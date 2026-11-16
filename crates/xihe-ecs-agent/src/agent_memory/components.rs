//! `AgentMemory`：一个智能体在 [`MemoryStore`] 之上的工作状态。
//!
//! 存储保存*发生了什么*（轮、事实、工具输出）；本层保存
//! *智能体当前正就此做什么*——关注点、任务计划、注释与最新摘要。
//! 把它们分开，正是让多个智能体共享一份长期存储的原因：记录
//! 归属到一个 [`Entity`] 所有者，而工作状态存于每智能体一条、
//! 每次写入即替换的记录中，因此重启能从同一存储
//! [`reload`](AgentMemory::reload) 它。
//!
//! 对话窗口又是另一回事
//! （[`crate::agent_memory::conversation::ConversationWindow`]）：组装成提示词的
//! 逐字轮。它不存在这里，因此一个会话可以被窗口化，
//! 而记忆为许多会话保留。

use std::sync::Arc;

use crate::ecs_core::Entity;
use serde::{Deserialize, Serialize};

use crate::agent_memory::embed::Embedding;
use crate::agent_memory::error::{MemoryError, MemoryResult};
use crate::agent_memory::record::{
    MemoryKind, MemoryQuery, MemoryRecord, RetentionPolicy, ScoredRecord, now_ms,
};
use crate::agent_memory::reflection::{ReflectionBudget, ReflectionStep, reflect, similar};
use crate::agent_memory::store::{MemoryStats, MemoryStore};

/// 标记持有智能体工作状态之记录的标签。
pub const STATE_TAG: &str = "xihe:state";

/// 智能体在轮之间携带的关注、计划、注释与摘要。
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkingState {
    /// 智能体当前正在优化什么，最旧的在前。
    pub focus: Vec<String>,
    /// 自由格式的 `kind: text` 观察，最新的在后。
    pub notes: Vec<String>,
    /// 最近的摘要；反思读取它。
    pub summary: String,
    #[serde(default)]
    /// 任务计划及其游标。
    pub plan: TaskPlan,
}

impl WorkingState {
    fn is_empty(&self) -> bool {
        self.focus.is_empty()
            && self.notes.is_empty()
            && self.summary.is_empty()
            && self.plan.is_empty()
    }
}

/// 一个智能体在任意存储之上的记忆桶。
///
/// 克隆廉价且共享存储：两个克隆只有在 [`reload`](AgentMemory::reload)
/// 之后才看到彼此的写入，这与存储本身提供的契约相同。
#[derive(Clone)]
pub struct AgentMemory {
    store: Arc<dyn MemoryStore>,
    owner: Option<Entity>,
    session: Option<String>,
    state: WorkingState,
    state_id: Option<u64>,
}

impl AgentMemory {
    /// 记录归属到 `owner` 的桶。
    pub fn new(store: Arc<dyn MemoryStore>, owner: Entity) -> Self {
        Self {
            store,
            owner: Some(owner),
            session: None,
            state: WorkingState::default(),
            state_id: None,
        }
    }

    /// 尚未归属到实体的桶，例如引导期间。
    pub fn unowned(store: Arc<dyn MemoryStore>) -> Self {
        Self {
            store,
            owner: None,
            session: None,
            state: WorkingState::default(),
            state_id: None,
        }
    }

    /// 把轮追踪归档到此对话下。
    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.session = Some(session.into());
        self
    }

    /// 此桶拥有其记录的实体。
    pub fn owner(&self) -> Option<Entity> {
        self.owner
    }

    /// 轮追踪所归档的对话。
    pub fn session(&self) -> Option<&str> {
        self.session.as_deref()
    }

    /// 底层存储，用于直接访问记录。
    pub fn store(&self) -> &Arc<dyn MemoryStore> {
        &self.store
    }

    /// 缓存的工作状态。
    pub fn state(&self) -> &WorkingState {
        &self.state
    }

    /// 关注点，最旧的在前。
    pub fn focus(&self) -> &[String] {
        &self.state.focus
    }

    /// 注释，最旧的在前。
    pub fn notes(&self) -> &[String] {
        &self.state.notes
    }

    /// 任务计划。
    pub fn plan(&self) -> &TaskPlan {
        &self.state.plan
    }

    /// 最新摘要（若已写入）。
    pub fn summary(&self) -> Option<&str> {
        (!self.state.summary.is_empty()).then_some(self.state.summary.as_str())
    }

    fn record_builder(&self, kind: MemoryKind, text: String) -> MemoryRecord {
        let mut record = MemoryRecord::new(kind, text, Embedding::default());
        if let Some(owner) = self.owner {
            record = record.with_owner(owner);
        }
        if let Some(session) = &self.session {
            record = record.with_session(session.clone());
        }
        record
    }

    /// 把工作状态写为一条记录，替换前一条。
    async fn persist(&mut self) -> MemoryResult<()> {
        if self.state.is_empty() {
            if let Some(id) = self.state_id.take() {
                self.store.remove(id).await?;
            }
            return Ok(());
        }
        let json = serde_json::to_string(&self.state).map_err(MemoryError::storage)?;
        // 记录的文本*就是*序列化后的状态：`reload` 把它解析回来，
        // 存储嵌入它，使摘要搜索能找到智能体的关注。
        let record = self
            .record_builder(MemoryKind::Reflection, json)
            .with_tag(STATE_TAG)
            .with_salience(2.0);
        if let Some(id) = self.state_id {
            let _ = self.store.remove(id).await;
        }
        let id = self.store.insert(record).await?;
        self.state_id = Some(id);
        Ok(())
    }

    /// 从存储中读回工作状态。
    ///
    /// 此所有者最新的状态记录胜出；较早进程遗留的记录会被忘记，
    /// 使存储不会在每次重启时增长。
    pub async fn reload(&mut self) -> MemoryResult<bool> {
        let tag = STATE_TAG.to_string();
        let owner = self.owner;
        let matches = |record: &MemoryRecord| {
            record.has_tag(&tag) && (owner.is_none() || record.owner == owner)
        };
        let mut found: Vec<MemoryRecord> = self.store.find(&matches).await;
        found.sort_by_key(|record| record.created_at_ms);
        let Some(latest) = found.pop() else {
            self.state = WorkingState::default();
            self.state_id = None;
            return Ok(false);
        };
        let state: WorkingState = serde_json::from_str(&latest.text).map_err(|error| {
            MemoryError::storage(format!(
                "working state record {} is unreadable: {error}",
                latest.id
            ))
        })?;
        for stale in found {
            self.store.remove(stale.id).await?;
        }
        self.state = state;
        self.state_id = Some(latest.id);
        Ok(true)
    }

    /// 持有工作状态之记录的 id（若已写入）。
    pub fn state_id(&self) -> Option<u64> {
        self.state_id
    }

    /// 替换关注点，裁掉最旧的以适配 `budget`。
    pub async fn set_focus(&mut self, items: &[String], budget: usize) -> MemoryResult<()> {
        if budget == 0 {
            return Err(MemoryError::config("the focus budget must be at least 1"));
        }
        let mut kept: Vec<String> = dedup(items.iter().map(|item| item.trim().to_string()));
        if kept.len() > budget {
            kept = kept.split_off(kept.len() - budget);
        }
        self.state.focus = kept;
        self.persist().await
    }

    /// 添加关注点，跳过已用其他措辞追踪的。
    pub async fn add_focus(
        &mut self,
        items: &[String],
        budget: &ReflectionBudget,
    ) -> MemoryResult<usize> {
        budget.validate()?;
        let mut added = 0;
        for item in items {
            let trimmed = item.trim();
            if trimmed.is_empty() || added >= budget.max_new_focus {
                continue;
            }
            let candidate = crate::agent_memory::reflection::normalize(trimmed);
            if self
                .state
                .focus
                .iter()
                .any(|seen| similar(&crate::agent_memory::reflection::normalize(seen), &candidate))
            {
                continue;
            }
            self.state.focus.push(trimmed.to_string());
            added += 1;
        }
        if added == 0 {
            return Ok(0);
        }
        while self.state.focus.len() > budget.max_focus {
            self.state.focus.remove(0);
        }
        self.persist().await?;
        Ok(added)
    }

    /// 替换注释；重复与空白被丢弃。
    pub async fn set_notes(&mut self, notes: &[String]) -> MemoryResult<()> {
        self.state.notes = dedup(notes.iter().map(|note| note.trim().to_string()));
        self.persist().await
    }

    /// 追加一条 `kind: text` 注释；重复同一注释不改变任何内容。
    pub async fn add_note(&mut self, kind: &str, text: &str) -> MemoryResult<bool> {
        let kind = kind.trim();
        let text = text.trim();
        if kind.is_empty() || text.is_empty() {
            return Err(MemoryError::config("a note needs both a kind and its text"));
        }
        let line = format!("{kind}: {text}");
        if self.state.notes.iter().any(|note| note == &line) {
            return Ok(false);
        }
        self.state.notes.push(line);
        self.persist().await?;
        Ok(true)
    }

    /// 存储下一次反思所读取的摘要。
    pub async fn set_summary(&mut self, summary: &str) -> MemoryResult<()> {
        self.state.summary = summary.trim().to_string();
        self.persist().await
    }

    /// 替换任务计划。
    pub async fn set_plan(&mut self, plan: TaskPlan) -> MemoryResult<()> {
        self.state.plan = plan;
        self.persist().await
    }

    /// 忘记计划；关注与注释不受影响。
    pub async fn clear_plan(&mut self) -> MemoryResult<()> {
        self.set_plan(TaskPlan::default()).await
    }

    /// 标记当前步骤完成并返回它，结束时返回 `None`。
    pub async fn complete_plan_step(&mut self) -> MemoryResult<Option<String>> {
        let done = self.state.plan.advance();
        if done.is_some() {
            self.persist().await?;
        }
        Ok(done)
    }

    /// 把存储的摘要变成关注点、事实、注释与一个计划。
    ///
    /// `tool_activity` 与 `failures` 描述摘要所覆盖的窗口。
    /// 预算限制一趟可添加的量，因此嘈杂的摘要不会淹没关注列表。
    /// 返回已应用的步骤。
    pub async fn apply_reflection(
        &mut self,
        tool_activity: usize,
        failures: usize,
        budget: &ReflectionBudget,
    ) -> MemoryResult<Vec<ReflectionStep>> {
        budget.validate()?;
        let steps = reflect(
            self.summary().unwrap_or_default(),
            &self.state.focus,
            !self.plan().is_empty(),
            tool_activity,
            failures,
        );
        let mut applied = Vec::new();
        let mut knowledge = 0;
        for step in steps {
            match &step {
                ReflectionStep::Focus(text) => {
                    let before = self.state.focus.len();
                    self.add_focus(std::slice::from_ref(text), budget).await?;
                    if self.state.focus.len() > before {
                        applied.push(step);
                    }
                }
                ReflectionStep::Note { kind, text } => {
                    if self.state.notes.len() > budget.max_notes {
                        continue;
                    }
                    if kind == "plan" {
                        let plan = TaskPlan::from_observation(text, budget.max_new_focus.max(3));
                        if plan.is_empty() {
                            continue;
                        }
                        self.set_plan(plan).await?;
                    } else if !self.add_note(kind, text).await? {
                        continue;
                    }
                    applied.push(step);
                }
                ReflectionStep::Knowledge(text) => {
                    if knowledge >= budget.max_knowledge {
                        continue;
                    }
                    self.remember(MemoryKind::Fact, text).await?;
                    knowledge += 1;
                    applied.push(step);
                }
            }
        }
        Ok(applied)
    }

    /// 以本智能体为所有者存储一条 `kind` 记录。
    pub async fn remember(&mut self, kind: MemoryKind, text: &str) -> MemoryResult<u64> {
        let record = self.record_builder(kind, text.trim().to_string());
        self.store.insert(record).await
    }

    /// 追加一轮的追踪：输入了什么、输出了什么、运行了哪些工具。
    pub async fn record_turn(
        &mut self,
        turn: u64,
        input: &str,
        output: &str,
        tool_names: &[String],
    ) -> MemoryResult<u64> {
        let text = format!("input={input} output={output}");
        let mut record = self
            .record_builder(MemoryKind::Conversation, text)
            .with_tag(format!("turn:{turn}"))
            .with_tag("agent_trace");
        for name in tool_names {
            record = record.with_tag(format!("tool:{name}"));
        }
        self.store.insert(record).await
    }

    /// 最近的轮追踪，最新的在前。
    pub async fn turn_trace(&self, limit: usize) -> MemoryResult<Vec<MemoryRecord>> {
        let owner = self.owner;
        let session = self.session.clone();
        let matches = |record: &MemoryRecord| {
            record.kind == MemoryKind::Conversation
                && (owner.is_none() || record.owner == owner)
                && (session.is_none() || record.session == session)
        };
        let mut records: Vec<MemoryRecord> = self.store.find(&matches).await;
        records.sort_by_key(|record| record.created_at_ms);
        records.reverse();
        records.truncate(limit.max(1));
        Ok(records)
    }

    /// 在存储上做语义搜索，当本智能体有所有者时限定到它。
    pub async fn recall(&self, query: &str, top_k: usize) -> MemoryResult<Vec<ScoredRecord>> {
        let mut builder = MemoryQuery::text(query).with_top_k(top_k.max(1));
        if let Some(owner) = self.owner {
            builder = builder.with_owner(owner);
        }
        self.store.search(&builder).await
    }

    /// 限制到一个类别的搜索，例如持久事实。
    pub async fn recall_kind(
        &self,
        query: &str,
        kind: MemoryKind,
        top_k: usize,
    ) -> MemoryResult<Vec<ScoredRecord>> {
        let mut builder = MemoryQuery::text(query)
            .with_kind(kind)
            .with_top_k(top_k.max(1));
        if let Some(owner) = self.owner {
            builder = builder.with_owner(owner);
        }
        self.store.search(&builder).await
    }

    /// 按 id 取一条记录，无论谁拥有它。
    pub async fn load(&self, id: u64) -> MemoryResult<Option<MemoryRecord>> {
        self.store.get(id).await
    }

    /// 删除一条记录。工作状态记录被重写，而非删除。
    pub async fn forget(&self, id: u64) -> MemoryResult<bool> {
        if self.state_id == Some(id) {
            return Err(MemoryError::config(
                "the working state record is rewritten, not deleted",
            ));
        }
        self.store.remove(id).await
    }

    /// 丢弃某智能体拥有的每条记忆；实体死亡时的常规清理。
    pub async fn forget_owner(&mut self, owner: Entity) -> MemoryResult<usize> {
        let removed = self.store.forget_entity(owner).await?;
        if self.owner == Some(owner) {
            self.state = WorkingState::default();
            self.state_id = None;
        }
        Ok(removed)
    }

    /// 对整个存储应用保留策略。
    pub async fn consolidate(&self, policy: &RetentionPolicy) -> MemoryResult<usize> {
        self.store.consolidate(policy).await
    }

    /// 底层存储的计数器。
    pub async fn stats(&self) -> MemoryStats {
        self.store.stats().await
    }

    /// 智能体喂回自身提示词的紧凑块。
    pub fn render_working_state(&self) -> String {
        let mut sections = Vec::new();
        if !self.state.focus.is_empty() {
            sections.push(format!("focus:\n{}", indent(&self.state.focus)));
        }
        if !self.state.plan.is_empty() {
            sections.push(format!("plan:\n{}", indent(&self.state.plan.render())));
        }
        if !self.state.notes.is_empty() {
            sections.push(format!("notes:\n{}", indent(&self.state.notes)));
        }
        if let Some(summary) = self.summary() {
            sections.push(format!("summary:\n  {summary}"));
        }
        sections.join("\n")
    }

    /// 用于过期检查的时间戳；放在这里，使调用者无需仅为
    /// 把记录与时钟比较而导入 [`crate::agent_memory::record::now_ms`]。
    pub fn now(&self) -> u64 {
        now_ms()
    }
}

fn indent(lines: &[String]) -> String {
    lines
        .iter()
        .map(|line| format!("  - {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn dedup(items: impl Iterator<Item = String>) -> Vec<String> {
    let mut out = Vec::new();
    for item in items.filter(|item| !item.is_empty()) {
        if !out.contains(&item) {
            out.push(item);
        }
    }
    out
}

/// 当前任务计划：有序步骤加一个游标。
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskPlan {
    /// 计划试图达成什么。
    pub goal: String,
    /// 按顺序的步骤。
    pub steps: Vec<String>,
    /// 进行中步骤的索引；`steps.len()` 表示计划已完成。
    pub current: usize,
}

impl TaskPlan {
    /// 显式计划，从第一步开始。
    pub fn new(goal: impl Into<String>, steps: Vec<String>) -> Self {
        Self {
            goal: goal.into(),
            steps,
            current: 0,
        }
    }

    /// 从一条观察派生计划：按句子边界拆分，
    /// 至多保留 `max_steps` 个子句。
    pub fn from_observation(text: &str, max_steps: usize) -> Self {
        let max_steps = max_steps.max(1);
        let mut steps: Vec<String> = text
            .split(['.', ';', '\n'])
            .map(str::trim)
            .filter(|clause| clause.chars().count() > 3)
            .map(as_sentence)
            .collect();
        if steps.is_empty() {
            let trimmed = text.trim();
            if trimmed.chars().count() > 3 {
                steps.push(as_sentence(trimmed));
            }
        }
        steps.truncate(max_steps);
        Self {
            goal: text.trim().to_string(),
            steps,
            current: 0,
        }
    }

    /// 当没有任何计划时为真。
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// 进行中的步骤（若有）。
    pub fn current_step(&self) -> Option<&str> {
        self.steps.get(self.current).map(String::as_str)
    }

    /// `(done, total)`；二者相等时计划完成。
    pub fn progress(&self) -> (usize, usize) {
        (self.current.min(self.steps.len()), self.steps.len())
    }

    /// 一旦每个步骤都完成即为真。
    pub fn complete(&self) -> bool {
        !self.steps.is_empty() && self.current >= self.steps.len()
    }

    /// 标记当前步骤完成；无剩余时返回 `None`。
    pub fn advance(&mut self) -> Option<String> {
        if self.complete() || self.steps.is_empty() {
            return None;
        }
        let done = self.steps[self.current].clone();
        self.current += 1;
        Some(done)
    }

    /// 带状态的步骤，用于提示词。
    pub fn render(&self) -> Vec<String> {
        self.steps
            .iter()
            .enumerate()
            .map(|(index, step)| {
                let mark = if index < self.current {
                    "done"
                } else if index == self.current {
                    "next"
                } else {
                    "later"
                };
                format!("[{mark}] {step}")
            })
            .collect()
    }
}

fn as_sentence(clause: &str) -> String {
    let mut chars = clause.chars();
    let head = chars
        .next()
        .map(|c| c.to_uppercase().collect::<String>())
        .unwrap_or_default();
    format!("{head}{}.", chars.as_str())
}

/// 为提示词格式化轮追踪记录。
pub fn agent_turn_summary(records: &[MemoryRecord]) -> String {
    let mut lines = Vec::new();
    for record in records {
        let turn = record
            .tags
            .iter()
            .find_map(|tag| tag.strip_prefix("turn:"))
            .unwrap_or("?");
        let tools: Vec<&str> = record
            .tags
            .iter()
            .filter_map(|tag| tag.strip_prefix("tool:"))
            .collect();
        let (input, output) = split_trace(&record.text);
        lines.push(format!(
            "turn {turn}: in={input} out={output} tools=[{}] access={}",
            tools.join(","),
            record.access_count
        ));
    }
    lines.join("\n")
}

fn split_trace(text: &str) -> (String, String) {
    let mut parts = text.splitn(2, " output=");
    let input = parts
        .next()
        .unwrap_or_default()
        .trim_start_matches("input=")
        .to_string();
    let output = parts.next().unwrap_or_default().to_string();
    (input, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs_core::Entity;
    use crate::agent_memory::embed::HashingEmbedder;
    use crate::agent_memory::store::InMemoryStore;
    use serde_json::json;

    fn shared_store() -> Arc<dyn MemoryStore> {
        Arc::new(InMemoryStore::with_embedder(HashingEmbedder::default()))
    }

    /// 在其自有存储之上的桶，使每个测试从空开始。
    fn memory_at(index: u32) -> (Entity, AgentMemory) {
        let owner = Entity::from_raw(index, 1);
        (owner, AgentMemory::new(shared_store(), owner))
    }

    fn entity(index: u32) -> Entity {
        Entity::from_raw(index, 1)
    }

    #[test]
    fn plans_derive_from_an_observation() {
        let plan = TaskPlan::from_observation(
            "check the failing tool call. retry with valid input; record the result. maybe more",
            3,
        );
        assert_eq!(plan.steps.len(), 3);
        assert_eq!(plan.current_step(), Some("Check the failing tool call."));
        assert_eq!(plan.progress(), (0, 3));
        assert!(!plan.complete());
    }

    #[test]
    fn plans_advance_one_step_at_a_time() {
        let mut plan = TaskPlan::from_observation("one thing. second thing", 3);
        assert_eq!(plan.advance().as_deref(), Some("One thing."));
        assert_eq!(plan.advance().as_deref(), Some("Second thing."));
        assert_eq!(plan.advance(), None, "past the end there is nothing left");
        assert!(plan.complete());
        assert_eq!(plan.render()[0], "[done] One thing.");
    }

    #[test]
    fn short_observations_collapse_to_one_step() {
        assert_eq!(TaskPlan::from_observation("do it", 2).steps.len(), 1);
        assert!(TaskPlan::from_observation("   ", 2).is_empty());
    }

    #[tokio::test]
    async fn working_state_survives_a_reload() {
        let store = shared_store();
        let owner = entity(1);
        let mut memory = AgentMemory::new(store.clone(), owner);
        memory
            .set_focus(&["keep replies short".to_string()], 5)
            .await
            .unwrap();
        memory
            .set_plan(TaskPlan::from_observation("fetch data. summarise it", 3))
            .await
            .unwrap();
        assert!(
            memory
                .add_note("correction", "use ISO dates")
                .await
                .unwrap()
        );
        memory
            .set_summary("the user wants per-region totals")
            .await
            .unwrap();

        let mut fresh = AgentMemory::new(store, owner);
        assert!(fresh.reload().await.unwrap());
        assert_eq!(fresh.focus(), ["keep replies short".to_string()]);
        assert_eq!(fresh.plan().steps.len(), 2);
        assert_eq!(fresh.notes(), ["correction: use ISO dates".to_string()]);
        assert_eq!(fresh.summary(), Some("the user wants per-region totals"));
        let rendered = fresh.render_working_state();
        assert!(rendered.contains("focus:"), "{rendered}");
        assert!(rendered.contains("[next] Fetch data."), "{rendered}");
        assert_eq!(fresh.stats().await.records, 1, "one state record per agent");
    }

    #[tokio::test]
    async fn an_empty_state_writes_nothing() {
        let (_, mut memory) = memory_at(2);
        memory.set_summary("  ").await.unwrap();
        assert_eq!(memory.stats().await.records, 0);
        assert!(!memory.reload().await.unwrap());
    }

    #[tokio::test]
    async fn duplicate_notes_are_ignored() {
        let (_, mut memory) = memory_at(3);
        assert!(memory.add_note("plan", "one").await.unwrap());
        assert!(!memory.add_note("plan", "one").await.unwrap());
        assert_eq!(memory.notes().len(), 1);
    }

    #[tokio::test]
    async fn a_note_needs_both_parts() {
        let (_, mut memory) = memory_at(4);
        assert!(memory.add_note("", "text").await.is_err());
        assert!(memory.add_note("kind", "").await.is_err());
    }

    #[tokio::test]
    async fn focus_budgets_trim_the_oldest_first() {
        let (_, mut memory) = memory_at(5);
        memory
            .set_focus(&["a".into(), "b".into(), "c".into(), "d".into()], 2)
            .await
            .unwrap();
        assert_eq!(memory.focus(), ["c".to_string(), "d".to_string()]);
        assert!(memory.set_focus(&["x".into()], 0).await.is_err());
    }

    #[tokio::test]
    async fn adding_focus_skips_near_duplicates() {
        let (_, mut memory) = memory_at(6);
        let budget = ReflectionBudget::default();
        memory
            .set_focus(&["report needs per-region totals".to_string()], 5)
            .await
            .unwrap();
        let added = memory
            .add_focus(
                &[
                    "The REPORT needs PER-REGION totals.".to_string(),
                    "cache exports yesterday".to_string(),
                ],
                &budget,
            )
            .await
            .unwrap();
        assert_eq!(added, 1, "{:?}", memory.focus());
        assert_eq!(memory.focus().len(), 2);
        assert_eq!(
            memory
                .add_focus(&["report needs per-region totals".to_string()], &budget)
                .await
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn reflection_writes_focus_facts_and_a_plan() {
        let (_, mut memory) = memory_at(7);
        memory
            .set_summary(
                "- the export always includes yesterday's rows\n- totals require a date filter\n- next step is to refresh the cache",
            )
            .await
            .unwrap();
        let steps = memory
            .apply_reflection(6, 2, &ReflectionBudget::default())
            .await
            .unwrap();
        assert!(!steps.is_empty(), "{steps:?}");
        assert!(!memory.focus().is_empty());
        assert!(!memory.plan().is_empty(), "{:?}", memory.plan().steps);
        let facts = memory
            .recall_kind("export", MemoryKind::Fact, 5)
            .await
            .unwrap();
        assert!(!facts.is_empty(), "the durable line became a fact");
    }

    #[tokio::test]
    async fn reflection_without_a_summary_still_plans() {
        let (_, mut memory) = memory_at(8);
        let steps = memory
            .apply_reflection(0, 0, &ReflectionBudget::default())
            .await
            .unwrap();
        assert!(steps.iter().any(|step| matches!(
            step,
            ReflectionStep::Note { kind, .. } if kind == "plan"
        )));
        assert!(!memory.plan().is_empty());
    }

    #[tokio::test]
    async fn invalid_reflection_budgets_are_rejected_before_work() {
        let (_, mut memory) = memory_at(9);
        let zero = ReflectionBudget {
            max_focus: 0,
            ..Default::default()
        };
        assert!(memory.apply_reflection(0, 0, &zero).await.is_err());
    }

    #[tokio::test]
    async fn turn_traces_are_recorded_and_retrievable() {
        let (owner, memory) = memory_at(10);
        let mut memory = memory.with_session("chat-1");
        memory
            .record_turn(3, "hello", "hi there", &["search".to_string()])
            .await
            .unwrap();
        let trace = memory.turn_trace(5).await.unwrap();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].owner, Some(owner));
        assert_eq!(trace[0].session.as_deref(), Some("chat-1"));
        let summary = agent_turn_summary(&trace);
        assert!(summary.contains("turn 3"), "{summary}");
        assert!(summary.contains("tools=[search]"), "{summary}");
        assert!(summary.contains("in=hello"), "{summary}");
    }

    #[tokio::test]
    async fn another_agents_state_is_not_reloaded() {
        let store = shared_store();
        let mut one = AgentMemory::new(store.clone(), entity(11));
        let mut two = AgentMemory::new(store.clone(), entity(12));
        one.set_focus(&["mine".to_string()], 3).await.unwrap();
        assert!(
            !two.reload().await.unwrap(),
            "the other agent's state is not ours"
        );
        assert!(two.focus().is_empty());
        assert!(one.reload().await.unwrap());
        assert_eq!(one.focus(), ["mine".to_string()]);
    }

    #[tokio::test]
    async fn forgetting_the_owner_clears_the_state() {
        let (owner, mut memory) = memory_at(13);
        memory.set_focus(&["a".to_string()], 2).await.unwrap();
        assert_eq!(memory.forget_owner(owner).await.unwrap(), 1);
        assert!(memory.focus().is_empty());
        assert_eq!(memory.state_id(), None);
    }

    #[tokio::test]
    async fn the_working_state_record_cannot_be_deleted_directly() {
        let (_, mut memory) = memory_at(14);
        memory.set_focus(&["a".to_string()], 2).await.unwrap();
        let id = memory.state_id().unwrap();
        assert!(memory.forget(id).await.is_err());
    }

    #[tokio::test]
    async fn recall_is_scoped_to_the_owner() {
        let store = shared_store();
        let mut one = AgentMemory::new(store.clone(), entity(15));
        let two = AgentMemory::new(store.clone(), entity(16));
        one.remember(MemoryKind::Fact, "the report needs per-region totals")
            .await
            .unwrap();
        assert_eq!(one.recall("per-region totals", 5).await.unwrap().len(), 1);
        assert!(
            two.recall("per-region totals", 5).await.unwrap().is_empty(),
            "another agent's fact must not surface"
        );
    }

    #[tokio::test]
    async fn an_unowned_bucket_sees_every_state_record() {
        let store = shared_store();
        let mut one = AgentMemory::new(store.clone(), entity(17));
        one.set_focus(&["written by one".to_string()], 2)
            .await
            .unwrap();
        let mut shared = AgentMemory::unowned(store);
        assert!(
            shared.reload().await.unwrap(),
            "unowned buckets do not filter"
        );
        assert_eq!(shared.focus(), ["written by one".to_string()]);
    }

    #[tokio::test]
    async fn a_corrupt_state_record_is_reported_not_panicked() {
        let store = shared_store();
        let owner = entity(18);
        let mut memory = AgentMemory::new(store.clone(), owner);
        memory.set_focus(&["a".to_string()], 2).await.unwrap();
        let id = memory.state_id().unwrap();
        let record = store.get(id).await.unwrap().unwrap();
        let bad = MemoryRecord {
            id: 0,
            text: "{ not json".to_string(),
            ..record
        };
        store.remove(id).await.unwrap();
        store.insert(bad).await.unwrap();
        let mut fresh = AgentMemory::new(store, owner);
        let error = fresh.reload().await.unwrap_err();
        assert!(error.to_string().contains("unreadable"), "{error}");
    }

    #[tokio::test]
    async fn stale_state_records_are_pruned_on_reload() {
        let store = shared_store();
        let owner = entity(19);
        let mut memory = AgentMemory::new(store.clone(), owner);
        memory.set_focus(&["first".to_string()], 2).await.unwrap();
        // 模拟一个较早进程遗留一条记录。
        let stale = MemoryRecord {
            id: 0,
            text: json!({"focus": ["older"], "notes": [], "summary": "", "plan": {"goal": "", "steps": [], "current": 0}}).to_string(),
            ..memory.record_builder(MemoryKind::Reflection, String::new())
        }
        .with_tag(STATE_TAG);
        store.insert(stale).await.unwrap();
        memory.set_focus(&["second".to_string()], 2).await.unwrap();
        let mut fresh = AgentMemory::new(store.clone(), owner);
        assert!(fresh.reload().await.unwrap());
        assert_eq!(fresh.focus(), ["second".to_string()]);
        assert_eq!(
            fresh.stats().await.records,
            1,
            "the stale record is gone, not stacked"
        );
    }

    #[tokio::test]
    async fn consolidation_and_stats_pass_through() {
        let store = shared_store();
        let owner = entity(20);
        let memory = AgentMemory::new(store.clone(), owner);
        // 较早运行中写入的一条感知：使它可回收的是时钟，
        // 而非类别。
        let ancient = MemoryRecord::at(
            MemoryKind::Observation,
            "a transient perception",
            Embedding::default(),
            1,
        )
        .with_owner(owner);
        store.insert(ancient).await.unwrap();
        let policy = RetentionPolicy::default().with_max_age(std::time::Duration::from_millis(1));
        assert_eq!(memory.consolidate(&policy).await.unwrap(), 1);
        let stats = memory.stats().await;
        assert_eq!(stats.records, 0);
        assert_eq!(stats.evictions, 1);
        assert!(memory.load(1).await.unwrap().is_none());
        assert!(memory.now() > 1);
        assert_eq!(memory.owner(), Some(owner));
    }
}
