//! `AgentMemory`: one agent's working state on top of a [`MemoryStore`].
//!
//! The store keeps *what happened* (turns, facts, tool output); this layer
//! keeps *what the agent is currently doing about it* — focus points, the task
//! plan, notes and the latest summary. Keeping them separate is what lets
//! several agents share one long-term store: records are attributed to an
//! [`Entity`] owner, and the working state lives in one record per agent that
//! is replaced on every write, so a restart can [`reload`](AgentMemory::reload)
//! it from the same store.
//!
//! The dialogue window is a different thing again
//! ([`crate::conversation::ConversationWindow`]): verbatim turns assembled into
//! a prompt. It is not stored here, so one session can be windowed while
//! memories are kept for many.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use xihe_core::Entity;

use crate::embed::Embedding;
use crate::error::{MemoryError, MemoryResult};
use crate::record::{MemoryKind, MemoryQuery, MemoryRecord, RetentionPolicy, ScoredRecord, now_ms};
use crate::reflection::{ReflectionBudget, ReflectionStep, reflect, similar};
use crate::store::{MemoryStats, MemoryStore};

/// Tag marking the record that holds an agent's working state.
pub const STATE_TAG: &str = "xihe:state";

/// The focus, plan, notes and summary an agent carries between turns.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkingState {
    /// What the agent is currently optimising for, oldest first.
    pub focus: Vec<String>,
    /// Free-form `kind: text` observations, newest last.
    pub notes: Vec<String>,
    /// The most recent summary; reflection reads this.
    pub summary: String,
    #[serde(default)]
    /// The task plan and its cursor.
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

/// One agent's memory bucket over an arbitrary store.
///
/// Cloning is cheap and shares the store: two clones see each other's writes
/// only after a [`reload`](AgentMemory::reload), which is the same contract the
/// store itself offers.
#[derive(Clone)]
pub struct AgentMemory {
    store: Arc<dyn MemoryStore>,
    owner: Option<Entity>,
    session: Option<String>,
    state: WorkingState,
    state_id: Option<u64>,
}

impl AgentMemory {
    /// A bucket whose records are attributed to `owner`.
    pub fn new(store: Arc<dyn MemoryStore>, owner: Entity) -> Self {
        Self {
            store,
            owner: Some(owner),
            session: None,
            state: WorkingState::default(),
            state_id: None,
        }
    }

    /// A bucket that is not attributed to an entity yet, e.g. during bootstrap.
    pub fn unowned(store: Arc<dyn MemoryStore>) -> Self {
        Self {
            store,
            owner: None,
            session: None,
            state: WorkingState::default(),
            state_id: None,
        }
    }

    /// File the turn trace under this conversation.
    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.session = Some(session.into());
        self
    }

    /// The entity whose records this bucket owns.
    pub fn owner(&self) -> Option<Entity> {
        self.owner
    }

    /// The conversation the turn trace is filed under.
    pub fn session(&self) -> Option<&str> {
        self.session.as_deref()
    }

    /// The underlying store, for direct record access.
    pub fn store(&self) -> &Arc<dyn MemoryStore> {
        &self.store
    }

    /// The cached working state.
    pub fn state(&self) -> &WorkingState {
        &self.state
    }

    /// Focus points, oldest first.
    pub fn focus(&self) -> &[String] {
        &self.state.focus
    }

    /// Notes, oldest first.
    pub fn notes(&self) -> &[String] {
        &self.state.notes
    }

    /// The task plan.
    pub fn plan(&self) -> &TaskPlan {
        &self.state.plan
    }

    /// The latest summary, if one was written.
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

    /// Write the working state as one record, replacing the previous one.
    async fn persist(&mut self) -> MemoryResult<()> {
        if self.state.is_empty() {
            if let Some(id) = self.state_id.take() {
                self.store.remove(id).await?;
            }
            return Ok(());
        }
        let json = serde_json::to_string(&self.state)
            .map_err(MemoryError::storage)?;
        // The record's text *is* the serialised state: `reload` parses it back,
        // and the store embeds it so a summary search can find the agent's focus.
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

    /// Read the working state back out of the store.
    ///
    /// The newest state record for this owner wins; a record left behind by an
    /// older process is forgotten so the store does not grow on every restart.
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
            MemoryError::storage(format!("working state record {} is unreadable: {error}", latest.id))
        })?;
        for stale in found {
            self.store.remove(stale.id).await?;
        }
        self.state = state;
        self.state_id = Some(latest.id);
        Ok(true)
    }

    /// The id of the record holding the working state, if written.
    pub fn state_id(&self) -> Option<u64> {
        self.state_id
    }

    /// Replace the focus points, trimming the oldest to fit `budget`.
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

    /// Add focus points, skipping ones already tracked in other words.
    pub async fn add_focus(&mut self, items: &[String], budget: &ReflectionBudget) -> MemoryResult<usize> {
        budget.validate()?;
        let mut added = 0;
        for item in items {
            let trimmed = item.trim();
            if trimmed.is_empty() || added >= budget.max_new_focus {
                continue;
            }
            let candidate = crate::reflection::normalize(trimmed);
            if self
                .state
                .focus
                .iter()
                .any(|seen| similar(&crate::reflection::normalize(seen), &candidate))
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

    /// Replace the notes; duplicates and blanks are dropped.
    pub async fn set_notes(&mut self, notes: &[String]) -> MemoryResult<()> {
        self.state.notes = dedup(notes.iter().map(|note| note.trim().to_string()));
        self.persist().await
    }

    /// Append a `kind: text` note; repeating the same note changes nothing.
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

    /// Store the summary that the next reflection reads.
    pub async fn set_summary(&mut self, summary: &str) -> MemoryResult<()> {
        self.state.summary = summary.trim().to_string();
        self.persist().await
    }

    /// Replace the task plan.
    pub async fn set_plan(&mut self, plan: TaskPlan) -> MemoryResult<()> {
        self.state.plan = plan;
        self.persist().await
    }

    /// Forget the plan; focus and notes are untouched.
    pub async fn clear_plan(&mut self) -> MemoryResult<()> {
        self.set_plan(TaskPlan::default()).await
    }

    /// Mark the current step complete and return it, or `None` when finished.
    pub async fn complete_plan_step(&mut self) -> MemoryResult<Option<String>> {
        let done = self.state.plan.advance();
        if done.is_some() {
            self.persist().await?;
        }
        Ok(done)
    }

    /// Turn the stored summary into focus points, facts, notes and a plan.
    ///
    /// `tool_activity` and `failures` describe the window the summary covers.
    /// The budget caps what one pass may add, so a noisy summary cannot flood
    /// the focus list. Returns the steps that were applied.
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

    /// Store a record of `kind` with this agent as its owner.
    pub async fn remember(&mut self, kind: MemoryKind, text: &str) -> MemoryResult<u64> {
        let record = self.record_builder(kind, text.trim().to_string());
        self.store.insert(record).await
    }

    /// Append one turn's trace: what came in, what went out, which tools ran.
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

    /// The most recent turn traces, newest first.
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

    /// Semantic search over the store, scoped to this agent when it has an owner.
    pub async fn recall(&self, query: &str, top_k: usize) -> MemoryResult<Vec<ScoredRecord>> {
        let mut builder = MemoryQuery::text(query).with_top_k(top_k.max(1));
        if let Some(owner) = self.owner {
            builder = builder.with_owner(owner);
        }
        self.store.search(&builder).await
    }

    /// Search restricted to one kind, e.g. durable facts.
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

    /// Fetch one record by id, whatever owns it.
    pub async fn load(&self, id: u64) -> MemoryResult<Option<MemoryRecord>> {
        self.store.get(id).await
    }

    /// Delete a record. The working-state record is rewritten, not deleted.
    pub async fn forget(&self, id: u64) -> MemoryResult<bool> {
        if self.state_id == Some(id) {
            return Err(MemoryError::config(
                "the working state record is rewritten, not deleted",
            ));
        }
        self.store.remove(id).await
    }

    /// Drop every memory owned by an agent; the usual cleanup when an entity dies.
    pub async fn forget_owner(&mut self, owner: Entity) -> MemoryResult<usize> {
        let removed = self.store.forget_entity(owner).await?;
        if self.owner == Some(owner) {
            self.state = WorkingState::default();
            self.state_id = None;
        }
        Ok(removed)
    }

    /// Apply a retention policy to the whole store.
    pub async fn consolidate(&self, policy: &RetentionPolicy) -> MemoryResult<usize> {
        self.store.consolidate(policy).await
    }

    /// Counters of the underlying store.
    pub async fn stats(&self) -> MemoryStats {
        self.store.stats().await
    }

    /// The compact block the agent feeds back into its own prompt.
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

    /// Timestamp used for expiry checks; kept here so callers do not import
    /// [`crate::record::now_ms`] just to compare a record against the clock.
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

/// The current task plan: ordered steps plus a cursor.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskPlan {
    /// What the plan is trying to achieve.
    pub goal: String,
    /// The steps in order.
    pub steps: Vec<String>,
    /// Index of the step in progress; `steps.len()` means the plan is finished.
    pub current: usize,
}

impl TaskPlan {
    /// An explicit plan, starting at the first step.
    pub fn new(goal: impl Into<String>, steps: Vec<String>) -> Self {
        Self {
            goal: goal.into(),
            steps,
            current: 0,
        }
    }

    /// Derive a plan from one observation: split on sentence boundaries and
    /// keep at most `max_steps` clauses.
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

    /// True when there is nothing planned.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The step in progress, if any.
    pub fn current_step(&self) -> Option<&str> {
        self.steps.get(self.current).map(String::as_str)
    }

    /// `(done, total)`; the plan is finished when they are equal.
    pub fn progress(&self) -> (usize, usize) {
        (self.current.min(self.steps.len()), self.steps.len())
    }

    /// True once every step is done.
    pub fn complete(&self) -> bool {
        !self.steps.is_empty() && self.current >= self.steps.len()
    }

    /// Mark the current step done; `None` when there is nothing left.
    pub fn advance(&mut self) -> Option<String> {
        if self.complete() || self.steps.is_empty() {
            return None;
        }
        let done = self.steps[self.current].clone();
        self.current += 1;
        Some(done)
    }

    /// The steps with their status, for the prompt.
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

/// Format turn-trace records for the prompt.
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
    use crate::embed::HashingEmbedder;
    use crate::store::InMemoryStore;
    use serde_json::json;
    use xihe_core::Entity;

    fn shared_store() -> Arc<dyn MemoryStore> {
        Arc::new(InMemoryStore::with_embedder(HashingEmbedder::default()))
    }

    /// A bucket over its own store, so each test starts empty.
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
        assert!(memory.add_note("correction", "use ISO dates").await.unwrap());
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
        assert!(
            !memory.plan().is_empty(),
            "{:?}",
            memory.plan().steps
        );
        let facts = memory.recall_kind("export", MemoryKind::Fact, 5).await.unwrap();
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
        assert!(!two.reload().await.unwrap(), "the other agent's state is not ours");
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
        one
            .remember(MemoryKind::Fact, "the report needs per-region totals")
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
        one.set_focus(&["written by one".to_string()], 2).await.unwrap();
        let mut shared = AgentMemory::unowned(store);
        assert!(shared.reload().await.unwrap(), "unowned buckets do not filter");
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
        // Simulate an older process leaving a record behind.
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
        // A perception written in an earlier run: the clock, not the kind, is
        // what makes it collectable.
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
