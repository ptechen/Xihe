//! ReAct 循环之上的多智能体模式。
//!
//! 编排器自己不调用模型。它把任务写到智能体实体上并读回它们的回答，
//! 因此运行单个智能体的同一个 [`xihe_agent::React`] 系统也运行一个团队。
//! 这里的模式就是为此所做的簿记：
//!
//! - [`Handoff`] —— 一个智能体完成，另一个以它的回答开始。
//! - [`Router`] —— 挑选描述与任务匹配的智能体。
//! - [`Supervisor`] —— 一个父级分派工作并收集结果。
//! - [`Blackboard`] —— 一个共享资源，智能体在轮之间读写它。
//! - [`FanOut`] —— 同一任务发给若干智能体，归约为一个回答。

use std::fmt::{self, Display, Formatter};

use async_trait::async_trait;
use xihe_agent::{AgentStatus, ReactAgent};
use xihe_core::{Entity, World};
use xihe_observer::{EventKind, Observer};
use xihe_schedule::{Access, System, SysResult, SystemContext, SystemError, Work};

/// 一个编排步骤为何无法运行。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchestratorError {
    /// 被点名的智能体不在名册中。
    UnknownAgent {
        /// 被请求的名字。
        name: String,
    },
    /// 一次交接指向了一个已消失的实体。
    MissingEntity {
        /// 被期望的实体。
        entity: Entity,
    },
    /// 名册中无人可路由。
    EmptyRoster,
    /// 一个归约步骤在每个分支完成之前就运行了。
    Incomplete {
        /// 有多少分支已有回答。
        finished: usize,
        /// 启动了多少分支。
        expected: usize,
    },
    /// 一个分支失败了，因此团队永远无法完全完成。
    BranchFailed {
        /// 失败的分支。
        name: String,
        /// 智能体报告了什么。
        error: String,
    },
    /// 一次交接在其源智能体完成之前就被应用了。
    SourceNotFinished {
        /// 尚无回答的源智能体。
        name: String,
        /// 它当前的状态。
        status: AgentStatus,
    },
    /// 没有覆盖规则触发，也没有描述与任务重叠。
    NoMatch {
        /// 无法被路由的任务。
        task: String,
    },
}

impl Display for OrchestratorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownAgent { name } => write!(f, "no agent named `{name}`"),
            Self::MissingEntity { entity } => write!(f, "entity {entity:?} has no agent"),
            Self::EmptyRoster => write!(f, "the roster is empty"),
            Self::Incomplete { finished, expected } => {
                write!(f, "{finished} of {expected} branches have finished")
            }
            Self::BranchFailed { name, error } => {
                write!(f, "branch `{name}` failed: {error}")
            }
            Self::SourceNotFinished { name, status } => {
                write!(f, "handoff source `{name}` is {}, not finished", status.as_str())
            }
            Self::NoMatch { task } => write!(f, "no agent matches the task `{task}`"),
        }
    }
}

impl std::error::Error for OrchestratorError {}

/// 编排器可寻址的一个智能体。
#[derive(Clone, Debug)]
pub struct RosterEntry {
    /// 在路由与交接中使用的稳定名字。
    pub name: String,
    /// 此智能体用途为何，供路由器使用。
    pub description: String,
    /// 承载 [`ReactAgent`] 的实体。
    pub entity: Entity,
}

impl RosterEntry {
    /// 为一个已存在的智能体建条目。
    pub fn new(name: impl Into<String>, description: impl Into<String>, entity: Entity) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            entity,
        }
    }
}

/// 一个模式可寻址的那些智能体。
#[derive(Clone, Debug, Default)]
pub struct Roster {
    entries: Vec<RosterEntry>,
}

impl Roster {
    /// 一个空名册。
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加一个智能体。
    pub fn add(&mut self, entry: RosterEntry) -> &mut Self {
        if let Some(existing) = self.entries.iter_mut().find(|item| item.name == entry.name) {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
        self
    }

    /// 把 `agent` 生成到 `world` 中，并以 `description` 注册它。
    pub fn spawn(
        &mut self,
        world: &mut World,
        description: impl Into<String>,
        agent: ReactAgent,
    ) -> Entity {
        let name = agent.name().to_string();
        let entity = world.spawn_with((agent,));
        self.add(RosterEntry::new(name, description, entity));
        entity
    }

    /// 每个条目，按注册顺序。
    pub fn entries(&self) -> &[RosterEntry] {
        &self.entries
    }

    /// 注册了多少智能体。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否无人注册。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 名为 `name` 的条目。
    pub fn get(&self, name: &str) -> Option<&RosterEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// 名字，按注册顺序。
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|entry| entry.name.as_str()).collect()
    }
}

/// 把一个智能体的回答作为新问题传给另一个。
#[derive(Clone, Debug)]
pub struct Handoff {
    /// 谁产出了回答。
    pub from: String,
    /// 谁应继续。
    pub to: String,
    /// 置于回答之前的文本，使下一个智能体知道它为何到来。
    pub preface: String,
}

impl Handoff {
    /// 把 `from` 的回答交给 `to`，以 `preface` 引入。
    pub fn new(from: impl Into<String>, to: impl Into<String>, preface: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            preface: preface.into(),
        }
    }

    /// 把交接写入世界。
    ///
    /// 返回被排队的问题，使调用者能记录它。源智能体必须已是
    /// [`AgentStatus::Finished`]；否则这返回
    /// [`OrchestratorError::SourceNotFinished`]，而非交接一个空回答。
    pub fn apply(&self, world: &mut World, roster: &Roster) -> Result<String, OrchestratorError> {
        let from = roster
            .get(&self.from)
            .ok_or_else(|| OrchestratorError::UnknownAgent {
                name: self.from.clone(),
            })?;
        let to = roster
            .get(&self.to)
            .ok_or_else(|| OrchestratorError::UnknownAgent {
                name: self.to.clone(),
            })?;
        let source = world
            .get::<ReactAgent>(from.entity)
            .ok_or(OrchestratorError::MissingEntity {
                entity: from.entity,
            })?;
        if source.status() != AgentStatus::Finished {
            return Err(OrchestratorError::SourceNotFinished {
                name: self.from.clone(),
                status: source.status(),
            });
        }
        let answer = source.answer().unwrap_or("").to_string();
        let question = if self.preface.is_empty() {
            answer
        } else {
            format!("{}\n{answer}", self.preface)
        };
        let target = world
            .get_mut::<ReactAgent>(to.entity)
            .ok_or(OrchestratorError::MissingEntity { entity: to.entity })?;
        target.push_question(question.clone());
        Ok(question)
    }
}

/// 通过把任务与描述匹配来挑选一个智能体。
///
/// 匹配刻意是词法的：需要模型的路由器可以去问一个，但默认必须
/// 能与模拟提供方一起工作，且必须可解释。
#[derive(Clone, Debug, Default)]
pub struct Router {
    /// 出现时即强制某个被点名智能体的词。在打分之前检查。
    overrides: Vec<(String, String)>,
}

impl Router {
    /// 一个没有强制路由的路由器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 当 `needle` 出现在任务中时，总是选择 `agent`。
    pub fn when_contains(mut self, needle: impl Into<String>, agent: impl Into<String>) -> Self {
        self.overrides.push((needle.into(), agent.into()));
        self
    }

    /// 此任务应去往的智能体，以及为何。
    ///
    /// 覆盖规则优先胜出；否则选择词重叠最多的描述。当没有任何重叠
    /// 且没有覆盖规则触发时，这返回 [`OrchestratorError::NoMatch`]，
    /// 而非猜测一个智能体。
    pub fn route<'a>(
        &self,
        task: &str,
        roster: &'a Roster,
    ) -> Result<(&'a RosterEntry, String), OrchestratorError> {
        if roster.is_empty() {
            return Err(OrchestratorError::EmptyRoster);
        }
        let folded = task.to_lowercase();
        for (needle, name) in &self.overrides {
            if folded.contains(&needle.to_lowercase()) {
                let entry = roster
                    .get(name)
                    .ok_or_else(|| OrchestratorError::UnknownAgent { name: name.clone() })?;
                return Ok((entry, format!("matched `{needle}`")));
            }
        }
        let mut best: Option<(&RosterEntry, usize)> = None;
        for entry in roster.entries() {
            let score = overlap(&folded, &entry.description.to_lowercase());
            if best.is_none_or(|(_, previous)| score > previous) {
                best = Some((entry, score));
            }
        }
        let (entry, score) = best.expect("the roster was checked");
        if score == 0 {
            return Err(OrchestratorError::NoMatch {
                task: task.to_string(),
            });
        }
        Ok((entry, format!("description overlap {score}")))
    }

    /// 把 `task` 排队到被选中的智能体上。
    pub fn dispatch(
        &self,
        world: &mut World,
        roster: &Roster,
        task: &str,
    ) -> Result<(String, String), OrchestratorError> {
        let (entry, reason) = self.route(task, roster)?;
        let agent = world
            .get_mut::<ReactAgent>(entry.entity)
            .ok_or(OrchestratorError::MissingEntity {
                entity: entry.entity,
            })?;
        agent.push_question(task);
        Ok((entry.name.clone(), reason))
    }
}

/// 一个分派工作并读回回答的父级。
#[derive(Clone, Debug)]
pub struct Supervisor {
    /// 监督智能体的名字。
    pub name: String,
    /// 它可分派到的智能体。
    roster: Roster,
}

impl Supervisor {
    /// 一个管理 `roster` 的监督者。
    pub fn new(name: impl Into<String>, roster: Roster) -> Self {
        Self {
            name: name.into(),
            roster,
        }
    }

    /// 此监督者可分派的那些智能体。
    pub fn roster(&self) -> &Roster {
        &self.roster
    }

    /// 把 `task` 排队到 `agent` 上。
    pub fn assign(
        &self,
        world: &mut World,
        agent: &str,
        task: impl Into<String>,
    ) -> Result<(), OrchestratorError> {
        let entry = self
            .roster
            .get(agent)
            .ok_or_else(|| OrchestratorError::UnknownAgent {
                name: agent.to_string(),
            })?;
        let target = world
            .get_mut::<ReactAgent>(entry.entity)
            .ok_or(OrchestratorError::MissingEntity {
                entity: entry.entity,
            })?;
        target.push_question(task);
        Ok(())
    }

    /// 来自已完成智能体的回答，形如 `(name, answer)`。
    pub fn collect(&self, world: &World) -> Vec<(String, String)> {
        self.roster
            .entries()
            .iter()
            .filter_map(|entry| {
                let agent = world.get::<ReactAgent>(entry.entity)?;
                if agent.status() != AgentStatus::Finished {
                    return None;
                }
                Some((entry.name.clone(), agent.answer().unwrap_or("").to_string()))
            })
            .collect()
    }

    /// 是否每个被分派的智能体都已到达终态。
    pub fn settled(&self, world: &World) -> bool {
        self.roster.entries().iter().all(|entry| {
            world
                .get::<ReactAgent>(entry.entity)
                .is_none_or(|agent| agent.status().is_terminal() || agent.status() == AgentStatus::Idle)
        })
    }
}

/// 任何智能体都可读取或追加的共享备注。
///
/// 作为世界资源存储，因此工具与系统按类型触及它，而无需知道是哪个
/// 实体写的。保留的备注可用 [`Blackboard::with_retention`] 加以限定，
/// 使一个长期运行的世界不会无限制地累积备注。
#[derive(Clone, Debug, Default)]
pub struct Blackboard {
    notes: Vec<Note>,
    retention: usize,
}

/// [`Blackboard`] 上的一行。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Note {
    /// 谁写的。
    pub author: String,
    /// 写了什么。
    pub text: String,
}

impl Blackboard {
    /// 一个空黑板。
    pub fn new() -> Self {
        Self::default()
    }

    /// 至多保留 `limit` 条最近备注；`0`（默认）表示无限制。
    /// 随着新备注被张贴，更旧的备注被丢弃。
    pub fn with_retention(mut self, limit: usize) -> Self {
        self.retention = limit;
        self
    }

    /// 追加一条备注。
    pub fn post(&mut self, author: impl Into<String>, text: impl Into<String>) -> &mut Self {
        self.notes.push(Note {
            author: author.into(),
            text: text.into(),
        });
        if self.retention > 0 && self.notes.len() > self.retention {
            let excess = self.notes.len() - self.retention;
            self.notes.drain(..excess);
        }
        self
    }

    /// 每条备注，最旧的在前。
    pub fn notes(&self) -> &[Note] {
        &self.notes
    }

    /// 由 `author` 写的备注。
    pub fn by(&self, author: &str) -> Vec<&Note> {
        self.notes.iter().filter(|note| note.author == author).collect()
    }

    /// 黑板按模型可读的行呈现。
    pub fn render(&self) -> String {
        if self.notes.is_empty() {
            return "the blackboard is empty".to_string();
        }
        self.notes
            .iter()
            .map(|note| format!("{}: {}", note.author, note.text))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 张贴了多少备注。
    pub fn len(&self) -> usize {
        self.notes.len()
    }

    /// 是否尚未张贴任何内容。
    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }
}

/// 同一任务发给若干智能体，然后折叠为一个回答。
#[derive(Clone, Debug)]
pub struct FanOut {
    /// 各自得到该任务的智能体。
    pub agents: Vec<String>,
    /// 它们的回答如何被组合。
    pub reduce: Reduce,
}

/// 扇出的回答如何成为一个。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reduce {
    /// 连接每个回答，按智能体标注。需要每个分支都完成；
    /// 一个失败的分支会中止归约。
    #[default]
    Concat,
    /// 保留最长的已完成回答。等待每个分支落定，但丢弃失败的分支，
    /// 而非等待它们或因它们而中止。
    Longest,
    /// 保留名册中第一个分支的回答。只需要那一个分支；
    /// 缓慢或失败的兄弟分支无关紧要。
    First,
}

/// 一个分支的状态，为一次归约而快照。
struct Branch {
    name: String,
    status: AgentStatus,
    answer: String,
    error: String,
}

impl FanOut {
    /// 给 `agents` 发送同一任务，并以 [`Reduce::Concat`] 组合。
    pub fn new(agents: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            agents: agents.into_iter().map(Into::into).collect(),
            reduce: Reduce::Concat,
        }
    }

    /// 选择回答如何被组合。
    pub fn with_reduce(mut self, reduce: Reduce) -> Self {
        self.reduce = reduce;
        self
    }

    /// 把任务排队到每个被点名的智能体上。
    pub fn dispatch(
        &self,
        world: &mut World,
        roster: &Roster,
        task: &str,
    ) -> Result<usize, OrchestratorError> {
        let mut queued = 0;
        for name in &self.agents {
            let entry = roster
                .get(name)
                .ok_or_else(|| OrchestratorError::UnknownAgent { name: name.clone() })?;
            let agent = world
                .get_mut::<ReactAgent>(entry.entity)
                .ok_or(OrchestratorError::MissingEntity {
                    entity: entry.entity,
                })?;
            agent.push_question(task);
            queued += 1;
        }
        Ok(queued)
    }

    /// 按 [`Reduce`] 组合已完成的回答。
    ///
    /// 每种模式只等待它所需之物：[`Reduce::Concat`] 需要每个分支完成，
    /// [`Reduce::Longest`] 需要每个分支落定（失败的分支被丢弃），
    /// 而 [`Reduce::First`] 只需要名册中第一个分支。当一种模式的要求
    /// 未满足时，它返回 [`OrchestratorError::Incomplete`]，并计数每个
    /// 已完成的分支。一种模式无法忽略的失败被报告为
    /// [`OrchestratorError::BranchFailed`]，而非被永远等待。
    pub fn reduce(&self, world: &World, roster: &Roster) -> Result<String, OrchestratorError> {
        let mut branches = Vec::with_capacity(self.agents.len());
        for name in &self.agents {
            let entry = roster
                .get(name)
                .ok_or_else(|| OrchestratorError::UnknownAgent { name: name.clone() })?;
            let agent = world
                .get::<ReactAgent>(entry.entity)
                .ok_or(OrchestratorError::MissingEntity {
                    entity: entry.entity,
                })?;
            branches.push(Branch {
                name: name.clone(),
                status: agent.status(),
                answer: agent.answer().unwrap_or("").to_string(),
                error: agent.error().unwrap_or("").to_string(),
            });
        }

        let expected = branches.len();
        if expected == 0 {
            return Ok(String::new());
        }
        let finished = branches
            .iter()
            .filter(|branch| branch.status == AgentStatus::Finished)
            .count();
        let settled = branches
            .iter()
            .filter(|branch| branch.status.is_terminal())
            .count();
        let incomplete = || OrchestratorError::Incomplete { finished, expected };
        let branch_failed = |branch: &Branch| OrchestratorError::BranchFailed {
            name: branch.name.clone(),
            error: branch.error.clone(),
        };

        match self.reduce {
            Reduce::First => match branches.first() {
                Some(branch) if branch.status == AgentStatus::Finished => Ok(branch.answer.clone()),
                Some(branch) if branch.status == AgentStatus::Failed => Err(branch_failed(branch)),
                Some(_) => Err(incomplete()),
                None => Ok(String::new()),
            },
            Reduce::Longest => {
                if settled < expected {
                    return Err(incomplete());
                }
                branches
                    .iter()
                    .filter(|branch| branch.status == AgentStatus::Finished)
                    .max_by_key(|branch| branch.answer.chars().count())
                    .map(|branch| branch.answer.clone())
                    .ok_or_else(|| branch_failed(&branches[0]))
            }
            Reduce::Concat => {
                if let Some(branch) = branches
                    .iter()
                    .find(|branch| branch.status == AgentStatus::Failed)
                {
                    return Err(branch_failed(branch));
                }
                if finished < expected {
                    return Err(incomplete());
                }
                Ok(branches
                    .iter()
                    .map(|branch| format!("{}: {}", branch.name, branch.answer))
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
        }
    }
}

fn overlap(task: &str, description: &str) -> usize {
    description
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|word| word.len() > 2 && task.contains(word))
        .count()
}

/// 告知驱动器刚刚有智能体工作被排队，即便 React 在同一帧更早时
/// 已报告自己落定。该认领属于 React 系统：它排空智能体工作并在
/// 智能体落定时清除认领，而如果 React 被隔离，调度器代它释放认领。
async fn mark_agents_busy(ctx: &SystemContext) {
    ctx.with_world_mut(|world| Work::ensure(world).set_owned("agents", true, "react"))
        .await;
}

/// 停止一次 [`Reduce::First`] 归约不再需要的那些分支，使它们在回答
/// 已在手后不再继续花费。每个被取消的分支以一个诊断原因失败，
/// 并被记入 [`Observer`]，与运行时报告被停止智能体的方式一致。
async fn cancel_open_branches(ctx: &SystemContext, fan: &FanOut, roster: &Roster) {
    ctx.with_world_mut(|world| {
        let mut events = Vec::new();
        for name in &fan.agents {
            let Some(entry) = roster.get(name) else {
                continue;
            };
            let Some(agent) = world.get_mut::<ReactAgent>(entry.entity) else {
                continue;
            };
            if agent.status() != AgentStatus::Idle && agent.status().is_open() {
                agent.fail_with("cancelled: fan-out reduced with First".to_string());
                events.push(EventKind::Agent {
                    name: agent.name().to_string(),
                    status: agent.status().as_str().to_string(),
                });
            }
        }
        if let Some(observer) = world.resource_mut::<Observer>() {
            for event in events {
                observer.record(event);
            }
        }
    })
    .await;
}

/// 一个把任务路由到最匹配智能体的一次性系统。
///
/// 在调度器中把它加在 [`xihe_agent::React`] 之前，使被选中的智能体
/// 在同一帧开始工作。
pub struct RouteTask {
    router: Router,
    roster: Roster,
    task: String,
    done: bool,
}

impl RouteTask {
    /// 通过 `router` 在 `roster` 上路由 `task`，一次。
    pub fn new(router: Router, roster: Roster, task: impl Into<String>) -> Self {
        Self {
            router,
            roster,
            task: task.into(),
            done: false,
        }
    }
}

#[async_trait]
impl System for RouteTask {
    fn name(&self) -> &str {
        "route-task"
    }

    fn access(&self) -> Access {
        Access::exclusive()
    }

    async fn run(&mut self, ctx: SystemContext) -> SysResult {
        if self.done {
            return Ok(());
        }
        let router = self.router.clone();
        let roster = self.roster.clone();
        let task = self.task.clone();
        let result = ctx
            .with_world_mut(move |world| router.dispatch(world, &roster, &task))
            .await;
        match result {
            Ok((name, reason)) => {
                tracing::info!(agent = %name, %reason, "routed task");
                self.done = true;
                mark_agents_busy(&ctx).await;
                Ok(())
            }
            Err(error) => Err(SystemError::new("route-task", error.to_string())),
        }
    }
}

/// 一旦源智能体完成就把它的回答交给另一个的系统。
///
/// 在调度器中把它加在 [`xihe_agent::React`] 之后，使它在源智能体
/// 完成的同一帧看到已完成状态。
pub struct HandoffWhenFinished {
    handoff: Handoff,
    roster: Roster,
    done: bool,
}

impl HandoffWhenFinished {
    /// 当源智能体已完成时，在 `roster` 上应用 `handoff`。
    pub fn new(handoff: Handoff, roster: Roster) -> Self {
        Self {
            handoff,
            roster,
            done: false,
        }
    }
}

#[async_trait]
impl System for HandoffWhenFinished {
    fn name(&self) -> &str {
        "handoff"
    }

    fn access(&self) -> Access {
        Access::exclusive()
    }

    async fn run(&mut self, ctx: SystemContext) -> SysResult {
        if self.done {
            return Ok(());
        }
        let handoff = self.handoff.clone();
        let roster = self.roster.clone();
        let result = ctx
            .with_world_mut(move |world| {
                let from = roster.get(&handoff.from).ok_or_else(|| {
                    OrchestratorError::UnknownAgent {
                        name: handoff.from.clone(),
                    }
                })?;
                let source = world.get::<ReactAgent>(from.entity).ok_or(
                    OrchestratorError::MissingEntity {
                        entity: from.entity,
                    },
                )?;
                if source.status() != AgentStatus::Finished {
                    return Ok(None);
                }
                handoff.apply(world, &roster).map(Some)
            })
            .await;
        match result {
            Ok(Some(question)) => {
                tracing::info!(%question, "handoff applied");
                self.done = true;
                mark_agents_busy(&ctx).await;
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(error) => Err(SystemError::new("handoff", error.to_string())),
        }
    }
}

/// 一个 [`FanOutTask`] 产出的归约回答，作为世界资源存储。
#[derive(Clone, Debug)]
pub struct FanOutResult(pub String);

/// 把同一任务发给若干智能体并归约它们回答的系统。
///
/// 在第一帧分派，然后一旦所配置的 [`Reduce`] 被满足就写入一个
/// [`FanOutResult`] 资源：[`Reduce::Concat`] 要每个分支完成，
/// [`Reduce::Longest`] 要每个分支落定（失败的被丢弃），
/// 而 [`Reduce::First`] 只要名册中第一个分支。
///
/// 把它加在 [`xihe_agent::React`] 之后：它在分支完成后归约，因此
/// 在 React 之前运行它会让驱动器在分派帧就落定，早于归约运行。
pub struct FanOutTask {
    fan: FanOut,
    roster: Roster,
    task: String,
    dispatched: bool,
    done: bool,
}

impl FanOutTask {
    /// 把 `task` 发给 `fan` 中的智能体，以其配置的策略归约。
    pub fn new(fan: FanOut, roster: Roster, task: impl Into<String>) -> Self {
        Self {
            fan,
            roster,
            task: task.into(),
            dispatched: false,
            done: false,
        }
    }
}

#[async_trait]
impl System for FanOutTask {
    fn name(&self) -> &str {
        "fan-out"
    }

    fn access(&self) -> Access {
        Access::exclusive()
    }

    async fn run(&mut self, ctx: SystemContext) -> SysResult {
        if self.done {
            return Ok(());
        }
        let fan = self.fan.clone();
        let roster = self.roster.clone();
        let task = self.task.clone();
        let dispatched = self.dispatched;
        let result = ctx
            .with_world_mut(move |world| {
                if !dispatched {
                    fan.dispatch(world, &roster, &task)?;
                    return Ok(None);
                }
                match fan.reduce(world, &roster) {
                    Ok(answer) => Ok(Some(answer)),
                    Err(OrchestratorError::Incomplete { .. }) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await;
        match result {
            Ok(Some(answer)) => {
                ctx.with_world_mut(|world| {
                    world.insert_resource(FanOutResult(answer));
                })
                .await;
                if self.fan.reduce == Reduce::First {
                    cancel_open_branches(&ctx, &self.fan, &self.roster).await;
                }
                self.done = true;
                Ok(())
            }
            Ok(None) => {
                if !self.dispatched {
                    mark_agents_busy(&ctx).await;
                }
                self.dispatched = true;
                Ok(())
            }
            Err(error) => Err(SystemError::new("fan-out", error.to_string())),
        }
    }
}

/// 一个通过监督者分派任务的一次性系统。
pub struct SupervisorAssign {
    supervisor: Supervisor,
    assignments: Vec<(String, String)>,
    done: bool,
}

impl SupervisorAssign {
    /// 通过 `supervisor` 把每个 `(agent, task)` 对排队，一次。
    pub fn new(supervisor: Supervisor, assignments: Vec<(String, String)>) -> Self {
        Self {
            supervisor,
            assignments,
            done: false,
        }
    }
}

#[async_trait]
impl System for SupervisorAssign {
    fn name(&self) -> &str {
        "supervisor-assign"
    }

    fn access(&self) -> Access {
        Access::exclusive()
    }

    async fn run(&mut self, ctx: SystemContext) -> SysResult {
        if self.done {
            return Ok(());
        }
        let supervisor = self.supervisor.clone();
        let assignments = self.assignments.clone();
        let result = ctx
            .with_world_mut(move |world| {
                for (agent, task) in &assignments {
                    supervisor.assign(world, agent, task.clone())?;
                }
                Ok::<_, OrchestratorError>(())
            })
            .await;
        match result {
            Ok(()) => {
                self.done = true;
                mark_agents_busy(&ctx).await;
                Ok(())
            }
            Err(error) => Err(SystemError::new("supervisor-assign", error.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn team(world: &mut World) -> Roster {
        let mut roster = Roster::new();
        roster.spawn(
            world,
            "writes short answers",
            ReactAgent::new("scribe", "be brief"),
        );
        roster.spawn(
            world,
            "deploys to production",
            ReactAgent::new("ops", "ship carefully"),
        );
        roster
    }

    fn finish(world: &mut World, name: &str, roster: &Roster, answer: &str) {
        let entity = roster.get(name).expect(name).entity;
        world
            .get_mut::<ReactAgent>(entity)
            .expect("agent")
            .finish(answer.to_string());
    }

    fn fail(world: &mut World, name: &str, roster: &Roster) {
        let entity = roster.get(name).expect(name).entity;
        world
            .get_mut::<ReactAgent>(entity)
            .expect("agent")
            .fail(&xihe_agent::AgentError::Stopped);
    }

    #[test]
    fn a_handoff_queues_the_answer_as_the_next_question() {
        let mut world = World::new();
        let roster = team(&mut world);
        finish(&mut world, "scribe", &roster, "the export is nightly");
        let question = Handoff::new("scribe", "ops", "act on this")
            .apply(&mut world, &roster)
            .expect("handed off");
        assert!(question.contains("nightly"));
        let ops = world
            .get::<ReactAgent>(roster.get("ops").unwrap().entity)
            .unwrap();
        assert_eq!(ops.status(), AgentStatus::Working);
        assert!(ops.awaits_model());
    }

    #[test]
    fn a_handoff_refuses_an_unfinished_source() {
        let mut world = World::new();
        let roster = team(&mut world);
        // scribe 被提问但尚未完成：没有回答可交接。
        let scribe = roster.get("scribe").unwrap().entity;
        world
            .get_mut::<ReactAgent>(scribe)
            .unwrap()
            .push_question("review");
        let error = Handoff::new("scribe", "ops", "act on this")
            .apply(&mut world, &roster)
            .expect_err("an unfinished source has no answer to hand off");
        assert!(
            matches!(&error, OrchestratorError::SourceNotFinished { name, .. } if name == "scribe"),
            "the unfinished source is named: {error:?}"
        );
    }

    #[test]
    fn an_unknown_name_is_an_error() {
        let mut world = World::new();
        let roster = team(&mut world);
        let error = Handoff::new("ghost", "ops", "")
            .apply(&mut world, &roster)
            .expect_err("no ghost");
        assert!(matches!(error, OrchestratorError::UnknownAgent { .. }));
    }

    #[test]
    fn routing_prefers_an_override_then_word_overlap() {
        let mut world = World::new();
        let roster = team(&mut world);
        let router = Router::new().when_contains("prod", "ops");
        let (entry, reason) = router.route("push to prod", &roster).expect("a route");
        assert_eq!(entry.name, "ops");
        assert!(reason.contains("prod"));

        let (entry, _) = Router::new()
            .route("please write a short note", &roster)
            .expect("scored");
        assert_eq!(entry.name, "scribe");
    }

    #[test]
    fn routing_a_task_that_matches_nothing_is_an_error() {
        let mut world = World::new();
        let roster = team(&mut world);
        let error = Router::new()
            .route("completely unrelated zzz", &roster)
            .expect_err("zero overlap must not silently pick the first agent");
        assert!(
            matches!(error, OrchestratorError::NoMatch { .. }),
            "a task that matches no description is reported, not guessed: {error:?}"
        );
    }

    #[test]
    fn a_supervisor_collects_only_finished_work() {
        let mut world = World::new();
        let roster = team(&mut world);
        let supervisor = Supervisor::new("lead", roster.clone());
        supervisor
            .assign(&mut world, "scribe", "summarise")
            .expect("assigned");
        assert!(supervisor.collect(&world).is_empty());
        finish(&mut world, "scribe", &roster, "three lines");
        assert_eq!(
            supervisor.collect(&world),
            vec![("scribe".to_string(), "three lines".to_string())]
        );
        assert!(supervisor.settled(&world));
    }

    #[test]
    fn a_blackboard_renders_in_post_order() {
        let mut board = Blackboard::new();
        board.post("scribe", "the window folds at 20 turns");
        board.post("ops", "deploy is frozen");
        assert_eq!(board.len(), 2);
        assert_eq!(board.by("ops").len(), 1);
        assert!(board.render().starts_with("scribe:"));
        assert!(board.render().contains("ops: deploy is frozen"));
    }

    #[test]
    fn blackboard_retention_bounds_the_notes() {
        let mut board = Blackboard::new().with_retention(2);
        for index in 0..5 {
            board.post("scribe", format!("note {index}"));
        }
        assert_eq!(board.len(), 2, "only the most recent notes are retained");
        assert!(board.render().contains("note 4"));
        assert!(
            !board.render().contains("note 0"),
            "the oldest notes are dropped: {}",
            board.render()
        );
    }

    #[test]
    fn fan_out_waits_for_every_branch() {
        let mut world = World::new();
        let roster = team(&mut world);
        let fan = FanOut::new(["scribe", "ops"]).with_reduce(Reduce::Longest);
        assert_eq!(fan.dispatch(&mut world, &roster, "review").expect("queued"), 2);
        assert!(matches!(
            fan.reduce(&world, &roster),
            Err(OrchestratorError::Incomplete { finished: 0, expected: 2 })
        ));
        finish(&mut world, "scribe", &roster, "ok");
        finish(&mut world, "ops", &roster, "not ok, and here is why");
        assert_eq!(
            fan.reduce(&world, &roster).expect("reduced"),
            "not ok, and here is why"
        );
    }

    #[test]
    fn fan_out_reports_a_failed_branch() {
        let mut world = World::new();
        let roster = team(&mut world);
        let fan = FanOut::new(["scribe", "ops"]);
        fan.dispatch(&mut world, &roster, "review").expect("queued");
        finish(&mut world, "scribe", &roster, "ok");
        let ops = roster.get("ops").unwrap().entity;
        world
            .get_mut::<ReactAgent>(ops)
            .unwrap()
            .fail(&xihe_agent::AgentError::Stopped);
        let error = fan.reduce(&world, &roster).unwrap_err();
        assert!(
            matches!(&error, OrchestratorError::BranchFailed { name, .. } if name == "ops"),
            "a failed branch is reported, not waited on: {error:?}"
        );
        assert!(error.to_string().contains("ops"), "{error}");
    }

    #[test]
    fn concat_labels_each_answer() {
        let mut world = World::new();
        let roster = team(&mut world);
        let fan = FanOut::new(["scribe", "ops"]);
        finish(&mut world, "scribe", &roster, "a");
        finish(&mut world, "ops", &roster, "b");
        assert_eq!(fan.reduce(&world, &roster).expect("joined"), "scribe: a\nops: b");
    }

    #[test]
    fn incomplete_counts_every_finished_branch_not_just_the_prefix() {
        let mut world = World::new();
        let roster = team(&mut world);
        let fan = FanOut::new(["scribe", "ops"]);
        fan.dispatch(&mut world, &roster, "review").expect("queued");
        // ops 在 scribe 仍开放时完成：计数必须包含 ops，
        // 而非停在名册顺序中第一个开放分支处。
        finish(&mut world, "ops", &roster, "done");
        assert!(matches!(
            fan.reduce(&world, &roster),
            Err(OrchestratorError::Incomplete { finished: 1, expected: 2 })
        ));
    }

    #[test]
    fn a_failed_branch_is_terminal_even_behind_an_open_one() {
        let mut world = World::new();
        let roster = team(&mut world);
        let fan = FanOut::new(["scribe", "ops"]);
        fan.dispatch(&mut world, &roster, "review").expect("queued");
        // scribe 仍开放，ops 已永久失败。团队永远无法完成，
        // 因此这必须浮现为 BranchFailed，而非一个等待信号。
        let ops = roster.get("ops").unwrap().entity;
        world
            .get_mut::<ReactAgent>(ops)
            .unwrap()
            .fail(&xihe_agent::AgentError::Stopped);
        let error = fan.reduce(&world, &roster).unwrap_err();
        assert!(
            matches!(&error, OrchestratorError::BranchFailed { name, .. } if name == "ops"),
            "a terminal failure wins over an open branch: {error:?}"
        );
    }

    #[test]
    fn first_reduces_as_soon_as_the_first_branch_finishes() {
        let mut world = World::new();
        let roster = team(&mut world);
        let fan = FanOut::new(["scribe", "ops"]).with_reduce(Reduce::First);
        fan.dispatch(&mut world, &roster, "review").expect("queued");
        finish(&mut world, "scribe", &roster, "the first answer");
        // ops 仍开放，但 First 只需要名册中第一个分支。
        assert_eq!(
            fan.reduce(&world, &roster).expect("reduced"),
            "the first answer"
        );
    }

    #[test]
    fn first_still_waits_for_the_first_branch() {
        let mut world = World::new();
        let roster = team(&mut world);
        let fan = FanOut::new(["scribe", "ops"]).with_reduce(Reduce::First);
        fan.dispatch(&mut world, &roster, "review").expect("queued");
        // ops 完成了但 scribe（第一个分支）仍开放：First 必须
        // 等待 scribe，而非交回 ops 的回答。
        finish(&mut world, "ops", &roster, "a later answer");
        assert!(matches!(
            fan.reduce(&world, &roster),
            Err(OrchestratorError::Incomplete { .. })
        ));
    }

    #[test]
    fn longest_ignores_a_failed_sibling() {
        let mut world = World::new();
        let roster = team(&mut world);
        let fan = FanOut::new(["scribe", "ops"]).with_reduce(Reduce::Longest);
        fan.dispatch(&mut world, &roster, "review").expect("queued");
        finish(&mut world, "scribe", &roster, "a usable answer");
        fail(&mut world, "ops", &roster);
        // 一个失败的兄弟分支被丢弃，不被等待也不致命：最长的
        // 已完成回答仍能归约。
        assert_eq!(
            fan.reduce(&world, &roster).expect("reduced"),
            "a usable answer"
        );
    }

    #[test]
    fn longest_still_waits_for_an_open_sibling() {
        let mut world = World::new();
        let roster = team(&mut world);
        let fan = FanOut::new(["scribe", "ops"]).with_reduce(Reduce::Longest);
        fan.dispatch(&mut world, &roster, "review").expect("queued");
        finish(&mut world, "scribe", &roster, "short");
        // ops 仍可能产出更长的回答，因此 Longest 等待它。
        assert!(matches!(
            fan.reduce(&world, &roster),
            Err(OrchestratorError::Incomplete { .. })
        ));
    }

    use xihe_observer::{EventKind, Observer};
    use xihe_schedule::{SystemContext, WorldHandle};

    #[tokio::test]
    async fn route_task_dispatches_once_and_marks_work() {
        let mut world = World::new();
        let roster = team(&mut world);
        let ctx = SystemContext::new(WorldHandle::new(world));
        let mut system = RouteTask::new(
            Router::new().when_contains("prod", "ops"),
            roster.clone(),
            "push to prod",
        );
        system.run(ctx.clone()).await.expect("routed");
        system.run(ctx.clone()).await.expect("idempotent");
        let world = ctx.world.read().await;
        let ops = world.get::<ReactAgent>(roster.get("ops").unwrap().entity).unwrap();
        assert_eq!(ops.status(), AgentStatus::Working);
        assert!(world.resource::<Work>().unwrap().is_busy());
    }

    #[tokio::test]
    async fn handoff_waits_for_the_source_to_finish() {
        let mut world = World::new();
        let roster = team(&mut world);
        let ctx = SystemContext::new(WorldHandle::new(world));
        let mut system = HandoffWhenFinished::new(
            Handoff::new("scribe", "ops", "act on this"),
            roster.clone(),
        );
        system.run(ctx.clone()).await.expect("not yet");
        {
            let world = ctx.world.read().await;
            let ops = world.get::<ReactAgent>(roster.get("ops").unwrap().entity).unwrap();
            assert_eq!(ops.status(), AgentStatus::Idle);
        }
        ctx.with_world_mut(|world| finish(world, "scribe", &roster, "the export is nightly"))
            .await;
        system.run(ctx.clone()).await.expect("handed off");
        let world = ctx.world.read().await;
        let ops = world.get::<ReactAgent>(roster.get("ops").unwrap().entity).unwrap();
        assert_eq!(ops.status(), AgentStatus::Working);
        assert!(ops.awaits_model());
    }

    #[tokio::test]
    async fn fan_out_system_dispatches_then_reduces() {
        let mut world = World::new();
        let roster = team(&mut world);
        let ctx = SystemContext::new(WorldHandle::new(world));
        let mut system = FanOutTask::new(
            FanOut::new(["scribe", "ops"]).with_reduce(Reduce::Concat),
            roster.clone(),
            "review",
        );
        system.run(ctx.clone()).await.expect("dispatched");
        system.run(ctx.clone()).await.expect("still waiting");
        assert!(ctx.with_world(|w| w.resource::<FanOutResult>().is_none()).await);
        ctx.with_world_mut(|world| {
            finish(world, "scribe", &roster, "a");
            finish(world, "ops", &roster, "b");
        })
        .await;
        system.run(ctx.clone()).await.expect("reduced");
        let result = ctx
            .with_world(|w| w.resource::<FanOutResult>().cloned())
            .await
            .expect("a result");
        assert_eq!(result.0, "scribe: a\nops: b");
    }

    #[tokio::test]
    async fn first_cancels_the_siblings_it_no_longer_needs() {
        let mut world = World::new();
        world.insert_resource(Observer::new());
        let roster = team(&mut world);
        let ctx = SystemContext::new(WorldHandle::new(world));
        let mut system = FanOutTask::new(
            FanOut::new(["scribe", "ops"]).with_reduce(Reduce::First),
            roster.clone(),
            "review",
        );
        system.run(ctx.clone()).await.expect("dispatched");
        // scribe（第一个分支）完成；ops 仍开放且在花费。
        ctx.with_world_mut(|world| finish(world, "scribe", &roster, "the first answer"))
            .await;
        system.run(ctx.clone()).await.expect("reduced");

        let world = ctx.world.read().await;
        let ops = world
            .get::<ReactAgent>(roster.get("ops").unwrap().entity)
            .unwrap();
        assert!(
            ops.status().is_terminal(),
            "the unneeded sibling is stopped, not left spending: {}",
            ops.status().as_str()
        );
        assert_eq!(world.resource::<FanOutResult>().unwrap().0, "the first answer");
        let observer = world.resource::<Observer>().unwrap();
        assert!(
            observer.events().iter().any(|event| matches!(
                &event.kind,
                EventKind::Agent { name, status } if name == "ops" && status == "failed"
            )),
            "the cancellation is visible in the log: {}",
            observer.render()
        );
    }

    #[tokio::test]
    async fn supervisor_assign_queues_every_task() {
        let mut world = World::new();
        let roster = team(&mut world);
        let ctx = SystemContext::new(WorldHandle::new(world));
        let mut system = SupervisorAssign::new(
            Supervisor::new("lead", roster.clone()),
            vec![
                ("scribe".into(), "summarise".into()),
                ("ops".into(), "deploy".into()),
            ],
        );
        system.run(ctx.clone()).await.expect("assigned");
        let world = ctx.world.read().await;
        for name in ["scribe", "ops"] {
            let agent = world.get::<ReactAgent>(roster.get(name).unwrap().entity).unwrap();
            assert_eq!(agent.status(), AgentStatus::Working);
        }
    }
}
