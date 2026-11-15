//! Multi-agent patterns on top of the ReAct loop.
//!
//! An orchestrator does not call a model itself. It writes tasks onto agent
//! entities and reads their answers back, so the same [`xihe_agent::React`]
//! system that runs one agent runs a team. The patterns here are the bookkeeping
//! for that:
//!
//! - [`Handoff`] — one agent finishes, another starts with its answer.
//! - [`Router`] — pick the agent whose description matches the task.
//! - [`Supervisor`] — a parent assigns work and collects results.
//! - [`Blackboard`] — a shared resource agents read and write between turns.
//! - [`FanOut`] — the same task to several agents, reduced to one answer.

use std::fmt::{self, Display, Formatter};

use async_trait::async_trait;
use xihe_agent::{AgentStatus, ReactAgent};
use xihe_core::{Entity, World};
use xihe_schedule::{Access, System, SysResult, SystemContext, SystemError, Work};

/// Why an orchestration step could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchestratorError {
    /// The named agent is not in the roster.
    UnknownAgent {
        /// The name that was asked for.
        name: String,
    },
    /// A hand-off pointed at an entity that is gone.
    MissingEntity {
        /// The entity that was expected.
        entity: Entity,
    },
    /// The roster has nobody to route to.
    EmptyRoster,
    /// A reduce step ran before every branch finished.
    Incomplete {
        /// How many branches have an answer.
        finished: usize,
        /// How many were started.
        expected: usize,
    },
    /// A branch failed, so the team can never fully finish.
    BranchFailed {
        /// The branch that failed.
        name: String,
        /// What the agent reported.
        error: String,
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
        }
    }
}

impl std::error::Error for OrchestratorError {}

/// One agent an orchestrator can address.
#[derive(Clone, Debug)]
pub struct RosterEntry {
    /// Stable name used in routes and hand-offs.
    pub name: String,
    /// What this agent is for, used by the router.
    pub description: String,
    /// The entity carrying the [`ReactAgent`].
    pub entity: Entity,
}

impl RosterEntry {
    /// An entry for an agent that already exists.
    pub fn new(name: impl Into<String>, description: impl Into<String>, entity: Entity) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            entity,
        }
    }
}

/// The agents a pattern may address.
#[derive(Clone, Debug, Default)]
pub struct Roster {
    entries: Vec<RosterEntry>,
}

impl Roster {
    /// An empty roster.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an agent.
    pub fn add(&mut self, entry: RosterEntry) -> &mut Self {
        if let Some(existing) = self.entries.iter_mut().find(|item| item.name == entry.name) {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
        self
    }

    /// Spawn `agent` into `world` and register it under `description`.
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

    /// Every entry, in registration order.
    pub fn entries(&self) -> &[RosterEntry] {
        &self.entries
    }

    /// How many agents are registered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nobody is registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry named `name`.
    pub fn get(&self, name: &str) -> Option<&RosterEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// Names, in registration order.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|entry| entry.name.as_str()).collect()
    }
}

/// Pass one agent's answer to another as a new question.
#[derive(Clone, Debug)]
pub struct Handoff {
    /// Who produced the answer.
    pub from: String,
    /// Who should continue.
    pub to: String,
    /// Text put in front of the answer, so the next agent knows why it arrived.
    pub preface: String,
}

impl Handoff {
    /// Hand `from`'s answer to `to`, introduced by `preface`.
    pub fn new(from: impl Into<String>, to: impl Into<String>, preface: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            preface: preface.into(),
        }
    }

    /// Write the hand-off into the world.
    ///
    /// Returns the question that was queued, so a caller can log it. The source
    /// agent must already be [`AgentStatus::Finished`].
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
        let answer = world
            .get::<ReactAgent>(from.entity)
            .ok_or(OrchestratorError::MissingEntity {
                entity: from.entity,
            })?
            .answer()
            .unwrap_or("")
            .to_string();
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

/// Pick an agent by matching the task against descriptions.
///
/// Matching is lexical on purpose: a router that needs a model can ask one, but
/// the default must work with the mock provider and must be explainable.
#[derive(Clone, Debug, Default)]
pub struct Router {
    /// Words that, when present, force a named agent. Checked before scoring.
    overrides: Vec<(String, String)>,
}

impl Router {
    /// A router with no forced routes.
    pub fn new() -> Self {
        Self::default()
    }

    /// When `needle` appears in the task, always choose `agent`.
    pub fn when_contains(mut self, needle: impl Into<String>, agent: impl Into<String>) -> Self {
        self.overrides.push((needle.into(), agent.into()));
        self
    }

    /// The agent this task should go to, and why.
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
        Ok((entry, format!("description overlap {score}")))
    }

    /// Queue `task` on the chosen agent.
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

/// A parent that assigns work and reads the answers back.
#[derive(Clone, Debug)]
pub struct Supervisor {
    /// The supervising agent's name.
    pub name: String,
    /// Agents it may assign to.
    roster: Roster,
}

impl Supervisor {
    /// A supervisor over `roster`.
    pub fn new(name: impl Into<String>, roster: Roster) -> Self {
        Self {
            name: name.into(),
            roster,
        }
    }

    /// The agents this supervisor may assign.
    pub fn roster(&self) -> &Roster {
        &self.roster
    }

    /// Queue `task` on `agent`.
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

    /// Answers from agents that have finished, as `(name, answer)`.
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

    /// Whether every assigned agent has reached a terminal status.
    pub fn settled(&self, world: &World) -> bool {
        self.roster.entries().iter().all(|entry| {
            world
                .get::<ReactAgent>(entry.entity)
                .is_none_or(|agent| agent.status().is_terminal() || agent.status() == AgentStatus::Idle)
        })
    }
}

/// A shared note any agent can read or append.
///
/// Stored as a world resource, so tools and systems reach it by type without
/// knowing which entity wrote it.
#[derive(Clone, Debug, Default)]
pub struct Blackboard {
    notes: Vec<Note>,
}

/// One line on a [`Blackboard`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Note {
    /// Who wrote it.
    pub author: String,
    /// What they wrote.
    pub text: String,
}

impl Blackboard {
    /// An empty board.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a note.
    pub fn post(&mut self, author: impl Into<String>, text: impl Into<String>) -> &mut Self {
        self.notes.push(Note {
            author: author.into(),
            text: text.into(),
        });
        self
    }

    /// Every note, oldest first.
    pub fn notes(&self) -> &[Note] {
        &self.notes
    }

    /// Notes written by `author`.
    pub fn by(&self, author: &str) -> Vec<&Note> {
        self.notes.iter().filter(|note| note.author == author).collect()
    }

    /// The board as lines a model can read.
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

    /// How many notes are posted.
    pub fn len(&self) -> usize {
        self.notes.len()
    }

    /// Whether nothing has been posted.
    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }
}

/// The same task sent to several agents, then folded into one answer.
#[derive(Clone, Debug)]
pub struct FanOut {
    /// Agents that each get the task.
    pub agents: Vec<String>,
    /// How their answers are combined.
    pub reduce: Reduce,
}

/// How fan-out answers become one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reduce {
    /// Join every answer, labelled by agent.
    #[default]
    Concat,
    /// Keep the longest answer.
    Longest,
    /// Keep the first answer in roster order.
    First,
}

impl FanOut {
    /// Send `agents` the same task and combine with [`Reduce::Concat`].
    pub fn new(agents: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            agents: agents.into_iter().map(Into::into).collect(),
            reduce: Reduce::Concat,
        }
    }

    /// Choose how answers are combined.
    pub fn with_reduce(mut self, reduce: Reduce) -> Self {
        self.reduce = reduce;
        self
    }

    /// Queue the task on every named agent.
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

    /// Combine finished answers.
    ///
    /// Returns [`OrchestratorError::Incomplete`] while any branch is still open,
    /// so a caller cannot reduce a partial team by accident. A branch that
    /// failed is terminal and will never finish, so it is reported as
    /// [`OrchestratorError::BranchFailed`] rather than waited on forever.
    pub fn reduce(&self, world: &World, roster: &Roster) -> Result<String, OrchestratorError> {
        let mut answers = Vec::with_capacity(self.agents.len());
        for name in &self.agents {
            let entry = roster
                .get(name)
                .ok_or_else(|| OrchestratorError::UnknownAgent { name: name.clone() })?;
            let agent = world
                .get::<ReactAgent>(entry.entity)
                .ok_or(OrchestratorError::MissingEntity {
                    entity: entry.entity,
                })?;
            let status = agent.status();
            if status == AgentStatus::Failed {
                return Err(OrchestratorError::BranchFailed {
                    name: name.clone(),
                    error: agent.error().unwrap_or("").to_string(),
                });
            }
            if status != AgentStatus::Finished {
                return Err(OrchestratorError::Incomplete {
                    finished: answers.len(),
                    expected: self.agents.len(),
                });
            }
            answers.push((name.clone(), agent.answer().unwrap_or("").to_string()));
        }
        Ok(match self.reduce {
            Reduce::Concat => answers
                .iter()
                .map(|(name, answer)| format!("{name}: {answer}"))
                .collect::<Vec<_>>()
                .join("\n"),
            Reduce::Longest => answers
                .into_iter()
                .max_by_key(|(_, answer)| answer.chars().count())
                .map(|(_, answer)| answer)
                .unwrap_or_default(),
            Reduce::First => answers
                .into_iter()
                .next()
                .map(|(_, answer)| answer)
                .unwrap_or_default(),
        })
    }
}

fn overlap(task: &str, description: &str) -> usize {
    description
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|word| word.len() > 2 && task.contains(word))
        .count()
}

/// Tell the driver that agent work was just queued, even if React already
/// reported itself settled earlier in the same frame.
async fn mark_agents_busy(ctx: &SystemContext) {
    ctx.with_world_mut(|world| Work::ensure(world).set("agents", true))
        .await;
}

/// A one-shot system that routes a task to the best-matching agent.
///
/// Add it before [`xihe_agent::React`] in the schedule so the chosen agent
/// starts working in the same frame.
pub struct RouteTask {
    router: Router,
    roster: Roster,
    task: String,
    done: bool,
}

impl RouteTask {
    /// Route `task` through `router` over `roster`, once.
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

/// A system that hands one agent's answer to another once the source finishes.
///
/// Add it after [`xihe_agent::React`] so it sees the finished status in the
/// same frame the source agent completed.
pub struct HandoffWhenFinished {
    handoff: Handoff,
    roster: Roster,
    done: bool,
}

impl HandoffWhenFinished {
    /// Apply `handoff` over `roster` when the source agent is finished.
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

/// The reduced answer a [`FanOutTask`] produced, stored as a world resource.
#[derive(Clone, Debug)]
pub struct FanOutResult(pub String);

/// A system that sends the same task to several agents and reduces their answers.
///
/// Dispatches on the first frame, then waits for every branch to finish before
/// writing a [`FanOutResult`] resource.
pub struct FanOutTask {
    fan: FanOut,
    roster: Roster,
    task: String,
    dispatched: bool,
    done: bool,
}

impl FanOutTask {
    /// Send `task` to the agents in `fan`, reducing with its configured strategy.
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
                    return Ok((true, None));
                }
                match fan.reduce(world, &roster) {
                    Ok(answer) => Ok((true, Some(answer))),
                    Err(OrchestratorError::Incomplete { .. }) => Ok((true, None)),
                    Err(error) => Err(error),
                }
            })
            .await;
        match result {
            Ok((true, Some(answer))) => {
                ctx.with_world_mut(|world| {
                    world.insert_resource(FanOutResult(answer));
                })
                .await;
                self.done = true;
                Ok(())
            }
            Ok((true, None)) => {
                if !self.dispatched {
                    mark_agents_busy(&ctx).await;
                }
                self.dispatched = true;
                Ok(())
            }
            Ok((false, _)) => Ok(()),
            Err(error) => Err(SystemError::new("fan-out", error.to_string())),
        }
    }
}

/// A one-shot system that assigns tasks through a supervisor.
pub struct SupervisorAssign {
    supervisor: Supervisor,
    assignments: Vec<(String, String)>,
    done: bool,
}

impl SupervisorAssign {
    /// Queue each `(agent, task)` pair through `supervisor`, once.
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
