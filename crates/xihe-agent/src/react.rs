//! The ReAct loop: reason, act, observe — until an answer or a limit.
//!
//! An agent is an entity carrying a [`ReactAgent`] component. The component
//! owns the conversation, the counters and the outcome; [`React`] is a
//! [`xihe_schedule::System`] that reads those components, calls the provider,
//! runs the tools and writes the result back. Nothing in the component talks to
//! the outside world, which is what makes an agent replayable: point a
//! [`xihe_llm::MockProvider`](xihe_llm::MockProvider) at it and the same
//! transcript comes out.
//!
//! # Lock discipline
//!
//! [`React`] never holds the world lock across an await on the provider or a
//! tool. A step is read-lock → snapshot → release → talk → write-lock → apply,
//! and the apply checks that the agent it is writing to is the one it planned
//! for (same turn count, same status). If anything else moved the agent in the
//! meantime, the step is dropped instead of applied to stale state.
//!
//! # Running one
//!
//! ```
//! use serde_json::json;
//! use xihe_agent::{LlmService, React, ReactAgent, ToolRegistry, EchoTool};
//! use xihe_core::World;
//! use xihe_llm::{MockProvider, Reply};
//! use xihe_schedule::{System, SystemContext, WorldHandle};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mut world = World::new();
//! let mut registry = ToolRegistry::default();
//! registry.register(EchoTool);
//! world.insert_resource(registry);
//! world.insert_resource(LlmService::from_provider(MockProvider::with_script([
//!     Reply::tool("echo", json!({"text": "hello"})),
//!     Reply::text("echoed hello"),
//! ])));
//! world.spawn_with((ReactAgent::new("scribe", "answer briefly").ask("say hello"),));
//!
//! let ctx = SystemContext::new(WorldHandle::new(world));
//! React::new().run(ctx.clone()).await?;
//!
//! let world = ctx.world.read().await;
//! let agent = world
//!     .single::<ReactAgent>()
//!     .and_then(|entity| world.get::<ReactAgent>(entity));
//! assert_eq!(agent.expect("an agent").answer(), Some("echoed hello"));
//! # Ok(())
//! # }
//! # tokio::runtime::Builder::new_current_thread()
//! #     .enable_all().build().unwrap().block_on(async { run().await.unwrap(); });
//! ```

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use serde_json::Value;
use xihe_core::Entity;
use xihe_llm::{ChatClient, ChatMessage, GenerationRequest, GenerationResponse, StreamChunk, ToolCall, ToolDescriptor};
use xihe_memory::{ConversationWindow, HeuristicSummariser, Speaker, Summariser, Turn};
use xihe_observer::{EventKind, Observer};
use xihe_schedule::{access, Access, System, SysResult, SystemContext, Work};

use crate::error::AgentError;
use crate::guardrail::{Guardrails, Spend};
use crate::hitl::{ApprovalRequest, Approver, Decision};
use crate::llm::LlmService;
use crate::memory::MemoryService;
use crate::output::ResponseFormat;
use crate::replay::{approval_key, Journal};
use crate::tool::{ToolContext, ToolOutcome, ToolRegistry};

/// Where the agent is in its task.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AgentStatus {
    /// Nothing has been asked of it.
    #[default]
    Idle,
    /// The loop owes the agent a model call, a tool run, or both.
    Working,
    /// A tool is waiting on a human; see [`ReactAgent::awaiting`].
    AwaitingApproval,
    /// The task ended with an answer.
    Finished,
    /// The task ended without one.
    Failed,
}

impl AgentStatus {
    /// Whether the loop will leave this agent alone.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Failed)
    }

    /// Whether the loop may still act on this agent.
    pub fn is_open(self) -> bool {
        !self.is_terminal()
    }

    /// Stable lowercase name for logs and metrics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Working => "working",
            Self::AwaitingApproval => "awaiting_approval",
            Self::Finished => "finished",
            Self::Failed => "failed",
        }
    }
}

impl Display for AgentStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Everything one ReAct agent carries.
#[derive(Clone, Debug)]
pub struct ReactAgent {
    name: String,
    instructions: String,
    model: Option<String>,
    status: AgentStatus,
    conversation: ConversationWindow,
    guardrails: Guardrails,
    spend: Spend,
    started: Option<Instant>,
    answer: Option<String>,
    parsed: Option<Value>,
    error: Option<String>,
    format: ResponseFormat,
    repairs_left: usize,
    awaiting: Option<ApprovalRequest>,
    decision: Option<Decision>,
    tool_errors: usize,
}

impl ReactAgent {
    /// An idle agent with `instructions` as its standing system prompt.
    pub fn new(name: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instructions: instructions.into(),
            model: None,
            status: AgentStatus::Idle,
            conversation: ConversationWindow::new(),
            guardrails: Guardrails::new(),
            spend: Spend::default(),
            started: None,
            answer: None,
            parsed: None,
            error: None,
            format: ResponseFormat::Text,
            repairs_left: 0,
            awaiting: None,
            decision: None,
            tool_errors: 0,
        }
    }

    /// Ask a different model than the service default.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Replace the ceilings.
    #[must_use]
    pub fn with_guardrails(mut self, guardrails: Guardrails) -> Self {
        self.guardrails = guardrails;
        self
    }

    /// Require the final answer to match `format`.
    #[must_use]
    pub fn with_format(mut self, format: ResponseFormat) -> Self {
        self.repairs_left = format.repairs();
        self.format = format;
        self
    }

    /// Replace the conversation window, e.g. to change its token budget.
    #[must_use]
    pub fn with_window(mut self, conversation: ConversationWindow) -> Self {
        self.conversation = conversation;
        self
    }

    /// Ask the agent something, starting a fresh task if the last one ended.
    #[must_use]
    pub fn ask(mut self, question: impl Into<String>) -> Self {
        self.begin();
        self.conversation.push_user(question);
        self.status = AgentStatus::Working;
        self
    }

    /// Queue a question without moving a builder chain along.
    pub fn push_question(&mut self, question: impl Into<String>) {
        self.begin();
        self.conversation.push_user(question);
        self.status = AgentStatus::Working;
    }

    /// Append something the model should see before answering.
    pub fn observe(&mut self, content: impl Into<String>) {
        self.conversation.push_user(content);
    }

    /// Zero the counters and start the clock, unless a task is still open.
    fn begin(&mut self) {
        let fresh = self.status.is_terminal() || self.status == AgentStatus::Idle;
        if fresh {
            self.spend = Spend::default();
            self.answer = None;
            self.parsed = None;
            self.error = None;
            self.awaiting = None;
            self.decision = None;
            self.tool_errors = 0;
            self.repairs_left = self.format.repairs();
            self.started = Some(Instant::now());
        } else if self.started.is_none() {
            self.started = Some(Instant::now());
        }
    }

    /// The agent's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Where the loop stands.
    pub fn status(&self) -> AgentStatus {
        self.status
    }

    /// The final answer, once the task is finished.
    pub fn answer(&self) -> Option<&str> {
        self.answer.as_deref()
    }

    /// The answer parsed as JSON, when the format asked for one.
    pub fn structured(&self) -> Option<&Value> {
        self.parsed.as_ref()
    }

    /// Why the task failed.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Model calls spent so far.
    pub fn turns(&self) -> usize {
        self.spend.turns
    }

    /// What the agent has spent.
    pub fn spend(&self) -> Spend {
        self.spend
    }

    /// The ceilings in force.
    pub fn guardrails(&self) -> &Guardrails {
        &self.guardrails
    }

    /// The answer format in force.
    pub fn format(&self) -> &ResponseFormat {
        &self.format
    }

    /// The transcript the agent is working from.
    pub fn conversation(&self) -> &ConversationWindow {
        &self.conversation
    }

    /// The transcript, mutable, for callers that seed or trim it directly.
    pub fn conversation_mut(&mut self) -> &mut ConversationWindow {
        &mut self.conversation
    }

    /// The request waiting for a human, while the loop is parked.
    pub fn awaiting(&self) -> Option<&ApprovalRequest> {
        self.awaiting.as_ref()
    }

    /// The call the model made that is waiting for a human.
    pub fn pending_call(&self) -> Option<&ToolCall> {
        match self.awaiting.as_ref() {
            Some(request) => Some(&request.call),
            None => None,
        }
    }

    /// Wall-clock time spent on the current task.
    pub fn elapsed(&self) -> Duration {
        self.started.map_or(Duration::ZERO, |started| started.elapsed())
    }

    /// Tool calls that came back failing.
    pub fn tool_errors(&self) -> usize {
        self.tool_errors
    }

    /// Repair chances left for a malformed answer.
    pub fn repairs_left(&self) -> usize {
        self.repairs_left
    }

    /// A one-line summary for logs.
    pub fn summary(&self) -> String {
        format!(
            "{}: {} turns={}, tools={}",
            self.name, self.status, self.spend.turns, self.spend.tool_calls
        )
    }

    /// The system prompt actually sent: instructions plus the answer format.
    pub fn system_text(&self) -> String {
        match self.format.instruction() {
            Some(extra) if !self.instructions.is_empty() => {
                format!("{}\n{extra}", self.instructions)
            }
            Some(extra) => extra,
            None => self.instructions.clone(),
        }
    }

    /// Whether a model call is owed: something was said that the model has not
    /// answered yet.
    pub fn awaits_model(&self) -> bool {
        if self.status != AgentStatus::Working {
            return false;
        }
        self.conversation
            .turns()
            .last()
            .is_some_and(|turn| matches!(turn.speaker, Speaker::User | Speaker::Tool))
    }

    /// Whether the loop is parked on a call a human has since answered.
    pub fn has_decision(&self) -> bool {
        self.status == AgentStatus::AwaitingApproval && self.decision.is_some()
    }

    /// Approve the parked call; `false` when nothing was parked.
    pub fn approve(&mut self) -> bool {
        self.decide(Decision::Approved)
    }

    /// Reject the parked call with `reason`; `false` when nothing was parked.
    pub fn reject(&mut self, reason: impl Into<String>) -> bool {
        self.decide(Decision::reject(reason))
    }

    fn decide(&mut self, decision: Decision) -> bool {
        if self.awaiting.is_none() {
            return false;
        }
        self.decision = Some(decision);
        true
    }

    /// The request for the next model call, given the tools on offer.
    pub fn build_request(&self, descriptors: Vec<ToolDescriptor>) -> GenerationRequest {
        let mut messages = Vec::with_capacity(self.conversation.turns().len() + 1);
        let system = self.system_text();
        if !system.is_empty() {
            messages.push(ChatMessage::system(system));
        }
        messages.extend(self.conversation.as_messages());
        GenerationRequest::new(self.model.clone().unwrap_or_default())
            .with_messages(messages)
            .with_tools(descriptors)
    }

    /// Charge a model call and append its answer, recording calls as the text
    /// the transcript needs.
    fn record_response(&mut self, response: &GenerationResponse) {
        self.spend.record_response(response);
        let content = if response.content.trim().is_empty() {
            if response.tool_calls.is_empty() {
                String::new()
            } else {
                let names = response
                    .tool_calls
                    .iter()
                    .map(|call| call.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("(calling tools: {names})")
            }
        } else {
            response.content.clone()
        };
        if !content.is_empty() {
            self.conversation
                .push(Turn::assistant(content).with_tool_calls(response.tool_calls.clone()));
        }
    }

    /// Append one tool result, whether the tool ran or refused to.
    pub fn record_tool_result(&mut self, call: &ToolCall, outcome: &ToolOutcome) {
        if !outcome.ok {
            self.tool_errors += 1;
        }
        self.spend.record_tool_call();
        self.conversation
            .push(Turn::tool(call.id.clone(), outcome.transcript_text()));
    }

    /// Accept a model answer as final, parsing it when the format asks.
    pub fn finish(&mut self, answer: impl Into<String>) {
        self.finish_text(answer.into());
    }

    fn finish_text(&mut self, answer: String) {
        self.status = AgentStatus::Finished;
        self.parsed = match &self.format {
            ResponseFormat::Text => None,
            ResponseFormat::Json { .. } => serde_json::from_str(&answer).ok(),
        };
        self.answer = Some(answer);
        self.awaiting = None;
        self.decision = None;
    }

    /// Accept a model answer that was already parsed.
    fn finish_parsed(&mut self, answer: String, value: Value) {
        self.status = AgentStatus::Finished;
        self.answer = Some(answer);
        self.parsed = Some(value);
        self.awaiting = None;
        self.decision = None;
    }

    /// Stop the task with a reason the caller can read.
    pub fn fail(&mut self, error: &AgentError) {
        self.status = AgentStatus::Failed;
        self.error = Some(error.to_string());
        self.awaiting = None;
        self.decision = None;
    }

    /// Stop the task with a message that is not an [`AgentError`].
    pub fn fail_with(&mut self, message: impl Into<String>) {
        self.status = AgentStatus::Failed;
        self.error = Some(message.into());
        self.awaiting = None;
        self.decision = None;
    }

    /// Park the loop on `request`, remembering the call so it can resume.
    fn park(&mut self, request: ApprovalRequest) {
        self.status = AgentStatus::AwaitingApproval;
        self.awaiting = Some(request);
        self.decision = None;
    }
}

/// The system that runs the loop.
///
/// One activation advances every busy agent through up to [`React::steps`]
/// reason-act rounds. Agents progress independently: each call runs outside the
/// world lock and is written back the moment it completes, so a fast agent never
/// waits on a slow one. Calls in flight together are applied one at a time, and
/// the journal keys each by a stable per-agent identity, so the final state is
/// deterministic even though completion order is not.
pub struct React {
    name: String,
    steps: usize,
    approver: Option<Arc<dyn Approver>>,
    fold_threshold: usize,
    keep_last: usize,
    stream_sink: Option<tokio::sync::mpsc::UnboundedSender<StreamChunk>>,
}

impl Default for React {
    fn default() -> Self {
        Self::new()
    }
}

impl React {
    /// Advance each busy agent up to 8 rounds per activation.
    pub fn new() -> Self {
        Self {
            name: "react".to_string(),
            steps: 8,
            approver: None,
            fold_threshold: 0,
            keep_last: 4,
            stream_sink: None,
        }
    }

    /// Name the system, which matters once several loops share a schedule.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Rounds per activation; `0` is treated as one.
    #[must_use]
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps.max(1);
        self
    }

    /// Answer approval requests instead of parking.
    #[must_use]
    pub fn with_approver<A: Approver + 'static>(mut self, approver: A) -> Self {
        self.approver = Some(Arc::new(approver));
        self
    }

    /// Share an approver handle, e.g. one gate in front of several loops.
    #[must_use]
    pub fn with_shared_approver(mut self, approver: Arc<dyn Approver>) -> Self {
        self.approver = Some(approver);
        self
    }

    /// Fold conversations longer than `threshold` turns, keeping the last
    /// `keep_last` turns verbatim. `0` disables folding.
    ///
    /// When an [`LlmService`] is present the fold asks the model to summarise;
    /// otherwise it falls back to sentence ranking.
    #[must_use]
    pub fn with_fold(mut self, threshold: usize, keep_last: usize) -> Self {
        self.fold_threshold = threshold;
        self.keep_last = keep_last.max(1);
        self
    }

    /// Forward every streamed chunk to `sink` as it arrives.
    ///
    /// When set, the loop uses [`ChatClient::stream`] instead of
    /// [`ChatClient::generate`], accumulating the full response for its own
    /// decision-making while the caller observes tokens incrementally.
    #[must_use]
    pub fn with_stream_sink(
        mut self,
        sink: tokio::sync::mpsc::UnboundedSender<StreamChunk>,
    ) -> Self {
        self.stream_sink = Some(sink);
        self
    }

    /// Rounds this system is willing to run per activation.
    pub fn steps(&self) -> usize {
        self.steps
    }
}

impl fmt::Debug for React {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("React")
            .field("name", &self.name)
            .field("steps", &self.steps)
            .field("approver", &self.approver.as_ref().map(|a| a.name()))
            .field("fold_threshold", &self.fold_threshold)
            .finish()
    }
}

#[async_trait::async_trait]
impl System for React {
    fn name(&self) -> &str {
        &self.name
    }

    fn access(&self) -> Access {
        access!(
            reads = [LlmService, ToolRegistry, MemoryService],
            writes = [ReactAgent, Observer, Work]
        )
    }

    async fn run(&mut self, ctx: SystemContext) -> SysResult {
        // A pool of in-flight steps. Each outcome is applied the moment it
        // completes and only that agent is re-planned, so agents advance at
        // their own pace instead of waiting on the slowest call in a round.
        let mut pool: FuturesUnordered<Pin<Box<dyn Future<Output = Outcome> + Send>>> =
            FuturesUnordered::new();
        let mut rounds: HashMap<Entity, usize> = HashMap::new();

        if ctx.is_stopping() {
            stop_agents(&ctx).await;
            update_work(&ctx, &self.name).await;
            return Ok(());
        }
        if self.fold_threshold > 0 {
            fold_overlong(&ctx, self.fold_threshold, self.keep_last).await;
        }
        for step in plan_batch(&ctx, self.approver.clone(), self.stream_sink.clone()).await {
            pool.push(Box::pin(run_step(step)));
        }

        loop {
            if ctx.is_stopping() {
                stop_agents(&ctx).await;
                break;
            }
            // Race the in-flight calls against shutdown: a stop signalled
            // mid-generation drops the provider future rather than waiting it out.
            let next = tokio::select! {
                next = pool.next() => next,
                _ = ctx.stop_token().wait() => {
                    stop_agents(&ctx).await;
                    break;
                }
            };
            let Some(outcome) = next else { break };
            let entity = outcome_entity(&outcome);
            if !apply(ctx.clone(), outcome).await {
                continue;
            }
            let used = rounds.entry(entity).or_insert(0);
            *used += 1;
            if *used >= self.steps {
                continue;
            }
            if self.fold_threshold > 0 {
                fold_overlong(&ctx, self.fold_threshold, self.keep_last).await;
            }
            if let Some(step) =
                plan_one(&ctx, entity, self.approver.clone(), self.stream_sink.clone()).await
            {
                pool.push(Box::pin(run_step(step)));
            }
        }
        update_work(&ctx, &self.name).await;
        Ok(())
    }
}

/// Run one planned step to its outcome, outside the world lock.
async fn run_step(step: Step) -> Outcome {
    match step {
        Step::Generate(generation) => (*generation).execute().await,
        Step::Resume(resume) => (*resume).execute().await,
        Step::Fail { entity, error } => Outcome::Failed { entity, error },
    }
}

/// A step that owes the world a model call.
struct Generation {
    entity: Entity,
    turn: usize,
    client: Arc<ChatClient>,
    request: GenerationRequest,
    registry: ToolRegistry,
    tool_ctx: ToolContext,
    approver: Option<Arc<dyn Approver>>,
    format: ResponseFormat,
    repairs_left: usize,
    max_tool_calls: usize,
    spend: Spend,
    stream_sink: Option<tokio::sync::mpsc::UnboundedSender<StreamChunk>>,
    journal: Option<Journal>,
}

/// A step that continues a parked call a human has answered.
struct Resume {
    entity: Entity,
    turn: usize,
    request: ApprovalRequest,
    decision: Decision,
    registry: ToolRegistry,
    tool_ctx: ToolContext,
    approver: Option<Arc<dyn Approver>>,
    journal: Option<Journal>,
}

enum Step {
    Generate(Box<Generation>),
    Resume(Box<Resume>),
    Fail { entity: Entity, error: AgentError },
}

/// What a step produced, ready to be written back.
enum Outcome {
    /// A final text answer.
    Answered {
        entity: Entity,
        turn: usize,
        response: Box<GenerationResponse>,
        parsed: Option<Value>,
    },
    /// A usable-looking answer that the format rejected; ask again.
    Repaired {
        entity: Entity,
        turn: usize,
        response: Box<GenerationResponse>,
        repair: String,
    },
    /// The model called tools; their results are ready to append.
    Acted {
        entity: Entity,
        turn: usize,
        response: Box<GenerationResponse>,
        results: Vec<(ToolCall, ToolOutcome)>,
        refused: usize,
        parked: Option<ApprovalRequest>,
    },
    /// A parked call was decided; its result is ready to append.
    Observed {
        entity: Entity,
        turn: usize,
        results: Vec<(ToolCall, ToolOutcome)>,
        refused: usize,
        parked: Option<ApprovalRequest>,
    },
    /// The task is over and did not produce an answer.
    Failed {
        entity: Entity,
        error: AgentError,
    },
}

/// Collect a step for every agent that owes one, releasing the lock before
/// returning so the calls can run concurrently.
async fn plan_batch(
    ctx: &SystemContext,
    approver: Option<Arc<dyn Approver>>,
    stream_sink: Option<tokio::sync::mpsc::UnboundedSender<StreamChunk>>,
) -> Vec<Step> {
    let service = LlmService::read(ctx).await;
    let registry = ctx
        .with_world(|world| world.resource::<ToolRegistry>().cloned())
        .await
        .unwrap_or_default();
    let store = MemoryService::read(ctx).await;

    ctx.with_world(|world| {
        let journal = world.resource::<Journal>().cloned();
        let mut steps = Vec::new();
        for (entity, agent) in world.query::<(Entity, &ReactAgent)>() {
            if let Some(step) = step_for(
                entity,
                agent,
                &service,
                &registry,
                store.as_ref(),
                &approver,
                &stream_sink,
                &journal,
            ) {
                steps.push(step);
            }
        }
        steps
    })
    .await
}

/// Collect the next step for a single agent, so it can advance without waiting
/// on the others. Returns `None` once the agent no longer owes a call.
async fn plan_one(
    ctx: &SystemContext,
    entity: Entity,
    approver: Option<Arc<dyn Approver>>,
    stream_sink: Option<tokio::sync::mpsc::UnboundedSender<StreamChunk>>,
) -> Option<Step> {
    let service = LlmService::read(ctx).await;
    let registry = ctx
        .with_world(|world| world.resource::<ToolRegistry>().cloned())
        .await
        .unwrap_or_default();
    let store = MemoryService::read(ctx).await;

    ctx.with_world(|world| {
        let journal = world.resource::<Journal>().cloned();
        let agent = world.get::<ReactAgent>(entity)?;
        step_for(
            entity,
            agent,
            &service,
            &registry,
            store.as_ref(),
            &approver,
            &stream_sink,
            &journal,
        )
    })
    .await
}

/// Build the step one agent owes, or `None` when it is settled or parked without
/// a decision. Shared by the batch seed and the per-agent continuation so both
/// plan identically.
#[allow(clippy::too_many_arguments)]
fn step_for(
    entity: Entity,
    agent: &ReactAgent,
    service: &Option<LlmService>,
    registry: &ToolRegistry,
    store: Option<&MemoryService>,
    approver: &Option<Arc<dyn Approver>>,
    stream_sink: &Option<tokio::sync::mpsc::UnboundedSender<StreamChunk>>,
    journal: &Option<Journal>,
) -> Option<Step> {
    let turn = agent.spend.turns;
    if agent.has_decision() {
        if let (Some(request), Some(decision)) = (agent.awaiting.clone(), agent.decision.clone()) {
            return Some(Step::Resume(Box::new(Resume {
                entity,
                turn,
                request,
                decision,
                registry: registry.clone(),
                tool_ctx: tool_context(entity, agent, store),
                approver: approver.clone(),
                journal: journal.clone(),
            })));
        }
        return None;
    }
    if !agent.awaits_model() {
        return None;
    }
    if let Err(error) = agent.guardrails.check(&agent.spend, agent.elapsed()) {
        return Some(Step::Fail { entity, error });
    }
    let Some(service) = service.clone() else {
        return Some(Step::Fail {
            entity,
            error: AgentError::config("this world has no LlmService resource for the agent to ask"),
        });
    };
    let descriptors = agent.guardrails.filter_tools(registry.descriptors());
    Some(Step::Generate(Box::new(Generation {
        entity,
        turn,
        client: service.shared(),
        request: agent
            .build_request(descriptors)
            .with_correlation(agent.name.clone(), turn),
        registry: registry.clone(),
        tool_ctx: tool_context(entity, agent, store),
        approver: approver.clone(),
        format: agent.format.clone(),
        repairs_left: agent.repairs_left,
        max_tool_calls: agent.guardrails.max_tool_calls,
        spend: agent.spend,
        stream_sink: stream_sink.clone(),
        journal: journal.clone(),
    })))
}

impl Generation {
    /// One model call plus the tool round it asks for.
    async fn execute(self) -> Outcome {
        let response = if let Some(sink) = &self.stream_sink {
            match self.client.stream(&self.request).await {
                Ok(mut stream) => {
                    let mut chunks = Vec::new();
                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(chunk) => {
                                let _ = sink.send(chunk.clone());
                                chunks.push(chunk);
                            }
                            Err(error) => {
                                return Outcome::Failed {
                                    entity: self.entity,
                                    error: AgentError::provider(&error),
                                };
                            }
                        }
                    }
                    Box::new(StreamChunk::reduce(chunks))
                }
                Err(error) => {
                    return Outcome::Failed {
                        entity: self.entity,
                        error: AgentError::provider(&error),
                    };
                }
            }
        } else {
            match self.client.generate(&self.request).await {
                Ok(response) => Box::new(response),
                Err(error) => {
                    return Outcome::Failed {
                        entity: self.entity,
                        error: AgentError::provider(&error),
                    };
                }
            }
        };
        if let Some(journal) = &self.journal {
            journal.record(&self.request, response.as_ref());
        }
        if response.tool_calls.is_empty() {
            return self.settle(response);
        }
        if let Some(journal) = &self.journal
            && let Some((results, refused)) = journal.lookup_tools(&self.request)
        {
            return Outcome::Acted {
                entity: self.entity,
                turn: self.turn,
                response,
                results,
                refused,
                parked: None,
            };
        }
        let room = if self.max_tool_calls == 0 {
            usize::MAX
        } else {
            self.max_tool_calls.saturating_sub(self.spend.tool_calls)
        };
        let (results, refused, parked) = run_calls(
            self.entity,
            self.turn,
            &response.tool_calls,
            room,
            &self.registry,
            &self.tool_ctx,
            self.approver.as_ref(),
        )
        .await;
        if parked.is_none()
            && let Some(journal) = &self.journal
        {
            journal.record_tools(&self.request, results.clone(), refused);
        }
        if let Some(parked) = &parked
            && let Some(journal) = &self.journal
            && let Some(correlation) = &self.request.correlation
        {
            journal.record_pending(
                approval_key(&correlation.agent, correlation.turn, &parked.call.id),
                self.request.clone(),
                results.clone(),
                refused,
            );
        }
        Outcome::Acted {
            entity: self.entity,
            turn: self.turn,
            response,
            results,
            refused,
            parked,
        }
    }

    /// Turn a text answer into a finished task, a repair round, or a failure.
    fn settle(self, response: Box<GenerationResponse>) -> Outcome {
        let content = response.content.clone();
        if self.format.is_text() {
            return Outcome::Answered {
                entity: self.entity,
                turn: self.turn,
                response,
                parsed: None,
            };
        }
        match self.format.parse(&content) {
            Ok(Some(value)) => Outcome::Answered {
                entity: self.entity,
                turn: self.turn,
                response,
                parsed: Some(value),
            },
            Ok(None) if self.repairs_left > 0 => Outcome::Repaired {
                entity: self.entity,
                turn: self.turn,
                response,
                repair: "The answer was empty. Reply again with only the JSON value.".to_string(),
            },
            Ok(None) => Outcome::Failed {
                entity: self.entity,
                error: AgentError::output("the answer was empty", content),
            },
            Err(error) if self.repairs_left > 0 => Outcome::Repaired {
                entity: self.entity,
                turn: self.turn,
                response,
                repair: self.format.repair(&error).content,
            },
            Err(error) => Outcome::Failed {
                entity: self.entity,
                error,
            },
        }
    }

}

impl Resume {
    /// Run (or refuse) the parked call, then carry on with whatever follows.
    ///
    /// When the recording journal parked this approval, the whole round — the
    /// results already in hand plus what the decision produces — is journaled
    /// under the original request, so a replay feeds it back without asking a
    /// human again or re-running the approved tool.
    async fn execute(self) -> Outcome {
        let call = self.request.call.clone();
        let session = self.tool_ctx.session.as_deref().unwrap_or_default();
        let turn = self.request.turn as usize;
        let pending = self
            .journal
            .as_ref()
            .and_then(|journal| journal.take_pending(&approval_key(session, turn, &call.id)));
        let (base_request, mut done, base_refused) = match pending {
            Some((request, done, refused)) => (Some(request), done, refused),
            None => (None, Vec::new(), 0),
        };

        let (results, refused, parked) = match &self.decision {
            Decision::Approved => {
                let first = match self.registry.execute(&self.tool_ctx, &call).await {
                    Ok(outcome) => vec![(call.clone(), outcome)],
                    Err(error) => vec![(
                        call.clone(),
                        ToolOutcome::error(format!("the tool could not run: {error}")),
                    )],
                };
                let rest = self.request.remaining.clone();
                if rest.is_empty() {
                    (first, 0, None)
                } else {
                    let (more, refused, parked) = run_calls(
                        self.entity,
                        turn,
                        &rest,
                        usize::MAX,
                        &self.registry,
                        &self.tool_ctx,
                        self.approver.as_ref(),
                    )
                    .await;
                    let mut results = first;
                    results.extend(more);
                    (results, refused, parked)
                }
            }
            Decision::Rejected { reason } => {
                let outcome =
                    ToolOutcome::error(format!("a human refused to run `{}`: {reason}", call.name));
                (vec![(call.clone(), outcome)], 1, None)
            }
        };

        if let Some(journal) = &self.journal
            && let Some(request) = &base_request
        {
            done.extend(results.clone());
            let total_refused = base_refused + refused;
            match &parked {
                Some(parked) => journal.record_pending(
                    approval_key(session, turn, &parked.call.id),
                    request.clone(),
                    done,
                    total_refused,
                ),
                None => journal.record_tools(request, done, total_refused),
            }
        }
        Outcome::Observed {
            entity: self.entity,
            turn: self.turn,
            results,
            refused,
            parked,
        }
    }
}

/// Run calls in the order the model asked for them.
///
/// When every call is read-only, registered, approved and within budget, they
/// run concurrently. Otherwise they run sequentially: a model that asks for two
/// calls usually means the second to see the first, and a tool that writes
/// cannot be rolled back if the loop stops halfway.
///
/// Returns the results, how many calls were refused outright, and a request to
/// park on if a call needs a human that no [`Approver`] can supply.
async fn run_calls(
    entity: Entity,
    turn: usize,
    calls: &[ToolCall],
    room: usize,
    registry: &ToolRegistry,
    tool_ctx: &ToolContext,
    approver: Option<&Arc<dyn Approver>>,
) -> (Vec<(ToolCall, ToolOutcome)>, usize, Option<ApprovalRequest>) {
    if calls.len() > 1
        && calls.len() <= room
        && calls.iter().all(|call| {
            registry.contains(&call.name)
                && registry.is_readonly(&call.name)
                && !registry.requires_approval(&call.name)
                && registry.validate(call).is_empty()
        })
    {
        let results = futures::future::join_all(calls.iter().map(|call| {
            let registry = registry.clone();
            let ctx = tool_ctx.clone();
            async move {
                let outcome = registry
                    .execute(&ctx, call)
                    .await
                    .unwrap_or_else(|error| ToolOutcome::error(error.to_string()));
                (call.clone(), outcome)
            }
        }))
        .await;
        return (results, 0, None);
    }
    let mut results = Vec::with_capacity(calls.len());
    let mut refused = 0usize;
    let mut ran = 0usize;
    for (index, call) in calls.iter().enumerate() {
        if ran >= room {
            refused += 1;
            results.push((
                call.clone(),
                ToolOutcome::error(format!(
                    "`{}` was not run: this task has used its tool-call budget",
                    call.name
                )),
            ));
            continue;
        }
        if !registry.contains(&call.name) {
            refused += 1;
            results.push((
                call.clone(),
                ToolOutcome::error(format!("unknown tool `{}`", call.name)),
            ));
            continue;
        }
        if registry.requires_approval(&call.name) {
            let request = ApprovalRequest::new(
                entity,
                turn as u64,
                call,
                calls
                    .iter()
                    .skip(index + 1)
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            let Some(approver) = approver else {
                return (results, refused, Some(request));
            };
            let decision = approver.decide(&request).await;
            if !decision.is_approved() {
                refused += 1;
                results.push((
                    call.clone(),
                    ToolOutcome::error(decision.transcript_text()),
                ));
                continue;
            }
        }
        let problems = registry.validate(call);
        if !problems.is_empty() {
            refused += 1;
            results.push((call.clone(), ToolOutcome::error(problems.join("; "))));
            continue;
        }
        match registry.execute(tool_ctx, call).await {
            Ok(outcome) => {
                ran += 1;
                results.push((call.clone(), outcome));
            }
            Err(error) => {
                refused += 1;
                results.push((call.clone(), ToolOutcome::error(error.to_string())));
            }
        }
    }
    (results, refused, None)
}

/// Write an outcome back, returning whether anything changed.
///
/// The turn count and status are checked first: a step planned against turn 3
/// is worthless once the agent is on turn 4, and applying it would duplicate
/// tool results in the transcript.
async fn apply(ctx: SystemContext, outcome: Outcome) -> bool {
    let expected_turn = match &outcome {
        Outcome::Answered { turn, .. }
        | Outcome::Repaired { turn, .. }
        | Outcome::Acted { turn, .. }
        | Outcome::Observed { turn, .. } => *turn,
        Outcome::Failed { .. } => usize::MAX,
    };
    ctx.with_world_mut(move |world| {
        let entity = outcome_entity(&outcome);
        let provider = LlmService::from_world(world)
            .map(|service| service.provider_id())
            .unwrap_or_default();
        let Some(agent) = world.get_mut::<ReactAgent>(entity) else {
            tracing::debug!(%entity, "the agent the step was planned for is gone");
            return false;
        };
        if outcome_turns(&outcome) && agent.spend.turns != expected_turn {
            tracing::debug!(
                %entity,
                expected_turn,
                actual = agent.spend.turns,
                "stale step dropped"
            );
            return false;
        }
        let old_status = agent.status;
        let mut events = Vec::new();
        let wrote = match outcome {
            Outcome::Answered {
                response,
                parsed,
                ..
            } => {
                if !agent.awaits_model() {
                    return false;
                }
                agent.record_response(&response);
                events.push(model_event(&provider, &response, false));
                let content = answer_text(&response);
                match parsed {
                    Some(value) => agent.finish_parsed(content, value),
                    None => agent.finish_text(content),
                }
                true
            }
            Outcome::Repaired {
                response, repair, ..
            } => {
                if !agent.awaits_model() {
                    return false;
                }
                agent.record_response(&response);
                events.push(model_event(&provider, &response, false));
                agent.repairs_left = agent.repairs_left.saturating_sub(1);
                agent.conversation.push_user(repair);
                true
            }
            Outcome::Acted {
                response,
                results,
                refused,
                parked,
                ..
            } => {
                if !agent.awaits_model() {
                    return false;
                }
                agent.record_response(&response);
                events.push(model_event(&provider, &response, false));
                for (call, outcome) in &results {
                    events.push(EventKind::Tool {
                        name: call.name.clone(),
                        ok: outcome.ok,
                    });
                }
                agent.spend.denied += refused;
                for (call, outcome) in &results {
                    agent.record_tool_result(call, outcome);
                }
                if let Some(request) = parked {
                    agent.park(request);
                }
                true
            }
            Outcome::Observed {
                results,
                refused,
                parked,
                ..
            } => {
                if agent.status != AgentStatus::AwaitingApproval {
                    return false;
                }
                for (call, outcome) in &results {
                    events.push(EventKind::Tool {
                        name: call.name.clone(),
                        ok: outcome.ok,
                    });
                }
                agent.spend.denied += refused;
                for (call, outcome) in &results {
                    agent.record_tool_result(call, outcome);
                }
                match parked {
                    Some(request) => agent.park(request),
                    None => {
                        agent.status = AgentStatus::Working;
                        agent.awaiting = None;
                        agent.decision = None;
                    }
                }
                true
            }
            Outcome::Failed { error, .. } => {
                if !agent.status.is_open() || agent.status == AgentStatus::Idle {
                    return false;
                }
                if error.kind() == "provider" {
                    events.push(EventKind::Model {
                        provider: provider.clone(),
                        model: String::new(),
                        input_tokens: 0,
                        output_tokens: 0,
                        failed: true,
                    });
                }
                agent.fail(&error);
                true
            }
        };
        if wrote && agent.status != old_status {
            events.push(EventKind::Agent {
                name: agent.name().to_string(),
                status: agent.status.as_str().to_string(),
            });
        }
        if let Some(observer) = world.resource_mut::<Observer>() {
            for event in events {
                observer.record(event);
            }
        }
        wrote
    })
    .await
}

fn model_event(provider: &str, response: &GenerationResponse, failed: bool) -> EventKind {
    EventKind::Model {
        provider: provider.to_string(),
        model: response.model.clone(),
        input_tokens: response.usage.prompt_tokens,
        output_tokens: response.usage.completion_tokens,
        failed,
    }
}

/// Fold every conversation that has grown past `threshold`, outside the lock.
///
/// Windows are cloned under a read lock, summarised concurrently, then written
/// back under a write lock only if the agent is still on the same turn.
async fn fold_overlong(ctx: &SystemContext, threshold: usize, keep_last: usize) {
    let service = LlmService::read(ctx).await;
    let (journal, snapshots): (Option<Journal>, Vec<(Entity, usize, String, ConversationWindow)>) =
        ctx.with_world(|world| {
            let journal = world.resource::<Journal>().cloned();
            let snapshots = world
                .query::<(Entity, &ReactAgent)>()
                .filter(|(_, agent)| {
                    agent.status.is_open()
                        && agent.conversation.turns().len() > threshold
                })
                .map(|(entity, agent)| {
                    (
                        entity,
                        agent.spend.turns,
                        agent.name().to_string(),
                        agent.conversation.clone(),
                    )
                })
                .collect();
            (journal, snapshots)
        })
        .await;
    if snapshots.is_empty() {
        return;
    }
    // One summariser per agent: the fold's model call carries that agent's
    // correlation so a recorded fold replays into the right window, even when
    // several agents fold identical transcripts in the same activation.
    let folded = futures::future::join_all(snapshots.into_iter().map(
        |(entity, turn, name, mut window)| {
            let service = service.clone();
            let journal = journal.clone();
            async move {
                let summariser: Box<dyn Summariser> = match &service {
                    Some(service) => {
                        let mut chat = crate::memory::ChatSummariser::new(service.shared())
                            .with_correlation(name, window.fold_count());
                        if let Some(journal) = journal {
                            chat = chat.with_journal(journal);
                        }
                        Box::new(chat)
                    }
                    None => Box::new(HeuristicSummariser::new(6, 160)),
                };
                let result = window.fold(summariser.as_ref(), keep_last).await;
                (entity, turn, window, result)
            }
        },
    ))
    .await;
    ctx.with_world_mut(|world| {
        for (entity, turn, window, result) in folded {
            let Ok(summary) = result else {
                tracing::warn!(%entity, "a conversation fold failed");
                continue;
            };
            let Some(agent) = world.get_mut::<ReactAgent>(entity) else {
                continue;
            };
            if agent.spend.turns != turn {
                continue;
            }
            agent.conversation = window;
            if summary.is_some() {
                tracing::debug!(%entity, "folded a long conversation");
            }
        }
    })
    .await;
}

async fn update_work(ctx: &SystemContext, label: &str) {
    let busy = ctx
        .with_world(|world| {
            // A parked agent waits on a human, not on compute; only a park a
            // human has already decided still owes a resume.
            world
                .query::<&ReactAgent>()
                .any(|agent| agent.status == AgentStatus::Working || agent.has_decision())
        })
        .await;
    ctx.with_world_mut(|world| Work::ensure(world).set_owned("agents", busy, label))
        .await;
}

fn outcome_entity(outcome: &Outcome) -> Entity {
    match outcome {
        Outcome::Answered { entity, .. }
        | Outcome::Repaired { entity, .. }
        | Outcome::Acted { entity, .. }
        | Outcome::Observed { entity, .. }
        | Outcome::Failed { entity, .. } => *entity,
    }
}

/// Whether this outcome must match the turn it was planned against.
fn outcome_turns(outcome: &Outcome) -> bool {
    !matches!(outcome, Outcome::Failed { .. })
}

/// The answer text, falling back to the calls made when the model said nothing.
fn answer_text(response: &GenerationResponse) -> String {
    if response.content.trim().is_empty() {
        let names = response
            .tool_calls
            .iter()
            .map(|call| call.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return if names.is_empty() {
            String::new()
        } else {
            format!("(called tools: {names})")
        };
    }
    response.content.clone()
}

/// The context tools are run with: who asked, which turn, and what memory.
fn tool_context(entity: Entity, agent: &ReactAgent, store: Option<&MemoryService>) -> ToolContext {
    let context = ToolContext::new(entity)
        .with_session(agent.name.clone())
        .with_turn(agent.spend.turns as u64);
    match store {
        Some(store) => context.with_store(store.shared()),
        None => context,
    }
}

/// Tell every busy agent that the runtime is shutting down.
async fn stop_agents(ctx: &SystemContext) {
    let reason = ctx
        .stop_token()
        .reason()
        .map(|reason| reason.into_owned())
        .unwrap_or_else(|| "the runtime stopped".to_string());
    ctx.with_world_mut(|world| {
        let busy: Vec<Entity> = world
            .query::<(Entity, &ReactAgent)>()
            .filter(|(_, agent)| agent.status != AgentStatus::Idle && agent.status.is_open())
            .map(|(entity, _)| entity)
            .collect();
        let mut events = Vec::new();
        for entity in busy {
            if let Some(agent) = world.get_mut::<ReactAgent>(entity) {
                agent.fail_with(format!("stopped: {reason}"));
                events.push(EventKind::Agent {
                    name: agent.name().to_string(),
                    status: agent.status.as_str().to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use xihe_core::World;
    use xihe_llm::{MockProvider, Reply};
    use xihe_schedule::{System, WorldHandle};

    use crate::{AlwaysApprove, AlwaysReject, EchoTool, FnTool, LlmService, ResponseFormat, Tool, ToolRegistry};

    fn world_with(provider: MockProvider, registry: ToolRegistry, agent: ReactAgent) -> SystemContext {
        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(LlmService::from_provider(provider));
        world.spawn_with((agent,));
        SystemContext::new(WorldHandle::new(world))
    }

    async fn only_agent(ctx: &SystemContext) -> ReactAgent {
        let world = ctx.world.read().await;
        let entity = world.single::<ReactAgent>().expect("one agent");
        world.get::<ReactAgent>(entity).expect("present").clone()
    }

    #[tokio::test]
    async fn a_tool_round_then_an_answer() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let ctx = world_with(
            MockProvider::tool_then_answer("echo", json!({"text": "hi"}), "echoed hi"),
            registry,
            ReactAgent::new("scribe", "be brief").ask("say hi"),
        );
        React::new().run(ctx.clone()).await.expect("the loop runs");
        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Finished);
        assert_eq!(agent.answer(), Some("echoed hi"));
        assert_eq!(agent.turns(), 2);
        assert_eq!(agent.spend().tool_calls, 1);
        assert!(agent.conversation().render().contains("hi"));
    }

    #[tokio::test]
    async fn a_tool_round_replays_the_assistant_tool_calls() {
        use std::sync::Mutex;
        use xihe_llm::{LlmError, Provider, Role};

        struct CapturingProvider {
            requests: Arc<Mutex<Vec<GenerationRequest>>>,
            calls: Mutex<usize>,
        }

        #[async_trait::async_trait]
        impl Provider for CapturingProvider {
            fn id(&self) -> &str {
                "capture"
            }
            fn default_model(&self) -> &str {
                "capture"
            }
            async fn generate(
                &self,
                request: &GenerationRequest,
            ) -> Result<GenerationResponse, LlmError> {
                self.requests.lock().unwrap().push(request.clone());
                let mut calls = self.calls.lock().unwrap();
                let response = if *calls == 0 {
                    GenerationResponse::default().with_tool_calls(vec![ToolCall::new(
                        "call_1",
                        "echo",
                        json!({"text": "hi"}),
                    )])
                } else {
                    GenerationResponse::text("echoed hi")
                };
                *calls += 1;
                Ok(response)
            }
        }

        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = CapturingProvider {
            requests: requests.clone(),
            calls: Mutex::new(0),
        };
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(LlmService::from_provider(provider));
        world.spawn_with((ReactAgent::new("scribe", "be brief").ask("say hi"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the loop runs");

        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Finished);

        let captured = requests.lock().unwrap();
        assert_eq!(captured.len(), 2, "a tool round then an answer round");
        let second = &captured[1];
        let tool_msg = second
            .messages
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("a tool result is replayed");
        let call_id = tool_msg
            .tool_call_id
            .clone()
            .expect("the tool result names its call");
        let assistant = second
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .expect("an assistant turn precedes the tool result");
        assert!(
            assistant.tool_calls.iter().any(|call| call.id == call_id),
            "the replayed assistant turn must carry the tool_calls its tool result answers: {assistant:?}"
        );
    }

    #[tokio::test]
    async fn a_recorded_run_replays_identically() {
        use std::sync::Mutex;
        use xihe_llm::{LlmError, Provider};

        use crate::replay::{Journal, ReplayProvider};

        // A provider whose answer changes on every call. Re-running it would
        // not reproduce the first run, so an identical replay proves the
        // journal fed the recorded responses back rather than the model
        // answering again.
        struct VaryingProvider {
            calls: Mutex<usize>,
        }

        #[async_trait::async_trait]
        impl Provider for VaryingProvider {
            fn id(&self) -> &str {
                "varying"
            }
            fn default_model(&self) -> &str {
                "varying"
            }
            async fn generate(
                &self,
                _request: &GenerationRequest,
            ) -> Result<GenerationResponse, LlmError> {
                let mut calls = self.calls.lock().unwrap();
                let n = *calls;
                *calls += 1;
                Ok(if n == 0 {
                    GenerationResponse::default().with_tool_calls(vec![ToolCall::new(
                        "call_1",
                        "echo",
                        json!({"text": format!("v{n}")}),
                    )])
                } else {
                    GenerationResponse::text(format!("answer-{n}"))
                })
            }
        }

        let journal = Journal::new();
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let mut world = World::new();
        world.insert_resource(registry.clone());
        world.insert_resource(LlmService::from_provider(VaryingProvider {
            calls: Mutex::new(0),
        }));
        world.insert_resource(journal.clone());
        world.spawn_with((ReactAgent::new("scribe", "be brief").ask("say hi"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the record run");
        let recorded = only_agent(&ctx).await;
        assert_eq!(recorded.status(), AgentStatus::Finished);
        assert!(journal.len() >= 2, "both model calls were journaled");

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(LlmService::from_provider(ReplayProvider::new(journal)));
        world.spawn_with((ReactAgent::new("scribe", "be brief").ask("say hi"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the replay run");
        let replayed = only_agent(&ctx).await;

        assert_eq!(replayed.status(), AgentStatus::Finished);
        assert_eq!(replayed.answer(), recorded.answer());
        assert_eq!(replayed.turns(), recorded.turns());
        assert_eq!(
            replayed.conversation().render(),
            recorded.conversation().render(),
            "replay must reproduce the transcript turn for turn"
        );
    }

    #[tokio::test]
    async fn parallel_agents_with_identical_requests_replay_their_own_answers() {
        use std::sync::Mutex;
        use xihe_llm::{LlmError, Provider};

        use crate::replay::{Journal, ReplayProvider};

        // A different answer on every call. Two agents with the same
        // instructions and the same question issue byte-identical requests, so
        // only a stable per-agent key can keep their recorded answers apart.
        struct CountingProvider {
            calls: Mutex<usize>,
        }

        #[async_trait::async_trait]
        impl Provider for CountingProvider {
            fn id(&self) -> &str {
                "counting"
            }
            fn default_model(&self) -> &str {
                "counting"
            }
            async fn generate(
                &self,
                _request: &GenerationRequest,
            ) -> Result<GenerationResponse, LlmError> {
                let mut calls = self.calls.lock().unwrap();
                let n = *calls;
                *calls += 1;
                Ok(GenerationResponse::text(format!("answer-{n}")))
            }
        }

        async fn answers(ctx: &SystemContext) -> (String, String) {
            let world = ctx.world.read().await;
            let mut alpha = None;
            let mut beta = None;
            for (_entity, agent) in world.query::<(Entity, &ReactAgent)>() {
                match agent.name() {
                    "alpha" => alpha = Some(agent.answer().unwrap_or("").to_string()),
                    "beta" => beta = Some(agent.answer().unwrap_or("").to_string()),
                    _ => {}
                }
            }
            (alpha.expect("alpha answered"), beta.expect("beta answered"))
        }

        let journal = Journal::new();
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let mut world = World::new();
        world.insert_resource(registry.clone());
        world.insert_resource(LlmService::from_provider(CountingProvider {
            calls: Mutex::new(0),
        }));
        world.insert_resource(journal.clone());
        world.spawn_with((ReactAgent::new("alpha", "be brief").ask("say hi"),));
        world.spawn_with((ReactAgent::new("beta", "be brief").ask("say hi"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the record run");
        let (recorded_alpha, recorded_beta) = answers(&ctx).await;
        assert_ne!(
            recorded_alpha, recorded_beta,
            "the provider must give the two agents different answers"
        );

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(LlmService::from_provider(ReplayProvider::new(journal)));
        world.spawn_with((ReactAgent::new("alpha", "be brief").ask("say hi"),));
        world.spawn_with((ReactAgent::new("beta", "be brief").ask("say hi"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the replay run");
        let (replayed_alpha, replayed_beta) = answers(&ctx).await;

        assert_eq!(
            replayed_alpha, recorded_alpha,
            "alpha must replay its own answer, not beta's"
        );
        assert_eq!(
            replayed_beta, recorded_beta,
            "beta must replay its own answer, not alpha's"
        );
    }

    #[tokio::test]
    async fn replay_does_not_re_execute_tools() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::replay::{Journal, ReplayProvider};

        let journal = Journal::new();
        let executions = Arc::new(AtomicUsize::new(0));

        let mut registry = ToolRegistry::new();
        let counter = executions.clone();
        registry.register(FnTool::new("deploy", "ship it", move |_ctx, _call| {
            counter.fetch_add(1, Ordering::SeqCst);
            ToolOutcome::ok(json!({"shipped": true}), "shipped to staging")
        }));

        let mut world = World::new();
        world.insert_resource(registry.clone());
        world.insert_resource(LlmService::from_provider(MockProvider::tool_then_answer(
            "deploy",
            json!({}),
            "staging is live",
        )));
        world.insert_resource(journal.clone());
        world.spawn_with((ReactAgent::new("ops", "ship carefully").ask("ship"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the record run");
        let recorded = only_agent(&ctx).await;
        assert_eq!(recorded.answer(), Some("staging is live"));
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "the tool ran once while recording"
        );

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(journal.clone());
        world.insert_resource(LlmService::from_provider(ReplayProvider::new(journal)));
        world.spawn_with((ReactAgent::new("ops", "ship carefully").ask("ship"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the replay run");
        let replayed = only_agent(&ctx).await;

        assert_eq!(replayed.answer(), recorded.answer());
        assert_eq!(
            replayed.conversation().render(),
            recorded.conversation().render()
        );
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "replay must feed the recorded tool result, not re-run the side effect"
        );
    }

    #[tokio::test]
    async fn a_journal_survives_a_byte_round_trip_for_replay() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::replay::{Journal, ReplayProvider};

        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        let counter = executions.clone();
        registry.register(FnTool::new("deploy", "ship it", move |_ctx, _call| {
            counter.fetch_add(1, Ordering::SeqCst);
            ToolOutcome::ok(json!({"shipped": true}), "shipped to staging")
        }));

        let journal = Journal::new();
        let mut world = World::new();
        world.insert_resource(registry.clone());
        world.insert_resource(LlmService::from_provider(MockProvider::tool_then_answer(
            "deploy",
            json!({}),
            "staging is live",
        )));
        world.insert_resource(journal.clone());
        world.spawn_with((ReactAgent::new("ops", "ship carefully").ask("ship"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the record run");
        let recorded = only_agent(&ctx).await;
        assert_eq!(executions.load(Ordering::SeqCst), 1, "the tool ran once");

        // Persist to bytes and rebuild a fresh journal from them alone: no
        // shared state with the recording journal, as a second process would.
        let bytes = journal.to_bytes().expect("serialize the journal");
        drop(journal);
        let replay_journal = Journal::from_bytes(&bytes).expect("deserialize the journal");

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(replay_journal.clone());
        world.insert_resource(LlmService::from_provider(ReplayProvider::new(replay_journal)));
        world.spawn_with((ReactAgent::new("ops", "ship carefully").ask("ship"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the replay run");
        let replayed = only_agent(&ctx).await;

        assert_eq!(replayed.answer(), recorded.answer());
        assert_eq!(
            replayed.conversation().render(),
            recorded.conversation().render()
        );
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "replay from bytes must not re-run the tool"
        );
    }

    #[tokio::test]
    async fn a_journal_saved_to_disk_replays_without_re_running_tools() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::replay::{Journal, ReplayProvider};

        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        let counter = executions.clone();
        registry.register(FnTool::new("deploy", "ship it", move |_ctx, _call| {
            counter.fetch_add(1, Ordering::SeqCst);
            ToolOutcome::ok(json!({"shipped": true}), "shipped to staging")
        }));

        let journal = Journal::new();
        let mut world = World::new();
        world.insert_resource(registry.clone());
        world.insert_resource(LlmService::from_provider(MockProvider::tool_then_answer(
            "deploy",
            json!({}),
            "staging is live",
        )));
        world.insert_resource(journal.clone());
        world.spawn_with((ReactAgent::new("ops", "ship carefully").ask("ship"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the record run");
        let recorded = only_agent(&ctx).await;
        assert_eq!(executions.load(Ordering::SeqCst), 1, "the tool ran once");

        // Persist to a real file, then drop the recording journal entirely.
        let path = std::env::temp_dir().join(format!("xihe-journal-{}.json", std::process::id()));
        journal.save(&path).expect("save to disk");
        drop(journal);

        // A fresh journal loaded from disk shares nothing with the recorder.
        let replay_journal = Journal::load(&path).expect("load from disk");
        std::fs::remove_file(&path).ok();

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(replay_journal.clone());
        world.insert_resource(LlmService::from_provider(ReplayProvider::new(replay_journal)));
        world.spawn_with((ReactAgent::new("ops", "ship carefully").ask("ship"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the replay run");
        let replayed = only_agent(&ctx).await;

        assert_eq!(replayed.answer(), recorded.answer());
        assert_eq!(
            replayed.conversation().render(),
            recorded.conversation().render()
        );
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "replay from disk must not re-run the tool"
        );
    }

    #[tokio::test]
    async fn a_human_approved_tool_replays_without_re_running_or_re_approving() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::replay::{Journal, ReplayProvider};

        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        let counter = executions.clone();
        registry.register(
            FnTool::new("deploy", "ship it", move |_ctx, _call| {
                counter.fetch_add(1, Ordering::SeqCst);
                ToolOutcome::ok(json!({"shipped": true}), "shipped to staging")
            })
            .requiring_approval(),
        );

        let journal = Journal::new();
        let mut world = World::new();
        world.insert_resource(registry.clone());
        world.insert_resource(LlmService::from_provider(MockProvider::tool_then_answer(
            "deploy",
            json!({}),
            "staging is live",
        )));
        world.insert_resource(journal.clone());
        world.spawn_with((ReactAgent::new("ops", "ship carefully").ask("ship"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the record run parks");
        assert_eq!(
            only_agent(&ctx).await.status(),
            AgentStatus::AwaitingApproval
        );
        assert_eq!(
            executions.load(Ordering::SeqCst),
            0,
            "nothing runs before a human approves"
        );

        {
            let mut world = ctx.world.write().await;
            let entity = world.single::<ReactAgent>().expect("one agent");
            world
                .get_mut::<ReactAgent>(entity)
                .expect("present")
                .approve();
        }
        React::new().run(ctx.clone()).await.expect("the resume run");
        let recorded = only_agent(&ctx).await;
        assert_eq!(recorded.status(), AgentStatus::Finished);
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "the approved tool ran once while recording"
        );

        let bytes = journal.to_bytes().expect("serialize the journal");
        drop(journal);
        let replay_journal = Journal::from_bytes(&bytes).expect("deserialize the journal");

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(replay_journal.clone());
        world.insert_resource(LlmService::from_provider(ReplayProvider::new(replay_journal)));
        world.spawn_with((ReactAgent::new("ops", "ship carefully").ask("ship"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the replay run");
        let replayed = only_agent(&ctx).await;

        assert_eq!(
            replayed.status(),
            AgentStatus::Finished,
            "replay must not park waiting for a human again"
        );
        assert_eq!(replayed.answer(), recorded.answer());
        assert_eq!(
            replayed.conversation().render(),
            recorded.conversation().render()
        );
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "replay must feed the recorded approval result, not re-run the side effect"
        );
    }

    #[tokio::test]
    async fn a_folded_run_replays_without_re_summarising() {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use xihe_llm::{LlmError, Provider};

        use crate::replay::{Journal, ReplayProvider};

        // A provider that answers generation calls (those offering tools) with
        // a few tool rounds then an answer, and fold calls (no tools) with a
        // fixed summary. The fold summary differs from anything the heuristic
        // fallback would produce, so a replay that re-summarises instead of
        // feeding the recorded one diverges.
        struct FoldAwareProvider {
            generations: Mutex<usize>,
        }

        #[async_trait::async_trait]
        impl Provider for FoldAwareProvider {
            fn id(&self) -> &str {
                "fold-aware"
            }
            fn default_model(&self) -> &str {
                "fold-aware"
            }
            async fn generate(
                &self,
                request: &GenerationRequest,
            ) -> Result<GenerationResponse, LlmError> {
                if request.tools.is_empty() {
                    return Ok(GenerationResponse::text("a folded summary of earlier turns"));
                }
                let mut calls = self.generations.lock().unwrap();
                let n = *calls;
                *calls += 1;
                Ok(if n < 3 {
                    GenerationResponse::default().with_tool_calls(vec![ToolCall::new(
                        format!("c{n}"),
                        "echo",
                        json!({"text": format!("t{n}")}),
                    )])
                } else {
                    GenerationResponse::text("all done")
                })
            }
        }

        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        let counter = executions.clone();
        registry.register(
            FnTool::new("echo", "echo it", move |_ctx, call| {
                counter.fetch_add(1, Ordering::SeqCst);
                ToolOutcome::text(call.string_argument("text").unwrap_or("").to_string())
            })
            .with_required_string("text", "the text to echo"),
        );

        let journal = Journal::new();
        let mut world = World::new();
        world.insert_resource(registry.clone());
        world.insert_resource(LlmService::from_provider(FoldAwareProvider {
            generations: Mutex::new(0),
        }));
        world.insert_resource(journal.clone());
        world.spawn_with((ReactAgent::new("scribe", "be brief").ask("go"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new()
            .with_fold(3, 2)
            .run(ctx.clone())
            .await
            .expect("the record run");
        let recorded = only_agent(&ctx).await;
        assert_eq!(recorded.status(), AgentStatus::Finished);
        assert_eq!(recorded.answer(), Some("all done"));
        assert!(
            recorded.conversation().fold_count() > 0,
            "the recording must actually have folded"
        );
        assert_eq!(executions.load(Ordering::SeqCst), 3, "three tool rounds ran");

        let bytes = journal.to_bytes().expect("serialize the journal");
        drop(journal);
        let replay_journal = Journal::from_bytes(&bytes).expect("deserialize the journal");

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(replay_journal.clone());
        world.insert_resource(LlmService::from_provider(ReplayProvider::new(replay_journal)));
        world.spawn_with((ReactAgent::new("scribe", "be brief").ask("go"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new()
            .with_fold(3, 2)
            .run(ctx.clone())
            .await
            .expect("the replay run");
        let replayed = only_agent(&ctx).await;

        assert_eq!(
            replayed.status(),
            AgentStatus::Finished,
            "a folded recording must replay, not drift: {:?}",
            replayed.error()
        );
        assert_eq!(replayed.answer(), recorded.answer());
        assert_eq!(
            replayed.conversation().render(),
            recorded.conversation().render(),
            "replay must reproduce the folded transcript turn for turn"
        );
        assert_eq!(
            executions.load(Ordering::SeqCst),
            3,
            "replay must not re-run the tools"
        );
    }

    #[tokio::test]
    async fn a_fast_agent_finishes_without_waiting_for_a_slow_one() {
        use std::time::Duration;
        use tokio::sync::Notify;
        use xihe_llm::{LlmError, Provider};

        // The slow agent's call blocks until the fast agent is observed
        // Finished. Under a round barrier the fast agent cannot be applied
        // (and so cannot finish) until the slow call returns, which never
        // happens: deadlock. Independent progression lets the fast agent
        // finish on its own and open the slow agent's gate.
        struct GateProvider {
            gate: Arc<Notify>,
        }

        #[async_trait::async_trait]
        impl Provider for GateProvider {
            fn id(&self) -> &str {
                "gate"
            }
            fn default_model(&self) -> &str {
                "gate"
            }
            async fn generate(
                &self,
                request: &GenerationRequest,
            ) -> Result<GenerationResponse, LlmError> {
                let name = request
                    .correlation
                    .as_ref()
                    .map(|c| c.agent.as_str())
                    .unwrap_or("");
                if name == "slow" {
                    self.gate.notified().await;
                }
                Ok(GenerationResponse::text(format!("{name} done")))
            }
        }

        let gate = Arc::new(Notify::new());
        let mut world = World::new();
        world.insert_resource(ToolRegistry::new());
        world.insert_resource(LlmService::from_provider(GateProvider { gate: gate.clone() }));
        let fast = world.spawn_with((ReactAgent::new("fast", "be brief").ask("hi"),));
        world.spawn_with((ReactAgent::new("slow", "be brief").ask("hi"),));
        let ctx = SystemContext::new(WorldHandle::new(world));

        // Open the slow agent's gate the moment the fast agent finishes.
        let handle = ctx.world.clone();
        let watcher_gate = gate.clone();
        let watcher = tokio::spawn(async move {
            loop {
                {
                    let world = handle.read().await;
                    if let Some(agent) = world.get::<ReactAgent>(fast)
                        && agent.status() == AgentStatus::Finished
                    {
                        watcher_gate.notify_one();
                        return;
                    }
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        });

        let ran = tokio::time::timeout(Duration::from_secs(3), React::new().run(ctx.clone())).await;
        watcher.abort();
        let ran = ran.expect("the fast agent must finish without waiting for the slow one");
        ran.expect("the loop runs");

        let world = ctx.world.read().await;
        for (_entity, agent) in world.query::<(Entity, &ReactAgent)>() {
            assert_eq!(agent.status(), AgentStatus::Finished, "{}", agent.name());
        }
    }

    #[tokio::test]
    async fn a_turn_ceiling_stops_the_loop() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let ctx = world_with(
            MockProvider::with_script([
                Reply::tool("echo", json!({"text": "a"})),
                Reply::tool("echo", json!({"text": "b"})),
                Reply::text("done"),
            ]),
            registry,
            ReactAgent::new("loop", "keep going")
                .with_guardrails(Guardrails::new().with_turns(1))
                .ask("go"),
        );
        React::new().run(ctx.clone()).await.expect("runs");
        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Failed);
        assert!(
            agent.error().unwrap_or("").contains("max_turns"),
            "{:?}",
            agent.error()
        );
    }

    #[tokio::test]
    async fn json_answers_are_repaired_then_accepted() {
        let ctx = world_with(
            MockProvider::with_script([
                Reply::text("sure, here you go"),
                Reply::text(r#"{"answer": 7}"#),
            ]),
            ToolRegistry::new(),
            ReactAgent::new("counter", "count")
                .with_format(ResponseFormat::json(json!({
                    "type": "object",
                    "required": ["answer"]
                })))
                .ask("how many?"),
        );
        React::new().run(ctx.clone()).await.expect("runs");
        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Finished);
        assert_eq!(agent.structured(), Some(&json!({"answer": 7})));
        assert_eq!(agent.repairs_left(), ResponseFormat::DEFAULT_REPAIRS - 1);
    }

    #[tokio::test]
    async fn a_tool_without_an_approver_parks_the_agent() {
        let mut registry = ToolRegistry::new();
        registry.register(
            FnTool::new("deploy", "ship it", |_ctx, _call| ToolOutcome::text("shipped"))
                .requiring_approval(),
        );
        let ctx = world_with(
            MockProvider::with_script([Reply::tool("deploy", json!({"target": "prod"}))]),
            registry,
            ReactAgent::new("ops", "deploy carefully").ask("ship"),
        );
        React::new().run(ctx.clone()).await.expect("runs");
        let mut agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::AwaitingApproval);
        assert_eq!(agent.awaiting().map(|request| request.tool()), Some("deploy"));
        assert!(
            agent.approve(),
            "a parked agent accepts a decision even off the world"
        );
    }

    #[tokio::test]
    async fn approving_a_parked_call_resumes_it() {
        let mut registry = ToolRegistry::new();
        registry.register(
            FnTool::new("deploy", "ship it", |_ctx, call| {
                let target = call.string_argument("target").unwrap_or("?");
                ToolOutcome::text(format!("shipped to {target}"))
            })
            .with_required_string("target", "where to ship")
            .requiring_approval(),
        );
        let ctx = world_with(
            MockProvider::with_script([
                Reply::tool("deploy", json!({"target": "staging"})),
                Reply::text("staging is live"),
            ]),
            registry,
            ReactAgent::new("ops", "deploy carefully").ask("ship"),
        );
        React::new().run(ctx.clone()).await.expect("parks");
        {
            let mut world = ctx.world.write().await;
            let entity = world.single::<ReactAgent>().expect("one");
            assert!(world.get_mut::<ReactAgent>(entity).unwrap().approve());
        }
        React::new()
            .with_approver(AlwaysApprove)
            .run(ctx.clone())
            .await
            .expect("resumes");
        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Finished, "{:?}", agent.error());
        assert_eq!(agent.answer(), Some("staging is live"));
        assert!(agent.conversation().render().contains("shipped to staging"));
    }

    #[tokio::test]
    async fn a_rejection_is_shown_to_the_model() {
        let mut registry = ToolRegistry::new();
        registry.register(
            FnTool::new("deploy", "ship it", |_ctx, _call| ToolOutcome::text("nope"))
                .requiring_approval(),
        );
        let ctx = world_with(
            MockProvider::with_script([
                Reply::tool("deploy", json!({})),
                Reply::text("I will not deploy"),
            ]),
            registry,
            ReactAgent::new("ops", "ask first").ask("ship"),
        );
        React::new()
            .with_approver(AlwaysReject::new("frozen for the weekend"))
            .run(ctx.clone())
            .await
            .expect("runs");
        let agent = only_agent(&ctx).await;
        assert_eq!(agent.answer(), Some("I will not deploy"));
        assert!(agent.conversation().render().contains("frozen for the weekend"));
        assert_eq!(agent.spend().denied, 1);
    }

    #[tokio::test]
    async fn an_allowlist_hides_tools_the_model_must_not_see() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry.register(FnTool::new("deploy", "ship it", |_ctx, _call| {
            ToolOutcome::text("shipped")
        }));
        let provider = MockProvider::text("nothing to call");
        let ctx = world_with(
            provider.clone(),
            registry,
            ReactAgent::new("scribe", "echo only")
                .with_guardrails(Guardrails::new().allow_tools(["echo"]))
                .ask("hi"),
        );
        React::new().run(ctx.clone()).await.expect("runs");
        let names = provider
            .last_request()
            .expect("a request")
            .tool_names()
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert_eq!(names, ["echo"]);
    }

    #[tokio::test]
    async fn no_service_fails_the_agent_instead_of_the_schedule() {
        let mut world = World::new();
        world.spawn_with((ReactAgent::new("orphan", "hello").ask("hi"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the system stays up");
        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Failed);
        assert!(agent.error().unwrap_or("").contains("LlmService"));
    }

    #[test]
    fn asking_again_resets_a_finished_task() {
        let mut agent = ReactAgent::new("scribe", "be brief").ask("first");
        agent.finish("done".to_string());
        agent.spend.turns = 4;
        agent.push_question("second");
        assert_eq!(agent.status(), AgentStatus::Working);
        assert_eq!(agent.turns(), 0);
        assert!(agent.answer().is_none());
        assert!(agent.awaits_model());
    }

    #[tokio::test]
    async fn folding_compresses_a_long_conversation_before_the_model_call() {
        let mut agent = ReactAgent::new("scribe", "be brief").ask("hi");
        for index in 0..20 {
            agent.conversation_mut().push_user(format!("turn {index}"));
        }
        let provider = MockProvider::echo("done");
        let ctx = world_with(provider.clone(), ToolRegistry::new(), agent);
        React::new()
            .with_fold(10, 2)
            .run(ctx.clone())
            .await
            .expect("runs");
        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Finished);
        assert!(agent.conversation().fold_count() > 0, "the fold ran");
        assert!(
            agent.conversation().turns().len() <= 10,
            "the window was compressed"
        );
        assert!(agent.conversation().summary().is_some());
        assert!(
            provider.call_count() >= 2,
            "one call for the summary, one for the answer"
        );
    }

    #[tokio::test]
    async fn folding_is_disabled_by_default() {
        let mut agent = ReactAgent::new("scribe", "be brief").ask("hi");
        for index in 0..20 {
            agent.conversation_mut().push_user(format!("turn {index}"));
        }
        let ctx = world_with(MockProvider::echo("done"), ToolRegistry::new(), agent);
        React::new().run(ctx.clone()).await.expect("runs");
        let agent = only_agent(&ctx).await;
        assert_eq!(agent.conversation().fold_count(), 0);
    }

    #[tokio::test]
    async fn streaming_forwards_chunks_and_still_finishes() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let provider = MockProvider::echo("hello world");
        let ctx = world_with(provider, ToolRegistry::new(), ReactAgent::new("scribe", "hi").ask("go"));
        React::new()
            .with_stream_sink(tx)
            .run(ctx.clone())
            .await
            .expect("runs");
        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Finished);
        assert_eq!(agent.answer(), Some("hello world"));
        let mut chunks = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            chunks.push(chunk);
        }
        assert!(!chunks.is_empty(), "the sink received chunks");
        let text: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(text, "hello world");
    }

    #[tokio::test]
    async fn readonly_tool_calls_run_concurrently() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct SlowRead {
            concurrent: Arc<AtomicUsize>,
            peak: Arc<AtomicUsize>,
        }

        #[async_trait::async_trait]
        impl Tool for SlowRead {
            fn descriptor(&self) -> ToolDescriptor {
                ToolDescriptor::new("slow", "a slow read")
                    .with_required_string("id", "identifier")
            }
            async fn run(&self, _ctx: &ToolContext, call: &ToolCall) -> ToolOutcome {
                let now = self.concurrent.fetch_add(1, Ordering::SeqCst) + 1;
                self.peak.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                self.concurrent.fetch_sub(1, Ordering::SeqCst);
                ToolOutcome::text(call.string_argument("id").unwrap_or("?"))
            }
            fn readonly(&self) -> bool {
                true
            }
        }

        let concurrent = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(SlowRead {
            concurrent: concurrent.clone(),
            peak: peak.clone(),
        });
        let provider = MockProvider::with_script([
            Reply::tools(vec![
                ToolCall::new("1", "slow", json!({"id": "a"})),
                ToolCall::new("2", "slow", json!({"id": "b"})),
            ]),
            Reply::text("done"),
        ]);
        let ctx = world_with(provider, registry, ReactAgent::new("scribe", "hi").ask("go"));
        React::new().run(ctx.clone()).await.expect("runs");
        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Finished);
        assert!(
            peak.load(Ordering::SeqCst) >= 2,
            "both calls were in flight at once"
        );
    }

    #[tokio::test]
    async fn a_stop_mid_generation_cancels_the_call_promptly() {
        use async_trait::async_trait;
        use tokio::sync::Notify;
        use xihe_llm::{LlmError, Provider};
        use xihe_schedule::StopToken;

        struct SlowProvider {
            started: Arc<Notify>,
        }

        #[async_trait]
        impl Provider for SlowProvider {
            fn id(&self) -> &str {
                "slow"
            }
            fn default_model(&self) -> &str {
                "slow"
            }
            async fn generate(
                &self,
                _request: &GenerationRequest,
            ) -> Result<GenerationResponse, LlmError> {
                self.started.notify_waiters();
                tokio::time::sleep(Duration::from_secs(30)).await;
                Ok(GenerationResponse::default())
            }
        }

        let started = Arc::new(Notify::new());
        let mut world = World::new();
        world.insert_resource(ToolRegistry::new());
        world.insert_resource(LlmService::from_provider(SlowProvider {
            started: started.clone(),
        }));
        world.spawn_with((ReactAgent::new("slow", "be brief").ask("hi"),));
        let token = StopToken::new();
        let ctx = SystemContext::new(WorldHandle::new(world)).with_stop_token(token.clone());

        let run_ctx = ctx.clone();
        let handle = tokio::spawn(async move { React::new().run(run_ctx).await });

        // Register interest before the task can run, then wait until the model
        // call is genuinely in flight so the stop lands mid-call, not before it.
        let notified = started.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        notified.await;

        token.stop("user cancelled");
        let joined = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("the run returns well before the 30s call would finish")
            .expect("the task joins");
        joined.expect("the system stays up");

        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Failed);
        assert!(
            agent.error().unwrap_or("").contains("stopped"),
            "{:?}",
            agent.error()
        );
    }

    #[tokio::test]
    async fn a_panicking_tool_does_not_abort_the_agent() {
        struct Boom;

        #[async_trait::async_trait]
        impl Tool for Boom {
            fn descriptor(&self) -> ToolDescriptor {
                ToolDescriptor::new("boom", "explodes")
            }
            async fn run(&self, _ctx: &ToolContext, _call: &ToolCall) -> ToolOutcome {
                panic!("tool exploded")
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(Boom);
        let provider = MockProvider::with_script([
            Reply::tool("boom", json!({})),
            Reply::text("recovered"),
        ]);
        let ctx = world_with(provider, registry, ReactAgent::new("scribe", "be brief").ask("go"));
        React::new()
            .run(ctx.clone())
            .await
            .expect("the loop survives a panicking tool");
        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Finished);
        assert_eq!(agent.answer(), Some("recovered"));
        assert!(
            agent.conversation().render().contains("panicked"),
            "the model saw the failure as a tool result"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "manual scaling measurement; run with --ignored --nocapture"]
    async fn measure_world_lock_scaling() {
        use xihe_llm::{LlmError, Provider};

        struct LatencyProvider {
            latency: Duration,
        }

        #[async_trait::async_trait]
        impl Provider for LatencyProvider {
            fn id(&self) -> &str {
                "latency"
            }
            fn default_model(&self) -> &str {
                "latency"
            }
            async fn generate(
                &self,
                _request: &GenerationRequest,
            ) -> Result<GenerationResponse, LlmError> {
                tokio::time::sleep(self.latency).await;
                Ok(GenerationResponse::text("done"))
            }
        }

        let latency = Duration::from_millis(50);
        println!("model latency per agent: {latency:?} (lock is released while it sleeps)");
        for n in [1usize, 8, 32, 64] {
            let mut world = World::new();
            world.insert_resource(ToolRegistry::new());
            world.insert_resource(LlmService::from_provider(LatencyProvider { latency }));
            for index in 0..n {
                world.spawn_with((ReactAgent::new(format!("agent-{index}"), "be brief").ask("hi"),));
            }
            let ctx = SystemContext::new(WorldHandle::new(world));
            let started = Instant::now();
            React::new().run(ctx.clone()).await.expect("the loop runs");
            let wall = started.elapsed();
            let finished = {
                let guard = ctx.world.read().await;
                guard
                    .query::<&ReactAgent>()
                    .filter(|agent| agent.status() == AgentStatus::Finished)
                    .count()
            };
            assert_eq!(finished, n, "every agent should finish for a valid measurement");
            let serial = latency.as_secs_f64() * n as f64;
            println!(
                "agents={n:3}  wall={wall:>10?}  per-agent={:>10?}  speedup-vs-serial={:5.1}x",
                wall / n as u32,
                serial / wall.as_secs_f64()
            );
        }

        // Zero latency strips away the model call so the lock + scheduling
        // overhead is the only cost left: the theoretical throughput ceiling.
        let latency = Duration::ZERO;
        println!("\nzero model latency: exposes lock + scheduling overhead alone");
        for n in [1usize, 8, 32, 64, 256] {
            let mut world = World::new();
            world.insert_resource(ToolRegistry::new());
            world.insert_resource(LlmService::from_provider(LatencyProvider { latency }));
            for index in 0..n {
                world.spawn_with((ReactAgent::new(format!("agent-{index}"), "be brief").ask("hi"),));
            }
            let ctx = SystemContext::new(WorldHandle::new(world));
            let started = Instant::now();
            React::new().run(ctx.clone()).await.expect("the loop runs");
            let wall = started.elapsed();
            println!(
                "agents={n:3}  wall={wall:>10?}  per-agent={:>10?}",
                wall / n as u32
            );
        }
    }
}
