//! ReAct 循环：推理、行动、观察——直到一个回答或一个限额。
//!
//! 一个智能体是携带 [`ReactAgent`] 组件的实体。该组件拥有对话、
//! 计数器与结果；[`React`] 是一个 [`crate::ecs_schedule::System`]，读取这些
//! 组件、调用提供方、运行工具并把结果写回。组件中没有任何东西
//! 与外部世界对话，这正是智能体可回放的原因：把一个
//! [`crate::agent_llm::MockProvider`] 指向它，
//! 同样的转录就会出来。
//!
//! # 锁纪律
//!
//! [`React`] 绝不在对提供方或工具的 await 期间持有世界锁。
//! 一个步骤是 读锁 → 快照 → 释放 → 对话 → 写锁 → 应用，
//! 应用时检查它正写入的智能体就是它当初规划的那个（相同轮计数、
//! 相同状态）。若期间有别的任何东西移动了该智能体，步骤被丢弃，
//! 而非应用到过期状态。
//!
//! # 运行一个
//!
//! ```
//! use serde_json::json;
//! use xihe::agent::{LlmService, React, ReactAgent, ToolRegistry, EchoTool};
//! use xihe::ecs_core::World;
//! use xihe::agent_llm::{MockProvider, Reply};
//! use xihe::ecs_schedule::{System, SystemContext, WorldHandle};
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

use crate::access;
use crate::ecs_core::Entity;
use crate::agent_llm::{
    ChatClient, ChatMessage, GenerationRequest, GenerationResponse, StreamChunk, ToolCall,
    ToolDescriptor,
};
use crate::agent_memory::{ConversationWindow, HeuristicSummariser, Speaker, Summariser, Turn};
use crate::agent_observer::{EventKind, Observer};
use crate::ecs_schedule::{Access, SysResult, System, SystemContext, Work};
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use serde_json::Value;

use crate::agent::error::AgentError;
use crate::agent::guardrail::{Guardrails, Spend};
use crate::agent::hitl::{ApprovalRequest, Approver, Decision};
use crate::agent::llm::LlmService;
use crate::agent::memory::MemoryService;
use crate::agent::output::ResponseFormat;
use crate::agent::replay::{Journal, approval_key};
use crate::agent::tool::{ToolContext, ToolOutcome, ToolRegistry};

/// 智能体在其任务中所处的位置。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AgentStatus {
    /// 尚未向它请求任何事。
    #[default]
    Idle,
    /// 循环欠该智能体一次模型调用、一次工具运行，或两者。
    Working,
    /// 一个工具正在等待人；见 [`ReactAgent::awaiting`]。
    AwaitingApproval,
    /// 任务以一个回答结束。
    Finished,
    /// 任务没有回答就结束。
    Failed,
}

impl AgentStatus {
    /// 循环是否会不再管这个智能体。
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Failed)
    }

    /// 循环是否仍可对这个智能体行动。
    pub fn is_open(self) -> bool {
        !self.is_terminal()
    }

    /// 用于日志与指标的稳定小写名字。
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

/// 一个 ReAct 智能体携带的一切。
#[derive(Clone, Debug)]
pub struct ReactAgent {
    name: String,
    instructions: String,
    model: Option<String>,
    service: Option<LlmService>,
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
    /// 一个空闲智能体，以 `instructions` 作为其常驻系统提示词。
    pub fn new(name: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instructions: instructions.into(),
            model: None,
            service: None,
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

    /// 请求一个不同于服务默认的模型。
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// 自带一个提供方客户端，覆盖世界默认的 [`LlmService`]。
    ///
    /// 多提供方基础架构：每个智能体可绑不同厂商/endpoint/重试策略，
    /// 世界默认只是不自带客户端者的回退。模型名仍可由 [`Self::with_model`]
    /// 在此提供方内进一步覆盖。
    #[must_use]
    pub fn with_service(mut self, service: LlmService) -> Self {
        self.service = Some(service);
        self
    }

    /// 替换上限。
    #[must_use]
    pub fn with_guardrails(mut self, guardrails: Guardrails) -> Self {
        self.guardrails = guardrails;
        self
    }

    /// 要求最终回答匹配 `format`。
    #[must_use]
    pub fn with_format(mut self, format: ResponseFormat) -> Self {
        self.repairs_left = format.repairs();
        self.format = format;
        self
    }

    /// 替换对话窗口，例如更改其 token 预算。
    #[must_use]
    pub fn with_window(mut self, conversation: ConversationWindow) -> Self {
        self.conversation = conversation;
        self
    }

    /// 问智能体某事，若上一个任务已结束则开始一个新任务。
    #[must_use]
    pub fn ask(mut self, question: impl Into<String>) -> Self {
        self.begin();
        self.conversation.push_user(question);
        self.status = AgentStatus::Working;
        self
    }

    /// 排入一个问题，而不推进构建器链。
    pub fn push_question(&mut self, question: impl Into<String>) {
        self.begin();
        self.conversation.push_user(question);
        self.status = AgentStatus::Working;
    }

    /// 追加模型在回答前应看到的内容。
    pub fn observe(&mut self, content: impl Into<String>) {
        self.conversation.push_user(content);
    }

    /// 清零计数器并启动时钟，除非一个任务仍未结。
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

    /// 智能体的名字。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 循环所处的位置。
    pub fn status(&self) -> AgentStatus {
        self.status
    }

    /// 最终回答，一旦任务完成。
    pub fn answer(&self) -> Option<&str> {
        self.answer.as_deref()
    }

    /// 解析为 JSON 的回答，当格式要求一个时。
    pub fn structured(&self) -> Option<&Value> {
        self.parsed.as_ref()
    }

    /// 任务为何失败。
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// 迄今花费的模型调用。
    pub fn turns(&self) -> usize {
        self.spend.turns
    }

    /// 智能体已花费什么。
    pub fn spend(&self) -> Spend {
        self.spend
    }

    /// 生效的上限。
    pub fn guardrails(&self) -> &Guardrails {
        &self.guardrails
    }

    /// 生效的回答格式。
    pub fn format(&self) -> &ResponseFormat {
        &self.format
    }

    /// 智能体据以工作的转录。
    pub fn conversation(&self) -> &ConversationWindow {
        &self.conversation
    }

    /// 可变的转录，供直接播种或裁剪它的调用者。
    pub fn conversation_mut(&mut self) -> &mut ConversationWindow {
        &mut self.conversation
    }

    /// 循环挂起时，正在等待人的请求。
    pub fn awaiting(&self) -> Option<&ApprovalRequest> {
        self.awaiting.as_ref()
    }

    /// 模型做出、正在等待人的那次调用。
    pub fn pending_call(&self) -> Option<&ToolCall> {
        match self.awaiting.as_ref() {
            Some(request) => Some(&request.call),
            None => None,
        }
    }

    /// 当前任务花费的墙钟时间。
    pub fn elapsed(&self) -> Duration {
        self.started
            .map_or(Duration::ZERO, |started| started.elapsed())
    }

    /// 返回时失败的工具调用。
    pub fn tool_errors(&self) -> usize {
        self.tool_errors
    }

    /// 畸形回答剩余的修复机会。
    pub fn repairs_left(&self) -> usize {
        self.repairs_left
    }

    /// 用于日志的一行摘要。
    pub fn summary(&self) -> String {
        format!(
            "{}: {} turns={}, tools={}",
            self.name, self.status, self.spend.turns, self.spend.tool_calls
        )
    }

    /// 实际发送的系统提示词：指令加上回答格式。
    pub fn system_text(&self) -> String {
        match self.format.instruction() {
            Some(extra) if !self.instructions.is_empty() => {
                format!("{}\n{extra}", self.instructions)
            }
            Some(extra) => extra,
            None => self.instructions.clone(),
        }
    }

    /// 是否欠一次模型调用：说了某事而模型尚未回答。
    pub fn awaits_model(&self) -> bool {
        if self.status != AgentStatus::Working {
            return false;
        }
        self.conversation
            .turns()
            .last()
            .is_some_and(|turn| matches!(turn.speaker, Speaker::User | Speaker::Tool))
    }

    /// 循环是否挂起在一个人类此后已回答的调用上。
    pub fn has_decision(&self) -> bool {
        self.status == AgentStatus::AwaitingApproval && self.decision.is_some()
    }

    /// 批准挂起的调用；无挂起时为 `false`。
    pub fn approve(&mut self) -> bool {
        self.decide(Decision::Approved)
    }

    /// 以 `reason` 拒绝挂起的调用；无挂起时为 `false`。
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

    /// 给定提供的工具，下一次模型调用的请求。
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

    /// 计费一次模型调用并追加其回答，把调用记录为转录所需的文本。
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

    /// 追加一个工具结果，无论工具运行了还是拒绝了。
    pub fn record_tool_result(&mut self, call: &ToolCall, outcome: &ToolOutcome) {
        if !outcome.ok {
            self.tool_errors += 1;
        }
        self.spend.record_tool_call();
        let text = clamp_tool_output(
            &outcome.transcript_text(),
            self.guardrails.max_tool_output_chars,
        );
        self.conversation.push(Turn::tool(call.id.clone(), text));
    }

    /// 接受一个模型回答为最终答案，当格式要求时解析它。
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

    /// 接受一个已被解析的模型回答。
    fn finish_parsed(&mut self, answer: String, value: Value) {
        self.status = AgentStatus::Finished;
        self.answer = Some(answer);
        self.parsed = Some(value);
        self.awaiting = None;
        self.decision = None;
    }

    /// 以调用者可读的原因停止任务。
    pub fn fail(&mut self, error: &AgentError) {
        self.status = AgentStatus::Failed;
        self.error = Some(error.to_string());
        self.awaiting = None;
        self.decision = None;
    }

    /// 以一条不是 [`AgentError`] 的消息停止任务。
    pub fn fail_with(&mut self, message: impl Into<String>) {
        self.status = AgentStatus::Failed;
        self.error = Some(message.into());
        self.awaiting = None;
        self.decision = None;
    }

    /// 把循环挂起在 `request` 上，记住该调用以便恢复。
    fn park(&mut self, request: ApprovalRequest) {
        self.status = AgentStatus::AwaitingApproval;
        self.awaiting = Some(request);
        self.decision = None;
    }
}

/// 运行循环的系统。
///
/// 一次激活把每个忙碌的智能体推进至多 [`React::steps`] 个
/// 推理-行动轮。智能体独立推进：每次调用在世界锁之外运行，
/// 完成的那一刻即被写回，因此快的智能体绝不等慢的。
/// 同时在飞的调用被逐个应用，日志以稳定的每智能体身份为每个键，
/// 因此最终状态是确定的，尽管完成顺序不是。
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
    /// 每次激活把每个忙碌智能体推进至多 8 轮。
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

    /// 命名系统，一旦多个循环共享一个调度器这就重要了。
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// 每次激活的轮数；`0` 被视为一。
    #[must_use]
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps.max(1);
        self
    }

    /// 回答审批请求而非挂起。
    #[must_use]
    pub fn with_approver<A: Approver + 'static>(mut self, approver: A) -> Self {
        self.approver = Some(Arc::new(approver));
        self
    }

    /// 共享一个审批器句柄，例如多个循环前的同一道闸。
    #[must_use]
    pub fn with_shared_approver(mut self, approver: Arc<dyn Approver>) -> Self {
        self.approver = Some(approver);
        self
    }

    /// 折叠长于 `threshold` 轮的对话，逐字保留最后 `keep_last` 轮。
    /// `0` 禁用折叠。
    ///
    /// 当存在 [`LlmService`] 时，折叠请模型做摘要；
    /// 否则回退到句子排名。
    #[must_use]
    pub fn with_fold(mut self, threshold: usize, keep_last: usize) -> Self {
        self.fold_threshold = threshold;
        self.keep_last = keep_last.max(1);
        self
    }

    /// 每个流式块到达时即转发给 `sink`。
    ///
    /// 设置后，循环使用 [`ChatClient::stream`] 而非
    /// [`ChatClient::generate`]，为自身决策累积完整响应，
    /// 同时调用者增量地观察 token。
    #[must_use]
    pub fn with_stream_sink(
        mut self,
        sink: tokio::sync::mpsc::UnboundedSender<StreamChunk>,
    ) -> Self {
        self.stream_sink = Some(sink);
        self
    }

    /// 本系统每次激活愿意运行的轮数。
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
        // 一个在飞步骤的池。每个结果在完成的那一刻被应用，
        // 且只重新规划那个智能体，因此智能体按自己的节奏推进，
        // 而非等待一轮中最慢的调用。
        let mut pool: FuturesUnordered<Pin<Box<dyn Future<Output = Outcome> + Send>>> =
            FuturesUnordered::new();
        let mut rounds: HashMap<Entity, usize> = HashMap::new();

        if ctx.is_stopping() {
            stop_agents(&ctx).await;
            update_work(&ctx, &self.name).await;
            return Ok(());
        }
        reject_duplicate_names(&ctx).await;
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
            // 让在飞调用与关停竞速：生成中途发出的停止信号
            // 会丢弃提供方 future，而非等它结束。
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
            if let Some(step) = plan_one(
                &ctx,
                entity,
                self.approver.clone(),
                self.stream_sink.clone(),
            )
            .await
            {
                pool.push(Box::pin(run_step(step)));
            }
        }
        update_work(&ctx, &self.name).await;
        Ok(())
    }
}

/// 在世界锁之外，把一个已规划步骤运行到其结果。
async fn run_step(step: Step) -> Outcome {
    match step {
        Step::Generate(generation) => (*generation).execute().await,
        Step::Resume(resume) => (*resume).execute().await,
        Step::Fail { entity, error } => Outcome::Failed { entity, error },
    }
}

/// 欠世界一次模型调用的步骤。
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

/// 继续一个人类已回答之挂起调用的步骤。
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

/// 一个步骤产出了什么，已准备好写回。
enum Outcome {
    /// 一个最终文本回答。
    Answered {
        entity: Entity,
        turn: usize,
        response: Box<GenerationResponse>,
        parsed: Option<Value>,
    },
    /// 一个看起来可用、却被格式拒绝的回答；再问一次。
    Repaired {
        entity: Entity,
        turn: usize,
        response: Box<GenerationResponse>,
        repair: String,
    },
    /// 模型调用了工具；其结果已准备好追加。
    Acted {
        entity: Entity,
        turn: usize,
        response: Box<GenerationResponse>,
        results: Vec<(ToolCall, ToolOutcome)>,
        refused: usize,
        parked: Option<ApprovalRequest>,
    },
    /// 一个挂起的调用已被决定；其结果已准备好追加。
    Observed {
        entity: Entity,
        turn: usize,
        results: Vec<(ToolCall, ToolOutcome)>,
        refused: usize,
        parked: Option<ApprovalRequest>,
    },
    /// 任务已结束且未产出回答。
    Failed { entity: Entity, error: AgentError },
}

/// 为每个欠一个步骤的智能体收集步骤，返回前释放锁，
/// 使调用能并发运行。
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

/// 为单个智能体收集下一步，使它无需等待其他智能体即可推进。
/// 一旦该智能体不再欠调用，返回 `None`。
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

/// 构建一个智能体所欠的步骤，当它已落定或无决定地挂起时为 `None`。
/// 由批量播种与每智能体续行共享，使两者规划一致。
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
    let Some(service) = agent.service.clone().or_else(|| service.clone()) else {
        return Some(Step::Fail {
            entity,
            error: AgentError::config(
                "no LlmService for the agent to ask: neither its own nor a world default",
            ),
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
    /// 一次模型调用，加上它请求的工具轮。
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
        if let Some(journal) = &self.journal
            && let Err(error) = journal.record(&self.request, response.as_ref())
        {
            return Outcome::Failed {
                entity: self.entity,
                error,
            };
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
            && let Err(error) = journal.record_tools(&self.request, results.clone(), refused)
        {
            return Outcome::Failed {
                entity: self.entity,
                error,
            };
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

    /// 把一个文本回答变成已完成的任务、一个修复轮，或一次失败。
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
    /// 运行（或拒绝）挂起的调用，然后继续其后的任何事。
    ///
    /// 当录制日志挂起了此审批时，整轮——已在手的结果加上决定
    /// 所产出的——被记入原请求之下，因此回放反馈它，
    /// 无需再次询问人或重新运行已批准的工具。
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
                None => {
                    if let Err(error) = journal.record_tools(request, done, total_refused) {
                        return Outcome::Failed {
                            entity: self.entity,
                            error,
                        };
                    }
                }
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

/// 按模型请求的顺序运行调用。
///
/// 当每个调用都只读、已注册、已批准且在预算内时，它们并发运行。
/// 否则顺序运行：请求两个调用的模型通常意在让第二个看到第一个，
/// 而一个写入的工具若循环中途停止则无法回滚。
///
/// 返回结果、被直接拒绝的调用数，以及当某调用需要人而
/// 没有 [`Approver`] 能提供时、用以挂起的请求。
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
                    .unwrap_or_else(|error| ToolOutcome::refused(error.to_string()));
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
                ToolOutcome::refused(format!(
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
                ToolOutcome::refused(format!("unknown tool `{}`", call.name)),
            ));
            continue;
        }
        if registry.requires_approval(&call.name) {
            let request = ApprovalRequest::new(
                entity,
                turn as u64,
                call,
                calls.iter().skip(index + 1).cloned().collect::<Vec<_>>(),
            );
            let Some(approver) = approver else {
                return (results, refused, Some(request));
            };
            let decision = approver.decide(&request).await;
            if !decision.is_approved() {
                refused += 1;
                results.push((
                    call.clone(),
                    ToolOutcome::refused(decision.transcript_text()),
                ));
                continue;
            }
        }
        let problems = registry.validate(call);
        if !problems.is_empty() {
            refused += 1;
            results.push((call.clone(), ToolOutcome::refused(problems.join("; "))));
            continue;
        }
        match registry.execute(tool_ctx, call).await {
            Ok(outcome) => {
                ran += 1;
                results.push((call.clone(), outcome));
            }
            Err(error) => {
                refused += 1;
                results.push((call.clone(), ToolOutcome::refused(error.to_string())));
            }
        }
    }
    (results, refused, None)
}

/// 把一个结果写回，返回是否有任何改变。
///
/// 先检查轮计数与状态：针对轮 3 规划的步骤，一旦智能体到了轮 4
/// 就毫无价值，应用它会在转录中重复工具体结果。
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
        let world_service = LlmService::from_world(world);
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
        let service = agent.service.clone().or(world_service);
        let provider = service
            .as_ref()
            .map(|service| service.provider_id())
            .unwrap_or_default();
        let default_model = service
            .as_ref()
            .map(|service| service.model())
            .unwrap_or_default();
        let old_status = agent.status;
        let mut events = Vec::new();
        let wrote = match outcome {
            Outcome::Answered {
                response, parsed, ..
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
                    if outcome.is_refused() {
                        events.push(EventKind::Refused {
                            name: call.name.clone(),
                            reason: outcome.text.clone(),
                        });
                    } else {
                        events.push(EventKind::Tool {
                            name: call.name.clone(),
                            ok: outcome.ok,
                        });
                    }
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
                    if outcome.is_refused() {
                        events.push(EventKind::Refused {
                            name: call.name.clone(),
                            reason: outcome.text.clone(),
                        });
                    } else {
                        events.push(EventKind::Tool {
                            name: call.name.clone(),
                            ok: outcome.ok,
                        });
                    }
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
                        model: agent.model.clone().unwrap_or(default_model),
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

/// 在锁之外，折叠每个已增长超过 `threshold` 的对话。
///
/// 窗口在读锁下被克隆、并发摘要，然后仅当智能体仍在同一轮时
/// 才在写锁下写回。
async fn fold_overlong(ctx: &SystemContext, threshold: usize, keep_last: usize) {
    let service = LlmService::read(ctx).await;
    let (journal, snapshots): (
        Option<Journal>,
        Vec<(
            Entity,
            usize,
            String,
            ConversationWindow,
            Option<LlmService>,
        )>,
    ) = ctx
        .with_world(|world| {
            let journal = world.resource::<Journal>().cloned();
            let snapshots = world
                .query::<(Entity, &ReactAgent)>()
                .filter(|(_, agent)| {
                    agent.status.is_open() && agent.conversation.turns().len() > threshold
                })
                .map(|(entity, agent)| {
                    (
                        entity,
                        agent.spend.turns,
                        agent.name().to_string(),
                        agent.conversation.clone(),
                        agent.service.clone().or_else(|| service.clone()),
                    )
                })
                .collect();
            (journal, snapshots)
        })
        .await;
    if snapshots.is_empty() {
        return;
    }
    // 每智能体一个摘要器：折叠的模型调用携带该智能体的关联标识，
    // 使录制的折叠回放进正确的窗口，即便多个智能体在同一次激活中
    // 折叠相同的转录。
    let folded = futures::future::join_all(snapshots.into_iter().map(
        |(entity, turn, name, mut window, service)| {
            let journal = journal.clone();
            async move {
                let report = crate::agent::memory::FoldReport::new();
                let summariser: Box<dyn Summariser> = match &service {
                    Some(service) => {
                        let mut chat = crate::agent::memory::ChatSummariser::new(service.shared())
                            .with_correlation(name, window.fold_count())
                            .with_report(report.clone());
                        if let Some(journal) = journal {
                            chat = chat.with_journal(journal);
                        }
                        Box::new(chat)
                    }
                    None => Box::new(HeuristicSummariser::new(6, 160)),
                };
                let result = window.fold(summariser.as_ref(), keep_last).await;
                (entity, turn, window, result, report.take(), service)
            }
        },
    ))
    .await;
    ctx.with_world_mut(|world| {
        let mut events = Vec::new();
        for (entity, turn, window, result, call, service) in folded {
            // 折叠的模型调用无论成败都记一条 Observer 事件：回退启发式后
            // 折叠仍成功，故据调用报告而非折叠结果计量。归因到该智能体
            // 实际使用的提供方，而非世界默认。
            if let Some(call) = call {
                let (provider, model) = service
                    .map(|service| (service.provider_id(), service.model()))
                    .unwrap_or_default();
                events.push(EventKind::Model {
                    provider,
                    model,
                    input_tokens: call.input_tokens,
                    output_tokens: call.output_tokens,
                    failed: call.failed,
                });
            }
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
        if let Some(observer) = world.resource_mut::<Observer>() {
            for event in events {
                observer.record(event);
            }
        }
    })
    .await;
}

async fn update_work(ctx: &SystemContext, label: &str) {
    let busy = ctx
        .with_world(|world| {
            // 挂起的智能体等待的是人，而非算力；只有人类已决定的
            // 挂起仍欠一次恢复。
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

/// 此结果是否必须匹配它当初规划所针对的轮。
fn outcome_turns(outcome: &Outcome) -> bool {
    !matches!(outcome, Outcome::Failed { .. })
}

/// 回答文本，当模型什么也没说时回退到所做的调用。
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

/// 限制单个工具结果的转录文本，标记被丢弃的尾部，使模型知道
/// 输出不完整。`0` 保留全部。完整结果仍在日志中传输；只有面向
/// 提示词的副本被裁剪，录制与回放时完全一致。
fn clamp_tool_output(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return text.to_string();
    }
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }
    let kept: String = text.chars().take(max_chars).collect();
    format!(
        "{kept}\n[output truncated: {} more chars omitted]",
        total - max_chars
    )
}

/// 运行工具时所带的上下文：谁请求、哪一轮、什么记忆。
fn tool_context(entity: Entity, agent: &ReactAgent, store: Option<&MemoryService>) -> ToolContext {
    let context = ToolContext::new(entity)
        .with_session(agent.name.clone())
        .with_turn(agent.spend.turns as u64);
    match store {
        Some(store) => context.with_store(store.shared()),
        None => context,
    }
}

/// 告知每个忙碌的智能体：运行时正在关停。
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

/// 失败重名的开放智能体。名字是关联标识（名字+轮）的一半，重名会让
/// 两个智能体共享日志键，静默复用彼此的模型响应与工具结果——记录时
/// 的冲突守卫只兜得住并发覆盖，兜不住顺序复用。名字所有权按所有
/// lingering 实体判定（含终态者）：一个跑完仍留在世界中的智能体依旧
/// 拥有其名字与日志键，复用它的新智能体会经 lookup_tools 静默接管旧
/// 轨迹。按实体顺序保留每个名字的首个拥有者，只失败开放的后来者
/// （终态者不被二次失败），使冲突在任何模型调用之前就响亮可见。
/// 合法复用须先 despawn 旧实体或改名。
async fn reject_duplicate_names(ctx: &SystemContext) {
    use std::collections::hash_map::Entry;
    ctx.with_world_mut(|world| {
        let mut agents: Vec<(Entity, String)> = world
            .query::<(Entity, &ReactAgent)>()
            .map(|(entity, agent)| (entity, agent.name().to_string()))
            .collect();
        agents.sort_by_key(|(entity, _)| *entity);
        let mut owners: HashMap<String, Entity> = HashMap::new();
        let mut events = Vec::new();
        for (entity, name) in agents {
            match owners.entry(name.clone()) {
                Entry::Vacant(slot) => {
                    slot.insert(entity);
                }
                Entry::Occupied(_) => {
                    if let Some(agent) = world.get_mut::<ReactAgent>(entity)
                        && agent.status.is_open()
                    {
                        agent.fail(&AgentError::config(format!(
                            "duplicate agent name `{name}`: agent names must be unique, \
                             so their correlation keys do not collide and share a journal"
                        )));
                        events.push(EventKind::Agent {
                            name: agent.name().to_string(),
                            status: agent.status.as_str().to_string(),
                        });
                    }
                }
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
    use crate::ecs_core::World;
    use crate::agent_llm::{MockProvider, Reply};
    use crate::ecs_schedule::{System, WorldHandle};
    use serde_json::json;

    use crate::agent::{
        AlwaysApprove, AlwaysReject, EchoTool, FnTool, LlmService, ResponseFormat, Tool,
        ToolRegistry,
    };

    fn world_with(
        provider: MockProvider,
        registry: ToolRegistry,
        agent: ReactAgent,
    ) -> SystemContext {
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
        use crate::agent_llm::{LlmError, Provider, Role};
        use std::sync::Mutex;

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
        use crate::agent_llm::{LlmError, Provider};
        use std::sync::Mutex;

        use crate::agent::replay::{Journal, ReplayProvider};

        // 一个每次调用回答都变的提供方。重新运行它不会复现第一次运行，
        // 因此一次相同的回放证明日志反馈了录制的响应，
        // 而非模型再次作答。
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
        use crate::agent_llm::{LlmError, Provider};
        use std::sync::Mutex;

        use crate::agent::replay::{Journal, ReplayProvider};

        // 每次调用给出不同回答。两个指令相同、问题相同的智能体
        // 发出逐字节相同的请求，因此只有稳定的每智能体键
        // 才能把它们录制的回答分开。
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

        use crate::agent::replay::{Journal, ReplayProvider};

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

        use crate::agent::replay::{Journal, ReplayProvider};

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

        // 持久化为字节，并仅从它们重建一个新日志：与录制日志
        // 不共享状态，正如第二个进程那样。
        let bytes = journal.to_bytes().expect("serialize the journal");
        drop(journal);
        let replay_journal = Journal::from_bytes(&bytes).expect("deserialize the journal");

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(replay_journal.clone());
        world.insert_resource(LlmService::from_provider(ReplayProvider::new(
            replay_journal,
        )));
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

        use crate::agent::replay::{Journal, ReplayProvider};

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

        // 持久化到一个真实文件，然后彻底丢弃录制日志。
        let path = std::env::temp_dir().join(format!("xihe-journal-{}.json", std::process::id()));
        journal.save(&path).expect("save to disk");
        drop(journal);

        // 从磁盘加载的新日志与录制者不共享任何东西。
        let replay_journal = Journal::load(&path).expect("load from disk");
        std::fs::remove_file(&path).ok();

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(replay_journal.clone());
        world.insert_resource(LlmService::from_provider(ReplayProvider::new(
            replay_journal,
        )));
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

        use crate::agent::replay::{Journal, ReplayProvider};

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
        React::new()
            .run(ctx.clone())
            .await
            .expect("the record run parks");
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
        world.insert_resource(LlmService::from_provider(ReplayProvider::new(
            replay_journal,
        )));
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
        use crate::agent_llm::{LlmError, Provider};
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::agent::replay::{Journal, ReplayProvider};

        // 一个对生成调用（提供工具的那些）以几个工具轮然后一个回答
        // 作答、对折叠调用（无工具）以固定摘要作答的提供方。
        // 折叠摘要不同于启发式回退会产出的任何内容，因此一次
        // 重新摘要而非反馈录制摘要的回放会发散。
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
                    return Ok(GenerationResponse::text(
                        "a folded summary of earlier turns",
                    ));
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
        assert_eq!(
            executions.load(Ordering::SeqCst),
            3,
            "three tool rounds ran"
        );

        let bytes = journal.to_bytes().expect("serialize the journal");
        drop(journal);
        let replay_journal = Journal::from_bytes(&bytes).expect("deserialize the journal");

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(replay_journal.clone());
        world.insert_resource(LlmService::from_provider(ReplayProvider::new(
            replay_journal,
        )));
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
    async fn a_failed_fold_is_visible_in_the_observer() {
        use crate::agent_llm::{LlmError, Provider};
        use std::sync::Mutex;

        // 折叠调用（无工具）以一个非重试错误失败，迫使 ChatSummariser
        // 回退启发式；生成调用（有工具）以工具轮然后回答作答，使智能体
        // 仍能完成。被容纳的折叠降级必须在 Observer 可见，而非仅 tracing。
        struct FailingFoldProvider {
            generations: Mutex<usize>,
        }

        #[async_trait::async_trait]
        impl Provider for FailingFoldProvider {
            fn id(&self) -> &str {
                "failing-fold"
            }
            fn default_model(&self) -> &str {
                "failing-fold"
            }
            async fn generate(
                &self,
                request: &GenerationRequest,
            ) -> Result<GenerationResponse, LlmError> {
                if request.tools.is_empty() {
                    return Err(LlmError::decode("the summariser endpoint returned garbage"));
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

        let mut registry = ToolRegistry::new();
        registry.register(
            FnTool::new("echo", "echo it", |_ctx, call| {
                ToolOutcome::text(call.string_argument("text").unwrap_or("").to_string())
            })
            .with_required_string("text", "the text to echo"),
        );

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(Observer::new());
        world.insert_resource(LlmService::from_provider(FailingFoldProvider {
            generations: Mutex::new(0),
        }));
        world.spawn_with((ReactAgent::new("scribe", "be brief").ask("go"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new()
            .with_fold(3, 2)
            .run(ctx.clone())
            .await
            .expect("the run");

        let agent = only_agent(&ctx).await;
        assert_eq!(
            agent.status(),
            AgentStatus::Finished,
            "the heuristic fallback keeps the agent working: {:?}",
            agent.error()
        );
        assert!(
            agent.conversation().fold_count() > 0,
            "the conversation must actually have folded"
        );

        let observer = {
            let world = ctx.world.read().await;
            world.resource::<Observer>().cloned().expect("an observer")
        };
        assert!(
            observer.snapshot().model_failures >= 1,
            "a failed fold model call must reach the Observer, not just tracing: {}",
            observer.render()
        );
        assert!(
            observer.events().iter().any(|event| matches!(
                &event.kind,
                EventKind::Model { provider, failed: true, .. } if provider == "failing-fold"
            )),
            "the degraded fold is recorded as a model failure: {}",
            observer.render()
        );
    }

    #[tokio::test]
    async fn a_successful_fold_is_counted_in_the_observer() {
        use crate::agent_llm::{LlmError, Provider, TokenUsage};
        use std::sync::Mutex;

        // 折叠调用（无工具）以带 token 计量的摘要作答；生成调用（有工具）
        // 以几个工具轮然后回答作答。成功的折叠也是一次模型调用，必须像
        // 生成轮一样计入 Observer 的 model_calls 与 token 总量，而非被漏计。
        struct AccountingFoldProvider {
            generations: Mutex<usize>,
        }

        #[async_trait::async_trait]
        impl Provider for AccountingFoldProvider {
            fn id(&self) -> &str {
                "accounting-fold"
            }
            fn default_model(&self) -> &str {
                "accounting-fold"
            }
            async fn generate(
                &self,
                request: &GenerationRequest,
            ) -> Result<GenerationResponse, LlmError> {
                if request.tools.is_empty() {
                    return Ok(
                        GenerationResponse::text("a folded summary of earlier turns")
                            .with_usage(TokenUsage::new(100, 20)),
                    );
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

        let mut registry = ToolRegistry::new();
        registry.register(
            FnTool::new("echo", "echo it", |_ctx, call| {
                ToolOutcome::text(call.string_argument("text").unwrap_or("").to_string())
            })
            .with_required_string("text", "the text to echo"),
        );

        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(Observer::new());
        world.insert_resource(LlmService::from_provider(AccountingFoldProvider {
            generations: Mutex::new(0),
        }));
        world.spawn_with((ReactAgent::new("scribe", "be brief").ask("go"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new()
            .with_fold(3, 2)
            .run(ctx.clone())
            .await
            .expect("the run");

        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Finished, "{:?}", agent.error());
        assert!(
            agent.conversation().fold_count() > 0,
            "the conversation must actually have folded"
        );

        let observer = {
            let world = ctx.world.read().await;
            world.resource::<Observer>().cloned().expect("an observer")
        };
        assert!(
            observer.events().iter().any(|event| matches!(
                &event.kind,
                EventKind::Model {
                    input_tokens: 100,
                    output_tokens: 20,
                    failed: false,
                    ..
                }
            )),
            "a successful fold must be counted with its token usage: {}",
            observer.render()
        );
        assert!(
            observer.snapshot().model_calls >= 5,
            "four generation turns plus at least one fold must all be counted: {}",
            observer.render()
        );
        assert_eq!(
            observer.snapshot().model_failures,
            0,
            "{}",
            observer.render()
        );
    }

    #[tokio::test]
    async fn duplicate_agent_names_are_rejected_before_they_share_a_journal() {
        use crate::agent::replay::Journal;

        // 名字是关联标识（名字+轮）的一半。两个同名智能体共享日志键，
        // 会静默复用彼此的模型响应与工具结果。预防性守卫必须在任何
        // 模型调用之前就失败后来者，而非靠记录时检测兜底。
        let mut world = World::new();
        world.insert_resource(ToolRegistry::new());
        world.insert_resource(Journal::new());
        world.insert_resource(LlmService::from_provider(MockProvider::text("done")));
        world.spawn_with((ReactAgent::new("worker", "be brief").ask("go"),));
        world.spawn_with((ReactAgent::new("worker", "be brief").ask("go"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the loop runs");

        let world = ctx.world.read().await;
        let agents: Vec<ReactAgent> = world
            .query::<(Entity, &ReactAgent)>()
            .map(|(_, agent)| agent.clone())
            .collect();
        let failed = agents
            .iter()
            .filter(|agent| agent.status() == AgentStatus::Failed)
            .count();
        let finished = agents
            .iter()
            .filter(|agent| agent.status() == AgentStatus::Finished)
            .count();
        assert_eq!(failed, 1, "the later duplicate must fail loudly");
        assert_eq!(finished, 1, "the first owner keeps the name and finishes");
        let duplicate = agents
            .iter()
            .find(|agent| agent.status() == AgentStatus::Failed)
            .unwrap();
        let error = duplicate.error().unwrap_or("");
        assert!(
            error.contains("duplicate"),
            "must name the problem: {error}"
        );
        assert!(error.contains("worker"), "must name the agent: {error}");
    }

    #[tokio::test]
    async fn a_name_owned_by_a_finished_agent_still_blocks_reuse() {
        use crate::agent::replay::Journal;

        // 一个已完成但仍留在世界中的智能体仍拥有其名字与日志键。
        // 复用该名字的新智能体会经 lookup_tools 静默复用旧轨迹，故
        // 名字所有权须按所有 lingering 实体判定，而非只看开放者。
        let mut world = World::new();
        world.insert_resource(ToolRegistry::new());
        world.insert_resource(Journal::new());
        world.insert_resource(LlmService::from_provider(MockProvider::text("done")));
        let first = world.spawn_with((ReactAgent::new("worker", "be brief").ask("go"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the first run");
        {
            let world = ctx.world.read().await;
            assert_eq!(
                world.get::<ReactAgent>(first).unwrap().status(),
                AgentStatus::Finished
            );
        }

        let second = {
            let mut world = ctx.world.write().await;
            world.spawn_with((ReactAgent::new("worker", "be brief").ask("go"),))
        };
        React::new().run(ctx.clone()).await.expect("the second run");

        let world = ctx.world.read().await;
        let duplicate = world.get::<ReactAgent>(second).unwrap();
        assert_eq!(
            duplicate.status(),
            AgentStatus::Failed,
            "reuse against a lingering finished agent must fail loudly"
        );
        assert!(
            duplicate.error().unwrap_or("").contains("duplicate"),
            "{:?}",
            duplicate.error()
        );
        assert_eq!(
            world.get::<ReactAgent>(first).unwrap().status(),
            AgentStatus::Finished,
            "the original owner is untouched"
        );
    }

    #[tokio::test]
    async fn a_huge_tool_result_is_capped_in_the_prompt() {
        use crate::agent_llm::{LlmError, Provider, Role};
        use std::sync::Mutex;

        // 捕获每个请求，使回答轮的提示词——把工具结果带回给模型的
        // 那一个——可被测量。
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
                let n = *calls;
                *calls += 1;
                Ok(if n == 0 {
                    GenerationResponse::default().with_tool_calls(vec![ToolCall::new(
                        "c1",
                        "dump",
                        json!({}),
                    )])
                } else {
                    GenerationResponse::text("done")
                })
            }
        }

        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = CapturingProvider {
            requests: requests.clone(),
            calls: Mutex::new(0),
        };
        let mut registry = ToolRegistry::new();
        registry.register(FnTool::new("dump", "return a lot", |_ctx, _call| {
            ToolOutcome::text("x".repeat(100_000))
        }));
        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(LlmService::from_provider(provider));
        world.spawn_with((ReactAgent::new("reader", "be brief").ask("read it"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the loop runs");
        assert_eq!(only_agent(&ctx).await.status(), AgentStatus::Finished);

        let captured = requests.lock().unwrap();
        assert_eq!(captured.len(), 2, "a tool round then an answer round");
        let answer_round = &captured[1];
        let tool_msg = answer_round
            .messages
            .iter()
            .find(|message| message.role == Role::Tool)
            .expect("the tool result is replayed to the model");
        assert!(
            tool_msg.content.chars().count() <= Guardrails::DEFAULT_TOOL_OUTPUT_CHARS + 64,
            "a single huge tool result must be capped, got {} chars",
            tool_msg.content.chars().count()
        );
        assert!(
            tool_msg.content.contains("truncated"),
            "the model must be told the output was cut: {}",
            tool_msg.content
        );
        assert!(
            answer_round.estimated_prompt_tokens()
                <= Guardrails::DEFAULT_TOOL_OUTPUT_CHARS / 4 + 512,
            "the answer-round prompt must stay bounded, got {} estimated tokens",
            answer_round.estimated_prompt_tokens()
        );
    }

    #[test]
    fn tool_output_clamp_marks_the_dropped_tail() {
        assert_eq!(clamp_tool_output("short", 10), "short");
        assert_eq!(
            clamp_tool_output("anything", 0),
            "anything",
            "zero keeps everything"
        );
        let capped = clamp_tool_output("abcdef", 3);
        assert!(capped.starts_with("abc"), "{capped}");
        assert!(capped.contains("truncated"), "{capped}");
        assert!(capped.contains("3 more chars"), "{capped}");
        let multibyte = clamp_tool_output("日本語のテキスト", 3);
        assert!(multibyte.starts_with("日本語"), "{multibyte}");
    }

    #[tokio::test]
    async fn a_fast_agent_finishes_without_waiting_for_a_slow_one() {
        use crate::agent_llm::{LlmError, Provider};
        use std::time::Duration;
        use tokio::sync::Notify;

        // 慢智能体的调用阻塞，直到快智能体被观察到 Finished。
        // 在轮屏障下，快智能体无法被落定（因而无法完成），直到
        // 慢调用返回，而那永远不会发生：死锁。独立推进让快智能体
        // 自行完成并打开慢智能体的闸门。
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
        world.insert_resource(LlmService::from_provider(GateProvider {
            gate: gate.clone(),
        }));
        let fast = world.spawn_with((ReactAgent::new("fast", "be brief").ask("hi"),));
        world.spawn_with((ReactAgent::new("slow", "be brief").ask("hi"),));
        let ctx = SystemContext::new(WorldHandle::new(world));

        // 在快智能体完成的那一刻打开慢智能体的闸门。
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
            FnTool::new("deploy", "ship it", |_ctx, _call| {
                ToolOutcome::text("shipped")
            })
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
        assert_eq!(
            agent.awaiting().map(|request| request.tool()),
            Some("deploy")
        );
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
        assert!(
            agent
                .conversation()
                .render()
                .contains("frozen for the weekend")
        );
        assert_eq!(agent.spend().denied, 1);
    }

    #[tokio::test]
    async fn a_refused_tool_is_not_counted_as_a_tool_failure() {
        // 人工拒绝是策略拦截，不是工具故障：它根本没运行，故不该进
        // tool_calls/tool_failures，而应作为可审计的拒绝单独计数。
        let mut registry = ToolRegistry::new();
        registry.register(
            FnTool::new("deploy", "ship it", |_ctx, _call| ToolOutcome::text("nope"))
                .requiring_approval(),
        );
        let mut world = World::new();
        world.insert_resource(registry);
        world.insert_resource(Observer::new());
        world.insert_resource(LlmService::from_provider(MockProvider::with_script([
            Reply::tool("deploy", json!({})),
            Reply::text("I will not deploy"),
        ])));
        world.spawn_with((ReactAgent::new("ops", "ask first").ask("ship"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new()
            .with_approver(AlwaysReject::new("frozen for the weekend"))
            .run(ctx.clone())
            .await
            .expect("runs");

        let observer = {
            let world = ctx.world.read().await;
            world.resource::<Observer>().cloned().expect("an observer")
        };
        let snapshot = observer.snapshot();
        assert_eq!(
            snapshot.tool_failures,
            0,
            "a refusal is not a tool failure: {}",
            observer.render()
        );
        assert_eq!(
            snapshot.tool_calls,
            0,
            "a refused tool never ran: {}",
            observer.render()
        );
        assert_eq!(
            snapshot.tool_refusals,
            1,
            "the human refusal is counted separately: {}",
            observer.render()
        );
        assert!(
            observer.events().iter().any(|event| matches!(
                &event.kind,
                EventKind::Refused { name, .. } if name == "deploy"
            )),
            "the refusal is visible in the log: {}",
            observer.render()
        );
    }

    #[tokio::test]
    async fn a_failed_model_call_records_the_requested_model() {
        use crate::agent_llm::{LlmError, Provider};

        // 失败的模型调用此前记空模型名，无法归因到具体模型。
        // 有效模型 = 智能体覆盖优先，否则服务默认；这里覆盖须胜出。
        struct FailingProvider;

        #[async_trait::async_trait]
        impl Provider for FailingProvider {
            fn id(&self) -> &str {
                "failing"
            }
            fn default_model(&self) -> &str {
                "provider-default"
            }
            async fn generate(
                &self,
                _request: &GenerationRequest,
            ) -> Result<GenerationResponse, LlmError> {
                Err(LlmError::decode("the endpoint returned garbage"))
            }
        }

        let mut world = World::new();
        world.insert_resource(ToolRegistry::new());
        world.insert_resource(Observer::new());
        world.insert_resource(LlmService::from_provider(FailingProvider));
        world.spawn_with((ReactAgent::new("scribe", "be brief")
            .with_model("specific-model")
            .ask("go"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the run");

        let agent = only_agent(&ctx).await;
        assert_eq!(agent.status(), AgentStatus::Failed);
        let observer = {
            let world = ctx.world.read().await;
            world.resource::<Observer>().cloned().expect("an observer")
        };
        assert!(
            observer.events().iter().any(|event| matches!(
                &event.kind,
                EventKind::Model { failed: true, model, .. } if model == "specific-model"
            )),
            "a failed call must name the requested model, not an empty string: {}",
            observer.render()
        );
    }

    #[tokio::test]
    async fn agents_can_carry_their_own_provider() {
        // 基础架构：每个智能体可自带提供方，世界默认只是回退。
        // 两个 agent 绑不同脚本/模型的提供方，各自的答案与 Observer
        // 归因须来自各自的提供方，而非世界默认。
        let mut world = World::new();
        world.insert_resource(ToolRegistry::new());
        world.insert_resource(Observer::new());
        world.insert_resource(LlmService::from_provider(
            MockProvider::text("default answer").with_model("default-model"),
        ));
        world.spawn_with((ReactAgent::new("alpha", "be brief")
            .with_service(LlmService::from_provider(
                MockProvider::text("alpha answer").with_model("alpha-model"),
            ))
            .ask("go"),));
        world.spawn_with((ReactAgent::new("beta", "be brief")
            .with_service(LlmService::from_provider(
                MockProvider::text("beta answer").with_model("beta-model"),
            ))
            .ask("go"),));
        let ctx = SystemContext::new(WorldHandle::new(world));
        React::new().run(ctx.clone()).await.expect("the run");

        let world = ctx.world.read().await;
        let answers: Vec<(String, String)> = world
            .query::<(Entity, &ReactAgent)>()
            .map(|(_, agent)| {
                (
                    agent.name().to_string(),
                    agent.answer().unwrap_or("").to_string(),
                )
            })
            .collect();
        assert!(
            answers.contains(&("alpha".to_string(), "alpha answer".to_string())),
            "alpha must answer from its own provider: {answers:?}"
        );
        assert!(
            answers.contains(&("beta".to_string(), "beta answer".to_string())),
            "beta must answer from its own provider: {answers:?}"
        );

        let observer = world.resource::<Observer>().unwrap();
        assert!(
            observer.events().iter().any(|event| matches!(
                &event.kind,
                EventKind::Model { model, failed: false, .. } if model == "alpha-model"
            )),
            "alpha's call must be attributed to alpha-model: {}",
            observer.render()
        );
        assert!(
            observer.events().iter().any(|event| matches!(
                &event.kind,
                EventKind::Model { model, failed: false, .. } if model == "beta-model"
            )),
            "beta's call must be attributed to beta-model: {}",
            observer.render()
        );
        assert!(
            !observer.events().iter().any(|event| matches!(
                &event.kind,
                EventKind::Model { model, .. } if model == "default-model"
            )),
            "the world default must not be used when both agents carry their own: {}",
            observer.render()
        );
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
        React::new()
            .run(ctx.clone())
            .await
            .expect("the system stays up");
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
        let ctx = world_with(
            provider,
            ToolRegistry::new(),
            ReactAgent::new("scribe", "hi").ask("go"),
        );
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
                ToolDescriptor::new("slow", "a slow read").with_required_string("id", "identifier")
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
        let ctx = world_with(
            provider,
            registry,
            ReactAgent::new("scribe", "hi").ask("go"),
        );
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
        use crate::agent_llm::{LlmError, Provider};
        use crate::ecs_schedule::StopToken;
        use async_trait::async_trait;
        use tokio::sync::Notify;

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

        // 在任务能运行前登记 interest，然后等到模型调用真正在飞，
        // 使停止落在调用中途，而非之前。
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
        let provider =
            MockProvider::with_script([Reply::tool("boom", json!({})), Reply::text("recovered")]);
        let ctx = world_with(
            provider,
            registry,
            ReactAgent::new("scribe", "be brief").ask("go"),
        );
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
        use crate::agent_llm::{LlmError, Provider};

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
                world
                    .spawn_with((ReactAgent::new(format!("agent-{index}"), "be brief").ask("hi"),));
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
            assert_eq!(
                finished, n,
                "every agent should finish for a valid measurement"
            );
            let serial = latency.as_secs_f64() * n as f64;
            println!(
                "agents={n:3}  wall={wall:>10?}  per-agent={:>10?}  speedup-vs-serial={:5.1}x",
                wall / n as u32,
                serial / wall.as_secs_f64()
            );
        }

        // 零延迟剥去模型调用，使锁 + 调度开销成为唯一剩下的成本：
        // 理论吞吐上限。
        let latency = Duration::ZERO;
        println!("\nzero model latency: exposes lock + scheduling overhead alone");
        for n in [1usize, 8, 32, 64, 256] {
            let mut world = World::new();
            world.insert_resource(ToolRegistry::new());
            world.insert_resource(LlmService::from_provider(LatencyProvider { latency }));
            for index in 0..n {
                world
                    .spawn_with((ReactAgent::new(format!("agent-{index}"), "be brief").ask("hi"),));
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
