//! 交互引擎：GUI 经通道驱动一个持有 xihe `App` 的 tokio 工作线程。
//!
//! 终端 App 里的 agent 模式：单个智能体通过 `run_command` 把命令注入
//! 真实 PTY（每次都要人批准），命令输出以块的形式回到终端，工具等到
//! 对应块落定后把输出交回智能体。

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, oneshot, watch};
use xihe::agent::{
    ApprovalRequest, Approver, Decision, Guardrails, LlmService, React, ReactAgent, ToolRegistry,
};
use xihe::agent_llm::{MockProvider, OpenAiConfig, OpenAiProvider, Reply, StreamChunk, ToolCall};
use xihe::agent_observer::{Observer, Snapshot};
use xihe::ecs_core::Entity;
use xihe::ecs_schedule::{Access, SysResult, System, SystemContext, SystemError, SystemSpec, Work};
use xihe::{AgentStatus, App, LlmPlugin, ObserverPlugin, StopReason, ToolsPlugin};

use crate::blocks::Block;
use crate::config::ModelConfig;
use crate::tools::{ListDir, ReadFile, RunInTerminal};

/// 每回合的帧预算。
const FRAME_BUDGET: u64 = 64;

const AGENT_INSTRUCTIONS: &str = "You are a terminal assistant embedded in the user's shell. \
    Use run_command to run shell commands in the user's terminal; every command needs human \
    approval. Prefer short, safe commands. Use read_file and list_dir for inspection without \
    approval. When you know enough, answer briefly in plain text.";

/// 引擎与终端之间的桥：写端注入命令，块快照供工具等待命令落定。
#[derive(Clone)]
pub struct TerminalLink {
    /// PTY 写端。
    pub writer: Arc<Mutex<Box<dyn Write + Send>>>,
    /// 块快照；UI 线程在块变化时发布。
    pub blocks: watch::Receiver<Arc<Vec<Block>>>,
}

/// GUI 发给引擎的命令。
#[derive(Debug)]
pub enum UiCommand {
    /// 一条新的用户消息。
    Ask(String),
    /// 对某次审批的决定。
    Decide {
        /// 审批请求编号。
        id: u64,
        /// 批准或拒绝。
        decision: Decision,
    },
    /// 更换模型配置，下一回合生效。
    Reconfigure(ModelConfig),
    /// 停止引擎。
    Shutdown,
}

/// 引擎发给 GUI 的事件。
#[derive(Clone, Debug)]
pub enum UiEvent {
    /// 智能体状态变化。
    Status {
        /// 智能体名。
        agent: String,
        /// 新状态。
        status: AgentStatus,
    },
    /// 一段流式回答文本。
    Stream {
        /// 正在回答的智能体。
        agent: String,
        /// 新增文本。
        text: String,
    },
    /// 一次工具调用开始运行（已通过审批）。
    ToolStarted {
        /// 调用者。
        agent: String,
        /// 调用本身。
        call: ToolCall,
    },
    /// 一次工具调用结束。
    ToolFinished {
        /// 调用者。
        agent: String,
        /// 工具名。
        name: String,
        /// 是否成功。
        ok: bool,
        /// 输出文本。
        output: String,
    },
    /// 一次工具调用在等待人。
    ApprovalNeeded {
        /// 用于回传决定的编号。
        id: u64,
        /// 请求者。
        agent: String,
        /// 想运行的调用。
        call: ToolCall,
        /// 工具为何需要人。
        reason: String,
    },
    /// 智能体给出了最终回答。
    Answer {
        /// 回答者。
        agent: String,
        /// 回答全文。
        text: String,
    },
    /// 一回合结束。
    EpisodeDone {
        /// 停止原因。
        reason: StopReason,
        /// 观测快照。
        snapshot: Snapshot,
    },
    /// 引擎侧的错误。
    Error(String),
}

/// 事件总线：发送即请求 GUI 重绘。
#[derive(Clone)]
pub struct EventBus {
    tx: mpsc::UnboundedSender<UiEvent>,
    repaint: Option<eframe::egui::Context>,
    names: Arc<HashMap<Entity, String>>,
}

impl EventBus {
    /// 测试用总线：无重绘、无名字表。
    pub fn for_test(tx: mpsc::UnboundedSender<UiEvent>) -> Self {
        Self { tx, repaint: None, names: Arc::new(HashMap::new()) }
    }

    /// 发一个事件并唤醒 GUI。
    pub fn send(&self, event: UiEvent) {
        let _ = self.tx.send(event);
        if let Some(ctx) = &self.repaint {
            ctx.request_repaint();
        }
    }

    /// 实体的智能体名；未知实体回退为调试表示。
    pub fn name(&self, entity: Entity) -> String {
        self.names
            .get(&entity)
            .cloned()
            .unwrap_or_else(|| format!("{entity}"))
    }
}

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Decision>>>>;

/// GUI 侧的引擎句柄。
pub struct EngineHandle {
    /// 命令发送端。
    pub commands: mpsc::UnboundedSender<UiCommand>,
    /// 事件接收端，GUI 每帧抽干。
    pub events: mpsc::UnboundedReceiver<UiEvent>,
}

/// 启动引擎线程；`repaint` 为 GUI 上下文，测试传 `None`。
pub fn spawn(
    config: ModelConfig,
    repaint: Option<eframe::egui::Context>,
    link: TerminalLink,
) -> EngineHandle {
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (ev_tx, ev_rx) = mpsc::unbounded_channel();
    std::thread::Builder::new()
        .name("xihe-engine".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("engine runtime");
            runtime.block_on(run_worker(config, repaint, link, cmd_rx, ev_tx));
        })
        .expect("spawn engine thread");
    EngineHandle { commands: cmd_tx, events: ev_rx }
}

/// 把审批请求发到 GUI，阻塞等待人的决定。
struct GuiApprover {
    bus: EventBus,
    pending: Pending,
    next_id: AtomicU64,
}

#[async_trait::async_trait]
impl Approver for GuiApprover {
    async fn decide(&self, request: &ApprovalRequest) -> Decision {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().expect("pending lock").insert(id, tx);
        self.bus.send(UiEvent::ApprovalNeeded {
            id,
            agent: self.bus.name(request.agent),
            call: request.call.clone(),
            reason: request.reason.clone(),
        });
        rx.await
            .unwrap_or_else(|_| Decision::reject("approval channel closed"))
    }

    fn name(&self) -> &str {
        "gui"
    }
}

/// 待分派的用户消息，由工作线程写入、Intake 取走。
struct Inbox(Option<String>);

/// 常驻的收信系统：把用户消息交给智能体。
struct Intake {
    entity: Entity,
}

#[async_trait::async_trait]
impl System for Intake {
    fn name(&self) -> &str {
        "intake"
    }

    fn access(&self) -> Access {
        Access::exclusive()
    }

    async fn run(&mut self, ctx: SystemContext) -> SysResult {
        let task = ctx
            .with_world_mut(|world| world.resource_mut::<Inbox>().and_then(|inbox| inbox.0.take()))
            .await;
        let Some(task) = task else {
            return Ok(());
        };
        let entity = self.entity;
        ctx.with_world_mut(move |world| -> Result<(), SystemError> {
            let agent = world
                .get_mut::<ReactAgent>(entity)
                .ok_or_else(|| SystemError::new("intake", "agent entity is missing"))?;
            agent.push_question(&task);
            Work::ensure(world).set_owned("agents", true, "react");
            Ok(())
        })
        .await?;
        Ok(())
    }
}

/// 常驻的汇报系统：把状态迁移与最终回答翻译成 GUI 事件。
struct UiBridge {
    entity: Entity,
    bus: EventBus,
    last: Option<AgentStatus>,
}

#[async_trait::async_trait]
impl System for UiBridge {
    fn name(&self) -> &str {
        "ui-bridge"
    }

    fn access(&self) -> Access {
        xihe::access!(reads = [ReactAgent])
    }

    async fn run(&mut self, ctx: SystemContext) -> SysResult {
        let entity = self.entity;
        let snapshot = ctx
            .with_world(move |world| {
                world
                    .get::<ReactAgent>(entity)
                    .map(|agent| (agent.status(), agent.answer().unwrap_or("").to_string()))
            })
            .await;
        if let Some((status, answer)) = snapshot
            && self.last != Some(status)
        {
            self.last = Some(status);
            self.bus.send(UiEvent::Status { agent: "agent".into(), status });
            if status == AgentStatus::Finished {
                self.bus.send(UiEvent::Answer { agent: "agent".into(), text: answer });
            }
        }
        Ok(())
    }
}

fn service_for(config: &ModelConfig) -> Result<LlmService, String> {
    if config.is_live() {
        let openai = OpenAiConfig::build(
            Some(&config.base_url),
            Some(&config.api_key),
            Some(&config.model),
        )
        .map_err(|error| error.to_string())?;
        Ok(LlmService::from_provider(OpenAiProvider::new(openai)))
    } else {
        Ok(LlmService::from_provider(MockProvider::with_script([
            Reply::tool("run_command", serde_json::json!({ "command": "ls" })),
            Reply::text("Listed the directory; see the block in the terminal."),
        ])))
    }
}

async fn build_app(
    config: &ModelConfig,
    tx: mpsc::UnboundedSender<UiEvent>,
    repaint: Option<eframe::egui::Context>,
    pending: Pending,
    link: TerminalLink,
    chunk_tx: mpsc::UnboundedSender<StreamChunk>,
) -> Result<(App, EventBus), String> {
    let mut app = App::new();
    app.add(ToolsPlugin).await.map_err(|e| e.to_string())?;
    app.add(ObserverPlugin).await.map_err(|e| e.to_string())?;
    app.add(LlmPlugin::new(service_for(config)?))
        .await
        .map_err(|e| e.to_string())?;

    let entity = app
        .world_mut()
        .spawn_with((ReactAgent::new("agent", AGENT_INSTRUCTIONS)
            .with_guardrails(Guardrails::new().with_turns(8)),));
    let names: HashMap<Entity, String> = [(entity, "agent".to_string())].into_iter().collect();
    let bus = EventBus { tx, repaint, names: Arc::new(names) };

    let workdir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    {
        let registry = app
            .world_mut()
            .resource_mut::<ToolRegistry>()
            .ok_or("tools plugin missing")?;
        registry.register(RunInTerminal { bus: bus.clone(), link });
        registry.register(ReadFile { bus: bus.clone(), workdir: workdir.clone() });
        registry.register(ListDir { bus: bus.clone(), workdir });
    }

    app.add_system(SystemSpec::new("intake", Intake { entity }));
    let react = React::new()
        .with_steps(6)
        .with_approver(GuiApprover { bus: bus.clone(), pending, next_id: AtomicU64::new(1) })
        .with_stream_sink(chunk_tx);
    app.add_system(SystemSpec::new("react", react));
    // 汇报必须在 react 之后：世界落定的那一帧没有下一帧，
    // 最终回答只有同帧观察才不会被驱动器吞掉。
    app.add_system(SystemSpec::new("ui-bridge", UiBridge { entity, bus: bus.clone(), last: None }));
    app.world_mut().insert_resource(Inbox(None));
    Ok((app, bus))
}

async fn run_worker(
    config: ModelConfig,
    repaint: Option<eframe::egui::Context>,
    link: TerminalLink,
    mut cmd_rx: mpsc::UnboundedReceiver<UiCommand>,
    ev_tx: mpsc::UnboundedSender<UiEvent>,
) {
    let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
    let config_slot = Arc::new(Mutex::new(config.clone()));
    let (ask_tx, mut ask_rx) = mpsc::unbounded_channel::<String>();

    let dispatcher = tokio::spawn({
        let pending = pending.clone();
        let config_slot = config_slot.clone();
        async move {
            while let Some(command) = cmd_rx.recv().await {
                match command {
                    UiCommand::Ask(text) => {
                        if ask_tx.send(text).is_err() {
                            break;
                        }
                    }
                    UiCommand::Decide { id, decision } => {
                        if let Some(tx) = pending.lock().expect("pending lock").remove(&id) {
                            let _ = tx.send(decision);
                        }
                    }
                    UiCommand::Reconfigure(config) => {
                        *config_slot.lock().expect("config lock") = config;
                    }
                    UiCommand::Shutdown => break,
                }
            }
        }
    });

    let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<StreamChunk>();
    let (mut app, bus) =
        match build_app(&config, ev_tx, repaint, pending, link, chunk_tx).await {
            Ok(built) => built,
            Err(error) => {
                eprintln!("xihe-desktop engine failed to start: {error}");
                return;
            }
        };

    let forwarder = tokio::spawn({
        let bus = bus.clone();
        async move {
            while let Some(chunk) = chunk_rx.recv().await {
                if !chunk.text.is_empty() {
                    bus.send(UiEvent::Stream { agent: "agent".into(), text: chunk.text });
                }
            }
        }
    });

    let mut active_config = config;
    while let Some(task) = ask_rx.recv().await {
        let config = config_slot.lock().expect("config lock").clone();
        if config != active_config {
            match service_for(&config) {
                Ok(service) => {
                    if let Some(slot) = app.world_mut().resource_mut::<LlmService>() {
                        *slot = service;
                    }
                    active_config = config;
                }
                Err(error) => bus.send(UiEvent::Error(error)),
            }
        }
        app.world_mut().insert_resource(Inbox(Some(task)));
        match app.run(FRAME_BUDGET).await {
            Ok(reason) => {
                let snapshot = app
                    .world()
                    .resource::<Observer>()
                    .map(|observer| observer.snapshot())
                    .unwrap_or_default();
                bus.send(UiEvent::EpisodeDone { reason, snapshot });
            }
            Err(error) => bus.send(UiEvent::Error(error.to_string())),
        }
    }

    dispatcher.abort();
    forwarder.abort();
}