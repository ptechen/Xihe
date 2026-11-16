//! 组装一个 Xihe 应用并驱动它。
//!
//! [`App`] 拥有世界、调度器与停止令牌。[`Plugin`] 添加资源与系统，
//! 而无需调用者知道顺序。[`Driver`] 运行调度器，直到每个智能体都已落定、
//! 帧预算耗尽，或有东西要求世界停止。

use std::collections::HashSet;
use std::fmt::{self, Display, Formatter};
use std::time::Duration;

use crate::agent::{LlmService, React, ToolRegistry};
use crate::ecs_core::World;
use crate::agent_observer::Observer;
use crate::ecs_schedule::{
    Schedule, StopToken, System, SystemContext, SystemError, SystemSpec, Work, WorldHandle,
};
use async_trait::async_trait;

/// 驱动器为何停止。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// 没有智能体仍在工作。
    Settled,
    /// 帧预算被用尽。
    FrameBudget {
        /// 运行了的帧数。
        frames: u64,
    },
    /// 有东西调用了 [`StopToken::stop`]。
    Stopped {
        /// 给出的原因，若有。
        reason: String,
    },
}

impl Display for StopReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Settled => write!(f, "every agent settled"),
            Self::FrameBudget { frames } => write!(f, "stopped after {frames} frames"),
            Self::Stopped { reason } => write!(f, "stopped: {reason}"),
        }
    }
}

/// 向 [`App`] 添加资源或系统之物。
#[async_trait]
pub trait Plugin: Send + Sync {
    /// 日志中用的简短名字。
    fn name(&self) -> &str;

    /// 变更应用。在第一帧之前，按注册顺序调用一次。
    async fn install(&self, app: &mut App) -> Result<(), SystemError>;
}

/// 组装好的应用。
pub struct App {
    world: World,
    schedule: Schedule,
    stop: StopToken,
    installed: Vec<String>,
}

impl App {
    /// 一个空世界与一个空调度器。
    pub fn new() -> Self {
        Self {
            world: World::new(),
            schedule: Schedule::new(),
            stop: StopToken::new(),
            installed: Vec::new(),
        }
    }

    /// 世界，用于在驱动器启动前播下实体。
    pub fn world(&self) -> &World {
        &self.world
    }

    /// 世界，可变。
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// 插件向其追加的调度器。
    pub fn schedule_mut(&mut self) -> &mut Schedule {
        &mut self.schedule
    }

    /// 驱动器与系统共享的停止令牌。
    pub fn stop_token(&self) -> &StopToken {
        &self.stop
    }

    /// 已安装的插件，按顺序。
    pub fn installed(&self) -> &[String] {
        &self.installed
    }

    /// 安装 `plugin`，记录其名字。
    pub async fn add<P: Plugin>(&mut self, plugin: P) -> Result<(), SystemError> {
        let name = plugin.name().to_string();
        plugin.install(self).await?;
        self.installed.push(name);
        Ok(())
    }

    /// 插入一个资源。
    pub fn insert_resource<R: crate::ecs_core::Resource>(&mut self, value: R) -> &mut Self {
        self.world.insert_resource(value);
        self
    }

    /// 向调度器添加一个系统。
    pub fn add_system(&mut self, spec: SystemSpec) -> &mut Self {
        self.schedule.add(spec);
        self
    }

    /// 是否有系统报告了未完成的工作量。
    pub fn busy(&self) -> bool {
        self.world
            .resource::<Work>()
            .map(|work| work.is_busy())
            .unwrap_or(false)
    }

    /// 运行直到世界落定或 `max_frames` 耗尽。
    pub async fn run(&mut self, max_frames: u64) -> Result<StopReason, SystemError> {
        Driver::new(max_frames).drive(self).await
    }

    /// 取出世界，留下一个空的。
    pub fn into_world(self) -> World {
        self.world
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for App {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("App")
            .field("entities", &self.world.len())
            .field("systems", &self.schedule.len())
            .field("plugins", &self.installed)
            .finish()
    }
}

/// 逐帧运行一个 [`App`] 的调度器。
pub struct Driver {
    max_frames: u64,
    observe: bool,
}

impl Driver {
    /// 在 `max_frames` 之后停止，或当世界落定时更早停止。
    pub fn new(max_frames: u64) -> Self {
        Self {
            max_frames: max_frames.max(1),
            observe: true,
        }
    }

    /// 跳过把帧写入 [`Observer`]，供对其断言的测试使用。
    pub fn without_observation(mut self) -> Self {
        self.observe = false;
        self
    }

    /// 驱动 `app` 直到一个 [`StopReason`]。
    pub async fn drive(&self, app: &mut App) -> Result<StopReason, SystemError> {
        app.schedule
            .validate()
            .map_err(|error| SystemError::new("schedule", error.to_string()))?;
        let handle = WorldHandle::new(std::mem::replace(&mut app.world, World::new()));
        let ctx = SystemContext::new(handle.clone()).with_stop_token(app.stop.clone());
        let mut frames = 0u64;
        let mut reported: HashSet<String> = HashSet::new();
        let reason = loop {
            if app.stop.is_stopped() {
                let reason = app
                    .stop
                    .reason()
                    .map(|reason| reason.into_owned())
                    .unwrap_or_else(|| "stopped".to_string());
                break StopReason::Stopped { reason };
            }
            frames += 1;
            ctx.advance_frame();
            if self.observe {
                let frame = ctx.frame_number();
                ctx.with_world_mut(|world| {
                    Observer::ensure(world).frame(frame);
                })
                .await;
            }
            let started = std::time::Instant::now();
            if let Err(error) = app.schedule.execute(&ctx).await {
                if self.observe {
                    let elapsed = started.elapsed();
                    let message = error.to_string();
                    let name = error.name.clone();
                    ctx.with_world_mut(|world| {
                        let observer = Observer::ensure(world);
                        observer.system(name, true, elapsed);
                        observer.note("driver", message);
                    })
                    .await;
                }
                app.world = take_world(&handle).await;
                return Err(error);
            }
            if self.observe {
                let elapsed = started.elapsed();
                // 隔离把失败容纳在 `execute` 内部，后者随后返回 Ok；
                // 在此浮现每个新失败的系统，使一个被容纳的失败
                // 仍在日志中可见，各一次。
                let failed = app.schedule.failed_labels();
                ctx.with_world_mut(|world| {
                    let observer = Observer::ensure(world);
                    observer.system("schedule", false, elapsed);
                    for label in failed {
                        if reported.insert(label.clone()) {
                            observer.system(label, true, Duration::ZERO);
                        }
                    }
                })
                .await;
            }
            let busy = ctx
                .with_world(|world| {
                    world
                        .resource::<Work>()
                        .map(|work| work.is_busy())
                        .unwrap_or(false)
                })
                .await;
            if !busy {
                break StopReason::Settled;
            }
            if frames >= self.max_frames {
                break StopReason::FrameBudget { frames };
            }
        };
        app.world = take_world(&handle).await;
        Ok(reason)
    }
}

async fn take_world(handle: &WorldHandle) -> World {
    let mut guard = handle.write().await;
    std::mem::replace(&mut *guard, World::new())
}

/// 当世界没有工具注册表时安装一个空的。
pub struct ToolsPlugin;

#[async_trait]
impl Plugin for ToolsPlugin {
    fn name(&self) -> &str {
        "tools"
    }

    async fn install(&self, app: &mut App) -> Result<(), SystemError> {
        if !app.world.has_resource::<ToolRegistry>() {
            app.world.insert_resource(ToolRegistry::new());
        }
        Ok(())
    }
}

/// 安装 ReAct 系统。
pub struct ReactPlugin {
    steps: usize,
}

impl ReactPlugin {
    /// 每帧至多运行 `steps` 轮的循环。
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }
}

impl Default for ReactPlugin {
    fn default() -> Self {
        Self::new(8)
    }
}

#[async_trait]
impl Plugin for ReactPlugin {
    fn name(&self) -> &str {
        "react"
    }

    async fn install(&self, app: &mut App) -> Result<(), SystemError> {
        app.schedule.add(SystemSpec::new(
            "react",
            React::new().with_steps(self.steps),
        ));
        Ok(())
    }
}

/// 安装一个空观察器。
pub struct ObserverPlugin;

#[async_trait]
impl Plugin for ObserverPlugin {
    fn name(&self) -> &str {
        "observer"
    }

    async fn install(&self, app: &mut App) -> Result<(), SystemError> {
        if !app.world.has_resource::<Observer>() {
            app.world.insert_resource(Observer::new());
        }
        Ok(())
    }
}

/// 把一个聊天客户端绑定为世界的 [`LlmService`]。
pub struct LlmPlugin {
    service: LlmService,
}

impl LlmPlugin {
    /// 对每个不自带客户端的智能体使用 `service`。
    pub fn new(service: LlmService) -> Self {
        Self { service }
    }
}

#[async_trait]
impl Plugin for LlmPlugin {
    fn name(&self) -> &str {
        "llm"
    }

    async fn install(&self, app: &mut App) -> Result<(), SystemError> {
        app.world.insert_resource(self.service.clone());
        Ok(())
    }
}

/// 一旦 `frames` 耗尽就停止世界的系统。
///
/// 作为 [`Driver::max_frames`] 旁的一道安全护栏很有用。
pub struct FrameLimit {
    frames: u64,
}

impl FrameLimit {
    /// 一旦帧计数器达到 `frames` 就停止。
    pub fn new(frames: u64) -> Self {
        Self { frames }
    }
}

#[async_trait]
impl System for FrameLimit {
    fn name(&self) -> &str {
        "frame-limit"
    }

    async fn run(&mut self, ctx: SystemContext) -> Result<(), SystemError> {
        if ctx.frame_number() >= self.frames {
            ctx.stop(format!("frame limit {}", self.frames));
        }
        let _ = Duration::from_millis(0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentStatus, EchoTool, FnTool, ReactAgent, ToolOutcome, ToolRegistry};
    use crate::agent_llm::{MockProvider, Reply};
    use crate::ecs_schedule::{SystemSpec, Work, system};
    use serde_json::json;

    async fn echo_app() -> App {
        let mut app = App::new();
        app.add(ToolsPlugin).await.expect("tools");
        app.add(ObserverPlugin).await.expect("observer");
        app.add(LlmPlugin::new(LlmService::from_provider(
            MockProvider::tool_then_answer("echo", json!({"text": "hi"}), "echoed hi"),
        )))
        .await
        .expect("llm");
        app.add(ReactPlugin::new(4)).await.expect("react");
        app.world_mut()
            .resource_mut::<ToolRegistry>()
            .expect("registry")
            .register(EchoTool);
        app.world_mut()
            .spawn_with((ReactAgent::new("scribe", "be brief").ask("say hi"),));
        app
    }

    #[tokio::test]
    async fn a_driver_runs_until_the_agent_finishes() {
        let mut app = echo_app().await;
        let reason = app.run(8).await.expect("driven");
        assert_eq!(reason, StopReason::Settled);
        let agent = app.world().query::<&ReactAgent>().next().expect("an agent");
        assert_eq!(agent.answer(), Some("echoed hi"));
        let snapshot = app.world().resource::<Observer>().unwrap().snapshot();
        assert!(snapshot.frames >= 1, "{snapshot}");
    }

    #[tokio::test]
    async fn a_frame_budget_stops_a_runaway_agent() {
        let mut app = App::new();
        app.add(LlmPlugin::new(LlmService::from_provider(
            MockProvider::with_script([Reply::tool("echo", json!({"text": "loop"}))]),
        )))
        .await
        .expect("llm");
        app.add(ToolsPlugin).await.expect("tools");
        app.add(ReactPlugin::new(1)).await.expect("react");
        app.world_mut()
            .resource_mut::<ToolRegistry>()
            .unwrap()
            .register(EchoTool);
        app.world_mut()
            .spawn_with((ReactAgent::new("loop", "never stop").ask("go"),));
        let reason = Driver::new(2)
            .without_observation()
            .drive(&mut app)
            .await
            .expect("ran");
        assert_eq!(reason, StopReason::FrameBudget { frames: 2 });
        assert!(app.busy());
    }

    #[tokio::test]
    async fn plugins_install_in_order() {
        let mut app = App::new();
        app.add(ToolsPlugin).await.unwrap();
        app.add(ObserverPlugin).await.unwrap();
        assert_eq!(app.installed(), ["tools", "observer"]);
        assert!(app.world().has_resource::<ToolRegistry>());
        assert!(app.world().has_resource::<Observer>());
    }

    #[tokio::test]
    async fn a_stop_token_ends_the_next_frame() {
        let mut app = App::new();
        app.add(ReactPlugin::default()).await.unwrap();
        app.world_mut()
            .spawn_with((ReactAgent::new("scribe", "hi").ask("go"),));
        app.stop_token().stop("tests are done");
        let reason = app.run(4).await.expect("stopped");
        assert_eq!(
            reason,
            StopReason::Stopped {
                reason: "tests are done".into()
            }
        );
    }

    #[tokio::test]
    async fn a_world_parked_on_a_human_settles_instead_of_burning_frames() {
        let mut app = App::new();
        app.add(LlmPlugin::new(LlmService::from_provider(
            MockProvider::with_script([Reply::tool("deploy", json!({}))]),
        )))
        .await
        .expect("llm");
        app.add(ToolsPlugin).await.expect("tools");
        app.add(ReactPlugin::new(4)).await.expect("react");
        app.world_mut()
            .resource_mut::<ToolRegistry>()
            .unwrap()
            .register(
                FnTool::new("deploy", "ship it", |_ctx, _call| {
                    ToolOutcome::text("shipped")
                })
                .requiring_approval(),
            );
        app.world_mut()
            .spawn_with((ReactAgent::new("ops", "ship carefully").ask("ship"),));

        let reason = Driver::new(4)
            .without_observation()
            .drive(&mut app)
            .await
            .expect("ran");
        assert_eq!(
            reason,
            StopReason::Settled,
            "waiting on a human is not owed work; the driver must not burn frames"
        );
        assert!(!app.busy(), "a parked world owes no compute");
        let agent = app.world().query::<&ReactAgent>().next().expect("an agent");
        assert_eq!(agent.status(), AgentStatus::AwaitingApproval);
    }

    async fn app_with_a_panicking_system() -> App {
        let mut app = App::new();
        app.add(LlmPlugin::new(LlmService::from_provider(
            MockProvider::with_script([Reply::text("done")]),
        )))
        .await
        .expect("llm");
        app.add(ToolsPlugin).await.expect("tools");
        app.add(ReactPlugin::new(4)).await.expect("react");
        app.add_system(SystemSpec::new(
            "boom",
            system("boom", |_ctx| async { panic!("the plugin exploded") }),
        ));
        app.world_mut()
            .spawn_with((ReactAgent::new("scribe", "be brief").ask("hi"),));
        app
    }

    #[tokio::test]
    async fn isolating_system_errors_lets_healthy_agents_finish() {
        let mut app = app_with_a_panicking_system().await;
        app.schedule_mut().isolate_system_errors();

        let reason = app
            .run(4)
            .await
            .expect("a bad system must not abort the whole app");
        assert_eq!(reason, StopReason::Settled);
        let agent = app.world().query::<&ReactAgent>().next().expect("an agent");
        assert_eq!(agent.answer(), Some("done"));
        assert!(
            app.schedule_mut()
                .failed_labels()
                .iter()
                .any(|label| label == "boom"),
            "the failing system is recorded and skipped, not silently retried"
        );
    }

    #[tokio::test]
    async fn without_isolation_a_failing_system_still_aborts_the_run() {
        let mut app = app_with_a_panicking_system().await;
        let result = app.run(4).await;
        assert!(result.is_err(), "fail-fast remains the default");
    }

    #[tokio::test]
    async fn isolating_a_failing_system_releases_its_work_so_the_world_settles() {
        let mut app = App::new();
        app.add(LlmPlugin::new(LlmService::from_provider(
            MockProvider::with_script([Reply::text("done")]),
        )))
        .await
        .expect("llm");
        app.add(ToolsPlugin).await.expect("tools");
        app.add(ReactPlugin::new(4)).await.expect("react");
        // 一个外围系统认领工作量，然后死掉。隔离此后永远跳过它，
        // 因此没有别人能清除它的标签：驱动器必须释放它。
        app.add_system(SystemSpec::new(
            "ingest",
            system("ingest", |ctx: SystemContext| async move {
                ctx.with_world_mut(|world| Work::ensure(world).set_owned("ingest", true, "ingest"))
                    .await;
                panic!("ingest exploded after claiming work");
            }),
        ));
        app.world_mut()
            .spawn_with((ReactAgent::new("scribe", "be brief").ask("hi"),));
        app.schedule_mut().isolate_system_errors();

        let reason = app
            .run(8)
            .await
            .expect("an isolated failure must not abort the app");
        assert_eq!(
            reason,
            StopReason::Settled,
            "a failed system's claimed work must be released, not orphaned"
        );
        assert!(!app.busy(), "no label keeps the driver awake");
        let agent = app.world().query::<&ReactAgent>().next().expect("an agent");
        assert_eq!(
            agent.answer(),
            Some("done"),
            "the healthy agent still finished"
        );
    }

    #[tokio::test]
    async fn isolated_failures_are_visible_in_the_observer() {
        let mut app = app_with_a_panicking_system().await;
        app.schedule_mut().isolate_system_errors();

        let reason = app.run(4).await.expect("isolated");
        assert_eq!(reason, StopReason::Settled);

        let observer = app.world().resource::<Observer>().expect("observer");
        let snapshot = observer.snapshot();
        assert!(
            snapshot.system_failures >= 1,
            "a contained failure must still be observable: {snapshot}"
        );
        assert!(
            observer.events().iter().any(|event| matches!(
                &event.kind,
                crate::agent_observer::EventKind::System { name, failed: true, .. } if name == "boom"
            )),
            "the failing system is named in the log: {}",
            observer.render()
        );
    }

    #[tokio::test]
    async fn a_fail_fast_failure_is_named_in_the_observer() {
        let mut app = app_with_a_panicking_system().await;
        let result = app.run(4).await;
        assert!(result.is_err(), "fail-fast remains the default");

        let observer = app.world().resource::<Observer>().expect("observer");
        let snapshot = observer.snapshot();
        assert!(
            snapshot.system_failures >= 1,
            "a fail-fast failure must be observable: {snapshot}"
        );
        assert!(
            observer.events().iter().any(|event| matches!(
                &event.kind,
                crate::agent_observer::EventKind::System { name, failed: true, .. } if name == "boom"
            )),
            "the failing system is named in the log: {}",
            observer.render()
        );
    }

    /// 一个在 `generate` 内部挂起直到被释放的提供方，使测试能在
    /// 一次调用真正在飞时发出关停信号。
    struct BlockingProvider {
        started: std::sync::Arc<tokio::sync::Notify>,
        release: std::sync::Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl crate::agent_llm::Provider for BlockingProvider {
        fn id(&self) -> &str {
            "blocking"
        }

        fn default_model(&self) -> &str {
            "blocking"
        }

        async fn generate(
            &self,
            _request: &crate::agent_llm::GenerationRequest,
        ) -> Result<crate::agent_llm::GenerationResponse, crate::agent_llm::LlmError> {
            self.started.notify_one();
            self.release.notified().await;
            Ok(crate::agent_llm::GenerationResponse::text("done"))
        }
    }

    #[tokio::test]
    async fn stopping_the_runtime_records_the_failed_agents_in_the_observer() {
        let started = std::sync::Arc::new(tokio::sync::Notify::new());
        let release = std::sync::Arc::new(tokio::sync::Notify::new());
        let provider = BlockingProvider {
            started: started.clone(),
            release: release.clone(),
        };

        let mut app = App::new();
        app.add(LlmPlugin::new(LlmService::from_provider(provider)))
            .await
            .expect("llm");
        app.add(ToolsPlugin).await.expect("tools");
        app.add(ReactPlugin::new(4)).await.expect("react");
        app.world_mut()
            .spawn_with((ReactAgent::new("scribe", "be brief").ask("hi"),));

        // 一旦智能体处于生成中途就停止，使 React 自己的关停路径
        //（而非驱动器的帧顶检查）才是让智能体失败的那个。
        let stop = app.stop_token().clone();
        let signaller = tokio::spawn(async move {
            started.notified().await;
            stop.stop("halt");
        });

        app.run(4).await.expect("ran");
        signaller.await.expect("signaller");

        let world = app.world();
        let agent = world.query::<&ReactAgent>().next().expect("an agent");
        assert_eq!(agent.status(), AgentStatus::Failed);
        let observer = world.resource::<Observer>().expect("observer");
        assert!(
            observer.events().iter().any(|event| matches!(
                &event.kind,
                crate::agent_observer::EventKind::Agent { name, status }
                    if name == "scribe" && status == "failed"
            )),
            "the stop-induced failure is visible in the log: {}",
            observer.render()
        );
    }
}
