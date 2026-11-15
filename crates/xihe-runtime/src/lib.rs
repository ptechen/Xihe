//! Assembling a Xihe application and driving it.
//!
//! [`App`] owns the world, the schedule and the stop token. [`Plugin`]s add
//! resources and systems without the caller having to know the order. [`Driver`]
//! runs the schedule until every agent has settled, the frame budget runs out,
//! or something asks the world to stop.

use std::collections::HashSet;
use std::fmt::{self, Display, Formatter};
use std::time::Duration;

use async_trait::async_trait;
use xihe_agent::{LlmService, React, ToolRegistry};
use xihe_core::World;
use xihe_observer::Observer;
use xihe_schedule::{Schedule, StopToken, System, SystemContext, SystemError, SystemSpec, Work, WorldHandle};

/// Why the driver stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// No agent was still working.
    Settled,
    /// The frame budget was used up.
    FrameBudget {
        /// Frames that ran.
        frames: u64,
    },
    /// Something called [`StopToken::stop`].
    Stopped {
        /// The reason that was given, when one was.
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

/// Something that adds resources or systems to an [`App`].
#[async_trait]
pub trait Plugin: Send + Sync {
    /// A short name for logs.
    fn name(&self) -> &str;

    /// Mutate the app. Called once, in registration order, before the first frame.
    async fn install(&self, app: &mut App) -> Result<(), SystemError>;
}

/// The assembled application.
pub struct App {
    world: World,
    schedule: Schedule,
    stop: StopToken,
    installed: Vec<String>,
}

impl App {
    /// An empty world and an empty schedule.
    pub fn new() -> Self {
        Self {
            world: World::new(),
            schedule: Schedule::new(),
            stop: StopToken::new(),
            installed: Vec::new(),
        }
    }

    /// The world, for seeding entities before the driver starts.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// The world, mutable.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// The schedule plugins append to.
    pub fn schedule_mut(&mut self) -> &mut Schedule {
        &mut self.schedule
    }

    /// The stop token the driver and systems share.
    pub fn stop_token(&self) -> &StopToken {
        &self.stop
    }

    /// Plugins that have been installed, in order.
    pub fn installed(&self) -> &[String] {
        &self.installed
    }

    /// Install `plugin`, recording its name.
    pub async fn add<P: Plugin>(&mut self, plugin: P) -> Result<(), SystemError> {
        let name = plugin.name().to_string();
        plugin.install(self).await?;
        self.installed.push(name);
        Ok(())
    }

    /// Insert a resource.
    pub fn insert_resource<R: xihe_core::Resource>(&mut self, value: R) -> &mut Self {
        self.world.insert_resource(value);
        self
    }

    /// Add a system to the schedule.
    pub fn add_system(&mut self, spec: SystemSpec) -> &mut Self {
        self.schedule.add(spec);
        self
    }

    /// Whether any system has reported outstanding work.
    pub fn busy(&self) -> bool {
        self.world
            .resource::<Work>()
            .map(|work| work.is_busy())
            .unwrap_or(false)
    }

    /// Run until the world settles or `max_frames` elapse.
    pub async fn run(&mut self, max_frames: u64) -> Result<StopReason, SystemError> {
        Driver::new(max_frames).drive(self).await
    }

    /// Take the world out, leaving an empty one behind.
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

/// Runs an [`App`]'s schedule frame by frame.
pub struct Driver {
    max_frames: u64,
    observe: bool,
}

impl Driver {
    /// Stop after `max_frames`, or sooner when the world settles.
    pub fn new(max_frames: u64) -> Self {
        Self {
            max_frames: max_frames.max(1),
            observe: true,
        }
    }

    /// Skip writing frames into the [`Observer`], for tests that assert on it.
    pub fn without_observation(mut self) -> Self {
        self.observe = false;
        self
    }

    /// Drive `app` to a [`StopReason`].
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
                // Isolation contains a failure inside `execute`, which then
                // returns Ok; surface each newly-failed system here so a
                // contained failure is still visible in the log, once each.
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

/// Installs an empty tool registry when the world does not have one.
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

/// Installs the ReAct system.
pub struct ReactPlugin {
    steps: usize,
}

impl ReactPlugin {
    /// A loop that runs up to `steps` rounds per frame.
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
        app.schedule
            .add(SystemSpec::new("react", React::new().with_steps(self.steps)));
        Ok(())
    }
}

/// Installs an empty observer.
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

/// Binds a chat client as the world's [`LlmService`].
pub struct LlmPlugin {
    service: LlmService,
}

impl LlmPlugin {
    /// Use `service` for every agent that does not carry its own client.
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

/// A system that stops the world once `frames` have elapsed.
///
/// Useful as a safety rail beside [`Driver::max_frames`].
pub struct FrameLimit {
    frames: u64,
}

impl FrameLimit {
    /// Stop once the frame counter reaches `frames`.
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
    use serde_json::json;
    use xihe_agent::{AgentStatus, EchoTool, FnTool, ReactAgent, ToolOutcome, ToolRegistry};
    use xihe_llm::{MockProvider, Reply};
    use xihe_schedule::{system, SystemSpec, Work};

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
        let agent = app
            .world()
            .query::<&ReactAgent>()
            .next()
            .expect("an agent");
        assert_eq!(agent.answer(), Some("echoed hi"));
        let snapshot = app.world().resource::<Observer>().unwrap().snapshot();
        assert!(snapshot.frames >= 1, "{snapshot}");
    }

    #[tokio::test]
    async fn a_frame_budget_stops_a_runaway_agent() {
        let mut app = App::new();
        app.add(LlmPlugin::new(LlmService::from_provider(MockProvider::with_script([
            Reply::tool("echo", json!({"text": "loop"})),
        ]))))
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
        let reason = Driver::new(2).without_observation().drive(&mut app).await.expect("ran");
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
        app.add(LlmPlugin::new(LlmService::from_provider(MockProvider::with_script([
            Reply::tool("deploy", json!({})),
        ]))))
        .await
        .expect("llm");
        app.add(ToolsPlugin).await.expect("tools");
        app.add(ReactPlugin::new(4)).await.expect("react");
        app.world_mut()
            .resource_mut::<ToolRegistry>()
            .unwrap()
            .register(
                FnTool::new("deploy", "ship it", |_ctx, _call| ToolOutcome::text("shipped"))
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
        let agent = app
            .world()
            .query::<&ReactAgent>()
            .next()
            .expect("an agent");
        assert_eq!(agent.status(), AgentStatus::AwaitingApproval);
    }

    async fn app_with_a_panicking_system() -> App {
        let mut app = App::new();
        app.add(LlmPlugin::new(LlmService::from_provider(MockProvider::with_script([
            Reply::text("done"),
        ]))))
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
        let agent = app
            .world()
            .query::<&ReactAgent>()
            .next()
            .expect("an agent");
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
        app.add(LlmPlugin::new(LlmService::from_provider(MockProvider::with_script([
            Reply::text("done"),
        ]))))
        .await
        .expect("llm");
        app.add(ToolsPlugin).await.expect("tools");
        app.add(ReactPlugin::new(4)).await.expect("react");
        // A peripheral system claims work, then dies. Isolation skips it forever
        // after, so nobody else can clear its label: the driver must release it.
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
        let agent = app
            .world()
            .query::<&ReactAgent>()
            .next()
            .expect("an agent");
        assert_eq!(agent.answer(), Some("done"), "the healthy agent still finished");
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
                xihe_observer::EventKind::System { name, failed: true, .. } if name == "boom"
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
                xihe_observer::EventKind::System { name, failed: true, .. } if name == "boom"
            )),
            "the failing system is named in the log: {}",
            observer.render()
        );
    }

    /// A provider that parks inside `generate` until released, so a test can
    /// signal shutdown while a call is genuinely in flight.
    struct BlockingProvider {
        started: std::sync::Arc<tokio::sync::Notify>,
        release: std::sync::Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl xihe_llm::Provider for BlockingProvider {
        fn id(&self) -> &str {
            "blocking"
        }

        fn default_model(&self) -> &str {
            "blocking"
        }

        async fn generate(
            &self,
            _request: &xihe_llm::GenerationRequest,
        ) -> Result<xihe_llm::GenerationResponse, xihe_llm::LlmError> {
            self.started.notify_one();
            self.release.notified().await;
            Ok(xihe_llm::GenerationResponse::text("done"))
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

        // Stop once the agent is mid-generation, so React's own shutdown path
        // (not the driver's top-of-frame check) is what fails the agent.
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
                xihe_observer::EventKind::Agent { name, status }
                    if name == "scribe" && status == "failed"
            )),
            "the stop-induced failure is visible in the log: {}",
            observer.render()
        );
    }
}
