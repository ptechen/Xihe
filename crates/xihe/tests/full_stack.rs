//! 跨 crate 集成：通过门面运行整个被调度的栈。
//!
//! 这些测试只用 `xihe::` 导入，因此它们也守护门面重导出构建一个
//! 完整应用所需的一切。

use xihe::prelude::*;
use xihe::{
    AgentStatus, EventKind, Handoff, HandoffWhenFinished, MockProvider, Reply, Roster, RouteTask,
    Router,
};

fn script() -> MockProvider {
    MockProvider::with_script([
        Reply::tool("echo", serde_json::json!({"text": "the export refreshes at 03:00"})),
        Reply::text("the export refreshes at 03:00"),
        Reply::text("The export refreshes at 03:00."),
    ])
}

#[tokio::test]
async fn a_scheduled_team_converges_in_one_run() {
    let mut app = App::new();
    app.add(ToolsPlugin).await.unwrap();
    app.add(ObserverPlugin).await.unwrap();
    app.add(LlmPlugin::new(LlmService::from_provider(script())))
        .await
        .unwrap();
    app.world_mut()
        .resource_mut::<ToolRegistry>()
        .unwrap()
        .register(EchoTool);

    let mut roster = Roster::new();
    roster.spawn(
        app.world_mut(),
        "looks facts up",
        ReactAgent::new("researcher", "find the fact"),
    );
    roster.spawn(
        app.world_mut(),
        "writes the answer",
        ReactAgent::new("scribe", "write one sentence"),
    );

    app.add_system(SystemSpec::new(
        "route",
        RouteTask::new(
            Router::new().when_contains("export", "researcher"),
            roster.clone(),
            "when does the export refresh?",
        ),
    ));
    app.add(ReactPlugin::new(6)).await.unwrap();
    app.add_system(SystemSpec::new(
        "handoff",
        HandoffWhenFinished::new(
            Handoff::new("researcher", "scribe", "write this up"),
            roster.clone(),
        ),
    ));

    let reason = app.run(8).await.unwrap();
    assert_eq!(reason, StopReason::Settled);

    let world = app.world();
    for name in ["researcher", "scribe"] {
        let entity = roster.get(name).unwrap().entity;
        let agent = world.get::<ReactAgent>(entity).unwrap();
        assert_eq!(agent.status(), AgentStatus::Finished, "{name}");
        assert!(agent.answer().is_some(), "{name} answered");
    }

    let observer = world.resource::<Observer>().unwrap();
    let snapshot = observer.snapshot();
    assert!(snapshot.frames >= 2, "{snapshot}");
    assert!(snapshot.model_calls >= 3, "{snapshot}");
    assert!(snapshot.tool_calls >= 1, "{snapshot}");
    assert!(snapshot.agent_events >= 2, "{snapshot}");
    assert_eq!(snapshot.model_failures, 0, "{snapshot}");

    let kinds: Vec<&EventKind> = observer.events().iter().map(|event| &event.kind).collect();
    assert!(kinds.iter().any(|kind| matches!(kind, EventKind::Model { .. })));
    assert!(kinds.iter().any(|kind| matches!(kind, EventKind::Tool { ok: true, .. })));
    assert!(kinds.iter().any(|kind| matches!(kind, EventKind::Agent { .. })));

    assert!(!app.busy(), "the driver settled the work");
}

#[tokio::test]
async fn streaming_and_folding_through_the_facade() {
    let mut agent = ReactAgent::new("scribe", "be brief").ask("hi");
    for index in 0..20 {
        agent.conversation_mut().push_user(format!("turn {index}"));
    }

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<StreamChunk>();
    let mut app = App::new();
    app.add(ToolsPlugin).await.unwrap();
    app.add(LlmPlugin::new(LlmService::from_provider(MockProvider::echo("done"))))
        .await
        .unwrap();
    app.add_system(SystemSpec::new(
        "react",
        React::new().with_fold(10, 2).with_stream_sink(tx),
    ));
    app.world_mut().spawn_with((agent,));

    let reason = app.run(4).await.unwrap();
    assert_eq!(reason, StopReason::Settled);

    let world = app.world();
    let entity = world.single::<ReactAgent>().unwrap();
    let agent = world.get::<ReactAgent>(entity).unwrap();
    assert_eq!(agent.status(), AgentStatus::Finished);
    assert!(agent.conversation().fold_count() > 0, "the fold ran");
    assert!(agent.conversation().summary().is_some());

    let mut text = String::new();
    while let Ok(chunk) = rx.try_recv() {
        text.push_str(&chunk.text);
    }
    assert_eq!(text, "done", "the sink saw the streamed answer");
}

#[tokio::test]
async fn a_custom_system_participates_in_work_driving() {
    struct Pending;

    struct Finisher {
        done: bool,
    }

    #[async_trait::async_trait]
    impl System for Finisher {
        fn name(&self) -> &str {
            "finisher"
        }
        async fn run(&mut self, ctx: SystemContext) -> Result<(), xihe::schedule::SystemError> {
            if !self.done {
                ctx.with_world_mut(|world| {
                    world.insert_resource(Pending);
                    Work::ensure(world).set("custom", true);
                })
                .await;
                self.done = true;
            } else {
                ctx.with_world_mut(|world| Work::ensure(world).set("custom", false))
                    .await;
            }
            Ok(())
        }
    }

    let mut app = App::new();
    app.add_system(SystemSpec::new("finisher", Finisher { done: false }));
    let reason = app.run(8).await.unwrap();
    assert_eq!(reason, StopReason::Settled);
    assert!(app.world().has_resource::<Pending>());
    assert!(!app.busy());
}

#[tokio::test]
async fn a_parallel_fan_out_replays_end_to_end_without_re_running_tools() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use xihe::llm::{GenerationRequest, GenerationResponse, LlmError, Provider, Role, ToolCall};
    use xihe::{FanOut, FanOutResult, FanOutTask, FnTool, Journal, ReplayProvider, ToolOutcome};

    // 每个智能体发布一次（一个工具轮）然后作答，且回答文本取决于
    // 全局调用顺序。重新运行会改变回答并再次触发副作用；
    // 回放必须精确复现两者。
    struct FanProvider {
        calls: Mutex<usize>,
    }

    #[async_trait::async_trait]
    impl Provider for FanProvider {
        fn id(&self) -> &str {
            "fan"
        }
        fn default_model(&self) -> &str {
            "fan"
        }
        async fn generate(
            &self,
            request: &GenerationRequest,
        ) -> Result<GenerationResponse, LlmError> {
            let answered = request.messages.iter().any(|message| message.role == Role::Tool);
            let mut calls = self.calls.lock().unwrap();
            let n = *calls;
            *calls += 1;
            Ok(if answered {
                GenerationResponse::text(format!("answer-{n}"))
            } else {
                GenerationResponse::default().with_tool_calls(vec![ToolCall::new(
                    format!("call_{n}"),
                    "deploy",
                    serde_json::json!({}),
                )])
            })
        }
    }

    async fn build_app<P: Provider>(
        provider: P,
        journal: Journal,
        ships: Arc<AtomicUsize>,
    ) -> (App, Roster) {
        let mut app = App::new();
        app.add(ToolsPlugin).await.unwrap();
        app.add(LlmPlugin::new(LlmService::from_provider(provider)))
            .await
            .unwrap();
        app.world_mut().insert_resource(journal);
        let counter = ships.clone();
        app.world_mut()
            .resource_mut::<ToolRegistry>()
            .unwrap()
            .register(FnTool::new("deploy", "ship the release", move |_ctx, _call| {
                counter.fetch_add(1, Ordering::SeqCst);
                ToolOutcome::ok(serde_json::json!({"shipped": true}), "shipped to staging")
            }));
        let mut roster = Roster::new();
        roster.spawn(
            app.world_mut(),
            "ships releases",
            ReactAgent::new("alpha", "ship carefully"),
        );
        roster.spawn(
            app.world_mut(),
            "ships releases",
            ReactAgent::new("beta", "ship carefully"),
        );
        app.add(ReactPlugin::new(6)).await.unwrap();
        app.add_system(SystemSpec::new(
            "fan",
            FanOutTask::new(
                FanOut::new(["alpha", "beta"]),
                roster.clone(),
                "ship the release",
            ),
        ));
        (app, roster)
    }

    fn answers(app: &App, roster: &Roster) -> (String, String) {
        let world = app.world();
        let answer = |name: &str| {
            world
                .get::<ReactAgent>(roster.get(name).unwrap().entity)
                .unwrap()
                .answer()
                .unwrap_or("")
                .to_string()
        };
        (answer("alpha"), answer("beta"))
    }

    let journal = Journal::new();
    let ships = Arc::new(AtomicUsize::new(0));

    let (mut record_app, record_roster) =
        build_app(FanProvider { calls: Mutex::new(0) }, journal.clone(), ships.clone()).await;
    assert_eq!(record_app.run(12).await.unwrap(), StopReason::Settled);
    let recorded_result = record_app.world().resource::<FanOutResult>().unwrap().0.clone();
    let (recorded_alpha, recorded_beta) = answers(&record_app, &record_roster);
    assert_eq!(ships.load(Ordering::SeqCst), 2, "each agent shipped once while recording");
    assert_ne!(recorded_alpha, recorded_beta, "the provider gave different answers");

    let (mut replay_app, replay_roster) =
        build_app(ReplayProvider::new(journal.clone()), journal, ships.clone()).await;
    assert_eq!(replay_app.run(12).await.unwrap(), StopReason::Settled);
    let replayed_result = replay_app.world().resource::<FanOutResult>().unwrap().0.clone();
    let (replayed_alpha, replayed_beta) = answers(&replay_app, &replay_roster);

    assert_eq!(replayed_alpha, recorded_alpha, "alpha replays its own answer");
    assert_eq!(replayed_beta, recorded_beta, "beta replays its own answer");
    assert_eq!(replayed_result, recorded_result, "the reduced fan-out answer is identical");
    assert_eq!(
        ships.load(Ordering::SeqCst),
        2,
        "replay must not re-run the deploy side effect"
    );
}