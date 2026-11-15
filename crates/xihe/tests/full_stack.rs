//! Cross-crate integration: the whole scheduled stack through the facade.
//!
//! These tests use only `xihe::` imports, so they also guard that the facade
//! re-exports everything needed to build a complete application.

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