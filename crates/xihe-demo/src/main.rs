//! A scripted team: one agent looks a fact up, then hands the answer to another.
//!
//! Run with `cargo run -p xihe-demo`. No network is required; the model is a
//! [`xihe::MockProvider`] with a fixed script.
//!
//! Routing and hand-off are scheduled systems, so the whole team runs inside
//! one [`App::run`] instead of between separate passes.

use xihe::prelude::*;
use xihe::{Handoff, HandoffWhenFinished, MockProvider, Reply, Roster, RouteTask, Router, SystemSpec};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::new();
    app.add(ToolsPlugin).await?;
    app.add(ObserverPlugin).await?;
    app.add(LlmPlugin::new(LlmService::from_provider(script()))).await?;

    app.world_mut()
        .resource_mut::<ToolRegistry>()
        .expect("tools plugin")
        .register(EchoTool);

    let mut roster = Roster::new();
    roster.spawn(
        app.world_mut(),
        "looks facts up",
        ReactAgent::new("researcher", "find the fact, then stop")
            .with_guardrails(xihe::agent::Guardrails::new().with_turns(3)),
    );
    roster.spawn(
        app.world_mut(),
        "writes the final answer",
        ReactAgent::new("scribe", "write one sentence"),
    );

    let router = Router::new().when_contains("export", "researcher");
    app.add_system(SystemSpec::new(
        "route",
        RouteTask::new(router, roster.clone(), "when does the export refresh?"),
    ));
    app.add(ReactPlugin::new(6)).await?;
    app.add_system(SystemSpec::new(
        "handoff",
        HandoffWhenFinished::new(
            Handoff::new("researcher", "scribe", "write this up for the reader"),
            roster.clone(),
        ),
    ));

    let reason = app.run(8).await?;
    println!("{reason}");
    report(&app, &roster);
    Ok(())
}

fn report(app: &App, roster: &Roster) {
    let world = app.world();
    for entry in roster.entries() {
        if let Some(agent) = world.get::<ReactAgent>(entry.entity) {
            println!(
                "  {} [{}] {}",
                entry.name,
                agent.status(),
                agent.answer().unwrap_or("(no answer)")
            );
        }
    }
    if let Some(observer) = world.resource::<Observer>() {
        println!("{}", observer.snapshot());
    }
}

fn script() -> MockProvider {
    MockProvider::with_script([
        Reply::tool(
            "echo",
            serde_json::json!({"text": "the export refreshes at 03:00"}),
        ),
        Reply::text("the export refreshes at 03:00"),
        Reply::text("The export refreshes at 03:00."),
    ])
}
