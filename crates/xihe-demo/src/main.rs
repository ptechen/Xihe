//! 一个脚本化的团队：一个智能体查找事实，然后把答案交给另一个。
//!
//! 用 `cargo run -p xihe-demo` 运行。无需网络；模型是一个带固定脚本的
//! [`xihe::MockProvider`]。
//!
//! 路由与交接是被调度的系统，因此整个团队在一次 [`App::run`] 内运行，
//! 而非分散在多个独立的遍次之间。

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
