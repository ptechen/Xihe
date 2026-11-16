# Xihe

Xihe (羲和) is a tokio-based entity-component-system for building agents. An agent is a component on an entity. A schedule runs systems against the world. The ReAct loop is one of those systems: it reads an agent, calls a model, runs tools, and writes the result back, without holding the world lock across the call.

## Modules

Everything lives in the `xihe` crate. Modules carry an `ecs_` or `agent_` prefix by domain; `agent` itself is the ReAct core.

| Module | What it owns |
| --- | --- |
| `xihe::ecs_core` | Entities, components, resources, queries |
| `xihe::ecs_schedule` | Systems, stages, deferred commands, per-system tracing spans, the stop token |
| `xihe::agent_llm` | Chat requests, providers, a scripted mock |
| `xihe::agent_memory` | Embeddings, a vector store, conversation windows |
| `xihe::agent` | Tools, guardrails, structured output, the ReAct loop |
| `xihe::agent_orchestrator` | Handoffs, routing, a supervisor, fan-out |
| `xihe::agent_observer` | An event log and a snapshot |
| `xihe::agent_runtime` | Plugins, the app, the frame driver |

## Run the demo

```bash
cargo run -p xihe-demo
```

The demo routes a question to a researcher, lets it call a tool, then hands the answer to a scribe. The model is scripted, so it needs no network and no key.

## A minimal agent

```rust
use xihe::prelude::*;
use xihe::MockProvider;

let mut app = App::new();
app.add(ToolsPlugin).await?;
app.add(LlmPlugin::new(LlmService::from_provider(MockProvider::text("hello")))).await?;
app.add(ReactPlugin::new(4)).await?;
app.world_mut()
    .spawn_with((ReactAgent::new("scribe", "be brief").ask("hi"),));
assert_eq!(app.run(4).await?, StopReason::Settled);
```

## Orchestration

Handoffs, routing, fan-out and a supervisor are ordinary systems, so a team is
just a schedule. `RouteTask` dispatches a question, `ReactPlugin` advances every
agent with work, and `HandoffWhenFinished` passes the answer on — all in one
`app.run`. The driver keeps going until the world settles.

On an `app` that already has `ToolsPlugin`, `LlmPlugin` and an `EchoTool` registered:

```rust
use xihe::prelude::*;
use xihe::{Handoff, HandoffWhenFinished, Roster, RouteTask, Router};

let mut roster = Roster::new();
roster.spawn(app.world_mut(), "looks facts up", ReactAgent::new("researcher", "find the fact"));
roster.spawn(app.world_mut(), "writes the answer", ReactAgent::new("scribe", "one sentence"));

app.add_system(SystemSpec::new("route", RouteTask::new(
    Router::new().when_contains("export", "researcher"), roster.clone(), "when does the export refresh?")));
app.add(ReactPlugin::new(6)).await?;
app.add_system(SystemSpec::new("handoff", HandoffWhenFinished::new(
    Handoff::new("researcher", "scribe", "write this up"), roster.clone())));

assert_eq!(app.run(8).await?, StopReason::Settled);
```

`cargo run -p xihe-demo` runs this team end to end; `crates/xihe-ecs-agent/tests/full_stack.rs`
asserts on it through the facade alone.

## Capabilities

- **Memory folding** — `React::new().with_fold(threshold, keep_last)` summarises old turns before the model call.
- **Streaming** — `.with_stream_sink(tx)` forwards `StreamChunk`s while the answer is still being assembled.
- **Parallel read-only tools** — tools marked `readonly()` (or `FnTool::with_readonly()`) run concurrently within a turn.
- **Custom systems drive the loop** — the driver runs until the `Work` resource is idle; any system can `Work::ensure(world).set(label, busy)`.
- **Tracing** — every system activation runs in a `system` span; install any `tracing` subscriber to see per-system timing and failures.
- **Responsive cancellation** — the stop token is honoured mid-model-call, dropping the in-flight request instead of waiting it out.
- **Observability** — `Observer` records frames, systems, models, tools and agents; `snapshot()` and `to_json()` export plain data.

## Features

- `openai` — the HTTP provider in `xihe::agent_llm`
- `hitl` — a terminal approver for tools that require a person
- `persist` — JSONL snapshots of a memory store

```bash
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```
