//! Xihe (羲和) —— 一个面向智能体的实体-组件-系统。
//!
//! 世界持有智能体、工具与记忆。调度器针对该世界运行系统。智能体是一个
//! 组件，而非一个线程：同一个 [`React`] 系统推进每个有工作量的智能体，
//! 而 [`Driver`] 持续运行直到它们落定。
//!
//! ```
//! use xihe::prelude::*;
//! use xihe::MockProvider;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mut app = App::new();
//! app.add(ToolsPlugin).await?;
//! app.add(LlmPlugin::new(LlmService::from_provider(MockProvider::text("hello")))).await?;
//! app.add(ReactPlugin::new(4)).await?;
//! app.world_mut().spawn_with((ReactAgent::new("scribe", "be brief").ask("hi"),));
//! let reason = app.run(4).await?;
//! assert_eq!(reason, StopReason::Settled);
//! # Ok(())
//! # }
//! # tokio::runtime::Builder::new_current_thread()
//! #     .enable_all().build().unwrap().block_on(async { run().await.unwrap(); });
//! ```
//!
//! 一个团队就是一个调度器：路由、ReAct 循环与交接都是系统，
//! 一次 [`App::run`] 驱动它们直到世界落定。
//!
//! ```
//! use xihe::prelude::*;
//! use xihe::{Handoff, HandoffWhenFinished, MockProvider, Reply, Roster, RouteTask, Router};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mut app = App::new();
//! app.add(ToolsPlugin).await?;
//! app.add(LlmPlugin::new(LlmService::from_provider(MockProvider::with_script([
//!     Reply::tool("echo", serde_json::json!({"text": "03:00"})),
//!     Reply::text("03:00"),
//!     Reply::text("It refreshes at 03:00."),
//! ])))).await?;
//! app.world_mut().resource_mut::<ToolRegistry>().unwrap().register(EchoTool);
//!
//! let mut roster = Roster::new();
//! roster.spawn(app.world_mut(), "looks facts up", ReactAgent::new("researcher", "find the fact"));
//! roster.spawn(app.world_mut(), "writes the answer", ReactAgent::new("scribe", "one sentence"));
//!
//! app.add_system(SystemSpec::new("route", RouteTask::new(
//!     Router::new().when_contains("export", "researcher"), roster.clone(), "when does the export refresh?")));
//! app.add(ReactPlugin::new(6)).await?;
//! app.add_system(SystemSpec::new("handoff", HandoffWhenFinished::new(
//!     Handoff::new("researcher", "scribe", "write this up"), roster.clone())));
//!
//! assert_eq!(app.run(8).await?, StopReason::Settled);
//! # Ok(())
//! # }
//! # tokio::runtime::Builder::new_current_thread()
//! #     .enable_all().build().unwrap().block_on(async { run().await.unwrap(); });
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub use xihe_agent as agent;
pub use xihe_core as core;
pub use xihe_llm as llm;
pub use xihe_memory as memory;
pub use xihe_observer as observer;
pub use xihe_orchestrator as orchestrator;
pub use xihe_runtime as runtime;
pub use xihe_schedule as schedule;

pub use xihe_agent::{
    AgentError, AgentStatus, EchoTool, FnTool, Guardrails, Journal, LlmService, LookupTool,
    MemoryService, React, ReactAgent, ReplayProvider, ResponseFormat, Tool, ToolContext, ToolOutcome,
    ToolRegistry,
};
pub use xihe_core::{Entity, World};
pub use xihe_llm::{ChatClient, MockProvider, Reply, StreamChunk};
pub use xihe_memory::{ConversationWindow, InMemoryStore, MemoryKind};
pub use xihe_observer::{EventKind, Observer, Snapshot};
pub use xihe_orchestrator::{
    Blackboard, FanOut, FanOutResult, FanOutTask, Handoff, HandoffWhenFinished, Roster, RouteTask,
    Router, Supervisor, SupervisorAssign,
};
pub use xihe_runtime::{App, Driver, LlmPlugin, ObserverPlugin, Plugin, ReactPlugin, StopReason, ToolsPlugin};
pub use xihe_schedule::{Schedule, System, SystemContext, SystemSpec, Work, WorldHandle};

/// demo 或二进制通常希望引入作用域的名字。
pub mod prelude {
    pub use crate::{
        App, Driver, EchoTool, Entity, LlmPlugin, LlmService, Observer, ObserverPlugin, React,
        ReactAgent, ReactPlugin, StopReason, StreamChunk, System, SystemContext, SystemSpec,
        ToolContext, ToolRegistry, ToolsPlugin, Work, World,
    };
}
