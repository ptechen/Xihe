//! Xihe 的智能体实体：对话状态、工具、ReAct 循环、
//! 结构化输出、护栏与人工介入审批。
//!
//! 本 crate 不决定关于调度或存储的任何*策略*；它组合下层——
//! [`crate::agent_llm`] 用于模型访问，[`crate::agent_memory`] 用于召回，
//! [`crate::ecs_core`] 与 [`crate::ecs_schedule`] 用于 ECS 世界及其循环。

pub mod error;
pub mod guardrail;
pub mod hitl;
pub mod llm;
pub mod memory;
pub mod output;
pub mod react;
pub mod replay;
pub mod tool;

pub use error::AgentError;
pub use guardrail::{Guardrails, Spend};
pub use hitl::{AlwaysApprove, AlwaysReject, ApprovalRequest, Approver, Decision, RuleApprover};
pub use llm::LlmService;
pub use memory::{ChatSummariser, MemoryService, fold_window, recall_context, recall_lines};
pub use output::{ResponseFormat, missing_required_keys, parse_json};
pub use react::{AgentStatus, React, ReactAgent};
pub use replay::{Journal, ReplayProvider};
pub use tool::{
    Clock, EchoTool, FixedClock, FnTool, LookupTool, NowTool, SystemClock, Tool, ToolContext,
    ToolOutcome, ToolRegistry,
};
