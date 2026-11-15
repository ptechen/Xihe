//! Agent entities for Xihe: conversation state, tools, the ReAct loop,
//! structured output, guardrails and human-in-the-loop approvals.
//!
//! Nothing in this crate decides *policy* about scheduling or storage; it
//! composes the layers underneath — [`xihe_llm`] for model access,
//! [`xihe_memory`] for recall, [`xihe_core`] and [`xihe_schedule`] for the ECS
//! world and its loop.

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
