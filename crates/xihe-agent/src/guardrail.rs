//! Limits that stop a loop before it costs more than it is worth.
//!
//! A [`Guardrails`] value is declarative: it says what an agent may spend, and
//! [`Guardrails::check`] says whether a given [`Spend`] still fits. The ReAct
//! system asks before every model call and every tool call, so an agent that
//! runs away from itself fails with [`AgentError::Budget`] instead of burning
//! tokens forever.
//!
//! ```
//! use xihe_agent::{Guardrails, Spend};
//!
//! let limits = Guardrails::new().with_turns(3).with_token_budget(1_000);
//! let mut spend = Spend::default();
//! assert!(limits.check_turns(&spend).is_ok());
//! spend.turns = 3;
//! assert!(limits.check_turns(&spend).is_err());
//! ```

use std::fmt::{self, Display, Formatter};
use std::time::Duration;

use xihe_llm::{GenerationResponse, ToolDescriptor};

use crate::AgentError;

/// What one agent has spent so far on its current task.
///
/// Counters only ever go up. [`crate::ReactAgent`] zeroes them when a new task
/// starts, so a budget is always "for this task", not "for this process".
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Spend {
    /// Model calls made.
    pub turns: usize,
    /// Tool calls executed.
    pub tool_calls: usize,
    /// Tool calls refused before execution.
    pub denied: usize,
    /// Input tokens charged.
    pub input_tokens: u64,
    /// Output tokens charged.
    pub output_tokens: u64,
}

impl Spend {
    /// Tokens charged for the whole task.
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Charge a provider response.
    pub fn record_response(&mut self, response: &GenerationResponse) {
        self.turns += 1;
        self.input_tokens += response.usage.prompt_tokens;
        self.output_tokens += response.usage.completion_tokens;
    }

    /// Charge a tool call that actually ran.
    pub fn record_tool_call(&mut self) {
        self.tool_calls += 1;
    }

    /// Charge a tool call that never ran.
    pub fn record_denial(&mut self) {
        self.denied += 1;
    }
}

/// Ceilings on a single agent task.
///
/// Every field is `0`/`None` for "no limit" by default except [`Guardrails::max_turns`],
/// which defaults to [`Guardrails::DEFAULT_TURNS`]: a runaway loop is the failure
/// mode worth guarding against by default.
#[derive(Clone, Eq, PartialEq)]
pub struct Guardrails {
    /// Model calls allowed per task. `0` means unlimited.
    pub max_turns: usize,
    /// Tool calls allowed per task. `0` means unlimited.
    pub max_tool_calls: usize,
    /// Total tokens allowed per task. `0` means unlimited.
    pub max_tokens: u64,
    /// Wall-clock seconds allowed per task. `0` means unlimited.
    pub max_seconds: u64,
    /// Tools the model may see and call; `None` means all registered tools.
    pub allowed_tools: Option<Vec<String>>,
}

impl Guardrails {
    /// Turns an agent gets before the loop is cut off.
    pub const DEFAULT_TURNS: usize = 8;

    /// Tool calls an agent gets before the loop is cut off.
    pub const DEFAULT_TOOL_CALLS: usize = 32;

    /// Reasonable defaults: a turn cap and a tool cap, no token or time ceiling.
    pub fn new() -> Self {
        Self {
            max_turns: Self::DEFAULT_TURNS,
            max_tool_calls: Self::DEFAULT_TOOL_CALLS,
            max_tokens: 0,
            max_seconds: 0,
            allowed_tools: None,
        }
    }

    /// Nothing is capped except the tool allowlist.
    pub fn unlimited() -> Self {
        Self {
            max_turns: 0,
            max_tool_calls: 0,
            max_tokens: 0,
            max_seconds: 0,
            allowed_tools: None,
        }
    }

    /// Allow `turns` model calls, `0` for unlimited.
    pub fn with_turns(mut self, turns: usize) -> Self {
        self.max_turns = turns;
        self
    }

    /// Allow `calls` tool calls, `0` for unlimited.
    pub fn with_tool_calls(mut self, calls: usize) -> Self {
        self.max_tool_calls = calls;
        self
    }

    /// Allow `tokens` input-plus-output tokens, `0` for unlimited.
    pub fn with_token_budget(mut self, tokens: u64) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Allow `limit` of wall-clock time, zero for unlimited.
    pub fn with_time_limit(mut self, limit: Duration) -> Self {
        self.max_seconds = limit.as_secs();
        self
    }

    /// Restrict the agent to `names`, which also hides other tools from the model.
    pub fn allow_tools(self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allowing(names.into_iter().map(Into::into).collect())
    }

    /// Restrict the agent to an explicit list of names.
    pub fn allowing(mut self, names: Vec<String>) -> Self {
        self.allowed_tools = Some(names);
        self
    }

    /// Whether `name` may be called at all.
    pub fn is_tool_allowed(&self, name: &str) -> bool {
        self.allowed_tools
            .as_ref()
            .is_none_or(|allowed| allowed.iter().any(|allowed| allowed == name))
    }

    /// Drop descriptors the allowlist hides.
    ///
    /// The model cannot call what it cannot see, which is a stronger guarantee
    /// than asking it to behave.
    pub fn filter_tools(&self, tools: Vec<ToolDescriptor>) -> Vec<ToolDescriptor> {
        let Some(allowed) = &self.allowed_tools else {
            return tools;
        };
        tools
            .into_iter()
            .filter(|tool| allowed.iter().any(|name| name == &tool.name))
            .collect()
    }

    /// The turn still fits the model-call ceiling.
    pub fn check_turns(&self, spend: &Spend) -> Result<(), AgentError> {
        check("max_turns", spend.turns as u64, self.max_turns as u64)
    }

    /// The turn still fits the tool-call ceiling.
    pub fn check_tool_calls(&self, spend: &Spend) -> Result<(), AgentError> {
        check("max_tool_calls", spend.tool_calls as u64, self.max_tool_calls as u64)
    }

    /// The turn still fits the token ceiling. Checked *before* the call, against
    /// what has already been spent: a provider quotes prices after the fact.
    pub fn check_tokens(&self, spend: &Spend) -> Result<(), AgentError> {
        check("max_tokens", spend.total_tokens(), self.max_tokens)
    }

    /// The turn still fits the wall-clock ceiling.
    pub fn check_time(&self, elapsed: Duration) -> Result<(), AgentError> {
        if self.max_seconds == 0 {
            return Ok(());
        }
        if elapsed.as_secs() < self.max_seconds {
            return Ok(());
        }
        Err(AgentError::budget(
            "max_seconds",
            elapsed.as_secs(),
            self.max_seconds,
        ))
    }

    /// `name` is on the allowlist.
    pub fn check_tool(&self, name: &str) -> Result<(), AgentError> {
        if self.is_tool_allowed(name) {
            return Ok(());
        }
        Err(AgentError::Refused {
            subject: name.to_string(),
            reason: "the tool is not on this agent's allowlist".to_string(),
        })
    }

    /// Turns left before the model-call ceiling, `None` when uncapped.
    pub fn turns_remaining(&self, spend: &Spend) -> Option<usize> {
        if self.max_turns == 0 {
            return None;
        }
        Some(self.max_turns.saturating_sub(spend.turns))
    }

    /// Every ceiling at once; the first one blown decides the error.
    pub fn check(&self, spend: &Spend, elapsed: Duration) -> Result<(), AgentError> {
        self.check_turns(spend)
            .and_then(|()| self.check_tokens(spend))
            .and_then(|()| self.check_time(elapsed))
    }
}

impl Default for Guardrails {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Guardrails {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "turns={}, tools={}, tokens={}, seconds={}",
            limit(self.max_turns as u64),
            limit(self.max_tool_calls as u64),
            limit(self.max_tokens),
            limit(self.max_seconds),
        )?;
        match &self.allowed_tools {
            Some(allowed) => write!(f, ", allowed={}", allowed.join("|")),
            None => write!(f, ", allowed=all"),
        }
    }
}

impl fmt::Debug for Guardrails {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Guardrails")
            .field("max_turns", &self.max_turns)
            .field("max_tool_calls", &self.max_tool_calls)
            .field("max_tokens", &self.max_tokens)
            .field("max_seconds", &self.max_seconds)
            .field("allowed_tools", &self.allowed_tools)
            .finish()
    }
}

fn check(limit: &str, used: u64, allowed: u64) -> Result<(), AgentError> {
    if allowed == 0 || used < allowed {
        return Ok(());
    }
    Err(AgentError::budget(limit, used, allowed))
}

fn limit(value: u64) -> String {
    if value == 0 {
        "none".to_string()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spent(turns: usize, tokens: u64) -> Spend {
        Spend {
            turns,
            tool_calls: 0,
            denied: 0,
            input_tokens: tokens,
            output_tokens: 0,
        }
    }

    #[test]
    fn defaults_stop_a_runaway_loop() {
        let limits = Guardrails::new();
        assert_eq!(limits.max_turns, Guardrails::DEFAULT_TURNS);
        assert!(limits.check(&Spend::default(), Duration::ZERO).is_ok());
        assert!(limits.check(&spent(Guardrails::DEFAULT_TURNS, 0), Duration::ZERO).is_err());
    }

    #[test]
    fn zero_means_unlimited() {
        let limits = Guardrails::unlimited();
        let spend = spent(10_000, 10_000_000);
        assert!(limits.check(&spend, Duration::from_secs(10_000)).is_ok());
    }

    #[test]
    fn the_first_ceiling_blown_wins() {
        let limits = Guardrails::new().with_turns(2).with_token_budget(100);
        let error = limits
            .check(&spent(5, 500), Duration::ZERO)
            .expect_err("both ceilings are blown");
        assert_eq!(error, AgentError::budget("max_turns", 5, 2));
    }

    #[test]
    fn an_allowlist_hides_tools_from_the_model() {
        let limits = Guardrails::new().allow_tools(["lookup"]);
        let tools = vec![
            ToolDescriptor::new("lookup", "read a table"),
            ToolDescriptor::new("deploy", "push to prod"),
        ];
        let kept = limits.filter_tools(tools);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].name, "lookup");
        assert!(!limits.is_tool_allowed("deploy"));
    }

    #[test]
    fn a_denied_tool_is_a_refusal_not_a_budget() {
        let limits = Guardrails::new().allow_tools(["echo"]);
        let error = limits
            .check_tool("rm")
            .expect_err("rm is not allowed");
        assert!(matches!(error, AgentError::Refused { .. }));
    }

    #[test]
    fn remaining_turns_counts_down() {
        let limits = Guardrails::new().with_turns(4);
        assert_eq!(limits.turns_remaining(&spent(1, 0)), Some(3));
        assert_eq!(limits.turns_remaining(&spent(4, 0)), Some(0));
        assert_eq!(
            limits.with_turns(0).turns_remaining(&spent(4, 0)),
            None,
            "uncapped agents never run out"
        );
    }

    #[test]
    fn display_shows_the_ceilings() {
        let limits = Guardrails::new()
            .with_turns(3)
            .with_tool_calls(0)
            .allow_tools(["echo", "lookup"]);
        assert_eq!(
            limits.to_string(),
            "turns=3, tools=none, tokens=none, seconds=none, allowed=echo|lookup"
        );
    }
}
