//! Failures an agent can report.
//!
//! [`AgentError`] is the single type a ReAct run returns. It wraps the failure
//! of whichever layer actually broke — the provider, the memory store, a tool,
//! or the agent's own limits — and keeps the retryable flag of the innermost
//! cause, so a caller can tell "call again" from "give up".

use std::fmt::{self, Display, Formatter};

use xihe_llm::LlmError;
use xihe_memory::MemoryError;

/// Anything that stopped an agent from finishing a turn.
#[derive(Clone, Debug, PartialEq)]
pub enum AgentError {
    /// The provider failed; the reason is kept as text plus retryability.
    Provider {
        /// What the provider layer reported.
        message: String,
        /// Whether the same request could succeed if repeated.
        retryable: bool,
    },
    /// Memory could not be read or written.
    Memory {
        /// What the memory layer reported.
        message: String,
    },
    /// A tool refused or failed.
    Tool {
        /// Name of the tool that failed.
        name: String,
        /// What it reported.
        message: String,
    },
    /// The agent asked for a tool that is not registered.
    UnknownTool {
        /// The name the model produced.
        name: String,
        /// Registered names, for the error message.
        available: Vec<String>,
    },
    /// A limit stopped the loop: turns, tokens, or wall-clock.
    Budget {
        /// Which limit, e.g. `max_turns`.
        limit: String,
        /// The value that was exceeded.
        used: u64,
        /// The configured ceiling.
        allowed: u64,
    },
    /// Input or configuration was unusable.
    Config {
        /// What was wrong.
        message: String,
    },
    /// A human declined to let the step run.
    Refused {
        /// The tool or step that was rejected.
        subject: String,
        /// Why, as given by the reviewer.
        reason: String,
    },
    /// The model's answer could not be read as the required structure.
    Output {
        /// Parse failure detail.
        message: String,
        /// The text that was parsed, truncated.
        raw: String,
    },
    /// Cooperative shutdown arrived mid-turn.
    Stopped,
}

impl AgentError {
    /// A configuration or argument problem.
    pub fn config(message: impl Display) -> Self {
        Self::Config {
            message: message.to_string(),
        }
    }

    /// A limit was reached.
    pub fn budget(limit: impl Into<String>, used: u64, allowed: u64) -> Self {
        Self::Budget {
            limit: limit.into(),
            used,
            allowed,
        }
    }

    /// A tool failed.
    pub fn tool(name: impl Into<String>, message: impl Display) -> Self {
        Self::Tool {
            name: name.into(),
            message: message.to_string(),
        }
    }

    /// The answer did not match the required shape.
    pub fn output(message: impl Display, raw: impl Into<String>) -> Self {
        let raw = raw.into();
        let cut = raw.chars().take(200).collect::<String>();
        Self::Output {
            message: message.to_string(),
            raw: cut,
        }
    }

    /// True when repeating the turn could plausibly work.
    ///
    /// Only provider failures and tool failures carry that signal; budgets,
    /// refusals and malformed output are the caller's decision to make.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Provider { retryable, .. } => *retryable,
            Self::Tool { .. } | Self::Memory { .. } => true,
            Self::Stopped
            | Self::Config { .. }
            | Self::Budget { .. }
            | Self::Refused { .. }
            | Self::Output { .. }
            | Self::UnknownTool { .. } => false,
        }
    }

    /// A short label for metrics and logs.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Provider { .. } => "provider",
            Self::Memory { .. } => "memory",
            Self::Tool { .. } => "tool",
            Self::UnknownTool { .. } => "unknown_tool",
            Self::Budget { .. } => "budget",
            Self::Config { .. } => "config",
            Self::Refused { .. } => "refused",
            Self::Output { .. } => "output",
            Self::Stopped => "stopped",
        }
    }

    /// Wrap a provider failure, keeping its retryability.
    pub fn provider(error: &LlmError) -> Self {
        Self::Provider {
            message: error.to_string(),
            retryable: error.is_retryable(),
        }
    }

    /// Wrap a memory failure.
    pub fn memory(error: &MemoryError) -> Self {
        Self::Memory {
            message: error.to_string(),
        }
    }
}

impl Display for AgentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider { message, .. } => write!(f, "provider failed: {message}"),
            Self::Memory { message } => write!(f, "memory failed: {message}"),
            Self::Tool { name, message } => write!(f, "tool `{name}` failed: {message}"),
            Self::UnknownTool { name, available } => write!(
                f,
                "tool `{name}` is not registered; available: [{}]",
                available.join(", ")
            ),
            Self::Budget {
                limit,
                used,
                allowed,
            } => write!(f, "{limit} limit reached: {used} of {allowed}"),
            Self::Config { message } => write!(f, "invalid agent configuration: {message}"),
            Self::Refused { subject, reason } => {
                write!(f, "`{subject}` was refused: {reason}")
            }
            Self::Output { message, raw } => {
                write!(f, "output could not be read ({message}): {raw}")
            }
            Self::Stopped => f.write_str("the run was stopped"),
        }
    }
}

impl std::error::Error for AgentError {}

impl From<LlmError> for AgentError {
    fn from(error: LlmError) -> Self {
        Self::provider(&error)
    }
}

impl From<MemoryError> for AgentError {
    fn from(error: MemoryError) -> Self {
        Self::Memory {
            message: error.to_string(),
        }
    }
}

impl From<serde_json::Error> for AgentError {
    fn from(error: serde_json::Error) -> Self {
        Self::Config {
            message: format!("json could not be handled: {error}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn timeout() -> LlmError {
        LlmError::Timeout {
            elapsed: Duration::from_secs(1),
        }
    }

    #[test]
    fn each_variant_names_its_layer() {
        let cases = [
            (AgentError::provider(&timeout()), "provider"),
            (AgentError::memory(&MemoryError::embedding("full")), "memory"),
            (AgentError::tool("search", "boom"), "tool"),
            (
                AgentError::UnknownTool {
                    name: "miss".into(),
                    available: vec!["hit".into()],
                },
                "unknown_tool",
            ),
            (AgentError::budget("max_turns", 9, 8), "budget"),
            (AgentError::config("no model"), "config"),
            (
                AgentError::Refused {
                    subject: "delete".into(),
                    reason: "unsafe".into(),
                },
                "refused",
            ),
            (AgentError::output("bad json", "{}"), "output"),
            (AgentError::Stopped, "stopped"),
        ];
        for (error, kind) in cases {
            assert_eq!(error.kind(), kind, "{error}");
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn retryability_follows_the_cause() {
        assert!(AgentError::provider(&timeout()).is_retryable());
        assert!(AgentError::tool("x", "network").is_retryable());
        assert!(AgentError::memory(&MemoryError::embedding("x")).is_retryable());
        assert!(!AgentError::budget("max_turns", 9, 8).is_retryable());
        assert!(!AgentError::config("bad").is_retryable());
        assert!(!AgentError::Stopped.is_retryable());
        let refused = AgentError::Refused {
            subject: "s".into(),
            reason: "r".into(),
        };
        assert!(!refused.is_retryable());
    }

    #[test]
    fn provider_retryability_is_preserved() {
        let retryable = AgentError::provider(&timeout());
        assert!(retryable.is_retryable(), "{retryable}");
        let permanent = AgentError::provider(&LlmError::config("no key"));
        assert!(!permanent.is_retryable(), "{permanent}");
    }

    #[test]
    fn long_output_previews_are_truncated() {
        let error = AgentError::output("missing field", "x".repeat(500));
        let AgentError::Output { raw, .. } = error else {
            panic!("wrong variant");
        };
        assert_eq!(raw.chars().count(), 200);
    }

    #[test]
    fn conversions_keep_the_message() {
        let from_llm: AgentError = timeout().into();
        assert!(from_llm.to_string().contains("timed out"), "{from_llm}");
        let from_memory: AgentError = MemoryError::disabled("persist").into();
        assert!(from_memory.to_string().contains("persist"));
        let broken: Result<u8, _> = serde_json::from_str("nope");
        let from_json: AgentError = broken.unwrap_err().into();
        assert_eq!(from_json.kind(), "config");
    }

    #[test]
    fn errors_are_std_errors() {
        let boxed: Box<dyn std::error::Error + Send + Sync> = Box::new(AgentError::Stopped);
        assert_eq!(boxed.to_string(), "the run was stopped");
    }
}
