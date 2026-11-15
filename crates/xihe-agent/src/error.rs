//! 智能体可报告的失败。
//!
//! [`AgentError`] 是一次 ReAct 运行返回的唯一类型。它包装
//! 真正出问题的那一层——提供方、记忆存储、工具，
//! 或智能体自身的限额——的失败，并保留最内层原因的可重试标志，
//! 使调用者能区分“再调一次”与“放弃”。

use std::fmt::{self, Display, Formatter};

use xihe_llm::LlmError;
use xihe_memory::MemoryError;

/// 任何阻止智能体完成一轮之事。
#[derive(Clone, Debug, PartialEq)]
pub enum AgentError {
    /// 提供方失败；原因以文本加可重试性保留。
    Provider {
        /// 提供方层报告了什么。
        message: String,
        /// 同一请求若重复是否可能成功。
        retryable: bool,
    },
    /// 记忆无法读取或写入。
    Memory {
        /// 记忆层报告了什么。
        message: String,
    },
    /// 工具拒绝或失败。
    Tool {
        /// 失败工具的名字。
        name: String,
        /// 它报告了什么。
        message: String,
    },
    /// 智能体请求了一个未注册的工具。
    UnknownTool {
        /// 模型产出的名字。
        name: String,
        /// 已注册的名字，用于错误消息。
        available: Vec<String>,
    },
    /// 一个限额停止了循环：轮数、token 或墙钟。
    Budget {
        /// 哪个限额，例如 `max_turns`。
        limit: String,
        /// 被超出的值。
        used: u64,
        /// 配置的上限。
        allowed: u64,
    },
    /// 输入或配置不可用。
    Config {
        /// 哪里出了问题。
        message: String,
    },
    /// 人工拒绝让该步骤运行。
    Refused {
        /// 被拒绝的工具或步骤。
        subject: String,
        /// 原因，由审查者给出。
        reason: String,
    },
    /// 模型的回答无法按所需结构读取。
    Output {
        /// 解析失败详情。
        message: String,
        /// 被解析的文本，已截断。
        raw: String,
    },
    /// 协作式关停在一轮中途到达。
    Stopped,
}

impl AgentError {
    /// 一个配置或参数问题。
    pub fn config(message: impl Display) -> Self {
        Self::Config {
            message: message.to_string(),
        }
    }

    /// 达到了一个限额。
    pub fn budget(limit: impl Into<String>, used: u64, allowed: u64) -> Self {
        Self::Budget {
            limit: limit.into(),
            used,
            allowed,
        }
    }

    /// 一个工具失败了。
    pub fn tool(name: impl Into<String>, message: impl Display) -> Self {
        Self::Tool {
            name: name.into(),
            message: message.to_string(),
        }
    }

    /// 回答不符合所需的形状。
    pub fn output(message: impl Display, raw: impl Into<String>) -> Self {
        let raw = raw.into();
        let cut = raw.chars().take(200).collect::<String>();
        Self::Output {
            message: message.to_string(),
            raw: cut,
        }
    }

    /// 当重复这一轮有可能奏效时为真。
    ///
    /// 只有提供方失败与工具失败携带该信号；预算、
    /// 拒绝与畸形输出由调用者决定。
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

    /// 用于指标与日志的简短标签。
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

    /// 包装一个提供方失败，保留其可重试性。
    pub fn provider(error: &LlmError) -> Self {
        Self::Provider {
            message: error.to_string(),
            retryable: error.is_retryable(),
        }
    }

    /// 包装一个记忆失败。
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
