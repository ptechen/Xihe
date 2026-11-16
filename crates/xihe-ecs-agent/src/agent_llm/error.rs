//! 提供方可返回的失败，以及运行时是否应重试它们。

use std::fmt::{self, Display, Formatter};
use std::time::Duration;

/// 我们与模型之间可能出错的一切。
///
/// 各变体按*处置方式*而非 HTTP 细节划分：重试、停止或重新配置。
/// [`LlmError::is_retryable`] 正是 [`crate::agent_llm::ChatClient`] 所信赖的。
#[derive(Clone, Debug, PartialEq)]
pub enum LlmError {
    /// 请求从未产出可用响应（DNS、连接、TLS、重置）。
    /// 请求从未到达服务器，或连接中断。
    Transport {
        /// 来自传输层的可读细节。
        message: String,
    },

    /// 服务器以非成功状态作答。
    Api {
        /// HTTP 状态码。
        status: u16,
        /// 响应体，或从中提取的 `error.message`。
        message: String,
        /// 服务器指明时应等待多久（可重试状态如 503 常带 `Retry-After`）。
        retry_after: Option<Duration>,
    },

    /// 凭据被拒绝；重试无济于事。
    Auth {
        /// 服务器对该密钥的说法。
        message: String,
    },

    /// 提供方要求我们放慢速度。
    RateLimited {
        /// 服务器指明时应等待多久。
        retry_after: Option<Duration>,
    },

    /// 调用超过了其截止时间。
    Timeout {
        /// 所配置的截止时间。
        elapsed: Duration,
    },

    /// 响应体不符合协议。
    Decode {
        /// 解析失败。
        message: String,
    },

    /// 运行时取消了调用（关停、停止令牌、用户中断）。
    Cancelled,

    /// 提供方未实现此操作。
    Unsupported {
        /// 哪个提供方拒绝了。
        provider: String,
        /// 缺失的操作，例如 `streaming`。
        operation: String,
    },

    /// 提供方配置有误。
    Config {
        /// 缺失或无效的内容。
        message: String,
    },

    /// 每次尝试都失败；附上最后一个错误。
    Exhausted {
        /// 进行了多少次尝试。
        attempts: u32,
        /// 最终的底层失败。
        last: Box<LlmError>,
    },
}

impl LlmError {
    /// 包装一条传输层消息。
    pub fn transport(message: impl Display) -> Self {
        Self::Transport {
            message: message.to_string(),
        }
    }

    /// 包装一条协议层消息。
    pub fn decode(message: impl Display) -> Self {
        Self::Decode {
            message: message.to_string(),
        }
    }

    /// 包装一条配置消息。
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// 非成功状态连同服务器的消息。
    pub fn api(status: u16, message: impl Into<String>) -> Self {
        Self::Api {
            status,
            message: message.into(),
            retry_after: None,
        }
    }

    /// 指明提供方及其缺失的操作。
    pub fn unsupported(provider: impl Into<String>, operation: impl Into<String>) -> Self {
        Self::Unsupported {
            provider: provider.into(),
            operation: operation.into(),
        }
    }

    /// 包装 `attempts` 次尝试后的最终失败。
    pub fn exhausted(attempts: u32, last: LlmError) -> Self {
        Self::Exhausted {
            attempts,
            last: Box::new(last),
        }
    }

    /// 重试循环是否应再次调用提供方？
    ///
    /// 瞬时的传输、超时、429 与大多数 5xx 应答值得再试一次；
    /// 鉴权、解码、取消与配置错误则不值得。
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Transport { .. } | Self::Timeout { .. } | Self::RateLimited { .. } => true,
            Self::Api { status, .. } => {
                matches!(status, 408 | 409 | 425 | 429) || (500..=599).contains(status)
            }
            Self::Exhausted { last, .. } => last.is_retryable(),
            Self::Cancelled
            | Self::Auth { .. }
            | Self::Decode { .. }
            | Self::Unsupported { .. }
            | Self::Config { .. } => false,
        }
    }

    /// 提供方要求等待多久（若它指明了）。
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after } | Self::Api { retry_after, .. } => *retry_after,
            Self::Exhausted { last, .. } => last.retry_after(),
            _ => None,
        }
    }

    /// 错误类别，用作低基数指标标签。
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Transport { .. } => "transport",
            Self::Api { .. } => "api",
            Self::Auth { .. } => "auth",
            Self::RateLimited { .. } => "rate_limited",
            Self::Timeout { .. } => "timeout",
            Self::Decode { .. } => "decode",
            Self::Cancelled => "cancelled",
            Self::Unsupported { .. } => "unsupported",
            Self::Config { .. } => "config",
            Self::Exhausted { .. } => "exhausted",
        }
    }

    /// 最内层的错误，解开任何 [`LlmError::Exhausted`] 包装。
    pub fn root(&self) -> &Self {
        match self {
            Self::Exhausted { last, .. } => last.root(),
            other => other,
        }
    }
}

/// 防止提供方错误体吞没整行日志。
fn clipped(message: &str) -> String {
    const LIMIT: usize = 240;
    let Some((end, _)) = message.char_indices().nth(LIMIT) else {
        return message.to_string();
    };
    let mut out = String::from(&message[..end]);
    out.push('…');
    out
}

impl Display for LlmError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport { message } => write!(f, "transport failure: {message}"),
            Self::Api {
                status, message, ..
            } => {
                write!(f, "provider returned {status}: {}", clipped(message))
            }
            Self::Auth { message } => write!(f, "authentication failed: {message}"),
            Self::RateLimited { retry_after } => match retry_after {
                Some(wait) => write!(f, "rate limited, retry after {wait:?}"),
                None => write!(f, "rate limited"),
            },
            Self::Timeout { elapsed } => write!(f, "timed out after {elapsed:?}"),
            Self::Decode { message } => write!(f, "malformed response: {message}"),
            Self::Cancelled => write!(f, "generation cancelled"),
            Self::Unsupported {
                provider,
                operation,
            } => write!(f, "{provider} does not support {operation}"),
            Self::Config { message } => write!(f, "invalid provider configuration: {message}"),
            Self::Exhausted { attempts, last } => {
                write!(f, "gave up after {attempts} attempts: {last}")
            }
        }
    }
}

impl std::error::Error for LlmError {}

impl From<LlmError> for std::io::Error {
    fn from(error: LlmError) -> Self {
        std::io::Error::other(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_policy_matches_status_codes() {
        for status in [408u16, 409, 425, 429, 500, 502, 503, 504] {
            assert!(
                LlmError::api(status, "nope").is_retryable(),
                "status {status} should be retried"
            );
        }
        for status in [400u16, 401, 403, 404, 422, 451] {
            assert!(
                !LlmError::api(status, "nope").is_retryable(),
                "status {status} must not be retried"
            );
        }
    }

    #[test]
    fn permanent_failures_are_not_retried() {
        for error in [
            LlmError::Cancelled,
            LlmError::Auth {
                message: "bad key".into(),
            },
            LlmError::decode("not json"),
            LlmError::unsupported("mock", "streaming"),
            LlmError::config("missing model"),
        ] {
            assert!(!error.is_retryable(), "{error} must not be retried");
        }
    }

    #[test]
    fn exhaustion_reports_the_attempt_count_and_inner_class() {
        let error = LlmError::exhausted(
            3,
            LlmError::Timeout {
                elapsed: Duration::from_secs(1),
            },
        );
        assert!(error.is_retryable(), "the inner timeout is transient");
        assert_eq!(error.kind(), "exhausted");
        assert_eq!(error.root().kind(), "timeout");
        assert_eq!(
            error.to_string(),
            "gave up after 3 attempts: timed out after 1s"
        );
        assert!(!LlmError::exhausted(2, LlmError::Cancelled).is_retryable());
    }

    #[test]
    fn retry_after_survives_wrapping() {
        let wait = Duration::from_millis(250);
        let error = LlmError::exhausted(
            2,
            LlmError::RateLimited {
                retry_after: Some(wait),
            },
        );
        assert_eq!(error.retry_after(), Some(wait));
        assert_eq!(LlmError::Cancelled.retry_after(), None);
        assert_eq!(
            LlmError::RateLimited { retry_after: None }.to_string(),
            "rate limited"
        );
    }

    #[test]
    fn long_api_messages_are_trimmed_in_display() {
        let error = LlmError::api(500, "x".repeat(2_000));
        let text = error.to_string();
        assert!(text.chars().count() < 300, "display stays readable");
        assert!(text.ends_with('…'));
        let LlmError::Api { message, .. } = &error else {
            panic!("expected an api error");
        };
        assert_eq!(
            message.len(),
            2_000,
            "clipping is display-only; the raw body stays available"
        );
    }

    #[test]
    fn errors_are_std_errors() {
        let boxed: Box<dyn std::error::Error + Send + Sync> =
            Box::new(LlmError::transport("connection refused"));
        assert!(boxed.to_string().contains("connection refused"));
    }
}
