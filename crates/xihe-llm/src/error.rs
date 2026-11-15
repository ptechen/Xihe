//! Failures a provider can return, and whether the runtime should retry them.

use std::fmt::{self, Display, Formatter};
use std::time::Duration;

/// Everything that can go wrong between us and a model.
///
/// The variants split by *disposition*, not by HTTP detail: retry, stop, or
/// reconfigure. [`LlmError::is_retryable`] is what [`crate::ChatClient`] trusts.
#[derive(Clone, Debug, PartialEq)]
pub enum LlmError {
    /// The request never produced a usable response (DNS, connect, TLS, reset).
    /// The request never reached the server, or the connection broke.
    Transport {
        /// Human-readable detail from the transport layer.
        message: String,
    },

    /// The server answered with a non-success status.
    Api {
        /// HTTP status code.
        status: u16,
        /// The body, or the `error.message` extracted from it.
        message: String,
    },

    /// Credentials were rejected; retrying cannot help.
    Auth {
        /// What the server said about the key.
        message: String,
    },

    /// The provider asked us to slow down.
    RateLimited {
        /// How long to wait, when the server says.
        retry_after: Option<Duration>,
    },

    /// The call exceeded its deadline.
    Timeout {
        /// The deadline that was configured.
        elapsed: Duration,
    },

    /// The body did not match the protocol.
    Decode {
        /// The parse failure.
        message: String,
    },

    /// The runtime cancelled the call (shutdown, stop token, user interrupt).
    Cancelled,

    /// The provider does not implement this operation.
    Unsupported {
        /// Which provider said no.
        provider: String,
        /// The operation that is missing, e.g. `streaming`.
        operation: String,
    },

    /// The provider was misconfigured.
    Config {
        /// What is missing or invalid.
        message: String,
    },

    /// Every attempt failed; the last error is attached.
    Exhausted {
        /// How many attempts were made.
        attempts: u32,
        /// The final underlying failure.
        last: Box<LlmError>,
    },
}

impl LlmError {
    /// Wrap a transport-level message.
    pub fn transport(message: impl Display) -> Self {
        Self::Transport {
            message: message.to_string(),
        }
    }

    /// Wrap a protocol-level message.
    pub fn decode(message: impl Display) -> Self {
        Self::Decode {
            message: message.to_string(),
        }
    }

    /// Wrap a configuration message.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// A non-success status together with the server's message.
    pub fn api(status: u16, message: impl Into<String>) -> Self {
        Self::Api {
            status,
            message: message.into(),
        }
    }

    /// Name the provider and the operation it lacks.
    pub fn unsupported(provider: impl Into<String>, operation: impl Into<String>) -> Self {
        Self::Unsupported {
            provider: provider.into(),
            operation: operation.into(),
        }
    }

    /// Wrap the final failure after `attempts` tries.
    pub fn exhausted(attempts: u32, last: LlmError) -> Self {
        Self::Exhausted {
            attempts,
            last: Box::new(last),
        }
    }

    /// Should the retry loop call the provider again?
    ///
    /// Transient transport, timeout, 429 and most 5xx answers are worth another
    /// try; auth, decode, cancellation and configuration errors are not.
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

    /// How long the provider asked to wait, if it said.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after } => *retry_after,
            Self::Exhausted { last, .. } => last.retry_after(),
            _ => None,
        }
    }

    /// The error class, used as a low-cardinality metric label.
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

    /// The innermost error, unwrapping any [`LlmError::Exhausted`] wrapper.
    pub fn root(&self) -> &Self {
        match self {
            Self::Exhausted { last, .. } => last.root(),
            other => other,
        }
    }
}

/// Keeps provider error bodies from swallowing the log line.
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
            Self::Api { status, message } => {
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
        let error = LlmError::exhausted(3, LlmError::Timeout {
            elapsed: Duration::from_secs(1),
        });
        assert!(error.is_retryable(), "the inner timeout is transient");
        assert_eq!(error.kind(), "exhausted");
        assert_eq!(error.root().kind(), "timeout");
        assert_eq!(error.to_string(), "gave up after 3 attempts: timed out after 1s");
        assert!(!LlmError::exhausted(2, LlmError::Cancelled).is_retryable());
    }

    #[test]
    fn retry_after_survives_wrapping() {
        let wait = Duration::from_millis(250);
        let error = LlmError::exhausted(2, LlmError::RateLimited {
            retry_after: Some(wait),
        });
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
