//! Failures specific to memory: embedding, storage and persistence.

use std::fmt::{self, Display, Formatter};

/// What can go wrong while remembering.
#[derive(Clone, Debug, PartialEq)]
pub enum MemoryError {
    /// The embedder could not produce a vector.
    Embedding {
        /// Why embedding failed.
        message: String,
    },
    /// A vector did not match the store's configured dimensionality.
    DimensionMismatch {
        /// Width the store expects.
        expected: usize,
        /// Width the caller supplied.
        found: usize,
    },
    /// The store could not be written or read.
    Storage {
        /// What the storage layer reported.
        message: String,
    },
    /// A persistence file was unreadable or malformed.
    Persistence {
        /// Path or line the operation choked on.
        message: String,
    },
    /// A caller-supplied budget or setting cannot be used as given.
    Configuration {
        /// What was rejected.
        message: String,
    },
    /// The feature is behind a cargo feature that is not enabled.
    Disabled {
        /// Name of the missing cargo feature.
        feature: &'static str,
    },
}

impl MemoryError {
    /// Wrap an embedding failure.
    pub fn embedding(message: impl Display) -> Self {
        Self::Embedding {
            message: message.to_string(),
        }
    }

    /// A vector of the wrong width reached a store of `expected` dimensions.
    pub fn dimension_mismatch(expected: usize, found: usize) -> Self {
        Self::DimensionMismatch { expected, found }
    }

    /// Wrap a storage failure.
    pub fn storage(message: impl Display) -> Self {
        Self::Storage {
            message: message.to_string(),
        }
    }

    /// Wrap a persistence failure.
    pub fn persistence(message: impl Display) -> Self {
        Self::Persistence {
            message: message.to_string(),
        }
    }

    /// Reject a budget or setting that would make memory behave badly.
    pub fn config(message: impl Display) -> Self {
        Self::Configuration {
            message: message.to_string(),
        }
    }

    /// Say which cargo feature the caller needs.
    pub const fn disabled(feature: &'static str) -> Self {
        Self::Disabled { feature }
    }
}

impl Display for MemoryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embedding { message } => write!(f, "embedding failed: {message}"),
            Self::DimensionMismatch { expected, found } => write!(
                f,
                "embedding has {found} dimensions, the store expects {expected}"
            ),
            Self::Storage { message } => write!(f, "memory storage failed: {message}"),
            Self::Persistence { message } => write!(f, "memory persistence failed: {message}"),
            Self::Configuration { message } => {
                write!(f, "invalid memory configuration: {message}")
            }
            Self::Disabled { feature } => {
                write!(f, "feature `{feature}` is not enabled in xihe-memory")
            }
        }
    }
}

impl std::error::Error for MemoryError {}

impl From<std::io::Error> for MemoryError {
    fn from(error: std::io::Error) -> Self {
        Self::Storage {
            message: error.to_string(),
        }
    }
}

impl From<serde_json::Error> for MemoryError {
    fn from(error: serde_json::Error) -> Self {
        Self::Persistence {
            message: error.to_string(),
        }
    }
}

/// The result type used across this crate.
pub type MemoryResult<T> = Result<T, MemoryError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_mismatch_names_both_sides() {
        let error = MemoryError::dimension_mismatch(64, 128);
        assert_eq!(
            error.to_string(),
            "embedding has 128 dimensions, the store expects 64"
        );
        assert_eq!(
            MemoryError::config("the focus budget must be at least 1").to_string(),
            "invalid memory configuration: the focus budget must be at least 1"
        );
    }

    #[test]
    fn io_and_json_errors_are_classified() {
        let io = MemoryError::from(std::io::Error::other("disk full"));
        assert_eq!(io.to_string(), "memory storage failed: disk full");
        let json = MemoryError::from(serde_json::from_str::<u32>("not a number").unwrap_err());
        assert!(json.to_string().starts_with("memory persistence failed:"));
    }

    #[test]
    fn disabled_features_say_the_feature_name() {
        assert_eq!(
            MemoryError::disabled("persist").to_string(),
            "feature `persist` is not enabled in xihe-memory"
        );
    }

    #[test]
    fn errors_are_std_errors() {
        let boxed: Box<dyn std::error::Error + Send + Sync> =
            Box::new(MemoryError::embedding("model unavailable"));
        assert!(boxed.to_string().contains("model unavailable"));
    }
}
