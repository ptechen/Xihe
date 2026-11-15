//! 记忆特有的失败：嵌入、存储与持久化。

use std::fmt::{self, Display, Formatter};

/// 记忆过程中可能出错的情形。
#[derive(Clone, Debug, PartialEq)]
pub enum MemoryError {
    /// 嵌入器无法产出向量。
    Embedding {
        /// 嵌入失败的原因。
        message: String,
    },
    /// 向量与存储配置的维度不匹配。
    DimensionMismatch {
        /// 存储期望的宽度。
        expected: usize,
        /// 调用者提供的宽度。
        found: usize,
    },
    /// 存储无法写入或读取。
    Storage {
        /// 存储层报告的内容。
        message: String,
    },
    /// 持久化文件不可读或格式错误。
    Persistence {
        /// 操作卡住的路径或行。
        message: String,
    },
    /// 调用者提供的预算或设置无法按原样使用。
    Configuration {
        /// 被拒绝的内容。
        message: String,
    },
    /// 该功能位于一个未启用的 cargo feature 之后。
    Disabled {
        /// 缺失的 cargo feature 名。
        feature: &'static str,
    },
}

impl MemoryError {
    /// 包装一个嵌入失败。
    pub fn embedding(message: impl Display) -> Self {
        Self::Embedding {
            message: message.to_string(),
        }
    }

    /// 宽度错误的向量到达了一个 `expected` 维的存储。
    pub fn dimension_mismatch(expected: usize, found: usize) -> Self {
        Self::DimensionMismatch { expected, found }
    }

    /// 包装一个存储失败。
    pub fn storage(message: impl Display) -> Self {
        Self::Storage {
            message: message.to_string(),
        }
    }

    /// 包装一个持久化失败。
    pub fn persistence(message: impl Display) -> Self {
        Self::Persistence {
            message: message.to_string(),
        }
    }

    /// 拒绝会使记忆行为异常的预算或设置。
    pub fn config(message: impl Display) -> Self {
        Self::Configuration {
            message: message.to_string(),
        }
    }

    /// 说明调用者需要哪个 cargo feature。
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

/// 本 crate 各处使用的结果类型。
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
