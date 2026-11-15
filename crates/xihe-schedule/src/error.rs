//! 调度与运行系统时产生的错误。

use std::fmt;

/// 某个系统报告了一次失败。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemError {
    /// 失败系统的名称。
    pub name: String,
    /// 系统写下的简短诊断。
    pub reason: String,
    /// 该失败值得在下一帧重试时为 `true`。
    pub retryable: bool,
}

impl SystemError {
    /// 构建一次失败。
    pub fn new(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
            retryable: false,
        }
    }

    /// 构建一次驱动器可重试的失败。
    pub fn retryable(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
            retryable: true,
        }
    }
}

impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "system `{}` failed{}",
            self.name,
            if self.reason.is_empty() {
                String::new()
            } else {
                format!(": {}", self.reason)
            }
        )?;
        if self.retryable {
            write!(f, " (retryable)")?;
        }
        Ok(())
    }
}

impl std::error::Error for SystemError {}

impl From<String> for SystemError {
    fn from(reason: String) -> Self {
        Self::new("unnamed", reason)
    }
}

impl From<&str> for SystemError {
    fn from(reason: &str) -> Self {
        Self::new("unnamed", reason.to_string())
    }
}

/// 无法构建或运行的调度器。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    /// 两个系统共用一个标签，排序会有歧义。
    DuplicateLabel {
        /// 引发问题的标签。
        label: String,
    },
    /// 一个并行阶段包含声明访问权限相互重叠的系统。
    Conflict {
        /// 并行阶段的名称。
        stage: String,
        /// 第一个系统。
        left: String,
        /// 第二个系统。
        right: String,
    },
    /// 排序提示引用了一个未被调度的标签。
    UnknownLabel {
        /// 缺失的标签。
        label: String,
    },
    /// 某阶段没有系统。
    EmptyStage {
        /// 空阶段的名称或索引。
        stage: String,
    },
    /// 某系统在执行期间失败。
    Failed {
        /// 底层失败。
        error: SystemError,
    },
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateLabel { label } => write!(f, "duplicate system label `{label}`"),
            Self::Conflict {
                stage,
                left,
                right,
            } => write!(
                f,
                "parallel stage `{stage}` runs `{left}` and `{right}` together, but their declared world access conflicts"
            ),
            Self::UnknownLabel { label } => write!(f, "no scheduled system named `{label}`"),
            Self::EmptyStage { stage } => write!(f, "stage `{stage}` has no systems"),
            Self::Failed { error } => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ScheduleError {}

impl From<SystemError> for ScheduleError {
    fn from(error: SystemError) -> Self {
        Self::Failed { error }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_covers_reason_and_retry() {
        let error = SystemError::retryable("planner", "provider timeout");
        assert_eq!(error.to_string(), "system `planner` failed: provider timeout (retryable)");
        let plain: SystemError = "boom".into();
        assert_eq!(plain.to_string(), "system `unnamed` failed: boom");
    }

    #[test]
    fn schedule_errors_render() {
        let error = ScheduleError::Conflict {
            stage: "perceive".into(),
            left: "a".into(),
            right: "b".into(),
        };
        assert!(error.to_string().contains("conflicts"));
        assert!(matches!(
            ScheduleError::from(SystemError::new("x", "y")),
            ScheduleError::Failed { .. }
        ));
    }
}
