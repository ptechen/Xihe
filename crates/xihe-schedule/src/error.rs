//! Errors produced while scheduling and running systems.

use std::fmt;

/// A system reported a failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemError {
    /// Name of the system that failed.
    pub name: String,
    /// Short diagnostic written by the system.
    pub reason: String,
    /// `true` when the failure is worth retrying on the next frame.
    pub retryable: bool,
}

impl SystemError {
    /// Builds a failure.
    pub fn new(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
            retryable: false,
        }
    }

    /// Builds a failure the driver may retry.
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

/// A schedule that cannot be built or run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    /// Two systems share a label, so ordering would be ambiguous.
    DuplicateLabel {
        /// The offending label.
        label: String,
    },
    /// A parallel stage contains systems whose declared access overlaps.
    Conflict {
        /// Name of the parallel stage.
        stage: String,
        /// First system.
        left: String,
        /// Second system.
        right: String,
    },
    /// An ordering hint refers to a label that is not scheduled.
    UnknownLabel {
        /// The missing label.
        label: String,
    },
    /// A stage has no systems.
    EmptyStage {
        /// Name or index of the empty stage.
        stage: String,
    },
    /// A system failed during execution.
    Failed {
        /// Underlying failure.
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
