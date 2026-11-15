//! Letting a human say "not without me".
//!
//! Some tools should never run on a model's say-so: writing to production,
//! spending money, deleting data. [`ToolRegistry::requires_approval`] marks
//! them, and the ReAct loop then needs a [`Decision`] from an [`Approver`]
//! before the call goes through. Without one configured, the loop does not
//! guess — it parks the agent in
//! [`AgentStatus::AwaitingApproval`](crate::AgentStatus::AwaitingApproval) with
//! the request stored on the agent, so whatever drives the world (a CLI, a web
//! form, a queue) can answer later.
//!
//! The trait is always available; only [`TerminalApprover`], which blocks on
//! stdin, needs the `hitl` feature.

use std::fmt::{self, Display, Formatter};

use async_trait::async_trait;
use serde_json::Value;
use xihe_core::Entity;
use xihe_llm::ToolCall;

/// A tool call that is waiting for a person.
///
/// It carries the whole call, not just its name, so a reviewer can see the
/// arguments the model chose, and it remembers the calls queued behind it —
/// approving the first one does not silently approve the rest.
#[derive(Clone, Debug, PartialEq)]
pub struct ApprovalRequest {
    /// The agent that wants to run the tool.
    pub agent: Entity,
    /// Turn the request was raised on.
    pub turn: u64,
    /// The call itself.
    pub call: ToolCall,
    /// Why the tool needs a human, from the tool itself.
    pub reason: String,
    /// Calls the model asked for after this one, not yet run.
    pub remaining: Vec<ToolCall>,
}

impl ApprovalRequest {
    /// The default reason, when the tool did not give one.
    pub const DEFAULT_REASON: &'static str = "the tool asked for a human before running";

    /// Build a request for `call`, with `remaining` queued behind it.
    pub fn new(agent: Entity, turn: u64, call: &ToolCall, remaining: Vec<ToolCall>) -> Self {
        Self {
            agent,
            turn,
            call: call.clone(),
            reason: Self::DEFAULT_REASON.to_string(),
            remaining,
        }
    }

    /// Explain why a human is needed.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// Name of the tool to be run.
    pub fn tool(&self) -> &str {
        &self.call.name
    }

    /// Arguments the model produced.
    pub fn arguments(&self) -> &Value {
        &self.call.arguments
    }

    /// A one-line summary safe to show in a log or a prompt.
    pub fn summary(&self) -> String {
        let arguments = self.call.arguments.to_string();
        let cut: String = arguments.chars().take(120).collect();
        let queued = if self.remaining.is_empty() {
            String::new()
        } else {
            format!(", {} more behind it", self.remaining.len())
        };
        format!("{}({cut}) — {}{queued}", self.call.name, self.reason)
    }
}

impl Display for ApprovalRequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "agent {:?} on turn {}: {}",
            self.agent,
            self.turn,
            self.summary()
        )
    }
}

/// What a reviewer decided.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Decision {
    /// Run the tool.
    Approved,
    /// Do not run it, with the reason the model gets to see.
    Rejected {
        /// Why a human said no.
        reason: String,
    },
}

impl Decision {
    /// A rejection with `reason` attached.
    pub fn reject(reason: impl Into<String>) -> Self {
        Self::Rejected {
            reason: reason.into(),
        }
    }

    /// Whether the tool may run.
    pub fn is_approved(&self) -> bool {
        matches!(self, Self::Approved)
    }

    /// The rejection reason, if any.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Approved => None,
            Self::Rejected { reason } => Some(reason),
        }
    }

    /// What the model is told when the decision is not approval.
    pub fn transcript_text(&self) -> String {
        match self {
            Self::Approved => "approved".to_string(),
            Self::Rejected { reason } => format!("a human refused to run this tool: {reason}"),
        }
    }
}

impl Display for Decision {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Approved => write!(f, "approved"),
            Self::Rejected { reason } => write!(f, "rejected: {reason}"),
        }
    }
}

/// Something that can answer an [`ApprovalRequest`].
#[async_trait]
pub trait Approver: Send + Sync {
    /// Decide whether `request` may run.
    async fn decide(&self, request: &ApprovalRequest) -> Decision;

    /// Short name used in logs.
    fn name(&self) -> &str {
        "approver"
    }
}

/// Approve everything: for tests and for tools that only *look* dangerous.
pub struct AlwaysApprove;

#[async_trait]
impl Approver for AlwaysApprove {
    async fn decide(&self, _request: &ApprovalRequest) -> Decision {
        Decision::Approved
    }

    fn name(&self) -> &str {
        "always-approve"
    }
}

/// Reject everything, with a fixed reason.
pub struct AlwaysReject {
    /// The reason handed back to the model.
    pub reason: String,
}

impl AlwaysReject {
    /// Reject with `reason`.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

#[async_trait]
impl Approver for AlwaysReject {
    async fn decide(&self, _request: &ApprovalRequest) -> Decision {
        Decision::reject(&self.reason)
    }

    fn name(&self) -> &str {
        "always-reject"
    }
}

/// Approve the calls a predicate accepts.
///
/// The predicate sees the request, not the world, so a rule like "only spend
/// under a limit" is expressible without writing a whole approver.
pub struct RuleApprover<F>
where
    F: Fn(&ApprovalRequest) -> Decision + Send + Sync,
{
    rule: F,
    label: String,
}

impl<F> RuleApprover<F>
where
    F: Fn(&ApprovalRequest) -> Decision + Send + Sync,
{
    /// Approve what `rule` approves.
    pub fn new(label: impl Into<String>, rule: F) -> Self {
        Self {
            rule,
            label: label.into(),
        }
    }
}

#[async_trait]
impl<F> Approver for RuleApprover<F>
where
    F: Fn(&ApprovalRequest) -> Decision + Send + Sync,
{
    async fn decide(&self, request: &ApprovalRequest) -> Decision {
        (self.rule)(request)
    }

    fn name(&self) -> &str {
        &self.label
    }
}

/// Read a yes/no answer from the terminal.
///
/// Blocking stdin is moved to a blocking thread so the async loop keeps its
/// other tasks running. Anything other than a bare `y` is a rejection, and the
/// rejected line becomes the reason the model sees — a human typing "no, use
/// the staging host instead" is information.
#[cfg(feature = "hitl")]
pub struct TerminalApprover {
    /// Question printed before reading.
    pub prompt: String,
}

#[cfg(feature = "hitl")]
impl Default for TerminalApprover {
    fn default() -> Self {
        Self {
            prompt: "run this tool? [y/N] ".to_string(),
        }
    }
}

#[cfg(feature = "hitl")]
impl TerminalApprover {
    /// Ask with a custom prompt.
    pub fn with_prompt(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
        }
    }
}

#[cfg(feature = "hitl")]
#[async_trait]
impl Approver for TerminalApprover {
    async fn decide(&self, request: &ApprovalRequest) -> Decision {
        use std::io::{BufRead, Write};

        let prompt = self.prompt.clone();
        let summary = request.summary();
        let answer = tokio::task::spawn_blocking(move || {
            let mut line = String::new();
            let stdin = std::io::stdin();
            let mut out = std::io::stdout();
            let _ = writeln!(out, "{summary}");
            let _ = out.flush();
            let _ = write!(out, "{prompt}");
            let _ = out.flush();
            let read = stdin.lock().read_line(&mut line);
            (read.is_ok(), line)
        })
        .await;
        match answer {
            Ok((true, line)) => {
                let answer = line.trim().to_lowercase();
                if answer == "y" || answer == "yes" {
                    Decision::Approved
                } else if answer.is_empty() {
                    Decision::reject("no answer given")
                } else {
                    Decision::reject(format!("the reviewer wrote: {}", line.trim()))
                }
            }
            _ => Decision::reject("the terminal was not readable"),
        }
    }

    fn name(&self) -> &str {
        "terminal"
    }
}

impl fmt::Debug for dyn Approver + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Approver({})", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> ApprovalRequest {
        ApprovalRequest::new(
            Entity::from_raw(1, 1),
            3,
            &ToolCall::new("c1", "deploy", json!({"target": "prod"})),
            Vec::new(),
        )
        .with_reason("writing to production")
    }

    #[tokio::test]
    async fn decisions_carry_their_reason() {
        assert!(Decision::Approved.is_approved());
        let rejected = Decision::reject("not today");
        assert!(!rejected.is_approved());
        assert_eq!(rejected.reason(), Some("not today"));
        assert!(rejected.transcript_text().contains("not today"));
        assert_eq!(rejected.to_string(), "rejected: not today");
    }

    #[tokio::test]
    async fn the_two_policies_are_opposites() {
        assert!(AlwaysApprove.decide(&request()).await.is_approved());
        let rejector = AlwaysReject::new("frozen");
        let decision = rejector.decide(&request()).await;
        assert_eq!(decision, Decision::reject("frozen"));
    }

    #[tokio::test]
    async fn a_rule_can_approve_by_argument() {
        let approver = RuleApprover::new("non-prod", |request| match request
            .arguments()
            .get("target")
            .and_then(Value::as_str)
        {
            Some("prod") => Decision::reject("prod needs a person"),
            _ => Decision::Approved,
        });
        assert_eq!(
            approver.decide(&request()).await,
            Decision::reject("prod needs a person")
        );
        let mut staging = request();
        staging.call.arguments = json!({"target": "staging"});
        assert!(approver.decide(&staging).await.is_approved());
    }

    #[test]
    fn a_summary_is_short_and_names_the_tool() {
        let summary = request().summary();
        assert_eq!(request().tool(), "deploy");
        assert!(summary.starts_with("deploy("), "{summary}");
        assert!(summary.contains("writing to production"), "{summary}");
        assert!(summary.len() < 200, "{summary}");
    }

    #[test]
    fn long_arguments_are_cut() {
        let mut request = request();
        request.call.arguments = json!({"blob": "x".repeat(500)});
        assert!(request.summary().chars().count() < 200);
    }
    #[test]
    fn a_dyn_approver_prints_its_name() {
        let approver: &dyn Approver = &AlwaysApprove;
        assert_eq!(format!("{approver:?}"), "Approver(always-approve)");
    }

    #[test]
    fn queued_calls_show_up_in_the_summary() {
        let second = ToolCall::new("c2", "notify", json!({}));
        let request = ApprovalRequest::new(
            Entity::from_raw(2, 1),
            1,
            &ToolCall::new("c1", "deploy", json!({"target": "prod"})),
            vec![second],
        );
        assert_eq!(request.reason, ApprovalRequest::DEFAULT_REASON);
        assert!(request.summary().contains("1 more behind it"));
        assert_eq!(request.arguments(), &json!({"target": "prod"}));
    }
}
