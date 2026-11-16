//! 让人类说“没有我不行”。
//!
//! 有些工具绝不应凭模型一句话就运行：写入生产、花钱、删除数据。
//! [`ToolRegistry::requires_approval`] 标记它们，随后 ReAct 循环
//! 在调用通过前需要来自 [`Approver`] 的一个 [`Decision`]。
//! 未配置时，循环不猜测——它把智能体挂起在
//! [`AgentStatus::AwaitingApproval`](crate::agent::AgentStatus::AwaitingApproval)，
//! 请求存于智能体上，因此驱动世界的任何东西（CLI、网页表单、
//! 队列）都能稍后作答。
//!
//! 该 trait 总是可用；只有阻塞于 stdin 的 [`TerminalApprover`]
//! 需要 `hitl` feature。

use std::fmt::{self, Display, Formatter};

use crate::ecs_core::Entity;
use crate::agent_llm::ToolCall;
use async_trait::async_trait;
use serde_json::Value;

/// 一个正在等待人的工具调用。
///
/// 它携带整个调用，而非仅其名字，使审查者能看到模型选择的
/// 参数；它还记得排在其后的调用——批准第一个不会静默批准其余。
#[derive(Clone, Debug, PartialEq)]
pub struct ApprovalRequest {
    /// 想要运行该工具的智能体。
    pub agent: Entity,
    /// 发起请求的轮。
    pub turn: u64,
    /// 调用本身。
    pub call: ToolCall,
    /// 工具为何需要人，来自工具本身。
    pub reason: String,
    /// 模型在此之后请求、尚未运行的调用。
    pub remaining: Vec<ToolCall>,
}

impl ApprovalRequest {
    /// 默认原因，当工具未给出时。
    pub const DEFAULT_REASON: &'static str = "the tool asked for a human before running";

    /// 为 `call` 构建一个请求，`remaining` 排在其后。
    pub fn new(agent: Entity, turn: u64, call: &ToolCall, remaining: Vec<ToolCall>) -> Self {
        Self {
            agent,
            turn,
            call: call.clone(),
            reason: Self::DEFAULT_REASON.to_string(),
            remaining,
        }
    }

    /// 解释为何需要人。
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// 将被运行的工具的名字。
    pub fn tool(&self) -> &str {
        &self.call.name
    }

    /// 模型产出的参数。
    pub fn arguments(&self) -> &Value {
        &self.call.arguments
    }

    /// 可安全显示在日志或提示词中的一行摘要。
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

/// 审查者决定了什么。
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Decision {
    /// 运行该工具。
    Approved,
    /// 不运行它，附上模型将看到的原因。
    Rejected {
        /// 人为何说不。
        reason: String,
    },
}

impl Decision {
    /// 附上 `reason` 的拒绝。
    pub fn reject(reason: impl Into<String>) -> Self {
        Self::Rejected {
            reason: reason.into(),
        }
    }

    /// 该工具是否可运行。
    pub fn is_approved(&self) -> bool {
        matches!(self, Self::Approved)
    }

    /// 拒绝原因（若有）。
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Approved => None,
            Self::Rejected { reason } => Some(reason),
        }
    }

    /// 当决定不是批准时，告知模型的内容。
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

/// 能回答 [`ApprovalRequest`] 之物。
#[async_trait]
pub trait Approver: Send + Sync {
    /// 决定 `request` 是否可运行。
    async fn decide(&self, request: &ApprovalRequest) -> Decision;

    /// 日志中使用的简短名字。
    fn name(&self) -> &str {
        "approver"
    }
}

/// 批准一切：用于测试，以及只是*看起来*危险的工具。
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

/// 拒绝一切，带一个固定原因。
pub struct AlwaysReject {
    /// 交回给模型的原因。
    pub reason: String,
}

impl AlwaysReject {
    /// 以 `reason` 拒绝。
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

/// 批准谓词接受的调用。
///
/// 谓词看到的是请求，而非世界，因此像“只在限额下花费”这样的规则
/// 无需写一整个审批器即可表达。
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
    /// 批准 `rule` 所批准的。
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

/// 从终端读取一个 yes/no 回答。
///
/// 阻塞的 stdin 被移到一个阻塞线程，使异步循环保持其他任务运行。
/// 除裸 `y` 之外的任何输入都是拒绝，被拒绝的那行成为模型看到的原因——
/// 一个人输入“不，改用 staging 主机”本身就是信息。
#[cfg(feature = "hitl")]
pub struct TerminalApprover {
    /// 读取前打印的问题。
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
    /// 用自定义提示词询问。
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
        let approver = RuleApprover::new("non-prod", |request| {
            match request.arguments().get("target").and_then(Value::as_str) {
                Some("prod") => Decision::reject("prod needs a person"),
                _ => Decision::Approved,
            }
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
