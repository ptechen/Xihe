//! 在循环花费超过其价值之前停止它的限额。
//!
//! [`Guardrails`] 值是声明式的：它说明一个智能体可花费什么，
//! 而 [`Guardrails::check`] 说明给定的 [`Spend`] 是否仍然适配。
//! ReAct 系统在每次模型调用与每次工具调用前询问，因此一个
//! 失控的智能体会以 [`AgentError::Budget`] 失败，而非永远烧 token。
//!
//! ```
//! use xihe::agent::{Guardrails, Spend};
//!
//! let limits = Guardrails::new().with_turns(3).with_token_budget(1_000);
//! let mut spend = Spend::default();
//! assert!(limits.check_turns(&spend).is_ok());
//! spend.turns = 3;
//! assert!(limits.check_turns(&spend).is_err());
//! ```

use std::fmt::{self, Display, Formatter};
use std::time::Duration;

use crate::agent_llm::{GenerationResponse, ToolDescriptor};

use crate::agent::AgentError;

/// 一个智能体迄今为止在其当前任务上花费了什么。
///
/// 计数器只增不减。[`crate::agent::ReactAgent`] 在新任务开始时把它们清零，
/// 因此预算总是“针对本任务”，而非“针对本进程”。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Spend {
    /// 已做的模型调用。
    pub turns: usize,
    /// 已执行的工具调用。
    pub tool_calls: usize,
    /// 执行前被拒绝的工具调用。
    pub denied: usize,
    /// 计费的输入 token。
    pub input_tokens: u64,
    /// 计费的输出 token。
    pub output_tokens: u64,
}

impl Spend {
    /// 整个任务计费的 token。
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// 计费一次提供方响应。
    pub fn record_response(&mut self, response: &GenerationResponse) {
        self.turns += 1;
        self.input_tokens += response.usage.prompt_tokens;
        self.output_tokens += response.usage.completion_tokens;
    }

    /// 计费一次实际运行的工具调用。
    pub fn record_tool_call(&mut self) {
        self.tool_calls += 1;
    }

    /// 计费一次从未运行的工具调用。
    pub fn record_denial(&mut self) {
        self.denied += 1;
    }
}

/// 单个智能体任务上的上限。
///
/// 默认每个字段为 `0`/`None` 表示“无限额”，唯独 [`Guardrails::max_turns`]
/// 默认为 [`Guardrails::DEFAULT_TURNS`]：失控的循环正是默认值得防范的失败模式。
#[derive(Clone, Eq, PartialEq)]
pub struct Guardrails {
    /// 每任务允许的模型调用。`0` 表示无限。
    pub max_turns: usize,
    /// 每任务允许的工具调用。`0` 表示无限。
    pub max_tool_calls: usize,
    /// 每任务允许的总 token。`0` 表示无限。
    pub max_tokens: u64,
    /// 每任务允许的墙钟秒数。`0` 表示无限。
    pub max_seconds: u64,
    /// 单个工具结果保留在转录中的字符数。`0` 表示
    /// 无限。约束提示词，防止一个工具倾倒巨量载荷。
    pub max_tool_output_chars: usize,
    /// 模型可见可调用的工具；`None` 表示所有已注册工具。
    pub allowed_tools: Option<Vec<String>>,
}

impl Guardrails {
    /// 循环被切断前一个智能体获得的轮数。
    pub const DEFAULT_TURNS: usize = 8;

    /// 循环被切断前一个智能体获得的工具调用数。
    pub const DEFAULT_TOOL_CALLS: usize = 32;

    /// 单个工具结果逐字保留在转录中的字符数。约为默认窗口
    /// token 预算的四分之一，因此单个工具倾倒巨量载荷不会
    /// 撑爆提示词；其余部分带一个标记被丢弃。
    pub const DEFAULT_TOOL_OUTPUT_CHARS: usize = 8_000;

    /// 合理默认：一个轮上限与一个工具上限，无 token 或时间上限。
    pub fn new() -> Self {
        Self {
            max_turns: Self::DEFAULT_TURNS,
            max_tool_calls: Self::DEFAULT_TOOL_CALLS,
            max_tokens: 0,
            max_seconds: 0,
            max_tool_output_chars: Self::DEFAULT_TOOL_OUTPUT_CHARS,
            allowed_tools: None,
        }
    }

    /// 除工具允许列表外，无任何上限。
    pub fn unlimited() -> Self {
        Self {
            max_turns: 0,
            max_tool_calls: 0,
            max_tokens: 0,
            max_seconds: 0,
            max_tool_output_chars: 0,
            allowed_tools: None,
        }
    }

    /// 允许 `turns` 次模型调用，`0` 表示无限。
    pub fn with_turns(mut self, turns: usize) -> Self {
        self.max_turns = turns;
        self
    }

    /// 允许 `calls` 次工具调用，`0` 表示无限。
    pub fn with_tool_calls(mut self, calls: usize) -> Self {
        self.max_tool_calls = calls;
        self
    }

    /// 允许 `tokens` 个输入加输出 token，`0` 表示无限。
    pub fn with_token_budget(mut self, tokens: u64) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// 允许 `limit` 的墙钟时间，零表示无限。
    pub fn with_time_limit(mut self, limit: Duration) -> Self {
        self.max_seconds = limit.as_secs();
        self
    }

    /// 在转录中至多保留任一工具结果的 `chars` 个字符，零表示
    /// 无限。被丢弃的尾部由一个截断标记替换。
    pub fn with_tool_output_chars(mut self, chars: usize) -> Self {
        self.max_tool_output_chars = chars;
        self
    }

    /// 把智能体限制到 `names`，这也对模型隐藏其他工具。
    pub fn allow_tools(self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allowing(names.into_iter().map(Into::into).collect())
    }

    /// 把智能体限制到一个显式的名字列表。
    pub fn allowing(mut self, names: Vec<String>) -> Self {
        self.allowed_tools = Some(names);
        self
    }

    /// `name` 是否可被调用。
    pub fn is_tool_allowed(&self, name: &str) -> bool {
        self.allowed_tools
            .as_ref()
            .is_none_or(|allowed| allowed.iter().any(|allowed| allowed == name))
    }

    /// 丢弃允许列表所隐藏的描述符。
    ///
    /// 模型无法调用它看不到的东西，这比要求它守规矩是更强的保证。
    pub fn filter_tools(&self, tools: Vec<ToolDescriptor>) -> Vec<ToolDescriptor> {
        let Some(allowed) = &self.allowed_tools else {
            return tools;
        };
        tools
            .into_iter()
            .filter(|tool| allowed.iter().any(|name| name == &tool.name))
            .collect()
    }

    /// 这一轮仍适配模型调用上限。
    pub fn check_turns(&self, spend: &Spend) -> Result<(), AgentError> {
        check("max_turns", spend.turns as u64, self.max_turns as u64)
    }

    /// 这一轮仍适配工具调用上限。
    pub fn check_tool_calls(&self, spend: &Spend) -> Result<(), AgentError> {
        check(
            "max_tool_calls",
            spend.tool_calls as u64,
            self.max_tool_calls as u64,
        )
    }

    /// 这一轮仍适配 token 上限。在调用*之前*、针对已花费的量检查：
    /// 提供方在事后才报价。
    pub fn check_tokens(&self, spend: &Spend) -> Result<(), AgentError> {
        check("max_tokens", spend.total_tokens(), self.max_tokens)
    }

    /// 这一轮仍适配墙钟上限。
    pub fn check_time(&self, elapsed: Duration) -> Result<(), AgentError> {
        if self.max_seconds == 0 {
            return Ok(());
        }
        if elapsed.as_secs() < self.max_seconds {
            return Ok(());
        }
        Err(AgentError::budget(
            "max_seconds",
            elapsed.as_secs(),
            self.max_seconds,
        ))
    }

    /// `name` 在允许列表上。
    pub fn check_tool(&self, name: &str) -> Result<(), AgentError> {
        if self.is_tool_allowed(name) {
            return Ok(());
        }
        Err(AgentError::Refused {
            subject: name.to_string(),
            reason: "the tool is not on this agent's allowlist".to_string(),
        })
    }

    /// 模型调用上限前剩余的轮数，无上限时为 `None`。
    pub fn turns_remaining(&self, spend: &Spend) -> Option<usize> {
        if self.max_turns == 0 {
            return None;
        }
        Some(self.max_turns.saturating_sub(spend.turns))
    }

    /// 一次性检查每个上限；第一个被突破的决定错误。
    pub fn check(&self, spend: &Spend, elapsed: Duration) -> Result<(), AgentError> {
        self.check_turns(spend)
            .and_then(|()| self.check_tokens(spend))
            .and_then(|()| self.check_time(elapsed))
    }
}

impl Default for Guardrails {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Guardrails {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "turns={}, tools={}, tokens={}, seconds={}, tool_output={}",
            limit(self.max_turns as u64),
            limit(self.max_tool_calls as u64),
            limit(self.max_tokens),
            limit(self.max_seconds),
            limit(self.max_tool_output_chars as u64),
        )?;
        match &self.allowed_tools {
            Some(allowed) => write!(f, ", allowed={}", allowed.join("|")),
            None => write!(f, ", allowed=all"),
        }
    }
}

impl fmt::Debug for Guardrails {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Guardrails")
            .field("max_turns", &self.max_turns)
            .field("max_tool_calls", &self.max_tool_calls)
            .field("max_tokens", &self.max_tokens)
            .field("max_seconds", &self.max_seconds)
            .field("max_tool_output_chars", &self.max_tool_output_chars)
            .field("allowed_tools", &self.allowed_tools)
            .finish()
    }
}

fn check(limit: &str, used: u64, allowed: u64) -> Result<(), AgentError> {
    if allowed == 0 || used < allowed {
        return Ok(());
    }
    Err(AgentError::budget(limit, used, allowed))
}

fn limit(value: u64) -> String {
    if value == 0 {
        "none".to_string()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spent(turns: usize, tokens: u64) -> Spend {
        Spend {
            turns,
            tool_calls: 0,
            denied: 0,
            input_tokens: tokens,
            output_tokens: 0,
        }
    }

    #[test]
    fn defaults_stop_a_runaway_loop() {
        let limits = Guardrails::new();
        assert_eq!(limits.max_turns, Guardrails::DEFAULT_TURNS);
        assert!(limits.check(&Spend::default(), Duration::ZERO).is_ok());
        assert!(
            limits
                .check(&spent(Guardrails::DEFAULT_TURNS, 0), Duration::ZERO)
                .is_err()
        );
    }

    #[test]
    fn zero_means_unlimited() {
        let limits = Guardrails::unlimited();
        let spend = spent(10_000, 10_000_000);
        assert!(limits.check(&spend, Duration::from_secs(10_000)).is_ok());
    }

    #[test]
    fn the_first_ceiling_blown_wins() {
        let limits = Guardrails::new().with_turns(2).with_token_budget(100);
        let error = limits
            .check(&spent(5, 500), Duration::ZERO)
            .expect_err("both ceilings are blown");
        assert_eq!(error, AgentError::budget("max_turns", 5, 2));
    }

    #[test]
    fn an_allowlist_hides_tools_from_the_model() {
        let limits = Guardrails::new().allow_tools(["lookup"]);
        let tools = vec![
            ToolDescriptor::new("lookup", "read a table"),
            ToolDescriptor::new("deploy", "push to prod"),
        ];
        let kept = limits.filter_tools(tools);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].name, "lookup");
        assert!(!limits.is_tool_allowed("deploy"));
    }

    #[test]
    fn a_denied_tool_is_a_refusal_not_a_budget() {
        let limits = Guardrails::new().allow_tools(["echo"]);
        let error = limits.check_tool("rm").expect_err("rm is not allowed");
        assert!(matches!(error, AgentError::Refused { .. }));
    }

    #[test]
    fn remaining_turns_counts_down() {
        let limits = Guardrails::new().with_turns(4);
        assert_eq!(limits.turns_remaining(&spent(1, 0)), Some(3));
        assert_eq!(limits.turns_remaining(&spent(4, 0)), Some(0));
        assert_eq!(
            limits.with_turns(0).turns_remaining(&spent(4, 0)),
            None,
            "uncapped agents never run out"
        );
    }

    #[test]
    fn display_shows_the_ceilings() {
        let limits = Guardrails::new()
            .with_turns(3)
            .with_tool_calls(0)
            .allow_tools(["echo", "lookup"]);
        assert_eq!(
            limits.to_string(),
            "turns=3, tools=none, tokens=none, seconds=none, tool_output=8000, allowed=echo|lookup"
        );
    }
}
