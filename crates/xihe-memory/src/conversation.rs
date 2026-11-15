//! 对话状态：有界历史、提示词组装与折叠。

use serde::{Deserialize, Serialize};
use xihe_llm::{ChatMessage, Role, ToolCall};

use crate::error::MemoryResult;
use crate::reflection::Summariser;

/// 谁说了某一轮，用记忆自己的词汇。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Speaker {
    /// 系统或人设指令。
    System,
    /// 人类，或冒充人类的上游智能体。
    User,
    /// 被建模的助手。
    Assistant,
    /// 折叠进对话的工具结果。
    Tool,
}

impl Speaker {
    /// 渲染提示词时此说话者映射到的角色。
    pub fn role(self) -> Role {
        match self {
            Self::System => Role::System,
            Self::User => Role::User,
            Self::Assistant => Role::Assistant,
            Self::Tool => Role::Tool,
        }
    }

    /// 渲染后的转录中此说话者的标签。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    /// 把一个角色读作对应的说话者。
    pub fn of_role(role: Role) -> Self {
        match role {
            Role::System => Self::System,
            Role::Assistant => Self::Assistant,
            Role::Tool => Self::Tool,
            Role::User => Self::User,
        }
    }
}

/// 一条被记住的话语。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    /// 谁说的。
    pub speaker: Speaker,
    /// 文本。
    pub content: String,
    /// 作者标签，当多个说话者共享一份转录时。
    #[serde(default)]
    pub author: Option<String>,
    /// 调用 id，保留以便工具轮能与其请求匹配。
    #[serde(default)]
    pub call_id: Option<String>,
    /// 助手轮请求的调用，保留以便回放转录时
    /// 仍能用产出它的调用应答每个工具结果。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl Turn {
    /// 来自任意说话者的一轮。
    pub fn new(speaker: Speaker, content: impl Into<String>) -> Self {
        Self {
            speaker,
            content: content.into(),
            author: None,
            call_id: None,
            tool_calls: Vec::new(),
        }
    }

    /// 一个用户轮。
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Speaker::User, content)
    }

    /// 一个助手轮。
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Speaker::Assistant, content)
    }

    /// 应答 `call_id` 的一个工具轮。
    pub fn tool(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            call_id: Some(call_id.into()),
            ..Self::new(Speaker::Tool, content)
        }
    }

    /// 标注作者。
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// 附上助手轮请求运行时执行的调用。
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = calls;
        self
    }

    /// 转换为给提供方的聊天消息。
    pub fn to_message(&self) -> ChatMessage {
        let mut message = ChatMessage::new(self.speaker.role(), self.content.clone());
        if let Some(author) = &self.author {
            message = message.with_name(author);
        }
        if let Some(call_id) = &self.call_id {
            message.tool_call_id = Some(call_id.clone());
        }
        message.tool_calls = self.tool_calls.clone();
        message
    }

    /// 从聊天消息构建一轮。
    pub fn from_message(message: &ChatMessage) -> Self {
        Self {
            speaker: Speaker::of_role(message.role),
            content: message.content.clone(),
            author: message.name.clone(),
            call_id: message.tool_call_id.clone(),
            tool_calls: message.tool_calls.clone(),
        }
    }

    /// 粗略 token 大小，用于预算算术。
    pub fn tokens(&self) -> usize {
        let mut size = message_size(&self.content, self.author.as_deref());
        for call in &self.tool_calls {
            size += message_size(&call.render(), None);
        }
        size
    }
}

/// 消息的类 token 大小：字符数除以四，加上每轮框架开销。
fn message_size(content: &str, author: Option<&str>) -> usize {
    let chars = content.chars().count() + author.map_or(0, |name| name.chars().count());
    chars.div_ceil(4) + 4
}

/// 窗口为何丢弃了某些内容。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimReason {
    /// 轮数超过窗口所能容纳。
    TooManyTurns,
    /// 超出 token 预算。
    TooManyTokens,
    /// 没有丢弃任何内容。
    None,
}

/// 一个有界、可提升的对话。
///
/// 系统提示词与最近的轮总是保留；当窗口溢出时，中间的轮
/// 由配置的 [`Summariser`] 折叠为摘要，因此重要内容不会被
/// 静默丢弃。
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConversationWindow {
    #[serde(default)]
    system: Option<String>,
    turns: Vec<Turn>,
    max_turns: usize,
    max_tokens: usize,
    summary_max_chars: usize,
    #[serde(default)]
    folded_summary: Option<String>,
    #[serde(default)]
    fold_count: usize,
}

impl ConversationWindow {
    /// 一个在 4k 估算 token 内保留 20 轮的窗口。
    pub fn new() -> Self {
        Self {
            system: None,
            turns: Vec::new(),
            max_turns: 20,
            max_tokens: 4_096,
            summary_max_chars: 400,
            folded_summary: None,
            fold_count: 0,
        }
    }

    /// 设置在每个提示词中最先渲染的常驻指令。
    pub fn with_system(mut self, instructions: impl Into<String>) -> Self {
        self.system = Some(instructions.into());
        self
    }

    /// 限制逐字保留的轮数。
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns.max(2);
        self
    }

    /// 限制提示词的估算 token 大小。
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens.max(32);
        self
    }

    /// 限制折叠摘要的长度。
    pub fn with_summary_budget(mut self, max_chars: usize) -> Self {
        self.summary_max_chars = max_chars.max(32);
        self
    }

    /// 追加一轮，返回较旧的轮发生了什么。
    pub fn push(&mut self, turn: Turn) -> TrimReason {
        self.turns.push(turn);
        self.enforce_budget()
    }

    /// 追加一个用户轮。
    pub fn push_user(&mut self, content: impl Into<String>) -> TrimReason {
        self.push(Turn::user(content))
    }

    /// 追加一个助手轮。
    pub fn push_assistant(&mut self, content: impl Into<String>) -> TrimReason {
        self.push(Turn::assistant(content))
    }

    /// 常驻系统指令（若已设置）。
    pub fn system(&self) -> Option<&str> {
        self.system.as_deref()
    }

    /// 逐字保留的轮，最旧的在前。
    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// 迄今有多少轮被折叠进摘要。
    pub fn fold_count(&self) -> usize {
        self.fold_count
    }

    /// 所有被折叠掉内容的摘要。
    pub fn summary(&self) -> Option<&str> {
        self.folded_summary.as_deref()
    }

    /// 此窗口将渲染的提示词的估算 token。
    pub fn tokens(&self) -> usize {
        let system = self.system.as_ref().map_or(0, |text| message_size(text, None));
        let summary = self
            .folded_summary
            .as_ref()
            .map_or(0, |text| message_size(text, None));
        system + summary + self.turns.iter().map(Turn::tokens).sum::<usize>()
    }

    /// 裁剪到轮预算，保留最新的轮。
    pub fn trim_to_turns(&mut self) -> Vec<Turn> {
        let mut dropped = Vec::new();
        while self.turns.len() > self.max_turns {
            dropped.push(self.turns.remove(0));
        }
        self.drop_orphaned_leading_tools(&mut dropped);
        dropped
    }

    /// 裁剪到 token 预算，保留最新的轮。
    pub fn trim_to_tokens(&mut self) -> Vec<Turn> {
        let mut dropped = Vec::new();
        while self.tokens() > self.max_tokens && self.turns.len() > 2 {
            dropped.push(self.turns.remove(0));
        }
        self.drop_orphaned_leading_tools(&mut dropped);
        dropped
    }

    /// 助手调用已被裁掉的工具结果无法单独回放，
    /// 因此也丢弃它，而非发出格式错误的提示词。
    fn drop_orphaned_leading_tools(&mut self, dropped: &mut Vec<Turn>) {
        while self
            .turns
            .first()
            .is_some_and(|turn| turn.speaker == Speaker::Tool)
        {
            dropped.push(self.turns.remove(0));
        }
    }

    /// 同时应用两个预算，保留最新的轮。先强制轮上限，
    /// 再强制 token 上限，因此同时突破两者的 push
    /// 不会让提示词超出其 token 预算。
    pub fn enforce_budget(&mut self) -> TrimReason {
        let mut reason = TrimReason::None;
        if self.turns.len() > self.max_turns {
            self.trim_to_turns();
            reason = TrimReason::TooManyTurns;
        }
        if self.tokens() > self.max_tokens {
            self.trim_to_tokens();
            reason = TrimReason::TooManyTokens;
        }
        reason
    }

    /// 把最旧的轮折叠进迄今的摘要，至少保留
    /// `keep_last` 个最近的轮。
    pub async fn fold<S: Summariser + ?Sized>(&mut self, summariser: &S, keep_last: usize) -> MemoryResult<Option<String>> {
        let keep_last = keep_last.max(1);
        if self.turns.len() <= keep_last {
            return Ok(self.folded_summary.clone());
        }
        let mut stale_count = self.turns.len() - keep_last;
        // 绝不让保留区从工具结果开始：回退到请求它的调用，
        // 使折叠边界不会孤立一个工具轮。
        while stale_count > 0 && self.turns[stale_count].speaker == Speaker::Tool {
            stale_count -= 1;
        }
        if stale_count == 0 {
            return Ok(self.folded_summary.clone());
        }
        let stale: Vec<Turn> = self.turns.drain(..stale_count).collect();
        let mut transcript = String::new();
        if let Some(previous) = &self.folded_summary {
            transcript.push_str(&format!("earlier context: {previous}\n"));
        }
        for turn in &stale {
            let speaker = turn.author.as_deref().unwrap_or(turn.speaker.as_str());
            transcript.push_str(&format!("{speaker}: {}\n", turn.content));
            for call in &turn.tool_calls {
                transcript.push_str(&format!("{speaker} called {}\n", call.render()));
            }
        }
        let summary = summariser.summarise(&transcript).await?;
        let summary = clamp_chars(&summary, self.summary_max_chars);
        self.folded_summary = Some(summary.clone());
        self.fold_count += stale.len();
        Ok(Some(summary))
    }

    /// 作为提供方消息的对话：系统、摘要、轮。
    pub fn as_messages(&self) -> Vec<ChatMessage> {
        let mut out = Vec::with_capacity(self.turns.len() + 2);
        if let Some(system) = &self.system {
            out.push(ChatMessage::system(system.clone()));
        }
        if let Some(summary) = &self.folded_summary {
            out.push(ChatMessage::system(format!(
                "summary of earlier turns: {summary}"
            )));
        }
        out.extend(self.turns.iter().map(Turn::to_message));
        out
    }

    /// 对话加上一条新的用户消息。
    pub fn as_prompt(&self, user_text: &str) -> Vec<ChatMessage> {
        let mut messages = self.as_messages();
        messages.push(ChatMessage::user(user_text));
        messages
    }

    /// 纯文本转录，适用于日志与非结构化提示词。
    pub fn render(&self) -> String {
        let mut out = String::new();
        if let Some(summary) = &self.folded_summary {
            out.push_str(&format!("[earlier] {summary}\n"));
        }
        for turn in &self.turns {
            let fallback = match turn.speaker {
                Speaker::System => "system",
                Speaker::User => "user",
                Speaker::Assistant => "assistant",
                Speaker::Tool => "tool",
            };
            let speaker = turn.author.as_deref().unwrap_or(fallback);
            out.push_str(&format!("{speaker}: {}\n", turn.content));
        }
        out
    }

    /// 忘记除系统提示词外的一切。
    pub fn clear(&mut self) {
        self.turns.clear();
        self.folded_summary = None;
        self.fold_count = 0;
    }

    /// 当尚未记录任何内容时为真。
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty() && self.folded_summary.is_none()
    }
}

fn clamp_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// 每实体一份的对话记录，作为组件保存。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionMemory {
    /// 对话标识符。
    pub session: String,
    /// 有界窗口本身。
    pub window: ConversationWindow,
    /// 见过的用户轮总数，含被丢弃的。
    pub user_turns: usize,
    /// 见过的助手轮总数，含被丢弃的。
    pub assistant_turns: usize,
}

impl SessionMemory {
    /// 以人设指令开始一个会话。
    pub fn new(session: impl Into<String>, system: impl Into<String>) -> Self {
        Self {
            session: session.into(),
            window: ConversationWindow::new().with_system(system),
            user_turns: 0,
            assistant_turns: 0,
        }
    }

    /// 记录一个用户轮。
    pub fn record_user(&mut self, content: impl Into<String>) -> TrimReason {
        self.user_turns += 1;
        self.window.push_user(content)
    }

    /// 记录一个助手轮。
    pub fn record_assistant(&mut self, content: impl Into<String>) -> TrimReason {
        self.assistant_turns += 1;
        self.window.push_assistant(content)
    }

    /// 记录一个工具轮。
    pub fn record_tool(&mut self, call_id: impl Into<String>, content: impl Into<String>) {
        self.window.push(Turn::tool(call_id, content));
    }

    /// 见过的轮总数，无论保留还是丢弃。
    pub fn turns_seen(&self) -> usize {
        self.user_turns + self.assistant_turns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::reflection::HeuristicSummariser;

    fn dialogue(window: &mut ConversationWindow, rounds: usize) {
        for index in 0..rounds {
            window.push_user(format!("question {index} with some padding words"));
            window.push_assistant(format!("answer {index} with some padding words"));
        }
    }

    #[test]
    fn speakers_map_to_roles() {
        assert_eq!(Speaker::Tool.role(), Role::Tool);
        assert_eq!(Speaker::of_role(Role::Assistant), Speaker::Assistant);
        assert_eq!(Speaker::of_role(Role::User), Speaker::User);
    }

    #[test]
    fn turns_round_trip_through_messages() {
        let turn = Turn::tool("call_1", "21C").with_author("weather");
        let message = turn.to_message();
        assert_eq!(message.role, Role::Tool);
        assert_eq!(message.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(message.name.as_deref(), Some("weather"));
        assert_eq!(Turn::from_message(&message), turn);
    }

    #[test]
    fn push_respects_the_turn_budget() {
        let mut window = ConversationWindow::new().with_max_turns(4);
        let reason = dialogue_with_return(&mut window, 4);
        assert_eq!(reason, TrimReason::TooManyTurns);
        assert_eq!(window.turns().len(), 4, "the newest four survive");
        assert_eq!(window.turns()[0].content, "question 2 with some padding words");
    }

    fn dialogue_with_return(window: &mut ConversationWindow, rounds: usize) -> TrimReason {
        let mut last = TrimReason::None;
        for index in 0..rounds {
            window.push_user(format!("question {index} with some padding words"));
            last = window.push_assistant(format!("answer {index} with some padding words"));
        }
        last
    }

    #[test]
    fn token_budget_wins_on_long_turns() {
        let mut window = ConversationWindow::new()
            .with_max_turns(100)
            .with_max_tokens(80);
        for index in 0..10 {
            window.push_user(format!("question {index} {}", "pad".repeat(10)));
        }
        assert!(window.tokens() <= 80, "{}", window.tokens());
        assert!(
            window.turns().len() < 10,
            "the oldest turns must go to fit the budget"
        );
        assert!(window.turns().last().unwrap().content.contains('9'));
    }

    #[test]
    fn budgets_keep_at_least_two_turns() {
        let mut window = ConversationWindow::new().with_max_tokens(32).with_max_turns(2);
        window.push_user("a very long question that blows the token budget on its own");
        window.push_assistant("a very long answer that also blows the token budget");
        assert_eq!(window.turns().len(), 2, "a live conversation cannot be emptied");
    }

    #[test]
    fn a_push_enforces_both_ceilings_after_reconfiguration() {
        let mut window = ConversationWindow::new()
            .with_max_turns(100)
            .with_max_tokens(10_000);
        for index in 0..6 {
            window.push_user(format!("question {index} {}", "pad".repeat(10)));
        }
        // 把两个上限都收紧到当前状态之下，再 push 一次，
        // 使真实的 push 路径运行 enforce_budget。
        window = window.with_max_turns(4).with_max_tokens(40);
        window.push_user("one more");
        assert!(window.turns().len() <= 4, "the turn ceiling holds");
        assert!(
            window.tokens() <= 40,
            "the token ceiling must hold too, not just the turn count: {}",
            window.tokens()
        );
    }

    #[test]
    fn prompts_render_system_summary_and_turns() {
        let mut window = ConversationWindow::new().with_system("be terse");
        window.push_user("hi");
        window.push_assistant("hello");
        assert_eq!(window.turns().iter().map(|turn| turn.speaker).collect::<Vec<_>>(), vec![Speaker::User, Speaker::Assistant]);
        let prompt = window.as_prompt("next question");
        assert_eq!(prompt.len(), 4);
        assert_eq!(prompt[0].role, Role::System);
        assert_eq!(prompt[0].content, "be terse");
        assert_eq!(prompt[3].role, Role::User);
        assert_eq!(prompt[3].content, "next question");
        assert_eq!(window.render(), "user: hi\nassistant: hello\n");
    }

    #[tokio::test]
    async fn folding_compresses_history_into_a_summary() {
        let mut window = ConversationWindow::new().with_max_turns(100);
        dialogue(&mut window, 6);
        let before = window.tokens();
        let summary = window
            .fold(&HeuristicSummariser::new(3, 200), 2)
            .await
            .unwrap()
            .expect("a summary");
        assert_eq!(window.turns().len(), 2, "only the last two turns stay verbatim");
        assert_eq!(window.fold_count(), 10);
        assert!(!summary.is_empty());
        assert!(window.tokens() < before, "folding must actually shrink");
        let prompt = window.as_prompt("continue");
        assert_eq!(prompt.len(), 4, "summary, two turns, new question");
        assert!(prompt[0].content.starts_with("summary of earlier turns:"));
    }

    #[tokio::test]
    async fn folding_twice_carries_the_previous_summary() {
        let mut window = ConversationWindow::new().with_max_turns(100).with_summary_budget(80);
        dialogue(&mut window, 3);
        let first = window.fold(&HeuristicSummariser::new(2, 200), 1).await.unwrap();
        dialogue(&mut window, 3);
        let second = window.fold(&HeuristicSummariser::new(2, 200), 1).await.unwrap();
        assert_ne!(first, second, "the second fold sees more history");
        assert!(window.summary().unwrap().chars().count() <= 80, "summary obeys its own cap");
    }

    #[tokio::test]
    async fn folding_an_already_short_window_is_a_noop() {
        let mut window = ConversationWindow::new();
        window.push_user("only one");
        let summary = window.fold(&HeuristicSummariser::new(2, 200), 4).await.unwrap();
        assert!(summary.is_none());
        assert_eq!(window.turns().len(), 1);
    }

    #[derive(Default)]
    struct CapturingSummariser {
        seen: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Summariser for CapturingSummariser {
        async fn summarise(&self, transcript: &str) -> MemoryResult<String> {
            self.seen.lock().unwrap().push(transcript.to_string());
            Ok("a summary".to_string())
        }

        fn id(&self) -> &str {
            "capturing"
        }
    }

    #[tokio::test]
    async fn folding_records_the_tool_calls_an_assistant_made() {
        let mut window = ConversationWindow::new().with_max_turns(100);
        window.push(
            Turn::assistant("let me check").with_tool_calls(vec![ToolCall::new(
                "c1",
                "weather",
                serde_json::json!({"city": "Paris"}),
            )]),
        );
        window.push(Turn::tool("c1", "21C"));
        window.push_user("thanks");
        let capture = CapturingSummariser::default();
        window.fold(&capture, 1).await.unwrap();
        let seen = capture.seen.lock().unwrap();
        let transcript = seen.last().expect("a fold happened");
        assert!(
            transcript.contains("weather"),
            "the transcript must record the action, not just the result: {transcript}"
        );
    }

    #[tokio::test]
    async fn folding_keeps_a_tool_result_with_its_call() {
        let mut window = ConversationWindow::new().with_max_turns(100);
        for index in 0..4 {
            window.push_assistant(format!("calling tool {index}"));
            window.push(Turn::tool(format!("c{index}"), format!("result {index}")));
        }
        // 八个轮；keep_last=5 朴素地让保留区从工具 `c1` 开始，
        // 把它与请求它的助手调用孤立开来。
        let summary = window
            .fold(&HeuristicSummariser::new(2, 200), 5)
            .await
            .unwrap();
        assert!(summary.is_some(), "folding still happens");
        assert_ne!(
            window.turns()[0].speaker,
            Speaker::Tool,
            "a tool result must keep the call that requested it"
        );
        assert_eq!(window.turns()[0].content, "calling tool 1");
        assert_eq!(
            window.fold_count(),
            2,
            "only the first complete call/result pair is folded"
        );
    }

    fn assert_no_orphan_tools(window: &ConversationWindow) {
        let turns = window.turns();
        for (index, turn) in turns.iter().enumerate() {
            if turn.speaker != Speaker::Tool {
                continue;
            }
            let id = turn.call_id.as_deref().expect("a tool turn names its call");
            let answered = turns[..index]
                .iter()
                .any(|prior| prior.tool_calls.iter().any(|call| call.id == id));
            assert!(
                answered,
                "tool result {id} was orphaned from its call: {turns:?}"
            );
        }
    }

    fn tool_dialogue(window: &mut ConversationWindow, rounds: usize) {
        for index in 0..rounds {
            window.push(
                Turn::assistant(format!("calling tool {index}")).with_tool_calls(vec![
                    ToolCall::new(format!("c{index}"), "echo", serde_json::json!({})),
                ]),
            );
            window.push(Turn::tool(format!("c{index}"), format!("result {index}")));
        }
    }

    #[test]
    fn turn_trim_keeps_a_tool_result_with_its_call() {
        let mut window = ConversationWindow::new().with_max_turns(5);
        tool_dialogue(&mut window, 6);
        assert_no_orphan_tools(&window);
    }

    #[test]
    fn token_trim_keeps_a_tool_result_with_its_call() {
        let mut window = ConversationWindow::new()
            .with_max_turns(100)
            .with_max_tokens(39);
        tool_dialogue(&mut window, 6);
        assert_no_orphan_tools(&window);
    }

    #[test]
    fn clamp_is_char_safe() {
        assert_eq!(clamp_chars("hello", 10), "hello");
        assert_eq!(clamp_chars("hello", 3), "he…");
        assert_eq!(clamp_chars("你好世界", 3), "你好…", "multibyte text is counted in chars");
        assert_eq!(clamp_chars("", 0), "");
    }

    #[test]
    fn session_memory_counts_everything_it_saw() {
        let mut session = SessionMemory::new("chat-1", "persona");
        session.record_user("one");
        session.record_assistant("two");
        session.record_assistant("three");
        session.record_tool("call", "output");
        assert_eq!(session.turns_seen(), 3);
        assert_eq!(session.window.turns().len(), 4);
        assert_eq!(session.session, "chat-1");
    }

    #[test]
    fn window_is_serialisable() {
        let mut window = ConversationWindow::new().with_system("persona");
        window.push_user("hi");
        let text = serde_json::to_string(&window).unwrap();
        let back: ConversationWindow = serde_json::from_str(&text).unwrap();
        assert_eq!(back, window);
        assert_eq!(back.tokens(), window.tokens());
    }
}
