//! Conversation state: bounded history, prompt assembly and folding.

use serde::{Deserialize, Serialize};
use xihe_llm::{ChatMessage, Role};

use crate::error::MemoryResult;
use crate::reflection::Summariser;

/// Who spoke a turn, in memory's own vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Speaker {
    /// The system or persona instruction.
    System,
    /// The human, or an upstream agent posing as one.
    User,
    /// The assistant being modelled.
    Assistant,
    /// A tool result folded into the dialogue.
    Tool,
}

impl Speaker {
    /// The role this speaker maps to when rendering a prompt.
    pub fn role(self) -> Role {
        match self {
            Self::System => Role::System,
            Self::User => Role::User,
            Self::Assistant => Role::Assistant,
            Self::Tool => Role::Tool,
        }
    }

    /// The speaker's label in a rendered transcript.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    /// Read a role as the matching speaker.
    pub fn of_role(role: Role) -> Self {
        match role {
            Role::System => Self::System,
            Role::Assistant => Self::Assistant,
            Role::Tool => Self::Tool,
            Role::User => Self::User,
        }
    }
}

/// One remembered utterance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    /// Who said it.
    pub speaker: Speaker,
    /// The text.
    pub content: String,
    /// Author label, when several speakers share a transcript.
    #[serde(default)]
    pub author: Option<String>,
    /// Call id, kept so a tool turn can be matched with its request.
    #[serde(default)]
    pub call_id: Option<String>,
}

impl Turn {
    /// A turn from any speaker.
    pub fn new(speaker: Speaker, content: impl Into<String>) -> Self {
        Self {
            speaker,
            content: content.into(),
            author: None,
            call_id: None,
        }
    }

    /// A user turn.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Speaker::User, content)
    }

    /// An assistant turn.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Speaker::Assistant, content)
    }

    /// A tool turn answering `call_id`.
    pub fn tool(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            call_id: Some(call_id.into()),
            ..Self::new(Speaker::Tool, content)
        }
    }

    /// Label the author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Convert into a chat message for the provider.
    pub fn to_message(&self) -> ChatMessage {
        let mut message = ChatMessage::new(self.speaker.role(), self.content.clone());
        if let Some(author) = &self.author {
            message = message.with_name(author);
        }
        if let Some(call_id) = &self.call_id {
            message.tool_call_id = Some(call_id.clone());
        }
        message
    }

    /// Build a turn from a chat message.
    pub fn from_message(message: &ChatMessage) -> Self {
        Self {
            speaker: Speaker::of_role(message.role),
            content: message.content.clone(),
            author: message.name.clone(),
            call_id: message.tool_call_id.clone(),
        }
    }

    /// Rough token size, for budget arithmetic.
    pub fn tokens(&self) -> usize {
        message_size(&self.content, self.author.as_deref())
    }
}

/// Token-ish size of a message: characters over four plus per-turn framing.
fn message_size(content: &str, author: Option<&str>) -> usize {
    let chars = content.chars().count() + author.map_or(0, |name| name.chars().count());
    chars.div_ceil(4) + 4
}

/// Why a window dropped something.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimReason {
    /// More turns than the window holds.
    TooManyTurns,
    /// The token budget was exceeded.
    TooManyTokens,
    /// Nothing was dropped.
    None,
}

/// A bounded, promotable conversation.
///
/// The system prompt and the most recent turns are always kept; when the window
/// overflows, middle turns are folded into a summary by the configured
/// [`Summariser`] so nothing important is silently thrown away.
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
    /// A window keeping 20 turns inside 4k estimated tokens.
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

    /// Set the standing instruction rendered first in every prompt.
    pub fn with_system(mut self, instructions: impl Into<String>) -> Self {
        self.system = Some(instructions.into());
        self
    }

    /// Cap the number of turns kept verbatim.
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns.max(2);
        self
    }

    /// Cap the estimated token size of the prompt.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens.max(32);
        self
    }

    /// Cap the length of a folded summary.
    pub fn with_summary_budget(mut self, max_chars: usize) -> Self {
        self.summary_max_chars = max_chars.max(32);
        self
    }

    /// Append a turn, returning what happened to the older ones.
    pub fn push(&mut self, turn: Turn) -> TrimReason {
        self.turns.push(turn);
        self.enforce_budget()
    }

    /// Append a user turn.
    pub fn push_user(&mut self, content: impl Into<String>) -> TrimReason {
        self.push(Turn::user(content))
    }

    /// Append an assistant turn.
    pub fn push_assistant(&mut self, content: impl Into<String>) -> TrimReason {
        self.push(Turn::assistant(content))
    }

    /// The standing system instruction, if set.
    pub fn system(&self) -> Option<&str> {
        self.system.as_deref()
    }

    /// Turns kept verbatim, oldest first.
    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// How many turns have been folded into the summary so far.
    pub fn fold_count(&self) -> usize {
        self.fold_count
    }

    /// The summary of everything folded away.
    pub fn summary(&self) -> Option<&str> {
        self.folded_summary.as_deref()
    }

    /// Estimated tokens of the prompt this window would render.
    pub fn tokens(&self) -> usize {
        let system = self.system.as_ref().map_or(0, |text| message_size(text, None));
        let summary = self
            .folded_summary
            .as_ref()
            .map_or(0, |text| message_size(text, None));
        system + summary + self.turns.iter().map(Turn::tokens).sum::<usize>()
    }

    /// Trim to the turn budget, keeping the newest turns.
    pub fn trim_to_turns(&mut self) -> Vec<Turn> {
        let mut dropped = Vec::new();
        while self.turns.len() > self.max_turns {
            dropped.push(self.turns.remove(0));
        }
        dropped
    }

    /// Trim to the token budget, keeping the newest turns.
    pub fn trim_to_tokens(&mut self) -> Vec<Turn> {
        let mut dropped = Vec::new();
        while self.tokens() > self.max_tokens && self.turns.len() > 2 {
            dropped.push(self.turns.remove(0));
        }
        dropped
    }

    /// Apply whichever budget is tighter.
    pub fn enforce_budget(&mut self) -> TrimReason {
        if self.turns.len() > self.max_turns {
            self.trim_to_turns();
            return TrimReason::TooManyTurns;
        }
        if self.tokens() > self.max_tokens {
            self.trim_to_tokens();
            return TrimReason::TooManyTokens;
        }
        TrimReason::None
    }

    /// Fold the oldest turns into the summary so far, keeping at least
    /// `keep_last` recent ones.
    pub async fn fold<S: Summariser + ?Sized>(&mut self, summariser: &S, keep_last: usize) -> MemoryResult<Option<String>> {
        let keep_last = keep_last.max(1);
        if self.turns.len() <= keep_last {
            return Ok(self.folded_summary.clone());
        }
        let stale_count = self.turns.len() - keep_last;
        let stale: Vec<Turn> = self.turns.drain(..stale_count).collect();
        let mut transcript = String::new();
        if let Some(previous) = &self.folded_summary {
            transcript.push_str(&format!("earlier context: {previous}\n"));
        }
        for turn in &stale {
            let speaker = turn.author.as_deref().unwrap_or(turn.speaker.as_str());
            transcript.push_str(&format!("{speaker}: {}\n", turn.content));
        }
        let summary = summariser.summarise(&transcript).await?;
        let summary = clamp_chars(&summary, self.summary_max_chars);
        self.folded_summary = Some(summary.clone());
        self.fold_count += stale.len();
        Ok(Some(summary))
    }

    /// The conversation as provider messages: system, summary, turns.
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

    /// The conversation plus a fresh user message.
    pub fn as_prompt(&self, user_text: &str) -> Vec<ChatMessage> {
        let mut messages = self.as_messages();
        messages.push(ChatMessage::user(user_text));
        messages
    }

    /// A plain transcript, useful for logs and prompts that are not structured.
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

    /// Forget everything except the system prompt.
    pub fn clear(&mut self) {
        self.turns.clear();
        self.folded_summary = None;
        self.fold_count = 0;
    }

    /// True when nothing has been recorded yet.
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

/// A per-entity conversation record, kept as a component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionMemory {
    /// Conversation identifier.
    pub session: String,
    /// The bounded window itself.
    pub window: ConversationWindow,
    /// Total user turns seen, including dropped ones.
    pub user_turns: usize,
    /// Total assistant turns seen, including dropped ones.
    pub assistant_turns: usize,
}

impl SessionMemory {
    /// Start a session with a persona instruction.
    pub fn new(session: impl Into<String>, system: impl Into<String>) -> Self {
        Self {
            session: session.into(),
            window: ConversationWindow::new().with_system(system),
            user_turns: 0,
            assistant_turns: 0,
        }
    }

    /// Record a user turn.
    pub fn record_user(&mut self, content: impl Into<String>) -> TrimReason {
        self.user_turns += 1;
        self.window.push_user(content)
    }

    /// Record an assistant turn.
    pub fn record_assistant(&mut self, content: impl Into<String>) -> TrimReason {
        self.assistant_turns += 1;
        self.window.push_assistant(content)
    }

    /// Record a tool turn.
    pub fn record_tool(&mut self, call_id: impl Into<String>, content: impl Into<String>) {
        self.window.push(Turn::tool(call_id, content));
    }

    /// Total turns seen, whether kept or dropped.
    pub fn turns_seen(&self) -> usize {
        self.user_turns + self.assistant_turns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
