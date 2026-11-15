//! A scripted, offline provider for tests, demos and deterministic replays.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream;
use serde_json::Value;

use crate::error::LlmError;
use crate::message::{
    ChatMessage, FinishReason, GenerationResponse, StreamChunk, TokenUsage, ToolCall,
};
use crate::provider::{Capabilities, Provider};
use crate::request::GenerationRequest;

/// One scripted answer from [`MockProvider`].
#[derive(Clone, Debug)]
pub enum Reply {
    /// Plain assistant text.
    Text(String),
    /// Assistant text plus the tool calls it wants executed.
    Tools {
        /// Assistant text accompanying the calls.
        content: String,
        /// The calls to execute.
        calls: Vec<ToolCall>,
    },
    /// A fully-specified response, for edge cases the helpers cannot express.
    Response(GenerationResponse),
    /// Fail the call, e.g. to exercise the retry policy.
    Fail(LlmError),
}

impl Reply {
    /// A plain text reply.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// A single tool call with no accompanying text.
    pub fn tool(name: impl Into<String>, arguments: impl Into<Value>) -> Self {
        Self::Tools {
            content: String::new(),
            calls: vec![ToolCall::new("", name, arguments)],
        }
    }

    /// Several tool calls with no accompanying text.
    pub fn tools(calls: Vec<ToolCall>) -> Self {
        Self::Tools {
            content: String::new(),
            calls,
        }
    }

    /// An injected failure.
    pub fn fail(error: LlmError) -> Self {
        Self::Fail(error)
    }

    /// The assistant message this reply would produce.
    pub fn response(&self) -> Option<GenerationResponse> {
        match self {
            Self::Text(text) => Some(GenerationResponse::text(text.clone())),
            Self::Tools { content, calls } => Some(
                GenerationResponse::text(content.clone()).with_tool_calls(calls.clone()),
            ),
            Self::Response(response) => Some(response.clone()),
            Self::Fail(_) => None,
        }
    }
}

/// How a mock behaves once its script runs out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Exhausted {
    /// Keep answering with the last scripted reply, so long runs terminate.
    RepeatLast,
    /// Replay the script from the beginning, for cadence and loop tests.
    Cycle,
    /// Fail with [`LlmError::Config`], so a test notices extra calls.
    Fail,
}

#[derive(Debug)]
struct MockState {
    script: VecDeque<Reply>,
    requests: Vec<GenerationRequest>,
    replies: usize,
    failures: usize,
    next_call_id: u64,
    last: Option<Reply>,
}

/// A deterministic provider that answers from a script and records requests.
///
/// Cloning shares the script and the log, so an `Arc<MockProvider>` handed to one
/// system and inspected from another sees the same state.
#[derive(Clone, Debug)]
pub struct MockProvider {
    state: Arc<Mutex<MockState>>,
    model: String,
    latency: Duration,
    exhausted: Exhausted,
    fallback: Reply,
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MockProvider {
    /// A mock with an empty script.
    pub fn new() -> Self {
        Self::with_script(Vec::new())
    }

    /// Answers taken in order; once consumed, [`MockProvider::exhausted`] decides.
    pub fn with_script(script: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            state: Arc::new(Mutex::new(MockState {
                script: script.into_iter().collect(),
                requests: Vec::new(),
                replies: 0,
                failures: 0,
                next_call_id: 1,
                last: None,
            })),
            model: "mock-model".to_string(),
            latency: Duration::ZERO,
            exhausted: Exhausted::RepeatLast,
            fallback: Reply::text(""),
        }
    }

    /// A one-shot mock returning `text`.
    pub fn text(text: impl Into<String>) -> Self {
        Self::with_script([Reply::text(text)])
    }

    /// A mock that always answers with the same text, however often it is called.
    pub fn echo(text: impl Into<String>) -> Self {
        Self::with_script([Reply::text(text)]).with_exhausted(Exhausted::Cycle)
    }

    /// A mock that asks for `tool` with `arguments` on the first call and then
    /// answers with `final_text` — the shape most ReAct tests need.
    pub fn tool_then_answer(
        tool: impl Into<String>,
        args: impl Into<Value>,
        final_text: impl Into<String>,
    ) -> Self {
        Self::with_script([Reply::tool(tool, args), Reply::text(final_text)])
    }

    /// A mock that fails `failures_before` times and then succeeds.
    pub fn flaky(failures_before: u32, message: impl Into<String>) -> Self {
        let message = message.into();
        let script = std::iter::repeat_n(
            Reply::fail(LlmError::transport(format!("{message} (injected)"))),
            failures_before as usize,
        )
        .chain(std::iter::once(Reply::text("recovered")));
        Self::with_script(script).with_exhausted(Exhausted::Cycle)
    }

    /// Override the advertised default model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Artificial delay per call, for timeout and concurrency tests.
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = latency;
        self
    }

    /// Choose what happens once the script runs out.
    pub fn with_exhausted(mut self, behaviour: Exhausted) -> Self {
        self.exhausted = behaviour;
        self
    }

    /// The reply used when the script is empty and `exhausted` is `RepeatLast`.
    pub fn with_fallback(mut self, reply: Reply) -> Self {
        self.fallback = reply;
        self
    }

    /// Append a reply to the end of the script.
    pub fn push(&self, reply: Reply) {
        self.lock().script.push_back(reply);
    }

    /// The per-call delay.
    pub fn latency(&self) -> Duration {
        self.latency
    }

    /// Every request received so far, oldest first.
    pub fn requests(&self) -> Vec<GenerationRequest> {
        self.lock().requests.clone()
    }

    /// The most recent request, if any.
    pub fn last_request(&self) -> Option<GenerationRequest> {
        self.lock().requests.last().cloned()
    }

    /// Total requests received, failures included.
    pub fn call_count(&self) -> usize {
        self.lock().requests.len()
    }

    /// How many calls were answered with an injected error.
    pub fn failure_count(&self) -> usize {
        self.lock().failures
    }

    /// The conversation the mock has been asked about, flattened for assertions.
    pub fn transcript(&self) -> Vec<ChatMessage> {
        self.lock()
            .requests
            .last()
            .map(|request| request.messages.clone())
            .unwrap_or_default()
    }

    /// Drop the script and the recorded history.
    pub fn reset(&self) {
        let mut state = self.lock();
        state.script.clear();
        state.requests.clear();
        state.replies = 0;
        state.failures = 0;
        state.next_call_id = 1;
        state.last = None;
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MockState> {
        // The guard is never held across an await point; see `pick` and `record`.
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Take the next scripted reply without holding the lock across the sleep.
    fn pick(&self) -> (Reply, Option<u64>) {
        let mut state = self.lock();
        let reply = match state.script.pop_front() {
            Some(reply) => {
                if self.exhausted == Exhausted::Cycle {
                    state.script.push_back(reply.clone());
                }
                state.last = Some(reply.clone());
                reply
            }
            None => match self.exhausted {
                Exhausted::Fail => {
                    // `record` counts the failure; here we only pick the reply.
                    return (Reply::Fail(LlmError::config("mock script exhausted")), None);
                }
                Exhausted::RepeatLast => state
                    .last
                    .clone()
                    .unwrap_or_else(|| self.fallback.clone()),
                Exhausted::Cycle => self.fallback.clone(),
            },
        };
        let id = state.next_call_id;
        state.next_call_id += 1;
        (reply, Some(id))
    }

    fn record(&self, request: &GenerationRequest, failed: bool) {
        let mut state = self.lock();
        state.requests.push(request.clone());
        if failed {
            state.failures += 1;
        } else {
            state.replies += 1;
        }
    }

    /// Fill in ids, usage and the model name so replies look provider-shaped.
    fn hydrate(
        &self,
        reply: GenerationResponse,
        request: &GenerationRequest,
        id: u64,
    ) -> GenerationResponse {
        let mut response = reply;
        if response.model.is_empty() {
            response.model = if request.model.is_empty() {
                self.model.clone()
            } else {
                request.model.clone()
            };
        }
        for (offset, call) in response.tool_calls.iter_mut().enumerate() {
            if call.id.is_empty() {
                call.id = format!("call_{id}_{}", offset + 1);
            }
        }
        if response.finish_reason.is_none() {
            response.finish_reason = Some(if response.calls_tools() {
                FinishReason::ToolCalls
            } else {
                FinishReason::Stop
            });
        }
        if response.usage.is_zero() {
            response.usage = TokenUsage::new(
                request.estimated_prompt_tokens() as u64,
                crate::estimate_tokens(response.content.len()) as u64,
            );
        }
        response
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn id(&self) -> &str {
        "mock"
    }

    fn default_model(&self) -> &str {
        &self.model
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            tools: true,
            streaming: true,
            vision: false,
            json_mode: false,
        }
    }

    async fn generate(&self, request: &GenerationRequest) -> Result<GenerationResponse, LlmError> {
        let (reply, id) = self.pick();
        if !self.latency.is_zero() {
            tokio::time::sleep(self.latency).await;
        }
        match reply {
            Reply::Fail(error) => {
                self.record(request, true);
                tracing::debug!(error = %error, "mock injected a failure");
                Err(error)
            }
            Reply::Text(text) => {
                self.record(request, false);
                Ok(self.hydrate(GenerationResponse::text(text), request, id.unwrap_or(0)))
            }
            Reply::Tools { content, calls } => {
                self.record(request, false);
                Ok(self.hydrate(
                    GenerationResponse::text(content).with_tool_calls(calls),
                    request,
                    id.unwrap_or(0),
                ))
            }
            Reply::Response(response) => {
                self.record(request, false);
                Ok(self.hydrate(response, request, id.unwrap_or(0)))
            }
        }
    }

    async fn stream(
        &self,
        request: &GenerationRequest,
    ) -> Result<crate::provider::StreamBox, LlmError> {
        let response = self.generate(request).await?;
        let pieces = split_words(&response.content);
        let usage = response.usage;
        let mut chunks: Vec<Result<StreamChunk, LlmError>> = pieces
            .into_iter()
            .map(|piece| Ok(StreamChunk::text(piece)))
            .collect();
        chunks.push(Ok(StreamChunk {
            text: String::new(),
            tool_calls: response.tool_calls,
            finish_reason: response.finish_reason,
            usage: Some(usage),
        }));
        Ok(Box::pin(stream::iter(chunks)))
    }
}

fn split_words(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut pieces = Vec::new();
    let mut current = String::new();
    for word in text.split_inclusive(' ') {
        current.push_str(word);
        if current.chars().count() >= 12 {
            pieces.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn request() -> GenerationRequest {
        GenerationRequest::new("").with_user("hello")
    }

    #[tokio::test]
    async fn script_is_consumed_in_order() {
        let provider = MockProvider::with_script([Reply::text("one"), Reply::text("two")]);
        assert_eq!(provider.generate(&request()).await.unwrap().content, "one");
        assert_eq!(provider.generate(&request()).await.unwrap().content, "two");
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn repeat_last_ends_a_loop() {
        let provider = MockProvider::with_script([Reply::text("only")]).with_exhausted(Exhausted::RepeatLast);
        for _ in 0..3 {
            assert_eq!(provider.generate(&request()).await.unwrap().content, "only");
        }
    }

    #[tokio::test]
    async fn empty_script_can_fail_loudly_instead_of_repeating() {
        let provider = MockProvider::new().with_exhausted(Exhausted::Fail);
        let error = provider.generate(&request()).await.unwrap_err();
        assert_eq!(error.kind(), "config");
        assert_eq!(provider.failure_count(), 1, "one call, one failure");
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn cycle_replays_the_script() {
        let provider = MockProvider::echo("again");
        assert_eq!(provider.generate(&request()).await.unwrap().content, "again");
        assert_eq!(provider.generate(&request()).await.unwrap().content, "again");
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn tool_replies_get_synthetic_ids_and_usage() {
        let provider = MockProvider::tool_then_answer("search", serde_json::json!({ "q": "ecs" }), "done");
        let first = provider.generate(&request()).await.unwrap();
        assert_eq!(first.step(), crate::GenerationStep::ExecuteTools);
        assert_eq!(first.tool_calls[0].id, "call_1_1");
        assert_eq!(first.tool_calls[0].string_argument("q"), Some("ecs"));
        assert!(!first.usage.is_zero(), "usage is synthesised for budget tests");
        let second = provider.generate(&request()).await.unwrap();
        assert_eq!(second.step(), crate::GenerationStep::Answer);
        assert_eq!(second.model, "mock-model");
    }

    #[tokio::test]
    async fn requests_are_recorded_for_inspection() {
        let provider = MockProvider::text("ok");
        let request = GenerationRequest::new("gpt-x")
            .with_system("be brief")
            .with_user("question");
        provider.generate(&request).await.unwrap();
        let recorded = provider.last_request().expect("recorded");
        assert_eq!(recorded.messages.len(), 2);
        assert_eq!(recorded.system_prompt(), Some("be brief"));
        assert_eq!(provider.transcript()[1].content, "question");
    }

    #[tokio::test]
    async fn failures_are_separate_from_replies() {
        let provider = MockProvider::flaky(2, "network down");
        assert!(provider.generate(&request()).await.is_err());
        assert!(provider.generate(&request()).await.is_err());
        assert_eq!(provider.failure_count(), 2);
        assert_eq!(provider.generate(&request()).await.unwrap().content, "recovered");
        assert_eq!(provider.call_count(), 3, "failures are still calls");
    }

    #[tokio::test]
    async fn streamed_text_arrives_in_words_then_a_terminal_chunk() {
        let provider = MockProvider::text("alpha beta gamma delta epsilon");
        let stream = provider.stream(&request()).await.unwrap();
        let chunks: Vec<StreamChunk> = stream.map(Result::unwrap).collect().await;
        assert!(chunks.len() > 2, "the text should be split up");
        assert!(chunks.last().unwrap().is_last());
        let joined = chunks.iter().map(|chunk| chunk.text.as_str()).collect::<String>();
        assert_eq!(joined, "alpha beta gamma delta epsilon");
    }

    #[tokio::test]
    async fn push_extends_a_running_script() {
        let provider = MockProvider::with_script([Reply::text("first")]);
        provider.push(Reply::text("second"));
        assert_eq!(provider.generate(&request()).await.unwrap().content, "first");
        assert_eq!(provider.generate(&request()).await.unwrap().content, "second");
    }

    #[tokio::test]
    async fn reset_clears_history() {
        let provider = MockProvider::text("x");
        provider.generate(&request()).await.unwrap();
        provider.reset();
        assert_eq!(provider.call_count(), 0);
        assert!(provider.requests().is_empty());
    }

    #[tokio::test]
    async fn concurrent_calls_do_not_deadlock() {
        let provider = MockProvider::with_script(
            (0..8).map(|index| Reply::text(format!("reply {index}"))).collect::<Vec<_>>(),
        )
        .with_latency(Duration::from_millis(2));
        let mut tasks = Vec::new();
        for _ in 0..8 {
            let provider = provider.clone();
            tasks.push(tokio::spawn(async move { provider.generate(&request()).await }));
        }
        let mut contents = Vec::new();
        for task in tasks {
            contents.push(task.await.unwrap().unwrap().content);
        }
        assert_eq!(contents.len(), 8);
        assert_eq!(provider.call_count(), 8);
        assert_eq!(contents.len(), contents.iter().collect::<std::collections::HashSet<_>>().len(), "each reply used once");
    }

    #[test]
    fn splitting_groups_words() {
        assert_eq!(split_words(""), Vec::<String>::new());
        assert_eq!(split_words("short"), vec!["short".to_string()]);
        let pieces = split_words("one two three four five six seven eight");
        assert!(pieces.len() > 1);
        assert_eq!(pieces.concat(), "one two three four five six seven eight");
    }
}
