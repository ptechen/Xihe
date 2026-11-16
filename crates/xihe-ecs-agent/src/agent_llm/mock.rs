//! 一个脚本化的离线提供方，用于测试、演示与确定性回放。

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream;
use serde_json::Value;

use crate::agent_llm::error::LlmError;
use crate::agent_llm::message::{
    ChatMessage, FinishReason, GenerationResponse, StreamChunk, TokenUsage, ToolCall,
};
use crate::agent_llm::provider::{Capabilities, Provider};
use crate::agent_llm::request::GenerationRequest;

/// 来自 [`MockProvider`] 的一个脚本化答案。
#[derive(Clone, Debug)]
pub enum Reply {
    /// 纯助手文本。
    Text(String),
    /// 助手文本加上它希望执行的工具调用。
    Tools {
        /// 伴随这些调用的助手文本。
        content: String,
        /// 要执行的调用。
        calls: Vec<ToolCall>,
    },
    /// 一个完整指定的响应，用于辅助方法无法表达的边缘情形。
    Response(GenerationResponse),
    /// 让调用失败，例如用于演练重试策略。
    Fail(LlmError),
}

impl Reply {
    /// 一个纯文本回复。
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// 一个不带伴随文本的工具调用。
    pub fn tool(name: impl Into<String>, arguments: impl Into<Value>) -> Self {
        Self::Tools {
            content: String::new(),
            calls: vec![ToolCall::new("", name, arguments)],
        }
    }

    /// 多个不带伴随文本的工具调用。
    pub fn tools(calls: Vec<ToolCall>) -> Self {
        Self::Tools {
            content: String::new(),
            calls,
        }
    }

    /// 一个注入的失败。
    pub fn fail(error: LlmError) -> Self {
        Self::Fail(error)
    }

    /// 此回复将产出的助手消息。
    pub fn response(&self) -> Option<GenerationResponse> {
        match self {
            Self::Text(text) => Some(GenerationResponse::text(text.clone())),
            Self::Tools { content, calls } => {
                Some(GenerationResponse::text(content.clone()).with_tool_calls(calls.clone()))
            }
            Self::Response(response) => Some(response.clone()),
            Self::Fail(_) => None,
        }
    }
}

/// 脚本耗尽后模拟提供方的行为。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Exhausted {
    /// 持续用最后一个脚本回复作答，使长运行能终止。
    RepeatLast,
    /// 从头回放脚本，用于节奏与循环测试。
    Cycle,
    /// 以 [`LlmError::Config`] 失败，使测试能察觉多余的调用。
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

/// 一个确定性提供方，从脚本作答并记录请求。
///
/// 克隆会共享脚本与日志，因此交给一个系统、又从另一个系统查看的
/// `Arc<MockProvider>` 看到的是同一份状态。
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
    /// 一个带空脚本的模拟提供方。
    pub fn new() -> Self {
        Self::with_script(Vec::new())
    }

    /// 按顺序取用的答案；一旦耗尽，由 [`MockProvider::exhausted`] 决定。
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

    /// 返回 `text` 的一次性模拟提供方。
    pub fn text(text: impl Into<String>) -> Self {
        Self::with_script([Reply::text(text)])
    }

    /// 无论被调用多少次都总是以相同文本作答的模拟提供方。
    pub fn echo(text: impl Into<String>) -> Self {
        Self::with_script([Reply::text(text)]).with_exhausted(Exhausted::Cycle)
    }

    /// 第一次调用时以 `arguments` 请求 `tool`、随后以 `final_text`
    /// 作答的模拟提供方——大多数 ReAct 测试所需的形态。
    pub fn tool_then_answer(
        tool: impl Into<String>,
        args: impl Into<Value>,
        final_text: impl Into<String>,
    ) -> Self {
        Self::with_script([Reply::tool(tool, args), Reply::text(final_text)])
    }

    /// 先失败 `failures_before` 次然后成功的模拟提供方。
    pub fn flaky(failures_before: u32, message: impl Into<String>) -> Self {
        let message = message.into();
        let script = std::iter::repeat_n(
            Reply::fail(LlmError::transport(format!("{message} (injected)"))),
            failures_before as usize,
        )
        .chain(std::iter::once(Reply::text("recovered")));
        Self::with_script(script).with_exhausted(Exhausted::Cycle)
    }

    /// 覆盖所宣称的默认模型名。
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// 每次调用的人为延迟，用于超时与并发测试。
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = latency;
        self
    }

    /// 选择脚本耗尽后发生什么。
    pub fn with_exhausted(mut self, behaviour: Exhausted) -> Self {
        self.exhausted = behaviour;
        self
    }

    /// 当脚本为空且 `exhausted` 为 `RepeatLast` 时使用的回复。
    pub fn with_fallback(mut self, reply: Reply) -> Self {
        self.fallback = reply;
        self
    }

    /// 向脚本末尾追加一个回复。
    pub fn push(&self, reply: Reply) {
        self.lock().script.push_back(reply);
    }

    /// 每次调用的延迟。
    pub fn latency(&self) -> Duration {
        self.latency
    }

    /// 迄今收到的每个请求，最旧的在前。
    pub fn requests(&self) -> Vec<GenerationRequest> {
        self.lock().requests.clone()
    }

    /// 最近的请求（若有）。
    pub fn last_request(&self) -> Option<GenerationRequest> {
        self.lock().requests.last().cloned()
    }

    /// 收到的请求总数，包含失败。
    pub fn call_count(&self) -> usize {
        self.lock().requests.len()
    }

    /// 有多少次调用以注入的错误作答。
    pub fn failure_count(&self) -> usize {
        self.lock().failures
    }

    /// 模拟提供方被问到的对话，展平以便断言。
    pub fn transcript(&self) -> Vec<ChatMessage> {
        self.lock()
            .requests
            .last()
            .map(|request| request.messages.clone())
            .unwrap_or_default()
    }

    /// 丢弃脚本与记录的历史。
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
        // 守卫绝不跨越 await 点持有；参见 `pick` 与 `record`。
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// 取下一个脚本回复，且不在睡眠期间持有锁。
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
                    // `record` 统计失败；这里只挑选回复。
                    return (Reply::Fail(LlmError::config("mock script exhausted")), None);
                }
                Exhausted::RepeatLast => {
                    state.last.clone().unwrap_or_else(|| self.fallback.clone())
                }
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

    /// 填入 id、用量与模型名，使回复看起来像提供方产出的。
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
                crate::agent_llm::estimate_tokens(response.content.len()) as u64,
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
    ) -> Result<crate::agent_llm::provider::StreamBox, LlmError> {
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
        let provider =
            MockProvider::with_script([Reply::text("only")]).with_exhausted(Exhausted::RepeatLast);
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
        assert_eq!(
            provider.generate(&request()).await.unwrap().content,
            "again"
        );
        assert_eq!(
            provider.generate(&request()).await.unwrap().content,
            "again"
        );
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn tool_replies_get_synthetic_ids_and_usage() {
        let provider =
            MockProvider::tool_then_answer("search", serde_json::json!({ "q": "ecs" }), "done");
        let first = provider.generate(&request()).await.unwrap();
        assert_eq!(first.step(), crate::agent_llm::GenerationStep::ExecuteTools);
        assert_eq!(first.tool_calls[0].id, "call_1_1");
        assert_eq!(first.tool_calls[0].string_argument("q"), Some("ecs"));
        assert!(
            !first.usage.is_zero(),
            "usage is synthesised for budget tests"
        );
        let second = provider.generate(&request()).await.unwrap();
        assert_eq!(second.step(), crate::agent_llm::GenerationStep::Answer);
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
        assert_eq!(
            provider.generate(&request()).await.unwrap().content,
            "recovered"
        );
        assert_eq!(provider.call_count(), 3, "failures are still calls");
    }

    #[tokio::test]
    async fn streamed_text_arrives_in_words_then_a_terminal_chunk() {
        let provider = MockProvider::text("alpha beta gamma delta epsilon");
        let stream = provider.stream(&request()).await.unwrap();
        let chunks: Vec<StreamChunk> = stream.map(Result::unwrap).collect().await;
        assert!(chunks.len() > 2, "the text should be split up");
        assert!(chunks.last().unwrap().is_last());
        let joined = chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<String>();
        assert_eq!(joined, "alpha beta gamma delta epsilon");
    }

    #[tokio::test]
    async fn push_extends_a_running_script() {
        let provider = MockProvider::with_script([Reply::text("first")]);
        provider.push(Reply::text("second"));
        assert_eq!(
            provider.generate(&request()).await.unwrap().content,
            "first"
        );
        assert_eq!(
            provider.generate(&request()).await.unwrap().content,
            "second"
        );
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
            (0..8)
                .map(|index| Reply::text(format!("reply {index}")))
                .collect::<Vec<_>>(),
        )
        .with_latency(Duration::from_millis(2));
        let mut tasks = Vec::new();
        for _ in 0..8 {
            let provider = provider.clone();
            tasks.push(tokio::spawn(
                async move { provider.generate(&request()).await },
            ));
        }
        let mut contents = Vec::new();
        for task in tasks {
            contents.push(task.await.unwrap().unwrap().content);
        }
        assert_eq!(contents.len(), 8);
        assert_eq!(provider.call_count(), 8);
        assert_eq!(
            contents.len(),
            contents
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            "each reply used once"
        );
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
