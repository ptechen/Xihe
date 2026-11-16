//! 提供方边界：一个异步 trait，置于 `Arc` 之后即对象安全。

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};

use crate::agent_llm::GenerationResponse;
use crate::agent_llm::error::LlmError;
use crate::agent_llm::message::{FinishReason, StreamChunk};
use crate::agent_llm::request::GenerationRequest;

/// 增量生成事件的装箱 `Send` 流。
pub type StreamBox = Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send + 'static>>;

/// 可在系统与 tokio 任务之间共享的任意提供方。
pub type DynProvider = Arc<dyn Provider>;

/// 一个模型后端：聊天补全，外加可选的流式。
///
/// 实现必须能廉价地按 `Arc` 克隆，并可安全地被多个智能体实体同时并发调用。
#[async_trait]
pub trait Provider: Send + Sync + 'static {
    /// 用于日志与指标的稳定标识符（`mock`、`openai`……）。
    fn id(&self) -> &str;

    /// 当请求的 `model` 为空时使用的模型。
    fn default_model(&self) -> &str;

    /// 请求下一个助手轮。
    async fn generate(&self, request: &GenerationRequest) -> Result<GenerationResponse, LlmError>;

    /// 以块流的形式请求下一轮。
    ///
    /// 默认实现委托给 [`Provider::generate`] 并发出一个块，
    /// 因此调用者可以流式读取任何只一次性作答的提供方。
    async fn stream(&self, request: &GenerationRequest) -> Result<StreamBox, LlmError> {
        let response = self.generate(request).await?;
        Ok(one_shot(response))
    }

    /// 运行时可据此分支、无需探测提供方的能力。
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }
}

/// 提供方声称支持的内容。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    /// 聊天补全接受工具定义。
    pub tools: bool,
    /// 提供方发出增量块。
    pub streaming: bool,
    /// 接受图像输入。
    pub vision: bool,
    /// 可请求结构化输出。
    pub json_mode: bool,
}

/// 将已完成的响应包装为单块流。
pub fn one_shot(response: GenerationResponse) -> StreamBox {
    let usage = response.usage;
    let chunks = vec![StreamChunk {
        text: response.content,
        tool_calls: response.tool_calls,
        finish_reason: Some(response.finish_reason.unwrap_or(FinishReason::Stop)),
        usage: Some(usage),
    }];
    Box::pin(stream::iter(chunks.into_iter().map(Ok)))
}

/// 将流收集为单个响应，遇到第一个错误即停止。
pub async fn collect_stream<S>(mut stream: S) -> Result<GenerationResponse, LlmError>
where
    S: Stream<Item = Result<StreamChunk, LlmError>> + Send + Unpin + 'static,
{
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk?);
    }
    Ok(StreamChunk::reduce(chunks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_llm::message::TokenUsage;
    use crate::agent_llm::mock::MockProvider;
    use futures::StreamExt;

    /// 只实现 `generate` 的提供方，用于测试默认的 `stream`。
    struct TextOnly;

    #[async_trait::async_trait]
    impl Provider for TextOnly {
        fn id(&self) -> &str {
            "text-only"
        }

        fn default_model(&self) -> &str {
            "text-only-model"
        }

        async fn generate(
            &self,
            _request: &GenerationRequest,
        ) -> Result<crate::agent_llm::GenerationResponse, LlmError> {
            Ok(crate::agent_llm::GenerationResponse::text("hello world"))
        }
    }

    #[tokio::test]
    async fn default_stream_emits_one_chunk() {
        let request = GenerationRequest::new("m").with_user("hi");
        let stream = TextOnly.stream(&request).await.expect("default impl");
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 1, "a non-streaming provider yields one chunk");
        assert_eq!(chunks[0].as_ref().unwrap().text, "hello world");
        assert!(chunks[0].as_ref().unwrap().is_last());
    }

    #[tokio::test]
    async fn collected_stream_rebuilds_the_response() {
        let provider = MockProvider::text("a) b) c)");
        let request = GenerationRequest::new("m").with_user("hi");
        let stream = provider.stream(&request).await.unwrap();
        let response = collect_stream(stream).await.unwrap();
        assert_eq!(response.content, "a) b) c)");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn one_shot_carries_usage_and_calls() {
        let response = GenerationResponse::text("done")
            .with_usage(TokenUsage::new(3, 4))
            .with_tool_calls(vec![crate::agent_llm::ToolCall::new(
                "c",
                "noop",
                serde_json::json!({}),
            )]);
        let stream = one_shot(response);
        let collected = futures::executor::block_on(collect_stream(stream)).unwrap();
        assert_eq!(collected.usage.total_tokens, 7);
        assert!(collected.calls_tools());
    }
}
