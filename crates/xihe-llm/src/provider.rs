//! The provider boundary: one async trait, object-safe behind an `Arc`.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};

use crate::error::LlmError;
use crate::message::{FinishReason, StreamChunk};
use crate::request::GenerationRequest;
use crate::GenerationResponse;

/// A boxed, `Send` stream of incremental generation events.
pub type StreamBox = Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send + 'static>>;

/// Any provider shareable across systems and tokio tasks.
pub type DynProvider = Arc<dyn Provider>;

/// A model backend: chat completion plus, optionally, streaming.
///
/// Implementations must be cheap to clone-by-`Arc` and safe to call
/// concurrently from several agent entities at once.
#[async_trait]
pub trait Provider: Send + Sync + 'static {
    /// Stable identifier used in logs and metrics (`mock`, `openai`, …).
    fn id(&self) -> &str;

    /// Model used when a request leaves `model` empty.
    fn default_model(&self) -> &str;

    /// Ask for the next assistant turn.
    async fn generate(&self, request: &GenerationRequest) -> Result<GenerationResponse, LlmError>;

    /// Ask for the next turn as a stream of chunks.
    ///
    /// The default delegates to [`Provider::generate`] and emits one chunk, so a
    /// caller can stream any provider that only answers in one shot.
    async fn stream(&self, request: &GenerationRequest) -> Result<StreamBox, LlmError> {
        let response = self.generate(request).await?;
        Ok(one_shot(response))
    }

    /// Capabilities a runtime can branch on without probing the provider.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }
}

/// What a provider claims to support.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    /// Chat completions accept tool definitions.
    pub tools: bool,
    /// The provider emits incremental chunks.
    pub streaming: bool,
    /// Image inputs are accepted.
    pub vision: bool,
    /// Structured output can be requested.
    pub json_mode: bool,
}

/// Wrap a finished response as a single-chunk stream.
pub fn one_shot(response: GenerationResponse) -> StreamBox {
    let usage = response.usage;
    let chunks = vec![
        StreamChunk {
            text: response.content,
            tool_calls: response.tool_calls,
            finish_reason: Some(response.finish_reason.unwrap_or(FinishReason::Stop)),
            usage: Some(usage),
        },
    ];
    Box::pin(stream::iter(chunks.into_iter().map(Ok)))
}

/// Collect a stream into a single response, stopping at the first error.
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
    use crate::message::TokenUsage;
    use crate::mock::MockProvider;
    use futures::StreamExt;

    /// A provider that implements only `generate`, to test the default `stream`.
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
        ) -> Result<crate::GenerationResponse, LlmError> {
            Ok(crate::GenerationResponse::text("hello world"))
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
            .with_tool_calls(vec![crate::ToolCall::new("c", "noop", serde_json::json!({}))]);
        let stream = one_shot(response);
        let collected = futures::executor::block_on(collect_stream(stream)).unwrap();
        assert_eq!(collected.usage.total_tokens, 7);
        assert!(collected.calls_tools());
    }
}
