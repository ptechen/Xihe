//! # xihe-llm
//!
//! A provider-agnostic chat-completion layer for agent runtimes: a message and
//! tool-call model, one async [`Provider`] trait, a [`ChatClient`] that owns the
//! retry policy, and an offline [`MockProvider`] so agent logic is testable with
//! no network.
//!
//! The crate deliberately knows nothing about entities or systems — it is plain
//! data plus async calls, which is why [`xihe_agent`] can depend on it while
//! staying independently testable.
//!
//! ## Offline by default
//!
//! ```
//! use xihe_llm::prelude::*;
//!
//! # #[tokio::main]
//! async fn main() -> Result<(), LlmError> {
//! // A scripted provider: two calls, the first asking for a tool.
//! let provider = MockProvider::with_script([
//!     Reply::tool("lookup", serde_json::json!({ "city": "Hangzhou" })),
//!     Reply::text("It is 21°C in Hangzhou."),
//! ]);
//! let client = ChatClient::from_provider(provider.clone());
//!
//! let first = client
//!     .generate(&GenerationRequest::new("").with_user("weather?"))
//!     .await?;
//! assert_eq!(first.step(), GenerationStep::ExecuteTools);
//! assert_eq!(first.tool_calls[0].string_argument("city"), Some("Hangzhou"));
//!
//! // Feed the tool result back exactly as a real provider would receive it.
//! let second = client
//!     .generate(&GenerationRequest::new("")
//!         .with_user("weather?")
//!         .with_message(first.to_message())
//!         .with_message(ChatMessage::tool_result(&first.tool_calls[0].id, serde_json::json!("21°C"))))
//!     .await?;
//! assert_eq!(second.content, "It is 21°C in Hangzhou.");
//! assert_eq!(provider.call_count(), 2);
//! # Ok(())
//! # }
//! ```
//!
//! ## Streaming
//!
//! Every provider can be streamed; a provider without native streaming emits
//! one terminal chunk.
//!
//! ```
//! use futures::StreamExt;
//! use xihe_llm::prelude::*;
//!
//! # #[tokio::main]
//! async fn main() -> Result<(), LlmError> {
//! let client = ChatClient::from_provider(MockProvider::text("the quick brown fox"));
//! let request = GenerationRequest::new("mock-model").with_user("say something");
//! let mut stream = client.stream(&request).await?;
//! let mut text = String::new();
//! while let Some(chunk) = stream.next().await {
//!     text.push_str(&chunk?.text);
//! }
//! assert_eq!(text, "the quick brown fox");
//! # Ok(())
//! # }
//! ```
//!
//! ## Real providers
//!
//! Enable the `openai` feature to get [`OpenAiProvider`], which speaks the
//! `/chat/completions` dialect shared by OpenAI, DashScope compatibility mode,
//! Ollama, vLLM and LiteLLM. `OpenAiConfig::from_env()` reads `OPENAI_API_KEY`,
//! `OPENAI_BASE_URL` and `XIHE_MODEL`.
//!
//! ```ignore
//! let client = ChatClient::from_provider(OpenAiProvider::from_env()?);
//! let answer = client.ask("You are concise.", "What is ECS?").await?;
//! ```

mod client;
mod error;
mod message;
mod mock;
mod provider;
mod request;
pub mod wire;

pub use client::{ChatClient, RetryPolicy};
pub use error::LlmError;
pub use message::{
    ChatMessage, FinishReason, GenerationParams, GenerationResponse, GenerationStep, Role,
    StreamChunk, TokenUsage, ToolCall, ToolChoice, ToolDescriptor,
};
pub use mock::{Exhausted, MockProvider, Reply};
pub use provider::{Capabilities, DynProvider, Provider, StreamBox, collect_stream, one_shot};
pub use request::{Correlation, GenerationRequest, RequestPatch};
pub use wire::{error_from_status, parse_completion, parse_stream_event, to_wire};

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "openai")]
pub use openai::{OpenAiConfig, OpenAiProvider};

/// Rough token count: providers average about four characters per token, and
/// every message carries a little framing.
pub fn estimate_tokens(chars: usize) -> usize {
    chars.div_ceil(4) + 4
}

/// Everything a caller usually needs.
pub mod prelude {
    pub use super::{
        Capabilities, ChatClient, ChatMessage, DynProvider, Exhausted, FinishReason, GenerationParams,
        GenerationRequest, GenerationResponse, GenerationStep, LlmError, MockProvider, Provider,
        Reply, RequestPatch, RetryPolicy, Role, StreamBox, StreamChunk, TokenUsage, ToolCall,
        ToolChoice, ToolDescriptor, estimate_tokens,
    };
    #[cfg(feature = "openai")]
    pub use super::{OpenAiConfig, OpenAiProvider};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimation_is_monotonic_and_never_zero() {
        assert_eq!(estimate_tokens(0), 4);
        assert!(estimate_tokens(40) < estimate_tokens(400));
        assert!(estimate_tokens(1_000) >= 250);
    }

    #[test]
    fn facade_types_are_object_safe() {
        // Providers are handed around as `Arc<dyn Provider>` by the runtime.
        let provider: DynProvider = std::sync::Arc::new(MockProvider::text("ok"));
        assert_eq!(provider.id(), "mock");
        assert!(!provider.default_model().is_empty());
    }
}
