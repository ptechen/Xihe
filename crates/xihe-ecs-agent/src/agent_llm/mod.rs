//! # xihe::agent_llm
//!
//! 面向智能体运行时、与提供方无关的聊天补全层：消息与工具调用模型、
//! 一个异步 [`Provider`] trait、持有重试策略的 [`ChatClient`]，
//! 以及离线的 [`MockProvider`]，使智能体逻辑无需网络即可测试。
//!
//! 本 crate 刻意不了解实体或系统——它只是纯数据加异步调用，
//! 这正是 [`crate::agent`] 能依赖它、同时保持可独立测试的原因。
//!
//! ## 默认离线
//!
//! ```
//! use xihe::agent_llm::prelude::*;
//!
//! # #[tokio::main]
//! async fn main() -> Result<(), LlmError> {
//! // 一个脚本化提供方：两次调用，第一次请求工具。
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
//! // 完全按真实提供方接收的方式把工具结果回传。
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
//! ## 流式
//!
//! 每个提供方都可流式读取；没有原生流式的提供方会发出一个终止块。
//!
//! ```
//! use futures::StreamExt;
//! use xihe::agent_llm::prelude::*;
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
//! ## 真实提供方
//!
//! 启用 `openai` feature 即可获得 [`OpenAiProvider`]，它使用 OpenAI、
//! DashScope 兼容模式、Ollama、vLLM 与 LiteLLM 共享的
//! `/chat/completions` 方言。`OpenAiConfig::from_env()` 读取 `OPENAI_API_KEY`、
//! `OPENAI_BASE_URL` 与 `XIHE_MODEL`。
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

/// 粗略的 token 计数：提供方平均每个 token 约四个字符，
/// 且每条消息都带一点框架开销。
pub fn estimate_tokens(chars: usize) -> usize {
    chars.div_ceil(4) + 4
}

/// 调用者通常需要的一切。
pub mod prelude {
    pub use super::{
        Capabilities, ChatClient, ChatMessage, DynProvider, Exhausted, FinishReason,
        GenerationParams, GenerationRequest, GenerationResponse, GenerationStep, LlmError,
        MockProvider, Provider, Reply, RequestPatch, RetryPolicy, Role, StreamBox, StreamChunk,
        TokenUsage, ToolCall, ToolChoice, ToolDescriptor, estimate_tokens,
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
        // 提供方由运行时以 `Arc<dyn Provider>` 的形式传递。
        let provider: DynProvider = std::sync::Arc::new(MockProvider::text("ok"));
        assert_eq!(provider.id(), "mock");
        assert!(!provider.default_model().is_empty());
    }
}
