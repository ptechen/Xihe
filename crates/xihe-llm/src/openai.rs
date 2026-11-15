//! An OpenAI-compatible chat-completions provider, behind the `openai` feature.
//!
//! Only the HTTP transport lives here; request and response encoding is in
//! [`crate::wire`], which is compiled and tested even without this feature.
//! Enabling `openai` pulls in `reqwest`, so the workspace needs registry access
//! the first time it is built with that feature.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::error::LlmError;
use crate::message::StreamChunk;
use crate::provider::{Capabilities, Provider, StreamBox};
use crate::request::GenerationRequest;
use crate::wire::{
    error_from_status, parse_completion, parse_retry_after, parse_stream_event, take_sse_line,
    to_wire,
};
use crate::GenerationResponse;

/// Where and how to reach the server.
#[derive(Clone, Debug)]
pub struct OpenAiConfig {
    /// Server root, without the `/chat/completions` suffix.
    pub base_url: String,
    /// Bearer token; `None` sends the request unauthenticated.
    pub api_key: Option<String>,
    /// Model name sent in the request body.
    pub model: String,
    /// The connect timeout, the total deadline for a non-streaming call, and the
    /// idle gap allowed between stream frames. A stalled connection surfaces as
    /// [`LlmError::Timeout`]; a healthy long stream is not cut off by it.
    pub timeout: Duration,
    /// Extra headers, e.g. an `api-version` for gateway deployments.
    pub headers: Vec<(String, String)>,
}

impl OpenAiConfig {
    /// Read `OPENAI_BASE_URL` (or `XIHE_OPENAI_BASE_URL`), `OPENAI_API_KEY` and
    /// `XIHE_MODEL` (or `OPENAI_MODEL`).
    pub fn from_env() -> Result<Self, LlmError> {
        let read = |keys: &[&str]| -> Option<String> {
            keys.iter()
                .find_map(|key| std::env::var(key).ok())
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        };
        Self::build(
            read(&["XIHE_OPENAI_BASE_URL", "OPENAI_BASE_URL"]).as_deref(),
            read(&["OPENAI_API_KEY"]).as_deref(),
            read(&["XIHE_MODEL", "OPENAI_MODEL"]).as_deref(),
        )
    }

    /// The env-reading rules, split out so tests can feed explicit values —
    /// edition 2024 makes `set_var` unsafe, so process env is never touched.
    pub fn build(
        base_url: Option<&str>,
        api_key: Option<&str>,
        model: Option<&str>,
    ) -> Result<Self, LlmError> {
        let api_key = api_key.map(str::to_string);
        if api_key.is_none() {
            return Err(LlmError::config(
                "OPENAI_API_KEY is not set; export it or use MockProvider for offline runs",
            ));
        }
        Ok(Self {
            base_url: trim_slashes(base_url.unwrap_or("https://api.openai.com/v1")),
            api_key,
            model: model.unwrap_or("gpt-4o-mini").to_string(),
            timeout: Duration::from_secs(120),
            headers: Vec::new(),
        })
    }

    /// Replace the request and stream-idle timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Replace the model sent when a request does not name one.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Send an extra header on every request.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// The chat-completions endpoint for this base URL.
    pub fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

fn trim_slashes(value: &str) -> String {
    value.trim_end_matches('/').to_string()
}

/// A chat-completions server reached over HTTP.
#[derive(Clone)]
pub struct OpenAiProvider {
    config: Arc<OpenAiConfig>,
    http: reqwest::Client,
}

impl OpenAiProvider {
    /// Build a provider, sharing one HTTP client across calls.
    pub fn new(config: OpenAiConfig) -> Self {
        // No total deadline on the shared client: it would also bound the SSE
        // body read and cut off a healthy long stream. `generate` sets its own
        // total deadline per request; `stream` relies on the idle gap instead.
        let http = reqwest::Client::builder()
            .connect_timeout(config.timeout)
            .build()
            .unwrap_or_default();
        Self {
            config: Arc::new(config),
            http,
        }
    }

    /// Build a provider from the environment.
    pub fn from_env() -> Result<Self, LlmError> {
        Ok(Self::new(OpenAiConfig::from_env()?))
    }

    /// The configuration this provider sends.
    pub fn config(&self) -> &OpenAiConfig {
        &self.config
    }

    /// Replace the model on this provider, leaving clones unaffected.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let mut config = self.config.as_ref().clone();
        config.model = model.into();
        self.config = Arc::new(config);
        self
    }

    fn request(&self, body: Value) -> reqwest::RequestBuilder {
        let mut builder = self
            .http
            .post(self.config.completions_url())
            .header("content-type", "application/json")
            .json(&body);
        if let Some(key) = &self.config.api_key {
            builder = builder.header("authorization", format!("Bearer {key}"));
        }
        for (name, value) in &self.config.headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn id(&self) -> &str {
        "openai"
    }

    fn default_model(&self) -> &str {
        &self.config.model
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            tools: true,
            streaming: true,
            vision: true,
            json_mode: true,
        }
    }

    async fn generate(&self, request: &GenerationRequest) -> Result<GenerationResponse, LlmError> {
        let mut body = to_wire(request, &self.config.model);
        body["stream"] = json!(false);
        let started = Instant::now();
        let response = self
            .request(body)
            .timeout(self.config.timeout)
            .send()
            .await
            .map_err(|error| map_send_error(error, started))?;
        let status = response.status().as_u16();
        let retry_after = retry_after_header(&response);
        let text = response
            .text()
            .await
            .map_err(|error| LlmError::transport(error.to_string()))?;
        if !(200..300).contains(&status) {
            return Err(error_from_status(status, text, retry_after));
        }
        let value: Value =
            serde_json::from_str(&text).map_err(|error| LlmError::decode(error.to_string()))?;
        tracing::debug!(
            model = %self.config.model,
            millis = started.elapsed().as_millis() as u64,
            "chat completion finished"
        );
        parse_completion(value)
    }

    async fn stream(&self, request: &GenerationRequest) -> Result<StreamBox, LlmError> {
        let mut body = to_wire(request, &self.config.model);
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });
        // Bound connection + headers, but not the body: a long healthy stream
        // must outlive this deadline, so only the idle gap below governs it.
        let response = tokio::time::timeout(self.config.timeout, self.request(body).send())
            .await
            .map_err(|_| LlmError::Timeout {
                elapsed: self.config.timeout,
            })?
            .map_err(|error| map_send_error(error, Instant::now()))?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let retry_after = retry_after_header(&response);
            let text = response.text().await.unwrap_or_default();
            return Err(error_from_status(status, text, retry_after));
        }
        // The SSE reader runs on its own task: when the consumer drops the stream
        // the send fails and the task stops reading, which is also how cancellation
        // propagates to the connection.
        let (sender, receiver) = mpsc::channel::<Result<StreamChunk, LlmError>>(32);
        let idle_timeout = self.config.timeout;
        let mut bytes = response.bytes_stream();
        tokio::spawn(async move {
            let mut buffer = String::new();
            loop {
                let frame = match tokio::time::timeout(idle_timeout, bytes.next()).await {
                    Err(_) => {
                        let _ = sender
                            .send(Err(LlmError::Timeout {
                                elapsed: idle_timeout,
                            }))
                            .await;
                        return;
                    }
                    Ok(None) => return,
                    Ok(Some(Err(error))) => {
                        let _ = sender
                            .send(Err(LlmError::transport(error.to_string())))
                            .await;
                        return;
                    }
                    Ok(Some(Ok(chunk))) => chunk,
                };
                buffer.push_str(&String::from_utf8_lossy(&frame));
                while let Some(line) = take_sse_line(&mut buffer) {
                    if line == "[DONE]" {
                        return;
                    }
                    match parse_stream_event(&line) {
                        Ok(Some(chunk)) => {
                            if sender.send(Ok(chunk)).await.is_err() {
                                return;
                            }
                        }
                        Ok(None) => {}
                        Err(error) => {
                            let _ = sender.send(Err(error)).await;
                            return;
                        }
                    }
                }
            }
        });
        Ok(stream_channel_to_pin_box(receiver))
    }
}

/// Turn the mpsc half into a `StreamBox`; kept separate so the coercion is clear.
fn stream_channel_to_pin_box(
    receiver: mpsc::Receiver<Result<StreamChunk, LlmError>>,
) -> StreamBox {
    Box::pin(stream::unfold(receiver, move |mut receiver| async move {
        receiver.recv().await.map(|item| (item, receiver))
    }))
}

/// Classify a transport-level failure, keeping timeouts distinguishable.
fn map_send_error(error: reqwest::Error, started: Instant) -> LlmError {
    if error.is_timeout() {
        return LlmError::Timeout {
            elapsed: started.elapsed(),
        };
    }
    if error.is_builder() {
        return LlmError::config(format!("request could not be built: {error}"));
    }
    LlmError::transport(error.to_string())
}

/// Read the `Retry-After` hint while the response headers are still available,
/// before the body is consumed.
fn retry_after_header(response: &reqwest::Response) -> Option<Duration> {
    response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(parse_retry_after)
}

/// Ask one question and return the answer text.
pub async fn ask(
    provider: &OpenAiProvider,
    system: &str,
    question: &str,
) -> Result<String, LlmError> {
    let request = GenerationRequest::new(provider.default_model().to_string())
        .with_system(system)
        .with_user(question);
    Ok(provider.generate(&request).await?.content)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> OpenAiConfig {
        OpenAiConfig::build(Some("http://127.0.0.1:1/v1/"), Some("test-key"), Some("m"))
            .unwrap()
            .with_timeout(Duration::from_millis(200))
    }

    #[test]
    fn configuration_is_validated_before_any_call() {
        assert!(
            OpenAiConfig::build(None, None, None).is_err(),
            "a missing key is a configuration error, not a silent 401 later"
        );
        let config = config();
        assert_eq!(config.base_url, "http://127.0.0.1:1/v1", "trailing slash trimmed");
        assert_eq!(
            config.completions_url(),
            "http://127.0.0.1:1/v1/chat/completions"
        );
        let defaults = OpenAiConfig::build(None, Some("k"), None).unwrap();
        assert_eq!(defaults.base_url, "https://api.openai.com/v1");
        assert_eq!(defaults.model, "gpt-4o-mini");
    }

    #[test]
    fn headers_and_model_are_overridable() {
        let provider = OpenAiProvider::new(
            config()
                .with_header("x-trace", "abc")
                .with_model("qwen-max"),
        );
        assert_eq!(provider.default_model(), "qwen-max");
        assert_eq!(
            provider.config().headers,
            vec![("x-trace".to_string(), "abc".to_string())]
        );
        assert_eq!(provider.id(), "openai");
        assert!(provider.capabilities().streaming);
        assert!(provider.capabilities().tools);
    }

    #[tokio::test]
    async fn an_unreachable_server_is_a_retryable_transport_failure() {
        let provider = OpenAiProvider::new(config());
        let error = provider
            .generate(&GenerationRequest::new("m").with_user("hi"))
            .await
            .unwrap_err();
        assert!(
            error.is_retryable(),
            "connection refused must be retried: {error}"
        );
        assert_eq!(error.root().kind(), "transport");
    }

    #[tokio::test]
    async fn streaming_an_unreachable_server_fails_before_the_stream_starts() {
        let provider = OpenAiProvider::new(config());
        // `StreamBox` is not `Debug`, so the result is matched rather than unwrapped.
        let error = match provider
            .stream(&GenerationRequest::new("m").with_user("hi"))
            .await
        {
            Ok(_) => panic!("an unreachable server must fail before the stream starts"),
            Err(error) => error,
        };
        assert_eq!(error.root().kind(), "transport");
    }

    #[tokio::test]
    async fn ask_uses_the_provider_default_model() {
        // No server: the failure must still name the transport, not a panic.
        let provider = OpenAiProvider::new(config());
        assert!(ask(&provider, "be brief", "hi").await.is_err());
    }

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    const SSE_HEAD: &[u8] =
        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
    const SSE_FRAME: &[u8] = b"data: {\"choices\":[{\"delta\":{\"content\":\"tok\"}}]}\n\n";

    /// Bind a one-shot HTTP server: drain the request headers, then hand the
    /// socket to `respond`. Returns the base URL to point a provider at.
    async fn spawn_server<F, Fut>(respond: F) -> String
    where
        F: FnOnce(TcpStream) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let mut seen = Vec::new();
            loop {
                let n = socket.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                seen.extend_from_slice(&buf[..n]);
                if seen.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            respond(socket).await;
        });
        format!("http://{addr}/v1")
    }

    #[tokio::test]
    async fn a_long_healthy_stream_outlives_the_idle_timeout() {
        let base = spawn_server(|mut socket| async move {
            socket.write_all(SSE_HEAD).await.unwrap();
            for _ in 0..10 {
                socket.write_all(SSE_FRAME).await.unwrap();
                socket.flush().await.unwrap();
                tokio::time::sleep(Duration::from_millis(60)).await;
            }
            socket.write_all(b"data: [DONE]\n\n").await.unwrap();
            socket.flush().await.unwrap();
        })
        .await;
        let provider = OpenAiProvider::new(
            OpenAiConfig::build(Some(&base), Some("k"), Some("m"))
                .unwrap()
                .with_timeout(Duration::from_millis(200)),
        );
        let mut stream = provider
            .stream(&GenerationRequest::new("m").with_user("hi"))
            .await
            .unwrap();
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            text.push_str(&chunk.expect("a healthy long stream must not error").text);
        }
        assert_eq!(
            text,
            "toktoktoktoktoktoktoktoktoktok",
            "10 frames over ~600ms survive a 200ms idle timeout: no total deadline"
        );
    }

    #[tokio::test]
    async fn a_stalled_stream_times_out_on_the_idle_gap() {
        let base = spawn_server(|mut socket| async move {
            socket.write_all(SSE_HEAD).await.unwrap();
            socket.write_all(SSE_FRAME).await.unwrap();
            socket.flush().await.unwrap();
            // Hold the connection open without sending: only the idle gap trips.
            tokio::time::sleep(Duration::from_millis(600)).await;
        })
        .await;
        let provider = OpenAiProvider::new(
            OpenAiConfig::build(Some(&base), Some("k"), Some("m"))
                .unwrap()
                .with_timeout(Duration::from_millis(150)),
        );
        let mut stream = provider
            .stream(&GenerationRequest::new("m").with_user("hi"))
            .await
            .unwrap();
        let first = stream.next().await.unwrap().expect("the first frame arrives");
        assert_eq!(first.text, "tok");
        let error = stream
            .next()
            .await
            .unwrap()
            .expect_err("the stall must surface as an error");
        assert_eq!(
            error.root().kind(),
            "timeout",
            "the idle gap, not a total deadline: {error}"
        );
    }
}
