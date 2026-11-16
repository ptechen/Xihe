//! 一个 OpenAI 兼容的聊天补全提供方，置于 `openai` feature 之后。
//!
//! 这里只有 HTTP 传输；请求与响应编码位于 [`crate::agent_llm::wire`]，
//! 即使没有此 feature 它也会被编译与测试。启用 `openai` 会引入
//! `reqwest`，因此工作区首次以该 feature 构建时需要注册表访问。

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::agent_llm::GenerationResponse;
use crate::agent_llm::error::LlmError;
use crate::agent_llm::message::StreamChunk;
use crate::agent_llm::provider::{Capabilities, Provider, StreamBox};
use crate::agent_llm::request::GenerationRequest;
use crate::agent_llm::wire::{
    error_from_status, parse_completion, parse_retry_after, parse_stream_event, take_sse_line,
    to_wire,
};

/// 在何处、以何种方式连接服务器。
#[derive(Clone, Debug)]
pub struct OpenAiConfig {
    /// 服务器根地址，不含 `/chat/completions` 后缀。
    pub base_url: String,
    /// Bearer 令牌；`None` 则以未鉴权方式发送请求。
    pub api_key: Option<String>,
    /// 在请求体中发送的模型名。
    pub model: String,
    /// 连接超时、非流式调用的总截止时间，以及流帧之间允许的空闲间隔。
    /// 停滞的连接会浮现为 [`LlmError::Timeout`]；
    /// 健康的长流不会被它截断。
    pub timeout: Duration,
    /// 额外头部，例如网关部署用的 `api-version`。
    pub headers: Vec<(String, String)>,
}

impl OpenAiConfig {
    /// 读取 `OPENAI_BASE_URL`（或 `XIHE_OPENAI_BASE_URL`）、`OPENAI_API_KEY`
    /// 与 `XIHE_MODEL`（或 `OPENAI_MODEL`）。
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

    /// 环境变量读取规则，单独拆出以便测试能喂入显式值——
    /// edition 2024 使 `set_var` 成为 unsafe，因此绝不触碰进程环境变量。
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

    /// 替换请求与流空闲超时。
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// 替换请求未指明模型时所发送的模型。
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// 在每个请求上发送一个额外头部。
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// 此 base URL 对应的聊天补全端点。
    pub fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

fn trim_slashes(value: &str) -> String {
    value.trim_end_matches('/').to_string()
}

/// 通过 HTTP 连接的聊天补全服务器。
#[derive(Clone)]
pub struct OpenAiProvider {
    config: Arc<OpenAiConfig>,
    http: reqwest::Client,
}

impl OpenAiProvider {
    /// 构建提供方，在各次调用间共享一个 HTTP 客户端。
    pub fn new(config: OpenAiConfig) -> Self {
        // 共享客户端上不设总截止时间：那也会限制 SSE 体读取，
        // 截断健康的长流。`generate` 为每个请求设置自己的总截止时间；
        // `stream` 则改为依赖空闲间隔。
        let http = reqwest::Client::builder()
            .connect_timeout(config.timeout)
            .build()
            .unwrap_or_default();
        Self {
            config: Arc::new(config),
            http,
        }
    }

    /// 从环境构建提供方。
    pub fn from_env() -> Result<Self, LlmError> {
        Ok(Self::new(OpenAiConfig::from_env()?))
    }

    /// 此提供方发送的配置。
    pub fn config(&self) -> &OpenAiConfig {
        &self.config
    }

    /// 替换此提供方上的模型，克隆不受影响。
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
        // 限制连接 + 头部，但不限制体：健康的长流必须比此截止时间
        // 活得更久，因此只有下面的空闲间隔来管控它。
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
        // SSE 读取器在自己的任务上运行：当消费者丢弃流时，
        // 发送失败、任务停止读取，这也是取消如何传播到连接的方式。
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

/// 将 mpsc 接收端转为 `StreamBox`；单独拆出以使类型强制转换清晰。
fn stream_channel_to_pin_box(receiver: mpsc::Receiver<Result<StreamChunk, LlmError>>) -> StreamBox {
    Box::pin(stream::unfold(receiver, move |mut receiver| async move {
        receiver.recv().await.map(|item| (item, receiver))
    }))
}

/// 对传输层失败分类，保持超时可区分。
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

/// 在响应头部仍可用、体尚未被消费时读取 `Retry-After` 提示。
fn retry_after_header(response: &reqwest::Response) -> Option<Duration> {
    response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(parse_retry_after)
}

/// 问一个问题并返回答案文本。
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
        assert_eq!(
            config.base_url, "http://127.0.0.1:1/v1",
            "trailing slash trimmed"
        );
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
        // `StreamBox` 不是 `Debug`，因此对结果做匹配而非 unwrap。
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
        // 没有服务器：失败仍必须指明传输层，而非 panic。
        let provider = OpenAiProvider::new(config());
        assert!(ask(&provider, "be brief", "hi").await.is_err());
    }

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    const SSE_HEAD: &[u8] =
        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
    const SSE_FRAME: &[u8] = b"data: {\"choices\":[{\"delta\":{\"content\":\"tok\"}}]}\n\n";

    /// 绑定一个一次性 HTTP 服务器：读尽请求头部，然后把套接字
    /// 交给 `respond`。返回可供提供方指向的 base URL。
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
            text, "toktoktoktoktoktoktoktoktoktok",
            "10 frames over ~600ms survive a 200ms idle timeout: no total deadline"
        );
    }

    #[tokio::test]
    async fn a_stalled_stream_times_out_on_the_idle_gap() {
        let base = spawn_server(|mut socket| async move {
            socket.write_all(SSE_HEAD).await.unwrap();
            socket.write_all(SSE_FRAME).await.unwrap();
            socket.flush().await.unwrap();
            // 保持连接打开但不发送：只有空闲间隔会触发。
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
        let first = stream
            .next()
            .await
            .unwrap()
            .expect("the first frame arrives");
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
