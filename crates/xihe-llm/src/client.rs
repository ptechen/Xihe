//! `ChatClient`：一个提供方加上重试策略与默认模型。

use std::sync::Arc;
use std::time::Duration;

use crate::error::LlmError;
use crate::message::{ChatMessage, GenerationParams, GenerationResponse};
use crate::provider::{DynProvider, Provider, StreamBox};
use crate::request::GenerationRequest;

/// 由挂钟的亚秒部分导出的 `0.0..1.0` 区间内的散布值。
///
/// 重试抖动只需让并发调用者去相关，因此一个廉价、无依赖的来源
/// 胜过引入一个随机数 crate。
fn sub_second_fraction() -> f32 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.subsec_nanos())
        .unwrap_or(0);
    (nanos % 1_000) as f32 / 1_000.0
}

/// 客户端如何重试瞬时失败。
#[derive(Clone, Debug, PartialEq)]
pub struct RetryPolicy {
    /// 总尝试次数，包含第一次。
    pub attempts: u32,
    /// 第一次重试前的延迟。
    pub initial_backoff: Duration,
    /// 基础延迟的指数增长。
    pub factor: f32,
    /// 任何单次延迟的上界。
    pub max_backoff: Duration,
    /// 加到每次延迟上的随机比例（0.0..=1.0），以避免惊群。
    pub jitter: f32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: 3,
            initial_backoff: Duration::from_millis(200),
            factor: 2.0,
            max_backoff: Duration::from_secs(5),
            jitter: 0.0,
        }
    }
}

impl RetryPolicy {
    /// 不重试：直接浮现第一次失败。
    pub const fn none() -> Self {
        Self {
            attempts: 1,
            initial_backoff: Duration::ZERO,
            factor: 1.0,
            max_backoff: Duration::ZERO,
            jitter: 0.0,
        }
    }

    /// 总尝试次数，包含第一次调用。
    pub fn attempts(mut self, attempts: u32) -> Self {
        self.attempts = attempts.max(1);
        self
    }

    /// 设置初始重试延迟。
    pub fn backoff(mut self, initial: Duration) -> Self {
        self.initial_backoff = initial;
        self
    }

    /// 随机化的额外延迟，作为基础值的比例（0.0..=1.0）。
    pub fn with_jitter(mut self, jitter: f32) -> Self {
        self.jitter = jitter.clamp(0.0, 1.0);
        self
    }

    /// `attempt` 之前的延迟（从 1 起：第一次失败后的延迟）。
    pub fn delay_for(&self, attempt: u32) -> Duration {
        let exponent = attempt.saturating_sub(1);
        let base = self.initial_backoff.as_secs_f64() * f64::from(self.factor).powi(exponent as i32);
        // 转换前先以秒为单位封顶：很大的尝试索引会让
        // Duration::from_secs_f64 溢出并 panic。min/max 还会把 NaN 或
        // 负的基础值折入 [0, max_backoff]，因此转换总是安全的。
        let capped = base.min(self.max_backoff.as_secs_f64()).max(0.0);
        let mut delay = Duration::from_secs_f64(capped);
        if self.jitter > 0.0 && !delay.is_zero() {
            let fraction = sub_second_fraction();
            delay = delay.mul_f32(1.0 + self.jitter * fraction);
        }
        delay
    }

    /// 第一次尝试之后可用的重试次数。
    pub fn retries(&self) -> u32 {
        self.attempts.saturating_sub(1)
    }
}

/// 带有归属模型名与重试策略的提供方。
///
/// 智能体持有 `ChatClient` 而非 `Arc<dyn Provider>`，
/// 使失败处理在各处都一致。
#[derive(Clone)]
pub struct ChatClient {
    provider: DynProvider,
    model: String,
    policy: RetryPolicy,
    params: Option<GenerationParams>,
}

impl ChatClient {
    /// 将提供方绑定到请求省略模型名时所用的模型名。
    pub fn new(provider: DynProvider, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            policy: RetryPolicy::default(),
            params: None,
        }
    }

    /// 采用提供方所宣称的默认模型。
    pub fn from_provider<P: Provider>(provider: P) -> Self {
        let arc: DynProvider = Arc::new(provider);
        let model = arc.default_model().to_string();
        Self::new(arc, model)
    }

    /// 替换重试策略。
    pub fn with_policy(mut self, policy: RetryPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// 当请求把采样参数留为其自身默认值时应用的采样默认值。
    pub fn with_params(mut self, params: GenerationParams) -> Self {
        self.params = Some(params);
        self
    }

    /// 底层提供方，用于能力检查与直接流式读取。
    pub fn provider(&self) -> &DynProvider {
        &self.provider
    }

    /// 提供方的短名，例如用于 span 属性。
    pub fn provider_id(&self) -> &str {
        self.provider.id()
    }

    /// 默认模型名。
    pub fn model(&self) -> &str {
        &self.model
    }

    /// 当前生效的重试策略。
    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }

    /// 准备请求：填充模型、合并默认值、保留显式选择。
    pub fn prepare(&self, request: &GenerationRequest) -> GenerationRequest {
        let mut prepared = request.clone();
        if prepared.model.is_empty() {
            prepared.model = self.model.clone();
        }
        if let Some(params) = &self.params
            && prepared.params == GenerationParams::default() {
            prepared.params = params.clone();
        }
        prepared
    }

    /// 生成，并对瞬时失败进行重试。
    pub async fn generate(
        &self,
        request: &GenerationRequest,
    ) -> Result<GenerationResponse, LlmError> {
        let prepared = self.prepare(request);
        let mut attempt = 1u32;
        loop {
            let result = self.provider.generate(&prepared).await;
            let Err(error) = result else {
                if attempt > 1 {
                    tracing::info!(attempts = attempt, "generation succeeded after retry");
                }
                return result;
            };
            if !error.is_retryable() || attempt >= self.policy.attempts {
                if attempt > 1 || error.is_retryable() {
                    return Err(LlmError::exhausted(attempt, error));
                }
                return Err(error);
            }
            let server_hint = error.retry_after();
            let delay = server_hint.unwrap_or_else(|| self.policy.delay_for(attempt));
            tracing::warn!(
                attempt,
                attempts = self.policy.attempts,
                kind = error.kind(),
                delay_ms = delay.as_millis() as u64,
                "retrying generation"
            );
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            attempt += 1;
        }
    }

    /// 一次性问答：系统提示词、用户文本、助手字符串。
    pub async fn ask(
        &self,
        system: impl Into<String>,
        question: impl Into<String>,
    ) -> Result<String, LlmError> {
        let request = GenerationRequest::new(self.model.clone())
            .with_message(ChatMessage::system(system))
            .with_message(ChatMessage::user(question));
        let response = self.generate(&request).await?;
        Ok(response.content)
    }

    /// 继续一段已组装好的对话。
    pub async fn complete(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<GenerationResponse, LlmError> {
        self.generate(&GenerationRequest::new(self.model.clone()).with_messages(messages))
            .await
    }

    /// 流式生成；重试方式与 [`ChatClient::generate`] 相同。
    pub async fn stream(
        &self,
        request: &GenerationRequest,
    ) -> Result<StreamBox, LlmError> {
        let prepared = self.prepare(request);
        let mut attempt = 1u32;
        loop {
            let result = self.provider.stream(&prepared).await;
            let Err(error) = result else {
                if attempt > 1 {
                    tracing::info!(attempts = attempt, "stream succeeded after retry");
                }
                return result;
            };
            if !error.is_retryable() || attempt >= self.policy.attempts {
                if attempt > 1 || error.is_retryable() {
                    return Err(LlmError::exhausted(attempt, error));
                }
                return Err(error);
            }
            let delay = error
                .retry_after()
                .unwrap_or_else(|| self.policy.delay_for(attempt));
            tracing::warn!(
                attempt,
                attempts = self.policy.attempts,
                kind = error.kind(),
                delay_ms = delay.as_millis() as u64,
                "retrying stream"
            );
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            attempt += 1;
        }
    }

    /// 将流收集为单个响应，适用于调用者无法处理增量输出、
    /// 但提供方只支持流式的情况。
    pub async fn generate_streamed(
        &self,
        request: &GenerationRequest,
    ) -> Result<GenerationResponse, LlmError> {
        let stream = self.stream(request).await?;
        crate::provider::collect_stream(stream).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockProvider, Reply};

    fn client(provider: MockProvider) -> ChatClient {
        ChatClient::from_provider(provider).with_policy(RetryPolicy {
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(2),
            ..RetryPolicy::default()
        })
    }

    #[tokio::test]
    async fn transient_failures_are_retried() {
        let provider = MockProvider::flaky(2, "unreachable");
        let calls = provider.clone();
        let client = client(provider);
        let response = client
            .generate(&GenerationRequest::new("").with_user("hi"))
            .await
            .expect("the third attempt succeeds");
        assert_eq!(response.content, "recovered");
        assert_eq!(calls.call_count(), 3);
    }

    #[tokio::test]
    async fn permanent_failures_stop_immediately() {
        let provider = MockProvider::with_script([Reply::fail(LlmError::Auth {
            message: "bad key".into(),
        })]);
        let calls = provider.clone();
        let error = client(provider)
            .generate(&GenerationRequest::new("m").with_user("hi"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), "auth", "the first error is reported verbatim");
        assert_eq!(calls.call_count(), 1, "auth failures must not be retried");
    }

    #[tokio::test]
    async fn exhausted_reports_the_attempt_count() {
        let provider = MockProvider::flaky(5, "unreachable");
        let client = ChatClient::from_provider(provider)
            .with_policy(RetryPolicy::default().attempts(2).backoff(Duration::ZERO));
        let error = client
            .generate(&GenerationRequest::new("m").with_user("hi"))
            .await
            .unwrap_err();
        let LlmError::Exhausted { attempts, last } = error else {
            panic!("expected Exhausted, got {error:?}");
        };
        assert_eq!(attempts, 2);
        assert_eq!(last.kind(), "transport");
    }

    #[tokio::test]
    async fn ask_and_complete_fill_the_model() {
        let provider = MockProvider::text("42");
        let calls = provider.clone();
        let client = ChatClient::from_provider(provider);
        assert_eq!(client.ask("be brief", "answer?").await.unwrap(), "42");
        let prepared = calls.last_request().unwrap();
        assert_eq!(prepared.model, "mock-model", "an empty model is filled in");
        assert_eq!(prepared.messages.len(), 2);
        let response = client
            .complete(vec![ChatMessage::user("continue")])
            .await
            .unwrap();
        assert_eq!(response.content, "42");
    }

    #[tokio::test]
    async fn an_explicit_model_wins_over_the_default() {
        let provider = MockProvider::text("ok");
        let calls = provider.clone();
        let client = ChatClient::from_provider(provider);
        client
            .generate(&GenerationRequest::new("specific-model").with_user("hi"))
            .await
            .unwrap();
        assert_eq!(calls.last_request().unwrap().model, "specific-model");
    }

    #[test]
    fn backoff_grows_and_caps() {
        let policy = RetryPolicy {
            initial_backoff: Duration::from_millis(100),
            factor: 2.0,
            max_backoff: Duration::from_millis(250),
            ..RetryPolicy::default()
        };
        assert_eq!(policy.delay_for(1), Duration::from_millis(100));
        assert_eq!(policy.delay_for(2), Duration::from_millis(200));
        assert_eq!(policy.delay_for(3), Duration::from_millis(250), "capped");
        assert_eq!(policy.delay_for(9), Duration::from_millis(250));
        assert_eq!(policy.retries(), 2);
    }

    #[test]
    fn a_large_attempt_index_stays_capped_without_overflowing() {
        let policy = RetryPolicy {
            attempts: 200,
            initial_backoff: Duration::from_millis(200),
            factor: 2.0,
            max_backoff: Duration::from_secs(5),
            ..RetryPolicy::default()
        };
        // 200ms * 2^199 会溢出 Duration；封顶必须在转换之前应用，
        // 否则 delay_for 会在一个面向不稳定服务器的策略上 panic。
        assert_eq!(policy.delay_for(200), Duration::from_secs(5));
    }

    #[test]
    fn jitter_stays_inside_its_band() {
        let policy = RetryPolicy {
            initial_backoff: Duration::from_millis(100),
            jitter: 0.5,
            max_backoff: Duration::from_secs(10),
            ..RetryPolicy::default()
        };
        for _ in 0..32 {
            let delay = policy.delay_for(1);
            assert!(delay >= Duration::from_millis(100) && delay < Duration::from_millis(151));
        }
    }

    #[test]
    fn no_policy_means_a_single_attempt() {
        let policy = RetryPolicy::none();
        assert_eq!(policy.attempts, 1);
        assert_eq!(policy.delay_for(1), Duration::ZERO);
    }

    #[tokio::test]
    async fn default_params_only_apply_when_untouched() {
        let provider = MockProvider::text("ok");
        let calls = provider.clone();
        let client = ChatClient::from_provider(provider).with_params(
            GenerationParams::default().with_temperature(0.1),
        );
        client
            .generate(&GenerationRequest::new("m").with_user("hi"))
            .await
            .unwrap();
        assert_eq!(calls.last_request().unwrap().params.temperature, 0.1);
        let explicit = GenerationRequest::new("m")
            .with_user("hi")
            .with_params(GenerationParams::default().with_temperature(0.9));
        client.generate(&explicit).await.unwrap();
        assert_eq!(
            calls.last_request().unwrap().params.temperature,
            0.9,
            "a request that set its own parameters is left alone"
        );
    }

    #[tokio::test]
    async fn streamed_output_reassembles() {
        let client = client(MockProvider::text("one two three four five six"));
        let response = client
            .generate_streamed(&GenerationRequest::new("m").with_user("hi"))
            .await
            .unwrap();
        assert_eq!(response.content, "one two three four five six");
    }

    #[tokio::test]
    async fn stream_retries_transient_failures() {
        let provider = MockProvider::flaky(2, "unreachable");
        let calls = provider.clone();
        let client = client(provider);
        let response = client
            .generate_streamed(&GenerationRequest::new("").with_user("hi"))
            .await
            .expect("the third attempt succeeds");
        assert_eq!(response.content, "recovered");
        assert_eq!(calls.call_count(), 3);
    }

    #[tokio::test]
    async fn stream_exhausted_reports_the_attempt_count() {
        let provider = MockProvider::flaky(5, "unreachable");
        let client = ChatClient::from_provider(provider)
            .with_policy(RetryPolicy::default().attempts(2).backoff(Duration::ZERO));
        let error = client
            .generate_streamed(&GenerationRequest::new("m").with_user("hi"))
            .await
            .unwrap_err();
        let LlmError::Exhausted { attempts, last } = error else {
            panic!("expected Exhausted, got {error:?}");
        };
        assert_eq!(attempts, 2);
        assert_eq!(last.kind(), "transport");
    }

    #[tokio::test]
    async fn stream_honors_the_server_retry_after() {
        // 为零的服务器提示必须覆盖巨大的本地退避：若忽略该提示，
        // 重试会睡眠 30s 并在此超时。
        let provider = MockProvider::with_script([
            Reply::fail(LlmError::RateLimited {
                retry_after: Some(Duration::ZERO),
            }),
            Reply::text("ok"),
        ]);
        let client = ChatClient::from_provider(provider).with_policy(RetryPolicy {
            attempts: 3,
            initial_backoff: Duration::from_secs(30),
            max_backoff: Duration::from_secs(30),
            ..RetryPolicy::default()
        });
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            client.generate_streamed(&GenerationRequest::new("m").with_user("hi")),
        )
        .await
        .expect("the server hint, not the local backoff, set the delay")
        .unwrap();
        assert_eq!(response.content, "ok");
    }
}
