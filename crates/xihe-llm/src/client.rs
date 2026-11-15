//! `ChatClient`: a provider plus a retry policy and a default model.

use std::sync::Arc;
use std::time::Duration;

use crate::error::LlmError;
use crate::message::{ChatMessage, GenerationParams, GenerationResponse};
use crate::provider::{DynProvider, Provider, StreamBox};
use crate::request::GenerationRequest;

/// A spread in `0.0..1.0` derived from the wall clock's sub-second part.
///
/// Retry jitter only needs to decorrelate concurrent callers, so a cheap,
/// dependency-free source beats pulling in a random number crate.
fn sub_second_fraction() -> f32 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.subsec_nanos())
        .unwrap_or(0);
    (nanos % 1_000) as f32 / 1_000.0
}

/// How the client re-attempts a transient failure.
#[derive(Clone, Debug, PartialEq)]
pub struct RetryPolicy {
    /// Total attempts, including the first one.
    pub attempts: u32,
    /// Delay before the first retry.
    pub initial_backoff: Duration,
    /// Exponential growth of the base delay.
    pub factor: f32,
    /// Upper bound on any single delay.
    pub max_backoff: Duration,
    /// Random fraction (0.0..=1.0) added to each delay to avoid thundering herds.
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
    /// No retries: surface the first failure.
    pub const fn none() -> Self {
        Self {
            attempts: 1,
            initial_backoff: Duration::ZERO,
            factor: 1.0,
            max_backoff: Duration::ZERO,
            jitter: 0.0,
        }
    }

    /// Total attempts, including the first call.
    pub fn attempts(mut self, attempts: u32) -> Self {
        self.attempts = attempts.max(1);
        self
    }

    /// Set the initial retry delay.
    pub fn backoff(mut self, initial: Duration) -> Self {
        self.initial_backoff = initial;
        self
    }

    /// Randomised extra delay, as a fraction of the base (0.0..=1.0).
    pub fn with_jitter(mut self, jitter: f32) -> Self {
        self.jitter = jitter.clamp(0.0, 1.0);
        self
    }

    /// The delay before `attempt` (1-based: the delay after the first failure).
    pub fn delay_for(&self, attempt: u32) -> Duration {
        let exponent = attempt.saturating_sub(1);
        let base = self.initial_backoff.as_secs_f64() * f64::from(self.factor).powi(exponent as i32);
        // Cap in seconds before converting: a large attempt index overflows
        // Duration::from_secs_f64, which panics. min/max also fold a NaN or
        // negative base into [0, max_backoff], so the conversion is always safe.
        let capped = base.min(self.max_backoff.as_secs_f64()).max(0.0);
        let mut delay = Duration::from_secs_f64(capped);
        if self.jitter > 0.0 && !delay.is_zero() {
            let fraction = sub_second_fraction();
            delay = delay.mul_f32(1.0 + self.jitter * fraction);
        }
        delay
    }

    /// Retries available after the first attempt.
    pub fn retries(&self) -> u32 {
        self.attempts.saturating_sub(1)
    }
}

/// A provider with a home model name and a retry policy.
///
/// Agents hold a `ChatClient` rather than an `Arc<dyn Provider>` so the failure
/// handling is identical everywhere.
#[derive(Clone)]
pub struct ChatClient {
    provider: DynProvider,
    model: String,
    policy: RetryPolicy,
    params: Option<GenerationParams>,
}

impl ChatClient {
    /// Bind a provider to the model name used when a request omits one.
    pub fn new(provider: DynProvider, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            policy: RetryPolicy::default(),
            params: None,
        }
    }

    /// Adopt whatever default model the provider advertises.
    pub fn from_provider<P: Provider>(provider: P) -> Self {
        let arc: DynProvider = Arc::new(provider);
        let model = arc.default_model().to_string();
        Self::new(arc, model)
    }

    /// Replace the retry policy.
    pub fn with_policy(mut self, policy: RetryPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Sampling defaults applied when a request leaves them at their own default.
    pub fn with_params(mut self, params: GenerationParams) -> Self {
        self.params = Some(params);
        self
    }

    /// The underlying provider, for capability checks and direct streaming.
    pub fn provider(&self) -> &DynProvider {
        &self.provider
    }

    /// The provider's short name, e.g. for span attributes.
    pub fn provider_id(&self) -> &str {
        self.provider.id()
    }

    /// The default model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The retry policy in force.
    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }

    /// Prepare a request: fill the model, merge defaults, keep explicit choices.
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

    /// Generate with retries for transient failures.
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

    /// One-shot question answering: system prompt, user text, assistant string.
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

    /// Continue a conversation that is already assembled.
    pub async fn complete(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<GenerationResponse, LlmError> {
        self.generate(&GenerationRequest::new(self.model.clone()).with_messages(messages))
            .await
    }

    /// Streamed generation; retried the same way as [`ChatClient::generate`].
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

    /// Collect a stream into one response, useful when the caller cannot handle
    /// incremental output but the provider only streams.
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
        // 200ms * 2^199 overflows Duration; the cap must apply before the
        // conversion, or delay_for panics on a policy meant for a flaky server.
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
        // A zero server hint must override a huge local backoff: were the hint
        // ignored, the retry would sleep for 30s and time out here.
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
