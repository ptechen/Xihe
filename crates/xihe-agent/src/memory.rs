//! 记忆，接入智能体循环。
//!
//! 三件事把 [`xihe_memory`] 连到一个智能体，本模块持有全部：
//!
//! - [`MemoryService`]——作为世界资源的共享存储，使工具与循环
//!   能按类型触及它。[`crate::React`] 把它作为
//!   [`ToolContext::with_store`](crate::ToolContext::with_store) 传给每次工具调用。
//! - [`ChatSummariser`]——一个询问模型而非挑选句子的 [`Summariser`]，
//!   在折叠长对话时使用。
//! - [`recall_context`]——智能体关于某主题已知之事，渲染为循环
//!   可置于模型面前的一条系统消息。

use std::fmt::{self, Debug, Formatter};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use xihe_core::{Entity, World};
use xihe_llm::ChatClient;
use xihe_memory::{
    AgentMemory, ConversationWindow, MemoryError, MemoryQuery, MemoryResult, MemoryStore,
    ScoredRecord, Summariser,
};
use xihe_schedule::SystemContext;

use crate::replay::Journal;

/// 共享记忆存储，保留在世界中，使系统按类型找到它。
#[derive(Clone)]
pub struct MemoryService(Arc<dyn MemoryStore>);

impl MemoryService {
    /// 包装调用者已拥有的一个存储。
    pub fn new<S: MemoryStore + 'static>(store: S) -> Self {
        Self(Arc::new(store))
    }

    /// 把存储包装在 `Arc` 之后，例如与非 ECS 代码共享的那个。
    pub fn from_arc(store: Arc<dyn MemoryStore>) -> Self {
        Self(store)
    }

    /// 共享句柄背后的存储。
    pub fn store(&self) -> &Arc<dyn MemoryStore> {
        &self.0
    }

    /// 指向同一存储的第二个句柄。
    pub fn shared(&self) -> Arc<dyn MemoryStore> {
        Arc::clone(&self.0)
    }

    /// 存储自己的标识符，例如 `in-memory`。
    pub fn id(&self) -> String {
        self.0.id().to_string()
    }

    /// 把服务放入世界，返回先前的那个（若有）。
    pub fn install(&self, world: &mut World) -> Option<Self> {
        world.insert_resource(self.clone())
    }

    /// 从已加锁的世界中读出服务。
    pub fn from_world(world: &World) -> Option<Self> {
        world.resource::<Self>().cloned()
    }

    /// 通过系统上下文读取服务。
    pub async fn read(ctx: &SystemContext) -> Option<Self> {
        ctx.read_resource(|world| world.and_then(Self::from_world))
            .await
    }

    /// 此存储之上、限定到某智能体的工作记忆视图。
    pub fn working_memory(&self, owner: Entity) -> AgentMemory {
        AgentMemory::new(self.shared(), owner)
    }
}

impl Debug for MemoryService {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryService")
            .field("store", &self.0.id())
            .finish()
    }
}

/// 用一个真正的摘要器折叠对话。
///
/// [`ConversationWindow::fold`] 需要对窗口的 `&mut` 访问并 await
/// 摘要器，因此它不能在持有世界锁时运行。进行折叠的调用者——
/// 任务间清理、demo、测试——通过这里、在它们拥有的窗口上做。
pub async fn fold_window<S: Summariser + ?Sized>(
    window: &mut ConversationWindow,
    summariser: &S,
    keep_last: usize,
) -> MemoryResult<Option<String>> {
    window.fold(summariser, keep_last).await
}

/// 一次折叠模型调用的结果，供调用者记入 Observer。
#[derive(Clone, Copy, Debug, Default)]
pub struct FoldCall {
    /// 提示词 token。
    pub input_tokens: u64,
    /// 补全 token。
    pub output_tokens: u64,
    /// 模型调用是否失败（失败时摘要回退启发式）。
    pub failed: bool,
}

/// 摘要器与调用者共享的折叠调用报告。
///
/// 折叠即便在模型失败时仍会成功（启发式兜底），故调用者无法从折叠
/// 结果看出这次模型调用的成败与计量；摘要器把它写入此报告，
/// [`crate::React`] 取出后为每次折叠调用记一条 Observer 模型事件。
#[derive(Clone, Default)]
pub struct FoldReport {
    call: Arc<Mutex<Option<FoldCall>>>,
}

impl FoldReport {
    /// 一个空报告。
    pub fn new() -> Self {
        Self::default()
    }

    fn set(&self, call: FoldCall) {
        *self.call.lock().expect("fold report lock") = Some(call);
    }

    /// 取出最近一次折叠调用（若有），取出后清空。
    pub fn take(&self) -> Option<FoldCall> {
        self.call.lock().expect("fold report lock").take()
    }
}

/// 用一个聊天模型做摘要。
///
/// [`xihe_memory`] 中的启发式摘要器挑选它能打分的句子；
/// 这一个询问模型，后者更擅长保留对话中对任务重要的部分。
/// 当调用失败时，它回退到启发式而非丢弃折叠，因为无法折叠的
/// 窗口会无界增长。
///
/// 给定一个 [`Journal`] 与一个关联标识，摘要调用像任何其他模型
/// 调用一样被录制与回放：回放反馈录制的摘要而非再次询问模型，
/// 因此一次折叠的运行能逐轮复现。
pub struct ChatSummariser {
    client: Arc<ChatClient>,
    fallback: Box<dyn Summariser>,
    max_chars: usize,
    instructions: String,
    journal: Option<Journal>,
    correlation: Option<(String, usize)>,
    report: Option<FoldReport>,
}

impl ChatSummariser {
    /// 请 `client` 压缩转录。
    pub fn new(client: Arc<ChatClient>) -> Self {
        Self {
            client,
            fallback: Box::new(xihe_memory::HeuristicSummariser::new(6, 160)),
            max_chars: 600,
            instructions: "Compress the transcript into at most five sentences. \
                            Keep decisions, facts that were established, and anything \
                            still unresolved. Do not add commentary."
                .to_string(),
            journal: None,
            correlation: None,
            report: None,
        }
    }

    /// 限制摘要长度。
    #[must_use]
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = max_chars.max(32);
        self
    }

    /// 替换框定摘要请求的指令。
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// 通过 `journal` 录制与回放摘要调用。
    #[must_use]
    pub fn with_journal(mut self, journal: Journal) -> Self {
        self.journal = Some(journal);
        self
    }

    /// 用发起它的智能体与折叠索引标记摘要调用，
    /// 使折叠相同转录的两个智能体保持独立的日志键。
    #[must_use]
    pub fn with_correlation(mut self, agent: impl Into<String>, fold: usize) -> Self {
        self.correlation = Some((agent.into(), fold));
        self
    }

    /// 把每次折叠模型调用的成败与 token 计量写入 `report`。
    ///
    /// 折叠本身即便在模型失败时仍会成功（启发式兜底），因此调用者无法
    /// 从折叠结果看出这次模型调用；这个报告让 [`crate::React`] 能为每次
    /// 折叠调用——无论成败——记一条 Observer 模型事件，而非只留在
    /// tracing 日志里。
    #[must_use]
    pub fn with_report(mut self, report: FoldReport) -> Self {
        self.report = Some(report);
        self
    }

    /// 此摘要器询问的模型。
    pub fn model(&self) -> String {
        self.client.model().to_string()
    }
}

#[async_trait]
impl Summariser for ChatSummariser {
    async fn summarise(&self, transcript: &str) -> MemoryResult<String> {
        if transcript.trim().is_empty() {
            return Ok(String::new());
        }
        let mut request = xihe_llm::GenerationRequest::new(self.client.model())
            .with_system(self.instructions.clone())
            .with_user(transcript)
            .with_max_tokens(256);
        if let Some((agent, fold)) = &self.correlation {
            request = request.with_correlation(format!("{agent}::fold"), *fold);
        }
        // 录制的摘要被逐字反馈；只有未命中才询问模型，
        // 成功的调用被记入日志，使下一次回放能找到它。
        let response = if let Some(journal) = &self.journal
            && let Some(recorded) = journal.lookup(&request)
        {
            recorded
        } else {
            match self.client.generate(&request).await {
                Ok(response) => {
                    if let Some(journal) = &self.journal {
                        journal
                            .record(&request, &response)
                            .map_err(|error| MemoryError::config(error.to_string()))?;
                    }
                    response
                }
                Err(error) => {
                    if let Some(report) = &self.report {
                        report.set(FoldCall {
                            input_tokens: 0,
                            output_tokens: 0,
                            failed: true,
                        });
                    }
                    tracing::warn!(
                        reason = error.to_string(),
                        "the model could not summarise; falling back to sentence ranking"
                    );
                    return self.fallback.summarise(transcript).await;
                }
            }
        };
        if let Some(report) = &self.report {
            report.set(FoldCall {
                input_tokens: response.usage.prompt_tokens,
                output_tokens: response.usage.completion_tokens,
                failed: false,
            });
        }
        let text = response.content.trim().to_string();
        if text.is_empty() {
            return self.fallback.summarise(transcript).await;
        }
        Ok(clamp(&text, self.max_chars))
    }

    fn id(&self) -> &str {
        "chat-summariser"
    }
}

/// 召回存储关于 `query` 已知之事，作为模型可读的行。
///
/// 记录连同其类别被引用，使模型能区分事实与半记得的反思，
/// 整块是一条消息而非 `top_k` 条——一个系统轮比 `top_k` 个轮更省。
pub async fn recall_lines(
    store: &Arc<dyn MemoryStore>,
    query: &str,
    top_k: usize,
) -> MemoryResult<Vec<String>> {
    let hits: Vec<ScoredRecord> = store.search(&MemoryQuery::text(query).with_top_k(top_k)).await?;
    Ok(hits
        .iter()
        .map(|hit| {
            format!(
                "[{}] {}",
                hit.record.kind.as_str(),
                clamp(hit.record.text.trim(), 400)
            )
        })
        .collect())
}

/// 构建“你已知之事”系统消息，无匹配时为 `None`。
pub async fn recall_context(
    store: &Arc<dyn MemoryStore>,
    query: &str,
    top_k: usize,
) -> MemoryResult<Option<String>> {
    let lines = recall_lines(store, query, top_k).await?;
    if lines.is_empty() {
        return Ok(None);
    }
    Ok(Some(format!(
        "what is already known about this:\n{}",
        lines.join("\n")
    )))
}

fn clamp(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut cut: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    cut.push('…');
    cut
}

#[cfg(test)]
mod tests {
    use super::*;
    use xihe_llm::{MockProvider, Reply};
    use xihe_memory::{HeuristicSummariser, InMemoryStore, MemoryKind};

    fn store() -> Arc<dyn MemoryStore> {
        Arc::new(InMemoryStore::new())
    }

    #[tokio::test]
    async fn a_service_is_found_by_type_and_shares_one_store() {
        let mut world = World::new();
        let service = MemoryService::new(InMemoryStore::new());
        let other = service.clone();
        assert!(Arc::ptr_eq(&service.0, &other.0));
        assert!(service.install(&mut world).is_none());
        assert_eq!(
            MemoryService::from_world(&world).expect("installed").id(),
            service.id()
        );
        assert!(MemoryService::from_world(&World::new()).is_none());
        assert_eq!(format!("{service:?}"), format!("MemoryService {{ store: {:?} }}", service.id()));
    }

    #[tokio::test]
    async fn a_working_memory_is_scoped_to_its_owner() {
        let service = MemoryService::new(InMemoryStore::new());
        let owner = Entity::from_raw(7, 1);
        let memory = service.working_memory(owner);
        assert_eq!(memory.owner(), Some(owner));
    }

    #[tokio::test]
    async fn the_chat_summariser_asks_the_model() {
        let mock = MockProvider::text("they decided to fold the window.");
        let summariser = ChatSummariser::new(Arc::new(ChatClient::from_provider(mock.clone())));
        let summary = summariser
            .summarise("user: please fold\nassistant: folding now")
            .await
            .expect("a summary");
        assert_eq!(summary, "they decided to fold the window.");
        assert_eq!(mock.call_count(), 1);
        assert!(
            mock.last_request()
                .expect("a request")
                .last_user()
                .is_some_and(|text| text.contains("please fold"))
        );
    }

    #[tokio::test]
    async fn an_empty_transcript_is_not_worth_a_call() {
        let mock = MockProvider::text("x");
        let summariser = ChatSummariser::new(Arc::new(ChatClient::from_provider(mock.clone())));
        assert_eq!(summariser.summarise("   ").await.expect("empty"), "");
        assert_eq!(mock.call_count(), 0, "no transcript, no request");
    }

    #[tokio::test]
    async fn a_failing_model_falls_back_instead_of_failing_the_fold() {
        let client = Arc::new(ChatClient::from_provider(MockProvider::with_script([
            Reply::fail(xihe_llm::LlmError::Timeout {
                elapsed: std::time::Duration::from_millis(5),
            }),
        ])));
        let summariser = ChatSummariser::new(client);
        let summary = summariser
            .summarise("user: the export refreshes nightly\nassistant: noted")
            .await
            .expect("the fallback answers");
        assert!(
            summary.contains("export"),
            "the heuristic kept the salient line, got {summary}"
        );
    }

    #[tokio::test]
    async fn a_failed_model_call_reports_a_failed_fold() {
        let report = FoldReport::new();
        let client = Arc::new(ChatClient::from_provider(MockProvider::with_script([
            Reply::fail(xihe_llm::LlmError::decode("the summariser returned garbage")),
        ])));
        let summariser = ChatSummariser::new(client).with_report(report.clone());
        let summary = summariser
            .summarise("user: the export refreshes nightly\nassistant: noted")
            .await
            .expect("the fallback answers");
        assert!(!summary.is_empty(), "the heuristic still summarises");
        let call = report.take().expect("a fold call was reported");
        assert!(
            call.failed,
            "the model failure must be reported so the fold degradation is observable"
        );
    }

    #[tokio::test]
    async fn a_successful_model_call_reports_usage() {
        let report = FoldReport::new();
        let client = Arc::new(ChatClient::from_provider(MockProvider::text("a good summary")));
        let summariser = ChatSummariser::new(client).with_report(report.clone());
        summariser
            .summarise("user: hello\nassistant: hi")
            .await
            .expect("a summary");
        let call = report.take().expect("a fold call was reported");
        assert!(!call.failed, "a successful summariser call is not a degradation");
        assert!(
            call.output_tokens > 0,
            "a successful fold must carry its token usage, not be undercounted"
        );
    }

    #[tokio::test]
    async fn a_replayed_summary_reports_a_successful_fold() {
        let journal = Journal::new();
        let transcript = "user: the export refreshes nightly\nassistant: noted";

        let recorder = ChatSummariser::new(Arc::new(ChatClient::from_provider(
            MockProvider::text("recorded summary"),
        )))
        .with_correlation("scribe", 0)
        .with_journal(journal.clone());
        recorder.summarise(transcript).await.expect("recorded");

        // 回放命中必须胜过失败的提供方：不询问模型，也不误报降级。
        let report = FoldReport::new();
        let replayer = ChatSummariser::new(Arc::new(ChatClient::from_provider(
            MockProvider::with_script([Reply::fail(xihe_llm::LlmError::decode("down"))]),
        )))
        .with_correlation("scribe", 0)
        .with_journal(journal.clone())
        .with_report(report.clone());
        let summary = replayer.summarise(transcript).await.expect("replayed");
        assert_eq!(summary, "recorded summary");
        let call = report.take().expect("a replayed fold call was reported");
        assert!(!call.failed, "a replay hit is not a model failure");
    }

    #[tokio::test]
    async fn a_long_summary_is_cut() {
        let client = Arc::new(ChatClient::from_provider(MockProvider::text("word ".repeat(400))));
        let summariser = ChatSummariser::new(client).with_max_chars(64);
        let summary = summariser.summarise("anything").await.expect("a summary");
        assert!(summary.chars().count() <= 64, "{} chars", summary.chars().count());
        assert!(summary.ends_with('…'));
    }

    #[tokio::test]
    async fn folding_keeps_the_recent_turns() {
        let mut window = ConversationWindow::new().with_max_turns(4);
        for index in 0..6 {
            window.push_user(format!("turn {index}"));
        }
        let before = window.turns().len();
        let summary = fold_window(&mut window, &HeuristicSummariser::new(4, 80), 2)
            .await
            .expect("a summary")
            .expect("not empty");
        assert!(!summary.is_empty(), "{summary}");
        assert!(window.turns().len() < before, "the fold dropped older turns");
        assert!(window.fold_count() > 0);
        assert!(window.summary().is_some());
    }

    #[tokio::test]
    async fn recall_quotes_what_the_store_knows() {
        let store = store();
        store
            .remember(MemoryKind::Fact, "the export refreshes at 03:00")
            .await
            .expect("remembered");
        store
            .remember(MemoryKind::Observation, "the reader wants one line")
            .await
            .expect("remembered");
        let lines = recall_lines(&store, "when does the export refresh", 3)
            .await
            .expect("a search");
        assert!(!lines.is_empty());
        assert!(
            lines.iter().any(|line| line.contains("export")),
            "{lines:?}"
        );
        assert!(lines[0].starts_with('['), "{}", lines[0]);
        let context = recall_context(&store, "export refresh", 2)
            .await
            .expect("a context");
        assert!(context.expect("some hits").contains("already known"));
    }

    #[tokio::test]
    async fn an_empty_store_yields_no_context() {
        let store = store();
        assert!(recall_lines(&store, "anything", 3).await.expect("empty").is_empty());
        assert!(recall_context(&store, "anything", 3).await.expect("none").is_none());
    }

    #[tokio::test]
    async fn a_clipped_text_keeps_its_shape() {
        assert_eq!(clamp("short", 10), "short");
        assert_eq!(clamp("abcdef", 4), "abc…");
        assert_eq!(clamp("日本語のテキスト", 3), "日本…");
    }
}
