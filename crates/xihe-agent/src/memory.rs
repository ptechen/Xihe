//! Memory, wired into the agent loop.
//!
//! Three things connect [`xihe_memory`] to an agent, and this module holds all
//! of them:
//!
//! - [`MemoryService`] — the shared store as a world resource, so tools and the
//!   loop can reach it by type. [`crate::React`] passes it to every tool call as
//!   [`ToolContext::with_store`](crate::ToolContext::with_store).
//! - [`ChatSummariser`] — a [`Summariser`] that asks the model instead of
//!   picking sentences, used when folding a long conversation.
//! - [`recall_context`] — what the agent knows about a topic, rendered as one
//!   system message the loop can put in front of the model.

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;

use async_trait::async_trait;
use xihe_core::{Entity, World};
use xihe_llm::ChatClient;
use xihe_memory::{
    AgentMemory, ConversationWindow, MemoryQuery, MemoryResult, MemoryStore,
    ScoredRecord, Summariser,
};
use xihe_schedule::SystemContext;

use crate::replay::Journal;

/// The shared memory store, kept in the world so systems find it by type.
#[derive(Clone)]
pub struct MemoryService(Arc<dyn MemoryStore>);

impl MemoryService {
    /// Wrap a store the caller already owns.
    pub fn new<S: MemoryStore + 'static>(store: S) -> Self {
        Self(Arc::new(store))
    }

    /// Wrap a store behind an `Arc`, e.g. one shared with non-ECS code.
    pub fn from_arc(store: Arc<dyn MemoryStore>) -> Self {
        Self(store)
    }

    /// The store behind the shared handle.
    pub fn store(&self) -> &Arc<dyn MemoryStore> {
        &self.0
    }

    /// A second handle to the same store.
    pub fn shared(&self) -> Arc<dyn MemoryStore> {
        Arc::clone(&self.0)
    }

    /// The store's own identifier, e.g. `in-memory`.
    pub fn id(&self) -> String {
        self.0.id().to_string()
    }

    /// Put the service in the world, returning any previous one.
    pub fn install(&self, world: &mut World) -> Option<Self> {
        world.insert_resource(self.clone())
    }

    /// Read the service out of a locked world.
    pub fn from_world(world: &World) -> Option<Self> {
        world.resource::<Self>().cloned()
    }

    /// Read the service through a system context.
    pub async fn read(ctx: &SystemContext) -> Option<Self> {
        ctx.read_resource(|world| world.and_then(Self::from_world))
            .await
    }

    /// An agent-scoped working memory view over this store.
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

/// Fold a conversation with a real summariser.
///
/// [`ConversationWindow::fold`] needs `&mut` access to the window and awaits the
/// summariser, so it cannot run while the world lock is held. Callers that fold
/// — a between-tasks cleanup, a demo, a test — do it through here, on a window
/// they own.
pub async fn fold_window<S: Summariser + ?Sized>(
    window: &mut ConversationWindow,
    summariser: &S,
    keep_last: usize,
) -> MemoryResult<Option<String>> {
    window.fold(summariser, keep_last).await
}

/// Summarise with a chat model.
///
/// The heuristic summariser in [`xihe_memory`] picks sentences it can score;
/// this one asks the model, which is better at keeping the parts of a
/// conversation that matter to the task. When the call fails, it falls back to
/// the heuristic rather than dropping the fold, because a window that cannot be
/// folded grows without bound.
///
/// Given a [`Journal`] and a correlation, the summarising call is recorded and
/// replayed like any other model call: a replay feeds back the recorded summary
/// instead of asking the model again, so a folded run reproduces turn for turn.
pub struct ChatSummariser {
    client: Arc<ChatClient>,
    fallback: Box<dyn Summariser>,
    max_chars: usize,
    instructions: String,
    journal: Option<Journal>,
    correlation: Option<(String, usize)>,
}

impl ChatSummariser {
    /// Ask `client` to compress transcripts.
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
        }
    }

    /// Cap the summary length.
    #[must_use]
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = max_chars.max(32);
        self
    }

    /// Replace the instruction that frames the summarising request.
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Record and replay the summarising call through `journal`.
    #[must_use]
    pub fn with_journal(mut self, journal: Journal) -> Self {
        self.journal = Some(journal);
        self
    }

    /// Tag the summarising call with the agent and fold index that raised it,
    /// so two agents folding identical transcripts keep separate journal keys.
    #[must_use]
    pub fn with_correlation(mut self, agent: impl Into<String>, fold: usize) -> Self {
        self.correlation = Some((agent.into(), fold));
        self
    }

    /// The model this summariser asks.
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
        // A recorded summary is fed back verbatim; only a miss asks the model,
        // and a successful call is journaled so the next replay finds it.
        let response = if let Some(journal) = &self.journal
            && let Some(recorded) = journal.lookup(&request)
        {
            recorded
        } else {
            match self.client.generate(&request).await {
                Ok(response) => {
                    if let Some(journal) = &self.journal {
                        journal.record(&request, &response);
                    }
                    response
                }
                Err(error) => {
                    tracing::warn!(
                        reason = error.to_string(),
                        "the model could not summarise; falling back to sentence ranking"
                    );
                    return self.fallback.summarise(transcript).await;
                }
            }
        };
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

/// Recall what the store knows about `query`, as lines the model can read.
///
/// Records are quoted with their kind so the model can tell a fact from a
/// half-remembered reflection, and the whole block is one message rather than
/// `top_k` of them — a system turn costs less than `top_k` turns.
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

/// Build the "what you already know" system message, or `None` when nothing matched.
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
