//! Record-and-replay for a world's model calls and tool results.
//!
//! [`Journal`] captures every [`GenerationResponse`] a run produces, keyed by a
//! deterministic fingerprint of the request that asked for it, along with the
//! tool results that followed. [`ReplayProvider`] answers from that journal
//! instead of a live model, and the loop feeds back the recorded tool results
//! instead of re-running them, so a second run reproduces the first turn for
//! turn even when the original provider was non-deterministic and the tools had
//! side effects.
//!
//! The fingerprint covers the correlation, the messages and the tool descriptors
//! (sorted by name), not the model name: the client fills the model in after the
//! request is built, so keying on it would differ between recording and replay.
//! A journal that already holds a request's tool results replays them; an empty
//! one records them, so the same resource handles both directions. Rounds that
//! park on a human approval still run live. To replay a run, install the
//! populated [`Journal`] as a world resource and back the
//! [`LlmService`](crate::LlmService) with a [`ReplayProvider`] that shares it.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use xihe_llm::{GenerationRequest, GenerationResponse, LlmError, Provider, ToolCall};

use crate::tool::ToolOutcome;

/// A shared log of model responses and tool results, stored as a world resource.
///
/// Cloning a journal shares the same entries, so the handle installed in the
/// recording world and the one handed to [`ReplayProvider`] see one log.
#[derive(Clone, Default)]
pub struct Journal {
    entries: Arc<Mutex<HashMap<String, GenerationResponse>>>,
    tools: Arc<Mutex<HashMap<String, ToolRecord>>>,
}

/// The tool results that followed one model response.
#[derive(Clone, Serialize, Deserialize)]
struct ToolRecord {
    results: Vec<(ToolCall, ToolOutcome)>,
    refused: usize,
}

/// The on-disk form of a journal. `BTreeMap` keeps the byte layout deterministic
/// (independent of the live maps' hash order), so two recordings of the same run
/// serialize identically and journals can be diffed or audited.
#[derive(Serialize, Deserialize)]
struct JournalData {
    entries: BTreeMap<String, GenerationResponse>,
    tools: BTreeMap<String, ToolRecord>,
}

impl Journal {
    /// An empty journal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `response` under the fingerprint of `request`.
    pub fn record(&self, request: &GenerationRequest, response: &GenerationResponse) {
        self.entries
            .lock()
            .expect("journal lock")
            .insert(fingerprint(request), response.clone());
    }

    /// The response recorded for `request`, if any.
    pub fn lookup(&self, request: &GenerationRequest) -> Option<GenerationResponse> {
        self.entries
            .lock()
            .expect("journal lock")
            .get(&fingerprint(request))
            .cloned()
    }

    /// Store the tool results that followed `request`.
    pub fn record_tools(
        &self,
        request: &GenerationRequest,
        results: Vec<(ToolCall, ToolOutcome)>,
        refused: usize,
    ) {
        self.tools
            .lock()
            .expect("journal lock")
            .insert(fingerprint(request), ToolRecord { results, refused });
    }

    /// The tool results recorded for `request`, if any.
    pub fn lookup_tools(
        &self,
        request: &GenerationRequest,
    ) -> Option<(Vec<(ToolCall, ToolOutcome)>, usize)> {
        self.tools
            .lock()
            .expect("journal lock")
            .get(&fingerprint(request))
            .map(|record| (record.results.clone(), record.refused))
    }

    /// How many distinct requests have a recorded model response.
    pub fn len(&self) -> usize {
        self.entries.lock().expect("journal lock").len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize the whole journal to bytes, so a recording can be persisted and
    /// replayed in another process. The bytes are self-contained: the fingerprint
    /// keys travel with the responses, so [`Journal::from_bytes`] needs nothing
    /// else to replay the run.
    pub fn to_bytes(&self) -> serde_json::Result<Vec<u8>> {
        let entries: BTreeMap<String, GenerationResponse> = self
            .entries
            .lock()
            .expect("journal lock")
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let tools: BTreeMap<String, ToolRecord> = self
            .tools
            .lock()
            .expect("journal lock")
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        serde_json::to_vec(&JournalData { entries, tools })
    }

    /// Rebuild a journal from [`Journal::to_bytes`] output. The result shares no
    /// state with the journal that produced the bytes, exactly as a second
    /// process loading the recording from disk would.
    pub fn from_bytes(bytes: &[u8]) -> serde_json::Result<Self> {
        let data: JournalData = serde_json::from_slice(bytes)?;
        Ok(Self {
            entries: Arc::new(Mutex::new(data.entries.into_iter().collect())),
            tools: Arc::new(Mutex::new(data.tools.into_iter().collect())),
        })
    }

    /// Persist the journal to a file so a recording can be replayed in another
    /// process. A thin wrapper over [`Journal::to_bytes`]; serialization errors
    /// surface as [`std::io::ErrorKind::InvalidData`].
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let bytes = self
            .to_bytes()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        std::fs::write(path, bytes)
    }

    /// Load a journal written by [`Journal::save`]. The result shares no state
    /// with the recorder, exactly as a second process reading the file would.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    }
}

/// A stable key for a request: its correlation (when present), its messages,
/// and its tool descriptors sorted by name so descriptor order cannot change the
/// key. The model name is deliberately excluded (see the module docs). The
/// correlation keeps two agents issuing identical requests apart.
fn fingerprint(request: &GenerationRequest) -> String {
    let mut tools = request.tools.clone();
    tools.sort_by(|a, b| a.name.cmp(&b.name));
    serde_json::json!({
        "correlation": request.correlation,
        "messages": request.messages,
        "tools": tools,
    })
    .to_string()
}

/// A provider that answers from a [`Journal`] instead of a live model.
///
/// A request with no recorded response is an error, not a silent fallback: a
/// replay that drifts from the recording should fail loudly.
pub struct ReplayProvider {
    journal: Journal,
}

impl ReplayProvider {
    /// Replay from `journal`.
    pub fn new(journal: Journal) -> Self {
        Self { journal }
    }
}

#[async_trait]
impl Provider for ReplayProvider {
    fn id(&self) -> &str {
        "replay"
    }

    fn default_model(&self) -> &str {
        "replay"
    }

    async fn generate(&self, request: &GenerationRequest) -> Result<GenerationResponse, LlmError> {
        self.journal.lookup(request).ok_or_else(|| {
            LlmError::unsupported("replay", "generate: no recorded response for this request")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xihe_llm::{ChatMessage, ToolDescriptor};

    fn request(text: &str) -> GenerationRequest {
        GenerationRequest::new("m").with_message(ChatMessage::user(text))
    }

    #[test]
    fn a_cloned_journal_shares_one_log() {
        let journal = Journal::new();
        let other = journal.clone();
        journal.record(&request("hi"), &GenerationResponse::text("hello"));
        assert_eq!(other.len(), 1);
        assert_eq!(
            other.lookup(&request("hi")).expect("recorded").content,
            "hello"
        );
        assert!(other.lookup(&request("absent")).is_none());
    }

    #[test]
    fn the_fingerprint_ignores_model_and_tool_order() {
        let mut a = request("hi");
        a.tools = vec![
            ToolDescriptor::new("zebra", "z"),
            ToolDescriptor::new("alpha", "a"),
        ];
        let mut b = request("hi");
        b.model = "a-different-model".into();
        b.tools = vec![
            ToolDescriptor::new("alpha", "a"),
            ToolDescriptor::new("zebra", "z"),
        ];
        assert_eq!(fingerprint(&a), fingerprint(&b));
    }

    #[test]
    fn different_messages_get_different_keys() {
        assert_ne!(fingerprint(&request("hi")), fingerprint(&request("bye")));
    }

    #[test]
    fn correlation_separates_identical_requests() {
        let alpha = request("hi").with_correlation("alpha", 0);
        let beta = request("hi").with_correlation("beta", 0);
        assert_ne!(
            fingerprint(&alpha),
            fingerprint(&beta),
            "two agents issuing the same request must not share a journal key"
        );
        let alpha_again = request("hi").with_correlation("alpha", 0);
        assert_eq!(fingerprint(&alpha), fingerprint(&alpha_again));
    }

    #[test]
    fn tool_results_round_trip_under_their_request_key() {
        let journal = Journal::new();
        let req = request("hi").with_correlation("ops", 0);
        assert!(journal.lookup_tools(&req).is_none());
        let results = vec![(
            ToolCall::new("c1", "deploy", serde_json::json!({})),
            ToolOutcome::ok(serde_json::json!({"shipped": true}), "shipped"),
        )];
        journal.record_tools(&req, results, 1);
        let (got, refused) = journal.lookup_tools(&req).expect("recorded");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0.name, "deploy");
        assert_eq!(got[0].1.text, "shipped");
        assert_eq!(refused, 1);
        assert!(
            journal
                .lookup_tools(&request("hi").with_correlation("other", 0))
                .is_none(),
            "another agent's identical request must not see these results"
        );
    }

    #[test]
    fn a_journal_round_trips_through_bytes() {
        let journal = Journal::new();
        let req = request("hi").with_correlation("ops", 0);
        journal.record(&req, &GenerationResponse::text("hello"));
        journal.record_tools(
            &req,
            vec![(
                ToolCall::new("c1", "deploy", serde_json::json!({})),
                ToolOutcome::ok(serde_json::json!({"shipped": true}), "shipped"),
            )],
            1,
        );

        let bytes = journal.to_bytes().expect("serialize");
        let restored = Journal::from_bytes(&bytes).expect("deserialize");
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.lookup(&req).expect("response").content, "hello");
        let (results, refused) = restored.lookup_tools(&req).expect("tools");
        assert_eq!(results[0].1.text, "shipped");
        assert_eq!(refused, 1);
        assert!(restored.lookup(&request("absent")).is_none());
    }

    #[test]
    fn a_journal_round_trips_through_a_file() {
        let journal = Journal::new();
        let req = request("hi").with_correlation("ops", 0);
        journal.record(&req, &GenerationResponse::text("hello"));

        let path =
            std::env::temp_dir().join(format!("xihe-journal-unit-{}.json", std::process::id()));
        journal.save(&path).expect("save");
        let restored = Journal::load(&path).expect("load");
        std::fs::remove_file(&path).ok();

        assert_eq!(restored.lookup(&req).expect("response").content, "hello");
    }

    #[tokio::test]
    async fn replay_answers_from_the_journal_and_fails_on_a_miss() {
        let journal = Journal::new();
        journal.record(&request("hi"), &GenerationResponse::text("hello"));
        let provider = ReplayProvider::new(journal);
        assert_eq!(provider.id(), "replay");
        let answered = provider.generate(&request("hi")).await.expect("recorded");
        assert_eq!(answered.content, "hello");
        let missed = provider.generate(&request("absent")).await;
        assert!(missed.is_err(), "an unrecorded request must fail loudly");
    }
}