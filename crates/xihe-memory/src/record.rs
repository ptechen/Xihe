//! What a memory looks like: kind, payload, provenance and scoring.

use serde::{Deserialize, Serialize};
use xihe_core::Entity;

use crate::embed::Embedding;

/// Identifier of a stored memory, assigned by the store.
pub type MemoryId = u64;

/// Milliseconds since the Unix epoch, the unit every timestamp in this crate uses.
pub type Timestamp = u64;

/// Wall-clock now, in [`Timestamp`] units.
pub fn now_ms() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

/// The sort of thing remembered, which drives retrieval filters and retention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MemoryKind {
    /// A turn or episode of dialogue.
    Conversation,
    /// A stable statement about the world or the user.
    Fact,
    /// A single perception, often from a tool.
    Observation,
    /// A summary derived from other memories.
    Reflection,
    /// The raw output of a tool call.
    ToolResult,
    /// Retrieved document text.
    Document,
}

impl MemoryKind {
    /// The kind's name in snake_case.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Conversation => "conversation",
            Self::Fact => "fact",
            Self::Observation => "observation",
            Self::Reflection => "reflection",
            Self::ToolResult => "tool_result",
            Self::Document => "document",
        }
    }

    /// Kinds worth keeping longest.
    pub fn is_durable(self) -> bool {
        matches!(self, Self::Fact | Self::Reflection)
    }

    /// Kinds produced during a single turn.
    pub fn is_ephemeral(self) -> bool {
        matches!(self, Self::Observation | Self::ToolResult)
    }
}

impl std::fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A remembered item plus everything needed to rank and expire it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Store-assigned identity.
    pub id: MemoryId,
    /// What kind of memory this is.
    pub kind: MemoryKind,
    /// The remembered text.
    pub text: String,
    /// Vector form of [`MemoryRecord::text`].
    pub embedding: Embedding,
    /// The agent entity that stored it, when there is one.
    #[serde(default)]
    pub owner: Option<Entity>,
    /// Conversation or task this memory belongs to.
    #[serde(default)]
    pub session: Option<String>,
    /// Free-form labels used for filtering.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Importance weight, normally `0.5..=2.0`.
    #[serde(default = "one")]
    pub salience: f64,
    /// When the memory was written.
    pub created_at_ms: Timestamp,
    /// When it was last retrieved.
    #[serde(default)]
    pub accessed_at_ms: Option<Timestamp>,
    /// How often it has been retrieved.
    #[serde(default)]
    pub access_count: u32,
    /// Hard expiry, independent of the retention policy.
    #[serde(default)]
    pub expires_at_ms: Option<Timestamp>,
}

fn one() -> f64 {
    1.0
}

impl MemoryRecord {
    /// A fresh record stamped with the current time.
    pub fn new(kind: MemoryKind, text: impl Into<String>, embedding: Embedding) -> Self {
        Self {
            id: 0,
            kind,
            text: text.into(),
            embedding,
            owner: None,
            session: None,
            tags: Vec::new(),
            salience: 1.0,
            created_at_ms: now_ms(),
            accessed_at_ms: None,
            access_count: 0,
            expires_at_ms: None,
        }
    }

    /// A record with a caller-supplied timestamp, for replays and tests.
    pub fn at(
        kind: MemoryKind,
        text: impl Into<String>,
        embedding: Embedding,
        created_at_ms: Timestamp,
    ) -> Self {
        Self {
            created_at_ms,
            ..Self::new(kind, text, embedding)
        }
    }

    /// Override the store-assigned id, for replays.
    pub fn with_id(mut self, id: MemoryId) -> Self {
        self.id = id;
        self
    }

    /// Attribute the record to an agent entity.
    pub fn with_owner(mut self, owner: Entity) -> Self {
        self.owner = Some(owner);
        self
    }

    /// Group the record under a conversation or task.
    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.session = Some(session.into());
        self
    }

    /// Add one filter label.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add several filter labels.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Weight the record in retrieval scoring.
    pub fn with_salience(mut self, salience: f64) -> Self {
        self.salience = salience;
        self
    }

    /// Expire after `ttl`, relative to creation.
    pub fn with_ttl(mut self, ttl: std::time::Duration) -> Self {
        self.expires_at_ms = Some(self.created_at_ms.saturating_add(ttl.as_millis() as u64));
        self
    }

    /// True when the record carries this exact tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|stored| stored == tag)
    }

    /// Record a retrieval, which is what makes frequently used memories stick.
    pub fn touch(&mut self, at: Timestamp) {
        self.accessed_at_ms = Some(at);
        self.access_count = self.access_count.saturating_add(1);
    }

    /// True once the hard expiry has passed.
    pub fn is_expired(&self, at: Timestamp) -> bool {
        self.expires_at_ms.is_some_and(|expiry| expiry <= at)
    }

    /// Age in milliseconds, clamped at zero for clock skew.
    pub fn age_ms(&self, at: Timestamp) -> Timestamp {
        at.saturating_sub(self.created_at_ms)
    }

    /// Time since the last retrieval, or the age if never retrieved.
    pub fn idle_ms(&self, at: Timestamp) -> Timestamp {
        let reference = self.accessed_at_ms.unwrap_or(self.created_at_ms);
        at.saturating_sub(reference)
    }

    /// Semantic relevance: cosine similarity, weighted by importance.
    pub fn relevance(&self, similarity: f64) -> f64 {
        similarity.max(0.0) * self.salience
    }

    /// Relevance with an exponential recency bonus.
    ///
    /// `half_life_ms` is how long it takes an otherwise-identical memory to lose
    /// half its recency weight; `0` disables the decay.
    pub fn scored(&self, similarity: f64, at: Timestamp, half_life_ms: u64) -> f64 {
        let base = self.relevance(similarity);
        if half_life_ms == 0 || base <= 0.0 {
            return base;
        }
        let age = self.age_ms(at) as f64;
        let decay = 0.5f64.powf(age / half_life_ms.max(1) as f64);
        // A little reinforcement for memories that keep getting used.
        let reinforcement = 1.0 + (self.access_count as f64).ln_1p() * 0.1;
        base * (0.5 + 0.5 * decay) * reinforcement
    }

    /// Truncate the text for display in logs and prompts.
    pub fn preview(&self, max_chars: usize) -> String {
        if self.text.chars().count() <= max_chars {
            return self.text.clone();
        }
        let mut out: String = self.text.chars().take(max_chars.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

/// A retrieval request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MemoryQuery {
    /// Text to embed when `vector` is absent.
    pub text: Option<String>,
    /// A pre-computed query vector.
    pub vector: Option<Embedding>,
    /// Allowed kinds; empty means all.
    pub kinds: Vec<MemoryKind>,
    /// Required tags; empty means any.
    pub tags: Vec<String>,
    /// Restrict to one agent entity.
    pub owner: Option<Entity>,
    /// Restrict to one session.
    pub session: Option<String>,
    /// Result cap.
    pub top_k: usize,
    /// Minimum score, in `0.0..=1.0` before salience weighting.
    pub min_score: f64,
    /// Include records past their expiry.
    pub include_expired: bool,
}

impl MemoryQuery {
    /// A text query, embedded by the store before matching.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            top_k: 5,
            min_score: 0.0,
            ..Self::default()
        }
    }

    /// A vector query, for callers that already embedded something.
    pub fn vector(vector: Embedding) -> Self {
        Self {
            vector: Some(vector),
            top_k: 5,
            min_score: 0.0,
            ..Self::default()
        }
    }

    /// Restrict the search to one kind.
    pub fn with_kind(mut self, kind: MemoryKind) -> Self {
        self.kinds.push(kind);
        self
    }

    /// Restrict the search to a set of kinds.
    pub fn with_kinds(mut self, kinds: impl IntoIterator<Item = MemoryKind>) -> Self {
        self.kinds.extend(kinds);
        self
    }

    /// Require one tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Restrict the search to one owner.
    pub fn with_owner(mut self, owner: Entity) -> Self {
        self.owner = Some(owner);
        self
    }

    /// Restrict the search to one session.
    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.session = Some(session.into());
        self
    }

    /// Return at most this many records.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Drop matches scoring below this.
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }

    /// Include records past their expiry.
    pub fn including_expired(mut self) -> Self {
        self.include_expired = true;
        self
    }

    /// Does this record pass the query's non-score filters?
    pub fn accepts(&self, record: &MemoryRecord, at: Timestamp) -> bool {
        if !self.include_expired && record.is_expired(at) {
            return false;
        }
        if !self.kinds.is_empty() && !self.kinds.contains(&record.kind) {
            return false;
        }
        if self
            .owner
            .is_some_and(|owner| record.owner != Some(owner))
        {
            return false;
        }
        if self.session.is_some() && record.session != self.session {
            return false;
        }
        self.tags.iter().all(|tag| record.has_tag(tag))
    }

    /// The similarity floor a record must clear to be returned at all.
    pub fn passes_score(&self, similarity: f64) -> bool {
        similarity >= self.min_score
    }
}

/// A record together with its retrieval score.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredRecord {
    /// The memory itself.
    pub record: MemoryRecord,
    /// Its score for this query, higher first.
    pub score: f64,
}

impl ScoredRecord {
    /// Build a scored record.
    pub fn new(record: MemoryRecord, score: f64) -> Self {
        Self { record, score }
    }

    /// Convenience for prompt assembly.
    pub fn text(&self) -> &str {
        &self.record.text
    }

    /// Sort by descending score with a stable tie-break on id.
    pub fn sort(results: &mut [Self]) {
        results.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.record.id.cmp(&b.record.id))
        });
    }
}

/// How a store trims itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetentionPolicy {
    /// Hard cap on records; the least valuable are dropped first.
    pub max_records: usize,
    /// Drop anything older than this, in milliseconds; `0` disables.
    pub max_age_ms: u64,
    /// Drop records scoring below this on `salience * reinforcement`.
    pub min_value: f64,
    /// Recency half-life used when scoring for eviction.
    pub half_life_ms: u64,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_records: 10_000,
            max_age_ms: 0,
            min_value: 0.0,
            half_life_ms: 60 * 60 * 1_000,
        }
    }
}

impl RetentionPolicy {
    /// A policy capped at `max_records`.
    pub const fn with_max_records(max_records: usize) -> Self {
        Self {
            max_records,
            max_age_ms: 0,
            min_value: 0.0,
            half_life_ms: 60 * 60 * 1_000,
        }
    }

    /// Keep memories for at most `ttl`.
    pub fn with_max_age(mut self, ttl: std::time::Duration) -> Self {
        self.max_age_ms = ttl.as_millis() as u64;
        self
    }

    /// Forget records whose salience and use cannot justify their space.
    pub const fn with_min_value(mut self, min_value: f64) -> Self {
        self.min_value = min_value;
        self
    }

    /// Long-run value of a record, independent of any query.
    pub fn value(&self, record: &MemoryRecord, at: Timestamp) -> f64 {
        let age = record.age_ms(at) as f64;
        let decay = if self.half_life_ms == 0 {
            1.0
        } else {
            0.5f64.powf(age / self.half_life_ms as f64)
        };
        let reinforcement = 1.0 + (record.access_count as f64).ln_1p() * 0.25;
        let durability = if record.kind.is_durable() { 1.5 } else { 1.0 };
        record.salience * reinforcement * durability * (0.25 + 0.75 * decay)
    }

    /// Decide which records to drop; pure, so the ordering is testable.
    pub fn plan_evictions(&self, records: &[MemoryRecord], at: Timestamp) -> Vec<MemoryId> {
        let mut stale: Vec<MemoryId> = records
            .iter()
            .filter(|record| {
                record.is_expired(at)
                    || (self.max_age_ms > 0 && record.age_ms(at) > self.max_age_ms)
                    || self.value(record, at) < self.min_value
            })
            .map(|record| record.id)
            .collect();
        if records.len() - stale.len() > self.max_records {
            let mut survivors: Vec<(f64, MemoryId)> = records
                .iter()
                .filter(|record| !stale.contains(&record.id))
                .map(|record| (self.value(record, at), record.id))
                .collect();
            survivors.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let keep = self.max_records;
            stale.extend(survivors.iter().skip(keep).map(|(_, id)| *id));
        }
        stale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashingEmbedder;

    fn record(kind: MemoryKind, text: &str, created_at_ms: Timestamp) -> MemoryRecord {
        MemoryRecord::at(
            kind,
            text,
            HashingEmbedder::new(64).features(text),
            created_at_ms,
        )
    }

    #[tokio::test]
    async fn kinds_describe_themselves() {
        assert_eq!(MemoryKind::ToolResult.as_str(), "tool_result");
        assert_eq!(
            serde_json::to_string(&MemoryKind::Conversation).unwrap(),
            "\"conversation\""
        );
        assert!(MemoryKind::Fact.is_durable());
        assert!(MemoryKind::Observation.is_ephemeral());
        assert!(!MemoryKind::Conversation.is_durable());
        assert_eq!(MemoryKind::Document.to_string(), "document");
    }

    /// An entity identified only by its index; tests do not need generations.
    fn owner_entity(index: u32) -> Entity {
        Entity::from_raw(index, 1)
    }

    #[test]
    fn builders_attach_provenance() {
        let entity = owner_entity(3);
        let record = record(MemoryKind::Fact, "the user likes rust", 1_000)
            .with_owner(entity)
            .with_session("chat-1")
            .with_tag("preference")
            .with_salience(1.5)
            .with_ttl(std::time::Duration::from_millis(5_000));
        assert_eq!(record.owner, Some(entity));
        assert_eq!(record.session.as_deref(), Some("chat-1"));
        assert!(record.has_tag("preference"));
        assert_eq!(record.expires_at_ms, Some(6_000));
        assert!(!record.is_expired(5_999));
        assert!(record.is_expired(6_000), "expiry is inclusive");
    }

    #[test]
    fn touch_reinforces_a_record() {
        let mut record = record(MemoryKind::Observation, "saw a cat", 1_000);
        record.touch(2_000);
        record.touch(3_000);
        assert_eq!(record.access_count, 2);
        assert_eq!(record.accessed_at_ms, Some(3_000));
        assert_eq!(record.idle_ms(4_000), 1_000);
        assert_eq!(record.age_ms(4_000), 3_000);
        assert_eq!(record.age_ms(0), 0, "clock skew must not underflow");
    }

    #[test]
    fn scoring_rewards_similarity_salience_and_recency() {
        let fresh = record(MemoryKind::Fact, "a", 10_000).with_salience(2.0);
        let dull = record(MemoryKind::Fact, "b", 10_000);
        assert!(fresh.relevance(0.5) > dull.relevance(0.5));
        assert_eq!(dull.relevance(-0.5), 0.0, "negative cosine is clipped");

        let older = record(MemoryKind::Fact, "c", 0);
        let scored_at = 100_000;
        assert!(
            fresh.scored(0.8, scored_at, 50_000) > older.scored(0.8, scored_at, 50_000),
            "the newer memory should win at equal similarity"
        );
        assert_eq!(
            older.scored(0.8, scored_at, 0),
            older.relevance(0.8),
            "half_life 0 disables decay"
        );

        let mut used = record(MemoryKind::Fact, "d", 0);
        for stamp in 1..=5 {
            used.touch(stamp);
        }
        assert!(
            used.scored(0.8, scored_at, 50_000) > older.scored(0.8, scored_at, 50_000),
            "frequent retrieval should compensate for age"
        );
    }

    #[test]
    fn previews_are_truncated() {
        let record = record(MemoryKind::Fact, "hello world, long text", 0);
        assert_eq!(record.preview(5), "hell…");
        assert_eq!(record.preview(1_000), "hello world, long text");
        assert_eq!(record.preview(0), "…");
    }

    #[test]
    fn queries_filter_before_scoring() {
        let entity = owner_entity(1);
        let fact = record(MemoryKind::Fact, "likes rust", 0)
            .with_owner(entity)
            .with_tag("preference")
            .with_session("s1");
        let tool = record(MemoryKind::ToolResult, "output", 0).with_tag("scratch");

        let query = MemoryQuery::text("rust").with_kind(MemoryKind::Fact);
        assert!(query.accepts(&fact, 0));
        assert!(!query.accepts(&tool, 0));

        let tagged = MemoryQuery::text("x").with_tag("preference");
        assert!(tagged.accepts(&fact, 0));
        assert!(!tagged.accepts(&tool, 0), "a tag filter is a conjunction");

        let owned = MemoryQuery::text("x").with_owner(entity);
        assert!(owned.accepts(&fact, 0));

        let other_owner = MemoryQuery::text("x").with_owner(owner_entity(99));
        assert!(!other_owner.accepts(&fact, 0));

        let session = MemoryQuery::text("x").with_session("other");
        assert!(!session.accepts(&fact, 0));

        let defaults = MemoryQuery::text("x");
        assert_eq!(defaults.top_k, 5);
        assert!(defaults.accepts(&tool, 0), "no filters means everything matches");
    }

    #[test]
    fn expiry_hides_records_unless_asked() {
        let short_lived = record(MemoryKind::Observation, "tick", 0)
            .with_ttl(std::time::Duration::from_millis(10));
        let query = MemoryQuery::text("tick");
        assert!(!query.accepts(&short_lived, 100));
        assert!(query.including_expired().accepts(&short_lived, 100));
    }

    #[test]
    fn scored_records_sort_by_value() {
        let mut results = vec![
            ScoredRecord::new(record(MemoryKind::Fact, "b", 0).with_id(2), 0.5),
            ScoredRecord::new(record(MemoryKind::Fact, "a", 0).with_id(1), 0.9),
            ScoredRecord::new(record(MemoryKind::Fact, "c", 0).with_id(3), 0.5),
        ];
        ScoredRecord::sort(&mut results);
        assert_eq!(
            results.iter().map(|hit| hit.record.id).collect::<Vec<_>>(),
            vec![1, 2, 3],
            "ties fall back to id so ordering is stable"
        );
        assert_eq!(results[0].text(), "a");
    }

    #[test]
    fn retention_drops_expired_then_least_valuable() {
        let at = 1_000_000;
        let mut records: Vec<MemoryRecord> = (0..5)
            .map(|index| {
                record(MemoryKind::Observation, &format!("o{index}"), index * 1_000)
                    .with_id(index as MemoryId)
            })
            .collect();
        records[0].expires_at_ms = Some(at - 1);
        let policy = RetentionPolicy::with_max_records(2);
        let evicted = policy.plan_evictions(&records, at);
        assert!(evicted.contains(&0), "the expired record goes first");
        assert_eq!(
            evicted.len(),
            3,
            "5 records, cap 2: three must go"
        );

        for record in &mut records {
            record.access_count = 10;
            record.expires_at_ms = None;
        }
        let kept = RetentionPolicy {
            max_records: 100,
            min_value: 0.5,
            ..RetentionPolicy::default()
        }
        .plan_evictions(&records, at);
        assert!(
            kept.is_empty(),
            "reinforced records clear the value floor"
        );
        let poor = RetentionPolicy {
            max_records: 100,
            min_value: 50.0,
            ..RetentionPolicy::default()
        }
        .plan_evictions(&records, at);
        assert_eq!(poor.len(), 5, "an unreachable floor evicts everything");
    }

    #[test]
    fn age_policy_measures_from_creation() {
        let policy = RetentionPolicy::default().with_max_age(std::time::Duration::from_millis(500));
        let young = record(MemoryKind::Fact, "young", 10_000).with_id(1);
        let old = record(MemoryKind::Fact, "old", 1_000).with_id(2);
        let evicted = policy.plan_evictions(&[young, old], 10_400);
        assert_eq!(evicted, vec![2], "only the record past its age limit goes");
    }

    #[test]
    fn durable_kinds_outlive_ephemeral_ones() {
        let policy = RetentionPolicy::default();
        let at = 10_000;
        let fact = record(MemoryKind::Fact, "f", at).with_id(1);
        let note = record(MemoryKind::ToolResult, "t", at).with_id(2);
        assert!(policy.value(&fact, at) > policy.value(&note, at));
        assert_eq!(
            RetentionPolicy {
                half_life_ms: 0,
                ..RetentionPolicy::default()
            }
            .value(&fact, 0),
            fact.salience * 1.5
        );
    }

    #[test]
    fn records_round_trip_through_json() {
        let original = record(MemoryKind::Reflection, "summary", 1_234)
            .with_id(9)
            .with_tag("daily")
            .with_owner(owner_entity(2));
        let text = serde_json::to_string(&original).unwrap();
        let back: MemoryRecord = serde_json::from_str(&text).unwrap();
        assert_eq!(back, original);
        assert_eq!(back.id, 9);
        assert_eq!(back.access_count, 0, "defaults survive the round trip");
    }
}
