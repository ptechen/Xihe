//! 记忆长什么样：类别、载荷、来源与评分。

use crate::ecs_core::Entity;
use serde::{Deserialize, Serialize};

use crate::agent_memory::embed::Embedding;

/// 已存储记忆的标识符，由存储分配。
pub type MemoryId = u64;

/// 自 Unix 纪元起的毫秒数，本 crate 每个时间戳所用的单位。
pub type Timestamp = u64;

/// 挂钟当前时刻，以 [`Timestamp`] 为单位。
pub fn now_ms() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

/// 被记住之物的类别，它驱动检索过滤与保留。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MemoryKind {
    /// 一轮或一段对话。
    Conversation,
    /// 关于世界或用户的稳定陈述。
    Fact,
    /// 单次感知，常来自工具。
    Observation,
    /// 从其他记忆派生的摘要。
    Reflection,
    /// 工具调用的原始输出。
    ToolResult,
    /// 检索到的文档文本。
    Document,
}

impl MemoryKind {
    /// 类别的 snake_case 名字。
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

    /// 值得保留最久的类别。
    pub fn is_durable(self) -> bool {
        matches!(self, Self::Fact | Self::Reflection)
    }

    /// 在单轮内产出的类别。
    pub fn is_ephemeral(self) -> bool {
        matches!(self, Self::Observation | Self::ToolResult)
    }
}

impl std::fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 一条被记住的项，加上排序与过期所需的一切。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// 存储分配的身份。
    pub id: MemoryId,
    /// 这是哪类记忆。
    pub kind: MemoryKind,
    /// 被记住的文本。
    pub text: String,
    /// [`MemoryRecord::text`] 的向量形式。
    pub embedding: Embedding,
    /// 存储它的智能体实体（若有）。
    #[serde(default)]
    pub owner: Option<Entity>,
    /// 此记忆所属的对话或任务。
    #[serde(default)]
    pub session: Option<String>,
    /// 用于过滤的自由格式标签。
    #[serde(default)]
    pub tags: Vec<String>,
    /// 重要性权重，通常 `0.5..=2.0`。
    #[serde(default = "one")]
    pub salience: f64,
    /// 记忆写入的时刻。
    pub created_at_ms: Timestamp,
    /// 上次被检索的时刻。
    #[serde(default)]
    pub accessed_at_ms: Option<Timestamp>,
    /// 被检索的频率。
    #[serde(default)]
    pub access_count: u32,
    /// 硬性过期，独立于保留策略。
    #[serde(default)]
    pub expires_at_ms: Option<Timestamp>,
}

fn one() -> f64 {
    1.0
}

impl MemoryRecord {
    /// 盖上当前时间戳的新记录。
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

    /// 带调用者提供时间戳的记录，用于回放与测试。
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

    /// 覆盖存储分配的 id，用于回放。
    pub fn with_id(mut self, id: MemoryId) -> Self {
        self.id = id;
        self
    }

    /// 把记录归属到一个智能体实体。
    pub fn with_owner(mut self, owner: Entity) -> Self {
        self.owner = Some(owner);
        self
    }

    /// 把记录归入一个对话或任务。
    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.session = Some(session.into());
        self
    }

    /// 添加一个过滤标签。
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// 添加若干过滤标签。
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// 在检索评分中给记录加权。
    pub fn with_salience(mut self, salience: f64) -> Self {
        self.salience = salience;
        self
    }

    /// 在 `ttl` 后过期，相对于创建时刻。
    pub fn with_ttl(mut self, ttl: std::time::Duration) -> Self {
        self.expires_at_ms = Some(self.created_at_ms.saturating_add(ttl.as_millis() as u64));
        self
    }

    /// 当记录恰好带有此标签时为真。
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|stored| stored == tag)
    }

    /// 记录一次检索，这正是让常用记忆留存的原因。
    pub fn touch(&mut self, at: Timestamp) {
        self.accessed_at_ms = Some(at);
        self.access_count = self.access_count.saturating_add(1);
    }

    /// 一旦硬性过期已过即为真。
    pub fn is_expired(&self, at: Timestamp) -> bool {
        self.expires_at_ms.is_some_and(|expiry| expiry <= at)
    }

    /// 以毫秒计的年龄，为时钟偏移钳制到零。
    pub fn age_ms(&self, at: Timestamp) -> Timestamp {
        at.saturating_sub(self.created_at_ms)
    }

    /// 自上次检索以来的时间，若从未检索则为年龄。
    pub fn idle_ms(&self, at: Timestamp) -> Timestamp {
        let reference = self.accessed_at_ms.unwrap_or(self.created_at_ms);
        at.saturating_sub(reference)
    }

    /// 语义相关性：余弦相似度，按重要性加权。
    pub fn relevance(&self, similarity: f64) -> f64 {
        similarity.max(0.0) * self.salience
    }

    /// 带指数新近性加成的相关性。
    ///
    /// `half_life_ms` 是一条其他方面相同的记忆失去一半新近性权重
    /// 所需的时间；`0` 禁用衰减。
    pub fn scored(&self, similarity: f64, at: Timestamp, half_life_ms: u64) -> f64 {
        let base = self.relevance(similarity);
        if half_life_ms == 0 || base <= 0.0 {
            return base;
        }
        let age = self.age_ms(at) as f64;
        let decay = 0.5f64.powf(age / half_life_ms.max(1) as f64);
        // 对持续被使用的记忆给一点强化。
        let reinforcement = 1.0 + (self.access_count as f64).ln_1p() * 0.1;
        base * (0.5 + 0.5 * decay) * reinforcement
    }

    /// 截断文本以便在日志与提示词中显示。
    pub fn preview(&self, max_chars: usize) -> String {
        if self.text.chars().count() <= max_chars {
            return self.text.clone();
        }
        let mut out: String = self
            .text
            .chars()
            .take(max_chars.saturating_sub(1))
            .collect();
        out.push('…');
        out
    }
}

/// 一个检索请求。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MemoryQuery {
    /// 当 `vector` 缺失时用于嵌入的文本。
    pub text: Option<String>,
    /// 预先算好的查询向量。
    pub vector: Option<Embedding>,
    /// 允许的类别；空表示全部。
    pub kinds: Vec<MemoryKind>,
    /// 必需的标签；空表示任意。
    pub tags: Vec<String>,
    /// 限制到单个智能体实体。
    pub owner: Option<Entity>,
    /// 限制到单个会话。
    pub session: Option<String>,
    /// 结果上限。
    pub top_k: usize,
    /// 最低分，位于显著性加权前的 `0.0..=1.0`。
    pub min_score: f64,
    /// 包含已过期的记录。
    pub include_expired: bool,
}

impl MemoryQuery {
    /// 文本查询，匹配前由存储嵌入。
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            top_k: 5,
            min_score: 0.0,
            ..Self::default()
        }
    }

    /// 向量查询，供已嵌入某物的调用者使用。
    pub fn vector(vector: Embedding) -> Self {
        Self {
            vector: Some(vector),
            top_k: 5,
            min_score: 0.0,
            ..Self::default()
        }
    }

    /// 把搜索限制到一个类别。
    pub fn with_kind(mut self, kind: MemoryKind) -> Self {
        self.kinds.push(kind);
        self
    }

    /// 把搜索限制到一组类别。
    pub fn with_kinds(mut self, kinds: impl IntoIterator<Item = MemoryKind>) -> Self {
        self.kinds.extend(kinds);
        self
    }

    /// 要求一个标签。
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// 把搜索限制到一个所有者。
    pub fn with_owner(mut self, owner: Entity) -> Self {
        self.owner = Some(owner);
        self
    }

    /// 把搜索限制到一个会话。
    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.session = Some(session.into());
        self
    }

    /// 至多返回这么多记录。
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// 丢弃评分低于此值的匹配。
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }

    /// 包含已过期的记录。
    pub fn including_expired(mut self) -> Self {
        self.include_expired = true;
        self
    }

    /// 此记录是否通过查询的非评分过滤？
    pub fn accepts(&self, record: &MemoryRecord, at: Timestamp) -> bool {
        if !self.include_expired && record.is_expired(at) {
            return false;
        }
        if !self.kinds.is_empty() && !self.kinds.contains(&record.kind) {
            return false;
        }
        if self.owner.is_some_and(|owner| record.owner != Some(owner)) {
            return false;
        }
        if self.session.is_some() && record.session != self.session {
            return false;
        }
        self.tags.iter().all(|tag| record.has_tag(tag))
    }

    /// 记录要被返回所必须达到的相似度下限。
    pub fn passes_score(&self, similarity: f64) -> bool {
        similarity >= self.min_score
    }
}

/// 一条记录连同其检索评分。
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredRecord {
    /// 记忆本身。
    pub record: MemoryRecord,
    /// 它在此查询下的评分，越高越靠前。
    pub score: f64,
}

impl ScoredRecord {
    /// 构建一条带评分的记录。
    pub fn new(record: MemoryRecord, score: f64) -> Self {
        Self { record, score }
    }

    /// 便于提示词组装。
    pub fn text(&self) -> &str {
        &self.record.text
    }

    /// 按评分降序排序，并以其 id 做稳定的平局裁决。
    pub fn sort(results: &mut [Self]) {
        results.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.record.id.cmp(&b.record.id))
        });
    }
}

/// 存储如何裁剪自身。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetentionPolicy {
    /// 记录数硬上限；价值最低的先被丢弃。
    pub max_records: usize,
    /// 丢弃比这更旧的任何记录，以毫秒计；`0` 禁用。
    pub max_age_ms: u64,
    /// 丢弃在 `salience * reinforcement` 上评分低于此值的记录。
    pub min_value: f64,
    /// 为驱逐评分时所用的新近性半衰期。
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
    /// 上限为 `max_records` 的策略。
    pub const fn with_max_records(max_records: usize) -> Self {
        Self {
            max_records,
            max_age_ms: 0,
            min_value: 0.0,
            half_life_ms: 60 * 60 * 1_000,
        }
    }

    /// 记忆至多保留 `ttl`。
    pub fn with_max_age(mut self, ttl: std::time::Duration) -> Self {
        self.max_age_ms = ttl.as_millis() as u64;
        self
    }

    /// 忘记显著性与使用不足以证明其占用空间合理的记录。
    pub const fn with_min_value(mut self, min_value: f64) -> Self {
        self.min_value = min_value;
        self
    }

    /// 记录的长期价值，独立于任何查询。
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

    /// 决定丢弃哪些记录；纯函数，因此排序可测试。
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
    use crate::agent_memory::embed::HashingEmbedder;

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

    /// 仅由其索引标识的实体；测试不需要世代。
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
        assert!(
            defaults.accepts(&tool, 0),
            "no filters means everything matches"
        );
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
        assert_eq!(evicted.len(), 3, "5 records, cap 2: three must go");

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
        assert!(kept.is_empty(), "reinforced records clear the value floor");
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
