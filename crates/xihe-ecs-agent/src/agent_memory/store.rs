//! 记忆存储：检索 trait 与一个内存向量索引。

use std::sync::atomic::{AtomicU64, Ordering};

use crate::ecs_core::Entity;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::agent_memory::embed::{Embedder, Embedding, HashingEmbedder};
use crate::agent_memory::error::{MemoryError, MemoryResult};
use crate::agent_memory::record::{
    MemoryId, MemoryKind, MemoryQuery, MemoryRecord, RetentionPolicy, ScoredRecord, Timestamp,
    now_ms,
};

/// 存储接缝：任何能记忆与召回之物。
///
/// 实现必须是 `Send + Sync`，使多个智能体实体能通过一个 `Arc`
/// 共享一份长期记忆。
#[async_trait]
pub trait MemoryStore: Send + Sync + 'static {
    /// 存储标识符，在 [`MemoryStats`] 中报告。
    fn id(&self) -> &str;

    /// 当记录不带向量被存储时所用的嵌入器。
    fn embedder(&self) -> Option<&dyn Embedder> {
        None
    }

    /// 记住一条记录，分配 id 并校验维度。
    ///
    /// 嵌入为空的记录先用存储的嵌入器编码，因此调用者可以
    /// 插入文本而无需关心向量；携带错误尺寸向量的记录会被拒绝，
    /// 而非重新编码。
    async fn insert(&self, record: MemoryRecord) -> MemoryResult<MemoryId>;

    /// 记住文本，用存储的嵌入器嵌入它。
    async fn remember(&self, kind: MemoryKind, text: &str) -> MemoryResult<MemoryId> {
        self.insert(MemoryRecord::new(kind, text, Embedding::default()))
            .await
    }

    /// 检索最佳匹配，评分最高者优先。
    async fn search(&self, query: &MemoryQuery) -> MemoryResult<Vec<ScoredRecord>>;

    /// 按 id 取一条记录。
    async fn get(&self, id: MemoryId) -> MemoryResult<Option<MemoryRecord>>;

    /// 忘记一条记录。
    async fn remove(&self, id: MemoryId) -> MemoryResult<bool>;

    /// 丢弃此实体记住的一切。
    async fn forget_entity(&self, owner: Entity) -> MemoryResult<usize>;

    /// 应用保留策略，返回丢弃了多少记录。
    async fn consolidate(&self, policy: &RetentionPolicy) -> MemoryResult<usize>;

    /// 存储中的一切，按插入顺序。
    async fn snapshot(&self) -> Vec<MemoryRecord>;

    /// 用于日志与测试的计数器。
    async fn stats(&self) -> MemoryStats;

    /// 匹配某谓词的记录，按插入顺序。
    async fn find(
        &self,
        predicate: &(dyn for<'a> Fn(&'a MemoryRecord) -> bool + Sync),
    ) -> Vec<MemoryRecord> {
        self.snapshot()
            .await
            .into_iter()
            .filter(|record| predicate(record))
            .collect()
    }

    /// 便利方法：单个最佳匹配（若有）。
    async fn recall(&self, query: &MemoryQuery) -> MemoryResult<Option<ScoredRecord>> {
        Ok(self
            .search(&query.clone().with_top_k(1))
            .await?
            .into_iter()
            .next())
    }
}

/// 描述一个存储的计数器。
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryStats {
    /// 当前持有的记录数。
    pub records: usize,
    /// 累计插入数，含后来被删除的。
    pub inserts: u64,
    /// 累计已服务的检索数。
    pub hits: u64,
    /// 累计被删除或保留策略移除的记录数。
    pub evictions: u64,
    /// `search` 调用总数，无论是否有匹配。
    pub queries: u64,
}

/// [`InMemoryStore::search`] 的调参。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchOptions {
    /// 收集候选时把所请求的 `top_k` 乘以此值，
    /// 使多样性有可供挑选的余地。
    pub candidate_multiplier: usize,
    /// `0.0` 为纯相关性，`1.0` 为纯多样性。
    pub mmf_lambda: f64,
    /// 施加到评分上的新近性半衰期；`0` 禁用它。
    pub half_life_ms: u64,
    /// 为返回的记录更新 `accessed_at_ms` 与 `access_count`。
    pub reinforce_hits: bool,
    /// 把无人拥有的记忆评分得仿佛相关性略低。
    pub unowned_penalty: f64,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            candidate_multiplier: 4,
            mmf_lambda: 0.0,
            half_life_ms: 0,
            reinforce_hits: true,
            unowned_penalty: 1.0,
        }
    }
}

impl SearchOptions {
    /// 仅按相关性排序。
    pub fn pure_relevance() -> Self {
        Self::default()
    }

    /// 使结果多样化，免得提示词被近重复项填满。
    pub fn with_diversity(mut self, lambda: f64) -> Self {
        self.mmf_lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// 按新近性给评分加权。
    pub fn with_recency(mut self, half_life: std::time::Duration) -> Self {
        self.half_life_ms = half_life.as_millis() as u64;
        self
    }

    /// 不动访问计数器，用于只读探查。
    pub const fn without_reinforcement(mut self) -> Self {
        self.reinforce_hits = false;
        self
    }
}

/// 一个暴力向量索引，这正是智能体会话实际持有的数万条记忆
/// 所适合的形态。
/// 检索计数器，分开保存以便各自独立饱和。
#[derive(Debug, Default)]
struct Counters {
    inserts: AtomicU64,
    hits: AtomicU64,
    evictions: AtomicU64,
    queries: AtomicU64,
}

/// 一个暴力向量索引。
///
/// 克隆会共享已存储的记忆——这正是多个智能体实体通过一个 `Arc`
/// 读取一份长期记忆的方式——而 `with_name` 与 `with_options`
/// 只调整被调用它们的那个句柄。
#[derive(Clone)]
pub struct InMemoryStore {
    records: std::sync::Arc<RwLock<Vec<MemoryRecord>>>,
    embedder: Option<std::sync::Arc<dyn Embedder>>,
    dimensions: usize,
    next_id: std::sync::Arc<AtomicU64>,
    counters: std::sync::Arc<Counters>,
    options: SearchOptions,
    name: String,
}

impl std::fmt::Debug for InMemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryStore")
            .field("name", &self.name)
            .field("dimensions", &self.dimensions)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::with_embedder(HashingEmbedder::default())
    }
}

impl InMemoryStore {
    /// 带哈希嵌入器、无标签的内存存储。
    pub fn new() -> Self {
        Self::with_embedder(HashingEmbedder::default())
    }

    /// 在 `remember` 时嵌入、尺寸取自嵌入器的存储。
    pub fn with_embedder<E: Embedder>(embedder: E) -> Self {
        let dimensions = embedder.dimensions();
        Self {
            records: std::sync::Arc::new(RwLock::new(Vec::new())),
            embedder: Some(std::sync::Arc::new(embedder)),
            dimensions,
            next_id: std::sync::Arc::new(AtomicU64::new(1)),
            counters: std::sync::Arc::new(Counters::default()),
            options: SearchOptions::default(),
            name: "in-memory".to_string(),
        }
    }

    /// 无嵌入器的存储：调用者自备向量。
    pub fn bare(dimensions: usize) -> Self {
        Self {
            records: std::sync::Arc::new(RwLock::new(Vec::new())),
            embedder: None,
            dimensions,
            next_id: std::sync::Arc::new(AtomicU64::new(1)),
            counters: std::sync::Arc::new(Counters::default()),
            options: SearchOptions::default(),
            name: "in-memory".to_string(),
        }
    }

    /// 给存储命名，使日志能区分两份记忆。
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// 替换搜索调参。
    pub fn with_options(mut self, options: SearchOptions) -> Self {
        self.options = options;
        self
    }

    /// 向量据以比较的嵌入器（若存储有）。
    pub fn embedder(&self) -> Option<&dyn Embedder> {
        self.embedder.as_deref()
    }

    /// 此存储接受的向量宽度。
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// 当前生效的搜索调参。
    pub fn options(&self) -> SearchOptions {
        self.options
    }

    fn bump(&self, field: Counter) {
        let counter = match field {
            Counter::Inserts => &self.counters.inserts,
            Counter::Hits => &self.counters.hits,
            Counter::Evictions => &self.counters.evictions,
            Counter::Queries => &self.counters.queries,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn bump_by(&self, field: Counter, amount: usize) {
        let counter = match field {
            Counter::Inserts => &self.counters.inserts,
            Counter::Hits => &self.counters.hits,
            Counter::Evictions => &self.counters.evictions,
            Counter::Queries => &self.counters.queries,
        };
        counter.fetch_add(amount as u64, Ordering::Relaxed);
    }

    /// 在保留记录前，对照存储维度校验它。
    fn check(&self, record: &MemoryRecord) -> MemoryResult<()> {
        if self.dimensions > 0 && record.embedding.dimensions() != self.dimensions {
            return Err(MemoryError::dimension_mismatch(
                self.dimensions,
                record.embedding.dimensions(),
            ));
        }
        Ok(())
    }

    /// 记录与查询向量的相似度，带存储级调整。
    fn similarity(&self, record: &MemoryRecord, vector: &Embedding, at: Timestamp) -> f64 {
        let cosine = record.embedding.cosine(vector);
        let weighted = record.scored(cosine, at, self.options.half_life_ms);
        if record.owner.is_none() {
            weighted * self.options.unowned_penalty
        } else {
            weighted
        }
    }

    /// 最大边际相关性选择：相关性减去相对已选记录的
    /// 冗余。
    fn diversify(candidates: &[ScoredRecord], lambda: f64) -> Vec<ScoredRecord> {
        if lambda <= 0.0 || candidates.len() < 2 {
            return candidates.to_vec();
        }
        let mut chosen: Vec<usize> = Vec::with_capacity(candidates.len());
        let mut remaining: Vec<usize> = (0..candidates.len()).collect();
        let mut order: Vec<usize> = Vec::with_capacity(candidates.len());
        while !remaining.is_empty() {
            let mut best_index = 0usize;
            let mut best_score = f64::MIN;
            for (position, candidate) in remaining.iter().enumerate() {
                let record = &candidates[*candidate];
                let redundancy = chosen
                    .iter()
                    .map(|picked| {
                        candidates[*picked]
                            .record
                            .embedding
                            .cosine(&record.record.embedding)
                    })
                    .fold(0.0_f64, f64::max);
                let score = record.score * (1.0 - lambda) - redundancy * lambda;
                if score > best_score {
                    best_score = score;
                    best_index = position;
                }
            }
            let picked = remaining.remove(best_index);
            chosen.push(picked);
            order.push(picked);
        }
        // 选出的顺序就是答案：按相关性重排会把近重复项
        // 放回前面，而被截断的调用者将永远看不到 MMR
        // 刚买来的多样性。
        order
            .iter()
            .map(|&index| candidates[index].clone())
            .collect()
    }
}

/// 一次操作触碰 [`Counters`] 的哪个字段。
#[derive(Clone, Copy)]
enum Counter {
    Inserts,
    Hits,
    Evictions,
    Queries,
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    fn id(&self) -> &str {
        &self.name
    }

    fn embedder(&self) -> Option<&dyn Embedder> {
        self.embedder.as_deref()
    }

    async fn insert(&self, mut record: MemoryRecord) -> MemoryResult<MemoryId> {
        if record.embedding.dimensions() == 0 {
            let embedder = self
                .embedder
                .as_deref()
                .ok_or_else(|| MemoryError::storage("this store has no embedder to encode with"))?;
            record.embedding = embedder.embed(&record.text).await?;
        }
        self.check(&record)?;
        if record.id == 0 {
            record.id = self.next_id.fetch_add(1, Ordering::Relaxed);
        } else {
            // 让 id 生成器领先于任何外部分配的 id，
            // 使无论谁铸造 id 都保持唯一。
            loop {
                let current = self.next_id.load(Ordering::Relaxed);
                if current > record.id {
                    break;
                }
                if self
                    .next_id
                    .compare_exchange_weak(
                        current,
                        record.id + 1,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    break;
                }
            }
        }
        let id = record.id;
        let mut records = self.records.write().await;
        if let Some(existing) = records.iter_mut().find(|stored| stored.id == id) {
            *existing = record;
        } else {
            records.push(record);
        }
        self.bump(Counter::Inserts);
        Ok(id)
    }

    async fn search(&self, query: &MemoryQuery) -> MemoryResult<Vec<ScoredRecord>> {
        self.bump(Counter::Queries);
        let vector = match (&query.vector, &query.text) {
            (Some(vector), _) => vector.clone(),
            (None, Some(text)) => match self.embedder() {
                Some(embedder) => embedder.embed(text).await?,
                None => {
                    return Err(MemoryError::storage(
                        "a text query needs a store built with an embedder",
                    ));
                }
            },
            (None, None) => {
                return Err(MemoryError::storage(
                    "a query must carry either text or a vector",
                ));
            }
        };
        if self.dimensions > 0 && vector.dimensions() != self.dimensions {
            return Err(MemoryError::dimension_mismatch(
                self.dimensions,
                vector.dimensions(),
            ));
        }
        let mut records = self.records.write().await;
        let at = now_ms();
        let mut candidates: Vec<ScoredRecord> = records
            .iter()
            .filter(|record| query.accepts(record, at))
            .map(|record| ScoredRecord::new(record.clone(), self.similarity(record, &vector, at)))
            .filter(|hit| query.passes_score(hit.score))
            .collect();
        ScoredRecord::sort(&mut candidates);
        if self.options.mmf_lambda > 0.0 {
            let window = (query.top_k * self.options.candidate_multiplier).max(query.top_k);
            let pool: Vec<ScoredRecord> = candidates.into_iter().take(window).collect();
            candidates = InMemoryStore::diversify(&pool, self.options.mmf_lambda);
        }
        let top_k = query.top_k.max(1);
        candidates.truncate(top_k);
        if self.options.reinforce_hits {
            for hit in &candidates {
                if let Some(stored) = records.iter_mut().find(|record| record.id == hit.record.id) {
                    stored.touch(at);
                }
            }
        }
        self.bump(Counter::Hits);
        Ok(candidates)
    }

    async fn get(&self, id: MemoryId) -> MemoryResult<Option<MemoryRecord>> {
        Ok(self
            .records
            .read()
            .await
            .iter()
            .find(|record| record.id == id)
            .cloned())
    }

    async fn remove(&self, id: MemoryId) -> MemoryResult<bool> {
        let mut records = self.records.write().await;
        let before = records.len();
        records.retain(|record| record.id != id);
        let removed = records.len() != before;
        if removed {
            self.bump(Counter::Evictions);
        }
        Ok(removed)
    }

    async fn forget_entity(&self, owner: Entity) -> MemoryResult<usize> {
        let mut records = self.records.write().await;
        let before = records.len();
        records.retain(|record| record.owner != Some(owner));
        let removed = before - records.len();
        self.bump_by(Counter::Evictions, removed);
        Ok(removed)
    }

    async fn consolidate(&self, policy: &RetentionPolicy) -> MemoryResult<usize> {
        let at = now_ms();
        let mut records = self.records.write().await;
        let evicted = policy.plan_evictions(&records, at);
        let count = evicted.len();
        records.retain(|record| !evicted.contains(&record.id));
        drop(records);
        self.bump_by(Counter::Evictions, count);
        Ok(count)
    }

    async fn snapshot(&self) -> Vec<MemoryRecord> {
        self.records.read().await.clone()
    }

    async fn stats(&self) -> MemoryStats {
        MemoryStats {
            records: self.records.read().await.len(),
            inserts: self.counters.inserts.load(Ordering::Relaxed),
            hits: self.counters.hits.load(Ordering::Relaxed),
            evictions: self.counters.evictions.load(Ordering::Relaxed),
            queries: self.counters.queries.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_memory::embed::{ConstantEmbedder, RandomEmbedder};

    fn store() -> InMemoryStore {
        InMemoryStore::with_embedder(HashingEmbedder::new(64)).with_name("test-store")
    }

    async fn seed(store: &InMemoryStore) {
        store
            .remember(MemoryKind::Fact, "the tokio runtime schedules tasks")
            .await
            .unwrap();
        store
            .remember(MemoryKind::Fact, "ecs stores components in sparse sets")
            .await
            .unwrap();
        store
            .remember(MemoryKind::Observation, "banana pancakes on a sunday")
            .await
            .unwrap();
    }

    /// 仅由其索引标识的实体；测试不需要世代。
    fn entity(index: u32) -> Entity {
        Entity::from_raw(index, 1)
    }

    #[tokio::test]
    async fn remember_embeds_and_assigns_ids() {
        let store = store();
        let id = store
            .remember(MemoryKind::Fact, "first memory")
            .await
            .unwrap();
        assert_eq!(id, 1);
        let other = store
            .remember(MemoryKind::Fact, "second memory")
            .await
            .unwrap();
        assert_eq!(other, 2, "ids increment");
        let record = store.get(id).await.unwrap().expect("stored");
        assert_eq!(record.text, "first memory");
        assert_eq!(record.embedding.dimensions(), 64);
        assert_eq!(store.id(), "test-store");
        assert_eq!(store.stats().await.records, 2);
    }

    #[tokio::test]
    async fn text_queries_rank_by_similarity() {
        let store = store();
        seed(&store).await;
        let hits = store
            .search(&MemoryQuery::text("tokio tasks scheduler").with_top_k(2))
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert!(
            hits[0].text().contains("tokio"),
            "got {:?}",
            hits.iter().map(ScoredRecord::text).collect::<Vec<_>>()
        );
        assert!(hits[0].score >= hits[1].score);
    }

    #[tokio::test]
    async fn a_store_without_an_embedder_refuses_text_queries() {
        let store = InMemoryStore::bare(8);
        let unit = Embedding(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        store
            .insert(MemoryRecord::new(MemoryKind::Fact, "x", unit.clone()))
            .await
            .unwrap();
        let error = store.search(&MemoryQuery::text("x")).await.unwrap_err();
        assert!(error.to_string().contains("embedder"));
        let hits = store.search(&MemoryQuery::vector(unit)).await.unwrap();
        assert_eq!(hits.len(), 1, "vector queries still work");
        let error = store.remember(MemoryKind::Fact, "nope").await.unwrap_err();
        assert!(error.to_string().contains("embedder"), "{error}");
    }

    #[tokio::test]
    async fn dimension_mismatch_is_rejected() {
        let store = store();
        let error = store
            .insert(MemoryRecord::new(
                MemoryKind::Fact,
                "wrong size",
                Embedding(vec![0.1; 8]),
            ))
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "embedding has 8 dimensions, the store expects 64"
        );
        assert_eq!(store.stats().await.records, 0);
    }

    #[tokio::test]
    async fn empty_queries_are_an_error_not_a_panics() {
        let store = store();
        let error = store.search(&MemoryQuery::default()).await.unwrap_err();
        assert!(error.to_string().contains("either text or a vector"));
    }

    #[tokio::test]
    async fn filters_narrow_before_scoring() {
        let store = store();
        let entity = entity(7);
        store
            .insert(
                MemoryRecord::new(
                    MemoryKind::ToolResult,
                    "tokio output",
                    store
                        .embedder()
                        .unwrap()
                        .embed("tokio output")
                        .await
                        .unwrap(),
                )
                .with_owner(entity)
                .with_tag("scratch"),
            )
            .await
            .unwrap();
        let facts_only = store
            .search(
                &MemoryQuery::text("tokio")
                    .with_kind(MemoryKind::Fact)
                    .with_top_k(10),
            )
            .await
            .unwrap();
        assert!(
            facts_only
                .iter()
                .all(|hit| hit.record.kind == MemoryKind::Fact),
            "kind filter must apply"
        );
        let owned = store
            .search(&MemoryQuery::text("tokio").with_owner(entity))
            .await
            .unwrap();
        assert_eq!(owned.len(), 1);
        let tagged = store
            .search(&MemoryQuery::text("tokio").with_tag("scratch"))
            .await
            .unwrap();
        assert_eq!(tagged[0].record.owner, Some(entity));
    }

    #[tokio::test]
    async fn min_score_drops_weak_matches() {
        let store = InMemoryStore::with_embedder(RandomEmbedder::new(256));
        for index in 0..8 {
            store
                .remember(MemoryKind::Fact, &format!("memory number {index}"))
                .await
                .unwrap();
        }
        let all = store
            .search(&MemoryQuery::text("memory number 3").with_top_k(8))
            .await
            .unwrap();
        let strict = store
            .search(
                &MemoryQuery::text("memory number 3")
                    .with_top_k(8)
                    .with_min_score(0.5),
            )
            .await
            .unwrap();
        assert!(all.len() > strict.len(), "a floor must remove something");
        assert!(strict.iter().all(|hit| hit.score >= 0.5));
    }

    #[tokio::test]
    async fn hits_reinforce_access_counters() {
        let store = store();
        let id = store
            .remember(MemoryKind::Fact, "remember me")
            .await
            .unwrap();
        store.search(&MemoryQuery::text("remember")).await.unwrap();
        store.search(&MemoryQuery::text("remember")).await.unwrap();
        let record = store.get(id).await.unwrap().unwrap();
        assert_eq!(record.access_count, 2);
        assert!(record.accessed_at_ms.is_some());
        let stats = store.stats().await;
        assert_eq!(stats.queries, 2);
        assert_eq!(stats.hits, 2);
    }

    #[tokio::test]
    async fn reinforcement_can_be_disabled() {
        let store = store().with_options(SearchOptions::default().without_reinforcement());
        let id = store.remember(MemoryKind::Fact, "read only").await.unwrap();
        store.search(&MemoryQuery::text("read")).await.unwrap();
        assert_eq!(store.get(id).await.unwrap().unwrap().access_count, 0);
    }

    #[tokio::test]
    async fn diversity_spreads_results() {
        let embedder = HashingEmbedder::new(64);
        let store = InMemoryStore::with_embedder(embedder.clone())
            .with_options(SearchOptions::default().with_diversity(0.6));
        for index in 0..3 {
            store
                .remember(
                    MemoryKind::Fact,
                    &format!("the ecs framework stores components in sparse set {index}"),
                )
                .await
                .unwrap();
        }
        store
            .remember(
                MemoryKind::Fact,
                "tokio schedules tasks on multiple threads",
            )
            .await
            .unwrap();
        let hits = store
            .search(&MemoryQuery::text("ecs components sparse").with_top_k(2))
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        let texts: Vec<&str> = hits.iter().map(ScoredRecord::text).collect();
        assert!(
            texts.iter().any(|text| text.contains("tokio")),
            "near-duplicates should make room: {texts:?}"
        );
    }

    #[tokio::test]
    async fn recency_weighting_changes_the_order() {
        let embedder = ConstantEmbedder::new(8);
        let store = InMemoryStore::with_embedder(embedder.clone()).with_options(
            SearchOptions::default().with_recency(std::time::Duration::from_millis(1)),
        );
        let mut old =
            MemoryRecord::new(MemoryKind::Fact, "old", embedder.embed("x").await.unwrap());
        old.created_at_ms = 0;
        let new_id = store
            .insert(MemoryRecord::new(
                MemoryKind::Fact,
                "new",
                embedder.embed("y").await.unwrap(),
            ))
            .await
            .unwrap();
        let old_id = store.insert(old).await.unwrap();
        let hits = store
            .search(&MemoryQuery::vector(Embedding(vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ])))
            .await
            .unwrap();
        assert_eq!(hits[0].record.id, new_id, "the fresh memory wins");
        assert_ne!(hits[0].record.id, old_id);
    }

    #[tokio::test]
    async fn unowned_penalty_downgrades_shared_memories() {
        let embedder = HashingEmbedder::new(64);
        let vector = embedder.embed("shared fact").await.unwrap();
        let base = InMemoryStore::with_embedder(embedder.clone());
        let penalised = InMemoryStore::with_embedder(embedder).with_options(SearchOptions {
            unowned_penalty: 0.5,
            reinforce_hits: false,
            ..SearchOptions::default()
        });
        let record = MemoryRecord::new(MemoryKind::Fact, "shared fact", vector.clone());
        base.insert(record.clone()).await.unwrap();
        penalised.insert(record).await.unwrap();
        let query = MemoryQuery::vector(vector.clone());
        let plain = base.search(&query).await.unwrap()[0].score;
        let discounted = penalised.search(&query).await.unwrap()[0].score;
        assert!(
            (discounted - plain * 0.5).abs() < 1e-9,
            "{discounted} should be half of {plain}"
        );
    }

    #[tokio::test]
    async fn remove_and_forget_entity() {
        let store = store();
        let entity = entity(4);
        let mine = store
            .insert(
                MemoryRecord::new(
                    MemoryKind::Fact,
                    "mine",
                    store.embedder().unwrap().embed("mine").await.unwrap(),
                )
                .with_owner(entity),
            )
            .await
            .unwrap();
        store.remember(MemoryKind::Fact, "nobody's").await.unwrap();
        assert!(store.remove(mine).await.unwrap());
        assert!(
            !store.remove(mine).await.unwrap(),
            "removing twice is false"
        );
        assert_eq!(store.stats().await.records, 1);
        let second = store
            .insert(
                MemoryRecord::new(
                    MemoryKind::Fact,
                    "also mine",
                    store.embedder().unwrap().embed("also mine").await.unwrap(),
                )
                .with_owner(entity),
            )
            .await
            .unwrap();
        assert_eq!(store.forget_entity(entity).await.unwrap(), 1);
        assert!(store.get(second).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn insert_with_a_chosen_id_reserves_it() {
        let store = store();
        let id = store
            .insert(
                MemoryRecord::new(
                    MemoryKind::Fact,
                    "preset",
                    store.embedder().unwrap().embed("preset").await.unwrap(),
                )
                .with_id(500),
            )
            .await
            .unwrap();
        assert_eq!(id, 500);
        let next = store.remember(MemoryKind::Fact, "after").await.unwrap();
        assert!(next > 500, "auto ids must not collide: {next}");
        let again = store
            .insert(
                MemoryRecord::new(
                    MemoryKind::Fact,
                    "updated",
                    store.embedder().unwrap().embed("preset").await.unwrap(),
                )
                .with_id(500),
            )
            .await
            .unwrap();
        assert_eq!(again, 500, "the same id updates in place");
        assert_eq!(store.stats().await.records, 2);
        assert_eq!(store.get(500).await.unwrap().unwrap().text, "updated");
    }

    #[tokio::test]
    async fn consolidate_applies_the_policy() {
        let store = store();
        for index in 0..10 {
            store
                .remember(MemoryKind::Observation, &format!("observation {index}"))
                .await
                .unwrap();
        }
        let dropped = store
            .consolidate(&RetentionPolicy::with_max_records(4))
            .await
            .unwrap();
        assert_eq!(dropped, 6);
        assert_eq!(store.stats().await.records, 4);
        assert_eq!(store.stats().await.evictions, 6);
    }

    #[tokio::test]
    async fn find_filters_by_predicate() {
        let store = store();
        seed(&store).await;
        let is_fact = |record: &MemoryRecord| record.kind == MemoryKind::Fact;
        let facts = store.find(&is_fact).await;
        assert_eq!(facts.len(), 2);
        let absent = |record: &MemoryRecord| record.text.contains("nothing");
        let none = store.find(&absent).await;
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn expired_records_are_hidden_but_stored() {
        let store = store();
        let id = store
            .insert(
                MemoryRecord::new(
                    MemoryKind::Observation,
                    "short lived fact about tokio",
                    store
                        .embedder()
                        .unwrap()
                        .embed("short lived fact about tokio")
                        .await
                        .unwrap(),
                )
                .with_ttl(std::time::Duration::from_millis(1)),
            )
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(3)).await;
        let hidden = store
            .search(&MemoryQuery::text("tokio").with_top_k(5))
            .await
            .unwrap();
        assert!(hidden.is_empty(), "expiry hides the record");
        let shown = store
            .search(&MemoryQuery::text("tokio").with_top_k(5).including_expired())
            .await
            .unwrap();
        assert_eq!(shown[0].record.id, id);
    }

    #[tokio::test]
    async fn snapshot_and_recall_cover_the_api() {
        let store = store();
        seed(&store).await;
        assert_eq!(store.snapshot().await.len(), 3);
        let best = store
            .recall(&MemoryQuery::text("sparse sets"))
            .await
            .unwrap()
            .expect("something matches");
        assert!(best.text().contains("sparse"));
        assert_eq!(best.record.kind, MemoryKind::Fact);
        assert_eq!(store.dimensions(), 64);
        assert_eq!(store.options(), SearchOptions::default());
    }

    #[tokio::test]
    async fn concurrent_remembers_keep_ids_unique() {
        let store = store();
        let mut tasks = Vec::new();
        for index in 0..16 {
            let store = store.clone();
            tasks.push(tokio::spawn(async move {
                store
                    .remember(MemoryKind::Fact, &format!("concurrent {index}"))
                    .await
            }));
        }
        let mut ids = Vec::new();
        for task in tasks {
            ids.push(task.await.unwrap().unwrap());
        }
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 16, "ids must not collide under load");
        assert_eq!(store.stats().await.records, 16);
    }

    #[tokio::test]
    async fn trait_object_dispatch_works() {
        let store: std::sync::Arc<dyn MemoryStore> = std::sync::Arc::new(store());
        store
            .remember(MemoryKind::Fact, "behind a dyn arc")
            .await
            .unwrap();
        assert_eq!(
            store.search(&MemoryQuery::text("dyn")).await.unwrap().len(),
            1
        );
    }
}
