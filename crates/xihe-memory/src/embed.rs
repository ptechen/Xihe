//! Vector embeddings: the numeric side of recall.
//!
//! A production agent would call a real embedding model; this crate ships a
//! deterministic [feature-hashing](HashingEmbedder) embedder so recall, ranking
//! and persistence are testable with no network and no weights to download. The
//! [`Embedder`] trait is the seam: swap in an HTTP or local-model embedder and
//! every store keeps working.

use async_trait::async_trait;

use crate::error::{MemoryError, MemoryResult};

/// A dense vector.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Embedding(pub Vec<f64>);

impl Embedding {
    /// Wrap a vector, normally already L2-normalised.
    pub fn new(values: Vec<f64>) -> Self {
        Self(values)
    }

    /// A zero vector of `dimensions`, used as a placeholder.
    pub fn zeros(dimensions: usize) -> Self {
        Self(vec![0.0; dimensions])
    }

    /// Vector width; the store validates against its embedder.
    pub fn dimensions(&self) -> usize {
        self.0.len()
    }

    /// Euclidean length.
    pub fn norm(&self) -> f64 {
        self.0.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    /// Unit-length copy; a zero vector normalises to itself.
    pub fn normalized(&self) -> Self {
        let norm = self.norm();
        if norm == 0.0 {
            return self.clone();
        }
        Self(self.0.iter().map(|value| value / norm).collect())
    }

    /// Dot product.
    pub fn dot(&self, other: &Self) -> f64 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Cosine similarity, in `-1.0..=1.0`; vectors of different widths score 0.
    pub fn cosine(&self, other: &Self) -> f64 {
        if self.dimensions() != other.dimensions() || self.0.is_empty() {
            return 0.0;
        }
        let denominator = self.norm() * other.norm();
        if denominator == 0.0 {
            return 0.0;
        }
        self.dot(other) / denominator
    }

    /// L2 distance; unlike cosine it grows with magnitude.
    pub fn distance(&self, other: &Self) -> f64 {
        if self.dimensions() != other.dimensions() {
            return f64::INFINITY;
        }
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Average several vectors into one, the cheap way to fold a multi-turn
    /// episode into a single memory.
    pub fn mean(vectors: &[Embedding]) -> Option<Embedding> {
        let first = vectors.first()?;
        let dimensions = first.dimensions();
        let mut total = vec![0.0; dimensions];
        let mut used = 0usize;
        for vector in vectors {
            if vector.dimensions() != dimensions {
                continue;
            }
            for (slot, value) in total.iter_mut().zip(vector.0.iter()) {
                *slot += value;
            }
            used += 1;
        }
        if used == 0 {
            return None;
        }
        Some(Embedding(
            total.iter().map(|value| value / used as f64).collect(),
        ))
    }

    /// Keep the `k` largest components, the trick that makes hashed embeddings
    /// sparse and comparable.
    pub fn top_k(&self, k: usize) -> Vec<(usize, f64)> {
        let mut pairs: Vec<(usize, f64)> = self.0.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        pairs.truncate(k);
        pairs
    }
}



/// Turn text into vectors.
#[async_trait]
pub trait Embedder: Send + Sync + 'static {
    /// Vector width this embedder always produces.
    fn dimensions(&self) -> usize;

    /// A short name for logs and stored metadata.
    fn id(&self) -> &str;

    /// Embed one string.
    async fn embed(&self, text: &str) -> MemoryResult<Embedding>;

    /// Embed several strings, preserving order.
    async fn embed_all(&self, texts: &[String]) -> MemoryResult<Vec<Embedding>> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            out.push(self.embed(text).await?);
        }
        Ok(out)
    }

    /// Whether two texts would land on the same vector.
    async fn is_same(&self, a: &str, b: &str) -> MemoryResult<bool> {
        Ok(self.embed(a).await? == self.embed(b).await?)
    }
}

/// Signed feature hashing over lower-cased words and character trigrams.
///
/// Deterministic, allocation-light and surprisingly effective for the short,
/// keyword-heavy texts agents exchange. It is not semantic: "buy milk" and
/// "purchase dairy" will not match. Real agents should swap in a model.
#[derive(Clone, Debug)]
pub struct HashingEmbedder {
    dimensions: usize,
    /// Weight of character trigrams relative to whole words.
    trigram_weight: f64,
}

impl Default for HashingEmbedder {
    fn default() -> Self {
        Self::new(256)
    }
}

impl HashingEmbedder {
    /// Build an embedder of the requested width.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions: dimensions.max(8),
            trigram_weight: 0.25,
        }
    }

    /// How much character-trigram signal to mix in, `0.0` to disable.
    pub const fn with_trigram_weight(mut self, weight: f64) -> Self {
        self.trigram_weight = weight;
        self
    }

    /// FNV-1a: small, fast and stable across runs, which matters because stored
    /// vectors must stay comparable after a restart.
    fn hash(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    fn add(values: &mut [f64], index: u64, weight: f64) {
        let slot = (index % values.len() as u64) as usize;
        // The top bit decides the sign, so unrelated features rarely cancel out.
        let sign = if index & (1 << 63) == 0 { 1.0 } else { -1.0 };
        values[slot] += sign * weight;
    }

    /// Fill a vector from text; exposed so other embedders can reuse the tokeniser.
    pub fn features(&self, text: &str) -> Embedding {
        let mut values = vec![0.0; self.dimensions];
        let lowered = text.to_lowercase();
        let mut words = lowered
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .peekable();
        let mut count = 0usize;
        while let Some(word) = words.next() {
            count += 1;
            Self::add(&mut values, Self::hash(word.as_bytes()), 1.0);
            // Bigrams keep "not good" distinguishable from "good".
            if let Some(next) = words.peek() {
                let pair = format!("{word} {next}");
                Self::add(&mut values, Self::hash(pair.as_bytes()), 0.6);
            }
            if self.trigram_weight > 0.0 && word.chars().count() >= 5 {
                let chars: Vec<char> = word.chars().collect();
                for window in chars.windows(3) {
                    let gram: String = window.iter().collect();
                    Self::add(&mut values, Self::hash(gram.as_bytes()), self.trigram_weight);
                }
            }
        }
        // Record length so a store can tell an empty text from a stop-word-only one.
        Self::add(&mut values, Self::hash(b"__len__"), (count as f64).sqrt() * 0.2);
        Embedding(values)
    }
}

#[async_trait]
impl Embedder for HashingEmbedder {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn id(&self) -> &str {
        "hashing"
    }

    async fn embed(&self, text: &str) -> MemoryResult<Embedding> {
        Ok(self.features(text).normalized())
    }
}

/// An embedder that maps each distinct text to a fixed, pseudo-random vector.
///
/// Useful when a test wants two texts to be *unrelated* — hashing would otherwise
/// give shared words a shared direction.
#[derive(Clone, Debug, Default)]
pub struct RandomEmbedder {
    dimensions: usize,
}

impl RandomEmbedder {
    /// Build an embedder of the requested width.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions: dimensions.max(8),
        }
    }

    fn vector(&self, text: &str) -> Embedding {
        let mut values = vec![0.0; self.dimensions];
        // A linear congruential walk seeded by the text hash: deterministic, and
        // any two different texts land on unrelated points.
        let mut state = HashingEmbedder::hash(text.as_bytes())
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(1);
        for slot in values.iter_mut() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 11) as f64) / (1u64 << 53) as f64;
            *slot = unit * 2.0 - 1.0;
        }
        Embedding(values).normalized()
    }
}

#[async_trait]
impl Embedder for RandomEmbedder {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn id(&self) -> &str {
        "random"
    }

    async fn embed(&self, text: &str) -> MemoryResult<Embedding> {
        Ok(self.vector(text))
    }
}

/// A fixed single-direction embedder, for tests that only need ordering.
#[derive(Clone, Debug)]
pub struct ConstantEmbedder {
    dimensions: usize,
}

impl ConstantEmbedder {
    /// Build an embedder of the requested width.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions: dimensions.max(2),
        }
    }
}

#[async_trait]
impl Embedder for ConstantEmbedder {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn id(&self) -> &str {
        "constant"
    }

    async fn embed(&self, _text: &str) -> MemoryResult<Embedding> {
        let mut values = vec![0.0; self.dimensions];
        values[0] = 1.0;
        Ok(Embedding(values))
    }
}

/// An embedder whose vectors are the characters of the text, for debugging.
///
/// Anything that fails to embed in a real deployment (a missing key, an offline
/// model) can be simulated with [`FailingEmbedder`] instead.
#[derive(Clone, Debug)]
pub struct CharacterEmbedder {
    dimensions: usize,
}

impl CharacterEmbedder {
    /// Build an embedder of the requested width.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions: dimensions.clamp(2, 128),
        }
    }
}

#[async_trait]
impl Embedder for CharacterEmbedder {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn id(&self) -> &str {
        "character"
    }

    async fn embed(&self, text: &str) -> MemoryResult<Embedding> {
        let mut values = vec![0.0; self.dimensions];
        for byte in text.bytes() {
            values[usize::from(byte % self.dimensions as u8)] += 1.0;
        }
        Ok(Embedding(values).normalized())
    }
}

/// Always fails, to exercise error paths.
#[derive(Clone, Debug)]
pub struct FailingEmbedder {
    dimensions: usize,
    message: String,
}

impl FailingEmbedder {
    /// Build an embedder that fails with `message`.
    pub fn new(dimensions: usize, message: impl Into<String>) -> Self {
        Self {
            dimensions,
            message: message.into(),
        }
    }
}

#[async_trait]
impl Embedder for FailingEmbedder {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn id(&self) -> &str {
        "failing"
    }

    async fn embed(&self, _text: &str) -> MemoryResult<Embedding> {
        Err(MemoryError::embedding(self.message.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_is_symmetric_and_bounded() {
        let a = Embedding(vec![1.0, 0.0, 0.0]);
        let b = Embedding(vec![1.0, 1.0, 0.0]);
        let c = Embedding(vec![-1.0, 0.0, 0.0]);
        assert!((a.cosine(&b) - 0.7071067811865475).abs() < 1e-9);
        assert_eq!(a.cosine(&b), b.cosine(&a));
        assert_eq!(a.cosine(&c), -1.0);
        assert_eq!(a.cosine(&a), 1.0);
    }

    #[test]
    fn degenerate_vectors_score_zero() {
        let zero = Embedding::zeros(4);
        let unit = Embedding(vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(zero.cosine(&unit), 0.0, "a zero vector has no direction");
        assert_eq!(
            Embedding(vec![1.0, 2.0]).cosine(&unit),
            0.0,
            "mismatched widths must not panic"
        );
        assert_eq!(Embedding(vec![1.0, 2.0]).distance(&unit), f64::INFINITY);
        assert_eq!(Embedding::default().cosine(&unit), 0.0);
    }

    #[test]
    fn normalisation_preserves_direction() {
        let raw = Embedding(vec![3.0, 4.0]);
        assert_eq!(raw.norm(), 5.0);
        let unit = raw.normalized();
        assert!((unit.norm() - 1.0).abs() < 1e-12);
        assert!((raw.cosine(&unit) - 1.0).abs() < 1e-12);
        assert_eq!(Embedding::zeros(3).normalized(), Embedding::zeros(3));
    }

    #[test]
    fn mean_ignores_ragged_vectors() {
        let a = Embedding(vec![1.0, 0.0]);
        let b = Embedding(vec![1.0, 2.0]);
        let short = Embedding(vec![5.0]);
        let pooled = Embedding::mean(&[a.clone(), b.clone(), short]).unwrap();
        assert_eq!(pooled, Embedding(vec![1.0, 1.0]));
        assert!(Embedding::mean(&[]).is_none());
    }

    #[test]
    fn top_k_keeps_the_largest_components() {
        let vector = Embedding(vec![0.1, -5.0, 3.0, 2.0]);
        let top = vector.top_k(2);
        // Ordering is by signed magnitude, so the negative outlier ranks last.
        assert_eq!(top[0], (2, 3.0));
        assert_eq!(top[1], (3, 2.0));
        assert_eq!(vector.top_k(0).len(), 0);
    }

    #[tokio::test]
    async fn hashing_is_deterministic_and_shared() {
        let embedder = HashingEmbedder::new(128);
        let first = embedder.embed("the agent plans with tools").await.unwrap();
        let second = embedder.embed("the agent plans with tools").await.unwrap();
        assert_eq!(first, second, "the same text must embed identically");
        assert_eq!(first.dimensions(), 128);
        assert!((first.norm() - 1.0).abs() < 1e-9, "vectors are unit length");
    }

    #[tokio::test]
    async fn related_text_scores_higher_than_unrelated() {
        let embedder = HashingEmbedder::new(512);
        let query = embedder.embed("rust tokio ecs framework").await.unwrap();
        let close = embedder.embed("an ecs framework built on tokio").await.unwrap();
        let far = embedder.embed("banana pancakes on a sunday").await.unwrap();
        assert!(
            query.cosine(&close) > query.cosine(&far),
            "{} must beat {}",
            query.cosine(&close),
            query.cosine(&far)
        );
        assert!(query.cosine(&far) < 0.3, "unrelated text should not look similar");
    }

    #[tokio::test]
    async fn negation_moves_the_vector_without_inverting_it() {
        // A hashing embedder has no notion of scope: `not good` shares most of its
        // n-grams with `good`, so negation shifts the vector without flipping it.
        // Recall that depends on polarity belongs to the model, not this embedder.
        let embedder = HashingEmbedder::new(512).with_trigram_weight(0.0);
        let positive = embedder.embed("the result is good").await.unwrap();
        let negated = embedder.embed("the result is not good").await.unwrap();
        let unrelated = embedder.embed("the sky is blue today").await.unwrap();
        let moved = positive.cosine(&negated);
        assert!(moved < 0.99, "the extra word must register: {moved}");
        assert!(
            moved > positive.cosine(&unrelated),
            "a negated restatement still beats unrelated text: {moved}"
        );
        assert_eq!(moved, positive.cosine(&negated), "scoring is pure");
    }

    #[tokio::test]
    async fn word_order_matters_little_but_word_choice_does() {
        let embedder = HashingEmbedder::new(256);
        let a = embedder.embed("alpha beta gamma").await.unwrap();
        let b = embedder.embed("gamma beta alpha").await.unwrap();
        let c = embedder.embed("delta epsilon zeta").await.unwrap();
        assert!(a.cosine(&b) > a.cosine(&c));
    }

    #[tokio::test]
    async fn batch_embedding_keeps_order() {
        let embedder = HashingEmbedder::new(64);
        let texts: Vec<String> = vec!["one".into(), "two".into(), "three".into()];
        let vectors = embedder.embed_all(&texts).await.unwrap();
        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[1], embedder.embed("two").await.unwrap());
        assert!(
            embedder
                .embed_all(&["ok".to_string(), "bad".to_string()])
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn random_embedder_decorrelates_texts() {
        let embedder = RandomEmbedder::new(256);
        let a = embedder.embed("alpha").await.unwrap();
        let b = embedder.embed("alpha-beta").await.unwrap();
        assert!(a.cosine(&b).abs() < 0.4, "shared prefixes must not dominate");
        assert_eq!(a, embedder.embed("alpha").await.unwrap());
        assert_eq!(a.dimensions(), 256);
    }

    #[tokio::test]
    async fn constant_embedder_makes_everything_identical() {
        let embedder = ConstantEmbedder::new(8);
        let a = embedder.embed("anything").await.unwrap();
        let b = embedder.embed("other").await.unwrap();
        assert_eq!(a.cosine(&b), 1.0);
        assert!(embedder.is_same("a", "b").await.unwrap());
    }

    #[tokio::test]
    async fn character_embedder_counts_alphabet() {
        let embedder = CharacterEmbedder::new(26);
        let text = embedder.embed("abcabc").await.unwrap();
        let half = embedder.embed("abc").await.unwrap();
        assert_eq!(text, half, "unit vectors erase the doubling");
        assert_ne!(text, embedder.embed("xyz").await.unwrap());
    }

    #[tokio::test]
    async fn failing_embedder_reports_its_message() {
        let embedder = FailingEmbedder::new(32, "model unavailable");
        let error = embedder.embed("text").await.unwrap_err();
        assert_eq!(error.to_string(), "embedding failed: model unavailable");
        let batch = embedder.embed_all(&["x".to_string()]).await.unwrap_err();
        assert_eq!(batch, error);
    }

    #[tokio::test]
    async fn is_same_compares_vectors() {
        let embedder = HashingEmbedder::new(128);
        assert!(embedder.is_same("same", "same").await.unwrap());
        assert!(!embedder.is_same("same", "different").await.unwrap());
    }

    #[test]
    fn tiny_dimension_requests_are_clamped() {
        assert_eq!(HashingEmbedder::new(1).dimensions(), 8);
        assert_eq!(RandomEmbedder::new(0).dimensions(), 8);
        assert_eq!(ConstantEmbedder::new(1).dimensions(), 2);
        assert_eq!(CharacterEmbedder::new(1_000).dimensions(), 128);
    }
}
