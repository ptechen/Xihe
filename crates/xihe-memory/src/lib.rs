//! Agent memory for Xihe: embeddings, vector recall, dialogue windowing,
//! working state and reflection.
//!
//! The crate is arranged as a stack, each layer speaking to the one below it
//! and never to the one above:
//!
//! | layer | what it answers |
//! |---|---|
//! | [`embed`] | how does a piece of text become a comparable vector? |
//! | [`record`] | what is a memory, and when is it worth keeping? |
//! | [`store`] | how do we insert and retrieve many records? |
//! | [`conversation`] | how do we keep a dialogue inside a token budget? |
//! | [`reflection`] | what does a compressed dialogue tell the agent to do next? |
//! | [`components`] | what does one agent currently believe, plan and note? |
//! | [`persist`] (feature `persist`) | how does any of it survive a restart? |
//!
//! Nothing here talks to a model: the [`Summariser`](reflection::Summariser)
//! and [`Embedder`](embed::Embedder) seams let an LLM be plugged in by the
//! agent layer, which already owns provider access. That keeps this crate
//! offline-testable and free of a network dependency.
//!
//! ```
//! use std::sync::Arc;
//!
//! use xihe_core::Entity;
//! use xihe_memory::components::AgentMemory;
//! use xihe_memory::record::MemoryKind;
//! use xihe_memory::store::{InMemoryStore, MemoryStore};
//!
//! # async fn run() -> Result<bool, xihe_memory::MemoryError> {
//! let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
//! let mut memory = AgentMemory::new(store, Entity::from_raw(1, 1));
//! memory
//!     .set_focus(&["answer in one sentence".to_string()], 3)
//!     .await?;
//! memory
//!     .remember(MemoryKind::Fact, "the export has yesterday's rows")
//!     .await?;
//! // The working-state record matches too: recall searches everything this
//! // agent owns, not only the facts.
//! let hits = memory.recall("when does the export refresh?", 3).await?;
//! Ok(!hits.is_empty())
//! # }
//! # tokio::runtime::Builder::new_current_thread()
//! #     .enable_all().build().unwrap().block_on(async {
//! #     assert!(run().await.unwrap());
//! # });
//! ```

pub mod components;
pub mod conversation;
pub mod embed;
pub mod error;
pub mod record;
pub mod reflection;
pub mod store;

#[cfg(feature = "persist")]
pub mod persist;

pub use components::{AgentMemory, TaskPlan, WorkingState};
pub use conversation::{ConversationWindow, SessionMemory, Speaker, TrimReason, Turn};
pub use embed::{Embedder, Embedding, HashingEmbedder};
pub use error::{MemoryError, MemoryResult};
pub use record::{
    MemoryId, MemoryKind, MemoryQuery, MemoryRecord, RetentionPolicy, ScoredRecord, Timestamp, now_ms,
};
pub use reflection::{HeuristicSummariser, ReflectionBudget, ReflectionStep, Summariser, reflect};
pub use store::{InMemoryStore, MemoryStats, MemoryStore, SearchOptions};

#[cfg(feature = "persist")]
pub use persist::{MemoryFile, append_journal, read_records, restore_snapshot, save_snapshot};

/// The kinds this crate knows about, re-exported where the agent layer picks them.
pub mod prelude {
    pub use crate::components::{AgentMemory, TaskPlan};
    pub use crate::conversation::{ConversationWindow, SessionMemory, Speaker, Turn};
    pub use crate::embed::{Embedder, Embedding, HashingEmbedder};
    pub use crate::error::{MemoryError, MemoryResult};
    pub use crate::record::{MemoryKind, MemoryQuery, MemoryRecord, RetentionPolicy, ScoredRecord};
    pub use crate::reflection::{
        HeuristicSummariser, ReflectionBudget, ReflectionStep, Summariser,
    };
    pub use crate::store::{InMemoryStore, MemoryStore, SearchOptions};
}
