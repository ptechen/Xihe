//! Xihe 的智能体记忆：嵌入、向量召回、对话窗口化、
//! 工作状态与反思。
//!
//! 本 crate 组织为一个栈，每层只与其下一层对话，
//! 绝不与上一层对话：
//!
//! | 层 | 它回答什么 |
//! |---|---|
//! | [`embed`] | 一段文本如何变成可比较的向量？ |
//! | [`record`] | 什么是记忆，何时值得保留？ |
//! | [`store`] | 如何插入与检索许多记录？ |
//! | [`conversation`] | 如何把对话保持在 token 预算内？ |
//! | [`reflection`] | 压缩后的对话告诉智能体接下来做什么？ |
//! | [`components`] | 一个智能体当前相信、计划与记录了什么？ |
//! | [`persist`]（feature `persist`） | 这一切如何在重启后存续？ |
//!
//! 这里没有任何东西与模型对话：[`Summariser`](reflection::Summariser)
//! 与 [`Embedder`](embed::Embedder) 接缝让智能体层插入 LLM，
//! 该层已持有提供方访问。这使本 crate 可离线测试，
//! 且不带网络依赖。
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
//! // 工作状态记录也会匹配：召回搜索此智能体拥有的一切，
//! // 而不仅是事实。
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

/// 本 crate 知道的类型，在智能体层取用处重新导出。
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
