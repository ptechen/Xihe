//! # xihe-core
//!
//! Xihe 智能体框架的实体-组件部分：[`World`]、[`Entity`]、
//! 组件存储、类型化查询、资源与帧事件。
//!
//! * **无 `unsafe`** —— 本 crate 在文件顶部禁止它，可变迭代 API 的设计
//!   让借用检查器（而非裸指针）来证明组件视图不会别名。
//! * **无异步、无锁** —— `World` 是纯数据。并发属于驱动器
//!   ([`xihe-schedule`](https://docs.rs/xihe-schedule))，由它决定哪些
//!   系统可以同时运行。
//! * **标记是全覆盖实现** —— 任何 `Send + Sync + 'static` 类型都是
//!   [`Component`] 和 [`Resource`]；加上 `Clone` 即成为 [`Message`]。
//!   无需 derive、无需导入、无需样板。
//!
//! ```
//! use xihe_core::World;
//!
//! #[derive(Debug)]
//! struct Plan { goal: String, steps: u32 }
//!
//! let mut world = World::new();
//! let agent = world.spawn_with((Plan { goal: "ship it".into(), steps: 3 },));
//!
//! world.edit::<Plan, _>(|_entity, plan| plan.steps += 1);
//! let plan = world.get::<Plan>(agent).unwrap();
//! assert_eq!(plan.steps, 4);
//!
//! for plan in world.query::<&Plan>() {
//!     assert_eq!(plan.goal, "ship it");
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod component;
pub mod entity;
pub mod events;
pub mod query;
pub mod resources;
pub mod storage;
pub mod world;

pub use component::{Component, Message, Resource};
pub use entity::{AliveIter, Entities, Entity, DANGLING};
pub use events::Events;
pub use query::{Fetch, FilteredQuery, Query};
pub use resources::Resources;
pub use storage::{EntityCursor, SparseSet, Storage, Storages};
pub use world::{Bundle, MAX_ENTITIES, ReadContext, World, WorldStats};

/// 用户通常一次性导入的全部内容。
pub mod prelude {
    pub use crate::{
        Bundle, Component, Entities, Entity, Events, Fetch, Message, Query, Resource, ReadContext,
        Storages, World, WorldStats, DANGLING,
    };
}
