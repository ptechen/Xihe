//! # xihe-core
//!
//! The entity-component half of the Xihe agent framework: [`World`], [`Entity`],
//! component storages, typed queries, resources and frame events.
//!
//! * **No `unsafe`** — the crate forbids it at the top of this file, and the
//!   mutable-iteration API is designed so the borrow checker, not raw pointers,
//!   proves that component views do not alias.
//! * **No async, no locks** — a `World` is plain data. Concurrency belongs to the
//!   driver ([`xihe-schedule`](https://docs.rs/xihe-schedule)), which decides which
//!   systems may run at the same time.
//! * **Markers are blanket impls** — every `Send + Sync + 'static` type is a
//!   [`Component`] and a [`Resource`]; adding `Clone` makes it a [`Message`]. No
//!   derive, no import, no boilerplate.
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

/// Everything a user normally imports at once.
pub mod prelude {
    pub use crate::{
        Bundle, Component, Entities, Entity, Events, Fetch, Message, Query, Resource, ReadContext,
        Storages, World, WorldStats, DANGLING,
    };
}
