//! Component, resource and message marker traits.

/// Data attached to an [`Entity`](crate::Entity).
///
/// The trait is implemented automatically: every `Send + Sync + 'static` type is
/// a component, so plain structs and `#[derive]`ed data models need no boilerplate
/// and no macro import. Keeping the bound `Sync` is what allows read-only systems
/// to share component storage across concurrently running tokio tasks.
///
/// ```
/// use xihe_core::World;
///
/// #[derive(Debug)]
/// struct Counter { value: u32 }
///
/// let mut world = World::new();
/// let e = world.spawn();
/// world.insert(e, Counter { value: 1 });
/// assert_eq!(world.get::<Counter>(e).unwrap().value, 1);
/// ```
pub trait Component: Send + Sync + 'static {}

impl<T: Send + Sync + 'static> Component for T {}

/// Shared, singleton state stored in the [`World`](crate::World) instead of on an
/// entity: provider clients, tool registries, token ledgers, configuration.
///
/// Like [`Component`], this is a blanket marker. A resource must be cheaply
/// constructible or inserted before first use.
pub trait Resource: Send + Sync + 'static {}

impl<T: Send + Sync + 'static> Resource for T {}

/// Payload of the [`Events`](crate::Events) buffer and of the async message bus.
///
/// Messages are `Clone` because a broadcast may deliver them to several agent
/// entities per frame; keep payloads small and pass heavy data by
/// [`Entity`](crate::Entity) handle instead.
pub trait Message: Send + Sync + Clone + 'static {}

impl<T: Send + Sync + Clone + 'static> Message for T {}
