//! The world: entity index, component storages and resources.

use crate::component::{Component, Resource};
use crate::entity::{Entities, Entity};
use crate::query::{Fetch, Query};
use crate::resources::Resources;
use crate::storage::Storages;

/// Hard ceiling on allocated entity slots (`u32` index space minus the sentinel).
pub const MAX_ENTITIES: usize = u32::MAX as usize - 1;

/// Components inserted together, typically at spawn time.
///
/// Implemented for tuples of up to six components, so
/// `world.spawn_with((profile, conversation))` needs no extra trait imports.
pub trait Bundle: Send {
    /// Inserts every member into `entity`.
    fn insert_into(self, world: &mut World, entity: Entity);
}

macro_rules! impl_bundle {
    ($($name:ident),+) => {
        impl<$($name: Component),+> Bundle for ($($name,)*) {
            fn insert_into(self, world: &mut World, entity: Entity) {
                #[allow(non_snake_case)]
                let ($($name,)*) = self;
                $(world.insert(entity, $name);)+
            }
        }
    };
}

impl_bundle!(A);
impl_bundle!(A, B);
impl_bundle!(A, B, C);
impl_bundle!(A, B, C, D);
impl_bundle!(A, B, C, D, E);
impl_bundle!(A, B, C, D, E, F);

/// Read-only view handed to [`World::edit_with`] closures, so one component type
/// can be mutated while every other component and resource stays readable.
pub struct ReadContext<'w> {
    storages: &'w Storages,
    resources: &'w Resources,
}

impl<'w> ReadContext<'w> {
    /// Shared component of `entity`.
    #[inline]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&'w T> {
        self.storages.get::<T>(entity)
    }

    /// `true` when `entity` holds `T`.
    #[inline]
    pub fn contains<T: Component>(&self, entity: Entity) -> bool {
        self.storages.contains::<T>(entity)
    }

    /// Shared resource.
    #[inline]
    pub fn resource<R: Resource>(&self) -> Option<&'w R> {
        self.resources.get::<R>()
    }

    /// Number of entities holding `T`.
    #[inline]
    pub fn count<T: Component>(&self) -> usize {
        self.storages.count_of::<T>()
    }
}

/// Owns every entity, component and resource of a simulation.
///
/// A `World` is plain data: no locks, no `unsafe`, no async. Concurrency is a
/// property of the driver (`xihe-schedule` and `xihe-runtime`), which shares it
/// through an `RwLock` and only ever hands systems shared or exclusive access.
#[derive(Debug, Default)]
pub struct World {
    entities: Entities,
    storages: Storages,
    resources: Resources,
}

impl World {
    /// Creates an empty world.
    pub fn new() -> Self {
        Self::default()
    }

    // --- entities -----------------------------------------------------------

    /// Allocates a bare entity.
    pub fn spawn(&mut self) -> Entity {
        self.entities.alloc()
    }

    /// Allocates an entity and fills it with a [`Bundle`].
    pub fn spawn_with<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.spawn();
        bundle.insert_into(self, entity);
        entity
    }

    /// Allocates `count` entities.
    pub fn spawn_many(&mut self, count: usize) -> Vec<Entity> {
        self.entities.alloc_many(count)
    }

    /// Frees an entity and drops its components. `true` when it was alive.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        self.storages.remove_entity(entity);
        self.entities.dealloc(entity)
    }

    /// Frees every live entity, keeping resources. Returns how many were dropped.
    pub fn despawn_all(&mut self) -> usize {
        let dead: Vec<Entity> = self.entities.iter().collect();
        let count = dead.len();
        for entity in dead {
            self.despawn(entity);
        }
        count
    }

    /// `true` when the handle still points at a live entity.
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// Live entity count.
    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// `true` when no entity is alive.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Iterates live entities in slot order.
    pub fn entities(&self) -> impl Iterator<Item = Entity> + Clone + '_ {
        self.entities.iter()
    }

    /// Borrowed entity index.
    #[inline]
    pub fn entity_index(&self) -> &Entities {
        &self.entities
    }

    // --- components ---------------------------------------------------------

    /// Inserts `component`, returning the value it replaced.
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) -> Option<T> {
        self.storages.insert(entity, component)
    }

    /// Inserts a whole [`Bundle`] into an existing entity.
    pub fn insert_bundle<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        bundle.insert_into(self, entity);
    }

    /// Shared component reference.
    #[inline]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.storages.get::<T>(entity)
    }

    /// Exclusive component reference.
    #[inline]
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        self.storages.get_mut::<T>(entity)
    }

    /// `true` when `entity` is alive and holds `T`.
    #[inline]
    pub fn contains<T: Component>(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity) && self.storages.contains::<T>(entity)
    }

    /// Removes and returns the component of `entity`.
    #[inline]
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.storages.remove::<T>(entity)
    }

    /// Number of entities holding `T`.
    #[inline]
    pub fn count<T: Component>(&self) -> usize {
        self.storages.count_of::<T>()
    }

    // --- queries ------------------------------------------------------------

    /// Runs a read-only query, e.g. `world.query::<(&Profile, &Conversation)>()`.
    pub fn query<'w, Q: Fetch<'w>>(&'w self) -> Query<'w, Q> {
        Query::new(&self.storages, &self.entities)
    }

    /// The only entity holding `T`, or `None` when no entity does.
    ///
    /// Panics when two entities hold it: singleton components name a process-wide
    /// resource by entity handle (a pipeline, a mailbox), and two of them is a
    /// programming error rather than a runtime choice.
    pub fn single<T: Component>(&self) -> Option<Entity> {
        let found: Vec<Entity> = self.storages.cursor_of::<T>().take(2).collect();
        match found.as_slice() {
            [] => None,
            [one] => Some(*one),
            _ => panic!(
                "`{}` is used as a singleton but {} entities hold it",
                std::any::type_name::<T>(),
                self.storages.count_of::<T>()
            ),
        }
    }

    /// Mutable access to every `T`, one component type at a time.
    ///
    /// This is the safe counterpart of a `&mut T` query term: the storage of `T`
    /// leaves the map for the duration of `f`, so the closure gets `&mut T` while
    /// the rest of the world stays readable — see [`World::edit_with`].
    pub fn edit<T: Component, F>(&mut self, mut f: F)
    where
        F: FnMut(Entity, &mut T),
    {
        self.storages.with_taken::<T, ()>(|set, _others| {
            for index in 0..set.len() {
                if let Some((entity, component)) = set.slot_at_mut(index) {
                    f(entity, component);
                }
            }
        });
    }

    /// [`World::edit`] plus a [`ReadContext`] over the other components.
    pub fn edit_with<T: Component, F>(&mut self, mut f: F)
    where
        F: FnMut(Entity, &mut T, &ReadContext<'_>),
    {
        let resources = &self.resources;
        self.storages.with_taken::<T, ()>(|set, others| {
            for index in 0..set.len() {
                let (entity, component) = match set.slot_at_mut(index) {
                    Some(slot) => slot,
                    None => continue,
                };
                let ctx = ReadContext {
                    storages: others,
                    resources,
                };
                f(entity, component, &ctx);
            }
        });
    }

    // --- resources ----------------------------------------------------------

    /// Inserts a resource, returning the previous value.
    pub fn insert_resource<R: Resource>(&mut self, value: R) -> Option<R> {
        self.resources.insert(value)
    }

    /// Ensures `R::default()` exists and returns it.
    pub fn init_resource<R: Resource + Default>(&mut self) -> &mut R {
        self.resources.init::<R>()
    }

    /// Shared resource reference.
    #[inline]
    pub fn resource<R: Resource>(&self) -> Option<&R> {
        self.resources.get::<R>()
    }

    /// Exclusive resource reference.
    #[inline]
    pub fn resource_mut<R: Resource>(&mut self) -> Option<&mut R> {
        self.resources.get_mut::<R>()
    }

    /// Removes a resource.
    #[inline]
    pub fn take_resource<R: Resource>(&mut self) -> Option<R> {
        self.resources.remove::<R>()
    }

    /// `true` when the resource exists.
    #[inline]
    pub fn has_resource<R: Resource>(&self) -> bool {
        self.resources.contains::<R>()
    }

    // --- introspection ------------------------------------------------------

    /// `(component type name, entity count)` pairs, sorted by name.
    pub fn component_counts(&self) -> Vec<(&'static str, usize)> {
        self.storages.counts()
    }

    /// Resource type names, sorted.
    pub fn resource_names(&self) -> Vec<&'static str> {
        self.resources.names()
    }

    /// Aggregate counters used by the observer layer.
    #[inline]
    pub fn stats(&self) -> WorldStats {
        WorldStats {
            entities: self.entities.len(),
            component_types: self.storages.len(),
            components: self.storages.total_components(),
            resources: self.resources.len(),
        }
    }

    /// Drops every entity, component and resource.
    pub fn clear(&mut self) {
        self.entities = Entities::default();
        self.storages.clear();
        self.resources.clear();
    }
}

/// Counters describing the size of a world.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct WorldStats {
    /// Live entities.
    pub entities: usize,
    /// Distinct component types in use.
    pub component_types: usize,
    /// Total component instances.
    pub components: usize,
    /// Resources in the type map.
    pub resources: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Counter {
        value: u32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Label(&'static str);

    #[derive(Debug, Default)]
    struct Frame(u64);

    #[test]
    fn spawn_insert_despawn_roundtrip() {
        let mut world = World::new();
        let entity = world.spawn_with((Counter { value: 1 }, Label("a")));
        assert_eq!(world.len(), 1);
        assert_eq!(world.get::<Label>(entity).unwrap().0, "a");
        world.remove::<Label>(entity);
        assert!(!world.contains::<Label>(entity));
        assert_eq!(world.count::<Counter>(), 1);

        assert!(world.despawn(entity));
        assert!(!world.despawn(entity));
        assert!(world.is_empty());
        assert_eq!(world.count::<Counter>(), 0);
    }

    #[test]
    fn resources_are_typed_singletons() {
        let mut world = World::new();
        world.insert_resource(Frame(3));
        assert_eq!(world.resource::<Frame>().unwrap().0, 3);
        world.resource_mut::<Frame>().unwrap().0 = 4;
        assert_eq!(world.init_resource::<Frame>().0, 4, "init keeps the value");
        assert_eq!(world.take_resource::<Frame>().unwrap().0, 4);
        assert!(!world.has_resource::<Frame>());
    }

    #[test]
    fn edit_mutates_in_place_and_restores_storage() {
        let mut world = World::new();
        let a = world.spawn_with((Counter { value: 1 }, Label("ab")));
        let b = world.spawn_with((Counter { value: 2 }, Label("b")));
        world.spawn_with((Counter { value: 3 },));

        world.edit::<Counter, _>(|_entity, counter| counter.value *= 10);
        assert_eq!(world.get::<Counter>(a).unwrap().value, 10);

        world.edit_with::<Counter, _>(|entity, counter, ctx| {
            if let Some(label) = ctx.get::<Label>(entity) {
                counter.value += label.0.len() as u32;
            }
        });
        assert_eq!(world.get::<Counter>(a).unwrap().value, 12);
        assert_eq!(world.get::<Counter>(b).unwrap().value, 21);
        assert_eq!(world.count::<Counter>(), 3, "storage was put back");
    }

    #[test]
    fn single_finds_and_panics() {
        let mut world = World::new();
        assert_eq!(world.single::<Label>(), None);
        let a = world.spawn_with((Label("only"),));
        assert_eq!(world.single::<Label>(), Some(a));
        world.spawn_with((Label("second"),));
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.single::<Label>()
        }));
        assert!(panic.is_err());
    }

    #[test]
    #[should_panic(expected = "singleton")]
    fn single_rejects_duplicates() {
        let mut world = World::new();
        world.spawn_with((Label("a"),));
        world.spawn_with((Label("b"),));
        world.single::<Label>();
    }

    #[test]
    fn stats_and_clear() {
        let mut world = World::new();
        world.spawn_with((Counter { value: 1 }, Label("x")));
        world.insert_resource(Frame(1));
        assert_eq!(
            world.stats(),
            WorldStats {
                entities: 1,
                component_types: 2,
                components: 2,
                resources: 1,
            }
        );
        assert!(
            world
                .component_counts()
                .iter()
                .any(|(name, count)| *count == 1 && name.ends_with("Counter"))
        );
        world.clear();
        assert_eq!(world.stats(), WorldStats::default());
    }

    #[test]
    fn bundles_up_to_six_components() {
        let mut world = World::new();
        let e = world.spawn_with((
            Counter { value: 1 },
            Label("l"),
            Frame(2),
            Counter { value: 3 },
            String::from("s"),
            7u8,
        ));
        assert_eq!(world.get::<Counter>(e).unwrap().value, 3, "last wins per type");
        assert_eq!(world.get::<u8>(e), Some(&7));
    }
}
