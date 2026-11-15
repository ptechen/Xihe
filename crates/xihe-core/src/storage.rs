//! Component storage: a `Send`-safe sparse set per component type.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

use crate::component::Component;
use crate::entity::{Entity, Entities};

/// Iterator over the entities of a storage, erased so queries stay monomorphic
/// in the *term* tuple rather than in every component type.
pub type EntityCursor<'w> = Box<dyn Iterator<Item = Entity> + Send + 'w>;

/// Type-erased component storage.
pub trait Storage: Any + Send + Sync {
    /// Component type this storage holds.
    fn component_type(&self) -> TypeId;

    /// Component type name, for reports and dashboards.
    fn component_name(&self) -> &'static str;

    /// Number of entities holding the component.
    fn len(&self) -> usize;

    /// `true` when no entity holds the component.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `true` when `entity` holds the component.
    fn contains(&self, entity: Entity) -> bool;

    /// Removes the component of `entity`, reporting whether it was present.
    fn remove(&mut self, entity: Entity) -> bool;

    /// Drops every stored component.
    fn clear_storage(&mut self);

    /// Downcast helper.
    fn as_any(&self) -> &dyn Any;

    /// Mutable downcast helper.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Snapshot of the stored entities (dense order).
    fn snapshot_entities(&self) -> Vec<Entity>;

    /// Borrowed cursor over the stored entities.
    fn cursor(&self) -> EntityCursor<'_>;
}

/// Dense-array component storage: `dense` holds `(Entity, T)` pairs, `sparse`
/// maps an entity slot index to its position in `dense`.
///
/// Insert/remove is O(1) with a swap-back, iteration is cache friendly, and an
/// entity index lookup never walks the storage.
pub struct SparseSet<T> {
    sparse: Vec<Option<u32>>,
    dense: Vec<(Entity, T)>,
}

impl<T> Default for SparseSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SparseSet<T> {
    /// Creates an empty storage.
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::new(),
        }
    }

    /// Number of stored components.
    #[inline]
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// `true` when empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Dense position of `entity`, if stored.
    #[inline]
    fn position(&self, entity: Entity) -> Option<usize> {
        self.sparse
            .get(entity.index() as usize)
            .and_then(|slot| *slot)
            .map(|pos| pos as usize)
    }

    /// `true` when `entity` has a component here.
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.position(entity).is_some()
    }

    fn grow(&mut self, entity: Entity) {
        let needed = entity.index() as usize + 1;
        if self.sparse.len() < needed {
            self.sparse.resize(needed, None);
        }
    }

    /// Inserts `value`, returning the component previously stored for `entity`.
    pub fn insert(&mut self, entity: Entity, value: T) -> Option<T> {
        self.grow(entity);
        match self.position(entity) {
            Some(pos) => Some(std::mem::replace(&mut self.dense[pos].1, value)),
            None => {
                self.sparse[entity.index() as usize] = Some(self.dense.len() as u32);
                self.dense.push((entity, value));
                None
            }
        }
    }

    /// Shared access to the component of `entity`.
    #[inline]
    pub fn get(&self, entity: Entity) -> Option<&T> {
        self.position(entity).map(|pos| &self.dense[pos].1)
    }

    /// Exclusive access to the component of `entity`.
    #[inline]
    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        self.position(entity)
            .map(|pos| &mut self.dense[pos].1)
    }

    /// Component of `entity` plus its dense position.
    #[inline]
    pub fn find(&self, entity: Entity) -> Option<(usize, &T)> {
        self.position(entity).map(|pos| (pos, &self.dense[pos].1))
    }

    /// Removes and returns the component of `entity`.
    pub fn remove(&mut self, entity: Entity) -> Option<T> {
        let pos = self.position(entity)?;
        let last = self.dense.len() - 1;
        let moved = self.dense.swap_remove(pos);
        if pos != last {
            let moved_entity = self.dense[pos].0;
            self.sparse[moved_entity.index() as usize] = Some(pos as u32);
        }
        // `swap_remove` returned the requested pair; the displaced pair was moved
        // into `pos` and is now reachable, so only clear the removed slot.
        self.sparse[entity.index() as usize] = None;
        Some(moved.1)
    }

    /// Iterates `(Entity, &T)` in dense order.
    pub fn iter(&self) -> std::slice::Iter<'_, (Entity, T)> {
        self.dense.iter()
    }

    /// Iterates `(Entity, &mut T)` in dense order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, (Entity, T)> {
        self.dense.iter_mut()
    }

    /// Borrows the pair at dense position `index`.
    #[inline]
    pub fn slot_at_mut(&mut self, index: usize) -> Option<(Entity, &mut T)> {
        self.dense.get_mut(index).map(|(entity, value)| (*entity, value))
    }

    /// Entity list in dense order.
    pub fn entities(&self) -> impl Iterator<Item = Entity> + Clone + '_ {
        self.dense.iter().map(|(entity, _)| *entity)
    }

    /// Takes ownership of every stored component, leaving the storage empty.
    pub fn drain(&mut self) -> std::vec::Drain<'_, (Entity, T)> {
        self.sparse.clear();
        self.dense.drain(..)
    }
}

impl<T: Component> Storage for SparseSet<T> {
    fn component_type(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn component_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    fn len(&self) -> usize {
        SparseSet::len(self)
    }

    fn contains(&self, entity: Entity) -> bool {
        SparseSet::contains(self, entity)
    }

    fn remove(&mut self, entity: Entity) -> bool {
        SparseSet::remove(self, entity).is_some()
    }

    fn clear_storage(&mut self) {
        self.sparse.clear();
        self.dense.clear();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn snapshot_entities(&self) -> Vec<Entity> {
        self.entities().collect()
    }

    fn cursor(&self) -> EntityCursor<'_> {
        Box::new(self.entities())
    }
}

impl<T: fmt::Debug + Component> fmt::Debug for SparseSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseSet")
            .field("type", &std::any::type_name::<T>())
            .field("len", &self.len())
            .finish()
    }
}

/// Collection of one [`SparseSet`] per component type.
#[derive(Default)]
pub struct Storages {
    map: HashMap<TypeId, Box<dyn Storage>>,
}

impl Storages {
    /// Creates an empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of component types that currently have a storage.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// `true` when no component type has been inserted yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Component type ids present, sorted for deterministic reporting.
    pub fn types(&self) -> Vec<TypeId> {
        let mut types: Vec<TypeId> = self.map.keys().copied().collect();
        types.sort();
        types
    }

    /// Concrete storage for `T`, creating it when needed.
    pub fn storage_of_mut<T: Component>(&mut self) -> &mut SparseSet<T> {
        let type_id = TypeId::of::<T>();
        self.map
            .entry(type_id)
            .or_insert_with(|| Box::new(SparseSet::<T>::new()))
            .as_any_mut()
            .downcast_mut::<SparseSet<T>>()
            .unwrap_or_else(|| {
                panic!(
                    "storage for `{}` was rebuilt by a different type",
                    std::any::type_name::<T>()
                )
            })
    }

    /// Concrete storage for `T` when it exists.
    pub fn storage_of<T: Component>(&self) -> Option<&SparseSet<T>> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|storage| storage.as_any().downcast_ref::<SparseSet<T>>())
    }

    /// Shared component of `entity`.
    #[inline]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.storage_of::<T>().and_then(|set| set.get(entity))
    }

    /// Exclusive component of `entity`.
    #[inline]
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        self.storage_of_mut::<T>().get_mut(entity)
    }

    /// `true` when `entity` holds `T`.
    #[inline]
    pub fn contains<T: Component>(&self, entity: Entity) -> bool {
        self.storage_of::<T>().is_some_and(|set| set.contains(entity))
    }

    /// Inserts `value`, returning the previous component of `entity`.
    #[inline]
    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) -> Option<T> {
        self.storage_of_mut::<T>().insert(entity, value)
    }

    /// Removes the component of `entity`.
    #[inline]
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|storage| storage.as_any_mut().downcast_mut::<SparseSet<T>>())
            .and_then(|set| set.remove(entity))
    }

    /// Removes `entity` from every storage, used when despawning.
    pub fn remove_entity(&mut self, entity: Entity) {
        for storage in self.map.values_mut() {
            storage.remove(entity);
        }
    }

    /// Number of entities holding `T`.
    #[inline]
    pub fn count_of<T: Component>(&self) -> usize {
        self.storage_of::<T>().map_or(0, SparseSet::len)
    }

    /// Drops all components of all types.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Total component instances across every type.
    pub fn total_components(&self) -> usize {
        self.map.values().map(|storage| storage.len()).sum()
    }

    /// `(component name, entity count)` pairs, sorted by name.
    pub fn counts(&self) -> Vec<(&'static str, usize)> {
        let mut counts: Vec<(&'static str, usize)> = self
            .map
            .values()
            .map(|storage| (storage.component_name(), storage.len()))
            .collect();
        counts.sort_by(|left, right| left.0.cmp(right.0));
        counts
    }

    /// Entity list of the storage of `T`.
    pub fn snapshot_of<T: Component>(&self) -> Vec<Entity> {
        self.storage_of::<T>()
            .map(|set| set.entities().collect())
            .unwrap_or_default()
    }

    /// Cursor over the entities of `T`, empty when the type is unknown.
    pub fn cursor_of<T: Component>(&self) -> EntityCursor<'_> {
        match self.storage_of::<T>() {
            Some(set) => Box::new(set.entities()),
            None => Box::new(std::iter::empty()),
        }
    }

    /// Cursor over every live entity of the world.
    pub fn cursor_of_all<'w>(&'w self, entities: &'w Entities) -> EntityCursor<'w> {
        Box::new(entities.iter())
    }

    /// Removes the storage of `T` from the map, hands it to `f` mutably together
    /// with the remaining read-only storages, then puts it back — including on
    /// unwind.
    ///
    /// This is the safe core of [`World::edit`](crate::World::edit): taking the
    /// storage out of the map gives one exclusive view of `T` while every other
    /// component type stays readable, which a `HashMap` of `&mut` values could
    /// never prove to the borrow checker.
    pub fn with_taken<T: Component, R>(&mut self, f: impl FnOnce(&mut SparseSet<T>, &Self) -> R) -> Option<R> {
        struct Restore<'s, T: Component> {
            storages: &'s mut Storages,
            taken: Option<Box<dyn Storage>>,
            marker: std::marker::PhantomData<T>,
        }

        impl<T: Component> Drop for Restore<'_, T> {
            fn drop(&mut self) {
                if let Some(storage) = self.taken.take() {
                    self.storages.map.insert(TypeId::of::<T>(), storage);
                }
            }
        }

        let taken = self.map.remove(&TypeId::of::<T>());
        let mut restore = Restore::<T> {
            storages: self,
            taken,
            marker: std::marker::PhantomData,
        };
        let taken = &mut restore.taken;
        let others = &*restore.storages;
        taken
            .as_mut()
            .and_then(|storage| storage.as_any_mut().downcast_mut::<SparseSet<T>>())
            .map(|set| f(set, others))
    }
}

impl fmt::Debug for Storages {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Storages")
            .field("types", &self.map.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Health(u32);

    #[derive(Clone, Debug, PartialEq)]
    struct Name(String);

    fn entity(index: u32) -> Entity {
        Entity::from_raw(index, 0)
    }

    #[test]
    fn insert_get_remove_roundtrip() {
        let mut set = SparseSet::<Health>::new();
        assert_eq!(set.insert(entity(1), Health(100)), None);
        assert_eq!(set.insert(entity(4), Health(50)), None);
        assert_eq!(set.insert(entity(1), Health(10)), Some(Health(100)));
        assert_eq!(set.get(entity(4)), Some(&Health(50)));
        assert_eq!(set.len(), 2);
        assert_eq!(set.remove(entity(1)), Some(Health(10)));
        assert_eq!(set.remove(entity(1)), None);
        assert_eq!(set.entities().collect::<Vec<_>>(), vec![entity(4)]);
        assert_eq!(set.get(entity(1)), None);
    }

    #[test]
    fn swap_remove_keeps_indices_consistent() {
        let mut set = SparseSet::<Health>::new();
        for i in 0..5u32 {
            set.insert(entity(i), Health(i));
        }
        assert_eq!(set.remove(entity(0)), Some(Health(0)));
        for i in 1..5u32 {
            assert_eq!(set.get(entity(i)), Some(&Health(i)), "entity {i} moved");
        }
        assert_eq!(set.len(), 4);
        assert!(!set.contains(entity(0)));
    }

    #[test]
    fn with_taken_restores_storage_and_mixes_borrows() {
        let mut storages = Storages::new();
        storages.insert(entity(0), Health(1));
        storages.insert(entity(0), Name("a".into()));
        storages.insert(entity(1), Health(2));

        let seen = storages.with_taken::<Health, Vec<(u32, u32)>>(|set, others| {
            let mut out = Vec::new();
            for i in 0..set.len() {
                let (e, health) = set.slot_at_mut(i).unwrap();
                health.0 *= 10;
                out.push((health.0, others.get::<Name>(e).map_or(0, |n| n.0.len() as u32)));
            }
            out
        });
        assert_eq!(seen, Some(vec![(10, 1), (20, 0)]));
        assert_eq!(storages.get::<Health>(entity(0)), Some(&Health(10)));
        assert_eq!(storages.get::<Name>(entity(1)), None);
        assert_eq!(storages.with_taken::<Name, u32>(|_, _| 7), Some(7));
        assert_eq!(storages.with_taken::<Health, u32>(|_, _| 1), Some(1));
    }

    #[test]
    fn with_taken_restores_storage_after_panic() {
        let mut storages = Storages::new();
        storages.insert(entity(9), Health(9));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            storages.with_taken::<Health, ()>(|_, _| panic!("boom"))
        }));
        assert!(result.is_err());
        assert_eq!(
            storages.get::<Health>(entity(9)),
            Some(&Health(9)),
            "storage must be reinserted even when the closure unwinds"
        );
    }
}
