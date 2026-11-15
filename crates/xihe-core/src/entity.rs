//! Generation-checked entity identifiers and the alive-entity index.

use std::fmt;

/// A stable, generation-checked handle to a world slot.
///
/// [`Entity`] is `u32`-sized, `Copy` and hashable, so it is cheap to keep inside
/// components, message payloads and tool arguments. Recycled slots bump
/// [`Entity::generation`], which makes stale handles fail lookups instead of
/// aliasing a new entity.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct Entity {
    index: u32,
    generation: u32,
}

/// Sentinel that never resolves to a live entity.
pub const DANGLING: Entity = Entity {
    index: u32::MAX,
    generation: u32::MAX,
};

impl Entity {
    /// Builds an entity from raw parts. Only used by the allocator and by
    /// (de)serialization of persisted worlds.
    #[inline]
    pub const fn from_raw(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Slot index inside [`Entities`].
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Number of times the slot has been recycled plus one.
    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    /// `true` for [`DANGLING`].
    #[inline]
    pub const fn is_dangling(self) -> bool {
        self.index == u32::MAX
    }

    /// Returns `self` when `alive`, `None` otherwise. Useful in `Option` chains.
    #[inline]
    pub fn take_if(self, alive: bool) -> Option<Self> {
        if alive { Some(self) } else { None }
    }
}

impl fmt::Debug for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}/{}", self.index, self.generation)
    }
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.index, self.generation)
    }
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    generation: u32,
    alive: bool,
}

/// Slot table backing a [`World`](crate::World).
#[derive(Debug, Default)]
pub struct Entities {
    slots: Vec<Slot>,
    free: Vec<u32>,
    alive_count: usize,
}

impl Entities {
    /// Creates an empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates an entity, reusing the most recently freed slot.
    ///
    /// # Panics
    /// If more than `u32::MAX - 1` slots would be required.
    pub fn alloc(&mut self) -> Entity {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.alive = true;
            slot.generation += 1;
            self.alive_count += 1;
            Entity::from_raw(index, slot.generation)
        } else {
            let index = self.slots.len() as u32;
            assert!(
                (index as usize) < crate::world::MAX_ENTITIES,
                "entity index space exhausted ({index} slots)"
            );
            self.slots.push(Slot {
                generation: 0,
                alive: true,
            });
            self.alive_count += 1;
            Entity::from_raw(index, 0)
        }
    }

    /// Allocates `count` entities in slot order.
    pub fn alloc_many(&mut self, count: usize) -> Vec<Entity> {
        (0..count).map(|_| self.alloc()).collect()
    }

    /// Frees `entity`, returning `false` when it was already dead or stale.
    pub fn dealloc(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        self.slots[entity.index() as usize].alive = false;
        self.free.push(entity.index());
        self.alive_count -= 1;
        true
    }

    /// `true` when the handle still points at a live slot.
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.slots
            .get(entity.index() as usize)
            .is_some_and(|slot| slot.alive && slot.generation == entity.generation())
    }

    /// Current generation of a slot index, alive or not.
    #[inline]
    pub fn generation(&self, index: u32) -> Option<u32> {
        self.slots.get(index as usize).map(|slot| slot.generation)
    }

    /// Maps an index plus observed generation back to a live entity, if any.
    pub fn resolve(&self, index: u32, generation: u32) -> Option<Entity> {
        let slot = self.slots.get(index as usize)?;
        if slot.alive && slot.generation == generation {
            Some(Entity::from_raw(index, generation))
        } else {
            None
        }
    }

    /// Number of live entities.
    #[inline]
    pub fn len(&self) -> usize {
        self.alive_count
    }

    /// `true` when no entity is alive.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.alive_count == 0
    }

    /// Highest allocated slot index (including dead slots).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Iterates every live entity in slot order.
    pub fn iter(&self) -> AliveIter<'_> {
        AliveIter {
            slots: &self.slots,
            index: 0,
        }
    }
}

/// Iterator over live entities produced by [`Entities::iter`].
#[derive(Debug, Clone)]
pub struct AliveIter<'e> {
    slots: &'e [Slot],
    index: usize,
}

impl Iterator for AliveIter<'_> {
    type Item = Entity;

    fn next(&mut self) -> Option<Entity> {
        while let Some(slot) = self.slots.get(self.index) {
            let index = self.index as u32;
            self.index += 1;
            if slot.alive {
                return Some(Entity::from_raw(index, slot.generation));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.slots.len() - self.index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recycled_slots_are_generation_checked() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        let b = entities.alloc();
        assert!(entities.is_alive(a) && entities.is_alive(b));
        assert!(entities.dealloc(a));
        assert!(!entities.dealloc(a));
        let c = entities.alloc();
        assert_eq!(c.index(), a.index());
        assert_eq!(c.generation(), a.generation() + 1);
        assert!(!entities.is_alive(a), "stale handle must be dead");
        assert_eq!(entities.len(), 2);
        assert_eq!(entities.iter().collect::<Vec<_>>(), vec![c, b]);
    }

    #[test]
    fn dangling_never_resolves() {
        let entities = Entities::new();
        assert!(DANGLING.is_dangling());
        assert!(!entities.is_alive(DANGLING));
    }
}
