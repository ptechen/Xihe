//! Which parts of a world a system is allowed to touch.

use std::any::TypeId;
use std::collections::HashSet;

/// Declared world access of a [`System`](crate::System).
///
/// The scheduler never parses system code; it trusts this declaration to decide
/// which systems may share a parallel stage. Declaring too much is safe (the
/// system simply runs alone), declaring too little is a user error: a system that
/// writes through the world handle must say so, otherwise two systems can write
/// the same component at the same time.
///
/// ```
/// use xihe_schedule::Access;
///
/// #[derive(Default)]
/// struct Position;
/// #[derive(Default)]
/// struct Velocity;
///
/// let physics = Access::read::<Velocity>().and_write::<Position>();
/// let render = Access::read::<Position>();
/// assert!(physics.conflicts(&render));
/// assert!(!render.conflicts(&Access::read::<Velocity>()));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Access {
    reads: HashSet<TypeId>,
    writes: HashSet<TypeId>,
    all_read: bool,
    all_write: bool,
}

impl Access {
    /// Touches nothing, so it never conflicts.
    pub fn empty() -> Self {
        Self {
            reads: HashSet::new(),
            writes: HashSet::new(),
            all_read: false,
            all_write: false,
        }
    }

    /// Reads exactly one component or resource type.
    pub fn read<T: 'static>() -> Self {
        Self::empty().and_read::<T>()
    }

    /// Writes exactly one component or resource type.
    pub fn write<T: 'static>() -> Self {
        Self::empty().and_write::<T>()
    }

    /// Reads an unknown part of the world: compatible with other readers only.
    pub fn read_only() -> Self {
        Self {
            all_read: true,
            ..Self::empty()
        }
    }

    /// May write anything. This is the default for systems that do not declare
    /// access, so forgetting to declare costs parallelism but never correctness.
    pub fn exclusive() -> Self {
        Self {
            all_write: true,
            ..Self::empty()
        }
    }

    /// Adds a read of `T`.
    pub fn and_read<T: 'static>(mut self) -> Self {
        self.reads.insert(TypeId::of::<T>());
        self
    }

    /// Adds a write of `T`.
    pub fn and_write<T: 'static>(mut self) -> Self {
        self.writes.insert(TypeId::of::<T>());
        self
    }

    /// Adds `other` to this declaration.
    pub fn merged(mut self, other: &Self) -> Self {
        self.reads.extend(other.reads.iter());
        self.writes.extend(other.writes.iter());
        self.all_read |= other.all_read;
        self.all_write |= other.all_write;
        self
    }

    /// `true` when nothing is accessed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.all_read && !self.all_write && self.reads.is_empty() && self.writes.is_empty()
    }

    /// `true` when the system may write anything.
    #[inline]
    pub fn is_exclusive(&self) -> bool {
        self.all_write
    }

    /// Declared read set.
    #[inline]
    pub fn reads(&self) -> &HashSet<TypeId> {
        &self.reads
    }

    /// Declared write set.
    #[inline]
    pub fn writes(&self) -> &HashSet<TypeId> {
        &self.writes
    }

    /// `true` when the two declarations cannot run at the same time.
    pub fn conflicts(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        if self.all_write || other.all_write {
            return true;
        }
        if self.all_read || other.all_read {
            // An unspecified reader conflicts with any writer.
            return !self.writes.is_empty() || !other.writes.is_empty();
        }
        !self.writes.is_disjoint(&other.writes)
            || !self.writes.is_disjoint(&other.reads)
            || !self.reads.is_disjoint(&other.writes)
    }

    /// `true` when the two declarations may share a parallel stage.
    #[inline]
    pub fn compatible(&self, other: &Self) -> bool {
        !self.conflicts(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct A;
    struct B;
    struct C;

    #[test]
    fn empty_never_conflicts() {
        assert!(!Access::empty().conflicts(&Access::exclusive()));
        assert!(!Access::empty().conflicts(&Access::read::<A>()));
    }

    #[test]
    fn exclusive_conflicts_with_everything_used() {
        assert!(Access::exclusive().conflicts(&Access::read::<A>()));
        assert!(Access::exclusive().conflicts(&Access::write::<A>()));
        assert!(Access::read_only().conflicts(&Access::exclusive()));
    }

    #[test]
    fn read_write_pairs() {
        assert!(Access::write::<A>().conflicts(&Access::write::<A>()));
        assert!(Access::write::<A>().conflicts(&Access::read::<A>()));
        assert!(!Access::write::<A>().conflicts(&Access::read::<B>()));
        assert!(!Access::read::<A>().conflicts(&Access::read::<B>()));
        assert!(!Access::read::<A>().conflicts(&Access::read::<A>()));
    }

    #[test]
    fn unspecified_reader_blocks_writers() {
        assert!(Access::read_only().conflicts(&Access::write::<A>()));
        assert!(!Access::read_only().conflicts(&Access::read_only()));
        assert!(!Access::read_only().conflicts(&Access::read::<A>()));
    }

    #[test]
    fn merging_accumulates() {
        let merged = Access::read::<A>().merged(&Access::write::<B>());
        assert!(merged.conflicts(&Access::read::<B>()));
        assert!(!merged.conflicts(&Access::read::<C>()));
        assert_eq!(merged.reads().len(), 1);
        assert_eq!(merged.writes().len(), 1);
        assert!(!Access::read::<C>().merged(&Access::empty()).is_empty());
        let _ = TypeId::of::<C>();
    }
}
