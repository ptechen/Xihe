//! 系统被允许触碰世界的哪些部分。

use std::any::TypeId;
use std::collections::HashSet;

/// [`System`](crate::System) 声明的世界访问权限。
///
/// 调度器从不解析系统代码；它信任此声明来决定哪些系统可共享
/// 一个并行阶段。声明过多是安全的（系统只是单独运行），
/// 声明过少则是用户错误：通过世界句柄写入的系统必须说明这一点，
/// 否则两个系统可能同时写入同一个组件。
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
    /// 不触碰任何东西，因此永不冲突。
    pub fn empty() -> Self {
        Self {
            reads: HashSet::new(),
            writes: HashSet::new(),
            all_read: false,
            all_write: false,
        }
    }

    /// 恰好读取一个组件或资源类型。
    pub fn read<T: 'static>() -> Self {
        Self::empty().and_read::<T>()
    }

    /// 恰好写入一个组件或资源类型。
    pub fn write<T: 'static>() -> Self {
        Self::empty().and_write::<T>()
    }

    /// 读取世界的未知部分：仅与其他读者兼容。
    pub fn read_only() -> Self {
        Self {
            all_read: true,
            ..Self::empty()
        }
    }

    /// 可写入任何东西。这是未声明访问权限的系统的默认值，
    /// 因此忘记声明会损失并行度，但绝不会损失正确性。
    pub fn exclusive() -> Self {
        Self {
            all_write: true,
            ..Self::empty()
        }
    }

    /// 添加对 `T` 的读取。
    pub fn and_read<T: 'static>(mut self) -> Self {
        self.reads.insert(TypeId::of::<T>());
        self
    }

    /// 添加对 `T` 的写入。
    pub fn and_write<T: 'static>(mut self) -> Self {
        self.writes.insert(TypeId::of::<T>());
        self
    }

    /// 将 `other` 加入此声明。
    pub fn merged(mut self, other: &Self) -> Self {
        self.reads.extend(other.reads.iter());
        self.writes.extend(other.writes.iter());
        self.all_read |= other.all_read;
        self.all_write |= other.all_write;
        self
    }

    /// 没有访问任何东西时为 `true`。
    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.all_read && !self.all_write && self.reads.is_empty() && self.writes.is_empty()
    }

    /// 系统可写入任何东西时为 `true`。
    #[inline]
    pub fn is_exclusive(&self) -> bool {
        self.all_write
    }

    /// 声明的读集合。
    #[inline]
    pub fn reads(&self) -> &HashSet<TypeId> {
        &self.reads
    }

    /// 声明的写集合。
    #[inline]
    pub fn writes(&self) -> &HashSet<TypeId> {
        &self.writes
    }

    /// 两个声明不能同时运行时为 `true`。
    pub fn conflicts(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        if self.all_write || other.all_write {
            return true;
        }
        if self.all_read || other.all_read {
            // 未指明的读者与任何写者冲突。
            return !self.writes.is_empty() || !other.writes.is_empty();
        }
        !self.writes.is_disjoint(&other.writes)
            || !self.writes.is_disjoint(&other.reads)
            || !self.reads.is_disjoint(&other.writes)
    }

    /// 两个声明可共享一个并行阶段时为 `true`。
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
