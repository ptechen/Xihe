//! 带世代校验的实体标识符与存活实体索引。

use std::fmt;

/// 指向世界槽位的稳定、带世代校验的句柄。
///
/// [`Entity`] 大小为 `u32`、可 `Copy` 且可哈希，因此可以廉价地存放在
/// 组件、消息载荷和工具参数中。回收槽位会递增
/// [`Entity::generation`]，使过期句柄查找失败，而不是别名到新实体。
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct Entity {
    index: u32,
    generation: u32,
}

/// 永远不会解析到存活实体的哨兵值。
pub const DANGLING: Entity = Entity {
    index: u32::MAX,
    generation: u32::MAX,
};

impl Entity {
    /// 从原始部分构建实体。仅供分配器以及持久化世界的
    /// （反）序列化使用。
    #[inline]
    pub const fn from_raw(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// 在 [`Entities`] 中的槽位索引。
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// 槽位被回收的次数加一。
    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    /// 对 [`DANGLING`] 为 `true`。
    #[inline]
    pub const fn is_dangling(self) -> bool {
        self.index == u32::MAX
    }

    /// `alive` 时返回 `self`，否则返回 `None`。便于用于 `Option` 链。
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

/// 支撑 [`World`](crate::World) 的槽位表。
#[derive(Debug, Default)]
pub struct Entities {
    slots: Vec<Slot>,
    free: Vec<u32>,
    alive_count: usize,
}

impl Entities {
    /// 创建一个空表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 分配一个实体，复用最近释放的槽位。
    ///
    /// # Panics
    /// 若需要超过 `u32::MAX - 1` 个槽位。
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

    /// 按槽位顺序分配 `count` 个实体。
    pub fn alloc_many(&mut self, count: usize) -> Vec<Entity> {
        (0..count).map(|_| self.alloc()).collect()
    }

    /// 释放 `entity`，若它已死亡或过期则返回 `false`。
    pub fn dealloc(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        self.slots[entity.index() as usize].alive = false;
        self.free.push(entity.index());
        self.alive_count -= 1;
        true
    }

    /// 句柄仍指向存活槽位时为 `true`。
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.slots
            .get(entity.index() as usize)
            .is_some_and(|slot| slot.alive && slot.generation == entity.generation())
    }

    /// 某槽位索引的当前世代，无论是否存活。
    #[inline]
    pub fn generation(&self, index: u32) -> Option<u32> {
        self.slots.get(index as usize).map(|slot| slot.generation)
    }

    /// 将索引加观察到的世代映射回存活实体（若有）。
    pub fn resolve(&self, index: u32, generation: u32) -> Option<Entity> {
        let slot = self.slots.get(index as usize)?;
        if slot.alive && slot.generation == generation {
            Some(Entity::from_raw(index, generation))
        } else {
            None
        }
    }

    /// 存活实体的数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.alive_count
    }

    /// 没有存活实体时为 `true`。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.alive_count == 0
    }

    /// 已分配的最高槽位索引（含死亡槽位）。
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// 按槽位顺序遍历每个存活实体。
    pub fn iter(&self) -> AliveIter<'_> {
        AliveIter {
            slots: &self.slots,
            index: 0,
        }
    }
}

/// 由 [`Entities::iter`] 产生的存活实体迭代器。
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
