//! 组件存储：每个组件类型一个 `Send` 安全的稀疏集。

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

use crate::component::Component;
use crate::entity::{Entity, Entities};

/// 遍历某存储中实体的迭代器，经过擦除，使查询在*项*元组上
/// 保持单态，而非在每个组件类型上单态。
pub type EntityCursor<'w> = Box<dyn Iterator<Item = Entity> + Send + 'w>;

/// 类型擦除的组件存储。
pub trait Storage: Any + Send + Sync {
    /// 此存储持有的组件类型。
    fn component_type(&self) -> TypeId;

    /// 组件类型名，用于报告与仪表盘。
    fn component_name(&self) -> &'static str;

    /// 持有该组件的实体数量。
    fn len(&self) -> usize;

    /// 没有实体持有该组件时为 `true`。
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `entity` 持有该组件时为 `true`。
    fn contains(&self, entity: Entity) -> bool;

    /// 移除 `entity` 的组件，并报告它原本是否存在。
    fn remove(&mut self, entity: Entity) -> bool;

    /// 丢弃每个已存储的组件。
    fn clear_storage(&mut self);

    /// 向下转换辅助方法。
    fn as_any(&self) -> &dyn Any;

    /// 可变向下转换辅助方法。
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// 已存储实体的快照（dense 顺序）。
    fn snapshot_entities(&self) -> Vec<Entity>;

    /// 遍历已存储实体的借用游标。
    fn cursor(&self) -> EntityCursor<'_>;
}

/// 稠密数组组件存储：`dense` 存放 `(Entity, T)` 对，`sparse`
/// 将实体槽位索引映射到它在 `dense` 中的位置。
///
/// 插入/移除借助交换回填为 O(1)，迭代对缓存友好，
/// 且按实体索引查找从不遍历整个存储。
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
    /// 创建一个空存储。
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::new(),
        }
    }

    /// 已存储组件的数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// 为空时为 `true`。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// `entity` 的 dense 位置（若已存储）。
    #[inline]
    fn position(&self, entity: Entity) -> Option<usize> {
        self.sparse
            .get(entity.index() as usize)
            .and_then(|slot| *slot)
            .map(|pos| pos as usize)
    }

    /// `entity` 在此处有组件时为 `true`。
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

    /// 插入 `value`，返回先前为 `entity` 存储的组件。
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

    /// 对 `entity` 组件的共享访问。
    #[inline]
    pub fn get(&self, entity: Entity) -> Option<&T> {
        self.position(entity).map(|pos| &self.dense[pos].1)
    }

    /// 对 `entity` 组件的独占访问。
    #[inline]
    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        self.position(entity)
            .map(|pos| &mut self.dense[pos].1)
    }

    /// `entity` 的组件及其 dense 位置。
    #[inline]
    pub fn find(&self, entity: Entity) -> Option<(usize, &T)> {
        self.position(entity).map(|pos| (pos, &self.dense[pos].1))
    }

    /// 移除并返回 `entity` 的组件。
    pub fn remove(&mut self, entity: Entity) -> Option<T> {
        let pos = self.position(entity)?;
        let last = self.dense.len() - 1;
        let moved = self.dense.swap_remove(pos);
        if pos != last {
            let moved_entity = self.dense[pos].0;
            self.sparse[moved_entity.index() as usize] = Some(pos as u32);
        }
        // `swap_remove` 返回了被请求的对；被挤出的对已移入 `pos`
        // 且现在可达，因此只清除被移除的槽位。
        self.sparse[entity.index() as usize] = None;
        Some(moved.1)
    }

    /// 按 dense 顺序遍历 `(Entity, &T)`。
    pub fn iter(&self) -> std::slice::Iter<'_, (Entity, T)> {
        self.dense.iter()
    }

    /// 按 dense 顺序遍历 `(Entity, &mut T)`。
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, (Entity, T)> {
        self.dense.iter_mut()
    }

    /// 借用 dense 位置 `index` 处的对。
    #[inline]
    pub fn slot_at_mut(&mut self, index: usize) -> Option<(Entity, &mut T)> {
        self.dense.get_mut(index).map(|(entity, value)| (*entity, value))
    }

    /// 按 dense 顺序的实体列表。
    pub fn entities(&self) -> impl Iterator<Item = Entity> + Clone + '_ {
        self.dense.iter().map(|(entity, _)| *entity)
    }

    /// 取得每个已存储组件的所有权，使存储变空。
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

/// 每个组件类型一个 [`SparseSet`] 的集合。
#[derive(Default)]
pub struct Storages {
    map: HashMap<TypeId, Box<dyn Storage>>,
}

impl Storages {
    /// 创建一个空集合。
    pub fn new() -> Self {
        Self::default()
    }

    /// 当前拥有存储的组件类型数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// 尚未插入任何组件类型时为 `true`。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// 存在的组件类型 id，排序以得到确定性报告。
    pub fn types(&self) -> Vec<TypeId> {
        let mut types: Vec<TypeId> = self.map.keys().copied().collect();
        types.sort();
        types
    }

    /// `T` 的具体存储，需要时创建。
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

    /// `T` 存在时的具体存储。
    pub fn storage_of<T: Component>(&self) -> Option<&SparseSet<T>> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|storage| storage.as_any().downcast_ref::<SparseSet<T>>())
    }

    /// `entity` 的共享组件。
    #[inline]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.storage_of::<T>().and_then(|set| set.get(entity))
    }

    /// `entity` 的独占组件。
    #[inline]
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        self.storage_of_mut::<T>().get_mut(entity)
    }

    /// `entity` 持有 `T` 时为 `true`。
    #[inline]
    pub fn contains<T: Component>(&self, entity: Entity) -> bool {
        self.storage_of::<T>().is_some_and(|set| set.contains(entity))
    }

    /// 插入 `value`，返回 `entity` 先前的组件。
    #[inline]
    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) -> Option<T> {
        self.storage_of_mut::<T>().insert(entity, value)
    }

    /// 移除 `entity` 的组件。
    #[inline]
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|storage| storage.as_any_mut().downcast_mut::<SparseSet<T>>())
            .and_then(|set| set.remove(entity))
    }

    /// 从每个存储中移除 `entity`，despawn 时使用。
    pub fn remove_entity(&mut self, entity: Entity) {
        for storage in self.map.values_mut() {
            storage.remove(entity);
        }
    }

    /// 持有 `T` 的实体数量。
    #[inline]
    pub fn count_of<T: Component>(&self) -> usize {
        self.storage_of::<T>().map_or(0, SparseSet::len)
    }

    /// 丢弃所有类型的全部组件。
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// 跨所有类型的组件实例总数。
    pub fn total_components(&self) -> usize {
        self.map.values().map(|storage| storage.len()).sum()
    }

    /// `(组件名, 实体数量)` 对，按名称排序。
    pub fn counts(&self) -> Vec<(&'static str, usize)> {
        let mut counts: Vec<(&'static str, usize)> = self
            .map
            .values()
            .map(|storage| (storage.component_name(), storage.len()))
            .collect();
        counts.sort_by(|left, right| left.0.cmp(right.0));
        counts
    }

    /// `T` 存储的实体列表。
    pub fn snapshot_of<T: Component>(&self) -> Vec<Entity> {
        self.storage_of::<T>()
            .map(|set| set.entities().collect())
            .unwrap_or_default()
    }

    /// 遍历 `T` 实体的游标，类型未知时为空。
    pub fn cursor_of<T: Component>(&self) -> EntityCursor<'_> {
        match self.storage_of::<T>() {
            Some(set) => Box::new(set.entities()),
            None => Box::new(std::iter::empty()),
        }
    }

    /// 遍历世界中每个存活实体的游标。
    pub fn cursor_of_all<'w>(&'w self, entities: &'w Entities) -> EntityCursor<'w> {
        Box::new(entities.iter())
    }

    /// 从映射中移除 `T` 的存储，将它与其余只读存储一起以可变方式交给 `f`，
    /// 然后再放回去——包括在 unwind 时。
    ///
    /// 这是 [`World::edit`](crate::World::edit) 的安全核心：把存储
    /// 取出映射，得到对 `T` 的一个独占视图，而其他每个组件类型保持可读，
    /// 这是一个存放 `&mut` 值的 `HashMap` 永远无法向借用检查器证明的。
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
