//! 世界：实体索引、组件存储与资源。

use crate::ecs_core::component::{Component, Resource};
use crate::ecs_core::entity::{Entities, Entity};
use crate::ecs_core::query::{Fetch, Query};
use crate::ecs_core::resources::Resources;
use crate::ecs_core::storage::Storages;

/// 已分配实体槽位的硬上限（`u32` 索引空间减去哨兵值）。
pub const MAX_ENTITIES: usize = u32::MAX as usize - 1;

/// 一起插入的组件，通常在 spawn 时。
///
/// 已为至多六个组件的元组实现，因此
/// `world.spawn_with((profile, conversation))` 无需额外的 trait 导入。
pub trait Bundle: Send {
    /// 将每个成员插入 `entity`。
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

/// 交给 [`World::edit_with`] 闭包的只读视图，使得一个组件类型可被修改，
/// 而其他所有组件与资源仍保持可读。
pub struct ReadContext<'w> {
    storages: &'w Storages,
    resources: &'w Resources,
}

impl<'w> ReadContext<'w> {
    /// `entity` 的共享组件。
    #[inline]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&'w T> {
        self.storages.get::<T>(entity)
    }

    /// `entity` 持有 `T` 时为 `true`。
    #[inline]
    pub fn contains<T: Component>(&self, entity: Entity) -> bool {
        self.storages.contains::<T>(entity)
    }

    /// 共享资源。
    #[inline]
    pub fn resource<R: Resource>(&self) -> Option<&'w R> {
        self.resources.get::<R>()
    }

    /// 持有 `T` 的实体数量。
    #[inline]
    pub fn count<T: Component>(&self) -> usize {
        self.storages.count_of::<T>()
    }
}

/// 拥有一次模拟的每个实体、组件与资源。
///
/// `World` 是纯数据：无锁、无 `unsafe`、无异步。并发是驱动器
/// （`xihe::ecs_schedule` 与 `xihe::agent_runtime`）的属性，它通过 `RwLock` 共享世界，
/// 且只向系统交出共享或独占访问。
#[derive(Debug, Default)]
pub struct World {
    entities: Entities,
    storages: Storages,
    resources: Resources,
}

impl World {
    /// 创建一个空世界。
    pub fn new() -> Self {
        Self::default()
    }

    // --- 实体 -----------------------------------------------------------

    /// 分配一个裸实体。
    pub fn spawn(&mut self) -> Entity {
        self.entities.alloc()
    }

    /// 分配一个实体并用 [`Bundle`] 填充它。
    pub fn spawn_with<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.spawn();
        bundle.insert_into(self, entity);
        entity
    }

    /// 分配 `count` 个实体。
    pub fn spawn_many(&mut self, count: usize) -> Vec<Entity> {
        self.entities.alloc_many(count)
    }

    /// 释放一个实体并丢弃其组件。它原本存活时为 `true`。
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        self.storages.remove_entity(entity);
        self.entities.dealloc(entity)
    }

    /// 释放每个存活实体，保留资源。返回被丢弃的数量。
    pub fn despawn_all(&mut self) -> usize {
        let dead: Vec<Entity> = self.entities.iter().collect();
        let count = dead.len();
        for entity in dead {
            self.despawn(entity);
        }
        count
    }

    /// 句柄仍指向存活实体时为 `true`。
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// 存活实体数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// 没有存活实体时为 `true`。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// 按槽位顺序遍历存活实体。
    pub fn entities(&self) -> impl Iterator<Item = Entity> + Clone + '_ {
        self.entities.iter()
    }

    /// 借用的实体索引。
    #[inline]
    pub fn entity_index(&self) -> &Entities {
        &self.entities
    }

    // --- 组件 ---------------------------------------------------------

    /// 插入 `component`，返回它替换掉的值。
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) -> Option<T> {
        self.storages.insert(entity, component)
    }

    /// 将整个 [`Bundle`] 插入一个已有实体。
    pub fn insert_bundle<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        bundle.insert_into(self, entity);
    }

    /// 共享组件引用。
    #[inline]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.storages.get::<T>(entity)
    }

    /// 独占组件引用。
    #[inline]
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        self.storages.get_mut::<T>(entity)
    }

    /// `entity` 存活且持有 `T` 时为 `true`。
    #[inline]
    pub fn contains<T: Component>(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity) && self.storages.contains::<T>(entity)
    }

    /// 移除并返回 `entity` 的组件。
    #[inline]
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.storages.remove::<T>(entity)
    }

    /// 持有 `T` 的实体数量。
    #[inline]
    pub fn count<T: Component>(&self) -> usize {
        self.storages.count_of::<T>()
    }

    // --- 查询 ------------------------------------------------------------

    /// 运行只读查询，例如 `world.query::<(&Profile, &Conversation)>()`。
    pub fn query<'w, Q: Fetch<'w>>(&'w self) -> Query<'w, Q> {
        Query::new(&self.storages, &self.entities)
    }

    /// 唯一持有 `T` 的实体，没有实体持有时为 `None`。
    ///
    /// 当两个实体持有它时 panic：单例组件以实体句柄命名一个进程级
    /// 资源（一条流水线、一个信箱），出现两个属于编程错误，
    /// 而非运行时的选择。
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

    /// 对每个 `T` 的可变访问，一次一个组件类型。
    ///
    /// 这是 `&mut T` 查询项的安全对应物：`T` 的存储在 `f` 执行期间
    /// 离开映射，因此闭包拿到 `&mut T`，而世界其余部分保持可读——
    /// 参见 [`World::edit_with`]。
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

    /// [`World::edit`] 外加一个覆盖其他组件的 [`ReadContext`]。
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

    // --- 资源 ----------------------------------------------------------

    /// 插入一个资源，返回先前的值。
    pub fn insert_resource<R: Resource>(&mut self, value: R) -> Option<R> {
        self.resources.insert(value)
    }

    /// 确保 `R::default()` 存在并返回它。
    pub fn init_resource<R: Resource + Default>(&mut self) -> &mut R {
        self.resources.init::<R>()
    }

    /// 共享资源引用。
    #[inline]
    pub fn resource<R: Resource>(&self) -> Option<&R> {
        self.resources.get::<R>()
    }

    /// 独占资源引用。
    #[inline]
    pub fn resource_mut<R: Resource>(&mut self) -> Option<&mut R> {
        self.resources.get_mut::<R>()
    }

    /// 移除一个资源。
    #[inline]
    pub fn take_resource<R: Resource>(&mut self) -> Option<R> {
        self.resources.remove::<R>()
    }

    /// 资源存在时为 `true`。
    #[inline]
    pub fn has_resource<R: Resource>(&self) -> bool {
        self.resources.contains::<R>()
    }

    // --- 自省 ------------------------------------------------------

    /// `(组件类型名, 实体数量)` 对，按名称排序。
    pub fn component_counts(&self) -> Vec<(&'static str, usize)> {
        self.storages.counts()
    }

    /// 资源类型名，已排序。
    pub fn resource_names(&self) -> Vec<&'static str> {
        self.resources.names()
    }

    /// 观察器层使用的聚合计数器。
    #[inline]
    pub fn stats(&self) -> WorldStats {
        WorldStats {
            entities: self.entities.len(),
            component_types: self.storages.len(),
            components: self.storages.total_components(),
            resources: self.resources.len(),
        }
    }

    /// 丢弃每个实体、组件与资源。
    pub fn clear(&mut self) {
        self.entities = Entities::default();
        self.storages.clear();
        self.resources.clear();
    }
}

/// 描述世界规模的计数器。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct WorldStats {
    /// 存活实体。
    pub entities: usize,
    /// 使用中的不同组件类型。
    pub component_types: usize,
    /// 组件实例总数。
    pub components: usize,
    /// 类型映射中的资源。
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
        let panic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| world.single::<Label>()));
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
        assert_eq!(
            world.get::<Counter>(e).unwrap().value,
            3,
            "last wins per type"
        );
        assert_eq!(world.get::<u8>(e), Some(&7));
    }
}
