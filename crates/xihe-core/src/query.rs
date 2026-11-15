//! 组件存储之上类型化、借用安全的查询项。

use std::marker::PhantomData;

use crate::component::Component;
use crate::entity::{Entities, Entity};
use crate::storage::{EntityCursor, Storages};

/// 查询的一个项：如何驱动游标，以及如何提取该项。
///
/// 已为 [`Entity`]、`&T`、`Option<&T>` 以及至多四个项的元组提供实现。
/// `&mut T` 刻意*不*作为项——独占迭代经由
/// [`World::edit`](crate::World::edit) 进行，由借用检查器主导，
/// 而非依赖裸指针。
pub trait Fetch<'w>: Copy + Sized {
    /// 该项想要遍历的实体。元组的第一个必需项决定游标，
    /// 因此查询稀有组件不会扫描整个世界。
    fn cursor(storages: &'w Storages, entities: &'w Entities) -> EntityCursor<'w> {
        let _ = storages;
        Box::new(entities.iter())
    }

    /// 提取 `entity` 对应的项，必需项缺失时返回 `None`。
    fn fetch(storages: &'w Storages, entity: Entity) -> Option<Self>;
}

impl<'w> Fetch<'w> for Entity {
    #[inline]
    fn fetch(_storages: &'w Storages, entity: Entity) -> Option<Self> {
        Some(entity)
    }
}

impl<'w, T: Component> Fetch<'w> for &'w T {
    fn cursor(storages: &'w Storages, _entities: &'w Entities) -> EntityCursor<'w> {
        storages.cursor_of::<T>()
    }

    #[inline]
    fn fetch(storages: &'w Storages, entity: Entity) -> Option<Self> {
        storages.get::<T>(entity)
    }
}

impl<'w, T: Component> Fetch<'w> for Option<&'w T> {
    #[inline]
    fn fetch(storages: &'w Storages, entity: Entity) -> Option<Self> {
        Some(storages.get::<T>(entity))
    }
}

macro_rules! impl_fetch_tuple {
    ($(#[$meta:meta])* $($name:ident),+) => {
        $(#[$meta])*
        impl<'w, $($name: Fetch<'w>),+> Fetch<'w> for ($($name),+) {
            fn cursor(storages: &'w Storages, entities: &'w Entities) -> EntityCursor<'w> {
                impl_fetch_tuple!(@first_cursor storages, entities, $($name),+)
            }

            #[inline]
            fn fetch(storages: &'w Storages, entity: Entity) -> Option<Self> {
                Some(($($name::fetch(storages, entity)?),+))
            }
        }
    };
    (@first_cursor $s:expr, $e:expr, $head:ident $(, $rest:ident)*) => {
        $head::cursor($s, $e)
    };
}

impl_fetch_tuple!(/// 两个项，例如 `(&Profile, &Conversation)`
    A, B);
impl_fetch_tuple!(/// 三个项，例如 `(Entity, &Conversation, Option<&Budget>)`
    A, B, C);
impl_fetch_tuple!(/// 四个项
    A, B, C, D);

/// 遍历匹配查询的实体的借用迭代器。
pub struct Query<'w, Q: Fetch<'w>> {
    storages: &'w Storages,
    cursor: EntityCursor<'w>,
    marker: PhantomData<Q>,
}

impl<'w, Q: Fetch<'w>> Query<'w, Q> {
    /// 在 `storages` 上构建查询，由第一个必需项驱动。
    pub(crate) fn new(storages: &'w Storages, entities: &'w Entities) -> Self {
        Self {
            storages,
            cursor: Q::cursor(storages, entities),
            marker: PhantomData,
        }
    }

    /// 用 `pred` 丢弃匹配项，同时保持迭代的借用安全。
    pub fn filter<F>(self, pred: F) -> FilteredQuery<'w, Q, F>
    where
        F: FnMut(Q) -> bool + Send + 'w,
    {
        FilteredQuery {
            inner: self,
            pred,
        }
    }

    /// 第一个匹配项（若有）。
    pub fn first(self) -> Option<Q> {
        self.into_iter().next()
    }

    /// 将匹配项收集到向量中。
    pub fn collect_vec(self) -> Vec<Q> {
        self.into_iter().collect()
    }

    /// 对每个匹配项运行 `f`。
    pub fn for_each<F>(self, f: F)
    where
        F: FnMut(Q),
    {
        Iterator::for_each(self, f)
    }
}

impl<'w, Q: Fetch<'w>> Iterator for Query<'w, Q> {
    type Item = Q;

    fn next(&mut self) -> Option<Q> {
        for entity in self.cursor.by_ref() {
            if let Some(item) = Q::fetch(self.storages, entity) {
                return Some(item);
            }
        }
        None
    }
}

impl<'w, Q: Fetch<'w>> std::fmt::Debug for Query<'w, Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Query")
            .field("terms", &std::any::type_name::<Q>())
            .finish()
    }
}

/// 通过 [`Query::filter`] 附加了谓词的 [`Query`]。
pub struct FilteredQuery<'w, Q: Fetch<'w>, F: FnMut(Q) -> bool> {
    inner: Query<'w, Q>,
    pred: F,
}

impl<'w, Q: Fetch<'w>, F: FnMut(Q) -> bool> Iterator for FilteredQuery<'w, Q, F> {
    type Item = Q;

    fn next(&mut self) -> Option<Q> {
        // 拆分借用：迭代器需要 `inner`，而谓词读取 `pred`。
        let Self { inner, pred } = self;
        inner.by_ref().find(|item| pred(*item))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Alive(bool);

    #[derive(Debug)]
    struct Score(u32);

    #[test]
    fn optional_terms_do_not_drive_the_cursor() {
        let mut world = crate::World::new();
        let a = world.spawn_with((Alive(true), Score(3)));
        let _b = world.spawn_with((Score(1),));

        let scores: Vec<u32> = world
            .query::<(&Alive, &Score)>()
            .map(|(alive, score)| {
                let _ = alive;
                score.0
            })
            .collect();
        assert_eq!(scores, vec![3]);

        let with_entity: Vec<(Entity, bool)> = world
            .query::<(Entity, &Alive)>()
            .map(|(e, alive)| (e, alive.0))
            .collect();
        assert_eq!(with_entity, vec![(a, true)]);
    }

    #[test]
    fn filter_and_count() {
        let mut world = crate::World::new();
        for i in 0..5u32 {
            world.spawn_with((Score(i),));
        }
        let big = world
            .query::<&Score>()
            .filter(|score| score.0 >= 2)
            .count();
        assert_eq!(big, 3);
        assert_eq!(world.query::<&Score>().count(), 5);
        assert_eq!(world.query::<Option<&Alive>>().count(), 5);
    }
}
