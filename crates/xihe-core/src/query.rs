//! Typed, borrow-safe query terms over component storage.

use std::marker::PhantomData;

use crate::component::Component;
use crate::entity::{Entities, Entity};
use crate::storage::{EntityCursor, Storages};

/// One term of a query: how to drive the cursor and how to extract the term.
///
/// Implementations exist for [`Entity`], `&T`, `Option<&T>` and tuples of up to
/// four terms. `&mut T` is deliberately *not* a term — exclusive iteration goes
/// through [`World::edit`](crate::World::edit), which keeps the borrow checker in
/// charge instead of relying on raw pointers.
pub trait Fetch<'w>: Copy + Sized {
    /// Entities this term wants to iterate. The first required term of a tuple
    /// decides the cursor, so querying a rare component does not scan the world.
    fn cursor(storages: &'w Storages, entities: &'w Entities) -> EntityCursor<'w> {
        let _ = storages;
        Box::new(entities.iter())
    }

    /// Extracts the term for `entity`, or `None` when a required term is missing.
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

impl_fetch_tuple!(/// Two terms, e.g. `(&Profile, &Conversation)`
    A, B);
impl_fetch_tuple!(/// Three terms, e.g. `(Entity, &Conversation, Option<&Budget>)`
    A, B, C);
impl_fetch_tuple!(/// Four terms
    A, B, C, D);

/// Borrowed iterator over the entities matching a query.
pub struct Query<'w, Q: Fetch<'w>> {
    storages: &'w Storages,
    cursor: EntityCursor<'w>,
    marker: PhantomData<Q>,
}

impl<'w, Q: Fetch<'w>> Query<'w, Q> {
    /// Builds a query over `storages`, driven by the first required term.
    pub(crate) fn new(storages: &'w Storages, entities: &'w Entities) -> Self {
        Self {
            storages,
            cursor: Q::cursor(storages, entities),
            marker: PhantomData,
        }
    }

    /// Drops matching items using `pred`, keeping iteration borrow-safe.
    pub fn filter<F>(self, pred: F) -> FilteredQuery<'w, Q, F>
    where
        F: FnMut(Q) -> bool + Send + 'w,
    {
        FilteredQuery {
            inner: self,
            pred,
        }
    }

    /// First match, if any.
    pub fn first(self) -> Option<Q> {
        self.into_iter().next()
    }

    /// Collects matches into a vector.
    pub fn collect_vec(self) -> Vec<Q> {
        self.into_iter().collect()
    }

    /// Runs `f` for every match.
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

/// [`Query`] with a predicate attached by [`Query::filter`].
pub struct FilteredQuery<'w, Q: Fetch<'w>, F: FnMut(Q) -> bool> {
    inner: Query<'w, Q>,
    pred: F,
}

impl<'w, Q: Fetch<'w>, F: FnMut(Q) -> bool> Iterator for FilteredQuery<'w, Q, F> {
    type Item = Q;

    fn next(&mut self) -> Option<Q> {
        // Split borrow: the iterator needs `inner` while the predicate reads `pred`.
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
