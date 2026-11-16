//! 附着在世界而非实体上的全局单例状态。

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

use crate::ecs_core::component::Resource;

/// [`Resource`] 的类型映射，外加报告所需的具体类型名。
#[derive(Default)]
pub struct Resources {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    names: HashMap<TypeId, &'static str>,
}

impl Resources {
    /// 创建一个空资源集。
    pub fn new() -> Self {
        Self::default()
    }

    /// 已插入资源的数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// 没有插入资源时为 `true`。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// 存在 `R` 时为 `true`。
    #[inline]
    pub fn contains<R: Resource>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<R>())
    }

    /// 对 `R` 的共享引用。
    #[inline]
    pub fn get<R: Resource>(&self) -> Option<&R> {
        self.map
            .get(&TypeId::of::<R>())
            .and_then(|value| value.downcast_ref::<R>())
    }

    /// 对 `R` 的独占引用。
    #[inline]
    pub fn get_mut<R: Resource>(&mut self) -> Option<&mut R> {
        self.map
            .get_mut(&TypeId::of::<R>())
            .and_then(|value| value.downcast_mut::<R>())
    }

    /// 插入 `value`，返回先前的值。
    #[inline]
    pub fn insert<R: Resource>(&mut self, value: R) -> Option<R> {
        let type_id = TypeId::of::<R>();
        self.names.insert(type_id, std::any::type_name::<R>());
        self.map
            .insert(type_id, Box::new(value))
            .and_then(|previous| previous.downcast::<R>().ok())
            .map(|boxed| *boxed)
    }

    /// 移除 `R`。
    #[inline]
    pub fn remove<R: Resource>(&mut self) -> Option<R> {
        let type_id = TypeId::of::<R>();
        self.names.remove(&type_id);
        self.map
            .remove(&type_id)
            .and_then(|value| value.downcast::<R>().ok())
            .map(|boxed| *boxed)
    }

    /// 若 `R::default()` 尚不存在则插入，并返回其可变引用。
    pub fn init<R: Resource + Default>(&mut self) -> &mut R {
        let type_id = TypeId::of::<R>();
        self.names.insert(type_id, std::any::type_name::<R>());
        self.map
            .entry(type_id)
            .or_insert_with(|| Box::new(R::default()));
        self.get_mut::<R>().expect("entry was just ensured")
    }

    /// 对 `R` 的共享引用，缺失时插入 `R::default()`。
    pub fn get_or_init<R: Resource + Default>(&mut self) -> &mut R {
        self.init::<R>()
    }

    /// 已插入资源的类型名，排序以得到确定性报告。
    pub fn names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.names.values().copied().collect();
        names.sort_unstable();
        names
    }

    /// 丢弃所有资源。
    pub fn clear(&mut self) {
        self.map.clear();
        self.names.clear();
    }
}

impl fmt::Debug for Resources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Resources")
            .field("count", &self.map.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Config {
        retries: u32,
    }

    impl Default for Config {
        fn default() -> Self {
            Self { retries: 3 }
        }
    }

    #[test]
    fn insert_get_remove() {
        let mut resources = Resources::new();
        assert!(resources.insert(Config { retries: 1 }).is_none());
        assert_eq!(resources.get::<Config>().unwrap().retries, 1);
        resources.get_mut::<Config>().unwrap().retries = 5;
        assert_eq!(
            resources.init::<Config>().retries,
            5,
            "init keeps the value"
        );
        assert_eq!(resources.remove::<Config>().unwrap().retries, 5);
        assert!(resources.is_empty());
        assert!(resources.names().is_empty());
    }

    #[test]
    fn init_creates_default_once() {
        let mut resources = Resources::new();
        resources.init::<Config>().retries = 9;
        assert_eq!(resources.get_or_init::<Config>().retries, 9);
        assert_eq!(
            resources.names(),
            vec!["xihe::ecs_core::resources::tests::Config"]
        );
    }
}
