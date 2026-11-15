//! 一个世界资源，告诉驱动器是否仍有系统欠着工作。
//!
//! 系统在某个标签下将自己标记为忙碌，落定时清除它。
//! 当没有标签剩余时驱动器停止，因此它无需了解
//! 智能体、对话或任何其他领域类型。

use std::collections::{BTreeSet, HashMap};

use xihe_core::World;

/// 仍有未完成工作的系统标签。
#[derive(Clone, Debug, Default)]
pub struct Work {
    busy: BTreeSet<String>,
    owners: HashMap<String, BTreeSet<String>>,
}

impl Work {
    /// 一个空工作量。
    pub fn new() -> Self {
        Self::default()
    }

    /// 将 `label` 标记为仍在工作，由 `label` 自身拥有。
    pub fn begin(&mut self, label: impl Into<String>) {
        let label = label.into();
        self.begin_owned(label.clone(), label);
    }

    /// 将 `label` 标记为已落定，丢弃每个所有者对它的认领。
    pub fn end(&mut self, label: &str) {
        self.busy.remove(label);
        self.owners.remove(label);
    }

    /// 一次调用将 `label` 设为忙碌或空闲，由 `label` 自身拥有。
    pub fn set(&mut self, label: impl Into<String>, busy: bool) {
        let label = label.into();
        self.set_owned(label.clone(), busy, label);
    }

    /// 代表 `owner`（负责它的系统）将 `label` 标记为仍在工作。
    /// 一个标签保持忙碌，直到每个所有者都已将它落定。
    pub fn begin_owned(&mut self, label: impl Into<String>, owner: impl Into<String>) {
        let label = label.into();
        self.busy.insert(label.clone());
        self.owners.entry(label).or_default().insert(owner.into());
    }

    /// 清除 `owner` 对 `label` 的认领；一旦没有所有者欠它，该标签即落定。
    pub fn end_owned(&mut self, label: impl Into<String>, owner: impl Into<String>) {
        let label = label.into();
        if let Some(owners) = self.owners.get_mut(&label) {
            owners.remove(&owner.into());
            if owners.is_empty() {
                self.owners.remove(&label);
                self.busy.remove(&label);
            }
        }
    }

    /// 一次调用代表 `owner` 将 `label` 设为忙碌或空闲。
    pub fn set_owned(&mut self, label: impl Into<String>, busy: bool, owner: impl Into<String>) {
        if busy {
            self.begin_owned(label, owner);
        } else {
            self.end_owned(label, owner);
        }
    }

    /// 释放 `owner` 仍欠着的每个标签。用于某系统失败后被隔离时：
    /// 它再也无法落定自己的工作，因此那个工作绝不能
    /// 让驱动器永远保持唤醒。
    pub fn release_owner(&mut self, owner: &str) {
        let labels: Vec<String> = self
            .owners
            .iter()
            .filter(|(_, owners)| owners.contains(owner))
            .map(|(label, _)| label.clone())
            .collect();
        for label in labels {
            self.end_owned(label, owner);
        }
    }

    /// 是否仍有系统欠着工作。
    pub fn is_busy(&self) -> bool {
        !self.busy.is_empty()
    }

    /// 当前忙碌的标签，按排序顺序。
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.busy.iter().map(String::as_str)
    }

    /// 从世界中读取工作量，缺失时插入一个空的。
    pub fn ensure(world: &mut World) -> &mut Self {
        if !world.has_resource::<Self>() {
            world.insert_resource(Self::new());
        }
        world.resource_mut::<Self>().expect("just inserted")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_track_busy_systems() {
        let mut work = Work::new();
        assert!(!work.is_busy());
        work.begin("react");
        work.begin("ingest");
        assert!(work.is_busy());
        work.end("react");
        assert!(work.is_busy());
        work.end("ingest");
        assert!(!work.is_busy());
        work.set("react", true);
        work.set("react", false);
        assert!(!work.is_busy());
    }

    #[test]
    fn ensure_inserts_once() {
        let mut world = World::new();
        Work::ensure(&mut world).begin("a");
        Work::ensure(&mut world).begin("b");
        assert_eq!(world.resource::<Work>().unwrap().labels().count(), 2);
    }

    #[test]
    fn a_label_settles_only_when_every_owner_releases_it() {
        let mut work = Work::new();
        work.begin_owned("agents", "react");
        work.begin_owned("agents", "fan-out");
        work.end_owned("agents", "react");
        assert!(work.is_busy(), "fan-out still owes the label");
        work.end_owned("agents", "fan-out");
        assert!(!work.is_busy());
    }

    #[test]
    fn release_owner_frees_only_that_owners_labels() {
        let mut work = Work::new();
        work.begin_owned("ingest", "ingest");
        work.begin_owned("agents", "react");
        work.release_owner("ingest");
        assert!(work.is_busy());
        assert_eq!(work.labels().collect::<Vec<_>>(), vec!["agents"]);
        work.release_owner("react");
        assert!(!work.is_busy());
    }
}
