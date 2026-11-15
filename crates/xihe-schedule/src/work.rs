//! A world resource that tells the driver whether any system still owes work.
//!
//! Systems mark themselves busy under a label and clear it when they settle.
//! The driver stops when no label remains, so it does not have to know about
//! agents, conversations, or any other domain type.

use std::collections::{BTreeSet, HashMap};

use xihe_core::World;

/// Labels of systems that still have unfinished work.
#[derive(Clone, Debug, Default)]
pub struct Work {
    busy: BTreeSet<String>,
    owners: HashMap<String, BTreeSet<String>>,
}

impl Work {
    /// An empty workload.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `label` as still working, owned by `label` itself.
    pub fn begin(&mut self, label: impl Into<String>) {
        let label = label.into();
        self.begin_owned(label.clone(), label);
    }

    /// Mark `label` as settled, dropping every owner's claim on it.
    pub fn end(&mut self, label: &str) {
        self.busy.remove(label);
        self.owners.remove(label);
    }

    /// Set `label` busy or idle in one call, owned by `label` itself.
    pub fn set(&mut self, label: impl Into<String>, busy: bool) {
        let label = label.into();
        self.set_owned(label.clone(), busy, label);
    }

    /// Mark `label` as still working on behalf of `owner`, the system responsible
    /// for it. A label stays busy until every owner has settled it.
    pub fn begin_owned(&mut self, label: impl Into<String>, owner: impl Into<String>) {
        let label = label.into();
        self.busy.insert(label.clone());
        self.owners.entry(label).or_default().insert(owner.into());
    }

    /// Clear `owner`'s claim on `label`; the label settles once no owner owes it.
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

    /// Set `label` busy or idle on behalf of `owner` in one call.
    pub fn set_owned(&mut self, label: impl Into<String>, busy: bool, owner: impl Into<String>) {
        if busy {
            self.begin_owned(label, owner);
        } else {
            self.end_owned(label, owner);
        }
    }

    /// Release every label `owner` still owes. Used when a system is isolated
    /// after failing: it can never settle its own work again, so that work must
    /// not keep the driver awake forever.
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

    /// Whether any system still owes work.
    pub fn is_busy(&self) -> bool {
        !self.busy.is_empty()
    }

    /// The labels currently busy, in sorted order.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.busy.iter().map(String::as_str)
    }

    /// Read the workload out of a world, inserting an empty one when absent.
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
