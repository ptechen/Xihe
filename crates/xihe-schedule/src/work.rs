//! A world resource that tells the driver whether any system still owes work.
//!
//! Systems mark themselves busy under a label and clear it when they settle.
//! The driver stops when no label remains, so it does not have to know about
//! agents, conversations, or any other domain type.

use std::collections::BTreeSet;

use xihe_core::World;

/// Labels of systems that still have unfinished work.
#[derive(Clone, Debug, Default)]
pub struct Work {
    busy: BTreeSet<String>,
}

impl Work {
    /// An empty workload.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `label` as still working.
    pub fn begin(&mut self, label: impl Into<String>) {
        self.busy.insert(label.into());
    }

    /// Mark `label` as settled.
    pub fn end(&mut self, label: &str) {
        self.busy.remove(label);
    }

    /// Set `label` busy or idle in one call.
    pub fn set(&mut self, label: impl Into<String>, busy: bool) {
        if busy {
            self.begin(label);
        } else {
            self.end(&label.into());
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
}
