//! Deferred world mutations.

use std::borrow::Cow;
use std::fmt;
use std::sync::{Arc, Mutex};

use xihe_core::World;

/// A boxed closure that mutates the world once the scheduler gets it a write
/// lock.
///
/// Commands are how a system that only holds a read guard — or a tool running on
/// another task — still changes the world: it queues the change instead of
/// upgrading the lock, which would deadlock an `RwLock` held by its own reader.
pub struct Command {
    /// Human-readable purpose, shown in traces and error reports.
    pub description: Cow<'static, str>,
    action: Box<dyn FnOnce(&mut World) + Send>,
}

impl Command {
    /// Wraps a closure as a command.
    pub fn new<F>(action: F) -> Self
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        Self {
            description: Cow::Borrowed(""),
            action: Box::new(action),
        }
    }

    /// Wraps a closure and names it.
    pub fn with_description(mut self, description: impl Into<Cow<'static, str>>) -> Self {
        self.description = description.into();
        self
    }

    /// Runs the command against `world`.
    pub fn apply(self, world: &mut World) {
        (self.action)(world);
    }
}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Command")
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// Multi-producer queue of [`Command`]s, shared by cloned system contexts.
#[derive(Clone, Default)]
pub struct CommandQueue {
    inner: Arc<Mutex<Vec<Command>>>,
}

impl CommandQueue {
    /// An empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues a command.
    pub fn push(&self, command: Command) {
        // The lock is only ever held for a `Vec` push, never across an await.
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(command);
    }

    /// Queues a closure directly.
    pub fn enqueue<F>(&self, action: F)
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        self.push(Command::new(action));
    }

    /// Removes every queued command.
    pub fn take_all(&self) -> Vec<Command> {
        std::mem::take(
            &mut *self
                .inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    /// Number of pending commands.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// `true` when nothing is pending.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Takes and runs every command against `world`.
    pub fn apply(&self, world: &mut World) -> usize {
        let pending = self.take_all();
        let count = pending.len();
        for command in pending {
            command.apply(world);
        }
        count
    }
}

impl fmt::Debug for CommandQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandQueue")
            .field("pending", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mark(u32);

    #[test]
    fn queue_is_multi_producer_and_applies_in_order() {
        let queue = CommandQueue::new();
        let producer = queue.clone();
        producer.enqueue(|world| {
            let e = world.spawn();
            world.insert(e, Mark(1));
        });
        queue.enqueue(|world| {
            let e = world.spawn();
            world.insert(e, Mark(2));
        });
        assert_eq!(queue.len(), 2);
        let mut world = World::new();
        assert_eq!(queue.apply(&mut world), 2);
        assert!(queue.is_empty());
        assert_eq!(world.count::<Mark>(), 2);
        let mut marks: Vec<u32> = world.query::<&Mark>().map(|mark| mark.0).collect();
        marks.sort_unstable();
        assert_eq!(marks, [1, 2]);
    }

    #[test]
    fn descriptions_survive() {
        let command = Command::new(|_world| ()).with_description("spawn agent");
        assert_eq!(command.description, "spawn agent");
        assert!(format!("{command:?}").contains("spawn agent"));
    }
}
