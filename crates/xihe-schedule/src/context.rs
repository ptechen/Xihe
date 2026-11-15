//! Handles handed to systems: the shared world, the frame counter and stop flags.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use xihe_core::{Resource, World};

use crate::command::{Command, CommandQueue};

/// `Send` handle to the world, shared by every system in a schedule.
///
/// Systems ask for a read or a write guard themselves; the scheduler only decides
/// *who may hold which guard at the same time*, based on declared
/// [`Access`](crate::Access).
#[derive(Clone)]
pub struct WorldHandle(Arc<RwLock<World>>);

impl WorldHandle {
    /// Wraps an existing world.
    pub fn new(world: World) -> Self {
        Self(Arc::new(RwLock::new(world)))
    }

    /// Wraps an empty world.
    pub fn empty() -> Self {
        Self::new(World::new())
    }

    /// Shared access. Any number of readers coexist, which is what makes
    /// read-only systems genuinely concurrent.
    pub async fn read(&self) -> RwLockReadGuard<'_, World> {
        self.0.read().await
    }

    /// Exclusive access.
    pub async fn write(&self) -> RwLockWriteGuard<'_, World> {
        self.0.write().await
    }

    /// Non-blocking shared access.
    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, World>> {
        self.0.try_read().ok()
    }

    /// Non-blocking exclusive access.
    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, World>> {
        self.0.try_write().ok()
    }

    /// `true` when a writer currently holds the lock.
    pub fn is_write_locked(&self) -> bool {
        self.0.try_read().is_err()
    }

    /// Underlying lock, for drivers that need to spawn extra tasks.
    pub fn inner(&self) -> &Arc<RwLock<World>> {
        &self.0
    }

    /// Borrow the world without async, failing when a guard is outstanding.
    pub fn blocking_read<R>(&self, f: impl FnOnce(&World) -> R) -> Option<R> {
        self.try_read().map(|guard| f(&guard))
    }
}

impl Default for WorldHandle {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Debug for WorldHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorldHandle")
            .field("locked", &self.is_write_locked())
            .finish()
    }
}

/// Cooperative shutdown flag shared by the driver and all systems.
#[derive(Clone, Default)]
pub struct StopToken(Arc<Inner>);

#[derive(Debug, Default)]
struct Inner {
    stopped: std::sync::atomic::AtomicBool,
    reason: RwLock<Option<Cow<'static, str>>>,
    notify: tokio::sync::Notify,
}

impl StopToken {
    /// A token that has not been signalled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests shutdown with a diagnostic reason.
    pub fn stop(&self, reason: impl Into<Cow<'static, str>>) {
        let changed = !self.0.stopped.swap(true, Ordering::AcqRel);
        if changed
            && let Ok(mut guard) = self.0.reason.try_write()
        {
            *guard = Some(reason.into());
        }
        self.0.notify.notify_waiters();
    }

    /// `true` once shutdown was requested.
    pub fn is_stopped(&self) -> bool {
        self.0.stopped.load(Ordering::Acquire)
    }

    /// Reason passed to [`StopToken::stop`].
    pub fn reason(&self) -> Option<Cow<'static, str>> {
        self.0.reason.read_blocking()
    }

    /// Resolves once shutdown is requested; safe to poll from many systems.
    pub async fn wait(&self) {
        loop {
            let notified = self.0.notify.notified();
            if self.is_stopped() {
                return;
            }
            notified.await;
        }
    }

    /// Future that resolves on the earlier of shutdown or the timeout.
    pub async fn wait_timeout(&self, timeout: Duration) -> bool {
        tokio::time::timeout(timeout, self.wait()).await.is_ok()
    }
}

impl fmt::Debug for StopToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StopToken")
            .field("stopped", &self.is_stopped())
            .finish()
    }
}

trait ReadBlocking {
    fn read_blocking(&self) -> Option<Cow<'static, str>>;
}

impl ReadBlocking for RwLock<Option<Cow<'static, str>>> {
    fn read_blocking(&self) -> Option<Cow<'static, str>> {
        self.try_read().ok().and_then(|guard| guard.clone())
    }
}

/// Everything a system needs, cheap to clone because every field is a handle.
///
/// The context is passed **by value** so that a system's future owns it and stays
/// `Send` — the requirement for spawning it onto a tokio worker in a parallel
/// stage.
#[derive(Clone)]
pub struct SystemContext {
    /// Shared world.
    pub world: WorldHandle,
    /// Deferred mutations applied at the next stage boundary.
    pub commands: CommandQueue,
    /// Live frame counter, updated by the driver.
    frame: Arc<AtomicU64>,
    /// Driver start time, used by time-based periodic systems.
    started: tokio::time::Instant,
    /// Cooperative shutdown flag.
    stop: StopToken,
}

impl SystemContext {
    /// Builds a context for a driver.
    pub fn new(world: WorldHandle) -> Self {
        Self {
            world,
            commands: CommandQueue::default(),
            frame: Arc::new(AtomicU64::new(0)),
            started: tokio::time::Instant::now(),
            stop: StopToken::new(),
        }
    }

    /// Replaces the frame counter with a shared one (used by the driver).
    pub fn with_frame_counter(mut self, frame: Arc<AtomicU64>) -> Self {
        self.frame = frame;
        self
    }

    /// Replaces the stop flag.
    pub fn with_stop_token(mut self, stop: StopToken) -> Self {
        self.stop = stop;
        self
    }

    /// Current frame number, starting at 0.
    #[inline]
    pub fn frame_number(&self) -> u64 {
        self.frame.load(Ordering::Acquire)
    }

    /// Advances the shared frame counter and returns the new value.
    pub fn advance_frame(&self) -> u64 {
        self.frame.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Time since the driver started.
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Stop flag of this run.
    pub fn stop_token(&self) -> &StopToken {
        &self.stop
    }

    /// `true` once shutdown was requested.
    pub fn is_stopping(&self) -> bool {
        self.stop.is_stopped()
    }

    /// Requests shutdown of the whole app.
    pub fn stop(&self, reason: impl Into<Cow<'static, str>>) {
        self.stop.stop(reason);
    }

    /// Queues a deferred world mutation.
    pub fn enqueue<F>(&self, command: F)
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        self.commands.push(Command::new(command));
    }

    /// Queues a named deferred world mutation.
    pub fn enqueue_named(&self, name: impl Into<Cow<'static, str>>, command: Command) {
        let mut command = command;
        if command.description.is_empty() {
            command.description = name.into();
        }
        self.commands.push(command);
    }

    /// Number of queued commands.
    pub fn pending_commands(&self) -> usize {
        self.commands.len()
    }

    /// Takes every queued command without applying them.
    pub fn take_commands(&self) -> Vec<Command> {
        self.commands.take_all()
    }

    /// Applies every queued command under one write lock. The driver calls this at
    /// stage boundaries; systems call it when they need their own writes visible
    /// immediately.
    pub async fn apply_commands(&self) -> usize {
        let pending = self.commands.take_all();
        if pending.is_empty() {
            return 0;
        }
        let mut world = self.world.write().await;
        let count = pending.len();
        for command in pending {
            command.apply(&mut world);
        }
        count
    }

    /// Runs `f` with shared world access.
    pub async fn with_world<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&World) -> R + Send,
        R: Send,
    {
        let world = self.world.read().await;
        f(&world)
    }

    /// Runs `f` with exclusive world access.
    pub async fn with_world_mut<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&mut World) -> R + Send,
        R: Send,
    {
        let mut world = self.world.write().await;
        f(&mut world)
    }

    /// Convenience: insert a resource.
    pub async fn insert_resource<R: Resource>(&self, value: R) -> Option<R> {
        self.with_world_mut(move |world| world.insert_resource(value)).await
    }

    /// Convenience: read a resource, `None` when absent.
    pub async fn read_resource<R, F>(&self, f: F) -> R
    where
        R: Send + Clone,
        F: FnOnce(Option<&xihe_core::World>) -> R + Send,
    {
        let world = self.world.read().await;
        f(Some(&*world))
    }
}

impl fmt::Debug for SystemContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemContext")
            .field("frame", &self.frame_number())
            .field("pending_commands", &self.pending_commands())
            .field("stopping", &self.is_stopping())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Counter(usize);

    #[tokio::test]
    async fn commands_apply_at_the_boundary() {
        let ctx = SystemContext::new(WorldHandle::empty());
        ctx.enqueue(|world| {
            let entity = world.spawn();
            world.insert(entity, Counter(1));
        });
        assert_eq!(ctx.pending_commands(), 1);
        assert_eq!(ctx.apply_commands().await, 1);
        assert_eq!(ctx.pending_commands(), 0);
        let world = ctx.world.read().await;
        assert_eq!(world.len(), 1);
        assert_eq!(world.count::<Counter>(), 1);
    }

    #[tokio::test]
    async fn several_readers_coexist() {
        let handle = WorldHandle::empty();
        let a = handle.clone();
        let b = handle.clone();
        let left = handle.read().await;
        let right = b.read().await;
        assert_eq!(left.len(), right.len());
        drop(left);
        drop(right);
        assert!(a.try_write().is_some(), "write lock free once readers drop");
    }

    #[tokio::test]
    async fn stop_token_is_idempotent_and_observable() {
        let token = StopToken::new();
        assert!(!token.is_stopped());
        token.stop("done");
        token.stop("ignored");
        assert!(token.is_stopped());
        assert_eq!(token.reason().as_deref(), Some("done"));
        token.wait().await;
        assert!(token.wait_timeout(Duration::from_millis(5)).await);
    }

    #[tokio::test]
    async fn frame_counter_is_shared_across_clones() {
        let ctx = SystemContext::new(WorldHandle::empty());
        let other = ctx.clone();
        ctx.advance_frame();
        assert_eq!(other.frame_number(), 1);
    }

    #[tokio::test]
    async fn resources_round_trip() {
        let ctx = SystemContext::new(WorldHandle::empty());
        ctx.insert_resource(Counter(7)).await;
        let seen = ctx.with_world(|world| world.resource::<Counter>().map(|c| c.0)).await;
        assert_eq!(seen, Some(7));
    }
}
