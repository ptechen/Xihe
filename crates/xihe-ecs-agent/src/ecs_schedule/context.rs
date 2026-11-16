//! 交给系统的句柄：共享世界、帧计数器与停止标志。

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::ecs_core::{Resource, World};
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::ecs_schedule::command::{Command, CommandQueue};

/// 指向世界的 `Send` 句柄，由调度器中的每个系统共享。
///
/// 系统自行请求读守卫或写守卫；调度器只基于声明的
/// [`Access`](crate::ecs_schedule::Access) 决定*谁可以同时持有哪个守卫*。
#[derive(Clone)]
pub struct WorldHandle(Arc<RwLock<World>>);

impl WorldHandle {
    /// 包装一个已有世界。
    pub fn new(world: World) -> Self {
        Self(Arc::new(RwLock::new(world)))
    }

    /// 包装一个空世界。
    pub fn empty() -> Self {
        Self::new(World::new())
    }

    /// 共享访问。任意数量的读者共存，这正是让只读系统
    /// 真正并发的原因。
    pub async fn read(&self) -> RwLockReadGuard<'_, World> {
        self.0.read().await
    }

    /// 独占访问。
    pub async fn write(&self) -> RwLockWriteGuard<'_, World> {
        self.0.write().await
    }

    /// 非阻塞共享访问。
    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, World>> {
        self.0.try_read().ok()
    }

    /// 非阻塞独占访问。
    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, World>> {
        self.0.try_write().ok()
    }

    /// 当前有写者持有锁时为 `true`。
    pub fn is_write_locked(&self) -> bool {
        self.0.try_read().is_err()
    }

    /// 底层锁，供需要派生额外任务的驱动器使用。
    pub fn inner(&self) -> &Arc<RwLock<World>> {
        &self.0
    }

    /// 非异步地借用世界，当有守卫未释放时失败。
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

/// 由驱动器与所有系统共享的协作式关停标志。
#[derive(Clone, Default)]
pub struct StopToken(Arc<Inner>);

#[derive(Debug, Default)]
struct Inner {
    stopped: std::sync::atomic::AtomicBool,
    reason: RwLock<Option<Cow<'static, str>>>,
    notify: tokio::sync::Notify,
}

impl StopToken {
    /// 一个尚未被触发的令牌。
    pub fn new() -> Self {
        Self::default()
    }

    /// 请求关停，并附带诊断原因。
    pub fn stop(&self, reason: impl Into<Cow<'static, str>>) {
        let changed = !self.0.stopped.swap(true, Ordering::AcqRel);
        if changed && let Ok(mut guard) = self.0.reason.try_write() {
            *guard = Some(reason.into());
        }
        self.0.notify.notify_waiters();
    }

    /// 一旦请求了关停即为 `true`。
    pub fn is_stopped(&self) -> bool {
        self.0.stopped.load(Ordering::Acquire)
    }

    /// 传给 [`StopToken::stop`] 的原因。
    pub fn reason(&self) -> Option<Cow<'static, str>> {
        self.0.reason.read_blocking()
    }

    /// 一旦请求关停即完成；可安全地从多个系统轮询。
    pub async fn wait(&self) {
        loop {
            let notified = self.0.notify.notified();
            if self.is_stopped() {
                return;
            }
            notified.await;
        }
    }

    /// 在关停或超时两者中较早者发生时完成的 Future。
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

/// 系统所需的一切，克隆廉价，因为每个字段都是句柄。
///
/// 上下文**按值**传递，使系统的 future 拥有它并保持 `Send`——
/// 这是在并行阶段中将它派生到 tokio worker 上的要求。
#[derive(Clone)]
pub struct SystemContext {
    /// 共享世界。
    pub world: WorldHandle,
    /// 在下一个阶段边界应用的延迟变更。
    pub commands: CommandQueue,
    /// 实时帧计数器，由驱动器更新。
    frame: Arc<AtomicU64>,
    /// 驱动器启动时间，供基于时间的周期系统使用。
    started: tokio::time::Instant,
    /// 协作式关停标志。
    stop: StopToken,
}

impl SystemContext {
    /// 为驱动器构建一个上下文。
    pub fn new(world: WorldHandle) -> Self {
        Self {
            world,
            commands: CommandQueue::default(),
            frame: Arc::new(AtomicU64::new(0)),
            started: tokio::time::Instant::now(),
            stop: StopToken::new(),
        }
    }

    /// 用一个共享的帧计数器替换当前的（由驱动器使用）。
    pub fn with_frame_counter(mut self, frame: Arc<AtomicU64>) -> Self {
        self.frame = frame;
        self
    }

    /// 替换停止标志。
    pub fn with_stop_token(mut self, stop: StopToken) -> Self {
        self.stop = stop;
        self
    }

    /// 当前帧号，从 0 开始。
    #[inline]
    pub fn frame_number(&self) -> u64 {
        self.frame.load(Ordering::Acquire)
    }

    /// 推进共享帧计数器并返回新值。
    pub fn advance_frame(&self) -> u64 {
        self.frame.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// 自驱动器启动以来的时间。
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// 本次运行的停止标志。
    pub fn stop_token(&self) -> &StopToken {
        &self.stop
    }

    /// 一旦请求了关停即为 `true`。
    pub fn is_stopping(&self) -> bool {
        self.stop.is_stopped()
    }

    /// 请求关停整个应用。
    pub fn stop(&self, reason: impl Into<Cow<'static, str>>) {
        self.stop.stop(reason);
    }

    /// 将一个延迟的世界变更入队。
    pub fn enqueue<F>(&self, command: F)
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        self.commands.push(Command::new(command));
    }

    /// 将一个命名的延迟世界变更入队。
    pub fn enqueue_named(&self, name: impl Into<Cow<'static, str>>, command: Command) {
        let mut command = command;
        if command.description.is_empty() {
            command.description = name.into();
        }
        self.commands.push(command);
    }

    /// 排队命令的数量。
    pub fn pending_commands(&self) -> usize {
        self.commands.len()
    }

    /// 取出每个排队的命令但不应用它们。
    pub fn take_commands(&self) -> Vec<Command> {
        self.commands.take_all()
    }

    /// 在一把写锁下应用每个排队的命令。驱动器在阶段边界调用它；
    /// 系统在需要自己的写入立即可见时调用它。
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

    /// 以共享世界访问运行 `f`。
    pub async fn with_world<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&World) -> R + Send,
        R: Send,
    {
        let world = self.world.read().await;
        f(&world)
    }

    /// 以独占世界访问运行 `f`。
    pub async fn with_world_mut<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&mut World) -> R + Send,
        R: Send,
    {
        let mut world = self.world.write().await;
        f(&mut world)
    }

    /// 便捷方法：插入一个资源。
    pub async fn insert_resource<R: Resource>(&self, value: R) -> Option<R> {
        self.with_world_mut(move |world| world.insert_resource(value))
            .await
    }

    /// 便捷方法：读取一个资源，缺失时为 `None`。
    pub async fn read_resource<R, F>(&self, f: F) -> R
    where
        R: Send + Clone,
        F: FnOnce(Option<&crate::ecs_core::World>) -> R + Send,
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
        let seen = ctx
            .with_world(|world| world.resource::<Counter>().map(|c| c.0))
            .await;
        assert_eq!(seen, Some(7));
    }
}
