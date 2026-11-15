//! 延迟的世界变更。

use std::borrow::Cow;
use std::fmt;
use std::sync::{Arc, Mutex};

use xihe_core::World;

/// 一个装箱闭包，在调度器为它拿到写锁后变更世界。
///
/// 命令是只持有读守卫的系统——或在另一个任务上运行的工具——
/// 仍能改变世界的方式：它把变更入队，而不是升级锁，
/// 升级锁会让由其自身读者持有的 `RwLock` 死锁。
pub struct Command {
    /// 人类可读的用途，显示在 trace 与错误报告中。
    pub description: Cow<'static, str>,
    action: Box<dyn FnOnce(&mut World) + Send>,
}

impl Command {
    /// 将一个闭包包装为命令。
    pub fn new<F>(action: F) -> Self
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        Self {
            description: Cow::Borrowed(""),
            action: Box::new(action),
        }
    }

    /// 包装一个闭包并为它命名。
    pub fn with_description(mut self, description: impl Into<Cow<'static, str>>) -> Self {
        self.description = description.into();
        self
    }

    /// 针对 `world` 运行该命令。
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

/// [`Command`] 的多生产者队列，由克隆的系统上下文共享。
#[derive(Clone, Default)]
pub struct CommandQueue {
    inner: Arc<Mutex<Vec<Command>>>,
}

impl CommandQueue {
    /// 一个空队列。
    pub fn new() -> Self {
        Self::default()
    }

    /// 将一个命令入队。
    pub fn push(&self, command: Command) {
        // 该锁只为一次 `Vec` push 持有，绝不跨 await。
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(command);
    }

    /// 直接将一个闭包入队。
    pub fn enqueue<F>(&self, action: F)
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        self.push(Command::new(action));
    }

    /// 移除每个排队的命令。
    pub fn take_all(&self) -> Vec<Command> {
        std::mem::take(
            &mut *self
                .inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    /// 待定命令的数量。
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// 没有待定项时为 `true`。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 取出每个命令并针对 `world` 运行。
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
