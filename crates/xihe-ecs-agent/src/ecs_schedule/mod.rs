//! # xihe::ecs_schedule
//!
//! Xihe ECS 的异步执行：[`System`] trait、声明的 [`Access`]、
//! 带并发阶段的 [`Schedule`]、延迟的 [`Command`] 以及类型化的
//! [`MessageBus`]。
//!
//! ## 并发如何工作
//!
//! 一个 [`World`](crate::ecs_core::World) 置于一把 `tokio::sync::RwLock` 之后。每个
//! 系统自行请求守卫，调度器用声明的 [`Access`] 来决定谁可以在同一个
//! [`Stage::Parallel`] 组中运行：
//!
//! * 只读系统被一起轮询，因此等待 LLM 响应的智能体不会阻塞它的邻居；
//! * 写入组件或资源的系统单独运行；
//! * `&mut World` 绝不跨越 `.await`，因此除非系统刻意选择
//!   `world.read().await`，否则不会跨挂起持有任何锁守卫。
//!
//! 两条规则保证无死锁：系统不得升级它已持有的读守卫
//! （那会等待它自己），任何需要在读取时写入的操作都应改为
//! [`SystemContext::enqueue`] 一个 [`Command`]。
//!
//! ```
//! use xihe::ecs_schedule::{system, SystemContext, Schedule, SystemSpec};
//! use xihe::access;
//!
//! #[derive(Default)]
//! struct InboxCount(usize);
//! #[derive(Default)]
//! struct Clock(u64);
//!
//! let mut schedule = Schedule::new();
//! schedule.add(SystemSpec::new(
//!     "tick",
//!     system("tick", |ctx: SystemContext| async move {
//!         ctx.with_world_mut(|world| world.init_resource::<Clock>().0 += 1).await;
//!         Ok(())
//!     })
//!     .with_access(access!(writes = [Clock])),
//! ));
//! schedule
//!     .add(SystemSpec::new(
//!         "observe",
//!         system("observe", |ctx: SystemContext| async move {
//!             let ticks = ctx.with_world(|world| world.resource::<Clock>().map(|c| c.0)).await;
//!             assert!(ticks >= Some(1));
//!             Ok(())
//!         })
//!         .with_access(access!(reads = [Clock])),
//!     ));
//!
//! # futures::executor::block_on(async {
//! let ctx = SystemContext::new(xihe::ecs_schedule::WorldHandle::empty());
//! schedule.execute(&ctx).await.unwrap();
//! # });
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod access;
pub mod bus;
pub mod command;
pub mod context;
pub mod error;
pub mod schedule;
pub mod system;
pub mod work;

pub use access::Access;
pub use bus::{BusHandle, BusStats, DEFAULT_CAPACITY, Inbox, MessageBus};
pub use command::{Command, CommandQueue};
pub use context::{StopToken, SystemContext, WorldHandle};
pub use error::{ScheduleError, SystemError};
pub use schedule::{Schedule, Stage, SystemSpec, one_shot};
pub use system::{FnSystem, Period, SysResult, System, SystemFuture, boxed, run_guarded, system};
pub use work::Work;

/// 为一个系统声明世界访问权限。
///
/// ```
/// use xihe::access;
///
/// #[derive(Default)]
/// struct Conversation;
/// #[derive(Default)]
/// struct AgentState;
///
/// let a = access!(reads = [Conversation], writes = [AgentState]);
/// let b = access!(reads = [AgentState]);
/// assert!(a.conflicts(&b), "a writes what b reads");
/// assert!(!a.conflicts(&access!(reads = [Conversation])), "same read is fine");
/// assert!(access!(none).compatible(&a));
/// assert!(access!(read_only).conflicts(&a));
/// ```
#[macro_export]
macro_rules! access {
    (none) => { $crate::ecs_schedule::Access::empty() };
    (read_only) => { $crate::ecs_schedule::Access::read_only() };
    (exclusive) => { $crate::ecs_schedule::Access::exclusive() };
    (reads = [$($read:ty),* $(,)?], writes = [$($write:ty),* $(,)?]) => {{
        let access = $crate::ecs_schedule::Access::empty();
        $(let access = access.and_read::<$read>();)*
        $(let access = access.and_write::<$write>();)*
        access
    }};
    (reads = [$($read:ty),* $(,)?]) => {{
        let access = $crate::ecs_schedule::Access::empty();
        $(let access = access.and_read::<$read>();)*
        access
    }};
    (writes = [$($write:ty),* $(,)?]) => {{
        let access = $crate::ecs_schedule::Access::empty();
        $(let access = access.and_write::<$write>();)*
        access
    }};
}

/// 构建一个命名的异步系统，可选地带有声明的访问权限与节奏。
///
/// ```
/// use xihe::ecs_schedule::{SysResult, SystemContext};
/// use xihe::system;
///
/// #[derive(Default)]
/// struct Steps(u32);
///
/// let plan = system!("plan",
///     |ctx|
///     reads = [],
///     writes = [Steps],
///     every = 2,
///     {
///         ctx.with_world_mut(|world| world.init_resource::<Steps>().0 += 1).await;
///         Ok(())
///     }
/// );
/// assert_eq!(plan.name(), "plan");
/// assert_eq!(plan.period(), Some(xihe::ecs_schedule::Period::Frames(2)));
/// ```
#[macro_export]
macro_rules! system {
    ($name:expr, |$ctx:ident| $body:block) => {
        $crate::ecs_schedule::system($name, move |$ctx: $crate::ecs_schedule::SystemContext| async move $body)
    };
    ($name:expr, |$ctx:ident| every = $frames:expr, $body:block) => {
        $crate::ecs_schedule::system($name, move |$ctx: $crate::ecs_schedule::SystemContext| async move $body)
            .every_frames($frames)
    };
    ($name:expr, |$ctx:ident| reads = [$($read:ty),* $(,)?], writes = [$($write:ty),* $(,)?], $body:block) => {
        $crate::ecs_schedule::system($name, move |$ctx: $crate::ecs_schedule::SystemContext| async move $body)
            .with_access($crate::access!(reads = [$($read),*], writes = [$($write),*]))
    };
    ($name:expr, |$ctx:ident| reads = [$($read:ty),* $(,)?], writes = [$($write:ty),* $(,)?], every = $frames:expr, $body:block) => {
        $crate::ecs_schedule::system($name, move |$ctx: $crate::ecs_schedule::SystemContext| async move $body)
            .with_access($crate::access!(reads = [$($read),*], writes = [$($write),*]))
            .every_frames($frames)
    };
    ($name:expr, $f:expr) => {
        $crate::ecs_schedule::system($name, $f)
    };
}

/// 用户通常一次性导入的全部内容。
pub mod prelude {
    pub use super::{
        Access, BusHandle, Command, CommandQueue, Inbox, MessageBus, Period, Schedule, Stage,
        StopToken, SysResult, System, SystemContext, SystemError, SystemSpec, Work, WorldHandle,
        access, boxed, one_shot, system,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Reader;
    struct Writer;

    #[test]
    fn access_macro_matches_manual_declaration() {
        let from_macro = access!(reads = [Reader], writes = [Writer]);
        let manual = Access::read::<Reader>().and_write::<Writer>();
        assert_eq!(from_macro, manual);
        assert_eq!(access!(none), Access::empty());
        assert_eq!(access!(exclusive), Access::exclusive());
        assert_eq!(access!(read_only), Access::read_only());
        assert_eq!(access!(writes = [Writer]), Access::write::<Writer>());
    }

    #[tokio::test]
    async fn system_macro_produces_a_runnable_spec_member() {
        let mut subject = system!("cadenced", |ctx| reads = [], writes = [], every = 3, {
            let _ = ctx.frame_number();
            Ok(())
        });
        assert_eq!(subject.name(), "cadenced");
        assert_eq!(subject.period(), Some(Period::Frames(3)));
        let ctx = SystemContext::new(WorldHandle::empty());
        subject.call(ctx.clone()).await.unwrap();
    }

    #[test]
    fn prelude_names_resolve() {
        let _: Option<fn() -> Access> = Some(|| access!(none));
    }
}
