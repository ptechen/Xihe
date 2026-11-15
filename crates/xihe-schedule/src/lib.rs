//! # xihe-schedule
//!
//! Async execution for the Xihe ECS: the [`System`] trait, declared [`Access`],
//! [`Schedule`]s with concurrent stages, deferred [`Command`]s and a typed
//! [`MessageBus`].
//!
//! ## How concurrency works
//!
//! A [`World`](xihe_core::World) lives behind one `tokio::sync::RwLock`. Each
//! system asks for a guard itself, and the scheduler uses the declared
//! [`Access`] to decide who may run in the same [`Stage::Parallel`] group:
//!
//! * read-only systems are polled together, so an agent awaiting an LLM response
//!   does not block its neighbours;
//! * a system that writes components or resources runs alone;
//! * `&mut World` never crosses an `.await`, so no lock guard is held across
//!   suspension unless the system opts into `world.read().await` deliberately.
//!
//! Two rules keep this deadlock-free: a system must not upgrade a read guard it
//! already holds (it would wait for itself), and anything that needs to write
//! while reading should [`SystemContext::enqueue`] a [`Command`] instead.
//!
//! ```
//! use xihe_schedule::{access, system, SystemContext, Schedule, SystemSpec};
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
//! let ctx = SystemContext::new(xihe_schedule::WorldHandle::empty());
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
pub use bus::{BusHandle, BusStats, Inbox, MessageBus, DEFAULT_CAPACITY};
pub use command::{Command, CommandQueue};
pub use context::{StopToken, SystemContext, WorldHandle};
pub use error::{ScheduleError, SystemError};
pub use schedule::{Schedule, Stage, SystemSpec, one_shot};
pub use system::{FnSystem, Period, SysResult, System, SystemFuture, boxed, run_guarded, system};
pub use work::Work;

/// Declares world access for a system.
///
/// ```
/// use xihe_schedule::access;
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
    (none) => { $crate::Access::empty() };
    (read_only) => { $crate::Access::read_only() };
    (exclusive) => { $crate::Access::exclusive() };
    (reads = [$($read:ty),* $(,)?], writes = [$($write:ty),* $(,)?]) => {{
        let access = $crate::Access::empty();
        $(let access = access.and_read::<$read>();)*
        $(let access = access.and_write::<$write>();)*
        access
    }};
    (reads = [$($read:ty),* $(,)?]) => {{
        let access = $crate::Access::empty();
        $(let access = access.and_read::<$read>();)*
        access
    }};
    (writes = [$($write:ty),* $(,)?]) => {{
        let access = $crate::Access::empty();
        $(let access = access.and_write::<$write>();)*
        access
    }};
}

/// Builds a named async system, optionally with declared access and cadence.
///
/// ```
/// use xihe_schedule::{system, SysResult, SystemContext};
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
/// assert_eq!(plan.period(), Some(xihe_schedule::Period::Frames(2)));
/// ```
#[macro_export]
macro_rules! system {
    ($name:expr, |$ctx:ident| $body:block) => {
        $crate::system($name, move |$ctx: $crate::SystemContext| async move $body)
    };
    ($name:expr, |$ctx:ident| every = $frames:expr, $body:block) => {
        $crate::system($name, move |$ctx: $crate::SystemContext| async move $body)
            .every_frames($frames)
    };
    ($name:expr, |$ctx:ident| reads = [$($read:ty),* $(,)?], writes = [$($write:ty),* $(,)?], $body:block) => {
        $crate::system($name, move |$ctx: $crate::SystemContext| async move $body)
            .with_access($crate::access!(reads = [$($read),*], writes = [$($write),*]))
    };
    ($name:expr, |$ctx:ident| reads = [$($read:ty),* $(,)?], writes = [$($write:ty),* $(,)?], every = $frames:expr, $body:block) => {
        $crate::system($name, move |$ctx: $crate::SystemContext| async move $body)
            .with_access($crate::access!(reads = [$($read),*], writes = [$($write),*]))
            .every_frames($frames)
    };
    ($name:expr, $f:expr) => {
        $crate::system($name, $f)
    };
}

/// Everything a user normally imports at once.
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
