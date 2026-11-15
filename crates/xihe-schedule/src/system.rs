//! The system trait: an async unit of agent logic with declared access.

use std::borrow::Cow;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;

use crate::access::Access;
use crate::context::SystemContext;
use crate::error::SystemError;

/// Result of one system run.
pub type SysResult<T = ()> = Result<T, SystemError>;

/// Boxed, `Send` future returned by boxed systems.
pub type SystemFuture<'a> = Pin<Box<dyn Future<Output = SysResult> + Send + 'a>>;

/// How often a system should run inside a schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Period {
    /// Every `n` frames (`n` >= 1).
    Frames(u64),
    /// At most once per `Duration`.
    Time(Duration),
}

impl Period {
    /// Runs every frame.
    pub fn every_frame() -> Self {
        Self::Frames(1)
    }

    /// Millisecond shorthand for time-based periods.
    pub fn millis(millis: u64) -> Self {
        Self::Time(Duration::from_millis(millis))
    }

}

/// A single step of agent logic.
///
/// Implement it by hand for stateful systems (a retry counter, a client pool) or
/// use [`system`] to wrap a closure. The context is passed **by value** so the
/// returned future owns everything it needs and stays `Send`, which is what lets
/// a parallel stage spawn its systems onto tokio workers.
///
/// ```
/// use xihe_schedule::{system, Access, SystemContext, SysResult};
///
/// #[derive(Default)]
/// struct Ticks(u64);
///
/// let counter = system("tick", |ctx: SystemContext| async move {
///     ctx.with_world_mut(|world| {
///         world.init_resource::<Ticks>().0 += 1;
///     })
///     .await;
///     Ok(())
/// })
/// .reads::<u32>();
/// assert!(!counter.access().conflicts(&Access::empty()));
/// ```
#[async_trait]
pub trait System: Send + Sync {
    /// Label used in logs, errors and ordering hints.
    fn name(&self) -> &str {
        "system"
    }

    /// World access this system needs. The default is [`Access::exclusive`]:
    /// systems that want to be parallelised must opt in.
    fn access(&self) -> Access {
        Access::exclusive()
    }

    /// Optional cadence override for the scheduler.
    fn period(&self) -> Option<Period> {
        None
    }

    /// Runs once per scheduled appearance.
    async fn run(&mut self, ctx: SystemContext) -> SysResult;
}

/// A [`System`] built from a closure plus declarative metadata.
///
/// Create it with [`system`], then chain access and cadence helpers.
pub struct FnSystem {
    name: Cow<'static, str>,
    access: Option<Access>,
    period: Option<Period>,
    action: Box<dyn FnMut(SystemContext) -> SystemFuture<'static> + Send + Sync + 'static>,
}

impl FnSystem {
    /// Wraps a closure that turns a context into a future.
    pub fn new<F, Fut>(name: impl Into<Cow<'static, str>>, mut f: F) -> Self
    where
        F: FnMut(SystemContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = SysResult> + Send + 'static,
    {
        let action: Box<dyn FnMut(SystemContext) -> SystemFuture<'static> + Send + Sync> =
            Box::new(move |ctx| Box::pin(f(ctx)));
        Self {
            name: name.into(),
            access: None,
            period: None,
            action,
        }
    }

    /// Replaces the access declaration.
    pub fn with_access(mut self, access: Access) -> Self {
        self.access = Some(access);
        self
    }

    /// Declares a read of `T`. The first declaration also drops the implicit
    /// exclusive default, otherwise `reads` would be a no-op.
    pub fn reads<T: 'static>(mut self) -> Self {
        self.access = Some(self.unwrap_access().and_read::<T>());
        self
    }

    /// Declares a write of `T`.
    pub fn writes<T: 'static>(mut self) -> Self {
        self.access = Some(self.unwrap_access().and_write::<T>());
        self
    }

    /// Declares "reads the world, writes nothing", the cheapest way to allow
    /// concurrency.
    pub fn read_only(mut self) -> Self {
        self.access = Some(Access::read_only());
        self
    }

    /// Declares exclusive access (the default).
    pub fn exclusive(mut self) -> Self {
        self.access = Some(Access::exclusive());
        self
    }

    fn unwrap_access(&self) -> Access {
        self.access.clone().unwrap_or_default()
    }

    /// Runs every `n` frames.
    pub fn every_frames(mut self, frames: u64) -> Self {
        self.period = Some(Period::Frames(frames.max(1)));
        self
    }

    /// Runs at most once per `period`.
    pub fn every(mut self, period: Duration) -> Self {
        self.period = Some(Period::Time(period));
        self
    }

    /// System name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Declared access, [`Access::exclusive`] when nothing was declared.
    pub fn access(&self) -> Access {
        self.access.clone().unwrap_or_else(Access::exclusive)
    }

    /// Cadence override.
    pub fn period(&self) -> Option<Period> {
        self.period
    }

    /// Erases the concrete system into a boxed trait object.
    pub fn into_boxed(self) -> Box<dyn System> {
        Box::new(self)
    }

    /// Borrows the boxed action, for drivers that must not move the system.
    pub fn call(&mut self, ctx: SystemContext) -> SystemFuture<'_> {
        (self.action)(ctx)
    }
}

#[async_trait]
impl System for FnSystem {
    fn name(&self) -> &str {
        &self.name
    }

    fn access(&self) -> Access {
        FnSystem::access(self)
    }

    fn period(&self) -> Option<Period> {
        self.period
    }

    async fn run(&mut self, ctx: SystemContext) -> SysResult {
        (self.action)(ctx).await
    }
}

impl fmt::Debug for FnSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnSystem")
            .field("name", &self.name)
            .field("access", &self.access())
            .field("period", &self.period)
            .finish()
    }
}

/// Wraps an async closure as a named system with exclusive access.
///
/// The closure receives an owned [`SystemContext`]:
///
/// ```
/// use xihe_schedule::{system, SystemContext};
///
/// let greet = system("greet", |ctx: SystemContext| async move {
///     assert!(ctx.frame_number() < 1_000_000);
///     Ok(())
/// });
/// assert_eq!(greet.name(), "greet");
/// ```
pub fn system<F, Fut>(name: impl Into<Cow<'static, str>>, f: F) -> FnSystem
where
    F: FnMut(SystemContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = SysResult> + Send + 'static,
{
    FnSystem::new(name, f)
}

/// Erases any concrete system for storage in a schedule.
pub fn boxed<S: System + 'static>(system: S) -> Box<dyn System> {
    Box::new(system)
}

/// Runs a future to completion, turning a panic into a [`SystemError`] so one
/// failing agent cannot unwind the whole schedule.
///
/// Each activation runs inside a `system` tracing span carrying the system's
/// name, so a host subscriber sees per-system timing and failures. The span is
/// entered with [`tracing::Instrument`] rather than a guard, which keeps it
/// correct across the future's await points.
///
/// Tokio locks are not poisoned, so a guard dropped by the unwinding future is
/// simply released and the next frame starts from a consistent world.
pub async fn run_guarded<F>(name: &str, future: F) -> SysResult
where
    F: Future<Output = SysResult>,
{
    use futures::FutureExt;
    use tracing::Instrument;
    let span = tracing::info_span!("system", name = name, failed = tracing::field::Empty);
    let result = std::panic::AssertUnwindSafe(future)
        .catch_unwind()
        .instrument(span.clone())
        .await;
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => {
            span.record("failed", true);
            Err(error)
        }
        Err(_panic) => {
            span.record("failed", true);
            Err(SystemError::new(name, "system panicked"))
        }
    }
}

/// Diagnostic view of a boxed system.
pub trait SystemName {
    /// Name of the boxed system.
    fn system_name(&self) -> String;
}

impl SystemName for Box<dyn System> {
    fn system_name(&self) -> String {
        self.name().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mark;

    #[tokio::test]
    async fn closure_system_runs_and_declares_access() {
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let seen = counter.clone();
        let subject = system("count", move |_ctx: SystemContext| {
            let seen = seen.clone();
            async move {
                seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
        })
        .reads::<Mark>()
        .every_frames(2);

        assert_eq!(subject.name(), "count");
        assert_eq!(subject.period(), Some(Period::Frames(2)));
        assert!(!subject.access().conflicts(&Access::empty()));

        let ctx = SystemContext::new(crate::context::WorldHandle::empty());
        let mut subject = subject;
        subject.run(ctx.clone()).await.unwrap();
        subject.run(ctx).await.unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn panics_are_contained_by_run_guarded() {
        let ctx = SystemContext::new(crate::context::WorldHandle::empty());
        let world = ctx.world.clone();
        let mut subject = system("boom", move |_ctx: SystemContext| {
            let world = world.clone();
            async move {
                let guard = world.read().await;
                assert_eq!(guard.len(), 0);
                panic!("agent exploded")
            }
        });
        let name = subject.name().to_string();
        let error = run_guarded(&name, subject.call(ctx.clone()))
            .await
            .unwrap_err();
        assert_eq!(error.name, "boom");
        assert!(error.to_string().contains("panicked"));
        // the read guard taken before the panic must not wedge the world
        assert_eq!(ctx.world.read().await.len(), 0);
    }

    #[derive(Clone, Default)]
    struct SpanCapture {
        names: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        next_id: std::sync::Arc<std::sync::atomic::AtomicU64>,
    }

    #[derive(Default)]
    struct NameField(Option<String>);

    impl tracing::field::Visit for NameField {
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            if field.name() == "name" {
                self.0 = Some(value.to_string());
            }
        }
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "name" {
                self.0 = Some(format!("{value:?}"));
            }
        }
    }

    impl tracing::Subscriber for SpanCapture {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn register_callsite(
            &self,
            _: &'static tracing::Metadata<'static>,
        ) -> tracing::subscriber::Interest {
            tracing::subscriber::Interest::always()
        }
        fn new_span(&self, attrs: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            let mut field = NameField::default();
            attrs.record(&mut field);
            let label = field
                .0
                .unwrap_or_else(|| attrs.metadata().name().to_string());
            self.names.lock().unwrap().push(label);
            let id = self
                .next_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            tracing::span::Id::from_u64(id)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, _: &tracing::Event<'_>) {}
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    #[tokio::test]
    async fn each_activation_runs_in_a_named_system_span() {
        let capture = SpanCapture::default();
        let seen = capture.names.clone();
        // tracing caches each callsite's interest process-wide. With a single
        // registered subscriber it rebuilds that cache against the thread-local
        // default of whichever thread happens to trigger a rebuild, so parallel
        // tests running on the no-op subscriber race to flip the shared `system`
        // callsite to `never` and these spans vanish nondeterministically. A
        // second registered subscriber forces the thread-independent
        // dispatcher-list rebuild path, keeping the capture's interest stable no
        // matter which thread rebuilds; the outer one only exists to do that.
        let _outer = tracing::subscriber::set_default(SpanCapture::default());
        let _guard = tracing::subscriber::set_default(capture);
        tracing::callsite::rebuild_interest_cache();
        let ctx = SystemContext::new(crate::context::WorldHandle::empty());

        let mut ok = system("alpha", |_ctx: SystemContext| async { Ok(()) });
        run_guarded("alpha", ok.call(ctx.clone())).await.unwrap();

        let mut boom = system("beta", |_ctx: SystemContext| async { panic!("exploded") });
        let error = run_guarded("beta", boom.call(ctx.clone()))
            .await
            .unwrap_err();
        assert_eq!(error.name, "beta");

        let spans = seen.lock().unwrap();
        assert!(spans.iter().any(|span| span == "alpha"), "spans: {spans:?}");
        assert!(spans.iter().any(|span| span == "beta"), "spans: {spans:?}");
    }
}
