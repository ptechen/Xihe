//! 系统 trait：带声明访问权限的智能体逻辑异步单元。

use std::borrow::Cow;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;

use crate::ecs_schedule::access::Access;
use crate::ecs_schedule::context::SystemContext;
use crate::ecs_schedule::error::SystemError;

/// 一次系统运行的结果。
pub type SysResult<T = ()> = Result<T, SystemError>;

/// 装箱系统返回的装箱 `Send` future。
pub type SystemFuture<'a> = Pin<Box<dyn Future<Output = SysResult> + Send + 'a>>;

/// 系统在调度器内应多久运行一次。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Period {
    /// 每 `n` 帧（`n` >= 1）。
    Frames(u64),
    /// 每个 `Duration` 至多一次。
    Time(Duration),
}

impl Period {
    /// 每帧运行。
    pub fn every_frame() -> Self {
        Self::Frames(1)
    }

    /// 基于时间的周期的毫秒简写。
    pub fn millis(millis: u64) -> Self {
        Self::Time(Duration::from_millis(millis))
    }
}

/// 智能体逻辑的一个步骤。
///
/// 为有状态系统（重试计数器、客户端池）手动实现它，或用 [`system`]
/// 包装一个闭包。上下文**按值**传递，使返回的 future 拥有它所需的
/// 一切并保持 `Send`，这正是让并行阶段能把其系统派生到
/// tokio worker 上的原因。
///
/// ```
/// use xihe::ecs_schedule::{system, Access, SystemContext, SysResult};
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
    /// 用于日志、错误与排序提示的标签。
    fn name(&self) -> &str {
        "system"
    }

    /// 此系统需要的世界访问权限。默认是 [`Access::exclusive`]：
    /// 想要被并行化的系统必须主动选择。
    fn access(&self) -> Access {
        Access::exclusive()
    }

    /// 供调度器使用的可选节奏覆盖。
    fn period(&self) -> Option<Period> {
        None
    }

    /// 每次被调度出现时运行一次。
    async fn run(&mut self, ctx: SystemContext) -> SysResult;
}

/// 由一个闭包加声明式元数据构建的 [`System`]。
///
/// 用 [`system`] 创建它，然后链式调用访问权限与节奏辅助方法。
pub struct FnSystem {
    name: Cow<'static, str>,
    access: Option<Access>,
    period: Option<Period>,
    action: Box<dyn FnMut(SystemContext) -> SystemFuture<'static> + Send + Sync + 'static>,
}

impl FnSystem {
    /// 包装一个将上下文转为 future 的闭包。
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

    /// 替换访问权限声明。
    pub fn with_access(mut self, access: Access) -> Self {
        self.access = Some(access);
        self
    }

    /// 声明对 `T` 的读取。第一个声明还会丢弃隐含的独占默认值，
    /// 否则 `reads` 将不起作用。
    pub fn reads<T: 'static>(mut self) -> Self {
        self.access = Some(self.unwrap_access().and_read::<T>());
        self
    }

    /// 声明对 `T` 的写入。
    pub fn writes<T: 'static>(mut self) -> Self {
        self.access = Some(self.unwrap_access().and_write::<T>());
        self
    }

    /// 声明“读取世界、不写入任何东西”，这是允许并发的最廉价方式。
    pub fn read_only(mut self) -> Self {
        self.access = Some(Access::read_only());
        self
    }

    /// 声明独占访问（默认值）。
    pub fn exclusive(mut self) -> Self {
        self.access = Some(Access::exclusive());
        self
    }

    fn unwrap_access(&self) -> Access {
        self.access.clone().unwrap_or_default()
    }

    /// 每 `n` 帧运行。
    pub fn every_frames(mut self, frames: u64) -> Self {
        self.period = Some(Period::Frames(frames.max(1)));
        self
    }

    /// 每个 `period` 至多运行一次。
    pub fn every(mut self, period: Duration) -> Self {
        self.period = Some(Period::Time(period));
        self
    }

    /// 系统名称。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 声明的访问权限，未声明时为 [`Access::exclusive`]。
    pub fn access(&self) -> Access {
        self.access.clone().unwrap_or_else(Access::exclusive)
    }

    /// 节奏覆盖。
    pub fn period(&self) -> Option<Period> {
        self.period
    }

    /// 将具体系统擦除为装箱 trait 对象。
    pub fn into_boxed(self) -> Box<dyn System> {
        Box::new(self)
    }

    /// 借用装箱的动作，供不能移动系统的驱动器使用。
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

/// 将一个异步闭包包装为带独占访问权限的命名系统。
///
/// 闭包接收一个拥有的 [`SystemContext`]：
///
/// ```
/// use xihe::ecs_schedule::{system, SystemContext};
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

/// 擦除任意具体系统以便存放在调度器中。
pub fn boxed<S: System + 'static>(system: S) -> Box<dyn System> {
    Box::new(system)
}

/// 将一个 future 运行至完成，把 panic 转为 [`SystemError`]，使一个
/// 失败的智能体无法 unwind 整个调度器。
///
/// 每次激活都在一个携带系统名称的 `system` tracing span 内运行，
/// 因此宿主订阅者能看到逐系统的计时与失败。该 span 用
/// [`tracing::Instrument`] 而非守卫进入，这使它在 future 的各
/// await 点之间保持正确。
///
/// Tokio 锁不会被毒化，因此被 unwind 的 future 丢弃的守卫只是
/// 被释放，下一帧从一个一致的世界开始。
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

/// 装箱系统的诊断视图。
pub trait SystemName {
    /// 装箱系统的名称。
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

        let ctx = SystemContext::new(crate::ecs_schedule::context::WorldHandle::empty());
        let mut subject = subject;
        subject.run(ctx.clone()).await.unwrap();
        subject.run(ctx).await.unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn panics_are_contained_by_run_guarded() {
        let ctx = SystemContext::new(crate::ecs_schedule::context::WorldHandle::empty());
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
        // panic 之前取得的读守卫绝不能卡死世界
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
        // tracing 在进程范围内缓存每个 callsite 的 interest。只有单个
        // 已注册订阅者时，它会针对恰好触发重建的那个线程的线程局部
        // 默认值来重建该缓存，因此运行在 no-op 订阅者上的并行测试
        // 会竞争把共享的 `system` callsite 翻转为 `never`，使这些 span
        // 非确定性地消失。第二个已注册订阅者会强制走与线程无关的
        // dispatcher 列表重建路径，无论哪个线程重建都保持 capture 的
        // interest 稳定；外层那个订阅者存在的唯一目的就是做到这一点。
        let _outer = tracing::subscriber::set_default(SpanCapture::default());
        let _guard = tracing::subscriber::set_default(capture);
        tracing::callsite::rebuild_interest_cache();
        let ctx = SystemContext::new(crate::ecs_schedule::context::WorldHandle::empty());

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
