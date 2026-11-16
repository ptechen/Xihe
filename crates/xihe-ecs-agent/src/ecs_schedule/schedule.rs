//! 调度器：系统的有序阶段，阶段内部可并发。

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::ecs_schedule::access::Access;
use crate::ecs_schedule::context::SystemContext;
use crate::ecs_schedule::error::{ScheduleError, SystemError};
use crate::ecs_schedule::system::{Period, System, SystemFuture, run_guarded};
use crate::ecs_schedule::work::Work;

/// 一个系统加上调度器所需的元数据。
pub struct SystemSpec {
    /// 唯一标签，用于排序提示与错误报告。
    pub label: Cow<'static, str>,
    /// 系统本身。
    pub system: Box<dyn System>,
    /// 生效的世界访问权限（除非被覆盖，否则是系统声明的副本）。
    pub access: Access,
    /// 节奏覆盖。
    pub period: Option<Period>,
}

impl SystemSpec {
    /// 接受任意具体系统，读取它自己的名称与访问权限。
    pub fn new<S: System + 'static>(label: impl Into<Cow<'static, str>>, system: S) -> Self {
        let access = system.access();
        let period = system.period();
        Self {
            label: label.into(),
            system: Box::new(system),
            access,
            period,
        }
    }

    /// 包装一个已装箱的系统。
    pub fn boxed(label: impl Into<Cow<'static, str>>, system: Box<dyn System>) -> Self {
        let access = system.access();
        let period = system.period();
        Self {
            label: label.into(),
            system,
            access,
            period,
        }
    }

    /// 覆盖访问权限声明，例如强制独占。
    pub fn with_access(mut self, access: Access) -> Self {
        self.access = access;
        self
    }

    /// 添加一个声明的读取。
    pub fn reads<T: 'static>(mut self) -> Self {
        self.access = self.access.and_read::<T>();
        self
    }

    /// 添加一个声明的写入。
    pub fn writes<T: 'static>(mut self) -> Self {
        self.access = self.access.and_write::<T>();
        self
    }

    /// 将系统声明为纯读者。
    pub fn read_only(mut self) -> Self {
        self.access = Access::read_only();
        self
    }

    /// 设置节奏。
    pub fn every(mut self, period: Period) -> Self {
        self.period = Some(period);
        self
    }

    /// 毫秒节奏。
    pub fn every_millis(self, millis: u64) -> Self {
        self.every(Period::Time(Duration::from_millis(millis)))
    }

    /// 帧数节奏。
    pub fn every_frames(self, frames: u64) -> Self {
        self.every(Period::Frames(frames))
    }

    /// 系统的标签。
    pub fn name(&self) -> &str {
        &self.label
    }
}

impl std::fmt::Debug for SystemSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemSpec")
            .field("label", &self.label)
            .field("access", &self.access)
            .field("period", &self.period)
            .finish()
    }
}

/// 调度器的一个步骤：顺序流水线或并发组。
#[derive(Debug)]
pub enum Stage {
    /// 系统按顺序一个接一个运行。
    Sequential(Vec<SystemSpec>),
    /// 系统同时运行；所有访问权限声明必须兼容。
    Parallel {
        /// 组的名称。
        name: Cow<'static, str>,
        /// 组的成员。
        systems: Vec<SystemSpec>,
    },
}

impl Stage {
    /// 由任意 spec 迭代器构建顺序阶段。
    pub fn sequential<I: IntoIterator<Item = SystemSpec>>(specs: I) -> Self {
        Self::Sequential(specs.into_iter().collect())
    }

    /// 并发阶段。
    pub fn parallel(name: impl Into<Cow<'static, str>>, specs: Vec<SystemSpec>) -> Self {
        Self::Parallel {
            name: name.into(),
            systems: specs,
        }
    }

    /// 阶段的成员。
    pub fn systems(&self) -> &[SystemSpec] {
        match self {
            Self::Sequential(specs) => specs,
            Self::Parallel { systems, .. } => systems,
        }
    }

    /// 阶段的可变成员。
    pub fn systems_mut(&mut self) -> &mut Vec<SystemSpec> {
        match self {
            Self::Sequential(specs) => specs,
            Self::Parallel { systems, .. } => systems,
        }
    }

    /// 阶段名称：第一个标签，或并行组的名称。
    pub fn name(&self) -> &str {
        match self {
            Self::Sequential(specs) => specs.first().map_or("empty", |spec| &spec.label),
            Self::Parallel { name, .. } => name,
        }
    }

    /// 并发阶段时为 `true`。
    pub fn is_parallel(&self) -> bool {
        matches!(self, Self::Parallel { .. })
    }

    /// 阶段中系统的数量。
    pub fn len(&self) -> usize {
        self.systems().len()
    }

    /// 阶段不运行任何东西时为 `true`。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 有序的系统，由驱动器每帧执行一次。
///
/// 各阶段按顺序运行；在 [`Stage::Parallel`] 内部，成员用 `join_all`
/// 一起轮询，因此等待 LLM 响应的智能体不会拖住其他智能体。
/// 调度器在每个阶段后应用命令队列，这是延迟写入变得可见的唯一时刻。
#[derive(Debug, Default)]
pub struct Schedule {
    stages: Vec<Stage>,
    last_run: HashMap<Cow<'static, str>, tokio::time::Instant>,
    flush_commands: bool,
    isolate: bool,
    failed: HashSet<String>,
}

impl Schedule {
    /// 在每个阶段后刷新命令的空调度器。
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            last_run: HashMap::new(),
            flush_commands: true,
            isolate: false,
            failed: HashSet::new(),
        }
    }

    /// 选择启用对失败系统的遏制：它被记录并在本次运行的剩余部分被跳过，
    /// 而不是中止其他每个系统与智能体。默认仍是快速失败，
    /// 因此坏系统仍会作为错误浮现。
    pub fn isolate_system_errors(&mut self) -> &mut Self {
        self.isolate = true;
        self
    }

    /// 失败并被隔离的系统的标签，按排序顺序。
    pub fn failed_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self.failed.iter().cloned().collect();
        labels.sort();
        labels
    }

    /// 禁用在阶段边界的自动命令刷新（此后由驱动器负责该步骤）。
    pub fn without_command_flush(mut self) -> Self {
        self.flush_commands = false;
        self
    }

    /// 向当前顺序阶段追加一个系统。
    pub fn add(&mut self, spec: SystemSpec) -> &mut Self {
        let label = spec.label.clone();
        match self.stages.last_mut() {
            Some(Stage::Sequential(specs)) => specs.push(spec),
            _ => self.stages.push(Stage::Sequential(vec![spec])),
        }
        let _ = label;
        self
    }

    /// 将多个系统作为一个顺序阶段追加。
    pub fn add_all<I: IntoIterator<Item = SystemSpec>>(&mut self, specs: I) -> &mut Self {
        let stage = Stage::sequential(specs);
        if !stage.is_empty() {
            self.stages.push(stage);
        }
        self
    }

    /// 追加一个并发阶段，拒绝冲突的访问权限声明。
    pub fn parallel<I: IntoIterator<Item = SystemSpec>>(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        specs: I,
    ) -> Result<&mut Self, ScheduleError> {
        let name = name.into();
        let systems: Vec<SystemSpec> = specs.into_iter().collect();
        if systems.is_empty() {
            return Err(ScheduleError::EmptyStage {
                stage: name.to_string(),
            });
        }
        for (left_index, left) in systems.iter().enumerate() {
            for right in systems[left_index + 1..].iter() {
                if left.access.conflicts(&right.access) {
                    return Err(ScheduleError::Conflict {
                        stage: name.to_string(),
                        left: left.label.to_string(),
                        right: right.label.to_string(),
                    });
                }
            }
        }
        self.stages.push(Stage::parallel(name, systems));
        Ok(self)
    }

    /// 在 `label` 正前方插入一个系统。
    pub fn insert_before(
        &mut self,
        label: &str,
        spec: SystemSpec,
    ) -> Result<&mut Self, ScheduleError> {
        let (stage, offset) = self.locate(label)?;
        self.stages[stage].systems_mut().insert(offset, spec);
        Ok(self)
    }

    /// 在 `label` 正后方插入一个系统。
    pub fn insert_after(
        &mut self,
        label: &str,
        spec: SystemSpec,
    ) -> Result<&mut Self, ScheduleError> {
        let (stage, offset) = self.locate(label)?;
        self.stages[stage].systems_mut().insert(offset + 1, spec);
        Ok(self)
    }

    /// 某标签的 `(阶段索引, 阶段内索引)`。
    fn locate(&self, label: &str) -> Result<(usize, usize), ScheduleError> {
        for (stage_index, stage) in self.stages.iter().enumerate() {
            if let Some(offset) = stage
                .systems()
                .iter()
                .position(|spec| spec.label.as_ref() == label)
            {
                return Ok((stage_index, offset));
            }
        }
        Err(ScheduleError::UnknownLabel {
            label: label.to_string(),
        })
    }

    /// 此调度器的各阶段。
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// 没有调度任何东西时为 `true`。
    pub fn is_empty(&self) -> bool {
        self.stages.iter().all(Stage::is_empty)
    }

    /// 已调度系统的数量。
    pub fn len(&self) -> usize {
        self.stages.iter().map(Stage::len).sum()
    }

    /// 按执行顺序的标签。
    pub fn labels(&self) -> Vec<Cow<'static, str>> {
        self.stages
            .iter()
            .flat_map(|stage| stage.systems())
            .map(|spec| spec.label.clone())
            .collect()
    }

    /// 追加 `other` 的每个阶段。
    pub fn merge(&mut self, other: Schedule) -> &mut Self {
        self.stages.extend(other.stages);
        self
    }

    /// 移除每个阶段与节奏时间戳。
    pub fn clear(&mut self) -> &mut Self {
        self.stages.clear();
        self.last_run.clear();
        self
    }

    /// 遇到重复标签、空阶段与并行冲突时失败。
    pub fn validate(&self) -> Result<(), ScheduleError> {
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for (stage_index, stage) in self.stages.iter().enumerate() {
            if stage.is_empty() {
                return Err(ScheduleError::EmptyStage {
                    stage: format!("{stage_index}:{}", stage.name()),
                });
            }
            for spec in stage.systems() {
                if let Some(first) = seen.insert(spec.label.as_ref(), stage_index) {
                    let _ = first;
                    return Err(ScheduleError::DuplicateLabel {
                        label: spec.label.to_string(),
                    });
                }
            }
            if let Stage::Parallel { name, systems } = stage {
                for (left_index, left) in systems.iter().enumerate() {
                    for right in systems[left_index + 1..].iter() {
                        if left.access.conflicts(&right.access) {
                            return Err(ScheduleError::Conflict {
                                stage: name.to_string(),
                                left: left.label.to_string(),
                                right: right.label.to_string(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// 运行每个到期的系统，然后刷新排队的命令。
    pub async fn execute(&mut self, ctx: &SystemContext) -> Result<(), SystemError> {
        let frame = ctx.frame_number();
        tracing::debug!(frame, stages = self.stages.len(), "executing schedule");
        let mut newly_failed: Vec<String> = Vec::new();
        for stage in &mut self.stages {
            match stage {
                Stage::Sequential(specs) => {
                    let flags = cadence(&mut self.last_run, specs, frame);
                    for (spec, is_due) in specs.iter_mut().zip(flags) {
                        if !is_due || self.failed.contains(spec.label.as_ref()) {
                            continue;
                        }
                        if let Err(error) =
                            run_guarded(&spec.label, spec.system.run(ctx.clone())).await
                        {
                            if self.isolate {
                                tracing::warn!(system = %spec.label, %error, "system failed; isolated");
                                newly_failed.push(spec.label.to_string());
                                release_work(ctx, spec.system.name()).await;
                            } else {
                                return Err(error);
                            }
                        }
                        if ctx.is_stopping() {
                            self.failed.extend(newly_failed);
                            return Ok(());
                        }
                    }
                }
                Stage::Parallel { systems, .. } => {
                    // 到期检查先运行，使下面每个系统借用都是每元素一个
                    // 不相交的 `&mut`，这正是让各 future 能在一个
                    // `join_all` 中共存的原因。
                    let flags = cadence(&mut self.last_run, systems, frame);
                    let mut pending: Vec<SystemFuture<'_>> = Vec::new();
                    let mut labels: Vec<Cow<'static, str>> = Vec::new();
                    let mut owners: Vec<String> = Vec::new();
                    for (spec, is_due) in systems.iter_mut().zip(flags) {
                        if is_due && !self.failed.contains(spec.label.as_ref()) {
                            labels.push(spec.label.clone());
                            owners.push(spec.system.name().to_string());
                            pending.push(Box::pin(run_guarded(
                                &spec.label,
                                spec.system.run(ctx.clone()),
                            )));
                        }
                    }
                    for ((label, owner), result) in labels
                        .into_iter()
                        .zip(owners)
                        .zip(futures::future::join_all(pending).await)
                    {
                        if let Err(error) = result {
                            if self.isolate {
                                tracing::warn!(system = %label, %error, "system failed; isolated");
                                newly_failed.push(label.to_string());
                                release_work(ctx, &owner).await;
                            } else {
                                return Err(error);
                            }
                        }
                    }
                }
            }
            if self.flush_commands {
                ctx.apply_commands().await;
            }
            if ctx.is_stopping() {
                self.failed.extend(newly_failed);
                return Ok(());
            }
        }
        self.failed.extend(newly_failed);
        Ok(())
    }
}

/// 释放刚被隔离的系统所认领的工作。被隔离的系统在本次运行的剩余部分
/// 被跳过，因此它再也无法落定自己的标签；若继续认领，它们会让驱动器
/// 保持唤醒直到帧预算耗尽。所有者是系统的 `name()`，即生产者声明的
/// 那个名字，它可能与注册标签不同。
async fn release_work(ctx: &SystemContext, owner: &str) {
    ctx.with_world_mut(|world| Work::ensure(world).release_owner(owner))
        .await;
}

/// 标记阶段中在本帧到期的每个成员，并为运行的那些打上时间戳。
/// 直接接收该映射，使调用者保持其阶段借用。
fn cadence(
    last_run: &mut HashMap<Cow<'static, str>, tokio::time::Instant>,
    specs: &[SystemSpec],
    frame: u64,
) -> Vec<bool> {
    specs
        .iter()
        .map(|spec| due(last_run, &spec.label, spec.period, frame))
        .collect()
}

/// 单个系统的节奏门控。
fn due(
    last_run: &mut HashMap<Cow<'static, str>, tokio::time::Instant>,
    label: &str,
    period: Option<Period>,
    frame: u64,
) -> bool {
    let Some(period) = period else {
        return true;
    };
    match period {
        Period::Frames(every) => every <= 1 || frame.is_multiple_of(every),
        Period::Time(duration) => {
            let now = tokio::time::Instant::now();
            let entry = lookup(last_run, label);
            let run = entry.is_none_or(|previous| now.duration_since(previous) >= duration);
            if run {
                let key = last_run_key(last_run, label);
                last_run.insert(key, now);
            }
            run
        }
    }
}

fn lookup(
    last_run: &HashMap<Cow<'static, str>, tokio::time::Instant>,
    label: &str,
) -> Option<tokio::time::Instant> {
    last_run.get(label).copied()
}

fn last_run_key(
    last_run: &HashMap<Cow<'static, str>, tokio::time::Instant>,
    label: &str,
) -> Cow<'static, str> {
    last_run
        .keys()
        .find(|key| key.as_ref() == label)
        .cloned()
        .unwrap_or_else(|| Cow::Owned(label.to_string()))
}

/// 将一组固定系统运行一次的调度器，便于用于测试与一次性流水线。
pub fn one_shot<I: IntoIterator<Item = SystemSpec>>(specs: I) -> Schedule {
    let mut schedule = Schedule::new();
    schedule.add_all(specs);
    schedule
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs_schedule::context::WorldHandle;
    use crate::ecs_schedule::system::system;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct A;
    struct B;

    fn counting(name: &'static str, counter: Arc<AtomicU64>, access: Access) -> SystemSpec {
        SystemSpec::new(
            name,
            system(name, move |_ctx: SystemContext| {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
            })
            .with_access(access),
        )
    }

    #[tokio::test]
    async fn sequential_stages_keep_declaration_order() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut schedule = Schedule::new();
        for name in ["first", "second", "third"] {
            let order = order.clone();
            schedule.add(SystemSpec::new(
                name,
                system(name, move |_ctx: SystemContext| {
                    let order = order.clone();
                    async move {
                        order.lock().unwrap().push(name);
                        Ok(())
                    }
                }),
            ));
        }
        let ctx = SystemContext::new(WorldHandle::empty());
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(*order.lock().unwrap(), vec!["first", "second", "third"]);
    }

    #[tokio::test]
    async fn parallel_members_wait_for_each_other() {
        // 只有当两个系统被并发轮询时，2 的屏障才会解除。
        let gate = Arc::new(tokio::sync::Barrier::new(2));
        let mut schedule = Schedule::new();
        let specs = ["left", "right"].map(|name| {
            let gate = gate.clone();
            SystemSpec::new(
                name,
                system(name, move |_ctx: SystemContext| {
                    let gate = gate.clone();
                    async move {
                        gate.wait().await;
                        Ok(())
                    }
                })
                .reads::<A>(),
            )
        });
        schedule.parallel("readers", specs).unwrap();
        let ctx = SystemContext::new(WorldHandle::empty());
        let result = tokio::time::timeout(Duration::from_secs(2), schedule.execute(&ctx)).await;
        result.expect("read-only systems run concurrently").unwrap();

        // 同一对系统，声明为独占，必须在一开始就被拒绝
        let mut strict = Schedule::new();
        let error = strict
            .parallel(
                "writers",
                ["x", "y"].map(|name| {
                    SystemSpec::new(name, system(name, |_c: SystemContext| async { Ok(()) }))
                }),
            )
            .unwrap_err();
        assert!(matches!(error, ScheduleError::Conflict { .. }));
    }

    #[tokio::test]
    async fn parallel_rejects_conflicting_access() {
        let counter = Arc::new(AtomicU64::new(0));
        let mut schedule = Schedule::new();
        let error = schedule
            .parallel(
                "bad",
                [
                    counting("writer", counter.clone(), Access::write::<A>()),
                    counting("reader", counter.clone(), Access::read::<A>()),
                ],
            )
            .unwrap_err();
        assert!(matches!(error, ScheduleError::Conflict { .. }));
        assert!(schedule.is_empty());
    }

    #[tokio::test]
    async fn frame_and_time_cadence() {
        let counter = Arc::new(AtomicU64::new(0));
        let mut schedule = Schedule::new();
        schedule.add(counting("half", counter.clone(), Access::empty()).every_frames(2));
        schedule.add(counting("slow", counter.clone(), Access::empty()).every_millis(40));

        let ctx = SystemContext::new(WorldHandle::empty());
        // 帧 0：两者都到期
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 2);
        // 帧 1：只有 2 帧节奏被跳过，计时器尚未到期
        ctx.advance_frame();
        counter.store(0, Ordering::Relaxed);
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        // 时间窗口过后两者再次到期：帧 2 匹配 2 帧节奏，
        // 且 60ms 超过 40ms 周期
        tokio::time::sleep(Duration::from_millis(60)).await;
        ctx.advance_frame();
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 2);
        // 帧 3 跳过 2 帧系统，计时器尚未到期
        counter.store(0, Ordering::Relaxed);
        ctx.advance_frame();
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn errors_stop_the_schedule() {
        let seen = Arc::new(AtomicU64::new(0));
        let mut schedule = Schedule::new();
        schedule.add(SystemSpec::new(
            "boom",
            system("boom", |_ctx: SystemContext| async move {
                Err(SystemError::new("boom", "no api key"))
            }),
        ));
        let seen_late = seen.clone();
        schedule.add(SystemSpec::new(
            "late",
            system("late", move |_ctx: SystemContext| {
                let seen_late = seen_late.clone();
                async move {
                    seen_late.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
            }),
        ));
        let ctx = SystemContext::new(WorldHandle::empty());
        let error = schedule.execute(&ctx).await.unwrap_err();
        assert_eq!(error.reason, "no api key");
        assert_eq!(seen.load(Ordering::Relaxed), 0, "later systems are skipped");
    }

    #[tokio::test]
    async fn isolation_contains_a_failing_parallel_member() {
        let healthy = Arc::new(AtomicU64::new(0));
        let mut schedule = Schedule::new();
        schedule
            .parallel(
                "workers",
                [
                    SystemSpec::new(
                        "boom",
                        system("boom", |_ctx: SystemContext| async { panic!("exploded") }),
                    )
                    .with_access(Access::read::<A>()),
                    counting("healthy", healthy.clone(), Access::read::<A>()),
                ],
            )
            .unwrap();
        schedule.isolate_system_errors();
        let ctx = SystemContext::new(WorldHandle::empty());

        schedule
            .execute(&ctx)
            .await
            .expect("a bad member must not abort the group");
        assert_eq!(
            healthy.load(Ordering::Relaxed),
            1,
            "the healthy member still ran"
        );
        assert_eq!(schedule.failed_labels(), vec!["boom".to_string()]);

        // 失败的成员在后续帧被跳过，健康的成员继续运行。
        ctx.advance_frame();
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(healthy.load(Ordering::Relaxed), 2);
        assert_eq!(schedule.failed_labels(), vec!["boom".to_string()]);
    }

    #[tokio::test]
    async fn a_failing_parallel_member_still_aborts_by_default() {
        let healthy = Arc::new(AtomicU64::new(0));
        let mut schedule = Schedule::new();
        schedule
            .parallel(
                "workers",
                [
                    SystemSpec::new(
                        "boom",
                        system("boom", |_ctx: SystemContext| async { panic!("exploded") }),
                    )
                    .with_access(Access::read::<A>()),
                    counting("healthy", healthy.clone(), Access::read::<A>()),
                ],
            )
            .unwrap();
        let ctx = SystemContext::new(WorldHandle::empty());
        assert!(
            schedule.execute(&ctx).await.is_err(),
            "fail-fast remains the default"
        );
        assert!(schedule.failed_labels().is_empty());
    }

    #[tokio::test]
    async fn isolation_releases_work_under_the_system_name_not_the_label() {
        let mut schedule = Schedule::new();
        schedule.add(SystemSpec::new(
            "route",
            system("route-task", |ctx: SystemContext| async move {
                ctx.with_world_mut(|world| {
                    Work::ensure(world).set_owned("agents", true, "route-task")
                })
                .await;
                Err(SystemError::new("route-task", "no api key"))
            }),
        ));
        schedule.isolate_system_errors();
        let ctx = SystemContext::new(WorldHandle::empty());
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(schedule.failed_labels(), vec!["route".to_string()]);
        let busy = ctx
            .with_world(|world| {
                world
                    .resource::<Work>()
                    .map(|work| work.is_busy())
                    .unwrap_or(false)
            })
            .await;
        assert!(
            !busy,
            "work is released under the system name, not the registration label"
        );
    }

    #[tokio::test]
    async fn isolation_releases_a_parallel_members_work_under_its_name() {
        let mut schedule = Schedule::new();
        schedule
            .parallel(
                "workers",
                [
                    SystemSpec::new(
                        "route",
                        system("route-task", |ctx: SystemContext| async move {
                            ctx.with_world_mut(|world| {
                                Work::ensure(world).set_owned("agents", true, "route-task")
                            })
                            .await;
                            Err(SystemError::new("route-task", "boom"))
                        }),
                    )
                    .with_access(Access::read::<A>()),
                    counting("healthy", Arc::new(AtomicU64::new(0)), Access::read::<A>()),
                ],
            )
            .unwrap();
        schedule.isolate_system_errors();
        let ctx = SystemContext::new(WorldHandle::empty());
        schedule.execute(&ctx).await.unwrap();
        let busy = ctx
            .with_world(|world| {
                world
                    .resource::<Work>()
                    .map(|work| work.is_busy())
                    .unwrap_or(false)
            })
            .await;
        assert!(!busy, "the failing parallel member's work is released");
    }

    #[tokio::test]
    async fn commands_flush_between_stages() {
        let mut schedule = Schedule::new();
        schedule.add(SystemSpec::new(
            "spawn",
            system("spawn", |ctx: SystemContext| async move {
                ctx.enqueue(|world| {
                    world.spawn();
                });
                Ok(())
            })
            .read_only(),
        ));
        let ctx = SystemContext::new(WorldHandle::empty());
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(
            ctx.with_world(|world| world.len()).await,
            1,
            "queued command was applied at the boundary"
        );
    }

    #[test]
    fn ordering_helpers_and_validation() {
        let counter = Arc::new(AtomicU64::new(0));
        let mut schedule = Schedule::new();
        schedule.add(counting("a", counter.clone(), Access::empty()));
        schedule.add(counting("b", counter.clone(), Access::empty()));
        schedule
            .insert_after("a", counting("a1", counter.clone(), Access::empty()))
            .unwrap();
        schedule
            .insert_before("b", counting("b0", counter.clone(), Access::empty()))
            .unwrap();
        assert_eq!(
            schedule.labels(),
            ["a", "a1", "b0", "b"]
                .iter()
                .map(|name| (*name).to_string().into())
                .collect::<Vec<Cow<'static, str>>>()
        );
        schedule.validate().unwrap();

        let duplicate = SystemSpec::new(
            "a",
            crate::ecs_schedule::system::system("a", |_c: SystemContext| async { Ok(()) }),
        );
        schedule.add(duplicate);
        assert!(matches!(
            schedule.validate(),
            Err(ScheduleError::DuplicateLabel { .. })
        ));
        assert!(matches!(
            schedule.insert_before("missing", counting("x", counter.clone(), Access::empty())),
            Err(ScheduleError::UnknownLabel { .. })
        ));
    }

    #[tokio::test]
    async fn one_shot_runs_each_system_once() {
        let counter = Arc::new(AtomicU64::new(0));
        let mut schedule = one_shot([
            counting("a", counter.clone(), Access::read::<A>()),
            counting("b", counter.clone(), Access::write::<B>()),
        ]);
        let ctx = SystemContext::new(WorldHandle::empty());
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }
}
