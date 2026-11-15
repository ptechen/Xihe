//! Schedules: ordered stages of systems, with concurrency inside a stage.

use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Duration;

use crate::access::Access;
use crate::context::SystemContext;
use crate::error::{ScheduleError, SystemError};
use crate::system::{Period, System, SystemFuture, run_guarded};

/// A system plus the metadata the scheduler needs.
pub struct SystemSpec {
    /// Unique label, used for ordering hints and error reports.
    pub label: Cow<'static, str>,
    /// The system itself.
    pub system: Box<dyn System>,
    /// Effective world access (a copy of the system's declaration unless
    /// overridden).
    pub access: Access,
    /// Cadence override.
    pub period: Option<Period>,
}

impl SystemSpec {
    /// Takes any concrete system, reading its own name and access.
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

    /// Wraps an already boxed system.
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

    /// Overrides the access declaration, e.g. to force exclusivity.
    pub fn with_access(mut self, access: Access) -> Self {
        self.access = access;
        self
    }

    /// Adds a declared read.
    pub fn reads<T: 'static>(mut self) -> Self {
        self.access = self.access.and_read::<T>();
        self
    }

    /// Adds a declared write.
    pub fn writes<T: 'static>(mut self) -> Self {
        self.access = self.access.and_write::<T>();
        self
    }

    /// Declares the system as a pure reader.
    pub fn read_only(mut self) -> Self {
        self.access = Access::read_only();
        self
    }

    /// Sets the cadence.
    pub fn every(mut self, period: Period) -> Self {
        self.period = Some(period);
        self
    }

    /// Millisecond cadence.
    pub fn every_millis(self, millis: u64) -> Self {
        self.every(Period::Time(Duration::from_millis(millis)))
    }

    /// Frame-count cadence.
    pub fn every_frames(self, frames: u64) -> Self {
        self.every(Period::Frames(frames))
    }

    /// Label of the system.
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

/// One step of a schedule: a sequential pipeline or a concurrent group.
#[derive(Debug)]
pub enum Stage {
    /// Systems run one after another, in order.
    Sequential(Vec<SystemSpec>),
    /// Systems run at the same time; all access declarations must be compatible.
    Parallel {
        /// Name of the group.
        name: Cow<'static, str>,
        /// Members of the group.
        systems: Vec<SystemSpec>,
    },
}

impl Stage {
    /// Sequential stage from any iterator of specs.
    pub fn sequential<I: IntoIterator<Item = SystemSpec>>(specs: I) -> Self {
        Self::Sequential(specs.into_iter().collect())
    }

    /// Concurrent stage.
    pub fn parallel(name: impl Into<Cow<'static, str>>, specs: Vec<SystemSpec>) -> Self {
        Self::Parallel {
            name: name.into(),
            systems: specs,
        }
    }

    /// Members of the stage.
    pub fn systems(&self) -> &[SystemSpec] {
        match self {
            Self::Sequential(specs) => specs,
            Self::Parallel { systems, .. } => systems,
        }
    }

    /// Mutable members of the stage.
    pub fn systems_mut(&mut self) -> &mut Vec<SystemSpec> {
        match self {
            Self::Sequential(specs) => specs,
            Self::Parallel { systems, .. } => systems,
        }
    }

    /// Stage name: the first label, or the parallel group name.
    pub fn name(&self) -> &str {
        match self {
            Self::Sequential(specs) => specs.first().map_or("empty", |spec| &spec.label),
            Self::Parallel { name, .. } => name,
        }
    }

    /// `true` for a concurrent stage.
    pub fn is_parallel(&self) -> bool {
        matches!(self, Self::Parallel { .. })
    }

    /// Number of systems in the stage.
    pub fn len(&self) -> usize {
        self.systems().len()
    }

    /// `true` when the stage runs nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Ordered systems, executed once per frame by the driver.
///
/// Stages run in sequence; inside a [`Stage::Parallel`] the members are polled
/// together with `join_all`, so an agent waiting on an LLM response does not
/// stall the other agents. The scheduler applies the command queue after each
/// stage, which is the only point where deferred writes become visible.
#[derive(Debug, Default)]
pub struct Schedule {
    stages: Vec<Stage>,
    last_run: HashMap<Cow<'static, str>, tokio::time::Instant>,
    flush_commands: bool,
}

impl Schedule {
    /// Empty schedule that flushes commands after every stage.
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            last_run: HashMap::new(),
            flush_commands: true,
        }
    }

    /// Disables the automatic command flush at stage boundaries (the driver then
    /// owns that step).
    pub fn without_command_flush(mut self) -> Self {
        self.flush_commands = false;
        self
    }

    /// Appends a system to the current sequential stage.
    pub fn add(&mut self, spec: SystemSpec) -> &mut Self {
        let label = spec.label.clone();
        match self.stages.last_mut() {
            Some(Stage::Sequential(specs)) => specs.push(spec),
            _ => self.stages.push(Stage::Sequential(vec![spec])),
        }
        let _ = label;
        self
    }

    /// Appends several systems as one sequential stage.
    pub fn add_all<I: IntoIterator<Item = SystemSpec>>(&mut self, specs: I) -> &mut Self {
        let stage = Stage::sequential(specs);
        if !stage.is_empty() {
            self.stages.push(stage);
        }
        self
    }

    /// Appends a concurrent stage, rejecting conflicting access declarations.
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

    /// Inserts a system right before `label`.
    pub fn insert_before(
        &mut self,
        label: &str,
        spec: SystemSpec,
    ) -> Result<&mut Self, ScheduleError> {
        let (stage, offset) = self.locate(label)?;
        self.stages[stage].systems_mut().insert(offset, spec);
        Ok(self)
    }

    /// Inserts a system right after `label`.
    pub fn insert_after(
        &mut self,
        label: &str,
        spec: SystemSpec,
    ) -> Result<&mut Self, ScheduleError> {
        let (stage, offset) = self.locate(label)?;
        self.stages[stage].systems_mut().insert(offset + 1, spec);
        Ok(self)
    }

    /// `(stage index, index inside the stage)` of a label.
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

    /// Stages of this schedule.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// `true` when nothing is scheduled.
    pub fn is_empty(&self) -> bool {
        self.stages.iter().all(Stage::is_empty)
    }

    /// Number of scheduled systems.
    pub fn len(&self) -> usize {
        self.stages.iter().map(Stage::len).sum()
    }

    /// Labels in execution order.
    pub fn labels(&self) -> Vec<Cow<'static, str>> {
        self.stages
            .iter()
            .flat_map(|stage| stage.systems())
            .map(|spec| spec.label.clone())
            .collect()
    }

    /// Appends every stage of `other`.
    pub fn merge(&mut self, other: Schedule) -> &mut Self {
        self.stages.extend(other.stages);
        self
    }

    /// Removes every stage and cadence timestamp.
    pub fn clear(&mut self) -> &mut Self {
        self.stages.clear();
        self.last_run.clear();
        self
    }

    /// Fails on duplicate labels, empty stages and parallel conflicts.
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

    /// Runs every due system, then flushes queued commands.
    pub async fn execute(&mut self, ctx: &SystemContext) -> Result<(), SystemError> {
        let frame = ctx.frame_number();
        tracing::debug!(frame, stages = self.stages.len(), "executing schedule");
        for stage in &mut self.stages {
            match stage {
                Stage::Sequential(specs) => {
                    let flags = cadence(&mut self.last_run, specs, frame);
                    for (spec, is_due) in specs.iter_mut().zip(flags) {
                        if !is_due {
                            continue;
                        }
                        run_guarded(&spec.label, spec.system.run(ctx.clone())).await?;
                        if ctx.is_stopping() {
                            return Ok(());
                        }
                    }
                }
                Stage::Parallel { systems, .. } => {
                    // The due check runs first so every system borrow below is a
                    // single disjoint `&mut` per element, which is what lets the
                    // futures coexist in one `join_all`.
                    let flags = cadence(&mut self.last_run, systems, frame);
                    let mut pending: Vec<SystemFuture<'_>> = Vec::new();
                    for (spec, is_due) in systems.iter_mut().zip(flags) {
                        if is_due {
                            pending.push(Box::pin(run_guarded(
                                &spec.label,
                                spec.system.run(ctx.clone()),
                            )));
                        }
                    }
                    for result in futures::future::join_all(pending).await {
                        result?;
                    }
                }
            }
            if self.flush_commands {
                ctx.apply_commands().await;
            }
            if ctx.is_stopping() {
                return Ok(());
            }
        }
        Ok(())
    }
}

/// Flags every member of a stage that is due on this frame, stamping the ones
/// that run. Takes the map directly so the caller keeps its stage borrow.
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

/// Cadence gate for one system.
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

fn lookup(last_run: &HashMap<Cow<'static, str>, tokio::time::Instant>, label: &str) -> Option<tokio::time::Instant> {
    last_run.get(label).copied()
}

fn last_run_key(last_run: &HashMap<Cow<'static, str>, tokio::time::Instant>, label: &str) -> Cow<'static, str> {
    last_run
        .keys()
        .find(|key| key.as_ref() == label)
        .cloned()
        .unwrap_or_else(|| Cow::Owned(label.to_string()))
}

/// A schedule that runs a fixed set of systems once, useful in tests and in
/// one-shot pipelines.
pub fn one_shot<I: IntoIterator<Item = SystemSpec>>(specs: I) -> Schedule {
    let mut schedule = Schedule::new();
    schedule.add_all(specs);
    schedule
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::WorldHandle;
    use crate::system::system;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

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
        // A barrier of 2 only resolves when both systems are polled concurrently.
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

        // the same pair, declared exclusive, must be refused up front
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
        schedule.add(counting(
            "half",
            counter.clone(),
            Access::empty(),
        ).every_frames(2));
        schedule.add(counting("slow", counter.clone(), Access::empty()).every_millis(40));

        let ctx = SystemContext::new(WorldHandle::empty());
        // frame 0: both due
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 2);
        // frame 1: only the 2-frame cadence is skipped, the timer is not due yet
        ctx.advance_frame();
        counter.store(0, Ordering::Relaxed);
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        // after the time window both are due again: frame 2 matches the 2-frame
        // cadence and 60ms exceeds the 40ms period
        tokio::time::sleep(Duration::from_millis(60)).await;
        ctx.advance_frame();
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 2);
        // frame 3 skips the 2-frame system, the timer is not due yet
        counter.store(0, Ordering::Relaxed);
        ctx.advance_frame();
        schedule.execute(&ctx).await.unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn errors_stop_the_schedule() {
        let seen = Arc::new(AtomicU64::new(0));
        let mut schedule = Schedule::new();
        schedule.add(SystemSpec::new("boom", system("boom", |_ctx: SystemContext| async move {
            Err(SystemError::new("boom", "no api key"))
        })));
        let seen_late = seen.clone();
        schedule.add(SystemSpec::new("late", system("late", move |_ctx: SystemContext| {
            let seen_late = seen_late.clone();
            async move {
                seen_late.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        })));
        let ctx = SystemContext::new(WorldHandle::empty());
        let error = schedule.execute(&ctx).await.unwrap_err();
        assert_eq!(error.reason, "no api key");
        assert_eq!(seen.load(Ordering::Relaxed), 0, "later systems are skipped");
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

        let duplicate = SystemSpec::new("a", crate::system::system("a", |_c: SystemContext| async { Ok(()) }));
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
