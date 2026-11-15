//! 一个 Xihe 世界的可观测性。
//!
//! [`Observer`] 是一个世界资源。系统与驱动器把事件记入其中；
//! 这里没有任何东西向外触及网络。快照是纯数据，因此测试可以
//! 对它断言，demo 可以打印它。

use std::fmt::{self, Display, Formatter};
use std::time::{Duration, Instant};

use xihe_core::World;

/// 发生了什么，按其发生的顺序。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventKind {
    /// 一个调度器帧开始。
    Frame {
        /// 帧编号，第一个被录制的帧从 1 开始。
        number: u64,
    },
    /// 一个系统完成了一次激活。
    System {
        /// 系统名。
        name: String,
        /// 它是否返回了错误。
        failed: bool,
        /// 此次激活耗时多久，以微秒计。
        elapsed_us: u64,
    },
    /// 一次模型调用完成。
    Model {
        /// 提供方标识符。
        provider: String,
        /// 模型名。
        model: String,
        /// 输入 token，已知时。
        input_tokens: u64,
        /// 输出 token，已知时。
        output_tokens: u64,
        /// 调用是否失败。
        failed: bool,
    },
    /// 一次工具调用完成。
    Tool {
        /// 工具名。
        name: String,
        /// 工具是否报告成功。
        ok: bool,
    },
    /// 一个智能体改变了状态。
    Agent {
        /// 智能体名。
        name: String,
        /// 状态名，与 [`xihe_agent::AgentStatus::as_str`] 一致。
        status: String,
    },
    /// 一条自由格式的备注。
    Note {
        /// 谁写的。
        source: String,
        /// 备注内容。
        text: String,
    },
}

impl Display for EventKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Frame { number } => write!(f, "frame {number}"),
            Self::System {
                name,
                failed,
                elapsed_us,
            } => write!(
                f,
                "system {name} {} in {elapsed_us}us",
                if *failed { "failed" } else { "ran" }
            ),
            Self::Model {
                provider,
                model,
                input_tokens,
                output_tokens,
                failed,
            } => write!(
                f,
                "model {provider}/{model} in={input_tokens} out={output_tokens} {}",
                if *failed { "failed" } else { "ok" }
            ),
            Self::Tool { name, ok } => {
                write!(f, "tool {name} {}", if *ok { "ok" } else { "failed" })
            }
            Self::Agent { name, status } => write!(f, "agent {name} -> {status}"),
            Self::Note { source, text } => write!(f, "{source}: {text}"),
        }
    }
}

/// 一个被录制的事件。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    /// 单调序列号，从 1 开始。
    pub seq: u64,
    /// 发生了什么。
    pub kind: EventKind,
}

/// 从事件日志派生的计数器。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// 录制的帧数。
    pub frames: u64,
    /// 系统激活次数。
    pub system_runs: u64,
    /// 失败的系统激活次数。
    pub system_failures: u64,
    /// 模型调用次数。
    pub model_calls: u64,
    /// 失败的模型调用次数。
    pub model_failures: u64,
    /// 录制的输入 token。
    pub input_tokens: u64,
    /// 录制的输出 token。
    pub output_tokens: u64,
    /// 录制的工具调用次数。
    pub tool_calls: u64,
    /// 失败的工具调用次数。
    pub tool_failures: u64,
    /// 智能体状态变更次数。
    pub agent_events: u64,
}

impl Snapshot {
    /// 跨每次模型调用录制的 token。
    pub fn tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

impl Display for Snapshot {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "frames={} systems={}/{} models={}/{} tokens={} tools={}/{}",
            self.frames,
            self.system_runs,
            self.system_failures,
            self.model_calls,
            self.model_failures,
            self.tokens(),
            self.tool_calls,
            self.tool_failures
        )
    }
}

/// 事件日志，作为世界资源存储。
///
/// 保留窗口可用 [`Observer::with_retention`] 加以限定，使一个长期运行的
/// 世界不会无限制地累积事件。[`Snapshot`] 计数器是累计的，在裁剪后依然存续。
#[derive(Clone, Debug, Default)]
pub struct Observer {
    events: Vec<Event>,
    next: u64,
    started: Option<Instant>,
    retention: usize,
    totals: Snapshot,
}

impl Observer {
    /// 一个空日志。
    pub fn new() -> Self {
        Self::default()
    }

    /// 至多保留 `limit` 个最近事件；`0`（默认）表示无限制。
    ///
    /// 裁剪从 [`Observer::events`] 丢弃最旧的事件，但保持累计的
    /// [`Snapshot`] 计数器完好，因此一个有界日志仍报告整个生命周期的总量。
    pub fn with_retention(mut self, limit: usize) -> Self {
        self.retention = limit;
        self.trim();
        self
    }

    /// 保留上限，`0` 表示无限制。
    pub fn retention(&self) -> usize {
        self.retention
    }

    /// 录制 `kind` 并返回其序列号。
    pub fn record(&mut self, kind: EventKind) -> u64 {
        if self.started.is_none() {
            self.started = Some(Instant::now());
        }
        self.next += 1;
        let seq = self.next;
        accumulate(&mut self.totals, &kind);
        self.events.push(Event { seq, kind });
        self.trim();
        seq
    }

    /// 丢弃超出保留上限的最旧事件。
    fn trim(&mut self) {
        if self.retention > 0 && self.events.len() > self.retention {
            let excess = self.events.len() - self.retention;
            self.events.drain(..excess);
        }
    }

    /// 录制一个帧边界。
    pub fn frame(&mut self, number: u64) -> u64 {
        self.record(EventKind::Frame { number })
    }

    /// 录制一次系统激活。
    pub fn system(&mut self, name: impl Into<String>, failed: bool, elapsed: Duration) -> u64 {
        self.record(EventKind::System {
            name: name.into(),
            failed,
            elapsed_us: u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
        })
    }

    /// 录制一次模型调用。
    pub fn model(
        &mut self,
        provider: impl Into<String>,
        model: impl Into<String>,
        input_tokens: u64,
        output_tokens: u64,
        failed: bool,
    ) -> u64 {
        self.record(EventKind::Model {
            provider: provider.into(),
            model: model.into(),
            input_tokens,
            output_tokens,
            failed,
        })
    }

    /// 录制一次工具调用。
    pub fn tool(&mut self, name: impl Into<String>, ok: bool) -> u64 {
        self.record(EventKind::Tool {
            name: name.into(),
            ok,
        })
    }

    /// 录制一次智能体状态变更。
    pub fn agent(&mut self, name: impl Into<String>, status: impl Into<String>) -> u64 {
        self.record(EventKind::Agent {
            name: name.into(),
            status: status.into(),
        })
    }

    /// 录制一条备注。
    pub fn note(&mut self, source: impl Into<String>, text: impl Into<String>) -> u64 {
        self.record(EventKind::Note {
            source: source.into(),
            text: text.into(),
        })
    }

    /// 保留的事件，最旧的在前。
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// 日志中保留了多少事件。
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// 是否尚未录制任何内容。
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// 自第一个事件以来的时间。
    pub fn elapsed(&self) -> Duration {
        self.started
            .map_or(Duration::ZERO, |started| started.elapsed())
    }

    /// 对曾经录制的每个事件的累计计数器，包括此后已从保留窗口
    /// 裁剪掉的任何事件。
    pub fn snapshot(&self) -> Snapshot {
        self.totals
    }

    /// 日志按行呈现，供 demo 或失败的测试使用。
    pub fn render(&self) -> String {
        self.events
            .iter()
            .map(|event| format!("{:04} {}", event.seq, event.kind))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 含快照与事件行的 JSON 对象。
    pub fn to_json(&self) -> serde_json::Value {
        let snapshot = self.snapshot();
        serde_json::json!({
            "frames": snapshot.frames,
            "system_runs": snapshot.system_runs,
            "system_failures": snapshot.system_failures,
            "model_calls": snapshot.model_calls,
            "model_failures": snapshot.model_failures,
            "input_tokens": snapshot.input_tokens,
            "output_tokens": snapshot.output_tokens,
            "tool_calls": snapshot.tool_calls,
            "tool_failures": snapshot.tool_failures,
            "events": self.events.iter().map(|event| event.kind.to_string()).collect::<Vec<_>>(),
        })
    }

    /// 从世界中读出观察器，缺失时插入一个空的。
    pub fn ensure(world: &mut World) -> &mut Self {
        if !world.has_resource::<Self>() {
            world.insert_resource(Self::new());
        }
        world.resource_mut::<Self>().expect("just inserted")
    }
}

/// 把一个事件折入累计计数器。
fn accumulate(snapshot: &mut Snapshot, kind: &EventKind) {
    match kind {
        EventKind::Frame { .. } => snapshot.frames += 1,
        EventKind::System { failed, .. } => {
            snapshot.system_runs += 1;
            if *failed {
                snapshot.system_failures += 1;
            }
        }
        EventKind::Model {
            input_tokens,
            output_tokens,
            failed,
            ..
        } => {
            snapshot.model_calls += 1;
            snapshot.input_tokens += input_tokens;
            snapshot.output_tokens += output_tokens;
            if *failed {
                snapshot.model_failures += 1;
            }
        }
        EventKind::Tool { ok, .. } => {
            snapshot.tool_calls += 1;
            if !ok {
                snapshot.tool_failures += 1;
            }
        }
        EventKind::Agent { .. } => snapshot.agent_events += 1,
        EventKind::Note { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_snapshot_counts_each_kind() {
        let mut observer = Observer::new();
        observer.frame(1);
        observer.system("react", false, Duration::from_micros(40));
        observer.system("react", true, Duration::from_micros(5));
        observer.model("mock", "m", 10, 4, false);
        observer.tool("echo", true);
        observer.tool("deploy", false);
        observer.agent("scribe", "finished");
        observer.note("demo", "hello");
        let snapshot = observer.snapshot();
        assert_eq!(snapshot.frames, 1);
        assert_eq!(snapshot.system_runs, 2);
        assert_eq!(snapshot.system_failures, 1);
        assert_eq!(snapshot.tokens(), 14);
        assert_eq!(snapshot.tool_failures, 1);
        assert_eq!(snapshot.agent_events, 1);
        assert_eq!(observer.len(), 8);
        assert!(observer.render().contains("tool deploy failed"));
    }

    #[test]
    fn json_lists_the_events_in_order() {
        let mut observer = Observer::new();
        observer.frame(1);
        observer.note("demo", "started");
        let json = observer.to_json();
        assert_eq!(json["frames"], 1);
        assert_eq!(json["events"][0], "frame 1");
        assert_eq!(json["events"][1], "demo: started");
    }

    #[test]
    fn ensure_inserts_once() {
        let mut world = World::new();
        Observer::ensure(&mut world).frame(1);
        Observer::ensure(&mut world).frame(2);
        assert_eq!(world.resource::<Observer>().unwrap().len(), 2);
    }

    #[test]
    fn display_names_the_outcome() {
        assert_eq!(
            EventKind::Model {
                provider: "mock".into(),
                model: "m".into(),
                input_tokens: 1,
                output_tokens: 2,
                failed: false,
            }
            .to_string(),
            "model mock/m in=1 out=2 ok"
        );
    }

    #[test]
    fn retention_bounds_the_log_but_keeps_cumulative_counts() {
        let mut observer = Observer::new().with_retention(3);
        for _ in 0..10 {
            observer.model("mock", "m", 1, 1, false);
        }
        assert_eq!(observer.len(), 3, "only the most recent events are retained");
        let snapshot = observer.snapshot();
        assert_eq!(
            snapshot.model_calls, 10,
            "cumulative counters survive trimming: {snapshot}"
        );
        assert_eq!(snapshot.tokens(), 20);
        let seqs: Vec<u64> = observer.events().iter().map(|event| event.seq).collect();
        assert_eq!(seqs, vec![8, 9, 10], "sequence numbers stay monotonic");
    }
}
