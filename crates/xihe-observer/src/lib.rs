//! Observability for a Xihe world.
//!
//! [`Observer`] is a world resource. Systems and the driver record events into
//! it; nothing here reaches out to a network. A snapshot is plain data, so a
//! test can assert on it and a demo can print it.

use std::fmt::{self, Display, Formatter};
use std::time::{Duration, Instant};

use xihe_core::World;

/// What happened, in the order it happened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventKind {
    /// A schedule frame started.
    Frame {
        /// Frame number, starting at 1 for the first recorded frame.
        number: u64,
    },
    /// A system finished one activation.
    System {
        /// System name.
        name: String,
        /// Whether it returned an error.
        failed: bool,
        /// How long the activation took, in microseconds.
        elapsed_us: u64,
    },
    /// A model call completed.
    Model {
        /// Provider identifier.
        provider: String,
        /// Model name.
        model: String,
        /// Input tokens, when known.
        input_tokens: u64,
        /// Output tokens, when known.
        output_tokens: u64,
        /// Whether the call failed.
        failed: bool,
    },
    /// A tool call completed.
    Tool {
        /// Tool name.
        name: String,
        /// Whether the tool reported success.
        ok: bool,
    },
    /// An agent changed status.
    Agent {
        /// Agent name.
        name: String,
        /// Status name, matching [`xihe_agent::AgentStatus::as_str`].
        status: String,
    },
    /// A free-form note.
    Note {
        /// Who wrote it.
        source: String,
        /// The note.
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

/// One recorded event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    /// Monotonic sequence, starting at 1.
    pub seq: u64,
    /// What happened.
    pub kind: EventKind,
}

/// Counters derived from the event log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// Frames recorded.
    pub frames: u64,
    /// System activations.
    pub system_runs: u64,
    /// System activations that failed.
    pub system_failures: u64,
    /// Model calls.
    pub model_calls: u64,
    /// Model calls that failed.
    pub model_failures: u64,
    /// Input tokens recorded.
    pub input_tokens: u64,
    /// Output tokens recorded.
    pub output_tokens: u64,
    /// Tool calls recorded.
    pub tool_calls: u64,
    /// Tool calls that failed.
    pub tool_failures: u64,
    /// Agent status changes.
    pub agent_events: u64,
}

impl Snapshot {
    /// Tokens recorded across every model call.
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

/// The event log, stored as a world resource.
#[derive(Clone, Debug, Default)]
pub struct Observer {
    events: Vec<Event>,
    next: u64,
    started: Option<Instant>,
}

impl Observer {
    /// An empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `kind` and return its sequence number.
    pub fn record(&mut self, kind: EventKind) -> u64 {
        if self.started.is_none() {
            self.started = Some(Instant::now());
        }
        self.next += 1;
        let seq = self.next;
        self.events.push(Event { seq, kind });
        seq
    }

    /// Record a frame boundary.
    pub fn frame(&mut self, number: u64) -> u64 {
        self.record(EventKind::Frame { number })
    }

    /// Record a system activation.
    pub fn system(&mut self, name: impl Into<String>, failed: bool, elapsed: Duration) -> u64 {
        self.record(EventKind::System {
            name: name.into(),
            failed,
            elapsed_us: u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
        })
    }

    /// Record a model call.
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

    /// Record a tool call.
    pub fn tool(&mut self, name: impl Into<String>, ok: bool) -> u64 {
        self.record(EventKind::Tool {
            name: name.into(),
            ok,
        })
    }

    /// Record an agent status change.
    pub fn agent(&mut self, name: impl Into<String>, status: impl Into<String>) -> u64 {
        self.record(EventKind::Agent {
            name: name.into(),
            status: status.into(),
        })
    }

    /// Record a note.
    pub fn note(&mut self, source: impl Into<String>, text: impl Into<String>) -> u64 {
        self.record(EventKind::Note {
            source: source.into(),
            text: text.into(),
        })
    }

    /// Every event, oldest first.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// How many events have been recorded.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Time since the first event.
    pub fn elapsed(&self) -> Duration {
        self.started
            .map_or(Duration::ZERO, |started| started.elapsed())
    }

    /// Counters over the whole log.
    pub fn snapshot(&self) -> Snapshot {
        let mut snapshot = Snapshot::default();
        for event in &self.events {
            match &event.kind {
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
        snapshot
    }

    /// The log as lines, for a demo or a failing test.
    pub fn render(&self) -> String {
        self.events
            .iter()
            .map(|event| format!("{:04} {}", event.seq, event.kind))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// JSON object with the snapshot and the event lines.
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

    /// Read the observer out of a world, inserting an empty one when absent.
    pub fn ensure(world: &mut World) -> &mut Self {
        if !world.has_resource::<Self>() {
            world.insert_resource(Self::new());
        }
        world.resource_mut::<Self>().expect("just inserted")
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
}
