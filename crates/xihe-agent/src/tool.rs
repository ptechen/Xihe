//! Tools: the only way an agent changes anything outside its own context.
//!
//! A [`Tool`] is described to the model by a [`ToolDescriptor`] (its name,
//! description and argument schema) and invoked by name with a [`ToolCall`].
//! The registry is a world resource, so every agent shares one set of
//! capabilities, while per-agent policy — which tools an agent may use, which
//! need a human — lives in [`crate::guardrail`] and [`crate::hitl`], not here.
//!
//! Two rules keep the loop trustworthy:
//!
//!   * A tool answers with a [`ToolOutcome`], never an `Err`, because "no rows
//!     matched" is *information for the model*: it goes back into the transcript
//!     as a tool result and the model decides what to do next. The registry
//!     enforces this even for a tool that panics, catching the unwind and
//!     returning an error outcome so one buggy tool cannot abort the schedule.
//!     Only the registry reports an [`AgentError`](crate::error::AgentError),
//!     and only for a call that was never dispatched — an unregistered name.
//!   * A tool gets a [`ToolContext`], never the world. Tools run while the
//!     ReAct system holds no world lock; handing the lock to arbitrary tool
//!     code would deadlock the apply step against itself. So anything a tool
//!     needs — the memory store, the clock, the calling entity — is handed
//!     over explicitly.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use xihe_core::Entity;
use xihe_llm::{ToolCall, ToolDescriptor};

use crate::error::AgentError;

/// What the calling agent is and what it can reach.
#[derive(Clone)]
pub struct ToolContext {
    /// The agent entity that asked for this call.
    pub agent: Entity,
    /// Conversation the call belongs to, when the agent has one.
    pub session: Option<String>,
    /// Turn number inside that conversation, counting from one.
    pub turn: u64,
    /// Shared memory store, when the run was given one.
    pub store: Option<Arc<dyn xihe_memory::MemoryStore>>,
    /// Arbitrary per-run values a harness wants to hand to its own tools.
    pub extras: BTreeMap<String, Value>,
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("agent", &self.agent)
            .field("session", &self.session)
            .field("turn", &self.turn)
            .field("store", &self.store.as_ref().map(|store| store.id()))
            .field("extras", &self.extras.keys())
            .finish()
    }
}

impl ToolContext {
    /// A context for one agent, with no store attached.
    pub fn new(agent: Entity) -> Self {
        Self {
            agent,
            session: None,
            turn: 0,
            store: None,
            extras: BTreeMap::new(),
        }
    }

/// Name the conversation the call was made in.
    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.session = Some(session.into());
        self
    }

/// Set the turn number.
    pub fn with_turn(mut self, turn: u64) -> Self {
        self.turn = turn;
        self
    }

/// Give the tool access to shared memory.
    pub fn with_store(mut self, store: Arc<dyn xihe_memory::MemoryStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Attach a named value, e.g. a file handle for a sandboxed tool.
    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extras.insert(key.into(), value);
        self
    }

/// Read an extra attached by the agent.
    pub fn extra(&self, key: &str) -> Option<&Value> {
        self.extras.get(key)
    }
}

/// The result of running a tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolOutcome {
    /// False when the tool ran and reported a problem.
    pub ok: bool,
    /// Structured payload returned to the caller of the loop.
    pub payload: Value,
    /// Text the model reads; must be enough on its own to decide the next step.
    pub text: String,
}

impl ToolOutcome {
    /// A successful call.
    pub fn ok(payload: Value, text: impl Into<String>) -> Self {
        Self {
            ok: true,
            payload,
            text: text.into(),
        }
    }

    /// A successful call whose result is only text.
/// A successful call whose result is only text.
    pub fn text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            ok: true,
            payload: Value::String(text.clone()),
            text,
        }
    }

    /// A call that ran and reported a problem, e.g. "no rows matched".
    pub fn error(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            ok: false,
            payload: json!({ "error": text }),
            text,
        }
    }

    /// A call that was not performed because it needs a human.
    pub fn awaiting_approval(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            ok: false,
            payload: json!({ "approval_required": true, "detail": text }),
            text,
        }
    }

    /// The message written back into the transcript as the tool result.
    pub fn transcript_text(&self) -> String {
        if self.ok {
            self.text.clone()
        } else {
            format!("error: {}", self.text)
        }
    }

    /// True when the call was parked for a human rather than executed.
    pub fn is_approval_request(&self) -> bool {
        self.payload
            .get("approval_required")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
}

/// One capability the model may ask for.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Name, description and argument schema as sent to the provider.
    fn descriptor(&self) -> ToolDescriptor;

    /// Run the call. Implementations return [`ToolOutcome::error`] rather than
    /// an `Err` for anything the model could react to.
    async fn run(&self, ctx: &ToolContext, call: &ToolCall) -> ToolOutcome;

    /// Ask a human before running. See [`crate::hitl`].
    fn requires_approval(&self) -> bool {
        false
    }

    /// Whether the tool only reads and can safely run alongside other calls.
    ///
    /// When every call in a batch is readonly, the loop executes them
    /// concurrently instead of sequentially.
    fn readonly(&self) -> bool {
        false
    }

    /// The tool's name, taken from its descriptor.
    fn name(&self) -> String {
        self.descriptor().name
    }
}

/// Name-indexed tools, shared as a world resource.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
/// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tool, replacing an earlier one of the same name.
    pub fn register<T: Tool>(&mut self, tool: T) -> Option<Arc<dyn Tool>> {
        self.tools.insert(tool.name(), Arc::new(tool))
    }

    /// Add a tool behind a shared handle, for registries built at runtime.
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) -> Option<Arc<dyn Tool>> {
        self.tools.insert(tool.name(), tool)
    }

/// Remove a tool, returning it if it was registered.
    pub fn unregister(&mut self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.remove(name)
    }

/// The tool registered under `name`.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

/// Whether `name` is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Names in sorted order, so error messages and prompts are stable.
    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

/// How many tools are registered.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

/// Whether nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Descriptors for every registered tool, ordered by name.
    pub fn descriptors(&self) -> Vec<ToolDescriptor> {
        let mut list: Vec<ToolDescriptor> = self
            .tools
            .values()
            .map(|tool| tool.descriptor())
            .collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }

    /// Descriptors restricted to an allowlist, in the allowlist's order.
    ///
    /// Unknown names are skipped rather than reported: an allowlist that
    /// mentions a tool the build does not have should shrink what the model can
    /// ask for, not stop the run.
    pub fn descriptors_for(&self, allow: &[String]) -> Vec<ToolDescriptor> {
        allow
            .iter()
            .filter_map(|name| self.tools.get(name))
            .map(|tool| tool.descriptor())
            .collect()
    }

/// Whether a call to `name` must be approved before running.
    pub fn requires_approval(&self, name: &str) -> bool {
        self.tools.get(name).is_some_and(|tool| tool.requires_approval())
    }

    /// Whether the named tool declared itself read-only.
    pub fn is_readonly(&self, name: &str) -> bool {
        self.tools.get(name).is_some_and(|tool| tool.readonly())
    }

    /// Names of the tools that ask for a human, for prompts and logs.
    pub fn approval_required(&self) -> Vec<String> {
        self.tools
            .iter()
            .filter(|(_, tool)| tool.requires_approval())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Run one call by name.
    ///
    /// An unknown name is a harness error, not something to send back to the
    /// model as text: the model cannot use a tool it was never shown, so this
    /// means the registry changed underneath it.
    ///
    /// A tool that panics is caught here and turned into an error
    /// [`ToolOutcome`], so one buggy tool cannot abort the schedule. The model
    /// sees the failure as a tool result and decides what to do next.
    pub async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ToolOutcome, AgentError> {
        let tool = self.tools.get(&call.name).ok_or_else(|| AgentError::UnknownTool {
            name: call.name.clone(),
            available: self.names(),
        })?;
        use futures::FutureExt;
        let outcome = std::panic::AssertUnwindSafe(tool.run(ctx, call))
            .catch_unwind()
            .await
            .unwrap_or_else(|_| {
                tracing::warn!(tool = %call.name, "a tool panicked");
                ToolOutcome::error(format!("tool `{}` panicked", call.name))
            });
        Ok(outcome)
    }

    /// Validate the arguments a model produced against the tool's schema.
    ///
    /// Returns the list of problems, empty when the call is well-formed. The
    /// loop sends these back to the model instead of running the tool, since a
    /// call missing its required argument would fail in a way the model cannot
    /// see.
    pub fn validate(&self, call: &ToolCall) -> Vec<String> {
        let Some(tool) = self.tools.get(&call.name) else {
            return vec![format!("unknown tool `{}`", call.name)];
        };
        let schema = tool.descriptor();
        let mut problems = Vec::new();
        let arguments = call
            .arguments
            .as_object()
            .cloned()
            .unwrap_or_default();
        if let Some(properties) = schema.parameters.get("properties").and_then(Value::as_object) {
            for key in arguments.keys() {
                if !properties.contains_key(key) {
                    problems.push(format!("`{key}` is not an argument of `{}`", call.name));
                }
            }
        }
        let required: Vec<String> = schema
            .parameters
            .get("required")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        for key in required {
            if !arguments.contains_key(&key) {
                problems.push(format!("required argument `{key}` is missing"));
            }
        }
        problems
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.names())
            .finish()
    }
}

/// A tool defined by a closure, for tests and one-off capabilities.
pub struct FnTool {
    descriptor: ToolDescriptor,
    approval: bool,
    readonly: bool,
    action: Box<dyn Fn(&ToolContext, &ToolCall) -> ToolOutcome + Send + Sync + 'static>,
}

impl FnTool {
    /// A tool whose result is computed synchronously from its arguments.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        action: impl Fn(&ToolContext, &ToolCall) -> ToolOutcome + Send + Sync + 'static,
    ) -> Self {
        Self {
            descriptor: ToolDescriptor::new(name, description),
            approval: false,
            readonly: false,
            action: Box::new(action),
        }
    }

    /// Attach the argument schema.
    pub fn with_parameters(mut self, schema: Value) -> Self {
        self.descriptor = self.descriptor.with_parameters(schema);
        self
    }

    /// Require an argument named `key`.
    pub fn with_required_string(mut self, key: &str, description: &str) -> Self {
        self.descriptor = self.descriptor.with_required_string(key, description);
        self
    }

    /// Ask a human before running.
    pub fn requiring_approval(mut self) -> Self {
        self.approval = true;
        self
    }

    /// Declare the tool read-only so batches of it can run concurrently.
    pub fn with_readonly(mut self) -> Self {
        self.readonly = true;
        self
    }
}

#[async_trait]
impl Tool for FnTool {
    fn descriptor(&self) -> ToolDescriptor {
        self.descriptor.clone()
    }

    async fn run(&self, ctx: &ToolContext, call: &ToolCall) -> ToolOutcome {
        (self.action)(ctx, call)
    }

    fn requires_approval(&self) -> bool {
        self.approval
    }

    fn readonly(&self) -> bool {
        self.readonly
    }
}

/// `echo` — returns its argument. The smallest tool that exercises the loop.
pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new("echo", "Repeat the given text back verbatim.")
            .with_required_string("text", "The text to repeat")
    }

    async fn run(&self, _ctx: &ToolContext, call: &ToolCall) -> ToolOutcome {
        match call.string_argument("text") {
            Some(text) => ToolOutcome::ok(json!({ "text": text }), text.to_string()),
            None => ToolOutcome::error("echo needs a `text` argument"),
        }
    }

    fn readonly(&self) -> bool {
        true
    }
}

/// `lookup` — answers from a fixed table, standing in for a real data source.
pub struct LookupTool {
    table: BTreeMap<String, String>,
}

impl LookupTool {
/// Build the table from pairs.
    pub fn new(entries: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>) -> Self {
        Self {
            table: entries
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }
}

#[async_trait]
impl Tool for LookupTool {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new("lookup", "Look up a term in the reference table.")
            .with_required_string("term", "The term to look up")
    }

    async fn run(&self, _ctx: &ToolContext, call: &ToolCall) -> ToolOutcome {
        let Some(term) = call.string_argument("term") else {
            return ToolOutcome::error("lookup needs a `term` argument");
        };
        match self.table.get(term) {
            Some(value) => ToolOutcome::ok(
                json!({ "term": term, "value": value }),
                format!("{term} = {value}"),
            ),
            None => ToolOutcome::error(format!("no entry for `{term}`")),
        }
    }

    fn readonly(&self) -> bool {
        true
    }
}

/// A clock a tool can read, so tests are not at the mercy of the wall clock.
pub trait Clock: Send + Sync + 'static {
    /// Whole seconds since the Unix epoch.
    fn now(&self) -> i64;
}

/// The real clock.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs() as i64)
            .unwrap_or(0)
    }
}

/// A clock that reads one value and moves by `step` every call, for prompts
/// that must be reproducible.
pub struct FixedClock {
    value: std::sync::Mutex<i64>,
    step: i64,
}

impl FixedClock {
/// Start the clock at `value`.
    pub fn new(value: i64) -> Self {
        Self {
            value: std::sync::Mutex::new(value),
            step: 1,
        }
    }

/// Advance by `step` seconds on every read.
    pub fn with_step(mut self, step: i64) -> Self {
        self.step = step;
        self
    }
}

impl Default for FixedClock {
    fn default() -> Self {
        Self::new(1_700_000_000)
    }
}

impl Clock for FixedClock {
    fn now(&self) -> i64 {
        let mut guard = self.value.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = *guard;
        *guard += self.step;
        current
    }
}

/// `now` — the current unix time, as a demonstration of a side-effect-free
/// tool whose result the model cannot compute itself.
pub struct NowTool {
    clock: Arc<dyn Clock>,
}

impl NowTool {
    /// Read the real system clock.
    pub fn new() -> Self {
        Self::with_clock(Arc::new(SystemClock))
    }

/// Read time from an injected clock.
    pub fn with_clock(clock: Arc<dyn Clock>) -> Self {
        Self { clock }
    }
}

impl Default for NowTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for NowTool {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new("now", "Report the current unix time in seconds.")
    }

    async fn run(&self, _ctx: &ToolContext, _call: &ToolCall) -> ToolOutcome {
        let value = self.clock.now();
        ToolOutcome::ok(json!({ "unix_seconds": value }), value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xihe_llm::ToolCall;

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall::new(format!("call-{name}"), name, arguments)
    }

    fn registry() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry.register(LookupTool::new([("xihe", "a river")].map(|(a, b)| (a.to_string(), b.to_string()))));
        registry.register(NowTool::with_clock(Arc::new(FixedClock::new(1000))));
        registry
    }

    #[tokio::test]
    async fn tools_are_described_in_a_stable_order() {
        let registry = registry();
        let names: Vec<String> = registry.descriptors().into_iter().map(|d| d.name).collect();
        assert_eq!(names, ["echo", "lookup", "now"]);
        assert_eq!(registry.len(), 3);
        assert!(!registry.is_empty());
        assert_eq!(registry.names(), names);
    }

    #[tokio::test]
    async fn echo_returns_its_argument() {
        let outcome = registry()
            .execute(&ToolContext::new(Entity::from_raw(1, 1)), &call("echo", json!({"text": "hi"})))
            .await
            .unwrap();
        assert!(outcome.ok);
        assert_eq!(outcome.text, "hi");
        assert_eq!(outcome.payload, json!({"text": "hi"}));
    }

    #[tokio::test]
    async fn a_missing_argument_becomes_tool_text_not_a_harness_error() {
        let outcome = registry()
            .execute(&ToolContext::new(Entity::from_raw(1, 1)), &call("echo", json!({})))
            .await
            .unwrap();
        assert!(!outcome.ok);
        assert!(outcome.text.contains("text"), "{}", outcome.text);
        assert_eq!(outcome.transcript_text(), "error: echo needs a `text` argument");
    }

    #[tokio::test]
    async fn an_unknown_tool_is_a_harness_error() {
        let error = registry()
            .execute(&ToolContext::new(Entity::from_raw(1, 1)), &call("nope", json!({})))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), "unknown_tool");
        assert!(error.to_string().contains("echo"), "{error}");
        assert!(!error.is_retryable());
    }

    #[tokio::test]
    async fn a_panicking_tool_is_contained_as_an_error_outcome() {
        struct Boom;

        #[async_trait]
        impl Tool for Boom {
            fn descriptor(&self) -> ToolDescriptor {
                ToolDescriptor::new("boom", "explodes")
            }
            async fn run(&self, _ctx: &ToolContext, _call: &ToolCall) -> ToolOutcome {
                panic!("tool exploded")
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(Boom);
        registry.register(EchoTool);
        let ctx = ToolContext::new(Entity::from_raw(1, 1));

        let outcome = registry
            .execute(&ctx, &call("boom", json!({})))
            .await
            .expect("the registry stays up");
        assert!(!outcome.ok);
        assert!(outcome.text.contains("panicked"), "{}", outcome.text);

        // the registry is unharmed: a well-behaved tool still runs afterwards
        let echo = registry
            .execute(&ctx, &call("echo", json!({"text": "hi"})))
            .await
            .unwrap();
        assert!(echo.ok);
        assert_eq!(echo.text, "hi");
    }

    #[tokio::test]
    async fn lookup_reports_misses_as_errors() {
        let ctx = ToolContext::new(Entity::from_raw(2, 1));
        let hit = registry().execute(&ctx, &call("lookup", json!({"term": "xihe"}))).await.unwrap();
        assert_eq!(hit.text, "xihe = a river");
        let miss = registry().execute(&ctx, &call("lookup", json!({"term": "nile"}))).await.unwrap();
        assert!(!miss.ok, "{miss:?}");
    }

    #[tokio::test]
    async fn a_fixed_clock_makes_time_reproducible() {
        let registry = registry();
        let ctx = ToolContext::new(Entity::from_raw(3, 1));
        let first = registry.execute(&ctx, &call("now", json!({}))).await.unwrap();
        let second = registry.execute(&ctx, &call("now", json!({}))).await.unwrap();
        assert_eq!(first.payload, json!({"unix_seconds": 1000}));
        assert_eq!(second.payload, json!({"unix_seconds": 1001}), "each call advances");
    }

    #[tokio::test]
    async fn a_closure_tool_is_enough_for_a_test() {
        let mut registry = ToolRegistry::new();
        registry.register(
            FnTool::new("add", "Add two numbers.", |_ctx, call| {
                let sum = call
                    .argument("a")
                    .and_then(Value::as_i64)
                    .unwrap_or_default()
                    + call
                        .argument("b")
                        .and_then(Value::as_i64)
                        .unwrap_or_default();
                ToolOutcome::text(sum.to_string())
            })
            .with_parameters(json!({
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            })),
        );
        let outcome = registry
            .execute(
                &ToolContext::new(Entity::from_raw(4, 1)),
                &call("add", json!({"a": 2, "b": 3})),
            )
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(outcome.text, "5");
    }

    #[test]
    fn registration_replaces_and_unregisters() {
        let mut registry = ToolRegistry::new();
        assert!(registry.register(EchoTool).is_none());
        let replaced = registry.register(EchoTool);
        assert!(replaced.is_some(), "the older tool is returned");
        assert!(registry.unregister("echo").is_some());
        assert!(registry.unregister("echo").is_none());
        assert!(registry.is_empty());
    }

    #[test]
    fn allowlists_filter_descriptors() {
        let registry = registry();
        let chosen = registry.descriptors_for(&["now".to_string(), "echo".to_string(), "ghost".to_string()]);
        let names: Vec<String> = chosen.into_iter().map(|descriptor| descriptor.name).collect();
        assert_eq!(names, ["now", "echo"], "the allowlist decides the order");
    }

    #[test]
    fn validation_catches_missing_and_extra_arguments() {
        let registry = registry();
        assert!(registry.validate(&call("echo", json!({"text": "hi"}))).is_empty());
        let missing = registry.validate(&call("echo", json!({})));
        assert_eq!(missing, ["required argument `text` is missing"]);
        let extra = registry.validate(&call("echo", json!({"text": "hi", "nope": 1})));
        assert_eq!(extra.len(), 1, "{extra:?}");
        assert!(extra[0].contains("nope"), "{extra:?}");
        assert_eq!(
            registry.validate(&call("ghost", json!({}))),
            vec!["unknown tool `ghost`".to_string()]
        );
    }

    #[test]
    fn approval_is_a_property_of_the_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(
            FnTool::new("drop", "Delete rows.", |_ctx, _call| ToolOutcome::text("dropped"))
                .requiring_approval(),
        );
        registry.register(EchoTool);
        assert!(registry.requires_approval("drop"));
        assert!(!registry.requires_approval("echo"));
        assert!(!registry.requires_approval("missing"));
        assert_eq!(registry.approval_required(), ["drop".to_string()]);
    }

    #[test]
    fn contexts_carry_their_environment() {
        let ctx = ToolContext::new(Entity::from_raw(7, 1))
            .with_session("chat-1")
            .with_turn(3)
            .with_extra("root", json!("/tmp"));
        assert_eq!(ctx.session.as_deref(), Some("chat-1"));
        assert_eq!(ctx.turn, 3);
        assert_eq!(ctx.extra("root"), Some(&json!("/tmp")));
        assert!(ctx.extra("other").is_none());
        assert!(ctx.store.is_none());
    }

    #[test]
    fn outcomes_distinguish_failure_from_approval() {
        let error = ToolOutcome::error("boom");
        assert!(!error.is_approval_request());
        assert_eq!(error.transcript_text(), "error: boom");
        let approval = ToolOutcome::awaiting_approval("delete");
        assert!(approval.is_approval_request());
        assert_eq!(approval.transcript_text(), "error: delete");
        assert_eq!(ToolOutcome::text("plain").transcript_text(), "plain");
    }

    #[test]
    fn debug_shows_the_tool_names() {
        let text = format!("{:?}", registry());
        assert!(text.contains("echo"), "{text}");
    }
}
