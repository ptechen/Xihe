//! 工具：智能体改变其自身上下文之外任何事物的唯一途径。
//!
//! [`Tool`] 由 [`ToolDescriptor`]（其名字、描述与参数 schema）
//! 向模型描述，并以 [`ToolCall`] 按名字调用。注册表是一个世界资源，
//! 因此每个智能体共享同一组能力，而每智能体策略——一个智能体可用
//! 哪些工具、哪些需要人——住在 [`crate::guardrail`] 与 [`crate::hitl`]，
//! 而非这里。
//!
//! 两条规则保持循环可信：
//!
//!   * 工具以 [`ToolOutcome`] 作答，绝不返回 `Err`，因为“没有匹配的行”
//!     是*给模型的信息*：它作为工具结果回到转录中，由模型决定下一步。
//!     即便工具 panic，注册表也强制执行这一点——捕获 unwind 并返回
//!     一个错误结果，使一个有 bug 的工具无法中止调度。只有注册表报告
//!     [`AgentError`](crate::error::AgentError)，且仅针对从未被分派的调用——
//!     一个未注册的名字。
//!   * 工具获得 [`ToolContext`]，绝不获得世界。工具在 ReAct 系统不持有
//!     世界锁时运行；把锁交给任意工具代码会使 apply 步骤与自身死锁。
//!     因此工具所需的一切——记忆存储、时钟、调用实体——都被显式递交。

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use xihe_core::Entity;
use xihe_llm::{ToolCall, ToolDescriptor};

use crate::error::AgentError;

/// 调用智能体是什么、它能触及什么。
#[derive(Clone)]
pub struct ToolContext {
    /// 请求此调用的智能体实体。
    pub agent: Entity,
    /// 调用所属的对话，当智能体有一个时。
    pub session: Option<String>,
    /// 该对话内的轮号，从一开始计数。
    pub turn: u64,
    /// 共享记忆存储，当运行被赋予一个时。
    pub store: Option<Arc<dyn xihe_memory::MemoryStore>>,
    /// 框架想交给其自有工具的任意每运行值。
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
    /// 一个智能体的上下文，未附加存储。
    pub fn new(agent: Entity) -> Self {
        Self {
            agent,
            session: None,
            turn: 0,
            store: None,
            extras: BTreeMap::new(),
        }
    }

/// 命名发起调用的对话。
    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.session = Some(session.into());
        self
    }

/// 设置轮号。
    pub fn with_turn(mut self, turn: u64) -> Self {
        self.turn = turn;
        self
    }

/// 给工具访问共享记忆的能力。
    pub fn with_store(mut self, store: Arc<dyn xihe_memory::MemoryStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// 附加一个命名值，例如沙箱化工具的文件句柄。
    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extras.insert(key.into(), value);
        self
    }

/// 读取智能体附加的一个 extra。
    pub fn extra(&self, key: &str) -> Option<&Value> {
        self.extras.get(key)
    }
}

/// 运行一个工具的结果。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolOutcome {
    /// 当工具运行并报告了一个问题时为 false。
    pub ok: bool,
    /// 返回给循环调用者的结构化载荷。
    pub payload: Value,
    /// 模型读取的文本；必须足以独立决定下一步。
    pub text: String,
}

impl ToolOutcome {
    /// 一次成功的调用。
    pub fn ok(payload: Value, text: impl Into<String>) -> Self {
        Self {
            ok: true,
            payload,
            text: text.into(),
        }
    }

    /// 结果仅为文本的成功调用。
/// 结果仅为文本的成功调用。
    pub fn text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            ok: true,
            payload: Value::String(text.clone()),
            text,
        }
    }

    /// 一次运行并报告了问题的调用，例如“没有匹配的行”。
    pub fn error(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            ok: false,
            payload: json!({ "error": text }),
            text,
        }
    }

    /// 因需要人而未执行的调用。
    pub fn awaiting_approval(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            ok: false,
            payload: json!({ "approval_required": true, "detail": text }),
            text,
        }
    }

    /// 作为工具结果写回转录的消息。
    pub fn transcript_text(&self) -> String {
        if self.ok {
            self.text.clone()
        } else {
            format!("error: {}", self.text)
        }
    }

    /// 当调用为人挂起而非执行时为真。
    pub fn is_approval_request(&self) -> bool {
        self.payload
            .get("approval_required")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// 运行前被策略拦截而未执行的调用：预算耗尽、未知工具、
    /// 人工拒绝或参数校验不过。文本与 [`ToolOutcome::error`] 一样
    /// 发回给模型，但 payload 标记使观察器能把它与运行后的失败区分。
    pub fn refused(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            ok: false,
            payload: json!({ "refused": true, "detail": text }),
            text,
        }
    }

    /// 当调用被策略拦截而非运行时为真。
    pub fn is_refused(&self) -> bool {
        self.payload
            .get("refused")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
}

/// 模型可请求的一项能力。
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// 发送给提供方时的名字、描述与参数 schema。
    fn descriptor(&self) -> ToolDescriptor;

    /// 运行该调用。对模型可回应的任何事，实现返回
    /// [`ToolOutcome::error`] 而非 `Err`。
    async fn run(&self, ctx: &ToolContext, call: &ToolCall) -> ToolOutcome;

    /// 运行前询问人。见 [`crate::hitl`]。
    fn requires_approval(&self) -> bool {
        false
    }

    /// 工具是否只读、可与其他调用安全地并行运行。
    ///
    /// 当一个批次中的每个调用都只读时，循环并发而非顺序执行它们。
    fn readonly(&self) -> bool {
        false
    }

    /// 工具的名字，取自其描述符。
    fn name(&self) -> String {
        self.descriptor().name
    }
}

/// 按名字索引的工具，作为世界资源共享。
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
/// 一个空注册表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加一个工具，替换同名的较早者。
    pub fn register<T: Tool>(&mut self, tool: T) -> Option<Arc<dyn Tool>> {
        self.tools.insert(tool.name(), Arc::new(tool))
    }

    /// 在共享句柄之后添加一个工具，用于运行时构建的注册表。
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) -> Option<Arc<dyn Tool>> {
        self.tools.insert(tool.name(), tool)
    }

/// 移除一个工具，若已注册则返回它。
    pub fn unregister(&mut self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.remove(name)
    }

/// 注册在 `name` 下的工具。
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

/// `name` 是否已注册。
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// 按排序顺序的名字，使错误消息与提示词稳定。
    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

/// 注册了多少工具。
    pub fn len(&self) -> usize {
        self.tools.len()
    }

/// 是否未注册任何工具。
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// 每个已注册工具的描述符，按名字排序。
    pub fn descriptors(&self) -> Vec<ToolDescriptor> {
        let mut list: Vec<ToolDescriptor> = self
            .tools
            .values()
            .map(|tool| tool.descriptor())
            .collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }

    /// 限制到允许列表的描述符，按允许列表的顺序。
    ///
    /// 未知名字被跳过而非报告：一个提到构建所没有之工具的
    /// 允许列表，应当缩减模型可请求的范围，而非停止运行。
    pub fn descriptors_for(&self, allow: &[String]) -> Vec<ToolDescriptor> {
        allow
            .iter()
            .filter_map(|name| self.tools.get(name))
            .map(|tool| tool.descriptor())
            .collect()
    }

/// 对 `name` 的调用是否必须在运行前被批准。
    pub fn requires_approval(&self, name: &str) -> bool {
        self.tools.get(name).is_some_and(|tool| tool.requires_approval())
    }

    /// 命名工具是否声明自己只读。
    pub fn is_readonly(&self, name: &str) -> bool {
        self.tools.get(name).is_some_and(|tool| tool.readonly())
    }

    /// 请求人的工具的名字，用于提示词与日志。
    pub fn approval_required(&self) -> Vec<String> {
        self.tools
            .iter()
            .filter(|(_, tool)| tool.requires_approval())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// 按名字运行一次调用。
    ///
    /// 未知名字是框架错误，而非作为文本发回给模型之物：
    /// 模型无法使用它从未见过的工具，因此这意味着注册表
    /// 在它底下变了。
    ///
    /// panic 的工具在这里被捕获并变成一个错误 [`ToolOutcome`]，
    /// 使一个有 bug 的工具无法中止调度。模型把失败看作工具结果，
    /// 并决定下一步。
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

    /// 针对工具的 schema 校验模型产出的参数。
    ///
    /// 返回问题列表，调用良构时为空。循环把这些发回给模型
    /// 而非运行工具，因为缺少必需参数的调用会以模型看不到的方式失败。
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

/// 由闭包定义的工具，用于测试与一次性能力。
pub struct FnTool {
    descriptor: ToolDescriptor,
    approval: bool,
    readonly: bool,
    action: Box<dyn Fn(&ToolContext, &ToolCall) -> ToolOutcome + Send + Sync + 'static>,
}

impl FnTool {
    /// 结果由其参数同步计算的工具。
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

    /// 附加参数 schema。
    pub fn with_parameters(mut self, schema: Value) -> Self {
        self.descriptor = self.descriptor.with_parameters(schema);
        self
    }

    /// 要求一个名为 `key` 的参数。
    pub fn with_required_string(mut self, key: &str, description: &str) -> Self {
        self.descriptor = self.descriptor.with_required_string(key, description);
        self
    }

    /// 运行前询问人。
    pub fn requiring_approval(mut self) -> Self {
        self.approval = true;
        self
    }

    /// 声明工具只读，使其批次可并发运行。
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

/// `echo`——返回其参数。演练循环的最小工具。
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

/// `lookup`——从固定表作答，充当真实数据源的替身。
pub struct LookupTool {
    table: BTreeMap<String, String>,
}

impl LookupTool {
/// 从键值对构建表。
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

/// 工具可读取的时钟，使测试不受墙钟摆布。
pub trait Clock: Send + Sync + 'static {
    /// 自 Unix 纪元以来的整秒数。
    fn now(&self) -> i64;
}

/// 真实时钟。
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs() as i64)
            .unwrap_or(0)
    }
}

/// 读取一个值、每次调用前进 `step` 的时钟，用于必须可复现的提示词。
pub struct FixedClock {
    value: std::sync::Mutex<i64>,
    step: i64,
}

impl FixedClock {
/// 把时钟起始于 `value`。
    pub fn new(value: i64) -> Self {
        Self {
            value: std::sync::Mutex::new(value),
            step: 1,
        }
    }

/// 每次读取前进 `step` 秒。
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

/// `now`——当前 unix 时间，作为一个无副作用工具的演示，
/// 其结果模型无法自行计算。
pub struct NowTool {
    clock: Arc<dyn Clock>,
}

impl NowTool {
    /// 读取真实系统时钟。
    pub fn new() -> Self {
        Self::with_clock(Arc::new(SystemClock))
    }

/// 从注入的时钟读取时间。
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

        // 注册表毫发无损：一个守规矩的工具之后仍能运行
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
