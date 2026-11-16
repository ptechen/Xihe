//! 聊天补全数据模型：角色、消息、工具调用与参数。
//!
//! 这些类型与提供方无关且对 serde 友好，因此对话可以写入磁盘，
//! 或针对 [`crate::agent_llm::MockProvider`] 回放。

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// 消息的产出者。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// 引导指令，通常是第一条消息。
    System,
    /// 人类（或上游智能体）输入。
    #[default]
    User,
    /// 模型输出。
    Assistant,
    /// [`Role::Assistant`] 工具调用的结果，回传给模型。
    Tool,
}

impl Role {
    /// 聊天补全 API 使用的传输层拼写。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    /// 对 [`Role::System`] 为真。
    pub fn is_system(self) -> bool {
        matches!(self, Self::System)
    }

    /// 对 [`Role::Assistant`] 为真。
    pub fn is_assistant(self) -> bool {
        matches!(self, Self::Assistant)
    }
}

/// 单条聊天消息：文本、工具调用或工具结果。
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// 消息的作者。
    pub role: Role,
    /// 文本载荷。
    pub content: String,
    /// 可选的显示名，供多智能体转录使用。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// 助手请求运行时执行的调用。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// 在 [`Role::Tool`] 消息上设置，用于标识所应答的调用。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// 为任意角色构建消息。
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            ..Self::default()
        }
    }

    /// 一个系统指令轮。
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// 一个用户轮。
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// 一个助手轮。
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// 一个工具结果，为需要字符串载荷的提供方渲染为文本。
    pub fn tool_result(call_id: impl Into<String>, payload: Value) -> Self {
        let content = match payload {
            Value::String(text) => text,
            other => other.to_string(),
        };
        Self {
            role: Role::Tool,
            content,
            tool_call_id: Some(call_id.into()),
            ..Self::default()
        }
    }

    /// 标注作者，供多智能体转录使用。
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// 附上模型希望执行的调用。
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = calls;
        self
    }

    /// 本轮是否请求工具工作。
    pub fn calls_tools(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// 近似的 token 开销，用于窗口化与预算护栏。
    pub fn tokens(&self) -> usize {
        let mut text = self.content.len() + self.name.as_ref().map_or(0, String::len);
        for call in &self.tool_calls {
            text += call.name.len() + call.arguments.to_string().len() + 8;
        }
        crate::agent_llm::estimate_tokens(text)
    }

    /// 将一串消息折叠为纯文本，每行以其角色为前缀。
    pub fn to_transcript_line(&self) -> String {
        let speaker = self.name.as_deref().unwrap_or(self.role.as_str());
        if self.calls_tools() {
            let calls = self
                .tool_calls
                .iter()
                .map(ToolCall::render)
                .collect::<Vec<_>>()
                .join(", ");
            return format!("{speaker}: {calls}");
        }
        format!("{speaker}: {}", self.content)
    }
}

impl std::fmt::Display for ChatMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_transcript_line())
    }
}

/// 允许模型调用的函数。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDescriptor {
    /// 模型必须调用的名称。
    pub name: String,
    /// 工具的作用，展示给模型。
    pub description: String,
    /// 参数对象的 JSON Schema。
    pub parameters: Value,
}

impl ToolDescriptor {
    /// 带空参数对象 schema 的描述符。
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    /// 附上 JSON Schema，通常用 `json!` 宏构建。
    pub fn with_parameters(mut self, schema: Value) -> Self {
        self.parameters = schema;
        self
    }

    /// 声明一个字符串参数，这是智能体工具 80% 的情形。
    pub fn with_string_arg(self, key: &str, description: &str, required: bool) -> Self {
        self.with_argument(
            key,
            json!({ "type": "string", "description": description }),
            required,
        )
    }

    /// 用任意 JSON Schema 子 schema 声明一个参数。
    pub fn with_argument(mut self, key: &str, schema: Value, required: bool) -> Self {
        if let Some(props) = self
            .parameters
            .get_mut("properties")
            .and_then(Value::as_object_mut)
        {
            props.insert(key.to_string(), schema);
        }
        if required && !self.is_required(key) {
            let entry = json!(key);
            match self.parameters.get_mut("required") {
                Some(list @ Value::Array(_)) => {
                    if let Some(items) = list.as_array_mut() {
                        items.push(entry);
                    }
                }
                Some(other) => *other = json!([entry]),
                None => self.parameters["required"] = json!([entry]),
            }
        }
        self
    }

    /// 一个必填字符串参数，附带面向模型的描述。
    pub fn with_required_string(self, key: &str, description: &str) -> Self {
        self.with_string_arg(key, description, true)
    }

    /// 已声明的参数名，按 schema 顺序。
    pub fn argument_names(&self) -> Vec<String> {
        self.parameters
            .get("properties")
            .and_then(Value::as_object)
            .map(|props| props.keys().map(String::from).collect())
            .unwrap_or_default()
    }

    /// `key` 是否出现在 schema 的 `required` 列表中。
    pub fn is_required(&self, key: &str) -> bool {
        self.parameters
            .get("required")
            .and_then(Value::as_array)
            .is_some_and(|list| list.iter().any(|item| item.as_str() == Some(key)))
    }
}

/// 模型请求的一次工具调用。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// 提供方分配的 id，由匹配的工具结果回显。
    pub id: String,
    /// 模型请求的工具。
    pub name: String,
    /// 已解析的参数；提供方发送 JSON 文本，我们即时解码。
    pub arguments: Value,
    /// 原始参数字符串，保留以便回放与调试能原样查看。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_arguments: Option<String>,
}

impl ToolCall {
    /// 由已解析的参数构建调用。
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<Value>,
    ) -> Self {
        let arguments = arguments.into();
        let raw_arguments = Some(arguments.to_string());
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
            raw_arguments,
        }
    }

    /// 解析提供方的参数载荷，失败时回退为空对象。
    pub fn from_json_text(
        id: impl Into<String>,
        name: impl Into<String>,
        text: &str,
    ) -> (Self, Option<Value>) {
        let id = id.into();
        let name = name.into();
        match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(map)) => (
                Self {
                    id,
                    name,
                    arguments: Value::Object(map),
                    raw_arguments: Some(text.to_string()),
                },
                None,
            ),
            Ok(other) => (
                Self {
                    id,
                    name,
                    arguments: json!({ "value": other }),
                    raw_arguments: Some(text.to_string()),
                },
                None,
            ),
            Err(reason) => (
                Self {
                    id,
                    name,
                    arguments: json!({}),
                    raw_arguments: Some(text.to_string()),
                },
                Some(json!({ "error": "invalid_tool_arguments", "reason": reason.to_string() })),
            ),
        }
    }

    /// 提供方所需的参数载荷：文本，优先使用提供方实际发送的内容。
    pub fn arguments_string(&self) -> String {
        self.raw_arguments
            .clone()
            .unwrap_or_else(|| self.arguments.to_string())
    }

    /// 查找一个参数值。
    pub fn argument(&self, key: &str) -> Option<&Value> {
        self.arguments.get(key)
    }

    /// 读取字符串参数；其他类型产出 `None`。
    pub fn string_argument(&self, key: &str) -> Option<&str> {
        self.argument(key).and_then(Value::as_str)
    }

    /// 供日志与转录使用的单行渲染。
    pub fn render(&self) -> String {
        format!("{}({})", self.name, self.arguments)
    }
}

/// 采样旋钮，对应常见的聊天补全参数。
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationParams {
    /// 采样温度；0.0 为贪心解码。
    ///
    /// 用 `f64` 而非 `f32`，因为这些值直接进入 JSON 体，
    /// 而 0.7 作为 `f32` 会序列化为 0.699999988079071。
    pub temperature: f64,
    /// 核采样截断；`None` 交由提供方决定。
    pub top_p: Option<f64>,
    /// 助手轮的上界。
    pub max_output_tokens: Option<u32>,
    /// 结束生成的序列。
    pub stop: Vec<String>,
    /// 在提供方尊重种子时请求确定性运行。
    pub seed: Option<u64>,
    /// 提供方特有的额外项，在请求体中原样转发。
    pub extensions: Value,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: None,
            max_output_tokens: None,
            stop: Vec::new(),
            seed: None,
            extensions: Value::Null,
        }
    }
}

impl GenerationParams {
    /// 带固定种子的贪心解码：测试所需。
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            ..Self::default()
        }
    }

    /// 设置温度。
    pub fn with_temperature(mut self, value: f64) -> Self {
        self.temperature = value;
        self
    }

    /// 为助手轮设置上限。
    pub fn with_max_output_tokens(mut self, value: u32) -> Self {
        self.max_output_tokens = Some(value);
        self
    }

    /// 固定种子。
    pub fn with_seed(mut self, value: u64) -> Self {
        self.seed = Some(value);
        self
    }
}

/// 运行时坚持要求工具调用的强度。
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ToolChoice {
    #[default]
    /// 让模型决定。
    Auto,
    /// 仅文本，即使附上了工具。
    None,
    /// 必须产出至少一个调用。
    Required,
    /// 强制使用某个特定工具。
    Only(String),
}

impl ToolChoice {
    /// 当此请求禁用工具时为真。
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// 生成停止的原因。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// 一个完整答案。
    Stop,
    /// 模型希望执行工具。
    ToolCalls,
    /// 被 token 预算截断。
    Length,
    /// 被提供方的安全过滤器拒绝。
    ContentFilter,
    /// 被运行时取消。
    Aborted,
}

impl FinishReason {
    /// 将提供方的自由格式字符串映射到我们的小集合。
    pub fn parse(value: &str) -> Self {
        match value {
            "tool_calls" | "function_call" => Self::ToolCalls,
            "length" | "max_tokens" => Self::Length,
            "content_filter" => Self::ContentFilter,
            "abort" | "cancelled" => Self::Aborted,
            _ => Self::Stop,
        }
    }

    /// 传输层拼写。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::ToolCalls => "tool_calls",
            Self::Length => "length",
            Self::ContentFilter => "content_filter",
            Self::Aborted => "abort",
        }
    }
}

/// 单次调用或整个运行累计的 token 计量。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// 提示词消耗的 token。
    pub prompt_tokens: u64,
    /// 回复产出的 token。
    pub completion_tokens: u64,
    /// 两者之和的报告值。
    pub total_tokens: u64,
}

impl TokenUsage {
    /// 由各部分计算总和。
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    /// 累加另一次调用的用量；总和重新计算，不信任传输层的值。
    pub fn accumulate(&mut self, other: &Self) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens = self.prompt_tokens + self.completion_tokens;
    }

    /// 当提供方未报告任何值时为真。
    pub fn is_zero(&self) -> bool {
        self.total_tokens == 0
    }
}

impl std::ops::Add for TokenUsage {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self.accumulate(&rhs);
        self
    }
}

/// 一次生成的结果。
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationResponse {
    /// 助手文本，纯工具调用时为空。
    pub content: String,
    /// 下一轮之前要执行的调用。
    pub tool_calls: Vec<ToolCall>,
    /// 生成停止的原因（若提供方指明了）。
    pub finish_reason: Option<FinishReason>,
    /// 此次调用计费的 token。
    pub usage: TokenUsage,
    /// 实际作答的模型，可能与请求中的不同。
    pub model: String,
    /// 原始提供方载荷，保留用于调试与透传字段。
    pub raw: Option<Value>,
}

impl GenerationResponse {
    /// 一个已完成的文本答案。
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            finish_reason: Some(FinishReason::Stop),
            ..Self::default()
        }
    }

    /// 附上调用并将原因标记为 `ToolCalls`。
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = calls;
        self.finish_reason = Some(FinishReason::ToolCalls);
        self
    }

    /// 设置 token 计量。
    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = usage;
        self
    }

    /// 记录作答的模型。
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// 运行时是否必须执行工具。
    pub fn calls_tools(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// 可重新编码为对话中助手轮的响应。
    pub fn to_message(&self) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            content: self.content.clone(),
            tool_calls: self.tool_calls.clone(),
            ..ChatMessage::default()
        }
    }

    /// 为 ReAct 循环建议的下一步。
    pub fn step(&self) -> GenerationStep {
        if self.calls_tools() {
            return GenerationStep::ExecuteTools;
        }
        match self.finish_reason {
            Some(FinishReason::Length) => GenerationStep::Truncated,
            Some(FinishReason::ContentFilter) => GenerationStep::Blocked,
            Some(FinishReason::Aborted) => GenerationStep::Aborted,
            _ => GenerationStep::Answer,
        }
    }
}

/// ReAct 循环应如何处理一个响应。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationStep {
    /// 文本答案：结束本轮。
    Answer,
    /// 有待执行的工具调用：执行它们并再次调用模型。
    ExecuteTools,
    /// 被 token 预算截断。
    Truncated,
    /// 被提供方的安全过滤器拒绝。
    Blocked,
    /// 被运行时取消。
    Aborted,
}

impl GenerationStep {
    /// 仅对 [`GenerationStep::ExecuteTools`] 为真。
    pub fn needs_tools(self) -> bool {
        matches!(self, Self::ExecuteTools)
    }

    /// 当本轮应结束而非循环时为真。
    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::ExecuteTools)
    }
}

/// 流式生成的一个增量片段。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamChunk {
    /// 自上一块以来新增的文本。
    pub text: String,
    /// 本块中完成的调用。
    pub tool_calls: Vec<ToolCall>,
    /// 在最后一块上设置。
    pub finish_reason: Option<FinishReason>,
    /// 只报告一次，通常在最后一块上。
    pub usage: Option<TokenUsage>,
}

impl StreamChunk {
    /// 一个纯文本块。
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// 对终止块为真。
    pub fn is_last(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// 将流折叠为单个响应，例如提供方只支持流式时。
    pub fn reduce(chunks: impl IntoIterator<Item = Self>) -> GenerationResponse {
        let mut out = GenerationResponse::default();
        let mut calls: Vec<ToolCall> = Vec::new();
        for chunk in chunks {
            out.content.push_str(&chunk.text);
            calls.extend(chunk.tool_calls);
            if chunk.finish_reason.is_some() {
                out.finish_reason = chunk.finish_reason;
            }
            if let Some(usage) = chunk.usage {
                out.usage.accumulate(&usage);
            }
        }
        if !calls.is_empty() {
            out.tool_calls = calls;
            out.finish_reason.get_or_insert(FinishReason::ToolCalls);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_have_wire_names() {
        assert_eq!(Role::Assistant.as_str(), "assistant");
        assert!(Role::System.is_system());
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
    }

    #[test]
    fn messages_round_trip() {
        let message = ChatMessage::tool_result("call_1", json!({ "temp": 21 }));
        let text = serde_json::to_string(&message).unwrap();
        let back: ChatMessage = serde_json::from_str(&text).unwrap();
        assert_eq!(back, message);
        assert_eq!(back.content, "{\"temp\":21}");
        assert_eq!(back.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn tool_results_pass_through_strings() {
        let message = ChatMessage::tool_result("c", json!("plain text"));
        assert_eq!(message.content, "plain text");
    }

    #[test]
    fn tool_call_argument_parsing_survives_bad_json() {
        let (call, error) = ToolCall::from_json_text("id", "search", "{ not json");
        assert_eq!(call.arguments, json!({}));
        assert_eq!(call.raw_arguments.as_deref(), Some("{ not json"));
        assert!(
            error.is_some(),
            "the caller must be able to report the failure"
        );
        let (ok, none) = ToolCall::from_json_text("id", "search", "{\"q\":\"tokio\"}");
        assert_eq!(ok.string_argument("q"), Some("tokio"));
        assert!(none.is_none());
    }

    #[test]
    fn descriptors_describe_arguments() {
        let tool = ToolDescriptor::new("weather", "lookup weather")
            .with_string_arg("city", "City name", true)
            .with_string_arg("unit", "C or F", false);
        assert_eq!(
            tool.parameters["required"],
            json!(["city"]),
            "only the required flag should be listed"
        );
        assert_eq!(
            tool.parameters["properties"]["unit"]["type"],
            json!("string")
        );
    }

    #[test]
    fn step_routing_covers_every_finish_reason() {
        assert_eq!(
            GenerationResponse::text("hi").step(),
            GenerationStep::Answer
        );
        let calls = vec![ToolCall::new("1", "search", json!({}))];
        assert_eq!(
            GenerationResponse::default().with_tool_calls(calls).step(),
            GenerationStep::ExecuteTools
        );
        let mut truncated = GenerationResponse::text("part");
        truncated.finish_reason = Some(FinishReason::Length);
        assert_eq!(truncated.step(), GenerationStep::Truncated);
        assert!(GenerationStep::Answer.is_terminal());
        assert!(!GenerationStep::ExecuteTools.is_terminal());
    }

    #[test]
    fn usage_accumulates() {
        let mut total = TokenUsage::new(10, 5);
        total.accumulate(&TokenUsage::new(1, 1));
        assert_eq!(total, TokenUsage::new(11, 6));
        assert_eq!((total + TokenUsage::new(0, 4)).total_tokens, 21);
        assert!(TokenUsage::default().is_zero());
    }

    #[test]
    fn chunks_fold_back_into_a_response() {
        let response = StreamChunk::reduce([
            StreamChunk::text("The answer"),
            StreamChunk::text(" is 42."),
            StreamChunk {
                text: String::new(),
                tool_calls: Vec::new(),
                finish_reason: Some(FinishReason::Stop),
                usage: Some(TokenUsage::new(7, 3)),
            },
        ]);
        assert_eq!(response.content, "The answer is 42.");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.usage.total_tokens, 10);
    }

    #[test]
    fn transcripts_label_speakers() {
        let line = ChatMessage::user("hello").to_transcript_line();
        assert_eq!(line, "user: hello");
        let named = ChatMessage::assistant("done").with_name("researcher");
        assert_eq!(named.to_transcript_line(), "researcher: done");
        let calling = ChatMessage::assistant("").with_tool_calls(vec![ToolCall::new(
            "c1",
            "search",
            json!({ "q": "tokio" }),
        )]);
        assert_eq!(
            calling.to_transcript_line(),
            "assistant: search({\"q\":\"tokio\"})"
        );
    }

    #[test]
    fn token_estimates_grow_with_text() {
        assert!(ChatMessage::user("a".repeat(4_000)).tokens() > 900);
        assert!(
            ChatMessage::user("").tokens() > 0,
            "every message has framing"
        );
    }

    #[test]
    fn finish_reason_parses_provider_strings() {
        assert_eq!(FinishReason::parse("tool_calls"), FinishReason::ToolCalls);
        assert_eq!(FinishReason::parse("max_tokens"), FinishReason::Length);
        assert_eq!(FinishReason::parse("stop"), FinishReason::Stop);
        assert_eq!(FinishReason::Stop.as_str(), "stop");
    }
}
