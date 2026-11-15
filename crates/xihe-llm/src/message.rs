//! The chat-completion data model: roles, messages, tool calls and parameters.
//!
//! These types are provider-agnostic and serde-friendly so a conversation can be
//! written to disk or replayed against [`crate::MockProvider`].

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Who produced a message.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Steering instructions, usually the first message.
    System,
    /// Human (or upstream agent) input.
    #[default]
    User,
    /// Model output.
    Assistant,
    /// The result of a [`Role::Assistant`] tool call, carried back to the model.
    Tool,
}

impl Role {
    /// The wire spelling used by chat-completion APIs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    /// True for [`Role::System`].
    pub fn is_system(self) -> bool {
        matches!(self, Self::System)
    }

    /// True for [`Role::Assistant`].
    pub fn is_assistant(self) -> bool {
        matches!(self, Self::Assistant)
    }
}

/// A single chat message: prose, tool calls, or a tool result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Who authored the message.
    pub role: Role,
    /// The text payload.
    pub content: String,
    /// Optional display name, used by multi-agent transcripts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Calls the assistant asked the runtime to execute.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Set on [`Role::Tool`] messages to identify the call being answered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Build a message for any role.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            ..Self::default()
        }
    }

    /// A system instruction turn.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// A user turn.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// An assistant turn.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// A tool result, rendered as text for providers that want a string payload.
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

    /// Label the author, used by multi-agent transcripts.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Attach the calls the model wants executed.
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = calls;
        self
    }

    /// Whether this turn requests tool work.
    pub fn calls_tools(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Approximate token cost, used for windowing and budget guards.
    pub fn tokens(&self) -> usize {
        let mut text = self.content.len() + self.name.as_ref().map_or(0, String::len);
        for call in &self.tool_calls {
            text += call.name.len() + call.arguments.to_string().len() + 8;
        }
        crate::estimate_tokens(text)
    }

    /// Collapse a run of messages into plain text, prefixing each line with its role.
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

/// A function the model is allowed to invoke.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDescriptor {
    /// The name the model must call.
    pub name: String,
    /// What the tool does, shown to the model.
    pub description: String,
    /// JSON Schema for the arguments object.
    pub parameters: Value,
}

impl ToolDescriptor {
    /// A descriptor with an empty argument-object schema.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    /// Attach a JSON Schema, typically built with the `json!` macro.
    pub fn with_parameters(mut self, schema: Value) -> Self {
        self.parameters = schema;
        self
    }

    /// Declare one string argument, the 80% case for agent tools.
    pub fn with_string_arg(self, key: &str, description: &str, required: bool) -> Self {
        self.with_argument(
            key,
            json!({ "type": "string", "description": description }),
            required,
        )
    }

    /// Declare an argument with an arbitrary JSON Schema subschema.
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

    /// A required string argument with a description for the model.
    pub fn with_required_string(self, key: &str, description: &str) -> Self {
        self.with_string_arg(key, description, true)
    }

    /// Declared argument names, in schema order.
    pub fn argument_names(&self) -> Vec<String> {
        self.parameters
            .get("properties")
            .and_then(Value::as_object)
            .map(|props| props.keys().map(String::from).collect())
            .unwrap_or_default()
    }

    /// Whether `key` appears in the schema's `required` list.
    pub fn is_required(&self, key: &str) -> bool {
        self.parameters
            .get("required")
            .and_then(Value::as_array)
            .is_some_and(|list| list.iter().any(|item| item.as_str() == Some(key)))
    }
}

/// A tool invocation requested by the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned id, echoed back by the matching tool result.
    pub id: String,
    /// The tool the model asked for.
    pub name: String,
    /// Already-parsed arguments; providers send JSON text which we decode eagerly.
    pub arguments: Value,
    /// The raw argument string, kept so replays and debugging can see it verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_arguments: Option<String>,
}

impl ToolCall {
    /// Build a call from already-parsed arguments.
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

    /// Parse a provider's argument payload, falling back to an empty object.
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

    /// The argument payload as providers want it: text, preferring what the
    /// provider actually sent.
    pub fn arguments_string(&self) -> String {
        self.raw_arguments
            .clone()
            .unwrap_or_else(|| self.arguments.to_string())
    }

    /// Look up one argument value.
    pub fn argument(&self, key: &str) -> Option<&Value> {
        self.arguments.get(key)
    }

    /// Read a string argument; other types yield `None`.
    pub fn string_argument(&self, key: &str) -> Option<&str> {
        self.argument(key).and_then(Value::as_str)
    }

    /// A one-line rendering for logs and transcripts.
    pub fn render(&self) -> String {
        format!("{}({})", self.name, self.arguments)
    }
}

/// Sampling knobs, mirroring the common chat-completion parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationParams {
    /// Sampling temperature; 0.0 is greedy decoding.
    ///
    /// `f64` rather than `f32` because these values go straight into a JSON body
    /// and 0.7 as an `f32` serialises as 0.699999988079071.
    pub temperature: f64,
    /// Nucleus sampling cut-off; `None` leaves it to the provider.
    pub top_p: Option<f64>,
    /// Upper bound on the assistant turn.
    pub max_output_tokens: Option<u32>,
    /// Sequences that end generation.
    pub stop: Vec<String>,
    /// Request a deterministic run where the provider honours seeds.
    pub seed: Option<u64>,
    /// Provider-specific extras forwarded verbatim in the request body.
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
    /// Greedy decoding with a fixed seed: what tests want.
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            ..Self::default()
        }
    }

    /// Set the temperature.
    pub fn with_temperature(mut self, value: f64) -> Self {
        self.temperature = value;
        self
    }

    /// Cap the assistant turn.
    pub fn with_max_output_tokens(mut self, value: u32) -> Self {
        self.max_output_tokens = Some(value);
        self
    }

    /// Pin the seed.
    pub fn with_seed(mut self, value: u64) -> Self {
        self.seed = Some(value);
        self
    }
}

/// How strongly the runtime insists on a tool call.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ToolChoice {
    #[default]
    /// Let the model decide.
    Auto,
    /// Text only, even when tools are attached.
    None,
    /// At least one call must be produced.
    Required,
    /// Force one specific tool.
    Only(String),
}

impl ToolChoice {
    /// True when tools are disabled for this request.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Why a generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// A complete answer.
    Stop,
    /// The model wants tools executed.
    ToolCalls,
    /// Cut off by the token budget.
    Length,
    /// Refused by the provider's safety filter.
    ContentFilter,
    /// Cancelled by the runtime.
    Aborted,
}

impl FinishReason {
    /// Map a provider's free-form string onto our small set.
    pub fn parse(value: &str) -> Self {
        match value {
            "tool_calls" | "function_call" => Self::ToolCalls,
            "length" | "max_tokens" => Self::Length,
            "content_filter" => Self::ContentFilter,
            "abort" | "cancelled" => Self::Aborted,
            _ => Self::Stop,
        }
    }

    /// The wire spelling.
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

/// Token accounting for one call or accumulated over a run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Tokens consumed by the prompt.
    pub prompt_tokens: u64,
    /// Tokens produced by the reply.
    pub completion_tokens: u64,
    /// The reported sum of the two.
    pub total_tokens: u64,
}

impl TokenUsage {
    /// Compute the total from its parts.
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    /// Add another call's usage; totals are recomputed, not trusted from the wire.
    pub fn accumulate(&mut self, other: &Self) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens = self.prompt_tokens + self.completion_tokens;
    }

    /// True when the provider reported nothing.
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

/// The outcome of one generation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationResponse {
    /// Assistant text, empty for a pure tool call.
    pub content: String,
    /// Calls to execute before the next turn.
    pub tool_calls: Vec<ToolCall>,
    /// Why generation stopped, when the provider said.
    pub finish_reason: Option<FinishReason>,
    /// Tokens charged for this call.
    pub usage: TokenUsage,
    /// The model that actually answered, which may differ from the request.
    pub model: String,
    /// Raw provider payload, kept for debugging and pass-through fields.
    pub raw: Option<Value>,
}

impl GenerationResponse {
    /// A finished text answer.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            finish_reason: Some(FinishReason::Stop),
            ..Self::default()
        }
    }

    /// Attach calls and mark the reason as `ToolCalls`.
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = calls;
        self.finish_reason = Some(FinishReason::ToolCalls);
        self
    }

    /// Set token accounting.
    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = usage;
        self
    }

    /// Record the model that answered.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Whether the runtime must execute tools.
    pub fn calls_tools(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// The response re-encodable as the assistant turn of a conversation.
    pub fn to_message(&self) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            content: self.content.clone(),
            tool_calls: self.tool_calls.clone(),
            ..ChatMessage::default()
        }
    }

    /// Suggested next step for a ReAct loop.
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

/// What a ReAct loop should do with a response.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationStep {
    /// Prose answer: end the turn.
    Answer,
    /// Tool calls pending: execute them and call the model again.
    ExecuteTools,
    /// Cut off by the token budget.
    Truncated,
    /// Refused by the provider's safety filter.
    Blocked,
    /// Cancelled by the runtime.
    Aborted,
}

impl GenerationStep {
    /// True only for [`GenerationStep::ExecuteTools`].
    pub fn needs_tools(self) -> bool {
        matches!(self, Self::ExecuteTools)
    }

    /// True when the turn should end instead of looping.
    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::ExecuteTools)
    }
}

/// One incremental piece of a streamed generation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamChunk {
    /// Text added since the previous chunk.
    pub text: String,
    /// Calls completed in this chunk.
    pub tool_calls: Vec<ToolCall>,
    /// Set on the final chunk.
    pub finish_reason: Option<FinishReason>,
    /// Reported once, usually on the final chunk.
    pub usage: Option<TokenUsage>,
}

impl StreamChunk {
    /// A text-only chunk.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// True for the terminal chunk.
    pub fn is_last(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// Fold a stream into a single response, e.g. when a provider only streams.
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
        assert!(error.is_some(), "the caller must be able to report the failure");
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
        assert_eq!(tool.parameters["properties"]["unit"]["type"], json!("string"));
    }

    #[test]
    fn step_routing_covers_every_finish_reason() {
        assert_eq!(GenerationResponse::text("hi").step(), GenerationStep::Answer);
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
        assert_eq!(calling.to_transcript_line(), "assistant: search({\"q\":\"tokio\"})");
    }

    #[test]
    fn token_estimates_grow_with_text() {
        assert!(ChatMessage::user("a".repeat(4_000)).tokens() > 900);
        assert!(ChatMessage::user("").tokens() > 0, "every message has framing");
    }

    #[test]
    fn finish_reason_parses_provider_strings() {
        assert_eq!(FinishReason::parse("tool_calls"), FinishReason::ToolCalls);
        assert_eq!(FinishReason::parse("max_tokens"), FinishReason::Length);
        assert_eq!(FinishReason::parse("stop"), FinishReason::Stop);
        assert_eq!(FinishReason::Stop.as_str(), "stop");
    }
}
