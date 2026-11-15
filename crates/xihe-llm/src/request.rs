//! The request a provider receives: model, conversation, tools, sampling.

use crate::message::{ChatMessage, GenerationParams, Role, ToolChoice, ToolDescriptor};
use serde_json::Value;

/// Everything needed to ask a model for the next turn.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationRequest {
    /// Target model; empty means the client's default.
    pub model: String,
    /// The conversation, oldest first.
    pub messages: Vec<ChatMessage>,
    /// Tools the model may call.
    pub tools: Vec<ToolDescriptor>,
    /// How strongly tool use is encouraged.
    pub tool_choice: ToolChoice,
    /// Sampling parameters.
    pub params: GenerationParams,
}

impl GenerationRequest {
    /// A request aimed at `model` with default parameters.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            params: GenerationParams::default(),
            ..Self::default()
        }
    }

    /// Seed the conversation with a system prompt.
    pub fn with_system(mut self, instructions: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::system(instructions));
        self
    }

    /// Append a user turn.
    pub fn with_user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::user(content));
        self
    }

    /// Append a turn verbatim.
    pub fn with_message(mut self, message: ChatMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// Extend the conversation.
    pub fn with_messages(mut self, messages: impl IntoIterator<Item = ChatMessage>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Replace the tool list.
    pub fn with_tools(mut self, tools: Vec<ToolDescriptor>) -> Self {
        self.tools = tools;
        self
    }

    /// Append one tool.
    pub fn with_tool(mut self, tool: ToolDescriptor) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set the tool-choice policy.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// Replace the sampling parameters.
    ///
    /// This overwrites anything an earlier `with_max_tokens` call set, so apply
    /// the parameter block first and the individual overrides after.
    pub fn with_params(mut self, params: GenerationParams) -> Self {
        self.params = params;
        self
    }

    /// Switch to greedy decoding.
    pub fn deterministic(mut self) -> Self {
        self.params = GenerationParams::deterministic();
        self
    }

    /// Cap the assistant turn.
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.params.max_output_tokens = Some(tokens);
        self
    }

    /// Drop the system message and return the rest, e.g. to re-fit a window.
    pub fn without_system(mut self) -> Self {
        self.messages.retain(|message| !message.role.is_system());
        self
    }

    /// The first system message, if present.
    pub fn system_prompt(&self) -> Option<&str> {
        self.messages
            .iter()
            .find(|message| message.role == Role::System)
            .map(|message| message.content.as_str())
    }

    /// The most recent user text.
    pub fn last_user(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .map(|message| message.content.as_str())
    }

    /// Rough prompt size, used for budget checks before the call is made.
    pub fn estimated_prompt_tokens(&self) -> usize {
        self.messages.iter().map(ChatMessage::tokens).sum::<usize>()
            + self.tools.iter().map(tool_schema_size).sum::<usize>()
    }

    /// Names of the attached tools.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|tool| tool.name.as_str()).collect()
    }
}

fn tool_schema_size(tool: &ToolDescriptor) -> usize {
    crate::estimate_tokens(
        tool.name.len() + tool.description.len() + tool.parameters.to_string().len(),
    )
}

/// A named override of selected request fields, applied by agent builders and
/// [`crate::ChatClient`] without cloning the whole request.
#[derive(Clone, Debug, Default)]
pub struct RequestPatch {
    /// Model override.
    pub model: Option<String>,
    /// Full parameter replacement.
    pub params: Option<GenerationParams>,
    /// Output cap override.
    pub max_output_tokens: Option<u32>,
    /// Merged into `params.extensions` object-wise.
    pub extra: Value,
}

impl RequestPatch {
    /// A patch that only changes the model.
    pub fn model(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..Self::default()
        }
    }

    /// Merge this patch into a request.
    pub fn apply(&self, request: &mut GenerationRequest) {
        if let Some(model) = &self.model {
            request.model = model.clone();
        }
        if let Some(params) = &self.params {
            request.params = params.clone();
        }
        if let Some(tokens) = self.max_output_tokens {
            request.params.max_output_tokens = Some(tokens);
        }
        if let (Value::Object(base), Value::Object(extra)) = (&mut request.params.extensions, &self.extra)
        {
            for (key, value) in extra {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn builders_chain_in_order() {
        let request = GenerationRequest::new("gpt-test")
            .with_system("be terse")
            .with_user("hi")
            .with_tool(ToolDescriptor::new("search", "find"))
            .deterministic()
            .with_max_tokens(64);
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.system_prompt(), Some("be terse"));
        assert_eq!(request.last_user(), Some("hi"));
        assert_eq!(request.tool_names(), vec!["search"]);
        assert_eq!(request.params.temperature, 0.0);
        assert_eq!(request.params.max_output_tokens, Some(64));
    }

    #[test]
    fn without_system_keeps_the_dialogue() {
        let request = GenerationRequest::new("m")
            .with_system("ignore me")
            .with_user("keep me")
            .without_system();
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, Role::User);
    }

    #[test]
    fn prompt_estimate_includes_tools() {
        let small = GenerationRequest::new("m").with_user("hi");
        let large = GenerationRequest::new("m")
            .with_user("hi")
            .with_tool(ToolDescriptor::new("a", "d".repeat(400)));
        assert!(
            large.estimated_prompt_tokens() > small.estimated_prompt_tokens(),
            "tool schemas are part of the prompt"
        );
    }

    #[test]
    fn patches_merge_extensions() {
        let mut request = GenerationRequest::new("old").with_params(GenerationParams {
            extensions: json!({ "keep": true }),
            ..GenerationParams::default()
        });
        let patch = RequestPatch {
            extra: json!({ "reasoning_effort": "low" }),
            ..RequestPatch::model("new")
        };
        patch.apply(&mut request);
        assert_eq!(request.model, "new");
        assert_eq!(request.params.extensions["keep"], json!(true));
        assert_eq!(
            request.params.extensions["reasoning_effort"],
            json!("low"),
            "patches must not clobber unrelated extensions"
        );
    }
}
