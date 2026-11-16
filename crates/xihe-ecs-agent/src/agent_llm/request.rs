//! 提供方接收的请求：模型、对话、工具、采样。

use crate::agent_llm::message::{ChatMessage, GenerationParams, Role, ToolChoice, ToolDescriptor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 模型调用的稳定身份，用作录制与回放的键，使两个发出字节相同请求的
/// 智能体不会冲突。绝不发送给提供方：[`to_wire`](crate::agent_llm::to_wire) 逐字段
/// 构建请求体。
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Correlation {
    /// 发起调用的智能体。
    pub agent: String,
    /// 发起调用时智能体的轮号。
    pub turn: usize,
}

/// 向模型请求下一轮所需的一切。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationRequest {
    /// 目标模型；为空表示使用客户端默认值。
    pub model: String,
    /// 对话，最旧的在前。
    pub messages: Vec<ChatMessage>,
    /// 模型可调用的工具。
    pub tools: Vec<ToolDescriptor>,
    /// 鼓励使用工具的强度。
    pub tool_choice: ToolChoice,
    /// 采样参数。
    pub params: GenerationParams,
    /// 可选的录制与回放身份；提供方与传输层忽略它。
    pub correlation: Option<Correlation>,
}

impl GenerationRequest {
    /// 以默认参数面向 `model` 的请求。
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            params: GenerationParams::default(),
            ..Self::default()
        }
    }

    /// 用系统提示词为对话播种。
    pub fn with_system(mut self, instructions: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::system(instructions));
        self
    }

    /// 追加一个用户轮。
    pub fn with_user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::user(content));
        self
    }

    /// 原样追加一轮。
    pub fn with_message(mut self, message: ChatMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// 扩展对话。
    pub fn with_messages(mut self, messages: impl IntoIterator<Item = ChatMessage>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// 替换工具列表。
    pub fn with_tools(mut self, tools: Vec<ToolDescriptor>) -> Self {
        self.tools = tools;
        self
    }

    /// 追加一个工具。
    pub fn with_tool(mut self, tool: ToolDescriptor) -> Self {
        self.tools.push(tool);
        self
    }

    /// 设置工具选择策略。
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// 为调用打上稳定的智能体与轮身份标签，用于录制与回放。
    pub fn with_correlation(mut self, agent: impl Into<String>, turn: usize) -> Self {
        self.correlation = Some(Correlation {
            agent: agent.into(),
            turn,
        });
        self
    }

    /// 替换采样参数。
    ///
    /// 这会覆盖先前 `with_max_tokens` 调用设置的任何值，
    /// 因此先应用参数块，再应用各个单独覆盖。
    pub fn with_params(mut self, params: GenerationParams) -> Self {
        self.params = params;
        self
    }

    /// 切换到贪心解码。
    pub fn deterministic(mut self) -> Self {
        self.params = GenerationParams::deterministic();
        self
    }

    /// 为助手轮设置上限。
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.params.max_output_tokens = Some(tokens);
        self
    }

    /// 丢弃系统消息并返回其余部分，例如用于重新适配窗口。
    pub fn without_system(mut self) -> Self {
        self.messages.retain(|message| !message.role.is_system());
        self
    }

    /// 第一条系统消息（若存在）。
    pub fn system_prompt(&self) -> Option<&str> {
        self.messages
            .iter()
            .find(|message| message.role == Role::System)
            .map(|message| message.content.as_str())
    }

    /// 最近的用户文本。
    pub fn last_user(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .map(|message| message.content.as_str())
    }

    /// 粗略的提示词大小，用于在发起调用前做预算检查。
    pub fn estimated_prompt_tokens(&self) -> usize {
        self.messages.iter().map(ChatMessage::tokens).sum::<usize>()
            + self.tools.iter().map(tool_schema_size).sum::<usize>()
    }

    /// 所附工具的名称。
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|tool| tool.name.as_str()).collect()
    }
}

fn tool_schema_size(tool: &ToolDescriptor) -> usize {
    crate::agent_llm::estimate_tokens(
        tool.name.len() + tool.description.len() + tool.parameters.to_string().len(),
    )
}

/// 对选定请求字段的命名覆盖，由智能体构建器与 [`crate::agent_llm::ChatClient`]
/// 应用，无需克隆整个请求。
#[derive(Clone, Debug, Default)]
pub struct RequestPatch {
    /// 模型覆盖。
    pub model: Option<String>,
    /// 完整参数替换。
    pub params: Option<GenerationParams>,
    /// 输出上限覆盖。
    pub max_output_tokens: Option<u32>,
    /// 按对象方式合并进 `params.extensions`。
    pub extra: Value,
}

impl RequestPatch {
    /// 只更改模型的补丁。
    pub fn model(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..Self::default()
        }
    }

    /// 将此补丁合并进请求。
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
        if let (Value::Object(base), Value::Object(extra)) =
            (&mut request.params.extensions, &self.extra)
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
