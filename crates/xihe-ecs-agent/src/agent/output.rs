//! 从本质上是一段散文的回答中取出可用数据。
//!
//! 被要求输出 JSON 的模型常常产出 JSON *外加*一段解释，或把它
//! 包在围栏块里，或发出一个多余的尾逗号。[`ResponseFormat`] 持有
//! 一个智能体的契约——自由文本或一个 schema——而 [`extract_json`]
//! 做读取中宽容的那部分。当回答无法解析时，格式产出 ReAct 循环
//! 发回的修复消息，使模型在它自己的错误面前再获一次机会。

use serde_json::Value;

use crate::agent_llm::ChatMessage;

use crate::agent::AgentError;

/// 应如何读取智能体的最终回答。
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ResponseFormat {
    /// 散文即回答；无需解析。
    #[default]
    Text,
    /// 回答必须是匹配 `schema` 的一个 JSON 值。
    Json {
        /// 调用者将据以校验的 JSON Schema。
        schema: Value,
        /// 解析失败时额外授予的修复机会。
        repairs: usize,
    },
}

impl ResponseFormat {
    /// 要求匹配 `schema` 的 JSON，带 [`ResponseFormat::DEFAULT_REPAIRS`] 次修复机会。
    pub fn json(schema: Value) -> Self {
        Self::Json {
            schema,
            repairs: Self::DEFAULT_REPAIRS,
        }
    }

    /// 要求匹配 `schema` 的 JSON，至多修复 `repairs` 次。
    pub fn with_repairs(self, repairs: usize) -> Self {
        match self {
            Self::Json { schema, .. } => Self::Json { schema, repairs },
            other => other,
        }
    }

    /// 一个 JSON 回答允许的修复机会。
    pub const DEFAULT_REPAIRS: usize = 2;

    /// 给系统提示词的额外指令；散文时为 `None`。
    pub fn instruction(&self) -> Option<String> {
        match self {
            Self::Text => None,
            Self::Json { schema, .. } => Some(format!(
                "Answer with one JSON value and nothing else. It must match this schema: {schema}"
            )),
        }
    }

    /// 此格式剩余的修复机会。
    pub fn repairs(&self) -> usize {
        match self {
            Self::Text => 0,
            Self::Json { repairs, .. } => *repairs,
        }
    }

    /// 循环是否应保留原始文本作为回答，而非解析。
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text)
    }

    /// 以此格式读取 `content`。
    ///
    /// `Ok(None)` 意为“散文，无需检查”；`Ok(Some(value))` 是解析后的
    /// JSON 回答；错误解释该把什么发回给模型。
    pub fn parse(&self, content: &str) -> Result<Option<Value>, AgentError> {
        match self {
            Self::Text => Ok(None),
            Self::Json { .. } => parse_json(content).map(Some),
        }
    }

    /// 向模型展示它自己那份不可读回答的消息。
    pub fn repair(&self, error: &AgentError) -> ChatMessage {
        let detail = match error {
            AgentError::Output { message, raw } => {
                format!("{message}. The answer was: {raw}")
            }
            other => other.to_string(),
        };
        ChatMessage::user(format!(
            "That answer could not be used: {detail}. Reply again with only the JSON value, \
             no explanation and no code fences."
        ))
    }
}

/// 从回答中解析 JSON，容忍散文与代码围栏。
pub fn parse_json(text: &str) -> Result<Value, AgentError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(AgentError::output("the answer is empty", text));
    }
    if let Ok(value) = serde_json::from_str(trimmed) {
        return Ok(value);
    }
    for candidate in candidates(trimmed) {
        if let Ok(value) = serde_json::from_str(candidate) {
            return Ok(value);
        }
    }
    Err(AgentError::output(
        "no JSON value could be read from the answer",
        text,
    ))
}

/// 值得尝试的子串，最可能的在前。
fn candidates(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    if let Some(fenced) = strip_fence(text) {
        out.push(fenced);
    }
    if let Some(balanced) = find_balanced(text)
        && !out.contains(&balanced)
    {
        out.push(balanced);
    }
    out
}

fn strip_fence(text: &str) -> Option<&str> {
    let body = text.strip_prefix("```")?;
    let body = body.strip_prefix("json").unwrap_or(body);
    let end = body.find("```")?;
    Some(body[..end].trim())
}

/// 第一个配平的 `{…}` 或 `[…]` 跨度，尊重字符串转义。
fn find_balanced(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let start = bytes
        .iter()
        .position(|byte| *byte == b'{' || *byte == b'[')?;
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, byte) in bytes[start..].iter().copied().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            byte if byte == open => depth += 1,
            byte if byte == close => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..start + offset + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

/// 宽松的 schema 检查：值是否携带 schema 所要求的键？
///
/// 这不是一个 JSON Schema 校验器。它回答智能体循环真正
/// 关心的问题——“模型是否漏了一个字段？”——而无需引入
/// 校验器 crate，并一次性报告每个缺失的键，使一轮修复
/// 就能把它们全部补齐。
pub fn missing_required_keys(schema: &Value, value: &Value) -> Vec<String> {
    let Some(object) = value.as_object() else {
        return Vec::new();
    };
    schema
        .get("required")
        .and_then(Value::as_array)
        .map(|required| {
            required
                .iter()
                .filter_map(Value::as_str)
                .filter(|key| !object.contains_key(*key))
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_plain_answer_parses() {
        assert_eq!(parse_json(r#"{"a": 1}"#).unwrap(), json!({"a": 1}));
    }

    #[test]
    fn prose_around_the_json_is_ignored() {
        let text = "Sure! Here is the result:\n{\"a\": 1}\nHope that helps.";
        assert_eq!(parse_json(text).unwrap(), json!({"a": 1}));
    }

    #[test]
    fn a_code_fence_is_stripped() {
        let text = "```json\n{\"a\": [1, 2]}\n```";
        assert_eq!(parse_json(text).unwrap(), json!({"a": [1, 2]}));
    }

    #[test]
    fn braces_inside_strings_do_not_end_the_span() {
        let text = r#"noise {"a": "} still json"} tail"#;
        assert_eq!(parse_json(text).unwrap(), json!({"a": "} still json"}));
    }

    #[test]
    fn an_unparseable_answer_reports_what_was_seen() {
        let error = parse_json("I cannot help with that").expect_err("not JSON");
        let AgentError::Output { message, raw } = &error else {
            panic!("expected an output failure, got {error:?}");
        };
        assert!(message.contains("no JSON value"), "{message}");
        assert_eq!(raw, "I cannot help with that");
    }

    #[test]
    fn an_empty_answer_is_its_own_reason() {
        let error = parse_json("   ").expect_err("empty");
        assert!(error.to_string().contains("empty"), "{error}");
    }

    #[test]
    fn text_format_never_parses() {
        assert!(ResponseFormat::Text.parse("anything").unwrap().is_none());
        assert_eq!(ResponseFormat::Text.instruction(), None);
        assert_eq!(ResponseFormat::Text.repairs(), 0);
    }

    #[test]
    fn json_format_states_its_contract() {
        let format = ResponseFormat::json(json!({"type": "object"}));
        let instruction = format.instruction().expect("json has one");
        assert!(instruction.contains(r#""type":"object""#), "{instruction}");
        assert_eq!(format.repairs(), ResponseFormat::DEFAULT_REPAIRS);
        assert_eq!(format.with_repairs(0).repairs(), 0);
    }

    #[test]
    fn a_repair_message_shows_the_model_its_answer() {
        let format = ResponseFormat::json(json!({}));
        let error = parse_json("not json at all").expect_err("bad");
        let message = format.repair(&error);
        assert!(message.content.contains("not json at all"));
        assert!(message.content.contains("only the JSON value"));
    }

    #[test]
    fn missing_keys_are_listed_together() {
        let schema = json!({"required": ["answer", "confidence"]});
        let value = json!({"answer": 42});
        assert_eq!(missing_required_keys(&schema, &value), ["confidence"]);
        assert!(
            missing_required_keys(&schema, &json!({"answer": 1, "confidence": 0.9})).is_empty()
        );
        assert!(
            missing_required_keys(&schema, &Value::String("nope".into())).is_empty(),
            "a non-object has no keys to compare, and the caller's parse step reports it"
        );
    }

    #[test]
    fn a_truncated_span_is_not_rescued() {
        assert!(find_balanced("{\"a\": 1").is_none());
        assert_eq!(
            find_balanced("] leading close then {\"a\":1}"),
            Some("{\"a\":1}")
        );
    }
}
