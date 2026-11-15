//! Getting usable data out of an answer that is, at heart, prose.
//!
//! A model asked for JSON often produces JSON *plus* an explanation, or wraps it
//! in a fenced block, or emits a stray trailing comma. [`ResponseFormat`] holds
//! the contract for one agent — free text or a schema — and [`extract_json`]
//! does the forgiving part of reading it. When the answer cannot be parsed, the
//! format produces the repair message that the ReAct loop sends back, so the
//! model gets one more chance with its own mistake in front of it.

use serde_json::Value;

use xihe_llm::ChatMessage;

use crate::AgentError;

/// How an agent's final answer should be read.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ResponseFormat {
    /// Prose is the answer; nothing to parse.
    #[default]
    Text,
    /// The answer must be one JSON value matching `schema`.
    Json {
        /// JSON Schema the caller will validate against.
        schema: Value,
        /// Extra repair chances granted when parsing fails.
        repairs: usize,
    },
}

impl ResponseFormat {
    /// Ask for JSON matching `schema`, with [`ResponseFormat::DEFAULT_REPAIRS`] repair shots.
    pub fn json(schema: Value) -> Self {
        Self::Json {
            schema,
            repairs: Self::DEFAULT_REPAIRS,
        }
    }

    /// Ask for JSON matching `schema`, repairing at most `repairs` times.
    pub fn with_repairs(self, repairs: usize) -> Self {
        match self {
            Self::Json { schema, .. } => Self::Json { schema, repairs },
            other => other,
        }
    }

    /// Repair chances allowed for a JSON answer.
    pub const DEFAULT_REPAIRS: usize = 2;

    /// Extra instructions for the system prompt; `None` for prose.
    pub fn instruction(&self) -> Option<String> {
        match self {
            Self::Text => None,
            Self::Json { schema, .. } => Some(format!(
                "Answer with one JSON value and nothing else. It must match this schema: {schema}"
            )),
        }
    }

    /// Repair chances left for this format.
    pub fn repairs(&self) -> usize {
        match self {
            Self::Text => 0,
            Self::Json { repairs, .. } => *repairs,
        }
    }

    /// Whether the loop should keep the raw text as the answer instead of parsing.
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text)
    }

    /// Read `content` as this format.
    ///
    /// `Ok(None)` means "prose, nothing to check"; `Ok(Some(value))` is a parsed
    /// JSON answer; the error explains what to send back to the model.
    pub fn parse(&self, content: &str) -> Result<Option<Value>, AgentError> {
        match self {
            Self::Text => Ok(None),
            Self::Json { .. } => parse_json(content).map(Some),
        }
    }

    /// The message that shows the model its own unreadable answer.
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

/// Parse JSON out of an answer, tolerating prose and code fences.
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

/// The substrings worth trying, most likely first.
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

/// The first balanced `{…}` or `[…]` span, honouring string escapes.
fn find_balanced(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|byte| *byte == b'{' || *byte == b'[')?;
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

/// Loose schema checking: does the value carry the keys the schema requires?
///
/// This is not a JSON Schema validator. It answers the question an agent loop
/// actually has — "did the model forget a field?" — without pulling in a
/// validator crate, and reports every missing key at once so one repair round
/// can fix all of them.
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
        assert!(missing_required_keys(&schema, &json!({"answer": 1, "confidence": 0.9})).is_empty());
        assert!(
            missing_required_keys(&schema, &Value::String("nope".into())).is_empty(),
            "a non-object has no keys to compare, and the caller's parse step reports it"
        );
    }

    #[test]
    fn a_truncated_span_is_not_rescued() {
        assert!(find_balanced("{\"a\": 1").is_none());
        assert_eq!(find_balanced("] leading close then {\"a\":1}"), Some("{\"a\":1}"));
    }
}
