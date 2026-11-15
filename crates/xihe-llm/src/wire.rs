//! `/chat/completions` 传输方言，其中不含 HTTP。
//!
//! 本模块总是被编译——`openai` feature 只在 [`crate::openai`] 中
//! 添加传输——因此即使离线构建，请求编码、响应解码与错误映射
//! 也保持被测试。

use std::time::Duration;

use serde_json::{Value, json};

use crate::error::LlmError;
use crate::message::{
    ChatMessage, FinishReason, StreamChunk, TokenUsage, ToolCall, ToolChoice,
};
use crate::request::GenerationRequest;
use crate::GenerationResponse;

/// 按聊天补全服务器期望的方式渲染请求。
///
/// 当请求把 `model` 留空时使用 `default_model`，这正是
/// [`crate::ChatClient`] 让客户端持有模型名的方式。
pub fn to_wire(request: &GenerationRequest, default_model: &str) -> Value {
    let model = if request.model.is_empty() {
        default_model
    } else {
        &request.model
    };
    let messages: Vec<Value> = request
        .messages
        .iter()
        .map(to_wire_message)
        .collect();
    let mut body = json!({
        "model": model,
        "messages": messages,
        "temperature": request.params.temperature,
    });
    if let Some(top_p) = request.params.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(max) = request.params.max_output_tokens {
        // 一些较新的服务器重命名了此字段；`extensions` 可以覆盖它，
        // 因此只填入尚不存在的那种拼写。
        if request.params.extensions.get("max_completion_tokens").is_none() {
            body["max_tokens"] = json!(max);
        }
    }
    if !request.params.stop.is_empty() {
        body["stop"] = json!(request.params.stop);
    }
    if let Some(seed) = request.params.seed {
        body["seed"] = json!(seed);
    }
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(
            request
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        },
                    })
                })
                .collect(),
        );
        body["tool_choice"] = tool_choice_to_wire(&request.tool_choice);
    } else if request.tool_choice != ToolChoice::Auto {
        // 即使没有工具，强制选择仍有意义：它表示“以文本作答”。
        body["tool_choice"] = tool_choice_to_wire(&request.tool_choice);
    }
    // 提供方特有字段被扁平合并：`reasoning_effort`、`response_format`、
    // `enable_thinking` 之类都放在这里。
    if let Value::Object(extra) = &request.params.extensions {
        for (key, value) in extra {
            body[key] = value.clone();
        }
    }
    body
}

fn tool_choice_to_wire(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Only(name) => json!({
            "type": "function",
            "function": { "name": name },
        }),
    }
}

fn to_wire_message(message: &ChatMessage) -> Value {
    let mut value = json!({
        "role": message.role.as_str(),
        "content": message.content,
    });
    if let Some(name) = &message.name {
        value["name"] = json!(name);
    }
    if let Some(call_id) = &message.tool_call_id {
        value["tool_call_id"] = json!(call_id);
    }
    if !message.tool_calls.is_empty() {
        value["tool_calls"] = Value::Array(
            message
                .tool_calls
                .iter()
                .map(|call| {
                    json!({
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            // 协议要求参数为 JSON *字符串*。
                            "arguments": call.arguments_string(),
                        },
                    })
                })
                .collect(),
        );
    }
    value
}

/// 解码一个非流式补全。
pub fn parse_completion(value: Value) -> Result<GenerationResponse, LlmError> {
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .cloned()
        .ok_or_else(|| LlmError::decode("response has no choices"))?;
    let message = choice
        .get("message")
        .or_else(|| choice.get("delta"))
        .cloned()
        .ok_or_else(|| LlmError::decode("choice carries no message"))?;
    let usage = parse_usage(&value).unwrap_or_default();
    let mut response = GenerationResponse {
        content: message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        model: value
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        usage,
        raw: Some(value),
        ..GenerationResponse::default()
    };
    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
        response.finish_reason = Some(FinishReason::parse(reason));
    }
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let function = call.get("function");
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let arguments = function
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
                .unwrap_or("{}");
            let id = call.get("id").and_then(Value::as_str).unwrap_or("");
            let (decoded, _problem) = ToolCall::from_json_text(id, name, arguments);
            response.tool_calls.push(decoded);
        }
        if !response.tool_calls.is_empty() {
            response
                .finish_reason
                .get_or_insert(FinishReason::ToolCalls);
        }
    }
    if response.finish_reason.is_none() {
        response.finish_reason = Some(if response.calls_tools() {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        });
    }
    Ok(response)
}

fn parse_usage(value: &Value) -> Option<TokenUsage> {
    let usage = value.get("usage")?.as_object()?;
    let field = |name: &str| {
        usage
            .get(name)
            .and_then(Value::as_u64)
            // 一些网关把 token 报告为字符串。
            .or_else(|| {
                usage
                    .get(name)
                    .and_then(Value::as_str)
                    .and_then(|text| text.parse::<u64>().ok())
            })
            .unwrap_or(0)
    };
    Some(TokenUsage {
        prompt_tokens: field("prompt_tokens"),
        completion_tokens: field("completion_tokens"),
        total_tokens: field("total_tokens"),
    })
}

/// 从不断增长的缓冲区中取出下一个 server-sent-events 载荷。
///
/// 当尚无完整行被缓冲时返回 `None`；对保活行与事件分隔符返回空字符串，
/// 调用者将其解码为“这里没有内容”。
pub fn take_sse_line(buffer: &mut String) -> Option<String> {
    let newline = buffer.find('\n')?;
    let line = buffer[..newline].trim().to_string();
    buffer.drain(..=newline);
    if line.is_empty() || line.starts_with(':') {
        return Some(String::new());
    }
    match line.strip_prefix("data:") {
        Some(data) => Some(data.trim().to_string()),
        // 一些代理发出不带 `data:` 前缀的裸 JSON。
        None => Some(line),
    }
}

/// 解码一个流式事件。`Ok(None)` 表示“此帧无内容”。
pub fn parse_stream_event(data: &str) -> Result<Option<StreamChunk>, LlmError> {
    if data.is_empty() {
        return Ok(None);
    }
    if data == "[DONE]" {
        return Ok(Some(StreamChunk {
            finish_reason: Some(FinishReason::Stop),
            ..StreamChunk::default()
        }));
    }
    let value: Value = serde_json::from_str(data)
        .map_err(|error| LlmError::decode(format!("stream frame is not json: {error}")))?;
    if let Some(error) = value.get("error") {
        let status = error
            .get("status")
            .or_else(|| error.get("code"))
            .and_then(Value::as_u64)
            .filter(|code| (100..600).contains(code))
            .map_or(500, |code| code as u16);
        return Err(LlmError::api(status, error.to_string()));
    }
    let mut chunk = StreamChunk {
        usage: parse_usage(&value),
        ..StreamChunk::default()
    };
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first());
    let Some(choice) = choice else {
        return Ok(chunk.usage.map(|_| chunk));
    };
    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
        chunk.finish_reason = Some(FinishReason::parse(reason));
    }
    let delta = choice.get("delta").or_else(|| choice.get("message"));
    if let Some(delta) = delta {
        chunk.text = delta
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let function = call.get("function");
                let name = function
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if name.is_empty() {
                    // 属于较早一帧的参数片段；需要完整参数的调用者
                    // 应改用 `generate`。
                    continue;
                }
                let arguments = function
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let id = call.get("id").and_then(Value::as_str).unwrap_or("");
                let (decoded, _) = ToolCall::from_json_text(id, name, arguments);
                chunk.tool_calls.push(decoded);
            }
        }
    }
    Ok(Some(chunk))
}

/// 对一个非成功的 HTTP 应答分类。
///
/// `retry_after` 是服务器的 `Retry-After` 提示，由传输层从响应头部解析；
/// 它只在 429 上携带，使 [`crate::ChatClient`] 能等待服务器要求的延迟，
/// 而非自己的退避。
pub fn error_from_status(status: u16, body: String, retry_after: Option<Duration>) -> LlmError {
    let message = extract_api_message(&body).unwrap_or_else(|| body.trim().to_string());
    match status {
        401 | 403 => LlmError::Auth { message },
        429 => LlmError::RateLimited { retry_after },
        _ => LlmError::Api { status, message },
    }
}

/// 解析以 delta-seconds 表示的 `Retry-After` 头部。
///
/// 不采纳 HTTP-date 形式：不是整数秒的值产出 `None`，
/// 调用者回退到自己的退避。
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}

/// 优先取 `error.message` 而非整个体，使日志保持可读。
pub fn extract_api_message(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let message = value.get("error")?;
    if let Value::String(text) = message {
        return Some(text.clone());
    }
    message
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| Some(message.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{GenerationParams, Role, ToolDescriptor};

    fn sample() -> GenerationRequest {
        GenerationRequest::new("test-model")
            .with_system("be brief")
            .with_user("what is ecs?")
            .with_tools(vec![
                ToolDescriptor::new("search", "look it up")
                    .with_required_string("q", "query")
            ])
            .with_params(GenerationParams {
                top_p: Some(0.9),
                seed: Some(7),
                stop: vec!["\n\n".to_string()],
                extensions: json!({ "reasoning_effort": "low" }),
                ..GenerationParams::default()
            })
            .with_max_tokens(128)
    }

    #[test]
    fn wire_body_carries_every_field() {
        let body = to_wire(&sample(), "fallback");
        assert_eq!(body["model"], json!("test-model"));
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        assert_eq!(body["messages"][0]["role"], json!("system"));
        assert_eq!(body["messages"][1]["content"], json!("what is ecs?"));
        assert_eq!(
            body["temperature"],
            json!(0.7),
            "sampling values survive the trip to JSON unchanged"
        );
        assert_eq!(body["top_p"], json!(0.9));
        assert_eq!(body["max_tokens"], json!(128));
        assert_eq!(body["seed"], json!(7));
        assert_eq!(body["stop"], json!(["\n\n"]));
        assert_eq!(
            body["reasoning_effort"], json!("low"),
            "extensions are merged flat into the body"
        );
        assert_eq!(body["tools"][0]["function"]["name"], json!("search"));
        assert_eq!(
            body["tools"][0]["function"]["parameters"]["required"],
            json!(["q"])
        );
        assert_eq!(body["tool_choice"], json!("auto"));
    }

    #[test]
    fn omitted_fields_stay_omitted() {
        let body = to_wire(&GenerationRequest::new("").with_user("hi"), "home");
        assert_eq!(body["model"], json!("home"), "an empty model uses the default");
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("top_p").is_none());
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("seed").is_none());
    }

    #[test]
    fn correlation_never_reaches_the_wire() {
        let request = GenerationRequest::new("m")
            .with_user("hi")
            .with_correlation("scribe", 3);
        let body = to_wire(&request, "m");
        assert!(
            body.get("correlation").is_none(),
            "record-and-replay metadata must not be sent to the provider: {body}"
        );
        assert!(body.get("agent").is_none());
        assert!(body.get("turn").is_none());
    }

    #[test]
    fn tool_choice_without_tools_is_still_sent() {
        let request = GenerationRequest::new("m")
            .with_user("hi")
            .with_tool_choice(ToolChoice::None);
        assert_eq!(to_wire(&request, "m")["tool_choice"], json!("none"));
    }

    #[test]
    fn assistant_tool_calls_round_trip_as_strings() {
        let message = ChatMessage::assistant("").with_tool_calls(vec![ToolCall::new(
            "call_9",
            "search",
            json!({ "q": "ecs" }),
        )]);
        let body = to_wire(&GenerationRequest::new("m").with_message(message), "m");
        let call = &body["messages"][0]["tool_calls"][0];
        assert_eq!(call["id"], json!("call_9"));
        assert_eq!(call["type"], json!("function"));
        assert_eq!(call["function"]["arguments"], json!("{\"q\":\"ecs\"}"));
    }

    #[test]
    fn tool_results_carry_their_call_id() {
        let request =
            GenerationRequest::new("m").with_message(ChatMessage::tool_result("call_1", json!("sunny")));
        let body = to_wire(&request, "m");
        assert_eq!(body["messages"][0]["role"], json!("tool"));
        assert_eq!(body["messages"][0]["tool_call_id"], json!("call_1"));
        assert_eq!(body["messages"][0]["content"], json!("sunny"));
    }

    #[test]
    fn named_authors_are_preserved() {
        let request = GenerationRequest::new("m")
            .with_message(ChatMessage::assistant("done").with_name("researcher"));
        assert_eq!(to_wire(&request, "m")["messages"][0]["name"], json!("researcher"));
    }

    #[test]
    fn completion_parses_text_usage_and_model() {
        let value = json!({
            "model": "gpt-test",
            "choices": [{
                "message": { "role": "assistant", "content": "ECS is a pattern." },
                "finish_reason": "stop",
            }],
            "usage": { "prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16 },
        });
        let response = parse_completion(value).unwrap();
        assert_eq!(response.content, "ECS is a pattern.");
        assert_eq!(response.model, "gpt-test");
        assert_eq!(response.usage, TokenUsage::new(11, 5));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.step(), crate::GenerationStep::Answer);
        assert!(response.raw.is_some(), "the payload stays available for debugging");
    }

    #[test]
    fn completion_parses_tool_calls_with_stringy_arguments() {
        let value = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "search", "arguments": "{\"q\":\"tokio\"}" },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        });
        let response = parse_completion(value).unwrap();
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "call_1");
        assert_eq!(response.tool_calls[0].string_argument("q"), Some("tokio"));
        assert_eq!(response.step(), crate::GenerationStep::ExecuteTools);
    }

    #[test]
    fn completion_infers_a_missing_finish_reason() {
        let with_calls = json!({
            "choices": [{ "message": { "tool_calls": [
                { "id": "c", "function": { "name": "t", "arguments": "{}" } }
            ] } }],
        });
        let response = parse_completion(with_calls).unwrap();
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        let plain = json!({ "choices": [{ "message": { "content": "hi" } }] });
        assert_eq!(
            parse_completion(plain).unwrap().finish_reason,
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn stringified_usage_is_accepted() {
        let value = json!({
            "choices": [{ "message": { "content": "hi" }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": "12", "completion_tokens": "3", "total_tokens": "15" },
        });
        assert_eq!(parse_completion(value).unwrap().usage, TokenUsage::new(12, 3));
    }

    #[test]
    fn malformed_payloads_become_decode_errors() {
        for value in [json!({}), json!({ "choices": [] }), json!({ "choices": [{}] })] {
            let error = parse_completion(value).unwrap_err();
            assert_eq!(error.kind(), "decode", "a broken body is a protocol failure");
        }
    }

    #[test]
    fn sse_lines_are_framed_without_splitting_events() {
        let mut buffer = "data: {\"a\":1}\n\ndata: {\"b\":2}\n\n".to_string();
        assert_eq!(take_sse_line(&mut buffer).unwrap(), "{\"a\":1}");
        assert_eq!(
            take_sse_line(&mut buffer).unwrap(),
            "",
            "a blank separator carries no payload"
        );
        assert_eq!(take_sse_line(&mut buffer).unwrap(), "{\"b\":2}");
        assert_eq!(take_sse_line(&mut buffer).unwrap(), "");
        assert!(
            take_sse_line(&mut buffer).is_none(),
            "a partial line must stay buffered"
        );
        // 跨两次网络读取被拆分的事件会重新合并为一帧。
        let mut buffer = String::new();
        buffer.push_str("data: {\"choices\":[{\"del");
        assert_eq!(take_sse_line(&mut buffer), None);
        buffer.push_str("ta\":{\"content\":\"Hi\"}}]}\n");
        let frame = take_sse_line(&mut buffer).unwrap();
        assert_eq!(
            parse_stream_event(&frame)
                .unwrap()
                .unwrap()
                .text,
            "Hi"
        );
    }

    #[test]
    fn keep_alive_and_done_are_handled() {
        let mut buffer = ": keep-alive\ndata: [DONE]\n".to_string();
        assert_eq!(take_sse_line(&mut buffer).unwrap(), "", "comments are skipped");
        assert_eq!(take_sse_line(&mut buffer).unwrap(), "[DONE]");
        let last = parse_stream_event("[DONE]").unwrap().unwrap();
        assert_eq!(last.finish_reason, Some(FinishReason::Stop));
        assert!(last.text.is_empty());
    }

    #[test]
    fn stream_events_decode_deltas_and_usage() {
        let chunk = parse_stream_event(r#"{"choices":[{"delta":{"content":"Hel"}}]}"#)
            .unwrap()
            .unwrap();
        assert_eq!(chunk.text, "Hel");
        assert!(!chunk.is_last());
        let final_frame = parse_stream_event(
            r#"{"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(final_frame.finish_reason, Some(FinishReason::Length));
        assert_eq!(final_frame.usage, Some(TokenUsage::new(2, 3)));
        assert!(parse_stream_event("").unwrap().is_none());
        // 仅含用量的帧仍携带计量，不附带文本。
        let usage_only = parse_stream_event(
            r#"{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
        )
        .unwrap()
        .expect("usage survives without a delta");
        assert_eq!(usage_only.usage, Some(TokenUsage::new(1, 1)));
        assert!(usage_only.text.is_empty());
    }

    #[test]
    fn stream_tool_calls_survive_their_fragment_frames() {
        let chunk = parse_stream_event(
            r#"{"choices":[{"delta":{"tool_calls":[{"id":"c1","function":{"name":"search","arguments":"{\"q\":\"x\"}"}}]}}]}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(chunk.tool_calls.len(), 1);
        assert_eq!(chunk.tool_calls[0].string_argument("q"), Some("x"));
        let fragment = parse_stream_event(
            r#"{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"more"}}]}}]}"#,
        )
        .unwrap()
        .unwrap();
        assert!(
            fragment.tool_calls.is_empty(),
            "a fragment with no name is not a new call"
        );
    }

    #[test]
    fn stream_errors_are_surfaced_to_the_consumer() {
        let error = parse_stream_event(r#"{"error":{"message":"context length"}}"#).unwrap_err();
        assert_eq!(error.kind(), "api");
        assert!(error.to_string().contains("context length"));
        assert_eq!(
            parse_stream_event("not json").unwrap_err().kind(),
            "decode"
        );
    }

    #[test]
    fn status_codes_map_to_the_right_failure_class() {
        let body = r#"{"error":{"message":"You exceeded your current quota"}}"#;
        assert!(matches!(
            error_from_status(429, body.to_string(), None),
            LlmError::RateLimited { retry_after: None }
        ));
        let LlmError::Auth { message } = error_from_status(401, body.to_string(), None) else {
            panic!("401 must be auth");
        };
        assert_eq!(
            message, "You exceeded your current quota",
            "only the api message is kept"
        );
        let LlmError::Api { status, .. } = error_from_status(500, "boom".to_string(), None) else {
            panic!("500 must be an api error");
        };
        assert_eq!(status, 500);
        assert!(extract_api_message("not json").is_none());
        assert_eq!(
            extract_api_message(r#"{"error":"plain string"}"#).as_deref(),
            Some("plain string")
        );
    }

    #[test]
    fn a_429_carries_the_server_retry_hint() {
        let LlmError::RateLimited { retry_after } =
            error_from_status(429, "{}".to_string(), Some(Duration::from_secs(7)))
        else {
            panic!("429 must be rate limited");
        };
        assert_eq!(retry_after, Some(Duration::from_secs(7)));
        // 该提示只在 429 上有意义；其他类别忽略它。
        assert!(matches!(
            error_from_status(500, "{}".to_string(), Some(Duration::from_secs(7))),
            LlmError::Api { .. }
        ));
    }

    #[test]
    fn retry_after_parses_delta_seconds_only() {
        assert_eq!(parse_retry_after("12"), Some(Duration::from_secs(12)));
        assert_eq!(parse_retry_after("  3 \n"), Some(Duration::from_secs(3)));
        assert_eq!(parse_retry_after("0"), Some(Duration::ZERO));
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), None);
        assert_eq!(parse_retry_after(""), None);
        assert_eq!(parse_retry_after("-1"), None);
    }

    #[test]
    fn roles_use_the_wire_spellings() {
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(Role::Assistant.as_str(), "assistant");
    }
}
