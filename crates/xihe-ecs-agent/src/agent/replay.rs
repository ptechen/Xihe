//! 对一个世界的模型调用与工具结果进行录制与回放。
//!
//! [`Journal`] 捕获一次运行产出的每个 [`GenerationResponse`]，以发起它的
//! 请求的确定性指纹为键，连同随后产生的工具结果。[`ReplayProvider`]
//! 从该日志而非实时模型作答，循环反馈录制的工具结果而非重新运行它们，
//! 因此第二次运行能逐轮复现第一次，即便原提供方是非确定性的、
//! 工具带有副作用。
//!
//! 指纹覆盖关联标识、消息与工具描述符（按名字排序），而非模型名：
//! 客户端在请求构建后才填入模型，因此以它为键会在录制与回放之间不同。
//! 已持有某请求工具结果的日志会回放它们；空日志则录制它们，
//! 因此同一资源处理两个方向。一个在人工审批处挂起的轮次，
//! 会在人工决定后把其结果记入日志，因此回放既不重新审批、
//! 也不重新运行已批准的工具。要回放一次运行，把已填充的 [`Journal`]
//! 安装为世界资源，并用共享它的 [`ReplayProvider`] 支撑
//! [`LlmService`](crate::agent::LlmService)。

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use crate::agent_llm::{GenerationRequest, GenerationResponse, LlmError, Provider, ToolCall};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::agent::error::AgentError;
use crate::agent::tool::ToolOutcome;

/// 模型响应与工具结果的共享日志，存为世界资源。
///
/// 克隆一个日志会共享同一批条目，因此安装在录制世界中的句柄
/// 与交给 [`ReplayProvider`] 的句柄看到的是同一份日志。
#[derive(Clone, Default)]
pub struct Journal {
    entries: Arc<Mutex<HashMap<String, GenerationResponse>>>,
    tools: Arc<Mutex<HashMap<String, ToolRecord>>>,
    pending: Arc<Mutex<HashMap<String, PendingApproval>>>,
}

/// 跟随一次模型响应的工具结果。
#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct ToolRecord {
    results: Vec<(ToolCall, ToolOutcome)>,
    refused: usize,
}

/// 一次运行中途挂起的审批：发起它的请求，加上已在手的
/// 工具结果，保留以便恢复时能把整轮记入日志。
/// 刻意不序列化：仍挂起时保存的日志是不完整的录制，
/// 回放它会再次挂起，而非猜测。
#[derive(Clone)]
struct PendingApproval {
    request: GenerationRequest,
    done: Vec<(ToolCall, ToolOutcome)>,
    refused: usize,
}

/// 日志的磁盘形式。`BTreeMap` 使字节布局确定
/// （独立于实时映射的哈希顺序），因此同一运行的两次录制
/// 序列化完全相同，日志可被 diff 或审计。
#[derive(Serialize, Deserialize)]
struct JournalData {
    entries: BTreeMap<String, GenerationResponse>,
    tools: BTreeMap<String, ToolRecord>,
}

impl Journal {
    /// 一个空日志。
    pub fn new() -> Self {
        Self::default()
    }

    /// 以 `request` 的指纹为键存储 `response`。
    ///
    /// 同一指纹已录有*不同*响应时返回错误而非覆盖：两个智能体共享
    /// 名字会让其关联标识（名字+轮）相撞，使非确定性模型的两次作答
    /// 落到同一个键上。无声覆盖会让回放静默发散，故冲突在录制时
    /// 就响亮失败。重复录入*相同*响应是幂等的。
    pub fn record(
        &self,
        request: &GenerationRequest,
        response: &GenerationResponse,
    ) -> Result<(), AgentError> {
        let key = fingerprint(request);
        let mut entries = self.entries.lock().expect("journal lock");
        if let Some(existing) = entries.get(&key)
            && existing != response
        {
            return Err(collision_error(request, "response"));
        }
        entries.insert(key, response.clone());
        Ok(())
    }

    /// 为 `request` 录制的响应（若有）。
    pub fn lookup(&self, request: &GenerationRequest) -> Option<GenerationResponse> {
        self.entries
            .lock()
            .expect("journal lock")
            .get(&fingerprint(request))
            .cloned()
    }

    /// 存储跟随 `request` 的工具结果。
    ///
    /// 与 [`Journal::record`] 同一守卫：同一指纹已录有*不同*工具结果时
    /// 返回错误而非覆盖。确定性模型下两个同名智能体的响应逐字相同、
    /// 绕过响应守卫，但带副作用的工具仍可能产出不同结果——无声覆盖会
    /// 让回放把一个智能体的结果喂给两者。重复录入*相同*结果幂等。
    pub fn record_tools(
        &self,
        request: &GenerationRequest,
        results: Vec<(ToolCall, ToolOutcome)>,
        refused: usize,
    ) -> Result<(), AgentError> {
        let key = fingerprint(request);
        let record = ToolRecord { results, refused };
        let mut tools = self.tools.lock().expect("journal lock");
        if let Some(existing) = tools.get(&key)
            && existing != &record
        {
            return Err(collision_error(request, "tool result"));
        }
        tools.insert(key, record);
        Ok(())
    }

    /// 为 `request` 录制的工具结果（若有）。
    pub fn lookup_tools(
        &self,
        request: &GenerationRequest,
    ) -> Option<(Vec<(ToolCall, ToolOutcome)>, usize)> {
        self.tools
            .lock()
            .expect("journal lock")
            .get(&fingerprint(request))
            .map(|record| (record.results.clone(), record.refused))
    }

    /// 有多少个不同请求拥有已录制的模型响应。
    pub fn len(&self) -> usize {
        self.entries.lock().expect("journal lock").len()
    }

    /// 是否尚未录制任何内容。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 把整个日志序列化为字节，使一次录制可被持久化并在
    /// 另一进程中回放。字节是自包含的：指纹键随响应一起传输，
    /// 因此 [`Journal::from_bytes`] 无需其他东西即可回放该运行。
    pub fn to_bytes(&self) -> serde_json::Result<Vec<u8>> {
        let entries: BTreeMap<String, GenerationResponse> = self
            .entries
            .lock()
            .expect("journal lock")
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let tools: BTreeMap<String, ToolRecord> = self
            .tools
            .lock()
            .expect("journal lock")
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        serde_json::to_vec(&JournalData { entries, tools })
    }

    /// 从 [`Journal::to_bytes`] 的输出重建一个日志。结果与产出
    /// 这些字节的日志不共享任何状态，正如第二个进程从磁盘
    /// 加载该录制一样。
    pub fn from_bytes(bytes: &[u8]) -> serde_json::Result<Self> {
        let data: JournalData = serde_json::from_slice(bytes)?;
        Ok(Self {
            entries: Arc::new(Mutex::new(data.entries.into_iter().collect())),
            tools: Arc::new(Mutex::new(data.tools.into_iter().collect())),
            pending: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// 把日志持久化到文件，使一次录制可在另一进程中回放。
    /// 对 [`Journal::to_bytes`] 的薄包装；序列化错误以
    /// [`std::io::ErrorKind::InvalidData`] 浮现。
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let bytes = self
            .to_bytes()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        std::fs::write(path, bytes)
    }

    /// 加载由 [`Journal::save`] 写出的日志。结果与录制者不共享
    /// 任何状态，正如第二个进程读取该文件一样。
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    }

    /// 记住一个挂起的审批，使其恢复能把整轮记入日志。
    pub(crate) fn record_pending(
        &self,
        key: String,
        request: GenerationRequest,
        done: Vec<(ToolCall, ToolOutcome)>,
        refused: usize,
    ) {
        self.pending.lock().expect("journal lock").insert(
            key,
            PendingApproval {
                request,
                done,
                refused,
            },
        );
    }

    /// 取出 `key` 的挂起审批（若本日志录制过一个）。
    pub(crate) fn take_pending(
        &self,
        key: &str,
    ) -> Option<(GenerationRequest, Vec<(ToolCall, ToolOutcome)>, usize)> {
        self.pending
            .lock()
            .expect("journal lock")
            .remove(key)
            .map(|pending| (pending.request, pending.done, pending.refused))
    }
}

/// 挂起审批的稳定键：智能体、发起它的生成轮，以及调用 id。
/// 轮是关联标识的轮（模型调用的），而非恢复的轮计数，
/// 因此一轮中的每个审批共享同一个轮。
pub(crate) fn approval_key(agent: &str, turn: usize, call_id: &str) -> String {
    format!("{agent}:{turn}:{call_id}")
}

/// 请求的稳定键：其关联标识（存在时）、其消息，以及按名字
/// 排序的工具描述符，使描述符顺序无法改变键。模型名被刻意
/// 排除（见模块文档）。关联标识把发出相同请求的两个智能体分开。
fn fingerprint(request: &GenerationRequest) -> String {
    let mut tools = request.tools.clone();
    tools.sort_by(|a, b| a.name.cmp(&b.name));
    serde_json::json!({
        "correlation": request.correlation,
        "messages": request.messages,
        "tools": tools,
    })
    .to_string()
}

/// 指纹冲突的错误，带上肇事关联标识使操作员一眼看到哪个智能体撞了键。
fn collision_error(request: &GenerationRequest, what: &str) -> AgentError {
    let who = request
        .correlation
        .as_ref()
        .map(|c| format!("agent `{}` turn {}", c.agent, c.turn))
        .unwrap_or_else(|| "an uncorrelated request".to_string());
    AgentError::config(format!(
        "journal fingerprint collision: a different {what} is already recorded for {who}; \
         two agents likely share a name, so their correlation keys collide and replay would \
         silently diverge"
    ))
}

/// 从 [`Journal`] 而非实时模型作答的提供方。
///
/// 没有已录制响应的请求是一个错误，而非静默回退：
/// 偏离录制的回放应当响亮地失败。
pub struct ReplayProvider {
    journal: Journal,
}

impl ReplayProvider {
    /// 从 `journal` 回放。
    pub fn new(journal: Journal) -> Self {
        Self { journal }
    }
}

#[async_trait]
impl Provider for ReplayProvider {
    fn id(&self) -> &str {
        "replay"
    }

    fn default_model(&self) -> &str {
        "replay"
    }

    async fn generate(&self, request: &GenerationRequest) -> Result<GenerationResponse, LlmError> {
        self.journal.lookup(request).ok_or_else(|| {
            LlmError::unsupported("replay", "generate: no recorded response for this request")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_llm::{ChatMessage, ToolDescriptor};

    fn request(text: &str) -> GenerationRequest {
        GenerationRequest::new("m").with_message(ChatMessage::user(text))
    }

    #[test]
    fn a_cloned_journal_shares_one_log() {
        let journal = Journal::new();
        let other = journal.clone();
        journal
            .record(&request("hi"), &GenerationResponse::text("hello"))
            .expect("record");
        assert_eq!(other.len(), 1);
        assert_eq!(
            other.lookup(&request("hi")).expect("recorded").content,
            "hello"
        );
        assert!(other.lookup(&request("absent")).is_none());
    }

    #[test]
    fn the_fingerprint_ignores_model_and_tool_order() {
        let mut a = request("hi");
        a.tools = vec![
            ToolDescriptor::new("zebra", "z"),
            ToolDescriptor::new("alpha", "a"),
        ];
        let mut b = request("hi");
        b.model = "a-different-model".into();
        b.tools = vec![
            ToolDescriptor::new("alpha", "a"),
            ToolDescriptor::new("zebra", "z"),
        ];
        assert_eq!(fingerprint(&a), fingerprint(&b));
    }

    #[test]
    fn different_messages_get_different_keys() {
        assert_ne!(fingerprint(&request("hi")), fingerprint(&request("bye")));
    }

    #[test]
    fn correlation_separates_identical_requests() {
        let alpha = request("hi").with_correlation("alpha", 0);
        let beta = request("hi").with_correlation("beta", 0);
        assert_ne!(
            fingerprint(&alpha),
            fingerprint(&beta),
            "two agents issuing the same request must not share a journal key"
        );
        let alpha_again = request("hi").with_correlation("alpha", 0);
        assert_eq!(fingerprint(&alpha), fingerprint(&alpha_again));
    }

    #[test]
    fn record_rejects_a_conflicting_duplicate_fingerprint() {
        let journal = Journal::new();
        let req = request("hi").with_correlation("worker", 0);
        journal
            .record(&req, &GenerationResponse::text("alpha"))
            .expect("first record");
        let conflict = journal.record(&req, &GenerationResponse::text("beta"));
        assert!(
            conflict.is_err(),
            "a different response under one key must fail"
        );
        assert!(
            conflict.unwrap_err().to_string().contains("collision"),
            "the error must name the collision"
        );
        assert_eq!(
            journal.lookup(&req).expect("recorded").content,
            "alpha",
            "the first recording must survive, not be overwritten"
        );
    }

    #[test]
    fn record_is_idempotent_for_an_identical_response() {
        let journal = Journal::new();
        let req = request("hi").with_correlation("worker", 0);
        let response = GenerationResponse::text("alpha");
        journal.record(&req, &response).expect("first record");
        journal
            .record(&req, &response)
            .expect("identical re-record");
        assert_eq!(journal.len(), 1);
    }

    #[test]
    fn record_tools_rejects_a_conflicting_duplicate_fingerprint() {
        let journal = Journal::new();
        let req = request("hi").with_correlation("worker", 0);
        let call = ToolCall::new("c1", "deploy", serde_json::json!({}));
        journal
            .record_tools(&req, vec![(call.clone(), ToolOutcome::text("a"))], 0)
            .expect("first record");
        let conflict = journal.record_tools(&req, vec![(call, ToolOutcome::text("b"))], 0);
        assert!(
            conflict.is_err(),
            "different tool results under one key must fail"
        );
        assert!(conflict.unwrap_err().to_string().contains("collision"));
        let (got, _) = journal.lookup_tools(&req).expect("recorded");
        assert_eq!(got[0].1.text, "a", "the first recording must survive");
    }

    #[test]
    fn record_tools_is_idempotent_for_identical_results() {
        let journal = Journal::new();
        let req = request("hi").with_correlation("worker", 0);
        let results = vec![(
            ToolCall::new("c1", "deploy", serde_json::json!({})),
            ToolOutcome::text("a"),
        )];
        journal
            .record_tools(&req, results.clone(), 0)
            .expect("first record");
        journal
            .record_tools(&req, results, 0)
            .expect("identical re-record");
    }

    #[test]
    fn a_collision_error_names_the_colliding_agent() {
        let journal = Journal::new();
        let req = request("hi").with_correlation("worker", 3);
        journal
            .record(&req, &GenerationResponse::text("alpha"))
            .expect("first record");
        let error = journal
            .record(&req, &GenerationResponse::text("beta"))
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("worker"), "must name the agent: {message}");
        assert!(message.contains("turn 3"), "must name the turn: {message}");
    }

    #[test]
    fn tool_results_round_trip_under_their_request_key() {
        let journal = Journal::new();
        let req = request("hi").with_correlation("ops", 0);
        assert!(journal.lookup_tools(&req).is_none());
        let results = vec![(
            ToolCall::new("c1", "deploy", serde_json::json!({})),
            ToolOutcome::ok(serde_json::json!({"shipped": true}), "shipped"),
        )];
        journal
            .record_tools(&req, results, 1)
            .expect("record tools");
        let (got, refused) = journal.lookup_tools(&req).expect("recorded");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0.name, "deploy");
        assert_eq!(got[0].1.text, "shipped");
        assert_eq!(refused, 1);
        assert!(
            journal
                .lookup_tools(&request("hi").with_correlation("other", 0))
                .is_none(),
            "another agent's identical request must not see these results"
        );
    }

    #[test]
    fn a_journal_round_trips_through_bytes() {
        let journal = Journal::new();
        let req = request("hi").with_correlation("ops", 0);
        journal
            .record(&req, &GenerationResponse::text("hello"))
            .expect("record");
        journal
            .record_tools(
                &req,
                vec![(
                    ToolCall::new("c1", "deploy", serde_json::json!({})),
                    ToolOutcome::ok(serde_json::json!({"shipped": true}), "shipped"),
                )],
                1,
            )
            .expect("record tools");

        let bytes = journal.to_bytes().expect("serialize");
        let restored = Journal::from_bytes(&bytes).expect("deserialize");
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.lookup(&req).expect("response").content, "hello");
        let (results, refused) = restored.lookup_tools(&req).expect("tools");
        assert_eq!(results[0].1.text, "shipped");
        assert_eq!(refused, 1);
        assert!(restored.lookup(&request("absent")).is_none());
    }

    #[test]
    fn a_journal_round_trips_through_a_file() {
        let journal = Journal::new();
        let req = request("hi").with_correlation("ops", 0);
        journal
            .record(&req, &GenerationResponse::text("hello"))
            .expect("record");

        let path =
            std::env::temp_dir().join(format!("xihe-journal-unit-{}.json", std::process::id()));
        journal.save(&path).expect("save");
        let restored = Journal::load(&path).expect("load");
        std::fs::remove_file(&path).ok();

        assert_eq!(restored.lookup(&req).expect("response").content, "hello");
    }

    #[tokio::test]
    async fn replay_answers_from_the_journal_and_fails_on_a_miss() {
        let journal = Journal::new();
        journal
            .record(&request("hi"), &GenerationResponse::text("hello"))
            .expect("record");
        let provider = ReplayProvider::new(journal);
        assert_eq!(provider.id(), "replay");
        let answered = provider.generate(&request("hi")).await.expect("recorded");
        assert_eq!(answered.content, "hello");
        let missed = provider.generate(&request("absent")).await;
        assert!(missed.is_err(), "an unrecorded request must fail loudly");
    }
}
