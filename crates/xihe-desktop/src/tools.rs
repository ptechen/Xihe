//! 桌面工具：把命令注入真实终端（需人工审批）、读文件与列目录（只读免审批）。

use std::path::PathBuf;
use std::time::Duration;

use xihe::agent::{Tool, ToolContext, ToolOutcome};
use xihe::agent_llm::{ToolCall, ToolDescriptor};

use crate::blocks::BlockState;
use crate::engine::{EventBus, TerminalLink, UiEvent};

const OUTPUT_CAP: usize = 8_000;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

fn cap(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut cut = limit;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}\n…[truncated, {} bytes total]", &text[..cut], text.len())
}

fn arg_str<'a>(call: &'a ToolCall, key: &str) -> Option<&'a str> {
    call.arguments.get(key).and_then(|value| value.as_str())
}

/// 把命令注入用户的真实终端；运行前必须有人批准，输出以块的形式回收。
pub struct RunInTerminal {
    /// 事件总线。
    pub bus: EventBus,
    /// 终端桥：PTY 写端 + 块快照。
    pub link: TerminalLink,
}

#[async_trait::async_trait]
impl Tool for RunInTerminal {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "run_command".to_string(),
            description: "Run a shell command in the user's terminal and return its exit code \
                          and output. The command is visible to the user as a block. Requires \
                          human approval before it runs."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The shell command to run." }
                },
                "required": ["command"]
            }),
        }
    }

    fn requires_approval(&self) -> bool {
        true
    }

    async fn run(&self, ctx: &ToolContext, call: &ToolCall) -> ToolOutcome {
        let agent = self.bus.name(ctx.agent);
        let Some(command) = arg_str(call, "command") else {
            return ToolOutcome::error("missing `command` argument");
        };
        if command.trim().is_empty() {
            return ToolOutcome::error("empty command");
        }
        self.bus.send(UiEvent::ToolStarted { agent: agent.clone(), call: call.clone() });

        let before = self.link.blocks.borrow().len();
        let injected = format!("{}\n", command.trim_end());
        if let Err(error) = self
            .link
            .writer
            .lock()
            .expect("pty writer lock")
            .write_all(injected.as_bytes())
        {
            let text = format!("cannot write to terminal: {error}");
            self.bus.send(UiEvent::ToolFinished {
                agent,
                name: call.name.clone(),
                ok: false,
                output: text.clone(),
            });
            return ToolOutcome::error(text);
        }

        // 等对应命令的块落定：命令文本匹配，避免抓到用户同时敲的命令。
        // 命令在当前提示符块里执行，所以回看一个块；再往前是历史，避免误匹配。
        let skip = before.saturating_sub(1);
        let mut blocks_rx = self.link.blocks.clone();
        let wanted = command.trim().to_string();
        let wait = async {
            loop {
                let snapshot = blocks_rx.borrow_and_update().clone();
                if let Some(block) = snapshot.iter().skip(skip).find(|block| {
                    block.state == BlockState::Done && block.command.trim() == wanted
                }) {
                    return Some(block.clone());
                }
                if blocks_rx.changed().await.is_err() {
                    return None;
                }
            }
        };
        let result = tokio::time::timeout(COMMAND_TIMEOUT, wait).await;
        let (ok, text) = match result {
            Ok(Some(block)) => {
                let code = block.exit_code.unwrap_or(-1);
                let output = cap(&block.output, OUTPUT_CAP);
                (block.exit_code == Some(0), format!("exit: {code}\n{output}"))
            }
            Ok(None) => (false, "terminal block stream closed".to_string()),
            Err(_) => (
                false,
                format!("command did not finish within {}s", COMMAND_TIMEOUT.as_secs()),
            ),
        };
        self.bus.send(UiEvent::ToolFinished {
            agent,
            name: call.name.clone(),
            ok,
            output: text.clone(),
        });
        if ok {
            ToolOutcome::ok(serde_json::json!({ "result": text }), text)
        } else {
            ToolOutcome::error(text)
        }
    }
}

/// 读取项目内的一个文本文件；只读，免审批。
pub struct ReadFile {
    /// 事件总线。
    pub bus: EventBus,
    /// 相对路径的根目录。
    pub workdir: PathBuf,
}

#[async_trait::async_trait]
impl Tool for ReadFile {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "read_file".to_string(),
            description: "Read a UTF-8 text file relative to the project directory.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path relative to the project." }
                },
                "required": ["path"]
            }),
        }
    }

    fn readonly(&self) -> bool {
        true
    }

    async fn run(&self, ctx: &ToolContext, call: &ToolCall) -> ToolOutcome {
        let agent = self.bus.name(ctx.agent);
        let Some(path) = arg_str(call, "path") else {
            return ToolOutcome::error("missing `path` argument");
        };
        self.bus.send(UiEvent::ToolStarted {
            agent: agent.clone(),
            call: call.clone(),
        });
        let full = self.workdir.join(path);
        let (ok, text) = match tokio::fs::read_to_string(&full).await {
            Ok(content) => (true, cap(&content, 16_000)),
            Err(error) => (false, format!("cannot read {}: {error}", full.display())),
        };
        self.bus.send(UiEvent::ToolFinished {
            agent,
            name: call.name.clone(),
            ok,
            output: text.clone(),
        });
        if ok {
            ToolOutcome::ok(serde_json::json!({ "content": text }), text)
        } else {
            ToolOutcome::error(text)
        }
    }
}

/// 列出项目内一个目录的条目；只读，免审批。
pub struct ListDir {
    /// 事件总线。
    pub bus: EventBus,
    /// 相对路径的根目录。
    pub workdir: PathBuf,
}

#[async_trait::async_trait]
impl Tool for ListDir {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "list_dir".to_string(),
            description: "List entries of a directory relative to the project; directories end \
                          with `/`."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory path, `.` for the project root." }
                },
                "required": ["path"]
            }),
        }
    }

    fn readonly(&self) -> bool {
        true
    }

    async fn run(&self, ctx: &ToolContext, call: &ToolCall) -> ToolOutcome {
        let agent = self.bus.name(ctx.agent);
        let path = arg_str(call, "path").unwrap_or(".");
        self.bus.send(UiEvent::ToolStarted {
            agent: agent.clone(),
            call: call.clone(),
        });
        let full = self.workdir.join(path);
        let (ok, text) = match tokio::fs::read_dir(&full).await {
            Ok(mut entries) => {
                let mut names = Vec::new();
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let suffix = if entry.path().is_dir() { "/" } else { "" };
                    names.push(format!("{}{suffix}", entry.file_name().to_string_lossy()));
                }
                names.sort();
                (true, cap(&names.join("\n"), OUTPUT_CAP))
            }
            Err(error) => (false, format!("cannot list {}: {error}", full.display())),
        };
        self.bus.send(UiEvent::ToolFinished {
            agent,
            name: call.name.clone(),
            ok,
            output: text.clone(),
        });
        if ok {
            ToolOutcome::ok(serde_json::json!({ "entries": text }), text)
        } else {
            ToolOutcome::error(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_keeps_short_text_and_marks_long_text() {
        assert_eq!(cap("abc", 8), "abc");
        let long = "x".repeat(20);
        let capped = cap(&long, 8);
        assert!(capped.starts_with("xxxxxxxx"));
        assert!(capped.contains("truncated"));
    }
}