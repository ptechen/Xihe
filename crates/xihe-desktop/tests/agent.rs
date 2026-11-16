//! 端到端：agent 提议命令 → 人批准 → 命令注入真实 zsh 并以块回收输出。

use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use xihe::agent::Decision;
use xihe_desktop::blocks::{Block, BlockState};
use xihe_desktop::config::ModelConfig;
use xihe_desktop::engine::{self, EngineHandle, TerminalLink, UiCommand, UiEvent};
use xihe_desktop::pty::{PtyMessage, PtySession};
use xihe_desktop::terminal::{TermListener, TerminalState};

const ZSH: &str = "/bin/zsh";

struct Harness {
    session: PtySession,
    state: TerminalState,
    pty_rx: mpsc::Receiver<PtyMessage>,
    block_tx: tokio::sync::watch::Sender<Arc<Vec<Block>>>,
    engine: EngineHandle,
    events: Vec<UiEvent>,
}

fn harness() -> Harness {
    let (tx, rx) = mpsc::channel();
    let session = PtySession::spawn_shell(ZSH, 80, 24, tx).expect("spawn zsh");
    let state = TerminalState::new(80, 24, TermListener::new(session.writer()));
    let (block_tx, block_rx) = tokio::sync::watch::channel(Arc::new(Vec::new()));
    // 强制离线 mock，避免测试环境里的真实密钥。
    let config = ModelConfig { api_key: String::new(), ..ModelConfig::default() };
    let engine = engine::spawn(
        config,
        None,
        TerminalLink { writer: session.writer(), blocks: block_rx },
    );
    Harness { session, state, pty_rx: rx, block_tx, engine, events: Vec::new() }
}

impl Harness {
    fn pump(&mut self) {
        while let Ok(message) = self.pty_rx.try_recv() {
            if let PtyMessage::Bytes(bytes) = message {
                self.state.advance(&bytes);
            }
        }
        self.block_tx.send_replace(Arc::new(self.state.blocks().to_vec()));
        while let Ok(event) = self.engine.events.try_recv() {
            self.events.push(event);
        }
    }

    fn wait_event(&mut self, pred: impl Fn(&UiEvent) -> bool) -> UiEvent {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            self.pump();
            if let Some(event) = self.events.iter().rev().find(|e| pred(e)) {
                return event.clone();
            }
            assert!(
                Instant::now() < deadline,
                "timed out; events: {:?}; blocks: {:?}",
                self.events,
                self.state
                    .blocks()
                    .iter()
                    .map(|b| (b.id, b.state, b.command.trim().to_string(), b.exit_code))
                    .collect::<Vec<_>>()
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn ask(&mut self, text: &str) {
        self.engine.commands.send(UiCommand::Ask(text.into())).expect("ask");
    }

    fn decide(&mut self, id: u64, decision: Decision) {
        self.engine.commands.send(UiCommand::Decide { id, decision }).expect("decide");
    }
}

#[test]
fn approved_command_runs_in_the_real_terminal() {
    let mut h = harness();
    h.ask("list the files");
    let approval = h.wait_event(|e| matches!(e, UiEvent::ApprovalNeeded { .. }));
    let UiEvent::ApprovalNeeded { id, call, .. } = approval else { unreachable!() };
    assert_eq!(call.name, "run_command");
    assert_eq!(call.arguments["command"], "ls");

    h.decide(id, Decision::Approved);
    h.wait_event(|e| matches!(e, UiEvent::EpisodeDone { .. }));

    assert!(
        h.state
            .blocks()
            .iter()
            .any(|b| b.state == BlockState::Done && b.command.trim() == "ls"),
        "the approved command must appear as a terminal block"
    );
    assert!(
        h.events.iter().any(|e| matches!(e, UiEvent::ToolFinished { name, ok: true, .. } if name == "run_command")),
        "tool must report success with the block output"
    );
    assert!(h.events.iter().any(|e| matches!(e, UiEvent::Answer { .. })));
    let _ = &h.session;
}

#[test]
fn rejected_command_never_runs() {
    let mut h = harness();
    h.ask("list the files");
    let approval = h.wait_event(|e| matches!(e, UiEvent::ApprovalNeeded { .. }));
    let UiEvent::ApprovalNeeded { id, .. } = approval else { unreachable!() };

    h.decide(id, Decision::reject("no"));
    let done = h.wait_event(|e| matches!(e, UiEvent::EpisodeDone { .. }));
    let UiEvent::EpisodeDone { snapshot, .. } = done else { unreachable!() };

    assert!(
        !h.state
            .blocks()
            .iter()
            .any(|b| b.state == BlockState::Done && b.command.trim() == "ls"),
        "a rejected command must never reach the shell"
    );
    assert!(snapshot.tool_refusals > 0, "rejection must be counted by the observer");
}