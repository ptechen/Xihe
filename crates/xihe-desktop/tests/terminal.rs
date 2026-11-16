//! 端到端：真实 zsh + OSC 133 集成，命令必须聚成带退出码的块。

use std::sync::mpsc;
use std::time::{Duration, Instant};

use xihe_desktop::blocks::BlockState;
use xihe_desktop::pty::{PtyMessage, PtySession};
use xihe_desktop::terminal::{TermListener, TerminalState};

const ZSH: &str = "/bin/zsh";

fn wait_for_block(
    state: &mut TerminalState,
    rx: &mpsc::Receiver<PtyMessage>,
    command: &str,
) -> Option<(String, Option<i32>)> {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(PtyMessage::Bytes(bytes)) => state.advance(&bytes),
            Ok(PtyMessage::Exit) => return None,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return None,
        }
        if let Some(block) = state
            .blocks()
            .iter()
            .find(|b| b.state == BlockState::Done && b.command.contains(command))
        {
            return Some((block.output.clone(), block.exit_code));
        }
    }
    None
}

#[test]
fn echo_command_forms_a_block_with_exit_zero() {
    let (tx, rx) = mpsc::channel();
    let session = PtySession::spawn_shell(ZSH, 80, 24, tx).expect("spawn zsh");
    let mut state = TerminalState::new(80, 24, TermListener::new(session.writer()));

    session.write_input(b"echo xihe-block-test\n").expect("write");
    let (output, exit) =
        wait_for_block(&mut state, &rx, "echo xihe-block-test").expect("block should finish");
    assert!(output.contains("xihe-block-test"), "output: {output:?}");
    assert_eq!(exit, Some(0));
}

#[test]
fn failing_command_records_nonzero_exit() {
    let (tx, rx) = mpsc::channel();
    let session = PtySession::spawn_shell(ZSH, 80, 24, tx).expect("spawn zsh");
    let mut state = TerminalState::new(80, 24, TermListener::new(session.writer()));

    session.write_input(b"false\n").expect("write");
    let (_, exit) = wait_for_block(&mut state, &rx, "false").expect("block should finish");
    assert_eq!(exit, Some(1));
}

#[test]
fn two_commands_form_two_blocks_in_order() {
    let (tx, rx) = mpsc::channel();
    let session = PtySession::spawn_shell(ZSH, 80, 24, tx).expect("spawn zsh");
    let mut state = TerminalState::new(80, 24, TermListener::new(session.writer()));

    session.write_input(b"echo first-command\n").expect("write");
    wait_for_block(&mut state, &rx, "echo first-command").expect("first block");
    session.write_input(b"echo second-command\n").expect("write");
    wait_for_block(&mut state, &rx, "echo second-command").expect("second block");

    let done: Vec<&xihe_desktop::blocks::Block> = state
        .blocks()
        .iter()
        .filter(|b| b.state == BlockState::Done && b.command.starts_with("echo"))
        .collect();
    assert_eq!(done.len(), 2);
    assert!(done[0].output.contains("first-command"));
    assert!(done[1].output.contains("second-command"));
    assert!(done[0].id < done[1].id);
}