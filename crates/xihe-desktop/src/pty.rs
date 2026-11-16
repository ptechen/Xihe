//! PTY 会话：起一个真实 shell，注入 OSC 133 shell integration，
//! 读线程把字节流送进通道，UI 线程负责喂给终端状态。

use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use portable_pty::{Child, ChildKiller, CommandBuilder, MasterPty, PtySize, native_pty_system};

/// 读线程发来的消息。
#[derive(Debug)]
pub enum PtyMessage {
    /// 一段原始输出字节。
    Bytes(Vec<u8>),
    /// shell 已退出（读到 EOF）。
    Exit,
}

/// 一个活着的 shell 会话。
pub struct PtySession {
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    master: Box<dyn MasterPty + Send>,
    child: Box<dyn Child + Send + Sync>,
    killer: Box<dyn ChildKiller + Send + Sync>,
    integration_dir: Option<PathBuf>,
}

impl PtySession {
    /// 用默认 shell（`$SHELL`，缺省 `/bin/zsh`）启动会话。
    pub fn spawn(cols: u16, rows: u16, tx: Sender<PtyMessage>) -> io::Result<Self> {
        Self::spawn_with_waker(cols, rows, tx, None)
    }

    /// 同 [`Self::spawn`]，但输出到达时额外调用 `waker`（用于唤醒 UI）。
    pub fn spawn_with_waker(
        cols: u16,
        rows: u16,
        tx: Sender<PtyMessage>,
        waker: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> io::Result<Self> {
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".into());
        Self::spawn_shell_with_waker(&shell, cols, rows, tx, waker)
    }

    /// 用指定 shell 启动会话；zsh/bash 会注入块标记集成脚本。
    pub fn spawn_shell(
        shell: &str,
        cols: u16,
        rows: u16,
        tx: Sender<PtyMessage>,
    ) -> io::Result<Self> {
        Self::spawn_shell_with_waker(shell, cols, rows, tx, None)
    }

    /// 同 [`Self::spawn_shell`]，附带输出唤醒回调。
    pub fn spawn_shell_with_waker(
        shell: &str,
        cols: u16,
        rows: u16,
        tx: Sender<PtyMessage>,
        waker: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> io::Result<Self> {
        let pty = native_pty_system();
        let pair = pty
            .openpty(PtySize { rows, cols, pixel_width: 0, pixel_height: 0 })
            .map_err(io::Error::other)?;

        let mut cmd = CommandBuilder::new(shell);
        cmd.arg("-i");
        cmd.env("TERM", "xterm-256color");
        cmd.env("COLORTERM", "truecolor");
        cmd.env("XIHE_SHELL_INTEGRATION", "1");

        let name = std::path::Path::new(shell)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let integration_dir = match name.as_str() {
            "zsh" => Some(write_integration("zsh")?),
            "bash" => Some(write_integration("bash")?),
            _ => None,
        };
        if let Some(dir) = &integration_dir {
            match name.as_str() {
                "zsh" => {
                    let user_zdotdir = std::env::var("ZDOTDIR").unwrap_or_else(|_| {
                        std::env::var("HOME").unwrap_or_else(|_| ".".into())
                    });
                    cmd.env("ZDOTDIR", dir);
                    cmd.env("__XIHE_USER_ZDOTDIR", user_zdotdir);
                }
                "bash" => {
                    cmd.arg("--rcfile");
                    cmd.arg(dir.join("xihe.bash"));
                }
                _ => {}
            }
        }

        let child = pair.slave.spawn_command(cmd).map_err(io::Error::other)?;
        let killer = child.clone_killer();
        let writer = Arc::new(Mutex::new(pair.master.take_writer().map_err(io::Error::other)?));
        let mut reader = pair.master.try_clone_reader().map_err(io::Error::other)?;

        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => {
                        let _ = tx.send(PtyMessage::Exit);
                        if let Some(waker) = &waker {
                            waker();
                        }
                        break;
                    }
                    Ok(n) => {
                        if tx.send(PtyMessage::Bytes(buf[..n].to_vec())).is_err() {
                            break;
                        }
                        if let Some(waker) = &waker {
                            waker();
                        }
                    }
                }
            }
        });

        Ok(Self { writer, master: pair.master, child, killer, integration_dir })
    }

    /// 向 shell 写入输入（按键或注入的命令）。
    pub fn write_input(&self, bytes: &[u8]) -> io::Result<()> {
        let mut writer = self.writer.lock().unwrap();
        writer.write_all(bytes)?;
        writer.flush()
    }

    /// 通知内核窗口尺寸变化。
    pub fn resize(&self, cols: u16, rows: u16) -> io::Result<()> {
        self.master
            .resize(PtySize { rows, cols, pixel_width: 0, pixel_height: 0 })
            .map_err(io::Error::other)
    }

    /// 共享写端：终端仿真器的应答（如 DSR）也要写回 PTY。
    pub fn writer(&self) -> Arc<Mutex<Box<dyn Write + Send>>> {
        Arc::clone(&self.writer)
    }
}

impl Drop for PtySession {
    fn drop(&mut self) {
        let _ = self.killer.kill();
        let _ = self.child.wait();
        if let Some(dir) = &self.integration_dir {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn write_integration(shell: &str) -> io::Result<PathBuf> {
    static SESSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = SESSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "xihe-shell-{}-{nonce}-{shell}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir)?;
    let script = match shell {
        "zsh" => include_str!("integration/xihe.zsh"),
        _ => include_str!("integration/xihe.bash"),
    };
    let file = if shell == "zsh" { ".zshrc" } else { "xihe.bash" };
    std::fs::write(dir.join(file), script)?;
    Ok(dir)
}