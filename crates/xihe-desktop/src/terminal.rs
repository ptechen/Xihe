//! 终端状态：alacritty 仿真器 + OSC 133 块解析，共用一条字节流。

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use alacritty_terminal::event::{Event, EventListener};
use alacritty_terminal::grid::Dimensions;
use alacritty_terminal::term::{Config, Term};
use alacritty_terminal::vte::ansi::Processor;

use crate::blocks::{Block, Blocks, OscSplitter, Segment};

/// 终端尺寸（列 × 行）。
#[derive(Clone, Copy)]
pub struct TermSize {
    /// 列数。
    pub columns: usize,
    /// 可视行数。
    pub screen_lines: usize,
}

impl Dimensions for TermSize {
    fn total_lines(&self) -> usize {
        self.screen_lines
    }

    fn screen_lines(&self) -> usize {
        self.screen_lines
    }

    fn columns(&self) -> usize {
        self.columns
    }
}

/// 把终端事件接回 PTY 与 UI。
#[derive(Clone)]
pub struct TermListener {
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    /// shell 设置的窗口标题。
    pub title: Arc<Mutex<String>>,
    /// shell 是否已退出。
    pub exited: Arc<AtomicBool>,
}

impl TermListener {
    /// 用 PTY 写端构造监听器。
    pub fn new(writer: Arc<Mutex<Box<dyn Write + Send>>>) -> Self {
        Self {
            writer,
            title: Arc::new(Mutex::new(String::new())),
            exited: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl EventListener for TermListener {
    fn send_event(&self, event: Event) {
        match event {
            Event::PtyWrite(text) => {
                let mut writer = self.writer.lock().unwrap();
                let _ = writer.write_all(text.as_bytes());
                let _ = writer.flush();
            }
            Event::Title(title) => *self.title.lock().unwrap() = title,
            Event::ResetTitle => self.title.lock().unwrap().clear(),
            Event::Exit => self.exited.store(true, Ordering::SeqCst),
            _ => {}
        }
    }
}

/// 一个终端的完整状态：网格 + 块。
pub struct TerminalState {
    term: Term<TermListener>,
    processor: Processor,
    splitter: OscSplitter,
    blocks: Blocks,
    segments: Vec<Segment>,
    size: TermSize,
}

impl TerminalState {
    /// 按给定尺寸创建终端。
    pub fn new(columns: usize, screen_lines: usize, listener: TermListener) -> Self {
        let size = TermSize { columns, screen_lines };
        let config = Config { scrolling_history: 10_000, ..Config::default() };
        Self {
            term: Term::new(config, &size, listener),
            processor: Processor::new(),
            splitter: OscSplitter::new(),
            blocks: Blocks::new(),
            segments: Vec::new(),
            size,
        }
    }

    /// 喂入一段 PTY 字节：普通字节进仿真器与文本捕获，133 标记推进块。
    pub fn advance(&mut self, bytes: &[u8]) {
        self.segments.clear();
        self.splitter.feed(bytes, &mut self.segments);
        for segment in std::mem::take(&mut self.segments) {
            match segment {
                Segment::Plain(plain) => {
                    self.processor.advance(&mut self.term, &plain);
                    self.blocks.capture(&plain);
                }
                Segment::Marker(marker) => self.blocks.apply(marker),
            }
        }
    }

    /// 调整尺寸（网格与 PTY 需同步调整）。
    pub fn resize(&mut self, columns: usize, screen_lines: usize) {
        self.size = TermSize { columns, screen_lines };
        self.term.resize(self.size);
    }

    /// 滚动视口（回看历史或回到最新）。
    pub fn scroll(&mut self, scroll: alacritty_terminal::grid::Scroll) {
        self.term.scroll_display(scroll);
    }

    /// 当前尺寸。
    pub fn size(&self) -> TermSize {
        self.size
    }

    /// 所有命令块。
    pub fn blocks(&self) -> &[Block] {
        self.blocks.blocks()
    }

    /// 底层仿真器（渲染用）。
    pub fn term(&self) -> &Term<TermListener> {
        &self.term
    }
}