//! eframe 界面：Warp 式块转录（命令卡片 + 退出码徽标）+ 活体终端网格。

use std::sync::Arc;
use std::sync::mpsc::Receiver;

use alacritty_terminal::grid::Scroll;
use alacritty_terminal::term::TermMode;
use alacritty_terminal::term::cell::Flags;
use alacritty_terminal::term::color::Colors;
use alacritty_terminal::vte::ansi::{Color, CursorShape, NamedColor, Rgb};
use eframe::egui::{self, Align2, Color32, FontId, Rect, Sense, vec2};
use xihe::agent::Decision;

use crate::blocks::{Block, BlockState};
use crate::config::ModelConfig;
use crate::engine::{self, EngineHandle, TerminalLink, UiCommand, UiEvent};
use crate::pty::{PtyMessage, PtySession};
use crate::terminal::{TermListener, TerminalState};

const FONT_SIZE: f32 = 14.0;
const ROWS: usize = 24;
const DEFAULT_FG: Color32 = Color32::from_rgb(0xd8, 0xde, 0xe9);
const DEFAULT_BG: Color32 = Color32::from_rgb(0x1a, 0x1b, 0x26);
const CARD_BG: Color32 = Color32::from_rgb(0x24, 0x28, 0x3b);

/// agent 面板里的一条记录。
enum AgentEntry {
    /// 用户提问。
    User(String),
    /// 流式回答（逐段追加）。
    Stream(String),
    /// 最终回答。
    Answer(String),
    /// 一次工具调用的结果。
    Tool { name: String, ok: bool, output: String },
    /// 一次待决/已决的审批。
    Approval { id: u64, command: String, decided: Option<bool> },
    /// 错误。
    Error(String),
}

/// 终端应用。
pub struct XiheApp {
    session: PtySession,
    state: TerminalState,
    rx: Receiver<PtyMessage>,
    listener: TermListener,
    exited: bool,
    focus_id: egui::Id,
    engine: EngineHandle,
    block_tx: tokio::sync::watch::Sender<Arc<Vec<Block>>>,
    block_sig: Option<(usize, BlockState, Option<i32>, usize, usize)>,
    agent_open: bool,
    agent_input: String,
    agent_log: Vec<AgentEntry>,
}

impl XiheApp {
    /// 启动 shell 并创建应用；`ctx` 用于 PTY 输出到达时唤醒 UI。
    pub fn new(ctx: egui::Context) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let repaint_ctx = ctx.clone();
        let waker = Arc::new(move || repaint_ctx.request_repaint());
        let session =
            PtySession::spawn_with_waker(80, ROWS as u16, tx, Some(waker)).expect("spawn shell");
        let listener = TermListener::new(session.writer());
        let state = TerminalState::new(80, ROWS, listener.clone());
        let (block_tx, block_rx) = tokio::sync::watch::channel(Arc::new(Vec::new()));
        let engine = engine::spawn(
            ModelConfig::load(),
            Some(ctx.clone()),
            TerminalLink { writer: session.writer(), blocks: block_rx },
        );
        Self {
            session,
            state,
            rx,
            listener,
            exited: false,
            focus_id: egui::Id::new("terminal"),
            engine,
            block_tx,
            block_sig: None,
            agent_open: false,
            agent_input: String::new(),
            agent_log: Vec::new(),
        }
    }

    fn drain(&mut self) -> bool {
        let mut got = false;
        for _ in 0..256 {
            match self.rx.try_recv() {
                Ok(PtyMessage::Bytes(bytes)) => {
                    self.state.advance(&bytes);
                    got = true;
                }
                Ok(PtyMessage::Exit) => {
                    self.exited = true;
                    got = true;
                }
                Err(_) => break,
            }
        }
        got
    }

    fn publish_blocks(&mut self) {
        let blocks = self.state.blocks();
        let sig = blocks.last().map(|b| {
            (blocks.len(), b.state, b.exit_code, b.output.len(), b.command.len())
        });
        if sig != self.block_sig {
            self.block_sig = sig;
            self.block_tx.send_replace(Arc::new(blocks.to_vec()));
        }
    }

    fn drain_engine(&mut self) {
        while let Ok(event) = self.engine.events.try_recv() {
            match event {
                UiEvent::Stream { text, .. } => {
                    if let Some(AgentEntry::Stream(buffer)) = self.agent_log.last_mut() {
                        buffer.push_str(&text);
                    } else {
                        self.agent_log.push(AgentEntry::Stream(text));
                    }
                }
                UiEvent::Answer { text, .. } => {
                    if matches!(self.agent_log.last(), Some(AgentEntry::Stream(_))) {
                        self.agent_log.pop();
                    }
                    self.agent_log.push(AgentEntry::Answer(text));
                }
                UiEvent::ToolFinished { name, ok, output, .. } => {
                    self.agent_log.push(AgentEntry::Tool { name, ok, output });
                }
                UiEvent::ApprovalNeeded { id, call, .. } => {
                    let command = call
                        .arguments
                        .get("command")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(unknown)")
                        .to_string();
                    self.agent_log.push(AgentEntry::Approval { id, command, decided: None });
                }
                UiEvent::Error(text) => self.agent_log.push(AgentEntry::Error(text)),
                _ => {}
            }
        }
    }

    fn draw_agent_panel(&mut self, ctx: &egui::Context) {
        if !self.agent_open {
            return;
        }
        egui::SidePanel::right("agent_panel")
            .resizable(true)
            .default_width(340.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.heading("Agent");
                ui.separator();
                let mut ask = None;
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for entry in &mut self.agent_log {
                            match entry {
                                AgentEntry::User(text) => {
                                    ui.colored_label(Color32::from_rgb(0x7a, 0xa2, 0xf7), format!("you: {text}"));
                                }
                                AgentEntry::Stream(text) | AgentEntry::Answer(text) => {
                                    ui.label(text.as_str());
                                }
                                AgentEntry::Tool { name, ok, output } => {
                                    let color = if *ok {
                                        Color32::from_rgb(0x9e, 0xce, 0x6a)
                                    } else {
                                        Color32::from_rgb(0xf7, 0x76, 0x8e)
                                    };
                                    ui.colored_label(color, format!("tool {name}"));
                                    ui.label(
                                        egui::RichText::new(output.as_str())
                                            .monospace()
                                            .size(FONT_SIZE - 2.0),
                                    );
                                }
                                AgentEntry::Approval { id, command, decided } => {
                                    egui::Frame::new()
                                        .fill(CARD_BG)
                                        .corner_radius(6.0)
                                        .inner_margin(8.0)
                                        .show(ui, |ui| {
                                            ui.label(
                                                egui::RichText::new(format!("$ {command}"))
                                                    .monospace()
                                                    .strong(),
                                            );
                                            match decided {
                                                None => {
                                                    ui.horizontal(|ui| {
                                                        if ui.button("Approve").clicked() {
                                                            let _ = self.engine.commands.send(
                                                                UiCommand::Decide {
                                                                    id: *id,
                                                                    decision: Decision::Approved,
                                                                },
                                                            );
                                                            *decided = Some(true);
                                                        }
                                                        if ui.button("Reject").clicked() {
                                                            let _ = self.engine.commands.send(
                                                                UiCommand::Decide {
                                                                    id: *id,
                                                                    decision: Decision::reject("rejected in terminal"),
                                                                },
                                                            );
                                                            *decided = Some(false);
                                                        }
                                                    });
                                                }
                                                Some(true) => {
                                                    ui.colored_label(Color32::from_rgb(0x9e, 0xce, 0x6a), "approved");
                                                }
                                                Some(false) => {
                                                    ui.colored_label(Color32::from_rgb(0xf7, 0x76, 0x8e), "rejected");
                                                }
                                            }
                                        });
                                }
                                AgentEntry::Error(text) => {
                                    ui.colored_label(Color32::from_rgb(0xf7, 0x76, 0x8e), text.as_str());
                                }
                            }
                            ui.add_space(4.0);
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    let edit = egui::TextEdit::singleline(&mut self.agent_input)
                        .hint_text("Ask the agent…")
                        .show(ui);
                    let enter = edit.response.lost_focus()
                        && ctx.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui.button("Send").clicked() || enter {
                        ask = Some(std::mem::take(&mut self.agent_input));
                    }
                });
                if let Some(text) = ask
                    && !text.trim().is_empty()
                {
                    self.agent_log.push(AgentEntry::User(text.clone()));
                    let _ = self.engine.commands.send(UiCommand::Ask(text));
                }
            });
    }

    fn cell_metrics(&self, ctx: &egui::Context) -> (f32, f32) {
        let font = FontId::monospace(FONT_SIZE);
        ctx.fonts_mut(|f| (f.glyph_width(&font, 'M'), f.row_height(&font)))
    }

    fn sync_width(&mut self, ctx: &egui::Context) {
        let (cw, _) = self.cell_metrics(ctx);
        let width = ctx.available_rect().width() - 24.0;
        let cols = ((width / cw).floor() as usize).max(20);
        if cols != self.state.size().columns {
            let _ = self.session.resize(cols as u16, ROWS as u16);
            self.state.resize(cols, ROWS);
        }
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        if self.exited || !ctx.memory(|m| m.has_focus(self.focus_id)) {
            return;
        }
        let events = ctx.input(|i| i.events.clone());
        let mode = *self.state.term().mode();
        let app_cursor = mode.contains(TermMode::APP_CURSOR);
        let bracketed = mode.contains(TermMode::BRACKETED_PASTE);
        let mut bytes = Vec::new();
        for event in events {
            match event {
                egui::Event::Text(text) => {
                    for c in text.chars() {
                        if !c.is_control() {
                            bytes.extend(c.to_string().as_bytes());
                        }
                    }
                }
                egui::Event::Paste(text) => {
                    if bracketed {
                        bytes.extend_from_slice(b"\x1b[200~");
                        bytes.extend_from_slice(text.as_bytes());
                        bytes.extend_from_slice(b"\x1b[201~");
                    } else {
                        bytes.extend_from_slice(text.as_bytes());
                    }
                }
                egui::Event::Key { key, pressed: true, modifiers, .. } => {
                    if modifiers.shift && key == egui::Key::PageUp {
                        self.state.scroll(Scroll::PageUp);
                        continue;
                    }
                    if modifiers.shift && key == egui::Key::PageDown {
                        self.state.scroll(Scroll::PageDown);
                        continue;
                    }
                    if let Some(encoded) = encode_key(key, modifiers, app_cursor) {
                        bytes.extend(encoded);
                    }
                }
                _ => {}
            }
        }
        // 终端独占键盘：吃掉事件，避免 Tab 等被 egui 当作焦点导航。
        ctx.input_mut(|i| i.events.clear());
        if !bytes.is_empty() {
            let _ = self.session.write_input(&bytes);
            self.state.scroll(Scroll::Bottom);
        }
    }

    fn draw_grid(&self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let (cw, ch) = self.cell_metrics(ctx);
        let size = self.state.size();
        let grid_size = vec2(size.columns as f32 * cw, size.screen_lines as f32 * ch);
        let (rect, response) = ui.allocate_exact_size(grid_size, Sense::click());
        if response.clicked() || ctx.memory(|m| m.focused().is_none()) {
            ctx.memory_mut(|m| m.request_focus(self.focus_id));
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, DEFAULT_BG);

        let content = self.state.term().renderable_content();
        let offset = content.display_offset as i32;
        let font = FontId::monospace(FONT_SIZE);
        for indexed in content.display_iter {
            let row = indexed.point.line.0 + offset;
            if row < 0 || row >= size.screen_lines as i32 {
                continue;
            }
            let cell = &indexed.cell;
            if cell.flags.intersects(Flags::WIDE_CHAR_SPACER | Flags::LEADING_WIDE_CHAR_SPACER) {
                continue;
            }
            if cell.flags.contains(Flags::HIDDEN) {
                continue;
            }
            let bold = cell.flags.contains(Flags::BOLD);
            let mut fg = resolve_color(cell.fg, bold, content.colors);
            let mut bg = match cell.bg {
                Color::Named(NamedColor::Background) => DEFAULT_BG,
                other => resolve_color(other, false, content.colors),
            };
            if cell.flags.contains(Flags::INVERSE) {
                std::mem::swap(&mut fg, &mut bg);
            }
            if cell.flags.contains(Flags::DIM) {
                fg = fg.gamma_multiply(0.6);
            }
            let wide = if cell.flags.contains(Flags::WIDE_CHAR) { 2.0 } else { 1.0 };
            let pos = rect.min + vec2(indexed.point.column.0 as f32 * cw, row as f32 * ch);
            if bg != DEFAULT_BG {
                painter.rect_filled(Rect::from_min_size(pos, vec2(cw * wide, ch)), 0.0, bg);
            }
            if cell.c != ' ' {
                painter.text(pos, Align2::LEFT_TOP, cell.c.to_string(), font.clone(), fg);
            }
        }

        if content.display_offset == 0 && content.cursor.shape != CursorShape::Hidden {
            let point = content.cursor.point;
            let row = point.line.0 + offset;
            if row >= 0 && row < size.screen_lines as i32 {
                let pos = rect.min + vec2(point.column.0 as f32 * cw, row as f32 * ch);
                let cell = self.state.term().grid()[point].clone();
                match content.cursor.shape {
                    CursorShape::Beam => {
                        painter.rect_filled(Rect::from_min_size(pos, vec2(2.0, ch)), 0.0, DEFAULT_FG);
                    }
                    CursorShape::Underline => {
                        painter.rect_filled(
                            Rect::from_min_size(pos + vec2(0.0, ch - 2.0), vec2(cw, 2.0)),
                            0.0,
                            DEFAULT_FG,
                        );
                    }
                    _ => {
                        painter.rect_filled(Rect::from_min_size(pos, vec2(cw, ch)), 0.0, DEFAULT_FG);
                        if cell.c != ' ' {
                            painter.text(pos, Align2::LEFT_TOP, cell.c.to_string(), font, DEFAULT_BG);
                        }
                    }
                }
            }
        }
    }
}

impl eframe::App for XiheApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let got = self.drain();
        self.publish_blocks();
        self.drain_engine();
        self.sync_width(ctx);
        self.handle_input(ctx);

        egui::TopBottomPanel::top("bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Xihe");
                ui.separator();
                ui.toggle_value(&mut self.agent_open, "Agent");
                let title = self.listener.title.lock().unwrap().clone();
                if !title.is_empty() {
                    ui.label(title);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.exited {
                        ui.colored_label(Color32::from_rgb(0xf7, 0x76, 0x8e), "shell exited");
                    }
                    let done = self
                        .state
                        .blocks()
                        .iter()
                        .filter(|b| b.state == BlockState::Done)
                        .count();
                    ui.label(format!("{done} blocks"));
                });
            });
        });

        self.draw_agent_panel(ctx);

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(DEFAULT_BG).inner_margin(12.0))
            .show(ctx, |ui| {
                self.draw_transcript(ui, ctx);
            });

        if got {
            ctx.request_repaint();
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = self.engine.commands.send(UiCommand::Shutdown);
    }
}

impl XiheApp {
    fn draw_transcript(&self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for block in self.state.blocks() {
                    if block.state == BlockState::Done && !block.command.trim().is_empty() {
                        draw_block(ui, block);
                        ui.add_space(6.0);
                    }
                }
                self.draw_grid(ui, ctx);
            });
    }
}

fn draw_block(ui: &mut egui::Ui, block: &Block) {
    egui::Frame::new()
        .fill(CARD_BG)
        .corner_radius(6.0)
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("$ {}", block.command.trim_end()))
                        .monospace()
                        .strong()
                        .color(DEFAULT_FG),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (text, color) = match block.exit_code {
                        Some(0) => ("exit 0".to_string(), Color32::from_rgb(0x9e, 0xce, 0x6a)),
                        Some(code) => (format!("exit {code}"), Color32::from_rgb(0xf7, 0x76, 0x8e)),
                        None => ("exit ?".to_string(), Color32::from_rgb(0x56, 0x5f, 0x89)),
                    };
                    ui.colored_label(color, text);
                });
            });
            if !block.output.is_empty() {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(block.output.trim_end_matches('\n'))
                        .monospace()
                        .size(FONT_SIZE - 1.0)
                        .color(Color32::from_rgb(0xa9, 0xb1, 0xd6)),
                );
            }
        });
}

fn encode_key(key: egui::Key, mods: egui::Modifiers, app_cursor: bool) -> Option<Vec<u8>> {
    if mods.ctrl && !mods.alt && !mods.shift {
        if let Some(c) = letter(key) {
            return Some(vec![(c as u8) & 0x1f]);
        }
        return match key {
            egui::Key::Space => Some(vec![0]),
            egui::Key::Enter => Some(vec![b'\n']),
            egui::Key::Backspace => Some(vec![0x08]),
            _ => None,
        };
    }
    let mut out = Vec::new();
    if mods.alt && let Some(c) = letter(key) {
        out.push(0x1b);
        out.push(c as u8);
        return Some(out);
    }
    let modifier_code = 1 + u8::from(mods.shift) + 2 * u8::from(mods.alt) + 4 * u8::from(mods.ctrl);
    let csi = |final_byte: u8, numbered: bool| -> Vec<u8> {
        if modifier_code > 1 && numbered {
            format!("\x1b[1;{modifier_code}{}", final_byte as char).into_bytes()
        } else if app_cursor && matches!(final_byte, b'A' | b'B' | b'C' | b'D' | b'H' | b'F') {
            vec![0x1b, b'O', final_byte]
        } else {
            vec![0x1b, b'[', final_byte]
        }
    };
    let tilde = |n: u8| -> Vec<u8> {
        if modifier_code > 1 {
            format!("\x1b[{n};{modifier_code}~").into_bytes()
        } else {
            format!("\x1b[{n}~").into_bytes()
        }
    };
    let encoded = match key {
        egui::Key::Enter => vec![b'\r'],
        egui::Key::Backspace => vec![0x7f],
        egui::Key::Tab if mods.shift => b"\x1b[Z".to_vec(),
        egui::Key::Tab => vec![b'\t'],
        egui::Key::Escape => vec![0x1b],
        egui::Key::ArrowUp => csi(b'A', true),
        egui::Key::ArrowDown => csi(b'B', true),
        egui::Key::ArrowRight => csi(b'C', true),
        egui::Key::ArrowLeft => csi(b'D', true),
        egui::Key::Home => csi(b'H', true),
        egui::Key::End => csi(b'F', true),
        egui::Key::Insert => tilde(2),
        egui::Key::Delete => tilde(3),
        egui::Key::PageUp => tilde(5),
        egui::Key::PageDown => tilde(6),
        _ => return None,
    };
    out.extend(encoded);
    Some(out)
}

fn letter(key: egui::Key) -> Option<char> {
    use egui::Key::*;
    Some(match key {
        A => 'a', B => 'b', C => 'c', D => 'd', E => 'e', F => 'f', G => 'g', H => 'h',
        I => 'i', J => 'j', K => 'k', L => 'l', M => 'm', N => 'n', O => 'o', P => 'p',
        Q => 'q', R => 'r', S => 's', T => 't', U => 'u', V => 'v', W => 'w', X => 'x',
        Y => 'y', Z => 'z',
        _ => return None,
    })
}

fn resolve_color(color: Color, bold: bool, colors: &Colors) -> Color32 {
    match color {
        Color::Spec(Rgb { r, g, b }) => Color32::from_rgb(r, g, b),
        Color::Indexed(i) => {
            let (r, g, b) = indexed_rgb(i);
            Color32::from_rgb(r, g, b)
        }
        Color::Named(named) => {
            if let Some(rgb) = colors[named] {
                return Color32::from_rgb(rgb.r, rgb.g, rgb.b);
            }
            let index = named as usize;
            match index {
                0..=15 => {
                    let i = if bold && index < 8 { index + 8 } else { index };
                    let (r, g, b) = ANSI_16[i];
                    Color32::from_rgb(r, g, b)
                }
                256 => DEFAULT_FG,
                257 => DEFAULT_BG,
                258 => DEFAULT_FG,
                259..=266 => {
                    let (r, g, b) = ANSI_16[index - 259];
                    Color32::from_rgb(r, g, b).gamma_multiply(0.6)
                }
                _ => DEFAULT_FG,
            }
        }
    }
}

fn indexed_rgb(i: u8) -> (u8, u8, u8) {
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    match i {
        0..=15 => ANSI_16[i as usize],
        16..=231 => {
            let i = i - 16;
            (LEVELS[(i / 36) as usize], LEVELS[((i % 36) / 6) as usize], LEVELS[(i % 6) as usize])
        }
        _ => {
            let gray = 8 + (i - 232) * 10;
            (gray, gray, gray)
        }
    }
}

const ANSI_16: [(u8, u8, u8); 16] = [
    (0x15, 0x16, 0x1e), (0xf7, 0x76, 0x8e), (0x9e, 0xce, 0x6a), (0xe0, 0xaf, 0x68),
    (0x7a, 0xa2, 0xf7), (0xbb, 0x9a, 0xf7), (0x7d, 0xcf, 0xff), (0xa9, 0xb1, 0xd6),
    (0x41, 0x48, 0x68), (0xf7, 0x76, 0x8e), (0x9e, 0xce, 0x6a), (0xe0, 0xaf, 0x68),
    (0x7a, 0xa2, 0xf7), (0xbb, 0x9a, 0xf7), (0x7d, 0xcf, 0xff), (0xc0, 0xca, 0xf5),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn transcript_renders_block_cards_and_live_grid() {
        let ctx = egui::Context::default();
        let mut app = XiheApp::new(ctx.clone());
        app.session.write_input(b"echo render-test\n").unwrap();

        let deadline = Instant::now() + Duration::from_secs(20);
        let mut found = false;
        while Instant::now() < deadline && !found {
            match app.rx.recv_timeout(Duration::from_millis(100)) {
                Ok(PtyMessage::Bytes(bytes)) => app.state.advance(&bytes),
                Ok(PtyMessage::Exit) => break,
                Err(_) => {}
            }
            found = app.state.blocks().iter().any(|b| {
                b.state == BlockState::Done && b.output.contains("render-test")
            });
        }
        assert!(found, "block should finish");

        let raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, vec2(900.0, 700.0))),
            ..Default::default()
        };
        ctx.begin_pass(raw);
        egui::CentralPanel::default().show(&ctx, |ui| {
            app.draw_transcript(ui, &ctx);
        });
        let output = ctx.end_pass();
        let texts: Vec<String> = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::epaint::Shape::Text(text) => Some(text.galley.text().to_string()),
                _ => None,
            })
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("$ echo render-test")),
            "block card header missing; texts: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("render-test") && !t.starts_with('$')),
            "block output missing"
        );
        assert!(texts.iter().any(|t| t.contains("exit 0")), "exit badge missing");
    }

    #[test]
    fn ctrl_c_encodes_as_etx() {
        let mods = egui::Modifiers { ctrl: true, ..Default::default() };
        assert_eq!(encode_key(egui::Key::C, mods, false), Some(vec![3]));
    }

    #[test]
    fn arrows_follow_application_cursor_mode() {
        let none = egui::Modifiers::default();
        assert_eq!(encode_key(egui::Key::ArrowUp, none, false), Some(b"\x1b[A".to_vec()));
        assert_eq!(encode_key(egui::Key::ArrowUp, none, true), Some(b"\x1bOA".to_vec()));
        let shift = egui::Modifiers { shift: true, ..Default::default() };
        assert_eq!(encode_key(egui::Key::ArrowUp, shift, false), Some(b"\x1b[1;2A".to_vec()));
    }
}