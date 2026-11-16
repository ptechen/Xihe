//! Warp 式块：从字节流里解析 OSC 133 shell integration 标记，
//! 把「命令 + 输出 + 退出码」聚合成块。
//!
//! 字节流先经 [`OscSplitter`] 切成 [`Segment`]：普通字节原样喂给终端
//! 仿真器，133 标记交给 [`Blocks`] 推进块状态机。捕获命令/输出文本时
//! 会剥掉 ANSI 转义并处理 `\r` 覆盖写，得到干净的纯文本。

/// OSC 133 语义标记（shell integration）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Marker {
    /// A：提示符开始。
    PromptStart,
    /// B：提示符结束，用户输入开始。
    InputStart,
    /// C：命令开始执行，输出开始。
    OutputStart,
    /// D：命令结束，附带退出码（可能缺失）。
    Finished(Option<i32>),
}

/// 切分后的一段字节流。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// 普通字节（含非 133 的 OSC），原样交给终端仿真器。
    Plain(Vec<u8>),
    /// 一个 133 标记（已从流中摘除）。
    Marker(Marker),
}

/// 块的生命周期阶段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockState {
    /// 提示符正在绘制。
    Prompting,
    /// 用户正在输入命令。
    Input,
    /// 命令执行中。
    Running,
    /// 命令已结束。
    Done,
}

/// 一个命令块：命令文本、输出文本、退出码。
#[derive(Debug, Clone)]
pub struct Block {
    /// 单调递增的块编号。
    pub id: u64,
    /// 用户输入的命令（来自 B..C 之间的回显）。
    pub command: String,
    /// 命令输出（来自 C..D 之间，已剥 ANSI）。
    pub output: String,
    /// 退出码；D 标记缺失或无参数时为 None。
    pub exit_code: Option<i32>,
    /// 当前阶段。
    pub state: BlockState,
}

/// 单个块输出文本的上限，防止 `yes` 之类的命令撑爆内存。
const OUTPUT_CAP: usize = 128 * 1024;
/// OSC 载荷的上限；超过则按普通字节放行。
const OSC_CAP: usize = 4096;

/// 把字节流切成普通段与 OSC 133 标记。
///
/// 跨 feed 的半个 OSC 序列会留在内部缓冲里等后续字节。
#[derive(Default)]
pub struct OscSplitter {
    state: OscState,
    buf: Vec<u8>,
}

#[derive(Default, Clone, Copy)]
enum OscState {
    #[default]
    Ground,
    /// 刚见到 ESC，还不知道是不是 OSC。
    AfterEsc,
    /// 已见到 ESC ]，正在收集载荷（buf 含开头的 ESC ]）。
    Osc,
    /// OSC 载荷里见到 ESC，等待 `\` 组成 ST。
    OscEsc,
}

impl OscSplitter {
    /// 创建一个空切分器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 喂入一段字节，切分结果追加到 `out`。
    pub fn feed(&mut self, bytes: &[u8], out: &mut Vec<Segment>) {
        let mut plain: Vec<u8> = Vec::new();
        for &b in bytes {
            match self.state {
                OscState::Ground => {
                    if b == 0x1b {
                        self.state = OscState::AfterEsc;
                    } else {
                        plain.push(b);
                    }
                }
                OscState::AfterEsc => {
                    if b == b']' {
                        self.buf.clear();
                        self.buf.push(0x1b);
                        self.buf.push(b']');
                        self.state = OscState::Osc;
                    } else if b == 0x1b {
                        plain.push(0x1b);
                    } else {
                        // 普通转义（CSI 等）：原样放行。
                        plain.push(0x1b);
                        plain.push(b);
                        self.state = OscState::Ground;
                    }
                }
                OscState::Osc => {
                    if b == 0x07 {
                        self.finish_osc(&mut plain, out);
                    } else if b == 0x1b {
                        self.state = OscState::OscEsc;
                    } else {
                        self.buf.push(b);
                        if self.buf.len() > OSC_CAP {
                            plain.extend_from_slice(&self.buf);
                            self.buf.clear();
                            self.state = OscState::Ground;
                        }
                    }
                }
                OscState::OscEsc => {
                    if b == b'\\' {
                        self.finish_osc(&mut plain, out);
                    } else {
                        // 不是 ST：把 ESC 和该字节当作载荷继续。
                        self.buf.push(0x1b);
                        self.buf.push(b);
                        self.state = OscState::Osc;
                    }
                }
            }
        }
        if !plain.is_empty() {
            out.push(Segment::Plain(plain));
        }
    }

    fn finish_osc(&mut self, plain: &mut Vec<u8>, out: &mut Vec<Segment>) {
        if !plain.is_empty() {
            out.push(Segment::Plain(std::mem::take(plain)));
        }
        // buf 以 ESC ] 开头，载荷是其余部分。
        if let Some(marker) = parse_133(&self.buf[2..]) {
            out.push(Segment::Marker(marker));
        } else {
            // 非 133 OSC：原样放行（统一用 BEL 终止）。
            let mut raw = std::mem::take(&mut self.buf);
            raw.push(0x07);
            out.push(Segment::Plain(raw));
        }
        self.buf.clear();
        self.state = OscState::Ground;
    }
}

fn parse_133(payload: &[u8]) -> Option<Marker> {
    let text = std::str::from_utf8(payload).ok()?;
    let mut parts = text.split(';');
    if parts.next()? != "133" {
        return None;
    }
    match parts.next()? {
        "A" => Some(Marker::PromptStart),
        "B" => Some(Marker::InputStart),
        "C" => Some(Marker::OutputStart),
        "D" => Some(Marker::Finished(parts.next().and_then(|c| c.parse().ok()))),
        _ => None,
    }
}

/// 块集合与状态机。
#[derive(Default)]
pub struct Blocks {
    blocks: Vec<Block>,
    next_id: u64,
    capture: Capture,
}

impl Blocks {
    /// 创建一个空块集合。
    pub fn new() -> Self {
        Self::default()
    }

    /// 所有块（含进行中的）。
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// 最近一个块。
    pub fn current(&self) -> Option<&Block> {
        self.blocks.last()
    }

    /// 应用一个 133 标记，推进块状态。
    pub fn apply(&mut self, marker: Marker) {
        match marker {
            Marker::PromptStart => {
                self.flush_line();
                if let Some(last) = self.blocks.last_mut()
                    && last.state != BlockState::Done
                {
                    last.state = BlockState::Done;
                }
                self.blocks.push(Block {
                    id: self.next_id,
                    command: String::new(),
                    output: String::new(),
                    exit_code: None,
                    state: BlockState::Prompting,
                });
                self.next_id += 1;
            }
            Marker::InputStart => {
                // 提示符重绘会再发 B：只在 Prompting 时推进，避免回退状态。
                if let Some(last) = self.blocks.last_mut()
                    && last.state == BlockState::Prompting
                {
                    last.state = BlockState::Input;
                }
            }
            Marker::OutputStart => {
                self.flush_line();
                if let Some(last) = self.blocks.last_mut()
                    && matches!(last.state, BlockState::Input | BlockState::Prompting)
                {
                    last.state = BlockState::Running;
                }
            }
            Marker::Finished(code) => {
                self.flush_line();
                if let Some(last) = self.blocks.last_mut()
                    && last.state != BlockState::Done
                {
                    last.state = BlockState::Done;
                    last.exit_code = code;
                }
            }
        }
    }

    /// 捕获普通字节里的文本：Input 阶段进命令，Running 阶段进输出。
    pub fn capture(&mut self, bytes: &[u8]) {
        let Some(target) = self.target() else { return };
        self.capture.feed(bytes, target, &mut self.blocks);
    }

    fn target(&self) -> Option<Target> {
        match self.blocks.last().map(|b| b.state) {
            Some(BlockState::Input) => Some(Target::Command),
            Some(BlockState::Running) => Some(Target::Output),
            _ => None,
        }
    }

    fn flush_line(&mut self) {
        if let Some(target) = self.target() {
            self.capture.flush_line(target, &mut self.blocks);
        } else {
            self.capture.line.clear();
        }
    }
}

#[derive(Clone, Copy)]
enum Target {
    Command,
    Output,
}

/// ANSI 剥离 + 行缓冲捕获。
#[derive(Default)]
struct Capture {
    state: CaptureState,
    /// 当前行（`\r` 后跟非 `\n` 时清空，`\n` 落盘）。
    line: String,
    /// 刚见到 `\r`：跟 `\n` 是换行，跟其他字节才是覆盖写。
    cr: bool,
    /// 跨 feed 的半个 UTF-8 字符。
    pending: Vec<u8>,
    /// 剥转义后待解码的文本字节。
    text: Vec<u8>,
}

#[derive(Default, Clone, Copy)]
enum CaptureState {
    #[default]
    Ground,
    Esc,
    EscIntermediate,
    Csi,
    Osc,
    OscEsc,
}

impl Capture {
    fn feed(&mut self, bytes: &[u8], target: Target, blocks: &mut [Block]) {
        for &b in bytes {
            match self.state {
                CaptureState::Ground => {
                    // 控制字符会改动行缓冲：先把已积累的文本落进线里，保序。
                    if matches!(b, 0x00..=0x1f) {
                        self.decode();
                    }
                    if self.cr && b != b'\r' {
                        if b != b'\n' {
                            self.line.clear();
                        }
                        self.cr = false;
                    }
                    match b {
                        0x1b => self.state = CaptureState::Esc,
                        b'\n' => self.flush_line(target, blocks),
                        b'\r' => self.cr = true,
                        0x08 => {
                            self.line.pop();
                        }
                        b'\t' => self.line.push('\t'),
                        0x00..=0x1f => {}
                        _ => self.text.push(b),
                    }
                }
                CaptureState::Esc => match b {
                    b'[' => self.state = CaptureState::Csi,
                    b']' => self.state = CaptureState::Osc,
                    0x20..=0x2f => self.state = CaptureState::EscIntermediate,
                    _ => self.state = CaptureState::Ground,
                },
                CaptureState::EscIntermediate => match b {
                    0x20..=0x2f => {}
                    _ => self.state = CaptureState::Ground,
                },
                CaptureState::Csi => match b {
                    0x40..=0x7e => self.state = CaptureState::Ground,
                    0x1b => self.state = CaptureState::Esc,
                    _ => {}
                },
                CaptureState::Osc => match b {
                    0x07 => self.state = CaptureState::Ground,
                    0x1b => self.state = CaptureState::OscEsc,
                    _ => {}
                },
                CaptureState::OscEsc => {
                    self.state = if b == b'\\' {
                        CaptureState::Ground
                    } else {
                        CaptureState::Osc
                    };
                }
            }
        }
        self.decode();
    }

    fn decode(&mut self) {
        let text = std::mem::take(&mut self.text);
        if text.is_empty() {
            return;
        }
        let mut bytes = std::mem::take(&mut self.pending);
        bytes.extend_from_slice(&text);
        let consumed = match std::str::from_utf8(&bytes) {
            Ok(s) => {
                self.line.push_str(s);
                bytes.len()
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid > 0 {
                    self.line.push_str(std::str::from_utf8(&bytes[..valid]).unwrap());
                }
                if e.error_len().is_none() {
                    // 结尾是不完整的 UTF-8：留到下一次 feed。
                    valid
                } else {
                    // 非法字节：丢弃一个，避免卡死。
                    valid + 1
                }
            }
        };
        self.pending = bytes[consumed..].to_vec();
    }

    fn flush_line(&mut self, target: Target, blocks: &mut [Block]) {
        if self.line.is_empty() {
            return;
        }
        if let Some(block) = blocks.last_mut() {
            let field = match target {
                Target::Command => &mut block.command,
                Target::Output => &mut block.output,
            };
            if field.len() < OUTPUT_CAP {
                field.push_str(&self.line);
                field.push('\n');
            }
        }
        self.line.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(bytes: &[u8]) -> Vec<Segment> {
        let mut splitter = OscSplitter::new();
        let mut out = Vec::new();
        splitter.feed(bytes, &mut out);
        out
    }

    fn osc(marker: &str) -> String {
        format!("\x1b]133;{marker}\x07")
    }

    fn plain_of(segments: &[Segment]) -> Vec<u8> {
        segments
            .iter()
            .flat_map(|s| match s {
                Segment::Plain(b) => b.clone(),
                _ => vec![],
            })
            .collect()
    }

    #[test]
    fn plain_bytes_pass_through() {
        assert_eq!(split(b"hello\r\n"), vec![Segment::Plain(b"hello\r\n".to_vec())]);
    }

    #[test]
    fn csi_escapes_pass_through_untouched() {
        let bytes = b"\x1b[31mred\x1b[0m\r\n";
        assert_eq!(plain_of(&split(bytes)), bytes);
    }

    #[test]
    fn markers_are_extracted() {
        let stream = format!(
            "{}prompt{}{}echo hi\r\noutput\r\n{}",
            osc("A"),
            osc("B"),
            osc("C"),
            osc("D;0")
        );
        let segments = split(stream.as_bytes());
        let markers: Vec<Marker> = segments
            .iter()
            .filter_map(|s| match s {
                Segment::Marker(m) => Some(*m),
                _ => None,
            })
            .collect();
        assert_eq!(
            markers,
            vec![
                Marker::PromptStart,
                Marker::InputStart,
                Marker::OutputStart,
                Marker::Finished(Some(0)),
            ]
        );
    }

    #[test]
    fn non_133_osc_passes_through() {
        let segments = split(b"\x1b]0;title\x07rest");
        assert!(segments.iter().all(|s| matches!(s, Segment::Plain(_))));
        assert_eq!(plain_of(&segments), b"\x1b]0;title\x07rest");
    }

    #[test]
    fn st_terminator_works() {
        let segments = split(b"\x1b]133;D;1\x1b\\");
        assert_eq!(segments, vec![Segment::Marker(Marker::Finished(Some(1)))]);
    }

    #[test]
    fn split_across_feeds() {
        let mut splitter = OscSplitter::new();
        let mut out = Vec::new();
        splitter.feed(b"\x1b]133;", &mut out);
        assert!(out.is_empty());
        splitter.feed(b"D;42\x07", &mut out);
        assert_eq!(out, vec![Segment::Marker(Marker::Finished(Some(42)))]);
    }

    #[test]
    fn esc_split_across_feeds() {
        let mut splitter = OscSplitter::new();
        let mut out = Vec::new();
        splitter.feed(b"a\x1b", &mut out);
        splitter.feed(b"]133;A\x07", &mut out);
        assert_eq!(
            out,
            vec![Segment::Plain(b"a".to_vec()), Segment::Marker(Marker::PromptStart)]
        );
    }

    #[test]
    fn finished_without_code() {
        assert_eq!(split(osc("D").as_bytes()), vec![Segment::Marker(Marker::Finished(None))]);
    }

    fn run_stream(stream: &str) -> Blocks {
        let mut blocks = Blocks::new();
        let mut splitter = OscSplitter::new();
        let mut segments = Vec::new();
        splitter.feed(stream.as_bytes(), &mut segments);
        for segment in segments {
            match segment {
                Segment::Plain(b) => blocks.capture(&b),
                Segment::Marker(m) => blocks.apply(m),
            }
        }
        blocks
    }

    #[test]
    fn full_lifecycle_captures_command_output_and_exit() {
        let stream = format!(
            "{}$ {}echo hello\r\n{}hello\r\n{}",
            osc("A"),
            osc("B"),
            osc("C"),
            osc("D;0")
        );
        let blocks = run_stream(&stream);
        assert_eq!(blocks.blocks().len(), 1);
        let block = &blocks.blocks()[0];
        assert_eq!(block.command, "echo hello\n");
        assert_eq!(block.output, "hello\n");
        assert_eq!(block.exit_code, Some(0));
        assert_eq!(block.state, BlockState::Done);
    }

    #[test]
    fn ansi_is_stripped_from_output() {
        let stream = format!(
            "{}{}{}\x1b[31mred\x1b[0m plain\r\n{}",
            osc("A"),
            osc("B"),
            osc("C"),
            osc("D;0")
        );
        let blocks = run_stream(&stream);
        assert_eq!(blocks.blocks()[0].output, "red plain\n");
    }

    #[test]
    fn carriage_return_overwrites_line() {
        let stream = format!(
            "{}{}{}10%\r50%\r100%\r\ndone\r\n{}",
            osc("A"),
            osc("B"),
            osc("C"),
            osc("D;0")
        );
        let blocks = run_stream(&stream);
        assert_eq!(blocks.blocks()[0].output, "100%\ndone\n");
    }

    #[test]
    fn utf8_split_across_feeds() {
        let mut blocks = Blocks::new();
        blocks.apply(Marker::PromptStart);
        blocks.apply(Marker::InputStart);
        blocks.apply(Marker::OutputStart);
        let text = "你好\r\n".as_bytes();
        blocks.capture(&text[..4]);
        blocks.capture(&text[4..]);
        blocks.apply(Marker::Finished(Some(0)));
        assert_eq!(blocks.blocks()[0].output, "你好\n");
    }

    #[test]
    fn two_commands_make_two_blocks() {
        let stream = format!(
            "{}{}{}first\r\n{}{}{}{}second\r\n{}",
            osc("A"),
            osc("B"),
            osc("C"),
            osc("D;0"),
            osc("A"),
            osc("B"),
            osc("C"),
            osc("D;1")
        );
        let blocks = run_stream(&stream);
        assert_eq!(blocks.blocks().len(), 2);
        assert_eq!(blocks.blocks()[0].output, "first\n");
        assert_eq!(blocks.blocks()[1].output, "second\n");
        assert_eq!(blocks.blocks()[1].exit_code, Some(1));
    }

    #[test]
    fn prompt_redraw_does_not_regress_state() {
        let stream = format!(
            "{}$ {}echo hi\r\n{}{}hi\r\n{}{}",
            osc("A"),
            osc("B"),
            osc("C"),
            osc("B"),
            osc("D;0"),
            osc("A")
        );
        let blocks = run_stream(&stream);
        // 运行中重绘的 B 被忽略；D 之后的 A 开新块。
        assert_eq!(blocks.blocks()[0].state, BlockState::Done);
        assert_eq!(blocks.blocks()[0].command, "echo hi\n");
        assert_eq!(blocks.blocks()[0].output, "hi\n");
        assert_eq!(blocks.blocks().len(), 2);
    }

    #[test]
    fn prompt_text_is_not_captured() {
        let stream = format!(
            "{}my-prompt$ {}ls\r\n{}{}",
            osc("A"),
            osc("B"),
            osc("C"),
            osc("D;0")
        );
        let blocks = run_stream(&stream);
        assert_eq!(blocks.blocks()[0].command, "ls\n");
        assert_eq!(blocks.blocks()[0].output, "");
    }
}