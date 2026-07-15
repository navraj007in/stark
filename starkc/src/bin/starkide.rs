//! A small, dependency-free, Turbo C++ inspired terminal IDE for STARK.
//!
//! The UI intentionally uses ANSI escape sequences and `stty` rather than a
//! TUI crate so the compiler keeps its zero-dependency bootstrap footprint.

use starkc::diag::Severity;
use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::cmp::min;
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BLUE: &str = "\x1b[44m";
const LIGHT_BLUE: &str = "\x1b[104m";
const WHITE: &str = "\x1b[97m";
const BLACK: &str = "\x1b[30m";
const YELLOW: &str = "\x1b[93m";
const CYAN: &str = "\x1b[96m";
const RED: &str = "\x1b[91m";
const RESET: &str = "\x1b[0m";

const MENUS: &[(&str, &[(&str, Action)])] = &[
    (
        "File",
        &[
            ("New", Action::New),
            ("Open...", Action::Open),
            ("Save", Action::Save),
            ("Save As...", Action::SaveAs),
            ("Quit", Action::Quit),
        ],
    ),
    (
        "Edit",
        &[
            ("Undo", Action::Unavailable),
            ("Cut", Action::Unavailable),
            ("Copy", Action::Unavailable),
            ("Paste", Action::Unavailable),
        ],
    ),
    (
        "Search",
        &[
            ("Find...", Action::Unavailable),
            ("Go to line...", Action::Unavailable),
        ],
    ),
    (
        "Run",
        &[
            ("Compile", Action::Compile),
            ("Run", Action::Run),
            ("Clear messages", Action::ClearOutput),
        ],
    ),
    (
        "Options",
        &[
            ("Toggle messages", Action::ToggleOutput),
            ("Program mode", Action::ProgramMode),
            ("Snippet mode", Action::SnippetMode),
        ],
    ),
    (
        "Help",
        &[("Keyboard", Action::Help), ("About", Action::About)],
    ),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    New,
    Open,
    Save,
    SaveAs,
    Quit,
    Compile,
    Run,
    ClearOutput,
    ToggleOutput,
    ProgramMode,
    SnippetMode,
    Help,
    About,
    Unavailable,
}

enum ScreenMode {
    Edit,
    Menu {
        menu: usize,
        item: usize,
    },
    Prompt {
        title: &'static str,
        value: String,
        action: Action,
    },
    Message {
        title: &'static str,
        lines: Vec<String>,
    },
}

struct Editor {
    lines: Vec<String>,
    row: usize,
    col: usize,
    scroll_row: usize,
    scroll_col: usize,
    path: Option<PathBuf>,
    dirty: bool,
}

impl Editor {
    fn empty() -> Self {
        Self {
            lines: vec![String::new()],
            row: 0,
            col: 0,
            scroll_row: 0,
            scroll_col: 0,
            path: None,
            dirty: false,
        }
    }

    fn open(path: PathBuf) -> io::Result<Self> {
        let source = fs::read_to_string(&path)?;
        let mut lines: Vec<String> = source.lines().map(str::to_owned).collect();
        if source.ends_with('\n') {
            lines.push(String::new());
        }
        if lines.is_empty() {
            lines.push(String::new());
        }
        Ok(Self {
            lines,
            row: 0,
            col: 0,
            scroll_row: 0,
            scroll_col: 0,
            path: Some(path),
            dirty: false,
        })
    }

    fn source(&self) -> String {
        self.lines.join("\n")
    }
    fn line_len(&self) -> usize {
        self.lines[self.row].chars().count()
    }
    fn byte_col(&self) -> usize {
        self.lines[self.row]
            .char_indices()
            .nth(self.col)
            .map_or(self.lines[self.row].len(), |(i, _)| i)
    }

    fn insert(&mut self, ch: char) {
        let byte = self.byte_col();
        self.lines[self.row].insert(byte, ch);
        self.col += 1;
        self.dirty = true;
    }

    fn newline(&mut self) {
        let byte = self.byte_col();
        let tail = self.lines[self.row].split_off(byte);
        self.row += 1;
        self.lines.insert(self.row, tail);
        self.col = 0;
        self.dirty = true;
    }

    fn backspace(&mut self) {
        if self.col > 0 {
            let end = self.byte_col();
            self.col -= 1;
            let start = self.byte_col();
            self.lines[self.row].replace_range(start..end, "");
            self.dirty = true;
        } else if self.row > 0 {
            let current = self.lines.remove(self.row);
            self.row -= 1;
            self.col = self.lines[self.row].chars().count();
            self.lines[self.row].push_str(&current);
            self.dirty = true;
        }
    }

    fn delete(&mut self) {
        let len = self.line_len();
        if self.col < len {
            let start = self.byte_col();
            self.col += 1;
            let end = self.byte_col();
            self.col -= 1;
            self.lines[self.row].replace_range(start..end, "");
            self.dirty = true;
        } else if self.row + 1 < self.lines.len() {
            let next = self.lines.remove(self.row + 1);
            self.lines[self.row].push_str(&next);
            self.dirty = true;
        }
    }
}

struct App {
    editor: Editor,
    mode: ScreenMode,
    output: Vec<String>,
    show_output: bool,
    parse_mode: ParseMode,
    status: String,
    quit: bool,
}

impl App {
    fn new(editor: Editor) -> Self {
        Self {
            editor,
            mode: ScreenMode::Edit,
            output: vec!["Ready. F9 compiles the current buffer.".into()],
            show_output: true,
            parse_mode: ParseMode::Program,
            status: "F10 Menu  F2 Save  F9 Compile  Ctrl+Q Quit".into(),
            quit: false,
        }
    }

    fn save(&mut self) {
        let Some(path) = self.editor.path.clone() else {
            self.mode = ScreenMode::Prompt {
                title: " Save file as ",
                value: "untitled.stark".into(),
                action: Action::SaveAs,
            };
            return;
        };
        match fs::write(&path, self.editor.source()) {
            Ok(()) => {
                self.editor.dirty = false;
                self.status = format!("Saved {}", path.display());
            }
            Err(e) => self.show_error("Save failed", &e.to_string()),
        }
    }

    fn compile(&mut self, run: bool) {
        let name = self
            .editor
            .path
            .as_ref()
            .and_then(|p| p.to_str())
            .unwrap_or("untitled.stark");
        let file = Arc::new(SourceFile::new(name, self.editor.source()));
        let (tree, mut diagnostics) = parse(&file, self.parse_mode);
        if diagnostics.is_empty() {
            let (hir, mut resolution_diagnostics) = resolve(&tree, file.clone());
            diagnostics.append(&mut resolution_diagnostics);
            if diagnostics.is_empty() {
                let checked = typecheck::analyze(&hir, file.clone());
                diagnostics.extend(checked.diagnostics.clone());
                if run
                    && checked
                        .diagnostics
                        .iter()
                        .all(|diagnostic| diagnostic.severity != Severity::Error)
                {
                    match interp::run(&hir, file.clone(), &checked.tables) {
                        Ok(execution) => {
                            self.output.clear();
                            self.output.push(format!("Running {name}"));
                            self.output
                                .extend(execution.output.lines().map(str::to_owned));
                            self.output.push("Program exited successfully.".into());
                            self.status = "Run successful".into();
                            self.show_output = true;
                            return;
                        }
                        Err(error) => diagnostics.push(starkc::diag::Diagnostic::error(
                            format!("runtime error: {}", error.message),
                            error.span,
                        )),
                    }
                }
            }
        }
        self.output.clear();
        let error_count = diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == Severity::Error)
            .count();
        if error_count == 0 {
            self.output.push(format!("Compiling {name}"));
            for diagnostic in &diagnostics {
                self.output
                    .extend(diagnostic.render(&file).lines().map(str::to_owned));
            }
            self.output.push(format!(
                "Success: syntax and semantic checks passed (0 errors, {} warning(s)).",
                diagnostics.len()
            ));
            self.status = "Compile successful".into();
        } else {
            self.output.push(format!("Compiling {name}"));
            for diagnostic in &diagnostics {
                self.output
                    .extend(diagnostic.render(&file).lines().map(str::to_owned));
            }
            self.output
                .push(format!("Build failed: {error_count} error(s)."));
            self.status = format!("{error_count} compile error(s)");
        }
        self.show_output = true;
    }

    fn show_error(&mut self, title: &'static str, message: &str) {
        self.mode = ScreenMode::Message {
            title,
            lines: vec![
                message.to_owned(),
                String::new(),
                "Press Enter or Esc".into(),
            ],
        };
    }

    fn action(&mut self, action: Action) {
        self.mode = ScreenMode::Edit;
        match action {
            Action::New => {
                self.editor = Editor::empty();
                self.status = "New file".into();
            }
            Action::Open => {
                self.mode = ScreenMode::Prompt {
                    title: " Open file ",
                    value: String::new(),
                    action,
                }
            }
            Action::Save => self.save(),
            Action::SaveAs => {
                let value = self
                    .editor
                    .path
                    .as_ref()
                    .map_or_else(|| "untitled.stark".into(), |p| p.display().to_string());
                self.mode = ScreenMode::Prompt {
                    title: " Save file as ",
                    value,
                    action,
                };
            }
            Action::Quit => self.quit = true,
            Action::Compile => self.compile(false),
            Action::Run => self.compile(true),
            Action::ClearOutput => self.output.clear(),
            Action::ToggleOutput => self.show_output = !self.show_output,
            Action::ProgramMode => {
                self.parse_mode = ParseMode::Program;
                self.status = "Parser mode: Program".into();
            }
            Action::SnippetMode => {
                self.parse_mode = ParseMode::Snippet;
                self.status = "Parser mode: Snippet".into();
            }
            Action::Help => {
                self.mode = ScreenMode::Message {
                    title: " Keyboard help ",
                    lines: vec![
                        "F10 / Alt+letter   Open menus".into(),
                        "Arrow keys         Navigate editor or menus".into(),
                        "F2                 Save".into(),
                        "F9                 Compile".into(),
                        "Ctrl+F9            Compile and run".into(),
                        "Ctrl+Q             Quit".into(),
                        "Esc                Close menu/dialog".into(),
                    ],
                }
            }
            Action::About => {
                self.mode = ScreenMode::Message {
                    title: " About STARK IDE ",
                    lines: vec![
                        "STARK IDE 0.1".into(),
                        "A Turbo C++ inspired terminal workbench.".into(),
                        "Directly connected to the starkc parser.".into(),
                    ],
                }
            }
            Action::Unavailable => {
                self.status = "This command is planned but not implemented yet.".into()
            }
        }
    }

    fn finish_prompt(&mut self) {
        let ScreenMode::Prompt { value, action, .. } =
            std::mem::replace(&mut self.mode, ScreenMode::Edit)
        else {
            return;
        };
        if value.trim().is_empty() {
            return;
        }
        let path = PathBuf::from(value.trim());
        match action {
            Action::Open => match Editor::open(path.clone()) {
                Ok(editor) => {
                    self.editor = editor;
                    self.status = format!("Opened {}", path.display());
                }
                Err(e) => self.show_error("Open failed", &e.to_string()),
            },
            Action::SaveAs => {
                self.editor.path = Some(path);
                self.save();
            }
            _ => {}
        }
    }
}

#[derive(Debug)]
enum Key {
    Char(char),
    Alt(char),
    Ctrl(char),
    CtrlF9,
    Enter,
    Esc,
    Backspace,
    Delete,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    F(u8),
}

fn poll_key(input: &mut impl Read) -> io::Result<Option<Key>> {
    let mut one = [0u8; 1];
    if input.read(&mut one)? == 0 {
        return Ok(None);
    }
    read_key(one[0], input).map(Some)
}

fn read_key(first_byte: u8, input: &mut impl Read) -> io::Result<Key> {
    let mut one = [0u8; 1];
    match first_byte {
        b'\r' | b'\n' => Ok(Key::Enter),
        8 | 127 => Ok(Key::Backspace),
        1..=26 => Ok(Key::Ctrl((b'a' + first_byte - 1) as char)),
        27 => {
            let mut seq = [0u8; 1];
            if input.read(&mut seq)? == 0 {
                return Ok(Key::Esc);
            }
            if seq[0] != b'[' && seq[0] != b'O' {
                return Ok(Key::Alt((seq[0] as char).to_ascii_lowercase()));
            }
            let mut tail = Vec::new();
            loop {
                input.read_exact(&mut seq)?;
                tail.push(seq[0]);
                if seq[0].is_ascii_alphabetic() || seq[0] == b'~' {
                    break;
                }
                if tail.len() > 6 {
                    return Ok(Key::Esc);
                }
            }
            match tail.as_slice() {
                b"A" => Ok(Key::Up),
                b"B" => Ok(Key::Down),
                b"C" => Ok(Key::Right),
                b"D" => Ok(Key::Left),
                b"H" | b"1~" => Ok(Key::Home),
                b"F" | b"4~" => Ok(Key::End),
                b"3~" => Ok(Key::Delete),
                b"5~" => Ok(Key::PageUp),
                b"6~" => Ok(Key::PageDown),
                b"12~" | b"Q" => Ok(Key::F(2)),
                b"20~" => Ok(Key::F(9)),
                b"20;5~" => Ok(Key::CtrlF9),
                b"21~" => Ok(Key::F(10)),
                _ => Ok(Key::Esc),
            }
        }
        byte if byte.is_ascii() => Ok(Key::Char(byte as char)),
        first => {
            let width = if first & 0b1111_0000 == 0b1111_0000 {
                4
            } else if first & 0b1110_0000 == 0b1110_0000 {
                3
            } else {
                2
            };
            let mut bytes = vec![first];
            for _ in 1..width {
                input.read_exact(&mut one)?;
                bytes.push(one[0]);
            }
            Ok(Key::Char(
                std::str::from_utf8(&bytes)
                    .ok()
                    .and_then(|s| s.chars().next())
                    .unwrap_or('?'),
            ))
        }
    }
}

fn terminal_size() -> (usize, usize) {
    // `Command::output` closes stdin by default; stty reads the terminal from
    // stdin, so it must be inherited explicitly or every probe falls back.
    let output = Command::new("stty")
        .arg("size")
        .stdin(Stdio::inherit())
        .output();
    let text = output
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    let mut fields = text
        .split_whitespace()
        .filter_map(|s| s.parse::<usize>().ok());
    (fields.next().unwrap_or(25), fields.next().unwrap_or(80))
}

struct Terminal;
impl Terminal {
    fn enter() -> io::Result<Self> {
        if !Command::new("stty")
            // Timed reads let the loop detect a resize without requiring a
            // keypress. `time 1` is one decisecond.
            .args(["raw", "-echo", "min", "0", "time", "1"])
            .status()?
            .success()
        {
            return Err(io::Error::other("could not put terminal in raw mode"));
        }
        print!("\x1b[?1049h\x1b[?25l");
        io::stdout().flush()?;
        Ok(Self)
    }
}
impl Drop for Terminal {
    fn drop(&mut self) {
        // `sane` is portable across BSD/macOS and GNU stty and guarantees the
        // user is not left in raw/no-echo mode after a panic or normal exit.
        let _ = Command::new("stty").arg("sane").status();
        print!("{RESET}\x1b[?25h\x1b[?1049l");
        let _ = io::stdout().flush();
    }
}

fn clip(text: &str, start: usize, width: usize) -> String {
    text.chars().skip(start).take(width).collect()
}
fn fit(text: &str, width: usize) -> String {
    let clipped: String = text.chars().take(width).collect();
    format!("{clipped:<width$}")
}
fn at(row: usize, col: usize) -> String {
    format!("\x1b[{row};{col}H")
}

fn draw_box(out: &mut String, top: usize, left: usize, height: usize, width: usize, title: &str) {
    out.push_str(&format!(
        "{}{}┌{}┐",
        at(top, left),
        WHITE,
        "─".repeat(width.saturating_sub(2))
    ));
    for row in top + 1..top + height - 1 {
        out.push_str(&format!(
            "{}│{}│",
            at(row, left),
            " ".repeat(width.saturating_sub(2))
        ));
    }
    out.push_str(&format!(
        "{}└{}┘",
        at(top + height - 1, left),
        "─".repeat(width.saturating_sub(2))
    ));
    if !title.is_empty() {
        out.push_str(&format!(
            "{}{}",
            at(top, left + 2),
            fit(title, min(title.len(), width.saturating_sub(4)))
        ));
    }
}

fn render(app: &mut App, (rows, cols): (usize, usize), clear: bool) -> io::Result<()> {
    let mut out = format!(
        "\x1b[?25l{BLUE}{WHITE}{}",
        if clear { "\x1b[2J\x1b[H" } else { "\x1b[H" }
    );
    if rows < 10 || cols < 40 {
        out.push_str(&format!("{}{}", at(1, 1), fit(" STARK IDE ", cols)));
        if rows > 2 {
            out.push_str(&format!(
                "{}{}",
                at(3, 1),
                fit(
                    &format!("Terminal too small: {cols}x{rows} (minimum 40x10)"),
                    cols
                )
            ));
        }
        out.push_str(RESET);
        io::stdout().write_all(out.as_bytes())?;
        return io::stdout().flush();
    }
    let output_height = if app.show_output { min(8, rows / 3) } else { 0 };
    let editor_top = 2;
    let editor_height = rows.saturating_sub(output_height + 3);
    let text_height = editor_height.saturating_sub(2);
    let gutter = 6usize;
    let text_width = cols.saturating_sub(gutter + 2);
    if app.editor.row < app.editor.scroll_row {
        app.editor.scroll_row = app.editor.row;
    }
    if app.editor.row >= app.editor.scroll_row + text_height {
        app.editor.scroll_row = app.editor.row + 1 - text_height;
    }
    if app.editor.col < app.editor.scroll_col {
        app.editor.scroll_col = app.editor.col;
    }
    if app.editor.col >= app.editor.scroll_col + text_width {
        app.editor.scroll_col = app.editor.col + 1 - text_width;
    }

    out.push_str(&format!("{}{}{}", at(1, 1), LIGHT_BLUE, BLACK));
    let mut menu_width = 0;
    for (i, (name, _)) in MENUS.iter().enumerate() {
        let active = matches!(app.mode, ScreenMode::Menu { menu, .. } if menu == i);
        if active {
            out.push_str(&format!("{BLUE}{WHITE} {name} {LIGHT_BLUE}{BLACK}"));
        } else {
            out.push_str(&format!(" {name} "));
        }
        menu_width += name.chars().count() + 2;
    }
    out.push_str(&" ".repeat(cols.saturating_sub(menu_width)));

    let title = app
        .editor
        .path
        .as_ref()
        .map_or("UNTITLED.STARK".into(), |p| p.display().to_string());
    draw_box(
        &mut out,
        editor_top,
        1,
        editor_height,
        cols,
        &format!(" {title}{} ", if app.editor.dirty { " *" } else { "" }),
    );
    for screen_row in 0..text_height {
        let line_index = app.editor.scroll_row + screen_row;
        let y = editor_top + 1 + screen_row;
        out.push_str(&format!("{}{}{}", at(y, 2), BLUE, CYAN));
        if let Some(line) = app.editor.lines.get(line_index) {
            out.push_str(&format!("{:>4} │", line_index + 1));
            out.push_str(WHITE);
            out.push_str(&fit(
                &clip(line, app.editor.scroll_col, text_width),
                text_width,
            ));
        } else {
            out.push_str(&fit("     │", cols - 2));
        }
    }

    if app.show_output {
        let top = editor_top + editor_height;
        draw_box(&mut out, top, 1, output_height, cols, " Messages ");
        for (i, line) in app
            .output
            .iter()
            .rev()
            .take(output_height.saturating_sub(2))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .enumerate()
        {
            let color = if line.contains("Error") || line.contains("failed") {
                RED
            } else if line.contains("Success") {
                YELLOW
            } else {
                WHITE
            };
            out.push_str(&format!(
                "{}{}{}",
                at(top + 1 + i, 2),
                color,
                fit(line, cols - 2)
            ));
        }
    }

    out.push_str(&format!(
        "{}{}{}{}",
        at(rows, 1),
        LIGHT_BLUE,
        BLACK,
        fit(
            &format!(
                " {}   Ln {}, Col {}   {} ",
                app.status,
                app.editor.row + 1,
                app.editor.col + 1,
                if app.parse_mode == ParseMode::Program {
                    "PROGRAM"
                } else {
                    "SNIPPET"
                }
            ),
            cols
        )
    ));

    match &app.mode {
        ScreenMode::Menu { menu, item } => {
            let left = MENUS
                .iter()
                .take(*menu)
                .map(|(n, _)| n.len() + 2)
                .sum::<usize>()
                + 1;
            let items = MENUS[*menu].1;
            let width = items.iter().map(|(n, _)| n.len()).max().unwrap_or(8) + 4;
            draw_box(&mut out, 2, left, items.len() + 2, width, "");
            for (i, (name, _)) in items.iter().enumerate() {
                let (bg, fg) = if i == *item {
                    (LIGHT_BLUE, BLACK)
                } else {
                    (BLUE, WHITE)
                };
                out.push_str(&format!(
                    "{}{}{}{}",
                    at(3 + i, left + 1),
                    bg,
                    fg,
                    fit(name, width - 2)
                ));
            }
        }
        ScreenMode::Prompt { title, value, .. } => {
            let width = min(64, cols.saturating_sub(4));
            let left = (cols - width) / 2 + 1;
            let top = rows / 2 - 2;
            draw_box(&mut out, top, left, 5, width, title);
            out.push_str(&format!("{}{}{}Path:", at(top + 1, left + 2), BLUE, WHITE));
            out.push_str(&format!(
                "{}{}{}{}",
                at(top + 2, left + 2),
                LIGHT_BLUE,
                BLACK,
                fit(value, width - 4)
            ));
            out.push_str(&format!(
                "{}{}Enter accepts · Esc cancels",
                at(top + 3, left + 2),
                WHITE
            ));
            out.push_str(&format!(
                "{}\x1b[?25h",
                at(top + 2, left + 2 + min(value.chars().count(), width - 5))
            ));
        }
        ScreenMode::Message { title, lines } => {
            let width = min(64, cols.saturating_sub(4));
            let height = min(lines.len() + 4, rows.saturating_sub(2));
            let left = (cols - width) / 2 + 1;
            let top = (rows - height) / 2 + 1;
            draw_box(&mut out, top, left, height, width, title);
            for (i, line) in lines.iter().take(height - 2).enumerate() {
                out.push_str(&format!(
                    "{}{}",
                    at(top + 1 + i, left + 2),
                    fit(line, width - 4)
                ));
            }
        }
        ScreenMode::Edit => {
            let y = editor_top + 1 + app.editor.row.saturating_sub(app.editor.scroll_row);
            let x = gutter + 2 + app.editor.col.saturating_sub(app.editor.scroll_col);
            out.push_str(&format!("{}\x1b[?25h", at(y, min(x, cols - 1))));
        }
    }
    out.push_str(RESET);
    io::stdout().write_all(out.as_bytes())?;
    io::stdout().flush()
}

fn handle_key(app: &mut App, key: Key) {
    match &mut app.mode {
        ScreenMode::Message { .. } => {
            if matches!(key, Key::Enter | Key::Esc) {
                app.mode = ScreenMode::Edit;
            }
        }
        ScreenMode::Prompt { value, .. } => match key {
            Key::Enter => app.finish_prompt(),
            Key::Esc => app.mode = ScreenMode::Edit,
            Key::Backspace => {
                value.pop();
            }
            Key::Char(ch) => value.push(ch),
            _ => {}
        },
        ScreenMode::Menu { menu, item } => match key {
            Key::Esc | Key::F(10) => app.mode = ScreenMode::Edit,
            Key::Left => {
                *menu = if *menu == 0 {
                    MENUS.len() - 1
                } else {
                    *menu - 1
                };
                *item = 0;
            }
            Key::Right => {
                *menu = (*menu + 1) % MENUS.len();
                *item = 0;
            }
            Key::Up => {
                *item = if *item == 0 {
                    MENUS[*menu].1.len() - 1
                } else {
                    *item - 1
                };
            }
            Key::Down => {
                *item = (*item + 1) % MENUS[*menu].1.len();
            }
            Key::Enter => {
                let action = MENUS[*menu].1[*item].1;
                app.action(action);
            }
            _ => {}
        },
        ScreenMode::Edit => match key {
            Key::Ctrl('q') => app.action(Action::Quit),
            Key::Ctrl('s') | Key::F(2) => app.save(),
            Key::F(9) => app.compile(false),
            Key::CtrlF9 => app.compile(true),
            Key::F(10) => app.mode = ScreenMode::Menu { menu: 0, item: 0 },
            Key::Esc => app.mode = ScreenMode::Menu { menu: 0, item: 0 },
            Key::Alt(ch) => {
                if let Some(menu) = MENUS
                    .iter()
                    .position(|(name, _)| name.to_ascii_lowercase().starts_with(ch))
                {
                    app.mode = ScreenMode::Menu { menu, item: 0 };
                }
            }
            Key::Left => {
                if app.editor.col > 0 {
                    app.editor.col -= 1;
                } else if app.editor.row > 0 {
                    app.editor.row -= 1;
                    app.editor.col = app.editor.line_len();
                }
            }
            Key::Right => {
                if app.editor.col < app.editor.line_len() {
                    app.editor.col += 1;
                } else if app.editor.row + 1 < app.editor.lines.len() {
                    app.editor.row += 1;
                    app.editor.col = 0;
                }
            }
            Key::Up => {
                if app.editor.row > 0 {
                    app.editor.row -= 1;
                    app.editor.col = min(app.editor.col, app.editor.line_len());
                }
            }
            Key::Down => {
                if app.editor.row + 1 < app.editor.lines.len() {
                    app.editor.row += 1;
                    app.editor.col = min(app.editor.col, app.editor.line_len());
                }
            }
            Key::Home => app.editor.col = 0,
            Key::End => app.editor.col = app.editor.line_len(),
            Key::PageUp => {
                app.editor.row = app.editor.row.saturating_sub(10);
                app.editor.col = min(app.editor.col, app.editor.line_len());
            }
            Key::PageDown => {
                app.editor.row = min(app.editor.row + 10, app.editor.lines.len() - 1);
                app.editor.col = min(app.editor.col, app.editor.line_len());
            }
            Key::Backspace => app.editor.backspace(),
            Key::Delete => app.editor.delete(),
            Key::Enter => app.editor.newline(),
            Key::Char(ch) => app.editor.insert(ch),
            _ => {}
        },
    }
}

fn main() -> ExitCode {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        eprintln!("starkide must be run in an interactive terminal");
        return ExitCode::FAILURE;
    }
    let editor = match env::args_os().nth(1) {
        Some(path) => match Editor::open(Path::new(&path).to_path_buf()) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("starkide: {e}");
                return ExitCode::FAILURE;
            }
        },
        None => Editor::empty(),
    };
    let _terminal = match Terminal::enter() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("starkide: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut app = App::new(editor);
    let mut stdin = io::stdin().lock();
    let mut last_size = (0, 0);
    let mut last_size_probe = Instant::now() - Duration::from_secs(1);
    let mut redraw = true;
    while !app.quit {
        let size = if last_size == (0, 0) || last_size_probe.elapsed() >= Duration::from_millis(200)
        {
            last_size_probe = Instant::now();
            terminal_size()
        } else {
            last_size
        };
        let resized = size != last_size;
        if redraw || resized {
            if render(&mut app, size, resized).is_err() {
                return ExitCode::FAILURE;
            }
            last_size = size;
            redraw = false;
        }
        match poll_key(&mut stdin) {
            Ok(Some(key)) => {
                handle_key(&mut app, key);
                redraw = true;
            }
            Ok(None) => {}
            Err(_) => break,
        }
    }
    ExitCode::SUCCESS
}

use std::io::IsTerminal;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn editor_handles_unicode_and_line_joins() {
        let mut e = Editor::empty();
        e.insert('λ');
        e.insert('x');
        e.newline();
        e.insert('!');
        assert_eq!(e.source(), "λx\n!");
        e.backspace();
        e.backspace();
        assert_eq!(e.source(), "λx");
        assert_eq!((e.row, e.col), (0, 2));
    }

    #[test]
    fn clipping_is_character_based() {
        assert_eq!(clip("αβγδε", 1, 3), "βγδ");
    }

    #[test]
    fn polling_handles_idle_escape_and_arrow_input() {
        assert!(poll_key(&mut Cursor::new([])).unwrap().is_none());
        assert!(matches!(
            poll_key(&mut Cursor::new([27])).unwrap(),
            Some(Key::Esc)
        ));
        assert!(matches!(
            poll_key(&mut Cursor::new(b"\x1b[A")).unwrap(),
            Some(Key::Up)
        ));
    }

    #[test]
    fn compile_runs_semantic_checks() {
        let mut editor = Editor::empty();
        editor.lines = vec!["fn main() { let x: Int32 = \"wrong\"; }".into()];
        let mut app = App::new(editor);
        app.compile(false);
        assert!(app.status.contains("compile error"));
        assert!(app.output.iter().any(|line| line.contains("E0001")));
    }

    #[test]
    fn run_executes_current_buffer() {
        let mut editor = Editor::empty();
        editor.lines = vec!["fn main() { println(42); }".into()];
        let mut app = App::new(editor);
        app.compile(true);
        assert_eq!(app.status, "Run successful");
        assert!(app.output.iter().any(|line| line == "42"));
        assert!(app
            .output
            .iter()
            .any(|line| line == "Program exited successfully."));
    }
}
