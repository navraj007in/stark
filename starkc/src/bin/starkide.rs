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
            ("Project files...", Action::BrowseProject),
            ("Recent files...", Action::RecentFiles),
            ("Save", Action::Save),
            ("Save As...", Action::SaveAs),
            ("Next buffer", Action::NextBuffer),
            ("Close buffer", Action::CloseBuffer),
            ("Quit", Action::Quit),
        ],
    ),
    (
        "Edit",
        &[
            ("Undo", Action::Undo),
            ("Redo", Action::Redo),
            ("Cut", Action::Cut),
            ("Copy", Action::Copy),
            ("Paste", Action::Paste),
        ],
    ),
    (
        "Search",
        &[
            ("Find...", Action::Find),
            ("Find next", Action::FindNext),
            ("Go to line...", Action::GoToLine),
            ("Next diagnostic", Action::NextDiagnostic),
        ],
    ),
    (
        "Run",
        &[
            ("Compile", Action::Compile),
            ("Run", Action::Run),
            ("Run again", Action::RunAgain),
            ("Show build output", Action::ShowBuildOutput),
            ("Show program output", Action::ShowProgramOutput),
            ("Clear messages", Action::ClearOutput),
        ],
    ),
    (
        "Options",
        &[
            ("Toggle messages", Action::ToggleOutput),
            ("Program mode", Action::ProgramMode),
            ("Snippet mode", Action::SnippetMode),
            ("Set project root...", Action::ProjectRoot),
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
    BrowseProject,
    RecentFiles,
    Save,
    SaveAs,
    NextBuffer,
    CloseBuffer,
    Quit,
    Undo,
    Redo,
    Cut,
    Copy,
    Paste,
    Find,
    FindNext,
    GoToLine,
    NextDiagnostic,
    Compile,
    Run,
    RunAgain,
    ShowBuildOutput,
    ShowProgramOutput,
    ClearOutput,
    ToggleOutput,
    ProgramMode,
    SnippetMode,
    ProjectRoot,
    Help,
    About,
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
    ConfirmDiscard {
        action: Action,
    },
    FilePicker {
        title: &'static str,
        files: Vec<PathBuf>,
        selected: usize,
    },
}

#[derive(Clone)]
struct EditorSnapshot {
    lines: Vec<String>,
    row: usize,
    col: usize,
    dirty: bool,
}

struct Editor {
    lines: Vec<String>,
    row: usize,
    col: usize,
    scroll_row: usize,
    scroll_col: usize,
    path: Option<PathBuf>,
    dirty: bool,
    selection_anchor: Option<(usize, usize)>,
    undo: Vec<EditorSnapshot>,
    redo: Vec<EditorSnapshot>,
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
            selection_anchor: None,
            undo: Vec::new(),
            redo: Vec::new(),
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
            selection_anchor: None,
            undo: Vec::new(),
            redo: Vec::new(),
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

    fn snapshot(&self) -> EditorSnapshot {
        EditorSnapshot {
            lines: self.lines.clone(),
            row: self.row,
            col: self.col,
            dirty: self.dirty,
        }
    }

    fn record_edit(&mut self) {
        self.undo.push(self.snapshot());
        if self.undo.len() > 512 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn restore(&mut self, snapshot: EditorSnapshot) {
        self.lines = snapshot.lines;
        self.row = snapshot.row.min(self.lines.len().saturating_sub(1));
        self.col = snapshot.col.min(self.line_len());
        self.dirty = snapshot.dirty;
        self.selection_anchor = None;
    }

    fn undo(&mut self) -> bool {
        let Some(previous) = self.undo.pop() else {
            return false;
        };
        self.redo.push(self.snapshot());
        self.restore(previous);
        true
    }

    fn redo(&mut self) -> bool {
        let Some(next) = self.redo.pop() else {
            return false;
        };
        self.undo.push(self.snapshot());
        self.restore(next);
        true
    }

    fn normalized_selection(&self) -> Option<((usize, usize), (usize, usize))> {
        let anchor = self.selection_anchor?;
        let cursor = (self.row, self.col);
        (anchor != cursor).then_some(if anchor <= cursor {
            (anchor, cursor)
        } else {
            (cursor, anchor)
        })
    }

    fn selection_contains(&self, row: usize, col: usize) -> bool {
        let Some(((start_row, start_col), (end_row, end_col))) = self.normalized_selection() else {
            return false;
        };
        ((row > start_row) || (row == start_row && col >= start_col))
            && ((row < end_row) || (row == end_row && col < end_col))
    }

    fn selected_text(&self) -> Option<String> {
        let ((start_row, start_col), (end_row, end_col)) = self.normalized_selection()?;
        if start_row == end_row {
            return Some(
                self.lines[start_row]
                    .chars()
                    .skip(start_col)
                    .take(end_col - start_col)
                    .collect(),
            );
        }
        let mut result: String = self.lines[start_row].chars().skip(start_col).collect();
        result.push('\n');
        for row in start_row + 1..end_row {
            result.push_str(&self.lines[row]);
            result.push('\n');
        }
        result.extend(self.lines[end_row].chars().take(end_col));
        Some(result)
    }

    fn delete_selection(&mut self) -> bool {
        let Some(((start_row, start_col), (end_row, end_col))) = self.normalized_selection() else {
            return false;
        };
        let start_byte = char_byte(&self.lines[start_row], start_col);
        let end_byte = char_byte(&self.lines[end_row], end_col);
        if start_row == end_row {
            self.lines[start_row].replace_range(start_byte..end_byte, "");
        } else {
            let suffix = self.lines[end_row][end_byte..].to_string();
            self.lines[start_row].truncate(start_byte);
            self.lines[start_row].push_str(&suffix);
            self.lines.drain(start_row + 1..=end_row);
        }
        self.row = start_row;
        self.col = start_col;
        self.selection_anchor = None;
        self.dirty = true;
        true
    }

    fn paste(&mut self, text: &str) {
        self.record_edit();
        self.delete_selection();
        let byte = self.byte_col();
        let suffix = self.lines[self.row].split_off(byte);
        let parts: Vec<&str> = text.split('\n').collect();
        self.lines[self.row].push_str(parts[0]);
        for part in parts.iter().skip(1) {
            self.row += 1;
            self.lines.insert(self.row, (*part).to_string());
        }
        self.col = self.lines[self.row].chars().count();
        self.lines[self.row].push_str(&suffix);
        self.dirty = true;
    }

    fn insert(&mut self, ch: char) {
        self.record_edit();
        self.delete_selection();
        let byte = self.byte_col();
        self.lines[self.row].insert(byte, ch);
        self.col += 1;
        self.dirty = true;
    }

    fn newline(&mut self) {
        self.record_edit();
        self.delete_selection();
        let byte = self.byte_col();
        let tail = self.lines[self.row].split_off(byte);
        self.row += 1;
        self.lines.insert(self.row, tail);
        self.col = 0;
        self.dirty = true;
    }

    fn backspace(&mut self) {
        if self.normalized_selection().is_some() {
            self.record_edit();
            self.delete_selection();
            return;
        }
        self.record_edit();
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
        if self.normalized_selection().is_some() {
            self.record_edit();
            self.delete_selection();
            return;
        }
        self.record_edit();
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

    fn select_movement(&mut self, selecting: bool) {
        if selecting {
            self.selection_anchor.get_or_insert((self.row, self.col));
        } else {
            self.selection_anchor = None;
        }
    }

    fn move_left(&mut self, selecting: bool) {
        self.select_movement(selecting);
        if self.col > 0 {
            self.col -= 1;
        } else if self.row > 0 {
            self.row -= 1;
            self.col = self.line_len();
        }
    }

    fn move_right(&mut self, selecting: bool) {
        self.select_movement(selecting);
        if self.col < self.line_len() {
            self.col += 1;
        } else if self.row + 1 < self.lines.len() {
            self.row += 1;
            self.col = 0;
        }
    }

    fn move_up(&mut self, selecting: bool) {
        self.select_movement(selecting);
        if self.row > 0 {
            self.row -= 1;
            self.col = min(self.col, self.line_len());
        }
    }

    fn move_down(&mut self, selecting: bool) {
        self.select_movement(selecting);
        if self.row + 1 < self.lines.len() {
            self.row += 1;
            self.col = min(self.col, self.line_len());
        }
    }

    fn find_from(&mut self, query: &str, advance: bool) -> bool {
        if query.is_empty() {
            return false;
        }
        let start_row = self.row;
        let start_col = if advance {
            self.col.saturating_add(1)
        } else {
            self.col
        };
        for offset in 0..self.lines.len() {
            let row = (start_row + offset) % self.lines.len();
            let from = if offset == 0 { start_col } else { 0 };
            let byte = char_byte(&self.lines[row], from.min(self.lines[row].chars().count()));
            if let Some(found) = self.lines[row][byte..].find(query) {
                self.row = row;
                self.col = self.lines[row][..byte + found].chars().count();
                self.selection_anchor = Some((row, self.col + query.chars().count()));
                return true;
            }
        }
        false
    }
}

fn char_byte(text: &str, column: usize) -> usize {
    text.char_indices()
        .nth(column)
        .map_or(text.len(), |(index, _)| index)
}

struct PersistentState {
    show_output: bool,
    snippet_mode: bool,
    project_root: Option<PathBuf>,
    recent_files: Vec<PathBuf>,
}

impl Default for PersistentState {
    fn default() -> Self {
        Self {
            show_output: true,
            snippet_mode: false,
            project_root: None,
            recent_files: Vec::new(),
        }
    }
}

impl PersistentState {
    fn path() -> Option<PathBuf> {
        env::var_os("HOME").map(|home| PathBuf::from(home).join(".starkide-state"))
    }

    fn load() -> Self {
        let Some(path) = Self::path() else {
            return Self::default();
        };
        let Ok(text) = fs::read_to_string(path) else {
            return Self::default();
        };
        let mut state = Self::default();
        for line in text.lines() {
            if let Some(value) = line.strip_prefix("show_output=") {
                state.show_output = value == "true";
            } else if let Some(value) = line.strip_prefix("snippet_mode=") {
                state.snippet_mode = value == "true";
            } else if let Some(value) = line.strip_prefix("project_root=") {
                state.project_root = Some(PathBuf::from(value));
            } else if let Some(value) = line.strip_prefix("recent=") {
                state.recent_files.push(PathBuf::from(value));
            }
        }
        state
    }

    fn save(&self) {
        let Some(path) = Self::path() else { return };
        let mut text = format!(
            "show_output={}\nsnippet_mode={}\n",
            self.show_output, self.snippet_mode
        );
        if let Some(root) = &self.project_root {
            text.push_str(&format!("project_root={}\n", root.display()));
        }
        for recent in &self.recent_files {
            text.push_str(&format!("recent={}\n", recent.display()));
        }
        let _ = fs::write(path, text);
    }
}

struct App {
    editor: Editor,
    buffers: Vec<Editor>,
    mode: ScreenMode,
    output: Vec<String>,
    show_output: bool,
    parse_mode: ParseMode,
    status: String,
    quit: bool,
    clipboard: String,
    find_query: String,
    diagnostic_targets: Vec<(usize, usize, String)>,
    diagnostic_index: usize,
    build_output: Vec<String>,
    program_output: Vec<String>,
    project_root: PathBuf,
    recent_files: Vec<PathBuf>,
    pending_after_save: Option<Action>,
}

impl App {
    fn new(editor: Editor) -> Self {
        let state = PersistentState::load();
        let mut app = Self {
            editor,
            buffers: Vec::new(),
            mode: ScreenMode::Edit,
            output: vec!["Ready. F9 compiles the current buffer.".into()],
            show_output: state.show_output,
            parse_mode: if state.snippet_mode {
                ParseMode::Snippet
            } else {
                ParseMode::Program
            },
            status: "F10 Menu  F2 Save  F4 Diagnostic  F9 Build  Ctrl+F9 Run".into(),
            quit: false,
            clipboard: String::new(),
            find_query: String::new(),
            diagnostic_targets: Vec::new(),
            diagnostic_index: 0,
            build_output: Vec::new(),
            program_output: Vec::new(),
            project_root: state
                .project_root
                .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
            recent_files: state.recent_files,
            pending_after_save: None,
        };
        if let Some(path) = app.editor.path.clone() {
            app.remember_file(path);
        }
        app
    }

    fn persist(&self) {
        PersistentState {
            show_output: self.show_output,
            snippet_mode: self.parse_mode == ParseMode::Snippet,
            project_root: Some(self.project_root.clone()),
            recent_files: self.recent_files.clone(),
        }
        .save();
    }

    fn remember_file(&mut self, path: PathBuf) {
        self.recent_files.retain(|recent| recent != &path);
        self.recent_files.insert(0, path);
        self.recent_files.truncate(10);
        self.persist();
    }

    fn request_destructive(&mut self, action: Action) -> bool {
        if self.editor.dirty || (action == Action::Quit && self.buffers.iter().any(|e| e.dirty)) {
            self.mode = ScreenMode::ConfirmDiscard { action };
            false
        } else {
            true
        }
    }

    fn open_buffer(&mut self, editor: Editor) {
        if self.editor.path.is_none()
            && !self.editor.dirty
            && self.editor.source().is_empty()
            && self.buffers.is_empty()
        {
            self.editor = editor;
            return;
        }
        let previous = std::mem::replace(&mut self.editor, editor);
        self.buffers.push(previous);
    }

    fn next_buffer(&mut self) {
        if self.buffers.is_empty() {
            self.status = "Only one buffer is open".into();
            return;
        }
        let next = self.buffers.remove(0);
        let previous = std::mem::replace(&mut self.editor, next);
        self.buffers.push(previous);
        self.status = format!("Buffer {}/{}", 1, self.buffers.len() + 1);
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
                self.remember_file(path);
            }
            Err(e) => self.show_error("Save failed", &e.to_string()),
        }
    }

    fn continue_after_save(&mut self) {
        if !self.editor.dirty && matches!(self.mode, ScreenMode::Edit) {
            if let Some(action) = self.pending_after_save.take() {
                self.action(action);
            }
        }
    }

    fn compile(&mut self, run: bool) {
        let mut runtime_started = false;
        self.diagnostic_targets.clear();
        self.diagnostic_index = 0;
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
                    runtime_started = true;
                    match interp::run(&hir, file.clone(), &checked.tables) {
                        Ok(execution) => {
                            self.program_output.clear();
                            self.program_output.push(format!("Running {name}"));
                            self.program_output
                                .extend(execution.output.lines().map(str::to_owned));
                            self.program_output
                                .push("Program exited successfully.".into());
                            self.output = self.program_output.clone();
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
        for diagnostic in &diagnostics {
            let (line, column) = file.line_col(diagnostic.span.lo);
            self.diagnostic_targets
                .push((line - 1, column - 1, diagnostic.message.clone()));
        }
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
        if runtime_started {
            self.program_output = self.output.clone();
        } else {
            self.build_output = self.output.clone();
        }
        self.show_output = true;
    }

    fn next_diagnostic(&mut self) {
        let Some((row, col, message)) = self
            .diagnostic_targets
            .get(self.diagnostic_index % self.diagnostic_targets.len().max(1))
            .cloned()
        else {
            self.status = "No diagnostics".into();
            return;
        };
        self.diagnostic_index = (self.diagnostic_index + 1) % self.diagnostic_targets.len();
        self.editor.row = row.min(self.editor.lines.len().saturating_sub(1));
        self.editor.col = col.min(self.editor.line_len());
        self.editor.selection_anchor = None;
        self.status = format!("Diagnostic: {message}");
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
                self.open_buffer(Editor::empty());
                self.status = "New file".into();
            }
            Action::Open => {
                self.mode = ScreenMode::Prompt {
                    title: " Open file ",
                    value: format!("{}/", self.project_root.display()),
                    action,
                }
            }
            Action::BrowseProject => {
                let mut files = Vec::new();
                collect_stark_files(&self.project_root, 0, &mut files);
                files.sort();
                self.mode = ScreenMode::FilePicker {
                    title: " Project files ",
                    files,
                    selected: 0,
                };
            }
            Action::RecentFiles => {
                self.mode = ScreenMode::FilePicker {
                    title: " Recent files ",
                    files: self.recent_files.clone(),
                    selected: 0,
                };
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
            Action::NextBuffer => self.next_buffer(),
            Action::CloseBuffer => {
                if !self.request_destructive(action) {
                    return;
                }
                self.editor = self.buffers.pop().unwrap_or_else(Editor::empty);
                self.status = "Buffer closed".into();
            }
            Action::Quit => {
                if self.request_destructive(action) {
                    self.persist();
                    self.quit = true;
                }
            }
            Action::Undo => {
                self.status = if self.editor.undo() {
                    "Undo".into()
                } else {
                    "Nothing to undo".into()
                }
            }
            Action::Redo => {
                self.status = if self.editor.redo() {
                    "Redo".into()
                } else {
                    "Nothing to redo".into()
                }
            }
            Action::Copy => {
                self.clipboard = self
                    .editor
                    .selected_text()
                    .unwrap_or_else(|| self.editor.lines[self.editor.row].clone());
                self.status = format!("Copied {} character(s)", self.clipboard.chars().count());
            }
            Action::Cut => {
                self.clipboard = self
                    .editor
                    .selected_text()
                    .unwrap_or_else(|| self.editor.lines[self.editor.row].clone());
                self.editor.record_edit();
                if !self.editor.delete_selection() {
                    self.editor.lines[self.editor.row].clear();
                    self.editor.col = 0;
                    self.editor.dirty = true;
                }
                self.status = "Cut selection".into();
            }
            Action::Paste => {
                if self.clipboard.is_empty() {
                    self.status = "Clipboard is empty".into();
                } else {
                    self.editor.paste(&self.clipboard);
                    self.status = "Pasted".into();
                }
            }
            Action::Find => {
                self.mode = ScreenMode::Prompt {
                    title: " Find ",
                    value: self.find_query.clone(),
                    action,
                };
            }
            Action::FindNext => {
                if self.find_query.is_empty() {
                    self.action(Action::Find);
                } else if self.editor.find_from(&self.find_query, true) {
                    self.status = format!("Found '{}'", self.find_query);
                } else {
                    self.status = format!("'{}' not found", self.find_query);
                }
            }
            Action::GoToLine => {
                self.mode = ScreenMode::Prompt {
                    title: " Go to line ",
                    value: (self.editor.row + 1).to_string(),
                    action,
                };
            }
            Action::NextDiagnostic => self.next_diagnostic(),
            Action::Compile => self.compile(false),
            Action::Run => self.compile(true),
            Action::RunAgain => self.compile(true),
            Action::ShowBuildOutput => {
                self.output = self.build_output.clone();
                self.show_output = true;
                self.status = "Build output".into();
            }
            Action::ShowProgramOutput => {
                self.output = self.program_output.clone();
                self.show_output = true;
                self.status = "Program output".into();
            }
            Action::ClearOutput => self.output.clear(),
            Action::ToggleOutput => {
                self.show_output = !self.show_output;
                self.persist();
            }
            Action::ProgramMode => {
                self.parse_mode = ParseMode::Program;
                self.status = "Parser mode: Program".into();
                self.persist();
            }
            Action::SnippetMode => {
                self.parse_mode = ParseMode::Snippet;
                self.status = "Parser mode: Snippet".into();
                self.persist();
            }
            Action::ProjectRoot => {
                self.mode = ScreenMode::Prompt {
                    title: " Project root ",
                    value: self.project_root.display().to_string(),
                    action,
                };
            }
            Action::Help => {
                self.mode = ScreenMode::Message {
                    title: " Keyboard help ",
                    lines: vec![
                        "F10 / Alt+letter   Open menus".into(),
                        "Arrow keys         Navigate editor or menus".into(),
                        "F2                 Save".into(),
                        "Ctrl+N/O/S         New / Open / Save".into(),
                        "Ctrl+Z/Y           Undo / Redo".into(),
                        "Ctrl+X/C/V         Cut / Copy / Paste".into(),
                        "Shift+Arrows       Select text".into(),
                        "Ctrl+F / Ctrl+G    Find / Go to line".into(),
                        "F4                 Next diagnostic".into(),
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
                        "Builds and runs through the starkc Gate 3 pipeline.".into(),
                    ],
                }
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
                    self.open_buffer(editor);
                    self.project_root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
                    self.remember_file(path.clone());
                    self.status = format!("Opened {}", path.display());
                }
                Err(e) => self.show_error("Open failed", &e.to_string()),
            },
            Action::SaveAs => {
                self.editor.path = Some(path.clone());
                self.save();
                self.remember_file(path);
                self.continue_after_save();
            }
            Action::Find => {
                self.find_query = value;
                if self.editor.find_from(&self.find_query, false) {
                    self.status = format!("Found '{}'", self.find_query);
                } else {
                    self.status = format!("'{}' not found", self.find_query);
                }
            }
            Action::GoToLine => match value.trim().parse::<usize>() {
                Ok(line) if line > 0 && line <= self.editor.lines.len() => {
                    self.editor.row = line - 1;
                    self.editor.col = self.editor.col.min(self.editor.line_len());
                    self.editor.selection_anchor = None;
                    self.status = format!("Line {line}");
                }
                _ => self.show_error("Invalid line", "Enter a line number in the current file."),
            },
            Action::ProjectRoot => {
                if path.is_dir() {
                    self.project_root = path;
                    self.persist();
                    self.status = format!("Project root: {}", self.project_root.display());
                } else {
                    self.show_error(
                        "Invalid project root",
                        "The selected path is not a directory.",
                    );
                }
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
    ShiftUp,
    ShiftDown,
    ShiftLeft,
    ShiftRight,
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
                b"1;2A" => Ok(Key::ShiftUp),
                b"1;2B" => Ok(Key::ShiftDown),
                b"1;2C" => Ok(Key::ShiftRight),
                b"1;2D" => Ok(Key::ShiftLeft),
                b"H" | b"1~" => Ok(Key::Home),
                b"F" | b"4~" => Ok(Key::End),
                b"3~" => Ok(Key::Delete),
                b"5~" => Ok(Key::PageUp),
                b"6~" => Ok(Key::PageDown),
                b"12~" | b"Q" => Ok(Key::F(2)),
                b"13~" | b"R" => Ok(Key::F(3)),
                b"14~" | b"S" => Ok(Key::F(4)),
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

fn collect_stark_files(root: &Path, depth: usize, files: &mut Vec<PathBuf>) {
    if depth > 4 || files.len() >= 500 {
        return;
    }
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if !matches!(
                path.file_name().and_then(|name| name.to_str()),
                Some(".git" | "target")
            ) {
                collect_stark_files(&path, depth + 1, files);
            }
        } else if path
            .extension()
            .is_some_and(|extension| extension == "stark")
        {
            files.push(path);
            if files.len() >= 500 {
                return;
            }
        }
    }
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
    let clipped = clip(text, 0, width);
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

#[derive(Debug, PartialEq, Eq)]
struct Layout {
    output_height: usize,
    editor_height: usize,
    text_height: usize,
    text_width: usize,
}

fn calculate_layout(rows: usize, cols: usize, show_output: bool) -> Option<Layout> {
    if rows < 14 || cols < 48 {
        return None;
    }
    let output_height = if show_output {
        min(12, (rows / 3).max(3))
    } else {
        0
    };
    let editor_height = rows.saturating_sub(output_height + 3).max(4);
    Some(Layout {
        output_height,
        editor_height,
        text_height: editor_height.saturating_sub(2),
        text_width: cols.saturating_sub(8),
    })
}

fn render(app: &mut App, (rows, cols): (usize, usize), clear: bool) -> io::Result<()> {
    let mut out = format!(
        "\x1b[?25l{BLUE}{WHITE}{}",
        if clear { "\x1b[2J\x1b[H" } else { "\x1b[H" }
    );
    let Some(layout) = calculate_layout(rows, cols, app.show_output) else {
        out.push_str(&format!("{}{}", at(1, 1), fit(" STARK IDE ", cols)));
        if rows > 2 {
            out.push_str(&format!(
                "{}{}",
                at(3, 1),
                fit(
                    &format!("Terminal too small: {cols}x{rows} (minimum 48x14)"),
                    cols
                )
            ));
        }
        out.push_str(RESET);
        io::stdout().write_all(out.as_bytes())?;
        return io::stdout().flush();
    };
    let output_height = layout.output_height;
    let editor_top = 2;
    let editor_height = layout.editor_height;
    let text_height = layout.text_height;
    let gutter = 6usize;
    let text_width = layout.text_width;
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
        &format!(
            " {title}{}  [1/{}] ",
            if app.editor.dirty { " *" } else { "" },
            app.buffers.len() + 1
        ),
    );
    for screen_row in 0..text_height {
        let line_index = app.editor.scroll_row + screen_row;
        let y = editor_top + 1 + screen_row;
        out.push_str(&format!("{}{}{}", at(y, 2), BLUE, CYAN));
        if let Some(line) = app.editor.lines.get(line_index) {
            out.push_str(&format!("{:>4} │", line_index + 1));
            out.push_str(WHITE);
            let visible: Vec<char> = line
                .chars()
                .skip(app.editor.scroll_col)
                .take(text_width)
                .collect();
            for (offset, ch) in visible.iter().enumerate() {
                if app
                    .editor
                    .selection_contains(line_index, app.editor.scroll_col + offset)
                {
                    out.push_str(&format!("{LIGHT_BLUE}{BLACK}{ch}{BLUE}{WHITE}"));
                } else {
                    out.push(*ch);
                }
            }
            out.push_str(&" ".repeat(text_width.saturating_sub(visible.len())));
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
            let label = if title.contains("line") {
                "Line:"
            } else if title.contains("Find") {
                "Text:"
            } else {
                "Path:"
            };
            out.push_str(&format!(
                "{}{}{}{label}",
                at(top + 1, left + 2),
                BLUE,
                WHITE
            ));
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
        ScreenMode::ConfirmDiscard { action } => {
            let width = min(68, cols.saturating_sub(4));
            let left = (cols - width) / 2 + 1;
            let top = rows / 2 - 2;
            draw_box(&mut out, top, left, 6, width, " Unsaved changes ");
            out.push_str(&format!(
                "{}{}",
                at(top + 1, left + 2),
                fit("The current buffer has unsaved changes.", width - 4)
            ));
            out.push_str(&format!(
                "{}{}",
                at(top + 2, left + 2),
                fit("S Save · D Discard · Esc Cancel", width - 4)
            ));
            let operation = match action {
                Action::New => "new file",
                Action::Open => "open file",
                Action::BrowseProject => "browse project files",
                Action::RecentFiles => "open a recent file",
                Action::Quit => "quit",
                Action::CloseBuffer => "close buffer",
                _ => "continue",
            };
            out.push_str(&format!(
                "{}{}",
                at(top + 3, left + 2),
                fit(&format!("Pending: {operation}"), width - 4)
            ));
        }
        ScreenMode::FilePicker {
            title,
            files,
            selected,
        } => {
            let width = min(78, cols.saturating_sub(4));
            let height = min(files.len().max(1) + 4, rows.saturating_sub(2));
            let left = (cols - width) / 2 + 1;
            let top = (rows - height) / 2 + 1;
            draw_box(&mut out, top, left, height, width, title);
            if files.is_empty() {
                out.push_str(&format!(
                    "{}{}",
                    at(top + 1, left + 2),
                    fit("No STARK files found.", width - 4)
                ));
            } else {
                let visible = height.saturating_sub(3);
                let start = selected.saturating_sub(visible.saturating_sub(1));
                for (screen_row, (index, path)) in files
                    .iter()
                    .enumerate()
                    .skip(start)
                    .take(visible)
                    .enumerate()
                {
                    let (bg, fg) = if index == *selected {
                        (LIGHT_BLUE, BLACK)
                    } else {
                        (BLUE, WHITE)
                    };
                    out.push_str(&format!(
                        "{}{}{}{}",
                        at(top + 1 + screen_row, left + 2),
                        bg,
                        fg,
                        fit(&path.display().to_string(), width - 4)
                    ));
                }
            }
            out.push_str(&format!(
                "{}{}Enter opens · Esc cancels",
                at(top + height - 2, left + 2),
                WHITE
            ));
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
        ScreenMode::FilePicker {
            files, selected, ..
        } => match key {
            Key::Esc => app.mode = ScreenMode::Edit,
            Key::Up => *selected = selected.saturating_sub(1),
            Key::Down => {
                if *selected + 1 < files.len() {
                    *selected += 1;
                }
            }
            Key::Enter => {
                let path = files.get(*selected).cloned();
                if let Some(path) = path {
                    match Editor::open(path.clone()) {
                        Ok(editor) => {
                            app.open_buffer(editor);
                            app.mode = ScreenMode::Edit;
                            app.remember_file(path.clone());
                            app.status = format!("Opened {}", path.display());
                        }
                        Err(error) => app.show_error("Open failed", &error.to_string()),
                    }
                }
            }
            _ => {}
        },
        ScreenMode::ConfirmDiscard { action } => {
            let pending = *action;
            match key {
                Key::Char('d' | 'D') => {
                    app.editor.dirty = false;
                    if pending == Action::Quit {
                        for editor in &mut app.buffers {
                            editor.dirty = false;
                        }
                    }
                    app.mode = ScreenMode::Edit;
                    app.action(pending);
                }
                Key::Char('s' | 'S') => {
                    app.mode = ScreenMode::Edit;
                    app.pending_after_save = Some(pending);
                    app.save();
                    app.continue_after_save();
                }
                Key::Esc => app.mode = ScreenMode::Edit,
                _ => {}
            }
        }
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
            Key::Ctrl('n') => app.action(Action::New),
            Key::Ctrl('o') => app.action(Action::Open),
            Key::Ctrl('s') | Key::F(2) => app.save(),
            Key::Ctrl('z') => app.action(Action::Undo),
            Key::Ctrl('y') => app.action(Action::Redo),
            Key::Ctrl('x') => app.action(Action::Cut),
            Key::Ctrl('c') => app.action(Action::Copy),
            Key::Ctrl('v') => app.action(Action::Paste),
            Key::Ctrl('f') => app.action(Action::Find),
            Key::Ctrl('g') => app.action(Action::GoToLine),
            Key::Ctrl('b') => app.action(Action::NextBuffer),
            Key::F(3) => app.action(Action::FindNext),
            Key::F(9) => app.compile(false),
            Key::F(4) => app.next_diagnostic(),
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
            Key::Left => app.editor.move_left(false),
            Key::Right => app.editor.move_right(false),
            Key::Up => app.editor.move_up(false),
            Key::Down => app.editor.move_down(false),
            Key::ShiftLeft => app.editor.move_left(true),
            Key::ShiftRight => app.editor.move_right(true),
            Key::ShiftUp => app.editor.move_up(true),
            Key::ShiftDown => app.editor.move_down(true),
            Key::Home => {
                app.editor.select_movement(false);
                app.editor.col = 0;
            }
            Key::End => {
                app.editor.select_movement(false);
                app.editor.col = app.editor.line_len();
            }
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
        assert!(matches!(
            poll_key(&mut Cursor::new(b"\x1b[1;2D")).unwrap(),
            Some(Key::ShiftLeft)
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
        assert_eq!(app.status, "Run successful", "{:?}", app.output);
        assert!(app.output.iter().any(|line| line == "42"));
        assert!(app
            .output
            .iter()
            .any(|line| line == "Program exited successfully."));
    }

    #[test]
    fn undo_redo_selection_and_multiline_paste_are_unicode_safe() {
        let mut editor = Editor::empty();
        editor.paste("αβ\nsecond");
        assert_eq!(editor.source(), "αβ\nsecond");
        assert!(editor.undo());
        assert_eq!(editor.source(), "");
        assert!(editor.redo());
        editor.row = 0;
        editor.col = 0;
        editor.selection_anchor = Some((1, 3));
        assert_eq!(editor.selected_text().as_deref(), Some("αβ\nsec"));
        editor.record_edit();
        assert!(editor.delete_selection());
        assert_eq!(editor.source(), "ond");
    }

    #[test]
    fn find_wraps_and_go_to_line_prompt_moves_cursor() {
        let mut editor = Editor::empty();
        editor.lines = vec!["alpha".into(), "beta alpha".into()];
        editor.row = 1;
        editor.col = 6;
        assert!(editor.find_from("alpha", true));
        assert_eq!((editor.row, editor.col), (0, 0));

        let mut app = App::new(editor);
        app.mode = ScreenMode::Prompt {
            title: " Go to line ",
            value: "2".into(),
            action: Action::GoToLine,
        };
        app.finish_prompt();
        assert_eq!(app.editor.row, 1);
    }

    #[test]
    fn dirty_buffers_require_confirmation_and_can_be_rotated() {
        let mut editor = Editor::empty();
        editor.insert('x');
        let mut app = App::new(editor);
        app.action(Action::Quit);
        assert!(matches!(app.mode, ScreenMode::ConfirmDiscard { .. }));
        assert!(!app.quit);

        app.editor.dirty = false;
        app.mode = ScreenMode::Edit;
        app.action(Action::New);
        assert_eq!(app.buffers.len(), 1);
        app.action(Action::NextBuffer);
        assert_eq!(app.editor.source(), "x");
    }

    #[test]
    fn compile_records_navigable_diagnostics() {
        let mut editor = Editor::empty();
        editor.lines = vec!["fn main() { let x: Int32 = \"wrong\"; }".into()];
        let mut app = App::new(editor);
        app.compile(false);
        assert!(!app.diagnostic_targets.is_empty());
        app.next_diagnostic();
        assert_eq!(app.editor.row, 0);
        assert!(app.status.starts_with("Diagnostic:"));
    }

    #[test]
    fn responsive_layout_has_safe_minimums_and_scales_output() {
        assert_eq!(calculate_layout(13, 80, true), None);
        assert_eq!(calculate_layout(30, 47, true), None);
        let compact = calculate_layout(14, 48, true).unwrap();
        let large = calculate_layout(60, 160, true).unwrap();
        assert!(compact.editor_height >= 4);
        assert!(compact.text_width >= 40);
        assert!(large.output_height > compact.output_height);
        assert!(large.text_height > compact.text_height);
    }
}
