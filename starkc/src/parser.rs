//! Parser for `02-Syntax-Grammar.md`.
//!
//! Hand-written recursive descent; the expression core follows the normative
//! 16-level precedence table literally (one function per level). Rules the
//! implementation honors:
//! - non-associative comparisons and ranges (structural: chaining is a
//!   dedicated syntax error);
//! - the struct-literal restriction in `if`/`while` conditions, `match`
//!   scrutinees, and `for` iterables;
//! - `>` splitting in generic-argument position (`>>`, `>>=`, `>=` split
//!   into `>` + remainder);
//! - block-formed expression statements (semicolon optional) with the
//!   trailing-expression rule (block-formed expression immediately before
//!   `}` is the block's value);
//! - `(e)` grouping vs `(e,)` 1-tuple, mirrored for types;
//! - panic-mode recovery at `;`, `}`, and item keywords, so one bad
//!   statement doesn't hide the rest of the file.
//!
//! Modes: `Program` (`Item*`, the source-language entry point) and `Snippet`
//! (`(Item | Statement)* Expression?`, the fixture-harness block-body form).

use crate::ast::*;
use crate::diag::Diagnostic;
use crate::lexer::{tokenize, Kw, Token, TokenKind};
use crate::options::LanguageOptions;
use crate::source::{SourceFile, Span};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ParseMode {
    Program,
    Snippet,
}

/// Lex and parse `file` in Core-only mode. The returned diagnostics include
/// lexer diagnostics. This is the Core v1 entry point; extension syntax is
/// rejected. Use [`parse_with_options`] to enable extensions.
pub fn parse(file: &SourceFile, mode: ParseMode) -> (Ast, Vec<Diagnostic>) {
    parse_with_options(file, mode, LanguageOptions::CORE)
}

/// Lex and parse `file` under `options`, which may enable extensions
/// (Gate 4+). The returned diagnostics include lexer diagnostics.
pub fn parse_with_options(
    file: &SourceFile,
    mode: ParseMode,
    options: LanguageOptions,
) -> (Ast, Vec<Diagnostic>) {
    match mode {
        ParseMode::Program => parse_project(file, options),
        ParseMode::Snippet => {
            let mut ast = Ast::default();
            let (root, diags) = parse_with_options_into(file, mode, options, &mut ast);
            ast.root = root;
            (ast, diags)
        }
    }
}

pub fn parse_project(root_file: &SourceFile, options: LanguageOptions) -> (Ast, Vec<Diagnostic>) {
    parse_project_inner(root_file, options, false)
}

/// DEV-036: a test-harness-only entry point, identical to [`parse_project`] except that a
/// genuinely missing `mod foo;` backing file is silently tolerated instead of producing
/// "file not found for module" (E0208). Exists solely so the one spec fixture that legitimately
/// omits a backing file (`07-Modules-and-Packages__01.stark`'s `mod math;`) can be parsed
/// without the diagnostic, via an explicit opt-in the caller names by exact fixture, rather than
/// the previous runtime string-match against the compiled file's own name/path (which could
/// silently swallow a genuinely missing module file for any real user project whose path
/// happened to contain `"spec-fixtures"`/`"STARKLANG"`, or whose entry file was named exactly
/// `test.stark` -- see `starkc/docs/conformance/KNOWN-DEVIATIONS.md` DEV-036). No production
/// caller (`starkc`/`stark` binaries, `stark::analysis`, the LSP) should ever call this; only
/// `starkc/tests/conformance.rs`'s harness does, and only for that one named fixture.
pub fn parse_project_allowing_missing_modules(
    root_file: &SourceFile,
    options: LanguageOptions,
) -> (Ast, Vec<Diagnostic>) {
    parse_project_inner(root_file, options, true)
}

fn parse_project_inner(
    root_file: &SourceFile,
    options: LanguageOptions,
    allow_missing_modules: bool,
) -> (Ast, Vec<Diagnostic>) {
    let mut ast = Ast::default();
    let mut diags = Vec::new();

    // Parse root
    let (root, mut root_diags) =
        parse_with_options_into(root_file, ParseMode::Program, options, &mut ast);
    diags.append(&mut root_diags);
    ast.root = root;

    // Recursively resolve and parse submodules
    let mut loaded_modules = std::collections::HashSet::new();
    loaded_modules.insert(root_file.name.clone());

    let root_items = match &ast.root {
        Root::Program(items) => items.clone(),
        _ => Vec::new(),
    };

    if let Err(mut loader_diags) = load_submodules_recursive(
        root_file,
        &root_items,
        options,
        &mut ast,
        &mut loaded_modules,
        allow_missing_modules,
    ) {
        diags.append(&mut loader_diags);
    }

    (ast, diags)
}

pub fn parse_package_graph(
    graph: &crate::package::PackageGraph,
    options: LanguageOptions,
) -> (Ast, Vec<Diagnostic>) {
    let mut ast = Ast::default();
    let mut diags = Vec::new();
    let mut parsed_packages = std::collections::HashMap::new();

    let root_items_res = parse_package_rec(
        &graph.root_package_name,
        graph,
        options,
        &mut ast,
        &mut diags,
        &mut parsed_packages,
    );

    match root_items_res {
        Ok(items) => {
            ast.root = Root::Program(items);
        }
        Err(err_msg) => {
            diags.push(Diagnostic::error(err_msg, Span { lo: 0, hi: 0 }).with_code("E0208"));
            ast.root = Root::Program(Vec::new());
        }
    }

    (ast, diags)
}

fn parse_package_rec(
    pkg_name: &str,
    graph: &crate::package::PackageGraph,
    options: LanguageOptions,
    ast: &mut Ast,
    diags: &mut Vec<Diagnostic>,
    parsed_packages: &mut std::collections::HashMap<String, Vec<ItemId>>,
) -> Result<Vec<ItemId>, String> {
    if let Some(items) = parsed_packages.get(pkg_name) {
        return Ok(items.clone());
    }

    let pkg = graph
        .packages
        .get(pkg_name)
        .ok_or_else(|| format!("Package '{}' not found in graph", pkg_name))?;

    let entry_src = std::fs::read_to_string(&pkg.entry).map_err(|e| {
        format!(
            "failed to read entry file '{}' for package '{}': {}",
            pkg.entry.display(),
            pkg_name,
            e
        )
    })?;
    let entry_file = std::sync::Arc::new(SourceFile::new(
        pkg.entry.to_string_lossy().into_owned(),
        entry_src,
    ));

    let (root, mut entry_diags) =
        parse_with_options_into(&entry_file, ParseMode::Program, options, ast);
    diags.append(&mut entry_diags);

    let mut root_items = match root {
        Root::Program(items) => items,
        _ => Vec::new(),
    };

    let mut loaded_modules = std::collections::HashSet::new();
    loaded_modules.insert(entry_file.name.clone());
    if let Err(mut sub_diags) = load_submodules_recursive(
        &entry_file,
        &root_items,
        options,
        ast,
        &mut loaded_modules,
        false,
    ) {
        diags.append(&mut sub_diags);
    }

    for dep_name in pkg.dependencies.keys() {
        let dep_items = parse_package_rec(dep_name, graph, options, ast, diags, parsed_packages)?;

        let synthetic_lo = 0x8000_0000 + ast.synthetic_spans.len() as u32;
        let synthetic_hi = synthetic_lo + dep_name.len() as u32;
        let name_span = Span {
            lo: synthetic_lo,
            hi: synthetic_hi,
        };
        ast.synthetic_spans.insert(name_span, dep_name.clone());

        let mod_item_id = ast.alloc_item(
            ItemKind::Mod {
                name: name_span,
                items: Some(dep_items),
            },
            Some(Vis::Pub),
            name_span,
        );

        ast.item_files.insert(mod_item_id, entry_file.clone());
        root_items.push(mod_item_id);
    }

    parsed_packages.insert(pkg_name.to_string(), root_items.clone());
    Ok(root_items)
}

fn load_submodules_recursive(
    current_file: &SourceFile,
    items: &[ItemId],
    options: LanguageOptions,
    ast: &mut Ast,
    loaded_modules: &mut std::collections::HashSet<String>,
    allow_missing_modules: bool,
) -> Result<(), Vec<Diagnostic>> {
    let mut diags = Vec::new();
    let parent_dir = std::path::Path::new(&current_file.name)
        .parent()
        .unwrap_or(std::path::Path::new(""))
        .to_path_buf();

    // Collect all mod items that need to be loaded
    let mut mods_to_load = Vec::new();
    for &item_id in items {
        let item = ast.item(item_id);
        if let ItemKind::Mod { name, items: None } = &item.kind {
            mods_to_load.push((item_id, *name));
        }
    }

    for (mod_item_id, name_span) in mods_to_load {
        let mod_name = &current_file.src[name_span.lo as usize..name_span.hi as usize];
        let candidates = [
            parent_dir.join(format!("{}.stark", mod_name)),
            parent_dir.join(format!("{}.st", mod_name)),
            parent_dir.join(mod_name).join("mod.stark"),
            parent_dir.join(mod_name).join("mod.st"),
        ];
        let existing: Vec<_> = candidates.iter().filter(|path| path.exists()).collect();

        if existing.len() > 1 {
            let paths = existing
                .iter()
                .map(|path| format!("'{}'", path.display()))
                .collect::<Vec<_>>()
                .join(", ");
            diags.push(
                Diagnostic::error(
                    format!(
                        "conflicting module files for '{}': {} exist",
                        mod_name, paths
                    ),
                    name_span,
                )
                .with_code("E0208")
                .with_file(std::sync::Arc::new(SourceFile::new(
                    current_file.name.clone(),
                    current_file.src.clone(),
                ))),
            );
            continue;
        }

        let (path, src) = if let Some(path) = existing.first() {
            if let Ok(src) = std::fs::read_to_string(path) {
                ((*path).clone(), src)
            } else {
                diags.push(
                    Diagnostic::error(format!("cannot read file '{}'", path.display()), name_span)
                        .with_code("E0208")
                        .with_file(std::sync::Arc::new(SourceFile::new(
                            current_file.name.clone(),
                            current_file.src.clone(),
                        ))),
                );
                continue;
            }
        } else {
            // WP-C1.1 (2026-07-17) removed a far more dangerous unconditional bypass keyed off
            // `std::env::args()` (DEV-014). DEV-036 (WP-C2.12) removes what WP-C1.1 kept in its
            // place: a runtime string-match against the compiled file's own name/path
            // (`== "test.stark"`, `.contains("spec-fixtures")`, `.contains("STARKLANG")`), which
            // could still silently swallow a genuinely missing module file for any real user
            // project whose path happened to collide with those substrings. The one legitimate
            // case that needs this suppressed (`STARKLANG/tests/spec-fixtures/
            // 07-Modules-and-Packages__01.stark`'s `mod math;`, which has no backing file on
            // disk by design) now goes through the explicit `allow_missing_modules` parameter,
            // set only by `parse_project_allowing_missing_modules` and named as an exact,
            // harness-side opt-in by `starkc/tests/conformance.rs` -- not inferred from the
            // path at all.
            if !allow_missing_modules {
                let expected = candidates
                    .iter()
                    .map(|path| format!("'{}'", path.display()))
                    .collect::<Vec<_>>()
                    .join(", ");
                diags.push(
                    Diagnostic::error(
                        format!(
                            "file not found for module '{}': expected one of {}",
                            mod_name, expected
                        ),
                        name_span,
                    )
                    .with_code("E0208")
                    .with_file(std::sync::Arc::new(SourceFile::new(
                        current_file.name.clone(),
                        current_file.src.clone(),
                    ))),
                );
            }
            (candidates[0].clone(), String::new())
        };

        let path_str = path.to_string_lossy().into_owned();
        if !loaded_modules.insert(path_str.clone()) {
            continue;
        }

        let child_file = SourceFile::new(path_str, src);
        let (child_root, mut child_diags) =
            parse_with_options_into(&child_file, ParseMode::Program, options, ast);
        diags.append(&mut child_diags);

        let child_items = match child_root {
            Root::Program(items) => items,
            _ => Vec::new(),
        };

        if let ItemKind::Mod { items, .. } = &mut ast.items[mod_item_id.0 as usize].kind {
            *items = Some(child_items.clone());
        }

        if let Err(mut sub_diags) = load_submodules_recursive(
            &child_file,
            &child_items,
            options,
            ast,
            loaded_modules,
            allow_missing_modules,
        ) {
            diags.append(&mut sub_diags);
        }
    }

    if diags.is_empty() {
        Ok(())
    } else {
        Err(diags)
    }
}

pub fn parse_with_options_into(
    file: &SourceFile,
    mode: ParseMode,
    options: LanguageOptions,
    ast: &mut Ast,
) -> (Root, Vec<Diagnostic>) {
    let (tokens, lex_diags) = tokenize(file);
    let mut p = Parser {
        file,
        tokens,
        pos: 0,
        diags: lex_diags,
        ast,
        options,
        in_impl_or_trait: false,
        depth: 0,
        depth_reported: false,
    };
    let root = match mode {
        ParseMode::Program => {
            let items = p.program();
            Root::Program(items)
        }
        ParseMode::Snippet => {
            let (stmts, tail) = p.snippet();
            Root::Snippet { stmts, tail }
        }
    };
    let file_arc = std::sync::Arc::new(SourceFile::new(file.name.clone(), file.src.clone()));
    for diag in &mut p.diags {
        if diag.file.is_none() {
            diag.file = Some(file_arc.clone());
        }
    }
    (root, p.diags)
}

/// Struct-literal restriction (02 "Struct Literal Restriction"): true in the
/// condition of `if`/`while`, the scrutinee of `match`, and the iterable of
/// `for`; cleared inside any parenthesized/bracketed subexpression.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Restrictions {
    no_struct_literal: bool,
}

const DEFAULT: Restrictions = Restrictions {
    no_struct_literal: false,
};
const NO_STRUCT: Restrictions = Restrictions {
    no_struct_literal: true,
};

/// Classification of a `[...]` group in generic-argument position
/// (`tensor` extension D2/D5 vs Core array/slice types).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ShapeGroupKind {
    /// `[T; N]` — a Core array type (has a top-level `;`).
    ArrayType,
    /// `[]` or `[a, b, ...]` — unambiguously a shape argument.
    ShapeLike,
    /// A single element with no `;`/`,`: a Core slice type `[T]` or a rank-1
    /// shape `[B]`, resolved by whether the extension is enabled.
    SingleElem,
}

struct Parser<'a> {
    file: &'a SourceFile,
    tokens: Vec<Token>,
    pos: usize,
    diags: Vec<Diagnostic>,
    ast: &'a mut Ast,
    /// Enabled language extensions (Core-only by default). Gates extension
    /// surface syntax such as tensor shape arguments and `model` items.
    options: LanguageOptions,
    /// `Self` paths and receivers are only valid inside `trait`/`impl`.
    in_impl_or_trait: bool,
    /// Recursion depth across expr/type/pattern/block/item nesting. Bounded
    /// so pathological input (`((((...`) degrades into a diagnostic instead
    /// of a stack overflow.
    depth: u32,
    depth_reported: bool,
}

const MAX_DEPTH: u32 = 200;

impl Parser<'_> {
    // ------------------------------------------------------- token cursor --

    fn peek(&self) -> Token {
        self.tokens[self.pos]
    }

    fn peek2(&self) -> TokenKind {
        self.tokens
            .get(self.pos + 1)
            .map_or(TokenKind::Eof, |t| t.kind)
    }

    fn bump(&mut self) -> Token {
        let t = self.tokens[self.pos];
        if t.kind != TokenKind::Eof {
            self.pos += 1;
        }
        t
    }

    fn at(&self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }

    fn at_kw(&self, kw: Kw) -> bool {
        self.peek().kind == TokenKind::Keyword(kw)
    }

    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn eat_kw(&mut self, kw: Kw) -> bool {
        self.eat(TokenKind::Keyword(kw))
    }

    fn text(&self, span: Span) -> &str {
        &self.file.src[span.lo as usize..span.hi as usize]
    }

    fn describe(&self, token: Token) -> String {
        match token.kind {
            TokenKind::Eof => "end of file".to_string(),
            TokenKind::Reserved => {
                format!("reserved word `{}`", self.text(token.span))
            }
            _ => format!("`{}`", self.text(token.span)),
        }
    }

    fn error(&mut self, message: impl Into<String>, span: Span) {
        self.diags.push(Diagnostic::error(message, span));
    }

    /// "expected X, found Y" at the current token. Reserved words get the
    /// dedicated message; lexer `Error` tokens already have a diagnostic and
    /// produce none here.
    fn expected(&mut self, what: &str) {
        let token = self.peek();
        if token.kind == TokenKind::Error {
            return;
        }
        let found = self.describe(token);
        if token.kind == TokenKind::Reserved {
            let word = self.text(token.span).to_string();
            self.diags.push(
                Diagnostic::error(format!("`{word}` is reserved for future use"), token.span)
                    .with_label(format!("expected {what}")),
            );
        } else {
            self.error(format!("expected {what}, found {found}"), token.span);
        }
    }

    fn expect(&mut self, kind: TokenKind, what: &str) -> bool {
        if self.eat(kind) {
            true
        } else {
            self.expected(what);
            false
        }
    }

    fn expect_ident(&mut self, what: &str) -> Option<Span> {
        if self.at(TokenKind::Ident) {
            Some(self.bump().span)
        } else {
            self.expected(what);
            None
        }
    }

    /// Consume a `>` in generic-argument position, splitting `>>`, `>>=`,
    /// and `>=` (02 "Parsing Notes").
    fn eat_gt(&mut self) -> bool {
        let t = self.tokens[self.pos];
        match t.kind {
            TokenKind::Gt => {
                self.bump();
                true
            }
            TokenKind::Shr => {
                self.tokens[self.pos] = Token {
                    kind: TokenKind::Gt,
                    span: Span {
                        lo: t.span.lo + 1,
                        hi: t.span.hi,
                    },
                };
                true
            }
            TokenKind::ShrEq => {
                self.tokens[self.pos] = Token {
                    kind: TokenKind::GtEq,
                    span: Span {
                        lo: t.span.lo + 1,
                        hi: t.span.hi,
                    },
                };
                true
            }
            TokenKind::GtEq => {
                self.tokens[self.pos] = Token {
                    kind: TokenKind::Eq,
                    span: Span {
                        lo: t.span.lo + 1,
                        hi: t.span.hi,
                    },
                };
                true
            }
            _ => false,
        }
    }

    fn span_from(&self, lo: u32) -> Span {
        let hi = if self.pos > 0 {
            self.tokens[self.pos - 1].span.hi
        } else {
            lo
        };
        Span { lo, hi: hi.max(lo) }
    }

    // ----------------------------------------------------------- recovery --

    fn item_start(kind: TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Keyword(
                Kw::Fn
                    | Kw::Struct
                    | Kw::Enum
                    | Kw::Trait
                    | Kw::Impl
                    | Kw::Const
                    | Kw::Type
                    | Kw::Use
                    | Kw::Mod
                    | Kw::Pub
                    | Kw::Priv
            )
        )
    }

    /// The contextual `model` item keyword (D4) begins an item when followed by
    /// a name, in both Core-only and tensor mode (Core-only produces a focused
    /// "requires extension `tensor`" diagnostic). The `peek2` guard keeps
    /// `model` usable as an ordinary identifier in expression position
    /// (`model.predict(x)`), since `model <ident>` is never a Core expression.
    fn at_model_item(&self) -> bool {
        self.peek().kind == TokenKind::Ident
            && self.text(self.peek().span) == "model"
            && self.peek2() == TokenKind::Ident
    }

    /// Whether the current token begins an item (Core keywords or the `model`
    /// contextual keyword).
    fn at_item_start(&self) -> bool {
        Self::item_start(self.peek().kind) || self.at_model_item()
    }

    /// Panic-mode recovery inside blocks: skip to just after a `;`, or to a
    /// `}` / item keyword / EOF (not consumed). Balances nested delimiters.
    fn recover_stmt(&mut self) {
        let mut depth = 0usize;
        loop {
            // Stop before a contextual `model` item so recovery does not skip a
            // following model declaration (Codex P2).
            if depth == 0 && self.at_model_item() {
                return;
            }
            match self.peek().kind {
                TokenKind::Eof => return,
                TokenKind::Semi if depth == 0 => {
                    self.bump();
                    return;
                }
                TokenKind::RBrace if depth == 0 => return,
                k if depth == 0 && Self::item_start(k) => return,
                TokenKind::LBrace | TokenKind::LParen | TokenKind::LBracket => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::RBrace | TokenKind::RParen | TokenKind::RBracket => {
                    depth = depth.saturating_sub(1);
                    self.bump();
                }
                _ => {
                    self.bump();
                }
            }
        }
    }

    /// Recovery between items: skip (balancing braces) to the next item
    /// keyword or EOF.
    fn recover_item(&mut self) {
        let mut depth = 0usize;
        loop {
            // Stop before a contextual `model` item (Codex P2) so malformed
            // input cannot cause every following model to be skipped to EOF.
            if depth == 0 && self.at_model_item() {
                return;
            }
            match self.peek().kind {
                TokenKind::Eof => return,
                k if depth == 0 && Self::item_start(k) => return,
                TokenKind::LBrace => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::RBrace => {
                    depth = depth.saturating_sub(1);
                    self.bump();
                }
                _ => {
                    self.bump();
                }
            }
        }
    }

    // -------------------------------------------------------------- paths --

    fn at_path_start(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::Ident
                | TokenKind::Keyword(Kw::SelfLower | Kw::SelfUpper | Kw::Super | Kw::Crate)
        ) || self.at_primitive().is_some()
    }

    fn at_primitive(&self) -> Option<Primitive> {
        let TokenKind::Keyword(kw) = self.peek().kind else {
            return None;
        };
        Some(match kw {
            Kw::Int8 => Primitive::Int8,
            Kw::Int16 => Primitive::Int16,
            Kw::Int32 => Primitive::Int32,
            Kw::Int64 => Primitive::Int64,
            Kw::UInt8 => Primitive::UInt8,
            Kw::UInt16 => Primitive::UInt16,
            Kw::UInt32 => Primitive::UInt32,
            Kw::UInt64 => Primitive::UInt64,
            Kw::Float32 => Primitive::Float32,
            Kw::Float64 => Primitive::Float64,
            Kw::Bool => Primitive::Bool,
            Kw::CharTy => Primitive::Char,
            Kw::StringTy => Primitive::String,
            Kw::Str => Primitive::Str,
            Kw::Unit => Primitive::Unit,
            _ => return None,
        })
    }

    /// One path segment. Primitive-type keywords are valid as the *first*
    /// segment only (`String::from`, per 02's amended `PathSegment`);
    /// `self`/`Self`/`crate` likewise; `super` may repeat at the front.
    fn path_segment(&mut self, first: bool, prev_super: bool) -> Option<PathSegment> {
        let token = self.peek();
        let kind = match token.kind {
            TokenKind::Ident => SegmentKind::Ident,
            TokenKind::Keyword(Kw::SelfLower) => SegmentKind::SelfValue,
            TokenKind::Keyword(Kw::SelfUpper) => SegmentKind::SelfType,
            TokenKind::Keyword(Kw::Super) => SegmentKind::Super,
            TokenKind::Keyword(Kw::Crate) => SegmentKind::Crate,
            TokenKind::Keyword(_) if self.at_primitive().is_some() => {
                if !first {
                    let word = self.text(token.span).to_string();
                    self.error(
                        format!("`{word}` is only valid as the first segment of a path"),
                        token.span,
                    );
                }
                SegmentKind::Ident
            }
            _ => {
                self.expected("a path segment");
                return None;
            }
        };
        if !first {
            match kind {
                SegmentKind::SelfValue | SegmentKind::SelfType | SegmentKind::Crate => {
                    let word = self.text(token.span).to_string();
                    self.error(
                        format!("`{word}` is only valid as the first segment of a path"),
                        token.span,
                    );
                }
                SegmentKind::Super if !prev_super => {
                    self.error("`super` may only appear at the front of a path", token.span);
                }
                _ => {}
            }
        }
        if kind == SegmentKind::SelfType && !self.in_impl_or_trait {
            self.error(
                "`Self` is only valid inside a `trait` or `impl` block",
                token.span,
            );
        }
        self.bump();
        Some(PathSegment {
            kind,
            span: token.span,
        })
    }

    fn at_segment_kind(kind: TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Ident
                | TokenKind::Keyword(Kw::SelfLower | Kw::SelfUpper | Kw::Super | Kw::Crate)
        )
    }

    /// `Path ::= PathSegment ('::' PathSegment)*`. Stops before `::` when the
    /// following token is not a segment (so `::<` turbofish and use-tree
    /// `::*` / `::{` remain for the caller).
    fn path(&mut self) -> Option<Path> {
        let lo = self.peek().span.lo;
        let first = self.path_segment(true, false)?;
        let mut prev_super = first.kind == SegmentKind::Super;
        let mut segments = vec![first];
        while self.at(TokenKind::ColonColon) && Self::at_segment_kind(self.peek2()) {
            self.bump(); // ::
            let Some(seg) = self.path_segment(false, prev_super) else {
                break;
            };
            prev_super = prev_super && seg.kind == SegmentKind::Super;
            segments.push(seg);
        }
        Some(Path {
            segments,
            span: self.span_from(lo),
        })
    }

    // ------------------------------------------------------------- types --

    fn generic_args(&mut self, single_ident_is_shape: bool) -> Option<GenericArgs> {
        let lo = self.peek().span.lo;
        if !self.expect(TokenKind::Lt, "`<`") {
            return None;
        }
        let mut args = Vec::new();
        loop {
            if self.eat_gt() {
                break;
            }
            if self.at(TokenKind::Eof) {
                self.expected("`>`");
                break;
            }
            // Shape / index-list argument `[DimExpr, ...]` (extension D2/D5).
            // In Core-only mode this only intercepts the `[]` and `[a, b]`
            // forms — which were parse errors before — to emit a focused
            // diagnostic; Core slice/array generic args (`[T]`, `[T; N]`) fall
            // through to `ty()` unchanged in every mode.
            if self.at(TokenKind::LBracket) {
                match self.shape_group_kind() {
                    ShapeGroupKind::ArrayType => {
                        let ty = self.ty();
                        args.push(GenericArg::Type(ty));
                    }
                    ShapeGroupKind::ShapeLike => {
                        let shape = self.shape_arg();
                        args.push(GenericArg::Shape(shape));
                    }
                    ShapeGroupKind::SingleElem => {
                        // A single-element `[X]` is a Core slice type unless the
                        // extension is on AND `X` is dim-capable. Type-only
                        // elements (`[Int32]`, `[&T]`, `[Foo<U>]`) stay Core
                        // slices even under the extension, so valid Core
                        // generic arguments are never stolen (Codex P1).
                        if self.tensor_enabled()
                            && !self.single_bracket_elem_is_type(single_ident_is_shape)
                        {
                            let shape = self.shape_arg();
                            args.push(GenericArg::Shape(shape));
                        } else {
                            let ty = self.ty();
                            args.push(GenericArg::Type(ty));
                        }
                    }
                }
            } else if self.at(TokenKind::Ident) && self.peek2() == TokenKind::Eq {
                // `Item = T` associated-type binding (also `device = D`, §8).
                let name = self.bump().span;
                self.bump(); // =
                let ty = self.ty();
                args.push(GenericArg::Binding { name, ty });
            } else if self.tensor_enabled() && matches!(self.peek().kind, TokenKind::Int { .. }) {
                args.push(GenericArg::Const(self.bump().span));
            } else {
                let ty = self.ty();
                args.push(GenericArg::Type(ty));
            }
            if !self.eat(TokenKind::Comma) {
                if !self.eat_gt() {
                    self.expected("`,` or `>`");
                }
                break;
            }
        }
        Some(GenericArgs {
            args,
            span: self.span_from(lo),
        })
    }

    // ------------------------------------ tensor extension (D2/D3/D5) --

    fn tensor_enabled(&self) -> bool {
        self.options.tensor()
    }

    /// Recognize the extension element-type identifiers `Float16`/`BFloat16`
    /// (D3) when the `tensor` extension is enabled. They lex as ordinary
    /// identifiers; in Core-only mode they are left to normal name resolution
    /// (so a Core program may still use them as ordinary type names).
    fn at_extension_primitive(&self) -> Option<Primitive> {
        if !self.tensor_enabled() || self.peek().kind != TokenKind::Ident {
            return None;
        }
        match self.text(self.peek().span) {
            "Float16" => Some(Primitive::Float16),
            "BFloat16" => Some(Primitive::BFloat16),
            _ => None,
        }
    }

    /// Classify a `[...]` group in generic-argument position without consuming
    /// it, by a bounded scan to the matching `]`. Priority: a top-level `;`
    /// means a Core array type `[T; N]`; an empty group or a top-level `,`
    /// means a shape argument; a single element is mode-dependent
    /// (slice type in Core, rank-1 shape under the extension).
    fn shape_group_kind(&self) -> ShapeGroupKind {
        debug_assert!(self.at(TokenKind::LBracket));
        let mut depth = 0i32;
        let mut i = self.pos;
        let mut saw_comma = false;
        let mut saw_content = false;
        while let Some(tok) = self.tokens.get(i) {
            match tok.kind {
                TokenKind::LBracket | TokenKind::LParen | TokenKind::LBrace => {
                    if depth == 1 {
                        saw_content = true;
                    }
                    depth += 1;
                }
                TokenKind::RParen | TokenKind::RBrace => depth -= 1,
                TokenKind::RBracket => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                TokenKind::Semi if depth == 1 => return ShapeGroupKind::ArrayType,
                TokenKind::Comma if depth == 1 => saw_comma = true,
                TokenKind::Eof => break,
                _ if depth == 1 => saw_content = true,
                _ => {}
            }
            i += 1;
        }
        // `[]` (no content) or a top-level comma is unambiguously a shape.
        if !saw_content || saw_comma {
            ShapeGroupKind::ShapeLike
        } else {
            ShapeGroupKind::SingleElem
        }
    }

    /// Decide whether the single element of a `[X]` generic argument (cursor at
    /// `[`) can only be a Core type, not a dimension expression. Dimension
    /// atoms are integer literals, bare identifiers, and `(`; anything else —
    /// a primitive keyword, `Self`, `fn`, `&`, a nested `[`, or an identifier
    /// that heads a path (`::`) or generic (`<`) — is a type. A lone
    /// identifier or parenthesized atom is ambiguous, so it becomes a rank-1
    /// shape only in a known shape position. `[Int32]`, `[T]`, `[Foo]`,
    /// `[&T]`, and `[Foo<U>]` otherwise remain Core slice types.
    fn single_bracket_elem_is_type(&self, single_ident_is_shape: bool) -> bool {
        let t1 = self.tokens.get(self.pos + 1).map(|t| t.kind);
        let t2 = self
            .tokens
            .get(self.pos + 2)
            .map_or(TokenKind::Eof, |t| t.kind);
        match t1 {
            Some(TokenKind::Ident) => {
                if t2 == TokenKind::RBracket {
                    !single_ident_is_shape
                } else {
                    matches!(t2, TokenKind::ColonColon | TokenKind::Lt)
                }
            }
            Some(TokenKind::Int { .. }) => false,
            Some(TokenKind::LParen) => !single_ident_is_shape,
            // Primitive keywords, `Self`, `fn`, `&`, nested `[`, tuples, etc.
            Some(_) => true,
            None => false,
        }
    }

    /// Parse a `[DimExpr, ...]` / `[]` shape argument (D2) or const index list
    /// (D5). Emits a focused diagnostic naming the `tensor` extension when it
    /// is disabled, then still consumes the group so recovery is clean.
    fn shape_arg(&mut self) -> ShapeArg {
        let lo = self.peek().span.lo;
        if !self.tensor_enabled() {
            self.error(
                "shape arguments require extension `tensor`",
                self.peek().span,
            );
        }
        self.expect(TokenKind::LBracket, "`[`");
        let mut dims = Vec::new();
        loop {
            if self.at(TokenKind::RBracket) || self.at(TokenKind::Eof) {
                break;
            }
            // Reserved named axes `[batch: B, ...]` (§11) — reject but recover
            // by parsing the dimension after the name.
            if self.at(TokenKind::Ident) && self.peek2() == TokenKind::Colon {
                self.error("named axes are reserved in `tensor` v0.1", self.peek().span);
                self.bump(); // name
                self.bump(); // :
            }
            dims.push(self.dim_expr());
            if matches!(
                self.peek().kind,
                TokenKind::Lt | TokenKind::LtEq | TokenKind::Gt | TokenKind::GtEq
            ) {
                self.error(
                    "dimension inequalities and range constraints are reserved in `tensor` v0.1",
                    self.peek().span,
                );
                self.bump();
                let _ = self.dim_expr();
            }
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBracket, "`]`");
        ShapeArg {
            dims,
            span: self.span_from(lo),
        }
    }

    /// A dimension expression (§3.2): `+`/`-` at the lowest precedence, `*`
    /// above them, atoms are integer literals, dim variables, and
    /// parenthesized sub-expressions. Left-associative.
    fn dim_expr(&mut self) -> DimId {
        if self.depth_exceeded() {
            let span = self.peek().span;
            return self.ast.alloc_dim(DimExprKind::Error, span);
        }
        self.depth += 1;
        let d = self.dim_add();
        self.depth -= 1;
        d
    }

    fn dim_add(&mut self) -> DimId {
        let lo = self.peek().span.lo;
        let mut lhs = self.dim_mul();
        loop {
            let op = if self.at(TokenKind::Plus) {
                DimBinOp::Add
            } else if self.at(TokenKind::Minus) {
                DimBinOp::Sub
            } else {
                break;
            };
            self.bump();
            let rhs = self.dim_mul();
            lhs = self
                .ast
                .alloc_dim(DimExprKind::Binary { op, lhs, rhs }, self.span_from(lo));
        }
        lhs
    }

    fn dim_mul(&mut self) -> DimId {
        let lo = self.peek().span.lo;
        let mut lhs = self.dim_atom();
        while self.at(TokenKind::Star) {
            self.bump();
            let rhs = self.dim_atom();
            lhs = self.ast.alloc_dim(
                DimExprKind::Binary {
                    op: DimBinOp::Mul,
                    lhs,
                    rhs,
                },
                self.span_from(lo),
            );
        }
        lhs
    }

    fn dim_atom(&mut self) -> DimId {
        let token = self.peek();
        match token.kind {
            TokenKind::Int { .. } => {
                self.bump();
                self.ast.alloc_dim(DimExprKind::Lit(token.span), token.span)
            }
            TokenKind::Ident => {
                self.bump();
                self.ast.alloc_dim(DimExprKind::Var(token.span), token.span)
            }
            TokenKind::LParen => {
                self.bump();
                let inner = self.dim_expr();
                self.expect(TokenKind::RParen, "`)`");
                inner
            }
            _ => {
                self.expected("a dimension expression (integer, identifier, or `(`)");
                self.ast.alloc_dim(DimExprKind::Error, token.span)
            }
        }
    }

    fn ty(&mut self) -> TypeId {
        if self.depth_exceeded() {
            let span = self.peek().span;
            return self.ast.alloc_type(TypeKind::Error, span);
        }
        self.depth += 1;
        let t = self.ty_inner();
        self.depth -= 1;
        t
    }

    fn ty_inner(&mut self) -> TypeId {
        let token = self.peek();
        let lo = token.span.lo;
        if let Some(p) = self.at_primitive() {
            // A primitive keyword followed by `::` heads a path
            // (`String::from` is a path even in type position).
            if self.peek2() != TokenKind::ColonColon {
                self.bump();
                return self.ast.alloc_type(TypeKind::Primitive(p), token.span);
            }
        }
        // Extension element types `Float16`/`BFloat16` (D3): lexed as
        // identifiers, recognized as primitives only when `tensor` is enabled.
        if let Some(p) = self.at_extension_primitive() {
            if self.peek2() != TokenKind::ColonColon {
                self.bump();
                return self.ast.alloc_type(TypeKind::Primitive(p), token.span);
            }
        }
        if self.at_path_start() {
            let Some(path) = self.path() else {
                return self.ast.alloc_type(TypeKind::Error, token.span);
            };
            if self.tensor_enabled() && path.segments.len() == 1 {
                let name = self.text(path.segments[0].span);
                let reserved = match name {
                    "QInt8" | "QUInt8" | "QInt16" | "Quantized" => {
                        Some("quantized dtypes are reserved in `tensor` v0.1")
                    }
                    "NCHW" | "NHWC" | "RowMajor" | "ColumnMajor" | "TensorLayout" => {
                        Some("memory layout types are reserved in `tensor` v0.1")
                    }
                    "PeakMemory" | "MemoryProfile" => {
                        Some("peak-memory deployment constraints are reserved in `tensor` v0.1")
                    }
                    "Gradient" | "Grad" | "Tape" | "Autodiff" => {
                        Some("training and autodiff types are reserved in `tensor` v0.1")
                    }
                    _ => None,
                };
                if let Some(message) = reserved {
                    self.error(message, path.span);
                }
            }
            // `[T]` is valid Core slice syntax and is otherwise lexically
            // indistinguishable from a rank-one symbolic shape. Only the
            // extension-owned `Tensor` constructor establishes a shape
            // position at parse time; all other bare identifiers remain Core
            // types until a later semantic extension signature says otherwise.
            let single_ident_is_shape = self.tensor_enabled()
                && path.segments.len() == 1
                && self.text(path.segments[0].span) == "Tensor";
            let args = if self.at(TokenKind::Lt) {
                self.generic_args(single_ident_is_shape)
            } else {
                None
            };
            return self
                .ast
                .alloc_type(TypeKind::Path { path, args }, self.span_from(lo));
        }
        match token.kind {
            TokenKind::LBracket => {
                self.bump();
                let elem = self.ty();
                if self.eat(TokenKind::Semi) {
                    let len = if matches!(self.peek().kind, TokenKind::Int { .. }) {
                        self.bump().span
                    } else {
                        self.expected("an integer array length");
                        self.peek().span
                    };
                    self.expect(TokenKind::RBracket, "`]`");
                    self.ast
                        .alloc_type(TypeKind::Array { elem, len }, self.span_from(lo))
                } else {
                    self.expect(TokenKind::RBracket, "`]`");
                    self.ast
                        .alloc_type(TypeKind::Slice(elem), self.span_from(lo))
                }
            }
            TokenKind::LParen => {
                self.bump();
                if self.eat(TokenKind::RParen) {
                    return self
                        .ast
                        .alloc_type(TypeKind::Tuple(Vec::new()), self.span_from(lo));
                }
                let first = self.ty();
                if self.eat(TokenKind::RParen) {
                    // `(T)` is grouping: the type T (02 type notes).
                    return first;
                }
                let mut elems = vec![first];
                while self.eat(TokenKind::Comma) {
                    if self.at(TokenKind::RParen) {
                        break;
                    }
                    elems.push(self.ty());
                }
                self.expect(TokenKind::RParen, "`)`");
                self.ast
                    .alloc_type(TypeKind::Tuple(elems), self.span_from(lo))
            }
            TokenKind::Amp => {
                self.bump();
                let mutable = self.eat_kw(Kw::Mut);
                let inner = self.ty();
                self.ast
                    .alloc_type(TypeKind::Ref { mutable, inner }, self.span_from(lo))
            }
            TokenKind::Keyword(Kw::Fn) => {
                self.bump();
                self.expect(TokenKind::LParen, "`(`");
                let mut params = Vec::new();
                while !self.at(TokenKind::RParen) && !self.at(TokenKind::Eof) {
                    params.push(self.ty());
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RParen, "`)`");
                let ret = if self.eat(TokenKind::Arrow) {
                    Some(self.ty())
                } else {
                    None
                };
                self.ast
                    .alloc_type(TypeKind::Fn { params, ret }, self.span_from(lo))
            }
            _ => {
                self.expected("a type");
                self.ast.alloc_type(TypeKind::Error, token.span)
            }
        }
    }

    // ------------------------------------------------------- expressions --

    /// True (and reports once) when the nesting budget is exhausted; the
    /// caller returns an error node. Consumes one token so recovery always
    /// makes progress.
    fn depth_exceeded(&mut self) -> bool {
        if self.depth < MAX_DEPTH {
            return false;
        }
        if !self.depth_reported {
            self.depth_reported = true;
            let span = self.peek().span;
            self.error("this code is nested too deeply to parse", span);
        }
        self.bump();
        true
    }

    /// Level 16: `AssignmentExpression` (right-associative).
    fn expr(&mut self, r: Restrictions) -> ExprId {
        if self.depth_exceeded() {
            let span = self.peek().span;
            return self.ast.alloc_expr(ExprKind::Error, span);
        }
        self.depth += 1;
        let e = self.expr_inner(r);
        self.depth -= 1;
        e
    }

    fn expr_inner(&mut self, r: Restrictions) -> ExprId {
        let lo = self.peek().span.lo;
        let lhs = self.range_expr(r);
        let op = match self.peek().kind {
            TokenKind::Eq => AssignOp::Assign,
            TokenKind::PlusEq => AssignOp::AddAssign,
            TokenKind::MinusEq => AssignOp::SubAssign,
            TokenKind::StarEq => AssignOp::MulAssign,
            TokenKind::SlashEq => AssignOp::DivAssign,
            TokenKind::PercentEq => AssignOp::RemAssign,
            TokenKind::StarStarEq => AssignOp::PowAssign,
            TokenKind::AmpEq => AssignOp::BitAndAssign,
            TokenKind::PipeEq => AssignOp::BitOrAssign,
            TokenKind::CaretEq => AssignOp::BitXorAssign,
            TokenKind::ShlEq => AssignOp::ShlAssign,
            TokenKind::ShrEq => AssignOp::ShrAssign,
            _ => return lhs,
        };
        self.bump();
        let rhs = self.expr(r);
        self.ast
            .alloc_expr(ExprKind::Assign { op, lhs, rhs }, self.span_from(lo))
    }

    /// Level 15: ranges, non-associative.
    fn range_expr(&mut self, r: Restrictions) -> ExprId {
        let lo_pos = self.peek().span.lo;
        let lo = self.or_expr(r);
        let inclusive = match self.peek().kind {
            TokenKind::DotDot => false,
            TokenKind::DotDotEq => true,
            _ => return lo,
        };
        self.bump();
        let hi = self.or_expr(r);
        if matches!(self.peek().kind, TokenKind::DotDot | TokenKind::DotDotEq) {
            let span = self.peek().span;
            self.diags.push(
                Diagnostic::error("range operators cannot be chained", span)
                    .with_help("parenthesize one of the ranges"),
            );
            self.bump();
            let _ = self.or_expr(r);
        }
        self.ast.alloc_expr(
            ExprKind::Range { lo, hi, inclusive },
            self.span_from(lo_pos),
        )
    }

    fn binary_left(
        &mut self,
        r: Restrictions,
        ops: &[(TokenKind, BinOp)],
        next: fn(&mut Self, Restrictions) -> ExprId,
    ) -> ExprId {
        let lo = self.peek().span.lo;
        let mut lhs = next(self, r);
        loop {
            let Some(&(_, op)) = ops.iter().find(|(k, _)| self.at(*k)) else {
                return lhs;
            };
            self.bump();
            let rhs = next(self, r);
            lhs = self
                .ast
                .alloc_expr(ExprKind::Binary { op, lhs, rhs }, self.span_from(lo));
        }
    }

    /// Level 14: `||`.
    fn or_expr(&mut self, r: Restrictions) -> ExprId {
        self.binary_left(r, &[(TokenKind::OrOr, BinOp::Or)], Self::and_expr)
    }

    /// Level 13: `&&`.
    fn and_expr(&mut self, r: Restrictions) -> ExprId {
        self.binary_left(r, &[(TokenKind::AndAnd, BinOp::And)], Self::equality_expr)
    }

    fn chained_comparison_error(&mut self) {
        let span = self.peek().span;
        self.diags.push(
            Diagnostic::error("comparison operators cannot be chained", span)
                .with_help("parenthesize to compare a Bool result explicitly")
                .with_note(
                    "generic arguments in expressions use `path::<T>` (turbofish), not `path<T>`",
                ),
        );
    }

    /// Level 12: `==` / `!=`, non-associative.
    fn equality_expr(&mut self, r: Restrictions) -> ExprId {
        let lo = self.peek().span.lo;
        let lhs = self.relational_expr(r);
        let op = match self.peek().kind {
            TokenKind::EqEq => BinOp::Eq,
            TokenKind::NotEq => BinOp::Ne,
            _ => return lhs,
        };
        self.bump();
        let rhs = self.relational_expr(r);
        if matches!(self.peek().kind, TokenKind::EqEq | TokenKind::NotEq) {
            self.chained_comparison_error();
            self.bump();
            let _ = self.relational_expr(r);
        }
        self.ast
            .alloc_expr(ExprKind::Binary { op, lhs, rhs }, self.span_from(lo))
    }

    /// Level 11: `<` `<=` `>` `>=`, non-associative.
    fn relational_expr(&mut self, r: Restrictions) -> ExprId {
        let lo = self.peek().span.lo;
        let lhs = self.bitor_expr(r);
        let op = match self.peek().kind {
            TokenKind::Lt => BinOp::Lt,
            TokenKind::LtEq => BinOp::Le,
            TokenKind::Gt => BinOp::Gt,
            TokenKind::GtEq => BinOp::Ge,
            _ => return lhs,
        };
        self.bump();
        let rhs = self.bitor_expr(r);
        if matches!(
            self.peek().kind,
            TokenKind::Lt | TokenKind::LtEq | TokenKind::Gt | TokenKind::GtEq
        ) {
            self.chained_comparison_error();
            self.bump();
            let _ = self.bitor_expr(r);
        }
        self.ast
            .alloc_expr(ExprKind::Binary { op, lhs, rhs }, self.span_from(lo))
    }

    /// Levels 10-8: `|`, `^`, `&`.
    fn bitor_expr(&mut self, r: Restrictions) -> ExprId {
        self.binary_left(r, &[(TokenKind::Pipe, BinOp::BitOr)], Self::bitxor_expr)
    }
    fn bitxor_expr(&mut self, r: Restrictions) -> ExprId {
        self.binary_left(r, &[(TokenKind::Caret, BinOp::BitXor)], Self::bitand_expr)
    }
    fn bitand_expr(&mut self, r: Restrictions) -> ExprId {
        self.binary_left(r, &[(TokenKind::Amp, BinOp::BitAnd)], Self::shift_expr)
    }

    /// Level 7: `<<` `>>`.
    fn shift_expr(&mut self, r: Restrictions) -> ExprId {
        self.binary_left(
            r,
            &[(TokenKind::Shl, BinOp::Shl), (TokenKind::Shr, BinOp::Shr)],
            Self::additive_expr,
        )
    }

    /// Level 6: `+` `-`.
    fn additive_expr(&mut self, r: Restrictions) -> ExprId {
        self.binary_left(
            r,
            &[
                (TokenKind::Plus, BinOp::Add),
                (TokenKind::Minus, BinOp::Sub),
            ],
            Self::multiplicative_expr,
        )
    }

    /// Level 5: `*` `/` `%`.
    fn multiplicative_expr(&mut self, r: Restrictions) -> ExprId {
        self.binary_left(
            r,
            &[
                (TokenKind::Star, BinOp::Mul),
                (TokenKind::Slash, BinOp::Div),
                (TokenKind::Percent, BinOp::Rem),
            ],
            Self::pow_expr,
        )
    }

    /// Level 4: `**` (right-associative).
    fn pow_expr(&mut self, r: Restrictions) -> ExprId {
        let lo = self.peek().span.lo;
        let lhs = self.cast_expr(r);
        if !self.eat(TokenKind::StarStar) {
            return lhs;
        }
        let rhs = self.pow_expr(r);
        self.ast.alloc_expr(
            ExprKind::Binary {
                op: BinOp::Pow,
                lhs,
                rhs,
            },
            self.span_from(lo),
        )
    }

    /// Level 3: `as` (left-associative chain).
    fn cast_expr(&mut self, r: Restrictions) -> ExprId {
        let lo = self.peek().span.lo;
        let mut e = self.unary_expr(r);
        while self.eat_kw(Kw::As) {
            let ty = self.ty();
            e = self
                .ast
                .alloc_expr(ExprKind::Cast { expr: e, ty }, self.span_from(lo));
        }
        e
    }

    /// Level 2: unary prefix.
    fn unary_expr(&mut self, r: Restrictions) -> ExprId {
        // Direct self-recursion (`----x`, `&&&&x`) bypasses `expr`; guard
        // here as well.
        if self.depth_exceeded() {
            let span = self.peek().span;
            return self.ast.alloc_expr(ExprKind::Error, span);
        }
        self.depth += 1;
        let e = self.unary_expr_inner(r);
        self.depth -= 1;
        e
    }

    fn unary_expr_inner(&mut self, r: Restrictions) -> ExprId {
        let token = self.peek();
        let lo = token.span.lo;
        let op = match token.kind {
            TokenKind::Minus => UnOp::Neg,
            TokenKind::Bang => UnOp::Not,
            TokenKind::Tilde => UnOp::BitNot,
            TokenKind::Star => UnOp::Deref,
            TokenKind::Amp => {
                self.bump();
                let mutable = self.eat_kw(Kw::Mut);
                let operand = self.unary_expr(r);
                return self.ast.alloc_expr(
                    ExprKind::Unary {
                        op: UnOp::Ref { mutable },
                        operand,
                    },
                    self.span_from(lo),
                );
            }
            _ => return self.postfix_expr(r),
        };
        self.bump();
        let operand = self.unary_expr(r);
        self.ast
            .alloc_expr(ExprKind::Unary { op, operand }, self.span_from(lo))
    }

    /// Level 1: postfix (calls, indexing, field access, `?`).
    fn postfix_expr(&mut self, r: Restrictions) -> ExprId {
        let lo = self.peek().span.lo;
        let mut e = self.primary_expr(r);
        loop {
            match self.peek().kind {
                TokenKind::LParen => {
                    self.bump();
                    let mut args = Vec::new();
                    while !self.at(TokenKind::RParen) && !self.at(TokenKind::Eof) {
                        args.push(self.expr(DEFAULT));
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(TokenKind::RParen, "`)`");
                    e = self
                        .ast
                        .alloc_expr(ExprKind::Call { callee: e, args }, self.span_from(lo));
                }
                TokenKind::LBracket => {
                    self.bump();
                    let index = self.expr(DEFAULT);
                    self.expect(TokenKind::RBracket, "`]`");
                    e = self
                        .ast
                        .alloc_expr(ExprKind::Index { base: e, index }, self.span_from(lo));
                }
                TokenKind::Question => {
                    self.bump();
                    e = self.ast.alloc_expr(ExprKind::Try(e), self.span_from(lo));
                }
                TokenKind::Dot => {
                    self.bump();
                    let t = self.peek();
                    match t.kind {
                        TokenKind::Ident => {
                            self.bump();
                            let turbofish = if self.at(TokenKind::ColonColon)
                                && self.peek2() == TokenKind::Lt
                            {
                                self.bump();
                                self.generic_args(true)
                            } else {
                                None
                            };
                            e = self.ast.alloc_expr(
                                ExprKind::Field {
                                    base: e,
                                    name: t.span,
                                    turbofish,
                                },
                                self.span_from(lo),
                            );
                        }
                        TokenKind::Int { .. } => {
                            self.bump();
                            e = self.ast.alloc_expr(
                                ExprKind::TupleField {
                                    base: e,
                                    index: t.span,
                                },
                                self.span_from(lo),
                            );
                        }
                        // `x.0.1`: the lexer maximally munched `0.1` as a
                        // float; split it into two tuple-field accesses.
                        TokenKind::Float { suffix: None } => {
                            let text = self.text(t.span);
                            let dot = text.find('.');
                            let splittable = dot.is_some_and(|i| {
                                text[..i].bytes().all(|b| b.is_ascii_digit())
                                    && !text[i + 1..].is_empty()
                                    && text[i + 1..].bytes().all(|b| b.is_ascii_digit())
                            });
                            if let (Some(i), true) = (dot, splittable) {
                                let mid = t.span.lo + i as u32;
                                let first = Span {
                                    lo: t.span.lo,
                                    hi: mid,
                                };
                                self.tokens[self.pos] = Token {
                                    kind: TokenKind::Dot,
                                    span: Span {
                                        lo: mid,
                                        hi: mid + 1,
                                    },
                                };
                                self.tokens.insert(
                                    self.pos + 1,
                                    Token {
                                        kind: TokenKind::Int {
                                            base: crate::lexer::Base::Dec,
                                            suffix: None,
                                        },
                                        span: Span {
                                            lo: mid + 1,
                                            hi: t.span.hi,
                                        },
                                    },
                                );
                                e = self.ast.alloc_expr(
                                    ExprKind::TupleField {
                                        base: e,
                                        index: first,
                                    },
                                    self.span_from(lo),
                                );
                            } else {
                                self.expected("a field name or tuple index");
                                return e;
                            }
                        }
                        _ => {
                            self.expected("a field name or tuple index");
                            return e;
                        }
                    }
                }
                _ => return e,
            }
        }
    }

    fn primary_expr(&mut self, r: Restrictions) -> ExprId {
        let token = self.peek();
        let lo = token.span.lo;
        match token.kind {
            TokenKind::Int { base, suffix } => {
                self.bump();
                self.ast
                    .alloc_expr(ExprKind::Lit(Lit::Int { base, suffix }), token.span)
            }
            TokenKind::Float { suffix } => {
                self.bump();
                self.ast
                    .alloc_expr(ExprKind::Lit(Lit::Float { suffix }), token.span)
            }
            TokenKind::Str { raw } => {
                self.bump();
                self.ast
                    .alloc_expr(ExprKind::Lit(Lit::Str { raw }), token.span)
            }
            TokenKind::CharLit => {
                self.bump();
                self.ast.alloc_expr(ExprKind::Lit(Lit::Char), token.span)
            }
            TokenKind::Keyword(Kw::True) => {
                self.bump();
                self.ast
                    .alloc_expr(ExprKind::Lit(Lit::Bool(true)), token.span)
            }
            TokenKind::Keyword(Kw::False) => {
                self.bump();
                self.ast
                    .alloc_expr(ExprKind::Lit(Lit::Bool(false)), token.span)
            }
            TokenKind::Keyword(Kw::If) => self.if_expr(),
            TokenKind::Keyword(Kw::Match) => self.match_expr(),
            TokenKind::Keyword(Kw::Loop) => {
                self.bump();
                let body = self.block();
                self.ast
                    .alloc_expr(ExprKind::Loop { body }, self.span_from(lo))
            }
            TokenKind::Keyword(Kw::While) => {
                self.bump();
                let cond = self.expr(NO_STRUCT);
                let body = self.block();
                self.ast
                    .alloc_expr(ExprKind::While { cond, body }, self.span_from(lo))
            }
            TokenKind::Keyword(Kw::For) => {
                self.bump();
                let var = self
                    .expect_ident("a loop variable name")
                    .unwrap_or(token.span);
                self.expect(TokenKind::Keyword(Kw::In), "`in`");
                let iter = self.expr(NO_STRUCT);
                let body = self.block();
                self.ast
                    .alloc_expr(ExprKind::For { var, iter, body }, self.span_from(lo))
            }
            TokenKind::LBrace => {
                let block = self.block();
                self.ast
                    .alloc_expr(ExprKind::Block(block), self.span_from(lo))
            }
            TokenKind::LParen => {
                self.bump();
                if self.eat(TokenKind::RParen) {
                    return self
                        .ast
                        .alloc_expr(ExprKind::Tuple(Vec::new()), self.span_from(lo));
                }
                let first = self.expr(DEFAULT);
                if self.eat(TokenKind::RParen) {
                    return first; // grouping
                }
                let mut elems = vec![first];
                while self.eat(TokenKind::Comma) {
                    if self.at(TokenKind::RParen) {
                        break;
                    }
                    elems.push(self.expr(DEFAULT));
                }
                self.expect(TokenKind::RParen, "`)`");
                self.ast
                    .alloc_expr(ExprKind::Tuple(elems), self.span_from(lo))
            }
            TokenKind::LBracket => {
                self.bump();
                if self.eat(TokenKind::RBracket) {
                    return self
                        .ast
                        .alloc_expr(ExprKind::Array(Vec::new()), self.span_from(lo));
                }
                let first = self.expr(DEFAULT);
                if self.eat(TokenKind::Semi) {
                    let count = self.expr(DEFAULT);
                    self.expect(TokenKind::RBracket, "`]`");
                    return self.ast.alloc_expr(
                        ExprKind::Repeat {
                            value: first,
                            count,
                        },
                        self.span_from(lo),
                    );
                }
                let mut elems = vec![first];
                while self.eat(TokenKind::Comma) {
                    if self.at(TokenKind::RBracket) {
                        break;
                    }
                    elems.push(self.expr(DEFAULT));
                }
                self.expect(TokenKind::RBracket, "`]`");
                self.ast
                    .alloc_expr(ExprKind::Array(elems), self.span_from(lo))
            }
            _ if self.at_path_start() => {
                let Some(path) = self.path() else {
                    return self.ast.alloc_expr(ExprKind::Error, token.span);
                };
                // Turbofish: `path::<Args>`.
                let turbofish = if self.at(TokenKind::ColonColon) && self.peek2() == TokenKind::Lt {
                    self.bump();
                    self.generic_args(false)
                } else {
                    None
                };
                // Struct literal, unless restricted in this position.
                if self.at(TokenKind::LBrace) && !r.no_struct_literal && turbofish.is_none() {
                    self.bump();
                    let mut fields = Vec::new();
                    while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                        let Some(name) = self.expect_ident("a field name") else {
                            break;
                        };
                        let expr = if self.eat(TokenKind::Colon) {
                            Some(self.expr(DEFAULT))
                        } else {
                            None
                        };
                        fields.push(FieldInit { name, expr });
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(TokenKind::RBrace, "`}`");
                    return self
                        .ast
                        .alloc_expr(ExprKind::StructLit { path, fields }, self.span_from(lo));
                }
                self.ast
                    .alloc_expr(ExprKind::Path { path, turbofish }, self.span_from(lo))
            }
            TokenKind::Error => {
                // The lexer already reported it; consume and continue.
                self.bump();
                self.ast.alloc_expr(ExprKind::Error, token.span)
            }
            _ => {
                self.expected("an expression");
                if token.kind == TokenKind::Reserved {
                    // A reserved word can never start anything; consume it
                    // so it isn't reported a second time by the caller.
                    self.bump();
                }
                self.ast.alloc_expr(ExprKind::Error, token.span)
            }
        }
    }

    fn if_expr(&mut self) -> ExprId {
        let lo = self.peek().span.lo;
        self.bump(); // if
        let cond = self.expr(NO_STRUCT);
        let then_block = self.block();
        let else_ = if self.eat_kw(Kw::Else) {
            if self.at_kw(Kw::If) {
                Some(self.if_expr())
            } else {
                let block_lo = self.peek().span.lo;
                let block = self.block();
                Some(
                    self.ast
                        .alloc_expr(ExprKind::Block(block), self.span_from(block_lo)),
                )
            }
        } else {
            None
        };
        self.ast.alloc_expr(
            ExprKind::If {
                cond,
                then_block,
                else_,
            },
            self.span_from(lo),
        )
    }

    fn match_expr(&mut self) -> ExprId {
        let lo = self.peek().span.lo;
        self.bump(); // match
        let scrutinee = self.expr(NO_STRUCT);
        self.expect(TokenKind::LBrace, "`{`");
        let mut arms = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let before = self.pos;
            let pat = self.pattern();
            self.expect(TokenKind::FatArrow, "`=>`");
            let body = self.expr(DEFAULT);
            arms.push(MatchArm { pat, body });
            self.eat(TokenKind::Comma);
            if self.pos == before {
                // Never loop without progress on malformed arms.
                self.bump();
            }
        }
        self.expect(TokenKind::RBrace, "`}`");
        self.ast
            .alloc_expr(ExprKind::Match { scrutinee, arms }, self.span_from(lo))
    }

    // ----------------------------------------------------------- patterns --

    fn pattern(&mut self) -> PatId {
        if self.depth_exceeded() {
            let span = self.peek().span;
            return self.ast.alloc_pat(PatKind::Wild, span);
        }
        self.depth += 1;
        let p = self.pattern_inner();
        self.depth -= 1;
        p
    }

    fn pattern_inner(&mut self) -> PatId {
        let token = self.peek();
        let lo = token.span.lo;
        match token.kind {
            TokenKind::Int { base, suffix } => {
                self.bump();
                self.ast
                    .alloc_pat(PatKind::Lit(Lit::Int { base, suffix }), token.span)
            }
            TokenKind::Float { suffix } => {
                self.bump();
                self.ast
                    .alloc_pat(PatKind::Lit(Lit::Float { suffix }), token.span)
            }
            TokenKind::Str { raw } => {
                self.bump();
                self.ast
                    .alloc_pat(PatKind::Lit(Lit::Str { raw }), token.span)
            }
            TokenKind::CharLit => {
                self.bump();
                self.ast.alloc_pat(PatKind::Lit(Lit::Char), token.span)
            }
            TokenKind::Keyword(Kw::True) => {
                self.bump();
                self.ast
                    .alloc_pat(PatKind::Lit(Lit::Bool(true)), token.span)
            }
            TokenKind::Keyword(Kw::False) => {
                self.bump();
                self.ast
                    .alloc_pat(PatKind::Lit(Lit::Bool(false)), token.span)
            }
            TokenKind::Ident if self.text(token.span) == "_" => {
                self.bump();
                self.ast.alloc_pat(PatKind::Wild, token.span)
            }
            TokenKind::LParen => {
                self.bump();
                let mut pats = Vec::new();
                while !self.at(TokenKind::RParen) && !self.at(TokenKind::Eof) {
                    pats.push(self.pattern());
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RParen, "`)`");
                self.ast.alloc_pat(PatKind::Tuple(pats), self.span_from(lo))
            }
            TokenKind::LBracket => {
                self.bump();
                let mut pats = Vec::new();
                while !self.at(TokenKind::RBracket) && !self.at(TokenKind::Eof) {
                    pats.push(self.pattern());
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RBracket, "`]`");
                self.ast.alloc_pat(PatKind::Array(pats), self.span_from(lo))
            }
            _ if self.at_path_start() => {
                let Some(path) = self.path() else {
                    return self.ast.alloc_pat(PatKind::Wild, token.span);
                };
                if self.eat(TokenKind::LParen) {
                    let mut pats = Vec::new();
                    while !self.at(TokenKind::RParen) && !self.at(TokenKind::Eof) {
                        pats.push(self.pattern());
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(TokenKind::RParen, "`)`");
                    return self
                        .ast
                        .alloc_pat(PatKind::TupleVariant { path, pats }, self.span_from(lo));
                }
                if self.eat(TokenKind::LBrace) {
                    let mut fields = Vec::new();
                    while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                        let name = if self.at(TokenKind::Ident) {
                            self.bump().span
                        } else {
                            // `_` in field-pattern position, or junk.
                            self.expected("a field name");
                            break;
                        };
                        let pat = if self.eat(TokenKind::Colon) {
                            Some(self.pattern())
                        } else {
                            None
                        };
                        fields.push(FieldPat { name, pat });
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(TokenKind::RBrace, "`}`");
                    return self
                        .ast
                        .alloc_pat(PatKind::Struct { path, fields }, self.span_from(lo));
                }
                if path.segments.len() == 1 && path.segments[0].kind == SegmentKind::Ident {
                    let span = path.segments[0].span;
                    self.ast.alloc_pat(PatKind::Binding(span), span)
                } else {
                    let span = path.span;
                    self.ast.alloc_pat(PatKind::Path(path), span)
                }
            }
            _ => {
                self.expected("a pattern");
                self.bump();
                self.ast.alloc_pat(PatKind::Wild, token.span)
            }
        }
    }

    // -------------------------------------------------- blocks/statements --

    fn block_formed_start(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::LBrace
                | TokenKind::Keyword(Kw::If | Kw::Match | Kw::Loop | Kw::While | Kw::For)
        )
    }

    fn block(&mut self) -> BlockId {
        if self.depth_exceeded() {
            let span = self.peek().span;
            return self.ast.alloc_block(BlockNode {
                stmts: Vec::new(),
                tail: None,
                span,
            });
        }
        self.depth += 1;
        let b = self.block_inner();
        self.depth -= 1;
        b
    }

    fn block_inner(&mut self) -> BlockId {
        let lo = self.peek().span.lo;
        if !self.expect(TokenKind::LBrace, "`{`") {
            return self.ast.alloc_block(BlockNode {
                stmts: Vec::new(),
                tail: None,
                span: self.peek().span,
            });
        }
        let (stmts, tail) = self.block_elements(TokenKind::RBrace, false);
        self.expect(TokenKind::RBrace, "`}`");
        self.ast.alloc_block(BlockNode {
            stmts,
            tail,
            span: self.span_from(lo),
        })
    }

    /// `Statement* Expression?` up to (not consuming) `terminator`.
    /// With `allow_items` (snippet mode), items parse into `StmtKind::Item`;
    /// inside real blocks an item keyword is an error (Core v1 blocks do not
    /// nest items) but the item is parsed anyway for recovery.
    fn block_elements(
        &mut self,
        terminator: TokenKind,
        allow_items: bool,
    ) -> (Vec<StmtId>, Option<ExprId>) {
        let mut stmts = Vec::new();
        let mut tail = None;
        while !self.at(terminator) && !self.at(TokenKind::Eof) {
            let before = self.pos;
            if self.at_item_start() {
                let span = self.peek().span;
                if !allow_items {
                    self.error("items are not allowed inside blocks in Core v1", span);
                }
                if let Some(item) = self.item() {
                    stmts.push(self.ast.alloc_stmt(StmtKind::Item(item), span));
                } else {
                    self.recover_stmt();
                    stmts.push(self.ast.alloc_stmt(StmtKind::Error, span));
                }
            } else {
                match self.statement_or_tail(terminator) {
                    Element::Stmt(id) => stmts.push(id),
                    Element::Tail(expr) => {
                        tail = Some(expr);
                        break;
                    }
                }
            }
            if self.pos == before {
                // Safety net: never loop without progress.
                let span = self.peek().span;
                self.expected("a statement");
                self.bump();
                stmts.push(self.ast.alloc_stmt(StmtKind::Error, span));
            }
        }
        (stmts, tail)
    }

    fn statement_or_tail(&mut self, terminator: TokenKind) -> Element {
        let token = self.peek();
        let lo = token.span.lo;
        match token.kind {
            TokenKind::Semi => {
                self.bump();
                Element::Stmt(self.ast.alloc_stmt(StmtKind::Empty, token.span))
            }
            TokenKind::Keyword(Kw::Let) => {
                self.bump();
                let mutable = self.eat_kw(Kw::Mut);
                let Some(name) = self.expect_ident("a binding name") else {
                    self.recover_stmt();
                    return Element::Stmt(self.ast.alloc_stmt(StmtKind::Error, self.span_from(lo)));
                };
                let ty = if self.eat(TokenKind::Colon) {
                    Some(self.ty())
                } else {
                    None
                };
                let init = if self.eat(TokenKind::Eq) {
                    Some(self.expr(DEFAULT))
                } else {
                    None
                };
                if ty.is_none() && init.is_none() {
                    self.diags.push(
                        Diagnostic::error(
                            "a `let` binding requires a type annotation or an initializer",
                            self.span_from(lo),
                        )
                        .with_help("write `let x: Type;` or `let x = value;`"),
                    );
                }
                if !self.eat(TokenKind::Semi) {
                    self.expected("`;`");
                    self.recover_stmt();
                }
                Element::Stmt(self.ast.alloc_stmt(
                    StmtKind::Let {
                        mutable,
                        name,
                        ty,
                        init,
                    },
                    self.span_from(lo),
                ))
            }
            TokenKind::Keyword(Kw::Return) => {
                self.bump();
                let expr = if self.at(TokenKind::Semi) {
                    None
                } else {
                    Some(self.expr(DEFAULT))
                };
                if !self.eat(TokenKind::Semi) {
                    self.expected("`;`");
                    self.recover_stmt();
                }
                Element::Stmt(
                    self.ast
                        .alloc_stmt(StmtKind::Return(expr), self.span_from(lo)),
                )
            }
            TokenKind::Keyword(Kw::Break) => {
                self.bump();
                let expr = if self.at(TokenKind::Semi) {
                    None
                } else {
                    Some(self.expr(DEFAULT))
                };
                if !self.eat(TokenKind::Semi) {
                    self.expected("`;`");
                    self.recover_stmt();
                }
                Element::Stmt(
                    self.ast
                        .alloc_stmt(StmtKind::Break(expr), self.span_from(lo)),
                )
            }
            TokenKind::Keyword(Kw::Continue) => {
                self.bump();
                if !self.eat(TokenKind::Semi) {
                    self.expected("`;`");
                    self.recover_stmt();
                }
                Element::Stmt(self.ast.alloc_stmt(StmtKind::Continue, self.span_from(lo)))
            }
            _ if self.block_formed_start() => {
                // Block-formed expression statement: `;` optional; a
                // block-formed expression immediately before the block's end
                // is its trailing expression (02 "Block-formed expression
                // statements"). Statement parsing is greedy: no operator may
                // extend it.
                let expr = self.block_formed_expr();
                if self.eat(TokenKind::Semi) {
                    Element::Stmt(
                        self.ast
                            .alloc_stmt(StmtKind::Expr { expr, semi: true }, self.span_from(lo)),
                    )
                } else if self.at(terminator) {
                    Element::Tail(expr)
                } else {
                    Element::Stmt(
                        self.ast
                            .alloc_stmt(StmtKind::Expr { expr, semi: false }, self.span_from(lo)),
                    )
                }
            }
            _ => {
                let expr = self.expr(DEFAULT);
                if self.eat(TokenKind::Semi) {
                    Element::Stmt(
                        self.ast
                            .alloc_stmt(StmtKind::Expr { expr, semi: true }, self.span_from(lo)),
                    )
                } else if self.at(terminator) {
                    Element::Tail(expr)
                } else {
                    self.expected("`;`");
                    self.recover_stmt();
                    Element::Stmt(
                        self.ast
                            .alloc_stmt(StmtKind::Expr { expr, semi: true }, self.span_from(lo)),
                    )
                }
            }
        }
    }

    /// A block-formed expression at statement position (greedy: not extended
    /// by postfix or binary operators, per the spec's statement rule).
    fn block_formed_expr(&mut self) -> ExprId {
        let lo = self.peek().span.lo;
        match self.peek().kind {
            TokenKind::Keyword(Kw::If) => self.if_expr(),
            TokenKind::Keyword(Kw::Match) => self.match_expr(),
            TokenKind::Keyword(Kw::Loop) => {
                self.bump();
                let body = self.block();
                self.ast
                    .alloc_expr(ExprKind::Loop { body }, self.span_from(lo))
            }
            TokenKind::Keyword(Kw::While) => {
                self.bump();
                let cond = self.expr(NO_STRUCT);
                let body = self.block();
                self.ast
                    .alloc_expr(ExprKind::While { cond, body }, self.span_from(lo))
            }
            TokenKind::Keyword(Kw::For) => {
                let kw_span = self.peek().span;
                self.bump();
                let var = self.expect_ident("a loop variable name").unwrap_or(kw_span);
                self.expect(TokenKind::Keyword(Kw::In), "`in`");
                let iter = self.expr(NO_STRUCT);
                let body = self.block();
                self.ast
                    .alloc_expr(ExprKind::For { var, iter, body }, self.span_from(lo))
            }
            _ => {
                let block = self.block();
                self.ast
                    .alloc_expr(ExprKind::Block(block), self.span_from(lo))
            }
        }
    }

    // -------------------------------------------------------------- items --

    fn generic_params(&mut self) -> Vec<GenericParam> {
        let mut params = Vec::new();
        if !self.eat(TokenKind::Lt) {
            return params;
        }
        loop {
            if self.eat_gt() {
                return params;
            }
            if self.at(TokenKind::Eof) {
                self.expected("`>`");
                return params;
            }
            let Some(name) = self.expect_ident("a generic parameter name") else {
                self.bump(); // skip junk so the loop can't get stuck
                continue;
            };
            let mut bounds = Vec::new();
            if self.eat(TokenKind::Colon) {
                loop {
                    if let Some(bound) = self.trait_ref() {
                        bounds.push(bound);
                    }
                    if !self.eat(TokenKind::Plus) {
                        break;
                    }
                }
            }
            params.push(GenericParam { name, bounds });
            if !self.eat(TokenKind::Comma) {
                if !self.eat_gt() {
                    self.expected("`,` or `>`");
                }
                return params;
            }
        }
    }

    /// `TraitBound ::= Path GenericArgs?`
    fn trait_ref(&mut self) -> Option<TraitRef> {
        let path = self.path()?;
        let args = if self.at(TokenKind::Lt) {
            self.generic_args(false)
        } else {
            None
        };
        Some(TraitRef { path, args })
    }

    fn fn_sig(&mut self) -> Option<FnSig> {
        let lo = self.peek().span.lo;
        if !self.expect(TokenKind::Keyword(Kw::Fn), "`fn`") {
            return None;
        }
        let name = self.expect_ident("a function name")?;
        let generics = if self.at(TokenKind::Lt) {
            self.generic_params()
        } else {
            Vec::new()
        };
        self.expect(TokenKind::LParen, "`(`");
        let mut receiver = None;
        let mut params = Vec::new();
        let mut first = true;
        while !self.at(TokenKind::RParen) && !self.at(TokenKind::Eof) {
            // Receiver: `self` | `&self` | `&mut self`, first position only.
            let is_receiver = self.at_kw(Kw::SelfLower)
                || (self.at(TokenKind::Amp)
                    && matches!(self.peek2(), TokenKind::Keyword(Kw::SelfLower | Kw::Mut)));
            if is_receiver {
                let recv_span = self.peek().span;
                let recv = if self.eat_kw(Kw::SelfLower) {
                    Receiver::Value
                } else {
                    self.bump(); // &
                    if self.eat_kw(Kw::Mut) {
                        self.expect(TokenKind::Keyword(Kw::SelfLower), "`self`");
                        Receiver::RefMut
                    } else {
                        self.expect(TokenKind::Keyword(Kw::SelfLower), "`self`");
                        Receiver::Ref
                    }
                };
                if !first {
                    self.error("a receiver must be the first parameter", recv_span);
                } else if !self.in_impl_or_trait {
                    self.error(
                        "a receiver is only valid on functions inside a `trait` or `impl` block",
                        recv_span,
                    );
                }
                receiver = Some(recv);
            } else {
                let mutable = self.eat_kw(Kw::Mut);
                let Some(pname) = self.expect_ident("a parameter name") else {
                    self.recover_param_list();
                    break;
                };
                self.expect(TokenKind::Colon, "`:`");
                let ty = self.ty();
                params.push(Param {
                    mutable,
                    name: pname,
                    ty,
                });
            }
            first = false;
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RParen, "`)`");
        let ret = if self.eat(TokenKind::Arrow) {
            if self.at(TokenKind::Bang) {
                let span = self.bump().span;
                RetTy::Never(span)
            } else {
                RetTy::Ty(self.ty())
            }
        } else {
            RetTy::Unit
        };
        Some(FnSig {
            name,
            generics,
            receiver,
            params,
            ret,
            span: self.span_from(lo),
        })
    }

    fn recover_param_list(&mut self) {
        let mut depth = 0usize;
        loop {
            match self.peek().kind {
                TokenKind::Eof => return,
                TokenKind::RParen if depth == 0 => return,
                TokenKind::LParen => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::RParen => {
                    depth -= 1;
                    self.bump();
                }
                _ => {
                    self.bump();
                }
            }
        }
    }

    fn field_list(&mut self) -> Vec<FieldDef> {
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let is_pub = self.eat_kw(Kw::Pub);
            let Some(name) = self.expect_ident("a field name") else {
                break;
            };
            self.expect(TokenKind::Colon, "`:`");
            let ty = self.ty();
            fields.push(FieldDef { is_pub, name, ty });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        fields
    }

    /// One item, or `None` when parsing failed badly enough to recover.
    fn item(&mut self) -> Option<ItemId> {
        if self.depth_exceeded() {
            return None;
        }
        self.depth += 1;
        let i = self.item_inner();
        self.depth -= 1;
        i
    }

    fn item_inner(&mut self) -> Option<ItemId> {
        let lo = self.peek().span.lo;
        let vis = self.item_visibility();
        let kind = match self.peek().kind {
            TokenKind::Keyword(Kw::Fn) => self.parse_fn()?,
            TokenKind::Keyword(Kw::Struct) => self.parse_struct()?,
            TokenKind::Keyword(Kw::Enum) => self.parse_enum()?,
            TokenKind::Keyword(Kw::Trait) => self.parse_trait()?,
            TokenKind::Keyword(Kw::Impl) => self.parse_impl(),
            TokenKind::Keyword(Kw::Const) => self.parse_const()?,
            TokenKind::Keyword(Kw::Type) => self.parse_type_alias()?,
            TokenKind::Keyword(Kw::Use) => self.parse_use()?,
            TokenKind::Keyword(Kw::Mod) => self.parse_mod()?,
            // `model` (D4) is a contextual keyword lexed as an identifier.
            TokenKind::Ident if self.text(self.peek().span) == "model" => self.parse_model()?,
            _ => {
                self.expected("an item");
                return None;
            }
        };
        let item_id = self.ast.alloc_item(kind, vis, self.span_from(lo));
        let file_arc = std::sync::Arc::new(SourceFile::new(
            self.file.name.clone(),
            self.file.src.clone(),
        ));
        self.ast.item_files.insert(item_id, file_arc);
        Some(item_id)
    }

    fn item_visibility(&mut self) -> Option<Vis> {
        if self.eat_kw(Kw::Pub) {
            Some(Vis::Pub)
        } else if self.eat_kw(Kw::Priv) {
            Some(Vis::Priv)
        } else {
            None
        }
    }

    /// Optional `GenericParams`, else empty.
    fn opt_generic_params(&mut self) -> Vec<GenericParam> {
        if self.at(TokenKind::Lt) {
            self.generic_params()
        } else {
            Vec::new()
        }
    }

    fn parse_fn(&mut self) -> Option<ItemKind> {
        let sig = self.fn_sig()?;
        let body = self.block();
        Some(ItemKind::Fn(FnDef { sig, body }))
    }

    fn parse_struct(&mut self) -> Option<ItemKind> {
        self.bump();
        let name = self.expect_ident("a struct name")?;
        let generics = self.opt_generic_params();
        self.expect(TokenKind::LBrace, "`{`");
        let fields = self.field_list();
        self.expect(TokenKind::RBrace, "`}`");
        Some(ItemKind::Struct {
            name,
            generics,
            fields,
        })
    }

    /// `model Name<...> { input p: T; output q: U; }` (extension D4). `model`,
    /// `input`, and `output` are contextual keywords (identifiers). Emits a
    /// focused diagnostic naming the `tensor` extension when it is disabled.
    fn parse_model(&mut self) -> Option<ItemKind> {
        let kw_span = self.peek().span;
        self.bump(); // `model`
        if !self.tensor_enabled() {
            self.error("`model` declarations require extension `tensor`", kw_span);
        }
        let name = self.expect_ident("a model name")?;
        let generics = self.opt_generic_params();
        self.expect(TokenKind::LBrace, "`{`");
        let mut ports = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            match self.parse_model_port() {
                Some(port) => ports.push(port),
                None => break,
            }
        }
        self.expect(TokenKind::RBrace, "`}`");
        Some(ItemKind::Model(ModelDef {
            name,
            generics,
            ports,
        }))
    }

    fn parse_model_port(&mut self) -> Option<ModelPort> {
        let lo = self.peek().span.lo;
        let dir = if self.at(TokenKind::Ident) && self.text(self.peek().span) == "input" {
            self.bump();
            PortDir::Input
        } else if self.at(TokenKind::Ident) && self.text(self.peek().span) == "output" {
            self.bump();
            PortDir::Output
        } else {
            self.expected("`input` or `output`");
            return None;
        };
        let name = self.expect_ident("a port name")?;
        self.expect(TokenKind::Colon, "`:`");
        let ty = self.ty();
        self.expect(TokenKind::Semi, "`;`");
        Some(ModelPort {
            dir,
            name,
            ty,
            span: self.span_from(lo),
        })
    }

    fn parse_enum(&mut self) -> Option<ItemKind> {
        self.bump();
        let name = self.expect_ident("an enum name")?;
        let generics = self.opt_generic_params();
        self.expect(TokenKind::LBrace, "`{`");
        let mut variants = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let Some(vname) = self.expect_ident("a variant name") else {
                break;
            };
            let kind = if self.eat(TokenKind::LParen) {
                let mut tys = Vec::new();
                while !self.at(TokenKind::RParen) && !self.at(TokenKind::Eof) {
                    tys.push(self.ty());
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RParen, "`)`");
                VariantKind::Tuple(tys)
            } else if self.eat(TokenKind::LBrace) {
                let fields = self.field_list();
                self.expect(TokenKind::RBrace, "`}`");
                VariantKind::Struct(fields)
            } else {
                VariantKind::Unit
            };
            variants.push(Variant { name: vname, kind });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBrace, "`}`");
        Some(ItemKind::Enum {
            name,
            generics,
            variants,
        })
    }

    fn parse_trait(&mut self) -> Option<ItemKind> {
        self.bump();
        let name = self.expect_ident("a trait name")?;
        let generics = self.opt_generic_params();
        self.expect(TokenKind::LBrace, "`{`");
        let was = self.in_impl_or_trait;
        self.in_impl_or_trait = true;
        let mut items = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let before = self.pos;
            if self.eat_kw(Kw::Type) {
                let Some(tname) = self.expect_ident("an associated type name") else {
                    self.recover_stmt();
                    continue;
                };
                self.expect(TokenKind::Semi, "`;`");
                items.push(TraitItem::AssocType { name: tname });
            } else if self.at_kw(Kw::Fn) {
                let Some(sig) = self.fn_sig() else {
                    self.recover_stmt();
                    continue;
                };
                let body = if self.at(TokenKind::LBrace) {
                    Some(self.block())
                } else {
                    self.expect(TokenKind::Semi, "`;` or a method body");
                    None
                };
                items.push(TraitItem::Method { sig, body });
            } else {
                self.expected("a trait item (`fn` or `type`)");
                self.recover_stmt();
            }
            if self.pos == before {
                // recover_stmt stops (without consuming) at item
                // keywords; never loop without progress.
                self.bump();
            }
        }
        self.in_impl_or_trait = was;
        self.expect(TokenKind::RBrace, "`}`");
        Some(ItemKind::Trait {
            name,
            generics,
            items,
        })
    }

    fn parse_impl(&mut self) -> ItemKind {
        self.bump();
        let generics = self.opt_generic_params();
        let was = self.in_impl_or_trait;
        self.in_impl_or_trait = true;
        let first_ty = self.ty();
        let (trait_, self_ty) = if self.eat_kw(Kw::For) {
            let trait_ref = self.type_as_trait_ref(first_ty);
            let self_ty = self.ty();
            (trait_ref, self_ty)
        } else {
            (None, first_ty)
        };
        self.expect(TokenKind::LBrace, "`{`");
        let mut items = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let before = self.pos;
            if self.eat_kw(Kw::Type) {
                let Some(tname) = self.expect_ident("an associated type name") else {
                    self.recover_stmt();
                    continue;
                };
                self.expect(TokenKind::Eq, "`=`");
                let ty = self.ty();
                self.expect(TokenKind::Semi, "`;`");
                items.push(ImplItem::AssocType { name: tname, ty });
            } else if self.at_kw(Kw::Fn)
                || ((self.at_kw(Kw::Pub) || self.at_kw(Kw::Priv))
                    && self.peek2() == TokenKind::Keyword(Kw::Fn))
            {
                let fvis = self.item_visibility();
                let Some(sig) = self.fn_sig() else {
                    self.recover_stmt();
                    continue;
                };
                if !self.at(TokenKind::LBrace) {
                    // API-notation `fn f(...) -> T;` is not valid in
                    // real code (06 "Notation").
                    self.expected("a method body");
                    self.eat(TokenKind::Semi);
                    continue;
                }
                let body = self.block();
                items.push(ImplItem::Fn {
                    vis: fvis,
                    def: FnDef { sig, body },
                });
            } else {
                self.expected("an impl item (`fn` or `type`)");
                self.recover_stmt();
            }
            if self.pos == before {
                // recover_stmt stops (without consuming) at item
                // keywords; never loop without progress.
                self.bump();
            }
        }
        self.in_impl_or_trait = was;
        self.expect(TokenKind::RBrace, "`}`");
        ItemKind::Impl {
            generics,
            trait_,
            self_ty,
            items,
        }
    }

    fn parse_const(&mut self) -> Option<ItemKind> {
        self.bump();
        let name = self.expect_ident("a constant name")?;
        self.expect(TokenKind::Colon, "`:`");
        let ty = self.ty();
        self.expect(TokenKind::Eq, "`=`");
        let value = self.expr(DEFAULT);
        self.expect(TokenKind::Semi, "`;`");
        Some(ItemKind::Const { name, ty, value })
    }

    fn parse_type_alias(&mut self) -> Option<ItemKind> {
        self.bump();
        let name = self.expect_ident("a type alias name")?;
        let generics = self.opt_generic_params();
        self.expect(TokenKind::Eq, "`=`");
        let ty = self.ty();
        self.expect(TokenKind::Semi, "`;`");
        Some(ItemKind::TypeAlias { name, generics, ty })
    }

    fn parse_use(&mut self) -> Option<ItemKind> {
        self.bump();
        let tree = self.use_tree()?;
        self.expect(TokenKind::Semi, "`;`");
        Some(ItemKind::Use(tree))
    }

    fn parse_mod(&mut self) -> Option<ItemKind> {
        self.bump();
        let name = self.expect_ident("a module name")?;
        if self.eat(TokenKind::Semi) {
            return Some(ItemKind::Mod { name, items: None });
        }
        self.expect(TokenKind::LBrace, "`{` or `;`");
        let mut items = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let before = self.pos;
            match self.item() {
                Some(item) => items.push(item),
                None => self.recover_item(),
            }
            if self.pos == before {
                self.bump();
            }
        }
        self.expect(TokenKind::RBrace, "`}`");
        Some(ItemKind::Mod {
            name,
            items: Some(items),
        })
    }

    /// Reinterpret the already-parsed type before `for` as the trait
    /// reference of `impl Trait for Type`.
    fn type_as_trait_ref(&mut self, ty: TypeId) -> Option<TraitRef> {
        let node = self.ast.ty(ty);
        match &node.kind {
            TypeKind::Path { path, args } => Some(TraitRef {
                path: path.clone(),
                args: args.clone(),
            }),
            _ => {
                let span = node.span;
                self.error("the trait in `impl Trait for Type` must be a path", span);
                None
            }
        }
    }

    fn use_tree(&mut self) -> Option<UseTree> {
        let lo = self.peek().span.lo;
        let first = self.path_segment(true, false)?;
        let mut prev_super = first.kind == SegmentKind::Super;
        let mut segments = vec![first];
        loop {
            if !self.at(TokenKind::ColonColon) {
                break;
            }
            match self.peek2() {
                TokenKind::Star => {
                    self.bump();
                    self.bump();
                    let prefix = Path {
                        segments,
                        span: self.span_from(lo),
                    };
                    return Some(UseTree::Glob { prefix });
                }
                TokenKind::LBrace => {
                    self.bump();
                    self.bump();
                    let prefix = Path {
                        segments,
                        span: self.span_from(lo),
                    };
                    let mut items = Vec::new();
                    while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                        if let Some(tree) = self.use_tree() {
                            items.push(tree);
                        } else {
                            break;
                        }
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(TokenKind::RBrace, "`}`");
                    return Some(UseTree::Group { prefix, items });
                }
                TokenKind::Keyword(Kw::SelfLower) => {
                    // `use a::b::self;` — trailing self-import.
                    self.bump();
                    self.bump();
                    let prefix = Path {
                        segments,
                        span: self.span_from(lo),
                    };
                    return Some(UseTree::SelfImport { prefix });
                }
                _ => {
                    self.bump(); // ::
                    let seg = self.path_segment(false, prev_super)?;
                    prev_super = prev_super && seg.kind == SegmentKind::Super;
                    segments.push(seg);
                }
            }
        }
        let path = Path {
            segments,
            span: self.span_from(lo),
        };
        let alias = if self.eat_kw(Kw::As) {
            self.expect_ident("an alias name")
        } else {
            None
        };
        Some(UseTree::Path { path, alias })
    }

    // ------------------------------------------------------- entry points --

    fn program(&mut self) -> Vec<ItemId> {
        let mut items = Vec::new();
        while !self.at(TokenKind::Eof) {
            let before = self.pos;
            if self.at_item_start() {
                match self.item() {
                    Some(item) => items.push(item),
                    None => self.recover_item(),
                }
            } else {
                self.expected("an item");
                self.recover_item();
            }
            if self.pos == before {
                self.bump();
            }
        }
        items
    }

    fn snippet(&mut self) -> (Vec<StmtId>, Option<ExprId>) {
        self.block_elements(TokenKind::Eof, true)
    }
}

enum Element {
    Stmt(StmtId),
    Tail(ExprId),
}

// ------------------------------------------------------------------ tests --

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_dump;

    fn parse_ok(src: &str, mode: ParseMode) -> Ast {
        let file = SourceFile::new("test.stark", src.to_string());
        let (ast, diags) = parse(&file, mode);
        assert!(
            diags.is_empty(),
            "unexpected diagnostics for {src:?}: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        ast
    }

    fn parse_err(src: &str, mode: ParseMode) -> Vec<String> {
        let file = SourceFile::new("test.stark", src.to_string());
        let (_ast, diags) = parse(&file, mode);
        assert!(!diags.is_empty(), "expected diagnostics for {src:?}");
        diags.iter().map(|d| d.message.clone()).collect()
    }

    fn dump_of(src: &str, mode: ParseMode) -> String {
        let file = SourceFile::new("test.stark", src.to_string());
        let (ast, diags) = parse(&file, mode);
        assert!(
            diags.is_empty(),
            "unexpected diagnostics for {src:?}: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        ast_dump::dump(&ast, &file)
    }

    // ----- tensor extension (M4.1) helpers -----

    fn tensor_ok(src: &str, mode: ParseMode) -> Ast {
        let file = SourceFile::new("test.stark", src.to_string());
        let (ast, diags) = parse_with_options(&file, mode, LanguageOptions::with_tensor());
        assert!(
            diags.is_empty(),
            "unexpected diagnostics for {src:?}: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        ast
    }

    fn tensor_err(src: &str, mode: ParseMode) -> Vec<String> {
        let file = SourceFile::new("test.stark", src.to_string());
        let (_ast, diags) = parse_with_options(&file, mode, LanguageOptions::with_tensor());
        assert!(!diags.is_empty(), "expected diagnostics for {src:?}");
        diags.iter().map(|d| d.message.clone()).collect()
    }

    fn tensor_dump(src: &str, mode: ParseMode) -> String {
        let file = SourceFile::new("test.stark", src.to_string());
        let (ast, diags) = parse_with_options(&file, mode, LanguageOptions::with_tensor());
        assert!(
            diags.is_empty(),
            "unexpected diagnostics for {src:?}: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        ast_dump::dump(&ast, &file)
    }

    #[test]
    fn tensor_static_symbolic_scalar_device_shapes() {
        // static, symbolic, scalar `[]`, device-qualified
        tensor_ok("let a: Tensor<Float32, [1024, 768]>;", ParseMode::Snippet);
        tensor_ok(
            "let b: Tensor<Float32, [B, 3, 224, 224]>;",
            ParseMode::Snippet,
        );
        tensor_ok("let c: Tensor<Float32, []>;", ParseMode::Snippet);
        tensor_ok(
            "let d: Tensor<Float16, [B, 1000], device = D>;",
            ParseMode::Snippet,
        );
        tensor_ok("let e: TensorDyn<Float32>;", ParseMode::Snippet);
        tensor_ok(
            "let gpu: Tensor<Float32, [1], device = Cuda<0>>;",
            ParseMode::Snippet,
        );
    }

    #[test]
    fn tensor_dim_expr_precedence_and_parens() {
        // `*` binds tighter than `+`/`-`; parens override.
        let d = tensor_dump("let x: Tensor<Float32, [B * H + 1]>;", ParseMode::Snippet);
        assert!(d.contains("dim-binary +"), "{d}");
        assert!(d.contains("dim-binary *"), "{d}");
        let p = tensor_dump("let x: Tensor<Float32, [B * (H + 1)]>;", ParseMode::Snippet);
        // Outer op is `*` when the sum is parenthesized.
        let mul_line = p.lines().find(|l| l.contains("dim-binary")).unwrap();
        assert!(mul_line.contains('*'), "{p}");
    }

    #[test]
    fn tensor_empty_and_trailing_comma_shapes() {
        tensor_ok("let a: Tensor<Float32, []>;", ParseMode::Snippet);
        tensor_ok("let b: Tensor<Float32, [B, 3,]>;", ParseMode::Snippet);
    }

    #[test]
    fn model_single_and_multiple_ports() {
        tensor_ok(
            "model M<B: Dim> { input x: Tensor<Float32, [B, 3]>; output y: Tensor<Float32, [B]>; }",
            ParseMode::Program,
        );
        tensor_ok(
            "model M<B: Dim> { input a: Tensor<Float32, [B]>; input b: Tensor<Float32, [B]>; \
             output p: Tensor<Float32, [B]>; output q: Tensor<Float32, [B]>; }",
            ParseMode::Program,
        );
        let d = tensor_dump(
            "model ResNet<B: Dim> { input image: Tensor<Float32, [B, 3, 224, 224]>; \
             output class: Tensor<Float32, [B, 1000]>; }",
            ParseMode::Program,
        );
        assert!(d.contains("model ResNet"), "{d}");
        assert!(d.contains("port input image"), "{d}");
        assert!(d.contains("port output class"), "{d}");
    }

    #[test]
    fn index_list_turbofish_argument() {
        // Index lists share the shape surface; usable in path-expr turbofish.
        let d = tensor_dump("let y = permute::<[0, 2, 1]>(x);", ParseMode::Snippet);
        assert!(d.contains("dim-lit 0"), "{d}");
        assert!(d.contains("dim-lit 2"), "{d}");
    }

    #[test]
    fn malformed_model_port_and_dim_and_index() {
        // malformed model port (missing direction)
        tensor_err("model M { x: Tensor<Float32, [B]>; }", ParseMode::Program);
        // malformed dim expression (dangling operator)
        tensor_err("let x: Tensor<Float32, [B * ]>;", ParseMode::Snippet);
        // invalid index-list syntax (missing element)
        tensor_err("let y = permute::<[0, , 1]>(x);", ParseMode::Snippet);
    }

    #[test]
    fn core_only_rejects_extension_constructs() {
        // D2: shape argument
        assert!(
            parse_err("let x: Tensor<Float32, [B, 3]>;", ParseMode::Snippet)
                .iter()
                .any(|m| m.contains("require extension `tensor`"))
        );
        // D4: model item
        assert!(parse_err(
            "model M { input x: Tensor<Float32, [B]>; output y: Tensor<Float32, [B]>; }",
            ParseMode::Program
        )
        .iter()
        .any(|m| m.contains("`model` declarations require extension `tensor`")));
    }

    #[test]
    fn reserved_named_axes_rejected_even_with_extension() {
        assert!(
            tensor_err("let x: Tensor<Float32, [batch: B, 3]>;", ParseMode::Snippet)
                .iter()
                .any(|m| m.contains("named axes are reserved"))
        );
    }

    #[test]
    fn all_tensor_future_type_and_constraint_classes_are_reserved() {
        let cases = [
            (
                "fn f(x: Tensor<QInt8<Scale, Zero>, [1]>) { }",
                "quantized dtypes are reserved",
            ),
            (
                "fn f(x: Tensor<Float32, [1], device = NCHW>) { }",
                "memory layout types are reserved",
            ),
            (
                "fn f(x: Gradient<Float32>) { }",
                "training and autodiff types are reserved",
            ),
            (
                "fn f(x: PeakMemory) { }",
                "peak-memory deployment constraints are reserved",
            ),
            (
                "fn f<B: Dim>(x: Tensor<Float32, [B <= 64]>) { }",
                "dimension inequalities and range constraints are reserved",
            ),
        ];
        for (source, expected) in cases {
            let messages = tensor_err(source, ParseMode::Program);
            assert!(
                messages.iter().any(|message| message.contains(expected)),
                "{messages:?}"
            );
        }
    }

    #[test]
    fn method_turbofish_preserves_refine_shape_arguments() {
        let dump = tensor_dump(
            "fn f(x: TensorAny) { let y = x.refine::<UInt8, [B, 3]>(); }",
            ParseMode::Program,
        );
        assert!(dump.contains("field refine turbofish(2)"), "{dump}");
        assert!(dump.contains("dim-var B"), "{dump}");
    }

    #[test]
    fn core_array_slice_generic_args_unchanged_in_both_modes() {
        // Core array/slice types as generic args must parse identically with
        // and without the extension (no regression from shape interception).
        parse_ok("let a: Vec<[Int32; 4]>;", ParseMode::Snippet);
        tensor_ok("let a: Vec<[Int32; 4]>;", ParseMode::Snippet);
        parse_ok("let b: Vec<[Int32]>;", ParseMode::Snippet);
        // Codex P1: a single-element type slice must survive tensor mode too.
        tensor_ok("let b: Vec<[Int32]>;", ParseMode::Snippet);
        tensor_ok("let c: Vec<[&Int32]>;", ParseMode::Snippet);
        // Bare generic and nominal type names are ambiguous with symbolic
        // dimensions. Outside `Tensor` they must remain Core slice types.
        let generic = tensor_dump(
            "struct Holder<T> { value: T } fn take<T>(x: Holder<[T]>) {}",
            ParseMode::Program,
        );
        assert!(generic.contains("type-slice"), "{generic}");
        assert!(!generic.contains("dim-var T"), "{generic}");
        let nominal = tensor_dump(
            "struct Foo {} struct Holder<T> { value: T } fn take(x: Holder<[Foo]>) {}",
            ParseMode::Program,
        );
        assert!(nominal.contains("type-slice"), "{nominal}");
        assert!(!nominal.contains("dim-var Foo"), "{nominal}");
        let grouped = tensor_dump(
            "struct Holder<T> { value: T } fn take<T>(x: Holder<[(T)]>) {}",
            ParseMode::Program,
        );
        assert!(grouped.contains("type-slice"), "{grouped}");
        // A single symbolic dim is still a rank-1 shape under the extension.
        let d = tensor_dump("let t: Tensor<Float32, [N]>;", ParseMode::Snippet);
        assert!(d.contains("dim-var N"), "{d}");
    }

    #[test]
    fn recovery_does_not_skip_following_model() {
        // Codex P2: malformed top-level input before a valid model must not
        // swallow the model; the model still parses into an item.
        let src = "@@@ garbage\nmodel M<B: Dim> { input x: Tensor<Float32, [B]>; \
                   output y: Tensor<Float32, [B]>; }";
        let file = SourceFile::new("test.stark", src.to_string());
        let (ast, diags) =
            parse_with_options(&file, ParseMode::Program, LanguageOptions::with_tensor());
        assert!(!diags.is_empty(), "expected a diagnostic for the garbage");
        let has_model = ast
            .items
            .iter()
            .any(|it| matches!(it.kind, ItemKind::Model(_)));
        assert!(has_model, "the model after garbage should still be parsed");
    }

    #[test]
    fn model_is_ordinary_identifier_in_expression_position() {
        // `model` followed by non-identifier stays a plain identifier.
        parse_ok("let model = 5; let z = model + 1;", ParseMode::Snippet);
        tensor_ok("let model = 5; let z = model + 1;", ParseMode::Snippet);
    }

    // ----- types -----

    #[test]
    fn types_roundtrip() {
        parse_ok("let a: Int32;", ParseMode::Snippet);
        parse_ok("let b: Vec<Vec<Int32>>;", ParseMode::Snippet);
        parse_ok("let c: [Int32; 5];", ParseMode::Snippet);
        parse_ok("let d: &[Int32];", ParseMode::Snippet);
        parse_ok("let e: (Int32, String);", ParseMode::Snippet);
        parse_ok("let f: (Int32,);", ParseMode::Snippet);
        parse_ok("let g: ();", ParseMode::Snippet);
        parse_ok("let h: &mut String;", ParseMode::Snippet);
        parse_ok("let i: fn(Int32) -> Bool;", ParseMode::Snippet);
        parse_ok("let j: fn();", ParseMode::Snippet);
        parse_ok("let k: Iterator<Item = Int32>;", ParseMode::Snippet);
    }

    #[test]
    fn paren_type_is_grouping_not_tuple() {
        let d = dump_of("let x: (Int32);", ParseMode::Snippet);
        assert!(d.contains("type-primitive Int32"), "{d}");
        assert!(!d.contains("type-tuple"), "{d}");
        let d = dump_of("let x: (Int32,);", ParseMode::Snippet);
        assert!(d.contains("type-tuple"), "{d}");
    }

    #[test]
    fn shr_split_in_generics() {
        parse_ok("let m: Vec<Vec<Int32>> = Vec::new();", ParseMode::Snippet);
        parse_ok(
            "let m: Vec<Vec<Vec<Int32>>> = Vec::new();",
            ParseMode::Snippet,
        );
    }

    #[test]
    fn shr_eq_splits_in_generics_with_no_space_before_assign() {
        // WP-C1.1: the `ShrEq` -> `GtEq` -> `Eq` split chain in eat_gt() (parser.rs) had no
        // test. `Foo<Bar<T>>=y` with no space lexes the trailing `>>=` as one ShrEq token; two
        // generic closes must peel it down to Eq for the assignment.
        parse_ok("let x: Vec<Vec<Int32>>=Vec::new();", ParseMode::Snippet);
    }

    #[test]
    fn bare_shift_expression_vs_generic_close() {
        // WP-C1.1: no prior test parsed `a >> b` as an actual binary shift expression to
        // contrast against the generic-closing case above -- the parser disambiguates by parse
        // position (eat_gt is only called from generic-argument sites), but that structural
        // guarantee had zero test evidence.
        let d = dump_of("let z = a >> b;", ParseMode::Snippet);
        assert!(
            d.contains("binary Shr"),
            "expected a real Shr binary expression, got:\n{d}"
        );
        parse_ok("let m: Vec<Vec<Int32>> = a >> b;", ParseMode::Snippet);
    }

    // ----- expressions -----

    #[test]
    fn precedence_shapes() {
        // 1 + 2 * 3 → Add(1, Mul(2,3)): Add is the outer (earlier) node.
        let d = dump_of("let x = 1 + 2 * 3;", ParseMode::Snippet);
        assert!(
            d.find("binary Add").unwrap() < d.find("binary Mul").unwrap(),
            "{d}"
        );
        // -x ** 2 → Pow(Neg(x), 2): unary binds tighter than `**`.
        let d = dump_of("let y = -x ** 2;", ParseMode::Snippet);
        assert!(
            d.find("binary Pow").unwrap() < d.find("unary Neg").unwrap(),
            "{d}"
        );
        // 2 ** 3 ** 2 → right-assoc.
        let d = dump_of("let z = 2 ** 3 ** 2;", ParseMode::Snippet);
        assert_eq!(d.matches("binary Pow").count(), 2, "{d}");
        // a || b && c → Or(a, And(b, c)).
        let d = dump_of("let w = a || b && c;", ParseMode::Snippet);
        assert!(
            d.find("binary Or").unwrap() < d.find("binary And").unwrap(),
            "{d}"
        );
        // shift vs additive: a << b + c → Shl(a, Add(b, c)).
        let d = dump_of("let s = a << b + c;", ParseMode::Snippet);
        assert!(
            d.find("binary Shl").unwrap() < d.find("binary Add").unwrap(),
            "{d}"
        );
    }

    #[test]
    fn cast_chain_and_range() {
        parse_ok("let x = y as Int64 as Float64;", ParseMode::Snippet);
        parse_ok("let r = 0..10;", ParseMode::Snippet);
        parse_ok("let r = 0..=9;", ParseMode::Snippet);
        parse_ok("let p = &v[1..4];", ParseMode::Snippet);
    }

    #[test]
    fn chained_comparison_is_syntax_error() {
        let msgs = parse_err("let x = a < b < c;", ParseMode::Snippet);
        assert!(
            msgs.iter()
                .any(|m| m.contains("comparison operators cannot be chained")),
            "{msgs:?}"
        );
        let msgs = parse_err("let x = a == b == c;", ParseMode::Snippet);
        assert!(
            msgs.iter()
                .any(|m| m.contains("comparison operators cannot be chained")),
            "{msgs:?}"
        );
        // Relational binds tighter than equality, so this parses without
        // chaining: (a<b) == (c>d).
        parse_ok("let x = a < b == c > d;", ParseMode::Snippet);
    }

    #[test]
    fn chained_range_is_syntax_error() {
        let msgs = parse_err("let x = 0..1..2;", ParseMode::Snippet);
        assert!(
            msgs.iter()
                .any(|m| m.contains("range operators cannot be chained")),
            "{msgs:?}"
        );
    }

    #[test]
    fn assignment_right_assoc_and_compound() {
        let d = dump_of("a = b = c;", ParseMode::Snippet);
        assert_eq!(d.matches("assign Assign").count(), 2, "{d}");
        parse_ok("x += 1;", ParseMode::Snippet);
        parse_ok("x **= 2;", ParseMode::Snippet);
        parse_ok("x >>= 1;", ParseMode::Snippet);
        parse_ok("*borrowed = 43;", ParseMode::Snippet);
        parse_ok("p.x = 10;", ParseMode::Snippet);
        parse_ok("arr[0] = 1;", ParseMode::Snippet);
    }

    #[test]
    fn postfix_chains() {
        parse_ok("let n = v.len();", ParseMode::Snippet);
        parse_ok("let c = map.get(&key)?.clone();", ParseMode::Snippet);
        parse_ok("let e = arr[i][j];", ParseMode::Snippet);
        parse_ok("let t = pair.0;", ParseMode::Snippet);
        parse_ok("let m = 42.abs();", ParseMode::Snippet);
        parse_ok("let s = String::from(\"hi\").len();", ParseMode::Snippet);
    }

    #[test]
    fn nested_tuple_field_float_split() {
        let d = dump_of("let x = pair.0.1;", ParseMode::Snippet);
        assert!(d.contains("tuple-field 1"), "{d}");
        assert!(d.contains("tuple-field 0"), "{d}");
    }

    #[test]
    fn turbofish() {
        parse_ok("let s = size_of::<Int32>();", ParseMode::Snippet);
        parse_ok("let s = align_of::<Vec<Int32>>();", ParseMode::Snippet);
    }

    #[test]
    fn literals_and_collections() {
        parse_ok("let u = ();", ParseMode::Snippet);
        parse_ok("let s = (42,);", ParseMode::Snippet);
        parse_ok("let p = (42, \"hello\");", ParseMode::Snippet);
        parse_ok("let a = [1, 2, 3];", ParseMode::Snippet);
        parse_ok("let z = [0; 1000];", ParseMode::Snippet);
        parse_ok("let e: [Int32; 0] = [];", ParseMode::Snippet);
    }

    #[test]
    fn struct_literals_and_restriction() {
        parse_ok("let p = Point { x: 1, y: 2 };", ParseMode::Snippet);
        parse_ok("let p = Point { x, y };", ParseMode::Snippet);
        // In these positions the `{` must start the block, not a literal.
        parse_ok("if config { do_thing(); }", ParseMode::Snippet);
        parse_ok(
            "if (Config { verbose: true }).verbose { do_thing(); }",
            ParseMode::Snippet,
        );
        parse_ok("while running { tick(); }", ParseMode::Snippet);
        parse_ok("for i in 0..10 { println(i.fmt()); }", ParseMode::Snippet);
        parse_ok("match c { Color::Red => 1, _ => 0 }", ParseMode::Snippet);
    }

    #[test]
    fn if_else_chains_and_match() {
        parse_ok(
            "let g = if s >= 90 { \"A\" } else if s >= 80 { \"B\" } else { \"C\" };",
            ParseMode::Snippet,
        );
        parse_ok(
            "match x { 0 => \"zero\", Option::Some(n) => \"some\", Point { x, y: _ } => \"pt\", (a, b) => \"pair\", [f] => \"one\", _ => \"other\" }",
            ParseMode::Snippet,
        );
        // No trailing comma on the last arm.
        parse_ok(
            "match c { Color::Red => \"red\", Color::Green => \"green\" }",
            ParseMode::Snippet,
        );
    }

    // ----- statements/blocks -----

    #[test]
    fn let_forms() {
        parse_ok("let x: Int32;", ParseMode::Snippet);
        parse_ok("let mut y: Int32;", ParseMode::Snippet);
        parse_ok("let x: Int32 = 42;", ParseMode::Snippet);
        parse_ok("let x = 42;", ParseMode::Snippet);
        parse_ok("let mut z: Int32 = 1;", ParseMode::Snippet);
        parse_ok("let mut w = 1;", ParseMode::Snippet);
        let msgs = parse_err("let x;", ParseMode::Snippet);
        assert!(
            msgs.iter()
                .any(|m| m.contains("type annotation or an initializer")),
            "{msgs:?}"
        );
    }

    #[test]
    fn block_formed_statement_without_semicolon() {
        // Mid-block `if` without `;` followed by another statement
        // (the 04-Semantic-Analysis__15 shape).
        parse_ok(
            "fn f(c: Bool) { let mut z = 0; if c { z = 42; } print(z.fmt()); }",
            ParseMode::Program,
        );
        // Trailing block-formed expression is the block's value.
        let d = dump_of(
            "fn max(a: Int32, b: Int32) -> Int32 { if a > b { a } else { b } }",
            ParseMode::Program,
        );
        assert!(d.contains("tail"), "{d}");
        // Bare block statement.
        parse_ok("{ let x = 1; } let y = 2;", ParseMode::Snippet);
    }

    #[test]
    fn return_break_continue() {
        parse_ok("fn f() { return; }", ParseMode::Program);
        parse_ok("fn f() -> Int32 { return 42; }", ParseMode::Program);
        parse_ok("fn f() { loop { break; } }", ParseMode::Program);
        parse_ok("fn f() { loop { continue; } }", ParseMode::Program);
        parse_ok("fn f() { while true { break; } }", ParseMode::Program);
    }

    // ----- items -----

    #[test]
    fn item_kinds() {
        parse_ok(
            "fn add(a: Int32, b: Int32) -> Int32 { a + b }",
            ParseMode::Program,
        );
        parse_ok(
            "fn greet(name: &str) { println(name); }",
            ParseMode::Program,
        );
        parse_ok("fn never() -> ! { panic(\"no\") }", ParseMode::Program);
        parse_ok(
            "struct Point { x: Float64, y: Float64 }",
            ParseMode::Program,
        );
        parse_ok(
            "struct P { name: String, pub email: String }",
            ParseMode::Program,
        );
        parse_ok("pub struct Empty { }", ParseMode::Program);
        parse_ok("enum Color { Red, Green, Blue }", ParseMode::Program);
        parse_ok("enum Option<T> { Some(T), None }", ParseMode::Program);
        parse_ok(
            "enum Shape { Circle { r: Float64 }, Dot }",
            ParseMode::Program,
        );
        parse_ok("const MAX: Int32 = 1000;", ParseMode::Program);
        parse_ok("type Age = Int32;", ParseMode::Program);
        parse_ok("type Pair<T> = (T, T);", ParseMode::Program);
        // DEV-036 (WP-C2.12): this line checks only that `mod math;` (an external,
        // backing-file module declaration) parses as a syntactically valid item -- it has never
        // cared whether a real `math.stark` exists. It used to pass incidentally because
        // `parse_ok`'s bare `SourceFile` is named `"test.stark"`, one of the three filenames the
        // (now-removed) module-loader bypass matched. Uses the explicit harness-only opt-in
        // directly so this syntax-shape assertion no longer depends on that removed mechanism.
        {
            let file = SourceFile::new("test.stark", "mod math;".to_string());
            let (_ast, diags) =
                parse_project_allowing_missing_modules(&file, LanguageOptions::CORE);
            assert!(
                diags.is_empty(),
                "unexpected diagnostics for \"mod math;\": {:?}",
                diags.iter().map(|d| &d.message).collect::<Vec<_>>()
            );
        }
        parse_ok(
            "mod inline { pub fn add(a: Int32, b: Int32) -> Int32 { a + b } }",
            ParseMode::Program,
        );
    }

    #[test]
    fn traits_and_impls() {
        parse_ok(
            "trait Eq { fn eq(&self, other: &Self) -> Bool; }",
            ParseMode::Program,
        );
        parse_ok(
            "trait Iterator { type Item; fn next(&mut self) -> Option<Self::Item>; }",
            ParseMode::Program,
        );
        parse_ok(
            "trait Greet { fn hello(&self) -> String { String::from(\"hi\") } }",
            ParseMode::Program,
        );
        parse_ok(
            "impl Eq for Point { fn eq(&self, other: &Point) -> Bool { self.x == other.x } }",
            ParseMode::Program,
        );
        parse_ok(
            "impl Iterator for Counter { type Item = Int32; fn next(&mut self) -> Option<Int32> { None } }",
            ParseMode::Program,
        );
        parse_ok(
            "impl<T: Copy> Clone for T { fn clone(&self) -> T { *self } }",
            ParseMode::Program,
        );
        parse_ok("impl Copy for Int8 { }", ParseMode::Program);
        parse_ok(
            "impl String { fn shout(&self) -> String { self.to_uppercase() } }",
            ParseMode::Program,
        );
        parse_ok(
            "fn max<T: Ord>(a: T, b: T) -> T { if a > b { a } else { b } }",
            ParseMode::Program,
        );
        parse_ok("fn both<T: Eq + Ord>(a: T) -> T { a }", ParseMode::Program);
        parse_ok(
            "fn collect_all<I: Iterator<Item = Int32>>(it: I) -> Vec<Int32> { Vec::new() }",
            ParseMode::Program,
        );
    }

    #[test]
    fn receiver_only_in_impl_or_trait() {
        let msgs = parse_err("fn free(self) { }", ParseMode::Program);
        assert!(
            msgs.iter().any(|m| m.contains("receiver is only valid")),
            "{msgs:?}"
        );
    }

    #[test]
    fn use_trees() {
        parse_ok("use crate::utils::math::add;", ParseMode::Program);
        parse_ok("use super::config;", ParseMode::Program);
        parse_ok("use super::super::x;", ParseMode::Program);
        parse_ok("use crate::utils::{math, io};", ParseMode::Program);
        parse_ok("use crate::utils::math as m;", ParseMode::Program);
        parse_ok("use crate::utils::*;", ParseMode::Program);
        parse_ok("use std::io;", ParseMode::Program);
    }

    #[test]
    fn primitive_keyword_as_path_head() {
        parse_ok("let s = String::from(\"hello\");", ParseMode::Snippet);
        parse_ok("let owned: String = String::from(s);", ParseMode::Snippet);
    }

    #[test]
    fn self_and_segment_position_rules() {
        let msgs = parse_err("fn f() -> Self { }", ParseMode::Program);
        assert!(
            msgs.iter()
                .any(|m| m.contains("`Self` is only valid inside")),
            "{msgs:?}"
        );
        let msgs = parse_err("use a::crate::b;", ParseMode::Program);
        assert!(
            msgs.iter()
                .any(|m| m.contains("only valid as the first segment")),
            "{msgs:?}"
        );
    }

    #[test]
    fn reserved_word_diagnostic() {
        let msgs = parse_err("let x = async;", ParseMode::Snippet);
        assert!(
            msgs.iter().any(|m| m.contains("reserved for future use")),
            "{msgs:?}"
        );
    }

    #[test]
    fn reserved_word_diagnostic_in_non_expression_positions() {
        // WP-C1.1: reserved_word_diagnostic above only covers expression position; confirm the
        // same rejection fires in parameter-name and struct-field-name positions too.
        for (src, where_) in [
            ("fn f(unsafe: Int32) -> Int32 { unsafe }", "parameter name"),
            ("struct S { macro: Int32 }", "struct field name"),
        ] {
            let msgs = parse_err(src, ParseMode::Snippet);
            assert!(
                msgs.iter().any(|m| m.contains("reserved for future use")),
                "reserved word in {where_} position should be rejected, got {msgs:?}"
            );
        }
    }

    #[test]
    fn error_recovery_continues() {
        // The bad first statement must not hide the bad third one.
        let file = SourceFile::new("test.stark", "let = 1;\nlet y = 2;\nlet z = ;".to_string());
        let (ast, diags) = parse(&file, ParseMode::Snippet);
        assert!(
            diags.len() >= 2,
            "{:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        let Root::Snippet { stmts, .. } = &ast.root else {
            panic!()
        };
        assert!(stmts.len() >= 2);
    }

    #[test]
    fn program_mode_rejects_bare_statements() {
        let msgs = parse_err("let x = 1;", ParseMode::Program);
        assert!(
            msgs.iter().any(|m| m.contains("expected an item")),
            "{msgs:?}"
        );
    }

    #[test]
    fn items_in_blocks_rejected() {
        let msgs = parse_err("fn outer() { fn inner() { } }", ParseMode::Program);
        assert!(
            msgs.iter()
                .any(|m| m.contains("items are not allowed inside blocks")),
            "{msgs:?}"
        );
    }

    #[test]
    fn snippet_mode_mixes_items_and_statements() {
        parse_ok(
            "fn take(s: String) { }\nlet s = String::from(\"x\");\ntake(s);",
            ParseMode::Snippet,
        );
        parse_ok("let x = 42; const MAX: Int32 = 10; x", ParseMode::Snippet);
    }

    #[test]
    fn snippet_tail_expression() {
        let file = SourceFile::new("t.stark", "let x = 1;\nx + 1".to_string());
        let (ast, diags) = parse(&file, ParseMode::Snippet);
        assert!(diags.is_empty());
        let Root::Snippet { tail, .. } = &ast.root else {
            panic!()
        };
        assert!(tail.is_some());
    }

    /// DEV-036 (WP-C2.12): ordinary `parse`/`parse_with_options`/`parse_project` must still
    /// report a missing module file for a bare in-memory `SourceFile` regardless of its name --
    /// there is no longer any filename/path string that suppresses this diagnostic on the
    /// ordinary path. `"test.stark"` in particular used to be one of the three bypassing names.
    #[test]
    fn ordinary_parse_reports_missing_module_even_for_bare_test_stark_name() {
        let file = SourceFile::new(
            "test.stark",
            "mod does_not_exist;\nfn main() {}".to_string(),
        );
        let (_ast, diags) = parse(&file, ParseMode::Program);
        assert!(
            diags.iter().any(|d| d.code.as_deref() == Some("E0208")),
            "expected E0208 even for a bare SourceFile literally named \"test.stark\", got {:?}",
            diags
        );
    }

    /// DEV-036 (WP-C2.12): the explicit, harness-only opt-in
    /// (`parse_project_allowing_missing_modules`) still suppresses the diagnostic when a caller
    /// deliberately asks for it -- confirming the positive case survived removing the
    /// path-string-match mechanism, independent of the spec-fixture corpus (which could be
    /// re-triaged later).
    #[test]
    fn allowing_missing_modules_suppresses_the_diagnostic_when_explicitly_requested() {
        let file = SourceFile::new(
            "test.stark",
            "mod does_not_exist;\nfn main() {}".to_string(),
        );
        let (_ast, diags) = parse_project_allowing_missing_modules(&file, LanguageOptions::CORE);
        assert!(
            !diags.iter().any(|d| d.code.as_deref() == Some("E0208")),
            "explicit opt-in must suppress the missing-module diagnostic, got {:?}",
            diags
        );
    }
}
