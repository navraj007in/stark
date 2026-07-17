//! Doc comment extraction: walks a parsed [`Ast`], associating each public
//! item with its immediately-preceding `///` comment run.
//!
//! There is no doc-comment storage in the AST/HIR (comments aren't part of
//! the language's semantic model at all — see
//! `starkc/docs/PHASE8_GRAMMAR_GAPS.md`'s WP8.2 entry). This mirrors the
//! formatter's own approach: reuse `lexer::tokenize_with_comments`'s
//! separately-collected trivia and re-associate it with AST nodes by
//! source position, rather than changing the grammar.

use crate::ast::{self, Ast, ItemKind, Vis};
use crate::lexer::{tokenize, Comment, CommentKind, Token, TokenKind};
use crate::source::{SourceFile, Span};

/// One documented item: a top-level `pub` item, or a `pub` member (field,
/// variant, method, associated type) of one.
pub struct DocItem {
    pub name: String,
    /// `pub mod` path this item was found under, joined by `::`; empty for
    /// items declared directly at the package's top level.
    pub module_path: String,
    pub kind: ItemDocKind,
    /// Concatenated, dedented `///` text immediately preceding the item
    /// (each line's leading `/// ` stripped); empty if undocumented.
    pub doc: String,
    /// Source-text signature (e.g. `fn add(a: Int32, b: Int32) -> Int32`),
    /// sliced from source and whitespace-normalized — not re-derived
    /// through the formatter, so it may not be in canonical `stark fmt`
    /// form, but it is always exactly what the source says.
    pub signature: String,
    pub span: Span,
    pub members: Vec<DocItem>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ItemDocKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Const,
    TypeAlias,
    Model,
    Field,
    Variant,
    Method,
    AssocType,
    Port,
}

impl ItemDocKind {
    pub fn label(self) -> &'static str {
        match self {
            ItemDocKind::Fn => "fn",
            ItemDocKind::Struct => "struct",
            ItemDocKind::Enum => "enum",
            ItemDocKind::Trait => "trait",
            ItemDocKind::Const => "const",
            ItemDocKind::TypeAlias => "type",
            ItemDocKind::Model => "model",
            ItemDocKind::Field => "field",
            ItemDocKind::Variant => "variant",
            ItemDocKind::Method => "method",
            ItemDocKind::AssocType => "assoc type",
            ItemDocKind::Port => "port",
        }
    }

    /// Top-level kinds get their own page; member kinds are rendered
    /// inline on their owning item's page.
    pub fn is_page_level(self) -> bool {
        matches!(
            self,
            ItemDocKind::Fn
                | ItemDocKind::Struct
                | ItemDocKind::Enum
                | ItemDocKind::Trait
                | ItemDocKind::Const
                | ItemDocKind::TypeAlias
                | ItemDocKind::Model
        )
    }
}

/// A `stark` fenced code block found inside a doc comment, kept for
/// compile validation (`doc_gen::validate_examples`).
pub struct DocExample {
    /// Name of the enclosing item, for error reporting.
    pub owner: String,
    pub code: String,
}

/// Read-only context threaded through extraction: the parsed tree, its
/// source, and its comment/token trivia (both pre-computed once per file
/// rather than re-derived per item).
struct Ctx<'a> {
    ast: &'a Ast,
    file: &'a SourceFile,
    comments: &'a [Comment],
    tokens: &'a [Token],
}

/// Extract every public top-level item (recursing into `pub mod`) from
/// `ast`, attaching doc comments from `comments` by position, and merging
/// `impl TypeName { ... }` blocks' public methods into the matching
/// struct/enum's members.
pub fn extract(ast: &Ast, file: &SourceFile, comments: &[Comment]) -> Vec<DocItem> {
    let (tokens, _) = tokenize(file);
    let ctx = Ctx {
        ast,
        file,
        comments,
        tokens: &tokens,
    };
    let mut items = Vec::new();
    let mut impls: Vec<(String, Vec<DocItem>)> = Vec::new();
    if let ast::Root::Program(ids) = &ast.root {
        let mut mod_path = Vec::new();
        collect_items(&ctx, ids, &mut mod_path, &mut items, &mut impls);
    }
    for (type_name, methods) in impls {
        if let Some(owner) = items
            .iter_mut()
            .find(|i| i.name == type_name && i.kind.is_page_level())
        {
            owner.members.extend(methods);
        }
    }
    items
}

/// Pull every `stark`-fenced example out of `items`' doc comments,
/// recursing into members.
pub fn collect_examples(items: &[DocItem]) -> Vec<DocExample> {
    let mut out = Vec::new();
    fn walk(item: &DocItem, out: &mut Vec<DocExample>) {
        for code in extract_stark_fences(&item.doc) {
            out.push(DocExample {
                owner: item.name.clone(),
                code,
            });
        }
        for member in &item.members {
            walk(member, out);
        }
    }
    for item in items {
        walk(item, &mut out);
    }
    out
}

/// Find ` ```stark ... ``` ` fenced blocks in raw doc markdown text.
pub fn extract_stark_fences(doc: &str) -> Vec<String> {
    let mut fences = Vec::new();
    let mut lines = doc.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        if let Some(lang) = trimmed.strip_prefix("```") {
            if lang.trim() == "stark" {
                let mut body = String::new();
                for inner in lines.by_ref() {
                    if inner.trim_start().starts_with("```") {
                        break;
                    }
                    body.push_str(inner);
                    body.push('\n');
                }
                fences.push(body);
            }
        }
    }
    fences
}

fn collect_items(
    ctx: &Ctx,
    ids: &[ast::ItemId],
    mod_path: &mut Vec<String>,
    out: &mut Vec<DocItem>,
    impls: &mut Vec<(String, Vec<DocItem>)>,
) {
    for &id in ids {
        let node = ctx.ast.item(id);

        if let ItemKind::Mod {
            name,
            items: Some(inner),
        } = &node.kind
        {
            if node.vis == Some(Vis::Pub) {
                mod_path.push(text(ctx.file, *name).to_string());
                collect_items(ctx, inner, mod_path, out, impls);
                mod_path.pop();
            }
            continue;
        }

        if let ItemKind::Impl { self_ty, items, .. } = &node.kind {
            let self_name = self_type_name(ctx, *self_ty);
            if !self_name.is_empty() {
                let methods: Vec<DocItem> = items
                    .iter()
                    .filter_map(|item| impl_item_doc(ctx, item, mod_path))
                    .collect();
                if !methods.is_empty() {
                    impls.push((self_name, methods));
                }
            }
            continue;
        }

        if node.vis != Some(Vis::Pub) {
            continue;
        }
        if let Some(doc_item) = doc_item_for(ctx, id, mod_path) {
            out.push(doc_item);
        }
    }
}

/// The span from an item's start up to (but not including) its body's
/// opening `{` or, for `;`-terminated items, the `;` — i.e. everything a
/// reader would want in a "signature" line, not the full field/variant/
/// statement list. Scans real tokens (not raw bytes), so braces inside
/// string literals or comments can't be mistaken for the body start.
fn header_span(tokens: &[Token], item_span: Span) -> Span {
    let mut depth: i32 = 0;
    for t in tokens {
        if t.span.lo < item_span.lo {
            continue;
        }
        if t.span.lo >= item_span.hi {
            break;
        }
        match t.kind {
            TokenKind::LBrace | TokenKind::LParen | TokenKind::LBracket => depth += 1,
            TokenKind::RBrace | TokenKind::RParen | TokenKind::RBracket => depth -= 1,
            TokenKind::Semi if depth == 0 => {
                return Span::new(item_span.lo, t.span.lo);
            }
            _ => {}
        }
        // The first `{` at depth 0 (i.e. the one that took depth from 0
        // to 1) is the body/field-list opener — everything before it is
        // the header, including any `<...>` generics or `(...)` params
        // already accounted for by the depth tracking above.
        if t.kind == TokenKind::LBrace && depth == 1 {
            return Span::new(item_span.lo, t.span.lo);
        }
    }
    item_span
}

/// Placeholder written by every member/nested `DocItem` literal; replaced
/// by the owning top-level item's actual path right before the item is
/// returned (`fix_up_module_path`), so nested match arms don't each need
/// `mod_path` threaded through separately.
const PENDING_MODULE_PATH: &str = "\0pending\0";

fn doc_item_for(ctx: &Ctx, id: ast::ItemId, mod_path: &[String]) -> Option<DocItem> {
    let (ast, file, comments, tokens) = (ctx.ast, ctx.file, ctx.comments, ctx.tokens);
    let node = ast.item(id);
    let doc = leading_doc_comment(file, comments, node.span.lo);

    let fn_sig_span = match &node.kind {
        ItemKind::Fn(def) => Some(def.sig.span),
        _ => None,
    };

    let (name_span, kind, members) = match &node.kind {
        ItemKind::Fn(def) => (def.sig.name, ItemDocKind::Fn, Vec::new()),
        ItemKind::Struct { name, fields, .. } => {
            let members = fields
                .iter()
                .filter(|f| f.is_pub)
                .map(|f| {
                    let field_span = Span::new(f.name.lo, ast.ty(f.ty).span.hi);
                    DocItem {
                        name: text(file, f.name).to_string(),
                        module_path: PENDING_MODULE_PATH.to_string(),
                        kind: ItemDocKind::Field,
                        doc: leading_doc_comment(file, comments, f.name.lo),
                        signature: signature_text(file, field_span),
                        span: f.name,
                        members: Vec::new(),
                    }
                })
                .collect();
            (*name, ItemDocKind::Struct, members)
        }
        ItemKind::Enum { name, variants, .. } => {
            let members = variants
                .iter()
                .map(|v| DocItem {
                    name: text(file, v.name).to_string(),
                    module_path: PENDING_MODULE_PATH.to_string(),
                    kind: ItemDocKind::Variant,
                    doc: leading_doc_comment(file, comments, v.name.lo),
                    signature: variant_signature(ast, file, v),
                    span: v.name,
                    members: Vec::new(),
                })
                .collect();
            (*name, ItemDocKind::Enum, members)
        }
        ItemKind::Trait { name, items, .. } => {
            let members = items
                .iter()
                .filter_map(|item| trait_item_doc(file, comments, item))
                .collect();
            (*name, ItemDocKind::Trait, members)
        }
        ItemKind::Const { name, .. } => (*name, ItemDocKind::Const, Vec::new()),
        ItemKind::TypeAlias { name, .. } => (*name, ItemDocKind::TypeAlias, Vec::new()),
        ItemKind::Model(def) => {
            let members = def
                .ports
                .iter()
                .map(|p| DocItem {
                    name: text(file, p.name).to_string(),
                    module_path: PENDING_MODULE_PATH.to_string(),
                    kind: ItemDocKind::Port,
                    doc: leading_doc_comment(file, comments, p.span.lo),
                    signature: signature_text(file, p.span),
                    span: p.span,
                    members: Vec::new(),
                })
                .collect();
            (def.name, ItemDocKind::Model, members)
        }
        ItemKind::Use(_) | ItemKind::Mod { .. } | ItemKind::Impl { .. } => return None,
    };

    let path = mod_path.join("::");
    // `FnSig::span` starts at the `fn` keyword (excludes `pub`); every
    // other item kind's own span starts at `pub` itself, so only the `Fn`
    // case needs the prefix added back manually.
    let signature = match fn_sig_span {
        Some(sig_span) => {
            let vis_prefix = if node.vis == Some(Vis::Pub) {
                "pub "
            } else {
                ""
            };
            format!("{vis_prefix}{}", signature_text(file, sig_span))
        }
        None => signature_text(file, header_span(tokens, node.span)),
    };
    let mut item = DocItem {
        name: text(file, name_span).to_string(),
        module_path: path.clone(),
        kind,
        doc,
        signature,
        span: node.span,
        members,
    };
    for member in &mut item.members {
        member.module_path = path.clone();
    }
    Some(item)
}

fn trait_item_doc(
    file: &SourceFile,
    comments: &[Comment],
    item: &ast::TraitItem,
) -> Option<DocItem> {
    match item {
        ast::TraitItem::Method { sig, .. } => Some(DocItem {
            name: text(file, sig.name).to_string(),
            module_path: PENDING_MODULE_PATH.to_string(),
            kind: ItemDocKind::Method,
            doc: leading_doc_comment(file, comments, sig.span.lo),
            signature: signature_text(file, sig.span),
            span: sig.span,
            members: Vec::new(),
        }),
        ast::TraitItem::AssocType { name } => Some(DocItem {
            name: text(file, *name).to_string(),
            module_path: PENDING_MODULE_PATH.to_string(),
            kind: ItemDocKind::AssocType,
            doc: leading_doc_comment(file, comments, name.lo),
            signature: text(file, *name).to_string(),
            span: *name,
            members: Vec::new(),
        }),
    }
}

fn impl_item_doc(ctx: &Ctx, item: &ast::ImplItem, mod_path: &[String]) -> Option<DocItem> {
    let (file, comments) = (ctx.file, ctx.comments);
    let path = mod_path.join("::");
    match item {
        ast::ImplItem::Fn { vis, def } => {
            if *vis != Some(Vis::Pub) {
                return None;
            }
            Some(DocItem {
                name: text(file, def.sig.name).to_string(),
                module_path: path,
                kind: ItemDocKind::Method,
                doc: leading_doc_comment(file, comments, def.sig.span.lo),
                // Only `pub` methods reach this point (checked above);
                // `FnSig::span` itself never includes the `pub` keyword.
                signature: format!("pub {}", signature_text(file, def.sig.span)),
                span: def.sig.span,
                members: Vec::new(),
            })
        }
        ast::ImplItem::AssocType { name, .. } => Some(DocItem {
            name: text(file, *name).to_string(),
            module_path: path,
            kind: ItemDocKind::AssocType,
            doc: leading_doc_comment(file, comments, name.lo),
            signature: text(file, *name).to_string(),
            span: *name,
            members: Vec::new(),
        }),
    }
}

fn variant_signature(ast: &Ast, file: &SourceFile, v: &ast::Variant) -> String {
    let name = text(file, v.name).to_string();
    match &v.kind {
        ast::VariantKind::Unit => name,
        ast::VariantKind::Tuple(tys) => {
            let parts: Vec<&str> = tys.iter().map(|&t| text(file, ast.ty(t).span)).collect();
            format!("{name}({})", parts.join(", "))
        }
        ast::VariantKind::Struct(fields) => {
            let parts: Vec<String> = fields
                .iter()
                .map(|f| format!("{}: {}", text(file, f.name), text(file, ast.ty(f.ty).span)))
                .collect();
            format!("{name} {{ {} }}", parts.join(", "))
        }
    }
}

fn self_type_name(ctx: &Ctx, ty: ast::TypeId) -> String {
    match &ctx.ast.ty(ty).kind {
        ast::TypeKind::Path { path, .. } => path
            .segments
            .last()
            .map(|s| text(ctx.file, s.span).to_string())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// Concatenate the contiguous run of `///` comments immediately preceding
/// `before_pos` (no blank line, no non-doc comment breaking the run),
/// stripping each line's `///` marker and at most one following space.
fn leading_doc_comment(file: &SourceFile, comments: &[Comment], before_pos: u32) -> String {
    let candidates: Vec<&Comment> = comments
        .iter()
        .filter(|c| c.span.hi <= before_pos && c.kind == CommentKind::LineDoc)
        .collect();

    // Walk backward from the item, requiring each comment to be on the
    // line immediately above the previous one we accepted (or immediately
    // above the item itself, for the last comment in the run).
    let mut contiguous: Vec<&Comment> = Vec::new();
    let mut expected_line = file.line_col(before_pos).0;
    for c in candidates.iter().rev() {
        let comment_line = file.line_col(c.span.lo).0;
        if comment_line + 1 == expected_line {
            contiguous.push(c);
            expected_line = comment_line;
        } else {
            break;
        }
    }
    contiguous.reverse();

    let mut out = String::new();
    for c in contiguous {
        let raw = &file.src[c.span.lo as usize..c.span.hi as usize];
        let body = raw.strip_prefix("///").unwrap_or(raw);
        let body = body.strip_prefix(' ').unwrap_or(body);
        out.push_str(body);
        out.push('\n');
    }
    out
}

fn text(file: &SourceFile, span: Span) -> &str {
    &file.src[span.lo as usize..span.hi as usize]
}

/// Slice `span`'s source text and collapse runs of whitespace into single
/// spaces, for a compact one-line-ish signature display.
fn signature_text(file: &SourceFile, span: Span) -> String {
    let raw = text(file, span);
    let mut out = String::new();
    let mut last_was_space = false;
    for ch in raw.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
            }
            last_was_space = true;
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    out.trim().to_string()
}
