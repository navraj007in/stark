//! AST-to-text renderer. Walks the parsed [`Ast`] (never the raw source
//! text) and produces canonical, idempotent output, re-attaching comment
//! trivia (`lexer::Comment`) by source position as it goes.
//!
//! Rules (see `PHASE8_PRODUCTION_TOOLING_PLAN.md` WP8.2):
//! - 4-space indent, no tabs.
//! - Soft-wrap delimited lists (params, args, fields, ...) at 100 columns:
//!   fits on one line -> flat, `, `-joined, no trailing comma; else -> one
//!   element per line, trailing comma, closing delimiter on its own line.
//! - Space around binary/assignment operators; space after `:`; space after
//!   `if`/`while`/`for`/`fn`/keyword before the following expression.
//! - `use` items are flattened (one leaf path per `use` statement) and the
//!   leaves of each original `use` sorted lexicographically.
//! - Comments: leading comments render on their own line(s) above the node
//!   they precede, blank-line-separated exactly when the source had a blank
//!   line; a comment on the same source line as the end of a node renders
//!   trailing that node. Comments in positions this printer does not
//!   specifically track (rare: inside an expression's interior) are still
//!   guaranteed to be emitted — see [`format`]'s end-of-file drain — never
//!   silently dropped, though they may be relocated to the nearest
//!   statement/item boundary.

use crate::ast::*;
use crate::lexer::{Comment, CommentKind};
use crate::source::{SourceFile, Span};

use super::comments::CommentStream;
use super::precedence::{self, Assoc, Position};

const MAX_WIDTH: usize = 100;
const INDENT_WIDTH: usize = 4;

/// Format `ast` (parsed from `file`) back to canonical source text.
/// `comments` is the trivia list from `lexer::tokenize_with_comments` for
/// the same file.
pub fn format(ast: &Ast, file: &SourceFile, comments: &[Comment]) -> String {
    let mut p = Printer {
        ast,
        file,
        comments: CommentStream::new(comments),
        out: String::new(),
        indent: 0,
        last_pos: 0,
    };
    match &ast.root {
        Root::Program(items) => p.item_seq(items, file.src.len() as u32),
        Root::Snippet { stmts, tail } => p.stmt_seq(stmts, *tail, file.src.len() as u32),
    }
    // No-loss safety net: anything the walk didn't specifically attach
    // (e.g. a comment inside an expression's interior) still gets emitted,
    // rather than silently dropped.
    let rest: Vec<Comment> = p.comments.take_rest().to_vec();
    for c in rest {
        p.blank_line_if_needed(c.span.lo);
        p.newline();
        p.write_comment(c);
        p.last_pos = c.span.hi;
    }
    if !p.out.ends_with('\n') && !p.out.is_empty() {
        p.out.push('\n');
    }
    p.out
}

struct Printer<'a> {
    ast: &'a Ast,
    file: &'a SourceFile,
    comments: CommentStream<'a>,
    out: String,
    indent: usize,
    /// Source position rendered up to so far; drives blank-line
    /// preservation and same-line trailing-comment attachment.
    last_pos: u32,
}

impl<'a> Printer<'a> {
    // ------------------------------------------------------------- basics --

    fn text(&self, span: Span) -> &'a str {
        &self.file.src[span.lo as usize..span.hi as usize]
    }

    fn line_of(&self, pos: u32) -> usize {
        self.file.line_col(pos).0
    }

    fn write(&mut self, s: &str) {
        self.out.push_str(s);
    }

    fn newline(&mut self) {
        if !self.out.is_empty() {
            self.out.push('\n');
        }
        for _ in 0..self.indent * INDENT_WIDTH {
            self.out.push(' ');
        }
    }

    fn col(&self) -> usize {
        match self.out.rfind('\n') {
            Some(i) => self.out[i + 1..].chars().count(),
            None => self.out.chars().count(),
        }
    }

    /// Render `f`'s writes into a scratch buffer instead of `self.out`, and
    /// return the result. `f` must not consume comments (list-item printers
    /// never do; comment attachment happens only at item/stmt/field/arm
    /// sequence level), so it is safe to discard this render and print the
    /// broken form for real afterward.
    fn measure_flat(&mut self, f: impl FnOnce(&mut Self)) -> String {
        let saved = std::mem::take(&mut self.out);
        f(self);
        std::mem::replace(&mut self.out, saved)
    }

    /// A `, `-joined delimited list of `n` elements, flat if it fits within
    /// [`MAX_WIDTH`] columns and contains no forced newline, else one
    /// element per line with a trailing comma. `{`/`}` lists (struct
    /// fields, struct literals, pattern struct fields) get an inner
    /// padding space in the flat form (`{ x: 1 }`, not `{x: 1}`); `(`/`[`
    /// lists do not.
    fn delimited_list(
        &mut self,
        open: &str,
        close: &str,
        n: usize,
        print_nth: impl Fn(&mut Self, usize),
    ) {
        if n == 0 {
            self.write(open);
            self.write(close);
            return;
        }
        let pad = if open == "{" { " " } else { "" };
        let start_col = self.col();
        let flat = self.measure_flat(|p| {
            p.write(open);
            p.write(pad);
            for i in 0..n {
                if i > 0 {
                    p.write(", ");
                }
                print_nth(p, i);
            }
            p.write(pad);
            p.write(close);
        });
        if !flat.contains('\n') && start_col + flat.chars().count() <= MAX_WIDTH {
            self.write(&flat);
        } else {
            self.write(open);
            self.indent += 1;
            for i in 0..n {
                self.newline();
                print_nth(self, i);
                self.write(",");
            }
            self.indent -= 1;
            self.newline();
            self.write(close);
        }
    }

    // ------------------------------------------------------ blank lines --

    fn blank_line_if_needed(&mut self, next_pos: u32) {
        if self.out.is_empty() {
            return;
        }
        if self.line_of(next_pos) > self.line_of(self.last_pos) + 1 {
            self.out.push('\n');
        }
    }

    // ----------------------------------------------------------- comments --

    fn write_comment(&mut self, c: Comment) {
        let raw = self.text(c.span);
        match c.kind {
            CommentKind::Line | CommentKind::LineDoc | CommentKind::LineInnerDoc => {
                let prefix = match c.kind {
                    CommentKind::LineDoc => "///",
                    CommentKind::LineInnerDoc => "//!",
                    _ => "//",
                };
                let body = raw.get(prefix.len()..).unwrap_or("").trim_end();
                if body.is_empty() {
                    self.write(prefix);
                } else if body.starts_with(char::is_whitespace) {
                    self.write(prefix);
                    self.write(body);
                } else {
                    self.write(prefix);
                    self.write(" ");
                    self.write(body);
                }
            }
            CommentKind::Block => {
                // Multi-line block comments are preserved verbatim
                // (continuation-line reindentation is not attempted); only
                // trailing whitespace on the whole trivia is trimmed.
                self.write(raw.trim_end());
            }
        }
    }

    /// Emit every unconsumed comment that starts before `before_pos`, each
    /// on its own line, preserving a single blank line wherever the source
    /// had one.
    fn emit_leading_comments(&mut self, before_pos: u32) {
        while let Some(c) = self.comments.take_before(before_pos) {
            self.blank_line_if_needed(c.span.lo);
            self.newline();
            self.write_comment(c);
            self.last_pos = c.span.hi;
        }
    }

    /// If the next unconsumed comment starts on the same source line as
    /// `self.last_pos`, consume and append it inline.
    fn emit_trailing_same_line_comment(&mut self) {
        let line = self.line_of(self.last_pos);
        if let Some(c) = self.comments.peek() {
            if self.line_of(c.span.lo) == line {
                self.comments.advance();
                self.write(" ");
                self.write_comment(c);
                self.last_pos = c.span.hi;
            }
        }
    }

    // -------------------------------------------------------------- paths --

    fn path(&mut self, path: &Path) {
        for (i, seg) in path.segments.iter().enumerate() {
            if i > 0 {
                self.write("::");
            }
            let t = self.text(seg.span);
            self.write(t);
        }
    }

    fn trait_ref(&mut self, t: &TraitRef) {
        self.path(&t.path);
        if let Some(args) = &t.args {
            self.generic_args(args);
        }
    }

    fn generic_args(&mut self, args: &GenericArgs) {
        self.write("<");
        for (i, a) in args.args.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            match a {
                GenericArg::Type(t) => self.ty(*t),
                GenericArg::Const(span) => {
                    let t = self.text(*span);
                    self.write(t);
                }
                GenericArg::Binding { name, ty } => {
                    let t = self.text(*name);
                    self.write(t);
                    self.write(" = ");
                    self.ty(*ty);
                }
                GenericArg::Shape(shape) => {
                    self.write("[");
                    for (j, d) in shape.dims.iter().enumerate() {
                        if j > 0 {
                            self.write(", ");
                        }
                        self.dim_expr(*d);
                    }
                    self.write("]");
                }
            }
        }
        self.write(">");
    }

    fn dim_expr(&mut self, id: DimId) {
        self.dim_expr_operand(id, 0, true);
    }

    fn dim_expr_operand(&mut self, id: DimId, parent_level: u8, is_lhs: bool) {
        let node = self.ast.dim(id);
        let (kind, span) = (&node.kind, node.span);
        let this_level = match kind {
            DimExprKind::Binary {
                op: DimBinOp::Mul, ..
            } => 1,
            DimExprKind::Binary { .. } => 0,
            _ => 2, // atoms never need parens
        };
        let needs_parens =
            this_level < parent_level || (this_level == parent_level && !is_lhs && this_level < 2);
        if needs_parens {
            self.write("(");
        }
        match kind {
            DimExprKind::Lit(s) | DimExprKind::Var(s) => {
                let t = self.text(*s);
                self.write(t);
            }
            DimExprKind::Binary { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                self.dim_expr_operand(lhs, this_level, true);
                self.write(" ");
                self.write(op.symbol());
                self.write(" ");
                self.dim_expr_operand(rhs, this_level, false);
            }
            DimExprKind::Error => {
                let t = self.text(span);
                self.write(t);
            }
        }
        if needs_parens {
            self.write(")");
        }
    }

    // ---------------------------------------------------------------- types --

    fn ty(&mut self, id: TypeId) {
        let node = self.ast.ty(id);
        match &node.kind {
            TypeKind::Primitive(p) => self.write(p.name()),
            TypeKind::Path { path, args } => {
                let path = path.clone();
                let args = args.clone();
                self.path(&path);
                if let Some(args) = &args {
                    self.generic_args(args);
                }
            }
            TypeKind::Array { elem, len } => {
                let (elem, len) = (*elem, *len);
                self.write("[");
                self.ty(elem);
                self.write("; ");
                let t = self.text(len);
                self.write(t);
                self.write("]");
            }
            TypeKind::Slice(elem) => {
                let elem = *elem;
                self.write("[");
                self.ty(elem);
                self.write("]");
            }
            TypeKind::Tuple(elems) => {
                let elems = elems.clone();
                if elems.len() == 1 {
                    self.write("(");
                    self.ty(elems[0]);
                    self.write(",)");
                } else {
                    self.delimited_list("(", ")", elems.len(), |p, i| p.ty(elems[i]));
                }
            }
            TypeKind::Ref { mutable, inner } => {
                let (mutable, inner) = (*mutable, *inner);
                self.write("&");
                if mutable {
                    self.write("mut ");
                }
                self.ty(inner);
            }
            TypeKind::Fn { params, ret } => {
                let (params, ret) = (params.clone(), *ret);
                self.write("fn");
                self.delimited_list("(", ")", params.len(), |p, i| p.ty(params[i]));
                if let Some(ret) = ret {
                    self.write(" -> ");
                    self.ty(ret);
                }
            }
            TypeKind::Never => self.write("!"),
            TypeKind::Error => {
                let t = self.text(node.span);
                self.write(t);
            }
        }
    }

    // -------------------------------------------------------------- patterns --

    fn pat(&mut self, id: PatId) {
        let node = self.ast.pat(id);
        match &node.kind {
            PatKind::Lit(_) => {
                let t = self.text(node.span);
                self.write(t);
            }
            PatKind::Wild => self.write("_"),
            PatKind::Binding(span) => {
                let t = self.text(*span);
                self.write(t);
            }
            PatKind::Path(path) => {
                let path = path.clone();
                self.path(&path);
            }
            PatKind::TupleVariant { path, pats } => {
                let (path, pats) = (path.clone(), pats.clone());
                self.path(&path);
                self.delimited_list("(", ")", pats.len(), |p, i| p.pat(pats[i]));
            }
            PatKind::Struct { path, fields } => {
                let (path, fields) = (path.clone(), fields.iter().collect::<Vec<_>>());
                self.path(&path);
                self.write(" ");
                self.delimited_list("{", "}", fields.len(), |p, i| {
                    let f = fields[i];
                    let t = p.text(f.name);
                    p.write(t);
                    if let Some(pat) = f.pat {
                        p.write(": ");
                        p.pat(pat);
                    }
                });
            }
            PatKind::Tuple(pats) => {
                let pats = pats.clone();
                self.delimited_list("(", ")", pats.len(), |p, i| p.pat(pats[i]));
            }
            PatKind::Array(pats) => {
                let pats = pats.clone();
                self.delimited_list("[", "]", pats.len(), |p, i| p.pat(pats[i]));
            }
        }
    }

    // ----------------------------------------------------------- expressions --

    /// Print `id` with no surrounding-context parenthesization concerns
    /// (statement tail, list element, etc.).
    fn expr(&mut self, id: ExprId) {
        self.expr_inner(id);
    }

    fn expr_operand(
        &mut self,
        id: ExprId,
        parent_level: precedence::Level,
        assoc: Assoc,
        position: Position,
    ) {
        if precedence::needs_parens(self.ast, id, parent_level, assoc, position) {
            self.write("(");
            self.expr_inner(id);
            self.write(")");
        } else {
            self.expr_inner(id);
        }
    }

    /// `if`/`while`/`for`/`match` condition position: guards against the
    /// struct-literal parse ambiguity (`02-Syntax-Grammar.md`).
    fn cond_expr(&mut self, id: ExprId) {
        if precedence::head_is_struct_lit(self.ast, id) {
            self.write("(");
            self.expr_inner(id);
            self.write(")");
        } else {
            self.expr_inner(id);
        }
    }

    fn expr_inner(&mut self, id: ExprId) {
        use precedence::*;
        let node = self.ast.expr(id);
        match &node.kind {
            ExprKind::Lit(_) => {
                let t = self.text(node.span);
                self.write(t);
            }
            ExprKind::Path { path, turbofish } => {
                let (path, turbofish) = (path.clone(), turbofish.clone());
                self.path(&path);
                if let Some(tf) = &turbofish {
                    self.write("::");
                    self.generic_args(tf);
                }
            }
            ExprKind::Unary { op, operand } => {
                let (op, operand) = (*op, *operand);
                self.write(unary_symbol(op));
                self.expr_operand(operand, UNARY, Assoc::Left, Position::Lhs);
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                let (lvl, assoc) = bin_op_level(op);
                self.expr_operand(lhs, lvl, assoc, Position::Lhs);
                self.write(" ");
                self.write(bin_op_symbol(op));
                self.write(" ");
                self.expr_operand(rhs, lvl, assoc, Position::Rhs);
            }
            ExprKind::Assign { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                self.expr_operand(lhs, ASSIGN, Assoc::Right, Position::Lhs);
                self.write(" ");
                self.write(assign_op_symbol(op));
                self.write(" ");
                self.expr_operand(rhs, ASSIGN, Assoc::Right, Position::Rhs);
            }
            ExprKind::Range { lo, hi, inclusive } => {
                let (lo, hi, inclusive) = (*lo, *hi, *inclusive);
                self.expr_operand(lo, RANGE, Assoc::None, Position::Lhs);
                self.write(if inclusive { "..=" } else { ".." });
                self.expr_operand(hi, RANGE, Assoc::None, Position::Rhs);
            }
            ExprKind::Cast { expr, ty } => {
                let (expr, ty) = (*expr, *ty);
                self.expr_operand(expr, CAST, Assoc::Left, Position::Lhs);
                self.write(" as ");
                self.ty(ty);
            }
            ExprKind::Call { callee, args } => {
                let (callee, args) = (*callee, args.clone());
                self.expr_operand(callee, POSTFIX, Assoc::Left, Position::Lhs);
                self.delimited_list("(", ")", args.len(), |p, i| p.expr(args[i]));
            }
            ExprKind::Field {
                base,
                name,
                turbofish,
            } => {
                let (base, name, turbofish) = (*base, *name, turbofish.clone());
                self.expr_operand(base, POSTFIX, Assoc::Left, Position::Lhs);
                self.write(".");
                let t = self.text(name);
                self.write(t);
                if let Some(tf) = &turbofish {
                    self.write("::");
                    self.generic_args(tf);
                }
            }
            ExprKind::TupleField { base, index } => {
                let (base, index) = (*base, *index);
                self.expr_operand(base, POSTFIX, Assoc::Left, Position::Lhs);
                self.write(".");
                let t = self.text(index);
                self.write(t);
            }
            ExprKind::Index { base, index } => {
                let (base, index) = (*base, *index);
                self.expr_operand(base, POSTFIX, Assoc::Left, Position::Lhs);
                self.write("[");
                self.expr(index);
                self.write("]");
            }
            ExprKind::Try(inner) => {
                let inner = *inner;
                self.expr_operand(inner, POSTFIX, Assoc::Left, Position::Lhs);
                self.write("?");
            }
            ExprKind::Tuple(elems) => {
                let elems = elems.clone();
                if elems.len() == 1 {
                    self.write("(");
                    self.expr(elems[0]);
                    self.write(",)");
                } else {
                    self.delimited_list("(", ")", elems.len(), |p, i| p.expr(elems[i]));
                }
            }
            ExprKind::Array(elems) => {
                let elems = elems.clone();
                self.delimited_list("[", "]", elems.len(), |p, i| p.expr(elems[i]));
            }
            ExprKind::Repeat { value, count } => {
                let (value, count) = (*value, *count);
                self.write("[");
                self.expr(value);
                self.write("; ");
                self.expr(count);
                self.write("]");
            }
            ExprKind::StructLit { path, fields } => {
                let (path, fields) = (path.clone(), fields.iter().collect::<Vec<_>>());
                self.path(&path);
                self.write(" ");
                self.delimited_list("{", "}", fields.len(), |p, i| {
                    let f = fields[i];
                    let t = p.text(f.name);
                    p.write(t);
                    if let Some(e) = f.expr {
                        p.write(": ");
                        p.expr(e);
                    }
                });
            }
            ExprKind::If { .. } => self.if_expr(id),
            ExprKind::Match { .. } => self.match_expr(id),
            ExprKind::Loop { body } => {
                let body = *body;
                self.write("loop ");
                self.block(body);
            }
            ExprKind::While { cond, body } => {
                let (cond, body) = (*cond, *body);
                self.write("while ");
                self.cond_expr(cond);
                self.write(" ");
                self.block(body);
            }
            ExprKind::For { var, iter, body } => {
                let (var, iter, body) = (*var, *iter, *body);
                self.write("for ");
                let t = self.text(var);
                self.write(t);
                self.write(" in ");
                self.cond_expr(iter);
                self.write(" ");
                self.block(body);
            }
            ExprKind::Block(b) => {
                let b = *b;
                self.block(b);
            }
            ExprKind::Error => {}
        }
    }

    fn if_expr(&mut self, id: ExprId) {
        let node = self.ast.expr(id);
        let ExprKind::If {
            cond,
            then_block,
            else_,
        } = &node.kind
        else {
            unreachable!()
        };
        let (cond, then_block, else_) = (*cond, *then_block, *else_);
        self.write("if ");
        self.cond_expr(cond);
        self.write(" ");
        self.block(then_block);
        if let Some(e) = else_ {
            self.write(" else ");
            match &self.ast.expr(e).kind {
                ExprKind::Block(b) => {
                    let b = *b;
                    self.block(b);
                }
                ExprKind::If { .. } => self.if_expr(e),
                _ => unreachable!("else_ is Block or If per ast.rs invariant"),
            }
        }
    }

    fn match_expr(&mut self, id: ExprId) {
        let node = self.ast.expr(id);
        let ExprKind::Match { scrutinee, arms } = &node.kind else {
            unreachable!()
        };
        let scrutinee = *scrutinee;
        let arms: &'a [MatchArm] = arms;
        let close_pos = node.span.hi;

        self.write("match ");
        self.cond_expr(scrutinee);
        self.write(" {");
        self.last_pos = node.span.lo;
        self.indent += 1;
        for arm in arms {
            let pat_lo = self.ast.pat(arm.pat).span.lo;
            self.emit_leading_comments(pat_lo);
            self.blank_line_if_needed(pat_lo);
            self.newline();
            self.pat(arm.pat);
            self.write(" => ");
            let blocky = matches!(
                self.ast.expr(arm.body).kind,
                ExprKind::Block(_)
                    | ExprKind::If { .. }
                    | ExprKind::Match { .. }
                    | ExprKind::Loop { .. }
                    | ExprKind::While { .. }
                    | ExprKind::For { .. }
            );
            self.expr(arm.body);
            self.last_pos = self.ast.expr(arm.body).span.hi;
            if !blocky {
                self.write(",");
            }
            self.emit_trailing_same_line_comment();
        }
        self.emit_leading_comments(close_pos);
        self.indent -= 1;
        self.newline();
        self.write("}");
        self.last_pos = close_pos;
    }

    // ------------------------------------------------------------- blocks --

    fn block(&mut self, id: BlockId) {
        let node = self.ast.block(id);
        let stmts: Vec<StmtId> = node
            .stmts
            .iter()
            .copied()
            .filter(|&s| !matches!(self.ast.stmt(s).kind, StmtKind::Empty))
            .collect();
        let tail = node.tail;
        let close_pos = node.span.hi;

        if stmts.is_empty()
            && tail.is_none()
            && self.comments.peek().is_none_or(|c| c.span.lo >= close_pos)
        {
            self.write("{}");
            self.last_pos = close_pos;
            return;
        }

        self.write("{");
        self.last_pos = node.span.lo;
        self.indent += 1;
        self.stmt_seq(&stmts, tail, close_pos);
        self.emit_leading_comments(close_pos);
        self.indent -= 1;
        self.newline();
        self.write("}");
        self.last_pos = close_pos;
    }

    /// Print a sequence of statements followed by an optional tail
    /// expression, with full leading/trailing comment attachment and
    /// blank-line preservation between elements. Used for block bodies and
    /// (with `Root::Snippet`) the harness's top-level statement sequence.
    fn stmt_seq(&mut self, stmts: &[StmtId], tail: Option<ExprId>, _end_pos: u32) {
        for &s in stmts {
            let lo = self.ast.stmt(s).span.lo;
            self.emit_leading_comments(lo);
            self.blank_line_if_needed(lo);
            self.newline();
            self.stmt(s);
            self.emit_trailing_same_line_comment();
        }
        if let Some(t) = tail {
            let lo = self.ast.expr(t).span.lo;
            self.emit_leading_comments(lo);
            self.blank_line_if_needed(lo);
            self.newline();
            self.expr(t);
            self.last_pos = self.ast.expr(t).span.hi;
            self.emit_trailing_same_line_comment();
        }
    }

    fn stmt(&mut self, id: StmtId) {
        let node = self.ast.stmt(id);
        let span = node.span;
        match &node.kind {
            StmtKind::Empty => {}
            StmtKind::Expr { expr, semi } => {
                let (expr, semi) = (*expr, *semi);
                self.expr(expr);
                if semi {
                    self.write(";");
                }
                self.last_pos = span.hi;
            }
            StmtKind::Let {
                mutable,
                name,
                ty,
                init,
            } => {
                let (mutable, name, ty, init) = (*mutable, *name, *ty, *init);
                self.write("let ");
                if mutable {
                    self.write("mut ");
                }
                let t = self.text(name);
                self.write(t);
                if let Some(ty) = ty {
                    self.write(": ");
                    self.ty(ty);
                }
                if let Some(init) = init {
                    self.write(" = ");
                    self.expr(init);
                }
                self.write(";");
                self.last_pos = span.hi;
            }
            StmtKind::Return(opt) => {
                let opt = *opt;
                self.write("return");
                if let Some(e) = opt {
                    self.write(" ");
                    self.expr(e);
                }
                self.write(";");
                self.last_pos = span.hi;
            }
            StmtKind::Break(opt) => {
                let opt = *opt;
                self.write("break");
                if let Some(e) = opt {
                    self.write(" ");
                    self.expr(e);
                }
                self.write(";");
                self.last_pos = span.hi;
            }
            StmtKind::Continue => {
                self.write("continue;");
                self.last_pos = span.hi;
            }
            StmtKind::Item(item_id) => {
                let item_id = *item_id;
                self.item(item_id);
                self.last_pos = span.hi;
            }
            StmtKind::Error => {
                self.last_pos = span.hi;
            }
        }
    }

    // -------------------------------------------------------------- items --

    fn item_seq(&mut self, items: &[ItemId], end_pos: u32) {
        for &id in items {
            let node = self.ast.item(id);
            let lo = item_leading_pos(node);
            self.emit_leading_comments(lo);
            self.blank_line_if_needed(lo);
            self.newline();
            self.item(id);
            self.emit_trailing_same_line_comment();
        }
        self.emit_leading_comments(end_pos);
    }

    fn item(&mut self, id: ItemId) {
        let node = self.ast.item(id);
        let vis = node.vis;
        let span = node.span;
        if let Some(v) = vis {
            self.write(match v {
                Vis::Pub => "pub ",
                Vis::Priv => "priv ",
            });
        }
        match &node.kind {
            ItemKind::Fn(def) => {
                let sig = def.sig.clone();
                let body = def.body;
                self.fn_sig(&sig);
                self.write(" ");
                self.block(body);
            }
            ItemKind::Struct {
                name,
                generics,
                fields,
            } => {
                let (name, generics, fields) = (*name, generics.clone(), fields.clone());
                self.write("struct ");
                let t = self.text(name);
                self.write(t);
                self.generics_decl(&generics);
                self.write(" ");
                self.delimited_list("{", "}", fields.len(), |p, i| p.field_def(&fields[i]));
            }
            ItemKind::Enum {
                name,
                generics,
                variants,
            } => {
                let (name, generics) = (*name, generics.clone());
                self.write("enum ");
                let t = self.text(name);
                self.write(t);
                self.generics_decl(&generics);
                self.write(" {");
                self.last_pos = span.lo;
                self.indent += 1;
                for (i, v) in variants.iter().enumerate() {
                    if i > 0 {
                        self.write(",");
                    }
                    self.newline();
                    self.variant(v);
                }
                if !variants.is_empty() {
                    self.write(",");
                }
                self.indent -= 1;
                self.newline();
                self.write("}");
            }
            ItemKind::Trait {
                name,
                generics,
                items,
            } => {
                let (name, generics) = (*name, generics.clone());
                self.write("trait ");
                let t = self.text(name);
                self.write(t);
                self.generics_decl(&generics);
                self.write(" {");
                self.indent += 1;
                for it in items {
                    self.newline();
                    self.trait_item(it);
                }
                self.indent -= 1;
                self.newline();
                self.write("}");
            }
            ItemKind::Impl {
                generics,
                trait_,
                self_ty,
                items,
            } => {
                let (generics, trait_, self_ty) = (generics.clone(), trait_.clone(), *self_ty);
                self.write("impl");
                self.generics_decl(&generics);
                self.write(" ");
                if let Some(t) = &trait_ {
                    self.trait_ref(t);
                    self.write(" for ");
                }
                self.ty(self_ty);
                self.write(" {");
                self.indent += 1;
                for it in items {
                    self.newline();
                    self.impl_item(it);
                }
                self.indent -= 1;
                self.newline();
                self.write("}");
            }
            ItemKind::Const { name, ty, value } => {
                let (name, ty, value) = (*name, *ty, *value);
                self.write("const ");
                let t = self.text(name);
                self.write(t);
                self.write(": ");
                self.ty(ty);
                self.write(" = ");
                self.expr(value);
                self.write(";");
            }
            ItemKind::TypeAlias { name, generics, ty } => {
                let (name, generics, ty) = (*name, generics.clone(), *ty);
                self.write("type ");
                let t = self.text(name);
                self.write(t);
                self.generics_decl(&generics);
                self.write(" = ");
                self.ty(ty);
                self.write(";");
            }
            ItemKind::Use(tree) => {
                let mut leaves = Vec::new();
                flatten_use_tree(self, Vec::new(), tree, &mut leaves);
                leaves.sort();
                leaves.dedup();
                for (i, leaf) in leaves.iter().enumerate() {
                    if i > 0 {
                        self.newline();
                    }
                    self.write("use ");
                    self.write(leaf);
                    self.write(";");
                }
            }
            ItemKind::Mod { name, items } => {
                let (name, items) = (*name, items.clone());
                self.write("mod ");
                let t = self.text(name);
                self.write(t);
                match items {
                    None => self.write(";"),
                    Some(items) => {
                        self.write(" {");
                        self.last_pos = span.lo;
                        self.indent += 1;
                        self.item_seq(&items, span.hi);
                        self.indent -= 1;
                        self.newline();
                        self.write("}");
                    }
                }
            }
            ItemKind::Model(def) => {
                let (name, generics, ports) = (def.name, def.generics.clone(), &def.ports);
                self.write("model ");
                let t = self.text(name);
                self.write(t);
                self.generics_decl(&generics);
                self.write(" {");
                self.indent += 1;
                for port in ports {
                    self.newline();
                    self.model_port(port);
                }
                self.indent -= 1;
                self.newline();
                self.write("}");
            }
        }
        self.last_pos = span.hi;
    }

    fn field_def(&mut self, f: &FieldDef) {
        if f.is_pub {
            self.write("pub ");
        }
        let t = self.text(f.name);
        self.write(t);
        self.write(": ");
        self.ty(f.ty);
    }

    fn variant(&mut self, v: &Variant) {
        let t = self.text(v.name);
        self.write(t);
        match &v.kind {
            VariantKind::Unit => {}
            VariantKind::Tuple(tys) => {
                let tys = tys.clone();
                self.delimited_list("(", ")", tys.len(), |p, i| p.ty(tys[i]));
            }
            VariantKind::Struct(fields) => {
                let fields = fields.clone();
                self.write(" ");
                self.delimited_list("{", "}", fields.len(), |p, i| p.field_def(&fields[i]));
            }
        }
    }

    fn trait_item(&mut self, it: &TraitItem) {
        match it {
            TraitItem::Method { sig, body } => {
                self.fn_sig(sig);
                match body {
                    Some(b) => {
                        self.write(" ");
                        self.block(*b);
                    }
                    None => self.write(";"),
                }
            }
            TraitItem::AssocType { name } => {
                self.write("type ");
                let t = self.text(*name);
                self.write(t);
                self.write(";");
            }
        }
    }

    fn impl_item(&mut self, it: &ImplItem) {
        match it {
            ImplItem::Fn { vis, def } => {
                if let Some(v) = vis {
                    self.write(match v {
                        Vis::Pub => "pub ",
                        Vis::Priv => "priv ",
                    });
                }
                self.fn_sig(&def.sig);
                self.write(" ");
                self.block(def.body);
            }
            ImplItem::AssocType { name, ty } => {
                self.write("type ");
                let t = self.text(*name);
                self.write(t);
                self.write(" = ");
                self.ty(*ty);
                self.write(";");
            }
        }
    }

    fn model_port(&mut self, port: &ModelPort) {
        self.write(port.dir.keyword());
        self.write(" ");
        let t = self.text(port.name);
        self.write(t);
        self.write(": ");
        self.ty(port.ty);
        self.write(";");
    }

    fn fn_sig(&mut self, sig: &FnSig) {
        self.write("fn ");
        let t = self.text(sig.name);
        self.write(t);
        self.generics_decl(&sig.generics);
        let receiver = sig.receiver;
        let params = sig.params.clone();
        let n = params.len() + receiver.is_some() as usize;
        self.delimited_list("(", ")", n, |p, i| {
            if let Some(r) = receiver {
                if i == 0 {
                    p.write(receiver_text(r));
                    return;
                }
                p.param(&params[i - 1]);
            } else {
                p.param(&params[i]);
            }
        });
        match sig.ret {
            RetTy::Unit => {}
            RetTy::Ty(t) => {
                self.write(" -> ");
                self.ty(t);
            }
            RetTy::Never(_) => self.write(" -> !"),
        }
    }

    fn param(&mut self, p: &Param) {
        if p.mutable {
            self.write("mut ");
        }
        let t = self.text(p.name);
        self.write(t);
        self.write(": ");
        self.ty(p.ty);
    }

    fn generics_decl(&mut self, generics: &[GenericParam]) {
        if generics.is_empty() {
            return;
        }
        self.write("<");
        for (i, g) in generics.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            let t = self.text(g.name);
            self.write(t);
            if !g.bounds.is_empty() {
                self.write(": ");
                for (j, b) in g.bounds.iter().enumerate() {
                    if j > 0 {
                        self.write(" + ");
                    }
                    self.trait_ref(b);
                }
            }
        }
        self.write(">");
    }
}

/// Doc comments (`///`) bind tighter to the item that follows than the
/// item's own span suggests when a blank line separates other leading
/// comments from the doc block; in practice `emit_leading_comments` already
/// handles this by position (a `///` line immediately above the item has no
/// blank line, so no gap is inserted). This helper only supplies the
/// position leading-comment collection should stop before: the item's own
/// span start (its visibility keyword, or the item keyword if no `pub`).
fn item_leading_pos(node: &ItemNode) -> u32 {
    node.span.lo
}

fn unary_symbol(op: UnOp) -> &'static str {
    match op {
        UnOp::Neg => "-",
        UnOp::Not => "!",
        UnOp::BitNot => "~",
        UnOp::Ref { mutable: false } => "&",
        UnOp::Ref { mutable: true } => "&mut ",
        UnOp::Deref => "*",
    }
}

fn receiver_text(r: Receiver) -> &'static str {
    match r {
        Receiver::Value => "self",
        Receiver::Ref => "&self",
        Receiver::RefMut => "&mut self",
    }
}

fn flatten_use_tree(p: &Printer, mut prefix: Vec<String>, tree: &UseTree, out: &mut Vec<String>) {
    match tree {
        UseTree::Path { path, alias } => {
            for seg in &path.segments {
                prefix.push(p.text(seg.span).to_string());
            }
            let mut line = prefix.join("::");
            if let Some(a) = alias {
                line.push_str(" as ");
                line.push_str(p.text(*a));
            }
            out.push(line);
        }
        UseTree::Glob { prefix: pre } => {
            for seg in &pre.segments {
                prefix.push(p.text(seg.span).to_string());
            }
            prefix.push("*".to_string());
            out.push(prefix.join("::"));
        }
        UseTree::SelfImport { prefix: pre } => {
            for seg in &pre.segments {
                prefix.push(p.text(seg.span).to_string());
            }
            prefix.push("self".to_string());
            out.push(prefix.join("::"));
        }
        UseTree::Group { prefix: pre, items } => {
            for seg in &pre.segments {
                prefix.push(p.text(seg.span).to_string());
            }
            for item in items {
                flatten_use_tree(p, prefix.clone(), item, out);
            }
        }
    }
}
