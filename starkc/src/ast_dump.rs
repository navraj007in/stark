//! Stable indented tree dump of the AST, for snapshot tests and
//! `starkc parse --dump`. Kept out of `ast.rs` so that module stays a pure
//! data model.
//!
//! Node headers carry `@line:col` of the node start. The output format is
//! covered byte-for-byte by `tests/snapshots`, so any change here that alters
//! the rendering must regenerate the snapshots (`UPDATE_SNAPSHOTS=1`).
//!
//! No cloning: `self.ast` is `&'a Ast` (Copy), so hoisting `let ast = self.ast`
//! gives node borrows tied to the arena lifetime `'a`, independent of the
//! `&mut self` that the output buffer needs — the tree is walked entirely by
//! reference.

use crate::ast::*;
use crate::source::{SourceFile, Span};

/// Render `ast` (parsed from `file`) as an indented tree.
pub fn dump(ast: &Ast, file: &SourceFile) -> String {
    let mut d = Dumper {
        ast,
        file,
        out: String::new(),
        depth: 0,
    };
    match &ast.root {
        Root::Program(items) => {
            d.line("program");
            d.depth += 1;
            for &item in items {
                d.item(item);
            }
        }
        Root::Snippet { stmts, tail } => {
            d.line("snippet");
            d.depth += 1;
            for &stmt in stmts {
                d.stmt(stmt);
            }
            if let Some(tail) = *tail {
                d.nested("tail".to_string(), |d| d.expr(tail));
            }
        }
    }
    d.out
}

struct Dumper<'a> {
    ast: &'a Ast,
    file: &'a SourceFile,
    out: String,
    depth: usize,
}

impl<'a> Dumper<'a> {
    fn line(&mut self, text: impl AsRef<str>) {
        for _ in 0..self.depth {
            self.out.push_str("  ");
        }
        self.out.push_str(text.as_ref());
        self.out.push('\n');
    }

    fn nested(&mut self, header: String, body: impl FnOnce(&mut Self)) {
        self.line(header);
        self.depth += 1;
        body(self);
        self.depth -= 1;
    }

    fn at(&self, span: Span) -> String {
        let (line, col) = self.file.line_col(span.lo);
        format!("@{line}:{col}")
    }

    fn text(&self, span: Span) -> &'a str {
        &self.file.src[span.lo as usize..span.hi as usize]
    }

    fn path(&self, path: &Path) -> String {
        let mut s = String::new();
        for (i, seg) in path.segments.iter().enumerate() {
            if i > 0 {
                s.push_str("::");
            }
            s.push_str(self.text(seg.span));
        }
        s
    }

    fn lit(&self, lit: &Lit, span: Span) -> String {
        let text = self.text(span);
        match lit {
            Lit::Int { .. } => format!("lit-int {text}"),
            Lit::Float { .. } => format!("lit-float {text}"),
            Lit::Str { raw } => format!("lit-str{} {text}", if *raw { " raw" } else { "" }),
            Lit::Char => format!("lit-char {text}"),
            Lit::Bool(b) => format!("lit-bool {b}"),
        }
    }

    // ------------------------------------------------------------- types --

    fn generic_args(&mut self, args: &'a GenericArgs) {
        for arg in &args.args {
            match arg {
                GenericArg::Type(ty) => self.ty(*ty),
                GenericArg::Binding { name, ty } => {
                    let header = format!("binding {}", self.text(*name));
                    let ty = *ty;
                    self.nested(header, |d| d.ty(ty));
                }
                GenericArg::Shape(shape) => {
                    let header = format!("shape {}", self.at(shape.span));
                    let dims = shape.dims.clone();
                    self.nested(header, |d| {
                        for dim in dims {
                            d.dim(dim);
                        }
                    });
                }
            }
        }
    }

    fn dim(&mut self, id: DimId) {
        let node = self.ast.dim(id);
        let at = self.at(node.span);
        match &node.kind {
            DimExprKind::Lit(s) => self.line(format!("dim-lit {} {at}", self.text(*s))),
            DimExprKind::Var(s) => self.line(format!("dim-var {} {at}", self.text(*s))),
            DimExprKind::Binary { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                self.nested(format!("dim-binary {} {at}", op.symbol()), |d| {
                    d.dim(lhs);
                    d.dim(rhs);
                });
            }
            DimExprKind::Error => self.line(format!("dim-error {at}")),
        }
    }

    fn ty(&mut self, id: TypeId) {
        let node = self.ast.ty(id);
        let at = self.at(node.span);
        match &node.kind {
            TypeKind::Primitive(p) => self.line(format!("type-primitive {} {at}", p.name())),
            TypeKind::Path { path, args } => {
                let header = format!("type-path {} {at}", self.path(path));
                self.nested(header, |d| {
                    if let Some(args) = args {
                        d.generic_args(args);
                    }
                });
            }
            TypeKind::Array { elem, len } => {
                let header = format!("type-array len={} {at}", self.text(*len));
                let elem = *elem;
                self.nested(header, |d| d.ty(elem));
            }
            TypeKind::Slice(elem) => {
                let elem = *elem;
                self.nested(format!("type-slice {at}"), |d| d.ty(elem));
            }
            TypeKind::Tuple(elems) => {
                self.nested(format!("type-tuple {at}"), |d| {
                    for &e in elems {
                        d.ty(e);
                    }
                });
            }
            TypeKind::Ref { mutable, inner } => {
                let header = format!("type-ref{} {at}", if *mutable { " mut" } else { "" });
                let inner = *inner;
                self.nested(header, |d| d.ty(inner));
            }
            TypeKind::Fn { params, ret } => {
                let ret = *ret;
                self.nested(format!("type-fn {at}"), |d| {
                    for &p in params {
                        d.ty(p);
                    }
                    if let Some(ret) = ret {
                        d.nested("ret".to_string(), |d| d.ty(ret));
                    }
                });
            }
            TypeKind::Never => self.line(format!("type-never {at}")),
            TypeKind::Error => self.line(format!("type-error {at}")),
        }
    }

    // ------------------------------------------------------- expressions --

    fn expr(&mut self, id: ExprId) {
        let node = self.ast.expr(id);
        let at = self.at(node.span);
        match &node.kind {
            ExprKind::Lit(lit) => {
                let line = format!("{} {at}", self.lit(lit, node.span));
                self.line(line);
            }
            ExprKind::Path { path, turbofish } => {
                let mut header = format!("path {} {at}", self.path(path));
                if turbofish.is_some() {
                    header.push_str(" turbofish");
                }
                self.nested(header, |d| {
                    if let Some(args) = turbofish {
                        d.generic_args(args);
                    }
                });
            }
            ExprKind::Unary { op, operand } => {
                let (op, operand) = (*op, *operand);
                self.nested(format!("unary {op:?} {at}"), |d| d.expr(operand));
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                self.nested(format!("binary {op:?} {at}"), |d| {
                    d.expr(lhs);
                    d.expr(rhs);
                });
            }
            ExprKind::Assign { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                self.nested(format!("assign {op:?} {at}"), |d| {
                    d.expr(lhs);
                    d.expr(rhs);
                });
            }
            ExprKind::Range { lo, hi, inclusive } => {
                let (lo, hi, inclusive) = (*lo, *hi, *inclusive);
                let header = format!("range{} {at}", if inclusive { " inclusive" } else { "" });
                self.nested(header, |d| {
                    d.expr(lo);
                    d.expr(hi);
                });
            }
            ExprKind::Cast { expr, ty } => {
                let (expr, ty) = (*expr, *ty);
                self.nested(format!("cast {at}"), |d| {
                    d.expr(expr);
                    d.ty(ty);
                });
            }
            ExprKind::Call { callee, args } => {
                let callee = *callee;
                self.nested(format!("call {at}"), |d| {
                    d.expr(callee);
                    for &a in args {
                        d.expr(a);
                    }
                });
            }
            ExprKind::Field { base, name } => {
                let base = *base;
                let header = format!("field {} {at}", self.text(*name));
                self.nested(header, |d| d.expr(base));
            }
            ExprKind::TupleField { base, index } => {
                let base = *base;
                let header = format!("tuple-field {} {at}", self.text(*index));
                self.nested(header, |d| d.expr(base));
            }
            ExprKind::Index { base, index } => {
                let (base, index) = (*base, *index);
                self.nested(format!("index {at}"), |d| {
                    d.expr(base);
                    d.expr(index);
                });
            }
            ExprKind::Try(expr) => {
                let expr = *expr;
                self.nested(format!("try {at}"), |d| d.expr(expr));
            }
            ExprKind::Tuple(elems) => {
                self.nested(format!("tuple {at}"), |d| {
                    for &e in elems {
                        d.expr(e);
                    }
                });
            }
            ExprKind::Array(elems) => {
                self.nested(format!("array {at}"), |d| {
                    for &e in elems {
                        d.expr(e);
                    }
                });
            }
            ExprKind::Repeat { value, count } => {
                let (value, count) = (*value, *count);
                self.nested(format!("repeat {at}"), |d| {
                    d.expr(value);
                    d.expr(count);
                });
            }
            ExprKind::StructLit { path, fields } => {
                let header = format!("struct-lit {} {at}", self.path(path));
                self.nested(header, |d| {
                    for f in fields {
                        let field_header = format!("field-init {}", d.text(f.name));
                        let expr = f.expr;
                        d.nested(field_header, |d| {
                            if let Some(expr) = expr {
                                d.expr(expr);
                            }
                        });
                    }
                });
            }
            ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                let (cond, then_block, else_) = (*cond, *then_block, *else_);
                self.nested(format!("if {at}"), |d| {
                    d.expr(cond);
                    d.block(then_block);
                    if let Some(else_) = else_ {
                        d.nested("else".to_string(), |d| d.expr(else_));
                    }
                });
            }
            ExprKind::Match { scrutinee, arms } => {
                let scrutinee = *scrutinee;
                self.nested(format!("match {at}"), |d| {
                    d.expr(scrutinee);
                    for arm in arms {
                        let (pat, body) = (arm.pat, arm.body);
                        d.nested("arm".to_string(), |d| {
                            d.pat(pat);
                            d.expr(body);
                        });
                    }
                });
            }
            ExprKind::Loop { body } => {
                let body = *body;
                self.nested(format!("loop {at}"), |d| d.block(body));
            }
            ExprKind::While { cond, body } => {
                let (cond, body) = (*cond, *body);
                self.nested(format!("while {at}"), |d| {
                    d.expr(cond);
                    d.block(body);
                });
            }
            ExprKind::For { var, iter, body } => {
                let header = format!("for {} {at}", self.text(*var));
                let (iter, body) = (*iter, *body);
                self.nested(header, |d| {
                    d.expr(iter);
                    d.block(body);
                });
            }
            ExprKind::Block(block) => {
                let block = *block;
                self.nested(format!("block-expr {at}"), |d| d.block(block));
            }
            ExprKind::Error => self.line(format!("expr-error {at}")),
        }
    }

    // ---------------------------------------------------------- patterns --

    fn pat(&mut self, id: PatId) {
        let node = self.ast.pat(id);
        let at = self.at(node.span);
        match &node.kind {
            PatKind::Lit(lit) => {
                let line = format!("pat-{} {at}", self.lit(lit, node.span));
                self.line(line);
            }
            PatKind::Wild => self.line(format!("pat-wild {at}")),
            PatKind::Binding(name) => self.line(format!("pat-binding {} {at}", self.text(*name))),
            PatKind::Path(path) => self.line(format!("pat-path {} {at}", self.path(path))),
            PatKind::TupleVariant { path, pats } => {
                let header = format!("pat-tuple-variant {} {at}", self.path(path));
                self.nested(header, |d| {
                    for &p in pats {
                        d.pat(p);
                    }
                });
            }
            PatKind::Struct { path, fields } => {
                let header = format!("pat-struct {} {at}", self.path(path));
                self.nested(header, |d| {
                    for f in fields {
                        let field_header = format!("field-pat {}", d.text(f.name));
                        let pat = f.pat;
                        d.nested(field_header, |d| {
                            if let Some(pat) = pat {
                                d.pat(pat);
                            }
                        });
                    }
                });
            }
            PatKind::Tuple(pats) => {
                self.nested(format!("pat-tuple {at}"), |d| {
                    for &p in pats {
                        d.pat(p);
                    }
                });
            }
            PatKind::Array(pats) => {
                self.nested(format!("pat-array {at}"), |d| {
                    for &p in pats {
                        d.pat(p);
                    }
                });
            }
        }
    }

    // -------------------------------------------------- statements/blocks --

    fn stmt(&mut self, id: StmtId) {
        let node = self.ast.stmt(id);
        let at = self.at(node.span);
        match &node.kind {
            StmtKind::Empty => self.line(format!("empty-stmt {at}")),
            StmtKind::Expr { expr, semi } => {
                let (expr, semi) = (*expr, *semi);
                let header = format!("expr-stmt{} {at}", if semi { "" } else { " nosemi" });
                self.nested(header, |d| d.expr(expr));
            }
            StmtKind::Let {
                mutable,
                name,
                ty,
                init,
            } => {
                let header = format!(
                    "let{} {} {at}",
                    if *mutable { " mut" } else { "" },
                    self.text(*name)
                );
                let (ty, init) = (*ty, *init);
                self.nested(header, |d| {
                    if let Some(ty) = ty {
                        d.ty(ty);
                    }
                    if let Some(init) = init {
                        d.expr(init);
                    }
                });
            }
            StmtKind::Return(expr) => {
                let expr = *expr;
                self.nested(format!("return {at}"), |d| {
                    if let Some(expr) = expr {
                        d.expr(expr);
                    }
                });
            }
            StmtKind::Break(expr) => {
                let expr = *expr;
                self.nested(format!("break {at}"), |d| {
                    if let Some(expr) = expr {
                        d.expr(expr);
                    }
                });
            }
            StmtKind::Continue => self.line(format!("continue {at}")),
            StmtKind::Item(item) => {
                let item = *item;
                self.item(item);
            }
            StmtKind::Error => self.line(format!("stmt-error {at}")),
        }
    }

    fn block(&mut self, id: BlockId) {
        let node = self.ast.block(id);
        let at = self.at(node.span);
        self.nested(format!("block {at}"), |d| {
            for &s in &node.stmts {
                d.stmt(s);
            }
            if let Some(tail) = node.tail {
                d.nested("tail".to_string(), |d| d.expr(tail));
            }
        });
    }

    // ------------------------------------------------------------- items --

    fn generic_params(&mut self, generics: &'a [GenericParam]) {
        if generics.is_empty() {
            return;
        }
        self.nested("generics".to_string(), |d| {
            for p in generics {
                let mut header = format!("param {}", d.text(p.name));
                if !p.bounds.is_empty() {
                    header.push_str(": ");
                    let joined = p
                        .bounds
                        .iter()
                        .map(|b| d.path(&b.path))
                        .collect::<Vec<_>>()
                        .join(" + ");
                    header.push_str(&joined);
                }
                d.nested(header, |d| {
                    for b in &p.bounds {
                        if let Some(args) = &b.args {
                            d.generic_args(args);
                        }
                    }
                });
            }
        });
    }

    fn fn_sig(&mut self, sig: &'a FnSig) {
        let header = format!("sig {} {}", self.text(sig.name), self.at(sig.span));
        self.nested(header, |d| {
            d.generic_params(&sig.generics);
            if let Some(receiver) = sig.receiver {
                d.line(format!("receiver {receiver:?}"));
            }
            for p in &sig.params {
                let param_header = format!(
                    "param{} {}",
                    if p.mutable { " mut" } else { "" },
                    d.text(p.name)
                );
                let ty = p.ty;
                d.nested(param_header, |d| d.ty(ty));
            }
            match sig.ret {
                RetTy::Unit => {}
                RetTy::Ty(ty) => d.nested("ret".to_string(), |d| d.ty(ty)),
                RetTy::Never(_) => d.line("ret never"),
            }
        });
    }

    fn use_tree(&mut self, tree: &'a UseTree) {
        match tree {
            UseTree::Path { path, alias } => {
                let mut line = format!("use-path {}", self.path(path));
                if let Some(alias) = alias {
                    line.push_str(" as ");
                    line.push_str(self.text(*alias));
                }
                self.line(line);
            }
            UseTree::Glob { prefix } => self.line(format!("use-glob {}::*", self.path(prefix))),
            UseTree::SelfImport { prefix } => {
                self.line(format!("use-self {}::self", self.path(prefix)))
            }
            UseTree::Group { prefix, items } => {
                let header = format!("use-group {}", self.path(prefix));
                self.nested(header, |d| {
                    for item in items {
                        d.use_tree(item);
                    }
                });
            }
        }
    }

    fn vis_prefix(vis: Option<Vis>) -> &'static str {
        match vis {
            Some(Vis::Pub) => "pub ",
            Some(Vis::Priv) => "priv ",
            None => "",
        }
    }

    fn field_def(&mut self, f: &FieldDef) {
        let header = format!(
            "field{} {}",
            if f.is_pub { " pub" } else { "" },
            self.text(f.name)
        );
        let ty = f.ty;
        self.nested(header, |d| d.ty(ty));
    }

    fn item(&mut self, id: ItemId) {
        let node = self.ast.item(id);
        let at = self.at(node.span);
        let vis = Self::vis_prefix(node.vis);
        match &node.kind {
            ItemKind::Fn(def) => {
                let body = def.body;
                self.nested(format!("{vis}fn {at}"), |d| {
                    d.fn_sig(&def.sig);
                    d.block(body);
                });
            }
            ItemKind::Struct {
                name,
                generics,
                fields,
            } => {
                let header = format!("{vis}struct {} {at}", self.text(*name));
                self.nested(header, |d| {
                    d.generic_params(generics);
                    for f in fields {
                        d.field_def(f);
                    }
                });
            }
            ItemKind::Enum {
                name,
                generics,
                variants,
            } => {
                let header = format!("{vis}enum {} {at}", self.text(*name));
                self.nested(header, |d| {
                    d.generic_params(generics);
                    for v in variants {
                        let vh = format!("variant {}", d.text(v.name));
                        d.nested(vh, |d| match &v.kind {
                            VariantKind::Unit => {}
                            VariantKind::Tuple(tys) => {
                                for &ty in tys {
                                    d.ty(ty);
                                }
                            }
                            VariantKind::Struct(fields) => {
                                for f in fields {
                                    d.field_def(f);
                                }
                            }
                        });
                    }
                });
            }
            ItemKind::Trait {
                name,
                generics,
                items,
            } => {
                let header = format!("{vis}trait {} {at}", self.text(*name));
                self.nested(header, |d| {
                    d.generic_params(generics);
                    for it in items {
                        match it {
                            TraitItem::AssocType { name } => {
                                d.line(format!("assoc-type {}", d.text(*name)))
                            }
                            TraitItem::Method { sig, body } => {
                                d.fn_sig(sig);
                                if let Some(body) = body {
                                    d.block(*body);
                                }
                            }
                        }
                    }
                });
            }
            ItemKind::Impl {
                generics,
                trait_,
                self_ty,
                items,
            } => {
                let header = match trait_ {
                    Some(tr) => format!("{vis}impl {} for {at}", self.path(&tr.path)),
                    None => format!("{vis}impl {at}"),
                };
                let self_ty = *self_ty;
                self.nested(header, |d| {
                    d.generic_params(generics);
                    if let Some(tr) = trait_ {
                        if let Some(args) = &tr.args {
                            d.nested("trait-args".to_string(), |d| d.generic_args(args));
                        }
                    }
                    d.nested("self-ty".to_string(), |d| d.ty(self_ty));
                    for it in items {
                        match it {
                            ImplItem::Fn { vis, def } => {
                                let fvis = Self::vis_prefix(*vis);
                                let body = def.body;
                                d.nested(format!("{fvis}method"), |d| {
                                    d.fn_sig(&def.sig);
                                    d.block(body);
                                });
                            }
                            ImplItem::AssocType { name, ty } => {
                                let header = format!("assoc-type {} =", d.text(*name));
                                let ty = *ty;
                                d.nested(header, |d| d.ty(ty));
                            }
                        }
                    }
                });
            }
            ItemKind::Const { name, ty, value } => {
                let header = format!("{vis}const {} {at}", self.text(*name));
                let (ty, value) = (*ty, *value);
                self.nested(header, |d| {
                    d.ty(ty);
                    d.expr(value);
                });
            }
            ItemKind::TypeAlias { name, generics, ty } => {
                let header = format!("{vis}type-alias {} {at}", self.text(*name));
                let ty = *ty;
                self.nested(header, |d| {
                    d.generic_params(generics);
                    d.ty(ty);
                });
            }
            ItemKind::Use(tree) => {
                self.nested(format!("{vis}use {at}"), |d| d.use_tree(tree));
            }
            ItemKind::Mod { name, items } => {
                let header = format!("{vis}mod {} {at}", self.text(*name));
                self.nested(header, |d| match items {
                    Some(items) => {
                        for &item in items {
                            d.item(item);
                        }
                    }
                    None => d.line("external"),
                });
            }
            ItemKind::Model(def) => {
                let header = format!("{vis}model {} {at}", self.text(def.name));
                self.nested(header, |d| {
                    d.generic_params(&def.generics);
                    for port in &def.ports {
                        let ph = format!(
                            "port {} {} {}",
                            port.dir.keyword(),
                            d.text(port.name),
                            d.at(port.span)
                        );
                        let ty = port.ty;
                        d.nested(ph, |d| d.ty(ty));
                    }
                });
            }
        }
    }
}
