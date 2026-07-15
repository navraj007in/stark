//! AST for `02-Syntax-Grammar.md`.
//!
//! Per PLAN.md T6: arena-allocated nodes referenced by typed IDs
//! (`ExprId`, `ItemId`, ...); every node carries a `Span`; no Rust
//! references or lifetimes in the tree. Types/ownership facts attach in
//! side tables keyed by these IDs from Gate 2 onward.
//!
//! Names are stored as `Span`s into the source, not owned strings.
//!
//! Grouping parentheses are not represented: `(expr)` parses to the inner
//! expression (a 1-tuple is distinguished by its trailing comma at parse
//! time), and `(T)` in type position parses to the inner type.

use crate::lexer::{Base, FloatSuffix, IntSuffix};
use crate::source::{SourceFile, Span};

macro_rules! arena_id {
    ($Id:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub struct $Id(pub u32);
    };
}

arena_id!(TypeId);
arena_id!(ExprId);
arena_id!(StmtId);
arena_id!(ItemId);
arena_id!(PatId);
arena_id!(BlockId);

#[derive(Default)]
pub struct Ast {
    pub types: Vec<TypeNode>,
    pub exprs: Vec<ExprNode>,
    pub stmts: Vec<StmtNode>,
    pub items: Vec<ItemNode>,
    pub pats: Vec<PatNode>,
    pub blocks: Vec<BlockNode>,
    pub root: Root,
}

/// What was parsed. `Program` is the source-language entry point
/// (`Program ::= Item*`); `Snippet` is the harness-only block-body form
/// `(Item | Statement)* Expression?` for spec examples written at statement
/// level (manifest `mode = "snippet"`).
pub enum Root {
    Program(Vec<ItemId>),
    Snippet {
        stmts: Vec<StmtId>,
        tail: Option<ExprId>,
    },
}

impl Default for Root {
    fn default() -> Self {
        Root::Program(Vec::new())
    }
}

// ---------------------------------------------------------------- types --

pub struct TypeNode {
    pub kind: TypeKind,
    pub span: Span,
}

pub enum TypeKind {
    /// `Int32`, `Bool`, `String`, `str`, ...
    Primitive(Primitive),
    /// `Vec<Int32>`, `Option<T>`, `Self::Item`, `T`.
    Path {
        path: Path,
        args: Option<GenericArgs>,
    },
    /// `[T; N]` — the length is an INTEGER literal, uninterpreted in Gate 1.
    Array { elem: TypeId, len: Span },
    /// `[T]`
    Slice(TypeId),
    /// `()`, `(T,)`, `(T1, T2)`. Never one element without a comma —
    /// `(T)` is grouping and constructs no node.
    Tuple(Vec<TypeId>),
    /// `&T` / `&mut T`
    Ref { mutable: bool, inner: TypeId },
    /// `fn(T1, T2) -> R`
    Fn {
        params: Vec<TypeId>,
        ret: Option<TypeId>,
    },
    /// `!` — produced only in function return position (`ReturnType`).
    Never,
    /// Placeholder for a type that failed to parse (a diagnostic exists).
    Error,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Primitive {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool,
    Char,
    String,
    Str,
    Unit,
}

impl Primitive {
    pub fn name(self) -> &'static str {
        match self {
            Primitive::Int8 => "Int8",
            Primitive::Int16 => "Int16",
            Primitive::Int32 => "Int32",
            Primitive::Int64 => "Int64",
            Primitive::UInt8 => "UInt8",
            Primitive::UInt16 => "UInt16",
            Primitive::UInt32 => "UInt32",
            Primitive::UInt64 => "UInt64",
            Primitive::Float32 => "Float32",
            Primitive::Float64 => "Float64",
            Primitive::Bool => "Bool",
            Primitive::Char => "Char",
            Primitive::String => "String",
            Primitive::Str => "str",
            Primitive::Unit => "Unit",
        }
    }
}

// ---------------------------------------------------------------- paths --

#[derive(Clone)]
pub struct Path {
    pub segments: Vec<PathSegment>,
    pub span: Span,
}

#[derive(Clone, Copy)]
pub struct PathSegment {
    pub kind: SegmentKind,
    pub span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SegmentKind {
    Ident,
    SelfValue, // `self`
    SelfType,  // `Self`
    Super,
    Crate,
}

#[derive(Clone)]
pub struct GenericArgs {
    pub args: Vec<GenericArg>,
    pub span: Span,
}

#[derive(Clone)]
pub enum GenericArg {
    Type(TypeId),
    /// `Item = T` associated-type binding.
    Binding {
        name: Span,
        ty: TypeId,
    },
}

// ---------------------------------------------------------- expressions --

pub struct ExprNode {
    pub kind: ExprKind,
    pub span: Span,
}

pub enum ExprKind {
    Lit(Lit),
    /// `x`, `String::from`, `Color::Red`, `size_of::<Int32>` (turbofish).
    Path {
        path: Path,
        turbofish: Option<GenericArgs>,
    },
    Unary {
        op: UnOp,
        operand: ExprId,
    },
    Binary {
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    /// `lhs = rhs`, `lhs += rhs`, ... Place-ness of `lhs` is a semantic
    /// check (04-Semantic-Analysis.md), not a parse error.
    Assign {
        op: AssignOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    /// `lo..hi` / `lo..=hi` — both operands required in Core v1.
    Range {
        lo: ExprId,
        hi: ExprId,
        inclusive: bool,
    },
    Cast {
        expr: ExprId,
        ty: TypeId,
    },
    Call {
        callee: ExprId,
        args: Vec<ExprId>,
    },
    /// `base.name` — field access or method reference (resolved in Gate 2).
    Field {
        base: ExprId,
        name: Span,
    },
    /// `base.0` — tuple field access; the index is an INTEGER literal.
    TupleField {
        base: ExprId,
        index: Span,
    },
    Index {
        base: ExprId,
        index: ExprId,
    },
    /// `expr?`
    Try(ExprId),
    /// `()`, `(a,)`, `(a, b)`. Never one element without a comma.
    Tuple(Vec<ExprId>),
    Array(Vec<ExprId>),
    /// `[value; count]`
    Repeat {
        value: ExprId,
        count: ExprId,
    },
    StructLit {
        path: Path,
        fields: Vec<FieldInit>,
    },
    /// `else_` is `None`, a `Block` expression, or another `If` expression.
    If {
        cond: ExprId,
        then_block: BlockId,
        else_: Option<ExprId>,
    },
    Match {
        scrutinee: ExprId,
        arms: Vec<MatchArm>,
    },
    Loop {
        body: BlockId,
    },
    While {
        cond: ExprId,
        body: BlockId,
    },
    For {
        var: Span,
        iter: ExprId,
        body: BlockId,
    },
    Block(BlockId),
    /// Placeholder for an expression that failed to parse.
    Error,
}

#[derive(Clone, Copy)]
pub enum Lit {
    Int {
        base: Base,
        suffix: Option<IntSuffix>,
    },
    Float {
        suffix: Option<FloatSuffix>,
    },
    Str {
        raw: bool,
    },
    Char,
    Bool(bool),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnOp {
    Neg,
    Not,
    BitNot,
    /// `&expr` / `&mut expr`
    Ref {
        mutable: bool,
    },
    Deref,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    PowAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    ShlAssign,
    ShrAssign,
}

pub struct FieldInit {
    pub name: Span,
    /// `None` is the shorthand `Point { x, y }`.
    pub expr: Option<ExprId>,
}

pub struct MatchArm {
    pub pat: PatId,
    pub body: ExprId,
}

// ------------------------------------------------------------ statements --

pub struct StmtNode {
    pub kind: StmtKind,
    pub span: Span,
}

pub enum StmtKind {
    /// `;`
    Empty,
    /// Expression statement. `semi` is false only for block-formed
    /// expression statements (`if c { }` without `;`).
    Expr {
        expr: ExprId,
        semi: bool,
    },
    Let {
        mutable: bool,
        name: Span,
        ty: Option<TypeId>,
        init: Option<ExprId>,
    },
    Return(Option<ExprId>),
    Break(Option<ExprId>),
    Continue,
    /// Item in snippet mode only (`Root::Snippet`); Core v1 blocks do not
    /// nest items.
    Item(ItemId),
    /// Placeholder for a statement that failed to parse.
    Error,
}

// ---------------------------------------------------------------- blocks --

pub struct BlockNode {
    pub stmts: Vec<StmtId>,
    /// Trailing expression (the block's value).
    pub tail: Option<ExprId>,
    pub span: Span,
}

// ----------------------------------------------------------------- items --

pub struct ItemNode {
    pub kind: ItemKind,
    pub vis: Option<Vis>,
    pub span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Vis {
    Pub,
    Priv,
}

pub enum ItemKind {
    Fn(FnDef),
    Struct {
        name: Span,
        generics: Vec<GenericParam>,
        fields: Vec<FieldDef>,
    },
    Enum {
        name: Span,
        generics: Vec<GenericParam>,
        variants: Vec<Variant>,
    },
    Trait {
        name: Span,
        generics: Vec<GenericParam>,
        items: Vec<TraitItem>,
    },
    Impl {
        generics: Vec<GenericParam>,
        /// `Some` for `impl Trait for Type`, `None` for inherent impls.
        trait_: Option<TraitRef>,
        self_ty: TypeId,
        items: Vec<ImplItem>,
    },
    Const {
        name: Span,
        ty: TypeId,
        value: ExprId,
    },
    TypeAlias {
        name: Span,
        generics: Vec<GenericParam>,
        ty: TypeId,
    },
    Use(UseTree),
    /// `items: None` is an external module declaration (`mod name;`).
    Mod {
        name: Span,
        items: Option<Vec<ItemId>>,
    },
}

pub struct FnDef {
    pub sig: FnSig,
    pub body: BlockId,
}

#[derive(Clone)]
pub struct FnSig {
    pub name: Span,
    pub generics: Vec<GenericParam>,
    pub receiver: Option<Receiver>,
    pub params: Vec<Param>,
    pub ret: RetTy,
    pub span: Span,
}

#[derive(Clone, Copy)]
pub enum RetTy {
    /// No `->`: returns `Unit`.
    Unit,
    Ty(TypeId),
    /// `-> !`
    Never(Span),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Receiver {
    /// `self`
    Value,
    /// `&self`
    Ref,
    /// `&mut self`
    RefMut,
}

#[derive(Clone, Copy)]
pub struct Param {
    pub mutable: bool,
    pub name: Span,
    pub ty: TypeId,
}

#[derive(Clone)]
pub struct GenericParam {
    pub name: Span,
    pub bounds: Vec<TraitRef>,
}

/// A trait named by path, e.g. `Ord`, `Iterator<Item = T>`.
#[derive(Clone)]
pub struct TraitRef {
    pub path: Path,
    pub args: Option<GenericArgs>,
}

#[derive(Clone, Copy)]
pub struct FieldDef {
    pub is_pub: bool,
    pub name: Span,
    pub ty: TypeId,
}

pub struct Variant {
    pub name: Span,
    pub kind: VariantKind,
}

pub enum VariantKind {
    Unit,
    Tuple(Vec<TypeId>),
    Struct(Vec<FieldDef>),
}

pub enum TraitItem {
    /// Required method (`body: None`) or method with default body.
    Method { sig: FnSig, body: Option<BlockId> },
    /// `type Item;`
    AssocType { name: Span },
}

pub enum ImplItem {
    Fn {
        vis: Option<Vis>,
        def: FnDef,
    },
    /// `type Item = Int32;`
    AssocType {
        name: Span,
        ty: TypeId,
    },
}

#[derive(Clone)]
pub enum UseTree {
    /// `use a::b::c;` / `use a::b as x;`
    Path { path: Path, alias: Option<Span> },
    /// `use a::b::*;`
    Glob { prefix: Path },
    /// `use a::b::self;`
    SelfImport { prefix: Path },
    /// `use a::{b, c as d, e::f};`
    Group { prefix: Path, items: Vec<UseTree> },
}

// ---------------------------------------------------------------- arena --

impl Ast {
    pub fn alloc_type(&mut self, kind: TypeKind, span: Span) -> TypeId {
        self.types.push(TypeNode { kind, span });
        TypeId(self.types.len() as u32 - 1)
    }
    pub fn alloc_expr(&mut self, kind: ExprKind, span: Span) -> ExprId {
        self.exprs.push(ExprNode { kind, span });
        ExprId(self.exprs.len() as u32 - 1)
    }
    pub fn alloc_stmt(&mut self, kind: StmtKind, span: Span) -> StmtId {
        self.stmts.push(StmtNode { kind, span });
        StmtId(self.stmts.len() as u32 - 1)
    }
    pub fn alloc_item(&mut self, kind: ItemKind, vis: Option<Vis>, span: Span) -> ItemId {
        self.items.push(ItemNode { kind, vis, span });
        ItemId(self.items.len() as u32 - 1)
    }
    pub fn alloc_pat(&mut self, kind: PatKind, span: Span) -> PatId {
        self.pats.push(PatNode { kind, span });
        PatId(self.pats.len() as u32 - 1)
    }
    pub fn alloc_block(&mut self, block: BlockNode) -> BlockId {
        self.blocks.push(block);
        BlockId(self.blocks.len() as u32 - 1)
    }

    pub fn ty(&self, id: TypeId) -> &TypeNode {
        &self.types[id.0 as usize]
    }
    pub fn expr(&self, id: ExprId) -> &ExprNode {
        &self.exprs[id.0 as usize]
    }
    pub fn stmt(&self, id: StmtId) -> &StmtNode {
        &self.stmts[id.0 as usize]
    }
    pub fn item(&self, id: ItemId) -> &ItemNode {
        &self.items[id.0 as usize]
    }
    pub fn pat(&self, id: PatId) -> &PatNode {
        &self.pats[id.0 as usize]
    }
    pub fn block(&self, id: BlockId) -> &BlockNode {
        &self.blocks[id.0 as usize]
    }
}

// -------------------------------------------------------------- patterns --

pub struct PatNode {
    pub kind: PatKind,
    pub span: Span,
}

pub enum PatKind {
    Lit(Lit),
    /// `_`
    Wild,
    /// Single identifier: new binding, or unit variant/const after name
    /// resolution (02's pattern note).
    Binding(Span),
    /// Multi-segment path: `Color::Red`.
    Path(Path),
    /// `Option::Some(x)`
    TupleVariant {
        path: Path,
        pats: Vec<PatId>,
    },
    /// `Point { x, y: 0 }`
    Struct {
        path: Path,
        fields: Vec<FieldPat>,
    },
    Tuple(Vec<PatId>),
    Array(Vec<PatId>),
}

pub struct FieldPat {
    pub name: Span,
    /// `None` is the shorthand (binds the field to a same-named variable).
    pub pat: Option<PatId>,
}

// ------------------------------------------------------------------ dump --

/// Stable indented tree dump for snapshot tests and `starkc parse --dump`.
/// Node headers carry `@line:col` of the node start.
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

impl Dumper<'_> {
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

    fn text(&self, span: Span) -> &str {
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

    fn generic_args(&mut self, args: &GenericArgs) {
        for arg in args.args.clone() {
            match arg {
                GenericArg::Type(ty) => self.ty(ty),
                GenericArg::Binding { name, ty } => {
                    let header = format!("binding {}", self.text(name));
                    self.nested(header, |d| d.ty(ty));
                }
            }
        }
    }

    fn ty(&mut self, id: TypeId) {
        let node = self.ast.ty(id);
        let at = self.at(node.span);
        match &node.kind {
            TypeKind::Primitive(p) => {
                let line = format!("type-primitive {} {at}", p.name());
                self.line(line);
            }
            TypeKind::Path { path, args } => {
                let header = format!("type-path {} {at}", self.path(path));
                let args = args.clone();
                self.nested(header, |d| {
                    if let Some(args) = &args {
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
                let elems = elems.clone();
                self.nested(format!("type-tuple {at}"), |d| {
                    for e in elems {
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
                let params = params.clone();
                let ret = *ret;
                self.nested(format!("type-fn {at}"), |d| {
                    for p in params {
                        d.ty(p);
                    }
                    if let Some(ret) = ret {
                        d.nested("ret".to_string(), |d| d.ty(ret));
                    }
                });
            }
            TypeKind::Never => {
                let line = format!("type-never {at}");
                self.line(line);
            }
            TypeKind::Error => {
                let line = format!("type-error {at}");
                self.line(line);
            }
        }
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
                let turbofish = turbofish.clone();
                self.nested(header, |d| {
                    if let Some(args) = &turbofish {
                        d.generic_args(args);
                    }
                });
            }
            ExprKind::Unary { op, operand } => {
                let operand = *operand;
                let op = *op;
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
                let args = args.clone();
                self.nested(format!("call {at}"), |d| {
                    d.expr(callee);
                    for a in args {
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
                let elems = elems.clone();
                self.nested(format!("tuple {at}"), |d| {
                    for e in elems {
                        d.expr(e);
                    }
                });
            }
            ExprKind::Array(elems) => {
                let elems = elems.clone();
                self.nested(format!("array {at}"), |d| {
                    for e in elems {
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
                let fields: Vec<(Span, Option<ExprId>)> =
                    fields.iter().map(|f| (f.name, f.expr)).collect();
                self.nested(header, |d| {
                    for (name, expr) in fields {
                        let field_header = format!("field-init {}", d.text(name));
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
                let arms: Vec<(PatId, ExprId)> = arms.iter().map(|a| (a.pat, a.body)).collect();
                self.nested(format!("match {at}"), |d| {
                    d.expr(scrutinee);
                    for (pat, body) in arms {
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
            ExprKind::Error => {
                let line = format!("expr-error {at}");
                self.line(line);
            }
        }
    }

    fn pat(&mut self, id: PatId) {
        let node = self.ast.pat(id);
        let at = self.at(node.span);
        match &node.kind {
            PatKind::Lit(lit) => {
                let line = format!("pat-{} {at}", self.lit(lit, node.span));
                self.line(line);
            }
            PatKind::Wild => {
                let line = format!("pat-wild {at}");
                self.line(line);
            }
            PatKind::Binding(name) => {
                let line = format!("pat-binding {} {at}", self.text(*name));
                self.line(line);
            }
            PatKind::Path(path) => {
                let line = format!("pat-path {} {at}", self.path(path));
                self.line(line);
            }
            PatKind::TupleVariant { path, pats } => {
                let header = format!("pat-tuple-variant {} {at}", self.path(path));
                let pats = pats.clone();
                self.nested(header, |d| {
                    for p in pats {
                        d.pat(p);
                    }
                });
            }
            PatKind::Struct { path, fields } => {
                let header = format!("pat-struct {} {at}", self.path(path));
                let fields: Vec<(Span, Option<PatId>)> =
                    fields.iter().map(|f| (f.name, f.pat)).collect();
                self.nested(header, |d| {
                    for (name, pat) in fields {
                        let field_header = format!("field-pat {}", d.text(name));
                        d.nested(field_header, |d| {
                            if let Some(pat) = pat {
                                d.pat(pat);
                            }
                        });
                    }
                });
            }
            PatKind::Tuple(pats) => {
                let pats = pats.clone();
                self.nested(format!("pat-tuple {at}"), |d| {
                    for p in pats {
                        d.pat(p);
                    }
                });
            }
            PatKind::Array(pats) => {
                let pats = pats.clone();
                self.nested(format!("pat-array {at}"), |d| {
                    for p in pats {
                        d.pat(p);
                    }
                });
            }
        }
    }

    fn stmt(&mut self, id: StmtId) {
        let node = self.ast.stmt(id);
        let at = self.at(node.span);
        match &node.kind {
            StmtKind::Empty => {
                let line = format!("empty-stmt {at}");
                self.line(line);
            }
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
            StmtKind::Continue => {
                let line = format!("continue {at}");
                self.line(line);
            }
            StmtKind::Item(item) => {
                let item = *item;
                self.item(item);
            }
            StmtKind::Error => {
                let line = format!("stmt-error {at}");
                self.line(line);
            }
        }
    }

    fn block(&mut self, id: BlockId) {
        let node = self.ast.block(id);
        let at = self.at(node.span);
        let stmts = node.stmts.clone();
        let tail = node.tail;
        self.nested(format!("block {at}"), |d| {
            for s in stmts {
                d.stmt(s);
            }
            if let Some(tail) = tail {
                d.nested("tail".to_string(), |d| d.expr(tail));
            }
        });
    }

    fn generic_params(&mut self, generics: &[GenericParam]) {
        /// One rendered bound: (path text, its generic args).
        type BoundDump = (String, Option<GenericArgs>);
        if generics.is_empty() {
            return;
        }
        let params: Vec<(Span, Vec<BoundDump>)> = generics
            .iter()
            .map(|p| {
                (
                    p.name,
                    p.bounds
                        .iter()
                        .map(|b| (self.path(&b.path), b.args.clone()))
                        .collect(),
                )
            })
            .collect();
        self.nested("generics".to_string(), |d| {
            for (name, bounds) in params {
                let mut header = format!("param {}", d.text(name));
                if !bounds.is_empty() {
                    header.push_str(": ");
                    header.push_str(
                        &bounds
                            .iter()
                            .map(|(p, _)| p.as_str())
                            .collect::<Vec<_>>()
                            .join(" + "),
                    );
                }
                d.nested(header, |d| {
                    for (_, args) in &bounds {
                        if let Some(args) = args {
                            d.generic_args(args);
                        }
                    }
                });
            }
        });
    }

    fn fn_sig(&mut self, sig: &FnSig) {
        let header = format!("sig {} {}", self.text(sig.name), self.at(sig.span));
        let generics = sig.generics.clone();
        let receiver = sig.receiver;
        let params = sig.params.clone();
        let ret = sig.ret;
        self.nested(header, |d| {
            d.generic_params(&generics);
            if let Some(receiver) = receiver {
                d.line(format!("receiver {receiver:?}"));
            }
            for p in params {
                let param_header = format!(
                    "param{} {}",
                    if p.mutable { " mut" } else { "" },
                    d.text(p.name)
                );
                d.nested(param_header, |d| d.ty(p.ty));
            }
            match ret {
                RetTy::Unit => {}
                RetTy::Ty(ty) => d.nested("ret".to_string(), |d| d.ty(ty)),
                RetTy::Never(_) => d.line("ret never"),
            }
        });
    }

    fn use_tree(&mut self, tree: &UseTree) {
        match tree {
            UseTree::Path { path, alias } => {
                let mut line = format!("use-path {}", self.path(path));
                if let Some(alias) = alias {
                    line.push_str(" as ");
                    line.push_str(self.text(*alias));
                }
                self.line(line);
            }
            UseTree::Glob { prefix } => {
                let line = format!("use-glob {}::*", self.path(prefix));
                self.line(line);
            }
            UseTree::SelfImport { prefix } => {
                let line = format!("use-self {}::self", self.path(prefix));
                self.line(line);
            }
            UseTree::Group { prefix, items } => {
                let header = format!("use-group {}", self.path(prefix));
                let items = items.clone();
                self.nested(header, |d| {
                    for item in &items {
                        d.use_tree(item);
                    }
                });
            }
        }
    }

    fn item(&mut self, id: ItemId) {
        let node = self.ast.item(id);
        let at = self.at(node.span);
        let vis = match node.vis {
            Some(Vis::Pub) => "pub ",
            Some(Vis::Priv) => "priv ",
            None => "",
        };
        match &node.kind {
            ItemKind::Fn(def) => {
                let sig = def.sig.clone();
                let body = def.body;
                self.nested(format!("{vis}fn {at}"), |d| {
                    d.fn_sig(&sig);
                    d.block(body);
                });
            }
            ItemKind::Struct {
                name,
                generics,
                fields,
            } => {
                let header = format!("{vis}struct {} {at}", self.text(*name));
                let generics = generics.clone();
                let fields = fields.clone();
                self.nested(header, |d| {
                    d.generic_params(&generics);
                    for f in fields {
                        let fh = format!(
                            "field{} {}",
                            if f.is_pub { " pub" } else { "" },
                            d.text(f.name)
                        );
                        d.nested(fh, |d| d.ty(f.ty));
                    }
                });
            }
            ItemKind::Enum {
                name,
                generics,
                variants,
            } => {
                let header = format!("{vis}enum {} {at}", self.text(*name));
                let generics = generics.clone();
                let variants: Vec<(Span, VariantDump)> = variants
                    .iter()
                    .map(|v| {
                        (
                            v.name,
                            match &v.kind {
                                VariantKind::Unit => VariantDump::Unit,
                                VariantKind::Tuple(tys) => VariantDump::Tuple(tys.clone()),
                                VariantKind::Struct(fields) => VariantDump::Struct(fields.clone()),
                            },
                        )
                    })
                    .collect();
                self.nested(header, |d| {
                    d.generic_params(&generics);
                    for (name, kind) in variants {
                        let vh = format!("variant {}", d.text(name));
                        d.nested(vh, |d| match kind {
                            VariantDump::Unit => {}
                            VariantDump::Tuple(tys) => {
                                for ty in tys {
                                    d.ty(ty);
                                }
                            }
                            VariantDump::Struct(fields) => {
                                for f in fields {
                                    let fh = format!(
                                        "field{} {}",
                                        if f.is_pub { " pub" } else { "" },
                                        d.text(f.name)
                                    );
                                    d.nested(fh, |d| d.ty(f.ty));
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
                let generics = generics.clone();
                let entries: Vec<TraitDump> = items
                    .iter()
                    .map(|it| match it {
                        TraitItem::Method { sig, body } => TraitDump::Method(sig.clone(), *body),
                        TraitItem::AssocType { name } => TraitDump::Assoc(*name),
                    })
                    .collect();
                self.nested(header, |d| {
                    d.generic_params(&generics);
                    for entry in entries {
                        match entry {
                            TraitDump::Assoc(name) => {
                                let line = format!("assoc-type {}", d.text(name));
                                d.line(line);
                            }
                            TraitDump::Method(sig, body) => {
                                d.fn_sig(&sig);
                                if let Some(body) = body {
                                    d.block(body);
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
                let generics = generics.clone();
                let trait_args = trait_.as_ref().and_then(|t| t.args.clone());
                let self_ty = *self_ty;
                let entries: Vec<ImplDump> = items
                    .iter()
                    .map(|it| match it {
                        ImplItem::Fn { vis, def } => ImplDump::Fn(*vis, def.sig.clone(), def.body),
                        ImplItem::AssocType { name, ty } => ImplDump::Assoc(*name, *ty),
                    })
                    .collect();
                self.nested(header, |d| {
                    d.generic_params(&generics);
                    if let Some(args) = &trait_args {
                        d.nested("trait-args".to_string(), |d| d.generic_args(args));
                    }
                    d.nested("self-ty".to_string(), |d| d.ty(self_ty));
                    for entry in entries {
                        match entry {
                            ImplDump::Fn(fvis, sig, body) => {
                                let fvis = match fvis {
                                    Some(Vis::Pub) => "pub ",
                                    Some(Vis::Priv) => "priv ",
                                    None => "",
                                };
                                d.nested(format!("{fvis}method"), |d| {
                                    d.fn_sig(&sig);
                                    d.block(body);
                                });
                            }
                            ImplDump::Assoc(name, ty) => {
                                let header = format!("assoc-type {} =", d.text(name));
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
                let generics = generics.clone();
                let ty = *ty;
                self.nested(header, |d| {
                    d.generic_params(&generics);
                    d.ty(ty);
                });
            }
            ItemKind::Use(tree) => {
                let tree = tree.clone();
                self.nested(format!("{vis}use {at}"), |d| d.use_tree(&tree));
            }
            ItemKind::Mod { name, items } => {
                let header = format!("{vis}mod {} {at}", self.text(*name));
                let items = items.clone();
                self.nested(header, |d| match items {
                    Some(items) => {
                        for item in items {
                            d.item(item);
                        }
                    }
                    None => d.line("external"),
                });
            }
        }
    }
}

enum VariantDump {
    Unit,
    Tuple(Vec<TypeId>),
    Struct(Vec<FieldDef>),
}

enum TraitDump {
    Method(FnSig, Option<BlockId>),
    Assoc(Span),
}

enum ImplDump {
    Fn(Option<Vis>, FnSig, BlockId),
    Assoc(Span, TypeId),
}
