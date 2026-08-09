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
use crate::source::Span;

macro_rules! arena_id {
    ($Id:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
        pub struct $Id(pub u32);
    };
}

arena_id!(TypeId);
arena_id!(ExprId);
arena_id!(StmtId);
arena_id!(ItemId);
arena_id!(PatId);
arena_id!(BlockId);
arena_id!(DimId);
// DEV-173: `StrLitId` indexes a string literal's DECODED value in `Ast::str_lits`.
arena_id!(StrLitId);

/// WP-FMT-001: arena high-water marks, so every node a sub-parse creates can be found afterwards.
#[derive(Clone, Copy)]
pub struct ArenaMarks {
    types: usize,
    exprs: usize,
    stmts: usize,
    pats: usize,
    blocks: usize,
}

#[derive(Default)]
pub struct Ast {
    pub types: Vec<TypeNode>,
    pub exprs: Vec<ExprNode>,
    pub stmts: Vec<StmtNode>,
    pub items: Vec<ItemNode>,
    pub pats: Vec<PatNode>,
    pub blocks: Vec<BlockNode>,
    /// Dimension expressions inside tensor shape arguments (`tensor`
    /// extension, D2/D5). Empty for Core-only programs.
    pub dims: Vec<DimExprNode>,
    pub root: Root,
    /// AS1b-ii-b: the source each item was parsed from, by IDENTITY.
    ///
    /// This was `ItemId -> Arc<SourceFile>`, which made it a second source *authority* alongside
    /// the registry: consumers selected a file from here and indexed spans against it. An id
    /// cannot be a rival authority — it can only name what the registry already decided.
    pub item_sources: std::collections::HashMap<ItemId, crate::source::SourceId>,
    /// Every source this AST was parsed from, interned in load order (AS1b-i).
    ///
    /// This is where `SourceId`s come from. Allocation used to happen in `build_source_map`, after
    /// the whole front end had run, which meant a span could not carry the identity of the file it
    /// indexes. Files are registered here as the parser loads them, so identity exists from the
    /// moment source does.
    pub sources: crate::source::SourceRegistry,
    /// AS1b-ii-d: names of items the compiler synthesised, keyed by ITEM.
    ///
    /// Dependency-package `mod` wrappers have no source text. Their names used to be encoded as
    /// spans at `lo >= 0x8000_0000` — a name wearing a location's clothes, which forced every span
    /// consumer to know that some spans index no file, and blocked span→location resolution from
    /// ever being total. A name is not a location, so it is no longer stored as one.
    pub synthetic_names: std::collections::HashMap<ItemId, String>,
    /// DEV-173: every string literal's DECODED value, in allocation order.
    ///
    /// A literal used to be re-decoded from its own source span on demand. That works only while
    /// a literal's span reads back as its own source — which an interpolation field breaks, since
    /// a nested string literal there is written `\"a\"` and carries the ENCLOSING literal's
    /// escapes. Decoding is done once, at parse time, from whatever buffer the parser was reading;
    /// spans are then purely diagnostic.
    pub str_lits: Vec<String>,
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
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
    /// `Float16` (IEEE 754 binary16) — `tensor` extension element type (D3).
    Float16,
    /// `BFloat16` (bfloat16) — `tensor` extension element type (D3).
    BFloat16,
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
            Primitive::Float16 => "Float16",
            Primitive::BFloat16 => "BFloat16",
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
    /// Integer const generic argument, currently used by `Cuda<N>`.
    Const(Span),
    /// `Item = T` associated-type binding. The `tensor` extension also uses
    /// this form for the `device = D` argument (§8); resolution distinguishes
    /// `device` from associated-type names.
    Binding {
        name: Span,
        ty: TypeId,
    },
    /// `[DimExpr, ...]` / `[]` shape argument, or a const index list such as
    /// `[0, 2, 1]` (`tensor` extension deltas D2/D5). The two share this
    /// surface form and are disambiguated semantically by the operation's
    /// signature (spec §6.4).
    Shape(ShapeArg),
}

/// A `[DimExpr, ...]` shape argument (D2) or const index list (D5).
#[derive(Clone)]
pub struct ShapeArg {
    pub dims: Vec<DimId>,
    pub span: Span,
}

// -------------------------------------------------- dimension expressions --

/// A node in a dimension expression (`tensor` extension, §3.2). Dimension
/// expressions are polynomials over dim variables; their algebra lives in the
/// extension's semantic layer (M4.2), not here.
pub struct DimExprNode {
    pub kind: DimExprKind,
    pub span: Span,
}

pub enum DimExprKind {
    /// An integer literal dimension (the span covers the digits).
    Lit(Span),
    /// A dimension variable (identifier) — resolved in semantic analysis.
    Var(Span),
    /// `lhs (+|-|*) rhs`.
    Binary {
        op: DimBinOp,
        lhs: DimId,
        rhs: DimId,
    },
    /// Placeholder for a dimension expression that failed to parse.
    Error,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DimBinOp {
    Add,
    Sub,
    Mul,
}

impl DimBinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            DimBinOp::Add => "+",
            DimBinOp::Sub => "-",
            DimBinOp::Mul => "*",
        }
    }
}

// ---------------------------------------------------------- expressions --

pub struct ExprNode {
    pub kind: ExprKind,
    pub span: Span,
}

pub enum ExprKind {
    Lit(Lit),
    /// WP-FMT-001: `f"pkg={name} n={count:04}"`.
    ///
    /// Not a macro, not a call and not a runtime-parsed format string: the segments below were
    /// split at COMPILE TIME, and the expression in each field is an ordinary `ExprId` parsed by
    /// the ordinary expression parser. Evaluating this expression produces an owned `String`.
    FormatString {
        segments: Vec<FormatSegment>,
    },
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
        /// Optional method turbofish (`value.method::<...>`).
        turbofish: Option<GenericArgs>,
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

// ------------------------------------------------- WP-FMT-001: interpolated string literals --

/// How a rendered field sits inside a wider one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FormatAlign {
    Left,
    Right,
    Center,
}

/// What prefix a non-negative number carries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FormatSign {
    Plus,
    Minus,
    Space,
}

/// The value-family conversion a field asks for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FormatKind {
    Bin,
    Oct,
    LowerHex,
    UpperHex,
    Fixed,
}

/// A field's format specification — everything after the top-level `:`.
///
/// Every part is a COMPILE-TIME constant. There is no dynamic width and no dynamic precision in
/// v0.1, which is what lets the whole specification be packed into a constant and lets a bad
/// combination be refused by the type checker rather than discovered at run time.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct FormatSpec {
    pub fill: Option<char>,
    pub align: Option<FormatAlign>,
    pub sign: Option<FormatSign>,
    pub alternate: bool,
    pub zero_pad: bool,
    pub width: Option<u32>,
    pub precision: Option<u32>,
    pub kind: Option<FormatKind>,
    /// The specification's own span (after the `:`), or `None` when none was written. Diagnostics
    /// about a bad type/spec pairing point HERE, not at the whole literal.
    pub span: Option<Span>,
}

impl FormatSpec {
    /// Whether this specification asks for anything beyond padding. A padding-only specification
    /// applies to any `Display` value; anything else constrains the type.
    pub fn is_padding_only(&self) -> bool {
        self.sign.is_none()
            && !self.alternate
            && !self.zero_pad
            && self.precision.is_none()
            && self.kind.is_none()
    }
}

/// One piece of an interpolated string literal.
#[derive(Clone, Debug)]
pub enum FormatSegment {
    /// Static text, with escapes and `{{`/`}}` already resolved.
    Literal { text: String, span: Span },
    /// `{ expression [: spec] }`.
    Field {
        expr: ExprId,
        spec: FormatSpec,
        /// The whole field including its braces.
        span: Span,
        /// The expression alone, for diagnostics that blame the value rather than the field.
        expr_span: Span,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
        /// DEV-173: the decoded value, resolved at parse time (see `Ast::str_lits`).
        value: StrLitId,
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
    /// `model Name<...> { input ...; output ...; }` (`tensor` extension, D4).
    Model(ModelDef),
}

/// A nominal `model` declaration (`tensor` extension, §7.1). Every generic
/// parameter is a `Dim` parameter (checked semantically); every port type is
/// a `Tensor`/`TensorDyn` type.
pub struct ModelDef {
    pub name: Span,
    pub generics: Vec<GenericParam>,
    pub ports: Vec<ModelPort>,
}

pub struct ModelPort {
    pub dir: PortDir,
    pub name: Span,
    pub ty: TypeId,
    pub span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PortDir {
    Input,
    Output,
}

impl PortDir {
    /// AS6: the spelling belongs to the extension that owns model ports, and the parser matches
    /// against the same table, so parse and print cannot drift.
    pub fn keyword(self) -> &'static str {
        crate::extensions::tensor::syntax::port_keyword(self)
    }
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
    /// Intern `file` and hand back the shared `Arc`.
    ///
    /// Callers hold a `&SourceFile` (the parser borrows one), so this copies it exactly once, on
    /// first sight, and returns the same `Arc` on every later call for that name.
    /// The source an item was parsed from. Mirrors `Hir::item_file`: the map names an id, the
    /// registry answers, and there is no second authority to disagree with it.
    pub fn item_file(&self, item: ItemId) -> Option<&std::sync::Arc<crate::source::SourceFile>> {
        self.item_sources
            .get(&item)
            .and_then(|id| self.sources.get(*id))
            .map(|source| source.file())
    }

    pub fn interned_source(
        &mut self,
        file: &crate::source::SourceFile,
    ) -> crate::source::RegisteredSource {
        if let Some(id) = self.sources.id_for_name(&file.name) {
            return self.sources.get(id).expect("just looked it up").clone();
        }
        let mut owned = crate::source::SourceFile::new(file.name.clone(), file.src.clone());
        if let Some(path) = &file.disk_path {
            owned = owned.with_disk_path(path.clone());
        }
        self.sources.intern(std::sync::Arc::new(owned))
    }

    /// WP-FMT-001: the current arena sizes, for [`Ast::remap_spans_since`].
    pub fn marks(&self) -> ArenaMarks {
        ArenaMarks {
            types: self.types.len(),
            exprs: self.exprs.len(),
            stmts: self.stmts.len(),
            pats: self.pats.len(),
            blocks: self.blocks.len(),
        }
    }

    /// DEV-173: translate every span allocated since `marks` from a scratch buffer's offsets back
    /// to the real file's, through `map`.
    ///
    /// The decoded sub-parse of an interpolation field reads a buffer whose offsets are its own.
    /// `map[i]` is the file offset the decoded byte `i` came from, so remapping restores REAL
    /// spans — a diagnostic inside such a field points at the sub-expression, not at the field.
    ///
    /// An earlier version collapsed every span to the field's instead. That was not merely coarse:
    /// a literal read its value from its span, so collapsing made a nested string literal read the
    /// whole field's source back. Values now come from `str_lits`, and spans are only ever
    /// locations.
    pub fn remap_spans_since(&mut self, marks: ArenaMarks, map: &[u32]) {
        let remap = |span: &mut Span| {
            let lo = map.get(span.lo as usize).copied();
            let hi = map.get(span.hi as usize).copied();
            if let (Some(lo), Some(hi)) = (lo, hi) {
                *span = Span::in_source(span.source, lo, hi);
            }
        };
        for node in &mut self.types[marks.types..] {
            remap(&mut node.span);
        }
        for node in &mut self.exprs[marks.exprs..] {
            remap(&mut node.span);
        }
        for node in &mut self.stmts[marks.stmts..] {
            remap(&mut node.span);
        }
        for node in &mut self.pats[marks.pats..] {
            remap(&mut node.span);
        }
        for node in &mut self.blocks[marks.blocks..] {
            remap(&mut node.span);
        }
    }

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
    pub fn alloc_dim(&mut self, kind: DimExprKind, span: Span) -> DimId {
        self.dims.push(DimExprNode { kind, span });
        DimId(self.dims.len() as u32 - 1)
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
    pub fn dim(&self, id: DimId) -> &DimExprNode {
        &self.dims[id.0 as usize]
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

/// DEV-214 (OD-9) — the deepest expression tree in this AST, computed **iteratively**.
///
/// # Why this exists
///
/// The parser bounds recursion with [`crate::parser::MAX_DEPTH`], which is what stops
/// `(((((...1...)))))` from overflowing the stack: each nesting level is a recursive call, so the
/// counter rises with the nesting.
///
/// A left-associative operator chain never recurses in the parser. The precedence table is
/// implemented as one `loop` per level, so `1 + 1 + 1 + ...` folds iteratively and the counter
/// never moves — **but the tree it builds is as deep as the chain is long.** Every recursive walk
/// downstream (type checking, and the index building in `analyze_project`) then descends that
/// depth and the process dies with a stack overflow, at ~65 terms on a 2 MiB thread stack and
/// ~250 on 8 MiB.
///
/// The bug was never the limit's VALUE. It was that the limit measured *the nesting the parser
/// recursed through* rather than *the depth of the tree that nesting produced*. This function
/// measures the second, against the same limit.
///
/// # Why the measurement itself is iterative
///
/// A recursive depth-measuring pass would overflow on exactly the input it exists to reject —
/// the guard would die measuring the thing it guards. So this is a forward dynamic-programming
/// pass over the expression arena:
///
/// ```text
/// depth[i] = 1 + max(depth[child] for child in children(i))
/// ```
///
/// A child is allocated before its parent (the parser builds sub-expressions first, and
/// [`Ast::alloc_expr`] pushes), so one left-to-right pass normally suffices. That ordering is a
/// property of the parser rather than of the type, so it is **iterated to a fixpoint instead of
/// assumed** — bounded by `MAX_PASSES`, after which the answer is reported as saturated rather
/// than guessed. No recursion at any point.
///
/// # Blocks are deliberately not traversed
///
/// `if`/`while`/`loop`/`for` bodies are [`BlockId`]s, and block nesting *does* raise the parser's
/// counter, so it is already bounded. Every expression in every block still appears in this arena
/// and is scored, so a deep chain inside a block is caught wherever it sits — what is not counted
/// is the block nesting itself, because it is not this guard's gap to close.
impl Ast {
    /// [`max_expr_depth`] as a method, for the one call site in `analyze_project`.
    pub fn max_expr_depth_of_program(&self) -> (u32, Option<ExprId>) {
        max_expr_depth(self)
    }
}

pub fn max_expr_depth(ast: &Ast) -> (u32, Option<ExprId>) {
    /// One pass is enough when children precede parents. A second proves the fixpoint. The rest
    /// are headroom for an allocation order this function does not want to depend on.
    const MAX_PASSES: usize = 4;

    let n = ast.exprs.len();
    if n == 0 {
        return (0, None);
    }
    let mut depth = vec![1u32; n];

    for _ in 0..MAX_PASSES {
        let mut changed = false;
        for i in 0..n {
            let mut best = 0u32;
            each_child_expr(&ast.exprs[i].kind, |child| {
                let c = child.0 as usize;
                if c < n {
                    best = best.max(depth[c]);
                }
            });
            let candidate = best.saturating_add(1);
            if candidate > depth[i] {
                depth[i] = candidate;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // The deepest node's span is what a diagnostic should blame -- "somewhere in this file" is
    // not a useful answer for an expression the reader has to find and break up.
    let mut best = (0u32, None);
    for (i, d) in depth.into_iter().enumerate() {
        if d > best.0 {
            best = (d, Some(ExprId(i as u32)));
        }
    }
    best
}

/// Every directly-nested expression of one expression, for [`max_expr_depth`].
///
/// **There is deliberately no `_ =>` arm.** A catch-all would silently score a new variant's
/// children as absent, and the guard would quietly stop guarding whatever that variant nests —
/// which is the shape `AS8-DA-006` records as "the sixth `MirTy` catch-all to swallow this
/// variant". Adding an `ExprKind` variant must be a compile error here.
fn each_child_expr(kind: &ExprKind, mut f: impl FnMut(ExprId)) {
    match kind {
        // Leaves, and the block-carrying forms whose nesting the parser already bounds.
        ExprKind::Lit(_)
        | ExprKind::Path { .. }
        | ExprKind::Loop { .. }
        | ExprKind::Block(_)
        | ExprKind::Error => {}
        // A format string's fields are parsed as their own expressions.
        ExprKind::FormatString { segments } => {
            for seg in segments {
                // No catch-all here either, for the same reason as the outer match.
                match seg {
                    FormatSegment::Literal { .. } => {}
                    FormatSegment::Field { expr, .. } => f(*expr),
                }
            }
        }
        ExprKind::Unary { operand, .. } => f(*operand),
        ExprKind::Binary { lhs, rhs, .. } => {
            f(*lhs);
            f(*rhs);
        }
        ExprKind::Assign { lhs, rhs, .. } => {
            f(*lhs);
            f(*rhs);
        }
        ExprKind::Range { lo, hi, .. } => {
            f(*lo);
            f(*hi);
        }
        ExprKind::Cast { expr, .. } => f(*expr),
        ExprKind::Call { callee, args } => {
            f(*callee);
            for a in args {
                f(*a);
            }
        }
        ExprKind::Field { base, .. } => f(*base),
        ExprKind::TupleField { base, .. } => f(*base),
        ExprKind::Index { base, index } => {
            f(*base);
            f(*index);
        }
        ExprKind::Try(e) => f(*e),
        ExprKind::Tuple(items) | ExprKind::Array(items) => {
            for e in items {
                f(*e);
            }
        }
        ExprKind::Repeat { value, count } => {
            f(*value);
            f(*count);
        }
        ExprKind::StructLit { fields, .. } => {
            for field in fields {
                // Shorthand (`S { x }`) carries no expression of its own.
                if let Some(e) = field.expr {
                    f(e);
                }
            }
        }
        ExprKind::If { cond, else_, .. } => {
            f(*cond);
            if let Some(e) = else_ {
                f(*e);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            f(*scrutinee);
            for arm in arms {
                f(arm.body);
            }
        }
        ExprKind::While { cond, .. } => f(*cond),
        ExprKind::For { iter, .. } => f(*iter),
    }
}
