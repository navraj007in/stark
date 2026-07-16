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
    pub item_files: std::collections::HashMap<ItemId, std::sync::Arc<crate::source::SourceFile>>,
    pub synthetic_spans: std::collections::HashMap<Span, String>,
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
    pub fn keyword(self) -> &'static str {
        match self {
            PortDir::Input => "input",
            PortDir::Output => "output",
        }
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
