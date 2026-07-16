//! High-Level Intermediate Representation (HIR) for STARK.
//!
//! Per PLAN.md M2.1: AST is lowered into this desugared HIR. Name resolution,
//! type checking, and all subsequent passes operate on HIR, never on the parser AST.

use crate::ast::{AssignOp, BinOp, Lit, Path, Primitive, UnOp, Vis};
use crate::source::Span;

macro_rules! hir_id {
    ($Id:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
        pub struct $Id(pub u32);
    };
}

hir_id!(TypeId);
hir_id!(ExprId);
hir_id!(StmtId);
hir_id!(ItemId);
hir_id!(PatId);
hir_id!(BlockId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

/// Compiler-provided functions available before the Core standard library is
/// loaded. These are not HIR items and must never be represented by a fake
/// `ItemId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Builtin {
    Print,
    Println,
    Panic,
    Assert,
    Sqrt,
    Drop,
    StringFrom,
    StringNew,
    StringWithCapacity,
    VecNew,
    VecWithCapacity,
    BoxNew,
    BoxIntoInner,
    ReadFile,
    WriteFile,
    Some,
    None,
    Ok,
    Err,
    TensorZeros,
    TensorOnes,
    TensorFull,
    TensorFromVec,
    TensorAdd,
    TensorSub,
    TensorMul,
    TensorDiv,
    TensorMin,
    TensorMax,
    TensorEq,
    TensorNe,
    TensorLt,
    TensorLe,
    TensorGt,
    TensorGe,
    TensorBroadcastTo,
    TensorMatMul,
    TensorBatchMatMul,
    TensorConcat,
    TensorPermute,
    TensorReshape,
    TensorSliceAxis,
    TensorTranspose,
    TensorSumAxis,
    TensorMeanAxis,
    TensorArgMax,
    TensorSum,
    TensorSoftmax,
    TensorCast,
    TensorToDevice,
    /// `scale_255()` — value-range transition ByteRange -> UnitRange (Gate 7).
    TensorScale255,
    /// `normalize()` — value-range transition UnitRange -> Normalized (Gate 7).
    TensorNormalize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoreType {
    String,
    Vec,
    Box,
    Option,
    Result,
    Range,
    RangeInclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoreTrait {
    Copy,
    Drop,
    Eq,
    Ord,
    Num,
}

/// Target of a resolved name or path segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Res {
    /// Function-local variable or parameter.
    Local(LocalId),
    /// Top-level item in a module/package.
    Item(ItemId),
    /// Enum variant.
    Variant(ItemId, u32),
    /// Method or associated type selected from a trait path.
    TraitMember(ItemId, u32),
    /// Receiverless function selected from an inherent impl.
    AssociatedFn(ItemId, Span),
    /// Primitive type.
    Primitive(Primitive),
    /// The `Self` type inside an impl/trait.
    SelfType,
    /// An associated type projection written as `Self::Name`.
    SelfAssoc(Span),
    /// A generic type parameter (like `T`).
    TypeParam,
    /// An associated type projected from a generic parameter (`T::Item`).
    ParamAssoc(Span, Span),
    /// The `self` value parameter in a method.
    SelfValue(LocalId),
    /// A compiler-provided function, distinct from an arena-backed item.
    Builtin(Builtin),
    /// A compiler-known Core marker trait supplied by the prelude.
    CoreTrait(CoreTrait),
    /// A nominal type supplied by the Core prelude.
    CoreType(CoreType),
    /// The load associated function for a nominal model type.
    ModelLoad(ItemId),
    /// Unresolved or error name (prevents cascading diagnostics).
    Err,
}

#[derive(Default)]
pub struct Hir {
    pub types: Vec<TypeNode>,
    pub exprs: Vec<ExprNode>,
    pub stmts: Vec<StmtNode>,
    pub items: Vec<ItemNode>,
    pub pats: Vec<PatNode>,
    pub blocks: Vec<BlockNode>,
    pub root: Root,
}

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

// ----------------------------------------------------------------- types --

pub struct TypeNode {
    pub kind: TypeKind,
    pub span: Span,
}

pub enum TypeKind {
    Primitive(Primitive),
    Path {
        path: Path,
        res: Res,
        args: Option<GenericArgs>,
    },
    Array {
        elem: TypeId,
        len: Span,
    },
    Slice(TypeId),
    Tuple(Vec<TypeId>),
    Ref {
        mutable: bool,
        inner: TypeId,
    },
    Fn {
        params: Vec<TypeId>,
        ret: Option<TypeId>,
    },
    Never,
    Error,
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
    Binding {
        name: Span,
        ty: TypeId,
    },
    /// `[DimExpr, ...]` shape / index-list argument (`tensor` extension,
    /// D2/D5). Dimension variables are carried as name spans; their kind and
    /// polynomial semantics are the extension checker's concern (M4.2+).
    Shape(ShapeArg),
}

#[derive(Clone)]
pub struct ShapeArg {
    pub dims: Vec<DimExpr>,
    pub span: Span,
}

#[derive(Clone)]
pub enum DimExpr {
    Lit(Span),
    Var(Span),
    Binary {
        op: crate::ast::DimBinOp,
        lhs: Box<DimExpr>,
        rhs: Box<DimExpr>,
    },
    Error,
}

// ----------------------------------------------------------- expressions --

pub struct ExprNode {
    pub kind: ExprKind,
    pub span: Span,
}

pub enum ExprKind {
    Lit(Lit),
    Path {
        path: Path,
        res: Res,
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
    Assign {
        op: AssignOp,
        lhs: ExprId,
        rhs: ExprId,
    },
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
    Field {
        base: ExprId,
        name: Span,
        turbofish: Option<GenericArgs>,
    },
    TupleField {
        base: ExprId,
        index: Span,
    },
    Index {
        base: ExprId,
        index: ExprId,
    },
    Try(ExprId),
    Tuple(Vec<ExprId>),
    Array(Vec<ExprId>),
    Repeat {
        value: ExprId,
        count: ExprId,
    },
    StructLit {
        path: Path,
        res: Res,
        fields: Vec<FieldInit>,
    },
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
        local: LocalId,
        iter: ExprId,
        body: BlockId,
    },
    Block(BlockId),
    Error,
}

pub struct FieldInit {
    pub name: Span,
    pub expr: Option<ExprId>,
}

pub struct MatchArm {
    pub pat: PatId,
    pub body: ExprId,
}

// ------------------------------------------------------------- statements --

pub struct StmtNode {
    pub kind: StmtKind,
    pub span: Span,
}

pub enum StmtKind {
    Empty,
    Expr {
        expr: ExprId,
        semi: bool,
    },
    Let {
        mutable: bool,
        name: Span,
        local: LocalId,
        ty: Option<TypeId>,
        init: Option<ExprId>,
    },
    Return(Option<ExprId>),
    Break(Option<ExprId>),
    Continue,
    Item(ItemId),
    Error,
}

// ----------------------------------------------------------------- blocks --

pub struct BlockNode {
    pub stmts: Vec<StmtId>,
    pub tail: Option<ExprId>,
    pub span: Span,
}

// ------------------------------------------------------------------ items --

pub struct ItemNode {
    pub kind: ItemKind,
    pub vis: Option<Vis>,
    pub span: Span,
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
    Mod {
        name: Span,
        items: Option<Vec<ItemId>>,
    },
    /// `model Name<...> { ... }` (`tensor` extension, D4). Full validation is
    /// deferred to the extension checker (M4.4); the front end only needs a
    /// span-preserving, name-resolved representation.
    Model(ModelDef),
}

pub struct ModelDef {
    pub name: Span,
    pub generics: Vec<GenericParam>,
    pub ports: Vec<ModelPort>,
}

pub struct ModelPort {
    pub dir: crate::ast::PortDir,
    pub name: Span,
    pub ty: TypeId,
    pub span: Span,
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
    pub receiver_local: Option<LocalId>,
    pub params: Vec<Param>,
    pub ret: RetTy,
    pub span: Span,
}

#[derive(Clone, Copy)]
pub enum RetTy {
    Unit,
    Ty(TypeId),
    Never(Span),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Receiver {
    Value,
    Ref,
    RefMut,
}

#[derive(Clone, Copy)]
pub struct Param {
    pub mutable: bool,
    pub name: Span,
    pub ty: TypeId,
    pub local: LocalId,
}

#[derive(Clone)]
pub struct GenericParam {
    pub name: Span,
    pub bounds: Vec<TraitRef>,
}

#[derive(Clone)]
pub struct TraitRef {
    pub path: Path,
    pub res: Res,
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
    Method { sig: FnSig, body: Option<BlockId> },
    AssocType { name: Span },
}

pub enum ImplItem {
    Fn { vis: Option<Vis>, def: FnDef },
    AssocType { name: Span, ty: TypeId },
}

#[derive(Clone)]
pub enum UseTree {
    Path { path: Path, alias: Option<Span> },
    Glob { prefix: Path },
    SelfImport { prefix: Path },
    Group { prefix: Path, items: Vec<UseTree> },
}

// --------------------------------------------------------------- arena --

impl Hir {
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

// ------------------------------------------------------------- patterns --

pub struct PatNode {
    pub kind: PatKind,
    pub span: Span,
}

pub enum PatKind {
    Lit(Lit),
    Wild,
    Binding {
        name: Span,
        local: LocalId,
    },
    Path {
        path: Path,
        res: Res,
    },
    TupleVariant {
        path: Path,
        res: Res,
        pats: Vec<PatId>,
    },
    Struct {
        path: Path,
        res: Res,
        fields: Vec<FieldPat>,
    },
    Tuple(Vec<PatId>),
    Array(Vec<PatId>),
    Error,
}

pub struct FieldPat {
    pub name: Span,
    pub pat: Option<PatId>,
    /// Binding allocated for shorthand fields such as `Point { x }`.
    pub local: Option<LocalId>,
}
