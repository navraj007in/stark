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
    AssertEq,
    AssertNe,
    Sqrt,
    Drop,
    StringFrom,
    StringNew,
    StringWithCapacity,
    CharFromU32,
    VecNew,
    VecWithCapacity,
    BoxNew,
    BoxIntoInner,
    ReadFile,
    WriteFile,
    FileOpen,
    FileCreate,
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
    SizeOf,
    AlignOf,
    Swap,
    Replace,
    Take,
    HashMapNew,
    HashMapWithCapacity,
    HashSetNew,
    // -- Phase 4E: Math, Random, I/O (`06-Standard-Library.md`) --
    MathPi,
    MathE,
    MathAbs,
    MathMin,
    MathMax,
    MathClamp,
    Pow,
    Log,
    Log10,
    Exp,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Floor,
    Ceil,
    Round,
    Trunc,
    Eprint,
    Eprintln,
    RandomNew,
    OrderingLess,
    OrderingEqual,
    OrderingGreater,
    IOErrorNotFound,
    IOErrorPermissionDenied,
    IOErrorAlreadyExists,
    IOErrorInvalidInput,
    IOErrorOther,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoreType {
    String,
    Vec,
    Box,
    Option,
    Result,
    Range,
    RangeInclusive,
    CharsIter,
    SplitIter,
    VecIter,
    HashMap,
    HashSet,
    KeysIter,
    ValuesIter,
    Iter,
    MapIter,
    FilterIter,
    Random,
    IOError,
    File,
    /// WP-C2.2 (DEV-027): `Ordering` is a normative prelude member
    /// (`06-Standard-Library.md` line 585, `enum Ordering { Less, Equal, Greater }`) required
    /// by the `Ord` trait's `cmp` signature; previously unresolvable anywhere in the compiler.
    Ordering,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoreTrait {
    Copy,
    Drop,
    Eq,
    Ord,
    Num,
    Clone,
    Hash,
    Default,
    Display,
    Error,
    From,
    Into,
    TryFrom,
    Index,
    IndexMut,
    Iterator,
    FromIterator,
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
    /// DEV-052: a method selected from a compiler-known `CoreTrait` path (`Eq::eq`, `Ord::cmp`,
    /// ...), distinct from `TraitMember` because a `CoreTrait` has no `hir::ItemKind::Trait`
    /// declaration item to index a member into -- the method-name segment's own span is carried
    /// instead, resolved to text on demand (the same idiom as `SelfAssoc`/`ParamAssoc`).
    CoreTraitMember(CoreTrait, Span),
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
    /// AS1b-ii-b: the source each item was parsed from, by identity (see `Ast::item_sources`).
    pub item_sources: std::collections::HashMap<ItemId, crate::source::SourceId>,
    /// Every source this program was parsed from, frozen after parsing.
    ///
    /// Carried here so that every phase holding a `&Hir` — type checking, borrow checking, MIR
    /// lowering, execution — can resolve a `SourceId` without being handed a separate lookup. It
    /// is the read-only authority the interpreter's ambient `self.file` used to stand in for.
    pub sources: crate::source::SourceRegistry,
    /// WP-C6.2b-F1: the defining module id of each item, so the type checker can enforce member
    /// and field visibility (private is exact-module, matching `resolve::item_is_visible_from`).
    pub item_modules: std::collections::HashMap<ItemId, u32>,
    pub publicly_nameable_items: std::collections::HashSet<ItemId>,
    /// C4.5f-3c: names for synthetic spans (dependency-package `mod` wrappers use spans at
    /// `lo >= 0x8000_0000` that index no real file). Copied from the AST so consumers past
    /// resolution (MIR lowering's module-path walk) can read them.
    pub synthetic_spans: std::collections::HashMap<crate::source::Span, String>,
    /// DEV-173: every string literal's decoded value, copied from the AST. A `Lit::Str` names its
    /// entry, so no pass re-decodes a literal from its span.
    pub str_lits: Vec<String>,
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
    /// WP-FMT-001: an interpolated string literal, already split into segments at parse time.
    /// Evaluates to an owned `String`.
    FormatString {
        segments: Vec<FormatSegment>,
    },
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

/// WP-FMT-001: one piece of an interpolated string literal.
#[derive(Clone)]
pub enum FormatSegment {
    /// Static text, escapes and `{{`/`}}` already resolved.
    Literal { text: String, span: Span },
    /// `{ expression [: spec] }`. The specification is a compile-time constant; `expr_span` blames
    /// the value and `spec.span` blames the specification, so a diagnostic can point at whichever
    /// half is wrong.
    Field {
        expr: ExprId,
        spec: crate::ast::FormatSpec,
        span: Span,
        expr_span: Span,
    },
}

#[derive(Clone)]
pub struct TraitRef {
    pub path: Path,
    pub res: Res,
    pub args: Option<GenericArgs>,
}

/// DEV-BOUND-TRAIT-IDENTITY: the trait a written bound denotes.
///
/// Core v1 has two kinds of trait — one with a declaration item, one compiler-known with none —
/// and both are reachable from a generic bound. This is their common identity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoundTrait {
    User(ItemId),
    Core(CoreTrait),
}

/// **The single answer to "which trait does this bound denote".**
///
/// Every pass downstream of name resolution reads the identity from [`TraitRef::res`], which the
/// resolver already computed against the bound's own module, imports and path. Nothing here looks
/// at how the bound was SPELLED.
///
/// DEV-BOUND-TRAIT-IDENTITY exists because two passes did look at the spelling. `typecheck`'s
/// `resolve_bound_trait` and `borrowck`'s `bound_method_receiver` each took
/// `text(bound.path.span)` and scanned every HIR item for a trait whose declared name matched.
/// Three failures followed, all reproduced before the repair:
///
/// * **A qualified bound never matched.** `T: traits::Render` compared the string
///   `"traits::Render"` against the declaration's name `"Render"`, so the bound contributed no
///   methods and `value.render()` was rejected.
/// * **An unrelated trait could win.** `mod unrelated { pub trait Display { fn other(&self); } }`
///   anywhere in the program captured every `T: Display` bound, because the search found a user
///   trait of that spelling and preferred it — so `x.fmt()` failed against the Core trait the
///   resolver had actually selected.
/// * **Declaration order decided ownership.** With two same-named traits, one taking `&self` and
///   one taking `self`, the borrow checker returned whichever appeared first in HIR item order.
///   The same program compiled or failed E0100 depending on the order its traits were written in.
///
/// A `TraitRef` whose `res` is not a trait yields `None` rather than a guess. That is a real
/// possibility — the resolver leaves `Res::Err` for a bound it could not resolve, and reports it
/// there — and a caller must treat it as "this bound contributes nothing", never as an invitation
/// to fall back to the spelling.
pub fn resolved_bound_trait(hir: &Hir, bound: &TraitRef) -> Option<BoundTrait> {
    bound_trait_of_res(hir, bound.res)
}

/// The trait identity a resolution names, if it names one.
///
/// Split out of [`resolved_bound_trait`] so a caller holding only a `Res` — a bound obligation
/// being discharged, which is carried as a resolution rather than as a `TraitRef` — asks the same
/// question and gets the same answer.
pub fn bound_trait_of_res(hir: &Hir, res: Res) -> Option<BoundTrait> {
    match res {
        Res::Item(item_id) => match &hir.item(item_id).kind {
            ItemKind::Trait { .. } => Some(BoundTrait::User(item_id)),
            // A bound position naming a non-trait item is already reported by the checker's own
            // bound validation; it contributes no methods here.
            ItemKind::Fn(_)
            | ItemKind::Struct { .. }
            | ItemKind::Enum { .. }
            | ItemKind::Impl { .. }
            | ItemKind::Const { .. }
            | ItemKind::TypeAlias { .. }
            | ItemKind::Use(_)
            | ItemKind::Mod { .. }
            | ItemKind::Model(_) => None,
        },
        Res::CoreTrait(core_trait) => Some(BoundTrait::Core(core_trait)),
        // Not a trait. `Res::Err` is an already-reported resolution failure; the rest cannot
        // appear in a bound position at all. None of them may be reconstructed from spelling.
        Res::Err
        | Res::Local(_)
        | Res::SelfValue(_)
        | Res::SelfType
        | Res::SelfAssoc(_)
        | Res::TypeParam
        | Res::ParamAssoc(_, _)
        | Res::Primitive(_)
        | Res::Builtin(_)
        | Res::CoreType(_)
        | Res::CoreTraitMember(_, _)
        | Res::TraitMember(_, _)
        | Res::Variant(_, _)
        | Res::AssociatedFn(_, _)
        | Res::ModelLoad(_) => None,
    }
}

/// The receiver form trait `trait_id` declares for `method`, or `None` if it declares no such
/// method. `name_of` reads a span against the TRAIT's declaring file (DEV-069 provenance), which
/// only the caller can do.
pub fn trait_method_receiver(
    hir: &Hir,
    trait_id: ItemId,
    method: &str,
    name_of: impl Fn(ItemId, Span) -> String,
) -> Option<Receiver> {
    let ItemKind::Trait { items, .. } = &hir.item(trait_id).kind else {
        return None;
    };
    items.iter().find_map(|trait_item| match trait_item {
        TraitItem::Method { sig, .. } if name_of(trait_id, sig.name) == method => sig.receiver,
        TraitItem::Method { .. } | TraitItem::AssocType { .. } => None,
    })
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
    /// The registered source with this logical name, if this program was parsed from it.
    ///
    /// The one supported way to recover a `RegisteredSource` from a compiled program — for callers
    /// that hold a `SourceFile` and need the identity this compilation gave it.
    pub fn source_named(&self, name: &str) -> Option<crate::source::RegisteredSource> {
        self.sources
            .id_for_name(name)
            .and_then(|id| self.sources.get(id))
            .cloned()
    }

    /// The source an item was parsed from.
    ///
    /// AS1b-ii-b: the one way from an item to its file. Callers used to hold an `Arc<SourceFile>`
    /// taken straight out of `item_files` and index spans against it, which made that map a rival
    /// source *authority*. Now the map names an id and the registry answers — one authority, and
    /// the id cannot disagree with it.
    pub fn item_file(&self, item: ItemId) -> Option<&std::sync::Arc<crate::source::SourceFile>> {
        self.item_sources
            .get(&item)
            .and_then(|id| self.sources.get(*id))
            .map(|source| source.file())
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
