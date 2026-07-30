//! STARK MIR v0.1 — data model and deterministic textual dump.
//!
//! Implements the APPROVED contract `STARKLANG/docs/compiler/mir.md` (CE3, CD-028). This module
//! is the *shape* of MIR; `lower` produces it from typed HIR (WP-C4.2, scalar core first). The
//! contract's load-bearing invariants, restated where the code enforces or relies on them:
//!
//! - Statements and rvalues are TOTAL: they never trap, never call user code, never diverge.
//!   Everything that can trap, diverge, or run user code — including `Drop` — is a terminator.
//! - There is NO unwinding anywhere: no cleanup edges, no landing pads. `Checked` has exactly
//!   one normal successor and an implicit aborting failure described by its `TrapInfo`.
//! - Verified MIR is monomorphised-only: no `Param`/`Infer` types survive lowering.
//! - `Option`/`Result` are logical enums (`EnumRef::CoreOption`/`CoreResult`) sharing the
//!   user-enum aggregate/discriminant/match machinery (CD-028 required change #2).
//! - Every statement and terminator carries `SourceInfo` with an explicit interned `FileId`
//!   (the DEV-006 lesson) and either a user span or a labeled synthetic origin.
//! - The textual dump is deterministic and versioned (`MIR_VERSION`).

pub mod drop_plan;
pub mod interp;
pub mod lower;
pub mod opt;
pub mod provider_lower;
pub mod provider_sig;
pub mod verify;

use crate::source::{SourceFile, Span};
use std::fmt::Write as _;
use std::sync::Arc;

/// Bumped whenever the MIR shape changes (contract §11). Consumers state the version they
/// accept; mismatch is a hard error.
///
/// `0.2` (A11, CD-224 — `mir-amendment-A11-host-resources.md`): adds [`MirTy::HostResource`]. A
/// `MirTy` variant, unlike A10's `Callee` variant, flows through every part of the compiler that
/// reasons about types, so this is a shape change rather than a surface revision. The increment
/// invalidates every build key, which is the point: a key that ignored a representation change
/// would serve a cached artifact produced under different type rules.
pub const MIR_VERSION: &str = "0.2";

/// Runtime-surface revision (Amendment A1, CD-031). Additive `RuntimeFn`/String/Vec growth
/// bumps this, not `MIR_VERSION`. Stamped onto every `MirProgram`; a consumer rejects a
/// program whose `runtime_surface` it does not support before consuming any body.
///
/// `0.1-A2` (C4.5f-2, per CD-032's activation rule — dated enumeration in the amendment doc
/// rev. 5): adds by-reference Vec iteration, `VecIterNew`/`VecIterNext` yielding
/// `Option<&T>` for `T: Copy`.
///
/// `0.1-A3` (C4.5f-3a/b, amendment rev. 6): the HashMap group (insertion-order per CD-009;
/// user-`Drop` key/value types excluded so no runtime op ever runs a user destructor), plus
/// the A1-approved-but-deferred Char ops (`StringPushChar`/`StringPopChar`,
/// `PrintlnChar`/`PrintChar`).
/// `0.1-A10` (WP-C7.8.2, CD-200, CE3 — `mir-amendment-A10-provider-invocation.md`): adds
/// [`Callee::Provider`] and [`ValidatedProviderCall`]. **Adds no `RuntimeFn` member.** A10 is a
/// new *invocation category*, not more runtime surface in the A1 sense, which is why it has its
/// own amendment document rather than a rev. 14 of the A1 string/collection surface. The constant
/// still advances because a consumer that cannot represent provider calls must reject an A10
/// program before consuming any body (V-SURFACE-1).
pub const MIR_RUNTIME_SURFACE: &str = "0.1-A10";

// ------------------------------------------------------------------ identity --

/// Interned source-file identity. MIR must never carry a file-less span (V-SRC-1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FileId(pub u32);

/// Nominal identity for enum types: user enums carry their HIR item; `Option`/`Result`/
/// `Ordering` are logical core enums with no user item (contract §3, CD-028 required change #2;
/// `CoreOrdering` added by MIR Amendment A2, CE3-approved 2026-07-19).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum EnumRef {
    User(crate::hir::ItemId),
    CoreOption,
    CoreResult,
    /// The prelude `Ordering` enum as a logical MIR enum (Amendment A2). Fixed logical
    /// discriminants: `Less = 0`, `Equal = 1`, `Greater = 2` — logical MIR only, not a physical
    /// ABI (C5.1 chooses the physical layout). Three fieldless variants; `Copy`, no drop glue.
    CoreOrdering,
}

/// A monomorphised function instance. Scalar core (WP-C4.2) only produces empty `type_args`;
/// the field exists so C4.5's monomorphisation extends, rather than reshapes, the model.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Instance {
    pub item: crate::hir::ItemId,
    pub type_args: Vec<MirTy>,
    /// Canonical symbol: deterministic and injective for identical inputs; NOT a stable
    /// external ABI (contract §2 qualification).
    pub symbol: String,
}

// --------------------------------------------------------------------- types --

// `Ord` (WP-C4.5c): `MirTy` vectors key the `TypeContext` maps per monomorphised nominal
// instantiation; the ordering is structural and carries no semantic meaning.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum MirTy {
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
    Unit,
    Never,
    /// Unsized; appears only behind `Ref` (V-TY-3).
    Str,
    String,
    Struct(crate::hir::ItemId, Vec<MirTy>),
    Enum(EnumRef, Vec<MirTy>),
    Tuple(Vec<MirTy>),
    Array(Box<MirTy>, u64),
    /// Unsized; appears only behind `Ref` (V-TY-3).
    Slice(Box<MirTy>),
    Ref {
        mutable: bool,
        inner: Box<MirTy>,
    },
    FnPtr {
        params: Vec<MirTy>,
        ret: Box<MirTy>,
    },
    /// Semantically opaque runtime types (Vec, Box, HashMap, …) — NOT Option/Result.
    Core(crate::hir::CoreType, Vec<MirTy>),
    /// **A11 (CD-224, CE3): a host resource.** Established by the COMPILER for a Core resource
    /// (`File`) or by a PACKAGE declaration for a package resource (`TcpStream`) — one
    /// representation, two authorities.
    ///
    /// **This type has no values the compiler can make.** There is no default rvalue, no
    /// `Aggregate`, no constant and no constructor that produces one (CD-234). A local of this type
    /// starts DEAD, with its drop flag clear, and becomes live only through a successful
    /// `HandleOut`, a move from an already-live resource, or an argument/return carrying one.
    ///
    /// The source-level nominal is a synthesized **zero-variant enum** (CD-234): opaque
    /// structurally rather than by a checker rule, because a zero-variant enum has no variant to
    /// name and no struct-literal form, so no expression and no pattern can manufacture a value.
    /// Its ordinary zero-variant-enum backend representation — including the default-init
    /// placeholder — **must not** apply once the nominal is classified as a host resource.
    ///
    /// **Boxed.** A11 §4 writes the form with inline fields; the three identities are logically
    /// inline and the box is representation only. `MirTy` is cloned constantly throughout the
    /// compiler, and 52 bytes of resource identity on every `MirTy` — hence on every `Rvalue` and
    /// every `Statement` — is a cost paid by all the code that never touches a resource. Boxing a
    /// rarely-instantiated variant's payload is the standard remedy, and `clippy::large_enum_variant`
    /// flagged `Statement::Assign` crossing its threshold as a direct result of not doing it.
    HostResource(Box<HostResourceTy>),
}

/// **A11 §4's nominal identity, widened (CD-235).** A host resource's STARK-side identity is either a
/// compiler-owned Core type or a package-owned item — A11's "one representation, two authorities".
///
/// A11 §4 wrote `nominal: ItemId`, which cannot name a Core resource: `File` resolves to
/// `CoreType::File` (`resolve.rs`), a different enum from `ItemId`, so there is no Core *item* to
/// point at. The widening makes §4's model expressible on both sides.
///
/// **Sequencing exception, not a second representation (CD-235).** Package resources use
/// `MirTy::HostResource` immediately. Core `File` temporarily stays on its pre-A11
/// `MirTy::Core(CoreType::File, [])` representation, which is the implemented and qualified path
/// behind C7.8.4's evidence, pending a separately requalified migration. `Core` exists here so that
/// migration is a registry change rather than a type change — and `V-HOSTRES-1` refuses a program
/// that half-performs it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum HostResourceNominal {
    /// A compiler-owned Core resource. **Not yet produced** — see CD-235's sequencing exception.
    Core(crate::hir::CoreType),
    /// A package-owned resource nominal: the synthesized zero-variant enum's item (CD-234).
    Item(crate::hir::ItemId),
}

/// A11 §4's three identities for a host resource. See [`MirTy::HostResource`] for why they are boxed.
///
/// Both identities are retained, as Packet 6 requires: `nominal` is what diagnostics and the source
/// language talk about, `provider`/`resource` is what the ABI talks about. Neither can be derived
/// from the other, and dropping either loses something a later stage needs.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct HostResourceTy {
    /// The STARK nominal this resource is, e.g. the item for `TcpListener`.
    pub nominal: HostResourceNominal,
    /// §2 identity of the provider that owns the resource type.
    pub provider: String,
    /// §13 resource-type name as that provider declares it, e.g. `"tcp_stream"`.
    pub resource: String,
}

impl MirTy {
    /// Constructs a host-resource type from A11 §4's three identities.
    pub fn host_resource(
        nominal: HostResourceNominal,
        provider: impl Into<String>,
        resource: impl Into<String>,
    ) -> Self {
        MirTy::HostResource(Box::new(HostResourceTy {
            nominal,
            provider: provider.into(),
            resource: resource.into(),
        }))
    }
}

// -------------------------------------------------------------------- bodies --

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LocalId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockId(pub u32);

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LocalKind {
    /// `Local(0)`, written before `Return`.
    Return,
    Param(u32),
    User(String),
    Temp,
    /// Drop-elaboration flag (C4.5); always `Bool` (V-DROP-2).
    DropFlag,
    /// Opaque index-proof token (contract §6, CD-028 required change #3): defined only by
    /// `Checked { op: CheckIndex }`, consumed only by `Projection::Index` on the same base.
    IndexProof,
}

#[derive(Clone, Debug)]
pub struct LocalDecl {
    pub ty: MirTy,
    pub kind: LocalKind,
}

#[derive(Clone, Debug)]
pub struct MirBody {
    pub instance: Instance,
    pub params: Vec<MirTy>,
    pub ret: MirTy,
    pub locals: Vec<LocalDecl>,
    pub blocks: Vec<BasicBlock>,
    pub entry: BlockId,
}

#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub statements: Vec<(Statement, SourceInfo)>,
    pub terminator: (Terminator, SourceInfo),
}

// ---------------------------------------------------------------- provenance --

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SyntheticKind {
    DropElaboration,
    ForLoopDesugar,
    DropFlagInit,
    ReturnSlot,
    ShortCircuit,
    MatchDesugar,
}

#[derive(Clone, Copy, Debug)]
pub enum Origin {
    UserCode,
    Synthetic(SyntheticKind),
}

#[derive(Clone, Copy, Debug)]
pub struct SourceInfo {
    pub file: FileId,
    pub span: Span,
    pub origin: Origin,
}

// ------------------------------------------------------- places and operands --

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Projection {
    /// Struct/tuple field by declaration-order index.
    Field(u32),
    /// Enum payload field; legal only after a discriminant test (V-DISC-1).
    VariantField(u32, u32),
    Deref,
    /// Element access consuming an index-proof token (never an ordinary integer local).
    Index(LocalId),
    /// MIR amendment A5 (CD-038): a STATICALLY KNOWN array element. Valid only on
    /// `Array<T, N>` with `index < N`, which the verifier checks directly — it needs no
    /// `CheckIndex` and no `IndexProof`, and is invalid on `Vec` and slice types, whose lengths
    /// are not statically known.
    ///
    /// It exists because a proof-backed `Index` cannot name a statically-known sub-place: a
    /// dynamic proof forces move analysis to treat the whole array as one unit, so moving one
    /// element out poisoned the rest. `ConstIndex` participates precisely in move analysis, which
    /// is what makes consuming array patterns and by-value array iteration expressible.
    /// Dynamic source indexing continues to use `CheckIndex` + `Index`.
    ConstIndex(u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place {
    pub local: LocalId,
    pub projection: Vec<Projection>,
}

impl Place {
    pub fn local(local: LocalId) -> Self {
        Place {
            local,
            projection: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Constant {
    Int(i128, MirTy),
    Float(f64, MirTy),
    Bool(bool),
    Unit,
    /// A function value: a bare instance reference (TYPE-FN-001 — Copy, no comparisons).
    FnPtr(Instance),
    /// A decoded immutable UTF-8 string literal (A1/CD-031). Denotes a `&str`
    /// (`Ref { mutable: false, inner: Str }`); identity is unobservable. Content is the
    /// resolved literal, never the source spelling; MUST be valid UTF-8 (V-STR-1).
    Str(String),
}

#[derive(Clone, Debug)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Const(Constant),
}

// ------------------------------------------------------------------- rvalues --

/// NON-TRAPPING unary operators only (integer negation traps on MIN → `Checked`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MirUnOp {
    Not,
    FloatNeg,
}

/// NON-TRAPPING binary operators only: comparisons, and float add/sub/mul (IEEE, CD-006).
/// Everything integer-arithmetic and float div/rem is a `Checked` terminator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MirBinOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    FloatAdd,
    FloatSub,
    FloatMul,
    /// A6 (CD-139): TOTAL IEEE division and remainder. NUM-FLOAT-OP-001 gives floating division by
    /// zero the IEEE infinity or NaN result rather than a trap, and gives `%` a NaN for a zero
    /// divisor, so neither owes a check — they belong with the other pure float operators, not
    /// among the checked ones. See `CheckedOp::FloatDiv`/`FloatRem`, now deprecated.
    FloatDiv,
    FloatRem,
    // A5: bitwise operators are pure (non-trapping) — for same-width two's-complement operands
    // the result is always representable in the operand width, so no range check is owed.
    BitAnd,
    BitOr,
    BitXor,
}

#[derive(Clone, Debug)]
pub enum AggKind {
    Struct(crate::hir::ItemId),
    Tuple,
    Array(MirTy),
    EnumVariant(EnumRef, u32),
}

/// A4 (CD-036): which target-layout property `LayoutQuery` asks for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayoutKind {
    SizeOf,
    AlignOf,
}

#[derive(Clone, Debug)]
pub enum Rvalue {
    Use(Operand),
    UnOp(MirUnOp, Operand),
    BinOp(MirBinOp, Operand, Operand),
    Aggregate(AggKind, Vec<Operand>),
    Discriminant(Place),
    RefOf {
        mutable: bool,
        place: Place,
    },
    /// A4 (amendment `mir-amendment-A4-layout.md`, CD-036): a target-layout query
    /// (`size_of::<T>()` / `align_of::<T>()`), typed `UInt64`. The queried type is PRESERVED —
    /// MIR is monomorphised, so `ty` is always concrete, and a backend answers from its own
    /// target layout. Pure: cannot trap, call user code, or diverge (§5 totality holds).
    LayoutQuery {
        kind: LayoutKind,
        ty: MirTy,
    },
}

/// The TOTAL statement set (contract §5/§6): assignments and nops only. `Drop` is a terminator.
#[derive(Clone, Debug)]
pub enum Statement {
    Assign(Place, Rvalue),
    Nop,
}

// --------------------------------------------------------------- terminators --

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrapCategory {
    IntegerOverflow,
    DivideByZero,
    IndexOutOfBounds,
    CastFailure,
    Panic,
    UnwrapNone,
    UnwrapErr,
    AssertFailure,
    /// A5 / NUM-SHIFT-001: a shift count that is negative or ≥ the operand width. Distinct from
    /// `IntegerOverflow` (which a left shift still raises when its *result* is not representable).
    InvalidShift,
    /// A7 / PROC-EXIT-001 (CD-150 CE3): `main` returned an `Int32`/`Ok(Int32)` outside `0..=255`.
    ///
    /// Provenance for this category is the ENTRY FILE at 1:1. The entry contract is violated by the
    /// signature's RESULT, not by an expression, so there is no sub-expression the three engines
    /// could agree to blame; one defined location beats three plausible ones.
    InvalidExitStatus,
}

#[derive(Clone, Copy, Debug)]
pub struct TrapInfo {
    pub category: TrapCategory,
    pub source: SourceInfo,
}

/// Trapping primitives (contract §6): one normal successor, implicit abort on failure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CheckedOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    Shl,
    Shr,
    /// A5: integer exponentiation (NUM-INT-ARITH-001) — nonnegative exponent required,
    /// each intermediate multiply checked; traps on overflow or negative exponent.
    Pow,
    /// **DEPRECATED (CD-139), unreachable.** Lowering no longer emits these: NUM-FLOAT-OP-001 makes
    /// float division and remainder TOTAL, so a checked form is a contradiction — a primitive
    /// declared trapping that is guaranteed never to trap. Superseded by `MirBinOp::FloatDiv`/
    /// `FloatRem`. Retained only so this amendment stays additive; removal is a separately versioned
    /// cleanup.
    FloatDiv,
    FloatRem,
    Cast,
    /// Defines an `IndexProof` local for `Projection::Index` (arrays/slices; Vec is runtime).
    CheckIndex,
}

/// The closed, versioned runtime surface (contract §7). Scalar core enumerates only what the
/// WP-C4.2 lowering emits; every extension of this enum is an extension of the MIR version's
/// runtime contract, and an unknown variant must fail loudly at any backend (V-RT-1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RuntimeFn {
    PrintlnInt64,
    PrintlnUInt64,
    PrintlnBool,
    PrintlnFloat64,
    /// 0.1-A9 (WP-C6.3e, DEV-105): WIDTH-PRESERVING `Float32` output. PRINT-DISPLAY-001 renders a
    /// float with "the fewest significant decimal digits that parse back to the same DECLARED IEEE
    /// value" — for a `Float32` that is the f32, so `0.1f32` must print `0.1`. Routing it through
    /// `PrintFloat64` widened first and printed `0.10000000149011612`, the shortest round-trip of the
    /// WIDENED value: conformant for the f64 it had become, wrong for the `Float32` that was
    /// declared. The operation identity carries the declared width, so every engine formats at f32.
    PrintFloat32,
    PrintlnFloat32,
    PrintInt64,
    PrintUInt64,
    PrintBool,
    PrintFloat64,
    // --- A1 (CD-031), C4.5e-1: String/str surface. Char-dependent ops (PrintlnChar/PrintChar,
    // StringPushChar/StringPopChar) are added with Char lowering in a later C4.5e sub-slice. ---
    PrintlnStr,
    PrintStr,
    StringNew,
    StringFromStr,
    StringLen,
    StringIsEmpty,
    StringPushStr,
    StringClear,
    StringAsStr,
    StringClone,
    StringContains,
    StrLen,
    StrIsEmpty,
    StrToString,
    StrBytes,
    StrEq,
    StrCmp,
    // --- A1 (CD-031), C4.5e-2: Vec data surface. Iteration (VecIterNew/VecIterNext) is NOT
    // here: STARK's `.iter()` is by-reference (`&T`), which A1 reserved to an interior-
    // reference sub-slice; activating it needs an owner-reviewed surface bump. ---
    VecNew,
    VecWithCapacity,
    VecPush,
    VecPop,
    VecLen,
    VecIsEmpty,
    VecIndexGet,
    VecReplace,
    VecRemove,
    VecClear,
    // --- 0.1-A7 (WP-C4.7-6.1): `Box<T>` construction and extraction. `Box<T>` stays
    // `MirTy::Core(Box, [T])` — an OPAQUE OWNING runtime type, deliberately not lowered
    // transparently as `T` (that would make recursive types through `Box` infinitely sized).
    // There is no public box-drop operation: ordinary destruction goes through the `Drop`
    // terminator's structural glue, which drops the contained `T` exactly once and then releases
    // the allocation. Core v1 has NO `Deref` trait, so `*box` is not a construct — extraction is
    // `into_inner` only. ---
    BoxNew,
    BoxIntoInner,
    // --- 0.1-A4 (C4.6 A4-2b): checked interior access — `get`/`get_mut` return `Option<&T>`/
    // `Option<&mut T>` and DO NOT trap on out-of-bounds (they return `None`), distinct from the
    // trapping `VecIndexGet`/`v[i]`. The reference is an interior borrow into the live Vec. ---
    VecGetRef,
    VecGetMutRef,
    // --- 0.1-A5 (C4.6 A4-2d): `str::chars`/`String::chars` iteration. The iterator is a
    // borrowed snapshot over the string's chars (Char is Copy, so a snapshot is sound);
    // `CharsIterNext` yields `Option<Char>` by value. ---
    CharsIterNew,
    CharsIterNext,
    // --- 0.1-A6 (C4.6 A4 slicing): slice views. `SliceNew(&base, lo, hi, inclusive) -> &[T]`
    // creates a view over an Array/Vec/slice referent and TRAPS IndexOutOfBounds on a
    // negative, inverted, or out-of-range bound (06-Standard-Library behavioral requirement);
    // re-slicing composes. `SliceLen`/`SliceIsEmpty` read the view length. ---
    SliceNew,
    /// 0.1-A8 (WP-C4.7-8.6): the EXCLUSIVE slice view. REF-SLICE-001: "writes through an
    /// exclusive slice reference update the original object", so unlike `SliceNew` this yields
    /// `&mut [T]` and writes through the window reach the base object.
    SliceNewMut,
    SliceLen,
    SliceIsEmpty,
    // --- 0.1-A2 (C4.5f-2, CD-032): by-reference Vec iteration. The iterator borrows the
    // source Vec (borrowck forbids mutation while live); `VecIterNext` yields `Option<&T>`
    // and requires `T: Copy` (V-COPY-1). ---
    VecIterNew,
    VecIterNext,
    // --- 0.1-A3 (C4.5f-3a): HashMap group. Insertion-order storage (CD-009); `Get` yields
    // an interior `Option<&V>`; user-`Drop` key/value types are excluded at lowering so no
    // runtime op ever runs a user destructor (`Insert` RETURNS the replaced `Option<V>` —
    // the caller drops it at a visible Drop, the VecReplace pattern). ---
    HashMapNew,
    HashMapInsert,
    HashMapGet,
    HashMapLen,
    HashMapIsEmpty,
    HashMapContainsKey,
    /// CD-180. Parity with `HashSet`, which reached these first: `map::remove`/`map::clear` already
    /// existed in the runtime because the set needed them, leaving the MAP less capable than the
    /// type built from it.
    HashMapRemove,
    HashMapClear,
    HashMapKeysIterNew,
    HashMapKeysIterNext,
    // --- DEV-116: HashSet. `HashSet<T>` is the map at `V = Unit`, so uniqueness is decided by the
    // SAME `Eq` comparator dispatch and first-insertion iteration order is inherited rather than
    // reimplemented. `iter` is deliberately absent — the admitted API includes it, but iteration is
    // out of this change's scope and `for` over a set stays refused at lowering with its own reason.
    HashSetNew,
    HashSetInsert,
    HashSetRemove,
    HashSetContains,
    HashSetLen,
    HashSetIsEmpty,
    HashSetClear,
    /// `HashSet::iter` (DEV-116-B). Deliberately SHARES the map's cursor implementation in every
    /// engine: a set is `StarkMap<T, ()>`, so its keys ARE its elements and `KeysIter` already
    /// traverses exactly the right sequence in exactly the right order.
    HashSetIterNew,
    HashSetIterNext,
    // --- 0.1-A3 (C4.5f-3b): the A1-approved Char ops, deferred from e-1 until Char lowered. ---
    PrintlnChar,
    PrintChar,
    StringPushChar,
    StringPopChar,
}

#[derive(Clone, Debug)]
pub enum Callee {
    Instance(Instance),
    /// Indirect call through a `FnPtr`-typed operand (CD-021/CD-027).
    FnValue(Operand),
    Runtime(RuntimeFn),
    /// **A10 (CD-200, CE3): a Native Provider ABI v0.1 call.** Indexes
    /// [`MirProgram::provider_calls`], resolving to a [`ValidatedProviderCall`] that carries the
    /// full declared contract.
    ///
    /// This is a distinct call form rather than more [`RuntimeFn`] members because the two have
    /// different trust models, not merely different callees. A `RuntimeFn` is compiler-owned: its
    /// identity is closed, its semantics are the compiler/runtime contract, it needs no provider
    /// selection, no external metadata, and its target compatibility is implicit. A provider call
    /// is externally *declared*, and carries provider identity, capability identity, a validated
    /// function declaration, target applicability, ABI parameter and output-slot shape, ownership
    /// transfer mode, borrowed-buffer constraints, resource-type identity, a declared recoverable
    /// status vocabulary, and failure-channel rules. Encoding that as `RuntimeFn` would either
    /// erase those distinctions or push provider metadata into `RuntimeFn` indirectly, leaving the
    /// distinction implicit and unverifiable.
    ///
    /// `RuntimeFn` therefore stays reserved for compiler-owned runtime operations, and a provider
    /// call encoded as one is a verification failure (V-PROV-10), not a style preference.
    Provider(ProviderCallId),
}

/// A10: index into [`MirProgram::provider_calls`].
///
/// Deliberately **not** a symbol string. Every A10 verifier invariant is checked *against the
/// declaration*; a call site carrying only a name would force the verifier to reconstruct the
/// contract it exists to check, and would let an unvalidated symbol reach the backend.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ProviderCallId(pub u32);

/// **A11 §5: the validated close for one host-resource type.**
///
/// A `HostResource` local's `Drop` terminator must call this, exactly once. §5's rule 4 is what makes
/// "exactly once" true: no other call site may invoke a close — a package cannot bind one, and a
/// `Callee::Provider` whose declaration is `is_close_for` is rejected outside a `Drop`. **MIR owns the
/// only path.**
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedProviderClose {
    /// The `HostResource` form this closes.
    pub resource: MirTy,
    /// The `is_close_for` function, already validated, in [`MirProgram::provider_calls`].
    pub close: ProviderCallId,
}

/// A10: one provider call site's fully resolved, already-validated contract.
///
/// **Resolution happens before MIR verification** (A10 §3): capability requirement → provider
/// selection for the target → metadata validation → `FunctionDecl` resolution → this record →
/// `Callee::Provider`. By the time verification runs, selection has happened and the metadata has
/// passed [`crate::provider_abi::validate`]. The backend never performs first-time provider
/// selection and never interprets unvalidated metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedProviderCall {
    /// §2 identity of the selected provider.
    pub provider: crate::provider_abi::ProviderIdentity,
    /// §5 capability this call belongs to. Must be one the provider declares, and must be the
    /// declaring function's own capability (V-PROV-3).
    pub capability: String,
    /// §6 declaration, copied from validated metadata rather than referenced, so a MIR program is
    /// self-contained: dumping, re-verifying, or replaying it needs no provider lookup.
    pub function: crate::provider_abi::FunctionDecl,
    /// §4 target this call was resolved for. Verification re-checks it rather than trusting that
    /// resolution ran (V-PROV-2).
    pub target_triple: String,
    /// The package's declared recoverable status vocabulary (Packet 1 §1.2, A10 §2's
    /// `status_binding`).
    ///
    /// Carried on the record because emission needs it where the call is emitted, and because an
    /// empty vocabulary is a *meaningful* value rather than missing data: it says every nonzero
    /// status from this provider is a contract violation.
    pub status_binding: crate::provider_bind::StatusBinding,
    /// The Cargo package name of the crate implementing this provider, e.g.
    /// `"stark-time-native"`.
    ///
    /// A **name**, never a path. A path here would be an absolute filesystem location baked into
    /// MIR and from there into build artefacts, which is exactly what §15.2's "no absolute path in
    /// semantic identity" rule and Gate C7.2's path remapping exist to prevent. The name is
    /// resolved to a location at build time, from `NativeToolchainOptions::provider_crates`.
    pub provider_crate: String,
    /// §13 resource-type list the provider declares.
    ///
    /// Carried because `RawResourceHandle::resource_type` is "a compiler-assigned index into the
    /// provider's declared resource-type list" (§7) — so validating a returned handle requires
    /// knowing that list, and a record that could not validate its own handles would not be
    /// self-verifying in the sense §4 invariant 2 already established for targets.
    pub provider_resource_types: Vec<String>,
    /// §4 target list the provider declares, copied so the record is **self-verifying**.
    ///
    /// Without it, verification could only take `target_triple` on faith: there would be nothing
    /// to check it against, and "re-checks rather than trusts" would be a comment rather than a
    /// rule. With it, a record whose selected target is not one the provider admits is rejected in
    /// MIR, even if resolution produced it.
    pub provider_target_triples: Vec<String>,
}

impl ValidatedProviderCall {
    /// The C symbol this call links against — [`crate::provider_abi::FunctionDecl::name`]
    /// **verbatim**.
    ///
    /// Never routed through `mangle::sanitize_symbol`: that encodes MIR canonical symbols into
    /// legal Rust identifiers, and a provider symbol is not a MIR instance. The same name has to
    /// resolve under a future `dlsym`, so a repaired name would make the metadata name differ from
    /// the linkage name — the one thing that must never be true (Packet 1 §1.3).
    pub fn symbol(&self) -> &str {
        &self.function.name
    }
}

#[derive(Clone, Debug)]
pub enum Terminator {
    Goto {
        target: BlockId,
    },
    SwitchInt {
        scrut: Operand,
        arms: Vec<(u128, BlockId)>,
        otherwise: BlockId,
    },
    Call {
        callee: Callee,
        args: Vec<Operand>,
        dest: Place,
        target: BlockId,
    },
    /// Run `place`'s destructor, continue at `target`. No unwind edge (CD-028 change #1).
    Drop {
        place: Place,
        target: BlockId,
    },
    Checked {
        op: CheckedOp,
        args: Vec<Operand>,
        dest: LocalId,
        target: BlockId,
        trap: TrapInfo,
    },
    Trap {
        info: TrapInfo,
        /// Optional user message (A1/CD-031): `panic(msg)` / failed `assert*` carry a `&str`
        /// operand; compiler-generated traps carry `None`. Participates in every operand
        /// analysis, not only typing.
        message: Option<Operand>,
    },
    Return,
    Unreachable,
}

// ------------------------------------------------------------------- program --

/// Nominal-type layout information the verifier and backends need to resolve projections:
/// struct field types and user-enum variant payload types, keyed per **monomorphised nominal
/// instance** `(ItemId, type arguments)` — contract §2 defines the type context over nominal
/// types reachable from the bodies, and a nominal type in monomorphised-only MIR *is* an
/// instance (WP-C4.5c; non-generic nominals key with an empty argument vector). `Option`/
/// `Result` payloads are derived from their type arguments directly (no table entry needed).
/// An implementation companion to the contract's nominal types — not part of the dump, not a
/// shape/version change.
#[derive(Clone, Debug, Default)]
pub struct TypeContext {
    pub struct_fields: std::collections::BTreeMap<(u32, Vec<MirTy>), Vec<MirTy>>,
    pub enum_variants: std::collections::BTreeMap<(u32, Vec<MirTy>), Vec<Vec<MirTy>>>,
    /// C4.5d: destructor instance symbol per nominal instance with an own `Drop` impl —
    /// how `Drop`-terminator glue (interpreter now, backends later) dispatches destructors.
    /// Populated for every nominal reachable through a lowered `Drop`'s glue.
    pub drop_impls: std::collections::BTreeMap<(u32, Vec<MirTy>), String>,
    /// A1 (CD-031): nominal instances carrying an `impl Copy`. `is_copy`/V-COPY-1 read it;
    /// populated during lowering like `drop_impls`. (The runtime-glue drop of String/Vec is
    /// recognized structurally, not via a table entry.)
    pub copy_types: std::collections::BTreeSet<(u32, Vec<MirTy>)>,
    /// WP-C6.3d (CD-133): the selected `Eq::eq` instance symbol per nominal instance used as a
    /// `HashMap`/`HashSet` KEY. Populated during lowering exactly as [`Self::drop_impls`] is
    /// (C4.5d), and read by both the MIR interpreter and the backend so key identity is decided by
    /// the user's lawful `Eq` (STD-HASH-001) rather than by structural comparison.
    ///
    /// This is metadata, not a runtime-surface change: no `RuntimeFn` gains or changes an argument,
    /// so the runtime-surface revision does not move. Only nominal keys need an entry — a primitive
    /// or `String` key has no user impl and compares structurally, which for those types IS its
    /// lawful `Eq`.
    pub eq_impls: std::collections::BTreeMap<(u32, Vec<MirTy>), String>,
    /// **A11 §5: host-resource type → its validated close** (`MirProgram::provider_closes`, keyed
    /// for lookup).
    ///
    /// Lives on the type context for the same reason `drop_impls` does: `drop_plan::plan_for`
    /// resolves destruction from the type alone, and a resource's destruction *is* its close. Keyed
    /// by the full `HostResource` form, so a listener's close cannot be selected for a stream — §5
    /// obligation 4, the case a structural check misses because both closes are `HandleConsumed` of
    /// *a* resource and differ only in which one they name.
    pub host_resource_closes: std::collections::BTreeMap<MirTy, ProviderCallId>,
    /// WP-C6.1g-a (OWN-COPY-001, amended): nominal items that are `Copy` when their type
    /// arguments are — impl-`Copy` plus structurally eligible. `is_copy` consults this and
    /// recurses on the arguments, mirroring the front end's `copy_eligible_types` so the verifier
    /// and backend agree with the checker. Supersedes the per-instance `copy_types` set for the
    /// copy decision; `copy_types` is retained for the build manifest.
    pub copy_eligible_items: std::collections::BTreeSet<u32>,
}

impl TypeContext {
    /// Whether MIR classifies `ty` as `Copy` — **the one rule every CONSUMER reads** (CD-065,
    /// folded into WP-C5.3d-1c). `mir::verify`'s V-COPY-1 and the backend's storage and drop
    /// decisions all route here; it had been written out identically in both, which is the same
    /// defect shape CD-064 closed for destruction order.
    ///
    /// `mir::lower::is_copy` deliberately does NOT delegate: it is the PRODUCER. It answers the
    /// nominal case from the HIR (`type_has_copy_impl`) precisely because it is what fills
    /// `copy_types`, and cannot read a table it has not written yet. Its structural arms must stay
    /// in step with these, which is what `lowered_copy_classification_matches_the_type_context`
    /// checks.
    ///
    /// An unmarked all-`Copy`-fields STARK struct is still Move; only an `impl Copy` makes it
    /// Copy, and the front end has already validated the all-Copy-fields / no-`Drop` rules for
    /// that impl to exist. §7.4 forbids a backend broadening the set from Rust traits, which is
    /// why the backend asks this rather than asking Rust.
    pub fn is_copy(&self, ty: &MirTy) -> bool {
        match ty {
            MirTy::Struct(item, args) | MirTy::Enum(EnumRef::User(item), args) => {
                self.copy_eligible_items.contains(&item.0) && args.iter().all(|a| self.is_copy(a))
            }
            MirTy::Enum(_, args) => args.iter().all(|a| self.is_copy(a)),
            MirTy::Tuple(elems) => elems.iter().all(|e| self.is_copy(e)),
            MirTy::Array(elem, _) => self.is_copy(elem),
            MirTy::Ref { mutable, .. } => !*mutable,
            MirTy::Slice(_) | MirTy::Core(..) | MirTy::String => false,
            // **A11/CD-234: a host resource is NEVER `Copy`, and this arm must stay explicit.**
            //
            // The wildcard below classified it `Copy`, with three consequences, none of which
            // announced themselves: `is_slot_backed` became false, so the local was declared through
            // `default_value_expr` -- which refuses a resource -- and emission failed before `Drop`
            // was ever reached; `emit_drop` refuses a `Copy` type outright, so the close could not
            // have run either; and "Copy" is the licence to DUPLICATE a handle, which would give two
            // owners of one resource and close it twice.
            //
            // Being non-`Copy` is what makes a resource slot-backed, and a slot is what gives it
            // `ValueSlot::dead()` -- CD-234's "the slot begins dead" is then the representation
            // itself rather than a rule anything has to enforce.
            MirTy::HostResource(_) => false,
            _ => true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MirProgram {
    /// Interned source files; `FileId` indexes here.
    pub files: Vec<Arc<SourceFile>>,
    /// Bodies sorted by canonical symbol (dump determinism).
    pub bodies: Vec<MirBody>,
    /// Nominal layout info for projection typing (verifier/backends).
    pub types: TypeContext,
    /// A1 (CD-031): the MIR shape version this program was produced against (`MIR_VERSION`).
    pub mir_version: String,
    /// A1 (CD-031): the runtime-surface revision (`MIR_RUNTIME_SURFACE`). A consumer rejects a
    /// program whose surface it does not support before consuming any body (V-SURFACE-1).
    pub runtime_surface: String,
    /// A10 (CD-200): validated provider-call records, indexed by [`ProviderCallId`].
    ///
    /// Program-level rather than per-body because one provider function is typically called from
    /// several bodies, and the *contract* is a property of the program's resolved provider set,
    /// not of any one call site. Empty for every program that makes no provider call, which is
    /// every program produced before A10.
    pub provider_calls: Vec<ValidatedProviderCall>,
    /// **A11/CD-234: the resource-name → nominal bindings this program uses.**
    ///
    /// Carried on the program for the same reason `provider_calls` copies its `FunctionDecl` rather
    /// than referencing it: a MIR program must be **self-contained**, so dumping, re-verifying or
    /// replaying it needs no registry lookup. Without this the verifier could only see
    /// `ResourceRegistry::builtin()` and would reject every package-declared resource as unbound.
    ///
    /// A `Vec` of pairs rather than a map so the serialised order is the program's own, and sorted by
    /// name at construction so it is a function of the manifest rather than of iteration.
    pub resource_bindings: Vec<(String, HostResourceNominal)>,
    /// **A11 §5: the close selected for each host resource, at RESOLUTION time.**
    ///
    /// `drop_plan` looks a resource up here and emits a provider call to the recorded id, rather
    /// than searching provider metadata during lowering. Selecting at resolution is what lets the
    /// verifier discharge its five obligations before emission — a close chosen at drop time could
    /// only be checked after the program was already being built.
    pub provider_closes: Vec<ValidatedProviderClose>,
}

impl MirProgram {
    /// A10: the validated contract behind a [`ProviderCallId`], or `None` if the id is out of
    /// range. Verification rejects a dangling id (V-PROV-1) rather than panicking, so this
    /// returns an `Option` instead of indexing.
    /// **The resource registry this program's own bindings describe**, plus the compiler built-ins.
    ///
    /// One derivation shared by the verifier and the backend. They previously each built their own —
    /// the verifier from `resource_bindings`, the backend from `ResourceRegistry::builtin()` — so a
    /// package resource verified and then failed to plan at emission with `UnboundResourceType`.
    /// Two consumers deriving the same thing separately is the defect class this method exists to
    /// remove, and it is the same fix `ResourceRegistry::resolve_ty` applied to plan/`provider_sig`.
    pub fn resource_registry(&self) -> crate::provider_bind::ResourceRegistry {
        use crate::provider_bind::{ResourceBinding, ResourceRegistry};
        let mut registry = ResourceRegistry::builtin();
        for (resource, nominal) in &self.resource_bindings {
            registry.register(
                resource.clone(),
                match nominal {
                    HostResourceNominal::Core(core) => ResourceBinding::LegacyCore(*core),
                    HostResourceNominal::Item(item) => ResourceBinding::Nominal(*item),
                },
            );
        }
        registry
    }

    pub fn provider_call(&self, id: ProviderCallId) -> Option<&ValidatedProviderCall> {
        self.provider_calls.get(id.0 as usize)
    }
}

// ---------------------------------------------------------------------- dump --

impl MirProgram {
    /// Deterministic, line-oriented dump (contract §11). Stable across runs for identical
    /// input; consumed by tests, review, and the C4.4 differential harness.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "// STARK MIR v{} (runtime-surface {})",
            self.mir_version, self.runtime_surface
        );
        for body in &self.bodies {
            let _ = writeln!(out);
            let _ = writeln!(out, "fn {} {{", body.instance.symbol);
            let mut locals_line = String::from("  locals:");
            for (i, decl) in body.locals.iter().enumerate() {
                if i > 0 {
                    locals_line.push(',');
                }
                let _ = write!(
                    locals_line,
                    " _{i}: {} [{}]",
                    dump_ty(&decl.ty),
                    dump_local_kind(&decl.kind)
                );
            }
            let _ = writeln!(out, "{locals_line}");
            for (bi, block) in body.blocks.iter().enumerate() {
                let _ = writeln!(out, "  bb{bi}:");
                for (stmt, info) in &block.statements {
                    let _ = writeln!(
                        out,
                        "    {}  // {}",
                        dump_statement(stmt),
                        self.dump_source(info)
                    );
                }
                let (term, info) = &block.terminator;
                let _ = writeln!(
                    out,
                    "    {}  // {}",
                    self.dump_terminator(term),
                    self.dump_source(info)
                );
            }
            let _ = writeln!(out, "}}");
        }
        out
    }

    fn dump_source(&self, info: &SourceInfo) -> String {
        let file = &self.files[info.file.0 as usize];
        let (line, col) = file.line_col(info.span.lo);
        let origin = match info.origin {
            Origin::UserCode => String::new(),
            Origin::Synthetic(kind) => format!(" synthetic:{kind:?}"),
        };
        format!("{}:{line}:{col}{origin}", file.name)
    }

    fn dump_terminator(&self, term: &Terminator) -> String {
        match term {
            Terminator::Goto { target } => format!("goto bb{}", target.0),
            Terminator::SwitchInt {
                scrut,
                arms,
                otherwise,
            } => {
                let arms_text = arms
                    .iter()
                    .map(|(v, b)| format!("{v} -> bb{}", b.0))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "switchInt({}) [{arms_text}] otherwise bb{}",
                    dump_operand(scrut),
                    otherwise.0
                )
            }
            Terminator::Call {
                callee,
                args,
                dest,
                target,
            } => {
                let callee_text = match callee {
                    Callee::Instance(instance) => instance.symbol.clone(),
                    Callee::FnValue(op) => format!("fnvalue({})", dump_operand(op)),
                    Callee::Runtime(rt) => format!("runtime:{rt:?}"),
                    // A10: the dump names the PROVIDER and the verbatim symbol, not the id --
                    // a bare index would make a dump unreadable without the arena beside it.
                    Callee::Provider(id) => match self.provider_call(*id) {
                        Some(call) => {
                            format!("provider:{}:{}", call.provider.name, call.symbol())
                        }
                        // Unresolvable ids are a verification failure (V-PROV-1); the dump still
                        // has to render something, and it renders the defect rather than hiding it.
                        None => format!("provider:<unresolved #{}>", id.0),
                    },
                };
                format!(
                    "{} = call {callee_text}({}) -> bb{}",
                    dump_place(dest),
                    args.iter().map(dump_operand).collect::<Vec<_>>().join(", "),
                    target.0
                )
            }
            Terminator::Drop { place, target } => {
                format!("drop {} -> bb{}", dump_place(place), target.0)
            }
            Terminator::Checked {
                op,
                args,
                dest,
                target,
                trap,
            } => format!(
                "_{} = checked {op:?}({}) -> bb{} trap:{:?}",
                dest.0,
                args.iter().map(dump_operand).collect::<Vec<_>>().join(", "),
                target.0,
                trap.category
            ),
            Terminator::Trap { info, message } => match message {
                Some(op) => format!("trap {:?} msg({})", info.category, dump_operand(op)),
                None => format!("trap {:?}", info.category),
            },
            Terminator::Return => "return".to_string(),
            Terminator::Unreachable => "unreachable".to_string(),
        }
    }
}

fn dump_local_kind(kind: &LocalKind) -> String {
    match kind {
        LocalKind::Return => "ret".to_string(),
        LocalKind::Param(i) => format!("param{i}"),
        LocalKind::User(name) => format!("user \"{name}\""),
        LocalKind::Temp => "tmp".to_string(),
        LocalKind::DropFlag => "dropflag".to_string(),
        LocalKind::IndexProof => "idxproof".to_string(),
    }
}

/// Deterministic, injective textual rendering of a type. Doubles as the type-argument
/// mangling inside canonical instance symbols (contract §2: deterministic + injective +
/// stable for identical inputs; NOT a stable external ABI).
pub(crate) fn dump_ty(ty: &MirTy) -> String {
    match ty {
        MirTy::Struct(item, args) => dump_generic("struct", &format!("#{}", item.0), args),
        MirTy::Enum(EnumRef::User(item), args) => {
            dump_generic("enum", &format!("#{}", item.0), args)
        }
        MirTy::Enum(EnumRef::CoreOption, args) => dump_generic("Option", "", args),
        MirTy::Enum(EnumRef::CoreResult, args) => dump_generic("Result", "", args),
        MirTy::Enum(EnumRef::CoreOrdering, args) => dump_generic("Ordering", "", args),
        MirTy::Tuple(elems) => {
            let inner = elems.iter().map(dump_ty).collect::<Vec<_>>().join(", ");
            format!("({inner})")
        }
        MirTy::Array(elem, len) => format!("[{}; {len}]", dump_ty(elem)),
        MirTy::Slice(elem) => format!("[{}]", dump_ty(elem)),
        MirTy::Ref { mutable, inner } => {
            format!("&{}{}", if *mutable { "mut " } else { "" }, dump_ty(inner))
        }
        MirTy::FnPtr { params, ret } => format!(
            "fn({}) -> {}",
            params.iter().map(dump_ty).collect::<Vec<_>>().join(", "),
            dump_ty(ret)
        ),
        MirTy::Core(core, args) => dump_generic(&format!("{core:?}"), "", args),
        // A11: rendered by `ItemId` index here, deliberately -- `dump_ty` renders every nominal that
        // way (`struct#3`), and it has no program context to resolve a content path from. The
        // CANONICAL, order-stable identity A11 Q5 specifies lives in `lower::symbol_ty`, which does.
        MirTy::HostResource(r) => {
            let nominal = match r.nominal {
                HostResourceNominal::Core(c) => format!("core:{c:?}"),
                HostResourceNominal::Item(item) => format!("#{}", item.0),
            };
            format!("hostres#{}/{}@{nominal}", r.provider, r.resource)
        }
        simple => format!("{simple:?}"),
    }
}

fn dump_generic(head: &str, id: &str, args: &[MirTy]) -> String {
    if args.is_empty() {
        format!("{head}{id}")
    } else {
        format!(
            "{head}{id}<{}>",
            args.iter().map(dump_ty).collect::<Vec<_>>().join(", ")
        )
    }
}

fn dump_statement(stmt: &Statement) -> String {
    match stmt {
        Statement::Assign(place, rvalue) => {
            format!("{} = {}", dump_place(place), dump_rvalue(rvalue))
        }
        Statement::Nop => "nop".to_string(),
    }
}

fn dump_rvalue(rvalue: &Rvalue) -> String {
    match rvalue {
        Rvalue::Use(op) => dump_operand(op),
        Rvalue::UnOp(op, operand) => format!("{op:?}({})", dump_operand(operand)),
        Rvalue::BinOp(op, lhs, rhs) => {
            format!("{op:?}({}, {})", dump_operand(lhs), dump_operand(rhs))
        }
        Rvalue::Aggregate(kind, operands) => {
            let kind_text = match kind {
                AggKind::Struct(item) => format!("struct#{}", item.0),
                AggKind::Tuple => "tuple".to_string(),
                AggKind::Array(ty) => format!("array<{}>", dump_ty(ty)),
                AggKind::EnumVariant(EnumRef::User(item), v) => format!("enum#{}::v{v}", item.0),
                AggKind::EnumVariant(EnumRef::CoreOption, v) => format!("Option::v{v}"),
                AggKind::EnumVariant(EnumRef::CoreResult, v) => format!("Result::v{v}"),
                AggKind::EnumVariant(EnumRef::CoreOrdering, v) => format!("Ordering::v{v}"),
            };
            format!(
                "aggregate {kind_text}({})",
                operands
                    .iter()
                    .map(dump_operand)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        Rvalue::Discriminant(place) => format!("discriminant({})", dump_place(place)),
        Rvalue::RefOf { mutable, place } => {
            format!(
                "&{}{}",
                if *mutable { "mut " } else { "" },
                dump_place(place)
            )
        }
        // A4: the queried type is part of the dump — that it survives lowering is the whole
        // point of the amendment, so it must be visible in the textual contract.
        Rvalue::LayoutQuery { kind, ty } => match kind {
            LayoutKind::SizeOf => format!("layout_size_of({})", dump_ty(ty)),
            LayoutKind::AlignOf => format!("layout_align_of({})", dump_ty(ty)),
        },
    }
}

fn dump_place(place: &Place) -> String {
    let mut text = format!("_{}", place.local.0);
    for projection in &place.projection {
        match projection {
            Projection::Field(i) => {
                let _ = write!(text, ".{i}");
            }
            Projection::VariantField(v, i) => {
                let _ = write!(text, ".v{v}.{i}");
            }
            Projection::Deref => text = format!("(*{text})"),
            Projection::Index(proof) => {
                let _ = write!(text, "[proof _{}]", proof.0);
            }
            Projection::ConstIndex(i) => {
                let _ = write!(text, "[{i}]");
            }
        }
    }
    text
}

fn dump_operand(op: &Operand) -> String {
    match op {
        Operand::Copy(place) => format!("copy {}", dump_place(place)),
        Operand::Move(place) => format!("move {}", dump_place(place)),
        Operand::Const(constant) => match constant {
            Constant::Int(v, ty) => format!("const {v}{}", dump_ty(ty)),
            Constant::Float(v, ty) => format!("const {v}{}", dump_ty(ty)),
            Constant::Bool(v) => format!("const {v}"),
            Constant::Unit => "const ()".to_string(),
            Constant::FnPtr(instance) => format!("const fnptr {}", instance.symbol),
            Constant::Str(s) => format!("const \"{}\"", s.escape_default()),
        },
    }
}
