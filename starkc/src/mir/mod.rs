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

pub mod interp;
pub mod lower;
pub mod verify;

use crate::source::{SourceFile, Span};
use std::fmt::Write as _;
use std::sync::Arc;

/// Bumped whenever the MIR shape changes (contract §11). Consumers state the version they
/// accept; mismatch is a hard error.
pub const MIR_VERSION: &str = "0.1";

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
pub const MIR_RUNTIME_SURFACE: &str = "0.1-A4";

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

#[derive(Clone, Debug)]
pub enum Rvalue {
    Use(Operand),
    UnOp(MirUnOp, Operand),
    BinOp(MirBinOp, Operand, Operand),
    Aggregate(AggKind, Vec<Operand>),
    Discriminant(Place),
    RefOf { mutable: bool, place: Place },
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
    // --- 0.1-A4 (C4.6 A4-2b): checked interior access — `get`/`get_mut` return `Option<&T>`/
    // `Option<&mut T>` and DO NOT trap on out-of-bounds (they return `None`), distinct from the
    // trapping `VecIndexGet`/`v[i]`. The reference is an interior borrow into the live Vec. ---
    VecGetRef,
    VecGetMutRef,
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
    HashMapKeysIterNew,
    HashMapKeysIterNext,
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
