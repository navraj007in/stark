# STARK MIR v0.1 — Mid-level Intermediate Representation Contract

**Status: APPROVED (CE3, 2026-07-19, CD-028) — verdict "approve with required changes", all
three required changes applied below.** The owner-confirmed decisions on the five §12 judgment
calls are recorded in §12; WP-C4.2 lowering may begin against this contract.

**Versioning policy (CE3 clarification, 2026-07-19, WP-C4.6 A2).** *Until Gate C4 closes and MIR
v0.1 is frozen for backend consumption, narrow additive shape amendments may remain within MIR
v0.1 only when individually approved under CE3 and recorded in this contract. After C4 exit, any
MIR shape change requires a MIR version bump.* This supersedes the earlier blanket "any shape
change requires a version bump" reading. Amendments approved under this policy to date:
**A1** (CD-031, `Constant::Str`, drop-elaborated `String`/`Vec`, `Trap.message`, runtime-surface
`RuntimeFn` groups — `mir-amendment-A1-strings-runtime.md`), **A2** (`EnumRef::CoreOrdering`,
the prelude `Ordering` as a logical MIR enum — `mir-amendment-A2-ordering.md`), **A3**
(bitwise/shift/exponentiation arithmetic — recorded in full below), and **A4**
(`Rvalue::LayoutQuery`, type-preserving `size_of`/`align_of` — `mir-amendment-A4-layout.md`,
CD-036). All are additive and remain MIR v0.1.

**A4 shape amendment (WP-C4.7-3, CD-036, approved 2026-07-20).** Adds one pure rvalue,
`LayoutQuery { kind: SizeOf | AlignOf, ty: MirTy }`, destination always `UInt64`. It replaces
WP-C4.6 A4-1's type-ERASING lowering of `size_of`/`align_of` to `Const 8`: 06-Standard-Library
classifies these as *target-layout queries* and LAYOUT-QUERY-001 makes them the only Core layout
observations, so a backend must be able to answer them from the MIR it is given — which it cannot
do if `T` was discarded. Because MIR is monomorphised, `ty` is always concrete (`size_of::<T>()`
in a generic body records the instantiation's type). The rvalue is pure — it cannot trap, call
user code, or diverge, so §5's totality invariant is unchanged, and it is deliberately NOT a
`RuntimeFn`: its only input is a type, it cannot trap, and layout is compile-time knowledge
rather than backend-supplied runtime. Each consumer answers it through a single layout service;
the C4 reference interpreter's returns `(8, 8)` for every type, i.e. **the same answers both
engines have always given — A4 changes the representation, not the behavior**, and the HIR oracle
is untouched. Real per-target numbers are C5.1's (LAYOUT-ABI-001 makes them target- and
version-dependent; CD-015 fixed none). Verifier rule: destination must be `UInt64` (MIR-0004);
the queried type is unconstrained, since `Sized`-ness is the checked front end's property. No
runtime-surface change (`0.1-A6` stands).

**A3 shape amendment (WP-C4.6 A5, under CD-033; recorded 2026-07-20 by WP-C4.7-1 as **CD-035**,
presented for post-hoc CE3 ratification).** CD-033 approved the A5 *class* of work (bitwise and shift
operators, integer exponentiation) but the per-amendment recording this policy requires was
missed at implementation time; this paragraph is that record. The amendment adds four shape
variants, all additive:

- **`MirBinOp::BitAnd | BitOr | BitXor`** — integer-only, both operands and the destination the
  same integer type. These are **pure `Rvalue::BinOp`s, not `Checked` terminators**: for
  same-width two's-complement operands the bitwise result is always representable in the operand
  width, so no range check is owed and the §5 totality invariant is preserved. Unary `~x`
  lowers to `x ^ mask` (the all-ones constant of the operand's width) rather than adding a
  `MirUnOp` — a deliberate choice to keep the shape addition to the binary set.
- **`CheckedOp::Pow`** — integer exponentiation (NUM-INT-ARITH-001). The exponent must be
  nonnegative; each intermediate multiply is checked and the final value range-checked against
  the destination type. A negative exponent and an unrepresentable result both trap as
  `IntegerOverflow` (the terminator's own category).
- **`CheckedOp::Shl | Shr`** — reserved in the original v0.1 draft, now **ACTIVE**. Per
  NUM-SHIFT-001 the shift count must be nonnegative and strictly less than the bit width of the
  left operand (= the destination type); there is no masking or count reduction. A left shift
  additionally traps when its result is not representable in the destination type. Right shift
  is arithmetic for signed destinations and logical for unsigned.
- **`TrapCategory::InvalidShift`** — a shift count that is negative or ≥ the operand width.
  Deliberately **distinct from `IntegerOverflow`**, which a left shift still raises when its
  result is out of range, so the two failure modes remain separately observable to a backend and
  to the differential comparator. Mechanically, a `Checked` terminator carries one category in
  its `TrapInfo`; the reference interpreter's `CheckedOutcome::Trap(Some(category))` **overrides**
  it for this case, which is the only category override in the evaluator. A backend must
  reproduce the same override rule for `Shl`/`Shr`.

No runtime-surface identifier changes: A3 adds no `RuntimeFn`, so `MIR_RUNTIME_SURFACE` is
untouched by this amendment.

This document is a **non-normative implementation contract**, not language specification. The
language's runtime authority is `STARKLANG/docs/spec/CORE-V1-ABSTRACT-MACHINE.md` and the
normative Core v1 chapters; where this contract and the abstract machine could ever disagree,
the abstract machine wins and this contract has a bug. MIR exists so that *one* validated
representation of Core execution semantics sits between the type-checked front end and every
backend (charter §1.2), so that backends cannot silently reinterpret the language.

## 1. Position in the pipeline and design constraints

```text
source → parse → resolve → type/flow/borrow check → typed HIR
      → [monomorphise + lower]  →  MIR  → [verify] → backend
                                      ↘  MIR interpreter (differential oracle vs HIR interpreter)
```

Fixed constraints this contract inherits (with sources):

- **Backend-neutral** (CD-026): the initial production backend is generated Rust, but the MIR
  must be equally consumable by a direct backend (Cranelift) — no construct may assume "the
  backend is a high-level language" (e.g. no re-use of Rust's own overflow or drop semantics as
  an implicit part of MIR meaning; everything observable is explicit in MIR).
- **Traps abort; there is no unwinding** (Core v1, `CORE-V1-ABSTRACT-MACHINE.md`). This is the
  single largest structural simplification versus Rust's MIR: **there are no cleanup edges, no
  landing pads, no unwind targets anywhere in this IR.** A trap terminates the program;
  destructors do not run.
- **Deterministic evaluation order is frozen** (CD-007/CD-010): strict left-to-right operands
  and call arguments, receiver before arguments, RHS before LHS-place for assignment,
  condition/scrutinee before branches. MIR's explicit statement sequencing makes this
  *structural*: a lowering that emits statements in the wrong order produces observably wrong
  MIR, catchable by the differential corpus.
- **All trait and method dispatch is static** (no `dyn` in Core v1): every call in MIR is either
  a direct call to a monomorphised instance or an indirect call through a function value
  (CD-021/CD-027). There are no vtables.
- **Function values are non-capturing code references**, `Copy`, with unobservable identity
  (TYPE-FN-001/002): a function value in MIR is a bare instance reference; no environment, no
  comparison operations.
- **Borrow checking is complete before MIR exists.** MIR carries references and borrows only as
  much as lowering and backends need (place semantics, aliasing-free by prior proof); it does
  not re-verify borrow rules.
- **No optimisation IR, no VM bytecode** (roadmap WP-C4.1 exclusion): this is a single-level,
  execution-faithful IR. C7.4's permitted optimisations operate on this same IR under
  differential tests; a production VM does not exist (charter §1.6 rule 11).

## 2. Compilation unit and monomorphisation representation

**Proposal: verified MIR is monomorphised-only.** A MIR *program* is the set of function
**instances** reachable from the entry point(s), where an instance is:

```text
Instance = (ItemId of the fn or method, Vec<MirTy> concrete type arguments)
```

- Generic functions have **no** MIR body of their own; each concrete instantiation gets one.
  `Ty::Param`/`Ty::Infer` are **prohibited** in verified MIR (verifier rule V-TY-2).
- Instance discovery starts at `main` (or the harness entry) and transitively collects every
  direct callee instance, every function-value constant's instance, every Drop impl instance of
  a dropped type, and every trait-method instance the front end resolved. Discovery is
  **deterministic and deduplicating**: identical instances reached through multiple call paths
  are one instance. Recursive or explosive instantiation fails through a **named
  compiler-resource limit** with a compiler-limit diagnostic (resource classification per
  C2.9) — never an arbitrary crash.
- **Deterministic symbol identity:** an instance's canonical name is
  `⟨package⟩::⟨module path⟩::⟨item name⟩@⟨mangled type args⟩`, where the mangling is a
  deterministic, injective encoding of the `MirTy` vector (exact scheme fixed at implementation
  with tests; requirement here is determinism + injectivity + stability across identical
  inputs). This satisfies C6.2's deterministic-symbol requirement, and — because TYPE-FN-001
  makes function-value identity unobservable — nothing more is semantically required of it.
  **The textual mangling is reproducible for identical compiler inputs but is *not* a stable
  external ABI** (CE3 qualification): Core v1 promises no ABI (C2.9), and the scheme may change
  with the MIR version.
- Rationale for monomorphised-only: the MIR interpreter (C4.4) must execute MIR without
  reinventing the HIR interpreter's type-erased model (which would validate nothing about
  monomorphisation); the direct backend requires it; the generated-Rust backend can consume it
  trivially (emitting concrete Rust). The rejected alternative — generic MIR with
  instantiation-at-codegen — would make C4.4's differential vacuous for exactly the generic
  workload items (9, 18, 21) the C3 workload froze. **DEV-064's fix belongs upstream of MIR**:
  the type checker must reject fn-value coercions whose instantiation is undetermined, so
  instance collection never sees an unnamed instantiation.

**Program shape (CD-029 amendment, CE3-reviewed):** a MIR program consists of (a) the interned
source-file table (`FileId` indexes it), (b) the body set, and (c) the **nominal type
context** — the struct field types and user-enum variant payload types for every nominal type
reachable from the bodies (`Option`/`Result` payloads derive from their type arguments and
need no table entry). The type context is *part of the in-memory MIR compilation unit*: the
verifier requires it to type projections step by step, and every backend requires it for
layout; it is not serialized in the textual dump. This is an additive amendment recorded under
CD-029; the MIR version remains v0.1. *Implementation note (WP-C4.5c):* because a nominal type
in monomorphised-only MIR is an instance, the context's entries are keyed per
`(ItemId, type arguments)` — `Pair<Int32>` and `Pair<Bool>` are distinct entries; non-generic
nominals key with an empty argument vector. This realizes the paragraph above for generic
nominals and is not a shape/version change. *Implementation note (WP-C4.5d):* the in-memory
context additionally carries the destructor instance symbol per nominal instance with an own
`Drop` impl (`drop_impls`), which is how `Drop`-terminator glue dispatches destructors —
same in-memory-companion status, not dump-serialized, no version change.

## 3. MIR types (`MirTy`)

A closed, first-order, fully concrete type language:

```text
MirTy ::= Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
        | Float32 | Float64 | Bool | Char | Unit | Never
        | Str                          -- unsized; appears only behind Ref
        | String
        | Struct(ItemId, Vec<MirTy>)   -- monomorphised nominal instance
        | Enum(EnumRef, Vec<MirTy>)    -- user enums AND Option/Result/Ordering (logical enums, below)
        | Tuple(Vec<MirTy>)
        | Array(Box<MirTy>, u64)
        | Slice(Box<MirTy>)            -- unsized; appears only behind Ref
        | Ref { mutable: Bool, inner: Box<MirTy> }
        | FnPtr { params: Vec<MirTy>, ret: Box<MirTy> }
        | Core(CoreType, Vec<MirTy>)   -- Vec, Box, HashMap, HashSet, Range, iterators, …
                                       --   (NOT Option/Result — see below)

EnumRef ::= User(ItemId) | CoreOption | CoreResult | CoreOrdering
```

- **`Option<T>`, `Result<T, E>`, and `Ordering` are logical MIR enums** (CE3 required change;
  `CoreOrdering` added by Amendment A2): they use exactly the same
  `Aggregate`/`Discriminant`/`VariantField`/`SwitchInt` machinery as user enums, with
  `EnumRef::CoreOption`/`CoreResult`/`CoreOrdering` supplying their nominal identity (they have
  no user `ItemId`). `CoreOption` has variants `None = 0`, `Some(T) = 1`; `CoreResult` has
  `Ok(T) = 0`, `Err(E) = 1`; `CoreOrdering` (the prelude `Ordering`) has the three fieldless
  variants `Less = 0`, `Equal = 1`, `Greater = 2`. Their **physical layout remains a C5.1/ABI
  decision** — MIR does not decide niche-vs-tag representation, and these logical discriminants
  are not a physical ABI. Rationale: one enum system, one match-lowering path,
  one discriminant discipline; the alternative (opaque runtime types) would have let the
  *current interpreter's* internal representation shape the IR, exactly the coupling this
  contract exists to prevent. Runtime calls may still implement higher-level *combinators*
  (`map`, `and_then`, formatting) in v0.1, but construction and pattern matching never require
  a runtime call.
- The remaining `Core(...)` types are **semantically opaque runtime types** at the MIR level:
  their operations are runtime calls (§7), their layout is a C5.1/ABI concern, and their
  behavior is fixed by the normative stdlib chapter (e.g. insertion-order HashMap iteration,
  CD-009). MIR does not open their representation.
- `Never` types check as any type's bottom (the type of `panic(...)`/diverging calls).
- Well-formedness: `Str`/`Slice` only under `Ref` (V-TY-3); no `Param`/`Infer` (V-TY-2).

## 4. Bodies, locals, and blocks

```text
MirBody {
    instance:  Instance,
    sig:       (params: Vec<MirTy>, ret: MirTy),
    locals:    Vec<LocalDecl { ty: MirTy,
                               kind: Return | Param(i) | User(name) | Temp | DropFlag
                                   | IndexProof }>,   -- opaque; see §6's proof tokens
    blocks:    Vec<BasicBlock { statements: Vec<Statement>, terminator: Terminator }>,
    entry:     BlockId,
    spans:     per-statement/terminator SourceInfo (see §9),
}
```

- `Local(0)` is the **return place**, written before `Return` (Rust-MIR convention, adopted
  deliberately for familiarity and tooling reuse).
- Every local is typed; `DropFlag` locals are `Bool` and exist only for drop elaboration (§8).
- Every block ends in exactly one terminator (V-CFG-1); all block/local references are in
  bounds (V-CFG-2); the graph is reachable-from-entry or the dead blocks are absent (lowering
  shouldn't emit unreachable blocks; verifier warns, does not fail, in v0.1).

## 5. Places, operands, rvalues

**Amended by A1 (CD-031, runtime surface `0.1-A1`):** `Constant` gains a `Str(String)` form (a
`&str`-typed decoded immutable UTF-8 literal); `String`/`Vec` become drop-elaborated runtime
values; and `Terminator::Trap` gains an optional `message: Option<Operand>` (§6/§8). All
additive within v0.1 — see `mir-amendment-A1-strings-runtime.md`.

```text
Place    ::= Local . Projection*
Projection ::= Field(u32)              -- struct/tuple field by index (declaration order)
             | VariantField(v, u32)    -- enum payload field, only after a discriminant test
             | Deref                   -- through Ref
             | Index(ProofLocal)       -- element of Array/Slice, consuming an index-proof
                                       --   token produced by a CheckIndex terminator (§6);
                                       --   ordinary integer locals are NOT accepted here
Operand  ::= Copy(Place) | Move(Place) | Const(Constant)
Constant ::= literal of a primitive/String type
           | FnPtr(Instance)           -- a function value (CD-021); Copy; no comparison ops
           | UnitConst
Rvalue   ::= Use(Operand)
           | UnOp(op, Operand)                     -- non-trapping unaries (not, float neg)
           | BinOp(op, Operand, Operand)           -- non-trapping only: comparisons on
                                                   --   comparable primitives, Bool ops,
                                                   --   float add/sub/mul (IEEE per CD-006),
                                                   --   integer bitwise and/or/xor (A3)
           | Aggregate(AggKind, Vec<Operand>)      -- struct/tuple/array/enum-variant construction
           | Discriminant(Place)                   -- read enum discriminant as an integer
           | RefOf { mutable, place }              -- take a reference
           | LayoutQuery { kind, ty }              -- A4: size_of/align_of, dest UInt64; the
                                                   --   queried type is PRESERVED (concrete,
                                                   --   since MIR is monomorphised)
```

- **The statement and rvalue sets are total: they never trap, never call user code, and never
  diverge.** Every operation that can trap, call user code, or diverge is a terminator (§6) —
  including `Drop`, which runs a user destructor (CE3 required change; the original draft had
  `Drop` as a statement, violating this very invariant). This is the contract's central honesty
  invariant: a backend that lowers only rvalues + terminators cannot accidentally skip a trap
  check or hide a user-code call, because both are control flow, not expression side effects.
  (Integer negation traps on `MIN`, so `Neg` on integers is a checked terminator, not a `UnOp`;
  float division and remainder trap on zero divisors per CD-006, so they are checked
  terminators too.)
- `Move` vs `Copy` on operands is decided by the front end's frozen Copy semantics (fn values,
  references, primitives, all-Copy aggregates are `Copy` — TYPE-FN-001, 03-Type-System §Copy
  and Drop). The verifier enforces move-before-use at the MIR level (V-MOVE-1) as a *defense in
  depth* behind borrowck, per WP-C4.3.
- `Aggregate` construction is complete-in-one-statement: partial initialization across
  statements is not representable in v0.1 (lowering uses temporaries; the abstract machine's
  partial-aggregate-trap-cleanup rule is preserved because the trap aborts — nothing to clean).

## 6. Terminators

```text
Terminator ::=
    Goto { target }
  | SwitchInt { scrut: Operand, arms: Vec<(u128, BlockId)>, otherwise: BlockId }
        -- if/match lowering; enum matching (incl. Option/Result) switches on Discriminant
  | Call { callee: Callee, args: Vec<Operand>, dest: Place, target: BlockId }
        -- no unwind edge exists; a trap inside the callee aborts the program
  | Drop { place: Place, target: BlockId }
        -- run the destructor for `place`, then continue at `target`. NO unwind edge: a
        --   destructor that traps aborts the program. A terminator (not a statement, CE3
        --   required change) because it invokes user code that may trap, diverge, or mutate
        --   observable state.
  | Checked { op: CheckedOp, args: Vec<Operand>, dest: Place | ProofLocal, target: BlockId,
              trap: TrapInfo }
        -- trapping primitives: integer add/sub/mul/div/rem/neg, shifts, pow (A3),
        --   float div/rem (CD-006), numeric `as` casts, and CheckIndex (below).
        -- Exactly ONE normal successor (`target`); the failure outcome is an implicit
        --   abort described by `trap` — there is no unwind or recovery successor of any kind.
  | Trap { info: TrapInfo }            -- unconditional: panic(msg), unwrap on None, assert, …
  | Return                             -- reads Local(0)
  | Unreachable                        -- statically-proven-impossible arms (verifier-guarded)

Callee ::= Instance(Instance)          -- direct, statically resolved (incl. trait methods)
         | FnValue(Operand)            -- indirect through a FnPtr-typed operand (CD-021)
         | Runtime(RuntimeFn)          -- stdlib/runtime surface: print/println, String/Vec/
                                       --   HashMap ops, Option/Result combinator plumbing where
                                       --   not inlined, provider calls (C5.1 ABI later)

TrapInfo ::= { category: TrapCategory, source: SourceInfo }
TrapCategory ::= IntegerOverflow | DivideByZero | IndexOutOfBounds | CastFailure
               | Panic | UnwrapNone | UnwrapErr | AssertFailure
               | InvalidShift (A3)   | …aligned with the
                 abstract machine's trap taxonomy
```

- **`Trap` and `Checked` failures abort** with the category and source location (feeding
  WP-C5.5's trap file:line requirement). There is no edge to a cleanup block because the
  language has none. **Execution outcomes carry the full `TrapInfo` — category AND
  `SourceInfo` — through every consumer** (CD-029): a differential comparator or backend that
  observes only the category is blind to wrong-location traps, so provenance is part of the
  observable trap outcome, not an implementation nicety.
- `Statement ::= Assign(Place, Rvalue) | Nop`. The statement set is total (§5); everything that
  can trap, diverge, or run user code — including `Drop` — is a terminator. (The original
  proposal placed `Drop` in the statement set; CE3 review correctly identified that as a
  violation of the totality invariant, since destructors are user code. The approved `Drop`
  terminator keeps every property the statement form wanted — no unwind edge, single normal
  continuation — while keeping the CFG honest about where user code runs.)
- **`CheckIndex` and index-proof tokens** (CE3 revised design): `CheckIndex { base, index }`
  succeeds by defining an **opaque index-proof token** into a dedicated `ProofLocal` — a local
  of kind `IndexProof` that semantically binds *(base place identity, index value, checked
  length)*. Proof tokens cannot be constructed by ordinary assignment, copied, moved, cast, or
  used with any base other than the one they were checked against; their only consumer is an
  `Index(ProofLocal)` projection on that base. Dominance of an ordinary integer local is *not*
  sufficient (the local could be reassigned, or the base/length could differ). In v0.1 the
  proof discipline covers fixed-length `Array` (verifier may validate against the compile-time
  length) and `Slice` through an unchanged reference; **`Vec` indexing lowers through `Runtime`
  operations** because its length is mutable — a future MIR version may extend the proof
  discipline to `Vec` with an explicit invalidation rule.

## 7. Runtime surface

**Amended by A1 (CD-031):** the String/str/Vec/slice `RuntimeFn` groups are enumerated in
`mir-amendment-A1-strings-runtime.md` §5 under runtime-surface identifier `0.1-A1`, stamped on
each `MirProgram` (`mir_version`/`runtime_surface`) and rejected by any consumer that does not
support it. Later reserved-op activation bumps `runtime_surface` via a dated reviewed
enumeration; `MIR_VERSION` stays `0.1`.

The `Runtime(RuntimeFn)` callee namespace covers the compiler-provided behavior the normative
stdlib specifies but Core programs cannot express: I/O (`print`/`println` with the normative
formatting incl. `canonical_float`), `String`/`Vec`/`HashMap`/`HashSet` operations with their
frozen contracts (insertion-order iteration per CD-009), `panic`, hashing (canonical FNV-1a per
CD-017), file/provider operations (signature only; the stable provider ABI is C5.1's). Each
`RuntimeFn` has a fixed MIR-level signature listed in an appendix table when C4.2 lowers it —
the contract requirement is: **the set is closed, enumerated, and versioned with the MIR**, so a
backend knows exactly what runtime it must supply and an unsupported `RuntimeFn` fails loudly at
codegen (charter rule: no unsupported construct reaches a backend silently).

## 8. Drop elaboration and drop flags

- Lowering emits explicit `Drop { place, target }` **terminators** at scope exits, early exits
  (`return`, `break`, `continue`), and assignment-overwrite points, in **reverse declaration
  order** (including struct-field internal order per CD-011), exactly once per initialized
  value — implementing the abstract machine's destruction rules structurally. Sequential drops
  form a chain of single-terminator blocks; this is the accepted block-count cost of keeping
  user-code execution visible in the CFG.
- Conditionally-initialized locals get a `DropFlag` boolean local; lowering emits flag
  assignments at init/move points and branches on the flag (ordinary `SwitchInt`) around the
  `Drop` terminator. **No special conditional-drop instruction exists** — drop flags are
  ordinary data + ordinary control flow, keeping the verifier and backends simple.
- On any trap path: no drops execute (abort). The differential corpus's instrumented-drop
  fixtures (C6.5) are the enforcement mechanism for order/exactly-once, complementing the
  verifier's structural checks (V-DROP-1: `Drop` only on droppable-typed places; V-DROP-2:
  `DropFlag` locals only assigned `Const(true/false)` and only read by `SwitchInt`).

## 9. Provenance

```text
SourceInfo ::= { file: FileId, span: Span, origin: UserCode | Synthetic(SyntheticKind) }
SyntheticKind ::= DropElaboration | ForLoopDesugar | DropFlagInit | ReturnSlot | …
```

Every statement and terminator carries `SourceInfo`. `FileId` is explicit — MIR must not repeat
the front end's DEV-006 mistake of file-less spans; the verifier fails MIR whose `SourceInfo` is
absent (V-SRC-1). Synthetic origins are labeled rather than borrowing an arbitrary user span, so
trap reports and future debug info never point at misleading source (roadmap WP-C4.2: "every
MIR instruction retains a source span or documented synthetic origin").

## 10. Verifier obligations (WP-C4.3 mapping)

```text
V-CFG-1  every block has exactly one terminator; targets in bounds
V-CFG-2  all Local/Place references in bounds; projections type-correct step by step
V-TY-1   Assign LHS/RHS types agree; Call/Checked dest and arg types match callee sigs
V-TY-2   no Param/Infer types anywhere (monomorphised-only invariant)
V-TY-3   Str/Slice appear only under Ref
V-MOVE-1 no use of a moved-from place on any path (move-before-use, defense in depth)
V-DISC-1 Discriminant/VariantField only on enum-typed places (user AND CoreOption/CoreResult);
         SwitchInt arms ⊆ variant set
V-DROP-1 Drop terminator only on places whose type can require dropping; no unwind successor
V-DROP-2 drop-flag locals: Bool, written only Const(true/false), read only by SwitchInt
V-IDX-1  Index(proof) projections consume an IndexProof local defined by a CheckIndex on the
         SAME base place; proof locals are never assigned by statements, never copied/moved,
         never used with a different base, and are single-base-bound (CE3 revised design)
V-IDX-2  IndexProof locals appear only as CheckIndex dests and Index projection arguments
V-FN-1   no BinOp/UnOp on FnPtr operands (TYPE-FN-001: no comparison, no arithmetic)
V-SRC-1  every statement/terminator has SourceInfo with a real FileId
V-RT-1   every Runtime callee is in the enumerated RuntimeFn set for this MIR version
```

Invalid MIR is a **compiler-internal diagnostic** (`MIR-xxxx` namespace, charter §5.1) and a
safe compilation failure — never undefined backend behavior.

## 11. Textual dump format and versioning

- `starkc --emit=mir` (or equivalent test-harness hook) prints a deterministic, line-oriented
  dump: header `// STARK MIR v0.1`, one section per instance in sorted canonical-name order,
  locals table, then blocks. Deterministic across runs (charter §2.5) and diffable — the C4.4
  differential and code review both consume it.
- The version tag is bumped whenever this contract changes shape (new statement/terminator/
  type). The MIR interpreter, verifier, and every backend state the version they consume;
  mismatch is a hard error, not a warning.

Example (illustrative, not yet emitted by any tool):

```text
// STARK MIR v0.1
fn demo::main@[] {
  locals: _0: Unit [ret], _1: Int64 [user "x"], _2: Int64 [tmp]
  bb0:
    Checked { op: AddInt64, args: [Const 2, Const 3], dest: _2, target: bb1,
              trap: { IntegerOverflow, main.stark:2:13 } }
  bb1:
    Call { callee: Runtime(PrintlnInt64), args: [Copy _2], dest: _0, target: bb2 }
  bb2:
    Return
}
```

**A4 dump grammar addition (CD-036).** A layout query renders with its queried type, using the
dump's ordinary `MirTy` rendering — the type surviving into the textual contract is the whole
point of the amendment, so it must be visible to a reader of the dump:

```text
_3 = layout_size_of(Int32)
_4 = layout_align_of(struct#7<Int32>)
```

## 12. CE3 review outcomes (owner decisions, 2026-07-19, CD-028)

Verdict: **APPROVE WITH REQUIRED CHANGES** — all applied above.

1. **Drop** — *Revise (required change, applied).* Owner decision: "Drop remains explicit and
   has no unwind edge, but it is represented as a terminator because it may invoke user code,
   diverge, or abort. Successful completion transfers to one normal target; failure aborts the
   process." The original statement form violated the totality invariant (§5) — destructors are
   user code.
2. **Trapping operations as terminators** — *Approved as written.* Owner decision: "Every
   primitive operation that may trap is represented by an explicit terminator with one normal
   successor and an aborting failure outcome. MIR size is secondary to semantic visibility in
   v0.1." The one-normal-successor / implicit-abort / no-recovery refinement is now explicit in
   §6.
3. **Monomorphised-only MIR** — *Approved for v0.1.* Owner decision: "Verified MIR contains
   only concrete instances; unresolved type parameters are prohibited. Instance discovery is
   deterministic, deduplicated, and resource-bounded. Symbol mangling is reproducible for
   identical compiler inputs but is not yet a stable external ABI." All three qualifications
   are now in §2.
4. **`Option`/`Result`** — *Revise (required change, applied).* Owner decision: "`Option` and
   `Result` are represented as logical enums in MIR and use the same aggregate, discriminant,
   payload, and match machinery as user enums. Their physical layout remains an ABI/backend
   concern. Runtime calls may implement library combinators but do not define their fundamental
   enum semantics." The opaque form would have let the current interpreter's representation
   shape the IR — the exact coupling this contract prohibits.
5. **`CheckIndex`/`Index`** — *Approved with revision (required change, applied).* Owner
   decision: "Confirm the split check/access design, but replace the ordinary checked-index
   integer local with an opaque index-proof token tied to the base, index, and applicable
   length. `Index` projections consume that proof. Dominance of an ordinary mutable local is
   not sufficient." See §6's proof tokens and V-IDX-1/2.

## 13. What this contract does not cover (owned elsewhere)

Value layout, calling convention, drop glue implementation, panic/trap ABI (WP-C5.1/CE4);
the Native Provider ABI (C5.1); optimisation passes (C7.4, on this IR, differential-tested);
closure environments and callable-ABI future-compatibility (the CD-021 callable memo,
recommended pre-C5.1); tensor-extension lowering (Track T; not part of Core MIR v0.1).
