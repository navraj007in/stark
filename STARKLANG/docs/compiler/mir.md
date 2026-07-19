# STARK MIR v0.1 — Mid-level Intermediate Representation Contract

**Status: PROPOSED — pending CE3 owner review (WP-C4.1).** Nothing below is binding until the
owner approves it; C4.2 lowering does not begin against an unapproved contract. Prepared
2026-07-19 under Gate C4.

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
  a dropped type, and every trait-method instance the front end resolved. Instantiation is
  finite or compilation fails with a compiler-limit diagnostic (resource classification per
  C2.9); the depth limit is a named constant in the implementation.
- **Deterministic symbol identity:** an instance's canonical name is
  `⟨package⟩::⟨module path⟩::⟨item name⟩@⟨mangled type args⟩`, where the mangling is a
  deterministic, injective encoding of the `MirTy` vector (exact scheme fixed at implementation
  with tests; requirement here is determinism + injectivity + stability across identical
  inputs). This satisfies C6.2's deterministic-symbol requirement, and — because TYPE-FN-001
  makes function-value identity unobservable — nothing more is semantically required of it.
- Rationale for monomorphised-only: the MIR interpreter (C4.4) must execute MIR without
  reinventing the HIR interpreter's type-erased model (which would validate nothing about
  monomorphisation); the direct backend requires it; the generated-Rust backend can consume it
  trivially (emitting concrete Rust). The rejected alternative — generic MIR with
  instantiation-at-codegen — would make C4.4's differential vacuous for exactly the generic
  workload items (9, 18, 21) the C3 workload froze. **DEV-064's fix belongs upstream of MIR**:
  the type checker must reject fn-value coercions whose instantiation is undetermined, so
  instance collection never sees an unnamed instantiation.

## 3. MIR types (`MirTy`)

A closed, first-order, fully concrete type language:

```text
MirTy ::= Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
        | Float32 | Float64 | Bool | Char | Unit | Never
        | Str                          -- unsized; appears only behind Ref
        | String
        | Struct(ItemId, Vec<MirTy>)   -- monomorphised nominal instance
        | Enum(ItemId, Vec<MirTy>)
        | Tuple(Vec<MirTy>)
        | Array(Box<MirTy>, u64)
        | Slice(Box<MirTy>)            -- unsized; appears only behind Ref
        | Ref { mutable: Bool, inner: Box<MirTy> }
        | FnPtr { params: Vec<MirTy>, ret: Box<MirTy> }
        | Core(CoreType, Vec<MirTy>)   -- Vec, Box, Option, Result, HashMap, HashSet, Range, …
```

- `Core(...)` types are **semantically opaque runtime types** at the MIR level: their operations
  are runtime calls (§7), their layout is a C5.1/ABI concern, and their behavior is fixed by the
  normative stdlib chapter (e.g. insertion-order HashMap iteration, CD-009). MIR does not open
  their representation. (`Option`/`Result` are deliberately *also* Core types rather than
  ordinary enums in v0.1, matching the current runtime; revisiting that is an explicit open
  question, §12.)
- `Never` types check as any type's bottom (the type of `panic(...)`/diverging calls).
- Well-formedness: `Str`/`Slice` only under `Ref` (V-TY-3); no `Param`/`Infer` (V-TY-2).

## 4. Bodies, locals, and blocks

```text
MirBody {
    instance:  Instance,
    sig:       (params: Vec<MirTy>, ret: MirTy),
    locals:    Vec<LocalDecl { ty: MirTy, kind: Return | Param(i) | User(name) | Temp | DropFlag }>,
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

```text
Place    ::= Local . Projection*
Projection ::= Field(u32)              -- struct/tuple field by index (declaration order)
             | VariantField(v, u32)    -- enum payload field, only after a discriminant test
             | Deref                   -- through Ref
             | Index(Local)            -- element of Array/Slice; MUST be dominated by a
                                       --   CheckIndex terminator that produced this Local (§6)
Operand  ::= Copy(Place) | Move(Place) | Const(Constant)
Constant ::= literal of a primitive/String type
           | FnPtr(Instance)           -- a function value (CD-021); Copy; no comparison ops
           | UnitConst
Rvalue   ::= Use(Operand)
           | UnOp(op, Operand)                     -- non-trapping unaries (not, float neg)
           | BinOp(op, Operand, Operand)           -- non-trapping only: comparisons on
                                                   --   comparable primitives, Bool ops,
                                                   --   float add/sub/mul (IEEE per CD-006)
           | Aggregate(AggKind, Vec<Operand>)      -- struct/tuple/array/enum-variant construction
           | Discriminant(Place)                   -- read enum discriminant as an integer
           | RefOf { mutable, place }              -- take a reference
```

- **Every operation that can trap is a terminator, not an rvalue** (§6). The rvalue set above is
  total (never traps, never calls user code). This is the contract's central honesty invariant:
  a backend that lowers only rvalues + terminators cannot accidentally skip a trap check,
  because the trap is control flow, not a side effect of an expression. (Integer negation traps
  on `MIN`, so `Neg` on integers is a checked terminator, not a `UnOp`; float division and
  remainder trap on zero divisors per CD-006, so they are checked terminators too.)
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
        -- if/match lowering; enum matching switches on Discriminant
  | Call { callee: Callee, args: Vec<Operand>, dest: Place, target: BlockId }
        -- no unwind edge exists; a trap inside the callee aborts the program
  | Checked { op: CheckedOp, args: Vec<Operand>, dest: Place, target: BlockId,
              trap: TrapInfo }
        -- trapping primitives: integer add/sub/mul/div/rem/neg, shifts,
        --   float div/rem (CD-006), numeric `as` casts, index bounds check
        --   (CheckIndex yields the proven in-bounds index local used by Index projections)
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
               | Panic | UnwrapNone | UnwrapErr | AssertFailure | …aligned with the
                 abstract machine's trap taxonomy
```

- **`Trap` and `Checked` failures abort** with the category and source location (feeding
  WP-C5.5's trap file:line requirement). There is no edge to a cleanup block because the
  language has none.
- `Statement ::= Assign(Place, Rvalue) | Drop(Place) | Nop`. **`Drop` is a statement, not a
  terminator** — justified precisely by abort semantics: a destructor that traps aborts the
  whole program, so Drop needs no failure edge, and its success path is simply the next
  statement. (This is a deliberate, load-bearing divergence from Rust MIR, where Drop is a
  terminator because of unwinding. It makes STARK MIR blocks materially larger and simpler.
  Flagged for CE3 attention, §12.)

## 7. Runtime surface

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

- Lowering emits explicit `Drop(place)` statements at scope exits, early exits (`return`,
  `break`, `continue`), and assignment-overwrite points, in **reverse declaration order**
  (including struct-field internal order per CD-011), exactly once per initialized value —
  implementing the abstract machine's destruction rules structurally.
- Conditionally-initialized locals get a `DropFlag` boolean local; lowering emits flag
  assignments at init/move points and branches on the flag (ordinary `SwitchInt`) around the
  `Drop`. **No special conditional-drop instruction exists** — drop flags are ordinary data +
  ordinary control flow, keeping the verifier and backends simple.
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
V-DISC-1 Discriminant/VariantField only on enum-typed places; SwitchInt arms ⊆ variant set
V-DROP-1 Drop only on places whose type can require dropping
V-DROP-2 drop-flag locals: Bool, written only Const(true/false), read only by SwitchInt
V-IDX-1  every Index(local) projection is dominated by the CheckIndex that defined that local
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

## 12. Open questions flagged for CE3 review

1. **Drop as a statement** (§6) — the abort-no-unwind justification is sound, but it is a
   divergence from the Rust-MIR shape contributors may know; confirm.
2. **All trapping ops as terminators** (§5/§6) — maximally explicit and differential-friendly,
   at the cost of more blocks; the alternative (trapping rvalues with an implicit abort) hides
   traps from the CFG. Recommend as written; confirm.
3. **Monomorphised-only MIR** (§2) — recommend as written; the alternative undermines C4.4's
   differential value. Confirm, including the whole-program-compilation assumption (Core
   promises no separate-compilation ABI, per C2.9).
4. **`Option`/`Result` as opaque Core runtime types rather than ordinary MIR enums** (§3) —
   matches the current runtime and keeps v0.1 small, but means `match` on them lowers through
   runtime tests rather than `Discriminant`. A future MIR version may migrate them to ordinary
   enums; doing it now would couple C4 to a runtime-representation change. Recommend opaque for
   v0.1; confirm.
5. **`CheckIndex`-dominates-`Index` discipline** (§5, V-IDX-1) vs. a single fused checked-index
   operation — fused is simpler for backends, split enables C7.4's bounds-check elimination
   later without changing MIR shape. Recommend split as written; confirm.

## 13. What this contract does not cover (owned elsewhere)

Value layout, calling convention, drop glue implementation, panic/trap ABI (WP-C5.1/CE4);
the Native Provider ABI (C5.1); optimisation passes (C7.4, on this IR, differential-tested);
closure environments and callable-ABI future-compatibility (the CD-021 callable memo,
recommended pre-C5.1); tensor-extension lowering (Track T; not part of Core MIR v0.1).
