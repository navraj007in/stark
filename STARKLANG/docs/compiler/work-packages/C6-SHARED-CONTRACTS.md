# C6 Shared Contracts — Frozen at the C6.0 Barrier

**Status:** FROZEN v1 (C6.0)
**Frozen:** 2026-07-23
**Integration base SHA:** `db73afe` (CD-077 — Gate C5 CLOSED; code state identical to qualification head `19254086`/CD-076, CD-077 being docs-only)
**Purpose:** the authority-bearing contracts every C6 track must consume unchanged. A track may not
alter anything here without a recorded CE decision (CE3 for MIR/verifier/version, CE4 for runtime
ABI/layout/Drop/resource/trap) and the integration lead's merge of the change. This document is the
single source of truth; tracks cite it rather than re-deriving it.

C6.0 adds no Core feature and changes no behaviour.

---

## 1. Versions (frozen)

| Contract | Value | Authority |
|---|---|---|
| MIR shape version | `0.1` | `starkc/src/mir/mod.rs` `MIR_VERSION` |
| MIR runtime surface | `0.1-A8` | `starkc/src/mir/mod.rs` `MIR_RUNTIME_SURFACE` |
| Generated-Rust backend version | `0.1` | `starkc/src/backend/version.rs` `BACKEND_VERSION` |
| Native runtime version | `0.1` | `starkc/stark-runtime/src/version.rs` `RUNTIME_VERSION` |
| Target-layout contract | `stark-64-v1` (rev `1`) | `starkc/src/layout.rs` (`contract_for`) |

Changing any value is a CE3 (MIR/version) or CE4 (runtime) escalation. A runtime-version bump
requires generated-code tests, installed-layout tests, and an offline-build proof (WP-C6.3 §30).

---

## 2. Backend precondition

The generated-Rust backend consumes a **`VerifiedMirProgram`**, constructible only via
`mir::verify::verify_program`. No backend path may bypass MIR verification. Verified MIR is
**monomorphised-only**: no unresolved type parameter or inference variable may reach a backend
(`starkc/src/backend/generated_rust/mod.rs`; `starkc/src/mir/verify.rs`).

STARK owns monomorphisation and instance discovery; the backend never re-runs resolution, lookup,
or instance discovery.

---

## 3. Instance / canonical-symbol contract

`mir::Instance { item: hir::ItemId, type_args: Vec<MirTy>, symbol: String }`.

- `symbol` is the **canonical callable identity**: deterministic and injective for identical
  inputs, produced by lowering (`mir::lower` `key_symbol`). It is **not** a stable external ABI.
- `MirProgram.bodies` is the authoritative concrete-body set, **sorted by canonical symbol**.
- Canonical identity is **relocation- and traversal-order-independent** — absolute paths, output
  directories, and discovery order must never enter it (proven: C5.4a + reference workspace
  relocation).
- The backend's sanitiser (`backend/generated_rust/mangle.rs`) is **injective** and maps the
  canonical symbol to a Rust identifier; function names, nominal-type names (`ty#…`), core-enum
  names (`core#…`), and function-pointer sentinels (`fn-sentinel#…`) occupy disjoint key spaces.
- Linkage validation (`backend/generated_rust/linkage.rs`) is the deterministic pre-rustc refusal
  boundary: unique/sorted symbols, unique generated names, every referenced instance resolving to
  exactly one body with matching `item`/`type_args`.

A track that finds two valid resolved instances sharing one canonical symbol, or one instance
receiving different symbols under relocation/order, **stops and escalates to the identity producer**
(CE3) — it does not patch the backend with a second identity scheme.

---

## 4. `ValueSlot<T>` invariants (ownership storage)

Authority: `starkc/stark-runtime/src/slot.rs`. A non-`Copy`, non-reference local is backed by a
`ValueSlot<T>`; a `Copy` or reference local is not slot-backed
(`backend/generated_rust/emit_types.rs::is_slot_backed`, reading `TypeContext::is_copy`).

- States: `SlotState` (dead / whole / partial). A slot starts **dead** (`ValueSlot::dead()`).
- API (the only sanctioned ownership operations): `write`, `get`, `get_mut`, `take` (moves out,
  records dead), `drop_value`, `drop_with(glue)`, `finish_partial`, `state`/`is_live`/`is_whole`.
- Liveness is **explicit** — whole-place and drop-unit liveness are tracked by the slot and by MIR
  drop flags, **never inferred from generated-Rust control flow**.
- `slot_violation(...)` aborts (a compiler defect, not a program fault).
- Function pointers are `Copy` (`TYPE-FN-001`) → never slot-backed.

C6.1's general non-`Copy` cross-block movement must preserve `ValueSlot<T>` **or an approved
successor** (CE4) and route every move/write/Drop through these reviewed helpers. Confined `unsafe`
lives only in reviewed runtime/helper modules, each invariant documented and tested.

**C6.1b additive amendment (not a CE4):** `ValueSlot::reinit(value)` — overwrite the slot regardless
of prior state, running no destructor. Sound ONLY for a no-drop type (a no-op `DropPlan`), where the
old value owns nothing to release. Used by the backend for a no-drop slot local's reassignment,
whose slot a MIR `Drop` never resets. Additive helper; no existing op, ABI, layout, or Drop-glue
behaviour changes.

---

## 5. `DropPlan` authority (destruction)

Authority: `starkc/src/mir/drop_plan.rs`. `DropPlan` (with `PlannedField`/`VariantPlan`) is the
**canonical destruction plan**, derived by `plan_for(ty, types)` and consumed by **both** the MIR
interpreter and the backend drop glue. The backend decides only how a step is spelled in Rust; it
decides nothing about order, coverage, or which components carry an obligation.

Frozen destruction semantics: own destructor before components; components in **reverse declaration
order** (`array_order`, `variant_payloads`); active-variant-only payload destruction; **exactly
once** (drop flags); **no Drop after an aborting trap**; no projected `Drop` collapses to a
whole-place drop over partial storage.

---

## 6. Trap observation contract

A trap carries `TrapCategory` + `SourceInfo { file: FileId, span, origin }`. Native behaviour:
abort (no unwind), process exit **101**, deterministic stderr rendering, source file/line/column
provenance pointing at **STARK source** (never generated Rust). `panic`/trap runs **no**
destructors. Message-carrying traps need `&str` values (WP-C6.3); message-less traps and
`assert`/`assert_eq`/`assert_ne` are supported today.

---

## 7. Runtime-call identity contract

`mir::Callee::Runtime(RuntimeFn)` (`starkc/src/mir/mod.rs`, `enum RuntimeFn`) is the frozen
runtime-surface identity at surface revision `0.1-A8`. The generated backend currently refuses
`Callee::Runtime` (`Unsupported`) — WP-C6.3 (Track C) implements the runtime surface behind
proven-equivalent wrappers. Adding or changing a `RuntimeFn` identity, or changing its signature,
is a CE3/CE4 escalation.

---

## 8. Three-engine comparator schema

Authority: `starkc/tests/three_engine_differential.rs`. Every case runs one source through HIR
oracle, verified-MIR interpreter, and native debug executable, each normalised to one `Outcome`:

```text
Completed { stdout: String, exit: i32 }
Trapped   { category: TrapCategory, file: String, line: u32, column: u32, stdout_before: String }
```

C6.5 extends this to the full observation shape (adds `stderr_bytes`, `returned_observation`,
`drop_log`, `message_class`) — see WP-C6-ENTRY §39. Only **normative observations** are compared,
never Cargo text or host backtraces. The comparator's semantic normalisation is owned by
Track B / integration; other tracks must not weaken it.

`NATIVE_STDOUT_SUPPORTED` is currently `false`; Track C flips it when native output lands, at which
point the comparator begins comparing real stdout bytes on all three sides.

---

## 9. Tier-1 targets

```text
aarch64-apple-darwin
x86_64-unknown-linux-gnu
```

A positive C6 claim requires both. `x86_64-pc-windows-msvc` is Tier-2 (Core v1 Compiler Stable, not
a C6 blocker); `x86_64-apple-darwin` is Tier-3. Unsupported targets must fail before linking with a
clear diagnostic (WP-C6.4).

---

## 10. Fixed decisions (not reopened in C6)

Native compilation mandatory; generated Rust is the backend; native emission consumes verified MIR;
STARK owns monomorphisation/instance discovery; MIR bodies concrete; Core dispatch static; trait
objects outside Core v1; function values non-capturing; panic/trap aborts; no Drop after aborting
trap; layout from a named versioned contract; stable Rust required; optimisation waits for C7; no
stable public ABI.

No host-semantic substitution: Rust Drop timing, HashMap iteration order, `derive`d
Eq/Ord/Hash/Display/Clone, Rust overflow, Rust panic unwinding, Rust `Option` ownership, Rust string
indexing, and Rust iterator behaviour must **not** define STARK semantics. Host types are allowed
only behind proven-equivalent wrappers.
