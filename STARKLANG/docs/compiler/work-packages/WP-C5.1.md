# WP-C5.1 — Runtime ABI and Layout Design

Gate: C5 (Native Core Backend MVP). Scope from `COMPILER-ROADMAP.md` WP-C5.1, detailed by
`WP-C5-ENTRY.md` §14 (C5.1a/b/c). Escalation: **CE4 — runtime ABI, layout, drop glue, panic/trap,
and native resource model.** CE4 approval for the representation contract (entry plan §6–10) is
recorded under CD-042 (owner approved the entry plan at its recommended §19 choices).

## C5.1a — Representation decision

### Status: representation contract approved (CD-042); MirTy matrix and host targets pinned below.

Per `WP-C5-ENTRY.md` §14, C5.1a must deliver: an approved version of §6–10, the exact C5 supported
`MirTy` matrix, the exact non-`Copy` storage strategy, the exact move and Drop invariants, the
exact enum/`Option`/`Result` representation, the exact function-pointer representation, the exact
layout-query rule, and the host target for the first native proof.

#### Approved version of §6–10

`WP-C5-ENTRY.md` §6 (generated-Rust representation contract), §7 (ownership/move/Drop strategy),
§8 (layout and `LayoutQuery`), §9 (minimal runtime), and §10 (Native Provider ABI v0.1 scope) are
**approved as drafted** (CD-042). No changes made in this WP. Cross-references below point at
those sections rather than restating them.

#### Exact C5 supported `MirTy` matrix

Enumerated directly against `starkc/src/mir/mod.rs::MirTy` (19 base variants + the opaque
`Core(CoreType, Vec<MirTy>)` catch-all, `CoreType` from `starkc/src/hir.rs`). "IN" means C5.2/5.3
must lower it per §6.2's type mapping; "OUT" means the front end must never hand C5 a program that
needs it (§3.2's deferred-feature rule: reject before backend invocation, or absent from the C5
source profile).

| `MirTy` | C5 status | Representation authority |
|---|---|---|
| `Int8`/`Int16`/`Int32`/`Int64` | IN | §6.2: `i8`/`i16`/`i32`/`i64` |
| `UInt8`/`UInt16`/`UInt32`/`UInt64` | IN | §6.2: `u8`/`u16`/`u32`/`u64` |
| `Float32`/`Float64` | IN, gated | §6.2: `f32`/`f64`, canonical-float policy. §3.1 admits these "where already present in the frozen C5 cases" — the frozen reference workspace (§4, not yet authored) fixes the exact subset actually exercised, per §4.2's baseline-output freeze |
| `Bool` | IN | §6.2: `bool` |
| `Char` | IN | §6.2: `char` |
| `Unit` | IN | §6.2: `()` |
| `Never` | IN, no runtime value | No `MirTy::Never` local is ever live at a native trap/return boundary (no value of this type is constructible); backend needs no storage representation, only type-checking pass-through |
| `Str` | IN, reference-only | Unsized; appears only behind `Ref` (V-TY-3) — see `Ref` row |
| `String` | IN, minimal | §6.2: "C5 runtime-owned or Rust `String`, according to the approved Drop strategy" — exact choice is a C5.1b implementation decision, not reopened here |
| `Struct(ItemId, Vec<MirTy>)` | IN | §6.3: one concrete generated Rust struct per reachable nominal instance |
| `Enum(EnumRef::User, _)` | IN | §6.3: one concrete generated Rust enum per reachable nominal instance |
| `Enum(EnumRef::CoreOption, _)` | IN | §6.2: ordinary `Option<T>` preferred if all observable semantics match |
| `Enum(EnumRef::CoreResult, _)` | IN | §6.2: ordinary `Result<T, E>` preferred if all observable semantics match |
| `Enum(EnumRef::CoreOrdering, _)` | IN, structural only | Shares the same enum machinery; not in §4.1's required application content. Supported if the reference workspace ends up using `cmp`/`Ord`, but not a C5.1 commitment to include it |
| `Tuple(Vec<MirTy>)` | IN | §6.2: generated concrete tuple or named internal aggregate (one canonical form, chosen in C5.1b) |
| `Array(Box<MirTy>, u64)` | IN | §6.2: `[Generated<T>; N]` |
| `Slice(Box<MirTy>)` | OUT (default) | §3.2 defers "complete slice and mutable-slice native parity"; §6.2: "Internal fat-view representation when admitted into the C5 subset" — not admitted by default |
| `Ref { mutable, inner }` | IN, narrow | §6.2: "Internal reference/pointer representation chosen in C5.1; broad parity remains C6" — basic reference passing is in scope, general reference-heavy programs are not |
| `FnPtr { params, ret }` | IN | §6.2/§6.4: typed non-capturing Rust `fn` pointer or generated equivalent, one deterministic symbol per instance |
| `Core(CoreType::Vec, _)` | OUT (default) | §3.2 defers "broad `Vec` and iterator parity"; §6.2: "Deferred except for a separately approved minimal path" |
| `Core(CoreType::Box, _)` | OUT (default) | §3.2 defers "`Box` native parity beyond any minimal case separately admitted"; §6.2: "Deferred unless explicitly added to the C5 subset" |
| `Core(CoreType::HashMap \| HashSet, _)` | OUT | §3.2 defers explicitly ("HashMap and HashSet native representation") |
| `Core(CoreType::Range \| RangeInclusive, _)` | OUT | Not named individually in §3.1/§3.2, but falls under §3.2's "broad ... iterator parity" deferral — see scope note below |
| `Core(CoreType::CharsIter \| SplitIter \| VecIter \| KeysIter \| ValuesIter \| Iter \| MapIter \| FilterIter, _)` | OUT | §3.2: "broad `Vec` and iterator parity" |
| `Core(CoreType::Random, _)` | OUT | Not in §3.1's required subset; no C5 fixture needs it |
| `Core(CoreType::IOError \| File, _)` | OUT | §3.2: "executable provider/file-resource support" |
| `Core(CoreType::String \| Option \| Result \| Ordering, _)` | N/A | These `CoreType` variants never materialize as `MirTy::Core` — `String` lowers to `MirTy::String`, `Option`/`Result`/`Ordering` lower to `MirTy::Enum(EnumRef::Core*, _)` per `mir/mod.rs`'s module doc comment |

**Scope note (loops and ranges):** because every `Core(CoreType::*Iter | Range*, _)` variant is OUT
by default, the "a loop" requirement in §4.1's reference-workspace content must be satisfied with a
`while` loop or array iteration (`Array` is IN), not a `for x in a..b` range loop or Vec/HashMap
iteration, unless a minimal Vec or Range path is separately approved before §4's workspace is
authored (C5.4d). Recorded here so C5.1's scope decision constrains C5.4's workspace design rather
than being rediscovered as a blocker then.

#### Exact non-`Copy` storage strategy, move/Drop invariants, enum representation, function-pointer
representation, layout-query rule

All approved as drafted in the entry plan, cited rather than restated:

- Non-`Copy` storage: `WP-C5-ENTRY.md` §7.2 (`MaybeUninit<ManuallyDrop<T>>` + explicit
  initialised/live state + explicit move/Drop helpers, simplifiable only when MIR dataflow proves
  always-initialised/never-partially-moved, evidence-based, no observable Drop change).
- Move invariants: §7.3 (five-step move: prove live, transfer without running Drop, mark source
  dead, initialise destination, preserve sub-place precision) — no `.clone()` substitution.
- Copy invariant: §7.4 (only for MIR types classified `Copy`; backend does not broaden the Copy set).
- Drop invariants: §7.5 (six-step: act only on a live place, run correct concrete Drop/structural
  glue, MIR-specified field order, mark dead, exactly once, never repeated by Rust scope exit).
  Generated nominal types do not implement Rust `Drop` in C5 (§7.5).
- Partial moves: §7.6 (every partial move in the C5 fixtures uses MIR's typed move/drop paths; no
  collapsing struct fields/variant fields/array indices into whole-local liveness).
- Trap path: §7.7 (abort, no pending Drop glue, no unwind, panic-abort crate profile).
- Enum/`Option`/`Result` representation: §6.2 (ordinary `Option<T>`/`Result<T, E>` preferred when
  observably equivalent) + §6.3 (no derived `Drop`/`Clone`/`Copy`/`Eq`/`Ord`/`Hash` as semantic
  shortcuts).
- Function-pointer representation: §6.2 + §6.4 (typed non-capturing Rust `fn` pointer or generated
  equivalent; deterministic, injective, non-ABI symbol per instance).
- Layout-query rule: §8.2 (`core::mem::size_of::<GeneratedTy>()`/`align_of` against the canonical
  generated representation, or an observationally equivalent constant from one central layout
  service; never the C4 interpreter placeholder `(8, 8)`).

None of these are reopened by this WP; C5.1b implements against them as written.

#### Host target for the first native proof

**Decision (owner, 2026-07-21): both, from the start.**

| Role | Target triple | Toolchain |
|---|---|---|
| Primary (local iteration) | `aarch64-apple-darwin` | rustc 1.93.0 (`254b59607`, 2026-01-19) |
| Secondary (CI-enforced) | `x86_64-unknown-linux-gnu` | stable channel via `dtolnay/rust-toolchain@stable` on `ubuntu-latest`, unpinned patch version |

C5.1's exit condition ("one verified empty/scalar MIR program becomes a standalone executable")
must be demonstrated on **both** triples before C5.1 is called done — not deferred to C5.6. This is
stricter than the entry plan's default (which allowed a single first-proof target) and matches the
project's existing dual-toolchain validation habit (`cargo clippy`/`fmt` already checked on 1.93
and 1.97 per `COMPILER-STATE.md`'s C4 close-out).

Follow-up for C5.1b, not decided here: minimum supported rustc/Cargo version floor, override
mechanism for local testing, and the missing-toolchain diagnostic (§12.4) — these are toolchain
*discovery* concerns, distinct from the target-triple decision recorded above.

### C5.1a exit

- [x] §6–10 approved version confirmed (CD-042).
- [x] Exact C5 `MirTy` matrix enumerated against `starkc/src/mir/mod.rs` and `starkc/src/hir.rs`.
- [x] Non-`Copy` storage, move/Drop invariants, enum/`Option`/`Result`, function-pointer
      representation, layout-query rule — confirmed against approved §6–10, no changes.
- [x] Host target recorded: `aarch64-apple-darwin` (primary) + `x86_64-unknown-linux-gnu`
      (secondary), both required for C5.1 exit.

## C5.1b — Backend/runtime skeleton

**Not started.** Deliverables per `WP-C5-ENTRY.md` §14: backend module, version constants, minimal
runtime crate/module, generated crate skeleton, build-manifest schema, one generated empty `main`
program compiled natively (on both C5.1a-pinned targets, per this WP's stricter host-target
decision).

## C5.1c — Native Provider ABI specification

**Not started.** Deliverable: `STARKLANG/docs/compiler/native-provider-abi-v0.1.md` per §10.1, plus
a compile-time ABI validator and mock provider metadata fixture per §10.2. No provider feature
expansion.

## C5.1 exit

Not yet reached. Per `WP-C5-ENTRY.md` §14: CE4 decision recorded (done, CD-042), one verified
empty/scalar MIR program becomes a standalone executable on both pinned targets, runtime/backend/
compiler version checks demonstrated, no language semantics hidden in the runtime.

**C5.1a CLOSED 2026-07-21. Next: WP-C5.1b (backend/runtime skeleton).**
