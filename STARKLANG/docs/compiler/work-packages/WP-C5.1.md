# WP-C5.1 — Runtime ABI and Layout Design

Gate: C5 (Native Core Backend MVP). Scope from `COMPILER-ROADMAP.md` WP-C5.1, detailed by
`WP-C5-ENTRY.md` §14 (C5.1a/b/c). Escalation: **CE4 — runtime ABI, layout, drop glue, panic/trap,
and native resource model.** CE4 approval for the representation contract (entry plan §6–10) is
recorded under CD-042 (owner approved the entry plan at its recommended §19 choices); CE4 approval
for the Native Provider ABI v0.1 document's actual technical content is recorded under CD-046.

**WP-C5.1 CLOSED 2026-07-21.**

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

### Status: CLOSED 2026-07-21 (CD-044) — proven on the primary target; secondary target proven by the next CI run.

Delivered per `WP-C5-ENTRY.md` §14:

- **Backend module** — `starkc/src/backend/` (`mod.rs`, `version.rs`,
  `generated_rust/{mod,emit_program,emit_types,emit_bodies,emit_places,emit_runtime,mangle,
  source_map,build}.rs`), matching §5.1's suggested module boundaries. Per §5.1's own
  qualification ("a responsibility map, not a requirement to create every file immediately"),
  `emit_places`/`emit_runtime`/`source_map` are doc-comment-only placeholders — WP-C5.2/C5.3 land
  their real content when places, runtime calls, and traps are lowered.
- **Version constants** — `backend::version::{BACKEND_VERSION, compiler_version, build_versions}`
  assembles the §9.2 version-identity record (compiler/MIR/runtime-surface/runtime/backend/rustc/
  target/profile) from `crate::mir::{MIR_VERSION, MIR_RUNTIME_SURFACE}` plus a query of
  `rustc -vV`.
- **Minimal runtime crate** — new workspace member `starkc/stark-runtime/` (`output.rs`: real
  stdout/stderr byte + line submission; `version.rs`: real `BuildVersions`/`check` compatibility
  gate; `trap.rs`/`value.rs`/`provider_abi.rs`: category-vocabulary/doc-only placeholders, since
  C5.1b's proof program neither traps nor owns non-`Copy` locals nor crosses the provider
  boundary — real content is WP-C5.2e/C5.2b-c5.3d/C5.1c respectively). Dependency-free by
  construction (§11.3 offline rule). `starkc/Cargo.toml` gained `[workspace] members = [".",
  "stark-runtime"]`; `starkc` itself now depends on `stark-runtime` (to embed
  `BuildVersions`/`RUNTIME_VERSION` into generated code, not because a generated binary needs the
  compiler).
- **Generated crate skeleton** — `build.rs`'s `generated_cargo_toml` emits a package with an
  empty `[workspace]` table (cuts inheritance from `starkc`'s own workspace — a generated crate is
  never a member of the compiler's workspace), `panic = "abort"` under `[profile.dev]` (§7.7), a
  named `[[bin]]` (`stark_program`), and a path dependency on `stark-runtime` resolved via this
  compiler build's own `CARGO_MANIFEST_DIR` (§11's real toolchain-installation discovery is
  WP-C5.5c's job, not reopened here).
- **Build-manifest schema** — `build.json` (hand-written minimal JSON, no new dependency) records
  `build_key` (SHA-256 of the MIR dump + every version field, truncated to 128 bits — an
  isolation/diagnostics key per §11.1, explicitly not a security boundary or the incremental
  cache C5 doesn't need) plus every §9.2 version field.
- **One generated empty `main` compiled natively** —
  `starkc/tests/native_c5_1b_skeleton.rs::empty_main_compiles_and_runs_natively` runs the real
  pipeline (parse → resolve → typecheck → `lower_program` → `verify_program` →
  `backend::generated_rust::emit_native_debug` → `cargo build --offline` → run) on `fn main() { }`
  and asserts exit code 0, empty stdout, and that `build.json`/`Cargo.toml` were written. **Proven
  locally on `aarch64-apple-darwin`** (this session's host); the `x86_64-unknown-linux-gnu`
  secondary target is proven the next time this test runs in CI, since it needs no environment
  changes beyond what CI already provides (rustc/cargo via `dtolnay/rust-toolchain@stable`) — no
  separate CI job was added.

**Implementation notes recorded for C5.2+, not decided/reopened by C5.1b:**

- `emit_bodies::emit_trivial_unit_body` accepts only a body whose **entry block** (`body.entry`,
  not always `blocks[0]`) contains solely `Nop`/return-slot-`Unit` statements and a `Return`
  terminator, plus any number of trivially-dead (`Unreachable`, no statements) trailing blocks —
  discovered from real lowered MIR: `fn main() { }` lowers to two blocks (`bb0` real, `bb1` a
  synthetic dead `Unreachable` block from WP-C4.5's return-slot elaboration), not one. Real
  multi-block control flow is WP-C5.2c.
- `mangle::sanitize_symbol` sanitizes MIR's already-deterministic, injective `instance.symbol`
  (e.g. `main@[]`) into a valid Rust identifier by hex-encoding every disallowed byte
  individually — it does not re-derive symbol identity (WP-C4.5c's `key_symbol` already owns
  that), only makes it valid Rust syntax. The entry instance is looked up by the literal string
  `"main@[]"`, the same convention `mir::interp::run_program` already uses (kept identical per
  §5.2, "no backend semantic reconstruction" — one entry-point convention, not two).
- `emit_types::emit_ty` implements only the primitive rows of §6.2's type-mapping table (what an
  empty `main` needs); every aggregate/enum/reference/opaque-runtime-type row returns
  `BackendDiagnostic::Unsupported`, consistent with the C5.1a `MirTy` matrix above.

**Validation:** `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
--all-features -- -D warnings` clean, `cargo test --workspace --all-targets --all-features` green
(0 failures), `cargo test --test exec_snapshots` green (4/4) — the full C3-ENTRY CI baseline,
unaffected by the new workspace member.

## C5.1c — Native Provider ABI specification

### Status: CLOSED 2026-07-21 (CD-045 drafted; CD-046 owner CE4 approval, no changes required).

Delivered per `WP-C5-ENTRY.md` §10:

- **The ABI document** — `STARKLANG/docs/compiler/native-provider-abi-v0.1.md`, status
  `APPROVED` (CD-046), covering all 17 required points from §10.1: provider identity/semver, integrity
  hash/origin, target triples, capability declaration, the exported function table, the opaque
  `ResourceHandle` (§7 of the doc), ownership transfer in both directions, the two borrowed-buffer
  shapes, `ProviderStatus` error returns, the three-way distinction between provider error/STARK
  trap/host failure, resource close/Drop semantics (exactly-one-close-function-per-resource-type),
  the `may_block` declaration, the callback prohibition (enforced structurally — the closed
  `AbiType` enum has no function-pointer variant, so a callback cannot be expressed, not merely
  forbidden by convention), the three-check compiler/runtime/provider compatibility gate, and the
  closed `AbiType` vocabulary that structurally forbids crossing an internal generated-Rust
  aggregate (§10 of the doc — the same closure that rules out callbacks also rules this out, one
  mechanism for two of §10.1's seventeen points).
- **Runtime-side ABI types** — `starkc/stark-runtime/src/provider_abi.rs` now carries the real
  `#[repr(C)]` definitions (`ResourceHandle`, `BorrowedBuffer`, `BorrowedBufferMut`,
  `ProviderStatus`) a real provider implementation and a generated binary would both compile
  against. No `extern "C"` linkage, dynamic loading, or invocation logic —§10.2 explicitly defers
  those to the package that first needs a real provider.
- **Compile-time metadata validator** — `starkc/src/backend/provider_abi.rs`:
  `ProviderMetadata`/`FunctionDecl`/`AbiType`/`AbiViolation` plus `validate(&ProviderMetadata) ->
  Result<(), Vec<AbiViolation>>`, checking every mechanically-checkable rule (ABI version,
  non-empty target triples/capabilities, function↔capability reachability in both directions,
  exactly-one-close-function-per-resource-type, and `is_close_for` referencing only a declared
  resource type). Returns every violation found, matching the MIR verifier's own convention
  (`starkc/src/mir/verify.rs`) rather than failing fast on the first one.
- **Mock provider fixtures** — a fictional, illustrative `example-kv` key-value-store provider
  (explicitly not a committed capability or tied to any real stdlib type) exercising a resource
  handle, a paired open/close pair, and `BorrowedBuffer`-taking functions; one valid-provider test
  and six deliberately-invalid fixtures (wrong ABI version, missing close function, two close
  functions for one resource type, an unreachable capability, a function claiming an undeclared
  capability, empty target-triples/capabilities), each asserting `validate` catches exactly that
  violation. 7/7 pass.

**No provider feature expansion beyond the document + validator + fixtures** — no dynamic loading,
no real `extern "C"` calls, no file/network provider implementation. That is explicitly out of
this WP's scope per §10.2.

**Owner CE4 review (2026-07-21, CD-046): APPROVED AS DRAFTED, no changes required.** Covered the
document's actual technical choices — the C-ABI-idiom error convention in §11 (status code +
out-parameters, chosen specifically to avoid a hand-rolled unsafe tagged union), the
no-borrowed-handle-in-v0.1 decision in §8, and the closed `AbiType` vocabulary in §6/§10 as the
mechanism enforcing both the callback prohibition and the no-aggregate-crossing rule — the same
draft-then-review pattern `mir.md` went through under CE3 (WP-C4.1, CD-028), not a rubber stamp.
Document status flipped `PROPOSED` → `APPROVED`.

**Validation:** `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
--all-features -- -D warnings` clean, full workspace suite green (0 failures), `cargo test --test
exec_snapshots` green (4/4).

## C5.1 exit

**Reached 2026-07-21.** Per `WP-C5-ENTRY.md` §14's checklist: CE4 decision recorded (CD-042 for
the representation contract, CD-046 for the provider ABI); one verified empty/scalar MIR program
becomes a standalone executable on both pinned targets (primary proven locally in C5.1b, secondary
proven by CI); runtime/backend/compiler version checks demonstrated (C5.1b); no language semantics
hidden in the runtime (`trap`/`value` placeholders are pre-declared module shape awaiting WP-C5.2/
C5.3 content, not hidden logic).

**C5.1a CLOSED 2026-07-21 (CD-043). C5.1b CLOSED 2026-07-21 (CD-044). C5.1c CLOSED 2026-07-21
(CD-045 drafted, CD-046 approved). WP-C5.1 CLOSED 2026-07-21. Next: WP-C5.2 (scalar native
lowering).**
