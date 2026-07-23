# C6 Integration Ledger

**Status:** OPEN (C6.0 established the barrier)
**Integration base SHA:** `db73afe` (CD-077 — Gate C5 CLOSED)
**Branch:** `main` (single-branch model per owner directive; §7C worktrees waived)
**Integration lead:** Track A (Claude)

This ledger is the running record of every track handoff, shared-file lease, contract change, merge,
and integration result across C6. Tracks append; the integration lead maintains merge order and
records integration results. Only integration-branch (here: `main`) evidence counts toward WP
closure (WP-C6-ENTRY §7K).

---

## 1. C5 closure packet (pinned)

| Item | Value |
|---|---|
| C5 closure commit | `db73afe` (CD-077); qualification head `19254086` (CD-076) |
| C5 exit report | `starkc/docs/compiler/C5-exit-report.md` — `NATIVE-CORE-MVP-WITH-LISTED-DEVIATIONS` |
| Supported/unsupported matrix | C5 exit report §2 (supported) / §3 (deferred) |
| Frozen three-package workspace | `starkc/tests/fixtures/c5-native-workspace/` (13 symbols, `EXPECTED-SYMBOLS.txt`) |
| Three-engine snapshot subset | `exec_snapshots/c5_native__01/02` (corpus v1.4.0) |
| Native Drop fixture | `native_c5_3_aggregates_enums.rs` + three-engine Drop cases |
| MIR version / runtime surface | `0.1` / `0.1-A8` |
| Runtime version / backend version | `0.1` / `0.1` |
| Target-layout identity | `stark-64-v1` (rev 1) |
| Toolchain (qualified) | rustc `1.93.0`, cargo `1.93.0`, host `aarch64-apple-darwin` |
| Open deviation list | C5 exit report §5 (DEV-098 defensive; DEV-101 follow-ups) |
| C6 deferral list | C5 exit report §3 |

**WP-C6-ENTRY §2 recheck items (re-pinned at `db73afe`), assigned to tracks:** multi-unit
enum-payload partial moves → **C6.1c (A)**; wider non-`Copy` cross-block moves → **C6.1b (A)**;
non-`Copy` by-value fixed-array iteration → **C6.1d (A)**; the **generic-impl receiver-inference
limitation** (recheck if still open) → **C6.2b (B)**; the WP-C2.12 deterministic generated corpus
and full cross-backend replay → **C6.5 (B/integration)**. (The first three appear in exit report §3;
the receiver-inference recheck and the WP-C2.12 corpus half are pinned here explicitly since they
are recheck/carry-forward items rather than C5 refusals.)

Handoff is read from these artifacts, not reconstructed from commit messages.

---

## 2. Baseline validation (integration base `db73afe`)

At the integration base (code identical to the C5 qualification head; CD-077 is docs-only):

```text
cargo test --workspace --all-targets --no-fail-fast → 59 binaries, 1098 passed, 0 failed
cargo fmt --all -- --check                           → clean
cargo clippy --workspace --all-targets --all-features -- -D warnings → clean
```

Full suite passes at the integration base. ✅

---

## 3. Contract freeze

`C6-SHARED-CONTRACTS.md` v1 FROZEN at `db73afe`. `C6-FILE-OWNERSHIP.md` ACTIVE. All three tracks
begin from `db73afe`.

---

## 4. Track ledger

| # | Track | Agent | Base SHA | Owned/leased files | Contracts assumed | Proposed contract changes | Tests | New deviations | Merge order | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| C6.1a | A | Claude | `db73afe` | `C6-OWNERSHIP-MATRIX.md` (new) | SHARED-CONTRACTS v1 | none | probe-grounded classification | **G3**, **G4** | §7J:3 | **CANDIDATE-COMPLETE** |
| C6.1b | A | Claude | `db73afe` | `emit_projections.rs`, `emit_places.rs`, `slot.rs` (owned); `emit_bodies.rs` (lease) | SHARED-CONTRACTS v1 | none (`ValueSlot::reinit` is a NEW runtime method, additive, no ABI/behaviour change to existing ops) | `native_c6_1_ownership.rs` (5) | G3, G4 fixed | §7J:3 | CANDIDATE-COMPLETE (CD-081) |
| C6.1c | A | Claude | (post-C6.1b) | `emit_bodies.rs`, `emit_projections.rs` (owned/lease); `mir/lower.rs` (lease) | SHARED-CONTRACTS v1 | none (MIR canonicalisation with existing ops only — refined Option A) | `native_c6_1_ownership.rs` (+7 `c61c_*`), `native_c5_3` positive multi-unit | G1 fixed | §7J:3 (front-end lowering + backend) | CANDIDATE-COMPLETE (CD-082) |
| C6.1d | A | Claude | (post-C6.1c) | `mir/lower.rs` (lease), `borrowck.rs` (lease) | SHARED-CONTRACTS v1 | none (unroll with existing `ConstIndex`; no new MIR op — Option (a)) | `native_c6_1_ownership.rs` (+12 `c61d_*`), `gate2_valid` accept flip | G2 fixed; DEV-090 closed | §7J:3 | CANDIDATE-COMPLETE (CD-083) |
| C6.1e | A | Claude | (post-C6.1d) | `C6-DROP-PATH-MATRIX.md` (new); `three_engine_differential.rs` (lease, tests only) | SHARED-CONTRACTS v1 | none (evidence only — no source change) | `three_engine_differential.rs` (+12 `c61e_*`) | none | §7J:5 (evidence) | **CANDIDATE-COMPLETE** |
| — | B | (Gemini) | `db73afe` | (see ownership doc) | SHARED-CONTRACTS v1 | none | — | — | §7J:2 | not started |
| — | C | (Codex) | `db73afe` | (see ownership doc) | SHARED-CONTRACTS v1 | none | — | — | §7J:4 | not started |

Merge order at each barrier (WP-C6-ENTRY §7J): (1) mechanical/shared-contract commits → (2) Track B
front-end/instance commits → (3) Track A MIR ownership/storage commits → (4) Track C runtime/emitter
commits → (5) Track B evidence/corpus commits → (6) integration fixes → (7) full validation → (8)
state/doc update.

---

## 5. Lease log

_(none yet — appended when a shared file is leased)_

| file | track | reason | base SHA | API impact | tests | lease start | lease release |
|---|---|---|---|---|---|---|---|
| `emit_bodies.rs` | A | C6.1b G4: `emit_assignment` chooses `reinit` vs `write` by drop-obligation | `db73afe` | none (internal codegen choice) | `native_c6_1_ownership.rs` | C6.1b | C6.1b (landed) |
| `mir/lower.rs` | A | C6.1c G1: `materialize_consumed_variant_payload` — tuple decomposition of multi-field non-Copy variant payloads (owner ruling). Track B informed & excluded from this file until release. | post-C6.1b | narrow: `consume_field`/`bind_field_local` take a pre-built `field_place`; new private helper | `native_c6_1_ownership.rs` `c61c_*` | C6.1c | C6.1c (landed) |
| `emit_bodies.rs` | A | C6.1c G1: `try_variant_payload_extraction` destructuring emitter | post-C6.1b | none (statement-local codegen) | as above | C6.1c | C6.1c (landed) |
| `mir/lower.rs` | A | C6.1d G2: `lower_for_over_array_unrolled` — unroll non-Copy array iteration | post-C6.1c | narrow: new private helper; Copy path unchanged | `native_c6_1_ownership.rs` `c61d_*` | C6.1d | C6.1d (landed) |
| `borrowck.rs` | A | C6.1d G2: remove the DEV-090 E0104 rejection now that MIR lowers non-Copy array iteration | post-C6.1c | none (removes a rejection) | `gate2_valid` accept test | C6.1d | C6.1d (landed) |
| `three_engine_differential.rs` | A | C6.1e: add 12 drop-path probe cases (tests only; reuses the existing trap comparator and sits with the sibling C5.3d-1c drop cases) | post-C6.1d | none (no comparator/normalisation change) | the cases themselves | C6.1e | C6.1e (landed) |

---

## 6. Integration barriers

| Barrier | Required | Status |
|---|---|---|
| I1 (pre-Wave 2) | Track A storage interface frozen; Track B dispatch/instance interface frozen; Track C runtime-call inventory frozen; no open CE3/CE4; focused green; merged full suite green; contracts doc updated | pending |
| I2 (pre-Wave 3) | C6.1/C6.2 candidates complete; storage/dispatch interactions merged; three-engine focused green; no wrong-output divergence; full suite green | pending |
| I3 (pre-gate-exit) | C6.1/2/3 closed; Tier-1 jobs available; full corpus assembled; full suite green; no semantic quarantine; contracts versioned | pending |

---

## 7. Decisions and CE escalations

- **CD-079 [2026-07-23] — WP-C6-ENTRY APPROVED.** Owner approved the Gate C6 entry plan
  (`starkc/docs/WP-C6-ENTRY.md`), discharging §1's last opening condition and the Wave-0
  "approve C6 entry" step. All ten §1 opening conditions are satisfied at Gate C5 closure
  (`db73afe`/CD-077). The §7C branch/worktree model is waived; all C6 tracks execute on `main`.
  Wave 1 is authorised.

- **G3 [2026-07-23, C6.1a, Track A] — multi-level partial move/drop refused.** Probing for the
  ownership matrix found that a partial move/drop through a projection chain of depth ≥2 (`o.a.x`,
  `Place.projection` length 2) is `BACKEND-REFUSED` — only one projection level is implemented
  (WP-C5.3d-0). Not named in WP-C6-ENTRY §2's re-pin (which listed multi-unit enum moves, wider
  cross-block moves, non-`Copy` array iteration). Assigned to **C6.1b**; its acceptance set now
  includes multi-level projected moves. Deterministic refusal, not a miscompile.

- **G4 [2026-07-23, C6.1b, Track A] — loop-carried no-`Drop` reassignment aborted at run time.** A
  no-`Drop` non-`Copy` local's slot is never reset by a MIR `Drop` (the verifier emits none for a
  non-droppable type), so a loop back-edge reassignment tripped `ValueSlot::write`'s dead-slot check
  and **aborted** (compile-then-abort — the CD-070 severity class). Newly surfaced by C6.1b native
  execution (the C6.1a probe checked only `emit` success). **Fixed**: additive `ValueSlot::reinit`
  (overwrite regardless of state, no drop — sound only for a no-drop type) + `emit_assignment`
  emitting it for a no-drop slot local. Not a CE4: `reinit` is a new helper method (Track A owned,
  WP-C6-ENTRY §10 "preserve `ValueSlot<T>`… route moves/writes through reviewed helpers"), additive,
  no change to any existing op, ABI, layout, or Drop glue. Recorded in `C6-SHARED-CONTRACTS.md §4`.

- **Owner ruling [2026-07-23, C6.1d] — Option (a) unconditional array-iteration unrolling, NOT a
  CE3.** By-value iteration over a non-`Copy` fixed array unrolls into `N` `ConstIndex(i)` moves (no
  size cap, no CE4 runtime bitmap; the `Copy` dynamic-index path stays). Uses only existing MIR ops,
  so not a CE3. Move the iterable once into a per-element-drop-tracked owner; fresh MIR binding local
  per iteration; scope cleanup body → binding → remaining elements; no cleanup after trap. **DEV-090
  CLOSED** (the front-end E0104 rejection removed — the HIR oracle moves each element, so no engine
  divergence). Pathological array length is a future compiler-resource-limit concern, not an
  `Unsupported` feature gap.

- **Owner ruling [2026-07-23, C6.1c] — refined Option A (enum payload partial moves), NOT a CE3.**
  Lower each consuming active-variant payload (multi-field, non-`Copy`) into ONE canonical
  `Rvalue::Aggregate(AggKind::Tuple, [VariantField(v,0..n)])` MIR statement, then use ordinary
  tuple-field movement (C6.1b). The backend implements a NARROW statement-local destructuring
  emitter for that canonical aggregate; **cross-block backend analysis is prohibited** (that was the
  rejected Option B). This uses only existing MIR operations — no new projection/rvalue/verifier
  rule/enum-identity change — so it is **not a CE3**; changing MIR dumps is not itself a CE3. Track A
  leased `mir/lower.rs`; Track B informed and excluded until release. Stop for CE3/CE4 if any of
  `Projection::VariantPayload`, `Rvalue::TakeVariantPayload`, a new verifier atomicity rule, an enum
  payload identity change in `TypeContext`, or altered MIR move semantics becomes necessary — none
  did. Recorded under this commit's decision ID.

_(CE3/CE4/CE8/CE9 recorded here before implementation continues — none yet.)_

---

## 8. C6.0 closure

- [x] Gate C5 closure packet pinned (§1)
- [x] Shared contracts written (`C6-SHARED-CONTRACTS.md`)
- [x] Track file ownership written (`C6-FILE-OWNERSHIP.md`)
- [x] Shared-file lease mechanism written (`C6-FILE-OWNERSHIP.md` §3)
- [x] Integration base SHA recorded (`db73afe`)
- [x] Mechanical extraction: none required at C6.0 (deferred just-in-time; §4 of ownership doc)
- [x] Full suite passes at the integration base (§2)
- [~] All three agents branch from the same SHA — **waived** (single-branch `main` model); all
      tracks instead begin from `db73afe` on `main`
- [x] No C6 semantic implementation has begun before this barrier

C6.0 complete: three agents can implement separate semantic areas without editing the same
authority-bearing code or making conflicting assumptions. Wave 1 may open (Track A ownership matrix,
Track B generic/trait matrix + corpus scaffolding, Track C runtime inventory + String/str).
