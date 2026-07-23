# C6 File Ownership and Shared-File Lease Protocol

**Status:** ACTIVE (C6.0)
**Integration base SHA:** `db73afe` (Gate C5 CLOSED)
**Branch model:** all C6 work proceeds on `main` (owner directive, 2026-07-23) — the WP-C6-ENTRY
§7C branch/worktree model is **waived**. The lease protocol below therefore governs concurrent edits
directly on `main` rather than across track branches.

Three tracks implement C6 concurrently (WP-C6-ENTRY §7A):

| Track | Default agent | Primary WP | Domain |
|---|---|---|---|
| **A** | Claude | C6.1 | ownership, partial moves, Drop; **semantic integration lead** |
| **B** | Gemini | C6.2 | generics, traits, method resolution, differential corpus, adversarial review |
| **C** | Codex | C6.3 (+ C6.4 mechanics) | runtime values, collections, resources, platform mechanics |

Role assignments, not capability claims; the owner may swap agents while preserving track
boundaries.

---

## 1. Owned files (single-writer, no lease needed)

A track edits its owned files freely. Files marked `# when extracted` are created just-in-time by
the integration lead via mechanical seam extraction (§4) — not pre-created empty.

### Track A — ownership / Drop
```text
starkc/src/mir/drop_plan.rs
starkc/stark-runtime/src/slot.rs
starkc/src/backend/generated_rust/emit_places.rs
starkc/src/backend/generated_rust/emit_projections.rs
starkc/src/backend/generated_rust/emit_ownership.rs        # when extracted
starkc/tests/native_c6_1_ownership.rs
STARKLANG/docs/compiler/work-packages/C6-OWNERSHIP-MATRIX.md
```

### Track B — generics / traits / evidence
```text
starkc/src/resolve.rs
starkc/src/typecheck.rs
starkc/src/mir/lower/instances.rs                          # when extracted
starkc/src/backend/generated_rust/emit_dispatch.rs         # when extracted
starkc/tests/native_c6_2_generics_traits.rs
starkc/tests/c6_generated_corpus.rs
starkc/tests/c6-corpus/
STARKLANG/docs/compiler/work-packages/C6-GENERICS-TRAITS-MATRIX.md
```

### Track C — runtime / collections / platform
```text
starkc/stark-runtime/src/string.rs
starkc/stark-runtime/src/vec.rs
starkc/stark-runtime/src/slice.rs
starkc/stark-runtime/src/boxed.rs
starkc/stark-runtime/src/iter.rs
starkc/stark-runtime/src/collections.rs
starkc/stark-runtime/src/format.rs
starkc/stark-runtime/src/file.rs
starkc/src/backend/generated_rust/emit_runtime.rs
starkc/src/backend/generated_rust/emit_core_values.rs      # when extracted
starkc/tests/native_c6_3_runtime.rs
starkc/tests/native_c6_4_platform.rs
STARKLANG/docs/compiler/work-packages/C6-RUNTIME-MATRIX.md
STARKLANG/docs/compiler/work-packages/C6-PLATFORM-MATRIX.md
```

Track A/B/C **must not** edit each other's owned files, nor the "must not edit" areas in
WP-C6-ENTRY §7D/§7E/§7F (e.g. Track A never edits method/trait selection in `typecheck.rs`/
`resolve.rs`; Track B never edits `ValueSlot`/`DropPlan`; Track C never edits ownership-liveness,
method/trait selection, or the comparator's semantic normalisation).

---

## 2. Shared files (require a lease — one writer between integration barriers)

These authority-bearing files are touched by more than one track:

```text
starkc/src/mir/lower.rs
starkc/src/mir/mod.rs
starkc/src/mir/verify.rs
starkc/src/backend/generated_rust/emit_bodies.rs
starkc/src/backend/generated_rust/emit_types.rs
starkc/src/backend/generated_rust/mod.rs
starkc/src/backend/generated_rust/linkage.rs
starkc/src/backend/generated_rust/mangle.rs
starkc/stark-runtime/src/lib.rs
starkc/stark-runtime/src/value.rs
starkc/stark-runtime/src/provider_abi.rs
starkc/src/backend/generated_rust/build.rs
starkc/src/native_build.rs
starkc/src/native_toolchain.rs
starkc/tests/three_engine_differential.rs
starkc/tests/exec_snapshots.rs
Cargo.lock
```

---

## 3. Lease protocol

On `main`, a lease is a coordination record + a narrow, promptly-landed commit — it replaces the
cross-branch lease of §7C.

Lease record (logged in `C6-INTEGRATION-LEDGER.md`):
```text
file · owning track · reason · base SHA · expected API impact · tests required · lease start · lease release
```

Rules:
1. The **integration lead (Track A / Claude)** grants leases.
2. The lease holder rebases to the latest `main` before editing.
3. No other track edits the file until the lease commit lands on `main`.
4. A lease does **not** authorise a CE3/CE4 change.
5. The lease commit must be **narrow** (the shared-file change only).
6. After it lands, dependent tracks pull `main`.
7. Emergency parallel edits to a leased file are prohibited — reconstruct the intended combined
   contract and add a regression instead of mechanically choosing one side.

---

## 4. Mechanical seam extraction (integration-lead, on demand)

Preferred alternative to a long-held lease: extract a track-owned module from a monolithic shared
file, expose a narrow interface, and keep the central dispatcher integration-owned. Candidates
(create **only when a track actually needs it**, never empty to match the list):

```text
starkc/src/mir/lower/ownership.rs        (from lower.rs — Track A)
starkc/src/mir/lower/instances.rs        (from lower.rs — Track B)
starkc/src/mir/lower/runtime.rs          (from lower.rs — Track C)
starkc/src/backend/generated_rust/emit_ownership.rs   (from emit_bodies.rs — Track A)
starkc/src/backend/generated_rust/emit_dispatch.rs    (from emit_bodies.rs — Track B)
starkc/src/backend/generated_rust/emit_core_values.rs (from emit_types/emit_bodies — Track C)
```

Extraction requirements: no accepted/rejected program change; no MIR shape change; no runtime ABI
change; source movement only; focused tests unchanged; **full suite green**; landed as **one
integration commit** before the dependent track's parallel work.

**C6.0 decision:** no extraction is performed now. `emit_bodies.rs` and `lower.rs` are the most
contended, but extracting them speculatively moves risk with no current consumer. Extraction is
deferred to just-in-time at the point a track's Wave-1/Wave-2 work would otherwise need a long lease.
