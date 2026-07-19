# WP-C3.4 — Backend and Runtime Architecture Selection (Analysis & Recommendation)

Gate C3 selection work package. Prepared 2026-07-19. Consolidates the WP-C3.1 workload/framework,
the WP-C3.2 (generated-Rust) and WP-C3.3 (direct-Cranelift) spikes, and the WP-C3 breadth run into
a decision.

> **This document is a recommendation, not the decision.** Backend selection is escalation **CE5
> — an owner decision** (charter §2.3). This analysis presents the three-way comparison and a
> reasoned recommendation; the owner records the selected strategy. Allowed outcomes (roadmap
> WP-C3.4): `SELECT-GENERATED`, `SELECT-DIRECT`, `REVISE`, `BLOCKED`. An interpreter-only end
> state is not allowed (CD-004).

## 1. The three-way comparison

The semantic comparator is the reference interpreter (the oracle). The architecture candidates:

| | Reference interpreter | Generated Rust/C (A) | Direct Cranelift (B) |
|---|---|---|---|
| Role | semantic oracle (not a candidate) | candidate | candidate |
| Frozen-corpus match | 17/17 (by definition) | **8/17** | 3/17 |
| Semantic mismatches on supported | — | 0 | 0 |
| Trap→abort parity | — | ✅ | ✅ |
| Build dependency | none (interpreter) | **`rustc` (heavy, mandatory)** | Cranelift crates + `cc`; **no rustc** |
| Total build time (tiny workload) | n/a | ~87 ms/case | ~49 ms/case (not a general multiple) |
| Breadth cost (structs/generics/enums/String) | n/a | **low — rustc absorbs it** | high — a subsystem per family |
| Cross-platform | n/a | free (rustc targets) | self-owned per target |
| Debug-info / trap file:line | n/a | via rustc (source-like) | must thread SourceLoc/DWARF |
| ABI control | n/a | indirect (through generated Rust) | **direct** |
| MIR benefit | n/a | cleaner input | **large — MIR ≈ Cranelift's model** |
| Runtime dependency of `stark build` | interpreter only | **full Rust toolchain** | linker only |

## 2. What the evidence does and does not settle

**Settled by the spikes:**
- Both candidates produce standalone native executables that match the interpreter exactly on
  their supported subset, including the trap-abort contract. Native compilation of Core is
  **feasible** on both paths — the gate's core question.
- Generated-Rust reaches broad aggregate/generic correctness **cheaply** (8/17 via ~250 lines of
  text emission) because rustc owns monomorphization, layout, ABI, and Drop.
- The direct backend's breadth cost is **real but largely mandatory MIR work** — struct layout,
  monomorphization, drop elaboration, and discriminant handling are all Gate C4 deliverables. The
  HIR-level spike comparison therefore **overstates** the direct backend's long-run cost.

**Not settled (and not resolvable at C3, by sequencing):**
- Executable size, startup time, steady-state runtime — unmeasured for both (C7 work).
- The direct backend's coverage once it consumes **verified MIR** (Gate C4) — the fair comparison,
  which cannot exist before C4. Requiring it would be circular: C3 precedes C4.
- Whether the `rustc` build-dependency weight is acceptable for STARK's audience.

## 3. The decision hinges on one strategic question

Both candidates are correct and feasible. The choice is a priority judgment:

> **Does the project prioritize (a) the fastest, lowest-risk path to a working native MVP, or
> (b) a self-contained native compiler with no `rustc` build dependency as the end state?**

- **(a) → generated Rust.** rustc's battle-tested middle/back end gives broad correctness now,
  free cross-platform and debug-info, lowest semantic-parity risk. Cost: `stark build` requires a
  full Rust toolchain forever, and builds are slower.
- **(b) → direct Cranelift.** A `stark build` that needs only a linker, fast builds, and the ABI
  control the Native Provider ABI (C5.1) wants. Cost: STARK owns monomorphization, aggregate
  layout, drop elaboration, and a runtime surface — higher up-front engineering and
  semantic-parity risk — though most of that is mandatory MIR work regardless.

## 4. Recommendation: SELECT-GENERATED (initial production backend), direct kept open

**Recommended outcome: `SELECT-GENERATED`** — generated Rust as the initial production backend
behind verified MIR — **with a backend-neutral MIR contract that preserves `SELECT-DIRECT` as a
C7-gated migration.**

Rationale, weighted by the charter's own priority order (§1.6 rule 7, "correctness precedes
optimisation"):

1. It is the **shortest, lowest-risk path to correct, broad native compilation** — the mandatory
   completion goal (CD-004). The breadth run proved this concretely (8/17 cheaply, zero
   mismatches).
2. STARK is currently a **research language** (Gate 7: RETAIN AS RESEARCH LANGUAGE), for which a
   `rustc` build dependency is an acceptable trade for speed-to-correctness. This weighting would
   change if/when the systems-platform ambition (S-roadmap) becomes primary — hence the C7
   re-evaluation trigger below.
3. It reuses the **old Gate 5 generated-Rust precedent** (`deploy/`).
4. **No backend lock-in** (charter §1.6 rule 9): a backend-neutral MIR keeps the direct backend a
   live option, to be revisited at C7 on measured evidence — the same evidence-gated pattern the
   charter uses for LLVM (§1.6 rule 10).

**Why not `SELECT-DIRECT` as the initial path:** its advantages (no rustc dep, ABI control) are
real and strategically attractive, and the MIR shrinks its cost — but building a from-scratch
backend (monomorphization, layout, drop, runtime) before a single correct native release is the
higher-risk, slower path, which inverts "correctness precedes optimisation." It is the right
*eventual* backend if the self-contained-compiler goal becomes primary; that is a C7 decision on
measured evidence, not a C3 commitment.

**Why not `REVISE` or `BLOCKED`:** both candidates demonstrated credible, correct native paths —
not `BLOCKED`. The missing data (exe size/startup/runtime, MIR-level comparison) is inherent to
the sequencing (it needs C4–C7), not a bounded pre-C4 follow-up — so not `REVISE`.

### If `SELECT-GENERATED` is chosen, the architecture commits to (roadmap WP-C3.4 requirements):

- **MIR consumption boundary:** the generated-Rust emitter consumes **verified MIR** (Gate C4),
  not typed HIR — the spike lowered from HIR only because MIR does not exist yet.
- **Runtime ownership / ABI direction:** a small STARK runtime library (print/panic/trap glue);
  Rust owns value layout and calling convention in the MVP; the Native Provider ABI (C5.1) is
  expressed as `extern "C"` provider calls from generated Rust.
- **Target-platform plan:** inherit rustc's targets; ship Tier-1 (linux-x64, macos-arm64) first.
- **Debug / source mapping:** map STARK spans → generated-Rust lines → rustc debug info; trap
  file:line via a span table (WP-C5.5).
- **Unsupported MVP features + closure plan:** floats (port `canonical_float`), `?`, tuple
  patterns, traits/Drop, references, Vec/HashMap — all mechanical follow-ups tracked into C4.5/
  C5/C6; function values (CD-021 items 16–23) per the callable-ABI memo.
- **Why the rejected candidate is not the initial path:** see above; direct-Cranelift is retained
  as a C7-gated migration via the backend-neutral MIR.

## 5. Alternatives the owner may prefer (and their consequences)

- **`SELECT-DIRECT`** — if the self-contained-compiler goal (no rustc dependency, ABI control) is
  judged primary now rather than at C7. Accept higher up-front backend engineering; the MIR makes
  it far cheaper than the HIR-level spike suggests. Requires committing to CE4 runtime-ABI work
  earlier.
- **`REVISE`** — if the owner wants one bounded pre-selection measurement first (the two most
  defensible: an exact Cranelift struct head-to-head via a ~150–200-line sret implementation, or
  an executable-size/startup comparison on the shared supported cases). The gate stays open.

## 6. Owner decision required (CE5)

The owner records one of `SELECT-GENERATED` / `SELECT-DIRECT` / `REVISE` / `BLOCKED` in
`COMPILER-STATE.md`'s Native-backend-selection section. Until then, status remains `SPIKING` and
no MIR/backend architecture is committed. This document does not change that status.
