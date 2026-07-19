# WP-C3.3 — Direct (Cranelift) Backend Spike Report

Gate C3 spike. Prepared 2026-07-19. Records the direct-backend candidate (Candidate B in
`STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md`) measured against the same frozen
workload as WP-C3.2, for a like-for-like comparison. **Disposable spike (charter §2.2), not a
production backend.** Artifact: `starkc/tests/spike_cranelift.rs` — an isolated integration test,
not wired into `stark build`. It does not select a backend — that is WP-C3.4 (CE5).

## What was built and run

An isolated HIR→Cranelift-IR lowerer that consumes programs already validated by the real front
end (no front-end check bypassed), lowers a supported subset to Cranelift IR, emits a **native
object file** via `cranelift-object` (no JIT — so no `unsafe`, which the crate forbids), links it
with the host `cc` against a ~6-line C runtime into a **standalone executable**, runs it, and
compares stdout + exit status against the reference interpreter over the frozen `exec_snapshots`
corpus (`corpus_version = 1.0.0`). Test: `cranelift_spike_matches_interpreter_on_frozen_corpus`
(skips MANUAL if `cc` absent).

Supported subset: signed integer primitives (Int8/16/32/64) + Bool; arithmetic with STARK trap
semantics — add/sub/mul overflow via explicit checks + `trapnz`, div/rem-by-zero and `INT_MIN/-1`
via Cranelift's trapping `sdiv`/`srem`; unary neg / bool not; comparisons; `let`/`let mut` +
assignment; `if`/`else`, `while`, `loop`+bare `break`, `continue`, `for x in a..b`; block-tail
values; non-generic functions + calls; `print`/`println` of an integer or bool.

## Result summary

- **Supported and matched the interpreter exactly: 3 / 17 frozen corpus cases.** Zero semantic
  mismatches.
- **Trap-abort parity demonstrated**: `primitive__02` (`Int8` `120 + 100`) — the object emits a
  `trapnz` on the computed overflow condition; the linked binary aborts with a non-zero exit,
  matching the interpreter's trap.
- **14 / 17 cleanly reported unsupported**, each with a reason.
- **Cranelift codegen: ~2 ms/case; `cc` link: ~47 ms/case** (mean over supported cases).
  **Read these carefully — they are not a "40× faster" claim** (see "Timing caveat" below).

| Case (frozen corpus) | Result | Note |
|---|---|---|
| `expr_stmt__01_arithmetic_and_precedence` | ✅ match | item 1 (arithmetic/precedence) |
| `expr_stmt__03_loops_break_continue` | ✅ match | item 2 (loops, `for`/range, break/continue) |
| `primitive__02_integer_overflow_traps` | ✅ match/trap | item 8 (trap → abort) |
| `primitive__01_integer_widths_and_overflow_traps` | — unsupported | `UInt8` (spike is signed-only) |
| `expr_stmt__02`, `expr_stmt__04` | — unsupported | `String` / `match` |
| `primitive__03` | — unsupported | `Float64` |
| `struct_enum_trait__01..04`, `ownership_drop__01/02` | — unsupported | struct/enum/trait/generic/refs |
| `option_result__01/02`, `collection_iter__01/02` | — unsupported | `Option`/`Result`/`Vec`/`HashMap` |

## Measurement dimensions (same record as WP-C3.2)

**Unsupported constructs (recorded).** Same families as the generated-Rust spike (String/Vec/
HashMap, floats, structs, enums incl. Option/Result, traits, generics, references, match, `?`),
**plus unsigned integers** — this direct spike implements signed overflow detection only, so
`UInt8`..`UInt64` are unsupported here where the generated-Rust spike handled them (its
`StarkChecked` trait covered `u8`..`u64`). That is why the direct spike matched 3 cases where
generated-Rust matched 4 (`primitive__01` uses unsigned widths). A trivial extension; recorded
honestly, not hidden.

**Source-to-generated-code traceability.** Lower — this is the direct backend's structural
disadvantage. There is no human-readable intermediate: STARK source lowers straight to Cranelift
IR (blocks, SSA values, `trapnz`). A source→machine mapping would require deliberately threading
Cranelift's source-location / debug-info facilities (`ir::SourceLoc`, DWARF via the object
backend) — real work, not the free item/line correspondence the generated-Rust text gives. This
is the main open cost for the WP-C5.5 trap-file:line requirement under this candidate.

**Build-tool dependencies.** The host `cc`/linker, plus the Cranelift crates as a compiler-side
dependency (`cranelift-codegen`/`-frontend`/`-module`/`-object`). **No `rustc` at build time.**
Notable finding: **Cranelift 0.133 (latest) requires rustc ≥ 1.94, but this environment's
toolchain is 1.93**, so the spike had to pin **Cranelift 0.110**. Cranelift's MSRV advances
quickly relative to a pinned project toolchain — a real, recurring maintenance cost for this
candidate (dimension 9), to weigh against generated-Rust's single heavyweight `rustc` dependency.

**Cross-platform behaviour.** Cranelift targets the host ISA (here aarch64-apple-darwin) via
`target_lexicon::Triple::host()`; other targets require per-target ISA configuration and a
per-target linker. Not exercised beyond the host. Generated-Rust inherits rustc's broad target
matrix for free; the direct backend owns cross-compilation itself.

**Semantic mismatches.** Zero on supported cases. The trap-abort contract holds through native
code: explicit overflow checks + `trapnz`, and Cranelift's trapping `sdiv`/`srem`, both abort the
process with a non-zero exit, matching the interpreter's trap.

**Amount of glue per language feature.** **Higher than generated Rust**, materially. The direct
backend requires, by hand: block creation and sealing (SSA), `Variable`-based mutable locals,
explicit control-flow graphs for `if`/`while`/`loop`/`for` with continue/break target stacks,
explicit overflow-detection sequences (`bxor`/`band`/`icmp`/`trapnz`; double-width `sextend`/
`imul`/`ireduce` for mul), sign-extension before the print runtime call, a C runtime, and a link
step. One representative subtlety cost real debugging time (a Unit-typed control-flow expression
in block-tail position had to be re-routed from value-lowering to statement-lowering). The
generated-Rust spike offloaded all of monomorphization, Drop, layout, and overflow-idiom choice
to rustc; here every one of those is the backend's job.

**Feasibility of consuming verified MIR.** *Higher value here than for generated Rust.* This
spike lowered typed HIR directly and had to reconstruct control-flow structure and re-query
`expr_types` while emitting blocks. A verified MIR (Gate C4) — already a basic-block CFG with
explicit terminators, typed locals, explicit move/drop, and explicit trap points — maps almost
1:1 onto Cranelift's block/terminator/`trapnz` model, and would remove most of the structural
glue this spike wrote by hand. The direct backend is the candidate that benefits *most* from the
mandatory MIR.

## Head-to-head: WP-C3.2 (generated Rust) vs WP-C3.3 (direct Cranelift)

| Dimension | Generated Rust (A) | Direct Cranelift (B) |
|---|---|---|
| Frozen-corpus cases matched | 4/17 | 3/17 (unsigned unsupported in spike) |
| Semantic mismatches on supported | 0 | 0 |
| Trap-abort parity | ✅ | ✅ |
| Codegen time | not separable (within rustc) | ~2 ms/case (codegen phase only, from built IR) |
| Total build time/case | ~87 ms (`rustc`, whole pipeline + link) | ~49 ms (2 ms codegen + 47 ms `cc` link) |
| Build dependency | `rustc` (heavy, single) | Cranelift crates + `cc`; **no rustc**; MSRV churn (0.133→1.94) |
| Traceability / debug-info | high (source-like text) | low (must thread SourceLoc/DWARF) |
| Glue per feature | low (rustc absorbs monomorph/Drop/layout) | **high** (we own CFG/SSA/overflow/link) |
| Cross-platform | free (rustc targets) | per-target work we own |
| MIR consumption benefit | cleaner input | **large** — MIR ≈ Cranelift's own model |
| Control of ABI/layout | indirect (via generated Rust) | direct |

Not measured for either spike (deferred to a breadth run / later gate): executable size, startup
time, steady-state runtime, and the whole unsupported breadth (aggregates, generics, traits,
references, Drop, function values).

### Timing caveat (do not quote a general multiple)

The `~2 ms` Cranelift codegen figure and the `~87 ms` rustc figure are **not** a like-for-like
comparison, and the difference must not be reported as "Cranelift is ~40× faster":

- The `2 ms` is Cranelift's **codegen phase only**, operating on an IR the STARK front end already
  produced — no parsing, no type/borrow checking, and **no linking**.
- The `87 ms` is rustc's **entire pipeline** on the generated `.rs`: re-parsing, re-type-checking,
  re-borrow-checking, monomorphization, LLVM codegen, **and** linking, in one number.
- The only defensible end-to-end comparison from this spike is **~49 ms (Cranelift + `cc` link)
  vs ~87 ms (rustc all-in) ≈ 1.8×**, and even that is over **three trivial programs, debug/
  unoptimised, single-file, no caching, one host** — the charter explicitly forbids claiming a
  general performance multiple from a small workload.
- On the direct path, the **`cc` link (~47 ms) dominates** the wall clock, not codegen.

What the data *does* support, directionally only: Cranelift's codegen phase is structurally much
lighter than rustc's front-end-plus-LLVM pipeline (its design purpose — fast iterative builds).
The magnitude is not measurable here and must be re-measured on real workloads at C7.

## Reading of the evidence (not a decision)

The two candidates now have symmetric, honest data on the same frozen subset. The tradeoff is
clear and matches the hypothesis in `NATIVE-CORE-ARCHITECTURE.md` §4:

- **Generated Rust** buys correctness cheaply — rustc absorbs monomorphization, Drop, layout, and
  overflow idioms, giving low glue and free cross-platform/debug-info — at the cost of a heavy
  mandatory `rustc` build dependency and slower builds.
- **Direct Cranelift** buys faster builds (defensibly ~1.8× end-to-end on this tiny workload —
  **not** the raw codegen ratio; see the timing caveat), no rustc dependency, and direct ABI
  control — and is the bigger beneficiary of the mandatory MIR — at the cost of substantially more
  backend engineering (CFG/SSA/overflow/Drop/layout all owned by us), weaker out-of-the-box
  debug-info, self-owned cross-compilation, and a faster-moving dependency MSRV.

Neither candidate was falsified; neither was cleared. The decisive open questions are the ones
neither spike resolved: the unsupported breadth (esp. generics/traits/Drop/references, where the
glue asymmetry will be largest), executable size/startup/runtime, and — for Cranelift — the
debug-info/trap-mapping cost. **Backend selection is WP-C3.4 / CE5 (owner).** This report is
evidence for that decision, not the decision.

## Dependency note (charter §1.10)

The Cranelift crates are declared **dev-dependencies only** in `starkc/Cargo.toml` (with an
inline necessity/scope/licence note) — they are **not** part of the shipped compiler's dependency
surface. Licence: Apache-2.0 with LLVM exception (Bytecode Alliance). Pinned to 0.110 for rustc
1.93 compatibility. These lines and `tests/spike_cranelift.rs` are removed when Gate C3 selects a
backend (WP-C3.4); the spike is not production architecture (charter §2.2).

## Breadth run (2026-07-19)

Coverage remains **3/17**. Extending the direct backend to the aggregate/generic breadth that the
generated-Rust spike reached (structs, generics, `Option`/`match`, `String`) requires, per
construct family, a dedicated subsystem — struct-by-value needs stack-slot layout + field offsets
+ load/store + an **sret ABI transform**; enums need **tagged-union layout** + discriminant
switching; generics need a **monomorphization engine**; String/Vec need a **runtime library**.
Rather than build a second real backend inside the C3 spike, the cost was measured concretely from
the ~600-line integer/control-flow Cranelift lowerer already built. The full head-to-head is in
`starkc/docs/compiler/spikes/WP-C3-breadth-comparison.md`. Key point for WP-C3.4: **most of this
breadth cost is mandatory MIR work anyway** (Gate C4 supplies monomorphization-ready,
drop-elaborated, layout-bearing MIR), so the HIR-level comparison overstates the direct backend's
long-run cost.

## Reproduce

```bash
cd starkc
cargo test --test spike_cranelift -- --nocapture   # prints the frozen-corpus coverage + timings
```

Skips (MANUAL) if `cc` is unavailable.
