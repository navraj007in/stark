# Native Core Architecture — Hypothesis, Workload Freeze, and Measurement Framework

WP-C3.1 deliverable (Gate C3, Native Compiler Architecture and Backend Selection Spike).
Prepared 2026-07-19. Non-normative proposal: it frames the backend-selection experiment,
freezes the workload the spikes run against, and defines how they are measured. It does **not**
select a backend (WP-C3.4, escalation CE5), define the MIR contract (Gate C4, CE3), or fix the
runtime ABI (WP-C5.1, CE4).

## 1. Purpose and the four separated questions

Gate C3 answers exactly one question: **which native backend architecture should implement
STARK's mandatory MIR-to-native Core compilation?** It does not answer *whether* STARK compiles
natively — that is settled: mandatory, per CD-004 / `COMPILER-CHARTER.md` §1.2. An
interpreter-only end state is not an allowed C3 outcome.

Keep separate (charter §1.4):

1. **Core correctness** — closed at Gate C2 (semantic freeze), the interpreter is the oracle.
2. **Native compiler architecture** — *this gate*.
3. Artifact-binding generality — ecosystem, not here.
4. AI-development methodology — not here.

The interpreter is the **semantic comparator** (native output must match it, charter §1.6 rule
6). The **architecture comparator** is not "no native compiler" — it is the strongest practical
candidate implementation paths, evaluated head to head.

## 2. Pipeline context

The mandatory end-to-end path (charter §1.2):

```text
STARK source
  -> parse + resolve                        (done)
  -> type / trait / ownership / borrow check (done, Gate C1)
  -> typed HIR                              (done)
  -> verified MIR                           (Gate C4, not yet built)
  -> selected native backend                (Gate C5, this gate selects which)
  -> standalone executable
```

Today the pipeline stops at typed HIR + reference interpreter. C3 chooses the backend
*architecture* that will consume verified MIR; C4 builds the MIR; C5 builds the backend. The
backend spikes in C3.2/C3.3 may lower from typed HIR directly (MIR does not exist yet) but must
record whether they could instead consume a verified MIR cleanly, since that is the mandatory
production shape.

## 3. What makes STARK unusually favorable to lower

Several frozen Core v1 decisions materially reduce native-backend complexity, and the hypothesis
below leans on them:

- **Traps and panic abort without unwinding** (`CLAUDE.md`; `CORE-V1-ABSTRACT-MACHINE.md`).
  Integer overflow, division by zero, out-of-bounds indexing, and failing `as` casts *always
  trap*, in every build mode; traps and `panic` **abort** and destructors do **not** run. There
  is no unwinding table, no landing-pad generation, no exception ABI. This is the single largest
  simplification versus a general systems language.
- **No trait objects / no `dyn`** (Core v1 reserved list). All trait dispatch is statically
  resolvable and monomorphizable. No vtables are required for Core v1.
- **No capturing closures.** Function values are non-capturing `fn(...) -> R` only (CD-021) —
  a bare code pointer, `Copy`, no environment, no escape analysis, no closure-conversion pass.
- **No `async`, no `unsafe`, no raw pointers, no general FFI** in Core. Host access is confined
  to the approved-provider boundary (metadata-bound), which the C3 provider experiment probes
  but does not finalize (C5.1 owns the stable Native Provider ABI).
- **Borrow checking completes before codegen.** References are lexically scoped with no lifetime
  annotations; by the time a backend sees the program, aliasing is already proven safe, so the
  backend may lower `&T`/`&mut T` to plain pointers without re-deriving lifetimes.
- **Deterministic, specified evaluation and drop order** (CD-007, CD-012, abstract machine).
  The backend has one legal observable order to preserve, not a permitted range.

The residual hard parts are ordinary compiler engineering, enumerated in the risk register (§6):
monomorphization, static trait/method dispatch lowering, reference/slice representation, exact
Drop elaboration (reverse declaration order, exactly once, skipped on abort), function-value
code-pointer ABI, and the opaque-resource/provider boundary.

## 4. Architecture hypothesis and falsifiers

Two candidate architectures are spiked against the frozen workload:

**Candidate A — Generated Rust/C (WP-C3.2).**
`typed HIR -> verified MIR -> generated Rust source -> rustc -> native executable`, with a small
STARK runtime library. Precedent: old Gate 5 already lowers the tensor deployment path to a
generated Rust host (`deploy/`), so some codegen-to-Rust machinery and its build orchestration
exist. Rust's own monomorphization, `Drop`, and borrow checker act as a correctness backstop;
STARK's trap-abort maps to Rust's `panic = "abort"` profile; STARK Drop (skipped on abort) maps
to Rust Drop under abort. Cross-platform reach comes free from rustc's target support.

**Candidate B — Direct Cranelift (WP-C3.3).**
`typed HIR -> verified MIR -> Cranelift IR -> native object -> link`, with the same STARK runtime
library. No rustc build dependency; fast compile; direct control of layout, calling convention,
and the runtime ABI; embeddable as a library. We implement monomorphization, trait-dispatch
lowering, and Drop elaboration ourselves (much of which is Gate C4 MIR work regardless of
backend).

**Leading hypothesis: SELECT-GENERATED (Candidate A) as the initial production backend, with
Candidate B as the primary challenger.** Rationale: it is the shortest path to *correctness*
(the charter's first priority, §1.6 rule 7), it reuses an existing lowering precedent, and it
borrows a mature toolchain's monomorphization/Drop/codegen so the first native release is spent
proving semantic parity rather than reimplementing a code generator. This is a hypothesis to be
*falsified by the spikes*, not a decision — WP-C3.4 selects under CE5.

**Falsifiers that would move the selection to Candidate B (SELECT-DIRECT) or REVISE:**

- rustc as a mandatory build dependency is judged unacceptable for the target user (toolchain
  weight, offline builds, contributor cost) and no bundling mitigation is adequate;
- generated-Rust compile time on the workload is materially worse than the direct backend by a
  margin that dominates the developer build loop and cannot be mitigated by incremental/caching
  within charter limits;
- the STARK→Rust semantic mapping requires generating constructs that re-introduce risk the
  front end already discharged (e.g. leaning on `unsafe` blocks to force a layout or an aliasing
  pattern Rust's borrow checker rejects but STARK's accepts), i.e. the backstop becomes a
  fight;
- binary size or startup time of the rustc-generated artifact is materially worse with no
  in-backend remedy.

**Falsifiers that would keep Candidate A / reject Candidate B:**

- the direct backend cannot reach representative generics + trait dispatch + reference/slice +
  Drop-bearing-resource parity within the spike budget (C3 requires spiking these, not full C6
  parity, but a candidate that cannot even spike them is not credible);
- Cranelift's debug-info / source-mapping story is too weak for `stark build`'s trap file:line
  requirement (WP-C5.5) and the gap is not closable.

**Not in scope as a candidate:** LLVM. It enters only via WP-C7.6/CE6 on measured evidence of a
material limitation the selected simpler backend cannot meet (charter §1.6 rule 10). A custom
bytecode VM is likewise excluded by default (charter rule 11).

## 5. Frozen workload

The workload is the fixed target every spike must attempt and every measurement is taken
against. Its semantic oracle is the reference interpreter; where a case exists in the frozen
`exec_snapshots` corpus (`corpus_version = 1.0.0`, `starkc/tests/exec_snapshots/corpus.lock`)
the golden `.snap` is the exact expected observable output. Items without a corpus case are
specified reference programs the spikes implement; when a spike needs the executable form, it is
added to the corpus under a `corpus_version` bump (per the freeze rules in WP-C3-ENTRY.md), not
silently.

| # | Workload item | Semantic oracle | Exercises |
|---|---|---|---|
| 1 | scalar arithmetic and branches | corpus `expr_stmt__01`, `expr_stmt__02` | scalar ops, control flow |
| 2 | loops and function calls | corpus `expr_stmt__03` | loops, break/continue, direct calls |
| 3 | structs and enums | corpus `struct_enum_trait__01`, `__02` | aggregate layout, discriminants |
| 4 | `Option`/`Result` and pattern matching | corpus `option_result__01`, `__02`, `expr_stmt__04` | enum payloads, match lowering, `?` |
| 5 | ownership moves and deterministic drops | corpus `ownership_drop__01` | move semantics, drop order/flags |
| 6 | strings/`Vec` operations through the runtime surface | corpus `collection_iter__01` | runtime-library calls, heap values |
| 7 | multi-file, multi-package CLI application | `exec_snapshots.rs::workspace_relocation...` (relocatable 2-package workspace); new corpus case at next bump | cross-package symbol linkage |
| 8 | one error/trap workload | corpus `primitive__01`, `primitive__02` (overflow traps) | deterministic trap + abort ABI |
| 9 | a generic function and a generic nominal type | corpus `struct_enum_trait__03` | monomorphization |
| 10 | user-defined trait method and operator dispatch | corpus `struct_enum_trait__04`, `metamorphic/trait_call_operator`+`_qualified` | static trait dispatch, operator desugaring |
| 11 | default trait method calling another trait method | new reference program (DEV-051/DEV-060 path; interp support confirmed) | default-method dispatch through `self` |
| 12 | references, mutable references, and slices | corpus `ownership_drop__02` (shared borrow) + new slice program | reference/slice ABI |
| 13 | a Drop-bearing nominal value passed through trait dispatch | new reference program | monomorphized Drop at a dispatch site |
| 14 | an opaque host resource with deterministic Drop | new reference program (`Value::File`-shaped) | provider-boundary resource Drop |
| 15 | basic file read/write through the proposed provider boundary | new reference program (interp: `Value::File` + `read_to_string`/`write`) | provider call ABI (disposable C3 experiment) |
| 16 | named function assigned to a typed function-value local (`fn(T) -> R`) | new reference program (interp: `Value::Function`) | function-value representation |
| 17 | a function value passed to another function and invoked indirectly | new reference program | indirect-call ABI |
| 18 | `Option::map`/`Result::map` called with a function value | new reference program | function value into a generic stdlib API |
| 19 | a function value stored in a struct field (route-table shape) | new reference program | function value in an aggregate |
| 20 | a cross-package function reference | new reference program | cross-package function symbol identity |
| 21 | a monomorphised generic function used as a function value | new reference program (record boundary if Core disallows) | generic instantiation as a value |
| 22 | repeated indirect invocation through one function-value local (`f(f(v))`) | new reference program (function values are `Copy`, `03-Type-System.md` §Copy and Drop) | no spurious move on re-invocation |
| 23 | a `Copy` aggregate containing a function-value field, copied, both copies invoked | new reference program | `Copy` aggregate with a code-pointer field |

Items 1–10 have frozen corpus coverage today. Items 11–23 are specified here and become
executable corpus cases (under version bumps) as the C3.2/C3.3 spikes and later C4.4/C6.5 need
them. **Two properties are deliberately unresolved and must be settled from the frozen spec or
by CE1/CE2 escalation before backend selection, never invented by a backend** (CD-022): whether
function values participate in `Eq`/`Ord`/`Hash`, and the canonical identity of a monomorphised
generic function value.

## 6. Risk register — the parts that are real engineering

Each risk is recorded for both candidates; the spikes must report on each (§7 measurement
dimensions map onto these).

| Risk | Generated Rust (A) | Direct Cranelift (B) |
|---|---|---|
| Monomorphization | rustc does it; we emit generic Rust | we implement it in MIR lowering (C4.5 anyway) |
| Static trait/method dispatch | emit as ordinary Rust trait impls or direct calls | resolve to concrete symbols in MIR; direct calls |
| References / `&mut` / slices | Rust refs (borrow checker backstops) *or* raw pointers if STARK accepts an aliasing Rust rejects | pointers + fat pointers for slices; validity already proven pre-codegen |
| Drop elaboration (reverse decl order, exactly once, skipped on abort) | Rust `Drop` under `panic=abort`; must confirm order matches STARK's | manual drop-flag lowering from MIR |
| Trap → abort ABI | `panic=abort` / explicit `abort()` with trap category + file:line | explicit trap block → runtime abort with category + location |
| Function values (code pointers, `Copy`) | Rust `fn` pointer types | Cranelift func refs / relocations |
| Drop-bearing resource through dispatch (13/14) | monomorphized concrete Drop; provider close in Drop glue | same, in our drop glue |
| Provider boundary / file I/O (14/15) | FFI-style call to runtime provider fn | same; ABI is the C3 experiment, C5.1 finalizes |
| Source mapping / trap file:line (WP-C5.5) | rustc debug info from generated source (mapping layer needed) | Cranelift debug info (more manual) |
| Build dependency / maintenance | requires rustc toolchain; codegen is "just text" | requires Cranelift crate; we own more passes |
| MIR consumption | must show it can consume verified MIR, not just typed HIR | same |

**Highest-attention items** (most likely to differentiate the candidates): the rustc
build-dependency question (A's chief liability), compile time (A's likely weakness, B's likely
strength), and source-mapped trap location (B's likely weakness). These three are where the
spikes should spend measurement effort first.

## 7. Measurement framework

Every spike (C3.2 generated Rust, C3.3 direct Cranelift) reports each dimension below, measured
against the frozen workload, with the reference interpreter as the semantic oracle and the
*stronger* of the two spikes — not "no compiler" — as the architecture comparator. All 13
roadmap dimensions:

1. **Implementation complexity** — lines/passes added, and which passes are reused vs.
   hand-written; qualitative "glue per language feature."
2. **Compile time** — wall-clock `stark build` per workload item and total, cold and warm.
3. **Executable size** — stripped binary size per representative item (esp. #7 the CLI app).
4. **Startup time** — process start to first output, to isolate runtime init cost.
5. **Runtime performance** — steady-state on the compute-bearing items; report the
   interpreter/native ratio, never a single headline multiple (charter caution).
6. **Source mapping / debug + stack-trace feasibility** — can a trap report file:line (WP-C5.5)?
   quality of a backtrace? demonstrated, not asserted.
7. **Cross-platform effort** — what each candidate needs per Tier-1 target (linux-x64,
   macos-arm64); Tier-2 windows-x64 noted.
8. **Semantic parity risk** — count and nature of observable mismatches vs. the interpreter on
   the workload; each mismatch is a defect, not a rounding difference.
9. **External dependency and maintenance burden** — toolchain/crate dependencies, their MSRV and
   licence, and the ongoing cost of owning the passes each candidate requires.
10. **MIR / runtime-ABI compatibility** — can the candidate consume a verified MIR (mandatory
    production shape) rather than only typed HIR? does it force ABI decisions that should belong
    to C5.1/CE4?
11. **Monomorphisation / generic-instantiation complexity** — how each handles items 9, 18, 21;
    duplicate-instantiation control; deterministic symbol names.
12. **Trait-dispatch representation and symbol-instantiation complexity** — items 10, 11, 13;
    symbol naming, dispatch lowering.
13. **Reference, slice, and Drop-bearing resource ABI risk** — items 12, 13, 14; pointer/fat-
    pointer representation, drop glue, provider-resource close.

A spike need not implement every item to report — an item it *cannot* lower is itself a
first-class measurement (an "unsupported construct" per WP-C3.2/C3.3), recorded, not hidden.

## 8. Decision-framework preview (WP-C3.4, CE5)

WP-C3.4 compares reference interpreter / generated-Rust spike / direct spike and records one
outcome (roadmap §3, WP-C3.4):

- **SELECT-GENERATED** — generated Rust/C is the initial production backend behind verified MIR.
- **SELECT-DIRECT** — the direct backend is the initial production backend behind verified MIR.
- **REVISE** — neither spike is yet sufficient; one specific bounded follow-up can resolve the
  blocker; the gate stays open.
- **BLOCKED** — no credible native path demonstrated; escalate to owner; C4 does not open and
  the roadmap is not complete. (Not "interpreter-only success" — that outcome does not exist.)

A selected architecture must specify its MIR consumption boundary, runtime ownership/ABI
direction, target-platform plan, debug/source-mapping approach, unsupported-MVP features with a
closure plan, and why the rejected candidate is not the initial path. `DEFER` is not a
completion outcome for mandatory gates C3–C7 (charter §5.3).

## 9. Companion: callable ABI / closure compatibility (CD-021, recommended pre-C5.1)

Because C4/C5.1 are about to freeze MIR call representation and the runtime ABI, a paper-only
**Callable ABI and Future Closure Compatibility Spike** should be drafted during C3 spike work
(recommendation, not approved scope). It has two parts: existing mandatory capability
(function-value representation, direct vs. indirect calls, cross-package function references,
generic function references, storage in aggregates, `Copy`/`Eq`/`Hash` properties, stack-trace
behaviour) and future compatibility (owned move closures through escaping borrows). Outcomes:
`GO` / `REVISE-ABI` / `DEFER-ESCAPING-BORROWS` / `ANNOTATIONS-LIKELY` / `NO-CURRENT-DESIGN`. Its
purpose is to keep the ABI chosen at C5.1 compatible with the function-value workload frozen
here (items 16–23) and with a plausible future closure design, without implementing closures.

## 10. Next

WP-C3.2 (generated Rust/C spike) and WP-C3.3 (direct Cranelift spike) implement the frozen
workload subset each can reach, reporting every measurement dimension in §7 and every
unsupported construct. WP-C3.4 selects under CE5. The interpreter remains the semantic reference
until a later gate records otherwise (charter §1.6 rule 6).
