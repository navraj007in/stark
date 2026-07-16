# Gate 7 decision memo — language direction

**Status:** Complete. Evidence gathered (G7-00 … G7-06b); decision recorded.

**Decision:** **RETAIN AS RESEARCH LANGUAGE** (owner-confirmed 2026-07-16),
time-boxed, with the demonstrated value understood as *deployment-scoped* (the
deployment-DSL direction) and the **verifier-validation track** (real external
developers, §13 of the proposal) as the gating next step before any
productisation. Rationale in *Product / Language / VM implication* below.

**Verdict at a glance (scoped to what Gate 7 actually tested):**

- **Tensor-track technical verdict: POSITIVE** — STARK demonstrably carries
  runtime-symbolic dimensions and *proves* compile-time shape-relationship facts,
  detects artifact drift earlier (deploy-time), and needs far less application
  code (18 source → 25 generated vs 260 handwritten Rust) than the strongest
  typed-Rust host; both run identical, bit-exact inference.
- **Tensor productisation verdict: DEFER** — no external adoption evidence yet.
- **Current project policy: RETAIN AS RESEARCH LANGUAGE.**
- **Broader language verdict: UNRESOLVED** — concurrency, reactive, cloud,
  backend, and AI-native tracks were outside Gate 7 and need independent
  proposals and evidence. This experiment neither proved nor disproved them.

---

## Hypothesis

> A realistic inference deployment containing computed tensor dimensions and
> value-range state can be expressed, checked and deployed through STARK more
> generally and clearly than through a competent stable-Rust generated host,
> without introducing a custom tensor runtime.

The experiment was built to be able to falsify this (a real, end-to-end typed-
Rust ORT comparator, not the Gate 6 stub).

## Workload

Tiny YOLOv2 (ONNX Model Zoo, MIT), frozen in G7-01. Input `image`
`Float32 [B,3,416,416]` (dynamic batch), output `grid`
`Float32 [B,125,13,13]` where `125 = A*(5+C)` (A=5 anchors, C=20 classes) — a
**computed** channel dimension. The **natively deployed** pipeline (scope
amended post-G7-05) is `refine → predict → reshape → permute`, producing the
per-cell tensor `[B,5,13,13,25]`. The `125 → 5×25` reshape over a symbolic batch
is the differentiator the Gate 5 ResNet wedge lacked. Full grid/anchor box
decode and the value-range transition ops are **frontend-checked only** and are
not lowered to the native host.

## STARK implementation

Frontend symbolic-shape checking (G7-02, source-level diagnostics), native
deployment lowering of symbolic dimensions (G7-03: `DeployDim`, runtime `DimEnv`,
reshape) with model-boundary shape enforcement (G7-03b), and one semantic
property — image value range (G7-04/04b: `range = R`, exact-match unification,
named transitions, correct per-op propagation). Evidence: `tests/results/gate7/`.

## Rust comparator (G7-06/06b)

A **real** typed-Rust ONNX Runtime host (correcting the Gate 6 stub): real
SHA-256, real `ort` session, real inference, and — after G7-06b — the reshape
and permute performed **by the typed `Tensor4`/`Tensor5` values themselves**
(not a parallel `ndarray` path). Strongest stable-Rust typing: per-rank const-
generic tensors, phantom device tag, phantom value-range markers.

## Defect comparison (both execute 13 corresponding defect intents)

Both reject all 13 test programs. But "same corpus" is a qualified claim: three
of the Rust symbolic-shape cases are **approximations**, not equivalent proofs —
`d1`/`d3` check disagreement with a *fixed generated signature* (they do not
prove the element relationship; the comparator records `proves_relationship:
false`), and `d2` compares *compile-time const* batches, not runtime-symbolic
dimensions. And the *stage* differs on drift:

| | STARK | typed-Rust comparator |
| --- | --- | --- |
| Rejected eventually | **13 / 13** | **13 / 13** |
| Rejected **before inference** | **13 / 13** | **12 / 13** (drift caught after the model runs) |
| Reshape product (`5·26≠125`) | compile-time, **proved** (`125==5·25`) | compile-time, **fixed signature only — not proved** |
| Symbolic/dynamic batch | carried symbolically, compile-time | **not expressible** — batch is a compile-time const |
| dtype / device | compile-time | compile-time (phantom) |
| Value-range (6 defects) | compile-time | compile-time (4 markers) — **cheap parity** |
| Declaration/artifact drift | **deploy-time** (before the host is built) | **runtime, after `session.run`** — the model has already executed |
| Runtime artifact swap | runtime, before inference (SHA-256) | runtime, before inference (real SHA-256) |

## Correctness

Both hosts run real inference and their output matches the independent Python
reference **bit-for-bit** on the identical input (`max_abs_diff = 0.0`); STARK
model correctness is additionally anchored by the ONNX-bundled oracle
(`2.19e-05`). The STARK native evidence records 13/13 defects before inference
(`defect-matrix.json`).

## Integration complexity

Comparing like with like — *application implementation*, separate from
evaluation/test infrastructure (present on both sides):

- **Application:** STARK **18** pipeline source lines → **25** generated-host
  lines, vs the Rust comparator's **260** handwritten lines (typed layer 150 +
  host 110). A ~14× expressiveness gap at the application level. Two per-rank
  tensor types, one bespoke per-reshape function, two runtime shape checks on
  the Rust side; one ONNX Runtime backend on both, no new kernels or runtime.
- **Evaluation infrastructure (counted separately):** the Rust side adds ~283
  lines (defect cases 66 + measurement harness ~217); STARK's evaluation harness
  is counted separately too. This is *not* part of the expressiveness comparison.
- The Rust comparator is **handwritten**, not generated; a reusable generated
  solution would add generator code on top of the 260.

## Diagnostics

STARK emits source-level tensor diagnostics (E0212 with axis/provenance, no
internal ids — G7-02); the Rust comparator emits generic `E0308`/`E0599` type
errors. Serviceable on both sides; STARK's are domain-specific.

## Deployment cost

A wash: both compile to a single ORT-backed native binary. STARK host: binary
24.8 MB, model 63.5 MB, startup ~474 ms, warm median ~50 ms, peak RSS ~420 MB
(single-sequence measurement).

## Limitations

- The advantage is demonstrated on **one batch-1 CV pipeline**; the deployed
  scope is reshape/permute (box decode and value-range transitions are
  frontend-only). Neither side executes semantic value-range preprocessing.
- **No product/adoption evidence** exists: the verifier-validation track (§13,
  real external developers) has not been run.
- The comparator is handwritten, not generated; a generated Rust host might
  narrow the ergonomic gap (though not the symbolic-batch or proof gaps).

## Product implication

The value is real but concentrated in the **tensor/deployment layer**, not Core:
symbolic-shape *proof*, a *dynamic* batch in the type system, *deploy-time* drift
detection, and far less code than the strongest typed-Rust host. Value-range is a
convenience, not the decisive edge (cheap on both sides). There is, however, **no
evidence of user demand**. Productising now — as either a full language or a DSL
— would commit ahead of that evidence.

## Language implication

**Scope discipline (important).** Gate 7 tested *only* the tensor/model-
deployment proposition. It deliberately excluded — per ROADMAP §4 and the
proposal's non-goals — networking, actors, distributed execution, cloud
deployment, and broad Core language work. So the finding is bounded: *in this
ONNX CV experiment* the Core language (ownership, generics, execution) was not
the differentiator; every measured advantage lived in the tensor extension and
the deployment backend.

That does **not** establish that Core, concurrency, reactive systems, cloud
integration, or AI-native development cannot be differentiators in *their own*
experiments — those propositions were never tested here. The honest conclusions:

- **Tensor/deployment product wedge:** the best-supported product forms are a
  *narrow deployment DSL* (generating typed ORT hosts) or the *verifier* —
  contingent on adoption evidence, not a decision to build now.
- **Broader language thesis:** **unresolved.** It requires its own bounded
  validation tracks (ROADMAP §5 proposals), each with its own evidence.

`RETAIN AS RESEARCH LANGUAGE` is therefore an owner-selected *current project
policy* — not a scientific verdict against the language forced by this one
tensor experiment.

## VM implication

A STARK VM may enter architectural evaluation **only after a `GO AS LANGUAGE`
outcome** (proposal §5), and even then the next step would be a VM-vs-Rust-
backend comparison, not implementation. This memo recommends RETAIN AS RESEARCH,
so **VM work remains deferred** — and the ORT-via-generated-Rust backend (T12)
has shown no evidence of being the bottleneck.

## Authorised next work

1. **Verifier-validation track (§13)** — the gating experiment: `stark verify`
   over ≥3 ONNX signature forms (fixed/symbolic/multi-port), deterministic JSON,
   a fresh-clone evaluator guide, and feedback from **≥3 real external
   developers** on whether they would add it to CI. This tests *demand*, which
   the technical experiment did not. Its own §5-style proposal is required first.
2. Keep the Gate 7 evidence reproducible (harnesses are committed and green).

## Deferred work

Native box decode (`broadcast_to`/`add`/`mul`, anchor math), executable
value-range transitions, additional semantic properties (colour space,
coordinate frame, units), multiple workloads/models, any VM/bytecode/JIT, and
broad Core language expansion — none authorised by this decision.

---

**Decision recorded: RETAIN AS RESEARCH LANGUAGE** (owner-confirmed 2026-07-16).
The next narrow experiment — the verifier-validation track — requires its own
roadmap-governed §5 proposal (owner, bounded scope, measurable exit criteria)
before any implementation. No language expansion, DSL productisation, or VM work
is authorised by this decision.
