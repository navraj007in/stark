# Gate 7 — Symbolic Shape and Semantic Tensor Deployment (Proposal)

**Status:** Proposed (authorised by the Gate 6 `REVISE` decision, 2026-07-16).
**Type:** Roadmap-governed bounded experiment (ROADMAP.md §5).
**Owner:** STARK maintainer (Navraj Singh).

This proposal is authorisation to *plan*, not to implement. Implementation of
any work package below begins only after this proposal is reviewed for scope.
It does **not** assert that STARK will win the experiment.

---

## 1. Hypothesis

> A realistic inference deployment containing computed tensor dimensions and
> value-range state can be expressed, checked and deployed through STARK more
> generally and clearly than through a competent stable-Rust generated host,
> without introducing a custom tensor runtime.

The experiment is designed to be able to **falsify** this. If a real,
end-to-end typed-Rust ONNX Runtime host achieves the same guarantees at
comparable complexity, the outcome is `STOP PRODUCTISATION`, not a rewrite of
the experiment.

### Why this experiment exists

Gate 6 measured safety parity between STARK and a generated typed-Rust contract
on the Gate 5 ResNet pipeline. That parity held for two reasons the Gate 6 memo
recorded as caveats, both of which this gate removes:

1. **No computed shapes.** ResNet preprocessing uses only permute + cast +
   elementwise ops, so stable Rust never hit the `generic_const_exprs` wall
   (`cases/limit_reshape.rs`). Gate 7 requires a genuinely *computed* dimension.
2. **The Gate 6 comparator was a stub.** `rust-comparator/lib.rs` used a stubbed
   `predict`, a fake model handle, and FNV-1a in place of the real SHA-256
   integrity path. Gate 7 requires a real end-to-end ORT comparator (G7-06).

---

## 2. Selected workload

A single-stage object-detector **raw prediction head** — the class of pipeline
that naturally forces computed shape relationships during decode.

Structural requirement (the exact model is pinned in G7-01):

```text
model output:   Tensor<Float32, [B, A * (5 + C), H, W]>
reshape:        Tensor<Float32, [B, A, 5 + C, H, W]>
permute:        Tensor<Float32, [B, A, H, W, 5 + C]>
decode:         grid + offset broadcast over [B, A, H, W, *]
```

`A` (anchors) and `C` (classes) are fixed for the pinned model; `B` (batch) is
symbolic; `A * (5 + C)` is a **computed** channel dimension that must equal the
model's actual output channel count, and the reshape must preserve element count
(`A * (5 + C) * H * W`). These are the relationships a fixed-shape backend cannot
represent and stable Rust cannot type without `generic_const_exprs`.

### Candidate models (final choice + SHA pinned in G7-01)

Selection is deferred to G7-01, which must record a **clear licence**, pinned
source, reproducible export, SHA-256, stable opset, and documented signatures.
A permissively-licensed export is preferred; an AGPL/GPL detector (e.g. some
Ultralytics YOLO weights) is acceptable **only** if its licence is compatible
with committing evaluation code (not weights) to this MIT repository and is
recorded explicitly. Large model binaries are not committed unless repository
policy already permits it (fetch-by-checksum instead, per Gate 5).

---

## 3. Required tensor operations

Only the operations the frozen workload actually needs. Expected minimum:

* `reshape` (computed product equality);
* `permute` (already lowered — reuse);
* `broadcast_to` and broadcasting elementwise `add` / `mul` (grid + offset
  decode);
* symbolic batch `B` propagated across function boundaries;
* polynomial equality of dimension expressions (`A*(5+C) == A*5 + A*C`).

`matmul` / `concat` are added **only if** the frozen pipeline requires them. The
full tensor-extension op catalogue is explicitly out of scope.

---

## 4. Selected semantic annotation

**Exactly one** new statically-tracked property: **image value range**, added as
an annotation that mirrors the existing `device =` form
(`Tensor<T, S, device = D>` → `Tensor<T, S, range = R>`).

Proposed states (narrow, closed set):

```text
ByteRange     // integer image values conceptually in [0,255]
UnitRange     // floating-point values conceptually in [0,1]
Normalized    // channel-wise mean/std normalised values
Unspecified   // default; no claim
```

Defined transitions (no other op may silently claim a transition):

```text
cast       : ByteRange UInt8   -> ByteRange Float32
scale_255  : ByteRange Float32 -> UnitRange Float32
normalize  : UnitRange Float32 -> Normalized Float32
```

The exact surface syntax requires spec approval; on approval the annotation
lands in `STARKLANG/docs/extensions/Tensor-Model-Types.md` **first** (canonical
source), then the frontend, per the repo's "spec-first" rule. Markers are a
**frontend/compile-time** property: generated Rust need not preserve them at
runtime once checking passes (runtime assertions only at external refinement
boundaries). The feature stays optional under the `tensor` extension.

**Not in scope:** colour space, coordinate frames, physical units, or any second
annotation. One property, this gate.

---

## 5. Explicit non-goals

Deferred / prohibited for this gate (superset of ROADMAP.md §4):

* STARK VM, bytecode, JIT, interpreter fallback in the deployed host;
* LSP, formatter, package manager, networking, actors, distributed execution;
* training, autodiff, custom tensor kernels, GPU kernels;
* broad semantic-unit systems or **multiple** new tensor annotations;
* general dynamic-shape deployment beyond what the frozen workload needs;
* a catalogue of unrelated models (exactly one workload);
* replacing ONNX Runtime or writing numerical kernels;
* modifying the workload, model, or comparator to make STARK win.

A STARK VM may enter *architectural evaluation* only after a `GO AS LANGUAGE`
outcome — and even then the next step is a VM-vs-Rust-backend comparison, not
implementation.

---

## 6. Implementation boundaries

* **Frontend (G7-02):** implement/verify typing only for the ops in §3. Do not
  implement the whole tensor spec. Diagnostics name operation, axis/expression,
  expected vs actual relationship, and source locations — never internal
  polynomial IDs.
* **Deployment (G7-03):** add `DeployOp` variants only for the workload's ops
  (expected: `Reshape`, `BroadcastTo`, `Add`, `Mul`). Replace the literal-only
  deploy dimension (`Vec<u64>` / `Vec<Option<u64>>` today) with an explicit
  symbolic representation, e.g.:

  ```rust
  enum DeployDim { Literal(u64), Symbol(String),
                   Add(Box<DeployDim>, Box<DeployDim>),
                   Mul(Box<DeployDim>, Box<DeployDim>) }
  ```

  Dimensions must originate from checked STARK types; the generated host must
  **not** re-infer shape relationships. Symbolic runtime values come only from a
  verified refinement or the model boundary; computed dims are evaluated from
  checked expressions with overflow detection; unsupported dynamic expressions
  fail at deployment generation (no silent erasure to untyped vectors, no
  interpreter fallback).
* **Execution:** ONNX Runtime via the existing generated Rust host (T12). No new
  runtime.
* **Semantic markers (G7-04):** compile-time only; erased after checking.

Existing Gate 4 and Gate 5 tests must remain unchanged and green throughout.

---

## 7. Comparator requirements (G7-06)

The comparator must be a **real, end-to-end ONNX Runtime application**, not a
type-only demonstration. Explicitly correcting Gate 6, it must **not** use a
stubbed `predict`, a fake model handle, or a stand-in hash for the integrity
path. Required architecture:

```text
ONNX signature inspection -> generated Rust contract -> typed tensor wrapper
-> semantic range markers -> shape operations -> SHA-256 model binding
-> real ORT session -> real preprocessing -> real model execution -> real
   postprocessing
```

Permitted (must be the *strongest* credible stable-Rust host): const generics,
phantom types, marker traits, build scripts, generated model-specific structs,
and runtime refinement where stable Rust cannot express a relationship. It must
run the **same** defect corpus and produce outputs matching the same reference.
For each guarantee, classify where it is enforced:

```text
rustc compile time | generator/build time | startup before ORT
| runtime before inference | runtime during inference | not enforced
```

A structurally-reasoned case is **not** counted as measured unless it was
executed.

---

## 8. Defect corpus

Run identically through STARK and the Rust comparator.

Shape / contract defects:

1. incorrect reshape product;
2. broken symbolic dimension relationship (different symbols used as equal);
3. invalid broadcast;
4. wrong dtype;
5. wrong device;
6. declaration/signature drift;
7. runtime artifact swap.

Semantic value-range defects:

8. omit `scale_255` (missing range conversion);
9. apply `scale_255` twice (duplicate conversion);
10. `normalize` a `ByteRange` tensor directly;
11. pass `UnitRange` where the model contract requires `Normalized`;
12. merge incompatible semantic states;
13. erase semantic state without an explicit boundary.

For every defect record: case, command, exit code, diagnostic code, detection
stage, whether inference was attempted, expected result, actual result.

---

## 9. Measurements

* **Correctness:** output shape, top detections/predictions, max numerical
  difference vs an independent Python/ORT reference, tolerance, stable ordering,
  intermediate checked shapes, semantic-state transitions.
* **Operational:** compiler build, generated-host build, binary size, model
  size, cold start, session startup, warm preprocessing, inference latency,
  postprocessing, warm end-to-end, peak RSS.
* **Comparator complexity (counted, not asserted):** handwritten vs generator vs
  generated Rust lines; model-specific generated types; semantic marker types;
  runtime shape checks; ranks requiring generated implementations; operations
  forced to runtime dimension erasure; build time; binary size; diagnostics.

Evaluation scripts obtain provenance dynamically (commit, model SHA, input SHA)
and **fail when the tracked working tree is dirty or a tracked file is modified**.
No git revision, hash, benchmark, expected output, or comparator outcome is
hardcoded.

---

## 10. Exit criteria (candidate outcomes)

The decision (G7-07) records exactly one outcome from ROADMAP-governed options:
`GO AS LANGUAGE`, `NARROW TO DEPLOYMENT DSL`, `PRODUCTISE VERIFIER`, `RETAIN AS
RESEARCH LANGUAGE`, or `STOP PRODUCTISATION`.

`GO AS LANGUAGE` is permitted **only** when all hold, from executed evidence:

* STARK carries computed symbolic shape relationships through native deployment;
* at least one material defect is caught statically that the real stable-Rust
  comparator must defer to runtime or represent with disproportionate generated
  machinery;
* semantic range checking gives a clear practical benefit;
* the deployment backend remains maintainable;
* the advantage does not depend on unimplemented future work.

## 11. Stop criteria

Record `STOP PRODUCTISATION` when any hold:

* the real Rust comparator achieves the same guarantees at comparable
  complexity;
* STARK's deployment backend becomes excessively complex to support the
  workload;
* the semantic system does not prevent meaningful defects;
* the experiment requires hardcoded, model-specific behaviour to succeed.

---

## 12. Work packages and branches

| WP | Branch | Deliverable |
| --- | --- | --- |
| G7-00 | `gate7/g7-00-proposal` | this proposal (docs only) |
| G7-01 | `gate7/g7-01-freeze-workload` | pinned model + reference + freeze doc |
| G7-02 | `gate7/g7-02-shape-frontend` | frontend symbolic-shape checking + tests |
| G7-03 | `gate7/g7-03-deployment-lowering` | symbolic-dim deploy lowering |
| G7-04 | `gate7/g7-04-semantic-range` | value-range annotation + defects |
| G7-05 | `gate7/g7-05-native-evaluation` | end-to-end native evidence |
| G7-06 | `gate7/g7-06-rust-comparator` | real end-to-end typed-Rust comparator |
| G7-07 | `gate7/g7-07-decision` | `starkc/docs/gate7-decision.md` |

Each WP: clean tracked tree at start, focused commits, and after any compiler
change: `cargo fmt --check`, `cargo clippy --all-targets --all-features -D
warnings`, `cargo test --all-targets --all-features`, `cargo build --release
--all-targets`, `cargo doc --no-deps`, `git diff --check`.

---

## 13. Evidence locations

```text
STARKLANG/docs/proposals/GATE7_SYMBOLIC_SEMANTIC_DEPLOYMENT.md   (this file)
starkc/examples/gate7/{README.md, model.stark, valid_pipeline.stark, fetch-artifacts.py}
starkc/tests/fixtures/gate7/{reference.py, inputs/}
starkc/tests/results/gate7/{environment,reference,stark-host,defect-matrix,benchmark-runs,comparison}.json
starkc/docs/gate7-decision.md                                    (G7-07)
```

---

## 14. Separate verifier-validation track (independent of this gate)

Gate 7 tests the **language** thesis only. Whether `stark verify` should become
an independent product is a *separate* evaluation, prepared after G7-05: at
least three ONNX signature forms (fixed, symbolic, multi-port), deterministic
JSON output, a fresh-clone evaluator guide, and feedback from **at least three
real external developers** on whether they would add the verifier to CI. Human
participants must be real; feedback must never be fabricated. The verifier and
language decisions are kept independent.
