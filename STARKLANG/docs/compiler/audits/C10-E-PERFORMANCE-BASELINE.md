# C10-E — performance baselines

**Packet:** C10-E, WP-C10.6. **Date:** 2026-08-09.
**Candidate SHA:** `01ba608` (branch `wp-c10/execution-plan`, a descendant of `develop` `1d20123`).
**Machine:** Darwin-arm64. **rustc:** 1.93.0 (254b59607 2026-01-19).
**Data:** `starkc/benchmarks/c10/darwin-arm64.json`.
**Harnesses:** `scripts/c10e-baseline.py`, `examples/c10e_phases.rs`,
`lsp::server::tests::c10e_lsp_latency` (`--ignored`).

> **Baselines only.** Plan §12.3 and WP-C10.6: regression thresholds may be added **only after
> stable baselines exist**, and this is the first. **No threshold is proposed here, and none should
> be inferred.** Nothing was optimised, and no number below was acted on.

---

# E0 — identity

```text
commit    01ba608          platform  Darwin-arm64        rustc  1.93.0 (254b59607 2026-01-19)
frozen    4650d475…        workload integrity  VERIFIED — 7 workloads, 16 files, 0 drift
reps      5 (median reported; min/max retained in the JSON)
```

**The harness refuses to measure a drifted workload.** `verify_frozen()` re-hashes every file
against `FROZEN.json` and exits before taking a single timing if any differs — a baseline against a
changed workload is worse than no baseline, because it looks comparable.

---

# E1 — the frozen workload set

Seven workloads, frozen at `4650d475` by per-file SHA-256 plus a `workload_hash`, unchanged by this
packet. **C10-E froze nothing new**: extending the set after seeing results is precisely the
denominator manipulation §7 forbids.

```text
w01_minimal   w02_arith_control   w03_generic_trait   w04_string_vec
w05_hash      w06_multi_package   w07_drop_ownership
```

Scaling inputs are **generated and labelled as such**, kept out of the frozen set: seven fixed
workloads have no size axis, so they can say what a representative program costs and cannot say how
cost grows.

---

# E2 — compiler front end: the phase split

Marginal cost of each stage over the previous one, in-process, median of 5.

| workload | lex | parse | resolve | check | front-end total |
| --- | ---: | ---: | ---: | ---: | ---: |
| w01_minimal | 6.5% | 22.5% | 20.8% | **50.2%** | 0.12 ms |
| w02_arith_control | 6.9% | 22.3% | 14.4% | **56.4%** | 0.74 ms |
| w03_generic_trait | 6.8% | 18.5% | 16.9% | **57.8%** | 0.84 ms |
| w04_string_vec | 7.6% | 23.0% | 15.2% | **54.3%** | 0.21 ms |
| w05_hash | 6.5% | 21.3% | 13.0% | **59.1%** | 0.25 ms |
| w06_multi_package | 6.7% | 22.6% | 26.8% | **43.9%** | 0.17 ms |
| w07_drop_ownership | 6.3% | 19.9% | 16.0% | **57.8%** | 0.27 ms |

**Type checking dominates at 44–59%; lexing is a flat ~6.5% everywhere.** `w06_multi_package` is the
only workload where resolution rises (26.8%) and checking falls — the one workload with a package
graph, which is where resolution has more to do.

This is the split `c7-baseline.py` cannot produce: that harness measures the STARK-vs-Cargo share of
a whole `stark build`, a real and coarser question.

## E2.1 Scaling — generated, and mildly superlinear

| functions | front-end | check share |
| ---: | ---: | ---: |
| 100 | 2.01 ms | 52.3% |
| 400 | 8.11 ms | 52.7% |
| 1,600 | 36.99 ms | 60.2% |
| 6,400 | 234.21 ms | **74.9%** |

**64× the input costs 117× the time** — roughly `O(n^1.17)` over this range — and the growth is
concentrated in checking, whose share climbs from 52% to 75%.

**Recorded, not acted on.** A rising share identifies where growth lives; it does not establish that
anything is wrong, and 234 ms for 6,400 functions is not a number that blocks qualification.
**Investigating it is a separate approved packet** (§12.3), and this baseline is what such a packet
would measure against.

---

# E3 — generated-program performance and artifact size

**Inherited, not re-measured:** `benchmarks/c7-workloads/c75-report-macos-arm64.json` carries
per-workload compile time, peak compiler RSS, executable bytes, and interpreter-vs-native runtime
ratios, taken under C7.5 on this same platform.

**It is one platform, and that limitation is E6's.** C10-E did not re-run it because nothing in this
packet changed code generation, and re-measuring on the same machine would produce a second number
of the same authority rather than more coverage.

---

# E4 — LSP latency, against the POST-DEV-213 implementation

| modules | cold open + analyse | edit → diagnostic | workspace symbol |
| ---: | ---: | ---: | ---: |
| 4 | 548 µs | 510 µs | 1 µs |
| 8 | 937 µs | 1,051 µs | 3 µs |
| 16 | 1,808 µs | 1,869 µs | 4 µs |
| 32 | 3,162 µs | 3,185 µs | 8 µs |

## E4.1 The finding: an edit now costs about a cold open

**`edit → diagnostic` tracks `cold open` almost exactly at every size.** That is C10-P's residual,
quantified: package-scoped invalidation drops every cached analysis of the package, so the first
query after an edit pays for a full re-analysis rather than an incremental one.

**This is the cost of correctness, and it was declared before it was measured.** C10-P recorded:
*"invalidation is now more eager, so a package with many open URIs recompiles more often… no
before/after was taken here and none is claimed."* E4 is that measurement. At 3.2 ms for a
32-module package it is comfortably inside interactive latency, so **nothing here argues for
reverting or optimising the repair.**

`workspace/symbol` is 1–8 µs — a merge over cached analyses, negligible against the analysis itself.

## E4.2 AS8's numbers are NOT a before/after, and are not used as one

AS8 measured 22 ms for one whole-package analysis and 181 ms for eight open URIs. **Those figures
describe a different architecture** — one analysis per open URI, invalidated per URI — and were
taken with a different harness on a differently-shaped package.

> **A comparison needs a demonstrably identical harness and workload. This is not one.** The
> numbers here are smaller; that is not evidence of a speedup and is not claimed as one. Owner
> ruling, and plan §12.

---

# E5 — variance

Median of 5 repetitions; `min` and `max` retained per workload in the JSON rather than discarded.
No outlier was removed. Repetition counts are low enough that **p95 is not computed** — reporting a
percentile from five samples would be arithmetic dressed as statistics.

Every derived share is range-checked: a value outside 0–100% aborts the harness. That check exists
because `c7-baseline.py`'s own header records a method error that produced a **−0.3%** host share,
caught only because a negative share is impossible.

---

# E6 — platform separation, stated honestly

```text
Darwin-arm64    MEASURED — everything in this document
Linux-x64       NOT MEASURED
Windows-x64     NOT MEASURED
```

**C10-E is a one-platform baseline.** The harnesses are portable (`c10e-baseline.py` records the
platform in the filename and the RSS unit in the payload, because macOS reports bytes and Linux
kilobytes), but no CI job runs them.

**C10-Q may not generalise these numbers to the platform matrix.** Producing Linux and Windows
baselines means adding a CI job that uploads `benchmarks/c10/<platform>.json` as an artifact — named
here as the concrete next step rather than left as an aspiration.

---

# APPENDIX (OD-8) — ONNX / tensor, and why it is not measured

> *These measurements qualify the already-supported, frozen tensor/ONNX maintenance surface only.
> They do not expand tensor capability, reopen the tensor productisation track, or support a claim
> of general tensor execution maturity.*

**No ONNX timing was taken, and the reason is availability rather than scope.** OD-8 ruled INCLUDE
(quarantined), and the inputs are not present:

```text
ONNX models      NOT COMMITTED. tests/fixtures/gate4/manifest.toml: ONNX fixtures "are generated
                 deterministically by test code" — there is no .onnx file anywhere in the tree
deploy inputs    tests/fixtures/gate5/fetch-input.sh downloads a reference image from
                 raw.githubusercontent.com. Deploy additionally needs ONNX Runtime
```

So import/verify could in principle be timed by driving the generator, and **deploy cannot be timed
offline at all**. Measuring one third of the appendix and labelling it "ONNX import/verify/deploy"
would be worse than measuring none.

**Owner decision required** — the smallest useful form: authorise driving the fixture generator for
import/verify timings and record deploy as unmeasurable offline; or accept the appendix as empty
with this explanation standing as the reason. **Either is consistent with OD-8; guessing is not.**

---

# What C10-E does NOT establish

```text
NOT a threshold             none proposed; §12.3 forbids inventing one after seeing the numbers
NOT a multi-platform claim  one machine, one platform. E6
NOT a DEV-213 before/after  E4.2 — different architecture, different harness
NOT an optimisation mandate the scaling shape is recorded; investigating it is a separate packet
NOT a runtime claim         E3 is inherited from C7.5 and was not re-measured
```
