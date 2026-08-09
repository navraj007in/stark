# AS8 — coverage baseline for the compiler crate

**Packet:** AS8. **Date:** 2026-08-09. Tool: `cargo-llvm-cov` 0.8.7 (installed for this packet;
neither it nor `llvm-tools-preview` was present before).

AS8's work item is explicit about what this must and must not be:

> *"Establish line/branch coverage baselines for compiler crates and report uncovered semantic
> arms — **without imposing an arbitrary percentage as a conformance claim**."*

So this file publishes numbers and **states no target**. A coverage percentage is not evidence that
a rule is correct; AS8's own mutation trials are the demonstration of that — `copy_canon_matrix`
covers `core_method_signature` completely and, per `AS8-MUT-003`, controls nothing.

---

## Full-corpus baseline — the headline number

**Completed 2026-08-09** under a disk watchdog that never had to fire.

```text
TOTAL     regions 83.05%     functions 84.92%     lines 83.64%
```

| Module | Regions | Lines |
| --- | ---: | ---: |
| `provider_synth.rs` | 96.05% | 96.07% |
| `typecheck/trait_contracts.rs` | 91.85% | 90.23% |
| `provider_resolve.rs` | 89.06% | 89.70% |
| `typecheck/body.rs` | 88.61% | 86.68% |
| `typecheck/types.rs` | 86.71% | 87.57% |
| `typecheck/traits.rs` | 82.77% | 87.09% |
| `typecheck/infer.rs` | 73.58% | 73.82% |

## THIS CORRECTS A CLAIM THIS DOCUMENT MADE FROM THE `--lib` RUN

The `--lib` version of this file said:

> *"The two files holding the most `INVISIBLE` shared authorities are the two least covered by
> unit-level tests in the type checker … Two independent measurements pointing at the same two
> files is worth more than either alone."*

**That correlation was a `--lib` artefact and does not survive the full corpus.** `traits.rs` is
82.77% and `types.rs` 86.71% — unremarkable against a project total of 83.05%, and neither is the
lowest (that is `infer.rs` at 73.58%). `provider_synth.rs` moved from **0.00% to 96.05%**.

The replacement finding is stronger than the one it retires:

```text
typecheck/traits.rs   82.77% region coverage   AND ESF-TRAIT-001 HAS NO CONTROL
                                               MUT-014 and MUT-015 made Core trait contracts
                                               arbitrarily wrong and BOTH SURVIVED
typecheck/types.rs    86.71% region coverage   AND MUT-019 (strip_ref) SURVIVED
                                               AND MUT-013 (TYPE-PRIM-001) SURVIVED
```

**High coverage and no control, in the same file, measured two ways.** That is the same point
`AS8-MUT-003` made about `copy_canon_matrix` — full coverage of a table proves nothing about the
rule the table encodes — and it is now demonstrated on the two files carrying the most `INVISIBLE`
authorities in the register. Coverage says a line RAN. It does not say anything would have noticed
had the line been wrong.

## Unit-test baseline (`--lib`)

**This measures what the IN-MODULE `#[cfg(test)]` tests reach — not what the compiler is tested by.**
The 209 integration binaries under `starkc/tests/` are excluded, and they carry most of the
compiler's real exercise. Read as "how much has a unit-level control", never as "how much is
tested".

```text
TOTAL     regions 46.69%     functions 58.00%     lines 48.34%
```

| Module | Regions | Lines | Reading |
| --- | ---: | ---: | --- |
| `typecheck/mod.rs` | 96.57% | 97.40% | the facade, exercised by every unit test that runs the pass |
| `typecheck/state.rs` | 91.94% | 94.01% | |
| `resolve.rs` | 83.34% | 86.28% | the strongest unit-tested module — and `AS8-MUT-038` was killed by exactly these tests |
| `typecheck/items.rs` | 82.25% | 75.77% | |
| `typecheck/trait_contracts.rs` | 78.77% | 71.74% | the module AS7/CD-393 created |
| `typecheck/bounds.rs` | 76.29% | 78.95% | |
| `typecheck/body.rs` | 70.71% | 66.74% | the largest module, 4,842 lines |
| `typecheck/convert.rs` | 68.03% | 66.72% | |
| `typecheck/patterns.rs` | 67.10% | 62.18% | |
| `typecheck/traits.rs` | 55.21% | 59.06% | **holds `ESF-COPY-001`, `ESF-DROP-001`, `ESF-TRAIT-001`** |
| `typecheck/infer.rs` | 54.02% | 54.89% | |
| `typecheck/types.rs` | 51.66% | 58.09% | **holds `ESF-TYPE-001`, `is_integer`, `strip_ref`** |
| `session.rs` | 32.91% | 33.11% | |
| `test_runner/mod.rs` | 15.66% | 10.57% | |
| `provider_resolve.rs` | **0.00%** | **0.00%** | 393 regions, no unit-level control at all |
| `provider_synth.rs` | **0.00%** | **0.00%** | 253 regions, no unit-level control at all |

## The `--lib` reading that survives

**`--lib` coverage is a map of where a unit-level control EXISTS.** A module at 0% is not untested —
`provider_synth.rs` is exercised end-to-end, and `AS8-MUT-032` was killed by the `stark-io` package
build. It means every check on that module runs through the whole pipeline, so a defect there is
only ever caught late, in aggregate, by a test that was aiming at something else.

The gap between the two runs is the useful quantity, and `provider_synth.rs` is the clearest case:
**0.00% under `--lib`, 96.05% under the full corpus.** Nothing about that module is untested; it is
that *every* check on it runs end to end. A defect there is caught late, in aggregate, by a test
aimed at something else — which is exactly how `AS8-MUT-032` died, in a `stark-io` package build.

That is a real property worth knowing per module. It is **not** a proxy for whether a rule is
controlled, and the retired claim above is what happens when it is used as one.

## What this baseline is NOT

```text
not a conformance claim      the work item forbids it, and AS8-MUT-003 shows why: full coverage of
                             a table proves nothing about the rule the table encodes
not a target                 no percentage is proposed, and none should be inferred
both runs are published      full corpus 83.05% is the headline; `--lib` 46.69% answers the
                             narrower question of where a UNIT-LEVEL control exists
not branch coverage          llvm-cov reports region and line coverage; the "branches" column is
                             empty for this toolchain and is not silently substituted
```

## Reproducing

```bash
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov --locked
CARGO_TARGET_DIR=/tmp/as8-cov cargo llvm-cov --manifest-path starkc/Cargo.toml --lib --summary-only
```

**Instrumented builds are large.** This packet filled the disk to 99% once already — from mutation
target directories, not from coverage — and a full instrumented build of all 209 integration
binaries is substantially bigger again. Check `df` before running, and use a scratch
`CARGO_TARGET_DIR` that can be deleted.
