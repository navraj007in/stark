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

## The reading that matters, and it is not the total

**`--lib` coverage is a map of where a unit-level control EXISTS.** A module at 0% is not untested —
`provider_synth.rs` is exercised end-to-end, and `AS8-MUT-032` was killed by the `stark-io` package
build. It means every check on that module runs through the whole pipeline, so a defect there is
only ever caught late, in aggregate, by a test that was aiming at something else.

Cross-referencing against the shared-fate register is where this becomes useful:

```text
typecheck/traits.rs   55.21%   ESF-COPY-001, ESF-DROP-001, ESF-TRAIT-001
typecheck/types.rs    51.66%   ESF-TYPE-001
```

**The two files holding the most `INVISIBLE` shared authorities are the two least covered by
unit-level tests in the type checker.** That is consistent with, and independent of, what the
mutation trials found: `ESF-TRAIT-001` has no control (`AS8-MUT-014/015` survived), and
`ESF-TYPE-001`'s recorded control does not control it (`AS8-MUT-013` survived).

Two independent measurements pointing at the same two files is worth more than either alone.

## What this baseline is NOT

```text
not a conformance claim      the work item forbids it, and AS8-MUT-003 shows why: full coverage of
                             a table proves nothing about the rule the table encodes
not a target                 no percentage is proposed, and none should be inferred
not total coverage           `--lib` only. The integration suites are excluded BY CHOICE, because
                             the useful question here is WHERE A UNIT-LEVEL CONTROL EXISTS
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
