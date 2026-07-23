# Gate C5 Exit Report — Native Core MVP

WP-C5.6 deliverable. Prepared and owner-approved 2026-07-23 against qualification head
`19254086d5f71db169fd1a1020bf30bddd284686` (`CD-076: close C5.5 and open C5.6
qualification`).

## Decision

**NATIVE-CORE-MVP-WITH-LISTED-DEVIATIONS.**

Gate C5 and WP-C5.6 are **CLOSED** by CD-077.

The compiler now takes a three-package Core workspace through the production `stark build`
command, canonical analysis, monomorphised MIR lowering, mandatory MIR verification, deterministic
generated-Rust emission, offline Cargo execution, and stable artifact installation. The resulting
standalone executable runs successfully. The admitted native subset agrees with the HIR and MIR
interpreters on completion, traps, source locations, layout answers, and destruction evidence.

This is deliberately not a full-Core native-conformance claim. The generated-Rust backend has a
smaller accepted profile than the HIR and MIR engines, and it refuses unsupported shapes before
rustc. The exact profile and the known boundaries are frozen below.

## Qualification identity

| Item | Qualified value |
|---|---|
| Git head | `19254086d5f71db169fd1a1020bf30bddd284686` |
| Branch | `main` |
| Host/target triple | `aarch64-apple-darwin` |
| Compiler crate | `starkc 0.1.0` |
| rustc | `rustc 1.93.0 (254b59607 2026-01-19)` |
| Cargo | `cargo 1.93.0 (083ac5135 2025-12-15)` |
| Crate MSRV | Rust 1.85 |
| MIR shape | `0.1` |
| MIR runtime surface | `0.1-A8` |
| Generated-Rust backend | `0.1` |
| Native runtime ABI | `0.1` |
| Native Provider ABI specification | `0.1` (specified, not executed in C5) |
| Target-layout contract | `stark-64-v1` |
| Layout contract/compiler revision | `1` / `1` |
| Native profile | `debug` |

The closure report is necessarily committed after the qualification head. The hash above is the
exact code-and-corpus state exercised by all evidence in this report; the closure commit changes
documentation only.

## Gate C5 exit criteria

| # | Exit condition | Evidence and disposition |
|---:|---|---|
| 1 | `stark build` accepts the frozen workspace | Production CLI built the relocated `app` workspace successfully. |
| 2 | MIR is verified before backend invocation | Build driver accepts only `VerifiedMirProgram`; 50 verifier tests pass. |
| 3 | Generated crate is deterministic | Build-key mutation coverage, two-lowering equality, frozen symbols, and relocation tests pass. |
| 4 | Cargo/rustc are invoked internally | CLI discovers and injects both tools; generated Cargo runs with `--offline` and the selected `RUSTC`. |
| 5 | Standalone executable at a documented path | `<package-root>/target/stark/debug/<package-name>`; reference artifact was `app/target/stark/debug/app`. |
| 6 | At least three packages | `app -> logic -> model`. |
| 7 | Cross-package concrete generic | `logic::wrap@[Int32/Int64]` and `logic::model::transform@[Int32/Int64]`. |
| 8 | Approved C5 baseline | Scalar/control, aggregates, enums, error values, concrete generics, function values, packages, traps, layout, ownership/Drop, and debug build are exercised. The String/output rows are the explicit CD-077 deviation below. |
| 9 | Three-engine agreement | 71 differential tests pass; dedicated trap, layout, Drop, and workspace comparisons pass. The reference app completes with empty stdout/stderr and status 0 in all applicable engines. |
| 10 | Approved snapshot subset replays | Corpus v1.4.0's two exact C5-native sources pass both the frozen snapshot harness and the HIR/MIR/native comparator. |
| 11 | Unsupported native features fail before rustc | Backend validation and CLI tests require `UnsupportedNative`/linkage diagnostics before generated compilation. |
| 12 | No known miscompilation/unsoundness/divergence in the subset | No such finding remains open. Bounded unsupported combinations are deterministic refusals. |
| 13 | Exact exit report | This document. |

## Frozen C5 native subset

The following table is the closure authority for what C5 native compilation actually supports. It
supersedes future-tense implementation notes in the C5 entry plan without changing Core semantics.

| Area | C5 native support |
|---|---|
| Input boundary | One verified, monomorphised `MirProgram` with source table, nominal type context, concrete bodies, and one entry instance. Malformed linkage is rejected before rustc. |
| Primitive values | Unit, Bool, Char, Int8/16/32/64, UInt8/16/32/64, Float32/64. |
| Constants | Unit, Bool, integer/Char, float, enum discriminants through lowered control flow, and concrete function-instance pointers. Source string constants are not admitted. |
| Operations | The MIR pure and checked scalar operations covered by the C5 matrix, including deterministic arithmetic overflow, division-by-zero, shift, bounds, assertion, and invalid-cast traps. |
| Control flow | Basic blocks, branch/switch, loops, lowered break/continue, calls, returns, recursion, and mutual recursion. |
| Functions/generics | Direct calls to concrete instances; concrete monomorphised generic functions and nominals emitted exactly once with no Rust generic parameters. |
| Function values | Non-capturing typed `fn(...) -> ...` values in locals, parameters, returns, tuples/structs, copies, value-only reachability, and indirect calls. Unsupported signatures are refused. |
| Aggregates | Tuples, fixed arrays, simple structs, user enums, and generated core `Option`, `Result`, and `Ordering` enums. |
| Patterns/errors | Already-lowered projections, discriminant switches, match paths, payload access, and `?` paths represented in verified MIR. |
| Ownership/Drop | MIR-directed move/copy, `ValueSlot` storage for admitted non-`Copy` values, canonical shared `DropPlan`, reverse component order, own destructor before components, active-variant destruction, exactly-once evidence, and supported partial moves. |
| References | The narrow ephemeral receiver/borrow lane approved by C5.3d: reference parameters and same-block borrows consumed without general storage, return, or cross-block lifetime. |
| Layout | Exact `size_of`/`align_of` answers from `stark-64-v1`, including primitive, tuple, array, struct, enum/core-enum, function-value, and generic/composite cases. Backend-private Rust layout is not observable authority. |
| Packages/linkage | One deterministic standalone link unit across path-dependency packages, with canonical sorted/unique symbols and complete instance reachability. |
| Traps | Deterministic category and source file/line/column evidence, stderr rendering, exit status 101, and abort without unwinding or post-trap destruction. |
| Build UX | `stark build [--locked] [--offline] [--keep-generated] [--emit-rust] [--verbose]`; debug profile; offline generated crate; stable final artifact; structured retained-directory/process evidence on backend failure. |
| Observable output | C5 native programs may complete with empty stdout/stderr and use assertions/traps as the observation channel. Source `String`, `str`, print/eprint, and Display-to-output runtime calls are not admitted. |

## Reference workspace and artifact

The frozen workspace is `starkc/tests/fixtures/c5-native-workspace/`:

```text
app 0.1.0
└── logic 0.1.0
    └── model 0.1.0
```

It contains 13 frozen canonical function symbols. The app exercises cross-package direct calls,
four concrete generic instances, returned/passed/copied/aggregate-stored function values,
value-only function reachability, indirect calls, struct construction/projection, `Option`
matching, a checked `while` loop, a target-layout query, and a checked cast.

The relocated clean build used:

```text
stark build --locked --offline --emit-rust --verbose
```

It reported 13 MIR bodies, completed verification, generated a deterministic crate, installed:

```text
<relocated-workspace>/app/target/stark/debug/app
```

and the installed executable exited 0. The production contract is:

```text
<package-root>/target/stark/debug/<package-name>
```

The generated crate is removed after a normal successful build unless retention or Rust emission
is requested. A cleaned backend-local binary is not reported as an extant artifact.

## Differential, snapshot, and qualification evidence

### Three-engine and focused C5 matrix

The C5.6 matrix ran these ten integration binaries:

```text
exec_snapshots                         4 passed
mir_verify                            50 passed
native_build_cli                       9 passed
native_c5_2e_traps                     4 passed
native_c5_3_aggregates_enums          21 passed
native_c5_4_function_values            8 passed
native_c5_4_generics                   3 passed
native_c5_4_linkage                   12 passed
native_c5_4_workspace                  6 passed
three_engine_differential             71 passed
                                      ----------
                                     188 passed
```

There were 0 failures and 0 ignored tests in this matrix. The 71-case differential suite includes
exact-value layout coverage, scalar/control/aggregate/generic/function-value agreement, deterministic
traps, and the dedicated Drop evidence. The six workspace tests include normal and trapping
three-engine outcomes, frozen symbols, linkage completeness, native execution, and relocation.

### Frozen execution snapshots

Execution-snapshot corpus `1.4.0` contains 65 locked `.stark`/`.snap` files, 25 primary cases,
seven metamorphic source pairs, and a workspace-relocation comparison. Most pre-C5 cases exercise
features outside the native subset, especially String/output and collections; they remain HIR
execution evidence and were not falsely counted as native coverage.

CD-076 approved exactly these two source files for C5 cross-backend replay:

- `c5_native__01_supported_completion.stark`;
- `c5_native__02_supported_overflow_trap.stark`.

The first completes with status 0 and empty output; the second reports `integer overflow` at its
frozen source location. Both pass their golden snapshots and the HIR/MIR/native comparator.

### Complete local qualification

Against the exact qualification head:

```text
cargo test --workspace --all-targets --all-features --no-fail-fast
  1,098 passed; 0 failed; 2 ignored
  55 test-bearing binaries

cargo test -p stark-runtime
  23 passed; 0 failed; 0 ignored

cargo fmt --all -- --check
  passed

cargo clippy --workspace --all-targets --all-features -- -D warnings
  passed
```

The two ignored tests remain the previously recorded opt-in/non-hermetic tests and are unrelated
to native C5. Runtime coverage proves both a matching runtime and the diagnostic direction of a
recorded-version/linked-runtime mismatch.

### Hosted CI

GitHub Actions run
`https://github.com/navraj007in/stark/actions/runs/29981161896` completed successfully for exact
head `19254086d5f71db169fd1a1020bf30bddd284686`. Both `spec fixture conformance` and
`fmt, clippy, test` jobs passed, including execution-snapshot verification, on the configured
`ubuntu-latest` stable-Rust environment.

## Listed deviations and future ownership

These are bounded scope boundaries, not hidden native support:

| Boundary | C5 behaviour | Owner |
|---|---|---|
| String/str, source string constants, print/eprint, Display output | Rejected before rustc. This is the explicit delta from the C5 entry-plan Output/Display rows accepted by CD-077. | C6 native Core parity |
| Vec, Box, slices, maps/sets, broad iterators/collections | Not in the admitted native profile; deterministic refusal where they reach the backend. | C6 |
| General references and full borrow/lifetime representation | Only the ephemeral C5 lane is admitted. | C6 |
| Broad non-`Copy` cross-block movement | Unsupported combinations are refused. | C6 |
| Partial move from a multi-drop-unit enum payload | Refused before rustc; single-unit/whole-payload paths remain supported. | C6 |
| By-value iteration over non-`Copy` arrays (DEV-090) | Front-end refusal, consistently before either execution backend. | Language completion/C6 review |
| Closures/captures, trait objects, async/concurrency | Outside C5. Function values are non-capturing concrete code references only. | C6 or later |
| General FFI/provider resource execution | Native Provider ABI v0.1 is specified but not executed. | Later provider/FFI gate |
| Windows and other target contracts | C5 qualified macOS arm64 locally and Linux x64 through hosted CI; Windows was explicitly deferred. | C6 platform matrix |
| Release/optimised builds, incremental compilation, stable public ABI, direct Cranelift | Debug generated-Rust pipeline only. | C7 or later |
| DEV-101 bounded follow-ups | Tensor-kind bound-name and callee-local associated-binding-type provenance remain outside the proven Core cross-package-generic fix; no admitted C5 Core miscompile is known. | C6/front-end follow-up |
| DEV-098 defensive reborrow reasoning | The formerly suspected valid-source double-use is unreachable; reachable reference-emission defects were fixed. The defensive `Operand::Copy` reborrow remains documented. | C6 MIR/reference audit |

No known native miscompilation, invalid-MIR acceptance, ownership unsoundness, or unexplained
three-engine divergence remains inside the frozen C5 native subset.

## Owner ruling and next gate

CD-077 accepts this report, the explicit String/output scope delta, and the evidence above;
closes WP-C5.6; and closes Gate C5 with verdict
**NATIVE-CORE-MVP-WITH-LISTED-DEVIATIONS**.

This closure does not silently open Gate C6. The next repository state is C6 entry planning and
owner approval, with the deferred matrix above as mandatory input.
