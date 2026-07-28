# WP-C7.0 — Baseline: the build path as it actually is

**Status:** `COMPLETE` (CD-185). Inventory, frozen workloads, stage-timing harness, two-path
clean-build comparison, panic-site audit and the C7/C8 lease map are all done and measured.
**Head at inventory:** `e5651b0` (the Gate C6 closure commit).
**Method:** every statement below was read out of the tree, not assumed. The directive's suggested
paths were checked first and several do not exist.

---

## 0. Paths that do NOT exist

The directive lists likely paths "but must be verified before editing". Verified:

| assumed | actual |
| --- | --- |
| `starkc/src/cli/**` | does not exist — CLI is `src/main.rs` and `src/bin/stark.rs` |
| `starkc/src/build/**` | does not exist — build orchestration is `src/native_build.rs` |
| `starkc/benches/**` | does not exist — no benchmark harness of any kind |
| `starkc/docs/compiler/evidence/c7/**` | created by this commit |

## 1. Two binaries, and only one of them builds

| binary | source | commands |
| --- | --- | --- |
| `starkc` | `src/main.rs` | `check`, `run`, `parse`, `lex`, `lsp`, `import`, `verify`, `deploy` |
| `stark` | `src/bin/stark.rs` | `check`, `build`, `run`, `test`, `fmt`, … |

`starkc run` is the **HIR interpreter**, not native execution — it calls `starkc::interp::run`. The
native path is `stark build`, via `native_build::build_current_package`.

## 2. `stark build` today

```text
stark build [--locked] [--offline] [--keep-generated] [--emit-rust] [--verbose]
```

`BuildCommandOptions { locked, offline, keep_generated, emit_rust, verbose }`.

**There is no `--release` and no `--target`.** The help text says "Compile a native debug
executable", and that is exact: debug is not a default among alternatives, it is the only mode.

### Pipeline

```text
find_package_root
  → PackageGraph::load_from_root_with_modes(locked, offline)
  → validate_binary_name
  → analyze_project (parse, resolve, typecheck)
  → lower_program → verify_program
  → native_toolchain::discover
  → emit_native_debug_with_toolchain   (generate Rust crate)
  → cargo build --locked --offline
  → copy artefact
```

## 3. Findings that shape C7.1

### 3.1 Overflow trapping does NOT depend on the Rust profile — already safe

The generated code emits **explicit** `checked_add` / `checked_mul` on widened `i128`
(`emit_bodies.rs`), not Rust's profile-dependent overflow checks. So Cargo's release default of
`overflow-checks = false` cannot silently turn STARK's always-trapping arithmetic into wrapping.

This is the single largest semantic hazard §3.2 anticipates, and it is already neutralised by
construction. Recorded here because the opposite would have been a blocking discovery.

### 3.2 Trap abort does NOT depend on the panic strategy — already safe

`stark_runtime::trap::abort` ends in `std::process::exit(101)`, not a Rust panic. `exit` runs no
destructors, so DROP-ABORT-001 holds irrespective of `panic = "abort"` vs `"unwind"`.

### 3.3 But the generated crate has NO `[profile.release]` — a real hazard

`generated_cargo_toml` emits only:

```toml
[profile.dev]
panic = "abort"
```

Adding `--release` without adding a release profile would inherit Cargo's defaults, including
`panic = "unwind"`. §3.1/§3.2 mean STARK's OWN traps stay correct — but any Rust-level panic
reachable in generated code or the runtime (a slice bound, an `unwrap`, an allocation failure)
would then UNWIND and run destructors, which DROP-ABORT-001 forbids. **`[profile.release]` must set
`panic = "abort"` explicitly**, and every setting §6.6 lists must be recorded rather than inherited.

A stale comment in `trap.rs` compounds this: it says no Drop glue runs because "every
locally-declared type so far is `Copy`", which was true in C5 and has not been true since C6.1.

### 3.4 Output layout has no target component

```rust
let target_root = package_root.join("target/stark");
let final_dir   = target_root.join("debug");
```

→ `<package>/target/stark/debug/<binary>`

No target triple, and "debug" is a literal rather than a profile variable. §3.4 requires layouts
that prevent profile, target and package collisions; a cross-target build would today overwrite the
host build's artefacts.

### 3.5 The target is never threaded through the build

`src/target.rs` defines the tier table — `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`
(Tier 1), `x86_64-pc-windows-msvc` (Tier 2), `x86_64-apple-darwin` (Tier 3) — and the backend's
`build.rs` can accept a target. But **`native_build.rs` mentions the target zero times**: the CLI
has no way to supply one and never does. `--target` is new wiring, not a new flag on existing
plumbing.

## 4. Reproducibility candidates identified statically

To be measured, not yet claimed:

- the generated crate is written into a **temporary directory whose name contains the process ID
  and thread ID** (`stark_c6pkg_…{pid}_{ThreadId}` in the test harness path; the build path needs
  the same audit);
- the generated `Cargo.toml` embeds an **absolute path** to the runtime crate;
- `stark.lock` is written into the package root during resolution;
- canonical symbols are content-addressed and relocation-independent (DEV-114, verified in C6) —
  this one is expected to reproduce;
- generated-source ordering is deterministic by construction in C6's evidence, but has never been
  compared across two checkout paths, which is §4.3's experiment.

---

## 5. Verified invariants (CD-185)

Two properties make release-profile work safe, and both are properties of the CODE rather than of
any build setting. They are recorded because the opposite would each have been blocking.

**5.1 Integer overflow is profile-independent.** The backend emits explicit `checked_add` /
`checked_mul` on widened `i128` (`emit_bodies.rs:1777`), not Rust's profile-dependent overflow
checks. Cargo's release default of `overflow-checks = false` therefore cannot turn STARK's
always-trapping arithmetic into wrapping arithmetic.

**5.2 Trap paths do not unwind.** `stark_runtime::trap::abort` and `abort_with_message` end in
`std::process::exit(101)`, which terminates without unwinding and so runs no destructors. That is
what makes DROP-ABORT-001 hold irrespective of the `panic` strategy.

## 6. Panic-site audit (§ directive item 8)

Every `panic!`, `unreachable!`, `unwrap` and `expect` in the shipped runtime, classified:

| site | class |
| --- | --- |
| `provider_abi.rs:164,209` | **test-only** — inside `#[cfg(test)] mod tests`, never in a shipped binary |
| `slot.rs:76` | **internal bug** — its own message says "STARK compiler defect, not a program fault" |
| `slot.rs:663` | **verifier-impossible** — `unreachable!("V-DISC-1")` |
| `format.rs:121` | **internal invariant** — parses Rust's own float-formatting output |
| `vec.rs:95` | **converted STARK trap** — `narrow_index` checks first; out-of-range routes to `trap::abort` |

**No user-reachable Rust panic exists in the runtime.** Every user-reachable failure is converted to
a STARK trap. This *weakens* §3.3's hazard from "unwinding would run destructors" to "unwinding has
no user-reachable path today" — but the guarantee then rests on this audit rather than on the build
configuration, so `panic = "abort"` must still be set explicitly in the release profile as
defence-in-depth, and this table must be re-run when the runtime grows.

## 7. Frozen workloads

Seven packages under `starkc/benchmarks/c7-workloads/`, hashed in `FROZEN.json` at `4650d47`:

| workload | covers |
| --- | --- |
| `w01_minimal` | smallest program |
| `w02_arith_control` | arithmetic, `while`, branches |
| `w03_generic_trait` | generics and trait dispatch |
| `w04_string_vec` | `String`/`Vec` allocation |
| `w05_hash` | `HashMap`/`HashSet` |
| `w06_multi_package` | two-package graph |
| `w07_drop_ownership` | `Drop`, moves |

The parser/structured-data workload (§2.2 item 8) and the P1 practical workloads are deliberately
absent; §8.1 requires a missing platform feature to be recorded as a dependency failure rather than
substituted with an unrelated benchmark.

## 8. Stage timings — host compilation dominates

`scripts/c7-baseline.py --measure`, median of 3 cold builds, macOS arm64.

| workload | total (s) | host cargo (s) | STARK (s) | host % | generated (B) | exe (B) |
| --- | --- | --- | --- | --- | --- | --- |
| w01_minimal | 0.359 | 0.234 | 0.125 | 65.1 | 1 230 | 484 656 |
| w02_arith_control | 0.374 | 0.244 | 0.130 | 65.2 | 5 889 | 493 296 |
| w03_generic_trait | 0.383 | 0.255 | 0.127 | 66.7 | 7 733 | 491 728 |
| w04_string_vec | 0.399 | 0.263 | 0.136 | 65.9 | 7 809 | 510 048 |
| w05_hash | 0.405 | 0.277 | 0.128 | 68.4 | 9 404 | 536 304 |
| w06_multi_package | 0.392 | 0.259 | 0.134 | 65.9 | 5 858 | 491 600 |
| w07_drop_ownership | 0.403 | 0.261 | 0.141 | 64.9 | 6 388 | 502 688 |

**Host Cargo/rustc is 65–68% of every cold build, remarkably stable across workload shape.** This is
the §5.1 decision-gate input for C7.3: a front-end-only cache cannot fix build latency, because
two-thirds of it is not front-end work.

**How the split is obtained, and how it must not be.** The first harness timed `stark build` against
`stark build --emit-rust`, assuming the latter stopped before Cargo. It does not — `--emit-rust`
only additionally writes the generated file — so both timings measured the same work and the
"host share" came out as noise, once at **−0.3%**. A negative share is the useful kind of wrong: it
is impossible, so it exposed the method instead of quietly biasing a number. The split is now
measured by building with `--keep-generated`, then `cargo clean`-ing the generated crate and timing
`cargo build` alone.

## 9. Reproducibility — measured, two distinct absolute paths

`scripts/c7-baseline.py --reproduce`, each workload built twice from temporary roots of deliberately
different names and lengths. All seven agree:

| artefact class | verdict |
| --- | --- |
| generated Rust (`main.rs`) | **BYTE-REPRODUCIBLE** |
| generated `Cargo.toml` | **BYTE-REPRODUCIBLE** on one machine; see 9.2 |
| `stark.lock` | **BYTE-REPRODUCIBLE** |
| linked executable | **NOT-YET-REPRODUCIBLE** — see 9.1 |

### 9.1 Why the executable differs, exactly

Each binary embeds **40 strings containing its own absolute build directory**, and the size
difference tracks the path-length difference (484 384 vs 484 784 bytes for a 400-byte-longer root).
This is `rustc` embedding build paths in debug info and panic locations, not a STARK-level leak.
`--remap-path-prefix` is the standard remedy and belongs to C7.2.

### 9.2 PID and thread-ID temporary paths do NOT leak

Directly tested: **zero** `stark_…{pid}_ThreadId(n)`-shaped strings in either binary. Temp-directory
naming affects the intermediate build location only, never the artefact.

### 9.3 The absolute runtime dependency path

`generated_cargo_toml` writes an absolute path to the runtime crate. It came out byte-identical
between the two checkouts because it points at the fixed compiler installation, not into the
workload copy. So it is constant per machine and **varies across machines or install locations**:
the correct classification is `SEMANTICALLY-REPRODUCIBLE`, and a cross-machine byte claim would be
false. This is the answer to "the exact reproducibility effect of the absolute runtime path" — it
does not affect same-machine reproduction at all, and it is fatal to a cross-machine byte claim.

## 10. C7/C8 lease map (from the real tree)

**C7-owned, no lease needed:** `src/native_build.rs`, `src/native_toolchain.rs`, `src/target.rs`,
`src/backend/**`, `src/bin/stark.rs`, `benchmarks/c7-workloads/**`, `scripts/c7-baseline.py`,
`docs/compiler/evidence/c7/**`, `STARKLANG/docs/compiler/work-packages/WP-C7*.md`.

**C8-owned, do not touch:** `src/lsp/**` (`mod.rs`, `position.rs`, `protocol.rs`, …). No editor
extension directory exists in this repository yet.

**Shared — lease required before editing:** `COMPILER-STATE.md`,
`STARKLANG/docs/compiler/COMPILER-ROADMAP.md`, `starkc/Cargo.toml`, `.github/workflows/ci.yml`,
`src/analysis.rs`, `src/package.rs`, `src/diag.rs`, `src/main.rs`.

`src/analysis.rs` deserves emphasis: `native_build.rs` calls `analyze_project`, and the directive
forbids rewriting the shared project-analysis API for build convenience. C7 consumes it and does not
reshape it.

## 11. Unresolved C7.1 requirements

Carried forward as explicit blockers rather than assumptions:

1. **The release profile inherits `panic = "unwind"`.** `[profile.release]` must be written with
   `panic = "abort"` and every §6.6 setting recorded rather than inherited.
2. **Output paths are not target-aware and hard-code `debug`.** `target/stark/debug/` must become
   profile- and target-parameterised before `--target` can exist without collisions.
3. **Target selection is absent from `native_build.rs`.** It must be threaded through the build model
   — package resolution, generated crate, cache key, output path — not appended to the final Cargo
   command, which would leave artefact paths and any future cache key blind to it.
