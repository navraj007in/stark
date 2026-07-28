# WP-C7.0 — Baseline: the build path as it actually is

**Status:** `PARTIAL` — §1 build-path inventory complete. Workload freeze, stage-timing harness and
clean-build experiments follow in the next commits.
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

## 5. Next in C7.0

1. freeze the workload set (§2.2);
2. build the stage-timing harness separating STARK time from `rustc` time (§2.3);
3. run the two-path clean-build comparison (§4.3);
4. record the C7/C8 lease map.

No optimisation flag is added until those exist. The directive is explicit that C7 is not
permission to optimise first and measure later, and §3.3 above is the concrete reason: the release
profile has to be *designed*, not switched on.
