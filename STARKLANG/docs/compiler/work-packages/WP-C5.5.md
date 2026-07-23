# WP-C5.5 — Debug Build Experience and Parallel Pre-Integration Plan

**Status:** CLOSED 2026-07-23 — CD-076
**Parallel status:** `C5.5-PRE` completed; `C5.5-INTEGRATION` completed after WP-C5.4
**Formal closure dependency:** satisfied by WP-C5.4d closure at `6e150b1`
**Prepared:** 2026-07-23  
**Repository baseline:** `0f39579d722b0851b7abe497df306222014f9730`  
**Intended parallel implementer:** Codex or Gemini while Claude implements WP-C5.4  
**Depends on now:** WP-C5.1, WP-C5.2, WP-C5.3 — closed  
**Depends on for final integration:** WP-C5.4 — multi-package linkage, concrete generics, and function values  
**Next:** WP-C5.6 — Gate C5 qualification
**Authority:** `COMPILER-CHARTER.md`, `COMPILER-STATE.md`, `COMPILER-ROADMAP.md`, `WP-C5-ENTRY.md`, this document

---

## 0. Executive directive

Implement WP-C5.5 in two stages:

```text
C5.5-PRE — may run in parallel with C5.4
    command parsing
    native build orchestration
    toolchain discovery
    stable output placement
    generated-source retention
    user-facing diagnostic classification
    single-package C5.2/C5.3 CLI proofs

C5.5-INTEGRATION — begins only after C5.4d closes
    rebase onto the completed linkage backend
    route the frozen three-package workspace through `stark build`
    integrate linkage-specific diagnostics
    finalize installed-runtime discovery
    run final C5.5 closure evidence
```

The parallel track must **not** duplicate or pre-empt C5.4.

C5.4 owns:

- canonical callable linkage validation;
- concrete-instance completeness;
- package-aware symbol proof;
- function-value representation;
- indirect calls;
- the frozen three-package reference workspace.

C5.5 owns:

- the user command;
- package-to-native orchestration;
- toolchain and runtime discovery;
- final artifact placement;
- generated-source retention and verbose output;
- user-facing build diagnostics;
- installation/offline behaviour.

The integration boundary is:

> C5.5 supplies one successfully verified `MirProgram` to the generated-Rust backend and consumes one `NativeArtifact`. It does not inspect, repair, discover, deduplicate, or reinterpret linked instances.

---

## 1. Why parallel execution is allowed

The Gate C5 roadmap places C5.4 before C5.5 because the final C5.5 proof uses the C5.4 reference workspace. That does not require all C5.5 implementation to wait.

At the baseline:

- `stark build` is already listed in the CLI usage;
- the CLI currently treats `build` like `check`, stopping after semantic analysis and printing `<package>: OK`;
- `analysis::analyze_project` is already the shared semantic project entry;
- `mir::lower::lower_program` already accepts multi-file/multi-package HIR;
- `mir::verify::verify_program` already produces the verified-program capability;
- `backend::generated_rust::emit_native_debug` already accepts only `VerifiedMirProgram`;
- the backend already emits a generated Cargo crate and returns `NativeArtifact`;
- native tests already prove parse/resolve/typecheck/lower/verify/emit/build/run.

Therefore C5.5-PRE can wire the existing proven pipeline into the user command using current C5.2/C5.3 programs while C5.4 extends what the backend can link.

Parallel execution is valid only while the boundary remains strict.

---

## 2. Current-state facts that the implementer must verify

Before editing, confirm these baseline facts and update this document if the branch has moved.

### 2.1 Current CLI behaviour

`starkc/src/bin/stark.rs` currently:

- advertises:
  - `stark check`;
  - `stark build`;
  - `stark run`;
- accepts `--locked` and `--offline` for check/build/run;
- loads the package graph;
- calls `analysis::analyze_project`;
- renders semantic diagnostics;
- returns success for both `check` and `build` by printing `<package>: OK`;
- runs `stark run` through the HIR interpreter.

C5.5 must replace only the placeholder `build` behaviour.

Do **not** change `stark run` from interpreter execution to native execution in C5.5. That is a separate product/CLI decision and is not required by the Gate C5 build claim.

### 2.2 Current native backend API

Expected API:

```rust
pub fn emit_native_debug(
    program: &VerifiedMirProgram<'_>,
    options: &NativeBuildOptions,
) -> Result<NativeArtifact, BackendDiagnostic>
```

Expected artifact fields:

```rust
pub struct NativeArtifact {
    pub binary_path: PathBuf,
    pub build_dir: PathBuf,
}
```

C5.5-PRE must call this API rather than calling `emit_program`, writing Cargo files, or invoking Cargo independently.

### 2.3 Current backend implementation limitations relevant to C5.5

At baseline, generated-Rust `build.rs`:

- discovers `rustc` with literal command name `rustc`;
- invokes Cargo with literal command name `cargo`;
- locates `stark-runtime` through `env!("CARGO_MANIFEST_DIR")/stark-runtime`;
- always uses Cargo `--offline`;
- writes under `<options.target_dir>/debug/<build-key>/`;
- returns the inner generated-crate binary path;
- returns raw `BackendDiagnostic::Io` and `BackendDiagnostic::BuildFailed`;
- does not place a stable user-facing executable;
- does not implement a user installation layout.

C5.5-PRE must work with this API without editing C5.4-owned backend files. The final toolchain/runtime-path API correction belongs to C5.5-INTEGRATION after C5.4 merges.

### 2.4 Current lowering pipeline

The user-facing native build path is conceptually:

```rust
let graph = PackageGraph::load_from_root_with_modes(...)?;
let analysis = analyze_project(ProjectInput::package(graph), LanguageOptions::CORE);

if analysis.has_errors() {
    render diagnostics;
    fail;
}

let hir = analysis.hir.as_ref().expect(...);
let tables = analysis.type_tables.as_ref().expect(...);

let mir = lower_program(hir, tables, analysis.root_file.clone())?;
let verified = verify_program(&mir)?;
let artifact = emit_native_debug(&verified, &native_options)?;
```

Do not create a second parser/resolver/typechecker pipeline in the build command.

---

## 3. Worktree and branch coordination

Parallel implementation creates a real merge-risk. Use explicit ownership.

Recommended branches/worktrees:

```text
work/c5.4       — Claude
work/c5.5-pre   — Codex or Gemini
```

Both should start from the same approved baseline or from a common plan-only commit.

### 3.1 Files owned by C5.4 during parallel work

C5.5-PRE must not edit these unless the C5.4 implementer explicitly transfers ownership:

```text
STARKLANG/docs/compiler/work-packages/WP-C5.4.md

starkc/src/backend/generated_rust/
  linkage.rs
  emit_program.rs
  emit_types.rs
  emit_bodies.rs
  emit_places.rs
  emit_projections.rs
  mangle.rs
  build.rs
  mod.rs

starkc/tests/
  native_c5_4_*.rs
  three_engine_differential.rs
  fixtures/c5-native-workspace/
```

C5.4 may not edit every listed file, but C5.5-PRE must assume it might.

### 3.2 Files owned by C5.5-PRE

Preferred C5.5-PRE ownership:

```text
STARKLANG/docs/compiler/work-packages/WP-C5.5.md

starkc/src/bin/stark.rs
starkc/src/native_build.rs          # new preferred orchestration module
starkc/src/native_toolchain.rs      # new preferred toolchain discovery module
starkc/src/lib.rs                   # one module export only

starkc/tests/native_build_cli.rs
starkc/tests/fixtures/c5-build-single/
```

Names may vary, but responsibilities must remain outside `backend/generated_rust/` during PRE.

### 3.3 Shared files

Avoid editing during the parallel phase:

```text
COMPILER-STATE.md
STARKLANG/docs/compiler/COMPILER-ROADMAP.md
STARKLANG/docs/compiler/work-packages/WP-C5-ENTRY.md
Cargo.lock
```

`Cargo.lock` may change only if genuinely required. C5.5 should add no dependency, so it should normally remain untouched.

Record PRE status and evidence in this file. Update `COMPILER-STATE.md` once, during integration after C5.4 is merged, to avoid competing state histories.

### 3.4 Required merge order

```text
1. Complete and merge C5.4.
2. Rebase C5.5-PRE onto completed C5.4.
3. Resolve only mechanical conflicts.
4. Perform C5.5-INTEGRATION.
5. Update COMPILER-STATE.md.
6. Close C5.5.
```

Do not merge a final C5.5 closure claim before C5.4d.

---

## 4. Purpose and claims

### 4.1 C5.5-PRE claim

C5.5-PRE may be marked complete when:

> A user can run `stark build` on an already-supported single-package C5.2/C5.3 program, the command uses the shared project analysis and verified-MIR pipeline, a native executable is placed at a stable documented path, toolchain and backend failures are classified in STARK-facing terms, and the CLI does not contain linkage or code-generation semantics.

### 4.2 Final C5.5 claim

WP-C5.5 may close only when:

> A user can run `stark build` in the frozen C5.4 three-package workspace without invoking Cargo manually, receive a standalone executable at a stable output path, locate retained generated source when requested, and receive clear STARK-facing diagnostics for source errors, unsupported native features, missing toolchains, backend defects, runtime installation failures, and environmental failures.

### 4.3 Consequence of false closure

A false closure could produce:

- `stark build` that only checks;
- a command that bypasses MIR verification;
- a second implementation of package linkage in the CLI;
- an executable that is built but cannot be found;
- raw rustc errors shown as user type errors;
- source errors misclassified as backend defects;
- a command that works only inside the compiler source checkout;
- accidental online dependency resolution;
- success output before the executable is safely installed;
- an old output binary left in place after a failed build and mistaken for the new result.

These are closure blockers.

---

## 5. Non-goals

C5.5 does **not** implement:

- C5.4 linkage, function values, or concrete generic-instance logic;
- a new project-analysis pipeline;
- native `stark run`;
- release builds;
- `--release`;
- arbitrary target selection;
- `--target`;
- Windows support;
- incremental compilation;
- a persistent build cache;
- parallel compiler jobs;
- watch mode;
- package publishing;
- dependency fetching;
- registry access;
- dynamic linking;
- external build scripts;
- user-configurable linker flags;
- optimisation;
- executable signing;
- installation packaging for every operating system;
- a stable public native ABI.

Do not add convenient adjacent flags.

---

## 6. User command contract

### 6.1 Supported syntax

C5.5 defines:

```text
stark build [--locked] [--offline] [--keep-generated] [--emit-rust] [--verbose]
```

Existing behaviour:

- `--locked` — package resolution must use the lock state as already defined;
- `--offline` — package resolution must not access the network.

New behaviour:

- `--keep-generated` — retain the full generated Cargo crate after a successful build and print its path;
- `--emit-rust` — retain generated output and print the generated `main.rs` path; implies `--keep-generated`;
- `--verbose` — print stage, toolchain, generated directory, backend artifact, and final artifact information.

No positional path argument is admitted in C5.5. Build the package containing the current working directory, using existing package-root discovery.

### 6.2 Parsing rules

- flags may appear in any order after `build`;
- repeated boolean flags are idempotent;
- unknown flags produce usage and exit code 2;
- unexpected positional arguments produce usage and exit code 2;
- `--emit-rust` sets `keep_generated = true`;
- `--help` remains global;
- `stark build --help` may either print command help or global help, but behaviour must be tested and documented;
- do not use an external CLI dependency.

### 6.3 Exit codes

```text
0 — build succeeded and final artifact exists
1 — source, package, lowering, verification, toolchain, backend, runtime, or filesystem failure
2 — invalid command-line usage
```

Do not return success because semantic checking succeeded if native build or final placement failed.

### 6.4 Success output

Default output should be one concise line:

```text
Built <package-name> [debug] -> <path>
```

The printed path must be the stable user-facing executable, not the hashed generated-crate binary.

With `--keep-generated`:

```text
Generated crate -> <path>
```

With `--emit-rust`:

```text
Generated Rust -> <path>/src/main.rs
```

With `--verbose`, print the stage details before the final success line.

Do not print Cargo’s full normal output unless verbose mode is explicitly intended to expose it.

---

## 7. Thin CLI, library-owned orchestration

Do not place the complete build pipeline in `src/bin/stark.rs`.

Create a library module, preferred:

```text
starkc/src/native_build.rs
```

### 7.1 Suggested public surface

```rust
#[derive(Clone, Debug, Default)]
pub struct BuildCommandOptions {
    pub locked: bool,
    pub offline: bool,
    pub keep_generated: bool,
    pub emit_rust: bool,
    pub verbose: bool,
}

#[derive(Clone, Debug)]
pub struct BuildCommandResult {
    pub package_name: String,
    pub artifact_path: PathBuf,
    pub generated_dir: Option<PathBuf>,
    pub generated_rust: Option<PathBuf>,
    pub toolchain: ToolchainInfo,
}

pub fn build_current_package(
    current_dir: &Path,
    options: &BuildCommandOptions,
) -> Result<BuildCommandResult, BuildCommandError>;
```

The exact names may differ. Preserve the responsibility split:

- binary:
  - parse arguments;
  - call library;
  - render result/error;
  - choose exit code;
- library:
  - find package;
  - load package graph;
  - analyze;
  - lower;
  - verify;
  - invoke backend;
  - place artifact;
  - cleanup;
  - return structured outcome.

### 7.2 No process termination in library code

The library module must not:

- call `std::process::exit`;
- write normal output directly;
- depend on terminal colour;
- read `std::env::args`.

Return structured values/errors so tests and future tooling can reuse the pipeline.

### 7.3 Shared semantic entry

Use:

```rust
analysis::analyze_project(ProjectInput::package(graph), LanguageOptions::CORE)
```

Do not call `parse_package_graph`, `resolve`, and `typecheck` independently.

### 7.4 Verified-MIR boundary

The backend call must receive only `VerifiedMirProgram`.

No escape hatch accepting arbitrary `MirProgram` may be added for CLI convenience.

---

## 8. Build stages

Implement and report these stages.

```text
1. command option validation
2. current-directory resolution
3. package-root discovery
4. package-graph loading
5. shared project analysis
6. source diagnostic rendering boundary
7. typed HIR availability check
8. MIR lowering
9. MIR verification
10. native toolchain preflight
11. native-backend invocation
12. backend artifact existence validation
13. stable artifact installation
14. optional generated-output retention
15. successful result
```

### 8.1 Stage ordering rule

Do not probe rustc/Cargo before source analysis succeeds.

A source syntax/type/borrow error should not be hidden by “rustc missing.” The command is a compiler command; source diagnostics come first.

Recommended order:

```text
package and source analysis
    then MIR lowering/verification
    then external toolchain preflight
    then backend build
```

### 8.2 No stale success

Before invoking the native build:

- determine the final artifact path;
- do not delete the previous successful artifact yet;
- build in the hashed backend directory;
- install the new artifact atomically or as close to atomically as the supported platforms permit;
- only after successful installation may the old artifact be replaced.

On failure, never print or imply that the existing artifact is the result of the failed build.

Optionally record a message that an older artifact remains, but do not run it automatically.

---

## 9. Stable artifact placement

### 9.1 Output directory

C5.5 debug output:

```text
<package-root>/target/stark/debug/<binary-name>
```

The backend’s hashed directory remains internal:

```text
<package-root>/target/stark/debug/<build-key>/
```

### 9.2 Binary name

Use the resolved root package name as the user-facing binary name.

Rules:

- package name comes from the loaded package graph, not source text;
- validate it as a safe single filesystem component;
- reject path separators, `.`/`..`, empty names, and platform-invalid names;
- preserve the executable suffix from the backend artifact where required;
- do not use canonical function symbols as the file name;
- do not use the generated crate’s internal `stark_program` name as the final user name.

For C5 Tier-1 targets, macOS/Linux normally have no suffix. Use the actual backend artifact suffix rather than hardcoding platform assumptions.

### 9.3 Installation algorithm

Conceptually:

```text
source = NativeArtifact.binary_path
temp   = final_path + ".tmp-<process-local>"
final  = target/stark/debug/<package>

copy source -> temp
verify temp exists and is a regular file
preserve executable permissions
rename temp -> final
verify final exists
```

No random dependency is required. A process ID plus a deterministic local counter is sufficient for a temporary sibling path; the name is not a reproducibility claim.

Clean a stale temp file before use.

### 9.4 Final artifact metadata

C5.5 need not create a new public manifest format.

The generated backend already writes `build.json`. In verbose mode, print its path if retained.

Do not duplicate backend version/layout/build-key metadata into a second independently maintained JSON schema unless the Gate C5 plan explicitly requires it.

---

## 10. Generated output retention

### 10.1 Default successful build

After final artifact installation:

- remove the hashed generated crate unless retention was requested;
- leave the stable final binary;
- do not implement a cache in C5.5.

C5 does not require incremental compilation.

### 10.2 `--keep-generated`

Retain:

```text
<target>/stark/debug/<build-key>/
```

Print that exact path.

### 10.3 `--emit-rust`

- implies `--keep-generated`;
- verify `<build_dir>/src/main.rs` exists;
- print its path;
- do not print the entire generated source to stdout;
- do not create a second copy unless a later decision requires one.

### 10.4 Failed backend build

Preserve generated files on backend/rustc failure whenever they exist.

During C5.5-PRE, the existing backend error does not return the build directory on failure. Do not scan by modification time or guess which directory failed.

Acceptable PRE behaviour:

- preserve the target build root;
- print that generated files, when created, remain under `<target>/stark/debug/`;
- record the missing exact-failure-directory field as an integration item.

Final recommended correction after C5.4 merge:

```rust
BackendDiagnostic::BuildFailed {
    summary: String,
    stdout: String,
    stderr: String,
    build_dir: PathBuf,
    command: Vec<String>,
    status: Option<i32>,
}
```

This is a backend diagnostic-structure improvement, not a linkage change. Implement only during C5.5-INTEGRATION to avoid file conflict.

---

## 11. Toolchain discovery

Create a separate preferred module:

```text
starkc/src/native_toolchain.rs
```

### 11.1 Toolchain structure

```rust
#[derive(Clone, Debug)]
pub struct ToolchainInfo {
    pub rustc: PathBuf,
    pub cargo: PathBuf,
    pub rustc_release: String,
    pub cargo_release: String,
    pub host_triple: String,
    pub sysroot: Option<PathBuf>,
    pub runtime_crate: PathBuf,
}
```

### 11.2 Command resolution precedence

For rustc:

```text
STARK_RUSTC
RUSTC
rustc from PATH
```

For Cargo:

```text
STARK_CARGO
CARGO
cargo from PATH
```

For runtime source:

```text
STARK_RUNTIME_DIR
installed-toolchain layout
source-checkout fallback
```

Environment overrides exist for deterministic testing and controlled installations. They are not language semantics.

### 11.3 Probes

Run:

```text
rustc -vV
cargo --version
```

Parse:

- rustc release;
- host triple;
- optional sysroot using `rustc --print sysroot`;
- Cargo release.

Failures must identify which command could not run.

### 11.4 Minimum version

Baseline compiler manifest declares:

```text
rust-version = "1.85"
```

Recommended C5.5 mechanism:

- keep the minimum in one named constant or derive it from package metadata;
- initial value: `1.85.0`;
- compare numeric major/minor/patch components, not lexicographic strings;
- accept stable version suffixes by parsing the numeric prefix;
- record actual qualification on the C5 primary toolchain and stable CI toolchain.

This is a toolchain policy decision, not Core semantics. Before formal C5.5 closure, record owner approval of the floor. C5.5-PRE may implement the mechanism and the baseline recommendation.

### 11.5 Cargo/rustc coherence

Do not attempt to prove perfect rustup toolchain identity.

At minimum:

- both commands must exist;
- rustc must meet the floor;
- Cargo must successfully parse its version;
- verbose output records both resolved paths and versions;
- backend compilation is the final compatibility proof.

### 11.6 Installed runtime layout

Recommended installed layout:

```text
<toolchain-root>/
  bin/
    stark
    starkc
  lib/
    stark/
      stark-runtime/
        Cargo.toml
        src/
```

Runtime discovery:

1. `STARK_RUNTIME_DIR`, if set;
2. relative to current executable:
   `../lib/stark/stark-runtime`;
3. development fallback:
   `env!("CARGO_MANIFEST_DIR")/stark-runtime`.

Validate:

- directory exists;
- `Cargo.toml` exists;
- `src/lib.rs` exists;
- path is canonicalizable or can be used as an absolute path.

Do not silently use a different runtime version from another source tree.

### 11.7 Parallel restriction

During C5.5-PRE:

- implement discovery and tests in the new module;
- preflight before backend invocation;
- do not modify `NativeBuildOptions` or generated backend commands.

During C5.5-INTEGRATION, after C5.4 merge:

extend the backend options so the discovered commands/runtime path are passed explicitly.

Recommended final shape:

```rust
pub struct NativeBuildOptions {
    pub target_dir: PathBuf,
    pub target_contract: String,
    pub rustc: PathBuf,
    pub cargo: PathBuf,
    pub runtime_crate: PathBuf,
}
```

A compatibility constructor/default may preserve existing tests.

Then:

- `query_rustc_verbose` uses `options.rustc`;
- Cargo invocation uses `options.cargo`;
- generated manifest uses `options.runtime_crate`;
- no literal source-checkout path remains in the production CLI path.

Do not implement this API change on the PRE branch while C5.4 owns backend files.

---

## 12. Offline behaviour

### 12.1 Package graph

Continue passing existing `locked` and `offline` modes into:

```rust
PackageGraph::load_from_root_with_modes
```

Do not reinterpret their package-manager semantics.

### 12.2 Generated Cargo build

Generated Cargo must always use:

```text
cargo build --offline
```

because the generated crate is allowed only local/pinned toolchain dependencies.

User `--offline` controls package resolution. Backend Cargo remains offline even without the flag.

### 12.3 Proof

Add a test using:

- an empty temporary `CARGO_HOME`;
- the local `stark-runtime` path;
- a supported single-package fixture.

The native build must succeed without registry index/network dependency.

Do not make the test rely on an actual network denial that could hang. Empty `CARGO_HOME` plus `--offline` is the deterministic proof.

### 12.4 No dependency expansion

C5.5 should add no crate dependency for:

- CLI parsing;
- version parsing;
- temporary directories;
- JSON;
- process execution;
- path copying.

Use the standard library and existing project helpers.

---

## 13. Diagnostic model

Create a structured build error type in the library.

Suggested shape:

```rust
pub enum BuildCommandError {
    CurrentDirectory(String),
    Package(String),
    Analysis {
        diagnostics_already_rendered: bool,
    },
    Lowering {
        message: String,
        source: Option<BuildSourceLocation>,
    },
    MirVerification(String),
    ToolchainMissing {
        tool: &'static str,
        attempted: PathBuf,
        detail: String,
    },
    ToolchainTooOld {
        found: String,
        required: String,
    },
    RuntimeMissing {
        attempted: Vec<PathBuf>,
    },
    UnsupportedNative(String),
    BackendBuild {
        summary: String,
        detail: Option<String>,
        build_dir: Option<PathBuf>,
    },
    ArtifactMissing(PathBuf),
    ArtifactInstall {
        from: PathBuf,
        to: PathBuf,
        detail: String,
    },
    Io {
        action: String,
        path: Option<PathBuf>,
        detail: String,
    },
}
```

Exact names may vary. Do not reduce every failure to one string before the CLI renders it.

### 13.1 Source diagnostics

Parse/name/type/borrow errors:

- render the existing `DiagnosticBatch`;
- return failure;
- do not wrap them as “native backend failed.”

### 13.2 MIR lowering

`LowerError` currently carries a message and span but not a complete standalone file identity in its public error value.

C5.5-PRE must not lie by attributing every package lowering error to the root file.

Use one of:

- an existing source-correct helper if available on the implementation branch;
- a non-fabricated message without a false file location;
- a narrowly scoped structured error improvement after coordination.

Do not redesign MIR merely for diagnostics.

### 13.3 MIR verification

A verified-MIR failure is an internal compiler defect, not a user language error.

Render conceptually:

```text
error: internal compiler error: generated MIR failed verification
```

Include detailed verifier output in verbose mode.

Do not continue to the backend.

### 13.4 Backend unsupported

Map:

```rust
BackendDiagnostic::Unsupported(message)
```

to:

```text
error: native build does not yet support this program: <message>
```

This is not a Rust compiler error and should occur before rustc for defined unsupported boundaries.

### 13.5 Missing toolchain

Example:

```text
error: Rust toolchain not found
help: install a supported Rust toolchain or set STARK_RUSTC and STARK_CARGO
```

Name the exact attempted executable in verbose mode.

### 13.6 Backend/rustc failure

Default mode:

```text
error: the STARK native backend generated a crate that Cargo could not build
```

This must not be shown as a user type error.

Verbose mode includes:

- command;
- exit status;
- generated path;
- stdout/stderr;
- rustc/Cargo versions.

### 13.7 Runtime installation failure

Example:

```text
error: STARK native runtime installation is missing
help: expected stark-runtime under <path> or set STARK_RUNTIME_DIR
```

Do not let Cargo report a missing path dependency as the first user-visible diagnosis.

### 13.8 Diagnostic codes

Before adding new `E` codes:

- inspect the current diagnostic catalogue/ranges;
- ensure no collision;
- add the code to the authoritative catalogue and tests.

C5.5-PRE may use stable textual categories without inventing codes. Formal closure should state whether build-environment failures are coded compiler diagnostics or CLI errors.

---

## 14. Verbose mode

With `--verbose`, print deterministic stage labels:

```text
[stark build] package root: ...
[stark build] package: ...
[stark build] analysis: complete
[stark build] MIR bodies: <count>
[stark build] MIR verification: complete
[stark build] rustc: <path> (<version>)
[stark build] cargo: <path> (<version>)
[stark build] host: <triple>
[stark build] runtime: <path>
[stark build] generated crate: <path>
[stark build] backend binary: <path>
[stark build] final artifact: <path>
```

Do not print:

- absolute paths in non-verbose success output except the final artifact path;
- source contents;
- environment secrets;
- entire package manifests;
- raw backend output on success.

The exact count of MIR bodies is observational build information, not semantic output.

---

## 15. C5.5-PRE work sequence

### 15.1 PRE-a — Command parsing and thin CLI

#### Deliver

- separate build option parser;
- new flags;
- usage text;
- `build` routed separately from `check`;
- `run` unchanged;
- usage/exit-code tests.

#### Exit

- `stark build` no longer returns semantic-check success without invoking a native-build driver;
- invalid flags return 2;
- `--emit-rust` implies keep;
- no backend or linkage logic is in argument parsing.

### 15.2 PRE-b — Toolchain and runtime preflight module

#### Deliver

- command resolution;
- rustc/Cargo probes;
- version parser;
- recommended version-floor mechanism;
- runtime discovery;
- structured errors;
- environment-override tests.

#### Exit

- missing rustc is diagnosed before backend invocation;
- missing Cargo is diagnosed;
- missing runtime is diagnosed;
- verbose metadata is available as structured data;
- no generated-backend file was edited.

### 15.3 PRE-c — Native build driver

#### Deliver

- package root/graph loading;
- shared project analysis;
- MIR lowering;
- MIR verification;
- backend call through `emit_native_debug`;
- stable artifact placement;
- cleanup/retention;
- structured result/error.

#### Exit

- an existing supported C5.2/C5.3 fixture builds from the command;
- final artifact exists at the stable path;
- running it produces the expected result;
- the build driver contains no symbol/linkage/function-value code.

### 15.4 PRE-d — CLI diagnostics and evidence

#### Deliver

- source-error path;
- unsupported-native path;
- toolchain failure paths;
- backend-failure path;
- output-path tests;
- retention tests;
- offline test;
- repeated-build replacement test;
- full focused validation.

#### PRE outcome

Record:

```text
C5.5-PRE-COMPLETE
```

This does **not** mean WP-C5.5 is closed.

---

## 16. C5.5-INTEGRATION after C5.4

Begin only when C5.4d is formally closed and merged.

### 16.1 Rebase and contract audit

Verify:

- backend entry API;
- `NativeArtifact` fields;
- C5.4 linkage preflight location;
- final three-package fixture path;
- whether C5.4 changed build-key inputs;
- whether new backend diagnostics exist;
- whether C5.4 touched current CLI assumptions.

Update this document’s inventory.

### 16.2 Explicit toolchain/runtime options

Make the bounded backend options change described in Section 11.7.

Coordinate modifications to:

```text
backend/generated_rust/mod.rs
backend/generated_rust/build.rs
```

Do not alter C5.4 linkage semantics.

### 16.3 Failed-build structured detail

Return exact failed build directory and process metadata.

Preserve generated source and print its path.

### 16.4 Frozen workspace through CLI

From the frozen C5.4 fixture:

```bash
stark build --locked --offline
```

Prove:

- package discovery;
- analysis;
- multi-package lowering;
- MIR verification;
- C5.4 linkage validation;
- native Cargo build;
- stable artifact placement;
- executable run;
- HIR/MIR/native expected outcome.

### 16.5 Linkage diagnostic integration

C5.5 must display C5.4 pre-rustc failures without reimplementing them.

Examples:

- missing concrete instance;
- duplicate canonical symbol;
- generated-name collision;
- unsupported function-value signature.

The CLI maps backend diagnostic category to text. It does not inspect the MIR graph itself.

### 16.6 Installed-runtime proof

Test the recommended toolchain layout in a temporary directory:

```text
tmp-toolchain/
  bin/stark
  lib/stark/stark-runtime/
```

The test may invoke library discovery helpers rather than physically self-installing the running test binary if that is more deterministic.

Prove the runtime path is not dependent on the repository’s source checkout.

---

## 17. Test architecture

### 17.1 Unit tests

Test:

- build flag parser;
- `--emit-rust` implication;
- version parser:
  - `1.85.0`;
  - `1.93.0`;
  - stable suffix;
  - malformed version;
- safe binary-name validation;
- executable suffix preservation;
- final output-path calculation;
- toolchain command precedence;
- runtime-path precedence;
- error rendering;
- cleanup decisions.

### 17.2 CLI integration tests

Use Cargo’s integration-test binary path where available:

```rust
env!("CARGO_BIN_EXE_stark")
```

Create deterministic temporary package fixtures without a new dependency.

Required PRE tests:

1. `stark build` succeeds for a supported scalar program;
2. aggregate/enum program succeeds;
3. source type error fails before rustc probe;
4. unknown flag exits 2;
5. stable final path is printed;
6. final executable runs;
7. `--keep-generated` retains and prints generated crate;
8. `--emit-rust` retains and prints `main.rs`;
9. default successful build removes generated crate;
10. missing rustc override produces toolchain diagnostic;
11. missing Cargo override produces toolchain diagnostic;
12. missing runtime override produces runtime diagnostic;
13. repeated successful build replaces the final artifact;
14. failed new build does not claim the previous artifact as new;
15. empty `CARGO_HOME` offline build succeeds.

### 17.3 Final integration tests

After C5.4:

1. frozen three-package workspace builds;
2. cross-package generic path executes;
3. cross-package function value executes;
4. final artifact path uses root package name;
5. verbose output reports concrete-body count without changing semantics;
6. linkage failure is mapped, not duplicated;
7. relocated workspace builds to the same logical output name;
8. installed-runtime layout works;
9. source trap provenance remains from the correct dependency file where the build/run harness exercises it.

### 17.4 Negative controls

- a fixture with a false in-program assertion must produce a failing executable when run, proving the test observes values;
- a source error must fail even when an old final binary exists;
- a missing runtime must fail before Cargo;
- an unsupported backend shape must fail before rustc;
- a fake successful backend path without a binary must produce `ArtifactMissing`;
- an unwritable target directory must fail without printing success.

### 17.5 No brittle tests

Do not assert full absolute temporary paths unless testing path selection.

Normalize path separators only where cross-platform support is claimed. C5 Tier-1 is macOS arm64 and Linux x64; Windows remains C6.

---

## 18. Parallel stop rules

C5.5-PRE must stop and record an integration note instead of editing C5.4-owned files when:

1. a desired CLI feature requires changing function-value representation;
2. package linkage information appears missing from MIR;
3. concrete body completeness must be repaired;
4. canonical symbols need changes;
5. build-key package identity requires C5.4 decisions;
6. `NativeArtifact` needs a linkage-specific field;
7. backend diagnostics need changes in a file currently modified by C5.4;
8. the frozen three-package fixture is required;
9. a test requires C5.4 indirect calls;
10. merge conflict would combine semantic edits rather than mechanical API changes.

Continue implementing independent PRE work. Do not block the entire track for an integration-only item.

---

## 19. General stop and escalation rules

Stop and escalate if implementation appears to require:

- a Core semantic change;
- a MIR shape/version change;
- a new stable ABI promise;
- online Cargo dependency resolution;
- source package visibility checks in the CLI;
- duplicate monomorphisation/linkage logic;
- unsafe code;
- a new crate dependency;
- changing `stark run` semantics;
- treating rustc as the user diagnostic engine;
- exposing release/target flags before C7;
- silently broadening the native subset.

A valid supported source program failing in rustc is a backend defect or missing deterministic boundary, not ordinary user error.

---

## 20. Validation cadence

### 20.1 During PRE

After bounded changes:

```bash
cargo test --test native_build_cli
cargo test native_build
cargo test native_toolchain
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Run existing native smoke tests affected by orchestration.

### 20.2 Observable build changes

Per CD-070, when changing:

- diagnostics;
- artifact paths;
- manifest values;
- generated retention;
- command output;
- build status;

run:

```bash
cargo test --workspace --all-targets --no-fail-fast
```

### 20.3 PRE close

Record:

- exact commit;
- test counts;
- ignored counts;
- host;
- rustc/Cargo versions;
- fixture names;
- target paths;
- known integration items.

### 20.4 Final close

Required:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --no-fail-fast
```

Plus:

- frozen C5.4 workspace build through CLI;
- final executable run;
- offline proof;
- runtime installation-layout proof;
- retained-source proof;
- missing-toolchain proof;
- failed-build classification proof;
- relocated-workspace proof;
- adversarial review.

Hosted CI is a C5.6 qualification item unless explicitly brought forward. Do not claim hosted evidence without a real workflow run.

---

## 21. Adversarial closure review

Before C5.5 closes, answer with evidence:

1. Can `stark build` still return success after only checking?
2. Can it bypass MIR verification?
3. Can source errors be hidden by a missing Rust toolchain?
4. Can an old executable remain and be mistaken for a failed new build?
5. Can the final path point into a temporary hashed crate?
6. Can a missing runtime reach Cargo as a raw path-dependency error?
7. Can a rustc generated-source failure be shown as a user type error?
8. Can `--emit-rust` delete the source before the user reads it?
9. Can default successful builds leak every generated crate indefinitely?
10. Can CLI code duplicate C5.4 linkage validation?
11. Can absolute workspace relocation alter the user-facing binary name?
12. Can a package name escape the target directory?
13. Can `STARK_RUSTC`, `STARK_CARGO`, or `STARK_RUNTIME_DIR` expose secrets in output?
14. Can an unsupported feature fail only in rustc?
15. Can the CLI accidentally switch `run` to a different execution engine?
16. Does Cargo attempt any network access?
17. Does the installed toolchain work outside the repository checkout?
18. Does the full no-fail-fast suite reveal stale expected paths/output?
19. Are all final closure claims based on the three-package reference workspace, not only a scalar fixture?
20. Did integration alter any C5.4 semantic decision?

Record every disposition.

---

## 22. C5.5-PRE completion checklist

C5.5-PRE may be marked complete when:

### CLI

- [ ] `stark build` has a distinct execution path from `check`.
- [ ] new flags parse correctly.
- [ ] invalid usage exits 2.
- [ ] `run` remains unchanged.
- [ ] binary main remains thin.

### Pipeline

- [ ] shared project analysis is used.
- [ ] source errors stop the build.
- [ ] MIR lowering runs.
- [ ] MIR verification runs.
- [ ] backend receives only `VerifiedMirProgram`.
- [ ] no linkage logic exists in C5.5 code.

### Toolchain

- [ ] rustc/Cargo paths are resolved.
- [ ] versions and host are parsed.
- [ ] minimum-version mechanism exists.
- [ ] runtime discovery exists.
- [ ] missing tools/runtime have structured errors.
- [ ] environment overrides are tested.

### Artifact

- [ ] stable output path is implemented.
- [ ] final name derives safely from root package name.
- [ ] binary permissions are preserved.
- [ ] successful installation is verified.
- [ ] default cleanup works.
- [ ] retention flags work.
- [ ] old artifacts are not misreported after failure.

### Evidence

- [ ] supported scalar fixture builds.
- [ ] supported aggregate/enum fixture builds.
- [ ] resulting executables run and are observed.
- [ ] offline test passes.
- [ ] focused tests pass.
- [ ] fmt/clippy pass.
- [ ] full no-fail-fast suite passes.
- [ ] integration blockers are listed.
- [ ] status is recorded as `C5.5-PRE-COMPLETE`, not `WP-C5.5 CLOSED`.

---

## 23. Final WP-C5.5 closure checklist

WP-C5.5 closes only when:

### Dependency

- [ ] C5.4d is formally closed and merged.
- [ ] C5.5-PRE is rebased onto the C5.4 closure head.
- [ ] no C5.4 semantic logic was duplicated.

### User command

- [ ] `stark build` builds the frozen three-package workspace.
- [ ] no manual Cargo invocation is required.
- [ ] final path is printed.
- [ ] executable exists and runs.
- [ ] `--keep-generated`, `--emit-rust`, and `--verbose` work.

### Toolchain/install

- [ ] explicit discovered rustc/Cargo paths reach the backend.
- [ ] installed runtime path reaches the generated manifest.
- [ ] source-checkout path is not required by the production CLI path.
- [ ] generated Cargo remains offline.
- [ ] missing and old toolchains fail clearly.

### Diagnostics

- [ ] source diagnostics remain source diagnostics.
- [ ] unsupported native features fail before rustc.
- [ ] MIR verification failure is internal.
- [ ] backend/rustc failure is correctly classified.
- [ ] exact failed generated directory is reported when available.
- [ ] runtime installation failure is preflighted.
- [ ] no raw rustc error is presented as an ordinary STARK type error.

### C5.4 integration

- [ ] cross-package direct call builds through CLI.
- [ ] cross-package generic call builds through CLI.
- [ ] function-value/indirect-call path builds through CLI.
- [ ] linkage failures are rendered from C5.4 diagnostics.
- [ ] root package binary name is stable after relocation.

### Evidence

- [ ] HIR/MIR/native expected outcome for the frozen workspace is unchanged.
- [ ] final executable is standalone within the C5 runtime/toolchain definition.
- [ ] offline proof passes.
- [ ] installed-runtime proof passes.
- [ ] relocation proof passes.
- [ ] full workspace no-fail-fast suite passes.
- [ ] fmt/clippy pass.
- [ ] adversarial dispositions are recorded.
- [ ] `COMPILER-STATE.md` is updated once.
- [ ] this document records exact counts, commit, targets, and toolchains.
- [ ] no known C5.5 user-experience blocker remains.

---

## 24. Expected repository changes

### Parallel PRE

```text
STARKLANG/docs/compiler/work-packages/WP-C5.5.md

starkc/src/bin/stark.rs
starkc/src/native_build.rs
starkc/src/native_toolchain.rs
starkc/src/lib.rs

starkc/tests/native_build_cli.rs
starkc/tests/fixtures/c5-build-single/
```

### Integration after C5.4

Potential bounded changes:

```text
starkc/src/backend/generated_rust/mod.rs
starkc/src/backend/generated_rust/build.rs
COMPILER-STATE.md
```

Do not modify other C5.4 backend modules unless a real integration defect requires it and the change is reviewed against WP-C5.4.

---

## 25. Codex/Gemini implementation protocol

The implementation agent must:

1. read `COMPILER-CHARTER.md`;
2. read current `COMPILER-STATE.md`;
3. read this file;
4. read only the relevant C5 entry-plan sections;
5. inspect the current CLI and backend public API;
6. verify the file-ownership matrix;
7. implement PRE-a through PRE-d in bounded commits;
8. add tests with each behaviour;
9. avoid all C5.4-owned files;
10. record integration-only needs instead of crossing the boundary;
11. run the no-fail-fast suite before PRE completion;
12. mark only `C5.5-PRE-COMPLETE`;
13. wait for C5.4 merge before integration;
14. rebase and perform the exact integration sequence;
15. never mark WP-C5.5 closed without the frozen three-package CLI proof.

Suggested PRE commits:

```text
C5.5-PRE-a: split native build command parsing from check
C5.5-PRE-b: add Rust toolchain and runtime discovery
C5.5-PRE-c: route verified MIR through native backend and place artifact
C5.5-PRE-d: add diagnostics, retention, offline and CLI evidence
```

Suggested integration commits:

```text
C5.5-INT-a: pass explicit toolchain/runtime paths into backend
C5.5-INT-b: build frozen C5.4 workspace through `stark build`
C5.5-INT-c: close WP-C5.5 with full evidence
```

Use the repository’s actual next decision/commit identifiers. Do not invent or reserve CD numbers without recording them in the authoritative state.

Each commit message/body should state:

- sub-package;
- scope;
- files intentionally not touched;
- tests added;
- defects found;
- deferred integration items;
- validation totals.

---

## 26. Handoff contract between parallel agents

Claude/C5.4 should provide at merge:

```text
- closure commit SHA;
- final backend entry signature;
- final NativeArtifact shape;
- frozen workspace path;
- command/test helper for building it;
- C5.4 linkage diagnostic variants;
- any build-key changes;
- any generated-directory changes;
- exact C5.4 supported/deferred boundaries.
```

Codex/Gemini/C5.5-PRE should provide:

```text
- PRE completion commit SHA;
- CLI command contract;
- build-driver public API;
- toolchain/runtime discovery API;
- stable output-path contract;
- structured build-error categories;
- focused test totals;
- exact backend API assumptions;
- integration-only change list.
```

The integration owner compares the two contracts before editing shared files.

---

## 27. Completion result

At C5.5-PRE completion:

> `stark build` is a real native build command for the already-supported C5.2/C5.3 subset, with verified-MIR enforcement, toolchain preflight, stable artifact placement, generated-source controls, and STARK-facing diagnostics. Multi-package C5.4 integration remains explicitly pending.

At final WP-C5.5 closure:

> A user can enter the frozen three-package STARK workspace, run `stark build`, and receive a runnable native debug executable at a stable path without invoking Cargo manually. The command preserves the compiler’s semantic pipeline, uses the completed C5.4 linker and function-value backend without duplicating them, operates offline with the installed STARK runtime, and reports source, toolchain, unsupported-feature, backend, and environmental failures through clear STARK-facing diagnostics.

C5.6 then qualifies the complete Gate C5 claim through the full supported snapshot subset, platform/CI evidence, and the final exit report.

---

## 28. C5.5-PRE implementation record (2026-07-23)

**Status: C5.5-PRE-COMPLETE. WP-C5.5 remains OPEN pending C5.4d closure and C5.5-INTEGRATION.**

Recorded at repository head `fef84d85c65be8e121a2da8c1b4114ad289ac367` while C5.4 changes were
present in the shared worktree. The PRE implementation did not edit any file under
`starkc/src/backend/generated_rust/` and did not alter C5.4 linkage, concrete-instance, symbol,
or function-value semantics.

### 28.1 Delivered PRE contract

- `stark build` has a distinct native-build path; `check` and interpreter-backed `run` retain
  their previous execution models.
- The binary parses `--locked`, `--offline`, `--keep-generated`, `--emit-rust`, and `--verbose`;
  `--emit-rust` implies retention and invalid flags/positionals exit 2.
- `native_build` owns package discovery, canonical project analysis, MIR lowering, mandatory MIR
  verification, backend invocation, stable artifact installation, cleanup, and structured
  outcomes/errors.
- `native_toolchain` implements `STARK_RUSTC`/`RUSTC`/PATH and
  `STARK_CARGO`/`CARGO`/PATH precedence, numeric Rust version-floor checking (`1.85.0`), host and
  sysroot probes, and runtime discovery through override, installed layout, then development
  fallback.
- Stable debug artifacts are installed at `<package-root>/target/stark/debug/<package-name>` by
  copying to a process-local sibling and renaming only after the backend artifact exists.
- Successful default builds remove the hashed generated crate. Retention and emitted-Rust modes
  return and print exact existing paths. Backend failure preserves the generated root and states
  the PRE limitation that the exact failed hashed directory is not yet available.
- Source diagnostics precede toolchain probes. MIR verification, unsupported-native, missing
  toolchain/runtime, backend build, missing artifact, installation, and filesystem failures have
  distinct STARK-facing categories.

### 28.2 PRE evidence

Host: `aarch64-apple-darwin`.

Toolchain:

```text
rustc 1.93.0 (254b59607 2026-01-19)
cargo 1.93.0 (083ac5135 2025-12-15)
```

Focused evidence:

```text
cargo test --test native_build_cli --no-fail-fast
  6 passed; 0 failed; 0 ignored

cargo test --lib native_
  3 passed; 0 failed; 0 ignored; 435 filtered out

cargo fmt --all -- --check
  passed

cargo clippy --workspace --all-targets --all-features -- -D warnings
  passed
```

The CLI suite observes a scalar final executable and an aggregate final executable, stable
package-derived output naming, repeated replacement, default cleanup, retained crate/source,
verbose stages, source-error-before-toolchain ordering, old-artifact non-claim, missing rustc,
missing Cargo, missing runtime, usage exit 2, and an empty-`CARGO_HOME` offline build.

Full closure evidence:

```text
cargo test --workspace --all-targets --no-fail-fast
  1067 passed; 0 failed; 2 ignored across 52 test-bearing binaries
```

### 28.3 Integration-only items (intentionally deferred)

After C5.4d closes and merges:

1. extend `NativeBuildOptions` with the discovered rustc, Cargo, and runtime paths and use them in
   generated-backend process/manifest construction;
2. enrich `BackendDiagnostic::BuildFailed` with exact build directory, command, status, stdout,
   and stderr;
3. route the frozen C5.4 three-package workspace through the CLI and run the direct-call,
   concrete-generic, and function-value/indirect-call proofs;
4. map final C5.4 linkage diagnostics without inspecting or repairing MIR in C5.5 code;
5. prove the installed runtime layout outside the source checkout and workspace relocation;
6. update `COMPILER-STATE.md` once and perform the final adversarial closure review.

---

## 29. C5.5-INTEGRATION implementation record (2026-07-23)

**Status: CLOSED by owner directive CD-076.** Implementation, adversarial review, and all required
C5.5 evidence are complete; WP-C5.6 owns final Gate C5 qualification.

Integration began from the completed C5.4 handoff at
`6e150b1` (`CD-075: close C5.4d and WP-C5.4`). The final C5.4 backend entry and
`NativeArtifact { binary_path, build_dir }` contract were unchanged. The frozen workspace is
`starkc/tests/fixtures/c5-native-workspace/`. The C5.5 integration implementation and its closure
evidence are committed at `2c96d99` (`C5.5: complete native build integration`), with reviewed
follow-ups at `e94e760` (reproducible backend-command evidence) and `496406c` (retained-artifact
lifecycle correction).

### 29.1 Delivered integration

- Added an explicit production backend entry accepting resolved rustc, Cargo, and runtime paths.
  The older direct-backend entry remains as a development/test compatibility wrapper; production
  backend construction contains no hidden source-checkout or literal-tool command assumption.
  The CLI's documented development runtime fallback remains last in the §11.2 precedence order.
- The selected rustc is used for backend version/host queries and is passed to Cargo through
  `RUSTC`; the selected Cargo executable performs the always-offline generated-crate build; the
  selected runtime path is written into the generated manifest.
- `BackendDiagnostic::BuildFailed` now carries a boxed structured record: summary, stdout,
  stderr, exact build directory, command, and optional exit status. Default CLI output classifies
  it as a native-backend build failure and prints the retained directory; verbose output includes
  all process evidence plus the preflighted rustc/Cargo paths and versions.
- The frozen relocated `app → logic → model` workspace builds through
  `stark build --locked --offline --emit-rust`, places the executable at
  `<relocated-app>/target/stark/debug/app`, retains generated Rust, and the executable exits 0.
  This consumes C5.4 linkage validation and concrete/function-value bodies without duplicating or
  inspecting linkage semantics in the CLI.
- Installed-layout discovery is proven from `bin/stark` to
  `lib/stark/stark-runtime`; a separate CLI proof supplies a relocated runtime through
  `STARK_RUNTIME_DIR`, verifies that exact canonical path reaches the generated Cargo manifest,
  uses an empty `CARGO_HOME`, and observes the selected Cargo wrapper receive both its version
  probe and offline build command.
- A deliberately failing Cargo wrapper proves exit-status/stderr classification and that the
  exact generated crate (including `src/main.rs`) remains available after failure.
- A normal unretained `stark build --verbose` returns no backend-artifact path after deleting the
  generated crate. The result models that path as `Option<PathBuf>` and prints it only while the
  generated crate is retained; the stable final artifact remains reported and verified to exist.

### 29.2 Focused evidence

```text
cargo test --test native_build_cli --no-fail-fast
  9 passed; 0 failed; 0 ignored

cargo test --lib native_toolchain
  2 passed; 0 failed; 0 ignored; 437 filtered out

cargo test --test native_c5_4_workspace --test native_c5_3_aggregates_enums --no-fail-fast
  27 passed; 0 failed; 0 ignored

cargo fmt --all -- --check
  passed

cargo clippy --workspace --all-targets --all-features -- -D warnings
  passed

cargo test --workspace --all-targets --no-fail-fast
  1096 passed; 0 failed; 2 ignored across 55 test-bearing binaries
```

### 29.3 Adversarial closure dispositions

1. **Check-only success:** impossible; `build` has its own driver and requires final artifact
   installation before success.
2. **MIR verification bypass:** impossible through the backend API; both backend entries require
   `VerifiedMirProgram`.
3. **Toolchain hiding source errors:** no; analysis/lowering/verification precede preflight.
4. **Old executable misreported:** no; installation occurs only after a new backend artifact and
   success is printed only after sibling-temp rename and final existence validation.
5. **Hashed/deleted backend path presented as live:** no; an unretained verbose build reports only
   the stable package-derived final artifact, while retained builds may also report the existing
   backend-local path.
6. **Missing runtime reaching Cargo:** no; runtime structure is preflighted first.
7. **rustc failure shown as source typing:** no; it is a structured backend-build failure.
8. **`--emit-rust` deleting output:** no; it implies retention and validates `src/main.rs`.
9. **Default generated-crate leak:** no; successful unretained builds remove the hashed crate.
10. **CLI linkage duplication:** none; C5.4 diagnostics cross the backend category boundary.
11. **Relocation changing binary name:** no; the relocated frozen workspace still emits `app`.
12. **Package-name path escape:** rejected by safe-component validation before building.
13. **Environment-secret output:** environment values are not printed; resolved paths and
    versions appear only in verbose mode.
14. **Unsupported feature reaching rustc:** C5.4/C5.3 preflight tests remain green; unsupported
    backend shapes map to `UnsupportedNative`.
15. **`run` engine drift:** none; `run` remains HIR-interpreter-backed.
16. **Cargo network access:** generated builds always pass `--offline`; empty-`CARGO_HOME` proof
    passes.
17. **Installed operation outside checkout:** installed-layout discovery and relocated-runtime
    manifest use are both proven.
18. **Stale expected output:** no; focused suites, fmt, strict clippy, and the 1,096-test workspace
    pass are green against the final implementation.
19. **Scalar-only closure:** no; closure uses the frozen three-package C5.4 workspace through the
    real CLI and runs its executable.
20. **C5.4 semantic alteration:** none; integration changes only build/toolchain orchestration and
    diagnostic transport.
