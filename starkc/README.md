# starkc

Compiler for the STARK Core v1 language. Rust, stable toolchain.
**Gates 1–7 (old numbering) are closed** — front end, semantic checker, interpreter, tensor/ONNX
front end, and a native ONNX-Runtime deployment demonstrator; see
[`docs/gate1-exit.md`](docs/gate1-exit.md) through [`docs/gate7-decision.md`](docs/gate7-decision.md).
Gate 7's decision was **RETAIN AS RESEARCH LANGUAGE**, authorizing only a `stark verify`
external-validation track as further tensor-track work. STARK Core programs can be checked and
executed by the tree-walking interpreter; the optional `tensor` extension provides static
tensor/model checks, bounded ONNX import/verify, and a bounded native deployment path (generated
Rust host + ONNX Runtime). The crate also has a source formatter (`stark fmt`), a
naming-convention test runner (`stark test`), a documentation generator (`stark doc`), and an LSP
server (`starkc lsp`). A separate, newer governance track (Gate C0–C10) is re-closing Core v1
conformance from first evidence — see [`docs/dev/compiler-map.md`](docs/dev/compiler-map.md) and
the repo-root `COMPILER-STATE.md` for current status; do not rely on this README's status claims
for that track.

- Language definition: [`../STARKLANG/docs/spec/`](../STARKLANG/docs/spec/)
  (normative), single file: `STARK-Core-v1.md`
- Delivery gates: [`../STARKLANG/docs/ROADMAP.md`](../STARKLANG/docs/ROADMAP.md)
- Engineering plan: [`../STARKLANG/docs/PLAN.md`](../STARKLANG/docs/PLAN.md)
  (standing decisions T1–T12, work packages)
- Conformance corpus: [`../STARKLANG/tests/spec-fixtures/`](../STARKLANG/tests/spec-fixtures/)

## Use

The crate builds **two separate CLI binaries with non-overlapping command sets** — `starkc`
(single-file/extension-aware compiler CLI, `cargo run` defaults to this one) and `stark`
(package-oriented build tool, always Core-only, requires a `starkpkg.json` root). See
`docs/dev/compiler-map.md` §2 for exactly how their `check`/`run` paths relate (they share the
same underlying library functions but currently disagree on whether warnings block execution).

```bash
# starkc — single-file / extension-aware
cargo run -- parse file.stark             # parse a program, report diagnostics
cargo run -- check file.stark             # resolve, type-check, and borrow-check
cargo run -- run file.stark               # check and execute a Core program
cargo run -- check --extension tensor file.stark  # check tensor/model source
cargo run -- import model.onnx --out model.stark  # generate a declaration
cargo run -- verify model.onnx --declaration model.stark  # detect drift
cargo run -- deploy pipeline.stark --model model.onnx --entry run --out dir/  # native deployment
cargo run -- parse --snippet --dump f.stark  # block-body mode, print the AST
cargo run -- lex file.stark               # dump the token stream
cargo run -- lsp                          # start the LSP server on stdio

# stark — package-oriented build tool (run from a directory with starkpkg.json)
cargo run --bin stark -- check            # resolve, type-check, borrow-check the package
cargo run --bin stark -- build            # (currently: alias for check; no native codegen yet)
cargo run --bin stark -- run              # check and execute the package's main entry
cargo run --bin stark -- test             # discover and run fn test_*()/test_ignored_*()
cargo run --bin stark -- fmt --check      # verify formatting; omit --check to rewrite in place
cargo run --bin stark -- doc --open       # generate and open HTML documentation

cargo test                                # everything, incl. the 112-fixture
                                          # conformance suite and pseudo-fuzz
UPDATE_SNAPSHOTS=1 cargo test --test snapshots   # regenerate AST snapshots
```

## Terminal IDE

`starkide` is a dependency-free, Turbo C++ inspired terminal workbench. It
provides the classic blue/white interface, keyboard-driven drop-down menus, a
Unicode-aware source editor, multiple buffers, project/recent-file pickers,
search, undo/redo, diagnostic navigation, and separate build/program output
wired to the complete Gate 3 pipeline.

```bash
cargo run --bin starkide                         # new buffer
cargo run --bin starkide -- ../Practice/Basics/hello.st
```

Use **F10** for menus, **F2** to save, **F9** to compile, **Ctrl+F9** to run,
**F4** to visit diagnostics, and **Ctrl+Q** to quit. Program output and runtime
diagnostics appear in the messages pane. See
[`docs/terminal-ide.md`](docs/terminal-ide.md) for the complete workflow.

## Release binaries

The dependency-free release builder packages both `starkc` and `starkide`
with the license, README, build metadata, and a SHA-256 checksum:

```bash
# macOS (build the current Intel or Apple Silicon host)
python3 scripts/build-release.py

# Windows PowerShell or Command Prompt
py -3 scripts/build-release.py
```

Packages are written to `target/packages/`. macOS produces a `.tar.gz` and
Windows produces a `.zip`. Each operating system and CPU architecture needs
its own binary; one executable cannot run on both macOS and Windows.

An explicit target is useful in CI or with a configured cross-linker:

```bash
python3 scripts/build-release.py --target aarch64-apple-darwin
python3 scripts/build-release.py --target x86_64-apple-darwin
py -3 scripts/build-release.py --target x86_64-pc-windows-msvc
```

Install the requested standard library first with `rustup target add <target>`.
Cross-compiling also requires a linker for the destination platform. Release
archives produced by this script are currently unsigned; public macOS builds
will still need Apple code signing/notarization, and public Windows builds
should be Authenticode-signed.

## Layout

| Module | Contents | Status |
| --- | --- | --- |
| `source` | `SourceFile`, `Span`, line/column mapping | done (WP1.1) |
| `diag` | diagnostics in the spec's normative render format | done (WP1.1) |
| `lexer` | `01-Lexical-Grammar.md` | done (WP1.2) |
| `tests/conformance` | manifest-driven fixture harness | done (WP1.3) |
| `ast` | arena + ID node tree (PLAN.md T6) + dump | done (WP1.4) |
| `parser` | recursive descent, 16-level precedence (`02-Syntax-Grammar.md`) | done (WP1.4) |
| `tests/snapshots` | AST snapshots for 15 representative fixtures | done (WP1.4) |
| `tests/robustness` | deterministic pseudo-fuzz (no panics/hangs) | done (WP1.4) |
| `tests/diag_format` | end-to-end diagnostic-format goldens | done (WP1.5) |
| `docs/gate1-exit.md` | Gate 1 exit report (T10 ledger, open questions) | done (WP1.5) |
| `hir`, `resolve` | HIR lowering, lexical/module names, imports, visibility | done (M2.1) |
| `typecheck`, `flow` | inference, control flow, generics, traits, coherence | done (M2.2–M2.3) |
| `borrowck` | moves, partial moves, reinitialization, lexical borrows | done (M2.4) |
| `tests/gate2-valid` | 26 end-to-end valid semantic programs | done (M2.5) |
| `docs/gate2-exit.md` | Gate 2 exit evidence and deferrals | done (M2.5) |
| `interp` | typed-HIR interpreter, runtime values, control flow, drops | done (M3.1) |
| Core runtime | String, Vec, Option/Result, Box, ranges, print and file I/O | done (M3.2) |
| `examples/gate3`, `tests/gate3_execution` | executable programs and CLI/runtime tests | done (M3.3) |
| `docs/gate3-exit.md` | Gate 3 exit evidence | done (M3.3) |
| `extensions/tensor`, tensor checker | shape, dtype, device, and operation typing | done (M4.0–M4.4) |
| `onnx`, `tests/gate4_onnx` | bounded import and signature verification | done (M4.5) |
| `examples/gate4`, `docs/gate4-exit.md` | Gate 4 demonstrations and exit evidence | done (M4.6) |
| `deploy` | HIR → Deployment IR lowering, deterministic Rust-host emission (ONNX Runtime backend) | done (Gate 5) |
| `docs/gate5-*.md` | Gate 5 backend decision and exit evidence | done (Gate 5) |
| `package` | `starkpkg.json` manifests, dependency resolution, `stark.lock` | done (Phase 2–3, see PLAN.md) |
| `formatter` | AST-driven, comment-preserving, idempotent source formatter (`stark fmt`) | done (WP8.2) |
| `test_runner` | naming-convention test discovery/execution (`stark test`) | done (WP8.3) |
| `lsp` | stdio JSON-RPC LSP server, document sync (`starkc lsp`) | done (WP8.1) |
| `doc_gen` | HTML documentation generator with search (`stark doc`) | done (WP8.5) |
| `docs/gate6-memo.md`, `docs/gate7-decision.md` | tensor-track decision checkpoints (REVISE; RETAIN AS RESEARCH) | done |
| `docs/terminal-ide.md` | terminal IDE editing, project, build and run workflow | stable |

Architecture target: `Source → Tokens → AST → HIR → typed HIR → backend`;
Gate 3 adds execution of typed HIR plus the `core-min` runtime. Spec defects found during implementation
follow the T10 protocol: spec fix + regenerated artifacts + fixture re-triage
+ compiler change in one commit.

**This table reflects the old (non-"C") gate/milestone numbering only.** For a
module-by-module purpose/input/output/entry-point map that is independent of any gate
numbering and was produced by direct source audit rather than milestone bookkeeping, see
[`docs/dev/compiler-map.md`](docs/dev/compiler-map.md). Where the two disagree, trust
`compiler-map.md` — it is the more recently and more rigorously verified of the two.
