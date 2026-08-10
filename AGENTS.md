# STARK Language Project — Codex AI Assistant Context

> **`CLAUDE.md` is the fuller version of this document and wins on any disagreement.** This file
> carries what a session needs before it touches anything; that one carries the detail.
>
> **Rewritten 2026-08-04.** Everything previously here described the *pre-pivot* design — an
> "AI-native" language with garbage collection, actors, a 240-opcode stack VM, bytecode
> generation, and TensorLib/DatasetLib/ModelLib/NetworkingLib. **None of that exists or is
> planned.** It is preserved in `STARKLANG/docs/archive/` (see that directory's `README.md` for
> the conflict table) and in git history. If you find a document describing actors, a VM, hybrid
> GC, lowercase `i32`/`f32` types, or a `Package.stark` TOML manifest, it is archive-era and wrong.

## What STARK is, today

A **pre-alpha general-purpose programming language with a working implementation**: a safe,
Rust-inspired ownership core with no garbage collector, compiled natively through generated Rust,
plus an optional tensor/model extension for AI/ML deployment that is currently a **deferred
research track**.

It is not specification-stage, and it is not a research-only language — Gate 7's project-wide
"RETAIN AS RESEARCH LANGUAGE" policy was superseded on 2026-08-04
(`starkc/docs/gate7-superseded.md`). It is also not production-ready: no stability guarantees,
breaking changes expected.

## Before you start a session

| Question | Authority |
| --- | --- |
| What is the compiler's current position? | `COMPILER-STATE.md` (repo root) — **read this before any compiler work** |
| What is planned? | `ROADMAP.md` (repo root) — the single live forward plan |
| What does the language mean? | `STARKLANG/docs/spec/`, source documents 00–07 + `CORE-V1-*.md` |
| What are the rules for compiler work? | `STARKLANG/docs/compiler/COMPILER-CHARTER.md` |

Never take this file's status summary as current. It is a snapshot and it will lag.

## Core v1 language facts

These are the ones most often got wrong:

- Primitive types are **PascalCase**: `Int32`, `UInt64`, `Float64`, `Bool`, `Char`, `String`,
  `str`, `Unit`. Never `i32`/`f32` — that is archive-era.
- Statements end with semicolons. Blocks are expressions; the last expression is the block value.
- `let` is immutable by default; `let mut` for mutable.
- Ownership and borrowing, Rust-like: one `&mut` XOR many `&`. **No lifetime annotations.**
- Integer overflow, division by zero, out-of-bounds indexing and failing `as` casts **always trap,
  in every build mode**. Traps and `panic` abort; destructors do not run.
- Errors via `Result<T, E>`/`Option<T>` and `?`. `panic(msg) -> !`.
- **Not in Core v1**: `async`/`await`, closures, `unsafe`, raw pointers, trait objects (`dyn`),
  lifetime annotations, `Rc`/`RefCell`, actors, tensors (extension only).
- Iterator combinators (`map`, `filter`, `collect`, `fold`, …) are **refused by the front end**
  with `E0105`. Iterate a borrow (`v.iter()`) in a `for` loop.
- Manifest is `starkpkg.json`; entry defaults to `src/main.stark`.

## What exists

- **Compiler** (`starkc/`): lexer, parser, name resolution, type/flow/borrow checking, a
  typed-HIR reference interpreter, a MIR interpreter, and **native compilation through generated
  Rust in debug and release on Linux, macOS and Windows** — over a *qualified* standard-library
  subset (87 methods audited, 59 with a verified invocation). Programs are compared across all
  four engine configurations, each case pinned against the specification rather than against
  another engine.
- **Tooling**: `stark fmt`, `stark test`, `stark doc`, `stark doctor`, `starkc lsp` with a VS Code
  extension, and `starkide`, a terminal IDE.
- **Packages**: 29 first-party libraries plus the `stark-get` application under `packages/`, with
  qualification consumers that must actually *call* the declared surfaces they cover. Includes an
  HTTP/1.1 and HTTPS client written in STARK
  (HC0–HC13, closed 2026-08-03), TLS, JSON, CSV, URL and the encoding family.
- **Host access**: capability-derived, envelope-checked and provider-backed. Vocabulary v1 names
  `filesystem-read`, `filesystem-write`, `environment-read`, `network-client`, `network-listen`,
  `clock`, `randomness`, `process-execution`, and `native-code`. The root manifest approves the
  transitive derived set; a native provider crate at `packages/<name>/native` satisfies it.
- **Distribution**: release archives, platform installers, a versioned install tree, uninstall,
  29 explicitly marked toolchain libraries, six provider crates, executable-relative local package
  resolution, and named package/provider checks in `stark doctor`.

## What does not exist

Garbage collection, actors, a virtual machine, bytecode, async/await, closures, a public package
registry, an HTTP server, structured concurrency, persistent storage, training or autodiff, GPU
kernel generation, and a signed distribution. Several of these are on `ROADMAP.md`; none of them
is in the code.

## Encoded procedures

`.claude/skills/` holds three project skills covering the work that has actually gone wrong here.
Read the relevant one before starting; they are written as checklists, not background reading.

| skill | when |
| --- | --- |
| `stark-layout-verification` | any change to paths, layout, the installer, provider crates, or runtime/provider discovery |
| `stark-doc-sweep` | before declaring any multi-document edit done |
| `stark-package-authoring` | creating, extending or reviewing a package under `packages/` |

## Working rules that have cost time when broken

- **`stark run` cannot execute anything that touches the host.** The interpreters have no provider
  layer, so capability-backed packages build with `stark build` or not at all.
- **Path dependencies may leave the package's parent directory.** They are canonicalized, and the
  lockfile records the resolved path plus content hash. First-party packages remain siblings for a
  relocatable repository layout.
- **Provider crates are depth-sensitive.** `packages/<name>/native/Cargo.toml` reaches the ABI
  through `../../../starkc/stark-provider-abi`; `include_str!` in provider *sources* needs one
  more level, because it resolves relative to the source file rather than the manifest. One
  `stark-provider-abi` must satisfy both the runtime's `../` and a provider's `../../../` — Cargo
  refuses a lockfile naming one package at two paths, and a symlink does not help.
- **Test code reaches package paths through `starkc/tests/support/paths.rs`**
  (`repo_package`, `repo_provider`, `repo_provider_root`), never by joining `"packages"` by hand.
- **The provider crates have their own test targets** that the `starkc` suite never compiles. They
  are built by `--manifest-path` in `.github/workflows/c78-native-capabilities.yml`. A change to
  provider sources is not verified until those run.
- **This checkout is often shared by parallel sessions.** Never `git add` broadly, never format the
  whole tree, and do not stage or commit files you did not write.
- Prefer scoped local tests and read CI for the full picture.

## Spec editing

- Editing anything in `STARKLANG/docs/spec/` means regenerating the compiled view in the same
  change: `python3 STARKLANG/tools/build-core-spec.py`. Never edit `STARK-Core-v1.md` directly.
- Grammar, prose and examples must agree. Regenerate the fixture corpus with
  `STARKLANG/tools/extract-spec-examples.sh`; it fails if the fixture set diverges from
  `STARKLANG/tests/spec-fixtures/manifest.toml`, and any added or renumbered block must be
  re-triaged in the same change.
- New language features land in the spec first, extensions second, README last. The archive is
  never updated for new features.

## Scope discipline

Work outside the current gate needs a roadmap-governed proposal — `COMPILER-CHARTER.md` §1.6/§6
for compiler work, `ROADMAP.md` §12 for current non-goals. **The tensor track is closed**: Gate 7
recorded productisation DEFER pending external-developer evidence that has not been gathered, and
nothing in the platform work reopens it.

---

**Last Updated**: 2026-08-04
**License**: MIT
