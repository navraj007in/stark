# STARK Language Project — Claude AI Assistant Context

## What STARK Is (Current)

STARK is a **pre-alpha general-purpose programming language with a working implementation**: a
safe, Rust-inspired ownership core (no GC), compiled natively through generated Rust, with an
optional tensor/model extension for AI/ML deployment. The `starkc/` implementation has a complete
Core v1 front end (lexer, parser, name resolution, type/borrow checking), a reference interpreter,
a MIR interpreter, native compilation on Linux/macOS/Windows, a tensor extension with bounded ONNX
import/verify/deploy, a formatter, test runner, documentation generator, and an LSP server. Beyond
the compiler there are 25 first-party packages under `packages/`, capability-declared host access
backed by native provider crates, and an installable toolchain — see "Implementation Status" below
for exactly what is closed vs. still open. It is not "specification-stage"; do not describe it
that way.

**Nor is it a research-only language any more.** Gate 7's project-wide "RETAIN AS RESEARCH
LANGUAGE" policy was superseded on 2026-08-04 (`starkc/docs/gate7-superseded.md`) by the evidence
that followed it — native compilation, the HTTP client written in STARK, the package ecosystem, the
installer, and the adopted application-platform `ROADMAP.md`. **The tensor track is the exception**
and remains deferred research on Gate 7's own terms: its productisation verdict is still DEFER,
gated on external-developer evidence nobody has gathered. Do not read platform progress as
permission to restart tensor work.

In early 2026 the project pivoted from an ambitious "AI-native, cloud-first"
design to a minimal, implementable **Core v1**. The rationale is in
`STARK_Analysis_and_Discussion.md`. The tensor/model extension remains the long-term
differentiator, but it is a non-core extension and a deferred track, not part of Core v1 and not
current work.

## Source of Truth

**Normative spec (Core v1): numbered source documents 00–07 and approved
`CORE-V1-*.md` semantic chapters under `STARKLANG/docs/spec/`**

| Document | Contents |
| --- | --- |
| `00-Core-Language-Overview.md` | Design goals, spec structure |
| `01-Lexical-Grammar.md` | Tokens, keywords, literals, operators |
| `02-Syntax-Grammar.md` | Full EBNF: items, generics, `self` receivers, patterns, ranges, casts |
| `03-Type-System.md` | Types, ownership/borrowing, references-and-lifetimes rules, generics, coherence, numeric semantics |
| `04-Semantic-Analysis.md` | Name resolution, borrow checking, exhaustiveness, definite assignment, error codes |
| `CORE-V1-ABSTRACT-MACHINE.md` | Values, places, execution, moves, references, destruction, traps |
| `CORE-V1-FUTURE-BOUNDARIES.md` | Reserved compatibility space, concurrency/FFI exclusions, extension isolation |
| `05-Memory-Model.md` | Memory-safety guarantees and authority boundaries |
| `06-Standard-Library.md` | Prelude, Option/Result, Vec/HashMap/String, Iterator, IO, math |
| `07-Modules-and-Packages.md` | `mod`/`use`, visibility, `starkpkg.json` manifest |
| `09-STARK-Language-Spec-v1.md` | Concise conformance summary |

`09-STARK-Language-Spec-v1.md` is a non-normative summary. Compiler-governance ledgers and
pending decisions live under `STARKLANG/docs/compiler/semantic-freeze/` and are non-normative;
only approved decisions transferred into the normative source documents define Core behavior.

`STARKLANG/docs/spec/STARK-Core-v1.md` (+ `.html`, `.pdf`) is a **generated
compilation** of the normative Core source documents. Never edit it directly — edit the
individual files, then regenerate:

```bash
python3 STARKLANG/tools/build-core-spec.py
```

**Extensions (optional, non-core): `STARKLANG/docs/extensions/`**
— `Tensor-Model-Types.md` is the normative draft of the tensor & model type
system (extension `tensor` v0.1): `Dim`/`DType` kinds, shape arguments
`Tensor<Float32, [B, 128]>`, polynomial dim equality, `TensorDyn`/`refine`
boundary, `model` declarations with verified ONNX import, device types.
`AI-Extensions.md` holds the remaining sketches (datasets, LLM blocks).

**Archived (do not treat as current): `STARKLANG/docs/archive/`, `web-docs/`,
`STARKLANG/compiler/`, `Practice/`** — pre-pivot design (actors, hybrid GC,
lowercase `i32`/`f32` types, `Package.stark` TOML manifest, ML pipeline DSL,
cloud annotations) and a Python prototype targeting it. Where archive and spec
conflict, the spec wins; see `STARKLANG/docs/archive/README.md` for the
conflict table.

## Core v1 Language Facts (for writing/reviewing STARK code)

- Primitive types are PascalCase: `Int32`, `UInt64`, `Float64`, `Bool`,
  `Char`, `String`, `str`, `Unit`. Never `i32`/`f32` (archive-era).
- Statements end with semicolons. Blocks are expressions; last expression is
  the block value.
- `let` immutable by default; `let mut` for mutable. Shadowing allowed only in
  nested scopes.
- Ownership + borrowing, Rust-like: one `&mut` XOR many `&`. **No lifetime
  annotations in Core v1**; instead: struct/enum declarations cannot write
  reference field types (generic instantiation like `Option<&T>` is allowed
  and produces a *borrow-carrying value* that behaves as a reference), a
  returned reference must derive from a reference parameter and takes the
  *shortest* input lifetime, and borrows bound with `let` are **lexically
  scoped** to end-of-block (temporary borrows end with their statement). See
  03-Type-System.md "References and Lifetimes".
- Generics with trait bounds (`fn max<T: Ord>(...)`), associated types, orphan
  rule. Generic args in expressions are inferred; turbofish (`size_of::<Int32>()`)
  exists only for the uninferable case.
- Operators on generic parameters desugar to traits: `==`→`Eq`, `<`→`Ord`,
  arithmetic→`Num` (compiler-known, primitives only). Method calls auto-borrow
  the receiver (`&`/`&mut`) and auto-dereference **repeatedly** — TYPE-METHOD-002
  removes one leading `&`/`&mut` at a time, trying by-value, shared- then
  exclusive-borrow form at each level, and stops at the first level with an
  applicable candidate. (Nested-reference receivers are therefore normative, not
  excluded.)
- Copy/Drop soundness: `Copy` requires all-Copy fields; `Copy`+`Drop` is
  forbidden; destructors run exactly once (drop flags); no explicit
  `Drop::drop` calls; no moves out of indexed places or `Drop` types.
- Function return types are never inferred; omitted `->` means `Unit`.
- Integer overflow, division by zero, out-of-bounds indexing, and failing `as`
  casts **always trap** — in every build mode. Traps and `panic` **abort**:
  destructors do not run.
- Errors via `Result<T, E>`/`Option<T>` and `?`. `panic(msg) -> !` (never
  type).
- Stdlib conformance profiles: `core-min` (MVP) and `std-full` — see
  06-Standard-Library.md.
- Not in Core v1 (reserved/future): `async`/`await`, closures/lambdas,
  `unsafe`, raw pointers, trait objects (`dyn`), lifetime annotations,
  `Rc`/`RefCell`, actors, tensors.
- Manifest: `starkpkg.json`; entry defaults to `src/main.stark`; std library
  under the `std` package name.

## Implementation Status

- Specification: Core v1 complete draft (numbered source documents 00–07 and approved
  `CORE-V1-*.md` semantic chapters normative; concise and generated views non-normative).
- Compiler: front end, semantic analysis and execution are done (`starkc/` — lexer, parser, name
  resolver, type/flow/borrow checker, a typed-HIR reference interpreter and a MIR interpreter; the
  114-fixture conformance suite is green). **Native compilation through generated Rust works in
  debug and release on Linux, macOS and Windows** (Gate C7), over a *qualified* standard-library
  subset — Gate C6 audited 87 methods and verified an invocation for 59. Programs are compared
  across four engine configurations (HIR interpreter, MIR interpreter, native debug, native
  release), each case pinned against the specification rather than against another engine.
  Also implemented: a `tensor` v0.1 extension front end with bounded ONNX signature
  import/verification, a Gate-5 native deployment path (generated Rust host + ONNX Runtime), a
  source formatter, a naming-convention test runner, a documentation generator, and an LSP server
  with a VS Code extension. The Python code in `STARKLANG/compiler/` and `Practice/Interpreter/`
  are pre-pivot prototypes and must not be extended for Core v1 work.
- Packages and host access: **25 first-party packages live under `packages/`** (moved there
  2026-08-04), each with a `*-consumer` package that must actually *call* its declared surface.
  A package reaching outside the process declares a capability in `starkpkg.json` — `clock`,
  `filesystem`, `process.env`/`process.args`, `random`, `tcp`/`dns`, `tls` — satisfied at build
  time by a native provider crate at `packages/<name>/native`. **Capability-backed packages build
  with `stark build` and cannot run under `stark run`**: the interpreters have no host access at
  all. An HTTP/1.1 and HTTPS client written in STARK closed 2026-08-03 (HC0–HC13).
- Distribution: Installer Phase I is implemented — release archives, platform installers, a
  versioned install tree (`lib/stark/versions/<v>` with `current`), uninstall, and `stark doctor`
  manifest verification. It proves **integrity, not authenticity**: archives are unsigned, the
  payload does not carry the first-party packages or providers, and an offline build of an
  HTTP/TLS program on a clean machine is not yet possible.
- Delivery has been governed by **two, non-overlapping gate sequences** — do not conflate them:
  - **Old sequence (`starkc/docs/gate1-exit.md` … `gate7-decision.md`)**, cited by
    `STARKLANG/docs/ROADMAP.md`: all seven gates are closed. Gates 1–5 built the front end,
    interpreter, tensor/ONNX front end, and a native ONNX-Runtime deployment demonstrator. Gates
    6–7 are decision checkpoints, not implementation gates: Gate 6 recorded **REVISE**, and Gate 7
    recorded a **tensor-track** verdict of technical POSITIVE with **productisation DEFER**
    (2026-07-16), gated on external-developer evidence that has still not been gathered — see
    `starkc/docs/gate7-decision.md`. That tensor verdict stands; **do not treat the tensor track as
    open.** Gate 7's separate project-wide **RETAIN AS RESEARCH LANGUAGE** policy, and its
    "only a `stark verify` track" scope limit, were **superseded 2026-08-04** —
    `starkc/docs/gate7-superseded.md`. `STARKLANG/docs/PLAN.md` has not been updated past Gate 5
    and should not be trusted for Gate 6/7 status.
  - **New sequence (Gate C0–C10)**, defined in
    `STARKLANG/docs/compiler/COMPILER-ROADMAP.md`/`COMPILER-CHARTER.md`, current status in
    `COMPILER-STATE.md` (repo root): a from-scratch, evidence-first re-closure of Core v1
    conformance, reference execution semantics, and (conditionally) native compilation. This is
    the **active governance track for compiler work** as of 2026-07-17; consult
    `COMPILER-STATE.md` before starting any compiler-track session, not this file's status
    summary, which is a snapshot only.
- Forward planning: **`ROADMAP.md` (repo root) is the single live roadmap** — the STARK
  Consolidated Roadmap, August 2026 – February 2027, adopted 2026-08-03. It governs package,
  application and platform work (operability → security/artifacts → REST server → structured
  concurrency → persistence → ecosystem), and its §0 states the authority boundary. It does
  **not** supersede the two live gate tracks below. All prior package/ecosystem roadmaps were
  consolidated into it and moved to `STARKLANG/docs/archive/roadmaps/`; do not schedule work
  from them. `STARKLANG/docs/ROADMAP.md` and `PLAN.md` are now historical records of the
  closed Gate 1–7 sequence, not forward plans.
- Scope discipline: work outside the current gate needs a roadmap-governed proposal — see
  `STARKLANG/docs/compiler/COMPILER-CHARTER.md` §1.6/§6 (compiler track), `ROADMAP.md` §12
  (current non-goals) or `STARKLANG/docs/ROADMAP.md` §4 (pre-existing non-goals).

## Working Conventions for This Repo

Three project skills encode the procedures that have actually failed here. Invoke them rather than
re-deriving the rules:

- **`stark-layout-verification`** — any change to paths, layout, the installer, provider crates or
  runtime/provider discovery. Three platform-divergent defects got past ad-hoc checking.
- **`stark-doc-sweep`** — before declaring any multi-document edit done.
- **`stark-package-authoring`** — creating, extending or reviewing a package under `packages/`.

- When editing any spec file in `docs/spec/`, regenerate `STARK-Core-v1.md`
  (+ HTML/PDF) in the same change, and keep the individual files as the
  editing surface.
- Spec changes must keep grammar, prose, and examples in agreement — the
  grammar in `02-Syntax-Grammar.md` must be able to parse every `stark` code
  block classified `parse-pass`/`semantic-error` in
  `STARKLANG/tests/spec-fixtures/manifest.toml` (semicolons included); blocks
  triaged `notation`/`lex-pass`/`parse-fail` are the sanctioned exceptions.
  Regenerate the corpus with `STARKLANG/tools/extract-spec-examples.sh` after
  spec edits — it fails if the fixture set diverges from the manifest, and any
  added/renumbered block must be re-triaged in the same change.
- New language features land in the spec first, extensions second, README
  last. The archive is never updated for new features.
- **Packages live under `packages/`, and dependency paths must stay siblings.** The workspace root
  is the *parent directory* of a package (`package.rs` `get_workspace_root`), so a dependency
  outside `packages/` is refused by name. Every first-party manifest therefore uses plain
  `../stark-<name>` paths.
- **Provider crates are depth-sensitive.** `packages/<name>/native/Cargo.toml` reaches the ABI
  through `../../../starkc/stark-provider-abi`, and `include_str!` in provider sources needs a
  further level (`../../../../starkc/providers/*.json`) because it resolves relative to the source
  file. The installed tree mirrors this at `lib/stark/packages/<name>/native`. One
  `stark-provider-abi` must satisfy both the runtime's `../` and a provider's `../../../` — Cargo
  refuses a lockfile naming one package at two paths, symlink or not.
- A new or changed package must pass `starkc/scripts/qualify-first-party-packages.py`, which
  requires that every declared public callable is actually *called* by the package's tests or its
  consumer, and that resource-shaped packages exercise acquire/use/release natively against a live
  peer.
- Test code must reach package paths through `starkc/tests/support/paths.rs`
  (`repo_package`, `repo_provider`, `repo_provider_root`), never by joining `"packages"` by hand.
- Do not run `cargo test --workspace` locally on a whim — this checkout is often shared by parallel
  sessions. Use scoped tests while iterating and read CI for the full picture; the provider crates
  have their *own* test targets that the `starkc` suite never compiles.

---

**Last Updated**: 2026-08-04
**Status**: STARK is a **pre-alpha general-purpose language with a working implementation**,
developed against the application-platform roadmap in `ROADMAP.md`. Core v1 specification
complete; front end, semantic analysis, execution, native compilation (Gate C7, three Tier-1
platforms) and compiler-backed language services done, over a qualified subset. 25 first-party
packages under `packages/`, capability-declared host access with native providers, an HTTP/1.1 and
HTTPS client written in STARK (HC0–HC13, closed 2026-08-03), and an installable toolchain
(Installer Phase I). The **tensor/ONNX extension is a deferred research track** on Gate 7's terms.
Gate 7's project-wide "research language" policy was superseded 2026-08-04
(`starkc/docs/gate7-superseded.md`). See `COMPILER-STATE.md` for the current compiler position.
**License**: MIT
