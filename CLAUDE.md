# STARK Language Project — Claude AI Assistant Context

## What STARK Is (Current)

STARK is a **specification-stage programming language**: a safe, compiled,
general-purpose core (Rust-inspired ownership, no GC) with optional AI/ML
extensions planned on top. The compiler front end (lexer + parser in
`starkc/`) exists; semantic analysis and execution do not yet.

In early 2026 the project pivoted from an ambitious "AI-native, cloud-first"
design to a minimal, implementable **Core v1**. The rationale is in
`STARK_Analysis_and_Discussion.md`. Long-term, the differentiator remains
AI/ML deployment (compile-time tensor shape checking, model loading), but
those are non-core extensions, not part of Core v1.

## Source of Truth

**Normative spec (Core v1): `STARKLANG/docs/spec/`**

| Document | Contents |
| --- | --- |
| `00-Core-Language-Overview.md` | Design goals, spec structure |
| `01-Lexical-Grammar.md` | Tokens, keywords, literals, operators |
| `02-Syntax-Grammar.md` | Full EBNF: items, generics, `self` receivers, patterns, ranges, casts |
| `03-Type-System.md` | Types, ownership/borrowing, references-and-lifetimes rules, generics, coherence, numeric semantics |
| `04-Semantic-Analysis.md` | Name resolution, borrow checking, exhaustiveness, definite assignment, error codes |
| `05-Memory-Model.md` | Ownership, moves, Copy vs Move, Drop, layout |
| `06-Standard-Library.md` | Prelude, Option/Result, Vec/HashMap/String, Iterator, IO, math |
| `07-Modules-and-Packages.md` | `mod`/`use`, visibility, `starkpkg.json` manifest |
| `09-STARK-Language-Spec-v1.md` | Concise conformance summary |

`STARKLANG/docs/spec/STARK-Core-v1.md` (+ `.html`, `.pdf`) is a **generated
compilation** of files 00–07. Never edit it directly — edit the individual
files, then regenerate:

```bash
cd STARKLANG/docs/spec
for f in 00-Core-Language-Overview 01-Lexical-Grammar 02-Syntax-Grammar \
         03-Type-System 04-Semantic-Analysis 05-Memory-Model \
         06-Standard-Library 07-Modules-and-Packages; do
  cat "$f.md"; printf '\n\n---\n\n'
done > STARK-Core-v1.md
pandoc STARK-Core-v1.md -s --metadata title="STARK-Core-v1" -o STARK-Core-v1.html
pandoc STARK-Core-v1.md -s --metadata title="STARK-Core-v1" --pdf-engine=weasyprint -o STARK-Core-v1.pdf
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
  the receiver (`&`/`&mut`) and auto-deref one reference level.
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

- Specification: Core v1 complete draft (all 8 documents normative).
- Compiler: front end done (`starkc/` — lexer WP1.2, parser WP1.4; the
  121-fixture conformance suite is green and required in CI). Semantic
  analysis (Gate 2) and execution (Gate 3) not started. The Python code in
  `STARKLANG/compiler/` is a pre-pivot prototype and must not be extended
  for Core v1 work.
- Delivery is governed by `STARKLANG/docs/ROADMAP.md` (Gates 1–6 with
  evidence-based exit criteria) and executed per `STARKLANG/docs/PLAN.md`
  (standing decisions T1–T12, work packages). Gate 1 is closed
  (`starkc/docs/gate1-exit.md`); next: Gate 2, the semantic checker
  (M2.1–M2.5).
- Scope discipline: work outside the current gate needs a roadmap-governed
  proposal; see ROADMAP.md §4 non-goals.

## Working Conventions for This Repo

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

---

**Last Updated**: July 2026
**Status**: Core v1 specification complete; compiler front end done (Gate 1
closed; Gate 2 next)
**License**: MIT
