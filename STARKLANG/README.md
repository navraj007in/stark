# STARKLANG

This directory contains the STARK language specification and related material.

**Current, normative:**
- [`docs/spec/`](./docs/spec/) — the Core v1 language specification
  (lexing, syntax, types, semantics, memory model, stdlib, modules)
- [`docs/extensions/`](./docs/extensions/) — optional, non-core AI/ML
  extension sketches
- [`docs/index.md`](./docs/index.md) — documentation index

**Historical (superseded):**
- [`docs/archive/`](./docs/archive/README.md) — the pre-pivot "AI-native,
  cloud-first" design documents (actors, GC hybrid, ML pipeline DSL,
  serverless annotations). Kept for reference; contradicts the current spec
  in several places — see the archive README's conflict table.
- `index.html`, `index1.html`, `index2.html` — old web pages for the
  pre-pivot design.
- [`compiler/`](./compiler/README.md) — a Python prototype type checker for
  the pre-pivot design; not a Core v1 implementation.

There is **no working compiler yet**. See the repository
[README](../README.md) for project status and roadmap.
