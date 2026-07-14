# Archived Documentation (Pre-Pivot Design)

**Status: SUPERSEDED — retained for historical reference only.**

Everything in this directory describes the original, pre-pivot STARK design
(the "AI-native, cloud-first" language with actors, a hybrid GC memory model,
lowercase primitive types like `i32`/`f32`, a `Package.stark` TOML manifest,
ML pipeline DSLs, and cloud/serverless annotations). In early 2026 the project
was refocused on a small, implementable core language; see
`STARK_Analysis_and_Discussion.md` at the repository root for the rationale.

**The normative specification is `docs/spec/` (Core v1).** Optional AI/ML
extensions are sketched in `docs/extensions/`. Where anything in this archive
contradicts those documents, the archive is wrong. Known conflicts include:

| Topic | Archive says | Current (normative) |
| --- | --- | --- |
| Primitive types | `i32`, `f32`, ... | `Int32`, `Float32`, ... (`docs/spec/03-Type-System.md`) |
| Memory model | Hybrid ownership + GC | Ownership/borrowing only (`docs/spec/05-Memory-Model.md`) |
| Concurrency | Actors, async/await | Not in Core v1 (reserved keywords; future extension) |
| Package manifest | `Package.stark` (TOML) | `starkpkg.json` (`docs/spec/07-Modules-and-Packages.md`) |
| Tensor types | `Tensor<f32, [1024, 768]>` and `Tensor<T>[dims]` | `Tensor<Float32, [1024, 768]>` (`docs/extensions/AI-Extensions.md`) |
| Grammar | `docs/archive/03-Syntax/STARK-Formal-Grammar.md` | `docs/spec/02-Syntax-Grammar.md` |
| Standard library | `docs/archive/06-Standard-Library/` (TensorLib, CloudLib, ...) | `docs/spec/06-Standard-Library.md`; AI/cloud libraries are non-core extensions |
| Bytecode/VM | Two overlapping instruction-set docs | Non-normative architectural sketches only |

Duplicate documents within the archive (e.g. two concurrency specs, two
bytecode instruction sets) reflect iterations of the old design; none of them
is authoritative.
