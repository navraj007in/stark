# STARK Language Spec v1 (Concise)

## Scope
This document summarizes the Core v1 language specification. The normative documents are in `docs/spec/` and are listed below. Non-core extensions are defined separately and are optional.

## Terminology
The key words **MUST**, **SHOULD**, and **MAY** are to be interpreted as follows:
- **MUST**: an absolute requirement for conformance.
- **SHOULD**: a strong recommendation; valid reasons may exist to deviate, but the implications must be understood and documented.
- **MAY**: an optional item that does not affect conformance.

## Normative Core v1 Documents
1. Core overview: `docs/spec/00-Core-Language-Overview.md`
2. Lexical grammar: `docs/spec/01-Lexical-Grammar.md`
3. Syntax grammar: `docs/spec/02-Syntax-Grammar.md`
4. Type system: `docs/spec/03-Type-System.md`
5. Semantic analysis: `docs/spec/04-Semantic-Analysis.md`
6. Memory model: `docs/spec/05-Memory-Model.md`
7. Standard library: `docs/spec/06-Standard-Library.md`
8. Modules and packages: `docs/spec/07-Modules-and-Packages.md`

## Non-Core Extensions
1. AI/ML extensions: `docs/extensions/AI-Extensions.md` (optional)

## Spec Status
| Area | Status | Notes |
| --- | --- | --- |
| Lexical grammar | Normative | Core v1 |
| Syntax grammar | Normative | Core v1 |
| Type system | Normative | Core v1 |
| Semantic analysis | Normative | Core v1 |
| Memory model | Normative | Core v1 |
| Standard library | Normative | Core v1 |
| Modules and packages | Normative | Core v1 |
| Extensions | Non-core | Optional |

## Conformance
An implementation claiming Core v1 conformance MUST implement the normative Core v1 documents listed above, and MUST document any deviations or extensions.

## Conformance Checklist (Core v1)
1. Lexer recognizes all tokens and keywords per `01-Lexical-Grammar.md`.
2. Parser accepts all grammar productions in `02-Syntax-Grammar.md`.
3. Type checker enforces ownership, borrowing, generics, and trait coherence rules in `03-Type-System.md`.
4. Semantic analysis enforces exhaustiveness, initialization, and runtime error semantics in `04-Semantic-Analysis.md`.
5. Memory model guarantees ownership and borrowing invariants in `05-Memory-Model.md`.
6. Standard library implements required behavior in `06-Standard-Library.md`.
7. Module and package resolution follow `07-Modules-and-Packages.md`.
