# Gate C2 Semantic-Freeze Preflight Audit

Status: completed 2026-07-18
Implementation baseline: `be43874` (`complete WP-C2.5 diagnostic transport`)
Preflight commit: `b34d2d02e94aca442c27c99a0cd5bc9daac43268`
Next work package: WP-C2.6

## Purpose and boundary

This audit is the required transition between WP-C2.5 and WP-C2.6. It reconciles the
Core v1 semantic-freeze execution plan with the repository as it exists after the compiler
service and diagnostic-transport work. It does not approve new language semantics, edit
normative rules, renumber the roadmap, or begin the full question-by-question inventory owned
by WP-C2.6.

The audit read the compiler charter, roadmap and state; the Core v1 overview and chapters
01–07; the concise and generated combined specifications; the reference-execution contract;
the conformance database and generated coverage evidence; the known-deviation ledger; and
`STARKLANG/docs/compiler/plans/CORE-V1-SEMANTIC-FREEZE-EXECUTION-PLAN.md`. The original
owner-provided plan has SHA-256
`145bc076c28020d5f48793390fda2c57a0241edaa2e777f90f222a3afc0203f8`; the repository copy
records its preparation baseline and any correction made while importing it.

## Repository-to-plan gap map

`Present` means the repository has useful material, not that the semantic domain is complete.
Compatibility cost is the likely cost of changing a decision after native compilation or
external packages depend on it.

| Domain | Present authority or evidence | Preflight finding | Cost | Owner |
| --- | --- | --- | --- | --- |
| Source encoding and lexing | `01-Lexical-Grammar.md`, LEX rules | UTF-8 is stated, but invalid input and byte/character boundary obligations need a granular inventory. | Medium | C2.6, C2.11 |
| Grammar and parsing | `02-Syntax-Grammar.md`, SYN rules | Broad grammar exists; edge productions, reserved future syntax, and parser-only DEV-036 need exact ownership. | Medium | C2.6, C2.12 |
| Names and scopes | `04-Semantic-Analysis.md`, `07-Modules-and-Packages.md` | Resolution rules are split between chapters; package-qualified identity is not complete. | High | C2.6, C2.8, C2.9 |
| Type identity and well-formedness | `03-Type-System.md` | Alias transparency, recursive finite sizedness, unsized boundaries, and edge types are not settled. | High | C2.8 |
| Inference and coercions | `03-Type-System.md` | Local inference is described, but constraint solving, expected types, ambiguity, normalization, and defaulting are incomplete. | High | C2.8 |
| Traits and coherence | `03-Type-System.md`, `06-Standard-Library.md` | Orphan/overlap rules are broad; selection, normalization, package identity, callable builtin obligations, and the semantic laws of `Eq`/`Ord`/`Hash` are incomplete. | High | C2.8, C2.9, C2.11 |
| Ownership and borrowing | `03-Type-System.md`, `05-Memory-Model.md` | Legality is duplicated; authority must be separated from execution and representation. | High | C2.6, C2.7, C2.8 |
| Places, moves, and temporaries | reference-execution contract, interpreter regressions | Implementation behavior is evidenced, but no implementation-independent abstract machine defines all legal expressions and failure cleanup. | High | C2.7 |
| Patterns and destruction | type/semantic/memory chapters, C1/C2.2 tests | Exhaustiveness exists; binding ownership, wildcard destruction, partial failure, and exact destruction categories are incomplete. | High | C2.7, C2.8 |
| Constants | `04-Semantic-Analysis.md`, E0009 evidence | Array-repeat checks exist; the evaluable subset, cycles, cross-package use, overflow, and traps are not defined as one contract. | High | C2.8 |
| Integer semantics | type and runtime-error sections | Overflow and division-by-zero policy exists; signed division/remainder, shifts, negation minimum, and detailed casts remain incomplete. | High | C2.9 |
| Floating-point semantics | `03-Type-System.md`, math APIs | IEEE 754 is named, but rounding, NaN propagation, signed zero, reproducibility, conversions, and `Eq`/`Ord`/`Hash` are unsettled. | High | C2.9 |
| Strings and Unicode | lexical, type, and standard-library chapters | UTF-8/String/Char basics exist; indexing units, invalid boundaries, case-data version, trim, and byte escapes need decisions. | High | C2.9 |
| Modules | `07-Modules-and-Packages.md` | Module layout and resolution exist, but their boundary with package identity and public API identity needs one authority map. | Medium | C2.6, C2.9 |
| Packages and package identity | `07-Modules-and-Packages.md` | Semver selection exists; relocation-stable logical source/package-instance identity, lockfiles, aliases, source substitution, and the separate one-version-per-major-line coexistence invariant are incomplete. | High | C2.9 |
| Entry and process behavior | package entry-file rule and CLIs | No complete executable `main` signature, exit mapping, stream encoding, startup, shutdown, or trap contract exists. | High | C2.9 |
| Standard-library language hooks | `06-Standard-Library.md`, interpreter builtins | Required APIs, language hooks, conformance profiles, and implementation/performance notes are mixed. DEV-009/023/024 remain relevant. | High | C2.6, C2.8, C2.9, C2.11 |
| Layout observability | `03-Type-System.md`, `05-Memory-Model.md` | Pointer sizes, stack/heap claims, and tagged-union language improperly imply representation guarantees. | High | C2.9 |
| Panic and trap termination | `04-Semantic-Analysis.md`, reference-execution contract | Abort behavior is partly specified; observable categories, source, cleanup boundary, and process mapping need one contract. | High | C2.7, C2.9 |
| Resource exhaustion and limits | scattered implementation behavior | Allocation/stack exhaustion, recursion/call-depth and object/source/package limits, host I/O failure, OS termination, and failures outside defined STARK traps are not classified. | High | C2.7, C2.9 |
| Target-defined behavior | scattered platform notes | There is no complete classification of specified, implementation-defined, target-defined, unspecified, and prohibited behavior. | High | C2.6, C2.9 |
| Extension boundaries | overview, extension documents, language options | Core isolation exists in code and policy; future syntax/ownership/concurrency/native-provider boundaries need a coherent compatibility document. | High | C2.10 |

## Authority conflicts and stale material

These are inventory inputs, not edits approved by this preflight.

| Location | Finding | Required disposition |
| --- | --- | --- |
| `00-Core-Language-Overview.md` introduction | Says Core has not been validated by a conforming implementation; this is stale relative to the current evidenced-but-qualified implementation status. | C2.6 must replace the status claim with an accurate, non-normative pointer. |
| `00-Core-Language-Overview.md` implementation phases, success criteria, and next steps | Project planning is embedded in the normative specification set. | Remove from the normative generated surface or label and relocate it as non-normative history. |
| `03-Type-System.md` implementation notes | Describes references as pointers and enums as tagged unions with an “optimal layout.” | Replace representation promises with language-level semantics and C2.9 layout classifications. |
| `05-Memory-Model.md` memory layout and implementation notes | Freezes stack/heap choices, 64-bit reference size, pointer representation, escape analysis, and compiler phases. | Remove implementation leakage; make C2.7 the execution authority and C2.9 the observability authority. |
| `03-Type-System.md` and `05-Memory-Model.md` | Both state ownership, borrowing, Copy, and Drop rules. | C2.6 must assign one normative home per concept and turn duplicate text into summaries/cross-references. |
| `04-Semantic-Analysis.md` | Mixes language constraints, diagnostic catalogue, pass order, and compiler data-structure suggestions. | Separate normative acceptance/rejection from non-normative implementation guidance. |
| `06-Standard-Library.md` | Mixes required APIs and language hooks with `core-min`/`std-full`, platform, and performance guidance. | Map every hook/API to one authority; do not broaden the public API during inventory. |
| `reference-execution.md` | Correctly records the current comparator, but necessarily mentions HIR/interpreter mechanisms. | Retain as implementation evidence; C2.7 must not copy those mechanisms into the abstract machine. |
| `STARK-Core-v1.md` | Generated combined specification. | Never hand-edit; regenerate only after approved normative source changes. |

## Conformance and deviation reconciliation

The current conformance database has 59 broad rules: 53 are marked implemented and 6 partial.
Only 20 rules have function-level positive/negative classifications. The other 39 are reported
as unclassified rather than treated as complete. Chapter counts are 14 LEX, 13 SYN, 8 TYP,
7 SEM, 6 MEM, 5 STD, and 6 PKG. This is useful coverage metadata, but it is not the granular
normative-question inventory required for semantic freeze.

The following deviation ownership must be preserved or clarified:

| Deviation | Preflight classification | Owner |
| --- | --- | --- |
| DEV-005 CLI warning-gating drift | Open cross-tool convergence issue. | C2.11 |
| DEV-009 first-class `File` representation | Open standard-library/runtime-hook question. | C2.9 decision, C2.11 alignment if approved |
| DEV-010 LSP hover/definition/reference stubs | C2.4 supplied compiler queries; consumer completion remains later developer-experience work. | Later tooling gate |
| DEV-011 doc comments absent from HIR | Not a semantic-freeze blocker unless a normative metadata behavior is identified. | Later documentation/tooling gate |
| DEV-017 classified evidence gap | Partially closed; 39 rules remain unclassified. | C2.6 rule split, C2.11 evidence |
| DEV-018 AST span-integrity coverage | C2.4 supplied production position indexing, but the ledger still reports incomplete exhaustive Type/Pat/Item containment verification. Requalify against the actual walker and tests; do not assume closure. | C2.6 ledger reconciliation, C2.11 evidence if still open |
| DEV-019 diagnostic-code collisions | Explicitly deferred; current pre-alpha allocations are not frozen. | C2.11 |
| DEV-022 private types in public signatures | Spec-silent and unimplemented. | C2.9 decision, C2.11 alignment |
| DEV-023 builtin `Display`/`Hash` callable methods | Depends on the language-hook contract. | C2.8 decision, C2.11 alignment |
| DEV-024 `From::from` associated resolution | Depends on trait selection/normalization. | C2.8 decision, C2.11 alignment |
| DEV-036 parser filename bypass | Explicitly deferred with the differential corpus. | C2.12 |

DEV-026–035 and DEV-037–043 were closed by WP-C2.2 and its correction pass. They are seeds for
abstract-machine examples and differential tests, not permission to reopen implementation work
without a newly demonstrated semantic gap.

## Work already complete

The transition preserves these completed results:

- C2.1 defines the current reference-interpreter comparator.
- C2.2 repaired the owned interpreter semantic deviations and added regressions.
- C2.3 established one compiler-owned project-analysis result.
- C2.4 established position, type, definition, reference, and symbol queries.
- C2.5 established structured, source-versioned diagnostic transport.
- C1 evidence remains valid as implementation evidence, but does not substitute for a complete
  normative-question inventory.
- C3, MIR, backend selection, and native compilation remain blocked until a frozen C2.13
  outcome.

## Owner-decision register

Recommendations below are audit defaults only. They are not approved semantic decisions.

| ID | Decision required | Recommended default | Owner | Approval |
| --- | --- | --- | --- | --- |
| SF-OD-001 | Normative authority split | Type System owns legality; Abstract Machine owns execution; Memory Model summarizes guarantees; C2.9 owns observable layout. | C2.6/C2.7 | Pending |
| SF-OD-002 | Type-alias identity | Core aliases are transparent and do not create nominal identity. | C2.8 | Pending |
| SF-OD-003 | Recursive and unsized boundary | Require finite sizedness for values; admit only explicitly supported unsized pointees behind references/views. | C2.8 | Pending |
| SF-OD-003A | General `Eq`/`Ord`/`Hash` laws | Require equivalence/total-order/hash-consistency laws, define hash stability scope, and explicitly classify behavior under unlawful user implementations. | C2.8/C2.9 | Pending |
| SF-OD-004 | Float equality, ordering, and hashing | Do not imply total `Eq`/`Ord`/`Hash`; specify explicit partial/bitwise operations where provided. | C2.9 | Pending |
| SF-OD-005 | Layout stability | Core v1 has no stable ABI; classify only observably required size/alignment behavior as target-defined. | C2.9 | Pending |
| SF-OD-006 | String units and boundaries | Use UTF-8 byte offsets for low-level boundaries; operations requiring scalar boundaries trap or reject deterministically. | C2.9 | Pending |
| SF-OD-007 | Executable entry contract | Accept a small explicit set of `main` signatures and define return-to-exit mapping. | C2.9 | Pending |
| SF-OD-008A | Package-instance identity | Use relocation-stable logical source identity, canonical package name, exact selected version, and locked content; aliases do not change identity. | C2.9 | Pending |
| SF-OD-008B | Version coexistence | Select at most one version per logical source/name/major line; allow different major lines to coexist visibly. | C2.9 | Pending |
| SF-OD-009 | Constant evaluation | Freeze a bounded, deterministic, side-effect-free subset with explicit cycle and trap behavior. | C2.8 | Pending |
| SF-OD-010 | Future execution boundary | Core v1 is safe and single-threaded; unsafe/FFI remain non-Core and host access requires declared native-provider capabilities. | C2.10 | Pending |
| SF-OD-011 | Resource exhaustion and implementation limits | Classify each failure/limit separately; do not turn host panics into normative STARK traps. | C2.7/C2.9 | Pending |

## Transition artifacts and exit check

This preflight is complete when all of the following exist together:

- this repository-to-plan audit;
- the retained source execution plan under `STARKLANG/docs/compiler/plans/`;
- an explicit roadmap transition between C2.5 and C2.6;
- a non-normative C2.6 completeness-ledger skeleton with fields and domain routing under
  `STARKLANG/docs/compiler/semantic-freeze/`;
- a non-normative open-question and owner-decision register skeleton in that same directory;
- a state record naming WP-C2.6 as next and stating that no semantics were approved here.

WP-C2.6 may now begin. It must expand the skeleton into a row for every independently
observable or rejectable normative question, assign stable granular rule IDs, resolve authority
duplication, and reconcile deviation statuses. No C2.7–C2.11 implementation correction should
start merely because this preflight identified it.
