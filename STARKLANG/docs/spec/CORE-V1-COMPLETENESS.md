# Core v1 Completeness Inventory

Status: **WP-C2.6 skeleton only — not a completed inventory**
Created by: Gate C2 semantic-freeze preflight, 2026-07-18

## Purpose

This will become the authoritative completeness ledger for Core v1. A chapter heading, a broad
conformance rule, or an implementation test does not by itself prove completeness. WP-C2.6
must create one row for every independently observable behavior and every independently
rejectable program condition.

The temporary `PF-*` IDs in this skeleton are audit identifiers, not normative rule IDs.
WP-C2.6 must replace them with stable granular rule IDs or explicitly classify the question as
non-normative.

## Required row schema

Every completed row must contain:

| Field | Required meaning |
| --- | --- |
| Rule ID | Stable, unique Core v1 identifier at independently testable granularity. |
| Domain | One domain from the domain ledger below. |
| Exact normative question | A question answerable without consulting compiler internals. |
| Completeness | `complete`, `partial`, `absent`, `contradictory`, or `non-normative`. |
| Behavior class | `specified`, `implementation-defined`, `target-defined`, `deliberately-unspecified`, or `prohibited`. |
| Normative home | Exactly one source file and section; duplicate prose is a cross-reference. |
| Positive evidence | Function-level test/fixture citations, or `none`. |
| Negative evidence | Function-level rejection/trap citations, `not-applicable` with rationale, or `none`. |
| Compatibility cost | `low`, `medium`, `high`, or `ecosystem-breaking`. |
| Owning WP | C2.7, C2.8, C2.9, C2.10, or C2.11. |
| Decision state | `settled`, `pending-owner-approval`, `blocked`, or `not-required`. |
| Deviation | Applicable DEV ID or `none`. |

## Draft authority map

This is routing input for WP-C2.6, not an approved rewrite.

| Concept | Proposed normative home | Notes |
| --- | --- | --- |
| Source encoding, tokens, literals, comments | `01-Lexical-Grammar.md` | Syntax chapter cross-references token definitions. |
| Grammar, precedence, parse classification | `02-Syntax-Grammar.md` | No runtime or type-system rules. |
| Type identity, well-formedness, inference, traits, coercions | `03-Type-System.md` | Legality only; no physical representation. |
| Static name, flow, initialization, and diagnostic conditions | `04-Semantic-Analysis.md` | Pass/data-structure guidance becomes non-normative. |
| Values, places, moves, temporaries, references, evaluation, destruction | `CORE-V1-ABSTRACT-MACHINE.md` | To be created by C2.7. |
| Memory-safety guarantees | `05-Memory-Model.md` | Summary and cross-references; no stack/pointer/layout promises. |
| Required library API and language hooks | `06-Standard-Library.md` | Profiles and performance guidance must be classified separately. |
| Module syntax/resolution and package contracts | `07-Modules-and-Packages.md` | C2.9 may split package identity into a dedicated normative section. |
| Future-reserved compatibility boundaries | `CORE-V1-FUTURE-BOUNDARIES.md` | To be created by C2.10. |
| Overview and generated combined spec | overview/combined artifacts | Navigation only; never a competing normative authority. |

## Domain audit ledger

| Domain | Initial state | Primary owner |
| --- | --- | --- |
| Source encoding and lexing | Not started | C2.6/C2.11 |
| Grammar and parsing | Not started | C2.6/C2.11/C2.12 |
| Names and scopes | Not started | C2.8/C2.9 |
| Type identity and well-formedness | Not started | C2.8 |
| Inference and coercions | Not started | C2.8 |
| Traits and coherence | Not started | C2.8/C2.9 |
| Ownership and borrowing | Not started | C2.7/C2.8 |
| Places, moves, and temporaries | Not started | C2.7 |
| Patterns and destruction | Not started | C2.7/C2.8 |
| Constants | Not started | C2.8 |
| Numeric behavior | Not started | C2.9 |
| Strings and Unicode | Not started | C2.9 |
| Modules | Not started | C2.9 |
| Packages and package identity | Not started | C2.9 |
| Entry and process behavior | Not started | C2.9 |
| Standard-library language hooks | Not started | C2.8/C2.9 |
| Layout observability | Not started | C2.9 |
| Panic and trap termination | Not started | C2.7/C2.9 |
| Target-defined behavior | Not started | C2.9 |
| Extension boundaries | Not started | C2.10 |

## Preflight seed questions

These rows ensure confirmed gaps are not lost. They are not a substitute for the full C2.6
chapter-by-chapter inventory.

| Audit ID | Domain | Exact question | Initial finding | Cost | Owner | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| PF-001 | Type identity | Does a type alias create identity or transparently denote its target? | Absent | High | C2.8 | Pending |
| PF-002 | Type well-formedness | Which recursive and unsized forms are legal and have finite value size? | Partial | High | C2.8 | Pending |
| PF-003 | Inference | How are expected types, ambiguity, defaulting, and generic normalization resolved? | Partial | High | C2.8 | Pending |
| PF-004 | Traits | How are candidates collected, normalized, selected, and checked for overlap across packages? | Partial | High | C2.8/C2.9 | Pending |
| PF-005 | Abstract execution | What are the value/place contexts and lifetime of every temporary on success and early exit? | Absent | High | C2.7 | Pending |
| PF-006 | Destruction | What is destroyed after partial aggregate, assignment, pattern, or call failure, and in what order? | Partial | High | C2.7 | Pending |
| PF-007 | Constants | What deterministic expression subset is permitted, and how do cycles and traps behave? | Partial | High | C2.8 | Pending |
| PF-008 | Integers | What are signed division/remainder, shift, minimum-negation, and cast semantics? | Partial | High | C2.9 | Pending |
| PF-009 | Floats | What are rounding, NaN, signed-zero, reproducibility, conversion, and trait semantics? | Partial/contradictory | High | C2.9 | Pending |
| PF-010 | Text | Are indices bytes or scalar values, and what happens at invalid UTF-8 boundaries? | Partial | High | C2.9 | Pending |
| PF-011 | Layout | Which size, alignment, discriminant, and address facts are observable? | Contradictory/implementation-leaking | High | C2.9 | Pending |
| PF-012 | Packages | What forms canonical package/public-item identity across versions and sources? | Absent | High | C2.9 | Pending |
| PF-013 | Process | Which `main` signatures are executable and how do result, trap, and streams map to a process? | Absent | High | C2.9 | Pending |
| PF-014 | Standard hooks | Which library names are compiler-recognized language hooks versus ordinary APIs? | Partial | High | C2.8/C2.9 | Pending |
| PF-015 | Future boundary | What safe single-threaded, closure/lifetime, native-provider, capability, and FFI boundary must Core preserve? | Partial/scattered | High | C2.10 | Pending |

## C2.6 completion procedure

1. Inventory every normative sentence and independently observable/rejectable implication.
2. Split broad LEX/SYN/TYP/SEM/MEM/STD/PKG coverage entries into stable rule IDs.
3. Assign exactly one normative home and replace duplicate normative prose with cross-references.
4. Classify every behavior and attach positive and negative evidence.
5. Link all open questions and deviations; do not silently treat implementation behavior as a
   decision.
6. Remove or relocate stale planning and representation-specific implementation prose.
7. Validate the source chapters, generated combined specification, conformance database,
   deviation ledger, state, and roadmap as one evidence set.
