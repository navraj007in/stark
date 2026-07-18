# STARK Core v1 Completeness and Semantic Freeze Execution Plan

Repository status: governance input retained for reproducibility. The preparation baseline below
is historical; current execution status is recorded in `COMPILER-STATE.md`.

Original owner-provided source SHA-256:
`145bc076c28020d5f48793390fda2c57a0241edaa2e777f90f222a3afc0203f8`.
The repository copy preserves the plan while incorporating the preflight correction that makes
package-relocation identity explicitly logical rather than filesystem-location-based.

**Audience:** Claude Code, Codex, or another repository-aware coding agent

**Repository:** `navraj007in/stark`

**Prepared:** 2026-07-18

**Baseline at preparation time:** `27fb3656f2824ee1b7d52f0adf7763d4491fa287` (`complete WP-C2.4 position queries`)

**Recorded test baseline:** 469 passed, 0 failed, 2 ignored

**Current compiler gate:** C2
**Next planned work at preparation time:** WP-C2.5 diagnostics transport

> This document is an execution brief, not a suggestion list. Before making changes, inspect the current repository head and authoritative status files. Do not repeat work already completed after the baseline above.

---

## 1. Mission

Complete the remaining specification-completeness work required to make STARK Core v1 safe to freeze before backend selection, MIR design, native ABI design, and general native compilation.

The objective is not to add every feature found in other programming languages. The objective is to ensure that:

1. every legal Core v1 program has defined typing, ownership, evaluation, destruction, linkage, failure, and process behaviour;
2. every illegal program is rejected under a defined rule;
3. every observable behaviour is classified as specified, implementation-defined, target-defined, deliberately unspecified, or prohibited;
4. no MIR or backend implementation decision accidentally becomes language semantics;
5. future additions such as closures, explicit lifetimes, concurrency, and native providers remain possible without breaking Core v1 foundations;
6. a second implementation could be built from the normative documents without consulting `starkc` behaviour as the source of truth.

The required end state is a frozen semantic foundation suitable for:

```text
STARK source
  -> shared analysis
  -> typed HIR
  -> verified MIR
  -> native backend
  -> standalone executable
```

---

## 2. Non-negotiable constraints

These constraints apply to every work package in this plan.

1. **The normative specification defines the language.** Do not standardise current interpreter behaviour merely because it already exists.
2. **Specification defects are real defects.** When the specification is wrong or incomplete, change the normative source, generated combined specification, implementation, tests, conformance records, and deviation ledger together.
3. **Do not add attractive language features during this audit.** Missing features and missing semantic definitions are different problems.
4. **No C3 backend spike, MIR design, runtime ABI freeze, or native lowering may begin before the semantic-freeze gate closes.**
5. **Representation is not semantics unless deliberately made observable.** HIR structs, Rust enums, interpreter frames, and future MIR instructions must not appear in normative definitions.
6. **Every soundness-relevant rule needs positive and negative executable evidence.**
7. **Do not copy Rust automatically.** Rust, Go, Zig, Swift, C#, Kotlin, Ada, and SPARK are comparators, not normative sources.
8. **Prefer conservative Core v1 rules.** A restrictive sound rule can be relaxed later. A permissive or ambiguous rule is harder to retract.
9. **Do not silently broaden the standard library or package system.** This plan may define contracts and identities, but public API expansion requires separate approval.
10. **Do not implement public unsafe code or general FFI as part of this work.** The preferred long-term boundary is approved native providers with explicit ABI, provenance, and capabilities.
11. **Do not commit, push, open a PR, or rewrite history unless explicitly instructed by the owner.**
12. **Update status files in the same change that alters implementation status.**

---

## 3. Mandatory preflight for every agent session

Before starting a work package:

```bash
git status --short
git log -10 --oneline
```

Read at minimum:

```text
COMPILER-STATE.md
STARKLANG/docs/compiler/COMPILER-CHARTER.md
STARKLANG/docs/compiler/COMPILER-ROADMAP.md
STARKLANG/docs/compiler/reference-execution.md
starkc/docs/conformance/KNOWN-DEVIATIONS.md
STARKLANG/conformance/core-v1-coverage.toml
STARKLANG/docs/spec/00-Core-Language-Overview.md
STARKLANG/docs/spec/01-Lexical-Grammar.md
STARKLANG/docs/spec/02-Syntax-Grammar.md
STARKLANG/docs/spec/03-Type-System.md
STARKLANG/docs/spec/04-Semantic-Analysis.md
STARKLANG/docs/spec/05-Memory-Model.md
STARKLANG/docs/spec/06-Standard-Library.md
STARKLANG/docs/spec/07-Modules-and-Packages.md
```

Then determine:

- the actual current head;
- the actual next work package;
- whether WP-C2.5 or any later work in this plan has already been completed;
- whether test counts and state metadata have drifted;
- whether there are uncommitted owner changes that must be preserved.

Do not overwrite or discard unrelated work.

---

## 4. Revised Gate C2 roadmap

The intended sequence is:

```text
WP-C2.5   Structured diagnostics transport
WP-C2.6   Core completeness inventory and specification authority
WP-C2.7   Abstract machine and execution semantics
WP-C2.8   Type, trait, pattern, and constant semantics
WP-C2.9   Numeric, layout, text, process, and package contracts
WP-C2.10  Future-extension compatibility boundaries
WP-C2.11  Implementation alignment and adversarial conformance
WP-C2.12  Differential interpreter corpus
WP-C2.13  Gate C2 exit and Core v1 semantic-freeze decision
```

The existing differential corpus and Gate C2 exit work must be renumbered accordingly when the roadmap is amended.

### Mandatory ordering

```text
decide semantics
  -> update normative specification
  -> align interpreter/compiler
  -> add positive and negative evidence
  -> build differential corpus
  -> freeze Gate C2
  -> begin C3
```

Do not build a broad differential corpus around behaviour that has not yet been intentionally specified.

---

## 5. Cross-cutting deliverables

The following files should exist by the end of this plan:

```text
STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md
STARKLANG/docs/spec/CORE-V1-ABSTRACT-MACHINE.md
STARKLANG/docs/compiler/semantic-freeze/CORE-V1-OPEN-QUESTIONS.md
STARKLANG/docs/spec/CORE-V1-FUTURE-BOUNDARIES.md
starkc/docs/compiler/C2-exit-report.md
```

The exact file names may be adjusted to match repository naming conventions, but the responsibilities must remain separate.

Also update as required:

```text
COMPILER-STATE.md
STARKLANG/docs/compiler/COMPILER-ROADMAP.md
STARKLANG/docs/compiler/COMPILER-CHARTER.md
STARKLANG/docs/spec/00-Core-Language-Overview.md
STARKLANG/docs/spec/01-Lexical-Grammar.md
STARKLANG/docs/spec/02-Syntax-Grammar.md
STARKLANG/docs/spec/03-Type-System.md
STARKLANG/docs/spec/04-Semantic-Analysis.md
STARKLANG/docs/spec/05-Memory-Model.md
STARKLANG/docs/spec/06-Standard-Library.md
STARKLANG/docs/spec/07-Modules-and-Packages.md
STARKLANG/docs/spec/STARK-Core-v1.md
starkc/docs/conformance/KNOWN-DEVIATIONS.md
STARKLANG/conformance/core-v1-coverage.toml
```

If `STARK-Core-v1.md` or HTML output is generated, update it through the repository’s generator rather than editing generated output by hand.

---

# WP-C2.5 — Structured diagnostics transport

## Objective

Create one structured compiler-wide diagnostic representation used by CLI text output, JSON output, LSP publication, and future native/MIR diagnostics.

## Required diagnostic fields

```text
stable diagnostic code
severity
message
primary SourceId and span
related SourceId/span entries
notes
help
source version
optional Core rule ID
optional deviation ID
deterministic JSON representation
```

## Requirements

- Cross-file related spans must be supported.
- Source identity must retain file, module, package, and extension provenance.
- CLI text and LSP diagnostics must derive from the same structured form.
- JSON ordering and encoding must be deterministic.
- The transport format may stabilise without freezing every current diagnostic code.
- Existing diagnostic code collisions must be repaired or explicitly assigned to WP-C2.11.
- No semantic behaviour should change merely to simplify diagnostic transport.

## Exit criteria

- CLI and LSP consume the same diagnostic structure.
- JSON output has golden tests.
- Cross-file related-span tests pass.
- Existing diagnostic behaviour remains equivalent except for approved code cleanup.
- `COMPILER-STATE.md` records the actual committed head and the next WP.

---

# WP-C2.6 — Core completeness inventory and specification authority

## Objective

Find every incomplete, contradictory, duplicated, accidental, or absent Core v1 rule before making further semantic decisions.

This work package primarily inventories and classifies. It should not mix every semantic repair into one uncontrolled change.

## Deliverable A: `CORE-V1-COMPLETENESS.md`

Create a structured matrix with at least these fields:

| Field | Required content |
|---|---|
| Domain | Numeric, types, ownership, modules, etc. |
| Question | One exact normative question |
| Status | Defined, incomplete, contradictory, duplicated, or absent |
| Behaviour class | Specified, implementation-defined, target-defined, deliberately unspecified, or prohibited |
| Normative home | Exact document, section, and rule ID |
| Implementation evidence | Source and test citations |
| Negative evidence | Rejected or dangerous case |
| Compatibility cost | Low, medium, high, or effectively irreversible |
| Owning WP | C2.7 through C2.11 |
| Decision | Pending or approved |

## Domains that must be audited

```text
source encoding and lexing
grammar and parsing
names, scopes, shadowing, and visibility
type identity and type well-formedness
inference, coercions, and casts
traits, coherence, and method selection
ownership, borrowing, and lifetimes
places, moves, copies, temporaries, and reinitialisation
patterns, exhaustiveness, moves, and destruction
constant evaluation
integer and floating-point behaviour
strings, Unicode, and indexing units
modules and imports
packages, lock identity, and package nominal identity
program entry and process behaviour
standard-library language hooks
layout observability
panic, trap, abort, and exit semantics
target-defined behaviour
extension and native-provider boundaries
```

## Deliverable B: normative authority map

Give each concept one authoritative normative home.

Recommended ownership:

```text
03-Type-System.md
  type identity
  inference
  coercions
  traits
  associated types

CORE-V1-ABSTRACT-MACHINE.md
  values
  places
  storage
  moves
  temporaries
  references
  evaluation
  drops
  traps

04-Semantic-Analysis.md
  legality checks
  control-flow analysis
  exhaustiveness
  diagnostics categories

07-Modules-and-Packages.md or split replacement documents
  module scopes and visibility
  package identity and resolution

06-Standard-Library.md
  public API contracts only
  not implementation representation
```

Where rules are duplicated, retain one normative definition and replace copies with references.

## Deliverable C: granular rule IDs

The current broad conformance rules are insufficient for semantic freezing. Introduce stable granular rule identifiers such as:

```text
NUM-INT-DIV-001
NUM-SHIFT-003
NUM-FLOAT-NAN-002
TYPE-ALIAS-001
TYPE-RECURSION-002
TRAIT-COHERENCE-004
PAT-MOVE-004
DROP-TEMP-003
CONST-CYCLE-001
TEXT-BOUNDARY-001
PROC-MAIN-002
PKG-IDENTITY-002
```

Each independently observable behaviour or independently rejectable legality rule should have its own rule ID.

## Deliverable D: `CORE-V1-OPEN-QUESTIONS.md`

For every unresolved question, record:

```text
question
decision required now or later
why it matters
cost of deciding later
recommended default
alternatives
compatibility impact
owning work package
owner approval status
```

## Exit criteria

- Every domain has been inspected.
- No unresolved question remains hidden in explanatory prose.
- Every gap has a compatibility cost and an owning WP.
- Status prose and implementation planning are removed from normative documents or clearly marked non-normative.
- No implementation behaviour is declared normative merely because it exists.

---

# WP-C2.7 — Abstract machine and execution semantics

## Objective

Define language-level execution independently of HIR, Rust implementation structures, interpreter frames, or future MIR.

## Required document

```text
STARKLANG/docs/spec/CORE-V1-ABSTRACT-MACHINE.md
```

## Required concepts

### Values and storage

Define:

```text
value
object
storage location
allocation
owner
local binding
temporary
place
projection
initialised state
moved-from state
```

### Evaluation

Define:

- value context versus place context;
- expression evaluation;
- place resolution;
- temporary creation;
- temporary lifetime and destruction point;
- evaluation order;
- partial evaluation;
- early return, `break`, `continue`, and `?` propagation;
- trap during subexpression evaluation.

### Moves and copies

Define:

- when a place read copies;
- when a place read moves;
- moves into function parameters;
- moves into aggregate fields;
- return-value ownership transfer;
- partial moves;
- reinitialisation;
- moves through pattern bindings;
- prohibited moves from indexed places.

### Assignment and replacement

Settle the exact abstract sequence for:

```text
evaluate right-hand side
resolve destination place
preserve or extract old destination value
write new value
destroy replaced value at a defined point
```

Also define the outcome if destination-place resolution traps after the right-hand side has already produced an owned value.

### Aggregate construction

Define:

- field and element evaluation order;
- ownership transfer into completed fields;
- destruction of completed fields when a later field traps;
- enum payload construction;
- array-repeat construction;
- struct update if later introduced;
- duplicate and missing field rejection.

### Destruction

Define:

- local destruction;
- temporary destruction;
- field destruction;
- collection-owned element destruction;
- loop-variable destruction per iteration;
- explicit `drop`;
- replacement assignment;
- partial-move drop flags;
- function argument destruction;
- return-value destruction;
- normal return versus aborting trap.

The existing exactly-once Drop rule remains mandatory.

### References

Define:

- reference identity;
- derivation from places;
- projections through references;
- auto-borrow and auto-dereference;
- returned references;
- method receiver references;
- references to slices;
- borrow-carrying generic values;
- validity after frame exit;
- reference behaviour through moves of the owner.

### Observable behaviour

Define the comparator later used by HIR, MIR, and native execution:

```text
stdout bytes
stderr bytes
exit-status category
returned value where harnessed
trap category
trap source location
observable Drop order
artifact verification result
```

## Required adversarial examples

At minimum include examples for:

- assignment where both sides have side effects;
- trap during destination index evaluation;
- trap during aggregate field construction;
- returned reference from a method;
- returned range slice from a method;
- moved and reinitialised locals;
- pattern wildcard over a Drop type;
- loop variable with Drop on `continue`, `break`, and normal iteration;
- collection clear/remove/destruction;
- panic/trap with live Drop values.

## Exit criteria

- Every legal expression form has defined execution behaviour.
- No rule depends on interpreter frame layout.
- Every previous runtime deviation can be stated as a violation of a named abstract-machine rule.
- C4 MIR can be designed as an implementation of this document rather than as a semantic authority.

---

# WP-C2.8 — Type, trait, pattern, and constant semantics

## Part A: type identity and well-formedness

### Type aliases

Recommended rule:

> `type Alias = ExistingType` creates a transparent alias, not a new nominal type.

Reserve a separate future newtype facility for nominal wrappers.

### Nominal type identity

Define nominal identity using:

```text
package identity
+ module path
+ item name
+ generic arguments
```

Two structurally identical structs or enums remain distinct types.

### Recursive types

Reject direct or indirect infinitely sized recursive values unless recursion passes through an approved indirection such as `Box`.

Examples:

```stark
struct Invalid {
    next: Invalid
}
```

must fail, while:

```stark
struct Valid {
    next: Option<Box<Valid>>
}
```

may succeed.

### Sizedness

Recommended Core v1 rule:

> All user-defined types and generic parameters are sized. Only implementation-provided `str` and `[T]` are unsized, and they may appear only behind references.

Explicitly decide whether `Box<[T]>` is legal in Core v1. If not needed, prohibit it now rather than leaving it ambiguous.

### Edge types

Define:

- zero-length arrays;
- empty structs;
- zero-variant enums;
- maximum tuple arity;
- maximum array length;
- inhabitation of `!`;
- whether `Unit` and `()` are exactly the same type.

## Part B: inference and coercion

Specify:

- integer and float literal defaulting;
- constraint generation;
- expected-type propagation;
- ambiguity;
- generic argument inference;
- associated-type normalization;
- inference failure;
- whether later uses of a local may influence its inferred type;
- when explicit turbofish is required;
- coercion ordering;
- error recovery expectations where observable through diagnostics.

## Part C: traits and coherence

Define an independent algorithm for:

- applicable implementation selection;
- generic substitution;
- associated type binding and normalization;
- blanket implementation overlap;
- orphan rules;
- duplicate implementations across packages;
- duplicate implementations across package major versions;
- inherent versus trait method priority;
- trait-qualified protocol operations such as `Eq::eq`, `Ord::cmp`, and `Iterator::next`;
- trait-associated functions such as `From::from`;
- ambiguity diagnostics.

Core v1 should not add specialization, negative impls, or trait objects during this work.

Define the semantic laws of `Eq`, `Ord`, and `Hash` independently of compiler-provable
well-formedness: `Eq` equivalence, total `Ord` consistent with `Eq`, equal-values-equal-hash,
the scope of hash stability, and the guarantee (or lack of one) when user implementations
violate those laws.

## Part D: pattern ownership

Create a dedicated normative pattern-ownership section covering:

- whether each binding copies, moves, or borrows;
- wildcard ownership and destruction;
- partial moves through struct and enum patterns;
- matching reference scrutinees;
- nested patterns;
- arm-local binding lifetime;
- scrutinee destruction;
- unmatched and unbound field destruction;
- array and tuple patterns;
- unreachable and subsumed arms;
- behaviour when a pattern test itself requires equality.

## Part E: constant evaluation

Define a bounded Core v1 constant subset.

Recommended allowed forms:

```text
literals
references to earlier constants
tuples and arrays of constant values
struct and enum construction
primitive arithmetic and comparison
if expressions
valid primitive casts
```

Recommended initially prohibited forms:

```text
I/O
heap-backed mutable collections
ordinary function calls
trait dispatch
loops
runtime state
artifact access
native-provider access
```

Specify:

- evaluation order;
- overflow;
- invalid cast behaviour;
- division by zero;
- cycle detection;
- cross-package constant dependencies;
- whether constants have addressable storage identity;
- deterministic float evaluation if floats are permitted in constants.

## Exit criteria

- Type alias identity is explicit.
- Recursive type legality is deterministic.
- Trait selection can be independently implemented.
- Pattern ownership can be lowered without guessing.
- Constant legality is deterministic.
- Package linkage and future monomorphisation have stable nominal type identities.

---

# WP-C2.9 — Numeric, layout, text, process, and package contracts

## Part A: complete integer semantics

Define for every integer type:

- overflow and underflow;
- unary negation of the minimum signed value;
- division rounding direction;
- remainder sign and algebraic relationship to division;
- division and modulo by zero;
- shift operand type;
- negative shift handling;
- shift amounts equal to or larger than bit width;
- signed right-shift behaviour;
- exponentiation operand rules;
- negative exponent handling;
- exponentiation overflow;
- cast success and failure;
- debug/release invariance.

Recommended defaults:

```text
signed division rounds toward zero
remainder has the dividend's sign and satisfies a = (a / b) * b + (a % b)
invalid shift counts trap
signed right shift is arithmetic
all overflow traps in every profile
```

## Part B: complete floating-point semantics

Define:

- IEEE-754 formats used by `Float32` and `Float64`;
- rounding mode;
- division by zero;
- NaN propagation;
- signed zero;
- primitive comparisons;
- casts;
- conversion rounding;
- whether fused multiply-add contraction is allowed;
- reproducibility requirements across targets.

### Float `Eq`, `Ord`, and `Hash`

Recommended Core v1 decision:

> Floating-point types do not implement `Eq`, `Ord`, or `Hash`. They retain primitive IEEE comparison operators. A future total-order wrapper may implement the traits.

This avoids violating ordinary equality and ordering laws because of NaN and signed zero.

If a different decision is chosen, define total equality, total ordering, and hash canonicalisation explicitly.

## Part C: layout observability

Because `size_of<T>()` and `align_of<T>()` are public, define their contract.

Recommended rule:

> Layout is target- and compiler-defined unless an explicit future representation contract is applied.

Specify:

- `size_of` and `align_of` are target-dependent;
- Core v1 promises no stable cross-compiler ABI;
- references and slices have implementation-defined physical representation;
- struct field access and Drop order are semantic, while physical layout is not automatically stable;
- enum niche optimisation is permitted only within the target/compiler-defined contract;
- packages must not infer raw layout beyond defined queries;
- C5 runtime ABI documents implementation layout without turning it into permanent language semantics.

Resolve any normative prose that currently claims fixed pointer sizes, stack placement, or “optimal” enum layout.

## Part D: strings and Unicode

Recommended contract:

```text
source files are valid UTF-8
String and str contain valid UTF-8
Char is a Unicode scalar value
len, find, and substring indices use UTF-8 byte offsets
slice boundaries must be valid scalar boundaries
grapheme clusters are outside Core v1
chars iterates Unicode scalar values
```

Define:

- invalid boundary behaviour;
- out-of-range substring behaviour;
- whether APIs trap or return `Result`;
- `find` return units;
- `trim` definition;
- case-conversion Unicode versioning;
- whether case conversion may expand one scalar into multiple scalars;
- escape validity;
- whether `\xNN` is restricted to ASCII in UTF-8 string and character literals.

## Part E: executable entry point and process model

Define exactly which entry signatures are legal.

Possible minimal choice:

```stark
fn main()
fn main() -> Int32
```

A richer option may include:

```stark
fn main() -> Result<Unit, E> where E: Display
```

Do not support a form unless its lowering, diagnostics, and exit mapping are specified.

Specify:

- exactly one executable entry point;
- library-package behaviour;
- duplicate or inaccessible `main` rejection;
- return-to-exit-code mapping;
- panic/trap exit category;
- standard stream encoding;
- argument and environment access through standard capability packages;
- startup and shutdown;
- destructor behaviour on normal completion and abort.

## Part F: package and public API identity

Separate language modules from package resolution if practical. At minimum, distinguish their normative sections clearly.

Define canonical package identity using relocation-stable logical sources rather than absolute
checkout paths:

```text
logical source or registry identity
+ package name
+ major-version line
+ exact selected version
+ content hash
```

Required rules:

- registry identity is the registry plus package name/version;
- repository identity is canonical origin plus revision plus package path;
- path dependency identity is workspace dependency identity plus manifest package identity,
  never an absolute filesystem path;
- one selected version per compatible `(logical source, name, major-version line)`;
- incompatible major versions may coexist;
- coexistence must be visible in lockfiles, dependency trees, diagnostics, and update summaries;
- application policy may deny multiple majors globally or selectively;
- package version/source participates in nominal type identity;
- source substitution cannot occur silently;
- relocating a complete workspace without changing manifests, lock data, or logical package
  sources does not change package or item identity;
- exact content is locked;
- import aliases are explicit;
- no automatic transformation from manifest name `tensor-lib` to source identifier `TensorLib`;
- private types must not leak through public signatures unless an explicit rule permits it;
- public re-exports are well-defined;
- package-level coherence is deterministic.

Operational public-registry machinery remains deferred. Do not implement MFA, reputation scoring, typosquat detection, advisory services, or quarantine here.

## Part G: resource exhaustion and implementation limits

Classify allocation failure, stack exhaustion, recursion/call-depth limits, maximum object,
array, source-file and package sizes, host I/O failures, operating-system termination, and
failures outside defined STARK traps. Each must be specified, implementation-defined,
target-defined, or outside language guarantees. A host-language panic is not automatically a
normative STARK trap.

## Exit criteria

- Primitive programs cannot vary accidentally between interpreter and backend.
- Float trait behaviour is explicit.
- Layout queries do not silently promise a stable ABI.
- String index units are unambiguous.
- `stark build` has a defined entry and process contract.
- Package major coexistence has unambiguous nominal type identity.

---

# WP-C2.10 — Future-extension compatibility boundaries

## Objective

Ensure Core v1 does not make likely future features impossible. Do not implement the features in this WP.

## Required document

```text
STARKLANG/docs/spec/CORE-V1-FUTURE-BOUNDARIES.md
```

## Closures

Record an intended ownership-aware callable model equivalent in capability to:

```text
call once
call mutably
call repeatedly through shared access
```

The final trait names need not copy Rust, but current iterator APIs should remain compatible with future capturing callables.

Document:

- capture by move;
- capture by immutable borrow;
- capture by mutable borrow;
- callable ownership and Drop;
- interaction with generic function parameters.

## Explicit lifetimes and reference fields

Document how future lifetime parameters may extend rather than contradict the conservative Core v1 rules.

Reserve design space for:

- lifetime arguments;
- reference fields;
- variance;
- associated types carrying lifetimes;
- higher-ranked bounds;
- lifetime erasure at native ABI boundaries.

## Concurrency

State explicitly:

> Core v1 execution is single-threaded unless an approved extension states otherwise.

Do not claim the current borrow system proves all data-race freedom across threads.

Record future requirements:

```text
atomics and memory ordering
thread-safety marker traits
shared ownership
synchronisation
thread-local storage
cross-thread Drop behaviour
panic propagation
concurrent native providers
```

## Native providers and FFI

Adopt the strategic boundary unless the owner decides otherwise:

> Core v1 exposes no public unsafe code or general FFI. Host access occurs through approved native-provider interfaces with explicit ABI, provenance, target support, and capability metadata.

This must remain compatible with static derivation of capabilities for pure STARK packages.

## Metaprogramming

Record that macros and compile-time code generation are not required for Core v1 and must not be added merely to compensate for package/build tooling gaps.

## Dynamic dispatch

Trait objects remain outside Core v1. Reserve a viable syntax and object-safety design space, but do not add runtime vtables during this work.

## Exit criteria

- Current syntax and APIs do not block plausible closures or explicit lifetimes.
- Core v1 guarantees are scoped to safe, single-threaded execution.
- Native providers have a documented strategic boundary.
- No new public language feature is implemented.

---

# WP-C2.11 — Implementation alignment and adversarial conformance

## Objective

Implement every approved semantic decision and repair every affected compiler/interpreter path.

## Atomic change rule

For each semantic correction, update together:

```text
normative source specification
generated combined specification
compiler/interpreter implementation
diagnostic catalogue
positive test
negative test
conformance database
known-deviation ledger
state and roadmap
```

Do not leave a specification decision without executable evidence or leave implementation changes undocumented.

## Required adversarial test groups

### Numeric matrix

Cover every primitive type and boundary:

```text
minimum and maximum values
overflow and underflow
unary negation
division
remainder
shifts
exponentiation
casts
NaN
infinity
signed zero
float comparisons
```

### Type identity and well-formedness

Test:

- transparent aliases;
- nominal struct/enum identity;
- direct recursive rejection;
- indirect recursive rejection;
- legal boxed recursion;
- sizedness;
- zero-length arrays;
- empty structs;
- zero-variant enums if permitted;
- package-major type distinction.

### Drop and temporaries

Test:

- replacement assignment;
- trap during destination-place resolution;
- partial aggregate construction;
- method return temporaries;
- returned slice references;
- loop iteration variables;
- `continue`, `break`, and early return;
- pattern wildcards;
- collection clear/remove/destruction;
- normal return;
- aborting trap.

### Patterns

Test:

- Copy bindings;
- move bindings;
- reference scrutinees;
- nested partial moves;
- wildcard ownership;
- unreachable arms;
- unmatched field destruction;
- arm-local binding destruction.

### Constants

Test:

- every permitted expression form;
- overflow;
- division by zero;
- invalid casts;
- cycle detection;
- forbidden calls;
- cross-package references;
- deterministic results.

### Unicode

Test:

- ASCII;
- two-, three-, and four-byte scalars;
- invalid scalar boundaries;
- `find` offsets;
- escape validation;
- case expansion;
- trimming semantics.

### Packages

Test:

- same-major unification;
- incompatible-major coexistence;
- duplicate-major policy;
- source identity;
- lockfile determinism;
- explicit aliases;
- path dependency identity;
- private-in-public rejection;
- nominal identity across versions.

### Entrypoints

Test every permitted and rejected `main` signature and every exit mapping.

## Exit criteria

- No confirmed semantic defect remains unowned.
- Every high-cost decision has positive and negative evidence.
- Granular conformance rules cover the frozen surface.
- Generated specifications match normative sources.
- The interpreter conforms or every remaining deviation is explicitly approved and blocks later work where necessary.

---

# WP-C2.12 — Differential interpreter corpus

## Objective

Build the semantic corpus only after the rules above are intentionally settled and implemented.

## Required coverage

```text
every expression and statement form
every primitive operation
structs and enums
generics
traits and associated types
methods and protocol dispatch
ownership and Drop edge cases
Option and Result propagation
collections and iterators
multi-file execution
multi-package execution
deterministic output
deterministic trap/failure behaviour
```

## Required metamorphic transformations

At minimum:

```text
alpha-renaming
harmless extra block scopes
explicit versus inferred generic arguments
equivalent trait-qualified calls
field shorthand versus explicit field initialisation
equivalent non-overlapping match-arm order
relocation of an entire workspace without changing manifests, lock data, or logical package
sources
```

## Corpus output contract

The corpus must produce stable records later consumable by:

```text
HIR interpreter
MIR interpreter
native debug build
native release build
```

Record:

```text
stdout
stderr
exit status
returned value where applicable
trap category
source location
observable Drop order
```

## DEV-036

Close filename-based missing-module suppression using an explicit test-harness/conformance input mode and permanent collision regressions for ordinary user paths containing names such as `STARKLANG`, `test.stark`, or `spec-fixtures`.

## Exit criteria

- Every frozen Core rule with executable behaviour is represented.
- Metamorphic tests are deterministic.
- The corpus can be reused without redesign by C4 and C5.
- No test merely codifies an unexplained interpreter quirk.

---

# WP-C2.13 — Gate C2 exit and Core v1 semantic freeze

## Required report

```text
starkc/docs/compiler/C2-exit-report.md
```

## Required conclusion

Use exactly one:

```text
CORE-V1-SEMANTIC-FOUNDATION-FROZEN
CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS
CORE-V1-SEMANTIC-FOUNDATION-NOT-YET-FROZEN
```

## Gate questions

The report must answer:

1. Can a second implementation be written from the normative documents without consulting `starkc` behaviour?
2. Does every observable behaviour have one of these classifications?

   ```text
   specified
   implementation-defined
   target-defined
   deliberately unspecified
   prohibited
   ```

3. Are all MIR-relevant concepts defined independently of MIR?
4. Are all high-cost future decisions settled or protected by explicit boundaries?
5. Does the interpreter pass the frozen differential corpus?
6. Are package identity and executable entry behaviour defined?
7. Are remaining deviations listed, non-silent, and assigned?
8. Are all soundness-relevant rules backed by negative tests?
9. Do generated specification documents match their normative sources?
10. Does the project state accurately identify the current committed head and next gate?

## Gate rule

C3 may open only with one of the two frozen outcomes.

If the result is `NOT-YET-FROZEN`, record blockers and keep Gate C2 open. Do not reinterpret native compilation as optional.

---

## 6. Priority tiers

### Tier 1 — must settle before C3

```text
abstract machine
numeric edge semantics
type identity
recursive type legality
pattern ownership
temporary and Drop semantics
constant evaluation
entry-point model
trait selection and coherence
package nominal identity
```

### Tier 2 — must settle before C4 MIR design

```text
layout observability
reference identity
aggregate-construction failure
string indexing
public API leakage
target-defined behaviour
```

### Tier 3 — future-proof design only

```text
capturing closures
explicit lifetime syntax
concurrency
general FFI
macros
dynamic dispatch
```

Tier 3 does not require implementation, but it requires documented extension boundaries before Core v1 is described as stable.

---

## 7. Required validation commands

Run from the appropriate repository directory and adapt only if the repository’s authoritative scripts differ.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
python scripts/check-conformance.py
```

Also run the conformance report generator when present, for example:

```bash
python scripts/generate-conformance-report.py
```

Run spec generation/checking commands defined by the repository whenever normative Markdown changes.

For each WP, record:

```text
exact command
pass/fail result
test count
ignored test count
new test files/functions
known environment limitations
current commit or working-tree baseline
```

Do not claim independent CI success unless a workflow run or status check was actually observed.

---

## 8. Work-package implementation protocol

For each WP:

1. Inspect the current head and state.
2. Produce a bounded change plan.
3. Identify owner decisions required before coding.
4. Avoid unrelated cleanup.
5. Implement the smallest coherent change.
6. Add positive and negative tests.
7. Run focused tests first.
8. Run the complete validation suite.
9. Update conformance and deviation records.
10. Update `COMPILER-STATE.md` and roadmap status.
11. Report changed files, decisions, evidence, deviations, and remaining blockers.
12. Do not commit unless explicitly instructed.

---

## 9. Decision and escalation protocol

Stop and request an owner decision when:

- two plausible choices have materially different compatibility or adoption effects;
- a change would add new public syntax or semantics;
- a decision would permanently constrain native ABI or package identity;
- the normative documents contradict each other and no existing recorded decision resolves them;
- a standard-library API must change to restore soundness;
- a chosen rule would reject significant existing conforming programs;
- a future feature boundary must be selected rather than merely documented;
- a finding could invalidate the current Core front-end conformance claim.

Do not stop merely because implementation is complex. Continue with bounded analysis and present a concrete recommendation with alternatives and tradeoffs.

---

## 10. Explicit non-goals

This plan does not authorise:

```text
async/await
capturing closures
macros
unsafe blocks
raw pointers
general C FFI
trait objects
garbage collection
actors
threads or atomics
exception unwinding
reflection
public registry operations
broad standard-library expansion
new tensor or AI syntax
MIR implementation
native backend implementation
```

A future-boundary document may discuss these topics, but implementation requires separate approval.

---

## 11. Definition of done

This plan is complete only when:

- WP-C2.5 through WP-C2.13 are closed or renumbered equivalents are closed;
- the normative specification has one authoritative home per concept;
- every observable behaviour is classified;
- abstract values, places, moves, references, temporaries, Drop, and traps are defined independently of implementation IRs;
- integer and floating-point edge semantics are complete;
- type identity, recursive types, sizedness, aliases, and traits are independently implementable;
- pattern ownership and constant evaluation are explicit;
- strings and Unicode indexing have defined units;
- layout observability does not accidentally freeze a permanent ABI;
- executable entry and process behaviour are defined;
- package and public API identity are stable enough for multi-major coexistence;
- future closures, lifetimes, concurrency, and native providers have viable documented boundaries;
- compiler and interpreter behaviour align with approved rules;
- adversarial positive and negative tests exist;
- the differential corpus is reusable by HIR, MIR, and native execution;
- the C2 exit report records a frozen outcome;
- Gate C3 has not opened prematurely.

---

## 12. Suggested bounded change sequence

This is a recommended sequence of reviewable changes, not permission to commit automatically.

```text
Change 1: WP-C2.5 diagnostic transport
Change 2: roadmap amendment and completeness document skeletons
Change 3: WP-C2.6 full inventory and rule-ID plan
Change 4: WP-C2.7 abstract machine decisions
Change 5: WP-C2.8 type/trait/pattern/const decisions
Change 6: WP-C2.9 numeric/layout/text/process/package decisions
Change 7: WP-C2.10 future-boundary document
Change 8+: WP-C2.11 implementation corrections grouped by semantic domain
Change N: WP-C2.12 differential corpus
Final change: WP-C2.13 exit report and state update
```

Do not combine all semantic decisions and all implementation repairs into one unreviewable commit.

---

## 13. Initial instruction to the coding agent

Use the following as the first execution prompt after providing this file:

> Read this execution plan and the repository’s current `COMPILER-STATE.md`, `COMPILER-CHARTER.md`, and `COMPILER-ROADMAP.md`. Inspect the current head and determine which listed work packages are already complete. Do not modify code yet. Produce a gap report mapping the current repository to this plan, identify contradictions or already-resolved items, propose the exact roadmap amendment, and list any owner decisions required before implementation. Preserve all uncommitted work and do not commit or push.

After the gap report is approved, execute one work package at a time.

---

## 14. Final governing statement

> Core v1 is ready for MIR and native compilation only when its semantics are intentional rather than inferred from the current interpreter, every observable behaviour is classified, and a second implementation could reproduce the same legal programs, failures, ownership effects, and externally visible results from the normative specification alone.
