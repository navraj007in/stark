# WP-C6-ENTRY — Native Semantic Parity and Cross-Platform Runtime

**Status:** APPROVED (owner, 2026-07-23, CD-079) — Gate C5 CLOSED (CD-077); three-track parallel execution on `main` (§7C branch/worktree model waived by owner)  
**Prepared:** 2026-07-23  
**Repository baseline inspected:** `e94e76089e85bd12c5373d6517622d0507b03bc7`  
**Gate:** C6 — Native Semantic Parity and Cross-Platform Runtime  
**Selected backend:** generated Rust consuming verified STARK MIR  
**Tier 1:** `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`  
**Tier 2:** `x86_64-pc-windows-msvc` for Core v1 Compiler Stable  
**Next after closure:** Gate C7 — Build Profiles, Reproducibility, and Evidence-Gated Optimisation

---

## 0. Directive

Gate C5 proves that one bounded multi-file, multi-package Core application can become a standalone native debug executable.

Gate C6 must prove that the **executable Core v1 surface** preserves:

- ownership and move/copy semantics;
- partial moves and exact Drop behaviour;
- static generic and trait dispatch;
- String, collections, slices, iterators, Box, formatting, and file/resource behaviour;
- traps, output, status, and source provenance;
- identical semantics on macOS-arm64 and Linux-x64;
- agreement between HIR, MIR, and native debug execution;
- both hand-written and deterministic generated differential coverage.

Gate C6 is a parity gate, not a language-expansion gate.

It does not add classes, inheritance, trait objects, closures, async, concurrency, reflection, macros, unsafe STARK source, general FFI, release optimisation, a package registry, networking, HTTP, or a new backend.

---

## 1. Opening conditions

No C6 implementation work may be marked open until (all satisfied 2026-07-23 at Gate C5 closure `db73afe`/CD-077):

- [x] WP-C5.5 is formally closed. (CD-076)
- [x] WP-C5.6 is complete. (CD-077)
- [x] Gate C5 has an owner-approved exit report. (`starkc/docs/compiler/C5-exit-report.md`, CD-077)
- [x] The C5 exit report records the exact native supported subset. (exit report §2)
- [x] Every deterministic C5 refusal is listed. (exit report §3)
- [x] Every C6-deferred boundary is listed. (exit report §3; re-pinned in `C6-INTEGRATION-LEDGER.md` §1)
- [x] The C5 full suite is green at the exact closure commit. (59 binaries / 1098 tests, `db73afe`)
- [x] No known C5-supported program silently miscompiles. (exit report §5)
- [x] No unexplained HIR/MIR/native divergence remains. (exit report §5)
- [x] This entry plan or an amended replacement is owner-approved. (CD-079)

The C5 handoff packet must include:

```text
closure commit
C5 exit report
supported/unsupported feature matrix
frozen three-package workspace
three-engine snapshot version and hashes
native Drop fixture
MIR version
MIR runtime-surface version
runtime version
backend version
target-layout identity
toolchain versions
open deviation list
C6 deferral list
```

Do not reconstruct the handoff from commit messages.

---

## 2. Baseline entering C6 preparation

At the inspected baseline:

- generated Rust is the production backend;
- backend entry requires `VerifiedMirProgram`;
- concrete generics, function values, indirect calls, and multi-package linkage exist;
- `stark build` integration exists but Gate C5 is not yet formally closed;
- `stark-runtime` remains a deliberately small C5 runtime;
- C5 defers broad String, Vec, slice, Box, iterator, HashMap/HashSet, formatting, and file-resource parity;
- C5.3 explicitly defers:
  - multi-unit enum-payload partial moves;
  - wider non-`Copy` cross-block moves;
- non-`Copy` by-value array iteration and a generic-impl receiver-inference limitation must be rechecked;
- WP-C2.12 still leaves the deterministic generated corpus and full cross-backend replay to C6.5.

Re-pin this inventory at the C5 closure commit.

---

## 3. Gate outcome

C6 closes only when:

> Every executable normative Core v1 feature required by the Core runtime profile executes natively with the same observable semantics as the HIR and MIR interpreters on both Tier-1 targets.

Required results:

1. full ownership and Drop parity;
2. full static generic/trait parity;
3. full normative Core runtime-value parity;
4. macOS-arm64 and Linux-x64 qualification;
5. full hand-written and generated differential replay;
6. no C5-style unsupported profile remaining for normative executable Core;
7. a CE8-ready C6 exit report.

C6 does not deliver:

- release builds;
- optimisation;
- incremental compilation;
- reproducible-binary claims;
- Windows stable support unless separately completed;
- networking/TLS/HTTP/database/GUI packages;
- dynamic plugins or runtime DI containers;
- trait objects;
- Core v1 Compiler Stable.

---

## 4. Closure vocabulary

The C6 exit report must choose one:

```text
NATIVE-CORE-TIER1-SEMANTIC-PARITY
NATIVE-CORE-TIER1-SEMANTIC-PARITY-WITH-LISTED-NON-SEMANTIC-DEVIATIONS
NATIVE-CORE-TIER1-PARITY-NOT-YET
```

A listed-deviations outcome may contain only non-semantic limitations such as:

- Tier-2/Tier-3 targets;
- documentation gaps;
- debug-symbol limitations;
- performance limitations;
- non-load-bearing diagnostic wording differences.

It may not contain:

- unsupported normative Core execution;
- wrong native output;
- wrong move/copy;
- missing, duplicate, or misordered Drop;
- wrong generic or trait dispatch;
- collection-order divergence;
- trap-category/provenance divergence;
- Tier-1 semantic divergence.

Any such issue forces `NATIVE-CORE-TIER1-PARITY-NOT-YET`.

A positive parity claim is CE8.

---

## 5. Fixed decisions

C6 does not reopen:

- native compilation is mandatory;
- generated Rust is the selected backend;
- native emission consumes verified MIR;
- STARK owns monomorphisation and instance discovery;
- MIR bodies are concrete;
- Core dispatch is static;
- trait objects are outside Core v1;
- function values are non-capturing;
- panic/trap aborts;
- no Drop runs after an aborting trap;
- layout answers come from a named versioned contract;
- stable Rust is required;
- optimisation waits for C7;
- no stable public ABI is claimed.

Stop for owner review before:

- CE1: normative semantic change;
- CE2: genuine spec ambiguity;
- CE3: MIR/verifier/version change;
- CE4: runtime ABI, value layout, Drop glue, resource, or trap change;
- CE5: backend replacement;
- CE8: conformance claim;
- CE9: trust-sensitive process/codegen/linking change.

---

## 6. Scope boundaries

### In scope

- moves, copies, borrows, partial moves, reinitialisation, and Drop;
- generic functions/types/methods;
- static trait and method dispatch;
- associated types after front-end resolution;
- operator-to-trait semantics;
- String/str;
- arrays, Vec, slices, Box;
- Option/Result;
- iterators;
- HashMap/HashSet where normative;
- formatting, output, assertions, panic, and traps;
- whole-file operations and file resources where normative;
- multi-file and multi-package execution;
- source provenance;
- Tier-1 portability;
- generated differential corpus.

### Out of scope

```text
class / inheritance
dyn / trait objects
closures / lambdas
async / await
concurrency / actors
macros
reflection
unsafe STARK
general FFI
garbage collection
runtime type registry
service container
public registry/publishing
HTTP/TLS/network packages
release/optimisation
incremental compilation
LLVM or direct Cranelift product backend
stable external ABI
self-hosting
```

### No host-semantic substitution

The implementation must not let:

- Rust Drop choose STARK Drop timing;
- Rust HashMap choose iteration order;
- Rust derives replace STARK Eq/Ord/Hash/Display/Clone;
- Rust overflow behaviour replace STARK checked operations;
- Rust panic unwinding replace STARK abort;
- Rust Option ownership replace STARK partial initialisation;
- Rust string indexing replace STARK Unicode contracts;
- Rust iterator behaviour define STARK borrow semantics.

Host types are allowed only behind proven-equivalent wrappers.

---

## 7. Gate sequence

```text
C6-ENTRY  freeze inventory and claims
C6.1      ownership and Drop parity
C6.2      generics and static trait dispatch
C6.3      runtime values and collections
C6.4      Tier-1 platform matrix
C6.5      full differential/generated corpus
C6.6      adversarial review and gate exit
```

Do not close later packages around an unresolved earlier semantic blocker.

---

# WP-C6.1 — Ownership and Drop Parity

## 8. Claim

C6.1 proves:

- every MIR move is a STARK move;
- every MIR copy is allowed only for a MIR-Copy type;
- every live drop unit is dropped exactly once;
- every moved drop unit is never dropped;
- all normal exits perform required cleanup;
- aborting traps perform no cleanup;
- partial values never need to form an invalid complete host value;
- generated Rust cannot accidentally use a moved STARK value.

## 9. C6.1a — Ownership matrix

Create:

```text
STARKLANG/docs/compiler/work-packages/C6-OWNERSHIP-MATRIX.md
```

Columns:

```text
source construct
type shape
Copy classification
MIR shape
HIR outcome
MIR outcome
native current outcome
native target outcome
positive test
negative test
deviation
status
```

Rows must cover:

- locals, parameters, returns;
- tuple/struct/enum fields;
- constant-index array elements;
- dynamic indexed places where movement is permitted/prohibited;
- Vec elements;
- Box inner values;
- Option/Result payloads;
- pattern bindings and wildcards;
- branch joins and loop-carried values;
- break/continue/return/`?`;
- function-value aggregates;
- destructor-bearing generic nominals.

C6.1a closes only when every normative ownership shape is classified.

## 10. C6.1b — General cross-block non-Copy movement

Remove C5’s control-flow limitation without asking Rust’s borrow checker to infer STARK liveness.

Requirements:

- preserve `ValueSlot<T>` or an approved successor;
- maintain explicit whole-place and drop-unit liveness;
- route every move, write, and Drop through reviewed helpers;
- consume the shared `DropPlan`;
- do not infer liveness from generated Rust control flow;
- do not use automatic Rust Drop as fallback;
- confine required unsafe code to reviewed runtime/helper modules;
- document and test each unsafe invariant.

Required cases:

- define in block A, move in block B;
- conditional definition then join;
- move in one branch;
- loop-carried reinitialisation;
- move then reinitialise;
- return and `?` transfer;
- call argument and return movement;
- recursive legal control flow.

A rustc borrow-check failure is not an acceptable boundary for normative Core.

## 11. C6.1c — Enum payload partial moves

Implement:

- one payload field moved while siblings remain live;
- multiple payload moves in different orders;
- wildcard cleanup of unbound fields;
- correct type-destructor and payload order;
- no Drop of moved payloads;
- exact branch-join liveness;
- reinitialisation only when a valid whole value is restored.

Negative controls:

- illegal moves remain rejected;
- moved payload use is rejected;
- corrupted liveness produces test failure;
- collapsing to whole-enum Drop produces duplicate-drop evidence.

## 12. C6.1d — Non-Copy array iteration

Close normative by-value iteration over fixed arrays with non-Copy elements.

Preserve:

- source order;
- one ownership transfer per element;
- cleanup of unconsumed elements after break/return/`?`;
- no whole-array Drop after complete consumption;
- no cleanup after trap;
- zero-, one-, and multi-element cases.

Candidate strategies:

```text
fixed-array MIR unrolling
runtime element-liveness bitmap
typed constant-index iteration
other CE3-approved MIR shape
```

Copying elements is prohibited.

## 13. C6.1e — Drop-path matrix

Cover normal exits from:

- scopes;
- functions/methods;
- loop body;
- break/continue;
- return;
- match arm;
- failed pattern binding;
- `?`;
- normal main completion.

Cover abnormal exits from:

- panic;
- arithmetic trap;
- cast trap;
- index trap;
- explicit MIR trap;
- IO/provider failure where normative.

Observe marker order and counts. Exit code alone is insufficient.

## 14. C6.1 mutation guards

Tests must fail if a developer deliberately:

1. clears a live drop unit early;
2. leaves a moved unit live;
3. reverses field Drop order;
4. drops fields before the type destructor;
5. cleans up after trap;
6. copies a non-Copy move;
7. whole-drops a partially moved enum;
8. skips remaining array-element cleanup;
9. double-drops Box/Vec contents;
10. treats function pointers as non-Copy.

## 15. C6.1 closure

- [ ] ownership matrix complete;
- [ ] all normative move/copy shapes have three-engine evidence;
- [ ] general cross-block non-Copy movement works;
- [ ] enum payload partial moves work;
- [ ] non-Copy array iteration works;
- [ ] normal Drop paths agree;
- [ ] trap paths perform no cleanup;
- [ ] Drop count/order agree;
- [ ] no normative C5 ownership refusal remains;
- [ ] mutation guards pass;
- [ ] full suite passes;
- [ ] deterministic tests pass twice;
- [ ] state/deviation records updated.

---

# WP-C6.2 — Generics and Static Trait Dispatch

## 16. Claim

Every front-end-resolved generic, inherent-method, trait-method, associated-type, and operator call has one deterministic concrete native meaning.

Trait objects remain out of scope.

## 17. C6.2a — Generic instance completeness

Audit:

- generic free functions;
- generic inherent and trait methods;
- generic structs/enums;
- nested generic nominals;
- concrete generic function values;
- recursive/mutually recursive instances;
- cross-package and dependency-to-dependency calls;
- inferred/explicit arguments;
- aliases;
- associated-type results;
- generic Drop/Copy/layout;
- generic collections and iterators.

Every build must contain:

- one body per canonical concrete function instance;
- one type per canonical concrete nominal;
- no type parameter or inference variable;
- no duplicate due to path spelling;
- no missing function-value-, Drop-, or trait-only reachability;
- deterministic order.

## 18. C6.2b — Method-resolution completion

Recheck every open limitation, including generic impl-head receiver inference if still open.

Matrix:

```text
inherent
user trait
CoreTrait
default trait method
fully qualified call
generic-parameter method
Self in default method
shared/mutable/nested-reference receiver
associated function
associated type
ambiguity
privacy
cross-package impl
```

The backend consumes the front-end-selected callee. It does not redo lookup.

## 19. C6.2c — Associated types — **CLOSED (CD-106)**

Prove:

- declarations and impl bindings; ✅
- `Self::Item`; ✅
- `T::Item`; ✅ (explicit binding AND inferred from the call argument via deferred projection obligations)
- explicit binding constraints; ✅
- nested associated results; ✅
- associated types in signatures; ✅
- cross-package use; ✅ (DEV-101 span provenance in `check_trait_member_call`)
- associated types inside runtime collections and Drop-bearing nominals. ✅ Drop-bearing; runtime
  collections resolve in HIR+MIR — returning a `Vec<..>` BY VALUE across the native linkage boundary
  is a separate C6.3 limitation (a plain `fn f() -> Vec<_>` fails identically), not a C6.2c gap.

Verified MIR must contain concrete types — enforced by native emit's C4.5 residual-param refusal;
reachable bodies compile and run. Evidence: `tests/c62c_associated_types.rs`. See
`C6-GENERICS-TRAITS-MATRIX.md` "C6.2c — the §19 associated-type matrix" for the per-row status.

## 20. C6.2d — Operator/CoreTrait semantics — **CLOSED (CD-107)**

Native execution must invoke STARK semantics for:

- Eq; ✅ native (`==`, `!=`)
- Ord; ✅ native (`cmp` + all four comparison operators)
- Hash; ✅ dispatch (HIR+MIR); native HashMap runtime is C6.3
- Display; ✅ dispatch (HIR+MIR); native `println`/String-return is C6.3
- Clone; ✅ native
- Default; ✅ native (via `P::default()`)
- conversion/index/iterator traits where normative. `From` ✅ native.

Adversarial types (all proven — `tests/c62d_operator_coretrait.rs`):

- Eq always true → distinct values compare equal; ✅
- reverse Ord → `a < b` false for `a.v < b.v`; ✅
- intentional Hash collision → constant `hash` keeps both distinct keys; ✅ (HIR+MIR)
- Display unlike structural debug → `fmt` returns a fixed string; ✅ (HIR+MIR)
- observable Clone → clone changes the value; ✅
- nonzero Default → `default()` returns 42; ✅

**Rust equivalents must not substitute — proven both ways.** The backend emits NO
`#[derive(PartialEq/Ord/Clone/Hash/..)]` on STARK nominals; every operator/CoreTrait call routes
through the written impl (the adversarial always-true `Eq` would be impossible otherwise). A MISSING
impl is rejected — `==`/`<` without `Eq`/`Ord` → E0500, `.clone()` without `Clone` → E0302 — never
filled by a Rust derive.

**Deferred (owner decision, DEV-103/DEV-104 in the CD-107 entry):** `.into()` deriving from a `From`
impl (blanket `Into`) and `Default::default()` with a type-inferred target. The spec lists
`From`/`Into` as independent traits and mandates only `fn default() -> Self`; `Fahrenheit::from(c)`
and `P::default()` are the supported forms. Display/Hash native output/collection runtime is C6.3
(Track C), not a C6.2d gap — the dispatch is correct in HIR+MIR.

## 21. C6.2e — Deterministic identity — **CLOSED (CD-108)**

Test clean rebuild, relocation, and dependency declaration reorder for:

- function/method/trait instances; ✅
- Drop instances; ✅
- generic nominals; ✅
- function-pointer sentinels; ✅
- helper/wrapper names. ✅

Absolute paths must not enter semantic symbol identity. ✅

**Defect found and fixed.** Generic type arguments rendered a nominal as `struct#N`/`enum#N` (raw
`ItemId` index). The index is assigned by item WALK ORDER, so a **dependency-declaration reorder**
swapped indices and changed the symbol (`callA@[struct#5]` ⇄ `callA@[struct#10]`) — a §21 violation.
`mir::lower::symbol_ty` now renders the nominal's **content path** (`struct#liba::A`): order-stable,
relocation- and rebuild-stable, and still distinct from an identically-named core type (a user may
declare `struct Vec`; the `struct#`/`enum#` head keeps it apart from core `Vec<..>`). Named-path
method/trait/Drop symbols were already content-based. Evidence:
`tests/c62e_deterministic_identity.rs` (relocation+rebuild; dependency reorder; no path/pid leak).

## 22. C6.2 closure — **CLOSED (CD-108)**

- [x] all executable generic forms covered; (C6.2a/b/c)
- [x] all accepted trait/method forms covered; (C6.2b matrix cleared)
- [x] associated types concrete in MIR; (C6.2c)
- [x] no normative method-resolution limitation remains; (F3 → C6.1f closed; the only open item is
      the F4 parser half — `&&T` is unspellable and inferred `&&T` fails MIR verify — a syntactic
      edge, not a normative resolution rule)
- [x] operator dispatch follows STARK impls; (C6.2d)
- [x] no Rust semantic derive shortcut; (C6.2d — no `#[derive]` on nominals; missing impls rejected)
- [x] one canonical instance emitted once; (C6.2a)
- [x] indirect/Drop/trait-only reachability works; (C6.2a native linkage)
- [x] deterministic relocation-stable identity; (C6.2e)
- [x] full suite and negative linkage tests pass; (per-WP targeted suites green at each closure; the
      full workspace suite is the Gate C6 exit gate, C6.6)
- [x] records updated.

---

# WP-C6.3 — Core Runtime Values and Collections

## 23. Runtime principles

`stark-runtime` must become the normative native Core runtime while remaining:

- versioned;
- deterministic;
- offline-buildable;
- independent of source checkout;
- free of host-semantic drift.

Suggested responsibility map:

```text
version.rs
output.rs
trap.rs
slot.rs
value.rs
string.rs
vec.rs
slice.rs
boxed.rs
iter.rs
collections.rs
format.rs
file.rs
provider_abi.rs
```

Do not create empty modules merely to match this list.

## 24. C6.3a — String and str — **PARTIAL (CD-109)**

Inventory and implement:

- construction; ✅ native (`String::from`, `String::new`)
- move/clone where normative; ✅ native (owned move; `clone`)
- length and emptiness; ✅ native (`len` = bytes, `is_empty`)
- UTF-8 validity; ⏳ (well-formed literals only so far; invalid-boundary failures with char ops)
- character iteration; ⏳ (`chars()` / `CharsIter` — remaining)
- byte/character distinctions; ⏳ (byte `len` done; char ops remaining)
- concatenation and mutation; ✅ native (`push_str`, `clear`)
- valid slicing/view behaviour; ⏳ (string slicing views — remaining)
- borrowed str; ✅ native for str VALUES (literals, `&str` params, `contains` pattern). A STORED
  interior `&str` borrowing an OWNED `String` across a block is deferred to WP-C6.1g-c (native
  dispatch-loop borrow; HIR+MIR pass).
- comparison; ✅ native for `str` values; owned-`String` `==`/`<` deferred to C6.1g-c (lowers
  through `String::as_str` → the stored-interior-borrow case above)
- canonical display; ⏳ (C6.3e formatting)
- nested formatting; ⏳ (C6.3e)
- Drop/reinitialisation; ✅ (slot-backed `String`; MIR-controlled drop)
- cross-package passing/return; ⏳ (return-across-fn done same-package; cross-package remaining)

Invalid boundaries must produce the specified failure and source location. Landed:
`stark-runtime/src/string.rs`, `emit_runtime`, `emit_ty`/`Constant::Str`. Evidence:
`tests/c63a_string.rs` (15, three-engine with native stdout-byte checks).

## 25. C6.3b — Vec, slices, Box

### Vec

- new/empty;
- push/pop;
- len/capacity where normative;
- indexing/mutable indexing;
- insert/remove/clear where normative;
- by-value/shared/mutable iteration;
- `as_slice`;
- growth/reallocation;
- partial consumption;
- nested Vec;
- Drop-bearing and generic elements.

### Slices

- array and Vec views;
- shared and mutable views;
- range slicing;
- nested indexing;
- mutation reflected in source;
- returned-reference provenance;
- no disconnected copy;
- bounds failures.

### Box

- new;
- borrow/deref;
- mutable borrow;
- `into_inner`;
- move;
- exact inner Drop;
- recursive types;
- generic/nested uses.

Do not add a general Deref trait unless Core v1 defines it.

## 26. C6.3c — Iterators

Matrix:

```text
Range
array value/shared/mutable
Vec value/shared/mutable
slice shared/mutable
HashMap keys/values/entries
HashSet
user Iterator impl
map/filter/collect where normative
```

Prove order, ownership, borrow duration, early termination, remaining-element Drop, associated Item, and `for` equivalence to explicit iteration.

No element cloning to avoid ownership work.

## 27. C6.3d — HashMap and HashSet

If normative at C6 entry, native support is mandatory.

Record a CE4 representation decision for:

- first-insertion order;
- replacement preserving position;
- remove/reinsert appending;
- STARK Eq and Hash;
- collisions;
- deterministic hashing;
- growth observability;
- Drop order;
- Tier-1 consistency.

Required adversarial cases:

- reversed insertion;
- replacement;
- remove/reinsert;
- total collision;
- custom Eq;
- custom Hash;
- Drop-bearing keys/values;
- second-run and cross-platform determinism.

A raw host HashMap is unacceptable if it violates these contracts.

## 28. C6.3e — Formatting and failure

Cover canonical formatting for:

- primitives;
- Float32/Float64;
- String/str;
- tuple/array;
- struct/enum;
- Option/Result;
- Vec/slice/Box;
- HashMap/HashSet;
- user Display;
- nested values.

Also cover:

- print/println;
- assertions;
- panic text;
- stdout/stderr bytes;
- line termination;
- Tier-1 equality.

Generated Rust Debug output is not STARK Display.

## 29. C6.3f — Files and resources

Where normative, implement:

- open/create;
- read/write;
- close;
- consuming close;
- automatic close/Drop;
- stable error mapping;
- UTF-8 failure;
- missing file;
- permission failure where deterministic;
- double-close/consumed-handle rejection;
- no cleanup after abort;
- source-mapped failure;
- Tier-1 paths.

Use the approved provider/resource ABI when required. Do not introduce general FFI.

## 30. Runtime compatibility

Record/validate:

```text
compiler
MIR
MIR runtime surface
backend
runtime
target triple
layout contract
profile
```

Mismatch must fail before user code.

Runtime additions require version review, generated-code tests, installed-layout tests, and offline-build proof.

## 31. C6.3 closure

- [ ] runtime API inventory complete;
- [ ] String/str parity;
- [ ] Vec/slice/Box parity;
- [ ] iterator parity;
- [ ] HashMap/HashSet parity where normative;
- [ ] formatting/output parity;
- [ ] whole-file/resource parity where normative;
- [ ] exact close/Drop;
- [ ] version checks;
- [ ] installed runtime outside checkout;
- [ ] offline generated build;
- [ ] no host-semantic substitution;
- [ ] full suite and records complete.

---

# WP-C6.4 — Tier-1 Platform Matrix

## 32. Targets

Tier 1:

```text
aarch64-apple-darwin
x86_64-unknown-linux-gnu
```

Tier 2:

```text
x86_64-pc-windows-msvc
```

Tier 3:

```text
x86_64-apple-darwin
```

A positive C6 claim requires both Tier-1 targets.

## 33. C6.4a — Target preflight

The build path must:

- identify host/selected target;
- accept only qualified targets;
- reject unsupported targets before linking;
- name supported targets;
- distinguish unsupported target from missing toolchain;
- select layout and executable naming;
- record target metadata.

C7 owns the full user-facing `--target` feature.

## 34. C6.4b — Portability audit

Audit:

- integer/pointer assumptions;
- alignment/layout;
- path separators;
- executable suffix;
- line endings;
- stdout/stderr bytes;
- filesystem errors/permissions;
- Unicode paths;
- temporary directories;
- toolchain/runtime discovery;
- manifest path escaping;
- shell-only tests.

Tier-1 semantic tests must not depend on Unix-only wrappers without equivalent evidence.

## 35. C6.4c — Platform evidence

Each Tier-1 platform runs:

```text
fmt
strict clippy
full no-fail-fast suite
C6 three-engine suite
generated corpus
frozen workspace build/run
installed-runtime test
offline generated-build test
determinism rerun
```

Evidence records commit, OS, architecture, rustc, Cargo, commands, counts, and artifacts.

No real platform run means no platform claim.

## 36. Windows disposition

Produce a Windows gap report:

```text
portable
path/process adaptation
runtime adaptation
harness adaptation
CI/toolchain blocked
semantic blocker
```

Windows may be implemented if bounded, but is not automatically a Tier-1 C6 blocker. Unsupported Windows must fail clearly.

## 37. C6.4 closure

- [ ] macOS-arm64 passes;
- [ ] Linux-x64 passes;
- [ ] Tier-1 observations agree;
- [ ] target metadata correct;
- [ ] unsupported targets reject clearly;
- [ ] no hidden host assumption;
- [ ] Windows report exists;
- [ ] evidence attached to exact commits;
- [ ] skipped tests are not counted as passing.

---

# WP-C6.5 — Full Differential and Generated Corpus

## 38. Ownership

C6.5 is the final owner of unfinished WP-C2.12 work:

- deterministic generated corpus;
- deeper per-category breadth;
- HIR/MIR/native-debug replay;
- Tier-1 replay;
- metamorphic equivalence;
- deterministic snapshots.

## 39. Observation comparator

Normalize:

```text
Completion {
  stdout_bytes,
  stderr_bytes,
  exit_status,
  returned_observation,
  drop_log
}

Trap {
  category,
  source_file,
  line,
  column,
  message_class,
  stdout_before_trap,
  stderr_observation,
  exit_status,
  drop_log_before_trap
}
```

Compare normative observations only, not Cargo text or host backtraces.

## 40. Corpus categories

Cover every executable Core category:

- every expression/statement/control transfer;
- every executable pattern;
- all primitives and aggregates;
- String/Vec/slice/Box/maps/sets/files;
- Option/Result;
- direct, method, trait, generic, and indirect calls;
- moves/copies/borrows/partial moves/Drop;
- numeric and indexing/cast traps;
- assertions/panic;
- multi-file/package/re-export/dependency-to-dependency cases;
- relocation;
- dependency declaration reorder;
- lock/offline build.

## 41. Deterministic generator

Requirements:

- explicit seed;
- checked-in generator version;
- deterministic IDs;
- valid type- and ownership-correct programs;
- bounded source/MIR/runtime size;
- no network;
- reproducible sources and hashes;
- failing-case retention;
- reduction path where practical.

Preferred design:

```text
semantic templates
× bounded types
× bounded control-flow
× bounded ownership modes
× bounded package layouts
```

Arbitrary token fuzzing is not the semantic generator.

## 42. Metamorphic families

Require equivalent observations for:

- alpha-renaming;
- harmless scope insertion;
- explicit/inferred generics;
- qualified/unqualified trait call;
- shorthand/explicit fields;
- equivalent pattern decomposition;
- non-overlapping arm reorder;
- workspace relocation;
- dependency declaration reorder;
- helper extraction;
- direct function vs equivalent function value;
- equivalent loop forms where normative.

Divergence opens a defect; do not redesign the pair to hide it.

## 43. Mutation controls

The harness must detect:

- wrong arithmetic;
- wrong trap line/category;
- omitted/duplicate/reversed Drop;
- copied move;
- wrong generic instance;
- wrong trait impl;
- wrong function-value target;
- changed collection order;
- slice view changed to copy;
- Float32 formatted as Float64;
- generated-Rust path replacing source path;
- missing output;
- incorrect exit normalization.

Record mutation evidence.

## 44. Corpus manifest

Create an approved equivalent of:

```text
starkc/tests/c6-corpus/manifest.toml
```

Each case records:

```text
case_id
category
sources
package graph
language options
expected completion/trap class
required engines
required targets
metamorphic family
generator seed/version
normative rules
deviation/quarantine
```

A semantic quarantine keeps C6 open unless the feature is genuinely outside Core.

Release replay is `NOT-APPLICABLE` until C7 provides release builds.

## 45. C6.5 closure

- [ ] hand-written category coverage complete;
- [ ] deterministic generator exists;
- [ ] generated corpus is nontrivial and bounded;
- [ ] HIR/MIR/native debug replay passes;
- [ ] Tier-1 replay passes;
- [ ] all metamorphic families pass;
- [ ] mutation controls prove sensitivity;
- [ ] no unexplained divergence;
- [ ] manifest and hashes recorded;
- [ ] two clean runs deterministic;
- [ ] WP-C2.12 remaining scope formally closed.

---

# WP-C6.6 — Gate Exit

## 46. Entry packet

C6.6 requires:

```text
ownership matrix
generic/trait matrix
runtime matrix
Tier-1 evidence
corpus manifest
generated-corpus evidence
mutation results
open deviations
version identities
exact commit
```

## 47. Adversarial review

Answer with executable evidence:

1. Can legal non-Copy code still fail only because generated Rust cannot express control flow?
2. Can moved units remain live?
3. Can live units be skipped?
4. Can Drop run after trap?
5. Can enum partial movement form an invalid whole host value?
6. Can array iteration copy non-Copy elements?
7. Can trait/Drop/function-value-only instances be omitted?
8. Can concrete instances collapse by item ID?
9. Can host Eq/Ord/Hash/Display replace STARK?
10. Can map/set order vary?
11. Can slices become copies?
12. Can Vec growth invalidate accepted borrows?
13. Can Box inner values drop twice?
14. Can file resources close twice or leak normally?
15. Can runtime mismatch execute user code?
16. Can provenance point to generated Rust?
17. Can Tier-1 platforms differ normatively?
18. Can skipped tests count as parity?
19. Can the generator produce only scalar cases?
20. Can normalization hide real differences?
21. Can a C5 Unsupported remain for normative Core?
22. Can rustc be the first detector of a known normative gap?
23. Can the backend redo semantic selection?
24. Can relocation change semantic symbols?
25. Did C6 import later-gate scope?

## 48. Validation

At the exact closure commit:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --no-fail-fast
```

Also run on both Tier-1 platforms:

```text
full C6 corpus
generated corpus from clean seed
frozen C5 workspace
ownership/Drop suite
runtime suite
installed-runtime suite
offline-build suite
determinism suite twice
```

Record exact counts.

## 49. Exit report

Create:

```text
starkc/docs/compiler/C6-exit-report.md
```

Include:

- exact commit and owner decision;
- selected outcome;
- compiler/MIR/runtime/backend versions;
- layout identities;
- OS/toolchain matrix;
- test and corpus counts;
- generator seed/version/count;
- mutation results;
- runtime API matrix;
- ownership and generic/trait summaries;
- open deviations;
- Windows status;
- exact next WP;
- explicit statement that C7 release/optimisation work is not included.

## 50. Gate closure checklist

### Semantics

- [ ] every executable normative Core feature has Tier-1 native support;
- [ ] HIR/MIR/native debug agree;
- [ ] ownership/Drop agree;
- [ ] traps/provenance agree;
- [ ] runtime values agree;
- [ ] generic/trait dispatch agrees;
- [ ] no known native wrong-output defect.

### Platforms

- [ ] macOS-arm64;
- [ ] Linux-x64;
- [ ] unsupported target diagnostics;
- [ ] Windows gap report;
- [ ] platform metadata.

### Corpus

- [ ] complete-category hand-written corpus;
- [ ] generated corpus;
- [ ] metamorphic corpus;
- [ ] mutation-sensitive comparator;
- [ ] deterministic snapshots;
- [ ] WP-C2.12 closure.

### Governance

- [ ] all CE3/CE4 decisions recorded;
- [ ] CE8 owner decision recorded;
- [ ] deviations/conformance/state updated;
- [ ] user docs do not overclaim;
- [ ] C7 named next.

Any unchecked required box keeps C6 open.

---

## 51. Likely repository areas

Compiler/runtime:

```text
starkc/src/mir/
starkc/src/backend/generated_rust/
starkc/stark-runtime/src/
starkc/src/typecheck.rs
starkc/src/resolve.rs
starkc/src/borrowck.rs
starkc/src/flow.rs
starkc/src/native_build.rs
starkc/src/native_toolchain.rs
```

Evidence:

```text
starkc/tests/three_engine_differential.rs
starkc/tests/mir_differential.rs
starkc/tests/native_c6_1_ownership.rs
starkc/tests/native_c6_2_generics_traits.rs
starkc/tests/native_c6_3_runtime.rs
starkc/tests/native_c6_4_platform.rs
starkc/tests/c6_generated_corpus.rs
starkc/tests/exec_snapshots.rs
starkc/tests/c6-corpus/
```

Documents:

```text
STARKLANG/docs/compiler/work-packages/WP-C6-ENTRY.md
STARKLANG/docs/compiler/work-packages/C6-OWNERSHIP-MATRIX.md
STARKLANG/docs/compiler/work-packages/C6-GENERICS-TRAITS-MATRIX.md
STARKLANG/docs/compiler/work-packages/C6-RUNTIME-MATRIX.md
STARKLANG/docs/compiler/work-packages/C6-PLATFORM-MATRIX.md
starkc/docs/compiler/C6-exit-report.md
```

Use existing responsibility owners where present. Do not duplicate pipelines.

---

## 52. Agent protocol

The implementation agent must:

1. read Charter, State, C5 exit, and this file;
2. confirm exact head and active sub-package;
3. load only relevant normative rules/source/tests;
4. build the acceptance matrix before coding;
5. add a failing regression/differential case before fixing;
6. preserve HIR as reference;
7. preserve verified-MIR backend boundary;
8. escalate CE decisions before implementation;
9. avoid later-gate features;
10. run focused validation;
11. run full validation at WP closure;
12. update state/deviations/conformance;
13. commit only on explicit owner request;
14. never infer closure or invent a decision ID.

Suggested labels:

```text
[C6.1a] freeze ownership matrix
[C6.1b] generalise cross-block non-Copy storage
[C6.1c] implement enum payload partial moves
[C6.1d] implement non-Copy array iteration
[C6.1e] close ownership/Drop parity

[C6.2a] complete generic inventory
[C6.2b] close method-resolution limitations
[C6.2c] prove associated types
[C6.2d] close operator/CoreTrait dispatch
[C6.2e] close deterministic instance emission

[C6.3a] String/str parity
[C6.3b] Vec/slice/Box parity
[C6.3c] iterator parity
[C6.3d] deterministic hash collections
[C6.3e] formatting/output parity
[C6.3f] file/resource parity

[C6.4] qualify Tier-1 platforms
[C6.5] complete generated differential corpus
[C6.6] close Gate C6
```

---

## 53. Stop rules

Stop and record a blocker when:

- semantics are ambiguous;
- accepted language must change;
- MIR lacks required information;
- verifier rules must change;
- runtime ABI/layout must change;
- host types cannot preserve STARK behaviour;
- Tier-1 cannot represent an approved contract;
- nightly Rust appears necessary;
- a solution requires dyn, closures, async, concurrency, reflection, or general FFI;
- HIR, MIR, or spec is wrong;
- a trust boundary changes;
- a failure can only be hidden by weakening evidence.

Do not silently broaden scope.

---

## 54. End result

At successful C6 closure:

> On macOS-arm64 and Linux-x64, STARK native debug executables preserve executable Core v1 ownership, partial-move, Drop, generic, trait-dispatch, runtime-value, collection, iterator, formatting, failure, and source-provenance semantics. HIR, MIR, and native debug execution agree across a complete hand-written and deterministic generated corpus. Remaining compiler work begins at C7: release profiles, reproducibility, measured optimisation, and performance qualification.

This result must be proven, not declared in advance.
