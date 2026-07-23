# WP-C5.4 — Multi-file/Package Linkage and Function Values

**Status:** PROPOSED IMPLEMENTATION DIRECTIVE  
**Prepared:** 2026-07-23  
**Repository baseline:** `0f39579d722b0851b7abe497df306222014f9730`  
**Depends on:** WP-C5.1, WP-C5.2, WP-C5.3 — all closed  
**Next after closure:** WP-C5.5 (`stark build` and debug build experience)  
**Escalation:** Follow `COMPILER-CHARTER.md`; any MIR shape change, Core semantic change, stable-ABI claim, or change to canonical instance identity requires the corresponding owner decision before implementation continues.

---

## 0. Executive directive

Implement WP-C5.4 as a bounded extension of the existing generated-Rust backend.

The package must deliver:

1. deterministic, validated linkage over the canonical symbols already carried by verified MIR;
2. exactly one generated Rust definition for each concrete `MirProgram` body;
3. native representation and execution of non-capturing function values;
4. indirect calls through `Callee::FnValue`;
5. a frozen three-package application containing a cross-package concrete generic call and a cross-package function reference;
6. three-engine agreement for the supported C5.4 cases;
7. deterministic refusal before rustc for every adjacent unsupported shape.

Do **not** redesign MIR, package resolution, monomorphisation, or source semantics. The backend consumes the concrete, transitively reachable, monomorphised program it is given. It must validate that input and emit it; it must not become a second linker, resolver, or monomorphiser.

The governing principle is:

> `Instance.symbol` is the canonical callable identity. `MirProgram.bodies` is the authoritative concrete-body set. C5.4 validates and emits those artifacts; it does not reconstruct them.

---

## 1. Purpose and closure claim

### 1.1 Purpose

WP-C5.4 closes the remaining semantic gap between the existing single-link-unit generated backend and the Gate C5 reference application:

- multiple source files and packages already lower into one verified `MirProgram`;
- direct concrete calls already emit;
- concrete nominal types already emit;
- deterministic symbol sanitisation already exists;
- `MirTy::FnPtr`, `Constant::FnPtr`, and `Callee::FnValue` already exist in MIR;
- the MIR interpreter already executes indirect calls.

C5.4 makes those existing MIR capabilities executable in the native backend and proves that cross-package identity survives end to end.

### 1.2 Closure claim

C5.4 may close only when the following claim is mechanically supported:

> For the approved C5.4 subset, one verified multi-package, monomorphised `MirProgram` emits as one deterministic generated Rust crate in which every referenced concrete function instance is defined exactly once, direct and indirect calls target the intended instance, cross-package generic calls execute correctly, and HIR, MIR, and native outcomes agree.

### 1.3 Consequence of a false closure

A false closure could cause:

- two distinct STARK functions to become one generated Rust function;
- a concrete generic instance to be missing or emitted twice;
- a function value to call the wrong concrete instance;
- a valid cross-package program to fail only in rustc;
- a native executable to disagree with HIR/MIR while appearing to link successfully;
- source/package identity to depend on workspace location or traversal order.

Therefore deterministic source generation and successful rustc linkage are necessary but not sufficient. Values and call targets must be observed through three-engine tests.

---

## 2. Authority and fixed inputs

The following inputs are fixed and must not be reopened by this package.

### 2.1 MIR authority

From `starkc/src/mir/mod.rs`:

- MIR version remains `0.1`.
- verified MIR is monomorphised-only;
- `Instance` contains:
  - `item`;
  - concrete `type_args`;
  - canonical `symbol`;
- `MirTy::FnPtr { params, ret }` is the function-value type;
- `Constant::FnPtr(Instance)` is a bare non-capturing function-instance value;
- `Callee::FnValue(Operand)` is the indirect-call form;
- `MirProgram.bodies` is sorted by canonical symbol;
- `MirProgram.bodies` is the concrete body set supplied to consumers.

No unresolved type parameter or inference variable may reach C5.4.

### 2.2 Backend authority boundary

The generated-Rust backend:

- consumes verified MIR;
- may map MIR types and operations into Rust syntax;
- may validate that the body set is internally complete;
- may deterministically order purely representational output;
- must not inspect HIR to reconstruct identities;
- must not redo package resolution;
- must not discover generic instances;
- must not infer type arguments;
- must not emit semantic Rust generics for STARK generic functions;
- must not choose a different callable based on Rust overload resolution;
- must not create a stable external ABI.

### 2.3 Existing canonical identity

`starkc/src/backend/generated_rust/mangle.rs` already establishes:

- `Instance.symbol` as the canonical MIR function identity;
- `function_name_for_symbol` as the generated Rust function-name mapping;
- injective byte escaping in `sanitize_symbol`;
- a separate nominal-type key space;
- the `main@[]` to Rust `main` mapping;
- collision and round-trip tests.

C5.4a is therefore an **audit, validation, and cross-package proof**, not permission to design a second symbol grammar.

### 2.4 Existing emission topology

`emit_program.rs` already:

- finds the canonical entry symbol;
- emits every body in `program.bodies`;
- emits one Rust function per body;
- uses canonical symbol-derived names;
- flattens the linked program into one generated Rust crate.

C5.4 keeps this one-crate, one-Rust-module topology unless a demonstrated compiler error makes a split necessary. Source package/module visibility has already been checked before MIR; generated Rust visibility is not a second implementation of STARK visibility.

### 2.5 Existing C5.3 boundaries

C5.4 must preserve all C5.3 deterministic refusals, including:

- partial movement from a multi-unit enum payload;
- wider ownership/reference cases explicitly deferred to C6;
- unsupported runtime-owning Core types;
- unsupported projections or reference escapes.

A C5.4 fixture must not accidentally depend on a C6-deferred feature.

---

## 3. Non-goals

C5.4 does **not** implement:

- closures or captured environments;
- trait objects;
- dynamic dispatch other than `Callee::FnValue`;
- function-value comparison, ordering, or hashing;
- function addresses as integers;
- reflection over function identities;
- a stable native ABI;
- separate shared libraries per package;
- dynamic linking;
- package loading at runtime;
- incremental compilation;
- linker-level dead-code elimination;
- generic Rust function definitions representing STARK generics;
- provider callbacks;
- C6-wide ownership, references, containers, or iterator parity;
- the user-facing `stark build` command, which belongs to C5.5.

Do not expand scope merely because Rust makes an adjacent feature easy.

---

## 4. Current implementation inventory

Before changing code, verify this inventory against the baseline and record any mismatch in `WP-C5.4.md`.

| Capability | Current expected state | C5.4 action |
|---|---|---|
| Canonical instance symbol | Present in `Instance.symbol` | Consume; do not re-derive |
| Body determinism | `MirProgram.bodies` sorted by symbol | Validate and preserve |
| Rust symbol escaping | Implemented and injective | Extend cross-package evidence |
| Direct concrete call | Implemented for `Callee::Instance` | Preserve |
| Concrete function definition emission | Implemented for every body | Validate uniqueness/completeness |
| Concrete generic MIR bodies | Produced by lowering | Prove with multi-instance and cross-package tests |
| `MirTy::FnPtr` | Exists; backend type emission missing | Implement |
| `Constant::FnPtr` | Exists; backend constant emission missing | Implement |
| `Callee::FnValue` | Exists; native call emission missing | Implement |
| Function-value local default | No valid backend representation yet | Implement deterministic sentinel strategy |
| Function value in aggregates | MIR/type system support exists | Prove native representation works |
| Multi-package native fixture | Not frozen | Create and freeze |
| Multi-file/package three-engine comparator | Not yet established as a C5.4 fixture | Implement or extend harness |

Do not replace already-correct mechanisms merely to make C5.4 look self-contained.

---

## 5. C5.4 supported subset

### 5.1 Required

C5.4 must support:

- direct calls to any concrete `Callee::Instance` present in the verified body set;
- multiple concrete instantiations of one generic function;
- concrete nominal generic instances already present in `TypeContext`;
- cross-file and cross-package direct calls;
- `MirTy::FnPtr` whose parameter and return types are themselves in the approved C5 native subset;
- `Constant::FnPtr`;
- function values in:
  - locals;
  - parameters;
  - return values;
  - tuple fields;
  - array elements;
  - user-struct fields;
  - single-unit enum payloads where no C5.3 ownership boundary is crossed;
- `Operand::Copy` and `Operand::Move` for function values, which are semantically identical because function values are `Copy`;
- indirect invocation through `Callee::FnValue`;
- cross-package function references;
- one concrete generic function referenced directly and as a function value;
- recursion and mutual recursion among concrete bodies when already present in verified MIR.

### 5.2 Bounded signature rule

A `FnPtr` signature is admitted only when every parameter and the return type already has a C5 generated-Rust representation.

No new container, slice, provider-resource, closure, trait-object, or general reference support may be introduced merely to admit a function signature.

`MirTy::Never` may be emitted as Rust `!` only in a return position if an existing accepted C5 source case requires it. It must not become a stored local value.

### 5.3 Unsupported adjacent shapes

Reject before rustc:

- a function constant naming no body in the linked program;
- a direct callee naming no body;
- duplicate bodies with one canonical symbol;
- one canonical symbol associated with different item/type-argument identities;
- two canonical symbols mapping to one generated Rust function name;
- an indirect call whose operand is not `FnPtr`;
- a function-value signature containing an unsupported type;
- any unresolved type parameter;
- any attempted closure/capture representation;
- any multi-unit enum-payload move exposed by a function-value aggregate case;
- any generated-Rust type or call failure caused by a backend omission.

A rustc error is not an acceptable diagnostic for an out-of-subset STARK program.

---

## 6. Linkage preflight

Add one explicit linkage-validation pass before generated source is assembled.

Suggested responsibility:

```text
starkc/src/backend/generated_rust/linkage.rs
```

The exact file name may differ, but do not scatter these checks across constant emission, call emission, and `emit_program`.

### 6.1 Linkage index

Build a deterministic read-only index:

```rust
struct LinkedInstance<'a> {
    body: &'a MirBody,
    rust_name: String,
}

struct LinkageIndex<'a> {
    by_symbol: BTreeMap<String, LinkedInstance<'a>>,
}
```

This is a backend validation/indexing structure, not a second instance-discovery graph.

### 6.2 Required validations

For every `MirBody`:

1. canonical symbols are strictly sorted in `MirProgram.bodies`;
2. the symbol is unique;
3. the generated Rust name is unique;
4. the body instance’s `item`, `type_args`, and `symbol` are internally consistent with every reference to that instance;
5. exactly one entry body has `mangle::ENTRY_SYMBOL`;
6. no non-entry body maps to Rust `main`;
7. all body parameter and return types are C5-representable.

Scan every body for references from:

- `Callee::Instance`;
- `Constant::FnPtr` in statements;
- `Constant::FnPtr` in call arguments;
- `Constant::FnPtr` in checked arguments if structurally possible;
- constants nested in aggregate operands;
- trap-message operands if ever admitted;
- any future operand-bearing MIR location through one shared exhaustive visitor.

For each referenced `Instance`:

- its symbol must resolve to exactly one body;
- the resolved body must have the same `item` and `type_args`;
- direct-call and function-value references share this same resolution path.

Do not silently omit a missing body and let rustc report an unresolved function.

### 6.3 Shared operand visitor

Use or extend one exhaustive MIR operand walker. Do not create one walker for projection collection, another for linkage, and a third for future liveness checks that disagree on which terminators contain operands.

A new `Rvalue`, `Terminator`, `Callee`, or `Constant` variant must stop compilation or fail a test until the linkage visitor is updated.

### 6.4 Backend does not repair the program

The linkage pass must not:

- add missing bodies;
- clone one body under another symbol;
- rewrite instance type arguments;
- sort and silently normalize a producer defect if the MIR contract promises sorted bodies;
- substitute a same-named function;
- search HIR for a missing definition.

A failure means the accepted source-to-MIR pipeline or verified MIR is incomplete. Classify it as an internal compiler/backend contract defect.

---

## 7. Exact function-value representation

### 7.1 Decision

Represent:

```text
MirTy::FnPtr { params, ret }
```

as the typed Rust function pointer:

```rust
fn(P0, P1, ...) -> R
```

Properties:

- non-capturing;
- no data/environment pointer;
- Rust ABI internal to the generated executable;
- `Copy`;
- no `Eq`, `Ord`, or `Hash` operation is exposed by C5.4;
- not a stable external ABI;
- not convertible to an integer or raw address.

Do not wrap function values in trait objects, `Box<dyn Fn>`, closures, raw pointers, or a dynamic signature-erased table.

### 7.2 Type emission

Add an explicit `MirTy::FnPtr` arm to `emit_types::emit_ty`:

1. recursively emit each parameter type;
2. recursively emit the return type;
3. produce `fn(<params>) -> <ret>`.

A signature containing an unsupported type must produce `BackendDiagnostic::Unsupported` before generated Rust compilation.

Add direct unit tests for:

- zero-argument function;
- one argument;
- several arguments;
- aggregate argument;
- function-pointer parameter;
- Unit return;
- aggregate return;
- distinct signatures producing distinct Rust type text.

### 7.3 Copy classification

Function values are `Copy` because MIR says so (`TYPE-FN-001`), not because the backend independently asks Rust.

Preserve the shared `TypeContext::is_copy` authority. Add a regression test explicitly pinning:

```text
TypeContext::is_copy(FnPtr) == true
```

No `ValueSlot` may be introduced for a `FnPtr` local.

### 7.4 Default initialisation problem

The CFG-to-`loop { match __bb { ... } }` emitter default-initialises locals because rustc cannot prove MIR definite assignment across dispatch iterations. A bare Rust function pointer has no language-level default.

Use one generated **aborting sentinel function per distinct `FnPtr` signature**.

For signature:

```rust
fn(i32) -> i32
```

generate conceptually:

```rust
fn stark_fn_sentinel_<signature>(_: i32) -> i32 {
    std::process::abort()
}
```

Rules:

- one sentinel per distinct canonical `MirTy::FnPtr`;
- deterministic name derived from a dedicated key space, for example:
  `fn-sentinel#<dump_ty>`;
- name generation must reuse the injective sanitizer;
- the sentinel namespace must be disjoint from source function and nominal type namespaces;
- every parameter is accepted and ignored;
- the body calls `std::process::abort()` immediately;
- no user destructor or STARK cleanup runs;
- the function never returns, so Rust `!` coerces to any declared return type;
- a sentinel call is an internal compiler/liveness defect, never valid STARK behaviour.

Do not create a sentinel that returns an arbitrary value. Such a sentinel could hide a use-before-initialisation defect.

### 7.5 Signature collection

Collect distinct `FnPtr` types recursively from:

- body params;
- body return types;
- body locals;
- tuple/array elements;
- struct fields;
- enum payloads;
- nested generic type arguments;
- other function-pointer signatures.

Use a deterministic `BTreeMap` keyed by canonical `dump_ty`.

Emit sentinel definitions once, before ordinary bodies. Rust item order is not semantically relevant, but generated source order must remain deterministic.

### 7.6 Default value emission

Add:

```text
default_value_expr(FnPtr) -> sentinel function name
```

This enables function values inside default-initialised tuples, arrays, structs, and enums without creating invalid Rust.

A mutation test or focused regression must fail if the `FnPtr` default falls back to zero, null, transmute, or a non-aborting arbitrary function.

---

## 8. Function constants

### 8.1 Constant emission

Implement:

```text
Constant::FnPtr(instance)
```

as the generated Rust function item name obtained from the linkage index and `function_name_for_symbol`.

Do not emit:

- a numeric address;
- a string lookup;
- a switch table;
- a closure;
- a wrapper allocating an environment.

The expected Rust context coerces the function item to the declared `fn(...) -> ...` pointer type.

### 8.2 Resolution requirement

Before emission, linkage preflight must establish that the instance:

- resolves to exactly one body;
- has matching item and concrete type arguments;
- is not an invented or unresolved symbol.

`emit_constant` itself should not search HIR or reconstruct signatures.

It is acceptable to extend constant emission with a read-only linkage context. Avoid a global mutable registry.

### 8.3 Entry function

Probe whether valid STARK source can form a function value referencing the entry `main`.

- If the front end forbids it, add a source-level negative test and record the fact.
- If it is valid, prove that its function-value type and generated reference are coherent with the Rust `main` wrapper.
- Do not guess or add a backend-only prohibition without testing source reachability.

This probe does not block the primary C5.4 path unless the shape is source-reachable.

---

## 9. Indirect calls

### 9.1 Emission

Implement:

```text
Callee::FnValue(operand)
```

as:

```rust
(<emitted function-pointer operand>)(<emitted args>)
```

The call target is a typed Rust `fn` pointer. No runtime signature switch is required.

### 9.2 Argument and result rules

Reuse the existing direct-call machinery for:

- left-to-right operand emission as already sequenced by MIR;
- move/copy handling;
- destination assignment;
- continuation block.

Do not reconstruct source call order.

### 9.3 Verification boundary

MIR verification is responsible for proving:

- the operand has `MirTy::FnPtr`;
- parameter count matches;
- each argument type matches;
- destination type matches the function return type.

The backend may validate the expected shape defensively but must not implement type inference or overload selection.

Add negative MIR/verifier tests for:

- non-function operand;
- wrong arity;
- wrong parameter type;
- wrong destination type.

A malformed verified program reaching rustc is a verifier/backend contract defect.

### 9.4 Function value as argument and return

Required end-to-end shapes:

```text
fn apply(f: fn(Int32) -> Int32, x: Int32) -> Int32
fn choose(...) -> fn(Int32) -> Int32
```

Use actual valid STARK syntax from existing parser/typechecker tests. Do not invent syntax in implementation code.

---

## 10. Concrete generic instances

### 10.1 Authority

The frontend/lowering pipeline owns:

- generic argument selection;
- reachability;
- semantic monomorphisation;
- concrete `Instance` construction;
- canonical symbols;
- concrete body production.

C5.4 does not emit Rust generics for STARK generics.

### 10.2 Emission

Each concrete `MirBody` emits as one ordinary Rust function with:

- concrete parameter types;
- concrete return type;
- no Rust type parameter list;
- canonical symbol-derived name.

Example conceptual output:

```rust
fn stark_identity_...Int32(x: i32) -> i32 { ... }
fn stark_identity_...Int64(x: i64) -> i64 { ... }
```

not:

```rust
fn identity<T>(x: T) -> T
```

### 10.3 Deduplication rule

The body set is authoritative. Do not deduplicate by source item alone.

Distinct concrete instances of one item must remain distinct when their `type_args` differ.

The same concrete instance reached through several paths must appear once:

- direct call;
- function constant;
- return of a function value;
- aggregate field;
- cross-package re-export or import path already normalized by resolution.

### 10.4 Required tests

1. one generic function at `Int32`;
2. the same function at `Int64`;
3. both instances in one program;
4. one instance reached by direct call and function value;
5. repeated use of one instance emits one definition;
6. recursive generic instance;
7. cross-package generic call;
8. generic function returned or stored as a function value when valid under Core v1;
9. missing concrete body is rejected before rustc;
10. body with unresolved type parameter cannot enter backend.

### 10.5 Reachability adversarial test

Create a case where a function instance is referenced **only** by `Constant::FnPtr`, never by `Callee::Instance`.

The native program must link and call it indirectly.

This is mandatory. It guards against a direct-call-only reachability assumption, the same defect class previously exposed by types reachable only through layout queries.

---

## 11. Cross-package generated linkage

### 11.1 Topology decision

For C5.4, emit one flattened generated Rust crate containing:

- all reachable nominal definitions;
- all function-pointer sentinels;
- projection helpers;
- every concrete function body;
- the one entry wrapper.

Do not mirror STARK package/module structure as Rust modules. Package boundaries are already represented in canonical instance symbols and source provenance.

### 11.2 Visibility

Do not reimplement STARK `pub`/private rules in generated Rust.

By the time verified MIR exists:

- illegal access has been rejected;
- every remaining call is authorized;
- generated Rust item visibility is an internal code-generation choice.

Functions may remain private within the generated crate unless Rust requires otherwise.

### 11.3 Package identity

The backend treats the canonical `Instance.symbol` as authoritative.

It must not independently combine:

- package names;
- package versions;
- aliases;
- file-system paths;
- module paths.

If cross-package tests show canonical symbols are not unique for a valid resolved package graph, stop and escalate to the producer/identity owner. Do not patch the backend with a second package identity scheme.

### 11.4 Relocation and order independence

The generated callable identities must not change when:

- the workspace is copied to another absolute directory;
- files are discovered in a different host iteration order;
- dependency declarations are reordered without semantic change;
- two clean builds use different temporary output directories.

Target/toolchain/build metadata may change the build key where already specified, but must not change STARK canonical symbols.

### 11.5 Cross-package diagnostics

Preserve `SourceInfo` and `FileId` provenance from MIR. A trap inside a dependency package must report the dependency source file, not the root package file or generated Rust path.

C5.4 need not redesign trap formatting, but its package fixture must include at least one focused source-attribution test if the current harness can execute it without broadening scope.

---

## 12. Frozen three-package reference workspace

### 12.1 Location

Create a checked-in fixture under a stable path, preferably:

```text
starkc/tests/fixtures/c5-native-workspace/
```

Use the repository’s actual manifest naming and workspace layout conventions. The illustrative package names below may be adjusted once valid source/manifest syntax is confirmed.

```text
c5-native-workspace/
  <workspace manifest>
  packages/
    c5-model/
      <package manifest>
      src/lib.stark
    c5-logic/
      <package manifest>
      src/lib.stark
    c5-app/
      <package manifest>
      src/main.stark
```

### 12.2 Freeze before backend implementation

Before using the workspace as native evidence:

1. make it parse, resolve, typecheck, flow-check, and borrow-check;
2. run it through the HIR interpreter;
3. lower and verify MIR;
4. run it through the MIR interpreter;
5. require HIR/MIR agreement;
6. record the concrete `Instance.symbol` list;
7. record expected observable output/outcome;
8. then freeze the fixture.

Do not alter the fixture merely to fit the first native implementation. Any semantic change after freeze requires an explicit fixture revision note.

### 12.3 Required semantic content

The frozen workspace must contain:

- at least three STARK packages;
- at least two source files outside the root entry file;
- one public concrete non-generic cross-package call;
- one generic function defined in a dependency package;
- at least two concrete instantiations of that generic function;
- one cross-package concrete generic call from the application;
- one non-capturing function defined in a dependency package;
- one cross-package `Constant::FnPtr`;
- one function value stored in a local;
- one function value copied;
- one function value passed as a parameter;
- one function value returned, if accepted by Core/source syntax;
- one function value stored in a tuple or user struct;
- one indirect invocation;
- one instance referenced only through a function value;
- one repeated reference proving single definition emission;
- a struct;
- a payload enum or `Option`/`Result` without entering the multi-unit enum partial-move boundary;
- a `while` loop;
- deterministic observable result;
- one target-layout query;
- one checked arithmetic or cast operation;
- normal process exit.

The workspace should remain small. It is a linkage proof, not a showcase application.

### 12.4 Suggested behavioural shape

Use an architecture equivalent to:

- `c5-model`:
  - generic identity/transform function;
  - generic nominal type;
  - scalar function suitable as a function value;
- `c5-logic`:
  - higher-order `apply`;
  - function returning or forwarding a function value;
  - call into the generic dependency function;
- `c5-app`:
  - constructs values;
  - invokes direct and indirect paths;
  - validates results with `assert`/`assert_eq`;
  - produces deterministic output if the current C5 output surface supports it.

Use actual language syntax discovered from the repository. Do not modify grammar to match this conceptual arrangement.

---

## 13. Test architecture

### 13.1 Pure unit tests

Add or extend unit tests for:

- `FnPtr` Rust type emission;
- function-sentinel naming;
- signature collection through nested aggregates;
- sentinel deduplication;
- `Constant::FnPtr` name emission;
- symbol/body linkage index;
- duplicate symbol detection;
- generated-name collision detection;
- missing referenced body;
- mismatched instance identity;
- strictly sorted body requirement.

### 13.2 Generated-source tests

Inspect generated Rust only for structural claims that execution cannot isolate:

- no Rust generic parameter list for concrete STARK instances;
- one function definition per canonical symbol;
- one sentinel per signature;
- function pointer types are `fn(...) -> ...`;
- no `dyn Fn`, closure, `Box`, raw-address cast, or function registry;
- indirect call syntax uses the emitted operand;
- canonical instance comments appear deterministically;
- direct and indirect references use the same generated function name.

Generated source is not the semantic oracle.

### 13.3 Three-engine cases

Add source-level cases to `three_engine_differential.rs` or a shared multi-file extension that compare:

- direct cross-package call;
- cross-package generic instance;
- two generic instantiations;
- function constant;
- local copy and repeated indirect invocation;
- function value parameter;
- function value return if source-supported;
- function value in tuple/struct;
- function value referenced only indirectly;
- recursion or mutual recursion;
- package-source trap attribution where practical.

Every completing case must observe values through assertions or output. Exit code alone is insufficient.

### 13.4 Negative controls

Every broad positive family needs a negative control:

- false assertion proves observations execute;
- missing function body proves preflight runs;
- duplicate symbol proves duplicate validation runs;
- unsupported signature proves native-subset validation runs;
- single valid function value proves a blanket `FnPtr` refusal cannot pass;
- direct call still works after indirect-call implementation;
- two concrete generic instances prove item-only deduplication cannot pass.

### 13.5 Mutation guards

At minimum, tests must fail if a developer deliberately makes any of these mutations:

1. removes the function-value-only body from the linked body set;
2. deduplicates bodies by `item` rather than canonical symbol;
3. emits all `Constant::FnPtr` values as the same function;
4. removes type arguments from generated function names;
5. uses a non-injective underscore sanitizer;
6. emits a non-aborting arbitrary default function;
7. treats `Callee::FnValue` as a direct call to a fixed instance;
8. skips the linkage preflight;
9. emits Rust semantic generics instead of concrete bodies.

These may be conventional regression tests rather than an external mutation-testing tool, but the enforcing mechanism must be directly exercised.

### 13.6 Determinism tests

For the same semantic workspace:

- two clean builds produce byte-identical canonical symbol lists;
- two clean builds produce byte-identical generated Rust after normalizing only explicitly variable build metadata;
- relocation does not change symbols;
- output directory does not change symbols;
- dependency traversal order does not change symbols;
- repeated references do not change definition count or order.

Do not assert that the full `build.json` is identical when it intentionally records environment/toolchain identity.

---

## 14. Work-package sequence

Each sub-package gets its own dated status and evidence section in this file or in the implementation-updated repository copy.

### 14.1 C5.4a — Linkage and deterministic-symbol proof

#### Deliver

- linkage preflight/index;
- unique canonical symbol validation;
- unique generated Rust name validation;
- referenced-instance completeness validation;
- shared exhaustive instance-reference walker;
- cross-package symbol tests;
- relocation and deterministic-order tests;
- exact generated-name tests;
- preservation of existing `mangle.rs` injectivity tests.

#### Do not

- redesign `Instance.symbol`;
- create package identity in the backend;
- add Rust modules for source packages;
- implement function values yet, except the scanning needed to recognize `Constant::FnPtr`.

#### Exit

- all references resolve to exactly one body;
- duplicate/missing/mismatched instances are refused before rustc;
- deterministic symbol evidence is green;
- no canonical identity decision remains implicit.

### 14.2 C5.4b — Concrete generic-instance emission

#### Deliver

- proof that every concrete body emits exactly once;
- multi-instantiation tests;
- direct + indirect reachability deduplication;
- recursive/mutually recursive concrete-body proof;
- no Rust semantic generics;
- body-definition count and order tests.

#### Exit

- `Int32` and `Int64` instances of one generic function execute correctly;
- a same-instance multi-path reference emits once;
- a missing concrete instance is caught by linkage preflight;
- generated source contains only concrete Rust definitions.

### 14.3 C5.4c — Function values and indirect calls

#### Deliver

- `MirTy::FnPtr` emission;
- function-pointer signature collection;
- deterministic aborting sentinels;
- `default_value_expr(FnPtr)`;
- `Constant::FnPtr` emission;
- `Callee::FnValue` emission;
- local/parameter/return/aggregate storage cases;
- function-value-only reachability case;
- verifier/backend negative tests.

#### Exit

- all required function-value shapes agree across HIR/MIR/native;
- function values are copied without slots;
- indirect calls target the correct instance;
- no closure or signature-erased mechanism was introduced;
- invalid shapes fail before rustc.

### 14.4 C5.4d — Frozen reference workspace

#### Deliver

- checked-in three-package fixture;
- frozen HIR/MIR expected outcome;
- frozen canonical instance list;
- native build/run through compiler test APIs;
- cross-package generic call;
- cross-package function reference;
- indirect call;
- deterministic clean-build comparison.

This work package does not wait for the user-facing `stark build` CLI. C5.5 will route the same pipeline through the CLI.

#### Exit

- one standalone executable is produced;
- executable exits normally and produces/validates the frozen outcome;
- HIR, MIR, and native agree;
- cross-package source provenance remains correct;
- no duplicate concrete body exists;
- all referenced bodies are present.

---

## 15. Stop and escalation rules

Stop implementation and report before broadening scope when any of the following occurs.

### 15.1 MIR shape or semantic change

Stop if implementation appears to require:

- adding a new MIR variant;
- changing `Instance`;
- changing `MirTy::FnPtr`;
- adding captures/environment to `Constant::FnPtr`;
- changing `Callee::FnValue`;
- unresolved generics in MIR;
- changing verification meaning.

This is not a local backend edit.

### 15.2 Canonical identity defect

Stop if two valid resolved instances share one canonical `Instance.symbol`, or one semantic instance receives different symbols based on relocation/order.

The backend must not patch this with extra path/version components. Escalate to the canonical identity producer.

### 15.3 Source/spec ambiguity

Stop if Core v1 is unclear about:

- whether `main` can be a function value;
- function-value return syntax/semantics;
- function-value participation in Copy;
- signature identity;
- comparisons or hashing.

Use the existing spec and tests first. Do not infer from Rust behaviour.

### 15.4 Backend-language leakage

Stop if the implementation relies on:

- Rust closure capture;
- Rust trait-object dispatch;
- Rust generic monomorphisation for STARK semantics;
- Rust function-address comparison;
- host Drop as STARK Drop;
- target linker name resolution as the primary missing-body check.

### 15.5 Accepted source reaches rustc failure

Any valid source program in the approved C5.4 subset that reaches rustc and fails is a C5.4 blocker.

Classify and fix or record a deterministic pre-rustc boundary. Do not call rustc rejection an acceptable unsupported-feature diagnostic.

### 15.6 Engine divergence

Any unexplained HIR/MIR/native disagreement blocks sub-package closure.

Do not update expected output merely to make the native result pass. Determine which authority is wrong.

---

## 16. Validation cadence

### 16.1 During implementation

Run focused suites after each bounded change:

- backend unit tests;
- MIR verifier tests;
- MIR differential function-value tests;
- native function-value tests;
- package/linkage tests;
- generated-source structural tests.

### 16.2 Observable changes

Per CD-070, use:

```bash
cargo test --workspace --all-targets --no-fail-fast
```

whenever a change can alter:

- observable output;
- traps or source spans;
- diagnostics;
- snapshots;
- symbol/build-manifest values;
- Drop events;
- generated artifact identity.

### 16.3 C5.4 closure

Required:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --no-fail-fast
```

Also run:

- focused three-engine C5.4 suite;
- frozen reference workspace HIR run;
- frozen reference workspace MIR run;
- frozen reference workspace native build/run;
- deterministic two-clean-build comparison;
- relocated-workspace comparison;
- relevant snapshot/corpus integrity tests;
- adversarial review.

Record exact counts, ignored tests, binaries, host target, rustc/Cargo versions, and commit SHA.

Hosted CI remains a real C5.6 exit item. Do not claim hosted validation unless a workflow run exists.

---

## 17. Adversarial closure review

Before formal C5.4 closure, review the implementation against these questions.

1. Can a function body referenced only through `Constant::FnPtr` be omitted?
2. Can two type instantiations collapse to one Rust function?
3. Can the same instance be emitted twice through different source/import paths?
4. Can a generated function name collide after sanitisation?
5. Can workspace relocation change callable identity?
6. Can a default-initialised `FnPtr` silently execute a real function?
7. Can an indirect call ignore or replace its operand?
8. Can Rust infer/coerce a wrong function signature that MIR verification missed?
9. Can a cross-package trap report the wrong source file?
10. Can a valid unsupported-adjacent program fail only in rustc?
11. Has any Rust generic, closure, trait object, or function registry replaced STARK semantics?
12. Do tests observe actual returned values and target selection?
13. Is any closure claim based only on generated-source appearance?
14. Are the C5.3 ownership boundaries still enforced?
15. Does the full no-fail-fast run reveal stale exact-value/symbol pins outside the focused suites?

Record each disposition, including false premises that uncover different reachable defects.

---

## 18. Exact C5.4 closure conditions

WP-C5.4 is complete only when all are true.

### 18.1 Linkage

- [ ] `Instance.symbol` remains the sole canonical callable identity consumed by the backend.
- [ ] Every body symbol is unique.
- [ ] Every generated Rust function name is unique.
- [ ] Bodies are in deterministic canonical order.
- [ ] Every direct callee resolves to one body.
- [ ] Every function constant resolves to one body.
- [ ] Missing or mismatched instances fail before rustc.
- [ ] The backend performs no instance discovery or semantic monomorphisation.

### 18.2 Concrete generics

- [ ] At least two concrete instantiations of one generic function emit and execute.
- [ ] Distinct type arguments produce distinct definitions.
- [ ] Repeated reachability of one instance produces one definition.
- [ ] Cross-package generic invocation executes correctly.
- [ ] No semantic Rust generic definition represents a STARK generic.

### 18.3 Function values

- [ ] `MirTy::FnPtr` maps to a typed Rust `fn` pointer.
- [ ] `FnPtr` remains MIR-authorized `Copy`.
- [ ] Deterministic aborting sentinels provide default local values.
- [ ] `Constant::FnPtr` emits the intended concrete function.
- [ ] `Callee::FnValue` invokes the operand.
- [ ] Local, parameter, copy, return-if-supported, tuple/struct, and indirect-call cases pass.
- [ ] A function referenced only through a function value links and executes.
- [ ] Unsupported signatures fail before rustc.
- [ ] No closure, capture, raw-address, trait-object, or registry representation exists.

### 18.4 Package workspace

- [ ] The frozen fixture contains at least three packages.
- [ ] HIR and MIR agree before native qualification.
- [ ] Canonical symbols are frozen and deterministic.
- [ ] One standalone native executable builds.
- [ ] Native outcome agrees with HIR and MIR.
- [ ] Cross-package direct, generic, and function-value paths are exercised.
- [ ] Relocation does not change canonical symbols.
- [ ] Dependency/traversal order does not change canonical symbols.

### 18.5 Evidence and process

- [ ] Focused tests green.
- [ ] Generated-source structural tests green.
- [ ] Negative controls green.
- [ ] Mutation guards green.
- [ ] `cargo fmt` clean.
- [ ] `cargo clippy` clean.
- [ ] Full workspace `--no-fail-fast` suite green.
- [ ] Adversarial review dispositions recorded.
- [ ] `COMPILER-STATE.md` updated.
- [ ] This work-package document updated from PROPOSED to CLOSED with exact totals.
- [ ] No known native miscompilation or unexplained engine divergence remains in the C5.4 subset.

---

## 19. Expected repository changes

Likely files, subject to keeping responsibilities coherent:

```text
STARKLANG/docs/compiler/work-packages/WP-C5.4.md
COMPILER-STATE.md

starkc/src/backend/generated_rust/
  linkage.rs                 # new, preferred
  emit_program.rs
  emit_types.rs
  emit_bodies.rs
  mangle.rs
  mod.rs

starkc/tests/
  native_c5_4_linkage.rs
  native_c5_4_function_values.rs
  three_engine_differential.rs
  fixtures/c5-native-workspace/
```

Possible supporting changes:

```text
starkc/src/mir/verify.rs      # tests or fixes only if a real verifier gap is found
starkc/tests/mir_verify.rs
starkc/tests/mir_differential.rs
```

Do not modify MIR shape files merely for convenience.

---

## 20. Claude implementation protocol

When this document is given to an implementation agent:

1. inspect the baseline and reconcile Section 4;
2. create/check in this document at the repository path;
3. implement one sub-package at a time;
4. add the enforcing tests before or with the implementation;
5. update the sub-package status and evidence after each closure;
6. do not silently broaden the supported subset;
7. do not reinterpret recorded owner decisions;
8. stop only for a genuine escalation condition in Section 15;
9. otherwise make the best bounded implementation and report exact evidence;
10. do not mark C5.4 closed until every Section 18 checkbox is discharged or explicitly owner-deferred.

Each commit should state:

- sub-package;
- claim discharged;
- tests added;
- defects found;
- boundaries recorded;
- validation totals.

Suggested commit progression:

```text
CD-071: approve WP-C5.4 implementation plan
CD-072: close C5.4a — linkage preflight and deterministic symbols
CD-073: close C5.4b — concrete generic instances
CD-074: close C5.4c — function values and indirect calls
CD-075: close C5.4d and WP-C5.4 — frozen three-package linkage proof
```

Decision numbers are illustrative; use the next actual repository sequence.

---

## 21. Completion output

At closure, the concise project-level result should be:

> WP-C5.4 delivers deterministic native linkage for one verified multi-package `MirProgram`, exact-once emission of concrete generic instances, typed non-capturing function values, indirect calls, and a frozen three-package standalone executable whose observable behaviour agrees across HIR, MIR, and native execution.

C5.5 then exposes this already-proven pipeline through `stark build`, stable artifact placement, installation/toolchain discovery, and user-facing diagnostics.

---

## 22. Implementation status and evidence

**Status:** IN PROGRESS
**Baseline reconciled:** 2026-07-23 at `0f39579`

### 22.1 Section 4 reconciliation (baseline `0f39579`)

Verified against the actual tree before any code change. No mismatches with the §4 inventory:

| Capability | Verified state at baseline | Citation |
|---|---|---|
| Canonical instance symbol | Present; consumed, not re-derived | `mir/mod.rs` `Instance.symbol` (l.73) |
| Body determinism | `MirProgram.bodies` documented sorted by symbol | `mir/mod.rs` l.567 |
| Rust symbol escaping | Injective, round-trip-tested | `backend/generated_rust/mangle.rs` |
| Direct concrete call | Implemented for `Callee::Instance` | `emit_bodies::emit_call` l.432 |
| Concrete body emission | One Rust fn per body | `emit_program::emit` l.63 loop |
| `MirTy::FnPtr` | Exists in MIR; `emit_ty` catch-all → `Unsupported` | `emit_types::emit_ty` l.113 |
| `Constant::FnPtr` | Exists in MIR; `emit_constant` catch-all → `Unsupported` | `emit_types::emit_constant` l.418 |
| `Callee::FnValue` | Exists in MIR; `emit_call` catch-all → `Unsupported` | `emit_bodies::emit_call` l.440 |
| `is_copy(FnPtr)` | Already `true` (falls to `_ => true`); no explicit pin | `mir/mod.rs` `TypeContext::is_copy` l.547 |
| Multi-package assembly | `PackageGraph`→`parse_package_graph`→`resolve`→`lower_program` | `parser.rs` l.118, `mir/lower.rs` l.166 |

**Bounded scope note (recorded, not a scope change):** the C5.4 observation channel remains
`assert`/`assert_eq`, unchanged from C5.2/C5.3 (`NATIVE_STDOUT_SUPPORTED == false`). Neither
`println`/`RuntimeFn` output nor `Constant::Str` is a C5.4 deliverable — §5.1's required subset
does not list them and §3 forbids adjacent-scope expansion. The forward reference in CD-066
("tracing … belongs to WP-C5.4c") is superseded by this directive, which scopes C5.4c to function
values and indirect calls only. Output support remains future work.

### 22.2 Sub-package log

#### C5.4a — linkage preflight and deterministic symbols — CLOSED 2026-07-23

**Delivered:**
- `backend/generated_rust/linkage.rs`: the read-only preflight (§6). One `LinkageIndex` keyed by
  canonical symbol; `LinkageIndex::resolve` proves §6.2.4 identity consistency. It discovers,
  resolves, monomorphises, and repairs nothing (§6.4) — it validates and refuses.
- The §6.3 shared exhaustive operand/instance walker (`visit_instance_refs`), matching every
  `Statement`/`Rvalue`/`Terminator`/`Callee`/`Operand`/`Constant` variant with **no wildcard**, so
  a new MIR variant fails to compile until the walker is updated.
- Validations: strict canonical-symbol sort (§6.2.1), symbol uniqueness (§6.2.2), generated-name
  uniqueness (§6.2.3), referenced-instance completeness + identity match (§6.2.4), exactly one
  entry (§6.2.5), no non-entry→`main` (§6.2.6), param/return C5-representability via the shared
  `emit_ty` oracle (§6.2.7).
- Wired into `emit_program::emit` as the deterministic pre-rustc refusal boundary. Existing
  `mangle.rs` injectivity/round-trip tests preserved unchanged.

**Evidence (`starkc/tests/native_c5_4_linkage.rs`, 12 tests, all green):** hand-built refusals for
unsorted / duplicate / missing-body (direct AND function-value) / one-symbol-two-identities /
no-entry; the function-value-only reference resolving through the same path (§10.5 in miniature);
the shared walker collecting both ref kinds; and source-driven **real cross-package** proofs — a
two-package workspace links and **runs natively** (exit 0, cross-package symbol package-qualified
and emitted exactly once), relocation to a different absolute path leaves canonical symbols
byte-identical (§11.4/§13.6), and two clean lowerings agree and are already in canonical order.

**No regressions:** native C5.1b/2b/2c/2d/2e/3 + three-engine (57) suites green; `cargo fmt`/`cargo
clippy --tests` clean.

**Boundaries recorded:** the generated-name-collision and non-entry→`main` checks are
structurally unreachable from valid input (mangle injectivity + `function_name_for_symbol`
returning `"main"` only for `ENTRY_SYMBOL`); they are retained as defensive backstops guarding a
future non-injective sanitiser (mutation guard §13.5 #5), not as reachable refusals. Duplicate
identical symbols are caught by the strict-sort guard before the duplicate-insert guard; both
refuse.

#### C5.4b — concrete generic-instance emission — CLOSED 2026-07-23

**Finding:** monomorphisation already occurs in lowering and `emit_program::emit` already emits one
concrete Rust `fn` per body with a sanitised canonical-symbol name and no Rust generic parameter
list. C5.4b required **no backend code change** — it is the proof that the existing emission
satisfies §10. The one function-value-crossed generic case (§10.4 #4/#8) and the §10.5
function-value-only reachability case require native `Constant::FnPtr` emission and are delivered
in **C5.4c**, where that capability exists; C5.4a already proved their *linkage* resolution.

**Evidence:**
- `three_engine_differential.rs` (+4 cases, full suite 61 green): a value (not just `size_of`)
  carried through `identity<T>` at Int32 **and** Int64; a recursive generic instance
  (`depth@[Int32]` calling itself); mutual recursion among concrete bodies (`is_even`/`is_odd`);
  one instance reached by two call paths computing correctly.
- `native_c5_4_generics.rs` (3 tests, green) — generated-source structure via `emit_program::emit`
  (no rustc): distinct type args → two distinct concrete definitions with the same item and
  different type_args (defeats item-only dedup, guard §13.5 #2); one shared instance emitted
  exactly once (§10.4 #5); a recursive instance is one self-calling definition; and **no generated
  `fn` carries a Rust generic parameter list** (guard §13.5 #9), asserted by scanning every `fn `
  header for a `<` before its `(`.

**Exit conditions (§14.2) discharged:** Int32/Int64 instances execute correctly; same-instance
multi-path reference emits once; a missing concrete instance is caught by the C5.4a linkage
preflight; generated source contains only concrete Rust definitions. `cargo fmt`/`clippy` clean.

