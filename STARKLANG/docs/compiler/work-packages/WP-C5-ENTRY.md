# WP-C5-ENTRY — Native Core Backend MVP Execution Plan

**Status:** APPROVED (owner, 2026-07-21, CD-042) — entry plan cleared for WP-C5.1 execution  
**Prepared:** 2026-07-21  
**Repository baseline:** `f87402d6f299f572599b44e9f424f2157050af21`  
**Current position:** Gate C4 closed; Gate C5 open  
**Initial backend:** generated Rust consuming verified MIR  
**MIR contract:** STARK MIR v0.1, frozen for backend consumption  
**Escalation:** CE4 owner approval — RECORDED (CD-042). The §19 decision table is approved at its recommended choices.

---

## 0. Purpose

This document turns the Gate C5 roadmap into an implementation-ready plan.

It freezes:

1. the exact Native Core MVP supported subset;
2. the generated-Rust representation contract;
3. the ownership, move, Drop, and partial-initialisation strategy;
4. the target-layout and `LayoutQuery` strategy;
5. the generated crate and runtime topology;
6. the build-command behaviour;
7. the incremental work-package sequence;
8. the native differential test matrix;
9. the stop and escalation rules;
10. the evidence required to close Gate C5.

This document does **not**:

- change Core v1 language semantics;
- reopen Gate C4;
- authorize a MIR v0.1 shape change;
- claim full native Core parity;
- replace Gate C6;
- select a direct Cranelift backend;
- create a stable public native ABI;
- require every cleanly rejected Core combination to be implemented before native compilation begins.

The goal is a bounded native MVP:

> One normal multi-file, multi-package STARK application compiles from verified MIR through generated Rust into a standalone executable, with native behaviour matching the HIR and MIR interpreters for the approved C5 subset.

---

## 1. Authority and fixed inputs

C5 implementation must obey the following documents, in descending order of authority:

1. normative Core v1 specification and abstract machine;
2. approved compiler decisions recorded in `COMPILER-STATE.md`;
3. frozen `STARK MIR v0.1` contract;
4. `COMPILER-CHARTER.md`;
5. `COMPILER-ROADMAP.md`;
6. this entry plan;
7. implementation notes and generated-Rust source.

If this plan conflicts with the normative language specification or approved MIR contract, this plan is wrong.

### 1.1 Fixed decisions entering C5

The following are not reopened by this work package:

- native compilation is mandatory;
- the initial production backend is generated Rust;
- generated Rust consumes **verified MIR**, not HIR;
- MIR is monomorphised-only;
- STARK owns instance discovery and semantic monomorphisation;
- verified MIR contains no unresolved type parameters or inference variables;
- all trait and method dispatch in Core v1 is static;
- function values are non-capturing code references;
- traps and panic abort without unwinding;
- destructors do not run after an aborting trap;
- MIR carries explicit evaluation order, move semantics, Drop points, traps, and source provenance;
- MIR runtime surface remains `0.1-A8` unless separately approved;
- any post-C4 MIR shape change requires a MIR version bump and owner review.

### 1.2 What Rust owns in the MVP

Rust is the implementation vehicle for C5, not the semantic authority.

Rust may own:

- machine-code generation;
- target calling convention inside the generated executable;
- register allocation;
- object generation and linking;
- target-specific physical layout used by the generated-Rust backend;
- standard allocator integration;
- debug information for generated Rust;
- optimisation performed by rustc, although C5 uses the debug profile only.

Rust must **not** silently decide:

- STARK evaluation order;
- whether a value is moved or copied;
- when a STARK destructor executes;
- which fields remain live after a partial move;
- STARK overflow or cast behaviour;
- trap category;
- whether a user method or trait implementation is called;
- source-level package or symbol identity;
- whether cleanup occurs after a trap.

Those decisions are already explicit in verified MIR.

---

## 2. Gate C5 outcome

Gate C5 closes when all of the following are true:

1. `stark build` accepts the frozen C5 reference workspace.
2. The compiler verifies MIR before invoking the backend.
3. The backend emits a deterministic generated Rust crate.
4. The compiler invokes Cargo/rustc without requiring the user to do so manually.
5. A standalone executable is produced at a documented output path.
6. The reference application contains at least three STARK packages.
7. The application contains a cross-package call to a monomorphised generic instance.
8. The application exercises the approved C5 baseline in Section 3.
9. HIR, MIR, and native execution agree on stdout, stderr, exit status, traps, and the C5 Drop evidence.
10. The approved C5 execution-snapshot subset replays through the three-engine harness.
11. Unsupported native features fail with a STARK diagnostic before generated Rust compilation.
12. No known native miscompilation, invalid MIR acceptance, ownership unsoundness, or unexplained engine divergence remains in the supported C5 subset.
13. A C5 exit report records the exact supported subset and all deferred native features.

C5 does not claim full Core v1 native conformance. That is Gate C6 work.

---

## 3. Frozen C5 supported subset

The C5 baseline is intentionally smaller than full Core v1 native parity.

### 3.1 Required in C5

| Area | Required native support |
|---|---|
| Compilation unit | One verified `MirProgram` with source table, bodies, nominal type context, and entry instance |
| Primitive types | Unit, Bool, Char, signed and unsigned integers required by the corpus, Float32/Float64 where already present in the frozen C5 cases |
| Constants | Primitive constants, string constants, enum discriminants, function-instance constants |
| Operations | MIR pure operations and checked operations used by the frozen C5 cases |
| Control flow | Basic blocks, branches, switches, loops, break/continue-lowered control flow, returns |
| Functions | Direct calls to concrete instances |
| Generics | Concrete monomorphised functions and nominal instances supplied by MIR |
| Aggregates | Tuples, fixed arrays used by the workload, simple structs |
| Enums | User enums, `Option`, `Result`, payload variants, discriminants |
| Patterns | Already-lowered MIR control flow and projections; the backend does not reimplement pattern semantics |
| Errors | `?` paths as represented in MIR |
| Function values | Non-capturing function references, local storage, aggregate storage, copying, indirect invocation |
| Packages | Deterministic cross-package symbols and one standalone link unit |
| Output | Existing MIR string-output operations and the minimum String path required by the reference application |
| Display | User `Display::fmt` works when already represented as ordinary MIR calls followed by string output |
| Traps | Deterministic category, source location, stderr format, exit behaviour, and abort |
| Layout queries | Real target-dependent `size_of`/`align_of` answers for every C5-supported `MirTy` |
| Ownership | Moves/copies and explicit Drops needed by the frozen C5 application and dedicated Drop fixture |
| Debug build | `stark build` produces a native debug executable |

### 3.2 Explicitly deferred to C6 or later

The following do not block C5 unless the frozen C5 reference application is changed to require them:

- full partial-move and Drop parity across every Core construct;
- by-value iteration over arrays with non-`Copy` elements;
- broad `Vec` and iterator parity;
- complete slice and mutable-slice native parity;
- `Box` native parity beyond any minimal case separately admitted;
- HashMap and HashSet native representation;
- all standard-library String operations;
- broad formatting parity beyond the C5 Display fixture;
- every generic/trait interaction accepted by the front end;
- trait objects;
- closures;
- async;
- concurrency;
- general FFI;
- executable provider/file-resource support;
- Windows support;
- release builds and optimisation;
- incremental compilation;
- stable ABI guarantees;
- direct Cranelift generation.

A deferred feature must either:

1. be rejected before backend invocation with a named diagnostic; or
2. be absent from the C5 accepted source profile.

It must not fail as a Rust compiler error, backend panic, malformed generated source, or native divergence.

### 3.3 Known front-end limitations entering C5

The following remain documented limitations and do not reopen C4:

- DEV-083: receiver inference through concrete positions of a generic impl head;
- DEV-088: cross-file `const` use;
- DEV-090: by-value iteration over arrays whose element type is not `Copy`.

C5 must preserve their deterministic front-end rejection. The backend must never receive these programs.

---

## 4. Frozen C5 reference workspace

Create a checked-in reference workspace under a dedicated compiler test fixture, for example:

```text
starkc/tests/fixtures/c5-native-workspace/
  Stark.toml
  packages/
    c5-model/
      Stark.toml
      src/lib.stark
    c5-logic/
      Stark.toml
      src/lib.stark
    c5-app/
      Stark.toml
      src/main.stark
```

Exact package names may change, but the structure and exercised semantics are frozen once the workload is approved.

### 4.1 Required application content

The workspace must exercise:

- at least three packages;
- public and private items already validated by the front end;
- one generic nominal instantiated concretely;
- one generic function instantiated concretely across a package boundary;
- a struct;
- a payload enum;
- `Option` or `Result`;
- a match lowered to MIR control flow;
- a loop;
- a direct call;
- a function value copied and invoked indirectly;
- a cross-package function reference;
- deterministic printed output;
- at least one target-layout query;
- at least one checked arithmetic or cast operation;
- normal process exit.

A separate native fixture must exercise an aborting trap so the successful application is not forced to terminate abnormally.

A separate native Drop fixture must contain one non-`Copy` value with observable destruction, proving that:

- it is moved at most once;
- it is destroyed exactly once;
- the observable Drop order agrees with HIR and MIR;
- no destructor runs after an aborting trap.

### 4.2 Baseline-output freeze

Before C5.2 begins:

1. run the workspace through the HIR interpreter;
2. run it through the MIR interpreter;
3. require agreement;
4. store the expected observable result in the execution-snapshot format;
5. record the corpus version and hashes;
6. assign stable case identifiers.

The C5 workload may grow only through a recorded corpus-version bump.

---

## 5. Backend architecture

The production C5 pipeline is:

```text
STARK source
  -> parse / resolve / type / flow / borrow check
  -> typed HIR
  -> instance discovery and monomorphisation
  -> MIR lowering
  -> MIR verification
  -> generated-Rust backend
  -> generated Cargo crate
  -> cargo/rustc
  -> standalone executable
```

The backend entry point must accept only a successfully verified `MirProgram`.

Conceptually:

```rust
pub fn emit_native_debug(
    program: &VerifiedMirProgram,
    options: &NativeBuildOptions,
) -> Result<NativeArtifact, BackendDiagnostic>;
```

The actual Rust API may differ, but it must encode the verified-program precondition rather than accepting an arbitrary unverified `MirProgram`.

### 5.1 Suggested module boundaries

Use a small backend surface with responsibilities separated enough for review:

```text
starkc/src/backend/
  mod.rs
  generated_rust/
    mod.rs
    emit_program.rs
    emit_types.rs
    emit_bodies.rs
    emit_places.rs
    emit_runtime.rs
    mangle.rs
    source_map.rs
    build.rs
```

This is a responsibility map, not a requirement to create every file immediately. Avoid a second oversized all-purpose lowering module.

### 5.2 No backend semantic reconstruction

The emitter must not:

- inspect HIR to recover meaning;
- redo trait selection;
- redo overload or method resolution;
- infer generic arguments;
- recompute Drop placement;
- reconstruct source-level patterns;
- choose a different trap for convenience;
- use Rust expression evaluation order as an implicit substitute for MIR sequencing.

Every MIR statement and terminator is emitted in explicit block order.

---

## 6. Generated-Rust representation contract

This section is a proposed CE4 decision and must be approved before broad implementation.

### 6.1 Representation principles

1. Generated definitions are internal to one compiler/runtime version.
2. They are not a stable external ABI.
3. The provider ABI is separate and uses explicitly specified C-compatible types.
4. Type names and symbols are deterministic for identical compiler inputs.
5. STARK source names are sanitised and never trusted as raw Rust identifiers.
6. Every concrete `MirTy` maps to one canonical generated Rust type in a build.
7. The backend emits concrete instances; it does not emit semantic Rust generics for STARK monomorphisation.
8. Rust layout is backend-, target-, and toolchain-version dependent, as permitted by Core’s target-layout query contract.
9. Any `unsafe` code is isolated behind a very small reviewed helper module and justified by explicit invariants.

### 6.2 Type mapping

| MIR type | C5 generated-Rust representation |
|---|---|
| Unit | `()` |
| Bool | `bool` |
| Char | `char` |
| Int8/16/32/64 | `i8/i16/i32/i64` |
| UInt8/16/32/64 | `u8/u16/u32/u64` |
| Float32/64 | `f32/f64` using the existing canonical-float policy |
| Tuple | Generated concrete tuple or named internal aggregate; choose one canonical form |
| Array<T, N> | `[Generated<T>; N]` |
| User struct instance | One concrete generated Rust struct |
| User enum instance | One concrete generated Rust enum |
| Option<T> | One canonical generated representation; ordinary `Option<T>` is preferred for C5 if all observable semantics match |
| Result<T, E> | One canonical generated representation; ordinary `Result<T, E>` is preferred for C5 if all observable semantics match |
| String | C5 runtime-owned or Rust `String`, according to the approved Drop strategy |
| `str`/string view | Borrowed string view used only where MIR reference semantics permit it |
| Reference | Internal reference/pointer representation chosen in C5.1; broad parity remains C6 |
| Slice | Internal fat-view representation when admitted into the C5 subset |
| Function value | Typed non-capturing Rust `fn` pointer or generated equivalent |
| Box | Deferred unless explicitly added to the C5 subset |
| Vec | Deferred except for a separately approved minimal path |
| Opaque provider resource | Not an ordinary Rust pointer; separate Native Provider ABI handle |

Do not apply `#[repr(C)]` to all internal language types merely for convenience. Internal generated layout and external provider ABI layout are different concerns.

### 6.3 Nominal type generation

For every reachable concrete nominal instance:

- emit exactly one Rust definition;
- derive no semantic traits unless their behaviour is proven observationally equivalent;
- do not derive `Drop`;
- do not derive `Clone`, `Copy`, `Eq`, `Ord`, or `Hash` as a shortcut for STARK semantics;
- emit explicit helper functions where MIR requires semantic operations;
- retain a reverse map from generated definition to STARK package/module/item and concrete type arguments.

### 6.4 Function and symbol generation

Every MIR instance gets one deterministic symbol.

The mangling input must include:

- package identity;
- module path;
- item identity;
- concrete type arguments;
- method-own type arguments where applicable;
- MIR version;
- a disambiguator when two legal source items otherwise collide.

Requirements:

- deterministic across identical inputs;
- injective within one linked program;
- independent of hash-map iteration order;
- valid as a Rust identifier after encoding;
- not advertised as a stable external ABI.

Keep a human-readable comment above each generated function with its canonical STARK instance name and source location.

---

## 7. Ownership, moves, and Drop strategy

This is the highest-risk C5 design area.

### 7.1 Required rule

Verified MIR is the authority for value liveness and destruction.

Generated Rust must not add a second, implicit destruction schedule.

### 7.2 Proposed C5 strategy

Use ordinary Rust values for trivially `Copy` MIR types.

Use controlled manual storage for non-`Copy` MIR locals and temporaries:

```text
MaybeUninit<ManuallyDrop<T>>
+ explicit initialised/live state where required
+ explicit move helpers
+ explicit Drop helpers
```

The precise generated shape may be simplified when MIR dataflow proves a local is always initialised and never partially moved, but such simplification must be evidence-based and must not change observable Drop behaviour.

### 7.3 Move operation

A MIR move from a non-`Copy` place must:

1. prove the source place is live through verified MIR assumptions;
2. transfer the bytes/value without running Drop at the source;
3. mark the source place or drop unit dead;
4. initialise the destination;
5. preserve sub-place precision already present in MIR move/drop paths.

The generated implementation may use a reviewed `ptr::read`-style helper inside the isolated unsafe module.

It must not use `.clone()` to implement a move.

### 7.4 Copy operation

A MIR copy is emitted only for a MIR type classified as `Copy`.

The backend does not independently broaden the Copy set based on Rust traits.

### 7.5 Drop operation

A MIR `Drop(place)` must:

1. act only on a live place;
2. execute the correct concrete user Drop instance or generated structural glue;
3. destroy fields in the MIR-specified order;
4. mark the place/drop unit dead;
5. execute exactly once;
6. never be repeated automatically by Rust scope exit.

Generated nominal types must not implement Rust `Drop` in C5 unless a later approved design proves that doing so cannot conflict with explicit MIR Drop.

### 7.6 Partial moves

C5 does not need to support every partial-move combination natively, but every partial move present in the approved C5 fixtures must use MIR’s typed move and drop paths.

A backend must not collapse:

- struct fields;
- enum variant fields;
- constant array indices;

into whole-local liveness when MIR has already distinguished them.

### 7.7 Trap paths

An aborting trap:

- submits the complete trap record;
- does not run pending Drop glue;
- does not unwind;
- terminates the process through the runtime abort path.

The generated crate must compile with panic-abort behaviour. Generated code must not rely on Rust unwinding for STARK control flow.

### 7.8 Unsafe-code containment

All backend-generated unsafe operations must route through a small fixed helper module, for example:

```text
stark_backend_rt::value_move
stark_backend_rt::value_drop
stark_backend_rt::place_read
stark_backend_rt::place_write
```

The helper module must document:

- initialisation requirements;
- aliasing requirements;
- live/dead flag transitions;
- permitted source and destination overlap;
- exact Drop responsibility.

No arbitrary MIR expression should emit ad hoc unsafe blocks.

---

## 8. Layout and `LayoutQuery`

### 8.1 C5 layout authority

C5 generated Rust uses the actual layout of the generated Rust representation for the selected:

- target triple;
- rustc version;
- backend version;
- runtime version;
- build profile.

Core v1 does not promise one stable layout across those inputs.

The result must nevertheless be deterministic within a recorded build environment.

### 8.2 Layout-query emission

For:

```text
Rvalue::LayoutQuery { SizeOf, ty }
Rvalue::LayoutQuery { AlignOf, ty }
```

the backend emits a compile-time Rust query against the canonical generated representation:

```rust
core::mem::size_of::<GeneratedTy>() as u64
core::mem::align_of::<GeneratedTy>() as u64
```

or an observationally equivalent constant generated by one central layout service.

The backend must not:

- return the C4 interpreter placeholder `(8, 8)`;
- erase the queried type;
- maintain separate layout calculations in unrelated emitter modules;
- answer using a provider-ABI type when the query concerns an internal STARK value type.

### 8.3 Layout validation

C5 tests must include:

- all primitive widths and alignments used by the target;
- a tuple;
- a fixed array;
- a user struct;
- a payload enum;
- `Option`;
- `Result`;
- a function value;
- every additional C5-supported runtime value.

For each case, compare:

1. the value observed by the native STARK program;
2. the generated Rust compile-time query;
3. any compiler-side recorded layout metadata.

A mismatch is a C5 blocker.

### 8.4 Provider ABI separation

The Native Provider ABI must use its own explicit layout contract.

Internal generated Rust values must never cross the provider boundary directly unless the ABI document explicitly defines that representation.

---

## 9. Minimal C5 runtime

Create a small versioned runtime crate or module consumed by the generated crate.

Suggested location:

```text
stark-runtime/
  Cargo.toml
  src/
    lib.rs
    output.rs
    trap.rs
    value.rs
    provider_abi.rs
```

The exact repository path is an implementation decision, but the runtime must be independently versioned from MIR.

### 9.1 Required runtime responsibilities

C5 runtime must provide only what the MVP uses:

- stdout byte submission;
- stderr byte submission;
- newline-appending variants;
- trap record formatting;
- process abort;
- minimal String bridging required by generated code;
- reviewed move/drop helper functions if they cannot remain generated locally;
- runtime/compiler compatibility check;
- source-file/span lookup needed by traps.

### 9.2 Runtime version identity

Every generated crate records:

- STARK compiler version;
- MIR version;
- MIR runtime-surface version;
- native runtime ABI version;
- backend version;
- rustc version;
- target triple;
- debug profile.

A runtime-version mismatch must fail before user code executes.

### 9.3 Not in the minimal runtime

Do not add during C5 unless required by the frozen workload:

- networking;
- async runtime;
- scheduler;
- general dynamic linking;
- plugin registry;
- package registry;
- broad collection library;
- reflection;
- garbage collector;
- JIT;
- provider callbacks;
- concurrency.

---

## 10. Native Provider ABI v0.1

C5.1 owns the **specification** of Native Provider ABI v0.1.

Provider execution does not block the C5 MVP unless separately added to the frozen C5 workload.

### 10.1 Required ABI document

Create:

```text
STARKLANG/docs/compiler/native-provider-abi-v0.1.md
```

It must specify:

- provider identity and semantic version;
- integrity hash and origin metadata;
- supported target triples;
- capability declaration;
- exported function table;
- opaque resource-handle representation;
- ownership transfer in both directions;
- borrowed immutable buffer representation;
- borrowed mutable buffer representation if admitted;
- error return representation;
- distinction among provider error, STARK trap, and host failure;
- resource close/Drop semantics;
- blocking declaration;
- callback prohibition;
- compiler/runtime/provider compatibility checks;
- no concurrent callbacks;
- no direct crossing of internal generated Rust aggregates.

### 10.2 C5 implementation boundary

C5 may include a compile-time ABI validator and a mock provider metadata fixture.

Actual file I/O or resource-provider execution may remain deferred to its owning later package, provided the ABI design is approved and no C5 claim says providers run natively.

---

## 11. Generated crate topology

For each build, generate a deterministic crate tree under the STARK target directory.

Recommended shape:

```text
target/stark/debug/<build-key>/
  Cargo.toml
  build.json
  src/
    main.rs
    stark_types.rs
    stark_functions.rs
    stark_runtime_bindings.rs
    stark_source_map.rs
  target/
```

The implementation may split files further as size grows.

### 11.1 Build key

The build key should include at least:

- package graph identity;
- entry package and target;
- source-content hashes;
- compiler version;
- MIR version;
- native backend version;
- runtime version;
- rustc version;
- target triple;
- profile.

C5 does not require an incremental cache. The key is for isolation, diagnostics, and future C7 work.

### 11.2 Generated Cargo manifest

The generated manifest must:

- pin the local STARK runtime dependency;
- use the debug profile;
- configure aborting panic;
- avoid network dependency resolution;
- avoid unapproved crates;
- record generated metadata where Cargo permits it;
- produce one binary with a deterministic internal crate name.

### 11.3 Offline rule

`stark build` must not fetch arbitrary dependencies from the network.

All backend/runtime dependencies required for the generated crate must be:

- part of the installed STARK toolchain;
- local path dependencies;
- or otherwise pinned and provisioned before the build.

A missing local runtime is a compiler installation error, not a Cargo dependency-resolution adventure.

---

## 12. `stark build` behaviour

### 12.1 Command

C5 introduces:

```bash
stark build
```

Optional C5 debugging flags may include:

```bash
stark build --keep-generated
stark build --emit-rust
stark build --verbose
```

Target selection and release builds remain C7 unless a target override is required internally for testing.

### 12.2 Build stages

The command performs:

1. workspace discovery;
2. package-graph validation;
3. front-end analysis;
4. HIR construction;
5. MIR instance discovery and lowering;
6. MIR verification;
7. native-subset validation;
8. generated-crate emission;
9. Cargo/rustc discovery;
10. generated-crate compilation;
11. executable placement;
12. build-manifest recording;
13. concise success output.

### 12.3 Output path

Use a predictable user-facing path, for example:

```text
target/stark/debug/<binary-name>
```

The internal generated crate may live under a hashed subdirectory, but the final artifact path must be stable and clearly printed.

### 12.4 Toolchain discovery

C5.1 must approve:

- the minimum supported rustc version;
- the minimum supported Cargo version;
- how STARK locates them;
- how a toolchain override is supplied for testing;
- the error shown when they are missing;
- whether the installed STARK distribution bundles or requires Rust.

The initial implementation should test at least the toolchain versions already used by the compiler project’s supported validation matrix.

### 12.5 Rust compiler failures

User source errors must be rejected before rustc.

A rustc failure after verified MIR is generated is classified as one of:

- STARK backend defect;
- unsupported target/toolchain;
- STARK runtime installation defect;
- generated-crate environmental failure.

It must not be presented as an ordinary user type error.

When rustc fails, report:

- concise STARK-facing classification;
- relevant STARK source origin where available;
- generated file and line;
- command exit status;
- retained generated path when requested.

---

## 13. Source mapping and traps

### 13.1 Source map

Generate a span table mapping a compact native span ID to:

- STARK `FileId`;
- file path;
- start line/column;
- end line/column;
- function instance;
- optional MIR block and statement/terminator index.

The generated crate carries only compact IDs at trap sites.

### 13.2 Trap ABI

A native trap record must include:

- trap category;
- source span ID;
- optional approved message;
- compiler/runtime version metadata if diagnostic mode requires it.

The runtime resolves the ID and prints the canonical STARK trap format.

### 13.3 Required trap cases

C5 native tests include at least:

- integer overflow;
- division by zero;
- invalid shift when in the subset;
- out-of-bounds indexing when in the subset;
- failing cast when in the subset;
- explicit panic/trap.

Compare:

- category;
- source file;
- source line;
- stderr;
- exit status;
- absence of post-trap Drop output.

---

## 14. Detailed work-package sequence

The roadmap’s C5.1–C5.6 numbering remains authoritative. Use the following bounded increments.

### WP-C5.1 — Runtime ABI and layout design

#### C5.1a — Representation decision

Deliver:

- approved version of Sections 6–10;
- exact C5 supported `MirTy` matrix;
- exact non-`Copy` storage strategy;
- exact move and Drop invariants;
- exact enum/Option/Result representation;
- exact function-pointer representation;
- exact layout-query rule;
- host target for the first native proof.

No broad backend emission begins before this decision.

#### C5.1b — Backend/runtime skeleton

Deliver:

- backend module;
- version constants;
- minimal runtime crate/module;
- generated crate skeleton;
- build-manifest schema;
- one generated empty `main` program compiled natively.

#### C5.1c — Native Provider ABI specification

Deliver the v0.1 ABI document and validation fixture. No provider feature expansion.

#### C5.1 exit

Done when:

- CE4 decision is recorded;
- one verified empty/scalar MIR program becomes a standalone executable;
- runtime/backend/compiler version checks are demonstrated;
- no language semantics are hidden in the runtime.

### WP-C5.2 — Scalar native lowering

#### C5.2a — Primitive values and constants

Implement canonical type and constant emission.

#### C5.2b — Locals, places, assignments, copies, and moves

Implement the C5 place subset and approved storage strategy.

#### C5.2c — Operations and control flow

Implement pure rvalues, checked terminators, branches, switches, loops, and returns.

#### C5.2d — Direct functions and calls

Implement concrete instances, parameters, return destinations, and call continuations.

#### C5.2e — Trap path

Implement native trap records, source IDs, abort, and no-unwind configuration.

#### C5.2 exit

Require three-engine agreement for:

- scalar arithmetic;
- branches;
- loops;
- direct calls;
- successful checked operations;
- each admitted trap category.

### WP-C5.3 — Aggregates, enums, and error values

#### C5.3a — Tuples, arrays, and structs

Implement construction, field projection, copying/moving, and layout queries.

#### C5.3b — Enums and discriminants

Implement user enums, payload access, discriminant switching, and verifier-assumed variant correctness.

#### C5.3c — Option, Result, matches, and `?`

Emit the already-lowered MIR control flow without reconstructing source patterns.

#### C5.3d — Bounded Drop proof

Implement explicit Drop for the types required by the C5 fixtures and prove exactly-once behaviour.

#### C5.3 exit

Require three-engine agreement for:

- aggregate values;
- payload variants;
- match paths;
- Option/Result;
- `?`;
- target layout queries;
- the dedicated C5 Drop fixture.

### WP-C5.4 — Multi-file/package linkage

#### C5.4a — Deterministic symbols

Implement package/module/item/type-argument mangling and collision tests.

#### C5.4b — Concrete generic instances

Emit monomorphised MIR bodies as concrete Rust definitions. Do not emit semantic Rust generics.

#### C5.4c — Function values

Implement:

- function-instance constants;
- local storage;
- copying;
- aggregate fields;
- indirect calls;
- cross-package function references.

#### C5.4d — Reference workspace

Compile and link the frozen three-package application.

#### C5.4 exit

Require:

- identical symbols across two clean identical builds;
- no duplicate concrete instance emission;
- one standalone executable;
- correct cross-package generic call;
- correct indirect invocation.

### WP-C5.5 — Debug build experience

#### C5.5a — User command

Implement `stark build` orchestration and final artifact placement.

#### C5.5b — Diagnostics

Implement:

- missing Rust toolchain diagnostic;
- unsupported native feature diagnostic;
- rustc/backend failure classification;
- generated-source retention;
- source-mapped traps.

#### C5.5c — Installation and offline behaviour

Prove the runtime is found locally and generated builds do not fetch arbitrary dependencies.

#### C5.5 exit

A user can build the reference workspace without invoking Cargo manually and can locate/run the executable from the printed output path.

### WP-C5.6 — Gate exit

Run:

- complete workspace tests;
- MIR verification suite;
- C5 three-engine differential suite;
- approved execution-snapshot subset;
- reference workspace build and run;
- trap matrix;
- layout matrix;
- symbol determinism checks;
- runtime-version mismatch check;
- clean build from a relocated workspace;
- formatting and clippy on supported compiler toolchains;
- configured hosted CI.

Write:

```text
starkc/docs/compiler/C5-exit-report.md
```

The report must state:

- exact head;
- target triple;
- rustc/Cargo versions;
- backend/runtime/MIR versions;
- exact C5 supported subset;
- exact deferred native features;
- test and snapshot counts;
- reference workspace structure;
- native artifact path;
- three-engine comparison results;
- open deviations and owning future gates;
- one of:

  NATIVE-CORE-MVP
  NATIVE-CORE-MVP-WITH-LISTED-DEVIATIONS
  NATIVE-CORE-MVP-NOT-YET

---

## 15. Test architecture

### 15.1 Native differential harness

Extend the existing differential model:

```text
source case
  -> HIR interpreter result
  -> verified MIR interpreter result
  -> native debug executable result
  -> comparator
```

Compare:

- stdout bytes;
- stderr bytes;
- exit code;
- trap category;
- trap source file and line;
- observable Drop events;
- build success/failure classification.

### 15.2 Test groups

Create distinct suites:

```text
native_scalar.rs
native_traps.rs
native_aggregates.rs
native_layout.rs
native_drop.rs
native_function_values.rs
native_packages.rs
native_build_cli.rs
native_snapshot_replay.rs
```

Names may differ, but the categories must remain visible.

### 15.3 Generated-source tests

Use focused golden tests for:

- symbol mangling;
- type definitions;
- basic-block structure;
- trap calls;
- source-map entries;
- runtime imports.

Do not use generated-Rust snapshots as the semantic oracle. Semantic correctness comes from execution comparison.

### 15.4 Negative tests

At minimum test:

- backend invoked with unverified MIR is impossible or rejected;
- unsupported `MirTy` fails with a STARK diagnostic;
- unsupported `RuntimeFn` fails before rustc;
- symbol collision is rejected;
- runtime version mismatch is rejected;
- invalid target is rejected;
- missing rustc/Cargo is diagnosed;
- rustc failure is classified as backend/environmental;
- non-`Copy` array iteration remains a front-end error;
- cross-file `const` use remains a front-end error.

---

## 16. Stop and escalation rules

Implementation stops and reports to the owner when any of the following occurs.

### 16.1 MIR-contract blocker

Stop if the C5 baseline cannot be represented faithfully in MIR v0.1.

Do not patch around it in generated Rust.

Required response:

- minimal reproducer;
- HIR and MIR evidence;
- exact missing MIR concept;
- whether a MIR v0.2 proposal is required;
- impact on the future direct backend.

### 16.2 Ownership or Drop blocker

Stop if safe or bounded-unsafe generated Rust cannot preserve MIR move/Drop semantics without:

- double destruction;
- leak of a value that MIR destroys;
- use after move;
- hidden automatic cleanup;
- cleanup after abort;
- loss of partial-move precision.

Do not replace moves with clones.

### 16.3 Engine divergence

Stop on a supported case when:

```text
HIR != MIR
HIR != native
or
MIR != native
```

Classify whether the defect is in:

- HIR oracle;
- MIR lowering;
- MIR interpreter;
- MIR verifier;
- generated backend;
- native runtime;
- specification.

### 16.4 Layout blocker

Stop if:

- `LayoutQuery` cannot be answered for a C5-supported type;
- native observed layout differs from the generated representation;
- internal layout is accidentally crossing the provider ABI;
- results vary without a recorded target/toolchain/version change.

### 16.5 Rust impedance blocker

Stop if accepted STARK semantics require generated Rust that the chosen safe representation rejects and the proposed unsafe solution lacks a small auditable invariant.

Do not broaden unsafe emission throughout the backend.

### 16.6 Scope control

Do not reopen a broad C4 audit because C5 discovers a cleanly rejected front-end combination.

Do not add a feature merely because rustc makes it easy.

Do not add an optimisation unless necessary for correctness or bounded compilation.

Do not begin C6 parity work while C5’s standalone baseline remains incomplete.

---

## 17. Defect classification during C5

Every discovered defect must be classified before implementation expands.

| Class | Default disposition |
|---|---|
| Generated-Rust syntax/emission defect | Fix in current C5 increment |
| Build orchestration defect | Fix in C5.5 or earlier if blocking |
| Native runtime defect | Fix if inside C5 runtime surface |
| Source-map diagnostic defect | Fix if required by C5.5 |
| Unsupported but cleanly rejected Core feature | Record for C6/later; do not automatically expand C5 |
| Front-end over-rejection with workaround | Record and assign; does not automatically reopen C4 |
| HIR/MIR divergence | Stop and investigate |
| Native/MIR divergence | Stop and investigate |
| Ownership/Drop unsoundness | Stop and fix before proceeding |
| MIR contract inadequacy | Owner escalation; possible MIR v0.2 |
| Spec ambiguity | Use the approved spec-decision protocol |
| Provider feature request | Defer unless required by the frozen C5 workload |

Track defect-class frequency during C5. A sustained stream of MIR-contract or ownership defects is an architecture warning; a stream of ordinary emitter/build defects is expected backend maturation.

---

## 18. Review checkpoints

### Checkpoint A — before broad code generation

Owner approves:

- C5 supported subset;
- representation mapping;
- Drop strategy;
- layout-query strategy;
- minimal runtime boundary;
- provider ABI scope;
- first target/toolchain.

### Checkpoint B — after scalar executable

Review:

- generated crate structure;
- MIR-to-Rust control-flow mapping;
- trap path;
- no-unwind behaviour;
- error classification.

### Checkpoint C — after bounded Drop proof

Review:

- all unsafe helpers;
- move/live-state transitions;
- exact Drop evidence;
- aggregate representation;
- layout-query evidence.

### Checkpoint D — before gate exit

Review:

- reference workspace;
- three-engine matrix;
- deferred feature list;
- C5 exit report;
- exact C6 entry point.

These checkpoints are reviews, not invitations to start open-ended audits.

---

## 19. Owner decision table

The owner should record one decision for each item before C5.1 implementation is treated as approved.

| Decision | Recommended choice |
|---|---|
| Initial backend | Generated Rust consuming verified MIR |
| Initial profile | Debug only |
| Initial target | Current owner/CI host target; record exact triple |
| Generic strategy | Emit concrete monomorphised instances only |
| Internal aggregate layout | Generated Rust target layout, version-recorded, not stable ABI |
| LayoutQuery | Query canonical generated Rust type |
| Non-Copy storage | `MaybeUninit<ManuallyDrop<T>>` or equivalent explicit-liveness strategy |
| STARK Drop | Explicit MIR-directed glue; no automatic Rust Drop schedule |
| Unsafe policy | Isolated fixed helpers only |
| Build driver | Cargo invoked internally by `stark build` |
| Generated dependencies | Local/pinned only; no network fetch |
| Provider ABI | Specify v0.1 in C5.1; execution not required for MVP |
| Native C5 baseline | Section 3 required subset |
| Full semantic parity | Deferred to C6 |

Allowed owner outcomes:

```text
APPROVE
APPROVE-WITH-CHANGES
REVISE
BLOCKED
```

---

## 20. Definition of ready

WP-C5.1 implementation is ready to begin only when:

- this document is checked into `STARKLANG/docs/compiler/work-packages/`;
- its status is changed from `PROPOSED` to `APPROVED`;
- the owner decision is recorded in `COMPILER-STATE.md`;
- the frozen C5 reference workspace is named;
- its HIR/MIR snapshot is green;
- the first host target and Rust toolchain are recorded;
- no unresolved CE4 decision remains.

---

## 21. Definition of done for the C5 plan

This planning increment is complete when a later implementation session can answer, without inventing policy:

- Which MIR programs are accepted by the C5 backend?
- How is every accepted `MirTy` represented?
- Who owns monomorphisation?
- How are moves distinguished from copies?
- How are non-`Copy` locals stored?
- How are explicit Drops prevented from running twice?
- How are partial moves represented?
- How does a trap abort without cleanup?
- How does `LayoutQuery` obtain real target values?
- What crosses the provider ABI?
- What files does the backend generate?
- How does `stark build` find and invoke the Rust toolchain?
- Where is the executable written?
- Which cases compare across HIR, MIR, and native execution?
- Which failures stop the gate?
- Which missing features are deliberately deferred to C6?

If any of those still requires an architectural guess, C5.1 is not ready to implement broadly.
