# Claude Code Marathon Execution Directive — WP-C7.9

**Status:** OWNER-AUTHORISED FOR EXECUTION  
**Mode:** One uninterrupted implementation phase, followed by one consolidated qualification phase  
**Repository:** `navraj007in/stark`  
**Work package:** `WP-C7.9 — Three-Engine Adversarial Conformance Correction`  
**Primary objective:** Accepted STARK source must receive the same **specified** outcome across HIR, MIR, native debug, and native release for the admitted pure-language surface.

---

## 0. Operating instruction

Execute WP-C7.9 as one marathon coding phase.

Do **not** run packet-level tests, the full Rust test suite, the differential corpus, CI scripts, or cross-platform qualification while implementing Packets A–I. Complete the entire code and documentation change set first. Then enter one qualification phase, run the complete test sequence, fix failures in batches, and repeat the complete qualification sequence until green.

Do not ask for decisions already resolved in §2. Do not stop after finding the first defect. Continue through every independent packet unless a hard blocker makes further code changes unsafe or impossible.

Use a dedicated worktree and branch. Do not work in a shared checkout.

---

## 1. Non-negotiable controls

### 1.1 Dedicated worktree

Before editing:

```bash
git status --short
git log -1 --oneline
git worktree add ../stark-c79 -b c79-three-engine-correction
cd ../stark-c79
```

If the target branch already exists, use a new unambiguous branch name. The worktree must start clean.

Do not use broad staging commands in a shared tree. In this dedicated worktree, still review every staged path before committing.

### 1.2 No unrelated cleanup

Do not:

- rename unrelated APIs;
- reorganise modules for style;
- upgrade dependencies;
- reformat untouched files;
- change provider ABI shapes unless this directive explicitly requires it;
- alter accepted language semantics outside the enumerated findings;
- absorb `WP-C7-Usage-Shape-Qualification`;
- rewrite historical decisions.

A nearby defect may be fixed only when it directly blocks a required packet and is recorded separately in the final report.

### 1.3 Implementation-first discipline

During the implementation phase:

- source inspection is allowed;
- repository search is allowed;
- editing is allowed;
- reviewing diffs is allowed;
- no `cargo test`;
- no corpus replay;
- no CI workflow execution;
- no package qualification;
- no native cross-profile run;
- no opportunistic “quick test” after each packet.

A compile-only command may be used once, after all packet code is written, if needed to transition into qualification. Treat that as the beginning of the qualification phase.

### 1.4 Preserve evidence boundaries

Maintain these evidence classes:

```text
Pure language semantics       : HIR == MIR == native-debug == native-release
Provider binding / lifecycle  : verifier + ABI tests + native execution
Real operating-system effects : native platform qualification
```

Do not describe provider-backed packages as three-engine qualified unless an interpreter provider model exists.

---

## 2. Owner rulings for this execution

These decisions are authorised. Do not pause for approval.

### D1 — Cause-dependent integer trap category

**Approved.**

Use the existing checked-operation result mechanism to report a cause-dependent category. Prefer the existing `CheckedOutcome::Trap(Some(category))` override path. Do not add a new MIR statement or terminator variant unless the existing representation proves incapable.

Document that `Div` and `Rem` have:

- default failure category `DivideByZero` for divisor zero;
- override category `IntegerOverflow` for signed `MIN / -1` and `MIN % -1`.

Do not bump `MIR_VERSION` merely for documentation or backend logic. Bump it only if the serialised/structural MIR contract changes.

### D2 — Trap identity

`IntegerOverflow` is the required category for both:

- minimum signed value divided by `-1`;
- minimum signed value remainder by `-1`.

The Core specification already states this. Do not amend Core numeric semantics. Update compiler/MIR documentation and evidence only.

### D3 — HIR-only iterator surfaces

For this marathon, choose **uniform frontend refusal**, not new iterator architecture, for:

- by-value `Vec<T>` iteration;
- `Iterator::map`;
- `Iterator::filter`;
- `Iterator::count`;
- `Iterator::collect`.

The goal is to eliminate the split where type checking accepts code that only HIR executes. Add a stable user-facing diagnostic at type checking. Reuse an existing suitable unsupported-language diagnostic code if one exists; otherwise allocate one free E-code after checking the registry and document it.

Do not implement `VecIntoIter`, adapter MIR types, or collection lowering in this work package.

### D4 — Native call-depth exhaustion

Add controlled call-depth/resource-limit failure to the HIR and MIR interpreters.

Do **not** add per-function native call-depth instrumentation in this work package. Record native stack/call-depth exhaustion as a bounded host/process deviation under `LIMIT-RESOURCE-001`, explaining why Rust/native stack overflow is not reliably recoverable.

### D5 — Provider model

Do not build a deterministic interpreter-side provider in this marathon.

Complete the mandatory evidence-class corrections only. Provider-backed packages remain native-qualified.

### D6 — Trait implementation diagnostic

Use `E0500` for Core-trait and user-trait implementation contract violations unless an existing more-specific code already covers the exact condition. Keep one code family with precise messages rather than minting multiple new codes.

### D7 — Permitted final claim

The final claim must remain bounded:

> At the qualified commit, every maintained admitted pure-Core conformance case agrees with an independently pinned specification expectation across HIR, MIR, native debug, and native release. Provider-backed capabilities remain verifier/ABI/native-qualified, and intentionally unsupported iterator forms are rejected uniformly by the frontend.

Do not claim that every type-correct STARK program is universally three-engine conformant.

---

## 3. Implementation order

Implement in this order without running tests between packets:

```text
A → B → C → D → E → F → G → H → I → J preparation
```

Packets F, H, and I are logically independent, but keep this order to reduce overlapping edits and make the final diff review easier.

Packet J’s documentation and claim text may be drafted during the implementation phase, but corpus version changes, evidence hashes, final status, and commit references must be completed only after qualification.

---

# Packet A — Integer trap correctness

## A.1 Defects to close

For every signed width:

```text
MIN / -1  -> IntegerOverflow
MIN % -1  -> IntegerOverflow
x / 0     -> DivideByZero
x % 0     -> DivideByZero
```

Current widened `i128` evaluation incorrectly allows `MIN % -1` to complete with `0`, and routes `MIN / -1` range failure through the static divide-by-zero category.

## A.2 MIR interpreter implementation

In `starkc/src/mir/interp.rs`, update checked `Div` and `Rem` evaluation.

Required order:

1. Read each operand exactly once.
2. Determine the destination integer width and signedness.
3. If divisor is zero, return the default trap.
4. If destination is signed, dividend equals that destination type’s minimum value, and divisor is `-1`, return:

```rust
CheckedOutcome::Trap(Some(TrapCategory::IntegerOverflow))
```

5. Otherwise perform the widened operation.
6. Range-filter the result as today.

Do not identify signed minimum from the `i128` carrier alone. Use the destination `MirTy`’s declared range.

Add a helper if needed:

```rust
fn is_signed_int(ty: &MirTy) -> bool
```

Keep unsigned behaviour unchanged.

## A.3 Generated-Rust backend implementation

In `starkc/src/backend/generated_rust/emit_bodies.rs`:

- bind both operands into generated temporaries once;
- branch on divisor zero;
- branch on signed destination minimum with divisor `-1`;
- call the correct abort function for each cause;
- perform widened checked operation only after those guards;
- preserve source location/provenance;
- preserve current result range checking.

Do not emit the left or right operand expression multiple times.

The generated shape should be logically equivalent to:

```rust
{
    let __a = <left> as i128;
    let __b = <right> as i128;

    if __b == 0 {
        abort_divide_by_zero(...);
    }

    if SIGNED && __a == TYPE_MIN && __b == -1 {
        abort_integer_overflow(...);
    }

    match __a.checked_div(__b) {
        Some(__v) if __v >= TYPE_MIN && __v <= TYPE_MAX => __v as Dest,
        _ => abort_integer_overflow(...),
    }
}
```

Use `checked_rem` for remainder.

## A.4 Lowering and MIR contract

Inspect `mir/lower.rs` where `Div` and `Rem` currently attach a static trap category.

Keep `DivideByZero` as the terminator’s default category if the interpreter/backend override mechanism is sufficient.

Update MIR documentation to state that checked operation evaluation may override the terminator category when the operation fails for a different normative cause. Use the same model already used for invalid shifts.

Do not alter the Core spec.

## A.5 Tests to add

Create maintained tests, not scratch-only probes.

Required exact cases for `Int8`, `Int16`, `Int32`, and `Int64`:

- `MIN / -1`;
- `MIN % -1`;
- `MIN / 1`;
- `MIN % 1`;
- `MIN / -2`;
- `MIN % -2`;
- `/ 0`;
- `% 0`;
- `/=` and `%=` at least once;
- correct source-location blame.

Because minimum literals are not directly writable, construct them through an in-range expression such as:

```stark
let min: Int32 = -2147483647 - 1;
```

### Exhaustive Int8 property evidence

Add one host-side independent mathematical oracle for all `Int8 × Int8` pairs.

Use a batched STARK execution test for all **non-trapping** pairs rather than compiling one native binary per pair. Emit or accumulate deterministic results and compare them with a Rust width-aware oracle.

Test the trapping pairs independently:

- all `a / 0`;
- all `a % 0`;
- `-128 / -1`;
- `-128 % -1`.

The expected outcome must be pinned independently; engine agreement alone is insufficient.

Add permanent corpus sentinels for the four widths’ `MIN / -1` and `MIN % -1` cases.

---

# Packet B — Core-trait and user-trait implementation conformance

## B.1 Architectural rule

A trait implementation must conform before any body is executable.

Malformed implementations must be rejected by the type checker with `E0500`. They must never reach:

- HIR execution;
- MIR lowering;
- MIR verification;
- native compilation.

## B.2 Canonical Core-trait contracts

Do not scatter signature checks through operator code.

Create one canonical descriptor for every `CoreTrait` variant that can be implemented. Derive the complete contract from the current `CoreTrait` enum and normative standard-library/type-system definitions.

The descriptor must cover, where applicable:

- required method names;
- receiver form: none, `self`, `&self`, `&mut self`;
- parameter count and types;
- return type;
- method generic parameters and bounds;
- associated types;
- required versus defaulted items;
- whether extra associated items are prohibited.

Normalise `Self`, trait generic parameters, and associated-type projections before comparison.

## B.3 Reuse one conformance engine

Generalise the existing user-trait signature comparison so Core traits and user traits use the same structural comparison machinery.

The comparison must include:

- receiver presence and mutability;
- parameter arity;
- parameter types;
- return type;
- generic arity;
- alpha-equivalent generic parameter names;
- generic bounds independent of source ordering;
- associated-type bindings;
- required item presence;
- extra item rejection;
- duplicate item rejection;
- default method override compatibility.

Make responsibility explicit. Do not leave “missing method” to an unrelated pass without tests proving that pass owns it.

## B.4 Diagnostics

Use `E0500`.

Messages must identify:

- trait name;
- implementation item;
- expected signature or contract component;
- actual signature or missing/extra state;
- offending item span.

Do not report a runtime trap such as “Ord::cmp must return Ordering”.

HIR and MIR may retain defensive assertions, but a violation there must be classified as an internal compiler/oracle invariant failure.

## B.5 Required negative matrix

Cover every relevant Core trait and representative user traits.

At minimum:

- `Eq::eq` wrong return type;
- `Ord::cmp` wrong return type;
- `Display::fmt` wrong return type;
- `Clone::clone` wrong return type;
- `Default::default` wrong return type;
- malformed `From`;
- malformed `Iterator::next`;
- malformed `Drop`;
- wrong receiver mutability;
- missing receiver;
- extra receiver;
- wrong parameter type;
- wrong parameter count;
- missing required method;
- extra trait item;
- duplicate method;
- wrong associated type;
- wrong generic arity;
- wrong bound;
- compatible alpha-renamed generic signature accepted;
- compatible default override accepted.

Add tests proving rejection occurs during type checking and that lower/execute is never attempted.

---

# Packet C — HIR place-aware pattern execution

## C.1 Normative target

Implement `PAT-BIND-001` in the HIR interpreter.

When a match scrutinee is a **place read through a reference**:

- a non-`Copy` component binds by shared reference;
- a `Copy` component binds by value;
- bindings remain shared even when the source path came through `&mut`;
- owned scrutinees continue to bind by value/move;
- each nested `match` decides its own mode;
- a reference-typed value is not implicitly destructured without explicit dereference.

## C.2 Required architecture

Do not patch individual pattern variants with clones.

Introduce an explicit pattern source model, conceptually:

```rust
enum PatternSource {
    Owned(Value),
    Place {
        place: Place,
        through_reference: bool,
    },
}
```

Exact naming may differ.

Add an `eval_match_source` path that preserves place identity for scrutinees that are places:

- local/path;
- field projection;
- tuple projection;
- enum payload projection;
- array/index projection where supported;
- explicit dereference;
- field access with a reference base.

Rvalues remain owned values.

Change `match_pattern` and recursive pattern matching to accept a source that can be projected without first cloning the whole value.

## C.3 Binding behaviour

For a binding:

```text
Owned + non-Copy                     -> move/value binding according to current owned semantics
Owned + Copy                         -> value copy
Place through reference + non-Copy  -> Value::Ref(projected place)
Place through reference + Copy      -> value read
```

For nested struct, tuple, enum, and named fields, project the original place and preserve frame/lifetime identity.

Do not create references to temporary cloned values.

Do not grant mutable bindings through `&mut`; PAT-BIND-001’s current floor is shared binding.

## C.4 Destruction and ownership

Ensure a by-reference match:

- does not consume the referent;
- does not schedule destruction of referenced payloads;
- permits repeated inspection;
- preserves the original owner’s eventual drop.

Ensure an owned match retains current move/drop behaviour.

## C.5 Required evidence

Convert the existing pinned HIR divergence test into a positive four-configuration case.

Do not delete the case.

All existing CE1/PAT-BIND-001 cases must agree, including:

- dereferenced `&self`;
- dereferenced reference parameter;
- field read through reference base;
- struct pattern;
- tuple pattern;
- nested explicit dereference;
- `Copy` payload by value;
- owned scrutinee by value;
- generic payload;
- repeated inspection;
- shared behaviour through `&mut`;
- negative direct matching of a reference-typed binding;
- existing Box-recursive limitation remains a frontend limitation unless separately supported.

Requalify `stark-json` execution, not just build success, after this fix.

---

# Packet D — stderr as a first-class compared channel

## D.1 Required semantics

Implement exact three/four-engine semantics for:

- `eprint`;
- `eprintln`;
- `Display` dispatch;
- newline byte `0x0A`;
- ordering within stderr;
- stdout and stderr in the same program;
- bytes emitted before a trap;
- flush before normal return or trap.

## D.2 HIR interpreter

Add an interpreter-owned stderr buffer.

Change `eprint`/`eprintln` so they append to that buffer rather than calling host `eprint!`/`eprintln!`.

`Execution.stderr` must contain:

- normal `eprint`/`eprintln` bytes;
- `Err(message)` entry-completion bytes in the specified order.

Do not allow program stderr to leak into the Rust test runner’s stderr.

Preserve exact `Display::fmt` call count, argument consumption, and drop order.

## D.3 MIR representation and interpreter

Mirror the existing stdout print lowering.

Use either:

- explicit stderr runtime functions; or
- a shared output operation parameterised by channel.

The design must preserve ordinary `Display` dispatch. Do not turn `eprintln` into a syntax-only or string-only special case.

Add captured stderr to both successful and failed MIR execution outcomes so pre-trap bytes are retained.

## D.4 Native runtime and generated Rust

Add native runtime support for stderr submission and flush.

Program stderr and runtime trap diagnostics share the host OS stream, so differential tests need a structured separation protocol.

Use this design:

1. The differential runner generates a random per-run trap token.
2. It sets a hidden environment variable, for example:

```text
STARK_DIFFERENTIAL_TRAP_TOKEN=<nonce>
```

3. On a language trap, the runtime:
   - flushes program stdout and stderr;
   - emits one final machine-readable trap record containing the exact nonce;
   - aborts/exits through the existing trap path.
4. The comparator splits captured native stderr at the exact nonce-bearing record:
   - bytes before the record are program stderr;
   - the record supplies trap category, provenance, and message;
   - no token means ordinary production CLI formatting remains unchanged.

Do not use a fixed delimiter that a STARK program can accidentally reproduce.

Provider-facing environment APIs must not expose or depend on this token. It is an internal runtime/test protocol.

## D.5 Comparator observation

Extend observations so both completion and trap outcomes carry program stderr.

Required relations:

```text
stdout bytes
stderr bytes
status or trap category
trap provenance
trap message where normative
drop/destruction log
```

A trap diagnostic is not program stderr.

## D.6 Required cases

- `eprint` no newline;
- `eprintln` one newline;
- repeated stderr writes preserve order;
- alternating stdout/stderr preserves each stream’s own order;
- stderr before trap is retained;
- `Display::fmt` user implementation;
- formatting trap produces no partially submitted formatted bytes unless the spec explicitly permits them;
- source argument evaluated once;
- formatting `String` and source argument destruction timing;
- `Err(message)` completion combined with earlier `eprint`;
- debug/release equality.

---

# Packet E — Uniform refusal of HIR-only iterator surfaces

## E.1 Audit first

Locate every test/helper that currently asserts:

- type checking succeeds;
- HIR executes;
- MIR lowering refuses.

Build a committed table of all such surfaces.

The minimum known set is:

- by-value `Vec<T>` iteration;
- `map`;
- `filter`;
- `count`;
- `collect`.

Add any additional surface found by the audit.

## E.2 Frontend refusal

Reject each surface during type checking with one stable, user-facing diagnostic.

Messages must say the form is not supported by the current Core compiler surface. Do not mention internal phases such as C4.5, MIR, or generated Rust.

Examples of acceptable message shape:

```text
by-value iteration over Vec<T> is not supported; use v.iter()
iterator adapter 'map' is not supported by this compiler profile
```

Do not reject supported reference iteration.

## E.3 Remove split-state tests

Convert HIR-only tests into specific frontend-rejection tests.

Each negative test must prove:

- parse succeeds;
- resolve succeeds;
- type checking rejects;
- the expected diagnostic code is present;
- the expected construct is blamed.

Add an audit guard so a future typechecker change cannot silently re-admit a form without lowering support.

---

# Packet F — Controlled interpreter resource exhaustion

## F.1 Classification

Call-depth exhaustion is a host/process resource failure under `LIMIT-RESOURCE-001`, not a STARK language trap.

Do not add a new `TrapCategory`.

## F.2 HIR interpreter

Add an explicit call-depth counter and implementation capacity.

The capacity must be:

- named in code;
- documented as implementation-defined;
- high enough for ordinary programs;
- low enough to prevent host stack abort on all Tier-1 test stacks.

Increment before entering a STARK call and decrement on every exit, including errors. Use an RAII guard if possible.

On limit exhaustion, return a structured host/resource failure rather than `RuntimeError::new` language trap text.

## F.3 MIR interpreter

Apply the same classification and comparable implementation guard.

The two interpreter capacities may share one constant if their host-stack behaviour is equivalent; otherwise document why they differ.

Do not let the comparator treat resource-limit failure as a semantic trap.

## F.4 CLI behaviour

`starkc run` must:

- remain alive;
- print a classified resource-limit diagnostic;
- return a stable nonzero process status;
- not report a STARK trap category;
- not emit a Rust panic or stack-overflow abort.

## F.5 Native execution

Do not add generated per-call guards.

Document:

- native stack capacity is target/runtime-defined;
- host stack exhaustion may terminate the process;
- it is outside semantic equality under `LIMIT-RESOURCE-001`;
- no claim is made that native and interpreter call-depth capacities match.

## F.6 Tests

Run recursion-limit cases only through subprocess tests.

Required:

- recursion below the interpreter limit completes;
- recursion above the limit returns classified failure;
- no test process abort;
- mutual recursion also triggers the guard;
- error paths restore the counter;
- repeated independent runs do not inherit depth state.

---

# Packet G — Comparator and qualification hardening

## G.1 Native debug and release

Refactor the comparator so admitted cases can execute:

```text
HIR
MIR
native-debug
native-release
```

Do not represent native release as an unrelated special helper.

The default maintained differential path must compare both profiles when Rust/native execution is available.

Record missing profiles explicitly.

## G.2 No silent macro skip

Change `three_engine_test!` and related macros so absence of `rustc` does not return before comparing HIR and MIR.

The result must state:

- which engines ran;
- which engine/profile was unavailable;
- why.

Tier-1 CI must treat missing native tooling as failure unless the job is explicitly interpreter-only.

## G.3 Structural trap categories

Inventory every HIR language-trap construction site.

Convert every normative trap to carry an explicit `TrapCategory`.

Separate:

- language trap;
- entrypoint/compiler-selection error;
- interpreter internal invariant failure;
- host/resource failure.

Remove substring-based semantic classification only after every language trap site is explicit.

Diagnostic prose must no longer determine category.

Add a guard test that fails if the normaliser still contains semantic phrase matching such as:

```text
integer overflow
division by zero
invalid shift
out of bounds
```

## G.4 Independently pinned expectations

Introduce or generalise an expectation type, conceptually:

```rust
enum ExpectedOutcome {
    Complete {
        stdout: ...,
        stderr: ...,
        status: u8,
        drop_log: ...,
    },
    Trap {
        category: TrapCategory,
        source: ...,
        stdout_before: ...,
        stderr_before: ...,
        message: ...,
    },
    FrontendReject {
        code: ...,
    },
    HostFailure {
        class: ...,
    },
}
```

Every adversarial test must assert:

1. each engine matches the independent expectation;
2. engines agree with one another.

Do not use HIR output as the expected answer.

## G.5 Shared evaluator compensation

For `canonical_float` and any other shared semantic helper:

- keep exact-value tests independent of engine execution;
- cover NaN, infinities, signed zero, subnormals, shortest round-trip formatting, and Float32 rounding;
- add mutation-style tests using deliberately incorrect helper variants or altered boundary logic, proving the tests fail on wrong algorithms.

No need to introduce a third-party mutation framework.

## G.6 Commit adversarial seeds

Turn the scratch probe battery into maintained test modules grouped by subject, for example:

```text
tests/adversarial_integer_semantics.rs
tests/adversarial_trait_impls.rs
tests/adversarial_patterns.rs
tests/adversarial_stderr.rs
tests/resource_exhaustion.rs
```

Keep deterministic seeds and print the seed on failure.

## G.7 Boundary/property expansion

At minimum add property coverage for:

- signed/unsigned integer boundaries;
- casts at exact min/max and one beyond;
- shifts at `-1`, `0`, `width-1`, `width`, and overflow-producing values;
- exponentiation intermediate overflow;
- Float32 operation-rounding boundaries;
- index and length conversion near `usize`/`UInt64` boundaries where host-independent tests are possible;
- exact trap category and source location.

---

# Packet H — Correct provider evidence claims

## H.1 Mandatory documentation changes

Audit and correct claims in:

- C7.8 capability records;
- package `README.md`;
- package `EVIDENCE.md`;
- `stark-io`;
- `stark-time`;
- TCP/REST workload records;
- any compiler-state or gate document saying “three-engine” for provider-backed execution.

Use the evidence classes from §1.4.

Examples:

```text
stark-json pure semantics:
    eligible for four-engine qualification

stark-io provider calls:
    verifier + provider ABI + native platform qualification

pure-STARK read_to_end logic over a deterministic adapter:
    not claimed until such an adapter exists
```

## H.2 No fake provider implementation

Do not add interpreter provider execution in this work package.

Record it as deferred, with one owner and one future work item, rather than leaving it ambiguous.

---

# Packet I — Close DEV-118 shared trait-bound omission

## I.1 Required rule

Enforce the normative bounds for:

```text
HashMap<K, V> : K must satisfy Hash + Eq
HashSet<T>    : T must satisfy Hash + Eq
```

The fact that the current implementation scans by equality is irrelevant to type conformance.

## I.2 Enforcement location

Use one general mechanism for implementation-provided generic type bounds.

Do not hard-code checks only in `insert`.

Enforcement must cover type instantiation and every path that constructs or uses the type sufficiently to prevent invalid programs from reaching execution.

Coordinate with existing generic-bound obligation machinery.

## I.3 Required tests

Reject:

- key type with `Eq` but no `Hash`;
- key type with `Hash` but no `Eq`;
- floating-point key types;
- generic functions whose bounds are insufficient;
- nested invalid key types.

Accept:

- primitives that normatively satisfy both;
- user nominal with valid `Eq` and `Hash`;
- generic function with both bounds;
- values and operations across all engines after frontend acceptance.

Close or update DEV-118 in the authoritative ledger.

---

# Packet J — Integration preparation

During implementation, prepare but do not finalise:

- packet records;
- amendment text;
- deviation updates;
- claim wording;
- corpus additions;
- coverage matrix changes;
- `COMPILER-STATE.md` append sections.

Do not record PASS, close the WP, bump evidence hashes, or claim qualification until the final test phase is green.

---

## 4. End-of-implementation review before tests

After Packets A–I are coded:

1. Review `git status --short`.
2. Review the full diff by packet.
3. Confirm no unrelated paths changed.
4. Search for:
   - old substring trap normalisation;
   - direct HIR host `eprint!`/`eprintln!`;
   - tests that still intentionally accept HIR-only surfaces;
   - claims that provider packages are three-engine qualified;
   - DEV-118 still marked open without explanation;
   - divergence test for PAT-BIND-001 still expecting failure.
5. Confirm every new diagnostic code is unique and documented.
6. Confirm every new enum variant has exhaustive consumer handling.
7. Run formatting once.

This is the transition into the qualification phase.

---

# 5. Consolidated qualification phase

Once implementation is complete, run tests in the following order. From this point onward, fixing failures and rerunning tests is allowed.

Use the exact canonical commands from the current repository CI/work-package scripts. Do not rely on stale commands in this document when the repository has a newer authoritative command.

## Q1 — Formatting and compile surface

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
```

Fix compile errors across the complete workspace in one batch.

## Q2 — Static analysis

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Do not suppress new warnings unless the code is intentionally unreachable and the reason is documented.

## Q3 — New and directly affected tests

Run all new C7.9/adversarial modules together, not one at a time.

Include:

- integer semantics;
- trait impl conformance;
- CE1 pattern binding;
- stderr;
- HIR-only frontend refusal;
- resource exhaustion;
- comparator tests;
- Hash/`Eq` bounds;
- stark-json execution qualification.

## Q4 — Full workspace test run

```bash
cargo test --workspace --all-targets --all-features --no-fail-fast
```

Use `--no-fail-fast` so one run exposes the complete failure set.

Fix failures in grouped batches by root cause.

After any fix, rerun:

1. affected test group;
2. full workspace test command.

## Q5 — Project-specific qualification commands

Read the current CI workflows and run every canonical project-specific command used for:

- conformance fixture validation;
- C6/C7 corpus replay;
- generator/replay suite;
- metamorphic suite;
- MIR verifier guards;
- native debug profile;
- native release profile;
- provider ABI validation;
- C6.4/C7 qualification;
- package graph tests;
- stark-json;
- stark-url;
- P1 REST workload;
- lifecycle/resource tests.

Copy the commands verbatim from the current workflow or authoritative work-package document.

## Q6 — Four-configuration differential corpus

Run the admitted pure-language corpus through:

```text
HIR
MIR
native-debug
native-release
```

For every case assert both:

```text
engine observation == pinned specification expectation
all engine observations are equal
```

No case may silently skip.

Record:

- total cases;
- completing cases;
- trapping cases;
- frontend rejection cases;
- host/resource cases;
- unavailable-engine count;
- observation hashes by engine/profile.

## Q7 — Release-specific checks

Pay particular attention to:

- `MIN / -1`;
- `MIN % -1`;
- overflow categories;
- Float32 rounding;
- trap provenance;
- stderr before trap;
- drop timing;
- loop storage reuse;
- partial moves;
- `?` inside loops;
- provider resource cleanup.

Debug-only success is insufficient.

## Q8 — Subprocess robustness

Run:

- HIR deep recursion;
- MIR deep recursion;
- CLI deep recursion;
- native documented-deviation probe;
- malformed program/trait cases;
- stderr/trap protocol cases.

No test runner process may abort.

## Q9 — Tier-1 CI

After local qualification is green, commit and push the branch and run the full Tier-1 CI matrix:

- Linux x64;
- macOS arm64;
- Windows x64;
- native debug;
- native release;
- provider/platform lanes.

Do not close the WP on local-host evidence alone.

---

## 6. Failure-handling policy during qualification

When the first consolidated run fails:

1. collect the entire failure set;
2. classify failures by root cause;
3. fix all failures from one root cause together;
4. rerun the affected group;
5. rerun the complete workspace and project qualification;
6. repeat until green.

Do not weaken expectations to match current output.

Do not regenerate snapshots or corpus expectations until the specification-derived expected value is independently confirmed.

A corpus re-pin requires an explicit scope statement showing which cases changed and why.

---

## 7. Commit and CD strategy

At execution time, re-read the highest CD number across all refs.

Allocate one CD per independently meaningful packet or closely coupled packet group. Suggested logical sequence:

```text
CD-N     Packet A
CD-N+1   Packet B
CD-N+2   Packet C
CD-N+3   Packet D
CD-N+4   Packet E
CD-N+5   Packet F
CD-N+6   Packet G
CD-N+7   Packet H
CD-N+8   Packet I
CD-N+9   Packet J closure
```

Do not assume `CD-270` is still free.

Because this is a marathon implementation, checkpoint commits may be made locally before qualification, but final history must not present unqualified work as closed. Before pushing:

- amend/squash checkpoint commits into coherent packet commits;
- include evidence only after tests pass;
- ensure commit titles match actual contents;
- verify staged paths explicitly.

---

## 8. Packet J finalisation after green qualification

After all local and Tier-1 qualification is green:

1. bump the corpus version once;
2. regenerate corpus evidence once;
3. update observation hashes;
4. reconcile F1–F5 and R1.1–R1.10;
5. update `COMPILER-STATE.md`;
6. append to C7 records without rewriting historical rulings;
7. update C10 blocker status;
8. record the native call-depth deviation;
9. record provider evidence-class boundaries;
10. close DEV-118;
11. close WP-C7.9;
12. state the authorised bounded conformance claim from D7.

Do not claim provider-backed three-engine qualification.

Do not claim universal conformance for every type-correct program.

---

## 9. Required final report

Return one report with these sections.

### 9.1 Repository state

- branch;
- final commit;
- base commit;
- CD range;
- clean worktree confirmation.

### 9.2 Changes by packet

For A–I:

- root cause;
- implementation;
- files changed;
- tests added;
- resulting status.

### 9.3 Owner rulings applied

State how D1–D7 were executed.

### 9.4 Test evidence

Include exact commands and totals for:

- format;
- check;
- clippy;
- workspace tests;
- adversarial tests;
- corpus;
- debug/release differential;
- provider ABI;
- packages;
- Tier-1 CI.

### 9.5 Observation matrix

Report counts and hashes for:

```text
HIR
MIR
native-debug
native-release
```

### 9.6 Deviations and deferrals

Only expected retained items:

- native call-depth exhaustion boundary;
- provider-backed capabilities remain native-qualified;
- refused iterator surfaces;
- `WP-C7-Usage-Shape-Qualification` remains adjacent.

Any additional item must be explicitly justified.

### 9.7 Final claim

Use exactly the bounded claim authorised in D7, adjusted only for factual evidence totals.

---

## 10. Hard stop conditions

Do not stop for ordinary compiler errors or test failures.

Stop and report only when:

- the dedicated worktree is not clean and proceeding risks contaminating another change;
- a required fix contradicts a normative Core rule;
- a change requires a new public language feature outside WP-C7.9;
- a provider ABI breaking change becomes unavoidable;
- repository corruption or data loss is possible;
- an external toolchain/platform failure prevents qualification after code is complete.

When one packet is blocked, continue every independent packet first. Report the blocker only after completing all safe work.

---

## 11. Completion definition

The marathon is complete only when:

- Packets A–I are implemented or dispositioned exactly as directed;
- the entire consolidated qualification sequence is green;
- Tier-1 CI is green;
- no maintained admitted pure-language case differs from its pinned expectation across HIR, MIR, native debug, and native release;
- malformed trait implementations are rejected in the frontend;
- HIR implements PAT-BIND-001;
- normal and pre-trap stderr are compared;
- HIR-only iterator surfaces are uniformly refused;
- interpreter call-depth exhaustion is classified without process abort;
- trap identity is structural, not prose-derived;
- DEV-118 is closed;
- provider evidence claims are accurate;
- Packet J records the bounded final claim.
