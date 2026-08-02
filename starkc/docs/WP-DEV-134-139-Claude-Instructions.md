# WP-DEV-134-139 v2 — Soundness and Control-Flow Repair Programme

**Status:** APPROVED WITH OWNER AMENDMENTS — READY FOR IMPLEMENTATION  
**Target branch:** `develop`  
**Repository:** `navraj007in/stark`  
**Owner:** Claude  
**Scope:** DEV-134 through DEV-139 discovered by the external 18-package sample suite; amended sequencing, staged DEV-135 closure, and independent external-suite CI  
**Priority:** P0/P1 compiler correctness before public binary release  
**Integration rule:** Do not merge to `main` until aggregate CI is green and all closure evidence is recorded.

---

## 1. Objective

Repair the six defects registered in CD-334 without weakening existing language rules, bypassing invariants, or rewriting sample programs around compiler limitations.

Required outcome:

1. incompatible `?` propagation is rejected unless the language specification explicitly authorises a conversion;
2. moves from struct fields are tracked precisely enough to reject a second move before execution;
3. moves on terminating control-flow paths do not poison states that remain reachable;
4. borrows created solely to evaluate a `while` condition end before entering the loop body;
5. iterator-yielded shared string references obey shared-reference Copy semantics across all engines;
6. impl-level generic bounds are visible to operator and trait resolution inside impl methods;
7. all six reproducers become permanent executable regression evidence;
8. no repair weakens ownership, borrowing, MIR verification, or existing negative controls.

This work package is a repair programme, not a feature expansion. Where the specification is ambiguous, stop at the conservative, sound behaviour and record the unresolved language-design question separately.

---

## 2. Governing rules

Apply these rules throughout the work:

- Register or update the deviation entry before implementing the repair.
- Keep one coherent defect per commit unless two findings are proven to share one mechanism and one repair.
- Do not add a special case when a shared authority or dataflow mechanism is the real defect.
- Do not downgrade an invariant, verifier rule, or ownership check to make a reproducer pass.
- Do not rewrite sample packages to avoid valid source shapes.
- Do not infer STARK semantics from Rust.
- Use the approved specification, `COMPILER-STATE.md`, the deviation ledger, current executable tests, and current implementation in that order.
- Any newly discovered defect must receive its own DEV entry before repair.
- Every commit message must state:
  - the exact defect mechanism;
  - why the previous implementation was wrong;
  - why the repair is sound;
  - which negative controls prevent over-broadening;
  - exact evidence run locally;
  - any checks intentionally left to CI.

---

## 3. Required implementation order

Use this order unless a direct dependency forces a different sequence:

1. **DEV-134 — incompatible `?` propagation**
2. **DEV-137 — `while`-condition borrow leaks into body**
3. **DEV-136 — terminating-path move treated as unconditional**
4. **DEV-135a — soundness gate for repeated field moves**
5. **DEV-139 — impl-level generic bounds missing from operator resolution**
6. **DEV-138 — iterator-yielded `&str` consumed after first use**
7. **DEV-135b — full field-sensitive partial-move precision, if justified by inventory**

Reasoning:

- DEV-134 is an immediate type-soundness blocker.
- DEV-137 and DEV-136 correct control-flow regions and reachability before field-sensitive move analysis is built on top of them.
- DEV-135's branch-sensitive obligations depend on DEV-136's join semantics.
- DEV-135 is split into a conservative soundness gate and a separate precision programme so release safety does not wait on a Rust-grade partial-move feature.
- DEV-139 is a bounded generic-environment propagation repair.
- DEV-138 may be an instance of the existing DEV-121 representation class and must be classified before being treated as independent.

Do not implement all defects in one large commit.

---

# Part A — DEV-134

## 4. DEV-134 — `?` propagates an incompatible error type

### 4.1 Problem statement

A function returning:

```stark
Result<T, High>
```

can use `?` on:

```stark
Result<T, Low>
```

even when:

- `Low` and `High` are different types;
- no `From<Low> for High` implementation exists;
- no explicit conversion occurs.

The `Low` runtime value is propagated while the surrounding expression is typed as `High`.

This violates type preservation and is a soundness defect.

### 4.2 Owner ruling for this work package

Implement the conservative rule first:

> `?` may propagate an error only when the source error type is exactly compatible with the enclosing function's error type under already-approved type equality/coercion rules.

Do **not** introduce implicit `From` conversion as part of this repair unless an approved specification already requires it.

If the current specification is silent or ambiguous, exact-type rejection is the required repair. File a separate language-design work package for optional conversion semantics.

### 4.3 Required semantic rule

For:

```stark
fn f(...) -> Result<U, E_out> {
    let x = expr?;
    ...
}
```

where:

```stark
expr: Result<T, E_in>
```

the checker must require:

```text
E_in == E_out
```

under the compiler's canonical type-equivalence relation.

If not, produce a user-facing diagnostic before MIR lowering.

### 4.4 Required diagnostic

Add or use a stable diagnostic code. The message should identify:

- the error type produced by the operand;
- the error type required by the enclosing function;
- that `?` does not currently perform implicit conversion;
- the source location of the `?` expression.

Example intent:

```text
error[E0xxx]: `?` cannot propagate `LowError` from a function returning `Result<_, HighError>`
help: return the same error type, or match the inner `Err`, construct the outer error explicitly, and return it
```

The help text must show or point to a source shape that compiles in current STARK. Do not recommend an implicit conversion mechanism that the language does not provide. A valid form is:

```stark
match inner_result {
    Ok(value) => value,
    Err(low) => return Err(HighError::from_low(low)),
}
```

Use the actual constructor or mapping function available in the reproducer; do not invent `from_low` if it does not exist. The diagnostic test must compile the advised spelling as a positive control.

Do not report an internal compiler error.

### 4.5 Required tests

Add a dedicated integration test file, for example:

```text
starkc/tests/dev134_try_error_type.rs
```

#### Must reject

1. Different fieldless enums.
2. Different payload-carrying enums.
3. Generic `Result<T, E1>` propagated from a function returning `Result<T, E2>`.
4. A `From<E1> for E2` impl exists but implicit conversion is not specified.
5. Nested helper function where the mismatch occurs inside a generic body.

#### Must pass

1. Identical error types.
2. Type aliases or normalized-equivalent types, if aliases exist.
3. Different success types where `?` legitimately extracts the inner success value and the surrounding expression accepts it.
4. Existing provider error propagation using the same error type.
5. Existing REST workload and package qualification.
6. **DEV-134 × DEV-136:** `?` error propagation is a terminating edge. Add a case where the error edge moves or returns a value while the success fallthrough continues to use its own still-live value; the terminating error predecessor must not poison the reachable join.

### 4.6 Required engine evidence

- `starkc check` rejects mismatched error types.
- HIR execution is unreachable for rejected programs.
- MIR lowering is unreachable for rejected programs.
- Existing correct `?` cases remain identical across HIR, MIR, and native.
- No weakening of MIR verification.

### 4.7 Closure criteria

DEV-134 may be marked closed only when:

- incompatible propagation is rejected before MIR;
- the diagnostic is stable and tested;
- all exact-type positive cases pass;
- no implicit conversion semantics were silently introduced;
- the external sample reproducer is added to the permanent suite;
- full CI is green.

---

# Part B — DEV-135

## 5. DEV-135 — individual struct fields can be moved twice

### 5.1 Problem statement

For a non-Copy field:

```stark
let first = value.field;
let second = value.field;
```

the compiler accepts both moves. The oracle later detects unavailable storage and reports an internal compiler error.

Ownership must be enforced before execution.

### 5.2 Two-stage closure model

Split this defect into:

- **DEV-135a — conservative soundness gate:** prevent any second use after a field move, using parent poisoning if necessary;
- **DEV-135b — precision follow-on:** preserve legal sibling-field use through hierarchical move paths.

Do not make public release safety depend automatically on full partial-move precision.

### 5.3 Mandatory inventory before choosing the stage-one repair

Before implementation, measure current use of partial moves across:

- all first-party packages;
- generated and frozen corpora;
- compiler integration tests;
- the external task-shaped sample suite;
- examples and consumers.

Classify at least these shapes:

```text
move field A, then use sibling B
move nested field, then use sibling
move field, then reassign it
move field, then move parent
branch-local partial move
```

Record counts and file locations in the DEV entry or an evidence document.

### 5.4 DEV-135a — conservative soundness repair

If sibling-after-partial-move is unused or rare enough to be an acceptable bounded limitation, land parent poisoning:

> Moving any non-Copy field marks the whole parent unavailable for subsequent reads, borrows, moves, or drops except through compiler-controlled drop elaboration.

This must:

- reject the second move statically;
- reject sibling use after a partial move;
- reject parent use after a partial move;
- avoid double-drop;
- produce a user-facing ownership diagnostic;
- state the temporary limitation in release notes;
- file DEV-135b as the precision follow-on.

Parent poisoning is a sound approximation, not full closure of partial-move ergonomics. Mark DEV-135a closed and DEV-135b open; do not claim field-sensitive partial moves are complete.

If the inventory shows sibling use is load-bearing, do not land poisoning silently. Proceed to DEV-135b, but keep it sequenced after DEV-134, DEV-137, and DEV-136.

### 5.5 DEV-135b — full field-sensitive move paths

The precision model must distinguish:

```text
local
local.field_a
local.field_b
local.field_a.inner
```

Required behaviour:

- moving `local.field_a` invalidates that path and descendants;
- moving `local.field_a` does not invalidate `local.field_b`;
- moving the whole parent after a partial move is rejected unless fully restored;
- moving or borrowing the same path again is rejected;
- a Copy field remains reusable;
- assignment may restore a path only where the language permits it;
- drop elaboration destroys each still-live owning field exactly once.

Reuse typed move/drop paths already present in MIR where possible. Do not add an AST spelling check for `x.field`.

### 5.6 Required tests

Add:

```text
starkc/tests/dev135_field_move_paths.rs
```

#### Stage-one must reject

1. Same non-Copy field moved twice.
2. Nested field moved twice.
3. Borrow after field move.
4. Whole parent moved after field move.
5. Parent passed by value after field move.
6. Tuple element moved twice.
7. Field moved on a reachable branch and used at the join — run only after DEV-136 establishes correct reachability semantics.

#### Inventory/precision controls

1. Move field A, then use sibling field B.
2. Move nested field A, then use unrelated nested field B.
3. Legal field restoration followed by use, if supported.
4. Drop order and exactly-once destruction for remaining live fields.
5. Existing partial-move MIR verifier tests.

For DEV-135a these controls may be expected rejections and must be documented as the bounded limitation. For DEV-135b they become required positives.

### 5.7 Closure criteria

DEV-135a closes the release-blocking soundness hole when:

- the second field move is rejected before execution;
- runtime internal errors are no longer the enforcement mechanism;
- parent poisoning is proven sound;
- current partial-move usage was inventoried;
- the limitation is explicit in release notes;
- DEV-135b is filed if precision is deferred.

DEV-135b closes only when sibling preservation, nested paths, restoration, branch joins, and exactly-once drop behaviour are proven across HIR, MIR, and native.

---

# Part C — DEV-137

## 6. DEV-137 — borrow in a `while` condition remains live across the body

### 6.1 Problem statement

This valid shape is rejected:

```stark
while i < values.len() {
    values[i] = 5;
}
```

The receiver auto-borrow used by `values.len()` is treated as live for the loop body, preventing mutation.

The condition borrow should end after condition evaluation and before entry to the body.

### 6.2 Required control-flow model

Model the loop as distinct regions:

```text
condition block
    evaluate receiver
    begin shared borrow
    call len
    end shared borrow
    compare
    branch to body or exit

body block
    mutation may occur
    jump back to condition
```

Do not solve this by hoisting `len()` outside the loop. That changes semantics when the length changes.

Do not introduce a special-case exception for `len`. The lifetime boundary must apply to temporary/autoborrows created solely for condition evaluation.

### 6.3 Required investigation

Identify whether the leak occurs in:

- HIR borrow analysis;
- MIR lowering;
- borrow-region construction;
- storage-end insertion;
- liveness dataflow;
- method receiver auto-borrow handling;
- loop back-edge joining.

Record the exact layer in the DEV entry before repair.

### 6.4 Required tests

Add:

```text
starkc/tests/dev137_while_condition_borrows.rs
```

#### Must pass

1. `while i < v.len() { v[i] = ... }`.
2. Condition calls a shared method, body mutates the same receiver.
3. Condition borrows through a field, body mutates the field owner.
4. Repeated iterations with changing length where semantics require re-evaluation.
5. Condition contains multiple temporary shared borrows.
6. Nested loops with independent receivers.
7. Equivalent explicit-block condition form, if syntax permits.
8. **DEV-137 × DEV-132:** a `VecGetRef`-backed indexed place borrow created in the `while` condition ends before the body, allowing a legal body mutation without extending the shared borrow.

#### Must reject

1. An explicitly stored shared reference that remains live into the body.
2. Condition returns or stores a reference used in the body.
3. Mutation while a genuine live shared borrow exists.
4. Mutable and shared aliasing violations unrelated to temporary condition evaluation.

### 6.5 Evidence

- HIR, MIR, and native agree.
- Index bounds and loop condition are re-evaluated each iteration.
- No borrow is ended earlier than the language permits.
- Existing borrow checker negative tests remain green.
- No workaround is added to packages.

### 6.6 Closure criteria

DEV-137 closes only when condition-only borrows end at the control-flow boundary and real body-live borrows remain rejected.

---

# Part D — DEV-136

## 7. DEV-136 — move on a terminating branch is treated as unconditional

### 7.1 Problem statement

This valid program is rejected:

```stark
if flag {
    return output;
}

output.push('a');
```

The move of `output` occurs only on a branch that terminates. That branch does not reach the later use.

The move-state join incorrectly includes a non-reaching predecessor.

### 7.2 Required dataflow rule

At a control-flow join, merge ownership state only from predecessors that reach the join.

Terminating edges include at least:

- `return`;
- unconditional trap/panic where modelled as non-returning;
- `break` relative to the relevant join;
- `continue` relative to the relevant join;
- diverging expressions, if represented.

Do not add a syntax-specific `if-return` exception.

### 7.3 Required tests

Add:

```text
starkc/tests/dev136_terminating_path_moves.rs
```

#### Must pass

1. Move then `return` on one branch; use on fallthrough.
2. Move then trap/panic on one branch; use on fallthrough, if trap is non-returning.
3. Nested early return.
4. Early return in one branch of a match.
5. Multiple terminating branches.
6. Loop `continue` and `break` cases where state joins differ.
7. Non-Copy values with Drop.

#### Must reject

1. Move on a branch that can reach the join.
2. Move on one of two reachable branches followed by use.
3. Maybe-moved state after a conditional without termination.
4. Partial field move where the moved path may reach the join.

### 7.4 Evidence

- Dataflow graph or debug snapshot shows unreachable predecessors excluded.
- Drop paths remain exactly once.
- Existing move checker tests remain green.
- No branch is incorrectly treated as terminating.

### 7.5 Closure criteria

DEV-136 closes only when ownership state is path-sensitive with respect to termination, not merely syntactically patched.

---

# Part E — DEV-139

## 8. DEV-139 — impl-level bounds are invisible to operator desugaring

### 8.1 Problem statement

Inside:

```stark
impl<T: Ord> Pair<T> {
    fn less_than(&self, other: &Pair<T>) -> Bool {
        self.value < other.value
    }
}
```

operator resolution rejects `<`, while the equivalent free generic function with `T: Ord` is accepted.

The method body does not inherit the impl's generic obligation environment.

### 8.2 Required semantic environment

A method inside an impl must be checked with:

```text
impl generic parameters
+ impl-level bounds
+ method generic parameters
+ method-level bounds
+ where-clause obligations, if supported
```

The same environment must be visible to:

- operator desugaring;
- CoreTrait dispatch;
- method lookup;
- trait-bound satisfaction;
- associated operations where supported.

Do not add an `Ord`-specific exception.

### 8.3 Required tests

Add:

```text
starkc/tests/dev139_impl_generic_bounds.rs
```

#### Must pass

1. `Ord` bound used by `<`, `<=`, `>`, `>=`.
2. `Eq` bound used by `==`, `!=`.
3. `Clone` bound used inside an impl method.
4. `Default` bound where already admitted.
5. `From` or conversion bound where already admitted.
6. Impl and method each contribute separate bounds.
7. Nested generic nominal types.

#### Must reject

1. Operator used without the required bound.
2. Wrong bound, such as `Eq` where `Ord` is required.
3. Bound present on an unrelated type parameter.
4. Ambiguous or unsatisfied trait obligations.

### 8.4 Cross-check

Compare with the generic environment used for free functions. Prefer reusing one canonical obligation-construction path rather than duplicating bound assembly.

### 8.5 Closure criteria

DEV-139 closes only when all trait-dependent operations inside impl methods see the complete impl-plus-method environment.

---

# Part F — DEV-138

## 9. DEV-138 — iterator-yielded `&str` is consumed after first use

### 9.1 Classification requirement

Do not assume this is independent.

First determine whether it is:

1. another DEV-121 runtime representation producer;
2. iterator binding emitting Move for a Copy item;
3. a `Str`/`String` equality or representation issue;
4. a borrow-lifetime issue;
5. a distinct defect.

### 9.2 Required diagnostic matrix

For the minimal reproducer, record:

- HIR static type of the iterator item;
- normalized type;
- MirTy;
- canonical Copy classification;
- emitted operand at loop binding;
- runtime `Value` variant in HIR;
- behaviour on first use;
- behaviour on second use;
- MIR interpreter result;
- native result.

Use the existing copy-canonicalization matrix style.

### 9.3 Decision rules

#### Fold into DEV-121 when

- static type is a shared reference;
- MIR uses Copy correctly;
- MIR/native are correct;
- HIR uses an owned or non-Copy runtime representation and consumes it.

Then:

- add the iterator producer to the permanent producer × use-mode × escape matrix;
- update DEV-121 residual exposure;
- repair the producer;
- do not allocate another independent root-cause class.

#### Treat as distinct when

- MIR emits Move for a Copy shared-reference item;
- all engines consume the item;
- iterator ownership semantics differ from ordinary shared references;
- the runtime representation is already canonical.

### 9.4 Required tests

At minimum:

1. iterator yields `&str`, used twice in loop body;
2. iterator yields `&String`, used twice;
3. iterator yields `&[UInt8]`, used twice;
4. iterator yields Copy scalar reference;
5. iterator yields non-Copy owned item and still moves once;
6. item passed to helper twice;
7. item stored or returned where legal;
8. direct producer control outside iteration.

### 9.5 Closure criteria

DEV-138 closes or folds into DEV-121 only after the engine matrix proves the mechanism.

---

## 10. External sample suite and in-tree regressions

Preserve the independence that allowed the external 18-package suite to find defects the compiler-owned corpora missed.

### 10.1 In-tree minimal reproducers

Add minimal, mechanism-focused reproducers to this repository as permanent regression tests.

Requirements:

- each DEV has a minimal case;
- expected pass/fail status is explicit;
- negative cases assert diagnostic code and relevant message content;
- positive cases execute through HIR, MIR, and native where applicable;
- no reproducer is rewritten to avoid a valid source construct;
- a machine-readable manifest links each case to its DEV and expected engine outcomes.

Suggested manifest fields:

```text
id
description
expected_frontend
expected_hir
expected_mir
expected_native
linked_dev
```

### 10.2 External task-shaped suite remains external

Keep the task-shaped suite in its own repository. Do not absorb or sanitize the full suite into the compiler repository.

Add a CI job that:

1. builds the compiler release artifacts from the candidate commit;
2. clones the external suite at a pinned commit SHA;
3. runs the suite against those built artifacts, not against an in-tree development shortcut;
4. uploads its manifest, logs, and results as CI evidence;
5. fails on any unexpected outcome.

The external repository must retain task-oriented programmes such as sorting a vector, graph traversal, and expression parsing. Its design should not be reorganized around compiler subsystems.

### 10.3 Outcome-history discipline

When a compiler fix changes an external task's expected result, update the external manifest in the same logical change set as the compiler repair. The suite history must show which compiler commit changed which task outcome.

Where cross-repository atomic commits are impossible, the compiler PR must pin the exact external-suite commit containing the matching expectation update before merge.

The external suite must fail on new accepted-but-unbuildable, accepted-but-trapping, or silently divergent behaviour.

---

## 11. Layer-audit hardening

The current layer audit reports findings but always passes.

Convert it from an informational printout into a registered-inventory gate.

Each probe should have an expected disposition:

```text
FrontEnd
Lowers
KnownDev("DEV-xxx")
```

The test must fail when:

- an unregistered layer defect appears;
- a closed DEV still reproduces;
- a probe unexpectedly changes from KnownDev to another state;
- a probe no longer reaches the intended construct;
- the total registered inventory changes without an explicit update.

Do not require zero findings immediately. Require zero **unregistered** findings.

---

## 12. CI and qualification requirements

For each defect commit, run the narrowest relevant suite plus the shared safety net.

Minimum local evidence before push:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features --no-fail-fast
```

Also run relevant targeted tests, including:

```bash
cargo test --test dev134_try_error_type
cargo test --test dev135_field_move_paths
cargo test --test dev136_terminating_path_moves
cargo test --test dev137_while_condition_borrows
cargo test --test dev139_impl_generic_bounds
cargo test --test copy_canon_matrix
cargo test --test mir_verify
cargo test --test mir_differential
cargo test --test c6_generated_corpus
```

Run first-party package qualification after any change affecting:

- `?`;
- ownership;
- loop borrowing;
- generic operator resolution;
- iterator items;
- lowering.

Run the exact CI qualification script over all ten packages.

For changes affecting the REST or provider path, rerun the P1 REST workload.

Before trusting any local qualification result:

```bash
git status --short
```

The working tree must be clean, or the evidence must explicitly state otherwise and must not be used for closure.

---

## 13. Documentation requirements

Update in the same commit as each repair:

- `starkc/docs/conformance/KNOWN-DEVIATIONS.md`
- `COMPILER-STATE.md`
- relevant work-package or evidence document
- test manifest, if introduced
- diagnostic map, if a new code is allocated

For DEV-134, record the owner ruling:

```text
`?` requires exact error-type compatibility.
Implicit From-based propagation is not part of this repair.
```

For DEV-138, record whether it is:

- folded into DEV-121;
- confirmed as a new root cause;
- still candidate/unconfirmed.

Do not mark candidate findings closed without mechanism evidence.

---

## 14. Commit structure

Recommended commits:

```text
CD-next    DEV-134 exact error-type enforcement for `?`
CD-next    DEV-137 condition-borrow lifetime boundary
CD-next    DEV-136 terminating-edge move-state joins
CD-next    DEV-135 inventory and DEV-135a soundness gate
CD-next    DEV-139 impl generic environment propagation
CD-next    DEV-138 classification and repair/fold
CD-next    DEV-135b field-sensitive precision, if required
CD-next    in-tree minimal regression manifest
CD-next    external-suite pinned CI integration
CD-next    layer audit converted to registered-inventory enforcement
CD-next    final reconciliation and qualification
```

Numbers are illustrative; use the repository's next available CD numbers.

Do not combine registration, broad refactoring, and several unrelated repairs into one commit.

---

## 15. Release gate

Do not publish or promote public compiler binaries until:

- DEV-134 is closed;
- DEV-137 is closed;
- DEV-136 is closed;
- DEV-135a is closed, or DEV-135b is complete because inventory proved parent poisoning unacceptable;
- aggregate CI is green;
- the external task-shaped suite runs from its own pinned repository in CI;
- no unregistered soundness defect remains from CD-334.

DEV-139 and DEV-138 should also be closed before calling the compiler broadly ready for early external users. DEV-135b may remain open only when DEV-135a is soundly closed and the partial-move limitation is explicit and bounded. If any remains open, release notes must state the exact bounded limitation.

---

## 16. Final acceptance checklist

The programme is complete only when all of the following are true:

- [ ] `?` cannot propagate an incompatible error type.
- [ ] No implicit conversion semantics were added without approval.
- [ ] A non-Copy struct field cannot be moved twice.
- [ ] Current sibling-after-partial-move usage is inventoried.
- [ ] DEV-135a rejects repeated field moves before execution.
- [ ] If parent poisoning ships, its sibling-use limitation is explicit and DEV-135b remains tracked.
- [ ] If DEV-135b is completed, sibling fields, nested paths, restoration, and parent/child move state are correct.
- [ ] Condition-only borrows end before the `while` body.
- [ ] Real body-live borrows remain rejected.
- [ ] Moves on terminating paths do not poison reachable joins.
- [ ] Reachable maybe-moves remain rejected.
- [ ] Impl-level bounds are available to all trait-dependent operations.
- [ ] Iterator-yielded shared references obey canonical Copy semantics.
- [ ] DEV-138 is either folded into DEV-121 with evidence or confirmed independently.
- [ ] All six reproducers are permanent tests.
- [ ] Minimal reproducers run in-tree and the independent task-shaped suite runs from a pinned external repository in CI.
- [ ] The layer audit fails on any unregistered finding.
- [ ] All ten first-party packages qualify.
- [ ] P1 REST remains green where relevant.
- [ ] `cargo fmt`, clippy, full test suite, differential suite, MIR verifier, and corpus are green.
- [ ] `COMPILER-STATE.md` and the deviation ledger match the implementation.
- [ ] The final integration commit has a clean aggregate `ci-complete` result.

---

## 17. Required final report

When complete, provide a report with:

1. commit hashes;
2. defect-by-defect root cause;
3. exact repair;
4. rejected alternatives;
5. new invariants or canonical authorities;
6. tests added;
7. full command list and results;
8. package qualification result;
9. external sample-suite result;
10. any residual open limitations;
11. final release recommendation.

Do not summarize a defect as fixed merely because the original reproducer passes. State what class of future program is now prevented from reproducing the same failure.

This is a standing closure law for every DEV, not only DEV-134 through DEV-139.
