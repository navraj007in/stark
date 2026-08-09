# AS7 exit qualification — pass modularisation and compiler API boundary

**Verdict: PASS on all five exit criteria. Unconditional — exact-head CI green.**

> **CRITERION 2's PASS BELOW IS SUPERSEDED — see §9, added 2026-08-09 (CD-393).** The forcing
> test that produced it observed 36 of 234 methods and five violations were live under its
> green, including the `traits -> convert` cycle §2 records as broken. **Everything above §9 is
> preserved exactly as written on 2026-08-08**, including the sentences now known to be wrong;
> §9 carries the repaired detector, the repair, and the re-qualification.

**Head:** `977b7a3`. **Date:** 2026-08-08. **Branch:** `wp-arch-stability/as7-modularization`,
worktree `/Users/nexper/Documents/GitHub/stark-as7`, forked from `3f18e49`.
**Authority:** Campaign B approved for execution 2026-08-08; AS6 CLOSED (CD-390) 2026-08-08.

---

## 1. What AS7 was asked to do

> Split the type checker by semantic ownership; replace ambient current-file/module/impl/generic
> state with scoped context objects; define a narrow supported compiler facade; document a
> cycle-free dependency direction.

Thirteen commits, `321c634` … `977b7a3` (this report is the fourteenth).

## 2. Exit criteria

### Criterion 1 — no semantic behaviour or diagnostic structure changes

**PASS, on the strongest available evidence: zero fixture changes.**

```text
git diff --stat 3f18e49..HEAD -- starkc/tests/fixtures
                                 STARKLANG/tests/spec-fixtures
                                 starkc/tests/c6-corpus
  (empty)
```

Across ten extraction packets, **no test fixture and no expected output was edited.** Behaviour is
not "unchanged as far as we tested" — the suites that define checker behaviour were never adjusted
to accommodate the split.

Two behaviour changes were made deliberately, both in Packet 2, and both are *fixes* to latent
defects rather than consequences of moving code:

```text
current_fn_ret    was CLEARED to None on scope exit; now RESTORES the enclosing value
current_module    was assigned per item and never restored; now enters and leaves a scope
```

Both were correct only under the invariant "item checking never nests" — the invariant AS7's own
splitting was most likely to break. Restoring is behaviour-identical while the invariant holds and
correct by construction after. Pinned by `as7_fn_scope_saves_and_restores_the_enclosing_function`
and `as7_item_scope_saves_and_restores_the_enclosing_module`, both proved in **both** directions:
reverting `exit_fn_scope` to clear fails exactly those tests and the structural check
`as7_no_ambient_field_is_cleared_on_scope_exit`.

### Criterion 2 — dependency direction documented and cycle-free

**PASS, with evidence regenerated at the exact head rather than accumulated from per-packet runs.**

```text
types <- state <- infer <- traits <- convert <- bounds <- patterns/body <- items <- mod
```

Documented in `AS7-MODULE-DAG.md`; enforced by `starkc/tests/as7_module_dependencies.rs`, green at
`977b7a3`, four tests. The check derives edges from explicit references **and** from `TypeChecker`
method ownership, because Rust's inherent-method resolution hides dependencies from the import
graph entirely: a method defined in `traits.rs` and called as `self.select_impl(..)` from `body.rs`
is a real edge that produces no `use` statement.

**The graph was revised once, under the packet's stop condition.** Packet 7 found `convert` and
`traits` strongly connected — converting `HashMap<K,V>` must prove `K: Hash + Eq`, and proving
`Iterator<Item = Foo>` must convert the written `Item`. Both directions are load-bearing. The owner
ruling rejected a permitted cycle and added `bounds` as the missing orchestration layer; `traits`
answers trait *identity* and may not convert HIR types, `bounds` owns the complete written
constraint. The original decomposition is preserved in the record with a dated correction.

**Why per-packet greens are not the evidence.** Four of the ten packets reported a green from a
checker that was blind or partly blind — `find` vs `rfind` truncating the ownership map at an early
inline test module, trait-impl methods conflated with inherent ones, a glob ban firing on `mod.rs`'s
own test module, and edges to items still living in `mod.rs` being invisible entirely. Each was
found by deliberately introducing a violation and observing that nothing failed. Only this
exact-head run is admissible.

### Criterion 3 — internal modules are not accidentally part of the supported public API

**PASS, and it removed one accidental export.**

```text
public API of `crate::typecheck` before AS7 : 31
public API of `crate::typecheck` now        : 31   identical
```

All nine submodules are declared `mod`, not `pub mod`; nothing reaches them except through
`mod.rs`'s re-exports. `crate::typecheck::{Ty, TypeTables, analyze, …}` resolve exactly as before,
and **AS7 required no import migration anywhere in the crate** — `flow.rs`, `borrowck.rs`,
`session.rs` and the integration tests are untouched.

One difference, and it is the criterion working: **`TypeChecker` was `pub` before AS7 and is now
`pub(super)`.** Nothing outside `typecheck` ever used it — the only mentions are doc comments. It
was accidentally public, and criterion 3 is exactly the requirement to stop that.

*A measurement note.* A first pass also reported `Token` as newly public. It is a regex artefact:
the string `pub use hidden::Token;` appears inside a **STARK source literal** in a test. Recorded
because this qualification's own measurements were not exempt from the failure mode the packet kept
finding.

### Criterion 4 — default dependency/build surfaces contain only active compiler architecture

**PASS.**

```text
git diff --stat 3f18e49..HEAD -- starkc/Cargo.toml Cargo.toml
  (empty)
```

AS7 added no dependency, no feature and no build target. The Cranelift retirement was taken in
Sprint 1 under its own audit gate and did not return here.

### Criterion 5 — file-size reduction reported as an outcome, not used as the acceptance criterion

**Reported as an outcome.** No criterion above is satisfied by a line count, and the packet was
never steered by one — Packet 7 *added* a module and Packet 9a moved 1,795 lines into an existing
one because the dependency graph required it, not because either improved a size figure.

```text
mod.rs production   14,432 -> 596

body 4,396 | traits 3,200 | infer 1,262 | types 1,153 | items   986
convert 901 | state  896 | patterns 429 | bounds  116 | mod     596 production
```

## 3. The ambient state, which the packet ordered first

The opening inventory (`AS7-OPENING-INVENTORY.md`, `3f18e49`) separated two questions the packet's
risk statement conflates:

```text
does control leave between a save and its restore?   -> a LEAK      none found, 14 pairs checked
is a field written on a path that never saved it?    -> WRONG CONTEXT   two latent sites
```

All eight ambient fields now enter and leave through named operations in `state.rs`. Every pair
saves and restores; none clears to a default. They are **not** one `AmbientContext`: the fields have
different dynamic scopes, and collapsing them would allow restoring a function's return type while
leaving an impl's `Self` behind.

`state` owns the mutation and not merely the storage, for a reason discovered while doing it: **the
dependency checker sees method calls and imports, but not raw field access.** A later module writing
`self.current_self_ty` directly would be an edge criterion 2 could not see.

## 4. The AS6 boundary, still frozen

`as7_does_not_reopen_the_as6_extension_boundary` passes at the head.
`extensions/tensor/check.rs` mentions `check_expr` exactly twice, both in doc comments stating its
deliberate absence. The extension consumes checked expression types; it does not cause expression
checking.

## 5. Recorded limits

1. **`pub(super)` is wide inside `typecheck`.** `TypeChecker`'s fields and many formerly private
   items are visible throughout the module tree so the passes can read them. Nothing escapes the
   crate's public API — criterion 3 measures that — but this is weaker encapsulation than a
   finished design would carry, and narrowing it is later work, not AS7's.
2. **`body.rs` is 4,396 lines.** Splitting expression checking further (calls, operators, control
   flow) is a plausible next cut and was not attempted; the approved decomposition names one `body`.
3. **35 historical `.md` files still reference `typecheck.rs`.** Deliberate: gate reports, the state
   archive and closed work packages are preserved as written. The stale-path census fixed every
   executable reference and every live document.

## 6. The finding worth carrying forward

**Four separate defects in this packet's own verification, each found only by deliberately trying to
make a check fail.**

```text
`find` vs `rfind`             the ownership map was built from mod.rs's first 384 lines
trait-impl vs inherent        `impl TensorCheckCtx for TypeChecker` produced phantom edges
"prose" classification        conformance TOMLs and two source-scanning tests were executable
cargo check --lib             does NOT compile #[cfg(test)] code — ten packets of "compiles
                              clean" meant less than reported
```

The common shape is one sentence: **a check that does not cover the thing being claimed cannot
support the claim.** Every one of these produced a confident green that was not evidence.

The compensating discipline that worked is also one sentence: **introduce the violation on purpose
and watch the check fail.** That is how the checker's blind spots were found, how the ambient
conversion was proved, and how the AS6 boundary test earned its place. AS8 is assurance work and
should adopt it as a standing requirement rather than a habit.

## 7. Evidence index

```text
local, head 977b7a3    665 tests across 19 targets, 0 failed
                       clippy --workspace --all-features --all-targets -- -D warnings CLEAN
                       cargo fmt --check CLEAN
                       conformance baseline verification CLEAN
                       as7_module_dependencies 4/4 at the exact head

CI, head 977b7a3       CI 24/24 success, C7.8 Native Capabilities 4/4 success, ZERO failing
CI, head 4c4311a       CI 24/24 success, C7.8 Native Capabilities 4/4 success, ZERO failing
                       Includes `fmt, clippy, test` on linux-x64, macos-arm64 AND windows-x64 —
                       the platform no local evidence in this packet ever covered — plus both
                       C6.4 tier-1 qualification runs, C6.5 corpus replay and tier-1 agreement,
                       spec fixture conformance, the pinned external sample suite, DEV-160 under
                       Miri, and first-party package qualification on all three platforms.
```

## 8. Verdict

**AS7 PASSES all five exit criteria. Unconditional — CI is green on the exact head across all
three Tier-1 platforms, 28 jobs across two workflows, zero failing.**

Propagation follows the order AS6 established: `COMPILER-STATE.md` first, then
`WP-ARCHITECTURE-STABILIZATION.md`, then the derived contributor documents.

**AS8 may open.** It is assurance written against the frozen AS6/AS7 result and cannot be batched
with the work it challenges.

---

## 9. Criterion 2 correction and re-qualification (2026-08-09, CD-393)

**Added after the fact under owner instruction. Sections 1–8 are unmodified.**

### The detector was blind to 85% of its input

`method_owners()` in `as7_module_dependencies` recognised an enumerated list of visibility
prefixes — `fn`, `pub fn`, `pub(crate) fn` — and not `pub(super) fn`, which is what nearly every
method AS7 extracted was given.

```text
methods the ownership map saw   36 of 234      about 15%
```

The repair stops enumerating and strips **any** visibility qualifier, `pub(...)` included. With it,
criterion 2 went RED with five violations:

```text
traits -> convert     the Packet 7 cycle §2 records as broken by the `bounds` ruling
infer  -> convert
state  -> body
state  -> infer
state  -> traits
```

### Why §2's account of Packet 7 was wrong

The owner ruling on Packet 7 rejected a permitted cycle and introduced `bounds`. That ruling was
applied to the **explicit references** and never to the **methods** — conversion-dependent trait
machinery stayed in `traits` and went on calling `convert_hir_type`. §2 records the cycle as
discharged because the check that would have contradicted it could not see method calls.

### The negative control that did not exist

Packet 1's proof injected `use super::X`, which exercises the explicit-reference detector alone.
The method-ownership detector — the half that matters for a packet whose entire content is moving
methods — was never injected against. **A check with two mechanisms needs a negative control per
mechanism**, and §6's lesson (*a check that does not cover the thing being claimed cannot support
the claim*) is exactly this, applied one level lower than it was written.

The missing proof now exists: injecting `self.convert_hir_type(id)` into `traits.rs` yields
`typecheck/traits.rs -> typecheck/convert.rs is not permitted`, and green returns on removal.

### The repair

`trait_contracts.rs` (1,191 lines) holds the eight methods that must know what a written type
means, above `convert`. Identity and impl selection stay in `traits` and convert nothing.

The three `state` edges were a misclassification of my own: the publication family was placed in
`state` because publication writes storage — classifying by **effect** rather than by **need**.
Each resolves, instantiates or selects trait candidates first.

```text
types <- state <- infer <- traits <- convert <- {bounds, trait_contracts} <- patterns/body <- items <- mod

body 4,842 | traits 2,043 | mod 1,846 | trait_contracts 1,191 | infer 1,156
types 1,153 | items 986 | convert 901 | state 569 | patterns 429 | bounds 116
```

Eleven modules. `state` 896 -> 569. Figures are `wc -l`; §5's "596 production" for `mod.rs` used a
method the report does not state and that I could not reproduce (474 by brace-matching its four
inline `#[cfg(test)]` modules, at both heads), so totals are used here instead.

### Re-qualified result

```text
2 dependency direction documented and cycle-free   PASS   234 of 234 methods owned, 4/4 green
1 no semantic behaviour change                     PASS   PROVED: the whitespace-stripped line
                                                          multiset over typecheck/ is identical
                                                          across the repair but for imports,
                                                          module docs, `impl` wrappers, comments
3 internals not accidentally public                PASS   crate::typecheck API 31 -> 31, same set
4 build/dependency surface unchanged               PASS   no manifest change
5 size reported, never the criterion                      the repair ADDED a module
```

759 tests across 18 targets, 0 failed; clippy `--workspace --all-features --all-targets -D
warnings` clean; conformance baseline clean.

### The measurement this qualification should have carried

**How much of its input did the check observe?** Not "did it pass", and not "did it fail on an
injected violation" — 36-of-234 would have been visible on day one to anyone who printed the size
of the ownership map. Checker coverage is now a required figure in any qualification that rests on
an executable check.

