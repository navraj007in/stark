# AC4 — adversarial architecture validation

**Packet:** WP-ARCH-CLOSE AC4, under CD-400. **Status: IN PROGRESS.** Two of the eleven required
authorities had no trial at all; one of those is now covered and found a real gap. §4 states exactly
what remains.

**Method:** trials run through `starkc/scripts/as8-mutate.py`, AS8's harness, extended rather than
replaced. Its structural guarantee is the reason: **a trial declares `expect` = KILLED or SURVIVED
before it runs**, and the harness reports CONFIRMED / UNEXPECTED against that declaration. A batch
whose SURVIVED-expected trials all come back KILLED is a harness detecting edits rather than
defects. The source file is always restored, including on interrupt.

---

## 1. Coverage against AC4's eleven authorities

AS8 selected its 26 trials under EI5, for a different question. Mapped onto AC4's required list:

```text
                                    trials       state
type identity and Copy              1,1b,1c,2    COVERED
runtime-function classification     da           COVERED   both copies killed
resolution / namespaces             9,9b,ac4-ns  COVERED   both halves: where names are filed
                                                 and where they are looked for (§2.4)
trait / bound dispatch              3,ac4-bound  PROBED    5 arms; 2 executed by any test.
                                                 AC4-F3 (§2.6) -- Ref, Core and Param arms are
                                                 never run
Drop determination                  1,ac4-drop   COVERED   4 arms, all killed. AC4-F4:
                                                 built-in destruction is observable only
                                                 incidentally (§2.7)
MIR lowering                        6,7          PARTIAL   trap-category assignment, array_order
MIR verification                    9            PARTIAL   paths_prefix_related only
provider/resource ownership         4,4b         PARTIAL   provider_sig::signature only
borrow / move ownership             9            PARTIAL   -> EXTENDED, see §2.2
pattern legality                    NONE         -> COVERED, see §2.1. FOUND A GAP
generic specialization env.         NONE         -> COVERED, see §2.3. TWO FINDINGS
```

## 2. Trials added by AC4

### 2.1 Pattern legality — `batch ac4-pat`. **An unguarded authority, now guarded**

Chosen first because it has the worst defect history in the tree: **DEV-222, 223, 225, 226 and 227
all landed in a single day**, and every one was a pattern that compiled, reported nothing, and
silently never matched. An authority with that record and zero adversarial coverage is what AC4
exists to find.

```text
AC4-MUT-PAT-001  Res::AssociatedFn admitted as a pattern     KILLED by 3   CONFIRMED
AC4-MUT-PAT-002  Res::Item(_) => true (any item legal)       SURVIVED      UNEXPECTED  <-- FINDING
                 ... control added ...
AC4-MUT-PAT-002  re-run                                      KILLED by 1   CONFIRMED
```

**The finding.** `resolution_is_pattern_legal` has three decision arms. Two were guarded. The
`Res::Item` arm — which admits a struct or a constant and rejects a function, a trait or a module
item — had **no control at all**: replacing it with `Res::Item(_) => true` left the entire suite
green. A regression there restores DEV-227's defect.

Reproduced concretely under the mutation:

```stark
mod m { pub fn f() -> Int32 { 1 } }
fn main() { let n: Int32 = 1; match n { m::f => { println(1); } _ => { println(2); } } }
```

```text
unmutated   E0200            rejected
mutated     (no diagnostic)  ACCEPTED as a pattern that never matches
```

The control added is `dev222_pattern_only_resolutions::a_qualified_path_naming_a_module_function_is_not_a_pattern`.

**Three invalid probes preceded the valid one, and the record says so.** The first used
`match n { helper => .. }` with a bare identifier — which **binds a fresh variable** and never
reaches the authority. It reported no diagnostics both mutated and unmutated, and was briefly read as
*"the defect is live"*. It was not: the probe never touched the code. Two further probes failed the
same way before it emerged that only a **qualified path** reaches `Res::Item`.

**The tell each time was the unmutated run agreeing with the mutated one.** That is the signal that
a probe does not reach its target, and it is the same signal that caught an unreaching program during
the `mir::borrows` trials. It is written into the new test so the next reader does not repeat it.

### 2.2 Borrow origin — `batch ac4-borrow`. Promoting a doc-comment residual into the harness

`mir::borrows` is the authority AC1 step 1 created (CE3, 2026-08-12); it postdates AS8 entirely. Its
own module-level trials found two of four rules uncontrolled, which was recorded in a doc comment —
a place nothing enforces. These trials put it under the harness.

```text
AC4-MUT-BOR-001  call-result guard removed          expect KILLED    KILLED by 3   CONFIRMED
AC4-MUT-BOR-002  a MOVE stops severing provenance   expect KILLED    KILLED by 1   CONFIRMED
AC4-MUT-BOR-003  aggregate component filter         expect SURVIVED  SURVIVED      CONFIRMED
```

**BOR-003 is declared SURVIVED and confirmed as such**, which is the point of declaring it. The
aggregate filter is precautionary and no control reaches it — **an explicit shared-fate style
classification rather than a silent gap**, which is what AC4's exit permits. A future run that comes
back KILLED is good news and must be re-declared.

BOR-002 is the rule whose absence over-refused `stark_http_client::follow` on the first DEV-160
repair attempt; it is now guarded by the module's pinned relation.

### 2.3 Generic specialization environment — `batch ac4-gen`. **Two findings**

The last AC4 authority with no trial at all.

```text
AC4-MUT-GEN-001  the binder->type map, emptied         SURVIVED   <-- AC4-F2, see below
AC4-MUT-GEN-002  substitute_ty's Ty::Param never subst  KILLED by 17
AC4-MUT-GEN-003  ARM-LEVEL: Ty::Fn substitutes params
                 but not `ret`                          KILLED by 1
```

**GEN-003 is the arm-level probe pattern legality argued for, and it passed** — killed by a test
named `substitution_reaches_every_position_a_parameter_can_hide_in`. Someone had already reasoned
about arm coverage for `substitute_ty`. That is what a covered authority looks like.

#### AC4-F1 — a convenience view with zero callers, and the duplicate it exists to prevent

**Class C.** `GenericEnvironment::substitutions()` states its own purpose:

> *"The name→type view to substitute with — **the same view `CallableInstantiation` publishes**, so a
> consumer or a test never has to build a second one."*

**It has zero callers**, and `bound_dispatch.rs:253` builds a byte-identical second one inline:

```rust
// GenericEnvironment::substitutions()        // bound_dispatch.rs:253
bindings.iter()                               environment.iter()
    .map(|(binder, ty)|                           .map(|(binder, ty)|
        (binder.name().to_string(), ty.clone()))      (binder.name().to_string(), ty.clone()))
    .collect()                                    .collect()
```

The view provided to prevent a duplicate is unused, and the duplicate exists.

**This finding came out of a mutation that was itself invalid.** GEN-001 first targeted
`substitutions()` and SURVIVED — because the method is unreached, not because a control is missing.
Recording that as a clean SURVIVED would have been a false result. The trial was retargeted at the
copy that IS reached, which produced AC4-F2.

#### AC4-F2 — the bound-specialisation environment has NO independent falsifier

**Class C. The strongest finding of the campaign so far.**

Emptying the substitution map that `specialize_bound_callable` actually uses — so every
bound-specialised callable keeps `T` instead of its instantiated type — is **not detected by
anything**:

```text
--lib (584)   native_c6_2_generics_traits   native_c5_4_generics
three_engine_differential (129)   mir_differential (132)   conformance
as3_invocation_authority
                                                    -> ALL PASS with the map emptied
```

**And the code is reached.** A probe counted the map built with **non-empty bindings 6 times** in
`native_c6_2_generics_traits` and `native_c5_4_generics` alone. So this is not an unreachable arm:
the authority runs, produces a specialised signature, and **nothing checks the result**.

That is precisely the AC4 exit condition failing — *"every critical authority has either an
independent falsifier, or an explicit shared-fate classification with an identified alternative
control."* This has neither yet; the trial is declared `SURVIVED` on measurement so the gap is
tracked mechanically rather than assumed closed.

**Disposition is an owner question, because the two repairs differ in kind.** Either the specialised
signature is consumed and needs a control, or it is not consumed and the construction is dead — and
which one it is decides whether the fix is a test or a deletion. Not taken here.

### 2.4 Resolution / namespaces — `batch ac4-ns`. **Well controlled**

AS8's only resolver trials are `item_is_visible_from` (batches 9, 9b), which is *visibility*, not
namespacing. DEV-228 rebuilt this surface — the resolver now carries the module/type/value
namespaces `NAME-RESOLVE-001` specifies — and AS8 predates it entirely, so the namespaces themselves
had never been mutated.

All three trials are arm-level, and all three died:

```text
AC4-MUT-NS-001  trait/struct/enum/alias/model filed under Value   KILLED by 85
AC4-MUT-NS-002  a FUNCTION filed under Type                       KILLED by  3
                first killer: a_type_and_a_value_may_share_a_spelling -- DEV-228's own
                motivating case, `struct Pair` alongside `fn Pair()`
AC4-MUT-NS-003  the READ side: NsHint::Type consults the VALUE map KILLED by 83
```

NS-001/002 break where names are *filed*; NS-003 breaks where they are *looked for*. Both halves are
watched.

### 2.5 A defect in the INSTRUMENT, found by disbelieving its output

`AC4-MUT-NS-002` first reported **"killed by 1 test"**, and that number was about to be recorded as
thin coverage for DEV-228 — an authority whose dedicated suite missed its own motivating case.

**That conclusion would have been false.** Applying the mutation and running `dev228_namespaces`
directly showed **two** of its tests failing, including `a_type_and_a_value_may_share_a_spelling`.

The cause was in the harness: it ran every target in one `cargo test` invocation **without
`--no-fail-fast`**, so cargo stopped at the first failing target. `--lib` failed first, and the
dedicated suites never ran at all.

```text
                     before      after --no-fail-fast
AC4-MUT-NS-001         76                85
AC4-MUT-NS-002          1                 3     first killer now DEV-228's own case
AC4-MUT-NS-003         76                83
```

**What was and was not affected.** KILLED/SURVIVED **verdicts were never wrong** — a kill is a kill,
and a SURVIVED trial ran every target by definition, since nothing failed to stop it. Only
`killer_count` was a lower bound. But a count that understates coverage invites precisely the wrong
conclusion about which authorities are watched, which is the conclusion this campaign exists to draw.

**Every AC4 trial was re-run under the corrected harness and all eleven CONFIRMED**, including
`AC4-MUT-GEN-001` still SURVIVING — so AC4-F2 stands on the corrected instrument, not the flawed one.

### 2.6 Trait / bound dispatch — `batch ac4-bound`. **AC4-F3: three of five arms are never executed**

AS8's only trial here is batch 3 (`core_trait_contract` receiver) — one function, and it SURVIVED.
`satisfies_bound_identity` is the authority, and it is a match over **five semantic arms**:
reference forwarding, the primitive matrix, the Core-type rules, the nominal impl witness, and the
generic-parameter discharge. One trial on one function says nothing about the other four.

Four arm-level trials were written. **All four survived**, and the campaign rule — challenge the
measuring path when a result is unexpectedly clean — is what turned that from a coverage claim into
the real finding.

**The arm census.** A probe on `satisfies_bound_identity`, whole lib suite:

```text
Primitive Hash   6      Nominal Hash    5
Primitive Eq     6      Nominal Eq      5
Primitive Ord    1      Nominal Sz      2      Nominal Source  1

Ref arm          0      Core arm        0      Param arm       0
```

**The authority is reached 26 times and touches two of its five arms**, for three bound names out of
the nine it decides. Never executed by any test in the compiler:

```text
Ty::Ref forwarding      whether `&T` forwards Eq/Ord/Clone/Hash/Display to `T` --
                        the `fn show<T: Display>(v: &T)` shape the file's own comment calls routine
Ty::Core rules          Clone/Display/Hash/Eq/Ord/Default over Core types, AND the entire
                        Iterator membership list (eight core iterator types)
Ty::Param discharge     DEV-067(a)'s own repair -- a bound on a generic parameter discharged by
                        the ENCLOSING function's declared bounds. Its absence was a real defect
                        that failed simple recursion with E0500
```

The `Primitive/Ord` arm is reached exactly **once**, and not with `Bool` — so admitting `Bool` as
`Ord`, one token's difference from the `Eq` arm directly above it, changes nothing observable. One
reach is not coverage of a matrix over eight primitives.

**Also measured: five integration suites reach this authority ZERO times** — `conformance`,
`three_engine_differential`, `adversarial_trait_impls`, `c46_class_a`, `dev075_operator_bounds`.
Only `--lib` reaches it at all. Not every suite was swept, and the claim is limited to those six.

**All four trials are now declared SURVIVED on measurement**, so the gap is tracked mechanically. A
run that comes back KILLED is good news and must be re-declared.

**Why this is worse than "no falsifier".** AC4-F2 was a constructed fact nothing consumed — the
repair was deletion. This is the opposite: live semantic rules, each one decided on real programs,
with **no test that runs them**. The difference matters for disposition, and neither is fixed by
adding a mutation.

### 2.7 Drop determination — `batch ac4-drop`. Controlled at arm level, **and AC4-F4**

AS8's only trial here is `nominals_with_destructor` (batch 1, SURVIVED) — which answers *"does this
nominal declare a destructor"*, one input to the real authority. `requires_drop_glue_with` is the
authority: nine arms, and the module header calls it exhaustive on purpose so *"a new MirTy variant
must be classified at this one authority"*.

```text
AC4-MUT-DRP-001  MirTy::String stops owning anything      KILLED by 2   <-- see AC4-F4
AC4-MUT-DRP-002  the STRUCT arm stops recursing into
                 fields                                    KILLED by 7
AC4-MUT-DRP-003  a HOST RESOURCE stops needing its close   KILLED by 2
AC4-MUT-DRP-004  the TUPLE arm stops recursing             KILLED by 4
```

DRP-002 is the realistic defect and is well controlled: a struct with no `Drop` impl but a `String`
field stops dropping the field, caught first by
`a_partially_moved_value_destroys_only_the_surviving_field`. Note that AS8's
`nominals_with_destructor` trial would still answer correctly under DRP-002 — which is why one
trial on one input is not coverage of the authority.

#### AC4-F4 — a built-in type's destruction has no observable control

**Class C.** DRP-001 was killed, so it counts as covered — and the *killers* are the finding.

```text
killers of "MirTy::String requires no drop glue":
    mir::borrows::tests::census_which_rules_a_program_reaches
    mir::borrows::tests::the_whole_relation_is_pinned_for_the_reported_shape
```

**Both are borrow-origin characterization tests written on 2026-08-12**, one day old, which pin
exact local numbering — numbering that shifts when drop elaboration changes. Neither is a drop test.
Detection was incidental.

Verified independently, with the mutation applied and the drop suites run directly:

```text
three_engine_differential   129 passed
native_c6_1_ownership        24 passed
as4_destructor_authority      6 passed
```

**The suites that exist to compare destruction do not notice that `String` stopped being
destroyed.** Had those two tests not been written the previous day, DRP-001 would have SURVIVED.

**The mechanism is stated in the comparator's own documentation.** `support/differential.rs`: *"A
drop-observing case emits a reserved frame from its own `Drop` impl; the harness extracts those
frames from stdout."* A `String` has no user `Drop` impl, so **its destruction cannot emit a frame,
and the drop log is structurally incapable of observing it**. The same holds for every built-in
owning type.

```text
user destructors     observable -- the case emits its own frame. Well controlled (DRP-002/004)
built-in ownership   NOT observable by this mechanism. A leak is not a wrong answer, and
                     nothing in the suite observes memory
```

**This is a shared-fate finding, not a missing test.** Adding another differential case would not
help: the observation channel cannot see built-in destruction at all. An alternative control is
needed — the Miri lane already runs the slot primitives under Stacked Borrows and is the natural
place, or a leak-observing harness. **Naming the alternative control is what AC4's exit requires
when independent falsification is unavailable**, and this one is named rather than assumed.

---

## 3. What AC4's exit requires, and what is not yet true

> *every critical authority has either an `independent falsifier`, or an explicit shared-fate
> classification with an identified alternative control. **No authority may be described as
> independently verified solely because HIR, MIR and native inherit the same answer.***

**Not yet met.** Seven authorities are PARTIAL — each has at least one trial, but a single trial on
one function is not the same as the authority having a falsifier. The distinction matters: pattern
legality would have read as PARTIAL-to-covered on a shallow count, and two of its three arms were
guarded while the third was not.

## 4. Remaining work, in priority order

```text
1  AC4-F2's disposition                  the bound-specialisation environment has no
                                         falsifier. Owner call: add a control, or delete a
                                         construction nothing consumes
2  the six other PARTIAL authorities     each needs its arms enumerated, as pattern legality's
                                         and substitute_ty's were, rather than counted
4  shared-fate register reconciliation   AC4's exit feeds ENGINE-SHARED-FATE-REGISTER.md; the
                                         register has not yet been updated with these results
```

**Do not read "2 covered, 7 partial" as 82% done.** The pattern-legality result is the argument
against that reading: a covered-looking authority had an entirely unguarded arm, and only enumerating
the arms found it.
