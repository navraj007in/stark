# AC4 — adversarial architecture validation

**Packet:** WP-ARCH-CLOSE AC4, under CD-400. **Status: MET — 2026-08-12, on owner dispositions of
F4 and F6.** All eleven authorities addressed; F1–F7 all closed; shared-fate register reconciled. Seven findings, F1–F7; three resolved (F1, F2 by deletion, F7 by a written
falsifier), four open. The shared-fate register is reconciled (§2.11). §4 states exactly what remains.

**The campaign's own headline result is methodological.** Four of the seven findings — F2, F3, F5,
F6 — were visible only because a **survival was challenged rather than recorded**. Each would have
entered the record as a pass, or as a false gap, had the first result been trusted. A mutation
result is not interpreted until the measuring path itself has been challenged, whenever the result
is unexpectedly clean, unexpectedly sparse, or inconsistent with a dedicated regression suite.

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
trait / bound dispatch              3,ac4-bound  COVERED   5 arms; AC4-F3 REPAIRED (§2.6) --
                                                 ac4_bound_arms executes all three that were
                                                 never run; 4 of 4 mutations now die
Drop determination                  1,ac4-drop   COVERED   4 arms, all killed. AC4-F4 PARTIALLY
                                                 repaired (§2.7): the DECISION is controlled;
                                                 the leak itself stays unobservable
MIR lowering                      6,7,ac4-lower  COVERED   language semantics it uniquely
                                                 owns. AC4-F7 (§2.10) -- CD-007's assignment
                                                 order had NO control; written, mutation dies
MIR verification                    9,ac4-verify COVERED   36 rules; AC4-F5 REPAIRED (§2.8) --
                                                 MIR-0029/0035/0037 now have malformed-MIR
                                                 cases; 4 of 4 trials die
provider/resource ownership       4,4b,ac4-prov  COVERED   signature + lifecycle. AC4-F6:
                                                 release is controlled by the PACKAGE lane, not
                                                 by starkc's own suite (§2.9)
borrow / move ownership             9,ac4-borrow COVERED   mir::borrows, 3 arms (§2.2)
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

**AC4-F3 REPAIRED, 2026-08-12.** `starkc/tests/ac4_bound_arms.rs` — eight controls that make the
three unexecuted arms execute. All four mutations now die, each to a purpose-built case:

```text
BND-001  Ref forwarding drops Display   -> a_reference_forwards_display_to_its_referent
BND-002  Primitive matrix admits Bool   -> bool_satisfies_eq_but_not_ord
BND-003  Iterator list drops VecIter    -> a_vec_cursor_satisfies_iterator
BND-004  Param discharge disabled       -> an_enclosing_bound_discharges_a_callees_obligation
```

**Three of the eight exist to stop a lazy arm passing.** `a_parameter_without_the_bound_does_not_
discharge_it` catches an arm that answers `true` unconditionally, which would satisfy every positive
case; `char_is_ordered` paired with the `Bool` rejection catches an arm excluding both, which would
satisfy the `Bool` test while being wrong.

**Every case was probe-confirmed to reach its arm before being written.**
`fn show<T: Display>(v: &T)` looks like it exercises reference forwarding and does not — the body
dereferences and the check sees `T`. The shape that works is `show(&n)`, with `T` INSTANTIATED to a
reference. Writing the plausible-looking version is how the gap arose in the first place.

**A language limitation was found while writing the Iterator case and recorded rather than filed.**
`for _x in it` where `it: I` with `I: Iterator` is refused — `E0001 "for-loop requires an iterable
value, found 'I'"`. A `for` loop needs a concrete iterable; the bound alone is not enough. Adjacent
to DEV-144, not the same thing. The test uses `.next()` because the bound check is what it is for.

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

> **PARTIALLY REPAIRED 2026-08-12.** See the note at the end of this subsection.

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

### 2.8 MIR verification — `batch ac4-verify`. **AC4-F5: three rules no test exercises**

AS8's only trial here is `paths_prefix_related` (batch 9) — one predicate out of a verifier carrying
**36 distinct `MIR-nnnn` rules across 60 functions**.

**This authority needs a different mutation strategy, and the difference is the point.** Removing a
CHECK does not break correct programs; it only matters if something feeds the verifier malformed
MIR. So a surviving mutation here is a statement about the verifier's **negative** cases — whether
any test constructs the bad shape the rule exists to reject.

Selected by census rather than by intuition:

```text
36  MIR-nnnn rule ids in verify.rs
33  named by at least one test
 3  named by NO test    MIR-0029   MIR-0035   MIR-0037
```

```text
AC4-MUT-VER-001  MIR-0035: storage_dead on a PROJECTED place accepted   SURVIVED   CONFIRMED
AC4-MUT-VER-002  paths_prefix_related always false (positive control)   KILLED by 3
```

**VER-002 is the control that makes VER-001 legible.** AS8 had already killed that predicate, so a
survival there would have meant the method stopped working on this authority rather than that a rule
was unguarded. It was killed, first by
`dev117_drop_elaboration_moves_are_exempt_but_user_moves_are_not`.

**AC4-F5 REPAIRED, 2026-08-12.** All three census-only rules now have malformed-MIR cases in
`mir_verify`, and all four trials in the batch are CONFIRMED KILLED:

```text
MIR-0035  projected storage_dead      -> rejects_storage_dead_on_a_projected_place
MIR-0029  dangling close binding      -> rejects_a_close_binding_naming_a_call_outside_the_arena
MIR-0037  undefined spec-word bits    -> rejects_a_format_spec_word_with_undefined_bits
paths_prefix_related (control)        -> dev117_drop_elaboration_moves_are_exempt_...
```

**The first MIR-0029 mutation was a NO-OP, and it read exactly like a missing control.** Falling
back to `provider_calls.first()` on an **empty** arena still yields `None`, so the check fired
anyway and the trial "survived" without the rule ever being disabled. Same class as the unreachable
mutations earlier, in a new disguise: **a mutation that does not change behaviour is
indistinguishable from an unguarded rule unless the mutation itself is checked.** The replacement
removes the diagnostic and keeps the control flow, which is what disabling a rule means.

For MIR-0037 the out-of-range **word** arm was chosen over the operand-shape arms deliberately: a
wrong constant is what a miscompiling lowering would emit, whereas a wrong operand *type* would
already have failed the runtime-callee signature check upstream.

**Original finding, retained.** MIR-0035 enforces A12's rule that storage liveness belongs to a
**whole local**
— *"ending 'part of' a local's storage is not a thing MIR can mean, so a projection here is a
lowering defect"*. Disabling the check entirely is not detected by `--lib`, `mir_verify`,
`mir_differential`, or `a12_storage_end_shapes`. **No test constructs a projected `storage_dead`**,
so nothing proves the rule is enforced.

Two caveats stated rather than glossed:

```text
NOT a soundness claim   the rule may be unreachable because lowering never emits the shape.
                        That is a good reason for it to be a verifier check -- defence in depth --
                        and a bad reason to believe it works
MIR-0029 / MIR-0037     named by no test either, and NOT mutated here. The census identifies them;
                        only MIR-0035 was falsified
```

**Disposition.** A verifier rule is worth a negative test precisely because its positive path is
always green: correct MIR passes whether or not the rule exists. The repair is one hand-built
malformed body per unexercised rule, which is `mir_verify`'s existing shape.

### 2.9 Provider / resource ownership — `batch ac4-provider`. **AC4-F6, with the control named**

AS8's trials here are `provider_sig::signature` (batches 4, 4b) — the ABI **signature**, not the
**lifecycle**. CD-347/348 are explicit that a resource-shaped provider must successfully acquire,
use *and release*, and that a failure-only path is the weaker claim. These two attack release.

```text
AC4-MUT-PRV-002  the registry stops recognising any resource type   KILLED by 15   CONFIRMED
AC4-MUT-PRV-001  select_closes iterates the map that is EMPTY at
                 that point -- the defect its own comment records   SURVIVED       <-- AC4-F6
```

**PRV-002 is the positive control and it matters here.** It rules out the easy explanation: the
provider suites *do* reach this area, 15 tests deep. So PRV-001's survival is not "these tests
ignore providers".

**But the measuring path was challenged anyway, and that is the finding.** A probe shows
`select_closes` reached **zero times** by all four controls:

```text
--lib                     0        a10_provider_resource   0
a11_host_resource         0        a10_provider_verify     0
```

So the survival means **never run**, not **not detected** — the same shape as the bound-dispatch
arms, and the reason a survival is never interpreted before reachability is established.

**The alternative control exists, was run today, and is named rather than assumed.** Close selection
is exercised by the package qualification lane, not by `cargo test -p starkc`:

```text
qualify-first-party-packages.py     EXIT 0
    tls: TLS 1.3 session verified, used and closed explicitly
    tls: TLS 1.2 session verified, used and closed explicitly
    tls: drop released the session and the socket under it
```

That last line is the release half, observed against a live peer — which is precisely what CD-347/348
required and why they refused a failure-only path.

**Classification: shared-fate, with an identified alternative control** — the branch AC4's exit
permits. The residual risk is stated plainly: the control lives in a **different workspace and a
different CI lane**, so a change to `select_closes` will not be caught by the compiler's own suite,
and a contributor running `cargo test -p starkc` will see green. That is a real gap in feedback
speed even though the claim itself is covered.

### 2.10 MIR lowering — `batch ac4-lower`. **AC4-F7: a rule enforced only by a comment**

At 13,008 lines this authority cannot be enumerated arm by arm, so selection follows a different
rule: **the language semantics lowering uniquely owns** — the ones no other phase can restate —
rather than a sample of its code. `mir/lower.rs`'s own header says evaluation order is *"preserved
**structurally**"*, by the order operands are lowered into temporaries. Nothing type-checks it and
no verifier rule restates it.

```text
AC4-MUT-LOW-001  `&&`/`||` evaluate BOTH sides      KILLED by 1  -> now 4
AC4-MUT-LOW-002  CD-007's RHS-before-LHS inverted   SURVIVED     -> now KILLED
```

**AC4-F7, Class C — RESOLVED in the campaign.** LOW-002 was *reached* — every assignment lowers
through that line — and survived anyway, because **every assignment in the suite has an inert
left-hand side**. `a = f()` cannot observe an ordering. `a[idx()] = val()` can, and no such case
existed.

`starkc/tests/cd007_evaluation_order.rs` is the falsifier, four cases:

```text
an_assignment_evaluates_its_rhs_before_the_lhs_place    both sides print; order is the assertion
a_false_left_operand_means_the_right_never_runs         && short-circuit, by effect
a_true_left_operand_of_or_means_the_right_never_runs    || mirror
a_guarded_index_is_safe_only_because_the_right_side     short-circuiting as a TRAP property:
    _is_skipped                                         `i < 2 && a[i] == 1` completes only
                                                        because the right side is skipped
```

**The failure signal is worth recording.** The inverted lowering fails as **`HIR/MIR DISAGREEMENT on
stdout_bytes`** — the HIR oracle evaluates in its own order, so a lowering divergence surfaces as
engines disagreeing. That is a shape a shared-fate defect can never produce, and it is the strongest
kind of evidence this architecture can generate.

LOW-001 was killed before the new cases, but by a **single** test. A language rule that basic
deserves a case that states it directly rather than catching it incidentally; it now has three.

### 2.11 Shared-fate register — reconciled

AC4's exit feeds `ENGINE-SHARED-FATE-REGISTER.md`; the reconciliation lives there. Three things are
worth stating at the campaign's own level.

**No visibility classification changed.** Eight of eleven entries remain INVISIBLE to some engine
pair. What AC4 changed is the *reason* behind two and the *confidence* in a third — a more useful
result than a moved number, because a row whose reason is wrong is a row that will be defended on
the wrong grounds.

**Five findings became real entries, not residuals** — a correction to this campaign's first
instinct. EI0 froze the **vocabulary**, not the inventory, so adding entries that use the existing
terms is legitimate; declining to add them would have left the register describing a compiler that
no longer matches it. `ESF-VERIFY-001` is `ENGINE_LOCAL` rather than a shared-fate row, because MIR
verification is deliberately an independent checker and F5 is an **evidence** gap in its negative
cases — **not "engines inherit one answer" but "no engine is ever asked the question"**.

**One pre-existing prose/JSON divergence was found and deliberately NOT resolved.**
`ESF-PROV-001`'s visibility is `UNKNOWN` in the prose and `INVISIBLE_MIR_NATIVE` in the JSON — one
cell, and the whole difference between the register's *"eight of eleven"* and the JSON's nine. AC4's
pass is *reconcile, not improve*, and EI0's binding rule says `UNKNOWN` never resolves silently,
including in the direction that flatters the register. Settling it needs the hir cell **measured**,
which is EI2's open work and not AC4's to close.

**F1 and F2 produced no register change, and that is correct.** The bound-specialisation signature
was deleted rather than controlled, so the authority no longer produces the fact. An authority that
stops existing needs no shared-fate row, and manufacturing a consumer so the register had something
to classify is the reversal CD-401 refused.

---

## 3. What AC4's exit requires, and what is not yet true

> *every critical authority has either an `independent falsifier`, or an explicit shared-fate
> classification with an identified alternative control. **No authority may be described as
> independently verified solely because HIR, MIR and native inherit the same answer.***

**Not yet met, and the reason has changed.** No authority is PARTIAL any more — all eleven have
arm-level or semantics-level trials. What blocks the exit is four specific gaps (§4), each of which
is a *named* deficiency rather than an unexamined surface:

```text
F3  an authority whose arms are not EXECUTED cannot be said to have a falsifier
F5  two censused verifier rules were never mutated; one was, and is unenforced
F4  a control is NAMED for built-in destruction but not built
F6  a control EXISTS for resource release but lives outside the compiler's suite
```

The distinction that got the campaign here: pattern legality would have read as nearly covered on a
shallow count, and two of its three arms were guarded while the third was not. **Coverage is a
property of an authority's arms, not of it having a trial.**

## 4. Dispositions — AC4 is MET

Seven findings, all closed. Five by repair, two by owner disposition (2026-08-12).

```text
F1  CLOSED   dead convenience view; deleted with F2
F2  CLOSED   dead specialised signature; deleted rather than given a manufactured consumer
F3  CLOSED   ac4_bound_arms -- all five bound-dispatch arms execute; 4 of 4 mutations die
F4  CLOSED   DISPOSITION ACCEPTED -- decision-level falsifier sufficient
F5  CLOSED   three malformed-MIR cases; 36 of 36 verifier rules named by a test
F6  CLOSED   DISPOSITION ACCEPTED -- external package-qualification control accepted
F7  CLOSED   cd007_evaluation_order -- CD-007 was enforced only by a comment

shared-fate register   RECONCILED, 11 entries -> 16
```

### AC4-F4 — disposition accepted, with the caveat preserved

```text
AC4-F4 = DISPOSITION ACCEPTED / CLOSED

Decision-level falsifier sufficient for the authority claim.
Native built-in deallocation consequence remains an explicit assurance residual and
must not be described as directly observed.
Leak harness = future assurance work, not WP-ARCH-CLOSE work.
```

`ac4_builtin_destruction` falsifies the authority being audited: a wrong decision in
`requires_drop_glue_with` changes the MIR, and a purpose-built control detects it. **That is
architecture qualification.** The boundary stays explicit:

```text
WHAT AC4 PROVES          the destruction DECISION is exercised and falsifiable
                         wrong decision -> MIR changes -> purpose-built control fails

WHAT AC4 DOES NOT PROVE  that every native allocation is eventually freed
                         that no generated program can leak memory
```

The second is a **runtime/memory-assurance** question, not evidence that the destruction authority
lacks a falsifier. Treating it as an architecture-closure blocker would expand AC4 after the fact.

### AC4-F6 — disposition accepted, with a standing requirement

```text
AC4-F6 = DISPOSITION ACCEPTED / CLOSED

Authoritative control:  first-party package/provider qualification lane
Residual:               `cargo test -p starkc` alone cannot detect this regression
Requirement:            package/provider qualification remains a REQUIRED closure and
                        release qualification input for changes affecting resource lifecycle
```

The external control is **semantically stronger** than a synthetic compiler-unit test would be: it
exercises acquire → use → explicit-and-drop release against a live TLS peer. Its deficiency is a
slow, separate feedback lane — **not absent evidence**.

**No second local lifecycle test is added**, deliberately. Duplicating the package lane inside
`starkc` for directory symmetry would recreate exactly the duplicate-evidence problem this campaign
has spent its time removing.

## 5. What AC4 established, beyond the findings

**The campaign repeatedly caught errors in the interpretation of its own evidence.** Four of the
seven findings — F2, F3, F5, F6 — were visible only because a **survival was challenged rather than
recorded**, and three separate instrumentation defects were found on the way:

```text
--no-fail-fast absent          killer_count was a lower bound; NS-002 read as "killed by 1"
                               when its dedicated suite killed it with 2
a no-op mutation               MIR-0029's first mutation could not fire on an empty arena, and
                               "survived" without the rule ever being disabled
unreachable mutations          three bound arms and select_closes were never executed by the
                               chosen controls, so their survivals said nothing
```

That is the argument for the verdict rather than a caveat on it: a campaign that can falsify its own
measuring assumptions is a campaign whose remaining green results mean something.
