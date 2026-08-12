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
resolution / namespaces             9,9b         PARTIAL   item_is_visible_from only; the
                                                 NAMESPACES themselves are untested and DEV-228
                                                 postdates AS8
trait / bound dispatch              3            PARTIAL   core_trait_contract receiver only
Drop determination                  1            PARTIAL   nominals_with_destructor only
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
2  resolution / NAMESPACES               DEV-228 rebuilt this and AS8 predates it entirely;
                                         the namespaces themselves have never been mutated
3  the six other PARTIAL authorities     each needs its arms enumerated, as pattern legality's
                                         and substitute_ty's were, rather than counted
4  shared-fate register reconciliation   AC4's exit feeds ENGINE-SHARED-FATE-REGISTER.md; the
                                         register has not yet been updated with these results
```

**Do not read "2 covered, 7 partial" as 82% done.** The pattern-legality result is the argument
against that reading: a covered-looking authority had an entirely unguarded arm, and only enumerating
the arms found it.
