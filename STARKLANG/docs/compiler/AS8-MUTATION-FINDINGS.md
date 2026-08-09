# AS8 — compiler-source mutation trials

**Packet:** AS8, consuming `WP-ENGINE-INDEPENDENCE` EI1–EI5 (CD-392). Harness:
`starkc/scripts/as8-mutate.py`. **Date:** 2026-08-09.

**Status: all EI5 batches run, plus three the packet added (1b/1c isolation, 4b re-selection, 9 resolver/verifier) and the AS8-DA pairs. THREE EI DOCUMENTS WERE WRONG AND ARE CORRECTED HERE.** These are the
first mutations in this repository to touch compiler source; EI2's §14.1 residual said so in
advance and it was accurate.

---

## The result table

Every trial declares `expect` **before** it runs, so a surprise is recorded as `UNEXPECTED` rather
than reinterpreted afterwards.

| Trial | Authority | Mutation | Expected | Actual | Verdict | Killers | Killer independence |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| `MUT-SELFTEST-LIVE` | harness | `Int8` stops reporting as an integer | KILLED | KILLED | **CONFIRMED** | — | — |
| `MUT-SELFTEST-NOOP` | harness | add a binding, use it | SURVIVED | SURVIVED | **CONFIRMED** | 0 | — |
| `AS8-MUT-001` | `ESF-COPY-001` | drop the Copy+Drop exclusion | SURVIVED | **KILLED** | UNEXPECTED | 25 | `CROSS_ENGINE_DERIVED` |
| `AS8-MUT-002` | `ESF-DROP-001` | collect `Clone` impls, not `Drop` | SURVIVED | **KILLED** | UNEXPECTED | 25 | `CROSS_ENGINE_DERIVED` |
| `AS8-MUT-003` | `ESF-COPY-001` | = MUT-001, vs `copy_canon_matrix` alone | SURVIVED | SURVIVED | **CONFIRMED** | 0 | — |
| `AS8-MUT-004` | `ESF-COPY-001` | drop the all-fields-Copy requirement | SURVIVED | **KILLED** | UNEXPECTED | 5 | `CROSS_ENGINE_DERIVED` |
| `AS8-MUT-005` | `ESF-COPY-001` | zero-variant enum vacuously `Copy` (reverts CD-251) | SURVIVED | SURVIVED | **CONFIRMED** | 0 | — |
| `AS8-MUT-006` | `ESF-COPY-001` | `&mut T` reports `Copy` | SURVIVED | SURVIVED | **CONFIRMED** | 0 | — |
| `AS8-MUT-009` | `ESF-COPY-001` | = MUT-004, **vs `c61f_structural_copy`** | KILLED | KILLED | **CONFIRMED** | 4 | **`HAND_AUTHORED`** |
| `AS8-MUT-010` | `ESF-COPY-001` | = MUT-005, **vs `c61f_structural_copy`** | KILLED | KILLED | **CONFIRMED** | 1 | **`HAND_AUTHORED`** |
| `AS8-MUT-011` | `ESF-COPY-001` | = MUT-006, **vs `c61f_structural_copy`** | KILLED | KILLED | **CONFIRMED** | 1 | **`HAND_AUTHORED`** |
| `AS8-MUT-007` | `ESF-TRAP-001b` | wrong category, **MIR path only** | KILLED | KILLED | **CONFIRMED** | 4 | `CROSS_ENGINE_DERIVED` |
| `AS8-MUT-008` | `ESF-TRAP-001a` | vocabulary no-op | SURVIVED | SURVIVED | **CONFIRMED** | 0 | — |
| `AS8-MUT-012` | `ESF-COPY-002` | `&mut T` reports `Copy` **over `MirTy`** | KILLED | **SURVIVED** | UNEXPECTED | 0 | — |
| `AS8-MUT-013` | `ESF-TYPE-001` | `()` stops canonicalising to `Unit` (reverts TYPE-PRIM-001) | KILLED | **SURVIVED** | UNEXPECTED | 0 | — |
| `AS8-MUT-014` | `ESF-TRAIT-001` | `Eq::eq` declared by-value | SURVIVED | SURVIVED | **CONFIRMED** | 0 | — |
| `AS8-MUT-015` | `ESF-TRAIT-001` | `Ord::cmp` returns `Bool` | SURVIVED | SURVIVED | **CONFIRMED** | 0 | — |
| `AS8-MUT-016` | `ESF-PROV-001` | provider params reversed | KILLED | SURVIVED | **VOID — see Finding 6** | 0 | — |
| `AS8-MUT-025` | `ESF-PROV-001` | = MUT-016, re-selected | KILLED | KILLED | **CONFIRMED** | 4 | `MIR-0005` via `mir/verify.rs` |
| `AS8-MUT-017` | `ESF-RES-001` | host resource is `Copy`-eligible | KILLED | KILLED | **CONFIRMED** | 3 | `a_host_resource_is_never_copy` |
| `AS8-MUT-018` | `ESF-TYPE-001` | `is_integer` drops `UInt64` | KILLED | KILLED | **CONFIRMED** | 9 | `entire_frozen_corpus_agrees` |
| `AS8-MUT-024` | `ESF-TYPE-001` | `is_integer_primitive` drops `UInt64` | KILLED | KILLED | **CONFIRMED** | 4 | `entire_frozen_corpus_agrees` |
| `AS8-MUT-019` | `ESF-TYPE-001` | `strip_ref` stops at one level | KILLED | **SURVIVED** | UNEXPECTED | 0 | — |
| `AS8-MUT-020` | `RA-OVERFLOW` | drop the destination-width range check | KILLED | KILLED | **CONFIRMED** | 3 | `c5_snapshot_overflow_replays_…` |
| `AS8-MUT-021` | `RA-SHIFT` | remove STARK's shift-count check | KILLED | KILLED | **CONFIRMED** | 1 | `invalid_shift_trap_agrees` |
| `AS8-MUT-022` | `RA-DROP` | array destruction order reversed | KILLED | **SURVIVED** | UNEXPECTED | 0 | — |
| `AS8-MUT-023` | `EV-CORPUS-C6` | trap assignment, vs the corpus alone | KILLED | KILLED | **CONFIRMED** | 1 | `the_corpus_replays_through_every_required_engine` |

Per EI5, kill rates are **not pooled**. Every trial above is `SHARED_AUTHORITY`, so the meaningful
split is by killer independence, not by tag:

```text
killed by CROSS_ENGINE_DERIVED evidence   4   MUT-001, 002, 004, 007
killed by HAND_AUTHORED spec controls     3   MUT-009, 010, 011
survived everything selected              6   MUT-003, 005, 006, 008, 012, 013
```

**Five of thirteen predictions were wrong, and they were wrong in BOTH directions** — three
predicted survivors were killed, two predicted kills survived. That ratio is the packet's most
useful single number: EI5's predictions were reasoned from EI2's reading of the evidence base, and
they are right about as often as they are wrong. Nothing in the register earns belief until it has
been mutated.

## Finding 1 — the differential detects a Copy *contradiction*, never a Copy *error*

EI4 predicted that a mutation in a front-end authority all three engines inherit **survives every
differential suite**. Batch 1 killed two, and the natural reading — "the differential is stronger
than EI4 thought" — is wrong. The captured divergence says what actually happened:

```text
--- HIR oracle ---                     --- MIR ---
100 9002 101 9003 9020 9001            100 9002 101 9003 9020 9001
9001 9020 9003 9002                    (nothing: destructors never ran)
```

`copy_eligible_types` **consults** `nominals_with_destructor` to exclude Copy+Drop. MUT-001 broke
that exclusion directly and MUT-002 broke it indirectly, and in both cases the result was a type
that is simultaneously `Copy` and `Drop`. MIR's drop planning asks *"is it Copy?"*
(`ESF-COPY-002`); the HIR interpreter's destruction walk asks *"does it have a destructor?"*
(`ESF-DROP-001`). **Two different shared authorities, one followed by each engine.** The
differential saw them disagree with each other.

The isolation trials settle it. Every one of the 25 + 25 + 5 killing tests is a **drop** test —
`box_drop_timing_agrees`, `variant_payload_drop_order_with_wildcards_agrees`,
`vec_clear_droppable_runs_destructors_agree`. And the two mutations with **no drop consequence**
survived completely:

```text
MUT-005   zero-variant enum vacuously Copy   0 killers   THE ACTUAL CD-251 DEFECT
MUT-006   `&mut T` reports Copy              0 killers   breaks one-&mut-XOR-many-&
```

> **A wrong Copy rule is invisible to the three-engine differential unless it also makes a
> destructor run in one engine and not another.** EI4's prediction was right; Batch 1's mutations
> were simply not the experiment that tests it.

MUT-005 is not a hypothetical. The code comment at that site records that vacuous `Copy` on
zero-variant enums broke exactly-once close for host resources in the front end — the CD-234/CD-251
shape. It can be reintroduced today and no differential suite notices.

## Finding 2 — `ESF-COPY-001` HAS an independent control, and three EI documents say it does not

EI2 recorded `none in-tree`, EI4 ranked `ESF-COPY-001` critical on `control is
IMPLEMENTATION_GENERATED`, and EI5's Selected-tests column listed `copy_canon_matrix` and the
differential suites. **All three missed `starkc/tests/c61f_structural_copy.rs`.**

Thirteen hand-authored tests derived from OWN-COPY-001, pinning the **negative** surface by
behaviour — reuse after move must be `E0100` — rather than by enumerating the checker's match arms:

```text
c61g_mutable_reference_field_stays_move             killed MUT-011  (1 test)
c251_a_zero_variant_enum_is_not_structurally_copy   killed MUT-010  (1 test)
c61g_mixed_copy_and_non_copy_fields_stays_move      killed MUT-009  (4 tests)
```

Batch 1c is Batch 1b with that one suite added and the predictions flipped to KILLED. **All three
confirmed.** The survivors were an artefact of **test selection**, not a gap in the tree.

This is a `HAND_AUTHORED` control in EI0's sense: its expectations come from the normative rule,
and it is capable of contradicting the implementation. It is the strongest control any register
authority has apart from `EV-SPEC-FIXTURES` and the external suites.

**And MUT-003 confirms EI2's other claim in the same breath.** The same mutation run against
`copy_canon_matrix` alone **survived** — the matrix enumerates from `core_method_signature`, so it
is a transcription, not a control. EI2 said so; the trial demonstrates it. EI5 scheduled that
question for Batch 8 and it is answered here.

## Finding 3 — `ESF-TRAP-001` is two authorities, and only one of them is invisible

The register makes one entry, ranks it `INVISIBLE`, and EI2-R3 states that *"a mis-categorisation is
invisible to every mechanism in the tree."* The measurement contradicts it:

```text
src/interp.rs                    28 assignment sites   all 10 categories
src/mir/lower.rs + mir/interp.rs 30 assignment sites   all 10 categories
src/backend/generated_rust       3 assignment sites    remainder inherited from the runtime
```

The same operation is categorised **twice, independently, in two files.** MUT-007 changed division
by zero to `IntegerOverflow` on the MIR path only, and the differential caught it immediately:

```text
divzero.stark: trap category mismatch — MIR IntegerOverflow vs oracle message "division by zero"
```

So:

```text
ESF-TRAP-001a   trap category VOCABULARY (the enum)        INVISIBLE          no control
ESF-TRAP-001b   trap category ASSIGNMENT at each site      PARTIALLY_VISIBLE  the HIR oracle
```

The genuine residual is **narrower and cannot be posed as a source mutation at all**: if the enum
names the wrong concept, or omits one, every engine and the corpus manifest are wrong together and
nothing in the tree can disagree. MUT-008 is the honest no-op that marks the boundary.

---

## Finding 4 — the "strongest control in the tree" does not cover the rule EI5 aimed at it

`EV-SPEC-FIXTURES` is `SPEC_DERIVED`, and EI2, EI4 and EI5 all name it the strongest control
available. EI5's Batch 2 row predicted it would kill a `ESF-TYPE-001` mutation and said what a
survivor would mean:

> *"A survivor would mean the spec fixtures do not cover TYPE-PRIM-001's own rule, which would be a
> notable gap given the rule has a fixture history."*

**MUT-013 survived.** `unit_or_tuple` was reverted so `()` no longer canonicalises to `Unit` — the
exact pre-TYPE-PRIM-001 defect — and `conformance` passed, spec fixtures included
(`spec_conformance ... ok`; the suite was in the selection, not omitted). Zero killers.

The reason is structural and worth stating, because it limits what `EV-SPEC-FIXTURES` can ever
control: **the fixtures classify programs by parse and semantic OUTCOME — accepted, rejected, which
diagnostic — and TYPE-PRIM-001 is about type IDENTITY.** Reverting the canonicalisation makes no
program fail to compile. It changes an internal representation that no fixture asserts, and the
divergence would only surface where the two spellings are distinguished downstream.

```text
EV-SPEC-FIXTURES controls   what the front end ACCEPTS AND REJECTS
it does not control         which internal type a construct canonicalises to
```

That is a narrower reading of the tree's best control than any EI document has, and it was
obtainable only by mutating.

## Finding 5 — `ESF-COPY-002` is unexercised, and EI5 predicted this correctly

MUT-012 made `&mut T` report `Copy` over `MirTy`. EI5 predicted a kill on the ground that the HIR
engine classifies over `Ty` rather than `MirTy` and should disagree. It survived `mir_differential`,
`three_engine_differential` and `c61f_structural_copy` alike, with zero killers.

EI5's own survivor consequence is the right reading and needs no revision:

> *"If it survives, the hir↔mir differential is not exercising the case; residual against
> `EV-DIFF-*` coverage rather than against the rule."*

**This is a coverage gap, not an authority gap.** No differential case duplicates a `&mut` and
observes the consequence. It pairs instructively with MUT-006 — the same rule broken on the
front-end side, also a survivor, but killed instantly by `c61f_structural_copy`. The front-end form
has a control; the MIR form has none, and neither engine notices.

---

## Finding 6 — `MUT-016` was VOID, and the reason is a third selection failure

`AS8-MUT-016` reversed provider parameter order and was recorded SURVIVED. **That result is void,
not negative.** EI5 named `EV-PROVIDER-LOOP` as its control and the harness cannot run it: the
loopback suites live in `packages/*/native/Cargo.toml`, separate crates, while every trial runs
`cargo test -p starkc`. The trial measured the harness's reach.

`AS8-MUT-025` re-ran the identical mutation against the suites that drive a provider program through
`mir/verify.rs`, its one in-tree consumer. **Killed by 4 tests**, the verifier reporting
`MIR-0005 call argument: expected Ref { mutable: true, inner: UInt64 }, found Ref { mutable: false,
inner: Slice(UInt8) }`.

**`ESF-PROV-001` therefore HAS an in-tree independent control, and it is the MIR verifier.** EI2
recorded only the external loopback, and EI2-R2 framed the gap as "two engines, not three". Both
were counting ENGINES — and `mir/verify.rs` is not an engine. It is a check that derives its answer
separately, which is what a control is.

> **Three selection failures in one packet.** MUT-005/006 (the control existed and was not
> selected), MUT-016 (the control was in another crate), MUT-013 (the control was selected and was
> structurally incapable). The rule EI5 gained after the first was necessary and twice
> insufficient. The general statement it should have carried from the start: **a mutation trial
> measures THE SELECTED SUITES. A survivor is a statement about them until the selection is
> independently justified.**

## Finding 7 — a Core trait contract can be arbitrarily wrong and nothing notices

`Eq::eq` declared to take `self` by value, and `Ord::cmp` declared to return `Bool` rather than
`Ordering`, both **survived** `copy_canon_matrix`, `conformance`, `gate4a_prelude_traits` and
`three_engine_differential`. EI5 predicted survivors; the prediction holds.

`copy_canon_matrix` cannot help, for the reason MUT-003 already demonstrated — it enumerates *from*
`core_method_signature`. `ESF-TRAIT-001` is `high` risk with no control, and the corpus census found
no test citing TRAIT-DEF-001 / TRAIT-LAW-001 / TRAIT-ASSOC-001 outside corpus cases. **AS8-R10.**

## Finding 8 — the rustc conversions are real; one of them is unguarded

EI3's central claim was that overflow, shift and drop order were **converted** from rustc
assumptions into STARK decisions. Two of three are now enforced rather than documented:

```text
RA-OVERFLOW  drop the destination-width range check  KILLED, 3 tests
             "MIR/NATIVE DISAGREEMENT on completion versus trap"
RA-SHIFT     remove STARK's own shift-count check    KILLED, 1 test
             "MIR/NATIVE DISAGREEMENT on trap category"
```

Both kills are `CROSS_ENGINE_DERIVED`: MIR traps independently of the generated Rust, which is
exactly the structure that makes the conversion meaningful rather than cosmetic.

**`RA-DROP` survived.** Reversing `drop_plan::array_order` was observed by nothing. EI5 stated the
consequence in advance — *"a survivor means mir↔native inheritance (`ESF-DROP-002`) is unguarded by
the one control that exists"*. The HIR engine walks a different structure and would disagree; no
case exercises it. **AS8-R8.**

## Finding 9 — both `is_integer` copies are live and guarded; `strip_ref` is not

`AS8-DA-001` resolved by measurement rather than argument:

```text
MUT-018  is_integer            drops UInt64   KILLED, 9 tests
MUT-024  is_integer_primitive  drops UInt64   KILLED, 4 tests
```

Both copies are reached and both are guarded, so neither is dormant and neither has drifted. That
settles the disposition the owner ruling left conditional: consolidation is safe **after AS8**,
because neither copy checks the other — there is no independence to lose.

`strip_ref` is the opposite. Making it stop after ONE level **survived** `conformance`, both
differentials and `c61f_nested_refs`. TYPE-METHOD-002 makes nested-reference receivers normative —
auto-deref removes one leading `&` at a time — and nothing distinguishes one level of stripping from
many. **AS8-R9.**

---

---

## The corpus census (AS8-R3)

`starkc/scripts/as8-control-census.py`, keyed on **normative rule IDs rather than function names**,
because `c61f_structural_copy.rs` never mentions `copy_eligible_types` — it cites OWN-COPY-001. A
symbol census finds tests that touch the implementation and misses every test that pins the rule,
which is the only kind that can act as a control.

```text
ESF-COPY-001/002   c61f_structural_copy.rs                            confirmed by MUT-009/010/011
ESF-RES-001        a11_host_resource, c788_resource_lifecycle,        FIVE front-end controls;
                   dev146_resource_borrow_weakening,                  EI2 recorded only the
                   dev151_resource_method_dispatch,                   EXTERNALLY_DERIVED loopback
                   dev153_slice_parameter_in_resource_method
seven others       none found by rule citation
```

**Stated limitation, in the script and here.** It finds tests that CITE a rule ID. The spec-fixture
manifest carries no rule IDs at all, so EI5's claim that `EV-SPEC-FIXTURES` controls `ESF-TYPE-001`
could be neither confirmed nor refuted by census — and MUT-013 then refuted it by measurement.
Absence of a hit is not evidence of absence of a control; where the census reports NONE, the honest
response is a trial.

---

## What this changes

```text
ENGINE-SHARED-FATE-REGISTER.md   split ESF-TRAP-001; ESF-COPY-001 gains a control and drops
                                 from critical-with-no-control
ENGINE-EVIDENCE-INDEPENDENCE.md  add EV-COPY-STRUCTURAL; correct EI2-R3; correct question 4,
                                 whose "No, for the six INVISIBLE authorities" is now measured
                                 to be wrong for two of them
ENGINE-RISK-PROFILES.md          the EI5 handover ranking is rebuilt on measurement
ENGINE-MUTATION-TARGETS.md       predictions revised; Selected-tests columns must name the
                                 HAND_AUTHORED controls, which is how MUT-005/006 were missed
```

---

---

## Disposition — what a survivor is, and what it is not (owner ruling, 2026-08-09)

**A surviving mutant does not mean the compiler is wrong.** It means:

> *"I can introduce this defect and the selected evidence cannot detect it."*

It does **not** mean:

> *"The current compiler contains this defect."*

Batch 2 is the clean demonstration. `MUT-012` and `MUT-013` survived and revealed **missing
evidence and coverage, not wrong behaviour at HEAD** — `&mut` is correctly non-`Copy` over `MirTy`
today, and `()` correctly canonicalises to `Unit` today. Allocating DEV numbers for those would
record defects that do not exist and would corrupt the ledger in the direction that is hardest to
undo.

```text
surviving hypothetical defect                      ->  AS8 residual / coverage gap / evidence gap
current HEAD demonstrably wrong                    ->  DEV
duplicated copies demonstrably DISAGREE            ->  DEV
dormant but genuinely incorrect code already at    ->  DEV candidate, on the DEV-179 precedent
    HEAD
historical defect reintroduced ONLY by mutation    ->  cite the historical DEV/CD + AS8 residual;
                                                       NO new DEV
```

`AS8-MUT-005` sits in the last row and is the sharpest case for it: it reintroduces the CD-251
zero-variant-enum defect, which was real, was found, and is fixed. The survivor is a statement
about the differential's reach, not about HEAD. It cites CD-251 and opens `AS8-R1`; it does not
open a DEV.

**No DEV number is allocated by this packet.** Every shared-authority survivor remains an `AS8-R*`
finding until something is shown to be wrong at HEAD.

## Residuals opened

```text
AS8-R1  A wrong Copy rule with NO drop consequence is invisible to every differential suite.
        The control that catches it is c61f_structural_copy, a FRONT-END test. The three-engine
        differential contributes nothing to ESF-COPY-001 except via ESF-DROP-001 contradiction.

AS8-R2  ESF-TRAP-001a (vocabulary) has no control and admits no source mutation. It can only be
        addressed by deriving the category set from the specification rather than from the enum.

AS8-R3  EI2/EI4/EI5 each independently missed an in-tree control. The evidence audit was
        conducted by reading the differential machinery and the register, not by enumerating the
        test corpus. A test-corpus census is owed before any further "no control exists" claim.

AS8-R4  `copy_canon_matrix` is confirmed a transcription (MUT-003 survived). It should be
        described as a DRIFT DETECTOR, not as evidence for the Copy rule.

AS8-R5  EV-SPEC-FIXTURES does not control TYPE-PRIM-001 (MUT-013 survived with the suite in the
        selection). The fixtures classify programs by what the front end ACCEPTS AND REJECTS;
        TYPE-PRIM-001 is about type IDENTITY, and reverting it makes no program fail to compile.
        The tree's strongest control is narrower than three EI documents claim.

AS8-R6  ESF-COPY-002 is unexercised: no differential case duplicates a `&mut` and observes the
        consequence (MUT-012 survived every selected suite). A COVERAGE residual against
        EV-DIFF-*, not an authority residual — EI5 predicted this reading and it stands.

AS8-R7  EIGHT of twenty-five predictions were falsified, in BOTH directions. Every error traces
        to reasoning about the evidence base by reading it rather than by mutating it. No
        register entry should be cited as controlled or uncontrolled until it has been mutated.

AS8-R8  ESF-DROP-002 array destruction order is unguarded. MUT-022 reversed
        `drop_plan::array_order` and nothing observed it. The HIR engine walks a different
        structure and WOULD disagree; no case exercises it. Coverage gap, not authority gap.

AS8-R9  `types::strip_ref` recursion is unguarded. MUT-019 made it stop after one level and
        survived both differentials and `c61f_nested_refs`. TYPE-METHOD-002 makes nested-reference
        receivers normative, so this is a normative rule with no executable control.

AS8-R10 ESF-TRAIT-001 has no control of any kind. MUT-014 and MUT-015 made Core trait contracts
        arbitrarily wrong — `Eq::eq` by value, `Ord::cmp` returning Bool — and all four selected
        suites passed. `copy_canon_matrix` enumerates FROM the table it would have to check.

AS8-R11 EI2-R2 is corrected, in the compiler's favour: ESF-PROV-001 DOES have an in-tree
        independent control, `mir/verify.rs` (MUT-025, 4 kills). EI2 counted engines, and the
        verifier is not an engine. "Two engines, not three" understated the evidence.
```

## Method note — the harness had to be repaired mid-packet, twice

Recorded because both defects had the shape this packet exists to find.

1. **`KILLED` alone is not a usable record.** EI5 makes `killer independence` required, and the
   harness captured neither the failing tests nor the divergence — so Batch 1's two surprises could
   not be interpreted without re-running by hand. Findings 1 and 3 are both *entirely* conclusions
   from the divergence text, not from the pass/fail bit.
2. **Restoring in `finally` was not enough.** A `git add` of a directory issued while a trial was
   in flight committed AS8-MUT-002 to a pushed branch, and every C6.5 job failed on it. The
   harness now refuses a target that differs from HEAD before the trial and after the restore.
   **A mutation harness is a parallel writer inside your own session.**

---

---
