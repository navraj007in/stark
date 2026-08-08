# AS8 — compiler-source mutation trials, batches 0/1/1b/1c/6

**Packet:** AS8, consuming `WP-ENGINE-INDEPENDENCE` EI1–EI5 (CD-392). Harness:
`starkc/scripts/as8-mutate.py`. **Date:** 2026-08-09.

**Status: five batches run. THREE EI DOCUMENTS ARE WRONG AND ARE CORRECTED HERE.** These are the
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

Per EI5, kill rates are **not pooled**. Every trial above is `SHARED_AUTHORITY`, so the meaningful
split is by killer independence, not by tag:

```text
killed by CROSS_ENGINE_DERIVED evidence   4   MUT-001, 002, 004, 007
killed by HAND_AUTHORED spec controls     3   MUT-009, 010, 011
survived everything selected              4   MUT-003, 005, 006, 008
```

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
