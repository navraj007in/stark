# Post-C10 deviation repair programme — final report

**Programme:** `STARK-Post-C10-Compiler-Deviation-Repair-Programme.md` (§22)
**Date:** 2026-08-10
**Branch:** `codex/post-c10-deviation-repair`

```text
baseline SHA   689d26d26990399d1de3026c13c271c403a45032   (origin/develop at start)
final SHA      5967a4288e6ec7b2c24c0204a79f204fc0408c31
```

## Population

```text
starting population (A)   13
non-reproducing            0     nothing in the population failed to reproduce
repaired                   6     DEV-180, 157, 168, 159, 220(new), and DEV-165 (population B)
reclassified               2     DEV-120 -> documented limit; DEV-098 -> accepted-indefinitely
closed by owner decision   1     DEV-167 -> documented non-promise (CE1)
newly registered           2     DEV-220 (found by P3), DEV-221 (an unnumbered residual)
remaining open (A)         8     DEV-140..145, DEV-160, DEV-221
dormant                    1     DEV-179
accepted-indefinitely      2     DEV-011, DEV-098
```

**Population A: 13 → 8.** Computed by `starkc/scripts/c10-deviation-populations.py` at every step
and never hand-edited, per §17.

The count is not a clean 13 → 6 because two IDs were *added* without anything getting worse.
DEV-220 was a real defect nobody had found; DEV-221 already existed as an unnumbered paragraph
inside DEV-168. Both are now trackable, which is the point of a number.

Separately, the tool's "named in COMPILER-STATE.md, owning no heading here" bucket went **16 → 0**.

## Category split (§22)

```text
semantic defects closed    DEV-220  a diverging arm captured the join's inference variable
                           DEV-157  `!` had no generated-Rust representation
                           DEV-168  a qualified core-trait call had no MIR lowering
                           DEV-180  `&mut self` carried an obsolete take/write-back model

native subset widened      NONE. Measured, not assumed: eight borrow/move shapes built with the
                           pre- and post-repair compilers, and the only cells that moved were two
                           `E0502` leaks becoming named refusals. Nothing that built stopped
                           building; nothing newly builds. §12.3 satisfied without a claim widening.

operational defects closed DEV-159  concurrent `stark build` of one program corrupted its own
                                    generated-crate directory

ergonomic defects closed   DEV-167  closed as a documented non-promise on an owner CE1 decision,
                                    not repaired -- the specification never promised the method form

outside population A       DEV-165  `connect_timeout` accepted and ignored (population B)
```

## HIR / MIR / native agreement

Every semantic repair carries three-engine cases comparing normalised outcomes — completion with
stdout, or trap category with exact source line — rather than one engine's output. DEV-220 (5
cases), DEV-157 (7), DEV-168 (3), plus the upgrade of
`qualified_calls_disambiguate_the_two_traits` from front-end-and-oracle to full three-engine
agreement.

**No engine disagreement remains in the repaired set.** DEV-160's remaining shapes are refused
identically by construction: they are refused before code generation, so there is no native
behaviour to disagree with.

## CI

```text
commit      5967a4288e6ec7b2c24c0204a79f204fc0408c31
workflow    CI                          run 31353396547   SUCCESS   24/24 jobs
workflow    C7.8 Native Capabilities    run 31353396543   SUCCESS
platforms   linux-x64, macos-arm64, windows-x64 (all Tier-1)
```

Jobs include `fmt, clippy, test` on all three platforms, `first-party package qualification` on all
three, `C6.4 tier-1 qualification` and `tier-1 agreement`, `C6.5 corpus replay` and `mutation
controls`, `C7 P1 REST workload` on all three, `spec fixture conformance`, `release package smoke`
on all three, `External sample suite (pinned)`, `C6.4 windows tier-2 gap probe`, and **`DEV-160 raw
slot primitives under Miri`** — the job DEV-160's own entry names as its guard.

### CI failed twice first, and that is the most useful part of this record

A report showing only the final green run would hide the two defects that only CI could find. Both
were in this programme's own work.

```text
run 31350802609  sha 45a8bc5  FAILURE  windows-x64: `two_acquisitions_of_one_build_directory
                                       _cannot_overlap` -- 578 passed, 1 failed, and the 1 was
                                       this programme's own new test
run 31352368457  sha 1913b19  FAILURE  all three platforms: `stark fmt --check` on
                                       stark-net-resource-consumer
run 31353396547  sha 5967a42  SUCCESS
```

**The Windows failure was a real portability defect in the DEV-159 repair.** `remove_dir` on
Windows marks a directory for deletion and the name lingers until the last handle closes; during
that window `create_dir` returns `ERROR_ACCESS_DENIED` — `PermissionDenied`, not `AlreadyExists` —
and the acquire loop treated any other error kind as fatal. A tight acquire/release cycle failed on
one platform and nowhere else. No macOS run could have caught it. Repaired in `fda97c4`.

**The formatting failure was a gap in this programme's own discipline.** `stark fmt` is STARK's own
source formatter and is separate from `cargo fmt`, which had been clean throughout. Editing a
`.stark` file means running both. Repaired in `5967a42`.

## Final table

| DEV | Baseline status | Final status | Repair SHA | Evidence | Residual |
|---|---|---|---|---|---|
| 180 | OPEN, scheduled | RESOLVED | `1db9760` | `as3_receiver_materialization` 7/7; `interp::tests` 144/144 | none |
| 120 | OPEN, documented | CLOSED — reclassified as a documented limit | `c6e3669` | interpreter exit 2 (512 frames) vs native exit 134 (SIGABRT), both measured | native recursion dies by signal; `MAX_CALL_DEPTH` unchanged |
| 167 | OPEN, deferred | CLOSED — CE1 owner decision, documented non-promise | `c6e3669` | 2 decision-pinning tests in `dev_display_dispatch` | `to_string()` on a `Display` bound stays refused, by decision |
| 220 | *not yet found* | RESOLVED (new) | `9107fb2` | 5 three-engine cases, 2 negative controls; all 5 fail with the repair reverted | none |
| 157 | OPEN | RESOLVED | `004d1fc` | 7 three-engine cases; 6/7 fail with the representation removed | `loop {}` verified by building, not running |
| 168 | OPEN | RESOLVED | `70725ee` | 3 three-engine cases + a refusal control; all 3 fail with the arm removed | bounded-generic form → DEV-221 |
| 159 | OPEN, unsettled | RESOLVED | `8c5526e`, `fda97c4` | 73/240 failures before, 0/240 debug and 0/200 release after; unit control fails deterministically when neutered | stress numbers are from one macOS host |
| 160 | OPEN | **OPEN** — rustc leak sealed, capability half remains | `9a0557f` | 8-shape pre/post corpus diff; `dev160_call_site_thunk` 8/8; Miri job green | provenance is "may derive from", not "a live borrow reaches here" |
| 140 | OPEN | OPEN — assessed, deferred | — | layer audit; no consumer | needs a 4-layer feature addition |
| 141 | OPEN | OPEN — a `std-full` profile boundary | — | layer audit; no consumer | not a defect; owner reclassification suggested |
| 142 | OPEN | OPEN — assessed, deferred | — | layer audit; no consumer | needs C6.3e generated lifetimes |
| 143 | OPEN | OPEN — assessed, deferred | — | layer audit; no consumer | the most tractable of the six if pressure appears |
| 144 | OPEN | OPEN — assessed, deferred | — | layer audit; no consumer | one cursor type at a time |
| 145 | OPEN | OPEN — assessed, deferred | — | layer audit; no consumer | same 4-layer shape as DEV-140 |
| 165 | OPEN (population B) | RESOLVED | `1913b19`, `5967a42` | a pre-existing pinned control fired; 2.479s measured against a 2s deadline | `connect_no_timeout` remains the honest unbounded spelling |
| 221 | *unnumbered residual* | OPEN (registered) | — | reproduced: `[E0500]` on a bounded generic receiver | ergonomic; `x.fmt()` works |
| 098 | untracked | ACCEPTED-INDEFINITELY | `45a8bc5` | never a defect — a verifier-accepted MIR shape | none |

## §23 exit condition

```text
 1  every deviation reproduced or reclassified                     MET
 2  DEV-180 resolved                                               MET
 3  DEV-160 resolved OR its boundary enforced by STARK             PARTIAL
 4  DEV-157 and DEV-168 repaired or bounded                        MET (repaired)
 5  DEV-159 experimentally settled                                 MET
 6  DEV-140..145 individually assessed, not bulk-refactored        MET
 7  DEV-167 deliberate API disposition                             MET
 8  DEV-120 deliberate resource-limit disposition                  MET
 9  population regenerated from repository authority               MET
10  broad CI green at the final candidate                          MET
11  docs distinguish historical C10 from post-C10 repairs          MET
```

**Criterion 3 is PARTIAL and is recorded as partial rather than ticked.** DEV-160's `E0502` leak is
sealed for the demonstrated shape, so the boundary is now enforced by STARK. But those programs are
valid STARK and still do not build, which is the capability half — DEV-160b's cross-block
absorption, its own deferred work package under the owner ruling of 2026-08-03.

## Known residuals

1. **DEV-160's detection is a heuristic.** `borrow_provenance` answers "this value may derive from
   that slot", not "a live borrow of that slot reaches this call". Severing propagation on moves
   removed the false positive that mattered, and neither the corpus nor the package surface found
   another — **but absence of a counterexample is not proof of precision.** The precise mechanism is
   a backward walk of the reference's own def-use chain, which cross-block absorption needs anyway
   and which should replace this heuristic when it lands.
2. **DEV-159's stress numbers are from one macOS host.** The mechanism is filesystem atomicity
   rather than anything platform-specific, and CI covers all three Tier-1 platforms — but the
   73/240 and 0/240 figures are not claimed for other hosts.
3. **DEV-097's column-level claim was not re-verified.** Its reconciliation confirms the trap fires
   and the engines agree; the entry's specific complaint about two ends of one bounds check blaming
   different columns is not re-checked.
4. **DEV-093 and DEV-094 closed on recorded evidence, not a probe.** Neither is a language
   behaviour a test program can reach. Their headings say so.
5. **A CI coverage gap, found while fixing the formatting failure.** The qualification script runs
   `stark fmt --check` on each package and on *resource* consumers, but not on ordinary consumers,
   and not on packages absent from its case list. Three packages are unformatted today and CI
   cannot see it: `stark-fmt-consumer` (an ordinary consumer), `stark-io` and `stark-random` (not
   in the list). Verified pre-existing by re-running the check with the pre-change compiler. This is
   the same mechanism the script's own comment records for five earlier packages. Left untouched —
   out of this programme's scope, and worth its own change.

## Closing note

The programme's governing principle was to repair demonstrated defects in bounded packets and not
redesign adjacent architecture. Two decisions were escalated rather than taken (DEV-167's CE1;
DEV-140..145's deferral), one repair was implemented, measured, and **reverted** for over-refusing
shipping code before a corrected version landed (DEV-160), and one deviation was found only because
the position matrix §9.1 required was built by probe instead of read from its entry (DEV-220).

The endpoint §23 asks for is not zero open deviations. It is that every remaining one is repaired,
intentionally bounded, or classified with evidence, and that no serious semantic defect is hidden
inside a broad "supported subset" statement. **That condition is met.** Of the eight that remain,
exactly one — DEV-160's capability half — is reached by code anyone has written.
