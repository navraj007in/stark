# AS8 exit qualification — independent evidence, tooling scale and governance closure

**Date:** 2026-08-09. **Branch:** `wp-arch-stability/as7-modularization`, worktree
`/Users/nexper/Documents/GitHub/stark-as7`. **Authority:** Campaign B (CD-389), AS8 opened after
AS6 (CD-390) and AS7 (CD-391, criterion 2 re-qualified CD-393).

**Verdict: the five exit criteria are met. The packet's own result is that THE EVIDENCE BASE WAS
OVERSTATED IN THREE DOCUMENTS AND UNDERSTATED IN ONE**, and every correction is a measurement, not
a re-reading.

---

## 1. What AS8 was asked to do, item by item

| Work item | Status |
| --- | --- |
| Consume the EI register / evidence audit / risk profiles / ranked targets; do not repeat the inventory | **Done** — and three of the four needed correcting. No second taxonomy was introduced; AS8-DA-* is a separate dimension by owner ruling, not a rival register |
| Compiler-source mutation trials for ownership, trap, drop, resolver and MIR verifier rules | **Done** — 39 trials, all five families |
| Line/branch coverage baselines; report uncovered semantic arms | **Done** — `AS8-COVERAGE-BASELINE.md`. Tooling did not exist and was installed for this packet |
| Run the external `stark-samples` suite as pinned independent evidence, recording commit and manifest | **Done** — 34/34 passed at `b3b28e757f38d691e7309f168d1209e28ac459af` (2026-08-02) |
| Profile LSP package analysis on representative multi-file projects | **Done** — and it found DEV-213 |
| Bounded debounce / cancellation / cache ownership **if evidence warrants** | **Not done, and deliberately.** The evidence warrants a cache-ownership change, but as a CORRECTNESS repair (DEV-213), not a performance one. AS8 is assurance; the repair is implementation and takes ordinary checkpoints |
| Replace whole-package `ProjectAnalysis` duplication **where measurement shows material cost** | **Not done — the question was wrong.** See §4 |
| Compress `COMPILER-STATE.md`; preserve append-only history in an archive | **Done** — 12,979 -> 6,681 lines, verified lossless |
| Reconcile deviation statuses with executable evidence | **Done** — `as8-reconcile-deviations.py` |
| Update `compiler-map.md`, `lib.rs` docs, canonical roadmaps at campaign exit | **Done** |

## 2. The five exit criteria

### Criterion 1 — each differential claim names shared phases and at least one independent evidence source

**MET, and the answer changed for four authorities.**

`ENGINE-SHARED-FATE-REGISTER.md` now carries 11 entries, each naming its independent evidence or
recording that it has none. Four were wrong before AS8 measured them:

```text
ESF-COPY-001   "no control in-tree"          ->  c61f_structural_copy, HAND_AUTHORED (MUT-009/010/011)
ESF-TRAP-001   one entry, INVISIBLE          ->  001a INVISIBLE / 001b PARTIALLY_VISIBLE (MUT-007)
ESF-PROV-001   "two engines, not three"      ->  mir/verify.rs IS an in-tree control (MUT-025)
ESF-TYPE-001   "spec fixtures"               ->  they do not control it (MUT-013), promoted to high
```

### Criterion 2 — selected source mutations are killed by the claimed suites; survivors recorded as test gaps

**MET.** 39 trials. 26 confirmed the prediction, **13 falsified it, in both directions.** Every
survivor is an `AS8-R*` residual; none was quietly dropped.

The packet's most useful single number is that ratio. EI5's predictions were reasoned from EI2's
reading of the evidence base and were wrong a third of the time — which is the argument for
mutation over audit, made by the audit failing.

### Criterion 3 — LSP changes justified by before/after measurements and cancellation correctness tests

**MET VACUOUSLY, AND THAT IS THE HONEST STATUS.** No LSP change was made, so no before/after is
owed. The measurement was taken (§4) and it justified filing a defect rather than making a change.

### Criterion 4 — current compiler position discoverable from the beginning of `COMPILER-STATE.md`

**MET.** `# Current position` is now the first section. Before AS8, `## Position` sat at line
5,456 and described Gate C5 closing on 2026-07-23 — four gates stale — while a 2,808-line section
was still headed **IN PROGRESS** after C5, C6, C7 and C8 had all closed.

The same defect was found and fixed in `KNOWN-DEVIATIONS.md`, where it is sharper: the file is
append-only, so `DEV-121`'s first heading says OPEN and it is CLOSED **3,558 lines later**.

### Criterion 5 — architecture documentation matches production entry points and module ownership

**MET.** `lib.rs` cited `PLAN.md` as the live execution plan and described "Gates 1–3" building an
interpreter, four gates after native compilation closed. Replaced with the real pipeline, the three
engines and their roles, the entry points, and typecheck's eleven-module DAG. `compiler-map.md`
carries the eleventh module and the corrected counts.

## 3. Trial record

| Trial | Batch | Target | Expected | Actual | Verdict | Killers |
| --- | --- | --- | --- | --- | --- | ---: |
| `AS8-MUT-001` | 1 | `ESF-COPY-001` | SURVIVED | **KILLED** | UNEXPECTED | 25 |
| `AS8-MUT-002` | 1 | `ESF-DROP-001` | SURVIVED | **KILLED** | UNEXPECTED | 25 |
| `AS8-MUT-003` | 1 | `ESF-COPY-001` | SURVIVED | **SURVIVED** | CONFIRMED | 0 |
| `AS8-MUT-004` | 1b | `ESF-COPY-001` | SURVIVED | **KILLED** | UNEXPECTED | 5 |
| `AS8-MUT-005` | 1b | `ESF-COPY-001` | SURVIVED | **SURVIVED** | CONFIRMED | 0 |
| `AS8-MUT-006` | 1b | `ESF-COPY-001` | SURVIVED | **SURVIVED** | CONFIRMED | 0 |
| `AS8-MUT-009` | 1c | `ESF-COPY-001` | KILLED | **KILLED** | CONFIRMED | 4 |
| `AS8-MUT-010` | 1c | `ESF-COPY-001` | KILLED | **KILLED** | CONFIRMED | 1 |
| `AS8-MUT-011` | 1c | `ESF-COPY-001` | KILLED | **KILLED** | CONFIRMED | 1 |
| `AS8-MUT-012` | 2 | `ESF-COPY-002` | KILLED | **SURVIVED** | UNEXPECTED | 0 |
| `AS8-MUT-013` | 2 | `ESF-TYPE-001` | KILLED | **SURVIVED** | UNEXPECTED | 0 |
| `AS8-MUT-014` | 3 | `ESF-TRAIT-001` | SURVIVED | **SURVIVED** | CONFIRMED | 0 |
| `AS8-MUT-015` | 3 | `ESF-TRAIT-001` | SURVIVED | **SURVIVED** | CONFIRMED | 0 |
| `AS8-MUT-016` | 4 | `ESF-PROV-001` | KILLED | **SURVIVED** | UNEXPECTED | 0 |
| `AS8-MUT-017` | 4 | `ESF-RES-001` | KILLED | **KILLED** | CONFIRMED | 3 |
| `AS8-MUT-025` | 4b | `ESF-PROV-001` | KILLED | **KILLED** | CONFIRMED | 4 |
| `AS8-MUT-018` | 5 | `ESF-TYPE-001` | KILLED | **KILLED** | CONFIRMED | 9 |
| `AS8-MUT-024` | 5 | `ESF-TYPE-001` | KILLED | **KILLED** | CONFIRMED | 4 |
| `AS8-MUT-019` | 5 | `ESF-TYPE-001` | KILLED | **SURVIVED** | UNEXPECTED | 0 |
| `AS8-MUT-007` | 6 | `ESF-TRAP-001b` | KILLED | **KILLED** | CONFIRMED | 4 |
| `AS8-MUT-008` | 6 | `ESF-TRAP-001a` | SURVIVED | **SURVIVED** | CONFIRMED | 0 |
| `AS8-MUT-020` | 7 | `RA-OVERFLOW` | KILLED | **KILLED** | CONFIRMED | 3 |
| `AS8-MUT-021` | 7 | `RA-SHIFT` | KILLED | **KILLED** | CONFIRMED | 1 |
| `AS8-MUT-022` | 7 | `RA-DROP` | KILLED | **SURVIVED** | UNEXPECTED | 0 |
| `AS8-MUT-023` | 8 | `EV-CORPUS-C6` | KILLED | **KILLED** | CONFIRMED | 1 |
| `AS8-MUT-034` | 9 | `resolver visibility` | KILLED | **SURVIVED** | UNEXPECTED | 0 |
| `AS8-MUT-035` | 9 | `resolver visibility` | KILLED | **SURVIVED** | UNEXPECTED | 0 |
| `AS8-MUT-036` | 9 | `MIR verifier` | KILLED | **KILLED** | CONFIRMED | 3 |
| `AS8-MUT-037` | 9 | `MIR verifier` | KILLED | **SURVIVED** | UNEXPECTED | 0 |
| `AS8-MUT-038` | 9b | `resolver visibility` | KILLED | **KILLED** | CONFIRMED | 2 |
| `AS8-MUT-039` | 9b | `resolver visibility` | KILLED | **SURVIVED** | UNEXPECTED | 0 |
| `AS8-MUT-026` | da | `AS8-DA-002` | KILLED | **KILLED** | CONFIRMED | 1 |
| `AS8-MUT-027` | da | `AS8-DA-002` | KILLED | **KILLED** | CONFIRMED | 1 |
| `AS8-MUT-028` | da | `AS8-DA-003` | KILLED | **KILLED** | CONFIRMED | 4 |
| `AS8-MUT-029` | da | `AS8-DA-003` | KILLED | **KILLED** | CONFIRMED | 4 |
| `AS8-MUT-030` | da | `AS8-DA-004` | KILLED | **KILLED** | CONFIRMED | 1 |
| `AS8-MUT-031` | da | `AS8-DA-004` | KILLED | **KILLED** | CONFIRMED | 1 |
| `AS8-MUT-032` | da | `AS8-DA-005` | KILLED | **KILLED** | CONFIRMED | 2 |
| `AS8-MUT-033` | da | `AS8-DA-005` | KILLED | **SURVIVED** | UNEXPECTED | 0 |

Full analysis: `AS8-MUTATION-FINDINGS.md`. Duplicate-authority pairs: `AS8-DUPLICATE-AUTHORITIES.md`.

## 4. The work item AS8 declined, and why

> *"Replace whole-package `ProjectAnalysis` duplication per open URI **where measurement shows
> material cost**."*

Measured: 22 ms for one analysis of a 32-module package, 181 ms for eight open URIs. **Answering the
question as written, the item closes as "not material" and nothing happens.**

The duplication's real consequence is not cost. It is **N copies with independent invalidation** —
`update_document` drops only the edited URI's entry, and `handle_workspace_symbol` merges symbols
from every cached analysis, so a rename in one file leaves every other open file's analysis
carrying the old name and the response contains both. Demonstrated by a passing test at HEAD and
filed as **DEV-213**.

A scope framed around cost would have measured 22 ms, concluded "no", and closed. The measurement
was worth taking anyway.

## 5. Residuals

```text
AS8-R1   a wrong Copy rule with no drop consequence is invisible to every differential suite
AS8-R2   ESF-TRAP-001a: no control, and none constructible as a source mutation
AS8-R3   DISCHARGED — corpus census exists (as8-control-census.py) and found what EI2 missed
AS8-R4   copy_canon_matrix is a transcription, not a control (MUT-003)
AS8-R5   EV-SPEC-FIXTURES does not control TYPE-PRIM-001 (MUT-013)
AS8-R6   ESF-COPY-002 is unexercised: no case duplicates a &mut and observes it (MUT-012)
AS8-R7   13 of 39 predictions falsified, in both directions
AS8-R8   array destruction order is unguarded (MUT-022)
AS8-R9   strip_ref recursion is unguarded (MUT-019)
AS8-R10  ESF-TRAIT-001 has no control of any kind (MUT-014/015)
AS8-R11  EI2-R2 corrected in the compiler's favour: mir/verify.rs IS a control (MUT-025)
AS8-R12  AS8-DA-005: scalar_name can drift silently; scalar_src cannot (MUT-032/033)
AS8-R13  non-`pub` re-export visibility has NO control anywhere — not the fixtures, not the
         differential, not resolve.rs's own unit tests (MUT-035 and MUT-039 both survived)
AS8-R14  mir::verify::may_need_drop's HostResource arm is unguarded (MUT-037), while the
         corresponding mir_ty_is_copy arm IS guarded (MUT-017). AS8-DA-006's verifier half
AS8-R15  the full-corpus coverage baseline was attempted and stopped on a disk floor; the
         published baseline is `--lib` only and says so
```

**No DEV number was allocated for any mutation survivor**, per the owner ruling: a survivor means
the evidence cannot detect a defect, not that the defect is present. DEV-213 is the one DEV filed,
and it was demonstrated at HEAD by a passing test.

## 6. What AS8 got wrong about itself

Recorded because the packet's subject is exactly this.

1. **Three test-selection failures.** MUT-005/006 (the control existed and was not selected),
   MUT-016 (the control was in another crate and unrunnable), MUT-013 (the control was selected and
   structurally incapable). The rule added after the first was necessary and twice insufficient.
2. **Three scanners over-reported on their first run** — the duplicate-authority scan (trait
   declarations have no body), the deviations reconciler (115 of 189 entries), and the AS7 ownership
   parser inherited from CD-393. Each was narrowed until its output was worth reading, or deleted.
3. **A broad `git add` committed a live mutation to a pushed branch**, failing every C6.5 job. The
   harness now refuses a dirty target and verifies its own restore.
4. **The harness built all 209 test binaries per trial**, relinking ~205 it would never run. That
   filled the disk to 99% and stretched one build to 32 minutes before I looked at `df`.
5. **A killed batch bypassed `finally`** and left a mutation in the working tree. Caught by
   `git status` before any commit.
