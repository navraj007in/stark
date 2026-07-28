# WP-C6.5 closure packet

**Qualified commit:** `e3ef603`. All 15 CI jobs green, including both `C6.4 tier-1 qualification`
targets, `C6.4 tier-1 agreement`, and `C6.5 corpus tier-1 agreement`.
**Recommended status:** `CLOSED`.

**The claim, stated exactly.** The admitted **executable** surface agrees across HIR, MIR and native
debug on both Tier-1 targets. This packet does **not** claim that every specified limit is enforced:
DEV-118 is open, and the `T: Hash + Eq` bound is not enforced for `HashMap` or `HashSet`.

**Tier-1, verified independently of CI's own verdict.** 160/160 cases on `aarch64-apple-darwin` and
`x86_64-unknown-linux-gnu`, same commit, corpus 1.3.0, identical `manifest_sha256` and
`generator_sha256`. Every per-case observation hash is byte-identical between targets (0 differing),
every case ran all three engines, and no case had a non-PASS engine. `full_evidence: true`,
`skipped_count: 0`, `result: PASS` on both.

## 1. What changed, in one page

C6.5 began this batch with thirteen §17 findings, three of them blocking. All thirteen are closed.
The work that mattered was rarely the stated finding:

| Finding | Stated as | What it actually was |
| --- | --- | --- |
| R-01 | two trap categories missing | a **sampling budget silently deleting coverage the corpus still reported** |
| R-02 | 23 suites use private comparators | those comparators asserted `status == 0` per engine **separately**, which is not a comparison |
| R-03 | 7 of 15 fields have controls | nothing **enumerated** the field set, so the gap was invisible |
| R-07 | 36 of 136 rows have evidence | **three fabrication classes**: 69 invented rule IDs, 36 false template arrows, 13 test names that exist nowhere |
| R-08 | retention untested | the §11.11 layout had **never been produced at all** |
| R-12 | counts, not identities | the count was a **literal `0`**, not a measurement |

## 2. Defects found by doing the work

Four were found and fixed during C6.5 (DEV-111 entry termination, DEV-112 `()`/`Unit`, DEV-113
package trap provenance, DEV-114 diamond-graph symbol nondeterminism). Two are open and are the
reason this packet does not recommend `CLOSED`:

**DEV-116 — `HashSet` refused at lowering. FIXED (CD-176/CD-177).** Normative in `std-full`, ran in
the oracle, refused by MIR at TYPE lowering. Now `StarkMap<T, ()>` in every engine, so uniqueness
goes through the comparator dispatch STD-HASH-001 already governs. All eight admitted methods lower,
including `iter` — closed in two parts because the data operations alone left the surface only
partly executable, which V19's per-TYPE row could not express.

**DEV-117 — MIR refuses reinitialisation. FIXED (CD-175).** The move-out was drop elaboration under a
drop flag, not a user read; the exemption the pass already documented for `Terminator::Drop` had
never covered the move-to-temp shape a reassignment uses.

```stark
fn take(s: String) -> UInt64 { s.len() }
fn main() {
    let mut slot: String = String::from("ab");
    assert_eq(take(slot), 2u64);
    slot = String::from("cde");
    assert_eq(take(slot), 3u64);
}
```

Both were found because a coverage row had no case, and both are now permanent regressions retained
under §11.11. Neither was a test problem, and neither could be quarantined: a divergence between the
semantic authority and MIR is precisely what this work package exists to detect.

**DEV-115 — `str::bytes` half-landed. FIXED.** Found by reviewing another author's diff, not by this
corpus — see limitation 4 below.

**DEV-118 — the `Hash` bound is not enforced. OPEN, CARRIED, NON-BLOCKING.** An element with `Eq`
and no `Hash` compiles and runs; the identical program over a `HashMap` key is equally accepted, so
it is pre-existing and shared rather than introduced here. It is an **enforcement omission, not a
differential defect**: all three engines accept it identically, so it cannot threaten the agreement
claim. It is unobservable today only because CE4 chose an `Eq`-only scan that never consults `Hash`
in storage, and it becomes real the moment any engine narrows candidates by hash.
**Owner: WP-C6.3**, which holds collection bound enforcement.
`dev116_hashset::dev118_the_hash_bound_is_not_enforced_for_either_collection` pins the current
behaviour so the day it changes, it fails there.

## 3. §22 checklist

Every box below was checked against the tree, not against intent.

### 22.1 Hand-written coverage — **MET**
Matrix covers every executable Core category; all 136 rows carry one machine-checked disposition
(0 UNATTRIBUTED); 10 of 10 admitted trap categories have exact witnesses; package and relocation
paths are covered by M08/M09 and the `pkg__*` cases; non-Core classifications state their reason
(P09 has no `ref`/`mut` pattern form in Core v1's grammar at all). O14 and V19 were the last two
BLOCKED rows and both are now covered, by fixing the defects rather than by reclassifying them.

### 22.2 Generator — **MET**
Explicit seed, checked-in version, deterministic content-addressed IDs, 74 generated cases (floor
64), 15 templates (floor 10), bounded size and trip count with the bound enforced, no network,
byte-reproducible, retention and reduction paths exercised for real.

### 22.3 Replay — **MET**
153/153 through every required engine, no refusal among manifest cases, exact observation comparison
over the full §39 shape, two runs identical.

### 22.4 Metamorphic — **MET**
M01–M12, 24 groups / 48 members, per-engine and cross-engine equality, kind-aware identity-transform
protection, no pair hidden or redesigned.

### 22.5 Mutation — **MET, exceeded**
MU01–MU16 plus MU17–MU23; 15 of 15 comparator fields; witnesses pass unmodified first; the intended
field is named; evidence written per mutation.

### 22.6 Manifest and hashes — **MET**
Corpus 1.0.0; manifest, lock, per-source and generator hashes complete; no unlisted files (with
`cases/retained/DEV-*/` accounted for by the stricter retention suite); regeneration byte-clean.

### 22.7 Tier-1 — **MET at `e3ef603`**
Both targets, same commit, 160/160, identical per-case observation hashes, no skips or quarantines.
The records at `8a23772` are **superseded, not false** — the qualified path changed underneath them,
and they are replaced rather than amended.

### 22.8 Governance — **MET on the items in this work package's gift**
Defects recorded with owners; DEV-117 retained under §11.11; leases recorded in advance; CE
decisions recorded (CD-150's CE3 discharged); WP-C2.12 closed by owner directive. `COMPILER-STATE.md`
and the integration ledger are updated in the closing commit.

## 4. What a reviewer should distrust

Three things in this work package are weaker than they look, and saying so is cheaper than having
them found:

1. **`c63c_iterators` and `c63e_formatting` take the HIR oracle's own output as their expectation.**
   Engine agreement is real; an independent pin is not. Their headers say so.
2. **Four mutation controls are insertion-only** (`stderr_bytes`, `drop_log_before_trap`,
   `trap exit_status`, `completion versus trap`) because the language makes the field empty or
   constant in every conformant observation. They prove the comparator reads the field, not that it
   distinguishes two real values.
3. **Attribution is per-row judgement.** 32 rows cite a comparator-backed test chosen by reading. The
   validator proves the identity exists and is comparator-backed; it cannot prove the test exercises
   the row. Automated matching was tried and produced confidently wrong answers.

4. **The coverage matrix is per-TYPE, not per-METHOD, and DEV-115 proves it matters.** V07 (`String`)
   and V08 (`str`) are both attributed to a generated case that exercises `String::from` and
   `push_str` — and no corpus case anywhere calls `.bytes()`. So both rows read as covered while a
   `str` method diverged across engines: it typechecked, lowered, verified, emitted natively, and
   died in the MIR interpreter. It was caught by a review of another author's diff, not by this
   corpus.

   "All 136 rows carry a machine-checked disposition" therefore certifies something narrower than it
   sounds. It means every row has real, verified evidence behind it. It does NOT mean every API
   surface reachable from that row is exercised, and a per-method defect can sit behind a covered
   row indefinitely. Closing that would mean rows at method granularity for the stdlib surface,
   which is a larger corpus than C6.5 scoped.
