# Gate C5–C7 closed detail — archived from COMPILER-STATE.md

**Archived 2026-08-09 under AS8 exit criterion 4** ("current compiler position is discoverable
from the beginning of `COMPILER-STATE.md` without reconstructing chronology") and the charter's
§2.4 rule to compress **closed gate detail** into summaries.

**Nothing is deleted and nothing is rewritten.** Every section below is verbatim as it stood at
the moment of archival; `COMPILER-STATE.md` carries a one-paragraph summary and a pointer here in
each section's place. The decision log and every `CD-###` record stay in the state file — the
append-only log is not touched by this compression.

The largest section here, `C5.3e — target-layout manifest`, was still headed **IN PROGRESS** while
Gates C5, C6, C7 and C8 had all since closed. A 2,808-line block describing itself as in progress
is the clearest single symptom of the problem this compression exists to fix.

---

## WP-C7.9 — three-engine adversarial conformance correction — **CLOSED** (CD-275…CD-278, 2026-07-31)

**Qualified at `144ceee` on `main`: 18 of 18 CI jobs green across linux-x64, macos-arm64 and
windows-x64.** Local: workspace 2047 passed / 0 failed; corpus replay 170 cases over four engine
configurations; subprocess robustness 6/6. The claim this supports, and its limits, are in
`WP-C7.9-CLOSURE.md` §7 — it is deliberately narrower than "every type-correct program agrees".

Follow-on CDs from the qualification phase itself: **CD-276** (a guard test read line endings, so it
was green on two platforms and red on the third) and **CD-277** (`c785_time_closeout` asserted
`reading > 0` to mean "the slot was written", while `0` is a legitimate clock reading — a latent
unsoundness, fixed at the cause with a sentinel the provider cannot produce). **CD-278** closed the
`chars()` scalar/byte confusion that a feature-example suite found afterwards.


**Corrective work on the tree Gate C7 closed over.** CD-274 closed C7; this landed after it, from
two adversarial review passes. CD-274's ruling stands as written and is not amended — but three of
the defects below were **live cross-engine divergences at the moment C7 closed**, so the claim this
work supports is stated separately rather than folded into C7's.

### The three divergences the reviews found

| | |
| --- | --- |
| `MIN % -1` | **Did not trap at all** in MIR or native: both evaluate on an `i128` carrier and range-filter, and the remainder `0` is in range. The program COMPLETED with a value where NUM-INT-DIV-001 requires a trap, while the oracle trapped. |
| `MIN / -1` | Trapped in all three engines with the **wrong identity** — `DivideByZero`, because `Div`/`Rem` carried one static category per operator regardless of cause. |
| borrowed payload binding | The oracle bound by clone, so a binding used AS a reference failed there and worked in MIR and native. CD-267 pinned and escalated it; Packet C closed it. |

### Three more this work package found itself

1. **Compound assignment skipped the range check entirely.** `acc /= -1` on `Int32::MIN` completed
   in the oracle, storing `2147483648` in an `Int32`. `eval_binary` range-checks against the type of
   the expression it is handed, and the compound path handed it the ASSIGNMENT — type `Unit`, no
   width, so the check passed vacuously. No maintained case had ever overflowed through `/=`.
2. **A function's generics were not in scope for its own signature.** They were installed after the
   return type was converted, so `fn build<T: Hash + Eq>() -> HashMap<T, Int32>` rejected itself
   once anything checked bounds during conversion.
3. **Interpreter recursion overflowed the host stack at ~100 STARK frames** — a depth ordinary
   programs reach. A depth cap alone could not fix it; execution now runs on a stack sized for the
   cap, and the cap reports exhaustion before the host runs out.

### What changed

- **MIR amendment A13**: checked evaluation may override the terminator's category when an
  operation fails for a different normative cause. `MIR_RUNTIME_SURFACE` `0.1-A10` → `0.1-A13`
  (fourteen stderr output operations); `MIR_VERSION` stays `0.3` — A13 adds no shape.
- **`E0105`** allocated: iteration forms this implementation does not support. Nine
  accepted-but-unlowerable surfaces now refused by the front end instead of being accepted and
  refused later by lowering.
- **Core-trait implementations are checked**, against one canonical contract table. A `CoreTrait`
  has no declaration item, so nothing had ever compared `impl Ord for T` against anything.
- **`eprint`/`eprintln` reach every engine**, and the channel is compared — including before a trap,
  separated from the runtime's own diagnostic by a per-run nonce.
- **Trap identity is structural**: every language trap states its category where it is raised, the
  prose normaliser is gone, and a guard test fails if phrase-matching returns.
- **`FailureClass`** replaces `is_trap`: language trap / entry rejection / host resource /
  interpreter invariant. Call-depth exhaustion is the third, never a trap (`LIMIT-RESOURCE-001`).
- Corpus **1.4.0 → 1.5.0** (nine cases: eight `MIN op -1` sentinels, one writing both streams before
  trapping — the corpus had no case with program stderr, because no engine could perform it).

### Carried

**DEV-120** (native call-depth exhaustion, bounded host limitation, ruling D4); provider-backed
capabilities stay verifier/ABI/native qualified (D5); the nine refused iterator surfaces (D3); two
CE4/CD-132-governed refusal points recorded with guard tests; `eprint`'s `&str`-only signature.

Full account: `STARKLANG/docs/compiler/work-packages/WP-C7.9-CLOSURE.md`.


## Gate C6 — CLOSURE (CD-183)

**Gate C6 closes with a qualified native executable subset. Of 87 audited normative
standard-library methods, 59 have verified executable invocations and 28 are explicitly refused or
excluded; none are unclassified. The audit establishes invocation support, not exhaustive validity
across every usage shape. Usage interactions are qualified through the differential corpus and
focused lifecycle regressions, including borrowed-iterator cleanup introduced by DEV-119. No claim
of full Core or standard-library native conformance is made.**

### The audit's limit, stated because the number reads stronger than it is

`59 of 87` means: each of those 59 has **at least one valid invocation** that passes the front end,
lowers to MIR, and verifies. It does **not** mean every valid use of them works. DEV-119 is the
demonstration — `HashMap::keys`, `HashSet::iter` and `Vec::iter` all passed the invocation audit
while an ordinary post-loop mutation failed native compilation. Fixed (CD-182), permanently covered
by `dev119_iterator_lifetime.rs`, and generalised as the risk-based follow-on
`WP-C7-Usage-Shape-Qualification`.

### Exclusions and carried work, all explicit

| | |
| --- | --- |
| `File` (5) | EXCLUDED — needs a host/provider contract, filesystem error semantics, and a way to compare environmental observations across engines. Deferred to the I/O gate. |
| `Random` (4) | EXCLUDED pending a normative PRNG algorithm and cross-engine sequence contract. **Not** excluded as "nondeterministic": a seeded generator is reproducible. |
| `String` extended (10) | CARRIED → `WP-C7-String-Surface` |
| `HashMap` remainder (4) | CARRIED → `WP-C7-HashMap-Completion` (`with_capacity`, `get_mut`, `values`, `iter`) |
| `Vec` remainder (3) | CARRIED → `WP-C7-Vec-Completion` |
| DEV-118 | **CLOSED by WP-C7.9 Packet I (CD-275)** — the `T: Hash + Eq` bound is enforced at type instantiation for both collections. It was an enforcement omission all three engines shared, which is why no differential could see it. |

### What C6 actually established

Native execution preserves Core ownership, Drop, failure and library semantics across HIR, MIR and
native debug, on two Tier-1 targets, at one commit, with identical per-case observation hashes —
rather than merely running scalar examples. Seven defects were found and fixed in the process
(DEV-111 … DEV-117, DEV-119), every one by closing a coverage gap rather than by inspection.

Updated: 2026-07-25 — **Gate C5 CLOSED (CD-077). Gate C6 OPEN: entry plan APPROVED (CD-079),
WP-C6.0 contract freeze CLOSED (CD-078), **WP-C6.1a–e (ownership and Drop parity, Track A) CLOSED
(CD-080…CD-084) and **WP-C6.1f CLOSED (CD-099)** (general reference storage — the C5 deferral the C6
entry plan never assigned), so **WP-C6.1 as a whole is CLOSED**; and WP-C6.2a (canonical callable identity — native method/trait/operator dispatch)
CLOSED (CD-086). WP-C6.2b PARTIAL (CD-087): DEV-102 closed, §18 matrix probed, **six findings
**C6.2b matrix CLEARED: F1/F5/F2/F6 CLOSED (CD-102/103/104/105)**; F3 → WP-C6.1f (closed); F4 split (parser half open, selection is Track B).**
**WP-C6.2c (associated types) CLOSED (CD-106): §19 matrix proven three-engine — `Self::Item`, `T::Item`
via explicit binding and inferred-from-argument (deferred projection obligations + program-wide
`assoc_projections`), cross-package projection (DEV-101 span provenance), Drop-bearing assoc types.**
**WP-C6.2d (operator/CoreTrait semantics) CLOSED (CD-107): §20 matrix proven — user impls invoked
natively (adversarial Eq/Ord/Clone/Default/From), no Rust-derive substitution, missing impls rejected
(E0500/E0302); Display/Hash dispatch in HIR+MIR (native output/collections → C6.3); `.into()` blanket
and `Default::default()` inference deferred (DEV-103/104).**
**WP-C6.2e (deterministic identity) CLOSED (CD-108) → WP-C6.2 as a WHOLE CLOSED: canonical symbols
render nominals by content path (`struct#liba::A`), not the order-dependent `ItemId` index — stable
across clean rebuild, relocation, and dependency-declaration reorder (§21/§22 met).**
**WP-C6.3 OPENED — native Core runtime (Track C). WP-C6.3a PARTIAL (CD-109/110): the runtime-call
bridge (`Callee::Runtime`) + String/str value + str-output + Char surface land three-engine (owned
String construction/query/mutation/clone/return; `println`/`print` of str & char incl. Unicode;
`push`/`pop` char) with native stdout-byte checks; `stark-runtime/src/string.rs`. **The Option-return
bridge (CD-110) — wrapping a runtime Rust `Option` into the generated Option enum — is the mechanism
every future collection accessor reuses.** Owned-`String` `==`/`<` and stored interior `&str` are
now NATIVE (unblocked by C6.1g-c, promoted CD-116). C6.3a remaining: `chars()` iteration (→ C6.3c),
slicing views (→ C6.3b), cross-package String.
**WP-C6.3b COMPLETE (CD-111 + CD-131): native Vec/Box VALUE surface (new/push/pop/len/is_empty/clear/
return, Box new/into_inner) three-engine, plus the SLOT BUFFER-RECLAIM FIX — `drop_with` now runs
`ManuallyDrop::drop` after the glue, freeing every owning value's allocation (a latent leak).
`Vec<String>`-style pushes are NATIVE (unblocked by C6.1g-c, promoted CD-116). CD-131 added the
deferred half: TRAPPING `v[i]`/`remove` with the USER's source location (**DEV-107 CLOSED** — the
terminator already carried a `SourceInfo`; no MIR change was needed), CHECKED `get`/`get_mut`
(`Option<&T>`, never traps), and SLICE VIEWS (`MirTy::Slice(T)` → `[T]`, `SliceNew`/`SliceNewMut`/
`SliceLen`/`SliceIsEmpty`, bounds SIGNED so a negative bound traps rather than wrapping). Remaining:
`VecReplace` (no method surface reaches it), Vec/Box of user-destructor elements (refused by design).**
Remaining C6: **WP-C6.1 CLOSED (CD-099)**. **WP-C6.1g-a LANDED (CD-100): structural Copy
(OWN-COPY-001 amended) + borrow-carrying nominals in locals.** **WP-C6.1g-c CLOSED (CD-112): dispatch-loop
linearisation — acyclic bodies emit as nested labelled blocks so a cross-block borrow is seen
once-through; the borrow-through-return refusal is lifted (`Option<&P>` returns build). This also
unblocked owned-`String` comparison, stored interior `&str`, and `Vec<String>`-style pushes.**
**GENERALISED by CD-127: emission is now STRUCTURED for cyclic bodies too (`break 'bbT` for forward
edges, `continue 'loopH` for back edges), so borrows flow-analyse INSIDE loops — previously loops had
no borrow precision at all, since the `match __bb` dispatch let rustc assume any block follows any
block. The dispatch loop survives only as the fallback for an irreducible CFG. CD-127 also retired
the LAST C6.1f refusal (CD-128): a slot-backed MOVE borrow-carrying nominal builds and runs, so
`refuse_borrow_carrying_nominals` is deleted and no reference shape is refused pre-rustc any more.**
Gate-C6 dependencies: `WP-C6.1g-b`
(return-source lifetime precision), and C6.3 (`Box`/`Vec`/slice, Track C).
**WP-C6.3c CLOSED (CD-128/129/130, owner ruling 2026-07-26): native ITERATORS, on a native-parity
basis with exclusions named. CLOSED WITH EVIDENCE (three-engine): range, array (order), user
`Iterator` impl, shared `Vec` iteration, early termination, empty iteration, and `String`/`str`
character iteration — the cursor forms via `stark_runtime::vec::VecIter` / `string::CharsIter`.
EXCLUDED as absent LANGUAGE features: slice iteration and `iter_mut`. EXCLUDED as pre-MIR CAPABILITY
gaps: `map`/`filter`, `count`/`collect`, by-value `Vec` iteration — HIR-only, so neither MIR nor
native can represent them and no native divergence exists for this gate. Those are recorded as a
bounded follow-on (`starkc/docs/WP-ITER-LOWERING-PROPOSAL.md`, PROPOSED — needs owner approval and a
roadmap slot) and pinned by four PERMANENT boundary tests. `HashMap`/`HashSet` iteration lands with
C6.3d.**
**WP-C6.3e PARTIAL (CD-113…123): native OUTPUT + formatting — primitives (ints/bool/Float64 via a
shared `stark_runtime::format`, interp delegates, no drift), user `Display` dispatch (clears the
C6.2d Display deferral), `panic(msg)` text, and COMPOSITE Display (tuple/array + `Option`/`Result`,
`Vec` via a runtime loop, owned `String`/`str` elements, and nested user `Display` in tuple/array —
recursively — now native AND in MIR, was HIR-only; via a print-sequence lowering, no runtime-surface
change). Owner decision (CD-123): language `Display` RECURSES — a user nominal at any depth runs its
own `fmt`, not the aggregate debug form; the interp oracle was fixed to match. Its observable contracts
(A sequencing / B partial-output-on-trap / C destructor-timing) are recorded (CD-120) and Contract C is
load-bearing (the owned Vec/composite is dropped after its render); the native trap ABI flushes stdout
before abort so a mid-render trap's prefix matches the interpreters. `three_engine_differential`
compares real stdout (`NATIVE_STDOUT_SUPPORTED = true`). `Option`/`Result` of a `String` or a user
`Display` nominal now render three-engine — the backend's trailing variant-field BORROW (CD-126) fixed
the enum-payload limit (E0716). Nested user `Display` inside a `Vec` also renders
three-engine — CD-127's structured emission gave loops borrow precision (E0502 gone). Bounded/refused
AT LOWERING (deterministic): arrays > 64 (unroll cap), and a droppable composite carrying a borrow
(generated lifetimes). **`Float32` is no longer refused anywhere — DEV-105 is CLOSED (CD-138) by the
approved CE3 `PrintFloat32`/`PrintlnFloat32` at `MIR_RUNTIME_SURFACE` 0.1-A9, which carries the
DECLARED width in the operation's identity; scalar and every composite context render three-engine.** CD-135/136 added `Vec` of OWNING elements —
`Vec<String>`, and aggregates one level down (`Vec<(String, Int32)>`, `Vec<[String; 2]>`,
`Vec<Option<String>>`, `Vec<Result<String, _>>`) — read by REFERENCE rather than by copy.
**DEV-106 (trap-message three-engine parity) and DEV-107 (native `v[i]` OOB provenance) are both
CLOSED** (CD-136, CD-131). **C6.3e is CLOSED (CD-142).** DEV-108 CLOSED (CD-138: FIXED by a
loop-aware block order, not refused — the payload type was never the cause); DEV-105 CLOSED
(CD-138); DEV-110 CLOSED (CD-139); DEV-109 CLOSED (CD-140); confirmed by a full three-platform run
(CD-142). Historical note on the two `Float32` VALUE-semantics defects, both found by DEV-105's own
evidence — DEV-109 (`Float32` arithmetic is
carried in f64 and rounded only at display, so casts and overflow observe the wrong precision) and
DEV-110 (ESCALATED: NUM-FLOAT-OP-001 says float division by zero does NOT trap, recorded owner
decision CD-006 says it does; HIR follows the spec and MIR follows CD-006). DEFERRED to a future decision (CD-125): composite `Box`
elements — `Box<T>` is not a Display type today (typechecker E0500) and making it one is a semantics
choice, not a lowering slice. ESCALATED (CD-136): whether a `HashMap`/bare struct renders under
`Display` at all, and in what form — CE-shaped, currently E0500 in the front end but with latent
HIR-only renderings that would diverge the day either is admitted.**
**WP-C6.3d CLOSED by amendment (CD-132/133/134): native `HashMap` on the CE4 insertion-ordered
representation, with identity by the key type's lawful `Eq` reaching MIR and the backend through one
shared `TypeContext::eq_impls` table. CD-133 fixed a LIVE HIR↔MIR divergence found on the way (MIR
compared keys structurally, ignoring user `Eq`). EXCLUDED and pinned by boundary tests: `HashSet`
(HIR-only — no MIR representation, so a lowering gap like C6.3c's adapters) and Drop-bearing keys/
values (refused before MIR, which keeps entry Drop order unobservable and legitimately unspecified).**
**WP-C6.3 is CLOSED (CD-142; PARTIAL under CD-138, which corrected CD-137).** a/b/c/d are closed and C6.3f (files) is
EXCLUDED — absent from every engine and in the optional, already-unclaimable `std-full` profile — and
the CD-116 CLOSURE EVIDENCE is discharged (installed-runtime + offline build + version-mismatch
detection, `tests/c63_closure_evidence.rs`). **C6.3e is not closed**: CD-137 claimed completion while
DEV-105 stood as a known WRONG OUTPUT inside the admitted domain rather than an excluded feature, and
those two statements cannot both hold. DEV-105 is now CLOSED, and so are the two defects its
evidence surfaced (DEV-109 via CD-140, DEV-110 via CD-139) and DEV-108 (CD-138). **WP-C6.3 is
CLOSED (CD-142)** on a full `cargo test --workspace --all-targets --all-features` across linux-x64,
macos-arm64 and windows-x64 — the confirming run CD-138 item 7 required. Escalations named above
(`Box`/`HashMap` Display semantics) are excluded by decision, not blocking.**
**WP-C6.4 is CLOSED (CD-162, owner directive) and WP-C2.12 is CLOSED (CD-162)**, both on the
`e3ef603` Tier-1 evidence. **WP-C6.5 is `CLOSED` at `e3ef603`** (CD-178) — see
`WP-C6.5-CLOSURE-PACKET.md`. All thirteen §17 findings closed, none superseded; all 136 matrix rows
carry one machine-checked disposition; all 23 forked suites migrated to the shared comparator.
Corpus `1.3.0`, 160 cases, 24 metamorphic groups over twelve families, 10 of 10 trap categories,
23 mutation controls over 15 of 15 comparator fields.

**Claim boundary, stated because it is narrower than "conformant":** the admitted EXECUTABLE surface
agrees across HIR, MIR and native on both Tier-1 targets. NOT every specified limit is enforced —
**DEV-118** (the `T: Hash + Eq` bound is unenforced for `HashMap` and `HashSet`) is carried open,
non-blocking, owned by WP-C6.3. It is an enforcement omission, not a differential defect: all three
engines accept the same programs, so it cannot threaten the agreement claim.

> **Superseded 2026-07-31 (CD-275, WP-C7.9 Packet I): DEV-118 is CLOSED.** The bound is enforced at
> type instantiation for both collections. The reasoning above stands as the reason it was
> *survivable* at C6 closure — and it is also the reason nothing found it: an omission every engine
> shares is invisible to a differential, which is why the comparator now pins expectations against
> the specification rather than against engine agreement.

**Seven defects found and fixed**, each by closing a coverage gap rather than by inspection:
DEV-111, DEV-112, DEV-113, DEV-114, DEV-115, DEV-116 (incl. `HashSet::iter`), DEV-117. Three
FABRICATION classes were also found and machine-checked shut: 69 invented rule IDs (CD-154), 36
false template arrows (CD-165), and 13 cited test functions that exist nowhere (CD-169).

**Row 24 is CLOSED as of CD-161 (`8a23772`)** — the C6.5 corpus replayed on both Tier-1
targets with identical per-case observations, both records carrying `generated_corpus_status: PASS`.
Row 24 was the only bar to `CLOSED`; the closure decision is the owner's. The historical record
follows.
**WP-C6.4 CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS — ACCEPTED BY THE OWNER (CD-146;
built CD-143, reviewed CD-144, evidenced CD-145) — Tier-1 platform matrix.**
Phases 0/a/b/c/d are built: the matrix is frozen (`C6-PLATFORM-MATRIX.md`, 25 rows), target
classification is CENTRALISED in the new `starkc/src/target.rs` (before this, the rustc host WAS the
target and `stark-64-v1` was inherited by any triple), the §34 portability audit found TEN host
assumptions of which eight are fixed, and the qualification harness + Tier-1 comparison gate + three
CI jobs exist. **BOTH TIER-1 RECORDS EXIST AND AGREE, at `4844702`** (CI run 30192449131,
**all 11 jobs green**): 1705 passed / 0 failed on EACH target, 2 ignores both classified, 0
unclassified, 0 self-skipped, no deviations, determinism `match`, TIER-1 AGREEMENT on identical
per-command counts — and the same verdict reproduced LOCALLY against the downloaded records, so the
claim does not rest on a CI job having exited zero. The earlier `61008f6` records also passed and
agreed but were DISCARDED (CD-144), because the strengthened comparator refuses them. Matrix row 25 REPORT-ONLY with
G1 and G3 closed (Windows passed the C6.4 suite 14/14); **row 24 (generated corpus) BLOCKED-BY-C6.5
by construction**, which is why `CLOSED` is not available and the ceiling is
`CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS`, **accepted by the owner 2026-07-26 (CD-146)**. Row 24
ticks — and C6.4 becomes `CLOSED` — when C6.5's corpus replays through the harness C6.4 already
built; no new platform work is needed for it. Details in `WP-C6.4.md`; evidence in
`starkc/docs/compiler/evidence/c6.4/`. **Those Tier-1 records are invalidated as of CD-148**, which
touches `starkc/tests` — expected, and exactly what §3.5 of the C6.5 plan requires (C6.4 evidence
regenerated at the exact final corpus commit, older records not reused).

**WP-C6.5 `PARTIAL` — phase 0 done (CD-147), phase C6.5-1 COMPLETE (CD-148 commit 2, CD-151
commit 3).** `starkc/tests/support/differential.rs` is the single three-engine comparator authority:
extracted mechanically first (88 passed, identical to V0), then extended to the **full §39
observation shape** — stderr bytes, exit status, returned observation and a parsed Drop log, with
trap stderr normalised rather than byte-matched and 18 comparator tests proving each field is
load-bearing (**109 passed / 0 ignored / 0 self-skipped**). **C6.5-2 (CD-152) added the corpus
itself** — `starkc/tests/c6-corpus/`, strict manifest, hash lock, 28 manifest/lock tests — and
**C6.5-3 (CD-153, PARTIAL) added the thirteen §10.3 adversarial sentinels**, each pinning its
observation in the manifest because a wrong implementation is usually wrong in all three engines at
once and they would otherwise agree. **C6.5-4 (CD-155) then built the deterministic generator: corpus `0.3.0`, 89
cases — 70 generated across 15 templates, 13 sentinels, 6 retained**, with §11.4's floor asserted by
test and §11.10's determinism proven by running the generator (same seed byte-identical, relocation
stable, seed and generator-version both part of case identity, no absolute paths). **C6.5-5 (CD-156) then built the §12 replay** — the named entry point, with §12.2 admission
classifications, per-case timeouts, content-addressed sharding, §12.6 filters that cannot be mistaken
for closure evidence, and §21 evidence output: **89 cases, 89 AGREEMENT, result PASS**. **C6.5-6 (CD-157, PARTIAL) added 20 metamorphic
groups** over ten of the twelve §13.1 families (40 members; corpus `0.4.0`, replay **129/129
AGREEMENT**), with each group's semantics-preserving precondition recorded and enforced. Still owed by
§10: per-row witnesses and package breadth; by §11: the retained-case workflow and package templates;
by §12: the package-graph step and shard-summary merging; by §13: **M08/M09 and the 24/48 floor**, all
blocked on package graphs. **C6.5-7 (CD-158) closed the mutation controls: all sixteen §14.3
mutations detected** against real witnesses by the production comparator, with source-level routing
controls for the two route-sensitive ones — the negative control that makes the rest of the evidence
mean something. **C6.5-8 (CD-159, PARTIAL) added package breadth** — corpus `0.5.0`, 131 cases,
**131/131 AGREEMENT** — and found two defects: **DEV-113** (absolute paths in package trap
provenance; blocks a trapping package case) and **DEV-114** (canonical package symbols
nondeterministic for a diamond graph; ESCALATED). **C6.5-9 (CD-160) built the Tier-1 machinery** —
`c65-corpus` jobs on both targets, §16.2 measured identity, the §16.4 comparator with per-case
observation-hash equality, §20.7's thirteen controls, and the C6.4 harness now running the corpus and
measuring row 24's fields. **CD-161 then produced them: TIER-1 CORPUS AGREEMENT at `8a23772`** —
131/131 on both targets with identical per-case observation hashes, **C6.4 row 24 flipped to PASS**,
and both C6.4 records refreshed at that commit carrying `generated_corpus_status: PASS`. Recommended
WP-C6.5 status **`PARTIAL`**: the Tier-1 evidence is complete, the breadth (metamorphic floor, per-row
witnesses, §17's eight review passes) is not.
**CD-154: the matrix's rule citations were 69/84 INVENTED and are now repaired and machine-checked** —
a fabrication, not a misjudgement, and the third phase-0 exit condition to fail on inspection. Two
tests now refuse any citation that resolves to nothing, in the matrix and in the corpus manifest.
**All 23 forked suites are now migrated (CD-165, R-02)**, so their C6.2/C6.3 evidence rests on the
shared comparator rather than on twenty-three local notions of agreement. Matrix roll-up: 127
EXISTING-EVIDENCE, 4 NOT-APPLICABLE-NON-CORE, 1 ADD-METAMORPHIC, **4 BLOCKED — V19 `HashSet`**
(a lowering gap, which §4.3 forbids as a non-Core exclusion) **and K15–K17, the entry contract
(DEV-111)**. O13 left the blocker list: the refusal it cited (CD-038) was superseded by C6.1d's
unrolling (CD-084 G2) and the program runs in all three engines today.

**DEV-111 (CD-149) — the executable entry contract diverged in all three engines.** PROC-EXIT-001
says an `Int32` entry returns that status and an `Err(message)` entry writes `message` + LF to
stderr and returns 1; the oracle did both, **MIR reported status 0 with no stderr for every
non-`Unit` entry** (including a MISSED TRAP on an out-of-range status), and **native refuses to
build any non-`Unit` entry at all**. MIR is FIXED (`entry_termination`; `MirExecution` gains
`stderr`; not a contract change — `MirExecution` is absent from `mir.md`). Native is ESCALATED as a
Gate C6 blocker under `WP-C6-ENTRY.md` §3 required result 6. Two further escalations flagged, not
resolved: `invalid-exit-status` has **no `TrapCategory`** (CE3 — trap identity is frozen), and the
Unit VALUE was unwritable. **Both dispositioned by CD-150:** the CE3 is BUNDLED with the native entry
work (the same increment must emit that trap), and the Unit gap was **DEV-112, FIXED** — TYPE-PRIM-001
says `Unit` and `()` are two spellings of one type, so it was a conformance bug rather than the
spec conflict I first called it; canonicalised in all three engines, which unblocked
PROC-EXIT-001's `Ok(Unit)` clause. Retained:
`starkc/tests/c65_entry_exit_contract.rs` (8 tests after CD-150, each remaining boundary naming the
condition that retires it). **The matrix had no row for any of this** — the second inherited disposition to fail on contact
with a run.

Still owed and not reduced by any of the above: the §39 observation shape, a
generated corpus (0 of ≥64 cases), metamorphic breadth (7 groups against a floor of 24; M08–M12 have
none), 16 mutation controls (0 exist), and adversarial sentinels.
Also open:
C4/C5/C6
C6.5 differential corpus, C6.6 gate exit. (F4 parser half `&&T`/`**x`, DEV-083,
DEV-105 CLOSED by CD-138 — none is C6.2.)**

**CD-053 (owner directive, 2026-07-21), four parts.** (1) The three-engine differential harness
was built NOW as the WP-C5.2 closure addendum rather than deferred to WP-C5.6 —
`starkc/tests/three_engine_differential.rs`, 20 tests, one source per case run through HIR, MIR
and native with all three results normalised to a common outcome (completion vs. trap, exit
status, trap category, exact source file/line/column, observable output) and required equal.
**WP-C5.2 is therefore CLOSED.** The harness was mutation-tested (a wrong native `+`, and a
native trap line off by one) to prove it fails before it was trusted to pass. (2) CE4
Amendment 1 to the Native Provider ABI v0.1 was **NOT approved as submitted**; the owner approved
its principles and directed a revision, now at
`native-provider-abi-v0.1-CE4-amendment-1.md` **revision 2** — awaiting owner approval, and
neither `provider_abi.rs` changes until then. (3) The ABI version stays **`0.1`** (nothing has
shipped or executed against it). (4) DEV-095 (build-key completeness) is confirmed as a
**mandatory WP-C5.3 opening condition**: no aggregate or Drop-bearing native generation begins
until every semantic input affecting generated code is in the build key, with cache-invalidation
tests.

**CD-054 (owner directive, 2026-07-21).** The WP-C5.2 closure was reviewed and **approved**; three
bounded corrections were required and made (the outcome comparison extracted into a testable
`compare_outcomes` helper and driven with deliberately disagreeing triples; the "implements §15.1
literally" claim replaced with the precise statement that it implements the §15.1 three-engine
pipeline with normalised trap comparison, raw stderr byte equality being uncomparable because the
HIR oracle has no canonical stderr format; and the full-workspace evidence completed —
884 passed / 0 failed / 2 ignored across 52 binaries). **CE4 Amendment 1 is APPROVED at revision 3
and applied in full**: the closed `AbiParam` model, the raw/owning handle split, and four new
normative rules (consumed-handle error, output initialisation, close failure, physical ABI
mapping). The close-function question was ruled: **exactly one parameter, the consumed handle,
nothing else** — MIR's `Drop(place)` supplies no argument list, so a close with a second parameter
is one generated code cannot call. ABI version stays `0.1`. No provider executes; §10.2's boundary
is unchanged.

Preceding context (unchanged): the
owner's DEV-089 close-out directive was executed: user `Display` dispatch implemented in both
engines, non-`Copy` array iteration and cross-file `const` use rejected in the front end, all
validation green. WP-C5.1 (Runtime ABI and Layout Design) closed in full — representation
contract, backend/runtime skeleton with a proven native empty-`main` executable, and the
owner-approved Native Provider ABI v0.1. Every WP-C5.2 sub-part (C5.2a-e) is closed: real
arithmetic with correct overflow/div-by-zero/shift trapping, comparisons, `if`/`else`, `while`
loops, multi-function programs with real parameters and direct calls, and now a real trap ABI
(category + exact source file/line on stderr, exit 101) all compile and run natively via a
block-index dispatch loop. (§14's C5.2 exit condition — three-engine automated agreement — was
open at that point and is what CD-053 above closed; the per-engine `native_c5_2*.rs` tests remain
as supplementary evidence.) **An external review of head 37828a07 then raised seven findings, all seven real
(CD-052)**: four fixed (DEV-091 float→int casts accepted out-of-range values at 64-bit widths in
BOTH the MIR interpreter and the native backend; DEV-092 symbol sanitization was not injective;
DEV-093 native success-path tests observed no computed values; DEV-094 reversed version-mismatch
labels), one recorded as a WP-C5.3 opening condition (DEV-095 build key omits nominal type
context), and two escalated to the owner as a CE4 amendment to the approved Native Provider ABI
v0.1. Fixing the first surfaced an eighth defect the review had not named (DEV-096: the HIR oracle
reported every out-of-range cast as an arithmetic overflow). The pass also completed C5.2e's
`Terminator::Trap` support, which CD-051 had recorded as closed while it was still `Unsupported`.
**WP-C5.3 OPEN (CD-056), C5.3a CLOSED.** DEV-095 was discharged first (CD-055: the build key now
covers all eight version axes, the entry symbol, the source table with content hashes, all four
`TypeContext` fields and the bodies, with seven mutation-verified cache-invalidation tests).
C5.3a delivered tuples, arrays and structs — §6.2 type mapping, §6.3 nominal definitions, the
projection-type walk, aggregate construction, constant and proof-backed indexing — with seven new
three-engine cases and four native-only ones. It found and fixed **DEV-097** (the HIR oracle
blamed two different columns for the two ends of one bounds check; the fourth defect this campaign
has found living only in the gap between engines).

**THREE OWNER DECISIONS ARE OPEN (CD-056), all flagged rather than resolved:** (1) what
"three-engine agreement on target layout queries" means, since §14 requires it but the
interpreters answer 8 for every type while native answers real target layout — the exit condition
cannot be satisfied as literally written; (2) the §6.3-vs-§7.4 `Copy`-derive reading, implemented
and reversible in one function; (3) the non-`Copy` storage strategy (§7.2), which **blocks
C5.3d** and is already visible as C5.3a's scope boundary — a non-`Copy` move across a
basic-block boundary is refused as `Unsupported` because the block-dispatch loop defeats Rust's
borrow checker.

**C5.3b CLOSED (CD-057)** — user enums, discriminants and payload access run natively; the
variant-field projection is emitted as a `match` expression, since Rust cannot project into a
variant otherwise. It also makes **decision 3 urgent**: conditionally constructing an enum and
then matching it is the ordinary shape, and it straddles a basic-block boundary, so the
non-`Copy` storage strategy is a **prerequisite for C5.3c** (`Option`/`Result` payloads are
frequently non-`Copy` and `?` is inherently cross-block), not a nicety.

**All three CD-056 decisions RESOLVED by CD-058**: layout agreement means exact values under one
injectable target-layout manifest (relations-only tests no longer discharge the exit condition);
the Copy-derive reading is approved with `copy_types` as the sole authority; and non-Copy storage
is §7.2's `ValueSlot<T>` over `MaybeUninit<ManuallyDrop<T>>` — plain `Option<T>` rejected for
introducing Rust-owned destruction, `Option<ManuallyDrop<T>>` rejected as the general form because
a partially moved value's bytes need not form a valid `T`.

**C5.3d-0 CLOSED (CD-059)** — `ValueSlot` is sound for partial moves (three-state machine, Miri
verified), generated projection helpers confine all `unsafe` to one module, and all five movement
shapes work. **C5.3c is unblocked.**

**One structural finding needs an owner decision**: a user `Drop` impl's receiver is `&mut Self`,
so `impl Drop` requires `MirTy::Ref`, which is outside the C5 subset. User destructors therefore
cannot be dispatched natively, and C5.3d-1's observable destruction fixture cannot be built as
planned — §7.7 is currently proven structurally instead. Admitting `Ref` for destructor receivers
is an owner-level scope question.

**C5.3c CLOSED (CD-061)** — Option, Result, matches and `?` run natively on generated core enums.

**The two remaining C5.3 gaps are one gap: no references.** User `Drop` impls need `&mut Self`;
`Ordering` needs `cmp(&other)`. A narrow destructor-reference lane, slightly widened, closes both
— and until it lands, C5.3d-1's observable destruction fixture cannot be built and the enum drop
glue fixed under CD-060 stays unexercised.

**All open decisions resolved by CD-062.** C5.3's remaining work is now **two closure packages**,
not four gaps: (a) references/Drop evidence — C5.3d-1a ephemeral reference lane → C5.3d-1b
canonical `DropPlan` → C5.3d-1c observable evidence; and (b) C5.3e, the exact target-layout
manifest, independent and parallelisable. §6.2 amended for generated core enums; universal
`NativeOperation` IR deferred.

**C5.3d-1a CLOSED (CD-063)** — the lane is implemented; `Ordering` is reachable and user
destructors compile and run natively. One deviation from CD-062's wording is flagged for the
owner: `cmp` consumes its borrow by a `Deref` READ, not by a direct call, because lowering inlines
primitive comparison.

**C5.3d-1b DONE** — `mir::drop_plan` is the single derivation of destruction order, consumed by
BOTH the MIR interpreter and the native emitter. It removes the defect class CD-060 was an instance
of: two independent reconstructions of one rule. Four invariants are now carried by the plan's
SHAPE rather than by convention — the type's own destructor nests *outside* its components (so
"fields before the destructor" is unrepresentable, not merely discouraged), components are stored
in destruction order, `Variants` is indexed by variant number with complete coverage and full
arity, and any component with no obligation is absent (which is where "never drop a `Copy` field"
now lives). `Vec`/`Box` name their element by type rather than inlining a sub-plan, because they
are Core v1's only indirection and therefore its only route to a recursive type. **MIR v0.1
unchanged**; runtime surface untouched. The variant-payload table, which existed three times,
moved into the same module. Tuples and arrays reach the native drop path for the first time as a
consequence. Evidence: 14 derivation tests plus CD-062's five representable mutations, each
corrupting the *shared* plan and showing the corruption reach the generated Rust — which is what
proves application rather than re-derivation; the sixth (Drop after a trap) was already covered by
existing differential/native fixtures and is unaffected by this package.

**CD-065: the process-driven re-engineering phase of C5 is CLOSED by owner assessment.** What
remains is evidence, manifest, linkage, build UX and qualification — not architecture. Deferred
explicitly: `NativeOperation` IR, operation-planning abstractions, dashboards, process metrics,
retroactive work-package conversion, general references, liveness bitmaps. Two process items
survive: an adversarial review at C5.3 closure and a gate-exit review at C5.6.

**C5.3d-1c DONE — and it was not purely evidence work.** The owner's predicted seam was real and
WIDER than predicted: the partial-move fixture failed to build, and so did the plain
**reverse-field-order** fixture. MIR's drop elaboration emits **one flag-guarded `Drop` per drop
unit on a PROJECTED place** (`drop _1.1` then `drop _1.0`), not one whole-local `Drop` — so any
struct with two droppable fields and no destructor of its own could not compile natively at all.
The backend's refusal of projected `Drop` was right rather than merely conservative (collapsing
per-unit drops into a whole-local one destroys a unit MIR's flags say is gone, §7.6), so it was
closed with a real per-unit operation: `HelperOp::Drop` wrappers over
`ValueSlot::drop_field_with`, plan baked into the wrapper, call sites still safe and glue-free.

**C5.3d-1 is CLOSED** (1a references, 1b `DropPlan`, 1c observable closure).

**C5.3e is now the ONLY remaining C5.3 exit condition.** Everything else in §14 is discharged.
**Process note:** full-workspace test runs are now reserved for WP/gate closure points,
not every intermediate change, per owner feedback.

## WP-C7.8 — IN PROGRESS (CD-212)

**All five packets are dispositioned. C7.8.0–C7.8.2 are closed: MIR represents provider calls,
verification enforces invariants 1–5, the binding plan and emission close 6, 8 and 9, the resource
framework structures 7, and `stark-time` executes natively through the ABI (CD-210). C7.8.3–C7.8.6
are unblocked.**

**Packet 4 / CE1 (CD-212) — no Core specification change.** The normative Core `File` surface stays
exactly as specified; arguments, environment, time, sleep and TCP are package capabilities;
`IOError` stays file-I/O-only; `NetworkError`/`ProcessError` are package-owned. No Core
`File::read`, `read_to_end`, `write_all`, `flush`, or networking API. Where a package byte-read
cannot be built over Core `File`, the package binding invokes the **provider** byte-read primitive
directly — reaching past Core to a provider it already owns, which adds no Core symbol.
Package conveniences must preserve short-read, short-write and successful-zero-write rules.

**Packet 5 / CE9 (CD-212) — explicit trust boundary.** Providers are admitted only through
package-declared capability requirements and target-compatible validated metadata: no implicit
discovery, no fallback, no priority rule, no dynamic loading, and **`stark build` fails when a
required capability has no unique selected provider**. Arguments and environment are read-only; no
environment mutation. Paths pass to the provider **verbatim** — no normalisation, shell/tilde/env
expansion, or hidden working-directory changes; relative paths resolve against the launched
process's working directory. Outbound TCP needs an explicit call and address. Inbound TCP is
admitted **only** via an explicit `TcpListener::bind(address)` — no hidden default, no implicit
`0.0.0.0`, no listener as a side effect of package loading. Loopback is mandatory for
qualification. Raw descriptors are never exposed. Contract violations and host failures stay
outside package error enums. Dynamic loading, sandboxing, allowlists and deployment policy are
deferred.

**CORRECTION (CD-220), because the earlier entries overstated this.** "Executes natively" has meant
*a hand-built MIR body calling the provider compiles, links and runs*. It has **not** meant a STARK
programmer can use the capability: `lower.rs` produces no `Callee::Provider` at all, and every
capability e2e hand-builds MIR. Those tests are backend and ABI evidence — they are what proved
emission, ownership and the three status channels — but they are **not** source-language capability.

| capability | provider executes (hand-built MIR) | reachable from STARK source |
| --- | --- | --- |
| time (`stark-time`) | yes, both symbols (CD-219) | **yes, via the compiler library** — `c788_source_time_e2e.rs` (2026-07-30); not yet via `starkc build`, see below |
| args/env (`stark-env`) | yes (CD-214, CD-216) | **no blocker left in lowering** — recoverable statuses lower as of 2026-07-30; needs its manifest binding written and an e2e |
| file (`stark-file`) | yes, create/write/complete/close (CD-217) | no — resource nominal (§3.1) |
| tcp (`stark-net`) | no — resource types unbound (CD-218) | no — resource nominals (§3.1) |

**AMENDED 2026-07-30.** The right-hand column is no longer uniformly "no": the source path exists
and one capability traverses it end to end. What blocks the other three is now specific and named,
not "lowering emits no provider call" — that was the general blocker and it is gone. `stark-env`
has no lowering blocker left (recoverable statuses lower as of 2026-07-30) and needs only its
manifest binding and an e2e; `stark-file` and `stark-net` still need a resource-nominal mechanism
(§3.1). P1's host-capability precondition is **partially** removed: a STARK program can
now call a scalar capability, which the closure statement
(`WP-C7.8-First-Party-Native-Host-Capabilities.md` §5.7) should read as narrowing rather than
lifting the amendment.

**Packet 6 / CE3 (CD-220) — Route B.** A package-declared host resource gets an explicit MIR
representation, not an ordinary struct and not a new `CoreType`. It retains the STARK nominal *and*
the provider resource identity, and emits as `OwnedResourceHandle`: no fields, no `Copy`/`Clone`, no
Rust `Drop`, MIR-owned exactly-once close, `resource_type` validated on `HandleOut`. Packet 4 stands
— TCP is not moved into Core to unblock a demo. Marking ordinary structs was rejected: a hidden
special case obliges every consumer to remember it, and the first that forgets emits fields where a
handle belongs.

**WP-C7.8.8 — source/package provider integration** is now the critical path. **Eight steps**, the
order CD-225 approved and the design's §16 carries verbatim: manifest `provider_api` parsing →
synthesis of package items and resource nominals → typed HIR bindings → resource-name-to-nominal
registry → resolution-time `MirTy::HostResource` → `Callee::Provider` lowering → close arena and
verifier rules → **source-level monotonic-time proof**. Proven in that order on real STARK source:
time, args/env, File, TCP bind/connect, accept, full echo. TCP sits behind this, not in front of it.

**CD-234 (2026-07-30) — the resource-nominal mechanism, and A11 IMPLEMENTED at MIR 0.2.**

The owner dispositioned the §3.1 gap: a resource nominal is a synthesized **zero-variant enum**
(`enum TcpStream {}`). Both alternatives were rejected — a compiler-injected spanless item
(reintroduces fabricated spans) and an ordinary struct plus a do-not-construct marker (soundness
resting on a rule every future construction path must remember, the same hidden special case Packet 6
already rejected). A zero-variant enum is opaque **structurally**: no fields, no variants, no
constructor expression, no pattern that can manufacture a value, and no marker to forget.

Attached condition: the nominal supplies **source identity only**. A provider-bound instance lowers
to `MirTy::HostResource` and must never receive an ordinary zero-variant enum's backend
representation or default-initialisation. A `HostResource` local becomes live only through a
successful `HandleOut`, a move from an already-live resource, or an argument/return carrying one.
Drop flags still decide whether a *live* resource closes, but may not excuse a forged placeholder
existing: **a dead host-resource slot contains no semantically valid STARK value, and native code
must never read or close it.** Recorded as a CE3 clarification to A11, not a new Core feature.

**A11 is now implemented** — it had been approved on paper since CD-224 and entirely unbuilt
(`MirTy::HostResource` existed nowhere; `MIR_VERSION` was still `0.1`). Landed: the variant with all
three identity fields, structural identity over `(nominal, provider, resource)`, the canonical
`hostres#<provider>/<resource>@<content path>` rendering in `symbol_ty` (content path, never
`ItemId` — CD-108), §Q6's rule that every host resource emits as `OwnedResourceHandle` regardless of
nominal, and CD-234's refusals: `MIR-0026` rejects any rvalue other than a move (no aggregate — including
an enum-variant aggregate — no constant, no discriminant, no borrow, **no copy**), and
`default_value_expr` refuses outright rather than fabricating a handle. Evidence:
`starkc/tests/a11_host_resource.rs` (13 tests).

**Adding the variant produced ZERO compile errors, which was the risk rather than the relief.** Every
`MirTy` match has a wildcard arm, so a host resource would silently have inherited ordinary-enum
treatment. The sites that matter were made explicit deliberately, not because the compiler forced it.

**CD-237 — A11 §5's close lifecycle: selection, the five obligations, and drop planning.**

`ValidatedProviderClose { resource, close }` and `MirProgram::provider_closes`. The close is selected
at **resolution**, not at drop time (`ProviderLowering::select_closes`) — which is what lets the
verifier discharge §5's obligations *before* emission; a close chosen at drop time could only be
checked once the program was already being built.

`DropPlan::HostResourceClose { close }`, and `plan_for` on a `HostResource` with **no** recorded close
is an **error, never a `Noop`**: planning nothing is obligation 5's leak itself, since the provider
never learns the handle was abandoned and nothing downstream can detect it. There is no `then` arm —
a host resource is opaque by construction (CD-234), so nothing is inside it to destroy after.

**The five obligations, all program-level** (`verify_provider_closes`): `MIR-0028` exactly one close
per resource; `MIR-0030` the close is declared `is_close_for` *that* resource; `MIR-0031` it belongs
to the same resolved provider; `MIR-0032` it takes exactly one `HandleConsumed` of it and no value
output (ABI §13.1); `MIR-0029` the binding is well-formed.

**`MIR-0030` is the one a structural check cannot make.** `stark_tcp_listener_close` and
`stark_tcp_stream_close` have identical shapes — both consume one handle — and differ only in the
resource they name, so a listener closed by the stream's close typechecks perfectly. Only comparing
`is_close_for` against the resource catches it.

**`MIR-0033` is what makes "exactly once" true** (§5 rule 4): a `Callee::Provider` naming an
`is_close_for` declaration is rejected outright. A package cannot bind a close, so any such call site
means another path found one — a second destruction path for a resource MIR already closes. MIR owns
the only path.

The reference interpreter refuses a host-resource close rather than pretending: closing needs a
linked native provider, so such a program is native-only, and saying so beats a silent no-op. Generic
drop glue refuses it too — a close is a provider call and must come from the `Drop` terminator's own
path, which has the arena.

**Native emission and the lifecycle rules (same slice).** `ProviderLowering` carries the selected
closes; lowering copies them to `MirProgram::provider_closes` and keys them into
`TypeContext::host_resource_closes`, because `drop_plan::plan_for` resolves destruction from the type
alone and a resource's destruction *is* its close. The `Drop` terminator routes a `HostResource` to
`emit_host_resource_close` rather than generic glue — a close needs the arena, the symbol and the
consuming-handle shape, none of which a glue expression has.

**CD-234's lifecycle rules fall out of the slot mechanism rather than needing separate checks.** The
close is emitted through the same `drop_with` every non-`Copy` local uses, so: a
declared-but-never-initialised resource has a clear flag and never closes; a failed `HandleOut` never
wrote the slot, so nothing closes; a moved-out resource leaves its source dead, so only the
destination closes; and a consuming call takes the value out, so the later implicit `Drop` finds it
dead and cannot close twice. The handle is *taken*, not borrowed — which is what makes a second close
impossible rather than merely unlikely.

**CD-240 — the bottleneck was one wildcard, and it is fixed.**

`TypeContext::is_copy` ends in `_ => true`, so `MirTy::HostResource` was silently classified **Copy**.
Three consequences, none of which announced themselves: `is_slot_backed` became false, so the local
was declared through `default_value_expr` — which refuses a resource — and emission failed before
`Drop` was reached; `emit_drop` refuses a `Copy` type outright, so the close could not have run
either; and `Copy` is the licence to *duplicate* a handle, which gives two owners of one resource and
closes it twice.

The arm is now explicit and `is_copy(HostResource) == false`. That single change makes a resource
local slot-backed, and a slot-backed local is already declared `ValueSlot::dead()` with **no default**
— so CD-234's "the slot begins dead, and no placeholder may make it live" is now the representation
itself rather than a rule anything has to enforce. The `Drop`→close path written in CD-239 is
reachable, and the emitted form is
`local.drop_with(|__v| unsafe { close(__v.take_raw()); })`: taken, not borrowed, so a second close is
impossible.

**This is the third time a `MirTy` catch-all has swallowed the new variant** (see the zero-compile-error
note under CD-234). The parallel session independently diagnosed the same root cause and left tripwire
assertions — `assert!(is_copy(&resource), "current failing point changed … upgrade this test")` — in
the two boundary tests. Both tripped as designed and are now upgraded to assert close emission and
success-only `HandleOut` writeback. `a11_host_resource.rs` carries the standing regression guard,
because the defect produced no compile error and only an assertion can catch its return.

**CORRECTION (superseded by CD-240 above) — the close emission is written but NOT YET REACHABLE.** A host-resource local still
fails earlier, at `default_value_expr`: the CFG dispatch loop default-initialises every local
**eagerly**, and CD-234 requires a resource to have no default. So emission refuses before `Drop`
ever runs. The parallel session's `c788_resource_lifecycle.rs` pins exactly that boundary with an
`expect_err`, and it is right to.

The missing piece is CD-234's remaining backend requirement: **generated Rust must use an
uninitialised slot or equivalent slot-backed representation**, so a resource local is not materialised
at all until a successful `HandleOut` writes it. Until that lands, the `Drop`→close path is dead code
that the lifecycle tests exercise only through hand-built MIR at the emission layer.

**Still open:** the slot-backed representation (the blocker above), driver-side close selection
(`select_closes` exists and `native_build.rs` does not call it), and the source-level lifecycle e2e (never-initialised does
not close; failed `HandleOut` does not close; successful closes exactly once; move then drop closes
only the destination; consuming close prevents a later implicit close).

**CD-257 — slice 7's closure matrix is filled, and it refuses to over-claim.**

`c78/closure-gate-slice7.md` separates frontend / HIR / MIR-lowered / native-runtime /
cross-platform, per §5.7's requirement that a single "supported" column would reproduce the
over-claiming C7.2 was corrected for. Every row cites evidence committed on `main`; a session's
working tree is not evidence.

That distinction immediately did work. **TCP's bind/accept/echo path exists and passes locally and
is NOT claimed**, because it is uncommitted. TCP's *resource lifecycle* is claimed, because
`c788_lifecycle_e2e` is committed and runs on all three platforms.

`stark_net.native_e2e` moves `pending → implemented`, and the CI assertion moves with it — plus a
new step that runs the test justifying the claim, so the record cannot say "implemented" while its
evidence is absent or failing. The lifecycle set is recorded as `partially_observed`: four cases
observed, one unreachable by construction, the rest defined-but-unrun.

**The gate also records a methodology finding.** Six `MirTy` catch-alls silently swallowed
`HostResource` (`dump_ty`, `emit_ty`, `default_value_expr`, both `is_copy` predicates,
`ty_needs_drop`, `may_need_drop`). Each compiled cleanly; each was found downstream, one at a time.
`ty_needs_drop`'s `_ => false` meant **no `Drop` was ever emitted for a resource — every resource
leaked while every unit test on the close machinery passed** — and only a test inspecting generated
code found it. Recommendation carried into the gate: remove `_ =>` fallbacks from the predicates that
decide semantics, which would have made all six compile errors at CD-234.

**SELECT-C (CD-253) — Core `File` remains entirely on the legacy MIR resource path.**

`CoreType::File` lowers unconditionally to `MirTy::Core(File, ..)`, independent of capability
declaration, provider selection, or build configuration. **Backend representation equivalence does
not establish MIR identity equivalence**: both the legacy and A11 paths emit `OwnedResourceHandle`,
which is precisely why the difference has to be enforced in the verifier rather than noticed
downstream.

**The invariant is broader than `File`: a type must not change MIR identity according to how the
build was configured.** Migrating `File` needs the provider name at type-conversion time, and that
is known only after selection — so its representation would depend on whether the program declared
the capability, giving one type two identities and violating CD-235's no-mixed-migration rule.

Rejected alternatives, both for reasons larger than this work package. **Capability-gating `File`**
would couple type *availability* to provider binding, so `let f: File;` would become invalid in
generic, unreachable or declaration-only code that performs no host I/O — a Core typing change
affecting library APIs, generic signatures, tooling and conformance fixtures. **A provider-less
`HostResource`** would move provider resolution from type construction into linking and raises its
own model questions (may unresolved resources reach verified MIR? which pass binds them? is provider
identity part of MIR equality? can cached MIR be reused under a different selection?). Either may be
right later; neither is required now.

**The loss is narrow and explicit: `File` does not participate in the A11 close arena in this
revision.** `MIR-0033` continues to exempt it — and the exemption exists because `File` is retained
as a *complete legacy resource path*, not because mixed representations are tolerated.

Closure conditions implemented: the mapping is frozen; **`MIR-0027` now rejects a Core-owned
resource as a `HostResource` by ANY route** — checking only the nominal was too weak, since
`resource: "file"` under an *Item* nominal is the same mixed identity; both build configurations are
tested to produce identical MIR identity; and legacy affinity is verified separately (non-`Copy`, so
moves invalidate the source, and the same owning handle in generated Rust). Evidence:
`a11_host_resource.rs` (34 tests).

**CD-235 — the nominal identity is widened, and the Core side is sequenced.** A11 §4 wrote
`nominal: ItemId`, which cannot name a Core resource: `File` resolves to `CoreType::File`, a different
enum from `ItemId`. So `nominal` is now `HostResourceNominal::Core(CoreType) | Item(ItemId)`, and §4's
"one representation, two authorities" is expressible on both sides.

**Package resources use `MirTy::HostResource` immediately. Core `File` stays on its pre-A11
`MirTy::Core(CoreType::File, [])` path**, which is what C7.8.4's evidence qualified. `ResourceRegistry`
maps `file → ResourceBinding::LegacyCore(..)`, so the migration is a registry change, not a type
change. **A sequencing exception, not a permanent second representation — and A11's Core side does not
count as implemented until the migration and its requalification close.**

`V-HOSTRES-1` / **`MIR-0027`** rejects a `HostResource` naming a Core nominal, which is what makes the
exception safe rather than merely documented: one Core resource with two representations in one
program means two drop-close paths for one handle kind, and the first consumer to pick the other
closes twice. The guard is removed **by** the migration step.

The named migration step carries bounded requalification: provider resolution and emission;
create/open output initialisation; borrowed read/write/complete; consuming close; implicit `Drop`
close; failed `HandleOut`; move and early-return lifecycle; no double close; generated representation
stays `OwnedResourceHandle`; C7.8.4's native e2e behaviourally equivalent.

**Package-resource lifecycle progress.** Synthesis emits resource nominals as zero-variant enums
(CD-234) and refuses a signature naming a nominal the package does not bind. Lowering handles
`HandleBorrowed` (shared borrow, never a move — the call only reads), `HandleConsumed` (move;
ownership transfers at entry and does not return on failure) and `HandleOut` (the argument names the
destination place, per the C7.8.4 convention, and the slot is **not** initialised — it begins dead and
only success makes it live). Handle outputs join the `Ok` payload after the scalar out-slots, matching
`provider_sig`'s derivation order. Still open: the close arena, the `Drop`-terminator close, drop-flag
verifier rules, and the slot-backed generated-Rust representation.

**A11 §3 and §9 disagree, and §9 is right.** §3's table claims the installed-runtime check gives
cross-version rejection and "needs no new logic". `stark_runtime::version::check` compares only
`runtime_version`, and that module documents the other fields as recorded-not-validated — putting
`mir_version` there would make the runtime an authority over a compiler-internal representation. §9
consequence 3 is instead satisfied by **V-SURFACE-1 / `MIR-0017`**, whose exact-equality check on
`mir_version` rejects in both directions already. Consequence 1 likewise already held: `build.rs`
folds `mir={}` into the build key, with a mutation test perturbing it.

**Verified, not assumed** (§9 consequence 5): build-cache, reproducibility, profile-agreement,
snapshot and closure-evidence suites all pass under `0.2` with **no re-pinning**, because nothing
derives the version string except the synthetic C6 tier-1 fixture — which stays `0.1`, exactly as
§9's immutability rule requires.

**Still open on the resource path:** synthesis of the zero-variant nominals, `ResourceRegistry`'s
change from resource-name→`MirTy` to resource-name→nominal identity, resolution-time construction of
the `HostResource`, drop-flag/close-arena rules, the slot-backed generated-Rust representation, and
CD-234's lifecycle negative tests (never-initialised does not close; failed `HandleOut` does not
close; successful `HandleOut` closes exactly once; move then drop closes only the destination;
consuming close prevents a later implicit close). `File` and TCP need those.

**DECISION-ID CORRECTION (2026-07-30).** Two commits landed with **already-used** CD subjects:
`cdba7c8` says `CD-196` and `ee85652` says `CD-197`, but CD-196 is "WP-C7.8 REVISE" (`4419d6c`) and
CD-197 is "Packet 3 dispositioned under CE2" (`9aa7482`). Their correct identities are **CD-228**
(step 3) and **CD-229** (steps 6 and 8). This entry and
`WP-C7.8.8-PACKAGE-API-DESIGN.md` §16 are the authority for the mapping.

The subjects are **not** rewritten. They are pushed, a parallel session works this repository, and
force-pushing shared history to correct a label risks destroying that session's work — a strictly
worse outcome than a subject line that needs this note. Cause: the CD sequence is allocated by
decision, not by commit order, so the last commit on `main` at session start (`CD-195`) was not a
reliable high-water mark. **Read the maximum from `git log --all | grep -oE "CD-[0-9]+"`, not from
`HEAD`.**

**POSITION (2026-07-30): the source-to-provider gap is CLOSED for a scalar capability.**
`c788_source_time_e2e.rs` compiles a `.stark` program that calls a manifest-bound function with
ordinary syntax, lowers it to `Callee::Provider`, links `stark-time-native`, runs the binary and
asserts the printed monotonic reading is nonzero. **No hand-built MIR anywhere in that path.** This
is what CD-220 named the critical path, and what every earlier provider e2e could not demonstrate:
`lower_program` hard-coded `provider_calls: Vec::new()`, so no STARK source could reach a provider
at all. §16 steps 1, 3, 6 and 8 are done; step 2 is done **for functions only**; steps 4, 5 and 7
were blocked on the resource-nominal gap, which CD-234 dispositions (design §3.2).

Step 6 is hooked at `Res::Item` in `lower_call` — after name resolution, type checking and borrow
checking have all seen an ordinary function, which is what keeps the front end free of provider
special cases. `ScalarOut` becomes a zero-initialised caller-owned local passed as `&mut`; the
call's `dest` takes the raw status code, not the STARK value; the `Result` is built afterwards from
the slots. `lower_program_with_providers` is a new entry point rather than a parameter added to
`lower_program`'s ~20 call sites.

**Stated precisely, because CD-220 had to correct an over-claim of exactly this shape once already:
the proof runs through the compiler *library*, not `starkc build`.** The test drives parse →
resolve → typecheck → `lower_program_with_providers` → emit → link → run itself. The driver
(`native_build.rs`) still calls plain `lower_program` and never invokes synthesis, so a package
with a `provider_api` block in its manifest does not yet build from the command line. Every
component of that path now exists and is tested; what is missing is the driver wiring — manifest →
derive → synthesize → prepend to the compilation unit → resolve → lower-with-providers. That is
the next slice, and it is integration rather than design.

> **Superseded 2026-07-31 (CD-285): the driver wiring is BUILT.** `native_build.rs` calls
> `synthesize_with_resources`, assembles the provider layer, and calls
> `lower_program_with_providers`. Demonstrated rather than read: the `c7-p1-rest` workload — a real
> `provider_api` package binding six TCP/env functions and two resource types — compiles, links and
> produces a binary from `stark build`, both in-repo and from an installed toolchain under
> `STARK_REQUIRE_INSTALLED_RUNTIME=1`.
>
> **Recorded because the paragraph above outlived its accuracy and cost a reader a wrong critical
> path.** With parallel sessions landing slices, an append-only file states a position that the work
> can overtake without anyone noticing; a POSITION entry is only as good as its most recent
> correction. This one was found by re-testing a claim rather than re-reading it.

**ONE refusal remains: a resource in any position (§3.1).**

**Recoverable statuses now lower (2026-07-30).** A capability with a declared vocabulary gets a
`SwitchInt` on the status: zero builds `Ok` from the out-slots, one arm per declared code builds
`Err(RawE::V)`, and `otherwise` is **`Unreachable`** — never a fallback error, because an undeclared
nonzero code already aborted inside the emitted call and a `_ =>` mapped to a generic package error
is the channel collapse Packet 1 §1.2 forbids. Each declared code gets its own block, since each
constructs a different variant.

**§7.2 clarified: the compiler generates the raw error enum.** That section says the manifest carries
"only the minimum raw error identity" with **no** code→variant table, and that the compiler
"produces the raw typed result" — together those leave no way for a package-declared enum to say
which variant means status 3. One variant per declared code, named by the vocabulary, ordered by
code. An empty vocabulary yields an **uninhabited** enum (`enum RawTimeError { }`), so `clock`'s
`Err` arm cannot be constructed at all: the type system now states what the three-channel rule states
in prose. Two capabilities may share a raw error type while they agree; a disagreement on any code is
refused.

**One backend change (§16.3).** An uninhabited enum had no generated-Rust representation, and it
surfaced as soon as a program bound one (`Err(e) => …`), because the CFG dispatch loop
default-initialises locals **eagerly** — so an aborting expression fires on entry rather than on
misuse, unlike the named `FnPtr` sentinel it sits beside. A zero-variant enum's Rust declaration now
carries a single placeholder variant. It is invisible to STARK: the front end sees zero variants, so
nothing can construct or match one, and MIR never reads such a local.

**Position (2026-07-29): steps 1–3 done, and step 3 collapsed into step 2.** Synthesis is generated
STARK source (`provider_synth.rs`) rather than constructed HIR, because every HIR name is a `Span`
into a `SourceFile` — so there is no separate "typed HIR binding" step to do; the ordinary front end
builds the HIR and the binding rides alongside in a side table. `c788_synth.rs` compiles the
generated layer through parse → resolve → typecheck rather than inspecting it as text, which is how
it caught the body needing to be a tail expression: `panic(…);` with a semicolon types as `Unit`.

**Finding — resource nominals have no mechanism yet (design §3.1).** *[SUPERSEDED by CD-234 above: a synthesized zero-variant enum. Retained as the dated record of why the question arose.]* Every source form that declares
a nominal is constructible, and a host resource must be opaque, so generating source for one would
let a program forge a handle no provider produced — `from_raw_checked` would not catch it, because
the `resource_type` would be whatever the forger wrote. Synthesis therefore **refuses** any signature
with a receiver or resource type rather than emitting something weaker. Steps 4–7 all touch resource
nominals and are blocked on deciding that mechanism (compiler-injected opaque item form, or a source
form the checker refuses to construct). **Step 8's target needs none**, so the remaining path to the
monotonic-time proof is step 6 alone — lowering a call to a synthesized item into `Callee::Provider`,
which the emitter and linker already execute from hand-built MIR (`a10_stark_time_e2e.rs`). CD-225's
"time before resource capabilities" ordering was load-bearing, not merely convenient.

**CD-224 dispositions — A11 APPROVED, and three rules that outlive it.**

**MIR `0.2` is approved** for A11's new `MirTy` form. A10's surface-bump precedent does not carry: a
`Callee` variant fails at one match site, a `MirTy` variant flows through every part of the compiler
that reasons about types. `MIR_RUNTIME_SURFACE` stays `0.1-A10` — A11 adds no `RuntimeFn`, because a
close is a provider call through MIR's `Drop` terminator.

> **Historical gate evidence remains immutable and valid for the version and commit it records. A
> representation-contract version increment does not retroactively reopen the gate, but current
> compiler claims that rely on the changed representation must be requalified under the new
> version.**

Closed C6 evidence is **not** rewritten or regenerated as though produced under `0.2`. **A version
bump alone is not a gate-reopening condition** — Gate C6 reopens only if the bounded non-regression
run finds an actual regression in a C6 *closure claim*. Seven consequences are in scope for the
implementing slice (build keys, re-pinned current snapshots and locks, explicit two-way version
rejection, serializer/validator support, tested cache invalidation, current differential/native
suites under `0.2`, and the bounded C6 ownership/Drop/native non-regression run).

> **There is one authoritative callable signature: validated provider metadata. The package
> declaration exposes and names that callable surface but does not mirror its physical or ownership
> signature.**

Package declarations name capability, provider symbol, public item identity, associated resource
where applicable, and error mapping where not derivable — never ABI parameter types or ownership
modes. The requested signature-mismatch diagnostic is withdrawn as structurally impossible and
replaced by six derivation-failure cases; CD-219 is the evidence that a mirrored signature drifts.

> **Application source and ordinary package APIs may name capabilities and package declarations
> only. Provider crate identities, raw symbols and physical ABI parameter forms are not part of
> application-visible STARK source.**

**Core and package resources are distinct authorities sharing one representation.** `file →
CoreType::File` stays compiler-owned and undeclarable by any package; `tcp_listener`/`tcp_stream`
are package-declared. Both lower to A11's host-resource form. Packet 4 holds on both sides: `File`
stays normative Core, TCP stays package-owned, and neither mechanism reaches into the other.

Drafts: `WP-C7.8.8-PACKAGE-API-DESIGN.md` (rev. 3) and `mir-amendment-A11-host-resources.md`
(approved). Open: design §7.1–§7.3 (item paths, error-mapping home, visibility) and A11 §8.3 (the
close arena).

**Remaining in C7.8:** CLI/package-manifest capability declarations and provider selection; C7.8.3
args/env; C7.8.4 File (registering `file` in the resource registry); C7.8.5 close-out (**done**, CD-219);
C7.8.6 TCP (**registered**, CD-218; execution blocked on Packet 6); **C7.8.8 source/package provider
integration (the critical path)**; C7.8.7 three-platform qualification and the P1 unblock
assessment, which must report backend evidence and source capability as separate columns. **Cross-platform architecture claims stay unticked until C7.8.7 evidence
exists** — work to date is proven on one host plus CI, not on a three-platform record.

| | |
| --- | --- |
| Plan | `STARKLANG/docs/compiler/work-packages/WP-C7.8-First-Party-Native-Host-Capabilities.md` |
| Decisions | `WP-C7.8.1-DECISION-PACKETS.md` — 3 of 5 dispositioned |
| MIR amendment | `mir-amendment-A10-provider-invocation.md` (rev. 1, CE3, `0.1-A10`) |
| Superseded | `STARKLANG/docs/compiler/plans/WP-C7.8-Native-Host-Capability-Foundation.md` — REVISE, CD-196 |

**Packet 1 / CE4 (CD-198, CD-199)** — first-party providers are **statically linked, ABI-semantic**:
ordinary Rust crates linked into the produced binary, direct `extern "C"` symbol reference,
conforming exactly to ABI v0.1 §7/§8/§9/§11/§12/§13 and constructed only through §6.1's boundary
helpers. Dynamic loading is a separate later WP. Panic containment is **already structural** — the
generated workspace sets `panic = "abort"` in both profiles, so a provider panic aborts rather than
unwinding into generated code, and no `catch_unwind` may be added to the static path. An
**undeclared provider status code is a contract violation**, never a generic `Other`. Provider
symbols are validated **verbatim, never sanitised** (`[A-Za-z_][A-Za-z0-9_]*`, identity-prefixed,
unique across the selected set). Provider selection is by capability + target triple; ambiguity is
a hard error with no priority mechanism.

**Packet 2 / CE3 (CD-200)** — `Callee` gains `Provider(ProviderCallId)` resolving to a validated
`FunctionDecl`; `MIR_RUNTIME_SURFACE` advances `0.1-A9` → `0.1-A10`. Provider calls may **not** be
`RuntimeFn` values or bare symbols — `RuntimeFn` stays reserved for compiler-owned operations. Nine
verifier invariants bind, plus `resource_type` validation before an owned resource is constructed.
Provider calls are target-resolved **before** MIR verification; the backend never performs
first-time selection nor interprets unvalidated metadata. `Instance` and `FnValue` are untouched —
A10 is purely additive.

**Packet 3 / CE2 (CD-197)** — STD-IO-001's "cannot surface a new language trap" and ABI §13.2's
fatal close are reconciled **without amending either text**: a failed provider close is a §12 **host
failure**, a channel already held distinct from a STARK trap. `close(self)` **consumes `self` at
call entry unconditionally**; a completion failure returns `Err(IOError)` and the resource still
passes through MIR `Drop`. Swallowing close failure is rejected on the record. Seven binding
conditions.

**Open:** Packet 4 / CE1 (Core-versus-package API placement — recommends the option needing no Core
change) and Packet 5 / CE9 (trust boundaries). Both gate C7.8.3 onward, neither gates C7.8.2.

**C7.8 does not close Gate C7.** It removes P1's native-capability precondition. C7 stays
`CANDIDATE-COMPLETE-BLOCKED-BY-P1` until P1's own exit criteria are met.

### C8 concurrency boundary (CD-201)

C8 (semantic language services) runs in parallel per `COMPILER-ROADMAP.md` §4.3 and is currently
active. Authority is split by surface, not by file proximity:

| Owner | Surface |
| --- | --- |
| C8 | LSP, editor integration, protocol behaviour, related front-end diagnostics; `starkc/src/lsp/`, `starkc/src/analysis*`, `editors/vscode/` |
| C7.8 | Provider metadata consumption, MIR provider calls, generated-Rust provider bindings, native host capabilities, runtime/provider conformance |

- **C8 must not add or modify provider ABI or MIR runtime-surface entries.**
- **C7.8 must not alter LSP protocol or editor-facing behaviour**, except where exposing
  already-approved diagnostics.
- **Changes to common MIR enums require coordination** — C8 compiles against `Callee` and `MirTy`
  even though it does not semantically use provider calls, so A10's added variant is a
  cross-track change even where it is not a cross-track *semantic* change.
- **Shared roadmap/state files.** No lease mechanism exists in the charter or roadmap today, so
  the operative rule is the weaker one already in use: updates to `COMPILER-STATE.md` are
  **additive to distinct sections**, never rewrites of a shared one, and the two tracks append
  under their own headings. If a lease mechanism is wanted, it needs to be specified before it can
  be cited.

## GATE C7 — CLOSED (CD-274, final owner ruling)

Full consolidation: `STARKLANG/docs/compiler/GATE-C7-CLOSURE.md`.

```text
GATE C7: CLOSED
P1: TIER-1 QUALIFIED
C7.5 SIZE: MEASURED
C7.5 RUNTIME: NOT MEASURABLE — NO CLAIM
NATIVE PATH: USABLE FOR THE ADMITTED WORKLOAD
FULL CORE/NATIVE CONFORMANCE: NOT CLAIMED
```

**Two commits, two evidentiary roles.** `d735b35` qualifies the P1 execution matrix — six rows,
three platforms, debug and release. `c5a97bfd918a3af1e293a4b5d0114d0ea8cbf084` (`c5a97bf`) qualifies the complete C7 tree.
`d735b35` is not the gate-qualifying commit and must not be cited as one.

**The supported claim**, and nothing wider:

> STARK has a usable generated-Rust native build path for its admitted workload. It builds and
> executes in debug and release on Linux x64, macOS arm64 and Windows x64; supports the first-party
> process, time and synchronous TCP capabilities required by the frozen P1 HTTP/JSON REST workload;
> preserves MIR-owned move-only resource lifecycle; and passes six Tier-1 P1 execution rows
> consisting of byte-exact HTTP exchanges and bounded clean exit. Executable-size profile effects
> are measured. No steady-state runtime, throughput, complete Core-library, unrestricted host-I/O or
> universal native-conformance claim is made.

**Not claimed:** steady-state runtime or throughput; native Core `File`; TLS/HTTP2/HTTP3/UDP/async
I/O/event loop/DNS/unrestricted FFI; universal Core-to-native conformance; usage-shape qualification
for reference-returning and borrow-retaining APIs (separately owned, and **not** retroactively
absorbed).

**Gate transition.** C7 no longer blocks roadmap work. The performance instrument is follow-on work,
not gate repair. `stark-io` and further host packages go through their own provider/package
qualification. WP-C7.9 continues to govern three-engine adversarial corrections. Future native
capability claims must retain the evidence distinctions this gate established: a build is not
execution evidence; a green component test is not whole-path evidence; cross-platform support is not
inferred from one host; and a runtime number is not a backend-performance result when fixed harness
costs dominate it.

## Gate C7 — RULING (CD-273): CLOSES WITHOUT A STEADY-STATE PERFORMANCE CLAIM

**P1 is Tier-1 qualified.** All six execution rows are green at `d735b35` — linux-x64, macos-arm64
and windows-x64, each in debug and release, each **executing** the artefact through 24 byte-exact
HTTP exchanges and a bounded clean exit, not merely building it.

**C7.5 closes with size measured and runtime explicitly not measured.**

```text
Executable-size profile effect:
    MEASURED — release materially smaller than debug (1.686x on P1).

Micro-workload runtime profile effect:
    NOT MEASURABLE — dominated by process-startup floor.

P1 REST end-to-end runtime profile effect:
    NOT MEASURABLE — dominated by harness startup, deliberate delay,
    process supervision and loopback exchanges.

Backend steady-state runtime claim:
    NONE.

Future measurement:
    requires a separate amortised or internally instrumented benchmark;
    the frozen P1 qualification workload will not be modified.
```

**The gate does not wait on a performance instrument.** An honest absence of a runtime claim beats a
number produced by a harness already known to be invalid. The `1.003x` debug/release ratio is not
evidence that the profiles perform alike — it is evidence that the measurement was of the harness.
`321 req/s` and `66 ms` must not be quoted as STARK server throughput.

**P1 stays frozen at 24 exchanges.** Extending it would fuse functional qualification with
performance measurement and make the workload's identity depend on benchmark requirements. The
instrument is a separate versioned artefact — specified in `WP-C7.5-PERFORMANCE-REPORT.md` §8, which
extracts `handle_request_bytes` and replays the frozen corpus in-process — and is **follow-on work,
not gate repair**.

### Gate state

| condition | status |
| --- | --- |
| native builds usable for admitted workload | MET |
| native host capability exists | MET |
| P1 implementation | MET |
| P1 Tier-1 qualification | **MET** — six execution rows green (CD-273) |
| C7.5 executable-size dimension | **MET** |
| C7.5 steady-state runtime | **EXPLICITLY NOT MEASURED** — no claim attached |
| native Core `File` support | KNOWN LIMITATION / DEFERRED (SELECT-C) |
| `DEFECT-C788-LOOP-TEMP` | DISCHARGED — fixed by A12 (CD-265) |
| resource lifecycle matrix | COMPLETE — 9 observed, 1 unreachable |

Remaining: final evidence consolidation and the closure ruling itself.

## DEFECT-C788-LOOP-TEMP — FIXED (CD-265, MIR amendment A12)

**Closed.** `Statement::StorageDead(Place, StorageEnd)` ends a local's storage where lowering knows
its units are all accounted for; `MIR_VERSION` `0.2` → `0.3`, runtime surface unchanged. Sixteen
shapes probed for MIR/native agreement and destructor counts, all agreeing.
`repeated_connect_and_release_reuses_slot_state` is un-ignored, its `CLASSIFIED_IGNORES` entry
removed, and `c788_lifecycle_e2e` is 9 passed / 0 ignored. Full argument:
`STARKLANG/docs/compiler/mir-amendment-A12-storage-end.md`.

**The recorded scope was too narrow, and the fix corrected it.** CD-263/264 said the defect affected
a temporary and not user locals. Measured: a user local with one field moved out inside a loop aborts
identically, with no `match` in the program. The root cause is any place whose storage is emptied
piecewise — a sub-place move or a field-precise drop — which nothing then finished. CD-264's
non-blocking verdict is unaffected (P1 uses whole-value bindings), but its stated reason was
narrower than the truth.

**Open for the owner:** A12 was implemented without a prior ruling, under CD-264's commission to fix
the defect "compiler-wide rather than TCP-specific". The charter records that changes to common MIR
enums require coordination because C8 compiles against them. C8 does not match on `Statement`
exhaustively today, so nothing breaks — but whether A12 should carry a retrospective CE-numbered
approval is a governance question, not an engineering one. See the amendment's §8.

## DEFECT-C788-LOOP-TEMP — RULING (CD-264): NON-BLOCKING C7 DEVIATION, MANDATORY NEAR-TERM FIX

**Does not block Gate C7 closure. Becomes a mandatory near-term compiler defect at P1 compiler
priority** — high priority, *not* the P1 workload; the two senses of "P1" are unrelated.

Classification:

> **C7 non-blocking known defect; mandatory before native resource support is declared generally
> usable beyond the admitted P1 workload.**

| question | ruling |
| --- | --- |
| blocks P1 qualification? | **no** |
| blocks C7 closure? | **no** |
| lifecycle matrix fully complete? | **yes, since A12 (CD-265)** — 9 observed, 1 unreachable by construction |
| may remain indefinitely? | **no** |
| must be fixed before a broad native-resource completeness claim? | **yes** |
| must be fixed before a public release recommending resource-producing calls in loops? | **yes** |

**Why not blocking.** C7's admitted closure question is whether the selected native path is usable
and qualified for the frozen workloads. P1's 24-accept REST loop executes; its resources are held in
user bindings whose lifecycle works; eight lifecycle cases pass; `?` propagation, early return,
call-boundary movement and independent listener/stream closing all work; reproduction needs a
compiler-generated temporary shape P1 does not emit. Making it blocking would retroactively change
the gate from *prove the admitted native workload and its required resource surface* to *prove every
valid looping shape involving resource-bearing intermediate values* — a guarantee that matters but
was not the frozen C7/P1 criterion.

**Why not a minor deferral.** The failing program is valid source (a `while` loop around
`match connect(addr) { … }`) that aborts on its second iteration. The defect is generic — repeated
provider operations, resource-producing expressions in loops, any future `Result<Resource, E>` or
`Option<Resource>` API, and confidence in exactly-once lowering for reusable control-flow regions.
TCP merely exposed it.

**Safety reading.** The compiler fails closed with a compiler-defect diagnostic rather than silently
overwriting a live resource. This is a language-correctness and availability defect, **not** a
demonstrated silent double-close, use-after-move or ownership corruption. That fail-closed behaviour
is the strongest reason it is admissible as a known limitation.

**Fix boundary** — compiler-wide, not TCP-specific:

> Every non-`Copy` compiler-generated temporary that may be assigned again must be proven dead
> before the next assignment. If live, lowering must emit the appropriate Drop or move-out
> transition on every predecessor edge.

Eight-point investigation scope and gate wording: `c78/closure-gate-slice7.md`;
`repeated_connect_and_release_reuses_slot_state` is the primary regression test and is unignored
(with its `CLASSIFIED_IGNORES` entry removed) by the fixing change.

**Final C7 position:** close C7 once the remaining cross-platform qualification and C7.5
measurements pass; carry `DEFECT-C788-LOOP-TEMP` as an explicit high-priority deviation, not a
hidden gap and not a C7 blocker.

## Gate C7 — RULING (CD-262): QUALIFICATION-BLOCKED, NOT CAPABILITY-BLOCKED

**Condition 1 ("native builds usable") is MET for the admitted C7/P1 scope.** A real STARK
application compiles from ordinary source, uses environment and TCP capabilities, lowers through
provider-aware MIR, links native providers and runs a non-trivial HTTP/JSON workload — stronger
evidence than the presence of every standard-library I/O type.

**The ruling separates two questions that had been conflated.** "Can native builds perform useful
host I/O?" (yes — args, env, time, TCP) is a *usability* criterion. "Does every Core I/O abstraction
execute natively?" (no — `File`) is a *completeness* criterion. Conflating them would silently expand
C7 from "usable native build path" to "complete native Core library", which is not what P1 tested —
P1 required TCP and environment, not filesystem.

**Core `File` is a known scoped limitation**, intentional under SELECT-C, not an unimplemented
capability, and does not hold the gate open.

| condition | status |
| --- | --- |
| native builds usable for admitted workload | MET |
| native host capability exists | MET |
| P1 implementation | MET |
| P1 Tier-1 qualification | PARTIAL |
| C7.5 deferred measurements | OPEN |
| native Core `File` support | KNOWN LIMITATION / DEFERRED |
| `DEFECT-C788-LOOP-TEMP` | DISCHARGED — fixed by A12 (CD-265) |
| **C7 overall** | **OPEN — QUALIFICATION REMAINS** |

Critical path: Linux x64 P1 run; Windows x64 P1 run; C7.5 steady-state runtime; C7.5 debug/release
comparison; final consolidation and closure ruling.

## Gate C7 — REASSESSED after C7.8 (CD-261): still open, for a narrower reason

C7.8 changes two verdicts. **Native I/O exists and executes from ordinary source** — args/env,
monotonic time, TCP bind/accept/connect/read/write — so the 2026-07 assessment's central claim
("`stark-runtime/src` has no file, network, time or environment module at all") is superseded.

**But its probe still holds, re-run rather than assumed.** A source-level `File::create` program
still fails with `native build does not yet support this program: type Core(File, []) (C4.5)`. The
backend emitting `OwnedResourceHandle` for `MirTy::Core(File, ..)` does **not** make such a program
buildable — the refusal is upstream of emission, and I asserted otherwise from inspection before
checking, which was wrong.

That refusal is now a **decision** rather than an omission: SELECT-C (CD-253) keeps `File` on the
legacy path unconditionally, because migrating it would make MIR identity depend on build
configuration.

**The block has changed shape.** The old assessment said P1 was "waiting on native capability".
It is not: the P1 REST workload is built on TCP and environment lookup, needs no `File`, and
self-assesses `P1 PARTIAL — Tier-1 cross-platform runs remain`. What remains is qualification —
cross-platform runs for P1, and C7.5's two deferred measurements which were blocked on P1 existing.

**One question is the owner's**, and is deliberately left open: whether "native builds usable"
requires the standard library's own `File`, or is satisfied by the provider capabilities P1
enumerates. That reading decides whether condition 1 can move to MET.

## Gate C7 — EXIT ASSESSMENT (CD-195): CANDIDATE-COMPLETE, BLOCKED BY P1

**Gate C7 does NOT close.** Of its four exit conditions, two are met, one is partial, and one is not
met. Full assessment in `STARKLANG/docs/compiler/work-packages/WP-C7.7-GATE-EXIT.md`.

| condition | verdict |
| --- | --- |
| native builds usable | **PARTIAL** — usable for Core-v1 compute; native I/O does not exist |
| reproducible to the documented degree | **MET** — per artefact, profile AND platform |
| performance claims bounded by measured evidence | **MET** — six of eight dimensions measured, two declared unmeasurable |
| P1 complete | **NOT MET** |

### The blocking fact, stated so it is not mistaken for a scheduling problem

`stark build` refuses any program touching `File`:

```
error: native build does not yet support this program: type Core(File, []) (C4.5)
```

`File` was already recorded EXCLUDED at Gate C6 closure ("deferred to the I/O gate", above). What
Gate C7 adds is the consequence: **that exclusion is what blocks C7.** P1's exit criteria —
arguments and environment, file read/write, monotonic time and sleep, TCP listener and stream — are
made almost entirely of surface that does not exist natively. `stark-runtime/src` has no file,
network, time or environment module at all. So P1 is not waiting to be scheduled; it is waiting on
native capability, and C7.5's remaining measurements are waiting on P1.

"C7 is done except P1" would be the wrong summary. The native path C7 delivered cannot yet run the
class of program P1 requires.

### What C7 delivered

| WP | outcome |
| --- | --- |
| C7.0 (CD-185) | baseline: host Cargo/rustc is 65-68 % of a cold build |
| C7.1 | `--release`, `--target`, profile-aware layout, target preflight |
| C7.2 (CD-187, 190, 191) | path remapping; reproducibility classified per artefact, profile and platform |
| C7.3 (CD-188, 189) | bounded build cache, size-capped LRU, 2.0x median rebuild |
| C7.4 (CD-192) | baseline MIR optimisations — measured to fire ZERO times on real workloads |
| C7.5 (CD-193) | performance report; two of eight dimensions declared unmeasurable |
| C7.6 (CD-194) | DEFER LLVM, **CE6 unopened** |

Two of those produced findings that CONSTRAIN what may be claimed rather than expanding it — C7.4's
inertness and C7.5's unmeasurable dimensions — and both are recorded as findings rather than
failures. Three over-generalised reproducibility claims were caught by CI during C7.2 and corrected;
the per-platform table is what replaced the habit that produced them.

### Re-opens when P1 exists

1. **WP-C7.5** — steady-state runtime, the debug/release runtime ratio, and a defensible
   interpreter/native ratio become measurable for the first time.
2. **WP-C7.4** — whether the folding passes ever fire on realistic code.
3. **WP-C7.6** — whether a generated-code deficit appears that would justify opening CE6.

## Position
**Gate C5 and WP-C5.6 CLOSED 2026-07-23 by owner directive CD-077.** Verdict:
**NATIVE-CORE-MVP-WITH-LISTED-DEVIATIONS.** The production `stark build` pipeline, relocated
three-package reference workspace, exact C5-native snapshot replay, 188-test focused matrix,
1,098/0/2 complete workspace run, runtime-version checks, formatting, strict clippy, and hosted CI
are green against qualification head `19254086d5f71db169fd1a1020bf30bddd284686`. The exact
supported subset, explicit String/output delta, deferred native features, toolchain identity,
artifact contract, and evidence are frozen in `starkc/docs/compiler/C5-exit-report.md`. Gate C6 is
not automatically open; an owner-approved C6 entry plan is next.

**WP-C5.3 (aggregates, enums, error values, Drop, layout) CLOSED 2026-07-23** by owner directive
after the adversarial review dispositions (CD-070). Sub-packages: C5.3a (CD-056), C5.3b, C5.3c
(CD-061), C5.3d-0 (CD-059), C5.3d-1a (CD-063), C5.3d-1b (CD-064), C5.3d-1c + C5.3d-1 (CD-066), the
`Copy` consolidation fold-in (CD-065), C5.3e (CD-067) with DEV-100 fixed (CD-068) and the corpus
re-pinned to 1.3.0 (CD-069). Every §14 exit dimension is discharged with three-engine agreement:
aggregate values, payload variants, match paths, `Option`/`Result`, `?`, the dedicated Drop
fixture (seven observable properties), and exact layout-query values under the versioned
`stark-64-v1` contract. Two bounded boundaries are recorded and enforced deterministically before
rustc rather than left latent: multi-unit enum payload partial moves (CD-070) and the wider
non-`Copy` cross-block cases, both deferred to C6. WP-C5.4 subsequently closed linkage and
function values under CD-071..CD-075.
The two open C5.3-adjacent items carried into the C5.4/C5.6 reviews are DEV-098's defensive
reborrow reasoning and the C6-deferred ownership boundaries.

Gate: **C5 (native compilation) — CLOSED 2026-07-23 (CD-077). WP-C5.1 CLOSED 2026-07-21 in full** (entry plan CD-042,
WP-C5.1a CD-043, WP-C5.1b CD-044, WP-C5.1c CD-045 drafted/CD-046 approved). **WP-C5.2 (scalar
native lowering) CLOSED 2026-07-21 in full**: C5.2a (CD-047), C5.2b (CD-048), C5.2c (CD-049),
C5.2d (CD-050), C5.2e (CD-051), and the §14 exit condition discharged by the three-engine
differential harness (CD-053). Gate **C4 CLOSED 2026-07-21** by owner directive, after the last blocker
(DEV-089) was resolved
rather than
deferred. The full WP-C4.7 close-out landed in two directives: the first (CD-038/039/040)
implemented DEV-086, deferred DEV-083, ratified surface revs 11/12, and refreshed the corpus to
1.2.0; the second (this one) resolved DEV-089 and the two residual over-rejections. Final
validation: workspace tests green, `cargo fmt` clean, `cargo clippy` clean on 1.93 and 1.97, corpus
1.2.0 lock integrity green, frozen-corpus + differential suites green.

**WP-C5-ENTRY.md APPROVED 2026-07-21 (CD-042).** The Gate C5 implementation-ready plan is checked
into `STARKLANG/docs/compiler/work-packages/WP-C5-ENTRY.md` and approved at its recommended
decision-table choices: generated Rust backend consuming verified MIR (per CD-026), debug-only
profile, concrete-monomorphised-instances-only generics, `MaybeUninit<ManuallyDrop<T>>`-style
non-`Copy` storage with explicit MIR-directed Drop glue, isolated unsafe helpers only, Cargo
invoked internally by `stark build`, local/pinned generated dependencies, and Native Provider ABI
v0.1 specified in WP-C5.1c without execution being required for the MVP. Next: WP-C5.1a
(representation decision write-up already covered by the entry plan's §6-10) proceeds straight to
WP-C5.1b (backend/runtime skeleton) once the frozen C5 reference workspace (§4) is named and its
HIR/MIR baseline snapshot is green.

**DEV-089 — RESOLVED by implementing user `Display` dispatch in both engines** (owner decision,
2026-07-21). `print`/`println`/`eprint`/`eprintln` are generic `<T: Display>` functions that
dispatch to the argument's own `Display::fmt`. Spec: **PRINT-DISPLAY-001** (06-Standard-Library,
nine-point contract); prelude + IO signatures and STD-FORMAT-001 updated to match. Oracle:
`display_text`/`finish_display` run the impl and destroy the by-value argument after its bytes are
submitted. MIR: `lower_print_display` — a static `Callee::Instance` call to `fmt`, then the
existing `StringAsStr` + `Print(ln)Str` surface, then visible `Drop`s. **No new MIR shape, no new
`RuntimeFn`, no runtime-surface bump** (`MIR_RUNTIME_SURFACE` stays `0.1-A8`). Eight differential
tests + checker positive/negative coverage.

**Two residual over-rejections made consistent and deferred** (not gate blockers under the
six-clause rule): **DEV-090** (split from DEV-086) — by-value iteration over a non-`Copy` array
element now rejected in the front end (`E0104`, `borrowck.rs`) before either engine, deferred to a
later language-completion package; **DEV-088 use-site** — using a `const` declared in another file
now rejected in the checker (`E0215`) before either engine, deferred to the front-end/multi-file
completion package with DEV-083. Both reject at a single deterministic point rather than diverging
between engines. The six-clause stopping rule (CD-040(c)) now holds in full — clause 3 ("no known
engine divergence remains") satisfied by DEV-089's resolution.

(Previously: C4 NOT CLOSED pending the DEV-089 decision; the bounded validation had surfaced it as
an engine divergence and §6 required stop-and-report.)
**Frozen corpus grown to `corpus_version` 1.1.0 (CD-037, owner-directed, ADDITIVE)** — five new
cases covering every construct the Class-A campaign and WP-C4.7 added; 22 cases, all agreeing
across both engines. Writing them found and closed **DEV-087** (the oracle treated a slice
reference as non-`Copy`, so passing one to a function consumed it) — the fourth defect in this
package that lived only in the gap between two engines. Decision-table item 4 is now discharged;
items 1, 2, 3 and 5 remain with the owner.
Report: `WP-C4.6.md`, final section "Gate C4 Closure (WP-C4.7 close-out, 2026-07-21)", which
records the closure under CD-041 and supersedes both the 2026-07-19 Verdict and the earlier
"Gate C4 Exit Report (WP-C4.7-9)" recommendation. **The gate is now CLOSED (see the Position
header); the text below this line is the historical pre-closure record.**
Recommendation in the report: **close C4, conditional on the owner disposing of DEV-086 and
DEV-083 by explicit dated decision** rather than leaving them undisposed. Exit conditions 1 and 3
are satisfied outright; condition 2 is satisfied except for those two over-rejections, which are
recorded, bounded, consistent across engines, and blocked on DECISIONS (a CE3 shape question and a
method-resolution design question) rather than on effort. The report also states the
counter-argument plainly: the defect-discovery rate has not visibly plateaued — 13 defects found
in this package, 11 of them in already-signed-off code — which is a fact about risk into C5.
Owner decision table (report §6): DEV-086, DEV-083, post-hoc ratification of surface revs 11/12
(`0.1-A7`/`0.1-A8`), whether to grow the frozen corpus (a `corpus_version` bump is
governance-controlled and was deliberately not touched), and gate closure itself.
**WP-C4.7-9 AUDIT SWEEP DONE 2026-07-20 — and it found six more items, as forecast.** Every
`unsupported(` site in `lower.rs` was enumerated, partitioned defensive-vs-construct, and each
construct candidate probed against BOTH engines. Owner-directed fixes for four of them landed:
**DEV-084** (`print`/`println` accepted ANY type — three engines gave three answers for a program
06 says is invalid; the CHECKER was the wrong one and now rejects), **DEV-085** (`for` over an
array: checker accepted, oracle ran, MIR alone refused — now lowers), the **trait-default method
with own generics** gap that WP-C4.7-8.4 left behind (both the checker's default-fallback path and
`FnKey::TraitDefault::method_args`), and the **droppable array pattern**, which turned out to need
a CE3 shape change and is recorded precisely instead (**DEV-086**).
Correctly reserved, not blockers: `HashMap::values`, `Vec::contains`, `String::insert` (std-full,
CD-033); or-patterns (**not in 02's Pattern grammar** — the parse error is correct).
Workspace 798/0/2. Frozen corpus green.

**WP-C4.7-8.4 DONE 2026-07-20 — method-own generic parameters, the last implementation item.**
Two halves had to meet: the checker instantiated only the IMPL's parameters, leaving a method's
own `U` a rigid `Ty::Param` no argument could unify with; and MIR could not monomorphise a method
at arguments the impl does not mention. `FnKey::ImplFn` now carries `method_args` beside the
impl's `type_args`, filled from a per-call-site record keyed by the call expression — the method
equivalent of C4.5c's machinery for top-level generic fns. **`FnKey` appears ZERO times in
`mir.md`**, so extending it is not a contract change and needed no CE3 (the plan asked for this to
be verified and stated). Symbols gain a second bracket for method args and stay injective; §2
already declares them non-ABI. Workspace 795/0/2.

**WP-C4.7-8.5 DONE 2026-07-20 — non-bare impl heads.** `impl<T> Holder<Option<T>>` now applies to
`Holder<Option<Int32>>` in BOTH engines. The checker's impl matching bound a parameter only when
it stood ALONE as a type argument and otherwise demanded `types_equal`, so `Option<T>` vs
`Option<Int32>` failed and every non-bare head was invisible (E0302). Replaced with `unify_impl_ty`
— one-way structural unification, parameters bound from the IMPLEMENTATION side only, with
consistency enforced when a parameter recurs (`Pair<T, T>`). Lowering gained the matching
`bind_written_impl_arg`, because the two must agree about which impls apply or the front end would
admit programs lowering then rejects — the DEV-079 failure shape. **DEV-083 recorded, not fixed:**
a CONCRETE position in an impl head still cannot match a receiver argument that is an unresolved
inference variable at resolution time; fixing it needs speculative binding during candidate
search, which can select the wrong impl and is a semantics change, not a bug fix. Narrow
over-rejection with a workaround (annotate the receiver). Workspace 794/0/2.

**OWNER DECISION 2026-07-20: implement 8.6, 8.5 and 8.4, then audit.** All three are normative
Core by the grammar and the abstract machine — `02:64`+`02:120` put `GenericParams?` on methods,
`02:117` admits any `Type` as an impl self type, and REF-SLICE-001 states that "writes through an
exclusive slice reference update the original object" — so under CD-033's strict reading
(deliberately chosen over the workload-subset reading) none of them may be silently deferred.
**WP-C4.7-8.6 DONE 2026-07-20 — exclusive slice views, surface `0.1-A7` → `0.1-A8` (A1 rev. 12).**
`SliceNewMut` yields `&mut [T]` from an exclusive receiver borrow; the interpreter's WRITE path
now composes a `Slice { start, len }` window with a following `Index(i)` exactly as its READ path
already did, which is what makes a write through the view reach the base object. Verifier: an
exclusive receiver is required (MIR-0012 otherwise); `len`/`is_empty` accept either mutability
since they only read. **DEV-082 found and closed:** `method_receiver` had no slice/array arm, so a
method call on a slice CONSUMED the receiver — harmless for `&[T]` (shared refs are `Copy`, which
is why shared slices shipped clean in A4-2e) but a real move for `&mut [T]`, making
`s.len(); s[0]` fail E0100. Invisible until exclusive views existed to expose it. Lowering
likewise now reads such a receiver by `Copy` — the MIR-level shared reborrow — instead of moving
it. Workspace 793/0/2.
**WP-C4.7-8.3b DONE 2026-07-20 — droppable scrutinee under NESTED patterns.** A consuming match
decomposes the scrutinee completely, so every leaf the pattern DISCARDS still owes a destructor.
`consume_unbound_leaves` generalizes C4.5d's flat rule to an arbitrary pattern tree (wildcards,
unmentioned struct fields, nested tuples/variants → arm-scoped temps), running BEFORE the binding
walk so reverse-registration order yields the oracle's order: bindings first (reverse binding
order), discarded leaves after. **A third pre-existing defect surfaced — DEV-081:**
`bind_shorthand` never registered a shorthand struct-field binding as droppable in ANY mode, so
`P { a, b }` moved the fields out and destroyed neither. A LEAK, not a double drop, which is
precisely why it had failed silently — no verifier rule broken, nothing to assert on, invisible
unless a destructor prints. It affected the FLAT path too, before 8.3b existed. Workspace 792/0/2.
**WP-C4.7-8.3a DONE 2026-07-20 — DEV-079 + DEV-080, both found while pinning oracle behaviour for
8.3 and both in the FLAT match path that A2/C4.5d had signed off.**
*DEV-079:* V-MOVE-1 collapsed every non-`Field` projection to the whole local, so moving a second
payload field out of an enum local read as a second move of the same place. **Every enum variant
with two or more droppable payload fields produced MIR that lowering accepted and verification
rejected** (MIR-0007) — an internal inconsistency between two components meant to be independent
readings of one contract, and strictly worse than a clean `Unsupported`. `VariantField(v, i)` now
contributes two path components, so siblings are distinguishable; `Deref`/`Index` still collapse.
*DEV-080:* fixing that immediately exposed a drop-ORDER divergence it had been masking — with a
mix of bound and wildcard payload fields, MIR used reverse-FIELD order while the oracle destroys
all bound bindings first (reverse binding order) then the discarded leaves. `consume_variant_payload`
now consumes unbound fields first and bound second, so reverse-registration yields the oracle's
order. Workspace 789/0/2.
**WP-C4.7-8.2 DONE 2026-07-20.** A user `Iterator` with a droppable `Item` now lowers: each
yielded value is destroyed at the END OF ITS OWN ITERATION, not accumulated to loop exit —
pinned against the oracle before any lowering was written. `break` destroys the current
iteration's value before leaving and `continue` before looping back, and both fall out for free
from one ordering decision: capture the loop's `scope_depth` BEFORE pushing the per-iteration
scope, so the existing break/continue handling (which drops every scope from `scope_depth`
onward) covers them with no special casing. Pushing the scope first would have leaked the value
on `break`. Workspace 787/0/2.
**WP-C4.7-8.1 COMPLETE 2026-07-20 (MIR half).** `unwrap_or` over a droppable payload/default now
lowers, matching the timing pinned against the corrected oracle: the DISCARDED value is destroyed
**at the call**, not at scope exit — on `Some`/`Ok` the payload is yielded and the default dropped
there; on `None` the default is yielded; on `Err` the default is yielded and the displaced error
payload dropped. The blocker was that consuming a payload out of a **drop-tracked** local through
a `VariantField` projection is refused outright (C4.5d). The fix is the one `lower_match` already
uses: materialize the receiver into a fresh temp first — the move clears the source's drop flags,
and a temp is never auto-dropped, so ownership transfers exactly once. Reusing that discipline
rather than inventing a second one is what made this small. Non-droppable lowering is unchanged
byte-for-byte. Workspace 785/0/2.
**WP-C4.7-8.1a DONE 2026-07-20 — DEV-076 CLOSED (oracle half).** `Option`/`Result::unwrap_or`
double-dropped the payload and never dropped the discarded default — a SOUNDNESS defect, same
root cause as DEV-077: it was handled on the borrowing method path, which operates on a CLONE, so
taking the payload emptied the clone while the original kept it and destroyed it again at scope
exit. It now consumes the real place and explicitly drops whichever value it discards.
**Pinned timing, which is what the MIR half must match and is not the obvious answer:** the
discarded default is destroyed **at the call**, not at end of scope —
`let t = Some(Tag{1}).unwrap_or(Tag{2})` observably prints `2` then `1`, where the defect gave
`1` twice and no `2`. The MIR half stays a clean `Unsupported` for now: moving a payload out of a
**drop-tracked** local through a `VariantField` projection hits the C4.5d guard, so it needs the
drop-flag machinery — real work, and now writable against a correct oracle rather than against a
double drop.
**DEV-075 CLOSED 2026-07-20 under an owner SPECIFICATION decision — the first spec change of
WP-C4.7.** The owner split the two types rather than treating them as one gap: **`Char`** is
totally ordered by **Unicode scalar value** (`Eq`+`Ord`+`Hash`; all four ordered operators;
`Char::cmp`), explicitly not collation; **`Bool`** is `Eq`+`Hash` but **not `Ord`**, so its
ordered operators and `Bool::cmp` are compile-time errors while `==`/`!=` stay valid. MIR was
already directionally right for `Char`, so the ORACLE was aligned to it (the divergence ran that
way round). New **`PRIM-TRAIT-001`** in 06-Standard-Library gives the full primitive
trait/operator matrix, replacing the illustrative `impl Eq for Int32` + "similar for other types"
that had been the only authority; 03's operator table cross-references it; compiled spec
regenerated and the fixture corpus re-extracted (manifest in sync).
**The matrix had to make one distinction explicit:** for primitives, operators have built-in
meaning and do NOT dispatch through the traits — `Float64` admits `<`/`==` as IEEE operations
while implementing neither `Eq` nor `Ord` (IEEE comparison is not an equivalence relation or a
total order), so it cannot satisfy `T: Ord` or key a `HashMap`. Conflating the operator gate with
the trait gate silently broke ordinary float comparison once during implementation; both
directions are now pinned.
**WP-C4.7-6.3 DONE 2026-07-20 (owner-decided: a real conformance defect, fix it) — DEV-078.**
An unsuffixed integer literal now ADOPTS an expected integer type. 03 says expected types flow
inward from annotations, **function parameters**, fields and assignment destinations, and that
step 5 defaults only an **unconstrained** literal — the checker was committing every literal to
`Int32` at the literal itself, before any expectation could reach it, so `v.get(0)`,
`takes_u64(0)`, `let a: UInt64 = 9` and a `UInt64` field initializer were all rejected. Fixed as
**general inference**, not a `Vec::get` special case: literals take integer-KINDED inference
variables, unification carries the expectation in, and step 5 is a real defaulting pass running
after all bodies and before the deferred bound checks. Binding range-checks (`takes_u8(300)` is
E0008); the kind restriction stops a literal standing in for a `Bool`; and because this is
propagation rather than coercion, a suffixed literal (`0i32`) and a typed `Int32` value both still
fail against `UInt64`. Method receivers and cast operands settle eagerly (they branch on a
concrete type with nothing later to wait for). **Subtlety:** a literal variable is often bound to
ANOTHER variable (`MyOpt::Some2(7)`), so defaulting must resolve first and default the end of the
chain — defaulting only variables absent from the substitution left such chains unbound, and they
escaped to MIR as `type Infer(N)`. Unnecessary `as UInt64` casts removed from the corpus.
Workspace 778/0/2; clippy clean 1.93/1.97.
**WP-C4.7-6.1 DONE 2026-07-20 (owner-decided, option (a)).** `Box<T>` reaches MIR as an OPAQUE
OWNING runtime type: `RuntimeFn::BoxNew`/`BoxIntoInner`, surface **`0.1-A6` → `0.1-A7`** (A1
amendment rev. 11), `MirTy::Core(Box, [T])` — **no new `MirTy`**, and deliberately NOT lowered
transparently as `T`. Drop goes through the existing `Drop` terminator's structural glue (no
public box-drop op): dropping a box destroys the contained `T` exactly once, `into_inner`
transfers it out without dropping. The audit's "`Box` deref" entry is **corrected**: Core v1 has
no `Deref` trait, TYPE-METHOD-002 peels only `&`/`&mut`, and 06 gives `Box` exactly
`new`/`into_inner` — so `*box` is spec-conformant to reject, now pinned by a negative test.
**Three pre-existing defects surfaced while implementing it:** (1) drop-instance discovery never
descended into `Core` container type arguments, so a `Box<Tag>`'s `Drop` terminator fired and
silently found no destructor; (2) that walk had no cycle guard, and `Box` makes types recursive —
`Node -> Option<Box<Node>> -> Box<Node> -> Node` overflowed the stack; (3) **DEV-077**, an oracle
double-drop in `Box::into_inner` (it operated on a CLONE of the receiver), fixed and closed here.
Workspace 775/0/2; clippy clean 1.93/1.97.
**DEV-076 OPENED (blocking WP-C4.7-8.1):** the oracle's `Option::unwrap_or` double-drops the
payload and never drops the discarded default — found by pinning drop timing BEFORE writing 8.1's
lowering, per §0.6. MIR must not be built to match it; the oracle is fixed first.
**WP-C4.7-7 DONE 2026-07-20 — DEV-067 and DEV-071 CLOSED.** With these, **every front-end
deviation the C4 track owned is closed**; the only open deviations are the four long-standing
unscheduled ones (DEV-005/010/011/012/017) plus DEV-075, opened yesterday by C4.7-6.2.
*DEV-071*: the prelude `Ordering` is `Ty::Core(CoreType::Ordering)` with `Res::Builtin` variants —
structurally like `Option`/`Result` and invisible to the `Ty::Enum` machinery for the same reason,
but unlike those two it had never been given an explicit arm, so it hit WP-C1.5's "unknown domain,
require a wildcard" default. Now tracks all three variants; a two-variant match is still E0303.
*DEV-067* was two causes, one per symptom: **(b)** the bounded-parameter method lookup tested the
UNPEELED receiver, so it matched `t: T` but never `t: &T` — TYPE-METHOD-002 requires the peel, and
the concrete-type path right below already computed one; the peel simply happened too late.
**(a)** `satisfies_bound` had **no `Ty::Param` arm at all**. Adding it was not enough: bound
obligations are verified in a DEFERRED pass that runs after every body, so `current_fn_generics`
belonged to whatever was checked last — each obligation now carries the generic environment it
arose in. Nothing newly accepted: a concrete type without the impl, and an unbounded parameter
forwarded into a bounded position, both still E0500 (pinned). Workspace 769/0/2; clippy clean.
**WP-C4.7-6.2 DONE 2026-07-20 — primitive `Ord::cmp`.** 06 specifies `impl Ord for Int32 {
fn cmp }` "and similar for other types" and `Ordering` is `core-min` prelude, but `3.cmp(&5)`
failed E0304, so a user `Ord` impl was the only way to obtain an `Ordering`. Added across all
three engines: checker surface returning `Core(Ordering)`; oracle via the existing `Ord for
Value` (the same comparison `<` uses); MIR via a new `lower_primitive_cmp` that CONSTRUCTS the
`CoreOrdering` variant from the comparisons `<`/`==` already lower (`StrCmp` for `String`/`str`)
— the exact inverse of `lower_user_ord`, and **no new MIR shape and no runtime-surface change**.
Scoped to integers + `String`/`str`; floats excluded per CD-015; **`Bool`/`Char` excluded because
of DEV-075** (below). Workspace 765/0/2; clippy clean 1.93/1.97.
**DEV-075 OPENED (found while scoping 6.2, pre-existing and unrelated to it):** the checker
accepts `<` on `Bool` and `Char`, but `false < true` fails in BOTH engines (accept-then-fail)
and `'a' < 'b'` **succeeds in MIR while the oracle rejects it — an engine divergence**, unnoticed
because no test compares an ordered operator on `Char`. Needs a spec reading (does 03 intend
`Bool`/`Char` to be ordered?), not just a code fix. C4-exit-report input.
**WP-C4.7-6.1 and 6.3 are with the OWNER** — see the dated record for the evidence; both findings
contradict the WP-C4.7 plan's framing of them.
**WP-C4.7-5 DONE 2026-07-20 — DEV-072 and DEV-073 CLOSED.**
*DEV-073* root cause sat one level below the two failing checks: `type_from_hir_without_diagnostics`
DROPS generic arguments, which was invisible while its only consumers compared non-generic
nominals but meant an impl's written `W<T>` became `W<>` and could never match `W<Int32>`. New
`impl_self_ty_with_args` preserves them, and both the operator-bound and for-loop-iterable checks
now unify through **`match_impl_type`** — the same one-way unification method resolution already
used, which is exactly why method calls on generic nominals worked while operators and `for` loops
on the same types did not. The iterable half also substitutes the associated `Item`
(`type Item = T` on `Repeat<Int32>` → `Int32`). **MIR needed no change** — A1 had already made
dispatch instantiation-ready, confirmed by the two differential tests this deviation had blocked.
*DEV-072*: borrowck's `match` handling inspected no patterns at all; it now mirrors MIR's
`scrutinee_reads_through_ref` exactly (so the engines agree by construction, which is what the
deviation was) and reports E0101 for any non-`Copy` binding under it, recursing through nested
and shorthand patterns. Wildcards, literals, and `Copy` bindings stay legal and are pinned by
positive tests — matching by reference is fine, only taking ownership is not. The MIR guard is
kept as documented defense in depth. Workspace 763/0/2; clippy clean 1.93/1.97.
**WP-C4.7-4 DONE 2026-07-20 — DEV-069 CLOSED** (multi-file span discipline; one root cause, not
four bugs: all three engines read spans against a single "current file", right for the item being
CHECKED and wrong for every item LOOKED UP. `item_text` + a per-body file swap in the oracle,
which had three body-execution funnels, not one). See the dated record.
**WP-C4.7-3 DONE 2026-07-20 — MIR amendment A4 (CD-036), owner-approved under CE3 as drafted.**
`Rvalue::LayoutQuery { kind: SizeOf|AlignOf, ty: MirTy }` (pure, dest `UInt64`) replaces WP-C4.6
A4-1's type-ERASING lowering of `size_of`/`align_of` to `Const 8`. 06 classifies these as
target-layout queries and LAYOUT-QUERY-001 makes them the only Core layout observations, so a C5
backend must be able to answer them from the MIR it is handed — impossible once `T` is discarded.
Because MIR is monomorphised the recorded type is always concrete (`size_of::<T>()` in a generic
body records the instantiation's type — pinned by a test). Each consumer answers through ONE
layout service; the reference one returns the frozen `(8, 8)` for every type, so **the
representation changed and the behavior did not** — the HIR oracle was not touched and
`size_of_align_of_agree` stays green unmodified, which is the proof. Research finding:
**CD-015/WP-C2.9 fixed no per-type numbers** — it approved only that `size_of`/`align_of` are the
sole layout observations and that Core promises no ABI; LAYOUT-ABI-001 makes the values target-
and version-dependent, so real numbers belong to C5.1's target contract, not C4. Rejected a
`RuntimeFn` encoding: its only input is a type, it cannot trap, and layout is compile-time
knowledge, not backend-supplied runtime. Workspace 756/0/2; clippy clean 1.93/1.97.
**WP-C4.7-2 DONE 2026-07-20** (evidence symmetry, CD-033's evidence rule): 6 hand-built verifier
negatives covering the Class-A classes (bitwise-on-float and Pow-on-float-dest → MIR-0004;
`VecGetRef` wrong schematic dest, `CharsIterNext` wrong operand, runtime call arity → MIR-0005;
`SwitchInt` on Float64 → MIR-0004, pinning that A2's Char widening stopped at Char) and 4
clean-Unsupported fixtures pinning every pinnable Class-A residual. **Finding that changes
WP-C4.7-8's shape:** two recorded "MIR residuals" are actually **front-end-blocked** and never
reach lowering — method-own generic params (`h.first(7, 9)` → E0001 "expected 'U', found
'Int32'") and non-bare impl heads (`Holder<Vec<T>>` → E0302 "method not found"). By the §1 rule
(a MIR gap must be typecheck-clean AND oracle-supported) both are front-end work first; C4.7-8.4
and 8.5 are annotated accordingly. Workspace 752/0/2; fmt + clippy clean 1.93/1.97.
**WP-C4.7-1 DONE 2026-07-20** (doc/evidence reconciliation, no code): the WP-C4.6 A5 arithmetic
additions are now recorded in `mir.md` as MIR **amendment A3** (`MirBinOp::BitAnd/BitOr/BitXor`
pure; `CheckedOp::Pow`; `Shl`/`Shr` ACTIVE under NUM-SHIFT-001; `TrapCategory::InvalidShift` with
the interpreter's category-override rule) — **awaiting post-hoc CE3 ratification by the owner**,
since CD-033 approved the A5 class but the per-amendment recording was missed. Consequently
C4.7-3's layout amendment is **renumbered A4** (`mir-amendment-A4-layout.md`). **DEV-074** opened
and closed at creation (the A4-2e oracle slice-message alignment, previously recorded only in A1
rev. 10); ledger count 71 → 72. A4's "complete" claim tightened everywhere to "MIR runtime
surface" (front-end `core-min` holes are WP-C4.7-6).
The executor-grade plan is
`STARKLANG/docs/compiler/work-packages/WP-C4.7.md`; work it increment by increment. C4 stays
OPEN until WP-C4.7 completes and the owner approves the fresh exit report (the Class-A
requirement of CD-033 is met, but the external review + self-audit identified corrections
required before an honest exit — most notably the type-erasing `size_of`/`align_of` lowering
vs. the spec's "target-layout queries" classification (both resolved — see the WP-C4.7-3/4
records), DEV-069 as a C5 prerequisite, and the
front-end deviations DEV-067/071/072/073 + Box deref + primitive `cmp`). **A1 DONE 2026-07-20**, the
last Class-A blocker: `FnKey::ImplFn`/`TraitDefault` carry the instantiation's type args
(symbols render them — `Stack::push_item@[Int32]`); impl-generic substitution aligns the
impl's written self-type args (bare params) with the instantiation; covered: methods on
generic nominal instantiations, associated fns (instantiation INFERRED by one-way sig
unification), trait impls + defaults, Drop impls per instantiation, user `Iterator` for-loops
(desugar to `next()` instance calls; oracle already supported). Residuals clean-Unsupported:
method-own generics, non-bare impl self args, droppable Iterator Item. **DEV-073** opened
(front end: generic impls unmatched in operator-trait/iterable bound checks — both engines
reject consistently; MIR dispatch is instantiation-ready). 3 A1 differential tests; workspace
746/0; clippy 1.93/1.97 clean. Earlier same day: A2 complete (DEV-070 closed both engines,
DEV-072 opened, general pattern engine; see WP-C4.6.md). **A4 COMPLETE (all 2026-07-20):** A4-1 `size_of`/`align_of` + `unwrap_or`; A4-2a
`map`/`and_then`/`map_err` + Range-as-value (MIR tuple `(start,end,inclusive)`); A4-2b
`Vec::get`/`get_mut` (`Option<&T>`, never trap) at `0.1-A4` (A1 rev. 8); A4-2c `println(Ordering)`
(no new op); A4-2d `chars()` iteration (`Option<Char>` by value) at `0.1-A5` (A1 rev. 9);
**A4-2e slicing** at **`0.1-A6`** (A1 rev. 10): `&base[range]` over Array/Vec/slice →
trap-capable `SliceNew` (**runtime-surface only — no new MIR shape, no CE3 escalation**);
re-slicing composes windows; `s[i]` via the existing CheckIndex proof discipline against the
VIEW length; `SliceLen`/`SliceIsEmpty`; interp `ConcreteProj::Slice{start,len}` window on `Ref`
paths; shared-only (`&mut base[range]` reserved); oracle slice-bound messages aligned to the
"out of bounds" family. 13 A4 differential + 2 verifier tests; workspace 733/0; clippy
1.93/1.97 clean.
Progress: **A5, A7, A6, and A3 (Eq+Ord) DONE 2026-07-19.** A5: pure bitwise `MirBinOp`,
`~` → `^ mask`, trapping `Shl`/`Shr`/`Pow`, new `TrapCategory::InvalidShift`. A7: `loop`-break
value, `[v;n]` repeat, Unit value-position `if`/`while`/`for`. A6: Vec iteration → borrowed
cursor (V-COPY-1 dropped for the iterator ops; amendment rev. 7). A3-Eq: `==`/`!=` → `Eq::eq`
dispatch (borrow-not-move). **A3-Ord: CE3-approved Amendment A2** (`mir-amendment-A2-ordering.md`,
approved with 5 clarifications) — `EnumRef::CoreOrdering` (prelude `Ordering` as a logical MIR
enum, Less=0/Equal=1/Greater=2) across lowering/verify/interp/dump; `Ordering::Less/Equal/Greater`
construction; direct `cmp`; all four ordered ops on non-generic user nominals → `cmp` +
discriminant-compare; v3-variant → MIR-0008; generic-nominal comparison stays `Unsupported`.
`mir.md` records the C4-open additive-amendment versioning policy + `CoreOrdering` in `EnumRef`.
13 new differential + 2 verifier tests across the session; workspace 720/0; clippy clean
1.93/1.97.
(Historical note, superseded: DEV-070 was CLOSED by A2 on 2026-07-20; A4/A2/A1 all completed
2026-07-20 — see the Position header above. Open front-end deviations as of 2026-07-20:
DEV-067, DEV-069 (since CLOSED by WP-C4.7-4), DEV-071, DEV-072, DEV-073, plus Box deref,
primitive `Ordering::cmp`, and
the `Vec::get` literal-typing quirk — all inventoried in `WP-C4.6.md` "Gate closure input"
and owned by `WP-C4.7.md`.)
**WP-C4.5f-3 done 2026-07-19, closing WP-C4.5** — three sub-slices in one increment:
- **f-3a HashMap surface (`0.1-A3`, amendment rev. 6):** `RuntimeFn` HashMap group
  (New/Insert/Get/Len/IsEmpty/ContainsKey/KeysIterNew/KeysIterNext); insertion-ordered
  (CD-009) `MirValue::Vec` of `[k,v]` aggregates; `insert` returns the displaced `Option<V>`
  (honesty rule §5a — caller drops it at a visible Drop; user-`Drop` K/V refused); `get` →
  interior `Option<&V>`; `keys()` a true borrowed cursor reusing the f-2 for-desugar;
  schematic-(K,V) `map_runtime_sig`. **`collection_iter__02` differential-green.**
- **f-3b Char + assert_eq/ne (rev. 6):** `MirTy::Char` (`Constant::Int` Unicode scalar),
  `PrintlnChar`/`PrintChar`, `StringPushChar`/`StringPopChar`; `assert_eq`/`assert_ne` →
  scalar `BinOp::Eq` or `StrEq`/`StrCmp` into conditional `Trap{AssertFailure}` (message
  fidelity deferred with the e-1 boundary).
- **f-3c multi-file lowering:** `ProgramMeta` interns all source files (FileId(0)=entry),
  maps items to declaring file + module path; all cross-item name reads go against the owning
  item's file; `synthetic_spans` for generated wrappers; **module-qualified canonical symbols**
  (`helper::add_self@[]`) — package-stable linkage identity for C5. **Found DEV-069 (open,
  front-end WP):** checker + HIR oracle read cross-file spans against the entry file
  (cross-file methods/literals/field reads break); the differential test pins the
  front-end-safe subset; MIR side is multi-file-clean.
- **Exit-sweep fixes:** MIR-interp call args were bound positionally over locals `1..n`,
  clobbering interleaved drop-flag locals for callees with droppable params (bit
  `largest::<String>` in `struct_enum_trait__03`) — now bound by declared `Param(i)` kind
  with arity checks; non-place method receivers/`&expr` (call results) materialize via
  `place_or_temp`. 6 new differential tests + `entire_frozen_corpus_agrees` (all 17).
  Workspace 707/0; fmt+clippy clean 1.93/1.97.
**WP-C4.5f-2 done 2026-07-19** (by-reference Vec iteration, surface `0.1-A2` per CD-032's
dated-enumeration rule, amendment rev. 5): `VecIterNew`/`VecIterNext -> Option<&T>` (`T: Copy`,
V-COPY-1/MIR-0016); interpreter iterator = snapshot aggregate `[Vec, cursor]` in a frame local
handing out interior `&T` refs — protected by f-1's frame generations (built first,
deliberately); `for value in v.iter()` desugar; Index-on-Vec projection arms;
`MIR_RUNTIME_SURFACE = "0.1-A2"`. **`collection_iter__01` corpus case differential-green.**
Workspace 701/0/2; fmt+clippy clean 1.93/1.97.
**WP-C4.5f-1 done 2026-07-19** (both CD-030 deferrals): `Frame.generation` (monotonic) +
`MirValue::Ref` carries the pointee's generation; every deref and runtime-op ref helper
validates (slot, generation) — stale references to reused frame slots fail loudly (adversarial
hand-built MIR test: verifies by design, interpreter rejects). Projected `Move`s now TAKE with
a `MirValue::Moved` poison; any read of the hole is a loud internal error; full suite green
with the poison live confirms the tested subset never re-reads a moved place. Workspace
699/0/2; fmt+clippy clean 1.93/1.97.
**Match-drop increment done 2026-07-19** (match on owned Drop-bearing scrutinees): oracle drop
timing pinned empirically (matched arm consumes the scrutinee; bound, unbound `_`, and
catch-all payloads all drop at **arm end**). `lower_enum_match` rewritten — each arm a drop
scope; every payload field moved out of the materialized-temp scrutinee (bound → registered
binding local; unbound droppable → registered temp; catch-all → whole value), so the shell is
fully consumed (no double-drop) and everything drops at arm-scope exit; a body-moved binding
clears its flag so only the callee drops. Blanket C4.5d restriction removed. **`option_result__02`
corpus case now differential-green.** 4 new differential tests. Workspace 698/0/2; fmt+clippy
clean 1.93/1.97.
**WP-C4.5e-3 done 2026-07-19** (`?` + Option/Result methods): `ExprKind::Try` lowering
(operand in a temp consumed by both switch arms; Ok/Some payload = expr value, None/Err
early-returns the enclosing fn's Option/Result after dropping live scopes);
`is_some`/`is_none`/`is_ok`/`is_err` + `unwrap` (SwitchInt; wrong variant →
`Trap{UnwrapNone|UnwrapErr}`). `option_result__01` corpus case differential-green.
**A1 iteration gap RESOLVED — CD-032 (owner, 2026-07-19):** Vec iteration folds into C4.5f.
STARK's `.iter()` binds `value: &T` (by-reference = an interior reference into a runtime
container); A1's by-value `VecIterNext -> Option<T>` had no STARK trigger and is struck.
Iteration (by-reference `Option<&T>`) activates via a future `0.1-A2` surface bump alongside
the interior-reference/frame-generation work in C4.5f. `collection_iter__01`'s iteration half
stays Unsupported until then.
**WP-C4.5e-2 done 2026-07-19** (Vec data surface, A1/CD-031): `RuntimeFn` Vec group +
`MirValue::Vec`; `Vec::new`/`with_capacity`, method dispatch (push/pop/remove/clear/len/
is_empty), `v[i]` read → `VecIndexGet` (Copy T), `v[i]=x` → `VecReplace`+drop-old, `clear()`
on droppable T → pop-and-drop loop (§5a — destructors only at visible Drop terminators),
`Vec<T>` a droppable leaf unit dropping elements **reverse index order** (matched to oracle);
verifier schematic-T `runtime_sig` + V-COPY-1 (MIR-0016, `copy_types` populated,
`mir_needs_drop` precise); interp Vec ops (in-place `&mut Vec` mutation, call-site trap
provenance). 4 differential + 2 verifier tests. Workspace 691/0/2; fmt+clippy clean 1.93/1.97.
**WP-C4.5e-1 done 2026-07-19** (strings, implementing Amendment A1/CD-031): A1 shape
foundation landed (`MIR_RUNTIME_SURFACE`, `MirProgram.mir_version`/`runtime_surface`,
`Constant::Str`, `Trap.message`, `TypeContext.copy_types`, String/str `RuntimeFn` group, dump
header + `const "…"`). String literals, `String::new`/`from`, String/str method dispatch,
`&str`/`String` print, String/str comparison via `StrEq`/`StrCmp` (V-STR-2), `panic(msg)`/
`assert(cond)` traps, String as a droppable leaf unit, and user `as` casts (were unlowered)
all lower; verifier surface gate (MIR-0017) + V-STR-1/2 (MIR-0015) + Trap.message threaded
through every operand analysis; MIR interp gained `Str`/`String` values, in-place `&mut String`
mutation, snapshot `as_str`, and trap-message comparison. **The two frozen `ownership_drop__*`
corpus cases are differential-green** (first String-dependent corpus cases). Deferred to later
e sub-slices: Char + Char String ops, `assert_eq`/`assert_ne`. Workspace 684/0/2; fmt+clippy
clean on 1.93 and 1.97.
**WP-C4.5e-0 done 2026-07-19** (pre-runtime-values hardening, CD-030 review disposition):
IndexProof definite-initialization dataflow (must-analysis + unique-definition rule; 4
adversarial negatives incl. the review's one-branch example); V-REF-1/MIR-0014
write-through-shared-reference rejection (write-path place typing); pre-trap stdout now
observable and compared by the differential (`run_with_partial_output` + `MirFailure`;
drop-output-before-trap regression test); DEV-068 fixed (user `impl Copy` structs were
always-Move → field-precise verifier rejected valid double-use programs). Deferred with
owners per CD-030: frame generations (C4.5f), projected-move take-and-poison (C4.5e proper).
Workspace 675/0/2.
**WP-C4.5d done 2026-07-19** (ownership and Drop): droppable locals decompose into per-unit
`DropFlag`-guarded drops (units = outermost dtor-bearing/enum/array sub-places through
dtor-less structs/tuples — partial moves clear exactly the covered units); emission at scope
exits (reverse decl order), early exits, assignment overwrite (install-then-destroy per
CD-012), discards, and the `drop(x)` builtin; dtor instances discovered + registered in
`TypeContext::drop_impls`; MIR-interp recursive glue (own dtor through `&mut` ref, then
fields/payload reverse, enums by runtime discriminant); verifier V-MOVE-1 refined
field-precise with Drop-of-possibly-moved legal by design, V-DROP-2 read half added. Oracle
drop timing pinned empirically before implementation; the differential then matched on first
run (no new oracle defects — first increment where that happened). Boundaries (clean
Unsupported): match on owned Drop-bearing scrutinee (C4.5e, needs drop_unbound), Drop impls
on generic nominals (needs generic impls). Workspace 668/0/2.
C4.1-C4.4 done; WP-C4.5 split into increments (WP-C4.5.md). Done so far: C4.5a
(methods/assoc-fns/trait dispatch incl. defaults; corpus __01 differential-green),
C4.5-contract-cleanup (CD-029: trap provenance through outcomes + differential span
comparison; VerifiedMirProgram wrapper — run_program consumes proof-of-verification only;
TypeContext amended into mir.md §2, still v0.1; canonical_float spec tests as the
compensating control for the intentionally-shared formatter), C4.5b (indexing via CheckIndex
proof tokens + real reference places; DEV-065/066 oracle fixes), and **C4.5c 2026-07-19**
(external framing per CD-030: *top-level generic monomorphisation and static bound dispatch*
— generic methods/impls stay later-increment work: checker-recorded instantiations in
`TypeTables::generic_insts` with E0004 undetermined-rejection — DEV-064 closed; monomorphised
`FnKey::Top(item, type_args)` instances, injective `name@[args]` symbols, named
`LIMIT-MIR-MONO-INSTANCES`=512 limit negatively tested on polymorphic recursion; generic
nominal instantiations registered per `(item, args)` in TypeContext; operator + trait-bound
method dispatch per instantiation; comparisons on user nominals clean-Unsupported until
C4.5e's Eq/Ord impl dispatch; DEV-067 recorded — pre-existing checker over-rejection of
bounded params at intra-generic call sites and `&T` receivers, owner: later C4.5 increment;
6 new differential + 3 lowering + 3 typecheck tests). Same session: fixed the CI break — a
`collapsible_match` lint new in CI's clippy 1.97 (verify.rs; local was 1.93, 1.97 installed
side-by-side and both fmt+clippy verified clean at CI parity), failing every run since the
WP-C4.3 push. Differential status: no difference in lowering and MIR execution for the tested
subset, with some runtime algorithms intentionally shared and separately spec-tested.
Workspace 658/0/2 (C4.5b-2 baseline re-measured 646; the previously recorded 640 was stale).
WP-C4.3 done 2026-07-19: `src/mir/verify.rs` implements all 13 contract §10 obligations with
the MIR-xxxx internal namespace (first allocation, see Diagnostic codes); every lowered program
verifies clean; 13 hand-crafted invalid bodies each rejected with their specific code; one
unsafe-failure bug (panic on broken CFG edge in the move dataflow) caught by the negative suite
and fixed. Workspace 625/0/2.
WP-C4.2 done 2026-07-19: `starkc/src/mir/` implements the approved MIR v0.1 model (all CD-028
shapes) + scalar-core lowering + deterministic dump; 5 frozen-corpus cases lower; fn-values,
Option/Result-as-logical-enums, checked-terminator arithmetic all verified by tests (6 new,
workspace 611/0/2). Out-of-subset constructs report clean Unsupported naming C4.5.
MIR v0.1 contract APPROVED under CE3 (CD-028, approve-with-required-changes — Drop terminator,
Option/Result as logical enums, index-proof tokens; all applied). `mir.md` is the binding
implementation contract; changes to its shape need a new CE3 review + version bump.
Gate C3 complete 2026-07-19: WP-C3.1 (workload freeze + framework), WP-C3.2 (generated-Rust spike
4/17→8/17 with breadth), WP-C3.3 (direct Cranelift spike 3/17), WP-C3 breadth run, and **WP-C3.4
backend selection = `SELECT-GENERATED`** (owner CE5 decision, CD-026): generated Rust as the
initial production backend behind verified MIR, backend-neutral MIR keeping direct-Cranelift open
as a C7 migration. Decision analysis:
`starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md`. Native backend selection
status: SELECTED. Next: Gate C4 (MIR contract + verified lowering) — WP-C4.1 defines the MIR
under CE3; the generated-Rust emitter will consume that verified MIR, not typed HIR.
Mandatory compiler path: Core=CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS (C2
closed)  Backend=SELECTED (generated Rust/C, CD-026)  MIR=open (Gate C4 next, WP-C4.1/CE3)
Native=blocked (behind C4, mandatory per CD-004)
Optional tracks: ArtifactInfra=blocked (no second artifact impl yet)  TensorExpansion=blocked (no approved workload, Conditional Track T)

## C5.3e — target-layout manifest (IN PROGRESS)

**Where the three engines stand today.** They do not agree, and only a relations-only placeholder
test hides it:

| Engine | Current answer |
| --- | --- |
| HIR oracle (`interp.rs`) | `Value::Int(8)` — hardcoded, and it does not even look at the queried type |
| MIR interpreter | `reference_layout(_ty) = (8, 8)` — type-erased by construction |
| Native backend | `core::mem::size_of::<RustTy>()` — the real HOST representation (`Int32` → 4) |

`assert_eq(size_of::<Int32>(), 4)` succeeds natively and traps in both interpreters.

**The authority question is already settled by the normative spec, so this is NOT CE-shaped.**
`07-Modules-and-Packages.md` LAYOUT-QUERY-001 says the queries return "positive **target-contract**
values", and LAYOUT-ABI-001 says "layout-query values may differ between named targets and compiler
versions". A layout query answers from a *declared target contract*, not from a measurement of
whatever the host compiler chose. On that reading the native backend is currently the
**non-conforming** engine: it reports the host's `repr(Rust)` representation instead of a contract.
Addresses, offsets, niches and discriminant representation are all explicitly unobservable, so
nothing in a STARK program can depend on the contract matching the host layout.

**Design.** One injectable `TargetLayout` manifest is the authority; all three engines read it and
the native backend emits its constants rather than `core::mem::size_of`. The algorithm lives in one
place as combinators (`primitive`, `aggregate`, `enum_layout`) and each engine walks its own type
representation into them — the type representations genuinely differ (HIR/checker types vs.
`MirTy`), so this is the same producer/consumer split as `TypeContext::is_copy`, and it gets the
same treatment: an empirical agreement check rather than a shared walk.

**The cross-check sub-decision was RESOLVED AGAINST the recommendation by CD-067** — see that
entry. Falsifiability comes from testing the declared algorithm and the frozen manifest values, not
from comparing against Rust's private representation. The generated crate emits contract constants
and asserts nothing about its own physical layout; generated nominals stay `repr(Rust)` and remain
free to reorder fields and use niches, none of which a STARK program can observe.

**Delivered (7 of 7 directive items).** `src/layout.rs` is the contract: `stark-64-v1`, identity
`(target_contract, layout_contract_version, compiler_layout_revision)`, one set of combinators
(`aggregate` / `array` / `sum`), and `contract_for` REJECTING an unknown target rather than
defaulting. Two independent adapters, as the directive required: `TypeChecker::contract_layout`
walks checker `Ty` (it owns type conversion, generic substitution and the nominal tables — the
oracle reproducing them would have been a fourth derivation) and `TargetLayout::layout_of` walks
`MirTy` for the MIR interpreter and the backend. Native emits `4u64`, never `core::mem::size_of`.
Five frozen exact-value matrices agree across all three engines (primitives, tuples, arrays,
structs, enums+`Option`/`Result`/`Ordering`); the CD-056 relations-only placeholder is deleted.
Eight mutation tests. Layout identity is in the build key and `build.json`, with a test that a
value changed WITHOUT bumping the identity leaves the key stable — deliberately, since the identity
is what a build is accountable to and hashing values would hide the drift it exists to expose.
DEV-099 fixed (`hir_field_ty` now handles arrays).

**Two things found while building it, both reported rather than absorbed:**

- **A mutation test that could not fail.** `dropping_the_field_alignment_rule_changes_the_answer`
  first used `(Int8, Int64)`, where correct and mutant both give 16 because the trailing round-up
  hides the missing gap. Rewritten on `(Int8, Int32, Int8)` — 12 correct, 8 mutant. A mutation
  test that cannot fail is worse than none.
- **DEV-100**, below: a real engine divergence the contract work exposed.

- DEV-100 [2026-07-23, found by WP-C5.3e, BLOCKS nothing in the frozen matrix but is a live engine
  divergence] **`size_of::<T>()` inside a generic body: the MIR interpreter answers correctly and
  the HIR oracle refuses.**

  - `fn f<T>() -> UInt64 { size_of::<T>() }` called as `f::<Int32>()` → MIR/native answer 4; the
    oracle errors with "the target layout contract does not describe this query's type".

  - **Root cause: the HIR oracle has NO generic type substitution at all** — `grep` finds no
    `param_subst`, no `type_args`, no `Ty::Param` handling anywhere in `interp.rs`. It is a fully
    dynamic interpreter that never needed instantiation types. The checker records one layout
    answer per query expression, and a generic body is checked ONCE with `Ty::Param`, so there is
    no per-instantiation answer to record.

  - **This divergence is newly VISIBLE, not newly created.** Before C5.3e both engines answered a
    hardcoded 8 for every type — they agreed by being equally wrong. Making the answer real made
    the oracle's missing machinery observable.

  - **Not reachable from the C5.3e exit evidence**: the frozen layout matrix is entirely concrete
    types, and the three-engine harness runs concrete programs. But it is an engine divergence
    under the charter's six-clause rule and needs an owner disposition — fix (oracle-side
    substitution: push each call's `generic_insts` entry, resolve `Ty::Param` at the query) or
    record as a bounded deferral.

- CD-068 [2026-07-23, DEV-100 FIXED by owner directive — deferral refused] **`size_of::<T>()`
  inside a generic body now agrees across all three engines. The HIR oracle has a call-time generic
  substitution stack, which it previously lacked entirely.**

  - **Owner's ruling on why it blocked closure**: a layout query in a generic function is not an
    exotic adjacent feature but the ordinary COMPOSITION of two capabilities already inside C5 —
    monomorphised generic functions and layout queries — and MIR amendment A4 states that a generic
    layout query is instantiated with the active substitution. Deferring would have meant claiming
    "generic functions work, and layout queries work, but their ordinary composition does not work
    in the reference oracle". The absence from the frozen matrix meant the MATRIX was incomplete
    for this interaction, not that the interaction fell outside Core.

  - **Delivered**: `Interpreter::generic_frames`, a stack of call-time substitutions behind an RAII
    guard (`GenericFrame`). Pushed from the checker's `generic_insts` entry paired with the
    callee's own generic parameter names; popped on every completion path including traps and
    interpreter errors. `Rc<RefCell<_>>` so the guard owns a handle rather than borrowing `self` —
    a guard holding `&mut self.generic_frames` cannot coexist with the `&mut self` call it wraps.

  - **Bounded exactly as directed.** The stack carries call-time type substitutions and nothing
    else: no HIR body cloning or specialisation, no effect on value execution, no inference, no
    second type checker. A missing `generic_insts` entry or an arity mismatch installs NOTHING, so
    the query then fails as an unsubstituted parameter rather than answering from a partial or
    stale frame. `ty_contains_param` makes a surviving parameter an oracle DEFECT, never a
    fallback layout.

  - **Substitution recurses**, per the directive's warning against handling only a bare
    `Ty::Param`: tuples, arrays, references, nominal generic arguments, `Option`/`Result`/core
    parameterised types, and function types.

  - **Design correction made while fixing it.** The published table changed from
    `layout_answers: HashMap<ExprId, Layout>` to `layout_queries: HashMap<ExprId, Ty>` plus a
    published `LayoutTables`. A precomputed answer cannot work for a generic body — the checker
    sees it ONCE with `Ty::Param`, so there is no per-instantiation answer to precompute. The
    checker now publishes the declaration-ordered nominal tables and generic parameter names
    instead, and the walker lives in one place (`LayoutTables::layout_of`) rather than being
    duplicated between checker and oracle.

  - **A second real gap the fixture exposed**: a nominal instance reachable ONLY through a layout
    query was never registered in the type context — nothing in `size_of::<Pair<Int32>>()`
    constructs a `Pair<Int32>`, and `register_reachable_nominal_instances` walked only local
    declaration types. MIR failed at run time with "no field table for struct #0" on a program the
    front end accepted. Fixed by also visiting `Rvalue::LayoutQuery`'s type.

  - **Evidence**: three three-engine cases (a generic body with `size_of` and `align_of` at several
    instantiations; composite substitution through `[T; 4]`, `Pair<T>`, `(T, Int8)` and
    `Option<T>`; nested and repeated instantiations where the inner frame must not leak and the
    outer must be restored — checked by re-reading `size_of::<T>()` after an inner generic call),
    plus three substitution unit tests including the directive's mutation case: with the push
    removed the parameter survives and is DETECTED rather than silently laid out.

- CD-069 [2026-07-23, owner-authorized] **Frozen corpus `corpus_version` 1.2.0 → 1.3.0 — a RE-PIN,
  and the first bump that changes an existing expectation rather than adding coverage.**

  - `option_result__03_box_and_layout_queries.snap` recorded the pre-contract placeholder from when
    every consumer answered one machine word for every type: `size_of::<Int32>()` → `8`,
    `align_of::<Bool>()` → `8`. Under the named target contract `stark-64-v1` they are `4` and `1`.

  - **Scope, verified before regenerating**: exactly ONE corpus file changed and exactly TWO output
    lines within it. Every hash from 1.0.0, 1.1.0 and 1.2.0 is otherwise untouched, so the original
    baseline survives byte-identically everywhere else and comparisons against it stay valid.

  - MIR amendment A4 predicted this precisely: its option (b) says real reference numbers "break
    the current differential's shared placeholder in a way that must be re-pinned in BOTH engines".

  - **Performed as four deliberate steps**, per WP-C3-ENTRY/CD-025: regenerate the `.snap`, bump
    `corpus_version` with a dated note in `corpus.lock`, update the changed hash line, and update
    the freeze-governance assertion in `exec_snapshots.rs`. That assertion exists as a speed bump
    against exactly this situation, so the bump was **held for explicit owner authorization** and
    not performed as a side effect of the change that caused it.

- CD-070 [2026-07-23, C5.3 adversarial review dispositions] **Both review items resolved. The
  premise of one was wrong; investigating it found two other live defects. The other found a real
  defect exactly as intended.**

  - **Validation policy, approved and adopted**: `cargo test --workspace --all-targets
    --no-fail-fast` whenever a change can alter observable output, traps or spans, layout values,
    snapshots, diagnostics, Drop events, or serialization/manifest values. The fail-fast run
    stopped at binary 21 and hid later stale pins. Also preserved as a distinction worth keeping:
    `gate4a_prelude_traits` is an exact-value test and had to change; `size_of_align_of_agree` is a
    differential AGREEMENT test and correctly survived the values becoming real.

  - **DEV-098 — the stated risk is NOT reachable; two other defects were.** The review was right
    that `validate_ephemeral_references` never counts uses. But passing a `&mut` binding to another
    function twice is rejected by the FRONT END (`E0100 use of moved value`), because STARK has no
    implicit source-level reborrow — so the double-use shape does not arise from valid source and
    the "refused before rustc" promise held, for a different reason than either the old record or
    the finding gave. **Both `a(c); a(c);` and every other route were probed; the only `&mut`
    operand a body actually produces is a `Move` of a freshly created borrow temp.**

    Investigating it found two defects that WERE reachable and are now fixed: (a) `Operand::Move`
    on a reference went to `emit_move_out` and was refused outright ("move out of the non-slot
    place") — a reference is non-`Copy` at MIR level but is never slot-backed, so **passing
    `&mut x` to any user function failed**; (b) a mutable `Rvalue::RefOf` emitted `&mut _1.get()`
    (borrowing a `&T` as mutable) and then, once corrected, `&mut _1.get_mut()` (a `&mut &mut T`
    over a temporary) — the accessor for a whole slot-backed local already IS the reference. Only
    the destructor path had exercised `&mut` before, and that one is emitted by the drop glue
    rather than through `RefOf`, which is why both stayed hidden.

    `Operand::Copy` on a `&mut` now emits a reborrow (`&mut *p`) as directed. It is defensive
    rather than fixing a reachable bug, and is recorded as such.

  - **Multi-unit enum payload — a REAL defect, found exactly as the review intended.**
    `enum E { V(A, B) }` with `match e { E::V(a, b) => take_a(a) }` **compiled and then aborted at
    run time** inside `slot_violation`, whose own message reads "STARK compiler defect, not a
    program fault". No deterministic refusal existed at all — the worst of both outcomes.

    Cause: an enum payload has no raw-pointer projection, so a payload move goes through
    `move_field_whole`, which requires a complete value and leaves the slot `Partial`. With more
    than one payload unit, the second move — or the whole-enum drop of the survivor — then needs
    `Whole` over partial storage.

    **Boundary recorded and now enforced before rustc**: *C5 supports whole enum payload movement
    and the approved single-unit consuming-match shapes. Partial movement of one field from a
    multi-drop-unit enum payload, followed by projected destruction of a sibling payload unit, is
    deferred to broad ownership/reference completion in C6.* Evidence: the adversarial fixture in
    both its unbound-sibling and both-bound forms, each required to be refused as `Unsupported`
    naming the boundary, plus a single-unit negative control — a refusal that rejected every
    payload move would pass the first test while breaking `Option`/`Result` entirely.

  - Lowering emits **no projected `Drop` on a `VariantField`** for either fixture, so the
    `HelperOp::Drop` + `Whole` refusal added under CD-066 stays correct and is now backed by a
    source fixture rather than by an explanatory comment alone.

- CD-071..CD-075 [2026-07-23, WP-C5.4 CLOSED] **Deterministic native linkage, concrete generic
  emission, non-capturing function values + indirect calls, and a frozen three-package standalone
  executable — plus DEV-101, a cross-package generic typecheck fix surfaced by the workspace.**
  See `STARKLANG/docs/compiler/work-packages/WP-C5.4.md` §22 for full evidence.

  - **C5.4a (CD-072)** — `backend/generated_rust/linkage.rs`: a read-only preflight validating the
    verified body set (strict-sorted/unique canonical symbols, unique generated names, every
    referenced instance resolving to exactly one body with matching identity) and refusing before
    rustc; one exhaustive instance-reference walker with no wildcard. 12 tests incl. a real
    two-package native run and relocation symbol-stability.

  - **C5.4b (CD-073)** — proof (no backend change) that monomorphised generics emit exactly-once
    as concrete Rust with **no** generic parameter list; +4 three-engine value cases (identity at
    Int32/Int64, recursion, mutual recursion, shared instance) and 3 generated-source structural
    tests.

  - **C5.4c (CD-074)** — `MirTy::FnPtr` → typed Rust `fn(..)->..` (coincides with the emitted
    calling convention, no ABI wrapper); one aborting sentinel per distinct signature
    (`mangle::fn_sentinel_name`); `default_value_expr(FnPtr)` = sentinel; `Constant::FnPtr` =
    function item name; `Callee::FnValue` = `(operand)(args)`. +8 three-engine cases (local,
    param, return, copy, tuple, struct, generic-as-value, and the mandatory §10.5 value-only
    reachability), +4 verifier negatives, 8 structural/unit. §8.3 probe: `let f = main;` is valid
    source and builds natively.

  - **DEV-101 (in CD-075)** — cross-package (cross-file) generic instantiation was entirely broken
    in `typecheck`: turbofish/inference/coercion/qualified/nominal all failed
    (`expected 'T', found '<concrete>'`) and a satisfied cross-package bound was wrongly rejected
    with a garbage name; non-generic cross-package and all same-file generics worked. **Owner-
    directed surgical item-provenance fix** (same class as DEV-069), entirely within `typecheck`,
    no resolver/HIR/MIR/linkage/backend change: read generic parameter / associated-binding /
    trait-bound NAMES via `item_text(item_id, …)` (they are callee-declared), and carry the
    declaring file with each deferred bound so `satisfies_bound` resolves the right trait. The
    turbofish ARGUMENT stays on the caller's file. 11 tests in
    `starkc/tests/cross_package_generics.rs`. **Bounded follow-up recorded (not fixed):** the
    tensor-kind `single_segment_name` read and a callee-local associated-binding TYPE conversion
    still read `self.file`; neither can cause a Core-v1 miscompile.

  - **C5.4d (CD-075)** — frozen `starkc/tests/fixtures/c5-native-workspace/` (`app`→`logic`→
    `model`) exercising every §12.3 shape; 13 canonical symbols frozen in `EXPECTED-SYMBOLS.txt`.
    6 tests: HIR/MIR agreement + completion, byte-exact frozen symbols, linkage completeness (two
    `wrap` + two `transform` instances), **one standalone native executable that exits 0**,
    relocation symbol-stability, and a false-assertion negative control trapping in all three
    engines.

  - **Validation:** `cargo fmt --check`, `cargo clippy --workspace --all-targets --all-features -D
    warnings`, and `cargo test --workspace --all-targets --no-fail-fast` all clean/green. Native
    tests build real crates via ONNX-free generated Rust + rustc on the host.

- CD-076 [2026-07-23, **WP-C5.5 CLOSED; WP-C5.6 OPEN**] Owner accepts the C5.5 implementation and
  its post-review corrections (`2c96d99`, `e94e760`, `496406c`, evidence commit `6c00f67`). The
  stale verbose backend-artifact report is resolved, the final 1,096/0/2 validation is accepted,
  and no C5.5 user-experience blocker remains. For the carried WP-C2.12 replay obligation, owner
  approves corpus v1.4.0: `c5_native__01_supported_completion` and
  `c5_native__02_supported_overflow_trap` are the exact non-String C5-native subset and must replay
  through both the frozen snapshot harness and HIR/MIR/native comparator during WP-C5.6.

- CD-077 [2026-07-23, **WP-C5.6 CLOSED; GATE C5 CLOSED**] Owner accepts
  `starkc/docs/compiler/C5-exit-report.md` and the verdict
  **NATIVE-CORE-MVP-WITH-LISTED-DEVIATIONS** against exact qualification head
  `19254086d5f71db169fd1a1020bf30bddd284686`.

  - **Qualification green:** focused C5.6 matrix 188/0/0; complete
    `cargo test --workspace --all-targets --all-features --no-fail-fast` 1,098/0/2 across 55
    test-bearing binaries; `stark-runtime` 23/0/0; formatting clean; strict all-target/all-feature
    clippy clean. GitHub Actions run `29981161896` succeeded for the exact SHA on both configured
    jobs.
  - **Required replay discharged:** corpus v1.4.0's two owner-approved C5-native sources pass the
    frozen snapshot harness and the HIR/MIR/native comparator. The older String/collection cases
    remain valid HIR corpus evidence but are not misrepresented as native coverage.
  - **Reference product proof:** a relocated `app -> logic -> model` workspace builds through the
    production CLI with `--locked --offline --emit-rust --verbose`; all 13 canonical bodies link;
    the stable `app/target/stark/debug/app` executable runs with status 0.
  - **Scope ruling:** CD-077 explicitly accepts the entry-plan Output/Display delta. C5 native has
    no source `String`/`str`, string constants, print/eprint, or Display-to-output runtime calls;
    those and the other exact report boundaries are C6-or-later work and are rejected before
    rustc. There is no known miscompilation, invalid-MIR acceptance, ownership unsoundness, or
    unexplained divergence inside the admitted native subset.
  - **Next state:** Gate C6 is not opened by implication. C6 entry planning and owner approval are
    next, with the exit report's deferred-feature matrix as mandatory input.

- WP-C5.5 implementation record [2026-07-23, commits `2c96d99`, `e94e760`, `496406c`, **CLOSED
  CD-076**]
  **Debug build integration is complete without changing C5.4 semantics or the `NativeArtifact`
  contract.** The production native-build driver supplies its resolved rustc, Cargo, and runtime
  paths explicitly to the generated-Rust backend. The selected rustc handles target discovery and
  is exported to Cargo as `RUSTC`; the selected Cargo performs `build --offline`; and the selected
  runtime path is the generated manifest dependency. `BackendDiagnostic::BuildFailed` carries a
  boxed structured failure with summary, command, exit status, stdout, stderr, and exact retained
  build directory. CLI diagnostics classify that boundary without parsing process text.

  - **Real CLI closure proof:** a relocated copy of the frozen
    `starkc/tests/fixtures/c5-native-workspace/` builds with
    `stark build --locked --offline --emit-rust`, installs the stable executable at
    `app/target/stark/debug/app`, and runs successfully. This exercises C5.4's cross-package direct
    calls, concrete generics, function values/indirect calls, structs, `Option`, loops, layout, and
    casts through the production build path.
  - **Installed/offline proof:** unit coverage discovers the runtime beside an installed
    `bin/stark` at `lib/stark/stark-runtime`; CLI coverage uses a relocated runtime and selected
    Cargo wrapper with an empty `CARGO_HOME`, verifies `--offline`, and observes the exact canonical
    runtime path in the generated manifest.
  - **Failure-retention proof:** a Cargo wrapper exiting 23 proves backend classification, status
    and stderr transport, the exact retained-directory note, and retained `src/main.rs`.
  - **Artifact-lifecycle correction (`496406c`):** `BuildCommandResult.backend_artifact` is present
    only when the generated crate is retained. A normal `stark build --verbose` no longer
    advertises the backend-local binary after cleanup; it still reports and verifies the stable
    final artifact. The stale C5.1/C5.4 future-tense backend comments were corrected with the fix.
  - **Focused validation:** 9 CLI tests; 2 native-toolchain unit tests; 27 C5.3/C5.4 native
    regression tests; formatting and strict workspace clippy all green. Full-workspace closure:
    **1,096 passed / 0 failed / 2 ignored across 55 test-bearing binaries.** Exact commands,
    toolchain versions, and adversarial dispositions are recorded in WP-C5.5 §29.

- CD-100 [2026-07-24, **WP-C6.1g-a LANDED — structural Copy; borrow-carrying nominals in locals**]
  OWN-COPY-001 amended (owner-worded): a recursively-`Copy`, non-`Drop`, non-owning nominal is
  `Copy` **structurally**, no `impl Copy` required — shared references participate, mutable
  references never do, and any owned/`Drop`-bearing field disqualifies. Implemented as ONE predicate
  (`typecheck::copy_eligible_types`, a fixpoint over field types) consumed by the type checker, move
  checker, MIR (`FnLowerer`/`TypeContext` `is_copy`), HIR interpreter, and native backend derive —
  a divergence there is the DEV-072 class.
  - **This resolves the C6.1g-a core:** a `Copy` borrow-carrying nominal (`Option<&T>`, a user
    generic at a reference) is non-slot-backed, so it flows through the CD-095 aggregate path and
    works **in a local and across blocks** — the two shapes CD-096 had to refuse.
  - **Landing boundary** (`emit_types::refuse_borrow_carrying_nominals`, owner ruling): Copy
    borrow-carrying nominal locals admitted; **Move** borrow-carrying nominal locals refused
    pre-rustc; **any function returning a borrow-carrying nominal** refused pre-rustc regardless of
    Copy; plain reference returns supported.
  - **Corrected diagnosis (my earlier "regression" was wrong):** `wrap(&p).unwrap()` fails
    **identically for a Move referent** (E0502), so it is a general borrow-through-return limitation
    — `unwrap`'s panic-branch match extends the borrow across dispatch-loop blocks, colliding with
    the referent's block-0 assignment. Referent-storage stabilization does NOT fix it (only changes
    E0506→E0502). Uniform borrow-carrier returns are **`WP-C6.1g-c`** (dispatch-loop linearisation),
    an independent backend package; the original "uniform returns green" acceptance bar is revised.
  - **A DEV-072-class divergence was found by the new fixtures and fixed:** `borrowck::is_copy_type`
    ignored a nominal's type arguments (`H<&mut P>` read Copy there, Move in the checker); it now
    recurses arguments, matching `is_copy_with_impls`.
  - **Test churn from the semantic change:** 3 lib + 6 native tests that used all-Copy-field structs
    as Move stand-ins switched to `Drop`-bearing (Move-but-native) types; the C5.3 lane test rotated
    its negative to a Move borrow-carrier.
  - **Spec regenerated** (`STARK-Core-v1.md`/`.html`/`.pdf`); 112-block fixture corpus in sync.
  - **Evidence:** `c61f_structural_copy.rs` (positive: primitive/nested/generic/borrow-carrying/enum;
    negative: `String`/`Vec`/`Box`/`&mut`/`Drop`/mixed stay Move), `native_c61f_nominals.rs`
    (Copy-local works, Move-local + any borrow-carrier return refused). `fmt --check` and strict
    `clippy` clean.

- CD-163 [2026-07-27, **review corrections landed; three decision packets prepared**]
  - **Landed (docs/ledger only, so the `8a23772` Tier-1 evidence stands):** R-06's two lease
    violations recorded retrospectively in `C6-INTEGRATION-LEDGER.md` with the process correction and
    the next batch's leases entered IN ADVANCE; R-10's wording corrected so no claim says stderr is
    "compared three ways" (it is parsed on the native side and CONSTRUCTED for both interpreters, so
    HIR-vs-MIR equality on that field is implied by the category); and the false "7 of 9 trap
    categories" line corrected to **5 of 9** with its cause.
  - **R-12 DEFERRED with cause.** The owner allowed it "provided this remains outside the qualified
    execution path" — it is not. The summary writer is `starkc/tests/c6_generated_corpus.rs`, so
    recording skip/ignore identities now would invalidate the Tier-1 records before the packets are
    dispositioned. It moves into the consolidated batch.
  - **Three decision packets prepared** (`WP-C6.5-DECISION-PACKETS.md`), each with root cause,
    normative requirement, choices, recommendation, compatibility impact, implementation surface and
    required regression evidence:
    - **DEV-113** — root cause split in two: (A) package `SourceFile`s are named by FILESYSTEM PATH
      (`parser.rs:173/340/401`), so provenance moves with the checkout; (B) `RuntimeError` carries a
      span and NO file (`interp.rs:35–58`), so the oracle blames the entry file for every trap — even
      though the interpreter tracks `self.file` per callable and discards it at the raise site.
      Recommended: logical `<package>/<relative>` names plus attaching the file to `RuntimeError`.
      PKG-IDENTITY-001 already says identity is "never an absolute checkout path".
    - **DEV-114** — root cause found exactly: `parser.rs:200` iterates
      `HashMap<String, Dependency>`, whose order is per-process random; each dependency becomes a
      synthetic `Mod`, and a memo means whichever path is walked FIRST fixes the nesting.
      Recommended: canonical prefix = **the package's own name**, independent of the path taken, plus
      sorted iteration. **TYPE-NOMINAL-001 settles it** — identity is "canonical package instance +
      module path + item name", so a dependency edge is not a module-path segment, and
      PKG-IDENTITY-001 adds that re-exports preserve identity.
    - **CD-150 CE3** — precise semantics proposed for `TrapCategory::InvalidExitStatus` (message
      class `CategoryOnly`, provenance at the `main` signature, status 101, range `0..=255` applied
      after unwrapping `Ok`), all four PROC-MAIN-001 entry signatures on all three engines, the
      `mir.md` trap-identity amendment, and the generated-`fn main()` shape change. Recommendation:
      implement both halves together, as CD-150 intended.
  - **Sequencing recorded:** packets 1 and 2 are independent; **R-04/R-05's metamorphic floor depends
    on packet 2**, because M08/M09 cannot be built while a diamond graph's symbols are
    nondeterministic. Awaiting the owner's disposition before any qualified-path change.

- CD-162 [2026-07-27, **OWNER DIRECTIVE — WP-C6.4 CLOSED; WP-C2.12 CLOSED; WP-C6.5 stays PARTIAL;
  §17 reviews run**]
  - **WP-C6.4 — CLOSED.** The owner accepts the refreshed same-commit Tier-1 evidence at `8a23772`:
    131/131 corpus agreement on macOS-arm64 and Linux-x64, identical per-case observation hashes, row
    24 `PASS`. The ceiling `CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS` (CD-146) is discharged.
  - **WP-C2.12 — CLOSED**, recorded as its own governance closure rather than folded into C6.5. Its
    inherited deliverable — a versioned, manifest-driven generated corpus replayed across HIR, MIR and
    native on both Tier-1 targets, with metamorphic and mutation controls — is delivered and evidenced
    at `starkc/docs/compiler/evidence/c6.5/`.
  - **WP-C6.5 — remains `PARTIAL`.** Not candidate-complete: breadth and review obligations stand.
  - **§17's eight adversarial closure reviews are COMPLETE** — `WP-C6.5-REVIEWS.md`, ~90 questions
    answered against artifacts, **13 findings**, none acted on (the owner's rule: record before
    correcting). Three are new blockers:
    - **R-01 (HIGH):** the corpus covers **5 of 9 admitted trap categories**, not the 7 the WP's own
      report claimed. `DivideByZero` and `AssertFailure` are in T16's dimension space but were
      **dropped by the per-template budget of 5**; `UnwrapNone`/`UnwrapErr` were never in it. A budget
      that can delete a required category is the wrong mechanism for a template whose dimensions ARE
      the coverage claim. §10.4 is not met.
    - **R-02 (HIGH):** **23 three-engine suites still use private comparators — zero migrated** since
      CD-148 chose incremental migration, and one of them (`c65_entry_exit_contract`) was ADDED by
      C6.5 while the finding was open. Most `EXISTING-EVIDENCE` matrix rows therefore rest on
      comparators the C6.5 authority has never seen, and no closure claim may cite them.
    - **R-07 (MEDIUM):** **36 of 136 matrix rows** have corpus evidence, and nothing validates that a
      case's `subcategories` names a real row — proven by ten metamorphic family IDs passing
      validation while naming rows that do not exist (R-13). Same failure shape as CD-154's fabricated
      rule citations, caught there and uncaught here.
  - Remaining findings: mutation controls cover **7 of 15 comparator fields** (R-03); the metamorphic
    floor is unmet (R-04) and **DEV-114 blocks M08/M09 outright, not merely the floor** (R-05, a link
    not previously stated); the **shared-file lease protocol was not followed** for
    `three_engine_differential.rs` and `mir/lower.rs` (R-06); retention and divergence-retention have
    never been exercised (R-08); `MAX_LOOP_ITERATIONS` is declared but unenforced (R-09);
    `stderr_observation` equality is tautological between the interpreters (R-10); no generator-side
    ID collision check (R-11); the summary records skip counts without identities (R-12).
  - **Cost note that shapes sequencing:** the Tier-1 evidence names commit `8a23772`, and C6.4's
    re-qualification rule invalidates a record once `starkc/src|tests|scripts` changes — so fixing
    R-01, R-03, R-07, R-09, R-11 or R-13 costs a fresh Tier-1 run. R-06 and R-12 are docs/schema only.

- CD-161 [2026-07-27, **TIER-1 CORPUS AGREEMENT at `8a23772`; C6.4 ROW 24 CLOSED; WP-C6.5 recommended
  `PARTIAL`**] CI run 30221728539, all 15 jobs green. Evidence DOWNLOADED from the runners, not
  regenerated locally (§16.6), and re-verified clause by clause against the artifacts.
  - **The claim.** Both Tier-1 targets replayed the corpus at ONE commit: **131 cases, 131
    AGREEMENT, 0 failed, 0 skipped, FULL evidence, clean worktrees** — and
    `compare-c65-evidence.py` found the same commit, corpus version 0.5.0, generator version, seed,
    manifest hash and generator hash, identical counts, two DIFFERENT triples, and **identical
    per-case observation hashes for all 131 cases**. That last clause is the claim: two records can
    agree on every total while having observed different bytes. Verified independently here — no
    duplicate IDs, no case on one target only, no differing hash, every result `AGREEMENT`.
    Platform metadata that differs (OS, arch, Python 3.14.6 vs 3.12.3) is reported, not treated as
    disagreement.
  - **C6.4 ROW 24: BLOCKED-BY-C6.5 → PASS.** Both C6.4 records at the same commit carry
    `generated_corpus_status: PASS`, `generated_corpus_version: 0.5.0`,
    `generated_corpus_case_count: 131`, MEASURED by the harness from the corpus lock. The C6.4
    records were REFRESHED at `8a23772` rather than amended — the earlier `4844702` records are
    superseded under C6.4's own re-qualification rule, since C6.5 changed `starkc/tests` extensively.
    Workspace 1560/1560 with 2 classified ignores, `three_engine_differential` 109/109, determinism
    `match`, no deviations, on both targets. **Row 24 was the only thing between WP-C6.4 and
    `CLOSED`**; that decision is the owner's to record.
  - **16 mutation controls PASS** in their own CI job, all with `unmodified_agrees` and
    `mutation_detected` true.
  - **Recommended WP-C6.5 status: `PARTIAL`, not `CANDIDATE-COMPLETE`.** §23 reserves the latter for
    "all implementation and local evidence complete, final Tier-1 evidence pending" — here the Tier-1
    evidence is the part that is DONE. Outstanding: the metamorphic floor (20 groups/40 members
    against 24/48; M08/M09 blocked on package graphs), per-row witnesses (21 against 136 matrix
    rows), `UnwrapNone`/`UnwrapErr` trap cases, the §11.11 retention workflow, templates T13/T14 and
    T17–T19, §15.1's dependency-trap provenance (blocked by DEV-113), **§17's eight review passes —
    not started**, and the open defects DEV-113, DEV-114 and the CD-150 CE3.
  - **WP-C2.12:** its deliverable — a versioned manifest-driven corpus replayed across three engines
    on both Tier-1 targets, with metamorphic and mutation controls — is delivered and evidenced. NOT
    recorded as closed here: closure is a governance act, and doing it inside a package whose own
    status is `PARTIAL` would bury it. Recommended for the owner on this evidence.
  - **Evidence committed:** `starkc/docs/compiler/evidence/c6.5/` (both summaries, both per-case
    files, `mutations.json`, `c65-tier1-agreement.md`) and refreshed `evidence/c6.4/` records. This
    commit touches no file under `starkc/src`, `starkc/tests` or `starkc/scripts`, so the records
    remain valid for the commit they name.

- CD-160 [2026-07-27, **WP-C6.5-9 commit 11 — Tier-1 machinery and the C6.4 handoff; row 24 NOT
  flipped**] The jobs, records and comparator exist; the Tier-1 CLAIM does not, and this entry does
  not make one. Two records at one commit have to come from CI first — asserting agreement from the
  machinery that would produce it is exactly the substitution §16.5 forbids.
  - **CI jobs (§16.1):** `c65-corpus` on macos-arm64 and linux-x64 (integrity, full replay,
    metamorphic, package breadth; platform-named artifact), `c65-mutation-controls` as its OWN job so
    a run where the mutations were skipped looks different from one where they passed, and
    `c65-tier1-comparison` with `if: always()` — a skipped comparison is an absence a reader must
    interpret, not a report.
  - **§16.2 identity, MEASURED in-process:** target triple from `rustc -vV`, OS, architecture,
    rustc/cargo/python, MIR/backend/runtime versions, and `dirty_worktree` from `git status`. A record
    whose triple came from its caller proves nothing about the machine that ran the corpus.
  - **The §16.4 comparator** requires same commit, corpus/generator version, seed, manifest and
    generator hashes, identical counts, both `PASS` and FULL evidence, clean worktrees, two DIFFERENT
    Tier-1 triples, and — the strongest clause — identical **per-case observation hashes**. Two
    records can agree on every count while having observed different bytes. Platform metadata expected
    to differ is reported, never treated as disagreement.
  - **§20.7's controls are tests** (`c6_tier1_controls`, 13): same platform twice, different commit /
    corpus version / seed / manifest hash, dirty worktree, filtered run, a skip, a failure, a missing
    record, a case present on only one target, a differing per-case observation — plus one VALID pair
    that must be accepted, without which "rejects everything" would pass the rest.
  - **C6.4 handoff (§16.5):** the qualification harness runs the five C6.5 corpus commands and
    MEASURES `generated_corpus_version` and `generated_corpus_case_count` from the corpus lock;
    `generated_corpus_status` is derived from whether those steps passed IN THAT RUN — `NOT-RUN`,
    `PARTIAL`, `FAIL` or `PASS`. Verified: a `--only fmt` probe correctly reports version 0.5.0, 131
    cases and status `NOT-RUN`. `compare-c64-evidence.py`'s expectation inverts from
    `BLOCKED-BY-C6.5` to `PASS` with a nonzero count — a record still reporting the old status now
    means the corpus steps did not execute.
  - **Row 24 is deliberately NOT flipped.** It flips when two records exist and agree, not when the
    machinery to produce them lands.
  - **Also this commit:** `1c47908` fixed a Windows-only failure CD-159 shipped — my DEV-113 pin used
    `ends_with("app/src/main.stark")`, and Windows returns a MIXED path (`…\ws\app\src/main.stark`)
    because the OS builds the directory part with backslashes while the entry suffix is composed with
    a literal `/` in the compiler. Separators are normalised now, and the inconsistency is noted in
    DEV-113's record rather than absorbed.
  - **Evidence:** `c6_tier1_controls` 13/13, `c6_package` 6/6, corpus replay 131/131, clippy and fmt
    clean. Still owed by §16: sharded jobs and merge (not needed at ~90s per target), the real Tier-1
    comparison, row 24, and evidence import.

- CD-159 [2026-07-27, **WP-C6.5-8 commit 10 PARTIAL — package breadth; DEV-113 and DEV-114 found**]
  Corpus `0.5.0`, **131 cases, replay 131/131 AGREEMENT**. Two package cases: a root package with a
  module, and a three-package workspace (`app → logic → model`) covering a dependency-to-dependency
  call, a re-export, a cross-package generic, a cross-package function value, and a **`Drop` type
  from the leaf package destroyed in the root**, observed through the §8.8 protocol. The replay now
  STAGES package cases before compiling — resolution writes `stark.lock` into the root package, so
  compiling in place would dirty the corpus and break its own lock, and concurrent cases sharing a
  root would race on that file. `C6_KEEP_TEMP` is honoured.
  - **DEV-113 — a package build puts ABSOLUTE PATHS in trap provenance.** §15.2 requires no absolute
    path in semantic identity and logical trap source names; for a package graph, file identity IS
    the filesystem path, so the same workspace staged at two locations reports different provenance.
    **Consequence: a trapping package case cannot join the corpus** — its observation would depend on
    where the repo was checked out. Second half: the HIR oracle attributes every trap to the ROOT
    file whatever file trapped, while MIR attributes it correctly, so a dependency-trap case would
    make the engines disagree about WHICH FILE. §15.1's "source provenance in dependency trap" is
    therefore NOT covered. Both halves pinned by tests that retire when the behaviour changes.
  - **DEV-114 — canonical package symbols are NONDETERMINISTIC for a diamond graph.** With
    `app → {logic, model}` and `logic → model`, the same function is `model::leaf@[]` in one process
    and `logic::model::leaf@[]` in the next — same sources, same manifests, same declaration order;
    six consecutive runs produced both forms. The prefix is assigned by whichever traversal path
    reaches the package first, and that traversal follows a per-process-seeded hash map. Canonical
    symbols are the identity that reaches the backend, so two builds of one workspace can produce
    differently-named code — against PKG-IDENTITY-001 and CD-108's deterministic identity.
    **ESCALATED, not fixed** (§18.5): choosing the canonical name for a package reachable by several
    paths is a compiler decision. The corpus workspace is a CHAIN, not a diamond, so no corpus case
    is flaky; the defect needed a purpose-built graph to surface.
  - **A methodological error recorded against myself.** My first reorder experiment reused the
    relocation helper, which compiles BEFORE rewriting the manifest and leaves a stale `stark.lock`.
    That made the result depend on run order and I briefly wrote it up as "symbols depend on
    declaration order" — plausible and wrong. The clean experiment showed order-independence and
    process-nondeterminism instead. A contaminated experiment that produces a believable defect is
    worse than no experiment.
  - **Still owed by §15:** dependency-trap provenance (blocked by DEV-113), cross-package trait impls,
    CLI `--locked --offline`, installed-runtime cases (held by `c63_closure_evidence`), and M08/M09 as
    corpus metamorphic groups — covered as harness checks, which does NOT raise the §13.2 group count.
  - **Evidence:** `c6_package` 6/6 (relocation, reorder, both DEV pins, offline resolution),
    `c6_generated_corpus` 131/131, `c6_metamorphic` 3/3, `c6_corpus_manifest` 30/30, clippy and fmt
    clean.

- CD-158 [2026-07-27, **WP-C6.5-7 commit 9 — all sixteen mutation controls detected**]
  The negative control for the whole package. Every other phase shows the corpus and comparator
  AGREE; a suite of passing tests cannot distinguish "the engines match" from "the harness cannot
  tell them apart". §14 is what separates those.
  - **Mechanism (§14.5), per mutation:** take a REAL passing corpus case, run it through the REAL
    engines, clone one normalised observation, apply ONE precise test-only mutation, invoke the
    PRODUCTION comparator, require rejection naming the intended field.
  - **The sixteen:** MU01 arithmetic (generated T01) → `stdout_bytes`; MU02/03 trap line/category
    (generated T16 overflow) → `trap line`/`trap category`; MU04–07 omitted/duplicated/reversed Drop
    and copied move (the three-event Drop sentinel) → `drop_log`; MU08/09/10 wrong generic instance /
    trait impl / function-value target (the three dispatch sentinels) → `stdout_bytes`; MU11 sorted
    instead of insertion order → `stdout_bytes`; MU12 slice view copied → `stdout_bytes`; MU13
    `Float32` rendered as `Float64` → `stdout_bytes`; MU14 generated-Rust path replacing user source
    → `trap source_file`; MU15/16 missing output / wrong exit → `stdout_bytes`/`exit_status`.
  - **Three rules enforced, not intended** (§14.6/§14.7): the witness must agree BEFORE mutation (a
    detection on an already-failing case proves nothing); the mutation must actually change the
    observation (asserted — the identity-transform trap CD-157's generator hit twice); and no
    mutation is simulated by asserting `false`, since the comparator under test is
    `compare_observations`, the function the replay itself uses.
  - **Routing controls (§14.5).** Mutating an observation shows the comparator would catch a wrong
    ANSWER, not that a wrong ROUTE produces one. So MU09 and MU12 additionally run the wrong route as
    a REAL PROGRAM — calling the other trait impl, and passing an array by value instead of taking a
    view — and assert the observation differs. Without those, both rest on my assertion that the
    sentinel discriminates.
  - **One recorded gap:** `returned_observation` has no corpus witness (the §8.7 framed-probe cases
    live in `three_engine_differential.rs`, not the corpus), so that field's sensitivity is proven
    against a constructed pair — comparator evidence, not corpus evidence, and the test says so.
  - **Evidence:** `c6_mutation` 4/4, `target/c6.5-evidence/mutations.json` in the §21.3 schema,
    `clippy --workspace --all-targets --all-features` clean, `fmt` clean.

- CD-157 [2026-07-26, **WP-C6.5-6 commit 8 PARTIAL — 20 metamorphic groups; the floor is not met and
  a test says so**] Corpus `0.4.0`, **129 cases, replay 129/129 AGREEMENT**. Ten of §13.1's twelve
  families, two independent groups each, 40 member cases.
  - **Families:** M01 renaming, M02 scope insertion, M03 explicit-vs-inferred generics, M04
    qualified-vs-unqualified trait call, M05 shorthand-vs-explicit fields, M06 nested-vs-sequential
    pattern, M07 non-overlapping arm reorder, M10 helper extraction, M11 direct-call-vs-function-value,
    M12 `while`-vs-range-`for`.
  - **The preconditions are CONSTRAINTS, not commentary.** Scope insertion is refused over a
    `Drop`-bearing base by an assertion, because there it is NOT semantics-preserving — the inner
    block ends earlier, destruction moves (DROP-ORDER-001), and the pair would fail against a CORRECT
    compiler. Arm reordering asserts no catch-all (§13.5). Loop equivalence asserts no owning value in
    the body (§13.6).
  - **Two FAKE PAIRS my own generator produced, both caught by its own guard.** `add()` asserts the
    transformed source differs from the base, and it fired twice: M12/g2, where a post-hoc
    `.replace("total + i", …)` broke the transform's anchor so it returned the input unchanged; and
    M05/g2, where a blind `.replace("3", "8")` turned `Int32` into `Int82`. Same root cause —
    **generating variants by substring surgery over source** — now fixed by making every base a
    parameterised builder. An identity-transform pair passes trivially and looks like evidence, which
    is why that assertion exists.
  - **§13.4 comparison:** per engine (`HIR(base) == HIR(transformed)`, same for MIR and native), then
    three-engine agreement for both members via the §12 replay, which runs metamorphic members as
    ordinary cases. Divergence reports name the engine, the first differing field AND the
    precondition, because §13.7 requires normative analysis to decide defect-vs-invalid-transformation
    and the precondition is where that starts.
  - **THE FLOOR IS NOT MET.** §13.2 requires 24 groups / 48 members over all twelve families; this is
    20/40 over ten. **M08 (workspace relocation) and M09 (dependency reorder) transform a PACKAGE
    GRAPH**, and every case is single-file until §15 — a single-file "relocation" pair proves nothing
    about relocation, so they are absent rather than approximated.
    `the_metamorphic_floor_is_reported_honestly` asserts both the present state and that it is BELOW
    the floor, so when M08/M09 become buildable the test fails and demands the expectation be raised.
    A shortfall recorded only in prose is one that gets forgotten.
  - **Also this commit:** `dc72136` fixed the clippy `field_reassign_with_default` errors CD-156
    shipped, which had turned main red on all six jobs (and, as at CD-154, was failing another
    author's commits). CI's exact clippy invocation was run before this push.
  - **Evidence:** `c6_metamorphic` 3/3, `c6_generated_corpus` 6/6 (129 cases, 81s),
    `c6_corpus_generator` 8/8, `c6_corpus_manifest` 30/30, `clippy --workspace --all-targets
    --all-features` clean, `fmt` clean.

- CD-156 [2026-07-26, **WP-C6.5-5 commit 7 — the full three-engine replay; 89/89 AGREEMENT**]
  `starkc/tests/c6_generated_corpus.rs`, the plan's named §12.1 entry point: validate manifest, verify
  lock, enumerate in case-ID order, run each case on the engines it declares, compare field by field,
  check against the manifest's expectations, write §21 evidence. The C6.5-3 bridge is RETIRED — it ran
  cases but produced no evidence, applied no timeout and could not be narrowed.
  - **Result: 89 cases, 89 AGREEMENT, 0 failed, `full_evidence: true`, `result: PASS`, 99s.** Evidence
    written to `target/c6.5-evidence/{summary,per-case}.json` in the §21.1/§21.2 schemas, with a
    per-case `observation_hash`.
  - **Failures are CLASSIFIED** (§12.2's ten admissions plus `TIMEOUT`) and the report says outright
    when a classification is a **C6 blocker**. "An accepted Core case refused by MIR/native is a
    blocker" only bites if refusal and disagreement look different in the output; now they do.
  - **A filtered run cannot be filed as closure evidence** (§12.6): every narrowing is recorded and
    the summary reads `PARTIAL-FILTERED`. **Sharding counts as narrowing** — a shard is complete
    evidence for the shard and for the corpus only once merged.
  - **A timeout is a failure, not a skip** (§12.4): 120s per case on a worker thread, 3600s whole-run
    ceiling. A hung native binary fails its case with the budget named instead of stalling the run —
    CD-127's infinite-loop shape. The worker is abandoned rather than killed; recorded as deliberate.
  - **Sharding (§12.7) is content-addressed**, `u64(SHA-256(case_id)[0..8]) % total`, not index-based:
    adding one case moves only the cases whose digests demand it rather than reshuffling every shard.
    Partition claims checked over the real corpus at six shard counts — each case in exactly one
    shard, none omitted, none duplicated.
  - **Determinism (§12.8)** proven by replaying a shard twice and comparing observation hashes. The
    hash is over an EXPLICIT canonical rendering, not `Debug` — `Debug` is stable in practice but not
    by contract, and an evidence hash that moved with a Rust release would invalidate stored records
    for no semantic reason.
  - **Still owed by §12:** the package-graph step (single-source only until §15), the generated-crate
    path in divergence reports, `C6_KEEP_TEMP` honoured by the native runner (parsed and recorded
    today), and deterministic shard-summary merging (CI work, commit 11).
  - **Evidence:** `c6_generated_corpus` 6/6 (99s full replay + determinism + sharding + filters),
    `c6_corpus_generator` 8/8, `c6_corpus_manifest` 30/30, `fmt` clean.

- CD-155 [2026-07-26, **WP-C6.5-4 commit 6 — the deterministic generator; corpus 0.3.0, 89 cases**]
  **70 generated cases across 15 templates**, plus the 13 sentinels and 6 retained. §11.4's floor
  (≥64 cases, ≥10 templates, completion AND trap, full provenance per case) is met and ASSERTED BY A
  TEST rather than counted by hand.
  - **Selection (§11.2):** dimension tuples enumerated in sorted order, ranked by
    `SHA-256(generator_version | seed | template_id | canonical_dimensions)`, truncated to a
    per-template budget of 5; case ID = template + digest prefix. Nothing host-dependent enters
    identity — no filesystem order, PID, timestamp, absolute path, or Python-representation
    dependence (the dimension tuple is canonicalised by an explicit function, not `repr`, which is
    stable in practice but not contractually).
  - **Expectations come from the TEMPLATE, not from an engine.** Same principle as the sentinels and
    the reason both exist: the corpus claims the three engines agree with the SPECIFICATION, and an
    expectation read back from one engine could only show the engines agree with each other.
  - **Registry: 15 of §11.5's 20 families** (T01–T12, T15, T16, T20). **T13/T14 absent** (borrow/
    reborrow/reference return, partial move/reinit — covered by handwritten cases today) and
    **T17/T18/T19 blocked on package graphs (§15)**. `--list-templates` prints the absent ones with
    reasons, so the registry never implies coverage it does not have.
  - **Valid by construction (§11.7), not by trial:** each template's dimension space excludes tuples
    that would produce invalid or accidentally-trapping programs (unsigned subtraction that would go
    negative is filtered — overflow traps, and T01 is a completion template). No case was found by
    generating and discarding failures. **All 70 pass on all three engines.**
  - **Determinism proven by RUNNING the generator (§11.10), 8 tests:** same seed twice byte-identical;
    relocation to a different and deeper root identical; pre-existing junk in the output directory
    irrelevant; a different seed reselects but stays reproducible with the same count; a GENERATOR
    VERSION change reselects, which is what makes "a version change requires corpus-version review"
    enforceable rather than advisory; no absolute path anywhere in the generated corpus; `--check`
    byte-identical.
  - **Two bugs in my own tooling that the generated DATA found, both fixed:** the manifest list parser
    split on `,` and tore `expected_stdout = ["[1, 2, 3]"]` in half (a rendered array is legitimate
    data — the parser now scans quoted items on both the Rust and Python sides), and the lock builder
    referenced a constant I had renamed. Worth recording: review had not caught either.
  - **Still owed by §11:** the §11.11 retained-case workflow has NOT been exercised with a synthetic
    failure (retention is documented and retained cases exist, but the
    `cases/retained/<DEV-ID>/original|reduced` flow is untested), and the package dimensions wait on
    §15.
  - **Evidence:** `c6_corpus_cases` 2/2 over **89 cases** (63s), `c6_corpus_generator` 8/8,
    `c6_corpus_manifest` 30/30, `generate.py --check` current, `fmt` clean.

- CD-154 [2026-07-26, **C65-F3 — the coverage matrix cited 69 INVENTED rule IDs; repaired and now
  machine-checked**]
  Found while choosing citations for the §10.3 sentinels. Of the **84 distinct normative rule IDs the
  matrix cited, 69 exist in no specification document** — 100 occurrences across ~130 rows.
  `OWN-DROP-001`, `FN-VALUE-001`, `MAP-001`, `TRAP-ABORT-001`, `CTRL-IF-001`, `PAT-WILD-001`,
  `VEC-001`, `SLICE-001`, `REF-001`: all plausible-looking, all fabricated. The real rules are
  `DROP-EXACT-001`, `TYPE-FN-001`, `STD-HASH-001`, `DROP-ABORT-001`, `EXEC-EVAL-001`,
  `SYN-PATTERN-001`, `DROP-COLLECTION-001`, `REF-SLICE-001`, `REF-IDENTITY-001`.
  - **This is the worst of the three phase-0 failures and a DIFFERENT KIND.** O13 was a wrong
    judgement inherited from a stale ledger entry; the missing entry-contract rows were an omission.
    This was invented content presented as grounding, and §7.5's exit condition "every row has a
    normative citation" was recorded as MET because nothing compared the citations to the spec. A
    fabricated citation is worse than a blank one: whoever follows it finds nothing, and everyone who
    does not follow it assumes someone did.
  - **Repaired:** all 136 rows re-cited against the spec's real rules, each chosen for what the rule
    SAYS rather than what its name resembles — `break`/`continue`/`return`/`?` all to EXEC-CFLOW-001
    (one rule about normal control transfer), Drop order to DROP-ORDER-001 and Drop-once to
    DROP-EXACT-001, trap rows to TRAP-CATEGORY-001 with DROP-ABORT-001 where the claim is about
    post-trap cleanup, `Box`/`Option`/`Result` payload destruction to DROP-ORDER-001's own bullet.
    Two substring collisions the mechanical pass introduced (`PRIM-TRAIT-001` → `PRIM-TRAIT-DEF-001`,
    `TEXT-ITER-001` → `TEXT-EXEC-FOR-001`) were caught by re-verifying every ID after the edit rather
    than trusting it.
  - **Guarded so it cannot recur silently:** `every_rule_id_the_matrix_cites_exists_in_the_spec` reads
    the matrix and fails on any ID the spec does not define; the corpus validator applies the same
    check to each case's `normative_rules`, and `a_manifest_citing_an_invented_rule_is_rejected`
    proves that check REFUSES rather than merely runs. The authority set is parsed from the numbered
    source documents only — the generated `STARK-Core-v1.md` is excluded, so a stale compilation
    cannot validate an ID the sources no longer define.
  - **Audited elsewhere, reported not silently fixed:** the same pattern exists at smaller scale in
    closed-gate records — `WP-C3-ENTRY.md` (7, incl. `STD-ITER-001`, `STD-OPTION-001`, `STD-VEC-001`),
    `WP-C1.3.md` (1), `WP-C1.6.md` (2). The `CORE-Q-0##` references in WP-C2.x are a separate
    question-numbering scheme, not spec rules, and are fine. Rewriting closed-gate documents is a
    governance decision, not a C6.5 edit, so they are named for the owner.
  - **Evidence:** `c6_corpus_manifest` 30/30 (two new citation tests), `c6_corpus_cases` 2/2, `fmt`
    clean.

- CD-153 [2026-07-26, **WP-C6.5-3 commit 5 PARTIAL — the thirteen §10.3 sentinels**]
  `corpus_version` **0.2.0**: 19 cases (13 handwritten sentinels, 6 retained). Each is built so the
  LIKELY WRONG implementation fails it, which is §10.3's stated bar — "a case that would still pass
  under the likely wrong implementation is insufficient". What each catches: structural key
  comparison in a `HashMap` (CD-133's live defect), comparing fields instead of the user's `cmp`,
  equal hashes treated as equal keys, a structural `Display` fallback, `Clone` as a structural copy,
  zero-initialisation instead of `Default`, monomorphising a generic once and reusing the body,
  picking the first matching impl, resolving an indirect call statically, copying elements into a
  slice view (§18.4's "slice copy instead of view"), sorting or hash-ordering a map, a
  declaration-order/omitted/duplicated Drop schedule, and carrying `Float32` arithmetic at f64 width
  (DEV-109's defect).
  - **The load-bearing decision: every sentinel PINS its observation in the manifest**
    (`expected_stdout` / `expected_drop_log`), and a test enforces that it does. A wrong
    implementation is usually wrong in ALL THREE ENGINES AT ONCE — a structural `Display` fallback, a
    sorted map iteration, a declaration-order Drop schedule — and those agree perfectly, so
    three-engine agreement alone would pass every sentinel above. Not theoretical: the `Float32`
    sentinel failed on first run against a wrong expectation of mine, which is the mechanism working.
  - **`c6_corpus_cases.rs`** runs each case on the engines its manifest entry declares — three-engine
    where native builds it, two-engine for the DEV-111 entry cases native refuses. Deliberately NOT
    §12's replay harness (commit 7: admission classification, timeouts, sharding, filters, evidence
    schema); it exists now so no case is added in a state where nothing runs it.
  - **Two surface findings while writing the cases, recorded not worked around.** (1) `T::assoc()`
    through a type PARAMETER does not resolve (`E0200 "undefined variable 'T::tag'"`); TRAIT-ASSOC-001
    covers `T::Item` for associated TYPES, so whether an associated FUNCTION is callable through a
    parameter is a spec question — flagged, and the sentinel rewritten onto a `&T` receiver.
    (2) No implicit array→slice coercion: `&mut xs[0..2]` is the normative view form. Correct as
    specified; recorded because the first draft assumed otherwise.
  - **C6.5-3 is PARTIAL and the remainder is named**: §10.2's per-row witnesses (13 sentinels against
    136 rows), §10.4's completion/trap balance (NO trap case is in the corpus yet), §10.5's package
    breadth (every case is single-file), and §10.3's "same filename in different package locations".
    Sentinels went first because nothing else in the plan substitutes for them and the roll-up named
    "adversarial sentinels: 0".
  - **Evidence:** `c6_corpus_cases` 2/2 (19 cases), `c6_corpus_manifest` 28/28,
    `c65_entry_exit_contract` 8/8, `generate.py --check` current, `fmt` clean.

- CD-152 [2026-07-26, **WP-C6.5-2 commit 4 — the corpus exists: manifest, layout, lock**]
  `starkc/tests/c6-corpus/` with the §9.1 layout, a strict manifest, a generated lock, and **28
  tests — 3 on the real corpus, 25 proving the validator REFUSES what §9.3 requires**. A validator
  whose only evidence is a valid manifest is a validator nobody has watched refuse anything.
  - **Parser (§9.4): option 2, a deliberately small strict reader** (`tests/support/corpus.rs`).
    Option 1 was checked and does not apply — the workspace has no TOML parser to reuse, and §9.4
    forbids adding a network-fetched dependency to parse a test manifest. Subset: `[[case]]` plus
    `key = "string" / ["a","b"] / true`. **Unknown keys are rejected**, because a parser that skips
    what it does not understand turns a typo'd attribute into an attribute nobody checks.
  - **Seeded with the 6 retained DEV-111/DEV-112 cases, not empty.** §18.3 requires a retained case
    to remain a permanent regression, and a lock that has never hashed a real file proves nothing.
    `c65_entry_exit_contract.rs` reads them via `include_str!`, so corpus source and expectation
    cannot drift — one edit changes the hash in `corpus.lock` AND the assertion pinning the
    observation. Deliberately NOT cases, with the reason in the README: the out-of-range status (no
    replayable observation until the CE3 lands) and the pre-DEV-112 `()` rejection (history).
  - **§4.4's disallowed quarantines are unspellable, not discouraged.** Three reason classes parse —
    `non-core-feature`, `external-artifact`, `environment` — each requiring a `CD-###` authority.
    There is no syntax for "the engines disagree", "wrong output", "wrong Drop order" or "native
    refuses an accepted program". `semantic_quarantine_rejected` proves the door is shut.
  - **Lock (§9.5):** per-source SHA-256, manifest and generator hashes, five counts. `generate.py
    --lock` writes it, `--check` is the CI question, and the generator hashes ITSELF in — changing
    how the corpus is produced invalidates the lock. `c6_corpus_manifest.rs` asserts
    `corpus_version` against a constant, so regenerating without a version bump fails rather than
    quietly redefining the baseline every later claim is measured against.
  - **Evidence:** `c6_corpus_manifest` 28/28, `c65_entry_exit_contract` 8/8, `generate.py --check`
    current, `fmt` clean. `corpus_version` **0.1.0**, `generator_version` 0.1.0, case_count 6
    (0 handwritten, 0 generated, 6 retained, 0 metamorphic groups).
  - **What this is not.** The generated corpus §11 requires — ≥64 cases across ≥10 templates — is
    entirely unbuilt. This phase built the container, and the container is not the evidence.

- CD-151 [2026-07-26, **WP-C6.5-1 commit 3 — the §39 observation model; the comparator now compares
  what the claim is about**] The plan's §8.3–§8.10, additive to commit 2's mechanical extraction.
  - **The shape.** `Outcome { stdout, exit }` → `Completed { stdout_bytes, stderr_bytes,
    exit_status, returned_observation, drop_log }` / `Trapped { category, source_file, line, column,
    message_class, stdout_before_trap, stderr_observation, exit_status, drop_log_before_trap }`.
    Every field participates in equality, and `first_difference` NAMES the field that disagreed —
    with nine fields on a trap, "these two structs differ" is not a usable failure.
  - **Trap stderr is normalised, not byte-matched** (§8.5): parsed from the native engine,
    CONSTRUCTED for the interpreters from `stark_runtime::trap`'s own category table — the same
    source the native ABI prints from, so the two cannot drift. Exhaustive over `TrapCategory` by an
    exhaustive `match`: a tenth category (the pending `invalid-exit-status` CE3) fails to compile
    until it is mapped.
  - **Drop events come from the PROGRAM** (§8.8): a `Drop` impl emits `@@stark-drop:<identity>@@`,
    the harness extracts frames in order, assigns sequence by position, and strips them from
    normative stdout. Inferring Drop order from generated Rust destructors or host traces would make
    the native engine's schedule unfalsifiable. Duplicate identities and mid-line frames are hard
    failures — a Drop event that vanished into stdout would under-report the log silently.
  - **Returned values go through a framed probe** (§8.7): `fn probe() -> T` plus a generated wrapper
    appended AFTER the case source (so user line numbers, and therefore trap provenance, are
    unchanged).
  - **Two deviations from the plan's sketch, recorded not silent.** (1) The sentinel is `@@`, not
    `##`: a case source is a Rust raw string and `"##` terminates `r#"…"#`, so `##` would have made
    every drop-observing case remember `r###"`. (2) Return frames are marker-delimited rather than
    length-delimited — Core v1 source cannot compute the byte length of an arbitrary `Display`
    rendering, so the probe is instead REQUIRED to emit no other stdout and `agree_returning`
    asserts it, making the ambiguity fail loudly rather than be prevented by a prefix.
  - **18 comparator unit tests** (§8.10's full list), one per dimension so a regression names the
    field it broke. Each perturbs exactly ONE field of an otherwise-agreeing triple. Three cover
    what stdout comparison cannot see: **Drop reversal** (same identities, same count, order only),
    **pre-trap Drop change** (TRAP-ABORT-001 makes the retained log an observation), and **internal
    MIR error**, which runs the real `fn main() -> Int32 { 300 }` — DEV-111's escalated case — and
    requires the harness to fail loudly rather than report a completion.
  - **Evidence:** `three_engine_differential` **109 passed / 0 failed / 0 ignored / 0 self-skipped**
    (was 89: +18 comparator tests, +2 framed-probe cases, +1 Drop-log-before-trap case, O13 converted
    to the protocol). `fmt` clean. Test-only change; CI's three platforms are the exhaustive net.
  - **Still forked: 22 suites.** Until each is migrated, its C6.2/C6.3 evidence rests on its own
    local notion of agreement — the unified comparator has not seen it. That is the gap C65-F1
    named, and commit 3 does not close it.

- CD-150 [2026-07-26, **owner decisions on DEV-111's two escalations; DEV-112 FIXED**]
  - **The `invalid-exit-status` trap category (CE3): BUNDLED with the native entry-signature work.**
    The backend increment that emits a non-`Unit` `main` must emit this trap anyway, so one `mir.md`
    amendment, one implementation and one set of three-engine evidence rather than three. Nothing is
    lost waiting: `c65_entry_exit_contract` pins the case and fails the day either half lands.
    Meanwhile MIR fails loudly there instead of completing with status 0.
  - **DEV-112 — `()` did not typecheck as `Unit`. FIXED, and my classification of it was wrong.**
    I recorded it as a spec-vs-checker conflict needing an owner decision. **TYPE-PRIM-001 settles it
    outright**: *"`Unit` and `()` are two spellings of the same single-inhabitant type"*, and
    03-Type-System repeats it in the tuple rules ("`()` is `Unit`"). So it was a plain conformance
    bug, not governance — the correction is recorded because "this needs your decision" was the
    expensive part of the mistake, not the diagnosis.
  - **Why it was not cosmetic.** `Ty::Tuple([])` unified with nothing, so **no value of type `Unit`
    could be written at all**, and PROC-EXIT-001 gives `Ok(Unit)` its own exit-status clause while
    PROC-MAIN-001 admits `Result<Unit, String>` entries. The success branch of a legal entry
    signature was unreachable from source; such a `main` could only ever return `Err`.
  - **Fixed by canonicalising at construction in all three engines**, not by teaching `unify` that
    two representations are interchangeable — so they are ONE type as the rule says, and
    `Ty::Tuple([])` is no longer constructible from source: `unit_or_tuple` (checker),
    `Constant::Unit` (`mir/lower.rs`), `Value::Unit` (oracle). **All three were required, and each
    announced itself separately:** checker-only produced `MIR-0004 "aggregate Tuple assigned to
    incompatible type Unit"`; checker+lowering left the oracle's `Ok(Tuple([]))` failing
    `main_result_to_status` ("entrypoint returned a value inconsistent with its checked signature").
    A single-engine fix would have looked complete against a single-engine test.
  - **Evidence:** `c65_entry_exit_contract` 8/8 (adds `ok_unit_entry_completes_with_status_zero` and
    the `Unit`-literal case; the former is the clause DEV-112 had made unreachable), `--lib` 463,
    `mir_differential` 132, `exec_snapshots`, `conformance` green, `fmt` clean. Type identity is
    cross-cutting, so the exhaustive net is CI's three-platform `--all-targets --all-features` run,
    per the standing rule — not a local full suite.

- CD-149 [2026-07-26, **DEV-111 — the entry/exit contract diverged in all three engines; MIR fixed,
  native escalated**] Owner decision: fix MIR, escalate native. Found while building §8.3's
  `stderr_bytes` field, by asking what each engine does with a `main` that returns something.
  - **The divergence**, against PROC-MAIN-001/PROC-EXIT-001 (07-Modules-and-Packages):
    `main -> Result<Unit, String>` returning `Err("boom")` — spec says status 1 with `boom\n` on
    stderr; oracle correct, **MIR status 0 with no stderr**, **native refuses to build**.
    `main -> Int32 { 3 }` — spec says status 3; oracle correct, **MIR status 0**, native refuses.
    `main -> Int32 { 300 }` — spec says trap `invalid-exit-status`; oracle traps, **MIR completes
    with status 0**, native refuses. `main()` returning `Unit` agrees three ways. So: two wrong
    outputs and a **missed trap**, §18.4's first two high-priority classes.
  - **Cause.** `run_program` matched `Ok(_)` on the entry call and hardcoded `status: 0`, discarding
    the entry's return value; `MirExecution` had no stderr field at all. The HIR oracle has
    implemented the rule correctly since Phase 4E, so the whole `Err`/`Int32` half of the entry
    contract was unobservable on the MIR side while looking like agreement on `Unit` programs — 0 is
    also what a `Unit` entry reports.
  - **MIR fixed** (`entry_termination`): status derived from the returned value, `MirExecution`
    gains `stderr`. **Not a contract change** — `MirExecution` appears nowhere in `mir.md`, the same
    test CD-084 applied to `FnKey`; no MIR shape, `RuntimeFn` or runtime-surface version moved.
  - **Native escalated as a Gate C6 blocker.** `Unsupported("the entry instance must return Unit to
    become Rust's fn main()")` refuses a program PROC-MAIN-001 declares a legal executable target —
    "a C5-style unsupported profile remaining for normative executable Core", which `WP-C6-ENTRY.md`
    §3 lists as **required result 6** for closing C6. A backend feature build does not belong inside
    a corpus package (§18.5).
  - **Two further escalations this produced, flagged not resolved.** (1) `invalid-exit-status` has
    **no `TrapCategory`** — the nine categories contain nothing for it, the oracle raises it
    uncategorised, and adding one is a **CE3** (WP-C6.0 froze trap identity); MIR therefore fails
    loudly there rather than completing with a wrong status. (2) **The Unit value is unwritable**:
    `02-Syntax-Grammar.md:324` declares `()` the Unit value, the checker rejects
    `let x: Unit = ()` (E0001 "expected 'Unit', found '()'"), and `Ok({})` fails at lowering — so
    PROC-EXIT-001's `Ok(Unit)` branch cannot be expressed in source. Spec-vs-checker conflict.
  - **A channel gap, recorded because it bounds §8.3.** `eprint`/`eprintln` are normative but
    observable in NO engine: the oracle writes them to the host process's stderr
    (`src/interp.rs:2779`) rather than into `Execution.stderr`, MIR has no lowering, native emits
    none. `stderr_bytes` can only compare the `Err`-completion write until that is closed. Not
    classified non-Core — §4.3 forbids exactly that reasoning.
  - **Retained** (§18.3): `starkc/tests/c65_entry_exit_contract.rs`, 7 tests — four two-engine cases
    checking every PROC-EXIT-001 field against the rule stated independently, and three boundary
    tests that each **name the condition that retires them** (native accepts a non-`Unit` entry; the
    trap gains a category; `()` typechecks as `Unit`). A boundary test that keeps passing after its
    boundary moves is exactly how O13 went stale.
  - **The matrix had NO row for any of this.** PROC-MAIN-001 and PROC-EXIT-001 appeared in none of
    the 133 rows; exit status was covered only as X12 (exit 101 after a trap). Rows **K15–K17**
    added, matrix now 136 rows, 4 BLOCKED (V19 + K15/K16/K17). So the §7.5 exit condition "no
    category silently omitted" did not hold when phase 0 was declared complete — **the second
    inherited disposition to fail on contact with a run**, after O13, which is the argument for
    C6.5-5's replay re-deriving all of them rather than trusting the matrix.

- CD-148 [2026-07-26, **OWNER DECISIONS on C65-F1, O13 and V19; WP-C6.5-1 comparator extracted**]
  Three dispositions and the plan's §19 commit 2.
  - **C65-F1 — option (1).** Extract the comparator, adopt it in `three_engine_differential.rs`,
    migrate the other 22 forked suites in COVERAGE-MATRIX order as C6.5 touches each category. Forks
    stay alive in the interim; a suite still on its own local helper is not evidence for the
    required claim until migrated, and §22's closure checklist is read that way.
  - **Commit 2 done, mechanically.** `starkc/tests/support/differential.rs` is now the comparator
    authority: engine runners, normalisation (`oracle_category`, `runtime_category`,
    `parse_native_trap`), `compare_outcomes`, the case entry points and the `three_engine_test!`
    macro, moved verbatim and made `pub`. `three_engine_differential.rs` keeps its case declarations
    and the comparator's own negative tests. Consumers include it with `#[macro_use] mod support;`
    (the existing `tests/common/mod.rs` convention); the macro uses absolute paths so a migrating
    suite needs that one line. **88 passed / 0 failed / 0 ignored / 0 self-skipped at the extraction
    commit `c789e4b` — identical to V0, which is the point of a mechanical move; 89 with the O13
    case below.** `fmt --check` clean, `clippy --tests` clean. **No observation-shape change**: §8.3–§8.10
    are commit 3, kept separate so a later disagreement is attributable to the extension, not the
    move.
  - **O13 (non-Copy array iteration) — the BLOCKER DID NOT EXIST; row was stale.** It was carried in
    from CD-038's "narrowed, not closed" (a runtime loop index names no `ConstIndex`; reading by
    copy would double-free). CD-038 also recorded what would close it — "unrolling or
    runtime-indexed drop flags" — and **WP-C6.1d took the unrolling option** (CD-084 G2, closing
    DEV-090). Two ledger records; the matrix inherited the older. Settled by EXECUTION, not by
    reading either: `o13_non_copy_array_by_value_iteration_agrees` pins stdout to `"idid\n"`
    independently of the engines, so a wrong Drop schedule (both elements at the end, or neither)
    fails even under unanimous agreement. All three engines produce it. Row → EXISTING-EVIDENCE.
    **Method note, deliberately recorded:** §3.6 exists to stop a legal Core program hiding behind a
    blocker, and here it was pointing at a program that already worked. The matrix's other 132
    dispositions were built the same way — from records rather than from runs — which is what
    C6.5-5's replay re-derives.
  - **V19 (`HashSet<T>`) — NOT-APPLICABLE-NON-CORE → BLOCKED-BY-OTHER-C6-WP.** §4.3(1) requires
    genuine absence from normative Core v1. `HashSet` is specified in 06-Standard-Library and named
    in the `std-full` profile; row V18 covers `HashMap` — equally `std-full` — as existing evidence,
    so "core-min only" is not the rule the matrix runs on; and CD-142's own words call the exclusion
    "a lowering gap like C6.3c's adapters", exactly the reason §4.3's closing line forbids.
    `c63d_map_key_identity::hashset_is_hir_only` pins the boundary and says so itself: *"if it now
    lowers, promote it to a three-engine case"*. A C6 blocker held for a lowering package, not a
    corpus exclusion.
  - **Matrix roll-up now:** 127 EXISTING-EVIDENCE, 4 NOT-APPLICABLE-NON-CORE (P08, P13, V20, K06),
    1 ADD-METAMORPHIC (K09), 1 BLOCKED (V19). 133 rows unchanged; the blocker count is unchanged at
    one and the row it names is not.
  - **This commit touches `starkc/tests`, so it INVALIDATES the WP-C6.4 Tier-1 records at
    `4844702`** under CD-146's re-qualification rule. Expected and already planned for: §3.5
    requires C6.4 evidence to be regenerated at the exact final corpus commit and forbids reusing
    older records once the corpus changes the commit. Row 24 remains BLOCKED-BY-C6.5 either way.

- CD-147 [2026-07-26, **WP-C6.5 OPENED — phase 0 done; the comparator is already forked 23 ways**]
  Baseline `b0d7a72` (the plan's `61008f6` had advanced six commits and is superseded). Tracked
  worktree clean; CI green, run 30192715611, all 11 jobs. V0: `exec_snapshots` 4,
  `mir_differential` 132, `three_engine_differential` 88, `c64_platform_matrix` 15, `fmt` clean,
  **0 ignored and 0 self-skipped in all four**. Full workspace not re-run locally — CI carries
  stronger exact-commit evidence for this commit and repeating a weaker single-platform version of
  it is not evidence.
  - **C65-F1, and it resizes phase C6.5-1.** The plan's §3.3/§8.2 assume ONE three-engine
    comparator to extract mechanically out of `three_engine_differential.rs`. Measured: **23 test
    files run all three engines, each with its own comparison logic** — every `c62*`, `c63*`,
    `native_c6*`, `native_c61f_*`, `cd139_float_division`, `native_c5_4_workspace`, and
    `three_engine_differential` itself. They share a SHAPE (assert HIR status, assert MIR status,
    assert HIR/MIR output equal, then native) without sharing CODE, and nothing calls the
    "shared" one — it is one of twenty-three, not the authority.
  - **Why that is a finding and not tidiness.** Every C6.2/C6.3/C6.4 claim about collections,
    strings, formatting, iterators, ownership and generics rests on one of these local helpers,
    each written to the standard its own work package needed. The union of 23 ad hoc definitions
    of "the engines agree" is not a definition — and C6.5's required claim is precisely that the
    three engines produce the same NORMATIVE observations. None of the 23 observes the §39 shape:
    no stderr bytes, no returned observation, no explicit Drop log. Every ownership row's Drop
    evidence today is printed stdout compared as ordinary output.
  - **Recorded for the owner with a recommendation, not resolved silently:** extract the
    `three_engine_differential` comparator, adopt it there, and migrate the other 22 incrementally
    as C6.5 touches each category (matrix order, not file order). The alternatives — migrate all 23
    at once, or leave inherited suites untouched — are stated in `WP-C6.5.md` §2 with their costs.
  - **`C6-CORPUS-COVERAGE-MATRIX.md`: 133 rows across the eight §7.3 groups**, every row carrying a
    normative citation and one of §7.4's dispositions, and citing existing evidence by exact case
    or test name. 126 EXISTING-EVIDENCE, 5 NOT-APPLICABLE-NON-CORE (P08 range patterns, P13 match
    guards, V19 `HashSet`, V20 files, K06 package alias — the last provisional pending a spec
    check), 1 ADD-METAMORPHIC (K09), **1 BLOCKED (O13, non-Copy array iteration — a real C6
    blocker under §3.6, narrowed and refused at CD-038, NOT a quarantine)**.
  - **126 EXISTING-EVIDENCE is not "nearly done", and the matrix says so.** It means the category
    SURFACE is exercised somewhere. Still owed: one comparator instead of 23; the full §39
    observation shape; a generated corpus (0 of ≥64 cases, 0 of ≥10 templates); metamorphic breadth
    (7 inherited groups against a floor of 24, and 5 of 12 families — M08–M12 — have no group at
    all); 16 mutation controls (0 exist); and adversarial sentinels, since the current dispatch and
    function-value cases prove a route WORKS rather than that the wrong route is observable.
  - **One flagged self-check:** V19's `NOT-APPLICABLE` rests on `HashSet` being absent from the
    `core-min` profile, not merely unrepresentable in MIR — §4.3 explicitly forbids the latter as a
    reason. If that reading is wrong, V19 becomes ESCALATION-REQUIRED. Stated in the matrix rather
    than assumed.

- CD-146 [2026-07-26, **OWNER DECISION — WP-C6.4 accepted as
  CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS**]
  The owner accepted the recommended status. Recorded so the ledger carries a decision rather than
  an open question.
  - **What this accepts.** Matrix rows 1–23 MET on two agreeing Tier-1 records at `4844702`
    (CI 30192449131, all 11 jobs green; 1705 passed / 0 failed each; TIER-1 AGREEMENT, reproduced
    locally). Row 25 REPORT-ONLY with G1 and G3 closed. The §34 portability audit and its ten
    findings, the owner's five review findings (R1–R5), and review passes A/B/C/D/E are complete.
  - **What this does NOT do: it is not closure, and no decision could have made it closure.**
    Row 24 — the deterministic generated corpus replayed on both Tier-1 targets — is WP-C6.5's, and
    the artifact does not exist. `CLOSED` becomes available only when C6.5's corpus replays through
    the harness C6.4 already built; that needs no new platform work, only the corpus.
  - **The re-qualification rule stands and is load-bearing.** Any commit touching `starkc/src`,
    `starkc/tests`, `starkc/scripts`, `starkc/target-matrix.json` or `stark-runtime` invalidates
    these records and requires a fresh qualification run. This is not boilerplate: it is exactly
    what forced the `61008f6` records to be discarded this round despite their having passed.
  - **Carried forward, open and named** — none blocking this status: row 24 (C6.5); gap-report G2
    (two installer scripts asserting the same thing) and G4 (`/tmp` in a gate-7 fixture), both
    harness, neither semantic; `LinkerOrExternalToolFailure` still conflated with generated-crate
    compile errors inside `BackendDiagnostic::BuildFailed`; and the file-not-found mapping probe,
    which is unrun because `std-full` file operations are absent from every engine, so there is no
    mapping to probe.
  - **The lesson this package leaves.** THREE controls shipped with indistinguishable success and
    failure states — the ignore classification, the skip detector that could not observe a skip
    (libtest hides passing output), and a Windows step that failed the job by asserting correctly
    (`$LASTEXITCODE` leaked through `pwsh`). Each was validated against its happy path only. The
    compensating discipline is `scripts/test_c64_scripts.py`: 43 tests, each mutating exactly one
    thing and asserting the REFUSAL. Apply that shape to C6.5's mutation controls (§43), which are
    the same problem stated as a work package.

- CD-145 [2026-07-26, **WP-C6.4 tier-1 evidence retaken under the strengthened gate; a check that
  failed by succeeding**]
  CI run 30192449131 at `4844702`, **all 11 jobs green**. Both tier-1 records: 1705 passed / 0
  failed, 2 ignores (both classified by full libtest name), 0 unclassified, 0 self-skipped, no
  deviations, determinism `match`, pointer width 64, `stark-64-v1` v1 rev 1. Identical per-command
  counts. `qualification-summary.md` reports TIER-1 AGREEMENT, **and I reproduced that verdict
  locally against the downloaded records** — the claim does not rest on a CI job having exited zero.
  - **The Windows release smoke failed twice, and BOTH times the check itself was correct.** The
    step logged `installed stark correctly refused to build without its installed runtime (exit 1)`
    and then failed the job on that same `1`: GitHub appends `exit $LASTEXITCODE` to every `pwsh`
    step, and `$LASTEXITCODE` was still 1 from the build the check DELIBERATELY makes fail. A
    passing assertion and a failing step were therefore the same observable state, which makes the
    step unreadable in both directions — a real regression would have looked identical. Fixed with
    `exit 0` after the assertions. The bash branch never had it (its last command is the echo),
    which the `279b4a7` run confirms: linux and macos smokes passed with the same negative check.
  - **This is the third control in this package whose success and failure states were
    indistinguishable until CI ran it** — after the ignore classification (CD-144 R-context) and the
    skip detector that could not see a skip (CD-144 R3/R4 neighbourhood). Recorded as a pattern
    rather than three unrelated fixes: every one was a check I wrote, validated locally against the
    happy path, and shipped without exercising its failure path. The compensating discipline that
    did work is the one now in `test_c64_scripts.py` — 43 tests, each mutating exactly one thing and
    asserting the REFUSAL.
  - **Evidence committed:** `docs/compiler/evidence/c6.4/{macos-arm64,linux-x64}.{json,md}` and
    `qualification-summary.md`, downloaded from the runners, not regenerated. Matrix Table B rows
    1–23 MET at `4844702`.
  - **Status: `CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS`.** Row 24 (deterministic generated corpus)
    is C6.5's and cannot be satisfied here, so `CLOSED` remains unavailable. Owner decision is the
    only outstanding step.

- CD-144 [2026-07-26, **WP-C6.4 owner review round — five findings, and the Tier-1 records withdrawn**]
  The owner reviewed the delivered package and found five defects. All fixed. Stated as the defect,
  not the fix, because the pattern in three of them is the same: **a control that was proven
  somewhere other than where it operates.**
  - **R1 — the installed-runtime proof was a unit test, not the real path.**
    `STARK_REQUIRE_INSTALLED_RUNTIME=1` was proven to disable the checkout fallback in a unit test
    while the actual release smoke — the one that installs a package and runs the installed `stark`
    — did not set it. The thing shipped to users was still the unproven path. Now set on all three
    platforms, AND paired with the negative half that makes the positive half mean anything: with
    the installed runtime moved aside and the checkout still present at the compiled-in path,
    `stark build` must FAIL. Without that step, a passing build proves nothing about WHICH runtime
    it used.
  - **R2 — a failed qualification job silently SKIPPED the comparison.** `c64-tier1-comparison`
    lacked `if: always()`. A skipped job is worse than a failing one: it reads as "not applicable"
    rather than "not established". §10.4 forbids workflow-level skipping standing in for an
    explicit TIER-1 DISAGREEMENT. The comparison now runs after success, failure or cancellation,
    and reports a missing or unreadable record AS a disagreement with a named cause.
  - **R3 — the comparator could reach agreement from incomplete evidence.** Two records that both
    omitted a field agreed on it. Each record is now VALIDATED before the two are compared:
    required metadata non-blank, self-consistent platform identity (selected==host, tier-1, 64-bit,
    declared contract), positive layout contract version and revision, every command of the fixed
    set present and passing, corpus exactly `BLOCKED-BY-C6.5` with zero cases, determinism matched
    with non-blank hashes, no deviation/dirty/quick/unclassified/self-skip. Then compared —
    per-command exit code, all four counts, normative argv, and FULL ignored-test identities.
  - **R4 — ignored-test identities were truncated to the last `::` component.** Two modules can each
    hold a `basic_case`; collapsing them would let a classified ignore vouch for an unrelated
    unclassified one. Complete libtest names now, in a list so the count survives, and the named
    count must equal Cargo's ignored count.
  - **R5 — two documentation claims were stale or overstated.** (a) Review A said float division
    follows CD-006; **CD-006 is SUPERSEDED by NUM-FLOAT-OP-001 and CD-139** — my own CD-139 entry
    records that succession, and I then cited the superseded decision three days later. (b) Review
    A(4) claimed the absence of `cfg` in the runtime PROVED the platforms cannot diverge. It does
    not: identical source can still diverge through the host toolchain, LLVM, libc or floating-point
    behaviour beneath it. Corrected in both this file and `WP-C6.4.md` to the accurate claim — no
    target-conditional semantic implementation, therefore REDUCED RISK, with actual equivalence
    established by the cross-platform observations.
  - **B-1 fully resolved, not merely checked.** `build-release.py` classified Windows with
    `"windows" in target` — wrong in two directions, since it misclassifies an unknown triple
    containing the word AND packages triples the compiler does not name at all. There is now ONE
    description, `starkc/target-matrix.json`, read by every Python consumer through
    `scripts/target_matrix.py` and pinned to `src/target.rs` in BOTH directions by
    `target_matrix_json_matches_the_compiler`. Checking one direction catches half the drift — the
    half noticed first. Gap-report G3 CLOSED.
  - **THE TIER-1 RECORDS WERE WITHDRAWN.** `61008f6`'s records passed and agreed, but the
    strengthened comparator REFUSES them — they lack `target_pointer_width`,
    `layout_contract_version`, `compiler_layout_revision` and `required_steps`, and their ignore
    identities are truncated. Verified by running the new comparator against them. Keeping them
    would claim qualification from evidence the current gate rejects, so they are deleted and
    retaken at the corrected commit. Matrix Table B rows 1–23 reset to `pending`.
  - **Evidence:** `c64_platform_matrix` 15, `test_c64_scripts.py` **43 (new)**,
    `test_build_release.py` 6 (4 new), `--list` ok, `fmt --check` clean, **`clippy --workspace
    --all-targets --all-features -D warnings` clean (5m28s)**. The full workspace suite is CI's,
    per the standing rule recorded in CD-142.

- CD-143 [2026-07-26, **WP-C6.4 OPENED and BUILT — the compiler had no notion of a target**]
  The Tier-1 platform matrix. Owner directed it to start ahead of the C6.3 confirming run; that run
  landed first anyway (CD-142), so C6.4 opens on an admitted runtime. Baseline `5d2c85d`.
  - **The finding that shapes the package.** Before this, *the compiler had no target
    classification at all*: `backend/generated_rust/build.rs` read `host:` out of `rustc -vV` and
    used it as the target, `stark-64-v1` was applied to whatever that turned out to be, and the
    executable suffix came from the COMPILER'S OWN `std::env::consts::EXE_SUFFIX`. Three separate
    places for one assumption, each correct exactly while host and target are the same string.
    Nothing rejected an unsupported target, so rustc or the linker would necessarily have been the
    first detector — a §14 stop condition. `starkc/src/target.rs` is now the single place a triple
    is interpreted; every other site asks it.
  - **Ten host assumptions found (F1–F10 in `WP-C6.4.md` §2), eight fixed.** The one worth naming:
    **F3** — `stark-runtime/src/vec.rs` checked bounds as `i as usize >= v.len()` with `i: u64`, at
    four sites. On a 32-bit target the cast TRUNCATES first, so `v[0x1_0000_0000]` on a one-element
    vector narrows to `0`, passes the check, and RETURNS ELEMENT 0 instead of trapping
    `IndexOutOfBounds`. Unreachable on Tier 1 (both targets are 64-bit) and therefore not a live
    defect — but it is exactly the class the audit exists to expose, and it was load-bearing the
    moment F2 (any triple inherits `stark-64-v1`) stopped being hypothetical. Fixed on BOTH axes
    independently: `narrow_index` compares in `u64`, and preflight admits only named 64-bit targets.
  - Also fixed: **F4** the compiled-in source-checkout runtime fallback could make an
    installed-runtime test pass for the wrong reason (`STARK_REQUIRE_INSTALLED_RUNTIME=1` now turns
    it off); **F5** the generated crate was built `--offline` but never `--locked` and had no lock at
    all (both added; the lock's runtime version is READ from the runtime being linked, not
    hardcoded); **F6** the generated `Cargo.toml` escaped paths with Rust's `Debug` used as TOML
    quoting, which diverges on control characters and non-UTF-8 bytes; **F10** three of §8.3's six
    error classes did not exist.
  - **NOT DONE — and this is the substance of C6.4, not a detail.** No Tier-1 platform record
    exists. The harness (`scripts/run-c64-qualification.py`), the comparison gate
    (`scripts/compare-c64-evidence.py`, which requires the two records to be for the two DIFFERENT
    Tier-1 targets at one commit with matching per-command observations) and three CI jobs are
    written, but `docs/compiler/evidence/c6.4/` holds no `.json`. That directory's README says so
    explicitly, because a locally simulated record would defeat the only purpose those files have.
    Formal review passes A/D/E (§12) are also not written up; C was performed as the §2 audit.
  - **Row 24 is permanently blocked inside this package.** The deterministic GENERATED corpus is
    C6.5's (the `WP-C6.5` chapter of `WP-C6-ENTRY.md`, §§38–45); `tests/exec_snapshots/corpus.lock` is the FROZEN
    execution corpus, a different artifact already covered by other rows. Every evidence record
    carries `generated_corpus_status: BLOCKED-BY-C6.5` so the state is asserted, not merely absent.
  - **FIRST CI RUN (`8d894e8`, run 30190825336): 9 of 11 jobs green, and the two failures were the
    HARNESS DOING ITS JOB.** (a) **The Windows gap probe PASSED 14/14** — the first run of the C6.4
    suite on a platform outside the claim: exact stdout bytes with no CRLF, identical trap category
    and `file:line:column`, exit 101, the flushed pre-trap prefix, `--locked --offline` under
    Windows Cargo, and builds under spaced and Unicode paths. Gap-report G1 closes as `portable`.
    That is evidence about the SHARED RUNTIME, not about Windows — and it was not guaranteed.
    (b) **Both Tier-1 qualification jobs FAILED, correctly**, on `workspace: 2 test(s) ignored in a
    required command`. The two ignores are pre-existing opt-in tensor-track tests needing external
    artifacts. The defect was MINE, in the harness: §10.4 permits an ignored test "unless explicitly
    classified outside the required matrix", and I built the refusal without the classification.
    Fixed by NAMING them — `CLASSIFIED_IGNORES` is a closed list with a reason per entry, not a
    count, because counting would let a new ignore silently replace a retired one. The harness now
    parses `test <name> ... ignored` lines, fails any unclassified name, ALSO fails when a nonzero
    ignored count cannot be attributed to names, and records both sets in the evidence.
  - **Review passes A/B/D/E performed** (`WP-C6.4.md` §4.5), against the tree rather than from
    memory. `grep -rn "cfg(" stark-runtime/src` excluding `cfg(test)` returns NOTHING, so the
    runtime contains **no target-conditional semantic implementation** — which REDUCES divergence
    risk rather than proving its absence. (An earlier draft of this entry said the two platforms
    "cannot take different semantic paths"; that overstated it. Identical source can still diverge
    through the host toolchain, LLVM, libc or the floating-point behaviour beneath it. Actual
    Tier-1 equivalence is established by the exact cross-platform qualification observations, not
    by the absence of `cfg`.) **Review B found a real duplication** (B-1): the Python qualification
    scripts carried their own tier table, exactly what §8.2 forbids. One probe is honestly NOT run:
    file-not-found mapping, because `std-full` file operations are absent from every engine.
  - **THE QUALIFYING RUN: CI 30191381334 at `61008f6`, both tier-1 jobs and the agreement gate
    green.** macos-arm64 and linux-x64 each: 1705 passed, 0 failed, 2 ignored (both classified), 0
    unclassified, 0 self-skipped, determinism `match`, rustc 1.97.1. Identical per-command counts —
    `c64_platform_matrix` 15, `three_engine_differential` 88, `mir_differential` 132,
    `exec_snapshots` 4, `c63_closure_evidence` 2, `conformance` 3, `workspace` 1461. The records are
    committed AS DOWNLOADED from the runners, and deliberately NOT taken from the two earlier
    passing runs (`9ff8d35`, `e80df80`): the harness changed after each, and evidence has to
    describe the commit it claims.
  - **A third harness defect, found by reasoning rather than by a run: THE SKIP DETECTOR COULD NOT
    SEE A SKIP.** Eleven native/differential suites print `SKIP:` and return SUCCESS when no rustc
    is present, and the harness failed a required command on that — except libtest DISCARDS a
    passing test's output, so the line was invisible under a plain `cargo test`. A detector that
    cannot observe what it detects is worse than none, because it reads as coverage. Every step
    whose suite can self-skip now runs `-- --nocapture`; the workspace step does not (its output
    would be enormous) and its narrower guarantee is stated in the harness docstring rather than
    left implied. PROVED by running the built c64 binary under `env -i PATH=/usr/bin`: 7 SKIP lines
    appear and the step fails.
  - **Evidence for this commit (scoped, per the CD-142 rule — CI is the exhaustive net):** `--lib`
    463 + `stark-runtime --lib` 23, `c64_platform_matrix` 14, `native_build_cli` 9,
    `c63_closure_evidence` 2, `native_c5_1b_skeleton` + `native_c5_3_aggregates_enums` 20,
    `native_c5_2b_locals` 2 — the last five are what prove `--locked` plus the emitted lock builds
    under real Cargo. `fmt --check` clean. Clippy and the full suite are CI's.
  - **Two `native_build_cli` tests pinned the old Cargo argv** and were updated, not worked around:
    they now assert `build --locked --offline`, which is a stronger assertion than the one they
    replaced.

- CD-142 [2026-07-26, **WP-C6.3 CLOSED — on a full three-platform run**]
  CD-138 item 7 required C6.3 to be re-closed "on a full clean run" and not before. That condition
  is now met, by the CI run for `1ef4e8b` (Actions run 30188909346), **all 7 jobs green**:
  `cargo test --workspace --all-targets --all-features` on **linux-x64, macos-arm64 and
  windows-x64**, plus release-package smoke on all three and spec-fixture conformance. `cargo fmt`
  and `clippy --workspace --all-targets --all-features -D warnings` are clean in the same jobs.
  - **This is stronger evidence than the local run it replaced**, in two ways that matter. It is
    `--all-targets --all-features` rather than the plain `--workspace` I had queued, and it is three
    platforms rather than one. The specific risk I had named — `loop_aware_order` changing block
    emission for every generated body with a loop, where a plan that VALIDATES but is wrong yields an
    infinite loop rather than a failure (the CD-127 precedent) — is exercised by every native target
    on every platform, including a Windows host whose toolchain and linker differ from this one.
  - **The local "wave 2" did NOT contribute to this closure.** I terminated it (`SIGTERM`, exit 143)
    after 3 of 24 targets once CI made it redundant. Recorded explicitly so the evidence trail is not
    read as 24 local targets plus CI.
  - **CI was RED for the four commits before this one** (`3f8e993` onward, CD-141) — so this is also
    the first green build since the C6.3e work began, and the first evidence that the whole series
    holds together on Linux and Windows rather than only on the development host.
  - **What C6.3 closes as.** a/b/c/d/e closed; f (files) EXCLUDED — absent from every engine and in
    the optional, already-unclaimable `std-full` profile. Carried forward as excluded-by-decision,
    not defects: `HashSet` (HIR-only, no MIR representation), Drop-bearing map keys/values,
    composite `Box` elements (CD-125), `HashMap`/bare-struct `Display` (CD-136, CE-shaped), and the
    six iterator forms in `WP-ITER-LOWERING-PROPOSAL.md`. Deprecated but present:
    `CheckedOp::FloatDiv`/`FloatRem`, whose removal is a separately versioned cleanup.

- CD-141 [2026-07-26, **CI RED SINCE `3f8e993` — the three-engine harness ignored the category the
  oracle states**]
  The GitHub Actions run for `3f8e993` failed on all three platforms (macOS, Linux, Windows), on
  `panic_message_agrees_across_engines` and `conditional_panic_message_agrees_across_engines`. Not
  product code — the harness. **This is the deferred failure from committing before the full suite
  finished**, on the owner's instruction "commit and push for now; if the full suite fails it will
  be resolved". Resolved here.
  - **The defect.** CD-136 (DEV-106) added `RuntimeError::trap_category` for exactly one reason:
    `panic(msg)` raises arbitrary USER text, so no prose table can classify it. The harness then
    kept classifying by prose anyway and never read the new field, so both `panic` cases hit
    `oracle_category`'s deliberate unrecognised-message failure — the guard doing its job against a
    caller that should not have reached it. The call site even tested `category ==
    TrapCategory::Panic` to decide whether to carry the message, a branch `oracle_category` could
    never produce.
  - **Fix:** the STATED category wins when the oracle supplies one; prose matching stays as the
    fallback for every trap raised without one. One line, plus the reason.
  - **Why it escaped locally:** the two cases were added in the same change and the suite was not
    run to completion before pushing. The lesson is the one already recorded for lowering refusals —
    a new test that exercises a NEW field must run before the commit that introduces both.

- CD-140 [2026-07-26, **DEV-109 CLOSED — `Float32` VALUES are binary32, not just `Float32` DISPLAY**]
  Owner directive: DEV-109 stays inside WP-C6.3 rather than being re-scoped to a C4-era defect.
  - **What was wrong.** DEV-105 gave `Float32` a print operation that honours the declared width,
    which fixed RENDERING. It did not make the VALUE binary32. Both interpreters carry every float
    in an f64, so a `Float32` local could hold a value no f32 can represent, and only the printer
    rounded it. That is the worse failure mode of the two: a number that PRINTS as `inf` while
    arithmetic still treats it as finite looks correct at the point a developer would check it.
    NUM-FLOAT-FORMAT-001 requires IEEE binary32 for `Float32`, and NUM-FLOAT-REPRO-001 requires the
    same result bits for the same declared type and sequence of operations — both about VALUES.
  - **The HIR oracle was already right**, via `normalize_numeric`, which narrows to f32 whenever the
    expression's static type is `Float32`. Only MIR was wrong, so this was a live HIR↔MIR divergence
    of the same family as CD-133 (HashMap keys) and CD-139 (float division) — the third this gate.
    Native was never wrong: it holds a real `f32`.
  - **Fixed at three points, mirroring the oracle.** (1) **Literal:** a `Float32` literal lowers to
    the nearest BINARY32 value carried in the f64 constant — NUM-FLOAT-LIT-001 converts a decimal
    literal directly to the DESTINATION format, so `0.1f32` denotes the f32 nearest 0.1, and storing
    the f64 nearest 0.1 made the constant observably wider than its own type. (2) **Cast:**
    integer-to-`Float32` rounds once (NUM-FLOAT-CONV-001); it had been sharing the `Float64` arm, so
    it did not round at all. Float-to-`Float32` already narrowed. (3) **Assignment:** any value
    stored into a `Float32` destination is rounded to binary32. Every float rvalue reaches a typed
    destination, so that one site covers arithmetic, negation and operand reads together — the
    destination's declared type is the MIR-level equivalent of the oracle's static expression type.
  - **Evidence: 8 new three-engine cases in `tests/c63e_float32.rs` (21 total).** Widening a literal
    exposes its narrowing (`0.1f32 as Float64` → `0.10000000149011612`), arithmetic rounds at every
    step, overflow becomes a REAL infinity so `inf - inf` is `NaN` (the case that exposed the
    defect — it previously stayed finite at `3.4e39` and merely printed as `inf`, so the subtraction
    gave `0.0`), underflow reaches exactly zero, `16777217 as Float32` rounds to `16777216.0` (the
    first integer binary32 cannot represent), `Float32` division by zero, NaN surviving a widening
    cast and staying unordered, and a ten-iteration accumulation where a missing per-step rounding
    compounds rather than cancelling.
  - **CD-139 is what made this testable.** Constructing an `inf` or a `NaN` requires a division by
    zero, and that trapped in MIR until CD-139, which is why `c63e_float32.rs` originally had to
    scope its edges to infinities reached by overflow and skip NaN entirely. The two defects were
    found together and had to be fixed in that order.

- CD-139 [2026-07-26, **DEV-110 CLOSED — float division/remainder are TOTAL; CD-006 superseded;
  MIR amendment A6**]
  **Owner ruling: "CD-006 is superseded — not reversed on its merits — by the later normative WP-C2.9
  drafting of NUM-INT-DIV-001 and NUM-FLOAT-OP-001; HIR, MIR, and native execution must align with
  those later rules."** Succession of authority, not a change of mind.
  - **The evidence that made this a succession rather than a conflict.** I first reported DEV-110 as
    a live spec-vs-decision standoff needing a merits ruling. That framing was wrong, and the
    primary sources say so: (a) CD-006 arbitrated the sentence "Division or modulo by zero is a
    runtime error and MUST trap" in `03-Type-System.md`, which is **no longer in that file**;
    (b) CD-006 landed 2026-07-18 08:47 (`785c1be`) and NUM-FLOAT-OP-001 landed the same day at
    17:29 (`b702a31`, WP-C2.9) — nine hours later; (c) WP-C2.9 deliberately SPLIT the cases into
    adjacent paired rules, NUM-INT-DIV-001 "integer division by zero and remainder by zero trap"
    and NUM-FLOAT-OP-001 "floating division by zero does not trap", which is authoring intent, not
    an oversight; (d) TRAP-CATEGORY-001 defers to "the owning numeric rule" and so does not
    re-create the ambiguity; and (e) CD-006's own text records "No spec or code edits made under
    this decision" — it was a do-not-re-litigate note pinned to prose that was then rewritten.
    (f) The HIR oracle already followed the spec: `interp.rs`'s "division by zero" error is inside
    the INTEGER arm, and there is no float trap anywhere in it. Only MIR had one. Charter §1.6 rule
    6 makes the interpreter the semantic reference, so MIR was the straggler.
  - **MIR amendment A6 (CE3, owner-approved), narrow and additive:** adds `MirBinOp::FloatDiv` and
    `MirBinOp::FloatRem`. **This was the owner's correction to my implementation plan.** I proposed
    "emit a plain `BinOp`" as if that avoided a shape change; it does not — `MirBinOp` held only
    `FloatAdd`/`FloatSub`/`FloatMul`, and `FloatDiv`/`FloatRem` existed ONLY under `CheckedOp`. The
    owner's reasoning for amending rather than economising: keeping a total IEEE operation inside
    `CheckedOp` would preserve the enum shape while corrupting its contract — a primitive declared
    trapping that is guaranteed never to trap. `MIR_VERSION` stays `0.1` (additive variant, the A5
    precedent); the runtime surface is untouched (no `RuntimeFn`).
  - `CheckedOp::FloatDiv`/`FloatRem` are **retained, deprecated, and unreachable**, so the amendment
    stays additive. Removal is a separately versioned cleanup, not part of this change.
  - **Evidence: `tests/cd139_float_division.rs`, 13 three-engine cases.** Signed infinities (both
    signs, and by a NEGATIVE-zero divisor — the sign of the divisor selects the sign of the
    infinity, which a "return infinity on a zero divisor" shortcut would miss), `0.0/0.0` → NaN,
    all three NaN producers for `%` (zero divisor, infinite dividend, NaN operand), an ordinary
    remainder that still computes, `Float32` on the same path, NaN propagation through `+`/`*`/`-`,
    NaN's unordered comparisons, and a shape assertion that lowering no longer emits the deprecated
    checked ops.
  - **Half the file guards the OVER-correction, and that is deliberate.** "Division by zero no
    longer traps" is true of floats and false of integers; a fix applied to the headline rather
    than to NUM-FLOAT-OP-001 specifically would silently make integer division total. Signed and
    unsigned integer `/` and `%` by zero must still trap in every engine, and are pinned here.
  - **Unblocks DEV-109's evidence.** `inf` and `NaN` previously could not be CONSTRUCTED in a test:
    every route ran through a division by zero, and that trapped in MIR. `c63e_float32.rs` had to
    scope its edge cases to infinities reached by overflow for exactly this reason. `inf - inf` is
    now a reachable case — and it is the one that exposes DEV-109 most sharply.

- CD-138 [2026-07-26, **C6.3 CLOSURE CORRECTION — DEV-105 CLOSED (0.1-A9); WP-C6.3 back to PARTIAL**]
  An external review rejected CD-137's closure claim, correctly. I had marked WP-C6.3 complete while
  DEV-105 sat as a KNOWN WRONG-OUTPUT defect inside the admitted domain — not an excluded feature.
  Those two statements cannot coexist, and the reviewer was right that the second invalidates the
  first. **WP-C6.3 and C6.3e are PARTIAL.**
  - **CE3 APPROVED and implemented (owner): `PrintFloat32`/`PrintlnFloat32`, `MIR_RUNTIME_SURFACE`
    0.1-A8 → 0.1-A9.** Additive; `PrintFloat64`'s arity and meaning are untouched.
  - **DEV-105 CLOSED.** PRINT-DISPLAY-001 renders a float at its DECLARED IEEE width, so `0.1f32`
    must print `0.1`. This was never an open semantics question — the spec answers it — only a
    missing width-preserving operation. `Float32` no longer passes through `widen_for_print` in
    EITHER the scalar or the composite path; the verifier REQUIRES a `Float32` operand (the declared
    width is part of the operation's identity, not a convention); the MIR interpreter narrows its f64
    storage at that boundary; the backend calls an `f32` runtime function. All three route through
    the one `canonical_float32`. **The composite Float32 refusal is removed** — tuple, array,
    `Option`, `Result` and `Vec` all render `Float32` elements.
  - **Correction to the review's instruction on the frozen corpus:** it asked for a non-binary-exact
    value to be added because `2.5` "cannot detect width substitution". The corpus ALREADY prints
    `0.1f32` and records `0.1`, so no change was needed — and that fact refined the diagnosis: since
    `mir_differential` passed, MIR was already printing `0.1`. Only NATIVE was wrong. MIR agreed by
    accident (its constant never actually narrowed to f32); it now agrees for the right reason.
  - **Evidence:** new `tests/c63e_float32.rs`, 11 three-engine cases — scalar `println`/`print`, a
    value whose f32 and f64 renderings visibly differ, tuple/array/`Option`/`Result`/`Vec` elements,
    negative zero, max finite, min subnormal, and infinities.
  - **THREE NEW DEFECTS, found by writing that evidence.** Each is value semantics, not formatting,
    so each is recorded rather than absorbed into a Display slice:
    - **DEV-109 — `Float32` arithmetic does not maintain binary32 precision.** Both interpreters hold
      a `Float32` as f64 and round only AT DISPLAY. So `0.1f32 as Float64` is a no-op in MIR (giving
      `0.1`) while HIR rounds (giving `0.10000000149011612`), and an overflowing `Float32` product is
      stored unrounded — `3.4e39`, not `inf` — so `inf - inf` yields `0.0` instead of `NaN`. The
      RENDERING becomes an infinity while the VALUE never does. NUM-FLOAT-FORMAT-001 requires IEEE
      binary32 for all value observations.
    - **DEV-110 [ESCALATED — a spec-vs-decision conflict, not a bug to pick a side on].**
      NUM-FLOAT-OP-001: "floating division by zero does not trap: it produces the IEEE infinity or
      NaN result." **CD-006** is a recorded OWNER decision (2026-07-18) to keep trapping for floats,
      taken when the spec text was ambiguous. The normative text is now unambiguous and contradicts
      it. HIR follows the spec (yields `inf`); MIR follows CD-006 (traps `DivideByZero`). Charter
      §1.6 rule 1 says the spec governs and rule 3 forbids inventing a third behaviour — but
      overriding a recorded owner decision is not mine to do, and CD-006 was itself flagged CE2-shaped
      rather than resolved unilaterally. It returns the same way.
    - Both blocked the obvious way to construct `inf`/`NaN` in a test, which is how they surfaced.
  - **CD-138 also hardens the C6.3d `Eq` dispatch (review item 4).** `Option<usize>` conflated "a
    primitive key, which legitimately compares structurally" with "a nominal key whose `eq_impls`
    entry is missing" — and the backend REFUSES the second, so the MIR interpreter would have
    silently executed structural equality for a program native declines to build. Replaced by an
    explicit `KeyEqMode { Structural, UserEq(index), MissingForNominal }`, where the third is an
    INTERNAL ERROR. A nominal key always has an entry (it needs `impl Eq` to satisfy the key bound),
    so a missing one is a compiler defect and now says so.
  - **DEV-108 CLOSED — FIXED, not refused, and the diagnosis inverted the framing.** The review
    asked for a precise pre-rustc refusal, suggesting a predicate over the payload's drop plan. That
    would have been wrong, because the payload was never the cause. The body fell back to the
    DISPATCH loop, where a `match` on a runtime value makes every local live in every arm, so the
    payload borrow appeared live across the slot's drop glue — `E0502`. It fell back because a plain
    RPO does not keep a natural loop's blocks CONTIGUOUS, and a `Loop` scope is an RPO SPAN: the
    `Vec` render loop's header landed at index 11 with its body at 20-29 and eight unrelated blocks
    between, so `structured_plan` correctly abandoned the plan rather than emit a loop that
    re-executes non-members. The DFS simply took the loop-EXIT successor first at that header.
    `Option<Vec<String>>` worked only because its DFS happened to go the other way — so a guard on
    the payload type would have refused a working program AND missed every other shape with the same
    ordering accident. Fixed by `loop_aware_order`: emit a block once every forward predecessor is
    emitted, preferring the innermost open loop's members, closing a loop when none is ready (sound
    because a reducible loop is single-entry). Both `Result<Vec<String>, Int32>` variants now render
    three-engine. **General consequence: fewer bodies fall back to dispatch, so borrow precision
    improves across the backend, not just for this shape.**
  - **Still open, and why C6.3 stays PARTIAL:** DEV-109 and DEV-110 — both `Float32` VALUE
    semantics, both outside Display, and DEV-110 needs an owner ruling rather than an implementation.
  - **Governance correction (review item 3):** CD-134's Drop-bearing exclusion is recorded as
    "per the owner's closure ruling", and that is accurate — it was a direct answer to a question put
    to the owner offering exclusion or full implementation. It is NOT derived from the earlier
    review, which presented both outcomes without choosing. Stated here explicitly so the record
    shows a superseding decision rather than an inherited one.

- CD-137 [2026-07-26, **WP-C6.3f EXCLUDED + the C6.3 CLOSURE EVIDENCE discharged (CD-116)**]
  The two remaining C6.3 items, resolved in opposite directions — one excluded on evidence, one
  satisfied with new evidence.
  - **WP-C6.3f (files) — EXCLUDED, not built.** `File` is implemented NOWHERE: zero mentions in
    `interp.rs`, zero in `mir/lower.rs` (only four in `typecheck.rs`). So it is not a native-parity
    gap at all — nothing exists for native to fall behind. Two further facts settle it: `std/io/` is
    its own module (the spec's own layout, analogous to `System.IO`), and file IO is **`std-full`**,
    which `STD-PROFILE-001` makes an OPTIONAL capability — Core v1 conformance requires only
    `core-min`. Building it would mean an entire std module across HIR, MIR, runtime and native, plus
    STD-IO-001's resource semantics (a non-`Copy` `File` whose ownership moves but cannot be cloned,
    UTF-8-validating reads, short-write handling, and "dropping an open file attempts close but
    cannot surface a new language trap" — which reaches into the Drop/trap machinery).
  - **Why excluding it costs nothing that was still available.** `std-full` is *indivisible* — a
    claim requires everything in it. `HashSet` (CD-134) and the iterator combinators (CD-130) are
    already excluded, and both are `std-full`, so the profile was ALREADY unclaimable. Excluding
    files changes what STARK implements, not what it can advertise: `core-min` plus a partial,
    unclaimable subset of `std-full`. If file IO is wanted it deserves its own std-library work
    package with its own scope, exactly like `WP-ITER-LOWERING-PROPOSAL.md`.
  - **Note for the record:** the io module's `core-min` half — `print`/`println` — has been native
    since CD-113. It is only the `std-full` half that is absent.
  - **C6.3 CLOSURE EVIDENCE discharged (CD-116).** That requirement — runtime version review plus
    installed-layout and offline-build proofs — was recorded as "must land before C6.3 closes" and
    had not. New `tests/c63_closure_evidence.rs`, 2 cases:
    (a) **installed runtime + offline build.** `NativeToolchainOptions::runtime_crate` is a PATH and
    every other native test points it at the working tree; this test COPIES the runtime (Cargo.toml +
    `src/*.rs` only — no `target/`, no `.git`) into a temp directory and builds against the copy, so a
    program that only compiled because of something in the checkout fails here. It exercises what
    C6.3 ADDED — composite formatting, `String`, `Vec`, `iter()`, `HashMap` — and asserts exact
    output. The offline half needs no separate test: `build_and_link` passes `--offline`
    unconditionally, and the copied crate has neither a vendored registry nor network, so a runtime
    dependency regression fails HERE.
    (b) **version identity is CHECKED, not merely recorded** — a stale linked runtime is rejected
    before user code runs (§9.2), with the matching case asserted first so the rejection is not
    vacuous. This matters precisely BECAUSE the runtime can now be installed separately.
  - **Both proven to fail before being trusted:** the installed-runtime assertion was inverted and
    observed to fail against the real binary's output.

- CD-136 [2026-07-26, **WP-C6.3e — DEV-106 CLOSED (trap-message parity); a CD-135 regression fixed**]
  Three changes: the deviation that was the point of this slice, a defect CD-135 introduced and an
  external probe caught, and one recorded deviation left open on purpose.
  - **DEV-106 CLOSED — trap MESSAGE parity across engines.** The three-engine harness compared trap
    category and location but not TEXT, because it REFUSED message-carrying traps outright ("needs
    string values — outside the C5.2-admitted surface"): `panic(msg)` was never compared at all. That
    refusal was stale once strings landed in C6.3a. `Outcome::Trapped` now carries
    `message: Option<String>`, filled by all three engines — MIR from its own trap payload, native by
    parsing the line `trap::abort_with_message` prints after the `-->` location, and HIR from the
    error text. **`RuntimeError` gained `trap_category: Option<TrapCategory>`** so the interpreter
    STATES a `panic`'s category instead of leaving it to be recovered from prose: a user message is
    arbitrary text that no prose table can classify, which is exactly why the harness had to reject
    it before. Every other trap keeps its prose-matched category.
  - **Proven to FAIL before being trusted to pass** (the CD-053 discipline): the comparator's own
    self-test now includes cases where only the MESSAGE differs, and where one engine loses it
    entirely, asserting rejection names the disagreeing pair. Plus two real three-engine cases —
    `panic("the sky is falling")` and a conditional panic after output, so the message is compared
    alongside the pre-trap stdout prefix.
  - **A CD-135 REGRESSION, found by an external probe and confirmed here.** CD-135 made an owning
    `Vec` element arrive as `&T` but only made the `Vec`/`String`/`str` arms reference-aware. An
    AGGREGATE element then reached the tuple/array/`Option`/`Result` arms behind a reference and they
    projected straight through it, emitting ILL-FORMED MIR: `Vec<(String, Int32)>` → MIR-0003,
    `Vec<[String; 2]>` → MIR-0010, `Vec<Option<String>>` and `Vec<Result<String, _>>` → MIR-0008.
    Verifier errors, not diagnostics — a compiler internal error surfaced to the user. I had tested
    the arms I CHANGED, not the arms that would now RECEIVE references. Fixed by peeling the
    reference (`deref_place`) in every arm that projects into a value; the arms that consume the
    reference itself deliberately do not peel. All four now render three-engine.
  - **DEV-108 [CLOSED by CD-138 — fixed, see there. Original record follows.]:** `Result<Vec<String>, Int32>` fails at `cargo build` of the
    generated crate with `E0502` — the drop-glue slot borrow colliding with the payload borrow held
    across the render. It is the ONE C6.3e shape that fails as a rustc error rather than a named
    pre-rustc refusal, so it is recorded rather than hidden. Deliberately NOT guarded: the
    neighbouring `Option<Vec<String>>` and `Result<Vec<Int32>, Int32>` both render three-engine, so
    any guard broad enough to catch it would refuse working programs, and the precise predicate needs
    a debugging pass on the generated crate. It WAS pinned by `result_of_vec_of_string_fails_at_rustc_dev_108`, which
    was written to fail loudly if the conflict was ever fixed so the case would get promoted — and
    that is exactly what happened. CD-138 replaced it with `result_of_vec_of_string_renders` plus a
    both-variants companion. The "deliberately not guarded" reasoning above turned out to be right
    for the wrong reason: no payload-type guard would have been correct, because the payload was
    never the cause (see CD-138).
  - **DEV-109 [OPEN, CD-138 — `Float32` arithmetic does not maintain binary32 precision].** Both
    interpreters hold a `Float32` in an f64 and round only AT DISPLAY, so the VALUE observes f64
    precision while its RENDERING observes f32. Two consequences, both engine-divergent: `0.1f32 as
    Float64` is a no-op in MIR (yielding `0.1`) while HIR rounds first (yielding
    `0.10000000149011612`); and a `Float32` product that overflows binary32 is stored unrounded
    (`3.4e39`), so it PRINTS as `inf` but is not infinite — `inf - inf` gives `0.0` instead of `NaN`.
    NUM-FLOAT-FORMAT-001 requires IEEE binary32 for value observations, not only for display.
    Surfaced by DEV-105's own evidence while trying to construct a `NaN`.
  - **DEV-110 [ESCALATED, CD-138 — float division by zero: NUM-FLOAT-OP-001 vs CD-006].**
    NUM-FLOAT-OP-001 states that "floating division by zero does not trap: it produces the IEEE
    infinity or NaN result". **CD-006** (owner, 2026-07-18) decided the opposite — keep trapping —
    when the spec text was read as ambiguous. It is no longer ambiguous. HIR follows the spec and
    yields `inf`; MIR follows CD-006 and traps `DivideByZero`; the engines disagree on a program
    both accept. Charter §1.6 rule 1 makes the spec govern, but overriding a recorded OWNER decision
    is not an implementation call, and CD-006 was itself flagged CE2-shaped rather than settled
    unilaterally — so it returns the same way rather than being silently reversed here.
  - **Evidence.** `c63e_formatting.rs` 51 (DEV-108 promoted to a three-engine case plus a
    both-variants companion; the two composite `Float32` refusals deleted — A9 admits those shapes); `c63e_float32.rs` 13; `c63d_map_key_identity.rs` 15;
    `three_engine_differential` +2 message cases and the extended comparator self-test.
  - **ESCALATED, not resolved — `HashMap`/bare-struct `Display` is CE-shaped.** `println(m)` for any
    map is E0500 today (`type_is_displayable` admits only `Option`/`Result`/`Vec`/tuple/array/slice
    plus user-`Display` nominals), as is a struct without a `Display` impl. But the HIR interpreter
    still carries renderings for both (`HashMap{k: v, …}`; `{v: 1}`), and `emit_display_value` has no
    map arm at all — so the day either is admitted to `Display`, it is an instant three-engine
    divergence. Whether a map renders, and in what form, is a language-`Display` semantics decision
    (the same class as CD-123), so it is flagged for the owner rather than settled here.

- CD-135 [2026-07-26, **WP-C6.3e — `Vec` of OWNING elements renders (by reference, not by copy)**]
  `Vec<String>` Display was refused because the Vec renderer read each element with `VecIndexGet`,
  which is BY COPY (V-COPY-1) and so demanded a `Copy` element — copying an owning value the `Vec`
  still holds. CD-131 wired `VecGetRef` natively, which made the fix small.
  - **The element read now splits on Copy-ness.** A `Copy` element is still read by value; an owning
    element is read BY REFERENCE through `VecGetRef` → `Option<&T>`, whose `Some` payload is reached
    by a trailing `VariantField` — borrowable since CD-126. The `None` arm is unreachable (`idx <
    len` holds) but is still emitted as a real discriminant switch rather than assumed away.
  - **The renderer's `Vec` borrow is now reference-aware.** The recursive case made this real: a
    `Vec<Vec<T>>` element arrives as `&Vec<T>`, and borrowing that again built `&&Vec<T>`, which the
    verifier rejected (MIR-0004). `vec_ref_for_display` yields the `&Vec<T>` operand whether the
    place holds the `Vec` or already holds a reference to one.
  - **Evidence.** `c63e_formatting.rs` 47: `Vec<String>` (multi-element and empty) three-engine.
  - **A FINDING, recorded not fixed: the `Vec`-of-`Vec` drop-glue refusal looks over-broad.**
    `Vec<Vec<Int32>>` Display type-checks, lowers and VERIFIES, then the native backend refuses it
    when the printed `Vec` is dropped (Contract C) with the C6.3b-era
    "destructor-in-runtime-collection" deferral. That guard's own comment lists "nested `Vec`/`Box`"
    among the element kinds carrying NO user destructor and therefore expected to pass — but it tests
    `DropPlan::is_noop()`, which is literally `matches!(self, Noop)`, and a `Vec<Int32>` element's
    plan is `VecElements { Int32 }`: non-`Noop` yet running no user destructor anywhere. The precise
    question is "does this plan run any USER destructor, RECURSIVELY", not "is the plan empty".
    Widening a drop-glue refusal is C6.3b's scope, not this formatting slice's, so it is pinned by
    `composite_vec_of_vecs_refused_by_drop_glue` and left for an owner-scoped decision.
  - **C6.3e remaining:** `Float32` (DEV-105 — needs a ruling on where `f32` rounding canonically
    occurs before implementation); trap-message three-engine parity (DEV-106); nested user `Display`
    inside a `Vec`/`Option`/`Result` payload where the payload is itself a non-Copy COMPOSITE.

- CD-134 [2026-07-26, **WP-C6.3d CLOSED by amendment — native `HashMap`; exclusions named**]
  The CE4 representation (CD-132) is implemented natively and the §27 matrix is proven three-engine
  for the admitted domain. Per the owner's closure ruling, C6.3d is closed **only** for that domain,
  with the exclusions stated rather than ticked.
  - **Native representation — the CE4 decision, unchanged.** `stark_runtime::map::StarkMap` is an
    insertion-ordered map with identity by a linear `Eq` scan; `Hash` is never consulted. Held as
    PARALLEL `keys`/`values` vectors rather than a `Vec<(K, V)>` for one concrete reason: STARK types
    the keys cursor as `KeysIter<K>` with no `V` to name, so a cursor over `&[K]` is expressible and
    one over `&[(K, V)]` is not. Ordering, identity and replacement semantics are unaffected.
  - **Identity reaches the backend the same way it reaches MIR.** `emit_bodies::map_key_eq_fn` reads
    the SAME `TypeContext::eq_impls` table the MIR interpreter reads (CD-133) and passes the user's
    selected `Eq::eq` to the runtime as a comparator; a primitive/`String` key gets
    `map::structural_eq`, whose Rust `==` IS its lawful `Eq`. The map never decides identity itself,
    and the backend cannot substitute a Rust trait — generated nominals deliberately derive no `Eq`.
  - **Proven three-engine (HIR == MIR == native), 9 cases in `tests/c63d_map_key_identity.rs`:**
    custom `Eq` decides identity; replacement retains the FIRST stored key; TOTAL hash collision
    keeps unequal keys distinct; custom `Eq` decides `contains_key`; CD-009 insertion order survives
    a custom `Eq`; primitive keys; `String` keys; plus the two boundary tests below.
  - **EXCLUDED — `HashSet` is HIR-only, and that is a LOWERING gap, not a native one.**
    `Core(HashSet, …)` has no MIR representation at all, so implementing it — even as the obvious
    "HashMap to Unit" — would add new MIR semantics, expanding a native-parity WP exactly as the
    C6.3c adapter iterators would have. Same precedent, same ruling. Pinned by
    `hashset_is_hir_only`, which asserts the HIR interpreter RUNS it and lowering REFUSES it.
  - **EXCLUDED — Drop-bearing keys/values remain refused before MIR** ("HashMap over user-Drop key/
    value types (reserved — std-full)"), in BOTH positions, pinned by
    `drop_bearing_keys_and_values_are_refused`. This is what keeps entry Drop order UNOBSERVABLE and
    therefore legitimately unspecified: no user destructor can run inside a map. Admitting them needs
    a Drop-order rule decided AND specified first — not invented here.
  - **§27's remaining matrix rows** (`values`/`entries` iteration, `remove`, HashSet adversarial
    cases) depend on those two exclusions and are out of scope with them.

- CD-133 [2026-07-26, **WP-C6.3d — MIR key identity FIXED: a live HIR↔MIR divergence closed**]
  A correctness fix to shipped code, not a new feature. MIR's `HashMapInsert`/`Get`/`ContainsKey`
  compared keys with `kv[0] == key` — structural `MirValue` equality — so a user `Eq` impl was
  IGNORED. HIR dispatches the user's `Eq` (`language_position` → `language_equal`) and is correct per
  STD-HASH-001, so the two engines disagreed: a key whose `Eq` ignores one field made HIR print `1`
  and MIR print `2` for the same program. It type-checked and ran in both engines; the differential
  never saw it because `HashMap` is absent from the corpus.
  - **Found by an external review, but not as reported.** The review placed the defect in HIR's
    `InsertionMap::position`. That helper exists but is not the path map methods take — HIR was
    right and MIR was wrong, which makes the finding a live divergence rather than merely "unproven
    for adversarial implementations". A probe settled it before any code moved.
  - **Fix — `TypeContext::eq_impls`, no CE3.** The selected `Eq::eq` instance per nominal key type,
    populated during lowering exactly as `drop_impls` has been since C4.5d (`eq_impl_key` mirrors
    `drop_impl_key`; the instance is queued for lowering through `discovered_callees`). The MIR
    interpreter resolves it at the CALL SITE — where the call's operands and the enclosing body's
    local types are still in scope — and calls it. `Eq::eq(&self, other: &K)` needs a place for both
    arguments, so the query key is parked in a scratch frame for the duration of the call. **No
    `RuntimeFn` gains or changes an argument, so the runtime-surface revision does not move.** The
    alternative (new `HashMapFindHash`/`KeyAt`/… ops making every comparison explicit in MIR) is a
    CE3 runtime-surface change and remains available to escalate if MIR-visible comparisons are
    wanted; it is NOT required for correctness.
  - **`HashMapInsert` restructured find-then-mutate**, matching HIR: dispatched `Eq` can run user
    code, so the scan cannot happen inside a `&mut` closure over the entries.
  - **Evidence.** New `tests/c63d_map_key_identity.rs` — 6 cases, HIR == MIR, and the §27 adversarial
    set: custom `Eq` decides identity; replacement retains the FIRST stored key (`b` stays 1 though
    the second insert supplied 2); TOTAL hash collision keeps unequal keys distinct; custom `Eq`
    decides `contains_key`; CD-009 insertion order survives a custom `Eq`; primitive keys unaffected
    (no user impl, structural comparison IS their lawful `Eq`). Regression: `--lib` 441,
    `mir_differential` 132, `three_engine_differential` 86, `exec_snapshots`, `conformance` green.
  - **C6.3d remaining:** the native `StarkMap` slice (the CE4 ordered vector), `HashSet` as
    map-to-Unit, and closure by amendment with the Drop-bearing exclusion named (CD-132).

- CD-132 [2026-07-26, **WP-C6.3d OPENED — the CE4 HashMap/HashSet representation decision (owner)**]
  §27 asks for a CE4 representation decision across nine items. Investigation found **seven of them
  are already normatively fixed**, so the decision put to the owner was much narrower than the
  checklist implies — recorded here so the closed items are not re-litigated:
  - **Already fixed, NOT open.** First-insertion iteration order, replacement preserving position, and
    remove/reinsert appending come from **CD-009** (owner decision, 2026-07-18) and
    `06-Standard-Library`'s "Iteration Order (Core v1)". **STD-HASH-001** additionally fixes: key
    identity by lawful `Eq` with `Hash` used ONLY to select candidate buckets; collisions resolved by
    `Eq` (unequal keys with equal hashes stay distinct); replacement retaining the FIRST stored key
    and its position; observable order independent of hash values, collision strategy, capacity,
    target and process; and a fully specified hash — 64-bit **FNV-1a** (basis `14695981039346656037`,
    prime `1099511628211`) over a canonical byte encoding given in the spec.
  - **Why a host `HashMap` is unacceptable** (§27's warning, made concrete): Rust's `RandomState`
    seeds per process, so iteration order varies between RUNS; it would key on Rust's `Hash`/`Eq`
    rather than STARK's lawful `Eq` (which dispatches to a user impl for user types); and rehashing
    on growth reorders iteration, which STARK requires be capacity-independent.
  - **OWNER DECISION (CE4): mirror the interpreter.** Native `HashMap`/`HashSet` use an
    INSERTION-ORDERED `Vec` of entries with linear scan by STARK `Eq` — structurally what
    `interp.rs`'s `InsertionMap(Vec<(Value, Option<Value>)>)` already is. Rationale: it satisfies
    every fixed contract BY CONSTRUCTION rather than by careful maintenance of a second index, which
    makes divergence from the reference near-impossible; and C6's charge is native semantic PARITY,
    not performance (charter §1.6 rule 7 — correctness precedes optimisation; performance work is
    C7). Lookup is O(n). Because the spec makes observable order independent of storage, switching
    later to an IndexMap-style order-plus-hash-index is an internal change with NO observable
    difference — so this decision does not foreclose the faster representation.
  - **Deliberately NOT decided: entry Drop order.** The spec states no rule, and it is currently
    UNOBSERVABLE — lowering excludes user-`Drop` keys/values, so no user destructor ever runs inside
    a map. Inventing a rule now would be unfounded; it must be decided (and specified) if and when
    droppable keys/values are admitted.
  - **Baseline — CORRECTED (this entry's first draft was wrong).** I recorded "HashMap already runs
    in both interpreters, so this is a native-only gap". It runs in both, but the KEY-IDENTITY
    semantics differ, which an external review flagged and a probe then settled. A key whose `Eq`
    deliberately ignores a field (so `K{1,1}` and `K{1,2}` are the SAME key under STD-HASH-001):
    **HIR prints `1`, MIR prints `2`.** HIR is correct — `HashMap` methods resolve the key through
    `language_position` → `language_equal`, which dispatches the user's `Eq`. MIR is WRONG — its
    `HashMapInsert`/`Get`/`ContainsKey` compare `kv[0] == key`, structural `MirValue` equality, so a
    user `Eq` impl is ignored entirely. (The review attributed the defect to HIR's
    `InsertionMap::position`; that helper exists but is not the path map methods take.) So C6.3d is
    **not** a native-only gap: it is a live HIR↔MIR divergence on a program that type-checks and runs
    in both engines, undetected because `HashMap` is absent from the differential corpus. Two further
    owner decisions were taken on the back of it:
  - **OWNER DECISION (identity): `Eq`-only scan, no cached hash.** Lookups compare with dispatched
    STARK `Eq` and never consult `Hash`. Rationale: hash-narrowing and `Eq`-only scanning are
    OBSERVABLY different when a user's `Hash` is inconsistent with their `Eq` — a TRAIT-LAW-001
    violation where either strategy is conformant alone, but the three engines must agree with each
    other. HIR scans by `Eq` today and is the semantic reference (charter §1.6 rule 6), so all three
    do. A hash index remains addable later, but only ACROSS ALL ENGINES TOGETHER and with the
    law-violating case ruled on. The spec's FNV-1a stays where it already correctly lives —
    `interp::standard_hash`, for direct `Hash::hash` calls — not in map storage.
  - **OWNER DECISION (closure): narrow by amendment.** §27 lists Drop-bearing keys/values among its
    REQUIRED adversarial cases, so C6.3d cannot be ticked complete while they are refused. It will be
    closed only for the admitted non-user-Drop domain, by explicit amendment, with user-`Drop` keys/
    values remaining refused before MIR and entry Drop order recorded as intentionally unspecified
    (it is unobservable while no user destructor can run inside a map). Same precedent as C6.3c.
  - **Implementation route (no CE3).** The selected `Eq` reaches both engines through a new
    `TypeContext::eq_impls` table — per-instance impl symbol, populated during lowering exactly as
    `drop_impls` already is (C4.5d). `RuntimeFn` signatures and arities are UNCHANGED, so the
    runtime-surface revision does not move. The alternative the review proposed — new
    `HashMapFindHash`/`KeyAt`/`ReplaceAt`/`Push`/`RemoveAt` ops making every `Eq` call explicit in
    MIR — is architecturally purer but IS a runtime-surface change (CE3) and a large lowering rewrite;
    it is recorded here as the option to escalate if the owner wants MIR-visible key comparisons.

- CD-131 [2026-07-26, **WP-C6.3b COMPLETED — trapping `Vec` ops, checked interior access, slice
  views; DEV-107 CLOSED**] C6.3b had landed the `Vec`/`Box` VALUE surface and deferred everything that
  either TRAPS on a bad index or hands out an INTERIOR reference. All of it is now native.
  - **DEV-107 closed — and it needed no MIR change.** The deviation was recorded (CD-121) as needing a
    MIR shape change because "the `RuntimeFn` call ABI carries no per-call `SourceInfo`". That was
    WRONG: `MirBlock::terminator` is `(Terminator, SourceInfo)`, so EVERY terminator already carries
    one, `Call` included — it was simply dropped on the way to `emit_call`. It is now threaded through
    as a `CallSite`, and a trapping runtime op bakes in the user's `file:line:col` exactly as
    `Terminator::Checked` does for array/arithmetic traps. The `"<vec index>":0:0` placeholder is gone.
  - **Now native:** `v[i]` (trapping, correct provenance), `v.remove(i)` (trapping), `v.get(i)` /
    `v.get_mut(i)` (CHECKED access that never traps — `Option<&T>`/`Option<&mut T>` through the
    existing `wrap_option` bridge), and SLICE VIEWS: `MirTy::Slice(T)` is Rust's unsized `[T]` (only
    ever named behind a reference), with `SliceNew`/`SliceNewMut`/`SliceLen`/`SliceIsEmpty` wired and
    `Projection::Index` extended to slices in BOTH the type walk and the rendering (only patching the
    latter left the type walk refusing, which the tests caught).
  - **Slice bounds are SIGNED (`i64`), deliberately.** A STARK range is `Int`-typed, so `&a[-1..2]` is
    expressible; taking `u64` would have wrapped a negative bound into a huge index. Bounds are
    widened at the call site and the runtime traps on negative, inverted (`lo > hi`), and past-the-end
    windows — a TRAP, never a clamp (06-Standard-Library).
  - **Evidence.** New `tests/c63b_trapping_ops.rs` — 13 cases. Success paths three-engine
    (HIR == MIR == native stdout); trap paths additionally assert the trap CATEGORY and the exact
    SOURCE LINE on stderr, so a trap firing with the wrong provenance fails rather than passing. Trap
    cases also assert the pre-trap stdout prefix (CD-120 Contract B). Covers: indexed read, OOB index
    (provenance), `get` Some/None, `get_mut`, `remove` + OOB `remove`, array slice view, out-of-range
    /inverted/negative bounds, an empty end window, an INCLUSIVE range, and a slice over a `Vec`.
  - **C6.3b remaining:** `VecReplace` (no method surface reaches it yet), and Vec/Box of
    user-destructor elements (still refused by design — destructor-in-runtime-collection).

- CD-130 [2026-07-26, **WP-C6.3c CLOSED (owner ruling) — native parity, with exclusions named**]
  The owner accepted the native-parity closure basis and ruled that the excluded forms must NOT be
  implemented inside C6.3c, because doing so would expand a backend/runtime parity WP into new
  front-end and MIR semantics. WP-C6.3c is **CLOSED**.
  - **Closed WITH three-engine evidence (HIR == MIR == native):** range iteration, array iteration
    (order), a user `Iterator` impl, shared `Vec` iteration (`v.iter()`), early termination via
    `break`, empty-source iteration, and `String`/`str` character iteration (`chars()` over a literal
    and over an owned `String`). 8 cases in `c63c_iterators.rs`.
  - **EXCLUDED — absent language features, not backend gaps:** slice iteration and mutable (`iter_mut`)
    iteration. Neither has any surface in the compiler or the spec.
  - **EXCLUDED — pre-MIR capability gaps:** `map`/`filter`, `count`/`collect`, and by-value `Vec`
    iteration. Neither MIR nor native can represent them; they run only in the HIR interpreter, so
    there is no native divergence for this gate to close.
  - **Follow-on recorded, NOT scheduled:** `starkc/docs/WP-ITER-LOWERING-PROPOSAL.md` — MIR
    representations for adapter iterators; method resolution/lowering for iterator values with
    non-nominal types; by-value collection iteration; remaining-element `Drop` on normal completion,
    `break`, trap and early return; slice iteration ONLY if the language surface is explicitly
    approved; mutable iteration ONLY through a separate language/spec decision. It requires owner
    approval and a roadmap slot before any implementation (charter §1.6 rule 4).
  - **The four boundary tests are PERMANENT regression evidence** (owner instruction). Each HIR-only
    test asserts both that the HIR interpreter RUNS the program and that lowering REFUSES it, which is
    what distinguishes "supported by HIR but not lowerable" from a native divergence and stops the
    boundary changing silently — if any starts lowering, its test fails and the case must be promoted
    to three-engine.
  - **Next:** the remaining EXISTING C6.3 packages (trapping Vec ops, HashMap/HashSet C6.3d, files
    C6.3f, C6.3 closure evidence) — the iterator-expansion work is not imported into this gate.

- CD-129 [2026-07-26, **WP-C6.3c CLOSED for native parity — the §26 boundary is now executable**]
  Every §26 row that MIR can lower is native and proven three-engine (CD-128). This entry establishes
  what remains and why none of it is a NATIVE gap, replacing prose with negative tests.
  - **Rows the language does not have.** `for x in <slice>` is rejected by the front end ("for-loop
    requires an iterable value, found `&[Int32]`"), and there is no `iter_mut` surface ANYWHERE in
    the compiler or spec — "Vec mutable iteration" is not deferred work, it is an absent feature.
  - **Rows that are HIR-ONLY (a C4.5-era LOWERING gap).** `map`/`filter` have no MIR type for
    `Core(MapIter/FilterIter, …)`; `count`/`collect` are method calls on a non-nominal (core)
    receiver, which lowering does not do; by-value `for x in v` is refused ("for over a non-range,
    non-Vec iterator"). Each RUNS in the HIR interpreter and stops at lowering — which is precisely
    what makes them lowering gaps, not backend ones: **the MIR interpreter cannot run them either, so
    there is no native/interpreter divergence for C6 to close, and the differential suite cannot even
    reach them.** Closing them is a front-end/MIR package; under the charter it needs its own scope,
    not an extension of a native-parity WP.
  - **Evidence.** `c63c_iterators.rs` is now 12: the 8 three-engine cases plus 4 boundary tests —
    `slice_iteration_is_not_a_language_form` (front-end rejection),
    `vec_by_value_iteration_is_hir_only`, `map_adapter_is_hir_only`, `count_and_collect_are_hir_only`
    (each asserting the HIR interpreter RUNS it and lowering REFUSES it). The boundary can no longer
    drift unnoticed, and a future lowering package inherits its starting point.
  - **Open (not C6.3c):** `HashMap`/`HashSet` iteration → C6.3d; the lowering gaps above → a
    front-end/MIR package.

- CD-128 [2026-07-25, **WP-C6.3c OPENED — native iterators; the Move borrow-carrier refusal RETIRED**]
  §26's matrix splits into two lowering families, and only one needed backend work — established
  empirically by building the matrix as a probe suite BEFORE writing any code:
  - **Counting loops — already native.** `for i in a..b` and `for x in <array>` lower to an index
    loop under the ordinary `CheckIndex` proof discipline (no iterator object exists at runtime), and
    a user `Iterator` impl is ordinary static calls to the user's `next`. All three passed on the
    first probe run.
  - **Runtime iterator CURSORS — added here.** `v.iter()` and `s.chars()` lower to
    `*IterNew`/`*IterNext` over a live cursor that BORROWS its source. `stark_runtime` gains
    `vec::VecIter<'a, T>` (slice + index) and `string::CharsIter<'a>` (over `std::str::Chars`), with
    `iter_next` lending `&'a T` out of the SOURCE rather than out of the `&mut` cursor borrow — which
    is what lets the loop variable outlive the `next` call, as the `for` desugaring requires.
    `emit_types` spells the cursors (they carry a lifetime in EVERY position — unlike `Vec<Int32>`,
    a cursor borrows even when its type arguments do not, so `nominal_needs_lifetime` reports true
    for them directly), and `emit_runtime` wires the four ops, `Next` through the existing
    `wrap_option` bridge.
  - **A CD-127 DIVIDEND: `refuse_borrow_carrying_nominals` is DELETED.** Native iteration first hit
    that C6.1f-era refusal — a slot-backed (Move) borrow-carrying nominal, refused because the
    `ValueSlot`'s destruction needs `&mut` while the reference it stores is still live (E0502). That
    is exactly the imprecision CD-127 removed. Verified rather than assumed: with the check bypassed,
    the iterator cases built AND the refusal's own hardest negative case — a `Drop`-bearing
    `H<&P>` — built and ran (exit 0). The check is gone, and with it the LAST lane negative: every
    shape `native_c5_3_aggregates_enums`'s lane test once pinned as "must be refused before rustc" is
    now supported, so that test is removed (following its own instruction to move supported shapes to
    positive tests) and `native_c61f_nominals`'s refusal case became
    `c61f_a_move_borrow_carrying_nominal_local_now_works`.
  - **Evidence.** New `tests/c63c_iterators.rs` — 8 three-engine cases: range, array order, user
    `Iterator` impl, `v.iter()` sum+order, early `break` mid-iteration, empty source, `chars()` over
    a literal and over an owned `String`. Order and early termination are asserted INSIDE the STARK
    programs and by printed output, so agreeing on the wrong order still fails.
    `native_c61f_nominals` 8, `native_c5_3_aggregates_enums` 20, `c61f_structural_copy` 11 green.
  - **C6.3c remaining:** `HashMap` keys/values/entries and `HashSet` (land with C6.3d), slice
    iteration, `map`/`filter`/`collect`, and by-value/mutable `Vec` iteration (no `iter_mut` surface
    exists in the language yet — confirm against the spec before adding one).

- CD-127 [2026-07-25, **backend — STRUCTURED control-flow emission; borrow precision inside loops
  (generalises CD-112)**] Every generated body with a loop was emitted as `loop { match __bb { … } }`,
  which switches on a RUNTIME value — so rustc must assume ANY block can follow ANY block. Every local
  read anywhere in the loop is therefore live everywhere in it, and a borrow held across a block
  boundary conflicts with every mutable use of its referent. Loops had **zero** borrow precision.
  CD-112 fixed this for ACYCLIC bodies (nested labelled blocks); cyclic bodies kept the dispatch loop.
  - **Diagnosis (empirical, not inferred).** The generated crate for a `Vec<P>` Display render was
    dumped and hand-patched: moving the borrow and its use into ONE block made the identical program
    compile. That isolates the cause to the dispatch loop's lost edge information, and rules out the
    borrow itself being ill-formed.
  - **Fix — `structured_plan` + `EmitMode::Structured`.** A body is now emitted as REAL Rust control
    flow: a **forward** edge to `t` is `break 'bbT`, where `'bbT` is a labelled block opened at `t`'s
    EARLIEST forward predecessor and closed immediately before `t`; a **back** edge to header `h` is
    `continue 'loopH`, where `'loopH: loop` spans `h` through its whole NATURAL LOOP. Scopes are
    opened widest-first per index (on a tie the `Block` is outer, so a loop-EXIT edge escapes the
    loop it shares a span with) and validated against a stack; a CFG whose scopes would partially
    overlap (irreducible) is not emitted this way at all but falls back to the dispatch loop, which
    remains for exactly that case. `linear_order` is superseded (kept only under `#[cfg(test)]`).
  - **Two defects found DURING this work — by a test that hung, not by review.** Both were in the
    first cut of the scope computation, and both are recorded because each is a trap the next
    control-flow change could fall into:
    1. **A loop's span must cover its whole natural loop, not just its latches.** RPO can place an
       INNER loop's latch AFTER the outer loop's, so an outer span measured by latches did not
       contain the inner loop and the two spans CROSSED. Now computed as the natural loop of each
       back edge (`h` plus every node reaching the latch without passing through `h`).
    2. **A `Loop` scope must never be widened.** The crossing above was "repaired" by an
       outward-extension rule that moved the inner loop's start earlier — off its header — which
       pulls the preceding blocks into the loop body and re-executes them every iteration. In
       nested-`while` code that reset the inner counter forever: an INFINITE LOOP in a previously
       passing test (`multi_iteration_loop_agrees` spun at 76% CPU for ten minutes). Extension is
       now restricted to `Block` labels; a genuinely crossing `Loop` is irreducible and falls back.
  - **Coverage gap this exposed:** the differential suite had NO `loop { … }` case at all — only
    `while`. Three were added (`infinite_loop_with_mid_body_break_agrees`,
    `loop_with_continue_and_break_agrees`, `nested_loop_scopes_agree`), covering a mid-body `break`
    as a loop's only exit, `continue`+`break` from inside a body, and nested loop scopes — precisely
    the shapes that stress scope nesting.
  - **Retires the loop-borrow deferral:** nested user `Display` inside a `Vec` — whose per-iteration
    `fmt` `String` is borrowed then dropped — now compiles and renders three-engine. More importantly
    this was a GENERAL limitation: every cross-block borrow inside every loop was blocked, which the
    iterator (C6.3c), `chars()` and HashMap (C6.3d) work would have hit constantly.
  - **Evidence.** `c63e_formatting.rs` 44 with `nested_user_display_in_vec` now POSITIVE three-engine;
    `three_engine_differential` **86** (83 + the three new `loop` cases) green; full suite + CI as the
    exhaustive check on a change that touches EVERY generated body.
  - **Still deferred (unrelated causes):** `Vec<String>` Display — the Vec arm reads elements by COPY
    (`VecIndexGet`, V-COPY-1), so a non-Copy element needs by-REFERENCE Vec access, not borrow
    precision; and a droppable composite carrying a borrow (generated lifetimes).

- CD-126 [2026-07-25, **WP-C6.3e / backend — enum-payload BORROW fixed (retires two deferrals)**]
  The native backend could not borrow an enum variant payload: `emit_places` emitted every
  `VariantField` projection as `match &e { V(p) => *p }` — a dereferenced VALUE. Reading by value
  needs a Copy payload (so non-Copy payloads were refused), and `Rvalue::RefOf` wrapped it as
  `&(match … *p)`, which borrows a temporary freed at statement end (rustc E0716). That blocked
  `Option`/`Result` of a `String` or a user-`Display` nominal.
  - **Fix (two edits, isolated to the shared `Callee::Runtime`/`RefOf` path):** in BORROW mode a
    TRAILING variant-field now emits `match &e { V(p) => p }` — the `&Payload` directly, valid for as
    long as `e` lives — and `RefOf` recognises that a trailing-variant-field place already yields a
    reference, so it does not re-wrap it in `&`. Borrowing needs no move, so it works for ANY payload
    type; the READ-by-value path is unchanged (`*p`, still Copy-required).
  - **Retires two deferrals:** `Option<String>`/`Result<String>` (CD-122's non-Copy refusal) AND
    nested user `Display` inside `Option`/`Result` (CD-123's E0716 refusal). Both lowering gates in
    `emit_display_value` are removed; a DEEPER non-Copy payload (a tuple owning a `String` inside an
    `Option`) still needs a non-trailing variant-field value read and gets a clean backend refusal —
    no lowering gate needed.
  - **Evidence.** `c63e_formatting.rs` 44 — `composite_option_of_string`, `composite_result_of_string`,
    `nested_user_display_in_option`, `nested_user_display_in_result` now POSITIVE three-engine.
    Regression (a cross-cutting codegen change): `--lib` 441, `three_engine_differential` 83,
    `mir_differential` 132, `native_c5_3_aggregates_enums` 21, `native_c61f_b3_stored_refs` 6,
    `native_c61f_reborrow` 5 — all green; fmt + clippy clean.
  - **Still deferred:** nested user `Display` / owner elements inside a `Vec` — that is the SEPARATE
    E0502 loop-carried-borrow limitation, not the enum-payload one.

- CD-125 [2026-07-25, **WP-C6.3e — composite `Box` elements DEFERRED (owner decision)**] Investigating
  the last item on the C6.3e "remaining" list found it is not a lowering slice: `Box<T>` is not a
  Display type at all. `typecheck::type_is_displayable` admits only `Option`/`Result`/`Vec` among Core
  types, so `Box` falls to `_ => false` and `println(box)` / `println((box, 1))` are rejected E0500;
  the spec (`06-Standard-Library`) says nothing about `Box` + `Display`. (The interpreter's
  `Display for Value` incidentally renders `Box(inner)`, but that path is unreachable — the
  typechecker blocks it — so it is dead code, not a de-facto contract.)
  - Making `Box` displayable is a SEMANTICS decision (charter §1.6 rule 4), not a mechanical
    continuation: it needs the displayable-set extended AND a render-form choice — transparent
    (`inner`, the Rust idiom, which would change the interp's `Box(...)` rendering) vs wrapped
    (`Box(inner)`, matching the interp's dead code).
  - **Owner decision: DEFER** — revisit as a future language-Display-semantics decision, not now.
    `Box` remains an opaque owning box in Core v1 (no `Deref`, `into_inner` only); today you
    `into_inner()` and print the value. No code change; the C6.3e "remaining" list drops `Box`
    elements as active scope and records it as deferred here.

- CD-124 [2026-07-25, **CI hotfix — CD-119's `Float32` refusal was too broad (broke the frozen
  corpus)**] CD-119 moved the `Float32` Display refusal into `widen_for_print`, the SHARED chokepoint
  for BOTH scalar `println(Float32)` and composite elements. That refused a scalar top-level
  `println(Float32)` at lowering — which the interpreter-only frozen corpus
  (`mir_differential::entire_frozen_corpus_agrees`, snapshot
  `primitive__03_float_arithmetic_and_casts.stark`) depends on and the HIR/MIR engines AGREE on (the
  f32→f64 divergence is native-only). CI went red at CD-119 and stayed red through CD-122; local
  scoped runs never included `mir_differential`, so it went unseen until flagged.
  - **Fix:** `widen_for_print` widens `Float32`→`Float64` again (scalar admitted, DEV-105); the
    refusal moved to `emit_display_value`'s primitive arm — the COMPOSITE path only, where a `Float32`
    element would otherwise reach the native binary silently (review #2's actual concern). Scalar
    native `println(Float32)` remains an admitted DEV-105 divergence (untested, as before CD-119).
  - **Test correction:** `c63e_formatting.rs` `float32_println_refused` (scalar) removed; the
    composite negatives (`float32_in_tuple_refused`, `float32_in_option_refused`) stay. CD-119's entry
    below overstates the refusal scope ("every Display path"); this entry is the correction.
  - **Process:** `mir_differential` (and the full `cargo test`) must run before a WP/gate closes —
    scoped runs miss the frozen corpus. Recorded in memory [[stark-test-run-frequency]].
  - **Evidence:** `mir_differential` 132 (was 131 + 1 failed), `c63e_formatting` 43; combined with the
    CD-123 change below and verified green together.

- CD-123 [2026-07-25, **WP-C6.3e — nested user `Display` in a composite (+ reference-oracle fix)**]
  **Owner decision (asked & answered):** language-level `Display` recurses — a user nominal at ANY
  depth runs its OWN `Display::fmt`, NOT the aggregate `{field: value}` debug form. This resolves a
  reference-implementation INCONSISTENCY: top-level `println(p)` already called `fmt` (→ `CUSTOM`),
  but `println((p, 1))` fell through to the generic `Display for Value` and rendered `({v: 7}, 1)`.
  - **Native lowering:** `emit_display_value` gains a user-nominal arm — it calls the element's
    `fmt(&self)` on the element BORROWED IN PLACE (the owning composite keeps and later drops it —
    Contract C), prints the returned `String` (no newline — an element), then drops that `String`.
    Same machinery as top-level `lower_print_display`, minus the arg-drop.
  - **Interp (oracle) fix:** `display_text` now routes a composite argument through a new recursive
    `display_deep`, which calls user `fmt` for nested nominals and renders composites with the SAME
    delimiters the lowering emits. A nested nominal is CLONED to give `fmt` a `&self` place (a Rust
    clone runs no STARK destructor) and the clone is discarded WITHOUT `drop_value` — so the real
    element is dropped exactly once by its owning composite (no double destructor). The composite is
    promoted to a place and dropped once by `finish_display` — also fixing a latent gap (droppable
    composite `println` args were not being dropped). Nominal-free composites render byte-identically
    to before (same delimiters), so no existing output changed (`--lib` 441 green).
  - **Works three-engine:** nested user `Display` in a tuple/array, INCLUDING a Drop-bearing nested
    nominal — `println((d, 1))` renders `(DROPPY, 1)` via the element's `fmt` with NO double
    destructor, then the tuple drops it once (`DROP`), proving the clone-discard discipline.
  - **Deferred — refused AT LOWERING (via `ty_mentions_user_nominal`):** nested user `Display` inside
    a `Vec` (the per-iteration `fmt` `String` borrow is loop-carried, rustc E0502) and inside
    `Option`/`Result` (the `VariantField`-payload borrow is a temporary freed too early, E0716).
  - **Evidence.** `c63e_formatting.rs` now 44: +3 positive (`nest_tuple`, `nest_array`, `nest_drop`
    — the Drop-bearing Contract C proof) and +2 `refused_by_lowering` (`nest_vec`, `nest_option`).
    `--lib` 441, `three_engine_differential` 83, `c63b_vec_box` 9 green.
  - **C6.3e remaining:** composite `Box` elements; nested user `Display` inside `Vec` (loop-borrow) /
    `Option`/`Result` (enum-payload borrow); `Vec<String>`/`Option<String>` (same backend gaps);
    `Float32` (DEV-105); trap-message parity (DEV-106); native `v[i]` OOB provenance (DEV-107).

- CD-122 [2026-07-25, **WP-C6.3e — String/str as composite elements (+ two bounded deferrals)**]
  `emit_display_value` now renders a `String`/`str` ELEMENT of a composite: its raw bytes (NO quotes —
  `Display for Value`, interp.rs line 501), via `&String -> as_str -> PrintStr`. The element is
  BORROWED in place, never moved out of the composite temp, so the whole composite is still dropped
  after the render (CD-120 Contract C). `lower_print_composite`'s gate is broadened — ANY droppable
  composite is admitted and `emit_display_value` is the real filter (it cleanly refuses what it
  cannot render).
  - **Works, three-engine:** owned `String` in a tuple/array (`(String::from("hi"), 1)`,
    `[String; 2]`) and `&str` in a Copy composite (`("hi", 1)`).
  - **Two deferrals — refused AT LOWERING (deterministic), not admitted-but-broken:**
    (1) a non-Copy payload inside `Option`/`Result` (`Option<String>`) — borrowing a non-Copy enum
    `VariantField` payload needs WP-C5.3d controlled storage (native `match &e` yields a reference
    and moving out hits C5.3a's cross-block-move limit); refused in the `Option`/`Result` arms.
    (2) a droppable composite that ALSO carries a borrow (`(String, &str, i32)`) — its slot-backed
    field read returns a borrow whose lifetime the backend does not emit (rustc E0106); refused via a
    new `ty_carries_ref` gate. A COPY borrow-carrier (`(&str, i32)`) is fine (no slot, no wrapper).
    `Vec<String>` also stays refused (the Vec arm needs a Copy element; by-reference Vec access is a
    separate slice).
  - **Evidence.** `c63e_formatting.rs` now 39: +3 positive (`tuple_str`, `tuple_string`, `arr_string`,
    three-engine) and +3 `refused_by_lowering` negatives (`option_of_string`, `result_of_string`,
    `droppable_tuple_carrying_borrow`). `--lib`, `three_engine_differential` 83 green.
  - **C6.3e remaining:** composite `Box` elements, nested user-`Display`; `Option`/`Result`-of-owner
    (WP-C5.3d), borrow-in-droppable-composite (generated lifetimes), `Vec<String>` (by-ref access);
    `Float32` (DEV-105); trap-message parity (DEV-106); native `v[i]` OOB provenance (DEV-107).

- CD-121 [2026-07-25, **WP-C6.3e — Vec Display (runtime loop; first non-Copy composite)**] `println`/
  `print` of a `Vec<T>` (T a Copy primitive or Copy composite) now renders `[e0, e1, …]` — the FIRST
  composite that needs a runtime LOOP rather than unrolling, and the FIRST non-Copy composite, so it
  activates CD-120 Contract C for real. Built directly against A/B/C. Native AND MIR.
  - **The loop (Contract A).** `emit_display_value`'s Vec arm reads `VecLen`, then loops
    `idx` in `0..len`, emitting `", "` before every element but the first and `VecIndexGet(&v, idx)`
    (by Copy, V-COPY-1) into a temp it renders recursively — the same per-element print-op sequence,
    in index order, as the interpreter's `Display for Value` (`[`/`, `/`]`). Empty → `[]`.
  - **Contract C (destructor timing) — now load-bearing.** The owned Vec is MOVED into the print
    temp and DROPPED after the whole render (including the trailing newline). This matches the
    interpreter, which also consumes+drops the by-value print argument; so a single `println(v)`
    agrees across engines, and `println(v); println(v)` is correctly rejected in BOTH (E0100
    use-after-move — `print`/`println` are `fn(T)`, and a non-Copy `T` moves). `println(&v)` never
    arises (E0500).
  - **Fresh-borrow discipline (the E0502 fix).** A single reused `&Vec` held across the loop is still
    live at the post-render mutable `drop_with`, which rustc rejects. Each runtime read (the length,
    and every element) now takes a FRESH short shared borrow that dies at its call, so the Vec's own
    drop is unobstructed.
  - **Native `VecIndexGet` wired** (`stark_runtime::vec::index_get`, by Copy). **DEV-107 [recorded]:**
    its out-of-bounds trap reports a runtime-internal location (`<vec index>`), not the user's `v[i]`
    span — the `RuntimeFn` call ABI carries no per-call `SourceInfo` (only `Terminator::Checked`/
    `Trap` do), so precise provenance awaits the native Vec-trapping-ops WP. Category and exit code
    (101) are correct. The Display loop guarantees `idx < len`, so this path is DEAD for Display; the
    deviation concerns only general native `v[i]` OOB, not yet differential-tested.
  - **Evidence.** `c63e_formatting.rs` now 33 (+6: `Vec<Int32>` multi/empty/singleton, `Vec<Bool>`,
    two-Vec print-then-println, `Vec<(Int32, Bool)>` recursing into tuple elements), each three-engine
    HIR == MIR == native stdout.
  - **C6.3e remaining:** composite `str`/`String`/`Box` elements, nested user-`Display`; `Float32`
    (DEV-105); trap-message parity (DEV-106); native `v[i]` OOB provenance (DEV-107).

- CD-120 [2026-07-25, **WP-C6.3e — composite Display observable-behaviour contracts + a trap-flush
  fix they surfaced**] Before `Vec` Display (the first composite needing a runtime LOOP rather than
  unrolling), the three observable-behaviour contracts it must satisfy are written down explicitly,
  and writing them surfaced+fixed a real native/interp divergence. No new MIR shape or `RuntimeFn`.
  - **Contract A — output sequencing.** Composite Display is a *print-sequence lowering*: a fixed
    left-to-right structural walk emitting `Print*`/`PrintStr` ops in structural order — opening
    delimiter, each element rendered recursively separated by `", "`, closing delimiter, and a
    trailing newline (`PrintlnStr("")`) for `println`. There is NO intermediate `String` assembly
    and no reordering buffer, so the byte stream is defined purely by op emission order on the one
    shared stdout. All three engines run the SAME ordered ops → byte-identical by construction
    (proven: `c63e_formatting` HIR==MIR==native). A runtime-length container (Vec) emits the same
    per-element op sequence per iteration in index order, separator before every element but the
    first.
  - **Contract B — partial output on trap.** If rendering an element traps (Vec index OOB,
    arithmetic/cast/`panic` in a nested user `fmt`), STARK trap semantics are unchanged: ABORT (exit
    101), NO unwind, NO destructors. The observable stdout is therefore exactly the prefix of ops
    completed before the trapping op — every opening delimiter / separator / fully-rendered earlier
    element, and NOTHING after (no closing delimiter, no newline). That prefix is byte-identical
    across engines: the interpreters already retain it (`interp::run_with_partial_output`, used by
    the MIR differential comparator), and native now does too (see the fix below).
  - **Contract C — destructor timing.** A Drop-bearing (non-Copy) printed value's destructor runs
    AFTER the complete rendering (including the trailing newline) is emitted — never interleaved with
    the bytes — on the success path (the scalar rule of CD-114, proven by
    `user_display_drop_bearing_runs_destructor_after_output`). A composite drops its owned elements
    as part of that single post-render destruction, in the language's declared drop order. On the
    trap path (Contract B) NO destructor runs. Composite Display is Copy-only today (nothing to
    drop), so this is currently vacuous for composites; it governs the `String`/`Vec`/`Box` element
    slices and is why they are sequenced after this CD.
  - **The fix Contract B surfaced (real bug).** `std::io::stdout()` is a `LineWriter`; `print(x)`
    with no trailing newline sits unflushed, and `std::process::exit(101)` in the trap ABI does NOT
    flush it — so a trap mid-output DROPPED its pre-trap prefix natively while the interpreters kept
    it, violating Contract B across engines. `stark_runtime::output::flush_stdout()` was added and is
    now called at the top of both `trap::abort` and `trap::abort_with_message`.
  - **Evidence.** `native_c5_2e_traps.rs` +1 (`output_before_trap_is_flushed_then_abort`:
    `print("before")` then an overflow trap ⇒ stdout is exactly `"before"`, exit 101 — 7 pass).
  - **DEV-106 [narrowed]:** partial *output* IS already cross-engine comparable (above); the residual
    gap is only trap *message/category* TEXT equality across engines — `interp::Outcome`'s trap arm
    carries no category/message field for the comparator to assert. That remains the deferred,
    CE-adjacent `Outcome::Trapped { message }` widening.
  - **C6.3e remaining:** composite `str`/`String`/`Box`/`Vec` (loop, built against A/B/C) elements,
    nested user-`Display`; `Float32` (DEV-105); trap-message parity (DEV-106).

- CD-119 [2026-07-25, **WP-C6.3e — composite formatting boundary hardening (external review)**] A
  bounded correctness pass on the composite Display foundation (CD-117/118) before extending it to
  `Vec`, closing two soundness/scalability gaps the reviewer flagged and one differential-coverage
  gap (recorded, not closed).
  - **DEV-105 no longer leaks into composites (the real fix).** `widen_for_print` was the single
    Float32 chokepoint for scalar printing, but `emit_display_value` recursed into composite elements
    THROUGH it — so `println((1, 0.1f32))` reached native with the very f32→f64 widening divergence
    DEV-105 defers, silently, inside a tuple. `widen_for_print`'s `Float32` arm now returns
    `unsupported(… DEV-105 …)`, refusing Float32 in EVERY Display path (scalar and every composite
    depth) BEFORE MIR — a refusal, never a wrong answer. Confirmed no existing test prints Float32
    (c63e/native_c5_2b/native_c5_2c/gate2 all green after the change).
  - **Array unrolling is bounded.** `emit_display_value`'s `Array` arm fully unrolls elements (one
    print-op sequence per index); it now caps at `MAX_UNROLL = 64` and `unsupported`s longer arrays
    rather than emitting an unbounded body. (A runtime loop is the eventual lift, tracked with `Vec`.)
  - **Evidence.** `c63e_formatting.rs` now 27: added boundary positives `Some(None)`, `Some(Ok(5))`,
    `[Some(1), None]`; and a `refused_by_lowering` helper with negatives `float32_println_refused`,
    `float32_in_tuple_refused`, `float32_in_option_refused`, `large_array_display_refused` (each
    asserting the lowering refuses, not that native mis-renders). Header rewritten to state the
    native/refused boundary. `--lib`, `three_engine_differential`, `gate2_valid` green; fmt + clippy
    clean.
  - **DEV-106 [recorded, deferred — CE-adjacent]:** the three-engine differential compares that all
    engines TRAP, not the trap MESSAGE. Native already proves category+location+user-message on
    stderr (`native_c5_2e_traps.rs`), and HIR/MIR carry their own messages, but `interp::Outcome::
    Trapped` has no `message` field, so the comparator cannot assert byte-equal trap text across
    engines. Closing it means widening `Outcome::Trapped { message: Option<String> }` and threading it
    through both interpreters — an interp-surface change I am flagging rather than folding into a
    formatting pass, so it can be scoped deliberately.
  - **C6.3e remaining:** composite `str`/`String`/`Box`/`Vec` (loop) elements, nested user-`Display`;
    `Float32` (DEV-105); assert-message + trap-message three-engine parity (DEV-106).

- CD-118 [2026-07-25, **WP-C6.3e slice 5 — native composite Display: Option/Result**] Extends the
  composite renderer to `Option`/`Result` (Copy payloads): `emit_display_value` reads the
  discriminant (`Rvalue::Discriminant`) and `SwitchInt`s to a `None`/`Some(v)` or `Ok(v)`/`Err(e)`
  branch, recursing into the payload via a `VariantField` projection. Still no runtime-surface change
  and still three-engine (the recursion also renders a nested composite inside the payload).
  - **Proven three-engine:** `Some(5)`, `None`, `Ok(7)`, `Err(true)`, and nested `Some((1, 2))`
    (composite inside the Some payload). `c63e_formatting.rs` now 20. Regression: `--lib` 441,
    `three_engine_differential` 83; fmt + clippy clean.
  - **C6.3e remaining:** composite `str`/`String` elements, `Box`, `Vec` (a runtime loop), nested
    user-`Display`; `Float32` (DEV-105); assert message text.

- CD-117 [2026-07-25, **WP-C6.3e slice 4 — native composite Display (tuple/array)**] `println`/`print`
  of a displayable COMPOSITE was HIR-only — the lowering (`widen_for_print`) rejected it before MIR,
  so neither MIR nor native rendered it. Now a tuple/array of primitive elements lowers to a SEQUENCE
  of primitive print ops matching the interpreter's `Display for Value` — `print("(")`,
  `print(elem0)`, `print(", ")`, …, `print(")")`, trailing newline for `println`. **No runtime-surface
  change** (0.1-A8 untouched): it reuses the `Print*` ops from slice 1, so no value→String `RuntimeFn`
  and no CE3 contract bump. This ALSO adds MIR support, not just native.
  - `lower_print_composite` + a recursive `emit_display_value` (primitives + `Tuple` + `Array`);
    restricted to `Copy` composites (nothing to drop) in this slice.
  - **Proven three-engine (HIR == MIR == native stdout):** `(1, 2)`, mixed `(1, true, 2.5)`,
    `[10, 20, 30]`, nested `((1, 2), 3)`, `[(1, 2), (3, 4)]`, print-then-println. `c63e_formatting.rs`
    now 17. Regression: `--lib` 441, `three_engine_differential` 83, `mir_lowering`; fmt + clippy
    clean.
  - **C6.3e remaining:** composite `str`/`String` elements, `Option`/`Result`/`Box`, `Vec` (a runtime
    loop), nested user-`Display`; `Float32` (DEV-105); assert message text.

- CD-116 [2026-07-25, **evidence precision + state sync (external review)**] A bounded correction
  pass on CD-113/114 before composite formatting — no implementation change, tightening tests and
  resynchronising governance docs.
  - **c63e evidence strengthened.** (a) `agree_out` now also asserts `mir_exec.output == expect` — the
    MIR oracle's STDOUT, not just its exit status, so each case is self-contained three-engine
    evidence. (b) `user_display_reads_field` now BRANCHES on `self.v` (`if self.v == 3 …`) so the
    output actually depends on the field (the prior body ignored `self`). (c) the Drop-bearing case
    now has an OBSERVABLE destructor (`fn drop { println("DROP"); }`) and the expected output
    `DROPPY\nDROP` proves the destructor runs exactly once, after the formatted bytes — the earlier
    empty destructor proved neither timing nor count. c63e 11 still green.
  - **DEV-105 recorded** for the `Float32`-println cross-engine cast-precision discrepancy (was noted
    without an id); the c63e header corrected from "Float32/Float64" to Float64-only.
  - **State docs resynchronised.** `COMPILER-STATE.md` header date → 2026-07-25; the C6.3a/b summaries
    no longer say owned-`String` `==`/`<`, stored interior `&str`, and `Vec<String>`-style pushes are
    "deferred to C6.1g-c" (they were promoted to native, CD-116) — they contradicted the CD-112
    closure line. `WP-C6-ENTRY` §24/§25 String/Vec rows updated to match. A C6.3e header summary added.
  - **Recorded as a C6.3 CLOSURE requirement (not yet done):** runtime version review + installed-
    layout + offline-build proofs for the CD-113 `stark_runtime::format` addition (generated-code
    tests exist; the install/offline evidence does not). Must land before C6.3 closes.
  - (`starkide` non-interactive tests were removed with the module per owner instruction; extracting
    the pure editor logic into a testable lib module is a possible future cleanup, out of scope here.)

- CD-115 [2026-07-25, **WP-C6.3e slice 3 — native `panic(msg)` text**] A `Terminator::Trap` carrying
  a `&str` message (an explicit `panic("...")`) was `Unsupported` natively; now that str values are
  native it is wired. Added `stark_runtime::trap::abort_with_message(category, message, file, line,
  col)` which reports the category header and `-->` location in the SAME shape as `abort` (so the
  three-engine stderr parser still reads category + provenance) and the user message on its own line;
  `emit_bodies` emits it with the resolved `&str` operand. Message-less traps (every
  compiler-generated trap and `assert*`, which lower with `message: None`) are unchanged.
  - **Proven:** `tests/native_c5_2e_traps.rs` now 6 — `panic("the sky is falling")` and a
    conditional `panic("too big")` each abort with exit 101, the `explicit panic` category, the exact
    `file:line`, and the user message in stderr. Regression: `--lib` 441, `three_engine_differential`
    83 (message-less traps unaffected); fmt + clippy clean.

- CD-114 [2026-07-25, **WP-C6.3e slice 2 — native user `Display` dispatch; C6.2d Display deferral
  CLEARED**] `println(x)` on a user struct/enum with a `Display` impl now runs the user's `fmt` and
  prints its `String` result natively — never Rust's `Debug`. This was already wired (`lower_print_
  display` → call `Display::fmt(&self) -> String`, then `PrintlnStr`); the pieces became native once
  C6.1g-c unblocked String-returning methods and C6.3a wired `PrintlnStr`.
  - **The one fix:** `lower_print_display` unconditionally dropped the by-value argument, but a `Copy`
    printed type (`struct P { v: Int32 }` with a `&self` `Display`) has no destructor — the emitted
    `Drop` on a `Copy` type is a no-op the interpreter ignores but the native backend refuses (Copy
    has no slot). Now the arg-drop is gated on `!is_copy`. A Drop-bearing (non-`Copy`) printed value
    still has its destructor run after the bytes are submitted (observable, oracle-matched).
  - **Proven native (stdout == HIR oracle):** `tests/c63e_formatting.rs` now 11 — the 7 primitive
    cases plus user `Display` on a Copy struct, a field-reading `fmt`, a Drop-bearing type, and an
    enum. Regression: `--lib` 441, `three_engine_differential` 83, `native_c6_2_generics_traits`,
    `gate2_valid`, `mir_lowering` — all green; fmt + clippy clean.
  - **C6.2d Display:** the deferral (native output → C6.3) is now satisfied for user `Display`.
  - **C6.3e remaining:** composite `Display` (tuple/struct/enum/Option/Result/Vec/Box field-by-field
    rendering), `Float32` println (the deferred cast-precision differential), panic/assert text bytes.

- CD-113 [2026-07-25, **WP-C6.3e slice 1 — native primitive formatting + output**] `println`/`print`
  of `Int*`/`UInt*` (widened to `i64`/`u64`), `Bool`, and `Float64` now emit natively, rendered per
  STARK's canonical form (not Rust `Debug`). Until now native supported ONLY str/char output; numbers
  and bools could not be printed.
  - **One shared formatter, no drift.** The canonical float renderer moved from `starkc::interp` into
    `stark_runtime::format` (dependency-free); `starkc::interp::canonical_float` now DELEGATES there,
    so the HIR oracle and the native binary format floats byte-identically by construction. Added
    `stark_runtime::format::{println_i64,print_i64,println_u64,…,println_f64}` and wired the primitive
    `Print*`/`Println*` `RuntimeFn`s in `emit_runtime`.
  - **`NATIVE_STDOUT_SUPPORTED` flipped to `true`** in `three_engine_differential`: the comparator now
    checks real stdout bytes across all three engines (83 pass).
  - **Proven:** `tests/c63e_formatting.rs` (7 — signed/unsigned ints incl. Int8/UInt8 widening, bool,
    Float64 canonical incl. `0.1`→`"0.1"` and `-0.0`, print-no-newline, mixed), each asserting native
    stdout == HIR oracle. `canonical_float` 6, `--lib` 441 (interp delegation), `three_engine` 83; fmt
    + clippy clean.
  - **DEV-105 [deferred]:** `println(Float32)` — the `f32→f64` widening (`widen_for_print`) makes the
    NATIVE binary see the f32-rounded value (`0.1f32 as f64 == 0.10000000149011612`) while the HIR
    interpreter keeps the wider `0.1`. A cross-engine **value-semantics** discrepancy in how the
    widening cast is evaluated, NOT a formatting issue (the canonical renderer is shared and correct).
    Fixing it needs a decision on where `Float32` rounding canonically occurs, then alignment across
    HIR/MIR/native. C6.3e remaining: composite `Display` (tuple/struct/enum/Option/Result/Vec/Box — a
    lowering feature, HIR-only today), `Float32` println (DEV-105), assert message text.

- CD-112 [2026-07-25, **WP-C6.1g-c CLOSED — dispatch-loop linearisation; the borrow-through-return
  refusal LIFTED**] The root cause of a broad class of native-build failures: every generated body
  was ONE `loop { match __bb { … } }`, so rustc could not see that a block runs once and treated a
  borrow held across blocks as live on the back-edge — colliding with the referent's single
  assignment (E0502/E0506). This blocked owned-`String` `==`/`<`, stored interior `&str`,
  `Vec<String>`-style pushes, and the `Option<&P>`-return shape.
  - **Fix.** An ACYCLIC body is now emitted as nested labelled blocks (`emit_bodies::linear_order`
    computes reverse-postorder + detects back-edges): later-RPO labels enclose earlier ones, so every
    forward `goto`/branch becomes `break 'bbTarget` and `Return` becomes `break 'stark_ret v`. rustc
    then flow-analyses each block as running once. A body WITH a real back-edge (while/for/loop) keeps
    the `loop { match __bb }` dispatch. Pure rendering change — same MIR, same control flow, same
    move/borrow/definite-assignment semantics; three-engine agreement preserved.
  - **The return-refusal is lifted.** `refuse_borrow_carrying_nominals` no longer refuses a function
    returning a borrow-carrying nominal (`Option<&P>`, etc.); it now builds and runs, consumed across
    the `Option::unwrap` blocks. The slot-backed Move borrow-carrying LOCAL refusal (part 2) stays —
    its `ValueSlot` drop still needs `&mut` while the stored borrow is live.
  - **Proven native (three-engine):** `wrap(&p) -> Option<&P>` then `o.unwrap().get()` and the inline
    `wrap(&p).unwrap().get()`; String `==`/`<`; stored `s.as_str()`; `Vec<String>::push`; plus
    `while`/`if` (dispatch + linear paths). Validation: `--lib` 441; the six `native_c61f_*` suites;
    the earlier 16-suite native/differential regression. `native_c61f_nominals`' return test flipped
    from refused→builds-and-runs.
  - **CI fixes folded in (CD-109 fallout).** Making `String` representable had silently invalidated
    two tests asserting the OLD "unsupported" boundary: `emit_types` `unsupported_constants…` (already
    fixed in `6be3428`) and `native_c5_4_function_values`' fnptr-over-`String` (now uses a bare
    `Slice`, still unsupported). Lesson recorded: run `cargo test --lib` after broadening an emitter's
    supported-set.
  - **`starkide` bin excluded from `cargo test`** (`Cargo.toml` `test = false`): the experimental
    terminal IDE (a side project, not the compiler) whose tests hung a local `--all-targets` run. It
    still builds; only its tests are skipped.

- CD-111 [2026-07-25, **WP-C6.3b PARTIAL — native Vec/Box value surface + the slot buffer-reclaim
  fix**] Extends the native runtime with the owning containers, and fixes a latent leak in the drop
  path that affected every owning value.
  - **The slot buffer-reclaim fix (load-bearing).** `ValueSlot<T>` holds `ManuallyDrop<T>`, and the
    MIR drop path emitted `slot.drop_with(|__v| <glue>)` where the glue runs USER destructors only.
    For an owning value with no user destructor (`String`, `Vec`, `Box`, and owning FIELDS of Drop
    structs) the glue was empty, so the allocation was never freed — a real leak (unobservable in
    the differential, which checks status/output not memory). Fix: `drop_with` now runs
    `ManuallyDrop::drop(held)` AFTER the glue. Rust's structural drop reclaims the buffer and drops
    elements (recursive-safe, at runtime); it never re-runs a user STARK destructor because
    generated nominal types implement no Rust `Drop`, and the glue frees no buffer — the two are
    disjoint, so exactly-once holds. The 24 `native_c6_1_ownership` destructor-order tests are
    unchanged (only unobservable buffer frees added).
  - **Vec/Box value surface:** `emit_ty` renders `Core(Vec,[T])→Vec<T>`, `Core(Box,[T])→Box<T>`;
    `stark-runtime/src/{vec,boxed}.rs`; wired `VecNew/WithCapacity/Push/Pop/Len/IsEmpty/Clear`
    (`Pop` reuses the Option bridge) and `BoxNew/IntoInner`. `VecElements`/`BoxInner` drop glue is
    now emitted (empty when the element has no user destructor — the slot's structural drop does the
    rest).
  - **Proven native (three-engine):** `Vec<Int32>` new/push/pop(Some/None)/len/is_empty/clear/
    return-across-fn; `Box::new`/`into_inner`; `Box<String>`. `tests/c63b_vec_box.rs` (9).
  - **Deferred:** (a) a `Vec`/`Box` whose element carries a USER destructor — refused pre-rustc
    (destructor-in-runtime-collection design); (b) `v.push(f(...))` where the pushed value is itself
    a runtime call, e.g. `Vec<String>::push` — the `&mut Vec` receiver borrow is held across the
    argument-evaluation block → **WP-C6.1g-c** (HIR+MIR pass). (c) trapping index/replace/remove,
    interior-ref `get`, iteration, slices — later slices.
  - **C6.1g-c is now the critical shared unblocker** — it gates owned-`String` comparison, stored
    interior `&str`, and `Vec<String>`-style pushes.

- CD-110 [2026-07-24, **WP-C6.3a cont. — native char ops + the Option-return bridge**] Extends
  CD-109 with the String Char surface and the foundational mechanism every collection accessor will
  reuse.
  - **Char ops (Char is a Copy scalar):** `PrintlnChar`/`PrintChar` (UTF-8 encode → runtime output
    sink; multi-byte scalars like `λ` verified), `StringPushChar`. Added
    `stark_runtime::string::{push_char, println_char, print_char}`.
  - **The Option-return bridge (foundational).** A `RuntimeFn` that yields a Rust `Option<T>`
    (`StringPopChar` now; `VecPop`/`VecGetRef`/`HashMapGet`/`CharsIterNext` later) is wrapped into
    the program's generated Option enum: `emit_call` threads the destination type to
    `emit_runtime_call`, which emits `match <rust option> { Some(__v) => Opt::V1(__v), None =>
    Opt::V0() }` (generated variants are TUPLE variants — the fieldless `None` needs `V0()`, the
    defect the first attempt hit). `stark_runtime::string::pop_char` added.
  - **Proven native (three-engine):** `println`/`print` of a char incl. Unicode, `push`, `pop`
    (Some/None/`unwrap_or`). `tests/c63a_string.rs` now 20.
  - **C6.3a remaining:** `chars()` iteration (`CharsIter{New,Next}` — shares the iterator
    representation, lands with C6.3c), string slicing views (C6.3b slices), cross-package String.
  - Regression: `mir_lowering` 4, `gate5_codegen` 14, `exec_snapshots` 4, `native_c6_1_ownership`
    24 — green. `fmt`/`clippy` clean.

- CD-109 [2026-07-24, **WP-C6.3a PARTIAL — native String/str value + output surface; WP-C6.3
  OPENED**] First slice of the Core native runtime (§23/§24). Until now `Callee::Runtime` was
  entirely unimplemented in the backend — native supported NO output or collection calls; every
  `Core(String/Vec/..)` type was refused by `emit_ty`. This slice builds the runtime-call bridge and
  the String/str surface end-to-end, three-engine (HIR/MIR/native).
  - **Landed:** `stark-runtime/src/string.rs` (STARK String/str semantics — byte `len`, UTF-8,
    lexicographic ordering, pinned in one reviewed place so they cannot drift with host `std`);
    `emit_ty` renders `MirTy::String → String`, `MirTy::Str → str`; `Constant::Str` → a Rust
    `&'static str` literal; `emit_runtime::emit_runtime_call` bridges `Callee::Runtime`; wired the
    String/str + str-output `RuntimeFn`s: `StringNew/FromStr/Clone/AsStr/Len/IsEmpty/Contains/
    PushStr/Clear`, `Str{Len,IsEmpty,ToString,Eq,Cmp}`, `Println/PrintStr`. `String` is Rust
    `String` (owning, non-`Copy`, slot-backed → MIR controls destruction).
  - **Proven native (three-engine):** construction (`from`/`new`), `len`/`is_empty`, `push_str`,
    `clear`, `contains`, `clone`, `str::to_string`, `str` len, return-`String`-across-fn, str-literal
    `==`/`<`, and `println`/`print` of a str with the native STDOUT BYTES checked against the
    oracle. `tests/c63a_string.rs` (15).
  - **Deferred to WP-C6.1g-c (native only; HIR+MIR pass):** a STORED interior `&str` borrowing an
    OWNED `String` held across a block — owned-`String` `==`/`<` (lowers through `String::as_str`)
    and an explicit `let v = s.as_str()` used after a branch. The stored borrow overlaps the
    `String`'s slot-drop across the block-dispatch `loop { match __bb }` back-edges (E0502) — the
    same dispatch-loop borrow-linearisation problem as C6.1g-c, NOT String-specific (`str`-value
    comparison works natively).
  - **C6.3a REMAINING (not in this slice):** char ops (`PrintlnChar`/`StringPushChar`/
    `StringPopChar`), `chars()` iteration (`CharsIter{New,Next}`), string slicing views, and
    cross-package String passing. Display/formatting of non-str values is C6.3e. Regression:
    `mir_lowering` 4, `native_c5_4_linkage`, `gate5_codegen` 14, `exec_snapshots` 4,
    `native_c6_1_ownership` 24 — all green. `fmt`/`clippy` clean (lib + `stark-runtime`).

- CD-108 [2026-07-24, **WP-C6.2e CLOSED — deterministic instance identity; WP-C6.2 as a whole
  CLOSED**] §21: a clean rebuild, relocation, and dependency-declaration reorder must leave every
  canonical symbol byte-identical, with no path/order artifact in semantic identity.
  - **Defect found and fixed.** Generic type arguments rendered a nominal as `struct#N`/`enum#N` —
    the raw `ItemId` INDEX, assigned by item walk order. Declaring two dependencies in the other
    order swapped the indices and changed the symbol (`callA@[struct#5]` ⇄ `callA@[struct#10]`), a
    §21 violation surfaced by a two-dependency reorder probe. `mir::lower::symbol_ty` now renders the
    nominal's CONTENT PATH (`struct#liba::A`): order-stable, relocation- and rebuild-stable, and
    still distinct from an identically-named core type (a user MAY declare `struct Vec` — the
    `struct#`/`enum#` head keeps it apart from core `Vec<..>`). `dump_ty` (debug body dump) is
    unchanged; the fix is scoped to the canonical symbol's five type-argument renderings in
    `key_symbol`. Named-path method/trait/Drop/assoc-fn symbols were already content-based.
  - Evidence: `tests/c62e_deterministic_identity.rs` (2: relocation+rebuild across two absolute paths
    of different length with a no-path/pid-leak assertion; dependency-declaration reorder). Regression:
    `native_c6_2_generics_traits` 20, `native_c5_4_linkage` 14, `native_c5_4_workspace` 12,
    `mir_lowering` 6, `cross_package_generics` 20, `c62c` 9, `c62d` 11 — all green; the linkage
    preflight accepts the content-based symbols. `fmt --check` and strict `clippy` clean.
  - **WP-C6.2 CLOSED.** §22 checklist met: all executable generic forms (a/b/c), all accepted
    trait/method forms (b), associated types concrete in MIR (c), operator dispatch follows STARK
    impls with no derive shortcut (d), one canonical instance emitted once (a), Drop/trait-only
    reachability (a), deterministic relocation-stable identity (e). Open remainders are NOT C6.2:
    the F4 parser half (`&&T`/`**x`), and DEV-083 (candidate-local inference snapshots) — neither a
    normative method-resolution rule. Next in Gate C6: **WP-C6.3** (runtime values/collections incl.
    output, Track C), then C6.4/5/6.

- CD-107 [2026-07-24, **WP-C6.2d CLOSED — operator/CoreTrait semantics**] The §20 matrix is proven:
  native execution invokes the user's STARK impl, and a Rust equivalent never substitutes. **No source
  change was required** — the dispatch was already correct; this WP proves it with an adversarial
  suite and documents the boundaries.
  - **Fully native (HIR+MIR+native), adversarial:** `Eq` always-true (distinct values compare equal —
    impossible under a Rust `PartialEq` derive), `!=` through the same `eq`, reversed `Ord` across all
    four comparison operators, observable `Clone` (+100), nonzero `Default` (via `P::default()`),
    `From` conversion.
  - **Anti-substitution, both directions.** The backend emits NO `#[derive(PartialEq/Ord/Clone/Hash)]`
    on STARK nominals; a MISSING impl is rejected — `==`/`<` without `Eq`/`Ord` → **E0500**, `.clone()`
    without `Clone` → **E0302** — never filled by a Rust derive.
  - **Dispatch proven in HIR+MIR; native runtime is C6.3 (Track C):** `Display` (`fmt` returns a fixed
    string, len 6 — a by-value `String` return) and `Hash` (constant `hash`, a nominal HashMap key
    that keeps both distinct keys). Same native-linkage boundary as C6.2c's `Vec` return; not a C6.2d
    gap.
  - **DEV-103 [deferred, owner decision]** — `.into()` deriving from a `From` impl (blanket `Into`) is
    not provided; `a.into()` with only `impl From<A> for B` in scope is E0302. The spec (06-Standard-
    Library) lists `From`/`Into` as INDEPENDENT traits with no mandated blanket impl. `Fahrenheit::from(c)`
    is the supported form. Ergonomic, not correctness.
  - **DEV-104 [deferred, owner decision]** — `Default::default()` with a type-inferred target (no
    receiver) is E0005 "qualified trait method requires a receiver". The spec mandates only
    `fn default() -> Self`; `P::default()` is the supported form. Ergonomic, not correctness.
  - Evidence: `tests/c62d_operator_coretrait.rs` (11: 6 native adversarial, 2 HIR+MIR dispatch, 3
    rejection). `fmt --check` and strict `clippy` clean. (No lib change → no broad relink; the suite
    and its dependencies build green.)
  - **C6.2 remaining:** C6.2e (deterministic instance identity — §21). The F4 parser half
    (`&&T`/`**x`) is still open.

- CD-106 [2026-07-24, **WP-C6.2c CLOSED — associated types**] The §19 matrix is proven across all
  three engines. Baseline already worked: an associated-type declaration + impl binding, `Self::Item`
  in return and parameter position, and an associated type that is a nominal or a tuple. Four gaps
  fixed:
  1. **`T::Item` through an explicit binding** (`fn f<T: Holder<Item = Int32>>`): the projection now
     normalises to the bound type. `check_trait_member_call` rewrites `Self::Item` in the method's
     return to the receiver's projection (`T::Item`), then `assoc_binding_map` + `normalize_projections`
     pin it from the in-scope `Trait<Item = ..>` binding.
  2. **`T::Item` inferred from the call argument** (`fn first<T: Holder>(t: T) -> T::Item`): a
     program-wide `assoc_projections` table `(nominal, assoc) -> bound` (front end AND MIR lowerer)
     resolves `<H as Holder>::Item`; where the base is still an inference variable at the call, a
     **deferred projection obligation** is recorded and discharged the moment the call's arguments
     unify (so `build(H {}).v` sees a concrete type). Verified MIR never carries a residual
     `Ty::Param("T::Item")` — native emit's C4.5 residual-param refusal enforces this and the reachable
     bodies compile+run.
  3. **Cross-package projection** (DEV-101 provenance): `check_trait_member_call` converts the
     signature's types (including `Self::Item` associated-name spans) against the TRAIT's file, not the
     caller's — previously produced a mangled `T:::Ite` and E0001. Fixed; the dependency-declared
     trait's projection resolves in an app-declared generic.
  4. **Drop-bearing associated types** flow through projections unchanged.
  - **Scope boundary:** returning a runtime collection (`Vec<..>`) BY VALUE across a function boundary
    is a separate native-linkage limitation (C6.3) — a plain `fn f() -> Vec<_>` hits the identical
    refusal — so it is not part of this closure. Associated-type resolution for such a signature is
    correct (HIR + MIR pass); only the native linkage of the value return is deferred to C6.3.
  - Evidence: `tests/c62c_associated_types.rs` (9: self-item return/param, assoc-nominal, assoc-tuple,
    inferred projection, explicit binding, by-value projected use with field access, nested
    projection-then-method, cross-package — three-engine where applicable). Regression: lib 441,
    `native_c6_2_generics_traits` 20, `cross_package_generics` 11, `conformance`/`gate4`/`gate5`/`gate7`
    semantics, `exec_snapshots` 4, `native_c5_4_linkage` 12, `native_c5_4_workspace` 6 — all green.
    `fmt --check` and strict `clippy` clean.
  - **C6.2 remaining:** C6.2d (operator/CoreTrait dispatch parity) and C6.2e (deterministic identity);
    the F4 parser half (`&&T`/`**x`) is still open.

- CD-105 [2026-07-24, **WP-C6.2b-F6 CLOSED — impl signatures may spell the concrete type for
  `Self`; C6.2b matrix cleared**] `impl Mk for G { fn make() -> G {..} }` for `trait Mk { fn make()
  -> Self; }` was rejected E0500 "signature incompatible", because the compatibility check keyed
  `Self` (trait) and the concrete `G` (impl) to different strings — yet in `impl … for G`, `Self`
  IS `G`. Fix: `typecheck` keys the impl's self type in the SAME format a path produces
  (`ty_signature_key`) and returns that for any `Self` mention, so `Self` and the written self type
  (`G`, `&G`, `W<Int32>`) compare equal. A DIFFERENT concrete type (`-> H`) still mismatches and is
  rejected — no over-accept. Evidence: `tests/c62b_f6_self_normalisation.rs` (5: return-Self-as-
  concrete, return-Self-as-Self, param-`&Self`-as-concrete, generic-self via a `&Self` param, and
  the wrong-type negative; native three-engine where applicable). **Found in passing (separate, not
  fixed):** `W::<Int32>::make(7)` — a generic associated-fn call via turbofish — reports E0005
  wrong-arity; unrelated to F6, worth a follow-up.
  - **C6.2b matrix CLEARED.** F1 (privacy, the only accepted-invalid), F2, F5, F6 closed; F3 closed
    (→ WP-C6.1f); F4 split (parser half `&&T`/`**x` — open; selection is Track B). C6.2b no longer
    blocks Gate C6 on findings.
  - `fmt --check` clean; F6 suite + lib 441 green. (Broad targeted regression not re-run for this
    commit per owner instruction; last full green at CD-100 confirmation, 70 suites.)

- CD-104 [2026-07-24, **WP-C6.2b-F2 CLOSED — specific-instance impl matches an inferred receiver**]
  `impl Get for W<Int32>` did not match `let w = W { v: 7 }; w.get()` (E0302, receiver `W<_infer>`).
  Not a "specific-instance impls unsupported" bug — an ANNOTATED `w: W<Int32>` already worked; the
  receiver's int-literal argument (`7`) was simply not defaulted to `Int32` before method
  resolution. `default_int_literals_deep` now defaults literals INSIDE the receiver type (03 solving
  step 5), so `W<_infer>` becomes `W<Int32>` and the concrete-instance impl matches. Only literal
  variables are touched (`int_literal_vars`); a genuine unbound inference var is left alone, so a
  different instance (`W<Bool>`) stays rejected — no over-accept. Evidence:
  `tests/c62b_f2_specific_instance.rs` (5, incl. native, a nested-literal case, and the negative
  guard). Regression green (lib 441, native_c6_2 11, three_engine 83, conformance 56,
  exec_snapshots, gate2_valid); `fmt --check` and strict `clippy` clean. C6.2b remaining: F6.

- CD-103 [2026-07-24, **WP-C6.2b-F5 CLOSED — impl-head bounds visible in method bodies**] The
  WP-C6-ENTRY §2 carry-forward. A method call on a bounded generic *function* parameter already
  resolved through its bound, but a bound on the IMPL head (`impl<T: Sh> W<T> { fn go(&self) {
  self.v.a() } }`) was invisible in the body (E0302 "method 'a' not found for type 'T'"). Fix:
  `typecheck` tracks `current_impl_generics` (set around each impl's method bodies in Pass 2) and
  consults it alongside `current_fn_generics` when resolving a method on a `Ty::Param` receiver.
  An unbounded impl param still rejects the method (no over-accept). Evidence:
  `tests/c62b_f5_impl_bounds.rs` (4, incl. native three-engine and the negative guard). Regression
  green (lib 441, native_c6_2 11, three_engine 83, conformance 56, gate2_valid, cross_package);
  `fmt --check` and strict `clippy` clean. C6.2b remaining: F2, F6.

- CD-102 [2026-07-24, **WP-C6.2b-F1 CLOSED — privacy enforcement for callable/member resolution**]
  F1 (the accepted-invalid privacy hole) is fixed at the FRONT END; invalid access stops before
  lowering. Module-level items were already enforced by `resolve::item_is_visible_from`; the gap was
  impl members and fields, which resolve in `typecheck` with no visibility check. Fix: `resolve`
  exposes its module map as `hir.item_modules`; `typecheck` tracks the use-site module
  (`current_module`) and enforces one shared predicate `check_member_visible` (private is
  exact-module, matching resolve; emits **E0207**) at four points — inherent-method selection,
  associated-function resolution, struct-field read, and struct-literal construction. Trait/default
  methods keep their trait-path visibility; a plain reference return etc. is unaffected.
  - **Probe/inventory (§4), all now rejected pre-lowering:** private inherent method `s.hidden()`,
    private associated fn `S::secret()`, private field read `s.v`, private field construction
    `S { v }`, and neither method syntax nor qualified syntax bypasses. Same-module private and
    public cross-module access stay accepted; private top-level fn stays enforced by resolve.
  - **Evidence:** `tests/c62b_f1_privacy.rs` (11: 4 positive + 7 negative). Regression green with no
    over-rejection: lib 441, `gate2_valid` 11, `native_c6_2_generics_traits` 11,
    `three_engine_differential` 83, `conformance` 56, `cross_package_generics` 20 — the WP-C6.2a
    canonical-identity fixtures unchanged. `fmt --check` and strict `clippy` clean.
  - **C6.2b matrix:** F1 struck from the finding list. F2/F5/F6 remain (after C6.1g), F3 is closed
    (→ WP-C6.1f), F4 is split. F1 no longer blocks C6.2b; the remaining findings do.

- CD-101 [2026-07-24, **WP-C6.1g-a follow-up — 5 full-suite test-churn failures fixed**] The CD-100
  full run surfaced 5 failures, all test-churn from the semantic change, no code regressions: four
  used all-Copy structs as Move stand-ins (`c61f_reference_boundary` move-while-borrowed;
  `native_c6_1_ownership` c61c/c61d/multi-level partial-move) → switched to the existing
  `Drop`-bearing variants; one was the conformance baseline greping `**OWN-COPY-001.**` (heading
  reformatted) → restored to `**OWN-COPY-001.** — Copy eligibility.` and spec regenerated.
  **Confirmation full workspace run: exit 0, 70 suites, 0 failures** — CD-100 + CD-101 fully
  validated. `fmt --check` and strict `clippy` clean.

- CD-099 [2026-07-24, **WP-C6.1 CLOSED**] All ten `WP-C6.1f.md` §2 scope items are implemented with
  native evidence or carry an owner-approved disposition, and all five exit criteria are met:
  reference storage, cross-block flow, reference parameters, nested references (representation +
  syntax), reborrowing (receiver + argument, incl. generic callees), reference returns, and
  borrow-carrying aggregates and most nominals all build and run natively with three-engine
  agreement; move-while-borrowed and the no-NLL case are correctly rejected and pinned. The full
  workspace suite is green (exit 0, 68 suites); `fmt --check` and strict `clippy` clean. Four
  limitations carried out of the package, all owner-dispositioned under CD-097 and **none blocking
  this closure**: borrow-carrying nominal slot/return shapes (`WP-C6.1g-a`), conservative return
  lifetimes (`WP-C6.1g-b`), `Box`/`Vec`/slice representability (C6.3), and `Box` deref (correct
  rejection, not a deviation). **C6.1f closure does NOT move Gate C6** — the first three remain
  explicit Gate-C6 dependencies. Packet: `WP-C6.1f-CLOSURE.md`. With C6.1a–e (CD-080…084),
  **WP-C6.1 as a whole is closed.**

- CD-098 [2026-07-24, **WP-C6.1f-b2 completion — generic-callee argument weakening**] The last
  unblocked implementation item in C6.1f. A generic callee's `fn_types` entry still names the
  callee's OWN parameters (`Ty::Param("T")`), which the CALLER's substitution cannot ground, so the
  expected type at the argument boundary was unresolvable and no `&mut T` -> `&T` weakening was
  applied — leaving the call to fail MIR verification. The call's concrete type arguments are
  already computed for the instance and are in the callee's generic declaration order, so they are
  exactly the substitution needed (`mir::lower::callee_param_types`, generic names read via
  `item_text` per DEV-101).
  - **Why the previous best-effort fallback was right as an interim and wrong as an end state:**
    resolving against the *caller's* map would be **worse than declining** — inside a generic body
    with a same-named parameter it would silently pick up the WRONG type instead of failing. The
    helper therefore substitutes explicitly rather than reusing ambient state, and stays
    best-effort per parameter (an unresolvable entry means no weakening, never a mislowering).
  - Closes the b2 boundary set: function arguments, fully qualified trait-call arguments, annotated
    local init, assignment, return expressions — and now generic callees. Aggregate fields remain
    open only because borrow-carrying nominals are (`WP-C6.1g-a`).
  - Evidence: 4 new tests in `native_c61f_b2_weakening.rs`.

- CD-097 [2026-07-24, **OWNER DISPOSITIONS — the four C6.1f recorded limitations**] None of the four
  prevents **WP-C6.1f package closure**; items 1–3 remain explicit **Gate C6** dependencies and item
  4 leaves the deviation list entirely. Full text in `C6-INTEGRATION-LEDGER.md` §7.
  - **1. Borrow-carrying nominal values and returns — temporary deviation, ASSIGNED** to
    **`WP-C6.1g-a` Borrow-Carrying Nominal Lifetime Emission** (Track A). Initial approach is
    generated lifetime-parameter threading; **no `ValueSlot` or CE4 runtime-layout change without a
    probe demonstrating necessity**. Blocks Gate C6.
  - **2. Conservative returned-reference lifetimes — temporary sound over-rejection, ASSIGNED** to
    **`WP-C6.1g-b` Return-Source Lifetime Precision** (Track A): a result derived only from `a` must
    not be tied to an unrelated `b`; may-derive-from-either stays tied to both. Blocks Gate C6
    native-conformance closure.
  - **3. `Box`/`Vec`/slice native representability — SCOPE-OUT TO C6.3 APPROVED.** Permits C6.1f
    closure; blocks Gate C6 while those normative forms are unsupported.
  - **4. `Box` dereference — CORRECT REJECTION, NOT A DEVIATION.** Core v1 defines `Box::new` and
    `Box::into_inner` and defines no `Box` dereference, `Deref` trait, or method auto-dereference
    through `Box`. **Removed from the deviation list.** Status documents calling it an
    implementation gap were corrected — CD-089's bullet here and two rows in
    `C6-REFERENCE-MATRIX.md`. **The correction was already on record earlier in this file and my
    CD-089 bullet contradicted it; the error was mine.**

- CD-096 [2026-07-24, **WP-C6.1f — borrow-carrying nominals; lifetime parameters on generated
  types**] A generated nominal is a *declared* Rust type, so unlike a tuple it cannot borrow
  implicitly: a reference in a field needs a lifetime parameter or rustc reports `E0106`. Generated
  nominals now carry one.
  - **Two spellings, not one.** `Name<'a>` in the type's own declaration; `Name<'_>` at every use
    site. They are not interchangeable — `'_` is illegal in a field type (no enclosing binder to
    infer from), while a named `'a` at a use site would demand every use site bind one.
    `emit_types::LifetimePosition` makes the distinction explicit and `emit_ty_at` threads it
    through nested types. Only instances that actually carry a borrow gain the parameter, so every
    existing generated type is byte-identical.
  - **Working natively:** `Some(&x)`/`None` at `Option<&T>`; matching on `Option<&P>` and using the
    bound reference; `Option<Option<&T>>`; `Option<&T>` inside a tuple; plain `Option<Int32>`
    unaffected.
  - **The C6.1f-a design question, finally located.** §5 predicted `ValueSlot`-versus-borrow-checker
    would be the crux. b3 showed it was not the blocker for plain references (that was definite
    assignment) and aggregates showed it was not for tuples (not slot-backed). **It is real here and
    only here**: a slot-backed borrow-carrying nominal, and a function returning one, both fail
    `E0502`. **Removing the slot is not an escape — it was tried**: the slot also carries MOVE
    liveness, so without it the mover fails instead. Both shapes are refused before rustc.
  - **Validation: full workspace suite exit 0 — 68 suites, zero failures**; `fmt --check` and strict
    `clippy` clean. Evidence: `starkc/tests/native_c61f_nominals.rs` (6).

- CD-095 [2026-07-24, **WP-C6.1f — borrow-carrying aggregates; tuples/arrays land, nominals
  refused before rustc**] OWN-CARRY-001 makes borrow provenance **structural** — through tuples,
  generic arguments and enum payloads — so a tuple or array of references is ordinary Core v1.
  Declared reference *fields* stay forbidden (03 rule 1, front-end E0001) and are pinned.
  - **The property is "carries a borrow", not "is a reference".** Relaxing the lane to admit
    aggregates only moved the failure: a **`Copy` aggregate of references** is not slot-backed, so it
    was default-initialised — and `default_value_expr` cannot fabricate a reference, one level down
    for exactly the reason it cannot fabricate one directly. Generalising b3's rule from *is* to
    *carries* (`ty_carries_reference`) fixed the class at once; non-`Copy` borrow-carrying
    aggregates are already slot-backed and untouched.
  - **Supported natively:** tuple of two references; tuple of struct references; mixed tuple; array
    of references; nested borrow-carrying tuple; a borrow-carrying tuple crossing basic blocks; a
    tuple of references to **`Drop`-bearing** values.
  - **Borrow-carrying NOMINALS are refused — deliberately, and before rustc.** `Option<&T>` and a
    user generic at a reference need lifetime parameters a generated Rust struct/enum does not have,
    so rustc would report `E0106` **in the generated crate**. That would break this backend's
    defining property: an unsupported program must be refused on *our* side of the boundary as a
    named STARK limitation, never as a compiler error in code the user never wrote. A new
    `refuse_borrow_carrying_nominals` raises it deterministically, naming the missing capability.
    Tuples work and nominals do not for one reason: a tuple is a **structural** Rust type whose
    lifetimes rustc infers; a generated nominal is a **declared** type needing explicit ones.
  - `native_c5_3_aggregates_enums.rs`'s lane test rotated its negative case a third time
    (`store` → b3, `ret` → the return step, `ref_in_tuple` → here), each time following its own
    "if it is now legitimately supported, move it to a positive test" instruction.
  - **Lifting the nominal restriction** needs lifetime parameters threaded through generated type
    declarations and every use site — field types, locals, signatures, drop glue, variant
    construction, match patterns — interacting with §11.2's shared-`'a` signature machinery. A
    self-contained next step, not a small edit.
  - **Evidence:** `starkc/tests/native_c61f_aggregates.rs` (6). **Validation: full workspace suite
    exit 0 — 67 suites, zero failures** (including `spike_cranelift`, confirming the temp-path fix);
    `fmt --check` and strict `clippy` clean.

- CD-094 [2026-07-24, **WP-C6.1f — returning a reference; lane check 5 removed**] The last of the
  five lane checks with real semantics behind it. **Provenance is the front end's**: OWN-RETURN-001
  rules 2/3 already reject (E0103) a returned reference not derived from a reference *parameter*, so
  the backend does not re-check it — the blanket "a reference may never be returned" is removed.
  Two mechanisms made the emission compile, both found by probing rather than predicted:
  - **The E0381 wall again, in new places.** A reference that is a `Call` destination or an
    `if`/`match` join result is written in one block and read in another — the same
    definite-assignment problem b3 hit in a `let`, now in the caller and at join points. b3's fix
    generalised: a reference **temporary spanning more than one block** is `Option<&T>`-backed,
    subsuming both concrete triggers into the property that actually matters. **Parameters are
    excluded** (initialised at entry by the caller) — an early over-broad version Option-backed them
    and broke, which is what forced the distinction. Same-block ephemeral temporaries stay bare.
  - **Return-position access moves out of the `Option`** (`unwrap()`), never re-borrows: a re-borrow
    would borrow from the dying return-slot local and dangle.
  - **Projecting through a returned reference** (`f(&p).field`, `f(&p).method()`) materialises the
    call result into a temp, via the same non-place fallback `RefOf` and receivers already used.
  - **Lifetimes = OWN-RETURN-001's shortest-input rule.** Two or more reference parameters leave the
    output lifetime ambiguous (E0106); a **single shared `'a`** on every reference parameter and the
    return encodes the intersection — the shortest of all inputs (03 rule 3). Zero or one reference
    parameter is handled by Rust's own elision, which is why a `&self` accessor never needed it.
    **Conservative and reported:** for `pick(a, b) -> a` STARK's shortest is `a`'s lifetime alone,
    but the shared `'a` also ties it to `b` — sound (never accepts what STARK rejects) though it can
    reject a valid program whose return derives from a longer-lived subset. Precise per-path
    provenance is a later refinement.
  - **Still refused:** returning a reference to a **local** (E0103, front end) and a reference stored
    in an **aggregate** (lane check 3). The `native_c5_3_aggregates_enums.rs` lane test's `ret` case
    followed its own "move it to a positive test" instruction — as `store` did at b3 — leaving the
    aggregate case as its remaining negative.
  - **Evidence:** `starkc/tests/native_c61f_ret_refs.rs` (8), incl. the two E0103 negatives; the
    C6.1f negative corpus (6) passes unaltered.
  - **Validation:** `fmt --check` and strict `clippy` clean. Full workspace suite **in progress at
    commit time — 50 suites green, 0 failures** (it had also caught a clippy-only regression earlier,
    now fixed); all scoped suites touching this change pass.

- CD-093 [2026-07-24, **WP-C6.1f-b3 — stored references; the lane replaced, not deleted**]
  - **The §5 design question had the wrong answer.** The matrix predicted the crux was `ValueSlot`
    versus Rust's borrow checker. Probing with the lane disabled showed otherwise: a same-block
    borrow bound to a user local **already built and ran, including for a `Drop`-bearing owner**.
    The blocker was `E0381 "used binding isn't initialized"` — rustc's **definite-assignment**
    analysis, not its borrow checker; a reference local is assigned in one arm of the generated
    block-dispatch `loop { match … }` and read in another. **No borrow error appeared in any case.**
    Third time in C6 that probing overturned a pre-measurement assumption.
  - **Fix:** a reference bound to a **user** local is declared `Option<&T> = None`, definitely
    initialised at its declaration; MIR liveness still decides legality and `unwrap` names a state
    MIR proved unreachable (the `slot_violation` posture). **Compiler temporaries keep the bare
    form** — same-block by construction, so rustc's definite-assignment check still guards them and
    every previously working reference path is byte-identical.
  - Two non-obvious details: **`Option<&mut T>` is not `Copy`**, so access re-borrows out of the
    `Option` rather than moving out of it; and **borrowing needs a place expression**, since read
    mode may substitute a raw-projection *copy* helper for a `Copy` field and `&<copy>` would
    reference a temporary rather than the field — a silently wrong reference, not a compile error.
    A distinct `PlaceMode::Borrow` keeps the place form.
  - **Lane checks narrowed, never deleted** (§4's requirement): checks 1 and 5 kept intact; 2 and 3
    admit user bindings only (aggregates still refused); 4 still binds temporaries to one block.
    The negative corpus passes unaltered, no-NLL case included. `native_c5_3_aggregates_enums.rs`'s
    lane test carried the instruction "if it is now legitimately supported, move it to a positive
    test" — its **store** case did exactly that; its **ret** case stays refused.
  - **Twelve shapes now build and run natively**, including references across `if`/`while`, `&mut`
    in a user local, borrows of fields/nested fields/array elements, a `Drop`-bearing owner,
    borrow-then-move, and the b2 annotated-local weakening that was waiting on the lane.
  - **Still open in C6.1f:** returning a reference (check 5) with OWN-RETURN-001 provenance
    validation; references in aggregates (check 3); b2's aggregate-field and generic-callee
    weakening; b4's parser half; b5's E0103 message.
  - **Validation: full workspace suite green — 65 suites, exit 0, zero failures** (warranted here:
    `emit_places.rs`/`emit_bodies.rs` are cross-cutting for the whole backend); `fmt --check` and
    strict `clippy` clean.

- CD-092 [2026-07-24, **WP-C6.1f-b2 — expected-type reference weakening; 5 of 6 boundaries**]
  Two defects had to be fixed **together**, because either alone leaves the boundary unusable:
  **borrowck** consumed a `&mut` argument (so `f(m); f(m);` was E0100) and now **re-borrows**;
  **lowering** never emitted the conversion (so MIR verification rejected the call) and now
  re-borrows at the expected mutability. `weaken_ref_to` also covers the **same-mutability** case —
  passing `&mut T` where `&mut T` is expected must re-borrow too, or the reference is moved and a
  second use fails V-MOVE-1, the MIR-level twin of the borrowck E0100.
  - Each re-borrow is a *temporary* borrow ending with its statement (03 rule 4), so **no borrow
    duration changed**: the C6.1f-a negative corpus passes unaltered, no-NLL case included.
  - **Boundaries:** function arguments ✅ native; fully qualified trait-call arguments ✅ native;
    annotated local init, assignment, and return expressions (both `return m;` and a tail `m`) all
    emit the weakening correctly and now **reach the ephemeral-reference lane**, i.e. they are
    blocked only by b3. **Aggregate fields are NOT done** — they need the expected field types of a
    generic nominal instantiation, and no nominal-generic substitution helper exists in lowering
    (`impl_generic_subst` covers impl heads, not struct instantiations). Substituting wrongly there
    would produce a **silent miscompile** rather than a refusal — the one failure mode this package
    has been free of — so it is reported rather than approximated.
  - **A full-suite run caught 6 regressions the scoped set missed**, all from one root cause: the
    call-arm resolved the callee's `fn_types` at the call site, but for a **generic** callee those
    are still `Ty::Param`, which the caller's substitution cannot ground. Expected-type resolution
    is now best-effort — an unresolvable parameter type means no weakening for that argument, never
    a lowering failure. Consequence to note: **generic callees do not yet get argument weakening.**
  - **Validation:** the six previously-failing suites re-run green (`native_c6_2_generics_traits`,
    `native_c5_4_workspace`, `exec_snapshots`, `three_engine_differential`, `cross_package_generics`)
    plus the C6.1f suites and the negative corpus; `fmt --check` and strict `clippy` clean. A second
    full-workspace run was deliberately **not** performed — the failing suites plus the scoped set
    are the signal, and b2 is not a closure point.

- CD-091 [2026-07-23, **OWNER RULING — b2 REVISED; my spec reading was wrong**] I claimed
  argument-position conversion "does not exist" in CD-090, citing TYPE-METHOD-002. **That was
  wrong, and the error was mine: I cited TYPE-METHOD-002 without checking the coercion rules it
  defers to.** A function parameter is an **expected-type boundary**, and the closed set of built-in
  coercions applies at expected-type boundaries: 03-Type-System "Reference Coercions" gives
  `&mut T -> &T`, and **TYPE-COERCE-003** gives `&[T; N] -> &[T]`, `&mut [T; N] -> &mut [T]`, and
  mutable-weakened-to-shared. TYPE-METHOD-002 prohibits argument-position **auto-borrow**,
  **auto-dereference** and **user-defined** coercion — not the fixed built-in set. So **the checker
  is correct to accept these forms and the verifier/backend refusal is an implementation gap, not
  front-end over-acceptance**; rejecting them would have contradicted frozen Core v1 coercion rules.
  - **TYPE-METHOD-002 clarified editorially** in `03-Type-System.md` (a clarification of existing
    frozen semantics, not an amendment): argument expressions may still undergo the closed built-in
    expected-type coercions. Spec regenerated; the 112-block fixture corpus stays in sync.
  - **C6.1f-b2 REVISED, not dropped — "Expected-type reference weakening", Track A.** `&mut T -> &T`
    at expected-type boundaries: ordinary function arguments, fully qualified trait-call arguments,
    annotated local initialisation, assignment, return expressions, and aggregate fields where
    applicable. Must **re-borrow rather than move**, preserving the lexical borrow rules b1 proved.
    Does not depend on slice representation and does not wait for C6.3.
  - **Array→slice coercion moves to C6.3b** with slice-parameter representability (TYPE-COERCE-003
    native execution), covering `n(&a)`, `n(&mut a)` and `n(&a[0..3])` together. The prerequisite is
    representation — `n(&a[0..3])`, with no coercion involved, is refused with "param 0 is not
    C5-representable" — and that prerequisite does **not** justify rejecting `n(&a)`.
  - **Checker behaviour fixed by the ruling:** it must not reject either normative coercion merely
    because native support is incomplete. Native build may issue a deterministic unsupported-profile
    diagnostic for slice parameters until C6.3b lands, but `check` must keep accepting valid Core
    source. **C6 cannot close while either normative coercion remains unsupported.**
  - **Probe of all six boundaries:** every one fails today, in two ways — five at MIR verification
    (the weakening is never emitted) and three with **E0100 "use of moved value"** (borrowck moves
    the `&mut` instead of re-borrowing). Both fixes land in **Track A files** (`borrowck.rs`,
    `mir/lower.rs`), so b2 needs **no typecheck lease** and does not collide with Track B's F1 work.

- CD-090 [2026-07-23, **WP-C6.1f-b1 CLOSED — receiver re-borrowing; b2 blocked**]
  - **A probe re-scoped both sub-packages before any code changed.** Explicit re-borrow syntax
    **already works end-to-end natively** (`f(&*m)`, `f(&mut *m); f(&mut *m);` all run). That makes
    TYPE-METHOD-002's closing sentence operative: *"No argument-position auto-borrow,
    auto-dereference, or user coercion exists."* The matrix's nine verifier refusals are therefore
    **two different problems split by position**: receiver position is a genuine lowering gap the
    spec *requires* (b1); argument position is a **front-end over-acceptance** where the spec says
    the conversion does not exist and the explicit form the user should write already works.
  - **b1 implemented.** Lowering passed an already-reference receiver through as a value, which was
    wrong twice: it never adjusted `&mut T` to `&T`, and it **moved** the reference (`&mut T` is not
    `Copy`), so `m.bump(); m.bump();` failed V-MOVE-1. Receivers are now dereferenced via the
    existing `lower_place_autoderef` and **re-borrowed at the method's required mutability**. Each
    re-borrow is a temporary borrow ending with its statement (03 rule 4), so **no borrow duration
    changed** — the C6.1f-a negative corpus passes unaltered, including the no-NLL case.
  - **Free gain: F4's representation half is done.** Peeling every layer means repeated auto-deref
    now lowers and verifies; the nested-receiver rows moved from verifier-refused to
    backend-lane-refused, pinned by a test that stops at MIR so b3 need not rediscover it. The
    parser half (`&&T`/`**x` unspellable) and selection (Track B) are untouched.
  - **b2 is blocked and was mis-scoped — needs a ruling.** Array→slice unsizing is argument-position
    coercion (which the spec says does not exist), *and* the explicit form fails anyway:
    `n(&a[0..3])` is refused with "param 0 is not C5-representable" — **slice parameters are not
    natively representable at all**, which is Track C's C6.3. Recommended: drop b2 as a sub-package
    and fold the argument-position question into one decision (reject at the checker naming the
    explicit form), since that narrows the accepted language and is the owner's call.
    **[SUPERSEDED by CD-091 — this recommendation rested on a wrong reading of the spec; the
    coercions are normative and b2 was revised rather than dropped.]**
  - **Validation:** twelve at-risk suites green including all 441 lib tests, three-engine (83) and
    the C6.1f negative corpus; **no snapshot re-pin needed** despite changing a very common lowering
    path. `fmt --check` and strict `clippy` clean. Evidence: `starkc/tests/native_c61f_reborrow.rs`.

- CD-089 [2026-07-23, **WP-C6.1f-a COMPLETE — the reference matrix**] 51 cases driven end-to-end
  across the ten `WP-C6.1f.md` §2 scope items. Classification only; no source change.
  `STARKLANG/docs/compiler/work-packages/C6-REFERENCE-MATRIX.md`.
  - **No miscompilation exists.** Every engine pair that ran agreed; nothing was accepted-but-wrong.
    Every gap is a refusal, so C6.1f is a **capability** package, not a soundness repair — the
    opposite of F1, which is why the ruling's ordering (F1 first) is right on severity grounds.
  - **MIR already represents and executes references-in-locals correctly.** All fifteen
    backend-refused rows verify *and run to a correct answer under the MIR interpreter*. The gap is
    **generated-Rust emission, not reference representation** — this removes the package's largest
    unknown, though not its difficulty.
  - **The lane boundary is "freshly-taken borrow", not "reference".** Reference *parameters* work
    natively today, including stored in a user local (`fn f(r: &P) { let q = r; q.get() }` runs).
    Only materialising a new `RefOf` outside a same-block compiler temporary is refused.
  - **Two missing mechanisms are not storage at all:** reborrow `&mut T` → `&T` (receiver and
    argument position) and array → slice unsizing account for all nine MIR-verifier refusals. `&mut`
    params are also **moved rather than reborrowed**, which surfaces as two different failures in
    two different phases (E0100 at typecheck; "move from possibly-moved place" at MIR verify).
  - **`Box` deref is a CORRECT REJECTION, not a gap** (owner disposition CD-097 item 4, and
    already recorded earlier in this file): Core v1 defines `Box::new`/`Box::into_inner` and has no
    `Deref` trait; TYPE-METHOD-002 peels only `&`/`&mut`. `*b`, `(*b).field` and method lookup
    through `Box` are therefore *supposed* to be rejected. **This bullet originally called it a
    front-end gap, contradicting the correction already on record — the error was mine.**
    (`Box`/`Vec`/`str` REPRESENTABILITY remains Track C's C6.3.)
  - **Six conformant refusals locked by permanent tests before implementation**
    (`starkc/tests/c61f_reference_boundary.rs`), including the no-NLL case Rust's NLL accepts and
    Core v1 does not. This is the §2 item 10 constraint made mechanical rather than aspirational.
  - **Awaiting approval:** the five-way C6.1f-b split in matrix §7, and specifically whether the
    reborrow and unsizing sub-packages land first as independent conformance fixes (they need no
    lane change and no CE3) or whether all of it waits on the lane replacement design.

- CD-088 [2026-07-23, **C6.2b F1/F3 OWNER RULINGS; WP-C6.1f OPENED**] Dispositions for the six
  C6.2b findings, and a scope correction to Gate C6.
  - **F1 → Track B, C6.2b BLOCKER.** The privacy under-rejection is fixed before F2, F5, DEV-083 or
    C6.2c. No lease: `resolve.rs`, `typecheck.rs`, the C6.2 tests and the generics/traits matrix are
    Track B-owned; a narrow lease is requested only if shared authority-bearing files prove
    necessary. Enforcement must sit at the **semantic access point** rather than block-listing the
    three discovered examples — field projection, method-call selection, associated-function
    selection, fully qualified calls to private impl members, generic and cross-package versions,
    defining-module access still accepted, public members of a private type not making that type
    externally nameable, and inherent-member privacy kept distinct from trait-member accessibility.
    Ranked first because it is the only finding that **expands the accepted language beyond Core
    v1** rather than temporarily rejecting valid code.
  - **F3 → new WP-C6.1f, Track A.** *General Reference Storage, Reborrowing, and Provenance*
    (`STARKLANG/docs/compiler/work-packages/WP-C6.1f.md`). NOT absorbed into C6.2b: method
    resolution merely exposed it, while the problem is reference storage, liveness, provenance, MIR
    verification and native emission. Track A owns it as semantic integration lead — the work
    intersects ownership-liveness, MIR lowering/verification, `ValueSlot` conventions and backend
    place emission; Track C is prohibited from changing ownership-liveness, and Track B keeps
    method-selection behaviour built on top of the resulting contract. **Status wording corrected to
    "C6.1a–e closed; C6.1f open because the C5 general-reference deferral was not assigned during
    C6 planning" — a scope correction, not evidence the completed Drop/ownership work was invalid.**
    Ten scope items incl. **no NLL expansion**; explicitly ruled that **removing a validator check
    so `let r = &p` passes would be an unsafe patch, not an implementation of F3**. CE3 for
    MIR/verifier contract changes, CE4 for runtime representation/ABI.
  - **Dependency order: F1 → C6.1f/F3 → F4 → remaining F2/F5/F6 → C6.3b.** F4 stays split —
    nested-reference type parsing and MIR/reference representation to C6.1f, repeated auto-deref
    *selection* to Track B afterwards.
  - **`CLAUDE.md` corrected immediately** (`0873308`, narrow docs commit): its "auto-deref one
    reference level" contradicted normative TYPE-METHOD-002 and would have led a future agent to
    implement the wrong limitation.

- CD-078…CD-084 [2026-07-23, **GATE C6 OPENED; WP-C6.0 and WP-C6.1a–e CLOSED**] Gate C6 (Native
  Semantic Parity) is a **three-track parallel** gate — Track A ownership/Drop (Claude), Track B
  generics/traits (Gemini), Track C runtime/collections (Codex) — executing on `main` (the owner
  waived the entry plan's §7C branch/worktree model). Governance lives in
  `STARKLANG/docs/compiler/work-packages/C6-{SHARED-CONTRACTS,FILE-OWNERSHIP,INTEGRATION-LEDGER}.md`.

  - **CD-078 — WP-C6.0 (contract freeze) CLOSED.** Froze the authority-bearing contracts every track
    consumes (versions, `VerifiedMirProgram` precondition, `Instance`/canonical-symbol identity,
    `ValueSlot` invariants, `DropPlan` authority, trap + runtime-call identity, the three-engine
    comparator schema, Tier-1 targets, no-host-semantic-substitution), per-track file ownership with
    a single-writer lease protocol, and the integration ledger. Integration base `db73afe`.
  - **CD-079 — WP-C6-ENTRY APPROVED**, discharging §1's opening conditions.
  - **CD-080…CD-084 — WP-C6.1 (ownership and Drop parity) CLOSED.** The C6.1a audit was
    **probe-grounded** (24 shapes driven through the real backend) rather than assumed, and found the
    C5 ownership surface far more complete than the exit report implied — all common cross-block
    movement already at parity. It surfaced four concrete gaps, **all now closed**:
    - **G3** multi-level (depth ≥2) partial move/drop — chained `addr_of_mut!` raw projection helpers
      at any depth (C6.1b).
    - **G4** loop-carried reassignment of a no-`Drop` non-`Copy` local — a **compile-then-abort** bug
      (the slot is never reset by a MIR `Drop` for a non-droppable type); fixed with the additive
      `ValueSlot::reinit` (C6.1b). Surfaced only because C6.1b re-probed by native *execution* — the
      C6.1a probe had checked `emit` success alone. **Method correction recorded.**
    - **G1** multi-unit enum-payload consuming match / partial move (the CD-070 boundary) — owner
      ruling "refined Option A": lowering canonicalises the payload into ONE
      `Aggregate(Tuple, [VariantField(v,0..n)])` statement and the backend emits a single
      destructuring `take()` match; per-field movement is then ordinary tuple machinery. Not a CE3
      (existing MIR ops only); cross-block backend analysis explicitly prohibited (C6.1c).
    - **G2** non-`Copy` array by-value iteration — owner ruling "Option (a)": unconditional
      unrolling into `ConstIndex(i)` moves with a fresh binding local per iteration; **DEV-090 fully
      CLOSED** (the front-end E0104 rejection removed — the HIR oracle moves each element, so the
      feared divergence does not exist) (C6.1d).
    - **C6.1e** — the Drop-path matrix (`C6-DROP-PATH-MATRIX.md`), evidence only. Reuses C5.3d-1c's
      **trapping-destructor position probe** (native has no stdout, but a trap's category and exact
      `file:line:column` are comparable in all three engines), adding the §13 exit paths the C5.3d-1c
      set did not reach: inner block scope, loop body per-iteration, `break`, `continue`, `return`,
      `?`, match-arm end, failed pattern test; and no-cleanup-after-trap for overflow, cast, index
      and assertion failures. Two rows genuinely wait on C6.3: byte-level Drop-*log* comparison and
      IO/provider-failure cleanup.

  - **Validation at closure:** `cargo fmt --check` and strict workspace `clippy` clean; full
    `cargo test --workspace --all-targets --no-fail-fast` green. Evidence lives in
    `starkc/tests/native_c6_1_ownership.rs` (24) and `three_engine_differential.rs`'s `c61e_*` (12).

- CD-087 [2026-07-23, **WP-C6.2b PARTIAL — DEV-102 closed; §18 matrix probed; F1–F6 opened**]
  The §18 method-resolution matrix was driven end-to-end
  (`parse → resolve → typecheck → HIR-run → lower → verify → emit → native-run`). Eleven of the
  fifteen rows are green natively, and two rejections were confirmed **correct**: two traits
  supplying `go` is E0203 (ambiguity), and `let r = &mut p; r.bump(); p.get()` is E0101 (Core v1
  borrows are lexically scoped to end-of-block — there is no NLL, so this is conformant, not a bug).
  - **DEV-102 CLOSED.** TYPE-METHOD-001 requires fully qualified `Trait::method(&recv)` and requires
    it to *bypass trait-name lookup*. Lowering gained a `Res::TraitMember` arm selecting through a
    new **trait-filtered** `find_trait_impl_fn`. Reusing `find_impl_fn` would have been wrong: it
    answers "what does `recv.m()` mean", so it prefers inherent methods and takes any in-scope
    trait. The qualified form is the spec's own remedy for E0203, proven by `A::go(&s)` and
    `B::go(&s)` selecting different impls while `s.go()` still prefers the inherent method. Because
    the receiver is written explicitly, no auto-borrow/auto-deref applies, so every argument lowers
    as an ordinary operand — which is why the arm is small. Not a CE3 (existing MIR ops only).
    Covered: plain call, the disambiguation pair, inherent-shadowing, default bodies, extra
    arguments, `&mut` receivers, `Drop`-bearing receivers; E0203/E0005 asserted to persist.
  - **F1–F6 opened, awaiting disposition** (`C6-GENERICS-TRAITS-MATRIX.md` §7). **F1 is the only one
    that accepts invalid programs**: private impl members (methods, associated fns) and private
    struct fields are reachable cross-module, though module-level items *are* enforced — a violation
    of MOD-VIS-001 and TYPE-METHOD-001 step 5, in Track B's front-end area. **F3 is a scope gap, not
    just a defect**: `let r = &p; r.get()` is refused by the backend ("C5 ephemeral reference lane"),
    yet §18 lists shared/nested-reference receivers as C6.2b rows while the C5 exit report defers
    "general references" to "C6" without naming a sub-package — so no C6 package currently owns it,
    and C6.3b's slices/Box ("borrow/deref", "returned-reference provenance") depend on it. F2
    (trait impl on a specific generic instantiation), F4 (nested-reference receivers; `&&T` is
    unspellable and inferred `&&T` fails MIR verify though TYPE-METHOD-002 makes repeated auto-deref
    normative), F5 (impl-head bounds invisible in method bodies — the §2 carry-forward, still open)
    and F6 (impl signatures do not normalise `Self`) are over-rejections.
  - **Doc defect:** repo `CLAUDE.md` says method calls "auto-deref one reference level"; normative
    TYPE-METHOD-002 says auto-dereference "repeatedly removes one leading `&`/`&mut`".
  - **Evidence:** `native_c6_2_generics_traits.rs` now 20 tests (8 new `c62b_*`). Scoped regression
    across ten at-risk suites green; `fmt --check` and strict workspace `clippy` clean.

  - **Remaining C6:** WP-C6.2 (generics and static trait dispatch) and WP-C6.3 (runtime
    values and collections — String/Vec/Box/iterators/maps/**output**/files, Track C) are the bulk of
    the gate; then C6.4 Tier-1 platform matrix, C6.5 full differential/generated corpus, C6.6
    adversarial review and gate exit.

- CD-086 [2026-07-23, **WP-C6.2a — canonical callable identity; native dispatch unblocked**]
  A probe of twelve generics/trait shapes found **nine refused before rustc** (two already worked;
  one is a separate lowering gap) — every method, trait, operator and associated-function call among
  them. Cause: `Instance` identity is
  `(item, type_args, symbol)`, and while **bodies** derived `item` from the `FnKey`, **call sites**
  passed the **receiver nominal**, so one canonical symbol carried two item identities and the C5.4a
  linkage preflight (correctly) refused the program. The full suite had stayed green only because no
  native test exercised an ordinary method call — destructors resolve through
  `TypeContext::drop_impls`, a different path. This confirms C6.1b's method correction a second time:
  **coverage of a mechanism is not coverage of the surface that uses it.**
  - **Owner ruling — a conformance correction, NOT a CE3/CE4.** `C6-SHARED-CONTRACTS.md §3` was
    *violated*, not changed; no MIR shape, verifier rule, `mir_version`, symbol scheme, ABI or
    accepted-language semantics moves. Ruling further directed: **do not patch the six sites
    independently** — introduce ONE lowering-internal constructor
    `FnLowerer::instance_from_key(&FnKey) -> Instance` and route **every** `Instance` through it
    (`MirBody.instance`, ordinary methods, trait-impl calls, default trait calls, `Eq` dispatch,
    `Ord` dispatch, associated functions), removing the defect *class*. Implemented exactly so.
  - **Result:** eleven of the twelve probe shapes now build and run natively, as do two further
    shapes added as regressions (a method on a generic nominal, and a cross-package trait call) —
    inherent, generic-nominal and
    method-level-generic methods; user-trait dispatch; bounded-generic bound calls; default trait
    methods; associated types and associated functions; cross-package trait calls. **`Eq` and `Ord`
    operator dispatch are proven adversarially** (an always-true `eq` and a reversed `cmp` both give
    answers a Rust `derive` would contradict), discharging §20's "STARK's impls, not Rust's".
  - **The linkage consistency check was not weakened** — `a_mismatched_item_is_still_rejected`
    proves it still fires; and every case now asserts directly that each `Callee::Instance` reference
    and its defining body share identical `symbol`/`item`/`type_args`.
  - **DEV-102 opened, deliberately kept separate** per the ruling: fully-qualified `Trait::method(&r)`
    still reports `LOWER: callee form (C4.5)`. It is a missing callee-lowering form unrelated to the
    identity defect, and belongs to **C6.2b method-resolution completion** (alongside the deferred
    DEV-083), not to this correction.
  - **Evidence:** `starkc/tests/native_c6_2_generics_traits.rs` (12) and
    `STARKLANG/docs/compiler/work-packages/C6-GENERICS-TRAITS-MATRIX.md`. Scoped regression across
    the ten at-risk suites (lib 441, `mir_differential`, `mir_lowering`, `mir_verify`,
    `exec_snapshots`, `conformance`, `three_engine_differential` 83, `native_c6_1_ownership` 24,
    `native_c5_3_aggregates_enums`, `gate4a_prelude_traits`) green; `fmt --check` and strict
    workspace `clippy` clean.

