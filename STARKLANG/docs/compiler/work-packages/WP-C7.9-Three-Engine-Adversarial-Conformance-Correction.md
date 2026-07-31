# WP-C7.9 — Three-Engine Adversarial Conformance Correction

**Status:** PROPOSED — awaiting owner approval. No packet is authorised to start.
**Parent gate:** C7 (open; see `COMPILER-STATE.md` "Gate C7 — RULING (CD-262): QUALIFICATION-BLOCKED,
NOT CAPABILITY-BLOCKED")
**Inputs:** two adversarial three-engine review passes — the current-head source/evidence pass
(§§1–10 of that review, referenced below as **R1.n**) and the later probe pass (findings **F1–F5**).
**Adjacent, not absorbed:** `WP-C7-Usage-Shape-Qualification` (proposed) — see §7.
**CD allocation:** highest number in `git log --all` at drafting is **CD-269**; this WP allocates
from **CD-270** upward, re-read at execution time.

---

## 1. Objective, and the claim it corrects

> Accepted STARK source must receive the same specified outcome across HIR, MIR, and native
> execution.

The first review states the gap precisely, and this WP adopts its framing as the objective's
motivation:

- **Not currently true:** "every type-correct STARK program behaves identically in HIR, MIR and
  native."
- **Still supportable:** "the admitted executable corpus agrees across HIR, MIR and native debug on
  the qualified targets" — itself bounded by C6's own limit of 59 of 87 audited standard-library
  methods, one verified invocation each, not every usage shape.

Closing this WP does not by itself promote the narrow claim to the broad one. It removes the known
obstacles to that promotion and makes the remaining distance measurable. **Packet J records which
claim the evidence then supports**, and that wording is an owner decision, not a drafting one.

## 2. Why one work package

Both passes inspected the same assurance boundary and overlap literally, not thematically: both
found stderr invisible to the comparator, both found surfaces the front end admits that no engine
below HIR executes, and both showed that agreement testing cannot see a defect the engines share by
construction. Two WPs would duplicate the findings inventory, severity rulings, comparator changes,
corpus version bump, requalification run and gate-impact analysis — and would produce two ledgers
that disagree.

The umbrella is governance and qualification. **Implementation stays packetised.** The affected
surface spans type checking, HIR execution, MIR lowering, the MIR interpreter, generated Rust, the
runtime, comparator normalisation, test infrastructure and the normative specification. One broad
branch across that is the change shape this repository has already paid for twice. Every packet
below carries an explicit path-ownership set and its own commit sequence; only Packet J integrates.

## 3. Findings ledger

Every entry was re-derived against the tree at `28a9ad1` before being written down. Where a review's
description and the code disagree, the code is recorded — DEV-047 is the precedent for why (a prior
review's `MIN / -1` claim was overstated and had to be corrected after the fact).

| ID | Claim as reviewed | Verified position at HEAD | Class | Packet |
| --- | --- | --- | --- | --- |
| F1 | `MIN / -1` does not trap | **Corrected: it traps in all three engines; the divergence is the trap *category*.** `mir/lower.rs:4602–4603` maps `Div`/`Rem` to one static `TrapCategory::DivideByZero` for every cause, so `MIN / -1` reports `DivideByZero` in MIR and native while the oracle reports "integer overflow" (`interp.rs:2110–2118`). | category divergence | A |
| F2 | `MIN % -1` does not trap | **Confirmed, and the more serious of the two.** MIR (`mir/interp.rs:1275`) and native (`emit_bodies.rs:2052`) compute `checked_rem` on an `i128` carrier then range-filter; `MIN % -1` yields `0`, in range, so no trap fires and the program **completes with a value**. HIR traps via an explicit guard (`interp.rs:2079–2085`, DEV-047). | completion-vs-trap divergence | A |
| F3 | Malformed trait impls (`Ord::cmp`) reach execution | **Confirmed, with a sharper cause than "the general rule is wrong".** User-declared traits *are* checked — `trait_method_signature_matches` (`typecheck.rs:3676`), rejected `E0500` (`:3584`). A `CoreTrait` has no declaration item (`typecheck.rs:6104–6106`: "each `impl <CoreTrait> for T` writes its own method signature directly"), so **no conformance check runs for any Core trait impl**. Separately, that check `continue`s when the impl has no matching item (`:3573–3581`), so "missing required method" is not its responsibility. | admission failure | B |
| R1.1 | HIR binds borrowed payloads by clone | **Confirmed live.** `match_pattern` still takes `&Value` (`interp.rs:4660–4665`) and has no place to build a `Value::Ref` from. Type checker, MIR and native bind `&String`; HIR clones. CD-267 pinned this as a divergence test (`tests/ce1_borrowed_payload_binding.rs`) and **escalated rather than patched it**, explicitly because the fix is structural. PAT-BIND-001 is now normative (`04-Semantic-Analysis.md:283`, CD-269), so the spec half is done and the oracle is the outstanding half. | semantic divergence | C |
| F4 / R1.2 | Accepted surfaces only HIR executes | Confirmed as a class: by-value `Vec` iteration, `map`, `filter`, `count`, `collect` — front end accepts, HIR runs, MIR lowering refuses. The tests deliberately pin that boundary. | split language | E |
| F5 / R1.3 | stderr has no three-engine semantics | Confirmed, and two defects under one name: HIR's `eprint`/`eprintln` write straight to the process stderr (`interp.rs:2830–2832`) rather than the captured `Execution.stderr` (`:89–90`, today only `Err`-completion bytes); MIR has no lowering at all (`mir/interp.rs:123`); native emits nothing; the comparator's `stderr_bytes` therefore observes empty in every ordinary case. | observation gap | D |
| R1.4 | Provider-backed programs are native-only by construction | Confirmed by design — the MIR interpreter does not execute `Callee::Provider`. Not a defect; a **mis-stated evidence class**. args/env, time, TCP, `File`, and every future host resource receive native + verifier + ABI evidence, never three-engine evidence. | claim scoping | H |
| R1.5 | Shared omissions pass all three engines | Confirmed: **DEV-118** — the `T: Hash + Eq` bound is unenforced for `HashMap`/`HashSet`; all engines accept the same invalid instantiations because storage scans by `Eq`. Carried open in `COMPILER-STATE.md:30`, owned by WP-C6.3. Agreement proves consistency, not conformance. | conformance gap | I |
| R1.6 | Usage-shape qualification incomplete | Confirmed: DEV-119 is the demonstration. `WP-C7-Usage-Shape-Qualification` exists but is **proposed, not complete**. | carried scope | §7 |
| R1.7 | Differential testing is native-debug centric | Confirmed: `run_native` uses the debug profile; release is a selected fourth mode, not the ordinary corpus. | methodology | G |
| R1.8 | Engines are not fully independent | Confirmed: the MIR interpreter calls `crate::interp::canonical_float` (`mir/interp.rs:28–29, 1519, 1523`). Reasonable sharing, but HIR and MIR cannot disagree on float rendering, so the differential proves nothing there. | methodology | G |
| R1.9 | Inconsistent skip behaviour | Confirmed: `three_engine_test!` prints `SKIP` and `return`s when `rustc` is absent (`tests/support/differential.rs:1177–1180, 1187–1190, 1197–1200`), whereas `agree_completing_available_engines` falls back to HIR+MIR. Two meanings for "native unavailable". | methodology | G |
| R1.10 | HIR trap identity derives from prose | Confirmed: substring matching over diagnostic text (`integer overflow`, `division by zero`, `invalid shift`, `out of bounds`). Wording changes would silently reclassify semantics. | methodology | G |
| — | Recursion depth aborts the interpreter process | Confirmed as a robustness gap. **No new normative limit is needed** — `LIMIT-RESOURCE-001` (`07-Modules-and-Packages.md:306–310`) already names *call-depth* and classifies exhaustion as a host/process failure implementations "must prevent … and report". | robustness | F |

Three ledger notes carry into scope:

- **F1 and F2 have one repair.** Both need the checked-arithmetic terminator to report the category
  of the *cause*, not of the *operator*. Fixing F2 alone leaves `MIN / -1` mis-categorised; fixing F1
  alone leaves a wrong value in a completed program. They ship together.
- **F4/R1.2 and F5/R1.3 are the same architectural fact seen twice.** `eprintln` is an
  accepted-HIR-only surface *and* the stderr gap. Packet D implements it end-to-end because the
  channel must exist before it can be compared; Packet E dispositions the rest of the class.
- **R1.10 is why F1 could hide.** A category derived from prose and a category fixed per operator are
  the same weakness at two ends of the comparison. Packet G item 3 and Packet A close it from both
  sides.

## 4. Packets

Each is independently closable with its own CD, commit sequence and evidence. A packet may not edit
paths outside its ownership set without an amendment recorded here.

### Packet A — Integer trap correctness *(P0, blocks C7 closure)*

**Scope.** F1 + F2 together. The checked-arithmetic path reports the category of the failure:
`MIN / -1` and `MIN % -1` are `IntegerOverflow` at every signed width; `x / 0` and `x % 0` remain
`DivideByZero`. MIR interpreter and generated-Rust backend, with the user's source location
preserved as CD-131 established for indexing.

**Design constraint.** A static per-operator category cannot express this; the terminator (or the
checked-op evaluation) must yield the category. That is a **MIR contract change → D1 in §6**. Do not
implement before the ruling.

**Exit criteria.**
1. `MIN / -1` and `MIN % -1` trap `IntegerOverflow` in HIR, MIR, native debug and native release at
   `Int8`/`Int16`/`Int32`/`Int64`.
2. `/ 0` and `% 0` still trap `DivideByZero` at every width, signed and unsigned.
3. Unsigned types unaffected (no negative `MIN`); non-`-1` divisors unaffected.
4. Exhaustive `Int8` × `Int8` div/rem cross-engine table; boundary-generated cases at wider widths.
5. Permanent corpus sentinels, not test-local cases.

**Paths.** `starkc/src/mir/{mod,lower,interp}.rs`,
`starkc/src/backend/generated_rust/emit_bodies.rs`, `starkc/tests/`,
`STARKLANG/docs/compiler/mir.md` (amendment), corpus.

### Packet B — Core-trait implementation conformance *(P0)*

**Scope.** Extend signature conformance to `CoreTrait` impls, where none runs today: receiver form,
parameter types and arity, return type, generic parameters and bounds, associated items, and
missing / extra / duplicate items. Audit the existing user-trait check on the same axes rather than
assuming it complete — missing required methods are not its responsibility today, and that
responsibility must land somewhere explicit.

**Exit criteria.** A malformed implementation of any Core or user trait is rejected during type
checking with a stable `E`-code and a span on the offending item; it never reaches HIR execution and
never reaches MIR verification. Negative fixtures on every axis, for both trait kinds.

**Paths.** `starkc/src/typecheck.rs`, `starkc/src/resolve.rs` (only if trait-ref resolution is
implicated), `starkc/tests/`, `STARKLANG/docs/spec/04-Semantic-Analysis.md` if a code is minted.

### Packet C — HIR place-aware pattern execution *(P1)*

**Scope.** Implement PAT-BIND-001 in the reference interpreter — the half CD-267 escalated. Pattern
execution becomes place-aware:

```text
match_pattern(pattern, source: Value | Place, binding_mode)
```

`Copy` components bind by value; non-`Copy` components matched through a reference bind as
`Value::Ref` to the projected place; owned scrutinees continue to move; nested matching preserves
the referent's frame and lifetime; bindings are shared even through `&mut`, per the spec's
deliberate floor.

**Exit criteria.** `tests/ce1_borrowed_payload_binding.rs`'s pinned divergence case is **converted
into a positive three-engine case**, not deleted; all 19 matrix cases agree across engines including
the two that currently do not; PAT-BIND-001's stated compatibility consequences hold in HIR.

**Dependency for claims.** Until this closes, `stark-json` should not be described as three-engine
qualified: its native path may be complete, but the oracle does not implement the language rule its
borrowed recursive traversal depends on.

**Paths.** `starkc/src/interp.rs`, `starkc/tests/`. Kept separate from Packet B despite both being
HIR-adjacent: this is a place model, not a check.

### Packet D — stderr as a compared channel *(P1)*

**Scope.** HIR appends `eprint`/`eprintln` to `Execution.stderr` instead of writing through to the
process; MIR gains stderr print operations; the native runtime writes and flushes stderr; the
comparator observes exact stderr bytes. Program stderr emitted *before* a trap must stay
distinguishable from the native runtime's own trap diagnostic — that separation is the packet's
point, not a detail of it.

**Required cases** (from R1.3, adopted verbatim as the acceptance set): `eprint` without newline;
`eprintln`; multiple writes preserving order; stdout and stderr in one program; stderr before a
trap; user `Display` dispatch through `eprintln`; formatting that traps before producing output.
Ordering between the two streams and flushing at trap/abort are qualified explicitly rather than
left to the host.

**Paths.** `starkc/src/interp.rs`, `starkc/src/mir/{mod,lower,interp}.rs`,
`starkc/src/backend/generated_rust/`, `starkc/stark-runtime/`, `starkc/tests/support/differential.rs`,
`starkc/tests/`, `STARKLANG/docs/compiler/mir.md` (amendment).

### Packet E — Accepted HIR-only surfaces *(P1/P2, decision packet)*

**Scope.** Disposition every surface the front end admits that only HIR executes — by-value `Vec`
iteration, `map`, `filter`, `count`, `collect`, plus whatever the audit adds. Per surface, one of:
**(a)** implement MIR types and lowering plus native support, or **(b)** refuse in the front end with
a diagnostic that names the limitation.

**Constraint.** "Accepted by type checking, executable only in HIR" is not permitted as a steady
state — that is the finding, and any row left in it re-opens the class. R1.2 recommends
implementation over refusal, on the grounds that parser, HTTP, CSV and data-processing packages all
pressure iterator composition; that recommendation is recorded, and the disposition remains **D3 in
§6**.

**Exit criteria.** Every audited surface is (a) or (b) with evidence; the audit table is committed.

### Packet F — Execution resource exhaustion *(P1)*

**Scope.** Deep recursion must not abort an interpreter test process without classification. Apply
`LIMIT-RESOURCE-001`, which already names call-depth and already classifies exhaustion as a
host/process failure — **not** a language trap. No new limit ID is minted; a documented capacity, if
needed, is declared as an implementation limit under that rule.

**Exit criteria.** HIR and MIR interpreters report a classified host/process failure rather than
crashing; tests exercise it in subprocesses; native execution either obtains a controlled failure or
its bounded limitation is documented as a deviation with the reason (**D4 in §6**). Independent of
Packets A and C.

**Paths.** `starkc/src/interp.rs`, `starkc/src/mir/interp.rs`, `starkc/tests/`,
`starkc/docs/conformance/KNOWN-DEVIATIONS.md` if the native half is documented rather than fixed.

### Packet G — Comparator and qualification hardening *(P1)*

The methodological findings — which are what let A–F survive earlier passes.

1. **Release profile.** Every admitted case runs native debug **and** release; the required relation
   becomes `HIR == MIR == native-debug == native-release`. Highest-value after MIR optimisation and
   for overflow/trap behaviour, `Float32` rounding, drop timing, loop control, partial moves, output
   before traps, and provider resource cleanup.
2. **Skip semantics (R1.9).** `three_engine_test!` delegates to
   `agree_completing_available_engines` (`tests/support/differential.rs:1060`) and records the
   missing engine explicitly. No macro arm returns without comparing something.
3. **Trap identity (R1.10).** Every language trap carries an explicit category at its construction
   site; prose becomes diagnostic content only. The substring normaliser is removed, not tuned.
4. **Independently pinned expectations (R1.5).** Each case pins its expected observation against the
   specification, so three engines agreeing on a wrong answer fails. Extended to rejection and
   trait-bound cases, not only stdout/return/drop/trap.
5. **Shared evaluators (R1.8).** Where engines share a normative algorithm — `canonical_float` today,
   any optimiser/interpreter evaluator later — the shared implementation carries exact-value tests
   independent of both engines, mutation tests proving those tests can fail, and boundary coverage
   for NaN, infinities, signed zero and `Float32` rounding.
6. **Adversarial seeds.** The probe seeds from both passes are committed as maintained tests.
7. **Boundary and property coverage** expanded on the axes A–F exposed.

Item 4 is the one that would have caught F2 unaided; it is not optional polish.

### Packet H — Provider evidence class *(P1)*

**Scope.** R1.4. Two parts, and the first is mandatory even if the second is declined.

1. **State the evidence class correctly** wherever a provider-backed capability or package is
   described as qualified:

   ```text
   Pure language semantics       :  HIR == MIR == native
   Provider binding / lifecycle  :  verifier + synthetic ABI tests + native execution
   ```

   `stark-io` in particular is not three-engine qualified while no interpreter provider model
   exists. Pure-STARK convenience functions over a fake provider can be; real filesystem integration
   stays native-qualified.
2. **Deterministic in-memory test provider** executable by the MIR interpreter, scripting success
   and failure status codes, borrowed and consumed handles, output buffers, failed `HandleOut`,
   exact close events, and short reads/writes — so provider-*call* semantics become comparable
   between MIR and native. Real OS behaviour remains native-only. (**D5 in §6.**)

**Paths.** `starkc/src/mir/interp.rs`, `starkc/src/provider_*.rs`, `starkc/tests/`, `c78/`
capability records, package `EVIDENCE.md`/`README.md` files whose claims change.

### Packet I — Shared-omission conformance *(P1)*

**Scope.** R1.5's concrete instance: close **DEV-118** — enforce `T: Hash + Eq` for `HashMap` and
`HashSet` — with specification-derived negative tests. The current implementation scans by `Eq` and
never uses the hash, which is exactly why all three engines accept the same invalid programs; that
becomes a live divergence the moment one implementation starts hashing.

**Exit criteria.** Invalid instantiations are rejected at type checking in every engine's front end;
negative fixtures derive from the specification's bound, not from current behaviour; DEV-118 is
closed or its remaining part explicitly re-carried with an owner. Coordinate with WP-C6.3, which
owns it today.

### Packet J — Integration, requalification and claim statement *(closes the WP)*

The only packet that integrates. Corpus version bump, full four-configuration requalification,
ledger reconciliation, `COMPILER-STATE.md` and C7 record updates, gate-impact statement, and the
**explicit statement of which conformance claim the resulting evidence supports** (§1).

## 5. Sequencing

```
A ──┐
B ──┤
C ──┼──> G ──> J
E ──┤
D ──┘
F ──────────── > J
H ──────────── > J
I ──────────── > J
```

- **A first**, and it blocks C7 closure. It is the only packet that changes a program's observable
  result rather than its acceptance or its observability.
- **G depends on D** for the stderr channel and on A/B/C/E for the expectations it pins; G's
  independent items (1, 2, 3, 5) may start in parallel.
- **F, H, I are independent** of the rest and of each other.
- **J last**, and only J.

R1's own priority ruling maps onto this as: fix-immediately → C, D (and the CE1 spec amendment,
already discharged by CD-269); next qualification track → §7's usage-shape matrix, G.1, E; before
broad host-package claims → H, I.

## 6. Decisions requiring an owner ruling before implementation

Flagged, not resolved.

| # | Decision | Shape | Blocks |
| --- | --- | --- | --- |
| D1 | Checked-arithmetic terminators carry a cause-dependent trap category (MIR contract amendment). | CE3 | Packet A |
| D2 | Confirm `IntegerOverflow` — not a new category — is the right identity for `MIN / -1` and `MIN % -1`, and that `03`/`07` need no amendment. | CE-spec (may be a no-op) | Packet A |
| D3 | Per-surface disposition in Packet E: implement vs. refuse. Refusal narrows the language for programs that compile today under HIR. | owner, per row | Packet E |
| D4 | Native call-depth exhaustion: fixed, or documented as a bounded deviation. | owner | Packet F |
| D5 | Build the deterministic interpreter-side provider model, or define provider-backed packages as native-qualified only. Part 1 of Packet H proceeds either way. | owner | Packet H (part 2) |
| D6 | Whether Packet B's new rejections extend `E0500` or mint a fresh code. | CE2-shaped | Packet B |
| D7 | Which conformance claim Packet J is authorised to state. | owner | Packet J |

## 7. Relationship to existing work packages

- **`WP-C7-Usage-Shape-Qualification` (proposed).** R1.6's finding is that WP's subject, and this WP
  does **not** absorb it — duplicating its risk matrix here would create two owners for one
  question. It is instead declared a **C7 closure dependency alongside this WP**: the highest-risk
  matrix (APIs returning or retaining references and resources, across immediate use, stored local,
  reassignment, pass-through, nested match, loop exhaustion, `break`, `continue`, early return, `?`,
  trap path, Drop-bearing referents, generic and package-boundary use) is where the next compiler
  defects are most likely, on DEV-119's precedent.
- **`WP-C6.3`** owns DEV-118 today; Packet I coordinates rather than reassigns.
- **CD-262/264/265** stand as written. This WP appends to the C7 record; it does not rewrite the
  ruling or the deviation history.

## 8. Closure criteria

WP-C7.9 closes when all of the following hold:

1. F1–F5 and R1.1–R1.10 each have an explicit, recorded disposition.
2. No valid admitted case differs in completion, trap, or trap category across HIR, MIR, native
   debug and native release.
3. No malformed trait implementation — Core or user — reaches execution.
4. Normal and pre-trap stderr are captured and compared, distinguishably from runtime diagnostics.
5. Every accepted-language surface either lowers or is cleanly refused; none remains HIR-only.
6. HIR implements PAT-BIND-001; no pinned semantic-divergence test remains in the suite.
7. Resource exhaustion cannot crash an interpreter test process without classification.
8. Provider-backed capabilities are described in the correct evidence class everywhere they are
   claimed, including package-level `EVIDENCE.md` files.
9. DEV-118 is closed or explicitly re-carried with an owner and a reason.
10. No engine can silently skip: an unavailable engine is a reported skip with a reason, or a
    failure.
11. Trap identity is structural everywhere; no semantic classification derives from diagnostic prose.
12. Expected observations are independently pinned against the specification, not merely agreed
    between engines; shared evaluators carry engine-independent tests.
13. The adversarial suite is committed and reproducible from seeds.
14. C7 and C10 records are updated without rewriting historical decisions, and Packet J states the
    supported claim in the words D7 authorises.

## 9. What this WP does not do

- It does not re-open Gate C6's closure, the Native Provider ABI, or any ruled CE decision.
- It does not extend the accepted language: Packet E may add lowering for surfaces already admitted;
  it may not admit new ones.
- It does not absorb `WP-C7-Usage-Shape-Qualification` (§7).
- It does not repair the `stark-*` first-party packages ad hoc; where a fix here changes their
  qualification, Packet J reports it.
