# Post-C10 deviation reproduction

**Programme:** `STARK-Post-C10-Compiler-Deviation-Repair-Programme.md`  
**Baseline branch:** `origin/develop`  
**Baseline SHA:** `689d26d26990399d1de3026c13c271c403a45032`  
**Repair branch:** `codex/post-c10-deviation-repair`  
**Worktree clean at branch creation:** no — pre-existing untracked `docs/` and `stark-http-get/`  
**Population command:** `python3 starkc/scripts/c10-deviation-populations.py`

## Population at baseline

```text
REPRODUCED_OPEN=13
NON_REPRODUCING=0
DORMANT=1
ACCEPTED=1
```

The live compiler population reported by repository authority is:

```text
DEV-120
DEV-140
DEV-141
DEV-142
DEV-143
DEV-144
DEV-145
DEV-157
DEV-159
DEV-160
DEV-167
DEV-168
DEV-180
```

`DEV-011` is accepted indefinitely and `DEV-179` is dormant. `DEV-214` remains in the tool's
manual `ADJUDICATE` bucket and is outside this post-C10 repair population.

## DEV-180

status: REPRODUCED, then REPAIRED in this working tree  
baseline SHA: `689d26d26990399d1de3026c13c271c403a45032`  
reproducer: `starkc/docs/conformance/KNOWN-DEVIATIONS.md` last `DEV-180` heading plus source
inspection of `starkc/src/interp.rs` method epilogue  
expected: `&mut self` binds a `Value::Ref` to caller storage; no method write-back path is needed  
actual at baseline: materialization already bound `Value::Ref`, but method error cleanup still
contained the legacy `RefMut` restoration/write-back path  
engine(s): HIR interpreter  
user impact: HIR oracle carried obsolete receiver machinery even after the materialization repair  
repair packet: P1 / DEV-180

Required questions:

1. Body-visible `self` for `&mut self` is `Value::Ref(caller_place)`.
2. The caller receiver place remains populated during the method call.
3. The legacy write-back path was reachable only on `eval_block` returning `Err`.
4. If reached, it restored the callee receiver local into the caller place; after current
   materialization that value is itself a `Value::Ref`, not the owned receiver.
5. Normal return already used reference write-through. `Result::Err` propagation is represented as
   `Flow::Propagate`, not this `Err` path. Language traps abort via `Err`. Returned references are
   still passed through `rebase_frame_refs`. `Drop` uses the separate `OwnedForDrop` backing path.
   Nested mutable receiver calls use the same caller-place reference representation.
6. A returned `&mut` points into caller storage after `rebase_frame_refs`.
7. `starkc/tests/as3_receiver_materialization.rs` and the interpreter mutation controls encode
   the normative receiver-boundary behavior; no test was updated for this repair.

Repair:

```text
starkc/src/interp.rs
  - removed obsolete `RefMut` error-path write-back
  - removed borrowed `RefMut` receiver locals before method-frame cleanup, same as `Ref`
```

Negative controls:

```text
interp::tests::audit_10c_a_mut_self_receiver_must_keep_place_identity
interp::tests::class_2_an_owned_value_behind_a_reference_receiver_is_refused
starkc/tests/as3_receiver_materialization.rs::a_by_value_self_receiver_consumes_the_resolved_place
starkc/tests/as3_receiver_materialization.rs::a_shared_self_receiver_does_not_consume_the_callers_place
```

Tests run:

```text
cargo test --manifest-path starkc/Cargo.toml --test as3_receiver_materialization -- --nocapture
  7 passed; 0 failed

cargo test --manifest-path starkc/Cargo.toml interp::tests:: -- --nocapture
  144 passed; 0 failed; 433 filtered out in src/lib.rs
  remaining integration/bin harnesses contained 0 matching tests and passed
```

Remaining limitation: repair commit SHA and CI run IDs are not available until this working-tree
change is committed and pushed.

## DEV-160

status: REPRODUCED, bounded by current backend refusal tests  
baseline SHA: `689d26d26990399d1de3026c13c271c403a45032`  
reproducer: `starkc/tests/dev160_call_site_thunk.rs` and
`packages/stark-http-client/src/lib.stark` workaround comment in `send`  
expected: supported disjoint accesses to one aggregate should agree across HIR, MIR and native;
unsupported cross-block absorption shapes should be refused by STARK backend diagnostics, not by
rustc `E0502` in generated code  
actual: in-block sibling borrow/move/read shapes are covered by call-site thunks; cross-block
borrow shapes are still refused by name (`DEV-160b` / `DEV-160d`)  
engine(s): HIR interpreter, MIR interpreter, native generated-Rust backend  
user impact: remaining limitation is native-backend completeness for cross-block borrow absorption,
not front-end over-acceptance and not an exposed rustc diagnostic  
repair packet: P2 / DEV-160, no further code change made in this pass

Tests run:

```text
cargo test --manifest-path starkc/Cargo.toml --test dev160_call_site_thunk -- --nocapture
  8 passed; 0 failed
```

Remaining limitation: this pass did not implement cross-block absorption. The current repository
already contains named backend refusals for those shapes, matching the programme rule that the
supported boundary must be enforced by STARK rather than delegated accidentally to rustc.

---

# Reproduction continued — the remaining eleven

Baseline SHA for every entry below: `689d26d26990399d1de3026c13c271c403a45032`.
Reproducers built with `starkc/target/debug/stark` at that SHA, run outside the repository tree.

## DEV-157

status: REPRODUCED, in the corrected shape the C10-Q pass recorded (not the shape the original
entry named)
reproducer:

```stark
fn main() { let x: Int32 = panic("p"); println(x); }
```

expected: a diverging call in initializer position is accepted and builds; `Never` coerces
actual:

```text
stark run   -> Error: runtime error: p          (correct — the trap fires)
stark build -> error: native build does not yet support this program:
               MirTy Never has no C5.3a generated-Rust representation yet
```

engine(s): HIR correct; native generated-Rust refuses. Accepted-but-unbuildable.
user impact: a program the front end and oracle accept cannot be built natively.
repair packet: P3. §9.1's position matrix is still to be built — this is one position
(initializer/local), and the entry also names argument position. Do not treat `Never` as one
homogeneous defect on the strength of this single probe.

## DEV-168

status: REPRODUCED, exactly as the entry describes
reproducer: a user `impl Display for P`, called in fully-qualified form:

```stark
let s: String = Display::fmt(&p);
```

expected: TYPE-METHOD-001 — "Trait methods can always be called in fully-qualified function form"
actual:

```text
stark run   -> P                                (HIR oracle runs it)
stark build -> error: native build does not yet support this program: callee form (C4.5)
```

engine(s): HIR correct; MIR lowering refuses.
user impact: the shape the spec offers as THE disambiguation mechanism for an ambiguous trait
method runs in one engine of three.
repair packet: P4.

## DEV-167

status: REPRODUCED, and refused at the front end rather than at a backend
reproducer:

```stark
fn show<T: Display>(value: &T) -> String { return value.to_string(); }
```

expected: contested — see below
actual: `[E0302] method 'to_string' not found for type 'T' — no trait in scope declares a method
named 'to_string'`
engine(s): all three, identically. This is a resolution refusal, not an engine disagreement.
user impact: ergonomic only. The workaround is real and shipped: `stark-fmt`'s
`to_string<T: Display>(value: &T) -> String` (`packages/stark-fmt/src/lib.stark:75`).
repair packet: P7 — **but the packet cannot decide this.** See the escalation below.

### DEV-167 is CE1, and is raised rather than resolved

`06-Standard-Library.md:817` declares `trait ToString { fn to_string(&self) -> String; }` and
`06-Standard-Library.md:446` gives `str::to_string`. **Neither promises that every `Display` type
has `to_string()`.** So there is no conformance gap to repair — the question is whether Core v1
should make that promise, which is a normative Core semantic change: **CE1**, Charter §2.3.

The three options and what each costs:

```text
A. blanket impl<T: Display> ToString for T
   Core v1 has no blanket implementations and no extension traits. Adding them is a language
   feature with coherence/overlap consequences far beyond this deviation.

B. resolver branch keyed on the method name "to_string"
   The entry's own objection, and it is correct: this reintroduces exactly the two-tier model
   DEV-166 removed (RESOLVED, DEV-DISPLAY-DISPATCH), trading a closed defect for ergonomics.

C. keep the free function; close DEV-167 as a documented non-promise
   Zero language change. stark-fmt already ships it. The spec never promised the method form.
```

**DECIDED (owner, CE1, 2026-08-10): option C — keep the free function.** DEV-167 is CLOSED as a
documented non-promise; see its closing heading in `KNOWN-DEVIATIONS.md`. Raised rather than
resolved by the packet, which is precisely what CE2 exists to prevent. The decision is pinned by
`dev_display_dispatch.rs::to_string_on_a_display_bound_is_refused_by_decision` and its `fmt()`
counterpart, so a later name-keyed resolver branch fails CI instead of quietly reversing an
owner decision.

## DEV-120

status: REPRODUCED, then CLOSED — RECLASSIFIED AS DOCUMENTED LIMIT
See the closing heading appended to `starkc/docs/conformance/KNOWN-DEVIATIONS.md` for the five
§14.1 answers, the measured interpreter/native split (exit 2 classified vs exit 134 SIGABRT), and
why owner ruling D4 already settled the repair question.
repair packet: P8 — complete. `MAX_CALL_DEPTH` unchanged; §14's warning observed.

## DEV-140 … DEV-145

status: ALL SIX REPRODUCE
reproducer: `cargo test --manifest-path starkc/Cargo.toml --test layer_audit`, which is an
enforcing gate since CD-342 — it fails on an unregistered finding AND on a registered one that
stops reproducing, so a green run is itself the reproduction evidence.

```text
TOTALS: 6 layer defects, 8 correctly refused up front, 6 lowered cleanly
  L7153  DEV-140  Vec::insert                       lowering: a later C4.5e sub-slice
  L8093  DEV-141  HashMap over user-Drop values     lowering: reserved — std-full
  L9130  DEV-142  droppable composite + borrow      lowering: needs generated lifetimes, C6.3e
  L5346  DEV-143  assert_eq on a user type          lowering: dispatch through its Eq impl
  L3698  DEV-144  for over ValuesIter               lowering: type Core(ValuesIter,[Int32]) (C4.5)
  L6450  DEV-145  to_uppercase on String            lowering: a later C4.5e sub-slice
```

engine(s): front end accepts, HIR oracle runs, MIR lowering refuses — the E0105 class, all six.
repair packet: P6. §12.2's grouping rule is not satisfied by any pair here on this evidence: the
six name four different missing authorities (C4.5e method sub-slice, std-full collections, C6.3e
generated lifetimes, Eq-impl dispatch). L7153 and L6450 are the only pair sharing a named slice
("a later C4.5e sub-slice") and even they differ in receiver type. **Do not batch.**

## DEV-159

status: NOT SETTLED BY THIS PASS — counted OPEN conservatively, unchanged from the C10-Q verdict
No reproduction was attempted here. §11.1 requires repeated cold builds of an HTTPS program with a
fresh target/cache, in debug and release, and a negative control that fails with the isolation
mechanism deliberately disabled. A single build — passing or failing — is not evidence about a
race. Left for P5 with its machine time budgeted honestly.
repair packet: P5.

## DEV-160

Covered in the section above. Reproduced; not repaired in this pass.

---

# P0 status

```text
REPRODUCED_OPEN = 9    (DEV-140..145 assessed+deferred, 159*, 160, 180 — *159 not probed)
NON_REPRODUCING = 0
RECLASSIFIED    = 1    (DEV-120 -> documented limit)
CLOSED BY CE1   = 1    (DEV-167 -> documented non-promise)
REPAIRED        = 3    (DEV-180 uncommitted; DEV-157 and DEV-168 this session)
NEWLY REGISTERED= 1    (DEV-220 — found by P3's position matrix, registered and repaired)
DORMANT         = 1    (DEV-179)
ACCEPTED        = 1    (DEV-011)
```

Population after P0/P3/P4/P6/P7/P8: **9**, computed by `c10-deviation-populations.py`, not
hand-edited. DEV-180 is still counted OPEN because its repair is uncommitted — the ledger closes
on a repair SHA, per §5.5, and there is not one yet.

**P3 found a defect the entry did not name.** Building §9.1's position matrix rather than trusting
the DEV-157 entry turned up an internal compiler error on `if c { x } else { return; }` —
registered as DEV-220 and repaired. The entry's own warning that it was "one probe away from being
filed as a false closure" was earned twice more: the matrix found two further defects, in two
phases the entry did not mention.

**P6's assessment came back against repair.** Not one of DEV-140..145's six shapes is used by any
first-party package, and the six name four different missing authorities. Recorded per deviation
in the ledger; deferred by owner decision, 2026-08-10.

**No deviation in this population failed to reproduce.** The programme's §6.2 path for stale
entries was not needed — the ledger is accurate at this baseline.
