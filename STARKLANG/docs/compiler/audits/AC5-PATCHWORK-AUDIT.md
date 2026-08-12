# AC5 — patchwork / special-case audit

**Packet:** WP-ARCH-CLOSE AC5, under CD-400. **Status: IN PROGRESS — not complete, and the closure
criterion is not met.** Categories covered and not covered are listed in §5 so a reader can see the
boundary rather than infer it.

**Tree:** `develop` at `8ebdaa6`. **Surface:** `starkc/src`, 106 files, 109,401 lines.

---

## 1. What this audit is looking for

AC5's search list, and the classification every finding receives:

```text
A — legitimate language/runtime special case
B — deliberate independent verifier implementation
C — architecture debt, safe and explicitly tracked
D — patchwork / semantic authority violation      <- BLOCKS CLOSURE
```

The objective is **not** zero special cases. It is zero *unclassified* semantic exceptions, zero
known symptom patches, and zero *accidental* duplicate authorities.

---

## 2. Mechanical sweeps, and what they returned

### 2.1 Residue markers

```text
TODO / FIXME / HACK / XXX          0 occurrences across 109,401 lines
"temporary"                       71, ALL false positives -- MIR temporaries and temp files
"for now"                          6, each naming a scope decision with its ruling
"workaround"                       3
```

**Zero TODO/FIXME/HACK is a real result, not an empty search.** The pattern was verified to match by
running it against text that contains those words elsewhere in the tree.

The six `for now` sites are scope markers (`provider_synth.rs` free functions only, CD-225 §7.1;
`provider_bind.rs` Core `File` pre-A11, CD-235; `lower.rs` droppable payload elaboration; and two in
`emit_call_thunk.rs`). Each cites a decision. **Classified A** — a stated scope boundary with a
ruling behind it is not residue.

### 2.2 Name-based semantic dispatch

`match <stringified path>` appears in 20 places. Triage:

```text
CLI / LSP argument and method parsing      16   NOT semantic -- protocol surface. Class A
resolve.rs:1033                             1   the ~26 hardcoded builtin spellings, DEV-229's
                                                own stated residual. Class C, owned, see §3.2
deploy/lower.rs                             2   tensor extension, a DEFERRED research track
provider_manifest.rs:219                    1   manifest form parsing, not language semantics
```

### 2.3 Spelling-versus-identity in trait handling

`typecheck/traits.rs` compares `bound_name` against `"Copy" | "Eq" | "Ord" | "Clone" | "Hash" |
"Display" | "Num" | "Default" | "Iterator"` in ~34 places, while the same repo settled — twice —
that trait identity must come from `Res::CoreTrait(..)` and never from spelling (CD-379 for
`Display`; DEV-210, where the borrow checker asked `.ends_with("Drop")` and refused a legal partial
move on any type implementing a user trait `MyDrop`).

**A hypothesis was formed here and then FALSIFIED, which is why it is not filed as a finding.**
`bound_impl_witness` accepts an impl when `Some(trait_ref.res) != bound_res && text(..) != bound_name`
— a spelling fallback that looked capable of admitting a user trait as a witness for a core bound.
Removing the fallback entirely and re-running the probes changed **nothing**, so it is not the cause
of the behaviour in §3.1 and no finding rests on it. Recorded because an audit that reports only its
confirmed hypotheses is not reporting its method.

---

## 3. Findings

### 3.1 AC5-F1 — `println` does not enforce its own `T: Display` bound — **NOW DEV-236**

**RECLASSIFIED 2026-08-12. Filed as `DEV-236`, a CONFORMANCE DEFECT, not architecture debt.**
Test: `starkc/tests/ac5_display_entry_points.rs` (3 cases).

> **The original classification here was wrong, and the correction is worth stating rather than
> overwriting.** This was filed as *Class C* on the reasoning that *"either policy is defensible;
> applying one to each entry point is not"* — that the divergence was a policy choice needing an
> owner. **It was not a choice.** `PRINT-DISPLAY-001` already fixes the policy (the print family are
> *"implementation-provided generic functions … **not** syntax hooks"*), and `TYPE-METHOD-003`
> already fixes identity-over-spelling (*"a `TYPE-NOMINAL-001` item identity **and not a
> spelling**"*). The current behaviour violates both.
>
> **The audit reached for a class before checking the normative text.** A/B/C/D classify
> architectural residue; a rule the specification already decides is not residue, it is
> non-conformance, and it belongs in the deviation ledger with a DEV number. Both citations were
> verified verbatim in `06-Standard-Library.md` and `03-Type-System.md` before this correction was
> made.

`Display` has two entry points, and the repo names them as two — `typecheck/body.rs` calls
interpolation *"the SECOND `Display` entry point"* (AS3 Boundary 4). They apply different policies to
the same rule:

```text
f"{x}"      on `fn show<T>(x: T)`      REFUSED at the definition, E0306
println(x)  on `fn show<T>(x: T)`      ACCEPTED; obligation deferred to the instantiation
```

Measured across five shapes:

```text
println(x), no bound at all                  front end ACCEPTS
println(x), T: Clone (unrelated bound)       front end ACCEPTS
println(x), T: Display (correct)             front end ACCEPTS
println(x), T: <user trait named Display>    front end ACCEPTS
f"{x}",     T: <user trait named Display>    REFUSED, E0306
```

Either policy is defensible on its own. **Applying one to each entry point is not**, and it has a
visible consequence:

```text
trait Display { fn unrelated(&self) -> Int32; }     // a USER trait, correctly resolved
impl Display for P { ... }
fn show<T: Display>(x: T) { println(x); }
fn main() { show(P { a: 1 }); }

front end   ACCEPTS
MIR         REFUSES -- "Display::fmt not found for printed type"
```

That is the **accepted-but-unbuildable** shape this repo tracks as the E0105 class and audits in
`layer_audit.rs`. The program is valid by the front end's account and cannot be built.

**Why not Class D, having established it is non-conformance.** D is *"patchwork / semantic authority
violation"*. The bounds here are present, resolved, and carry their identities — the obligation
checker is simply not consulted for this callee. **One authority, not consulted, is not a bypassed
authority**, and the same reasoning is why DEV-236's triage records `Architecture trigger: NONE`
rather than AC7-D. If a repair attempt shows the authority *cannot* express the obligation, both
classifications are revisited on that evidence.

It fails **safe** — a compile-time refusal, never wrong code — but it is not benign: the refusal
reaches the user from the wrong layer, which is the E0105 class this programme exists to remove.

**RULED 2026-08-12 under CE1 (CD-401): enforce at the generic DEFINITION; interpolation is not
weakened.** The normative text substantially settles it — `PRINT-DISPLAY-001` defines the print
family as ordinary generic functions constrained `T: Display`, **not syntax hooks**, and
`TYPE-METHOD-003` says a parameter's capabilities come from its declared bounds *and their resolved
identities*. So `fn show<T>(x: T) { println(x); }` and the `T: Clone` variant are both rejected where
they are written, and a user trait spelled `Display` satisfies the bound only if it resolves to the
Core `Display` identity.

**The repair must not be a `println` special case.** It belongs in the authority that checks generic
callee obligations; if `println` bypasses ordinary bound checking because it is a builtin, it is
routed through the existing mechanism rather than gaining another `if callee == println`.

> **The repair is itself an architecture test.** If this obligation cannot be expressed through the
> existing generic-call/bound authority, that is a **more serious finding than F1** — §4's *"consumer
> patched because the owning authority cannot express the rule"* — and is recorded as such rather
> than worked around.

**Blast radius, measured: zero.** No first-party generic function prints — every `println` under
`packages/` is on a concrete type. The repair rejects programs that build today, and none of them
are in this tree. It will reject `fn show<T>(x: T) { println(x); }`, which a newcomer writes early;
`E0306` already carries the right remedy and should be reused rather than a new code minted.

**REPAIRED 2026-08-12, after AC3's two runs completed and released the freeze.** DEV-236 is
RESOLVED: `type_is_displayable`'s `Ty::Param` arm now asks `param_declares_bound` with
`Res::CoreTrait(CoreTrait::Display)` as the required identity. **The architecture test in CD-401's
Decision 2 passes** — the obligation was expressible at the existing bound authority, with no
`println` special case. Had it not been, that would have been a finding more serious than F1 itself.

The repair also exposed a second defect: making a Pass-3 obligation scope-sensitive without carrying
its scope, which refused a bound plainly written. `DeferredDisplayPlan`'s own doc comment had
already stated the general rule, and `display_checks` now obeys it.

### 3.2 AC5-F2 — the ~26 hardcoded builtin spellings

**Class C.** Owner: compiler track. Already stated as a residual by DEV-229's own entry:

> *"The twenty-six spellings remain a string-matched table rather than entries in the value and type
> namespaces. Making them ordinary namespace entries a user declaration shadows by the normal rule
> would delete the special case entirely; the fallback keeps it, in a position where it can no
> longer pre-empt."*

Re-verified at this tree: the table is still in `resolve.rs`'s fallback position, after the namespace
walk rather than before it, which is what DEV-229 moved it to. **Classified, not re-litigated** —
AC5's job is to ensure it is owned, and it is.

### 3.3 AC5-F3 — a refusal message that outlived its cause

**Class C, REPAIRED in this audit.** `emit_call_thunk.rs`'s DEV-160b refusal still told the user the
thunk *"can only take over evaluation within the call's OWN block"* and that the case was *"deferred
to its own work package"*. Both became false when cross-block absorption landed hours earlier. The
refusal is still reachable — absorption declines shapes that fail its admission conditions — so the
message now states which conditions were not met instead of describing a mechanism that exists.

A diagnostic that misdescribes its own cause sends the reader looking for the wrong thing, and this
one would have sent them to a closed work package.

### 3.4 The known duplicate authorities, re-confirmed as deliberate

**Class B.** `AS8-DA-002/003/004` — `mir::interp::is_vec_runtime` versus
`mir::verify::is_vec_runtime_fn`, and the Box and Slice equivalents. Owner ruling 2026-08-09:
**REMAIN SEPARATE**, because `verify.rs` exists to CHECK what `interp.rs` executes and an independent
table is what lets it disagree. Both copies were mutation-tested and **both killed**.

Re-confirmed present and unchanged at this tree. **This is the model outcome for a duplicate**: two
implementations, a stated reason, and a control proving each is load-bearing.

---

## 4. Class-D findings

```text
NONE SO FAR.
```

This is a statement about the categories in §5's "covered" list, not about the whole surface.

**Findings by disposition, after the §3.1 correction:**

```text
DEV-236   conformance defect, left the A/B/C/D scheme entirely   (was filed here as Class C)
AC5-F2    Class C, owned by DEV-229's stated residual
AC5-F3    Class C, repaired in this audit
AS8-DA-*  Class B, re-confirmed deliberate, both copies mutation-killed
```

**One finding has now moved out of this audit's scheme on inspection**, which is worth watching for
in the unswept categories: the A/B/C/D classes describe architectural residue, and a behaviour the
specification already decides is non-conformance instead. The question to ask of each remaining
finding is *does a normative rule already settle this?* before reaching for a class.

---

## 5. Coverage — what this audit has and has not examined

```text
COVERED
  TODO / FIXME / workaround / temporary markers        whole tree, mechanically
  name-based semantic dispatch (match on a spelling)   whole tree, mechanically, then triaged
  spelling-vs-identity in trait/bound handling         typecheck/traits.rs, probed empirically
  the Display obligation's two entry points            probed across five shapes
  known duplicate classifiers (AS8-DA-*)               re-confirmed

NOT YET COVERED
  engine-local reconstruction of type/generic information
  backend-specific acceptance rules beyond emit_call_thunk
  precedence exceptions (DEV-228 removed the resolver's; others not swept)
  copy/paste semantic tables outside the AS8-DA set
  consumer/package workarounds for compiler limitations (packages/ not swept)
  special handling keyed to individual builtins beyond resolve.rs:1033
```

**AC5 is not complete and must not be reported as complete.** The uncovered categories are where a
Class-D finding is most likely to live — `engine-local reconstruction` in particular, since that is
the category AC1 step 1 found a real instance of (the borrow-origin analysis living in the native
emitter) before it was moved to MIR.
