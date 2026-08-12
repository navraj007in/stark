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

### 3.5 Engine-local reconstruction — swept 2026-08-12, **no reconstruction found**

The category AC1 found a real instance of (the borrow-origin analysis living in the native emitter),
and therefore the one most likely to hold a Class-D. **It does not, on this sweep.**

```text
mir/interp.rs        1 reference to HIR or the typecheck tables, in 2,878 lines
interp.rs          288 -- expected and not reconstruction: reading HIR IS the HIR oracle's job
backend/*           29 -- ALL of them `hir::ItemId` / `hir::CoreType` IDENTITY payloads that
                        `MirTy` embeds. The backend matches on MIR variants; it performs no HIR
                        lookup and re-derives no semantic answer
```

The MIR interpreter's single HIR reference is the strongest evidence here: an engine that had to
reconstruct type or generic information could not do it from one.

### 3.6 AC5-F4 — duplicated primitive classifiers with no stated rationale

**Class C.** Owner: compiler track. Found by scanning all 3,716 `fn` definitions across 106 files
for one name defined in more than one module — 214 such names, 8 semantic-looking.

```text
is_signed_int    mir/interp.rs:2862   backend/generated_rust/emit_bodies.rs:2115
                 BYTE-IDENTICAL: matches!(ty, Int8 | Int16 | Int32 | Int64)
is_integer       interp.rs:9469       typecheck/types.rs:957
                 same domain (Primitive), same predicate, two implementations
                 (mir/verify.rs:2824 states the same question over MirTy -- a different IR)
is_numeric       typecheck/types.rs:913 (Primitive)   mir/verify.rs:2838 (MirTy)
```

**Why C and not B.** AS8-DA-002/003/004 are duplicates *with a stated reason* — `verify.rs` exists to
check what `interp.rs` executes, and an independent table is what lets it disagree; both copies were
mutation-tested and both killed. **These carry no such statement.** They read as copy-paste, and
nothing enforces their agreement.

**Why C and not D.** The sets are closed `MirTy`/`Primitive` enums, and a divergence in signedness
or integer-ness changes trap behaviour, which the three-engine differential would catch. Safe, but
unowned.

**Disposition.** Either give each a stated reason (promoting it to B, as AS8 did) or collapse it to
one authority. Not repaired here: `is_signed_int` is used by two engines whose independence may be
deliberate and undocumented, and deciding that is an owner call, not an audit call.

### 3.7 AC5-F5 — one selection predicate written twice, kept in sync by a comment

**Class C.** `interp.rs`'s `find_drop` (9261) and `drop_impl_is_generic` (9243) share the identical
`Drop`-impl selection:

```rust
reference.res == Res::CoreTrait(hir::CoreTrait::Drop)
    && matches!(&self.hir.ty(*self_ty).kind,
        hir::TypeKind::Path { res: Res::Item(actual), .. } if *actual == item)
```

The second's doc comment states the risk itself: *"Matched the same way `find_drop` matches, so the
two cannot disagree about which impl is in question — a check that looked at a different impl than
the one about to run would be worse than no check."*

**Nothing enforces that.** This repository's own history is the argument: DEV-162 shipped an `E0425`
because an emitter named a helper the collector never generated, and the fix was *"a shared structure
replaces the comment that used to say 'the two must agree'"*. This is the same shape, one comment
earlier. The repair is a small extraction — one function returning the selected impl, both callers
reading it — and is deliberately not taken inside the HIR oracle without an owner's word.

### 3.8 Two candidates that are NOT findings, on inspection

Recorded because an audit that lists only its hits is not reporting its method.

```text
has_user_destructor   drop_rule.rs declares a TRAIT; lower.rs and verify.rs implement it.
                      Two implementations by design -- the AS8-DA pattern, Class B.
                      Observation, not a defect: lowering's answer is argument-INDEPENDENT
                      (`_args`, justified by A1) while verification's is keyed by
                      `(item, args)`. They agree only because A1 holds; if a `Drop` impl
                      could ever apply to some instantiations and not others, they would not
is_copy               mir/lower.rs and mir/mod.rs are two thin WRAPPERS over one authority,
                      `mir_ty_is_copy`, whose doc says "The one structural Copy rule.
                      There is no second copy of this match." They differ only in how each
                      supplies the copy-eligible set. Class A
```

### 3.9 AC5-F6 — `packages/` swept; one stale workaround found and REMOVED

**Class C, repaired.** Denominators: **102 `.stark` files, 27,933 lines, 110 marker hits.**

**Twelve distinct DEVs are cited in first-party STARK source, and every one is CLOSED or RESOLVED.**
That sounds like twelve stale workarounds. It is not — and the distinction is the finding:

```text
HISTORICAL NOTES, correctly updated when the defect was fixed          11 of 12
    "unit aborted natively (DEV-158). That is fixed, so this reads
     from `default_config()` again"
    "DEV-165 REPAIRED (2026-08-10). This assertion used to require FAILURE"
    "until DEV-148 was fixed it could not be"
  -> someone went back and updated the code AND the comment. Not residue. Class A

A LIVE WORKAROUND whose stated expiry condition had just been met           1
    packages/stark-http-client/src/lib.stark, `send`
```

The one exception named its own expiry:

> *"CD-374 fixed the in-block form with a generated call thunk. This one is NOT in-block: `as_str()`
> is itself a call, so the `&str` it returns arrives from an earlier block and the thunk cannot take
> over its evaluation. … **so this stays until cross-block absorption lands**."*

Cross-block absorption landed the same day (DEV-160 RESOLVED). The four locals were removed and the
inline form restored:

```stark
let once = send_once(client, builder.method, builder.url.as_str(),
                     builder.headers, builder.body)?;
```

**This function is where DEV-160 was reported from.** Restoring it converts the repair's evidence
from *"the reproducer works"* to *"the motivating consumer's workaround is gone"*, which is a
materially stronger claim and the reason this was repaired rather than merely filed.

```text
stark-get build                       OK -- the application that consumes the client
33 first-party applications           built, 0 failures
qualify-first-party-packages.py       EXIT 0: 31 test targets, 1,222 tests, 0 failed,
                                      ending with a live TLS 1.3/1.2 session verified, used and
                                      closed, and an untrusted root rejected
```

**Not every local binding is a workaround, and one was deliberately left alone.** `follow`, in the
same file, also binds the fields to locals — but it is a redirect *loop* that reassigns them, so the
bindings are load-bearing. The original comment even says `follow` "already had this shape, which is
why only this path failed". Removing them there would have been a change dressed as a cleanup.

**Not counted as evidence:** `stark check --target-native` passing on the client. It scans for
unsupported runtime functions and never reaches `plan_for_call`, so it cannot see this change at
all — the same trap AC1 fell into and withdrew.

**A note on `DEV-156`'s six citations.** It owns two ledger headings, OPEN (backfilled) then CLOSED;
the last heading decides, so it is closed. Those six comments explain why a doc comment sits where it
does, are cosmetic, and are left as written.

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
  the Display obligation's two entry points            probed across five shapes -> DEV-236
  known duplicate classifiers (AS8-DA-*)               re-confirmed
  engine-local reconstruction of type/generic info     swept 2026-08-12, NONE FOUND (§3.5)
  copy/paste semantic tables outside the AS8-DA set    all 3,716 fns scanned -> F4, F5 (§3.6-3.7)

  consumer/package workarounds for compiler limitations  102 .stark files, 27,933 lines,
                                                        12 DEVs cited -> F6 (§3.9)

NOT YET COVERED
  backend-specific acceptance rules beyond emit_call_thunk
  precedence exceptions (DEV-228 removed the resolver's; others not swept)
  special handling keyed to individual builtins beyond resolve.rs:1033
```

**AC5 is not complete and must not be reported as complete.** Three categories remain.

**The category that was most likely to hold a Class-D has now been swept and holds none** (§3.5),
and so has `packages/` (§3.9). That materially lowers, without eliminating, the chance that AC5 ends
in FAIL-ARCHITECTURE — the remaining three are narrower surfaces inside the compiler itself.

**A method note, recorded because it nearly produced a false clean result.** The first duplicate-
classifier scan reported *zero* duplicates across the whole tree. The scan was broken — it resolved
a relative path against a working directory already inside `src`, found **zero files**, and reported
a clean sweep. It was caught by disbelieving the result, not by the harness. Every mechanical sweep
in this audit now prints its denominator (files scanned, definitions found) so an empty search cannot
masquerade as a clean one.
