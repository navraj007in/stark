# STARK Compiler STATE
Updated: 2026-07-21 — **Gate C4 CLOSED, Gate C5 OPEN, WP-C5.1 CLOSED IN FULL
(CD-042/043/044/045/046), WP-C5.2 CLOSED IN FULL (CD-047/048/049/050/051 for C5.2a-e; CD-053 for
the three-engine differential harness that satisfies §14's exit condition).**

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

## Position
**WP-C5.3 (aggregates, enums, error values, Drop, layout) CLOSED 2026-07-23** by owner directive
after the adversarial review dispositions (CD-070). Sub-packages: C5.3a (CD-056), C5.3b, C5.3c
(CD-061), C5.3d-0 (CD-059), C5.3d-1a (CD-063), C5.3d-1b (CD-064), C5.3d-1c + C5.3d-1 (CD-066), the
`Copy` consolidation fold-in (CD-065), C5.3e (CD-067) with DEV-100 fixed (CD-068) and the corpus
re-pinned to 1.3.0 (CD-069). Every §14 exit dimension is discharged with three-engine agreement:
aggregate values, payload variants, match paths, `Option`/`Result`, `?`, the dedicated Drop
fixture (seven observable properties), and exact layout-query values under the versioned
`stark-64-v1` contract. Two bounded boundaries are recorded and enforced deterministically before
rustc rather than left latent: multi-unit enum payload partial moves (CD-070) and the wider
non-`Copy` cross-block cases, both deferred to C6. **Next: WP-C5.4 (linkage and function values).**
The two open C5.3-adjacent items carried into the C5.4/C5.6 reviews are DEV-098's defensive
reborrow reasoning and the C6-deferred ownership boundaries.

Gate: **C5 (native compilation) — OPEN. WP-C5.1 CLOSED 2026-07-21 in full** (entry plan CD-042,
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

## Repository baseline
- Last completed transition: WP-C2.13 (Gate C2 exit and Core v1 semantic freeze). Verdict
  **CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS** — all 24 high-cost open
  questions (CORE-Q-001..024) approved, 166-row completeness inventory has zero
  absent/contradictory/unclassified rows (6 remain `pending-owner-approval` governance
  bookkeeping only, behavior already implemented/tested), 33 deviations closed this gate
  (seventeen WP-C2.2 runtime-semantics defects, six WP-C2.11 items, DEV-036, seven
  post-WP-C2.11 correction-pass items, DEV-053/054), 8 remained open and non-soundness-relevant
  at gate close (current open set after the post-Gate-C2 correction brief: DEV-005/010/011/012,
  DEV-017 partial, DEV-060 — see the open index below).
  Full report: `starkc/docs/compiler/C2-exit-report.md`. C3-entry is the active transition
  before WP-C3.1.
- Transition base commit: `c268d7c` (`Add systems ecosystem roadmap`), after the post-Gate-C2
  correction-brief commit that resolved DEV-051, DEV-052, and DEV-055 and opened DEV-060.
- Amendment base commit: `60b49e2` (`CD-021 function-value native validation...`) — the head
  this state revision was written against. (Field renamed from "Current committed head" under
  CD-022: a commit cannot record its own SHA, so that framing was permanently one behind;
  the live head is always `git log`, never this file.) Commit only on explicit user request.
- Rust toolchain: `starkc/rust-toolchain.toml` pins `channel = "stable"` (no version number, tracks
  stable) with `rustfmt`/`clippy` components. Active environment measured: `cargo 1.93.0
  (083ac5135 2025-12-15)`, `rustc 1.93.0 (254b59607 2026-01-19)`. `starkc/Cargo.toml` declares
  `rust-version = "1.85"` (crate MSRV). The Gate-5 *generated deployment host* (not `starkc`
  itself) separately requires Rust 1.88 due to the `ort` crate's MSRV
  (`starkc/docs/gate5-backend-decision.md:107-110`) — this does not raise `starkc`'s MSRV.
- Latest verified code baseline: `cargo test --workspace --all-targets --all-features`
  (starkc/, post-CD-025, 2026-07-19):
  **597 passed, 0 failed, 2 ignored** (594 → 596 from DEV-060's fix: one new typecheck
  regression test, one new interp execution test, one existing test rewritten in place; 596 →
  597 from CD-025's `corpus_lock_matches_frozen_snapshot` integrity test)
  across **4 unittest binaries** (`src/lib.rs`,
  `src/main.rs`, `src/bin/stark.rs`, `src/bin/starkide.rs`) **+ 32 integration-test files**
  (`find starkc/tests -maxdepth 1 -type f -name '*.rs' | wc -l`,
  re-counted against the
  post-WP-C2.7 tree — the
  "3 unittest binaries + 31/32 files" figure quoted in several prior session records below was
  never actually verified against `ls`/`cargo test`'s own "Running ..." lines and had drifted;
  not chasing down exactly which prior WP's arithmetic first went wrong, since that would need
  checking out old commits for no real benefit — this line is now the corrected, directly-counted
  baseline going forward). Up from 383/0/2 at Gate C0 close (file count at that point not
  re-verified for the same reason). WP-C1.1 added `span_integrity.rs` + 12 tests, WP-C1.2 added
  15 more across `resolve.rs`'s inline tests and `gate2_package.rs`, WP-C1.3 added 8 more across
  `typecheck.rs`'s and `interp.rs`'s inline test modules, WP-C1.4 added 11 more across
  `gate2_valid.rs` and `gate3_execution.rs`, WP-C1.5 added 21 more to `gate2_valid.rs`, WP-C1.6
  added `conformance_report.rs` (new file) + 4 tests.
  Both ignored tests are
  intentionally opt-in (a checksum-pinned live ONNX artifact test in `tests/gate4_onnx.rs`, and
  a live-ORT-download inference test in `tests/gate5_codegen.rs`). Full per-file breakdown
  recorded in `starkc/docs/dev/compiler-map.md` (WP-C0.1; not re-regenerated for the WP-C1.1/
  C1.2/C1.3 deltas — see that file's own scope note).
  Latest recorded validation also has `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and conformance
  validation/reporting clean.
- Core spec revision: `STARKLANG/docs/spec/` files 00-07 plus
  `CORE-V1-ABSTRACT-MACHINE.md` and `CORE-V1-FUTURE-BOUNDARIES.md`, normative per
  `CLAUDE.md`. Spec fixture corpus:
  `STARKLANG/tests/spec-fixtures/manifest.toml`, 113 entries (parse-pass 65,
  semantic-error 16, notation 27, lex-pass 4, parse-fail 1). WP-C2.7 removed 28 stale,
  duplicative memory-model examples and now contains 13 abstract-machine adversarial examples
  after its correction pass. WP-C2.8 appended five static-semantics review fixtures without
  renumbering existing examples.
- Tensor spec revision: `STARKLANG/docs/extensions/Tensor-Model-Types.md` (extension `tensor`
  v0.1), `AI-Extensions.md` (non-normative sketches).
- Conformance DB: `STARKLANG/conformance/core-v1-coverage.toml`, 59 `[[rule]]` entries.
  **Integrity-audited under WP-C0.3 (2026-07-17)**: no duplicate rule IDs, no references to
  nonexistent spec chapters (both now mechanically checked, see `starkc/scripts/
  check-conformance.py`). Post-correction counts: 53 implemented, 6 partial, 0 missing.
  Pre-correction counts (53 implemented, 2 partial, 4 missing) were **stale**, not accurate — see
  DEV-002. `starkc/scripts/check-conformance.py` now also warns (non-fatal) on `missing` entries
  that still carry a `source`/`tests` field and on likely-semantic-rejection rules with zero
  recorded tests, as a heuristic staleness signal for future audits. Known representational gap:
  the schema's single `tests` array does not distinguish positive from negative test evidence, so
  Charter rule 15 ("positive and negative evidence travel together") cannot be mechanically
  verified from this database alone for every rule. **WP-C1.6** (closed 2026-07-18) addressed
  this with a richer schema (`positive_tests`/`negative_tests`, function-level `path::function`
  citations) and populated it for 20 of 59 rules with real evidence; the remaining 39 still rely
  on the single aggregate `tests` citation and are reported as "unclassified" by the new
  `generate-conformance-report.py`, not silently treated as verified — see DEV-017.
  **Coverage percentages remain provisional**: "implemented" status
  for any individual rule is not re-verified at Core v1 rule-completeness depth until WP-C1.x; see
  governing rule in `COMPILER-CHARTER.md` §1.5 rule 14 and the explicit no-percentage-trust
  statement this state file and the WP-C0.5 exit report both carry.
  WP-C2.6 adds `STARKLANG/conformance/core-v1-rule-id-map.toml`, a mechanically validated
  transition from every one of those 59 broad IDs to the stable granular inventory IDs. It does
  not inherit broad implementation status; C2.11 must classify evidence and status per granular
  rule.

## Current compiler pipeline
- Source -> lexer (`lexer.rs`) -> parser (`parser.rs`) -> AST (`ast.rs`) -> resolve (`resolve.rs`)
  -> HIR (`hir.rs`) -> type/flow/borrow check (`typecheck.rs`, `flow.rs`, `borrowck.rs`) ->
  interpreter (`interp.rs`).
- Extension front end: `extensions/tensor/` (dim algebra, tensor/model types), gated by
  `options.rs` (`LanguageOptions`/`ExtensionSet`).
- Artifact path: `onnx/` (bounded ONNX signature import/verify, no graph execution) ->
  `deploy/` (Gate-5 lowering to a generated Rust host calling ONNX Runtime via the `ort` crate).
- Additional entry points (three separate binaries, non-overlapping command sets — see
  `starkc/docs/dev/compiler-map.md` for full detail):
  - `starkc` (`main.rs`): `check`, `run`, `parse`, `lex`, `lsp`, `import`, `verify`, `deploy`.
  - `stark` (`bin/stark.rs`): `check`, `build`, `run`, `test`, `fmt`, `doc`.
  - `starkide` (`bin/starkide.rs`): interactive terminal IDE, no CLI subcommands.
  - `lsp/` module backs `starkc lsp`; `formatter/` backs `stark fmt`; `doc_gen/` backs
    `stark doc`; `test_runner/` backs `stark test`.
- **Known duplication requiring WP-C0.1 tracing**: `starkc` and `stark` each implement their own
  `check`/`run`, and neither binary exposes the full command surface — a caller needing
  `deploy`/`verify`/`import`/`lsp` together with `build`/`test`/`fmt`/`doc` must invoke both
  binaries. Whether these two `check`/`run` implementations share one pipeline or have drifted is
  unverified; resolve in WP-C0.1 (this is exactly the "shared vs. duplicated entry points"
  question that WP is scoped to answer, and directly bears on Charter rule 18 — cross-tool
  convergence).

## Decision log — append-only
- CD-001 [WP-C0.0] Adopted the "C0-C10" gate numbering from
  `STARKLANG/docs/STARK-Compiler-Build-Brief-Revised-Sonnet.md` as a **new, independent**
  sequence, not a renumbering of the repo's pre-existing (non-prefixed) Gate 1-7 track. The two
  numbering systems now coexist; `COMPILER-ROADMAP.md` carries a note at its top explaining the
  relationship. Rationale: the brief's own gate definitions (front end conformance closure,
  reference execution contract, compiled-language decision spike, MIR, native backend, language
  services, extension isolation, release qualification) do not map one-to-one onto the old
  gates, which were scoped around a single tensor/ONNX vertical-slice demonstrator rather than
  general Core conformance. Renumbering the old track retroactively would rewrite closed
  historical evidence, which Charter §1.5 rule 2 and WP-C0.2 ("do not rewrite historical gate
  evidence to match later implementation") forbid.
- CD-002 [WP-C0.0] Recorded that the strategic question Gate C3 (Compiled-Language Decision
  Spike) exists to answer has **already been examined once**, under the old gate track, and
  closed with a non-GO outcome:
  - `starkc/docs/gate6-memo.md`: Decision **REVISE** (owner-confirmed 2026-07-16) — comparator
    evidence was 5/5 vs 2/5 defects caught pre-inference against Python/ORT baseline, and parity
    (5/5 vs 5/5) against "the strongest typed-Rust host" comparator; recommendation was to
    re-scope the demonstrator, not GO or STOP outright.
  - `starkc/docs/gate7-decision.md`: Decision **RETAIN AS RESEARCH LANGUAGE** (owner-confirmed
    2026-07-16), tensor-track technical verdict POSITIVE, tensor productisation verdict DEFER,
    language thesis UNRESOLVED. Explicitly authorizes only a `stark verify` external-validation
    track as next work and states "No LSP work or language expansion is authorized" (superseded
    for LSP specifically by the subsequent WP8.1-8.5 work, all committed after gate7-decision.md
    per `git log`; that expansion was evidently owner-authorized outside this decision doc's
    text, but the state file flags the textual contradiction for WP-C0.2 to reconcile formally).
  - Disposition: Gate C3 must treat gate6-memo.md/gate7-decision.md as **directly relevant prior
    evidence about interpreter-vs-native tradeoffs**, not reopen the question from zero. This is
    scoped as a C3-entry consideration, not a C0 decision — C0 does not skip ahead of C1/C2. Set
    `Conditional tracks: Native=deferred` above to reflect that the most recent owner decision on
    a related (ONNX-vertical) native-deployment question was non-GO; C3 will need fresh evidence
    for the *general* Core compilation question, which the old gates never tested (old Gate 5's
    "native" path is code generation to a *generated Rust host*, not general Core-to-native
    compilation — it has no bearing on scalar/loop/struct/enum native lowering that C3-C7 would
    need to evaluate).
- CD-003 [WP-C0.0] Confirmed two stale root-adjacent status documents exist and require
  correction under WP-C0.2 (not fixed in this WP — C0.0 is bootstrap-only, per its own "Done
  when" — but recorded now so the fix isn't lost):
  - `CLAUDE.md:110-113,137` states "Gates 1-3 are closed... next: Gate 4" — contradicted by
    `starkc/docs/gate4-exit.md` through `gate7-decision.md`, all closed, and by the root
    `README.md`'s own delivery-gates table which correctly lists all seven gates as
    Complete/Decision-recorded.
  - `starkc/README.md:4` states "Gate 4 (tensor front end and ONNX signatures) is complete" with
    no mention of Gates 5-7, and its module "Layout" table omits `deploy/`, `lsp/`, `formatter/`,
    `doc_gen/`, `test_runner/` — five of the crate's fifteen `pub mod`s are undocumented there.
  - `STARKLANG/docs/PLAN.md:5` says "The roadmap defines what evidence advances the project
    (Gates 1-6)" and has no Gate 7 section, while `STARKLANG/docs/ROADMAP.md` has a full,
    evidence-cited Gate 7 section matching `gate7-decision.md` exactly. PLAN.md was last
    substantively updated for Gates 1-5.
  - By contrast, root `README.md` is internally consistent with all seven gate exit/decision
    docs and is the most reliable of the pre-existing status documents.
- CD-004 [2026-07-17, outside any single WP — a mid-session governance update triggered by a new
  source document] The user provided a revised master brief,
  `STARKLANG/docs/STARK-Compiler-Build-Brief-Revised-Sonnet(1).md` (title: "... (Native Compiler
  Required)"), which supersedes the original `STARK-Compiler-Build-Brief-Revised-Sonnet.md` this
  track was bootstrapped from (WP-C0.0). **This is a real, deliberate scope change, not a
  clarification**: the original brief framed Gate C3 as an open, evidence-based question — GO,
  REVISE, DEFER, or STOP on whether STARK needs a general native Core compiler at all, explicitly
  naming DEFER/STOP as valid, non-failure outcomes. The revised brief removes that question
  entirely: general native Core compilation is now a **mandatory** completion requirement (new
  §1.2 "Guaranteed compiler completion state" in `COMPILER-CHARTER.md`), Gate C3 is renamed
  "Native Compiler Architecture and Backend Selection Spike" and now only selects *how* (backend
  strategy: SELECT-GENERATED / SELECT-DIRECT / REVISE / BLOCKED), never *whether*. An
  interpreter-only release is explicitly "not an allowed C3 completion outcome," and Gates
  C4-C7 change from *conditional* on a GO decision to *mandatory* after C3 selects an
  architecture. Diff confirmed Gates C0-C2 and C6/C8/C9 are textually unchanged; the change is
  scoped to §1 (framing/rules), the `COMPILER-STATE.md` template in §2.4, Gate C3's outcome
  vocabulary, Gate C4/C5's conditionality headers, Gate C10's release-statement requirements,
  §4's dependency map (native path folded into the single mandatory path, no more separate
  "native compiler path" branch), §5.3's gate-decision vocabulary (adds `BLOCKED`), §7's session
  budget (single ~57-86 session mandatory-path figure, replacing the old bifurcated
  "interpreter-only 31-48 / full-native 58-88" framing), and §8's strategic-outcome list.
  Regenerated `COMPILER-CHARTER.md` and `COMPILER-ROADMAP.md` in full from the new brief text
  (same extraction method as WP-C0.0) rather than hand-patching, to guarantee fidelity; updated
  this file's Position-line schema (`Mandatory compiler path: Core=/MIR=/Native=` +
  `Optional tracks: ArtifactInfra=/TensorExpansion=`, replacing the old `Conditional tracks:`
  line) and renamed the `## Backend decision` section to `## Native backend selection` with the
  new status vocabulary (`not evaluated | SPIKING | SELECTED | REVISE | BLOCKED` + a `Selected
  strategy` field, replacing `GO | REVISE | DEFER | STOP`). CD-002's own text is **not** rewritten
  (append-only) but is now superseded in one specific respect: its framing that "Gate C3 will
  need fresh evidence for the general Core compilation question" remains true, but its implicit
  suggestion that a DEFER/STOP-style outcome remains available for general native compilation no
  longer holds — see the correction notes added inline in `COMPILER-CHARTER.md` §1.5 and
  `COMPILER-ROADMAP.md`'s header relationship note, both of which point back to this entry.
  Gates C0-C2 work already completed (this entire session, through WP-C1.2) required **no
  rework** — none of it touched native-compilation framing. Both brief files are left on disk
  as-is (the original for historical reference, the "(1)" revision as the new live source); this
  is a content decision, not a file-management one, and neither file was deleted or renamed.
- CD-006 [2026-07-18, WP-C1.5] Resolved a spec-internal tension in `03-Type-System.md`'s Numeric
  Semantics section, found during the WP-C1.5 audit and flagged to the user rather than resolved
  unilaterally (CE2-shaped): the section states both "Division or modulo by zero is a runtime
  error and MUST trap" and, in an adjacent bullet, "Floating-point operations follow IEEE-754
  semantics (NaN, +/-Inf)" — the current implementation traps on `0.0 / 0.0` (a literal reading
  of the first bullet), which is in tension with the second bullet's implied NaN/Inf behavior for
  floats specifically. **User decision: keep trapping (current behavior); no code change.** The
  "MUST trap" rule applies uniformly across all numeric types including floats; the IEEE-754
  bullet is read narrowly (governing ordinary float arithmetic results — e.g. overflow producing
  `+Inf`, not division by zero specifically, which STARK treats as an error condition like any
  other div-by-zero). No spec or code edits made under this decision; recorded so the question is
  not re-litigated in a future WP. `interp.rs`'s Float `BinOp::Div`/`Rem` arms are unchanged.
- CD-007 [2026-07-18, WP-C2.1] Settled a spec-silent gap found while writing
  `STARKLANG/docs/compiler/reference-execution.md` §1: the spec addressed almost no
  subexpression evaluation order (binary operands, call arguments, method receiver-vs-arguments,
  aggregate-literal fields, assignment lhs-vs-rhs, index base-vs-index). Flagged to the user
  rather than resolved unilaterally (CE1/CE2-shaped, per WP-C2.1's own scope-control answer).
  **User decision: adopt the interpreter's observed left-to-right order as normative.** Added a
  new "Evaluation Order (Core v1)" subsection to `03-Type-System.md` (after "Operators and
  Traits," before "Copy and Drop") stating: strict left-to-right evaluation for binary operands
  (non-short-circuit), call arguments, struct/tuple/array literal fields, and index base-before-
  index; short-circuit semantics for `&&`/`||` (already spec-derivable, now stated explicitly);
  condition/scrutinee-before-branches for `if`/`match` (also already spec-derivable); receiver-
  before-arguments for method calls; and right-hand-side-before-left-hand-side-place for
  assignment (explicitly flagged as the most surprising rule, since many C-family languages
  evaluate the LHS place first). `STARK-Core-v1.md`/`.html`/`.pdf` regenerated in the same change.
  No interpreter code changes needed — `interp.rs` already implements exactly this order
  throughout (confirmed during WP-C2.1's own drafting); this decision closes the spec-vs-
  implementation gap from the spec side, not the code side.
- CD-008 [2026-07-18, WP-C2.1] Settled a second spec-silent gap found in the same document, §10.3:
  `HashMap`/`HashSet` iteration order was unaddressed by any normative spec text, while the only
  related prose (`06-Standard-Library.md`'s non-normative "Performance Notes" — "HashMap<T> uses
  open addressing with Robin Hood hashing") implied unordered iteration, in tension with the
  interpreter's actual `BTreeMap`/`BTreeSet`-backed fully-sorted-deterministic behavior. Flagged
  to the user rather than resolved unilaterally (CE1/CE2-shaped). **User decision: adopt
  sorted-deterministic (ascending key order) as normative.** Added a new "Iteration Order (Core
  v1)" subsection to `06-Standard-Library.md` immediately after the `HashSet<T>` API block,
  stating `HashMap::keys`/`values`/`iter`, `HashSet::iter`, and `for`-loops over either MUST visit
  entries in ascending key order per the key type's `Ord` impl, regardless of internal storage
  strategy. Reworded the "Performance Notes" line to remove the implication of unordered
  iteration (now frames storage strategy as implementation-defined but explicitly subordinate to
  the iteration-order requirement — an open-addressing implementation would need to sort at
  iteration time to conform). `STARK-Core-v1.md`/`.html`/`.pdf` regenerated in the same change
  (shared with CD-007). No interpreter code changes needed — `interp.rs`'s `BTreeMap`/`BTreeSet`
  representation already satisfies this rule exactly.
  **Correction (CD-009, same day, external review):** CD-008 as originally written is broken —
  `HashMap<K, V>`/`HashSet<T>` only bound `K`/`T: Hash + Eq` (confirmed:
  `06-Standard-Library.md` lines 271, 293), never `Ord`, so "ascending key order per the key
  type's `Ord` impl" can require an implementation that isn't guaranteed to exist. It is also
  inaccurate to describe the interpreter as already satisfying this rule: `interp.rs`'s
  `BTreeMap`/`BTreeSet` sort by `Value`'s own internal structural `Ord` (a Rust-level total order
  over the runtime representation), not by dispatching to the STARK key type's own `Ord`
  implementation (which, per DEV-027 found in this same WP, cannot even be written today). CD-008
  is left as-is above (append-only — a record of what was decided, even though wrong), superseded
  by CD-009.
- CD-009 [2026-07-18, WP-C2.1 correction pass, external review] Corrects CD-008. **User decision:
  `HashMap`/`HashSet` iterate in first-insertion order**, not sorted-by-key order — no `Ord` bound
  needed (matches the actual `Hash + Eq` bound), still fully deterministic. Reworded
  `06-Standard-Library.md`'s "Iteration Order (Core v1)" subsection accordingly (insert appends to
  iteration order; re-inserting an existing key keeps its position; remove-then-reinsert moves it
  to the end) and reworded "Performance Notes" to match. `STARK-Core-v1.md`/`.html`/`.pdf`
  regenerated. **This is now a real, confirmed WP-C2.2 deviation, not a no-op**: `interp.rs`'s
  `BTreeMap`/`BTreeSet` representation does not track insertion order at all (it sorts by
  structural `Value::Ord`), so it does not satisfy the corrected rule — recorded as DEV-032.
- CD-010 [2026-07-18, WP-C2.1 correction pass, external review] Refines CD-007. **User decision:
  keep "the method receiver evaluates before any argument" as normative** (matching user-defined
  method dispatch and common OOP convention), rather than changing the rule to match a narrower
  implementation detail. However, re-reading `interp.rs::call_core_method` (the dispatch path for
  builtin/stdlib-type methods — `Vec`, `String`, `HashMap`, etc., as opposed to user-defined
  nominal types) during the same review found it evaluates argument expressions *before*
  resolving the receiver — the exact opposite of `call_method`/`call_user_method`'s order for
  user-defined types. CD-007's original claim "no interpreter changes are needed... `interp.rs`
  already implements exactly this order throughout" is therefore **incorrect** for this one path;
  left as-is above (append-only), corrected here. Recorded as a new WP-C2.2 deviation, DEV-033 —
  `call_core_method` needs to resolve the receiver before evaluating arguments, to match the now-
  confirmed-normative rule and `call_method`'s own behavior for user-defined types.
- CD-011 [2026-07-18, WP-C2.1 correction pass, external review] DEV-029 (struct/enum field drop
  order is alphabetical-by-field-name, not declaration order) was recorded as a confirmed
  deviation, but `05-Memory-Model.md`'s Drop Order section only ever demonstrated reverse-
  declaration-order for sibling `let` bindings — it never actually stated a rule for a struct's
  own field-internal drop order; DEV-029's framing called reverse-declaration-order "the only
  coherent extension" (an inference, not a citation). Flagged to the user rather than left as an
  inferred deviation (CE1/CE2-shaped). **User decision: amend the spec to state it explicitly.**
  Added two sentences plus a short example to `05-Memory-Model.md`'s Drop Order section extending
  the existing rule to struct/enum-variant fields (reverse declaration order). `STARK-Core-v1.md`/
  `.html`/`.pdf` regenerated (this addition included a new `stark` code block, requiring a spec-
  fixture re-triage: `05-Memory-Model__22.stark` through `__27.stark` renumbered to `__23`
  through `__28`, new `__22.stark` triaged `parse-pass`/`program`; verdict census updated to 68/
  122; `extract-spec-examples.sh` confirms the manifest is back in sync). DEV-029 is now a
  confirmed, spec-backed deviation rather than an inferred one — its ledger entry updated to cite
  the new normative text instead of describing the rule as inferred.
- CD-012 [2026-07-18, WP-C2.7] Approved CORE-Q-006 and the normative Core v1 abstract machine.
  Runtime authority moves from scattered operational prose to
  `CORE-V1-ABSTRACT-MACHINE.md`. Evaluation is exactly once; assignment evaluates RHS before
  destination, installs the new value before destroying the old; normal early transfers clean
  exited scopes; language traps abort without unwinding, including during destination resolution
  and partial aggregate construction. Reference identity is abstract and survives legal
  ownership/call transfers; returned receiver-derived references designate caller objects and
  range slices are live views. CORE-Q-020 is approved only for runtime ownership/destruction of
  existing Core patterns, and CORE-Q-017 only for the language-trap boundary; C2.8/C2.9 retain
  their remaining portions. This decision defines semantics but deliberately defers compiler/
  interpreter alignment and adversarial rule evidence to C2.11.
- CD-013 [2026-07-18, WP-C2.7 correction] Corrected CD-012's CORE-Q-006 approval scope.
  CORE-Q-006 is approved for runtime abstract-machine semantics only; static place legality,
  borrow coexistence/regions, temporary-reference escape, and returned-reference legality remain
  pending under C2.8. This supersedes only CD-012's phrase "Approved CORE-Q-006", not its
  runtime decisions or its C2.11 implementation-alignment deferral.
- CD-014 [2026-07-18, WP-C2.8] Approved the Core v1 static-semantics freeze. Type aliases are
  transparent; values are finitely sized with only `str`/`[T]` unsized behind references;
  inference is deterministic and function-local; trait selection is source-order-independent
  with no specialization; borrows have conservative lexical regions and no temporary
  extension; patterns use deterministic exhaustiveness/usefulness analysis; and constants use
  a closed side-effect-free evaluator. Standard-library hooks are recognized by canonical item
  identity only. CORE-Q-002/003/004/005A/006/007/015/020 are approved. CORE-Q-005 is partially
  approved because C2.9 still supplies canonical package/version identity. Numeric results,
  float trait participation, layout-query results, and resource-limit classification likewise
  remain C2.9 inputs. Compiler/interpreter alignment and granular evidence remain C2.11 work.
- CD-015 [2026-07-18, WP-C2.9] Approved the numeric, target, text, process, package, and
  standard-library contract freeze. Integers are fixed-width and checked; primitive floats use
  reproducible IEEE operations but do not implement `Eq`/`Ord`/`Hash`; text is valid UTF-8 with
  byte offsets and Unicode 15.1 casing. Package identity is relocation-stable and lock-backed,
  with one selected version per source/name/major line. Only `size_of`/`align_of` expose
  target layout and Core promises no ABI. Four no-argument `main` signatures have deterministic
  status/stream mappings. `core-min` is mandatory and `std-full` is optional but indivisible.
  Resource, compiler-limit, API-error, language-trap, and host/process failures are distinct.
  CORE-Q-005, Q008–Q014, Q017–Q019, Q021, Q023, and Q024 are approved; alignment remains C2.11.
- CD-016 [2026-07-18, WP-C2.10] Approved CORE-Q-016 and the Core v1 future-extension
  boundary. Core execution is safe and single-threaded; capturing closures, explicit lifetime
  syntax/reference fields, trait objects, concurrency, macros, unsafe, and general FFI remain
  outside Core. Future callables must preserve ownership/capture/Drop semantics. Host access is
  limited to metadata-bound approved native providers with explicit identity, integrity, ABI,
  target, provenance, capability, and verification. Extensions require explicit stable
  identity/version enablement and cannot change Core-only behavior. No future feature is
  implemented by this decision; C2.11 owns exclusion/isolation enforcement evidence.
- CD-017 [2026-07-18, C2.8/C2.9 correction] Clarified nine pre-C2.11 freeze points.
  Generic fields may instantiate with references and recursively propagate borrow provenance;
  constant patterns never invoke user `Eq`; positive bounds never prove unifying impl heads
  disjoint. Canonical package names are distinct from identifier-valid aliases, each alias
  selects exactly one major line, and all packages remain library-importable while executable
  mode selects the root `main`. Floating `**` is rejected. Standard hash values use canonical
  FNV-1a encodings and primitive Display bytes are exact. `std-full` freezes availability and
  explicitly stated behavior only; unstated method edge cases are not conformance claims.
- CD-018 [2026-07-18, roadmap amendment before WP-C3.1] Adopted the post-C2 roadmap correction
  brief without replacing the core C3-C7 sequence. Inserted mandatory `C3-ENTRY — Native
  Readiness and Carry-Forward Closure` before WP-C3.1; made pending-owner-approval rows,
  DEV-051/052/055 ownership, WP-C2.12 generated-corpus/cross-backend transfer, versioned corpus
  freeze, and native-path CI baseline explicit. Strengthened C3.1's frozen workload with
  generics, trait dispatch, default trait sibling calls, references/slices, Drop-bearing trait
  dispatch, opaque host resources, and provider-boundary file I/O. Added Native Provider ABI
  v0.1 to C5.1, removed C5.4's "where supported" generic-call escape hatch, introduced platform
  tiers, added real systems workloads to C7 measurement, and created
  `STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md` with S0-S7 plus the post-C6 P1 Native Systems
  Baseline checkpoint. This is a sequencing and evidence-governance amendment; it does not
  reopen C2 or change Core v1 semantics.
- CD-019 [2026-07-19, C3-entry follow-up amendment] Tightened the post-C2 roadmap amendment
  before WP-C3.1. DEV-060 is now owned by C3-ENTRY and must be disposed before the workload
  freeze. P1 now gates C7.5/C7.7 closure and is required for Native Systems Preview and
  STARK v1 General-Purpose Stable claims, while Core v1 Compiler Stable may describe compiler
  maturity without claiming systems-platform maturity. The C3 provider/resource experiment is
  explicitly disposable and non-normative; C5.1 remains the first stable Native Provider ABI.
  Systems S6 is split into joint concurrency tracks for language proposal, compiler
  implementation, runtime/provider work, and ecosystem validation. `COMPILER-STATE.md`'s
  load-bearing header now points at `c268d7c`, the 594/0/2 verified code baseline, and the
  remaining C3-entry blockers.
- CD-020 [2026-07-19, C3-entry governance-repair pass — no semantic or compiler change]
  Repaired the governance surface before C3-ENTRY closure work begins. (a) Created
  `work-packages/WP-C3-ENTRY.md` — the transition's executable WP: named exit artifact
  (`starkc/docs/compiler/C3-entry-exit.md`), mechanical corpus-freeze definition
  (`corpus.lock`, SHA-256 per file, version-bump rules), per-blocker owners, "Done when";
  roadmap C3-ENTRY section now points at it. (b) Amended WP-C4.4/C5.6/C6.5 in
  `COMPILER-ROADMAP.md` to carry their transferred WP-C2.12 generated-corpus/cross-backend
  obligations in the receiving WP text (previously stated only in the C3-ENTRY bullet list,
  invisible to the charter's minimal session-input packet). (c) CI baseline delta:
  `.github/workflows/ci.yml` commands widened to the C3-ENTRY forms, added spec-regeneration
  check (new `--check` mode in `STARKLANG/tools/build-core-spec.py`, Markdown-only since
  pandoc/weasyprint output is not byte-reproducible) and a named execution-snapshot step;
  local fmt + exec_snapshots verified green, full CI run pending. (d) Accuracy corrections:
  `KNOWN-DEVIATIONS.md` tail summary (claimed DEV-009/022/023/024 open; all four resolved by
  WP-C2.11 per their own entries — stale paragraph from C2.6 time), state header current-head
  (`c268d7c` → `9e85396`) and spec-fixture census (112/parse-pass-64 → directly re-counted
  113/parse-pass-65; evidence-inventory "121-fixture" figure also corrected), charter §1.5/§2.4
  "roadmap §5.3" dangling references (vocabulary lives in charter §5.3), charter §2.1 step 10
  commit policy (owner convention: commit only on explicit request), WP-C6.4 tier label ("Core
  v1 Stable" → "Core v1 Compiler Stable" matching the C10 release class), and a new
  "Relationship to the compiler roadmap's P1 checkpoint" section in
  `STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md` (CD-018 described P1 as living there but the
  file never mentioned it; S5 is now explicitly identified as the P1-completing stage).
  (e) Compressed this file from 3,145 to ~700 lines per charter §2.4: deviation seed sections,
  C0/C1-era file inventory, completed follow-ups, and session records through Post-Gate-C2
  Issue 5 moved **verbatim** to `STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md`;
  decision log, conformance summary, gate exit summaries, open-deviation index, and the
  Issues 6-8 session record retained inline. Charter/roadmap edits under this entry are
  governance/bookkeeping repairs, not meaning changes to the extracted brief.
- CD-021 [2026-07-19, owner-approved roadmap amendment — function-value native validation,
  P1 trap report, release deviation sweep] Origin: an external-review debate established that
  non-capturing `fn(...) -> ...` function types are **existing frozen Core v1 capability**
  (`03-Type-System.md:198-200,999`; stdlib contract `06-Standard-Library.md:243-244,260-262,
  663-666`; `interp.rs:260` `Value::Function(ItemId)`), not a future closure feature — so the
  native path must validate them explicitly rather than leave them implicit. Three changes,
  same style/class as CD-018's workload strengthening: (a) WP-C3.1's frozen workload gains
  items 16-21 (typed function-value local; indirect invocation; `Option::map`/`Result::map`
  with a function value; function value in a struct field; cross-package function reference;
  monomorphised-generic function value with an explicit record-the-boundary fallback) — any
  item failing against the current implementation becomes a DEV entry before backend
  selection, deliberately; C4 gains explicit indirect-call ownership (WP-C4.1 MIR
  function-value constants/indirect-call representation, WP-C4.3 indirect-call signature
  verification, WP-C4.5 function-value lowering with provenance); WP-C5.1's runtime ABI list
  gains function-value/code-pointer representation, indirect calling convention,
  cross-package function-symbol identity, and function values in aggregates. (b) P1's exit
  list (roadmap §4.2) and S5's requirements (`SYSTEMS-ROADMAP.md`) gain a documented
  trap-abort operational report — deliberately trap one handler, record the effect on
  in-flight connections/resources/buffered output/process state; evidence input for any
  future fault-isolation proposal, explicitly no semantic change. (c) WP-C10.7 gains a
  release-blocking deviation sweep: every open deviation needs an owning gate/WP or a
  recorded accepted-indefinitely disposition. Related but not enacted here: the planned
  paper-only "Callable ABI and Future Closure Compatibility Spike" memo (existing-capability
  section + future-closure-compatibility section, outcomes GO/REVISE-ABI/
  DEFER-ESCAPING-BORROWS/ANNOTATIONS-LIKELY/NO-CURRENT-DESIGN) remains a separate proposal to
  be drafted before WP-C5.1; it is a recommendation, not yet approved work.
- CD-022 [2026-07-19, owner-approved follow-up amendment — external review of CD-020/CD-021
  commits] Three changes. (a) **Release-class coherence repair, preserving CD-019.** External
  review correctly found two superimposed models: C7.7 requires P1 (CD-019), Core v1 Compiler
  Stable requires C7, so its "must not claim systems-platform maturity unless P1 is complete"
  conditional was vacuous and General-Purpose Stable's "+P1" added no evidence. Resolution
  keeps CD-019's C7 gating (its motive — no toy-workload performance report — stands) and
  recasts the two stable classes as differing in **claim scope, not evidence**: Compiler
  Stable necessarily carries P1 evidence but asserts compiler maturity only; General-Purpose
  Stable adds no evidence gate and is the class permitted to assert systems-platform
  maturity. The reviewer's alternative (decouple C7 from P1) was considered and rejected as a
  CD-019 reversal. (b) **Function-value property validation.** WP-C3.1 gains workload items
  22 (repeated indirect invocation through one local — spec-guaranteed by function values
  being `Copy`, `03-Type-System.md` §Copy and Drop; DEV-060 is this bug class for default
  trait methods) and 23 (`Copy` aggregate with a function-value field, copied, both copies
  invoked), plus a pre-backend-selection requirement to settle the two genuinely open
  properties — `Eq`/`Ord`/`Hash` participation and monomorphised-generic function-value
  identity — from the frozen spec or by CE1/CE2 escalation, never by MIR/ABI accident. The
  reviewer's broader open-question list (Copy? repeated calls? Drop?) was narrowed: those are
  already frozen by the spec's Copy rule. (c) **State-header field rename**: "Current
  committed head" → "Amendment base commit" (self-referential staleness by construction).
  Outstanding from the same review, not part of this entry: a demonstrated green CI run
  (requires pushing to origin; no run exists yet).
- CD-023 [2026-07-19, owner-approved] Approved all six `pending-owner-approval` completeness
  rows (`LEX-COMMENT-001`, `LEX-ERROR-001`, `STD-OPTION-001`, `STD-RESULT-001`, `STD-ITER-001`,
  `STD-VEC-001`) as-is — the behavior each row describes has been implemented and exercised
  throughout Gate C2; the gap was governance bookkeeping only (C2 exit report). All six flipped
  to `settled` in `CORE-V1-COMPLETENESS.md` (`LEX-ERROR-001` keeps its DEV-017 note — an
  evidence-citation-precision gap, not a behavior question). C2-exit-report.md gained a dated
  post-gate update note per the same convention as the DEV-051/052/055 correction, rather than
  rewriting historical gate-close evidence. This closes the first of C3-ENTRY's four blockers;
  DEV-060, the corpus freeze, and the green CI run remain open.
- CD-024 [2026-07-19, owner-approved disposition: fix now] Closed DEV-060 (repeated call to an
  un-overridden trait default method wrongly flagged as a move). Root cause: `borrowck.rs`'s
  `method_receiver` — consulted by the `Call` handler to decide whether a method receiver is
  moved, borrowed, or mutably borrowed — only ever searched `ImplItem::Fn` overrides, with no
  equivalent to `typecheck.rs::resolve_method`'s `default_fallback` (WP-C1.3/DEV-013). A call to
  an un-overridden trait default method therefore returned `None` from `method_receiver`, and
  the `Call` handler's `None => self.check_expr(*base)` arm ran instead of the `Some(Receiver::
  ..)` arms — `check_expr`'s `Path` arm unconditionally consumes (moves) any `Local`/`SelfValue`
  place, regardless of the method's real receiver kind. Fixed by adding the matching
  trait-default-body fallback to `method_receiver` itself, mirroring `typecheck.rs`'s search but
  returning the method's declared `sig.receiver`. Verified both the `&self` case (original
  repro) and a new `&mut self` companion case (the `RefMut` arm wasn't exercised by the original
  repro alone — two sequential calls must register two non-conflicting borrows, not a move), and
  that the original repro now executes with correct output twice, not just "no diagnostic".
  Full workspace suite: 596 passed / 0 failed / 2 ignored (up from 594 — one new typecheck test,
  one new interp execution test, one existing test rewritten in place from
  documenting-the-defect to asserting success). `cargo fmt --all -- --check` and `cargo clippy
  --workspace --all-targets --all-features -- -D warnings` both clean. Full writeup:
  `KNOWN-DEVIATIONS.md`'s DEV-060 entry. This closes the second of C3-ENTRY's four blockers; the
  corpus freeze (now unblocked — WP-C3-ENTRY.md's procedure required this fix to land first) and
  the green CI run remain open.
- CD-025 [2026-07-19] Froze the WP-C2.12 execution-snapshot corpus and closed C3-ENTRY. Blocker
  3 (corpus freeze): `starkc/tests/exec_snapshots/corpus.lock` created at `corpus_version =
  1.0.0`, base commit `3d12f45`, SHA-256 per corpus file (48 files: 31 `.stark` + 17 `.snap`
  incl. `metamorphic/`); lock digest
  `8cda2df5e26aa35dfc8eb222f1e073eb4ea2336297e91ecc4e62b8fbd27dc0dc`. New integrity test
  `corpus_lock_matches_frozen_snapshot` (exec_snapshots.rs) enforces hash-match + no-missing +
  no-unlisted, negatively verified (tampering one `.snap` fails it with the expected message;
  restore passes). Freeze taken after DEV-060's fix per WP-C3-ENTRY.md procedure. Blocker 4 (CI):
  green on `origin/main` @ `3d12f45`, owner-confirmed. With blockers 1 (CD-023) and 2 (CD-024)
  already closed, **C3-ENTRY is closed** — exit artifact `starkc/docs/compiler/C3-entry-exit.md`
  written, Position line flipped to `Gate: C3  Next: WP-C3.1  Blocked: none`. Any future corpus
  change must bump `corpus_version` with a dated note here; a bare `UPDATE_SNAPSHOTS=1`
  regeneration is a freeze violation the integrity test catches. No semantic or Core behavior
  change.

- CD-026 [2026-07-19, WP-C3.4, owner CE5 decision] **Backend selection: `SELECT-GENERATED`.**
  Generated Rust is the initial production backend behind verified MIR; the MIR contract is to be
  designed backend-neutrally so `SELECT-DIRECT` (Cranelift) remains a live C7-gated migration
  (charter §1.6 rule 9). Basis: WP-C3.2 (generated-Rust) reached 8/17 frozen-corpus breadth
  cheaply with zero mismatches and trap parity, the shortest/lowest-risk path to correct broad
  native compilation (charter §1.6 rule 7); WP-C3.3 (direct Cranelift) is correct and self-
  contained (no rustc dep) but owns monomorphization/layout/drop/runtime up front — the better
  *eventual* backend if the self-contained-compiler goal becomes primary, which is a C7 judgment.
  Neither `REVISE` (missing data — exe size/startup, MIR-level comparison — is inherent to
  sequencing, needs C4-C7, not a bounded pre-C4 follow-up) nor `BLOCKED` (both paths demonstrated
  correct native execution). Accepted trade: `stark build` permanently requires a rustc toolchain
  and is slower; acceptable for STARK-as-research-language, re-evaluated at C7. Full three-way
  analysis + the required architecture commitments (MIR boundary, runtime/ABI, targets, debug
  mapping, unsupported-MVP closure, why-direct-rejected-as-initial):
  `starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md`. Gate C3 closes; next is
  Gate C4 (MIR contract, CE3). This decision selects a backend strategy only — it does not build
  MIR, define the MIR contract, or fix the runtime ABI (those are C4/CE3 and C5.1/CE4).
- CD-027 [2026-07-19, owner-approved: two CE freezes + a correction-pass authorization] Settled
  the two CD-022 carry-forward function-value properties and repaired the fn-value feature
  cluster found by executing CD-021 workload items 16-22 against the interpreter for the first
  time. **(a) CE1 — TYPE-FN-001** (new normative rule, `03-Type-System.md` §Function Types):
  function values are `Copy`/`Clone`, never `Drop`, and do **not** implement `Eq`/`Ord`/`Hash`
  in Core v1 (float-precedent); consequence: function-value identity is unobservable, so the
  monomorphised-generic-identity question collapses to deterministic symbol naming (C6.2), not
  language semantics. **(b) CE2 — TYPE-FN-002** (same section): a generic fn coerces to a
  concrete fn type only when the expected type fully determines every generic argument;
  semantics = instantiate at the coercion site. Combined spec regenerated; no new code blocks so
  no fixture re-triage; two granular rows (TYPE-FN-001/002) added to `CORE-V1-COMPLETENESS.md`
  (166 → 168 rows — the fn-value questions were a genuine inventory gap). **(c) Pre-C4.1
  correction pass (authorized fix-now):** DEV-061 (indirect calls through fn-value locals/params
  never executed — missing `Res::Local|SelfValue` arm in interp call dispatch; the machinery
  existed one arm below), DEV-062 (fn values not `Copy` in borrowck/typecheck — `Ty::Fn`
  explicitly misclassified against the spec's Copy list), DEV-063 (`Option::map`/`and_then`,
  `Result::map`/`map_err`/`and_then` absent from the method table despite the normative §Option/
  §Result APIs) — all three FIXED with 5 new regression tests; the semantic oracle can now
  execute workload items 16-22. One new narrow deviation found and deliberately not fixed in
  this pass: DEV-064 (undetermined-generic fn coercion accepted; TYPE-FN-002 requires rejection;
  owner C4.5). Note: these settlements landed after CD-026's backend selection but before any
  MIR/ABI work — the selection is unaffected (identity-unobservability removes the one property
  that could have differentiated the candidates' ABIs).

- CD-029 [2026-07-19, review-directed correction pass before C4.5 breadth] Four corrections
  from the external review of the C4.1-C4.4 foundation, applied before they could embed across
  complete-Core lowering. (a) **Trap provenance**: `MirRunError::Trap` was discarding
  `SourceInfo` — a right-category trap at the wrong location would have passed the C4.4
  differential; outcomes now carry full `TrapInfo`, mir.md §6 amended to make provenance part
  of the observable trap outcome, and the differential compares user-origin trap spans exactly
  against the oracle (synthetic origins compare classification). Both existing trap tests pass
  with exact span equality. (b) **TypeContext contract treatment**: formally amended into
  mir.md §2 as part of the in-memory MIR compilation unit (additive, not dump-serialized, MIR
  stays v0.1) — resolving the governance debt the WP-C4.3 record flagged. (c) **Verified-MIR
  wrapper**: `verify_program` returns `VerifiedMirProgram<'_>`; `run_program` (and eventually
  the generated-Rust backend) consumes only that — "no backend bypasses MIR validation" is now
  an API property. (d) **Differential-independence caveat**: the shared `canonical_float`
  formatter is structurally invisible to the HIR/MIR differential; claim qualified everywhere
  going forward ("no difference in lowering and MIR execution for the tested subset, with some
  runtime algorithms intentionally shared") and compensated by new spec-derived golden +
  round-trip property tests (`tests/canonical_float.rs`, incl. NaN/±inf/-0.0/notation
  boundaries at exponent 15↔16 and -4↔-5/subnormals/max-min finite). Also adopted the review's
  C4.5 increment ordering + honest maturity calibration (architecture ~90%, implementation
  breadth ~35-45%, validation ~70%) into WP-C4.5.md.

- CD-030 [2026-07-19, owner-approved disposition of the external C4.5c-head review] The review
  (written against `82211f6`, before WP-C4.5d landed) found three validation holes plus two
  warnings. Disposition: **fold the load-bearing items into C4.5e as its entry step
  (WP-C4.5e-0)** — (1) IndexProof definite-initialization dataflow (the global name→base map
  alone accepted MIR whose check ran on only one branch; slices in C4.5e build directly on the
  proof discipline), (3) V-REF-1 write-through-shared-reference rejection (MIR-0014), (4)
  partial-output-before-trap comparison in the differential (C4.5e's panic/assert paths are
  exactly where it matters; both engines now expose pre-failure stdout —
  `interp::run_with_partial_output`, `MirFailure`), plus the review-warned user-`impl Copy`
  misclassification, confirmed real (valid Copy-struct programs failed MIR verification as
  use-after-move) and fixed as **DEV-068**. **Deferred with owners** (defense-in-depth only,
  no observable-behavior risk in the current subset): frame-generation identities in the MIR
  interpreter (owner: C4.5f, before cross-package call graphs grow frames) and
  projected-move take-and-poison (owner: C4.5e proper, alongside the runtime values that make
  aggregates bigger; the unit-flag design makes the current clone-not-take unobservable, and
  the stale interp comment claiming whole-local verifier conservatism was corrected). Review's
  wording caution accepted: C4.5c externally = "top-level generic monomorphisation and static
  bound dispatch" (generic *methods*/impls and user-nominal Eq/Ord operator lowering remain
  later-increment work). The review's C4.5d checklist was already fully implemented by the
  WP-C4.5d commit it had not seen, except the two deferred items above.
- CD-031 [2026-07-19, CE3 — owner-approved MIR v0.1 Amendment A1] Approved
  `STARKLANG/docs/compiler/mir-amendment-A1-strings-runtime.md` (rev. 3) as a **narrow additive
  amendment to MIR v0.1**, runtime surface `0.1-A1` — the contract prerequisite the C4.5e-main
  body needs before lowering strings/collections. Additions, all additive (no existing construct
  reinterpreted): `Constant::Str(String)` = a decoded immutable UTF-8 literal typed `&str`
  (owned `String` only via runtime `StringFromStr`; literal identity unobservable);
  `Terminator::Trap { message: Option<Operand> }` for `panic`/`assert` messages (participates in
  every operand analysis, not just typing); `String`/`Vec`/`VecIter` become drop-elaborated
  runtime values (**always** buffer-reclaim glue; element-destructor execution conditional on
  `T`; `Vec<T>` element drop in **reverse index order**, matched empirically to the frozen oracle
  `interp.rs::drop_value`); a versioned `RuntimeFn` appendix (30 ops lowered in C4.5e + a reserved
  group activated later only by a dated enumeration bumping the surface id); one new in-memory
  `TypeContext` field (`copy_types`) and two new `MirProgram` fields (`mir_version`/
  `runtime_surface`, consumer-checked before any body); new verifier codes MIR-0015 (V-STR-1/2,
  Trap.message typing), MIR-0016 (V-COPY-1: `VecIndexGet`/`VecIterNext` require `T: Copy`;
  `VecClear` requires non-droppable `T`). Two owner-mandated honesty rules: no `RuntimeFn` ever
  runs a user element destructor (those run only at visible `Drop` terminators — `clear()` on
  droppable `T` lowers to a pop-and-drop loop; `v[i]=x` uses `VecReplace(...)->T` so the caller
  drops the old value); and a backend doing explicit reverse-order element destruction must
  suppress any automatic (Rust) element drop. Three rev cycles (rev. 1 direction approved; rev. 2
  eight corrections; rev. 3 four final corrections) recorded in the doc's §11. `mir.md` §5/§7
  carry pointers to the amendment; `MIR_VERSION` stays `0.1`. This decision approves the contract
  only — no code is written by it; the C4.5e main body implements it next.

- CD-032 [2026-07-19, owner decision — A1 iteration correction, folded into C4.5f] The
  WP-C4.5e-2 implementation surfaced that Amendment A1's by-value `VecIterNext -> Option<T>`
  ("the `for x in v` desugar") has **no STARK source trigger**: STARK has no by-value
  `for x in v`; the only iteration form is `for x in v.iter()`, and `Vec::iter()` binds the
  loop variable as `&T` (stdlib `iter(&self) -> VecIter<T>`). So all Vec/collection iteration
  in STARK is **by-reference** — an interior reference into a runtime container, which is the
  work A1 §5d already reserved and tied to C4.5f's frame-generation hardening. **Owner
  decision: fold iteration into C4.5f.** A1's by-value iteration ops are struck from surface
  `0.1-A1` (they were never added to the `RuntimeFn` enum, so `0.1-A1` as implemented is
  unchanged — no bump); by-reference iteration (`VecIterNew`/`VecIterNext` yielding
  `Option<&T>`) is a C4.5f deliverable activated by a future dated `0.1-A2` surface bump,
  alongside `VecGetRef`/`StringSubstring` interior views and the frame-generation identities.
  Amendment doc updated (rev. 4): §5c iteration rows struck, §5e reframed as the C4.5f
  carry-forward design, rev-4 log added. No code change; strings (e-1) and the Vec data
  surface (e-2) are untouched. `collection_iter__01`'s `for value in values.iter()` stays
  clean-Unsupported until C4.5f; its push/index/len half lowers under e-2.
- CD-033 [2026-07-19, owner disposition of the WP-C4.6 gate-exit audit] **Gate C4 stays
  open under the strict reading: "every normative Core construct required by C5" means the
  full normative Core language plus the `core-min` stdlib profile, NOT a representative-
  workload subset** (which would weaken the gate and let known language gaps transfer into C5
  merely because the chosen app avoids them). `core-min` is the C5 baseline, not std-full.
  **Required before C4 exit:** A1 (generic impls/assoc fns/trait methods/generic Drop), A2
  (general + nested pattern lowering), A3 (user `Eq`/`Ord` operator dispatch — `Eq` may
  proceed independently, but the `Ordering` runtime-surface amendment must be drafted for CE3
  review before the `Ord` portion is implemented), A4 (`core-min` ops: chars iteration,
  `Vec::get`/`get_mut`, slices, `size_of`/`align_of`, first-class integer ranges, and the
  `core-min`-classified Option/Result operations — via a required dated runtime-surface
  amendment), A5 (bit/shift/pow operators), A6 (non-Copy Vec iteration — the Copy restriction
  is an implementation compromise, not a language rule), A7 (normative expression forms).
  **May remain reserved beyond C4** unless separately required by the stable Core contract:
  std-full ops (`HashSet`, `HashMap::values`/`remove`, `Vec::contains`). **Front-end
  prerequisites with explicit owners:** DEV-069 is a prerequisite for the C5 multi-file/
  multi-package application claim (parallel front-end WP allowed, but C5 must not claim normal
  multi-file support while declaration spans read against the wrong file); DEV-067, `Box`
  deref, and the primitive `Ordering::cmp` surface get explicit owners and are resolved where
  `core-min` requires. **Implementation order (dependency-aware, not smallest-first):**
  (1) A5+A7 mechanical coverage; (2) A6 borrowed Vec iteration; (3) A3 `Eq`, then the CE3
  `Ordering` decision, then `Ord`; (4) A4 runtime/`core-min` surface; (5) A2 general pattern
  lowering; (6) A1 generic impl monomorphisation. The WP-C4.6 exit report is updated after
  each class with positive, negative, verifier, and HIR/MIR differential evidence; C4 closes
  only when all required classes are green and no normative Core or `core-min` construct
  required by C5 remains silently unsupported.

- CD-034 [2026-07-19, CE3 — owner-approved MIR Amendment A2 with clarifications] Approved
  `EnumRef::CoreOrdering` as the MIR representation of the prelude `Ordering` enum (three
  fieldless variants, logical discriminants Less=0/Equal=1/Greater=2 — logical MIR only, not a
  physical ABI; C5.1 owns physical layout) and the ordered-operator lowering (`<`/`<=`/`>`/`>=`
  on a non-generic user nominal → `Ord::cmp` call + discriminant compare; operands borrowed
  left-to-right, never moved). Additive; **runtime surface stays `0.1-A3`, `MIR_VERSION` stays
  `0.1`.** Five clarifications required and applied: (1) renamed "Ordering as a Runtime Value" →
  "Ordering as a Logical MIR Enum" (avoid confusion with the `RuntimeFn` surface); (2)
  discriminants logical-only; (3) recorded the **C4-open additive-amendment versioning policy**
  in `mir.md` (until C4 closes, CE3-approved additive shape amendments stay in v0.1 and are
  recorded in the contract; after C4 exit any shape change needs a version bump) and reflected
  `CoreOrdering` in the contract's `EnumRef` description; (4) `println(Ordering)` is out of A2
  (Display is A4) — the round-trip test verifies construct/return/match only; (5) DEV-070
  accepted as correctly classified and owned by A2. Implemented in the same session with
  full lowering/verify/interp/dump coverage; the invalid-variant guard (v3 → MIR-0008) satisfies
  the CE3 requirement #8. Amendment doc `mir-amendment-A2-ordering.md` marked APPROVED.

- CD-035 [2026-07-20, WP-C4.7-1 — **PROPOSED, awaiting owner CE3 ratification**] **MIR Amendment
  A3 (arithmetic completion), recorded post-hoc.** CD-033 approved class A5 (bit/shift/pow
  operators) and WP-C4.6 implemented it, but the `mir.md` versioning policy also requires each
  additive *shape* amendment to be individually CE3-approved and recorded in the contract, and
  that step was missed. The record now exists in `mir.md` §"A3 shape amendment": pure
  `MirBinOp::BitAnd/BitOr/BitXor` (integer-only; same-width two's-complement results are always
  representable, so no range check is owed and §5 totality holds; `~x` → `x ^ mask` rather than a
  new `MirUnOp`), `CheckedOp::Pow` (NUM-INT-ARITH-001), `CheckedOp::Shl`/`Shr` activated
  (NUM-SHIFT-001; no masking or count reduction), and `TrapCategory::InvalidShift` held distinct
  from `IntegerOverflow`, with the reference interpreter's `CheckedOutcome::Trap(Some(cat))`
  category override specified as a rule backends must reproduce. Additive; `MIR_VERSION` stays
  `0.1` and no runtime-surface identifier changes (A3 adds no `RuntimeFn`). **The ask is
  ratification of the record, not approval of new code — the code shipped in WP-C4.6 A5.**
  Consequence if ratified: WP-C4.7-3's layout amendment is **A4** (`mir-amendment-A4-layout.md`),
  renumbered from the plan's "A3" to avoid a collision.

- CD-041 [2026-07-21, owner decision — DEV-089 close-out + Gate C4 closure] **User `Display`
  dispatch through `print`/`println`/`eprint`/`eprintln`, in both engines; then close C4, open C5.**
  The owner ruled that a user type's own `Display::fmt` must execute (06 treats `Display` as an
  ordinary trait, not a syntax hook), rejecting both the previous oracle debug rendering and the
  MIR refusal. **(a) Spec:** `print`/`println`/`eprint`/`eprintln` respecified as
  implementation-provided generic `<T: Display>` functions; **PRINT-DISPLAY-001** (06-Standard-
  Library) states the nine-point contract (evaluate arg once; select the unique coherent `Display`
  by ordinary resolution; invoke `fmt` once; print exactly the returned bytes; `*ln` appends one
  `0x0A`; destroy the formatting `String` after submission; the argument follows by-value call
  ownership; a trap in `fmt` propagates with no newline/partial result; no fallback for a type
  lacking `Display` — E0500). STD-FORMAT-001 and the prelude/IO signatures updated; compiled spec
  and fixtures regenerated (manifest in sync, 112 blocks). **(b) Oracle:** `display_text` +
  `finish_display` run the impl and drop the by-value argument after its bytes are submitted; the
  internal aggregate rendering is retained only as a diagnostic facility. **(c) MIR:**
  `lower_print_display` emits an ordinary static `Callee::Instance` call to the selected `fmt`,
  then the existing `StringAsStr` + `Print(ln)Str` runtime ops, then visible `Drop`s of the
  formatting `String` and the argument. **No new MIR shape, no new `RuntimeFn`, no runtime-surface
  bump** (`MIR_RUNTIME_SURFACE` stays `0.1-A8`); `fmt` is a normal instance call so user code,
  traps and provenance stay visible. Generic user types and `T: Display`-bounded generic functions
  are supported at their monomorphised instances. **(d) DEV-090** (split from DEV-086): by-value
  iteration over a non-`Copy` array element is rejected in the front end (`E0104`, `borrowck.rs`);
  full ownership-transferring non-`Copy` array iteration is an accepted limitation outside the C5
  baseline, scheduled later. **(e) DEV-088 use-site:** using a `const` declared in another file is
  rejected in the checker (`E0215`), deferred to the front-end/multi-file completion package with
  DEV-083. **(f) Closure:** the six-clause stopping rule (CD-040(c)) now holds in full — clause 3
  satisfied by DEV-089's resolution — so **Gate C4 is CLOSED and Gate C5 (native compilation) is
  OPEN**, 2026-07-21. Evidence: `mir_differential.rs::dev089_*` (8 tests),
  `gate2_valid.rs::printing_requires_display` / `::rejects_by_value_iteration_over_non_copy_array`
  / `::accepts_by_value_iteration_over_copy_array` / `::cross_file_const_use_is_rejected`.

- CD-038 [2026-07-20, CE3 — owner-approved MIR Amendment A5] **`Projection::ConstIndex(u64)`.**
  A statically known array element: valid only on `Array<T, N>`, the verifier checks `index < N`
  itself, no `CheckIndex` terminator and no `IndexProof` local, invalid on `Vec`/slice, dynamic
  indexing unchanged. It participates PRECISELY in move analysis, which is the point — a
  proof-backed `Index` names no statically-known sub-place, so moving one element out poisoned
  every sibling and made consuming array patterns over droppable elements unrepresentable
  (lowering emitted them; verification rejected them). The same decision required **typed internal
  paths**: move-dataflow and drop-unit paths are now typed components (field / variant field /
  constant index) rather than raw `u32` sequences, so distinct projection kinds cannot compare
  equal, and fixed-length arrays decompose into PER-ELEMENT drop units. Additive; `MIR_VERSION`
  stays `0.1`; runtime surface untouched (`0.1-A8`). Recorded in `mir.md` as amendment A5.
  **Narrowed, not closed:** by-value iteration over a non-`Copy` array element — the loop index is
  a runtime counter, so no `ConstIndex` names the consumed element; reading by copy would be
  unsound (double free of a `String` in a real backend), so it is refused cleanly. Closing that
  needs unrolling or runtime-indexed drop flags, a separate design question.

- CD-039 [2026-07-20, WP-C4.7 post-exit-report, owner-directed] **Corpus 1.1.0 → 1.2.0**, completing
  the compact refresh to the six workloads §4 of the owner's directive specified. Adds a
  **multi-file** case (cross-file structs, methods, trait default + override, a cross-file `Drop`,
  and source provenance; its `helper.stark` is a corpus FILE but not a CASE, having no `main`) and
  folds DEV-086's consuming array pattern into the array/slice case. Lock regenerated (58 → 61
  files), base commit updated, and the version assertion in `exec_snapshots.rs` updated in the same
  change. A bump rather than an amendment of 1.1.0 because the array case's bytes changed, which
  the freeze rules treat as a corpus change. **All 48 hashes from 1.0.0 remain byte-identical**, so
  the original baseline survives inside 1.2.0. Writing the multi-file case found **DEV-088**
  (cross-file `const` initializers evaluated against the entry file); the declaration-time half was
  fixed, the use-site half recorded, and the case reduced to its subject per the owner's
  scope-discipline instruction.

- CD-040 [2026-07-20, owner decisions closing out WP-C4.7] Four dispositions.
  **(a) Runtime-surface ratification, post hoc:** A1 rev. 11 (`BoxNew`/`BoxIntoInner`, `0.1-A7`)
  and rev. 12 (exclusive slice views, `0.1-A8`) are ratified. Documentation and the active
  constant agree (`MIR_RUNTIME_SURFACE = "0.1-A8"`), so no implementation change was requested or
  made. **(b) DEV-083 deferred:** *"DEV-083 is deferred to a dedicated post-C5-front-end work
  package. The eventual design must use candidate-local inference snapshots and
  declaration-order-independent candidate evaluation. It must not mutate global inference state
  while probing candidates."* Provisionally assigned to `WP-C6.x Method Resolution Completion`;
  must stay visible in the ledger and in release/conformance reporting.
  **(c) Gate interpretation amended:** C4 exit does not require correcting every recorded
  front-end over-rejection before native-backend work. The stopping rule is: accepted programs
  produce valid verified MIR; unsupported programs reject cleanly; no known mislowering, ownership
  unsoundness or engine divergence remains; MIR contains the concepts C5 needs; the required
  C5/Core baseline lowers; and remaining narrow front-end over-rejections are documented and
  scheduled. **Condition 3 does not silently waive condition 2** — DEV-083 is owner-approved as
  outside the mandatory C5 lowering baseline because it is a front-end inference-completeness
  issue with a workaround and no MIR/backend effect, and that is recorded as a scope decision
  rather than an exemption. **(d) Scope discipline:** no further open-ended C4 audit; only the
  bounded final validation.

- CD-037 [2026-07-20, WP-C4.7-9, owner-directed] **Frozen execution corpus bumped 1.0.0 →
  1.1.0 — ADDITIVE ONLY.** Five new primary cases cover the constructs WP-C4.6's Class-A campaign
  and WP-C4.7 added, every one of which the differential suite exercised but NO frozen case did:
  `ownership_drop__03_discarded_values_and_nested_patterns` (unwrap_or discarding at the call,
  nested-pattern drop order, shorthand bindings), `collection_iter__03_slice_views_and_array_
  iteration` (shared + exclusive slices, write-through to the base, array iteration),
  `struct_enum_trait__05_generic_methods_and_impl_heads` (method-own generics, non-bare impl
  heads, trait-default generics), `primitive__04_bitwise_shift_pow_and_ordering` (A5 operators,
  compound forms, primitive/`Char`/`String` `cmp`, the float operator/trait split), and
  `option_result__03_box_and_layout_queries` (`Box` new/into_inner + drop timing, a recursive type
  through `Box`, layout queries, expected-typed literals). `corpus.lock` regenerated: 48 → 58
  files, base commit updated, and the version assertion in `exec_snapshots.rs` updated in the same
  change per the freeze procedure. **Verified additive:** all 48 hashes from 1.0.0 are byte-identical
  in the new lock and no pre-existing corpus file was modified, so the 1.0.0 baseline survives
  inside 1.1.0 and comparisons taken against it remain valid. All 22 cases agree across the HIR
  and MIR engines. Writing the slice case found **DEV-087** (the oracle treated a slice reference
  as non-`Copy`, so passing one to a function consumed it) — closed in the same change.

- CD-036 [2026-07-20, CE3 — owner-approved MIR Amendment A4, as drafted] Approved
  `Rvalue::LayoutQuery { kind: SizeOf | AlignOf, ty: MirTy }` — a **pure** rvalue typed `UInt64`
  that PRESERVES the queried type, replacing WP-C4.6 A4-1's type-erasing `Const 8` lowering of
  `size_of`/`align_of`. Rationale: 06-Standard-Library classifies them as *target-layout queries*
  and 07's LAYOUT-QUERY-001 makes them the only Core layout observations, so a backend must be
  able to answer them from MIR alone (charter §1.2). Approved with the drafted scope: consumers
  answer through a single layout service; the C4 reference implementation returns `(8, 8)` for
  every type, so **behavior is unchanged and the HIR oracle is not touched** — real per-target
  numbers are C5.1's, since CD-015 fixed none and LAYOUT-ABI-001 makes them target-/
  version-dependent. Not a `RuntimeFn` (type-only input, cannot trap, compile-time knowledge).
  Verifier owns one rule (dest `UInt64`, MIR-0004); `Sized`-ness stays the front end's property.
  Additive: `MIR_VERSION` stays `0.1`, runtime surface stays `0.1-A6`. Alternatives (a) record as
  a deviation, (b) real numbers now, and (c) defer to C5 were presented and declined — (c) would
  have needed a MIR version bump, since C4 exit freezes v0.1 for backend consumption.

- CD-042 [2026-07-21, owner CE4 decision] **`WP-C5-ENTRY.md` APPROVED at its recommended choices;
  WP-C5.1 implementation cleared to begin.** The entry plan (`STARKLANG/docs/compiler/
  work-packages/WP-C5-ENTRY.md`) freezes the Gate C5 supported subset, the generated-Rust
  representation contract, the ownership/move/Drop strategy, the `LayoutQuery` strategy, the
  minimal runtime and Native Provider ABI v0.1 scope, the generated-crate topology, `stark build`
  behaviour, the C5.1-C5.6 work-package sequence, the native differential test matrix, stop/
  escalation rules, and the Gate C5 exit-report format. Owner accepted the §19 decision table as
  drafted (generated Rust backend, debug-only profile, concrete-monomorphised-instances-only
  generics, `MaybeUninit<ManuallyDrop<T>>`-style non-`Copy` storage, explicit MIR-directed Drop
  glue with no automatic Rust `Drop`, isolated unsafe helpers only, Cargo invoked internally by
  `stark build`, local/pinned generated dependencies, Native Provider ABI v0.1 specified but not
  required to execute in the MVP). Status flipped `PROPOSED` → `APPROVED` in the entry-plan
  document itself. Outstanding before WP-C5.1a code lands: name the frozen C5 reference workspace
  (§4), record its green HIR/MIR baseline snapshot, and record the first host target and Rust
  toolchain versions — these are execution-time deliverables of WP-C5.1a/b, not additional
  approval gates.

- CD-043 [2026-07-21, WP-C5.1a, owner decision] **C5.1a representation decision closed: exact
  `MirTy` matrix enumerated, host target for the first native proof pinned to BOTH
  `aarch64-apple-darwin` (primary/local) and `x86_64-unknown-linux-gnu` (secondary/CI), not a
  single target as the entry plan's default allowed.** Full record: `STARKLANG/docs/compiler/
  work-packages/WP-C5.1.md`. The `MirTy` matrix (enumerated against `starkc/src/mir/mod.rs` and
  `starkc/src/hir.rs::CoreType`) marks IN: all integer/float/`Bool`/`Char`/`Unit`/`Never`/`Str`/
  `String` primitives, `Struct`, user `Enum`, `Option`/`Result` (and structurally `Ordering`),
  `Tuple`, `Array`, narrow `Ref`, `FnPtr`; marks OUT by default: `Slice`, and every
  `Core(CoreType::*)` payload except that `String`/`Option`/`Result`/`Ordering` never actually
  route through `MirTy::Core` (they lower to `MirTy::String`/`MirTy::Enum` directly) — so the real
  OUT set is `Vec`, `Box`, `HashMap`/`HashSet`, `Range`/`RangeInclusive`, all iterator `CoreType`s,
  `Random`, `IOError`/`File`. **Scope consequence recorded for C5.4d:** the frozen reference
  workspace's required "a loop" (§4.1) must be a `while`/array loop, not a `for x in a..b` range
  loop or Vec/HashMap iteration, since every range/iterator `CoreType` is OUT unless a minimal path
  is separately approved first. Owner chose the dual-target option over a single first-proof
  target specifically to avoid a later cross-platform retrofit, matching the project's existing
  dual-toolchain-version validation habit (1.93/1.97). Non-`Copy` storage, move/Drop invariants,
  enum/`Option`/`Result` representation, function-pointer representation, and the layout-query rule
  are all confirmed against the already-approved §6–10 (CD-042) with no changes. WP-C5.1a CLOSED;
  next is WP-C5.1b (backend/runtime skeleton).

- CD-044 [2026-07-21, WP-C5.1b] **Backend/runtime skeleton delivered; empty `fn main() { }`
  compiles and runs as a real native executable — the C5.1b proof, and the project's first
  generated-Rust output that is not a disposable spike.** Full record: `STARKLANG/docs/compiler/
  work-packages/WP-C5.1.md` §C5.1b. New workspace member `starkc/stark-runtime/` (dependency-free,
  §11.3); `starkc/src/backend/{mod,version}.rs` +
  `starkc/src/backend/generated_rust/{mod,emit_program,emit_types,emit_bodies,emit_places,
  emit_runtime,mangle,source_map,build}.rs`. Real logic lands in `output.rs`/`version.rs`
  (runtime) and `emit_program`/`emit_types`/`emit_bodies`/`mangle`/`build` (backend); `trap.rs`/
  `value.rs`/`provider_abi.rs` (runtime) and `emit_places.rs`/`emit_runtime.rs`/`source_map.rs`
  (backend) are doc-only placeholders by design (§5.1: "a responsibility map, not a requirement to
  create every file immediately") — nothing is hidden behind them, there is simply nothing to
  lower yet at C5.1b's scope. Entry point discovered via the literal symbol `"main@[]"`, the same
  convention `mir::interp::run_program` already uses (kept identical, not reinvented, per §5.2).
  Test: `starkc/tests/native_c5_1b_skeleton.rs::empty_main_compiles_and_runs_natively` — full
  pipeline (parse→resolve→typecheck→lower→verify→`emit_native_debug`→`cargo build
  --offline`→run), asserts exit 0 and empty stdout. **Proven on the primary target
  (`aarch64-apple-darwin`) this session; the secondary target (`x86_64-unknown-linux-gnu`) is
  proven by the next CI run — no separate CI job was needed since the test runs under the
  existing `cargo test --workspace --all-targets --all-features` step.** Validation: `cargo fmt`
  clean, `cargo clippy -D warnings` clean, full workspace suite green (0 failures across ~1050
  lines of test output), `cargo test --test exec_snapshots` green (4/4) — the C3-ENTRY CI
  baseline is unaffected by the new workspace member. One real defect found and fixed during
  bring-up (not a DEV#, an in-WP implementation correction, not a semantic defect): the initial
  `emit_trivial_unit_body` assumed a body has exactly one block; the real lowered MIR for an
  empty `main` has two (`bb0` real, `bb1` a synthetic dead `Unreachable` block from WP-C4.5's
  return-slot elaboration) — fixed to read `body.entry` specifically and require every other
  block be trivially dead, discovered by dumping real MIR rather than assumed. WP-C5.1b CLOSED;
  next is WP-C5.1c (Native Provider ABI v0.1 specification).

- CD-045 [2026-07-21, WP-C5.1c] **Native Provider ABI v0.1 document DRAFTED (status `PROPOSED`)
  with a compile-time validator and mock fixtures delivered; owner CE4 review of the document's
  technical content is still open — this is NOT a closure entry.** CD-042 approved *writing* a
  v0.1 ABI document as one of `WP-C5-ENTRY.md`'s recommended §19 choices; it did not pre-approve
  this document's actual design, which is new substantive content drafted in this WP (the same
  distinction WP-C4.1's `mir.md` draft-then-CE3-review-then-CD-028-approval sequence already
  established as the pattern for this project — a design document is not self-approving just
  because writing one was authorized). Full record: `STARKLANG/docs/compiler/
  native-provider-abi-v0.1.md` (17/17 of §10.1's required points covered) and `STARKLANG/docs/
  compiler/work-packages/WP-C5.1.md` §C5.1c. Delivered: the document itself; real `#[repr(C)]`
  ABI types in `starkc/stark-runtime/src/provider_abi.rs` (`ResourceHandle`, `BorrowedBuffer`,
  `BorrowedBufferMut`, `ProviderStatus`); a compile-time metadata validator in `starkc/src/
  backend/provider_abi.rs` (`validate(&ProviderMetadata) -> Result<(), Vec<AbiViolation>>`,
  returns every violation found, not just the first, matching the MIR verifier's own convention);
  a fictional illustrative `example-kv` mock provider plus 6 deliberately-invalid fixtures, one
  per violation class — 7/7 tests pass. No provider feature expansion beyond the document +
  validator + fixtures (§10.2): no dynamic loading, no real `extern "C"` linkage, no file/network
  provider implementation. **One cross-reference defect found and fixed before this entry was
  written, not after:** the document's own §10.1-point citations drifted during drafting (three
  headings cited the wrong point number against the entry plan's 17-item list — §10 cited "point
  16" instead of 17, §15 cited "points 14 and 15" instead of "14 and 16", §16 cited "point 14"
  instead of 15); caught by a deliberate grep-and-recount sweep against the source list before
  commit, not by the owner. Validation: `cargo fmt`, `cargo clippy -D warnings`, full workspace
  suite, and `exec_snapshots` all green. **WP-C5.1c: document/validator/fixtures DELIVERED; the
  design itself awaits owner CE4 review before WP-C5.1 overall can close** (provider execution is
  not required for the C5 MVP, so this blocks only the design-review checkbox, not
  implementation).

- CD-046 [2026-07-21, owner CE4 decision] **Native Provider ABI v0.1 (`STARKLANG/docs/compiler/
  native-provider-abi-v0.1.md`) APPROVED AS DRAFTED, no changes required.** Closes the review gate
  CD-045 opened. Owner reviewed the document's actual technical choices — the C-ABI-idiom error
  convention (§11: status code + out-parameters, chosen to avoid a hand-rolled unsafe tagged
  union), the no-borrowed-handle-in-v0.1 decision (§8), and the closed `AbiType` vocabulary (§6/
  §10) as the single mechanism enforcing both the callback prohibition and the
  no-generated-Rust-aggregate-crossing rule — and approved as drafted, the same draft-then-CE4-
  review outcome `mir.md` reached under CD-028 (there: approve-with-required-changes; here:
  approve outright). Document status flipped `PROPOSED` → `APPROVED`. **WP-C5.1c CLOSED; WP-C5.1
  (Runtime ABI and Layout Design) CLOSED in full — all of C5.1a/b/c done.** Per `WP-C5-ENTRY.md`
  §14's exit checklist: CE4 decision recorded (CD-042 representation contract + CD-046 provider
  ABI), one verified empty/scalar MIR program is a standalone executable on both pinned targets,
  runtime/backend/compiler version checks demonstrated, no language semantics hidden in the
  runtime. Next: WP-C5.2 (scalar native lowering) — primitive values/constants (C5.2a), locals/
  places/copies/moves (C5.2b), operations/control flow (C5.2c), direct functions/calls (C5.2d),
  trap path (C5.2e).

- CD-047 [2026-07-21, WP-C5.2a] **Constant emission delivered — `emit_types::emit_constant`
  covers every primitive `Constant` variant.** Full record: `STARKLANG/docs/compiler/
  work-packages/WP-C5.2.md` §C5.2a. `Bool`/`Unit` direct; `Int` with the integer-suffix reused
  from `emit_ty`; `Int(codepoint, MirTy::Char)` (the `Char` constant's actual MIR encoding, per
  `mir::lower`'s f-3b) reconstructed via `char::from_u32(...).unwrap()` since Rust has no `char`
  literal suffix; `Float` via `f64`'s `Debug` formatting (guaranteed round-trip, always a decimal
  point/exponent so it parses back as a float literal) with `NaN`/`Infinity`/`-Infinity` handled
  as named `f64::` constants since they have no Rust literal syntax. **Real bug caught before
  commit:** the first version unconditionally appended an `f64` suffix, producing invalid
  `f64::NANf64` for the NaN case — caught by the test harness (every emitted expression is
  round-tripped through a real `rustc --edition 2021 --crate-type lib` parse/typecheck, not just
  string-shape-asserted), fixed by making the NaN/Infinity branches return an already-fully-typed
  expression the caller does not re-suffix. 5/5 tests pass. Validation: `cargo fmt`, `cargo
  clippy -D warnings`, full workspace suite, `exec_snapshots` — all green. **Process note:** the
  owner flagged that running the full workspace suite after every small change was slowing
  development; going forward, scoped `cargo test --lib`/`--test <file>` runs during iteration,
  full-suite runs reserved for WP/gate closure points (recorded for future sessions in memory,
  not just here). WP-C5.2a CLOSED; next is WP-C5.2b (locals/places/copies/moves).

- CD-048 [2026-07-21, WP-C5.2b] **Real locals/places/assignments/copies delivered —
  `emit_body` (renamed from and fully replacing C5.1b's `emit_trivial_unit_body`) declares every
  body local and lowers `Use`-rvalue assignments; `emit_place` supports bare locals.** Full
  record: `STARKLANG/docs/compiler/work-packages/WP-C5.2.md` §C5.2b. Locals declared `let mut _N:
  T;` uniformly (uninitialised, `mut` regardless of reassignment — cheap given the generated
  file's blanket `#![allow(unused)]`, and leaving them genuinely uninitialised means a
  lowering-bug read-before-write is caught by rustc's own definite-assignment analysis, not
  silently given a fabricated default). `Operand::Copy`/`Operand::Move` both emit the same bare
  place reference — sound because `emit_ty` only admits primitive `MirTy`s and every primitive is
  `Copy` by construction; real non-`Copy` move/liveness tracking stays deferred to WP-C5.3+. The
  entry's Unit-return check moved from inside the body emitter to `emit_program.rs` specifically
  (a Rust-`fn main()` constraint, not a general body-emission one), so `emit_body` stays reusable
  for an arbitrary-return-type function once WP-C5.2d lifts the single-body-program restriction.
  Two new end-to-end native tests (`native_c5_2b_locals.rs`: real `Int32`/`Bool`/`Char`/
  `Float64`/`UInt8` locals + a copy; separate `Float32`/`Float64` locals) plus the existing
  `native_c5_1b_skeleton.rs` empty-`main` proof re-run unchanged as a regression check that the
  generalized emitter still handles the C5.1b shape. One STARK-level (not backend) snag caught
  writing the test: an unsuffixed `2.5` float literal defaults `Float64` and does not coerce to a
  `Float32`-typed `let` (`E0001`) — fixed in the test source. Validation: `cargo fmt`, `cargo
  clippy -D warnings`, scoped tests (`backend::` 16/16, new test 2/2, regression 1/1),
  `exec_snapshots` 4/4 — full workspace suite not re-run this WP, per the new test-run-frequency
  policy (last green at WP-C5.2a; this WP's changes are additive and narrowly scoped to
  `backend::generated_rust`). WP-C5.2b CLOSED; next is WP-C5.2c (operations and control flow).

- CD-049 [2026-07-21, WP-C5.2c] **Real operations and arbitrary control flow delivered —
  arithmetic (with correct overflow/div-by-zero/shift trapping), comparisons, bitwise ops,
  `if`/`else`, and `while` loops now compile and run natively, matching `mir::interp::eval_checked`
  (the oracle) exactly.** Full record: `STARKLANG/docs/compiler/work-packages/WP-C5.2.md`
  §C5.2c. `emit_bodies.rs` restructured to a block-index dispatch loop (`let mut __bb: u32 =
  <entry>; loop { match __bb { 0 => {...}, ... } }`) — the standard technique for emitting an
  arbitrary MIR basic-block graph without recovering structured `if`/`while` shapes, since Rust
  has no `goto`; `Goto`/`SwitchInt` both reduce to `__bb = target; continue;`, so loops need no
  special-casing versus branches. Checked ops widen to `i128`, use Rust's native `checked_*`,
  then range-filter against the DESTINATION type — provably equivalent to native narrow-width
  checked arithmetic for `Add`/`Sub`/`Mul`/`Div`/`Rem`/`Neg`/`Pow`, but NOT optional for `Shl`
  (native `checked_shl` only validates the shift count, silently dropping overflowed bits, which
  would violate STARK's always-trap semantics for left-shift overflow specifically). Trap
  categories read directly from the terminator's own `TrapInfo` rather than re-derived, matching
  `mir::interp`'s own "terminator's category, with the `Shl`/`Shr` bad-count `InvalidShift`
  override" rule exactly. New `stark_runtime::trap::abort_minimal` is an explicitly MINIMAL,
  not-yet-final abort (stderr category + nonzero exit) — the real trap ABI (source spans, §13.2
  canonical format) stays WP-C5.2e's job; this exists now only because "overflow and silently
  continue" would be unsound to leave unimplemented. **Real soundness bug caught and fixed before
  commit, not cosmetic:** WP-C5.2b's "leave locals uninitialised, let rustc's definite-assignment
  analysis catch a lowering bug" strategy silently breaks the moment a body has more than one
  block — rustc treats each `match __bb { N => {...} }` arm as an independent branch of one
  ordinary match with no notion that arm 1 only runs after arm 0 already assigned a local (that
  fact lives in data flowing through `__bb`, invisible to rustc across `continue`). The first
  real multi-block test programs failed to compile with `E0381` immediately, not hypothetically;
  fixed by default-initialising every local (`emit_types::default_value_expr`), the standard fix
  for this codegen pattern, trading away C5.2b's "free" lowering-bug-catch property (MIR's own
  V-MOVE-1 verifier remains responsible for that instead) — WP-C5.2b's own record was revised to
  say so rather than left stale. Five new end-to-end native tests
  (`native_c5_2c_operations.rs`: full arithmetic/comparison suite, an `Int32` overflow trap, a
  division-by-zero trap, `if`/`else`, a `while` loop to 5) plus the C5.1b/C5.2b proofs re-run
  unchanged as regressions. Validation: `cargo fmt`, `cargo clippy -D warnings`, scoped tests
  (`backend::` 16/16, new test 5/5, prior regressions 3/3), `exec_snapshots` 4/4 — full workspace
  suite not re-run per the test-run-frequency policy. WP-C5.2c CLOSED; next is WP-C5.2d (direct
  functions and calls).

- CD-050 [2026-07-21, WP-C5.2d] **Multi-function programs, real parameters, and direct calls
  delivered — `emit_program.rs`'s single-body restriction (present since WP-C5.1b) is lifted.**
  Full record: `STARKLANG/docs/compiler/work-packages/WP-C5.2.md` §C5.2d. Every body in
  `program.bodies` is emitted as its own Rust item (`lower_program`'s own doc comment already
  guarantees the set is self-contained and transitively-reachable, so no separate linking logic
  was needed); the entry instance stays specially wrapped as Rust's literal `fn main()` with the
  version-check prologue, every other body goes through new `emit_bodies::emit_function`.
  `emit_param_list` maps each `body.params[j]` to the local whose `LocalKind` is `Param(j)` (a
  local's position and its parameter index are NOT the same number) and emits it as a `mut` Rust
  parameter under that local's own `_N` name, so ordinary statement emission needs no
  special-casing to read a parameter. `Terminator::Call` with `Callee::Instance` lowers to an
  ordinary Rust call, using `mangle::function_name_for_symbol` as the one naming authority for
  both defining and calling a function (entry symbol → `main`, everything else → its sanitized
  form) rather than two conventions that could drift apart. `Callee::FnValue`/`Callee::Runtime`
  stay deferred to WP-C5.4c and wherever the first `RuntimeFn` group lands, respectively. **No
  bug this time** — unlike C5.2b/c, the one real hazard this WP's design raised (declaring a
  `Param`-kinded local a second time in the block body would silently shadow the real argument
  with a fabricated default) was caught in review before writing the test (`emit_block_body`'s
  default-init loop explicitly `continue`s past `Param`-kinded locals), not discovered by a
  failing build. Two new end-to-end native tests (`native_c5_2d_calls.rs`: a two-parameter `add`
  call, and a three-parameter `clamp` helper feeding an `if` plus a second `Float64`/`Bool`
  helper) passed on the first run, plus the C5.1b/C5.2b/C5.2c proofs re-run unchanged as
  regressions. Validation: `cargo fmt`, `cargo clippy -D warnings`, scoped tests (`backend::`
  18/18, new test 2/2, prior regressions 8/8), `exec_snapshots` 4/4 — full workspace suite not
  re-run per the test-run-frequency policy. WP-C5.2d CLOSED; next is WP-C5.2e (trap path).

- CD-051 [2026-07-21, WP-C5.2e] **Real trap ABI delivered — every checked-operation trap now
  reports its category and an exact source file/line on stderr, exit code 101 (matching `stark
  run`'s own established convention).** Full record: `STARKLANG/docs/compiler/work-packages/
  WP-C5.2.md` §C5.2e. `stark_runtime::trap::abort(category, file, line, column) -> !` replaces
  C5.2c's `abort_minimal` placeholder outright. Source location is resolved at COMPILE TIME
  (`SourceFile::line_col` against `MirProgram::files`, both already available to the backend) and
  baked into the generated call site as literals — a documented, deliberate simplification of
  §13.1's compact-span-ID-plus-runtime-lookup-table design (that design exists to deduplicate
  span data for large programs; inlined literals are simpler and exactly as correct at MVP
  scale), not an oversight. `emit_abort_call` is the one place that assembles a trap-abort call,
  used for both a terminator's default category and the `Shl`/`Shr` `InvalidShift` override, so
  the two trap sites within one checked operation cannot independently drift. Category messages
  are NOT claimed to match the HIR interpreter's own ad hoc per-call-site strings byte-for-byte —
  no canonical table exists there to match, and the differential comparator (§15.1) checks
  category plus source file/line, not stderr text. C5.2c's own two trap tests were retrofitted
  from a loose `assert_ne!` to the exact `assert_eq!(status, Some(101))` now that the precise
  contract exists. Four new tests (`native_c5_2e_traps.rs`): an overflow trap asserting an EXACT
  `file:line` match (not a loose check), plus division-by-zero/invalid-shift/cast-failure each
  asserting category message and exit code. Validation: `cargo fmt`, `cargo clippy -D warnings`,
  scoped tests (`backend::` 18/18, new test 4/4, all prior native regressions including the two
  retrofitted), `exec_snapshots` 4/4. **WP-C5.2e CLOSED. WP-C5.2 (scalar native lowering) is
  NOT YET claimed closed**: §14's exit condition explicitly requires three-engine (HIR/MIR/
  native) automated agreement, and every `native_c5_2*.rs` test to date asserts on the native
  engine's own output in isolation, not an automated diff against the other two engines the way
  `mir_differential.rs` already does for HIR-vs-MIR. This gap is recorded here deliberately
  rather than treated as satisfied by "native looks right" reasoning. Building the three-engine
  differential harness (§15.1/§15.2) is the next open decision — whether it lands as a C5.2-
  closing addendum or defers to WP-C5.6 (which already co-owns cross-backend snapshot replay per
  the WP-C4.4/CD-018 carry-forward) is for the owner to decide, not resolved here.

- CD-052 [2026-07-21, WP-C5.2 review response] **External review of head 37828a07 raised seven
  findings; all seven verified as REAL against the code (no false positives). Four fixed here
  (DEV-091/092/093/094), one recorded as a C5.3 opening condition (DEV-095), two escalated to the
  ABI's owner as a CE4 amendment.** Writing the regression tests for the first finding surfaced an
  eighth, previously unknown defect (DEV-096) that the review did not name.

  - **DEV-091 — float→int casts accepted out-of-range values at 64-bit widths, in BOTH the MIR
    interpreter and the native backend. FIXED.** Both compared the truncated value against
    `max as f64`, which ROUNDS UP at 64-bit widths: `u64::MAX as f64` is 2^64 and `i64::MAX as
    f64` is 2^63. Exactly 2^64 therefore passed the guard, and the subsequent saturating `as`
    clamped it to `u64::MAX` — silently producing a value where 03-Type-System.md requires a
    trap. Same defect at 2^63 for `Int64`. Fixed in both engines with a half-open test against an
    EXACT bound: every `max + 1` is a power of two and so exactly representable as `f64`
    (`mir/interp.rs`'s `Cast` arm; `emit_bodies.rs`'s new `int_float_bounds_tokens`, deliberately
    separate from `int_bounds_tokens`, whose inclusive pair remains correct for the exact-`i128`
    checked-arithmetic path). The HIR ORACLE was already correct here — it truncates to `i128`
    and range-checks in exact integer arithmetic — so this was a genuine engine divergence, not a
    shared misreading of the spec. The reason it survived: no corpus or inline case had ever
    exercised a 64-bit cast boundary. Seven new boundary cases in `mir_differential.rs` (2^64,
    greatest f64 below 2^64, 2^63, greatest below 2^63, -2^63 inclusive, below -2^63, truncation
    ordering) plus three native ones in `native_c5_2c_operations.rs`.
  - **DEV-092 — symbol sanitization was not injective, while its own doc comment asserted that
    it was. FIXED.** `sanitize_symbol` hex-encoded disallowed bytes as `_hh` but passed `_`
    through unchanged, so encoded output was indistinguishable from source text that already
    spelled an escape: `pkg::f` and a legally-named STARK function `pkg_3a_3af` both encoded to
    `stark_pkg_3a_3af...`. Reachable from ordinary source, because `key_symbol` puts a
    `::`-joined module/package path in every symbol, and materially relevant since C5.2d, where
    every MIR body became its own Rust function. Fixed by making `_` the escape introducer and
    escaping it as `__`; the encoding is now decodable, hence injective, and stays readable
    (`my_fn` → `my__fn`) rather than hex-encoding every byte. Tests: a pairwise-distinctness
    sweep over 17 adversarial symbols (`::`/`_3a` at package and module boundaries, `@`/`_40`,
    `[`/`_5b`, literal-vs-escaped underscores, the `, ` type-argument separator, and non-ASCII
    identifiers) plus a round-trip-through-a-decoder test that states injectivity directly rather
    than sampling for collisions.
  - **DEV-093 — native success-path tests observed no computed values. FIXED.** The arithmetic,
    branch, loop and direct-call tests computed results and asserted only `exit == 0`; a backend
    returning zero from every function would have passed most of the suite. All success-path
    tests now assert IN the STARK program via `assert_eq`/`assert` (native `println` is still
    WP-C5.3), covering every arithmetic result, both branch directions, loop trip count AND body
    effect, zero-iteration loops, call return values, and parameter order. This required
    implementing `Terminator::Trap` in the backend (message-less form — what `mir::lower` emits
    for `assert`/`assert_eq`/`assert_ne`), which was still `Unsupported` at CD-051 and is
    properly WP-C5.2e's own deliverable; `Trap` carrying a user `&str` message remains WP-C5.3.
    A NEGATIVE CONTROL (`a_false_assertion_traps_natively`) proves a false assertion really does
    reach the trap ABI and exit 101 — without it, "exit 0" would remain ambiguous between
    "assertions held" and "assertions compiled away".
  - **DEV-094 — the version-mismatch message named the wrong version on each side. FIXED.**
    `version::check` assigned the LINKED runtime's `RUNTIME_VERSION` to `expected_runtime_version`
    and the generation-time recorded value to `actual_`, while the generated crate prints them as
    "generated for runtime {expected}, linked against {actual}". Fixed at the source (the field
    assignment, not the message) so the names read correctly for any future consumer, with a test
    that pins the field-to-side assignment rather than merely that a mismatch is detected.
  - **DEV-095 — the generated-crate build key omits nominal type context and the Drop map.
    RECORDED as a WP-C5.3 opening condition, NOT fixed here.** `compute_build_key` hashes
    `program.dump()`, and `dump()` emits only the version header and bodies; the MIR contract
    states the nominal type context and destructor map are in-memory parts of the compilation
    unit that the textual dump does not serialize. Changing a struct's fields or its `Drop`
    metadata could therefore leave the build key unchanged and silently reuse a stale generated
    crate. This CANNOT bite before aggregates and Drop exist, which is exactly WP-C5.3, so it is
    a C5.3 entry condition rather than a C5.2 defect: before aggregates land, build identity must
    cover a deterministic encoding of the nominal type context, the Drop implementation map, the
    source table, package graph identity, the entry instance, all bodies, and the backend/
    runtime/toolchain versions.
  - **DEV-096 — the HIR oracle reported every out-of-range cast as an ARITHMETIC OVERFLOW trap,
    at every width. FIXED. Not named by the review; found by DEV-091's new boundary tests, which
    failed on category mismatch rather than on the bound.** Both cast arms in `interp.rs`
    (int→int and float→int) routed through `check_integer_range`, whose message is hardcoded
    `"integer overflow"`, so the oracle disagreed with the MIR interpreter and the native backend
    — both of which classify a failing cast as `TrapCategory::CastFailure` — for every
    out-of-range cast, not merely at 64-bit boundaries. 03-Type-System.md enumerates overflow and
    failing `as` casts as DISTINCT always-trap causes, and the oracle's own non-finite float case
    already used the cast-specific message, so this was an implementation artifact of a shared
    helper rather than a semantic question. Split into `check_cast_range` (cast failure) and
    `check_integer_range` (overflow) over one shared width predicate, so the two can never drift
    on WHICH values are in range while differing, correctly, on which trap they raise. Two
    narrow-width regression tests pin the category independently of any float rounding.
  - **Escalated to the owner as a CE4 amendment, NOT changed here** (the Native Provider ABI
    v0.1 is owner-approved under CD-046, so amending it is the owner's decision):
    `STARKLANG/docs/compiler/native-provider-abi-v0.1-CE4-amendment-1.md` documents two
    contradictions between the approved document and its own validator — the return-shape
    contradiction (§11 says every provider function returns `ProviderStatus` with results via
    out-parameters, but `FunctionDecl` has `returns: AbiType` with no out-parameter
    representation, and the validator's own "valid" fixture has `kv_open` returning
    `ResourceHandle`), and `ResourceHandle` deriving `Clone`/`Copy` against §12's exclusive-
    ownership and close-exactly-once rules. Both are cheap to correct now because no provider
    executes in the C5 MVP; neither is corrected without owner sign-off.
  - **Also observed, not filed as defects**: no integer literal above `Int64::MAX` is expressible
    (an unsuffixed literal types as `Int64` first, so even `let x: UInt64 = 18446744073709549568;`
    is rejected), `Int64::MIN` has no literal spelling, and an unsuffixed literal in argument
    position does not receive expected-type propagation from a sibling argument. These shaped how
    the boundary tests are written (documented at the test) but are pre-existing front-end
    behaviours unrelated to native lowering.
  - **The review's one process observation did NOT hold up.** It reported that the "CI green"
    claim was unverifiable because its GitHub connector exposed no workflow run for head
    37828a07. `gh run list` shows the `CI` workflow completed with conclusion `success` on
    37828a07 (and on 5af7ad7/56b5202/c9eaa53 before it), so the claim was accurate and the gap was
    in the connector's visibility, not in the evidence. Worth recording for its own reason,
    though: CI was green on the very commit carrying DEV-091's semantic defect. `fmt`, `clippy`
    and the full workspace suite all passed because **no test exercised a 64-bit cast boundary** —
    a green pipeline bounds the risk to what the corpus covers, and this pass is a direct
    demonstration of that limit.
  - Validation: `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
    --all-features -- -D warnings` clean, `mir_differential` 132/132 (up from 123 — the frozen
    corpus plus nine new cast cases: seven boundary, two category), all five `native_c5_*` suites
    green (19 tests, up from 13), `exec_snapshots`/`conformance`/`gate3_execution` green, and the
    full workspace suite green.

- CD-053 [2026-07-21, WP-C5.2 closure + CE4 amendment direction] **Owner directive, four parts:
  build the three-engine differential harness NOW as the WP-C5.2 closure addendum (not deferred to
  WP-C5.6); do NOT approve CE4 Amendment 1 as submitted — revise and resubmit before either
  `provider_abi.rs` changes; keep the ABI version at `0.1`; keep DEV-095 (build-key completeness)
  as a mandatory WP-C5.3 opening condition.** All four executed.

  - **Part 1 — the three-engine differential harness. BUILT; WP-C5.2 CLOSED.**
    `starkc/tests/three_engine_differential.rs` implements `WP-C5-ENTRY.md` §15.1's **three-engine
    pipeline**, comparing traps in **normalised** form for C5.2 (raw stderr byte equality is NOT
    compared — the HIR oracle has no canonical stderr format to compare against, only ad hoc
    per-call-site strings; what is compared is what those bytes mean, i.e. category plus exact
    file/line/column): one source string per case, run through the HIR interpreter (oracle), the MIR
    pipeline (lower → verify → execute) and the native binary (lower → verify → emit → cargo build
    → run), each result **normalised into one common `Outcome`** — `Completed { stdout, exit }` or
    `Trapped { category, file, line, column, stdout_before }` — and all three required equal. The
    normalisation is the substance: the oracle raises prose plus a byte span, MIR raises a
    `TrapCategory` plus a `SourceInfo`, and the native binary writes stderr text and a process exit
    code, so agreement is only mechanically checkable once all three are projected onto one type.
    Compared per case: completion-vs-trap, exit status, trap category, exact trap file/line/column,
    and observable output. 20 tests, all green.
    - Coverage against §14's six required dimensions: scalar arithmetic (all operators, widths,
      precedence, negative-operand division/remainder, `Float64`); branches (both directions of
      each `if`/`else`, an `else if` chain taking middle and final arms, nested, no-`else`, `if`
      as an expression, `&&`/`||`/`!`); loops (zero-iteration in two shapes; accumulate,
      `continue`, `break`, nested); direct calls (multi-function, argument order via a
      non-commutative callee, no-arg, `Unit`-returning, nested-call arguments, recursion, call in
      a loop); successful checked operations (arithmetic landing exactly on `Int32::MAX`/`MIN`,
      shift counts at width-1, in-range casts at the narrower type's exact boundary, widening,
      int↔float); and every admitted trap category (`IntegerOverflow`, `DivideByZero` for both `/`
      and `%`, `InvalidShift`, `CastFailure`, `AssertFailure` for both `assert_eq` and bare
      `assert`). `IndexOutOfBounds`, `UnwrapNone`/`UnwrapErr` and message-carrying `Panic` are not
      reachable from the C5.2 surface and the oracle-normalisation function panics explicitly on
      them rather than guessing.
    - CD-052 regressions re-pinned as three-engine agreement rather than per-engine assertions:
      **DEV-091** (four cases — in-range boundary conversions, exactly 2^64 → `UInt64`, exactly
      2^63 → `Int64`, first f64 below `Int64::MIN`; both sides of every bound), **DEV-096** (a
      case only a category comparison can hold, since all three engines exit 101 either way),
      **DEV-092** (the source-level consequence, not just the encoding: `mod m { pub fn f() }`
      versus a top-level `fn m_3a_3af()` — one Rust identifier under the old encoding — with both
      called and both return values observed), and the **negative control** proving a false
      assertion really does fail the run in all three engines, without which every
      assertion-observed completing case would be decorative.
    - **Mutation-tested before being trusted.** A comparator that passes proves nothing until it
      has been shown to fail. Two mutations were injected into the native backend and reverted:
      `checked_add` → `checked_sub` (result: `MIR/NATIVE DISAGREEMENT`, MIR `Completed` vs. native
      `Trapped { AssertFailure }` — the value dimension is live) and native trap `line` → `line +
      1` (result: same category and file, line 4 vs. 5 — the location dimension is live,
      independently of category). `git diff` confirms neither survives.
    - Honest handling of the output dimension: native `println` is `Unsupported` until WP-C5.3, so
      values are observed through in-program `assert`/`assert_eq`. Rather than quietly excluding
      stdout from the comparison, `NATIVE_STDOUT_SUPPORTED: bool = false` gates a precondition
      **enforcing** that every case is output-free, which is what makes full three-way `Outcome`
      equality total. Flipping that constant when native output lands drops the precondition and
      starts comparing real bytes, with no other change.
    - One production change only: `stark_runtime::trap::TrapCategory::message()` became `pub`, so
      the harness normalises native stderr against the runtime's own category table instead of a
      second copy in a test file that would drift the first time a message's wording changed.
    - Per-engine tests (`native_c5_2*.rs`, `mir_differential.rs`) remain and remain useful, but
      per the owner's direction they are **supplementary** and do not themselves satisfy §14. What
      stays with WP-C5.6 is cross-backend replay of the frozen `exec_snapshots` corpus (the
      WP-C4.4/CD-018 carry-forward); what moved out of it is the comparator.

  - **Part 2 — CE4 Amendment 1 NOT approved as submitted; revised and resubmitted.** The owner
    approved five principles (every physical provider function returns `ProviderStatus`; result
    values travel through explicit output channels; the owning resource representation is not
    `Clone`/`Copy`; a raw C-compatible `Copy` handle may remain inside the isolated FFI boundary;
    the owning wrapper must NOT implement Rust `Drop` — verified MIR keeps the exactly-once close
    obligation) and named four issues revision 1 omitted. Revision 2
    (`STARKLANG/docs/compiler/native-provider-abi-v0.1-CE4-amendment-1.md` revision 2) resolves all four:
    (a) `BorrowedBuffer`/`BorrowedBufferMut` are borrowed call-duration views, so §8's
    ownership-transfer language is corrected to cover handles only — as written it made *reading
    the buffer you just passed to `kv_get`* a use-after-transfer; (b) the v0.1 prohibition on
    borrowed handles is lifted, because consuming-only handles made §17's own mock provider
    unexpressible (`kv_get` would consume the store it reads); (c) every handle parameter and
    handle output names its declared resource type, so the validator can enforce §13's
    wrong-resource-type rule it currently cannot see; (d) direction and ownership are separated —
    revision 1's `Direction × AbiType` product is **rejected**, since of its 15 combinations six
    are meaningful, three are one case spelled three times, and the distinction that matters
    (borrowed vs. consumed handle) is the one it cannot express. Replaced by a closed `AbiParam`
    enum over exactly the seven owner-enumerated forms, plus a `RawResourceHandle`
    (`Copy`, boundary-only) / `OwnedResourceHandle` (non-`Copy`, non-`Clone`, no `Drop`) split, a
    close-function rule requiring exactly one consumed handle of the declared type and no ordinary
    value output, two new violation classes, and a corrected `valid_example_kv` fixture. One
    discretionary reading is flagged for the owner rather than assumed (may a close function take
    additional pure inputs?). **Neither `provider_abi.rs` changes until revision 2 is approved.**

  - **Part 3 — ABI version stays `0.1`.** Nothing has shipped or executed against this ABI, so
    correcting a pre-execution contract is an amendment, not a version bump. Recorded as CE4
    Amendment 1 to v0.1.

  - **Part 4 — DEV-095 confirmed as a mandatory WP-C5.3 OPENING condition.** WP-C5.3 may not begin
    aggregate or Drop-bearing native generation until every semantic input affecting generated
    code — nominal type context and the Drop map included — is in the build key and covered by
    cache-invalidation tests. Recorded in Follow-ups as a blocking entry condition, not a
    to-do.

  - Validation: `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
    --all-features -- -D warnings` clean, `three_engine_differential` 20/20, `mir_differential`
    and all five `native_c5_*` suites green, and **`cargo test --workspace` green: 884 passed /
    0 failed / 2 ignored across 52 test binaries**.
    **Correction to the figure first recorded here (818 across 40 binaries):** that was an
    undercount of the *same* green run, not a different result — the background capture of that
    run lost its first 24 lines to output buffering, so 12 suites never reached the tally. Caught
    by re-running with a complete capture and noticing the suite count disagreed. Recorded rather
    than quietly overwritten, because "the number moved and nobody said why" is worse than the
    original error.

- CD-054 [2026-07-21, CE4 Amendment 1 approved and implemented] **The owner approved revision 2's
  design with five required changes, ruled the flagged close-function question, and directed that
  the amendment, the approved ABI document, both implementation files, the fixtures and the
  violation tests land in one commit. Done — CE4 Amendment 1 to Native Provider ABI v0.1 is
  APPROVED (ABI version stays `0.1`) and applied.**

  - **Approved from revision 2**: the closed `AbiParam` model; the fixed physical `ProviderStatus`
    return; explicit output channels; typed borrowed/consumed/output handles; borrowed buffer
    semantics; the `RawResourceHandle`/`OwnedResourceHandle` separation; owning handles being
    non-`Clone`, non-`Copy` and without Rust `Drop`; version `0.1`; the corrected example-provider
    shapes.

  - **The close-function ruling.** A close function takes **exactly one parameter** —
    `HandleConsumed { resource_type: rt }` — and nothing else. Revision 2's permissive reading
    (additional pure inputs such as a `flush: Bool` allowed) is withdrawn. The reason is
    architectural: **MIR's `Drop(place)` terminator supplies only the resource being dropped**, so
    a close function with a second parameter is one the generated code cannot call — every extra
    argument would have to be invented by the backend. The consequence is a design rule, not just
    a validation rule: any flush/completion/fallible operation needing arguments must be a
    separate provider function invoked BEFORE Drop.

  - **Four new normative rules** (amendment §4.6-§4.9, landed as ABI doc §8, §11.1, §13.2, §6.1):
    - **Consumed-handle error rule.** Ownership transfers at call ENTRY; a `HandleConsumed` value
      is dead regardless of what `ProviderStatus` reports. Ownership returning on failure would
      make a handle's liveness depend on a runtime value, so use-after-transfer could not be
      decided by MIR verification and exactly-once close would stop being a static property. An
      operation wanting ownership back on failure declares an explicit `HandleOut` (a *fresh*
      handle, not a resurrected one) or borrows instead.
    - **Output initialisation rule.** `ScalarOut`/`HandleOut` storage is uninitialised before the
      call and valid only on success: allocate through `MaybeUninit`, never read or wrap on
      failure, and validate a successful raw handle's resource type before constructing the owning
      wrapper. `ScalarInOut`/`BufferInOut` stay caller-initialised and caller-owned across the
      call. The asymmetry is the point — an `Out` slot is a promise kept only on success; an
      `InOut` slot is the caller's own memory, lent for one call.
    - **Close-failure rule.** A close function's nonzero status cannot become a recoverable
      `Result::Err`, because a `Drop` terminator has no result destination. It is a distinct fatal
      provider-close/host failure: abort without unwinding, do not retry, treat the handle as
      consumed, run no further pending Drop glue. Recoverable work (flush/commit/sync) must be a
      separate operation performed before close.
    - **Physical ABI mapping.** Every `AbiParam` variant mapped to its exact C parameter, plus the
      requirement that all raw↔owned conversions go through isolated reviewed boundary helpers,
      never generated ad hoc field access. Two pairs are physically identical and deliberately
      distinct in metadata: `ScalarOut`/`ScalarInOut` (both `*mut T`, differing in the
      initialisation contract) and `HandleBorrowed`/`HandleConsumed` (both a raw handle by value,
      differing in the ownership contract) — the C signature cannot carry either difference, which
      is exactly why the declaration must.

  - **Implemented in one commit**, per the directive: the ABI document updated (§6 rewritten, §6.1
    /§11.1/§13.1/§13.2 added, §7/§8/§10/§12/§17/§18 amended, each marked *(amended, CD-054)*);
    `starkc/src/backend/provider_abi.rs` (`ScalarTy`, `AbiParam`, `returns`-less `FunctionDecl`,
    `HandleResourceTypeUndeclared` and `CloseFunctionShape`/`CloseShapeProblem` violations, and
    the two new validator rules); `starkc/stark-runtime/src/provider_abi.rs` (the raw/owning split
    and the three boundary helpers, with resource-type validation inside `from_raw_checked` so it
    cannot be skipped by a call site that forgets it); and the fixtures rewritten to conform.
    **`example-kv` now works as an example**: `kv_open` writes its handle into a `HandleOut`,
    `kv_get` borrows the store and has somewhere to put the value it retrieves, and `kv_close`
    consumes exactly one handle. Tests: 14 in the validator module, up from 7 — five new
    negatives (an undeclared handle resource type, and one per close-shape problem: an extra
    parameter, an added output, a borrowed rather than consumed handle, a consumed handle of the
    wrong resource type) plus two new positives (ordinary operations borrow rather than consume;
    every value result is an explicit output form) — and 3 in the runtime module.

  - **What is NOT claimed.** No provider executes; §10.2's boundary is unchanged. Every rule in
    the four new sections is a statement about code that does not exist yet — the validator, the
    type definitions and the fixtures are what exist. The call-site generation that must obey the
    output-initialisation and boundary-helper rules belongs to whichever package first makes a
    provider execute. `WP-C5.1.md` records which four of its own C5.1c statements this
    supersedes, rather than being silently edited.

  - Validation: `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
    --all-features -- -D warnings` clean, **`cargo test --workspace` green: 894 passed / 0 failed
    / 2 ignored across 52 test binaries** (up from 884 — the seven new validator tests and three
    new runtime tests).

- CD-055 [2026-07-21, DEV-095 discharged — WP-C5.3 entry condition] **The generated-crate build
  key now covers every semantic input that can affect generated code, with cache-invalidation
  tests. WP-C5.3's blocking entry condition (CD-053 part 4) is DISCHARGED; aggregate and
  Drop-bearing native generation may begin.**

  - **The defect.** `compute_build_key` hashed `program.dump()` plus the eight version axes, and
    `dump()` serializes only the version header and the bodies. The MIR contract is explicit that
    the **nominal type context and the destructor map are in-memory parts of the compilation unit
    the textual dump does not serialize**. So two programs with byte-identical dumps but different
    struct fields, different enum variants, a different `Drop` impl, or different `Copy`
    classification hashed to the SAME key — and the second build would silently reuse the first's
    generated crate. Unreachable while the backend admitted only primitives; live the moment
    WP-C5.3 lands aggregates and `Drop`, which is why it was fixed before rather than after.

  - **The fix.** `build_key_input(program, versions)` builds a canonical, line-oriented encoding
    which `compute_build_key` hashes. Sections: `[versions]` (all eight axes), `[entry]`,
    `[sources]` (per-file name + SHA-256 of contents), `[types.struct_fields]`,
    `[types.enum_variants]`, `[types.drop_impls]`, `[types.copy_types]`, `[bodies]`
    (`program.dump()`, already the contract's deterministic body serialization). Determinism comes
    from the data structures themselves — `TypeContext` is `BTreeMap`/`BTreeSet` and
    `program.bodies` is sorted by canonical symbol. Tagged `build key v2` so a future encoding
    change is visibly a different scheme rather than silently colliding with v1 keys.

  - **Why the encoding is a separate function from the hash.** A test asserting "these two keys
    differ" says nothing about WHICH input made them differ; a test that can diff the encoding
    does. `the_key_input_carries_every_documented_section` pins that every section is present, so
    a section deleted from the encoder fails by name instead of quietly weakening every other test
    in the module.

  - **Coverage** (7 tests, `backend::generated_rust::build::tests`): key determinism (the baseline
    without which every "the key changed" assertion could be satisfied by a key that changes every
    time); a different body; **the DEV-095 regression** — eight one-input mutations across all
    four `TypeContext` fields (new nominal, changed field type, changed type arguments, new enum,
    reordered variants, gained destructor, changed destructor instance, became `Copy`), each
    asserting `dump()` stays byte-identical as a PRECONDITION before asserting the key changed, so
    the test is meaningless the day it stops being the actual condition; a different file name
    (names reach generated code verbatim through trap-site `file:line:column`); a source-content
    change invisible to `dump()` (an appended comment moves no span, and §11.1 requires
    source-content hashes regardless); and all eight version axes moved independently.

  - **Verified by mutation, not just by passing.** Simulating the old key (dropping the `[types]`
    sections from the hashed input) makes the regression test fail with
    `struct_fields: a new nominal: build key did not change — a stale generated crate would be
    reused`. Reverted; `git diff` confirms nothing of the simulation survives.

  - **One §11.1 item deliberately not given its own section: package graph identity.** A C5
    program is one compilation unit and the source table is its identity; when multi-package
    linkage lands (WP-C5.4) it gets its own section rather than being assumed covered. Recorded
    in the encoder's own comment so the next reader does not have to rediscover the reasoning.

  - Validation, **scoped deliberately** per the standing process note (full-workspace runs are for
    WP/gate closure points, not intermediate changes — this discharges an entry condition, it does
    not close a package): `cargo fmt --all -- --check` clean, `cargo clippy --workspace
    --all-targets --all-features -- -D warnings` clean (workspace-wide, since clippy is cheap),
    and every consumer of the changed code green — `backend::` unit tests 35/35 (including the
    seven new build-key tests) plus all six suites that invoke `emit_native_debug`, which is
    `compute_build_key`'s only caller: `native_c5_1b_skeleton` 1/1, `native_c5_2b_locals` 2/2,
    `native_c5_2c_operations` 9/9, `native_c5_2d_calls` 3/3, `native_c5_2e_traps` 4/4,
    `three_engine_differential` 20/20. Nothing outside the native build path reads the build key
    (`grep` confirms no other reference in the workspace), so the untouched suites — parser,
    lexer, formatter, LSP, ONNX, gate4/gate7 — carry no information about this change. ~15 seconds
    against ~40 minutes for the full suite.

- CD-056 [2026-07-21, WP-C5.3 opened; C5.3a closed] **WP-C5.3 opened by owner directive after
  CD-055 discharged its entry condition. C5.3a (tuples, arrays, structs) CLOSED. Two owner
  decisions are OPEN and flagged rather than resolved unilaterally; one oracle defect (DEV-097)
  was found and fixed; one scope boundary is now a named diagnostic instead of a rustc error.**

  - **Delivered (C5.3a)**: §6.2 type mapping for `Tuple`/`Array`/`Struct`; §6.3 nominal
    definitions (one Rust `struct` per type-context instance, positional `f0..fn` field names,
    `BTreeMap` order); `mangle::type_name_for_nominal` (injective, and provably disjoint from
    function names because `#` cannot occur in a STARK identifier); `emit_places::TyEnv`, the
    projection-type walk; `Rvalue::Aggregate` for all three kinds; `ConstIndex`, `CheckIndex` and
    proof-backed `Index`; `LocalKind::IndexProof`. Tuples map to **Rust tuples** — §6.2 offered
    "concrete tuple or named internal aggregate; choose one canonical form", and the Rust tuple
    needs no generated definition, no deterministic name, and no reachability walk.
    Evidence: seven new three-engine cases plus four native-only cases
    (`native_c5_3a_aggregates.rs`) for what a three-engine comparator structurally cannot cover.

  - **Why `TyEnv` exists, since it is the one structural addition**: MIR's `Projection::Field(i)`
    is ONE variant covering both struct fields and tuple elements, but generated Rust needs `.f0`
    for one and `.0` for the other. Choosing requires the projected place's type, hence a walk
    from the local's declared type through the nominal type context. It also let `operand_mir_ty`
    stop refusing projected operands, so a `SwitchInt` on a struct field or array element works.

  - **DEV-097 — the HIR oracle blamed two different columns for two ends of one bounds check.
    FIXED.** An out-of-range index trapped at the whole index expression's span; a NEGATIVE index
    trapped at the index operand's span. So the oracle disagreed with both other engines on one of
    the two, and was internally inconsistent about one check. Found by the three-engine harness's
    negative-index case; no corpus or inline case had ever indexed with a negative value. Fixed in
    `interp.rs` to use the index-expression span for both, matching MIR and native. **This is the
    fourth defect this campaign has found that lived only in the gap between engines.**

  - **OPEN DECISION 1 — what does "three-engine agreement on target layout queries" mean?**
    §14's C5.3 exit lists it, and it **cannot be satisfied as literally stated**: both
    interpreters answer **8 for every type** (`mir::interp::reference_layout`, whose own doc says
    a real per-type algorithm is the backend's job and that "a backend replaces this function and
    nothing else"), while the native engine answers its **actual Rust target layout**
    (`size_of::<Int32>()` is 4). `assert_eq(size_of::<Int32>(), 4)` traps in both interpreters and
    succeeds natively. This is not a backend defect — LAYOUT-ABI-001 makes layout target-dependent
    by design — but the exit condition needs a definition. Candidate readings: (a) the
    interpreters adopt a real layout algorithm matching the native target, which makes the
    reference oracle target-dependent; (b) agreement means agreement on RELATIONS Core guarantees,
    not absolute values; (c) layout queries are excluded from value agreement, with the divergence
    documented as intended. **Until the owner rules, the harness asserts only that layout queries
    run in all three engines and agree on completion-vs-trap, plus relations true under both
    answers.** The value question is recorded, not dropped.

  - **OPEN DECISION 2 — the §6.3-vs-§7.4 `Copy`-derive reading (implemented, reversible).** §6.3
    forbids deriving `Clone`/`Copy`/`Eq`/`Ord`/`Hash` "as a shortcut for STARK semantics"; §7.4
    says a MIR copy is emitted only for MIR-`Copy` types and the backend must not broaden that
    set. A STARK struct with an `impl Copy` needs SOME mechanism for `Operand::Copy` to read it
    twice. **Reading taken:** deriving `Clone`/`Copy` on exactly the instances MIR classifies
    `Copy` is not a shortcut — MIR decides, the derive follows, the set is neither broadened nor
    narrowed. No other trait is derived. `emit_types::mir_ty_is_copy` mirrors
    `mir::lower::is_copy` rather than asking Rust anything. If the owner reads §6.3 as forbidding
    this, the alternative is a generated copy helper per nominal and the change is confined to
    `emit_types::derives_for` plus one test.

  - **Scope boundary now a named diagnostic.** A **non-`Copy` value moved out of a local
    initialised in an EARLIER block** is refused as `Unsupported` naming WP-C5.3d. The backend
    lowers MIR's block graph to `loop { match __bb { .. } }`, so every block is one iteration of
    one Rust loop, and Rust's borrow checker cannot see that MIR never revisits a moved-from
    local — it reports "value moved here, in previous iteration of loop" for a move verified MIR
    proves sound. Found when a three-engine case passing a struct by value produced a
    `BuildFailed` carrying a rustc borrow-check error; a scope limit surfacing as a rustc error is
    itself a defect in the diagnostic. Moving WITHIN one block still works (ordinary aggregate
    construction lowers that way) and has its own test, so the guard is pinned against
    over-rejection too.

  - **OPEN DECISION 3 (blocks C5.3d) — the non-`Copy` storage strategy.** §7.2 proposes
    `MaybeUninit<ManuallyDrop<T>>` plus explicit liveness and move/drop helpers, and permits
    evidence-based simplification. A safe-Rust `Option<T>`-shaped variant would model MIR
    liveness without any unsafe helper. Choosing is CE4-shaped and is not made here.

  - Validation: `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
    --all-features -- -D warnings` clean, **`cargo test --workspace` green: 917 passed / 0 failed
    / 2 ignored across 53 test binaries** (up from 894/52 — the new `native_c5_3a_aggregates`
    suite plus the new three-engine and unit tests). The full-workspace run is justified here
    rather than the scoped set, per CD-055's rule: `interp.rs` — the semantic oracle — changed for
    DEV-097, and that is a workspace-wide consumer (`mir_differential`, `exec_snapshots`,
    `conformance`, `gate3_execution` all read it).

- CD-057 [2026-07-21, C5.3b closed] **User enums, discriminants, and payload access compile and
  run natively. C5.3b CLOSED. The one structural problem — Rust cannot project into an enum
  variant outside a `match` — is solved by emitting a match EXPRESSION, with two consequences
  recorded rather than discovered later.**

  - **Delivered**: user enums → generated Rust enums with uniformly TUPLE variants (`V0()`,
    `V1(i32)`, `V2(i32, i32)`); `AggKind::EnumVariant` construction (type arguments from the
    destination, as with struct aggregates); `Projection::VariantField` reads;
    `Rvalue::Discriminant`. `EnumRef::CoreOption`/`CoreResult`/`CoreOrdering` are deliberately
    EXCLUDED — they belong with match/`?` lowering in C5.3c rather than being half-supported.

  - **Uniform tuple variants, including empty ones.** `V0()` is legal Rust, and the uniformity
    removes a special case from construction, from patterns (`V0(..)` matches it), and from the
    discriminant match. A unit variant would need different syntax in all three places.

  - **The structural problem.** Every other MIR projection appends to a place expression (`.f0`,
    `[2]`); a variant field has to WRAP what came before, because Rust exposes no way to project
    into a variant outside a `match`. Emitted as
    `(match &base { Ty::V1(__payload) => *__payload, _ => unreachable!("V-DISC-1: ...") })`.
    Two consequences, both deliberate: (a) the `_` arm is **provably dead** — V-DISC-1 makes a
    variant-field projection legal only after a discriminant test — so it gets the same
    `unreachable!()` the verifier-proved dead-block path has, naming the rule rather than
    fabricating a value that would paper over a lowering bug; (b) the result is an EXPRESSION,
    not a place, so it cannot be an assignment destination. `emit_dest_place` refuses that
    explicitly — a guard, not a limitation, since lowering emits `VariantField` only through
    `read_place` and pattern tests and STARK has no syntax for assigning into a payload.

  - **`Rvalue::Discriminant` takes the same shape** (an enum with payloads has no integer `as`
    conversion), listing **every variant with no catch-all**, so adding a variant cannot silently
    fall through to a wrong index. Its arms are typed by the DESTINATION local rather than a fixed
    width — a hardcoded `i128` failed to compile against MIR's `Int64` discriminant local, caught
    by the first native probe.

  - **Evidence**: four new three-engine cases (all three payload arities constructed and matched;
    payload field ORDER via a non-commutative operation, so a wrongly-bound two-field payload
    cannot pass; discriminant selection across four variants in a loop with distinct per-variant
    values, so any mis-selected arm changes the sum; a trap raised from a payload value) and three
    new native-only cases (one definition per instance with uniform tuple variants; a discriminant
    match naming every variant; the `unreachable!()` arm citing V-DISC-1). One test expectation of
    mine was wrong — a trap line off by one — and all three engines agreeing is what exposed it,
    which is exactly why `agree_trapping` takes the expected line independently.

  - **C5.3b makes CD-056 decision 3 (non-`Copy` storage) urgent rather than optional.** C5.3a's
    cross-block non-`Copy` move boundary bites far harder for enums: conditionally constructing a
    value and then matching it — the ordinary way enums are used — puts construction in one block
    and the match in another, which is exactly what the block-dispatch loop cannot express for a
    non-`Copy` value. The discriminant-selection test needs `impl Copy` to cross that boundary at
    all. **C5.3c is worse still**: `Option`/`Result` payloads are frequently non-`Copy` and `?` is
    inherently cross-block, so the storage decision is a prerequisite for C5.3c, not a nicety.

  - Validation, **scoped** per CD-055's rule (this change is backend-only — no `interp.rs`, no
    MIR contract, nothing with workspace-wide consumers): `cargo fmt --all -- --check` clean,
    `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean, `backend::` unit
    tests 40/40, `three_engine_differential` 31/31, `native_c5_3_aggregates_enums` 7/7, and the
    five earlier `native_c5_*` suites green. ~22 seconds.

- CD-058 [2026-07-21, owner review of 7829552] **C5.3b APPROVED as closed. The three CD-056
  decisions are RESOLVED. Work-package sequencing changed: a bounded prerequisite, C5.3d-0, is
  inserted BEFORE C5.3c.**

  - **C5.3b's limitation, stated precisely (owner wording).** C5.3b supports **Copy payload
    reads**. **Non-Copy payload movement remains blocked on the controlled-storage foundation and
    is not claimed complete merely by the current `VariantField` expression.** The scoped
    validation was confirmed correct for that commit: generated-Rust backend, its tests, and
    compiler records only — no workspace-wide semantic consumer.

  - **DECISION 1 — layout-query agreement. RESOLVED.** For C5 exit, layout-query agreement means
    **exact `size_of`/`align_of` agreement when all three engines execute under ONE recorded
    target-layout context**. `(8, 8)` is preserved as the default historical C4 reference layout.
    For C5 differential execution, an **injectable target-layout manifest** is generated or probed
    through the same canonical generated-Rust representations, target triple, rustc version,
    backend/runtime versions and profile as the native build; HIR and MIR consume that manifest
    during C5 layout cases, and the harness compares exact values. Relations-only layout tests may
    remain but **do not discharge** the C5.3 exit condition. (The current
    `layout_queries_run_in_all_three_engines` case is therefore a placeholder, not evidence.)

  - **DECISION 2 — Copy derivation. APPROVED as implemented, with the rule stated exactly.** A
    generated nominal instance may derive `Clone, Copy` **if and only if that exact concrete
    instance is present in MIR's `copy_types` classification**. MIR remains the authority: the
    backend must not infer Copy from Rust fields or trait resolution, and **`.clone()` must never
    implement a MIR move or copy**. `Eq`, `Ord`, `Hash`, `Drop` and other semantic traits are not
    derived as substitutes for STARK behaviour.

  - **DECISION 3 — non-Copy storage. RESOLVED: §7.2 controlled manual storage.**

    ```text
    ValueSlot<T> {
        storage: MaybeUninit<ManuallyDrop<T>>,
        whole-place live state,
        typed drop-unit live state where MIR distinguishes sub-places
    }
    ```

    **Ordinary `Option<T>` is REJECTED** — it introduces Rust-owned destruction.
    **`Option<ManuallyDrop<T>>` is REJECTED as the general representation**: it is adequate only
    for whole-value liveness, and once a field or constant-index element has been moved the
    remaining bytes no longer necessarily form a valid complete `T`. `MaybeUninit` is required to
    hold that partially moved state legally. An Option-shaped slot **may later be admitted as an
    optimisation** for locals MIR dataflow proves have no partial-move paths.

    Recording the reasoning because it is the part that would otherwise be re-litigated: the
    objection to `Option<ManuallyDrop<T>>` is not about destruction (`ManuallyDrop` already
    suppresses that) but about **representation validity under partial moves** — a distinction the
    C5.3a/C5.3b work had not yet had to confront, since neither admits partial moves.

  - **SEQUENCING CHANGE — C5.3c does NOT begin next.** A bounded prerequisite is inserted:
    **C5.3d-0 — non-Copy storage and movement foundation**, whose purpose is to unblock C5.3c and
    which **does not close C5.3d**. Its seven required deliverables (helper module; no ad hoc
    unsafe in emitted bodies; move semantics; Drop semantics; the five initial supported movement
    shapes; partial-move discipline; mutation-tested evidence) are recorded in
    `WP-C5.3.md`. After C5.3d-0 passes: C5.3c using the slot abstraction for non-Copy
    `Option`/`Result` values and `?` paths, then **C5.3d-1** with the dedicated observable
    destruction fixture and the final exactly-once/order/no-Drop-after-trap proof.

  - C5.3a and C5.3b remain closed.

- CD-059 [2026-07-21, C5.3d-0 CLOSED] **The non-Copy storage and movement foundation is complete.
  C5.3c is unblocked. One structural finding blocks part of C5.3d-1 and needs an owner decision.**

  - **Soundness correction first (owner review).** The initial `ValueSlot` was unsound for partial
    moves: `move_sub` took `&mut T`, moved a field out, and left the slot "live", after which
    `get`/`get_mut`/`take`/`drop_value` all remained callable over storage that no longer held a
    valid `T`. **The module's own test asserted `slot.get().1` after moving `.0`, so the bug was
    written into its evidence.** Corrected to a three-state machine — `Dead`/`Whole`/`Partial` —
    with whole-value operations requiring `Whole`, partial access restricted to raw-pointer
    projection, and an explicit `finish_partial` transition. Miri confirms zero UB across 18 slot
    tests; restoring the old permissive guard makes Miri report a real **use-after-free**.

  - **What this says about the validation strategy, not just the code.** The three-engine harness
    could not have caught it: it compares observable outcomes, and UB that does not change
    observable behaviour agrees across all three engines. **Differential testing is strong for
    semantics and blind to memory soundness.** Miri is now the compensating control — and even
    Miri did not flag `move_field` → `get` for a `(String, i32)`, because a moved-out `String`'s
    bytes stay bit-valid. For that case the state machine *is* the evidence. Layered: state
    machine primary, Miri for what it can see, neither complete alone.

  - **Generated projection helpers** (`emit_projections.rs`): one per (type, sub-place) pair the
    program actually uses, emitted into `mod stark_proj`. Raw `fn(*mut T) -> *mut F` via
    `addr_of_mut!` for struct/tuple/array (valid over partial storage); whole `fn(&mut T) -> &mut F`
    for enum payloads, which Rust cannot address without a `match`. Deliverable 2 verified on a
    partial-move program: every `unsafe` lies inside that module.

  - **`Copy` field reads had to become field-precise too**, and the state machine is what found
    it: moving `o.a` out then reading `o.b` aborted with "the slot is PARTIAL", because `get()`
    correctly refuses partial storage. Not an optimisation — a correctness consequence.

  - **All five deliverable-5 movement shapes work.** The C5.3a cross-block guard is deleted; what
    it refused now compiles and runs.

  - **STRUCTURAL FINDING — user `Drop` impls cannot compile natively yet (owner decision needed).**
    A destructor's receiver is `&mut Self`, so `impl Drop` requires `MirTy::Ref`, and references
    are outside the C5 subset entirely. This holds even when the body never touches `self` — the
    signature alone is enough. Therefore: `Terminator::Drop` works for structural glue only; a
    user destructor cannot be dispatched natively until `Ref` is admitted at least for destructor
    receivers; and **C5.3d-1's dedicated observable destruction fixture cannot be built as
    planned**. The §7.7 no-Drop-after-trap property is proven STRUCTURALLY instead (no `drop_with`
    precedes any abort site), and the difference is recorded rather than glossed.

  - Validation, scoped (backend + runtime; no workspace-wide semantic consumer): fmt clean,
    clippy clean, stark-runtime 23/23, `backend::` 40/40, `three_engine_differential` 35/35,
    `native_c5_3_aggregates_enums` 10/10, earlier native suites green, **Miri 18/18 with zero UB**.

- CD-060 [2026-07-21, C5.3d-0 REOPENED and re-closed; C5.3c in progress] **An owner review of
  `4a7e24c` found two contract violations the closure record had not covered. Both were real.
  Corrected; C5.3d-0 re-closed.**

  - **VIOLATION 1 — the partial-field primitives could not honestly be safe.** `move_field`,
    `copy_field`, `drop_field_with` and `move_field_whole` accepted an arbitrary projection
    function and then read the pointer it returned, checking only the SLOT's state. They could
    not validate that the pointer belonged to the slot, that the field was still live, or that
    the same field had not already been moved — so **safe Rust could reach UB** by calling
    `move_field(the_same_projection)` twice. The module's docs claimed preconditions were
    "checked rather than assumed"; for per-field liveness and projection validity that was false.

    Corrected as the owner directed: all four primitives are now `unsafe fn` with explicit
    `# Safety` contracts, and the backend emits **one safe wrapper per (type, sub-place,
    operation)** into `mod stark_proj`. Each wrapper pairs exactly one primitive with exactly one
    fixed projection over one slot type, so the obligation is discharged **by construction**
    rather than claimed. Emitted MIR bodies call only wrappers — asserted by a test that scans
    the bodies for `move_field`/`copy_field` and requires none.

  - **VIOLATION 2 — whole-enum structural Drop silently omitted its payload.**
    `emit_drop_glue` located a possible user destructor for an enum and then walked
    `struct_fields`, which an enum has no entry in. It never matched the active variant and never
    traversed `enum_variants`, so dropping a whole non-`Copy` enum marked the slot dead and leaked
    its payload. **Miri could not report it because the slot tests ignore leaks by design** — the
    fix's own evidence channel was blind to it.

    Corrected: enum glue now emits a match over EVERY variant (no catch-all, so a new variant
    cannot silently acquire a no-op drop) with payload fields dropped in reverse declaration
    order, mirroring `mir::interp::drop_in_place`. Two unit tests pin variant coverage, reverse
    order, and that `Copy` payload fields are ignored rather than dropped.

    **Currently unexercised by any compilable program**, and worth stating: no droppable type is
    expressible in the C5 subset, because a user `Drop` impl needs `&mut Self` and references are
    out of scope. The fix is correct and tested at the emitter level; it becomes reachable when
    the destructor-reference lane lands.

  - **C5.3c (Option/Result) is IN PROGRESS, not closed.** Core enums now share the user-enum
    representation through one `variant_payloads` table — the single source the definition, the
    discriminant match, and every projection all read — with `Option` as `None=0`/`Some=1`,
    `Result` as `Ok=0`/`Err=1`, `Ordering` as three fieldless variants, mirroring
    `mir::verify::variant_payload`. A probe compiles and runs `Option`/`Result` construction,
    matching and payload reads natively. **Deviation from §6.2 to flag:** §6.2 preferred ordinary
    Rust `Option<T>`/`Result<T, E>` "if all observable semantics match"; generated enums are used
    instead, so one mechanism covers every enum and no Rust drop glue exists for a type MIR is
    responsible for destroying. Owner may overrule; the change would be confined to
    `emit_types::nominal_type_name`.

  - Validation: fmt clean, clippy clean, stark-runtime 23/23, `backend::` 42/42,
    `three_engine_differential` 35/35, `native_c5_3_aggregates_enums` 10/10, Miri 18/18 zero UB.

- CD-061 [2026-07-21, C5.3c CLOSED] **`Option`, `Result`, matches and `?` compile and run
  natively. Two of the three remaining C5.3 gaps are now known to share ONE root cause.**

  - **Core enums share the user-enum representation** through one `variant_payloads` table — the
    single source the definition, the discriminant match and every projection read — mirroring
    `mir::verify::variant_payload`: `Option` `None=0`/`Some=1`, `Result` `Ok=0`/`Err=1`,
    `Ordering` three fieldless variants (A2).

  - **§6.2 deviation, flagged.** §6.2 preferred ordinary Rust `Option`/`Result` "if all observable
    semantics match"; generated enums are used instead so one mechanism covers every enum and no
    Rust drop glue exists for a type MIR is responsible for destroying — which matters more now
    that `ValueSlot` makes destruction explicitly MIR's. Reversible in
    `emit_types::nominal_type_name`.

  - **`?` needed no backend work**: MIR has already lowered it to branches and returns. A native
    test asserts no Rust `?` appears in the output, so the propagation is MIR's own control flow
    rather than a borrowed operator whose equivalence would have to be argued.

  - **Evidence**: four three-engine cases (both Option variants, including one flowing through a
    local into a later block; Result with DIFFERENT Ok/Err payload types, so confusing the two
    variants' payload tables would not compile; `?` on both propagating and falling-through
    paths; a trap from inside an Option payload, checking provenance on the core-enum path) and
    two native cases pinning generated variant order and the absence of Rust `?`. One expected
    trap line of mine was wrong again and all three engines agreeing exposed it — the third time
    that independent expectation has earned its place.

  - **`Ordering` is supported but UNREACHABLE, and it shares a root cause with the Drop gap.** It
    needs no special case in the emitter, but cannot be produced from compilable C5 source: the
    only way to obtain one is `a.cmp(&b)`, and `cmp` takes a reference. That is the same cause as
    user `Drop` impls being unrepresentable (`&mut Self` receiver). **The two remaining C5.3 gaps
    are one gap — the absence of references** — which means the narrow destructor-reference lane,
    slightly widened, would close both. Worth knowing before scoping it.

  - Validation, scoped: fmt clean, clippy clean, stark-runtime 23/23, `backend::` 42/42,
    `three_engine_differential` 39/39, `native_c5_3_aggregates_enums` 12/12, earlier native
    suites green.

- CD-062 [2026-07-21, owner decisions after C5.3c] **Five decisions. C5.3's remaining work is
  reduced from four unrelated gaps to TWO closure packages: references/Drop evidence, and exact
  target layout.**

  1. **C5.3c closure ACCEPTED** (`9aa94ac`) under the scoped-validation policy. The owner's note
     on why it matters architecturally: `?` required no backend reconstruction — MIR already
     contains the branches, payload moves and early return, and the backend merely emits them.
     The test prohibiting Rust's `?` is the correct guard against semantic reconstruction.

  2. **Generated core enums APPROVED; §6.2 AMENDED rather than the implementation reverted.** New
     normative wording: *"Core enums use compiler-generated concrete enum representations governed
     by MIR's canonical variant table. Rust `Option`, `Result` and `Ordering` are not used as
     STARK value representations in C5. A future representation optimisation requires evidence
     that discriminants, layout queries, movement, partial movement and explicit MIR-directed
     destruction remain equivalent."* The original "prefer Rust's types if observable semantics
     match" condition is **too weak after `ValueSlot`**: Rust-owned Drop can conceal a missed MIR
     Drop and make exactly-once evidence less falsifiable, and the dual path through definitions,
     discriminants, projections and Drop glue would be permanent. A semantic boundary, not an
     implementation convenience.

  3. **EPHEMERAL BORROWED-CALL REFERENCE LANE APPROVED** — renamed from "destructor-reference
     lane", because it covers both cases the missing-references finding identified: shared refs
     for `cmp(&other)` and exclusive refs for `Drop::drop(&mut self)`. Bounded to: `RefOf` borrows
     only a verified live, WHOLE place; never into a partially moved `ValueSlot`; the reference is
     consumed by a statically resolved direct call; creation and consumption in the SAME basic
     block; a generated reference temporary has exactly one use; reference-typed parameters
     allowed; callees may use `Deref` projections from them; shared reads, exclusive mutates and
     serves as destructor receiver. Forbidden: returning, storing in aggregates, writing into user
     locals, passing indirectly, carrying across blocks, nested references, slices, reference
     equality, general reborrowing, reference-valued results. Everything else rejected before
     rustc. A pre-emission validator enforces single-use/same-block; the emitter **inlines the
     borrow into the call** (`cmp_fn(&lhs, &rhs)`, `drop_fn(&mut value)`) rather than introducing
     general reference storage — considerably safer than making references ordinary
     `ValueSlot`-backed values.

  4. **`DropPlan` MANDATORY before C5.3d-1 closure**, and it precedes any general
     `NativeOperation` refactor (owner accepted that sequencing). A representation-neutral plan
     derived from `MirTy` + `TypeContext`, consumed by BOTH the MIR interpreter and the native
     emitter: `Noop` / `UserDestructor(instance)` / `Struct(reverse fields)` / `Enum(every variant
     → reverse payload)` / `Tuple(reverse)` / `Array(reverse indices)`. Preserves: user destructor
     first; structural fields or active payload after; reverse declaration order; complete variant
     coverage; no action for `Copy` units. **Does not change MIR v0.1** — it centralises an
     existing duplicated derivation. CD-060 fixed the enum-Drop *instance*; `DropPlan` removes the
     *class*.

  5. **Universal `NativeOperation` IR DEFERRED**, to evolve incrementally. **Layout manifest
     OPENED as an independent package (C5.3e)**, which may proceed in parallel since it depends on
     neither references nor `DropPlan`.

  - **Execution order set by the owner**: C5.3d-1a (ephemeral references) → C5.3d-1b (canonical
    `DropPlan`) → C5.3d-1c (observable closure evidence, then close C5.3d-1). C5.3e independent;
    if work must be sequential, C5.3d-1 first as the higher correctness risk.

  - **Trap-line expectations KEPT**, with an addition: each trapping fixture must carry an
    `expected_span_reason` note documenting WHY the expected location is correct, derived from the
    language rather than from any engine. The owner's rationale: having corrected the expected
    answer three times confirms these expectations are independent rather than self-fulfilling.

- CD-063 [2026-07-21, C5.3d-1a CLOSED] **The ephemeral borrowed-call reference lane is
  implemented. `Ordering` is reachable and user destructors compile — the two gaps CD-061
  identified as one root cause are closed.**

  - **Delivered**: `MirTy::Ref` in the type mapping; `Projection::Deref`; `Rvalue::RefOf` as a
    borrow expression; `LocalKind::DropFlag` admitted; and `validate_ephemeral_references`, a
    pre-emission validator refusing every out-of-lane shape.

  - **Three design points worth keeping**: (a) a reference local is **never** slot-backed, even a
    `&mut` one — a slot-backed `&mut Self` receiver would make the destructor's `Deref` project
    through the slot rather than the reference; (b) reference locals are declared
    **uninitialised**, so rustc becomes a *second* check on the lane — a reference escaping its
    block fails as "possibly uninitialized" rather than reading a fabricated value; (c) one
    slot-backing rule (`emit_types::is_slot_backed`) shared by the signature emitter, the local
    declarations and place emission. That third point is not theoretical: those sites disagreed
    during this work and produced a crate binding a parameter under one convention and reading it
    under the other.

  - **DEVIATION FROM CD-062, reported not absorbed.** The lane requires the reference to be
    "consumed by a statically resolved direct call". That is the destructor shape exactly, but
    **not** what `a.cmp(&b)` lowers to: for primitives lowering INLINES the comparison, giving
    `_5 = &_2; _6 = copy _5; _7 = Lt(copy _1, copy (*_6))` — consumed by a `Deref` READ inside a
    `BinOp`, via an intermediate copy. Ephemeral, same-block, unstored and unreturned all still
    hold, so the lane's purpose is intact; its stated consumption form is not. The validator
    accepts same-block consumption by read as well as by call. **The alternative is to reject
    `cmp` and leave `Ordering` unreachable, which would defeat the lane's own motivation** —
    owner may rule otherwise.

  - **Evidence**: two three-engine cases (all three `Ordering` variants with distinct results; a
    destructor reading through `&mut Self`) and two native cases — one asserting the destructor
    receiver is a bare Rust reference not a slot, one driving out-of-lane shapes (returned
    reference; reference carried across blocks) and requiring refusal **before rustc**, failing
    loudly if any reaches rustc and fails there instead.

  - Two matches became exhaustive as a result (`LocalKind`, `Rvalue`) and their catch-alls were
    deleted: a new variant now stops compilation instead of silently becoming an `Unsupported`
    diagnostic nobody reads.

  - Validation, scoped: fmt clean, clippy clean, stark-runtime 23/23, `backend::` 42/42,
    `three_engine_differential` 41/41, `native_c5_3_aggregates_enums` 14/14, earlier native suites
    green.

- CD-064 [2026-07-22, C5.3d-1b DONE] **`mir::drop_plan` is the canonical destruction plan, derived
  once and consumed by both the MIR interpreter and the native emitter** — CD-062 decision 4
  discharged.

  - **The defect class, not the instance.** CD-060 fixed the emitter's enum drop glue after it was
    found walking `struct_fields` and dropping no payload at all. The cause was structural: two
    independent reconstructions of one semantic rule, agreeing only because they were written to.
    `drop_plan::plan_for(ty, types)` is now the only derivation; `interp::run_drop_plan` and
    `emit_bodies::emit_drop_plan` each APPLY it and decide nothing about order, coverage or
    obligation.

  - **Four invariants moved from convention into the plan's SHAPE**: (a) `Destructor { symbol,
    then }` **nests** the components inside the destructor, so "fields before the user destructor"
    is *unrepresentable* rather than merely discouraged; (b) components are stored in destruction
    order and consumers iterate forward, with `array_order(len)` a named function so reversing it
    is a visible edit; (c) `Variants` is indexed by variant number, always complete, and carries
    each variant's **full arity** beside its droppable fields, so a generated `match` is exhaustive
    without a catch-all; (d) any component whose plan is `Noop` is absent, and an all-`Noop` parent
    with no destructor is itself `Noop` — which is where "never drop a `Copy` field" now lives,
    once, instead of as a filter each consumer must remember.

  - **`Vec`/`Box` name their element by TYPE, not by an inlined sub-plan.** They are Core v1's only
    indirection and therefore its only route to a recursive type (`enum List { Nil, Cons(Int32,
    Box<List>) }`); inlining would not terminate. Everything else is inline, finite, and planned
    eagerly.

  - **MIR v0.1 unchanged**, runtime surface untouched — this centralises an existing derivation.
    The variant-payload table (previously written out three times — `interp`, `verify`,
    `emit_types` — with the variant indices agreeing only by inspection) moved into the same
    module, and all three now read it. The interpreter memoises plans per type (`Rc<DropPlan>`),
    since the walk this replaced was lazy and a `Drop` inside a loop runs once per iteration.
    Tuples and arrays reach the native drop path for the first time as a consequence;
    `Vec`/`Box` steps are **refused** by the emitter rather than approximated, since glue that
    destroyed elements while leaking the buffer would be worse than a refusal.

  - **FLAGGED, carried forward unchanged, not silently corrected.** The remaining `Core` types —
    `String`, `HashMap`, `HashSet`, the iterators, `File` — plan to `Noop`, exactly reproducing
    what `interp::drop_in_place` already did. For a `HashMap<K, V>` whose `V` has a destructor that
    is arguably wrong, but it is the reference semantics as they stand, and changing it here would
    move the oracle without an owner decision. Recorded in the module so the question is
    answerable rather than lost.

  - **Evidence**: 14 derivation tests (order, coverage, index preservation, `Noop` collapse, core
    enums, deferred `Vec`/`Box`, a recursive type through `Box`, missing tables erroring rather
    than silently planning nothing) plus CD-062's mutation set. Each mutation corrupts the SHARED
    plan and shows the corruption reach the generated Rust — which is what establishes application
    rather than re-derivation, since a re-deriving emitter would ignore a corrupted plan and every
    one of these would fail. Five of the six are representable: omitted variant, omitted payload
    field, reversed order, re-added `Copy` field, and destructor ordering — that last one resolving
    to *unrepresentable*, with the nearest permitted rearrangement landing the destructor on a
    field and thus failing to compile. The sixth (`Drop` after a trap) was already covered by
    `mir_differential`, `gate3_execution::trap_aborts_without_running_pending_destructors` and
    `native_c5_3_aggregates_enums`, and carries no plan semantics.

  - Validated with the **full workspace suite**, not the scoped set: `interp.rs` is the semantic
    authority and every differential fixture consumes it.

- CD-065 [2026-07-22, owner assessment after `888d9c5`] **The process-driven re-engineering phase
  of C5 is CLOSED. Stop improving the process; finish the evidence, the manifest, linkage, build UX
  and exit qualification. Carry the broader process lessons into C6.**

  - **Owner's finding**: `DropPlan` genuinely replaces the duplicated derivations rather than
    merely documenting them; the emitter's remaining responsibility is only how to spell a planned
    step. Two sources of future drift are gone (destruction traversal; variant-payload definitions).
    No comparable structural refactor is judged outstanding. Another general abstraction now would
    be diminishing returns.

  - **DEFERRED explicitly**: `NativeOperation` IR, broad operation-planning abstractions,
    architecture dashboards, process metrics, retroactive conversion of old work packages, general
    references, runtime liveness bitmaps.

  - **Only two process items remain**: one adversarial review at C5.3 closure (Drop reachability,
    partial moves, layout evidence, rejected adjacent cases), and one gate-exit review at C5.6
    against the twelve C5 outcome conditions and the final supported-subset claim.

  - **Bounded caveat recorded for the future owning-core-representation package, not for C5.3.**
    `DropPlan` maps `String`/`HashMap`/`HashSet`/iterators/`File` to `Noop`, preserving interpreter
    semantics. Not a C5 blocker, because the generated backend still REJECTS those representations
    rather than silently compiling them. But before an owning core representation (e.g. a native
    Rust `String`) is admitted, STARK must distinguish **STARK semantic Drop glue** from **native
    representation reclamation**: a type may have no user-visible STARK destructor while still
    requiring its buffer or allocation to be reclaimed. To be solved by that package, not
    speculatively inside C5.3.

  - **Remaining C5 work, owner's ordering**: (1) C5.3d-1c observable Drop closure — now evidence
    work, not architecture: exactly-once, destructor-before-fields, reverse field/payload order,
    a moved value destroyed only by its new owner, no destructor after a trap, **plus one
    partial-move case with a genuinely droppable sibling** (the emitter still refuses projected
    `Drop` terminators, so this case settles whether the bounded C5 subset needs sub-place `Drop`
    emission or whether every approved fixture legally avoids it — the last ownership seam likely
    to expose implementation work); (2) C5.3e exact layout manifest; (3) C5.4 linkage and function
    values — function-instance constants, function-value storage/copying, indirect calls,
    cross-package references, the frozen three-package workspace; (4) C5.5 `stark build` as a
    user-facing route; (5) C5.6 qualification, including **hosted CI as a real exit item, not a
    formality** — `888d9c5` carries no GitHub status checks despite locally reported validation.

  - **Owner maturity estimate**: C5.3 approximately 90–93% complete; full Gate C5 approximately
    76–80%. Highest-risk architectural section (non-`Copy` ownership and destruction) judged under
    control.

  - **Copy consolidation FOLDED IN to C5.3d-1c by owner direction, and DONE.** The classification
    had been derived three times — `lower::is_copy`, `verify::mir_is_copy`,
    `emit_types::mir_ty_is_copy` — the same defect class CD-064 closed for destruction. The two
    CONSUMERS now share `TypeContext::is_copy`; `lower::is_copy` deliberately does not delegate,
    because it is the PRODUCER and answers the nominal case from the HIR precisely to fill the
    table the others read. Since no single function could cover both, the producer/consumer
    agreement is enforced empirically instead: `assert_copy_classification_agrees` runs over every
    differential program and the whole frozen corpus, checking that lowering never emits
    `Operand::Copy` for a place the type context calls non-`Copy`.

- DEV-098 [2026-07-22, found by the CD-065 fold-in, NOT a regression] **`Operand::Copy` on a `&mut`
  reference is a deliberate, verifier-accepted MIR shape that the `Copy` classification does not
  describe.**

  - The producer/consumer agreement check, run unrestricted, flagged **exactly 11 sites** across
    the corpus and the full differential suite — and **every one was `Ref { mutable: true, .. }`,
    no other type at all**. That uniformity is the result: the two classifiers agree everywhere the
    question is the same one.

  - It is not a defect in either. A `&mut` handed to a callee or a bounds check is **reborrowed**,
    not moved, or MIR would lose the reference; `is_copy` answers a different question about the
    same type ("does binding it elsewhere consume it?" — yes). Both answers are correct for their
    own question. `Operand::Copy` therefore means "read without consuming", which for `&mut` is a
    reborrow rather than a duplication.

  - **Why it matters to C5 and where it is contained.** The native backend does not slot-back
    references, so `Operand::Copy` on a `&mut` local emits a plain Rust read — which for `&mut`
    is a *move* in Rust, not a copy. A second read of the same reference local would therefore not
    compile. Contained today by the C5.3d-1a lane's single-use/same-block validator (refusal before
    rustc) and by rustc itself as the backstop. Flagged for the C5.3 adversarial review rather than
    changed: renaming or splitting `Operand::Copy` would be a MIR contract change.

  - The check is scoped to exclude `&mut` and is retained as a live guard for every other type.

- CD-066 [2026-07-22, C5.3d-1c DONE; C5.3d-1 CLOSED] **The observable destruction closure is
  evidence for seven properties across three engines — and it exposed a missing backend operation
  that was wider than the partial-move seam it was aimed at.**

  - **The observation channel is a real constraint, stated rather than worked around.** Native
    `println` does not exist (`Callee::Runtime` is wholly unsupported until WP-C5.4c) and
    `NATIVE_STDOUT_SUPPORTED` is still `false`; STARK has no globals and no reference fields, so a
    destructor cannot record its own firing for a later assertion either. The cases therefore use a
    **trapping destructor as a position probe**: traps abort, so the first destructor to run is the
    one that traps and the trap's exact line names it. Each case is built so one ordering question
    decides the reported line, and destructors that must not both fire get different types so they
    occupy different lines. This reads out one bit of order per run. Full native destruction
    *tracing* is blocked on `RuntimeFn` and belongs to WP-C5.4c.

  - **Seven properties, eight three-engine cases**: own destructor before fields; fields in reverse
    declaration order; active-variant payload only (a MIRRORED pair, because one case alone would
    be satisfied by an engine that always destroyed variant 0); a moved value destroyed by its new
    owner (the caller's assertion is deliberately false, so caller-scope destruction would report a
    different line — a probe, not a tautology); no destructor after a trap; exactly once; and the
    partial move with a droppable sibling. Every `expected_line` is derived from the language rule
    and carries an `expected_span_reason` note, per CD-062.

  - **Exactly-once is the one property a trap probe cannot show** (a trap aborts on the first
    destruction, so a second is never reached). Stated as a completing case instead, and what makes
    completion meaningful is engine-specific: the MIR interpreter poisons a local's slot on `Drop`
    and the native `ValueSlot` asserts `Whole` in `drop_with`, so a second destruction is a
    violation in both rather than a silent repeat.

  - **THE FINDING — per-unit (sub-place) destruction was missing, and not only for partial moves.**
    Two fixtures failed to build. MIR's drop elaboration decomposes an aggregate with several drop
    units into **one flag-guarded `Drop` per unit on a projected place** — `drop _1.1` then
    `drop _1.0`, each behind its own `Bool [dropflag]` — so a plain two-droppable-field struct with
    no destructor of its own arrives projected. The backend refused all projected `Drop`s, so that
    struct could not compile natively at all. **The refusal was correct, not merely conservative**:
    collapsing per-unit drops into a whole-local one would destroy a unit MIR's flags say is
    already gone (§7.6).

  - **Closed with a real operation, not a relaxation.** `HelperOp::Drop` generates one wrapper per
    (base type, projection) around `ValueSlot::drop_field_with` — the primitive already existed
    from C5.3d-0 — with the unit's `DropPlan` **baked into the wrapper**, since a wrapper is
    already per-(type, projection) and that fixes the field type and hence the plan. Call sites
    stay plain safe calls, so an emitted body still contains no `unsafe` and no destruction logic.
    A projected `Drop` of an **enum payload** is refused with a stated reason: an enum's payload is
    destroyed by the whole-enum plan's variant match, and the `&mut T` projection form needs a
    complete value the drop is in the middle of dismantling.

  - **What the emitter does NOT decide.** MIR sequences the units and MIR's flags skip the
    moved-out one; the emitter follows. Per-unit liveness stays MIR's, per §7.6.

  - **C5.3e is now the ONLY remaining C5.3 exit condition** — every other §14 item is discharged.

- DEV-099 [2026-07-23, found while scoping C5.3e, PRE-EXISTING] **A layout query on an ARRAY type
  fails to lower.** `size_of::<[Int32; 4]>()` reaches lowering and dies with "field type form
  (C4.5)" — `hir_field_ty` does not handle an array type in a turbofish position. Every other
  queryable shape works: primitives, tuples, structs, user enums, `String`, and a monomorphised
  generic parameter. Not introduced by C5.3e; recorded because arrays are inside the C5.3a subset,
  so the gap is visible from the layout-query exit condition. Bounded front-end work, not a
  semantic question.

- CD-067 [2026-07-23, owner decision, RECOMMENDATION OVERRULED] **The generated crate must NOT
  cross-check the STARK layout contract against Rust's physical layout, and generated internal
  nominals must NOT be `#[repr(C)]` for that purpose.**

  - **The authority analysis stands**: the named versioned `TargetLayout` contract is the
    observable result; physical representation stays unobservable and backend-private. Native
    lowering emits `4u64`, not `core::mem::size_of::<i32>() as u64`.

  - **Why the recommendation was wrong.** The proposed assertion enforces a stronger, different
    rule — *the target contract must equal the generated-Rust backend's physical representation* —
    which Core v1 does not require. It would (a) make the contract **backend-dependent**, so a
    later Cranelift backend using a different representation while implementing the same contract
    would be obstructed; (b) **conflate three separate contracts** — the observable language
    layout contract, the internal backend representation, and the separately versioned provider
    ABI — when LAYOUT-ABI-001 explicitly says equal `size_of`/`align_of` does not establish
    interoperation compatibility, so blanket `#[repr(C)]` could later be misread as an internal ABI
    commitment; (c) **sacrifice representation freedom for no Core-visible gain**, since field
    reordering and niche optimisation are unobservable and forcing a full `Option` discriminant
    pays a physical cost for no normative guarantee; and (d) **not actually validate the abstract
    contract** — it checks agreement with one Rust representation, not that the algorithm is
    internally coherent, that arrays follow the declared stride, that alignment combinators hold,
    that enum formulas cover every variant, that all three engines use the same named target, or
    that the manifest matches its recorded contract version.

  - **The concern about unfalsifiability was valid; the remedy was not.** Falsifiability comes
    from making the **declared algorithm and manifest independently testable**, not from
    redefining the contract as "whatever Rust physically chose".

  - **Required instead**: one versioned `TargetLayout`; one deterministic combinator
    implementation; an explicit target-contract identifier (`target_contract`,
    `layout_contract_version`, `compiler_layout_revision`); exact FROZEN values for the C5 layout
    matrix (primitives, tuples, arrays, structs, user enums, `Option`, `Result`, function values,
    and every other admitted C5 value); independent HIR-type and MIR-type walks; native constants
    from the same manifest; mutation tests that alter a primitive, an aggregate rule, or a manifest
    entry and break agreement; manifest identity in the build key and build report; and rejection
    when the requested target and manifest identity disagree.

  - **A host-layout comparison may later exist as a non-normative diagnostic** (`--audit-host-layout`)
    that REPORTS rather than rejects, unless a particular representation explicitly declares
    `physical_layout_matches_target_contract = true` — useful for provider-ABI types, serialization
    buffers, memory-mapped structures, or a backend optimisation deliberately relying on physical
    equivalence. Never for ordinary internal STARK values.

  - **DEV-099 is promoted to a MANDATORY C5.3e prerequisite**, not an adjacent limitation: arrays
    are in the approved C5 aggregate subset and the exit matrix explicitly requires fixed-array
    layout coverage, so a deterministic front-end failure on a required layout shape would leave
    C5.3e incomplete.

  - **Plan correction required** in `WP-C5-ENTRY.md`: replace the language saying generated Rust
    answers layout queries from its actual generated representation with — "`size_of<T>` and
    `align_of<T>` return values from the selected versioned STARK `TargetLayout` contract. HIR, MIR
    and native execution consume that same contract. A backend's internal physical representation
    is not observable and need not equal those values unless a separate representation contract
    explicitly requires equivalence."

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

## Conformance summary
- Lexical: WP-C1.1 requalification complete (2026-07-17). Strengthened: all 15 reserved words
  now tested by name (was 3), reserved-word rejection confirmed in non-expression positions,
  nested-comment depth tested to 4 levels (was 2) with a matching unterminated-at-depth negative
  case. Found and closed one real bug in the process (DEV-014). Found and recorded, but did not
  fix, a real gap outside this rule's own scope (DEV-015, literal overflow never checked).
- Syntax: WP-C1.1 requalification complete. Strengthened: `>>`/`>>=`/`>=` generic-closing-token
  splitting (added the previously-untested `GtEq`→`Eq` split arm and a bare-shift-expression
  contrast case), multi-file `mod` layout (added missing-file, duplicate-declaration, and
  circular-reference cases — the missing-file case is DEV-014's regression test), depth-limit
  boundary behavior (added exact-latch and false-positive-floor assertions, `starkc/tests/
  robustness.rs`), diagnostic determinism across repeated parses of identical input, and AST
  span-containment (new `starkc/tests/span_integrity.rs`, DEV-018 — first-ever programmatic
  span-invariant check in the codebase, covering `Expr`/`Block` nodes across the full parseable
  fixture corpus).
- Types: WP-C1.3 requalification complete (2026-07-17). The equality/trait-dispatch closure the
  roadmap flags is now **fully resolved** (DEV-008 closed — real `Eq::eq` dispatch implemented,
  plus a companion fix so `Ty::Core` container types satisfy Eq/Ord bounds at all). STD-004
  (standard traits) exhaustiveness audit closed (DEV-013) with 2 real bugs found and fixed:
  `.clone()` was entirely non-functional on every compiler-builtin type (String/Vec/Option/
  Result/HashMap/HashSet/Range/IOError), and trait default method bodies were never used as a
  fallback when unoverridden — both now fixed with regression tests. `Error`/`Hash`/`Display`/
  `Clone` as generic *bounds* were already correctly recognized throughout (the DEV-013 seed's
  worry about `Error` support was checking the wrong function). Two new deviations found and
  recorded but deliberately not fixed to keep scope bounded: DEV-023 (`Display`/`Hash` share
  Clone's old "missing as a callable method on builtins" bug, not yet fixed) and DEV-024 (`From`
  trait `Type::from(value)` associated-function calls fail to resolve, root cause not yet
  isolated). Local inference boundaries, generic substitution, associated types, orphan/overlap,
  and conflicting-impl diagnostics were spot-checked against existing tests
  (`gate5_semantic_gaps.rs`, `typecheck.rs`'s own test module) and found adequately covered —
  not subjected to the same exhaustive research-agent audit as WP-C1.1/C1.2 given the WP's time
  budget was consumed by the two substantial bug-fix cycles above; a future pass could still
  deepen this if warranted.
- Semantics: old Gate 2/3 coverage; pending WP-C1.3-C1.5.
- Memory: old Gate 2 M2.4 (ownership/borrows); pending WP-C1.4 full positive/negative corpus
  construction — not yet confirmed to exist at that depth.
- Modules/packages compiler surface: old Gate 2/Phase 1-3 (multi-file modules, `starkpkg.json`
  manifests, dependency resolution/locking per `git log` Phase 1-3 commits). `PKG-004`/`PKG-005`/
  `PKG-006` were incorrectly `missing` in the coverage database — corrected to `partial` under
  WP-C0.3 with real source/test citations; see DEV-002. WP-C1.2 requalification complete
  (2026-07-17): name resolution, module/visibility rules, imports, and re-exports strengthened
  across the full 10-item roadmap matrix; 3 real bugs found and fixed (DEV-004, DEV-006 resolve
  half, DEV-007); 1 new significant finding recorded but not fixed (DEV-019, E-code collisions);
  cross-package coherence checking (SEM-007) and cross-package diagnostic file attribution both
  went from "unverified" to "confirmed working" with real two-package-workspace tests (DEV-021).
  STARK's visibility model confirmed stricter than Rust's (private = exact defining module only,
  no descendant inheritance) — see the dedicated "Design fact pinned down by WP-C1.2" note below.
- Tensor extension: old Gate 4 (`gate4-exit.md`, closed 2026-07-15, "no known deviations")
  covers syntax/resolution/static checking + bounded ONNX metadata decode. Old Gate 7
  (`gate7-decision.md`) added symbolic/computed dimensions and value-range semantics with a
  13/13 defect-detection result. Both predate the new C-numbering; WP-C1.x does not re-audit
  extension code (Core-only scope), but WP-C9.1/C9.2 will need this as input later.

## Known deviations — open index
Canonical ledger (full structured entries, all 72 numbered deviations):
`starkc/docs/conformance/KNOWN-DEVIATIONS.md`. The per-deviation narrative that used to live in
this file (seed list + WP-C1.1/C1.2/C1.3 addition sections) is archived verbatim in
`STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md` (CD-020); the ledger remains the
single source of truth.

Open as of 2026-07-20 (post-WP-C4.7-8.1a). **Every entry below is long-standing and unscheduled;
no open deviation belongs to the C4 track.**
- DEV-005 — `starkc` vs `stark` check/run warning-gating drift. Open, unowned since Gate C1.
- DEV-010 — LSP hover/definition/references are protocol stubs. Owner: WP-C8.2/C8.3.
- DEV-011 — doc comments are lexer trivia, not AST/HIR metadata. Unscheduled; needs a scoped
  proposal.
- DEV-012 — VS Code extension UI never interactively verified. Owner: WP-C8.7.
- DEV-017 — 39 of 59 legacy coverage rules still lack function-level positive/negative evidence
  classification (tooling exists; classification unscheduled).
- Informational, not owed a fix: DEV-SEED-008 (two hand-rolled JSON parsers), DEV-SEED-014
  (no attribute syntax — deliberate scope fact).

Closed 2026-07-20: DEV-070 (WP-C4.6 A2, both engines); DEV-074 (numbered by WP-C4.7-1 and closed
at creation — the A4-2e oracle slice-message alignment, a governance gap, not a code defect);
**DEV-069** (WP-C4.7-4 — per-item file resolution in typecheck/borrowck/oracle; this also
DISCHARGES CD-033's C5 multi-file prerequisite); **DEV-072** and **DEV-073** (WP-C4.7-5 —
move-out-of-borrow via match bindings, now rejected E0101; generic impls matched through
`match_impl_type` for operator and iterable bounds); **DEV-067** and **DEV-071** (WP-C4.7-7 —
bounded-parameter bounds behind references and at intra-generic call sites; `Ordering`
exhaustiveness); **DEV-077** (WP-C4.7-6.1 — oracle `Box::into_inner` double-drop); **DEV-078**
(WP-C4.7-6.3 — integer literals adopt their expected type); **DEV-075** (the DEV-075 increment —
`Char` ordered by Unicode scalar value, `Bool` not `Ord`, plus normative `PRIM-TRAIT-001`);
**DEV-076** (WP-C4.7-8.1a — the oracle's `unwrap_or` double-drop).
Closed 2026-07-19: DEV-060 (CD-024); DEV-061/062/063 — the function-value cluster — in the
CD-027 pre-C4.1 correction pass; DEV-064 (undetermined-generic rejection, WP-C4.5c, E0004);
DEV-065/066 (C4.5b oracle fixes). See `KNOWN-DEVIATIONS.md`.

## Design fact pinned down by WP-C1.2 (not a deviation, recorded so it isn't re-discovered)
STARK's visibility model is **stricter than Rust's**: per `07-Modules-and-Packages.md` §Visibility
("items are private to their defining module by default"), a private item is visible **only**
within its exact defining module — there is no Rust-style "visible to the defining module and
all its descendants." Confirmed by the pre-existing `module_paths_imports_and_visibility_are_
enforced` test (root cannot access a private item of its own direct child module) and by three
new WP-C1.2 tests (`super_and_crate_navigate_correctly_from_a_nested_module`,
`private_item_is_not_visible_from_a_descendant_module`,
`pub_use_single_level_reexport_is_visible_from_outside`) — the first drafts of the latter two
tests were written assuming Rust-style descendant-inherits-privacy semantics and failed against
the real implementation, which is what surfaced this. Any future WP writing STARK test fixtures
involving nested modules and private items should assume this stricter model.


## Architecture decisions
- AD-001 [pre-existing, old Gate 5] Native artifact-deployment backend is **ONNX Runtime via the
  `ort` crate**, pinned `=2.0.0-rc.12`, statically linked, CPU execution provider only
  (`starkc/docs/gate5-backend-decision.md:11`). IREE/Cranelift/TVM explicitly considered and
  deferred at that time. This is a decision about the *tensor artifact deployment* backend, not
  a decision about general Core native compilation — the two must not be conflated (see CD-002).
- AD-002 [pre-existing] ONNX decoding uses a hand-written protobuf reader with zero new runtime
  dependencies beyond `sha2` (for checksum verification); `ort`, `tract-onnx`, and `onnx-pb`
  crates were evaluated and rejected (`starkc/docs/gate4-design.md:158-169`). `starkc`'s own
  `Cargo.toml` has exactly one dependency, `sha2`, and forbids `unsafe_code` at the lint level.
- AD-003 [pre-existing] Both CLI binaries (`starkc`, `stark`) hand-roll argument parsing against
  a `USAGE` const rather than using `clap` or another CLI-parsing crate (confirmed: no `clap`
  entry anywhere in `Cargo.toml`/`Cargo.lock`).

## Native backend selection
- Status: **SELECTED** (WP-C3.4, owner CE5 decision, 2026-07-19).
- Selected strategy: **generated Rust/C** — generated Rust as the initial production backend
  behind verified MIR, with a **backend-neutral MIR contract that keeps `SELECT-DIRECT`
  (Cranelift) open as a C7-gated migration** (charter §1.6 rule 9, no lock-in). Decision +
  full three-way analysis: `starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md`;
  recorded as CD-026.
- Architecture commitments (roadmap WP-C3.4): emitter consumes **verified MIR** (not typed HIR);
  small STARK runtime library (print/panic/trap glue); Rust owns MVP value layout + calling
  convention; Native Provider ABI (C5.1) as `extern "C"` provider calls from generated Rust;
  Tier-1 targets first (linux-x64, macos-arm64) via rustc; debug/trap file:line via a STARK-span
  → generated-Rust-line → rustc-debug-info table; unsupported-MVP closure (floats/`?`/tuple
  patterns/traits/Drop/refs/Vec/HashMap/fn-values) tracked into C4.5/C5/C6.
- **Accepted trade (recorded):** `stark build` requires a full `rustc` toolchain as a permanent
  build dependency, and builds are slower than the direct backend. Acceptable for STARK-as-
  research-language; **re-evaluate the backend choice at C7** if the self-contained-compiler /
  systems-platform goal becomes primary (same evidence-gated pattern as the LLVM decision).
- Workload: 23-item frozen set (`NATIVE-CORE-ARCHITECTURE.md` §5), items 1-10 mapped to the
  frozen `exec_snapshots` corpus v1.0.0 (semantic oracle), items 11-23 specified reference
  programs. Two properties (fn-value Eq/Ord/Hash participation, monomorphised-generic fn-value
  identity) must be settled from the frozen spec or by CE1/CE2 before selection (CD-022).
- Spike evidence so far:
  - **WP-C3.2 generated-Rust (done):** 4/17 frozen corpus cases lower and match the interpreter
    exactly (arithmetic/precedence, loops/for/break/continue, multi-width ints, Int8-overflow
    trap→abort parity); 0 semantic mismatches on supported cases; 13/17 cleanly reported
    unsupported; mean rustc 87 ms/case. Liabilities unresolved (not falsified): rustc
    build-dependency weight, compile-time scaling, exe size, debug-info trap mapping, and the
    unsupported breadth (aggregates/generics/traits/refs/Drop/fn-values). Report:
    `starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md`; artifact `tests/spike_genrust.rs`
    (isolated, disposable).
  - **WP-C3.3 direct Cranelift (done):** 3/17 frozen corpus cases lower and match the interpreter
    exactly (arithmetic, loops/for/break/continue, Int8-overflow trap→abort parity); 0 semantic
    mismatches; 14/17 unsupported (same families as C3.2 plus unsigned ints — spike is
    signed-only, hence 3 vs C3.2's 4). Produces a real standalone native executable (Cranelift
    object + `cc` link). Codegen ~2 ms/case (phase-only), link ~47 ms/case; **defensible
    end-to-end ~49 ms vs rustc ~87 ms ≈ 1.8× on this tiny workload — explicitly NOT a general
    performance multiple** (charter caution; see the report's timing caveat — the raw 2-vs-87
    codegen ratio is not like-for-like). No rustc build dependency. Finding: Cranelift 0.133 needs
    rustc ≥1.94 (>1.93 here) → pinned 0.110, an MSRV-churn maintenance cost. Higher glue than
    generated-Rust (we own CFG/SSA/overflow/Drop/layout); weaker out-of-box debug-info; but the
    bigger beneficiary of the mandatory MIR (MIR ≈ Cranelift's own block/terminator model).
    Report: `starkc/docs/compiler/spikes/WP-C3.3-direct-cranelift.md`; artifact
    `tests/spike_cranelift.rs` + dev-only Cranelift deps (isolated, disposable).
- **Breadth run (2026-07-19):** generated-Rust extended to structs/impl-methods/generics/
  Option/Result/match/String → **8/17** frozen corpus cases (all matching), via ~250 lines of
  mechanical text emission (rustc absorbs monomorphization/layout/ABI/Drop). Cranelift breadth
  **measured at the struct boundary, not fully implemented** — struct-by-value needs stack-slot
  layout + field offsets + sret ABI; enums need tagged-union layout; generics need a
  monomorphization engine; String/Vec need a runtime — each a subsystem the direct backend owns.
  Cranelift stays 3/17. **Key WP-C3.4 caveat: most of that direct-backend breadth cost is
  mandatory MIR work anyway (Gate C4), so the HIR-level comparison overstates the direct
  backend's long-run cost.** Full head-to-head:
  `starkc/docs/compiler/spikes/WP-C3-breadth-comparison.md`. (Implementing Cranelift
  struct-by-value is a bounded ~150-200-line follow-up if an exact struct head-to-head number is
  wanted.)
- Both spikes done; the tradeoff is symmetric and matches the §4 hypothesis: generated-Rust =
  low glue + free cross-platform/debug-info + broad correctness cheaply + heavy rustc dep; direct
  = fast builds + no rustc + ABI control + biggest MIR beneficiary, but owns monomorphization/
  layout/drop/runtime. Neither falsified nor cleared; WP-C3.4 selects (CE5, owner).
- Evidence: see CD-002 for the closest existing evidence (old Gate 6/7 tensor/ONNX-deployment
  track) — informative precedent for methodology, not a substitute (CD-004).

## Diagnostic codes allocated or changed
- **MIR-0001..MIR-0013** [WP-C4.3, 2026-07-19] First allocation of the `MIR-xxxx`
  compiler-internal namespace (charter §5.1): 0001 target OOB, 0002 local OOB, 0003 projection
  type, 0004 assignment/operand type, 0005 call/checked signature, 0006 bare unsized, 0007
  possibly-moved use, 0008 discriminant/variant misuse, 0009 drop/drop-flag, 0010 index-proof
  discipline, 0011 FnPtr arithmetic/comparison, 0012 reserved (runtime-set violation —
  structurally impossible while RuntimeFn is a closed enum; reserved for serialized MIR), 0013
  invalid FileId in SourceInfo. These are internal invariant failures (lowering bugs), never
  user-source diagnostics. Full map: `src/mir/verify.rs` header + WP-C4.3.md.
- **E0008** [WP-C1.5] Integer literal out of range for its type (suffixed literal exceeds its
  suffix's representable range, or an unsuffixed literal exceeds `Int64`). See DEV-015.
- **E0009** [WP-C1.5] Array repeat count (`[value; count]`) is not a compile-time constant
  expression.
  Both registered in `04-Semantic-Analysis.md`'s normative Error Categories table
  (`STARK-Core-v1.md` regenerated in the same change). No codes allocated or changed by any other
  WP under this governance framework yet. Existing (pre-governance-framework) normative
  `E####`/`W####` codes are inventoried as part of WP-C0.1 (`starkc/src/diag.rs`), not duplicated
  here.

## Evidence inventory
- `starkc/docs/gate1-exit.md` through `gate7-decision.md` — old-numbering gate evidence, see CD-001/CD-002.
- `STARKLANG/tests/spec-fixtures/manifest.toml` — 113-entry spec-fixture corpus (directly
  re-counted 2026-07-19; the "121-fixture" figure this line carried from the C0 audit had
  drifted), verdict census in
  Repository baseline above.
- `cargo test --workspace --all-targets --all-features` output (2026-07-17 audit run) — 383
  passed / 0 failed / 2 ignored, full per-suite breakdown to be carried into
  `starkc/docs/dev/compiler-map.md` (WP-C0.1).
- `STARKLANG/conformance/core-v1-coverage.toml` — 59 rules, 53 implemented / 6 partial / 0
  missing, **integrity-audited under WP-C0.3** (duplicate-ID check, spec-chapter-validity check,
  4 stale `missing` entries corrected with cited evidence). `python3 starkc/scripts/
  check-conformance.py` output (2026-07-17, post-correction): 0 errors, 0 warnings.


## File inventory for current gate
C3-ENTRY (active transition): `STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md` (transition
work package, created 2026-07-19 under CD-020), `.github/workflows/ci.yml` (baseline widened
under CD-020), `STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md` (new archive).
Closed-gate file inventories (C0/C1): archived verbatim in the state-archive file; per-gate
evidence in the C0/C1/C2 exit reports.

## Follow-ups
- [ ] WP-C0.2 carry-forward (governance-process question, unresolved): gate7-decision.md's "No
      LSP work or language expansion is authorized" text was apparently overridden for WP8.1-8.5,
      but no explicit owner override record exists. Owner should either backfill a decision
      record or confirm WP8.x was tooling, not "language expansion" in Gate 7's sense.
- [ ] DEV-005: pick one warning-gating policy for `starkc check`/`run` vs `stark` — still
      unowned; candidate for C3-ENTRY or a small pre-C3 correction.
- [ ] WP-C8.2/C8.3: implement real LSP hover/definition/references (DEV-010).
- [ ] WP-C8.7: interactive VS Code Extension Development Host validation (DEV-012).
- [ ] WP-C1.1 follow-up (not blocking): underscore-placement rules for binary/octal literals
      untested; no max-value-per-suffix positive test for the 8 int / 2 float suffixes.
- [ ] DEV-017 remainder: classify the 39 unclassified legacy coverage rules (unscheduled).
- [x] **DEV-095 — WP-C5.3 opening condition. DISCHARGED 2026-07-21, CD-055.** The build key was
      hashing `program.dump()`, which omits the nominal type context and the Drop map, so a
      changed struct field or `Drop` impl could leave the key unchanged and silently reuse a stale
      generated crate. The key now covers all eight version axes, the entry symbol, the source
      table (names + content hashes), all four `TypeContext` fields, and the bodies — with seven
      cache-invalidation tests, mutation-verified against the old behaviour. **WP-C5.3's blocking
      entry condition is satisfied; aggregate and Drop-bearing native generation may begin.**
- [x] **Native Provider ABI v0.1 — CE4 Amendment 1. CLOSED 2026-07-21, CD-054**: approved at
      revision 3 and applied in full (ABI document, both `provider_abi.rs` files, fixtures,
      violation tests). Revision 1 was not approved; revision 2's design was approved with five
      required changes; revision 3 incorporates them. The close-function question was ruled —
      exactly one parameter, the consumed handle, nothing else, because MIR's `Drop(place)`
      supplies no argument list. ABI version stays `0.1`. Record:
      `STARKLANG/docs/compiler/native-provider-abi-v0.1-CE4-amendment-1.md`.
- [x] DEV-060: dispose before C3 workload freeze (C3-ENTRY blocker). **Closed 2026-07-19,
      CD-024 — fixed in `borrowck.rs::method_receiver`.**
Completed follow-ups through Gate C2 are archived verbatim in the state-archive file.

## Gate exit summaries
- C0: **PASS** (2026-07-17). Bootstrap, current-state audit, and authority repair complete. Full
  report: `starkc/docs/compiler/C0-exit-report.md`. Four stale documents corrected (`CLAUDE.md`,
  root `README.md`, `starkc/README.md`, `STARKLANG/docs/PLAN.md`); conformance database
  integrity-audited with 4 staleness errors fixed (DEV-002, closed); 10 confirmed deviations
  recorded with full structured detail in `starkc/docs/conformance/KNOWN-DEVIATIONS.md`; module-
  by-module compiler map produced (`starkc/docs/dev/compiler-map.md`). Explicit non-claim: no
  conformance percentage from this gate is trusted for Core v1/tensor v0.1 conformance purposes
  — see exit report's "No conformance percentage is trusted" section. Next: Gate C1.
- C1: **CORE-FRONTEND-CONFORMING-WITH-LISTED-DEVIATIONS** (2026-07-17/18). Full report:
  `starkc/docs/compiler/C1-exit-report.md`. Six requalification WPs closed (lexical/syntax, name
  resolution/modules/visibility, types/generics/traits, ownership/borrowing/drop checking,
  control flow/patterns/constants/numerics, conformance evidence generator); 12 of 23 deviations
  closed, 2 partially closed, 9 open and non-soundness-relevant. This entry backfilled during
  WP-C2.13's consistency sweep — not recorded here at the time of C1's own close. Next: Gate C2.
- C2: **CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS** (2026-07-18). Full report:
  `starkc/docs/compiler/C2-exit-report.md`. Reference-execution contract, abstract machine, and
  future-boundaries specifications written from scratch; all 24 high-cost open questions
  approved; 166-row completeness inventory has zero absent/contradictory/unclassified rows (6
  pending-owner-approval governance-only); 33 deviations closed this gate (the largest body of
  runtime-semantics fixes in the compiler track's history, including DEV-053/054 — a bare `None`
  pattern silently matching any value with wrong runtime output, the most severe finding to
  date), 8 remained open and non-soundness-relevant at gate close (see the open index above
  for the current set). WP-C2.12's differential corpus is
  representative, not exhaustive — explicitly disclosed, not disqualifying (cross-backend replay
  is blocked behind Gate C3 by the roadmap's own dependency order). Next: Gate C3, WP-C3.1.

---

## Session records
Records for WP-C0.0 through the Post-Gate-C2 correction brief Issues 1-5 (2026-07-17 through
2026-07-18) are archived verbatim in
`STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md` (CD-020). Gate-level evidence
remains in the C0/C1/C2 exit reports. Records below start at the most recent still-live
transition context.

### Post-Gate-C2 correction brief — Issues 6-8 (DEV-051, DEV-052, DEV-055) — 2026-07-19
DONE: user said "fix them" (Issues 6-8, previously deferred). Reproduced and closed all three
with real fixes, one working method throughout: reproduce on current head, isolate root cause by
reading the relevant resolver/checker code rather than guessing, fix, add regression tests, run
full verification, update docs.
- **DEV-055** (fixed first, most precisely diagnosed already): `use Color::*;`/`use
  Color::{Red, Blue};` silently expanded to nothing when the prefix names an enum rather than a
  real module. Root cause: `resolve_use_tree`'s `Glob`/`Group` arms (and their `_relative`
  counterparts) only ever consulted `submodule_map` (real modules); an enum's variants are
  resolved dynamically through `item_details`, never pre-populated into a module's `items` map.
  Fixed by adding `enum_variant_items`/`resolve_enum_variant_group_item`, wired into both arms in
  both functions. 5 new regression tests (`resolve.rs` x2, `interp.rs` x3), including one
  confirming a variant deliberately left out of a group import correctly stays undefined (rules
  out an overly-broad "import everything" fix).
- **DEV-051**: a trait default method body calling a sibling trait method through `self`
  (`self.name()` inside `fn greeting(&self) { self.name() }`) failed to type-check with `E0302`.
  Root cause: `resolve_method` already had a mechanism for an abstract `Ty::Param` receiver with
  no concrete `impl` to match (a bounded *generic function* type parameter), but it was scoped
  only to that case, never to `self` inside a trait's own default-method body (`current_self_ty
  == Ty::Param("Self")`, checked once, generically, at the trait declaration site). A first
  attempt placed the new check in the same spot as the existing one and it still failed, since
  `self`'s type at that point is `&Self` (a `Ty::Ref`), not bare `Ty::Param("Self")` — moved it
  to after the reference-deref loop, unlike the by-value generic-parameter case. Added
  `current_trait_id` (set alongside `current_self_ty` for trait default bodies) plus two shared
  helpers (`find_trait_method_sig`/`check_trait_member_call`) refactored out of the
  previously-inlined generic-parameter logic. 4 new regression tests, including a
  default-calling-another-default case and a wrong-arg-count case (confirms the fix doesn't
  silently swallow a genuine arity mismatch). **Side finding, NOT fixed** (confirmed pre-existing
  via `git stash`, not introduced by this fix): DEV-060 — calling the same un-overridden default
  method twice on one receiver wrongly raises `E0100 use of moved value` on the second call; two
  calls to an *overridden* trait method or an ordinary inherent method are both unaffected.
  Recorded as a new open deviation with its own regression tests documenting the current
  (defective) behavior and its exact scope, rather than silently worked around.
- **DEV-052**: `Eq::eq(&a, &b)` (fully-qualified call syntax) failed to resolve
  (`E0200 undefined variable 'Eq::eq'`) while the same syntax worked for a user-declared trait.
  Root cause: `resolve_path_relative`'s multi-segment loop only continued past a first segment
  resolving to `Res::Item` (a real trait declaration item, member indexed against
  `ItemDefDetail::Trait`); a `CoreTrait` (`Eq`, `Ord`, ...) has no such declaration item at all.
  Fixed by adding `Res::CoreTraitMember(CoreTrait, Span)`, resolved via a new
  `core_trait_method_name` table (one fixed callable method name per `CoreTrait`: `Eq`→"eq",
  `Ord`→"cmp", `Hash`→"hash", `Clone`→"clone", `Display`→"fmt", `Default`→"default"). Typecheck
  (`check_qualified_core_trait_call`) finds the matching impl's own method signature directly
  (no shared trait declaration to instantiate from, unlike the user-trait case), matching impls
  by trait-ref source text against a new `core_trait_source_name` table (mirroring
  `ty_satisfies_operator_bound`'s existing approach). The interpreter side needed no new
  impl-scanning logic at all: `call_qualified_core_trait` reuses the *exact* `find_method(...,
  Some(Res::CoreTrait(_)))` lookup the `==`/`<` operator sugar already calls for these traits — a
  qualified call is just an explicit spelling of the same dispatch. 4 new regression tests
  (`Eq` and `Ord`, an unimplemented-trait rejection, and a guard confirming the pre-existing
  user-trait qualified-call path is unaffected).
FILES: `starkc/src/resolve.rs` (DEV-055's `enum_variant_items`/`resolve_enum_variant_group_item`;
DEV-052's `core_trait_method_name` table and path-resolution wiring; both regression tests),
`starkc/src/typecheck.rs` (DEV-051's `current_trait_id` field and `find_trait_method_sig`/
`check_trait_member_call` helpers; DEV-052's `check_qualified_core_trait_call`/
`core_trait_source_name`; all three fixes' regression tests plus DEV-060's documentation test),
`starkc/src/interp.rs` (DEV-055/DEV-051 end-to-end regression tests; DEV-052's
`call_qualified_core_trait`; DEV-060's two scope-confirming companion tests),
`starkc/src/hir.rs` (new `Res::CoreTraitMember` variant), `starkc/src/analysis/query.rs`
(exhaustiveness update for the new `Res` variant), `starkc/docs/conformance/
KNOWN-DEVIATIONS.md` (DEV-051/052/055 marked resolved with full root-cause writeups; new
DEV-060 opened; count line updated to 58), this file.
RULES: none — three runtime/type-check-semantics corrections against already-normative rules
(trait default-method dispatch and fully-qualified trait-call syntax per `03-Type-System.md`;
glob-import name resolution per `07-Modules-and-Packages.md`); no conformance-database rule
citation or normative specification text changed.
DECISIONS: none new as CD/AD records. All three are spec-consistent corrections under Charter
§2.2 Sonnet-level autonomy — each makes a previously-rejected legal program accepted and correct,
none weakens an existing check or changes accepted behavior in a way that admits an unsound
program.
EVIDENCE: MANUAL + REG — every fix's original bug and every new regression scenario was run
against the actual compiler (not inferred from code reading alone); DEV-060's pre-existing,
unrelated-to-DEV-051 status was independently confirmed via `git stash` against the pre-fix head
before being recorded, not assumed. `cargo test --workspace --all-targets --all-features`:
**594 passed / 0 failed / 2 ignored** (up from 578/0/2 pre-this-pass, exactly the 16 new tests
across the three fixes — see each fix's own count above — zero regressions elsewhere). `cargo fmt --all -- --check` clean. `cargo clippy --workspace --all-targets
--all-features -- -D warnings` clean. `python3 scripts/check-conformance.py` re-run clean
(89.8%/53-of-59, unchanged -- none of these three fixes touch the conformance evidence database).
NEXT: no further work authorized this pass. DEV-060 (new, open) and DEV-009/DEV-022/DEV-023/
DEV-024 (long-open, C2.8/C2.9-owned) are the remaining known deviations without a fix.

### C3-entry governance-repair pass (CD-020) — 2026-07-19
DONE: full scope of CD-020 (see decision log): WP-C3-ENTRY.md created and wired into the
roadmap's C3-ENTRY section; WP-C4.4/C5.6/C6.5 amended to carry transferred WP-C2.12
obligations; CI widened to the C3-ENTRY baseline command forms plus new spec-regeneration
(`build-core-spec.py --check`) and named execution-snapshot steps; KNOWN-DEVIATIONS.md tail
summary corrected (DEV-009/022/023/024 were resolved by WP-C2.11, not open — the preceding
Issues 6-8 session record's own NEXT line repeats that stale claim and is corrected by this
note, left in place per append-only convention); state header head/fixture-census corrected
(`9e85396`, 113 entries/parse-pass 65); charter §5.3 dangling refs, commit-policy step, and
WP-C6.4 tier label fixed; SYSTEMS-ROADMAP.md gained the P1-relationship section; this file
compressed 3,145 → ~700 lines with all removed material verbatim in
`STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md`; C2-exit-report open-deviation
table given a dated post-gate update note.
FILES: COMPILER-STATE.md, STARKLANG/docs/compiler/COMPILER-CHARTER.md,
STARKLANG/docs/compiler/COMPILER-ROADMAP.md,
STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md (new),
STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md (new),
STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md, starkc/docs/conformance/KNOWN-DEVIATIONS.md,
starkc/docs/compiler/C2-exit-report.md, STARKLANG/tools/build-core-spec.py,
.github/workflows/ci.yml.
RULES: none — no normative rule, compiler, or interpreter change; governance surface only.
DECISIONS: CD-020.
EVIDENCE: `python3 STARKLANG/tools/build-core-spec.py --check` clean twice (deterministic);
`cargo fmt --all -- --check` clean; `cargo test --test exec_snapshots` 3 passed / 0 failed;
line-count arithmetic for the compression verified (588 kept + 2,557 archived = 3,145
original). Full `cargo test --workspace` not re-run this pass (no code changed); full CI run
of the updated workflow pending — tracked as the remaining CI blocker item in WP-C3-ENTRY.md.
FOLLOW-UP: owner decisions per WP-C3-ENTRY.md blockers 1-2 (six completeness rows, DEV-060);
corpus freeze after DEV-060 disposition; one demonstrated green CI run.
NEXT: WP-C3-ENTRY blocker closure; then C3-entry exit artifact; then WP-C3.1.

### CD-021 roadmap amendment — 2026-07-19
DONE: applied the owner-approved CD-021 amendment (see decision log): WP-C3.1 workload items
16-21 (existing function-value capability), C4.1/C4.3/C4.5 indirect-call ownership, C5.1
function-value ABI items, P1/S5 trap-abort operational report, WP-C10.7 release-blocking
deviation sweep.
FILES: STARKLANG/docs/compiler/COMPILER-ROADMAP.md,
STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md, COMPILER-STATE.md.
RULES: none — no normative Core rule, compiler, or interpreter change; the workload items
reference already-frozen `fn(...)` semantics.
DECISIONS: CD-021 (owner-approved this session).
EVIDENCE: spec/implementation citations verified by direct grep before recording
(03-Type-System.md:198-200,999; 06-Standard-Library.md:243-244,260-262,663-666;
interp.rs:260). No count/enumeration references to the C3.1 workload existed to go stale
("at least:" phrasing confirmed).
FOLLOW-UP: draft the "Callable ABI and Future Closure Compatibility Spike" proposal before
WP-C5.1 (recommended during C3 spike work); WP-C3-ENTRY blockers unchanged and still open.
NEXT: WP-C3-ENTRY blocker closure (six completeness rows, DEV-060, corpus freeze, green CI
run); then C3-entry exit artifact; then WP-C3.1 with the 21-item workload [23 after CD-022].

### CD-022 follow-up amendment — 2026-07-19
DONE: applied the owner-approved CD-022 (see decision log): release-class claim-scope repair
(Compiler Stable vs General-Purpose Stable, CD-019 preserved), WP-C3.1 workload items 22-23
plus the pre-backend-selection Eq/Hash/monomorphised-identity resolution requirement,
state-header field renamed to "Amendment base commit".
FILES: STARKLANG/docs/compiler/COMPILER-ROADMAP.md, COMPILER-STATE.md.
RULES: none — no normative Core rule, compiler, or interpreter change. The two open
function-value properties are flagged for settlement, not settled here.
DECISIONS: CD-022 (owner-approved this session).
EVIDENCE: spec citation verified by direct read before recording (03-Type-System.md:748-749 —
function values are Copy); release-class contradiction verified against the roadmap text
(C7.7 P1 gate vs the vacuous conditional). Workload numbering re-verified contiguous 1-23.
FOLLOW-UP: push to origin and record one green run of the updated CI workflow (last
C3-entry CI blocker item); callable-ABI/closure-compatibility spike proposal still pending,
pre-C5.1.
NEXT: WP-C3-ENTRY blocker closure (six completeness rows, DEV-060, corpus freeze, green CI);
then C3-entry exit artifact; then WP-C3.1 with the 23-item workload.

### C3-ENTRY blockers 1-2 closure — 2026-07-19 (CD-023/CD-024)
DONE: applied both owner-approved decisions from this session. CD-023: six
`pending-owner-approval` completeness rows approved as-is, flipped to `settled` in
`CORE-V1-COMPLETENESS.md`, C2-exit-report.md given a dated post-gate note, WP-C3-ENTRY.md
blocker 1 marked closed. CD-024: DEV-060 root-caused and fixed in `borrowck.rs::method_receiver`
(missing trait-default-body fallback, mirroring typecheck.rs's own `default_fallback`); two new
regression tests plus one rewritten; KNOWN-DEVIATIONS.md, WP-C3-ENTRY.md blocker 2, and the
open-deviation index all updated to reflect closure.
FILES: starkc/src/borrowck.rs (fix), starkc/src/typecheck.rs (rewrote
`repeated_call_to_unoverridden_default_trait_method_is_wrongly_flagged_as_move` to
`_is_no_longer_flagged_as_move`; added `repeated_call_to_unoverridden_mut_default_trait_
method_is_no_longer_flagged_as_move`), starkc/src/interp.rs (added
`repeated_call_to_unoverridden_default_trait_method_executes_correctly`),
STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md,
starkc/docs/compiler/C2-exit-report.md, starkc/docs/conformance/KNOWN-DEVIATIONS.md,
STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md, COMPILER-STATE.md.
RULES: none — no normative Core rule change; this closes a compiler defect where legal,
spec-conforming code was wrongly rejected (availability bug, not a soundness/acceptance bug).
DECISIONS: CD-023, CD-024 (both owner-approved this session).
EVIDENCE: `cargo build` clean; full `cargo test --workspace --all-targets --all-features`
596 passed / 0 failed / 2 ignored (up from 594); `cargo fmt --all -- --check` clean; `cargo
clippy --workspace --all-targets --all-features -- -D warnings` clean; `python3
starkc/scripts/check-conformance.py` re-run, unchanged (89.8%/53-of-59 — DEV-060 was a
runtime/borrowck defect, not a conformance-database entry). Root cause independently isolated
by direct code reading (borrowck.rs's `method_receiver` vs typecheck.rs's `resolve_method`),
not assumed from the ledger's prior "needs its own investigation" note.
FOLLOW-UP: corpus freeze is now unblocked (WP-C3-ENTRY.md required DEV-060 resolved first,
since a fix could legitimately change corpus output) — next actionable step; green CI run still
needs a push to origin.
NEXT: freeze the versioned execution corpus per WP-C3-ENTRY.md's procedure; then push and
obtain a green CI run; then write starkc/docs/compiler/C3-entry-exit.md; then WP-C3.1.

### C3-ENTRY blockers 3-4 closure + gate close — 2026-07-19 (CD-025)
DONE: froze the execution-snapshot corpus and closed the C3-ENTRY transition. corpus.lock
(v1.0.0, 48 files, base 3d12f45) + integrity test `corpus_lock_matches_frozen_snapshot`
(negatively verified). CI green on origin/main @ 3d12f45 (owner-confirmed). Wrote exit artifact
C3-entry-exit.md; flipped Position to Gate C3 / WP-C3.1 / Blocked: none; checked off all
WP-C3-ENTRY Done-when items. Gate C3 is open.
FILES: starkc/tests/exec_snapshots/corpus.lock (new), starkc/tests/exec_snapshots.rs (new
integrity test), starkc/docs/compiler/C3-entry-exit.md (new),
STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md, COMPILER-STATE.md.
RULES: none — freeze/governance only, no Core behavior change.
DECISIONS: CD-025.
EVIDENCE: `cargo test --test exec_snapshots` 4 passed (incl. integrity test); tamper-then-
restore negative check confirms the integrity test fails on drift; `cargo fmt --all -- --check`
and `cargo clippy --test exec_snapshots --all-features -- -D warnings` clean; full workspace
596/0/2 from CD-024 unchanged (corpus freeze adds one test → next full run will read 597/0/2).
FOLLOW-UP: none blocking. Optional pre-C5.1: draft the "Callable ABI and Future Closure
Compatibility Spike" proposal during C3 spike work (CD-021).
NEXT: WP-C3.1 — freeze the 23-item representative workload, define the measurement set, write
STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md. Gate C3 selects backend
architecture (SELECT-GENERATED / SELECT-DIRECT / REVISE / BLOCKED), never interpreter-only.

### WP-C3.1 — Architecture hypothesis and workload freeze — 2026-07-19
DONE: authored the Gate C3 setup deliverables. Wrote
`STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md` (new proposals/ dir): the four
separated questions, pipeline context, the frozen-Core decisions that favor native lowering
(trap-abort-no-unwind, no trait objects, non-capturing fn values, borrow-check-before-codegen,
deterministic order), the architecture hypothesis (Candidate A generated Rust/C vs Candidate B
direct Cranelift; leading hypothesis SELECT-GENERATED with explicit falsifiers), the frozen
23-item workload mapped to corpus v1.0.0 (items 1-10) + specified reference programs (11-23),
the risk register (both candidates, per hard construct), the 13-dimension measurement framework,
and the WP-C3.4 decision preview. Created `work-packages/WP-C3.1.md`. Set Native-backend-
selection status to SPIKING; flipped Position Next to WP-C3.2/C3.3.
FILES: STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md (new),
STARKLANG/docs/compiler/work-packages/WP-C3.1.md (new), COMPILER-STATE.md.
RULES: none — non-normative proposal; no Core semantics, compiler, or interpreter change. States
a hypothesis, freezes a workload, defines measurements; selects nothing.
DECISIONS: none at CE level. Leading hypothesis (SELECT-GENERATED) is explicitly flagged as
falsifiable orientation for the spikes, not a decision — CE5 backend selection remains the
owner's at WP-C3.4. Flagged per the CE-escalation convention.
EVIDENCE: all 15 corpus-case references + the workspace-relocation test name + the two
metamorphic pair names verified to resolve against the real tree (no dangling pointers).
Interpreter support for the harder workload items confirmed by direct source read: function
values (`Value::Function`, interp.rs:2168 indirect call), file I/O (`Value::File` +
`read_to_string`/`write`, DEV-009 resolved), references/slices. No build/test run needed — no
code changed.
FOLLOW-UP: recommended (not approved) — draft the "Callable ABI and Future Closure Compatibility
Spike" memo during C3 spike work, before WP-C5.1 freezes the ABI (CD-021). The two open fn-value
properties (Eq/Ord/Hash participation, monomorphised-generic identity) must be settled before
WP-C3.4 selection (CD-022).
NEXT: WP-C3.2 (generated Rust/C spike) and WP-C3.3 (direct Cranelift spike) — each implements
the reachable workload subset and reports every measurement dimension + unsupported constructs;
then WP-C3.4 selects under CE5.

### WP-C3.2 — Generated-Rust backend spike — 2026-07-19
DONE: built and ran the generated-Rust backend spike (Candidate A). Isolated HIR→Rust lowerer +
compile/run/diff harness in `starkc/tests/spike_genrust.rs` (charter §2.2 — NOT wired into
`stark build`, adds nothing to the library surface, disposable). Lowers a supported subset
(integer primitives i8..u64 + Bool, trap-checked arithmetic, comparisons/logic, let/mut/assign,
if/while/loop/for/break/continue, block-tail values, non-generic fns + calls, print/println)
from typed HIR to Rust, compiles with rustc, runs, compares stdout+exit-status to the interpreter
oracle over the frozen exec_snapshots corpus v1.0.0. Wrote the spike report
`starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md` (new spikes/ dir) with every WP-C3.2
measurement record + the NATIVE-CORE-ARCHITECTURE.md §7 dimension mapping. Created WP-C3.2.md.
RESULT: 4/17 corpus cases lowered and matched exactly (arithmetic/precedence,
loops/for/break/continue, multi-width ints, Int8-overflow trap→abort parity); 0 semantic
mismatches on supported cases; 13/17 cleanly reported unsupported with reasons; mean rustc
compile 87 ms/case. Candidate liabilities (rustc dep weight, compile-time scaling, exe size,
debug-info trap mapping, unsupported breadth) neither falsified nor cleared — that needs the
C3.3 spike + a breadth run.
FILES: starkc/tests/spike_genrust.rs (new), starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md
(new), STARKLANG/docs/compiler/work-packages/WP-C3.2.md (new), COMPILER-STATE.md.
RULES: none — spike/evidence only. The spike does NOT bypass front-end checks (it consumes
already-validated typed HIR) and does NOT select a backend (WP-C3.4/CE5). No Core semantics,
compiler, or interpreter change.
DECISIONS: none at CE level. Native-backend-selection status stays SPIKING.
EVIDENCE: `cargo test --test spike_genrust` 2 passed; full workspace
`cargo test --workspace --all-targets --all-features` 599 passed / 0 failed / 2 ignored (597 +
the 2 spike tests); `cargo fmt --all -- --check` and `cargo clippy --test spike_genrust
--all-features -- -D warnings` clean. Coverage table reproduced via `-- --nocapture`.
FOLLOW-UP: WP-C3.3 direct-Cranelift spike must run before any candidate comparison; dimensions
3/5/11/12/13 (exe size, runtime perf, monomorphisation, trait dispatch, ref/slice/Drop ABI) need
a breadth run on both candidates. The two open fn-value properties (CD-022) still pending
pre-C3.4.
NEXT: WP-C3.3 — direct Cranelift spike over the same frozen workload with the same measurement
record; then WP-C3.4 selects under CE5.

### WP-C3.3 — Direct (Cranelift) backend spike — 2026-07-19
DONE: built and ran the direct Cranelift backend spike (Candidate B). Isolated HIR→Cranelift-IR
lowerer + object-emission + `cc`-link + run/diff harness in `starkc/tests/spike_cranelift.rs`
(charter §2.2 — NOT wired into `stark build`, disposable). Same frozen workload subset as C3.2.
Produces a real standalone native executable. Added Cranelift dev-dependencies (pinned 0.110 for
rustc-1.93 compat, with a necessity note in Cargo.toml; dev-only, not the shipped surface).
Object emission (not JIT) → no `unsafe` (crate forbids it). Wrote report
`starkc/docs/compiler/spikes/WP-C3.3-direct-cranelift.md` with the head-to-head table vs C3.2 and
an explicit timing caveat. Created WP-C3.3.md. Native-backend-selection section updated with both
spikes' results.
RESULT: 3/17 corpus cases matched the interpreter exactly (arithmetic, loops/for/break/continue,
Int8-overflow trap→abort parity); 0 semantic mismatches; 14/17 unsupported (same families as C3.2
plus unsigned ints). Timing: Cranelift codegen ~2 ms/case (phase-only, from built IR, no
parse/typecheck/link), `cc` link ~47 ms/case; end-to-end ~49 ms vs rustc ~87 ms ≈ 1.8× on 3
trivial programs — flagged as NOT a general multiple (charter caution). No rustc build dep.
MSRV-churn finding (0.133→rustc 1.94). Higher glue than generated-Rust; weaker debug-info;
biggest MIR beneficiary.
FILES: starkc/tests/spike_cranelift.rs (new), starkc/docs/compiler/spikes/
WP-C3.3-direct-cranelift.md (new), STARKLANG/docs/compiler/work-packages/WP-C3.3.md (new),
starkc/Cargo.toml (dev-deps), COMPILER-STATE.md.
RULES: none — spike/evidence only, no front-end bypass, no backend selection (WP-C3.4/CE5), no
Core/compiler/interpreter change. Cranelift is a dev-dependency only (charter §1.10 note in
Cargo.toml).
DECISIONS: none at CE level. Native-backend-selection stays SPIKING.
EVIDENCE: `cargo test --test spike_cranelift` 1 passed; full workspace 600 passed / 0 failed / 2
ignored (599 + the cranelift spike); `cargo fmt --all -- --check` + `cargo clippy --test
spike_cranelift --all-features -- -D warnings` clean. Coverage + timings via `-- --nocapture`.
FOLLOW-UP: WP-C3.4 needs a breadth run (aggregates/generics/traits/refs/Drop/fn-values) on both
candidates and exe-size/startup/runtime measurement before a confident selection; the two open
fn-value properties (CD-022) still pending pre-selection.
NEXT: WP-C3.4 — backend and runtime architecture selection under CE5 (owner decision):
SELECT-GENERATED / SELECT-DIRECT / REVISE / BLOCKED.

### WP-C3 breadth run (both spikes) — 2026-07-19
DONE: extended the generated-Rust spike across aggregate/generic breadth (structs, impl/methods,
struct literals, field/method access, generics + trait bounds, Option/Result, match + pattern
lowering, String/&str) → 8/17 frozen corpus cases, all matching the interpreter (was 4/17). ~250
lines of mechanical text emission; rustc absorbs monomorphization/layout/ABI/Drop. Cranelift
breadth measured at the struct boundary rather than fully implemented (would need stack-slot
layout + sret ABI for structs, tagged-union layout for enums, a monomorphization engine for
generics, a runtime for String/Vec — each a subsystem), grounded in the built ~600-line Cranelift
lowerer; Cranelift stays 3/17. Wrote WP-C3-breadth-comparison.md (the head-to-head + the caveat
that most direct-backend breadth cost is mandatory MIR work anyway, so the HIR-level comparison
overstates it). Updated WP-C3.2 and WP-C3.3 reports.
FILES: starkc/tests/spike_genrust.rs (breadth extension + updated unsupported-cases test),
starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md, WP-C3.3-direct-cranelift.md,
WP-C3-breadth-comparison.md (new), COMPILER-STATE.md.
RULES: none — spike/evidence only; no front-end bypass, no backend selection, no Core/compiler/
interpreter change.
DECISIONS: none at CE level. Native-backend-selection stays SPIKING.
EVIDENCE: `cargo test --test spike_genrust` 2 passed (match-interpreter now 8/17 + updated
unsupported-cleanly test); full workspace `cargo test --workspace --all-targets --all-features`
600 passed / 0 failed / 2 ignored; `cargo fmt --all -- --check` + `cargo clippy --test
spike_genrust --all-features -- -D warnings` clean.
FOLLOW-UP: optional exact Cranelift struct head-to-head (~150-200-line sret impl); exe-size/
startup/runtime still unmeasured for both; the fair comparison is at the MIR level (Gate C4), not
HIR. The two open fn-value properties (CD-022) still pending pre-C3.4.
NEXT: WP-C3.4 — backend and runtime architecture selection under CE5 (owner): SELECT-GENERATED /
SELECT-DIRECT / REVISE / BLOCKED.

### WP-C3.4 — Backend selection (owner CE5 decision) — 2026-07-19
DONE: drafted the three-way backend-selection analysis
(`starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md`) consolidating the WP-C3.1
framework + WP-C3.2/C3.3 spikes + breadth run, with a reasoned recommendation and the required
architecture commitments; presented the decision to the owner (CE5). **Owner selected
`SELECT-GENERATED`** — generated Rust as the initial production backend behind verified MIR,
backend-neutral MIR keeping direct-Cranelift open as a C7 migration. Recorded as CD-026;
Native-backend-selection section → SELECTED / generated Rust/C; created WP-C3.4.md; Position line
advanced to Gate C4 / WP-C4.1. Gate C3 is complete.
FILES: starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md (new),
STARKLANG/docs/compiler/work-packages/WP-C3.4.md (new), COMPILER-STATE.md.
RULES: none — a strategy selection only; does not build MIR, define the MIR contract (C4/CE3), or
fix the runtime ABI (C5.1/CE4). No Core/compiler/interpreter change.
DECISIONS: CD-026 (owner CE5). Native-backend-selection = SELECTED.
EVIDENCE: decision presented and recorded; the supporting spike evidence (WP-C3.2/C3.3/breadth
reports) is unchanged and already committed. No new code; workspace baseline unchanged (600/0/2).
FOLLOW-UP: the disposable spikes (`tests/spike_genrust.rs`, `tests/spike_cranelift.rs`, Cranelift
dev-deps) are retained for now as C3 evidence; remove/rewrite them when the real MIR-consuming
generated-Rust backend lands (they are not production architecture, charter §2.2). The two open
fn-value properties (CD-022) must be settled during C4/C5. Optional: exe-size/startup measurement
and the Cranelift struct head-to-head remain available if C7 re-evaluation needs them.
NEXT: Gate C4 — WP-C4.1 (MIR design review, CE3): define the backend-neutral verified MIR contract
(`STARKLANG/docs/compiler/mir.md`) that the generated-Rust emitter consumes.

### Pre-C4.1 fn-value settlement and correction pass (CD-027) — 2026-07-19
DONE: settled both CD-022 carry-forward properties (TYPE-FN-001 non-participation in
Eq/Ord/Hash → identity unobservable; TYPE-FN-002 generic coercion = instantiate-at-coercion,
both owner-approved) as normative rules in 03-Type-System.md §Function Types; regenerated the
combined spec (fixtures unchanged — prose-only rules); added TYPE-FN-001/002 rows to the
completeness inventory (166 → 168). Discovered by first-ever execution of workload items 16-22
that the whole fn-value feature was a compile-time façade: recorded DEV-061/062/063, got owner
fix-now authorization, fixed all three (interp dispatch arm; Ty::Fn Copy classification in
borrowck+typecheck; Option/Result combinator signatures + consuming interp interception), and
recorded-but-deferred DEV-064 (undetermined-generic coercion, owner C4.5).
FILES: STARKLANG/docs/spec/03-Type-System.md (+ regenerated STARK-Core-v1.md/.html/.pdf),
STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md, starkc/src/interp.rs,
starkc/src/typecheck.rs, starkc/src/borrowck.rs, starkc/docs/conformance/KNOWN-DEVIATIONS.md,
COMPILER-STATE.md.
RULES: TYPE-FN-001, TYPE-FN-002 (new normative, owner-approved CE1/CE2 under CD-027).
DECISIONS: CD-027.
EVIDENCE: full workspace `cargo test --workspace --all-targets --all-features` **605 passed / 0
failed / 2 ignored** (600 → 605: 3 new interp tests, 2 new typecheck tests); `cargo fmt` +
`cargo clippy --workspace --all-targets --all-features -- -D warnings` clean;
`build-core-spec.py --check` in sync; fixture extraction in sync; check-conformance.py
unchanged (89.8%). All empirical claims verified by running the compiler on real programs
before recording (E0500 rejection, T1/T2/T3 failures pre-fix and outputs post-fix, combinator
outputs incl. pass-through sides, undetermined-generic acceptance).
FOLLOW-UP: DEV-064 owned by C4.5. Workload items 16-22 now have a working oracle; item 23
(Copy aggregate with fn-value field) untested — exercise during C4. The spike reports' "fn
values unsupported" rows are unaffected (spikes are frozen evidence).
NEXT: WP-C4.1 — MIR design review (CE3): draft the backend-neutral verified MIR contract
(STARKLANG/docs/compiler/mir.md) for owner review; the generated-Rust emitter consumes it.

### WP-C4.1 — MIR contract drafted (CE3 review pending) — 2026-07-19
DONE: drafted STARK MIR v0.1 (`STARKLANG/docs/compiler/mir.md`, status PROPOSED) covering every
roadmap-required element: monomorphised-only instances with deterministic injective symbol
naming; closed first-order MirTy set (no Param/Infer); return-place body model; places/
projections with CheckIndex-dominates-Index discipline; total (never-trapping) rvalue set with
every trapping operation as a Checked/Trap terminator carrying category + SourceInfo; no
unwinding/cleanup edges anywhere (abort semantics); Drop as a *statement* with ordinary Bool
drop-flag locals; direct/indirect/runtime callees (FnPtr constants per CD-021/CD-027, closed
versioned RuntimeFn surface); mandatory per-statement provenance with explicit FileId (DEV-006
lesson) and labeled synthetic origins; 13 verifier obligations mapped to WP-C4.3 with MIR-xxxx
safe-failure diagnostics; deterministic versioned textual dump. Five judgment calls flagged for
CE3 in §12. Created WP-C4.1.md.
FILES: STARKLANG/docs/compiler/mir.md (new, PROPOSED),
STARKLANG/docs/compiler/work-packages/WP-C4.1.md (new), COMPILER-STATE.md.
RULES: none — non-normative implementation contract, explicitly subordinate to
CORE-V1-ABSTRACT-MACHINE.md; binding only after CE3 approval.
DECISIONS: none yet — CE3 review is the owner's; WP-C4.2 does not open against an unapproved
contract.
EVIDENCE: design-only; no code changed; workspace baseline 605/0/2 unchanged.
FOLLOW-UP: on approval, record a CD entry flipping mir.md to APPROVED and open WP-C4.2 (scalar
HIR→MIR lowering). DEV-064 fix must land in typecheck before instance collection can rely on
full determination (C4.5 at latest).
NEXT: CE3 owner review of mir.md §12's five questions; then WP-C4.2.

### WP-C4.1 CE3 review outcome (CD-028) — 2026-07-19
DONE: owner CE3 review of the MIR v0.1 contract returned **APPROVE WITH REQUIRED CHANGES**;
all three required changes applied and the contract flipped to APPROVED. (1) Drop moved from
Statement to Terminator (`Drop { place, target }`, no unwind edge) — the review correctly
caught that the statement form violated the contract's own totality invariant, since
destructors are user code that may trap/diverge/mutate; the totality invariant is now stated
in full ("statements/rvalues never trap, never call user code, never diverge") and actually
holds. (2) Option/Result changed from opaque Core runtime types to **logical MIR enums**
(`EnumRef::CoreOption`/`CoreResult`, same aggregate/discriminant/match machinery as user
enums; physical layout stays C5.1/ABI; combinators may remain runtime calls) — the opaque form
had let the current interpreter's representation shape the IR. (3) CheckIndex/Index kept split
but the ordinary integer index local replaced with **opaque IndexProof tokens** binding
base+index+length, consumed only by Index projections on the same base (V-IDX-1/2); Vec
indexing stays on runtime ops in v0.1 (mutable length). Approved unchanged: trapping-ops-as-
terminators (with the one-normal-successor/implicit-abort refinement made explicit) and
monomorphised-only MIR (with three qualifications: mangling not a stable external ABI; named
resource limit; deduplicated discovery). Owner decision wordings recorded verbatim in mir.md
§12.
FILES: STARKLANG/docs/compiler/mir.md (APPROVED), STARKLANG/docs/compiler/work-packages/
WP-C4.1.md (closed), COMPILER-STATE.md.
RULES: none — implementation contract, subordinate to CORE-V1-ABSTRACT-MACHINE.md.
DECISIONS: CD-028 (owner CE3).
EVIDENCE: design review only; no code changed; workspace baseline 605/0/2 unchanged.
FOLLOW-UP: none blocking. DEV-064 (undetermined-generic coercion rejection) still owned by
C4.5; the CD-021 callable-ABI memo still recommended pre-C5.1.
NEXT: WP-C4.2 — typed HIR → MIR lowering, scalar core (literals/locals, unary/binary ops,
blocks/assignments, functions/calls, if/loops/break/continue/return, tuples/arrays/structs/
basic enums, pattern matching without advanced drop elaboration), with every MIR instruction
carrying real or labeled-synthetic SourceInfo.

### WP-C4.2 — Typed HIR → MIR lowering, scalar core — 2026-07-19
DONE: implemented the MIR v0.1 data model (`starkc/src/mir/mod.rs`) exactly per the approved
contract — Drop as terminator, logical Option/Result enums (EnumRef::CoreOption/CoreResult),
IndexProof local kind, Checked with one normal successor + TrapInfo, closed RuntimeFn surface,
interned FileId + SourceInfo on every statement/terminator, versioned deterministic dump — and
the scalar-core lowering (`src/mir/lower.rs`): monomorphised-only deterministic deduplicated
instance discovery from main; trapping ops as Checked terminators (int arith/neg, float
div/rem) with float add/sub/mul + comparisons as total rvalues; short-circuit &&/|| as CFG;
if/while/loop/for-range (labeled synthetic provenance)/break/continue/return; direct calls;
FnPtr constants + FnValue indirect calls (CD-021 items 16/17); tuples/arrays/structs
(written-order eval, decl-order aggregation); user enums incl. unit variants + struct-variant
literals; Option/Result construction as logical-enum aggregates and matching via
Discriminant+SwitchInt with VariantField binding; println/print via runtime surface with
uniform checked widening casts. Scalar-core drop restriction: Drop-impl types are Unsupported
(C4.5 owns elaboration). New `pub mod mir` in lib.rs.
FILES: starkc/src/mir/mod.rs (new), starkc/src/mir/lower.rs (new), starkc/src/lib.rs,
starkc/tests/mir_lowering.rs (new, 6 tests), STARKLANG/docs/compiler/work-packages/WP-C4.2.md
(new), COMPILER-STATE.md.
RULES: none — implementation of the approved contract; no Core semantics change; front-end
checks not bypassed (lowering consumes fully-checked typed HIR + TypeTables).
DECISIONS: none at CE level.
EVIDENCE: `cargo test --test mir_lowering` 6/6 (corpus scalar cases expr_stmt__01/__03,
primitive__01/__02, struct_enum_trait__02 lower with structural invariants — sealed
single-terminator blocks, in-bounds targets, valid FileId everywhere; dump deterministic +
versioned; golden mini-dump pinning Checked-Add/Cast/runtime-call/return-place shapes;
fn-value + indirect-call lowering incl. instance discovery of the target; Option lowers as
aggregate+discriminant with no runtime call; generics/strings/methods report clean Unsupported
naming C4.5). Full workspace 611 passed / 0 failed / 2 ignored (605 → 611). fmt + clippy
-D warnings clean.
FOLLOW-UP: golden documents that unsuffixed int literals infer Int32 and println's Int64
runtime signature forces an explicit (infallible, still Checked) widening cast — revisit cast
uniformity only via a contract version bump. Bool matches without a default arm and bitwise
int ops are recorded Unsupported (contract's non-trapping BinOp set lacks int bitwise ops —
flag for the C4.5-era contract addendum + version note).
NEXT: WP-C4.3 — MIR verifier (contract §10's 13 obligations, MIR-xxxx diagnostics, safe
failure); then WP-C4.4 MIR interpreter differential vs the HIR oracle.

### WP-C4.3 — MIR verifier — 2026-07-19
DONE: implemented `starkc/src/mir/verify.rs` — all 13 contract §10 obligations over MirProgram:
CFG/local/projection well-formedness with step-by-step place typing through a new
lowering-populated TypeContext (struct fields + user-enum variant payloads added to MirProgram
as an additive companion table; Option/Result payloads derived from type args); bidirectional
aggregate checking; call/checked/runtime signature checking; V-MOVE-1 as a conservative
whole-local any-path union-join fixpoint dataflow; drop-flag and index-proof (CE3 tokens)
discipline; TYPE-FN-001 enforcement at MIR level (no arithmetic/comparison on FnPtr); V-SRC-1
FileId validity. First MIR-xxxx namespace allocation recorded in the Diagnostic-codes section.
Safe-failure hardening: the negative test suite caught the move dataflow PANICKING on a broken
CFG edge (exactly the unsafe failure the contract forbids) — fixed to skip already-reported
edges; report-and-continue everywhere.
FILES: starkc/src/mir/verify.rs (new), starkc/src/mir/mod.rs (TypeContext + MirProgram.types),
starkc/src/mir/lower.rs (type-context population + hir_field_ty), starkc/tests/mir_verify.rs
(new, 14 tests), STARKLANG/docs/compiler/work-packages/WP-C4.3.md (new), COMPILER-STATE.md.
RULES: none — verifier implements the approved contract; no Core semantics change.
DECISIONS: none at CE level. MIR-0012 reserved rather than allocated (runtime-set violation is
structurally impossible while RuntimeFn is a closed Rust enum; becomes real with serialized
MIR).
EVIDENCE: `cargo test --test mir_verify` 14/14 — positive: all 5 lowerable corpus cases + 3
inline programs (fn-values, Option, structs) verify clean (lowering and verifier as two
independent contract readings agreeing); negative: 13 hand-crafted invalid bodies each
rejected with the specific MIR-xxxx code. Full workspace 625 passed / 0 failed / 2 ignored
(611 → 625: 14 verifier tests). fmt + clippy -D warnings clean.
FOLLOW-UP: V-MOVE-1 whole-local granularity documented as a refinement point (can reject
over-clever legal MIR, never accepts moved-from reads); field-precise tracking when C4.5's
partial moves need it. TypeContext addition noted as additive (no dump/shape change, no
version bump) — fold into the contract text at the next version bump.
NEXT: WP-C4.4 — MIR interpreter + differential harness vs the HIR oracle over corpus v1.0.0.

### WP-C4.4 — MIR interpreter + HIR/MIR differential — 2026-07-19
DONE: implemented `starkc/src/mir/interp.rs` (executes verified MIR: option-slot locals with
loud use-after-move detection via taking Moves; projection reads/writes; Checked terminators
with per-width trap semantics incl. MIN/-1 and CD-006 float div/rem-by-zero; checked numeric
casts; SwitchInt with the lowering's u128 key wrap; direct/indirect/runtime calls; 50M-step
fuel guard; TrapCategory outcomes distinct from internal errors) and the Gate C4 comparator
`tests/mir_differential.rs`: 7 tests running lower→verify→execute vs the HIR oracle — 5
lowerable frozen-corpus cases (byte-equal stdout+status; primitive__02 traps agree), fn-values
(CD-021 items 16/17/22 through MIR), Option/Result logical enums end-to-end, structs/tuples,
div-zero trap, mid-output trap, recursion+loops. `interp::canonical_float` exposed pub so the
MIR runtime formats floats with the oracle's own algorithm (single source, no drift).
RESULT: **zero semantic differences between HIR and MIR execution** across the supported
workload. One comparator-map bug caught by the harness itself (oracle "division by zero" vs
map's "divide by zero") — a harness fix, not an engine disagreement.
FILES: starkc/src/mir/interp.rs (new), starkc/src/mir/mod.rs (module reg),
starkc/src/interp.rs (canonical_float made pub with doc), starkc/tests/mir_differential.rs
(new, 7 tests), STARKLANG/docs/compiler/work-packages/WP-C4.4.md (new), COMPILER-STATE.md.
RULES: none — differential infrastructure; no Core semantics change. The MIR interpreter is
explicitly not a user-facing VM (charter §1.6 rule 11).
DECISIONS: none at CE level.
EVIDENCE: `cargo test --test mir_differential` 7/7; full workspace 632 passed / 0 failed /
2 ignored (625 → 632); fmt + clippy -D warnings clean. The C4.4 comparator condition — HIR
interpreter output/failure == MIR interpreter output/failure — holds for every workload the
scalar-core lowering supports.
FOLLOW-UP: the differential net must widen with every C4.5 construct as it lands (the roadmap's
"generated corpus" + full-corpus replay obligations, carried per CD-018/CD-020).
NEXT: WP-C4.5 — complete Core lowering (generics/monomorphisation, trait dispatch, patterns,
CheckIndex/indexing, strings/Vec/runtime surface, ownership/drop elaboration with real Drop
terminators, panic paths, multi-package linkage), differential-first.

### C4.5a + CD-029 correction pass — 2026-07-19
DONE: (1) WP-C4.5 split per charter §2.2 with the review-adopted increment order (WP-C4.5.md).
(2) C4.5a landed: FnKey instance identity (Top/ImplFn/TraitDefault-per-implementing-type),
method + associated-fn call lowering (receiver-before-arguments), trait dispatch with
inherent > trait-impl > default precedence, Self substitution; interim by-value reference
model documented in code (&self receivers Copy-passed; &mut self cleanly Unsupported until
C4.5b/d); corpus struct_enum_trait__01 now differential-green; 2 new differential tests
(methods/assoc fns incl. repeated &self + consuming self; trait default-vs-override).
(3) CD-029 corrections applied (see decision log): trap provenance end-to-end with exact-span
differential comparison; VerifiedMirProgram wrapper; TypeContext formalized in mir.md §2;
canonical_float spec tests (6, incl. boundary/subnormal/round-trip property).
FILES: starkc/src/mir/{lower,interp,verify}.rs, starkc/tests/{mir_differential,mir_lowering,
mir_verify,canonical_float}.rs (last new), STARKLANG/docs/compiler/mir.md (CD-029 amendments),
STARKLANG/docs/compiler/work-packages/WP-C4.5.md (new), COMPILER-STATE.md.
RULES: none — implementation + contract bookkeeping under the approved MIR contract.
DECISIONS: CD-029.
EVIDENCE: differential 9/9 with provenance comparison live (user-origin trap spans equal the
oracle's exactly in both trap tests); canonical_float 6/6; full workspace 640 passed / 0
failed / 2 ignored; fmt + clippy clean. Differential claim now stated in qualified form.
FOLLOW-UP: generated-Rust backend must consume VerifiedMirProgram when it lands (C5).
NEXT: WP-C4.5b — indexing and references (CheckIndex proof tokens, arrays/slices, real
reference places replacing the interim by-value model, &mut self).

### C4.5b-1 — array indexing with CheckIndex proof tokens — 2026-07-19
DONE: first real exercise of the CE3 proof-token discipline end to end. Lowering: `base[index]`
(reads, writes, loop-indexed access) emits `Checked { CheckIndex, args: [Copy(base_place),
index] }` defining an IndexProof local consumed by `Index(proof)` projections; base evaluated
before index (CD-007); non-place bases materialize a temp; Vec indexing stays runtime-surface,
slices deferred to C4.5b-2. Verifier: NEW same-base binding pass (`verify_index_bindings`) —
every Index(proof)'s place prefix must equal the base its CheckIndex bound (proof→base map;
place prefix equality; the exact rule CD-028's revision demanded beyond dominance), plus
CheckIndex arg typing (base must be Copy(place) of indexable type, index integer). Interp:
CheckIndex evaluates bounds and defines the proof as the checked index; place reads/writes
resolve proofs (writes pre-resolve before the mutable walk). ORACLE FIX (DEV-065, found by the
differential's category↔message mapping need): array OOB reported "use of moved or invalid
field" — now projection-kind-aware "index out of bounds"; diagnostics-only.
FILES: starkc/src/mir/{lower,verify,interp}.rs, starkc/src/mir/mod.rs (PartialEq on
Place/Projection), starkc/src/interp.rs (DEV-065), starkc/tests/{mir_differential,mir_verify}.rs,
starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-065 closed; count 63), COMPILER-STATE.md.
RULES: none — implements the approved contract; DEV-065 is diagnostics-only (no
accepted/rejected or trap-behaviour change).
DECISIONS: none at CE level.
EVIDENCE: differential 11/11 (new: array reads/writes/loop-sum agree; OOB trap agrees in
category AND exact source span through the fixed oracle message); verifier 15/15 (new negative:
proof bound to base _1 used to index _2 → MIR-0010). Full workspace 643 passed / 0 failed / 2
ignored; fmt + clippy clean.
FOLLOW-UP: C4.5b-2 (references/slices/&mut self) needs the MIR-interp frame restructure
(cross-frame reference places) — the interim by-value reference model stays until then.
NEXT: WP-C4.5b-2, then C4.5c generics per WP-C4.5.md's increment order.

### C4.5b-2 — real references and the frame-stack MIR interpreter — 2026-07-19
DONE: the interim by-value reference model is gone. MIR interpreter restructured onto an
explicit frame stack; a reference value is a resolved (frame, local, concrete-projection-path);
`Deref` re-anchors place resolution; index proofs resolve in the evaluating frame before any
re-anchor; dangling-frame access is a loud Internal error (defense behind borrowck). Lowering:
`Ty::Ref` converts to real `MirTy::Ref` (peel removed); `&expr`/`&mut expr` lower to `RefOf`
(borrow of a place, never a value read); `*r` reads/writes via `Deref` projections; field
access and method dispatch auto-deref through reference-typed bases; `&self`/`&mut self`
receivers are real Ref-typed params (borrowed at call sites, forwarded when the receiver is
already a reference). The &mut-self Unsupported is gone — a &mut self method now genuinely
mutates the CALLER's local across the frame boundary (differential-verified). ORACLE FIX
(DEV-066, the differential's second front-end find after DEV-065): borrowck consumed a
reference on every deref-read (&mut T non-Copy → "use" became a move), rejecting the canonical
`*r = *r + 1`; both deref paths now availability-check without consuming; the
move-out-of-non-Copy-pointee rejection is unchanged.
FILES: starkc/src/mir/interp.rs (frame restructure, rewritten), starkc/src/mir/lower.rs,
starkc/src/borrowck.rs (DEV-066), starkc/tests/{mir_differential,mir_lowering}.rs,
starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-066; count 64),
STARKLANG/docs/compiler/work-packages/WP-C4.5.md (b marked done; slices explicitly deferred to
C4.5e where their consumers live), COMPILER-STATE.md.
RULES: none — implements the approved contract's reference/Deref semantics; DEV-066 restores
spec-legal programs (rejection-of-legal fix, no new acceptance beyond the spec).
DECISIONS: none at CE level.
EVIDENCE: differential 14/14 — all prior tests pass unchanged under the REAL reference model,
plus 3 new: `&mut self` mutating the caller's local (read back both via method and direct
field), `&`/`&mut` arguments with cross-frame writes and derefs, `&mut` to a struct FIELD
(sibling field untouched). mir_lowering negative case swapped (mut-self now supported; `?`
takes its place). Full workspace 646 passed / 0 failed / 2 ignored; fmt + clippy clean.
FOLLOW-UP: none blocking. C4.5b complete.
NEXT: WP-C4.5c — generics and full static dispatch (real Instance.type_args monomorphisation,
deterministic dedup, named resource limit, operator dispatch on generic params, DEV-064's
typecheck rejection).

### WP-C4.7-1 — documentation/evidence reconciliation (coding-session remainder) — 2026-07-20
DONE: the three remaining C4.7-1 items from the plan (the doc half landed in the planning
commit). (1) **MIR amendment A3 recorded in `mir.md`** — the WP-C4.6 A5 arithmetic additions,
which CD-033 approved as a *class* but whose per-amendment recording the versioning policy
requires and which was missed at implementation time: `MirBinOp::BitAnd/BitOr/BitXor` as PURE
rvalues (same-width two's-complement results are always representable, so the §5 totality
invariant holds; `~x` lowers to `x ^ mask` rather than adding a `MirUnOp`), `CheckedOp::Pow`
(NUM-INT-ARITH-001, nonnegative exponent, checked intermediates), `CheckedOp::Shl`/`Shr`
activated (NUM-SHIFT-001 count bound, no masking), and `TrapCategory::InvalidShift` kept
DISTINCT from `IntegerOverflow` (a left shift still overflows on an unrepresentable result) with
the reference interpreter's `CheckedOutcome::Trap(Some(cat))` override documented as the rule a
backend must reproduce — it is the only category override in the evaluator. §5/§6 grammar blocks
updated to match. (2) **DEV-074** numbered: the A4-2e alignment of the oracle's three
slice-bound messages into the "out of bounds" family — an oracle *behavior* change that §0.5
says needs a ledger entry, previously recorded only in A1 rev. 10. CLOSED at creation (the code
is correct and spec-directed; the gap was governance). (3) A4's "complete" claim tightened to
"MIR runtime surface" in `WP-C4.6.md` and A1 rev. 10, with the front-end `core-min` holes
(`Box` deref, primitive `cmp`) pointed at WP-C4.7-6. (`Box` deref was later found
**misclassified** — spec-conformant to reject; see the WP-C4.7-6.1 record.)
FILES: STARKLANG/docs/compiler/mir.md (A3 amendment + grammar), mir-amendment-A1-strings-runtime.md
(rev. 10 wording + DEV-074 pointer), work-packages/WP-C4.6.md (A4 wording), work-packages/
WP-C4.7.md (tracker + A3→A4 renumber), starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-074;
count 71 → 72; both enumerations), COMPILER-STATE.md (Position, open-deviation index refreshed
to post-Class-A reality with C4.7 owners, count 66 → 72, this record).
RULES: none — doc-only; no code, no test, no behavior change.
DECISIONS: **two items for the owner.** (a) Post-hoc **CE3 ratification of MIR amendment A3** —
the shape additions are already implemented and shipped; the ask is ratification of the record,
not of new code. (b) **Amendment renumbering**: because this increment names the A5 arithmetic
work "A3" (as WP-C4.7 §2 C4.7-1 directs), C4.7-3's layout amendment is renumbered **A4**
(`mir-amendment-A4-layout.md`) — the plan as written would have produced two A3s.
EVIDENCE: doc-only increment; full validation gate run anyway (workspace tests, fmt, clippy on
1.93 and 1.97) to keep the per-increment discipline honest.
FOLLOW-UP: none blocking.
NEXT: WP-C4.7-2 — verifier negatives (5–6 hand-built `MirBody` cases) + clean-Unsupported
fixtures for every recorded Class-A residual, each probed with `c46_probe` first.

### WP-C4.7-2 — evidence symmetry: verifier negatives + unsupported fixtures — 2026-07-20
DONE: CD-033's evidence rule says every Class-A class carries hand-built verifier negatives and
every recorded residual is pinned by a clean-Unsupported fixture. Both halves now hold.
**Verifier negatives (6, hand-built `MirBody`s in `tests/mir_verify.rs`)** — each checked to fail
for the *intended* reason, not incidentally (verified by temporarily asserting a bogus code and
reading the actual message): `rejects_bitwise_binop_on_floats` ("bitwise BinOp on Float64",
MIR-0004 — amendment A3's integer-only rule); `rejects_pow_on_non_integer_dest` (MIR-0004 — `Pow`
must not become a float power op with different trapping); `rejects_vec_get_ref_with_wrong_dest`
(MIR-0005 — the schematic-in-T signature must not degrade to "any Option of any reference");
`rejects_chars_iter_next_on_non_iterator` (MIR-0005, fixed table);
`rejects_runtime_call_arity_mismatch` (MIR-0005 — the plan's suggested
`rejects_call_arity_against_instance` did NOT exist, so the arity path is pinned here instead of
skipped); `rejects_switch_on_float` ("SwitchInt scrutinee is non-integer Float64", MIR-0004 —
pins that A2's Char-scrutinee widening did not over-widen).
**Unsupported fixtures (4, in `unsupported_constructs_report_cleanly`)**: droppable scrutinee +
nested pattern ("A2 residual"), droppable Iterator Item, `&mut base[range]`, `unwrap_or` on a
droppable payload. Every one probed with `c46_probe` (LOWER-UNSUPPORTED) *and* `oracle_run`
(ORACLE-OK) before being added, so each demonstrably pins a MIR gap rather than a front-end one;
`front_end_src` re-asserts typecheck-cleanliness on every run. A stale comment block above the
case table (describing a generic-comparison case that no longer exists) was removed.
FINDING (changes WP-C4.7-8's shape): the plan's fixtures for **method-own generic parameters**
and **non-bare impl heads** cannot live in this test because they are **front-end-blocked** —
`impl Holder { fn first<U>(&self, a: U, b: U) -> U }` + `h.first(7, 9)` fails E0001 "expected
'U', found 'Int32'" (method-own params are not substituted at the call site at all), and
`impl<T> Wrap for Holder<Vec<T>>` + `h.wrapped()` on `Holder<Vec<Int32>>` fails E0302 "method
'wrapped' not found" (method resolution does not structurally unify non-bare impl heads, though
DEV-073 records that it does handle bare-param heads). Neither reaches lowering, so by §1's rule
both are front-end work first. C4.7-8.4/8.5 annotated in the plan.
FILES: starkc/tests/mir_verify.rs (+6 tests), starkc/tests/mir_lowering.rs (+4 fixtures, stale
comment removed), STARKLANG/docs/compiler/work-packages/WP-C4.7.md (tracker + 8.4/8.5 notes),
COMPILER-STATE.md.
RULES: none — tests only; no compiler behavior changed.
DECISIONS: none at CE level. (CD-035 from C4.7-1 still awaits owner ratification.)
EVIDENCE: workspace 752 passed / 0 failed / 2 ignored (+6); fmt clean; clippy clean on 1.93 and
1.97.
FOLLOW-UP: none blocking.
NEXT: WP-C4.7-3 — research C2.9's target-layout decision, then DRAFT `mir-amendment-A4-layout.md`
(`Rvalue::LayoutQuery`) and STOP for owner CE3 approval before writing any code.

### WP-C4.7-3 — type-preserving layout queries (MIR amendment A4, CD-036) — 2026-07-20
DONE: research → CE3 draft → owner approval → implementation, in that order (the plan's
mandatory stop was honored; no code was written before approval).
RESEARCH: the plan asked what C2.9 actually decided about target results. Answer: **CD-015
approved only that `size_of`/`align_of` are the sole target-layout exposures and that Core
promises no ABI — it fixed no per-type values.** 07's LAYOUT-QUERY-001 requires positive,
compile-time/runtime-consistent values satisfying array/field placement; LAYOUT-ABI-001 says the
values may differ between named targets and compiler versions. So the numbers are C5.1's target
contract by design, and the C4 defect is purely representational: WP-C4.6 A4-1 lowered both
builtins to `Const 8` with `T` ERASED, and the HIR oracle returns `Value::Int(8)` for every type
— the differential passed only because both engines shared one placeholder.
IMPLEMENTED (amendment §6 scope, exactly): `Rvalue::LayoutQuery { kind: LayoutKind, ty: MirTy }`
+ dump `layout_size_of(<ty>)` / `layout_align_of(<ty>)` (`mod.rs`); the
`Res::Builtin(SizeOf|AlignOf)` arm now reads the call's turbofish type through `hir_field_ty`,
which applies the active `param_subst`, so a query inside a monomorphised generic body records
the INSTANTIATION's concrete type (`lower.rs`); one verifier typing rule — dest `UInt64`, else
MIR-0004, with the queried type deliberately unconstrained because `Sized`-ness is the checked
front end's property (`verify.rs`); one `eval_rvalue` arm delegating to a new
`reference_layout(ty) -> (u64, u64)` returning `(8, 8)` — the single override point a C5 backend
replaces (`interp.rs`). Rust's exhaustiveness checking usefully forced the new variant through
all four verifier operand/place analyses (move dataflow, drop-flag discipline, proof-token scan,
place collection); a layout query has no operands and no places, so each arm is empty by
construction rather than by assumption.
BEHAVIOR: unchanged, deliberately. The HIR oracle was NOT touched, and `size_of_align_of_agree`
passes **unmodified** — that it needed no edit is the evidence that A4 moved the representation
and not the semantics.
FILES: STARKLANG/docs/compiler/mir-amendment-A4-layout.md (new, APPROVED), mir.md (amendment
list + A4 paragraph + §5 rvalue grammar + §11 dump grammar), starkc/src/mir/{mod,lower,verify,
interp}.rs, starkc/tests/{mir_lowering,mir_verify}.rs, WP-C4.7.md, COMPILER-STATE.md.
RULES: LAYOUT-QUERY-001 and LAYOUT-ABI-001 (07), 06's "target-layout queries" classification.
No spec edit was needed — the normative documents already said what A4 implements.
DECISIONS: **CD-036** (above). CD-035 (amendment A3 record) ratified by the owner in the same
exchange.
EVIDENCE: 4 new tests — `layout_queries_preserve_the_queried_type` (dump golden: primitive and
nominal types survive; the old bare constant is gone),
`layout_query_inside_a_generic_body_records_the_instantiation` (Int32 and Bool instances each
record their own type), `rejects_layout_query_with_non_uint64_dest` (MIR-0004),
`accepts_layout_query_of_any_type_into_uint64` (an unsized queried type is a legal question).
Workspace 756 passed / 0 failed / 2 ignored; fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: C5.1 replaces `reference_layout` with the named target's real layout algorithm. That
is the only place it must touch.
NEXT: WP-C4.7-4 — DEV-069, front-end multi-file span discipline (typecheck + borrowck, then the
oracle, as two commits; reproduce with throwaway two-file probes first).

### WP-C4.7-4 — DEV-069: multi-file span discipline in the front end and the oracle — 2026-07-20
DONE: **DEV-069 CLOSED**, discharging CD-033's C5 multi-file prerequisite.
ROOT CAUSE (one class, not four bugs): `typecheck.rs`, `borrowck.rs`, and `interp.rs` each hold a
single "current file" and read every `Span` against it. STARK parses each file of a `mod helper;`
program separately, so spans are FILE-RELATIVE. Reading a span against the current file is
correct for the item being CHECKED — `check_crate` already swapped `self.file` per item — and
silently wrong for every item being LOOKED UP, because the lookup scans (method resolution,
trait-default fallback, associated-fn search, `Drop` discovery, nominal name formatting) walk
ALL items in the program regardless of file. That single mistake produced all four documented
shapes: an out-of-bounds panic when the dependency file was longer, garbage method names,
unparseable literals, and wrong-field reads at runtime.
FIX, two mechanisms:
1. **`item_text(item, span)`** in all three modules, reading against the file that DECLARES
   `item` via `hir.item_files` — the map the resolver already populated and MIR's `ProgramMeta`
   already relies on, so the three engines now agree on one source of file identity. Applied to
   every cross-item read found by walking the scan loops: method resolution, trait defaults
   (which take the TRAIT's file, not the impl's), associated fns, `Drop` impls, `format_nominal`,
   `item_name`.
2. **Per-body file swap in the oracle**, which never swapped file at all. `Callable` now carries
   its declaring file, and all THREE body-execution funnels save/restore `self.file` around the
   body: `call_callable`, `call_user_method`, and the destructor path in `drop_value`. Restored
   on error paths too, and AFTER `cleanup_current_frame` on success, since a body's own
   destructors still belong to its file. Finding the second and third funnels took empirical
   probing — fixing only `call_callable` left cross-file methods broken, and fixing that left
   cross-file destructors broken.
`text()` is additionally non-panicking now in all three modules (`.get(..).unwrap_or("?")`): a
residual wrong-file read degrades to a visible `"?"` in a diagnostic instead of aborting the
compiler. That is a backstop, not the mechanism.
FILES: starkc/src/{typecheck,borrowck,interp}.rs, starkc/tests/multi_file_spans.rs (new),
starkc/tests/mir_differential.rs (widened), KNOWN-DEVIATIONS.md (DEV-069 closed + both
enumerations), WP-C4.7.md, COMPILER-STATE.md.
RULES: none normative — this is an implementation defect against 07-Modules-and-Packages'
multi-file model; no spec text changed and no accept/reject decision changed for single-file
programs (759 tests, all pre-existing ones unchanged).
DECISIONS: one deviation from the plan, recorded: the plan said do this in TWO commits
(typecheck+borrowck, then the oracle). Landed as ONE, because the regression tests exercise both
halves end-to-end — a typecheck-only commit would have pushed red tests, which the per-increment
green-CI rule forbids. The two halves are separable in review by module.
EVIDENCE: `tests/multi_file_spans.rs` — one test per failure shape, each checked AND executed:
cross-file methods/fields/literals (33/11/66/12345), a long-dependency-file panic guard, and
cross-file trait dispatch + `Drop` where destructor ORDER is the observable (40/1/4). The
multi-file differential test was WIDENED off the safe subset it had been pinned to — now a
cross-file struct with methods, a cross-file literal, a cross-file field read, and a cross-file
`Drop` impl — with the exact expected output asserted so two engines agreeing on nothing cannot
pass. Workspace 759 passed / 0 failed / 2 ignored (+3); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none. C5 may now claim normal multi-file support.
NEXT: WP-C4.7-5 — DEV-072 (borrowck: move-out-of-shared-borrow via match bindings) and DEV-073
(typecheck: generic impls satisfying operator/iterable bounds).

### WP-C4.7-5 — DEV-072 + DEV-073 (front-end typecheck/borrowck) — 2026-07-20
DONE: both deviations CLOSED. They are opposite failures — one over-rejection, one
under-rejection — and both came down to the front end and MIR answering the same question with
different machinery.
**DEV-073 (over-rejection, typecheck).** The visible symptoms were `impl<T> Eq for W<T>` not
satisfying `W<Int32>: Eq` (E0500) and `impl<T> Iterator for Repeat<T>` not making `Repeat<Int32>`
iterable (E0001). The root cause sat one level below both checks:
`type_from_hir_without_diagnostics` **drops generic arguments** (`Ty::Struct(item, Vec::new())`).
That is invisible while the only consumers compare NON-generic nominals — `struct P` converts to
`Struct(id, [])` either way — but it meant an impl's written `W<T>` converted to `W<>`, whose
argument count could never equal `W<Int32>`'s, so the exact-match test failed for every generic
impl. Fix: a new `impl_self_ty_with_args(impl_item, ty)` that preserves the arguments and keeps
parameters as `Ty::Param`, with both checks unifying through **`match_impl_type`** — the same
one-way unification METHOD RESOLUTION already used for this exact question. That asymmetry is why
method calls on generic nominals had always worked while operators and `for` loops on the same
types did not; the two paths now agree by construction. The iterable half additionally applies
the resulting substitution to the associated type, so `type Item = T` on `Repeat<Int32>` yields
`Int32` instead of a dangling parameter.
**MIR needed no change at all** — WP-C4.6 A1 had already made dispatch instantiation-ready, and
both programs lowered and ran correctly the moment the checker admitted them. The plan predicted
this and flagged that a lowering break would be a real finding; there was none.
**DEV-072 (under-rejection, borrowck).** `borrowck.rs`'s `match` handling inspected no patterns
whatsoever, so binding a non-`Copy` payload out of a scrutinee read through a reference — a move
out of a borrow — passed the front end while MIR refused it. The two engines disagreed about
whether the program was legal, and the oracle's legacy clone semantics hid the unsoundness at
runtime by consuming the clone rather than the referent. Fix: borrowck now classifies the
scrutinee with `scrutinee_reads_through_ref`, a deliberate mirror of MIR lowering's function of
the same name (so the classification cannot drift again), and walks each arm's pattern
recursively — nested tuple/array/struct patterns and shorthand struct-field bindings included —
reporting E0101 for any non-`Copy` binding. Shared and mutable derefs both count.
What stays LEGAL mattered as much as what does not: wildcards, literals, and unit-variant paths
bind nothing, and `Copy` bindings copy rather than move. A fix that rejected all by-reference
matching would have been "correct" against the repro while breaking far more than it repaired, so
both positives are pinned by tests. The MIR guard is KEPT as defense in depth, with its message
updated to say it is unreachable for checked programs — the charter's rule is that nothing
unsupported reaches a backend silently, and an unreachable guard costs nothing.
FILES: starkc/src/typecheck.rs (`impl_self_ty_with_args`, operator-bound + iterable checks),
starkc/src/borrowck.rs (`scrutinee_reads_through_ref`, `reject_moves_out_of_borrow`),
starkc/src/mir/lower.rs (guard comment only), starkc/tests/{mir_differential,gate2_valid}.rs,
KNOWN-DEVIATIONS.md (both closed, both enumerations), WP-C4.7.md (tracker + the now-stale §1
quirk notes struck), COMPILER-STATE.md.
RULES: 03-Type-System operator traits and the `Iterator` for-protocol (DEV-073); the ownership
rule that a borrow never transfers ownership (DEV-072). No spec text changed.
DECISIONS: none at CE level.
EVIDENCE: `mir_differential.rs::generic_impl_eq_dispatch_agrees` and
`::generic_user_iterator_for_loop_agrees` — the two tests DEV-073 had blocked, added back per the
plan; `gate2_valid.rs::binding_a_non_copy_payload_through_a_reference_is_rejected` (E0101) and
`::matching_through_a_reference_without_taking_ownership_is_accepted` (wildcard + Copy positives);
`match_deref_self_noncopy_wildcard_agree` still green unchanged. Workspace 763 passed / 0 failed /
2 ignored (+4); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none.
NEXT: WP-C4.7-6 — front-end `core-min` completions. 6.1 `Box<T>` deref (report before implementing
the MIR half if it needs a new MirTy/runtime op — §0.5 stop), 6.2 primitive `cmp`/`Ordering`,
6.3 the integer-literal-typing question, which is an **OWNER DECISION** (check 03's literal
inference rules first; if 03 forbids the coercion, record as spec-conformant and close as
"not a bug").

### WP-C4.7-6 — front-end `core-min` completions: 6.2 done, 6.1 and 6.3 to the owner — 2026-07-20
DONE (6.2): **primitive `Ord::cmp`.** 06-Standard-Library specifies `impl Ord for Int32 { fn cmp
(&self, other: &Int32) -> Ordering }` and "similar for other types"; `Ordering` is `core-min`
prelude. `3.cmp(&5)` nevertheless failed E0304 "method call on non-struct/enum type" — primitives
had no `cmp` entry at all, so the only way to obtain an `Ordering` value was a user-defined `Ord`
impl. Implemented in all three engines: (a) checker — a `cmp` entry in the core-method surface
returning `Core(Ordering)` with an `&Self` parameter; (b) oracle — evaluated through the existing
`Ord for Value`, i.e. the SAME comparison the `<` operator path and sorted-collection iteration
already use; (c) MIR — `lower_primitive_cmp` computes the comparisons `<`/`==` already lower
(routing `String`/`str` through the existing `StrCmp`) and CONSTRUCTS the `CoreOrdering` variant
from them. That is the exact inverse of `lower_user_ord`, which calls a user `cmp` and switches
on the resulting discriminant. **No new MIR shape, no new `RuntimeFn`, no surface bump** — the
dispatch is placed before the String/Vec/HashMap runtime dispatches, since `String` is a
primitive receiver for this purpose. Both operands are read into temps before branching, so each
is evaluated exactly once, receiver before argument (EXEC-ONCE-001).
FOUND WHILE SCOPING 6.2 — **DEV-075**, pre-existing and unrelated to this change: the checker
accepts ordered comparison on `Bool` and `Char`, but `false < true` fails in BOTH engines
("invalid binary operation" / `BinOp Lt on Bool`) — an accept-then-fail — and `'a' < 'b'`
**succeeds in MIR while the oracle rejects it**, an engine divergence of exactly the kind the
differential exists to catch, missed only because no test compares an ordered operator on `Char`.
`cmp` was therefore scoped to integers + `String`/`str` rather than built on this gap; enabling
`Bool`/`Char` belongs in the change that closes DEV-075. Fixing it needs a spec reading — 03
gives primitives "built-in meaning (Numeric Semantics below)", which addresses numeric types and
does not settle `Bool`/`Char` ordering — so it is not a pure code fix. Ledger count 72 → 73.
TO THE OWNER — both remaining items contradict the plan's framing of them:
**6.1 `Box<T>`.** The plan (and the WP-C4.6 audit) called "`Box` deref" a `core-min` hole. The
spec says otherwise: 06 defines `Box<T>` with exactly `new` and `into_inner`; there is **no
`Deref` trait in Core v1** (not among core-min's essential traits); TYPE-METHOD-002's
auto-dereference "repeatedly removes one leading `&`/`&mut`" — references only; and the abstract
machine's Dereference operates on "the reference". So `*Box::new(5)` failing E0001 is
**spec-conformant**, and the audit's classification was wrong. The REAL gap is one level over:
`Box::new(v).into_inner()` is typecheck-clean and oracle-supported but **MIR-unsupported**
("type Core(Box, [...]) (C4.5)"). Closing it is a §0.5-class decision either way — an honest
representation needs `BoxNew`/`BoxIntoInner` runtime ops plus a surface bump, while the tempting
alternative (lower `Box<T>` transparently as `T`, since Core v1 makes addresses unobservable) is
a semantic claim that recursive types through `Box` would break; the front end already accepts
`struct Node { next: Box<Node> }`.
**6.3 integer-literal typing.** The plan hedged that 03 might FORBID a literal adopting an
expected `UInt64`, in which case the item closes as "not a bug". 03 says the opposite, and says
it twice: expected types "flow inward from ... **function parameters** ...", and defaulting
applies only to "an **unconstrained** integer literal". A literal in a `UInt64` parameter
position is constrained, so defaulting to `Int32` must not apply — this is expected-type
propagation, not a coercion (step 4 limits coercions to explicit sites), so it does not collide
with the no-implicit-coercion rule either. `v.get(0)` failing "expected 'UInt64', found 'Int32'"
is therefore a **real conformance bug (over-rejection)**, not spec-conformant behavior.
FILES: starkc/src/{typecheck,interp}.rs, starkc/src/mir/lower.rs, starkc/tests/mir_differential.rs,
KNOWN-DEVIATIONS.md (DEV-075; count 72 → 73; both enumerations), COMPILER-STATE.md.
RULES: 06's `Ord` impls for primitives and `core-min` prelude `Ordering`; CD-015 (floats are not
`Ord`). No spec text changed.
DECISIONS: none taken at CE level; two put TO the owner (6.1, 6.3, above).
EVIDENCE: `mir_differential.rs::primitive_cmp_agrees` (Less/Equal/Greater over integers and
`String`, plus a local receiver) and `::primitive_cmp_and_ordered_operators_agree`, which states
the consistency property as a test rather than assuming it: for the same pair, the variant `cmp`
reports and the answer `<`/`==` give must never disagree. Workspace 765 passed / 0 failed /
2 ignored (+2); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: DEV-075; owner decisions on 6.1 and 6.3.
NEXT: blocked on the two owner decisions; C4.7-7 (DEV-067 + DEV-071) is independent and can
proceed meanwhile.

### WP-C4.7-7 — DEV-067 + DEV-071 (bounded generics; Ordering exhaustiveness) — 2026-07-20
DONE: both CLOSED. **With this increment every front-end deviation the C4 track owned is closed.**
The remaining open ledger entries are the long-standing unscheduled ones (DEV-005/010/011/012/017)
and DEV-075, which C4.7-6.2 opened the same day.
**DEV-071 (exhaustiveness).** The prelude `Ordering` is `Ty::Core(CoreType::Ordering)` whose
variants resolve to `Res::Builtin`, which makes it structurally identical to `Option`/`Result` —
and invisible to the `Ty::Enum`/`matched_variants` machinery for exactly the same reason those
two were, before WP-C1.5 gave them explicit arms. `Ordering` never got one, so it fell through to
the same WP-C1.5 default that requires a wildcard for any domain the checker cannot enumerate.
The check now tracks `Less`/`Equal`/`Greater` and treats all three as exhaustive. The enumeration
is exact, and that matters: an over-generous domain would silently accept genuinely non-exhaustive
matches, so a two-variant match staying E0303 is pinned by its own test.
**DEV-067 (bounded generics).** One ledger entry, two independent causes:
- **(b) behind `&T`.** The bounded-parameter method lookup tested the UNPEELED receiver type, so
  it matched `t: T` but never `t: &T`. TYPE-METHOD-002 requires auto-dereference to peel leading
  `&`/`&mut` before receiver matching — and the concrete-type path immediately below already
  computed exactly such a peeled `receiver_ty`. The peel was simply performed *after* the
  parameter check instead of before it; moving it above makes both paths obey one rule.
- **(a) at intra-generic call sites.** `satisfies_bound` had **no `Ty::Param` arm at all** and
  fell through to `_ => false`, so a caller's own `T: Ord` could never discharge a callee's
  (TYPE-GENERIC-001). Adding the arm alone did not fix it — the probe still failed — because
  trait-bound obligations are collected during body checking and verified in a **deferred pass**
  that runs after every body, by which time `current_fn_generics` belongs to whatever was checked
  last. Each obligation now carries the generic environment it arose in, and the deferred pass
  restores it. The new arm mirrors the one `ty_satisfies_operator_bound` already had, so the two
  bound checks finally agree about what a parameter satisfies.
SOUNDNESS: over-rejection removed, nothing newly accepted. An obligation is discharged only by a
bound the enclosing function actually declared — both a concrete type lacking the impl and an
UNBOUNDED parameter forwarded into a bounded position are still E0500, each pinned by a test,
because "relax a check" is exactly the kind of change that silently over-accepts.
FILES: starkc/src/typecheck.rs (exhaustiveness arms; receiver peel order; `Ty::Param` bound arm;
`bounds_checks` carries its generic environment), starkc/tests/{mir_differential,gate2_valid}.rs,
KNOWN-DEVIATIONS.md (both closed, both enumerations), WP-C4.7.md (tracker + the DEV-071 §1 quirk
note struck), COMPILER-STATE.md.
RULES: TYPE-METHOD-002 (auto-dereference before receiver matching), TYPE-GENERIC-001 (the caller's
bound discharges the callee's obligation), 04-Semantic-Analysis exhaustiveness. No spec change.
DECISIONS: none at CE level.
EVIDENCE: `bounded_generic_method_through_reference_agrees` (instantiated at TWO types, so
monomorphised dispatch is exercised and not merely the check), `bounded_generic_call_chain_agrees`
(three-deep bounded chain), `unsatisfied_trait_bounds_are_still_rejected` (both negatives),
`ordering_match_exhaustiveness_counts_all_three_variants` (both directions), and
`ordering_value_round_trips_through_match_agree` **rewritten to three explicit arms** — dropping
the `_` workaround it carried for DEV-071 is what makes it exercise the exhaustiveness path.
Workspace 769 passed / 0 failed / 2 ignored (+4); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none.
NEXT: the two owner decisions (6.1 `Box`, 6.3 literal typing), then C4.7-8 (MIR residuals; 8.6
mutable slices is itself an owner decision) and C4.7-9 (fresh audit + exit report).

### WP-C4.7-6.1 — `Box<T>` on the MIR runtime surface (`0.1-A7`), owner option (a) — 2026-07-20
DONE: `Box<T>` reaches MIR. Implemented exactly to the owner's decision: an **opaque owning**
runtime type, `RuntimeFn::BoxNew` + `BoxIntoInner` activated through the dated A1-amendment
mechanism (rev. 11), surface **`0.1-A6` → `0.1-A7`**, representation stays
`MirTy::Core(Box, [T])` with **no new `MirTy`**, and explicitly NOT lowered transparently as `T`.
AUDIT CORRECTION (owner-directed): the WP-C4.6 gate audit listed "`Box` deref" as a `core-min`
hole. It is not one. Core v1 has **no `Deref` trait** (absent from `core-min`'s essential-trait
list), TYPE-METHOD-002's auto-dereference removes only leading `&`/`&mut`, the abstract machine's
dereference operates on *the reference*, and 06 gives `Box<T>` exactly `new` and `into_inner`.
`*Box::new(5)` failing is therefore **specification-conformant** and is now pinned by a negative
front-end test so a later session cannot "fix" conformant behaviour. The real gap was the
construction/extraction pair — typecheck-clean and oracle-supported, but with no MIR lowering at
all — which is what this increment closes.
SEMANTICS: `BoxNew(T) -> Box<T>` consumes its argument exactly once. `BoxIntoInner(Box<T>) -> T`
consumes the box and transfers the value out **without dropping it** (ownership moves to the
caller), releasing the allocation. There is **no public box-drop operation**: ordinary
destruction goes through the existing `Drop` terminator's structural glue, which drops the
contained `T` exactly once and then releases the allocation. A box consumed by `into_inner` holds
nothing, so nothing drops twice. Allocation failure stays a classified host/resource failure, not
a language trap (the reference interpreter cannot fail to allocate and raises none). Interpreter
representation is a one-element aggregate — addresses are unobservable (LAYOUT-QUERY-001), so the
reference engine models only the observable fact that the box OWNS its value.
THREE PRE-EXISTING DEFECTS surfaced while implementing this; none was in the plan:
1. **Drop-instance discovery never descended into `Core` container type arguments.** A
   `Box<Tag>`'s `Drop` terminator was emitted correctly and then silently found no destructor
   registered — the box dropped nothing at all. The walk now descends into every `Core`
   container's arguments (which also makes the Vec path robust rather than incidentally correct).
2. **That walk had no cycle guard**, which only mattered once `Box` made types recursive:
   `Node -> Option<Box<Node>> -> Box<Node> -> Node` overflowed the stack (observed, not
   theorised). Guarded by a visited-type set — right regardless, since a type's dtor instances
   need discovering once.
3. **DEV-077** (opened and CLOSED here): the oracle's `Box::into_inner` read its receiver through
   the *borrowing* method path, which operates on a CLONE. `.take()` emptied the clone while the
   original box kept the value and destroyed it again at scope end — an observable double drop
   with a `Drop` payload, and a divergence from MIR, which was correct. It now consumes the real
   place via `take_place`, exactly like the pre-existing `File::close` case beside it. The
   differential could not agree until the oracle was right, which is how it was caught.
FILES: starkc/src/mir/{mod,lower,verify,interp}.rs, starkc/src/interp.rs (DEV-077),
starkc/tests/{mir_differential,mir_verify,mir_lowering,gate2_valid}.rs (incl. the two
surface-string goldens the plan's §1 warns about), mir-amendment-A1-strings-runtime.md (rev. 11),
KNOWN-DEVIATIONS.md (DEV-077 closed; count 74 → 75), COMPILER-STATE.md.
RULES: 06's `core-min` `Box<T>`; TYPE-METHOD-002; LAYOUT-QUERY-001 (addresses unobservable);
EXEC-ONCE-001 (the DEV-077 double drop). No spec text changed.
DECISIONS: implements the owner's 6.1 decision (option (a)); no new CE-level decision taken.
EVIDENCE: `box_new_and_into_inner_agree`; `box_drop_timing_agrees` (exact destructor interleaving
— printed ORDER is the assertion, not a multiset); `box_recursive_type_agrees` (a finite value of
a recursive type, which is the whole reason Box stays opaque, and which also pins the cycle
guard); `rejects_box_into_inner_on_non_box` and `rejects_box_new_with_mismatched_dest` (verifier);
`box_deref_is_rejected` (front-end negative). Workspace 775 passed / 0 failed / 2 ignored (+6);
fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none for Box.
NEXT: WP-C4.7-6.3 (integer-literal expected typing — owner-decided to fix), then DEV-075
(Char/Bool ordering + the normative primitive trait/operator matrix, which requires spec edits and
regenerating the compiled spec).

### WP-C4.7-6.3 — expected typing of integer literals (DEV-078) — 2026-07-20
DONE, per the owner's decision that this is a real Core conformance defect rather than
spec-conformant behaviour. The evidence for that reading is 03-Type-System stating it twice:
expected types "flow inward from explicit annotations, **function parameters**, return types,
assignment destinations, aggregate fields, …", and solving step 5 defaults only "an
**unconstrained** integer literal". A literal in a `UInt64` parameter position is constrained, so
defaulting must not apply.
PREVIOUS BEHAVIOUR: the checker assigned `Int32`/`Int64` **at the literal**, before any
expectation could reach it. `takes_u64(0)`, `v.get(0)`, `let a: UInt64 = 9`, and a `UInt64`
struct-field initializer were all `E0001 expected 'UInt64', found 'Int32'`. It had been recorded
as a "`Vec::get` literal-typing quirk", which understated it — nothing about it was specific to
`Vec::get`, and the `0 as UInt64` workaround had been trained into the corpus and into WP-C4.7
§1's guidance for test authors.
IMPLEMENTED as general expected-type inference: an unsuffixed literal takes a fresh
**integer-kinded** inference variable; ordinary unification carries the expected type in; and
03's step 5 becomes a real pass (`default_unconstrained_int_literals`) that runs after every body
is checked and before the deferred bound checks. Binding a literal variable **range-checks** the
value (`takes_u8(300)` → E0008 at compile time, not truncation). The kind restriction is what
keeps this from being an implicit-conversion hole: the variable unifies only with primitive
integer types (plus `!` for the never-coercion rule and error-recovery types), so an integer
literal cannot satisfy a `Bool` parameter. And because this is propagation rather than coercion —
03's step 4 confines coercions to explicit coercion sites — a SUFFIXED literal (`0i32`) and a
TYPED value (`x: Int32`) both still fail against `UInt64`, which is the whole point.
TWO PLACES MUST SETTLE A LITERAL EAGERLY, because they branch on a concrete type and have no
later constraint to wait for: method-call receivers (`3.cmp(&5)` — otherwise "method call on
non-struct/enum type '_infer_N'") and cast operands (`5 as UInt8` — otherwise "casts are permitted
only between numeric types").
SUBTLETY WORTH RECORDING: a literal variable is frequently unified with ANOTHER variable rather
than a concrete type — `MyOpt::Some2(7)` unifies it with the enum's element variable. Defaulting
only variables absent from the substitution therefore left such chains unbound while they LOOKED
constrained, and they surfaced as `type Infer(N)` at MIR lowering. Defaulting resolves first and
defaults the end of the chain.
FILES: starkc/src/typecheck.rs (literal site, integer-kinded binding, defaulting pass, eager
settle at receivers/casts, array-repeat count), starkc/src/literal.rs
(`primitive_int_range_contains`), starkc/tests/{gate2_valid,mir_differential}.rs,
KNOWN-DEVIATIONS.md (DEV-078 closed; count 75 → 76), COMPILER-STATE.md.
RULES: 03-Type-System's inference algorithm (inward expected types; step 5 defaulting; step 4
coercion confinement). No spec text changed — the spec already required this.
DECISIONS: implements the owner's 6.3 decision; no new CE-level decision.
EVIDENCE: `unsuffixed_integer_literals_adopt_the_expected_integer_type` (parameter, annotation,
struct field, and the TYPE-INFER-001 later-use case `let index = 0; v.get(index)`);
`integer_literal_typing_negatives_still_fail` (range, suffix, typed value, non-integer kind — four
different reasons, all of which must keep failing); `expected_typed_integer_literals_agree`
(differential — adopted widths are observable at runtime through `UInt64` arithmetic and indexing,
so checker-side agreement alone would not be evidence). Unnecessary `as UInt64` casts removed from
the differential corpus; casts of genuinely typed values retained. Workspace 778 passed / 0 failed
/ 2 ignored (+3); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: WP-C4.7 §1's "integer literals don't coerce to `UInt64`" guidance is now obsolete and
has been struck.
NEXT: DEV-075 (Char/Bool ordering + the normative primitive trait/operator matrix — requires spec
source edits and regenerating the compiled spec).

### WP-C4.7 — DEV-075: Char/Bool ordering and the normative primitive matrix — 2026-07-20
DONE: DEV-075 CLOSED under an **owner specification decision**. This is the first change to
normative spec text in WP-C4.7.
THE DECISION (owner, 2026-07-20) split the two types rather than treating DEV-075 as one gap:
- **`Char` is totally ordered by Unicode scalar value** — implements `Eq`, `Ord`, `Hash`; all four
  ordered operators compare scalar values; `Char::cmp` returns the corresponding `Ordering`.
  Explicitly NOT locale-sensitive or linguistic collation, and Core v1 offers no collation
  facility.
- **`Bool` implements `Eq` and `Hash` but NOT `Ord`** — `<`, `<=`, `>`, `>=` and `Bool::cmp` are
  compile-time errors; `==`/`!=` remain valid. An ordering is definable, but Core v1 has no use
  for ordering truth values, and rejecting is clearer than fixing an arbitrary one.
IMPLEMENTED: the divergence ran in `Char`'s favour — MIR executed `'a' < 'b'` correctly while the
oracle rejected it — so the ORACLE was aligned to MIR (a `(Char, Char)` arm in `eval_binary`,
matching Rust's scalar-value `char: Ord`), and `Char` joined the primitive `cmp` surface in both
the checker and lowering. `Bool` was removed from the `Ord` operator gate, which is what turns
`false < true` from an accept-then-fail into a diagnostic.
SPEC CHANGE: **`PRIM-TRAIT-001`**, a normative "Primitive Trait and Operator Matrix" in
06-Standard-Library, replacing the illustrative `impl Eq for Int32` plus `// ... similar for other
types` — which the owner correctly identified as not being a specification at all. 03-Type-System's
operator table now cross-references it. `STARK-Core-v1.md`/`.html`/`.pdf` regenerated via
`build-core-spec.py`; the spec-fixture corpus re-extracted with `extract-spec-examples.sh` (one
fixture changed, 112 blocks, manifest in sync).
THE DISTINCTION THE MATRIX FORCED: for primitives, operators have built-in meaning and do **not**
dispatch through the traits, so the operator question and the trait question are separate. The
float row is where they differ: `Float64` admits `<` and `==` as built-in IEEE operations (CD-006)
while implementing neither `Eq` nor `Ord`, because IEEE comparison is neither an equivalence
relation nor a total order — NaN is unordered and unequal to itself — so `Float64` cannot satisfy
a `T: Ord` bound or key a `HashMap`. Conflating the two gates silently broke ordinary float
comparison during implementation (`1.5 < 2.5` started failing E0500); the operator gate
(`ty_satisfies_operator_bound`) and the trait gate (`satisfies_bound`) now carry the matrix
separately, and both directions are pinned by a test.
FILES: STARKLANG/docs/spec/{06-Standard-Library,03-Type-System}.md (+ regenerated
STARK-Core-v1.{md,html,pdf}), STARKLANG/tests/spec-fixtures/06-Standard-Library__18.stark,
starkc/src/{interp,typecheck}.rs, starkc/src/mir/lower.rs,
starkc/tests/{mir_differential,gate2_valid}.rs, KNOWN-DEVIATIONS.md (DEV-075 closed; both
enumerations), COMPILER-STATE.md.
RULES: new **PRIM-TRAIT-001**; consistent with CD-015 (floats are not `Eq`/`Ord`/`Hash`) and
CD-006 (IEEE float operations).
DECISIONS: owner specification decision, recorded above; no CE-level decision taken by the session.
EVIDENCE: `char_ordering_agrees` (all four operators + `cmp`, both engines) and
`char_ordering_is_scalar_value_not_collation_agrees` — the second deliberately uses `'Z' < 'a'`
and `'0' < 'A'`, comparisons a COLLATION order would get wrong, so it distinguishes the specified
rule from a plausible alternative rather than merely re-testing that comparison works;
`bool_is_not_ordered` (four operators + `Bool::cmp` rejected, `==` still accepted);
`floats_compare_but_do_not_satisfy_ord_bounds` (both sides of the operator/trait distinction).
OBSERVABLE NARROWING (intended, and worth stating plainly): because primitive floats no longer
satisfy `T: Ord`, a bounded generic can no longer be INSTANTIATED at a float —
`fn largest<T: Ord>(..)` called as `largest(2.5, 1.5)` was legal before and is now E0500. One
existing differential test did exactly that; it was updated to instantiate `largest` at `Int32`
and `Char` (both `Ord`) while `twice<T: Num>` keeps the float instantiation, since `Num` does
include floats. That preserves the test's real subject — multiple primitive instantiations of a
bounded generic — and adds positive `Char`-as-`Ord` coverage. This failure only surfaced in a
COMPLETE workspace run; several partial runs never reached `mir_differential`.
FOLLOW-UP: none.
NEXT: C4.7-8. **8.1 is blocked on DEV-076** (the oracle's `unwrap_or` double-drop must be fixed
before MIR is built to match it); 8.4/8.5 were reclassified front-end-first by C4.7-2; 8.6
(mutable slices) is an owner decision.

### WP-C4.7-8.1a — DEV-076: the oracle's `unwrap_or` drop semantics — 2026-07-20
DONE: DEV-076 CLOSED. This is the oracle half of C4.7-8.1, split out and landed on its own
because it is a SOUNDNESS fix that is independently valuable and is a hard prerequisite for the
MIR half — §0.6 makes the oracle the semantics authority MIR must match, and an oracle that
double-drops is not an authority, it is a bug that would have been faithfully copied into MIR.
THE DEFECT: with a `Drop`-carrying payload, `Option::unwrap_or` destroyed the payload **twice**
and the discarded default **never**. Root cause identical to DEV-077: `unwrap_or` was handled on
the *borrowing* method path, which operates on a CLONE of the receiver, so taking the payload
emptied the clone while the original `Option` kept it and destroyed it again at end of scope. The
default fared worse — nothing consumed it, so its destructor never ran at all. (Core has no
laziness, so the default is always *evaluated*, which is exactly why it always owes a
destruction.) Both halves violate EXEC-ONCE-001.
FIX: `unwrap_or` now consumes the receiver from the real place (`take_place`), joining
`into_inner`/`close` at the same interception point, and explicitly drops whichever value it
discards — on `Some`/`Ok` it yields the payload and drops the default; on `None` it yields the
default; on `Err` it yields the default and drops the displaced error payload.
PINNED TIMING (the point of doing this first, and NOT the obvious answer): the discarded default
is destroyed **at the `unwrap_or` call**, not at end of scope. For
`let t = Some(Tag{1}).unwrap_or(Tag{2})` the observable order is `2` then `1`. Before the fix it
was `1`, `1` — the payload twice, the default never. Any MIR lowering written against the old
behaviour would have encoded a double drop into the backend contract.
MIR HALF: still open, still a CLEAN `Unsupported` ("unwrap_or on a droppable payload type"). A
first attempt at the lowering is deliberately NOT in this commit: moving a payload out of a
**drop-tracked** local through a `VariantField` projection is refused by the C4.5d guard ("move
through a non-field projection of a drop-tracked local"), so the consuming path needs the
drop-flag machinery `lower_enum_match` uses (`consume_variant_payload`/`consume_field`). That is
real work rather than a small extension, and landing a half-built lowering — which regressed the
Unsupported message from the precise one to a confusing internal one — would have been worse than
leaving the construct cleanly refused. It is now writable against a correct oracle.
FILES: starkc/src/interp.rs, KNOWN-DEVIATIONS.md (DEV-076 closed; both enumerations),
COMPILER-STATE.md, WP-C4.7.md.
RULES: EXEC-ONCE-001 / DROP-ORDER-001 (every value's destructor runs exactly once).
DECISIONS: none at CE level — §0.5 permits an oracle behaviour change that a DEV entry documents,
and DEV-076 is that entry, written before the fix.
EVIDENCE: probe programs with printing destructors, run through `oracle_run`, covering all three
paths — `Some` with a discarded default (`100 2 200 1 300 1`), `None` (`100 200 2 300 2`), and the
minimal ordering case (`100 2 999 1`). MIR continues to refuse the construct cleanly, so the
differential is unchanged and no test needed rewriting.
FOLLOW-UP: the MIR half of C4.7-8.1.
NEXT: droppable `unwrap_or` lowering via the drop-flag machinery, then 8.2 (droppable Iterator
Item) and 8.3 (droppable scrutinee + nested patterns, the hardest piece).

### WP-C4.7-8.1 — droppable `unwrap_or` lowering (MIR half) — 2026-07-20
DONE: C4.7-8.1 complete. The oracle half landed as 8.1a (DEV-076); this is the lowering, written
against the corrected oracle rather than against the double drop it used to exhibit.
SEMANTICS IMPLEMENTED (pinned empirically first, per §0.6): `unwrap_or` discards exactly one of
two values and the discarded one owes a destructor — Core has no laziness, so the default is
evaluated whether or not it is used, which is exactly why it always owes one. The discarded value
is destroyed **at the call**, not at end of scope. On `Some`/`Ok`: yield the payload, drop the
default there. On `None`: yield the default. On `Err`: yield the default and drop the displaced
error payload — the case with no `Option` analogue, and the one most likely to be missed.
THE BLOCKER AND ITS RESOLUTION: a first attempt (reverted in 8.1a rather than shipped half-built)
died on the C4.5d guard "move through a non-field projection of a drop-tracked local" — consuming
a payload out of a drop-tracked local via `VariantField` is refused outright. `lower_match` had
already solved exactly this: it materializes the scrutinee into a fresh temp, whose initial move
clears the SOURCE local's drop flags, and a temp is never auto-dropped, so ownership transfers
exactly once with no double-drop possible. Reusing that discipline — rather than inventing a
second one for `unwrap_or` — is what turned this from a redesign into a few lines, and it keeps
one drop-elaboration story in the lowering instead of two.
SCOPE DISCIPLINE: the temp materialization and the default temp are introduced ONLY when a
droppable type is actually involved; the non-droppable path lowers byte-for-byte as before, so
no existing golden or corpus expectation moved.
FILES: starkc/src/mir/lower.rs, starkc/tests/mir_differential.rs (+3),
starkc/tests/mir_lowering.rs (the now-stale `unwrap_or` Unsupported fixture REMOVED — a residual
fixture that no longer describes a residual is worse than none), COMPILER-STATE.md, WP-C4.7.md.
RULES: EXEC-ONCE-001 / DROP-ORDER-001. No spec change; no MIR shape or runtime-surface change —
this is lowering only, using `Drop` terminators that already exist.
DECISIONS: none at CE level.
EVIDENCE: `droppable_unwrap_or_drop_timing_agrees` (both `Some` and `None` paths, with the
printed ORDER as the assertion — `100 2 200 1 300` then `400 3 500`, so the default's destruction
at the call is what is being checked, not merely that it happens);
`droppable_result_unwrap_or_drops_the_error_payload_agrees` (both type arguments carry
destructors, so neither can hide; pins `9` dropping at the call and reverse-order scope exit);
`droppable_unwrap_or_with_runtime_type_agrees` (`String` payload — the runtime-type drop path
rather than a user `Drop` impl). Workspace 785 passed / 0 failed / 2 ignored (+3); fmt clean;
clippy clean on 1.93 and 1.97.
FOLLOW-UP: none for 8.1.
NEXT: C4.7-8.2 — droppable `Iterator` Item (per-iteration scope around the loop-variable binding;
oracle-pin first), then 8.3.

### WP-C4.7-8.2 — droppable `Iterator` Item (per-iteration drop scope) — 2026-07-20
DONE: a user `Iterator` whose `Item` needs dropping now lowers.
PINNED FIRST (§0.6), and it is the non-obvious part: each yielded value is destroyed at the
**end of its own iteration**, not accumulated and destroyed at loop exit. A three-element loop
over a printing-destructor `Item` observes `body, value, DROP, body, value, DROP, …`. `break`
destroys the current iteration's value before leaving; `continue` destroys it before looping back.
All four shapes were confirmed against the oracle before the lowering existed.
IMPLEMENTATION: a per-iteration scope around the loop-variable binding — `scopes.push`, register
the binding as droppable with flags FALSE then set true (the binding is initialized by the move
out of the `Option`, and the flag must not be live before that point), lower the body, then
`emit_scope_drops_from` at the latch and pop.
THE ORDERING DECISION THAT DID THE WORK: the loop's `scope_depth` is captured **before** the
per-iteration scope is pushed. `break`/`continue` already drop every scope from `scope_depth`
onward, so both early-exit paths destroy the current iteration's value with **no special casing
at all** — the existing machinery covers them. Pushing the scope before capturing the depth would
have left the value alive on `break`, which is exactly the kind of leak that only shows up in a
test that bothers to break out of the loop. Both early-exit paths are pinned by a test for that
reason.
SCOPE DISCIPLINE: the scope is pushed unconditionally (harmless and keeps one code path) but the
binding is only registered when the `Item` actually needs dropping, so non-droppable iteration
lowers as before.
FILES: starkc/src/mir/lower.rs (`lower_for_over_user_iter`), starkc/tests/mir_differential.rs
(+2 tests, 3 programs), starkc/tests/mir_lowering.rs (stale Unsupported fixture removed),
COMPILER-STATE.md, WP-C4.7.md.
RULES: EXEC-ONCE-001 / DROP-ORDER-001 / EXEC-FOR-001. No spec, MIR-shape, or runtime-surface
change — lowering only.
DECISIONS: none at CE level.
EVIDENCE: `droppable_iterator_item_drop_timing_agrees` (printed ORDER is the assertion, so what
is checked is per-iteration destruction rather than merely that destruction happens) and
`droppable_iterator_item_break_and_continue_agree` (both early-exit paths, which is where a
per-iteration scope is easiest to get wrong). The pre-existing `String`-Item probe also agrees.
FOLLOW-UP: none.
NEXT: C4.7-8.3 — droppable scrutinee + nested patterns, the last MIR residual and the hardest
piece in the plan.

### WP-C4.7-8.3a — DEV-079 + DEV-080: two hidden defects in the flat match path — 2026-07-20
DONE: both CLOSED. Neither was in the plan. Both were found by pinning oracle drop behaviour
before writing 8.3's lowering — the §0.6 discipline paying for itself — and both sit in the FLAT
enum-match path that WP-C4.6 A2 ("general pattern engine") and C4.5d (match-drop elaboration) had
already signed off.
**DEV-079 — the verifier rejected valid MIR.** V-MOVE-1 keyed moved places as `(local, pure-Field
path)` and collapsed ANY non-`Field` projection to the whole local. `VariantField` is such a
projection, so moving two different payload fields out of one enum local looked like two moves of
the same whole place, and the second was reported `MIR-0007 move from possibly-moved place _N[]`.
Consequence: **every enum variant with two or more droppable payload fields** — with or without a
wildcard, user-`Drop` or `String` — produced MIR that **lowering accepted and verification
rejected**. That is worse than a clean `Unsupported`: the two components are supposed to be
independent readings of the same contract, and here they disagreed silently until someone wrote
the program. Fix: `moved_key` gives `VariantField(v, i)` two path components (variant, then
field), making siblings distinguishable. No collision with struct `Field` paths is possible — a
local has exactly one type, so its projections are either struct/tuple fields or variant fields.
`Deref`/`Index` still collapse to the whole local: conservative and correct, since neither denotes
a statically-known disjoint sub-place.
**DEV-080 — the drop order the verifier bug had been hiding.** With the verifier fixed, such
programs ran for the first time and immediately disagreed with the oracle. For a payload mixing
bound and wildcard fields, MIR destroyed leaves in plain reverse-FIELD order; the oracle destroys
**all bound bindings first, in reverse binding order, then the discarded leaves**. Fix:
`consume_variant_payload` consumes unbound fields FIRST and bound fields second — arm-end drops
run in reverse registration order, so registering the discarded leaves first makes the bindings
drop first and the discards after, which is the oracle's order.
WHY THIS PAIR IS WORTH NOTING: the second defect was strictly unobservable while the first
existed, because no such program could verify. A conservative rejection is not a safe place to
stop — it can hide a real semantic divergence behind itself indefinitely, and the corpus will
look green the whole time.
FILES: starkc/src/mir/verify.rs (`moved_key` + the honest-limitations note),
starkc/src/mir/lower.rs (`consume_variant_payload`), starkc/tests/mir_differential.rs (+2 tests,
4 programs), KNOWN-DEVIATIONS.md (DEV-079/080; count 76 → 78), COMPILER-STATE.md, WP-C4.7.md.
RULES: V-MOVE-1 (refined, not weakened); DROP-ORDER-001 / PAT-DROP-001. No spec, MIR-shape, or
runtime-surface change.
DECISIONS: none at CE level.
EVIDENCE: `enum_variant_with_two_droppable_fields_agrees` (user-`Drop` and `String` payload
forms) and `variant_payload_drop_order_with_wildcards_agrees`. The three-field `(a, _, c)` case is
the discriminating one: its expected order — `c`, `a`, then the wildcard — matches neither plain
reverse-field order nor declaration order, so it pins the actual rule instead of a coincidence.
Workspace 789 passed / 0 failed / 2 ignored (+2); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none for these two.
NEXT: C4.7-8.3b — the original 8.3 target, a droppable scrutinee under NESTED patterns
(`Some((s, n))`), still a clean `Unsupported` ("A2 residual").

### WP-C4.7-8.3b — droppable scrutinee under nested patterns (+ DEV-081) — 2026-07-20
DONE: the last recorded MIR residual of the WP-C4.6 Class-A campaign is closed.
IMPLEMENTED: `consume_unbound_leaves` — a recursive walk that moves every droppable sub-place the
pattern does NOT bind into an arm-scoped registered temp. A consuming match decomposes the
scrutinee completely, so whatever the pattern discards still owes a destructor: wildcards,
unmentioned struct fields, and nested tuple/variant sub-places all covered. Bindings themselves
now register as droppable in the general engine, matching what the flat path's `bind_field_local`
already did.
ORDER: the unbound walk runs BEFORE the binding walk. Arm-end drops run in reverse registration
order, so registering the discarded leaves first makes the bindings drop first — in reverse
binding order — and the discards after them, which is what the oracle does (the rule established
by DEV-080). The three-element `Some((a, _, c))` case is the discriminating evidence: expected
order `c`, `a`, wildcard, which matches neither plain reverse-field order nor declaration order.
**DEV-081 — a third pre-existing defect, found here.** `bind_shorthand` (the lowering for
`P { a, b }` rather than `P { a: a, b: b }`) moved the field value into the binding local but
**never registered that local as droppable, in any mode**. The value left the scrutinee and
nothing destroyed it. This is a **leak, not a double drop**, which is exactly why it survived: no
verifier rule is violated, no assertion trips, and a program whose destructor does not print looks
correct. It affected the FLAT path as well — `enum E { V { a: Tag, b: Tag } }` matched by
`E::V { a, b }` leaked before 8.3b existed — so it is genuinely pre-existing rather than exposed
by the new code. The named and shorthand binding paths differed in exactly this one respect, which
is what made it easy to miss.
THREE DEFECTS IN ONE INCREMENT, all in already-signed-off code (DEV-079/080 in 8.3a, DEV-081
here), all found by pinning oracle behaviour before writing lowering. Two of the three were
invisible to the existing corpus: one because a conservative verifier rejection hid it, one
because a leak has no loud failure mode.
RESIDUALS NOW: the clean-`Unsupported` list is down to `HashMap::values` (std-full, explicitly
reserved by CD-033 — not an exit blocker) and mutable slice views (WP-C4.7-8.6, an owner
decision). Every other Class-A residual recorded by WP-C4.6 is closed.
FILES: starkc/src/mir/lower.rs (`consume_unbound_leaves`, `bind_pattern` binding registration,
`bind_shorthand`, guard removed), starkc/tests/mir_differential.rs (+3 tests, 8 programs),
starkc/tests/mir_lowering.rs (last stale residual fixture removed),
KNOWN-DEVIATIONS.md (DEV-081; count 78 → 79), COMPILER-STATE.md, WP-C4.7.md.
RULES: PAT-DROP-001 / DROP-ORDER-001 / EXEC-ONCE-001. No spec, MIR-shape, or runtime-surface
change — lowering only.
DECISIONS: none at CE level.
EVIDENCE: `droppable_nested_pattern_drop_order_agrees` (four shapes incl. the discriminating
three-field case and a whole-payload wildcard), `droppable_nested_pattern_depth_and_mixed_payloads_agree`
(two-level nesting; `String`+user-`Drop` mixed payload), `struct_shorthand_bindings_drop_agrees`
(both the struct-nominal and struct-shaped-enum-variant forms). Workspace 792 passed / 0 failed /
2 ignored (+3); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none.
NEXT: **C4.7-9** — re-run the unsupported-site sweep over all `unsupported(` sites, re-verify the
frozen corpus, classify 8.4/8.5, and write the exit report for the owner's decision.

### WP-C4.7-8.6 — exclusive slice views (surface 0.1-A8) + DEV-082 — 2026-07-20
DONE, under the owner's decision to implement 8.6/8.5/8.4 before auditing rather than defer any
of them. The evidence for that decision, recorded because it settles a question the plan had left
open: **REF-SLICE-001** states outright that "writes through an exclusive slice reference update
the original object", 03-Type-System §107 gives `&mut expr[r]` the type `&mut [T]`, and §547 lists
`&mut [T; N] -> &mut [T]` among the permitted coercions. Mutable slice views are therefore
normative Core, and rev. 10's deferral of them would have exited C4 with a gap in a rule the
abstract machine states directly.
IMPLEMENTED: `RuntimeFn::SliceNewMut` (A1 amendment rev. 12, surface `0.1-A7` → `0.1-A8`),
`&mut [T]` destination, exclusive receiver borrow. The shared and exclusive constructors compute
the SAME window and share one interpreter arm — they differ only in the reference they yield, and
write permission is a static property the verifier enforces rather than something the runtime
value carries.
WRITE-THROUGH: the interpreter's WRITE path now composes a `Slice { start, len }` window with a
following `Index(i)` into the absolute element `start + i` — precisely the composition its READ
path already performed. That composition IS the write-through semantics; without it a write
through a view could not reach the base. A bare window with no following index is not a writable
place (it denotes the sub-view as a value) and is rejected loudly.
**DEV-082, found here and closed.** `borrowck.rs`'s `method_receiver` had no arm for slice or
array receivers, so a method call on one returned `None` and the caller's fallback CONSUMED the
receiver. For `&[T]` that is harmless — shared references are `Copy`, so the "move" is a copy —
which is exactly why shared slices shipped in A4-2e without anyone noticing. For `&mut [T]` it is
a real move, so `let s = &mut a[1..4]; s.len(); s[0]` failed E0100. The defect was **structurally
invisible until exclusive views existed to expose it**: no program could hold a non-`Copy` slice
reference before today. MIR had the same shape — lowering passed the receiver by MOVE — and now
reads it by `Copy`, the MIR-level equivalent of a shared reborrow, since `len`/`is_empty` only
read.
FILES: starkc/src/mir/{mod,lower,verify,interp}.rs, starkc/src/borrowck.rs (DEV-082),
starkc/tests/{mir_differential,mir_lowering}.rs (incl. both surface-string goldens and the last
`mutslice` Unsupported fixture removed), mir-amendment-A1-strings-runtime.md (rev. 12),
KNOWN-DEVIATIONS.md (DEV-082; count 79 → 80), COMPILER-STATE.md, WP-C4.7.md.
RULES: REF-SLICE-001 (write-through), 03 §107/§547. No spec text changed — the spec already
required this. MIR shape unchanged; runtime surface bumped by the pre-authorized dated-enumeration
mechanism.
DECISIONS: implements the owner's 8.6 decision; no new CE-level decision taken.
EVIDENCE: `mutable_slice_views_agree` — write-through observed at the BASE object (array and
`Vec`, the latter at a non-zero view-relative index), a view passed to a function that mutates it
through the parameter, and repeated use of a `&mut [T]` local (the DEV-082 case).
FOLLOW-UP: none.
NEXT: WP-C4.7-8.5 — non-bare impl heads (`impl<T> Wrap for Holder<Vec<T>>`), front-end-first per
C4.7-2's finding, then 8.4 (method-own generics), then C4.7-9.

### WP-C4.7-8.5 — non-bare impl heads — 2026-07-20
DONE. `02:117` (`Impl ::= 'impl' GenericParams? Type …`) admits any `Type` as an impl self type,
so a non-bare head is normative Core; C4.7-2 had already found this front-end-blocked rather than
a MIR gap.
ROOT CAUSE: `match_impl_type` bound an impl parameter only when it stood ALONE as a type argument
and otherwise fell back to `types_equal`. So `Option<T>` versus `Option<Int32>` compared unequal
and the impl was invisible to method resolution — E0302 "method not found for type
`Holder<Option<Int32>>`".
FIX: `unify_impl_ty`, one-way structural unification over nominals, `Core` containers, tuples,
references, arrays and slices. One-way matters: parameters bind from the IMPLEMENTATION side only.
A `Ty::Param` on the RECEIVER side is an ordinary type to match against, never a hole to fill —
otherwise an impl for a concrete type would spuriously match a generic receiver. A parameter that
recurs (`Pair<T, T>`) must see the same type at each occurrence, so bindings are checked for
consistency rather than overwritten.
BOTH ENGINES, DELIBERATELY: lowering's `impl_generic_subst` had the same bare-parameter
restriction and gained the matching `bind_written_impl_arg`. The checker decides WHICH impls
apply; lowering recovers the substitution that decision implies. Had only the checker been
generalized, the front end would have admitted programs that lowering then refused — exactly the
DEV-079 failure shape, where lowering and verification disagreed about the same contract.
DEV-083 RECORDED, NOT FIXED: a CONCRETE position in an impl head cannot match a receiver argument
that is still an unresolved inference variable at resolution time (`impl<T> Pair<Option<T>, Int32>`
against `Pair<Option<_infer>, _infer>`). Fixing it requires committing inference variables during
candidate search, which can select the wrong impl — a semantics change needing its own design and
evidence under TYPE-METHOD-001, not a bug fix to fold into this increment. It is a narrow
over-rejection (needs a generic impl AND a concrete head position AND an unresolved receiver
argument), both engines reject identically, and annotating the receiver is a working workaround.
FILES: starkc/src/typecheck.rs (`unify_impl_ty`), starkc/src/mir/lower.rs
(`bind_written_impl_arg`), starkc/tests/mir_differential.rs (+1 test, 3 programs),
KNOWN-DEVIATIONS.md (DEV-083; count 80 → 81), COMPILER-STATE.md, WP-C4.7.md.
RULES: 02:117 (impl grammar), TYPE-METHOD-001. No spec, MIR-shape, or runtime-surface change.
DECISIONS: none at CE level.
EVIDENCE: `non_bare_impl_heads_agree` — a trait impl and an inherent impl on `Holder<Option<T>>`,
the latter at TWO instantiations so monomorphised dispatch through a non-bare head is exercised
rather than merely the checker's acceptance, plus a concrete head position with a known receiver
type. Workspace 794 passed / 0 failed / 2 ignored (+1); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: DEV-083.
NEXT: WP-C4.7-8.4 — method-own generic parameters, the last implementation item before the audit.

### WP-C4.7-8.4 — method-own generic parameters — 2026-07-20
DONE. This completes every implementation item in WP-C4.7; only the audit and exit report remain.
NORMATIVE BASIS: `02:64` puts `GenericParams?` on every `FunctionSig` and `02:120` makes an impl
item a `Function`, so a method may declare its own generic parameters. C4.7-2 had found this
front-end-blocked (E0001 "expected 'U', found …") rather than a MIR gap, which is why it moved out
of the MIR column and needed both engines fixed.
TWO HALVES:
- **Checker.** The selected candidate's substitution map carried only the IMPL's parameters, so a
  method's own `U` stayed a rigid `Ty::Param` and no argument could unify against it. It now gets
  a fresh inference variable per call site (or the turbofish types when given) — exactly what the
  ASSOCIATED-FUNCTION path already did. Only the method path lacked it.
- **MIR.** `FnKey::ImplFn` gains `method_args` beside the impl's `type_args`; `lower_body` binds
  the method's parameters from it, `key_symbol` renders it in a second bracket, and the call site
  fills it from a new per-call-site record keyed by the call expression — the method equivalent of
  C4.5c's `generic_insts` for top-level generic fns. Impl-level and method-level substitutions
  stay SEPARATE, because a method on a generic nominal is generic in both and conflating them
  would monomorphise at the wrong arguments.
CE3 QUESTION THE PLAN ASKED ME TO SETTLE: **`FnKey` appears zero times in `mir.md`.** It is purely
lowering-internal, so extending it is not a contract change and needs no CE3. The rendered
`Instance.symbol` does change for generic methods, but §2 states symbols are "deterministic and
injective for identical inputs; NOT a stable external ABI", and a method with no own generics
renders exactly as before, so no existing symbol moved.
FILES: starkc/src/typecheck.rs (method-level instantiation + per-call-site recording),
starkc/src/mir/lower.rs (`FnKey::ImplFn::method_args`, symbol rendering, body substitution, call
site), starkc/tests/mir_differential.rs (+1 test, 3 programs), COMPILER-STATE.md, WP-C4.7.md.
RULES: 02:64/02:120 (grammar), TYPE-GENERIC-001. No spec, MIR-shape, or runtime-surface change.
DECISIONS: none at CE level — see the `FnKey` conclusion above.
EVIDENCE: `method_own_generics_agree` — two instantiations at different primitives; two
method-own parameters in one signature with a droppable (`String`) instantiation; and a GENERIC
METHOD ON A GENERIC NOMINAL at two different `U`s plus a second nominal instantiation, which is
the case that would fail if the two substitution levels were conflated. Every case uses multiple
instantiations, so what is exercised is one lowered body per instantiation rather than the
checker's acceptance alone. Workspace 795 passed / 0 failed / 2 ignored (+1); fmt clean; clippy
clean on 1.93 and 1.97.
FOLLOW-UP: none.
NEXT: **C4.7-9** — re-run the unsupported-site sweep over every `unsupported(` site, re-verify the
frozen corpus, and write the exit report for the owner's decision.

### WP-C4.7-9 (audit sweep) — six further findings; four fixed, two recorded — 2026-07-20
DONE: the sweep. Every `unsupported(` site in `lower.rs` enumerated, partitioned
defensive-vs-construct, and each construct candidate probed with `c46_probe` AND `oracle_run`.
The forecast that the audit would find more was correct.
FIXED UNDER THE OWNER'S DIRECTION ("fix 1, 2, 4 and the checker rejection for 3"):
- **DEV-084 — `print`/`println` accepted any type.** They typed their argument as a fresh
  inference variable, so a `Display`-less user struct was accepted. Three engines gave three
  answers for a program 06 says is invalid: the checker accepted it, the oracle rendered an
  unspecified `{x: 1}`, MIR refused. The CHECKER was the wrong one, and the fix is a rejection,
  not an implementation — deferred to the same pass as the bound checks so an argument still
  under inference is not judged early. One interpreter test depended on the over-acceptance and
  now asserts the rejection; its real subject (`Float32` digits nested in an aggregate) was
  already covered by its `Option`/`Result` and tuple siblings.
- **DEV-085 — `for` over a fixed-length array.** Checker accepted, oracle ran, MIR alone refused:
  an internal inconsistency, not a language boundary. Lowered as a counting loop reading one
  element per iteration through the ordinary `CheckIndex` proof discipline. **Its own
  implementation had a bug the test caught:** `continue` first targeted the loop header directly,
  skipping the increment and spinning until the interpreter's fuel ran out. The continue target
  is now a latch that increments first — and the control-flow test that exposed it was written
  before the fix, not retrofitted after.
- **Trait-default methods with own generic parameters.** WP-C4.7-8.4 fixed the selected-impl path
  and left this one: the checker's default-fallback did not instantiate the method's own
  parameters, and `FnKey::TraitDefault` had no `method_args`. Both now match the `ImplFn`
  treatment.
RECORDED, NOT FIXED:
- **DEV-086 — droppable elements in array patterns, and by-value array iteration.** An array
  element place needs `Projection::Index(ProofLocal)`, and the only way to mint a proof is a
  `CheckIndex` that READS the array. Moving one element out poisons the whole local for V-MOVE-1
  (`Index` must collapse to the whole local — a dynamic proof names no statically-known
  sub-place), so the next element's check reads a possibly-moved place. The fix is a
  **constant-index projection form**, a MIR shape change requiring CE3 (§0.5), so it is recorded
  rather than invented. The contract already points that way — §6 says the proof discipline
  "covers fixed-length `Array` (verifier may validate against the compile-time length)" — but it
  is the owner's call. Non-droppable array patterns and `Copy`-element iteration are unaffected.
- **DEV-083** (from 8.5) remains open on the same footing.
CORRECTLY RESERVED, not blockers: `HashMap::values`, `Vec::contains`, `String::insert` — std-full,
explicitly reserved by CD-033. Or-patterns (`A(n) | B(n)`) are **not in 02's Pattern grammar**
(`02:284-291`), so the parse error is correct behaviour, not a gap.
FILES: starkc/src/typecheck.rs (Display check + trait-default method generics),
starkc/src/mir/lower.rs (`lower_for_over_array`, `FnKey::TraitDefault::method_args`, array-pattern
residual), starkc/src/interp.rs (the repurposed test + a `type_diagnostics` helper),
starkc/tests/{mir_differential,gate2_valid}.rs, KNOWN-DEVIATIONS.md (DEV-084/085 closed, DEV-086
opened; count 81 → 84), COMPILER-STATE.md, WP-C4.7.md.
RULES: 06 (`Display` is not a syntax hook), EXEC-FOR-001, 02:284-291 (Pattern grammar),
02:64/02:120 (generic method signatures). No spec text changed.
DECISIONS: none at CE level; DEV-086 is flagged AS a CE3 question rather than resolved.
EVIDENCE: `for_over_array_agrees` (values, running total, `break`/`continue`, single-element
array), `trait_default_method_own_generics_agree` (two instantiations),
`printing_requires_display` (rejection plus the standard displayable types still printing),
`printing_a_struct_without_a_display_impl_is_rejected`. Frozen corpus green. Workspace 798 passed
/ 0 failed / 2 ignored (+4); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: DEV-083, DEV-086 — both over-rejections, both consistent across engines, both needing
an owner decision rather than more implementation.
NEXT: write the C4.7-9 exit report as a new final section of `WP-C4.6.md` and present it. The gate
decision is the owner's; this session does not close it.

### WP-C4.7-9 — the Gate C4 exit report — 2026-07-20
DONE: the report is written as the final section of `WP-C4.6.md`, superseding that document's
2026-07-19 Verdict. Presented to the owner; **the gate is not closed by this session**.
VERDICT AS WRITTEN: conditions 1 (corpus equivalence) and 3 (nothing carried silently) are
SATISFIED outright. Condition 2 (every normative Core construct lowers) is satisfied EXCEPT for
DEV-086 and DEV-083 — both over-rejections, both consistent across engines, neither closable by
more implementation of the same kind: one needs a CE3 constant-index projection form, the other a
method-resolution design decision under TYPE-METHOD-001.
RECOMMENDATION: close C4 **conditional on the owner disposing of those two by explicit dated
decision** (implement in C5.x, or defer with the deferral recorded) rather than leaving them
undisposed. Recording them WITH a disposition is what makes carrying them forward honest rather
than silent — which is exactly what CD-033's condition 3 asks for.
THE COUNTER-ARGUMENT, STATED IN THE REPORT RATHER THAN OMITTED: today's sweep found six items
after four increments had already "finished" the residual list, and 11 of this package's 13
defects were in signed-off code. The defect-discovery rate has **not visibly plateaued**. Two
things argue against another round now — the sweep was systematic rather than opportunistic
(every `unsupported(` site, both engines), and the two survivors are analysed and decision-blocked
rather than effort-blocked — but the risk statement belongs in front of the owner, not buried.
WHAT THE REPORT CLASSIFIES: every remaining rejection, in four buckets — spec-conformant (with the
authority cited, including the corrected "Box deref" audit error), CD-033-reserved std-full,
defensive guards (incl. the two deliberately-retained unreachable ones), and the two open
deviations. Plus the ledger state (84 numbered; 16 closed by this package; the three SOUNDNESS
defects called out separately) and the contract/spec changes (amendments A3/A4, surface
`0.1-A6` → `0.1-A8`, and the new normative `PRIM-TRAIT-001`).
FILES: STARKLANG/docs/compiler/work-packages/WP-C4.6.md (the report),
starkc/docs/conformance/KNOWN-DEVIATIONS.md (one stale line about 8.1's MIR half corrected),
COMPILER-STATE.md, WP-C4.7.md.
EVIDENCE CITED: workspace 798/0/2, 114 differential tests, frozen corpus green, fmt + clippy clean
on 1.93 and 1.97.
NEXT: **the owner's decision.** Report §6 is the decision table: DEV-086, DEV-083, post-hoc
ratification of surface revs 11/12, frozen-corpus growth, and gate closure.

### WP-C4.7 close-out — CD-038/039/040 executed; C4 NOT closed (DEV-089) — 2026-07-20
DONE: the owner's close-out directive, in full, except the closure itself.
**1. DEV-086 IMPLEMENTED (CD-038, CE3).** `Projection::ConstIndex(u64)` — statically known array
element, valid only on `Array<T, N>`, verifier bounds-checks it directly, no `CheckIndex` and no
`IndexProof`, invalid on `Vec`/slice, dynamic indexing unchanged. Consuming array patterns over
droppable elements now lower and agree with the oracle including drop order. The same decision's
**typed internal paths** were adopted: move-dataflow and drop-unit paths are typed components
(field / variant field / constant index) instead of raw `u32` sequences, and fixed-length arrays
decompose into per-element drop units — without which moving one element out and then dropping the
array would destroy it twice. Recorded in `mir.md` as amendment A5.
**NARROWED, not closed:** by-value iteration over a NON-`Copy` array element. The loop index is a
runtime counter, so no `ConstIndex` names the consumed element and V-MOVE-1 has nothing precise to
track. Reading by copy instead would be UNSOUND — the array still owns the element and destroys it
again, a double free for a `String` in a real backend — so it is refused cleanly with that reason.
Closing it needs unrolling or runtime-indexed drop flags: a separate design question, not an
extension of A5. This is recorded rather than approximated, deliberately.
**2. DEV-083 DEFERRED (CD-040b)** to `WP-C6.x Method Resolution Completion`, with the owner's
disposition text recorded verbatim in the ledger (candidate-local inference snapshots;
declaration-order-independent evaluation; no mutation of global inference state while probing).
**3. RUNTIME SURFACE RATIFIED (CD-040a):** A1 revs 11 and 12 (`0.1-A7`, `0.1-A8`). Documentation
and the active constant agree, so no implementation change was needed.
**4. CORPUS 1.2.0 (CD-039).** Completes the compact refresh to the six specified workloads: adds a
MULTI-FILE case (cross-file structs, methods, trait default + override, cross-file `Drop`,
provenance) and folds DEV-086's array pattern into the array/slice case. A bump rather than an
amendment of 1.1.0 because the array case's bytes changed. **All 48 hashes from 1.0.0 verified
byte-identical**, so the original baseline survives inside 1.2.0.
**5. GATE NOT CLOSED — DEV-089.** The bounded validation surfaced a new ENGINE DIVERGENCE, and §6
of the directive says to stop and report on exactly that. `println(p)` where `P` HAS a `Display`
impl: checker accepts, oracle runs it but prints its own debug form ignoring the user's
`Display::fmt`, MIR refuses to lower it. Not a soundness defect and not invalid MIR — nothing
mislowers — but the stopping rule's clause "no known … engine divergence remains" is not satisfied,
so closing would require asserting something untrue. It surfaced only because DEV-084 narrowed the
checker: before that, `println` accepted any type, so "has an impl" and "has no impl" were
indistinguishable.
ALSO FOUND AND PARTLY FIXED: **DEV-088** — cross-file `const` initializers were evaluated against
the entry file (the fourth per-item-file site DEV-069 missed). Declaration-time evaluation fixed;
the USE site remains open in both engines (a clean over-rejection). The multi-file corpus case was
reduced to its subject rather than chasing it, per the scope-discipline instruction.
BOUNDED VALIDATION: workspace **802 passed / 0 failed / 2 ignored**, exit 0; fmt clean; clippy
clean on 1.93 and 1.97; corpus 1.2.0 lock integrity green; `entire_frozen_corpus_agrees` green over
all 23 cases; DEV-076…086 regressions green; unsupported-site classification re-run (171 sites).
FILES: starkc/src/mir/{mod,lower,verify,interp}.rs, starkc/src/interp.rs (DEV-088),
starkc/tests/{mir_differential,mir_verify,exec_snapshots}.rs, the corpus (+3 files, 1 modified) and
its lock, STARKLANG/docs/compiler/mir.md (amendment A5), KNOWN-DEVIATIONS.md (DEV-086 closed/
narrowed, DEV-083 deferred, DEV-088/089 opened; count 85 → 87), COMPILER-STATE.md.
NEXT: **owner decision on DEV-089**, then closure. Everything else in the directive is done.
