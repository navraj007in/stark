# WP-C4.6 — Gate C4 Exit Audit (2026-07-19)

Gate: C4. Roadmap exit conditions:

1. the Core execution corpus lowered in this gate runs equivalently through HIR and MIR
   interpreters;
2. every normative Core construct required by C5 has verified MIR lowering;
3. any remaining unsupported normative construct is recorded as a gate blocker rather than
   being carried forward silently.

"If a blocker remains, C4 stays open."

## Verdict

- **Condition 1: SATISFIED.** `tests/mir_differential.rs::entire_frozen_corpus_agrees` — all
  17 frozen corpus cases agree on output, trap category, provenance, message, and pre-trap
  stdout (WP-C4.5 exit, 2026-07-19).
- **Condition 3: SATISFIED by this document.** Every unsupported normative construct is
  enumerated below with probe evidence; nothing is carried silently.
- **Condition 2: NOT satisfied.** Seven blocker classes remain (§Class A). **C4 stays open**
  until they are implemented or the owner narrows "required by C5" by explicit decision
  (§Decision required). This is the roadmap-compliant state: blockers recorded, gate open.

## Method

1. Enumerated every `unsupported(...)` site in `starkc/src/mir/lower.rs` (~105 sites).
2. Partitioned: defensive internal guards (unreachable for checker-approved programs) vs.
   construct rejections.
3. For each candidate construct, ran a minimal probe program through the full pipeline
   (`starkc/examples/c46_probe.rs`: parse → resolve → typecheck → lower → verify → MIR-run).
   A probe stopping at PARSE/RESOLVE/TYPECHECK is front-end scope (Class B/C); stopping at
   LOWER-UNSUPPORTED with a clean front end is a real C4 lowering gap (Class A).
4. Classified each gap against the normative spec (01-Lexical-Grammar operators;
   02-Syntax-Grammar patterns/expressions/impl grammar; 03-Type-System operator traits;
   06-Standard-Library `core-min` profile, STD-PROFILE-001).

## Class A — real MIR-lowering gaps (front end accepts; normative; gate blockers)

Ordered by estimated effort. "Probe" quotes the program shape that reproduces it.

### A1. Generic impls: methods, associated fns, Drop, and trait impls on generic nominals
- Probes: `impl<T> Holder<T> { fn get(&self) -> &T }` + `h.get()`;
  `Holder::make(7)`; `impl<T> Drop for Res<T>`; user `Iterator` impl driving a `for` loop.
- Rejections: "generic impl/trait method (monomorphisation of methods…)", "associated fn on a
  generic nominal type", "Drop impl on a generic nominal type", "for over a non-range,
  non-Vec iterator".
- Normative: yes — `Impl ::= 'impl' GenericParams? …` (02 §Impl); generic params on impl
  blocks (02 rule 54). `Iterator` is `core-min` prelude.
- Why it matters for C5: a "normal multi-file, multi-package Core application" uses methods
  on generic types; this is the largest expressiveness hole left.
- Shape of the fix: impl-level substitution — `FnKey::ImplFn`/`TraitDefault` instances carry
  the impl's type arguments; the C4.5c worklist/`param_subst` machinery extends to impl
  generics; `drop_impls`/`find_impl_fn` keyed by instantiation. **Estimate: 1–2 sessions**
  (largest single blocker).

### A2. General pattern lowering (tuple/array scrutinees, nested patterns, Char/Float/String literal patterns)
- Probes: `match t { (a, b) => … }`; `match o { Some(Some(x)) => … }`; `match c { 'a' => … }`;
  `match s.as_str() { "hi" => … }`; `match x { 1.5 => … }`; `match a { [x, y] => … }`.
- Rejections: "match scrutinee type", "nested pattern in match arm", "literal pattern form".
- Normative: yes — 02's `Pattern` grammar is compositional (tuple, array, nested
  `Path(PatternList)`), and pattern literals include Char/String.
- Shape of the fix: recursive pattern decomposition in `lower_match` (decision-tree or
  sequential-test lowering), Char patterns via integer switch, String patterns via `StrEq`
  chains, Float via `BinOp::Eq` tests (spec-exact equality). **Estimate: 1–2 sessions.**

### A3. User `Eq`/`Ord` operator dispatch on nominal types
- Probes: `impl Eq for P { fn eq(&self, other: &P) -> Bool }` + `a == b` →
  "comparison on a user-defined type dispatches through its Eq/Ord impl";
  `assert_eq` on user types likewise. (Structural `==` *without* an impl is already
  front-end-rejected — E0500 — so only impl dispatch is owed.)
- Normative: yes — 03 operator traits.
- Shape of the fix: lower `==`/`!=` to the impl's `eq` instance call; `<`… to `Ord` dispatch,
  which needs `Ordering` as a runtime value — the short design note A1 §10 reserved. Eq can
  land before Ord. **Estimate: 0.5–1 session** (+ the Ordering design note, CE-shaped if it
  amends the runtime surface).

### A4. `core-min` stdlib operations that typecheck but don't lower
- Probes and rejections:
  - `s.chars()` → "type Core(CharsIter, [])" — **core-min** (`String`…`chars`).
  - `v.get(0 as UInt64)` → "Vec::get (a later C4.5e sub-slice)" — **core-min**
    (`get`/`get_mut`).
  - `size_of::<Int32>()` → "builtin" — **core-min** (`size_of`, `align_of`).
  - `&a[0..2]` → "type Slice(…)" — slices, normative in 03 and in 06's behavioral
    requirements (range slicing traps); deferred since C4.5b.
  - `o.map(f)` / `o.unwrap_or(9)` → "Option/Result method … (a later C4.5e sub-slice)" —
    06 defines these in the Option/Result API; whether the full method set is `core-min` or
    only the types is ambiguous in STD-PROFILE-001 (flagged for the owner).
  - Range as a first-class value (`let r = 0..3; for i in r`) → "type Range(…)" — `core-min`
    lists `Range` "with integer `Iterator` impls (for `for` loops)"; inline `for i in 0..3`
    works, a bound range local does not.
- Shape of the fix: a dated `0.1-A4` surface enumeration (CharsIter group, `VecGetRef` for
  `get`, slice views), `size_of`/`align_of` as lowering-time constants, `MirTy::Range` as a
  small aggregate. **Estimate: 1–1.5 sessions**, includes a CE3 surface amendment.

### A5. Bitwise/shift/pow operators
- Probes: `6 & 3`, `6 | 3`, `6 ^ 3`, `1 << 3`, `16 >> 2`, `2 ** 5`, `~5`, `x &= 3`,
  `x <<= 1` — all typecheck, all rejected ("binary operator", "unary operator",
  "compound bit/pow assignment").
- Normative: yes — 01 §Bitwise/§Assignment lists all of them; 03 defines "invalid shift"
  as a trap category alongside overflow (rule at 03:965), and `**` participates in checked
  integer semantics.
- Shape of the fix: `MirBinOp` additions + `Checked` terminators for `Shl`/`Shr` (invalid
  shift trap) and `Pow` (overflow trap); compound forms reuse the existing desugar.
  **Estimate: 0.5 session.** Mechanical.

### A6. Non-Copy Vec element iteration
- Probe: `Vec<String>` + `for s in v.iter()` → "Vec iteration over a non-Copy element type
  (reserved beyond 0.1-A2)".
- Normative: iteration over `Vec<String>` is ordinary Core; the Copy bound was an interpreter
  representation compromise (snapshot iterator), not spec-derived.
- Shape of the fix: borrowed-cursor iterator (the f-3a `KeysIter` representation) for the
  Vec case, dropping the `T: Copy` restriction; V-COPY-1 relaxed for `VecIterNew`/`Next`.
  **Estimate: 0.5 session** (+ rev. 7 surface note).

### A7. Small expression forms
- Probes: `let x = loop { break 5; };` ("expression form"; `'break' Expression?` is 02
  grammar); `[7; 3]` repeat expression (02:329); `let u = if true { println(1); };`
  (Unit-typed if-as-value without else); `while`/`for` in value position (Unit-typed).
- **Estimate: 0.5 session combined.**

**Total to clear Class A: ~5–7 sessions.**

## Class B — front-end gaps blocking normative programs before MIR (recorded, different owner)

These never reach lowering; they are conformance gaps of the checker/resolver, recorded here
so the gate decision sees them (they cap what C5 programs can be written regardless of MIR):

- **DEV-067** — bounded generic params lose bounds at intra-generic call sites (E0500) and
  behind `&T` receivers (E0302). Probe: `fn call_speak<T: Speak>(t: &T) { t.speak() }` →
  E0302. Open since C4.5c.
- **DEV-069** — front end + HIR oracle not multi-file-span-clean (found f-3c). Caps
  multi-file programs — directly adjacent to C5's "multi-package application" outcome.
- **`Box<T>`** — `core-min` requires it; the checker cannot deref it (probe: `*Box::new(5)`
  → E0001 "cannot dereference non-reference type 'Box<Int32>'"). No deviation number yet;
  recorded by this audit.
- **`Ordering`/`cmp` surface** — `3.cmp(&5)` → E0304 (no method surface on primitives).
  `Ordering` is `core-min` prelude. Ties into A3's Ordering design note.
- **`Vec::contains`** — E0304; 06 lists it in the Vec API (std-full tier). Minor.
- **`Vec::get` literal typing** — `v.get(0)` fails E0001 (`UInt64` vs `Int32` literal);
  requires an explicit cast. Integer-literal inference gap against the `UInt64` index
  convention. Minor but user-facing.

## Class C — spec-conformant rejections (no action owed)

- Nested items in blocks: parse error "items are not allowed inside blocks in Core v1" —
  matches 02.
- `let`-destructuring (`let (a, b) = t`): parse error — 02's let takes a binding name, not a
  pattern.
- Move out of a value whose type implements Drop: lowering rejects — matches 03/05 ("no
  moves out of … `Drop` types"); ideally front-end-rejected, but the lowering guard is
  spec-aligned.
- User-`Drop` K/V in HashMap, `values()`/`remove()`, `HashSet`: std-full tier, explicitly
  reserved by A1 rev. 6 (§10) — reserved-not-silent is the sanctioned state; whether
  std-full is a C5 target is an owner call (§Decision).

## Class D — defensive internal guards (not constructs)

~60 of the ~105 `unsupported` sites are can't-happen guards for checker-approved programs:
`FnKey` shape mismatches, arity guards duplicating checker errors (`break` outside loop,
wrong generic-arg counts, unknown fields/variants, print/assert/drop arity), unparseable
literals (checker parsed them first), "not a call"/"not a match" dispatch preconditions,
struct/enum shape crosschecks. No spec exposure; they fail loudly per the C4.3 charter.

## Decision — RESOLVED (CD-033, owner, 2026-07-19)

**Reading (i) — full normative Core + `core-min` — adopted.** Gate C4 stays open. `core-min`
is the C5 baseline, not std-full. All Class-A classes A1–A7 are required before C4 exit
(including the `core-min` items in A4). std-full ops (`HashSet`, `HashMap::values`/`remove`,
`Vec::contains`) may stay reserved beyond C4 unless separately required by the stable Core
contract. Front-end prerequisites get explicit owners: DEV-069 blocks the C5 multi-file claim
(parallel front-end WP allowed); DEV-067, `Box` deref, and primitive `Ordering::cmp` resolved
where `core-min` requires. A3's `Ord` portion waits on a CE3 `Ordering` runtime-surface
amendment (`Eq` may proceed first); A4 needs a dated runtime-surface amendment.

**Implementation order (dependency-aware):** (1) A5+A7 mechanical; (2) A6 borrowed Vec
iteration; (3) A3 `Eq` → CE3 `Ordering` decision → `Ord`; (4) A4 `core-min` surface;
(5) A2 general patterns; (6) A1 generic impl monomorphisation. This report is updated after
each class with positive/negative/verifier/differential evidence. C4 closes only when every
required class is green.

### Class progress
- A5 (bit/shift/pow): **DONE 2026-07-19** — see below
- A7 (expr forms): **DONE 2026-07-19** — see below
- A6 (non-Copy Vec iter): **DONE 2026-07-19** — see below
- A3 (Eq + Ord): **DONE 2026-07-19** (Ord under CE3-approved Amendment A2) — see below
- A4 (core-min surface): **in progress** — A4-1 (size_of/align_of, unwrap_or) DONE 2026-07-19; A4-2 (Vec::get/get_mut, chars, slices — needs the dated runtime-surface amendment) + combinators (map/and_then/map_err) + Range-as-value pending — see below
- A2 (patterns): _pending_
- A1 (generic impls): _pending_

### A5 + A7 — DONE 2026-07-19

**A5 (bitwise / shift / power).** `MirBinOp::BitAnd`/`BitOr`/`BitXor` are pure (for same-width
two's-complement operands on the i128 carrier the result is always representable, so no range
check is owed); `~a` desugars to `a ^ mask` (mask = −1 for signed widths, `(1<<W)−1` for
unsigned width W), avoiding a type-carrying MIR unary op. `CheckedOp::Shl`/`Shr`/`Pow` are
trapping: shifts enforce the NUM-SHIFT-001 count bound (nonnegative, `< width` of the left
operand = dest type) and range-filter the result; `**` requires a nonnegative exponent
(`u32::try_from`) with each intermediate multiply checked (`checked_pow`). Compound assigns
(`&=`, `<<=`, `**=`, …) route through the same `lower_arith_operands`. New faithful trap
category **`TrapCategory::InvalidShift`** distinguishes a bad shift count (oracle message
"invalid shift count") from a non-representable left-shift result (`IntegerOverflow`); the
interpreter's checked-op path gained a `CheckedOutcome` return so a shift can override the
terminator's default category. Verifier: bitwise arm (integer-only, result-typed as operands);
shifts/pow reuse the existing checked-binary arm (the checker unifies both operands to the same
type). Evidence: `bit_shift_pow_operators_agree`, `unsigned_bitnot_is_width_masked_agree`,
`oversized_shift_count_traps_agree`, `pow_overflow_traps_agree` (4 differential).

**A7 (normative expression forms in value position).** `loop { break <value>; }` carries its
value through a new `LoopTargets.value_target` local that `break <value>` writes before the
scope-drops + jump; the exit block reads it (Unit-typed loops and `while`/`for` lower as
statements and yield `Unit`). `[value; count]` repeat replicates the once-evaluated Copy
operand `count` times (count from the array type). `then`-only `if`, `while`, and `for` in
value position lower for effects and yield `Unit`. Evidence: `loop_break_value_agree`,
`repeat_and_unit_value_forms_agree` (2 differential). Workspace 713/0; clippy clean 1.93/1.97.

### A6 — DONE 2026-07-19

Vec iteration converted from the rev. 5 *snapshot* iterator (which forced `T: Copy`) to a
**true borrowed cursor** identical to the HashMap `KeysIter`: `VecIterNew` keeps the `&Vec`
reference (`[vec-ref, cursor]`) instead of snapshotting, and `VecIterNext` indexes the *live*
Vec through it to hand out an interior `&T`, protected by the C4.5f-1 frame generations. The
`T: Copy` gate (V-COPY-1/MIR-0016) is dropped from lowering and the verifier for
`VecIterNew`/`VecIterNext`; `VecIndexGet` keeps it (it returns `T` by value). Signatures
unchanged. `Vec<String>` iteration now lowers. Amendment rev. 7 records the representation
change (no surface bump — stays `0.1-A3`). Evidence: `non_copy_vec_iteration_agrees`
(1 differential; the existing `collection_iter__01`/`__02` corpus cases stay green under the
new representation).

### A3 — Eq DONE 2026-07-19; Ord BLOCKED on CE3 (Amendment A2 drafted)

**Eq (done).** `==`/`!=` on a (non-generic) user nominal now dispatches to the type's
`Eq::eq(&self, &other) -> Bool` impl (`!=` negates), routed before eager operand lowering so
both sides are **borrowed** (`&Self`) not moved — matching the oracle's `Eq::eq` dispatch and
its borrow-not-move semantics. `find_impl_fn(nominal, "eq", …)` resolves the impl; a shared
`borrow_value_ref` helper builds the `&Self` operands (materializing a temp for a non-place
operand). Ordered comparison and comparison on a *compound* type containing a user nominal stay
clean `Unsupported`. Evidence: `user_struct_eq_dispatch_agrees` (struct with a `Drop` impl:
dispatch + borrow-not-move + correct Drop ordering all agree). **Found DEV-070** (open, owned by
A2): `match` on a scrutinee behind a shared reference (`match *self` in a `&self` method) moves
it out and poisons the borrowed place on a second read — this blocks realistic *enum* `Eq`
bodies (which match `*self`), not A3's dispatch mechanism, so the enum differential test waits on
A2.

**Ord (done, under CE3-approved Amendment A2).** The owner approved
`mir-amendment-A2-ordering.md` with five clarifications (renamed "logical MIR enum" not "runtime
value"; discriminants are logical MIR only, not physical ABI; C4-open additive-amendment
versioning policy recorded in `mir.md`; no Display requirement; DEV-070 accepted under A2).
Implemented: `EnumRef::CoreOrdering` (the prelude `Ordering` as a logical MIR enum, Less=0/
Equal=1/Greater=2) across lowering/verify/interp/deterministic dump; construction of
`Ordering::Less`/`Equal`/`Greater`; direct `cmp` calls returning `Ordering`; all four ordered
operators on non-generic user nominals lowered to `cmp` + discriminant-compare (`<`→`d==0`,
`<=`→`d!=2`, `>`→`d==2`, `>=`→`d!=0`); valid-variant checking (v3 → MIR-0008); clean
`Unsupported` for generic-nominal comparison; no change to primitive comparison. Evidence:
`user_ord_all_operators_agree`, `ordering_value_round_trips_through_match_agree`,
`user_ord_borrows_and_drops_normally_agree` (differential), `rejects_invalid_core_ordering_variant`
+ `accepts_valid_core_ordering_variants` (verifier). **Found DEV-071** (open, front-end): an
all-three-variant `Ordering` match is wrongly flagged non-exhaustive (exhaustiveness gap); the
round-trip test uses an explicit-plus-wildcard match, which fully exercises the MIR path.
DEV-070 remains open under A2 (enum `Eq`/`Ord` bodies that `match *self`).

### A4 — core-min surface (in progress)

Sub-sliced (like A1's e-1/e-2/e-3). **A4-1 DONE 2026-07-19** (no runtime-surface amendment
needed): `size_of::<T>()` / `align_of::<T>()` lower to the fixed word constant the reference
implementation reports (the HIR oracle returns 8 for every type — MIR matches exactly, type
erased, result `UInt64`); `Option::unwrap_or` / `Result::unwrap_or` select payload-or-default
via a discriminant switch, default evaluated once before the switch (non-droppable payload only
for now — a droppable payload/default needs drop-of-unused elaboration, deferred). Evidence:
`size_of_align_of_agree`, `option_result_unwrap_or_agree`.

**Remaining A4 (pending):** `map`/`and_then`/`map_err` combinators (fn-value calls, no
amendment); Range/RangeInclusive as first-class bound values; and the interior-ref/iterator/slice
surface — `Vec::get`/`get_mut` (`Option<&T>`), `chars()` iteration, array/Vec slicing — which
**needs the dated runtime-surface amendment** CD-033 pre-authorized (A4-2). `println` of the
core-min types (the Display path deferred from A2) also lands here.

### Original decision framing (retained for the record)

Exit condition 2 hinged on what "required by C5" means. Two coherent readings:

- **(i) Full normative Core + `core-min`:** all of Class A (and the `core-min` items in
  Class B) are gate blockers. C4 stays open ~5–7 sessions of Class-A work (plus front-end
  fixes owned separately). Highest-integrity reading; matches STD-PROFILE-001 ("`core-min`
  is required for every Core v1 implementation").
- **(ii) C5-representative workload:** the owner names the representative C5 application
  shape (per CD-021's workload-first practice), and only constructs that shape needs are
  blockers; the rest transfer to recorded C5-entry blockers. Faster, but the C5 app must
  then be written inside the supported subset, and every waiver must be dated here.

Also owner-owned: whether C5 claims `core-min` or std-full (drives A4/Class-B stdlib
scope), and the A3 `Ordering` design note (touches the runtime surface, CE3).

**Recommendation:** reading (i) for Class A1–A3 + A5 (they are unambiguous language-level
normative constructs — impls, patterns, operators — not stdlib tiering), with A4/A6/A7
sequenced behind the profile decision. But the choice is the owner's; this audit records
blockers either way. **[Adopted in full — see CD-033 above; reading (i) for all of A1–A7.]**

## Status

C4 **OPEN** (CD-033). All Class-A classes required; none carried silently. Work proceeds in
the dependency-aware order under the Class progress tracker above.

Probe harness: `starkc/examples/c46_probe.rs` (committed); probe sources are inlined above.
