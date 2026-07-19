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

## Decision required (CE-shaped — flagged, not resolved here)

Exit condition 2 hinges on what "required by C5" means. Two coherent readings:

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
blockers either way.

## Status

C4 **OPEN**. Blockers recorded above; none carried silently. Next work: owner disposition of
the Decision section, then Class-A increments in effort order (A5 → A7 → A6 → A3 → A4 →
A2 → A1 is smallest-first; A1 → A2 first is highest-value-first).

Probe harness: `starkc/examples/c46_probe.rs` (committed); probe sources are inlined above.
