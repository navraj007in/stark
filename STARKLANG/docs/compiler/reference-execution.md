# STARK Reference Execution Contract (Core v1)

Status: WP-C2.1 deliverable, Gate C2 ("Reference Execution Semantics and Compiler Service
Foundation"). Documentation only — no `starkc/src/*.rs` changes were made to produce this
document.

**Authority update (WP-C2.7, 2026-07-18):** this remains historical implementation/deviation
evidence. Runtime semantics are now defined solely by
`STARKLANG/docs/spec/CORE-V1-ABSTRACT-MACHINE.md`; line-numbered interpreter descriptions and
pre-C2.2 deviation states below are not normative and must not override that document.

## Purpose and method

This document states, for each topic in `COMPILER-ROADMAP.md`'s WP-C2.1 scope, what the
*correct* reference-execution behavior is for a conforming Core v1 implementation, derived from
and cited to the normative spec (`STARKLANG/docs/spec/00`–`07`). It is **not** a description of
"whatever `starkc/src/interp.rs` currently does" — where the interpreter's actual behavior was
checked against a rule and found to diverge, that is called out explicitly in a **Known
deviation** block under the rule, not folded into the rule itself. Every such deviation was
verified empirically by building the interpreter with `cargo build --manifest-path
starkc/Cargo.toml` and running a minimal `.stark` repro with `cargo run --manifest-path
starkc/Cargo.toml --bin starkc -- run <file>`; each block states the exact command output
observed.

Function/line references to `interp.rs` are to `starkc/src/interp.rs` at the working-tree state
used for this WP (post–WP-C1.7, pre–WP-C2.2; see `COMPILER-STATE.md`'s Position line, Gate C2,
next WP-C2.1). All findings below are recorded in the deviation ledger
(`starkc/docs/conformance/KNOWN-DEVIATIONS.md`) as DEV-026 through DEV-036, cited by number
throughout. **Two rounds of findings are folded into this document.** The initial drafting pass
found six deviations (DEV-026–031) and two spec-silent gaps (evaluation order, `HashMap`/
`HashSet` iteration order); three of the six were independently re-verified via a fresh empirical
repro during the drafting WP's own review pass before the document was finalized. An external
review of that finalized document then caught that the drafting pass's proposed resolutions for
both spec-silent gaps had real problems (see §2.4 and §10.3 below for what changed and why), and
independently found two further, more severe deviations the drafting pass missed entirely
(DEV-034, a confirmed double-evaluation bug; DEV-035, a confirmed dangling-reference crash on an
ordinary `&self` accessor pattern — see §2.6 and §4.3a). Both spec-silent gaps are now settled
(CD-007/CD-009/CD-010/CD-011); every deviation below is a confirmed, ledgered finding, not a
provisional one.

---

## 1. Evaluation order

**Status: settled, CD-007/CD-010.** This section originally found the spec almost entirely
silent on subexpression evaluation order and flagged it for a decision rather than asserting a
rule unilaterally. The user settled it the same day: `03-Type-System.md`'s new "Evaluation
Order (Core v1)" section (added under CD-007) adopts the interpreter's observed left-to-right
order as normative for every construct below. A same-day external review then caught that the
section's opening line ("strictly left to right") was itself in tension with the assignment
rule (right-hand side before the left-hand-side place, even though the place is written first)
— corrected in the spec to "evaluation order is defined construct by construct below," with the
assignment exception called out explicitly. The subsections below now state the **normative**
rule for each construct (cited to the corrected spec section) alongside the confirming
`interp.rs` evidence, rather than "observed, not yet normative" language.

Two sub-cases *are* spec-derivable, not silent, and are marked as such below: short-circuit
`&&`/`||`, and `if`/`match` condition-before-branch. Both follow necessarily from how those
constructs are defined, not from an arbitrary evaluator choice.

### 1.1 Short-circuit `&&` / `||` — spec-derived
`03-Type-System.md` "Operators and Traits" (line 531): "`&&`, `||`, and `!` (on `Bool`) are
built-in, short-circuiting, and not overloadable." Short-circuiting necessarily means the left
operand is evaluated first and the right operand is evaluated only conditionally on its result —
no other reading of "short-circuit" is coherent.

`interp.rs::eval_expr`, `ExprKind::Binary` (lines 729–737): for `BinOp::And`, evaluates `lhs` via
`expect_bool`, and only evaluates `rhs` if `left` is `true`; symmetrically for `BinOp::Or`,
`rhs` is evaluated only if `left` is `false`. **Matches spec.**

### 1.2 `if` / `match`: condition/scrutinee before branches — spec-derived
Necessarily true of any operational reading of `03-Type-System.md`'s "Control Flow Typing"
(lines 471–478: `if`/`else`, `match`) — you cannot select a branch without first evaluating what
selects it.

`interp.rs::eval_expr`, `ExprKind::If` (lines 816–828): evaluates `cond` via `expect_bool` before
evaluating either branch. `ExprKind::Match` (lines 829–844): evaluates `scrutinee` via
`expect_value` exactly once, then tries arms **in source (declaration) order**, testing each
arm's pattern against the already-computed scrutinee value (`match_pattern`, line 2569) until one
matches. **Matches spec** in the only way "match" can operationally be read; arm order itself is
also spec-implied (an earlier arm that would match must win, since exhaustiveness and pattern
matching in `04-Semantic-Analysis.md` §5 treat arms as an ordered sequence of guards, e.g. a
wildcard arm placed last is not treated as ambiguous with earlier specific arms).

### 1.3 Binary operator operands (non-short-circuit) — normative, CD-007
`interp.rs::eval_expr`, `ExprKind::Binary` (lines 738–742, non-`And`/`Or` case): evaluates `lhs`
via `expect_value`, then `rhs` via `expect_value` — left operand fully evaluated (including any
side effects, e.g. from a call in `lhs`) before the right operand is evaluated at all.

`03-Type-System.md` "Evaluation Order (Core v1)": "the left operand evaluates fully before the
right operand begins." **Matches spec** (this rule was adopted *from* this observed behavior
under CD-007).

### 1.4 Function call arguments — normative, CD-007
`interp.rs::eval_call` (lines 1164–1238): for every callee shape (`Res::Builtin`, `Res::Item`,
`Res::Variant`, `Res::AssociatedFn`, function-pointer value), arguments are evaluated via
`args.iter().map(|arg| self.expect_value(*arg)).collect::<Result<Vec<_>, _>>()` — Rust's
`Iterator::map`/`collect` over a slice evaluates left to right and stops at the first `Err`, so
observed behavior is strict left-to-right evaluation with a short-circuit on the first runtime
error/trap.

`03-Type-System.md` "Evaluation Order (Core v1)": "arguments evaluate left to right, before the
call itself executes." **Matches spec.**

### 1.5 Method call: receiver vs. arguments — normative for user-defined types, CONFIRMED DEVIATION for core/builtin types (DEV-033)
`interp.rs::call_method` (lines 1538–1563): for a user/nominal-type method, the receiver place is
resolved and cloned (`clone_expr_place(base)`, line 1551) and dereferenced **before** any argument
expression is evaluated (`args.iter().map(...)`, lines 1557–1560). Symmetrically,
`call_qualified_trait` (lines 1565–1593) resolves the first (receiver) argument before the rest.

`03-Type-System.md` "Evaluation Order (Core v1)": "the receiver evaluates before any argument;
arguments then evaluate left to right." **Matches spec for user-defined (nominal) types.**

**Confirmed deviation (DEV-033).** For core/builtin-type methods (`call_core_method`, line 1784
— `Vec`, `String`, `HashMap`, etc.), argument expressions are evaluated *first* (line 1792–1795),
and the receiver place is resolved lazily per-operation inside each `name == "..."` branch
afterward — the opposite order from the now-confirmed-normative rule, and inconsistent with
`call_method`'s own behavior for user-defined types. This was originally flagged in this
document's drafting as "an internal inconsistency worth noting even before any spec citation is
available"; once the receiver-first rule became normative (CD-007, refined under CD-010 during
the same-day correction pass after external review), this inconsistency became a confirmed,
numbered deviation rather than just an observation. See DEV-033 in
`starkc/docs/conformance/KNOWN-DEVIATIONS.md` for full detail.

### 1.6 Struct / tuple / array literal fields — normative, CD-007
`interp.rs::eval_struct_lit` (lines 2528–2567): fields evaluated via a `for field in fields`
loop in the order the HIR carries them, which is parse/source order — left to right, sequential.
`ExprKind::Tuple`/`ExprKind::Array` (lines 794–805): same left-to-right `.map`/`.collect` pattern
as call arguments.

`03-Type-System.md` "Evaluation Order (Core v1)": "fields/elements evaluate left to right, in the
order written." **Matches spec.**

### 1.7 Assignment: right-hand side vs. left-hand-side place — normative, CD-007 (the one non-left-to-right exception)
`interp.rs::eval_expr`, `ExprKind::Assign` (lines 744–754):
```rust
let right = self.expect_value(*rhs)?;
let place = self.expr_place(*lhs)?;
let value = if *op == AssignOp::Assign { right } else { ... };
self.write_place(&place, value, expr.span)?;
```
The **right-hand side is fully evaluated before the left-hand-side place is resolved at all** —
including any side effects the place expression's own subexpressions might have (e.g. an index
expression on the LHS: `arr[f()] = g();` evaluates `g()` before `f()`).

`03-Type-System.md` "Evaluation Order (Core v1)": "the right-hand side evaluates fully before the
left-hand-side place is resolved... side effects in a place expression's own subexpressions...
run *after* the right-hand side." **Matches spec.** This is the construct the spec's own
"Evaluation Order" section calls out as the exception to its general left-to-right framing (a
same-day external review caught that the section's original "strictly left to right" opening
line was in tension with this exact rule — corrected).

### 1.8 Index: base vs. index — normative, CD-007
`interp.rs::expr_place`, `ExprKind::Index` (lines 2784–2789): resolves the *base* place first
(`self.expr_place(*base)?`), then evaluates the index expression (`self.expect_int(*index)?`).
The value-context Index path (`eval_expr`, lines 772–780) similarly resolves/clones the base
place before consulting the index's static type to route to `slice_value` or a scalar place read.

`03-Type-System.md` "Evaluation Order (Core v1)": "the base expression resolves to a place before
the index expression evaluates." **Matches spec.**

---

## 2. Function and method dispatch

### 2.1 Free-function and associated-function calls
A path expression resolving to `Res::Item` (free function) or `Res::AssociatedFn` (inherent
associated function, e.g. `Vec::new`) calls the corresponding HIR function body directly, no
dynamic dispatch involved — Core v1 has no `dyn Trait`/trait objects (`03-Type-System.md` line
663–665: "Dynamic dispatch via `dyn Trait` is a future extension; `dyn` is a reserved keyword"),
so **all dispatch in Core v1 is static** (resolved to a single concrete callee, whether at
compile time via name resolution or, for trait methods, via the runtime nominal-type lookup
described in 2.2 below — the latter is still *static* dispatch in the sense of "exactly one
implementation exists per (type, method) pair," per Core v1's coherence rules, `03-Type-System.md`
"Trait Coherence," lines 631–637).

`interp.rs::eval_call` (lines 1164–1238), `Res::Item`/`Res::AssociatedFn` arms; `find_associated_fn`
(lines 1665–1695) — searches only `trait_: None` (inherent) impl blocks for a matching
receiverless function name.

**Known deviation (DEV-024, existing, cited in `C1-exit-report.md`):** `find_associated_fn` only
searches inherent impl blocks (`trait_: None`), never trait-provided associated functions. A
trait associated function like `From::from` (`06-Standard-Library.md` "Conversion Traits," lines
508–512) called as `Point::from(x)` or `From::from(x)` cannot resolve through this path. Status:
open, unscheduled, needs root-cause investigation (per `C1-exit-report.md`'s deviation table).
Not re-investigated in this WP; cited as-is.

### 2.2 Method-call resolution to a nominal type's `impl`
Normative algorithm: `03-Type-System.md` "Method Calls and Auto-Borrowing" (lines 486–514):
1. Candidate collection: inherent methods first, then in-scope trait methods; inherent methods
   shadow trait methods of the same name (line 493–494).
2. Ambiguity between two in-scope traits with no inherent method is a compile-time error.
3. Receiver coercion tries `self: S`, then `self: &S` (auto-borrow), then `self: &mut S`
   (auto-mutable-borrow), in that order.
4. Auto-dereference: if no candidate matches on `S` and `S` is `&T`/`&mut T`, retry with `T`,
   repeated for nested references.
5. Visibility per module rules (`07-Modules-and-Packages.md`).

`interp.rs`'s runtime counterpart:
- `call_method` (lines 1538–1563) routes core/builtin-typed receivers (`is_core_value`, lines
  1768–1782, walking through `Ty::Ref` layers at the *type* level) to `call_core_method`
  (string-name dispatch, §9 below), and nominal (struct/enum) receivers to `find_method`
  (lines 1595–1663).
- `find_method` fully dereferences the receiver value first (`clone_expr_place` +
  `deref_value`, which loops `while let Value::Ref(place) = value` — i.e. it collapses "one
  level, applied repeatedly" (spec step 4) into a single fully-dereferencing pass at the *value*
  level; behaviorally equivalent to iterated single-level deref for any receiver depth Core v1
  can construct, since Core v1 has no way to construct arbitrarily-deep nested references beyond
  what auto-borrow itself introduces).
- `find_method` then does `self.hir.items.iter().find_map(...)`: **linear scan over every item
  in the whole program, in HIR item order** (source/declaration order across the file, not
  sorted by inherent-vs-trait), looking for the first `impl` block matching the nominal type
  (and, if `trait_filter` is set for a qualified call, the specific trait) that defines the
  method name; falls back to the trait's own default-method body if the impl block doesn't
  override it (WP-C1.3/DEV-013 fix, closed).

**Confirmed deviation (DEV-026).** `find_method`'s "first impl block found, in
HIR item order" rule does not implement "inherent methods shadow trait methods" (spec rule 1
above, line 493). With a struct `Thing`, a trait `Speak` supplying a default `fn say(&self) ->
String { "trait-default" }`, `impl Speak for Thing { }` (using the default), and a separate
inherent `impl Thing { fn say(&self) -> String { "inherent" } }`, calling `t.say()` prints:
- `"trait-default"` when the `impl Speak for Thing` block textually precedes the inherent
  `impl Thing` block in the source file;
- `"inherent"` when the inherent block precedes the trait impl block.

Per spec, the inherent method must win **unconditionally**, regardless of textual order. This is
a distinct bug from DEV-008 (which was about `==`/`!=` dispatching structurally instead of to
`Eq::eq`, already closed in WP-C1.3) — this one is about *priority among multiple candidates that
do exist*, not about a missing dispatch path. Not in `KNOWN-DEVIATIONS.md` under any existing
DEV-NNN as of this WP.

### 2.3 Trait-qualified calls (`TraitName::m(&recv, args)`)
Spec: `03-Type-System.md` line 512–514 — always legal, a method is "an ordinary function whose
first parameter is the receiver." `interp.rs::call_qualified_trait` (lines 1565–1593) implements
this: splits the first argument off as the receiver, looks up the method name from the trait
item's declared signature, and calls `find_method` with `trait_filter` set to the named trait —
this correctly disambiguates when a bare `recv.m(...)` would otherwise be ambiguous (spec rule 2).

### 2.4 Operator-trait desugaring
Spec table (`03-Type-System.md` lines 516–531): `==`/`!=` → `Eq::eq` (negated for `!=`);
`<`/`<=`/`>`/`>=` → `Ord::cmp` compared against `Ordering`; arithmetic/bitwise/shift → `Num`
(primitive operation after monomorphization — `Num` is not user-implementable, `03-Type-System.md`
line 527–529, so this is never a *dispatch* question at runtime, just a direct primitive op).

`interp.rs::eval_binary` (lines 1003–1109):
- `==`/`!=` (lines 1014–1041): dereferences both operands, then if the (dereferenced) left value
  is a `Value::Struct`/`Value::Enum` (`nominal_item`), looks up a user `eq` method via
  `find_method` and calls it (DEV-008 fix, closed, WP-C1.3); otherwise falls back to structural
  `Value` equality — sound for primitives and `Ty::Core` container types since Core v1 has no
  user-overridable `Eq` for those (operator overloading for user types is a future extension,
  spec line 530).
- `<`/`<=`/`>`/`>=`: only three arms exist — `(Int, Int)`, `(Float, Float)`, `(String|Str,
  String|Str)`. **There is no struct/enum arm at all** — anything else falls through to the
  final `_ => Err("invalid binary operation")`.

**Confirmed deviation (DEV-027, independently re-verified during this WP's own
review pass — see "Findings requiring a decision"), two-part finding:**

(a) The prelude `Ordering` enum (`06-Standard-Library.md` lines 76–81, "enum Ordering { Less,
Equal, Greater }", and the `Ord` trait's required signature `fn cmp(&self, other: &Self) ->
Ordering` at line 111–113) **does not exist as a resolvable name anywhere in the compiler**.
Every occurrence of `Ordering` in `starkc/src/*.rs` is Rust's own `std::cmp::Ordering` used
internally by the interpreter's own `Value: Ord` impl; `hir.rs`'s `CoreType` enum (the registry
of implementation-provided prelude/std types — `String`, `Vec`, `Box`, `Option`, `Result`,
`Range`, `RangeInclusive`, iterator/view types, `HashMap`, `HashSet`, `Random`, `IOError`) has no
`Ordering` entry. Verified (re-confirmed independently during this WP's review): a program
declaring `impl Ord for Point { fn cmp(&self, other: &Point) -> Ordering { ... } }` and returning
`Ordering::Less`/`Ordering::Greater`/`Ordering::Equal` fails to compile with `[E0202] undefined
type 'Ordering'` and further `[E0200] undefined variable 'Ordering::...'` errors. **A conforming
`impl Ord` per the spec's own trait signature cannot currently be written at all.**

(b) Consequently — and independently confirmable even setting (a) aside — `eval_binary` has no
`Ord`/`cmp`-dispatch arm for struct/enum values, unlike the `Eq`/`eq` fix from DEV-008.
`typecheck.rs::ty_satisfies_operator_bound` (lines 5841–5852) *does* accept `Ty::Struct`/
`Ty::Enum` for the `"Ord"` bound whenever a matching `impl Ord for T` exists in the HIR (i.e. the
type-checker's static check is structurally ready for this, independent of whether `Ordering`
resolves) — so if (a) were fixed, a struct/enum `<` comparison would type-check successfully and
then crash at runtime with `"invalid binary operation"` on reaching `eval_binary`, the same class
of gap `<`/`>` had for `==`/`!=` before DEV-008's fix. This is a compile-time/runtime **mismatch**
of exactly the kind Gate C2 exists to surface: typecheck.rs is (partially) ready for a feature
that has no working runtime implementation, and no working way to even author the required trait
method body today. Recorded as DEV-027.

### 2.5 Default trait-method fallback
Spec: `03-Type-System.md` implies default bodies are usable by any implementer (trait method with
a body, `fn m(&self) { ... }` inside a `trait` block, distinct from Core v1's ban on body-less
free functions outside traits — `06-Standard-Library.md` line 10–11). `find_method` (lines
1633–1661, WP-C1.3/DEV-013 fix, closed): if the impl block doesn't override the method, falls
back to the trait's own default-method body (`TraitItem::Method { body: Some(body), .. }`).
**Matches spec** (modulo the priority-ordering deviation in §2.2 above, which affects which impl
block's default/override is reached first, not whether defaults work at all).

### 2.6 By-value receiver expressions are evaluated twice — CONFIRMED DEVIATION (DEV-034), found by external review

**§1's evaluation-order rules implicitly assume each subexpression evaluates exactly once** — an
ordering rule presupposes a single evaluation to order against others. `call_method` (§2.2, line
1551) evaluates the receiver expression once via `clone_expr_place` purely to determine dispatch
(which method/impl to call) — for a non-place receiver expression (e.g. a function call), this
stores the one-time result in a synthetic temporary place (§3). But `call_user_method`'s
`hir::Receiver::Value` arm (line 1710, reached when the called method takes `self` by value, not
`&self`/`&mut self`) calls `self.expect_value(base)?` — **re-evaluating the original receiver
expression from scratch**, completely independent of the dispatch-time evaluation, rather than
reusing the already-computed `borrowed_receiver` value `call_method` passes in.

Confirmed empirically:
```stark
struct Counter { n: Int32 }
impl Counter { fn consume(self) -> Int32 { self.n } }
fn make_counter() -> Counter { println("making"); Counter { n: 1 } }
fn main() -> Unit { let r = make_counter().consume(); println(r); }
```
prints `making` **twice** for one call to `make_counter()` — any observable side effect in a
by-value method's receiver expression is silently duplicated. This is not a rare shape:
`expr.consume_style_method()` where `expr` is itself a call or computed expression is ordinary
method-chaining. See DEV-034 in `starkc/docs/conformance/KNOWN-DEVIATIONS.md` for full detail and
the proposed fix (reuse `borrowed_receiver` instead of re-evaluating `base`).

---

## 3. Place evaluation

A *place* denotes an addressable location: a local, a struct/tuple field projection, an array/Vec
index projection, or a dereference. `03-Type-System.md`'s Array Types section (lines 92–98)
normatively frames `expr[i]` and `expr[r]` as denoting *places* (bounds-checked, trap on
violation); `05-Memory-Model.md`'s "Reference Layout" (lines 45–54) frames `&expr` as producing a
value that "contains the address of" the place.

`interp.rs`'s runtime representation of a place: `struct Place { frame: usize, local: LocalId,
projections: Vec<Projection> }` (lines 360–365), where `Projection` is `Field(String)`,
`Index(usize)`, or `MapKey(Value)` (lines 353–358). `expr_place` (lines 2757–2812) builds a
`Place` recursively:
- `Path` to a local/`self` → base place, no projections (`frame: self.frames.len() - 1, local,
  projections: vec![]`) — i.e. always the *current* (topmost) call frame; STARK has no way to
  name an outer frame's local directly, so this is sound as long as references never outlive
  their frame (enforced at compile time — see §11).
- `Field { base, name }` → recurse on `base`, push `Projection::Field(name)`.
- `TupleField { base, index }` → recurse, push `Projection::Index(index)`.
- `Index { base, index }` → recurse on `base`, evaluate `index` as an int (`expect_int`), push
  `Projection::Index(...)`. **Only supports a scalar integer index at the place level** — see
  Known Deviation below for the range-index (`expr[range]` place) case.
- `Unary { op: Deref, operand }` → evaluate `operand` to a `Value::Ref(place)` and return that
  place directly (no new projection: dereferencing an existing reference just yields its
  underlying place).
- Any other expression shape → falls to a temporary: the expression is evaluated to a value
  (`expect_value`), stashed in a synthetic out-of-range `LocalId` (`1_000_000 +` frame-local-
  count, line 2803) in the current frame, and a place pointing at that synthetic local is
  returned. This is how `&(a + b)` or `&f()` produce a valid (if short-lived) place for a
  temporary.

A place is *read* via `place_value`/`place_value_mut` (lines 3040–3072): walk `frames[place.frame]
.values[place.local]`, then apply each `Projection` in order via the free functions `project`/
`project_mut` (lines 3250–3286), which return `None` (→ a `RuntimeError`, i.e. a trap) on a
missing field, an out-of-bounds index, or an absent map key. This directly implements the
bounds-checked-place / trap-on-violation rule (`03-Type-System.md` lines 92–98,
`06-Standard-Library.md` "Behavioral Requirements," line 619: "Indexing `Vec<T>` with `[]` MUST
perform bounds checking and MUST trap on out-of-bounds access").

A place is *taken* (moved-from, for non-Copy values) via `take_place` (lines 3074–3082): clone the
value, and if it is not a Copy type (`value_is_copy`, §4 below), `.take()` the slot, replacing it
with `None`, and error with `"use of moved value"` if the slot was already `None`. A place is
*written* via `write_place` (lines 3084–3090): replace the slot's value, and if a previous value
existed, run `drop_value` on it (see §6).

**Confirmed deviation (DEV-028).** `03-Type-System.md`'s Array Types section
(lines 95–98) is explicit and normative: `expr[r]` where `r` is a `Range` denotes a place of
unsized slice type `[T]`, and specifically states `&expr[r]` has type `&[T]` and `&mut expr[r]`
has type `&mut [T]` — i.e. taking a reference to a range-indexed place is spec-mandated syntax.
But `expr_place`'s `Index` arm (line 2786) unconditionally calls `self.expect_int(*index)`,
which fails whenever `index`'s runtime value is a `Value::Range` rather than a `Value::Int`.
Verified with two minimal repros:
```stark
let arr: [Int32; 5] = [1, 2, 3, 4, 5];
let s: &[Int32] = &arr[1..4];
```
→ `Error: runtime error: expected integer` (pointing at the `1..4` span), and identically for
`let mut arr: [Int32; 5] = [...]; let s: &mut [Int32] = &mut arr[1..4];`. Both the shared and
mutable spec-mandated slice-place forms crash unconditionally. (The *value*-context sibling path,
`eval_expr`'s `Index` arm at lines 772–777, does handle a `Range`-typed index correctly via
`slice_value` — see §4's note on slice materialization — but that path is only reached when the
index expression is not being placed under `&`/`&mut`, i.e. `expr_place` is never consulted for
it; `&`/`&mut` always route through `expr_place` via `UnOp::Ref`, line 714.) Recorded as
DEV-028.

---

## 4. Moves, copies, borrows, and runtime representation

### 4.1 `Value` as the runtime representation of an owned value
`interp.rs`'s `Value` enum (lines 33–79) is the runtime representation of every STARK value:
primitives (`Bool`, `Int(i128)` — a single widened representation for all integer widths, with
range re-validated against the statically-known target type at points that matter, see §8 —
`Float(f64)`, `Char`, `Str`/`String`), aggregates (`Tuple`, `Array`, `Struct`, `Enum`), and
implementation-provided types (`Vec`, `Boxed`, `Option`, `Result`, `Range`, `HashMap`, `HashSet`,
iterator/view types, `Random`, `IOError`). This matches `06-Standard-Library.md`'s framing of
`Box`/`Vec`/`String`/`HashMap`/`HashSet` as "implementation-provided" types whose "internal
representation is not expressible in Core v1 source and is implementation-defined" (lines 13–19)
— the `Value` enum's variant shapes for these types are exactly that implementation-defined
representation for the reference interpreter.

### 4.2 Copy vs. Move at runtime
Spec: `05-Memory-Model.md` "Copy vs Move Types" (lines 313–346): a type is Copy if all its fields
are Copy (built-in Copy set: `Int8`–`Int64`, `UInt8`–`UInt64`, `Float32`, `Float64`, `Bool`,
`Char`, `Unit`, `&T` — not `&mut T` — and `[T; N]` where `T: Copy`); everything else moves on
assignment/pass/return.

`interp.rs::value_is_copy` (lines 3114–3153) implements the runtime side of this check,
structurally: primitives, `Str` (the borrowed `&str`/string-literal variant), `Ref` (any
reference — the spec's "`&T` (immutable references)" rule; the interpreter does not distinguish
`&T` from `&mut T` representation-wise at this check, both are `Value::Ref` — see §4.3 below on
why this is sound), and `Function` are unconditionally Copy; `Tuple`/`Array` are Copy iff every
present element is; `Struct`/`Enum` are Copy iff the item is in `self.copy_items` (built once at
interpreter construction from every `impl Copy for T` found in the HIR, lines 536–558); `Option`/
`Result` are Copy iff their payload is; everything else (`String`, `Vec`, `Boxed`, `Range`,
iterator/view types, `HashMap`, `HashSet`, `Random`, `IOError`) is unconditionally Move.

`take_place` (§3, lines 3074–3082) is where this matters operationally: Copy values are cloned
without touching the source slot; Move values are physically taken out of the slot (`slot.take()`
→ `None`), and any later read of that now-`None` slot traps with `"use of moved value"` — a
runtime backstop for the borrow checker's static move-tracking (`03-Type-System.md` "Copy and
Drop," lines 539–557; `04-Semantic-Analysis.md` "Move Semantics Validation," lines 90–100).

### 4.3 What `&`/`&mut` look like at runtime — a real place-path, not a clone
`05-Memory-Model.md` "Reference Layout" (lines 45–54) states references are pointers (8 bytes on
64-bit) containing "the address of" the referent. `interp.rs` cannot have literal machine
addresses (it is a tree-walker over an HIR, not a codegen backend), but its representation is the
closest faithful analog: `Value::Ref(Place)`, where `Place` is `(frame index, local id,
projection path)` (§3). Critically, **a `Value::Ref` is not a snapshot/clone of the referent** —
`eval_expr`'s `UnOp::Ref` case (line 714) constructs it directly from `expr_place(operand)`
without evaluating/copying the referent's *value* at all, and every read through it
(`place_value`/`place_value_mut`, `clone_place_value`, `deref_value`) re-walks the frame/local/
projection path **at the time of the read**, observing whatever is currently stored there. This
means a `&mut` obtained early and used later (within its lexically-scoped borrow region — see
§11) sees live mutations made through other paths to the same place in between, exactly as a real
pointer would — this is the property that makes the representation a faithful (if
implementation-specific) stand-in for "pointer," matching the spirit if not the letter of
"references are just pointers" (`05-Memory-Model.md` line 395, under "Zero-Cost Abstractions").

For a **method-call receiver** specifically, `call_method`/`call_qualified_trait` first take a
value-level *clone* of the (dereferenced) receiver purely to decide *which* method/impl to
dispatch to (`clone_expr_place` + `deref_value`, §2.2) — that clone is discarded after dispatch
selection. The value actually bound as `self` inside the method body follows the method's
declared `hir::Receiver` kind:
- `Receiver::Value` **re-evaluates the receiver expression itself**
  (`self.expect_value(base)?`) rather than reusing the dispatch-time clone — this is not a benign
  redundancy: it is DEV-034 (§2.6), a confirmed double-evaluation bug whenever the receiver
  expression is not a simple place.
- `Receiver::RefMut` round-trips through the real `Place` via `take`/later `write_place`
  (`call_user_method`, lines 1712–1719, 1752–1758), so mutations through `&mut self` are
  genuinely written back to the caller's storage.
- `Receiver::Ref` rebinds the dispatch-time snapshot clone directly, reasoned in this document's
  original drafting as "sound, since Core v1 has no interior mutability... so a `&self` method
  cannot observably distinguish a live alias from a frozen snapshot taken at call time." **That
  reasoning is correct only for what the method body itself *reads* through `self` during the
  call — it does not account for a value *returned* from the method that is derived from `self`
  and outlives the call.** `self` is stored as a value inside the method's own call frame; a
  returned `&self.field` is a `Value::Ref` pointing into that frame, and the frame is popped
  before the return value reaches the caller. See DEV-035 (a confirmed, high-severity finding)
  immediately below — the snapshot-clone design for `Receiver::Ref` is the direct cause.

### 4.3a Returned references derived from `&self` dangle after the method frame is popped — CONFIRMED DEVIATION (DEV-035), found by external review

`03-Type-System.md`'s shortest-input-lifetime rule (References and Lifetimes, requalified under
WP-C1.4's borrow checker work) makes returning a reference derived from a reference parameter —
including `&self` — an entirely ordinary, spec-legal, borrow-checker-approved pattern. A method
such as `fn value_ref(&self) -> &Int32 { &self.value }` must work; the borrow checker correctly
accepts it (the *static* analysis is sound). But at runtime, `self` lives in the method's own
call frame (pushed by `call_user_method`), so `&self.value` evaluates to a `Value::Ref` whose
`Place` points into that frame. `call_user_method` pops the frame (`cleanup_current_frame` then
`self.frames.pop()`) **before** the return value reaches the caller — the returned `Value::Ref`
now points at a frame slot that either no longer exists or has been reused by an unrelated frame.

Confirmed empirically:
```stark
struct BoxedValue { value: Int32 }
impl BoxedValue { fn value_ref(&self) -> &Int32 { &self.value } }
fn main() -> Unit {
    let b = BoxedValue { value: 42 };
    let r = b.value_ref();
    println(*r);
}
```
fails with `runtime error: dangling reference` at `println(*r)`. This affects essentially every
idiomatic accessor/getter method returning a reference into `self`, unconditionally — a
compile-accepts/runtime-always-crashes gap for a large, common, spec-legal program shape, and
arguably the single most severe finding in this document (DEV-035 is recorded as the highest
priority in the entire ledger). Root-cause description here is inferred from reading
`call_user_method`'s frame lifecycle alongside the empirical crash, not yet traced line-by-line
to a specific fix — see DEV-035 in `starkc/docs/conformance/KNOWN-DEVIATIONS.md` for the proposed
disposition and the explicit caveat that this needs confirmation before a fix is attempted.

### 4.4 Slice materialization — copies, not views
`03-Type-System.md` (line 96–98) and `05-Memory-Model.md` (lines 51–54) describe `&[T]`/`&mut [T]`
as pointer-plus-length *views* into existing storage — mutating through `&mut expr[r]` should be
observable through the original collection. `interp.rs::slice_value` (lines 2729–2755, the
value-context path reached from `eval_expr`'s `Index` arm at lines 772–777) instead **clones the
underlying elements into a brand-new `Value::Array`** (`values[start..end].to_vec()`, line 2754)
— it materializes a disconnected copy, not a live view. Combined with §3's Known Deviation (the
place-context `&expr[r]`/`&mut expr[r]` form crashing outright), there is currently **no way to
observe** whether a working mutable-slice-through-range would round-trip to the source collection
in this interpreter, since the only code path that produces a slice value at all (the
value-context path) is a copy by construction. This is recorded here as directly relevant
context for whoever repairs the §3 deviation: fixing `expr_place`'s Range-index crash alone would
not be sufficient if the fix simply reused `slice_value`'s copy semantics — the result would
compile and run, but `&mut expr[r]` mutations still would not propagate, silently diverging from
the spec's view semantics. Flagged as part of the same underlying gap, not a second independent
one.

---

## 5. Aggregate construction and destructuring

### 5.1 Construction
Struct/enum-variant literals: `eval_struct_lit` (lines 2528–2567) evaluates each field expression
(or, for shorthand `Point { x, y }` syntax, takes the value from the same-named local via
`find_local_by_name` + `take_place`, lines 2540–2550 — this is a real *move* of the local, not a
read, matching move semantics for a non-Copy shorthand field) and collects them into a
`BTreeMap<String, Option<Value>>` keyed by field name. Tuple/array literals (`ExprKind::Tuple`/
`Array`, lines 794–805) collect into a `Vec<Option<Value>>` in written order. Both aggregate
kinds' field order is captured in §1.6 above (normative, CD-007: left to right, source order).

**Representation note directly relevant to §6:** struct fields (and enum struct-like-variant
named fields) are stored in a `BTreeMap<String, Option<Value>>`, keyed and therefore *iterated*
in **alphabetical field-name order**, not declaration order. Tuple/array/tuple-variant-enum
fields are stored in a `Vec`, which *does* preserve declaration order. This asymmetry is the
direct cause of the Known Deviation in §6.2 below.

### 5.2 Destructuring (pattern matching)
`match_pattern` (lines 2569–2687) recursively matches a pattern against a value:
- `Wild` (`_`): matches unconditionally, binds nothing.
- `Binding` (a plain name, or `x @ pat`-style capture): pushes `(local, value.clone())` — i.e.
  **every binding is a clone of the matched (sub)value**, regardless of whether the source-level
  binding mode would conceptually be "by value" (moving) or "by reference" (borrowing a place).
  When the matched value itself is a `Value::Ref(place)` (e.g. matching `&x` or a reference
  produced by `.iter()`), cloning it is cheap and preserves reference semantics (the clone is
  just another path to the same place, per §4.3); when the matched value is an owned aggregate,
  cloning it is a real deep/structural duplication at the `Value` representation level. This is
  *not* the double-allocation it might look like: the pre-match scrutinee (`eval_expr`'s `Match`
  arm, line 830, `let value = self.expect_value(*scrutinee)?`) already took ownership of (moved)
  the original place if it was non-Copy — the "original" `value` that gets cloned from during
  binding is a Rust-level local inside the interpreter, not a second live STARK value; only the
  clone becomes the new, single logical owner. See §6.4, however, for what happens to portions of
  that scrutinee that a pattern does **not** bind.
- `Lit`: literal-value equality against the freshly-evaluated pattern literal (uses
  `eval_lit`/`literal::eval_lit_value`, the same shared module DEV-025's fix (WP-C1.5, closed)
  put in place for value-correct literal comparison).
- `Path` (unit variant, `None`, `IOError::NotFound`-style paths): variant/kind identity check.
- `TupleVariant` (`Some(x)`, `Ok(x)`, a tuple-shaped enum variant): destructures the payload
  Vec/boxed value and recurses via `match_sequence`.
- `Struct` (struct pattern or struct-like enum variant pattern): for each *named field the
  pattern mentions*, look up that field in the value's field map and recurse/bind; **fields the
  pattern does not mention are never touched** (see §6.4).
- `Tuple`/`Array`: positional destructure via `match_sequence`, which requires
  `patterns.len() == values.len()` (Core v1 has no `..`-rest array/tuple pattern per the grammar
  as implemented here — a completeness note, not itself investigated further in this WP).

---

## 6. Drop order and drop flags

### 6.1 Block-scoped `let` bindings — reverse declaration order (spec-conforming)
Spec: `05-Memory-Model.md` "Drop Order" (lines 283–291): three sequential `let` bindings in a
block drop in reverse declaration order (last-declared, first-dropped) at end of scope. "Drop
Soundness" (lines 300–311) additionally requires exactly-once semantics via drop-flag tracking
for partially-moved values.

`interp.rs::eval_block` (lines 640–660) accumulates each `let`-introduced `LocalId` into a
`locals: Vec<LocalId>` **in declaration order** as it walks the block's statements, then calls
`cleanup_locals(&locals)` (lines 3155–3167) at the end of the block (or immediately, if a
statement produced non-`Value` control flow — see §6.5). `cleanup_locals` iterates `locals.iter()
.rev()` — i.e. **last-declared first** — and for each still-present (`Some`) slot, takes it and
calls `drop_value` (§6.3). A slot that is `None` (already moved out, e.g. via `let n = p.name;`)
is silently skipped — this **is** the drop-flag mechanism: the `Option<Value>` slot itself
doubles as the initialization/drop flag (`Some` = live and owned here, `None` = moved-from or
never-initialized), satisfying "track initialization state... to preserve exactly-once
semantics" (`03-Type-System.md` line 549–550). Function-level cleanup (`cleanup_current_frame`,
lines 3169–3172, called at the end of `call_callable`/`call_user_method`) does the same over
`Frame.order` — the full first-insertion order of every local (receiver, params, then body
locals) — so parameters and the receiver are dropped last (in reverse of their insertion,
which happens before any body local). **This matches the spec's example directly** and is the
behavior the interpreter's own regression test `runs_drop_in_reverse_declaration_order`
(line 3567) exercises.

### 6.2 Struct-field / enum-named-field drop order — NOT declaration order
**Confirmed deviation (DEV-029, spec citation added under CD-011, independently re-verified during this WP's own
review pass — see "Findings requiring a decision").** The spec's only explicit drop-order example
(§6.1 above) is about sibling `let` bindings in a block; it does not separately spell out
struct-*internal* field drop order. The only reasonable extension of "reverse declaration order"
to struct fields — and the one every Rust-family language actually implements — is: fields drop
in reverse of their *declaration* order in the `struct`/`enum` item. `interp.rs::drop_value`
(lines 3174–3222), however, drops a `Value::Struct`'s fields via `fields.values_mut().rev()`
(line 3194) where `fields: BTreeMap<String, Option<Value>>` — i.e. **reverse-alphabetical
field-*name* order**, not reverse-declaration order (same for `Value::Enum`'s `named` map, line
3202). Verified: a struct
```stark
struct Pair { beta: Loud, alpha: Loud }   // (also tried: alpha declared first, beta second)
```
where `Loud` has a `Drop` impl that prints its label — **both declaration orders produce the same
output, `beta` then `alpha`** — conclusively showing drop order tracks field-name alphabetical
sort, not source declaration order, and is *invariant* to how the fields were actually declared.
Tuple/array/tuple-enum-variant fields (`Vec`-backed, lines 3188–3192, 3199–3201) do **not** have
this problem, since a `Vec` preserves insertion (= declaration) order; only the `BTreeMap`-backed
named-field cases are affected. Recorded as DEV-029 (spec citation added under CD-011).

### 6.3 User `Drop::drop` invocation, then recursive field drop
Spec: `05-Memory-Model.md` lines 264–281 (automatic drop calls the user's `drop` method at scope
exit) and `03-Type-System.md` lines 544–545 (`drop(value)` is the only legal way to force a
drop; `Drop::drop` itself cannot be called explicitly). `drop_value` (lines 3174–3222): if the
value is a `Struct`/`Enum` and an `impl Drop for T` exists (`find_drop`, lines 3224–3233), the
user's `drop(&mut self)` body is run first (against a *clone* of the value bound to a synthetic
frame — the clone is discarded after the call, since by the time `drop_value` runs on it, this
copy is the sole conceptual owner and there is nothing further to write back), and *then*
`drop_value` recurses into the value's children (fields/elements) regardless of whether the type
itself had a `Drop` impl — this correctly implements "even a struct without its own `Drop` still
drops its `String`/`Vec`/etc. fields" and "the user body runs before recursing into children,"
which is the only order consistent with the user's `drop` method still being able to read `self`'s
fields before they're gone (`05-Memory-Model.md`'s `FileHandle` example, lines 265–275, reads
`self.handle` inside `drop`). **Matches spec**, modulo the field-order-within-the-recursion issue
in §6.2.

### 6.4 Destructured-but-unbound sub-values are never dropped
**Confirmed deviation (DEV-030, independently re-verified during this WP's own
review pass — see "Findings requiring a decision"), high severity.** Per `03-Type-System.md`
line 548–550, "every owned value's destructor runs exactly once: at end of scope, at explicit
`drop`, or when its owner is consumed — never twice," and drop-flag tracking exists precisely so
"exactly once" holds even under partial moves. Consuming a value by pattern-matching it (moving
it into the scrutinee, per §5.2) and leaving part of it unbound (`_`, an unmentioned struct
field, a `Wild` sub-pattern) is exactly this "owner consumed" case — the unbound portion's
ownership passed into the match and was never rebound to anything, so per spec it should be
dropped once, at the point the match consumes it (or, at latest, at the end of the arm). Instead,
`match_pattern`'s `Binding`/`Wild` handling (§5.2) only ever **clones** bound sub-values into new
locals (which *do* get properly `cleanup_locals`'d later); anything matched by `Wild` or simply
omitted from a struct/tuple-variant pattern is read via `project`/field lookup for the *purpose*
of testing the pattern, but the original scrutinee `Value` (built once at line 830, `let value =
self.expect_value(*scrutinee)?`) is a plain Rust local that goes out of Rust scope at the end of
`eval_expr`'s `Match` arm **without ever being passed to `self.drop_value(...)`**. Since `Value`
does not implement Rust's own `Drop` trait, nothing STARK-observable happens to it — its
user-defined `impl Drop for T::drop` method is simply never invoked for that sub-value, ever, for
the remainder of the program. Verified with:
```stark
struct Loud { label: String }
impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } }
fn main() -> Unit {
    let pair = (Loud { label: String::from("first") }, Loud { label: String::from("second") });
    match pair {
        (a, _) => { println("matched"); }
    }
    println("after match");
}
```
prints exactly `matched`, `first`, `after match` — **`"second"` is never printed, at any point in
the program**, including after `main` returns (the process exits normally, code 0, after `after
match`). This is not merely a missing-cleanup-timing issue (e.g. "dropped too late") — it is a
genuine, permanent skip: that `Loud` value's destructor runs zero times, not exactly once, in
violation of the spec's core Drop soundness invariant. This affects any pattern with a
`_`/unmentioned-field/`Wild` sub-pattern matched against an owned (by-value, not by-reference)
scrutinee whose unbound portion has a `Drop` impl anywhere in its type. Recorded as DEV-030,
recommended as a **high-priority** WP-C2.2
candidate given it is a silent correctness bug (no error, no crash, just a resource/destructor
that never runs), not merely an ergonomic gap like most of the currently-open DEV-NNN entries.
This finding was independently re-verified (fresh empirical repro, not just trusted from
drafting) during this WP's own review pass — see "Findings requiring a decision" below.

### 6.5 Early return / `?` / `break` out of a scope with live Drop values
`eval_block` (lines 640–660): for each statement, if `eval_stmt` yields anything other than
`Flow::Value` (i.e. `Return`, `Break`, `Continue`, or `Propagate` from a `?`), the function
**still calls `self.cleanup_locals(&locals)?` before returning that flow** (lines 648–651) — so
locals already declared earlier in the same block *are* dropped before an early exit propagates
further outward, and this repeats at each enclosing block as the flow value threads back up
through nested `eval_block` calls, and again via `cleanup_current_frame` at the function boundary
(`call_callable`, lines 623–630, though note: only on the **success** path — see §7 for why an
`Err(RuntimeError)` deliberately skips this). This matches the spec's general Drop-Soundness
requirement for normal (non-trapping) control-flow exits — `break`/`return`/`?` are ordinary
scope exits, not aborts, so destructors must still run for values already in scope. **Matches
spec** for the traps-vs-normal-exit boundary; see §6.4 for the separate, orthogonal bug about
match-arm-internal unbound values.

---

## 7. Panic and trap abort behaviour

Spec: `04-Semantic-Analysis.md` "Runtime Error Semantics" (lines 310–316): a runtime error
(overflow, division/modulo by zero, out-of-bounds indexing, failing `as` cast) or `panic(...)`
terminates the program as an **abort** — non-zero exit status, **no destructors run for live
values**, no unwinding/recovery. `05-Memory-Model.md` line 309–311 restates this under Drop
Soundness. This applies "in every build mode" per `CLAUDE.md`'s summary and `05-Memory-Model.md`
line 387–389 ("Integer overflow (traps in ALL build configurations)").

`interp.rs` represents every trap and every `panic(...)` call uniformly as `Result::Err
(RuntimeError)` (struct defined lines 12–25), propagated via ordinary `?` through
`eval_expr`/`eval_stmt`/`eval_block`. The abort property falls out of a single structural
decision, verified by reading every propagation point: **`?` always short-circuits *before* any
cleanup/drop call is reached.** Concretely:
- `eval_block` (line 647): `let flow = self.eval_stmt(*stmt)?;` — an `Err` here returns
  immediately, *never reaching* the `cleanup_locals(&locals)` call at line 649/658 that runs on
  every other path.
- `call_callable` (lines 623–627): `if result.is_err() { self.frames.pop(); return
  result.map(|_| Value::Unit); }` — explicitly pops the frame **without** calling
  `cleanup_current_frame()` first, then re-propagates the error. Every enclosing call frame up
  the Rust call stack does the same (each is itself inside another `?`-propagating call), so no
  frame at any depth runs its locals'/parameters' destructors once an `Err` is in flight.
- `call_user_method` (lines 1729–1740): same pattern — on error, pops the frame, and if the
  receiver was a `&mut self` receiver, restores the *place* (so the caller's aliasing view is
  left in a defined state) but still does **not** call `cleanup_current_frame`, i.e. still does
  not run Drop for anything else live in that frame.
- `drop_value` itself can fail (its recursive calls use `?`, e.g. line 3184, 3190) — if a user
  `Drop::drop` body itself traps, that also propagates as an ordinary `Err`, abandoning any
  remaining not-yet-dropped siblings, consistent with "abort, no further destructors."

At the top: `run`/`run_item` (lines 397–427) return `Result<Execution, RuntimeError>` to the CLI,
which reports the error and exits non-zero (verified: `panic("boom")` after constructing a live
`Drop`-implementing local printed only the `Error: runtime error: boom` diagnostic — **no drop
message** — and exited with status `1`). **Matches spec exactly**: no observed case where a trap
or `panic` runs a destructor, at any nesting depth, for either a compiler-detected trap (overflow,
div-by-zero, OOB, failed cast — see §8) or an explicit `panic` call.

---

## 8. Numeric conversion and failure

### 8.1 `as` cast semantics
Spec: `03-Type-System.md` "Subtyping and Coercion" → "Numeric Coercions" (lines 357–363):
- int→int: value-preserving if in range, else a trapping runtime error;
- float→float: rounds to nearest representable value;
- int→float and float→int: float-to-int truncates toward zero and **must trap** if the result
  does not fit the target type, "including NaN/Inf."

`interp.rs::eval_cast` (lines 1111–1137), dispatched on the statically-known target type
(`self.tables.expr_types.get(&expr)`, populated by `typecheck.rs`):
- `Int → Int-primitive target`: `check_integer_range` (lines 1139–1162, `i8::try_from`/.../
  `u64::try_from` against the target primitive) — traps (`"integer overflow"`) if out of range.
- `Int → Float-primitive target`: `value as f64` — always succeeds (widening; Core v1's largest
  int type, `i128`-backed at the `Value` level but statically bounded to `Int64`/`UInt64` by the
  type checker, always fits `f64` closely enough that this path is not itself a spec-cited
  precision concern).
- `Float → Float-primitive target`: direct value copy (`Float64`↔`Float32` narrowing/widening is
  handled entirely by the *value*'s own `f64` representation here — note: `interp.rs`'s `Value`
  has no separate `f32` representation, so a `Float64 as Float32` narrowing cast does not
  actually round to `f32` precision at the value level, since both are stored as Rust `f64`; this
  was flagged during drafting but not independently re-verified during this WP's review pass —
  see "Findings requiring a decision" below, listed as unconfirmed).
- `Float → Int-primitive target`: explicitly checks `!value.is_finite() || value.fract() != 0.0`
  (lines 1129–1131) before truncating — this covers NaN and ±Inf (`is_finite()` is `false` for
  both) and non-integral values, matching "including NaN/Inf" from the spec verbatim; the
  truncated `i128` is then still run through `check_integer_range` for the target's exact range.
  **Matches spec.**

### 8.2 Integer overflow / division / modulo — traps in every build mode
Spec: `03-Type-System.md` "Numeric Semantics" (lines 639–644): overflow/underflow and
division/modulo by zero are runtime errors that must trap, in every build configuration (no
wrapping mode exists).

`eval_binary`'s `(Int, Int)` arm (lines 1044–1079): every arithmetic operator uses Rust's
`checked_*` (`checked_add`/`checked_sub`/`checked_mul`/`checked_div`/`checked_rem`/`checked_pow`/
`checked_shl`/`checked_shr`), and a `None` result maps to a `RuntimeError` distinguishing
`"division by zero"` (when `right == 0` and the op is `Div`/`Rem`) from `"integer overflow"`
otherwise (lines 1069–1078); the successful value is then re-validated against the statically
known target width via `check_integer_range` (line 1079) — this is what catches overflow of the
*narrower* target type even when the `i128`-backed intermediate arithmetic itself didn't overflow
(e.g. `Int8` addition that fits in `i128` but not `i8`). **Matches spec**; there is no separate
"release mode" that skips these checks (`starkc` has no such build-mode distinction for
interpreted execution).

### 8.3 Integer literal typing — `Int32` if it fits, else `Int64`
Spec: `03-Type-System.md` line 28: "Default integer type is `Int32` for literals that fit,
`Int64` otherwise."

`typecheck.rs::check_expr`, `Lit::Int` arm (lines 3083–3123, the **DEV-015 fix**, closed in
WP-C1.5): for a suffixed literal (`300u8`), checks the parsed value against
`literal::int_suffix_range_contains` and emits `E0008` if out of range; for an unsuffixed
literal, checks `i32::try_from(value).is_ok()` first (→ `Int32`), else `i64::try_from(value)
.is_ok()` (→ `Int64`), else emits `E0008` (out of range for `Int64` too). `interp.rs::eval_lit`
(lines 914–943) carries a defensive re-check of the same suffix-range condition at evaluation
time (deliberate defense-in-depth in case a literal ever reaches evaluation through a path that
bypassed the typecheck-time check — it does not re-check the unsuffixed-vs-inferred-type case,
since that already depends on the type table `typecheck.rs` produced). **Matches spec exactly**,
and is the concrete site where DEV-015 (fixed this cycle, `C1-exit-report.md` deviation table) is
directly relevant to this document's numeric conversion topic, as WP-C2.1's scope explicitly
anticipated.

---

## 9. Standard-library builtin dispatch

Two structurally distinct dispatch mechanisms coexist in `interp.rs`; describing this topic as
simply "a `Res::Builtin`/`Builtin` enum path" would only fully capture one of them. Both are
documented precisely, since conflating them would misrepresent the reference model:

### 9.1 Free functions and associated (static) functions — via `Res::Builtin`
`hir::Builtin` (`hir.rs` lines 30–124) enumerates every free/associated builtin: `Print`,
`Println`, `Panic`, `Assert*`, `Sqrt`, `Drop`, `StringFrom`/`StringNew`/`StringWithCapacity`,
`VecNew`/`VecWithCapacity`, `BoxNew`/`BoxIntoInner`, `ReadFile`/`WriteFile`, `Some`/`None`/`Ok`/
`Err`, the `Math*`/trig/rounding functions, `SizeOf`/`AlignOf`/`Swap`/`Replace`/`Take`,
`HashMapNew`/`HashMapWithCapacity`/`HashSetNew`, `RandomNew`, the `IOError*` constructors, plus
the tensor-extension builtins (out of Core v1 scope). Name resolution (`resolve.rs`) maps a call
like `String::from(...)`, `Vec::new()`, `println(...)`, `sqrt(...)` to `Res::Builtin(Builtin::X)`
at compile time. `interp.rs::eval_call`'s `Res::Builtin` arm (lines 1172–1179) evaluates every
argument left-to-right (§1.4) and dispatches into `call_builtin` (lines 1240–1535), one `match
builtin { ... }` arm per `Builtin` variant. This is a genuine enum-tag dispatch, and covers every
item in `06-Standard-Library.md`'s Prelude/Core/Math/IO modules that is a free or associated
function (constructors like `Vec::new`, `String::from`, `Box::new`; the `Some`/`None`/`Ok`/`Err`
constructors; `drop`/`size_of`/`align_of`/`swap`/`replace`/`take`; `print`/`println`/`eprint`/
`eprintln`/`panic`/`assert`/`assert_eq`/`assert_ne`; the math module's free functions).

### 9.2 Instance methods on implementation-provided types — via string-name dispatch, NOT `Builtin`
Calls like `v.push(x)`, `s.len()`, `map.get(&k)`, `iter.next()` are **not** routed through the
`Builtin` enum at all. `call_method` (lines 1538–1550) first asks `is_core_value(base)` (lines
1768–1782: is the receiver's *static* type, after stripping any `Ty::Ref` layers, one of
`String`/`str`/a `Ty::Core(..)` container/`Ty::Array`/`Ty::Slice`?); if so, it routes to
`call_core_method` (lines 1784 onward), which dispatches purely on the method-name **string**
(`match name { "clone" => ..., "get" => ..., "iter" => ..., "push" => ..., ... }`, spanning
roughly lines 1784–2527). This is a materially different dispatch mechanism from §9.1: there is
no `Builtin`-enum tag distinguishing `Vec::push` from `String::push`; the *string* `"push"` is
matched, and the receiver's runtime `Value` shape (`Value::Vec`/`Value::String`/etc., inspected
via `self.place_value(&receiver_place, span)?`) determines what actually happens.
`06-Standard-Library.md`'s `Vec<T>`/`HashMap<K,V>`/`HashSet<T>`/`String`/`str` `impl` blocks
(lines 236–347) are the normative source for which method names must exist and their
signatures/behavior; this string-dispatch path is where those signatures are actually realized
at runtime. Worth stating explicitly: "is this a `Builtin`" is not a reliable test for "is this a
normative stdlib operation" in this codebase — most of the stdlib's *method* surface (as opposed
to its free-function/constructor surface) bypasses `Builtin` entirely.

### 9.3 Iterator-combinator "methods" (`map`, `filter`, `fold`, ...) — also string-dispatch, backed by function-pointer values
`06-Standard-Library.md`'s `Iterator` trait (lines 478–493) declares `map`/`filter`/`fold`/
`reduce`/`any`/`all`/`find`/`count`/`collect` as trait default methods taking non-capturing `fn`
values (spec limitation, line 500–504: "the combinator parameters above are *non-capturing*
function types... Capturing closures are a future extension"). These are handled by further
`call_core_method` string-name branches (e.g. `MapIter`/`FilterIter` wrap the source iterator
value plus an `ItemId` naming the passed function, and `iterator_step`, lines 2839–3010, threads
them lazily on each `.next()`/`for`-loop pull). This confirms the combinators are real
lazy-iterator adaptors at the `Value` level (`Value::MapIter`/`Value::FilterIter` wrapping a
boxed source `Value` plus a function `ItemId`), not eagerly materialized — a correctness-relevant
detail for anyone reasoning about side effects inside a `map`/`filter` callback and infinite/
unbounded ranges.

### 9.4 `for`-loop iteration — a narrower, separate mechanism from §9.3
**Confirmed deviation (DEV-031), scope note.** `03-Type-System.md`'s "For Loops"
section (lines 459–469) states `for x in expr` "requires `expr` to have a type that implements
`Iterator`," and explicitly cites `.iter()` methods as a normal way to produce such an expression
("slices and collections provide `.iter()` methods"). `06-Standard-Library.md`'s Iterator trait
(§9.3) is the general mechanism the spec describes for this. However, `interp.rs::eval_expr`'s
`ExprKind::For` (lines 862–877) does not call the generic `next()`/`iterator_step` protocol at
all — it calls `iter_values` (lines 2709–2727), which **only** accepts `Value::Range` (materializes
the whole range eagerly into a `Vec<Value>`, line 2722) and `Value::Array`/`Value::Vec` (consumes/
moves the elements out eagerly, line 2724); anything else — including the exact `.iter()` case
the spec calls out by name — falls to `_ => Err("value is not iterable")`. Verified that this is
actually caught earlier, at **compile time**: `for x in v.iter() { ... }` (`v: Vec<Int32>`) fails
with `[E0001] for-loop requires an iterable value, found 'VecIter<Int32>'` — i.e.
`typecheck.rs`'s `for`-loop type-checking independently only recognizes the same Range/Array/Vec
shapes `iter_values` handles, not the general `Iterator`-trait-bound check the spec describes.
Both layers agree with each other (no compile-succeeds/runtime-crashes mismatch here, unlike
§2.4(b)'s finding), but together they represent a real, current gap between the spec's general
"any `Iterator`-typed expression" `for`-loop rule and the implementation's narrower "only
`Range`/`Array`/`Vec` by value" rule — `HashMap::keys()`, `.iter()`, `MapIter`/`FilterIter`
chains, and any user type implementing `Iterator` cannot currently be used directly as a
`for`-loop's iterable, only iterated manually via `.next()` in a `while`/`loop`. Recorded as
DEV-031; flagged as adjacent to, but distinct from, this
document's core `interp.rs` focus (the compile-time half of this gap lives in `typecheck.rs`).

### 9.5 Related, already-known deviations affecting this section
- **DEV-023** (open, unscheduled): `Display`/`Hash` are declared in the prelude
  (`06-Standard-Library.md` lines 73, 115–117, 123–125) but are not callable as builtin instance
  methods on primitive types via this dispatch mechanism. Cited, not re-investigated here.
- **DEV-013** (closed, WP-C1.3): `.clone()` was previously non-functional for every
  builtin/core-type value; fixed by adding the `name == "clone"` branch in `call_core_method`
  (lines 1834–1852) that directly clones the underlying `Value` for exactly the core types the
  type-checker's `core_method_signature` accepts `.clone()` for (`String`, `Str`, `Vec`, `Boxed`,
  `Option`, `Result`, `HashMap`, `HashSet`, `Range`, `IOError`) — sound because none of these have
  a user-overridable `Clone` impl in Core v1 (operator/method overloading for built-in types is
  out of scope). Cited as confirmation, not re-verified independently in this WP.

---

## 10. Deterministic output expectations

### 10.1 Two different senses of "determinism"
- **Compiler-internal determinism** (diagnostic ordering, report generation, etc.) — **not** in
  this document's scope; DEV-007 (glob-import nondeterminism, closed WP-C1.2) was this class of
  bug, about name-resolution ordering, not compiled-program semantics.
- **Executed-program determinism** — whether a STARK program, run twice with the same input,
  produces the same observable output (`println`/`print`/file writes/exit behavior). This is
  this document's actual topic.

### 10.2 What the spec says
There is no single "Determinism" section in the spec. The closest normative hooks are:
- `06-Standard-Library.md` "Implementation-Provided Types" (lines 13–19): `Box`/`Vec`/`String`/
  `HashMap` internals are implementation-defined; only their public APIs are normative.
- `06-Standard-Library.md` "Performance Notes" (lines 612–616, **non-normative** — falls under
  "Platform Considerations," not "Behavioral Requirements" or "Conformance"): "`HashMap<T>` uses
  open addressing with Robin Hood hashing." This describes a *hash-table* structure, which in a
  typical implementation has iteration order dependent on hash values/insertion history/table
  resizing — i.e. **not** sorted or otherwise predictable from the source alone.
- `04-Semantic-Analysis.md`/`03-Type-System.md` specify every operator/trap's behavior exhaustively
  with no "implementation-defined"/"unspecified" categories outside the types explicitly marked
  implementation-provided above, and outside genuinely environment-dependent operations (file
  IO's success/failure depends on the filesystem, not the language).

Taken together: for any program that does not (a) rely on `HashMap`/`HashSet` iteration order or
(b) depend on filesystem/environment state, the spec's exhaustive, trap-everywhere-else numeric
and control-flow semantics give no room for two conforming implementations to diverge in
observable output — but this is an *inference* from the shape of the rest of the spec, not a
sentence the spec states outright. Flagging this as a soft gap: a conforming-implementations
document at this level of formality would benefit from an explicit determinism statement, but
this document does not invent one.

### 10.3 `HashMap`/`HashSet` iteration order — normative (CD-009), CONFIRMED DEVIATION in the current interpreter (DEV-032)

**Status: settled, CD-009 (corrects an earlier, broken CD-008).** This section originally found
the spec silent on `HashMap`/`HashSet` iteration order, with the only related prose (a
non-normative "Performance Notes" line about open-addressing hash tables) implying unordered
iteration while the interpreter's actual `BTreeMap`/`BTreeSet`-backed behavior was fully
sorted-deterministic — flagged as a CE1/CE2-shaped gap. The user's first answer (adopt
sorted-by-key as normative, CD-008) turned out to be unimplementable as stated: `HashMap<K, V>`/
`HashSet<T>` only bound `K`/`T: Hash + Eq`, never `Ord` (`06-Standard-Library.md` lines 271, 293),
so "ascending key order per the key type's `Ord` impl" could require an implementation that isn't
guaranteed to exist. An external review of the finalized document caught this the same day; the
decision was corrected under CD-009 to **first-insertion order** instead — no `Ord` bound needed,
matching the actual `Hash + Eq` bound, still fully deterministic. `06-Standard-Library.md`'s
"Iteration Order (Core v1)" section (added under CD-009) is now normative:

> `HashMap`/`HashSet` iteration MUST follow first-insertion order: inserting a new key appends
> it; re-inserting an existing key keeps its position; remove-then-reinsert moves it to the end.

**Confirmed deviation (DEV-032).** `interp.rs::Value::HashMap`/`Value::HashSet` (lines 67–68) are
backed by `BTreeMap<Value, Option<Value>>`/`BTreeSet<Value>`, sorted by `Value`'s own internal
structural `Ord` impl (lines 204–321) — not first-insertion order, and not dispatched through the
STARK key type's own `Ord` (which, independently, cannot even be implemented today — see DEV-027,
§2.4). This tracks insertion order only when keys happen to be inserted in already-ascending
structural order; for any other insertion sequence, observed iteration order now diverges from
the normative rule. See DEV-032 in `starkc/docs/conformance/KNOWN-DEVIATIONS.md` for full detail
and the proposed fix (replace the `BTreeMap`/`BTreeSet` representation with an
insertion-order-preserving structure).

### 10.4 Other candidate nondeterminism sources checked and ruled out
- `Random` (`Value::Random(u64)`, an LCG seed): deterministic function of the seed per
  `06-Standard-Library.md`'s own framing ("simple linear congruential generator," line 385) — not
  a real entropy source, so seeded runs are always reproducible; not a determinism concern.
  Float `NaN`/`Inf` comparisons in `Value`'s custom `Ord` use `f64::total_cmp` (line 251), which
  totally orders all f64 bit patterns including NaNs — deterministic.
- No threads/concurrency exist in Core v1 (`06-Standard-Library.md` line 601, explicitly out of
  scope for any conformance profile).
- File IO (`ReadFile`/`WriteFile`, `File` struct — see DEV-009) depends on external filesystem
  state, which is inherently outside the language's own determinism guarantees and not a
  language-level nondeterminism source.

---

## 11. Compile-time-only properties with no runtime representation

### 11.1 Generic type parameters — effectively erased at the `Value` level, not monomorphized
Spec: `03-Type-System.md` "Generics" (line 603): "Instantiation occurs at use sites; Core v1
permits monomorphization or dictionary-passing, but the observable behavior MUST be equivalent."
`interp.rs` is a tree-walking interpreter operating directly over already-type-checked HIR; it
never re-specializes a generic function body per instantiation, and `Value` carries **no generic
type-argument tag at all** — a `Value::Vec(Vec<Option<Value>>)` representing a `Vec<Int32>` and
one representing a `Vec<String>` are structurally identical except for the shape of their
elements; nothing in the runtime representation names the element type `T` independently of what
the elements themselves happen to be. This is neither monomorphization nor dictionary-passing in
the conventional sense — it is closer to full erasure, relying entirely on the type-checker
having already proven every operation sound for the (single, already-substituted) concrete types
each call site uses. Per the spec's own "observable behavior MUST be equivalent" clause, this is
conforming as long as no STARK-visible operation can distinguish the strategies — and this
document found none in the areas investigated (§§1–10): every dispatch decision documented above
resolves via a value's concrete runtime shape (`nominal_item`, `Value` discriminant) or the
type-checker's already-resolved static types (`self.tables.expr_types`), never via a separate
generic-identity check.

### 11.2 Borrow-checker state — purely compile-time, no runtime flag
Spec: `03-Type-System.md` "References and Lifetimes" (lines 232–307) describes an entirely
static system: "either one mutable reference or any number of immutable references," lexically
scoped borrow regions, the shortest-input-lifetime rule for returned references — all enforced
"by the borrow checker" over source-level regions, with no runtime component described anywhere
in the spec. Confirmed by reading `interp.rs` for this document: there is no "is this place
currently borrowed" flag, no aliasing-exclusivity check, and no runtime rejection of a
theoretically-conflicting `&`/`&mut` pair anywhere in the interpreter — `Value::Ref(Place)` (§4.3)
is freely cloned and re-resolved with no bookkeeping about how many other `Ref`s might exist to
the same place. This is exactly right per spec: borrow checking is a compile-time-only analysis
(`borrowck.rs`, requalified in Gate C1's WP-C1.4), and by the time a program reaches `interp.rs`
at all, it has already been proven free of aliasing violations — the interpreter enforcing them
again at runtime would be redundant work with no spec mandate to do so (contrast with
move-tracking, §4.2, which the interpreter *does* redundantly re-check at runtime via the
`Option<Value>` slot state, as an explicit defense-in-depth backstop, not because the spec
requires a runtime borrow flag).

### 11.3 `Copy`/`Drop` trait bounds — compile-time markers, reconstructed (not carried) at runtime
Spec: `03-Type-System.md` "Copy and Drop" (lines 539–557) frames `Copy`/`Drop` purely as
compile-time-checked properties of a *type* (soundness rules about field composition, mutual
exclusivity, moved-out-of-Drop-types restrictions) — nothing about a runtime tag attached to
values. `interp.rs` does not store a "this value is Copy"/"this value implements Drop" bit inside
`Value` itself; instead, `value_is_copy` (§4.2) and `find_drop`/`drop_value`'s Drop lookup (§6.3)
both **reconstruct** the answer on demand by consulting the HIR's `impl` blocks for the value's
nominal `ItemId` (`self.copy_items`, a `HashSet<ItemId>` built once from every `impl Copy for T`
found at interpreter-construction time, lines 536–558; `find_drop`'s linear scan over `impl Drop
for T` blocks, lines 3224–3233). This is a runtime *lookup against compile-time-declared facts*,
not a runtime *representation* of the trait bound itself — consistent with treating `Copy`/`Drop`
as compile-time-only properties whose only runtime footprint is "the interpreter needs to know,
for a given `ItemId`, what was declared about it," which is categorically different from carrying
type-class/vtable information inside each value.

### 11.4 Lifetime information — no representation of any kind, at compile time or runtime
Spec: `CLAUDE.md`'s project-level summary and `03-Type-System.md` lines 232–234 are explicit:
"Core v1 has no lifetime annotations" at all — not "elided at runtime" but never present as
concrete syntax or a first-class checker concept in the first place; the conservative rules
(shortest-input-lifetime, lexical borrow scoping) apply *without* named lifetime parameters
existing anywhere in the type system (`'a`-style syntax is explicitly reserved for a "Future
Feature," `05-Memory-Model.md` lines 217–222). There is therefore nothing for `interp.rs` to erase
— lifetimes are not a compile-time-only property with a runtime-erasure story the way generics or
borrow-checker state are; they are simply absent as a concept at every stage of this
implementation, by spec design. Recorded here for completeness against the roadmap's checklist,
not because anything was found to investigate.

---

## Summary of findings

Every finding below is now a confirmed, numbered ledger entry — none remain as open escalations.
Severity, highest first: **DEV-035** (compile-accepts/runtime-always-crashes for an ordinary,
spec-legal program shape) and **DEV-034** (unconditional duplication of observable side effects)
are the two most severe, both found only during the external-review correction pass, not the
original drafting. **DEV-030** (never-dropped match wildcards) is the most severe finding from
the original drafting pass. The rest are missing-feature, ordering, or residual-risk gaps.

| # | Topic section | One-line description | DEV | Found |
|---|---|---|---|---|
| A | §2.2 | Method dispatch priority (`find_method`) is source-textual-order-dependent, not "inherent shadows trait" per spec | DEV-026 | drafting |
| B | §2.4 | `Ordering` prelude enum unresolvable; `Ord`/`cmp` cannot be conformingly implemented; no `<`/`<=`/`>`/`>=` struct/enum dispatch arm exists | DEV-027 | drafting, re-verified |
| C | §3, §4.4 | `&expr[range]` / `&mut expr[range]` (spec-mandated slice-place syntax) crashes at runtime; the one working slice path materializes a copy, not a view | DEV-028 | drafting |
| D | §6.2 | Struct/enum-named-field drop order is reverse-alphabetical-by-field-name (`BTreeMap` artifact), not reverse-declaration-order | DEV-029 | drafting, re-verified, spec citation added (CD-011) |
| E | §6.4 | Pattern-match `_`/unbound sub-values of an owned scrutinee are never dropped — a permanent, silent skip, not a timing issue | DEV-030 | drafting, re-verified, high severity |
| F | §9.4 | `for` loops accept only `Range`/`Array`/`Vec` directly; general `Iterator`-typed expressions (incl. the spec's own `.iter()` example) rejected at compile time | DEV-031 | drafting |
| G | §10.3 | `HashMap`/`HashSet`: interpreter sorts by structural `Ord`, not the now-normative first-insertion order (CD-009) | DEV-032 | drafting (as spec-silence flag) → confirmed after CD-009 |
| H | §1.5 | `call_core_method` evaluates arguments before the receiver, contradicting the now-normative receiver-first rule (CD-007/CD-010) | DEV-033 | drafting (as inconsistency note) → confirmed after CD-010 |
| I | §2.6, §4.3 | By-value method receiver expressions are evaluated twice, duplicating observable side effects | DEV-034 | **external review** |
| J | §4.3a | References returned from `&self` methods dangle after the method's call frame is popped — breaks an ordinary, spec-legal accessor pattern unconditionally | DEV-035 | **external review** |
| — | (parser, not interpreter) | `parser.rs`'s filename-based module-file-lookup bypass is a residual risk for real user projects, not previously flagged when DEV-014 closed | DEV-036 | **external review** |
| — | §2.1 | `find_associated_fn` only searches inherent impls, not trait associated functions (e.g. `From::from`) | DEV-024 (existing, cited) | Gate C1 |
| — | §9.5 | `Display`/`Hash` not callable as builtin methods | DEV-023 (existing, cited) | Gate C1 |
| — | §10.4 | `File` has no runtime representation | DEV-009 (existing, cited) | Gate C0 |

Both spec-silent gaps this document originally raised (evaluation order; `HashMap`/`HashSet`
iteration order) are settled: CD-007/CD-010 (evaluation order, receiver-before-arguments refined)
and CD-008/CD-009 (iteration order, corrected from sorted-by-key to first-insertion-order after
CD-008 was found broken). DEV-029's field-drop-order rule is settled under CD-011. See
`COMPILER-STATE.md`'s `### WP-C2.1` session record (and its correction-pass addendum) for the
full decision history.
