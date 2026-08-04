# STARK Compiler STATE

## CD-381 — WP-FMT-001 correction packet: six defects, one of them mine to admit (2026-08-04)

An external review of `987369b` reopened WP-FMT-001. It was right on every point, and one of them is
a **false statement in my own closure report**: §6 said the MIR runtime-surface version had changed.
It had not. `MIR_RUNTIME_SURFACE` was still `0.1-A13` while twelve `RuntimeFn` members had been
added across CD-378 and CD-380.

### The six

1. **Runtime surface unversioned.** `0.1-A14` now covers all twelve additions — CD-378's seven
   `Fmt*` members as well as CD-380's five `Fmt*Spec` ones. Both work packages added runtime surface
   without advancing the constant, so a consumer built against A13 would have accepted a program it
   cannot represent instead of rejecting it (V-SURFACE-1). `MIR_VERSION` stays `0.3`: additive
   surface, not a structural change.
2. **The field scanner did not know comments.** `f"{value /* } */}"` mis-scanned, because the
   scanner skipped escapes, strings and char literals but not `//` or `/* */`. A field delegates to
   the ordinary expression parser, so it must admit ordinary comments — including NESTED block
   comments, which 01-Lexical-Grammar §6 requires and which a one-character patch would not handle.
3. **The verifier typed the specification operands but did not require them to be constants.**
   Verified MIR could therefore have carried `FmtIntSpec(v, computed_word, computed_fill)` — dynamic
   formatting beneath a feature defined as statically specified, with a specification word no front
   end had validated. **MIR-0037** now requires both operands to be constants, requires the word to
   decode to a valid specification with zero unused bits, and bounds width and precision by
   LIMIT-FMT-WIDTH/PRECISION. A side effect worth stating: `Spec::unpack`'s defaults for unknown
   align/sign/kind encodings are now unreachable in verified MIR rather than silently normalising
   malformed compiler output.
4. **Inert flags were accepted.** `f"{42:#}"` set `alternate` and rendered nothing different;
   `f"{42:0}"` set `zero_pad` with width zero; `f"{1.25:f}"` asked for fixed-point with no
   precision; `f"{n:<06}"` wrote an alignment that zero-padding then overrode. LEX-FORMAT-003 says
   an implementation must reject a specification it does not act on — these were exactly the case it
   names, and they are refused now.
5. **LIMIT-FMT-SEGMENTS had an off-by-one hole.** The check ran after each field, so the trailing
   literal segment could push the count one past the limit. Checked once over the finished list.
6. **`println(f"...")` was only ever tested through `.as_str()`.** Proving the advertised form
   failed immediately — which is the point. See DEV-174.

### DEV-174, found by fixing the test rather than the code

`eprint`/`eprintln` were typed `&str` while 06-Standard-Library declares
`fn eprintln<T: Display>(value: T)` and PRINT-DISPLAY-001 names all four output functions together.
`eprintln(s)` with an owned `String` was rejected; `println(s)` was accepted. The stderr half of the
runtime surface has carried the full display family since 0.1-A13 and lowering already redirects by
channel — **only the signature lagged**, and no test had ever passed `eprintln` anything but a
`&str`. Both pairs are now typed alike and both go through the same deferred `Display` check.

Testing the convenient form instead of the advertised one is how a gap survives a suite that looks
thorough. Worth remembering beyond this work package.

### Scope, corrected

WP-FMT-001 is **IMPLEMENTED — v0.1 partial interpolation**, not closed. DEV-173 blocks a field
containing an escape sequence, and the original acceptance matrix included
`f"{choose(\"yes\", \"no\")}"`. Declaring closure while that is refused was an overclaim; the
closure report now says so at the top rather than having the claim edited away. Tier-1 qualification
is also still unobserved — the first push failed CI on clippy before reaching those lanes.

### Evidence

`tests/wp_fmt_001_interpolation.rs` — 39 tests, adding the direct `println(f"...")`/`eprintln`
form, comments inside fields (including a nested block comment and `}`/`:` inside one), and the six
inert-flag refusals.

### CI found three more, and one of them is a guard working as designed

The first push of this packet failed CI on three targets:

* `a10_provider_call::runtime_surface_is_current` and
  `a11_host_resource::the_mir_version_records_every_shape_amendment` both PIN the surface constant.
  Bumping to A14 fired them, which is their purpose. **But note what that means:** these guards
  pin the constant, not the surface, so they fail when the constant moves and stay silent when the
  surface grows without it. They were green through CD-378 and CD-380 while twelve `RuntimeFn`
  members were added unannounced. Recorded in `a10_provider_call.rs` rather than left implicit; a
  guard that could fail for the right reason would have to derive something from the `RuntimeFn`
  set itself.
* `adversarial_stderr::the_eprint_family_accepts_only_str_today` — the WP-C7.9 test that pinned
  DEV-174's restriction. Its own doc comment said the lowering already supported every `Display`
  shape, that widening would need "only a signature change and cases", and that this test "fails
  the day that happens, which is the right moment to add them." It did, and they are added: the
  three shapes it rejected now render byte for byte on stderr, plus a negative pinning that a type
  without `Display` is still refused there.

A recorded limitation carrying a test that fails when it is lifted is worth more than a to-do
comment. It turned this repair into one commit instead of a rediscovery.

### Status

DEV-174 CLOSED. DEV-173 remains open and is what stands between "v0.1 partial" and complete
ordinary-expression interpolation. The architecture was not the problem and is unchanged.


## CD-380 — WP-FMT-001: interpolated string literals (2026-08-04)

```stark
let message = f"pkg={name} n={count:04} r={ratio:.2} ok={ok}";
```
```text
pkg=stark n=0042 r=0.76 ok=true
```

**STARK has a complete, compile-time-checked string formatting feature.** `f"..."` produces an
ordinary owned `String` through the `Display` architecture CD-378 and CD-379 established. It is not
a macro, not a variadic call, and not a runtime-parsed format string: segments are split at parse
time, every specification is validated against its field's type at type checking, and no format
string exists in a running program.

### One implementation of the rules, three engines

Alignment, fill, width, sign, radix, alternate prefix and fixed precision live in
`stark_runtime::fmt_spec` and nowhere else. `starkc` already depends on that crate and every native
binary links it, so the HIR oracle, the MIR interpreter and generated code call the SAME functions —
the arrangement that already keeps `x.fmt()` and `println(x)` from drifting. A specification reaches
the runtime as a packed `UInt64` word plus a `Char` fill, both compile-time constants.

MIR gained **five** operations — `FmtPad`, `FmtIntSpec`, `FmtUIntSpec`, `FmtFloat64Spec`,
`FmtFloat32Spec` — not one per syntax combination. Everything about *how* a value renders is in the
word; the operation only says which value family it is.

### Tokenizing without a token soup

`f"..."` is ONE token, scanned exactly like a cooked string. The parser splits it and, for each
field, lexes **the original file over that field's own byte range** (`lexer::tokenize_range`) and
parses it with the ordinary expression parser. So spans inside an interpolation are real file spans
— `tests/span_integrity.rs` now asserts a field's expression is spanned inside its literal — and
nesting, calls, indexing, struct literals and paths are handled by the parser that already handles
them. No source text is ever reconstructed or rewritten.

The scanner tracks depth over `(`/`[`/`{` and consumes escapes whole, so `Point { x: 1, y: 2 }`'s
`:` and `}` stay inside the expression, `module::CONST`'s `::` is not a specification separator, and
`\u{1F600}`'s braces are an escape rather than a field.

### Rulings taken, and why

* **Width counts Unicode scalars**, not bytes and not terminal cells — the only choice that renders
  identically on every platform. It never truncates.
* **Odd centring puts the extra fill on the right**, so `{"x":^4}` is `| x  |`.
* **Sign, then prefix, then zero-padding, then digits**: `-00042`, `0x000000ff`.
* **A negative value in another base keeps its sign and renders its magnitude**: `-255` in hex is
  `-ff`. The host's two's-complement pattern is never exposed.
* **`0x` prefixes both hex cases** — the prefix names the base, the type character chooses digit case.
* **Rounding is half-to-even**; `Float32` renders at its declared width, never widened first;
  non-finite values ignore precision (`NaN`, not `NaN.00`).
* **Precision on a string is REFUSED.** It could only mean truncation, and Core v1 has no ruling on
  scalar-versus-grapheme-versus-byte cutting. Refusing beats guessing.
* **Alignment without a width is refused** — it is a no-op, and almost always a typo.
* **A numeric mode on a generic `T: Display` is refused.** `Display` does not prove integer
  formatting, and inventing a numeric bound to make it compile was explicitly out of scope.

### Ownership

A field **borrows**. `Display::fmt` is `&self`, so a place expression is read, not moved: `f"{x}"`
twice then `use_value(x)` is legal, for a non-`Copy` value and for an affine `Drop`-bearing one. A
temporary field is destroyed exactly once after its bytes are appended. Fields evaluate strictly
left to right, exactly once each — never a second time to discover a width.

### Spec first

**LEX-FORMAT-001/002/003** (01-Lexical-Grammar), **EXPR-FORMAT-001** (02-Syntax-Grammar) and
**STD-FORMAT-002…005** (06-Standard-Library) state the grammar, evaluation order, ownership,
type/spec compatibility, byte-exact rendering, and that interpolation is human-readable formatting
rather than an escaping mechanism for JSON, HTML, SQL, shell or URLs. Compiled spec regenerated; the
fixture corpus is now **114** blocks, the new one triaged in the same change.

### Evidence

- `starkc/tests/wp_fmt_001_interpolation.rs` — 36 tests. Every positive case runs the three-engine
  comparator with stdout pinned in the test; plus debug-vs-release native agreement, and a
  generated-source check that interpolation reaches `stark_runtime::fmt_spec` and that the crate
  contains no `format!`, `write!`, `writeln!`, `std::fmt::Display`, `std::fmt::Debug` or
  `#[derive(Debug`.
- `stark_runtime::fmt_spec` — 16 unit tests on the rules themselves, including `i64::MIN`, `-ff`,
  half-to-even, `Float32` width preservation and scalar-counted width.
- `src/format_syntax.rs` — 12 unit tests on the scanner and specification grammar, including the
  malformed inputs that must diagnose rather than panic.
- `stark fmt` round-trips interpolated literals byte-identically; `packages/stark-fmt` is unchanged
  and green, and interpolation needs no dependency on it.

### Deliberate exclusions, recorded rather than implied

* **No multiline interpolated form.** STARK has no multiline string literal to prefix; §2.5 said not
  to invent one.
* **No raw interpolated form** (`rf"..."`) — deferred, as §2.5 permits.
* **The source formatter reprints an interpolated literal verbatim** rather than re-formatting its
  embedded expressions. Reconstructing the literal risks changing what the program prints, which is
  a semantic difference, not a formatting one. §19 permits this trade and asks that it be recorded.

### Opened

- **DEV-172 — no signed type can express its own minimum value.** `let a: Int8 = -128;` is
  rejected: the magnitude is range-checked before the unary minus. Pre-existing, unrelated to
  formatting, found while testing that formatting a minimum does not overflow. The RENDERER handles
  `i64::MIN` correctly; no STARK program can produce the value to hand it.
- **DEV-173 — an interpolation field may not contain an escape sequence.** A nested string literal
  inside a field necessarily carries the outer literal's escapes, and parsing a decoded copy makes a
  string literal read its own source back (`\"slice\"` for `slice`) because literals read their value
  from their span. Refused rather than mis-parsed; workaround is to bind the value first.

### Status

WP-FMT-001 CLOSED for FMT-0 through FMT-5 as scoped. Formatting is sufficient for CLI output and
structured log lines; see the closure report for the REST-server assessment.


## CD-379 — DEV-BOUND-TRAIT-IDENTITY: a bound denoted whatever trait was spelled the same (2026-08-04)

**A follow-up to CD-378, and a correction to it.** CD-378 unified method candidate *collection*
across user and compiler-known traits. The step before that — deciding WHICH trait a bound denotes —
was still done by spelling, in two passes, and below the front end execution did not use the answer
at all.

### Four failures, all reproduced before any code changed

`typecheck::resolve_bound_trait` and `borrowck::bound_method_receiver` each took
`text(bound.path.span)` and scanned every HIR item for a trait declared with that name.

1. **A qualified bound matched nothing.** `T: traits::Render` compared `"traits::Render"` against
   the declaration's name `"Render"`. The bound contributed no methods, and `value.render()` was
   rejected with *"method 'render' requires the bound 'T: Render'"* — on a function whose signature
   already wrote exactly that bound. Every bound on a trait a package exports through a module was
   unusable.
2. **An unrelated trait captured the name.** `mod unrelated { pub trait Display { fn other(&self); } }`
   anywhere in the program took over every `T: Display` bound. CD-378's own §2 stated this as a
   design — "a user trait of the same spelling wins, exactly as `resolve_path` does" — which was the
   defect written down as a rule: `resolve_path` resolves against the bound's module and imports; a
   global name scan does not.
3. **Declaration order decided ownership.** Two same-named traits, one `&self` and one `self`: the
   borrow checker returned whichever appeared FIRST in HIR item order. The same program compiled or
   failed E0100 depending only on the order its two trait declarations were written in. The
   regression test is that pair, both halves of which must compile.
4. **Execution ignored the identity entirely.** Even with the front end fixed, both engines selected
   an implementation by method NAME on the receiver's nominal, so a type implementing two same-named
   `Render` traits ran the same body for both bounds. The type checker was right and every engine
   below it was wrong the same way — which is exactly what three-engine agreement cannot detect.

**Failures 1 and 2 are refusals. Failures 3 and 4 are acceptances of the wrong program** — an
order-dependent move check, and a call executing a different trait's body than the one type checking
approved. That is the more serious half of this entry.

### The repair: one identity, read from the resolver

`hir::resolved_bound_trait(hir, bound)` reads `TraitRef::res` and nothing else, with exhaustive
matches over `Res` and `ItemKind` — a new resolution or item category forces a decision here rather
than falling into a `_ => None`. `hir::BoundTrait` moved out of `typecheck` so both front-end passes
consume the same type and the same answer. **No spelling-based bound lookup remains in either pass.**

Below the front end: the checker records the selected trait per call site
(`TypeTables::bound_trait_calls`, `Res::Item` or `Res::CoreTrait`); the HIR interpreter passes it to
`find_method`'s already-existing trait filter, and MIR lowering passes it to a new one on
`find_impl_fn`. A filtered lookup considers only that trait's impl — never an inherent method and
never another trait's — exactly as a qualified call does.

**Canonical symbols now carry the trait's module path.** `impl left::Render for Item` and
`impl right::Render for Item` both produced `Item::Render::tag@[]`; the C5.4a linkage preflight
refused the program as "one symbol, two identities", and it was right to. A top-level trait's prefix
is empty, so every pre-existing symbol is byte-identical.

### What CD-378 got right, kept

Candidate collection, selection, ambiguity, the single Core-trait signature table, the `&self`
ruling for `Display::fmt`, and the missing-bound diagnostic all stand unchanged. All 21 cases in
`tests/dev_display_dispatch.rs` pass unmodified; `stark-fmt`'s public API, its 7 tests and both
consumer paths are unchanged.

### Evidence

- `starkc/tests/dev_bound_trait_identity.rs` — 15 tests: qualified bounds through nested generics
  and an impl head; two same-named traits in two modules dispatching to `L` and `R` (which pins
  which BODY ran, not merely that it compiled); an unrelated `Display` failing to capture a Core
  bound and an imported one correctly winning; receiver identity across `&self`, `&mut self` and
  `self`; the declaration-order pair; and a direct assertion that `resolved_bound_trait` returns the
  resolver's own `Res::Item`.
- Correction appended to `starkc/docs/compiler/WP-DEV-DISPLAY-DISPATCH.md` — append-only, stating
  what that report examined and what it did not, rather than editing it to look prescient.

### Opened

- **DEV-171 — an unrelated trait satisfies an OPERATOR bound by spelling.** `use fake::Eq;` then
  `fn compare<T: Eq>(a: T, b: T) -> Bool { a == b }` is ACCEPTED; written qualified (`T: fake::Eq`)
  it is correctly rejected. `ty_satisfies_operator_bound` compares the bound's text against `"Eq"`.
  Not fixed here: it is bound *satisfaction* rather than method identity, the same function also
  serves built-in obligations that have no `TraitRef` (DEV-118), and the repair decides what a
  user trait shadowing a Core trait's name means for operators — a semantics ruling, CE2-shaped.

### Status

DEV-170 CLOSED. **DEV-DISPLAY-DISPATCH (CD-378) is now fully closed**: it was closed for the
property it stated and open on one it did not state — that a bound denotes the trait the resolver
selected. Both hold.


## CD-378 — DEV-DISPLAY-DISPATCH: a compiler-known trait bound was not a trait bound (2026-08-04)

**`fn show<T: Display>(x: T) -> String { x.fmt() }` was rejected.** `[E0302] method 'fmt' not found
for type 'T'` — while the identical shape over a user-declared trait compiled and ran. The bound was
*checked*; it contributed nothing to method resolution.

### The defect is the trait model, not formatting

`typecheck.rs::resolve_method`'s bounded-generic branch resolved each bound by searching
`hir::ItemKind::Trait` items for a matching name. A compiler-known trait has no declaration item —
`resolve.rs` turns `Display` into `Res::CoreTrait(CoreTrait::Display)` and there the trait ends — so
the search returned `None`, the loop fell through, and the impl scan below could not match a
`Ty::Param` receiver either. Method visibility depended on whether a trait happened to be
compiler-known. That is two trait models, and the same hole covered `Ord::cmp`, `Clone::clone`,
`Hash::hash`, `Iterator::next` and `Into::into` on a bounded parameter. `Display` is where it was
noticed only because a `Display` bound has no purpose except calling `fmt`.

**DEV-023 (WP-C2.11) recorded that `Display`/`Hash` as bounds were "already correctly recognized".**
That was true of bound CHECKING and false of everything downstream. It fixed the concrete half
(`"hi".fmt()`) and left the generic half open, and nothing in the entry distinguished the two claims.

### Two more defects were in the same branch, and both are pre-existing

* **The move checker had no bounded-generic receiver at all.** `borrowck.rs::method_receiver`
  returned `None` for `Ty::Param`, and its caller's `None` arm CONSUMES the receiver. Every `&self`
  method reached through any bound moved it — for USER traits too:
  `fn f<T: Named>(x: T) { x.name(); x.name(); }` failed E0100 "use of moved value". Confirmed
  empirically before the fix. This had to be fixed here, because "format a value and keep using it"
  is the property the work package exists to establish.
* **Bound order was a resolution rule.** The branch returned on the first bound supplying the name,
  so `T: A + B` with both declaring `m` picked `A` silently instead of reporting ambiguity.

### What landed: one candidate path

`BoundTrait` makes both kinds of trait *an identity a bound resolves to* — `User(ItemId)` or
`Core(CoreTrait)`. Candidates are collected additively from every bound, de-duplicated by trait
identity, and then ONE selection runs: zero is a missing-bound diagnostic, one is checked, more than
one is E0203 naming both traits. Argument checking, `Self` substitution, associated-type
normalisation and diagnostics are shared from that point. A user trait of the same spelling wins,
the same precedence `resolve_path` already applies.

**No second signature registry was added.** `core_trait_contract` — WP-C7.9 Packet B's table for
checking user `impl` blocks against a Core trait's required shape — already carried
`fmt / Some(Ref) / [] / String`. A bound now reads that table. What a bound makes callable is by
construction what an implementation must provide. The filter on which Core methods a bound exposes
is `receiver.is_some()`, a property of the contract: `Default::default` and `From::from` have no
receiver and therefore no method spelling to resolve. **No method-name branch exists anywhere in the
change** — nothing keys on the string `"fmt"`.

### The missing-bound diagnostic

```text
[E0302] method 'fmt' requires the bound 'T: Display'
   |
   |     x.fmt()
   |     ^^^^^^^ 'T' has no bound that declares 'fmt'
```

Derived from the traits actually in scope, user and compiler-known alike, so it also names a user
trait and says nothing when no trait declares the name (that case keeps the plain "not found"
wording). Fires for `fn bad<T>(..)` and for `fn bad<T: Named>(..)` identically.

### The concrete tail: primitives had no `Display` impl to find

Monomorphisation grinds `T` down before MIR sees it. For a user nominal the ordinary impl path
resolved `fmt` already; for a PRIMITIVE there is no impl item, because 06 declares
`impl Display for Int32` "and similar for other types" and no source file writes those blocks. Seven
`RuntimeFn` variants (`FmtInt64`, `FmtUInt64`, `FmtBool`, `FmtFloat64`, `FmtFloat32`, `FmtChar`,
`FmtUnit`) are the lowering of exactly those declarations, sharing `stark_runtime::format`'s
renderers with the `Print*` family — so `x.fmt()` and `println(x)` cannot disagree in any engine.
`String`/`str` reuse `StringAsStr`/`StrToString`. The `RuntimeFn` matches in `emit_runtime.rs`,
`mir/verify.rs` and `mir/interp.rs` are exhaustive, so all three were forced open by the addition.

### Spec first

**TYPE-METHOD-003** (03-Type-System.md) states that a generic parameter's candidates come from its
bounds and nowhere else, that collection is additive, that written order is not a selection rule,
and that a compiler-known trait contributes through the same collection with no priority.
**STD-TRAIT-002** (06-Standard-Library.md) states the same property from the library side and names
the program a conforming implementation must accept. STD-FORMAT-001 gained the sentence the
ownership work depends on: `Display::fmt`'s receiver is `&self`; formatting borrows and never
consumes, which is what makes `Display` usable at all for an affine type. Compiled spec regenerated.

### Evidence

- `starkc/tests/dev_display_dispatch.rs` — 21 tests. Every positive case goes through the shared
  three-engine comparator with stdout pinned in the test rather than taken from an engine: all
  `Display` primitives through one generic function, user impls, non-`Copy` and affine values used
  after formatting, nested `outer<T>`→`inner<U>` forwarding, both bound orders, an impl-head bound,
  and debug-vs-release native agreement. Negative: missing bound, wrong bound, unknown method,
  non-`Display` concrete type, arity, and ambiguity in BOTH orders.
- `native_selects_stark_formatting_not_rusts` reads the generated crate and requires
  `stark_runtime::format::fmt_i64` present and `format!`, `std::fmt::Display`, `std::fmt::Debug`,
  `#[derive(Debug`, `ToString` absent.
- `packages/stark-fmt` + `packages/stark-fmt-consumer` — the proof workload, registered in the
  qualification gate. `Line::value<T: Display>` and `to_string<T: Display>` are the whole surface.
  7 package tests; consumer runs identically under the interpreter and as a native binary.
- Full report: `starkc/docs/compiler/WP-DEV-DISPLAY-DISPATCH.md`.

### One transitional compromise, stated plainly

`core_trait_contract` is not an ordinary trait DECLARATION, and the preferred architecture asks for
one. Core trait method metadata must eventually be derived from real prelude trait items carrying a
lang-item-like classification, at which point that table and `BoundTrait::Core` both disappear and
`BoundTrait` collapses to a single `ItemId`. That is a resolver-bootstrap change — the prelude has
no source file today — and is a tracked follow-up, not part of this work package.

### Opened

- **DEV-167** — no method-form `to_string()`; needs blanket implementations. `stark-fmt` ships the
  free function. Deferred by decision, NOT by resolver special-casing.
- **DEV-168** — `Display::fmt(&x)` has no MIR lowering ("callee form (C4.5)"). TYPE-METHOD-001 names
  this call as the way to disambiguate an ambiguous trait method, and it runs in one engine of three.
  Found while proving the ambiguity this work package introduces is resolvable.
- **DEV-169** — an explicit `.drop()` call type-checks. Pre-existing, in the CONCRETE path;
  `Drop::drop` was included in the bound surface so the generic path matches it rather than
  disagreeing for no stated reason. Needs a spec-vs-implementation ruling.
- Untracked follow-up: `Clone::clone`, `Hash::hash`, `Iterator::next` and `Into::into` are now
  callable through a bound at the front end, and their concrete lowering is uneven — a program using
  them generically now fails at LOWERING rather than at type checking. Worse diagnostic position for
  shapes that were rejected outright before; not a regression in what compiles.

### Status

DEV-166 CLOSED. The REST server's formatting prerequisite is **met** for rendering values into text;
see the work-package report §8 for the two limits to scope around (no format strings; `Display` is
not a serialisation format — use `stark-json` for payloads).


## CD-377 — installer Phase I: the layout the compiler could not find (2026-08-03)

**The installed toolchain could not build anything on macOS or Windows.** CI caught the symptom on
Linux, where it was a stale path assertion; underneath it was a real defect that Linux alone would
never have shown.

### The installer and the compiler disagreed about the layout

The installer now writes a VERSIONED tree — `lib/stark/current` → `versions/<v>`, payload beneath —
and puts a **symlink** (Unix) or a **copy** (Windows) at `<prefix>/bin/stark`. `discover_runtime`
searched only

```text
<bin>/../lib/stark/starkc/stark-runtime
<bin>/../lib/stark/stark-runtime
```

neither of which exists in that tree.

**It worked on Linux by accident.** `current_exe()` there resolves `/proc/self/exe`, so invoking the
`bin/` symlink already reported the real location and the flat form matched. macOS does not resolve
it; Windows installs a copy, so there is no link to resolve. Same package, three platforms, one
working — the DEV-163 shape exactly.

Reproduced without a Windows machine, by invoking both paths:

```text
/tmp/prefix/bin/stark build                      -> runtime installation is missing
/tmp/prefix/lib/stark/current/bin/stark build    -> Built app
```

Fixed by teaching `discover_runtime` the versioned forms FIRST, so the lookup no longer depends on
the exe path having been resolved through a symlink.

**My earlier "verified end to end" missed this because I never set
`STARK_REQUIRE_INSTALLED_RUNTIME=1`.** Without it the compiler falls back to a source checkout, so
every one of those builds was proving the checkout worked. The environment variable is the whole
experiment.

### `stark doctor`, hardened

Three findings from external review, all confirmed before fixing:

- **Windows executable name.** `("bin", "bin/stark")` was hardcoded, and `install.ps1` runs
  `stark.exe doctor --root` during staging and throws on failure — so a correct Windows package was
  rejected with "staged STARK installation failed manifest verification". Install-blocking, and
  invisible on Unix. Now read from the manifest's `host_target`, which also makes `doctor --root`
  work when inspecting a package built for another platform.
- **The manifest reader was formatting-dependent, and its failure mode was silence.** It split the
  file array on the literal `"\n    {"`. A compact manifest yielded zero entries — and the old
  binary reports that as `manifest_files: ok (0/0 files verified)`. **A verifier that silently
  checks nothing and calls it a pass is worse than one that errors.** Replaced with a real
  recursive-descent parser: escapes including surrogate pairs, bounded nesting, duplicate keys
  rejected rather than last-wins, and sizes that must be whole and non-negative.
- **Manifest paths are now validated.** Relative, no `..`, no drive or absolute form, and unique
  after case folding — Windows and macOS filesystems are case-insensitive, so two entries differing
  only in case name one file and the second certifies whatever the first wrote. A path escaping the
  root would let a manifest certify a file the package never installed.

`serde_json` was recommended and is **not** taken: `starkc` has three dependencies, and adding
`serde` plus a proc-macro chain is a supply-chain decision for the owner, not a code fix. The
defect is closed either way. Nine adversarial tests cover the parser and the path rules.

### Classification — Phase I, not a distribution

```text
Installer Phase I / compiler distribution   IMPLEMENTED
Standalone first-party toolchain            PARTIAL      packages are not in the payload
Offline package/provider build              NOT PROVEN
Public signed distribution                  NOT PROVEN   integrity, not authenticity
```

`manifest.json` detects corruption. It does not establish that the manifest came from a STARK
release — anyone who can replace the payload replaces the manifest with it. Signing, a trusted key,
verification before installation and notarisation are all outstanding.

## CD-376 — HC13 correction: two remote aborts, and a timeout claim counted wrong (2026-08-03)

**External review of `bfceaa0`. Every point was correct and every one is verified in the code
below, not accepted on assertion.** The HTTP client is reclassified **feature-track complete, not
security-release complete**.

### SEC-HTTP-001 and SEC-HTTP-002 — availability vulnerabilities, not parse errors

STARK traps on integer overflow in **every build mode**, so an arithmetic boundary in the parser is
not a wrong error message — it is a **remote process abort**. A hostile server choosing its own
response could stop any client reading it.

```text
SEC-HTTP-001   Content-Length: 18446744073709551616
               guard rejected `value > 1844674407370955161` but ADMITTED the boundary, then added
               a digit up to 9 on top of ...610
SEC-HTTP-002   chunked: "1\r\nx\r\nFFFFFFFFFFFFFFFF"
               `FFFFFFFFFFFFFFFF` accumulates to exactly u64::MAX WITHOUT overflowing, so the size
               parses legitimately; `body.len() + size` then overflowed on any non-empty body
```

**Why HC13's eleven malformed routes missed them.** Both sit exactly where the magnitude guard
stops and the final accumulation still happens. `not-a-number` and `zz` are refused long before the
boundary, so ordinary malformed-input coverage cannot reach either. Adversarial infrastructure is
necessary and not sufficient; the routes have to be aimed at the arithmetic.

Fixed: the Content-Length guard now checks the final digit at the boundary, and the cumulative chunk
check is a **subtraction** — with a `>=` guard first, because a subtraction that underflows traps
exactly as an addition that overflows does, and swapping one for the other would have moved the
defect rather than fixed it.

**Falsified.** Reverting either fix makes its test fail with `integer overflow`. Two new wire routes
(`/bad-length-overflow`, `/bad-chunk-cumulative-overflow`) prove the same against a live peer — and
there, reaching the *next line of output at all* is half the assertion, because before the fix the
process died.

### The timeout evidence was counted wrong, by me

Three stalling routes prove **two** phases, not three: `/slow-headers` and `/slow-body` both report
`ReadResponse`. The report said "three different phases" while its own case inventory showed
otherwise, and the limitations document said "two are not proved" and then listed three.

Worse than the arithmetic: **two of those were filed as "unproven" when they are not implemented.**

```text
ReadResponse    PROVEN
TlsHandshake    PROVEN
WriteRequest    UNPROVEN          deadline installed; no peer fills a receive window
Connect         NOT IMPLEMENTED   DEV-165
Resolve         ABSENT            no mechanism could produce it
```

**DEV-165 — `ClientConfig.connect_timeout` is advertised and never enforced.** The client calls
`connect_no_timeout`, and `stark-net::connect` refuses every non-zero timeout with `Unsupported`.
A caller setting it gets no error and no effect. My limitations document claimed it "IS applied to
the socket" — that was simply false, and it is the worst kind of documentation error because it
reads as reassurance. Deferred to the networking roadmap (it needs a non-blocking connect and a
poll, i.e. a provider ABI change), but the false claim is removed now.

**`Resolve` is ABSENT.** `stark-net::resolve` takes a host, a port and size/count limits, and passes
no duration to the provider. Filing it under "unproven" invited a reader to assume it merely lacked
a test.

### Status

```text
HC0-HC12 feature programme        CLOSED
HC13 adversarial qualification    CLOSED (corrected here)
HTTP client FEATURE track         COMPLETE
SEC-HTTP-001, SEC-HTTP-002        CLOSED (this)
DEV-163, DEV-164                  CLOSED (CD-375)
DEV-165                           OPEN -- deferred to the networking roadmap
PUBLIC RELEASE readiness          BLOCKED -- DEV-165, and no installer exists
```

Evidence: 42 executed cases (13 malformed, 4 oversized, 3 stalls), 36 parser unit tests, 16
packages through the full gate.

## CD-375 — HC13 CLOSED: adversarial peers, DEV-163 and DEV-164 (2026-08-03)

**The HTTP client track is complete: HC0–HC13.** HC13's job was to prove the client **fails
correctly**, which is a different property from proving it works and the one that had never been
tested end to end.

### The finding, which is the point of the packet

**DEV-163 — a read timeout did not report as a timeout on Unix.**

```text
Unix     SO_RCVTIMEO expires -> EAGAIN       -> ErrorKind::WouldBlock -> NetworkError::Interrupted
Windows  SO_RCVTIMEO expires -> WSAETIMEDOUT -> ErrorKind::TimedOut   -> NetworkError::TimedOut
```

So `stark-http-client` reported **"the connection failed"** on Linux and macOS and **"timed out
reading the response"** on Windows — identical peer, identical STARK source. An operator reading the
Unix message would have gone to look at the network instead of at the peer deliberately holding the
socket.

The deadline always worked. Only its **report** was wrong, which is exactly why nothing caught it:
every test through HC0–HC12 used a peer that *answers*, and a timeout that never fires cannot
misreport. It was found within an hour of a peer existing that stalls.

Fixed in `stark-net`'s native provider, where the socket mode is known. A provider stream is always
blocking — the only `set_nonblocking(true)` in that file is the test harness's listener — so
`WouldBlock` from a read or write can mean one thing: the deadline expired. Both platforms now
report `STATUS_TIMED_OUT`.

### What was built

```text
11 malformed routes    status line, version, header name, obs-fold, bare LF, two lengths,
                       length+TE, length value, transfer coding, chunk size, chunk terminator
 4 oversized routes    status line, header line, header count, body ceiling
 3 stalls              slow headers, slow body, a TCP peer that never speaks TLS
```

`stark-http-parser` already had 34 unit tests over an error type with 23 variants — but they assert
that malformed input is *rejected*, not *which* error it produces, and they hand the parser a
literal rather than delivering it over a socket. **No test anywhere would have noticed a bare LF
being reported as a chunk-size error.** These eighteen do, on the wire.

**Each case asserts the NAMED reason, not merely failure.** Eighteen cases all reporting "the
response was bad" would also pass against a client that rejected the valid responses above them in
the same run. The reason is what distinguishes a parser from a wall.

Two design points worth keeping:

- `/big-body` declares 12 MiB and **actually sends it**, against a lowered ceiling. The limit is
  enforced on total bytes read, not on the parsed body, so a peer cannot evade it by under-declaring
  `Content-Length`. A header check alone would pass the peer that lies.
- `tls_stall_peer` **holds** each accepted connection rather than closing it. Closing gives a
  connection error, which is a different outcome from a handshake that never progresses — and the
  wrong one to be testing.

### A second finding, from my own test: DEV-164

Adding the DEV-163 regression test made `a_detached_socket_is_live_and_this_provider_has_forgotten_it`
fail about **one run in five**. It was green in twelve consecutive runs without the new test, so
this was mine — not a flake I merely uncovered.

`stark-net`'s provider table is process-global and `cargo test` runs tests in parallel in one
process. Two tests hand a raw socket OUT of the provider (`detach` -> `into_raw_fd` -> `adopt`),
which leaves a live fd outside any Rust owner for a window; a third test opening and closing sockets
alongside them makes that window observable. The symptom was a detached socket that connected,
accepted its writes, and then reported `UnexpectedEof` instead of the echo.

**The product is not at fault** — `next_id` is monotonic under the lock so handle ids are never
reused, and `detach` consumes the stream with `into_raw_fd` so nothing closes the fd. The defect is
in the test suite's sharing of process-global state. Fixed with the writer-serialising mutex this
repository already uses for the TLS lifecycle tests: **every test that opens a socket takes it, not
only the ones that assert on the table.** A test that merely opens a socket perturbs a table
assertion just as much as one that reads it.

Verified 0/20 with the guard against ~1/5 without.

Two things went wrong on the way to that, and both are worth recording because they cost the most
time:

- I first blamed the echo harness's 5-second handler timeout and **tested it** — raised it to 60s,
  and the failure persisted. Refuted in one run; had I "fixed" it instead, I would have shipped a
  change that did nothing and claimed a cause.
- My first version of the test parked a thread for three seconds. Removing that made the failure
  *rarer* rather than gone, which is the worst possible outcome and was only caught by running the
  suite twenty times instead of once.

### One criterion is partial, and says so

Three of five timeout phases are proved on the wire. `Connect` and `Resolve` are not: a loopback
cannot black-hole a SYN deterministically, and a flaky negative test is worse than an absent one
because it teaches people to re-run the suite. They are recorded as **unproven**, not as working —
which is the distinction DEV-163 exists to justify.

Marking the criterion ✅ on three of five would be exactly the overstatement this packet punished.

### Evidence

```text
40  executed cases for stark-http-client, all native, all against live loopback peers
    (18 of them new: 11 malformed, 4 oversized, 3 stalls)
16  first-party packages through the full gate, exit 0
 3  Tier-1 platforms, no platform gating anywhere in the harness
10  of 10 required fixture servers built
 5  evidence documents
```

Every peer **asserts its bind rather than attempting it**. A skipped peer would silently downgrade
lifecycle evidence to lowering evidence while the gate still reported success — which matters most
on Windows, where a loopback TLS listener is likeliest to fail to bind.

### Status

```text
HC0-HC12  CLOSED
HC13      CLOSED -- one acceptance criterion partial and reported as partial
DEV-163   CLOSED (this)
DEV-164   CLOSED (this) -- provider test suite shared process-global state
```

Still open from the track, each deliberately in its own packet: dot-segment resolution in
`stark-url`; making `Header`/`HeaderMap.entries` private (an API break); and **no installable
toolchain** — no release has been published and `build-release.py` does not stage provider crates,
so a package release would produce a client nobody can run.

## CD-374 — DEV-160: the call thunk, and the shapes it still refuses (2026-08-03)

**Owner ruling accepted (2026-08-03): DEV-160 is NOT closed as a family.** The call-thunk
architecture, the Miri evidence mechanism and the named-refusal boundaries are approved; the defect
splits into four, of which one closes here. Cross-block absorption is DEFERRED to its own work
package, and the HTTP workaround is KEPT.

```text
DEV-160a  same-block direct-call disjoint projections      CLOSED (this)
DEV-160b  borrow returned by an EARLIER call               OPEN / DEFERRED
DEV-160c  conflicting provider-call argument sequence      OPEN / DEFERRED
DEV-160d  borrow surviving beyond the sibling move/call    OPEN / DEFERRED
```

**b, c and d are over-refusals, not unsound execution.** Each is refused by name before rustc, which
is the correct outcome for a shape the backend cannot emit: without it they reach the user as
`E0502` inside `mod stark_proj` — a correct compiler error about code they never wrote.

### What a thunk is

One generated function per conflicting call site, in `mod stark_proj` beside the wrappers it calls:

```rust
pub fn stark_thunk_23main_40_5b_5d_23bb2<'a>(
    s0: &'a mut stark_runtime::slot::ValueSlot<stark_ty_230_40_5b_5d>,
) -> u32 {
    let p0: *mut stark_runtime::slot::ValueSlot<stark_ty_230_40_5b_5d> = s0;
    unsafe {
        let a0 = stark_refraw_23struct_230_23f0::<'a>(p0);
        let a1 = stark_moveraw_23struct_230_23f1(p0);
        let a2 = stark_copyraw_23struct_230_23f2(p0);
        stark_consume_40_5b_5d(a0, a1, a2)
    }
}
```

The slot arrives ONCE through a real `&'a mut`, one raw pointer is derived from it, and every
operand is evaluated through that pointer **in MIR order**. There is one `&mut` in existence, so
there is nothing left to conflict with; `'a` comes from a real reference, so a borrow the thunk
hands on has honest provenance. The call site is `stark_proj::NAME(&mut _1)` — one safe call, §7.8
intact.

### The part that was not in the plan: absorbing the borrow

The conflicting `&` is usually **not in the argument list at all**. `f(&p.name, p.body)` lowers to a
`RefOf` STATEMENT filling a temporary, and only the temporary is an argument. A thunk that took over
the argument list alone would leave that borrow live beside its own `&mut` and change nothing.

So the thunk takes over the borrow's statements too, and `emit_bodies` suppresses them. Three
conditions gate it, and each was needed:

```text
same block          moving the RefOf inside the thunk must not move it past a branch
projected base      a whole-slot borrow has no raw twin (and STARK rejects it beside a move anyway)
every read is here  the definition is suppressed, so nothing may need the value afterwards
```

Delaying the borrow is sound because the front end has already proved nothing between it and the
call can mutate what is borrowed. A *disjoint* sibling may be moved in that gap — and re-deriving
through a raw projection reads the untouched field either way, which is exactly what a whole-value
accessor could not do.

**`let r = &p.name;` lowers to a PAIR** — `_8 = &_1.0` then `_7 = copy _8` — and only `_7` reaches
the call. Following that chain, and suppressing every statement along it, is the difference between
absorbing the reported idiom and absorbing nothing.

### DEV-160d and DEV-160b, and why each is a refusal rather than a gap

**DEV-160d — a borrow that outlives the call.** `let r = &p.name; f(r, p.body); use(r);` cannot be absorbed —
suppressing the definition breaks the later read — and cannot be left alone. Refused, naming the
local and the field.

**DEV-160b — a borrow arriving through an earlier call.** This is the shape DEV-160 was reported as:

```text
send_once(builder.url.as_str(), builder.headers, builder.body)
```

`as_str` runs in an earlier block and returns a `&str` borrowing `builder`. By the outer call it is
an ordinary non-slot local carrying no sign of where it came from, so the backend now **traces
borrow provenance** — `RefOf` seeds it, copies and borrow-carrying aggregates propagate it
(OWN-CARRY-001 makes provenance structural), and a call's result inherits its arguments', which is
STARK's own shortest-input rule read as may-alias. A by-value argument whose type could carry a
borrow and whose provenance meets a participating slot is refused by name, with the workaround
stated.

Absorbing it means absorbing the intermediate CALL, across a block boundary, turning that block's
terminator into a `goto`. That is a second mechanism, not an extension of this one, and it changes
control flow — **flagged for an owner ruling rather than taken unilaterally.** The HTTP client's
`send()` workaround therefore stays, and the comment above it in `stark-http-client/src/lib.stark`
now says which half of DEV-160 closed and which did not.

Provenance over-approximates, filtered by type: without the type filter,
`consume(p.taken, p.kept.len())` would be refused, because `len` takes `&p.kept` and the relation
propagates — but the result is a `UInt64` and borrows nothing.

### DEV-160c — the provider audit

A provider call never reaches `emit_call`. It is emitted as a statement SEQUENCE — one
`let __prov_aN = ...;` per argument (A10/CD-200) — which has the SAME conflict: `__prov_a0` holding
a shared borrow is a live local when `__prov_a1` moves a sibling through `&mut`. The thunk does not
apply (there is no single expression to replace, and the ABI's out-parameters and handle transfers
are not arguments a thunk could carry), so it is refused by name. This was the audit the addendum
asked for, and it found a real path rather than confirming an unreachable one.

### What bounds the change

A call that does not conflict reaches none of this. Both mechanisms that could touch it — the plan
lookup in `emit_call`, the statement suppression in `emit_one_block` — are gated on a plan existing,
so `ordinary_calls_plan_nothing` asserts the detector stays silent on four shapes each ONE condition
away from conflicting: two `Copy` reads, a lone borrow, a lone move, a whole-value move.

One plan, three consumers. `emit_projections::collect` skips the argument lists the plans cover,
`emit_projections::emit` renders the helpers each plan names, and `emit_call` looks its plan up. The
addendum required this and it is not decoration: DEV-162 shipped an `E0425` precisely because the
emitter named a helper the collector had never been asked for.

### Miri, and keeping the fixture honest

A thunk is generated code, and Miri cannot run what has not been generated. So `stark-runtime`
carries a hand-written one and a pinned CI job (`nightly-2026-07-20`,
`-Zmiri-strict-provenance`) runs the slot primitives under it — the only check here that can tell a
sound raw projection from one that merely happens to work.

That arrangement has an obvious failure mode: the generator changes, the fixture does not, and the
Miri job keeps proving something about code the compiler no longer emits. So the fixture publishes
`GENERATED_THUNK_SHAPE`, and `the_miri_fixture_matches_what_the_generator_emits` derives the same
sequence from a freshly generated thunk by resolving each wrapper to the primitive inside it. The
two must agree. **Neither check is worth much without the other.**

`-Zmiri-ignore-leaks` is required and does not weaken the aliasing check: three `should_panic` tests
hold heap values when the panic aborts them, which is what those tests are for.

### Evidence

```text
8   DEV-160 tests -- 4 executed through HIR, MIR, native-debug AND native-release,
                     2 refusal assertions, 1 bounding invariant, 1 fixture-drift check
26  stark-runtime slot tests, all green under Miri with strict provenance
23  suites re-run green locally: MIR lowering/verification/differential, the
    three-engine differential, ownership, aggregates, generics, function values,
    providers, host resources, and DEV-135/150/154/162
```

Local runs are scoped by design; `cargo test --workspace` belongs to CI, which is what the totals
should be read from.

The four executed cases are compared across engines rather than each asserted to exit 0 — including
the ordering case, where the `Copy` read is deliberately the THIRD argument, after the move, so a
reordered thunk would read storage a sibling had already left.

### Status

```text
DEV-158   install through a whole-value accessor      CLOSED (CD-371)
DEV-162   read through a whole-value accessor         CLOSED (CD-372)
DEV-160a  same-block conflicting evaluation           CLOSED (this)
DEV-160b  borrow through an earlier call              REFUSED by name; DEFERRED by ruling
DEV-160c  provider-call argument sequences            REFUSED by name; DEFERRED by ruling
DEV-160d  borrow outliving the call                   REFUSED by name; DEFERRED by ruling
```

### Why DEV-160b is a work package and not a follow-up commit

It is not an extension of the thunk. It has to absorb an EARLIER call terminator, replace that
terminator with a `goto`, preserve the failure and control-flow behaviour of the call it absorbed,
preserve the returned reference's provenance, potentially span several blocks, and coordinate more
than one call result. Every one of those is a property the current mechanism does not touch.

## CD-373 — DEV-160 foundation: the raw-slot primitives, and an order finding (2026-08-03)

**Owner ruling accepted (raw-pointer call-site thunk; argument reordering PROHIBITED because CD-007
freezes left-to-right evaluation).** This lands the foundation only. **DEV-160 is still OPEN** — no
thunk is generated yet and the HTTP workaround stays.

### What landed

Four `unsafe` raw-pointer primitives on `ValueSlot`: `field_ref_raw`, `move_field_raw`,
`copy_field_raw`, `take_raw`. They take `*mut ValueSlot<T>` and never form a reference to the slot,
so a borrow of one field and a move of a disjoint sibling can be live together — which the
`&self`/`&mut self` forms cannot express, because each borrows the whole slot. That inexpressibility
IS DEV-160.

**The ruling's lifetime point was decisive and I would have got it wrong.** My plan was to change
the existing helpers to take raw pointers and keep returning `&F`. A safe function returning a
reference derived from a raw pointer alone has no lifetime source — the borrow would be unbounded
and the signature a lie. So these are `unsafe`, carry an explicit `'a`, and are callable only from a
generated thunk that takes the slot ONCE through a real `&'a mut ValueSlot<T>`, which is what
anchors every reference it hands on.

The aliasing rule is written into the module: inside such a thunk no `&ValueSlot` or
`&mut ValueSlot` may be reconstructed after a field reference has been derived. Every access goes
through the raw pointer for the thunk's whole body.

Four tests, including the shape that motivates the whole thing — a field borrow and a sibling move
alive simultaneously — plus dead-slot and partial-slot refusals through the raw path, so the checks
are not skipped merely because the caller holds a pointer.

### A finding the thunk design has to absorb

Working through the emission, the thunk cannot take only the CONFLICTING slot and receive the other
arguments pre-evaluated. Evaluating a non-conflicting argument at the call site would place it
BEFORE the projections performed inside the thunk, which is the argument-order change the ruling
prohibits.

So the thunk must take **every distinct local an argument reads**, by `&mut ValueSlot<..>`, and
perform **every** operand read inside itself, in MIR order. That is consistent with the ruling's
"performs the fixed disjoint accesses internally ... in MIR order, and invokes the callee" — stated
here because it is a bigger obligation than "hand the thunk the conflicting slot", and it decides
the thunk's signature.

Constants and non-slot scalar locals may still be passed by value: their reads are unobservable and
order-insensitive.

### Remaining for DEV-160

```text
conflict detector       same slot base in >= 2 argument places, at least one requiring &mut
thunk plan identity     body/call-site, callee identity + signature, base slot type,
                        ordered argument modes, ordered projection chains, return type
thunk generation        into mod stark_proj, safe signature, raw body, MIR order, callee call
call-site emission      one safe call, no unsafe in the generated MIR body
negative controls       the owner's fifteen, incl. drop-exactly-once, overlap refusals,
                        indirect/runtime/provider call audit, debug AND release agreement
```

### Evidence

30 runtime tests, 161 across MIR verification, the three-engine differential and DEV-162's
regression, clippy clean. Nothing behaves differently yet: the primitives are unreferenced by any
emission path.

## CD-372 — DEV-162 CLOSED; DEV-160's obvious fix does not work, and here is why (2026-08-03)

### DEV-162 — reading a sibling field of partially-moved storage

Sibling of DEV-158, same root cause. Once a field is moved out the storage is `Partial`, and a read
of an UNTOUCHED sibling was emitted as `&slot.get().f1`, where `get` requires a complete value:

```text
_7.reinit(stark_proj::stark_move_23struct_230_23f0(&mut _1));
_13 = (&_1.get().f1);   // aborts: the slot is PARTIAL
```

`copy_field` already covered the `Copy` case by value (WP-C6.1b). This is the rest: a non-`Copy`
field, borrowed. `ValueSlot::field_ref` reads through a raw projection, so it never materialises a
reference to the surrounding value. `HelperOp::Ref` joins Move/Copy/Drop/Write, and the emitted form
is `(*stark_proj::stark_ref_…(&_1))` — dereferenced, because callers in `Borrow` mode prepend their
own `&` and need a place expression.

**The part missed first.** `Rvalue::RefOf` carries a PLACE, not an operand, so `rvalue_operands`
returns nothing for it and the collector never generated the helper the emitter had already named.
That surfaces as `E0425` inside the generated crate — a name error in code nobody wrote — not as any
diagnostic the compiler produces. Collector and emitter must agree and nothing but a build proves
it, which is now what the regression test does, across three engines.

### DEV-160 — the obvious fix does NOT work, recorded before anyone tries it

```stark
consume(p.url.as_str(), p.headers, p.body)   // accepted by STARK, E0502 in generated Rust
```

The instinct — and my own first plan — is to hoist each argument into a temporary before the call:

```rust
let __a0 = stark_ref_…url(&_1);
let __a1 = stark_move_…headers(&mut _1);   // still E0502
```

**It does not help.** `__a0` holds a shared borrow of `_1` that stays live until the call consumes
it, so every later `&mut _1` still conflicts. Sequencing the statements changes nothing about the
borrow's extent.

Two options actually remain, and both have a real cost:

```text
reorder    emit every `&mut` argument BEFORE any borrow that lives into the call. Sound here —
           a borrowed field and a moved field are necessarily disjoint or MIR would have refused
           the program — but it changes ARGUMENT EVALUATION ORDER, which CD-007 fixes. Needs a
           decision, not just an edit.

raw ptr    give the helpers `*mut ValueSlot<T>` parameters, which do not participate in borrowck.
           Conflicts with §7.8's rule that generated MIR bodies contain no `unsafe` of their own,
           unless the unsafety is pushed entirely inside `mod stark_proj`.
```

Recorded rather than attempted. Getting this wrong quietly changes evaluation order for every
call in the language.

### Where the family stands

```text
DEV-158  install through a whole-value accessor      CLOSED (CD-371)
DEV-162  read through a whole-value accessor         CLOSED (this)
DEV-160  whole-slot borrows, disjoint projections    OPEN — needs an evaluation-order ruling
```

Two of three closed. The remaining one is not a bug to fix but a decision to take.

### Evidence

378 across the MIR, native, ownership and aggregate suites; 26 runtime; clippy clean; all 16
packages green. `dev162_partial_field_read.rs` compares three engines rather than asserting each
exits 0 separately, and covers the `Copy` sibling alongside the non-`Copy` one so a regression in
either is visible against the other.

## CD-371 — DEV-158 CLOSED; the diagnosis was wrong twice before it was right (2026-08-03)

**Assigning over a struct field whose old value is a drop unit now works natively.** Both HTTP
workarounds are removed and the packages still build and pass.

### The defect was in TWO places, and I had only found the second

I documented DEV-158 twice as "no operation returns a slot from `Partial` to `Whole`". True, and not
the abort. Reading the generated Rust — which is what I should have done first — showed it:

```rust
_7.reinit(stark_proj::stark_move_23struct_231_23f0(&mut _3));  // slot -> PARTIAL
_3.get_mut().f0 = _6.take();                                   // <- ABORTS: get_mut needs WHOLE
```

The INSTALL uses a whole-value accessor on storage the matching move-out just made partial. The
missing state transition is real and necessary, and it runs after a line that never completed. So
the fix is two halves:

```text
ValueSlot::write_field    a raw-projection field write, valid over partially-moved storage
ValueSlot::mark_whole     the state transition, guarded by MIR's drop flags
```

`HelperOp::Write` joins Move/Copy/Drop in the projection-wrapper generator, and `emit_assignment`
routes a projected destination in slot-backed storage through it. `ptr::write`, not assignment: the
field is uninitialised at every generated call site because CD-012 requires the old unit to be moved
out first, so there is nothing to drop and assigning would drop garbage.

### A gate copied without re-deriving its reason

`emit_storage_whole` was written by copying `emit_storage_dead`'s gating, including its no-op
drop-plan check. That check is right for `finish_partial` — a no-drop slot is written with `reinit`,
which has no dead-slot check for a storage END to satisfy — and **wrong** for `mark_whole`: a slot is
made partial by a field MOVE, and `take` aborts on it whether or not the whole-type plan is a no-op.

Worse, a no-op whole-type plan is the COMMON case, because MIR decomposes an aggregate's drop into
per-unit flag-guarded drops. So the copied gate suppressed emission for exactly the shape DEV-158 is
about. The reproducer's `Config` reported `plan noop = true` and `mark_whole` was never emitted — the
statement was in the MIR and produced no code. Found by instrumenting rather than by reading it
again.

### The guard is correct in both directions

Proved, not asserted. A struct with two droppable fields, one moved out and never restored, the
other assigned: the guard must NOT fire, and a wrong fire is observable rather than silent — the
scope-end `finish_partial` would hit a WHOLE slot and abort by name. It passes.

### What this did NOT fix

**Reading one field of partially-moved storage still aborts.** `t.b.as_str()` after `t.a` was moved
out goes through `get()`, which requires WHOLE. Same family — a whole-value accessor over partial
storage — and the same family as DEV-160's whole-slot borrows. Filed as DEV-162.

That makes three in one class, and they want one fix rather than three:

```text
DEV-158  install through a whole-value accessor        CLOSED
DEV-162  READ through a whole-value accessor           OPEN
DEV-160  whole-slot borrows for disjoint projections   OPEN
```

### Evidence

3 new runtime tests (write_field's siblings survive, mark_whole in all three states), 315 across the
MIR and native suites, 26 runtime, clippy clean, all 16 packages green. The original reproducer and
the HC11 three-field shape both run natively and agree with the interpreter, and both HTTP
workarounds are deleted rather than merely marked removable.

### The process note

Three diagnoses, two wrong, and the two wrong ones were both reasoning from the source rather than
from the artefact. The generated Rust was available the whole time and named the failing line in one
read. When a backend defect is about what the backend EMITS, read the emission first.

## CD-370 — the diagnostic-injection hole I opened while closing the wire one; DEV-161 (2026-08-03)

**From a second Codex review of CD-369. Both findings were right, and the first is a mistake worth
naming precisely.**

### The repair reintroduced the injection one layer out

CD-369's commit message argued that a rejected VALUE must never be echoed, because it is
attacker-influenced and echoing it moves the injection into the log. Correct — and then the same
commit carried the rejected NAME verbatim into the error text. An invalid name may itself contain
CRLF. My own regression test asserted the reported name was exactly `X-Test\r\nInjected`.

So the reasoning was right and the code did the opposite of it, in the adjacent case.

Fixed **structurally** rather than by escaping:

```stark
InvalidHeaderName            carries NOTHING — the name is what failed, so there is no safe
                             version of it to report
InvalidHeaderValue(name)     carries the name, safe HERE and only here because the name is
                             checked FIRST and this variant is unreachable until it passed
```

The order of the two checks is the safety argument, and it is stated in the source. Escaping was
rejected as the primary fix: a sanitiser is something a future call site can forget, whereas a
variant carrying no string cannot leak one. The new test renders the error and scans it for control
bytes, so it asserts the property rather than the shape.

### `Content-Type` gets the same singleton policy as `Location`

`json_checked` used `get_first`; two `Content-Type` headers are two contradictory claims about the
same bytes, which is the same class of silent choice. Now `AmbiguousContentType`. And
`RequestBuilder::json` REFUSES when the caller already set one, rather than appending a second —
appending would put the contradiction on the wire and leave the winner to the server.

### DEV-161 — an ambient `CARGO_TARGET_DIR` breaks every native build

Cargo's default output is `<manifest dir>/target`, which is where the backend looks. An exported
`CARGO_TARGET_DIR` overrides it, the child inherits it, the build SUCCEEDS elsewhere, and the
backend reports "Cargo succeeded but the expected binary is missing" — naming neither the cause nor
the variable. `CARGO_TARGET_DIR` is a common global setting, so any developer with it exported could
not `stark build` at all.

Fixed by passing `--target-dir` explicitly, with the read path reusing the same value, so nothing
about the environment can separate where the build writes from where the backend looks.

**How it was found is the uncomfortable part.** It broke `mir_statement_consumers` and
`c788_resource_lifecycle`, and I reported both as pre-existing environmental failures unrelated to
my changes — twice. The second time I "confirmed" it by stashing every change and re-running. **That
control was worthless: the stashed run had the same variable exported.** Controlling for the code
while holding the environment fixed proves nothing about the environment. The review pushed back on
the dismissal, which is the only reason it got looked at.

Both suites now pass, including under the hostile variable. `StorageWhole`'s handling by every
statement consumer is therefore execution-evidenced, not merely compile-evidenced — which was the
review's specific concern.

### Still open, unchanged

```text
dot-segment reference resolution   bounded RFC 3986 resolver, belongs in stark-url
Header/HeaderMap field privacy     an API break, its own change
DEV-158                            lowering + runtime guard, the hard half
DEV-160                            field-granular generated projections
DEV-159                            native build racing its own dependency build
HC13                               not started
```

## CD-369 — HC12.1: a proven CRLF-injection hole closed, plus two P1 redirect gaps (2026-08-03)

**From an external Codex review of CD-368. All three findings were real; the first is a security
defect that predates HC12 and I verified it by exploit before fixing it.**

### P0 — header validation was bypassable, and it was reachable

`stark-http-core::header()` validates on construction, and the serializer trusted that. But
`Header`'s fields and `HeaderMap.entries` are PUBLIC, so a header can come into being without ever
touching the constructor. Written as a probe and run:

```text
value: "safe\r\nInjected: yes"

GET / HTTP/1.1\r\n
Host: a.test\r\n
X-Test: safe\r\n
Injected: yes\r\n        <- a header the caller never wrote
Connection: close\r\n
```

CRLF header injection, from safe STARK, no `unsafe` and no provider.

**The invariant is now enforced where the bytes are emitted**, because that is the only place that
cannot be bypassed by constructing the value differently. `SerializeError::InvalidHeader(name)`
carries the NAME only — a value rejected for containing CRLF is attacker-influenced by definition,
and echoing it into a log moves the injection one layer out instead of stopping it.

The regression test IS the exploit, plus bare CR, bare LF, NUL, and four invalid name shapes — and
one control asserting a well-formed hand-built header still serializes, so the repair rejects what
cannot be written rather than everything built without the constructor.

**Still open, recorded not fixed:** making `Header`/`HeaderMap.entries` private behind validated
accessors. That is an API break and belongs in its own change; this closes the hole.

### P1 — two URI-reference forms were silently mis-resolved

| base + Location | was | now |
| --- | --- | --- |
| `/one/two?q=1` + `?page=2` | `/one/?page=2` — a DIFFERENT resource | `/one/two?page=2` |
| `/one/two` + `ftp://other.test/f` | `http://a.test/one/ftp://other.test/f` | refused |

The first silently requested something the server did not name. The second fell through to the
relative-path branch: not dialling FTP is not the same as being correct. Fragment-only references
are refused too — they address a position in the current document, so there is nothing to fetch.

The scheme check follows RFC 3986 (`ALPHA *( ALPHA / DIGIT / "+" / "-" / "." ) ":"`, colon before
any `/?#`), so ordinary paths containing a colon still resolve — pinned by test, because a check
that swallowed `a/b:c` would be its own bug.

### P1 — a duplicate `Location` was first-wins

Now `get_singleton`, and `AmbiguousLocation`. Two `Location` headers are two destinations, and
picking one silently is a choice between things the server said — the class of disagreement request
smuggling is built on. `headers_for_next_hop` also propagates a validation failure instead of
silently omitting the header, since "the second request quietly lost a header" is indistinguishable
from a bug at the far end.

### Still open from the review

**Dot-segment removal (`.` / `..`) is not implemented.** Codex is right that the real answer is a
bounded RFC 3986 resolver in `stark-url` rather than a second URL implementation growing inside the
HTTP client, and a half-written normaliser is worse than none. Recorded for HC13's packet.

### DEV-158 — the fix is in progress, not landed

`ValueSlot::mark_whole` exists and is proven (3 tests: partial→whole with the field written back and
every whole-value operation working afterwards, idempotent on whole, refused on dead).
`Statement::StorageWhole` is defined and wired through the verifier (MIR-0036), the interpreter
(inert), linkage and the emitter. **Lowering does not emit it yet**, so nothing behaves differently
and the workarounds stay.

One finding while sizing it: the cheap static shortcut — "if the assigned place covers all the
local's drop units, wholeness follows" — is TOO WEAK. `RequestBuilder` has three droppable fields,
so `out.body = body` covers one of three: exactly HC11's case, still broken. It would have looked
like a fix and left the motivating instance failing. The real emission needs the runtime conjunction
of the local's drop flags, which is what remains.

## CD-368 — HC12 CLOSED: safe redirects; DEV-160 found (2026-08-03)

**Redirect support is opt-in, bounded, and cannot silently forward credentials to another origin.**
All three words are separate mechanisms. Full record:
`STARKLANG/docs/http-client/HC12-REDIRECT-EVIDENCE.md`.

```text
opt-in        follow_redirects defaults false; off, a 3xx is RETURNED, not errored — a redirect
              is a valid answer and hiding it would misreport what the server said
bounded       max_redirects (5) AND loop detection over every visited URL — two different faults,
              two different errors, because raising the limit should fix one and not the other
not silently  Authorization and Cookie stripped on any origin change; opting out is possible and
              is named `preserve_authorization_same_origin_only`
```

### Two rulings worth stating

**301/302 rewrite POST to GET**, contradicting a literal reading of the RFCs and matching every
browser and `curl -L`. The letter would send a POST body to a target the origin server redirected a
POST *away* from — both surprising and the more dangerous reading. 307/308 preserve and replay,
which is safe only because a body is a buffered `Vec<UInt8>`.

**Origin comparison uses the EFFECTIVE port**, so `https://h/` and `https://h:443/` are one origin.
Otherwise a redirect that merely spelled the port differently would strip credentials for no reason,
and callers would learn to turn the stripping off — which is how a safety default dies.

### A bug the pure tests could not have found

The 303 case asserts against what the PEER received. Method and body were already correct, and
`Content-Type: text/plain` was still riding along on a bodyless `GET` — a claim about content that
is not there. Dropping a body now drops every header that describes one. The rewrite-table test
alone would have passed; the echo route reflecting the actual wire is what caught it.

### DEV-160 — place-granular borrows, whole-value projections (OPEN)

STARK's borrow checker is place-granular (DEV-154) and correctly accepts disjoint-field borrows in
one call:

```stark
send_once(client, builder.method, builder.url.as_str(), builder.headers, builder.body)
```

The generated projections take `&slot` and `&mut slot`, losing that granularity, so **rustc rejects
the generated code**:

```text
error[E0502]: cannot borrow `_2` as mutable because it is also borrowed as immutable
```

A correct program refused by the backend. Worked around by moving the fields into locals first.

**This is the same shape as DEV-158** — the slot abstraction is whole-value while the ownership
model is place-granular — and it is the third defect in that family. Whatever fixes the
`Partial`/`Whole` transition should be scoped to look at projection granularity generally rather
than at field assignment alone.

### Evidence

38 `stark-http-client` tests (9 new) and 22 consumer cases (10 new), all against live peers. The
credential case reads the WIRE rather than the policy flag: the cleartext peer redirects to the TLS
peer, and the echo route reflecting `GET|-|-|` proves the header was absent on the second request.
The bound and the loop are proved separately — `/r-loop` revisits one target, `/r-hopN` walks an
ever-lengthening chain of distinct ones.

## CD-367 — HC11 CLOSED: JSON convenience, and a strict UTF-8 decoder (2026-08-03)

**Common JSON REST calls no longer require manual byte conversion or header construction, and HTTP
core still knows nothing about JSON.** Full record:
`STARKLANG/docs/http-client/HC11-JSON-EVIDENCE.md`.

```text
stark-http-core     TextDecodeError, decode_utf8, HttpResponse::body_text
stark-http-client   RequestBuilder::json, HttpResponse::json / json_checked, JsonBodyError
```

The split is forced: `stark-http-core` must not depend on `stark-json`, or everything that parses a
header pulls in a JSON parser. `body_text` lives in core because `HttpResponse` is declared there.

### The substantial part was a UTF-8 decoder

There is no `String::from_utf8` in the core surface, so HC11 wrote one. The accepted set is explicit
RANGES, not "leading byte then N continuations", because the short form accepts three things it must
not: overlong forms (`C0 80` is NUL in two bytes, invisible to a checker scanning decoded text),
surrogates (`ED A0 80`–`ED BF BF`), and anything above `U+10FFFF`. Each is a documented
parser-differential bug class — two components disagreeing about what a byte string means is how a
filter gets bypassed.

Strict also means **no replacement characters**. An invalid sequence is an error carrying the byte
offset; substituting `U+FFFD` would hand a caller a body that differs from what the server sent,
undetectably.

**The gap was found twice, independently.** Before HC11 there was no `body_text`, and two people —
the author of the first consumer and an outside reviewer writing their own client — each looked for
the obvious method, did not find it, and copied the same manual
`Char::from_u32(body[i] as UInt32)` loop out of an existing consumer. That loop is Latin-1: it
treats each byte as a code point, so `é` returns as two garbage characters. Fine for ASCII, silently
wrong otherwise. Two people reaching the same wrong idiom is the argument the helper had to exist.

### Independent corroboration of HC10, recorded but NOT gate evidence

An outside reviewer built their own client at the HC10 HEAD and ran it against real hosts:
`GET https://api.github.com/rate_limit` returned 200 over TLS validated against the **system** trust
store, headers reaching the server and response headers parsed back. That covers the one direction
the offline tests cannot — `SystemRoots` is tested here NEGATIVELY, since the fixture CA is in no
machine's store. HC13 forbids qualification depending on internet services, so it stays
corroboration.

### A coherence hazard, noted not exploited further

STARK permits an inherent `impl` on a FOREIGN type — verified, and that is how `HttpResponse::json`
is declared from the client package. Rust forbids it. Nothing stops two packages adding a `json`
method to the same foreign type and colliding. Harmless today, and it is what let the roadmap's
frozen call shape be matched exactly, but it is a real gap in the orphan rule.

### DEV-158 hit a SECOND time, and that is the finding

`RequestBuilder::json` did `out.body = body` — assigning over a `Vec<UInt8>`, a drop unit. Green
under the interpreter, aborted natively. Same workaround: build the struct as a literal from moved
fields.

**Two workarounds for one defect in one work package, both caught only by a native run.** The
three-engine divergence means the cheap engine cannot be trusted to find it, and every future
package writing `x.field = <owned value>` is exposed. This is the argument for prioritising the fix
recorded in CD-366.

### DEV-159 — a native build can race its own dependency build

Reported by the same outside reviewer: a first native build of an HTTPS program FAILED and succeeded
on retry, the generated crate having raced its `aws-lc-rs` dependency build. A user hitting this
sees a confusing failure. At minimum the diagnostic should say to retry; better, the build should
not race.

### Evidence

29 `stark-http-core` tests (10 new, the decoder's), 29 `stark-http-client` tests (8 new), and a
twelfth consumer case that encodes a value containing a four-byte scalar, POSTs it over verified
TLS, and re-encodes what comes back. Comparing RE-ENCODED values rather than destructuring is
deliberate: it exercises decode and encode together, so a decoder and an encoder wrong in the same
direction cannot agree their way past it.

## CD-366 — HC10 CLOSED: HTTPS from the URL alone; DEV-158 found (2026-08-03)

**`Client::send` now selects HTTP or HTTPS from the scheme, and there is no other way to ask.** No
per-request TLS switch, no insecure flag, no route to `https://` without certificate and hostname
verification. Full record: `STARKLANG/docs/http-client/HC10-HTTPS-EVIDENCE.md`.

`SystemRoots` is implemented (`rustls-native-certs` 0.8.2) and is `default_config()`'s policy —
CD-361's point delivered: the platform's trust anchors WITHOUT handing the protocol to a platform
TLS stack. `BundledRoots` stays refused; vendoring a CA list is a distribution decision nobody has
taken, and falling back to the system store would give a caller the opposite of what they asked for.

### DEV-158 — assigning over a drop-unit field aborts natively (OPEN)

```stark
enum Policy { None, Explicit(String) }
struct Config { policy: Policy, tag: UInt32 }

fn with_roots(pem: String) -> Config {
    let mut config = base();               // base() yields Policy::None
    config.policy = Policy::Explicit(pem);
    config                                 // aborts here
}
```

```text
generated-code invariant violated: mutable access to a dead slot: the slot is PARTIAL
```

**Cause — and note this is NOT `drop_field_with`, which was the first guess.** `lower_overwriting_assign`
(`mir/lower.rs`) implements CD-012's rule that the new value installs *before* the old is destroyed:

```text
1. save each covered drop unit into a temp   Assign(tmp, Move(unit_place))   <- slot -> PARTIAL
2. install the new value                     Assign(place, rhs)
3. drop the saved temps, flag-guarded
4. set the covered units' drop flags true
```

Step 1's move-out is what marks the slot `Partial`, via `move_field`. Step 2 writes the field back.
But **no operation returns a slot from `Partial` to `Whole`**: the API has `write`, `reinit`,
`take`, `drop_value`, `move_field`, `drop_field_with` and `finish_partial`, and the last goes to
`Dead`. The slot stays `Partial`, and the next whole-struct use hits the guard.

**Why it is not a one-line fix.** A slot may return to `Whole` only when EVERY drop unit is live.
Writing back the unit this assignment covers does not establish that: a SIBLING unit may have been
moved out earlier, in which case the slot is legitimately still partial. Per-unit liveness lives in
MIR's drop flags rather than in the slot, and `slot.rs`'s own docs record the owner review that
caught those two being conflated — the three-state design is that repair. A naive `restore_whole()`
reintroduces exactly the unsoundness it exists to prevent.

**The candidate fix, for whoever takes this.** MIR already holds per-unit liveness as ordinary
locals, so the backend CAN see it: after step 4, emit a `mark_whole()` guarded by the conjunction of
all of the local's drop flags. That is sound on MIR's own record of liveness rather than on a guess,
and it needs no cross-block analysis — the whole sequence is emitted by one function. What it does
need is a new runtime operation, emission for it, tests, and a soundness review. That is a compiler
work package, not an HC10 edit, which is why it is filed rather than patched.

**Bisected to one shape:**

| | native |
| --- | --- |
| assign over a NON-drop field (`config.tag = 9u32`) | fine |
| build the whole struct as ONE literal | fine |
| assign over a DROP-UNIT field, then use the struct | **aborts** |

**The worst property: the interpreter accepts the same program.** `stark test` and `stark run` are
green and only the native build fails, at runtime, as an abort. Any package writing
`config.field = <owned value>` over a pre-existing struct is exposed.

HC10's workaround is one struct literal instead of a field assignment — same semantics, same API,
recorded inline. Remove it when this closes.

### A language question, raised not resolved

Core v1 has no mutable binding of an enum payload in a pattern. So
`enum Transport { Plain(TcpStream), Secure(TlsStream) }` cannot carry a `&mut self` method —
`E0400 mutable method receiver requires a mutable place` — and with no trait objects and no
closures either, the plain and secure request flows are written out TWICE. That is a **language**
decision for the owner, not a defect, and the duplication is deliberate and commented rather than
hidden behind something that looks abstract and is not.

### Two process notes

**An experiment run only under the interpreter proves only the interpreter.** The enum-payload move
that DEV-158 eventually broke on was validated with `stark run` early in HC9 and never natively,
which is why it surfaced three layers later in an HTTPS build rather than in a 25-line probe.

**A stale copy of the harness cost real time.** Several minutes of chasing a hostname mismatch ended
at a `/tmp` snapshot of `qualify-first-party-packages.py` taken before the fixture change. The code
under test had been correct the whole time. Regenerate the filtered copy, or run the real script.

## CD-365 — HC9 CLOSED: verified TLS, and CD-360's rule found in a fourth place (2026-08-03)

**A STARK program can now establish a verified TLS 1.2/1.3 stream over a `stark-net` TCP connection
and release both layers exactly once, without touching a raw ABI symbol.** rustls 0.23.43 over
aws-lc-rs 1.17.3, Profile N, exactly the versions CD-361 observed.

Full record: `STARKLANG/docs/http-client/HC9-TLS-EVIDENCE.md`.

### CD-360's rule had a FOURTH site, and it was the verifier

CD-360 found the transfer-ownership rule implemented in three places and fixed each separately. The
MIR verifier was a fourth. It stayed hidden because CD-360's fixture built its
`ValidatedProviderCall` by hand and emitted from it — never running the verifier over a transfer.
HC9's first native build:

```text
MIR-0005 stark_tls::connect bb53: call argument:
  expected HostResource(… provider: "stark-std-tls", resource: "tcp_stream"),
  found    HostResource(… provider: "stark-std-net", resource: "tcp_stream")
```

**The planner was right and the verifier was wrong** — a correct program refused by the compiler,
which is the worse of the two ways to be inconsistent. The rule now lives in ONE function,
`mir::provider_sig::owner_of`, which both callers use. A fifth site cannot restate it slightly
differently, and a test asserts the planner's actual type and the verifier's expected type are the
same value rather than each being separately plausible.

**The lesson is about the fixture, not the code.** A hand-built `ValidatedProviderCall` skips every
stage between planning and emission. Three sites were fixed, the ruling was recorded as implemented,
and the first real caller found the fourth immediately.

### A package can now NAME another package's resource

The gap CD-360 did not reach: the derived signature for `stark_tls_stream_connect` takes a
`TcpStream`, which is `stark-net`'s nominal, so derivation failed with
`UnboundResourceInSignature`. A transfer was declarable in a *provider* manifest and not in a
*package* one.

```json
"foreign_resources": { "tcp_stream": { "package": "stark_net", "nominal": "TcpStream" } }
```

Resolves to `stark_net::TcpStream` and **synthesizes nothing**. Binding it as an ordinary resource
instead would generate a SECOND `enum TcpStream {}` — a distinct `ItemId`, the same spelling, and a
handle the program could not pass anywhere. Inferring the owner from the graph would make a typo
resolve to nothing far from its cause. So it is declared, names the owner, and is refused if the
alias is not a dependency, if the resource is also owned, or if it is a Core type.

### How the socket physically crosses

CD-360 conveyed ownership but not the object: a `RawResourceHandle` indexes the OWNER's private
table. `stark_provider_abi::RawOsHandle` now documents a detach convention —
`stark_<resource>_detach(handle, *mut RawOsHandle)` — resolved **by the linker**, since every
provider is statically linked into one binary. No Cargo edge, no path assumption, and deliberately
NOT in the provider manifest: a manifest describes the STARK-callable surface, and `detach` is
callable by no package and emitted by no lowering.

**Open, recorded rather than rediscovered:** a missing detach symbol is a LINK error naming a
symbol, not a compiler diagnostic.

### The ordering inside `connect` is the cleanup story

```text
detach the socket FIRST  ->  validate  ->  handshake
```

The handle is consumed whatever the function returns, so any early return before the socket is
adopted strands it in the net provider's table. Detaching first makes every later error path a plain
Rust drop. There is no cleanup code in that function, and its absence is the design.

### Evidence

19 provider tests (the full certificate matrix, both protocol versions distinguishable, handshake
deadline, peer-close, fragmented records, leak-freedom on every failure path), 16 new compiler tests,
8 package tests, and `stark-tls` as the **16th package** in the qualification gate — declared surface
14 callables, all called. All provider-related starkc suites re-run green.

**CD-360's runtime proving case is closed by this**: a real transfer against a live peer, both
outcomes, release observed exactly once.

### DEV-156 — `stark fmt` evicts member doc comments (OPEN)

A doc comment on a struct FIELD is relocated to after the struct; one on an `impl` METHOD is
relocated INSIDE the body. Reproducer:

```stark
pub struct Config {
    pub first: UInt32,
    /// PROBE DOC
    pub last: UInt32,
}
```

becomes `pub struct Config { pub first: UInt32, pub last: UInt32 }` followed by a dangling
`/// PROBE DOC`.

Cause: `printer::field_def` never consumes leading comments, so they survive only via
`CommentStream::take_rest`'s no-loss net, which flushes at the next position the printer does
attach. `item_seq` calls `emit_leading_comments` correctly, which is why top-level items are fine.
Fixing it needs `measure_flat` to snapshot the comment cursor, a member comment to force the
multi-line branch, and per-member emission in that branch.

**Both forms are idempotent after one pass, so `fmt --check` passes and the gate never noticed.**
`stark-net` has its method commentary inside method bodies — almost certainly this defect, absorbed
rather than reported.

Not fixed under HC9: it changes canonical form repo-wide, so every affected package must be
reformatted in the same commit, and this checkout is shared. `stark-tls` uses the surviving
placement with an inline note pointing at this entry.

**On reducing it:** three attempts reported "PRESERVED" falsely, because the baseline copy was kept
INSIDE the package directory and `stark fmt` formats every `.stark` file in a package — mangling the
baseline identically and emptying the diff. A formatter reducer must keep its baseline outside.

### Two other findings

* **DEV-157** — the native backend has no representation for `MirTy::Never`, so
  `Err(_) => panic(..)` in match-arm VALUE position checks and then fails to build. Known C5.3 gap;
  `stark-tls-consumer` nests instead, as `stark-net-resource-consumer` already does.
* `c788_resource_lifecycle::build_driver_selects_closes_for_bound_resource_nominals` fails in this
  checkout with "Cargo succeeded but the expected binary is missing". **Verified pre-existing on
  HEAD** by stashing every HC9 change. Environmental, tied to the shared `target/`.

### Not claimed

`SystemRoots`/`BundledRoots` are declared and REFUSED — HC10's, and refused by name rather than
silently substituted. Profile F is not qualified: it needs CMake and Go, neither present. HTTPS is
HC10.

## CD-364 — `crate_location` deleted; P0.2 complete (2026-08-03)

**The last piece of the mechanism that made every native capability a compiler-source change is
gone.** A provider's crate location now comes from its manifest, resolved against a root the caller
supplies — the compiler's own root for a built-in, the manifest's directory for an external one.

```rust
// before: a hardcoded match over five names
crate_location("stark-net-native", repo_root) -> repo_root/stark-net/native

// after: the manifest says
repo_root.join(&provider.crate_path)
```

`crate_path` is constrained at parse time to be relative and free of `..`, so the join cannot escape
the root. For an external provider that root is the only containment there is.

**`built_in_crate_location` is not `crate_location` returning under a new name.** It is a lookup
OVER the manifests, so the path data still lives in exactly one place and the function cannot
disagree with it. Adding a provider means adding a manifest and nothing else. Built-in only, by
design: an external provider's root comes from the application's declaration, which is what makes it
containable.

### P0.2 exit criteria

| | |
| --- | --- |
| a provider supplied outside the compiler repo is discovered, validated, linked | DONE |
| `first_party()` expressed the same way an external provider is | DONE (CD-362) |
| ABI mismatch, unsupported target, duplicate capability, missing checksum each refused by name | DONE |
| a provider not enabled in the application manifest cannot be activated by a dependency | DONE |
| release builds record provider hashes | `AdmittedProvider` carries identity, version and hash; **wiring into build metadata remains** |

### Verification

`cargo clippy --all-targets` clean (0 warnings), `cargo fmt --check` clean, the four P0.2/CD-360
suites green (51 tests), and the 15-package gate green — including native builds and live-peer
resource lifecycles, which is the evidence that matters, since it exercises the new location path
end to end.

### Two process notes worth keeping

**Clippy earned its place three times in this stretch alone** — `derivable_impls`, then two rounds of
`crate_location` callers that `cargo build` and targeted `cargo test` never compiled. `--all-targets`
is the only local command that compiles what CI compiles.

**Twice I let a partial signal stand in for a complete one.** A fix loop that grepped for ONE error
kind reported "no more sites" when the build had failed for a different reason; and I chased
`crate_location` callers one clippy run at a time — three four-minute runs to find five callers that
`grep -rn` listed in one second. The compiler's output is deliberately truncated; the tree is not.
Ask the source directly.

## CD-363 — P0.2 external provider discovery, trust tiers, and `crate_path` containment (2026-08-03)

### The crate-location ruling

> **A provider's manifest declares its crate path, resolved against a root the caller supplies —
> the compiler's own root for a built-in, the manifest's directory for an external one.**
> `crate_location()` is deleted.

One RULE, two roots. The alternative considered and rejected was keeping a layout convention for
built-ins: that is the `first_party()` shape again — a hardcoded path surviving beside a declared
one — merely moved rather than removed. The root differs by how the provider was ADMITTED, which is
already a first-class distinction (`ProviderTrust`), so it is a visible parameter rather than a
hidden special case.

Neutral on an existing fragility, not worse: built-in `crate_path` values are repo-layout relative,
which is exactly what `crate_location`'s match arms already assumed. Moving it from Rust to JSON
makes it visible and fixable without a compiler change — worth something given a stale install
layout has dropped a provider before.

### `crate_path` containment — a gap found while implementing, not while designing

Nothing in the ruling as chosen constrained `crate_path` to be relative. An external manifest is
written by a third party BY DEFINITION, and `"crate_path": "/etc"` or `"../../elsewhere"` would
escape the root it was admitted under — **the only containment this mechanism has.**

Now refused, and stricter than the obvious form:

* enforced at BOTH the parse and the resolution entry point, so neither is a route around the other;
* checked on the STRING, not the joined path — `provider/../../elsewhere` normalises into something
  that looks contained, so canonicalising first is how the check gets defeated, and a symlink beats
  post-hoc canonicalisation anyway. Refusing the components does not depend on the filesystem's
  cooperation;
* Windows drive prefixes refused on every host, since a manifest may have been written elsewhere.

### Trust is explicit, not enforced

```text
pure STARK package             no native code, no provider
first-party native provider    ships with the compiler, versioned with it
approved third-party provider  declared by the APPLICATION, pinned by version AND checksum
untrusted / local provider     path-based, development only, never in a release build
```

**No sandboxing is attempted** — a partial isolation story invites misplaced confidence, whereas a
visible tier is honest and achievable now. What the mechanism guarantees is that native third-party
code cannot enter a build BY ACCIDENT: every route in is deliberate, recorded, pinned and refusable.

Four properties, all refusal-tested:

1. **off by default** — declaring a provider is not enough;
2. **no transitive activation** — only the application may activate one. A library must not pull
   native code into a program that never asked for it, which is the difference between a dependency
   graph and an attack surface;
3. **pinned exactly** — version and checksum both, or the provider on disk is not the provider that
   was approved. Both hashes are reported so the reader can tell which artefact moved;
4. **development trust does not survive release** — an unpinned path provider works while developing
   and is refused in a release build.

Every failing provider is reported, not just the first: an application pinning three wrongly should
learn all three in one build.

### Evidence

32 tests across `p02_provider_manifest.rs` (11) and `p02_external_provider_trust.rs` (21). The
15-package gate is green through the manifest path, including native builds and live-peer resource
lifecycles.

### Still open in P0.2

Wiring discovery into `native_build.rs` and deleting `crate_location`, which has four real callers.
Deliberately not sprinted: that path produced two red CI runs this session, and it is the wrong
place for blind edits. The discovery surface exists and is tested; the old path still works; nothing
is half-rewired.

## CD-361 — joint HC9/CRYPTO0 decision: rustls + aws-lc-rs (2026-08-03)

> **Select `rustls` with `aws-lc-rs` as STARK's TLS and general native-cryptography foundation.
> Reject `native-tls` for the first-party TLS provider.**

Recorded in `WP-CRYPTO0-TLS-BACKEND.md` — which also CREATES the CRYPTO0 record, since none
existed. HC9's roadmap section is updated at source: **backend selection is no longer part of the
HC9 estimate.**

### Why not native-tls

It is not one TLS implementation — SChannel, Secure Transport and OpenSSL by platform. That would
give STARK three error surfaces, three certificate behaviours, three security policies and three
FIPS stories, and it is directly contrary to what this track has spent its effort on: one rule every
engine satisfies by construction. It also multiplies CD-347/348's obligation, since lifecycle
evidence would be needed per platform stack rather than once. Permitted later as an external
provider under WP-EXTERNAL-PROVIDERS; not as the first-party implementation.

### The sharpest point in the ruling

```text
trust-anchor source  ≠  TLS implementation
```

System roots can be used without handing the protocol to a platform stack. That defuses the only
strong argument for native-tls, and it mirrors a separation this codebase already makes —
`crate_location`'s doc: a crate's path is a property of the checkout, its name a property of the
program. HC9's fixture uses `ExplicitRoots` with a test CA; `SystemRoots` is HC10's concern.

### Verified before freezing, not carried over

The external claims were fetched and checked rather than transcribed. **The ruling held up**, with
two refinements:

| claim | result |
| --- | --- |
| FIPS 140-3 certificate **#4816**, AWS-LC-backed | confirmed exactly |
| rustls 0.23.42 | documentation had already moved to **0.23.43** |
| aws-lc-rs 1.17.x | confirmed, 1.17.3, released 2026-07-17 |
| normal build needs a C/C++ compiler; FIPS adds CMake and Go | confirmed — CMake/Go/bindgen are *never* needed for Profile N |
| a Cargo feature alone is not a FIPS claim | confirmed, and **more specific** than stated |

The version drifting between the ruling and the check, within one day, is itself the argument for
the pin-exactly policy. Recorded as versions OBSERVED; the pin comes from HC9's qualification
output, because you pin what you qualified.

**Profile F is a two-step activation, not a flag:** install `default_fips_provider().install_default()`
and verify `ClientConfig::fips()` at runtime. Both are checkable, so they belong in Profile F's
qualification criteria rather than in prose.

**A correction to my own objection:** I had called the build cost understated. Verification showed
the ruling's split was accurate — Profile N needs only a C/C++ compiler. The residual point stands
but is smaller: providers link statically into the generated workspace, so that compiler is required
of every user building a TLS program, not only of the provider's authors. Recorded as a named cost.

### Two things recorded so they are not rediscovered

* A provider manifest's `targets` field declares triples but **cannot express toolchain
  prerequisites**, so a provider may declare a target it cannot build on without extra tooling.
  Belongs to WP-EXTERNAL-PROVIDERS.
* `stark-http-client::parse_http_url` refuses `https://` outright today, deliberately. **HC10 turns
  that refusal into scheme dispatch** — the visible edge of this decision in already-shipped code.

## CD-360 — cross-provider transfer ruled and implemented; P0.1 closed (2026-08-03)

**Ruling, from the language owner:**

> A cross-provider `HandleConsumed` transfer consumes the source handle regardless of whether the
> provider operation succeeds or fails. Failure does not restore the source resource. The consuming
> provider is responsible for releasing any underlying native resource when it fails before
> producing the destination handle.

`HandleConsumed<T>` therefore keeps the meaning it has always had — ownership leaves the caller
unconditionally — which is precisely why this needed **no change to drop elaboration**, no
branch-dependent move state, and no place live on one result arm and dead on another. Option B
would have required conditional move restoration across provider boundaries; that is ownership
machinery, and it is not justified by making failed handshakes recoverable. It remains available as
a future extension.

Recorded in `native-provider-abi-v0.1-CD360-amendment-2.md`.

### Three enforcement sites, not one

The packet predicted a validator amendment. Implementation found the rule enforced in **three**
places, and only reading the first two would have shipped a P0 that could not lower:

| site | what it checked | change |
| --- | --- | --- |
| `provider_abi::validate` | a provider may only name resource types it declares | foreign types nameable in `HandleConsumed` position only, carrying no close obligation |
| `ProviderSet::select` | (nothing — could not see across providers) | a foreign consumption resolves to EXACTLY ONE owner, and to the owner the consumer named |
| `provider_bind` planner | handle type id and MirTy derived from the CALLING provider | for a transferred handle both come from the OWNER |

The third is the one that mattered. `mir/lower.rs`'s `HandleConsumed` arm already carried a comment
stating CD-360's rule verbatim — written for A11 §8, long before the question was asked — so the
move semantics and drop behaviour genuinely were already correct. **But the call could not be
planned at all**: `UndeclaredResourceType`. Nothing had ever lowered a transfer.

A handle carries its OWNER's type id, because it was created with it, and the consuming provider
must present it unchanged. Deriving it from the consumer would hand the provider a tag naming a
different resource. `ValidatedProviderCall` now carries `ForeignResourceCall` for that reason.

### Why the declaration is explicit

`foreign_resources` is declared, not inferred. Treating "any handle type I did not declare" as
foreign would silently accept `HandleConsumed { resource_type: "tcp_strem" }` and defer the typo to
a link failure. Naming the owning provider keeps the check at the three-part identity
`{nominal, provider, resource}` the type system already uses — which is also why
`ForeignResourceOwnerMismatch` exists: a matching resource NAME under a different owner is a
DIFFERENT resource.

### Evidence

19 tests. `cd360_cross_provider_transfer.rs` — 11 declaration rules (2 allowed, 9 refused) and 4
resolution rules; `cd360_transfer_lowering.rs` — 4 lowering assertions on a synthetic net→wrap
transfer, deliberately not TLS, so the proving case does not wait on a certificate chain.

**The fixture earned its keep twice.** It caught the planner refusal, and it caught a bad assertion
of my own: the first double-release check grepped the whole generated file and failed on the
`extern "C"` declaration rather than a call. That form would have passed for the wrong reason had
the code been broken differently — a declaration is not an invocation, and the test now cuts the
extern block before checking the body.

All ten provider suites re-run green (132 tests).

### What P0.1 does NOT include

The **runtime** proving case — a transfer executed against a live peer, both outcomes, release
observed exactly once — remains open and belongs with HC9, since it needs a TLS peer with a
controlled certificate chain. §3 of the amendment (a failing provider must leave no live native
resource) is a provider-author obligation **no compiler check can enforce**; it is recorded so
review can carry it.

**P0.2 (external provider discovery) is now the critical path.**

## CD-359 — HC9 paused; two P0 platform-architecture packets opened (2026-08-03)

**Two items previously carried as backlog are release-architecture blockers, and HC9 must not be
implemented before the first is frozen.** Recorded by the language owner; packets written to scope,
deliberately NOT combined.

### Revised priority

```text
P0  Cross-provider resource-transfer ABI      WP-PROVIDER-HANDLE-TRANSFER.md
P0  External provider discovery/registration  WP-EXTERNAL-PROVIDERS.md
P1  HC9 TLS implementation                    DESIGN-BLOCKED by P0.1
P1  Database provider foundation              blocked by P0.2
P1  HC10 HTTPS                                blocked by HC9
P2  HC11-HC13
```

The two design tracks may run in parallel. **DB0 (STARK-facing value, error, connection, transaction
and cursor contracts) may proceed now** — it is pure STARK and does not prejudge either decision.

### Why HC9 stops

TLS wraps TCP, so the TLS provider must take a `TcpStream` the net provider created. The ABI has no
way to express that, and without a frozen rule an implementation would duplicate ownership, smuggle
raw handles, bypass the validator, fuse TCP and TLS into one provider, or leave Drop authority
unclear. Each weakens the resource model A11/CD-234/CD-237/CD-240 exist to guarantee.

### The scope finding that shrinks P0.1

Probing `provider_abi::validate` established that **most of the transfer contract already exists**:

| already true | consequence |
| --- | --- |
| resource identity is structural over `{nominal, provider, resource}` | provider identity is part of the TYPE; a transfer is a genuine type change |
| `HandleOut` writes its slot only on success | the destination's failure disposition is settled |
| close is selected per resource, and a closeless resource is refused | "which provider releases" is answered structurally |
| every function returns `ProviderStatus`, no direct returns | `Result<HandleOut<TlsStream>, TlsError>` is the shape it already has |

So the packet does not design a mechanism. It authorizes **one referencing rule** — a provider may
name a foreign resource type in `HandleConsumed` position without inheriting its close — and freezes
**one failure rule**.

The two existing refusals are CORRECT and must survive; the new rule sits alongside them:

```text
ResourceTypeMissingClose      declaring a foreign type would give it a second, competing close
HandleResourceTypeUndeclared  a provider may only reference types it declares
```

### The hard question, and the recommendation

What happens to the SOURCE handle when a transfer fails. Three candidates are set out in the packet;
the recommendation is **(A) failure also consumes the source**, because it is the only option
requiring **no change to drop elaboration** — `HandleConsumed` keeps meaning exactly what it means
today, unconditionally consumed. Returning ownership on failure would make ownership depend on a
runtime value, which is precisely the class of conditional invariant this compiler has repeatedly
failed to get right first time. It also states the real-world truth: a failed handshake does not
leave a usable socket.

### Why P0.2 is broader than databases

`first_party()` is a hardcoded `Vec` and `crate_location` a hardcoded `match`. Providers are
compiler-integrated extensions, not an ecosystem mechanism: every native capability needs a compiler
change, nobody outside the repo can publish one, provider versioning is welded to compiler releases,
and trust policy is implicit because we wrote everything that exists. **The public package system is
incomplete for host capabilities.**

The packet keeps static linking and changes only DISCOVERY — manifests instead of hardcoded tables,
with `provider_abi::validate` unchanged and merely fed from a different source. Trust is made
explicit rather than enforced: four tiers, external providers off by default, no transitive
activation, exact version and checksum, no sandboxing attempted.

Its exit criterion is an executable claim:

> Adding PostgreSQL, MongoDB, MySQL or SQL Server requires no compiler-source change.

## CD-358 — the file-provenance audit, and borrow conflicts made place-granular (2026-08-03)

Two items from the post-CD-357 list, plus a CI failure that CD-357 caused and this fixes.

### 1. The provenance audit closed the class by EXERCISE, not by inspection

`self.text(span)` slices the file currently being CHECKED. A name belonging to a DECLARATION —
an impl's generic parameter, a signature's, a trait default's return type — belongs to the file
that declared it. Across a module boundary those differ, and the failure is **silent**: the
comparison succeeds against garbage.

The same bug has now been repaired at six sites across four decisions:

| | site | found by |
| --- | --- | --- |
| DEV-069 | a trait method's name | a trait default across files |
| DEV-101 | cross-package generic typecheck | a package consumer |
| DEV-148 | an associated function's name, then its generic parameters | `stark-url` calling its own `Url::parse` |
| **DEV-155** | a METHOD's impl generics, and a trait default's signature TYPES | **this audit** |

**Inspection was the wrong tool and had already failed four times.** There are ~90 `self.text`
calls in `typecheck.rs`, most legitimately reading the file under check. Classifying them by eye is
exactly the process that missed this repeatedly. A probe that actually compiles two-file packages
found the remaining live site in ONE run:

```text
*w.get() != 11   ->  E0001 expected 'S', found an integer literal
```

`'S'` is `T`'s offset in `lib.stark` landing on an `S` in `inner.stark`.

The repair is a `decl_text` helper that resolves against `foreign_sig_item` when a declaring item is
in scope — a helper rather than a habit, precisely because remembering `item_text` at 29 sites is
what has not worked. `tests/cd358_cross_module_provenance.rs` drives every construct across a module
boundary, so a future site added without it fails there rather than in a package months later.

**The near miss worth recording:** `item_text` returns `"?"` for an out-of-range span, so two
mis-sliced parameter names could COLLIDE on one key and substitute each other's types — a WRONG
program rather than a rejected one. Every failure seen so far was a refusal; that one would not
have been. A two-parameter test pins it.

**Also answered:** associated TYPES resolve correctly across a module boundary — the open question
DEV-148 left behind.

### 2. DEV-154: borrow conflicts compare PLACES

OWN-BORROW-001 has always said "Disjoint field projections do not overlap". Every comparison in
`borrowck` tested `b.local == local`, so a borrow of `p.a` blocked a read of `p.b`. The `Borrow`
record now carries the borrowed place, and every comparison — creation, assignment, move, method
receiver, read — goes through `places_overlap`, field-precise since DEV-135.

**This repair makes the checker accept more, so the refusals are the load-bearing half.** Identity,
parent-over-child, whole-local-over-field, two exclusive borrows of one field, assignment to a
borrowed place, and move-out-of-borrowed-storage all stay refused. The move check is deliberately
stricter than the read check — it rejects under ANY live borrow, shared included, because moving
invalidates storage a live view still points into — and going place-granular did not weaken it.

### 3. CD-357 broke the AST snapshots, and blessing them would have hidden it

CI went red on `tests/snapshots`. Inserting OWN-BORROW-002's example as `03-Type-System__19`
shifted every later fixture by one, and the snapshot cases name fixtures **by number** — so
`__20`, `__31`, `__37`, `__40` silently came to mean different constructs.

`UPDATE_SNAPSHOTS=1` would have gone green while repointing each snapshot at a different construct.
The cases were RENUMBERED to follow their content instead, and the `.ast` files renamed with them —
**the snapshot contents did not change**, which is the proof the mapping is right. A comment on
`CASES` now records that renumbering, not re-blessing, is the correct response.

The extractor's "manifest is in sync" check covers the manifest only; the snapshots are a second
artefact keyed to the same numbering, with no such check. That gap is real and remains open.

**Verification:** 15/15 packages qualify, external sample suite 39/39, and the three new suites
(8 provenance + 10 place-granular + the CD-357 15) are green. Full workspace coverage is CI's.

## CD-357 — DEV-150 ruled: uniform rejection, hoisting required (2026-08-02)

**Ruling (B), from the language owner. Now normative as OWN-BORROW-002 in `03-Type-System.md`:**

> A call may not create an exclusive borrow of a place while another argument in the same call
> reads from or borrows an overlapping place. Such reads must be evaluated into locals before the
> exclusive borrow is created.

```stark
fill(&mut buffer, buffer.len());   // rejected
let count = buffer.len();          // hoist
fill(&mut buffer, count);          // accepted
```

Uniform in the base — a local, a place reached through `&mut`, a field projection, an index, a free
function or a method receiver — and independent of argument order. **Core v1 therefore does not
define argument evaluation as providing two-phase borrow semantics**, and says so; adopting them
stays reserved and this ruling stays reversible.

Chosen over blessing the accepted case because that would have required accepting the LOCAL case
too — widening the borrow rule into two-phase borrows, with evaluation-order machinery and a real
semantics commitment. (B) keeps one backend-neutral rule every engine satisfies by construction.

### What had to change

The rule already existed and already fired for a local base. It stopped one indirection away:
passing a `&mut`-typed place REBORROWS, which registers no active borrow, so the read that followed
saw nothing to conflict with.

`check_argument_overlap` now runs as its own pass over the whole argument list — **a method
receiver included**, since `v.push(v.len())` is the same conflict as `push(&mut v, len(&v))` —
BEFORE the left-to-right walk. It has to be a separate pass: a check that falls out of the walk can
only ever catch the borrow-first order, and the ruling is order-independent. `exclusive_borrow_of`
treats an explicit `&mut place` and a `&mut`-typed place alike, which is the whole repair. A
report-once set keeps one mistake to one diagnostic, rather than the new check and the old one both
reporting the same read in different words.

### Livability, checked rather than assumed

**All 15 first-party packages pass the gate under the rule with zero new diagnostics.** The only
site that ever hit it was `stark-http-parser`'s four `take_line` calls, hoisted when the defect was
first found. A rule that had broken working code across the tree would have been the wrong rule to
implement without saying so.

### Engine agreement is by construction

The front end rejects, so nothing reaches the HIR oracle or MIR. `check`, `run` and `build` all
refuse the previously-accepted program with the same diagnostic — which is the point: the old
behaviour was accepted by the checker, executed correctly by the oracle, and refused by rustc.

### Evidence

`tests/dev150_argument_overlap.rs` — 15 tests, negatives varying the base and the order, positives
for every hoisted and non-overlapping form (different locals, literals, successive borrows,
successive reborrows of a parameter, two shared reads). Plus spec fixture
`03-Type-System__19.stark`, classified `semantic-error` with `errors = "E0101"`, so **the spec's own
example is an executable test of the rule.**

Supersedes `dev150_argument_conflict_through_reference.rs`, which pinned the INCONSISTENCY while the
ruling was open. Its own doc required it to be rewritten around whichever ruling landed, and both of
its "the two bases disagree" tests went red the moment they agreed — the mechanism working as
designed, twice in two commits now.

### One defect uncovered on the way: DEV-154

CD-357's overlap check is place-granular and correctly declined to fire on `f(&mut p.a, p.b)`. The
OLDER `check_read_borrow_conflict` then reported it anyway, because it compares only the LOCAL and
ignores projections — so **disjoint field projections over-reject, contradicting OWN-BORROW-001's
"Disjoint field projections do not overlap".** Pre-existing; visible only because two checks in the
same area now disagree about granularity. Filed OPEN and deliberately NOT bundled here: loosening a
borrow check is its own change with its own negative controls, and must not ride along with a ruling
that tightens one.

## CD-356 — DEV-148 CLOSED: the name was sliced out of the wrong file (2026-08-02)

**Filed as a language limitation about associated functions. It was a text bug, and the gate built
one commit earlier is what forced it into the open.**

`Wrap::make(2)` from a submodule of its own package failed with "associated function 'make' not
found". Path resolution was correct — it reached `Res::AssociatedFn`. `typecheck`'s lookup then
compared member names with `self.text(span)`, which slices **the file currently being checked**,
while a member's name span belongs to the file that declared the `impl`. Instrumented, `impl Wrap`'s
two members read back as:

```text
member name_text="rap:"  has_receiver=false     // `make`'s offsets applied to the other file
member name_text="?"     has_receiver=true      // a span running past the shorter file's end
```

No candidate could ever match. **Methods were unaffected because method lookup selects on the
receiver's TYPE rather than by slicing a name** — and that asymmetry is the whole reason this looked
like a rule about associated functions instead of a bug about files.

### A second site, one layer down

Fixing the comparison made plain associated functions work and immediately exposed the same defect
in generics:

```text
error: [E0500] type 'r' does not satisfy operator trait 'Eq'
```

`'r'` is `T` sliced from the wrong file. The substitution map's keys and the `Ty::Param`s they
substitute into must be read from the SAME file or substitution silently fails to fire, so
`foreign_sig_item` now carries the declaring item across the whole signature conversion. Note also
that `item_text` yields `"?"` for an out-of-range span, so several mis-sliced parameter names could
COLLIDE on one key and substitute each other's types; a two-parameter test pins that they cannot.

### The rule was already written down twice

DEV-069 fixed exactly this for trait methods — "the trait's method names belong to the TRAIT's
declaring file" — and `build_assoc_projections` converts "against the impl's own file". This site
simply missed it. The general statement, worth keeping where someone will read it: **`self.text` is
correct only for spans from the file under check; every lookup that reads a name off a foreign
declaration needs `item_text`.** Worth auditing the remaining `self.text` call sites against that.

### What closing it unblocked, and what it then found

The three items CD-355 recorded as `surface_blocked` became callable, and **the gate refused its own
stale records** — the self-cleaning rule firing for real rather than in principle:

```text
stark-url: these are recorded as blocked, but are now called:
      Url::parse
```

With the records removed and the three exercised, **all 15 packages qualify with zero blocked
items: every public callable in the tree is now called by its package's own tests or consumers.**

**And calling `TcpStream::connect` for the first time found a third dead API.** It refuses EVERY
non-zero timeout with `Unsupported`:

```stark
pub fn connect(address: SocketAddress, timeout: Duration) -> Result<TcpStream, NetworkError> {
    if !timeout.is_zero() { return Err(NetworkError::Unsupported); }
    connect_socket_address(&address)
}
```

So the natural connect API — the one that takes a deadline — has never connected to anything. It
succeeds only for a ZERO duration, which reads as "no timeout" and is the opposite of what passing a
`Duration` means. There is no connect-timeout in the provider ABI to implement it against: the
declaration ran ahead of the capability. Pinned in the consumer as a required failure, so landing a
real connect timeout forces the assertion to change.

That makes **three dead APIs in `stark-net` found by calling things nothing had called** —
`shutdown_write` (permanent stub), `connect`-with-timeout (unsupported for every meaningful
argument), and the timeout setters (unbuildable at a call site, DEV-151). All three concern
timeouts or lifecycle, and all three were invisible for the same reason.

**Evidence:** `tests/dev148_associated_fn_across_modules.rs` — 7 tests over a real two-file package
graph, because a single-file fixture cannot reproduce a provenance bug. Vacuity-checked by reverting
the repair: the three cross-boundary positives go RED, all four controls stay green.

## CD-355 — the gate now requires that a package's declared surface is CALLED (2026-08-02)

**The gap this closes has cost three separate stretches, each time closing the instance and leaving
the class open:**

| | what happened | what was fixed |
| --- | --- | --- |
| CD-345 | `stark-net` passed all seven steps while `connect`/`read`/`write`/`close` had never been called, hiding a build-breaking defect (DEV-146) | that package's consumer |
| CD-347/348 | resource LIFECYCLES made executable, against a live peer | the resource category |
| CD-354 (DEV-151) | the same failure one level in: `set_read_timeout` was declared under CD-346, qualified, documented and **unbuildable at every call site**, because nothing had ever called it | that one method |

Each round fixed an instance. **The class is: the gate proves a package builds and its consumer
runs; it never proved that what a package DECLARES is reached by anything.**

### The check

`qualify-first-party-packages.py` gains a step: every public callable must be CALLED by the
package's own tests or its own consumers. The declared surface comes from `stark doc` — the
compiler's own AST walk — not a regex over `pub fn`, so it cannot drift from the source.

**The bar is the package's OWN evidence**, not "called by something, anywhere in the tree". A
downstream caller can be deleted, and proves nothing about the package in isolation.

**Matching is textual and deliberately biased toward FALSE PASSES.** Comments are stripped first, so
prose never counts as a call; but an alias or a generic dispatch can credit a call that does not
happen. That bias is chosen: a false FAILURE would push someone to add a fake call to satisfy the
gate, which is worse than a missed one, because it teaches that gate output is noise.

### What it found immediately

**12 uncalled public callables across 3 of 15 packages** — and the concentration is the finding:

- `stark-net`: **all seven** `impl TcpStream` methods. The entire method surface was dead
  end-to-end; every consumer used the free functions instead. DEV-151 was one instance of a block
  that had never been called at all. Now exercised by the native resource consumer against the echo
  peer, including the DEV-151 reproducer as a real call site.
- `stark-mime`: four `MediaType` methods, wrapping free functions the tests already covered. A
  wrapper no test calls is not a thinner API — it is a second implementation nobody has run.
- `stark-url`: `Url::parse`.

Also surfaced: **`shutdown_write` is a stub** that always returns `Unsupported`. Calling it is what
made that visible. The consumer now asserts it fails, so implementing it forces the assertion to be
updated rather than letting a permanently-broken promise sit in the surface.

### Blocked items are counted, not waived

Three of the twelve are ASSOCIATED functions and cannot be called at all — DEV-148. They are
recorded per package with the defect that blocks them, and **the gate refuses a record whose item
has become callable**. A fix to DEV-148 therefore forces the records out rather than letting them
rot; the same self-cleaning rule as the sample suite's "an unexpected PASS is a failure". The
purpose is to make the cost of an open defect countable instead of invisible.

### Two compiler defects had to be fixed first

- **DEV-152** — `doc_gen::extract` silently DISCARDED the methods of any `impl` whose type had no
  page-level item. A synthesized resource nominal (CD-234) has none, so all seven `stark-net`
  methods were absent from its documentation. A surface gate built on that extractor would have
  certified the package as fully covered. It also explains part of why nobody called them: the docs
  did not say they existed.
- **DEV-153** — `hir_field_ty` had no arm for an unsized slice, so `owned.write_all(input)` refused
  to lower while `write_all(&mut owned, input)` built. This is **DEV-151's second-order cost**:
  opening method dispatch on a resource receiver routed declared parameter types through that
  conversion for the first time, and met a form it had never had to handle. A repair that widens
  what is reachable will expose whatever the newly reachable path never handled — that is the price
  of the DEV-151 class, not an argument against paying it.

### DEV-148's scope was wrong

Filed as cross-PACKAGE; it is cross-MODULE, which is strictly wider. A submodule of the same
package cannot call `Wrap::make` either, and neither can the fully qualified `super::Wrap::make`.
So a package cannot even TEST its own associated functions. The failure is not in the resolver —
the path reaches `Res::AssociatedFn` — but in `typecheck.rs`'s associated-function lookup. Methods
are unaffected because method lookup goes by the receiver's TYPE rather than by path resolution,
which is exactly why the two diverge.

**Status: 15 packages qualify with the surface check enforcing**, 3 items recorded blocked.

## CD-354 — three compiler defects found by qualifying HC7/HC8; one escalated, not repaired (2026-08-02)

**Writing two packages and running them through the gate found three compiler defects and one
semantics question. None was found by a reproducer; every one was found by executing something that
had never been executed.**

| | what | disposition |
| --- | --- | --- |
| DEV-149 | a `&self` method on a `&mut` base is neither weakened nor reborrowed | FIXED |
| DEV-150 | the argument read-conflict rule does not fire through a reference base | **ESCALATED** |
| DEV-151(a) | a method on a host-resource receiver did not lower | FIXED |
| DEV-151(b) | a written-out `()` lowered to `Tuple([])`, not `Unit` | FIXED |

### DEV-149 is my own DEV-147 repair, narrowed on the wrong axis

DEV-147 taught the four `borrow_*_receiver` sites to reborrow rather than move, then gated the
repair on the mutability the METHOD wants. The gate belongs on the mutability the BASE has:

```stark
fn count(v: &mut Vec<UInt8>) -> UInt64 { v.len() }   // check: OK, run: 1, build: REFUSED
```

Two failures from one omission — MIR-0005 (the `&mut` handed over unweakened) and MIR-0007 (the
caller's reference moved). One reborrow fixes both, because `&*base` from a `&mut` base IS the
weakening. The shape it blocked is "measure a caller's buffer, then modify it", which is what
`stark-http-parser::drop_front` does and what surfaced it.

### DEV-151 is CD-345's lesson one level down

CD-345 found `stark-net` passing all seven gate steps while `connect`/`read`/`write`/`close` had
never been called. CD-347 fixed that by requiring a native consumer to exercise each resource's
lifecycle. **This is the same failure one level in: a declared surface whose CALL SITES were still
unexecuted.**

CD-346 ruled that a resource operation moving a cursor or consuming bytes takes `&mut self`.
`stark-net` declared `set_read_timeout`/`set_write_timeout` as methods on that ruling and qualified.
Lowering refused every method call on a host-resource receiver, so **CD-346's ruling was
unbuildable at every call site** — and nothing learned that, because nothing had called one.
`stark-http-client` was the first caller and failed immediately.

The refusal was a missing match arm, not a missing capability: `HostResourceTy.nominal` already
holds the item the `impl` hangs off. Fixing it then exposed (b) — a written `()` reaching the tuple
arm and producing `Tuple([])` where every synthesized site uses `MirTy::Unit`, so
`fn f() -> Result<(), E>` declared a return type no constructed value could match. `Result<(), E>`
is a very common signature; it took two unexecuted paths crossing to make the divergence reachable.
The structural test now asserts no lowered signature or local carries an empty tuple, which catches
a divergence that has not yet MET a conflicting value.

**What this says about the gate.** Both halves of DEV-151 were reachable only by CALLING a declared
surface natively. The seven steps check that a package builds and its consumer runs; they do not
check that everything the package DECLARES is called by something. That is the next gap of the same
family, and it is not closed by this CD.

### DEV-150 is escalated, not repaired

`f(&mut x, x.field)` is refused for a local base and accepted through a `&mut` parameter; the
interpreter runs the accepted form and the native backend emits Rust that rustc refuses (E0503).
Two defensible rulings:

- **(A)** the checker is right and evaluation should be sequenced — close to Rust's two-phase
  borrows, but it requires the LOCAL case to start being accepted too, so it widens the borrow rule;
- **(B)** the checker is wrong and the rule must fire uniformly — conservative and matches the spec
  as written, but `f(buf, buf.len())` stops compiling.

They disagree about whether a real program is sound, so this is a language decision rather than a
repair-commit decision, and it is left OPEN for the language owner. The test suite pins the
INCONSISTENCY rather than either ruling: whichever lands, the test contradicting it fails and the
entry must be revisited. `stark-http-parser`'s four `take_line` call sites were rewritten to hoist
the read, which is required under either ruling.

**Evidence:** `tests/dev149_shared_receiver_over_mutable_base.rs` (13),
`tests/dev150_argument_conflict_through_reference.rs` (4),
`tests/dev151_resource_method_dispatch.rs` (4). All three include negative controls, because each
repair's own risk is handing out access that was never held.

## CD-353 — HC7 and HC8 delivered and qualified; the gate grows an HTTP peer (2026-08-02)

**`stark-http-parser` (HC7) and `stark-http-client` (HC8), both qualified through the seven-step
gate. All 15 first-party packages now qualify, with both resource-bearing ones observed against
live peers.**

### HC7 — the parser, and the exit criterion that shaped its tests

The roadmap's HC7 exit criterion is "the parser can consume any legal fragmentation pattern without
socket knowledge". That is not a claim a few hand-picked splits support, so the suite parses each
message at EVERY two-part split and requires every result to agree — n-1 boundaries per message,
each landing mid-token somewhere different: inside `HTTP/1.1`, between CR and LF, inside a header
name, inside a chunk size. The consumer does the same rather than parsing one buffer, because a
one-buffer consumer would prove nothing this package is for.

34 tests. Ten states, four framings (fixed, chunked, close-delimited, none), 1xx skipping, HEAD
responses, and the rejection half: bare LF, obs-fold, conflicting `Content-Length`, `Content-Length`
with `Transfer-Encoding`, unsupported codings, malformed chunk sizes and terminators, truncated
bodies, and each limit.

**Two real parser defects, both found by the whole-vs-fragmented differential rather than by any
single case:** the OWS-skip after a header colon used an `n + 1` sentinel that destroyed the index
it had just found, so every header value was mis-sliced; and the `UntilClose` transition returned
before the drain arm could run, dropping every close-delimited body that arrived in one buffer.

### HC8 — the client, and what a capability-bearing package can be tested with

Every useful operation in `stark-http-client` requires a provider, and `stark test` runs on the HIR
interpreter, which has no provider layer. So the split is forced, not chosen:

- **`stark test` (14 tests)** covers what is decidable without a socket — URL targeting, config
  budgets, builders.
- **`stark-http-client-consumer`** is native-only and requires a live HTTP peer. It PANICS without
  one rather than reporting success, per CD-348.

Step 5 (`stark run`) is therefore unreachable for this package. The gate now has an
`interpreter_exempt` flag that skips it with a printed reason — and REFUSES to accept the flag
unless the case also declares resources and a resource consumer, so an exempt package is executed
MORE than an ordinary one, never less. Validated in code rather than left to reviewer discipline,
because CD-345 is the record of what an unexecuted step costs.

**Two real client defects, found by the tests:** URL fragments reached the request target (an
information leak — a fragment is client-side only and must never go on the wire), and an empty or
invalid authority was accepted, including `http://h:/` silently defaulting to port 80 rather than
being reported as the typo it is.

### The HTTP peer

`qualify-first-party-packages.py` grows `http_peer()` beside `echo_peer()`, serving four routes that
each pin a response shape the client must handle differently: `/fixed`, `/chunked`, `/fragmented`
(head and body split across several writes with pauses), and `/close-early`. The last two matter
most — they are what a client that assumes one `recv()` per response, or that treats a short body as
complete, gets wrong. Binding is asserted, never skipped.

Observed natively, end to end: resolve, connect, set timeouts, write, read across fragmentation,
decode chunks, detect an early close, release the stream.

### Also recorded

DEV-148 (a cross-package associated function is unresolvable) was found mid-sprint and is OPEN.
`Type::new()` is simply unavailable to a consumer, so every first-party package exposes free
constructors instead — a convention adopted without anyone recording why, which is how a defect
becomes a house style. `stark-time` gained `duration_seconds`/`duration_millis`/`duration_nanos` as
the forced workaround.

## CD-348 — CD-347's claim was stronger than its evidence; the gate now earns it (2026-08-02)

**CD-347 said the gate requires a consumer to "acquire, use and close each resource". It did not.
The checked-in consumer connected to a port expected to REFUSE, so on the path CI actually took:**

```text
connect fails
  -> no TcpStream acquired
  -> write_all never executed
  -> close never executed
  -> drop-release never executed
```

The program compiled and linked every one of those branches, so it was valid evidence that the
source type-checks, provider calls lower, symbols link, the executable starts, and the failure path
runs. **It was not evidence that an acquired resource is used and released.** The honest claim for
that version was:

> every resource-bearing package ships a native consumer that COMPILES AND LINKS its
> acquire/use/release surface and EXECUTES AT LEAST ONE provider path.

Recorded because the gap between that sentence and CD-347's is precisely the gap CD-345 was about:
a claim satisfied by a path that never calls the product.

**The fix is the peer, not a weaker sentence.** `qualify-first-party-packages.py` now starts a
loopback echo listener before running a resource consumer, so the full lifecycle executes:

```text
acquire -> write -> read -> EXPLICIT close      the affine release the package exposes
acquire -> write -> IMPLICIT drop release       MIR drop elaboration emitting the close
```

**It cannot silently degrade.** If the port cannot be bound, qualification FAILS with a message
saying why, rather than falling back to the failure path — falling back would restore the weaker
claim while still reporting success. And the consumer PANICS if the peer is absent: verified, exit
code 101, so a peerless run can never be mistaken for a pass.

### EXECUTED SURFACE, by package category

The standing rule needed this precision, or a future team satisfies it through an expected error:

| Category | Bar |
| --- | --- |
| pure package | the ordinary consumer executes each principal public behaviour |
| function-shaped provider | the native consumer SUCCESSFULLY invokes each capability family |
| resource-shaped provider | the native consumer SUCCESSFULLY acquires, uses and releases every resource type — BOTH release paths |
| failure-only environment | a deterministic negative path is allowed, but must be LABELLED lowering/linking evidence, never lifecycle evidence |

The fourth row is the important one: it keeps the escape hatch open for environments where success
genuinely cannot be arranged, while making it impossible to use one and call the result lifecycle
evidence.

EVIDENCE: hardened eleven-package gate exit 0 with the peer, `STARK_NET_RESOURCE_OK` observed;
the consumer exits 101 with no peer; echo peer refuses to skip on a bind failure; fmt clean.
FILES: starkc/scripts/qualify-first-party-packages.py, stark-net-resource-consumer/src/main.stark,
COMPILER-STATE.md.
NEXT: unchanged — HC3/HC4 and the OPS resource items are unblocked; HC5/HC6 and the pure fills
never were.

## CD-346 / CD-347 — DEV-146 repaired with its ruling; the gate's surface coverage made executable (2026-08-02)

**The two toll items. Resource-track work (HC3/HC4, OPS stdio/signals/process) unblocks on these;
HC5/HC6 and the pure OPS fills never depended on either and should not have waited.**

### CD-346 — DEV-146, and the layer the first diagnosis got wrong

`weaken_ref_to` was never the problem. Its mutability arm is type-agnostic and would have handled
`HostResource` fine. **Provider calls never reached it**: the `HandleBorrowed` arm of
`lower_provider_call` pushed its operand with no expected-type coercion at all.

DEV-133 routed SIX coercion sites through `weaken_ref_to` and its comment warned that "whichever
site was forgotten would keep this defect". Provider calls were the seventh. It stayed invisible
because no first-party package called a resource function until `stark-net` did — the same
blindness CD-345 found in the gate, one layer down.

**THE RULING, which is what outlives the repair:**

```text
AbiParam::HandleBorrowed   always derives a SHARED reference   (ABI fact, unchanged)
package surface            may declare &mut; the compiler weakens
```

The two need not match, so the surface question is answered by SEMANTICS rather than by the ABI:

- an operation that consumes or produces bytes, or moves a cursor, takes `&mut` — a shared borrow
  would let a caller hold two readers of one stream, making byte-consumption order non-local and
  unreviewable;
- a purely observational operation stays `&`;
- neither choice changes what crosses the ABI.

Settled once, here, rather than re-litigated per package: io v0.2 streams, signals, process
handles and crypto keys all face it. **Recorded caveat:** the ruling was made from what the ABI
verifiably does. The CRYPTO0 convergence was NOT in evidence when it was written and should be
checked against it before the first crypto package declares a surface — if CRYPTO0 says something
narrower, this ruling yields to it.

**Negative control, because the risk is weakening the wrong way.** If `&R` could satisfy a `&mut R`
parameter the repair would hand out exclusive access from a shared borrow — an aliasing hole worse
than the defect. Pinned.

`stark-net`'s `&mut` signatures are restored, with the ruling recorded at their definition.

### CD-347 — the gate's executed-surface requirement

A `PackageCase` now declares the resource types it exposes, and a package that declares any must
ship a NATIVE consumer whose run acquires, uses and closes each one. Missing consumer, missing
directory, or a failing run all fail qualification.

**The split is forced, not chosen.** Step 5 is `stark run`, and the interpreter has no provider
layer — any consumer touching a bound resource dies with "provider binding not lowered". So the
resource exercise cannot live in the ordinary consumer, and the gate runs the resource consumer
without a `stark run` step.

`stark-net-resource-consumer` is the first: acquire+close, acquire+use+close (through the `&mut`
path DEV-146 broke, so the package would not have built before CD-346), and acquire-then-let-drop-
release. Deterministic in CI — it needs no peer, because what it proves is that the resource path
LOWERS, LINKS and EXECUTES, which is exactly what was unobserved.

Verified to bite: removing the resource consumer fails the gate with a message naming CD-345.

### The standing rule for the Codex lane

**Definition of done now includes executed surface, stated in each directive.** CD-344's failure
was not Codex writing wrong code — the behaviour was sound, and the end-to-end run proves it. It
was the lane's evidence standard being satisfiable by a consumer that never called the product.
Every future package directive names its required consumer exercises the way the repair packets
named their must-pass sets. One paragraph per directive; the difference between two lanes having
one discipline or two.

EVIDENCE: `dev146_resource_borrow_weakening` 3 cases; `mir_verify`/`mir_lowering`/
`c788_lifecycle_e2e`/`conformance`/`gate3_execution` 87 green; provider and resource suites
(`a10_provider_call`, `a11_host_resource`, `c786_tcp`, `c788_resource_lifecycle`) green; hardened
eleven-package gate exit 0 with `STARK_NET_RESOURCE_OK` observed; end-to-end native TCP client
`wrote / 5 / closed` against a listener that received `b'PING\n'`; fmt clean.
FILES: starkc/src/mir/lower.rs, starkc/tests/dev146_resource_borrow_weakening.rs (new),
starkc/scripts/qualify-first-party-packages.py, stark-net/src/lib.stark,
stark-net-resource-consumer/ (new), starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: HC3/HC4 and the OPS resource items unblock. HC5/HC6 and the pure OPS fills were never blocked.

## CD-345 — HC1 and HC2 qualified with evidence; HC2 was qualified in name only (2026-08-02)

**HC1 (`stark-url`) and HC2 (`stark-net`) landed as plain commits with no CD entry and no evidence
statement. Both are in the eleven-package gate and both pass it. For HC1 that means what it sounds
like. FOR HC2 IT DID NOT.**

### The finding: a happy-path gate cannot qualify a resource-holding package

`stark-net` is the first first-party package that holds host resources. The seven-step gate ran
`check` / `test` / `fmt` on the package, then `check` / `run` / `build` / execute on its consumer —
and **the consumer only formatted addresses**:

```stark
let address = socket_address(ipv4(127u8, 0u8, 0u8, 1u8), 1u16);
if socket_address_text(&address).as_str() != "127.0.0.1:1" { panic(..) }
```

The package's own tests are two cases, both address formatting. So `connect`, `read`, `write`,
`write_all`, `close` and `shutdown_write` — the entire reason the package exists — were qualified
**in name only**. Nothing had ever lowered a call into the raw bindings.

### What that concealed: DEV-146, a build-breaking defect, on develop

The CD-344 signature change (`&TcpStream` -> `&mut TcpStream` on `read`/`write`/`write_all`,
Codex's work, committed by me) makes any program that CALLS those functions fail to build:

```text
MIR-0005 call argument: expected Ref { mutable: false, inner: HostResource(tcp_stream) },
                        found    Ref { mutable: true,  inner: HostResource(tcp_stream) }
```

`weaken_ref_to` does not weaken `&mut T` to `&T` when `T` is a `HostResource`. Accepted by the
front end, refused by MIR verification — the DEV-132/DEV-133 class, third mechanism. Registered as
**DEV-146**; the signatures are reverted to shared borrows with the defect named at their
definition.

**My CD-344 verification was insufficient and I can name how.** I ran `stark check` (front end
only, which accepts) and the package gate (whose consumer never calls the affected functions). I
also checked that no package consumes the changed API — true, and exactly why nothing caught it.
The check I did not run is the one that matters for a resource package: build a program that
actually calls the thing.

### First end-to-end observation of the resource path

Never done before. A native client against a real loopback listener:

```text
client: wrote / 5 / closed          exit 0
server: received b'PING\n' / closed
```

connect, write, read, close all work. The package's behaviour is sound; only the build was broken.

**Drop elaboration verified, not assumed.** `close()`'s comment claims MIR emits
`stark_tcp_stream_close` exactly once for an owned stream. Confirmed in the generated Rust:
`_7.drop_with(|__v| unsafe { stark_tcp_stream_close(__v.as_raw()) })` for a program that never
calls `close`.

**Affine lifecycle negatives, observed for the first time:**

| Shape | Outcome |
| --- | --- |
| double `close` | REFUSED, E0100 |
| use after `close` | REFUSED, E0100 |
| never closed | accepted — drop elaboration emits the close (verified above) |
| closed on one branch only | accepted — same |
| stream stored in a `Vec` | accepted |

### A STRUCTURAL LIMIT OF THE GATE, which is the durable finding

I tried to close the hole by making the consumer exercise the resource path. **It cannot.** Step 5
is `stark run` on the consumer, and the interpreter has no provider layer — any consumer touching a
bound resource dies with "provider binding not lowered". So the seven-step gate is CONSTITUTIONALLY
unable to qualify a resource path, for `stark-net` or any future resource package.

That is not a `stark-net` problem and should not be patched inside one. Resource lifecycle belongs
in a native-only test alongside the existing provider e2e suites (`a10_*`, `c788_lifecycle_e2e`) and
the C7.8 native-capabilities workflow. **Filed as the recommended next step, not done here** — it
is a gate change, and gate changes need their own scope.

### HC1 by contrast

`stark-url` is genuinely qualified: 19 tests, 9 exercising the new absolute-URL surface, and a
consumer that calls `parse_url`/`Url::parse`. Pure parsing, no resources, nothing deferred.

### Correction

Commit messages CD-337 … CD-344 say "all ten packages qualify". It has been **eleven** since
`56a78b4` added `stark-net`, which landed between CD-336 and CD-337. The RUNS covered all eleven
and passed; the descriptions were stale. Corrected here.

EVIDENCE: eleven-package qualification exit 0; `stark-net` check/test/fmt clean; end-to-end native
TCP client against a Python listener; generated-Rust inspection for drop elaboration; five affine
lifecycle probes.
FILES: stark-net/src/lib.stark (signatures reverted, DEV-146 named),
starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-146), COMPILER-STATE.md.
NEXT: a native resource-lifecycle test for `stark-net` before HC3/HC4 consume these APIs, and
DEV-146 before the `&mut` signatures can be restored.

## CD-343 — WP-DEV-134-139 final report; programme complete pending CI (2026-08-02)

**All six CD-334 defects repaired, all infrastructure tasks delivered. §17 report at
`STARKLANG/docs/compiler/work-packages/WP-DEV-134-139-FINAL-REPORT.md`.**

**Recommendation: release, CONDITIONAL on CD-340/341/342 reporting green.** WP §15's gate is
otherwise satisfied, including its DEV-135 branch: the inventory proved parent poisoning
unacceptable, and the precision a DEV-135b would have built already existed.

**The qualification that must not be smoothed over.** CD-337 NEVER WENT GREEN — it failed
`clippy::collapsible_match`, the fix landed in CD-338, and every commit from there is green, so
DEV-136's code is transitively covered. But "aggregate CI green" is a release-gate item, and CI is
the SOLE workspace authority since CD-337 dropped local workspace runs. CD-341 and CD-342 each add
a required job to `ci-complete` and neither has yet been observed passing.

**Root cause of that miss, which matters more than the lint.** The repo pins `channel = "stable"`;
CI's resolves to 1.97.0, this machine's had gone stale at 1.93.0. Every "clippy clean" before
CD-338 was against an older lint set than CI's. Gate is now `cargo +1.97.0 clippy`.

**What the programme actually found.** None of the six needed a design change; four were a single
wrong line or a single missing consultation, and DEV-135 — sized by the work package as "full
field-sensitive move paths" — was one enum variant, because the move model was already
field-precise and only field IDENTITY was broken. Four of six were WIDER than filed, and in every
case the extra half was found by the repair's own must-pass tests rather than by the reproducer.

**Residual, all registered, none from CD-334:** DEV-121 stays open with its blind spot now named
(INV-VALUE-REP-001 checks `let` bindings; a for-loop binding is not a `let`, and both known
instances were loop items). DEV-140…145 registered at CD-342. DEV-083 open. `types_equal`'s
missing `Ty::Param` arm is symptomless and unowned. `?` conversion semantics is a language-design
question with no owner.

FILES: STARKLANG/docs/compiler/work-packages/WP-DEV-134-139-FINAL-REPORT.md, COMPILER-STATE.md.
NEXT: owner review; then the DEV-121 invariant extension is the highest-value unowned item, since
it would close a class rather than another instance.

## CD-342 — the layer audit is an enforcing gate; its six findings are now registered (2026-08-02)

**WP-DEV-134-139 §11. The audit reported and passed unconditionally, so a NEW layer defect could
appear and the suite would stay green — it could only ever be read by a human who happened to
look. It now fails on any UNREGISTERED finding.**

**The bar is not zero findings.** Six reachable lowering refusals exist and are NOT repaired by
this programme. They are now numbered, which is the actual change: CD-331 found and printed them
and they had carried no deviation number since.

| DEV | Probe | Reachable lowering refusal |
| --- | --- | --- |
| DEV-140 | L7153 | `Vec::` method outside the implemented lowering set |
| DEV-141 | L8093 | `HashMap` over a user-`Drop` value type |
| DEV-142 | L9130 | droppable composite carrying a borrowed element |
| DEV-143 | L5346 | `assert_eq` on a user-defined type |
| DEV-144 | L3698 | `for` over a non-range, non-`Vec` iterator |
| DEV-145 | L6450 | method on a peeled type outside the implemented slice |

Every probe now declares the disposition it is expected to have — `FrontEnd`, `Lowers`, or
`KnownDev("DEV-xxx")` — and the test compares actual against registered.

**It fails in BOTH directions, which is the part worth stating.** A registered defect that stops
reproducing fails too, because that means either the DEV was fixed and its registration is stale,
or the probe no longer reaches the construct it was written for. Both need a human decision; both
are invisible to a test that only looks for regressions. The failure was verified by deliberately
mis-registering one probe and confirming the gate reports "registered as Lowers but actually
KnownDev".

**Disposition of the six is unscheduled and per-site, not global.** Two repair shapes exist —
raise the refusal into semantic analysis (E0105) or teach lowering the construct (DEV-132,
DEV-133). CD-294 is the precedent for why raising is not always cheap: E0106 was reverted because
`v[i]` appears in value AND place positions that only later phases distinguish.

Local: `cargo test --test layer_audit` green; negative case verified by mis-registration.
FILES: starkc/tests/layer_audit.rs, starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: reconciliation and the §17 report — the last two programme tasks.

## CD-341 — the external sample suite is published, pinned, and gated in CI (2026-08-02)

**WP-DEV-134-139 §10.1/§10.2. The suite that found all six CD-334 defects is now a repository CI
clones at a fixed SHA, not a directory on one machine.**

```text
repo   navraj007in/stark-samples   (public)
pin    b3b28e757f38d691e7309f168d1209e28ac459af
job    external-sample-suite  ->  required via ci-complete
```

**Kept EXTERNAL on purpose (§10.2).** The fixture corpus and the generated C6 corpus both grow
from this compiler's own model of what programs look like. The sample suite grew from TASKS — sort
a vector, walk a graph, parse an expression — and that independence is why it found six defects the
in-tree suites did not. Absorbing it would reorganise it around compiler subsystems and destroy
the property that makes it useful.

**PINNED BY SHA, not tracking a branch.** §10.3 requires that when a compiler fix changes an
external task's expected outcome, the suite's manifest is updated and the pin moves to the commit
carrying that update, in the same logical change set. A floating `main` would let the suite drift
green or red for reasons unrelated to the commit under test — precisely the confusion the pin
prevents. The job also asserts the resolved HEAD equals the pin, because `ref:` accepts a branch
name and would otherwise float silently.

**It runs the BUILT artifacts**, never `cargo run`, so what is tested is what would ship.

**A machine-readable manifest now exists (§10.1)**, which the suite previously lacked —
`run-all.sh` printed pass/fail and nothing more. `manifest.json` records 39 cases with, per case:
id, description, linked DEV, and the expected outcome for EACH engine (front end, HIR, MIR,
native), using an explicit vocabulary that distinguishes `not_reached` (rejected earlier) from
`not_supported` (the engine lacks the construct) from `not_exercised` (this case does not drive
it). `verify.py` drives it, writes `results.json`, and CI uploads both as evidence.

**An unexpected PASS fails the job.** A reproducer that silently starts working means an
expectation went stale, not that the suite is healthy — the six `defects/` cases are exactly this
shape, since every one of them now does the OPPOSITE of what its file header describes.

Local: `verify.py` — 39/39 cases matched, 1.8s. CI YAML validated; `ci-complete` now needs twelve
jobs, and forgetting to add one remains visible there rather than silently unprotected.
FILES: .github/workflows/ci.yml, COMPILER-STATE.md. Suite content lives in its own repository.
NEXT: §11 layer-audit inventory enforcement, then reconciliation and the §17 report.

## CD-340 — DEV-138 CLOSED as a DEV-121 instance; all six CD-334 defects repaired (2026-08-02)

**WP-DEV-134-139 Part F. The classification came first and decided the repair, as §9 required.**

```text
declared item type   &str            06-Standard-Library.md: SplitIter / String::split / &str
HIR runtime value    Value::String   OWNED  <- the defect
value_is_copy        Value::Str -> true, Value::String -> false
front end            ACCEPTS (sees a Copy shared reference)
MIR / native         VACUOUS - both refuse SplitIter outright (C4.5)
```

**The MIR and native rows are vacuous, not confirming**, and are recorded that way rather than
counted as agreement: those engines do not implement `SplitIter`, so they could not have
disagreed. §9.3's "treat as distinct" test requires MIR to emit `Move` for a Copy shared-reference
item AND all engines to consume it. Neither holds; every testable fold criterion does.

**Producer-specific, which is what identifies it as DEV-121 rather than something new.** Six shapes
were probed: `&Vec<String>`, `&Vec<Int32>`, `chars()`, and a plain `&str` outside a loop were
already correct. Only `split` was wrong — and `trim`/`substring`, with the SAME declared return
type, already yielded `Value::Str`. The repair makes `split` consistent with its siblings rather
than introducing a rule. One line, no new `Value` variant, no amendment.

**THE MORE USEFUL FINDING IS WHY THE INVARIANT MISSED IT.** INV-VALUE-REP-001 exists precisely to
catch this class, and checks at every **`let`** that a binding declared `&str`/`&[T]` does not hold
owned storage. A **for-loop binding is not a `let`**. Both known DEV-121 instances —
`String::bytes()` at CD-305 and `String::split()` here — were reachable through a loop item, and
both were found by a user-facing program rather than by the invariant. Extending it to loop
bindings and call arguments is what would close the class; finding a third instance by hand would
not. Recorded against DEV-121, unowned.

**ALL SIX CD-334 DEFECTS ARE NOW REPAIRED.** Three were soundness holes; none required a design
change, and four turned out to be a single wrong line or a single missing consultation:

| DEV | Root cause in one line |
| --- | --- |
| 134 | operand and return type were never compared |
| 137 | a condition is neither a block nor a statement, so nothing popped its borrows |
| 136 | move state merged syntactic children instead of reaching predecessors |
| 135 | a field was identified by the span it was written at |
| 139 | two bound lookups each read half the generic environment |
| 138 | one producer returned an owned value for a borrowed type |

Local:
- cargo test --test dev138_iterator_item_representation -- 10 cases, green
- cargo test --test c63a_string --test copy_canon_matrix --test three_engine_differential --test
  exec_snapshots --test conformance --test gate2_valid --test gate3_execution -- 209 green
- cargo fmt --all -- --check -- clean; rustfmt on the two touched files only
- qualify-first-party-packages.py over all ten packages -- exit 0
- external task-shaped suite -- 34/34, and its `defects/05` reproducer now runs correctly
- full workspace NOT run, per the amended evidence policy

CI:
- the aggregate gate is the authority for this commit

FILES: starkc/src/interp.rs, starkc/tests/dev138_iterator_item_representation.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: the three non-defect programme tasks — in-tree regression manifest (§10.1), pinned
external-suite CI (§10.2, BLOCKED: the suite has no git remote), layer-audit inventory enforcement
(§11) — then final reconciliation and the §17 report.

## CD-339 — DEV-139 CLOSED: a method body reads the impl's bounds, not only its own (2026-08-02)

**WP-DEV-134-139 Part E. Five of six defects closed; only DEV-138 remains.**

**Wider than filed: it was TWO lookups, and the second was deferred.** The entry names operator
desugaring, but `satisfies_bound` — ordinary trait-bound satisfaction — had the identical gap. Each
kept its OWN copy of the parameter lookup and each consulted `current_fn_generics` alone; they
agreed only by coincidence. And the trait-bound half is deferred: DEV-067(a) captures "the generic
environment this obligation was recorded in" and replays it at drain, and that capture was also
fn-generics-only — so an obligation raised inside `impl<T: Ord> Pair<T>` replayed against half its
environment and still failed after the operator half was repaired. Two of this defect's own tests
found that second half, which is the argument for writing the must-pass set before assuming the
first fix was the whole fix.

**Nothing new was brought into scope.** WP-C6.2b-F5 already installed impl-head generics in
`current_impl_generics` for method bodies. The lookups never asked. This repair is a READ, not a
new binding — it cannot change which names are in scope, only which declared bounds are found.

**Two helpers, each written once:**

```
param_declares_bound(param, required)   both lookups call it
current_generic_env()                   the deferred capture calls it
```

DEV-128 and DEV-130 are both "the rule was written twice and the copies drifted". This was already
two copies; it is now one each.

**Negative controls, because WIDENING an environment risks discharging obligations never
declared:** no bound at all, `Eq` where `Ord` is required, `Ord` where `Num` is required, a bound
on a DIFFERENT parameter (pins that the lookup still matches on parameter NAME rather than finding
any bound in scope), an unbounded method-level parameter, and an undischarged callee obligation.

**DEV-083 is NOT closed by this.** It is impl-head *matching* — a concrete position in an impl head
against an unresolved receiver type argument. This was impl-head *bounds being read*. Different
mechanism; DEV-083 remains OPEN and unowned.

**What class of program is now prevented from failing:** any generic CONTAINER whose methods use
the bounds its impl declares — `Heap<T: Ord>`, `SortedVec<T: Ord>`, a `max` method. The rule is on
the environment, not on `Ord` or on operators, so it covers `Eq`/`Num`/user traits, inherent and
trait impls, and trait-bound obligations as well as operator desugaring.

Local:
- cargo test --test dev139_impl_generic_bounds -- 16 cases (10 accept, 6 reject), green
- cargo test --test c62b_f5_impl_bounds --test c62b_f6_self_normalisation --test
  c62c_associated_types --test c62d_operator_coretrait --test cross_package_generics --test
  native_c6_2_generics_traits -- 60 green; the generics/bounds subsystem closest to this change
- cargo test --test conformance --test gate2_valid --test gate3_execution -- 69 green
- cargo test --test dev134_try_error_type --test dev135_field_move_paths --test
  dev136_terminating_path_moves --test dev137_while_condition_borrows -- 62 green, all four
  previously closed defects in this programme
- cargo test --test mir_verify --test three_engine_differential -- 160 green
- cargo fmt --all -- --check -- clean; rustfmt on the two touched files only
- qualify-first-party-packages.py over all ten packages -- exit 0
- external task-shaped suite -- 34/34, and its `defects/06` reproducer now checks OK
- full workspace NOT run, per the amended evidence policy

CI:
- the aggregate gate is the authority for this commit, including clippy on CI's own stable

FILES: starkc/src/typecheck.rs, starkc/tests/dev139_impl_generic_bounds.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: DEV-138 — build the engine matrix first and apply WP §9.3's decision rules, rather than
assuming it is independent.

## CD-338 — DEV-135 CLOSED: a field is one place however many times it is written (2026-08-02)

**WP-DEV-134-139 Part B. Also carries a `collapsible_match` fix for CD-337 that CI caught and the
local gate did not, and two Codex documentation updates to the frozen P1 workload.**

**THE ESTIMATE WAS WRONG AND THE RECORD IS CORRECTED, NOT REWRITTEN.** The CD-334 inventory said
"the gap is in the front end's `moved_places`, which is keyed on whole locals". It is not.
`moved_places` is a `HashSet<Place>`, `Place` already carries `projections`, and `places_overlap`
already does prefix matching. The front end was ALREADY field-precise: moving `pair.left` already
left `pair.right` live, and moving the parent afterwards was already refused.

**The actual defect was field IDENTITY, one enum variant wide:**

```rust
Projection::Field(name.lo, name.hi)   // the SPAN the name was written at
```

Two mentions of one field sit at different byte offsets, so `owner.handle` on line 5 and
`owner.handle` on line 6 were two DIFFERENT projections that `places_overlap` correctly reported as
disjoint. Nothing was missing from the move model; the comparison could never succeed. Storing the
resolved NAME fixes it. Same class as DEV-122 — identity taken from a span rather than from what
the span denotes.

**So the WP's two-stage model was never entered, and that is a real outcome rather than a shortcut.**
§5.2 split this into a conservative "DEV-135a parent poisoning" gate and a "DEV-135b precision"
follow-on. The inventory ruled poisoning out — sibling survival is asserted by the conformance
fixture set and four differential suites. But the precision DEV-135b was meant to BUILD already
existed. The repair is neither stage. **No DEV-135b is filed and none is owed**: sibling survival,
nested paths, parent/child ordering, and exactly-once drop are all covered, which is exactly what
DEV-135b's closure criteria asked for. WP §15's release gate resolves on its second branch.

**What class of program is now prevented:** any program that moves the same owned field, tuple
element, or nested field out twice — and, by the same prefix rule that already worked, any that
moves a parent after a field or reads a field after the parent. The check is on the PLACE, not on
syntax, so it holds however the field is reached.

**A CI-vs-local gate divergence, worth more than the lint it caught.** CD-337 went red on
`clippy::collapsible_match`. The lint is real; the reason it was missed is that the repo pins
`channel = "stable"` and CI's stable resolves to **1.97.0** while this machine's stable had gone
stale at **1.93.0**. Every "clippy clean" reported earlier in this programme was against an OLDER
lint set than CI's. Corrected here and going forward: the gate is `cargo +1.97.0 clippy`. This
matters disproportionately now that CD-337 made CI the sole workspace authority — a local gate that
silently differs from CI undermines exactly that arrangement.

**Codex changes included at the owner's instruction, reviewed not rubber-stamped.** Two docs on the
frozen P1 REST workload: the plan's status moves to `IMPLEMENTED — TIER-1 QUALIFIED`, and the report
identifies `P1-COMPILER-001` as a local label for the already-governed `DEFECT-C788-LOOP-TEMP`
(discharged by MIR amendment A12) while demoting a stale `P1 PARTIAL` handoff to quoted history.
Cross-references verified: `a12_storage_end_shapes.rs`, `mir-amendment-A12-storage-end.md`, and
CD-263/264/265/273 all exist. **CD-269 is cited and is absent from this file** — it is a real
decision (commit `28a9ad1`, cited in five other documents), so the Codex text is correct and the gap
is in this ledger. Recorded, not silently patched.

Local:
- cargo test --test dev135_field_move_paths -- 16 cases (6 reject, 10 accept), green
- cargo test --test conformance --test gate2_valid --test gate3_execution -- 65 green
- cargo test --test mir_verify --test mir_differential --test three_engine_differential -- 292 green
- cargo test --test dev134_try_error_type --test dev136_terminating_path_moves --test
  dev137_while_condition_borrows -- 46 green, all three previously closed defects
- cargo test --test c61f_reference_boundary --test c61f_structural_copy --test
  native_c6_1_ownership --test operand_move_inventory --test copy_canon_matrix -- 51 green
- cargo test --test native_c5_3_aggregates_enums --test c6_generated_corpus -- 27 green; these are
  the suites that assert partial-move field precision at the MIR and native layers
- cargo fmt --all -- --check -- clean; rustfmt on the two touched files only
- full workspace NOT run, per the amended evidence policy
- `cargo +1.97.0 clippy --workspace --all-targets --all-features -- -D warnings` was IN FLIGHT when
  this was committed, at the owner's instruction to let CI decide. It is NOT claimed as passing.

CI:
- aggregate workspace gate is the authority for this commit, including clippy on CI's own stable

FILES: starkc/src/borrowck.rs, starkc/tests/dev135_field_move_paths.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md, and the two Codex P1 documents.
NEXT: DEV-139 — merge impl-level generics into the obligation environment the operator check reads.

## CD-337 — DEV-136 CLOSED: only branches that reach a join contribute to it (2026-08-02)

**WP-DEV-134-139 Part D, and the second of four milestone points. `if flag { return out; }
out.push('a');` compiles.**

**Layer: `borrowck.rs`.** The `If` arm unioned the then-branch's move set into the post-state
unconditionally; the `Match` arm extended the merged set from every arm. Neither asked whether the
branch reaches the join. A branch that `return`s is not a predecessor of the statement after the
`if`, so its moves were being attributed to a path they never happened on.

**Divergence is read from existing authorities, not re-derived from syntax.** A
`Return`/`Break`/`Continue` statement anywhere in the sequence, plus the type checker's own
`Ty::Never` for `panic(..)` and any call returning `!`. Composite forms recurse: an `if` diverges
only when both sides do, a `match` only when every arm does.

**THE DIRECTION OF CONSERVATISM IS THE ENTIRE SAFETY ARGUMENT.** The predicate answers "does this
definitely NOT reach the join?":

```
wrong `true`   -> drops a real move from the join -> accepts use-after-move -> UNSOUND
wrong `false`  -> keeps the old false positive     -> merely annoying
```

So it reports `true` only on evidence, and anything unrecognised falls through to `false`. `loop`
without a reachable `break` is deliberately NOT treated as diverging — judging it needs
reachability analysis the checker does not have, and guessing would land on the unsound side.

**Two merge subtleties, both found while writing the repair and both pinned:**

| Case | Naive answer | Correct answer |
| --- | --- | --- |
| `if` with no `else`, branch terminates | branch's move set | the state from BEFORE the `if` |
| `match` where ALL arms terminate | empty merged set | the pre-match state |

The second is the dangerous one: an empty merge would silently resurrect a value moved BEFORE the
`match`, turning a false positive into a false negative. `a_move_before_an_all_diverging_match_
is_still_rejected` exists precisely for it.

**Drop obligations, not just diagnostics.** `a_droppable_value_survives_a_terminating_branch`
executes both paths and asserts each `Guard` is destroyed exactly once — the false path drops at
end of scope, the true path moves into a callee that drops it there.

**What class of program is now prevented from failing:** any program whose move happens only on a
path that leaves the function or the loop. The rule is on the control-flow edge, not on the
syntax, so it covers `return`, `break`, `continue`, `panic`, nested `if`, and `match` arms
uniformly — and it does NOT cover a branch that can fall through, which is what keeps
maybe-moves rejected.

Local:
- cargo test --test dev136_terminating_path_moves -- 14 cases (9 accept, 5 reject), green
- cargo test --test conformance --test gate2_valid --test gate3_execution -- 65 green
- cargo test --test mir_verify --test mir_differential --test three_engine_differential -- 292 green
- cargo test --test dev134_try_error_type --test dev137_while_condition_borrows -- 32 green,
  the two previously closed defects in this programme
- cargo test --test c61f_reference_boundary --test native_c6_1_ownership --test
  operand_move_inventory --test copy_canon_matrix --test exec_snapshots --test snapshots -- 43 green
- cargo clippy --release --lib --tests --all-features -- -D warnings -- clean
- cargo fmt --all -- --check -- clean; rustfmt on the two touched files only
- qualify-first-party-packages.py over all ten packages -- exit 0
- external task-shaped suite -- 34/34
- FULL WORKSPACE (milestone 2 of 4) -- see the commit for the recorded result

CI:
- aggregate workspace gate PENDING

FILES: starkc/src/borrowck.rs, starkc/tests/dev136_terminating_path_moves.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: DEV-135b — full field-sensitive move paths, which the inventory established is the
release-gating repair rather than the DEV-135a poisoning gate.

## CD-336 — DEV-137 CLOSED: condition-only borrows end at the branch boundary (2026-08-02)

**WP-DEV-134-139 Part C. `while i < v.len() { v[i] = 5; }` compiles. So does the `if` form, which
had the identical defect and was found by this repair's own test.**

**The layer, recorded before the repair as the work package required: `borrowck.rs`.** Not MIR,
not liveness, not the back-edge. `active_borrows` is a stack scoped by exactly two mechanisms —
`check_block` truncates at block end, `check_stmt` truncates after each expression statement. A
CONDITION is neither: it is an expression evaluated outside any statement of its own. So

```rust
hir::ExprKind::While { cond, body } => {
    self.check_expr(*cond);      // pushes the auto-borrow `values.len()` takes
    self.check_block(*body);     // records its entry depth AFTER that push
}
```

left the condition's temporaries on the stack for the whole body, and `check_block` restored to a
depth that already included them. Nothing popped them until the enclosing statement ended.

**Wider than filed, same mechanism.** `if` conditions were identical. The growing-vector must-pass
case is what exposed it: `if values.len() < 5u64 { values.push(1); }` inside a loop body was
refused for the same reason. One repair, `check_condition`, written ONCE and used by both arms —
DEV-128 and DEV-130 are both "the rule was written twice and the copies drifted".

**The scope boundary is the whole design, and it is not "loop and branch headers".** `match`
scrutinees and `for` iterators are deliberately NOT routed through `check_condition`:

| Position | Borrow must | Why |
| --- | --- | --- |
| `while` / `if` condition | END at the branch | value is consumed by the branch |
| `match` scrutinee | SPAN the arms | PAT-BIND-001 binds payloads by reference into it |
| `for` iterator | SPAN the body | yields references into the iterated value |

Truncating either of the bottom two would hand out references to storage the checker had stopped
tracking. Both are pinned by negative controls that fail if someone later generalises the repair.

**Why depth-based rather than clearing the borrow set.** A borrow created before the loop
(`let view = &values;`) sits at a shallower depth than the snapshot, so the truncate cannot reach
it and a body mutation through its owner is still refused. That is
`borrow_predating_the_loop_stays_live`, and it is the difference between modelling a region and
just forgetting.

**Execution, not merely acceptance.** `a_growing_vector_re_evaluates_its_condition` runs through
the oracle and asserts output. It also settles a question the workaround raised: hoisting
`let n = v.len()` was a SEMANTIC change, not a stylistic one — that loop grows the vector it is
measuring, so a hoisted bound stops early. The samples that carried the hoist workaround were
working around a defect at the cost of a different meaning.

**What class of program is now prevented from failing:** any program that reads a receiver in a
condition and mutates it in the guarded branch. The fix is on the borrow REGION, not on `len` or
on `Vec`, so it holds for every method, every receiver type, `&mut` parameters included, and for
indexed place reads (`while values[0] < 3`) as well as method calls.

Local:
- cargo test --test dev137_while_condition_borrows -- 16 cases (12 accept, 4 reject), green
- cargo test --test conformance --test gate2_valid --test gate3_execution -- 65 green
- cargo test --test mir_verify --test mir_differential --test three_engine_differential
  -- 292 green (78s for the three-engine differential)
- cargo test --test c61f_reference_boundary --test c61f_nested_refs --test native_c6_1_ownership
  --test dev132_vec_index_place --test operand_move_inventory -- 44 green, the borrow/ownership
  subsystem closest to this change
- cargo clippy --release --lib --tests --all-features -- -D warnings -- clean
- cargo fmt --all -- --check -- clean; rustfmt on the two touched files only
- qualify-first-party-packages.py over all ten packages -- exit 0
- external task-shaped suite -- 34/34
- full workspace NOT run for this commit, per the 2026-08-02 evidence ruling; the next milestone
  run is after DEV-136

CI:
- aggregate workspace gate PENDING

FILES: starkc/src/borrowck.rs, starkc/tests/dev137_while_condition_borrows.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: DEV-136, then the milestone full-workspace run.

## CD-335 — DEV-134 CLOSED: `?` now relates its operand to the return type (2026-08-02)

**WP-DEV-134-139 Part A. `?` required exact error-type compatibility; it now does. The ruling is
REJECT, not convert — recorded as a decision, because it is a language question and not an
implementation detail.**

```text
`?` requires exact error-type compatibility.
Implicit From-based propagation is not part of this repair.
```

**The defect was one missing relation, not two.** The `Try` arm asked "is the enclosing return
type `?`-capable?" and "is the operand `?`-capable?" as INDEPENDENT questions and never compared
them. That single omission produced two symptoms, and the repair work found the second:

| Accepted before | Propagated value | Caller sees |
| --- | --- | --- |
| `Result<_, Low>?` in a fn returning `Result<_, High>` | a `Low` | tag from another enum |
| `Option<_>?` in a fn returning `Result<_, _>` | a `None` | tag from another enum |
| `Result<_, _>?` in a fn returning `Option<_>` | an `Err` | tag from another enum |

The constructor half was NOT in DEV-134 as filed. It is the same mechanism and the same repair, so
it widened the existing entry rather than taking a new number (WP §2, one mechanism one repair).

**Deferred, like `display_checks`, and for the same reason.** The operand's error type is routinely
an inference variable while the body is being checked (`Err(make())?`), so an eager comparison
would either reject valid code or force a premature binding. `check_try_compatibility` is recorded
during checking and drained after inference settles.

**E0006 widened rather than a new code allocated.** The spec's E0006 now covers the whole
return-type contract for `?` — wrong constructor, mismatched error type, or a function that
returns neither. One code per concept, normative table stable, conditions distinguished by
message. `non_result_return_reports_once_not_twice` pins that the pre-existing condition still
reports once rather than twice.

**A LATENT GAP FOUND BY THIS WORK'S OWN NEGATIVE CONTROL, and deliberately not repaired.**
`types_equal` has no `Ty::Param` arm: two occurrences of the same type parameter compare unequal
and fall to `_ => false`. Its existing callers are coherence and overlap paths where `Ty::Param`
is pre-handled or where a conservative `false` is safe, so it has no demonstrated symptom there —
but it made the first version of this repair reject

```stark
fn low<E>(e: E) -> Result<Int32, E> { Err(e) }
fn same<E>(e: E) -> Result<Int32, E> { let v = low(e)?; Ok(v) }
```

which `error_type_as_a_generic_parameter_is_accepted` caught before it could ship. Widening a
shared coherence primitive to fix a symptomless gap was rejected as out of scope; instead the
structural walk takes the `Ty::Param` behaviour as a PARAMETER (`types_equal_inner`) — written
ONCE, reached by two entry points, because DEV-128 and DEV-130 are both "the rule was written
twice and the copies drifted". Whether `types_equal` itself should be widened is unowned and gets
a DEV number only if a symptom is found.

**What class of program is now prevented, which is the closure question rather than "the
reproducer passes":** no program can propagate a value into a return type that cannot represent
it. The check is on the TYPES at the propagation site, not on any syntactic shape, so it holds for
`?` in any position — nested helpers, generic bodies, chained propagation — and it cannot be
evaded by adding a `From` impl, which is the shape a reader coming from Rust would expect to work.

EVIDENCE (all run locally, at this head):
`cargo test --test dev134_try_error_type` — 16 cases, 7 reject / 9 accept, green.
`cargo test --test conformance --test gate2_valid --test gate3_execution` — 65 green.
`cargo test --test c788_synth --test a10_provider_call --test c788_source_time_e2e` — 32 green;
these are the provider paths that use `?` most heavily and were the main over-rejection risk.
`qualify-first-party-packages.py` over all ten packages — exit 0.
External task-shaped suite — 34/34, unchanged.
`cargo fmt --all -- --check` — clean. `rustfmt` was run on the two touched files only, never
tree-wide, because this checkout is shared with parallel sessions.
Spec regenerated (`build-core-spec.py`) and the 112-block fixture corpus re-extracted: manifest in
sync, no block added or renumbered.
LEFT TO CI: `cargo clippy --workspace --all-targets --all-features -- -D warnings` and
`cargo test --workspace --all-targets --all-features` were still running locally when this was
prepared; both are required and are the aggregate gate.
FILES: starkc/src/typecheck.rs, starkc/tests/dev134_try_error_type.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, STARKLANG/docs/spec/04-Semantic-Analysis.md and the
regenerated STARK-Core-v1.{md,html,pdf}, COMPILER-STATE.md.
NEXT: DEV-137, per the work package's required order.

**PROGRAMME STATUS — two counts, deliberately kept apart.** They are not the same number and
conflating them misreports progress in both directions: the defect count understates the work
(regression manifest, external-suite CI, and layer-audit hardening are none of them defects), and
the task count understates release readiness (only the defects gate the release).

```text
Defects (WP-DEV-134-139 Parts A-F)     ALL SIX CLOSED (134, 135, 136, 137, 138, 139)
                                       DEV-135b NOT FILED — see CD-338
                                       DEV-138 closed as a DEV-121 instance; that class stays OPEN
                                       DEV-135b conditional on the DEV-135 inventory

Programme tasks (WP-DEV-134-139)       16 of 16 complete; release recommendation is
                                       CONDITIONAL on CD-340/341/342 CI going green
                                       includes the six defect repairs plus the in-tree
                                       regression manifest (§10.1), the pinned external-suite
                                       CI job (§10.2), layer-audit inventory enforcement (§11),
                                       and final reconciliation (§16)
```

**LOCAL EVIDENCE POLICY, owner ruling 2026-08-02, in force from DEV-137 onward.** Per-commit local
evidence is TARGETED — fmt, clippy over affected targets, the dedicated DEV suite, closest
subsystem suites, MIR differential/verifier when lowering or ownership changes, affected package
qualification, the external sample suite, and a clean `git status --short`.

The original ruling additionally required full local workspace runs at four milestones.
**AMENDED the same day after measurement: local workspace runs are DROPPED entirely after
CD-337.** Each takes ~17 minutes, two more were scheduled, and they duplicate a gate CI already
enforces on every pushed commit. Milestones 1 (CD-335, DEV-134) and 2 (CD-337, DEV-136) were run
and their results are recorded; no further local workspace run is required, including at
programme completion.

**CI's aggregate required check is therefore the SOLE workspace authority**, which makes the merge
gate strictly more important rather than less. A targeted local run supports a commit's evidence
statement and never replaced CI, but from CD-338 onward nothing local covers the workspace at all.
Any commit whose CI run is red is unverified regardless of how much local evidence it carries.

**A procedural rule learned the hard way (CD-337): while a workspace run is in flight, nothing
else may touch cargo.** The first attempt at milestone 2 was invalidated because a
`cargo build --bin starkc` for the NEXT defect landed mid-run, and ~49 test files invoke the
compiler through `CARGO_BIN_EXE` — so an unknown number of suites ran against a binary carrying an
unrelated change. Two other measurement errors in the same session point the same way: a
`head -40` that truncated a run and made `head`'s exit code look like cargo's, and a
`grep '^test result'` that counted tests NAMED `result_*` as suite summaries. Capture the tool's
own exit code, and validate counts against a known total rather than trusting a summary line.

## CD-334 — six defects filed from an external sample suite; three are soundness (2026-08-02)

**An 18-package sample suite was written OUTSIDE this repository, against the release binaries, to
answer "what does it feel like to write ordinary STARK today?". It found six defects, numbered
DEV-134…DEV-139. Three of them are soundness gaps the fixture corpus does not reach.**

| DEV | One line | Class |
| --- | --- | --- |
| 134 | `?` neither converts the error type nor requires a conversion to exist | **soundness** — type confusion |
| 135 | moves of individual struct FIELDS are not tracked; second move surfaces as an ICE | **soundness** (bounded by the oracle) |
| 136 | a move on a `return`ing path is treated as unconditional (E0100) | false positive |
| 137 | a receiver auto-borrow in a `while` CONDITION is live across the body (E0101) | false positive |
| 138 | an iterator-yielded `&str` is consumed by its first use | **soundness-adjacent**; candidate DEV-121 instance |
| 139 | impl-level generic bounds are invisible to operator desugaring (E0500) | false positive |

Full structured entries — normative expectation, reproducer, engine behaviour, impact, workaround,
disposition — are in the canonical ledger, `starkc/docs/conformance/KNOWN-DEVIATIONS.md`. This
record is the index and the finding about method, not a second copy.

**DEV-134 is the one that needs an owner decision rather than an implementation.** The spec does not
scope a `From` conversion at the propagation site, so "convert" would be new semantics — CE-shaped —
while "reject" is the conservative half and can land alone. Filing it does not presume which.

**DEV-138 is filed as a hypothesis, not a finding.** It is plausibly an instance of the still-open
DEV-121 value-representation class rather than an independent defect; INV-VALUE-REP-001 is the
instrument that would settle it, and it has not been run against this reproducer. Recorded that way
deliberately, so the count is not inflated by a duplicate.

**Why an external suite found things the corpus did not, which is the durable point.** The fixture
corpus and the generated C6 corpus both grow from the compiler's own model of what programs look
like. The sample suite grew from *tasks* — sort a vector, walk a graph, parse an expression, encode
a run-length string — and the defects cluster exactly where those two diverge:

```
DEV-136, DEV-137   ordinary imperative loop and early-return shapes
DEV-135, DEV-138   ownership of things the corpus rarely uses twice
DEV-139            generic CONTAINERS with methods, not generic functions
```

DEV-137 is the most disruptive in practice: `while i < v.len()` is how an indexed loop is written,
and every in-place algorithm hits it. Its workaround — hoist the length — **fails when the length
changes**, so a growing queue must track its length by hand. That is worth weighting above the other
two false positives when this is scheduled.

**A limitation found alongside them, filed as neither defect nor deviation because it may be
intended:** `Box<T>` cannot be dereferenced, so a recursive tree built with `Box` can be constructed
but never walked by reference — traversal requires consuming it with `Box::into_inner`. The suite
routes around this with an arena (nodes in one `Vec`, children as indices), which is a legitimate
technique rather than a workaround. If by-reference traversal of a boxed tree is meant to be
possible, this is a seventh defect; if not, it is a documented consequence of having no `Deref`.
The owner's call, which is why it carries no DEV number.

EVIDENCE: six runnable reproducers, one per DEV, verified against
`starkc/target/release/{stark,starkc}` at this head — each shown to reproduce its stated
`starkc check` and `starkc run` outcomes. The suite itself is 18 packages / 55 files / ~5,500 lines
and passes 34 of 34 checks with the six workarounds applied and commented.
SCOPE: **documentation only.** No compiler source, test, or fixture was modified under this CD, and
no defect was repaired. Nothing here is gate evidence.
FILES: starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-134…139 appended), COMPILER-STATE.md.
NEXT: owner triage — DEV-134's reject-vs-convert ruling, and whether DEV-138 folds into DEV-121.

## CD-294 — E0106 reverted: the layer migration was not the cheap kind (2026-07-31)

**CD-293 moved `v[i]`-on-a-non-`Copy`-element from MIR verification into semantic analysis as
E0106, on the E0105 precedent that acceptance and executability should agree. It broke three
working programs and is reverted.**

| What broke | Why it was never a move |
| --- | --- |
| `holder.key == values[0]` | dispatches through `Eq::eq(&self, &Key)` — auto-borrowed |
| `vs[idx()].push(arg())` | method receiver — borrowed |
| `v[1u64] = Loud { id: 20 }` | assignment target — never read at all |

**The premise was wrong, not the execution.** MIR reaches `VecIndexGet` only from the value-read
path; a receiver, an assignment target, a borrow, and an auto-borrowed comparison operand all index
a Vec and never arrive there. The front end sees only the syntax `v[i]`. Refusing on the syntax
refuses four things to catch one.

Scoping it correctly means enumerating every place context — `&`, receiver, field base, assignment
LHS, comparison operand, nested index, match scrutinee — and **missing one breaks working code**,
which is what happened three times. That is a value-context analysis the checker does not have, and
building it is a design change, not a diagnostic fix.

**The ergonomic win survives where it can tell the two apart.** MIR-0016 now reads: "…`v[i]` reads
by value, which would move the element out of the Vec; borrow it instead with `v.get(i)` (yields
`Option<&T>`), `&v[i]`, or read it in place by iterating with `for x in &v`". A message cannot
produce a false positive.

**CD-293's other two changes stand and are untouched:** `for x in &v` and `&v[i]` are pure
additions — they can only make previously-rejected programs work, never break working ones.

**WHAT THIS MEANS FOR THE LAYER AUDIT, which is the durable finding.** The estimate of "1–2 days,
each fix small" for batch-migrating MIR refusals into semantic analysis is wrong. E0105 was cheap
because by-value `Vec` iteration has exactly ONE syntactic form. E0106 was not, because `v[i]`
appears in value and place positions that only later phases distinguish. The audit's output
therefore needs a fourth classification beyond reachable/unreachable:

```
CHEAP     one syntactic form, unambiguous            (E0105-shaped)
EXPENSIVE appears in value AND place positions;      (E0106-shaped)
          needs context analysis the checker lacks
```

Which one a site is cannot be read off its message. It takes a probe per site.

**Also recorded: how this was missed locally.** Two of the three failures are unit tests inside
`src/interp.rs`, which `cargo test --test <name>` never compiles. Only
`cargo test --workspace --all-targets` sees them — the same command CI runs.

EVIDENCE: full-workspace verification was still running when this was pushed, at the owner's
direction. fmt clean and clippy clean under CI's exact flags
(`--workspace --all-targets --all-features -- -D warnings`) were confirmed before the revert.

## CD-293 — the three Vec ergonomics edges, and the guard that was a name filter (2026-07-31)

**Found by writing CD-292's file surface against the real API, not by review.** Reading a
`Vec<DirectoryEntry>` — a Vec of owning structs, which is the shape of most real data — failed
three ways in a row, and only the fourth spelling worked.

| Spelling | Was | Now |
| --- | --- | --- |
| `for x in &v` | E0001 "requires an iterable value" | **works** — same cursor as `v.iter()` |
| `for x in v` | E0105, names `.iter()` | unchanged (by-value moves elements out) |
| `v[i]`, non-`Copy` | **MIR-0016 at verification** | **E0106 in semantic analysis**, with a help |
| `&v[i]` | unrepresentable | **works** — `VecGetRef`, `None` arm traps |

**`for x in &v`** was not an architectural limit — it was a missing arm, in three engines: the type
checker, MIR lowering (builds the cursor with the same `VecIterNew` the method call emits), and the
HIR oracle (builds `Value::VecIter` at the same place). The differential harness caught the oracle
half; a two-engine change would have shipped a divergence.

**`v[i]` on a non-`Copy` element is correctly refused** — it would move the element out of a place
the Vec still owns. What was wrong was WHERE: it type-checked, ran in the oracle, and died at MIR
verification. **An accepted program no compiler could build** — precisely the defect class WP-C7.9
Packet E fixed for by-value `Vec` iteration (E0105) and left unfixed for indexing. E0106 now raises
during semantic analysis and its help names both borrowing reads, because there are two and neither
is guessable from "requires a Copy element type". Added to the normative spec; compiled spec
regenerated.

**`&v[i]` was unrepresentable, and that part IS architectural.** A Vec is `MirTy::Core` — an opaque
runtime type, not a projectable place — so there is no `Projection::Index` to borrow, which is why
`&a[i]` on an ARRAY always worked and `&v[i]` never did. Closing it needed no representation change:
`VecGetRef` already yields `Option<&T>`, and the `None` arm IS the out-of-bounds case, so it raises
`IndexOutOfBounds` — same category, same observable behaviour as `v[i]`, reached by another route.

**One column of disagreement, caught by the harness.** MIR blamed the index expression, the oracle
blamed the enclosing `&`. Same category, same line. MIR now matches the oracle: three-engine
agreement is the authority on provenance, and either span alone would have been defensible.

**CD-292's CI failure: a name filter standing in for a semantic rule.**
`no_environment_mutating_function_is_declared` scanned EVERY first-party provider for
`set|put|unset|remove|clear|exec|spawn`. It failed on `stark_iofile_set_len` and `stark_iofile_remove`
while `stark_iofile_write` and `stark_file_create` — which mutate no less — sat beside them and
passed, because their names miss the list. Packet 5's rule is that the **process environment** is
read-only, not that no provider mutates anything; a filesystem provider that cannot write is not a
filesystem provider. Split accordingly: `exec`/`spawn` stay whole-registry (nothing runs a process),
the mutation list applies to `process.*` capabilities. The guard tests the rule again instead of the
spelling.

EVIDENCE: `c63c_iterators` **22/22** three-engine, including the out-of-bounds trap case.
`c783_args_env` 9/9, `c788_starkc_build` 6/6, `c784_file` 11/11. fmt clean.

## CD-292 — the rest of the file surface, executed rather than declared (2026-07-31)

**`stark-io` had four types with nothing behind them.** `OpenOptions`, `SeekFrom`, `FileMetadata`
and `DirectoryEntry` were declared and referenced by no function — `OpenOptions` had validation but
no `open_with_options`, so append/truncate/create-new were unreachable. The API looked like it had
open options; it did not.

Provider: 6 `io_file` symbols → 19. Adds open-with-options, seek, durable sync, set-length, metadata
(by handle and by path), path existence, remove, rename, copy, directory create/remove/list.
Package: ~20 functions consuming all four types, plus `path_join`.

**Encodings, each chosen rather than defaulted** — ABI §10 admits no aggregate parameter:
- open options travel as a **bitmask**; an unknown bit is `InvalidInput`, not a dropped mode;
- a seek origin is a **discriminant byte**; a `Start` offset above `Int64::MAX` is refused rather
  than cast, because a failing `as` traps in every build mode;
- metadata is a **row of out-slots**, with each timestamp a (seconds, valid) pair. No sentinel:
  every `Int64` is a real instant, so there is no in-band value meaning "absent" — the same defect
  CD-277 found in the clock reading, avoided by construction;
- a directory listing is a **bounded NUL-separated snapshot** into a caller buffer, not a cursor.
  A cursor would be a second resource type with its own lifecycle to get wrong; a snapshot owns
  nothing past return. Truncation is reported and raised as `LimitExceeded` rather than returned as
  a short list, which would be indistinguishable from a directory that small.

**Deliberately absent:** recursive directory creation and recursive delete. Both are unbounded
effects from a single call, and the second is the most destructive filesystem primitive there is.
Callers walk with `read_dir` and act on what they have actually seen.

**FOUR LANGUAGE SHARP EDGES, found by writing real code against a real API** — the evidence P1 was
expected to produce, arriving early:
1. `if v.len() > 0 && !f(v[v.len()-1]) { v.push(..) }` — **E0101**. The index in the condition holds
   a borrow across the mutation in the body. Reading the byte into a local first ends it.
2. `entries[i]` where the element is a non-`Copy` struct — **MIR-0016**. `VecIndexGet` requires a
   `Copy` element, and borrowing the indexed place does not help. **A `Vec` of non-`Copy` structs is
   not readable by index at all.**
3. `for x in &v` — **E0001**, `&Vec<T>` is not iterable.
4. `for x in v` — **E0105**, but the diagnostic names the fix: "iterate over a borrow with
   `v.iter()`". The one refusal here that teaches instead of only refusing.

(2)+(3)+(4) compound: a `Vec` of owning structs is reachable **only** through `.iter()`, and two of
the three natural spellings fail first. Worth `WP-C7.8-RB0`'s attention, or its own ergonomics item.

**A behavioural correctness point the test caught:** `set_length` on a handle from `open_file`
fails, because `open_file` is read-only. The API was right and the test was wrong; the test now uses
`open_with_options` with `write`, which exercises that path too.

EVIDENCE: new `io_expanded_surface_executes_from_source_through_stark_io_package` — seek positions,
metadata lengths, listing composition and cleanup asserted on **observed values**, not on absence of
error, so an operation that silently did nothing fails it. `c788_starkc_build` 6/6,
`c788_provider_api_manifest` 10/10, `a11_host_resource` 38/38, `c784_file` 11/11. fmt clean.
Workspace and clippy left to CI.

## CD-291 — file IO works, because the package stopped asking for Core's identity (2026-07-31)

**`io_minimal_executes_from_source_through_stark_io_package` passes.** Ordinary STARK source opens,
writes, reads and closes a real file through the first-party provider in a natively built binary.
CD-290 shipped that test `#[ignore]`d; this removes the attribute rather than the guards.

**The question was never "how do we let a package use Core `file`".** It was "why does a package
need Core's resource identity at all", and the answer is that it does not. `stark-io`'s type is
`NativeFile`, not `File`. The only thing binding it to `file` bought was the provider's existing
symbols — and it cost every guard that protects Core `File`'s single destruction path.

`stark-io` now binds **`io_file`**: a second resource type on the same provider, with its own
symbols (`stark_iofile_*`) and its own handle tag. Consequences, none of them exemptions:

| Guard | Why it passes now |
| --- | --- |
| CD-224 — a package may not claim a Core resource | `io_file` is absent from `ResourceRegistry::builtin()` |
| MIR-0027 — a Core-owned resource may not be a `HostResource` | `io_file` is not `LegacyCore` |
| A11 §5 rule 4 — MIR owns a resource's only close | `io_file` is wholly on the `HostResource` path |

**The verifier caught the one real defect in this design, which is the point.** With `io_file` a
genuine resource, `stark-io`'s `file_close` calling the provider close directly became MIR-0033: a
second destruction path. Correct — drop elaboration already emits that close. `file_close(file:
NativeFile)` now takes the handle by value and calls nothing; taking ownership IS the close. The
signature keeps its `Result` but can only return `Ok`, because a destructor has nowhere to put an
error — `file_flush` is where a flush failure is observable. That is a real API consequence and it
is documented rather than papered over.

**What this does NOT do:** migrate Core `File`. That remains open, and having traced it, it is a
three-engine change — the reference interpreter implements `File` natively (`Builtin::FileOpen`,
`FileCreate`), so checker, interpreter and backend must move together and be requalified for
agreement. An earlier estimate of "a session or two" in this session's discussion was wrong. Nothing
in `stark-io` waits on it any more; only the spelling `File` does.

**Also fixes the CI failure CD-290 caused.** The C6.4 qualification harness rejects any `#[ignore]`
not registered in `CLASSIFIED_IGNORES` — "either the observation is required, in which case fix the
test, or classify it with a reason". CD-290's unclassified ignore failed `C6.4 tier-1 qualification`
on both tier-1 platforms and the dependent agreement job, while all three `fmt, clippy, test` jobs
passed, which is exactly the split that rule exists to produce. Taking the first branch — fixing the
test — removes the deviation at its cause.

EVIDENCE: `c788_starkc_build` **5 passed / 0 ignored** (was 4 + 1 ignored), `c788_provider_api_manifest`
10/10, `a11_host_resource` 38/38, `stark-file/native` 8/8, provider suites green. fmt clean, clippy
clean. No file under `starkc/src/mir/` or `package.rs` is touched by this change.

## CD-290 — WP-IO.1 lands with its guards intact; its e2e is blocked on Route B (2026-07-31)

**The minimal native file-IO slice is committed. Its end-to-end test is `#[ignore]`d, deliberately,
and that is the honest state rather than a failure to finish.**

The slice binds `stark-io`'s nominal `NativeFile` to the provider resource `file`. That resource is
Core-owned (`ResourceRegistry::builtin()` maps it to `LegacyCore(CoreType::File)`). The slice first
ran by removing three compiler guards:

| Guard | Removal |
| --- | --- |
| CD-224, `package.rs` — a package may not declare a Core resource | deleted outright |
| MIR-0027, `verify.rs` — a Core-owned resource may not be a `HostResource` | `&& !(provider == "stark-std-file" && resource == "file")` |
| A11 §5 — MIR owns a resource's only close | early `continue` on the same string pair |

Together those put `file` on the `HostResource` path for selected rules while it kept legacy
direct-close semantics: **one resource name, two MIR representations, two destruction paths.** That
is the half-migration SELECT-C exists to refuse and that CD-235's `partially_migrated_core` was
written to catch. The in-code comment beside the first exemption argues specifically against
named exemptions, because one "would still exempt a program that HAD migrated, which is the very
state the guard exists to catch" — a string pair is strictly weaker than the form it rejects.

**All three guards are restored, and `a_package_may_not_declare_a_core_resource` is restored to its
original assertion** (it had been inverted to `a_package_may_declare_the_file_resource_for_stark_io`).
`io_minimal_executes_from_source_through_stark_io_package` is `#[ignore]`d with the reason on it.

**What unblocks it:** migrating `file` off the legacy path WHOLLY — Route B's representation and
lifecycle work. A complete migration is already permitted; only the partial one is refused. When it
lands, delete the attribute; nothing in `stark-io` needs to change. Recorded in
`stark-io/BLOCKERS.md`, whose "Closed in the minimal native slice" heading was corrected to
"Written" — it claimed a closure the guards do not allow.

Also in this change, from the same work: cross-file item resolution in `native_build.rs`
(`resolve_resource_items` read spans against the entry file), a non-panicking `span_text`, `pub` on
the synthesized resource nominal, and the `stark-file` status vocabulary. The `IOError::` labels in
that vocabulary name variants Core's `IOError` does not have; they are consumed only as text in a
generated-code comment (`emit_provider.rs`), so this is a naming defect, not a conformance break —
left as-is and recorded here rather than fixed silently inside another change.

EVIDENCE: `a11_host_resource` 38/38, `c788_provider_api_manifest` 10/10, `c788_starkc_build` 4 passed
/ 1 ignored, `c64_platform_matrix` 15/15. fmt clean, clippy clean.

## CD-289 — a guard test matched the layout the guard's own decision introduced (2026-07-31)

**CI had been red for four consecutive commits (CD-284, CD-285, CD-286, CD-287) on one test**, on
linux-x64, macos-arm64 and windows-x64. Not one of those changes caused it in the sense the run
implied, and the switch under test worked correctly throughout.

`portability_installed_runtime_requirement_refuses_the_checkout_fallback` asserted that under
`REQUIRE_INSTALLED_RUNTIME` no attempted path ends with `starkc/stark-runtime` — an accurate way to
name the checkout fallback until **CD-284 introduced the installed MIRROR layout**,
`<prefix>/lib/stark/starkc/stark-runtime`, which mirrors the repository precisely so that the runtime
crate and the provider crates resolve `stark-provider-abi` to one path. That legitimate installed
candidate ends with the same two components, so the guard began rejecting the very layout the
decision it guards had just added.

Fixed at the assertion, which now compares against the checkout path itself
(`<starkc manifest dir>/stark-runtime`) rather than a suffix, plus a check that the installed
locations are still attempted and reported. A suffix was always the wrong instrument for identifying
one specific directory.

EVIDENCE: `c64_platform_matrix` 15/15 locally; the three-platform result is the CI run on this commit.

## CD-287 — every `MirTy` predicate that asserts a property is now exhaustive (2026-07-31)

**Generalises CD-240 from the instance to the shape.** CD-240 found `MirTy::HostResource` classified
`Copy` by a `_ => true` arm and fixed that arm, while recording the real finding in its message:
"THIRD TIME A MirTy CATCH-ALL HAS SWALLOWED THIS VARIANT." By the time A11 was working the count was
six, each found by an e2e observing generated code, none by a test — because the type checker cannot
notice a new variant falling into a wildcard.

**The rule applied.** A wildcard is safe when its arm DECLINES to handle a type (`unsupported(...)`,
`unreachable!`, "don't fold this"). It is unsafe when the arm ASSERTS A PROPERTY — `Copy`, `Noop`,
"needs no drop", "carries no borrow" — because it then makes that claim on behalf of every variant
nobody has classified yet, silently and with the suite green. Twelve predicates were of the second
kind and are now exhaustive; the ~40 decline-shaped wildcards in `lower.rs` are deliberately
untouched.

| Site | Was |
| --- | --- |
| `mir/mod.rs` `TypeContext::is_copy` | `_ => true` |
| `mir/lower.rs` `FnLowerer::is_copy` | `_ => true` |
| `mir/lower.rs` `ty_needs_drop` | `_ => false` |
| `mir/lower.rs` `ty_has_user_drop_guarded` | `_ => false` |
| `mir/lower.rs` `ty_mentions_user_nominal` | `_ => false` |
| `mir/lower.rs` `ty_carries_ref` | `_ => false` |
| `mir/verify.rs` `may_need_drop` | `_ => false` |
| `mir/verify.rs` `mir_needs_drop` | `_ => false` — **latent defect, see below** |
| `mir/drop_plan.rs` `plan_for` | `_ => Ok(DropPlan::Noop)` |
| `backend/…/emit_types.rs` `ty_carries_reference` | `_ => false` |
| `backend/…/emit_types.rs` `ty_contains_ref` | `_ => false` |
| `backend/…/emit_types.rs` `nominal_needs_lifetime` | `_ => false` |

**THE SEVENTH SWALLOWED INSTANCE, AND THE FIRST FOUND BY THE COMPILER.** `verify::mir_needs_drop`
was still classifying `MirTy::HostResource` as needing no drop. The verifier therefore held two
copies of "does this need dropping" that disagreed about resources — `may_need_drop` said true,
`mir_needs_drop` said false — with nothing in the suite distinguishing them. Every previous instance
of this defect was found by an e2e observing generated code, after a leak; this one was found by
making the match exhaustive, which is the entire argument for doing it.

**WHAT IT GOVERNED IS NARROWER THAN THAT DISAGREEMENT SUGGESTS, and the first draft of this entry
overstated it.** `mir_needs_drop` has exactly ONE consumer: V-COPY-1's rule that `VecClear` requires
a non-droppable element type, because clearing discards elements without running their glue. It does
not participate in the `Drop` terminator path at all — that path runs through `drop_plan::plan_for`
and `may_need_drop`. So the wrong answer was **latent, not active**: reachable only through a
`Vec<HostResource>`, where `clear()` would have discarded live handles without closing them, which
is exactly the leak MIR-0016 exists to prevent. A real defect and the right fix, but it was not
mislowering resource drops today, and this entry should not be read as saying it was.

**A second behavioural fix:** `ty_contains_ref` did not recurse into `MirTy::Core`'s arguments, so a
`Vec<&T>` was reported reference-free. Also surfaced by exhaustiveness.

**Three answers the wildcards were hiding, now written down rather than decided quietly:**
- `ty_needs_drop`'s `Core` arm is asymmetric — `VecIter`/`KeysIter`/`Iter` need glue,
  `CharsIter`/`SplitIter`/`ValuesIter`/`MapIter`/`FilterIter` do not. Preserved exactly as it stood.
  Whether the second group is right is a question for iterator lowering, and it is now visible.
- `ty_carries_ref` (lowering) and `ty_carries_reference` (backend) disagree on `FnPtr`: the backend
  descends into params/return, lowering calls every fn value borrow-free. Defensible — a Rust
  `fn(&T)` is higher-ranked and needs no lifetime parameter, which is all this guards — but the two
  copies had never been checked against each other.
- `drop_plan::plan_for` on `String` returns `Noop` because the backend lowers it to a Rust `String`
  whose own destructor reclaims it. Previously that answer came from the wildcard.

**The duplication is still the defect, and it is worse than CD-240 recorded.** Twelve sites are
twelve implementations of four rules: "is Copy" lives in two places, "needs drop" in **four**
(`lower::ty_needs_drop`, `lower::ty_has_user_drop_guarded`, `verify::may_need_drop`,
`verify::mir_needs_drop`), "carries a reference" in three. Each has historically been corrected
separately, after a leak, and the `mir_needs_drop` finding above is what a fourth undiscovered copy
costs. Exhaustiveness makes the next omission a compile error at every copy; it does NOT make the
copies agree — `ty_carries_ref` and `ty_carries_reference` still disagree on `FnPtr`. Unifying them
is a design change and is deliberately NOT this one, but it now has a concrete cost attached.

**Why now, and not after C7.8.** Route B (`OwnedResourceHandle`, MIR-owned exactly-once close,
`resource_type` on `HandleOut`) reshapes `MirTy`. That is the exact event that has cost six silent
leaks. This converts the seventh into a compile error before the variant is added, not after.

**BEHAVIOURAL QUALIFICATION, per the owner's ruling that an implementation-predicate fix does not
close a disagreement.** Two tests in `a11_host_resource.rs`, deliberately separate because they prove
different things:
- `vec_clear_on_a_host_resource_element_is_rejected` — the regression guard, placed at the corrected
  arm's one real consumer. `VecClear` over `Vec<HostResource>` must raise MIR-0016. **This fails on
  the pre-CD-287 code**, which is what makes it a regression test rather than a restatement.
- `the_verifier_accepts_a_drop_emitted_for_a_host_resource` — the anchor the ruling asked for: with a
  close recorded, a `Drop` terminator on a resource local is accepted by the real verifier, not
  merely planned. It passes before AND after CD-287, because the `Drop` path never consults
  `mir_needs_drop`. Its doc comment says so explicitly so it is not later mistaken for the guard.

The inverse guard the ruling suggested already exists and was left alone:
`dropping_a_resource_with_no_recorded_close_fails` (a resource with no close must not plan) and
`rejects_vec_clear_on_droppable_element` in `mir_verify.rs` (the same rule for a user `Drop` type).

EVIDENCE: `cargo check --lib` clean across all twelve sites — which for this change is the load-
bearing check, since a missed variant is a compile error and nothing else would report one.
`cargo test --test a11_host_resource`: **38 passed / 0 failed**, including both new tests. `cargo fmt
--check` clean; `cargo clippy --test a11_host_resource` clean. Full workspace verification left to
CI: the shared checkout currently holds a parallel session's in-flight WP-IO.1 edit that does not
compile (`c788_provider_api_manifest.rs` calls an undefined `pkg`), so a workspace run right now
would report their breakage, not this change's.
FILES: starkc/src/mir/{mod,lower,verify,drop_plan}.rs,
starkc/src/backend/generated_rust/emit_types.rs, starkc/tests/a11_host_resource.rs (its doc comment
described the wildcard in the present tense), COMPILER-STATE.md.
NOT MINE, PRESENT IN THE SAME SHARED CHECKOUT: `stark-io/`, `starkc/src/provider_registry.rs` and
`WP-IO.1-Minimal-Native-File-IO.md` belong to a parallel session's WP-IO.1 work and must not be
staged with this change. The `native_build.rs` refactor that was also in this tree was that
session's, and landed as CD-286 (`manual_find` clippy fix) while this entry was being written —
which is why this is CD-287.

## Gate C9 — OPEN (2026-07-31)

Active WP: C9 Part A closeout.

Part A is complete for C9.0 baseline/governance, C9.1 extension-isolation conformance, and C9.2
tensor/ONNX provider map. Part B is blocked pending second-artifact evidence; no provider
generalisation is authorised from ONNX alone.

Current policy recorded for C9.1: Core-only is the default; `tensor` must be explicitly enabled;
unknown and duplicate extension configuration is rejected at CLI/internal/LSP configuration
surfaces.

## DEV-012 — interactive editor validation, partially recorded (2026-07-31)

The first interactive record this deviation has ever had. It has been open since Gate C1 with the
text "VS Code extension UI never interactively verified", and what closed that gap is a person
using the editor rather than another protocol test.

**Setup.** Extension `starklang.stark-language@0.2.0`, built from this tree with esbuild, packaged
and installed with `--force`. Compiler `662842c`, binaries installed to `~/.local/bin`
(`stark`, `starkc`, `starkide`). VS Code 1.130.0 on macOS 26.5.2 arm64. Workspace
`~/Desktop/stark-extension-test`, a real STARK package that checks and runs.

**One thing worth carrying forward:** the extension defaults `stark.compiler.path` to `starkc`, and
VS Code launched from Finder does not inherit a shell `PATH` — so a `~/.local/bin` install is
invisible to it unless the setting is given an absolute path. The test workspace pins it. Anyone
validating from a fresh install will hit this first.

**Confirmed by the owner, interactively:**

| Feature | Result |
| --- | --- |
| Hover | works |
| Go-to-definition | works |
| Find-references | works |

**Not exercised, and therefore still unverified in an editor:** rename, diagnostics (on save and on
type), formatting, completion, signature help, document symbols, semantic tokens. Each is covered by
protocol tests only, which is what DEV-012 exists to distinguish from.

**Gate C8 remains CANDIDATE-COMPLETE, and closing it is the owner's call.** Its exit report names
missing interactive validation as the single reason it is not closed. That reason is now partly
answered: the three core navigation queries are confirmed against real compiler analysis. Whether
three of ten features is the record the gate's claim requires is a governance decision, not one this
entry makes.


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
- CD-006 [2026-07-18, WP-C1.5] **SUPERSEDED 2026-07-26 by CD-139 — succession of authority, NOT
  reversal on the merits. Do not cite this decision for float behaviour.** It arbitrated wording in
  `03-Type-System.md` that WP-C2.9 replaced the same day (08:47 → 17:29) with the explicit paired
  rules NUM-INT-DIV-001 (integer zero division traps) and NUM-FLOAT-OP-001 (floating zero division
  does not). The sentence it read is gone from the spec; its integer half survives under
  NUM-INT-DIV-001. Original entry follows.
  Resolved a spec-internal tension in `03-Type-System.md`'s Numeric
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
Canonical ledger (full structured entries): the file now carries **108 distinct numbered
deviations** as of 2026-08-02, counted as unique `## DEV-NNN` headings (DEV-121 has two — an
original and an UPDATE — and is counted once). CD-334 added six. NOTE: this line previously read
"97 numbered deviations as of 2026-08-01", which did not match the file then either (102 by the
same count); the discrepancy predates CD-334 and is recorded rather than silently rewritten,
because whichever convention produced 97 may be the intended one. Path:
`starkc/docs/conformance/KNOWN-DEVIATIONS.md`. The per-deviation narrative that used to live in
this file (seed list + WP-C1.1/C1.2/C1.3 addition sections) is archived verbatim in
`STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md` (CD-020); the ledger remains the
single source of truth.

Open as of 2026-08-02. Entries DEV-005…DEV-017 are long-standing and unscheduled, and no open
deviation belongs to the C4 track. **DEV-134…DEV-139 were opened 2026-08-02 by CD-334 and are new,
not long-standing** — three of them are soundness gaps and none has an owning gate yet.
- DEV-005 — `starkc` vs `stark` check/run warning-gating drift. Open, unowned since Gate C1.
- DEV-011 — doc comments are lexer trivia, not AST/HIR metadata. Unscheduled; needs a scoped
  proposal.
- DEV-012 — VS Code extension UI interactively verified **in part** (2026-07-31). Hover,
  go-to-definition and find-references were exercised by the owner in a real VS Code session and
  behaved. **Rename, diagnostics-on-save/on-type, formatting, completion, signature help, document
  symbols and semantic tokens were NOT exercised** and remain protocol-tested only. Open for that
  remainder; owner: post-C8 editor validation.
- DEV-017 — 39 of 59 legacy coverage rules still lack function-level positive/negative evidence
  classification (tooling exists; classification unscheduled).
- DEV-134 — CLOSED CD-335 (WP-DEV-134-139 Part A). `?` now requires exact error-type and
  constructor compatibility; the ruling was REJECT, not convert. Whether Core v1 should gain
  `From` conversion at `?` is a separate, still-open language-design question with no owner.
- DEV-135 — CLOSED CD-338 (WP-DEV-134-139 Part B). The move model was already field-precise; the
  defect was field IDENTITY taken from a span. No DEV-135b was filed: the precision that follow-on
  would have built already existed.
- DEV-136 — CLOSED CD-337 (WP-DEV-134-139 Part D). Move state now merges only from predecessors
  that reach the join; `loop` without a reachable `break` is deliberately still treated as
  reaching, because proving otherwise needs reachability analysis the checker lacks.
- DEV-137 — CLOSED CD-336 (WP-DEV-134-139 Part C). Condition-only borrows now end at the branch
  boundary, for `if` as well as `while`; `match` scrutinees and `for` iterators deliberately keep
  theirs.
- DEV-138 — CLOSED CD-340 as a CONFIRMED DEV-121 instance (WP-DEV-134-139 Part F). DEV-121's
  class stays OPEN, and its blind spot is now named: INV-VALUE-REP-001 checks `let` bindings, and
  a for-loop binding is not a `let`, so no loop item is covered.
- DEV-139 — CLOSED CD-339 (WP-DEV-134-139 Part E). Both the operator and trait-bound lookups now
  read the combined impl+method environment. DEV-083 is a different mechanism and remains OPEN.
- Informational, not owed a fix: DEV-SEED-008 (two hand-rolled JSON parsers), DEV-SEED-014
  (no attribute syntax — deliberate scope fact).

Closed 2026-07-31: **DEV-010** (C8 candidate closeout) — LSP hover, definition, and references
are no longer protocol stubs. They are backed by `ProjectAnalysis` semantic queries and covered by
`hover_uses_compiler_symbol_signature` and
`definition_and_references_use_resolved_symbol_identity`.
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
- **MIR-0036** [WP-COPY-CANON Phase 3, CD-311] INV-MOVE-001: a `Move` operand from a place whose
  type is `Copy`. A `Copy` type's contract is that reading leaves the source intact; `Move` empties
  it and transfers drop responsibility. Emitting both about one value lets every consumer believe
  whichever it prefers. Unconditional, with no exemption mechanism — see CD-311 for why an
  "unobservable move" escape hatch was refused. Found four latent defects on its first runs
  (DEV-124, DEV-125, DEV-127).
  **This section is stale between MIR-0013 and MIR-0036**: MIR-0014..MIR-0027 and MIR-0034/0035
  were allocated by later WPs (A1/A5/A11/A12) and recorded only in `src/mir/verify.rs`'s header
  map, which is the working registry. Reconciling them here is unscheduled and is noted rather
  than silently papered over by this entry.
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
- [x] WP-C8.2/C8.3: implement real LSP hover/definition/references (DEV-010). Closed by C8
      candidate closeout; semantic query tests pass.
- [ ] Post-C8: interactive VS Code Extension Development Host validation (DEV-012). C8 is
      candidate-complete until this record exists.
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

### WP-COPY-CANON — Phases 0–4 done, Phase 5 partial — 2026-08-01

**Reconciliation gap, stated first because it is the largest fact here.** Before this entry the
highest CD recorded in this file was **CD-294**. CD-295 through CD-306 remain unrecorded — they are
other work (package-track fixes, the HTTP substrate packages, `stark test` defects, DevOps) and
some of it belongs to parallel sessions. This entry covers **CD-307..CD-316 only** and does not
close that gap.

**The packet.** WP-COPY-CANON governs one law: *after expression typing, Copy/move behaviour — and
the runtime representation that carries it — is determined exclusively by the normalized semantic
type, never by the expression that produced the value.* It binds the checker, MIR lowering, the
native backend and each interpreter equally. Registered under CD-307 before any investigation, per
the packet's ordering rule.

**Phase 1 — the sentinel matrix (CD-308).** Six producers of a reference-typed value against six
use modes, checked on three axes: MirTy copy-eligibility, the emitted MIR call operand (asserted
from the dump, so a wrong operand fails even when runtime behaviour is green), and the runtime value
kind. A per-producer matrix rather than a regression for `bytes()`, because DEV-121 was per-producer:
`bytes()` and `as_slice()` share a normalized type, were built by different code paths, and only one
was wrong.

On its first run it failed on **CD-305's own fix**, not on DEV-121. CD-305 promoted `bytes()`'s
materialised storage into the *current* frame; correct locally, dangling the moment the view is
returned. `fn borrow_of(s: &String) -> &[UInt8] { s.bytes() }` is valid Core v1 and produced
"dangling reference". CD-305's regression tests had no escaping-view case; the matrix's
ordinary-language producer controls do.

**Phase 2 — the escape fix (CD-308) and DEV-126 (CD-313).** `promote_to_temp_place_in` takes the
owning frame explicitly. That was not sufficient: CI failed `stark-json` 9/10 on all three platforms
with "dangling reference", because `as_str` returned `Value::Str(string.clone())` — a detached copy
with no link to its origin — so the chained `c.input.as_str().bytes()` had nothing to anchor to.
`as_str` now returns `Value::Ref(receiver_place)`. Consequence: `s.as_str()` reaches builtins as a
`Value::Ref`, and `flatten_string_refs` derefs a reference argument **when its referent is a
string** — keyed on the referent's kind, not the callee's name, unlike the pre-existing
`remove`/`contains_key`/`contains` special case which only ever covered the three reported.

**Phase 3 — INV-MOVE-001 / MIR-0036 (CD-311..CD-315).** Unconditional, no exemption mechanism. It
found the same defect at seven sites across four DEVs, each invisible until a workload of the right
shape ran:

| DEV | Sites | What reached it |
| --- | --- | --- |
| 124 | for-loop desugar, both forms | any `for` loop — 12 unit tests |
| 125 | provider status→`Result`; out-slot tuple; `?`'s `Err` payload | the REST workload and C7.8 only |
| 127 | `borrow_set_receiver` | the DEV-116 HashSet corpus only |

In every case the correct idiom sat next to the defect: `assign_provider_ok` read its slots through
`read_place` then hand-built the `Move` wrapping them; `borrow_map_receiver` used `read_place` while
`borrow_set_receiver` three lines away did not. **The fix is never "write `copy`"** — a non-`Copy`
payload must still move; the defect is that the site had an opinion at all.

Two structural consequences (CD-315, DEV-128): the `Copy` rule now exists **once**, in
`mir::mir_ty_is_copy`, with the nominal case passed as a predicate — it had been two byte-identical
matches differing in one lookup, and the comment naming
`lowered_copy_classification_matches_the_type_context` as the test keeping them in step referred to
**a test that does not exist**. And `operand_move_inventory` pins all eleven `Operand::Move`
occurrences in `lower.rs` with a reason each, so a new one fails at authoring time.

**Phase 4 — `diag::resolve_span` (CD-309).** The one checked path from a span to a location; never
panics, never falls back to another source. An interim guard, not the architecture: filed as
`WP-SPAN-SOURCEID.md`, which CD-309 committed to and did not do until now.

**Two test fixtures retyped (CD-314), recorded because the distinction matters.** `mir_verify`'s
`partial_move_of_one_field_leaves_sibling_readable` and `dev117_...` hand-build MIR moving `Int32`
locals; `Int32` was incidental filler, and under INV-MOVE-001 an `Int32` move is invalid MIR on its
own account, so both failed for a reason neither test concerns. Retyped to `&mut Int32` with every
assertion unchanged. The weakening NOT done: exempting `Copy` moves in the invariant.

**Method finding, recorded because it cost the most.** Four instances of one defect reached CI one
round at a time, because each local run covered a different slice — lib suite, then four iterator
tests, then the provider workloads, then the C6 corpus. INV-MOVE-001 was correct every time; the
local evidence was too narrow for a change that constrains every lowering site in the compiler. The
compensating measures are CD-315's authoring-time inventory and CD-316's matrix chaining axis.

**Phase 5 — PARTIAL.** This reconciliation and `WP-SPAN-SOURCEID.md` are done. Not done:
qualification evidence, and the frozen-corpus question. On the latter: the new matrix and chaining
cases are plain `#[test]`s in `copy_canon_matrix.rs`, not corpus cases, so **no corpus bump may be
owed at all** — an earlier claim in this session that the corpus was "locked at 1.2.0" was wrong
(CD-069 re-pinned it to 1.3.0, and `exec_snapshots` and the generated corpus carry their own
versions). Establishing which corpus, if any, is affected is the remaining Phase 5 work.

**Still open from the packet:** INV-VALUE-REP-001, the actual class-closer for DEV-121. Not
attempted. It needs the normalized type available at interpreter binding sites, and the HIR
interpreter is largely untyped at runtime.

EVIDENCE: lib 495/495; mir_verify 51/51; copy_canon_matrix 7/7; operand_move_inventory 1/1;
c6_generated_corpus 7/7 over 170 cases; c788_lifecycle_e2e 9/9; stark-json 10/10; C7 P1 REST
workload 24/24 byte-exact HTTP cases on all three platforms. CI on develop is the outstanding judge
for CD-313..CD-316.
FILES: starkc/src/mir/{mod,lower,verify}.rs, starkc/src/{interp,diag}.rs,
starkc/tests/{copy_canon_matrix,operand_move_inventory,mir_verify}.rs, KNOWN-DEVIATIONS.md
(DEV-121..DEV-128), STARKLANG/docs/compiler/work-packages/WP-SPAN-SOURCEID.md, COMPILER-STATE.md.

### WP-COPY-CANON — CLOSED 2026-08-01 (Phase 5)

**Verdict: the packet's law is enforced in one direction each for behaviour and representation,
with the remainder filed rather than claimed.**

**Phase 5 disposition.**
- **Corpus: NO BUMP OWED, established rather than assumed.** `git diff 0bd4d54..HEAD` over
  `tests/c6-corpus/` and `exec_snapshots` is EMPTY: no corpus case was added, changed or removed.
  The packet's new tests are three plain `#[test]` files (`copy_canon_matrix.rs`,
  `operand_move_inventory.rs`, and edits to `mir_verify.rs`). The generated corpus stays at
  `EXPECTED_CORPUS_VERSION = "1.5.0"`.
  **Two version claims made earlier in this session were wrong and are corrected here**: "locked at
  1.2.0" was stale memory, and 1.3.0 is the FROZEN EXEC corpus re-pinned by CD-069 — a different
  corpus from the C6 generated one. Three corpora with independent versions is the trap; naming
  which one is meant is the fix.
- **Qualification.** `qualify-first-party-packages.py` — the exact script CI runs — passed locally
  at exit 0 over JSON, URL, Base64, Hex and UUID plus their consumers, including native builds.
  Package suites: json 10/10, percent 3/3, ascii 4/4, and mime/query/form 10/11/11 where all three
  previously had ZERO (CD-320).
- **Reconciliation.** CD-307..CD-316 recorded under CD-317; CD-317..CD-322 recorded here.
  **CD-295..CD-306 remain unrecorded** — other work, some from parallel sessions. That gap is
  restated rather than quietly closed.

**What the packet actually established.**

| Half of the law | Invariant | Status |
| --- | --- | --- |
| Copy/move behaviour follows the type | INV-MOVE-001 (MIR-0036) | ENFORCED, unconditional |
| The representation carrying it follows the type | INV-VALUE-REP-001 | NARROW — one direction, one pairing |

INV-MOVE-001 found four latent defects on its first runs: DEV-124 (for-loop desugar, both forms),
DEV-125 (provider status→`Result`, out-slot tuple, `?`'s `Err` payload), DEV-127
(`borrow_set_receiver`). In every case the correct idiom sat beside the defect — `assign_provider_ok`
read its slots through `read_place` then hand-built the `Move` wrapping them; `borrow_map_receiver`
used `read_place` while its sibling three lines away did not.

INV-VALUE-REP-001 is narrow deliberately and DEV-121 is recorded **NARROWED, not class-closed**,
with residual exposure named: `&T` for scalar `T`, and the `Str`/`String` duality. Deferred by owner
direction to `WP-VALUE-REP-TOTAL.md`.

**Structural changes that outlive the packet.**
- The `Copy` rule exists once (`mir::mir_ty_is_copy`), not twice. The comment claiming a test kept
  the two copies in step named a test that **does not exist** (DEV-128).
- Structural equality exists once (`values_equal`), not in four places with the `Str`/`String`
  pairing in only one of them (DEV-130).
- `operand_move_inventory` pins all eleven `Operand::Move` sites in `lower.rs` with a reason each,
  so the next one fails at authoring time rather than when a workload of the right shape runs.
- The matrix crosses producers with each other, not only with use modes (CD-316) — the axis whose
  absence let DEV-126 reach CI.

**Method finding, recorded because it cost the most.** Four instances of one defect reached CI one
round at a time, because each local run covered a different slice: lib suite, then four iterator
tests, then the provider workloads, then the C6 corpus, then `gate4a_prelude_traits`. The invariants
were correct every time; the local evidence was too narrow for changes that constrain every lowering
site and every binding site in the compiler.

**Two defects were introduced by this packet's own fixes and are recorded as such**: CD-305's
escaping-view flaw (found by the matrix on its first run) and DEV-131's over-broad string flattening
(which broke `take`, one commit after its own DEV entry criticised name-keyed derefs).

FOLLOW-UPS FILED, NOT PENDING: `WP-VALUE-REP-TOTAL.md` (owner-deferred),
`WP-SPAN-SOURCEID.md` (CD-317). Neither blocks other work.

### CD-295..CD-306 — backfill of the gap restated by CD-317 and CD-323 — 2026-08-01

Recorded from the commits themselves, not reconstructed. These sit between the last entry the
ledger carried (CD-294) and WP-COPY-CANON's registration (CD-307). They are not one work package:
they are a Windows-encoding fix, a package-tooling batch, a DevOps change, and two compiler defects
that the packages exposed. Grouped by what they were, in commit order.

**Windows encoding — CD-295, CD-296.**
- **CD-295** — `qualify-first-party-packages.py` decoded UTF-8 and then re-encoded to cp1252.
  `13c4eb0` had fixed the READ; the WRITE failed one line later, so a STARK program printing an
  emoji killed the script REPORTING its result while the program itself had already emitted correct
  bytes and passed. `sys.stdout`/`stderr` reconfigured to UTF-8 with `errors="replace"` — a
  reporting path must not fail a qualification run over a byte it cannot render, and the comparison
  happens on decoded text so substitution cannot mask a real mismatch. Verified by forcing
  `PYTHONIOENCODING=cp1252`.
- **CD-296** — the §9.5 output-contract test used `héllo wörld`, every character of which is present
  in cp1252, so a host round-tripping stdout through the console codepage would still have passed.
  Replaced with `😀` (4-byte UTF-8, no cp1252 representation). **The compiler was right and its test
  was incomplete** — same shape as CD-276.

**Package tooling — CD-297, CD-297a, CD-298, CD-300, CD-302.**
- **CD-297** — `stark-random` plus an EXECUTION test, which is the point: three compiler-side tests
  and four native crate tests all passed while the package's STARK code could not lex, had no
  imports, had never compiled `fill_bytes`, and trapped in `next_u64` on its second call. The last
  is a **language-level finding worth carrying**: STARK traps on integer overflow in every build
  mode, and a shift discarding set bits IS an overflow — so every wrapping-arithmetic algorithm
  (hashes, PRNGs, checksums, bit mixers) needs explicit masking, and the failure mode is a runtime
  trap rather than a compile error. Also corrected `c63c_iterators`: **CD-293's E0106 was
  redundant** — E0100 had always refused moving a non-`Copy` value out of an indexed place, and
  E0106 was reasoned from a MIR message without checking a source program could reach it.
- **CD-297a** — `assert_eq!(x, false)` is `clippy::bool_assert_comparison` under `-D warnings`. It
  failed TWO jobs: the lint job and the C6.4 qualification, whose gate runs clippy — **a single lint
  failure invalidates qualification evidence, not just the lint step.**
- **CD-298** — `stark-io` docs four commits out of date. Established that the recorded "library
  packages cannot test themselves" blocker is **narrower than written**: `stark test` already works
  on a library package (parse, resolve, type-check, run through the interpreter, no `main`); what a
  library cannot do is be NATIVELY qualified without an artificial entrypoint. Docs only.
- **CD-300** — `stark test` never synthesized `provider_api`, so every generated `*_raw` was E0200
  and a provider-bound package failed before discovering one test (`stark-io`: 18 undefined
  variables). Added, with `target::host_triple_of_this_build()` derived from `std::env::consts`
  rather than probing `rustc` — testing runs through the interpreter and compiles nothing. Also:
  `stark-random/stark.lock` was malformed, and `stark-random-native` depended on `getrandom` from
  crates.io, which broke every runner under `cargo generate-lockfile --offline`.
- **CD-302** — `stark test` PANICKED on any package with a dependency: `item_text` sliced the root
  file with a dependency's span (`byte index 2147483648` — 2^31, a synthetic span). Every package
  depending on another was untestable, which is most of them; this is why reviewing the package
  batch was impossible before it. `item_text` returns `Option` and callers skip an item whose span
  does not fit its file — not clamped, not guessed. Also added `# Safety` to all 37 unsafe extern
  fns across four provider crates; **CI lints only the `starkc` workspace, so none was checked.**

**DevOps — CD-301.** `develop` branch flow, so a red run cannot land on `main`. The reasoning worth
keeping: `ci.yml` has eleven jobs, three of them matrices, so real check names are generated —
naming them in a protection rule **fails OPEN**, because a renamed matrix entry is simply not found
and GitHub reports the rule satisfied. One `ci-complete` aggregator with `if: always()` and explicit
`needs.*.result` checks is the only name protection needs.

**Compiler defects the packages exposed — CD-303, CD-304, CD-305, CD-306.**
- **CD-303** — **PAT-BIND-001 was never enforced.** `Ty::Ref` fell through every classifier, giving
  the worst combination: exhaustiveness demanded a wildcard on a match that already covered every
  variant (E0303 pointing at the wrong problem), and the `_` arm added to satisfy it then ABSORBED
  EVERY CASE at run time. So the obvious response to a misleading error produced a function silently
  returning the wildcard's answer for every input. Now rejected with help naming `match *r`.
  **Deliberately not done:** making `match r` work by peeling the scrutinee type — that is Rust
  match ergonomics, contradicts PAT-BIND-001, and is a language-design proposal requiring
  coordinated checker/MIR/interpreter change. Caught only when opening the spec to document it.
- **CD-304** — landed Gemini's five HTTP-substrate packages with what does and does not work
  recorded: ascii 4/4 and percent 3/3 passing; mime, query and form with **zero tests and failing
  consumers**, characterised but not fixed. (Their tests were written later, under CD-320.)
- **CD-305** — `String::bytes()` returned an owned `Value::Vec` for a declared `&[UInt8]`, so
  passing the view consumed it. **Which engine was wrong was established, not assumed**: emitted MIR
  was `copy` on both calls, so the checker and MIR were right and the HIR interpreter alone was
  wrong. Predates the session, verified by A/B against a compiler built at `77d763e`. This became
  DEV-121 and the whole of WP-COPY-CANON; its own fix later proved incomplete for escaping views
  (CD-308) and for chained producers (CD-313).
- **CD-306** — a dependency's runtime span was rendered against the root consumer's source: a fault
  inside `stark-mime` reported at line 31 of a 21-LINE consumer. Two causes, one per layer —
  `cmd_run` never read `error.file`, and `SourceFile::line_col` CLAMPS an out-of-range offset so a
  foreign span produces a plausible WRONG location rather than a failure. **It cost real
  investigation time**: the wrong file sent the first characterisation of CD-305 to the wrong shape
  entirely. This became DEV-122, later given a checked resolution path (CD-309) and a filed
  correction (`WP-SPAN-SOURCEID.md`).

**Why this gap existed.** These twelve were pushed across a single long session in which the ledger
was not updated once. The lesson is the same one CD-317 recorded for CD-307..CD-316 and is now
stated for both: `COMPILER-STATE.md` is the live status source, and a session that pushes twelve CDs
without touching it leaves the ledger describing a compiler that no longer exists.
