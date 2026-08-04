# WP-FMT-001 — design note (FMT-0)

Written before implementation, as §21 requires. Base: `70aca83`.

---

## 1. What the investigation found

| question | answer |
| --- | --- |
| string literal forms | `"..."` (cooked) and `r"..."` (raw). **There is no multiline literal.** |
| lexer prefix dispatch | `Lexer::run`, one `match` on the leading byte; `r"` is `b'r' if peek(1) == b'"'` |
| lexer entry | `tokenize(file)` / `tokenize_with_comments(file)`; `Lexer` holds `src: &[u8]`, `pos` |
| AST | arena (`Ast { exprs: Vec<ExprNode> }`), `ExprId`, every node carries a `Span` |
| parser | `Parser { tokens: Vec<Token>, pos, ast: &mut Ast, depth }`, `MAX_DEPTH = 200` |
| `Display::fmt` dispatch | CD-378/CD-379: `hir::resolved_bound_trait` → `TypeTables::bound_trait_calls` → both engines |
| primitive rendering | `stark_runtime::format` — `fmt_i64`, `fmt_u64`, `fmt_bool`, `fmt_f64`, `fmt_f32`, `fmt_char`, `fmt_unit`, shared with `print`/`println` |
| float canonicalisation | `stark_runtime::format::canonical_float` / `canonical_float32`; `starkc::interp::canonical_float` delegates |
| dependency direction | `starkc` **depends on** `stark-runtime`; native binaries **link** `stark-runtime`. Both sides already reach it. |

**§2.5 is not implementable and is excluded.** It says "if STARK already has a multiline string
literal form, the `f` prefix must be supported for that existing form" and "do not invent an
unrelated multiline-string syntax". STARK has none, so there is nothing to prefix. Recorded as an
exclusion, not a gap.

**Raw-string interpolation (`rf"..."` / `fr"..."`) is deferred**, using the escape §2.5 offers for
it: combining the prefix rules widens the lexer change without adding expressive power, since a raw
string's whole point is that `\` is literal and interpolation needs no escapes of its own.

---

## 2. Tokenization

`f"` is one more prefix arm in `Lexer::run`, scanning to the closing quote exactly as `string()`
does — same escape handling, same "unterminated string literal" diagnostic, same UTF-8 validation —
and emitting **one token**, `TokenKind::FormatStr`, spanning the whole literal.

**The interior is not tokenized by the main lexer.** A brace-nesting scanner would have to run
inside the token stream, and every downstream consumer (formatter, LSP) would have to understand a
token soup. One token keeps `f"..."` a literal to everything that does not care.

### Why the parser re-lexes rather than the lexer emitting pieces

The parser splits the token's span into segments and, for each expression segment, lexes **the
original file over that exact byte range** and parses it with the real expression parser. That
means:

* spans inside an interpolation are real file spans — a diagnostic points at the expression itself,
  not at the whole literal;
* nesting, calls, indexing, struct literals, paths and nested string literals are handled by the
  parser that already handles them, not by a second implementation;
* no source text is ever reconstructed or rewritten (§9's prohibition).

This needs `lexer::tokenize_range(file, lo, hi)` — the existing lexer with a `pos` start and an
`end` bound, so spans stay absolute.

### The segment scanner

Walks the literal body once, tracking:

* `{{` and `}}` → one literal brace, no field;
* a lone `}` outside a field → error;
* inside a field: depth over `(`/`)`, `[`/`]`, `{`/`}`, and "inside a nested string literal"
  (with `\` escape awareness);
* the **top-level** `:` ends the expression and begins the spec — a `:` at depth > 0, or the second
  half of `::`, is part of the expression;
* the **top-level** `}` closes the field.

`Point { x: 1, y: 2 }` therefore parses: the `{` raises depth, so neither its `:` nor its `}` is
seen as a separator.

---

## 3. AST and HIR

```rust
// ast.rs
pub struct FormatSpec {          // all fields compile-time constant
    pub fill: Option<char>,
    pub align: Option<FormatAlign>,   // Left | Right | Center
    pub sign: Option<FormatSign>,     // Plus | Minus | Space
    pub alternate: bool,
    pub zero_pad: bool,
    pub width: Option<u32>,
    pub precision: Option<u32>,
    pub kind: Option<FormatKind>,     // Bin | Oct | LowerHex | UpperHex | Fixed
    pub span: Option<Span>,
}

pub enum FormatSegment {
    Literal { text: String, span: Span },
    Field { expr: ExprId, spec: FormatSpec, span: Span },
}

ExprKind::FormatString { segments: Vec<FormatSegment> }
```

HIR mirrors it (`hir::FormatSegment`, `hir::ExprKind::FormatString`). Each field keeps its
expression id, its spec, its own span and the spec's span — §9's requirement, and what makes
"`{text:x}` is not formattable" point at `:x` rather than at the literal.

---

## 4. Type checking

Per field:

* **no spec, or align/width/fill only** → the expression type must satisfy canonical `Display`.
  Checked with `type_is_displayable`, the same predicate `println` uses — which routes through
  `ty_satisfies_operator_bound`/bound identity, so CD-379's rule holds and a user trait merely
  *named* `Display` cannot satisfy it.
* **`b`/`o`/`x`/`X`, `sign`, `#`, `0`** → the type must be a concrete integer primitive.
* **`.precision`, `f`** → the type must be `Float32` or `Float64`, and the declared width is kept.
* A generic `T: Display` is accepted for the default form and **rejected for numeric specs**
  (§11.5): `Display` does not prove integer formatting, and inventing a numeric trait to make it
  compile is out of scope.

---

## 5. Where the algorithms live

**`stark-runtime::fmt_spec`, one implementation, three callers.**

```text
stark-runtime::fmt_spec          ← alignment/fill/width, radix, sign, prefix, fixed precision
        ↑            ↑
        │            └──────────── generated native code (links stark-runtime)
        └── starkc (HIR interp, MIR interp)   [starkc already depends on stark-runtime]
```

No new crate and no cycle: the dependency already runs `starkc → stark-runtime`, and native
binaries link the same crate. This is the same arrangement `stark_runtime::format` already has for
`print`/`println`, which is why `fmt()` and `println` cannot disagree today.

A `FormatSpec` reaches the runtime as **two operands**: a `UInt64` bitfield and a `Char` fill. Both
are compile-time constants, so the verifier can check them as constants and no format string is
ever parsed at runtime (§12.1).

---

## 6. MIR surface — Option A plus five operations

Literal segments and the output string use what already exists (`StringNew`, `StringPushStr`,
`StringAsStr`). Five new runtime operations carry what MIR cannot express:

| op | operands | result |
| --- | --- | --- |
| `FmtPad` | `&str`, `UInt64` spec, `Char` fill | `String` |
| `FmtIntSpec` | `Int64`, `UInt64` spec, `Char` fill | `String` |
| `FmtUIntSpec` | `UInt64`, `UInt64` spec, `Char` fill | `String` |
| `FmtFloat64Spec` | `Float64`, `UInt64` spec, `Char` fill | `String` |
| `FmtFloat32Spec` | `Float32`, `UInt64` spec, `Char` fill | `String` |

Five operations, not one per syntax combination (§13's explicit warning). Everything about *how* a
value is rendered lives in the spec word; the operation only says which value family it is.

`RuntimeFn` matches in `emit_runtime.rs`, `mir/verify.rs` and `mir/interp.rs` are exhaustive, so
adding these forces every consumer to be updated.

---

## 7. Evaluation-once and ownership

Lowering emits, per field, in order:

1. evaluate the expression **once** into a temporary (or borrow the place, if it is one);
2. render — `Display::fmt` through ordinary dispatch, or a runtime op for a spec'd primitive;
3. apply padding if the spec asks for it;
4. `StringPushStr` the rendered text onto the output;
5. drop the rendered `String`; drop the temporary if we made one.

An lvalue is **borrowed**, never moved: `Display::fmt` is `&self` (STD-FORMAT-001), and the
scalar path reads by copy. That is what makes `f"{value}"` twice, then `use_value(value)`, legal.
Left-to-right order falls out of emitting fields in source order.

---

## 8. Versions that must move

* MIR runtime surface → new `RuntimeFn` variants;
* the MIR/runtime contract version and the build-cache identity input, so a cached crate built
  before these ops is not reused.

---

## 9. Limits (§17)

| limit | value | why |
| --- | --- | --- |
| segments per literal | 1,024 | bounds the AST a single token can produce |
| nesting depth in a field | the parser's existing `MAX_DEPTH` (200) | one depth rule, not two |
| maximum width | 1,000,000 | a static request above this is a mistake, not an intent |
| maximum precision | 10,000 | far past any real fixed-point need |

Each is a compile-time diagnostic, so `f"{value:999999999999}"` fails at type checking and never
reaches an allocator.

---

## 10. Slice order

FMT-1 basic → FMT-2 width/align/fill → FMT-3 integer modes → FMT-4 float precision →
FMT-5 tooling, docs, conformance. The definition of done is the whole set; FMT-1 alone is not
closure, per §22.
