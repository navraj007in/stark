# WP-FMT-001 — closure report

**Status: COMPLETE. Ledger: CD-380, corrected by CD-381, completed by CD-382.**
**Base: `70aca83`. Design note: `WP-FMT-001-DESIGN.md`.**

> **Correction, 2026-08-04 (CD-381).** This report originally declared FMT-0 through FMT-5 closed
> and stated that the MIR runtime-surface version had moved. **It had not** —
> `MIR_RUNTIME_SURFACE` was still `0.1-A13`, and §6 below asserted a change that did not exist. An
> external review found that, four further defects, and that declaring closure while DEV-173 blocks
> ordinary string-literal expressions inside fields was an overclaim. All of the code defects are
> repaired in CD-381; the scope statement is corrected here rather than by editing the original
> claims away. **Formatting is not "complete ordinary-expression interpolation" until DEV-173 is
> solved.**
>
> **DEV-173 was solved in CD-382**, and Tier-1 qualification came back green on `7e41a1e`. Both of
> the reasons this report was not closable are gone; the status line above reflects that, and the
> history of how it got there is left standing.
>
> **A third wording correction, and the pattern behind all three.** CD-382 first described the
> result as "complete ordinary-expression interpolation". It is not:
> `f"{lookup(\"a\nb\")}"` is a valid ordinary expression and is refused. The accurate statement is
> the one in §1 — complete for the defined Core v1 interpolation surface, with data-bearing escapes
> inside a nested literal an explicitly rejected extension. Together with §6's version claim and the
> `println(f"...")` evidence gap, that is three summaries in one work package written from what the
> change was *for* rather than from what it *does*, each caught by review. The restriction now lives
> in the grammar (LEX-FORMAT-004), not only in a defect record.

```stark
let message = f"pkg={name} n={count:04} r={ratio:.2} ok={ok}";
```
```text
pkg=stark n=0042 r=0.76 ok=true
```

---

## 1. Final syntax and grammar

```text
FORMAT_STRING := 'f"' (CHAR | ESCAPE_SEQUENCE | '{{' | '}}' | FIELD)* '"'
FIELD         := '{' EXPRESSION [ ':' FORMAT_SPEC ] '}'
FORMAT_SPEC   := [ [FILL] ALIGN ] [ SIGN ] [ '#' ] [ '0' ] [ WIDTH ] [ '.' PRECISION ] [ TYPE ]
ALIGN         := '<' | '>' | '^'          SIGN := '+' | '-' | ' '
TYPE          := 'b' | 'o' | 'x' | 'X' | 'f'
```

Normative in **LEX-FORMAT-001/002/003** (01-Lexical-Grammar), **EXPR-FORMAT-001**
(02-Syntax-Grammar) and **STD-FORMAT-002…005** (06-Standard-Library).

## 2. Architecture

```text
f"..."  ──one token──►  scanner (format_syntax)  ──►  segments
                                                        │
                                    literal ────────────┤
                                    field ── expr span ─┴─► lexer::tokenize_range
                                                             └─► ordinary expression parser
                                                                   │
        ast::ExprKind::FormatString ◄──────────────────────────────┘
                    │
        hir::ExprKind::FormatString  ──► typecheck: Display + spec/type compatibility
                    │
        ┌───────────┴────────────┬─────────────────────┐
   HIR oracle              MIR lowering          native codegen
        └───────────┬────────────┴─────────────────────┘
                    ▼
        stark_runtime::fmt_spec   ← the ONE implementation of the rules
```

`starkc` already depended on `stark-runtime`, and native binaries already link it, so the shared
core needed no new crate and creates no cycle.

## 3. Lexer and parser representation

One token (`TokenKind::FormatStr`), scanned like a cooked string. The parser splits it and lexes
each field over its **own byte range in the original file** (`lexer::tokenize_range`), so spans stay
real — `tests/span_integrity.rs` asserts a field's expression is spanned inside its literal. Depth
is tracked over `(`/`[`/`{` and escapes are consumed whole, so a struct literal's `:`/`}`, a path's
`::`, and `\u{1F600}`'s braces all stay where they belong.

## 4. AST / HIR / MIR

* `ast::{FormatSegment, FormatSpec, FormatAlign, FormatSign, FormatKind}`,
  `ast::ExprKind::FormatString`; `Ast::marks`/`retag_spans_since`.
* `hir::FormatSegment`, `hir::ExprKind::FormatString`.
* MIR: five runtime operations — `FmtPad`, `FmtIntSpec`, `FmtUIntSpec`, `FmtFloat64Spec`,
  `FmtFloat32Spec`. Output construction reuses `StringNew`/`StringPushStr`/`StringAsStr`.

Adding the variants forced `emit_runtime.rs`, `mir/verify.rs` and `mir/interp.rs` open — those
matches are exhaustive — and the new AST/HIR variants forced nine passes open the same way.

## 5. Runtime surface

`stark_runtime::fmt_spec`: `Spec::pack`/`unpack`, `pad`, `scalar_len`, `fmt_int_spec`,
`fmt_uint_spec`, `fmt_float64_spec`, `fmt_float32_spec`, `fmt_pad_spec`.

## 6. Versions

**Corrected in CD-381.** The five new `RuntimeFn` variants extend the closed, versioned runtime
contract, and `MIR_RUNTIME_SURFACE` must advance with them. It did not: this report claimed a
version change that had not been made. `MIR_RUNTIME_SURFACE` is now `0.1-A14`, covering **twelve**
unversioned additions — the seven `Fmt*` members CD-378 added as well as these five — so a consumer
built against A13 rejects an A14 program before consuming a body, which is what V-SURFACE-1 exists
for. `MIR_VERSION` stays `0.3`: this is additive runtime surface, not a structural MIR change.

## 7. Changed files

| file | change |
| --- | --- |
| `stark-runtime/src/fmt_spec.rs` | new — the shared rules, 16 unit tests |
| `stark-runtime/src/lib.rs` | module registration |
| `src/format_syntax.rs` | new — segment scanner + spec grammar, 12 unit tests |
| `src/lexer.rs` | `FormatStr` token, `format_string`, `tokenize_range` |
| `src/ast.rs` | segment/spec types, `ExprKind::FormatString`, arena marks |
| `src/hir.rs` | `FormatSegment`, `ExprKind::FormatString` |
| `src/parser.rs` | `format_string`, `subparse_expr` |
| `src/resolve.rs`, `src/flow.rs`, `src/typecheck.rs`, `src/interp.rs` | lowering, definite assignment, checking, evaluation |
| `src/mir/{mod,lower,verify,interp}.rs` | five ops; `lower_format_string`, `render_format_field` |
| `src/backend/generated_rust/emit_runtime.rs` | native emission |
| `src/analysis/query.rs`, `src/ast_dump.rs`, `src/formatter/printer.rs` | index, dump, verbatim round-trip |
| `tests/wp_fmt_001_interpolation.rs` | new — 39 tests |
| `tests/span_integrity.rs` | field spans |
| spec: `01`, `02`, `06` + regenerated `STARK-Core-v1.{md,html,pdf}` | grammar and semantics |
| `STARKLANG/tests/spec-fixtures/manifest.toml` | new fixture triaged; corpus 113 → 114 |
| `COMPILER-STATE.md`, `KNOWN-DEVIATIONS.md`, `packages/stark-fmt/README.md` | records |

## 8. Positive matrix (39 tests; every case three-engine, stdout pinned in the test)

headline example · plain/simple fields · escaped braces · `\u{...}` not a field · arithmetic, call,
method, field, index, parenthesised and struct-literal expressions · user `Display` · interpolation
nested inside a `Display::fmt` body · generic `T: Display` · borrow of an owned parameter, a
non-`Copy` value and an affine value · evaluate-once and left-to-right by side effect · all three
alignments, custom fill, odd centring, scalar-counted width · zero-padding, `+`/` `/`-`,
`b`/`o`/`x`/`X`, `#` prefixes, `#010x`, negative in base · every integer width · `.N`, `.Nf`, `.0`,
half-to-even, `-0.0`, `NaN`/`inf`/`-inf`, `Float32` declared width, width+sign+precision · bool,
unit, char, `String`, `&str` · `print`/`println`/`eprint`/`eprintln` on both streams, **both via
`.as_str()` and in the direct `println(f"...")` form** · comments inside fields, including a nested
block comment and `}`/`:` inside one · no package dependency required · debug-vs-release native
agreement.

## 9. Negative matrix

unterminated field · unmatched `}` · empty and blank field · `{{}` · unknown type char · two-char
fill · alignment without width · width overflow (LIMIT-FMT-WIDTH) · precision overflow
(LIMIT-FMT-PRECISION) · escape inside a field (DEV-173) · type without `Display` · generic without a
`Display` bound · numeric mode on a generic `Display` · hex on `String` · precision on `Bool`,
`Int32` and `String` · binary on `Float64` · sign on `String` · **inert flags: `#` without a base,
`#` on a float, `0` without a width, `f` without a precision, `0` combined with an alignment or a
fill**. **No malformed input panics the compiler** — asserted directly.

## 10. Ownership and evaluation evidence

`a_field_borrows_an_affine_value` prints `first=…`, `second=…`, then `released`, then `id=5`: the
`Drop` line lands *after* both interpolations, so the value was alive while being formatted and was
destroyed exactly once, by the later `consume`. `fields_evaluate_once_and_left_to_right` prints
three side-effect lines in order and once each — a second evaluation to size a field would duplicate
a line, and a right-to-left walk would reorder them.

## 11. Float rounding evidence

`0.125 → 0.12` and `0.375 → 0.38` — exact binary halves rounding to the even digit, which
round-half-away-from-zero would render `0.13`/`0.38`. Pinned in the STARK suite and again at the
Rust level.

## 12. Four-engine parity

Every positive case goes through the shared comparator (HIR oracle, MIR interpreter, native debug),
comparing stdout bytes, stderr, exit status, Drop log and returned observation, with the expected
text stated in the test. `release_and_debug_native_agree` adds the fourth configuration.

## 13. Generated-source evidence

`native_routes_through_stark_formatting` builds a program using `{name}`, `{count:04}`,
`{ratio:.2}` and `{count:#x}`, then requires the generated crate to contain
`stark_runtime::fmt_spec::` and **not** `format!`, `write!`, `writeln!`, `std::fmt::Display`,
`std::fmt::Debug` or `#[derive(Debug`.

**Narrowed honestly:** inside `fmt_spec`, fixed-point digit production calls Rust's numeric decimal
conversion (`{:.*}` on an `f64`), and integer digits are produced by an explicit radix loop written
here. The claim is that *generated STARK program code* cannot select Rust formatting, and that all
three engines share one implementation — not that the implementation avoids the host's
number-to-decimal conversion. Replacing that is a separate design and performance decision.

## 14. Tier-1 qualification

**Still outstanding.** CI runs the three Tier-1 lanes on push and is the qualifying evidence; the
first push of this work failed CI on two clippy lints before reaching them, so no Tier-1 result has
been observed yet. The suite is platform-independent by construction (no host formatting, no locale,
scalar-counted width), but that is an argument, not a result, and it is not counted as one.

## 15. Remaining exclusions

No multiline interpolated form (**STARK has no multiline string literal to prefix** — §2.5 forbade
inventing one). No raw interpolated form (`rf"..."`), deferred as §2.5 permits. No string
truncation, no dynamic width or precision, no positional or named arguments, no variadic `println`,
no localization, no scientific/hex-float/percentage modes, no `Debug`, no method-form `to_string`.
The source formatter reprints an interpolated literal verbatim rather than re-formatting its
embedded expressions — §19 permits this and asks that it be recorded; it is.

## 16a. CD-381 correction packet

An external review of `987369b` found six defects. All are repaired:

| # | defect | repair |
| --- | --- | --- |
| 1 | `MIR_RUNTIME_SURFACE` never advanced, and §6 claimed it had | `0.1-A14`, covering CD-378's seven `Fmt*` members as well as these five |
| 2 | the field scanner did not know comments, so `f"{v /* } */}"` mis-scanned | comment-aware scanning, with NESTED block comments, mirroring the lexer's own rule |
| 3 | the verifier typed the specification operands but did not require them to be constants | **MIR-0037** — both operands must be constants, the word must decode to a valid specification, unused bits must be zero, and width/precision must be within LIMIT-FMT-WIDTH/PRECISION |
| 4 | inert flag combinations were silently accepted | `#` without a base, `0` without a width, `f` without a precision, and `0` with an alignment or fill are now refused |
| 5 | LIMIT-FMT-SEGMENTS was checked inside the loop, missing the trailing literal | checked once over the finished list |
| 6 | `println(f"...")` was tested only through `.as_str()` | direct form proved — and it **failed**, exposing DEV-174 |

Item 6 is the one worth dwelling on: testing the convenient form instead of the advertised one is
exactly how a gap survives a suite. `eprint`/`eprintln` had been typed `&str` since Phase 4E while
the specification declared them generic over `Display`, and no test had ever passed them anything
else.

`Spec::unpack`'s defaults for unknown align/sign/kind encodings are now unreachable in verified MIR:
MIR-0037 rejects those bit patterns before an engine can normalise them.

## 16. New defect records

* **DEV-172** — no signed type can express its own minimum (`let a: Int8 = -128;` is rejected).
  Pre-existing, unrelated to formatting; found because formatting a minimum is exactly the case
  where taking a magnitude in-width overflows. The renderer handles `i64::MIN`; the language cannot
  produce it.
* **DEV-173** — an interpolation field may not contain an escape sequence. Refused rather than
  mis-parsed, because a literal reads its value from its span and a decoded copy has none.
  **This is why the status above is "v0.1 partial interpolation" rather than closed.** The original
  acceptance matrix included `f"{choose(\"yes\", \"no\")}"`, and forms like `f"{lookup(\"name\")}"`
  and `f"{parse(\"42\").unwrap()}"` are blocked. The durable fix is for literals to carry decoded
  values in the AST/HIR rather than recovering them from source spans — an arena change that would
  improve the compiler beyond interpolation.
* **DEV-174** — `eprint`/`eprintln` took `&str` rather than a `Display` value, contradicting
  06-Standard-Library. Fixed in CD-381.

## 17. Is STARK formatting complete enough for CLI tools, structured logging and the REST server?

**CLI tools: yes.** Aligned columns, zero-padded numbers, fixed-precision floats and hex all work,
identically on every engine and platform.

**Structured logging: yes, for the line-oriented kind.** `f"level=info route={route} ms={elapsed:.1}"`
is exactly the shape, and fields interpolate without consuming the values a handler still needs.

**REST server: yes for the observable surface, no for payloads, and the distinction is not a
detail.** Interpolation performs no escaping — STD-FORMAT-005 says so normatively. Building a JSON
body with `f"{{\"name\":\"{name}\"}}"` is wrong for any `name` containing a quote or backslash, and
that is an injection defect, not a cosmetic one. Use `stark-json` for payloads and reserve
interpolation for logs, status lines and header values.

One gap worth scheduling before the REST work rather than during it: **there is no `Vec`/collection
interpolation shape** beyond whatever `Display` a type provides, so a repeated field must be built
with a loop and `stark-fmt`'s `Line`. That is a library question, not a language one.
