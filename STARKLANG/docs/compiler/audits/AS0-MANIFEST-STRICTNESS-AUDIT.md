# AS0 — Manifest strictness audit

**Status:** COMPLETE. Read-only measurement; nothing was rewritten.
**Date:** 2026-08-06.
**Owning packet:** `WP-ARCHITECTURE-STABILIZATION.md` §5, Sprint 1 opening — manifest strictness
audit. Its purpose is to decide **AS5's shape** before Sprint 2 commits to it.
**Method:** the real in-tree parsers (`starkc::package::parse_json`,
`starkc::lsp::protocol::parse_json`) against Python 3 `json` as the RFC 8259 reference. Escape
inputs were constructed from a backslash byte rather than written inline — the first version of the
probe silently tested the *character* instead of the *escape* and proved nothing.

---

## 1. Verdict

| Classification | Required? | Basis |
| --- | --- | --- |
| Repository migration | **NO** | the checked-in corpus is strict-clean, 108/108 |
| Compatibility correction | **YES** | `package.rs` rejects valid RFC 8259 input: all `\u` escapes, all exponent numbers |
| Correctness defect | **YES — since FIXED** (DEV-182/CD-384) | the LSP parser silently corrupted valid surrogate pairs to the empty string |
| Tightening | **YES** | both parsers accept invalid input; the sets differ |

**AS5 is not a tightening.** It is a tightening *plus* a compatibility correction *plus* at least
one correctness defect that needs its own DEV record. Sizing it as "adopt a strict parser" would
have been wrong.

## 2. Corpus

108 files under `packages/` (54 `starkpkg.json`, 54 `stark.lock`; `target/` excluded).

| Parser | Accept | Reject |
| --- | ---: | ---: |
| current `package.rs` | 108 | 0 |
| strict RFC 8259 | 108 | 0 |

No checked-in manifest or lockfile depends on the current parser's leniency, and none uses a
construct the current parser cannot read. **Tightening the parser breaks no first-party file today.**
That is a statement about the corpus as it stands, not about what authors may write next — §3 is
where the exposure is.

## 3. Construct-level results

`PKG` = `package.rs`, `LSP` = `lsp/protocol.rs`, `STRICT` = RFC 8259.

| # | Construct | PKG | LSP | STRICT |
| ---: | --- | --- | --- | --- |
| 1 | trailing comma in object | accept | reject | **reject** |
| 2 | trailing comma in array | accept | reject | **reject** |
| 3 | trailing input after value | reject | accept | **reject** |
| 4 | raw U+0001 in string | accept | accept | **reject** |
| 5 | `\u0041` (escaped `A`) | reject | `A` | **`A`** |
| 6 | `\ud83d\ude00` (valid pair) | reject | **empty string** | **U+1F600** |
| 7 | `\ud83d` (unpaired) | reject | `""` | **reject** |
| 8 | `\u0000` | reject | U+0000 | **U+0000** |
| 9 | duplicate keys | last | last | **last** |
| 10 | leading-zero number `01` | accept | accept | **reject** |
| 11 | exponent `1e3` | reject | 1000 | **1000** |
| 12 | negative exponent `1.5e-3` | reject | 0.0015 | **0.0015** |

Conformance against RFC 8259: **`package.rs` 3/12, `lsp/protocol.rs` 7/12.**

**The two in-tree parsers disagree with each other on 9 of 12 constructs** (all but 4, 9, 10).
They are not two implementations of one grammar that drifted; they are two different grammars.

## 4. Findings, in severity order

### F1 — the LSP parser silently corrupts valid surrogate pairs (correctness defect)

`{"a":"\ud83d\ude00"}` parses to the **empty string** instead of U+1F600. `parse_json_string`
(`src/lsp/protocol.rs:208-215`) reads four hex digits, calls `char::from_u32`, and **discards the
escape when that fails** — which it always does for a surrogate half — with no pairing step and no
error. An unpaired `\ud83d` is accepted the same way, so invalid input is also swallowed.

This is the delta-3 cell: both parsers accept, and the value is wrong. A verdict-only audit records
it as agreement. Any editor sending a non-BMP character in a diagnostic string, file path, or
completion label loses it silently.

**Disposition:** own DEV record, own fails-before-the-repair test, repaired outside AS5's
consolidation commit.

**RESOLVED 2026-08-06 — DEV-182 / CD-384.** Repaired on its own branch under the §3 live-defect
pre-emption rule and merged to `develop`: surrogates are paired per RFC 8259 §7, and a lone or
mis-paired surrogate, a malformed hex escape and a truncated escape are now rejected instead of
swallowed. Eight tests, five of which failed before the repair. **AS5 inherits the class, not the
instance** — the remaining verdict-level gaps in F2 and F3 are untouched.

### F2 — `package.rs` rejects valid JSON (compatibility correction)

Two families, both valid RFC 8259 that a hand-written or third-party-generated manifest may contain:

- **all `\u` escapes** — `parse_string` has no `'u'` arm, so it falls to
  `_ => Err("Unsupported escape")` (`src/package.rs:166`). A package name or description containing
  any escaped character is unreadable.
- **all exponent numbers** — `parse_number` consumes only digits and `.`
  (`src/package.rs:201-209`), so `1e3` and `1.5e-3` fail with a misleading
  `Expected ',' or '}'` pointing at the exponent character.

These are not lenience to be tightened; they are input the parser *should* accept and does not.

### F3 — both parsers accept invalid JSON, in different sets (tightening)

`package.rs` accepts trailing commas, raw control characters and leading-zero numbers.
`lsp/protocol.rs` accepts trailing input after a complete value and the same raw control characters.
Neither set is a subset of the other, so consolidation must pick a single grammar rather than adopt
either parser's current behaviour.

### F4 — divergence is the argument for AS5, and it is larger than assumed

9 of 12 constructs differ between the two parsers. `package.rs` is the *less* conformant of the two
(3/12 vs 7/12) despite being the one that reads files users write by hand.

## 5. What this changes in the programme

- AS5's classification is recorded as **tightening + compatibility correction + correctness
  defect**; the migration axis is closed as NOT REQUIRED.
- F1 did not wait for AS5. It was taken under the §3 live-defect pre-emption rule the same day as
  DEV-182 / CD-384 and is **fixed**; F2 and F3 remain AS5's.
- AS5's exit criterion 2 ("a standard JSON test corpus ... passes") should be read against the
  12-construct table above as the project-specific minimum, since it already distinguishes the two
  parsers.
- No first-party file changes. The corpus needed nothing.

## 6. Reproducing

The probe was temporary and is not in the tree. It walked `packages/` for `starkpkg.json` and
`stark.lock`, called each parser directly, and printed verdict plus parsed value; the strict side is
`json.loads` in Python 3. Rebuilding it is a ~150-line integration test. If AS5 wants this as a
standing check, the construct table in §3 is the fixture set, and it belongs in the tree rather than
as a probe.
