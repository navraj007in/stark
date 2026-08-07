# AS5 — opening analysis

**Packet:** AS5, protocol/manifest/version-surface contracts. Campaign B, executing in Sprint 2.
**Branch:** `wp-arch-stability/sprint-2`. **Date:** 2026-08-07.
**Status:** OPEN. This records what the packet is starting from, including three defects the Sprint 1
audit did not cover because it looked at parsing and these are on the **emit** side.

---

## 1. What Sprint 1 already established

`AS0-MANIFEST-STRICTNESS-AUDIT.md` compared the two in-tree JSON **parsers** across twelve
constructs and classified AS5 as *tightening + compatibility correction + correctness defect*, with
the repository-migration axis closed as NOT REQUIRED. Its findings stand:

- **F1** (LSP decoding every escaped non-BMP character to the empty string) — fixed already, under
  the live-defect pre-emption rule, as DEV-182 / CD-384.
- **F2, compatibility correction** — `package.rs` rejects valid RFC 8259: *all* `\u` escapes (no
  `'u'` arm in `parse_string`) and *all* exponent numbers (`1e3` fails with a misleading
  `Expected ',' or '}'`). This is input the parser should accept and does not.
- **F3, tightening** — both parsers accept invalid JSON in **non-overlapping** sets. `package.rs`
  takes trailing commas, raw control characters and leading-zero numbers; `lsp/protocol.rs` takes
  trailing input after a complete value. Neither is a subset of the other, so consolidation has to
  pick a grammar rather than adopt either parser's behaviour.
- 9 of 12 constructs diverge, and `package.rs` — the one reading files people write by hand — is the
  *less* conformant at 3/12 against `lsp/protocol.rs`'s 7/12.

---

## 2. F5 — there are **four** escaping authorities, and three emit invalid JSON

The audit counted parsers. Production also contains four independent JSON string escapers, and only
one of them is correct.

| Authority | Escapes | Verdict |
| --- | --- | --- |
| `diag.rs::escape_json` | `"` `\` `\b` `\f` `\n` `\r` `\t`, and **every** C0 control as `\u00xx` | **correct** |
| `lsp/protocol.rs::escape_json_string` | `"` `\` `\n` `\r` `\t` | emits raw C0 controls |
| `onnx/verifier.rs::escape_json` | `"` `\` `\n` `\r` `\t` | emits raw C0 controls |
| `bin/stark.rs::json_escape` | `"` `\` `\n` only | emits raw controls **including TAB** |

RFC 8259 §7 requires that U+0000–U+001F be escaped. Three of the four do not, so they can produce
output that no conforming parser accepts.

(`lexer.rs::escape` and `doc_gen/highlight.rs::escape` are unrelated — STARK string-literal escapes
and HTML entity escaping. They are not JSON authorities and are not in AS5's scope.)

### This is reachable, and demonstrated

**`stark doctor --json` produces invalid JSON.** With an install root whose path contains a TAB —
legal on every POSIX filesystem — the emitted document is rejected by a standard parser:

```text
$ stark doctor --json --root "$(pwd)/probe<TAB>root"
{
  "ok": false,
  "install_root": "/…/probe<TAB>root",      ← raw U+0009 inside a JSON string
  …

python3 -c 'import json,sys; json.load(sys.stdin)'
→ Invalid control character at: line 3 column 134
```

The command advertises machine-readable output and emits something no conforming parser accepts.
That is a **correctness defect**, not a strictness preference.

**The LSP and ONNX escapers have the same hole**, confirmed directly: given `"before␁after"` both
emit the raw U+0001 rather than ``.

The LSP case is the serious one, and it lands exactly on the limit `GATE-C8-CLOSURE.md` §4 records
and this packet's dependencies section already warns about: **C8's protocol validation compared
verdicts, not values.** A client lenient enough to accept a raw control character reports success,
and the wire message is still invalid. DEV-182 passed that same evidence.

### Consequence for the packet

Exit criterion 1 — "production code contains one JSON parser and one escaping authority" — is
carrying more weight than it appears to. The parser half is consolidation of two known-divergent
implementations. The escaper half is the repair of three defects, and needs its own fails-first
tests, per the packet's rule that "any value-divergence finding is repaired under its own DEV record
with a fails-before-the-repair test, not absorbed into the consolidation commit."

---

## 3. What makes consolidation cheap

The two `JsonValue` enums are **textually identical**:

```rust
pub enum JsonValue { Null, Bool(bool), Number(f64), String(String), Array(Vec<JsonValue>), Object(HashMap<String, JsonValue>) }
```

So the shared layer does not have to reconcile two data models — only two grammars and four
escapers. The packet's requirement to "preserve protocol-specific data models above the shared JSON
layer" costs nothing here: `package.rs` and `lsp/protocol.rs` keep their own accessors and message
types and delegate parsing and serialization.

---

## 4. Decisions this packet must take, and which are policy

Three of these are determined by RFC 8259 and need no ruling. Two change what the toolchain
**accepts** from existing users and are policy — flagged rather than resolved here.

| # | Decision | Kind |
| ---: | --- | --- |
| 1 | Escape every C0 control on every emit surface | determined — RFC 8259 §7 |
| 2 | Accept `\u` escapes and exponent numbers in manifests | determined — F2, valid input currently rejected |
| 3 | Reject unpaired surrogates rather than substituting | determined — F1's class; DEV-182 already chose this for decoding |
| 4 | **Reject trailing commas and leading-zero numbers in `starkpkg.json`** | **policy** — tightening; every first-party manifest is already strict-clean (audit §5: "no first-party file changes"), but a third-party manifest that parses today would stop |
| 5 | **Reject trailing input after a complete JSON-RPC value** | **policy** — tightening on the LSP wire; C8's baseline requires this rejection, so the two point the same way, but it is still a behaviour change on a shipped protocol surface |

Both policy items are strictly-narrowing and both have a directly stated justification, so the
recommendation is to take them. They are called out because exit criterion 5 asks for CE9 review of
security-sensitive parsing decisions, and "what the toolchain refuses to load" is that kind of
decision even when the refusal is right.

---

## 5. Exit criterion 4 is a separate piece of work

"A runtime/MIR surface change cannot compile or pass tests without updating its compatibility
identity" is not about JSON at all. AS1b-iii is a worked example of the problem and of the manual
discipline that currently substitutes for it: `MIR_VERSION` went 0.3 → 0.4 because a person decided
it should, and `a11_host_resource::the_mir_version_records_every_shape_amendment` pins the constant
but not the *shape it describes*. The A14 history in that constant's own doc comment records two
surface additions that shipped without advancing it.

The packet permits either a deterministic schema fingerprint or an exact-set test. An exact-set test
over the `RuntimeFn` members and the `Statement`/`Terminator`/`MirTy` variant names is the smaller
change and fails for the right reason: adding a variant without touching the identity breaks it.
