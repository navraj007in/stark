# AS5 — opening analysis

**Packet:** AS5, protocol/manifest/version-surface contracts. Campaign B, executing in Sprint 2.
**Branch:** `wp-arch-stability/sprint-2`. **Date:** 2026-08-07.
**Status:** **CLOSED 2026-08-07** (a–g). Opened as this analysis; §6 carries the CE9 record and §7
the AS7 forward note. What follows records what the packet started from, including three defects the
Sprint 1 audit did not cover because it looked at parsing and these are on the **emit** side.

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

The packet permits either a deterministic schema fingerprint or an exact-set test.

**This section originally recommended the exact-set test for both constants, over the `RuntimeFn`
members and the `Statement`/`Terminator`/`MirTy` variant names. That recommendation was wrong and is
superseded** — owner review caught it before implementation. An exact set of variant names **would
not have moved for AS1b-iii**, the change that caused the most recent `MIR_VERSION` increment:
removing `SourceInfo.file` and `MirProgram.files`, introducing `MirProgram.sources` and eliminating
`FileId` touches no variant name at all. A guard that cannot see its own most recent trigger is
theatre.

What was built instead gives the two constants two mechanisms, matching what each one means:

| Constant | Identifies | Mechanism |
| --- | --- | --- |
| `MIR_RUNTIME_SURFACE` | the set of runtime **operations** | exact canonical set of `RuntimeFn` members |
| `MIR_VERSION` | the structural **shape** of the model | schema fingerprint over every public type's variant names, payload shapes, field names and field types |

The fingerprint is computed over extracted *declarations*, not source text: hashing `mir/mod.rs`
would make a comment edit or a `rustfmt` run read as a compatibility change, and those types carry
doc comments that are edited constantly.

`tests/as5_compatibility_identity.rs` mutation-tests the guard against its own source **in memory**,
because mutating `mir/mod.rs` on disk stops the crate compiling — the test binary never runs, and a
broken guard would look identical to a working one. Six shape changes move the fingerprint
(including AS1b-iii in both directions, a field rename with no variant touched, and a field type
change); three non-shape edits do not.


---

## 6. CE9 record — the three parsing decisions taken (2026-08-07)

Exit criterion 5 asks that security-sensitive parsing decisions receive CE9 review. Three were
taken. All three narrow what the toolchain accepts, and all three are recorded here as decisions
rather than as consequences of replacing a parser.

### CE9-1 — `starkpkg.json` rejects trailing commas and leading-zero numbers

**Decision:** ACCEPT the tightening. **Authority:** the shared JSON parser, applied to manifests.

A manifest is a durable configuration contract; accepting non-JSON syntax creates compatibility debt
for no benefit. `AS0-MANIFEST-STRICTNESS-AUDIT.md` §5 established that every first-party manifest is
already strict-clean, so this narrows what *third-party* manifests may contain rather than requiring
a repository migration. Verified by checking all 41 first-party packages under the new parser: all
pass.

### CE9-2 — the LSP transport rejects trailing input after the JSON value

**Decision:** ACCEPT the tightening. **Authority:** the shared JSON parser, applied to JSON-RPC.

A JSON-RPC frame contains exactly one JSON value. Accepting `{"jsonrpc":"2.0",...} garbage` weakens
framing and hides malformed clients — the remainder simply vanished. C8's protocol baseline already
expected the rejection, so this makes the implementation match the contract it was written against.

### CE9-3 — JSON nesting is bounded at 128

**Decision:** ACCEPT `MAX_DEPTH = 128`, unconfigurable. **Authority:** the shared JSON parser.

| | |
| --- | --- |
| **Reason** | A resource/safety property of recursive descent, not consumer-specific document semantics. The recursion happens inside the shared authority, so the authority that owns the recursion must guarantee that adversarial input produces a `JsonError` rather than stack exhaustion. |
| **Behaviour** | depth ≤ 128 — accepted subject to normal JSON validity; depth > 128 — deterministic `JsonError`; **no stack-overflow or abort path**. Only arrays and objects consume depth. |
| **Value** | The measured maximum in a first-party manifest is 3. An LSP structure approaching 128 levels would already be pathological. |
| **Compatibility** | Implementation-defined under RFC 8259 §9. JSONTestSuite's `i_structure_500_nested_arrays` is an externally sourced boundary case and its rejection is pinned. |
| **Not configurable** | A `parse_with_limits()` API would create policy surface with no demonstrated consumer needing deeper JSON. If one appears, that is when configuration is justified. |

### Registered, not decided here

- **DEV-186 — the LSP transport allocates an unbounded `Content-Length` before parsing.** Found
  during this review. `MAX_DEPTH` cannot help: the allocation happens before the parser runs. A
  framing-layer cap belongs to an LSP hardening packet, with the request-id work.
- **The LSP request-id model** (DEV-185's adjacent note): string ids and non-`i64` numeric ids are
  refused. Also transport, also not AS5.

---

## 7. Forward note for AS7

`tests/as5_compatibility_identity.rs` derives the MIR schema fingerprint by reading declarations
from **`src/mir/mod.rs`**, which is correct today because that file is the MIR data-model authority.

**AS7 modularises passes and files.** If a schema-bearing MIR type moves into a submodule without the
fingerprint's canonical input set being widened to follow it, the guard silently stops covering that
type — it would keep passing, over a smaller model. That is not a defect in `96b5cbb`; it is a
migration invariant, and it belongs on the AS7 checklist.
