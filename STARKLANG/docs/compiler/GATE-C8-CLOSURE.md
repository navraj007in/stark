# Gate C8 — Closure Ruling

**Decision:** `CLOSED`
**Owner ruling:** Final (CD-385, 2026-08-06)
**Gate:** C8 — Semantic Language Services
**Exit report:** `starkc/docs/compiler/C8-exit-report.md` (qualified commit `6556a0d`, 2026-07-31)
**Interactive record:** CD-281 / DEV-012 entry, `COMPILER-STATE.md` (2026-07-31)

---

## 1. Ruling

> **Gate C8 is CLOSED.**

C8 establishes compiler-backed semantic language services: diagnostics, hover, definition,
references, completion, signature help, rename, document symbols, semantic tokens and inlay
information, all derived from the shared compiler analysis rather than from a parallel index, plus
protocol conformance and a packaged VS Code extension.

The gate closes **with interactive validation recorded for three of ten advertised features, not
ten.** That limit is the substance of this ruling and is stated in full in §2. It is not a
technicality to be discovered later by someone reading the exit report.

## 2. What the closure claims, and what it does not

C8's exit condition (WP-C8.8) required that advertised features "reflect real compiler semantics and
are interactively validated". The first half is met by protocol tests over the shared analysis. The
second half is met **in part**.

**Interactively confirmed by the owner**, VS Code 1.130.0, extension `starklang.stark-language@0.2.0`,
macOS 26.5.2 arm64, against a real STARK package (2026-07-31):

| Feature | Interactive evidence |
| --- | --- |
| Hover | confirmed |
| Go-to-definition | confirmed |
| Find-references | confirmed |

**Protocol-tested only — no editor session has exercised these:**

| Feature | Evidence |
| --- | --- |
| Diagnostics (on type, on save) | protocol tests |
| Formatting / format-on-save | protocol tests |
| Completion | protocol tests |
| Signature help | protocol tests |
| Rename | protocol tests |
| Document symbols | protocol tests |
| Semantic tokens | protocol tests |

**The claim this gate makes is therefore:** the language services are compiler-derived and
protocol-conformant, and the three core navigation queries are additionally confirmed against a real
editor. It does **not** claim that the packaged extension's full UI behaviour has been exercised by a
person.

`DEV-012` stays **OPEN, narrowed** to the seven features above. Closing the gate does not close the
deviation; the deviation is what carries the residue honestly, in the same way C7 closed without a
steady-state runtime-performance claim.

## 2a. Against the plan's own `C8-CLOSED` requirements

`WP-C8-Semantic-Language-Services-Execution-Plan.md` §17.2 lists ten requirements for
`C8-CLOSED`. This ruling is checked against them rather than around them.

| # | Requirement | Status |
| ---: | --- | --- |
| 1 | no advertised semantic capability remains a placeholder | MET |
| 2 | diagnostics are version-safe and package-aware | MET |
| 3 | hover is compiler-derived | MET — and interactively confirmed |
| 4 | definition and references use resolved identity | MET — and interactively confirmed |
| 5 | symbols are semantic rather than textual | MET |
| 6 | formatting uses the established compiler formatter | MET |
| 7 | raw protocol tests pass | MET, with the limit in §4 |
| 8 | **real VS Code validation passes** | **PARTIALLY MET — owner override** |
| 9 | C7 parallel changes reconciled on the qualified commit | MET (exit report §Reconciliation) |
| 10 | evidence registry contains no unresolved capability claim | MET |

**Item 8 is an explicit owner override, not a claim that the requirement is satisfied.** Real VS Code
validation passed for three of ten advertised features. The gate closes anyway, for the reasons in
§3, and the shortfall is carried by DEV-012 remaining open and narrowed rather than by weakening the
requirement's wording. Anyone auditing this gate should read item 8 as "deliberately closed short",
not as "met".

The closure claim itself uses §17.3's prescribed wording, which already accommodates a partial
record by referring to "the recorded VS Code environment" rather than to complete coverage:

> Gate C8 provides compiler-backed semantic language services for the documented STARK project and
> package configurations. Advertised diagnostics, hover, navigation, references, symbols,
> completion, signature and rename capabilities are derived from shared compiler analysis and have
> been validated through protocol tests and the recorded VS Code environment. Known limitations and
> unsupported configurations remain explicitly listed.

## 3. Why closing is the right call anyway

The gate has been candidate-complete since 2026-07-31 on a single blocking reason. Three
considerations decide it:

1. **The blocking reason was environmental, not technical.** DEV-012 records "no `code` CLI /
   Extension Development Host has been available in the implementing environment". That has since
   been answered for the navigation queries, and nothing about the remaining seven suggests a
   different outcome — they share the same analysis path and the same protocol layer.
2. **Holding the gate open does not generate the missing evidence.** It is gathered by a person
   using an editor, which is scheduled work, not a consequence of a gate's state.
3. **Downstream work is being blocked by the ambiguity rather than by the risk.**
   `WP-ARCHITECTURE-STABILIZATION.md` gates AS5 and AS8 on this decision precisely so they do not
   silently reopen or duplicate C8's territory. An unmade decision blocks them indefinitely.

## 4. A limit this ruling adds, which the exit report did not have

**Protocol validation checked verdicts, not values, and a value defect survived it.**

DEV-182 (CD-384, 2026-08-06) found that the LSP JSON parser silently decoded every escaped non-BMP
character to the empty string: an emoji in a completion label, a file path, or a diagnostic message
was lost without an error. Both the parse and the response *succeeded*. Only the value was wrong.

WP-C8.7 is "Protocol and editor validation", and this defect passed it — because a protocol test that
asserts a well-formed exchange records agreement when both sides say "ok". It was found by an audit
that compared **values**, not verdicts.

This is recorded here as a **standing limit on what C8's protocol evidence demonstrates**, not as a
reason to withhold closure: the defect is fixed, and the class it belongs to is now known. Any future
claim resting on "C8's protocol tests pass" must account for the fact that those tests do not compare
decoded values end to end.

## 5. Consequences

- **AS5** (`WP-ARCHITECTURE-STABILIZATION.md`) is unblocked. Its dependency clause takes the
  "C8 closes first" branch: AS5 preserves C8's protocol and interactive baseline while consolidating
  the shared JSON implementation used by the non-LSP surfaces. It does not reopen C8's scope.
- **AS8's** post-C8 LSP work — measured debounce, cancellation, one-analysis-per-package cache
  ownership — is unblocked, and remains bounded by the charter's continuing deferral of full
  incrementality.
- **DEV-012** remains open and narrowed to seven features, owned by WP-C8.7, to be closed by an
  editor session rather than another protocol test.
- **Gate C9** is unaffected; it was already permitted to begin from C8's candidate-complete state.

## 6. Evidence of record

| Item | Location |
| --- | --- |
| Exit report and per-WP evidence | `starkc/docs/compiler/C8-exit-report.md`, `starkc/docs/compiler/evidence/c8/` |
| Interactive session record | `COMPILER-STATE.md`, DEV-012 entry (2026-07-31) |
| Narrowed deviation | `starkc/docs/conformance/KNOWN-DEVIATIONS.md`, DEV-012 |
| Protocol value-defect finding | `COMPILER-STATE.md` CD-384; `STARKLANG/docs/compiler/audits/AS0-MANIFEST-STRICTNESS-AUDIT.md` F1 |
