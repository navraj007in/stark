# WP-C2.5 — Structured Diagnostics Transport

Gate: C2 (Reference Execution Semantics and Compiler Service Foundation).

Status: **Completed 2026-07-18.**

## Scope delivered

- one compiler-owned structured diagnostic batch produced by `ProjectAnalysis`;
- stable code, severity, message, primary source/span, related source/spans, notes, help, source
  version, and optional rule/deviation identifiers;
- deterministic schema-v1 JSON with deterministic source-map and object ordering;
- CLI text/JSON and package CLI text rendering from the shared batch;
- LSP `publishDiagnostics` notifications from the shared batch, with document versions,
  cross-file related information, and close-time clearing;
- explicit deferral of the known pre-alpha code collisions in DEV-019 to WP-C2.11.

Uncatalogued compiler diagnostics receive reserved transport fallbacks `E0000`/`W0000`.
This makes the code field total without treating the current catalogue as frozen.

## Evidence

Focused tests cover deterministic JSON object ordering, LSP publication/version transport,
single-session cross-file primary/related provenance, text related-location rendering, and
end-to-end `starkc check --message-format json` determinism.

See `starkc/docs/compiler/diagnostic-transport.md` for the transport contract and
`COMPILER-STATE.md` for full-suite evidence.

## Scope control

WP-C2.5 does not reassign existing public codes, define new language semantics, implement C2.6's
completeness audit, or begin backend/MIR work.
