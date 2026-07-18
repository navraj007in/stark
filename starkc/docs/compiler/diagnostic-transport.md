# Compiler Diagnostic Transport v1

Status: implemented by WP-C2.5. This is a compiler-service contract, not a freeze of the
pre-alpha diagnostic-code catalogue.

## Canonical record

`ProjectAnalysis::diagnostic_batch` is the only semantic-diagnostic transport producer. Each
`StructuredDiagnostic` contains:

- a non-empty stable code;
- severity and message;
- a primary session `SourceId`, byte span, source name, and optional label;
- zero or more related `SourceId`/span/message entries, including entries in other files;
- ordered notes and help entries;
- an optional source version;
- optional Core rule and known-deviation IDs.

Compiler emitters that have not yet been assigned a specific catalogue code are transported as
`E0000` or `W0000`. Those reserved fallback codes are stable transport values, not claims that
distinct uncatalogued diagnostics share one language rule.

Source versions are populated by versioned clients such as LSP. Disk/CLI analysis encodes an
unknown version as JSON `null`; the field is never omitted.

## Deterministic JSON

`DiagnosticBatch::to_json` emits schema version 1 with fixed field order and source IDs from the
analysis source map. The source map keeps the root first and sorts remaining files by source
name. Arrays retain compiler order except reference-like related data whose producer already
defines its order.

The top-level object contains `schemaVersion`, `tool`, `toolVersion`, enabled `extensions`, a
deterministic `sources` table, and `diagnostics`. Each source records its session ID, file,
root/module provenance, and package identity.
Each diagnostic contains `code`, `severity`, `message`, `primary`, `related`, `notes`, `help`,
`sourceVersion`, `ruleId`, and `deviationId`.

## Consumer mappings

- `starkc check --message-format text|json` renders the canonical batch.
- Package `stark check`/`build`/`run` text diagnostics render the same batch.
- LSP `textDocument/publishDiagnostics` is produced from the same batch. Primary spans become
  LSP ranges, related entries become `relatedInformation`, document versions are published in
  both notification parameters and diagnostic `data`, and closing a document publishes an empty
  diagnostic array.

The minimal LSP JSON object encoder sorts keys, so notifications and responses are deterministic.

## Catalogue status

DEV-019 lists five confirmed code-collision classes. Stabilizing this transport does not
legitimize or freeze those assignments. WP-C2.11 owns their evidence-complete reassignment
together with normative catalogue, compiler, positive/negative test, conformance, and deviation
updates.
