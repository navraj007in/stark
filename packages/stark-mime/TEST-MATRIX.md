# stark-mime Test Matrix

| ID | Category | Description | Target | Engines | Status |
|---|---|---|---|---|---|
| MIME-PRS-001 | Parsing | Bounded media type parser (`parse_media_type`) | Parser | HIR | PASS |
| MIME-FMT-001 | Formatting | Format MediaType with parameters (`format_media_type`) | Formatter | HIR | PASS |
| MIME-IS-001 | Comparison | Case-insensitive `media_type_is` and `media_type_parameter` lookup | Matching | HIR | PASS |
| MIME-CLI-001 | Compiler Check | Package compilation via `stark check` | Manifest & Types | HIR | PASS |
| MIME-RUN-001 | Interpreter Run | Interpreter execution holding `MediaType` | Interpreter | HIR interpreter | BLOCKED (BLOCKER-STARKC-INTERP-VEC-STRUCT-DROP) |
