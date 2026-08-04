# stark-query Test Matrix

| ID | Category | Description | Target | Engines | Status |
|---|---|---|---|---|---|
| QRY-PRS-001 | Parsing | Query pair parsing, duplicate key & order preservation | Parser | HIR | PASS |
| QRY-SER-001 | Serialization | Query pair serialization with percent encoding | Serializer | HIR | PASS |
| QRY-PLUS-001 | Literal Plus | Preserves literal `+` without replacing with space | Decoders | HIR | PASS |
| QRY-CLI-001 | Compiler Check | Package compilation via `stark check` | Manifest & Types | HIR | PASS |
| QRY-RUN-001 | Interpreter Run | Interpreter execution holding `QueryPair` | Interpreter | HIR interpreter | BLOCKED (BLOCKER-STARKC-INTERP-VEC-STRUCT-DROP) |
