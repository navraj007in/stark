# stark-form Test Matrix

| ID | Category | Description | Target | Engines | Status |
|---|---|---|---|---|---|
| FRM-PRS-001 | Parsing | Form pair parsing with `+`-as-space decoding | Parser | HIR | PASS |
| FRM-SER-001 | Serialization | Form pair serialization with space-to-`+` encoding | Serializer | HIR | PASS |
| FRM-CLI-001 | Compiler Check | Package compilation via `stark check` | Manifest & Types | HIR | PASS |
| FRM-RUN-001 | Interpreter Run | Interpreter execution holding `FormPair` | Interpreter | HIR interpreter | BLOCKED (BLOCKER-STARKC-INTERP-VEC-STRUCT-DROP) |
