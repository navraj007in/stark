# stark-percent Test Matrix

| ID | Category | Description | Target | Engines | Status |
|---|---|---|---|---|---|
| PCT-ENC-001 | Encoding | Encode under PathSegment, Path, QueryComponent sets | Encode sets | HIR/interpreter (consumer) | PASS |
| PCT-DEC-001 | Decoding | Decode valid percent sequences with uppercase/lowercase/mixed hex | Decoders | HIR/interpreter (consumer) | PASS |
| PCT-ERR-001 | Errors | Offset accuracy for incomplete escape and invalid hex digit | Decoders | HIR/interpreter (consumer) | PASS |
| PCT-PLUS-001 | Literal Plus | Preserves literal + without replacing with space | Decoders | HIR/interpreter (consumer) | PASS |
| PCT-CLI-001 | Compiler Check | Package type checking via `stark check` | Manifest & Types | HIR | PASS |
| PCT-TST-001 | Unit Tests CLI | Unit testing via `stark test` | Test runner | `stark test` CLI | BLOCKED (BLOCKER-STARKC-TEST-RUNNER-SYNTHETIC-SPAN) |
