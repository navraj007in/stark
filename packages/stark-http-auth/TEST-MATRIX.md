# stark-http-auth test matrix

| ID | Coverage |
| --- | --- |
| API-001 | Frozen public API |
| BASIC-001 | Aladdin vector |
| BASIC-002 | user/pass |
| BASIC-003 | password containing colon |
| BASIC-004 | empty password |
| BASIC-005 | empty username |
| BASIC-006 | username colon rejected by constructor precondition |
| BEARER-001 | simple token |
| BEARER-002 | dotted token |
| BEARER-003 | allowed punctuation |
| BEARER-004 | padding |
| BEARER-005 | empty rejected |
| BEARER-006 | whitespace rejected |
| BEARER-007 | control chars rejected |
| PARSE-001 | Basic |
| PARSE-002 | Basic case-insensitive scheme |
| PARSE-003 | Bearer |
| PARSE-004 | Bearer case-insensitive scheme |
| PARSE-005 | unsupported scheme |
| PARSE-006 | malformed value |
| BASE64-001 | invalid Base64 |
| BASE64-002 | decoded credentials missing colon |
| CANON-001 | canonical Basic casing |
| CANON-002 | canonical Bearer casing |
| CANON-003 | round trip |
| SEC-001 | CRLF bearer rejection |
| SEC-002 | CRLF Basic rejection |
| SEC-003 | errors do not echo credentials |
| IMM-001 | input immutability |
| DET-001 | deterministic output |
| CROSS-001 | consumer uses full public API |
| ENG-001 | HIR/MIR/native agreement |
