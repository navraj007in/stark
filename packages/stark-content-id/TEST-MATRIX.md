# stark-content-id test matrix

| ID | Coverage |
| --- | --- |
| API-001 | Frozen public surface: `from_digest`, `parse`, `digest`, `to_string`, `equals` |
| PARSE-001 | Canonical SHA-256 content IDs |
| PARSE-002 | Uppercase digest accepted when `stark-digest` accepts it |
| PARSE-003 | Mixed-case digest accepted when `stark-digest` accepts it |
| FORMAT-001 | Missing separator and empty parts |
| FORMAT-002 | Leading, trailing and separator-adjacent whitespace |
| FORMAT-003 | Malformed separator with extra colon |
| ALG-001 | SHA1 rejected |
| ALG-002 | MD5 rejected |
| ALG-003 | Uppercase SHA256 rejected |
| DIGEST-001 | Short digest |
| DIGEST-002 | Long digest |
| DIGEST-003 | Invalid first byte |
| DIGEST-004 | Invalid middle byte |
| DIGEST-005 | Invalid final byte |
| CANON-001 | Lowercase canonical rendering |
| CANON-002 | Parse/render idempotence |
| EQ-001 | Identical IDs compare equal |
| EQ-002 | Unequal IDs compare unequal |
| OWN-001 | Shared digest access through `digest` |
| IMM-001 | `digest`, `to_string`, and `equals` do not mutate IDs |
| DET-001 | Deterministic rendering and parsing |
| CROSS-001 | Consumer calls all public APIs |
| ENG-001 | Engine agreement through package tests and consumer runs |
