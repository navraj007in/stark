# stark-urlencoded Test Matrix

| ID | Area | Case | Expected |
| --- | --- | --- | --- |
| URLENC-QRY-001 | Query parse | empty, `a=b`, missing `=`, empty name/value, duplicate names | Baseline `stark-query` semantics |
| URLENC-QRY-002 | Query plus | `+`, `%20`, `%2B` | Bare `+` remains plus |
| URLENC-QRY-003 | Query serialize | spaces, plus, separators | Percent-only query output |
| URLENC-FRM-001 | Form parse | bare `+`, `%2B`, multiple pairs | Bare `+` becomes space |
| URLENC-FRM-002 | Form serialize | space and literal plus | Space becomes `+`, plus becomes `%2B` |
| URLENC-SEP-001 | Separator | `a=b=c`, both modes | Only the first `=` separates |
| URLENC-SEP-002 | Separator | `a=1&&b=2`, `a=1&`, both modes | Empty segments are preserved as pairs |
| URLENC-LIM-001 | Limits/errors | pair/value limits, invalid percent | Shared limit and percent errors preserved |
| URLENC-LIM-002 | Limits | each of the four limits, one at a time | Each limit reports its own error |
| URLENC-LIM-003 | Limits | `%41%42%43=%61%62%63` under a 3-byte limit | Limits measure decoded, not escaped, length |
| URLENC-RTP-001 | Round trip | serialize then parse, both modes | Names and values survive unchanged |
