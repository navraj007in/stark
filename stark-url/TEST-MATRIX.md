# stark-url v0.1 Test Matrix

| ID | Category | Input | Expected | Status |
|---|---|---|---|---|
| ENC-001 | component encode | unreserved ASCII | unchanged | pass |
| ENC-002 | component encode | reserved delimiters | uppercase `%HH` | pass |
| ENC-003 | component encode | controls LF/TAB | `%0A%09` | pass |
| DEC-001 | percent decode | `a%20b`, `%2f%2F` | ASCII decoded | pass |
| DEC-002 | malformed `%HH` | `%`, `%A`, `%G0`, `%0G`, `abc%`, `abc%A` | `InvalidPercentEscape` exact offset | pass |
| DEC-003 | UTF-8 validation | overlong/surrogate/out-of-range escapes | `InvalidUtf8` exact offset | pass |
| DEC-004 | non-ASCII construction boundary | `caf%C3%A9` | `PercentDecodedNonAsciiBlocked` at 3 | pass |
| DEC-005 | ASCII controls | `%00`, `%09`, `%0A`, `%0D`, `%1F`, `%7F` | exact decoded ASCII scalars | pass |
| TARGET-001 | examples | `/health`, `/users/123`, `/files/a%20b.txt` | path decoded, no query | pass |
| TARGET-002 | query model | repeated keys and empty value | ordered `Vec<QueryParameter>` | pass |
| TARGET-003 | separator semantics | `%2F`, `%3D`, `%26`, `%3F` | encoded separators are data | pass |
| TARGET-004 | error offsets | path/query malformed escapes | exact byte offsets | pass |
| LIMIT-001 | limits | small input/query limits | `InputTooLarge`, `TooManyQueryParameters` | pass |
| QUERY-001 | query encode | repeated keys, spaces, slash, `&`, `=` | canonical ordered query string | pass |
| CROSS-001 | consumer check/run | `stark-url-consumer` | `q=stark%20url&tag=compiler&tag=language` | pass |
| CROSS-002 | consumer native build | `stark-url-consumer` | native executable | blocked: `str.bytes()` native lowering |
