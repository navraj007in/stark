# stark-cookie Test Matrix

| ID | Area | Case | Expected |
| --- | --- | --- | --- |
| COOKIE-001 | Cookie parse | `session=abc123` | One pair, name and value split on the first `=` |
| COOKIE-002 | Cookie parse | `session=abc123; theme=dark; lang=en` | Three pairs, order preserved |
| COOKIE-003 | Cookie parse | `a=one; a=two; a=three` | Duplicates preserved in order; `cookie_value` returns the first |
| COOKIE-004 | Whitespace | `a=1;  b=2;\tc=3`, leading OWS, `""`, `a=` | OWS around separators accepted; empty input and empty value are legal |
| COOKIE-005 | Quoted value | `session="abc123"`, `session="hello world"` | Quotes are syntax; SP accepted inside quotes |
| SET-001 | Set-Cookie | `session=abc123` | First segment is the cookie, never an attribute |
| SET-002 | Set-Cookie | `Path=/account` | Value preserved, no path matching |
| SET-003 | Set-Cookie | `Domain=example.com` | Value preserved, no DNS or PSL |
| SET-004 | Set-Cookie | `Expires=Wed, 09 Jun 2021 10:18:14 GMT` | Validated opaque string |
| SET-005 | Set-Cookie | `Max-Age=3600`, `=0`, `=-1` | Signed `Int64`, sign preserved |
| SET-006 | Set-Cookie | `Secure` | Boolean attribute |
| SET-007 | Set-Cookie | `HttpOnly` | Boolean attribute |
| SET-008 | Set-Cookie | `SameSite=Lax` | Recognised policy |
| SET-009 | Set-Cookie | `Priority=High`, `Foo` | Unknown attributes preserved, valued and valueless |
| SET-010 | Set-Cookie | `Path=/one; Path=/two` | Duplicate attributes preserved in order |
| CASE-001 | Case handling | `secure`, `SECURE`, `httponly`, `HTTPONLY`, `path`, `max-age`, `domain` | Attribute names compare ASCII case-insensitively |
| CASE-002 | Case handling | `samesite=strict`, `SAMESITE=NONE` | SameSite values compare ASCII case-insensitively |
| FORMAT-001 | Formatting | Same value formatted twice | Byte-identical output; order never sorted |
| FORMAT-002 | Formatting | `hello world` vs `abc123` | Quoted only when the bare form would not parse back |
| FORMAT-003 | Formatting | Every attribute kind | Canonical casing; extensions keep their input casing |
| FORMAT-004 | Formatting | `a="plain"` | Unnecessary quotes canonicalised away |
| ROUND-001 | Round trip | Cookie with quoted value and duplicates | `parse(format(v))` equivalent; formatting stable |
| ROUND-002 | Round trip | Set-Cookie with every attribute kind, duplicates, extensions | `parse(format(v))` equivalent; formatting stable |
| LIMIT-001 | Limits | `max_total_bytes` at limit, over limit | `ExceededTotalBytesLimit` |
| LIMIT-002 | Limits | `max_cookie_pairs` at limit, over limit | `ExceededPairCountLimit` |
| LIMIT-003 | Limits | `max_name_bytes` at limit, over limit | `ExceededNameLimit` |
| LIMIT-004 | Limits | `max_value_bytes` at limit, over limit, quoted over limit | `ExceededValueLimit`, measured on content not quotes |
| LIMIT-005 | Limits | `max_attribute_count` at limit, over limit | `ExceededAttributeCountLimit` |
| LIMIT-006 | Limits | `max_attribute_name_bytes` over limit | `ExceededAttributeNameLimit` |
| LIMIT-007 | Limits | `max_attribute_value_bytes` over limit | `ExceededAttributeValueLimit` |
| ERR-001 | Errors | `=value`, `a b=1`, `a` | `InvalidName` / `UnexpectedCharacter` with byte offsets |
| ERR-002 | Errors | `a=one,two` | `InvalidValue` naming the offending byte |
| ERR-003 | Errors | `a="unterminated` | `InvalidQuote` at the opening quote |
| ERR-004 | Errors | `Secure=yes`, `HttpOnly=1`, `Path`, `; ;` | `MalformedAttribute` |
| ERR-005 | Errors | `Max-Age=abc`, `Max-Age=-`, 28 digits, `Int64` max ± 1 | `InvalidNumber` / `NumberOverflow`; max accepted, max+1 rejected |
| ERR-006 | Errors | `SameSite=Bogus` | `InvalidSameSite`, never folded onto a recognised value |
| SEC-001 | Security | 200-byte name against a 64-byte total limit | Bounded by `ExceededTotalBytesLimit` |
| SEC-002 | Security | Newline in a bare value, newline inside quotes, control byte in an attribute value | Rejected, not treated as a terminator |
| SEC-003 | Security | 100 pairs against a 10-pair limit | Bounded by `ExceededPairCountLimit` |
| SEC-004 | Security | Trailing junk after a value | `UnexpectedCharacter`, never silently dropped |
| CONSUMER-001 | Qualification | `stark-cookie-consumer` | Every public callable executed, not merely imported |
| ENGINE-001 | Engines | Interpreter, native debug, native release | Same observable result in all three |

Positions are byte offsets into the input handed to the parser, counted from zero, and are the same
under every engine.
