# stark-ascii Test Matrix

| ID | Category | Description | Target | Engines | Status |
|---|---|---|---|---|---|
| ASC-CLASS-001 | Classification | All 256 byte values tested for is_ascii, is_ascii_alpha, is_ascii_uppercase, is_ascii_lowercase, is_ascii_digit, is_ascii_hex_digit, is_ascii_whitespace, is_ascii_control | All 256 bytes | HIR/interpreter | PASS |
| ASC-TCHAR-001 | tchar Validation | RFC 9110 tchar validation across all 256 bytes | All 256 bytes | HIR/interpreter | PASS |
| ASC-CONV-001 | Case Conversion | Upper to lower, lower to upper, non-alpha unchanged | All 256 bytes | HIR/interpreter | PASS |
| ASC-CMP-001 | Comparison | Case-insensitive slice & string comparison | Slices & Strings | HIR/interpreter | PASS |
