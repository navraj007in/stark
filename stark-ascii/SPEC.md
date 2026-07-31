# stark-ascii Specification

## Purpose
Provide pure byte-first ASCII classification, RFC 9110 token character (`tchar`) validation, ASCII case conversion, and case-insensitive slice comparison for protocol parsers without requiring UTF-8 validation or host dependencies.

## Public API
- `is_ascii(byte: UInt8) -> Bool`
- `is_ascii_alpha(byte: UInt8) -> Bool`
- `is_ascii_uppercase(byte: UInt8) -> Bool`
- `is_ascii_lowercase(byte: UInt8) -> Bool`
- `is_ascii_digit(byte: UInt8) -> Bool`
- `is_ascii_hex_digit(byte: UInt8) -> Bool`
- `is_ascii_whitespace(byte: UInt8) -> Bool`
- `is_ascii_control(byte: UInt8) -> Bool`
- `is_tchar(byte: UInt8) -> Bool`
- `to_ascii_lowercase(byte: UInt8) -> UInt8`
- `to_ascii_uppercase(byte: UInt8) -> UInt8`
- `eq_ignore_ascii_case(left: &[UInt8], right: &[UInt8]) -> Bool`
- `string_eq_ignore_ascii_case(left: &String, right: &String) -> Bool`

## Semantics
- `is_tchar` validates HTTP token characters: `!`, `#`, `$`, `%`, `&`, `'`, `*`, `+`, `-`, `.`, `^`, `_`, `` ` ``, `|`, `~`, `0-9`, `A-Z`, `a-z`.
- Non-ASCII bytes (`0x80..=0xFF`) are classified accurately as non-ASCII, non-tchar, non-control.
- Case conversions map `A-Z` to `a-z` and vice versa; non-alpha bytes are left unchanged.

## Exclusions
- Unicode case folding, locale awareness, UTF-8 normalization.
