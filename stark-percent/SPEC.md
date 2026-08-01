# stark-percent Specification

## Purpose
Provide strict RFC percent encoding and decoding for URL path segments, paths, and query string components.

## Public Types & Functions
- `PercentEncodeSet`: Closed enum containing `PathSegment`, `Path`, `QueryComponent`.
- `PercentError`: `IncompleteEscape(UInt64)`, `InvalidHexDigit(UInt64, UInt8)`, `OutputTooLarge`.
- `encode(input: &[UInt8], set: PercentEncodeSet) -> String`
- `decode(input: &String) -> Result<Vec<UInt8>, PercentError>`

## Semantics
- Decoding accepts both uppercase and lowercase hex digits (`%2f`, `%2F`).
- Decoding reports exact offsets of malformed sequences.
- Plus (`+`) is treated strictly as literal character `+` (byte 0x2B), NOT space.
