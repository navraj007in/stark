# stark-hex

`stark-hex` v0.1 is a pure-STARK hexadecimal package. It encodes arbitrary bytes as lowercase or
uppercase hexadecimal and strictly decodes hexadecimal text back into raw bytes.

## Public API

```stark
pub enum HexError {
    InvalidLength,
    InvalidCharacter(UInt64, UInt8),
}

pub fn encode_lower(input: &[UInt8]) -> String;
pub fn encode_upper(input: &[UInt8]) -> String;
pub fn decode(input: &str) -> Result<Vec<UInt8>, HexError>;
```

## Behaviour

`encode_lower` uses exactly `0123456789abcdef`. `encode_upper` uses exactly
`0123456789ABCDEF`. Both emit two ASCII characters per input byte with no prefix, whitespace, or
separators.

`decode` accepts lowercase, uppercase, and mixed-case hexadecimal digits. It first scans left to
right for the first invalid byte, then rejects odd length, then decodes pairs. Offsets are
zero-based UTF-8 byte offsets and `InvalidCharacter` carries the raw invalid byte.

## Examples

```text
[]                 -> ""
[0x00, 0x7F, 0xFF] -> "007fff" / "007FFF"
"aB"               -> [0xAB]
"0x00"             -> InvalidCharacter(1, 120)
"abc"              -> InvalidLength
```

## Exclusions

No `0x` prefixes, whitespace tolerance, separators, streaming APIs, custom alphabets, checksums,
cryptographic claims, filesystem access, native provider, compiler intrinsic, or host-language
implementation.

## Status

`IMPLEMENTATION COMPLETE — LOCAL NATIVE QUALIFICATION PASS`.

Package `stark check`, `stark test`, and `stark fmt --check` pass. The test suite covers all byte
values for encoding/round-trip and every invalid ASCII byte for decoder rejection. The
cross-package consumer checks, runs under `stark run`, builds natively, and the generated native
binary prints `48656c6c6f`. Final exact-commit package qualification should be re-run after the
C6.5 compiler baseline freezes. See `EVIDENCE.md`.
