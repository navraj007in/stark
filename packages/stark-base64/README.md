# stark-base64

`stark-base64` is an official optional standard library package (Layer 3) for the **STARK Core v1** programming language. It provides strict, production-disciplined Base64 encoding and decoding adhering to **RFC 4648, Section 4**.

## Features

- **100% Pure STARK Core v1**: Zero native code, no C/Rust extensions, no compiler modifications, and no dependencies.
- **Strict Decoding**: Enforces canonical Base64 input formatting, exact padding placement, and zero unused trailing bits.
- **Deterministic Error Handling**: Typed `Base64Error` variants with exact byte-offset indicators.

## Standard Alphabet

Uses the standard RFC 4648 Base64 alphabet:
```text
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/
```

Encoding always includes `=` padding. Decoding requires canonical padding (`=`, `==`, or no padding for complete quartets) and rejects non-canonical or malformed inputs.

## Public API

```stark
pub enum Base64Error {
    InvalidLength,
    InvalidCharacter(UInt64, UInt8),
    InvalidPadding(UInt64),
    NonCanonicalTrailingBits(UInt64)
}

pub fn encode(input: &[UInt8]) -> String;

pub fn decode(input: &str) -> Result<Vec<UInt8>, Base64Error>;
```

## Usage Examples

### Encoding Bytes to Base64

```stark
use stark_base64::{encode};

fn main() {
    let data: &[UInt8] = &[102u8, 111u8, 111u8]; // "foo"
    let encoded = encode(data);
    println(encoded.as_str()); // Outputs "Zm9v"
}
```

### Decoding Base64 to Bytes

```stark
use stark_base64::{decode, Base64Error};

fn main() {
    match decode("Zm9v") {
        Ok(bytes) => {
            println("Decoded successfully!");
        }
        Err(Base64Error::InvalidCharacter(idx, val)) => {
            println("Invalid character found");
        }
        Err(Base64Error::InvalidLength) => {
            println("Invalid Base64 string length");
        }
        Err(Base64Error::InvalidPadding(idx)) => {
            println("Invalid padding structure");
        }
        Err(Base64Error::NonCanonicalTrailingBits(idx)) => {
            println("Non-canonical trailing bits detected");
        }
    }
}
```

## Explicit Exclusions (v0.1)

The following features are explicitly out of scope for v0.1:
- URL-safe Base64 (`-` and `_` are rejected as invalid characters).
- Whitespace or line-break tolerance (spaces, newlines are rejected).
- Unpadded decoding or custom padding rules.
- Streaming or reader/writer interfaces.

## License

MIT OR Apache-2.0
