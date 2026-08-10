# stark-digest

`stark-digest` provides validated value objects for cryptographic digest bytes.

Version 0.1.0 supports SHA-256 identity only. It does not compute hashes, access the host, or use a
native provider. Hexadecimal parsing and rendering are delegated to `stark-hex`; output is canonical
lowercase hexadecimal.

## Public API

```stark
pub enum DigestAlgorithm {
    Sha256,
}

pub enum DigestError {
    InvalidLength(UInt64, UInt64),
    InvalidHex(UInt64, UInt8),
}
```

```stark
pub fn from_bytes(algorithm: DigestAlgorithm, bytes: Vec<UInt8>) -> Result<Digest, DigestError>;
pub fn parse_hex(algorithm: DigestAlgorithm, text: &str) -> Result<Digest, DigestError>;
pub fn algorithm(digest: &Digest) -> DigestAlgorithm;
pub fn bytes_copy(digest: &Digest) -> Vec<UInt8>;
pub fn to_hex(digest: &Digest) -> String;
pub fn equals(left: &Digest, right: &Digest) -> Bool;
```

For `DigestAlgorithm::Sha256`, construction requires exactly 32 bytes, or 64 hexadecimal
characters after syntax validation.
