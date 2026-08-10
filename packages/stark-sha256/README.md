# stark-sha256

`stark-sha256` provides a pure STARK Core v1 one-shot SHA-256 implementation.

Public API:

```stark
pub fn hash(input: &[UInt8]) -> Digest;
pub fn hash_hex(input: &[UInt8]) -> String;
```

The package hashes in-memory byte slices, pads messages according to SHA-256, processes 512-bit
blocks, and returns `DigestAlgorithm::Sha256` values through `stark-digest`.

Hex output is canonical lowercase text produced by `stark-digest`; this package does not implement
or depend directly on hexadecimal encoding.

SHA-256 is not a password hashing function.

This package does not claim constant-time execution, side-channel resistance, HMAC, file hashing,
digital signatures, cryptographic audit, or certification.
