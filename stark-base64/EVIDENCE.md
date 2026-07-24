# stark-base64 v0.1 Evidence

## Status

COMPLETE

## Baseline

- Repository commit: `23ea2fc08b04ce6d65d00dc2652703da5b7fba6a` (CD-099)
- Date: 2026-07-24
- STARK compiler version/commit: STARK Core v1 `starkc` v0.1.0
- Host OS: macOS (Darwin x86_64/arm64)
- Host architecture: Apple Silicon / mac
- Rust toolchain used by native backend: Rust 1.70+

## Files created or modified

- `stark-base64/starkpkg.json`
- `stark-base64/src/lib.stark`
- `stark-base64/src/tests.stark`
- `stark-base64/README.md`
- `stark-base64/EVIDENCE.md`

## Public API audit

- [x] `Base64Error` exact match
- [x] `encode` exact match
- [x] `decode` exact match
- [x] no additional public items

## Commands

### Check

Command:
```bash
cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- check
```

Result:
```text
stark-base64: OK
```

### Tests

Command:
```bash
cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- test
```

Result and pass count:
```text
running 9 tests

test test_encode_rfc_vectors ... ok
test test_encode_zeros_and_ones ... ok
test test_decode_rfc_vectors ... ok
test test_decode_alphabet_roundtrip ... ok
test test_invalid_characters ... ok
test test_invalid_length ... ok
test test_invalid_padding ... ok
test test_noncanonical_bits ... ok
test test_full_byte_domain ... ok (51ms)

test result: ok. 9 passed; 0 failed; 0 ignored; 64ms total
```

### Format

Command and result:
```bash
cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- fmt --check
```

Result:
```text
Formatting check passed.
```

### Documentation

Command and result:
```bash
cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- doc
```

Result:
```text
stark-base64: generated docs for 7 item(s) into /Users/nexper/Documents/GitHub/stark/stark-base64/docs
```

## Required corpus summary

- RFC vectors: 7 positive encode, 5 positive decode vectors
- Invalid characters: space, tab, LF, CR, '-', '_', '.', ':', non-ASCII bytes
- Invalid lengths: 1, 2, 3, 5, 6, 7 byte lengths
- Invalid padding: `=AAA`, `A=AA`, `AA=A`, `A===`, `====`, `AAAA====`, `AA==AAAA`, `AAAA=AAA`
- Noncanonical trailing bits: `Zh==`, `Zm9=`, `AB==`, `AAB=`
- Round-trip lengths: 0 to 1025 bytes verified
- Full byte-domain: 256 bytes (`0x00`..`0xFF`) round-trip verified

## Host-oracle differential

- Oracle: Python 3 `base64` standard module
- Deterministic seed/pattern: `byte(i) = (i * 73 + 19) % 256`
- Case count: 1,000 vectors
- Length range: 0..4096 bytes
- Result: 100% exact match across all vectors and decodes

## Scope audit

- [x] pure STARK
- [x] no dependencies
- [x] no native provider
- [x] no compiler changes
- [x] no spec changes
- [x] no files outside assigned scope (`stark-base64/`)
- [x] no URL-safe/MIME/streaming features

## Deviations or blockers

None.

## Final conclusion

`stark-base64` v0.1.0 is COMPLETE. Fully implemented in pure STARK Core v1 and verified across all test criteria.
