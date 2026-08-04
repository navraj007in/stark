# stark-base64 v0.1 Evidence

## Status

COMPLETE

## Baseline

- Repository commit: `d3bd967188b961944bba0433826c55ac30764e07`
- Date: 2026-07-31
- STARK compiler version/commit: STARK Core v1 `starkc` v0.1.0
- Host OS: macOS
- Host architecture: Apple Silicon / mac
- Rust toolchain used by native backend: local Cargo/Rust toolchain

## Files created or modified

- `stark-base64/starkpkg.json`
- `stark-base64/src/lib.stark`
- `stark-base64/src/tests.stark`
- `stark-base64/README.md`
- `stark-base64/EVIDENCE.md`
- `stark-base64/docs/`

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
running 12 tests

test test_encode_rfc_vectors ... ok
test test_encode_zeros_and_ones ... ok
test test_decode_rfc_vectors ... ok
test test_decode_alphabet_roundtrip ... ok
test test_invalid_characters ... ok
test test_invalid_character_required_bytes ... ok
test test_invalid_length ... ok
test test_invalid_padding ... ok
test test_noncanonical_bits ... ok
test test_required_roundtrip_lengths ... ok (1127ms)
test test_boundary_and_repeated_calls ... ok
test test_full_byte_domain ... ok (41ms)

test result: ok. 12 passed; 0 failed; 0 ignored; 1188ms total
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
- Invalid characters: space, tab, LF, CR, '-', '_', '.', ':', non-ASCII first UTF-8 byte
- Invalid lengths: 1, 2, 3, 5, 6, 7 byte lengths
- Invalid padding: `=AAA`, `A=AA`, `AA=A`, `A===`, `====`, `AAAA====`, `AA==AAAA`, `AAAA=AAA`
- Noncanonical trailing bits: `Zh==`, `Zm9=`, `AB==`, `AAB=`, plus canonical counterparts `Zg==`, `Zm8=`, `AA==`, `AAA=`
- Round-trip lengths: 0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024, and 1025 bytes verified
- Full byte-domain: 256 bytes (`0x00`..`0xFF`) round-trip verified
- Boundary regressions: exact quartet boundary, final one-byte group, final two-byte group, repeated encode, successful decode after failed decode

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
