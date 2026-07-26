# stark-hex v0.1 Test Matrix

| ID | Category | Input/fixture | Expected result | Expected error | Offset/value | Engines | Status |
|---|---|---|---|---|---|---|---|
| API-001 | API | public enum/functions | frozen API only | none | n/a | check | pass |
| ENC-EMPTY-001 | ENCODE_EMPTY | `[]` | `""` | none | n/a | test | pass |
| ENC-BOUND-001 | ENCODE_BOUNDARIES | `00 01 0f 10 7f 80 ab ff` | exact lower/upper vectors | none | n/a | test | pass |
| ENC-ALL-001 | ENCODE_ALL_BYTES | `0x00..0xFF` | frozen 512-byte lower/upper strings | none | n/a | test | pass |
| DEC-VALID-001 | DECODE_EMPTY | `""` | `[]` | none | n/a | test | pass |
| DEC-VALID-002 | DECODE_LOWER | `00 01 0f 10 7f 80 ab ff 007f80ff` | bytes | none | n/a | test | pass |
| DEC-VALID-003 | DECODE_UPPER | `0F 7F AB FF 007F80FF` | bytes | none | n/a | test | pass |
| DEC-VALID-004 | DECODE_MIXED | `aB` | `[0xAB]` | none | n/a | test | pass |
| DEC-LEN-001 | DECODE_INVALID_LENGTH | `0 f abc ABC 000 12345` | reject | InvalidLength | n/a | test | pass |
| DEC-CHAR-001 | DECODE_INVALID_CHARACTER | `0x00` | reject | InvalidCharacter | 1/120 | test | pass |
| DEC-CHAR-002 | DECODE_INVALID_CHARACTER | leading/trailing whitespace | reject | InvalidCharacter | exact | test | pass |
| DEC-CHAR-003 | DECODE_INVALID_CHARACTER | separators `: - _` | reject | InvalidCharacter | exact | test | pass |
| DEC-CHAR-004 | DECODE_INVALID_CHARACTER | `gg`, `0g`, non-ASCII | reject | InvalidCharacter | exact | test | pass |
| PREC-001 | ERROR_PRECEDENCE | `x`, `0x0`, `abc`, `abz`, `é` | specified precedence | exact | exact | test | pass |
| RT-001 | ROUND_TRIP | representative + full domain | decode(encode(bytes)) == bytes | none | n/a | test | pass |
| CAN-001 | CANONICALITY | `aB` | lower `ab`, upper `AB` | none | n/a | test | pass |
| DET-001 | DETERMINISM | repeated encode/decode | equal outputs | none | n/a | test | pass |
| IMM-001 | INPUT_IMMUTABILITY | reused Vec/String | unchanged inputs | none | n/a | test | pass |
| FIX-001 | FIXTURES | checked-in valid/invalid files | mapped to tests | none | n/a | documentation | pass |
| CROSS-001 | CROSS_PACKAGE | `stark-hex-consumer` | prints `48656c6c6f` | none | n/a | check/run | pass |
| CROSS-002 | CROSS_PACKAGE_NATIVE | `stark-hex-consumer` | native executable | none | n/a | build | blocked: `Vec::as_slice` |
| ENG-001 | THREE_ENGINE | package/consumer | observations agree | none | n/a | HIR/MIR/native | blocked: native build |
