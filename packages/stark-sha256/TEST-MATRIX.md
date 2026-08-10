# stark-sha256 test matrix

| ID | Coverage |
| --- | --- |
| API-001 | Frozen public API: `hash`, `hash_hex` |
| HASH-EMPTY-001 | Empty input known-answer vector |
| HASH-ABC-001 | `abc` known-answer vector |
| HASH-LONG-001 | Standard multi-block known-answer vector |
| HASH-BINARY-001 | Bytes `00..FF` known-answer vector |
| PAD-055 | 55 repeated `a` bytes |
| PAD-056 | 56 repeated `a` bytes; catches one-block boundary errors |
| PAD-063 | 63 repeated `a` bytes |
| PAD-064 | 64 repeated `a` bytes |
| PAD-065 | 65 repeated `a` bytes |
| ARITH-WRAP-001 | `UInt32` modular addition carry and wrap cases in private self-check |
| ARITH-ROT-001 | ROTR across high bits and sigma fixture values in private self-check |
| ARITH-ENDIAN-001 | Big-endian `UInt32` read/write and `UInt64` length write in private self-check |
| DET-001 | Repeated `hash_hex` calls on same input |
| IMM-001 | Input vector unchanged after hashing |
| DIGEST-001 | Result algorithm is `DigestAlgorithm::Sha256` |
| DIGEST-002 | Result digest contains exactly 32 bytes |
| HEX-001 | Hex output is 64 lowercase characters via `stark-digest` |
| CROSS-001 | Consumer calls `hash` and `hash_hex` |
| CROSS-002 | Native consumer expected stdout is `sha256:ok` |
| ENG-001 | Engine agreement intended over package tests and consumer qualification |

Mutation controls covered by pinned tests:

| Mutation | Control |
| --- | --- |
| Little-endian input words | `ARITH-ENDIAN-001`, `HASH-ABC-001` |
| Little-endian output words | `HASH-ABC-001`, `HEX-001` |
| Wrong initial H value | All known-answer vectors |
| One wrong K constant | All compression vectors |
| ROTL instead of ROTR | `ARITH-ROT-001`, all compression vectors |
| Checked `UInt32` addition instead of modular addition | `ARITH-WRAP-001`, compression vectors |
| Omitted `0x80` padding byte | `HASH-EMPTY-001`, `HASH-ABC-001` |
| Length encoded in bytes instead of bits | `HASH-ABC-001`, `ARITH-ENDIAN-001` |
| Length encoded little-endian | `ARITH-ENDIAN-001`, `HASH-ABC-001` |
| 56-byte boundary handled as one block | `PAD-056` |
| Only 63 rounds | All compression vectors |
| Incorrect W schedule index | `HASH-LONG-001`, `HASH-BINARY-001` |

The million-`a` vector is retained in the work package and omitted from routine tests until package
runtime cost is measured.
