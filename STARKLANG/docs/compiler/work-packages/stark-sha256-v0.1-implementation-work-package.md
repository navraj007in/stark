# STARK SHA-256 v0.1 — Codex Implementation Work Package

**Package:** `stark-sha256`
**Version:** `0.1.0`
**Implementation language:** STARK Core v1 only
**Native code:** PROHIBITED
**Compiler changes:** PROHIBITED
**Native provider:** PROHIBITED
**Host capabilities:** NONE
**Third-party dependencies:** PROHIBITED
**Required first-party dependency:** `stark-digest` v0.1
**Algorithm:** SHA-256 only
**Streaming API:** OUT OF SCOPE for v0.1
**Status:** Public API and behaviour frozen for this packet

---

# 1. Instruction to Codex

Implement one bounded pure-STARK package providing a conforming one-shot SHA-256 implementation.

Treat every MUST, MUST NOT, SHALL, SHALL NOT, REQUIRED and EXACTLY statement in this document as normative for this work package.

Do not:

* modify the compiler;
* add runtime intrinsics;
* add native Rust/C code;
* add providers;
* add host capabilities;
* implement SHA-224;
* implement SHA-384 or SHA-512;
* implement HMAC;
* implement password hashing;
* implement filesystem hashing;
* implement streaming state;
* introduce SIMD;
* introduce platform-specific optimisations;
* duplicate `stark-digest`;
* duplicate hexadecimal encoding;
* weaken STARK's checked arithmetic semantics to make SHA-256 easier.

If the current compiler cannot express the required algorithm, stop and register a reduced compiler/runtime blocker instead of changing compiler-owned code.

---

# 2. Objective

Create:

```text
arbitrary bytes
     ↓
stark-sha256
     ↓
SHA-256 compression
     ↓
32 digest bytes
     ↓
stark-digest
     ↓
DigestAlgorithm::Sha256
```

The implementation MUST:

* hash arbitrary in-memory byte slices;
* implement standard SHA-256 message padding;
* process 512-bit blocks;
* implement all 64 SHA-256 rounds;
* implement arithmetic modulo (2^{32}) without relying on STARK integer overflow;
* return a validated `Digest`;
* provide canonical lowercase hexadecimal output through `stark-digest`;
* produce identical observations under HIR, MIR and native execution;
* contain no host dependency.

This package is intended to become a primitive for:

```text
artifact hashing
content identity
build provenance
lockfile verification
EvidenceSnapshot identity
future file hashing
```

Those consumers are outside this packet.

---

# 3. Mandatory repository preflight

Before editing:

```bash
git fetch origin
git switch <owner-assigned-branch>
git rev-parse HEAD
git status --short
```

Record:

```text
BASELINE_SHA=
BASELINE_BRANCH=
WORKTREE_CLEAN=
```

Inspect the current versions of:

```text
COMPILER-STATE.md
packages/stark-checksum/
packages/stark-hex/
packages/stark-digest/
scripts/qualify-first-party-packages.py
STARKLANG/docs/spec/03-Type-System.md
STARKLANG/docs/spec/CORE-V1-ABSTRACT-MACHINE.md
```

Do not assume code or commands named in this document remain current.

---

# 4. Required prerequisite: `stark-digest`

`stark-sha256` MUST use the existing canonical digest representation.

The required dependency surface is conceptually:

```stark
Digest
DigestAlgorithm::Sha256

from_bytes(
    DigestAlgorithm,
    Vec<UInt8>
) -> Result<Digest, DigestError>

to_hex(&Digest) -> String
```

If `stark-digest` is not present on the implementation baseline:

```text
STOP
→ report PREREQUISITE_NOT_MET
```

unless the owner explicitly authorises bringing the existing `stark-digest` package into the branch.

Do NOT:

* define another `Digest`;
* define another SHA-256 digest enum;
* duplicate `stark-hex`;
* copy the implementation of `stark-digest`.

---

# 5. Package structure

Preferred structure:

```text
packages/stark-sha256/
├── starkpkg.json
├── stark.lock
├── README.md
├── EVIDENCE.md
├── TEST-MATRIX.md
└── src/
    ├── lib.stark
    └── tests.stark

packages/stark-sha256-consumer/
├── starkpkg.json
└── src/
    └── main.stark
```

Follow the actual package-test convention at the implementation baseline if it differs.

---

# 6. Manifest

Expected shape:

```json
{
  "name": "stark-sha256",
  "version": "0.1.0",
  "entry": "src/lib.stark",
  "distribution": {
    "toolchain": true
  },
  "dependencies": {
    "stark_digest": {
      "package": "stark-digest",
      "path": "../stark-digest",
      "version": "0.1.0"
    }
  }
}
```

`stark-sha256` MUST NOT directly depend on `stark-hex`.

Hexadecimal rendering belongs to `stark-digest`.

No capability declaration is permitted.

---

# 7. Frozen public API

Implement EXACTLY:

```stark
use stark_digest::Digest;

pub fn hash(input: &[UInt8]) -> Digest;

pub fn hash_hex(input: &[UInt8]) -> String;
```

No additional public functions, structs, enums or constants in v0.1.

Private helpers are unrestricted within the scope of this specification.

---

# 8. API semantics

## 8.1 `hash`

```stark
pub fn hash(input: &[UInt8]) -> Digest
```

MUST:

1. read `input` without mutating it;
2. compute SHA-256 over exactly those bytes;
3. produce exactly 32 output bytes;
4. construct:

```text
DigestAlgorithm::Sha256
```

through `stark-digest`;
5. return the resulting `Digest`.

The SHA-256 implementation itself MUST guarantee the 32-byte invariant.

If `stark_digest::from_bytes` rejects the implementation-produced result, this is an internal package invariant failure, not a user-input error.

Do not expose `DigestError` from `hash`.

---

## 8.2 `hash_hex`

```stark
pub fn hash_hex(input: &[UInt8]) -> String
```

MUST behave as:

```text
hash(input)
    ↓
stark_digest::to_hex(...)
```

It MUST NOT contain its own hexadecimal encoder.

Output MUST therefore be exactly:

```text
64 lowercase ASCII hexadecimal characters
```

for every input.

---

# 9. No wrapping arithmetic assumption

This is a load-bearing rule.

SHA-256 specifies additions modulo:

```text
2^32
```

STARK integer addition MUST NOT be allowed to overflow merely because SHA-256 requires modular arithmetic.

The implementation MUST NOT use:

```stark
a + b
```

on `UInt32` when the mathematical sum may exceed `UInt32::MAX`.

Instead, implement private modular-add helpers by widening first.

Conceptually:

```stark
fn add2(a: UInt32, b: UInt32) -> UInt32 {
    ((a as UInt64 + b as UInt64) & 0xFFFF_FFFFu64) as UInt32
}
```

For more operands, either chain `add2`:

```text
add2(add2(a, b), c)
```

or use a `UInt64` sum known to remain inside `UInt64`.

Five maximum `UInt32` values sum to less than (2^{35}), so widening round expressions to `UInt64` is sufficient.

Required helpers may include:

```text
add2
add4
add5
```

The exact helper breakdown is private.

Tests MUST demonstrate carry/wrap behaviour directly.

Examples:

```text
FFFFFFFF + 00000001 -> 00000000
FFFFFFFF + FFFFFFFF -> FFFFFFFE
80000000 + 80000000 -> 00000000
```

These are SHA-256 modular-add expectations, not changes to STARK arithmetic semantics.

---

# 10. Rotation semantics

Implement private:

```text
rotr32(x, n)
```

for the fixed SHA-256 rotation counts.

Conceptually:

```text
(x >> n) | (x << (32 - n))
```

However STARK left shift is checked.

Therefore the implementation MUST ensure that no left-shift operation is rejected merely because high bits are discarded by a mathematical rotate.

If direct `UInt32 << n` cannot safely express rotation under current Core semantics, implement rotation through widened/masked arithmetic or another Core-conforming representation.

Do NOT request wrapping-shift semantics from the compiler.

The package implementation MUST demonstrate every rotation used by SHA-256 through tests or through SHA-256 intermediate-vector tests.

---

# 11. SHA-256 logical functions

Implement these exact 32-bit functions.

## 11.1 Choice

```text
Ch(x, y, z) =
    (x & y) ^ ((~x) & z)
```

## 11.2 Majority

```text
Maj(x, y, z) =
    (x & y) ^ (x & z) ^ (y & z)
```

## 11.3 Big sigma 0

```text
Σ0(x) =
    ROTR32(x, 2)
  ^ ROTR32(x, 13)
  ^ ROTR32(x, 22)
```

## 11.4 Big sigma 1

```text
Σ1(x) =
    ROTR32(x, 6)
  ^ ROTR32(x, 11)
  ^ ROTR32(x, 25)
```

## 11.5 Small sigma 0

```text
σ0(x) =
    ROTR32(x, 7)
  ^ ROTR32(x, 18)
  ^ (x >> 3)
```

## 11.6 Small sigma 1

```text
σ1(x) =
    ROTR32(x, 17)
  ^ ROTR32(x, 19)
  ^ (x >> 10)
```

All operations are on exactly 32 bits.

---

# 12. Initial SHA-256 state

Initialize the eight working hash words EXACTLY as:

```text
H0 = 6a09e667
H1 = bb67ae85
H2 = 3c6ef372
H3 = a54ff53a
H4 = 510e527f
H5 = 9b05688c
H6 = 1f83d9ab
H7 = 5be0cd19
```

Represent them as `UInt32`.

Do not generate these constants dynamically.

---

# 13. Round constants

Use these EXACTLY, in order:

```text
428a2f98 71374491 b5c0fbcf e9b5dba5
3956c25b 59f111f1 923f82a4 ab1c5ed5
d807aa98 12835b01 243185be 550c7dc3
72be5d74 80deb1fe 9bdc06a7 c19bf174
e49b69c1 efbe4786 0fc19dc6 240ca1cc
2de92c6f 4a7484aa 5cb0a9dc 76f988da
983e5152 a831c66d b00327c8 bf597fc7
c6e00bf3 d5a79147 06ca6351 14292967
27b70a85 2e1b2138 4d2c6dfc 53380d13
650a7354 766a0abb 81c2c92e 92722c85
a2bfe8a1 a81a664b c24b8b70 c76c51a3
d192e819 d6990624 f40e3585 106aa070
19a4c116 1e376c08 2748774c 34b0bcb5
391c0cb3 4ed8aa4a 5b9cca4f 682e6ff3
748f82ee 78a5636f 84c87814 8cc70208
90befffa a4506ceb bef9a3f7 c67178f2
```

Preferred representation:

```stark
const K: [UInt32; 64] = [ ... ];
```

if the current compiler accepts this cleanly.

If not, use a private deterministic equivalent.

Do not calculate SHA-256 constants from primes at runtime.

---

# 14. Message padding

For an input of `L` bytes:

```text
original bytes
+
80
+
zero bytes
+
64-bit big-endian original bit length
```

The padded message length MUST be divisible by:

```text
64 bytes
```

and the final eight bytes MUST begin at byte offset:

```text
56 mod 64
```

before those eight length bytes are appended.

Equivalent invariant:

```text
padded_len % 64 == 0
```

---

## 14.1 Padding algorithm

Conceptually:

```text
message = copy(input)
append 0x80

while message.len() % 64 != 56:
    append 0x00

bit_length = original_byte_length * 8

append bit_length as UInt64 big-endian
```

Do not mutate the input slice.

---

## 14.2 Length overflow

Before computing:

```text
byte_length * 8
```

ensure the multiplication cannot overflow `UInt64`.

The maximum representable SHA-256 input byte length for the 64-bit bit-length field is:

```text
0x1FFF_FFFF_FFFF_FFFF
```

The current in-memory implementation cannot realistically materialize such an input on supported machines, but the arithmetic MUST still not silently wrap.

Use a private guard.

Do not add a public error merely for this physically unreachable current-host case.

If the guard is ever violated, trap with a package-internal invariant message rather than allowing a checked-arithmetic trap at an unrelated expression.

---

# 15. Block parsing

Process the padded message in:

```text
64-byte blocks
```

For each block construct:

```text
W[0..63] : UInt32
```

The first 16 words MUST be parsed as big-endian:

```text
W[i] =
    b0 << 24
  | b1 << 16
  | b2 << 8
  | b3
```

where bytes are widened to `UInt32` before shifts.

Example:

```text
61 62 63 80 -> 61626380
```

No host endian conversion is permitted.

---

# 16. Message schedule

For:

```text
i = 16 .. 63
```

compute:

```text
W[i] =
    W[i-16]
  + σ0(W[i-15])
  + W[i-7]
  + σ1(W[i-2])
    mod 2^32
```

Use the modular-add helper.

Do not use ordinary overflow on `UInt32`.

---

# 17. Compression rounds

Initialize:

```text
a = H0
b = H1
c = H2
d = H3
e = H4
f = H5
g = H6
h = H7
```

For every round:

```text
i = 0 .. 63
```

compute:

```text
T1 =
    h
  + Σ1(e)
  + Ch(e, f, g)
  + K[i]
  + W[i]
    mod 2^32

T2 =
    Σ0(a)
  + Maj(a, b, c)
    mod 2^32
```

Then update EXACTLY:

```text
h = g
g = f
f = e
e = d + T1 mod 2^32
d = c
c = b
b = a
a = T1 + T2 mod 2^32
```

After all 64 rounds:

```text
H0 = H0 + a mod 2^32
H1 = H1 + b mod 2^32
H2 = H2 + c mod 2^32
H3 = H3 + d mod 2^32
H4 = H4 + e mod 2^32
H5 = H5 + f mod 2^32
H6 = H6 + g mod 2^32
H7 = H7 + h mod 2^32
```

---

# 18. Final digest bytes

Serialize:

```text
H0 H1 H2 H3 H4 H5 H6 H7
```

in big-endian order.

For each word:

```text
byte0 = word >> 24
byte1 = word >> 16
byte2 = word >> 8
byte3 = word
```

mask/cast each to `UInt8`.

The output MUST contain:

```text
32 bytes exactly
```

Then call:

```stark
stark_digest::from_bytes(
    DigestAlgorithm::Sha256,
    bytes
)
```

and return the resulting `Digest`.

Do not construct `Digest` by bypassing its public API.

---

# 19. Private implementation shape

A straightforward implementation is preferred.

Suggested helpers:

```text
add2
add4
add5

rotr32

choose
majority

big_sigma0
big_sigma1
small_sigma0
small_sigma1

read_be_u32
append_be_u32
append_be_u64

pad_message
compress_block
```

The exact private decomposition may differ.

Do not make algorithm internals public.

---

# 20. Optimisation policy

Correctness first.

v0.1 MUST NOT contain:

* SIMD;
* lookup-table compression tricks;
* loop unrolling solely for speed;
* unsafe code;
* native acceleration;
* architecture-specific instructions;
* parallel block processing;
* runtime-generated constants.

A normal 64-round implementation is expected.

Performance may be measured, but no optimisation is required for v0.1.

---

# 21. Required known-answer vectors

The following MUST pass.

## 21.1 Empty input

```text
input:
""

SHA-256:
e3b0c44298fc1c149afbf4c8996fb924
27ae41e4649b934ca495991b7852b855
```

Combined:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

---

## 21.2 `"abc"`

```text
ba7816bf8f01cfea414140de5dae2223
b00361a396177a9cb410ff61f20015ad
```

Combined:

```text
ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
```

---

## 21.3 `"hello world"`

```text
b94d27b9934d3e08a52e52d7da7dabf
ac484efe37a5380ee9088f7ace2efcde9
```

Combined:

```text
b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
```

---

## 21.4 Standard multi-block vector

Input:

```text
abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq
```

Expected:

```text
248d6a61d20638b8e5c026930c3e6039
a33ce45964ff2167f6ecedd419db06c1
```

Combined:

```text
248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
```

---

## 21.5 Quick-brown-fox vector

Input:

```text
The quick brown fox jumps over the lazy dog
```

Expected:

```text
d7a8fbb307d7809469ca9abcb0082e4f
8d5651e46d3cdb762d02d0bf37c9e592
```

---

## 21.6 Binary domain vector

Input:

```text
bytes 0x00 through 0xFF in ascending order
```

Expected:

```text
40aff2e9d2d8922e47afd4648e696749
7158785fbd1da870e7110266bf944880
```

This is important because text-only fixtures are insufficient evidence for a byte algorithm.

---

# 22. Padding-boundary vectors

Use repeated ASCII `a`.

These MUST pass:

```text
55 bytes:
9f4390f8d30c2dd92ec9f095b65e2b9a
e9b0a925a5258e241c9f1e910f734318

56 bytes:
b35439a4ac6f0948b6d6f9e3c6af0f5
f590ce20f1bde7090ef7970686ec6738a

63 bytes:
7d3e74a05d7db15bce4ad9ec0658ea98
e3f06eeecf16b4c6fff2da457ddc2f34

64 bytes:
ffe054fe7ae0cb6dc65c3af9b61d5209
f439851db43d0ba5997337df154668eb

65 bytes:
635361c48bb9eab14198e76ea8ab7f1a
41685d6ad62aa9146d301d4f17eb0ae0
```

These pin:

```text
one-block padding
extra-block padding
block-exact input
first byte beyond block
```

Do not replace them with only generic round-trip tests.

A hash has no reversible round-trip property.

---

# 23. Long-message vector

If runtime cost remains reasonable, include:

```text
1,000,000 copies of ASCII 'a'
```

Expected:

```text
cdc76e5c9914fb9281a1c7e284d73e67
f1809a48a497200e046d39ccc7112cd0
```

If this is too expensive for routine package tests:

* retain it as a qualification/slow test;
* record execution time;
* do not delete the vector.

---

# 24. Arithmetic-specific tests

Do not rely only on final known-answer vectors.

Test enough private algorithm machinery to distinguish likely wrong implementations.

Required classes:

```text
modular addition with carry

choice function

majority function

ROTR across high bits

small sigma 0
small sigma 1

big sigma 0
big sigma 1

big-endian UInt32 read

big-endian UInt32 write

big-endian UInt64 length write
```

If package conventions prohibit testing private functions directly, test them through deterministic internal fixtures.

---

# 25. Required negative controls

A cryptographic implementation can pass a small fixture set while still being structurally wrong.

Include controls that would fail these mutations:

```text
little-endian input words

little-endian output words

wrong initial H value

one wrong K constant

ROTL instead of ROTR

ordinary checked UInt32 addition instead of modular addition

omitted 0x80 padding byte

length encoded in bytes instead of bits

length encoded little-endian

56-byte boundary handled as one block instead of two

only 63 rounds

incorrect W schedule index
```

At least one test MUST be demonstrably sensitive to each class.

You do not need to keep mutation code in the repository.

Record the control mapping in `TEST-MATRIX.md`.

---

# 26. Input immutability

`hash` and `hash_hex` MUST NOT mutate or consume the input.

Test:

```text
construct input Vec
take slice
hash it
inspect original Vec
assert identical bytes and length
```

---

# 27. Determinism

For representative inputs:

```text
hash(input)
hash(input)
hash(input)
```

MUST return equal digest values.

Likewise:

```text
hash_hex(input)
```

MUST produce byte-identical lowercase output repeatedly.

---

# 28. Cross-package consumer

Create a small consumer.

It MUST actually call every public callable.

Example behaviour:

```text
hash("abc")
→ compare Digest against expected parsed digest

hash_hex("abc")
→ compare exact 64-character lowercase string

print:
sha256:ok
```

Expected stdout:

```text
sha256:ok
```

Do not merely import the package.

The consumer MUST exercise the declared public surface.

---

# 29. Engine qualification

Where supported, establish equivalent observations through:

```text
HIR reference execution
MIR execution
native debug
native release
```

For pure hashing inputs, compare exact lowercase digest text.

Required representative set:

```text
empty
abc
55 bytes
56 bytes
64 bytes
65 bytes
binary 0..255
multi-block standard vector
```

No engine may use a separate host SHA-256 implementation as its expected value.

The expected hashes are frozen fixtures.

---

# 30. Native implementation prohibition

Search the final patch.

It MUST contain no SHA-256 implementation in:

```text
.rs
.c
.cc
.cpp
.h
```

and no new provider manifest.

There MUST be no call to:

```text
OpenSSL
CommonCrypto
CryptoAPI
ring
sha2 Rust crate
system crypto APIs
```

The point of this package is to implement SHA-256 in STARK.

---

# 31. Dependency rule

The allowed dependency graph is:

```text
stark-sha256
      ↓
stark-digest
      ↓
stark-hex
```

Do not add:

```text
stark-checksum
stark-io
stark-file
stark-random
```

or any host-backed package.

`stark-checksum` may be inspected as an implementation example but is not a dependency.

---

# 32. Security claim boundary

This package MAY claim:

> Implements the SHA-256 message-digest algorithm and matches the package's pinned known-answer vectors.

It MUST NOT claim:

* constant-time execution;
* side-channel resistance;
* password hashing suitability;
* collision resistance beyond the algorithm's standard definition;
* protection from malicious host memory;
* keyed authentication;
* HMAC;
* digital signatures;
* file integrity unless the file's bytes were actually passed to the package;
* cryptographic audit or certification.

README MUST state:

```text
SHA-256 is not a password hashing function.
```

---

# 33. TEST-MATRIX.md

Create a matrix following the existing first-party package convention.

Minimum entries:

```text
API-001          frozen public API
HASH-EMPTY-001   empty
HASH-ABC-001     abc
HASH-LONG-001    standard multi-block
HASH-BINARY-001  00..FF

PAD-055
PAD-056
PAD-063
PAD-064
PAD-065

ARITH-WRAP-001
ARITH-ROT-001
ARITH-ENDIAN-001

DET-001
IMM-001

DIGEST-001        DigestAlgorithm::Sha256
DIGEST-002        exactly 32 bytes
HEX-001           64 lowercase chars

CROSS-001         consumer
CROSS-002         native consumer
ENG-001           engine agreement
```

If million-`a` is included:

```text
LONG-1M-001
```

---

# 34. EVIDENCE.md

Record EXACTLY what was measured.

Include:

```text
baseline SHA
final SHA

package tests
test count

stark check
stark test
stark fmt --check

consumer check
consumer run

HIR result
MIR result
native-debug result
native-release result

qualification script
CI run ID
CI conclusion

known residuals
compiler/runtime blockers
```

Do not write:

```text
all tests passed
```

without naming the command/count.

---

# 35. Qualification script

If `stark-sha256` is intended to become first-party/toolchain-distributed, add it to the current first-party package qualification mechanism using the existing convention.

The consumer must call every public surface.

Do not weaken the qualification script to accommodate the package.

If the package reveals a qualification-script defect, report it separately.

---

# 36. README requirements

README MUST contain:

```text
purpose
API
one example
dependency on stark-digest
pure-STARK implementation statement
no host/provider requirement
supported algorithm: SHA-256 only
explicit exclusions
security claim boundary
```

Example usage:

```stark
use stark_sha256::hash_hex;

fn main() {
    let bytes = "abc".bytes();
    let digest = hash_hex(bytes.as_slice());
    println(digest.as_str());
}
```

Use the actual valid current string/byte API if syntax differs.

---

# 37. Explicitly excluded from v0.1

Do NOT implement:

```text
Sha256 state object
new/update/finalize
streaming
reader APIs
filesystem hashing
SHA-224
SHA-384
SHA-512
SHA-512/224
SHA-512/256
HMAC-SHA256
HKDF
PBKDF2
password hashing
Merkle trees
content IDs
signatures
native acceleration
SIMD
hardware SHA instructions
parallel hashing
constant-time guarantees
```

Future packages can layer:

```text
stark-sha256
   ↓
stark-content-id
```

and later:

```text
stark-io
   +
streaming sha256
   ↓
stark-file-digest
```

Those are separate packets.

---

# 38. Compiler blocker protocol

If implementation reaches a compiler/runtime problem:

1. reduce it to the smallest STARK program;
2. identify whether it occurs in:

   * parser;
   * checker;
   * HIR;
   * MIR lowering;
   * MIR verification;
   * native backend;
3. state expected behaviour;
4. state actual behaviour;
5. record exact compiler SHA;
6. determine whether an equivalent legal Core expression avoids the issue without changing package semantics.

If not:

```text
STOP
STATUS = PARTIAL — COMPILER_RUNTIME_BLOCKER
```

Do not:

* patch `starkc`;
* add native SHA-256;
* reduce the test corpus;
* alter SHA-256 semantics;
* silently use a different integer representation;
* expose implementation details merely to work around the compiler.

---

# 39. Completion status

The packet may finish as exactly one of:

```text
COMPLETE

PARTIAL — COMPILER_RUNTIME_BLOCKER

BLOCKED — PREREQUISITE_NOT_MET
```

No "mostly complete".

---

# 40. Exit criteria

`stark-sha256` v0.1 is COMPLETE only when:

1. `stark-digest` is an actual resolved dependency.
2. The frozen two-function API is implemented.
3. Implementation is pure STARK.
4. No provider or host capability is used.
5. Empty-string vector passes.
6. `abc` vector passes.
7. Standard multi-block vector passes.
8. Binary `00..FF` vector passes.
9. 55/56/63/64/65-byte padding boundaries pass.
10. Modular addition is implemented without relying on STARK overflow.
11. Rotation implementation is legal under STARK's checked-shift semantics.
12. Output is exactly 32 bytes.
13. Output digest algorithm is `Sha256`.
14. `hash_hex` emits exactly 64 lowercase characters.
15. Input immutability is demonstrated.
16. Repeated hashing is deterministic.
17. Cross-package consumer calls the full public surface.
18. HIR/MIR/native observations agree where currently supported.
19. `stark fmt --check` is clean.
20. First-party qualification is green if registered.
21. Broad CI is green on the final PR candidate.
22. `TEST-MATRIX.md` exists.
23. `EVIDENCE.md` records exact evidence.
24. No compiler/runtime/native workaround was introduced.
25. Known residuals are stated rather than hidden.

---

# 41. Desired final report to owner

Return:

```text
STARK SHA-256 v0.1

baseline:
final SHA:

status:
COMPLETE | PARTIAL | BLOCKED

public API:
hash
hash_hex

known-answer vectors:
<passed>/<total>

padding boundaries:
<passed>/<total>

engines:
HIR:
MIR:
native-debug:
native-release:

consumer:
qualification:

CI:
run:
result:

compiler changes:
none

providers:
none

host capabilities:
none

dependencies:
stark-digest 0.1.0

residuals:
...

new compiler deviations:
...
```

Do not claim the package merged until it actually merges.

---

# 42. Governing design principle

The objective is not merely to make this output appear:

```text
ba7816bf...
```

The package must demonstrate that STARK can implement a real bit-oriented cryptographic primitive while preserving its own language semantics:

```text
checked arithmetic
        +
explicit modular arithmetic
        +
byte-order discipline
        +
pure-STARK implementation
        +
three-engine comparison
        +
package qualification
        ↓
reusable SHA-256 primitive
```

If implementing SHA-256 requires weakening those semantics, stop.

The package exists to exercise them.
