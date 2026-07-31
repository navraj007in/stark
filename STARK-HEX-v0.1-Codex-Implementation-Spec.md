# STARK Hex v0.1 — Codex Implementation Work Package

**Package:** `stark-hex`  
**Version:** `0.1.0`  
**Implementation language:** STARK Core v1 only  
**Native code:** Prohibited  
**Compiler changes:** Prohibited  
**Provider access:** Prohibited  
**Third-party dependencies:** Prohibited  
**Primary behaviour:** Strict hexadecimal encoding and decoding  
**Supported alphabets:** Lowercase `0123456789abcdef` and uppercase `0123456789ABCDEF`  
**Status:** Implementation specification — public API and behaviour frozen for this work package  
**Required first action:** Inspect and pin the actual STARK repository head before implementation

---

## 1. Instruction to Codex

You are the implementation engineer for one bounded STARK ecosystem package.

Implement **only** the package specified in this document. Treat every normative statement containing **MUST**, **MUST NOT**, **SHALL**, **SHALL NOT**, **REQUIRED**, or **EXACTLY** as binding.

Do not redesign the public API. Do not add convenience APIs, compiler intrinsics, runtime builtins, native providers, package-manager changes, macros, reflection, generated source, or host-language implementations.

Your responsibilities are:

1. inspect the current repository and verify the prerequisites in Section 5;
2. create the package in the owner-assigned directory;
3. implement the frozen API in pure STARK;
4. add the complete positive, negative, boundary, canonicality, and round-trip test corpus;
5. validate through every currently supported execution engine;
6. reduce and report compiler/runtime blockers without modifying compiler-owned files;
7. record exact evidence;
8. stop when the v0.1 exit criteria are satisfied.

If a required compiler or runtime capability is unavailable, **do not fix the compiler inside this work package**. Follow Section 18, “Compiler blocker protocol”.

---

## 2. Objective

Create a small, production-disciplined hexadecimal package that proves an independently versioned pure-STARK package can:

- encode arbitrary bytes into lowercase hexadecimal;
- encode arbitrary bytes into uppercase hexadecimal;
- strictly decode mixed-case hexadecimal text into bytes;
- reject malformed input deterministically;
- report stable error categories and byte offsets;
- preserve input immutability;
- run without native code, privileged hooks, operating-system access, or third-party dependencies;
- operate consistently through HIR, MIR, and native execution where supported.

The package should be simple enough to qualify fully under the current compiler unless a concrete slice, string, package, or native-lowering blocker exists.

---

## 3. Scope

### 3.1 Included

The package MUST support:

- empty input;
- arbitrary byte input from `0x00` through `0xFF`;
- lowercase encoding;
- uppercase encoding;
- strict decoding of lowercase hexadecimal;
- strict decoding of uppercase hexadecimal;
- strict decoding of mixed-case hexadecimal;
- deterministic invalid-length rejection;
- deterministic invalid-character rejection;
- zero-based UTF-8 byte offsets for invalid input;
- round-trip tests;
- canonical lowercase and uppercase output;
- package-local tests;
- cross-package consumer evidence;
- native evidence when the compiler supports the required program shape.

### 3.2 Explicitly excluded

Do not implement:

- `0x` or `0X` prefixes;
- whitespace-tolerant decoding;
- separators such as `:`, `-`, `_`, or spaces;
- odd-length decoding;
- streaming encoders or decoders;
- reader/writer APIs;
- filesystem APIs;
- in-place encoding or decoding;
- custom alphabets;
- case-preserving decoding metadata;
- constant-time or cryptographic claims;
- SIMD or platform-specific optimisation;
- parallel processing;
- native Rust/C implementation;
- compiler intrinsic or runtime builtin;
- macros, reflection, or generated code;
- checksums, hashing, or cryptographic digests;
- a command-line application, except an optional tiny consumer used solely for acceptance evidence.

Record excluded requests as future work; do not implement them in v0.1.

---

## 4. Frozen design decisions

### 4.1 Encoding output

`encode_lower` MUST emit exactly two lowercase ASCII hexadecimal characters for each input byte.

`encode_upper` MUST emit exactly two uppercase ASCII hexadecimal characters for each input byte.

Examples:

```text
[]                -> ""
[0x00]            -> "00"
[0x0F]            -> "0f" / "0F"
[0x10]            -> "10"
[0xAB]            -> "ab" / "AB"
[0xFF]            -> "ff" / "FF"
[0x00,0x7F,0xFF]  -> "007fff" / "007FFF"
```

### 4.2 Decoding acceptance

`decode` MUST accept hexadecimal digits from both alphabets:

```text
0..9
A..F
a..f
```

Case may be mixed within the same input.

Examples:

```text
"00"      -> [0x00]
"ff"      -> [0xFF]
"FF"      -> [0xFF]
"aB"      -> [0xAB]
"007fFF"  -> [0x00, 0x7F, 0xFF]
```

### 4.3 Strictness

The decoder MUST reject:

- odd-length input;
- prefixes;
- whitespace;
- separators;
- non-ASCII digits;
- any character outside the hexadecimal alphabets.

The decoder MUST NOT trim, normalize, skip, or repair malformed input.

### 4.4 Error precedence

Use this exact order:

1. scan input bytes from left to right and reject the first invalid character;
2. after character validation, reject odd total byte length;
3. decode pairs left to right.

Therefore:

```text
"0x0"
```

returns `InvalidCharacter(1, 'x')`, not `InvalidLength`.

And:

```text
"abc"
```

returns `InvalidLength` because every byte is hexadecimal but the total length is odd.

### 4.5 Input model

Offsets are UTF-8 byte offsets.

All valid hexadecimal input is ASCII. Any multibyte UTF-8 character is invalid, and the error index points to the first byte of that character.

---

## 5. Preconditions and repository inspection

Before editing, inspect the current repository. Do not assume any previously observed SHA is current.

Read only what is necessary to confirm:

- current `COMPILER-STATE.md`;
- current ecosystem/package roadmap;
- current package-manifest implementation;
- existing package examples such as `stark-base64`, `stark-csv`, and `stark-json`;
- current Core rules for `String`, `str`, `Vec`, slices, indexing, loops, casts, and `Result`;
- current package-local test convention;
- current `stark check`, `stark test`, `stark fmt --check`, `stark build`, and `stark run` behaviour.

Confirm these capabilities:

1. `Vec<UInt8>::new`, `push`, `len`, equality, and Drop.
2. `&[UInt8]` parameters.
3. Indexed reads or an equivalent deterministic iteration path over byte slices.
4. `String::new` and `String::push(Char)`.
5. `&str::bytes()` or the current normative equivalent.
6. `UInt8`, `UInt32`, or `UInt64` masks, shifts, arithmetic, and comparisons.
7. Checked or well-defined numeric casts required by the implementation.
8. `Result<T, E>` and enum payload pattern matching.
9. Package entry through `starkpkg.json` or the current approved manifest.
10. Cross-package dependency aliases.
11. Package-local tests.
12. Native compilation for a small consumer executable.

### 5.1 Allowed precondition outcomes

#### `READY`

All required capabilities are available through HIR, MIR, and native debug execution.

#### `INTERPRETER_READY`

The pure-STARK package and HIR tests can run, but MIR/native qualification is incomplete.

Status MUST remain:

```text
PARTIAL — WAITING_COMPILER_RUNTIME
```

#### `BLOCKED`

The package cannot be meaningfully implemented or tested without a compiler/runtime change.

Stop and report the blocker.

Do not silently weaken the API or semantics.

---

## 6. Package placement and allowed files

The owner assigns the package directory.

Preferred structure:

```text
stark-hex/
├── starkpkg.json
├── README.md
├── EVIDENCE.md
├── TEST-MATRIX.md
├── src/
│   ├── lib.stark
│   └── tests.stark
└── fixtures/
    ├── valid/
    └── invalid/
```

If current package tooling cannot load `src/tests.stark`, tests may be placed in `src/lib.stark`. Record the reason in `EVIDENCE.md`.

### 6.1 Allowed modifications

- Files inside the assigned `stark-hex` directory.
- One owner-approved consumer package or fixture for cross-package/native evidence.

### 6.2 Prohibited modifications

Do not modify:

- `starkc/`;
- `stark-runtime/`;
- compiler work packages;
- `COMPILER-STATE.md`;
- specification files;
- package-manager implementation;
- unrelated package code;
- shared CI or release scripts;
- conformance fixtures unrelated to this package.

If one of those areas must change, report a blocker.

---

## 7. Manifest

Use the current approved package format.

Unless the repository has migrated, use:

```json
{
  "name": "stark-hex",
  "version": "0.1.0",
  "entry": "src/lib.stark",
  "dependencies": {}
}
```

The package MUST:

- have no dependencies;
- declare no native provider;
- declare no compiler extension;
- run no build script;
- run no install script;
- require no network access.

---

## 8. Frozen public API

Implement exactly this public API:

```stark
pub enum HexError {
    InvalidLength,
    InvalidCharacter(UInt64, UInt8),
}

pub fn encode_lower(input: &[UInt8]) -> String;

pub fn encode_upper(input: &[UInt8]) -> String;

pub fn decode(input: &str) -> Result<Vec<UInt8>, HexError>;
```

The semicolons describe the API contract; actual STARK source must contain bodies.

### 8.1 Error payload meaning

#### `InvalidLength`

Returned when the UTF-8 byte length is odd after all bytes have been validated as hexadecimal digits.

No position payload is required.

#### `InvalidCharacter(index, value)`

- `index` is the zero-based UTF-8 byte offset of the first invalid byte;
- `value` is the raw invalid byte;
- spaces, tabs, newlines, `x`, punctuation, separators, and bytes above `0x7F` are invalid.

### 8.2 API prohibitions

Do not add:

- generic alphabet parameters;
- `decode_into`;
- `encode_into`;
- `is_hex`;
- `parse_u64`;
- prefix support;
- whitespace-tolerant variants;
- lowercase-only or uppercase-only decoder variants;
- public helper functions;
- public constants;
- error strings or nested causes.

Private helpers are allowed.

---

## 9. Encoding semantics

### 9.1 Lowercase alphabet

Use exactly:

```text
0123456789abcdef
```

### 9.2 Uppercase alphabet

Use exactly:

```text
0123456789ABCDEF
```

### 9.3 Byte mapping

For each byte `b`:

```text
high = (b >> 4) & 0x0F
low  = b & 0x0F
```

Append:

```text
alphabet[high]
alphabet[low]
```

### 9.4 Required properties

Both encoders MUST:

1. accept any byte sequence;
2. accept empty input;
3. process left to right;
4. produce exactly `input.len() * 2` output bytes;
5. emit ASCII only;
6. emit no prefix;
7. emit no whitespace;
8. emit no separators;
9. be deterministic across engines and targets;
10. never mutate or consume the input;
11. never use a native provider or compiler builtin specific to hexadecimal.

### 9.5 Overflow

If computing output capacity requires checked arithmetic and capacity overflow is possible under the current runtime API, do not trap or wrap.

Either:

- build incrementally without precomputing capacity; or
- use a private checked path and report a compiler/runtime blocker if no safe implementation exists.

Do not change the public API to add an allocation error.

---

## 10. Decoding semantics

### 10.1 Hex digit mapping

Use exactly:

```text
'0'..'9' -> 0..9
'a'..'f' -> 10..15
'A'..'F' -> 10..15
```

### 10.2 Validation pass

Scan every UTF-8 byte left to right.

For each byte:

- if hexadecimal, continue;
- otherwise return `InvalidCharacter(index, value)` immediately.

Do not perform pair decoding during this pass if doing so would alter the required error precedence.

### 10.3 Length check

After character validation:

- empty input succeeds;
- even length proceeds;
- odd length returns `InvalidLength`.

### 10.4 Pair decoding

For each pair `(high_char, low_char)`:

```text
high = hex_value(high_char)
low  = hex_value(low_char)
byte = (high << 4) | low
```

Append `byte` to the output vector.

### 10.5 Required properties

`decode` MUST:

1. be strict;
2. preserve byte order;
3. accept mixed case;
4. emit one byte per two input bytes;
5. produce deterministic errors;
6. never mutate the input;
7. never ignore trailing bytes;
8. never use host-language parsing;
9. never interpret the decoded bytes as text;
10. never perform Unicode normalization.

The decoder returns raw bytes. It does not validate whether those bytes form UTF-8.

---

## 11. Private implementation structure

A single-module implementation is acceptable.

Suggested private helpers:

```text
hex_value(byte) -> Option<UInt8>
lower_hex_char(nibble) -> Char
upper_hex_char(nibble) -> Char
validate_hex_input(input) -> Result<UInt64, HexError>
```

Do not expose helpers publicly.

### 11.1 Alphabet representation

Preferred options, in order:

1. fixed `[Char; 16]` arrays;
2. private nibble-to-character functions;
3. another deterministic Core-only representation.

Do not use general string indexing if Core does not define safe element indexing for strings.

---

## 12. Test strategy

Tests MUST be deterministic and self-contained.

No network.

No clock.

No filesystem dependency except package-local checked-in fixtures if supported deterministically.

### 12.1 Required test groups

```text
API
ENCODE_EMPTY
ENCODE_BOUNDARIES
ENCODE_ALL_BYTES
ENCODE_LOWER
ENCODE_UPPER
DECODE_EMPTY
DECODE_LOWER
DECODE_UPPER
DECODE_MIXED
DECODE_INVALID_LENGTH
DECODE_INVALID_CHARACTER
ERROR_PRECEDENCE
ROUND_TRIP
CANONICALITY
DETERMINISM
INPUT_IMMUTABILITY
CROSS_PACKAGE
THREE_ENGINE
```

### 12.2 Required positive vectors

At minimum:

| Bytes | Lowercase | Uppercase |
|---|---|---|
| `[]` | `""` | `""` |
| `[0x00]` | `"00"` | `"00"` |
| `[0x01]` | `"01"` | `"01"` |
| `[0x0F]` | `"0f"` | `"0F"` |
| `[0x10]` | `"10"` | `"10"` |
| `[0x7F]` | `"7f"` | `"7F"` |
| `[0x80]` | `"80"` | `"80"` |
| `[0xAB]` | `"ab"` | `"AB"` |
| `[0xFF]` | `"ff"` | `"FF"` |
| `[0x00,0x7F,0x80,0xFF]` | `"007f80ff"` | `"007F80FF"` |

### 12.3 Full byte-domain test

Construct the byte sequence:

```text
0x00, 0x01, ..., 0xFF
```

Required assertions:

- lowercase encoding matches the frozen expected string;
- uppercase encoding matches the frozen expected string;
- decoding either string returns the original 256-byte sequence;
- mixed-case transformation decodes to the same sequence.

### 12.4 Required valid decode cases

```text
""
"00"
"01"
"0f"
"0F"
"10"
"7f"
"7F"
"80"
"ab"
"AB"
"aB"
"ff"
"FF"
"007f80ff"
"007F80FF"
```

### 12.5 Required invalid-length cases

```text
"0"
"f"
"abc"
"ABC"
"000"
"12345"
```

Each MUST return `InvalidLength`.

### 12.6 Required invalid-character cases

At minimum:

```text
"0x00"      invalid 'x' at 1
" 00"       invalid space at 0
"00 "       invalid space at 2
"00\n"      invalid LF at 2
"00\t"      invalid TAB at 2
"00:11"     invalid ':' at 2
"00-11"     invalid '-' at 2
"00_11"     invalid '_' at 2
"gg"        invalid 'g' at 0
"0g"        invalid 'g' at 1
"é0"        invalid first UTF-8 byte at 0
```

Assert exact error variant, offset, and byte value.

### 12.7 Error precedence tests

Required:

```text
"x"       -> InvalidCharacter(0, 'x'), not InvalidLength
"0x0"     -> InvalidCharacter(1, 'x'), not InvalidLength
"abc"     -> InvalidLength
"abz"     -> InvalidCharacter(2, 'z'), not InvalidLength
"é"       -> InvalidCharacter(0, first UTF-8 byte), not InvalidLength
```

### 12.8 Round-trip tests

For representative byte vectors and the full byte domain:

```text
decode(encode_lower(bytes)) == bytes
decode(encode_upper(bytes)) == bytes
```

### 12.9 Canonicality tests

For any accepted hex input `text`:

```text
encode_lower(decode(text))
```

MUST produce canonical lowercase output.

Similarly:

```text
encode_upper(decode(text))
```

MUST produce canonical uppercase output.

Examples:

```text
"aB" -> lower "ab"
"aB" -> upper "AB"
```

### 12.10 Determinism tests

- same input encoded twice gives identical bytes;
- same text decoded twice gives identical vectors;
- HIR/MIR/native observations agree;
- clean rebuild and relocation do not alter results;
- error classification and offsets are stable.

### 12.11 Input immutability tests

Where the language test surface permits:

- encoding does not consume or mutate the byte vector;
- decoding does not mutate the source string;
- returned output remains valid after local temporaries end.

---

## 13. Fixtures

Create package-local fixtures if the test runner can consume them meaningfully.

Suggested files:

```text
fixtures/valid/empty.txt
fixtures/valid/lower.txt
fixtures/valid/upper.txt
fixtures/valid/mixed.txt
fixtures/invalid/odd.txt
fixtures/invalid/prefix.txt
fixtures/invalid/space.txt
fixtures/invalid/separator.txt
fixtures/invalid/non_ascii.txt
```

Checked-in fixture existence is not evidence by itself.

Every fixture MUST be executed or explicitly mapped to a package test.

---

## 14. Cross-package consumer

Create one owner-approved tiny consumer package or fixture proving:

```stark
use stark_hex::decode;
use stark_hex::encode_lower;

fn main() {
    let bytes = decode("48656c6c6f")?;
    println(encode_lower(bytes.as_slice()).as_str());
}
```

Adjust syntax to current Core rules.

The consumer must demonstrate:

- package resolution;
- public API visibility;
- `Result` crossing the package boundary;
- `Vec<UInt8>` crossing the package boundary;
- output correctness;
- native build and execution where available.

Expected output:

```text
48656c6c6f
```

Do not classify `stark run` as native evidence unless the current CLI contract proves it invokes the native path.

---

## 15. Implementation milestones

### H0 — Repository and capability audit

Deliver:

- current compiler SHA;
- package path;
- toolchain/platform;
- prerequisite table;
- readiness classification.

### H1 — Package skeleton and API

Create:

- manifest;
- public error enum;
- public function signatures;
- README/EVIDENCE/TEST-MATRIX skeletons.

Exit when package checking succeeds.

### H2 — Lowercase encoder

Implement and test `encode_lower`.

Exit when all lowercase vectors and full-byte-domain tests pass.

### H3 — Uppercase encoder

Implement and test `encode_upper`.

Exit when all uppercase vectors and full-byte-domain tests pass.

### H4 — Strict decoder

Implement:

- character validation;
- error precedence;
- length validation;
- pair decoding.

Exit when valid, invalid, and exact-offset tests pass.

### H5 — Round-trip and canonicality

Add complete round-trip and case-canonicalization tests.

### H6 — Cross-package and multi-engine evidence

Run:

- package check;
- package tests;
- formatter check;
- consumer check;
- consumer HIR/run evidence;
- MIR/native evidence;
- Tier-1 targets where infrastructure permits.

### H7 — Closure

Complete:

- README;
- EVIDENCE;
- TEST-MATRIX;
- exact test counts;
- exact commits;
- final status.

---

## 16. Required commands

Adapt to the actual CLI, but record exact commands.

Expected forms:

```bash
stark check
stark test
stark fmt --check
stark build
stark run
```

Run package validation from the package directory.

Run native acceptance from the consumer package if library packages do not have `main`.

Do not treat a library build failure caused solely by absence of `main` as a package semantic failure.

---

## 17. Performance and complexity

Expected complexity:

```text
encode_lower  O(n)
encode_upper  O(n)
decode        O(n)
memory        O(n)
```

The package should not require asymptotically expensive operations.

### 17.1 Bounded measurements

Record, without broad performance claims:

- empty input;
- 16 bytes;
- 1 KiB;
- 1 MiB if current runtime/testing infrastructure handles it safely.

Measure only after correctness is complete.

Record:

- engine;
- target;
- compiler SHA;
- build profile;
- input size.

### 17.2 No optimisation scope expansion

Do not introduce:

- lookup tables requiring unsafe/global mutation;
- SIMD;
- host calls;
- parallelism;
- compiler-specific fast paths.

---

## 18. Compiler blocker protocol

Codex MUST NOT modify compiler/runtime code.

When blocked, create a minimal report containing:

```text
BLOCKER ID
PACKAGE COMMIT
COMPILER COMMIT
EXECUTION ENGINE
MINIMISED STARK SOURCE
COMMAND
EXPECTED RESULT
ACTUAL RESULT
DIAGNOSTICS
REQUIRED CAPABILITY
NORMATIVE BASIS
WORKAROUND CONSIDERED
WHY WORKAROUND WAS REJECTED OR TEMPORARY
```

### 18.1 Blocker classes

Use one:

```text
FRONTEND_REJECTION
TYPE_SYSTEM
BORROW_OR_LIFETIME
MIR_LOWERING
MIR_VERIFIER
NATIVE_BACKEND
RUNTIME_STRING
RUNTIME_VEC
RUNTIME_SLICE
PACKAGE_SYSTEM
TEST_RUNNER
FORMATTER
UNKNOWN
```

### 18.2 No semantic workaround

Do not:

- accept odd lengths;
- ignore invalid characters;
- strip whitespace;
- accept `0x` prefixes;
- replace bytes with text;
- return placeholder output;
- implement in Rust/Python and check in generated answers;
- weaken offsets;
- skip full-byte-domain testing.

A package limitation caused by the compiler must remain explicit.

---

## 19. Documentation requirements

### README.md

Include:

- package purpose;
- exact supported scope;
- public API;
- strict decoding behaviour;
- accepted alphabets;
- error precedence;
- examples;
- exclusions;
- current engine support;
- current package status.

### EVIDENCE.md

Include:

```text
package commit
compiler commit
platform
toolchain
precondition result
commands
test counts
fixture counts and execution status
HIR result
MIR result
native result
cross-package result
formatter result
known blockers
deviations
final status
```

### TEST-MATRIX.md

Each row:

```text
ID
category
input/fixture
expected result
expected error
expected offset/value
engines
status
```

Do not use broad rows as substitutes for executable coverage.

---

## 20. Acceptance criteria

### 20.1 API

- [ ] frozen public API implemented;
- [ ] no extra public API;
- [ ] no dependencies;
- [ ] no native/provider/compiler changes;
- [ ] cross-package import works.

### 20.2 Encoding

- [ ] lowercase alphabet exact;
- [ ] uppercase alphabet exact;
- [ ] empty input works;
- [ ] all byte boundaries work;
- [ ] full 0x00–0xFF domain works;
- [ ] output length is exactly double input length;
- [ ] no prefixes, whitespace, or separators;
- [ ] input is not consumed or mutated.

### 20.3 Decoding

- [ ] lowercase accepted;
- [ ] uppercase accepted;
- [ ] mixed case accepted;
- [ ] empty input accepted;
- [ ] odd lengths rejected;
- [ ] first invalid byte reported exactly;
- [ ] prefixes rejected;
- [ ] whitespace rejected;
- [ ] separators rejected;
- [ ] non-ASCII rejected;
- [ ] error precedence exact;
- [ ] output byte order exact.

### 20.4 Tests

- [ ] required vectors pass;
- [ ] full byte-domain test passes;
- [ ] invalid-length matrix passes;
- [ ] invalid-character matrix passes;
- [ ] exact offsets and byte payloads pass;
- [ ] round trips pass;
- [ ] canonicality passes;
- [ ] determinism passes;
- [ ] every checked-in fixture is executed.

### 20.5 Engines

For full closure:

- [ ] HIR passes;
- [ ] MIR passes;
- [ ] native debug passes;
- [ ] observations agree;
- [ ] cross-package native consumer passes;
- [ ] required Tier-1 evidence recorded.

### 20.6 Evidence

- [ ] README complete;
- [ ] EVIDENCE complete;
- [ ] TEST-MATRIX complete;
- [ ] exact package/compiler commits recorded;
- [ ] no stale evidence SHA;
- [ ] no unsupported completeness claim.

---

## 21. Status vocabulary

Use exactly one.

### `READY — COMPLETE`

Use only when every v0.1 acceptance criterion passes on the required execution engines.

### `IMPLEMENTATION COMPLETE — EXECUTION QUALIFICATION BLOCKED`

Use when source, tests, fixtures, and documentation are complete, but one or more compiler/runtime blockers prevent required engine qualification.

### `PARTIAL — IMPLEMENTATION IN PROGRESS`

Use while package work remains.

### `BLOCKED — COMPILER CAPABILITY`

Use when a compiler/runtime limitation prevents meaningful implementation.

### `BLOCKED — DESIGN AUTHORITY`

Use when a normative/API decision outside Codex authority is required.

Do not use “complete” for HIR-only success unless the status explicitly says execution qualification is blocked.

---

## 22. Adversarial review checklist

Before final report, inspect:

### Encoding

1. Is every byte encoded into exactly two characters?
2. Can signed conversion corrupt bytes above `0x7F`?
3. Can a nibble index exceed `0x0F`?
4. Are lowercase and uppercase alphabets exact?
5. Is input mutated or consumed?
6. Can output capacity arithmetic overflow?

### Decoding

1. Is invalid-character validation performed before odd-length rejection?
2. Is the first invalid byte always reported?
3. Can `0x` be accepted accidentally?
4. Can whitespace or separators be skipped?
5. Can non-ASCII input be misclassified by scalar index instead of byte offset?
6. Can a pair be decoded with one invalid nibble?
7. Can the last nibble be ignored?
8. Is mixed case handled correctly?
9. Are decoded bytes above `0x7F` preserved exactly?

### Tests

1. Is the full byte domain exercised?
2. Are both alphabets frozen by exact expected strings?
3. Are offsets and invalid byte payloads asserted?
4. Are fixtures actually executed?
5. Are skipped tests visible?
6. Is interpreter success misreported as native success?
7. Is the consumer native path actually used?
8. Are evidence SHAs current?

---

## 23. Final report format

Return:

```text
STATUS
PACKAGE PATH
PACKAGE HEAD
COMPILER HEAD
PLATFORM
TOOLCHAIN
FILES CREATED
FILES MODIFIED OUTSIDE PACKAGE
PUBLIC API
PRECONDITION RESULT
LOWERCASE ENCODER RESULT
UPPERCASE ENCODER RESULT
DECODER RESULT
FULL BYTE-DOMAIN RESULT
VALID TEST COUNT
INVALID-LENGTH TEST COUNT
INVALID-CHARACTER TEST COUNT
ROUND-TRIP TEST COUNT
CANONICALITY TEST COUNT
FIXTURE COUNT
FIXTURE EXECUTION RESULT
HIR RESULT
MIR RESULT
NATIVE RESULT
CROSS-PACKAGE RESULT
FORMAT RESULT
PERFORMANCE OBSERVATIONS
COMPILER BLOCKERS
PACKAGE LIMITATIONS
DEVIATIONS FROM SPEC
RECOMMENDED STATUS
```

Expected normal answer:

```text
FILES MODIFIED OUTSIDE PACKAGE: stark-hex-consumer only
```

or:

```text
FILES MODIFIED OUTSIDE PACKAGE: NONE
```

---

## 24. Direct execution prompt

Implement `stark-hex` v0.1 as a pure-STARK package using this document as the binding work package.

Re-pin the repository first and classify prerequisites as `READY`, `INTERPRETER_READY`, or `BLOCKED`. Work only inside the owner-assigned package directory plus one explicitly approved consumer fixture. Do not modify the compiler, runtime, specifications, package manager, shared test infrastructure, compiler state, or unrelated packages.

Implement exactly:

```stark
pub enum HexError {
    InvalidLength,
    InvalidCharacter(UInt64, UInt8),
}

pub fn encode_lower(input: &[UInt8]) -> String;
pub fn encode_upper(input: &[UInt8]) -> String;
pub fn decode(input: &str) -> Result<Vec<UInt8>, HexError>;
```

Encoding must emit exactly two ASCII hexadecimal characters per input byte with no prefix, whitespace, or separators. Lowercase and uppercase outputs must use their exact frozen alphabets.

Decoding must accept lowercase, uppercase, and mixed case. It must first scan left to right for the first invalid byte, then reject odd length, then decode pairs. Reject `0x`, whitespace, separators, non-ASCII, and all non-hexadecimal bytes. Error offsets are zero-based UTF-8 byte offsets.

Build the full positive, invalid-length, invalid-character, exact-offset, full-byte-domain, round-trip, canonicality, determinism, input-immutability, cross-package, and multi-engine test matrix. Every checked-in fixture must be executed. Do not return placeholder output and do not weaken strictness to accommodate compiler limitations.

When a compiler/runtime issue appears, minimize and report it through the blocker protocol. Finish with README.md, EVIDENCE.md, TEST-MATRIX.md, exact commands/counts, exact package/compiler commits, and the final status report in Section 23.
