# STARK Base64 v0.1 — Gemini Implementation Work Package

**Package:** `stark-base64`  
**Version:** `0.1.0`  
**Implementation language:** STARK Core v1 only  
**Native code:** Prohibited  
**Compiler changes:** Prohibited  
**Primary standard:** RFC 4648, Section 4, standard Base64 alphabet  
**Status:** Implementation specification — public API and behaviour are frozen for this work package  
**Repository baseline inspected:** STARK `main` at CD-099 (`23ea2fc08b04ce6d65d00dc2652703da5b7fba6a`, 24 July 2026)

---

## 1. Instruction to Gemini

You are the implementation engineer for one bounded STARK ecosystem package.

Implement **only** the package specified in this document. Treat every normative statement containing **MUST**, **MUST NOT**, **SHALL**, **SHALL NOT**, or **EXACTLY** as binding.

Do not redesign the API. Do not add convenience functions. Do not modify STARK syntax, compiler semantics, runtime representation, package resolution, standard-library specifications, or test infrastructure.

Your job is to:

1. inspect the current repository and verify the prerequisites in Section 4;
2. create the package in the package directory assigned by the owner;
3. implement the frozen public API in pure STARK;
4. add the complete positive, negative, canonicality, boundary, and round-trip test corpus;
5. run all required checks that the current toolchain supports;
6. record exact evidence and any blockers;
7. stop without widening scope.

If a required compiler/runtime capability is unavailable, **do not fix the compiler**. Follow the blocker protocol in Section 17.

---

## 2. Objective

Create a small but production-disciplined Base64 package that proves an ordinary independently versioned STARK package can be implemented without:

- compiler modifications;
- privileged hooks;
- native providers;
- operating-system access;
- unsafe code;
- third-party dependencies.

The package shall encode and strictly decode the standard RFC 4648 Base64 alphabet:

```text
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/
```

Encoding always emits padding when the final input group contains fewer than three bytes. Decoding is strict and accepts only canonical inputs whose total byte length is a multiple of four: complete quartets may have no padding, while a final one-byte or two-byte group must use two or one `=` characters respectively.

---

## 3. Scope

### 3.1 Included

- Encoding arbitrary bytes into standard Base64.
- Strict decoding from `&str` to bytes.
- Empty input.
- Binary data containing every byte value from `0x00` through `0xFF`.
- Required `=` padding on encoder output.
- Strict padding validation on decoder input.
- Rejection of whitespace.
- Rejection of URL-safe `-` and `_` characters.
- Rejection of non-ASCII input bytes.
- Rejection of noncanonical unused trailing bits.
- Deterministic error classification and byte offsets.
- Package documentation and examples.
- Interpreter tests.
- Native evidence when the current C6.3 implementation supports all required runtime values.

### 3.2 Explicitly excluded

Do not implement any of the following:

- URL-safe Base64.
- Optional omission of padding.
- MIME Base64.
- Line wrapping.
- Whitespace-tolerant decoding.
- Streaming encoders or decoders.
- Reader/writer APIs.
- In-place encode/decode.
- Custom alphabets.
- Constant-time or cryptographic claims.
- SIMD or platform-specific optimisation.
- Parallel processing.
- Native Rust implementation.
- Compiler intrinsic or runtime builtin.
- Macros, reflection, generated source, or compiler plugins.
- A command-line application, except an optional tiny consumer used solely for native acceptance evidence.

Record excluded requests as future work; do not implement them in v0.1.

---

## 4. Preconditions and repository inspection

Before editing, inspect the current repository state. Do not assume the baseline SHA above is still current.

Read only the files necessary to confirm:

- current `COMPILER-STATE.md`;
- current ecosystem charter/roadmap, when present;
- `STARKLANG/docs/spec/03-Type-System.md`;
- `STARKLANG/docs/spec/06-Standard-Library.md`;
- `STARKLANG/docs/spec/07-Modules-and-Packages.md`;
- current package-manifest implementation or package examples;
- current `stark test` naming convention;
- any active C6.3 work-package status affecting `String`, `Vec`, slices, indexing, and output.

Confirm all of these capabilities:

1. `String::new`, `String::push`, `String::as_str` and related ownership work in the intended execution engine.
2. `Vec<UInt8>::new`, `push`, `len`, `as_slice`, equality, and Drop work.
3. `&[UInt8]` parameters and indexed reads work.
4. `&str::bytes()` or the current normative equivalent provides UTF-8 bytes.
5. Fixed arrays of `Char` and array indexing work.
6. `UInt8`, `UInt32` or `UInt64` bitwise shifts, masks, addition, and subtraction work.
7. Numeric casts required by the implementation work with checked STARK semantics.
8. `Result`, enum payloads, pattern matching, `Option`, and `?` work.
9. A package entry can be `src/lib.stark` through `starkpkg.json`.
10. `stark check`, `stark test`, and `stark fmt --check` work for an ordinary package.

### 4.1 Allowed precondition outcomes

- **READY:** all capabilities needed for implementation and native acceptance are available.
- **INTERPRETER_READY:** pure STARK implementation and interpreter tests can run, but native C6.3 support is incomplete. Implementation may proceed, but the package remains `PARTIAL — WAITING_C6.3`.
- **BLOCKED:** the package cannot be meaningfully implemented or tested without a compiler/runtime change.

Never convert `INTERPRETER_READY` or `BLOCKED` into a compiler task inside this work package.

---

## 5. Package placement and allowed files

The owner shall assign the package directory. Do not invent or restructure repository-level ecosystem directories.

Inside the assigned directory, create this structure unless current package tooling requires a narrowly documented variation:

```text
stark-base64/
├── starkpkg.json
├── README.md
├── EVIDENCE.md
└── src/
    ├── lib.stark
    └── tests.stark
```

If the current test runner does not discover tests in a module file, tests may be placed in `src/lib.stark`. Record the reason in `EVIDENCE.md`.

### 5.1 Allowed modifications

- Files inside the assigned `stark-base64` package directory only.
- An optional tiny consumer package inside a location explicitly assigned by the owner, solely for cross-package/native evidence.

### 5.2 Prohibited modifications

Do not modify:

- `starkc/`;
- `stark-runtime/`;
- `STARKLANG/docs/spec/`;
- `COMPILER-STATE.md`;
- compiler work-package or decision ledgers;
- root package-manager implementation;
- conformance fixtures unrelated to this package;
- release scripts;
- CI outside a package-local workflow explicitly requested by the owner.

If the package cannot pass because one of those areas needs a change, report a blocker.

---

## 6. Manifest

Use this manifest unless the current package format has changed through an approved versioned migration:

```json
{
  "name": "stark-base64",
  "version": "0.1.0",
  "entry": "src/lib.stark",
  "dependencies": {}
}
```

The package MUST have no dependencies.

Do not add a native provider declaration, build script, install script, compiler extension, capability declaration, or network access.

---

## 7. Frozen public API

Implement exactly this public API:

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

Because ordinary STARK function definitions require bodies, the semicolons above describe the API contract; actual source must contain function bodies.

### 7.1 Error payload meaning

- `InvalidLength`
  - The UTF-8 byte length is not divisible by four.
  - No index payload is needed.

- `InvalidCharacter(index, value)`
  - `index` is the zero-based UTF-8 byte offset of the first byte that is neither a standard Base64 alphabet byte nor `=`.
  - `value` is the invalid raw UTF-8 byte.
  - Space, tab, CR, LF, `-`, `_`, and every byte greater than `0x7F` are invalid.

- `InvalidPadding(index)`
  - `index` is the first byte offset that violates padding placement or count.

- `NonCanonicalTrailingBits(index)`
  - `index` identifies the alphabet byte in the final quartet whose unused low bits are nonzero.

Do not add error strings, nested causes, source spans, or additional variants in v0.1.

---

## 8. Encoding semantics

`encode` MUST:

1. accept any byte sequence, including empty input;
2. use the exact standard alphabet;
3. process input left-to-right in groups of three bytes;
4. emit four Base64 characters for each complete three-byte group;
5. emit `==` for a final group containing one byte;
6. emit `=` for a final group containing two bytes;
7. emit no whitespace or line breaks;
8. return valid UTF-8;
9. produce deterministic output independent of target and execution engine;
10. never mutate the input;
11. never use a native provider or compiler builtin specific to Base64.

### 8.1 Bit mapping

For bytes `a`, `b`, and `c`, promoted to a sufficiently wide unsigned integer type:

```text
s0 = a >> 2
s1 = ((a & 0x03) << 4) | (b >> 4)
s2 = ((b & 0x0F) << 2) | (c >> 6)
s3 = c & 0x3F
```

For one remaining byte `a`:

```text
s0 = a >> 2
s1 = (a & 0x03) << 4
output = alphabet[s0], alphabet[s1], '=', '='
```

For two remaining bytes `a`, `b`:

```text
s0 = a >> 2
s1 = ((a & 0x03) << 4) | (b >> 4)
s2 = (b & 0x0F) << 2
output = alphabet[s0], alphabet[s1], alphabet[s2], '='
```

Each sextet index is in `0..64`.

### 8.2 Alphabet representation

Prefer a fixed `[Char; 64]` constant and indexed lookup.

Do not use string indexing, because STARK string offsets are UTF-8 byte offsets and Core defines no general string element operator.

If the current compiler cannot use a fixed `Char` array constant, use a private deterministic mapping function. Do not change the public API or compiler.

---

## 9. Decoding semantics

`decode` MUST be strict.

### 9.1 Accepted bytes

Only these bytes are alphabet bytes:

```text
'A'..'Z' => 0..25
'a'..'z' => 26..51
'0'..'9' => 52..61
'+'      => 62
'/'      => 63
```

`=` is padding, not an alphabet byte.

### 9.2 Validation and error precedence

Use this exact order so malformed input produces deterministic errors:

1. **Character scan**
   - Scan the UTF-8 bytes left-to-right.
   - Alphabet bytes and `=` are provisionally valid.
   - On the first other byte, return `InvalidCharacter(index, value)`.

2. **Length check**
   - If byte length modulo four is not zero, return `InvalidLength`.
   - Empty input succeeds.

3. **Padding structure**
   - Padding count may be zero, one, or two.
   - Padding may occur only in the final quartet.
   - If `=` first appears, every following byte must also be `=`.
   - The first two positions of a quartet may never be padding.
   - One padding byte is legal only at the fourth position of the final quartet.
   - Two padding bytes are legal only at the third and fourth positions of the final quartet.
   - Three or more padding bytes are invalid.
   - Return `InvalidPadding` with the first byte offset that demonstrates the violation.

4. **Canonical trailing-bit check**
   - For `xx==`, the low four bits of the second sextet MUST be zero. Otherwise return `NonCanonicalTrailingBits` at the second sextet's byte offset.
   - For `xxx=`, the low two bits of the third sextet MUST be zero. Otherwise return `NonCanonicalTrailingBits` at the third sextet's byte offset.

5. **Decode**
   - Decode complete quartets left-to-right.
   - Do not emit bytes represented only by padding.

### 9.3 Output mapping

For sextets `s0`, `s1`, `s2`, `s3`:

```text
b0 = (s0 << 2) | (s1 >> 4)
b1 = ((s1 & 0x0F) << 4) | (s2 >> 2)
b2 = ((s2 & 0x03) << 6) | s3
```

- No padding: emit `b0`, `b1`, `b2`.
- One `=`: emit `b0`, `b1`.
- Two `=`: emit `b0`.

Use explicit checked casts to `UInt8` only after masking or constructing values known to fit.

### 9.4 Safety requirements

Malformed input MUST return `Err`; it MUST NOT cause:

- out-of-bounds indexing;
- arithmetic traps;
- a language panic;
- a native host failure;
- partial output observable to the caller.

It is acceptable for an allocation failure on an extremely large valid input to remain a classified host/resource failure under normal STARK runtime rules.

---

## 10. Private implementation helpers

The following private helpers are recommended. Names may vary, but their behaviour must remain private and deterministic:

```stark
fn encode_char(value: UInt8) -> Char;
fn decode_sextet(value: UInt8) -> Option<UInt8>;
fn validate_padding(bytes: &[UInt8]) -> Result<UInt64, Base64Error>;
```

`validate_padding` may return padding count or another private result convenient to the implementation.

Do not expose helper functions publicly.

Avoid cleverness. Prefer a two-pass decoder:

1. validate characters, length, padding, and canonical bits;
2. allocate/construct output and decode.

This ensures no partial result is produced before validation completes.

---

## 11. Required examples and known vectors

The following mappings MUST pass exactly:

| Input bytes / text | Encoded output |
|---|---|
| empty | `""` |
| `f` | `Zg==` |
| `fo` | `Zm8=` |
| `foo` | `Zm9v` |
| `foob` | `Zm9vYg==` |
| `fooba` | `Zm9vYmE=` |
| `foobar` | `Zm9vYmFy` |
| `[0x00]` | `AA==` |
| `[0x00, 0x00]` | `AAA=` |
| `[0x00, 0x00, 0x00]` | `AAAA` |
| `[0xFF]` | `/w==` |
| `[0xFF, 0xFF]` | `//8=` |
| `[0xFF, 0xFF, 0xFF]` | `////` |

Both encode and decode directions must be tested.

---

## 12. Required test corpus

Tests use the current STARK naming convention:

```text
fn test_<descriptive_name>()
```

Do not introduce `@test`, `#[test]`, macros, or new syntax. Verify error values by `match` and payload assertions; do not widen the package API merely to make `Base64Error` directly comparable.

### 12.1 Positive encoding tests

At minimum:

- empty input;
- RFC `f`, `fo`, `foo`, `foob`, `fooba`, `foobar` vectors;
- zero-byte vectors;
- `0xFF` vectors;
- all three remainder classes: length `% 3 == 0`, `1`, `2`;
- a vector whose output uses `+`;
- a vector whose output uses `/`;
- deterministic repeated encode of identical input.

### 12.2 Positive decoding tests

At minimum:

- empty input;
- every RFC vector;
- zero and `0xFF` vectors;
- an unpadded quartet such as `Zm9v`;
- valid one-padding and two-padding final quartets;
- a valid string containing `+` and `/`;
- the complete alphabet string:

```text
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/
```

Decode it and encode the resulting bytes; the result must equal the original alphabet string.

### 12.3 Invalid character tests

Test the exact error variant, index, and byte for:

- space;
- tab;
- LF;
- CR;
- `-`;
- `_`;
- `.`;
- `:`;
- a non-ASCII UTF-8 character, verifying the first invalid UTF-8 byte offset;
- invalid character at the first, middle, and final positions.

Examples:

```text
"@AAA"  -> InvalidCharacter(0, 0x40)
"AA_A"  -> InvalidCharacter(2, 0x5F)
"AAA-"  -> InvalidCharacter(3, 0x2D)
"Zm 9v" -> InvalidCharacter(2, 0x20)
```

### 12.4 Invalid length tests

After the character scan succeeds, these must return `InvalidLength`:

```text
"A"
"AA"
"AAA"
"AAAAA"
"AAAAAA"
"AAAAAAA"
"AAA==="
```

Include at least one alphabet-only invalid-length input longer than one quartet.

### 12.5 Invalid padding tests

Test these exact results:

```text
"=AAA"     -> InvalidPadding(0)
"A=AA"     -> InvalidPadding(1)
"AA=A"     -> InvalidPadding(3)
"A==="     -> InvalidPadding(1)
"===="     -> InvalidPadding(0)
"AAAA====" -> InvalidPadding(4)
"AA==AAAA" -> InvalidPadding(4)
"AAAA=AAA" -> InvalidPadding(4)
```

Also test:

- a non-padding byte after the first `=`;
- padding in a non-final quartet;
- three padding bytes;
- four padding bytes;
- padding in position 0 or 1 of the final quartet.

### 12.6 Noncanonical trailing-bit tests

These inputs may map to bytes in permissive decoders but MUST be rejected:

```text
"Zh==" -> NonCanonicalTrailingBits(1)
"Zm9=" -> NonCanonicalTrailingBits(2)
"AB==" -> NonCanonicalTrailingBits(1)
"AAB=" -> NonCanonicalTrailingBits(2)
```

Also include the corresponding canonical forms to prove the check is not overbroad:

```text
"Zg=="
"Zm8="
"AA=="
"AAA="
```

### 12.7 Round-trip tests

For each length below, construct deterministic bytes, encode, decode, and assert exact equality:

```text
0, 1, 2, 3, 4, 5, 6, 7,
15, 16, 17,
31, 32, 33,
63, 64, 65,
255, 256, 257,
1023, 1024, 1025
```

Use a deterministic pattern such as:

```text
byte(i) = (i * 73 + 19) modulo 256
```

Use sufficiently wide arithmetic and cast only after reducing modulo 256.

### 12.8 Full byte-domain test

Construct a vector containing every byte from `0x00` through `0xFF` exactly once. Require:

```text
decode(encode(all_bytes)) == all_bytes
```

### 12.9 Boundary and regression tests

- No index expression may evaluate past the end of input.
- Inputs ending exactly at a quartet boundary must work.
- Final one-byte and two-byte groups must work.
- Repeated calls must not retain state from earlier calls.
- Decoding an error must not affect a subsequent successful decode.
- Empty input must not access index zero.

---

## 13. Host-oracle differential check

In addition to committed STARK tests, perform a development-time differential comparison against a trusted host implementation, such as Python's standard `base64` module or Rust's established Base64 crate.

Requirements:

1. Use a fixed deterministic seed or deterministic byte construction.
2. Compare at least 1,000 byte vectors with lengths distributed across `0..4096`.
3. Compare exact encoded output.
4. Decode the host output with STARK and compare original bytes.
5. Decode STARK output with the host and compare original bytes.
6. Include all lengths around multiples of three.
7. Do not add the host implementation as a runtime package dependency.

A temporary generator or comparison script may be used outside package runtime code. Do not commit generated megabytes of vectors. Record the command, seed, count, and result in `EVIDENCE.md`.

Strict malformed-input behaviour need not equal a permissive host decoder. The STARK package's error semantics in this specification are authoritative.

---

## 14. Documentation requirements

### 14.1 `README.md`

Document:

- package purpose;
- standard alphabet;
- strict decoder policy;
- frozen public API;
- encode example;
- decode example;
- error handling example;
- explicit exclusions for URL-safe, MIME, whitespace-tolerant, and streaming forms;
- statement that the package is pure STARK and has no native code or dependencies;
- package version.

Do not claim cryptographic security, constant-time behaviour, or exceptional performance.

### 14.2 Source documentation

Add concise documentation comments to:

- `Base64Error`;
- each error variant;
- `encode`;
- `decode`.

Do not duplicate the entire algorithm in comments. Comments should describe contract and error behaviour.

### 14.3 `EVIDENCE.md`

Use the template in Section 18 and fill it with exact commands, toolchain identity, pass counts, native status, and blockers.

---

## 15. Required commands and evidence

Run commands from the package directory using the current repository's `stark` binary or the equivalent Cargo invocation.

At minimum run:

```bash
stark check
stark test
stark fmt --check
stark doc
```

If using the repository binary rather than an installed binary, record the exact equivalent, for example:

```bash
cargo run --manifest-path <repo>/starkc/Cargo.toml --bin stark -- check
cargo run --manifest-path <repo>/starkc/Cargo.toml --bin stark -- test
cargo run --manifest-path <repo>/starkc/Cargo.toml --bin stark -- fmt --check
cargo run --manifest-path <repo>/starkc/Cargo.toml --bin stark -- doc
```

### 15.1 Native acceptance

When supported by current C6.3 and package tooling:

1. create or use a tiny consumer application with a path dependency on `stark-base64`;
2. build it with the normal native `stark build` path;
3. run positive encode/decode smoke cases;
4. run at least one invalid-input case and verify the expected error variant;
5. record platform, target, compiler commit, generated backend, command, and result.

Do not claim native completion when only the HIR interpreter has executed the package.

### 15.2 Repository regression check

Because this package must not modify compiler code, a full compiler workspace run is not automatically required. However, run any package/ecosystem regression suite specified by the current ecosystem charter or active gate.

If the owner requests a full workspace check, run it and record exact results.

---

## 16. Definition of done

The package is **COMPLETE** only when all are true:

1. Manifest is valid and dependency-free.
2. Public API exactly matches Section 7.
3. Encoder behaviour exactly matches Section 8.
4. Decoder validation order and errors exactly match Section 9.
5. All required tests in Section 12 pass.
6. Host-oracle differential check in Section 13 passes.
7. Formatting check passes.
8. Documentation generation passes.
9. No files outside the assigned package/consumer scope changed.
10. No compiler, runtime, or specification modification was required.
11. Native acceptance evidence exists on the currently supported host target.
12. `EVIDENCE.md` is complete and honest.

If items 1–10 pass but item 11 is blocked solely by an explicitly open C6.3 capability, status is:

```text
PARTIAL — IMPLEMENTATION COMPLETE, WAITING_C6.3_NATIVE_EVIDENCE
```

Do not relabel that state as complete.

---

## 17. Blocker and escalation protocol

Stop and report a blocker when any of these occurs:

- required normative API is not accepted by the compiler;
- `String`, `Vec`, slices, byte access, indexing, enum payloads, or bitwise operations are unavailable in the required engine;
- ordinary cross-package use cannot expose the API without compiler modification;
- native code generation refuses a required C6.3 value shape;
- package test discovery cannot run package-local tests without changing the test runner;
- the current package manifest format conflicts with Section 6 and no approved migration exists;
- correctness would require a public API change.

### 17.1 Required blocker report

Write the following in `EVIDENCE.md` and in your final response:

```text
BLOCKED: <short identifier>

Exact source shape:
<minimal STARK program or package structure>

Expected behaviour:
<what this specification requires>

Actual behaviour:
<diagnostic, refusal, trap, or tooling limitation>

Stage:
parse | resolve | typecheck | borrow-check | HIR | MIR lower | MIR verify | emit | rustc | link | run | package tooling

Why package-local alternatives fail:
1. <alternative attempted>
2. <alternative attempted>

Files changed before stopping:
<list>

Recommended owner routing:
C6.3 | package tooling | ecosystem contract | other
```

Do not implement the recommended upstream fix.

---

## 18. `EVIDENCE.md` template

```markdown
# stark-base64 v0.1 Evidence

## Status

COMPLETE | PARTIAL — IMPLEMENTATION COMPLETE, WAITING_C6.3_NATIVE_EVIDENCE | BLOCKED

## Baseline

- Repository commit:
- Date:
- STARK compiler version/commit:
- Host OS:
- Host architecture:
- Rust toolchain used by native backend:

## Files created or modified

- ...

## Public API audit

- [ ] `Base64Error` exact
- [ ] `encode` exact
- [ ] `decode` exact
- [ ] no additional public items

## Commands

### Check

Command:

```text
...
```

Result:

```text
...
```

### Tests

Command:

```text
...
```

Result and pass count:

```text
...
```

### Format

Command and result:

```text
...
```

### Documentation

Command and result:

```text
...
```

### Native consumer

Command and result, or exact C6.3 blocker:

```text
...
```

## Required corpus summary

- RFC vectors:
- invalid characters:
- invalid lengths:
- invalid padding:
- noncanonical trailing bits:
- round-trip lengths:
- full byte-domain:
- state/regression cases:

## Host-oracle differential

- Oracle:
- Version:
- Deterministic seed/pattern:
- Case count:
- Length range:
- Result:

## Scope audit

- [ ] pure STARK
- [ ] no dependencies
- [ ] no native provider
- [ ] no compiler changes
- [ ] no spec changes
- [ ] no files outside assigned scope
- [ ] no URL-safe/MIME/streaming features

## Deviations or blockers

None | <full blocker report>

## Final conclusion

...
```

---

## 19. Expected implementation approach

A straightforward implementation should be approximately:

- one public error enum;
- two public functions;
- two to four private helpers;
- one fixed alphabet constant;
- roughly 100–250 lines of package implementation, depending on current STARK verbosity;
- a larger test file containing the complete required corpus.

Do not optimise for minimum line count. Optimise for:

- obvious bounds safety;
- exact error behaviour;
- easy review;
- deterministic tests;
- no hidden state;
- no upstream changes.

Avoid combining validation, output allocation, and decoding into one difficult loop if a clear two-pass decoder is possible.

---

## 20. Final response required from Gemini

Return a compact implementation report containing:

1. status: `COMPLETE`, `PARTIAL`, or `BLOCKED`;
2. files created/modified;
3. exact public API implemented;
4. test count and categories;
5. commands and results;
6. host-oracle result;
7. native evidence or exact C6.3 blocker;
8. confirmation that no compiler/spec/runtime files changed;
9. commit SHA, when committed;
10. any follow-up explicitly outside v0.1.

Do not claim success without command evidence. Do not describe an interpreter-only result as native completion.

---

## 21. Owner acceptance checklist

The owner/reviewer should reject the implementation if any answer is “no”:

- Is the package pure STARK?
- Is the API exact?
- Does encoding always use standard padded Base64?
- Does decoding reject whitespace and URL-safe characters?
- Are invalid character offsets UTF-8 byte offsets?
- Is padding restricted to the final quartet?
- Are noncanonical unused trailing bits rejected?
- Are all 256 byte values round-tripped?
- Are boundary lengths around multiples of three tested?
- Is malformed input returned as typed `Err` rather than a trap?
- Are tests deterministic?
- Is the package dependency-free?
- Did implementation avoid compiler and spec changes?
- Is native evidence clearly distinguished from interpreter evidence?
- Is `EVIDENCE.md` complete?

Only after every item passes should `stark-base64` v0.1 be accepted.
