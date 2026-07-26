# stark-hex v0.1 Evidence

## Audit

- Package: `stark-hex`
- Package path: `/Users/nexper/Documents/GitHub/stark/stark-hex`
- Compiler/repository head at implementation start: `7b5c10b0fe181e76aeef3c008ae186cc107b01ca`
- Platform: macOS host, `Australia/Sydney` local timezone
- Toolchain: repository `starkc/target/debug/stark`
- Precondition result: `INTERPRETER_READY`

## Prerequisites

| Capability | Result | Evidence |
|---|---|---|
| `Vec<UInt8>::new`, `push`, `len`, equality, Drop | available | package check/tests |
| `&[UInt8]` parameters | available | `stark-base64` precedent and package check |
| deterministic indexed reads over byte slices | available | encoders/tests |
| `String::new` and `String::push(Char)` | available | encoders/tests |
| `&str::bytes()` | available | decoder/tests |
| UInt masks, shifts, arithmetic, comparisons | available | encoders/decoder |
| numeric casts | available | full-domain vector construction |
| `Result<T,E>` and enum payload matching | available | decoder/tests |
| `starkpkg.json` package entry | available | package check |
| cross-package dependency aliases | available | `stark-hex-consumer` check/run |
| package-local tests | available | `mod tests;` |
| native compilation for consumer | blocked | `Vec::as_slice` native build gap |

## Commands

```bash
# from /Users/nexper/Documents/GitHub/stark/stark-hex
../starkc/target/debug/stark check
# result: stark-hex: OK

../starkc/target/debug/stark test
# result: 9 passed; 0 failed; 0 ignored; 212ms total

../starkc/target/debug/stark fmt --check
# result: pass

../starkc/target/debug/stark build
# result: blocked for library package: program without a `main` function

# from /Users/nexper/Documents/GitHub/stark/stark-hex-consumer
../starkc/target/debug/stark check
# result: stark-hex-consumer: OK

../starkc/target/debug/stark run
# result stdout: 48656c6c6f

../starkc/target/debug/stark fmt --check
# result: pass

../starkc/target/debug/stark build
# result: blocked: native build does not yet support this program: Vec::as_slice
```

## Test Counts

- Package tests: 9 passed, 0 failed, 0 ignored
- Valid decode cases: 16 minimum, covered by tests
- Invalid-length cases: 6
- Invalid-character cases: 11 plus precedence cases
- Round-trip cases: full domain lower/upper
- Canonicality cases: mixed-case `aB`
- Fixtures: 9 checked in, mapped to package tests

## Fixture Status

Fixtures are checked in under `fixtures/valid` and `fixtures/invalid`. The current package tests use
the same byte/text cases directly because the STARK test runner does not provide package-local file
I/O for pure Core tests.

## Engine Results

- HIR/package check: pass
- Package test runner: pass
- Formatter: pass
- MIR: not separately exposed by the package CLI in this run
- Native: blocked by `Vec::as_slice` in consumer; library native build also has no `main`
- Cross-package: check pass; interpreter run pass; native build blocked

## Known Blockers

### BLOCKER HEX-NATIVE-001

```text
BLOCKER ID: HEX-NATIVE-001
PACKAGE COMMIT: uncommitted worktree
COMPILER COMMIT: 7b5c10b0fe181e76aeef3c008ae186cc107b01ca
EXECUTION ENGINE: native build
MINIMISED STARK SOURCE:
  use stark_hex::decode;
  use stark_hex::encode_lower;
  fn main() {
      match decode("48656c6c6f") {
          Ok(bytes) => println(encode_lower(bytes.as_slice()).as_str()),
          Err(_) => panic("decode failed"),
      }
  }
COMMAND: ../starkc/target/debug/stark build
EXPECTED RESULT: native executable builds and prints 48656c6c6f when run
ACTUAL RESULT: error: native build does not yet support this program: Vec::as_slice (a later C4.5e sub-slice)
DIAGNOSTICS: native build refusal before executable generation
REQUIRED CAPABILITY: native support for Vec::as_slice when calling a public &[UInt8] API
NORMATIVE BASIS: STARK-HEX-v0.1 frozen API requires encode_lower(input: &[UInt8])
WORKAROUND CONSIDERED: change public API to &Vec<UInt8>
WHY WORKAROUND WAS REJECTED OR TEMPORARY: rejected because it weakens the frozen public API
CLASS: NATIVE_BACKEND
```

## Deviations

The fixture files are not read at runtime; their cases are executed in `src/tests.stark`. This avoids
filesystem APIs, which are excluded from the package implementation.

The consumer `main` returns `Unit` rather than `Result<(), HexError>` because the current executable
entry contract only admits `Unit`, `Int32`, `Result<Unit, String>`, or `Result<Int32, String>`.
The consumer still exercises `Result<Vec<UInt8>, HexError>` across the package boundary via `decode`.

## Final Status

`IMPLEMENTATION COMPLETE — EXECUTION QUALIFICATION BLOCKED`
