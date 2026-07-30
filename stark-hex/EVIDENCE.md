# stark-hex v0.1 Evidence

## Audit

- Package: `stark-hex`
- Package path: `/Users/nexper/Documents/GitHub/stark/stark-hex`
- Compiler/repository head at latest qualification: `159c7aa64c96e6ce7f7734ece3e0602f656fcd9c`
- Platform: macOS host, `Australia/Sydney` local timezone
- Toolchain: repository `starkc/target/debug/stark`
- Precondition result: `READY` for local package/native consumer execution; final baseline
  qualification pending C6.5 freeze

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
| native compilation for consumer | available | native build and generated binary run |

## Commands

```bash
# from /Users/nexper/Documents/GitHub/stark/stark-hex
../starkc/target/debug/stark check
# result: stark-hex: OK

../starkc/target/debug/stark test
# result: 10 passed; 0 failed; 0 ignored; 333ms total

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
# result: Built stark-hex-consumer [debug] -> target/stark/debug/stark-hex-consumer

./target/stark/debug/stark-hex-consumer
# result stdout: 48656c6c6f
```

## Test Counts

- Package tests: 10 passed, 0 failed, 0 ignored
- Valid decode cases: 16 minimum, covered by tests
- Invalid-length cases: 6
- Invalid-character cases: 11 named cases, all 106 invalid ASCII bytes, plus precedence cases
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
- MIR: package CLI does not expose a separate final MIR evidence command in this run
- Native: consumer native build pass; generated native binary run pass
- Cross-package: check pass; interpreter run pass; native build pass; native binary output pass

## Known Blockers

None remaining for the `stark-hex` check/test/run/build path exercised here.

Retired blockers during this qualification:

- `Vec::as_slice` native lowering: implemented via existing `SliceNew(&Vec<T>, 0, len, false)`.
- `str.bytes()` native lowering: implemented as `StrBytes(&str) -> &[UInt8]` and generated Rust
  `.as_bytes()`.

## Deviations

The fixture files are not read at runtime; their cases are executed in `src/tests.stark`. This avoids
filesystem APIs, which are excluded from the package implementation.

The consumer `main` returns `Unit` rather than `Result<(), HexError>` because the current executable
entry contract only admits `Unit`, `Int32`, `Result<Unit, String>`, or `Result<Int32, String>`.
The consumer still exercises `Result<Vec<UInt8>, HexError>` across the package boundary via `decode`.

## Final Status

`IMPLEMENTATION COMPLETE — LOCAL NATIVE QUALIFICATION PASS`

Final exact-commit qualification should be re-run after the C6.5 compiler baseline freezes.
