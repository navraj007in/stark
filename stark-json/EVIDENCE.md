# stark-json v0.1 Evidence

## Audit

- Package path: `/Users/nexper/Documents/GitHub/stark/stark-json`
- Consumer path: `/Users/nexper/Documents/GitHub/stark/stark-json-consumer`
- Compiler/package commit observed for evidence: `3b20f9203e924e7c65c64ffac41cca82f5bb3855`
- Initial head observed before edits: `3a686c1c03ca1ee07f3b9790158bdb1780e67860`
- Package format: `starkpkg.json` with `name`, `version`, `entry`, `dependencies`
- CLI commands: `stark check`, `stark test`, `stark fmt --check`, `stark build`, `stark run`
- Toolchain: `rustc 1.93.0`, `cargo 1.93.0`
- Platform: macOS Darwin 25.5.0 arm64

## Preconditions

- Recursive enums: partial. Type declarations check, but recursive non-Copy borrow patterns block
  compliant borrowed encoding.
- `Vec<JsonValue>` and `Vec<JsonMember>`: partial. Declarations and basic construction check.
- String construction and append: partial. Basic operations work; scalar-from-codepoint is missing.
- `&str` string view: available.
- UTF-8 byte iteration: available through `.bytes()`.
- Byte-offset tracking: implemented.
- Safe indexing or iterator equivalent: byte/vector indexing available.
- `Result<T,E>` and `Option<T>`: available.
- Recursive pattern matching: partial for owned values; borrowed payload inspection is blocked.
- Checked arithmetic: implemented manually for relevant counters.
- Cross-module package compilation: package-local `mod tests` works.
- String equality: available.
- Recursive Drop: not fully proven.
- Package-local tests: available.
- Cross-package imports: available with dependency alias plus `package` manifest field.
- Deterministic execution: package tests deterministic on current interpreter.

Readiness classification: `INTERPRETER_READY` for the reduced package surface;
`PARTIAL - WAITING_COMPILER_RUNTIME` for the frozen work package.

## Commands

From `/Users/nexper/Documents/GitHub/stark/stark-json`:

- `../starkc/target/debug/stark check`: passed, `stark-json: OK`
- `../starkc/target/debug/stark test`: passed, 9 tests passed
- `../starkc/target/debug/stark fmt --check`: passed
- `../starkc/target/debug/stark build`: not used as native acceptance evidence because this is a
  library package without `main`

From `/Users/nexper/Documents/GitHub/stark/stark-json-consumer`:

- `../starkc/target/debug/stark check`: passed, `stark-json-consumer: OK`
- `../starkc/target/debug/stark run`: passed, output `null`
- `../starkc/target/debug/stark fmt --check`: passed
- `../starkc/target/debug/stark build`: failed, `native build does not yet support this program:
  unit expression form (C4.5)`

Fixtures:

- Valid fixture files: 17
- Invalid fixture files: 32

## Blockers

### JSON-BLOCKER-001

- PACKAGE COMMIT: working tree, evidence head `3b20f9203e924e7c65c64ffac41cca82f5bb3855`
- COMPILER COMMIT: `3b20f9203e924e7c65c64ffac41cca82f5bb3855`
- EXECUTION ENGINE: package checker/frontend
- MINIMISED STARK SOURCE:

```stark
pub enum JsonValue {
    Bool(Bool),
    String(String),
}
```

- COMMAND: `../starkc/target/debug/stark check`
- EXPECTED RESULT: enum variants named exactly `Bool` and `String` accepted
- ACTUAL RESULT: frontend rejects `Bool` as a variant name
- REQUIRED CAPABILITY: enum variants may use names that also identify built-in types
- NORMATIVE BASIS: frozen public API in work package Section 8
- WORKAROUND: used `BoolValue` and `StringValue`
- CLASSIFICATION: `FRONTEND_REJECTION`

### JSON-BLOCKER-002

- EXECUTION ENGINE: package checker/borrow checker
- MINIMISED STARK SOURCE:

```stark
enum E { Text(String) }
fn f(e: &E) -> String {
    match *e {
        E::Text(text) => String::from(text.as_str()),
    }
}
```

- EXPECTED RESULT: borrow enum payload by reference
- ACTUAL RESULT: binding `text` would move a non-Copy value out of a borrow
- REQUIRED CAPABILITY: ref-binding or equivalent borrowed payload access
- NORMATIVE BASIS: `encode(value: &JsonValue) -> String` must inspect without consuming input
- WORKAROUND: `encode` placeholder returns `null`
- CLASSIFICATION: `BORROW_OR_LIFETIME`

### JSON-BLOCKER-003

- EXECUTION ENGINE: Core runtime/String
- EXPECTED RESULT: construct Unicode scalar values from validated `\uXXXX` code points
- ACTUAL RESULT: no public scalar-from-codepoint constructor found in current package surface
- REQUIRED CAPABILITY: append decoded Unicode scalar to `String`
- NORMATIVE BASIS: string and Unicode semantics in work package Section 15
- WORKAROUND: direct UTF-8, simple escapes, and ASCII Unicode escapes decode; non-ASCII Unicode
  escapes return `InvalidUtf8`
- CLASSIFICATION: `RUNTIME_STRING`

## Final Status

STATUS: `PARTIAL - WAITING_COMPILER_RUNTIME`

FILES MODIFIED OUTSIDE PACKAGE: `stark-json-consumer/` only, as the cross-package consumer fixture.
