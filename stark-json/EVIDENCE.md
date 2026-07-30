# stark-json v0.1 Evidence

## Audit

- Package path: `/Users/nexper/Documents/GitHub/stark/stark-json`
- Consumer path: `/Users/nexper/Documents/GitHub/stark/stark-json-consumer`
- Compiler/package commit observed for evidence: working tree after
  `88f834b8b6c0bf490c7d51e7b0e7a9b0f976b772`
- Package format: `starkpkg.json` with `name`, `version`, `entry`, `dependencies`
- CLI commands: `stark check`, `stark test`, `stark fmt --check`, `stark build`, `stark run`
- Toolchain: `rustc 1.93.0`, `cargo 1.93.0`
- Platform: macOS Darwin 25.5.0 arm64

## Preconditions

- Recursive enums: available for this package, including borrowed non-Copy payload binding.
- `Vec<JsonValue>` and `Vec<JsonMember>`: available for construction, `push`, length, and borrowed `get`.
- String construction and append: available, including scalar construction through `Char::from_u32`.
- `&str` string view: available.
- UTF-8 byte iteration: available through `.bytes()`.
- Byte-offset tracking: implemented.
- Safe indexing or iterator equivalent: byte/vector indexing available.
- `Result<T,E>` and `Option<T>`: available.
- Recursive pattern matching: available for owned values and borrowed enum payload inspection.
- Checked arithmetic: implemented manually for relevant counters.
- Cross-module package compilation: package-local `mod tests` works.
- String equality: available.
- Recursive Drop: available for this package's generated Rust native build.
- Package-local tests: available.
- Cross-package imports: available with dependency alias plus `package` manifest field.
- Deterministic execution: package tests deterministic on current interpreter.

Readiness classification: `LOCAL_NATIVE_READY` on macOS arm64 for the frozen package surface.
Tier-1 Linux x64 and Windows x64 qualification remains unrun.

## Commands

From `/Users/nexper/Documents/GitHub/stark/stark-json`:

- `../starkc/target/debug/stark check`: passed, `stark-json: OK`
- `../starkc/target/debug/stark test`: passed, 10 tests passed
- `../starkc/target/debug/stark fmt --check`: passed
- `../starkc/target/debug/stark build`: not used as native acceptance evidence because this is a
  library package without `main`

From `/Users/nexper/Documents/GitHub/stark/stark-json-consumer`:

- `../starkc/target/debug/stark check`: passed, `stark-json-consumer: OK`
- `../starkc/target/debug/stark build --no-build-cache`: passed
- `../starkc/target/debug/stark run`: passed, output
  `{"name":"stark","items":[1,true,null],"unicode":"😀"}`
- `../starkc/target/debug/stark fmt --check`: passed

From `/Users/nexper/Documents/GitHub/stark/starkc`:

- `cargo fmt --all --check`: passed
- `cargo check --workspace --all-features`: passed
- `cargo check --workspace --all-targets --all-features`: inconclusive; command did not complete
  within the local investigation window after earlier diagnostics were fixed
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: inconclusive; command did
  not complete within the local investigation window
- `cargo test parser::tests::item_kinds --lib`: inconclusive; test binary started but did not
  complete within the local investigation window

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
- ACTUAL RESULT: accepted after compiler parser fix
- REQUIRED CAPABILITY: enum variants may use names that also identify built-in types
- NORMATIVE BASIS: frozen public API in work package Section 8
- WORKAROUND: removed; package uses `Bool` and `String`
- CLASSIFICATION: `RESOLVED`
- EVIDENCE: `stark-json` check passes with `JsonValue::Bool(Bool)` and
  `JsonValue::String(String)`. Compiler parser regression added for declaration and use-site
  paths using primitive-named enum variants.

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
- ACTUAL RESULT: accepted after type checker and MIR lowering add borrowed non-Copy payload binding
- REQUIRED CAPABILITY: ref-binding or equivalent borrowed payload access
- NORMATIVE BASIS: `encode(value: &JsonValue) -> String` must inspect without consuming input
- WORKAROUND: removed; `encode` now performs borrowed recursive traversal
- CLASSIFICATION: `RESOLVED`
- EVIDENCE: minimized source checks, builds, and runs natively, printing `ok` twice after repeated
  borrowed inspection. `stark-json-consumer` encodes after parsing, inspects nested payloads, and
  encodes the same value again.

### JSON-BLOCKER-003

- EXECUTION ENGINE: Core runtime/String
- EXPECTED RESULT: construct Unicode scalar values from validated `\uXXXX` code points
- ACTUAL RESULT: `Char::from_u32` returns `Option<Char>` and rejects surrogate/out-of-range values
- REQUIRED CAPABILITY: append decoded Unicode scalar to `String`
- NORMATIVE BASIS: string and Unicode semantics in work package Section 15
- WORKAROUND: removed; BMP escapes and valid surrogate pairs decode to UTF-8
- CLASSIFICATION: `RESOLVED`
- EVIDENCE: package tests cover `\u00E9`, `\u20AC`, `\uD83D\uDE00`, and lone surrogate errors.
  Isolated native checks verified `Char::from_u32(0x41u32)` returns `Some('A')`,
  `Char::from_u32(0xD800u32)` returns `None`, and `Char::from_u32(0x110000u32)` returns `None`.

## Final Status

STATUS: `PARTIAL - PLATFORM_QUALIFICATION_PENDING`

FILES MODIFIED OUTSIDE PACKAGE: `starkc` compiler/runtime changes for JSON-BLOCKER-001,
JSON-BLOCKER-002, JSON-BLOCKER-003, and native generated Rust support.

The frozen package behavior is implemented and locally qualified on macOS arm64. Linux x64 and
Windows x64 Tier-1 qualification remains to be run before claiming full v0.1 completion.
