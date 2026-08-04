# stark-json v0.1 Evidence

## Requalification, 2026-07-31 (CD-269) — after MIR amendment A12 and the CE1 binding rule

Re-run against the compiler at `1029350`+, because both changes touch this package: A12 changed how
a place's storage is ended, and CE1 is the borrowed-payload binding this parser is built on.

| command | package | result |
| --- | --- | --- |
| `stark check` | stark-json | PASS — `stark-json: OK` |
| `stark test` | stark-json | PASS — 10 passed, 0 failed |
| `stark fmt --check` | stark-json | PASS |
| `stark check` | stark-json-consumer | PASS |
| `stark run` | stark-json-consumer | PASS — `{"name":"stark","items":[1,true,null],"unicode":"😀"}` |
| `stark fmt --check` | stark-json-consumer | PASS |
| `stark build --no-build-cache` | stark-json-consumer | PASS |
| **executing the built binary** | stark-json-consumer | **PASS — first time actually observed** |

### What requalification found

**The previous native evidence did not hold, and the record could not have shown that.** `NATIVE-001`
was recorded PASS on `stark build --no-build-cache` succeeding. Building is not running: the binary
had never been executed. When it was, it aborted immediately:

```
generated-code invariant violated: write to a live slot
(MIR must Drop or move out before reassigning a live place)
(STARK compiler defect, not a program fault)
```

This was a surviving instance of `DEFECT-C788-LOOP-TEMP`, in the one shape A12's matrix did not
contain: **`?` inside a loop**. `lower_try` builds its own scrutinee temporary rather than going
through `lower_match`, so the storage end added for match arms never covered it — and unlike a
propagating path, the `Ok` path keeps executing, so the next iteration wrote over a partially moved
slot. This parser is `?` in loops throughout, which is why it failed on its first parse and why
sixteen deliberately chosen `match` shapes had not.

Fixed under CD-269; the regression is `starkc/tests/a12_storage_end_shapes.rs`, whose `?` cases were
added from this finding rather than from further matrix design.

The interpreter path (`stark run`) passed throughout, before and after. That is the point worth
carrying: **an interpreter pass is not native evidence**, and neither is a successful build. Only
executing the artefact is.

### Status change

`NATIVE-001` moves from PASS-by-build to PASS-by-execution. No other row changes. Tier-1 Linux x64
and Windows x64 remain unrun, as before.

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
