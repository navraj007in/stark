# stark-digest Evidence

Baseline:

```text
BASELINE_SHA=fa4a4643f2f17983a11d8cd52a64892736b11036
PRECONDITION=READY
```

Preflight:

- `stark-hex` 0.1.0 is present, first-party, and already registered in package qualification.
- `stark-hex::decode` returns decoded bytes or `HexError::InvalidCharacter(index, byte)` /
  `HexError::InvalidLength`; `stark-digest` maps these without implementing a second decoder.
- Existing first-party packages and consumers use sibling path dependencies and package aliases
  with underscores.
- Existing package tests use `src/tests.stark` included by `mod tests;`.
- Pure-package qualification supports `stark check`, `stark test`, surface execution, formatter
  check, consumer interpreter run, consumer native build, and native execution.

The package has no capabilities, no provider crate, no host access, no build script, and no
compiler changes.
