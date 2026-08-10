# stark-sha256 evidence

Baseline SHA: `3557783a1eb0f41c05efdcf4e9918170140a50e0`

Final SHA: not committed in this working tree

Package tests:

- `../../starkc/target/release/stark check` from `packages/stark-sha256`: PASS
- `../../starkc/target/release/stark test` from `packages/stark-sha256`: PASS

Test count: 22 passed, 0 failed, 0 ignored

Consumer:

- `../../starkc/target/release/stark run` from `packages/stark-sha256-consumer`: PASS
- stdout: `sha256:ok`

Native consumer:

- `../../starkc/target/release/stark build --no-build-cache` from `packages/stark-sha256-consumer`: PASS
- `./target/stark/debug/stark-sha256-consumer`: PASS
- stdout: `sha256:ok`
- `../../starkc/target/release/stark build --release --no-build-cache` from `packages/stark-sha256-consumer`: PASS
- `./target/stark/release/stark-sha256-consumer`: PASS
- stdout: `sha256:ok`

Documentation/surface:

- `../../starkc/target/release/stark doc --output /tmp/stark-sha256-docs` from `packages/stark-sha256`: PASS
- Generated docs for 2 public item(s): `hash`, `hash_hex`

Qualification:

- Full `python3 starkc/scripts/qualify-first-party-packages.py --stark starkc/target/release/stark --repo-root /Users/nexper/Documents/GitHub/stark` reached and passed `stark-digest` and `stark-sha256`.
- `stark-sha256` qualification observations: check PASS, tests PASS, docs PASS, surface check PASS with 2 public callables all called, fmt check PASS, consumer check PASS, consumer run PASS with `sha256:ok`, native debug consumer build/run PASS with `sha256:ok`.
- The full qualification run later stopped at pre-existing `stark-net` provider resolution: `UnknownFunction { capability: "network-client", function: "stark_tcp_stream_connect_timeout", provider: "stark-std-net" }`.

Native-code prohibition:

- `rg` over `packages/stark-sha256`, `packages/stark-sha256-consumer`, `packages/stark-digest`, and `packages/stark-digest-consumer` for `.rs`, `.c`, `.cc`, `.cpp`, `.h` crypto implementations/references found no matches.

Notes:

- `stark-digest` was imported from `origin/codex/stark-digest` as the required prerequisite.
- The SHA-256 implementation is pure STARK and has no host capability declaration.
- No native provider or crypto library was added.
- Direct test-module access to private package helpers is not supported by the current compiler.
  Arithmetic, rotation, endian and padding helper checks are kept as private self-checks exercised
  through `hash`, while package tests assert the public digest observations.
