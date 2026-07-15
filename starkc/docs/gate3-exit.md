# Gate 3 exit — minimal execution path

Gate 3 closed on 2026-07-15. A STARK Core source file now travels through the
complete executable pipeline:

```text
Source → Tokens → AST → HIR → semantic checks → typed-HIR interpreter
```

## M3.1 — tree-walking interpreter

`src/interp.rs` executes the typed HIR produced by Gate 2. It supports:

- primitive values and checked numeric operations/casts;
- functions, arguments, returns, recursion-ready frames, and constants;
- blocks, `if`, `match`, `loop`, `while`, `for`, `break`, and `continue`;
- tuples, arrays, structs, enum variants, field access, patterns, and indexes;
- range iteration and bounds-checked array/Vec slicing;
- inherent methods and fully-qualified trait methods;
- `Option`, `Result`, and `?` early propagation;
- moves, partial runtime slots, assignments, and mutable method receivers;
- explicit and automatic destruction, including reverse declaration order;
- abort-on-panic/runtime-error behavior without destructor unwinding.

Runtime failures carry source spans and are rendered through the normal
diagnostic formatter. Integer overflow, division by zero, invalid casts,
invalid UTF-8 substring boundaries, failed unwraps, and out-of-bounds
operations terminate execution with a non-zero status.

## M3.2 — `core-min`

The Rust-backed runtime provides the Core surface needed by the Gate 3
programs and the later inference host:

- `print`, `println`, `panic`, `assert`, `sqrt`, and explicit `drop`;
- owned `String` construction and common query/mutation methods;
- `Vec<T>` construction, capacity, push/pop, indexing, safe `get`, insertion,
  removal, clearing, and slices;
- `Option<T>`/`Result<T,E>` construction, inspection, unwrap/unwrap-or, and
  `?` propagation;
- `Box<T>` construction and ownership recovery;
- integer ranges and iteration;
- `read_file` and `write_file`, returning `Result` on every I/O outcome.

The normative safety distinctions are tested: `[]` traps out of bounds,
`Vec::get` returns `None`, invalid slices trap, substring boundaries are
validated, and I/O failures are represented rather than ignored.

## M3.3 — executable workflow

```bash
cargo run -- run examples/gate3/01_hello.stark
```

`starkc run` parses and checks the complete program before execution. Compile
errors and runtime errors return failure; successful program output is written
to stdout. The Turbo-style terminal IDE uses the same in-process pipeline for
its Run command and displays output in the messages pane.

`examples/gate3/` contains seven executable programs covering functions,
control flow, aggregates, matching, `?`, core containers, mutable methods, and
drop order. `tests/gate3_execution.rs` executes all examples with exact-output
assertions and separately checks file I/O and CLI exit behavior.

## Reproduce the exit check

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release --all-targets
cargo run -- run examples/gate3/05_core_min.stark
```

## Next gate

Gate 4 owns tensor syntax and typing, ONNX import, shape/dtype/device checking,
and extension gating. Those concerns are intentionally absent from the Core
interpreter.
