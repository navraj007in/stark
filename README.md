# STARK Language

> A safe, compiled, general-purpose language core — with AI/ML deployment as its long-term direction.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-specification-yellow.svg)](STARKLANG/docs/spec/)

## Project Status — Read This First

STARK is currently a **language specification, not a working compiler**. There
is no toolchain to install and nothing to run yet. What exists today:

- **Core v1 specification** (normative, complete draft): lexical grammar,
  syntax, type system, ownership/borrowing memory model, semantic analysis,
  standard library surface, and module system — in
  [`STARKLANG/docs/spec/`](STARKLANG/docs/spec/).
- **AI/ML extension sketches** (non-core, optional, early):
  [`STARKLANG/docs/extensions/`](STARKLANG/docs/extensions/).
- **Archived pre-pivot design docs** in
  [`STARKLANG/docs/archive/`](STARKLANG/docs/archive/README.md) and
  [`web-docs/`](web-docs/README.md) — historical only.

The project originally targeted a much broader "AI-native, cloud-first"
language. That scope was deliberately cut back to a small, implementable core;
the reasoning is documented in
[`STARK_Analysis_and_Discussion.md`](STARK_Analysis_and_Discussion.md).

## What Core v1 Looks Like

```stark
struct Point {
    x: Float64,
    y: Float64
}

fn distance(a: &Point, b: &Point) -> Float64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    sqrt(dx * dx + dy * dy)
}

fn main() {
    let p1 = Point { x: 0.0, y: 0.0 };
    let p2 = Point { x: 3.0, y: 4.0 };
    println(distance(&p1, &p2).fmt());
}
```

Core v1 design commitments:

- **Memory safety without GC** — ownership, moves, and borrow checking
  (conservative, annotation-free lifetime rules)
- **Static types with local inference** — explicit function signatures,
  inferred locals, generics with trait bounds
- **Explicit, predictable semantics** — no implicit numeric conversions,
  integer overflow always traps, bounds-checked indexing, exhaustive `match`
- **Result/Option error handling** with the `?` operator
- **A minimal standard library**: `Vec`, `HashMap`, `String`, IO, math

Tensors, model loading, and LLM constructs are **optional extensions** layered
on top of the core — see
[`AI-Extensions.md`](STARKLANG/docs/extensions/AI-Extensions.md).

## Documentation

Start at the [documentation index](STARKLANG/docs/index.md). The single-file
compiled spec (Markdown/HTML/PDF) is
[`STARK-Core-v1.md`](STARKLANG/docs/spec/STARK-Core-v1.md).

## Roadmap

- [x] Core v1 specification (grammar, types, ownership, stdlib, modules)
- [ ] Lexer + parser for Core v1
- [ ] Type checker with ownership/borrow analysis
- [ ] Interpreter or bytecode backend (MVP execution)
- [ ] Minimal standard library implementation
- [x] Tensor & model type system spec, v0.1 draft (the long-term differentiator) —
      [`Tensor-Model-Types.md`](STARKLANG/docs/extensions/Tensor-Model-Types.md)
- [ ] Go/no-go prototype: ONNX import → typed signatures → static shape/dtype/device
      checking → native binary via an existing backend (IREE / ONNX Runtime)

## Repository Layout

```
STARKLANG/docs/spec/        Normative Core v1 specification
STARKLANG/docs/extensions/  Optional AI/ML extension sketches
STARKLANG/docs/archive/     Superseded pre-pivot design docs (historical)
STARKLANG/compiler/         Pre-pivot Python prototype (not Core v1; see its README)
web-docs/                   Archived pre-pivot HTML docs (historical)
Practice/                   Early experiments
```

## Contributing

The most valuable contribution right now is a Core v1 **lexer and parser**
implementing `docs/spec/01` and `02`, with a test suite derived from the spec
examples. Spec bug reports (ambiguities, contradictions, unparseable examples)
are equally welcome.

## License

MIT.

## Acknowledgments

STARK's design draws on Rust (ownership model), Swift for TensorFlow and Mojo
(ML-language direction), and Julia (numerical computing).
