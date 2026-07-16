# Gate 6 — comparator defect matrix (measured)

Two comparators run against the same five defects that STARK catches before
inference. Every cell below was produced by running the tool, not asserted.

Raw data: `python-baseline.json` (G6-04), `rust-comparator.json` (G6-05).

| Defect class | STARK | Python / ORT | Strongest typed-Rust host |
| --- | --- | --- | --- |
| Incompatible tensor dimensions | compile (`E0212`) | runtime @ `session.run` (`InvalidArgument`) | **compile** (`E0308`: expected `224`, found `100`) |
| Incorrect element type | compile (`E0212`) | runtime @ `session.run` (`InvalidArgument`) | **compile** (`E0308`) |
| Incompatible device placement | compile (`E0212`) | **never** — silent CPU fallback | **compile*** (`E0308`: expected `Cpu`, found `Cuda<0>`) |
| Declaration/signature drift | build/deploy | **never** — no declaration exists | load — regenerate + digest gate |
| Runtime artifact swap | load (SHA gate) | **never** — silent wrong output (258→739) | load — `ArtifactMismatch`, exit 1 |
| **Defects caught before inference** | **5 / 5** | **2 / 5** (both only at runtime) | **5 / 5** |

`*` Compile-time device catching in Rust is **non-idiomatic**: it required a
phantom-device tensor API that re-implements STARK's device typing. An ordinary
`ort` host manages device via execution providers at session-build time and
would catch this at runtime, or silently — like Python.

## The decisive caveat (measured): the shape-arithmetic wall

`cases/limit_reshape.rs` is a *valid* program (flatten `[1,3,224,224] → [1, C*H*W]`)
that the strongest **stable**-Rust host cannot even express:

```
error: generic parameters may not be used in const operations
```

Computed output dims (reshape, flatten, matmul contraction, conv stride,
broadcast) need `feature(generic_const_exprs)` — unstable. This CV
*preprocessing* pipeline needs only permutation + cast + elementwise ops (no
shape arithmetic), which is exactly why the Rust host reaches parity here. Any
pipeline that reshapes/flattens before a head, does conv-stride math, broadcasts,
or carries more than one symbolic dim forces the stable-Rust host to drop those
dims to runtime. STARK types all of them.

## Reading

- STARK crushes the **operational** baseline (Python): 5/5 vs 2/5, and Python's
  two are caught only at `session.run`, after full deploy + load + preprocess.
- Against the **strongest** comparator, STARK reaches parity (5/5) on *this*
  pipeline — but only because a generated Rust host paid to re-implement STARK's
  shape/device typing in bespoke per-rank/per-permutation/phantom-device
  boilerplate, and only because this pipeline avoids shape arithmetic. STARK's
  genuine edge is **generality** (arbitrary shape arithmetic), **ergonomics**
  (general shape lists + full op set vs. generated boilerplate), **diagnostics**
  (axis/provenance vs. generic `E0308`), and a **single-language** pipeline.
