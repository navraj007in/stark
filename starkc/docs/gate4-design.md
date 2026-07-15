# STARK Gate 4 Extension Design Note

This document specifies the architectural boundary for integrating the optional `tensor` v0.1 extension into the STARK compiler, satisfying milestone **M4.0**.

---

## 1. Extension Isolation and Configuration

To prevent tensor-specific logic from polluting the STARK Core-only compiler pipeline, we introduce a capability boundary. 

### Language Options and Configuration
We define a deterministic configuration passed through the stages of compilation:

```rust
// starkc/src/options.rs (implemented in M4.1)
pub enum ExtensionId { Tensor }

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct ExtensionSet { /* private flags */ }
impl ExtensionSet {
    pub fn contains(self, id: ExtensionId) -> bool;
    pub fn enable(&mut self, id: ExtensionId) -> bool; // false if already set
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct LanguageOptions { pub extensions: ExtensionSet }
```

The flags are private behind `contains`/`enable` so adding a second extension
never becomes a struct-field breaking change. `options_from_extension_flags`
builds a `LanguageOptions` from CLI `--extension` values, rejecting unknown and
duplicate ids as usage errors.

- **Core-only Mode (Default)**: `tensor` is disabled. Any usage of tensor surface syntax (such as dynamic dimensions `[B, 3, 224, 224]`, model declarations, or new primitives `Float16`/`BFloat16`) is rejected during parsing or checking with a diagnostic naming the `tensor` extension.
- **Tensor Mode**: `tensor` is enabled via `starkc check --extension tensor` and `starkc parse --extension tensor`.

### API Compatibility
To minimize churn on existing tests, the existing entry points default to Core-only mode:
- `parse(file, mode)` calls `parse_with_options(file, mode, LanguageOptions::default())`
- `resolve(tree, file)` calls `resolve_with_options(tree, file, LanguageOptions::default())`
- `typecheck::check(hir, file)` calls `typecheck::analyze_with_options(hir, file, LanguageOptions::default()).diagnostics`

---

## 2. AST and HIR Deltas

Optional syntax is parsed and represented using optional fields and gated variants in the AST/HIR.

- **Types**: New primitives `Float16` and `BFloat16` are recognized as
  `Primitive` type variants (only when `tensor` is enabled; they lex as
  ordinary identifiers). Generic arguments gain a single `GenericArg::Shape`
  variant carrying a `ShapeArg` (`[DimExpr, ...]`, including `[]`). Shape
  arguments (D2) and const index lists (D5) share this surface form and are
  disambiguated semantically by the operation signature (spec §6.4), so they
  are one AST node, not two. Dimension expressions are a small arena
  (`DimId` → `DimExprNode`) of literals, variables, and `+`/`-`/`*` binaries.
- **Items**: Nominal model declarations are added:
  ```rust
  pub struct ModelDef {
      pub name: Span,
      pub generics: Vec<GenericParam>,
      pub ports: Vec<ModelPort>,
  }
  ```
- **Disambiguation**: In generic-argument position a `[...]` group is a Core
  array type when it contains a top-level `;` (`[T; N]`); otherwise, under the
  extension, it is a shape argument (`[]`, `[B]`, `[a, b, ...]`). In Core-only
  mode `generic_args` is unchanged except that `[]`/`[a, b]` — previously parse
  errors — now yield the focused diagnostic below, so the 121-fixture
  conformance suite is untouched.
- **Diagnostics**: Encountering tensor surface syntax (D2 shape args, D4
  `model`) with `tensor` disabled produces a focused diagnostic naming the
  extension (e.g. `` shape arguments require extension `tensor` ``). Reserved
  §11 named-axis syntax is rejected even with the extension on. `model`,
  `input`, and `output` are contextual keywords lexed as identifiers, so the
  lexer stays extension-agnostic.

---

## 3. Extension Type Representation

To avoid scattering `if tensor` logic throughout the core `Ty` representation, we isolate tensor types behind a single extension variant:

```rust
pub enum Ty {
    // ... Core variants ...
    Extension(Box<ExtensionTy>),
}
```

Where `ExtensionTy` is defined in `starkc/src/extensions/tensor/types.rs`:
```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtensionTy {
    Tensor {
        elem: Box<Ty>,
        shape: TensorShape,
        device: Device,
    },
    TensorDyn {
        elem: Box<Ty>,
    },
    TensorAny,
}
```

All Core unification, displaying, and substitution logic delegates to the extension module if `Ty::Extension` is encountered:
- **Unification**: `Tensor` unifies iff element type unifies, shape unifies positionally by rank-equality and dim-equality, and devices unify.
- **Copy/Drop**: Tensors are strictly `Move` (non-`Copy`) types.

---

## 4. Dimension Algebra & Provenance

Dimension expressions (`DimExpr`) are represented as canonical polynomials. 
For example, a symbolic dimension is normalized to a sum of products:
```text
c0 + c1 * x1 + c2 * x2 + ...
```
Normalization easily proves mathematical identities (e.g. `B * (H + 1) == B*H + B`) using normal-form equality.

To ensure high-quality error reporting, every dimension tracks its **provenance**:
```rust
pub struct Provenance {
    pub span: Span,
    pub origin: OriginCategory,
    pub label: String,
}
```
Origins include literals, generic parameters, refinement points, imported model ports, and operation results.

---

## 5. Dependency Evaluation (ONNX Decoder)

To read and verify ONNX model metadata without bringing in a heavy runtime/engine, we evaluate several options:

1. **`ort`** (ONNX Runtime):
   - *Status*: Maintained bindings to ONNX Runtime.
   - *Transitive Cost*: High. Links to C++ ONNX Runtime binary, which is heavy and unnecessary for compile-time signature checks.
   - *Verdict*: **Rejected** per roadmap guidance (do not include runtime in Gate 4).
2. **`tract-onnx`**:
   - *Status*: Maintained inference engine in pure Rust.
   - *Transitive Cost*: High. Pulls in a massive graph optimization and execution pipeline.
   - *Verdict*: **Rejected** to keep compiler binary light.
3. **`onnx-pb`** (version 0.1.4):
   - *Status*: Unmaintained (last released 2020). Generates ONNX bindings at
     build time via an old `prost-build 0.6.1`.
   - *Transitive Cost*: **Higher than it appears.** `prost-build 0.6.1`
     requires a system `protoc` binary on `PATH` at build time; there is no
     bundled `protoc` for this platform. On a machine without `protoc` the
     whole crate — including Core-only builds and tests — fails to compile.
     This was observed directly: adding `onnx-pb = "0.1.4"` turned the green
     baseline red with *"Failed to find the protoc binary"*.
   - *Verdict*: **Rejected.** A build-time system-binary dependency violates
     reproducibility and blocks Core work that has nothing to do with tensors.

**Decision (confirmed in M4.5):** decode the handful of ONNX metadata fields we actually read
(`GraphProto.input`/`output` → `ValueInfoProto.name`, `TypeProto.Tensor.elem_type`,
and `TensorShapeProto.dim`) with **no build-time `protoc`**. M4.5 selected:

1. **A minimal hand-written protobuf reader** (varint, fixed-width, and
   length-delimited
   wire format) over only the ~5 message fields Gate 4 needs. Zero external
   decoder dependencies, no `protoc`, no `unsafe`, fully under our control,
   and directly mutation-tested at the byte boundary.
2. **`sha2 0.11`** with default features disabled for the required artifact
   provenance hash. It is runtime-independent and does not generate code or
   invoke a system binary during the build.

The rejected fallback remains `prost` with a checked-in, pre-generated trimmed
schema: it adds machinery that is unnecessary for the bounded signature-only
surface. Per PLAN.md T3, the hashing dependency was added only when M4.5 first
consumed it; `Cargo.toml` remained dependency-free through M4.4.

---

## 6. Fixtures and Validation Strategy

- **Automated Fixtures**: Tiny valid ONNX models are generated programmatically by `tests/gate4_onnx.rs`; no opaque binary artifact is committed.
- **CV Representative Signature**: The chosen target is a ResNet-style CV model:
  - Input: `image` of type `Tensor<Float32, [B, 3, 224, 224]>`
  - Output: `class` of type `Tensor<Float32, [B, 1000]>`
- **Manifest**: A Gate 4 test manifest will explicitly cover parser-pass, Core-only rejection, and semantic-error check fixtures.
