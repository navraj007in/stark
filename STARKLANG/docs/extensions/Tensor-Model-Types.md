# STARK Tensor & Model Type System Extension

**Extension id:** `tensor` · **Version:** 0.1 · **Status:** normative draft
**Requires:** Core v1 (`core-min` standard library profile or higher)

## 1. Overview

This extension defines STARK's reason to exist: a type system in which tensor
shapes, element types, devices, and model input/output signatures are checked
at compile time across an entire inference program. It is designed for the
deployment boundary — the pipeline from decoded request data through
preprocessing, model execution, and postprocessing — where today's stacks
discover shape and dtype defects only at runtime.

Design goals:
1. **Static tensor contracts.** Shape, dtype, and device errors in code that
   uses this extension are compile-time errors, except at explicitly marked
   refinement boundaries.
2. **Symbolic dimensions.** Batch and sequence dimensions are checked
   symbolically without being fixed to constants.
3. **Model signatures as types.** An imported model's input/output contract
   is a nominal type; calling it with a mismatched tensor is a compile error.
4. **No new runtime.** This extension defines *frontend* semantics only.
   Implementations are expected to lower to existing backends (StableHLO/IREE,
   ONNX Runtime, generated native code); see §10.

Non-goals for v0.1: training and autodifferentiation, a kernel language,
quantization arithmetic (reserved, §11), distributed execution, named/labeled
axes (reserved, §11).

## 2. Deltas Against Core v1

This extension modifies the Core v1 grammar and type system as follows. Each
delta is specified in the sections noted.

| # | Delta | Against | Section |
| --- | --- | --- | --- |
| D1 | `Dim` and `DType` kinds for generic parameters | 03-Type-System.md Generics | §3 |
| D2 | Shape arguments `[DimExpr, ...]` as generic arguments | 02-Syntax-Grammar.md `GenericArg` | §3.2 |
| D3 | New primitive element types `Float16`, `BFloat16` | 03-Type-System.md primitives | §4.1 |
| D4 | `model` item declaration | 02-Syntax-Grammar.md `Item` | §7 |
| D5 | Const index-list generic arguments (for `permute`) | 02-Syntax-Grammar.md `GenericArg` | §6.4 |

A Core v1 implementation without this extension MUST reject programs using
D1–D5 with a diagnostic naming the extension.

## 3. Dimensions

### 3.1 Dim parameters
`Dim` is a **kind**, not a trait: a generic parameter declared `: Dim` ranges
over dimension values (non-negative integers), not types. It reuses the Core
v1 generic parameter syntax:

```stark
fn matmul<T: DType, M: Dim, K: Dim, N: Dim>(
    a: &Tensor<T, [M, K]>,
    b: &Tensor<T, [K, N]>
) -> Tensor<T, [M, N]>
```

Rules:
- A parameter bounded by `Dim` MUST NOT carry trait bounds, be used in type
  position, or be mixed with type parameters in a single bound.
- Dim parameters are inferred at call sites exactly like type parameters;
  the turbofish form supplies them explicitly when inference is impossible.
- Scope rules follow Core v1 generics: every dim variable used by an item
  must be declared by that item (or by an enclosing `model` declaration, §7).

### 3.2 Shape arguments and dimension expressions
The `GenericArg` production is extended:

```ebnf
GenericArg ::= Type
             | IDENTIFIER '=' Type
             | ShapeArg                      // (extension)
             | IndexListArg                  // (extension, §6.4)

ShapeArg  ::= '[' DimExpr (',' DimExpr)* ','? ']'
            | '[' ']'                        // rank-0 (scalar tensor)

DimExpr ::= INTEGER
          | IDENTIFIER                       // dim variable
          | DimExpr ('+' | '-' | '*') DimExpr
          | '(' DimExpr ')'
```

### 3.3 Dimension expression semantics
- Dim expressions denote non-negative integers. They form polynomials over
  dim variables with integer coefficients.
- **Equality** of dim expressions is decided by polynomial normalization:
  two expressions are equal iff their normal forms are identical
  (e.g. `B * (H + 1)` equals `B*H + B`). This is the *only* arithmetic fact
  the checker knows; it does not do range reasoning in v0.1.
- Subtraction that could produce a negative value for some assignment of the
  variables is a compile-time error unless the checker can prove
  non-negativity from literal constants alone.
- A shape whose dims are all literals is **fully static**; otherwise it is
  **symbolic**. There are no "unknown rank" tensor types: rank is always
  static.

## 4. Tensor Types

### 4.1 Element types (`DType`)
`DType` is a kind-like bound satisfied by exactly: `Int8`–`Int64`,
`UInt8`–`UInt64`, `Float32`, `Float64`, `Bool`, and the two new primitives
this extension adds:

```stark
Float16    // IEEE 754 binary16
BFloat16   // bfloat16 (8-bit exponent, 7-bit mantissa)
```

`Float16`/`BFloat16` are `Copy`, follow IEEE-754 semantics where defined, and
are valid only as tensor element types and in explicit `as` casts in v0.1
(scalar arithmetic on them is implementation-optional).

### 4.2 The Tensor type
```stark
Tensor<T, S>                      // element type T: DType, shape S
Tensor<T, S, device = D>          // with explicit device (§8)
```

Examples:
```stark
Tensor<Float32, [1024, 768]>              // fully static
Tensor<Float32, [B, 3, 224, 224]>         // symbolic batch
Tensor<Float32, []>                       // rank-0 scalar tensor
Tensor<Float16, [B, 1000], device = Cuda<0>>
```

Properties:
- `Tensor` is an implementation-provided, owned, `Move` (never `Copy`) type.
- Two tensor types are equal iff their element types are equal, their shapes
  are equal dim-by-dim (§3.3), and their devices unify (§8).
- All operations in §6 take `&self`/`&other` and return newly owned tensors.
  Implementations MAY share storage internally (views), but the observable
  semantics MUST be as if the result were freshly allocated. This keeps
  tensors out of the borrow-carrying machinery of Core v1.

### 4.3 Dynamic tensors and refinement
Data entering from outside the type system (files, network requests, model
artifacts) has unverified shape. Two type-erased forms exist:

```stark
TensorDyn<T>    // dtype static, shape dynamic (rank and dims runtime values)
TensorAny       // dtype and shape dynamic
```

The **only** bridge from dynamic to static is `refine`:

```stark
impl<T: DType> TensorDyn<T> {
    fn refine<S>(self) -> Result<Tensor<T, S>, ShapeError>;
}

impl TensorAny {
    fn refine<T: DType, S>(self) -> Result<Tensor<T, S>, ShapeError>;
}
```

Refinement semantics (checked at runtime, once, at the boundary):
- Each **literal** dim in `S` must equal the actual dim, else `Err(ShapeError)`.
- Each dim **variable** in `S` that is already bound in the enclosing scope
  (e.g. a `Dim` parameter of the enclosing function) must equal its bound
  value, else `Err`.
- Each dim variable in `S` that is **unbound** is bound *existentially* to the
  actual runtime value, and acts as a fresh symbolic dim for the rest of the
  enclosing block (precedent: Futhark's existential sizes). Two tensors
  refined with the *same* fresh variable in one `refine` call are thereby
  known to agree in that dim; two separate `refine` calls introduce distinct
  variables even if spelled alike — reuse of a bound variable is what asserts
  equality.
- After a successful `refine`, no shape or dtype error involving that tensor
  can occur at runtime.

```stark
fn handle(request: TensorAny) -> Result<Tensor<Float32, [1000]>, ServeError> {
    // B is unbound here: bound existentially to the request's first dim
    let images = request.refine::<UInt8, [B, 224, 224, 3]>()?;
    let logits = classify(&model, &images);   // statically checked from here on
    Ok(logits.mean_axis::<0>())
}
```

## 5. Static Semantics

- **Unification.** Dim variables unify with dim expressions the same way type
  variables unify with types, using §3.3 equality. Shape unification is
  positional and requires equal rank.
- **No implicit rank/shape coercions.** There is no implicit broadcasting
  except the statically-provable elementwise rule in §6.2, and no implicit
  dtype conversion (Core v1's no-implicit-numeric-coercion rule extends to
  tensors).
- **Failure is compile-time.** Any operation in §6 whose constraint cannot be
  *proven* from the types is a compile-time error — including
  symbolic-vs-symbolic cases the checker cannot decide. The diagnostic MUST
  state the unsatisfied constraint and SHOULD suggest `refine` or
  `broadcast_to` where applicable. Provable-but-false and unprovable are both
  rejections; they differ only in wording.

## 6. Core Tensor Operations

All operations live in the `tensor` module of the extension's library
(`use tensor::*;`). Signatures below are normative (API notation, per the
Core v1 stdlib convention). `T: DType` throughout; device rules in §8.

### 6.1 Construction
```stark
fn zeros<T: DType, S>() -> Tensor<T, S>       // S via inference or turbofish
fn ones<T: DType, S>() -> Tensor<T, S>
fn full<T: DType, S>(value: T) -> Tensor<T, S>
fn from_vec<T: DType, N: Dim>(data: Vec<T>) -> Result<Tensor<T, [N]>, ShapeError>
```

### 6.2 Elementwise and broadcasting
Binary elementwise operations (`add`, `sub`, `mul`, `div`, `min`, `max`,
comparison ops producing `Tensor<Bool, S>`):

```stark
fn add<T: DType, S>(a: &Tensor<T, S>, b: &Tensor<T, S>) -> Tensor<T, S>
```

Broadcasting rule: a binary elementwise op also accepts operands whose shapes
differ, iff for every trailing position the dims are **provably equal** or
one of them is the **literal** `1` (NumPy alignment, checked statically).
The result dim is the non-1 dim. A dim-variable-vs-different-dim-variable
position is a compile error — broadcast explicitly:

```stark
fn broadcast_to<T: DType, S1, S2>(a: &Tensor<T, S1>) -> Tensor<T, S2>
// Constraint: S1 broadcasts to S2 under the rule above; S2 is the result.
```

### 6.3 Matrix multiplication
```stark
fn matmul<T: DType, M: Dim, K: Dim, N: Dim>(
    a: &Tensor<T, [M, K]>, b: &Tensor<T, [K, N]>
) -> Tensor<T, [M, N]>

fn batch_matmul<T: DType, B: Dim, M: Dim, K: Dim, N: Dim>(
    a: &Tensor<T, [B, M, K]>, b: &Tensor<T, [B, K, N]>
) -> Tensor<T, [B, M, N]>
```
The shared `K` makes inner-dimension mismatch a type error.

### 6.4 Shape manipulation
`permute` takes a **const index list** generic argument (delta D5):

```ebnf
IndexListArg ::= '[' INTEGER (',' INTEGER)* ','? ']'   // in GenericArg position
```

(Note: an `IndexListArg` whose elements are all integer literals in a context
expecting axes is distinguished from a `ShapeArg` by the operation's
signature; the two share surface syntax.)

```stark
fn permute<const P>(self: &Tensor<T, S>) -> Tensor<T, permute(S, P)>
// P must be a permutation of 0..rank(S); the result shape is S reordered by
// P. Both are checked at compile time.

fn reshape<S2>(self: &Tensor<T, S1>) -> Tensor<T, S2>
// Constraint: product(S1) == product(S2), decided by §3.3 polynomial
// equality. If not provable, compile error (route through TensorDyn/refine
// for genuinely dynamic reshapes).

fn concat<const AXIS>(a: &Tensor<T, S1>, b: &Tensor<T, S2>) -> Tensor<T, S3>
// Constraint: S1 and S2 agree in every dim except AXIS;
// S3[AXIS] = S1[AXIS] + S2[AXIS].

fn slice_axis<const AXIS, START: Dim, LEN: Dim>(self: &Tensor<T, S>)
    -> Tensor<T, replace(S, AXIS, LEN)>
// Constraint: START + LEN == S[AXIS] must be provable, or START and LEN are
// literals with START + LEN <= a literal S[AXIS].
```

Convenience: `transpose` = `permute<[1, 0]>` on rank-2 tensors.

### 6.5 Reduction
```stark
fn sum_axis<const AXIS>(self: &Tensor<T, S>) -> Tensor<T, remove(S, AXIS)>
fn mean_axis<const AXIS>(self: &Tensor<T, S>) -> Tensor<T, remove(S, AXIS)>
fn argmax<const AXIS>(self: &Tensor<T, S>) -> Tensor<Int64, remove(S, AXIS)>
fn sum(self: &Tensor<T, S>) -> Tensor<T, []>          // full reduction
```
`AXIS` must be a literal `< rank(S)`; `remove(S, AXIS)` deletes that
position. `softmax<const AXIS>` preserves the shape.

### 6.6 Dtype conversion
```stark
fn cast<U: DType>(self: &Tensor<T, S>) -> Tensor<U, S>
```
Cast value semantics follow the scalar `as` rules of Core v1
(03-Type-System.md), applied elementwise, with one difference: an
out-of-range elementwise cast **saturates** rather than trapping
(deployment code must not abort on a single bad pixel). Float-to-int of
NaN saturates to 0.

## 7. Model Declarations and Import

### 7.1 The `model` item (delta D4)
```ebnf
Model ::= 'model' IDENTIFIER GenericParams? '{' ModelPort* '}'

ModelPort ::= ('input' | 'output') IDENTIFIER ':' Type ';'
```

Rules:
- Every `ModelPort` type MUST be a `Tensor`/`TensorDyn` type.
- Generic parameters of a `model` MUST all be `Dim` parameters; they are
  universally quantified **per predict call** (each call may use a different
  batch size).
- At least one input and one output are required. Ports are ordered as
  written.

```stark
model ResNet50<B: Dim> {
    input  image: Tensor<Float32, [B, 3, 224, 224]>;
    output class: Tensor<Float32, [B, 1000]>;
}
```

### 7.2 Generated interface
A `model` declaration introduces a nominal type with:

```stark
impl ResNet50 {
    // Load and VERIFY a runtime artifact against this declaration (§7.3)
    fn load(path: &str) -> Result<ResNet50, ModelError>;

    // One parameter per input port, one result per output port
    // (multiple outputs are returned as a tuple in declaration order)
    fn predict<B: Dim>(&self, image: &Tensor<Float32, [B, 3, 224, 224]>)
        -> Tensor<Float32, [B, 1000]>;
}
```

`predict` cannot fail with a shape, dtype, or signature error: those are
excluded by the types (execution-level failures such as out-of-memory are
runtime errors per Core v1 semantics).

### 7.3 Artifact verification (`load`)
`load` MUST verify the runtime artifact (e.g. an ONNX file) against the
declaration and return `Err(ModelError::SignatureMismatch(...))` on any
deviation, before any inference runs (fail-fast):
- port count, order/names, and element types must match exactly;
- a **literal** dim in the declaration must equal the artifact's static dim;
- a **dim variable** in the declaration matches only a *dynamic* dim in the
  artifact (an artifact-static dim under a variable is a mismatch — the
  declaration over-promises flexibility the artifact does not have).

### 7.4 Import tooling
Implementations SHOULD provide an importer that generates the declaration
from an artifact, so signatures are derived, not hand-maintained:

```
stark import model.onnx --out resnet50.stark
```

The importer maps ONNX dynamic dims to fresh `Dim` parameters and static dims
to literals. The generated file is ordinary STARK source and is the canonical
form reviewed in version control.

## 8. Devices

```stark
// Device types (kind: Device)
Cpu
Cuda<const N>       // CUDA device index N
```

- The optional `device = D` argument fixes a tensor's placement. When
  omitted, the tensor type carries an implicit device *variable* that unifies
  like a dim variable (device-polymorphic code by default).
- Every multi-operand operation in §6 requires its operands' devices to
  unify; mixing `Cpu` and `Cuda<0>` operands is a compile-time error.
- Transfers are explicit:

```stark
fn to_device<D>(self: &Tensor<T, S>) -> Tensor<T, S, device = D>
```

- A `model`'s ports adopt the device the implementation loads it on;
  `load` variants MAY take a device argument (implementation-defined
  surface, but placement mismatches MUST remain compile-time errors).

Memory layout types (`NCHW`, row/column-major) are **reserved** for a future
version (§11); v0.1 layout is implementation-defined behind the types.

## 8a. Value Range (semantic property, Gate 7)

The optional `range = R` argument statically tracks the *image value range* a
tensor conceptually holds. It is a **compile-time-only** semantic property: the
marker is checked by the front end and does **not** survive into generated code
(§10). It is orthogonal to shape, dtype, and device.

```stark
// Value-range states (kind: Range)
Unspecified    // default — no claim
ByteRange      // integer image values conceptually in [0, 255]
UnitRange      // floating-point values conceptually in [0, 1]
Normalized     // channel-wise mean/std normalised values

Tensor<UInt8,   [B, H, W, 3], range = ByteRange>
Tensor<Float32, [B, 3, H, W], range = UnitRange>
Tensor<Float32, [B, 3, H, W], range = Normalized>
```

- An omitted `range` is `Unspecified`. `range = R` may appear in a tensor type
  in either order relative to `device = D`, and on a `refine` boundary
  (`raw.refine::<UInt8, [..], range = ByteRange>()`), which is how a pipeline
  assigns the initial range to decoded input.
- **Exact match, no widening.** Two tensor types unify only if their ranges are
  equal. `Unspecified` is *not* a supertype: a ranged value cannot be silently
  laundered into `Unspecified` (that erase is a compile-time error). This makes
  a `model` port's declared input range a checked contract.
- **Transitions are explicit, named operations** — ordinary arithmetic never
  claims a transition:

  ```stark
  fn cast<U>(self: Tensor<T, S, range = R>) -> Tensor<U, S, range = R>  // preserves
  fn scale_255(self: Tensor<Float32, S, range = ByteRange>)
      -> Tensor<Float32, S, range = UnitRange>
  fn normalize(self: Tensor<Float32, S, range = UnitRange>)
      -> Tensor<Float32, S, range = Normalized>
  ```

- **Range propagation is per operation:**
  - *preserve* — shape/layout/dtype-only ops (`permute`, `reshape`,
    `broadcast_to`, `slice`, `transpose`, `cast`, `to_device`) carry the range
    through unchanged;
  - *combine* — `concat` and elementwise ops combine operand ranges
    (`Unspecified` neutral; two different specified ranges are an error);
  - *transition* — only the named ops (`scale_255`, `normalize`) change the
    range;
  - *clear* — operations that change the meaning of the values
    (`matmul`/`batch_matmul`, reductions, `softmax`, `argmax`) produce
    `Unspecified`: an image value-range never survives an operation that
    invalidates it (e.g. `argmax` yields indices, never ranged pixels).

  `R` in the operation signatures above is metavariable notation. This version
  permits only the **concrete** range states listed; there is no `R: Range`
  generic kind (no range-polymorphic user functions) — that is reserved (§11).

The canonical checked progression for a model whose input contract is
`Normalized`:

```text
decoded bytes (ByteRange, UInt8)
  → cast     → ByteRange Float32
  → scale_255 → UnitRange Float32
  → normalize → Normalized Float32
  → model input (range = Normalized)
```

Only one semantic property is defined in this version; colour space, coordinate
frame, and physical units remain **reserved** (§11).

## 9. Diagnostics Requirements
Shape errors are this extension's product surface; minimum quality bar:
- Name the operation and the unsatisfied constraint in dim-expression form
  (`matmul requires K == K2; K = 768 (from `weights`), K2 = 512 (from
  `features`)`).
- Trace each symbolic dim to its origin (parameter, `refine`, or literal).
- When the fix is mechanical, say it (`insert .permute<[0, 3, 1, 2]>()` /
  `broadcast_to` / `refine`).

## 10. Lowering and Backends (Informative)
This extension deliberately defines no VM, kernels, or execution semantics
beyond value-level results. Expected lowering targets:
- **StableHLO → IREE** for compiled CPU/GPU deployment;
- **ONNX Runtime** for model execution with generated host glue;
- **generated Rust/C** calling an existing tensor library.

Because every shape/dtype/device fact is static after refinement, lowering
can monomorphize dims where literal and emit symbolic-dim kernels where not,
and buffer sizes for fully static graphs are computable at compile time
(enabling ahead-of-time memory planning).

## 11. Reserved / Future
- Quantized dtypes (`QInt8<scale, zero_point>`) and quantization checking
- Named axes (`[batch: B, channel: 3, ...]`) and axis-name-aware ops
- Memory layout types and layout-change tracking
- Peak-memory profiles as compile-time deployment constraints
- Training, autodiff, and gradient types
- Open-ended symbolic constraints (`B <= 64`) and dimension range reasoning
- Other semantic tensor properties beyond the Gate 7 value-range (§8a): colour
  space, coordinate frame, and physical units
- A `Range` generic kind (`R: Range`) for range-polymorphic functions; v0.1
  permits concrete value-range states only (§8a)

## 12. Conformance
An implementation claiming support for extension `tensor` v0.1 MUST implement
§3–§9 as specified, MUST document its backend mapping (§10), and MUST reject
reserved constructs (§11) with a diagnostic. Deviations MUST be documented.
This extension MUST NOT alter the behavior of programs that do not use it.

## Appendix A: Worked Example

```stark
use tensor::*;

model ResNet50<B: Dim> {
    input  image: Tensor<Float32, [B, 3, 224, 224]>;
    output class: Tensor<Float32, [B, 1000]>;
}

fn classify<B: Dim>(
    model: &ResNet50,
    images: &Tensor<UInt8, [B, 224, 224, 3]>
) -> Tensor<Float32, [B, 1000]> {
    let normalized = images
        .permute::<[0, 3, 1, 2]>()      // [B, 3, 224, 224], NHWC -> NCHW order
        .cast::<Float32>()
        .div(&IMAGENET_STD)             // broadcast: [3, 1, 1] vs [B, 3, 224, 224]
        .sub(&IMAGENET_MEAN);

    model.predict(&normalized)
}

fn main() -> Result<Unit, ServeError> {
    let model = ResNet50::load("resnet50.onnx")?;   // signature verified here
    let raw: TensorAny = read_request()?;
    let images = raw.refine::<UInt8, [B, 224, 224, 3]>()?;  // B bound here
    let logits = classify(&model, &images);         // no runtime shape checks
    respond(&logits)
}
```

Defects this catches at compile time that Python catches at runtime (or in
production): channel-order mistakes (missing `permute`), dtype mismatches
(forgotten `cast`), normalizing with a wrongly-shaped constant, calling a
model whose artifact signature drifted from the code, and batch-dimension
loss in postprocessing.
