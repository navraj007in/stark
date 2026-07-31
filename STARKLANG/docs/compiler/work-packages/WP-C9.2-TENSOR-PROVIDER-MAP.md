# WP-C9.2 Tensor/ONNX Provider Map

Date: 2026-07-31

## Status

Closed for Part A. This map records the current ONNX path and candidate responsibilities. It
deliberately does not introduce a provider trait or rename ONNX/tensor domain types to generic
artifact terminology.

## Stage Map

| Stage | Owner | Input | Output | Tests / diagnostics | Classification |
| --- | --- | --- | --- | --- |
| Parse extension syntax | `parser.rs`, `ast.rs` | STARK source + `LanguageOptions` | AST tensor/model nodes or diagnostics | parser Gate 4/C9 tests; disabled-extension diagnostics | TENSOR-SPECIFIC |
| Resolve extension symbols | `resolve.rs` | AST + options | HIR with extension-reserved names | resolver Gate 4 tests for reserved names and builtins | TENSOR-SPECIFIC |
| Type-check tensor/model semantics | `typecheck.rs`, `extensions/tensor/*` | HIR + type tables | extension types, model contracts, diagnostics | tensor signature/predict/drift tests; `E0211` extension diagnostics | TENSOR-SPECIFIC |
| Read ONNX artifact | `onnx/importer.rs::read_signature`, `decode_signature` | filesystem path / bytes | `ModelSignature` + SHA-256 | ONNX importer/verifier tests; `OnnxError` | POSSIBLY-REUSABLE |
| Normalise artifact content | `onnx/importer.rs` protobuf readers | ONNX graph/value info | supported tensor ports, dtype, dynamic identity | malformed/limit errors through `OnnxError` | EVIDENCE-INSUFFICIENT |
| Extract typed signature | `ModelSignature`, `Port`, `DType`, `Dimension` | normalised ONNX model | model inputs/outputs/dtypes/shapes | Gate 5/7 signature tests | POSSIBLY-REUSABLE |
| Record identity/provenance | `read_signature`, `format_declaration` | artifact bytes/path/hash | lowercase SHA-256 in generated header and deploy manifest | Gate 5/7 hash/drift tests | POSSIBLY-REUSABLE |
| Generate/register declarations | `format_declaration`, `model_identifier` | model signature + artifact stem/hash | STARK `model` declaration | import/golden tests | TENSOR-SPECIFIC |
| Verify drift | `onnx/verifier.rs::verify_declaration_*`, `Difference` | declaration + current artifact | `VerificationReport` | Gate 5/7 drift tests | POSSIBLY-REUSABLE |
| Lower deployment pipeline | `deploy/lower.rs::lower_pipeline`, `deploy/mod.rs::deploy` | checked tensor pipeline + ONNX model | deploy IR/host crate | Gate 5/7 deploy lowering tests | MUST-REMAIN-SEPARATE |
| Backend/runtime obligations | `deploy/emit.rs`, `deploy/template/runtime.rs.in` | verified model/deploy IR | runtime hash/load/shape checks | runtime template tests; `ArtifactMismatch` | MUST-REMAIN-SEPARATE |
| Structured diagnostics | `diag.rs`, ONNX/typecheck/deploy modules | phase-specific failures | stable CLI/LSP reports | C9/LSP diagnostics and Gate 5/7 tests | POSSIBLY-REUSABLE |

## Data Flow

```text
source -> parser with tensor enabled -> AST model/tensor nodes -> resolver -> typed semantic model
artifact path -> ONNX reader -> normalised ONNX graph -> typed model signature
typed model signature -> deterministic STARK model declaration
artifact bytes -> SHA-256 -> generated declaration provenance / runtime identity check
declaration + later artifact -> verifier drift comparator -> diagnostics
verified model + pipeline -> deployment lowering -> ONNX Runtime host obligations
```

## Questions Answered So Far

The artifact identity is the lowercase SHA-256 of the ONNX artifact bytes. `read_signature` returns
the typed signature and hash; `format_declaration` records the hash in the generated declaration
header, and deploy emission records it in the generated host constants and manifest.

Artifact reading happens in import, verify, and deployment-related paths. It is not a Core language
feature and is not available through arbitrary compiler plugins. Compiler phases that explicitly
know ONNX today include `onnx/*`, tensor type checking where model declarations are interpreted, and
deploy lowering/emission for ONNX Runtime.

Drift is defined by `verify_declaration_source`: port count, port order/name, dtype, rank, static
dimension values, and dynamic dimension identity relationships must agree between the declaration
and the current artifact signature. Runtime artifact replacement is a separate deployment-host
obligation: generated hosts compute the model file SHA-256 before creating an ONNX Runtime session
and fail with `ArtifactMismatch` if it differs.

The ONNX decoder is bounded by `DEFAULT_LIMITS`: 256 MiB file and protobuf field limits, nesting
depth 32, 4096 inputs, 4096 outputs, 1,000,000 initializers, tensor rank 64, and 4096-byte names.
Malformed protobuf data returns `OnnxError`; no artifact code executes during compilation.

There is no persistent artifact cache in the ONNX importer/verifier. Build-cache state belongs to
native build output reuse, not ONNX signature extraction. Deterministic stages include protobuf
field traversal for supported fields, declaration formatting, SHA-256 identity, and drift
comparison; filesystem reads are the only external input.

## Candidate Abstraction Ledger

- TENSOR-SPECIFIC: tensor syntax, shape algebra, dtype semantics, model declaration grammar,
  tensor deployment lowering.
- POSSIBLY-REUSABLE: artifact byte identity, provenance recording, typed contract extraction shape,
  declaration/artifact drift comparison shape, structured diagnostic plumbing.
- EVIDENCE-INSUFFICIENT: normalisation boundaries, cache policy, resource limits across artifact
  kinds, typed declaration registration for non-tensor contracts.
- MUST-REMAIN-SEPARATE: ONNX Runtime deployment obligations, tensor graph semantics, runtime model
  loading and inference host code.

## Part B Blocker

C9.3 requires a second working artifact implementation before any internal artifact-provider
contract can be proposed. This map is evidence inventory only.

## Closeout

C9.2 Part A is closed: the ONNX path is traceable, stage ownership is explicit, candidate
responsibilities are classified without marking anything `REUSABLE`, and no ONNX refactor was
performed. The concrete C9.3 entry requirement is a second implementation with artifact reading,
typed contract extraction, identity/provenance, drift verification, structured diagnostics, and
tests.
