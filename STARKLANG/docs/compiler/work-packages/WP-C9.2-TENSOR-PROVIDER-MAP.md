# WP-C9.2 Tensor/ONNX Provider Map

Date: 2026-07-31

## Status

Open. This map records the current ONNX path and candidate responsibilities. It deliberately does
not introduce a provider trait or rename ONNX/tensor domain types to generic artifact terminology.

## Stage Map

| Stage | Owner | Input | Output | Classification |
| --- | --- | --- | --- | --- |
| Parse extension syntax | `parser.rs`, `ast.rs` | STARK source + `LanguageOptions` | AST tensor/model nodes or diagnostics | TENSOR-SPECIFIC |
| Resolve extension symbols | `resolve.rs` | AST + options | HIR with extension-reserved names | TENSOR-SPECIFIC |
| Type-check tensor/model semantics | `typecheck.rs`, `extensions/tensor/*` | HIR + type tables | extension types, model contracts, diagnostics | TENSOR-SPECIFIC |
| Read ONNX artifact | `onnx/importer.rs`, `onnx/verifier.rs` | filesystem path | decoded ONNX model/signature | POSSIBLY-REUSABLE |
| Normalise artifact content | `onnx/importer.rs` | ONNX graph/types | supported tensor signature | EVIDENCE-INSUFFICIENT |
| Extract typed signature | `onnx/importer.rs::ModelSignature` | normalised ONNX model | model inputs/outputs/dtypes/shapes | POSSIBLY-REUSABLE |
| Record identity/provenance | `onnx/importer.rs`, diagnostics | artifact bytes/path/hash | SHA-256 comments and provenance | POSSIBLY-REUSABLE |
| Generate/register declarations | `onnx/importer.rs::format_declaration` | model signature + artifact stem/hash | STARK `model` declaration | TENSOR-SPECIFIC |
| Verify drift | `onnx/verifier.rs` | declaration + current artifact | mismatch diagnostics | POSSIBLY-REUSABLE |
| Lower deployment pipeline | `deploy/lower.rs`, `deploy/emit.rs` | checked tensor pipeline + ONNX model | deploy IR/host crate | MUST-REMAIN-SEPARATE |
| Backend/runtime obligations | `backend/generated_rust/*`, deploy emitter | verified model/deploy IR | runtime hash/load checks | MUST-REMAIN-SEPARATE |
| Structured diagnostics | `diag.rs`, ONNX/typecheck modules | phase-specific failures | stable error reports | POSSIBLY-REUSABLE |

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

The artifact identity is the SHA-256 of the ONNX artifact bytes. The generated declaration records
that hash in the header comment and carries source provenance through diagnostics. Drift is defined
as a mismatch between the declared model contract and the signature extracted from the current ONNX
artifact, plus runtime artifact hash mismatch in generated hosts.

Artifact reading happens in import, verify, and deployment-related paths. It is not a Core language
feature and is not available through arbitrary compiler plugins. Compiler phases that explicitly
know ONNX today include `onnx/*`, tensor type checking where model declarations are interpreted, and
deploy lowering/emission for ONNX Runtime.

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
