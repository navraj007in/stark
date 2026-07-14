# STARK Extensions

## Overview
Extensions define non-core language features that may evolve independently of Core v1. Implementations MAY support extensions, but Core v1 conformance does not require them.

## Policy
- Each extension MUST be documented in this directory.
- Extensions SHOULD be versioned independently from Core v1.
- Extensions MUST NOT change the behavior of Core v1 features unless explicitly specified.
- Implementations claiming extension support MUST document the supported extension versions and any deviations.

## Current Extensions
- `Tensor-Model-Types.md` — the tensor & model type system (extension id
  `tensor`, v0.1, normative draft): symbolic dimensions, shape/dtype/device
  checking, `model` signature declarations, ONNX import contract.
- `AI-Extensions.md` — overview sketch of the remaining AI/ML surface
  (LLM blocks, datasets); the tensor and model sections defer to
  `Tensor-Model-Types.md`.
