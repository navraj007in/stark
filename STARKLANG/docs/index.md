# STARK Documentation Index

## Project Direction

- [Implementation Roadmap](./ROADMAP.md) — delivery gates, prototype success
  criteria, decision checkpoints, and explicitly deferred scope
- [Implementation Plan](./PLAN.md) — the engineering layer under the roadmap:
  standing technical decisions, per-gate work packages, test strategy, risks

## Normative Specification (Core v1)
The current, authoritative language specification lives in [`spec/`](./spec/):

1. [Core Language Overview](./spec/00-Core-Language-Overview.md)
2. [Lexical Grammar](./spec/01-Lexical-Grammar.md)
3. [Syntax Grammar](./spec/02-Syntax-Grammar.md)
4. [Type System](./spec/03-Type-System.md)
5. [Semantic Analysis](./spec/04-Semantic-Analysis.md)
6. [Memory Model](./spec/05-Memory-Model.md)
7. [Standard Library](./spec/06-Standard-Library.md)
8. [Modules and Packages](./spec/07-Modules-and-Packages.md)

The concise conformance summary is [`spec/09-STARK-Language-Spec-v1.md`](./spec/09-STARK-Language-Spec-v1.md).
A single-file compilation is available as [`spec/STARK-Core-v1.md`](./spec/STARK-Core-v1.md)
(with generated HTML and PDF alongside it); it is regenerated from the
individual files above and must not be edited directly.

## Non-Core Extensions (Optional)
- [Tensor & Model Type System](./extensions/Tensor-Model-Types.md) —
  extension `tensor` v0.1 (normative draft): symbolic dimensions,
  shape/dtype/device checking, `model` signatures, ONNX import
- [AI/ML Extensions Overview](./extensions/AI-Extensions.md) — sketches of the
  remaining AI surface (datasets, LLM blocks)

## Archive
[`archive/`](./archive/README.md) contains the superseded pre-pivot design
documents (AI-native/cloud-first era). They are retained for historical
reference only and contradict the normative spec in several places; see the
archive README for the conflict table.
