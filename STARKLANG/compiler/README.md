# Prototype Type Checker (Pre-Pivot — Not Core v1)

This Python code is an early prototype written against the **pre-pivot** STARK
design: its AST includes `ACTOR_DECL`, `MODEL_DECL`, and `PIPELINE_DECL` nodes
and a `TensorType`, none of which exist in the current Core v1 language
(`docs/spec/`). It does not implement the Core v1 grammar, type rules, or
ownership analysis.

It is kept as a reference/experiment only. A Core v1 implementation should be
started fresh against `docs/spec/STARK-Core-v1.md`.
