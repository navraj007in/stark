# Gate 7 (G7-05) — native deployment evidence

Commit `94b37fb0e29c6b74e2b224a54d7435c4a8362e57` (dirty: False)

- Deploy: 7850 ms · host build: 83.0 s
- Binary: 24.76 MB · model: 63.48 MB
- Startup: 474.2 ms · warm median: 49.95 ms · peak RSS: 420.2 MB
- Host output: Float32 [1, 5, 13, 13, 25] (checksum 9b6b21a8), shape matches reference: True
- Model oracle max_abs_diff: 2.19e-05 (tol 0.001)
- Defects caught before inference: **9/9**

| defect | stage | code | caught |
|---|---|---|---|
| d1_reshape_product | compile-time (type checking) | [E0212] | True |
| d2_symbol_relationship | compile-time (type checking) | [E0212] | True |
| d3_broadcast | compile-time (type checking) | [E0212] | True |
| d4_range_missing | compile-time (type checking) | [E0212] | True |
| d5_range_double | compile-time (type checking) | [E0212] | True |
| d6_dtype | compile-time (type checking) | [E0212] | True |
| d7_device | compile-time (type checking) | [E0212] | True |
| d8_declaration_drift | deploy-time (artifact verification) | does not match the model declaration | True |
| d9_runtime_artifact_swap | runtime (generated-host load, before inference) | ArtifactMismatch | True |
