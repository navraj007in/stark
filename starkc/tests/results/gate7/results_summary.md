# Gate 7 (G7-05) — native deployment evidence

Commit `b263c9332099f4faed57aef07d9402aae1fb43e9` (dirty: False)

- Deploy: 903 ms · host build: 15.1 s
- Binary: 24.77 MB · model: 63.48 MB
- Startup: 184.2 ms · warm median: 14.25 ms · peak RSS: 440.4 MB
- Host output: Float32 [1, 5, 13, 13, 25] (checksum 9b6b21a8), shape matches reference: True
- Host-vs-reference (identical input) max_abs_diff: 0.00e+00 (tol 0.001, within: True)
- Model oracle max_abs_diff: 2.19e-05 (tol 0.001)
- Defects caught before inference: **13/13**
- Latency is a single 100-iteration sequence; min/median/p95 are reported, and tail variance reflects OS scheduling on a shared machine rather than the pipeline.

| defect | stage | code | caught |
|---|---|---|---|
| d1_reshape_product | compile-time (type checking) | [E0212] | True |
| d2_symbol_relationship | compile-time (type checking) | [E0212] | True |
| d3_broadcast | compile-time (type checking) | [E0212] | True |
| d4_range_missing | compile-time (type checking) | [E0212] | True |
| d5_range_double | compile-time (type checking) | [E0212] | True |
| d6_dtype | compile-time (type checking) | [E0212] | True |
| d7_device | compile-time (type checking) | [E0212] | True |
| d10_range_normalize_bytes | compile-time (type checking) | [E0212] | True |
| d11_range_wrong_model | compile-time (type checking) | [E0212] | True |
| d12_range_merge | compile-time (type checking) | [E0212] | True |
| d13_range_erase | compile-time (type checking) | [E0212] | True |
| d8_declaration_drift | deploy-time (artifact verification) | does not match the model declaration | True |
| d9_runtime_artifact_swap | runtime (generated-host load, before inference) | ArtifactMismatch | True |
