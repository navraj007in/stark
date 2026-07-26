# C6.4 Tier-1 qualification summary

- Commit: `4844702e1f1c9c8d6c11bde5fec9dfca793f096c`
- Records: `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`
- Generated corpus: BLOCKED-BY-C6.5 (see `WP-C6.4.md` §1.2)

## Platform identities

| Field | aarch64-apple-darwin | x86_64-unknown-linux-gnu |
|---|---|---|
| host_triple | aarch64-apple-darwin | x86_64-unknown-linux-gnu |
| selected_target_triple | aarch64-apple-darwin | x86_64-unknown-linux-gnu |
| os_name | Darwin | Linux |
| os_version | 23.6.0 | 6.17.0-1020-azure |
| architecture | arm64 | x86_64 |
| rustc_version_verbose | rustc 1.97.1 (8bab26f4f 2026-07-14) | rustc 1.97.1 (8bab26f4f 2026-07-14) |
| cargo_version | cargo 1.97.1 (c980f4866 2026-06-30) | cargo 1.97.1 (c980f4866 2026-06-30) |
| runner_provider | GitHub Actions 1000001251 | GitHub Actions 1000001249 |
| python_version | 3.14.6 | 3.12.3 |

## Required agreement

| Field | Value | Agrees |
|---|---|---|
| schema_version | c6.4-evidence-1 | yes |
| work_package | WP-C6.4 | yes |
| commit_sha | 4844702e1f1c9c8d6c11bde5fec9dfca793f096c | yes |
| compiler_version | 0.1.0 | yes |
| mir_version | 0.1 | yes |
| mir_runtime_surface | 0.1-A9 | yes |
| backend_version | 0.1 | yes |
| runtime_version | 0.1 | yes |
| layout_contract | stark-64-v1 | yes |
| layout_contract_version | 1 | yes |
| compiler_layout_revision | 1 | yes |
| target_tier | tier-1 | yes |
| target_pointer_width | 64 | yes |
| profile | debug | yes |
| determinism_result | match | yes |
| failed_count | 0 | yes |
| ignored_count | 2 | yes |
| skipped_count | 0 | yes |
| unclassified_ignores | [] | yes |
| classified_ignores | {'imports_and_verifies_checksum_pinned_reference_model': 'gate-4 tensor track; needs a checksum-verified ResNet50 named by STARK_GATE4_REFERENCE_ONNX. Outside the C6.4 Core-runtime matrix.', 'real_inference_agrees_with_reference': 'gate-5 tensor track; downloads and links ONNX Runtime and runs Python. Outside the C6.4 Core-runtime matrix.'} | yes |
| generated_corpus_status | BLOCKED-BY-C6.5 | yes |
| generated_corpus_version | None | yes |
| generated_corpus_case_count | 0 | yes |
| required_steps | ['fmt', 'clippy', 'c64_platform_matrix', 'three_engine_differential', 'mir_differential', 'exec_snapshots', 'c63_closure_evidence', 'conformance', 'release_package', 'workspace'] | yes |
| overall_result | PASS | yes |

## Per-command results

| Command | aarch64-apple-darwin | x86_64-unknown-linux-gnu |
|---|---|---|
| c63_closure_evidence | PASS (2 passed, 0 ignored) | PASS (2 passed, 0 ignored) |
| c64_platform_matrix | PASS (15 passed, 0 ignored) | PASS (15 passed, 0 ignored) |
| clippy | PASS (0 passed, 0 ignored) | PASS (0 passed, 0 ignored) |
| conformance | PASS (3 passed, 0 ignored) | PASS (3 passed, 0 ignored) |
| exec_snapshots | PASS (4 passed, 0 ignored) | PASS (4 passed, 0 ignored) |
| fmt | PASS (0 passed, 0 ignored) | PASS (0 passed, 0 ignored) |
| mir_differential | PASS (132 passed, 0 ignored) | PASS (132 passed, 0 ignored) |
| release_package | PASS (0 passed, 0 ignored) | PASS (0 passed, 0 ignored) |
| three_engine_differential | PASS (88 passed, 0 ignored) | PASS (88 passed, 0 ignored) |
| workspace | PASS (1461 passed, 2 ignored) | PASS (1461 passed, 2 ignored) |

## Result

**TIER-1 AGREEMENT**
