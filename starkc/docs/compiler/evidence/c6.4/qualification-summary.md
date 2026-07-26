# C6.4 Tier-1 qualification summary

- Commit: `61008f628312fa059d095b553d5a0f77f50a27bc`
- Records: `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`
- Generated corpus: BLOCKED-BY-C6.5 (see WP-C6.4 §1.2)

## Platform identities

| Field | aarch64-apple-darwin | x86_64-unknown-linux-gnu |
|---|---|---|
| host_triple | aarch64-apple-darwin | x86_64-unknown-linux-gnu |
| selected_target_triple | aarch64-apple-darwin | x86_64-unknown-linux-gnu |
| os_name | Darwin | Linux |
| os_version | 23.6.0 | 6.17.0-1020-azure |
| architecture | arm64 | x86_64 |
| cargo_version | cargo 1.97.1 (c980f4866 2026-06-30) | cargo 1.97.1 (c980f4866 2026-06-30) |
| runner_provider | GitHub Actions 1000001190 | GitHub Actions 1000001192 |
| python_version | 3.14.6 | 3.12.3 |

## Required agreement

| Field | Value | Agrees |
|---|---|---|
| schema_version | c6.4-evidence-1 | yes |
| commit_sha | 61008f628312fa059d095b553d5a0f77f50a27bc | yes |
| compiler_version | 0.1.0 | yes |
| mir_version | 0.1 | yes |
| mir_runtime_surface | 0.1-A9 | yes |
| backend_version | 0.1 | yes |
| runtime_version | 0.1 | yes |
| layout_contract | stark-64-v1 | yes |
| profile | debug | yes |
| determinism_result | match | yes |
| failed_count | 0 | yes |
| ignored_count | 2 | yes |
| skipped_count | 0 | yes |
| unclassified_ignores | [] | yes |
| overall_result | PASS | yes |

## Per-command results

| Command | aarch64-apple-darwin | x86_64-unknown-linux-gnu |
|---|---|---|
| c63_closure_evidence | PASS (2) | PASS (2) |
| c64_platform_matrix | PASS (15) | PASS (15) |
| clippy | PASS (0) | PASS (0) |
| conformance | PASS (3) | PASS (3) |
| exec_snapshots | PASS (4) | PASS (4) |
| fmt | PASS (0) | PASS (0) |
| mir_differential | PASS (132) | PASS (132) |
| release_package | PASS (0) | PASS (0) |
| three_engine_differential | PASS (88) | PASS (88) |
| workspace | PASS (1461) | PASS (1461) |

## Result

**TIER-1 AGREEMENT**
