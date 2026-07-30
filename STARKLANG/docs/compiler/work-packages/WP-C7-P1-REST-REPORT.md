# WP-C7.P1 Native HTTP/JSON REST Workload Report

**Date:** 2026-07-30
**Local qualification:** macOS arm64
**Recommendation:** `P1 PARTIAL — Tier-1 cross-platform runs remain`

## Implementation status

The workload is implemented at `starkc/tests/workloads/c7-p1-rest`. An ordinary
`stark build --no-build-cache` parses, resolves, typechecks, lowers to MIR 0.2, emits Rust, links
the selected first-party providers, and produces the native REST server.

Application behavior implemented in STARK:

- bounded HTTP/1.1 request-line/header/body parsing;
- case-insensitive required header matching;
- byte-exact response encoding and decimal `Content-Length`;
- pure-STARK JSON validation for object, array, string, integer, Boolean, and null values;
- JSON simple escapes, BMP `\uXXXX` validation, UTF-8 validation, and string-token preservation;
- `/health`, fixture item lookup, and `POST /items` routing;
- partial-read assembly and partial-write-safe `write_all`;
- a fixed 24-accept deterministic lifecycle.

Host capabilities are limited to TCP bind/accept/read/write and optional `STARK_P1_BIND`
environment lookup. Package source neither declares nor calls a provider close function.

## Correctness evidence

- Pure STARK: `python3 scripts/pure_tests.py` — 7 passed.
- Native build: `stark build --no-build-cache --verbose` — succeeded; 36 MIR bodies verified.
- Native raw TCP: `python3 scripts/e2e.py` — 24/24 cases passed.
- Formatting: `stark fmt --check` — passed.
- Python harness syntax: `python3 -m py_compile scripts/*.py` — passed.

The raw corpus covers repeated health requests, both fixture items, absent/bad/empty/signed/
overflowing IDs, successful ASCII/escaped/UTF-8 POST bodies, malformed/missing/duplicate/non-string/
empty `name`, missing and conflicting lengths, transfer coding, unknown path, unsupported method,
missing Host, lowercase Host, and a request split across client writes. Every response is compared
byte-for-byte, including headers, length, and body. The process exits successfully after exactly 24
accepted connections.

The standalone Python runner is both CI-suitable test infrastructure and the local qualification
authority; it directly exposes subprocess stdout/stderr and raw socket failures.

## Local measurements

Raw observations are retained in `measurements/latest.json`.

| Observation | macOS arm64 debug |
|---|---:|
| STARK source lines | 1,152 |
| generated Rust lines | 9,033 |
| binary size | 859,280 bytes |
| cold native build | 1.772 s |
| no-change build median, 3 samples | 0.733 s |
| validated 24-request run median, 5 samples | 0.0747 s |
| derived sequential throughput | 321.5 requests/s |

The throughput figure includes process startup and shutdown and is only a workload-local baseline,
not a framework comparison. Each timed run validates all output. The first functional sample is
retained as a visible cold/outlier observation.

## Compiler finding and workload adaptation

### P1-COMPILER-001 — repeated enum-result assignment retains a live generated slot

Classification: **COMPILER DEFECT AGAINST EXISTING SPEC**

Minimal shape:

```stark
while condition {
    match enum_returning_function() {
        Some(value) => { use(value); }
        None => {}
    }
}
```

Expected: each iteration completes the prior result lifetime before the next assignment.

Actual native result: the second assignment can trap in `stark-runtime/src/slot.rs` with:

```text
generated-code invariant violated: write to a live slot
```

Stage: generated-Rust execution after verified MIR.

P1 initially exposed this in repeated parse, accept, read, and write result sites. No compiler,
provider, resource identity, close arena, or Drop-planning code was changed. The workload uses
single-call helper activation records (`serve_next`, `read_once`, and `write_once`) and a Copy-only
framing readiness scan so every enum-returning call site executes once per function invocation.
The full HTTP parser still decides the request exactly once.

## Explicit limitations and deferrals

- JSON floating point/exponent forms are rejected; P1 accepts integer numbers.
- Unicode escape surrogate pairs are rejected; non-surrogate BMP escapes are validated.
- The bounded request count is the documented test-build constant (24), the work package's third
  termination preference. Bind address selection uses `STARK_P1_BIND`.
- The provider does not expose the selected address for port zero, so the harness reserves and
  supplies a free loopback port.
- Linux and Windows functional qualification and CI evidence have not been produced in this local
  run.
- Release-profile and optimiser-on/off runtime samples remain for the C7.4/C7.5 qualification rerun;
  the harness supports reproducing the functional baseline.

## Gate handoff

All mandatory application behavior and the macOS native path are demonstrated. P1 should not yet be
recorded as fully satisfied because the required Tier-1 Linux/Windows evidence is outstanding.

```text
P1 PARTIAL — implementation and macOS arm64 qualification pass;
Tier-1 Linux/Windows qualification remains.
```

This report does not declare Gate C7 closed.
