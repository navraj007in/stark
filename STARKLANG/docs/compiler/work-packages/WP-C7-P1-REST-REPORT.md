# WP-C7.P1 Native HTTP/JSON REST Workload Report

**Date:** 2026-07-30
**Local qualification:** macOS arm64
**Recommendation:** `P1 TIER-1 QUALIFIED` — all six execution rows green (CD-273).

**Status history.** The Tier-1 Linux and Windows rows were framed as manual runs and had stayed
unrun, holding Gate C7 open. CD-271 made them a CI job (`C7 P1 REST workload`, §7); CD-272 fixed the
three path assumptions its first run exposed; the run at `d735b35` is green on all six rows:

| platform | debug execution | release execution |
| --- | --- | --- |
| linux-x64 | **PASS** | **PASS** |
| macos-arm64 | **PASS** | **PASS** |
| windows-x64 | **PASS** | **PASS** |

Each row is the artefact **executed** — 24 raw HTTP exchanges compared byte-for-byte, then a bounded
clean exit — not merely built. Pure-STARK tests (7/7) also pass on all three.

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

Raw observations are retained in `measurements/debug.json` and `measurements/release.json`; what they do and do not support is stated in `measurements/README.md`.

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

**This is `DEFECT-C788-LOOP-TEMP`, and it is DISCHARGED. `P1-COMPILER-001` is a local label for a
defect already under governance, not a second finding.** Recorded CD-263, ruled a non-blocking C7
deviation CD-264, fixed by MIR amendment A12 (`Statement::StorageDead`, MIR `0.2` → `0.3`) at CD-265
and approved retrospectively as CE3; a surviving `?`-in-a-loop instance — the one shape A12's
sixteen-shape matrix missed, because `lower_try` builds its own scrutinee temporary — was found by
`stark-json` and fixed under CD-269. Regression: `starkc/tests/a12_storage_end_shapes.rs`. Root
cause: **any** place whose storage is emptied piecewise, not temporaries specifically. Full argument:
`../mir-amendment-A12-storage-end.md`; ledger: `COMPILER-STATE.md`.

The adaptation described below therefore documents what P1 did at implementation time, on a compiler
that still had the defect. It is **not** a live constraint on the language, and it is not a reason to
edit P1 — the workload is frozen at its qualifying commit.

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

**Superseded by CD-273 — see the recommendation at the head of this report.** The handoff below was
written at the 2026-07-30 macOS-only state and is retained as history. The Linux and Windows rows it
was waiting on are green (§7); the current status is `P1 TIER-1 QUALIFIED`, not `P1 PARTIAL`.

> All mandatory application behavior and the macOS native path are demonstrated. P1 should not yet be
> recorded as fully satisfied because the required Tier-1 Linux/Windows evidence is outstanding.
>
> ```text
> P1 PARTIAL — implementation and macOS arm64 qualification pass;
> Tier-1 Linux/Windows qualification remains.
> ```

This report does not declare Gate C7 closed.

---

## 7. Tier-1 cross-platform qualification (CD-271)

**These rows were never a research problem.** CI already ran on all three Tier-1 platforms; the
workload simply was not wired into it, and its harness had two portability gaps that would have
failed on Windows regardless:

- `e2e.py` built its default artefact path without an `.exe` suffix;
- `measure.py` invoked `python3`, which Windows runners do not provide (now `sys.executable`).

Both are fixed, and the job runs per push:

| step | why it is in the job |
| --- | --- |
| pure STARK tests | application logic with **no host capabilities**, so a later failure is attributable to the provider path or to the logic, not to both |
| `stark build --no-build-cache` (debug) | the compile path, uncached |
| **`e2e.py --profile debug`** | **executes the artefact**: 24 raw HTTP exchanges, every response compared byte-for-byte, then a bounded clean exit |
| `stark build --release --no-build-cache` | the release compile path |
| **`e2e.py --profile release`** | executes the release artefact under the same 24 cases |
| `stark fmt --check` | formatting |

**The job builds *and* runs, and that split is deliberate.** `stark-json`'s native evidence was
recorded on a successful `stark build` whose binary had never been executed; when it finally was, it
aborted immediately on a real compiler defect (`?` in a loop, CD-269). A build-only Tier-1 job would
have reproduced that mistake on three platforms instead of one.

Local dry-run of the exact sequence at `cd9405b` on macOS arm64: pure tests 7/7, debug 24/24,
release 24/24, `fmt --check` clean. The rows moved from PENDING to PASS on CI evidence at
`d735b35`, not on the job existing.

**The job's first run earned its keep** (CD-272): `pure_tests.py` hardcoded `target/debug/stark`
while the job builds `--release`, so linux-x64 failed before running a test. Two sibling assumptions
were fixed in the same pass — a relative `--compiler` read against a temporary working directory,
and trusting each platform's shell to resolve `.../stark` to `stark.exe`. All three were path
assumptions that held on the machine the scripts were written on, which is also why these rows had
stayed unrun: not difficulty, just assumptions nothing had forced anyone to check.

## 8. Measurements

Re-taken on both profiles at `cd9405b`, replacing `measurements/latest.json` — a single debug run at
`a6964d9`, a commit predating `MIR_VERSION` 0.2 → 0.3, whose numbers describe a compiler that no
longer exists.

Executable size is a real result: **860,784 B debug vs 510,592 B release, 1.686×**.

Runtime is not. `functional_run_seconds` times the `e2e.py` subprocess, and the debug/release ratio
of **1.003×** is the proof: C7.5 measured 1.5× between profiles on startup alone. The harness floor
— Python startup, a deliberate sleep, process spawn, 24 loopback round trips — dominates the
microseconds of STARK compute. C7.5's two runtime dimensions therefore stay `NOT MEASURABLE`, now
on two workloads' evidence. Detail in `measurements/README.md` and `WP-C7.5-PERFORMANCE-REPORT.md`
§7, §8.

**Do not quote `321 req/s` or `66 ms` as server throughput** (CD-273). They describe the harness.

**P1 is frozen at 24 exchanges and will not be extended to serve as a benchmark** (CD-273). Raising
the request count would fuse functional qualification with performance measurement and make this
workload's identity depend on benchmark requirements — and its byte-exact corpus is precisely what
the Tier-1 evidence above is defined against. The performance instrument is a separate versioned
artefact, specified in `WP-C7.5-PERFORMANCE-REPORT.md` §8, and is follow-on work rather than a C7
prerequisite.