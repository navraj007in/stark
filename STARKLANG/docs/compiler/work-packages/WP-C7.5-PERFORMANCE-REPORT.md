# WP-C7.5 — performance and complexity report

**Status:** `PARTIAL — OPEN, BLOCKED ON P1`. Six of the eight required dimensions are measured. Two
are **not measurable on this corpus** and are reported as such rather than estimated. C7.5 cannot
close until P1 supplies practical systems workloads; that is a roadmap condition, and nothing here
attempts to work around it.
**Measured at:** this change, macOS arm64, seven frozen C7 workloads, `scripts/c7-baseline.py
--report`.

---

## 1. The finding that governs how the rest should be read

**The heaviest frozen workload's compiled runtime is indistinguishable from an empty program.**

| binary | min wall time (25 runs) | compute above the startup floor |
| --- | --- | --- |
| `w01_minimal`, release — the startup floor | 1.547 ms | — |
| `w02_arith_control`, release | 1.527 ms | **−0.019 ms — below measurement resolution** |
| `w02_arith_control`, debug | 2.284 ms | 0.737 ms |
| `w02_arith_control`, HIR interpreter | 86.174 ms | 84.627 ms |

`w02` runs 300 Collatz sequences — the most arithmetic in the corpus — and natively that costs less
than the run-to-run noise on process startup. Every other workload does less.

So this corpus can measure compile time, memory, size and the interpreter's cost. It **cannot**
measure steady-state native runtime, and therefore cannot measure the debug/release runtime ratio
either. That is not a gap in the harness; it is the corpus being made of programs that finish before
they can be timed, and it is exactly why the roadmap blocks closure on P1.

**No general performance multiple is claimed**, per the roadmap's explicit instruction. The one
ratio below is a single-workload figure and is labelled as one.

## 2. The eight dimensions

| dimension | status | value |
| --- | --- | --- |
| compile time | **measured** | 0.39–0.44 s cold, per workload, either profile |
| peak compiler memory | **measured** | 148.6–156.3 MB, essentially flat |
| executable size | **measured** | debug 482–532 KB; release 443–447 KB |
| startup time | **measured** | 1.55 ms (release), 2.2–2.3 ms (debug) |
| steady-state runtime | **NOT MEASURABLE** | §1 (micro-workloads: process-startup floor); §7 (P1 end-to-end: harness floor) |
| interpreter/native ratio | **measured, one workload** | ≈115× against native debug — §4 |
| debug/release ratio | **size MEASURED (closed); runtime NOT MEASURABLE** | size 1.09–1.19× (micro-workloads), **1.686× (P1)** — §5, §7 |
| backend maintenance complexity | **measured** | §6 |

## 3. Compile time, memory and size

| workload | compile debug | compile release | peak RSS | size debug | size release |
| --- | --- | --- | --- | --- | --- |
| w01_minimal | 1.04 s* | 0.42 s | 156.3 MB | 482,608 | 443,328 |
| w02_arith_control | 0.39 s | 0.40 s | 149.5 MB | 490,848 | 445,408 |
| w03_generic_trait | 0.40 s | 0.41 s | 148.7 MB | 489,296 | 444,160 |
| w04_string_vec | 0.43 s | 0.43 s | 149.2 MB | 505,952 | 447,280 |
| w05_hash | 0.43 s | 0.44 s | 148.9 MB | 531,584 | 447,456 |
| w06_multi_package | 0.39 s | 0.41 s | 149.7 MB | 489,120 | 445,408 |
| w07_drop_ownership | 0.41 s | 0.41 s | 148.6 MB | 498,928 | 445,408 |

\* w01's debug figure is the first build of the session and carries the `stark-runtime` compile;
it is left in rather than discarded, with the cause named.

Three things are worth stating because they are not obvious from the numbers:

**Release does not cost more to compile than debug here** (0.40 s vs 0.41 s median). On a real
codebase `opt-level=3` costs real time; on programs this small, the fixed cost of starting Cargo and
rustc swamps it. This is a fact about the corpus, not about the profiles.

**Peak memory is flat at ~149 MB regardless of workload**, which means it is a floor — the toolchain
process footprint — not a function of program size. Note precisely what is measured:
`ru_maxrss` over child processes is the largest single child, almost certainly rustc, not the sum of
concurrent processes. It bounds the largest resident process, not total system pressure. This
partially answers the peak-memory limitation WP-C7.3 recorded as unmeasured; the multi-package
dependency-invalidation half of that limitation remains open.

**Release binaries are uniform at 443–447 KB across every workload**, a 4 KB spread over programs
ranging from `print` to hash maps. That is the runtime and std floor; STARK code contributes
single-digit kilobytes. Debug adds 39–88 KB of debug information.

## 4. The interpreter/native ratio

Measured on `w02_arith_control`, the only workload where both sides clear the noise floor:

**≈115× — the HIR interpreter's 84.6 ms of compute against the native debug binary's 0.737 ms.**

Every qualification this number needs:

- It is **one workload**, arithmetic-and-loops, and the roadmap forbids generalising from it.
- It is against **debug**, not release, because release compute is below resolution. The ratio
  against release is larger by an unknown factor and is deliberately not stated.
- The interpreter figure includes `stark run` re-doing parse, resolve and typecheck. That overhead
  is ~1.2 ms (from `w01`, whose interpreter run is 2.72 ms against a 1.55 ms floor), so it is about
  1.4 % of the 84.6 ms and does not change the conclusion — but the number is "interpret including
  front end", not pure interpretation.

The other six workloads produce ratios of 1.2–3.8×, and **those figures are meaningless**: both
sides are dominated by process startup, so the ratio is measuring how long it takes to start two
different programs. They are excluded rather than averaged in.

## 5. The debug/release ratio

**Size: measured.** Debug is 1.09–1.19× the release size — 482,608/443,328 = 1.089 at the low end
(`w01`) and 531,584/447,456 = 1.188 at the high end (`w05_hash`).

**Runtime: not measurable.** Release compute on the heaviest workload is below resolution, so the
ratio has no defined value. The raw wall-time ratios the harness computes (0.93–1.75) are startup
noise and must not be quoted; `w05`'s 1.75 in particular is a 1.6 ms difference on a 2 ms baseline.
They are recorded in the JSON for completeness and contradicted here on purpose, so that anyone
reading the machine output alone does not take them as findings.

## 6. Backend maintenance complexity

Counts, not adjectives — C7.6's DEFER decision needs a quantified basis for "the current backend is
maintainable", and an unquantified assertion is precisely what that decision must not rest on.

| component | files | lines |
| --- | --- | --- |
| `src/backend/generated_rust` | 11 | 6,608 |
| `src/mir` (lower, verify, interp, opt, drop_plan) | 6 | 16,872 |
| `stark-runtime/src` | 12 | 1,833 |

The backend proper is **6,608 lines**, and it emits Rust rather than machine code — so it inherits
register allocation, instruction selection, and every target rustc supports, at zero maintenance
cost. Lines of code is a crude proxy and is labelled as one; what it supports is a bounded claim:
the backend is a small fraction of the compiler, and the MIR layer it consumes is 2.5× its size.

## 7. What blocks closure

C7.5 closes when P1 supplies the practical systems workloads the roadmap names: file-processing CLI,
JSON parser, sequential HTTP server, request-routing benchmark, allocation-heavy String/Vec. Those
are the first workloads on which steady-state runtime, the debug/release runtime ratio, and a
defensible interpreter/native ratio can be measured at all.

Two further things should be re-measured when they exist, and are recorded here so they are not
forgotten:

1. **Whether the C7.4 folding passes ever fire.** They fire zero times on all seven frozen
   workloads. Configuration-heavy code is the plausible place they would, and P1's workloads are the
   first realistic sample.
2. **Whether compile time is still ~0.4 s** at a realistic program size, and whether the 65–68 %
   host share C7.0 measured holds there. Both current figures are from programs of a few dozen
   lines.

---

## 7. Re-tested against P1 (CD-271), and the verdict does not move

The two open dimensions were expected to close once a *real* workload existed, on the reasoning that
the micro-workloads of §1 were simply too small. P1 — a native HTTP/JSON REST server, 1,152 lines of
STARK, 24 byte-exact request/response exchanges — is that workload, and it was measured on both
profiles at compiler commit `cd9405b`.

**Executable size closes. Runtime does not.**

| | debug | release | ratio |
| --- | --- | --- | --- |
| executable size | 860,784 B | 510,592 B | **1.686×** |
| `functional_run_seconds` median | 66.72 ms | 66.50 ms | **1.003×** |

The size row is a real measurement and a wider spread than the micro-workloads' 1.09–1.19×, which
is what a larger program should show.

**The runtime row is not a measurement of the workload, and its own value proves it.** A 1.003×
debug/release ratio cannot describe STARK compute when §5 measured 1.5× between the profiles on
startup alone and §4 measured ~115× between the interpreter and native. What the harness times is
its own subprocess: ~15.7 ms of Python startup, a deliberate 10 ms sleep in the split-write case,
a server spawn, and 24 sequential loopback round trips. Handling 24 small requests is microseconds
of compute inside a floor of tens of milliseconds.

So §1's conclusion is unchanged, and now rests on two independent workloads rather than one: **the
generated code is fast enough that this project's measurement apparatus cannot see it.** What has
changed is the reason — the micro-workloads were below the *process startup* floor, and P1 is below
its *harness* floor. Neither is a statement about the backend.

Closing these two dimensions needs an instrument, not a bigger program: timing inside the server or
amortised across enough requests that a fixed floor stops mattering, and a request count above P1's
frozen 24-accept lifecycle. Raising that count changes the workload P1 is qualified against, so it
is an owner decision rather than a harness fix. Recorded in
`starkc/tests/workloads/c7-p1-rest/measurements/README.md`.

---

## 8. CLOSING RECORD (CD-273, owner ruling)

```text
Executable-size profile effect:
    MEASURED — release materially smaller than debug.

Micro-workload runtime profile effect:
    NOT MEASURABLE — dominated by process-startup floor.

P1 REST end-to-end runtime profile effect:
    NOT MEASURABLE — dominated by harness startup, deliberate delay,
    process supervision and loopback exchanges.

Backend steady-state runtime claim:
    NONE.

Future measurement:
    requires a separate amortised or internally instrumented benchmark;
    the frozen P1 qualification workload will not be modified.
```

**No throughput or speedup number from this workload may be quoted.** Specifically, the `321 req/s`
and `66 ms` figures describe the `e2e.py` harness and must not be reported as STARK server
throughput. They appear in `measurements/*.json` as raw observations, and
`measurements/README.md` states what they are.

**C7 does not wait on a performance instrument.** The gate closes with steady-state runtime
explicitly not measured, and with no runtime claim attached. An honest absence beats a number
produced by a harness already known to be invalid, and building the instrument is follow-on work
rather than gate repair.

### Why P1 is not extended to serve as the benchmark

Raising P1's request count would fuse two purposes into one artefact: semantic and lifecycle
qualification, and performance measurement. The qualification workload's identity would then depend
on benchmark requirements, and its byte-exact 24-exchange corpus is what P1's Tier-1 evidence is
defined against. **P1 stays frozen as the byte-exact functional workload.**

### The follow-on instrument, specified so it is not redesigned from scratch

A *separate, versioned* benchmark that extracts the computational request path rather than driving a
server:

```stark
fn handle_request_bytes(request: &[UInt8]) -> Result<Vec<UInt8>, HttpError>
```

Replay the same 24 frozen request byte sequences in-process, many times. That measures parse → route
→ JSON → response encoding, and excludes process startup, Python, TCP scheduling, deliberate sleeps,
server spawn and client round trips — every term that dominates §7.

Requirements:

- identical source at one commit, built debug and release;
- the same request corpus and iteration count for both;
- enough accumulated work for **at least ~1 second per measured run**;
- **at least five measured runs**, reporting median *and* dispersion;
- low run-to-run variance required before any ratio is called measurable;
- **response hashes verified**, so optimisation cannot delete the work being timed;
- timing code outside the measured loop wherever possible.

Optionally, a second measurement may instrument the real server internally — from receipt of a
complete request to completion of response encoding — kept separate from end-to-end HTTP latency.
