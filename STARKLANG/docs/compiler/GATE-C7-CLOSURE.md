# Gate C7 — Final Evidence Consolidation and Closure Ruling

**Decision:** `CLOSED`
**Owner ruling:** Final (CD-274)
**Gate:** C7 — Native Build, Qualification and First-Party Host Capability
**P1 execution evidence commit:** `d735b35dbb2148cae5c61dc17376adc8cb50be55`
**Gate-qualifying commit:** `c5a97bfd918a3af1e293a4b5d0114d0ea8cbf084` (`c5a97bf`)
**Prior rulings retained:** CD-183, CD-195, CD-264, CD-265, CD-268, CD-269, CD-270–CD-273

---

## 1. Ruling

> **Gate C7 is CLOSED.**

C7 establishes a usable and qualified generated-Rust native build path for the admitted workload,
including first-party native host capabilities required by P1, deterministic MIR-owned resource
lifecycle, debug and release execution, Tier-1 cross-platform qualification, build reproducibility
classifications, retained build caching, and bounded performance evidence.

The gate closes **without a steady-state runtime-performance claim**.

The absence of that claim is deliberate. Existing runtime measurements are dominated by either
process-startup cost or the P1 end-to-end test harness and therefore do not measure sustained STARK
execution. C7 does not manufacture a performance number from an instrument known to be unsuitable.

---

## 2. Qualification commit rule

Two commits have distinct evidentiary roles, and conflating them would repeat exactly the error this
gate spent its last weeks correcting.

- **`d735b35` qualifies the P1 execution matrix** — Linux x64, macOS arm64 and Windows x64, each in
  debug and release.
- **`c5a97bf` qualifies the complete C7 tree** — formatting, static analysis, compiler tests, C6.4
  qualification, C7 qualification, P1 execution, and the associated record checks. Confirmed green
  across **18 jobs in both workflows** (`CI` and `C7.8 Native Capabilities`) at that exact commit,
  not inferred from a later or earlier run.

`d735b35` must **not** be described as the overall C7 qualifying commit unless the complete required
CI set is independently confirmed green at that exact commit. It is cited here for the six execution
rows and nothing wider.

---

## 3. Exit-condition disposition

| C7 exit condition | Final status | Evidence and boundary |
| --- | --- | --- |
| Native builds are usable | **MET** | Ordinary STARK source builds and executes through the selected generated-Rust path in debug and release |
| Build reproducibility is classified | **MET** | Reproducibility is stated per artefact, profile and platform rather than as one universal claim |
| Generated build retention/cache behaviour is implemented | **MET** | Unchanged generated work is retained and reused without unbounded accumulation |
| Native safety behaviour is preserved | **MET** | Checked arithmetic, panic-abort configuration, verified MIR and generated-code invariants remain active |
| First-party host capability exists | **MET** | Source-to-provider lowering, validated provider metadata, static provider linkage and native execution are implemented |
| P1 implementation exists | **MET** | Native HTTP/JSON REST workload implemented |
| P1 executes on every Tier-1 platform | **MET** | Six debug/release execution rows green |
| P1 response behaviour is qualified | **MET** | Each row executes 24 raw HTTP exchanges compared byte-for-byte |
| P1 terminates cleanly | **MET** | Bounded clean exit after the frozen exchange sequence |
| Pure-STARK P1 tests | **MET** | 7/7 on Linux, macOS and Windows |
| Native resource lifecycle | **MET** | Nine observable cases qualified; one case unreachable by construction |
| `DEFECT-C788-LOOP-TEMP` | **DISCHARGED** | MIR amendment A12 and expanded storage-end shape qualification |
| C7.5 executable-size measurement | **MET** | Debug 860,784 bytes; release 510,592 bytes; ratio 1.686× on P1 |
| C7.5 steady-state runtime | **CLOSED AS NOT MEASURABLE** | No backend runtime claim attached |
| Native Core `File` surface | **KNOWN LIMITATION / DEFERRED** | Not required by the admitted P1 workload; retained under SELECT-C |
| Full Core/native conformance | **NOT CLAIMED** | C7 qualifies the admitted native workload and capability surface, not every normative Core program |

---

## 4. P1 Tier-1 execution matrix

| Platform | Debug | Release |
| --- | ---: | ---: |
| Linux x64 | **PASS** | **PASS** |
| macOS arm64 | **PASS** | **PASS** |
| Windows x64 | **PASS** | **PASS** |

A passing row means all six of:

1. the relevant profile was built;
2. **that artefact was executed**;
3. the pure-STARK suite completed 7/7;
4. 24 raw HTTP exchanges were performed;
5. every response was compared byte-for-byte;
6. the server completed a bounded clean exit.

**A successful build without execution does not satisfy this matrix.** That distinction is not
pedantry: `stark-json`'s native evidence had been recorded on a successful build whose binary had
never been run, and when it finally was, it aborted immediately on a real compiler defect.

P1 remains **frozen at 24 exchanges**. It is a functional and lifecycle qualification workload, not
a performance benchmark.

---

## 5. Native-host capability established

C7 began with a native path capable primarily of computation and console output. C7.8 extended it
with a provider model reachable from ordinary STARK source.

The admitted capability surface now includes what P1 requires: process arguments; environment
lookup; monotonic time; TCP listener bind; TCP accept; outbound TCP connect; stream read; stream
write; move-only listener and stream resources; MIR-owned exactly-once release; declared recoverable
statuses; fatal provider-contract-violation handling; and debug and release native linkage.

**This is not a general FFI claim.** It is a validated first-party provider mechanism with explicit
resource identity, capability selection and ABI contracts.

---

## 6. Resource lifecycle ruling

The resource model is retained as designed:

- provider resources are move-only;
- they are neither `Copy` nor `Clone`;
- Rust `Drop` does not own provider release;
- MIR determines the unique close operation;
- failed resource creation produces no live resource;
- moving a resource transfers its close obligation;
- early return and `?` propagation preserve cleanup;
- listener and stream handles close through their own declared release functions;
- repeated resource-producing operations may reuse storage only after MIR has accounted for the
  prior value.

The lifecycle matrix is complete for the admitted provider surface:

```text
9 observable cases qualified
1 case unreachable by construction
0 ignored cases
```

The earlier loop-temporary defect is **not** carried as an accepted deviation. It was fixed through
A12, including the wider user-local and `?`-inside-loop shapes discovered afterwards — the second of
which was found by requalifying a package rather than by extending a test matrix.

---

## 7. Performance disposition

### 7.1 Measurements that support a claim

```text
debug:    860,784 bytes
release:  510,592 bytes
ratio:    1.686×
```

C7 may claim that the measured release executable is materially smaller than the corresponding debug
executable for P1.

### 7.2 Measurements that do not support a claim

```text
Micro-workloads:
    dominated by native process-startup floor.

P1 end-to-end execution:
    dominated by Python startup, a deliberate 10 ms delay,
    process supervision, server startup and loopback exchanges.
```

Consequently:

```text
Backend steady-state runtime claim:
    NONE

Debug/release runtime ratio:
    NOT MEASURABLE

P1 server throughput:
    NOT MEASURED
```

The recorded `321 req/s`, approximately `66 ms`, and the `1.003×` profile ratio **describe the
harness observation**. They must not be quoted as STARK server throughput, request latency or
backend profile performance.

### 7.3 Follow-on instrument

Performance measurement is **follow-on work, not gate repair**. The specified instrument will
extract `handle_request_bytes`; replay the frozen 24-request corpus in-process; accumulate at least
approximately one second of measured work; run at least five measured trials; report median and
dispersion; require sufficiently low variance before reporting a profile ratio; verify response
hashes so optimisation cannot remove the measured computation; and remain versioned separately from
P1. Full specification: `work-packages/WP-C7.5-PERFORMANCE-REPORT.md` §8.

---

## 8. Reproducibility and build behaviour

C7's reproducibility claim remains **artefact- and platform-specific**. The gate does not claim that
every debug executable is byte-identical across every host and path; it records where reproducibility
was demonstrated and where platform toolchain metadata prevents it or has not yet demonstrated it.

The native build system also retains the generated content-addressed workspace instead of deleting it
after each default build. An unchanged rebuild reuses that workspace, and tests ensure generated
directories do not accumulate without bound.

---

## 9. Known limitations that do not block closure

**Native Core `File`.** The normative Core `File` surface is not yet generally reachable through the
selected source-level native representation. P1 does not depend on it. File/package work proceeds
separately.

**General network stack.** C7 qualifies the bounded synchronous TCP capability P1 needs. It does not
claim TLS, HTTP/2, HTTP/3, UDP, asynchronous I/O, an event loop, DNS as a general package contract,
or unrestricted native FFI.

**Runtime-performance claims.** None made.

**Universal language conformance.** C7 does not claim that every accepted Core program lowers and
executes natively. Language-surface completion and adversarial three-engine corrections remain
separately governed (WP-C7.9).

**Future usage-shape qualification.** Risk-based qualification for reference-returning and
borrow-retaining APIs remains separate work and is **not** retroactively absorbed into C7.

---

## 10. Evidence discipline retained

C7's most important methodological outcome is that evidence is classified by what it actually proves:

```text
Build success:
    proves construction and linkage.

Native execution:
    proves the generated artefact starts and runs the tested path.

Byte-exact workload comparison:
    proves tested external behaviour.

Lifecycle instrumentation:
    proves declared ownership and close events.

Three-engine comparison:
    applies only where all three engines execute the same semantic surface.

Provider-backed capability:
    verifier + ABI + native platform evidence, unless an interpreter provider exists.
```

A build is not execution evidence. A green component test is not whole-path evidence.
Cross-platform support is not inferred from one host. A runtime number is not a backend-performance
result when fixed harness costs dominate it.

Each of those four sentences was learned from a specific failure in this gate, not asserted as
principle: a package whose native evidence was a build; a close-emission defect that every unit test
passed over; Tier-1 rows that had never left one machine; and a `1.003×` ratio that measured Python
startup.

---

## 11. Final supported claim

> STARK has a usable generated-Rust native build path for its admitted workload. It builds and
> executes in debug and release on Linux x64, macOS arm64 and Windows x64; supports the first-party
> process, time and synchronous TCP capabilities required by the frozen P1 HTTP/JSON REST workload;
> preserves MIR-owned move-only resource lifecycle; and passes six Tier-1 P1 execution rows
> consisting of byte-exact HTTP exchanges and bounded clean exit. Executable-size profile effects are
> measured. No steady-state runtime, throughput, complete Core-library, unrestricted host-I/O or
> universal native-conformance claim is made.

---

## 12. Gate transition

With C7 closed:

- C7 no longer blocks subsequent roadmap work;
- the separate performance instrument remains follow-on work;
- `stark-io` and additional host packages proceed through their own provider/package qualification;
- three-engine adversarial corrections remain governed by WP-C7.9;
- usage-shape qualification remains independently owned;
- future native capability claims must retain the evidence distinctions established here.

**Final state:**

```text
GATE C7: CLOSED
P1: TIER-1 QUALIFIED
C7.5 SIZE: MEASURED
C7.5 RUNTIME: NOT MEASURABLE — NO CLAIM
NATIVE PATH: USABLE FOR THE ADMITTED WORKLOAD
FULL CORE/NATIVE CONFORMANCE: NOT CLAIMED
```
