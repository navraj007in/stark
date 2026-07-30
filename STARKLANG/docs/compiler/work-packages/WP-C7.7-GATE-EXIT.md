# WP-C7.7 — Gate C7 exit assessment

**Outcome (superseded):** `CANDIDATE-COMPLETE — BLOCKED BY P1`. Gate C7 does not close.
**Assessed at:** this change, after WP-C7.0 through WP-C7.6.

> **SUPERSEDED — Gate C7 is CLOSED (CD-274).** The final consolidation and closure ruling is
> `../GATE-C7-CLOSURE.md`. This document is retained as the assessment history that led there: it
> is where the gate was first found blocked, where the block was re-diagnosed as qualification
> rather than capability (CD-262), and where the `File` and loop-temporary dispositions were argued.
> Nothing below is edited to match the outcome — the reasoning is the record.

---

## 0. RULING (CD-262): C7 is qualification-blocked, not capability-blocked

**Condition 1 is MET for the admitted C7/P1 scope.** C7's usability test is no longer hypothetical:
a real STARK application compiles from ordinary source, uses environment and TCP host capabilities,
lowers through provider-aware MIR, links native providers, and runs a non-trivial HTTP/JSON workload.
That is stronger evidence of usable native builds than the presence of every standard-library I/O
type.

**Two questions had been conflated**, and separating them is the substance of this ruling:

1. *Can STARK native builds perform useful host I/O?* — **yes**: args, environment, time and TCP
   execute from source.
2. *Does every existing Core I/O abstraction execute natively?* — **no**: `File` does not.

The first is a native-build **usability** criterion; the second is a native-standard-library
**completeness** criterion. Treating them as identical would silently expand C7 from "usable native
build path" into "complete native Core library", which is not what P1 tested. P1 confirms the
distinction: it required TCP and environment access, not filesystem access, and the workload is
implemented.

**Core `File` is therefore a known scoped limitation, recorded rather than blocking:**

> Filesystem operations through the standard-library `File` abstraction are not supported by the
> native source build path in this revision. This is intentional under SELECT-C, not an
> unimplemented provider capability.

### Revised gate state

| condition | status |
| --- | --- |
| native builds usable for admitted workload | **MET** |
| native host capability exists | **MET** |
| P1 implementation | **MET** |
| P1 Tier-1 qualification | **MET** — six execution rows green at `d735b35` (CD-273) |
| C7.5 executable-size dimension | **MET** — 1.686× on P1 (CD-273) |
| C7.5 steady-state runtime | **EXPLICITLY NOT MEASURED** — no claim attached (CD-273) |
| native Core `File` support | **KNOWN LIMITATION / DEFERRED** |
| `DEFECT-C788-LOOP-TEMP` | **DISCHARGED** — fixed by A12 (CD-265); it was an admitted non-blocking deviation under CD-264 |
| **C7 overall** | **OPEN — QUALIFICATION REMAINS** |

### Remaining critical path

Four of the five items are discharged (CD-273); one remains.

| # | item | status |
| --- | --- | --- |
| 1 | Linux x64 P1 run | **DONE** — debug and release, executed, green |
| 2 | Windows x64 P1 run | **DONE** — debug and release, executed, green |
| 3 | C7.5 steady-state runtime measurement | **CLOSED AS NOT MEASURABLE** — no claim attached |
| 4 | C7.5 debug/release comparison | **size MEASURED (1.686×); runtime NOT MEASURABLE** |
| 5 | final evidence consolidation and closure ruling | **OPEN** |

Items 3 and 4's runtime halves are closed by ruling, not by measurement, and deliberately so: the
gate does not wait on a performance instrument. The `1.003×` ratio P1 produced is not evidence that
the profiles perform alike — it is evidence that `functional_run_seconds` times the `e2e.py`
harness. An honest absence of a runtime claim beats a number from a harness known to be invalid.
Building the instrument is follow-on work (`WP-C7.5-PERFORMANCE-REPORT.md` §8), and **P1 stays
frozen at 24 exchanges** rather than being extended to serve it.

### Closure wording

> **C7 is no longer blocked by absence of native host capability.** Native source builds now execute
> environment, clock and TCP operations, and the P1 HTTP/JSON workload demonstrates a useful
> end-to-end native application.
>
> Core `File` remains unsupported on the native source path by explicit SELECT-C decision. This is a
> scoped standard-library limitation, not evidence that native builds are unusable.
>
> Gate C7 remains open for qualification: P1 requires Linux x64 and Windows x64 execution evidence,
> and C7.5 requires the two workload-dependent runtime measurements previously deferred.
>
> Resource lifecycle evidence is complete: nine cases are observed against real providers and one
> is unreachable by construction. It read "eight observed, one blocked" until A12 (CD-265) fixed
> `DEFECT-C788-LOOP-TEMP`, which had been admitted below as a non-blocking deviation.

### DEFECT-C788-LOOP-TEMP — admitted non-blocking deviation (CD-264), since DISCHARGED (CD-265)

> **DEFECT-C788-LOOP-TEMP is admitted as a non-blocking C7 deviation.** A resource-bearing
> match-scrutinee temporary reused across loop iterations is not dropped before reassignment, causing
> the runtime to abort on the second write to the live slot. The runtime detects the compiler
> violation and fails closed; no silent ownership corruption has been observed.
>
> The defect does not invalidate the frozen P1 workload or its qualification because P1 does not
> generate the affected temporary-reuse shape and its user-bound resources close correctly. It
> therefore does not block Gate C7.
>
> The defect is nevertheless a mandatory compiler correction before STARK claims general native
> support for repeated resource-producing expressions or recommends such expressions for application
> use. The classified ignored regression remains committed and must be unignored by the fixing
> change.

Priority: **P1 compiler priority** — high priority, not the P1 workload. Full disposition, fix
boundary and the eight-point investigation scope are in `c78/closure-gate-slice7.md`.

The reason it is admissible rather than blocking is that the compiler **fails closed**: it does not
silently overwrite a live resource and continue, so this is a correctness and availability defect,
not a demonstrated double-close or ownership corruption. Making it blocking would retroactively
widen C7 from *prove the admitted native workload and its required resource surface* to *prove
every valid looping shape involving resource-bearing intermediate values*.

### Final position

> **Close C7 once the remaining cross-platform qualification and C7.5 measurements pass. Carry
> `DEFECT-C788-LOOP-TEMP` as an explicit high-priority deviation, not as a hidden gap and not as a
> C7 blocker.**

**Discharged (CD-265).** The deviation no longer needs carrying: A12
(`mir-amendment-A12-storage-end.md`) fixed the defect, and the regression test came off `#[ignore]`
with the fix. The rulings above are kept as written rather than edited away — they were correct when
made, and the disposition they set is what let the fix be sequenced deliberately instead of in a
scramble.

The fix did correct one thing they recorded. "P1 does not generate the affected temporary-reuse
shape and its user-bound resources close correctly" is true of P1, so the non-blocking verdict
stands — but the general claim behind it was not: a **user local** with one field moved out inside a
loop failed identically, with no `match` in the program. The defect was about any place whose storage
is emptied piecewise, not about temporaries. See `c78/closure-gate-slice7.md`, "What the fix
corrected".

C7's critical path is unchanged: the two cross-platform P1 runs and C7.5's two measurements.

---

## 0.1 Reassessment detail after WP-C7.8 (CD-261)

WP-C7.8 landed after this assessment was written. It changes two of the four verdicts below and
**leaves the gate open**, for a narrower reason than before.

| condition | was | now | why |
| --- | --- | --- | --- |
| native builds usable | PARTIAL | **PARTIAL, narrowed** | Native I/O now exists and executes from ordinary source — args/env, monotonic time, and TCP bind/accept/connect/read/write — through the provider path (`c78/closure-gate-slice7.md`). **Core `File` from source still refuses**, verbatim as §2 records |
| reproducible | MET | MET | unchanged |
| performance bounded by evidence | MET | MET | unchanged; the two deferred dimensions were waiting on P1, which now has a workload |
| P1 complete | NOT MET | **NOT MET, in progress** | A native HTTP/JSON REST workload exists and self-assesses `P1 PARTIAL — Tier-1 cross-platform runs remain` (`WP-C7-P1-REST-REPORT.md`) |

**§2's probe was re-run, not assumed.** A program calling `File::create` still fails with exactly
the error quoted below. That the backend now emits `OwnedResourceHandle` for `MirTy::Core(File, ..)`
does **not** make a source-level `File` program buildable, and inferring otherwise from the emitter
would have been wrong — the refusal is upstream of emission.

**Why `File` is still refused is now a decision, not an omission.** SELECT-C (CD-253) keeps
`CoreType::File` on the legacy `MirTy::Core` path unconditionally, because migrating it would make a
type's MIR identity depend on build configuration. The consequence is deliberate and recorded: `File`
does not participate in the A11 close arena, and no source-level `File` path was built.

**So the shape of the block has changed.** This assessment said P1 "is not waiting to be scheduled;
it is waiting on native capability". That is no longer true — P1's REST workload is built on TCP and
environment lookup and needs no `File`. What remains is qualification, not capability:

1. Tier-1 cross-platform runs for the P1 workload (its own stated caveat; measurements are macOS-only);
2. C7.5's two deferred measurements, which were blocked on P1 existing;
3. a source-level `File` path, **if** the roadmap's "native builds usable" is read as requiring the
   standard library's own I/O type rather than the provider capabilities P1 enumerates. That reading
   is a judgement for the gate owner and is deliberately not made here.

Item 3 is the one that decides whether condition 1 can move to MET. Items 1 and 2 are runs.

---

## 1. The four exit conditions, one verdict each

The roadmap: *"C7 closes when native builds are usable, reproducible to the documented degree,
performance claims are bounded by measured evidence, and P1 has completed so the performance report
includes the practical systems baseline."*

| condition | verdict | basis |
| --- | --- | --- |
| native builds usable | **PARTIAL** | usable for Core-v1 compute; native I/O does not exist — §2 |
| reproducible to the documented degree | **MET** | WP-C7.2, per artefact, profile and platform — §3 |
| performance claims bounded by measured evidence | **MET** | WP-C7.5, including two dimensions declared unmeasurable — §4 |
| P1 complete | **NOT MET** | P1 has not started; §5 |

One blocked condition is sufficient to keep the gate open, and there are two.

## 2. "Usable" is a partial, and this is the finding that matters most

Native builds work, are cached, run in both profiles, and pass a three-platform matrix. What they
cannot do is any I/O beyond writing to stdout.

Probed directly rather than inferred:

```
$ stark build            # a program calling File::open
error: native build does not yet support this program: type Core(File, []) (C4.5)
```

`File` exists in the type checker with `open`, `read_to_string`, `write` and `write_str`, and it
lowers to nothing. There is no `RuntimeFn` for it, no backend support, and no runtime module —
`stark-runtime/src` contains no file, network, time or environment surface at all. The C6.6 surface
audit records **59 of 87** probed standard-library methods as natively executable; the shortfall is
concentrated exactly here.

**None of that is new information.** `COMPILER-STATE.md` already records `File` as EXCLUDED at Gate
C6 closure — "needs a host/provider contract, filesystem error semantics, and a way to compare
environmental observations across engines. Deferred to the I/O gate." The contribution of this
section is not the exclusion but its consequence: *that recorded exclusion is what blocks Gate C7*,
because P1 is built almost entirely out of the surface it defers, and C7.5's remaining measurements
are built on P1.

This matters more than a missing feature list, because **P1's exit criteria are almost entirely made
of it**: arguments and environment, file read/write, monotonic time and sleep, TCP listener and
stream. None of those has a native path today. So P1 is not blocked on someone scheduling it — it is
blocked on native capability that does not exist yet, and building it is a prerequisite to C7.5's
measurements, not a consequence of them.

Stating this plainly is the point of this section. "C7 is done except P1" would suggest C7's own work
is complete and only an unrelated milestone is outstanding. It is not: the native path C7 delivered
cannot yet run the class of program P1 requires.

## 3. Reproducibility — MET

WP-C7.2 classifies per artefact, per profile **and per platform**, with each cell backed by a
measurement and none generalised from another platform:

- generated Rust and `stark.lock`: byte-reproducible;
- generated `Cargo.toml`: semantically reproducible (embeds the compiler's own runtime path);
- release executables: byte-reproducible on macOS and Linux, **not** on Windows;
- debug executables: not reproducible on macOS, reproducible on Linux, unmeasured on Windows.

"To the documented degree" is the condition, and the degree is documented with its exceptions rather
than rounded up to a clean claim. Three separate over-generalisations were caught by CI during C7.2
and corrected (CD-190, CD-191); the per-platform table is what replaced the habit that produced
them.

Outstanding within C7.2 and carried rather than hidden: the machine-readable build manifest (§4.6)
and cross-machine reproducibility comparison (§4.3). Neither blocks this condition, which is about
documenting the degree accurately, not about maximising it.

## 4. Performance claims bounded — MET

WP-C7.5 measures six of eight required dimensions and declares two **not measurable on this corpus**
rather than estimating them. The single ratio it reports (~115× interpreter over native debug) is
labelled as one workload, against debug, with its front-end overhead accounted for. The harness's
own raw debug/release runtime ratios are contradicted in the report on purpose, so a reader of the
JSON does not mistake startup noise for a finding.

No general performance multiple is claimed. That is the condition, and it is met — the claims are
bounded, and where evidence does not exist the report says so instead of filling the gap.

WP-C7.6 records DEFER on LLVM with CE6 unopened, on the same evidentiary basis, and separates
"should STARK use LLVM" (deferred) from "should STARK have a direct backend" (open, and motivated by
self-containment rather than by any performance measurement).

## 5. What C7 delivered

| WP | outcome |
| --- | --- |
| C7.0 | baseline: host Cargo/rustc is 65–68 % of a cold build |
| C7.1 | `--release` and `--target`, profile-aware layout, target preflight |
| C7.2 | path remapping; reproducibility classified per artefact, profile and platform |
| C7.3 | bounded build cache, size-capped LRU, 2.0× median rebuild |
| C7.4 | baseline MIR optimisations — measured to fire zero times on real workloads |
| C7.5 | performance report; two dimensions declared unmeasurable |
| C7.6 | DEFER LLVM; CE6 unopened |

Two of these produced findings that constrain what may be claimed rather than expanding it — C7.4's
inertness and C7.5's unmeasurable dimensions. Both are recorded as findings, not as failures.

## 6. Recommended disposition

**Keep Gate C7 OPEN as `CANDIDATE-COMPLETE-BLOCKED-BY-P1`.**

The work that unblocks it is not further C7 work. It is native runtime capability — file I/O,
environment, time, sockets — which P1 requires and which C7's measurements cannot proceed without.
Whether that is scoped as P1 itself or as a preceding native-capability work package is an owner
decision; this document does not assume one.

> **RESOLVED — owner decision, 2026-07-28 (CD-201).** Native host capability work is scoped as a
> preceding work package, **WP-C7.8**, rather than being absorbed implicitly into P1. P1 remains
> responsible for package-level JSON, HTTP/1.1 routing, and three REST endpoints after the native
> provider foundation is available.
>
> This resolves the open question **without changing the C7 exit verdict**: Gate C7 remains
> `CANDIDATE-COMPLETE-BLOCKED-BY-P1`, and WP-C7.8 does not close it. See
> `work-packages/WP-C7.8-First-Party-Native-Host-Capabilities.md` and `COMPILER-ROADMAP.md` §4.1.

When P1 exists, three things re-open here rather than being taken as settled:

1. **WP-C7.5** — steady-state runtime, the debug/release runtime ratio, and a defensible
   interpreter/native ratio become measurable for the first time.
2. **WP-C7.4** — whether the folding passes ever fire on realistic code.
3. **WP-C7.6** — whether a generated-code deficit appears that would justify opening CE6.
