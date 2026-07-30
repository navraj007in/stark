# WP-C7.7 — Gate C7 exit assessment

**Outcome:** `CANDIDATE-COMPLETE — BLOCKED BY P1`. Gate C7 does not close.
**Assessed at:** this change, after WP-C7.0 through WP-C7.6.

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
| P1 Tier-1 qualification | **PARTIAL** |
| C7.5 deferred measurements | **OPEN** |
| native Core `File` support | **KNOWN LIMITATION / DEFERRED** |
| **C7 overall** | **OPEN — QUALIFICATION REMAINS** |

### Remaining critical path

1. Linux x64 P1 run;
2. Windows x64 P1 run;
3. C7.5 steady-state runtime measurement;
4. C7.5 debug/release runtime comparison;
5. final evidence consolidation and closure ruling.

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
> Resource lifecycle evidence is substantially complete: seven cases are observed against real
> providers, one is unreachable by construction, and two remain defined but unobserved — `?`
> propagation with a live resource and repeated connect/release.

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
