# WP-C7.7 — Gate C7 exit assessment

**Outcome:** `CANDIDATE-COMPLETE — BLOCKED BY P1`. Gate C7 does not close.
**Assessed at:** this change, after WP-C7.0 through WP-C7.6.

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

When P1 exists, three things re-open here rather than being taken as settled:

1. **WP-C7.5** — steady-state runtime, the debug/release runtime ratio, and a defensible
   interpreter/native ratio become measurable for the first time.
2. **WP-C7.4** — whether the folding passes ever fire on realistic code.
3. **WP-C7.6** — whether a generated-code deficit appears that would justify opening CE6.
