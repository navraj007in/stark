# C7 P1 measurements

`debug.json` and `release.json`, produced by `scripts/measure.py --profile <p>` at compiler commit
`cd9405b` on macOS arm64. They replace `latest.json`, which recorded a single debug run at `a6964d9`
— a commit predating `MIR_VERSION` 0.2 → 0.3, so its build keys and its numbers describe a compiler
that no longer exists.

## What these numbers support, and what they do not

| C7.5 dimension | this workload |
| --- | --- |
| executable size, debug vs release | **measured** — 860,784 vs 510,592 bytes, **1.686×** |
| cold and warm build time | **measured** |
| **steady-state runtime** | **still not measured — see below** |
| **debug/release runtime ratio** | **still not measured — see below** |

## Why the runtime dimensions are still open

`functional_run_seconds` times the whole `e2e.py` subprocess. **It measures the harness, not the
workload**, and the measurement says so itself:

```
debug   run_median = 66.72 ms
release run_median = 66.50 ms   →   ratio 1.003×
```

A 1.003× debug/release ratio is not a finding about optimisation. It is what you get when the thing
being timed is not the thing you meant. C7.5 measured 1.5× between the profiles on *startup alone*,
and ~115× between the interpreter and native — so a workload genuinely dominated by STARK compute
could not come out flat.

Decomposing the ~66 ms:

- ~15.7 ms — Python interpreter startup (measured, `python -c pass`);
- ≥10 ms — a deliberate `time.sleep(0.01)` in the split-write case;
- the server process spawn;
- 24 sequential loopback connect/send/recv round trips;
- and, somewhere inside that, the STARK request handling.

Parsing 24 small HTTP requests is microseconds of compute. It is below the resolution of a harness
whose floor is tens of milliseconds, which is the *same* conclusion C7.5 reached on the
micro-workloads — reached again for a different reason, at a larger workload.

The first sample of every run is 6–7× the rest (464 ms debug, 417 ms release) and is filesystem and
loader warm-up. The median is used, so it does not distort the figure; it is noted because a mean
would have.

## What a real steady-state measurement would need

1. **Many more requests.** The workload has a *frozen 24-accept lifecycle*, so raising the count is
   a change to the workload, not to the harness — an owner decision, since P1's byte-exact
   qualification is defined against those 24 exchanges.
2. **Timing that excludes process startup** — measure inside the server, or amortise across enough
   requests that a fixed ~30 ms floor stops mattering.
3. **A load shape with real work in it.** Even unbounded requests against these fixtures may stay
   below resolution; the handler does bounded parsing of small inputs by design.

Until those exist, "steady-state runtime" and "debug/release runtime ratio" stay `NOT MEASURABLE`,
now with a second workload's evidence behind the verdict rather than one.
