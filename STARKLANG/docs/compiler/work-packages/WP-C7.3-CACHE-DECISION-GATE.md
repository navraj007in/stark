# WP-C7.3 — cache decision gate

**Status:** `GATE DATA COMPLETE — implementation not started`, per §5.1's requirement that profiling
precede any cache.
**Measured at:** `95dd492`, macOS arm64, seven frozen workloads, median of three runs.

---

## 1. The gate data §5.1 asks for

| measurement | value |
| --- | --- |
| cold build, total | 0.41–0.97 s |
| host Cargo/rustc share of a cold build | **65–68 %** (C7.0, CD-185) |
| STARK compiler share | 32–35 % |
| rebuild with no source change, current default | 0.39–0.47 s — **essentially the cold cost** |
| rebuild with the generated crate preserved | **0.18–0.23 s** |
| speedup from preserving it | **2.1× median** |
| disk cost | ~7–12 MB per package, per distinct source version |

## 2. The finding that decides the boundary

**There is no build cache today, and the reason is a deletion.** The generated crate directory is
content-addressed — its key already covers source content, compiler/MIR/runtime/backend versions,
target and (since WP-C7.1) profile — and it is **deleted after every build** unless
`--keep-generated` is passed. So every build recompiles the generated crate *and* `stark-runtime`
from scratch, and a no-change rebuild costs almost exactly what a cold build costs.

Preserving it turns the existing mechanism into a working cache:

- a no-change rebuild drops to 0.19 s (2.1×);
- an edit produces a new key and a cold build, correctly;
- **returning to a previously-built source version hits the cache** — measured at 0.19 s after five
  intervening edits. That is content-addressed behaviour, not timestamp behaviour.

## 3. What this rules out

§5.1 warns: *"If host compilation dominates, do not pretend a front-end-only cache solves total build
latency."* It does dominate, at 65–68 %.

A **perfect** front-end cache — parse, resolve, typecheck, MIR, emission all free — would cap at the
32–35 % that is STARK's share, and would need new keys, new invalidation rules and new failure
modes. Preserving the generated crate delivers 52 % with no new cache infrastructure at all, because
Cargo already performs the invalidation and already gets it right.

**A demand-driven incremental query engine is not justified by any measurement here** (§5.2). The
edit-rebuild cost is 0.4 s on these workloads; the query engine would target a fraction of that.

## 4. The correctness risk, measured

**Unbounded disk growth.** Five successive edits left five crate directories and 34 MB. Each
distinct source version costs ~7 MB and nothing removes it. This is the reason the deletion exists,
and it is the real work in C7.3 — not the caching, which already functions, but the eviction that
was never needed while everything was deleted immediately.

The safety properties §5.5 requires are mostly already met by the existing design: the key is
content-addressed rather than mtime-based; an incompatible compiler/MIR/runtime version produces a
different key rather than a stale hit; and `--keep-generated` already exists as the disable switch
for qualification. What is missing is eviction and an explicit cache-clear command.

## 5. Recommendation

Implement the smallest cache that the evidence supports:

1. **stop deleting the generated crate by default**, making the existing content-addressed directory
   the cache;
2. **add eviction** — a bounded number of versions or total size per package, evicting
   least-recently-used, since the leak is what the deletion was protecting against;
3. **add `stark clean`** (or equivalent) and keep a documented way to disable the cache for
   qualification runs;
4. **do not build a front-end or MIR cache**, and do not build a query engine, until a measurement
   shows the remaining 32–35 % is the binding constraint.

**Open for owner decision before implementation:** the eviction policy is user-visible — a size cap,
a version count, or an age bound — and it trades disk against rebuild latency for older versions.
The measurements above support any of the three; the choice is a product decision rather than one
the profiling settles.

## 6. Not yet measured

- Cache behaviour across a **dependency** change in a multi-package graph (§5.4) — `w06` is the
  workload for it.
- Public-interface versus private-implementation change: whether a private edit in `lib` should
  rebuild `app` at all. The current key is whole-source, so it does.
- Peak memory (§5.7).
