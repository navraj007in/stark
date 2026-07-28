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
the profiling settles. **Resolved — see §7.** The owner chose a size cap with an age bound as
secondary hygiene, and the implementation landed as CD-189.

## 6. Not yet measured

- Cache behaviour across a **dependency** change in a multi-package graph (§5.4) — `w06` is the
  workload for it.
- Public-interface versus private-implementation change: whether a private edit in `lib` should
  rebuild `app` at all. The current key is whole-source, so it does.
- Peak memory (§5.7).

## 7. Implementation outcome (CD-189)

**Owner decision:** size-capped LRU. 2 GB per cache root, 30-day age bound as secondary hygiene,
retained by default, eviction after a successful build, current and pinned entries never evicted.

**The closure claim, as scoped by the owner and not to be restated more strongly:**

> the implementation reuses complete content-addressed generated crates and Cargo artefacts; it is
> a bounded build cache, not fine-grained incremental compilation.

Nothing in it models functions, packages or interfaces. A one-character source edit changes the key
and produces a cold build. That is the intended ceiling, not a defect to be fixed later without a
measurement first — §5.4 still stands.

**Measured:** 2.0× median cold→cached across the seven frozen workloads (0.36–0.50 s cold,
0.18–0.26 s cached); `w07` 0.44 s → 0.20 s. End-to-end on a fresh package: 1.07 s → 0.19 s, with
`stark cache status` reporting 8.2 MB against the 2147 MB cap.

**Two deliberate divergences from §5's wording, recorded so the trail does not read as drift:**

1. §5.3 proposed `stark clean` "(or equivalent)". It landed as `stark cache status|clean` — one
   command with two verbs, because the status view is what makes a size cap legible to a user, and
   two top-level commands would widen the CLI further than the requirement needs.
2. The 2 GB cap applies **per cache root** — per package and per profile — because that is where
   the content-addressed entries already live. A machine-wide cap would have meant relocating the
   cache, which is a larger change than the evidence asks for. Recorded as an interpretation rather
   than left implicit, since "2 GB" reads as global until you ask.

**Disable path for qualification:** `stark build --no-build-cache`, which also removes the entry
afterwards, so a qualification run leaves no residue. `--keep-generated` and `--emit-rust` instead
**pin** an entry, which is what keeps user-requested generated source separate from ordinary
evictable cache.

**Robustness properties, each with a test:** metadata is written through a temporary and renamed, so
an interrupted build leaves the old file or the new one and never a truncated one; missing or
corrupt metadata makes an entry oldest, so it becomes the first eviction candidate rather than
breaking the sweep; the sweep takes an advisory whole-root lock and **skips** rather than fails if
it cannot acquire it, because no build should fail because a cache could not be tidied.

**Tests:** `starkc/tests/c73_build_cache.rs`, ten cases. Two are worth naming because they pin
absences rather than behaviour — an edit must produce a *new* entry (a future change that quietly
introduced interface-level reuse would have to update that test deliberately), and returning to an
earlier source version must *reuse* its entry rather than create a fresh one, which is the property
that distinguishes content-addressing from "keep the last build".

**Limitations carried forward as explicitly unmeasured, per §6 and unchanged by this
implementation:** multi-package dependency invalidation (whether a private edit in a dependency
should rebuild its dependents — the key is whole-source, so it does) and peak memory. Neither is
claimed to be bounded; both are stated as untested.
