# C10-Q — reproduction of every population-A deviation

**Anchor:** `3a53473`. Every commit between `076b4dc` (the C10-Q evidence anchor) and `3a53473`
touches documentation only — verified with `git diff --name-only 076b4dc..HEAD -- starkc/src
starkc/tests packages STARKLANG/tests`, which is empty. A reproduction at `3a53473` is therefore a
reproduction at the anchor.

**Why this pass exists.** C10-Q's recommendation was made conditional on it. A claim naming N
deviations is only as good as its list, and naming a deviation that no longer exists is its own
false claim. Three entries had already been found not to reproduce; the base rate justified
checking the rest rather than trusting them.

**Population and denominator, declared before measurement.** Population A as computed by
`starkc/scripts/c10-deviation-populations.py` — the canonical enumerator, which applies the
append-only "last heading wins" rule. It stood at **21**. DEV-172 was reproduced before this pass,
so the target population here is the remaining **20**. No entry was added to or removed from that
list after seeing a result.

---

## Result

| | count |
| --- | --- |
| reproduce | 14 |
| do not reproduce | 5 |
| not settled by this method | 1 |
| **target population** | **20** |

Adding DEV-172 (reproduced earlier): **15 of 21 reproduce.**

Population A after this pass: **16** — the 15 reproduced, plus DEV-159, which is counted as open
because this pass could not settle it, not because it was confirmed.

### Reproduce (14)

| DEV | how it was reproduced | observed |
| --- | --- | --- |
| DEV-120 | native vs interpreter, runaway recursion | native `fatal runtime error: stack overflow, aborting`; interpreter a classified diagnostic at the recursive call — the documented divergence exactly |
| DEV-140 | `tests/layer_audit.rs` probe `L7153` | registered `KnownDev`, actual `KnownDev` |
| DEV-141 | probe `L8093` | as registered |
| DEV-142 | probe `L9130` | as registered |
| DEV-143 | probe `L5346` | as registered |
| DEV-144 | probe `L3698` | as registered |
| DEV-145 | probe `L6450` | as registered |
| DEV-156 | `stark fmt` on a struct with field doc comments | both field docs evicted from the struct body and left orphaned before `fn main()` — worse than the entry's "relocated after the struct" |
| DEV-157 | native build, `let x: Int32 = panic("p")` and `f(panic("p"))` | `MirTy Never has no C5.3a generated-Rust representation yet` — verbatim the entry's claim |
| DEV-160 | native build of the shape `stark-http-client` works around | interpreter runs it; native emits `E0502: cannot borrow _1 as mutable` inside `mod stark_proj` |
| DEV-167 | `starkc check` on a `T: Display` receiver | `[E0302] method 'to_string' not found for type 'T'` |
| DEV-168 | native build of `Display::fmt(&p)` | `callee form (C4.5)`; runs correctly under the HIR oracle |
| DEV-180 | source inspection, `interp.rs` ~2542 | take/write-back intact: "the caller's place **was emptied** to make the binding" |
| DEV-186 | source inspection, `src/lsp/server.rs:60-63` | byte-for-byte the code the entry cites; `vec![0u8; content_length]` still unbounded |

### Do not reproduce (5)

| DEV | what the entry predicted | what happens now | why |
| --- | --- | --- | --- |
| DEV-083 | `E0302 method 'tag' not found for type 'Pair<Option<_infer_4>, _infer_5>'` | compiles, prints 42 | method resolution was rewritten by `5b5edd3` — "AS3 Boundary 4 EXIT: `find_method` and `find_impl_fn` no longer exist". The one-way match the entry describes is gone. **Incidental**: no commit names DEV-083 |
| DEV-122 | a fault in one file reported against another | fault in `src/helper.stark` reported at `src/helper.stark:3:5` in both engines | `Span` now carries `pub source: SourceId` (`source.rs:19`). This is the entry's own "platform correction… filed as a separate future WP"; it landed under AS1b. **Incidental** |
| DEV-161 | an ambient `CARGO_TARGET_DIR` breaks every native build | builds and runs; the hijack directory is never created | the builder passes `--target-dir` explicitly, documented at `backend/generated_rust/build.rs:104-119` |
| DEV-162 | a read through a whole-value accessor fails | partial move plus live-sibling read runs correctly | `COMPILER-STATE.md:2176` already records `DEV-162 … CLOSED (CD-372)`. The ledger and the state file **contradicted each other** |
| DEV-178 | generic context lost for associated-function calls and function values | `Holder::tsize(1)` → 4, `Holder::tsize(true)` → 1 | `b39c49d` — "DEV-178: a function value carries the instantiation it was created with". The `size_of` result **discriminates** `Int32` from `Bool`, so the environment is genuinely present |

### Not settled by this method (1)

**DEV-159** — a native build racing its own dependency build. Non-deterministic; a single build that
succeeds does not falsify a race, and no fix commit exists. Recorded as evidence in neither
direction and counted as open. Settling it needs repeated cold builds of an HTTPS program, which is
a distinct exercise.

---

## Two probes that were wrong before they were right

Recorded because the method matters more than the tally.

**DEV-157 was very nearly filed as "does not reproduce".** The entry names one shape — `Err(_) =>
panic(..)` in match-arm value position — and that shape now builds and runs correctly. Only a
robustness pass across other `Never` positions found the defect alive and unchanged. A one-shape
probe would have produced a false closure in a release claim. Every "does not reproduce" call above
was re-tested with variants for this reason; DEV-083 was checked against three receiver shapes
including `Vec::new()` and a bare `None`.

**DEV-160's first two probes hit the wrong sub-shape.** CD-374 split it into a/b/c/d and closed only
`a`. Both early probes landed on `a` and passed. The reproducer that works came from reading the
workaround comment in `packages/stark-http-client/src/lib.stark:1215`, which describes `b` precisely.

## The layer audit is a standing reproduction check, and it was verified as one

The six layer defects are not reproduced by hand. `tests/layer_audit.rs` compares each probe's
registered disposition against its actual one and **fails in either direction** — a new defect, and
equally a registered one that stopped reproducing (`layer_audit.rs:292-299`).

Per the AS8 rule, a passing suite does not establish the claim, so the check was made to fail on
purpose: re-registering `L7153` as `Expect::Lowers` produced

```
  - L7153 Vec:: method outside the implemented set
      registered as Lowers but actually KnownDev("")
```

which demonstrates the comparison is real and that the refusal is genuinely still present. The
mutation was reverted; `git diff` on the file is empty.

## What DEV-160 shows beyond reproducing

CD-374 records b/c/d as "refused by name before rustc, which is the correct outcome… without it they
reach the user as `E0502` inside `mod stark_proj` — a correct compiler error about code they never
wrote." The reproducer here is **not** refused by name. It reaches rustc and produces exactly that
outcome:

```
error[E0502]: cannot borrow `_1` as mutable because it is also borrowed as immutable
  --> src/main.rs:75:28
70 |     _8 = (&(*stark_proj::stark_ref_23struct_230_23f0(&_1)));
```

The named-refusal boundary does not cover this shape. That is a finding about the *quality* of the
deviation's containment, not a new defect, and it is recorded rather than acted on.

## Ledger hygiene — three distinct causes, not one

Across C10-Q's verification, eight population-A entries did not reproduce. They failed in three
different ways, and only the first is the one originally diagnosed:

1. **A named repair, with the ledger never updated.** DEV-005, DEV-177, DEV-181, DEV-178. The commit
   message names the DEV number. A ledger-vs-git reconciliation would catch all four.
2. **An incidental repair by an unrelated consolidation.** DEV-083 (AS3 method resolution),
   DEV-122 (AS1b span identity), DEV-161. No commit names the deviation, because fixing it was not
   the point of the work. **No reconciliation against commit messages can catch these** — only
   re-running the reproducer can.
3. **Two sources of record contradicting each other.** DEV-162 is `CLOSED (CD-372)` in
   `COMPILER-STATE.md` and was backfilled as OPEN in the ledger on the same day.

Cause 2 is the one with teeth: it means the ledger cannot be trusted to a git-log audit, and a
periodic reproduction pass is the only mechanism that keeps it honest.

## Residuals recorded, not closed

**DEV-122's clamp survives its closure.** `SourceFile::line_col` still does
`offset.min(self.src.len() as u32)`, and compile-time and runtime rendering are still separate
paths. What made that clamp dangerous — a span resolvable against the wrong file — is now prevented
structurally by `SourceId` on every span, which is why the entry is closed. The hardening the entry
also asked for (`start <= end`, a column within its line, one shared `resolve_span`) has **not**
been done. `Span::in_source` carries a `debug_assert!(lo <= hi)`, which is not a release check.
Whether that residual deserves its own number is an owner call; it is not tracked by DEV-122 any
more.
