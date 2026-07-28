# WP-C7.4 — baseline optimisations

**Status:** `IMPLEMENTED`. All five permitted optimisations are in place, on by default, with
differential evidence against unoptimised MIR and against native execution.
**Measured at:** this change, macOS arm64, the seven frozen C7 workloads.

---

## 1. The headline measurement, stated before the design

**On every one of the seven frozen workloads, only dead-block elimination fires. Constant folding,
constant propagation and branch folding fire zero times, and the linked binary is byte-identical in
size with and without the optimiser.**

| workload | rvalues folded | checked folded | proven trapping | branches | constants | dead blocks |
| --- | --- | --- | --- | --- | --- | --- |
| w01_minimal | 0 | 0 | 0 | 0 | 0 | 1 |
| w02_arith_control | 0 | 0 | 0 | 0 | 0 | 2 |
| w03_generic_trait | 0 | 0 | 0 | 0 | 0 | 5 |
| w04_string_vec | 0 | 0 | 0 | 0 | 0 | 2 |
| w05_hash | 0 | 0 | 0 | 0 | 0 | 1 |
| w06_multi_package | 0 | 0 | 0 | 0 | 0 | 3 |
| w07_drop_ownership | 0 | 0 | 0 | 0 | 0 | 3 |

`w03_generic_trait`, built both ways: **489,296 bytes either way.**

This is worth stating first rather than burying under the design, because the natural way to write
this document would be to describe five optimisations and let a reader assume they do something.
They are correct and they are exercised — by the C7.4 tests, which construct the constant
expressions the workloads do not contain. Real STARK code does not compute `6 * 7` at
compile time, so the folding passes have nothing to fold.

**What this does and does not justify.** It justifies keeping the passes: they are cheap, they are
proven observationally transparent, and dead-block elimination genuinely shrinks the MIR handed to
the backend on every workload. It does **not** justify any claim of a performance improvement, and
none is made. It also confirms C7.0's finding from the other direction — the compiler is not where
build time goes, so front-end-side optimisation was never going to move the number.

Where folding would matter is code with compile-time-known configuration — a `const`-like pattern, a
feature flag, a size parameter threaded through generics. C7's frozen workloads contain none, and
C7.5's practical systems workloads (P1) are the place to re-measure rather than to speculate.

## 2. What was implemented

`starkc/src/mir/opt.rs`, applied after lowering and **before** verification, so `mir::verify` checks
the program the backend actually receives.

| pass | what it does |
| --- | --- |
| constant propagation | replaces reads of a local proven to hold one constant |
| constant folding | folds `UnOp`/`BinOp` on constants |
| checked folding | resolves `Checked` terminators with constant arguments |
| branch folding | `SwitchInt` on a constant scrutinee becomes `Goto` |
| dead-block elimination | removes blocks with no path from the entry |

Backend-native optimisation settings — the fifth permitted item — landed in WP-C7.1 as the explicit
`[profile.release]` block, and are not re-done here.

Passes iterate to a fixpoint (bounded at 8 rounds) because they feed each other: propagation exposes
folds, folds expose constant branches, folded branches strand blocks.

## 3. The three decisions that constrain the passes

### 3.1 Folding calls the interpreter's own evaluator

`eval_binop`, `eval_unop` and `eval_checked` were changed from methods on the MIR interpreter into
free functions, and the optimiser calls exactly those. A second arithmetic implementation inside an
optimiser is the standard way a compiler comes to disagree with its own interpreter on one edge case
in ten thousand — overflow at a type boundary, a shift-count rule, a cast range. Sharing the code
makes that disagreement structurally impossible, which is why the differential tests can be about
the passes' *reasoning* rather than a re-test of arithmetic.

### 3.2 A folded trap is still a trap

Integer overflow, division by zero, bad shifts and failing casts trap in every build mode. When a
`Checked` terminator's arguments are all constants, the fold takes one of two branches:

- it yields a value → the terminator becomes an assignment plus a `Goto`;
- it traps → the terminator becomes `Trap` **carrying the original `TrapInfo`**.

So a folded trap fires at the same point, with the same category, file, line and column, after the
same preceding statements — including anything they printed. The A5 shift override is applied too:
an out-of-range constant shift reports `InvalidShift`, not the terminator's own category, exactly as
the interpreter does. An optimiser that folded `1 / 0` into a value would delete a required abort;
this one proves the abort instead.

### 3.3 The drop log is observable, so "dead" code mostly is not

STARK's drop log is program output (DROP-ORDER-001, §8.8). A store whose local is later dropped is
not dead, and a block that looks unused but carries a `Drop` cannot be removed. **Dead-store
elimination is therefore deliberately not implemented** — it is not in C7.4's permitted set, and the
analysis it would need is larger than this WP. Dead-*block* elimination removes only blocks with no
path from the entry, which by definition contribute nothing to any observation.

## 4. What is deliberately NOT folded

**Floating-point arithmetic.** The interpreter computes in `f64`; a native backend may compute a
`Float32` expression in `f32`. Folding with the interpreter's answer would bake interpreter rounding
into the binary and make the native result depend on whether an operand happened to be a literal.
Integers have no such freedom — every STARK integer type is a fixed width, there is no pointer-sized
integer, and two's-complement results are identical on every target. `c74_mir_opt.rs` asserts the
float counters are hard zero, so a later change that admits floats has to confront the reason.

**`CheckIndex`.** It yields an opaque index-proof token rather than a value; folding it would break
the proof discipline the verifier enforces. A constant in-bounds index keeps its check, and a
constant out-of-bounds index still traps at run time.

**Anything borrowed or projected.** Propagation applies only to whole locals that are assigned
exactly once, never borrowed, never used as a projection base, and never dropped.

## 5. Evidence

`starkc/tests/c74_mir_opt.rs` — 13 tests. Every case runs the program twice, as lowered and as
optimised, and requires **identical** §39 observations. Two further properties are asserted
throughout, and both close a specific way this suite could rot:

1. **The optimised program still verifies** under `mir::verify`. An optimiser that produced
   ill-formed MIR but happened to interpret correctly would pass an agreement-only test and fail
   later in a backend.
2. **The pass actually fired.** Agreement alone is satisfied by an optimiser that does nothing, so
   each case names the counter that must be non-zero. Given §1's finding — that these passes are
   inert on real workloads — this is the only thing keeping them honest.

Native agreement is checked as well: the optimised program must compile and observe identically once
compiled, not only under the MIR interpreter.

Specific cases worth naming: a folded division by zero keeps its category **and line 4**; output
printed before a folded trap survives; a folded shift keeps the `InvalidShift` override; a constant
condition removes the untaken arm; drop order and count are unchanged; optimising twice changes
nothing (fixpoint); and optimisation is deterministic — which WP-C7.3 depends on, because the build
key is computed from the optimised program, so a nondeterministic pass would give one source two
cache entries.

## 6. The flag

`stark build --no-mir-opt` compiles MIR exactly as lowered. It exists so a suspected divergence can
be bisected against unoptimised MIR — the higher authority under Gate C7 — without rebuilding the
compiler. Because the build key is computed from the optimised program, the two modes occupy
different cache entries and cannot contaminate each other.

The passes are **on by default**. A default-off optimiser is one the corpus never exercises, and the
requirement here is differential evidence against the real corpus, not against a synthetic one.

## 7. Not measured

- Effect on **compile time**. The counters show near-zero work on the frozen workloads, so no effect
  is expected, and none was measured; C7.5 owns compile-time measurement.
- Effect on **steady-state runtime**. The binaries are byte-identical on w03, so there is nothing to
  measure on this corpus. Re-measure when P1's workloads exist.
- Whether folding would fire on **configuration-heavy** code. Stated as a hypothesis in §1, not a
  finding.
