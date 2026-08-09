# C10-B — Robustness qualification

**Packet:** C10-B, WP-C10.2. **Date:** 2026-08-09.
**Baseline:** `f12ececca6d4bdabf828d657c4a4f719a7f9c39a`.
**Suite:** `starkc/tests/c10b_robustness.rs` (12 tests). **Repro tools:** `examples/c10b_repro.rs`,
`examples/c10b_thread.rs`.

**Verdict: the gate FAILS on one target, and passes on the rest. One new deviation — DEV-214.**

---

# 1. The gate, restated before the result

```text
no panic          including no unreachable!(), no compiler-side arithmetic overflow, no unwrap
                  on generated input
no hang           every case under a wall-clock bound; a timeout is a finding
bounded failure   a diagnostic or a clean error, never an unbounded allocation
deterministic     the same seed produces byte-identical diagnostics
```

**Not a claim that random programs are semantically meaningful.** A generated program that is
rejected is a pass, provided it is rejected the same way twice.

**Not "fuzzing".** Charter §1.10 rule 8 forbids nightly Rust and libFuzzer requires it. This extends
the deterministic seeded-generator pattern `tests/robustness.rs` established. **The release wording
must say *bounded deterministic robustness testing*.**

---

# 2. Target population — declared in C10-0 §8 before any of it ran

| Target | Status | Where |
| --- | --- | --- |
| T1 lexer/parser | **PASS** — 800 cases, whole front end (`robustness.rs` drove the parser alone) | `t1_random_token_soup…`, `t1_random_character_soup…` |
| T2 malformed-source corpus | **PASS** — 26 named hostile shapes | `t2_malformed_source_corpus_fails_boundedly` |
| T3 resolver/package/module graphs | **NOT RUN** — see §5 | — |
| T4 type checker | **PASS** — 500 generated programs, ≥50 reaching a semantic diagnostic (asserted, so the target cannot go vacuous) | `t4_t5_generated_programs…` |
| T5 borrow checker | **PASS** — same generator, ownership-hostile templates | as above |
| T6 MIR verifier | **PASS** — 300 programs lowered and verified; 19 reached MIR | `t6_mir_lowering_and_verification_never_panic` |
| T7 malformed artifacts | **NOT RUN** — see §5 | — |
| T8 LSP/diagnostic/protocol | **PASS** — 26 malformed framings and payloads | `t8_malformed_protocol_input_fails_boundedly` |
| T9 hostile-input resource limits | **FAIL** — DEV-214 | `t9_dev214…`, `t9_dev186…`, `t9_pathological…` |

Determinism is checked separately and passes, twice over: the generator is stable for a seed, and
the compiler is stable for a source.

---

# 3. The forcing function ran first, and it is why the passes count

`aaa_harness_self_test_detects_an_injected_panic` — named to sort first — proves the driver
**reports** a panic in the driven region rather than swallowing it, and, in its second half, that an
ordinary program does **not** trip the detector. Two-sided, like `as8-mutate.py --batch 0`.

> A clean run of an uncalibrated harness is worth nothing. Eleven tests in this file assert the
> *absence* of a panic, and an assertion of absence is vacuous until the detector is shown to fire.

---

# 4. DEV-214 — the finding

**A left-associative operator chain aborts the compiler with a stack overflow.**

The parser **has** the right guard — `MAX_DEPTH = 200`, *"this code is nested too deeply to parse"* —
and it bounds **syntactic nesting**, because that is what recurses in a recursive-descent parser. A
chain does not recurse there: `parser.rs` implements *"the 16-level precedence table literally (one
function per level)"* and each level folds operands in a `loop`. The counter never moves.

**The AST is still `n` deep**, and the recursive walks after the parser descend it.

```text
(((((...1...)))))  300 deep   ->  REJECTED cleanly          <- the gate's bounded failure
1 + 1 + ... + 1     65 terms  ->  SIGABRT, process death    <- DEV-214
```

## 4.1 Severity scales with the thread's stack, and that is the serious part

```text
8 MiB stack   a process main thread                       n = 240 OK,  n = 250 ABORTS
2 MiB stack   Rust's default for a SPAWNED thread, and
              what `cargo test` gives each test           n =  60 OK,  n =  65 ABORTS
```

~30 KB of stack per AST level. **Sixty-five `+` operators kill the process on a default-stack
thread** — and the LSP analyses on a server thread, so an embedding sits on the low number. The
shape is not exotic: any left-associative chain qualifies, including string concatenation, a long
boolean condition, and machine-generated arithmetic.

It was found by `cargo test` overflowing at a size that `cargo run` survived — the difference
between the two *is* the finding, and a suite run only on a main thread would have reported a
threshold four times too generous.

## 4.2 Not repaired, and the reason is a rule rather than reluctance

Each available fix crosses a line C10 draws:

1. **Count chain depth against `MAX_DEPTH`** — expressions of 200–245 terms that compile today
   would start being rejected. That changes the normative accepted/rejected program set:
   **CE1/CE2**, Charter §2.2, plan stop condition 5.
2. **Convert the walks to an explicit worklist** — a structural change to the type checker and the
   index builders, squarely inside plan §3.2's forbidden "broad refactoring".
3. **Raise or relocate the stack** — an architectural decision that moves the cliff rather than
   removing it.

**Owner decision required.** "Reject deep chains cleanly" needs a number only the owner can set.

---

# 5. What C10-B did NOT do, stated rather than left to be inferred

```text
T3  resolver / package / module graphs   NOT RUN. Needs generated package trees on disk (cyclic
                                         deps, missing entries, alias collisions, malformed
                                         starkpkg.json). In scope, not attempted here
T7  malformed artifacts                  NOT RUN. ONNX, build.json, stark.lock, manifest.json,
                                         corpus.lock. In scope, not attempted here
minimisation                             no minimiser was built. Both findings arrived already
                                         minimal because the corpora are hand-shaped
corpus retention                         nothing to retain: no generated case failed. The
                                         generators and their seeds are committed, per policy —
                                         generated corpora are not committed wholesale
```

**Two of nine declared targets were not run.** They are named here rather than dropped from the
population, because a target quietly removed after the fact is precisely the denominator
manipulation plan §7 forbids. **C10-Q may not claim robustness qualification over T3 or T7.**

---

# 6. Findings

| ID | Finding | Class (plan §11.2) | Disposition |
| --- | --- | --- | --- |
| **DEV-214** | A left-associative operator chain aborts the compiler with a stack overflow — 65 terms on a 2 MiB thread stack, 250 on 8 MiB. The parser's depth guard bounds syntactic nesting, not resulting AST depth | A (compiler) + C (DoS surface S13) | **OPEN. Owner decision** — every fix changes the accepted set, the architecture, or neither-but-moves-the-cliff |
| **DEV-186** | Confirmed at HEAD and characterised: `Server::run` does `vec![0u8; content_length]` **before** reading, so a header alone decides an allocation | A + C | Already OPEN. Test written so a future limit flips it |
| **B-F1** | The parser's `MAX_DEPTH` guard **works** — 300-deep paren nesting is rejected cleanly. DEV-214 is a gap in what it measures, not a missing guard | — | Recorded; it is what makes DEV-214 a bounded, fixable defect |
| **B-F2** | Everything else declared and run — T1, T2, T4, T5, T6, T8 — met the gate. No panic, no hang, no non-determinism across ~2,000 cases | — | Believed because the harness self-test fires |

---

# 7. Reproducing

```bash
cargo test --manifest-path starkc/Cargo.toml --test c10b_robustness          # 12 tests
cargo run  --manifest-path starkc/Cargo.toml --example c10b_repro  -- 250    # DEV-214, aborts
cargo run  --manifest-path starkc/Cargo.toml --example c10b_repro  -- 245    # completes
cargo run  --manifest-path starkc/Cargo.toml --example c10b_thread -- 65 2097152   # aborts
cargo run  --manifest-path starkc/Cargo.toml --example c10b_thread -- 60 2097152   # completes
```

Seeds are fixed in the source (`0xC10B_0001`, `0xC10B_0002`, `0xC10B_0045`, `0xC10B_0006`,
`0xC10B_DE7E`). Nothing about the host — clock, PID, path, hash seed — enters a case's identity, so
a failure reported here reproduces elsewhere.
