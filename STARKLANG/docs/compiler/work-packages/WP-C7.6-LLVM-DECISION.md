# WP-C7.6 — LLVM decision

**Outcome:** `DEFER` — the roadmap's default. **CE6 is NOT opened.**
**Basis:** WP-C7.5's measurements, plus the explicit limits on what those measurements can support.
**Re-evaluation trigger:** stated in §6, not left to judgement.

---

## 1. The gate condition, and whether it is met

The roadmap says: *"Open CE6 only if measured workloads show a material limitation in the selected
backend."*

**No measured workload shows a material limitation.** But the honest form of that sentence matters,
because there are two very different ways to arrive at it:

- *We measured for a limitation and found the backend adequate.*
- *We measured, and the corpus cannot resolve the question either way.*

**It is the second.** WP-C7.5 established that the heaviest frozen workload's release compute is
below measurement resolution — 1.527 ms against a 1.547 ms empty-program floor. A corpus that cannot
distinguish a Collatz loop from an empty program cannot demonstrate a code-generation limitation,
and equally cannot demonstrate its absence.

So DEFER here rests on **absence of evidence**, and is the correct outcome for that reason: opening
CE6 requires positive evidence of a material limitation, and there is none. It does not rest on a
finding that the backend generates good code, because that has not been measured.

## 2. Missing optimisation or target capability

**None identified.** The backend emits Rust and hands it to rustc, so it inherits rustc's
optimisation pipeline — which is LLVM. Adopting LLVM directly would not add an optimisation the
current path lacks; it would remove a layer above the same optimiser.

Target capability is likewise inherited: every target rustc supports is reachable in principle, and
WP-C7.1's `--target` validation already names them. Cross-compilation is validated-but-refused for
reasons in the toolchain layer, not the backend.

## 3. Expected benefit

**Not quantifiable from current evidence, and plausibly near zero for code quality.** The generated
Rust already reaches LLVM. The realistic benefits of a direct backend are compile *time* (skipping
Rust parsing and type checking) and self-containment (§5) — not runtime performance.

On compile time, WP-C7.0 measured host Cargo/rustc at 65–68 % of a cold build. A direct backend
could attack that share. But WP-C7.3 already recovered a 2.0× median on rebuilds by caching, and
0.4 s cold builds are not a reported problem. The benefit is real but small against the cost in §4.

## 4. Integration and build complexity

The current backend is **6,608 lines across 11 files** (WP-C7.5 §6), and it delegates register
allocation, instruction selection, calling conventions, and every target's ABI to rustc.

A direct LLVM backend takes all of that on: LLVM IR generation for every MIR construct, the ABI for
each supported target, debug-info emission, and version-tracking against an LLVM C++ API that
changes shape between releases. This is not a 6,608-line component. It is the single largest piece
of engineering the compiler could take on, and it would be undertaken to reach an optimiser the
project already reaches.

## 5. Binary and toolchain burden — the strongest argument for change, stated fairly

The current backend's real limitation is not code quality. It is that **`stark build` requires cargo
and rustc on the user's machine.** STARK is not a self-contained toolchain, and that is a genuine
product limitation, independent of any performance measurement.

LLVM is the **worst** available answer to it. Linking LLVM means tens of megabytes of toolchain, a
C++ build dependency, and version coupling — trading a rustc dependency for a heavier one.

**Cranelift remains the live alternative**, as the charter's `SELECT-DIRECT` option already records:
a Rust-native code generator, no C++ dependency, materially smaller than LLVM, at the cost of weaker
optimisation than LLVM produces. That trade is acceptable precisely where STARK sits today.

**This distinction is the substantive content of this WP.** "Should STARK have a direct backend?" and
"Should STARK use LLVM?" are different questions with different answers on current evidence. The
first is open and motivated by self-containment; the second is DEFER. Collapsing them would either
defer a live option or adopt LLVM for a goal Cranelift serves better.

## 6. Contributor impact, and the re-evaluation trigger

**Contributor impact of LLVM adoption:** a contributor to the current backend needs Rust and an
understanding of the MIR contract. A contributor to an LLVM backend needs LLVM IR, the C++ API, and
target ABI knowledge. For a research language, this narrows the contributor pool sharply for a
benefit §3 cannot quantify.

**Re-evaluate when any of these becomes true** — stated as conditions rather than left to judgement:

1. P1's practical systems workloads measure a **generated-code** deficit — not a compile-time or
   startup deficit — against a comparable Rust or C implementation.
2. Self-containment becomes a primary project goal. That reopens `SELECT-DIRECT`, and the candidate
   to evaluate first is **Cranelift**, not LLVM.
3. A required target is reachable by LLVM and not by rustc. None is known today.

## 7. Alternatives within the current backend

Cheaper than any backend change, and available now:

- **Build caching** — done (WP-C7.3, 2.0× median rebuild).
- **Baseline MIR optimisations** — done (WP-C7.4), and measured to fire zero times on real
  workloads, which is itself evidence that code-generation quality is not where the constraint is.
- **`--release` profile settings** — done (WP-C7.1).
- **Reducing the 443 KB release floor** — untried. WP-C7.5 found every release binary within a 4 KB
  spread regardless of program, so the floor is runtime and std, not codegen. If binary size ever
  matters, that floor is the target, and no backend change addresses it.

## 8. What this decision does not claim

- It does not claim the generated code is fast. That is unmeasured (§1).
- It does not claim a direct backend is unwarranted — only that **LLVM** is the wrong one, and that
  the question is not settled by any evidence now in hand.
- It does not close the self-containment question, which is a product decision outside C7's remit.

**CE6 remains unopened.** Opening it requires positive evidence of a material limitation, and the
right moment to look for that is P1 — the same dependency that blocks WP-C7.5's closure.
