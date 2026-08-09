# C10-0 — Opening Inventory and Qualification Freeze

**Packet:** C10-0, the opening packet of Gate C10 (Compiler Release Qualification).
**Authority:** `plans/WP-C10-COMPILER-RELEASE-QUALIFICATION.md` §6.3, approved with amendments
2026-08-09; owner rulings OD-1…OD-6 recorded in that plan's §2.
**Date:** 2026-08-09.

> **No substantive C10 qualification work may begin until this inventory exists and its
> contradictions are resolved or explicitly carried.** That is the packet's whole purpose.

**Status: COMPLETE for items 1–4 and 6–14; item 5 (population A) is complete as an ENUMERATION and
carries 14 entries needing owner adjudication (§5.4). Seven findings are recorded in §15.**

---

# 1. Frozen head and CI baseline

```text
QUALIFICATION BASELINE SHA   f12ececca6d4bdabf828d657c4a4f719a7f9c39a
                             "Merge main into develop — sync only, zero file content"
branch of record             develop
execution branch             wp-c10/execution-plan (branched from develop at f12ecec)
```

**Why the baseline is `f12ecec` and not the execution branch head.** The execution branch adds
**documentation-only** commits on top of `f12ecec`. No compiler source, test, workflow or manifest
differs. Every measurement in this inventory was taken against a tree whose compiler content is
byte-identical to `f12ecec`, and that is verified rather than asserted — see §6.2, where all 12
mutation-authority files and all 13 control suites hash identically at `f12ecec` and at this head.

## 1.1 The CI run this freeze rests on

```text
run id       31292404920          workflow: CI          event: push
head sha     f12ececca6d4bdabf828d657c4a4f719a7f9c39a
status       completed
conclusion   success
jobs not succeeding   NONE  (queried explicitly, not inferred from the summary badge)
```

Companion workflow on the same SHA:

```text
run id       31292404936          workflow: C7.8 Native Capabilities     conclusion: success
```

**Naming the run id is the point.** A `develop` push fires both the push trigger and the open
`develop -> main` PR trigger for the same commit; the `concurrency:` group queues them by tested
COMMIT so they do not race for the fixed peer ports (39187–39191), and **both report**. "CI is
green" without a run id is not a citable fact. At the time the C10 plan was drafted this same run
was still `in_progress`, and the plan required C10-0 to read a completed run rather than assume one
— which is why this section names one.

**Plan §15.3 stop condition 1 is DISCHARGED.** The frozen head's CI is green.

---

# 2. Compiler, toolchain and artifact versions

```text
starkc                     0.1.0        starkc/Cargo.toml
stark-runtime              0.1.0
stark-provider-abi         0.1.0        crate version
Native Provider ABI        "0.1"        stark-provider-abi/src/lib.rs::ABI_VERSION
MIR_VERSION                "0.4"        src/mir/mod.rs:59
MIR_RUNTIME_SURFACE        "0.1-A14"    src/mir/mod.rs:92 — bumped INDEPENDENTLY of MIR_VERSION
layout contract            stark-64-v1  every target, from target-matrix.json
MSRV (declared)            1.85         starkc/Cargo.toml `rust-version`
rustc (this machine)       1.93.0 (254b59607 2026-01-19)
rustc (CI)                 dtolnay/rust-toolchain@stable — FLOATING, not pinned
rustc (perf baseline)      1.93.0, recorded in benchmarks/c7-workloads/FROZEN.json
lints                      unsafe_code = "forbid" at the crate root
```

**There is no STARK language-version constant.** Core v1 is identified by the normative
specification set, not by a number in the source tree. C10-F must either define one or state that
absence explicitly (plan §13.2); it is not an input to C10 and must not be invented here.

**CI's toolchain floats.** `dtolnay/rust-toolchain@stable` resolves at run time, so two runs of the
same commit can use different rustc releases. That is fine for a CI gate and **not** fine for a
release claim that says "built with". C10-F owns the decision; C10-Q must record the actual rustc
of the run it cites, not the action name.

---

# 3. Platform matrix — FOUR targets, not three

Read from `starkc/target-matrix.json` (schema `stark-target-matrix-1`), whose compiler-owned fields
are pinned in **both** directions to `src/target.rs` by
`c64_platform_matrix.rs::target_matrix_json_matches_the_compiler`.

| Triple | Tier | Suffix | Archive | Installers | CI coverage |
| --- | --- | --- | --- | --- | --- |
| `aarch64-apple-darwin` | **tier-1** | — | tar.gz | install.sh / uninstall.sh | full |
| `x86_64-unknown-linux-gnu` | **tier-1** | — | tar.gz | install.sh / uninstall.sh | full |
| `x86_64-pc-windows-msvc` | **tier-2** | `.exe` | zip | install.ps1 / uninstall.ps1 | partial — gap probe + smoke + P1 + packages |
| `x86_64-apple-darwin` | **tier-3** | — | tar.gz | install.sh / uninstall.sh | **NONE** |

> **FINDING F1 (§15).** The compiler names **four** targets and packages all four. `x86_64-apple-darwin`
> is **tier-3 and is exercised by no CI job at all** — it is a target the release tooling will build
> an archive and an installer for, with zero executed evidence. The C10 plan's own §1.5 listed three
> targets and missed it. **C10-Q may not include tier-3 in any conformance claim**, and C10-F must
> state what tier-3 means: packaged, named by the compiler, unexercised.

---

# 4. Release classes under evaluation (OD-2, RULED)

```text
EVALUATE          Core v1 Compiler Stable          the class C10 exists to qualify
                  Native Systems Preview           fallback claim, already satisfiable on C6+P1

DO NOT EVALUATE   STARK v1 General-Purpose Stable  materially wider claim on much the same
                                                   evidence; a separate owner act (CD-022)
                  Native Developer Preview         subsumed
```

---

# 5. The three deviation/residual populations (OD-3, RULED)

Frozen separately. **Only population A is the denominator for CD-021's compiler-conformance rule.**
B constrains release/distribution wording; C constrains the strength of evidence claims and asserts
no defect at all.

Regenerate A with:

```bash
python3 starkc/scripts/c10-deviation-populations.py          # human-readable
python3 starkc/scripts/c10-deviation-populations.py --json   # machine-readable
```

## 5.1 Ledger shape — the counts, and why three of them differ

```text
186   `## DEV-` headings in KNOWN-DEVIATIONS.md
170   distinct DEV ids OWNING a heading
178   distinct DEV ids MENTIONED anywhere in the file   (the gap is in-body cross-references)
190   "deviation entries" reported by as8-reconcile-deviations.py
```

> **FINDING F4 (§15).** Four different counts of "how many deviations are there", all correct for
> their own question, none interchangeable. The C10 plan cited 186/178 as though they answered one
> question. **C10 documents must always say which count they mean.** The reconciler's 190 uses a
> broader entry regex than `^## DEV-`; the difference is not an error in either tool.

The file is **APPEND-ONLY**: a deviation gets a NEW heading each time it is touched, so the first
heading is not its status — the last one is. `DEV-121` opens OPEN and is CLOSED 3,558 lines later.
Both extractors above key on the **last** heading.

## 5.2 Population A — compiler deviations

**A1. Live OPEN by the last ledger heading — 18.**

```text
DEV-012  interactive editor validation, 3 of 10 features        -> C10-P (OD-4)
DEV-120  native call-depth exhaustion, bounded host limitation
DEV-122  span source-identity gap (guarded; instance fixed CD-306)
DEV-140  `Vec::` method outside the implemented lowering set     ) the six CD-342
DEV-141  `HashMap` over a user-`Drop` value type                 ) "layer defect"
DEV-142  droppable composite carrying a borrowed element         ) registrations —
DEV-143  `assert_eq` on a user-defined type                      ) these bound the
DEV-144  `for` over a non-range, non-`Vec` iterator              ) SUPPORTED SUBSET
DEV-145  method on a peeled type outside the implemented slice   ) and are load-bearing
DEV-167  `Display::fmt` has no method-form `to_string()`         (deferred by decision)
DEV-168  qualified call to a compiler-known trait's method has no MIR lowering
DEV-172  no signed type can express its own minimum value        (pre-existing)
DEV-177  generic-parameter shadowing accepted, contrary to NAME-SHADOW-001
DEV-178  generic context not retained for associated-fn calls or function values
DEV-180  HIR interpreter flattens `&mut self` into owned receiver storage
DEV-181  a borrow taken by an assignment's own RHS blocks the assignment
DEV-186  the LSP transport allocates an unbounded `Content-Length` before parsing
DEV-213  LSP per-URI analysis cache, stale workspace/symbol      -> C10-P (OD-4)
```

**A2. OPEN in `COMPILER-STATE.md`, owning NO ledger heading — 6.**

```text
DEV-156  `stark fmt` evicts member doc comments                       OPEN
DEV-157  native backend has no representation for `MirTy::Never`      OPEN
DEV-159  a native build can race its own dependency build             OPEN
DEV-160  place-granular borrows, whole-value projections              OPEN
DEV-161  an ambient `CARGO_TARGET_DIR` breaks every native build      OPEN
DEV-162  READ through a whole-value accessor                          OPEN
```

> **FINDING F3 (§15) — the most consequential of this packet.** These six are real, open,
> compiler-affecting deviations that **a C10.7 check reading only `KNOWN-DEVIATIONS.md` would not
> see.** OD-3's clause "plus any `DEV-NNN` that appears in `COMPILER-STATE.md` but not in that
> file" was written to catch exactly this, and it caught six. Three of them (DEV-157, DEV-160,
> DEV-162) are Core semantic or backend limitations that bear directly on a native-conformance
> claim; DEV-161 is an operational trap already recorded in the plan's own §8.1.

**A3. The last heading does not settle it — 8, needing owner adjudication.**

```text
DEV-005  `starkc` vs `stark` CLI gating drift on warnings
DEV-010  LSP hover/definition/references are protocol stubs        (superseded in substance by C8?)
DEV-011  doc comments are trivia, not AST/HIR metadata
DEV-020  `pub use` of a private item leaks it                      ("confirmed design, not a defect")
DEV-021  cross-package coherence checking verified working         (reads as a CLOSED note)
DEV-083  concrete position in an impl head vs unresolved receiver type argument
DEV-179  `MapIter`/`FilterIter` discard a generic callback's instantiation   (DORMANT)
DEV-196  legacy Core `File` has no drop plan                       ("NARROWED, not a live defect")
```

Four of these (DEV-020, DEV-021, DEV-179, DEV-196) read as *not live defects* and probably resolve
to `closed` or `accepted-indefinitely` on a sentence of owner text each. The extractor deliberately
does not guess: a regex deciding "confirmed design, not a defect" means CLOSED would be doing the
reviewer's job badly.

**Population A total requiring disposition before C10-Q: 18 + 6 + 8 = 32.**

> **FINDING F2 (§15).** `COMPILER-STATE.md`'s *"Known open, at a glance"* block lists **DEV-012 and
> DEV-213 only**, plus C9 Part B and the AS8 residual ranges. It omits the other 30. The block
> already says of itself *"This block is a summary and is not authoritative over the records below
> it"*, so this is not a false claim — but a release-qualification session that trusted it would
> have carried 2 deviations instead of 32. **The summary is corrected as C10-0's own output**, and
> the correction is forward-only (§16.1): the block is a current-state summary, not a historical
> decision.

## 5.3 Population B — release/distribution deviations

Constrains the release **wording**. Not compiler conformance. Not the CD-021 denominator.

```text
DEV-165                        `connect_timeout` accepted and ignored. Named a public-release
                               blocker by ROADMAP.md §1. Present in COMPILER-STATE.md; absent
                               from KNOWN-DEVIATIONS.md entirely
standalone toolchain           PARTIAL — the payload carries compiler, runtime and provider ABI,
                               NOT the first-party package/provider set
offline package build          NOT PROVEN — a clean machine cannot build an HTTP/TLS program
                               without obtaining the packages separately
signed distribution            NOT PROVEN — `stark doctor` re-hashes the payload against
                               manifest.json, establishing INTEGRITY. Anyone who can replace the
                               payload can replace the manifest and its sidecar. AUTHENTICITY
                               requires a signed manifest, a trusted release key, verification
                               before installation, and platform notarisation. None exists
tier-3 packaging               x86_64-apple-darwin is packaged and never executed (F1)
```

## 5.4 Population C — assurance residuals

Constrains the **strength** of evidence claims. **Asserts no defect.** This is the distinction AS8
paid for, and OD-3 exists to keep it.

```text
AS8-R1   a wrong Copy rule with no drop consequence is invisible to every differential suite
AS8-R2   ESF-TRAP-001a — no control, and NONE CONSTRUCTIBLE. Permanent residual
AS8-R4   copy_canon_matrix is a transcription, not a control
AS8-R5   EV-SPEC-FIXTURES does not control TYPE-PRIM-001
AS8-R6   ESF-COPY-002 unexercised: no case duplicates a `&mut` and observes it
AS8-R8   array destruction order is unguarded
AS8-R9   strip_ref recursion is unguarded
AS8-R10  ESF-TRAIT-001 has NO CONTROL OF ANY KIND — the highest-value C10-D target
AS8-R12  AS8-DA-005: scalar_name can drift silently
AS8-R13  non-`pub` re-export visibility has no control anywhere
AS8-R14  mir::verify::may_need_drop's HostResource arm is unguarded
AS8-R7   13/39 predictions falsified — a METHOD finding, not a residual to close
AS8-R11  corrected in the compiler's favour: mir/verify.rs IS a control
AS8-R3   DISCHARGED (corpus census exists)
AS8-R15  DISCHARGED (full-corpus coverage completed) — see OD-5; two records still say otherwise
AS8-DA-001, DA-005   owner ruling CONSOLIDATE, "after Sprint 4" — schedulable, NOT C10 work
AS8-DA-002/003/004   owner ruling REMAIN SEPARATE + build the RuntimeFn parity/drift test.
                     Test-only, therefore permissible in C10-D
AS8-DA-006           KEEP — the positive exemplar. No action
RA-LAYOUT            unmeasured (EI3 residual)
RA-LINTS             two deny-by-default lints suppressed in generated code, narrowing what
                     rustc refuses — i.e. narrowing "rustc is a genuine external control"
DEV-017              the coverage database cannot express per-rule positive/negative evidence.
                     THE reason most granular rules are unclassified — measured by C10-A1 as
                     85 of 168 AGGREGATE. Closed in the record
                     and named by no test (reconciler)
branch coverage      unavailable from this toolchain. Not fabricated, not claimed, never stated
```

---

# 6. Mutation authority inventory, and the §8.2a freshness verdict

## 6.1 Inherited authorities

```text
11   ESF-* shared-fate authorities        ENGINE-SHARED-FATE-REGISTER.md (after the TRAP-001 split)
 6   AS8-DA-* duplicated authorities      AS8-DUPLICATE-AUTHORITIES.md
41   trials declared in as8-mutate.py     = 39 recorded trials + 2 Batch-0 self-tests
12   distinct compiler-source files those trials mutate
26   CONFIRMED / 13 FALSIFIED             the 39 recorded trials, both directions
```

## 6.2 Freshness — MEASURED, not assumed (plan §8.2a)

Reference commit for the recorded verdicts: **`e7bb95d`** (CD-394, AS8 CLOSED). Compared against
this head by git blob SHA.

**Clause 1 — the semantic authority targeted by each trial:**

```text
src/backend/generated_rust/emit_bodies.rs   b2d41fab = b2d41fab   FRESH
src/mir/drop_plan.rs                        45870416 = 45870416   FRESH
src/mir/interp.rs                           b511ef10 = b511ef10   FRESH
src/mir/lower.rs                            10fce02b = 10fce02b   FRESH
src/mir/mod.rs                              ba38241e = ba38241e   FRESH
src/mir/provider_sig.rs                     d3bc803e = d3bc803e   FRESH
src/mir/verify.rs                           bc760e25 = bc760e25   FRESH
src/provider_derive.rs                      1c8c7af8 = 1c8c7af8   FRESH
src/provider_synth.rs                       4a7bff8e = 4a7bff8e   FRESH
src/resolve.rs                              b679c581 = b679c581   FRESH
src/typecheck/traits.rs                     37119f27 = 37119f27   FRESH
src/typecheck/types.rs                      86cfc444 = 86cfc444   FRESH
```

**Clause 2 — the claimed killing / control evidence** (including the two suites AS8 found
*structurally incapable*, whose freshness matters just as much: a change there could turn a
recorded survivor into a kill):

```text
c61f_structural_copy.rs  three_engine_differential.rs  mir_differential.rs
copy_canon_matrix.rs     conformance.rs                c6_generated_corpus.rs
a11_host_resource.rs     c788_resource_lifecycle.rs    mir_verify.rs
exec_snapshots.rs        c6_mutation.rs                c6_metamorphic.rs
c65_entry_exit_contract.rs
                                                        ALL 13 FRESH
```

> **VERDICT: every inherited AS8 mutation result is FRESH and citable at this head.** No trial
> needs re-running for C10-A2's dashboard. **This is re-checked at C10-Q against the final head**,
> because C10-P's DEV-213 repair and any §3.3 defect repair move source between now and then —
> `src/lsp/` is not in the list above, so a DEV-213 repair confined to the LSP does not disturb any
> of it, and that is a prediction this inventory is making in advance rather than a reassurance.

Reproduce with `git rev-parse e7bb95d:<path>` versus `git rev-parse HEAD:<path>`.

---

# 7. Test and evidence inventory

```text
210   integration test targets      top-level `.rs` under starkc/tests/ (each is a binary)
  3   test module directories       tests/common, tests/fixtures (21 files), tests/support (4)
116   spec fixtures                 STARKLANG/tests/spec-fixtures/manifest.toml entries.
                                    (The DIRECTORY holds 117 files — the 116 fixtures plus the
                                    manifest itself. README.md states 116; this line said 117 and
                                    was counting files, not fixtures.)
 89   C6.5 differential corpus      70 generated / 13 handwritten sentinels / 6 retained
 12   metamorphic families          M01-M12
  7   frozen performance workloads  w01-w07
```

AS8 recorded **209** test binaries; the tree now has **210**. One target was added between AS8's
run and this freeze. The number is a measurement of the tree, not of any run: CD-394's *"581 tests
across 5 targets"* was a **scoped** run and must never be cited as a tree total (plan §7.3).

## 7.1 The conformance-evidence populations — C10-A1's inputs

```text
 59   legacy broad rules            core-v1-coverage.toml — `tests` field does NOT distinguish
                                    positive from negative and often cites only the aggregate
                                    conformance.rs runner (DEV-017)
168   granular semantic-freeze IDs  CORE-V1-COMPLETENESS.md — THE INVENTORY OF RECORD
      *** was stated as 161 here on 2026-08-09; CORRECTED by C10-A1 the same day. Seven
      three-segment `NUM-*` IDs were invisible to the counting method. See A1-F1 ***
 36   granular rules at test-fn     core-v1-c2.11-evidence.toml
      precision
125   granular rules NOT at that precision
```

**DECLARED DENOMINATOR for C10-A1 and C10-A2 (plan §7.2), frozen here, before measurement:**

```text
metric        per-rule conformance evidence classification
population    the granular IDs in semantic-freeze/CORE-V1-COMPLETENESS.md
              declared as 161 (C10-0); CORRECTED to 168 (C10-A1, same day) — the POPULATION
              is unchanged, the ENUMERATOR was undercounting. Two independent enumerators
              agree on 168. Recorded as a dated line per plan §7.2, not as an edit
enumerated by the ID column of that file's matrix tables
exclusions    NONE. Rules classed intentionally-deferred / prohibited / spec-defect are
              retained in the denominator and bucketed N/A with the reason, so the denominator
              cannot shrink to flatter the result
frozen at     f12ececca6d4bdabf828d657c4a4f719a7f9c39a
changed by    (no changes)
```

---

# 8. Fuzz target population (declared BEFORE any run — plan §9.2)

```text
T1  lexer + parser                    robustness.rs, extended
T2  malformed-source corpus           truncation, encoding, BOM, mixed line endings, oversized
                                      identifiers (LEX-IDENT-002's 255 limit), deep nesting
T3  resolver / package / module graphs cyclic modules, cyclic package deps, missing entry,
                                      duplicate module names, alias collisions, deep re-export
                                      chains, malformed starkpkg.json
T4  type checker                      generated ill-typed programs, deep generic instantiation,
                                      alias cycles, recursive types
T5  borrow checker                    generated ownership-hostile programs, AS4 hostile shapes
T6  MIR verifier                      malformed MIR via generated source; direct input where the
                                      API allows
T7  malformed artifacts               ONNX (truncated / wrong magic / oversized dims / hostile
                                      shape metadata); build.json, stark.lock, manifest.json,
                                      corpus.lock
T8  LSP / diagnostic / protocol       malformed JSON-RPC, wrong Content-Length, non-BMP escapes
                                      (the DEV-182 shape), out-of-range positions, stale document
                                      versions, unknown methods
T9  hostile-input resource limits     time and memory bounds for every target above
```

**Existing assets:** `robustness.rs` (9 fixed-seed pseudo-fuzz tests over both `ParseMode`s) is the
only fuzz-shaped asset. `adversarial_*.rs` (7 files), `as4_hostile_combinations.rs` and
`resource_exhaustion.rs` are hand-authored adversarial cases, not generators. **T2–T9 have no
generator today.**

**Constraint:** stable Rust only (Charter §1.10 rule 8) — `cargo-fuzz`/libFuzzer needs nightly.
C10-B extends the deterministic seeded-generator pattern. The release wording must therefore say
*bounded deterministic robustness testing*, not *fuzzing*.

**T8 has a live head start:** `DEV-186` — the LSP transport allocates an unbounded `Content-Length`
before parsing — is an OPEN deviation at HEAD and is precisely a T8/T9 finding that already exists.
It is population A, and it is also a C10-C surface item (S11/S13).

---

# 9. Security surface (frozen BEFORE any finding is reviewed — plan §11.1)

```text
S01 source and module path traversal      S09 archive extraction
S02 package/cache filesystem access       S10 dependency/package provenance
S03 artifact parsing and limits           S11 LSP workspace trust
S04 generated Rust/source escaping        S12 executable/tool paths
S05 process execution (cargo)             S13 denial-of-service inputs
S06 linker arguments                      S14 dependency vulnerabilities
S07 environment propagation               S15 licences
S08 temporary files/directories           S16 installer/release authenticity
```

Method inherited from `HC13-THREAT-MODEL.md`: **every defence names the test that would fail if the
defence were removed.** A defence with no named falsifier is recorded as unverified, not as a
defence. Findings are classified A/B/C/D per plan §11.2 and mapped onto §5's populations.

Known starting points already in the tree: C6.4 row 17 (temp dirs — `env::temp_dir` + PID +
counter, no shared root, **one known survivor outside the matrix using `/tmp`**), row 18 (generated
manifest escaped to TOML rules, not Rust `Debug`), row 19 (installed runtime cannot silently fall
back to the checkout, switchable via `STARK_REQUIRE_INSTALLED_RUNTIME`), and `DEV-186`.

---

# 10. Performance workload set (frozen — plan §12.1)

```text
FROZEN.json     frozen_at_commit 4650d4753e831d2c7e7b09a0b5cab4b8e79d07ef
                rustc 1.93.0 (254b59607 2026-01-19)
                host aarch64-apple-darwin
                7 workloads, each pinned by per-file SHA-256 plus a workload_hash

w01_minimal   w02_arith_control   w03_generic_trait   w04_string_vec
w05_hash      w06_multi_package   w07_drop_ownership
```

**Inherited reports: `c75-report-macos-arm64.json` — ONE platform.** Linux-x64 and Windows-x64
have no performance record. C10-E either produces them through CI or states the limitation; it does
not generalise one platform's numbers to three or four.

**Not covered by the frozen set** — extension is an owner decision *before* measurement, never
after: large-module scaling, multi-package scaling beyond w06's app+lib, and LSP
change-to-diagnostic latency.

### OD-8 — ONNX timings: RULED (owner, 2026-08-09). INCLUDE, QUARANTINED.

C10-0 recommended excluding ONNX import/verify/deploy. **The owner declined outright exclusion**,
and the reason is the contract: WP-C10.6 explicitly lists *"ONNX import/verify/deploy time"*, so
excluding it would need an owner override of C10's own contract rather than a C10-0 recommendation.

The ruling is **include it, quarantine it**:

```text
C10-E CORE PERFORMANCE SET
    lex / parse / resolve / check      package scaling       compiler memory
    LSP latency                        native build/runtime  binary size

C10-E OPTIONAL-EXTENSION APPENDIX  (separately scoped, separately reported)
    frozen ONNX import      frozen ONNX verify      frozen ONNX deploy

DO NOT
    aggregate ONNX into any "STARK compiler performance" number
    optimise ONNX            add tensor capability
    reopen Gate 7            treat the measurement as tensor-track progress
```

The appendix must carry this sentence verbatim:

> *These measurements qualify the already-supported, frozen tensor/ONNX maintenance surface only.
> They do not expand tensor capability, reopen the tensor productisation track, or support a claim
> of general tensor execution maturity.*

**LSP latency additionally absorbs DEV-213's residual** (owner ruling, same day). C10-P traded an
unmeasured amount of recomputation for correctness; C10-E already owes an LSP baseline, so it
measures the post-fix architecture — single open URI, multi-file package, several open URIs, edit
one file, change-to-diagnostic latency. **AS8's pre-fix numbers are historical context and may not
be called a before/after** unless the harness and workload are demonstrably identical. **No
optimisation follows automatically.**

---

# 11. External pinned evidence

```text
external sample suite   navraj007in/stark-samples
                        FROZEN AT   b3b28e757f38d691e7309f168d1209e28ac459af
                        MOVED TO    5cac025f131f5b8d8de4ceb3112ca11c913c53fc  (2026-08-09)
                        CI verifies the pin RESOLVED to that SHA — `ref:` accepts a branch, so an
                        accidental branch name would float silently
C6.5 corpus             manifest.toml + corpus.lock + generator-version.txt; generator_sha256 and
                        templates_sha256 pin the producer
exec_snapshots          the inherited frozen execution corpus, v1.4.0, its OWN lock — neither lock
                        is valid for the other tree
FROZEN.json             per-file SHA-256 of every performance workload
```

**Moving the sample-suite pin during C10 is forbidden** (plan §14.4).

### THE PIN MOVED ONCE, by explicit owner decision — 2026-08-09

```text
from   b3b28e757f38d691e7309f168d1209e28ac459af   the C10-0 freeze
to     5cac025f131f5b8d8de4ceb3112ca11c913c53fc   15-capabilities migrated to vocabulary v1
```

**Why this is not the failure §14.4 exists to prevent.** That rule forbids advancing the pin to
quiet a red external control. This is the opposite: the control went red because it **correctly
detected a real breaking change**, and the pin moved only after the break was diagnosed, reproduced
and recorded.

```text
what broke   pkg/15-capabilities/native, expected `run_ok`, failed to build:
               "no provider supplies capability `process.args`"
why          capability vocabulary v1 renames the pre-v1 names. INTENTIONAL, and documented in
             spec/packages/capabilities.md "Migration from the pre-v1 implementation names" —
             which also states a capability is never silently renamed WITHIN a vocabulary version,
             so renaming ACROSS versions is legitimate and the absent alias shim is deliberate
scope        1 of 18 samples. `15-capabilities` is the only one declaring any capability at all
repair       the spec's own mapping, applied to `capabilities` AND to `provider_api`'s `errors`
             map and per-function `capability` field — the second half is easy to miss and fails
             the build for a separate reason if skipped
result       39/39, up from 38/39
```

**The disclosure C10-Q owes.** `EV-SAMPLES` is the register's only `EXTERNALLY_DERIVED` control, and
C10's external evidence now **spans two pins** — taken at `b3b28e7` before the vocabulary change and
at `5cac025` after. Any claim resting on the external suite must say which side of that boundary it
was measured on.

**What this episode demonstrates is worth more than the repair.** The break was invisible to every
in-repo test, because the compiler's own 28 first-party packages were migrated in the same branch
that made the change. **The external control caught a compatibility break no internal evidence
could have** — precisely the value EI2 assigned it, realised rather than argued for the first time.

**Consequence for C10-F:** the capability-vocabulary axis stops being PENDING and becomes
evidenced-as-breaking.

---

# 12. Excluded scope

Restated as this campaign's refusals, each citable when refusing:

```text
compiler redesign                    Charter §1.10
broad refactoring                    Charter §1.10; WP-ARCHITECTURE-STABILIZATION is COMPLETE
new language features                Charter §1.6 rule 4, §1.7, §2.2
optimisation                         Charter §1.6 rule 7, §6; plan §12.3
provider generalisation              Charter §1.3, §1.6 rule 19, CE7; C9 Part B is DEFERRED
cleanup sweeps                       plan §3.2
reopening C8                         CD-385 closed it; C10-P is a NEW bounded packet
reopening the tensor track           Gate 7 productisation DEFER stands
AS8-DA-001 / DA-005 consolidation    schedulable, but they change compiler source -> outside C10
moving the sample-suite pin          plan §14.4
tier-3 conformance claims            F1 — packaged, never executed
```

---

# 13. Expected artefacts from every C10 packet

```text
C10-0   work-packages/C10-0-OPENING-INVENTORY.md                    THIS FILE
        scripts/c10-deviation-populations.py                        the population A extractor
        COMPILER-STATE.md CD entry recording OD-1..OD-6             pending, §14
C10-P   audits/C10-P-LANGUAGE-SERVICES.md
C10-A1  audits/C10-A1-EVIDENCE-CENSUS.md
C10-A2  C10-CONFORMANCE-DASHBOARD.md + conformance/c10-dashboard.json
C10-B   audits/C10-B-ROBUSTNESS.md + generators + minimised regression fixtures
C10-C   C10-THREAT-MODEL.md (frozen first) + audits/C10-C-SECURITY-REVIEW.md
C10-D   audits/C10-D-DIFFERENTIAL.md + C10-MUTATION-LEDGER.md
C10-E   audits/C10-E-PERFORMANCE-BASELINE.md + benchmarks/c10/<platform>.json
C10-F   C10-F-COMPATIBILITY-POLICY.md
C10-Q   GATE-C10-CLOSURE.md + C10-RELEASE-STATEMENT.md
```

---

# 14. What C10-0 still owes

```text
[ ]  transcribe OD-1..OD-6 into COMPILER-STATE.md as one dated CD entry
[ ]  OD-5's superseding note: the full-corpus coverage run COMPLETED; AS8-R15 DISCHARGED;
     AS8-COVERAGE-BASELINE.md is the live figure. CD-394 and AS8-EXIT-QUALIFICATION untouched
[ ]  OD-6's ROADMAP.md §0.1 current-state correction; §6.0's gate text preserved, marked satisfied
[ ]  correct COMPILER-STATE.md's "Known open, at a glance" block (F2), forward-only
[ ]  OWNER: adjudicate the 8 in §5.2 A3
[ ]  OWNER: rule on the ONNX performance row (§10) — recommendation EXCLUDE
```

---

# 15. Findings

| # | Finding | Severity | Disposition |
| --- | --- | --- | --- |
| **F1** | The target matrix names **four** targets. `x86_64-apple-darwin` is **tier-3, packaged with an archive and installers, and exercised by no CI job**. The C10 plan's §1.5 listed three and missed it | material to the release claim | C10-Q excludes tier-3 from every conformance claim; C10-F states what tier-3 means. Plan §1.5 corrected |
| **F2** | `COMPILER-STATE.md`'s "Known open, at a glance" lists 2 deviations; population A has **32** | material | Corrected as C10-0 output, forward-only |
| **F3** | **Six open deviations (DEV-156/157/159/160/161/162) exist only in `COMPILER-STATE.md` with no ledger heading** — invisible to any C10.7 check reading only `KNOWN-DEVIATIONS.md` | **most consequential** | Population A2. OD-3's second clause is what caught them |
| **F4** | Four different, individually correct counts of "how many deviations": 186 / 170 / 178 / 190 | methodological | Every C10 document states which count it means |
| **F5** | `starkc/tests/c6-corpus/README.md` cites `c6_corpus_cases.rs` **twice**, including a runnable command `cargo test --test c6_corpus_cases`. **That target does not exist**; the real enforcer is `c6_generated_corpus.rs` | dangling evidence pointer in an evidence-bearing document | Repair in C10-A1 (documentation, no behaviour change). Plan stop-condition 9 in miniature |
| **F6** | **Every inherited AS8 mutation result is FRESH** — 12/12 authority files and 13/13 control suites hash identically at `e7bb95d` and at this head | positive | All 39 trials citable without re-running. Re-checked at C10-Q |
| **F7** | `DEV-186` (unbounded `Content-Length` before parsing) is an OPEN deviation that is simultaneously population A, fuzz target T8/T9, and security surface S11/S13 | scope overlap | Tracked once in A; referenced, not duplicated, by C10-B and C10-C |

**F1, F3 and F5 are each a case of the same thing: a claim that looked settled and was not checked.
That is what an opening inventory is for, and finding three of them in the first packet is the
argument for having one.**

---

**C10-0 is COMPLETE as an inventory.** §14's six items are its remaining transcription work, two of
which are owner decisions. **No contradiction found here blocks C10 from proceeding** — F1 and F3
change what C10-Q may claim, not whether C10 may run.
