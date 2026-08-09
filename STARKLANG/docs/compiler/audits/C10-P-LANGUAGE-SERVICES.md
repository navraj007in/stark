# C10-P — Language-services prerequisite packet

**Packet:** C10-P, authorised by owner ruling **OD-4** (CD-395) as a bounded prerequisite running
alongside C10-A1.
**Date:** 2026-08-09. **Baseline:** `f12ececca6d4bdabf828d657c4a4f719a7f9c39a`.

**Scope, verbatim from OD-4:** close `DEV-213` and `DEV-012` rather than carry them, *because*
C10 evaluates **Core v1 Compiler Stable** (OD-2). Neither blocks C10 opening; both gate the claim
through **C10-G**, the gate immediately before C10-Q.

```text
DEV-213   CLOSED   this packet
DEV-012   OPEN     blocked on a human in an editor — see §3
```

---

# 1. DEV-213 — CLOSED

Full record: `starkc/docs/conformance/KNOWN-DEVIATIONS.md`, live heading *"DEV-213 — CLOSED
(C10-P, 2026-08-09)"*. Summary here; that entry is authoritative.

**Defect.** The LSP cached one whole-package `ProjectAnalysis` **per open URI** and invalidated only
the edited URI, while `handle_workspace_symbol` merged symbols across every cached analysis. A
rename in one open file left every other open file's analysis carrying the old name, and the
response contained both.

**Repair.** `CompilationResult` records the `package_root` its analysis was built against;
`invalidate_package_of` drops the edited URI's entry and every sibling sharing that root. Called
from `open_document`, `update_document` and `close_document` — all three change the overlay set the
analysis is computed from, not only the edit path the defect was demonstrated on.

**Boundary honoured.** No LSP redesign, no incremental analysis, no change to C8's scope. Two files
touched (`lsp/state.rs`, `lsp/server.rs`), one field added, one function added, one function
extracted so that "which package is this URI in" has a single implementation.

## 1.1 The negative control — why the pass is believed

Gate C10's binding rule (plan §3.4, inherited from AS8): *a material evidence claim is not
established merely because its suite passes.*

```text
repair in place            dev213_editing_one_file_invalidates_every_analysis_of_its_package  PASS
sibling sweep disabled     same test                                                          FAIL
                           got ["alpha_symbol", "renamed_symbol"] — the defect's exact signature
control removed            restore verified byte-identical, then re-run                        PASS
```

The test is AS8's, **renamed and polarity-flipped rather than deleted**, exactly as its own
assertion message instructed. It also now checks that recompiling the sibling yields the new name
and *only* the new name — so the repair is demonstrated to be an invalidation, not a purge that
hides the stale entry by emptying the cache.

**A mechanical cross-check, not a claim.** `starkc/scripts/c10-deviation-populations.py` re-run
after the closure reports population A's live-OPEN set dropping **18 → 17** with `DEV-213` absent.
The extractor is independent of the repair and of this document.

## 1.2 Evidence

```text
cargo test --manifest-path starkc/Cargo.toml --lib lsp::            48 passed, 0 failed
cargo test --manifest-path starkc/Cargo.toml --lib                 569 passed, 0 failed
cargo clippy --manifest-path starkc/Cargo.toml --workspace \
      --all-features --all-targets -- -D warnings                  exit 0, zero warnings
cargo fmt --manifest-path starkc/Cargo.toml --check                clean
```

Evidence class (Charter §5.2): **REG** (regression test for a discovered bug) plus **UNIT**. Not
MANUAL, and not editor-validated — see §3.

## 1.3 Freshness consequence for C10-D (plan §8.2a)

C10-0 predicted in advance that a DEV-213 repair confined to the LSP would disturb none of the 12
mutation-authority files or 13 control suites, because `src/lsp/` appears in neither list. **That
prediction is now testable and holds:** the files this packet changed are `src/lsp/state.rs` and
`src/lsp/server.rs`. Every inherited AS8 mutation result remains FRESH. Re-verified at C10-Q
regardless, per the rule.

---

# 2. C10-G status after this packet

```text
DEV-213   CLOSED     -> the workspace/symbol claim may be stated WITHOUT the AS8 qualification,
                        within the bound in §1
DEV-012   OPEN       -> the language-services claim must still be NARROWED at C10-Q, unless the
                        session in §3 happens first
```

---

# 3. DEV-012 — NOT closed, and the reason is not effort

**This packet cannot close DEV-012.** Seven advertised features (diagnostics, formatting,
completion, signature help, rename, document symbols, semantic tokens) have protocol evidence only.
Closing the deviation requires a person exercising them in a real editor — **MANUAL evidence** under
Charter §5.2, which must be disclosed as manual and never described as automated coverage.

An autonomous session cannot produce it. Recorded here as owed work rather than silently deferred.

**What the session requires**, so it can be done in one sitting:

```text
environment   VS Code with the packaged extension (C8 used VS Code 1.130.0,
              starklang.stark-language@0.2.0, macOS 26.5.2 arm64)
subject       a real multi-file STARK package, not a scratch file
per feature   exercise it, record the observed BEHAVIOUR, not just that a response arrived
```

**Check values, not verdicts.** `GATE-C8-CLOSURE.md` §4 records why: DEV-182 — the LSP JSON parser
silently decoded every escaped non-BMP character to the empty string — **passed** protocol
validation, because both parse and response succeeded and only the *value* was wrong. A session
that confirms "the feature responded" reproduces exactly the gap that let DEV-182 through.

**Outcomes, per OD-4:**

```text
all seven behave correctly   -> CLOSE DEV-012
any feature fails            -> allocate or reuse a bounded DEV, decide from the evidence
session not obtained         -> C10-Q states the language-services claim as THREE
                                interactively-confirmed navigation queries plus SEVEN
                                protocol-conformant features, with DEV-012 named
```

The fallback is the fallback. It is not the plan.

---

# 4. What C10-P does not claim

```text
NOT "the LSP is correct"          one demonstrated defect is repaired; DEV-186 (unbounded
                                  Content-Length before parsing) is still OPEN and is a C10-B/T8
                                  and C10-C/S11 item
NOT a reopening of C8             CD-385 closed it. This is a new bounded packet, which the AS8
                                  record itself anticipated
NOT editor validation             §3. No UI was exercised by this packet
NOT a performance claim           AS8 measured the duplication's cost as immaterial (22 ms for one
                                  analysis, 181 ms for eight open URIs) and the defect was never
                                  about cost. This repair invalidates MORE eagerly than before,
                                  so a package with many open URIs recompiles more often. No
                                  before/after was taken and none is claimed
```

That last line is the honest residual: **C10-P traded an unmeasured amount of recomputation for
correctness, deliberately and without measuring it.** If C10-E's LSP workload is added to the frozen
performance set (an owner decision, plan §12.1), this is the change it should be measured against.
