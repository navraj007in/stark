# `stark verify` validation track (Proposal)

**Status:** Proposed (authorised by the Gate 7 `RETAIN AS RESEARCH` decision,
2026-07-16). Not yet started.
**Type:** Roadmap-governed bounded **product-validation** experiment (ROADMAP §5).
**Owner:** STARK maintainer (Navraj Singh).

This proposal is authorisation to *plan* only. It does **not** presume the
verifier is worth productising, and it is deliberately independent of the
language-direction decision: a positive result here would support a *verifier*
product regardless of whether anyone adopts the STARK language.

---

## 1. Hypothesis

> A standalone `stark verify` — checking an ONNX artifact against a declared
> model signature (port names, dtypes, static dims, and dynamic-dim identity)
> and emitting deterministic JSON — is useful enough as a low-friction CI guard
> that external developers would add it to a real pipeline **without adopting
> the STARK language**.

Built to be falsified: if developers would not add it to CI — because existing
tooling already covers it, the friction is too high, or the failure it catches
is not one they hit — the outcome is *stop the verifier track*.

## 2. Why this track exists

Gate 6 raised, and Gate 7 deferred, a question the technical experiments never
answered: is `stark verify` a useful CI tool *even when users do not adopt the
STARK language*? The Gate 7 decision (`starkc/docs/gate7-decision.md`) validated
the tensor-track *technology* but recorded **no product/adoption evidence**.
This track tests **demand**, with real developers — the one thing the compiler
experiments could not.

## 3. What already exists (this track validates it; it does not build it)

- `starkc verify <model.onnx> --declaration <model.stark> [--model <Name>]
  [--message-format text|json]` — verifies an artifact against a declared
  signature (tested in `starkc/tests/gate4_onnx.rs`).
- `starkc import <model.onnx> --out <model.stark>` — generates the declaration
  from an artifact, so a user writes at most a signature stub, not STARK code.

The verifier is therefore the *existing* deploy-time/import machinery exposed as
a standalone command. This track packages, exercises, and **validates** it.

## 4. Selected signature forms (the technical coverage to demonstrate)

The verifier must be shown to handle, with recorded evidence, three ONNX
signature shapes on real (or realistically-shaped) artifacts:

1. **Fixed** — all-static dims (e.g. a classifier `[1,3,224,224] → [1,1000]`).
2. **Symbolic** — a dynamic dimension whose identity is checked across ports
   (e.g. Tiny YOLOv2 `[B,3,416,416] → [B,125,13,13]`).
3. **Multi-port** — more than one input and/or output. *(Deploy currently
   restricts to single-input/single-output; V-01 confirms whether `verify`
   shares that limit and, if so, lifts it for verification only — no deployment
   change.)*

For each form: a valid artifact/declaration pair (passes) and a drifted one
(fails) — with the recorded command, exit code, and JSON diagnostic.

## 5. Deterministic JSON output contract

`--message-format json` must emit a **stable, documented** object: a top-level
result (`match` / `mismatch`), and for a mismatch an ordered list of typed
differences (port, axis, expected, actual, kind). Determinism is a first-class
requirement (a CI tool that reorders findings is unusable): running twice on the
same inputs is byte-identical, and the schema is versioned. V-02 pins and
documents this schema.

## 6. Implementation boundaries (what may be built for the evaluation)

- Package the existing `verify`/`import` commands; add the three signature-form
  fixtures (§4) and the deterministic JSON schema (§5).
- Lift the single-port restriction **for verification only** if V-01 finds it —
  no change to deployment lowering, no new language features.
- A fresh-clone **evaluator guide** (§8) and a runnable example (a CI snippet:
  GitHub Actions / a shell step) using only `import` + `verify`.
- **No** language surface changes, no new compiler analyses, no VM/DSL/runtime
  work. This track is packaging + validation, not language development.

## 7. Explicit non-goals

- Building a new verifier from scratch (it exists).
- Any conclusion about the STARK *language* (kept independent, per §1).
- Networking, a hosted service, a package registry, cloud CI infrastructure.
- Fabricating, simulating, or inferring developer feedback of any kind (§9).
- Broadening scope to the language tracks left UNRESOLVED by Gate 7.

## 8. Human-validation protocol (the core of this track)

Conducted by the **owner** with **real, external** developers — not by the
assistant, and not simulated:

1. Recruit **≥ 3 external developers** who ship ML/ONNX in CI (not STARK
   contributors).
2. Give each the fresh-clone evaluator guide: clone, install the pinned
   toolchain, run `import` + `verify` on their *own* (or a provided) ONNX model
   in fixed/symbolic/multi-port forms, and trigger a drift failure.
3. Collect, per developer, in their own words: whether they would add
   `stark verify` to a real CI pipeline (yes/no/conditional), the friction they
   hit, what would change their answer, and what existing tool (if any) they
   would use instead.
4. Record responses verbatim with attribution the developer consents to
   (name/handle or anonymised), plus the date and the artifact they used.

## 9. No-fabrication clause (binding)

Human participants must be real and their feedback recorded verbatim. The
assistant must **never** invent, paraphrase into existence, simulate, or
"reasonably infer" developer responses, adoption counts, or quotes. If real
feedback has not been collected, the human-evidence sections stay empty and the
decision (V-04) cannot be recorded — full stop. Machine-checkable artifacts
(fixtures, JSON determinism, the guide running end-to-end) may be produced and
measured by the assistant; *opinions and adoption intent may not*.

## 10. Measurements

- **Technical (assistant-measurable):** the three signature forms pass/fail
  correctly with recorded commands + JSON; JSON output is byte-deterministic
  across runs; the fresh-clone guide runs end to end on a clean machine; verify
  wall-time and binary footprint.
- **Human (owner-collected, real):** per-developer CI-adoption intent
  (yes/no/conditional), enumerated friction points, and named alternatives.

## 11. Exit criteria (candidate outcomes, recorded in V-04)

- **PRODUCTISE VERIFIER** — only if the three signature forms are demonstrated
  *and* a **majority of ≥ 3 real external developers** say they would add it to
  a real CI pipeline (with friction points that are addressable), grounded in
  recorded verbatim feedback.
- **ITERATE** — clear, addressable friction blocks adoption but interest is
  real; record the specific changes and re-run.
- **SHELVE THE VERIFIER** — see stop criteria.

## 12. Stop criteria

Record **SHELVE THE VERIFIER** when any hold, from real evidence: developers
would not add it to CI (existing tooling suffices, or the drift class is not one
they hit); the friction is fundamental rather than fixable; or fewer than three
real external evaluators can be recruited (demand cannot be measured — do not
substitute assumed demand).

## 13. Work packages and branches

| WP | Branch | Deliverable |
| --- | --- | --- |
| V-00 | `verify/v-00-proposal` | this proposal (docs only) |
| V-01 | `verify/v-01-signature-forms` | three signature-form fixtures + evidence; multi-port for `verify` if needed |
| V-02 | `verify/v-02-json-contract` | pinned, versioned, deterministic JSON schema + docs |
| V-03 | `verify/v-03-evaluator-guide` | fresh-clone guide + runnable CI example |
| V-04 | `verify/v-04-decision` | **owner** collects real developer feedback; decision recorded |

Each WP begins with a clean tracked tree and the git preamble; after any
compiler change: `cargo fmt --check`, `clippy -D warnings`, `test --all-targets`,
`build --release`, `doc --no-deps`, `git diff --check`. V-04 cannot be recorded
until §8 feedback exists (§9).

## 14. Evidence locations

```text
STARKLANG/docs/proposals/VERIFIER_VALIDATION_TRACK.md   (this file)
starkc/tests/fixtures/verify/{fixed,symbolic,multiport}/  (artifacts + declarations)
starkc/tests/results/verify/{signature-forms,json-schema,determinism}.json
starkc/docs/verifier-evaluator-guide.md                  (fresh-clone guide)
starkc/docs/verifier-decision.md                         (V-04, incl. verbatim real feedback)
```
