# STARK Implementation Plan

**Status:** active engineering plan · **Subordinate to:** [ROADMAP.md](./ROADMAP.md)

The roadmap defines *what evidence* advances the project (Gates 1–6). This
document defines *how the work is executed*: technical decisions, work
packages, test strategy, and risks. Where the two disagree, the roadmap wins.
Effort figures are relative sizing for sequencing, not calendar commitments
(per roadmap governance, no gate opens on a date).

Resolution is deliberately uneven: Gate 1 is planned at work-package detail,
Gates 2–3 at milestone detail, Gates 4–5 at decision-and-shape detail. Each
gate's plan is refined when the preceding gate is near exit — planning Gate 4
task-by-task today would encode guesses that Gate 2 will invalidate.

---

## 0. Standing Technical Decisions

Decisions that apply across all gates. Changing one requires noting the
reversal here with rationale.

| # | Decision | Choice | Rationale |
| --- | --- | --- | --- |
| T1 | Implementation language | Rust, stable toolchain only | Memory-safe compiler for a memory-safe language; best-in-class parsing/diagnostics ecosystem; matches spec's own recommendation. No nightly features — contributors and CI stay simple. |
| T2 | Repo location | New top-level `starkc/` Cargo workspace | Clean break from the pre-pivot Python in `STARKLANG/compiler/` (which stays archived-in-place with its README notice). |
| T3 | Crate layout | Single crate `starkc`, internal modules (`lexer`, `ast`, `parser`, `diag`, later `resolve`, `types`, `borrow`) | Premature crate-splitting creates interface churn; split only when compile times or ownership boundaries demand it. |
| T4 | Lexer | Hand-written | The spec requires nested block comments and maximal munch with `>>` re-tokenization support — awkward in lexer generators, trivial by hand. |
| T5 | Parser | Hand-written recursive descent; Pratt (precedence-climbing) for expressions | Core v1's grammar was written for RD (one-token lookahead + the two documented parsing notes). Best diagnostics control. No parser generator dependency. |
| T6 | AST | Arena-allocated nodes referenced by typed IDs (`ExprId`, `ItemId`, …), every node carries a `Span`; no Rust references or lifetimes in the tree | Gate 2 attaches types/ownership facts in side tables keyed by node ID — adopting IDs on day one costs minor parser boilerplate and avoids a mechanical Gate 2 refactor. The separate lowered HIR still lands at M2.1, not Gate 1. *(Amended from "plain owned tree" per architecture review.)* |
| T7 | Diagnostics | `codespan-reporting` (or equivalent single-purpose crate) emitting the exact format specified in 04-Semantic-Analysis.md, with `E####`/`W####` codes from the spec | The error format is normative; adopting it from day one makes diagnostics testable against the spec. |
| T8 | Testing | Fixture manifest + snapshot tests (`insta` or golden files) + unit tests | See §2. Fixtures are the conformance backbone; snapshots make diagnostic regressions visible in review. |
| T9 | CI | GitHub Actions: `cargo fmt --check`, `clippy -D warnings`, `cargo test`, fixture conformance job | The fixture job is the mechanized version of the spec's own "every example must parse" rule. |
| T10 | Spec-bug protocol | Implementation never diverges silently: a grammar/semantics defect found while implementing produces, in one commit — the spec fix, regenerated `STARK-Core-v1.{md,html,pdf}`, regenerated fixtures, and the compiler change | Continues the discipline that fixed the earlier drift; per CLAUDE.md conventions. |
| T11 | Execution strategy (Gate 3) | Tree-walking interpreter over the checked AST | Fastest path to "programs run"; the roadmap explicitly permits it and forbids requiring a VM. Lowering/codegen decisions belong to Gate 5. |
| T12 | Model backend (Gates 4–5) | ONNX Runtime via generated Rust host glue — `ort` crate first, dropping to the ORT C API only if the wrapper blocks something. Backend ladder beyond the prototype: Cranelift for native scalar STARK code *if* Gate 6 shows the interpreter is the bottleneck; IREE (via its stable C embedding API — Rust bindings are unofficial) *if* evidence demands compiled tensor graphs. Neither ladder rung is built before its evidence exists. | ORT is the lowest-integration-cost path to a native artifact; the prototype's value is in the frontend checks, not the executor. Revisit at Gate 6. |

---

## 1. Gate 1 — Core Front End (lexer + parser)

**Objective (from roadmap):** parse the normative Core v1 grammar with useful
diagnostics; every classified fixture has a deterministic expected result.

**Relative size:** ~4 work packages of a few days each, one of ~1–2 weeks
(WP1.4). Sequence is strict: each WP builds on the previous.

### WP1.1 — Scaffold and CI (small) ✅
- `starkc/` workspace: `cargo init`, module skeleton, `Span`/`SourceFile`
  types, CLI stub `starkc parse <file>` (prints "not implemented").
- GitHub Actions workflow per T9 (fixture job initially allowed to fail).
- README note in `starkc/` linking to spec, roadmap, and this plan.

*Done when:* CI is green on the skeleton; `starkc parse x.stark` runs and
reports a stub error with correct file/line plumbing.

### WP1.2 — Lexer (medium) ✅
Implement `01-Lexical-Grammar.md` in full:
- All token kinds; keywords vs reserved words vs identifiers (reserved words
  lex as distinct tokens so the parser can say "reserved for future use").
- Literals with suffixes and the strict underscore rule; nested block
  comments; raw strings; char escapes including `\u{...}`.
- Maximal munch; `>>` emitted as one token with parser-side splitting support
  (store as `Shr` carrying two joinable `Gt` spans).
- Lexer errors exactly as specified (unterminated string, invalid number,
  unexpected character) with spans.

*Tests:* unit tests per token class, including the spec's own literal
examples and pathological cases (`1__2`, `12_`, `0x_FF`, nested `/* /* */ */`,
unterminated raw string). Property test (optional): lex→concat round-trip
stability on generated token streams.

*Done when:* every `01-*` fixture and unit case lexes to the expected stream.

### WP1.3 — Fixture triage and conformance harness (small, high leverage) ✅
- Hand-review all 121 fixtures; record verdicts in
  `STARKLANG/tests/spec-fixtures/manifest.toml`:
  - `parse-pass` — must parse cleanly (most examples);
  - `parse-fail` — must be rejected by the lexer or parser (the two
    lifetime-annotation "future feature" blocks);
  - `semantic-error` — must parse, fails in Gate 2 (the `// Error:` borrow/
    type examples); the manifest records the expected `E####` code now,
    enforced later;
  - `lex-pass` — token-level examples from 01: must lex cleanly, are not
    programs (added during triage; the four `01-*` fixtures);
  - `notation` — API listings, type catalogs, and typing-rule metavariables
    excluded from parsing, per 06's Notation section.
- Parseable entries also record a *mode*: `program` (Item*) or `snippet`
  (block-body form, `(Item | Statement)* Expression?`) — a harness-only
  convenience for spec examples written at statement level.
- Harness: `starkc/tests/conformance.rs` — manifest schema validation and
  lex-level conformance always on; full verdict enforcement `#[ignore]`d
  (run with `--include-ignored`) until the parser exists, and emits a
  per-class summary table.
- Extend `tools/extract-spec-examples.sh` to fail if extraction produces a
  fixture set that diverges from the manifest's file list (detects spec edits
  that added/renumbered blocks without re-triage).

*Done when:* manifest covers 121/121 fixtures; harness runs in CI (still
red — the parser doesn't exist yet — but *deterministically* red).
**Done 2026-07-15.** Verdict census: parse-pass 67, semantic-error 18,
notation 30, lex-pass 4, parse-fail 2. Already green: lex-pass and parse-fail
(the lifetime blocks are rejected by the lexer). T10 spec fixes shipped with
the triage: three `{ ... }` ellipsis bodies made parseable (03 associated
types, 04 `?` example, 05 move example), a same-scope `let` redeclaration in
03's logical-operators example renamed, and 04's error-code table extended
(E0005–E0007, E0204, E0400/E0401, E0500) with unreachable code reclassified
from error E0300 to warning W0005 to match 04's own example.

### WP1.4 — Parser (large; the core of Gate 1) ✅
Recursive descent over `02-Syntax-Grammar.md`, built bottom-up so each layer
is testable before the next:

1. **Types** — primitives, paths + generic args (incl. associated-type
   bindings and shape-arg tolerance points left for Gate 4), arrays/slices,
   tuples, references, `fn` types, `Self` path rules.
2. **Expressions** — Pratt core with the normative precedence table
   (16 levels), non-associative comparisons/ranges enforced structurally,
   cast chains, unary incl. `&mut`, postfix (calls, index, field, tuple
   field, `?`), turbofish on path expressions, literals incl. tuple/unit/
   array-repeat, struct literals with the block-position restriction
   (context flag: "no struct literal at statement-head expression positions
   `if`/`while`/`match`/`for`").
3. **Statements & blocks** — `let` forms, `return`/`break`/`continue`, the
   trailing-expression rule (statement-first, expression-before-`}`).
4. **Items** — functions (receivers, generics), structs/enums (all variant
   forms), traits (signature-only, defaults, associated types), impls
   (inherent, trait-for, generic, blanket), `use` trees, `mod`, `const`,
   `type`.
5. **Patterns** — all forms incl. path patterns, struct patterns with
   shorthand, the identifier-vs-path note carried as an AST distinction for
   Gate 2 to resolve.
6. **Error recovery** — panic-mode sync at `;` and item keywords so multiple
   diagnostics emerge per file (the spec's error-recovery section is the
   requirement).

*Tests:* per-production unit tests + the WP1.3 harness turning green class by
class. Snapshot the AST (pretty-printed) for ~15 representative fixtures.
Add a `cargo-fuzz` target over the lexer+parser once the item grammar lands;
its gate is "no panics, no hangs" on arbitrary input — grammar *correctness*
remains owned by the fixtures, not the fuzzer.

*Done when:* all `parse-pass` fixtures parse, all `parse-fail` fixtures fail
with the right diagnostic, `notation` skipped — the conformance job is green.
**Done 2026-07-15.** Conformance: 91/91 (67 parse-pass, 18 semantic-error, 2
parse-fail, 4 lex-pass), 30 notation skipped; the CI fixture job is now
required-green. Extras beyond the letter of the plan: recursion-depth guard
(deep nesting is a diagnostic, not a stack overflow), tuple-field float
splitting (`pair.0.1`), `>=`/`>>=` splitting alongside `>>` in generic
argument position. The fuzz gate runs as a deterministic fixed-seed
pseudo-fuzz on the stable toolchain (`tests/robustness.rs`) instead of
nightly `cargo-fuzz` — same gate ("no panics, no hangs"), CI-runnable; it
immediately caught a parser hang (recovery at item keywords inside
`trait`/`impl` bodies looped without progress), which is fixed. T10 spec
fixes shipped with the parser: block-formed expression statements
(`Statement ::= BlockExpression ';'?` — the old grammar could not parse 04's
own `if` statement example), `(T)` grouping vs `(T,)` 1-tuple in type
position (mirroring the value side), and `PrimitiveType` as a leading
`PathSegment` (without it, `String::from(s)` was formally unparseable
because `String` lexes as a keyword).

### WP1.5 — Diagnostics polish + Gate 1 exit review (small) ✅
- Verify diagnostic rendering matches the normative format; wire `E`-codes
  where the spec defines them.
- Write the Gate 1 exit report (a short doc in `starkc/docs/`): fixtures
  status, list of spec defects found and fixed under T10, deviations (should
  be none), open questions handed to Gate 2.

**Done 2026-07-15.** Exit report: `starkc/docs/gate1-exit.md`. End-to-end
diagnostic-format goldens added (`starkc/tests/diag_format.rs`); they caught
and fixed a renderer misalignment (line-number `|` and carets off by one
column vs the gutter). E-codes: the spec assigns none to lexical/syntax
errors — recorded as Gate 2 open question #1 rather than inventing codes.
One last T10 fix: 02's Parsing Notes now specify the full `>`-splitting
family and the `pair.0.1` FLOAT split. **Gate 1 is closed** — see the exit
report for the ten-defect T10 ledger and six open questions handed to
Gate 2.

**Gate 1 exit == roadmap exit criteria + conformance job required-green in CI.**

*Expected side effect, planned for:* WP1.2/1.4 will surface spec bugs (the
first implementation always does). Budget roughly a third of Gate 1 effort
for T10 spec-fix loops; finding them is the point, not a delay.

---

## 2. Gate 2 — Core Semantic Checker (milestone detail)

Strict internal order — each milestone is a usable checkpoint:

- **M2.1 Name resolution + HIR.** Lower the ID-based AST (T6) into a
  desugared HIR here — resolution and everything after it operate on HIR,
  never on the parser AST. Scope trees per 04 §1 (lexical vs module
  resolution split), single-file first; `mod`/`use` resolution after.
- **M2.2 Type checking, no generics.** Local inference by unification within
  bodies; explicit signatures; control-flow typing rules (if/match/loop
  unification, `!` coercion); place-expression and mutability checks;
  definite assignment incl. the immutable-deferred-init rule.
- **M2.3 Generics + traits.** Bound checking, coherence/orphan rule,
  associated types, operator→trait desugaring (`Eq`/`Ord`/`Num`), method
  resolution with auto-borrow/auto-deref per 03. Monomorphization-style
  instantiation checking (no dictionary passing).
- **M2.4 Ownership & borrows.** Move tracking with drop flags, partial
  moves, reinitialization; then the lexical borrow checker (block-scoped
  `let` borrows, statement-scoped temporaries); Copy/Drop soundness rules.
  Borrow-carrying taint propagation comes *last* and only to the depth the
  prototype needs (returning `Option<&T>` from builtins).
- **M2.5 Conformance.** All `semantic-error` fixtures produce their recorded
  `E####`; a set of ~20 new valid programs (written for this gate) pass
  end-to-end analysis.

Deliberately deferred within Gate 2: match *usefulness* warnings, trait
default-method bodies' full checking, `pub use` re-export graphs beyond what
fixtures require.

**Done 2026-07-15.** M2.1–M2.5 are implemented in `starkc`; the exit evidence,
coverage inventory, and intentional deferrals are recorded in
`starkc/docs/gate2-exit.md`. Gate 2 is closed.

## 3. Gate 3 — Minimal Execution Path (milestone detail)

- **M3.1** Tree-walking interpreter (T11): values, calls, control flow,
  structs/enums/match, `?`, panic-as-abort semantics, drop order (observable
  via `Drop` impls).
- **M3.2** `core-min` builtins backed by Rust: `Vec`, `String`, `Option`/
  `Result` methods, `print`/`println`, `Range` iteration, `Box`, the handful
  of `std::io` calls the prototype needs. Behavioral requirements from 06
  (trap on OOB, `get` returns `None`, …) enforced and tested.
- **M3.3** `starkc run file.stark` + an examples directory that doubles as
  integration tests.

Exit: the Core-only parts of the Appendix-A-style pipeline (everything except
tensors) run correctly.

**Done 2026-07-15.** M3.1–M3.3 are implemented in `starkc`: typed-HIR
execution, the Rust-backed `core-min` runtime, `starkc run`, terminal IDE Run,
and executable integration examples. Exit evidence is recorded in
`starkc/docs/gate3-exit.md`. Gate 3 is closed.

## 4. Gate 4 — Tensor Front End + ONNX Import (decision detail)

Shape of the work (task breakdown when Gate 2 nears exit):
- Extend parser/checker behind an explicit `--extension tensor` flag: `Dim`/
  `DType` kinds, shape args, const index lists, `model` items — a Core-only
  build must reject them by name (extension conformance rule). In the type
  representation, tensor constructors are *extension-owned* (registered
  behind the flag), not variants baked into the Core `TyKind` — the Core
  checker must remain testable with no knowledge that tensors exist.
- Dim engine: polynomial normal form + equality; unification with dims;
  existential binding at `refine`.
- Op typing for exactly the §6 set the demo pipeline uses (matmul, permute,
  cast, elementwise+broadcast rule, reductions, reshape) — table-driven so
  ops are data, not code.
- `stark import`: ONNX protobuf read (via `prost`/existing ONNX crate),
  signature extraction, declaration generation, and the §7.3 verification
  rules incl. the variable-matches-only-dynamic-dims rule.
- Diagnostics per §9: constraint + provenance + suggested fix. This is
  product surface; budget accordingly.

## 5. Gate 5 — Go/No-Go Prototype (decision detail)

- One real CV model (ResNet-class or YOLO-class) through: `import` →
  generated signature → STARK pre/post-processing → generated Rust host
  program calling ONNX Runtime (T12) → single native binary.
- Defect corpus: the four roadmap defect classes, each as a one-line source
  mutation with its captured compile-time (or load-time) diagnostic —
  demo-ready.
- Measurement harness: reference-output tolerance check, artifact size,
  startup, peak RSS, steady-state latency vs a Python/ORT baseline —
  reported, not gated (per roadmap).
- Gate 6 decision memo template prepared at the start of Gate 5, so evidence
  is collected against it rather than assembled afterward.

---

## 6. Risks and Mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Grammar defects stall WP1.4 | High (expected) | T10 loop is budgeted; fixture harness localizes each break; spec fixes are small since the grammar is now internally consistent. |
| Borrow checker (M2.4) balloons | Medium | Lexical rule chosen for exactly this; borrow-carrying implemented shallowly (builtin returns only); anything deeper needs a roadmap-governed proposal. |
| Fixture renumbering on spec edits corrupts triage | Medium | WP1.3 manifest-vs-extraction consistency check; re-triage is part of any spec-editing commit (T10). |
| ONNX signature variety (dynamic dims, multiple outputs) exceeds spec | Medium | Gate 4 imports *one* representative model first; spec §7.3 gaps found there are fixed under T10 before generalizing. |
| Interpreter too slow for the demo's preprocessing | Low | Preprocessing is small; if needed, hot builtins (resize/normalize) become native builtins — allowed, they're library not language. |
| Scope creep via exciting extensions (meaning tags, units, audio) | High | Roadmap §4 non-goals + governance; this plan adds nothing beyond Gate 5; new work needs its own proposal. |
| Single-maintainer bus factor / motivation | Medium | Milestones are individually shippable and demoable (`parse` → `check` → `run` → `import` → binary); each WP ends with something visible. |

## 7. Immediate Next Actions (start of Gate 1)

1. WP1.1: create `starkc/` scaffold + CI (one sitting).
2. WP1.2: lexer with unit tests (first real code).
3. WP1.3: triage the 121 fixtures into `manifest.toml` (tedious, do early —
   it also constitutes a full close-read of the hardened spec).
4. WP1.4 step 1: type grammar + tests; proceed bottom-up.

## 8. Change Log
- v0.5 — WP1.5 done; **Gate 1 closed** (exit report in
  `starkc/docs/gate1-exit.md`). Renderer alignment fix behind new format
  goldens; final T10 fix to 02's Parsing Notes; syntax-error E-codes
  deferred to Gate 2 as an open question.
- v0.4 — WP1.4 done (parser; conformance 91/91, fixture CI job flipped to
  required). Amendment: the fuzz target is a stable-toolchain deterministic
  pseudo-fuzz in `starkc/tests/robustness.rs` rather than nightly
  `cargo-fuzz`; same no-panics/no-hangs gate, runs in ordinary CI. T10 fixes
  recorded under WP1.4.
- v0.3 — WP1.1–WP1.3 done (scaffold+CI, lexer, fixture triage). WP1.3
  additions over v0.2: fifth verdict `lex-pass` for the token-level `01-*`
  fixtures, and per-fixture parse modes (`program`/`snippet`) for spec
  examples written at statement level. T10 fixes recorded under WP1.3.
- v0.2 — architecture-review amendments: T6 arena/ID AST from day one with
  HIR scheduled at M2.1; T12 `ort`-first with the Cranelift/IREE evidence
  ladder; WP1.4 fuzz target; Gate 4 extension-owned type constructors.
- v0.1 — initial plan; Gate 1 at WP detail, Gates 2–5 at milestone/decision
  detail, standing decisions T1–T12.
