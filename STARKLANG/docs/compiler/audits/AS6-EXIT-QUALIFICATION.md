# AS6 exit qualification — extension quarantine

**Verdict: PASS.** Four recorded residue entries (across three files) and three recorded limits, all enumerated below.

**Head:** `6050efa` (code) / `89b400b` (governance record on top; no code change).
**Date:** 2026-08-08. **Branch:** `wp-arch-stability/sprint-4`.
**Authority:** Campaign B approved for execution 2026-08-08 —
`WP-ARCHITECTURE-STABILIZATION.md` §1. That approval ratified the landed packets as
*execution*; this document is the qualification evidence it deliberately did **not** substitute
for.

---

## 1. What AS6 was asked to do

> Quarantine extension-specific compiler knowledge: move extension-owned names, type rules, methods
> and diagnostics behind sealed internal tensor modules selected by the existing per-session
> `LanguageOptions`, without introducing a generic public abstraction.

Fifteen commits, `46ae2ec` … `6050efa`. Net shape across `src/` and `tests/`: **3,643 insertions,
2,477 deletions over 16 files** — and **zero changes to any test fixture or expected output**, which
is the central fact of criterion 3 below.

## 2. Exit criteria

### Criterion 1 — Core-only sessions load no tensor-owned name or semantic rule

**PASS.** Pinned in both directions, which is what this packet's checkpoint evidence demanded:

> A quarantine that suppresses tensor semantics passes the first test and fails the second; one that
> leaks passes the second and fails the first. Neither is visible to `cargo check`.

| Evidence | What it proves |
| --- | --- |
| `as6_core_session_isolation.rs`, 4 tests | every case is a **pair** — a Core-only session must not know the name, *and* a tensor session must answer differently |
| `c91_extension_isolation.rs`, 5 tests | the pre-existing C9.1 per-session isolation suite, retained unchanged |
| `as6_core_module_vocabulary.rs::surfaces_without_tensor_references_keep_none` | the structural half: the three surfaces the inventory found clean stay at **zero** |

Tensor references in production code, by surface:

```text
lexer          0        formatter    6     (printing ItemKind::Model, via the extension's keyword)
diagnostics    0        lsp          4     (enablement + SymbolKind mapping)
parser        38        resolver    25
checker      529
```

`lexer.rs`, `diag.rs` and `format_syntax.rs` import nothing from `extensions::tensor`.

### Criterion 2 — no open-ended tensor spelling tables or method catalogues in central Core modules

**PASS — and this was the criterion that actually failed twice before it passed.**

Three catalogues and six vocabulary tables were moved:

| Was in Core | Now |
| --- | --- |
| `resolve.rs` — 33-arm builtin name→`Builtin` map and its membership test | `extensions::tensor::builtins` (fe80129) |
| `hir::Builtin` — 33 `Tensor*` variants | one sealed `Builtin::Tensor(TensorBuiltin)` (fe80129, 33cb0a7) |
| `typecheck.rs` — `TENSOR_OPS` and the five rule types | `extensions::tensor::rules` (62ef6b0) |
| `parser.rs` — 21 spellings at 6 sites, incl. the 15-name reserved-type table | `extensions::tensor::syntax` (5190d1b) |
| `resolve.rs` — 15-name `extension_reserved_name` table | `syntax::extension_type_name` (6050efa) |
| `typecheck.rs` — `Dim`/`DType`/`Device`, `Cpu`/`Cuda`, value-range classifiers **and the three diagnostic phrases that recite them** | `syntax::{tensor_param_kind, device_constructor, value_range_state}` + `*_EXPECTATION` (6050efa) |
| `typecheck.rs` — 4-arm type-constructor table; `deploy/lower.rs` — a **third** copy; `deploy/lower.rs` `dtype_by_name` — a **fourth** copy of the element-type spellings | `syntax::{tensor_type_constructor, dtype_by_name}` (6050efa) |

**The checker's 529 remaining references contain no table.** They are, exhaustively: boundary glue
(`check_tensor_op`, `check_tensor_builtin_call`, `check_tensor_method_call`, `check_tensor_refine`,
`check_model_def`, `check_model_method_call` — each now validating call form and delegating its
rules), written-type-syntax conversion (`build_tensor_type`, `build_shape`, `build_refine_shape`,
`build_device`, `build_cuda_device`, `tensor_dtype`, `tensor_arity`), generic-parameter scope
management (`enter`/`exit_tensor_param_scope`, `as_tensor_param`), unification bridging
(`unify_tensor_types`, `emit_tensor_unify_error`, `ground_tensor_dims`) and the two service
implementations (`tensor_state`, `tensor_error`).

**Pinned by `as6_core_module_vocabulary.rs::core_front_end_modules_do_not_spell_extension_names`**
over sixteen files, checking *string literals* rather than reference counts, with the vocabulary
read back out of `extensions/tensor/syntax.rs` so a name added there is automatically a name Core
may not spell.

**Recorded residue — four entries, each with a reason, held as a set-equality ledger:**

```text
ast.rs "Float16"/"BFloat16"   Primitive::name renders a CLOSED Core enum. Adding a dtype means
                              adding a Primitive variant, which the compiler forces to be handled
                              everywhere — not a table that grows silently. Sealing it is the same
                              cut fe80129 made for hir::Builtin and is wider than AS6 scoped.
deploy/ir.rs "TensorAny"      Display for DeployTy, the deployment IR's own closed enum.
deploy/emit.rs "Tensor"       the GENERATED RUST host's type name. It coincides with the STARK
                              spelling without being one.
```

The ledger is set equality, not a skip-list: a new violation fails, **and so does removing an
accepted entry without updating the list**.

### Criterion 3 — tensor behaviour and ONNX verification unchanged for their documented scope

**PASS, on the strongest available evidence: zero fixture changes.**

AS6 changed **no test fixture and no expected output** in `starkc/tests/fixtures` or
`STARKLANG/tests/spec-fixtures`. Behaviour is not "unchanged as far as we tested" — the suites
that define tensor and deployment behaviour were never edited to accommodate the refactor.

Green at qualification: `gate4_tensor`, `gate4_onnx`, `gate5_lower`, `gate5_emit`, `gate7_deploy`,
`source_extensions`, `conformance`, `span_integrity`, `--lib` (564) — **629 tests, 0 failed, 1
ignored, `TESTS_EXIT=0`**.

Two moves were additionally verified by **identity rather than by compilation**: the 1,276-line
tensor rule block and `check_tensor_refine` were normalised (comments stripped, rustfmt line breaks
and trailing commas collapsed, the `self.X`→`cx.X` rename reversed) and compared byte-for-byte
against `HEAD` — `22925 == 22925` and `1690 == 1690`, identical. "It still compiles" would not have
caught a dropped match arm in 1,276 lines.

**Diagnostic order is preserved, including at the two sites where it was at risk.** `check_expr` in
`check_model_method_call` sat inside a `zip` over instantiated ports, and in `check_model_def` the
port-type conversion can itself emit. Both were **staged rather than hoisted**: Core drives the loop
and calls the extension rule per item, reproducing the original interleaving exactly.

### Criterion 4 — no public extension/plugin/provider API introduced

**PASS.** `extensions/tensor/check.rs` and `extensions/tensor/syntax.rs` export **zero** `pub`
items; everything is `pub(crate)`. Pinned by
`as6_core_module_vocabulary.rs::as6_added_no_public_extension_api`.

The boundary is enforced by the compiler, not by convention: `TypeChecker`'s fields and methods are
private to the `typecheck` module, so `extensions::tensor::check` can reach **only** what
`TensorCheckCtx` names.

### Criterion 5 — Part B generic artifact-provider work remains blocked

**PASS.** `WP-C9.1-EXTENSION-ISOLATION.md` still records "Part B remains blocked", C9.3's
independent evidence has not appeared, and AS6 introduced no artifact-provider, plugin-registry or
extension-registration code. The `extensions::tensor` modules are internal and tensor-specific by
construction.

## 3. The work-package deliverable that was missing

AS6's work list included one item no implementation packet produced:

> Add dependency/lint tests preventing new tensor imports in designated Core-only modules.

Exit qualification found it absent. `starkc/tests/as6_core_module_vocabulary.rs` (3 tests, 16 files)
is it, delivered in `6050efa`.

## 4. The finding

**Criterion 2 is the only one of the five with no behavioural signature, and that is why it failed
twice after its surfaces had been declared clean.**

Criteria 1 and 3 are pinned in both directions on every run. Criterion 2 is pinned by nothing that
executes. A spelling table therefore does not regress in one visible commit — it returns one arm at
a time, because the next contributor adds `"Float8"` to a match in `parse_type` where the
surrounding code already is, and every test still passes.

That is precisely what had happened. `resolve.rs` carried a 15-name table through three packets that
each reported their surface clean; the census then found the type-constructor spellings in **three**
places and the element-type spellings in **four**. A shell census run during this qualification
first reported *zero* violations across eleven files — the loop passed an unsplit shell variable, so
`grep` matched nothing and empty output read as success. The executable check found the first
violation immediately.

**Carry into AS7:** a structural criterion needs a structural check committed *with* the cut — not a
procedure, not a reviewer's grep. AS7's criterion 2, "dependency direction between submodules is
documented and cycle-free", has exactly the same shape and no behavioural signature. Its executable
check should exist before the modularisation begins, not after.

## 5. Recorded limits

1. **The `ast::Primitive` residue is real and deferred, not resolved.** The extension's two element
   types are variants of a Core enum. Sealing them is the `hir::Builtin` cut applied to
   `Primitive`, touching every `match` in the checker, interpreter, MIR and backends. Out of AS6's
   scope; recorded in the ledger rather than hidden.
2. **`src/deploy/` is extension-only code living outside `src/extensions/`.** Its *location* is
   residue, not its ownership. Relocating the deployment subsystem was not in scope.
3. **`grep`-style reference counts are not evidence for this packet and are not quoted as such.**
   Moving 21 spellings out of `parser.rs` *raised* its match count from 225 to 227, because
   `tensor_syntax::` is itself a match.

## 6. Evidence index

```text
local, head 6050efa      629 tests, 0 failed, 1 ignored, TESTS_EXIT=0, across:
                         --lib, as6_core_session_isolation, as6_core_module_vocabulary,
                         c91_extension_isolation, gate4_tensor, gate4_onnx, source_extensions,
                         conformance, span_integrity, gate5_lower, gate5_emit, gate7_deploy
                         cargo fmt --check clean

CI, 5190d1b (4C)         28/28 jobs success, zero failing
CI, 6050efa (4D)         CI 24/24 success, C7.8 Native Capabilities 4/4 success, ZERO failing
                         jobs. Includes `fmt, clippy, test` on linux-x64, macos-arm64 AND
                         windows-x64; `spec fixture conformance`; C6.4 and C6.5 corpus replay,
                         tier-1 agreement and mutation controls; the pinned external sample
                         suite; DEV-160 under Miri; first-party package qualification and
                         release package smoke on all three platforms.
```

## 7. Verdict

**AS6 PASSES its exit criteria. Unconditional — CI on `6050efa` is fully green on all three
Tier-1 platforms, 28 jobs across two workflows, zero failing.**

Closure propagation follows the order fixed in `AS6-PACKET-4-PLAN.md` — `COMPILER-STATE.md` first,
then `WP-ARCHITECTURE-STABILIZATION.md`, then `CLAUDE.md`, `AGENTS.md` and the remaining downstream
documents.

AS7 may open.
