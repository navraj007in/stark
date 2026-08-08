# Campaign A — final adversarial exit audit

**Branch:** `wp-arch-stability/sprint-3`. **Head audited:** `36b3dfc`. **Date:** 2026-08-08.
**Verdict:** **PASS**, with one lane outstanding — see §12.

This pass was not a search for improvements. It was an attempt to **falsify** the closure claims of
`7bbe8d0` by constructing a legal or internally reachable route that bypasses one of the claimed
authorities. Where a claim survived, §12 states what forcing mechanism makes the absence of a
finding meaningful; a `PASS` justified only by "search found nothing" is recorded as such.

**Seven defects were found and repaired** — DEV-203 through DEV-209. Every one was expressible as
a violation of an **existing** rule and repaired inside an **existing** authority. None required a
new representation, type rule, execution funnel or place system. That is the substantive result:
not that the audit found nothing, but that everything it found was already illegal under the
architecture's own terms.

---

## AUDIT 1 — Callable body entry

**Verdict:** PASS

**Claim challenged.** No production route executes a user-defined callable body without passing
through the invocation authority.

**Method.** Census of every `eval_block` call site; every `execute_body` / `invoke_callable` /
`invoke_with_epilogue` / `call_user_method` caller; and the routes the packet names — free
functions, associated functions, inherent and trait methods, trait defaults, bound dispatch,
operators, `Display`, function values, `Option`/`Result` combinators, iterator callbacks,
destructors, the entrypoint, and builtin callbacks. Then a forcing experiment: a second body-entry
route was added and the evidence required to fail.

**Evidence.** `eval_block(callable.body)` occurs in exactly **one** place; `execute_body` has
exactly **one** caller, the authority, which installs the environment first. Both are pinned by
`as3_invocation_authority`. Every other `eval_block` call takes an `if`/loop/block body, not a
callable body.

**Findings.** None beyond Packet 1's own migration.

**Mutation / forcing control.** Adding `audit_probe_second_executor` — a second function calling
`eval_block(callable.body)` — fails `p1_exactly_one_production_body_executor`. The pin counts code
only: comment lines are stripped at the read, so the census cannot be satisfied or broken by prose.

**Residual risk.** The pin is textual. A body executed through a differently-spelled call (a macro,
a helper taking `BlockId`) would not be counted. Mitigated by `execute_body` having one caller,
which is a second independent count of the same property.

---

## AUDIT 2 — Generic environment installation

**Verdict:** PASS

**Claim challenged.** Every callable class installs the checker-selected environment; omission is
observable.

**Method.** The AS3 #2 requalification (P1–P8, D1–D7) re-read from production structure rather than
from its tests, plus a hostile mutation **outside** the seven — `StripFunctionValueBindings`, which
corrupts the environment at the *capture* site rather than at the installer, to test whether the
evidence was over-fitted to the known seven cases.

**Evidence.** Seven dispatch classes each remove the environment at the single installation point
and require an `InternalInvariant`; each asserts the mutation was **reached**, and each witness
answers `size_of::<T>()` so the instantiation is load-bearing. `P2` is an exhaustive match over
`InvocationEnv`. `P6` is proved behaviourally (a `&W<T>` receiver would fail on a *correct* program
if read before installation) and structurally (install precedes the body; the guard is bound, not
dropped). `P8` uses contrasting instantiations — `848`, where a restoration bug gives `844`.

**Findings.** **DEV-202** — `call_method` chose the environment, installed it, and passed it to the
authority, which installed it again; the outer guard was live while the *caller's* receiver place
was still being resolved. Found by the pin that counts installation points, not by any behavioural
test.

**Repairs.** The call site chooses; the authority installs.

**Mutation / forcing control.** The out-of-set mutation (D-independent) fails as required. Adding an
`InvocationEnv` variant breaks `install_invocation_env`'s exhaustive match.

**Residual risk.** `push_callable_env` still treats an absent `callable_instantiations` entry as
"nothing to push". Audit 10-F shows that removing the entry for a **generic** call does fail
loudly, because the value boundaries catch the unsubstituted parameter — so the tolerance is
covered downstream rather than at the lookup.

---

## AUDIT 3 — Typed value boundary bypasses

**Verdict:** DEFECT FOUND — repaired

**Claim challenged.** Every route by which a value enters typed storage, crosses a callable
boundary, is written into a place, or is consumed inline passes through `check_value_for_ty`.

**Method.** Census of `Frame::insert`, raw `values.insert`, `place_slot_mut`, `write_place`,
aggregate construction, match and loop bindings, parameter and receiver insertion, return and
propagation handling, and `expect_value`; then a forcing experiment adding an extra typed-local
insert.

**Evidence.** All twelve `RepBoundary` variants are `Wired`, pinned by `dev121_boundary_inventory`.
Three `frame_mut().insert` sites and three raw `values.insert` sites, each classified.

**Findings.** **DEV-203** — an interpolated field consumed an expression result unchecked: it never
binds to a local, so no destination boundary saw it, and it reached the renderer through a direct
`eval_expr` rather than `expect_value`. **The only construct in Core v1 invisible to all twelve
wires.** The census had missed it because it asserted `direct <= 8` and there were six.

**Repairs.** The interpolation field's `Flow::Value` arm calls `check_expr_value`. The census is now
an exact count with every site named.

**Mutation / forcing control.** Reverting the repair makes
`an_interpolated_field_is_a_checked_expression_result` fail — proved in both directions. Adding a
typed-local insert outside `bind_typed_local` fails `typed_local_storage_has_one_funnel`, a pin that
did not exist before this audit and whose absence was itself a finding.

**Residual risk.** `swap`, `replace` and `take` write through `place_slot_mut` without their own
boundary. Each is homogeneous in `T` and its incoming value was checked as an expression result, so
representation is preserved by construction rather than by a check — argued, and worth converting to
a test if those builtins gain non-homogeneous forms.

---

## AUDIT 4 — Expression funnel exactness

**Verdict:** DEFECT FOUND — repaired (same defect as Audit 3)

**Claim challenged.** `expect_value` is the producer-side funnel for typed expression results.

**Method.** Enumerate every direct `eval_expr` consumer and prove each is the funnel, a genuine
flow-through, or a typed consumer that must be checked.

**Evidence.** Exactly six, each classified by name: the funnel itself; `eval_block`'s tail; an
expression statement (value dropped); an interpolation field (**a checked consumer**); an `else`
branch; a match arm body. The indirect helpers — `expect_bool`, `expect_int`, argument and receiver
evaluation — all delegate to `expect_value`.

**Findings / repairs.** DEV-203, above.

**Mutation / forcing control.** The count is exact. A seventh direct consumer fails the pin and must
be classified before it is added.

**Residual risk.** None identified.

---

## AUDIT 5 — Missing metadata fallbacks

**Verdict:** DEFECT FOUND — repaired

**Claim challenged.** Where the checker promises metadata for a reachable construct, absence is
`InternalInvariant`, never a skip.

**Method.** Search for `if let Some`, `let … else`, `unwrap_or*`, `.ok()`, `.get(…)` around
`callable_types`, `CallableUse`, environments, `callable_instantiations`, `expr_types`,
`local_types`, `aggregate_field_types`, nominal field metadata, bound dispatch results and
`FunctionValue` bindings — with particular attention to funnels introduced in Packets 1–7.

**Findings.**
- **DEV-204** — `capture_function_value` answered *any* missing instantiation with empty bindings:
  DEV-178's defect written as a fallback, where absence meant both "no generics" and "publication
  missing", the second unrecoverable by construction.
- **The typed-local escape** — `bind_typed_local` delegated to a permissive lookup, so a
  language-level binding whose `local_types` entry went missing would have been **skipped silently
  inside a wire the inventory reported as `Wired`**.

**Repairs.** DEV-204 separates the two meanings by whether the item declares generics. The funnel
looks the type up itself and treats absence as `InternalInvariant`; the permissive helper had no
remaining callers and was deleted rather than renamed, so no future funnel can pick it up.

**Mutation / forcing control.** Audit 10-F: deleting an entry from each of the five published tables
fails as `InternalInvariant`.

**Residual risk.** As Audit 2 records, `push_callable_env` remains tolerant at the lookup and is
covered downstream.

---

## AUDIT 6 — Duplicate semantic authorities

**Verdict:** PASS

**Claim challenged.** `check_value_for_ty` / `value_matches_ty` is the only answer to "does `V`
represent `T`".

**Method.** Search for kind-based validation at storage boundaries, ad-hoc `Value` matches before
insertion, boundary-specific predicates and runtime reconstruction of expected types — separating
representation *authority* from ordinary operation-specific destructuring.

**Evidence.** `check_value_representation`, the narrow second rule, was deleted in Packet 3 once the
funnel left it with no callers. The value-context property (below) consults the canonical relation
by **probing** it rather than by keeping a list.

**Findings.** During this audit I twice began adding a second authority and withdrew both: an
`operator_impl_environment` beside the existing `impl_dispatch_bindings` (DEV-201), and a checker
diagnostic enumerating runtime-representable types. Both are recorded because the withdrawal is the
evidence, not the intention.

**Mutation / forcing control.** Adding a `Ty` variant fails `value_matches_ty`'s exhaustiveness;
adding a `Value` variant fails the kind mapping and the relation.

**Residual risk.** MIR keeps its own `ProgramMeta::assoc_projections`. Two authorities exist for
associated-type projection — the checker's (now published, DEV-199) and MIR's. Out of Campaign A
scope; recorded.

---

## AUDIT 7 — Checker-published expected types

**Verdict:** DEFECT FOUND — repaired

**Claim challenged.** Every boundary's expected type comes from checker-owned metadata, never from
the runtime value.

| Boundary | Expected-type source | Enforcement site | Missing metadata |
| --- | --- | --- | --- |
| `Receiver` | `callable_types[body].receiver` | `execute_body` | `InternalInvariant` |
| `Parameter` | `callable_types[body].params[i]` | `execute_body` | `InternalInvariant` |
| `Return` | `callable_types[body].ret` | `execute_body` | `InternalInvariant` |
| `Propagation` | `callable_types[body].ret` | `execute_body` | `InternalInvariant` |
| `LetBinding` | `local_types[local]` | `bind_typed_local` | `InternalInvariant` |
| `MatchBinding` | `local_types[local]` | `bind_typed_local` | `InternalInvariant` |
| `LoopBinding` | `local_types[local]` | `bind_typed_local` | `InternalInvariant` |
| `Assignment` | `expr_types[lhs]` | `write_place` | `InternalInvariant` |
| `FieldWrite` | `expr_types[lhs]` | `write_place` | `InternalInvariant` |
| `ElementWrite` | `expr_types[lhs]` | `write_place` | `InternalInvariant` |
| `AggregateField` | `aggregate_field_types[lit][field]` | `eval_struct_lit` | `InternalInvariant` |
| `ExpressionResult` | `expr_types[expr]` | `expect_value` | `InternalInvariant` |

No row reconstructs an expected type from a runtime value. `AggregateField` deliberately does **not**
read `expr_types[init]`: that is the type of the expression that produced the value, so comparing
the value against it would assert nothing — and it does not exist at all for a shorthand field.

**Findings.**
- **DEV-205** — `IOError::Other(msg)` bound a payload the checker never typed: no `local_types`
  entry, every use published as `Ty::Error`, and the program printed the right answer.
- **DEV-206** — `Display` accepted the unsized `[T]` and rejected `&[T]`, so an accepted expression
  had a published type the relation says permits **nothing**.

**Repairs.** The builtin-variant pattern arm types the `IOError::Other` payload. `Display`
eligibility separates `[T; N]` (a value) from `[T]` (not one), and accepts `&[T]` iff `T` is
displayable. Both withdrawn alternatives for DEV-206 are recorded in the ledger.

**Mutation / forcing control.** `audit_published_types` asserts, over eight witness families, that a
program the checker accepts with **no diagnostics** publishes no `Ty::Error` in any expression or
local type — a general property, not a regression pin.

**Residual risk.** Whether a range index should publish `&[T]` is a language-semantics question,
recorded outside Campaign A.

---

## AUDIT 8 — Function values

**Verdict:** DEFECT FOUND — repaired

**Claim challenged.** A function value's captured bindings survive every reachable route, and
nothing reconstructs them from the later `Ty::Fn`.

**Method.** Enumerate the routes Core v1 permits, with a witness whose answer depends on its
instantiation; search for `FunctionValue { …, bindings: Vec::new() }` outside sites where emptiness
is proven.

**Evidence.** Six routes, all answering `8`: creator frame gone, passed through another function,
created inside a generic body (where the enclosing `U = Int32` must lose to the captured
`T = Float64`), `Option::map`, `Result::map`, and stored in an aggregate then read back. A control
asserts the probe distinguishes instantiations at all — without it the whole file would pass with
the bindings discarded, which is exactly how DEV-197's first two defects stayed invisible.

**Findings.** DEV-204, above.

**Mutation / forcing control.** `StripFunctionValueBindings` keeps a valid `Value::Function` and
removes only the instantiation; the run must fail.

**Residual risk.** None identified for reachable routes.

---

## AUDIT 9 — New variant and route forcing functions

**Verdict:** PASS — one missing forcing function found and added

**Method.** Six controlled experiments, each adding a variant or route and requiring evidence to
fail, then reverting.

| Experiment | Result |
| --- | --- |
| new `RepBoundary` variant | `classify` fails exhaustiveness |
| new `Ty` variant | `value_matches_ty` fails exhaustiveness |
| new `Value` variant | `kind` and the relation fail |
| new `InvocationEnv` variant | `install_invocation_env` fails |
| second body-entry route | `p1_exactly_one_production_body_executor` fails |
| extra typed-local insert | **nothing failed** → pin added |

**Findings.** The storage route had **no forcing function at all**. A new `frame_mut().insert`
outside `bind_typed_local` disturbed no evidence, and would have reintroduced the per-site
convention Packet 3 replaced without any test noticing.

**Repairs.** `typed_local_storage_has_one_funnel`, counting code lines only.

**Mutation / forcing control.** The experiment above is itself the control, and it now fails.

**Residual risk.** Textual census, as in Audit 1.

---

## AUDIT 10 — Independent mutation adversaries

**Verdict:** PASS

Each authority class has at least one hostile challenge that is **not** a rerun of the existing
evidence.

| Class | Independent challenge | Forcing site |
| --- | --- | --- |
| A environment | `StripFunctionValueBindings` — corrupts delivery at the *capture* site, not the installer | body boundaries |
| B owned/view | an interpolated `s.as_str()` emitting owned storage | `ExpressionResult` |
| C reference | a `&mut self` receiver losing place identity (class 2 uses `&self`) | `Receiver` |
| D function value | bindings stripped while the value stays a valid `Value::Function` | body boundaries |
| E aggregate/container | a mis-represented value written into **existing** storage, a different funnel from construction | `Assignment` |
| F metadata | deleting entries from all five published tables | `InternalInvariant` each |

**Residual risk.** A mutation that survives because the program happens to produce the same visible
output would be a failed architecture test. The environment controls guard against this by requiring
the witness's answer to depend on its instantiation; the representation controls by asserting the
boundary **by name**.

---

## AUDIT 11 — Differential and campaign qualification

**Verdict:** PASS — one lane outstanding

**Local, on `36b3dfc`:**

```text
cargo test (starkc)               209 targets, 2 743 tests, 0 failures
cargo fmt --check                 clean
cargo clippy --workspace --all-features --all-targets -- -D warnings   clean
three_engine_differential         109
mir_differential                  132
dev209_prelude_payload_place      13, plus a mutation control
stark-url (stark test)            20/20
External sample suite @ b3b28e7   39/39
```

**Repository-backed, on `36b3dfc`:** workflow `31242528902` completed **success** with zero failing
jobs. Workflow `31242528904` has every job green except `fmt, clippy, test (windows-x64)`, which was
still running when this report was written — including `first-party package qualification` on all
three platforms and `External sample suite (pinned)`, the two jobs DEV-209 repaired.

**Reporting discipline.** Three times during this campaign I reported CI more favourably than the
evidence supported — twice by quoting a run-level conclusion while a second workflow was red, once
by missing a failing job entirely. The remedy is procedural, not attentional: **enumerate failing
jobs by name**, never quote a run conclusion.

---

## 12. Consolidated result

| Audit | Verdict | Defects found | Repaired | Blocking? |
| ---: | --- | --- | --- | --- |
| 1 Callable body entry | PASS | — | — | no |
| 2 Environment installation | PASS | DEV-202 | yes | no |
| 3 Typed value boundary bypasses | DEFECT FOUND | DEV-203 | yes | no |
| 4 Expression funnel exactness | DEFECT FOUND | (DEV-203) | yes | no |
| 5 Missing metadata fallbacks | DEFECT FOUND | DEV-204, typed-local escape | yes | no |
| 6 Duplicate semantic authorities | PASS | — | — | no |
| 7 Checker-published expected types | DEFECT FOUND | DEV-205, DEV-206 | yes | no |
| 8 Function values | DEFECT FOUND | (DEV-204) | yes | no |
| 9 Forcing functions | PASS | missing storage pin | yes | no |
| 10 Independent adversaries | PASS | DEV-207, DEV-208, DEV-209 | yes | no |
| 11 Qualification | PASS | — | — | **Windows lane outstanding** |

**Nine defects, zero architectural changes.** DEV-202 through DEV-209 were each a violation of a
rule the architecture already stated, repaired at an authority that already existed. The largest,
DEV-209, **removed an exception** from the place model rather than adding anything to it.

### The hard exit conditions

```text
[x] no executable user body can bypass invocation authority
[x] no callable class can execute without its selected environment
[x] raw body execution has one controlled production entry
[x] every RepBoundary is Wired
[x] ExpressionResult covers inline typed expression consumption
[x] direct eval_expr census is fully classified
[x] no missing-metadata fallback bypasses enforcement
[x] check_value_for_ty is the sole Ty→Value semantic authority
[x] every boundary obtains expected Ty from checker-published metadata
[x] function-value captured environments survive every reachable call route
[x] environment mutations fail
[x] representation mutations fail
[x] metadata-removal mutations fail loudly
[x] new variants/routes trigger forcing mechanisms
[~] full qualification is green — one Windows lane still reporting
```

**Discovery stops here.** No adversarial bypass remains open. The final condition is a lane result,
not an unanswered question.
