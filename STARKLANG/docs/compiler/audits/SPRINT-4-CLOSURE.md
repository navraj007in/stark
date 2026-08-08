# Sprint 4 — closure report

**Branch:** `wp-arch-stability/sprint-3`. **Heads:** `04f0391`, `d047e13`, `f55bcc4`, `627cadf`.
**Date:** 2026-08-08.

```text
Phase 1 — AS4 consolidation                PASS
Phase 2 — AS4 adversarial qualification    PASS
Phase 3 — Campaign A convergence           PASS (local); CI pending on the exact head
Phase 4 — formal closure                   this document
```

---

## 1. Commits

| Commit | Content |
| --- | --- |
| `04f0391` | AS4 destruction authority; DEV-210, DEV-211; the hostile-combination suite. **CI green, both workflows** |
| `d047e13` | DEV-212, both engines. **CI red** — see §3 |
| `f55bcc4` | DEV-212 follow-up: removes the abandoned edit that made `d047e13` red |
| `627cadf` | Phase 3 qualification evidence, the AS4 re-census, and the deferment statements |

---

## 2. Defects discovered, rules violated, repairs

**DEV-210 — the borrow checker identified `Drop` by spelling.** It asked whether the written trait
name `.ends_with("Drop")`, so `impl MyDrop for S` gave `S` a destructor it did not have and a legal
partial move out of one of its fields was refused. **Valid Core rejected because a user trait's name
ended in four particular letters.** Violates CD-379 (a core trait is satisfied by *resolved
identity*). Repaired by publishing `nominals_with_destructor` — the set `copy_eligible_types`
already computed correctly, by identity, and kept private.

**DEV-211 — a matched component could move out of a `Drop` nominal.** Violates OWN-PARTIAL-001
("moving a field from a type that implements `Drop` is prohibited, because its destructor requires
the complete value"). Both engines agreed, making it a front-end conformance defect: the checker had
the rule for struct fields and never applied it to a matched component.

**DEV-212 — a `match` skipped a `Drop` nominal's own destructor**, even with nothing moving out.
Violates PAT-DROP-001's destroyed-exactly-once guarantee. Repaired in HIR (`drop_unbound`) and MIR
(`lower_enum_match`).

---

## 3. Four wrong turns on DEV-212, and what they share

This is the most transferable part of the sprint.

| # | Attempt | Disproved by |
| ---: | --- | --- |
| 1 | Destroy the value whole in `drop_unbound` | **Double drop** — the guard preceded the `Binding` arm and destroyed components already moved into their bindings |
| 2 | Fix MIR in `lower_arms_consuming` | An instrumented probe printed **nothing**: enum matches take their own route, `lower_enum_match` → `consume_variant_payload` |
| 3 | Guard with `ty_has_user_drop` | **Fourteen `MIR-0007` failures** — it means "contains a destructor *anywhere*, including a nested payload", so it fired for `Option<Droppable>` |
| 4 | Leave attempt 2's edit in place after moving the fix | **CI**, on `d047e13` — `droppable_array_pattern_agrees` and the frozen corpus |

The first three are one mistake: **reaching for a predicate that was nearby instead of establishing
which types reach the site.** `ty_has_user_drop` versus `type_has_drop_impl` is the same trap as
`ends_with("Drop")` versus resolved identity, one defect apart.

That is the argument for AS4's *second* requirement. One authority per property was already
satisfied for six of seven families. What failed twice in a single sprint was **naming** — the
compiler holds several predicates whose names begin with the same word and answer different
questions, and both times the wrong one compiled, ran, and looked correct.

The fourth is not a reasoning error but a process one: a fix was moved and the abandoned version was
left behind. It is the only one of the four that reached CI, and the only one no amount of local
reasoning would have caught.

**Attempt 2 deserves its own note.** The probe printing *nothing* is what revealed a second
match-lowering path. No test could have found that: the code compiled, ran, and changed no
behaviour. The question that found it was not "is my logic right" but "did my code execute at all".

---

## 4. Forcing controls added

- `as4_destructor_authority` — identity, not spelling: `MyDrop` and `DropLike` do not count, the
  published set is asserted directly, and a real destructor still refuses the move (the control that
  stops the suite passing against a borrow checker that simply stopped checking).
- `as4_hostile_combinations` — twelve property combinations, each run through **HIR and MIR with
  agreement required**, because Campaign A has repeatedly found the oracle to be the wrong engine.
- The DEV-211/DEV-212 pair, which together distinguish *"the destructor runs"* from *"a `Drop` enum
  cannot be matched at all"* — either alone would be satisfied by the wrong repair.

---

## 5. Qualification results

Local, at `f55bcc4`: `--lib` 558 · `mir_differential` 132 · `native_c6_1_ownership` 24 · AS4 suites
18 · **first-party packages 53/0** · **external sample suite 39/39** at pinned `b3b28e7` · fmt and
clippy clean.

The two application suites carried the weight: DEV-211 is a new **rejection**, and a refusal cannot
be qualified by compiler tests alone — it changes what compiles, so the only question that matters
is whether a real program relied on the old acceptance.

---

## 6. Known explicit limitations

Unchanged, and stated rather than promoted (exit report §4.0a): generic user `Drop` in the HIR
oracle; native drop glue for `Vec`/`Box` of a custom-destruction element; AS4 #4's structurally
unavailable lanes. Sprint 4 verified the first two have not silently widened into another engine.

---

## 7. Residual risk

- **CI on the exact head is outstanding.** `04f0391` is green; `f55bcc4` and `627cadf` are running.
  Campaign A's exit rule names this explicitly, and this campaign has produced a red lane after a
  green local run more than once.
- **The census pins are textual.** A body executed through a differently-spelled call would not be
  counted; mitigated by `execute_body` having exactly one caller, an independent count of the same
  property.
- **The boundaries live in the HIR oracle only.** MIR and native have no equivalent, so a
  representation defect there is caught by the differential comparing answers rather than by an
  invariant — which is why DEV-201 surfaced as a *missed trap*.

---

## 8. Verdict

Every Campaign A exit condition is satisfied by work that has landed, **except the last**:

```text
[ ] exact pushed head CI is green
```

That is a lane result, not an open question — and it is deliberately not being pre-empted. The
final verdict is issued in `CAMPAIGN-A-EXIT-REPORT.md` when CI on `627cadf` reports, by
**enumerating failing jobs by name** rather than reading a run-level conclusion.

If it reports clean:

```text
CAMPAIGN A — PASS
```

If it does not, the finding is repaired under §13's policy — ordinary implementation work, not an
architecture reopening — and the verdict follows the repair.
