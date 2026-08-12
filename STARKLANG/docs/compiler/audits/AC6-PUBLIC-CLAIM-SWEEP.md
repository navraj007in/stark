# AC6 — public architecture claim, verification sweep

**Packet:** WP-ARCH-CLOSE AC6, under CD-400. **Status: COMPLETE — one overstatement found and
corrected.**

AC6 was scoped as a **verification sweep with a reconciliation duty**, not a rewrite: EI6 (CD-392)
had already measured the public copy on 2026-08-09 and its corrections had landed. The live
obligation was the second half — *reconcile the published wording with AC4's findings, because AC4
can demote an authority and make today's accurate copy overstated tomorrow.*

It did.

---

## 1. The prohibited claim is still absent

```text
surfaces swept          README.md (891)   website/src/content.ts (251)
                        CLAUDE.md (235)   AGENTS.md (136)
                        ROADMAP.md (1603) STARKLANG/docs/ROADMAP.md (287)

"three independent implementations" and equivalents      ZERO occurrences
```

The single grep hit is a **prohibition comment** in `content.ts` telling a future editor not to
restore the claim — not the claim itself. EI6's rule holds.

## 2. The finding — a qualifying clause was dropped in the consuming change

EI6 approved this wording:

> *"…those rules are listed in a public register and checked separately — against the
> specification, by mutation testing, by executable gates, **and by recorded residuals where no
> control yet exists**."*

The website shipped:

> *"Those rules are listed in a public register and checked separately."*

**The clause that made the claim true was truncated.** With it, the sentence says *some of these are
checked and the rest are recorded as unchecked*. Without it, the sentence says *all of these are
checked* — and the paragraph names three examples.

**AC4 measured those three:**

```text
Copy eligibility          c61f_structural_copy, 13 tests, killed AS8-MUT-009/010/011   CHECKED
destructor eligibility    independent_evidence: "none"                     risk critical
trap category vocabulary  independent_evidence: "none, and none is constructible"
```

**Two of the three named examples had no independent control, and one of them cannot have one.** The
public copy asserted the opposite of its own register for the majority of the cases it chose to
name.

## 3. The correction

```text
BEFORE  Those rules are listed in a public register and checked separately.

AFTER   Those rules are listed in a public register, and checked separately where a separate
        check is possible: against the specification, by mutation testing, by executable gates.
        Where no independent check exists the register records that instead, including one rule
        for which none can be constructed.
```

This restores EI6's meaning rather than inventing new wording, and it is **stronger copy, not
weaker**: a project that publishes which of its own claims it cannot yet check is making a
harder-to-fake statement than one that asserts uniform coverage.

The final clause — *"including one rule for which none can be constructed"* — is deliberate.
`ESF-TRAP-001a` is not an unbuilt control but an **unbuildable** one: if the `TrapCategory` enum
names the wrong concept, every engine and the corpus manifest are wrong together, and AS8's MUT-008
is the honest no-op that marks the boundary. That is worth saying out loud.

## 4. What AC4 did NOT require changing

```text
"Three engines, four configurations, one answer"   accurate; EI6 correction 1, already landed
the rustc paragraph                                accurate; EI6's addition, already landed
"pinned against the specification, not against
    each other"                                    accurate and load-bearing -- it is the
                                                   sentence AC4-F7 vindicated, since a lowering
                                                   defect surfaced as engines DISAGREEING
the supporting anecdote                            EI6 recommended keeping it verbatim; unchanged
```

**No claim about MIR verification needed correction, because none is published.** `ESF-VERIFY-001`
was created during the AC4 reconciliation specifically so that a future claim of *"independently
verified MIR"* would have a machine-readable sensor. This sweep confirms no such claim exists yet —
so the sensor is in place **before** the claim, which is the correct order.

## 5. Residual

**AC6 is a sweep, not a gate.** Nothing mechanically prevents the next consuming change from
truncating an approved sentence the way this one was — which is exactly how the defect arose:
EI6's wording was correct, and the copy that shipped was not.

A drift gate for public claims would be the durable fix, in the shape of
`cohort_limitations_are_current.rs` — assert that the register's own verdict for each named rule
matches what the public copy says about it. **Not built here**: AC6's scope is the sweep, and
building it is a work packet. Recorded so the gap is owned rather than rediscovered.
