# Resolution Audit — scope

**Status:** draft scope, for a charter slot. Not a chartered packet.
**Proposed baseline:** `0e276eb` (`develop`), the commit that repaired DEV-222/223/225/226/227.
**Author's position:** this is the fourth consecutive unchartered compiler-track activity arising
from `stark-cookie`. The three before it were each justified in the moment. An audit that finds
defects creates pressure to fix them, so it should be scheduled rather than continued on momentum.

## Why

DEV-222, DEV-223, DEV-225, DEV-226 and DEV-227 were all the same shape: **the resolver returned a
resolution that was correct for one context and was then accepted in another where it was not
legal.** Four of the five produced *silent wrong behaviour* — the program compiled, no diagnostic
was issued, and an arm simply never matched. A wildcard arm hid the mistake completely.

None of them was found by reading code. Every one was found by running a small program and
comparing what happened against what should have happened; reading `resolve.rs` only confirmed
root causes afterwards. The audit therefore probes rather than reviews.

## The three questions

The existing conformance suite asks two questions per case. This class of defect needs a third.

1. Does it **compile** when it should?
2. Does it **reject** when it should — and does the diagnostic name the right thing? DEV-222's
   `E0303 non-exhaustive` pointed at the `match` rather than the typo, which is what invited the
   wildcard that made it silent. A correct rejection with a misleading message is a partial defect.
3. **When it compiles, does it behave correctly at run time?** This is where the class lives.
   DEV-223's constructor face passed both front-end stages and trapped at runtime; the first
   regression test written for it passed against the defect because it stopped at `typecheck`.

## Scopes, in priority order

### A — Silent-acceptance sweep *(highest expected yield)*

The specific pathology: compiles, no diagnostic, behaves as a no-op.

Method: a table of pattern constructs, each paired with a scrutinee it *should* match and an
expectation of `HIT`, `BIND`, or `REJECT`. Every case runs with a wildcard arm present, because the
wildcard is what converts the defect from loud to silent. A construct that compiles and reports
`MISS` where `HIT` was expected is a silent no-op; a construct that compiles where `REJECT` was
expected is an accepted invalid program.

### B — Precedence matrix *(second highest)*

For each qualifier kind — module, enum, struct, trait, core trait, primitive, alias — against each
name category, with a **same-named decoy planted in every other namespace**. DEV-223 and DEV-225
were two cells of this table. There is no evidence about the other cells; they have never been
probed.

### C — Pattern legality matrix

Every `Res` variant against every pattern kind, probed rather than reasoned. `resolve.rs`'s
`resolution_is_pattern_legal` is exhaustive *by construction*, which is not the same as verified by
execution: an arm can be exhaustively present and still classify wrongly.

### D — `Res` well-formedness

Whether `Res::Variant(item, idx)` can be constructed naming a non-enum or an out-of-range index.
Cheapest, least likely to yield, and speculative — no reproducer exists. Last.

## Deliberately out of scope

- **Fixing anything.** The audit reports; repairs are separate and separately scheduled. A finding
  that turns out to be intended behaviour is a documentation outcome, not a defect.
- **DEV-228.** Already filed, already understood, and a resolver-model decision rather than an
  audit question. The audit may add evidence about its cost; it does not decide it.
- **Type checking, borrow checking, MIR, native lowering.** Resolution and its consumers only.

## Exit

A report listing every probe, its expectation, its observed result, and a verdict of AGREES /
DISAGREES / INTENDED-BUT-UNDOCUMENTED. Disagreements become ledger entries with reproducers.
Nothing is repaired under this scope.

---

# Report — scopes A and B, run 2026-08-11

Harness: `starkc/scripts/resolution-audit-probe.py`, run against `starkc/target/debug/stark` built
from `0e276eb`. **33 probes, 1 disagreement.**

Each probe is a whole program executed through the interpreter, not a `check`. That is deliberate:
DEV-223's second face and DEV-230 both pass `stark check` and fail only when run.

## Scope A — silent acceptance: 23 probes, 1 disagreement

| Expectation | Probes | Result |
| --- | ---: | --- |
| `HIT` — must compile and the arm must fire | 9 | all agree |
| `BIND` — must compile as a fresh binding | 6 | all agree |
| `REJECT` — must not compile | 8 | 7 agree, **1 disagrees** |

Confirmed correct after the DEV-222/226/227 repairs: misspelled enum variants, struct-qualified
variant paths, builtin functions used as patterns, core-trait and trait member paths, and
module-qualified misspellings are all rejected with a diagnostic that names the path. Valid unit
variants, tuple variants, struct patterns, enum struct-variants, module-qualified variants,
imported variants, `Some`/`None` and constants all still match. Bare function, struct, enum, trait,
primitive and module names all bind, per SYN-PATTERN-001.

**The disagreement is DEV-230**: a struct pattern naming a field that does not exist compiles, and
silently does not match. Filed with its reproducer and root cause. It is the same failure mode one
stage further on — type checking rather than resolution — and the DEV-222 repair cannot reach it,
because the *path* resolves correctly and the bad name is a field inside the pattern.

## Scope B — precedence: 10 probes, 0 disagreements

Every cell probed with a same-named decoy planted in another namespace:

- enum variant beats a module function of the same name (DEV-223's shape)
- enum variant beats a module **type** of the same name (DEV-223 exactly)
- struct associated fn beats a module function (DEV-225)
- enum associated fn beats a module function
- a module qualifier still searches the **module** namespace, not some enum's variants
- a lexical binding still shadows a module-level name
- a user type occupying a hardcoded builtin spelling resolves to the user's declaration
- a user trait named like a core trait resolves to the user's
- a nested module shadows an outer name
- an imported variant beats a same-named local function

All agree. The precedence repairs hold, and the reorder did not steal names from the module
namespace — which was its main risk.

**This does not clear DEV-229.** The builtin probe shows a user declaration winning for a
*non-colliding* spelling; it still does not separate "the user's declaration won" from "the builtin
won and happened to agree" for one of the ~30 reserved spellings. DEV-229 stays UNCONFIRMED.

## What the audit did not cover

Scopes C and D were not run. C (every `Res` variant against every pattern kind) is largely
subsumed by scope A's results, which probe the same surface from the program side. D (`Res`
well-formedness) remains speculative with no reproducer.

The enum struct-variant form of DEV-230 **was** probed after the first report draft, and shares the
defect: `Rec::One { nope: v }` compiles and does not match. DEV-230 therefore covers both struct
patterns and enum struct-variant patterns. Leaving that as a suspicion would have understated it.

## Verdict

The five repaired defects are repaired, and the repairs did not break the precedence they moved.
One new defect of the same class was found in the next stage down. The method — execute whole
programs and compare against an expectation, with a wildcard present — is what found it, and is
worth keeping.

---

# Report — scopes C and D, run 2026-08-11

Harness: `starkc/scripts/resolution-audit-scope-cd.py`, run against a `stark` built from `2532b72`.
**29 probes, 5 disagreements — of which 4 are one defect and 1 was the audit being wrong.**

## Scope C — pattern legality matrix: 25 probes, 5 disagreements

Scope A probed the pattern kinds a program most often writes. This walks the matrix the other way:
each path-bearing pattern kind against each resolution category the resolver can produce, plus the
non-path kinds against mismatched types.

**Clean:** struct patterns reject a function, module, primitive, trait or constant in the path
position, and reject an unknown field (DEV-230's repair holding). Tuple and array patterns reject
arity mismatches and type mismatches. Literal patterns reject a mismatched type. Path patterns
reject a trait member.

**DEV-231, filed and since REPAIRED:** tuple-variant patterns checked neither the shape of their constructor nor their
arity. `Colour::Red(_v)` on a payload-less variant, `Shape::Line(_a, _b)` on a one-field variant,
`Thing(_v)` on a named-field struct and `LIMIT(_v)` on a constant all compile and silently never
match. Three mechanisms in one arm: a non-tuple variant yields `None` and the check is skipped,
`zip` truncates on arity mismatch, and a resolution that is neither `Res::Variant` nor
`Res::Builtin` reaches neither branch.

**The audit was wrong once, and the specification said so.** `Rec::One` — a bare path pattern
naming a struct variant — matches, and the probe expected a rejection. SYN-PATTERN-001:
"Multi-segment `Path` patterns always match by value", and Core v1 has no rest patterns, so a bare
path is the only way to match a struct variant without binding its fields. Rust intuition, not a
defect. Recorded because an audit that only reports the compiler's errors and never its own is
not measuring itself.

## Scope D — resolution well-formedness: 4 probes, 0 disagreements

The concern was whether a `Res::Variant(item, index)` can be consumed against the wrong enum or an
index the enum does not have — `typecheck/patterns.rs` indexes `variants[*variant_idx as usize]`
directly, so a malformed one would be an internal panic rather than a diagnostic.

**Every producer derives the index from the enum's own variant list.** All three sites in
`resolve.rs` build it from `enumerate()` or `position()` over that enum's variants, or propagate an
already-valid one. There is no surface syntax that separates the index from the enum it came from.

Probed anyway, from the language rather than from the code: two enums with same-named variants, an
index valid in a wide enum and out of range for a narrow one, a re-exported variant competing with
a same-named local variant, and a generic enum's variant surviving instantiation. All four behave.

**Verdict: no defect, and the invariant holds by construction rather than by check.** The unchecked
index remains a latent internal-panic surface if a future producer ever derives one differently.
That is a hardening argument, not a finding, and is deliberately not filed as a deviation.

## Where the audit now stands

All four scopes have run. Two defects found across 62 probes: DEV-230 (scope A) and DEV-231
(scope C), both in `typecheck/patterns.rs`, both the same failure mode — compiles, no diagnostic,
silently never matches. Both are now repaired, and re-running C and D against the repairs gives
**29 probes, 0 disagreements**.

Both were one stage past where the audit was aimed, which is the most useful thing it established:
**the resolution repairs held, and the defects of this class were in type checking.** A sweep of
the same shape aimed at type checking, borrow checking and MIR lowering is the obvious next one —
`typecheck/patterns.rs` alone yielded two defects and its own comments record a third (DEV-205) of
the identical class, fixed one entry at a time before the general check existed.
