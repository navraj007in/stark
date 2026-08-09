# C10 — conformance dashboard

**Packet:** C10-A2, WP-C10.1. **Date:** 2026-08-09. **Head:** `fe4c902`.
**Machine-readable source:** `STARKLANG/conformance/c10-dashboard.json` (168 rows).
**Generator:** `starkc/scripts/c10-dashboard.py`, from `c10-a2-resolve.py`.

> **No aggregate conformance percentage appears in this document.** Plan §7.3 forbids mixing
> precise and unclassified rows into one number, and §6.5 forbids a glossy figure that hides
> evidence quality. Per-state counts are below; a single headline would be a worse answer than
> none.

---

# 1. Evidence state over the declared population

```text
POPULATION   168 granular rules   (semantic-freeze/CORE-V1-COMPLETENESS.md; declared in C10-0,
                                   corrected from 161 by C10-A1's A1-F1, no exclusions)

PRECISE-C211           36    positive AND negative evidence at test-function precision
RESOLVED-BY-TREE       20    a TEST FUNCTION cites the rule id — and the inventory did not say so
CORPUS-OR-FILE-LEVEL   27    cited at file or corpus level; real evidence, not per-rule attributed
IMPLEMENTATION-ONLY     1    cited ONLY where the rule is implemented — provenance, not evidence
UNRESOLVED             84    no citation found. NOT "untested" — see §3
```

**Function-precision evidence rose from 36 to 56 of 168, and no test was written to do it.** The
twenty were already in the tree; the inventory simply never recorded them. That is A1-F3 paying off
exactly as predicted: the buckets were citation states, not evidence states.

## 1.1 The resolver needed three corrections, and every one was found by reading NAMES

Recorded because the counts were plausible at every stage, and only the names were wrong:

```text
1  implementation citations counted as evidence      `interp.rs::eval_expr` was offered as
                                                     evidence for TYPE-PRIM-001. An implementation
                                                     cannot be its own control — AS8-R4 exactly
2  `///` citations attributed BACKWARDS              a doc comment above a `#[test]` named the
                                                     PRECEDING function. Five attributions in
                                                     c91_extension_isolation.rs came back shifted
                                                     by one — every name a real function in the
                                                     right file, which is why it would have
                                                     survived review
3  corpus cases counted at function precision        a `.stark` case has `fn main`, and
                                                     attributing a rule to "main" is precision
                                                     theatre
```

**A dashboard citing the wrong test is worse than one citing none, because it looks checked.**
Each correction lowered the numbers.

**One residual, stated rather than engineered away:** four rows (`TYPE-FN-001`, `TYPE-INFER-001`,
`PROC-MAIN-001`, `PROC-EXIT-001`) still attribute to `main`, because the rule id sits inside an
**embedded STARK source string** in a Rust test and the scan finds that program's `fn main`. The
FILE is correct evidence; only the function name is wrong. Excluding `main` outright would drop
legitimate citations, so the imprecision is documented instead.

---

# 2. What each column means, and the two that are deliberately not automatic

Plan §6.5's columns, in its order: rule id, normative home, implementation location, positive and
negative evidence, engines/configurations, evidence class (Charter §5.2), shared-fate authority,
independent control, mutation/challenge status, deviation/residual, last verified.

```text
independent control      `none` is a LEGITIMATE and EXPECTED value. Six of eleven ESF authorities
                         are INVISIBLE to all three engines, so a differential cannot contradict
                         them however many engines agree (EI0's frozen rule). A dashboard with no
                         `none` cells has been massaged
mutation/challenge       inherited from AS8 where a trial exists, otherwise NOT-CHALLENGED.
                         Nothing here invents a challenge that was not run. C10-0 verified the
                         inherited results are FRESH under plan §8.2a — all 12 authority files and
                         13 control suites hash identically at `e7bb95d` and at the baseline
```

**The implementation column is provenance, not evidence.** The first version of the resolver
counted `interp.rs::eval_expr` and `lower.rs::lower_expr_to_operand` as evidence for
`TYPE-PRIM-001`. Those are the implementation naming the rule it implements, and **an
implementation cannot be its own control** — `AS8-R4` exactly, where `copy_canon_matrix` is a
transcription of `core_method_signature` and so "would pass just as happily if the reverse were
true". Implementation citations are now separated into their own column and never counted.

---

# 3. The 85 UNRESOLVED, and why the word is `UNRESOLVED` rather than `none`

**The resolver finds tests that CITE a rule id.** A test that pins a rule without naming it is
invisible to it, and the spec-fixture corpus carries **no rule ids at all**. So a missing hit is a
prompt to look by hand — never a finding that no control exists. That limitation is stated in the
tool's own header, and §3.1 is what happens when you take it seriously.

## 3.1 Two thirds of the unresolved set is one predictable class

```text
LEX   17        SYN   17        STD  8   TYPE 7   FLOW 6   AM  6   MOD 4   FUTURE 4
EXEC   3        REF    3        PKG  2   others 1 each
```

`EXT-ISOLATION-001` was in this list until this packet, and it is the exemplar: the inventory
recorded `none; none`, C10-A1 found nine tests running in CI, and the **resolver could not find
them either** — because nothing in `c91_extension_isolation.rs` named the rule it pinned. Adding
one module-header note plus five per-test attributions moved it to `RESOLVED-BY-TREE` with its five
real test functions named. **No test was written and no behaviour changed.** That is the shape of
the remaining work: a control that does not name its rule is invisible to every mechanical audit,
gets recorded as absent, and is then re-litigated.

**LEX and SYN are 34 of 84**, and both are exercised by the spec-fixture corpus — the one suite
that carries no rule ids. Their absence from the resolver's output is the tool's stated limitation
firing precisely where it predicted, not a discovery about the compiler.

## 3.2 But the two families are not in the same condition, and measuring said so

```text
LEXICAL      lexer.rs in-module tests    26 test fns, 32 error assertions
             spec fixtures                7 lex-pass

SYNTACTIC    parser.rs in-module tests   47 test fns, only 5 asserting a rejection
             spec fixtures               64 parse-pass ... and exactly ONE parse-fail
             -> negative evidence is THIN, and this is a real finding rather than a bookkeeping gap
```

### The lexical conclusion, in the four states it passed through

**All four are kept.** Git history can recover the old sentence; a qualification record should not
require archaeology to explain why a conclusion changed. Same discipline as `C10-THREAT-MODEL.md`
§2a.

```text
1  ORIGINAL CONCLUSION   (C10-A2)
     "negative evidence is DENSE. These rules are controlled; the attribution is missing."
     Basis: lexer.rs has 26 test fns and 32 error assertions.

2  CHALLENGE             (C10-D, C10D-MUT-001)
     delete `"mut" => Mut` from the keyword table
       lexer.rs unit suite      26 / 26 PASS   <- including keywords_reserved_and_idents,
                                                  a test named for exactly this rule
       conformance              FAIL
       gate2_valid              11 of 56 FAIL
     Every kill was a program ceasing to PARSE. No lexical assertion fired.

3  WITHDRAWAL
     The measurement was valid; the INFERENCE was not. 32 assertions measured lexical testing
     DENSITY — literal forms, escapes, malformed input — not keyword-to-token IDENTITY.
     Inferring control from an assertion count measures the wrong property, and C10-A2 did it in
     the same session that recorded EI2 doing the same thing.

4  CURRENT CONCLUSION    C10-R1
     keyword MEMBERSHIP    has coarse downstream control — remove a word from the keyword set
                           and every program using it stops parsing
     keyword IDENTITY      is NOT independently pinned. Nothing establishes that `mut` maps to
                           TokenKind::Mut rather than to some other valid keyword token
     class                 assurance/evidence-quality residual, population C.
                           NO DEV — this is not evidence of a misclassification in the
                           implementation, and none has been demonstrated
```

**One `parse-fail` fixture in the entire corpus.** Charter §1.6 rule 15 requires that "positive and
negative evidence travel together — every semantic rule needs valid and invalid cases where
rejection is meaningful". The fixture corpus demonstrates the grammar *accepts the specification's
own examples*. It barely demonstrates the grammar *rejects* anything.

That is not the same as saying the parser over-accepts: `over_acceptance_audit.rs` (8 tests),
`adversarial_boundaries.rs` (14), `gate2_valid.rs` (56) and C10-B's 800 soup cases all push on it
from other directions. It does mean **no C10-Q claim of syntactic conformance may rest on the
fixture corpus alone**, and the corpus's own positive/negative balance should be stated wherever it
is cited.

---

# 4. What this dashboard does NOT establish

```text
NOT a conformance verdict         it records what evidence EXISTS and at what precision. A rule
                                  in PRECISE-C211 may be pinned by two weak assertions; a rule in
                                  CORPUS-OR-FILE-LEVEL may be thoroughly exercised
NOT coverage                      coverage appears nowhere here. Coverage is not conformance:
                                  typecheck/traits.rs is 82.77% covered and ESF-TRAIT-001 has no
                                  control of any kind
NOT a claim about the 85          UNRESOLVED means the resolver found no citation. For LEX that
                                  almost certainly understates reality; for SYN it points at a
                                  real thinness. Neither is settled by this packet
NOT challenged                    only 5 rules carry an AS8 trial. The other 163 are
                                  NOT-CHALLENGED, and C10-D decides which of them earn one
```

---

# 5. Reproducing

```bash
python3 starkc/scripts/c10-a2-resolve.py               # per-rule resolution against the tree
python3 starkc/scripts/c10-a2-resolve.py --json        # machine-readable
python3 starkc/scripts/c10-a2-resolve.py --unresolved-only
python3 starkc/scripts/c10-dashboard.py                # regenerate c10-dashboard.json
```
