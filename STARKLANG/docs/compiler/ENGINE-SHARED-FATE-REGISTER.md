# Engine shared-fate register

**Packet:** `WP-ENGINE-INDEPENDENCE.md`, approved 2026-08-09 (CD-392), executed as an AS8
prerequisite. **AS0 remains closed.**

**Status: EI0 COMPLETE — vocabulary frozen. EI1 in progress.**

---

# EI0 — Frozen classification vocabulary

**Frozen 2026-08-09, before any authority was examined.** That ordering is the point of EI0: a
vocabulary chosen after looking at results can be shaped, consciously or not, to make the results
come out independent. Nothing below may be widened, narrowed or renamed during EI1–EI6; a
classification that does not fit is a **residual**, recorded as such.

## 0.1 Dependency classification

```text
DIRECT      the engine calls or reads the authority directly
INDIRECT    the engine consumes IR, metadata, normalized types, or generated structures
            produced by the authority
NONE        no relevant dependency found
UNKNOWN     insufficient evidence; resolve, or register as a residual
```

## 0.2 Differential visibility

```text
VISIBLE             a defect should affect one engine independently and create disagreement
PARTIALLY_VISIBLE   some defect classes diverge while others are inherited
INVISIBLE           the relevant engines inherit the same authority or output
UNKNOWN             visibility has not been established
```

## 0.3 Evidence independence

```text
SPEC_DERIVED                strongest
EXTERNALLY_DERIVED          strongest
HAND_AUTHORED               strongest when independently authored
CROSS_ENGINE_DERIVED        correlated unless an independent control exists
IMPLEMENTATION_GENERATED    correlated unless an independent control exists
SHARED_FIXTURE_GENERATOR    correlated unless an independent control exists
UNKNOWN
```

## 0.4 Authority category

```text
SHARED_PREDICATE        SHARED_TYPE_TABLE       SHARED_NORMALIZATION
SHARED_LOWERING         SHARED_IR               SHARED_PROVIDER_SCHEMA
SHARED_ABI_MAPPING      SHARED_ERROR_MAPPING    SHARED_TEST_ORACLE
RUSTC_ASSUMPTION        ENGINE_LOCAL
```

## Acceptance criteria, and how each is met

| Criterion | How EI0 meets it |
| --- | --- |
| classifications are frozen | Recorded here before any authority was examined; the register below was empty at the moment of freezing |
| `UNKNOWN` is not silently treated as independent | **Binding rule:** an `UNKNOWN` in any of the four axes propagates to the entry's risk as `UNKNOWN`, never as `NONE`, `VISIBLE` or independent. An entry carrying any `UNKNOWN` cannot be cited as evidence of independence in EI2 or EI4, and is a **residual** until resolved |
| direct and indirect dependencies are distinguished | Two separate values in 0.1, and EI1 records **both** per engine rather than collapsing to a boolean |
| evidence-source correlation is included | 0.3 is a required column of every EI1 entry, not an EI2-only concern — so correlation is visible at registration time rather than discovered during the audit |

## The rule EI0 exists to prevent being bent

> A shared authority whose defects are **INVISIBLE** to differential comparison, and whose only
> supporting evidence is **CROSS_ENGINE_DERIVED**, is **not** independently evidenced — regardless
> of how many engines agree.

Agreement between engines that inherit the same authority is not corroboration. Recording that as a
frozen rule now is what stops it being argued away against a specific entry later.

## Evidence invariant inherited from CD-392

> **No evidence mechanism may support a claim until its ability to distinguish success from failure
> has itself been demonstrated.**

Binding on every artefact this packet produces and on the AS8 work that consumes them.

---

# EI1 — Shared-fate register

**In progress.** Entries are appended below as authorities are identified. Each carries an `ESF-`
identifier which AS8 trials must reference; AS8 assigns its own trial IDs but **may not invent a
semantic classification independent of this register**.

| Field | Meaning |
| --- | --- |
| `id` | `ESF-<AREA>-<NNN>`, stable once assigned |
| `authority` | the shared thing: a predicate, table, normalization, lowering, IR, schema, mapping or oracle |
| `category` | one of 0.4 |
| `hir` / `mir` / `native` | dependency classification per engine, from 0.1 |
| `visibility` | from 0.2 |
| `evidence` | from 0.3 — the strongest source currently supporting it |
| `control` | an independent control, if one exists; otherwise `none` |

*(no entries yet — EI1 begins after this file is committed, so the freeze is a separate act from the
findings it constrains)*
