# WP-LOWERING-COVERAGE-MATRIX — find accepted-but-unbuildable programs on purpose

**Status:** FILED, not started.
**Filed by:** CD-329, on owner direction, after DEV-132 and DEV-133 were both found by a package
build rather than by the compiler's own corpus.
**Owning track:** compiler (Gate C-series governance, `COMPILER-CHARTER.md` §1.6).
**Motivating deviations:** DEV-132, DEV-133.

---

## 1. The defect class

> The checker accepts a program, the HIR oracle executes it correctly, and MIR lowering either
> refuses it or emits MIR the verifier rejects.

Not unsoundness — **over-refusal**. The program is valid Core v1, the oracle proves it has a
meaning, and the compiler cannot build it. For a language whose deliverable is a native binary,
that is a capability hole that reports itself as an internal compiler error.

Two instances, found within an hour of each other on 2026-08-01:

| DEV | Construct | Refused by | Mechanism |
| --- | --- | --- | --- |
| 132 | `&v[i].field` on `Vec<NonCopy>` | MIR-0016 (V-COPY-1) | place context not preserved through indexing; base lowered as a by-value read |
| 133 | `let s: &[UInt8] = &[b];` | MIR-0004 | array→slice unsize coercion never emitted |

Both were pre-existing. Both were invisible to the entire in-tree corpus. Both surfaced the first
time a real package was built natively.

## 2. What already exists, and why it did not catch them

**The rule is already written and already gating.** `c6_generated_corpus.rs` classifies
`Admission::MirVerifyFailure` as `MIR-VERIFY-FAILURE` and prints *"an accepted Core case refused by
an engine is a C6 BLOCKER (§12.2)"*. So "checker accepts ⇒ lowers ⇒ verifies" is enforced today.

**Neither defect slipped past the rule. No corpus case contained the construct.** That is the
finding this package exists to act on: the gap is INPUT COVERAGE, not policy, and no new invariant
is needed.

`tests/layer_audit.rs` holds a static inventory of `lower.rs`'s refusal sites (194 `unsupported(...)`
calls, ~136 classified as auditable). That is a partial instrument — see §3.1 for why it is only
partial.

## 3. Three audits, in increasing cost

### 3.1 Reachability of the refusal sites — cheapest, and provably incomplete

Every `unsupported(...)` in `lower.rs` is a claim: *the checker will never send me this*. For each,
ask whether a checker-accepted program can reach it. Reachable ⇒ an accepted-but-unbuildable defect.
Unreachable ⇒ genuinely defensive, and worth recording as such.

**Its limit must be stated up front, because it is the reason this alone is not the package:
neither DEV-132 nor DEV-133 was an `unsupported()` site.** Both were VERIFIER rejections on MIR that
lowering produced willingly. This audit covers only the half where lowering knows it is refusing.

### 3.2 The type × position matrix — the one that would have caught both

Both defects are the same shape: a valid TYPE COMBINATION in a SYNTACTIC POSITION that lowering
mishandled. That is a finite cross product, and generating it is mechanical:

- **element/inner types** — `Copy` scalar, non-`Copy` (`String`), nested aggregate, unit, reference
- **containers** — array `[T; N]`, slice `&[T]`, `Vec<T>`, `HashMap<K, V>`, tuple, struct, enum payload
- **positions** — `let` with a declared type, `let` without, call argument, method receiver, return,
  return-expression, assignment LHS, assignment RHS, field base, index base, match scrutinee
- **forms** — by value, `&`, `&mut`

Each admitted cell asserts the full chain: checker accepts ⇒ lowering succeeds ⇒ verifier passes ⇒
three engines agree ⇒ native build succeeds. Each REFUSED cell is recorded with its diagnostic rather
than skipped, so the acceptance surface is documented and a later widening is visible.

The harness pattern exists twice already — `copy_canon_matrix.rs` (producers × use modes) and
`c6_generated_corpus.rs`. This generalises the first from Copy-ness to the type/position surface.

**DEV-132 is the cell `Vec<non-Copy>` × `index base` × `&`. DEV-133 is `[T; N]` × `let with declared
type` × `&`.** Both fall out of the enumeration without anyone thinking of them, which is the
argument for the package.

### 3.3 Package-driven qualification — weakest guarantee, already in place

Real packages exercise combinations nobody would write deliberately. This is what found both
defects, and as of CD-329 all ten first-party packages run through
`qualify-first-party-packages.py` in CI on three platforms including native builds. No further work
is owed here; it is listed so the package does not duplicate it.

## 4. Scope

**In:** §3.2 in full, and §3.1 as a companion — converting `layer_audit.rs` from a static inventory
into reachability probes.

**Out:** changing what the checker accepts. If the matrix finds a cell the checker admits and the
language should not, that is a spec question routed through the normal path, not a fix here.
**Out:** performance of the generated suite; if the full cross product is too slow for every CI run,
shard it as `c6_generated_corpus` already does, rather than trimming the space.

## 5. Acceptance

1. The cross product is enumerated in code, not hand-written, so adding a container or position adds
   its whole row.
2. Every admitted cell asserts the full chain through native build. A cell that stops at "lowers" is
   the gap that let DEV-133 through — it lowered fine and failed verification.
3. Every refused cell records its diagnostic. A silently skipped cell is indistinguishable from a
   passing one.
4. The two known cells (DEV-132, DEV-133) are present and would have failed before their fixes —
   demonstrated by running the matrix against the parent commits, not asserted.
5. Findings are registered as DEVs before repair, per the ordering rule.

## 6. Risk

The likely outcome is that the first run finds several more instances, and the package becomes a
queue of small independent repairs rather than one change. That is the point, and it should be
planned for: land the matrix with the failing cells recorded as registered DEVs and marked expected,
then close them one at a time. Do not hold the matrix back until every cell is green — an audit that
cannot land until the code is perfect never lands.
