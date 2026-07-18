# WP-C2.2 — Interpreter Semantic Repair

Gate: C2 (Reference Execution Semantics and Compiler Service Foundation). Extracted verbatim in
scope from `STARKLANG/docs/compiler/COMPILER-ROADMAP.md`.

Status: **Completed 2026-07-18.** All ten inherited owned findings (DEV-026 through DEV-035) are
closed with regression coverage; adjacent DEV-037 was found and closed in the same work. The
explicitly optional candidates DEV-009/023/024 remain open, and parser-owned DEV-036 remains out
of scope. See `COMPILER-STATE.md` for evidence and the next work package.

## Scope

Resolve all C0/C1 deviations that affect executed behaviour, including any confirmed issues in:

- operator/trait dispatch;
- default trait methods;
- method receiver mutation;
- structural versus semantic equality;
- enum and pattern representation;
- drop order and partial moves;
- panic/trap paths;
- iterator state and aliasing;
- collection mutation;
- standard builtin resolution;
- extension builtin gating.

No new public standard-library API belongs here unless required by the normative spec and
approved under the ecosystem compatibility process.

## Inherited findings (owned by this WP per prior WPs' disposition)

The repair list, in severity order per the ledger
(`starkc/docs/conformance/KNOWN-DEVIATIONS.md`) after WP-C2.1 and its correction pass:

1. **DEV-035** (highest) — references returned from `&self`/`&mut self` methods dangle after the
   method's call frame is popped; breaks every idiomatic accessor returning `&self.field`.
2. **DEV-034** — by-value method receiver expressions are evaluated twice, duplicating
   observable side effects.
3. **DEV-030** — pattern-match `_`/unbound sub-values of an owned scrutinee are never dropped
   (silent violation of the "exactly once" Drop invariant).
4. **DEV-026** — method dispatch priority is source-textual-order-dependent instead of
   "inherent shadows trait."
5. **DEV-033** — `call_core_method` evaluates arguments before resolving the receiver,
   contradicting the CD-007/CD-010 normative order.
6. **DEV-029** — struct/enum named-field drop order is reverse-alphabetical, not
   reverse-declaration (normative per CD-011).
7. **DEV-032** — `HashMap`/`HashSet` iterate in structural-`Ord` sorted order, not the
   CD-009-normative first-insertion order.
8. **DEV-027** — `Ordering` prelude type unresolvable; no runtime `Ord`/`cmp` dispatch for
   struct/enum comparison operators.
9. **DEV-031** — `for` loops only accept `Range`/`Array`/`Vec`, not general `Iterator`-typed
   expressions.
10. **DEV-028** — `&expr[range]`/`&mut expr[range]` crash at runtime; slices materialize copies
    instead of spec-required views.

Also in scope per the roadmap's "standard builtin resolution" bullet, as capacity allows:
**DEV-023** (`Display`/`Hash` not callable as builtin methods), **DEV-024** (`From::from`
associated-function resolution), **DEV-009** (`File` has no runtime representation — explicitly
"candidate WP-C2.2" per its disposition). **DEV-036** (parser filename bypass) is NOT owned here
— it is a parser fix, not interpreter semantic repair; separately triaged.

## Scope-control answers (Charter §2.6)

- **Exact compiler claim tested:** that after this WP, the interpreter's executed behaviour
  matches `reference-execution.md`'s (now decision-complete) contract for every deviation listed
  above — verified by the same empirical repros that established each deviation, now inverted
  into permanent regression tests.
- **Later mechanism that would make the result impossible to attribute:** fixing a deviation
  without inverting its establishing repro into a permanent test (the fix could silently regress),
  or fixing one deviation in a way that silently alters behaviour adjacent to another (e.g. the
  receiver-handling cluster DEV-034/DEV-035 shares code paths with `&mut self` write-back — a fix
  must not break the existing mutation semantics that currently work).
- **Strongest existing comparator:** the 454-test suite plus each deviation's establishing repro;
  the differential corpus (WP-C2.6) does not exist yet, so this WP's own before/after repro
  discipline is the strongest available evidence.
- **Negative result that would stop this WP/gate:** discovering a deviation whose fix requires
  changing normative spec text again (all spec-side questions were settled in WP-C2.1 +
  correction pass; a fix that reopens one means the contract is still wrong and must go back to
  the user, not be papered over in code).

## CE-escalation watch

Per the standing preference (`stark-ce-escalation-flagging` memory): flag any CE1-CE9-shaped
decision before resolving it. The contract-side decisions are settled (CD-007 through CD-011),
so this WP should be mostly mechanical repair against a fixed target — but watch for: any fix
that would require *weakening* a compile-time check (Charter §2.2, forbidden without
escalation); DEV-027's `Ordering` introduction touching the prelude's public surface (the
roadmap's own "no new public standard-library API... unless required by the normative spec"
clause applies — `Ordering` IS required by the normative spec, `06-Standard-Library.md` line
585, so it qualifies, but the implementation shape should stay minimal); and DEV-028's slice
places, which may force an architectural choice about the `Place`/`Projection` model (CE3-shaped
if it grows beyond a local change).

## Input packet

`COMPILER-CHARTER.md` + `COMPILER-STATE.md` + `STARKLANG/docs/compiler/reference-execution.md`
(the contract) + this file, `starkc/src/interp.rs`, `starkc/src/typecheck.rs`,
`starkc/src/resolve.rs`, `starkc/src/hir.rs`, the deviation ledger's per-DEV repros.

## Execution log

See `COMPILER-STATE.md` session record `### WP-C2.2` for dated evidence, files touched, and
decisions.
