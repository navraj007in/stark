# WP-ITER-LOWERING — iterator adapters and by-value collection iteration (PROPOSAL)

**Status:** PROPOSED — not scheduled, not started. Requires owner approval and a roadmap slot before
any implementation (COMPILER-CHARTER §1.6 rule 4: no new Core semantics inside an implementation WP).

**Origin:** WP-C6.3c (CD-128/CD-129) closed for *native parity*: every §26 iterator row that MIR can
lower is native and proven three-engine. The rows that remain stop **before MIR**, so they are
capability gaps in the front end and MIR — not native-backend gaps. Gate C6's charge is native
semantic parity, and implementing these inside it would have expanded a backend/runtime WP into new
front-end and MIR semantics. The owner ruled (2026-07-26) that they are excluded from C6.3c and
recorded here instead.

**Why this is not a C6 concern.** Each excluded form runs in the HIR interpreter and is refused at
lowering. The MIR interpreter cannot run them either, so there is **no native/interpreter divergence**
for a native-parity gate to close, and the three-engine differential cannot reach them at all.

## Scope

1. **MIR representations for adapter iterators.** `Core(MapIter, [Src, U])` and
   `Core(FilterIter, [Src])` exist in the type system (`typecheck.rs` produces them) but have no MIR
   type and no lowering. Decide their MIR representation and destruction obligations.
2. **Method resolution/lowering for iterator values with NON-NOMINAL types.** `count`/`collect` fail
   today as "method call on non-nominal receiver `Core(VecIter, …)` (C4.5b+)". Lowering can only
   resolve methods on nominals; core-typed receivers need a resolution path.
3. **By-value collection iteration.** `for x in v` (consuming a `Vec`) — refused as "for over a
   non-range, non-Vec iterator". Runs in HIR today.
4. **Remaining-element `Drop` on every exit path.** Normal completion, `break`, a trap, and an early
   `return` must each destroy exactly the not-yet-yielded elements, once. This is the semantic core
   of by-value iteration and the reason it is not a mechanical addition; it interacts with drop flags
   and with §7.7's no-unwind trap rule (a trap runs NO destructors).
5. **Slice iteration — ONLY if the language surface is explicitly approved.** `for x in <slice>` is
   currently rejected by the front end ("for-loop requires an iterable value, found `&[Int32]`"). It
   is an absent language feature; adding it is a spec change, not a lowering fix.
6. **Mutable iteration — ONLY through a separate language/spec decision.** There is no `iter_mut`
   surface anywhere in the compiler or the spec. Mutable iteration cannot be "implemented"; it must
   first be specified.

Items 5 and 6 are gated on spec decisions and must not be started on the strength of this proposal
alone.

## Evidence already in place

Four executable boundary tests in `starkc/tests/c63c_iterators.rs` are **permanent regression
evidence** and must be kept:

| Test | Pins |
| --- | --- |
| `slice_iteration_is_not_a_language_form` | the front end rejects slice iteration |
| `vec_by_value_iteration_is_hir_only` | HIR runs it, lowering refuses it |
| `map_adapter_is_hir_only` | `MapIter` has no MIR type |
| `count_and_collect_are_hir_only` | non-nominal receiver methods do not lower |

Each HIR-only test asserts BOTH that the HIR interpreter runs the program AND that lowering refuses
it. That is what distinguishes "supported by HIR but not lowerable" from a native divergence, and it
means the boundary cannot move silently: if any of these starts lowering, its test fails and the case
must be promoted to a three-engine case.

## Exit criteria (when scheduled)

- Each implemented form proven three-engine (HIR == MIR == native), including order and early
  termination, matching the WP-C6.3c evidence standard.
- Remaining-element `Drop` proven observably (a `Drop`-bearing element type printing on destruction)
  for completion, `break`, trap, and early return.
- The corresponding boundary test promoted from HIR-only to three-engine in the same change, so the
  table above never overstates what is supported.
