# MIR Amendment A12 — ending a local's storage

**Status:** **IMPLEMENTED at MIR `0.3` (CD-265, 2026-07-31)** — fixes `DEFECT-C788-LOOP-TEMP`.
**Governs:** the defect CD-263 recorded and CD-264 admitted as a non-blocking C7 deviation.
**Scope class:** **MIR shape change**, not a runtime-surface revision. §1 says why. It adds no
`RuntimeFn`, so `MIR_RUNTIME_SURFACE` stays at `0.1-A10`.

**This amendment was written after the fact and approved retrospectively.** §8 records the
governance question as it was put; the ruling below closed it.

```text
CE3 — APPROVED RETROSPECTIVELY

MIR Amendment A12 is admitted as the minimal correct compiler-wide
representation of storage becoming reusable after all ownership units
have been accounted for.

MIR_VERSION 0.3 is authoritative.
MIR_RUNTIME_SURFACE remains 0.1-A10.
C8 is notified of the new Statement variant and must preserve compatibility.
```

**The grounds are architectural, not procedural.** A12 is approved because MIR is the correct owner
of the fact that a place's ownership units have all been accounted for — not because nothing
happened to break. Lowering is the only layer that knows it; repairing this in the backend would
mean inferring ownership facts MIR should state; and weakening the `ValueSlot` checks would convert
a detectable compiler defect into a possible silent leak.

The required coordination consequence — a permanent guard covering every MIR consumer — is
`starkc/tests/mir_statement_consumers.rs`. See §6.

---

## 1. Why the statement set is part of the shape

The contract described the statement set as closed: *"assignments and nops only. `Drop` is a
terminator."* A consumer written against that sentence has two statement cases and no default. A
third statement is therefore not additive in the way a `RuntimeFn` is — it is a change to what a
MIR body may contain at all.

The version increment is not about consumers failing loudly, though. It is about **caching**. A
build key that ignored this change would serve an artifact produced by a lowering that could not
end a partially moved local's storage — which is to say, an artifact with the defect this amendment
exists to remove. `MIR_VERSION` `0.2` → `0.3` invalidates those keys, which is the point.

## 2. The defect

Two liveness models coexist, and only one of them was complete.

MIR tracks **per-unit** liveness in drop flags: which field, element, or payload of a place is still
owned. The generated-Rust backend additionally tracks **whole-storage** liveness in `ValueSlot` —
`Dead`, `Whole`, or `Partial` — because a Rust binding cannot express MIR's liveness across a
`loop { match __bb { .. } }` dispatch (see `slot.rs`).

Exactly two MIR operations reset whole-storage liveness: a whole-place `Drop` (`drop_with`) and a
whole-value move-out (`take`). A place emptied **unit by unit** — a sub-place move, a field-precise
drop — is left `Partial`, and *nothing ever finished it*.

In straight-line code this is invisible, because such a place is never assigned again. Across a
**loop back edge** it is fatal: the next iteration's assignment finds storage that MIR considers
empty and the backend considers live, and `ValueSlot::write` refuses it — correctly, and loudly:

```
generated-code invariant violated: write to a live slot
(MIR must Drop or move out before reassigning a live place)
(STARK compiler defect, not a program fault)
```

### Three shapes, not one

CD-263 found this through a `match` scrutinee temporary and recorded it as affecting temporaries
rather than user locals. **That scope was too narrow**, established by measurement during the fix:

| shape | storage after | why it was missed |
| --- | --- | --- |
| `match` scrutinee temp, arm moves a non-`Copy` payload out | `Partial` | the case CD-263 found |
| `match` scrutinee temp, arm moves nothing out (unit variant, or all-`Copy` payload) | `Whole` | needs the opposite remedy — see §4 |
| **user local** with a field moved out, reused across the back edge | `Partial` | no `match` anywhere in the program |

The third is an ordinary user binding. `let t = pair(i); let a = t.0;` inside a loop aborts on the
second iteration. The defect was never about temporaries specifically — it was about *any* place
whose storage is emptied piecewise.

## 3. The form

```rust
pub enum Statement {
    Assign(Place, Rvalue),
    Nop,
    StorageDead(Place, StorageEnd),   // A12
}

pub enum StorageEnd {
    Accounted,
    OwnsNothing,
}
```

Contract:

- the place is a **whole local** — no projection. `MIR-0035` enforces it. Storage liveness belongs
  to a storage cell; a projection names part of one, and "ending part of a local's storage" is not
  a thing MIR can mean.
- at this point the local owns no live drop unit. The `StorageEnd` says *why*, and the two reasons
  are checked differently.
- it is **idempotent** on an already-dead local, which is what lets drop elaboration emit it
  unconditionally at the end of a local's sequence instead of proving which path reached it.

## 4. Why the reason is not bookkeeping

A single unconditional "make this local writable again" would have been smaller, and it would have
been wrong in the direction that matters: it would end storage over a live value without complaint,
turning a lowering defect from a loud abort into a **silent leak**. The whole-storage check is what
surfaced this defect at all; the fix preserves it rather than relaxing it to make itself easier.

| reason | backend | check kept |
| --- | --- | --- |
| `Accounted` | `finish_partial()` | **refuses a `Whole` slot** — a complete value that is still there owes a real drop or move |
| `OwnsNothing` | discarded `take()` | **requires a `Whole` slot** — `take` enforces it |

`OwnsNothing` covers a match arm whose active variant has an empty payload or an entirely `Copy`
one (and `Copy` excludes `Drop`, so a `Copy` payload can own nothing). Its storage is still whole,
and ending it abandons nothing.

**It may not simply drop the value instead**, and this is the case that forced the discriminator to
exist. A whole-value drop runs the enum's glue for **every** variant — including one holding a host
resource that this arm never had. The backend rightly refuses that:

```
host-resource close (provider call 6) must be emitted by the Drop terminator,
not by generic drop glue
```

That is the `Err` arm of every `Result<Resource, E>` — the most ordinary shape in the whole
capability surface, and P1's own. A first attempt at this fix used one unconditional form, and
`c788_lifecycle_e2e::a_failed_handle_out_does_not_close` failed to build because of it.

Rust's structural drop of the discarded `take()` reclaims the value's own storage while running no
user destructor — generated nominal types implement no Rust `Drop` (§6.3) — so this cannot become
the second destruction schedule §7.1 forbids. It is the same structural reclaim `drop_with` performs
as its last step, reached by the one operation that is legal when nothing is owed.

## 5. Where lowering emits it

- **end of each local's drop-elaboration sequence** (`emit_scope_drops_from`), as `Accounted`. True
  on every path into that point, and idempotent, so no path analysis is needed.
- **after a consuming match arm's payload consumption** (`consume_variant_payload`): `Accounted`
  when a non-`Copy` field came out, `OwnsNothing` when nothing did.
- **the C6.1c decomposition temporary**, as `Accounted`. It is emptied field by field exactly as the
  scrutinee is — a *second* compiler-generated temporary on the same reassignment path, and missing
  it kept the multi-field payload case failing after the single-field one was fixed.

The two catch-all arms needed nothing: a binding moves the whole value out, and
`drop_whole_scrutinee_at_arm_end` reads it whole. Both already reset the storage, which is why only
the variant-payload path was affected.

## 6. Consumers

- **generated-Rust backend** — the only consumer with a partial-storage model; see §4.
- **reference interpreter** — a nop. It holds values in a map keyed by place and has no state a
  storage end could correct. Inert by nature, not unimplemented.
- **verifier** — `MIR-0035` (§3's projection rule).
- **linkage validation** — a nop; the statement names a local and references no instance.

## 7. Evidence

Sixteen shapes, each checked for MIR/native **agreement** and for correct destructor counts — not
merely for building:

payload arm · unit arm · `Copy`-payload arm of a droppable enum · empty variant of a droppable enum ·
multi-field payload · `Option<Res>` · `Result<Res, E>` · nested matches · wildcard arm · binding arm ·
move-through-call · two temporaries in one body · `continue` out of an arm · `break` out of an arm ·
match-as-expression · user-local partial move.

All sixteen agree. All sixteen aborted, or failed to build, somewhere before this change.

`c788_lifecycle_e2e::repeated_connect_and_release_reuses_slot_state` — committed `#[ignore]`d with a
classification by CD-263 rather than deleted — is **un-ignored**, with its `CLASSIFIED_IGNORES` entry
removed, exactly as that classification said would happen. The suite is 9 passed, 0 ignored.

## 8. Open for the owner

**This amendment was implemented without a prior ruling**, under CD-264's commission to fix the
defect and its instruction that the fix be "compiler-wide rather than TCP-specific". The commission
did not mention the MIR surface, and the charter records that *changes to common MIR enums require
coordination* because C8 compiles against them.

The judgement made was that no smaller fix is correct: the user-local shape (§2) cannot be repaired
by changing lowering shapes alone, because only lowering knows where a place's units have all been
accounted for, and deriving it in the backend would mean guessing — with a wrong guess in the
permissive direction silently leaking the resources the storage check exists to catch.

What is open is the **governance** question, not the engineering one: whether A12 should carry a
CE-numbered approval retrospectively, and whether C8 needs notification of the statement-set
change. C8 does not match on `Statement` exhaustively today, so nothing there breaks; that is a
fact about the current tree, not a guarantee about its direction.
