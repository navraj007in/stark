# C6-DROP-PATH-MATRIX — Track A / WP-C6.1e

**Status:** COMPLETE (C6.1e) — every normative exit path classified with three-engine evidence
**Track:** A (ownership, partial moves, Drop)
**Base:** `main`, post-C6.1d

## 1. Observation channel (why the cases look like this)

Native execution still has **no stdout** (`Callee::Runtime` is unsupported until C6.3, and
`NATIVE_STDOUT_SUPPORTED` is `false`), and STARK has no globals or reference fields, so a destructor
cannot record its own firing anywhere a later assertion could read.

What **is** observable in all three engines is a **trap**: its category and exact
`file:line:column`. So C6.1e reuses the C5.3d-1c technique — a **trapping destructor as a position
probe** — in two shapes:

| Shape | Construction | What a wrong answer looks like |
|---|---|---|
| **Normal exit** — the destructor MUST run | the destructor traps; the case is built so it is the first trap reachable | completing, or reporting a different line, means the value was never destroyed |
| **Abnormal exit** — the destructor must NOT run | the destructor traps *and* the body traps first | reporting the destructor's line means cleanup ran after an abort |

Every expected line is derived from the language rule, not from any engine's answer. Byte-level
Drop *log* comparison (a destructor that prints a marker) is **not** required by this matrix and
lands naturally with C6.3 output + the C6.5 `drop_log` comparator; the probe above already decides
order and firing without it.

## 2. Normal exits

| Exit path | Rule | Evidence | Status |
|---|---|---|---|
| inner block scope | a block-local is destroyed at block end | `c61e_an_inner_block_local_is_destroyed_at_block_end` | ✅ 3-engine |
| function / method body | a local is destroyed at function end | C5.3d-1c `a_user_destructor_with_a_self_receiver_agrees`, and every normal-exit case below | ✅ 3-engine |
| **loop body** | a body-local is destroyed **each iteration**, not accumulated | `c61e_a_loop_body_local_is_destroyed_each_iteration` | ✅ 3-engine |
| **`break`** | a live binding is destroyed when leaving the loop | `c61e_a_local_live_at_break_is_destroyed` | ✅ 3-engine |
| **`continue`** | a live binding is destroyed before the next iteration | `c61e_a_local_live_at_continue_is_destroyed` | ✅ 3-engine |
| **`return`** | live locals are destroyed on return | `c61e_a_local_is_destroyed_on_return` | ✅ 3-engine |
| **`?` propagation** | live locals are destroyed when `?` propagates | `c61e_a_local_is_destroyed_when_question_mark_propagates` | ✅ 3-engine |
| **match arm** | an arm binding is destroyed at arm end | `c61e_a_match_arm_binding_is_destroyed_at_arm_end` | ✅ 3-engine |
| **failed pattern binding** | a failed pattern test neither consumes nor leaks the scrutinee | `c61e_a_failed_pattern_test_leaves_the_scrutinee_for_the_matching_arm` | ✅ 3-engine |
| normal `main` completion | top-level locals destroyed | C5.3d-1c drop-order cases | ✅ 3-engine |

### Order and count (C5.3d-1c, still in force)

| Rule | Evidence |
|---|---|
| a type's own destructor runs **before** its fields | `own_destructor_runs_before_fields` |
| fields destroyed in **reverse declaration order** | `struct_fields_are_destroyed_in_reverse_declaration_order` |
| an enum destroys only the **active** variant payload | `enum_destroys_the_active_variant_payload_a/_b` |
| a moved value is destroyed by its **new owner** | `a_moved_value_is_destroyed_by_its_new_owner` |
| a moved value is destroyed **exactly once** | `a_moved_value_is_destroyed_exactly_once` |
| a partially moved value destroys only the **surviving** field | `a_partially_moved_value_destroys_only_the_surviving_field` |
| arrays destroy elements in **reverse index order**, only still-initialised ones | array `DropPlan`; exercised by C6.1d `c61d_break_*` / `c61d_return` / `c61d_try` |

## 3. Abnormal exits — a trap performs NO cleanup

| Trap category | Evidence | Status |
|---|---|---|
| divide by zero | C5.3d-1c `no_destructor_runs_after_a_trap` | ✅ 3-engine |
| **integer overflow** | `c61e_no_destructor_runs_after_an_overflow_trap` | ✅ 3-engine |
| **cast failure** | `c61e_no_destructor_runs_after_a_cast_trap` | ✅ 3-engine |
| **index out of bounds** | `c61e_no_destructor_runs_after_an_index_trap` | ✅ 3-engine |
| **assertion failure** (the message-less `panic` equivalent) | `c61e_no_destructor_runs_after_an_assertion_failure` | ✅ 3-engine |
| explicit MIR trap | same abort path as the above (`Terminator::Trap`); message-carrying `panic("…")` needs `&str` values and is C6.3 | ✅ via the above / ⏸ message form C6.3 |
| IO / provider failure | not normative until the runtime resource surface exists | ⏸ C6.3 (Track C) |

## 4. Ownership-gap interaction (C6.1b–d)

Each closed gap carries its own drop evidence, all three-engine:

- **G3** multi-level partial move — the moved deep unit is not re-dropped (`multi_level_partial_move_of_drop_types_does_not_double_drop`).
- **G4** loop-carried no-`Drop` reassignment — no false live-slot write.
- **G1** multi-unit enum payload — moved units not dropped, unbound siblings dropped exactly once (`c61c_*`).
- **G2** array by-value iteration — binding dropped per iteration; unconsumed tail dropped on `break`/`return`/`?`; nothing after a trap (`c61d_*`).

A wrong drop in any of these trips `slot_violation` (a non-zero native exit), so the exit-0/expected-trap
assertions are themselves double/missing-drop guards.

## 5. C6.1e closure

- [x] Normal exits: scope, function, loop body, break, continue, return, match arm, failed pattern binding, `?`, main completion
- [x] Abnormal exits: overflow, divide-by-zero, cast, index, assertion — no cleanup after any
- [x] Order/count rules (own-before-fields, reverse order, active variant, exactly-once, partial survivor, array reverse index)
- [x] Every row observed through category + exact source location in **all three engines** (exit code alone is not relied on)
- [~] Byte-level Drop-log comparison — not required by the probe method; arrives with C6.3 output and the C6.5 `drop_log` comparator
- [~] IO/provider-failure cleanup — not normative until C6.3's resource surface

**C6.1e complete.** With C6.1a–e closed, **WP-C6.1 (ownership and Drop parity) is ready for closure**
pending the full-suite run.
