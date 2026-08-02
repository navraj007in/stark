//! **DEV-128: every hand-built `Operand::Move` in lowering is inventoried here, with a reason.**
//!
//! # Why this test exists
//!
//! INV-MOVE-001 (MIR-0036) rejects a `Move` from a `Copy` place, which is the semantic guarantee.
//! But it only fires when a *program* reaches the site, and that is exactly how four separate
//! instances of one defect surfaced one CI round at a time:
//!
//! | DEV | Site | What reached it |
//! | --- | --- | --- |
//! | 124 | for-loop desugar, both forms | any `for` loop — 12 unit tests |
//! | 125 | provider status→`Result`, out-slot tuple, `?`'s `Err` payload | the REST workload and C7.8, nothing else |
//! | 127 | `borrow_set_receiver` | the DEV-116 HashSet corpus, nothing else |
//!
//! Each was invisible until a workload with the right shape ran. `assign_provider_ok` read its
//! slots through `read_place` and then hand-built the `Move` that wrapped them; `borrow_map_receiver`
//! used `read_place` while `borrow_set_receiver` three lines away did not. In both cases the correct
//! idiom was already present next to the defect.
//!
//! So the operand rule needs a check that fires at AUTHORING time, not execution time. `read_place`
//! is the one function that decides `copy` vs `move` from a type. Anything else spelling
//! `Operand::Move` is asserting the answer, and must say why it is allowed to.
//!
//! # What failing means
//!
//! If this test fails you added, removed, or reworded an `Operand::Move` in `mir/lower.rs`.
//!
//! - **Adding one**: prefer `self.read_place(place, &ty, span)?`. It picks the operand from the
//!   type and, on the move path, transfers drop-flag responsibility — which a hand-built `Move`
//!   silently skips. If the site genuinely cannot use it, add it below WITH the argument for why
//!   the moved place can never be `Copy`. "The tests pass" is not that argument; every one of the
//!   defects above passed the tests that existed when it was written.
//! - **Removing one**: delete its row.
//! - **Reformatting**: update the text. Rows are matched on the trimmed source line, so they
//!   survive line-number churn but not rewording.

use std::collections::BTreeMap;

/// Trimmed source lines mentioning `Operand::Move`, with occurrence counts and the reason each is
/// allowed to name the operand itself.
fn expected_inventory() -> BTreeMap<&'static str, usize> {
    BTreeMap::from([
        // ---- the one legitimate decider: `read_place`'s own move path ----
        //
        // This is the function every other site should be calling. It reaches `Move` only after
        // `is_copy` said no and after clearing the drop flags the moved place covers.
        ("Ok(Operand::Move(place))", 1),
        // ---- provider ABI: types that are structurally never `Copy` ----
        //
        // A `&mut T` out-slot reference. `MirTy::Ref { mutable: true, .. }` is non-`Copy` by the
        // rule itself, so no instantiation of this site can be a `Copy` move.
        ("ops.push(Operand::Move(Place::local(r)));", 1),
        // An A11 `HandleOut` destination. `MirTy::HostResource` is non-`Copy` unconditionally
        // (CD-234) — being non-`Copy` is what makes a resource slot-backed at all.
        ("ops.push(Operand::Move(Place::local(slot)));", 1),
        // ---- drop machinery: droppable implies non-`Copy`, since `Copy + Drop` is forbidden ----
        //
        // Saving a drop unit's value before the place is overwritten. A drop unit exists only for a
        // type with drop glue, and `Copy + Drop` is rejected by the front end.
        (
            "Statement::Assign(Place::local(tmp), Rvalue::Use(Operand::Move(unit_place))),",
            1,
        ),
        // `lower_vec_clear_droppable` pops each element to drop it. It runs ONLY for a droppable
        // element type, so the same `Copy + Drop` argument applies. Examined during DEV-125 and
        // deliberately left alone.
        ("Rvalue::Use(Operand::Move(Place {", 1),
        // ---- ownership transfers where a `Copy` type would be a lowering bug in itself ----
        //
        // The consuming-match scrutinee temp: reached only when `consuming` is set, having already
        // gone through `read_place` to produce the value being re-homed.
        ("Operand::Move(Place::local(temp))", 1),
        // The iterator cursor built for `for x in &v`. A cursor is a `MirTy::Core` iterator type,
        // which the rule classifies non-`Copy`.
        ("Operand::Move(Place::local(cursor))", 1),
        // ---- pattern positions: these READ an operand, they do not construct one ----
        ("Some(op @ (Operand::Copy(_) | Operand::Move(_)))", 1),
        ("if matches!(op, Operand::Move(_)) {", 1),
        (
            "let (Operand::Copy(place) | Operand::Move(place)) = &op else {",
            1,
        ),
        ("let Operand::Move(default_place) = &default_op else {", 1),
        // DEV-146: the borrowed-handle provider argument reads its operand's local to decide
        // whether a `&mut R` needs weakening to `&R`. It INSPECTS the operand it was handed; it
        // constructs nothing, and the weakening it may then perform goes through `weaken_ref_to`,
        // which is itself inventoried above.
        ("Operand::Copy(place) | Operand::Move(place)", 1),
    ])
}

#[test]
fn every_hand_built_move_in_lowering_is_accounted_for() {
    let source = include_str!("../src/mir/lower.rs");

    let mut found: BTreeMap<&str, usize> = BTreeMap::new();
    for line in source.split('\n') {
        // Normalise CRLF at the read: a Windows checkout otherwise leaves a trailing `\r` on every
        // line, so no row would match and the failure would look like a wholesale rewrite.
        let line = line.trim_end_matches('\r').trim();
        // Comments are not operands. `lower.rs` documents this defect family at length — DEV-124
        // through DEV-127 all left an explanation next to their fix — and every one of those
        // paragraphs mentions `Operand::Move` in prose. Counting them made the inventory report a
        // "hand-built move" that was a sentence about hand-built moves.
        if line.starts_with("//") || line.starts_with("*") {
            continue;
        }
        if line.contains("Operand::Move") {
            *found.entry(line).or_insert(0) += 1;
        }
    }

    let expected = expected_inventory();

    let added: Vec<String> = found
        .iter()
        .filter(|(line, count)| expected.get(**line).copied().unwrap_or(0) < **count)
        .map(|(line, count)| {
            format!(
                "  + {line}   (found {count}, inventoried {})",
                expected.get(*line).copied().unwrap_or(0)
            )
        })
        .collect();
    let removed: Vec<String> = expected
        .iter()
        .filter(|(line, count)| found.get(**line).copied().unwrap_or(0) < **count)
        .map(|(line, count)| {
            format!(
                "  - {line}   (inventoried {count}, found {})",
                found.get(*line).copied().unwrap_or(0)
            )
        })
        .collect();

    assert!(
        added.is_empty() && removed.is_empty(),
        "the `Operand::Move` inventory in mir/lower.rs no longer matches this test.\n\n\
         Prefer `self.read_place(place, &ty, span)?` — it selects the operand from the TYPE and \n\
         transfers drop-flag responsibility on the move path, both of which a hand-built `Move` \n\
         skips. If a site truly cannot use it, add a row with the argument for why the moved place \n\
         can never be `Copy`.\n\n\
         unaccounted for:\n{}\n\
         inventoried but gone:\n{}\n",
        if added.is_empty() {
            "  (none)".to_string()
        } else {
            added.join("\n")
        },
        if removed.is_empty() {
            "  (none)".to_string()
        } else {
            removed.join("\n")
        },
    );
}
