//! DEV-116 — `HashSet<T>` across all three engines, and the boundaries that keep it honest.
//!
//! V19 could not be excluded from C6: `HashSet` is normative in the admitted `std-full` surface and
//! the HIR oracle ran it, so the failure was a MIR/native **lowering gap**, and §4.3 forbids
//! recording a lowering gap as a non-Core exclusion. Refusing it would have contradicted the frozen
//! scope rather than narrowed it.
//!
//! The positive evidence is corpus cases (`dev116__hashset_*`), replayed through all three engines
//! by §12. What lives here is the part a passing case cannot show: that the boundaries REFUSE, that
//! no structural fallback substitutes for a user's `Eq`, and that every method of the admitted
//! surface actually lowers — the per-method audit V19's per-type row cannot give.
//!
//! `iter` was refused at lowering when the data operations landed, which left the admitted surface
//! only partly executable. DEV-116-B closes it: `iter(&self) -> Iter<T>` yielding `&T`, sharing the
//! map's `KeysIter` cursor in every engine, because a set is `StarkMap<T, ()>` and its keys ARE its
//! elements. No new runtime code, and no new aliasing machinery — the borrow rules fall out of
//! `iter(&self)` versus `insert(&mut self)`.
//!
//! **Implementation note, because it is the reason the three engines agree.** `HashSet<T>` is
//! `StarkMap<T, ()>` in every engine — the insertion-ordered entry vector the HIR oracle already
//! used for maps, the same vector in the MIR interpreter, and `stark_runtime::map::StarkMap` in the
//! native backend. The element IS the key, so uniqueness is decided by exactly the comparator
//! dispatch STD-HASH-001 already governs. A second container would have been a second place for
//! identity to drift.

mod support;

use support::differential::{front_end, rustc_available, three_engine, Observation};

/// A nominal element with no `Eq` must be REFUSED, and this pins the phase it is refused at rather
/// than assuming one.
///
/// **The phase moved with WP-C7.9 Packet I (DEV-118).** It used to be MIR *verification*
/// (`MIR-0018`) — reached because nothing checked the declared bound, so an element type with no
/// `Eq` got as far as lowering and was caught only when the verifier found no instance to call.
/// The reasoning recorded at the time was that a missing instance is a compiler defect rather than
/// a rejectable program; that was true of the SYMPTOM, and the cause was that `HashSet<T>`'s own
/// `T: Hash + Eq` bound was never enforced anywhere.
///
/// It is enforced now, at type checking, where every other unsatisfied bound is rejected — so the
/// program never reaches lowering at all and the verifier's guard becomes the defence in depth it
/// was always meant to be.
fn refused_at_type_checking(tag: &str, source: &str) {
    support::differential::rejects_at_typecheck(&format!("dev116_{tag}.stark"), source, "E0500");
}

const TAG_WITHOUT: &str = "struct Tag { id: Int32 }\n";

/// Found by writing this test: the guard existed for `HashMap` and I had not extended it to
/// `HashSet`, so a nominal element with no `Eq` was accepted everywhere.
#[test]
fn an_element_without_eq_is_refused() {
    refused_at_type_checking(
        "no_eq",
        &format!(
            "{TAG_WITHOUT}impl Hash for Tag {{ fn hash(&self) -> UInt64 {{ 1u64 }} }}\n\
             fn main() {{ let mut s: HashSet<Tag> = HashSet::new(); s.insert(Tag {{ id: 1 }}); }}\n"
        ),
    );
}

#[test]
fn an_element_with_neither_eq_nor_hash_is_refused() {
    refused_at_type_checking(
        "neither",
        &format!(
            "{TAG_WITHOUT}fn main() {{ let mut s: HashSet<Tag> = HashSet::new(); s.insert(Tag {{ id: 1 }}); }}\n"
        ),
    );
}

/// **DEV-118 is FIXED (WP-C7.9 Packet I).** The `Hash` half of `T: Hash + Eq` is enforced for both
/// collections, at type checking.
///
/// This test previously recorded the opposite, and said what would make it fail: *"the `Hash` bound
/// is now enforced — DEV-118 is fixed, so update this test to require the refusal instead of
/// recording its absence."* That is what happened, so it now requires the refusal.
///
/// Why the omission mattered even while it was benign: CE4 (CD-132) chose an insertion-ordered
/// vector scanned by `Eq`, and `Hash` is never consulted in storage — so a missing `Hash` could not
/// affect membership, ordering, or any observation, and **all three engines agreed** on every
/// invalid program. No differential could see it. It would have become a real divergence the moment
/// one engine began narrowing candidates by hash.
#[test]
fn dev118_the_hash_bound_is_enforced_for_both_collections() {
    for (tag, source) in [
        (
            "set",
            "struct Tag { id: Int32 }\n\
             impl Eq for Tag { fn eq(&self, o: &Tag) -> Bool { self.id == o.id } }\n\
             fn main() { let mut s: HashSet<Tag> = HashSet::new(); s.insert(Tag { id: 1 }); }\n",
        ),
        (
            "map",
            "struct Tag { id: Int32 }\n\
             impl Eq for Tag { fn eq(&self, o: &Tag) -> Bool { self.id == o.id } }\n\
             fn main() { let mut m: HashMap<Tag, Int32> = HashMap::new(); m.insert(Tag { id: 1 }, 2); }\n",
        ),
    ] {
        support::differential::rejects_at_typecheck(&format!("dev118_{tag}.stark"), source, "E0500");
    }
}

/// CE4 (CD-132) is unchanged for sets: a user-`Drop` element is refused before MIR, which is what
/// keeps entry destruction order legitimately unspecified.
#[test]
fn a_drop_bearing_element_is_still_refused_before_mir() {
    let source = "struct D { v: Int32 }\n\
                  impl Drop for D { fn drop(&mut self) {} }\n\
                  impl Eq for D { fn eq(&self, other: &D) -> Bool { self.v == other.v } }\n\
                  impl Hash for D { fn hash(&self) -> UInt64 { 1u64 } }\n\
                  fn main() { let mut s: HashSet<D> = HashSet::new(); s.insert(D { v: 1 }); }\n";
    let front = front_end("dev116_drop.stark", source);
    let refusal = starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone())
        .err()
        .map(|e| e.what)
        .unwrap_or_default();
    assert!(
        refusal.contains("user-Drop"),
        "a Drop-bearing element must be refused at lowering, got: {refusal:?}"
    );
}

/// **The substitution control.** This is the one that matters most: it proves the user's `Eq` is
/// what decides membership, in every engine, rather than a structural comparison that happens to
/// agree with it.
///
/// `Tag` is equal on `id` alone. Two values with the same `id` and DIFFERENT `note` are equal by the
/// user's implementation and unequal structurally, so the two rules give different answers — one
/// entry versus two. Running it through all three engines is what makes this a substitution test
/// rather than a restatement of the corpus case.
#[test]
fn user_eq_decides_membership_in_every_engine_not_structural_equality() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let observation = three_engine(
        "dev116_user_eq_authoritative",
        "struct Tag { id: Int32, note: Int32 }\n\
         impl Eq for Tag { fn eq(&self, other: &Tag) -> Bool { self.id == other.id } }\n\
         impl Hash for Tag { fn hash(&self) -> UInt64 { 7u64 } }\n\
         fn main() {\n\
             let mut s: HashSet<Tag> = HashSet::new();\n\
             s.insert(Tag { id: 1, note: 100 });\n\
             s.insert(Tag { id: 1, note: 999 });\n\
             println(s.len());\n\
         }\n",
    );
    match observation {
        Observation::Completed(done) => assert_eq!(
            String::from_utf8_lossy(&done.stdout_bytes),
            "1\n",
            "the set must hold ONE element — structural comparison would give 2, which is exactly \
             the substitution this proves does not happen"
        ),
        other => panic!("expected completion, got {other:#?}"),
    }
}

/// A lawful `Hash` that is CONSTANT puts every element in one bucket, so if anything ever narrows
/// candidates by hash, `Eq` still has to separate them. STD-HASH-001: "unequal keys with equal
/// hashes remain distinct."
#[test]
fn total_hash_collision_keeps_unequal_elements_distinct() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let observation = three_engine(
        "dev116_hash_collision",
        "struct Tag { id: Int32 }\n\
         impl Eq for Tag { fn eq(&self, other: &Tag) -> Bool { self.id == other.id } }\n\
         impl Hash for Tag { fn hash(&self) -> UInt64 { 42u64 } }\n\
         fn main() {\n\
             let mut s: HashSet<Tag> = HashSet::new();\n\
             s.insert(Tag { id: 1 });\n\
             s.insert(Tag { id: 2 });\n\
             s.insert(Tag { id: 3 });\n\
             println(s.len());\n\
         }\n",
    );
    match observation {
        Observation::Completed(done) => assert_eq!(
            String::from_utf8_lossy(&done.stdout_bytes),
            "3\n",
            "three unequal elements sharing one hash must remain three entries"
        ),
        other => panic!("expected completion, got {other:#?}"),
    }
}

// ---------------------------------------------------- DEV-116-B: iteration --

/// The set may not be MUTATED while its iterator, or a reference it yielded, is live.
///
/// `iter(&self)` takes a shared borrow and `insert`/`remove`/`clear` take an exclusive one, so this
/// is the ordinary borrow rule rather than a special case — which is the point: iteration needed no
/// new aliasing machinery. Rejected at TYPE-CHECK with E0101, before any engine runs.
#[test]
fn the_set_cannot_be_mutated_while_an_iterator_is_live() {
    for (tag, source) in [
        (
            "insert_while_iterating",
            "fn main() { let mut s: HashSet<Int32> = HashSet::new(); s.insert(1); \
             for x in s.iter() { s.insert(2); } }",
        ),
        (
            "remove_while_iterating",
            "fn main() { let mut s: HashSet<Int32> = HashSet::new(); s.insert(1); \
             for x in s.iter() { s.remove(&1); } }",
        ),
        (
            "clear_while_iterating",
            "fn main() { let mut s: HashSet<Int32> = HashSet::new(); s.insert(1); \
             for x in s.iter() { s.clear(); } }",
        ),
        (
            // The cursor is held in a local, so the borrow outlives the call that made it.
            "mutate_while_a_held_cursor_is_live",
            "fn main() { let mut s: HashSet<Int32> = HashSet::new(); s.insert(1); \
             let it = s.iter(); s.insert(2); }",
        ),
    ] {
        let file = std::sync::Arc::new(starkc::source::SourceFile::new(
            format!("dev116b_{tag}.stark"),
            source.to_string(),
        ));
        let (ast, parse_diags) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
        assert!(parse_diags.is_empty(), "{tag}: must parse");
        let (hir, resolve_diags) = starkc::resolve::resolve(&ast, file.clone());
        assert!(resolve_diags.is_empty(), "{tag}: must resolve");
        let checked = starkc::typecheck::analyze(&hir);
        let errors: Vec<_> = checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == starkc::diag::Severity::Error)
            .collect();
        assert!(
            errors.iter().any(|d| d.code.as_deref() == Some("E0101")),
            "{tag}: mutating a set while it is borrowed for iteration must be rejected with \
             E0101, got: {errors:?}"
        );
    }
}

/// **The per-method admitted-surface audit.** V19 is a per-TYPE row, and DEV-115 showed that a
/// per-type row can hide a method-level gap indefinitely — `str` was "covered" while `str::bytes`
/// diverged across engines. So the surface is enumerated from the specification and each member
/// checked to LOWER, which is where every gap in this work package actually lived.
///
/// `06-Standard-Library.md`: `new`, `insert`, `remove`, `contains`, `len`, `is_empty`, `clear`,
/// `iter`. All eight, no more — `with_capacity` is described for collections generally but is not
/// in `HashSet`'s impl block.
#[test]
fn every_admitted_hashset_method_lowers() {
    let bodies = [
        ("new", "let s: HashSet<Int32> = HashSet::new();"),
        (
            "insert",
            "let mut s: HashSet<Int32> = HashSet::new(); let a: Bool = s.insert(1);",
        ),
        (
            "remove",
            "let mut s: HashSet<Int32> = HashSet::new(); let a: Bool = s.remove(&1);",
        ),
        (
            "contains",
            "let s: HashSet<Int32> = HashSet::new(); let a: Bool = s.contains(&1);",
        ),
        (
            "len",
            "let s: HashSet<Int32> = HashSet::new(); let a: UInt64 = s.len();",
        ),
        (
            "is_empty",
            "let s: HashSet<Int32> = HashSet::new(); let a: Bool = s.is_empty();",
        ),
        (
            "clear",
            "let mut s: HashSet<Int32> = HashSet::new(); s.clear();",
        ),
        (
            "iter",
            "let s: HashSet<Int32> = HashSet::new(); for x in s.iter() { print(*x); }",
        ),
    ];
    let mut refused = Vec::new();
    for (method, body) in bodies {
        let source = format!("fn main() {{ {body} }}\n");
        let front = front_end(&format!("dev116_surface_{method}.stark"), &source);
        if let Err(e) =
            starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone())
        {
            refused.push(format!("{method}: {}", e.what));
        }
    }
    assert!(
        refused.is_empty(),
        "{} admitted HashSet method(s) do not lower, so the admitted surface is not executable \
         across all three engines:\n  {}",
        refused.len(),
        refused.join("\n  ")
    );
}

/// Iteration BORROWS: elements are not moved out, so a non-Copy element survives traversal and the
/// set is fully usable afterwards. `&T` is what the specification's iterator table says `iter`
/// yields, and this is the observation that would fail if it yielded `T`.
#[test]
fn iteration_borrows_non_copy_elements_rather_than_moving_them() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let observation = three_engine(
        "dev116b_borrowing_iteration",
        "fn main() {\n\
             let mut s: HashSet<String> = HashSet::new();\n\
             s.insert(String::from(\"aa\"));\n\
             s.insert(String::from(\"bbb\"));\n\
             let mut total: UInt64 = 0u64;\n\
             for value in s.iter() { total = total + value.len(); }\n\
             let mut second: UInt64 = 0u64;\n\
             for value in s.iter() { second = second + value.len(); }\n\
             print(total);\n\
             print(\"|\");\n\
             print(second);\n\
             print(\"|\");\n\
             println(s.len());\n\
         }\n",
    );
    match observation {
        Observation::Completed(done) => assert_eq!(
            String::from_utf8_lossy(&done.stdout_bytes),
            "5|5|2\n",
            "iteration must borrow: both traversals see every element and the set survives intact"
        ),
        other => panic!("expected completion, got {other:#?}"),
    }
}

/// First-insertion order, which 06-Standard-Library makes NORMATIVE for `HashSet::iter` — "MUST
/// visit entries in first-insertion order", deterministic across conforming implementations.
///
/// Pinned directly rather than through an order-independent aggregate: the order is guaranteed, so
/// asserting only a sum or a count would assert less than the specification promises. The elements
/// are inserted in an order that is neither sorted nor reverse-sorted, so a backend that happened to
/// sort, or to reverse, is caught.
#[test]
fn iteration_visits_elements_in_first_insertion_order() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let observation = three_engine(
        "dev116b_insertion_order",
        "fn main() {\n\
             let mut s: HashSet<Int32> = HashSet::new();\n\
             s.insert(30);\n\
             s.insert(10);\n\
             s.insert(20);\n\
             s.insert(10);\n\
             for value in s.iter() { print(*value); print(\",\"); }\n\
             println(\"\");\n\
         }\n",
    );
    match observation {
        Observation::Completed(done) => assert_eq!(
            String::from_utf8_lossy(&done.stdout_bytes),
            "30,10,20,\n",
            "first-insertion order is normative; a duplicate insert must not reposition an element"
        ),
        other => panic!("expected completion, got {other:#?}"),
    }
}
