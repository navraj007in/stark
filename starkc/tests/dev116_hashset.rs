//! DEV-116 — `HashSet<T>` across all three engines, and the boundaries that keep it honest.
//!
//! V19 could not be excluded from C6: `HashSet` is normative in the admitted `std-full` surface and
//! the HIR oracle ran it, so the failure was a MIR/native **lowering gap**, and §4.3 forbids
//! recording a lowering gap as a non-Core exclusion. Refusing it would have contradicted the frozen
//! scope rather than narrowed it.
//!
//! The positive evidence is corpus cases (`dev116__hashset_*`), replayed through all three engines
//! by §12. What lives here is the part a passing case cannot show: that the boundaries REFUSE, and
//! that no structural fallback substitutes for a user's `Eq`.
//!
//! **Implementation note, because it is the reason the three engines agree.** `HashSet<T>` is
//! `StarkMap<T, ()>` in every engine — the insertion-ordered entry vector the HIR oracle already
//! used for maps, the same vector in the MIR interpreter, and `stark_runtime::map::StarkMap` in the
//! native backend. The element IS the key, so uniqueness is decided by exactly the comparator
//! dispatch STD-HASH-001 already governs. A second container would have been a second place for
//! identity to drift.

mod support;

use support::differential::{front_end, rustc_available, three_engine, Observation};

/// A nominal element with no recorded `Eq` must be REFUSED, and this pins the phase it is refused
/// at rather than assuming one.
///
/// The phase is MIR **verification** (MIR-0018), which is where `HashMap` already refuses the same
/// shape. Writing this first assumed type-check, because `impl<T: Hash + Eq> HashSet<T>` is a bound
/// — and the assumption was wrong for both collections. The established rule is that a missing
/// instance is a COMPILER defect (lowering failed to record the impl it selected) rather than a
/// rejectable program, so verification is where it surfaces: ahead of either engine, instead of as
/// a wrong answer in the interpreter and a refusal in the backend.
fn refused_at_verification(tag: &str, source: &str) {
    let front = front_end(&format!("dev116_{tag}.stark"), source);
    let program = starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone())
        .unwrap_or_else(|e| panic!("{tag}: expected lowering to succeed, got: {}", e.what));
    let errors = starkc::mir::verify::verify_program(&program)
        .err()
        .unwrap_or_else(|| {
            panic!(
                "{tag}: a set element with no recorded `Eq` was ACCEPTED — its \
                                   identity would fall back to structural comparison in the \
                                   interpreter while the backend refused the same program"
            )
        });
    assert!(
        errors.iter().any(|e| e.code == "MIR-0018"),
        "{tag}: expected MIR-0018, got: {errors:?}"
    );
}

const TAG_WITHOUT: &str = "struct Tag { id: Int32 }\n";

/// Found by writing this test: the guard existed for `HashMap` and I had not extended it to
/// `HashSet`, so a nominal element with no `Eq` was accepted everywhere.
#[test]
fn an_element_without_eq_is_refused() {
    refused_at_verification(
        "no_eq",
        &format!(
            "{TAG_WITHOUT}impl Hash for Tag {{ fn hash(&self) -> UInt64 {{ 1u64 }} }}\n\
             fn main() {{ let mut s: HashSet<Tag> = HashSet::new(); s.insert(Tag {{ id: 1 }}); }}\n"
        ),
    );
}

#[test]
fn an_element_with_neither_eq_nor_hash_is_refused() {
    refused_at_verification(
        "neither",
        &format!(
            "{TAG_WITHOUT}fn main() {{ let mut s: HashSet<Tag> = HashSet::new(); s.insert(Tag {{ id: 1 }}); }}\n"
        ),
    );
}

/// **DEV-118, recorded rather than fixed.** The `Hash` half of the `T: Hash + Eq` bound is NOT
/// enforced: an element with `Eq` and no `Hash` compiles and runs in all three engines. This is a
/// pre-existing gap shared with `HashMap` — the identical program over a `HashMap` key is equally
/// accepted — so it is not something DEV-116 introduced, and fixing bound enforcement for
/// collections generally is outside this change.
///
/// It is benign TODAY for a specific reason worth writing down: CE4 (CD-132) chose an
/// insertion-ordered vector scanned by `Eq`, and `Hash` is never consulted in storage at all. So a
/// missing `Hash` cannot affect membership, ordering or any observation. It becomes a real defect
/// the moment any engine starts narrowing candidates by hash.
///
/// This test pins the CURRENT behaviour so the day that changes, it fails here.
#[test]
fn dev118_the_hash_bound_is_not_enforced_for_either_collection() {
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
        let front = front_end(&format!("dev118_{tag}.stark"), source);
        let program =
            starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone())
                .unwrap_or_else(|e| panic!("{tag}: lowering: {}", e.what));
        assert!(
            starkc::mir::verify::verify_program(&program).is_ok(),
            "{tag}: the `Hash` bound is now enforced — DEV-118 is fixed, so update this test to \
             require the refusal instead of recording its absence"
        );
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

/// `iter` is in the admitted API and is NOT implemented here. Recorded as a test so the boundary is
/// a checked fact rather than a comment: iteration is its own surface (a cursor core type, its ops,
/// and `for` desugaring), and DEV-116 is scoped to the data operations.
#[test]
fn hashset_iteration_is_refused_with_its_own_reason() {
    let front = front_end(
        "dev116_iter.stark",
        "fn main() { let mut s: HashSet<Int32> = HashSet::new(); s.insert(1); let it = s.iter(); }\n",
    );
    let refusal = starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone())
        .err()
        .map(|e| e.what)
        .unwrap_or_default();
    // Refused by the `Iter<T>` RETURN type before reaching the named arm in `lower_set_method_call`
    // — which is still an explicit refusal at lowering, and the honest thing to pin is the refusal
    // that actually happens rather than the one I wrote the arm for.
    assert!(
        refusal.contains("Core(Iter") || refusal.contains("iteration is not part of DEV-116"),
        "iteration must be refused at lowering, got: {refusal:?}"
    );
}
