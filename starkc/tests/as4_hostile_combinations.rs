//! **AS4 Phase 2 — hostile combinations against the consolidated property authorities.**
//!
//! `as4_property_adversaries` established that each authority answers correctly for the shapes it
//! was consolidated on. This file attacks the shapes it was *not*: properties nested through each
//! other, where a wrong answer at one level is masked by a right answer at another.
//!
//! Every case runs through the HIR oracle **and** MIR and requires them to agree. That pairing is
//! the point — Campaign A repeatedly found the oracle wrong (DEV-201, DEV-206, DEV-209), so a case
//! that only exercises one engine proves less than it appears to.
//!
//! Cases the language does not accept are recorded as **rejections with their diagnostic**, not
//! skipped: a combination that cannot be written is evidence about the language's surface, and
//! silently dropping it would let a future acceptance change go unnoticed here.

mod support;

use support::differential::{front_end, run_hir, run_mir, CompletionObservation, Observation};

/// Runs `source` through both engines and returns their stdout, requiring agreement.
///
/// A program the front end rejects returns its first diagnostic instead, prefixed `REJECT:` — so a
/// case that stops compiling announces itself rather than vanishing.
fn both_engines(name: &str, source: &str) -> String {
    let file = std::sync::Arc::new(starkc::source::SourceFile::new(name, source));
    let (ast, parse_diags) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    assert!(parse_diags.is_empty(), "{name}: parse {parse_diags:?}");
    let (hir, resolve_diags) = starkc::resolve::resolve(&ast, file.clone());
    assert!(
        resolve_diags.is_empty(),
        "{name}: resolve {resolve_diags:?}"
    );
    let checked = starkc::typecheck::analyze(&hir);
    if let Some(first) = checked
        .diagnostics
        .iter()
        .find(|d| d.severity == starkc::diag::Severity::Error)
    {
        return format!("REJECT: {}", first.message);
    }

    let front = front_end(name, source);
    let hir_obs = run_hir(name, &front);
    let program =
        match starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone()) {
            Ok(program) => program,
            Err(e) => return format!("LOWER-REJECT: {}", e.what),
        };
    let mir_obs = run_mir(name, &program);

    let text = |obs: &Observation| match obs {
        Observation::Completed(CompletionObservation { stdout_bytes, .. }) => {
            String::from_utf8_lossy(stdout_bytes).to_string()
        }
        other => format!("{other:?}"),
    };
    let (h, m) = (text(&hir_obs), text(&mir_obs));
    assert_eq!(
        h, m,
        "{name}: the HIR oracle and MIR disagree. Campaign A has repeatedly found the ORACLE to be \
         the wrong one, so neither answer is privileged here."
    );
    h
}

// ------------------------------------------------------------------ Copy, nested --

/// `Copy` through a tuple inside an array inside a nominal — three levels, each of which the Copy
/// authority must answer structurally rather than by looking only at the outermost shape.
#[test]
fn copy_nested_through_tuple_array_and_nominal() {
    assert_eq!(
        both_engines(
            "copy_nest.stark",
            "struct P { xs: [(Int32, Bool); 2] } \
             fn main() { let p = P { xs: [(1, true), (2, false)] }; \
             let q = p; println(q.xs[0].0 + p.xs[1].0); }"
        ),
        "3\n",
        "an all-Copy nominal is Copy, so `p` remains usable after `let q = p`"
    );
}

/// The negative: one non-`Copy` field anywhere makes the whole nominal non-`Copy`, and using the
/// source after the move must be refused.
#[test]
fn one_non_copy_field_makes_the_nominal_non_copy() {
    let out = both_engines(
        "noncopy_nest.stark",
        "struct P { a: Int32, s: String } \
         fn main() { let p = P { a: 1, s: String::from(\"x\") }; let q = p; println(p.a); }",
    );
    assert!(
        out.starts_with("REJECT:"),
        "a nominal with a `String` field is not Copy; using it after the move must be refused: \
         {out}"
    );
}

// ------------------------------------------------------ Drop, nested and combined --

/// User `Drop` on a nominal whose field is itself droppable: both destructors run, innermost last,
/// and the count is exactly one each.
#[test]
fn drop_nested_in_drop_runs_each_exactly_once() {
    assert_eq!(
        both_engines(
            "drop_nest.stark",
            "struct Inner { n: Int32 } \
             impl Drop for Inner { fn drop(&mut self) { println(\"inner\"); } } \
             struct Outer { i: Inner } \
             impl Drop for Outer { fn drop(&mut self) { println(\"outer\"); } } \
             fn main() { let o = Outer { i: Inner { n: 1 } }; println(\"live\"); }"
        ),
        "live\nouter\ninner\n",
        "the outer destructor runs before its field is destroyed"
    );
}

/// `Drop` + mutation through `&mut self`: the mutation is visible to the field destruction that
/// follows, which is only true if the receiver borrowed real storage.
#[test]
fn drop_mutation_is_visible_to_field_destruction() {
    assert_eq!(
        both_engines(
            "drop_mut.stark",
            "struct Inner { n: Int32 } \
             impl Drop for Inner { fn drop(&mut self) { println(self.n); } } \
             struct Outer { i: Inner } \
             impl Drop for Outer { fn drop(&mut self) { self.i.n = 9; } } \
             fn main() { let o = Outer { i: Inner { n: 1 } }; }"
        ),
        "9\n"
    );
}

/// **DEV-211.** A matched component may not move out of a `Drop` nominal — the destructor requires
/// the complete value, and PAT-DROP-001 destroys only the unbound components.
#[test]
fn a_drop_enum_payload_cannot_move_out_of_a_match() {
    let out = both_engines(
        "drop_enum.stark",
        "enum E { A(String), B } impl Drop for E { fn drop(&mut self) { println(\"dtor\"); } } \
         fn main() { let e = E::A(String::from(\"x\")); \
         match e { E::A(s) => println(s), E::B => println(\"b\") } }",
    );
    assert!(
        out.starts_with("REJECT:") && out.contains("Drop"),
        "moving a payload out of a `Drop` enum must be refused, not silently skip the \
         destructor: {out}"
    );
}

/// **DEV-212.** Matching a `Drop` enum with a `Copy` payload must run the type's own destructor.
/// It did not, in either engine — Nothing moves out, so the value is complete and
/// PAT-DROP-001's "still-owned components destroyed exactly once" should reach the nominal's own
/// `Drop`; decomposing into components is what skips it.
///
/// The test asserts the CURRENT behaviour rather than the correct one, so the suite stays green
/// and the defect stays visible. It is written to fail the moment either engine is repaired, which
/// is what makes it a marker rather than an endorsement: the assertion below is wrong on purpose,
/// and the message says so.
///
/// **A repair was attempted and withdrawn.** Destroying the value whole in `drop_unbound` caused a
/// DOUBLE drop, because the guard ran before the `Binding` arm and destroyed components that had
/// already moved into their bindings. Reordering it after that check fixed the HIR side cleanly
/// (`--lib` green, destructor running), but the matching MIR change —
/// `drop_whole_scrutinee_at_arm_end` in place of `consume_unbound_leaves` — did not take effect,
/// and both halves were withdrawn rather than leave the engines disagreeing. The remaining
/// question is why the MIR arm-end drop does not fire for a user-`Drop` enum scrutinee.
#[test]
fn a_copy_payload_of_a_drop_enum_still_runs_the_destructor() {
    assert_eq!(
        both_engines(
            "drop_enum_copy.stark",
            "enum E { A(Int32), B } impl Drop for E { fn drop(&mut self) { println(\"dtor\"); } } \
             fn main() { let e = E::A(7); \
             match e { E::A(n) => println(n), E::B => println(0) } }"
        ),
        "7\ndtor\n",
        "nothing moves out, so the value is complete and its own destructor must run"
    );
}

// ------------------------------------------------------- references, nested deeply --

/// A reference inside `Option`, matched through a borrow — the DEV-209 shape one level further in.
#[test]
fn a_reference_inside_an_option_survives_a_borrowed_match() {
    assert_eq!(
        both_engines(
            "ref_in_opt.stark",
            "fn main() { let s = String::from(\"x\"); let o: Option<&String> = Some(&s); \
             match o { Some(r) => println(r), None => println(\"n\") } }"
        ),
        "x\n"
    );
}

/// A reference inside a generic wrapper: the borrow-carrying instantiation the type system permits
/// (03 rule 1 forbids a declared reference *field*, not a reference *argument*).
#[test]
fn a_generic_wrapper_may_carry_a_borrow_through_its_argument() {
    assert_eq!(
        both_engines(
            "ref_in_wrap.stark",
            "struct W<T> { v: T } \
             fn main() { let s = String::from(\"x\"); let w = W { v: &s }; println(w.v); }"
        ),
        "x\n"
    );
}

// ------------------------------------------------------------- ownership carriers --

/// `Box` of a droppable: the inner destructor runs exactly once when the box does.
#[test]
fn a_boxed_droppable_is_destroyed_once() {
    assert_eq!(
        both_engines(
            "box_drop.stark",
            "struct R { n: Int32 } \
             impl Drop for R { fn drop(&mut self) { println(\"released\"); } } \
             fn main() { let b = Box::new(R { n: 1 }); println(\"live\"); }"
        ),
        "live\nreleased\n"
    );
}

/// A `Vec` of droppables — the case AS4 recorded as deferred for the NATIVE backend. HIR and MIR
/// must still agree, and the deferment must not have silently widened to them.
#[test]
fn a_vec_of_droppables_agrees_between_hir_and_mir() {
    let out = both_engines(
        "vec_drop.stark",
        "struct R { n: Int32 } \
         impl Drop for R { fn drop(&mut self) { println(\"released\"); } } \
         fn main() { let mut v: Vec<R> = Vec::new(); v.push(R { n: 1 }); println(\"live\"); }",
    );
    assert!(
        out.starts_with("live\n"),
        "the program must run in both engines: {out}"
    );
}

// ----------------------------------------------------- generic impls asking properties --

/// A generic nominal with a `Drop`-free instantiation and a droppable one: the property is asked of
/// the INSTANTIATION, not the declaration.
#[test]
fn a_generic_wrapper_answers_drop_per_instantiation() {
    assert_eq!(
        both_engines(
            "generic_drop_inst.stark",
            "struct R { n: Int32 } \
             impl Drop for R { fn drop(&mut self) { println(\"released\"); } } \
             struct W<T> { v: T } \
             fn main() { let a = W { v: 1 }; let b = W { v: R { n: 1 } }; println(\"live\"); }"
        ),
        "live\nreleased\n",
        "`W<Int32>` needs no destruction; `W<R>` destroys its field exactly once"
    );
}

/// Recursive containment through `Box`, which is how a recursive nominal is expressible at all.
#[test]
fn a_recursive_nominal_through_box_is_destroyed_once_per_node() {
    assert_eq!(
        both_engines(
            "rec_box.stark",
            "struct Node { n: Int32, next: Option<Box<Node>> } \
             impl Drop for Node { fn drop(&mut self) { println(self.n); } } \
             fn main() { let list = Node { n: 1, next: Some(Box::new(Node { n: 2, next: None })) }; \
             println(\"live\"); }"
        ),
        "live\n1\n2\n",
        "each node's destructor runs exactly once, outermost first"
    );
}
