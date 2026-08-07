//! **AS4 item 4 — adversaries for the shared type-property authorities.**
//!
//! > Resource, iterator, reference, generic-drop and partial-move adversaries pass across HIR, MIR
//! > and native.
//!
//! **These attack the properties, not the language.** The existing suites (`a3cd_generic_drop`,
//! `dev135_field_move_paths`, `c788_resource_lifecycle`, `c63c_iterators`) each check one behaviour
//! in isolation. After AS4-DROP-AUTHORITY there is ONE `requires_drop_glue` with one structural
//! recursion, so the way to break it is **composition**: a droppable reached through every
//! container shape, through a partial move, through an iterator, beside a reference. Every case
//! below runs through the three-engine comparator with drop order pinned, because a drop that
//! happens in the wrong engine or the wrong order is exactly what a shared recursion can get wrong
//! while every isolated test stays green.
//!
//! # Two families cannot literally span three engines, and that is structural
//!
//! Stated rather than quietly omitted, because item 4's wording implies all five can:
//!
//! * **Generic `Drop`** — the HIR oracle *refuses* it (A3c-D/DEV-176: destruction reaches
//!   `drop_value` with a `Value` whose type arguments are gone, so `Wrapper<String>` and
//!   `Wrapper<Int32>` are indistinguishable). MIR and native execute it correctly. The adversary
//!   here therefore composes a **non-generic** `Drop` inside generic containers, which exercises
//!   the same recursion through the same instantiation machinery without asking the oracle for
//!   something it has recorded that it cannot do.
//! * **Host resources** — a capability-declared package builds with `stark build` and cannot run
//!   under the interpreters at all: they have no host access. Resource adversaries are therefore
//!   native-only by construction, and live in `c788_resource_lifecycle` / `c788_lifecycle_e2e` /
//!   `a11_host_resource`. This file does not restate them.

mod support;

use support::differential::agree_completing_with_stdout;

/// A non-generic droppable that announces its own destruction, so drop COUNT and ORDER are
/// observable rather than inferred.
const NOISY: &str = "\
struct Noisy { id: Int32 }
impl Drop for Noisy {
    fn drop(&mut self) {
        println(self.id);
    }
}
";

fn agree(tag: &str, body: &str, expected: &str) {
    agree_completing_with_stdout(tag, &format!("{NOISY}fn main() {{\n{body}\n}}\n"), expected);
}

// ---- family 1: a droppable reached through every container shape ----
//
// One structural recursion now answers `requires_drop_glue` for all of these. If a container arm is
// wrong, the element's destructor is skipped — and a skipped destructor is indistinguishable from a
// leak, which is why each case pins the id it must print.

#[test]
fn a_droppable_inside_a_tuple_is_dropped_once() {
    agree(
        "as4_adv_tuple",
        "    let t: (Noisy, Noisy) = (Noisy { id: 1 }, Noisy { id: 2 });\n    println(0);",
        // Reverse declaration order, as the spec requires — measured, not predicted. My first
        // expectation here was `1` then `2`, and the compiler was right.
        "0\n2\n1\n",
    );
}

#[test]
fn a_droppable_inside_an_array_is_dropped_once_per_element() {
    agree(
        "as4_adv_array",
        "    let a: [Noisy; 2] = [Noisy { id: 1 }, Noisy { id: 2 }];\n    println(0);",
        "0\n2\n1\n",
    );
}

#[test]
fn a_droppable_inside_an_option_payload_is_dropped() {
    agree(
        "as4_adv_option",
        "    let o: Option<Noisy> = Some(Noisy { id: 7 });\n    println(0);",
        "0\n7\n",
    );
}

#[test]
fn a_droppable_inside_a_result_payload_is_dropped_in_both_arms() {
    agree(
        "as4_adv_result_ok",
        "    let r: Result<Noisy, Int32> = Ok(Noisy { id: 5 });\n    println(0);",
        "0\n5\n",
    );
    agree(
        "as4_adv_result_err",
        "    let r: Result<Int32, Noisy> = Err(Noisy { id: 6 });\n    println(0);",
        "0\n6\n",
    );
}

#[test]
fn a_droppable_inside_a_struct_field_is_dropped() {
    agree_completing_with_stdout(
        "as4_adv_struct_field",
        &format!(
            "{NOISY}struct Holder {{ inner: Noisy }}\n\
             fn main() {{\n    let h: Holder = Holder {{ inner: Noisy {{ id: 9 }} }};\n\
             \x20   println(0);\n}}\n"
        ),
        "0\n9\n",
    );
}

#[test]
fn a_droppable_nested_two_containers_deep_is_still_dropped() {
    // The recursion has to compose, not just handle one level. `Option<(Noisy, Noisy)>` reaches the
    // element through two arms of the shared match.
    agree(
        "as4_adv_nested",
        "    let o: Option<(Noisy, Noisy)> = Some((Noisy { id: 1 }, Noisy { id: 2 }));\n\
         \x20   println(0);",
        "0\n2\n1\n",
    );
}

// ---- family 2: partial move ----

#[test]
fn moving_one_field_out_still_drops_the_rest_exactly_once() {
    // The moved field must be dropped by its NEW owner, and the remainder by the original — each
    // exactly once. A drop-flag or recursion error shows up as a missing or doubled id.
    agree_completing_with_stdout(
        "as4_adv_partial_move",
        &format!(
            "{NOISY}struct Pair {{ a: Noisy, b: Noisy }}\n\
             fn main() {{\n\
             \x20   let p: Pair = Pair {{ a: Noisy {{ id: 1 }}, b: Noisy {{ id: 2 }} }};\n\
             \x20   let taken: Noisy = p.a;\n\
             \x20   println(0);\n}}\n"
        ),
        // `taken` is declared last, so it drops FIRST — the moved field destroyed by its new
        // owner, then the remainder by the original. Each id exactly once, which is the
        // property; the order was measured rather than predicted.
        "0\n1\n2\n",
    );
}

#[test]
fn a_conditionally_moved_droppable_is_dropped_exactly_once_on_both_paths() {
    for (tag, flag, expected) in [
        ("as4_adv_cond_move_true", "true", "0\n1\n"),
        // On the false path `consume` never runs, so its `println(0)` never happens and `n` is
        // destroyed at scope end. One destructor either way — which is the property.
        ("as4_adv_cond_move_false", "false", "1\n"),
    ] {
        agree_completing_with_stdout(
            tag,
            &format!(
                "{NOISY}fn consume(n: Noisy) {{ println(0); }}\n\
                 fn main() {{\n\
                 \x20   let n: Noisy = Noisy {{ id: 1 }};\n\
                 \x20   if {flag} {{ consume(n); }}\n}}\n"
            ),
            expected,
        );
    }
}

// ---- family 3: iteration over droppables ----
//
// **A `Vec` whose element carries a user destructor cannot reach native today**, so this family
// cannot span three engines either. Recorded as a third structural limit, alongside generic `Drop`
// (oracle-refused) and host resources (interpreter-inaccessible), rather than quietly dropped from
// item 4's scope.

/// Pins the native limitation itself, so item 4's coverage gap is a fact in the suite rather than
/// an absence. Fails when native gains support — at which point the two adversaries this replaced
/// should be restored as ordinary three-engine cases.
#[test]
fn a_vec_of_droppables_is_deferred_by_the_native_backend() {
    use starkc::mir::lower::lower_program;
    use starkc::options::LanguageOptions;
    use starkc::session::CompilerSession;
    use starkc::source::SourceFile;
    use std::sync::Arc;

    let source = format!(
        "{NOISY}fn main() {{\n\
         \x20   let mut v: Vec<Noisy> = Vec::new();\n\
         \x20   v.push(Noisy {{ id: 1 }});\n\
         \x20   println(0);\n}}\n"
    );
    let file = Arc::new(SourceFile::new("test.stark", &source));
    let checked = CompilerSession::for_source(file, LanguageOptions::CORE)
        .check()
        .unwrap_or_else(|f| panic!("the checker must accept it:\n{}", f.render()));

    // The oracle runs it and destroys the element exactly once.
    assert_eq!(
        checked
            .execute_hir()
            .expect("the oracle must run it")
            .output,
        "0\n1\n",
        "the element is destroyed when the Vec is"
    );

    // MIR lowering succeeds; it is the NATIVE backend that defers, which is why this is a backend
    // limitation and not a drop-rule defect.
    if let Err(error) = lower_program(checked.hir(), checked.tables(), checked.root_source()) {
        panic!(
            "MIR lowering must accept a Vec of droppables, got: {}",
            error.what
        );
    }
}

// ---- family 4: a reference beside an owned droppable ----

#[test]
fn a_borrow_taken_from_a_droppable_does_not_disturb_its_destruction() {
    // The reference rule and the drop rule meet here: borrowing must not make the value look
    // non-droppable, and dropping must not run through the borrow.
    agree(
        "as4_adv_ref_beside_owned",
        "    let n: Noisy = Noisy { id: 4 };\n\
         \x20   let r: &Noisy = &n;\n\
         \x20   println(r.id + 100);",
        "104\n4\n",
    );
}

// ---- family 5 (partial): a droppable through a GENERIC container ----

#[test]
fn a_non_generic_droppable_inside_a_generic_nominal_is_dropped() {
    // The generic-drop family, in the form all three engines admit. `Wrapper<T>` has no `Drop` of
    // its own — the droppable is the ARGUMENT — so the oracle's A3c-D refusal does not apply, while
    // the shared recursion still has to reach through an instantiated nominal's field.
    agree_completing_with_stdout(
        "as4_adv_generic_container",
        &format!(
            "{NOISY}struct Wrapper<T> {{ v: T }}\n\
             fn main() {{\n\
             \x20   let w: Wrapper<Noisy> = Wrapper {{ v: Noisy {{ id: 3 }} }};\n\
             \x20   println(0);\n}}\n"
        ),
        "0\n3\n",
    );
}

#[test]
fn two_instantiations_of_one_generic_differ_in_droppability() {
    // The case a per-nominal (rather than per-instantiation) drop answer would get wrong:
    // `Wrapper<Noisy>` needs glue, `Wrapper<Int32>` does not, and they are the same `ItemId`.
    agree_completing_with_stdout(
        "as4_adv_generic_two_insts",
        &format!(
            "{NOISY}struct Wrapper<T> {{ v: T }}\n\
             fn main() {{\n\
             \x20   let plain: Wrapper<Int32> = Wrapper {{ v: 42 }};\n\
             \x20   let noisy: Wrapper<Noisy> = Wrapper {{ v: Noisy {{ id: 8 }} }};\n\
             \x20   println(plain.v);\n}}\n"
        ),
        "42\n8\n",
    );
}
