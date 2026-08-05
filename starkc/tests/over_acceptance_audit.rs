//! **Over-acceptance repairs: DEV-169 and DEV-171.**
//!
//! Both were programs the language forbids and the checker accepted. That is a different and worse
//! category than the refusals recorded elsewhere in the ledger (DEV-167, DEV-168, DEV-172), which
//! reject valid programs: a refusal is visible the moment you compile, and an over-acceptance ships.
//!
//! * **DEV-169** — `r.drop()` on a type with `impl Drop` was accepted, and the destructor then ran
//!   **twice**: once for the call and again at scope end. Verified before the fix, output
//!   `dropped / after / dropped`. 03-Type-System.md, "Copy and Drop": "`Drop::drop` MUST NOT be
//!   called explicitly; use the free function `drop(value)`."
//! * **DEV-171** — an unrelated trait imported as `Eq` authorised `==`, because the operator check
//!   compared the bound's SPELLING against `"Eq"`. Written qualified the same program was rejected,
//!   which is the tell.
//!
//! Each fix is paired here with the cases that must keep working, because the risk in both repairs
//! is over-rejection: a method merely NAMED `drop`, and a real `Eq`/`Ord`/`Num` bound.

mod support;

use support::differential::{agree_completing_with_stdout, rejects_at_typecheck};

// --------------------------------------------------------------- DEV-169: explicit destructors --

/// The reported program. It must be rejected, and the reason must name the free function.
#[test]
fn an_explicit_drop_call_is_rejected() {
    let messages = rejects_at_typecheck(
        "dev169_explicit_drop",
        "\
struct Resource {
    id: Int32,
}

impl Drop for Resource {
    fn drop(&mut self) {
        println(\"dropped\");
    }
}

fn main() {
    let mut resource = Resource { id: 1 };
    resource.drop();
}
",
        "E0307",
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("'Drop::drop' cannot be called explicitly")),
        "expected the explicit-drop rejection, got {messages:?}"
    );
}

/// A `drop` method that is NOT a `Drop` implementation is an ordinary method and stays callable.
/// The check keys on the impl the call resolved INTO, not on the method's name — this is what
/// would fail if it keyed on the name.
#[test]
fn an_inherent_method_named_drop_is_unaffected() {
    agree_completing_with_stdout(
        "dev169_inherent_drop",
        "\
struct Queue {
    n: Int32,
}

impl Queue {
    fn drop(&mut self) -> Int32 {
        self.n = self.n - 1;
        self.n
    }
}

fn main() {
    let mut q = Queue { n: 3 };
    println(q.drop());
    println(q.drop());
}
",
        "2\n1\n",
    );
}

/// The sanctioned way to destroy early still works, and still destroys **exactly once**: `released`
/// appears before `after`, and not again at scope end. That ordering is the whole point — it is
/// what the rejected form got wrong.
#[test]
fn the_free_function_drop_destroys_exactly_once() {
    agree_completing_with_stdout(
        "dev169_free_drop",
        "\
struct Res {
    id: Int32,
}

impl Drop for Res {
    fn drop(&mut self) {
        println(\"released\");
    }
}

fn main() {
    let r = Res { id: 1 };
    drop(r);
    println(\"after\");
}
",
        "released\nafter\n",
    );
}

/// And automatic destruction is untouched: exactly one `released`, at scope end.
#[test]
fn automatic_destruction_still_runs_once() {
    agree_completing_with_stdout(
        "dev169_automatic_drop",
        "\
struct Res {
    id: Int32,
}

impl Drop for Res {
    fn drop(&mut self) {
        println(\"released\");
    }
}

fn main() {
    let r = Res { id: 1 };
    println(\"before\");
}
",
        "before\nreleased\n",
    );
}

// ------------------------------------------------------------- DEV-171: operator bound identity --

/// The reported program: a trait imported as `Eq` that is not the Core `Eq`.
#[test]
fn an_unrelated_trait_named_eq_does_not_authorise_equality() {
    rejects_at_typecheck(
        "dev171_fake_eq",
        "\
mod fake {
    pub trait Eq {
        fn unrelated(&self) -> Int32;
    }
}

use fake::Eq;

fn compare<T: Eq>(a: T, b: T) -> Bool {
    a == b
}

fn main() {}
",
        "E0500",
    );
}

/// The qualified spelling was already rejected. It must stay rejected — and now for the same
/// reason as the imported spelling, rather than by accident of how the name was written.
#[test]
fn a_qualified_unrelated_eq_is_still_rejected() {
    rejects_at_typecheck(
        "dev171_fake_eq_qualified",
        "\
mod fake {
    pub trait Eq {
        fn unrelated(&self) -> Int32;
    }
}

fn compare<T: fake::Eq>(a: T, b: T) -> Bool {
    a == b
}

fn main() {}
",
        "E0500",
    );
}

/// The same defect shape for the other operator hooks, so the repair is not `Eq`-specific.
#[test]
fn unrelated_traits_named_ord_or_num_do_not_authorise_their_operators() {
    rejects_at_typecheck(
        "dev171_fake_ord",
        "\
mod fake {
    pub trait Ord {
        fn unrelated(&self) -> Int32;
    }
}

use fake::Ord;

fn smaller<T: Ord>(a: T, b: T) -> Bool {
    a < b
}

fn main() {}
",
        "E0500",
    );
    rejects_at_typecheck(
        "dev171_fake_num",
        "\
mod fake {
    pub trait Num {
        fn unrelated(&self) -> Int32;
    }
}

use fake::Num;

fn total<T: Num>(a: T, b: T) -> T {
    a + b
}

fn main() {}
",
        "E0500",
    );
}

/// Genuine bounds keep working — including a user type with its own `impl Eq`, which is the case a
/// too-eager identity check would break.
#[test]
fn genuine_operator_bounds_are_unaffected() {
    agree_completing_with_stdout(
        "dev171_genuine_bounds",
        "\
struct P {
    v: Int32,
}

impl Eq for P {
    fn eq(&self, other: &P) -> Bool {
        self.v == other.v
    }
}

fn same<T: Eq>(a: T, b: T) -> Bool {
    a == b
}

fn less<T: Ord>(a: T, b: T) -> Bool {
    a < b
}

fn total<T: Num>(a: T, b: T) -> T {
    a + b
}

fn both<T: Eq + Ord>(a: T, b: T) -> Bool {
    a == b || a < b
}

fn main() {
    println(same(1, 1));
    println(less(1, 2));
    println(total(2, 3));
    println(both(1, 2));
    println(same(P { v: 1 }, P { v: 2 }));
}
",
        "true\ntrue\n5\ntrue\nfalse\n",
    );
}
