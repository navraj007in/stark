//! **DEV-BOUND-TRAIT-IDENTITY** — a generic bound denotes the trait the RESOLVER selected, not the
//! first trait in the program that happens to be spelled the same way.
//!
//! DEV-DISPLAY-DISPATCH unified method candidate *collection* across user and compiler-known
//! traits. It left the step before that wrong in two passes: `typecheck`'s `resolve_bound_trait`
//! and `borrowck`'s `bound_method_receiver` each took the bound's source text and scanned every
//! HIR item for a trait declared with that name. Three failures followed, and all three are
//! reproduced by the cases below:
//!
//! * **A qualified bound matched nothing.** `T: traits::Render` compared `"traits::Render"`
//!   against the declaration's name `"Render"`, so the bound contributed no methods at all and
//!   `value.render()` was rejected with "requires the bound 'T: Render'" — on a function whose
//!   signature already wrote that bound.
//! * **An unrelated trait captured the name.** A `mod unrelated { pub trait Display { .. } }`
//!   anywhere in the program took over every `T: Display` bound, because a user trait of that
//!   spelling was found and preferred, so `x.fmt()` no longer reached the Core trait the resolver
//!   had actually selected.
//! * **Declaration order decided ownership.** With two same-named traits, one `&self` and one
//!   `self`, the borrow checker returned whichever appeared first in HIR item order. The pair
//!   `declaration_order_does_not_decide_ownership` is the same program twice with the two trait
//!   declarations swapped; before this work one order compiled and the other failed E0100.
//!
//! `hir::resolved_bound_trait` is now the single answer, read from `TraitRef::res` by both passes.
//! The invariant these tests defend: **the trait a method is selected from and the trait its
//! receiver form is read from are the same trait**, and neither depends on spelling or order.

mod support;

use starkc::hir::{BoundTrait, Res};
use support::differential::{agree_completing_with_stdout, front_end, rejects_at_typecheck};

/// Two traits with the same local name in different modules, plus a type implementing both.
/// Every cross-module identity case builds on this.
const TWO_RENDERS: &str = "\
mod left {
    pub trait Render {
        fn tag(&self) -> String;
    }
}

mod right {
    pub trait Render {
        fn tag(&self) -> String;
    }
}

struct Item {
    n: Int32,
}

impl left::Render for Item {
    fn tag(&self) -> String {
        let mut out = String::new();
        out.push_str(\"L\");
        out
    }
}

impl right::Render for Item {
    fn tag(&self) -> String {
        let mut out = String::new();
        out.push_str(\"R\");
        out
    }
}
";

// ------------------------------------------------------------------------ qualified bound paths --

/// A bound written through a module path resolves and dispatches. This is the case that could not
/// work at all under a spelling comparison: no trait is *declared* with the name
/// `"traits::Render"`.
#[test]
fn a_qualified_trait_bound_dispatches() {
    agree_completing_with_stdout(
        "bti_qualified",
        "\
mod traits {
    pub trait Render {
        fn render(&self) -> String;
    }
}

struct Item {
    value: Int32,
}

impl traits::Render for Item {
    fn render(&self) -> String {
        self.value.fmt()
    }
}

fn show<T: traits::Render>(value: &T) -> String {
    value.render()
}

fn main() {
    let item = Item { value: 7 };
    println(show(&item).as_str());
}
",
        "7\n",
    );
}

/// Nested forwarding through qualified bounds, so monomorphisation carries the identity too.
#[test]
fn a_qualified_bound_forwards_through_nested_generics() {
    agree_completing_with_stdout(
        "bti_qualified_nested",
        "\
mod traits {
    pub trait Render {
        fn render(&self) -> String;
    }
}

struct Item {
    value: Int32,
}

impl traits::Render for Item {
    fn render(&self) -> String {
        self.value.fmt()
    }
}

fn outer<T: traits::Render>(value: &T) -> String {
    inner(value)
}

fn inner<U: traits::Render>(value: &U) -> String {
    value.render()
}

fn main() {
    let item = Item { value: 11 };
    println(outer(&item).as_str());
}
",
        "11\n",
    );
}

/// A qualified bound written on the IMPL head (WP-C6.2b-F5), reached from a method body.
#[test]
fn a_qualified_impl_head_bound_dispatches() {
    agree_completing_with_stdout(
        "bti_qualified_impl_head",
        "\
mod traits {
    pub trait Render {
        fn render(&self) -> String;
    }
}

struct Item {
    value: Int32,
}

impl traits::Render for Item {
    fn render(&self) -> String {
        self.value.fmt()
    }
}

struct Wrapper<T> {
    inner: T,
}

impl<T: traits::Render> Wrapper<T> {
    fn show(&self) -> String {
        self.inner.render()
    }
}

fn main() {
    let w = Wrapper { inner: Item { value: 3 } };
    println(w.show().as_str());
}
",
        "3\n",
    );
}

// ------------------------------------------------------------- same local name, distinct traits --

/// `left::Render` and `right::Render` declare a method with the SAME name. Each bound must reach
/// its own trait's implementation.
#[test]
fn same_local_name_in_two_modules_stays_distinct() {
    agree_completing_with_stdout(
        "bti_two_modules",
        &format!(
            "{TWO_RENDERS}
fn use_left<T: left::Render>(value: &T) -> String {{
    value.tag()
}}

fn use_right<T: right::Render>(value: &T) -> String {{
    value.tag()
}}

fn main() {{
    let item = Item {{ n: 1 }};
    println(use_left(&item).as_str());
    println(use_right(&item).as_str());
}}
"
        ),
        "L\nR\n",
    );
}

/// An unrelated user trait spelled `Display` must not capture a `T: Display` bound. The resolver
/// selected the Core trait; a trait declared in some other module is not it.
#[test]
fn an_unrelated_same_named_trait_does_not_capture_a_core_bound() {
    agree_completing_with_stdout(
        "bti_core_not_captured",
        "\
mod unrelated {
    pub trait Display {
        fn other(&self) -> String;
    }
}

fn show<T: Display>(value: &T) -> String {
    value.fmt()
}

fn main() {
    let n: Int32 = 42;
    println(show(&n).as_str());
}
",
        "42\n",
    );
}

/// The reverse direction: a user trait that IS the resolved one still wins over the Core trait of
/// the same spelling, because `res` names it.
#[test]
fn an_imported_same_named_user_trait_is_the_resolved_one() {
    agree_completing_with_stdout(
        "bti_user_display_wins",
        "\
mod shadow {
    pub trait Display {
        fn fmt(&self) -> String;
    }
}

use shadow::Display;

struct Item {
    n: Int32,
}

impl shadow::Display for Item {
    fn fmt(&self) -> String {
        let mut out = String::new();
        out.push_str(\"shadowed\");
        out
    }
}

fn show<T: Display>(value: &T) -> String {
    value.fmt()
}

fn main() {
    let item = Item { n: 1 };
    println(show(&item).as_str());
}
",
        "shadowed\n",
    );
}

// ------------------------------------------------------------------------- receiver identity --

/// Two same-named traits with DIFFERENT receiver forms. The bound names the `&self` one, so the
/// receiver is borrowed and the value survives two calls.
///
/// Under the spelling lookup this depended on which trait the borrow checker found first.
#[test]
fn receiver_form_follows_the_resolved_trait_borrowed() {
    agree_completing_with_stdout(
        "bti_receiver_borrowed",
        "\
mod consuming {
    pub trait Action {
        fn act(self) -> Int32;
    }
}

mod borrowed {
    pub trait Action {
        fn act(&self) -> Int32;
    }
}

struct Cell {
    n: Int32,
}

impl borrowed::Action for Cell {
    fn act(&self) -> Int32 {
        self.n
    }
}

fn twice<T: borrowed::Action>(x: T) -> Int32 {
    let first = x.act();
    let second = x.act();
    first + second
}

fn main() {
    let c = Cell { n: 4 };
    println(twice(c));
}
",
        "8\n",
    );
}

/// **The order pair.** These two programs differ only in the order the two `Observe` declarations
/// are written. Both must compile: the bound names the top-level `&self` trait either way.
#[test]
fn declaration_order_does_not_decide_ownership() {
    let body = "\
fn twice<T: Observe>(x: T) -> Int32 {
    let first = x.observe();
    let second = x.observe();
    first + second
}

struct Cell {
    n: Int32,
}

impl Observe for Cell {
    fn observe(&self) -> Int32 {
        self.n
    }
}

fn main() {
    let c = Cell { n: 5 };
    println(twice(c));
}
";
    let borrowed_trait = "\
trait Observe {
    fn observe(&self) -> Int32;
}
";
    let consuming_trait = "\
mod other {
    pub trait Observe {
        fn observe(self) -> Int32;
    }
}
";
    agree_completing_with_stdout(
        "bti_order_borrowed_first",
        &format!("{borrowed_trait}\n{consuming_trait}\n{body}"),
        "10\n",
    );
    agree_completing_with_stdout(
        "bti_order_consuming_first",
        &format!("{consuming_trait}\n{borrowed_trait}\n{body}"),
        "10\n",
    );
}

/// A `&mut self` trait method through a qualified bound: the call requires a mutable receiver and
/// does not consume the value, so a second call and a later read both succeed.
#[test]
fn a_mutable_receiver_follows_the_resolved_trait() {
    agree_completing_with_stdout(
        "bti_receiver_mut",
        "\
mod m {
    pub trait Update {
        fn update(&mut self);
        fn peek(&self) -> Int32;
    }
}

struct Counter {
    n: Int32,
}

impl m::Update for Counter {
    fn update(&mut self) {
        self.n = self.n + 1;
    }

    fn peek(&self) -> Int32 {
        self.n
    }
}

fn bump_twice<T: m::Update>(x: &mut T) -> Int32 {
    x.update();
    x.update();
    x.peek()
}

fn main() {
    let mut c = Counter { n: 0 };
    let bumped = bump_twice(&mut c);
    println(bumped);
    println(c.peek());
}
",
        "2\n2\n",
    );
}

// ------------------------------------------------------- DEV-DISPLAY-DISPATCH properties kept --

/// Bound order still does not select a method, now with a qualified user bound alongside a Core
/// one. Both orders must produce the same output.
#[test]
fn bound_order_still_does_not_select_a_method() {
    for (tag, bounds) in [
        ("bti_order_core_first", "Display + traits::Named"),
        ("bti_order_user_first", "traits::Named + Display"),
    ] {
        agree_completing_with_stdout(
            tag,
            &format!(
                "\
mod traits {{
    pub trait Named {{
        fn name(&self) -> String;
    }}
}}

struct Item {{
    n: Int32,
}}

impl Display for Item {{
    fn fmt(&self) -> String {{
        self.n.fmt()
    }}
}}

impl traits::Named for Item {{
    fn name(&self) -> String {{
        let mut out = String::new();
        out.push_str(\"item\");
        out
    }}
}}

fn describe<T: {bounds}>(value: &T) -> String {{
    let rendered = value.fmt();
    let name = value.name();

    let mut out = String::new();
    out.push_str(name.as_str());
    out.push_str(\"=\");
    out.push_str(rendered.as_str());
    out
}}

fn main() {{
    let item = Item {{ n: 9 }};
    println(describe(&item).as_str());
}}
"
            ),
            "item=9\n",
        );
    }
}

// ------------------------------------------------------------------------------ negative cases --

/// A bound on `a::Render` must not make `b::Render`'s method callable, even though both traits are
/// spelled `Render`.
#[test]
fn a_bound_does_not_admit_a_same_named_traits_method() {
    let messages = rejects_at_typecheck(
        "bti_wrong_trait_method",
        "\
mod a {
    pub trait Render {
        fn a(&self) -> String;
    }
}

mod b {
    pub trait Render {
        fn b(&self) -> String;
    }
}

fn wrong<T: a::Render>(x: &T) -> String {
    x.b()
}

fn main() {}
",
        "E0302",
    );
    assert!(
        messages.iter().any(|m| m.contains("'b'")),
        "expected the missing method to be blamed, got {messages:?}"
    );
}

/// Two same-named traits, only ONE of them bound. That must not read as an ambiguity — the other
/// declaration is not in the candidate set at all.
#[test]
fn a_same_named_trait_elsewhere_is_not_an_ambiguity() {
    agree_completing_with_stdout(
        "bti_no_false_ambiguity",
        &format!(
            "{TWO_RENDERS}
fn only_left<T: left::Render>(value: &T) -> String {{
    value.tag()
}}

fn main() {{
    let item = Item {{ n: 1 }};
    println(only_left(&item).as_str());
}}
"
        ),
        "L\n",
    );
}

/// A bound naming the CONSUMING trait moves the receiver, so a second call is a use-after-move.
/// The borrowed trait of the same name must not rescue it.
#[test]
fn a_consuming_receiver_is_still_a_move() {
    rejects_at_typecheck(
        "bti_consuming_move",
        "\
mod borrowed {
    pub trait Action {
        fn act(&self) -> Int32;
    }
}

mod consuming {
    pub trait Action {
        fn act(self) -> Int32;
    }
}

fn twice<T: consuming::Action>(x: T) -> Int32 {
    let first = x.act();
    let second = x.act();
    first + second
}

fn main() {}
",
        "E0100",
    );
}

/// A `TraitRef` resolving to a non-trait item contributes no methods, and the call is rejected
/// rather than being given some trait picked by spelling.
///
/// The bound names a STRUCT. `hir::resolved_bound_trait` returns `None` for it — the exhaustive
/// `ItemKind` match has no arm that would treat a struct as a trait — so the parameter has no
/// bound-provided methods and the call fails as "not found". A spelling fallback would have
/// searched for a *trait* named `NotATrait`, found none, and then reached for the Core table.
#[test]
fn a_bound_naming_a_non_trait_item_contributes_nothing() {
    let messages = rejects_at_typecheck(
        "bti_non_trait_bound",
        "\
struct NotATrait {
    n: Int32,
}

fn bad<T: NotATrait>(x: &T) -> String {
    x.render()
}

fn main() {}
",
        "E0302",
    );
    assert!(
        messages.iter().any(|m| m.contains("not found")),
        "expected a not-found rejection, got {messages:?}"
    );
    assert!(
        !messages.iter().any(|m| m.contains("requires the bound")),
        "a non-trait bound must not be reported as a missing trait bound: {messages:?}"
    );
}

// ------------------------------------------------------------------- the helper, tested directly --

/// The identity helper reads `TraitRef::res` and nothing else. Asserted against the resolved HIR
/// rather than only through program behaviour, so a future pass that reintroduces a spelling
/// lookup fails here too.
#[test]
fn resolved_bound_trait_reads_the_resolution() {
    let front = front_end(
        "bti_helper",
        "\
mod left {
    pub trait Render {
        fn tag(&self) -> String;
    }
}

mod right {
    pub trait Render {
        fn tag(&self) -> String;
    }
}

fn use_left<T: left::Render>(value: &T) -> String {
    value.tag()
}

fn use_right<T: right::Render>(value: &T) -> String {
    value.tag()
}

fn with_core<T: Display>(value: &T) -> String {
    value.fmt()
}

fn main() {}
",
    );

    let mut resolved: Vec<(String, BoundTrait)> = Vec::new();
    for item in &front.hir.items {
        let starkc::hir::ItemKind::Fn(def) = &item.kind else {
            continue;
        };
        let fn_name =
            front.file.src[def.sig.name.lo as usize..def.sig.name.hi as usize].to_string();
        for param in &def.sig.generics {
            for bound in &param.bounds {
                let identity = starkc::hir::resolved_bound_trait(&front.hir, bound)
                    .unwrap_or_else(|| panic!("{fn_name}: bound did not resolve to a trait"));
                resolved.push((fn_name.clone(), identity));
            }
        }
    }

    let left = resolved
        .iter()
        .find(|(f, _)| f == "use_left")
        .expect("use_left");
    let right = resolved
        .iter()
        .find(|(f, _)| f == "use_right")
        .expect("use_right");
    assert_ne!(
        left.1, right.1,
        "two same-named traits in different modules must be distinct identities"
    );
    assert!(
        matches!(left.1, BoundTrait::User(_)) && matches!(right.1, BoundTrait::User(_)),
        "both are user traits: {:?} / {:?}",
        left.1,
        right.1
    );

    let core = resolved
        .iter()
        .find(|(f, _)| f == "with_core")
        .expect("with_core");
    assert_eq!(
        core.1,
        BoundTrait::Core(starkc::hir::CoreTrait::Display),
        "a Core bound resolves to the Core identity"
    );

    // And each user identity is the item the RESOLVER named, not one found by scanning.
    for item in &front.hir.items {
        let starkc::hir::ItemKind::Fn(def) = &item.kind else {
            continue;
        };
        let fn_name =
            front.file.src[def.sig.name.lo as usize..def.sig.name.hi as usize].to_string();
        if fn_name != "use_left" {
            continue;
        }
        for param in &def.sig.generics {
            for bound in &param.bounds {
                let Res::Item(expected) = bound.res else {
                    panic!("use_left's bound should resolve to an item");
                };
                assert_eq!(
                    starkc::hir::resolved_bound_trait(&front.hir, bound),
                    Some(BoundTrait::User(expected)),
                    "the identity must be the resolver's own answer"
                );
            }
        }
    }
}
