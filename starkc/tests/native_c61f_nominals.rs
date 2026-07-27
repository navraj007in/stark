//! WP-C6.1f — borrow-carrying **nominals**: `Option<&T>`, and user generics instantiated at a
//! reference.
//!
//! A generated nominal is a *declared* Rust type, so unlike a tuple it cannot borrow implicitly:
//! a reference in one of its fields needs a lifetime parameter, or rustc reports `E0106`. Generated
//! nominals therefore carry one — `Name<'a>` in the declaration, `Name<'_>` at every use site.
//! The two spellings are not interchangeable: `'_` is illegal in a field type, which has no
//! enclosing binder to infer from, so declaration and use positions are rendered separately
//! (`emit_types::LifetimePosition`).
//!
//! **Nothing here is refused any more.** Two shapes once were — both the
//! `ValueSlot`-versus-Rust-borrow-region tension the C6.1f-a matrix flagged as this package's
//! central design question — and both fell to control-flow precision in the generated crate rather
//! than to any change in the borrow rules:
//!
//! 1. A **function returning** a borrow-carrying nominal, whose elided output lifetime kept the
//!    borrow live across the referent's own slot destruction — lifted by CD-112 (acyclic bodies
//!    emitted as nested labelled blocks instead of one `loop { match __bb }`).
//! 2. A **slot-backed** (non-`Copy`) borrow-carrying nominal, whose slot destruction needs `&mut`
//!    while the reference it stores still borrows its referent — lifted by CD-128, once CD-127
//!    extended structured emission to cyclic bodies. rustc had been treating the two as overlapping
//!    for the local's whole lexical region (`E0502`) only because the dispatch loop hid the real
//!    control flow; with it visible, MIR's own ordering (borrower dropped first) type-checks as
//!    written.
//!
//! So this file is entirely positive, and `refuse_borrow_carrying_nominals` no longer exists.

mod support;

/// Delegates to the shared comparator (R-02).
fn agree(tag: &str, src: &str) {
    support::differential::agree_completing_available_engines(tag, src);
}

/// Drive a program to a native binary, asserting HIR and MIR agree on the way.
// CD-128: the `refused_before_rustc` helper is GONE with the last refusal it checked. Every
// borrow-carrying shape this file covers now builds and runs, so the file is entirely positive.

const P: &str = "struct P { v: Int32 }\nimpl P { fn get(&self) -> Int32 { self.v } }\n";

#[test]
fn c61f_option_holding_a_reference() {
    agree(
        "some",
        "fn main() { let x: Int32 = 5; let o: Option<&Int32> = Some(&x); assert_eq(*o.unwrap(), 5); }",
    );
    agree(
        "none",
        "fn main() { let o: Option<&Int32> = None; assert(o.is_none()); }",
    );
}

#[test]
fn c61f_matching_on_an_option_holding_a_reference() {
    agree(
        "match",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let o: Option<&P> = Some(&p); \
                  match o {{ Some(r) => assert_eq(r.get(), 3), None => assert(false) }} }}"
        ),
    );
}

#[test]
fn c61f_nested_and_embedded_borrow_carrying_nominals() {
    agree(
        "nested",
        "fn main() { let x: Int32 = 5; let o: Option<Option<&Int32>> = Some(Some(&x)); \
         assert(o.is_some()); }",
    );
    agree(
        "in_tuple",
        "fn main() { let x: Int32 = 5; let t: (Option<&Int32>, Int32) = (Some(&x), 1); \
         assert(t.0.is_some()); }",
    );
}

#[test]
fn c61f_a_nominal_without_a_borrow_is_unaffected() {
    // The lifetime parameter appears only when the instance actually carries a borrow.
    agree(
        "plain_option",
        "fn main() { let o: Option<Int32> = Some(5); assert_eq(o.unwrap(), 5); }",
    );
}

// WP-C6.1g-a: a COPY borrow-carrying nominal in a local WORKS (structural Copy makes it
// non-slot-backed; it flows through the CD-095 aggregate path). CD-128: the MOVE case — dragged Move
// by an owned/Drop-bearing field alongside the generic argument that supplies the borrow — works too
// now that structured emission (CD-127) gives rustc real control flow.
#[test]
fn c61f_a_copy_borrow_carrying_nominal_local_now_works() {
    agree(
        "generic_struct_local",
        &format!(
            "{P}struct H<T> {{ r: T }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; let h: H<&P> = H {{ r: &p }}; \
                  assert_eq(h.r.get(), 3); }}"
        ),
    );
    // Across basic blocks.
    agree(
        "generic_struct_xblock",
        &format!(
            "{P}struct H<T> {{ r: T }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; let h: H<&P> = H {{ r: &p }}; \
                  let n = if h.r.get() > 1 {{ h.r.get() }} else {{ 0 }}; assert_eq(n, 3); }}"
        ),
    );
    // A user enum carrying a borrow, matched.
    agree(
        "user_enum_local",
        &format!(
            "{P}enum E<T> {{ A, B(T) }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; let e: E<&P> = E::B(&p); \
                  match e {{ E::A => assert(false), E::B(r) => assert_eq(r.get(), 3) }} }}"
        ),
    );
}

/// CD-128: a MOVE borrow-carrying nominal local now BUILDS AND RUNS. The generic argument supplies
/// the borrow and the `Drop`-bearing field makes the whole nominal Move, so it is slot-backed — the
/// shape whose `ValueSlot` destruction (`&mut` on the slot) used to collide with the borrow it
/// stores (E0502) and was refused pre-rustc. CD-127's structured control-flow emission removed the
/// imprecision that caused the collision: rustc now sees the real region and MIR's own ordering
/// (borrower dropped before referent) type-checks as written.
#[test]
fn c61f_a_move_borrow_carrying_nominal_local_now_works() {
    agree(
        "move_generic_struct",
        &format!(
            "{P}struct D {{ w: Int32 }}\nimpl Drop for D {{ fn drop(&mut self) {{}} }}\n\
                  struct H<T> {{ r: T, d: D }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; let h: H<&P> = H {{ r: &p, d: D {{ w: 0 }} }}; \
                  assert_eq(h.r.get(), 3); }}"
        ),
    );
}

#[test]
fn c61f_returning_a_borrow_carrying_nominal_builds_and_runs() {
    // WP-C6.1g-c: the return-refusal is LIFTED. An acyclic body is emitted as nested labelled
    // blocks (not one `loop { match __bb }`), so the borrow `wrap` returns and `main` consumes
    // across the `Option::unwrap` blocks is seen by rustc with its real once-through lifetime and
    // no longer collides with the referent's assignment (the former E0502/E0506).
    agree(
        "return_option_ref",
        &format!(
            "{P}fn wrap(r: &P) -> Option<&P> {{ Some(r) }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; let o = wrap(&p); \
                  assert_eq(o.unwrap().get(), 3); }}"
        ),
    );
}

/// The same borrow returned and consumed inline in one expression (`wrap(&p).unwrap().get()`).
#[test]
fn c61f_returning_a_borrow_carrying_nominal_consumed_inline() {
    agree(
        "return_option_ref_inline",
        &format!(
            "{P}fn wrap(r: &P) -> Option<&P> {{ Some(r) }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; assert_eq(wrap(&p).unwrap().get(), 3); }}"
        ),
    );
}
