//! WP-C6.1f-b2 — expected-type reference weakening (`&mut T` → `&T`).
//!
//! 03-Type-System "Reference Coercions" makes `&mut T -> &T` normative, and a function parameter,
//! annotated `let`, assignment destination and return position are all **expected-type
//! boundaries**. TYPE-METHOD-002 excludes argument-position auto-borrow, auto-dereference and
//! *user-defined* coercion — not this fixed built-in set. (That distinction is the CD-091
//! correction: an earlier reading treated the exclusion as covering all argument-position
//! conversion, which would have contradicted the frozen coercion rules.)
//!
//! Two defects had to be fixed together, because either alone leaves the boundary unusable:
//!   * **borrowck** consumed a `&mut` argument, so `f(m); f(m);` was E0100; it now **re-borrows**.
//!   * **lowering** never emitted the conversion, so the MIR verifier rejected the call; it now
//!     re-borrows at the expected mutability — including `&mut` → `&mut`, where a plain move would
//!     fail V-MOVE-1 on a second use.
//!
//! Each re-borrow is a *temporary* borrow ending with its statement (03 "References and
//! Lifetimes" rule 4: "`f(&x); g(&mut x);` is legal"), so no borrow duration changed. The
//! `c61f_reference_boundary.rs` negative corpus still passes unaltered.

mod support;

/// Delegates to the shared comparator (R-02). Was a private helper that ran three engines and
/// asserted `status == 0` on each separately -- which let three engines each exit 0 while printing
/// three different things.
fn agree(tag: &str, src: &str) {
    support::differential::agree_completing_available_engines(tag, src);
}

const P: &str = "struct P { v: Int32 }\n\
                 impl P { fn get(&self) -> Int32 { self.v } \
                 fn bump(&mut self) { self.v = self.v + 1; } }\n";

#[test]
fn c61f_b2_mut_weakens_to_shared_at_a_function_argument() {
    agree(
        "fn_argument",
        &format!(
            "{P}fn f(r: &P) -> Int32 {{ r.get() }}\n\
                  fn g(m: &mut P) -> Int32 {{ f(m) }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; assert_eq(g(&mut p), 3); }}"
        ),
    );
}

#[test]
fn c61f_b2_a_weakened_argument_is_reborrowed_not_moved() {
    // Was E0100 "use of moved value" at the second `f(m)`.
    agree(
        "fn_argument_twice",
        &format!(
            "{P}fn f(r: &P) -> Int32 {{ r.get() }}\n\
                  fn g(m: &mut P) -> Int32 {{ f(m) + f(m) }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; assert_eq(g(&mut p), 6); }}"
        ),
    );
}

#[test]
fn c61f_b2_mut_to_mut_argument_is_also_reborrowed() {
    // No weakening here — the types already match — but passing it must still not MOVE the
    // reference, or the second call fails (E0100 in the checker, V-MOVE-1 in MIR).
    agree(
        "mut_to_mut_twice",
        &format!(
            "{P}fn f(r: &mut P) {{ r.bump(); }}\n\
                  fn g(m: &mut P) {{ f(m); f(m); }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; g(&mut p); assert_eq(p.v, 5); }}"
        ),
    );
}

#[test]
fn c61f_b2_weakening_applies_to_a_fully_qualified_trait_call_receiver() {
    agree(
        "fq_trait_arg",
        "trait S { fn a(&self) -> Int32; }\nstruct Q { n: Int32 }\n\
         impl S for Q { fn a(&self) -> Int32 { self.n } }\n\
         fn g(m: &mut Q) -> Int32 { S::a(m) }\n\
         fn main() { let mut q = Q { n: 4 }; assert_eq(g(&mut q), 4); }",
    );
}

#[test]
fn c61f_b2_shared_arguments_and_borrows_still_work() {
    agree(
        "shared_unaffected",
        &format!(
            "{P}fn f(r: &P) -> Int32 {{ r.get() }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; assert_eq(f(&p) + f(&p), 6); }}"
        ),
    );
}

// ------------------------------------------- generic callees (b2 completion) --
//
// A generic callee's `fn_types` entry still mentions the callee's OWN parameters (`Ty::Param("T")`),
// which the CALLER's substitution cannot ground — so the expected type was unresolvable and no
// weakening was applied, leaving these to fail MIR verification. The call's concrete type arguments
// are already computed for the instance and are in the callee's generic declaration order, so they
// are exactly the substitution needed (`mir::lower::callee_param_types`).
//
// Resolving against the caller's map would have been worse than failing: inside a generic body with
// a same-named parameter it would silently pick up the WRONG type rather than decline.

const SH: &str = "trait Sh { fn a(&self) -> Int32; }\nstruct S { n: Int32 }\n\
                  impl Sh for S { fn a(&self) -> Int32 { self.n } }\n";

#[test]
fn c61f_b2_generic_callee_gets_argument_weakening() {
    agree(
        "generic_callee",
        &format!(
            "{SH}fn f<T: Sh>(r: &T) -> Int32 {{ r.a() }}\n\
                  fn g(m: &mut S) -> Int32 {{ f(m) }}\n\
                  fn main() {{ let mut s = S {{ n: 4 }}; assert_eq(g(&mut s), 4); }}"
        ),
    );
}

#[test]
fn c61f_b2_a_weakened_generic_argument_is_reborrowed_not_moved() {
    agree(
        "generic_callee_twice",
        &format!(
            "{SH}fn f<T: Sh>(r: &T) -> Int32 {{ r.a() }}\n\
                  fn g(m: &mut S) -> Int32 {{ f(m) + f(m) }}\n\
                  fn main() {{ let mut s = S {{ n: 4 }}; assert_eq(g(&mut s), 8); }}"
        ),
    );
}

#[test]
fn c61f_b2_generic_callee_with_a_later_non_reference_parameter() {
    // The substitution must line up positionally with the callee's declaration order.
    agree(
        "generic_two_params",
        &format!(
            "{SH}fn f<T: Sh>(r: &T, k: Int32) -> Int32 {{ r.a() + k }}\n\
                  fn g(m: &mut S) -> Int32 {{ f(m, 1) }}\n\
                  fn main() {{ let mut s = S {{ n: 4 }}; assert_eq(g(&mut s), 5); }}"
        ),
    );
}

#[test]
fn c61f_b2_generic_callee_shared_argument_is_unaffected() {
    agree(
        "generic_shared",
        &format!(
            "{SH}fn f<T: Sh>(r: &T) -> Int32 {{ r.a() }}\n\
                  fn g(r: &S) -> Int32 {{ f(r) }}\n\
                  fn main() {{ let s = S {{ n: 4 }}; assert_eq(g(&s), 4); }}"
        ),
    );
}
