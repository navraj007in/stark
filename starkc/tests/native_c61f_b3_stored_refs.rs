//! WP-C6.1f-b3 — stored references: a borrow bound to a user local, flowing across basic blocks.
//!
//! The C5 "ephemeral reference lane" admitted references only in **same-block compiler
//! temporaries**. The C6.1f-a matrix showed why that was stricter than necessary: all fifteen
//! backend-refused rows already verified under the MIR verifier *and ran correctly under the MIR
//! interpreter* — the gap was generated-Rust emission, not reference representation.
//!
//! Probing with the lane disabled identified the actual blocker, and it was **not** the
//! `ValueSlot`/borrow-checker conflict the matrix flagged as the design question: a same-block
//! borrow bound to a user local already built and ran, **including for a `Drop`-bearing owner**.
//! What failed was `E0381 "used binding isn't initialized"` — rustc's *definite-assignment*
//! analysis, not its borrow checker. A reference local is assigned in one arm of the generated
//! block-dispatch `loop { match … }` and read in another, which rustc cannot follow.
//!
//! Fix: a reference bound to a **user** local is declared `Option<&T> = None`, definitely
//! initialised at its declaration. Compiler temporaries keep the bare form — they are same-block by
//! construction, so rustc's definite-assignment check still guards them exactly as before, and
//! every previously working reference path is untouched.
//!
//! Still refused (unchanged): **returning** a reference, and storing one in an aggregate. The
//! `c61f_reference_boundary.rs` negative corpus — including the no-NLL case — passes unaltered.

mod support;

/// Delegates to the shared comparator (R-02). Was a private helper that ran three engines and
/// asserted `status == 0` on each separately -- which let three engines each exit 0 while printing
/// three different things.
fn agree(tag: &str, src: &str) {
    support::differential::agree_completing_available_engines(tag, src);
}

/// HIR, MIR and native must all complete with exit 0.
const P: &str = "struct P { v: Int32 }\n\
                 impl P { fn get(&self) -> Int32 { self.v } \
                 fn bump(&mut self) { self.v = self.v + 1; } }\n";

#[test]
fn c61f_b3_reference_bound_to_a_user_local() {
    agree(
        "shared_local",
        &format!("{P}fn main() {{ let p = P {{ v: 3 }}; let r = &p; assert_eq(r.get(), 3); }}"),
    );
    agree(
        "shared_local_field",
        &format!("{P}fn main() {{ let p = P {{ v: 3 }}; let r = &p; assert_eq(r.v, 3); }}"),
    );
    agree(
        "primitive",
        "fn main() { let x: Int32 = 5; let r = &x; assert_eq(*r, 5); }",
    );
    agree(
        "two_shared_borrows",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let a = &p; let b = &p; \
                  assert_eq(a.get() + b.get(), 6); }}"
        ),
    );
}

#[test]
fn c61f_b3_reference_flows_across_basic_blocks() {
    // The E0381 case: assigned in one dispatch-loop arm, read in another.
    agree(
        "across_if",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let r = &p; \
                  let n = if r.get() > 1 {{ r.get() }} else {{ 0 }}; assert_eq(n, 3); }}"
        ),
    );
    agree(
        "into_loop",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 2 }}; let r = &p; let mut s: Int32 = 0; \
                  let mut i: Int32 = 0; while i < 3 {{ s = s + r.get(); i = i + 1; }} \
                  assert_eq(s, 6); }}"
        ),
    );
}

#[test]
fn c61f_b3_mutable_reference_in_a_user_local() {
    // `Option<&mut T>` is not `Copy`, so access re-borrows out of the Option rather than
    // moving out of it — moving would make the second use fail to compile.
    agree(
        "mut_local",
        &format!(
            "{P}fn main() {{ let mut p = P {{ v: 3 }}; let r = &mut p; r.bump(); \
                  assert_eq(r.get(), 4); }}"
        ),
    );
}

#[test]
fn c61f_b3_references_into_fields_and_elements() {
    // A borrow of a `Copy` field must be a place expression: read mode may substitute a raw
    // projection COPY helper, and `&<copy>` would reference a temporary, not the field.
    agree(
        "struct_field",
        "struct P { v: Int32 }\nfn main() { let p = P { v: 3 }; let r = &p.v; assert_eq(*r, 3); }",
    );
    agree(
        "nested_field",
        "struct I { v: Int32 }\nstruct O { i: I }\n\
         fn main() { let o = O { i: I { v: 7 } }; let r = &o.i; assert_eq(r.v, 7); }",
    );
    agree(
        "array_element",
        "fn main() { let a: [Int32; 3] = [1, 2, 3]; let r = &a[1]; assert_eq(*r, 2); }",
    );
}

#[test]
fn c61f_b3_borrowing_a_drop_bearing_owner() {
    // The matrix flagged slot/drop-flag interaction as the design risk; it is not one.
    agree(
        "drop_owner",
        "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) { } }\n\
         fn main() { let d = D { v: 1 }; let r = &d; assert_eq(r.v, 1); }",
    );
    // Borrow ends with its block, then the owner is moved.
    agree(
        "borrow_then_move",
        &format!(
            "{P}fn take(p: P) -> Int32 {{ p.v }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; {{ let r = &p; assert_eq(r.get(), 3); }} \
                  assert_eq(take(p), 3); }}"
        ),
    );
}

#[test]
fn c61f_b3_unblocks_the_b2_boundaries_that_were_waiting_on_the_lane() {
    // b2 emitted the `&mut T` -> `&T` weakening correctly, but binding the result to a user local
    // hit the lane. Both halves are needed for this to run.
    agree(
        "annotated_local_weakening",
        &format!(
            "{P}fn g(m: &mut P) -> Int32 {{ let r: &P = m; r.get() }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; assert_eq(g(&mut p), 3); }}"
        ),
    );
}
