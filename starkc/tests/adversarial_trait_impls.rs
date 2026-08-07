//! WP-C7.9 Packet B — a trait implementation must conform before any body is executable.
//!
//! **The defect.** A user-declared trait is an HIR item, so `impl Trait for T` could be compared
//! against its declaration, and was. A compiler-known `CoreTrait` has no declaration item — every
//! `impl Ord for T` writes its own signature — and **nothing checked it at all**. The consequence
//! was not a cosmetic one: `fn cmp(&self, other: &Self) -> Bool` passed type checking, reached the
//! HIR interpreter, and produced whatever each engine happened to do with a `Bool` where a
//! comparison result was owed. The frontend is the only place that can decide this, because it is
//! the only phase every engine sits behind.
//!
//! Each case here proves rejection **at type checking** — not at lowering, not at verification, and
//! not at execution. `rejects_at_typecheck` asserts the phase, so a fix that merely made one engine
//! fail more loudly would not pass.
//!
//! The positive cases at the end matter as much as the negatives: a conformance check that rejects
//! valid implementations is a worse defect than the one it replaced.

mod support;

use support::differential::rejects_at_typecheck;

/// A struct with `Eq`, used as the subject of most cases.
const POINT: &str = "struct P { x: Int32 }\n";

fn reject(name: &str, source: &str) -> Vec<String> {
    rejects_at_typecheck(&format!("{name}.stark"), source, "E0500")
}

fn accept(name: &str, source: &str) {
    // A full three-engine run is not what these prove; the point is that the front end ADMITS
    // them. Executing them is the business of the suites that own those semantics.
    let file = std::sync::Arc::new(starkc::source::SourceFile::new(
        format!("{name}.stark"),
        source.to_string(),
    ));
    let (ast, pd) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    assert!(pd.is_empty(), "{name}: parse: {pd:?}");
    let (hir, rd) = starkc::resolve::resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{name}: resolve: {rd:?}");
    let checked = starkc::typecheck::analyze(&hir);
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "{name}: expected acceptance, got {errors:?}"
    );
}

// ------------------------------------------------------------------- wrong return types --

/// The finding's own case. `Ord::cmp` returns `Ordering`; this returns `Bool`.
#[test]
fn ord_cmp_with_wrong_return_type_is_rejected() {
    let messages = reject(
        "ord_cmp_bool",
        &format!(
            "{POINT}impl Ord for P {{ fn cmp(&self, other: &P) -> Bool {{ self.x == other.x }} }}\n\
             fn main() {{ }}\n"
        ),
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("cmp") && m.contains("Ordering")),
        "the diagnostic must name the method and the expected type: {messages:?}"
    );
}

#[test]
fn eq_with_wrong_return_type_is_rejected() {
    reject(
        "eq_int",
        &format!(
            "{POINT}impl Eq for P {{ fn eq(&self, other: &P) -> Int32 {{ 1 }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

#[test]
fn display_fmt_with_wrong_return_type_is_rejected() {
    reject(
        "display_int",
        &format!(
            "{POINT}impl Display for P {{ fn fmt(&self) -> Int32 {{ self.x }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

#[test]
fn clone_with_wrong_return_type_is_rejected() {
    reject(
        "clone_int",
        &format!(
            "{POINT}impl Clone for P {{ fn clone(&self) -> Int32 {{ self.x }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

#[test]
fn default_with_wrong_return_type_is_rejected() {
    reject(
        "default_int",
        &format!(
            "{POINT}impl Default for P {{ fn default() -> Int32 {{ 0 }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

#[test]
fn hash_with_wrong_return_type_is_rejected() {
    reject(
        "hash_int32",
        &format!(
            "{POINT}impl Hash for P {{ fn hash(&self) -> Int32 {{ self.x }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

/// `Drop::drop` returns Unit. Declaring a return type is a contract violation even though the body
/// would type-check on its own.
#[test]
fn drop_with_a_return_type_is_rejected() {
    reject(
        "drop_returns",
        &format!(
            "{POINT}impl Drop for P {{ fn drop(&mut self) -> Int32 {{ 0 }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

/// `From::from` returns `Self`, not the source type.
#[test]
fn from_with_wrong_return_type_is_rejected() {
    reject(
        "from_wrong_ret",
        "struct C { v: Int32 }\nstruct F { v: Int32 }\n\
         impl From<C> for F { fn from(c: C) -> C { c } }\n\
         fn main() { }\n",
    );
}

/// `Iterator::next` returns `Option<Self::Item>` — here the associated type says `Int32` and the
/// method says `Option<Bool>`, which no engine could reconcile.
#[test]
fn iterator_next_disagreeing_with_its_item_type_is_rejected() {
    reject(
        "iter_item_mismatch",
        "struct G { n: Int32 }\n\
         impl Iterator for G { type Item = Int32; fn next(&mut self) -> Option<Bool> { None } }\n\
         fn main() { }\n",
    );
}

// ------------------------------------------------------------------------- receivers --

#[test]
fn wrong_receiver_mutability_is_rejected() {
    // `Drop::drop` takes `&mut self`.
    reject(
        "drop_shared_receiver",
        &format!("{POINT}impl Drop for P {{ fn drop(&self) {{ }} }}\nfn main() {{ }}\n"),
    );
    // `Eq::eq` takes `&self`.
    reject(
        "eq_mut_receiver",
        &format!(
            "{POINT}impl Eq for P {{ fn eq(&mut self, other: &P) -> Bool {{ true }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

#[test]
fn missing_receiver_is_rejected() {
    reject(
        "eq_no_receiver",
        &format!(
            "{POINT}impl Eq for P {{ fn eq(a: &P, other: &P) -> Bool {{ true }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

/// `Default::default` is an associated function; giving it a receiver makes it uncallable as the
/// trait declares it.
#[test]
fn extra_receiver_is_rejected() {
    reject(
        "default_receiver",
        &format!(
            "{POINT}impl Default for P {{ fn default(&self) -> P {{ P {{ x: 0 }} }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

// -------------------------------------------------------------------------- parameters --

#[test]
fn wrong_parameter_type_is_rejected() {
    reject(
        "eq_wrong_param",
        &format!(
            "{POINT}impl Eq for P {{ fn eq(&self, other: Int32) -> Bool {{ true }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

#[test]
fn wrong_parameter_count_is_rejected() {
    reject(
        "eq_two_params",
        &format!(
            "{POINT}impl Eq for P {{ fn eq(&self, a: &P, b: &P) -> Bool {{ true }} }}\n\
             fn main() {{ }}\n"
        ),
    );
    reject(
        "clone_extra_param",
        &format!(
            "{POINT}impl Clone for P {{ fn clone(&self, n: Int32) -> P {{ P {{ x: n }} }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

// --------------------------------------------------------------- item membership --

#[test]
fn missing_required_method_is_rejected() {
    reject(
        "eq_empty",
        &format!("{POINT}impl Eq for P {{ }}\nfn main() {{ }}\n"),
    );
    reject(
        "iterator_without_next",
        "struct G { n: Int32 }\n\
         impl Iterator for G { type Item = Int32; }\nfn main() { }\n",
    );
}

#[test]
fn extra_trait_item_is_rejected() {
    // A marker trait declares nothing at all.
    reject(
        "copy_with_method",
        &format!(
            "{POINT}impl Copy for P {{ fn extra(&self) -> Int32 {{ self.x }} }}\n\
             fn main() {{ }}\n"
        ),
    );
    reject(
        "eq_with_extra",
        &format!(
            "{POINT}impl Eq for P {{ fn eq(&self, o: &P) -> Bool {{ true }} \
             fn other(&self) -> Int32 {{ 1 }} }}\nfn main() {{ }}\n"
        ),
    );
}

#[test]
fn duplicate_method_is_rejected() {
    reject(
        "eq_twice",
        &format!(
            "{POINT}impl Eq for P {{ fn eq(&self, o: &P) -> Bool {{ true }} \
             fn eq(&self, o: &P) -> Bool {{ false }} }}\nfn main() {{ }}\n"
        ),
    );
}

#[test]
fn missing_or_wrong_associated_type_is_rejected() {
    reject(
        "iterator_no_item",
        "struct G { n: Int32 }\n\
         impl Iterator for G { fn next(&mut self) -> Option<Int32> { None } }\nfn main() { }\n",
    );
    reject(
        "iterator_extra_assoc",
        "struct G { n: Int32 }\n\
         impl Iterator for G { type Item = Int32; type Other = Bool; \
         fn next(&mut self) -> Option<Int32> { None } }\nfn main() { }\n",
    );
}

#[test]
fn method_generics_on_a_core_trait_method_are_rejected() {
    reject(
        "clone_generic",
        &format!(
            "{POINT}impl Clone for P {{ fn clone<T>(&self) -> P {{ P {{ x: self.x }} }} }}\n\
             fn main() {{ }}\n"
        ),
    );
}

// ------------------------------------------------------- user-declared traits, same rules --

/// The user-trait path already compared signatures; these pin that it still does, and that the two
/// paths agree on what a violation is.
#[test]
fn user_trait_violations_are_still_rejected() {
    reject(
        "user_wrong_ret",
        "trait Speak { fn speak(&self) -> Int32; }\nstruct D { }\n\
         impl Speak for D { fn speak(&self) -> Bool { true } }\nfn main() { }\n",
    );
    reject(
        "user_missing",
        "trait Speak { fn speak(&self) -> Int32; }\nstruct D { }\n\
         impl Speak for D { }\nfn main() { }\n",
    );
    reject(
        "user_extra",
        "trait Speak { fn speak(&self) -> Int32; }\nstruct D { }\n\
         impl Speak for D { fn speak(&self) -> Int32 { 1 } fn other(&self) -> Int32 { 2 } }\n\
         fn main() { }\n",
    );
}

/// Duplicates in a user-trait impl block. The membership checks there are set differences, and a
/// set cannot see the same name twice — so this was accepted before Packet B.
#[test]
fn duplicate_method_in_a_user_trait_impl_is_rejected() {
    reject(
        "user_duplicate",
        "trait Speak { fn speak(&self) -> Int32; }\nstruct D { }\n\
         impl Speak for D { fn speak(&self) -> Int32 { 1 } fn speak(&self) -> Int32 { 2 } }\n\
         fn main() { }\n",
    );
}

// ---------------------------------------------------------------------- valid impls --

/// Every modelled Core trait, implemented correctly, must still be accepted. A conformance check
/// that over-rejects is a worse defect than the one it replaced.
#[test]
fn conformant_core_trait_impls_are_accepted() {
    accept(
        "valid_eq_ord",
        "struct P { x: Int32 }\n\
         impl Eq for P { fn eq(&self, other: &P) -> Bool { self.x == other.x } }\n\
         impl Ord for P { fn cmp(&self, other: &P) -> Ordering { \
         if self.x < other.x { Ordering::Less } else { Ordering::Greater } } }\n\
         fn main() { }\n",
    );
    accept(
        "valid_clone_display_default_hash",
        "struct P { x: Int32 }\n\
         impl Clone for P { fn clone(&self) -> P { P { x: self.x } } }\n\
         impl Display for P { fn fmt(&self) -> String { String::from(\"P\") } }\n\
         impl Default for P { fn default() -> P { P { x: 0 } } }\n\
         impl Hash for P { fn hash(&self) -> UInt64 { 7u64 } }\n\
         fn main() { }\n",
    );
    accept(
        "valid_drop",
        "struct P { x: Int32 }\nimpl Drop for P { fn drop(&mut self) { } }\nfn main() { }\n",
    );
    accept(
        "valid_iterator",
        "struct G { n: Int32 }\n\
         impl Iterator for G { type Item = Int32; \
         fn next(&mut self) -> Option<Int32> { None } }\nfn main() { }\n",
    );
    accept(
        "valid_from",
        "struct C { v: Int32 }\nstruct F { v: Int32 }\n\
         impl From<C> for F { fn from(c: C) -> F { F { v: c.v } } }\nfn main() { }\n",
    );
}

/// `Self` and the written self type are interchangeable spellings, in both directions, and an
/// impl may use either — the same normalisation WP-C6.2b-F6 established for user traits.
#[test]
fn self_and_the_written_type_are_interchangeable() {
    accept(
        "valid_self_spelling",
        "struct P { x: Int32 }\n\
         impl Eq for P { fn eq(&self, other: &Self) -> Bool { self.x == other.x } }\n\
         impl Clone for P { fn clone(&self) -> Self { P { x: self.x } } }\n\
         fn main() { }\n",
    );
}

/// An `Iterator` impl may write its item type either way: `Option<Self::Item>` or the concrete
/// type the associated declaration binds.
#[test]
fn iterator_item_may_be_written_either_way() {
    accept(
        "valid_iter_assoc_spelling",
        "struct G { n: Int32 }\n\
         impl Iterator for G { type Item = Int32; \
         fn next(&mut self) -> Option<Self::Item> { None } }\nfn main() { }\n",
    );
}
