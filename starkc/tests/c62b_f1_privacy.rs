//! WP-C6.2b-F1 — privacy enforcement for callable and member resolution.
//!
//! MOD-VIS-001 (07-Modules-and-Packages) and TYPE-METHOD-001 step 5 (03-Type-System): a private
//! item is usable only in its defining module (and descendants — STARK is not Rust's
//! descendant-inherits rule; private is exact-module). Module-level items are already enforced by
//! `resolve::item_is_visible_from` at path resolution. F1: impl members (methods,
//! associated functions) and struct fields resolve in `typecheck`, which applied no visibility
//! check — so a private one is reachable cross-module, an accepted-invalid program.
//!
//! Every case here is a FRONT-END decision (invalid programs stop before lowering), so the harness
//! runs parse -> resolve -> typecheck and inspects diagnostics only.

use starkc::diag::Severity;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn errors(tag: &str, src: &str) -> Vec<starkc::diag::Diagnostic> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    let mut ds: Vec<_> = pd
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    let (hir, rd) = resolve(&ast, file.clone());
    ds.extend(rd.into_iter().filter(|d| d.severity == Severity::Error));
    let checked = typecheck::analyze(&hir, file);
    ds.extend(
        checked
            .diagnostics
            .into_iter()
            .filter(|d| d.severity == Severity::Error),
    );
    ds
}
fn accepted(tag: &str, src: &str) {
    let ds = errors(tag, src);
    assert!(ds.is_empty(), "{tag}: expected acceptance, got {ds:?}");
}
fn rejected(tag: &str, src: &str) {
    let ds = errors(tag, src);
    assert!(
        !ds.is_empty(),
        "{tag}: expected a privacy rejection, but the program was ACCEPTED (F1 accepted-invalid)"
    );
}

// ------------------------------------------------------------------ positive --

#[test]
fn f1_same_module_private_function_call_is_accepted() {
    accepted(
        "same_mod_fn",
        "fn hidden() -> Int32 { 3 }\nfn main() { assert_eq(hidden(), 3); }",
    );
}

#[test]
fn f1_same_module_private_inherent_method_is_accepted() {
    accepted(
        "same_mod_method",
        "struct S { v: Int32 }\nimpl S { fn hidden(&self) -> Int32 { self.v } }\n\
         fn main() { let s = S { v: 3 }; assert_eq(s.hidden(), 3); }",
    );
}

#[test]
fn f1_public_cross_module_function_is_accepted() {
    accepted(
        "pub_cross_fn",
        "mod m { pub fn shown() -> Int32 { 3 } }\nfn main() { assert_eq(m::shown(), 3); }",
    );
}

#[test]
fn f1_public_cross_module_method_and_assoc_fn_are_accepted() {
    accepted(
        "pub_cross_method",
        "mod m { pub struct S { pub v: Int32 } \
         impl S { pub fn make() -> S { S { v: 3 } } pub fn get(&self) -> Int32 { self.v } } }\n\
         fn main() { let s = m::S::make(); assert_eq(s.get(), 3); }",
    );
}

// ------------------------------------------------------------------ negative --

#[test]
fn f1_private_top_level_function_cross_module_is_rejected() {
    // Already enforced by resolve -- regression guard.
    rejected(
        "priv_cross_fn",
        "mod m { fn hidden() -> Int32 { 3 } }\nfn main() { let _ = m::hidden(); }",
    );
}

#[test]
fn f1_private_inherent_method_cross_module_is_rejected() {
    rejected(
        "priv_cross_method",
        "mod m { pub struct S { pub v: Int32 } impl S { fn hidden(&self) -> Int32 { self.v } } }\n\
         fn main() { let s = m::S { v: 3 }; let _ = s.hidden(); }",
    );
}

#[test]
fn f1_private_associated_function_cross_module_is_rejected() {
    rejected(
        "priv_cross_assoc",
        "mod m { pub struct S { pub v: Int32 } impl S { fn secret() -> S { S { v: 3 } } } }\n\
         fn main() { let _ = m::S::secret(); }",
    );
}

#[test]
fn f1_private_field_read_cross_module_is_rejected() {
    rejected(
        "priv_cross_field",
        "mod m { pub struct S { v: Int32 } pub fn mk() -> S { S { v: 3 } } }\n\
         fn main() { let s = m::mk(); let _ = s.v; }",
    );
}

#[test]
fn f1_private_field_construction_cross_module_is_rejected() {
    rejected(
        "priv_cross_field_ctor",
        "mod m { pub struct S { v: Int32 } }\n\
         fn main() { let _ = m::S { v: 3 }; }",
    );
}

#[test]
fn f1_method_syntax_does_not_bypass_privacy() {
    // The same private method must be inaccessible however it is called.
    rejected(
        "no_bypass_method",
        "mod m { pub struct S { pub v: Int32 } impl S { fn hidden(&self) -> Int32 { self.v } } }\n\
         fn main() { let s = m::S { v: 3 }; let _ = s.hidden(); }",
    );
}

#[test]
fn f1_qualified_syntax_does_not_bypass_privacy() {
    // Fully qualified / associated-function path to a private impl member.
    rejected(
        "no_bypass_qualified",
        "mod m { pub struct S { pub v: Int32 } impl S { fn hidden(s: &S) -> Int32 { s.v } } }\n\
         fn main() { let s = m::S { v: 3 }; let _ = m::S::hidden(&s); }",
    );
}
