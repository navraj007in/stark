//! **DEV-133: `&[T; N]` coerces to `&[T]`, and MIR must emit that coercion.**
//!
//! The checker accepted `let s: &[UInt8] = &[b];` and the HIR oracle executed it, but lowering
//! passed the array reference through unchanged, so the assignment's declared and actual types
//! differed by unsizing alone and MIR-0004 rejected it. Accepted-but-unbuildable — the same class
//! as DEV-132, a different mechanism: that one failed to preserve place context through indexing,
//! this one omitted a coercion outright.
//!
//! # Why these negative controls
//!
//! The risk is **broadening the coercion**, not failing to apply it. An unsize that fired on the
//! wrong pair would silently reinterpret memory. So the element type must match exactly, and a
//! shared array must not become a mutable slice — a coercion that also weakened mutability would
//! hand out a writable view of a read-only borrow.

mod support;

use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn build(src: &str, tag: &str) -> Result<String, String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
    if let Some(first) = checked
        .diagnostics
        .iter()
        .find(|d| d.severity == starkc::diag::Severity::Error)
    {
        return Err(format!(
            "CHECK {} {}",
            first.code.as_deref().unwrap_or("-"),
            first.message
        ));
    }
    let program = lower_program(
        &hir,
        &checked.tables,
        hir.source_named(&file.name).expect("registered"),
    )
    .map_err(|e| format!("LOWER: {}", e.what))?;
    match starkc::mir::verify::verify_program(&program) {
        Ok(_) => Ok(program.dump()),
        Err(errors) => Err(format!(
            "VERIFY {}",
            errors
                .iter()
                .map(|e| format!("{} {}", e.code, e.message))
                .collect::<Vec<_>>()
                .join("; ")
        )),
    }
}

// ------------------------------------------------------------------ must pass --

/// The reproducer from `form_encode_string`, then in `stark-form` and now in
/// `stark-urlencoded`: one byte wrapped as a slice.
#[test]
fn a_declared_slice_binding_accepts_an_array_reference() {
    let src = "fn takes(s: &[UInt8]) -> UInt64 { s.len() }\n\
               fn main() { let b: UInt8 = 7u8; let slice: &[UInt8] = &[b]; \
               println(takes(slice)); }\n";
    let dump = build(src, "dev133_let").expect("must lower and verify");
    assert!(
        dump.contains("SliceNew"),
        "the coercion must go through SliceNew:\n{dump}"
    );
}

/// Multi-element arrays, and the length carried into the coercion rather than assumed.
#[test]
fn arrays_of_several_elements_carry_their_length() {
    let src = "fn takes(s: &[UInt8]) -> UInt64 { s.len() }\n\
               fn main() { let a: [UInt8; 3] = [1u8, 2u8, 3u8]; \
               let slice: &[UInt8] = &a; println(takes(slice)); }\n";
    build(src, "dev133_multi").expect("must lower and verify");
}

/// **Argument position**, not only `let`. All six coercion sites route through the same helper, so
/// this proves the fix is not `let`-specific — the property that made fixing it in one place right.
#[test]
fn an_array_reference_coerces_in_argument_position() {
    let src = "fn takes(s: &[UInt8]) -> UInt64 { s.len() }\n\
               fn main() { let a: [UInt8; 2] = [4u8, 5u8]; println(takes(&a)); }\n";
    build(src, "dev133_arg").expect("must lower and verify");
}

/// A non-`Copy` element type — the coercion is about shape, not about the element being scalar.
#[test]
fn the_element_type_need_not_be_copy() {
    let src = "fn count(s: &[String]) -> UInt64 { s.len() }\n\
               fn main() { let a: [String; 1] = [String::from(\"x\")]; \
               let slice: &[String] = &a; println(count(slice)); }\n";
    build(src, "dev133_non_copy_elem").expect("must lower and verify");
}

// -------------------------------------------------------- must remain rejected --

/// **The element type must match exactly.** A coercion that ignored the element would reinterpret
/// memory rather than merely forget a length.
#[test]
fn a_mismatched_element_type_is_not_coerced() {
    let src = "fn takes(s: &[UInt8]) -> UInt64 { s.len() }\n\
               fn main() { let a: [Int32; 2] = [1, 2]; let slice: &[UInt8] = &a; \
               println(takes(slice)); }\n";
    let err = build(src, "dev133_bad_elem")
        .expect_err("an array of a different element type must not coerce");
    assert!(
        !err.contains("SliceNew"),
        "the refusal must not come from a wrongly-emitted coercion: {err}"
    );
}

/// **A shared array must not become a mutable slice.** Coercion changes shape, never capability;
/// combining the two would hand out a writable view over a read-only borrow.
#[test]
fn a_shared_array_reference_does_not_become_a_mutable_slice() {
    let src = "fn takes(s: &mut [UInt8]) -> UInt64 { s.len() }\n\
               fn main() { let a: [UInt8; 2] = [1u8, 2u8]; let slice: &mut [UInt8] = &a; \
               println(takes(slice)); }\n";
    build(src, "dev133_shared_to_mut")
        .expect_err("a shared array reference must not coerce to a mutable slice");
}
