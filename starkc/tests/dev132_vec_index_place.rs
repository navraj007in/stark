//! **DEV-132: `&v[i].field` borrows the element; it does not read it by value.**
//!
//! # The distinction under test
//!
//! ```text
//! v[i].field       value read; may require Copy or move rules
//! &v[i].field      shared borrow; must NOT read the element by value
//! &mut v[i].field  mutable borrow; separate capability, not admitted here
//! v[i].field = x   mutation; must not become admitted through shared borrowing
//! ```
//!
//! Lowering used to emit `VecIndexGet` — a by-value element read — for the base of `&v[i].field`,
//! so V-COPY-1 required `Copy` and MIR-0016 refused every `Vec<NonCopy>`. The refusal was correct
//! for the MIR emitted; emitting it was the defect. The place path now goes through `VecGetRef`,
//! which already existed with a verified `(&Vec<T>, u64) -> Option<&T>` signature and no `Copy`
//! requirement.
//!
//! # Why the negative controls carry as much weight as the positives
//!
//! The risk in this change is not that borrowing stays broken — it is **accidentally broadening
//! place support**. Making a borrow representable must not make the element assignable, must not
//! admit a move out of the Vec, and must not let the place path leak into ordinary value lowering
//! and bypass MIR-0016. Each of those is pinned below.

mod support;

use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// The shared fixture: a non-`Copy` element with both a `Copy` and a non-`Copy` field.
const ITEM: &str = "struct Item { label: String, code: Int32 }\n";

fn build(src: &str, tag: &str) -> Result<String, String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
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

/// The ruling's headline case: a `Copy` field borrowed out of a non-`Copy` element, returned as a
/// reference from a function.
#[test]
fn borrowing_a_copy_field_of_a_non_copy_element_lowers_and_verifies() {
    let src = format!(
        "{ITEM}fn read_code(values: &Vec<Item>, i: UInt64) -> &Int32 {{ &values[i].code }}\n\
         fn main() {{ let mut v: Vec<Item> = Vec::new(); \
         v.push(Item {{ label: String::from(\"a\"), code: 7 }}); \
         println(*read_code(&v, 0u64)); }}\n"
    );
    let dump = build(&src, "dev132_copy_field").expect("must lower and verify");
    assert!(
        dump.contains("VecGetRef"),
        "the place path must use VecGetRef, not VecIndexGet:\n{dump}"
    );
}

/// A NON-`Copy` field borrowed out of a non-`Copy` element — `&values[i].label` is a `&String`.
/// This is `stark-mime`'s exact shape.
#[test]
fn borrowing_a_non_copy_field_of_a_non_copy_element_lowers_and_verifies() {
    let src = format!(
        "{ITEM}fn label_len(values: &Vec<Item>, i: UInt64) -> UInt64 {{ values[i].label.len() }}\n\
         fn main() {{ let mut v: Vec<Item> = Vec::new(); \
         v.push(Item {{ label: String::from(\"abc\"), code: 1 }}); \
         println(label_len(&v, 0u64)); }}\n"
    );
    build(&src, "dev132_non_copy_field").expect("must lower and verify");
}

/// Repeated borrows of the same element, and a borrow passed to a helper. A shared reference is
/// `Copy`, so neither use may consume anything.
#[test]
fn repeated_borrows_and_a_borrow_passed_to_a_helper_lower_and_verify() {
    let src = format!(
        "{ITEM}fn takes(text: &String) -> UInt64 {{ text.len() }}\n\
         fn main() {{ let mut v: Vec<Item> = Vec::new(); \
         v.push(Item {{ label: String::from(\"abc\"), code: 1 }}); \
         let a = takes(&v[0u64].label); \
         let b = takes(&v[0u64].label); \
         let c = v[0u64].code; \
         println(a); println(b); println(c); }}\n"
    );
    build(&src, "dev132_repeated").expect("must lower and verify");
}

/// **The index expression must be evaluated exactly once.** A place path that re-lowered its index
/// would double any side effect in it; asserting the emitted call count is how that stays true
/// regardless of what the runtime happens to do.
#[test]
fn the_index_expression_is_evaluated_once_per_access() {
    let src = format!(
        "{ITEM}fn main() {{ let mut v: Vec<Item> = Vec::new(); \
         v.push(Item {{ label: String::from(\"abc\"), code: 1 }}); \
         println(v[0u64].label.len()); }}\n"
    );
    let dump = build(&src, "dev132_once").expect("must lower and verify");
    let calls = dump.matches("VecGetRef").count();
    assert_eq!(
        calls, 1,
        "one indexed access must emit exactly one VecGetRef:\n{dump}"
    );
}

// -------------------------------------------------------- must remain rejected --

/// **Moving an element out of a Vec stays rejected.** The place path must not leak into ordinary
/// value lowering and bypass the `Copy` requirement `VecIndexGet` carries.
#[test]
fn a_by_value_read_of_a_non_copy_element_is_still_rejected() {
    let src = format!(
        "{ITEM}fn main() {{ let mut v: Vec<Item> = Vec::new(); \
         v.push(Item {{ label: String::from(\"a\"), code: 1 }}); \
         let item = v[0u64]; println(item.code); }}\n"
    );
    let err = build(&src, "dev132_move_element")
        .expect_err("moving a non-Copy element out of a Vec must not be admitted");
    assert!(
        err.contains("E0100") || err.contains("MIR-0016") || err.contains("Copy"),
        "expected a Copy/move refusal, got: {err}"
    );
}

/// **Moving a non-`Copy` FIELD out of an indexed element stays rejected** — the field is owned by
/// the Vec just as the element is.
#[test]
fn a_by_value_read_of_a_non_copy_field_is_still_rejected() {
    let src = format!(
        "{ITEM}fn main() {{ let mut v: Vec<Item> = Vec::new(); \
         v.push(Item {{ label: String::from(\"a\"), code: 1 }}); \
         let text = v[0u64].label; println(text.len()); }}\n"
    );
    let err = build(&src, "dev132_move_field")
        .expect_err("moving a non-Copy field out of an indexed element must not be admitted");
    assert!(
        err.contains("E0100") || err.contains("MIR-0016") || err.contains("Copy"),
        "expected a Copy/move refusal, got: {err}"
    );
}

/// **The decisive negative control.** A shared `VecGetRef` must not become a WRITABLE place merely
/// because the lowering now returns a dereferenceable projection. Assignment through an indexed
/// element must stay refused — by the checker, or by V-REF-1 (MIR-0014), which rejects a write
/// crossing the `Deref` of a shared reference.
#[test]
fn assignment_through_an_indexed_element_is_not_admitted() {
    let src = format!(
        "{ITEM}fn main() {{ let mut v: Vec<Item> = Vec::new(); \
         v.push(Item {{ label: String::from(\"a\"), code: 1 }}); \
         v[0u64].code = 9; println(v[0u64].code); }}\n"
    );
    let err = build(&src, "dev132_assign")
        .expect_err("assignment through a shared indexed borrow must not be admitted");
    assert!(
        !err.contains("VecGetRef requires"),
        "the refusal must not be an internal VecGetRef complaint: {err}"
    );
}
