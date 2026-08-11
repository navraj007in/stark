//! **DEV-230: a struct pattern may not name a field the struct does not have.**
//!
//! `Thing { valu: v }` — a typo for `value` — type-checked clean, bound nothing, and stopped
//! matching. A struct pattern over its own type is IRREFUTABLE, so the typo quietly made it
//! refutable: with a wildcard arm present the program took the wrong branch, and without one
//! `stark check` still reported OK and the program trapped at run time with "non-exhaustive match
//! reached". The enum struct-variant form behaved identically.
//!
//! `typecheck/patterns.rs` looked each field name up and simply skipped the miss:
//!
//! ```ignore
//! if let Some(expected_f_ty) = expected_fields.get(f_name) { ... }
//! // no `else`
//! ```
//!
//! Struct LITERALS have always rejected an unknown field with `E0001 field '<name>' does not
//! exist`. Patterns now agree with them, which is where the message and code come from.
//!
//! # Why these negative controls
//!
//! The repair adds a rejection inside a loop that runs for every struct-shaped pattern, so the
//! risk is rejecting valid ones. Valid struct patterns, valid enum struct-variant patterns, and
//! the shorthand binding form are all pinned below.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn errors(src: &str, tag: &str) -> Vec<String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file);
    let mut out: Vec<String> = rd
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
        .collect();
    let checked = typecheck::analyze(&hir);
    out.extend(
        checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == starkc::diag::Severity::Error)
            .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message)),
    );
    out
}

/// The silent face: a wildcard absorbs the mistake and the program takes the wrong branch.
#[test]
fn an_unknown_field_is_rejected_with_a_wildcard_present() {
    let errs = errors(
        "\
struct Thing { value: Int64 }
fn main() {
    let t = Thing { value: 7i64 };
    match t { Thing { valu: v } => println(v), _other => println(\"MISS\") }
}
",
        "dev230_wildcard",
    );
    assert!(
        errs.iter().any(|e| e.contains("'valu'")),
        "the diagnostic must name the field that does not exist: {errs:?}"
    );
}

/// The louder face, and the sharper statement of the defect: a struct pattern over its own type is
/// irrefutable, so this program has no missing case -- yet it used to trap at run time.
#[test]
fn an_unknown_field_is_rejected_without_a_wildcard() {
    let errs = errors(
        "\
struct Thing { value: Int64 }
fn main() { let t = Thing { value: 7i64 }; match t { Thing { valu: v } => println(v), } }
",
        "dev230_no_wildcard",
    );
    assert!(
        errs.iter().any(|e| e.contains("'valu'")),
        "this must fail in the front end, not as `non-exhaustive match reached` at run time: \
         {errs:?}"
    );
}

#[test]
fn an_unknown_field_is_rejected_on_an_enum_struct_variant() {
    let errs = errors(
        "\
enum Rec { One { n: Int64 }, Two }
fn main() {
    let r = Rec::One { n: 5i64 };
    match r { Rec::One { nope: v } => println(v), Rec::Two => println(\"two\"), _o => println(\"M\") }
}
",
        "dev230_enum_variant",
    );
    assert!(
        errs.iter().any(|e| e.contains("'nope'")),
        "the enum struct-variant arm reads the same field table and shares the defect: {errs:?}"
    );
}

// -- controls -----------------------------------------------------------------------------------

#[test]
fn valid_struct_and_variant_patterns_still_bind() {
    assert_eq!(
        errors(
            "\
struct Thing { value: Int64 }
enum Rec { One { n: Int64 }, Two }
fn main() {
    let t = Thing { value: 7i64 };
    match t { Thing { value: v } => println(v), }
    let r = Rec::One { n: 3i64 };
    match r { Rec::One { n } => println(n), Rec::Two => println(\"two\"), }
}
",
            "dev230_control"
        ),
        Vec::<String>::new(),
        "valid struct and enum struct-variant patterns, including the shorthand binding form, \
         must be unaffected"
    );
}
