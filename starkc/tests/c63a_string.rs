//! WP-C6.3a — native String and str runtime.
//!
//! The first slice of the Core native runtime (WP-C6.3 §23/§24). `stark_runtime::string` defines
//! the STARK semantics for `String`/`str` (byte `len`, UTF-8, lexicographic ordering — matching
//! 06-Standard-Library), and the generated backend emits every `String`/`str` `RuntimeFn` and the
//! str-output path as calls into it. `String` is Rust `String` (owning, non-`Copy`, slot-backed so
//! MIR controls destruction); `str` is `&str`; a `str` literal is a Rust `&'static str` literal.
//!
//! Each case runs through all three engines (HIR interpreter, MIR interpreter, native binary) and,
//! where it prints, the native stdout bytes are checked against the expectation.
//!
//! Deferred boundary (native only; HIR+MIR pass): a STORED interior `&str` that borrows an OWNED
//! `String` and is held across a block — including `String`'s own `==`/`<` (which lowers through
//! `String::as_str`) and an explicit `let v = s.as_str();` used after a branch. The stored borrow
//! overlaps the `String`'s slot-drop across the generated block-dispatch `loop { match __bb }`
//! back-edges (E0502) — the SAME dispatch-loop borrow-linearisation problem as WP-C6.1g-c, not a
//! String-specific defect. `str`-value comparison (literals, `&str` params) works natively.

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions};
use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::run_program;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

struct Front {
    hir: starkc::hir::Hir,
    tables: starkc::typecheck::TypeTables,
    file: Arc<SourceFile>,
}

fn front(tag: &str, src: &str) -> Front {
    let file = Arc::new(SourceFile::new(
        format!("c63a_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "{tag} typecheck: {errs:?}");
    Front {
        hir,
        tables: checked.tables,
        file,
    }
}

/// HIR + MIR + native all exit 0; the native stdout must equal `expect_out`. In-program `assert_eq`
/// carries value checks; `expect_out` carries the output-byte check.
fn agree_out(tag: &str, src: &str, expect_out: &str) {
    let f = front(tag, src);

    let hir_exec = interp::run_with_partial_output(&f.hir, f.file.clone(), &f.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag} HIR: {}", e.message));
    assert_eq!(hir_exec.status, 0, "{tag}: HIR must exit 0");
    assert_eq!(hir_exec.output, expect_out, "{tag}: HIR output");

    let program = lower_program(&f.hir, &f.tables, f.file.clone())
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");

    if rustc_available() {
        let verified = verify_program(&program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_c63a_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let artifact = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: dir.clone(),
                target_contract: "stark-64-v1".to_string(),
            },
        )
        .unwrap_or_else(|e| panic!("{tag} native build: {e:?}"));
        let run = std::process::Command::new(&artifact.binary_path)
            .output()
            .expect("run");
        assert!(run.status.success(), "{tag}: native must exit 0");
        assert_eq!(
            String::from_utf8_lossy(&run.stdout),
            expect_out,
            "{tag}: native stdout"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// No output expected.
fn agree(tag: &str, src: &str) {
    agree_out(tag, src, "");
}

/// HIR + MIR only — native deferred to WP-C6.1g-c (dispatch-loop borrow linearisation).
fn agree_hir_mir(tag: &str, src: &str) {
    let f = front(tag, src);
    let hir_exec = interp::run_with_partial_output(&f.hir, f.file.clone(), &f.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag} HIR: {}", e.message));
    assert_eq!(hir_exec.status, 0, "{tag}: HIR must exit 0");
    let program = lower_program(&f.hir, &f.tables, f.file.clone())
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");
}

#[test]
fn from_and_len() {
    agree(
        "from_len",
        "fn main() { let s = String::from(\"hello\"); assert_eq(s.len(), 5); }",
    );
}

#[test]
fn new_and_is_empty() {
    agree(
        "new_empty",
        "fn main() { let s = String::new(); assert_eq(s.is_empty(), true); }",
    );
}

#[test]
fn push_str_grows() {
    agree(
        "push_str",
        "fn main() { let mut s = String::from(\"ab\"); s.push_str(\"cd\"); assert_eq(s.len(), 4); }",
    );
}

#[test]
fn clear_empties() {
    agree(
        "clear",
        "fn main() { let mut s = String::from(\"ab\"); s.clear(); assert_eq(s.is_empty(), true); }",
    );
}

#[test]
fn contains_substring() {
    agree(
        "contains",
        "fn main() { let s = String::from(\"hello\"); assert_eq(s.contains(\"ell\"), true); }",
    );
}

#[test]
fn clone_copies_value() {
    agree(
        "clone",
        "fn main() { let a = String::from(\"hi\"); let b = a.clone(); assert_eq(b.len(), 2); }",
    );
}

#[test]
fn str_len_of_literal() {
    agree("str_len", "fn main() { assert_eq(\"abc\".len(), 3); }");
}

#[test]
fn str_to_string() {
    agree(
        "to_string",
        "fn main() { let s = \"xy\".to_string(); assert_eq(s.len(), 2); }",
    );
}

#[test]
fn string_returned_across_function() {
    agree(
        "ret_string",
        "fn mk() -> String { String::from(\"made\") }\n\
         fn main() { let s = mk(); assert_eq(s.len(), 4); }",
    );
}

#[test]
fn println_str_literal() {
    agree_out(
        "println_lit",
        "fn main() { println(\"greetings\"); }",
        "greetings\n",
    );
}

#[test]
fn print_str_no_newline() {
    agree_out(
        "print_lit",
        "fn main() { print(\"ab\"); print(\"cd\"); }",
        "abcd",
    );
}

#[test]
fn str_literal_equality() {
    agree(
        "lit_eq",
        "fn main() { let r = if \"a\" == \"a\" { 1 } else { 0 }; assert_eq(r, 1); }",
    );
}

#[test]
fn str_literal_ordering() {
    agree(
        "lit_ord",
        "fn main() { let r = if \"a\" < \"b\" { 1 } else { 0 }; assert_eq(r, 1); }",
    );
}

// ---- Char operations (Char is a Copy scalar) ----

#[test]
fn println_char_value() {
    agree_out("println_char", "fn main() { println('A'); }", "A\n");
}

#[test]
fn print_char_unicode() {
    // A multi-byte scalar exercises UTF-8 encoding on the output path.
    agree_out("unicode_char", "fn main() { print('\u{3bb}'); }", "\u{3bb}");
}

#[test]
fn push_char_grows() {
    agree(
        "push_char",
        "fn main() { let mut s = String::from(\"ab\"); s.push('c'); assert_eq(s.len(), 3); }",
    );
}

/// `String::pop` returns `Option<Char>` — the runtime `Option` is wrapped into the program's
/// generated Option enum (the bridge every collection accessor reuses).
#[test]
fn pop_char_some() {
    agree(
        "pop_some",
        "fn main() { let mut s = String::from(\"aX\"); let c = s.pop(); assert_eq(c.unwrap_or('?'), 'X'); }",
    );
}

#[test]
fn pop_char_none_on_empty() {
    agree(
        "pop_none",
        "fn main() { let mut s = String::new(); let c = s.pop(); assert_eq(c.is_some(), false); }",
    );
}

// ---- Deferred to WP-C6.1g-c (native); HIR+MIR pass. ----

/// `String` `==` lowers through `String::as_str`, producing a stored `&str` that borrows the owned
/// `String` across the branch — conflicts with the slot-drop under the block-dispatch loop.
#[test]
fn owned_string_equality_hir_mir() {
    agree_hir_mir(
        "string_eq",
        "fn main() { let a = String::from(\"x\"); let b = String::from(\"x\"); \
         let r = if a == b { 1 } else { 0 }; assert_eq(r, 1); }",
    );
}

/// An explicit stored interior `&str` used after being bound.
#[test]
fn stored_as_str_hir_mir() {
    agree_hir_mir(
        "as_str_store",
        "fn main() { let s = String::from(\"hello\"); let v = s.as_str(); assert_eq(v.len(), 5); }",
    );
}
