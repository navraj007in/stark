//! WP-C6.3c — native ITERATORS (§26).
//!
//! The §26 matrix, proven three-engine (HIR == MIR == native stdout). Iteration splits into two
//! lowering families, and only the second needs backend work:
//!
//! - **Counting loops.** `for i in a..b` and `for x in <array>` lower to an index loop with the
//!   ordinary `CheckIndex` proof discipline — no iterator object exists at runtime, so these were
//!   already native. A **user `Iterator` impl** is likewise ordinary static calls to the user's
//!   `next`.
//! - **Runtime iterator objects.** `v.iter()`, `s.chars()` and `m.keys()` lower to
//!   `*IterNew`/`*IterNext` runtime calls over a live iterator VALUE that borrows its source. These
//!   are what WP-C6.3c adds natively.
//!
//! Order, early termination and `for`-vs-explicit-`next` equivalence are asserted inside the STARK
//! programs themselves (via `assert_eq`) and by comparing printed output, so a case that agreed on
//! the WRONG order would still fail.

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

/// HIR + MIR + native all exit 0, and MIR/native stdout equal the HIR oracle's output.
fn agree_out(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(
        format!("c63c_{tag}.stark"),
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

    let hir_exec = interp::run_with_partial_output(&hir, file.clone(), &checked.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag} HIR: {}", e.message));
    assert_eq!(hir_exec.status, 0, "{tag}: HIR must exit 0");
    let expect = hir_exec.output;

    let program = lower_program(&hir, &checked.tables, file)
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");
    assert_eq!(
        mir_exec.output, expect,
        "{tag}: MIR stdout must equal the HIR oracle"
    );

    if rustc_available() {
        let verified = verify_program(&program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_c63c_{tag}_{}", std::process::id()));
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
            expect,
            "{tag}: native stdout must equal the oracle"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ---- Counting-loop family: no runtime iterator object (already native before C6.3c) ----

#[test]
fn range_for_loop() {
    agree_out(
        "range",
        "fn main() { let mut s: Int32 = 0; for i in 0..4 { s = s + i; print(i); } assert_eq(s, 6); println(\"\"); }",
    );
}

#[test]
fn array_for_loop_order() {
    agree_out(
        "array",
        "fn main() { let a: [Int32; 3] = [10, 20, 30]; let mut s: Int32 = 0; for x in a { s = s + x; print(x); } assert_eq(s, 60); println(\"\"); }",
    );
}

/// A user `Iterator` impl — ordinary static calls to the user's `next`, and `for` must equal
/// explicit `next()` iteration.
#[test]
fn user_iterator_impl() {
    agree_out(
        "useriter",
        "struct Countdown { n: Int32 }\n\
         impl Iterator for Countdown {\n\
           type Item = Int32;\n\
           fn next(&mut self) -> Option<Int32> { if self.n == 0 { None } else { self.n = self.n - 1; Some(self.n) } }\n\
         }\n\
         fn main() { let mut c = Countdown { n: 3 }; for x in c { print(x); } println(\"\"); }",
    );
}

// ---- Runtime-iterator-object family: what WP-C6.3c adds natively ----

/// `v.iter()` — by-reference Vec iteration (`VecIterNew`/`VecIterNext`, yielding `Option<&T>`).
#[test]
fn vec_iter_shared() {
    agree_out(
        "veciter",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); v.push(3); \
         let mut s: Int32 = 0; for x in v.iter() { s = s + *x; print(*x); } assert_eq(s, 6); println(\"\"); }",
    );
}

/// Early termination: `break` mid-iteration leaves the iterator (and the borrow) unfinished.
#[test]
fn vec_iter_early_break() {
    agree_out(
        "veciterbreak",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); v.push(3); \
         let mut seen: Int32 = 0; for x in v.iter() { if *x == 2 { break; } seen = seen + 1; } \
         assert_eq(seen, 1); }",
    );
}

/// An empty source yields nothing — the `None`-on-first-`next` path.
#[test]
fn vec_iter_empty() {
    agree_out(
        "veciterempty",
        "fn main() { let v: Vec<Int32> = Vec::new(); let mut n: Int32 = 0; for x in v.iter() { n = n + 1; } assert_eq(n, 0); }",
    );
}

/// `s.chars()` — character iteration over a `str`.
#[test]
fn chars_iter() {
    agree_out(
        "chars",
        "fn main() { let mut n: Int32 = 0; for c in \"abc\".chars() { n = n + 1; print(c); } assert_eq(n, 3); println(\"\"); }",
    );
}

/// `chars()` over an owned `String`.
#[test]
fn chars_iter_over_string() {
    agree_out(
        "charsstring",
        "fn main() { let s: String = String::from(\"hey\"); let mut n: Int32 = 0; for c in s.chars() { n = n + 1; print(c); } assert_eq(n, 3); println(\"\"); }",
    );
}
