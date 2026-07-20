//! WP-C4.7-4 — DEV-069: the front end and the HIR oracle must be multi-file-span-clean.
//!
//! STARK parses each file of a multi-file program (`mod helper;`) separately, so every `Span` is
//! **file-relative**: byte offsets into the file that declares the item, and meaningless against
//! any other file's text. The type checker, borrow checker, and reference interpreter each hold
//! one "current file" and used to read every span against it. That was correct for spans of the
//! item being checked and silently wrong for spans of any item being *looked up* — another
//! type's name, another impl's method names, a trait's method names — producing four failure
//! shapes documented in DEV-069: an out-of-bounds panic, garbage method names, unparseable
//! literals, and wrong-field reads at runtime.
//!
//! These tests pin the fix from the outside: a two-file program that exercises every cross-file
//! read must typecheck clean, borrow-check clean, and produce the right output. The MIR side was
//! already clean (`ProgramMeta` per-item files) and is covered by
//! `mir_differential.rs::multi_file_module_program_agrees_with_qualified_symbols`.
//!
//! The dependency file is deliberately made LONGER than the entry file. That is what turns a
//! wrong-file read from "reads the wrong bytes" into "reads out of bounds", and it is how
//! DEV-069's panic shape was originally hit.

use starkc::diag::Severity;
use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::PathBuf;
use std::sync::Arc;

/// Writes a two-file program to a unique temp directory and returns the entry file's path.
fn write_program(tag: &str, helper_src: &str, main_src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("stark_dev069_{}_{}", tag, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("helper.stark"), helper_src).unwrap();
    let main_path = dir.join("main.stark");
    std::fs::write(&main_path, main_src).unwrap();
    main_path
}

struct Checked {
    hir: starkc::hir::Hir,
    file: Arc<SourceFile>,
    tables: typecheck::TypeTables,
}

/// Runs parse → resolve → typecheck (which runs borrowck) and asserts every stage is clean.
fn front_end(main_path: &PathBuf) -> Checked {
    let src = std::fs::read_to_string(main_path).unwrap();
    let file = Arc::new(SourceFile::new(main_path.to_string_lossy(), src));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "a multi-file program must check clean; got: {errors:?}"
    );
    Checked {
        hir,
        file,
        tables: checked.tables,
    }
}

/// DEV-069 shapes (b) cross-file methods, (c) cross-file literals, (d) cross-file field reads —
/// all three in one program, checked AND executed.
#[test]
fn cross_file_methods_fields_and_literals_check_and_run() {
    // The helper is longer than the entry file on purpose (see the module comment).
    let helper = "pub struct Point { pub x: Int32, pub y: Int32 }\n\
                  impl Point {\n\
                      pub fn sum(&self) -> Int32 { self.x + self.y }\n\
                      pub fn scaled(&self, k: Int32) -> Int32 { (self.x + self.y) * k }\n\
                  }\n\
                  pub fn make() -> Point { Point { x: 11, y: 22 } }\n\
                  pub fn big_literal() -> Int32 { 12345 }\n";
    let main = "mod helper;\n\
                fn main() {\n\
                    let p = helper::make();\n\
                    println(p.sum());\n\
                    println(p.x);\n\
                    println(p.scaled(2));\n\
                    println(helper::big_literal());\n\
                }\n";
    let path = write_program("mfl", helper, main);
    let checked = front_end(&path);
    let run = interp::run_with_partial_output(&checked.hir, checked.file.clone(), &checked.tables)
        .expect("the oracle must run a multi-file program");
    let stdout = run.output;
    assert_eq!(stdout, "33\n11\n66\n12345\n");
}

/// DEV-069 shape (a): the panic. A dependency file much longer than the entry file used to make
/// `text(span)` index out of bounds and abort the compiler. Nothing here is exotic — the length
/// difference alone was the trigger.
#[test]
fn a_long_dependency_file_does_not_panic_the_front_end() {
    let mut helper = String::from("pub struct Wide { pub v: Int32 }\n");
    helper.push_str("impl Wide {\n    pub fn get(&self) -> Int32 { self.v }\n}\n");
    for i in 0..40 {
        helper.push_str(&format!(
            "pub fn filler_{i}() -> Int32 {{ {i} }}  // padding to outrun the entry file\n"
        ));
    }
    helper.push_str("pub fn make_wide() -> Wide { Wide { v: 7 } }\n");
    let main = "mod helper;\nfn main() { println(helper::make_wide().get()); }\n";
    let path = write_program("long", &helper, main);
    let checked = front_end(&path);
    let run = interp::run_with_partial_output(&checked.hir, checked.file.clone(), &checked.tables)
        .expect("the oracle must run a multi-file program");
    let stdout = run.output;
    assert_eq!(stdout, "7\n");
}

/// Cross-file `Drop` and trait dispatch: destructor discovery and trait-method lookup both scan
/// every impl in the program, so both read trait/method names out of other files. Drop order is
/// the observable, so a wrong-file read here would be a silent behavior bug, not a diagnostic.
#[test]
fn cross_file_trait_impls_and_drop_run_correctly() {
    let helper = "pub trait Speak { fn speak(&self) -> Int32 { 1 } }\n\
                  pub struct Loud { pub n: Int32 }\n\
                  impl Speak for Loud { fn speak(&self) -> Int32 { self.n * 10 } }\n\
                  impl Drop for Loud { fn drop(&mut self) { println(self.n); } }\n\
                  pub struct Quiet { pub n: Int32 }\n\
                  impl Speak for Quiet {}\n\
                  pub fn loud(n: Int32) -> Loud { Loud { n: n } }\n\
                  pub fn quiet(n: Int32) -> Quiet { Quiet { n: n } }\n";
    let main = "mod helper;\n\
                use helper::Speak;\n\
                fn main() {\n\
                    let l = helper::loud(4);\n\
                    let q = helper::quiet(9);\n\
                    println(l.speak());\n\
                    println(q.speak());\n\
                }\n";
    let path = write_program("drop", helper, main);
    let checked = front_end(&path);
    let run = interp::run_with_partial_output(&checked.hir, checked.file.clone(), &checked.tables)
        .expect("the oracle must run a multi-file program");
    let stdout = run.output;
    // 40 = the overriding impl; 1 = the trait default (not overridden by `impl Speak for Quiet`);
    // 4 = `Loud`'s destructor, which must still run at end of scope across the file boundary.
    assert_eq!(stdout, "40\n1\n4\n");
}
