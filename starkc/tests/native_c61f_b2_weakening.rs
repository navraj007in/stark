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

fn agree(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(format!("b2_{tag}.stark"), src.to_string()));
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
    let program = lower_program(&hir, &checked.tables, file)
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");
    if !rustc_available() {
        return;
    }
    let verified = verify_program(&program).unwrap();
    let dir = std::env::temp_dir().join(format!("b2_{tag}_{}", std::process::id()));
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
    assert!(
        run.status.success(),
        "{tag}: native must exit 0; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
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
