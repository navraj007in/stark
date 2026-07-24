//! WP-C6.2b-F5 — a bound on an impl head is visible in the impl's method bodies.
//!
//! WP-C6-ENTRY §2 carry-forward. A method call on a bounded generic *function* parameter already
//! resolved through the bound (`fn f<T: Sh>(t: T) { t.a() }`), but a bound written on the IMPL head
//! (`impl<T: Sh> W<T> { fn go(&self) { self.v.a() } }`) was invisible in the body — E0302 "method
//! 'a' not found for type 'T'". `typecheck` now tracks `current_impl_generics` alongside
//! `current_fn_generics` and consults both when resolving a method on a `Ty::Param` receiver.

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

/// HIR + MIR + (if rustc) native all complete at exit 0.
fn agree(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(format!("f5_{tag}.stark"), src.to_string()));
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
    let dir = std::env::temp_dir().join(format!("f5_{tag}_{}", std::process::id()));
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

const SH: &str = "trait Sh { fn a(&self) -> Int32; }\nstruct S { n: Int32 }\n\
                  impl Sh for S { fn a(&self) -> Int32 { self.n } }\n";

#[test]
fn c62b_f5_impl_head_bound_is_visible_in_a_method_body() {
    agree(
        "impl_head_bound",
        &format!(
            "{SH}struct W<T> {{ v: T }}\n\
             impl<T: Sh> W<T> {{ fn go(&self) -> Int32 {{ self.v.a() }} }}\n\
             fn main() {{ let w = W {{ v: S {{ n: 4 }} }}; assert_eq(w.go(), 4); }}"
        ),
    );
}

#[test]
fn c62b_f5_impl_head_bound_with_a_by_value_receiver() {
    agree(
        "impl_head_byvalue",
        &format!(
            "{SH}struct W<T> {{ v: T }}\n\
             impl<T: Sh> W<T> {{ fn go(self) -> Int32 {{ self.v.a() }} }}\n\
             fn main() {{ let w = W {{ v: S {{ n: 5 }} }}; assert_eq(w.go(), 5); }}"
        ),
    );
}

#[test]
fn c62b_f5_method_and_impl_generics_both_in_scope() {
    // The method has its own generic too; both the impl-head bound and the method generic resolve.
    agree(
        "both_generics",
        &format!(
            "{SH}struct W<T> {{ v: T }}\n\
             impl<T: Sh> W<T> {{ fn combine<U>(&self, x: U) -> U {{ let _ = self.v.a(); x }} }}\n\
             fn main() {{ let w = W {{ v: S {{ n: 4 }} }}; let n: Int32 = 9; \
             assert_eq(w.combine(n), 9); }}"
        ),
    );
}

#[test]
fn c62b_f5_an_unbounded_impl_param_still_rejects_the_method() {
    // Without the bound, the method genuinely does not exist -- must stay rejected (no over-accept).
    let src = format!(
        "{SH}struct W<T> {{ v: T }}\n\
         impl<T> W<T> {{ fn go(&self) -> Int32 {{ self.v.a() }} }}\n\
         fn main() {{ let w = W {{ v: S {{ n: 4 }} }}; assert_eq(w.go(), 4); }}"
    );
    let file = Arc::new(SourceFile::new("f5_neg.stark".to_string(), src));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file);
    assert!(
        checked
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error),
        "an unbounded impl parameter has no method `a`; must stay rejected"
    );
}
