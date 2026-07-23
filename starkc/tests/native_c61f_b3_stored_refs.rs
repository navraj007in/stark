//! WP-C6.1f-b3 — stored references: a borrow bound to a user local, flowing across basic blocks.
//!
//! The C5 "ephemeral reference lane" admitted references only in **same-block compiler
//! temporaries**. The C6.1f-a matrix showed why that was stricter than necessary: all fifteen
//! backend-refused rows already verified under the MIR verifier *and ran correctly under the MIR
//! interpreter* — the gap was generated-Rust emission, not reference representation.
//!
//! Probing with the lane disabled identified the actual blocker, and it was **not** the
//! `ValueSlot`/borrow-checker conflict the matrix flagged as the design question: a same-block
//! borrow bound to a user local already built and ran, **including for a `Drop`-bearing owner**.
//! What failed was `E0381 "used binding isn't initialized"` — rustc's *definite-assignment*
//! analysis, not its borrow checker. A reference local is assigned in one arm of the generated
//! block-dispatch `loop { match … }` and read in another, which rustc cannot follow.
//!
//! Fix: a reference bound to a **user** local is declared `Option<&T> = None`, definitely
//! initialised at its declaration. Compiler temporaries keep the bare form — they are same-block by
//! construction, so rustc's definite-assignment check still guards them exactly as before, and
//! every previously working reference path is untouched.
//!
//! Still refused (unchanged): **returning** a reference, and storing one in an aggregate. The
//! `c61f_reference_boundary.rs` negative corpus — including the no-NLL case — passes unaltered.

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

/// HIR, MIR and native must all complete with exit 0.
fn agree(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(format!("b3_{tag}.stark"), src.to_string()));
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
    let dir = std::env::temp_dir().join(format!("b3t_{tag}_{}", std::process::id()));
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
fn c61f_b3_reference_bound_to_a_user_local() {
    agree(
        "shared_local",
        &format!("{P}fn main() {{ let p = P {{ v: 3 }}; let r = &p; assert_eq(r.get(), 3); }}"),
    );
    agree(
        "shared_local_field",
        &format!("{P}fn main() {{ let p = P {{ v: 3 }}; let r = &p; assert_eq(r.v, 3); }}"),
    );
    agree(
        "primitive",
        "fn main() { let x: Int32 = 5; let r = &x; assert_eq(*r, 5); }",
    );
    agree(
        "two_shared_borrows",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let a = &p; let b = &p; \
                  assert_eq(a.get() + b.get(), 6); }}"
        ),
    );
}

#[test]
fn c61f_b3_reference_flows_across_basic_blocks() {
    // The E0381 case: assigned in one dispatch-loop arm, read in another.
    agree(
        "across_if",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let r = &p; \
                  let n = if r.get() > 1 {{ r.get() }} else {{ 0 }}; assert_eq(n, 3); }}"
        ),
    );
    agree(
        "into_loop",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 2 }}; let r = &p; let mut s: Int32 = 0; \
                  let mut i: Int32 = 0; while i < 3 {{ s = s + r.get(); i = i + 1; }} \
                  assert_eq(s, 6); }}"
        ),
    );
}

#[test]
fn c61f_b3_mutable_reference_in_a_user_local() {
    // `Option<&mut T>` is not `Copy`, so access re-borrows out of the Option rather than
    // moving out of it — moving would make the second use fail to compile.
    agree(
        "mut_local",
        &format!(
            "{P}fn main() {{ let mut p = P {{ v: 3 }}; let r = &mut p; r.bump(); \
                  assert_eq(r.get(), 4); }}"
        ),
    );
}

#[test]
fn c61f_b3_references_into_fields_and_elements() {
    // A borrow of a `Copy` field must be a place expression: read mode may substitute a raw
    // projection COPY helper, and `&<copy>` would reference a temporary, not the field.
    agree(
        "struct_field",
        "struct P { v: Int32 }\nfn main() { let p = P { v: 3 }; let r = &p.v; assert_eq(*r, 3); }",
    );
    agree(
        "nested_field",
        "struct I { v: Int32 }\nstruct O { i: I }\n\
         fn main() { let o = O { i: I { v: 7 } }; let r = &o.i; assert_eq(r.v, 7); }",
    );
    agree(
        "array_element",
        "fn main() { let a: [Int32; 3] = [1, 2, 3]; let r = &a[1]; assert_eq(*r, 2); }",
    );
}

#[test]
fn c61f_b3_borrowing_a_drop_bearing_owner() {
    // The matrix flagged slot/drop-flag interaction as the design risk; it is not one.
    agree(
        "drop_owner",
        "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) { } }\n\
         fn main() { let d = D { v: 1 }; let r = &d; assert_eq(r.v, 1); }",
    );
    // Borrow ends with its block, then the owner is moved.
    agree(
        "borrow_then_move",
        &format!(
            "{P}fn take(p: P) -> Int32 {{ p.v }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; {{ let r = &p; assert_eq(r.get(), 3); }} \
                  assert_eq(take(p), 3); }}"
        ),
    );
}

#[test]
fn c61f_b3_unblocks_the_b2_boundaries_that_were_waiting_on_the_lane() {
    // b2 emitted the `&mut T` -> `&T` weakening correctly, but binding the result to a user local
    // hit the lane. Both halves are needed for this to run.
    agree(
        "annotated_local_weakening",
        &format!(
            "{P}fn g(m: &mut P) -> Int32 {{ let r: &P = m; r.get() }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; assert_eq(g(&mut p), 3); }}"
        ),
    );
}
