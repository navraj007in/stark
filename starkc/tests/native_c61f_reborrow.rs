//! WP-C6.1f-b1 — receiver-position auto-dereference and re-borrowing.
//!
//! 03-Type-System TYPE-METHOD-002: auto-dereference "examines `S`, then **repeatedly** removes one
//! leading `&`/`&mut`; at each level receiver matching tries by-value, shared-borrow, then
//! exclusive-borrow form". A receiver that is already a reference must therefore be dereferenced
//! and **re-borrowed at the method's required mutability**.
//!
//! Lowering previously passed such a receiver through as a value, which was wrong twice over: it
//! never adjusted `&mut T` to `&T` (so a `&self` method reached through a `&mut` receiver failed
//! MIR verification), and it MOVED the reference — `&mut T` is not `Copy` — so `m.bump();
//! m.bump();` failed V-MOVE-1 on the second call. The front end accepted both shapes already; only
//! lowering could not express them.
//!
//! Each re-borrow is a **temporary** borrow that ends with its statement (03, "References and
//! Lifetimes" rule 4: "`f(&x); g(&mut x);` is legal"), so this adds no borrow the checker has not
//! approved and does not touch Core v1's lexical borrow duration. The negative corpus in
//! `c61f_reference_boundary.rs` pins that.

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions};
use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::run_program;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::mir::MirProgram;
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

/// Lower + verify + run under the MIR interpreter, asserting the HIR oracle agrees.
fn lower_verify_run(tag: &str, src: &str) -> MirProgram {
    let file = Arc::new(SourceFile::new(
        format!("c61f_{tag}.stark"),
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
    let program = lower_program(&hir, &checked.tables, file)
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} MIR verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR run: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");
    program
}

/// All three engines complete with exit 0.
fn agree(tag: &str, src: &str) {
    let program = lower_verify_run(tag, src);
    if !rustc_available() {
        return;
    }
    let verified = verify_program(&program).unwrap();
    let dir = std::env::temp_dir().join(format!("c61f_b1_{tag}_{}", std::process::id()));
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

// ------------------------------------------------- the three rows b1 unblocks --

#[test]
fn c61f_b1_mut_receiver_calls_a_shared_method() {
    // `&mut P` receiver, `&self` method: auto-deref to `P`, then auto-borrow `&P`.
    agree(
        "mut_recv_shared_method",
        &format!(
            "{P}fn f(m: &mut P) -> Int32 {{ m.get() }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; assert_eq(f(&mut p), 3); }}"
        ),
    );
}

#[test]
fn c61f_b1_mut_receiver_used_twice_is_reborrowed_not_moved() {
    // The V-MOVE-1 case: `&mut T` is not Copy, so passing it through moved it.
    agree(
        "mut_recv_twice",
        &format!(
            "{P}fn f(m: &mut P) {{ m.bump(); m.bump(); }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; f(&mut p); assert_eq(p.v, 5); }}"
        ),
    );
}

#[test]
fn c61f_b1_mut_receiver_mixes_mutable_and_shared_methods() {
    agree(
        "mut_recv_mixed",
        &format!(
            "{P}fn f(m: &mut P) -> Int32 {{ m.bump(); m.get() }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; assert_eq(f(&mut p), 4); }}"
        ),
    );
}

// --------------------------------------------- the paths that must not regress --

#[test]
fn c61f_b1_shared_receiver_paths_still_work() {
    agree(
        "shared_twice",
        &format!(
            "{P}fn f(r: &P) -> Int32 {{ r.get() + r.get() }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; assert_eq(f(&p), 6); }}"
        ),
    );
    // `self` forwarding to a sibling method — shared and mutable.
    agree(
        "self_forward_shared",
        &format!(
            "{P}impl P {{ fn both(&self) -> Int32 {{ self.get() + self.get() }} }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; assert_eq(p.both(), 6); }}"
        ),
    );
    agree(
        "self_forward_mut",
        &format!(
            "{P}impl P {{ fn twice(&mut self) {{ self.bump(); self.bump(); }} }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; p.twice(); assert_eq(p.v, 5); }}"
        ),
    );
    // Non-place receiver: the call result is materialised, then borrowed.
    agree(
        "call_result_receiver",
        &format!("{P}fn mk() -> P {{ P {{ v: 7 }} }}\nfn main() {{ assert_eq(mk().get(), 7); }}"),
    );
    // Field receiver, and a Drop-bearing receiver.
    agree(
        "field_receiver",
        "struct I { v: Int32 }\nimpl I { fn get(&self) -> Int32 { self.v } }\nstruct O { i: I }\n\
         fn main() { let o = O { i: I { v: 7 } }; assert_eq(o.i.get(), 7); }",
    );
    agree(
        "drop_receiver",
        "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) { } }\n\
         impl D { fn get(&self) -> Int32 { self.v } }\n\
         fn main() { let d = D { v: 4 }; assert_eq(d.get(), 4); }",
    );
}

// ------------------------------------ progress pin for the F4 representation half --

#[test]
fn c61f_b1_nested_reference_receiver_now_lowers_and_verifies() {
    // A nested (`&&P`) receiver used to fail MIR verification, because lowering passed the
    // reference through instead of peeling every layer. `lower_place_autoderef` peels all of them,
    // so repeated auto-deref (TYPE-METHOD-002) is now expressed correctly in MIR and the MIR
    // engine runs it.
    //
    // These programs are still refused by the generated-Rust backend's ephemeral reference lane
    // (they bind a borrow to a user local), which is WP-C6.1f-b3's job — NOT a defect in b1. This
    // test deliberately stops at "lowers, verifies and runs under the MIR interpreter" so the b1
    // gain is pinned and b3 does not have to rediscover it.
    lower_verify_run(
        "nested_recv_inferred",
        &format!("{P}fn main() {{ let p = P {{ v: 3 }}; let r = &p; let rr = &r; assert_eq(rr.get(), 3); }}"),
    );
    lower_verify_run(
        "nested_recv_from_param",
        &format!(
            "{P}fn f(r: &P) -> Int32 {{ let rr = &r; rr.get() }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; assert_eq(f(&p), 3); }}"
        ),
    );
    // Same for a `&mut` bound to a user local: verification now passes; only the lane blocks it.
    lower_verify_run(
        "mut_local_recv",
        &format!("{P}fn main() {{ let mut p = P {{ v: 3 }}; let r = &mut p; r.bump(); assert_eq(r.get(), 4); }}"),
    );
}
