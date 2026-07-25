//! WP-C6.3e — native formatting and output (§28), slice 1: primitives.
//!
//! `println`/`print` of `Int*`/`UInt*` (widened to `i64`/`u64`), `Bool`, and `Float64` now emit
//! natively, rendered per STARK's canonical form — NOT Rust's `Debug`. The canonical float formatter
//! lives in `stark_runtime::format` and `starkc::interp` delegates to it, so the HIR oracle and the
//! native binary format identically by construction. (`Float32` println is DEFERRED — DEV-105: the
//! `f32 -> f64` widening cast is evaluated differently across engines, a value-semantics
//! discrepancy, not a formatting one.)
//!
//! Each case checks that HIR, MIR, and native all exit 0 AND that the native stdout equals the HIR
//! oracle's output byte-for-byte. (Composite types — tuple/struct/enum/Option/Result/Vec — and user
//! `Display` land in later C6.3e slices.)

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

/// HIR + MIR + native all exit 0, and native stdout equals the HIR oracle's output.
fn agree_out(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(
        format!("c63e_{tag}.stark"),
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
    // These are self-contained three-engine cases, so the MIR oracle's OUTPUT is compared too, not
    // just its exit status.
    assert_eq!(
        mir_exec.output, expect,
        "{tag}: MIR stdout must equal the HIR oracle"
    );

    if rustc_available() {
        let verified = verify_program(&program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_c63e_{tag}_{}", std::process::id()));
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

#[test]
fn println_signed_ints() {
    agree_out("int", "fn main() { println(42); println(-7); println(0); }");
}

#[test]
fn println_int32_widens() {
    agree_out(
        "int32",
        "fn main() { let x: Int32 = 5; println(x); let y: Int8 = -3; println(y); }",
    );
}

#[test]
fn println_unsigned_ints() {
    agree_out(
        "uint",
        "fn main() { let x: UInt64 = 300; println(x); let y: UInt8 = 255; println(y); }",
    );
}

#[test]
fn println_bools() {
    agree_out("bool", "fn main() { println(true); println(false); }");
}

#[test]
fn println_float64_canonical() {
    // Canonical shortest form: 0.1 renders as "0.1", not "0.100000...".
    agree_out(
        "float",
        "fn main() { println(3.5); println(0.1); println(-0.0); println(100.0); }",
    );
}

// DEV-105 (deferred): `println` of a `Float32` widens `f32 -> f64` (`widen_for_print`), and the
// native binary sees the f32-rounded value (`0.1f32 as f64 == 0.10000000149011612`) while the HIR
// interpreter keeps the wider `0.1`. That is a cross-engine Float32 value-semantics discrepancy in
// how the widening cast is evaluated, NOT a formatting-wiring issue — the canonical renderer is
// shared and correct. A Float32 formatting case lands once the cast agrees across all three engines.

#[test]
fn print_without_newline_then_println() {
    agree_out("print", "fn main() { print(1); print(2); println(3); }");
}

#[test]
fn mixed_primitive_output() {
    agree_out(
        "mixed",
        "fn main() { print(true); print(1); println(2.5); println(false); }",
    );
}

// ---- User `Display` dispatch (clears the C6.2d native-Display deferral) ----

/// `println(p)` on a Copy struct with a user `Display` calls the user's `fmt` and prints its result
/// (never Rust's `Debug`). The by-value arg has no destructor (Copy), so no drop is emitted.
#[test]
fn user_display_copy_struct() {
    agree_out(
        "display_struct",
        "struct P { v: Int32 }\n\
         impl Display for P { fn fmt(&self) -> String { String::from(\"CUSTOM\") } }\n\
         fn main() { let p = P { v: 7 }; println(p); }",
    );
}

/// A `Display` whose `fmt` genuinely BRANCHES on the receiver's field, so the printed text depends
/// on the value (the oracle-vs-native comparison would diverge if `self` were ignored).
#[test]
fn user_display_reads_field() {
    agree_out(
        "display_field",
        "struct P { v: Int32 }\n\
         impl Display for P { fn fmt(&self) -> String { if self.v == 3 { String::from(\"v=3\") } else { String::from(\"other\") } } }\n\
         fn main() { let p = P { v: 3 }; println(p); let q = P { v: 9 }; println(q); }",
    );
}

/// A Drop-bearing (non-Copy) `Display` type whose destructor is OBSERVABLE: the printed output is
/// `DROPPY` (the `fmt` result) THEN `DROP` (the arg's destructor, run after the bytes are
/// submitted). Native stdout == HIR oracle proves the destructor runs exactly once, in order —
/// exercising the arg-drop path the Copy case skips.
#[test]
fn user_display_drop_bearing_runs_destructor_after_output() {
    agree_out(
        "display_drop",
        "struct D { v: Int32 }\n\
         impl Drop for D { fn drop(&mut self) { println(\"DROP\"); } }\n\
         impl Display for D { fn fmt(&self) -> String { String::from(\"DROPPY\") } }\n\
         fn main() { let d = D { v: 1 }; println(d); }",
    );
}

/// User `Display` on an enum.
#[test]
fn user_display_enum() {
    agree_out(
        "display_enum",
        "enum E { A, B }\n\
         impl Display for E { fn fmt(&self) -> String { String::from(\"variant\") } }\n\
         fn main() { let e = E::A; println(e); }",
    );
}
