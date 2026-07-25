//! WP-C6.3e — native formatting and output (§28).
//!
//! `println`/`print`, rendered per STARK's canonical form — NOT Rust's `Debug` — now native for:
//! primitives (`Int*`/`UInt*` widened to `i64`/`u64`, `Bool`, `Float64`); user `Display` (calls the
//! user's `fmt`); and displayable COMPOSITES of primitive elements — tuple/array (`(a, b)` /
//! `[a, b]`) and `Option`/`Result` (`Some(v)`/`None`, `Ok(v)`/`Err(e)`), rendered as a print sequence
//! matching the interpreter's `Display for Value`, recursively. The canonical float formatter lives
//! in `stark_runtime::format` and `starkc::interp` delegates to it, so the HIR oracle and the native
//! binary format identically by construction.
//!
//! Each case checks that HIR, MIR, and native all exit 0 AND that native/MIR stdout equal the HIR
//! oracle's output byte-for-byte.
//!
//! Deferred and REFUSED pre-rustc (a bounded, tested boundary rather than an admitted divergence):
//! `Float32` anywhere in the Display path (DEV-105 — the `f32 -> f64` widening diverges across
//! engines) and an array longer than 64 (the renderer unrolls per element). Still deferred: composite
//! `str`/`String` elements, `Box`, `Vec` (a runtime loop), and nested user-`Display`.

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

// ---- Composite Display (C6.3e): tuple/array of primitives, rendered as a print sequence matching
// the interpreter's `Display for Value` — `(a, b)`, `[a, b]`. Native + MIR gain support (HIR-only
// before). `str`/`String` elements, Option/Result/Box, Vec, and nested user-Display are later
// slices. ----

#[test]
fn composite_tuple_of_primitives() {
    agree_out("tuple2", "fn main() { println((1, 2)); }");
}

#[test]
fn composite_tuple_mixed_primitives() {
    agree_out("tuple_mixed", "fn main() { println((1, true, 2.5)); }");
}

#[test]
fn composite_array() {
    agree_out("array", "fn main() { let a = [10, 20, 30]; println(a); }");
}

#[test]
fn composite_nested_tuple() {
    agree_out("nested", "fn main() { println(((1, 2), 3)); }");
}

#[test]
fn composite_array_of_tuples() {
    agree_out(
        "arr_tup",
        "fn main() { let a = [(1, 2), (3, 4)]; println(a); }",
    );
}

#[test]
fn composite_print_then_println() {
    agree_out(
        "comp_print",
        "fn main() { print((1, 2)); println((3, 4)); }",
    );
}

#[test]
fn composite_option_some_none() {
    agree_out(
        "opt",
        "fn main() { let s: Option<Int32> = Some(5); println(s); let n: Option<Int32> = None; println(n); }",
    );
}

#[test]
fn composite_result_ok_err() {
    agree_out(
        "result",
        "fn main() { let a: Result<Int32, Bool> = Ok(7); println(a); let b: Result<Int32, Bool> = Err(true); println(b); }",
    );
}

/// A composite NESTED inside an Option payload — the recursion renders `Some((1, 2))`.
#[test]
fn composite_option_of_tuple() {
    agree_out(
        "opt_tuple",
        "fn main() { let o: Option<(Int32, Int32)> = Some((1, 2)); println(o); }",
    );
}

// ---- Boundary shapes: exercise the recursion's edges before owning composites arrive. ----

#[test]
fn composite_nested_option() {
    agree_out(
        "opt_opt",
        "fn main() { let o: Option<Option<Int32>> = Some(None); println(o); }",
    );
}

#[test]
fn composite_option_of_result() {
    agree_out(
        "opt_res",
        "fn main() { let o: Option<Result<Int32, Bool>> = Some(Ok(5)); println(o); }",
    );
}

#[test]
fn composite_array_of_options() {
    agree_out(
        "arr_opt",
        "fn main() { let a: [Option<Int32>; 2] = [Some(1), None]; println(a); }",
    );
}

// ---- Vec Display (CD-121): a runtime LOOP renders `[e0, e1, …]`; the owning Vec is dropped after
// the render (Contract C). Built against the CD-120 contracts. ----

#[test]
fn composite_vec_of_ints() {
    agree_out(
        "vec_int",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(10); v.push(20); v.push(30); println(v); }",
    );
}

#[test]
fn composite_vec_empty() {
    agree_out(
        "vec_empty",
        "fn main() { let v: Vec<Int32> = Vec::new(); println(v); }",
    );
}

#[test]
fn composite_vec_singleton() {
    agree_out(
        "vec_one",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(7); println(v); }",
    );
}

#[test]
fn composite_vec_of_bools() {
    agree_out(
        "vec_bool",
        "fn main() { let mut v: Vec<Bool> = Vec::new(); v.push(true); v.push(false); println(v); }",
    );
}

// `print`/`println` MOVE an owned Vec (non-Copy), so the same Vec cannot be printed twice; two
// separate Vecs exercise `print` (no newline) then `println`.
#[test]
fn composite_vec_print_then_println() {
    agree_out(
        "vec_print",
        "fn main() { let mut a: Vec<Int32> = Vec::new(); a.push(1); a.push(2); print(a); \
         let mut b: Vec<Int32> = Vec::new(); b.push(3); println(b); }",
    );
}

/// A Vec whose elements are themselves composites (tuples) — the loop recurses into each element.
#[test]
fn composite_vec_of_tuples() {
    agree_out(
        "vec_tuple",
        "fn main() { let mut v: Vec<(Int32, Bool)> = Vec::new(); v.push((1, true)); v.push((2, false)); println(v); }",
    );
}

// ---- Nested user `Display` as a composite ELEMENT (CD-123): a user nominal at any depth runs its
// own `fmt` (NOT the aggregate `{field: value}` debug form); the interp oracle recurses the same way.
// ----

#[test]
fn nested_user_display_in_tuple() {
    agree_out(
        "nest_tuple",
        "struct P { v: Int32 }\n\
         impl Display for P { fn fmt(&self) -> String { String::from(\"CUSTOM\") } }\n\
         fn main() { let p = P { v: 7 }; println((p, 1)); }",
    );
}

/// The nested `fmt` genuinely branches on the field, so a wrong (aggregate) rendering would diverge.
#[test]
fn nested_user_display_in_array() {
    agree_out(
        "nest_array",
        "struct P { v: Int32 }\n\
         impl Display for P { fn fmt(&self) -> String { if self.v == 3 { String::from(\"v=3\") } else { String::from(\"other\") } } }\n\
         fn main() { let a: [P; 2] = [P { v: 3 }, P { v: 9 }]; println(a); }",
    );
}

/// A Drop-bearing (non-Copy) nested nominal: the tuple renders `(DROPPY, 1)` via the element's `fmt`
/// WITHOUT a double destructor, then the owning tuple drops it once (Contract C) — `DROP`. Native
/// stdout == HIR oracle proves the nested `fmt` clone is not separately dropped.
#[test]
fn nested_user_display_drop_bearing_in_tuple() {
    agree_out(
        "nest_drop",
        "struct D { v: Int32 }\n\
         impl Drop for D { fn drop(&mut self) { println(\"DROP\"); } }\n\
         impl Display for D { fn fmt(&self) -> String { String::from(\"DROPPY\") } }\n\
         fn main() { let d = D { v: 1 }; println((d, 1)); }",
    );
}

// ---- String/str as composite ELEMENTS (CD-122): rendered raw (no quotes), borrowed in place; the
// owning composite is dropped after the render (Contract C). ----

#[test]
fn composite_tuple_with_str() {
    agree_out("tuple_str", "fn main() { println((\"hi\", 1)); }");
}

#[test]
fn composite_tuple_with_string() {
    agree_out(
        "tuple_string",
        "fn main() { println((String::from(\"hi\"), 1)); }",
    );
}

#[test]
fn composite_array_of_strings() {
    agree_out(
        "arr_string",
        "fn main() { let a: [String; 2] = [String::from(\"a\"), String::from(\"b\")]; println(a); }",
    );
}

// ---- Refused pre-rustc (a bounded, TESTED boundary — not an admitted divergence). Lowering must
// reject these; typecheck accepts them (they are well-typed). ----

/// The program is well-typed but LOWERING must refuse it (deterministic pre-rustc boundary).
fn refused_by_lowering(tag: &str, src: &str) {
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
    assert!(
        errs.is_empty(),
        "{tag}: expected it to type-check, got {errs:?}"
    );
    assert!(
        lower_program(&hir, &checked.tables, file).is_err(),
        "{tag}: lowering must refuse this shape"
    );
}

// DEV-105: a SCALAR top-level `println(Float32)` is ADMITTED (the interpreters agree; only the
// native binary sees the f32-rounded value — an interpreter-only frozen-corpus snapshot relies on
// this). `Float32` is refused only where it would leak silently through native: the COMPOSITE path.
#[test]
fn float32_in_tuple_refused() {
    refused_by_lowering(
        "f32_tuple",
        "fn main() { let x: Float32 = 0.1f32; println((x, 1)); }",
    );
}

#[test]
fn float32_in_option_refused() {
    refused_by_lowering(
        "f32_opt",
        "fn main() { let x: Float32 = 0.1f32; let o: Option<Float32> = Some(x); println(o); }",
    );
}

/// A large array exceeds the unroll cap and is refused (bounded, not silently quadratic).
#[test]
fn large_array_display_refused() {
    refused_by_lowering(
        "big_arr",
        "fn main() { let a: [Int32; 100] = [0; 100]; println(a); }",
    );
}

// CD-122 deferrals: a non-Copy payload inside `Option`/`Result` needs WP-C5.3d enum-payload storage
// to borrow through a `VariantField`; a droppable composite carrying a borrow (`&str` beside an
// owned field) needs generated lifetimes. Both are refused at lowering, not admitted-but-broken.
#[test]
fn option_of_string_refused() {
    refused_by_lowering(
        "opt_string",
        "fn main() { let o: Option<String> = Some(String::from(\"x\")); println(o); }",
    );
}

#[test]
fn result_of_string_refused() {
    refused_by_lowering(
        "res_string",
        "fn main() { let r: Result<String, Int32> = Ok(String::from(\"ok\")); println(r); }",
    );
}

#[test]
fn droppable_tuple_carrying_borrow_refused() {
    refused_by_lowering(
        "tuple_mixed_str",
        "fn main() { println((String::from(\"owned\"), \"borrowed\", 42)); }",
    );
}

// CD-123 deferrals: nested user `Display` inside a `Vec` (loop-carried borrow, E0502) or an
// `Option`/`Result` (VariantField-payload borrow, E0716) — refused at lowering. Straight-line
// tuple/array/struct-field nesting works (see the positive tests above).
#[test]
fn nested_user_display_in_vec_refused() {
    refused_by_lowering(
        "nest_vec",
        "struct P { v: Int32 }\n\
         impl Display for P { fn fmt(&self) -> String { String::from(\"CUSTOM\") } }\n\
         fn main() { let mut v: Vec<P> = Vec::new(); v.push(P { v: 1 }); v.push(P { v: 2 }); println(v); }",
    );
}

#[test]
fn nested_user_display_in_option_refused() {
    refused_by_lowering(
        "nest_option",
        "struct P { v: Int32 }\n\
         impl Display for P { fn fmt(&self) -> String { String::from(\"CUSTOM\") } }\n\
         fn main() { let o: Option<P> = Some(P { v: 7 }); println(o); }",
    );
}
