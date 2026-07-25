//! WP-C6.3b (continued) — native TRAPPING `Vec` operations, checked interior access, and slice views.
//!
//! C6.3b landed the `Vec`/`Box` VALUE surface; what stayed deferred was everything that either TRAPS
//! on a bad index or hands out an INTERIOR reference:
//!
//! - `v[i]` — traps `IndexOutOfBounds`, and must report the USER'S source location (DEV-107).
//! - `v.get(i)` / `v.get_mut(i)` — checked access that never traps, yielding `Option<&T>`.
//! - `v.remove(i)` — traps on a bad index.
//! - `&v[a..b]` / `&a[a..b]` — slice views, which trap on an inverted or out-of-range bound, plus
//!   `len`/`is_empty` on the view.
//!
//! Success paths are proven three-engine (HIR == MIR == native stdout). Trap paths additionally
//! assert the trap CATEGORY and the exact SOURCE LINE on stderr, so a trap that fires with the wrong
//! provenance fails here rather than being silently accepted.

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

struct Compiled {
    program: starkc::mir::MirProgram,
    expect: String,
}

fn front_end(tag: &str, src: &str) -> Compiled {
    let file = Arc::new(SourceFile::new(
        format!("c63bt_{tag}.stark"),
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
    // On a trapping program the HIR interpreter fails; its partial output is still the oracle.
    let expect = match interp::run_with_partial_output(&hir, file.clone(), &checked.tables) {
        Ok(exec) => exec.output,
        Err((_, partial)) => partial,
    };
    let program = lower_program(&hir, &checked.tables, file)
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    Compiled { program, expect }
}

/// HIR + MIR + native all complete, with MIR/native stdout equal to the HIR oracle.
fn agree_out(tag: &str, src: &str) {
    let Compiled { program, expect } = front_end(tag, src);
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");
    assert_eq!(mir_exec.output, expect, "{tag}: MIR stdout vs HIR oracle");

    if rustc_available() {
        let (run, _) = build_and_run(tag, &program);
        assert!(run.status.success(), "{tag}: native must exit 0");
        assert_eq!(
            String::from_utf8_lossy(&run.stdout),
            expect,
            "{tag}: native stdout vs oracle"
        );
    }
}

fn build_and_run(tag: &str, program: &starkc::mir::MirProgram) -> (std::process::Output, String) {
    let verified = verify_program(program).unwrap();
    let dir = std::env::temp_dir().join(format!("stark_c63bt_{tag}_{}", std::process::id()));
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
    let _ = std::fs::remove_dir_all(&dir);
    (run, format!("c63bt_{tag}.stark"))
}

/// The program traps in every engine, and NATIVELY reports the given category at the given source
/// line — the provenance check DEV-107 exists for.
fn traps_at(tag: &str, src: &str, category: &str, line: u32) {
    let Compiled { program, expect } = front_end(tag, src);
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir = run_program(verified);
    assert!(mir.is_err(), "{tag}: MIR must trap");

    if rustc_available() {
        let (run, file_name) = build_and_run(tag, &program);
        assert_eq!(run.status.code(), Some(101), "{tag}: trap exit code");
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(
            stderr.contains(category),
            "{tag}: stderr missing category {category:?}: {stderr}"
        );
        assert!(
            stderr.contains(&format!("{file_name}:{line}:")),
            "{tag}: stderr must name the USER's location {file_name}:{line}: {stderr}"
        );
        // Output produced before the trap still reaches stdout (CD-120 Contract B).
        assert_eq!(
            String::from_utf8_lossy(&run.stdout),
            expect,
            "{tag}: pre-trap stdout vs oracle"
        );
    }
}

// ---- Indexed read: success and trap ----

#[test]
fn vec_index_read() {
    agree_out(
        "index",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(10); v.push(20); \
         println(v[0u64]); println(v[1u64]); }",
    );
}

/// DEV-107: an out-of-bounds `v[i]` must name the USER's line, not a runtime-internal location.
#[test]
fn vec_index_out_of_bounds_reports_user_location() {
    traps_at(
        "indexoob",
        "fn main() {\n    let mut v: Vec<Int32> = Vec::new();\n    v.push(1);\n    println(v[5u64]);\n}\n",
        "index out of bounds",
        4,
    );
}

// ---- Checked interior access: never traps ----

#[test]
fn vec_get_some_and_none() {
    agree_out(
        "get",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(7); \
         println(v.get(0u64).is_some()); println(v.get(9u64).is_none()); }",
    );
}

#[test]
fn vec_get_mut_writes_through() {
    agree_out(
        "getmut",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); \
         let r = v.get_mut(0u64); assert(r.is_some()); println(v[0u64]); }",
    );
}

// ---- remove: success and trap ----

#[test]
fn vec_remove_shifts() {
    agree_out(
        "remove",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); v.push(3); \
         let x = v.remove(1u64); println(x); println(v.len()); println(v[1u64]); }",
    );
}

#[test]
fn vec_remove_out_of_bounds_traps() {
    traps_at(
        "removeoob",
        "fn main() {\n    let mut v: Vec<Int32> = Vec::new();\n    v.push(1);\n    let x = v.remove(4u64);\n}\n",
        "index out of bounds",
        4,
    );
}

// ---- Slice views ----

#[test]
fn array_slice_view() {
    agree_out(
        "slice",
        "fn main() { let a: [Int32; 4] = [1, 2, 3, 4]; let s = &a[1..3]; \
         println(s.len()); println(s[0u64]); println(s.is_empty()); }",
    );
}

#[test]
fn slice_out_of_range_traps() {
    traps_at(
        "sliceoob",
        "fn main() {\n    let a: [Int32; 3] = [1, 2, 3];\n    let s = &a[1..9];\n    println(s.len());\n}\n",
        "index out of bounds",
        3,
    );
}

/// An INVERTED window (`lo > hi`) traps rather than yielding an empty view.
#[test]
fn inverted_slice_bounds_trap() {
    traps_at(
        "sliceinv",
        "fn main() {\n    let a: [Int32; 3] = [1, 2, 3];\n    let s = &a[2..1];\n    println(s.len());\n}\n",
        "index out of bounds",
        3,
    );
}

/// A NEGATIVE bound traps. STARK ranges are `Int`-typed, so this is expressible and must not wrap
/// into a huge unsigned index.
#[test]
fn negative_slice_bound_traps() {
    traps_at(
        "sliceneg",
        "fn main() {\n    let a: [Int32; 3] = [1, 2, 3];\n    let lo: Int32 = -1;\n    let s = &a[lo..2];\n    println(s.len());\n}\n",
        "index out of bounds",
        4,
    );
}

/// An empty window at the very end (`lo == hi == len`) is legal and yields an empty view.
#[test]
fn empty_slice_at_end_is_legal() {
    agree_out(
        "sliceempty",
        "fn main() { let a: [Int32; 3] = [1, 2, 3]; let s = &a[3..3]; \
         println(s.len()); println(s.is_empty()); }",
    );
}

/// The INCLUSIVE form `a..=b` covers one more element than `a..b`.
#[test]
fn inclusive_slice_range() {
    agree_out(
        "sliceincl",
        "fn main() { let a: [Int32; 4] = [1, 2, 3, 4]; let s = &a[1..=2]; \
         println(s.len()); println(s[0u64]); println(s[1u64]); }",
    );
}

/// A slice over a `Vec` (not just an array) — `&Vec<T>` and `&[T; N]` reach the same view.
#[test]
fn slice_over_a_vec() {
    agree_out(
        "slicevec",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(5); v.push(6); v.push(7); \
         let s = &v[1..3]; println(s.len()); println(s[0u64]); }",
    );
}
