//! WP-C6.2d — operator / `CoreTrait` semantics (WP-C6-ENTRY §20).
//!
//! The load-bearing property: **native execution invokes the user's STARK impl; a Rust equivalent
//! must never substitute.** The generated Rust emits NO `#[derive(PartialEq/Ord/Clone/Hash/..)]`
//! on STARK nominals — every operator and `CoreTrait` call routes explicitly through the written
//! impl. The adversarial types prove it: an `Eq` that is always true, a reversed `Ord`, an
//! observable `Clone`, a nonzero `Default`. If Rust's own trait were used, each of these would give
//! a different answer than the assertions demand.
//!
//! Anti-substitution is symmetric: a missing impl is REJECTED (E0500 for the operator traits, E0302
//! for a method call), never silently satisfied by a Rust derive.
//!
//! Coverage by engine:
//!   * Fully native (HIR + MIR + native): `Eq`/`!=`, `Ord` and all four comparison operators,
//!     `Clone`, `Default` (via `P::default()`), `From`.
//!   * HIR + MIR (native output/collection deferred to C6.3, Track C — see below): `Display`
//!     (`fmt` returns `String`, a by-value collection return) and `Hash` (a nominal HashMap key).
//!     The DISPATCH is proven here; the native runtime for the returned/collected value is C6.3.
//!
//! Deferred (owner decision, recorded as DEV-103/DEV-104 in the CD-107 ledger entry): `.into()`
//! deriving from a `From` impl (blanket `Into`), and `Default::default()` with a type-inferred
//! target. The spec lists `From`/`Into` as independent traits and mandates only `fn default() ->
//! Self`; `Fahrenheit::from(c)` and `P::default()` are the supported forms.

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
    errors: Vec<(Option<String>, String)>,
}

fn front(tag: &str, source: &str) -> Front {
    let file = Arc::new(SourceFile::new(
        format!("c62d_{tag}.stark"),
        source.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errors = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| (d.code.clone(), d.message.clone()))
        .collect();
    Front {
        hir,
        tables: checked.tables,
        file,
        errors,
    }
}

/// HIR + MIR must exit 0; native too when rustc is present. The in-program `assert_eq` carries the
/// value check that distinguishes the user impl from any Rust equivalent.
fn agree(tag: &str, source: &str) {
    let f = front(tag, source);
    assert!(f.errors.is_empty(), "{tag} typecheck: {:?}", f.errors);

    let hir_exec = interp::run_with_partial_output(&f.hir, f.file.clone(), &f.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag} HIR: {}", e.message));
    assert_eq!(hir_exec.status, 0, "{tag}: HIR must exit 0");

    let program = lower_program(&f.hir, &f.tables, f.file.clone())
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");

    if rustc_available() {
        let verified = verify_program(&program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_c62d_{tag}_{}", std::process::id()));
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
}

/// HIR + MIR must exit 0. Native output/collection runtime is deferred to C6.3, so native is not
/// exercised — the dispatch to the user impl is what is proven here.
fn agree_hir_mir(tag: &str, source: &str) {
    let f = front(tag, source);
    assert!(f.errors.is_empty(), "{tag} typecheck: {:?}", f.errors);

    let hir_exec = interp::run_with_partial_output(&f.hir, f.file.clone(), &f.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag} HIR: {}", e.message));
    assert_eq!(hir_exec.status, 0, "{tag}: HIR must exit 0");

    let program = lower_program(&f.hir, &f.tables, f.file.clone())
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");
}

/// The program must be REJECTED at type-check with `code` — anti-substitution: no Rust-derive
/// fallback fills a missing impl.
fn rejected(tag: &str, source: &str, code: &str) {
    let f = front(tag, source);
    assert!(
        f.errors.iter().any(|(c, _)| c.as_deref() == Some(code)),
        "{tag}: expected {code}, got {:?}",
        f.errors
    );
}

// ---- Fully native: the user impl is invoked, not a Rust equivalent ----

/// Adversarial `Eq` that is ALWAYS true: distinct values compare equal. A Rust `PartialEq` derive
/// would answer false — the assertion forces the user impl.
#[test]
fn eq_always_true() {
    agree(
        "eq_always_true",
        "struct P { v: Int32 }\n\
         impl Eq for P { fn eq(&self, o: &P) -> Bool { true } }\n\
         fn main() { let a = P { v: 1 }; let b = P { v: 2 }; let r = if a == b { 1 } else { 0 }; assert_eq(r, 1); }",
    );
}

/// `!=` desugars through the same user `Eq::eq` (always true → `a != b` is false).
#[test]
fn ne_routes_through_eq() {
    agree(
        "ne_via_eq",
        "struct P { v: Int32 }\n\
         impl Eq for P { fn eq(&self, o: &P) -> Bool { true } }\n\
         fn main() { let a = P { v: 1 }; let b = P { v: 2 }; let r = if a != b { 0 } else { 1 }; assert_eq(r, 1); }",
    );
}

/// Reversed `Ord`: `a < b` for `a.v < b.v` is false. All four comparison operators desugar through
/// the user `cmp`.
#[test]
fn reverse_ord_all_operators() {
    agree(
        "reverse_ord",
        "struct P { v: Int32 }\n\
         impl Eq for P { fn eq(&self, o: &P) -> Bool { self.v == o.v } }\n\
         impl Ord for P { fn cmp(&self, o: &P) -> Ordering { o.v.cmp(&self.v) } }\n\
         fn main() {\n\
           let a = P { v: 1 }; let b = P { v: 2 };\n\
           let lt = if a < b { 0 } else { 1 };\n\
           let gt = if a > b { 1 } else { 0 };\n\
           let le = if a <= b { 0 } else { 1 };\n\
           let ge = if a >= b { 1 } else { 0 };\n\
           assert_eq(lt + gt + le + ge, 4);\n\
         }",
    );
}

/// Observable `Clone`: the clone changes the value (+100), so the user impl is provably run.
#[test]
fn observable_clone() {
    agree(
        "observable_clone",
        "struct P { v: Int32 }\n\
         impl Clone for P { fn clone(&self) -> P { P { v: self.v + 100 } } }\n\
         fn main() { let a = P { v: 1 }; let b = a.clone(); assert_eq(b.v, 101); }",
    );
}

/// Nonzero `Default` via the supported `P::default()` form.
#[test]
fn nonzero_default() {
    agree(
        "nonzero_default",
        "struct P { v: Int32 }\n\
         impl Default for P { fn default() -> P { P { v: 42 } } }\n\
         fn main() { let a = P::default(); assert_eq(a.v, 42); }",
    );
}

/// `From` conversion invokes the user impl.
#[test]
fn from_conversion() {
    agree(
        "from_conv",
        "struct Celsius { v: Int32 }\n\
         struct Fahrenheit { v: Int32 }\n\
         impl From<Celsius> for Fahrenheit { fn from(c: Celsius) -> Fahrenheit { Fahrenheit { v: c.v * 2 + 30 } } }\n\
         fn main() { let c = Celsius { v: 10 }; let fr = Fahrenheit::from(c); assert_eq(fr.v, 50); }",
    );
}

// ---- HIR + MIR: dispatch proven; native output/collection is C6.3 ----

/// Adversarial `Display`: `fmt` returns a fixed string unlike any structural rendering. Dispatched
/// to the user impl (result length 6 = "CUSTOM"). Native `println`/String-return is C6.3.
#[test]
fn display_custom_dispatch() {
    agree_hir_mir(
        "display_custom",
        "struct P { v: Int32 }\n\
         impl Display for P { fn fmt(&self) -> String { String::from(\"CUSTOM\") } }\n\
         fn main() { let a = P { v: 1 }; let s = a.fmt(); assert_eq(s.len(), 6); }",
    );
}

/// Adversarial `Hash` (constant 0 → every key collides). A struct WITH the user `Hash` is usable as
/// a HashMap key and both distinct keys are retained (collision resolved via `Eq`). Native HashMap
/// runtime is C6.3.
#[test]
fn hash_collision_dispatch() {
    agree_hir_mir(
        "hash_collision",
        "struct K { v: Int32 }\n\
         impl Eq for K { fn eq(&self, o: &K) -> Bool { self.v == o.v } }\n\
         impl Hash for K { fn hash(&self) -> UInt64 { 0 } }\n\
         fn main() {\n\
           let mut m: HashMap<K, Int32> = HashMap::new();\n\
           m.insert(K { v: 1 }, 10); m.insert(K { v: 2 }, 20);\n\
           assert_eq(m.len(), 2);\n\
         }",
    );
}

// ---- Anti-substitution: a missing impl is rejected, never derived ----

/// `==` on a type with no `Eq` impl is rejected (E0500), not satisfied by a Rust `PartialEq` derive.
#[test]
fn eq_without_impl_rejected() {
    rejected(
        "eq_no_impl",
        "struct P { v: Int32 }\n\
         fn main() { let a = P { v: 1 }; let b = P { v: 2 }; let _r = a == b; }",
        "E0500",
    );
}

/// `<` on a type with no `Ord` impl is rejected (E0500).
#[test]
fn ord_without_impl_rejected() {
    rejected(
        "ord_no_impl",
        "struct P { v: Int32 }\n\
         fn main() { let a = P { v: 1 }; let b = P { v: 2 }; let _r = a < b; }",
        "E0500",
    );
}

/// `.clone()` on a type with no `Clone` impl is rejected (E0302).
#[test]
fn clone_without_impl_rejected() {
    rejected(
        "clone_no_impl",
        "struct P { v: Int32 }\n\
         fn main() { let a = P { v: 1 }; let _b = a.clone(); }",
        "E0302",
    );
}
