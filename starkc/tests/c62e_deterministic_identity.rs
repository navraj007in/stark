//! WP-C6.2e — deterministic instance identity (WP-C6-ENTRY §21).
//!
//! A canonical MIR symbol must be a pure function of the program's LOGICAL structure — package and
//! module path, source name, and type arguments — never of the build. §21 requires that a clean
//! rebuild, a relocation to a different absolute path, and a dependency-declaration reorder all
//! leave every symbol byte-identical, and that no absolute path (or path-order artifact) enters
//! semantic symbol identity.
//!
//! The defect this closes: generic type arguments rendered a nominal as `struct#N` / `enum#N`, where
//! `N` is the raw `ItemId` index. That index is assigned by the item walk order, so declaring two
//! dependencies in the other order swapped the indices and changed the symbol
//! (`callA@[struct#5]` ⇄ `callA@[struct#10]`). `symbol_ty` now renders the nominal's content path
//! (`struct#liba::A`) instead — order-stable while still distinct from an identically-named core
//! type (a user may declare `struct Vec`).
//!
//! The forms exercised (§21): a plain function instance, a generic function instance (a nominal type
//! argument), a method / trait instance, a `Drop` instance, a generic nominal, and a
//! function-pointer sentinel.

use starkc::diag::Severity;
use starkc::mir::lower::lower_program;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::parse_package_graph;
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The sorted list of every lowered body's canonical symbol for the workspace rooted at `root`.
fn symbols(root: &Path) -> Vec<String> {
    let app_dir = root.join("app");
    let manifest = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest).unwrap();
    let (ast, pd) = parse_package_graph(&graph, LanguageOptions::CORE);
    assert!(pd.is_empty(), "parse: {pd:?}");
    let src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let rf = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        src,
    ));
    let (hir, rd) = resolve(&ast, rf.clone());
    assert!(rd.is_empty(), "resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, rf.clone());
    let errs: Vec<String> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    assert!(errs.is_empty(), "typecheck: {errs:?}");
    let program =
        lower_program(&hir, &checked.tables, rf).unwrap_or_else(|e| panic!("lower: {}", e.what));
    let mut s: Vec<String> = program
        .bodies
        .iter()
        .map(|b| b.instance.symbol.clone())
        .collect();
    s.sort();
    s
}

fn write_pkg(root: &Path, name: &str, manifest: String, src: &str) {
    let dir = root.join(name);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("starkpkg.json"), manifest).unwrap();
    std::fs::write(dir.join("src/main.stark"), src).unwrap();
}

/// Two absolute locations of DIFFERENT length, plus a second read of the first, must all agree —
/// and no symbol may contain an absolute path fragment or the process id.
#[test]
fn relocation_and_rebuild_are_stable() {
    let lib = "pub struct Wrap<T> { pub v: T }\n\
               pub trait Speak { fn say(&self) -> Int32; }\n\
               pub struct Dog { pub v: Int32 }\n\
               impl Speak for Dog { fn say(&self) -> Int32 { self.v } }\n\
               pub struct Res { pub v: Int32 }\n\
               impl Drop for Res { fn drop(&mut self) { } }\n\
               pub fn ident<T: Speak>(t: T) -> Int32 { t.say() }\n\
               pub fn plain(x: Int32) -> Int32 { x + 1 }\n\
               fn main() { }";
    let app = "use lib::{Wrap, Dog, Res, Speak, ident, plain};\n\
               fn apply(f: fn(Int32) -> Int32, x: Int32) -> Int32 { f(x) }\n\
               fn main() {\n\
                 let d = Dog { v: 7 }; let a = ident(d);\n\
                 let w = Wrap { v: 3 }; let r = Res { v: 1 };\n\
                 let g = apply(plain, 5);\n\
                 assert_eq(a + w.v + g, 16);\n\
               }";
    let build = |root: &Path| {
        let deps = r#""dependencies": { "lib": { "path": "../lib" } }"#;
        write_pkg(
            root,
            "app",
            format!(
                r#"{{ "name": "app", "version": "0.1.0", "entry": "src/main.stark", {deps} }}"#
            ),
            app,
        );
        write_pkg(
            root,
            "lib",
            r#"{ "name": "lib", "version": "0.1.0", "entry": "src/main.stark" }"#.to_string(),
            lib,
        );
    };

    let pid = std::process::id();
    let a = std::env::temp_dir().join(format!("stark_c62e_A_{pid}"));
    let b = std::env::temp_dir().join(format!("stark_c62e_much_longer_location_{pid}"));
    let _ = std::fs::remove_dir_all(&a);
    let _ = std::fs::remove_dir_all(&b);
    build(&a);
    build(&b);

    let first = symbols(&a);
    let rebuild = symbols(&a);
    let relocated = symbols(&b);

    assert_eq!(first, rebuild, "a clean rebuild changed the symbols");
    assert_eq!(
        first, relocated,
        "relocation to a different path changed the symbols"
    );

    // The generic instance must carry the nominal by content path, and nothing build-specific.
    assert!(
        first.contains(&"lib::ident@[struct#lib::Dog]".to_string()),
        "expected content-path generic instance symbol, got {first:?}"
    );
    let pid_str = pid.to_string();
    for s in &first {
        assert!(
            !s.contains(std::path::MAIN_SEPARATOR) && !s.contains("tmp") && !s.contains(&pid_str),
            "symbol leaks a build/path artifact: {s}"
        );
    }

    let _ = std::fs::remove_dir_all(&a);
    let _ = std::fs::remove_dir_all(&b);
}

/// Swapping the order two dependencies are declared in the manifest must not change any symbol.
#[test]
fn dependency_declaration_order_is_stable() {
    let liba = "pub struct A { pub v: Int32 }\n\
                pub trait SpeakA { fn say(&self) -> Int32; }\n\
                impl SpeakA for A { fn say(&self) -> Int32 { self.v } }\n\
                fn main() { }";
    let libb = "pub struct B { pub v: Int32 }\n\
                pub trait SpeakB { fn say(&self) -> Int32; }\n\
                impl SpeakB for B { fn say(&self) -> Int32 { self.v * 2 } }\n\
                fn main() { }";
    let app = "use liba::{A, SpeakA};\n\
               use libb::{B, SpeakB};\n\
               fn callA<T: SpeakA>(t: T) -> Int32 { t.say() }\n\
               fn callB<T: SpeakB>(t: T) -> Int32 { t.say() }\n\
               fn main() { let a = A { v: 3 }; let b = B { v: 4 }; assert_eq(callA(a) + callB(b), 11); }";

    let build = |root: &Path, a_first: bool| {
        let deps = if a_first {
            r#""dependencies": { "liba": { "path": "../liba" }, "libb": { "path": "../libb" } }"#
        } else {
            r#""dependencies": { "libb": { "path": "../libb" }, "liba": { "path": "../liba" } }"#
        };
        write_pkg(
            root,
            "app",
            format!(
                r#"{{ "name": "app", "version": "0.1.0", "entry": "src/main.stark", {deps} }}"#
            ),
            app,
        );
        write_pkg(
            root,
            "liba",
            r#"{ "name": "liba", "version": "0.1.0", "entry": "src/main.stark" }"#.to_string(),
            liba,
        );
        write_pkg(
            root,
            "libb",
            r#"{ "name": "libb", "version": "0.1.0", "entry": "src/main.stark" }"#.to_string(),
            libb,
        );
    };

    let pid = std::process::id();
    let a: PathBuf = std::env::temp_dir().join(format!("stark_c62e_reorder_a_{pid}"));
    let b: PathBuf = std::env::temp_dir().join(format!("stark_c62e_reorder_b_{pid}"));
    let _ = std::fs::remove_dir_all(&a);
    let _ = std::fs::remove_dir_all(&b);
    build(&a, true);
    build(&b, false);

    let a_first = symbols(&a);
    let b_first = symbols(&b);
    assert_eq!(
        a_first, b_first,
        "dependency declaration order changed the symbols"
    );
    assert!(
        a_first.contains(&"callA@[struct#liba::A]".to_string())
            && a_first.contains(&"callB@[struct#libb::B]".to_string()),
        "expected content-path generic instances, got {a_first:?}"
    );

    let _ = std::fs::remove_dir_all(&a);
    let _ = std::fs::remove_dir_all(&b);
}
