//! WP-C6.2c — associated types.
//!
//! Proves the §19 matrix across all three engines (HIR interpreter, MIR interpreter, native debug
//! binary): trait associated-type declarations and impl bindings; `Self::Item`; `T::Item` resolved
//! both through an explicit `Trait<Item = ..>` binding and inferred from a call argument; nested
//! projections; associated types in signatures, cross-package, and inside a Drop-bearing nominal.
//!
//! Resolution work landed here:
//!   * `assoc_projections` — a program-wide `(nominal, assoc-name) -> bound type` table in both the
//!     front end and the MIR lowerer, so a concrete projection `<H as Holder>::Item` becomes the
//!     impl's binding at any call site (`Ty::Param("H::Item")` never survives into verified MIR).
//!   * deferred projection obligations — `fn first<T: Holder>(t: T) -> T::Item` whose base is fixed
//!     only by unifying the argument; discharged as soon as the call's arguments unify.
//!   * `check_trait_member_call` reads `Self::Item` spans against the TRAIT's file (DEV-101
//!     provenance), so a cross-package trait's projection is not mangled.
//!
//! Returning a runtime collection (`Vec<..>`) BY VALUE across a function boundary is a separate
//! native-linkage limitation (C6.3), independent of associated types — a plain `fn f() -> Vec<_>`
//! hits the same refusal — so it is intentionally not exercised here.

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions};
use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::run_program;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::{parse, parse_package_graph, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::Path;
use std::sync::Arc;

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// HIR + MIR + native must all complete with exit 0 (the in-program assertions carry the values).
fn agree(tag: &str, source: &str) {
    let file = Arc::new(SourceFile::new(
        format!("c62c_{tag}.stark"),
        source.to_string(),
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
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");

    if rustc_available() {
        let verified = verify_program(&program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_c62c_{tag}_{}", std::process::id()));
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

/// Declaration + impl binding + `Self::Item` return.
#[test]
fn self_item_return() {
    agree(
        "self_item",
        "trait Holder { type Item; fn get(&self) -> Self::Item; }\n\
         struct H { v: Int32 }\n\
         impl Holder for H { type Item = Int32; fn get(&self) -> Int32 { self.v } }\n\
         fn main() { let h = H { v: 6 }; assert_eq(h.get(), 6); }",
    );
}

/// An associated type in a signature PARAMETER position.
#[test]
fn self_item_parameter() {
    agree(
        "self_item_param",
        "trait Store { type Item; fn put(&self, x: Self::Item) -> Int32; }\n\
         struct S { }\n\
         impl Store for S { type Item = Int32; fn put(&self, x: Int32) -> Int32 { x } }\n\
         fn main() { let s = S {}; assert_eq(s.put(7), 7); }",
    );
}

/// The associated type is a nominal struct value.
#[test]
fn associated_type_is_nominal() {
    agree(
        "assoc_nominal",
        "struct P { v: Int32 }\n\
         trait Holder { type Item; fn get(&self) -> Self::Item; }\n\
         struct H { }\n\
         impl Holder for H { type Item = P; fn get(&self) -> P { P { v: 5 } } }\n\
         fn main() { let h = H {}; assert_eq(h.get().v, 5); }",
    );
}

/// The associated type is a tuple.
#[test]
fn associated_type_is_tuple() {
    agree(
        "assoc_tuple",
        "trait Holder { type Item; fn get(&self) -> Self::Item; }\n\
         struct H { }\n\
         impl Holder for H { type Item = (Int32, Int32); fn get(&self) -> (Int32, Int32) { (1, 2) } }\n\
         fn main() { let h = H {}; let t = h.get(); assert_eq(t.0 + t.1, 3); }",
    );
}

/// `T::Item` in a generic function, resolved from the concrete argument (no explicit binding). The
/// deferred projection obligation is discharged once `first`'s argument fixes `T = H`.
#[test]
fn projection_inferred_from_argument() {
    agree(
        "t_item_inferred",
        "trait Holder { type Item; fn get(&self) -> Self::Item; }\n\
         struct H { v: Int32 }\n\
         impl Holder for H { type Item = Int32; fn get(&self) -> Int32 { self.v } }\n\
         fn first<T: Holder>(t: T) -> T::Item { t.get() }\n\
         fn main() { assert_eq(first(H { v: 9 }), 9); }",
    );
}

/// Explicit binding constraint `T: Holder<Item = Int32>` pins the projection to a concrete type.
#[test]
fn explicit_binding_constraint() {
    agree(
        "explicit_bind",
        "trait Holder { type Item; fn get(&self) -> Self::Item; }\n\
         struct H { v: Int32 }\n\
         impl Holder for H { type Item = Int32; fn get(&self) -> Int32 { self.v } }\n\
         fn f<T: Holder<Item = Int32>>(t: T) -> Int32 { t.get() }\n\
         fn main() { assert_eq(f(H { v: 4 }), 4); }",
    );
}

/// A projected value used immediately (field access) through an unbounded generic projection: the
/// obligation must be discharged eagerly, at the call, not only at end of checking.
#[test]
fn projection_used_by_value() {
    agree(
        "assoc_drop_gen",
        "struct D { v: Int32 }\n\
         impl Drop for D { fn drop(&mut self) { } }\n\
         trait Holder { type Item; fn make(&self) -> Self::Item; }\n\
         struct H { }\n\
         impl Holder for H { type Item = D; fn make(&self) -> D { D { v: 8 } } }\n\
         fn build<T: Holder>(t: T) -> T::Item { t.make() }\n\
         fn main() { let d = build(H {}); assert_eq(d.v, 8); }",
    );
}

/// Nested: project `T::Item` then call a method on the projected value, under an explicit binding.
#[test]
fn nested_projection_then_method() {
    agree(
        "assoc_nested",
        "trait Holder { type Item; fn make(&self) -> Self::Item; }\n\
         struct Inner { v: Int32 }\n\
         impl Inner { fn val(&self) -> Int32 { self.v } }\n\
         struct H { }\n\
         impl Holder for H { type Item = Inner; fn make(&self) -> Inner { Inner { v: 11 } } }\n\
         fn build<T: Holder<Item = Inner>>(t: T) -> Int32 { t.make().val() }\n\
         fn main() { assert_eq(build(H {}), 11); }",
    );
}

/// Cross-package: a dependency declares the trait, its associated type and impl; the app projects
/// `T::Item` through a generic bounded by the dependency's trait. The projection's `Self::Item`
/// spans read against the TRAIT's file, not the caller's.
#[test]
fn cross_package_projection() {
    let lib = "pub trait Holder { type Item; fn get(&self) -> Self::Item; }\n\
               pub struct H { pub v: Int32 }\n\
               impl Holder for H { type Item = Int32; fn get(&self) -> Int32 { self.v } }\n\
               fn main() { }";
    let app = "use lib::{Holder, H};\n\
               fn first<T: Holder>(t: T) -> T::Item { t.get() }\n\
               fn main() { let h = H { v: 9 }; assert_eq(first(h), 9); }";

    let root = std::env::temp_dir().join(format!("stark_c62c_xpkg_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for (name, dep, src) in [("app", "lib", app), ("lib", "", lib)] {
        let dir = root.join(name);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let manifest = if dep.is_empty() {
            format!(r#"{{ "name": "{name}", "version": "0.1.0", "entry": "src/main.stark" }}"#)
        } else {
            format!(
                r#"{{ "name": "{name}", "version": "0.1.0", "entry": "src/main.stark", "dependencies": {{ "{dep}": {{ "path": "../{dep}" }} }} }}"#
            )
        };
        std::fs::write(dir.join("starkpkg.json"), manifest).unwrap();
        std::fs::write(dir.join("src/main.stark"), src).unwrap();
    }

    let app_dir = root.join("app");
    let manifest = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest).unwrap();
    let (ast, pd) = parse_package_graph(&graph, LanguageOptions::CORE);
    assert!(pd.is_empty(), "xpkg parse: {pd:?}");
    let src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        src,
    ));
    let (hir, rd) = resolve(&ast, root_file.clone());
    assert!(rd.is_empty(), "xpkg resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, root_file.clone());
    let errs: Vec<String> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    assert!(errs.is_empty(), "xpkg typecheck: {errs:?}");

    let program = lower_program(&hir, &checked.tables, root_file)
        .unwrap_or_else(|e| panic!("xpkg lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("xpkg verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("xpkg MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "xpkg: MIR must exit 0");

    if rustc_available() {
        let verified = verify_program(&program).unwrap();
        let out = root.join("out");
        let artifact = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: out,
                target_contract: "stark-64-v1".to_string(),
            },
        )
        .unwrap_or_else(|e| panic!("xpkg native build: {e:?}"));
        let run = std::process::Command::new(&artifact.binary_path)
            .output()
            .expect("run");
        assert!(run.status.success(), "xpkg: native must exit 0");
    }
    let _ = std::fs::remove_dir_all(Path::new(&root));
}
