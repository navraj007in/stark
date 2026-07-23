//! WP-C6.2a — canonical callable identity, and the generics/trait-dispatch shapes it unblocks.
//!
//! Owner ruling (conformance fix, NOT a CE3): every `Instance` derived from a `FnKey` — the body's
//! and every `Callee::Instance` reference — is built by ONE constructor
//! (`mir::lower::instance_from_key`), so the defining item and concrete type arguments always
//! agree with the canonical symbol. Call sites previously passed the RECEIVER NOMINAL as
//! `Instance.item`, producing "one symbol, two identities"; the C5.4a linkage preflight correctly
//! refused every method, trait, operator and associated-function call in the process.
//!
//! The linkage consistency check is NOT weakened — `a_mismatched_item_is_still_rejected` proves it
//! still fires on a deliberately inconsistent program.

use starkc::backend::generated_rust::{emit_native_debug, linkage, NativeBuildOptions};
use starkc::diag::Severity;
use starkc::hir::ItemId;
use starkc::interp;
use starkc::mir::interp::run_program;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::mir::{
    BasicBlock, BlockId, Callee, FileId, Instance, LocalId, MirBody, MirProgram, MirTy, Origin,
    Place, SourceInfo, Terminator, TypeContext,
};
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::{parse, parse_package_graph, ParseMode};
use starkc::resolve::resolve;
use starkc::source::{SourceFile, Span};
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

fn lower(
    tag: &str,
    source: &str,
) -> (
    MirProgram,
    starkc::hir::Hir,
    starkc::typecheck::TypeTables,
    Arc<SourceFile>,
) {
    let file = Arc::new(SourceFile::new(
        format!("c6_2_{tag}.stark"),
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
    let program = lower_program(&hir, &checked.tables, file.clone())
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    (program, hir, checked.tables, file)
}

/// HIR + MIR + native must all complete with exit 0 (the in-program assertions carry the values).
fn agree(tag: &str, source: &str) {
    let (program, hir, tables, file) = lower(tag, source);

    let hir_exec = interp::run_with_partial_output(&hir, file, &tables)
        .unwrap_or_else(|(e, _)| panic!("{tag} HIR: {}", e.message));
    assert_eq!(hir_exec.status, 0, "{tag}: HIR must exit 0");

    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");

    // Every call reference agrees with its body on symbol, item AND type_args.
    assert_reference_identity_matches_bodies(tag, &program);

    if rustc_available() {
        let verified = verify_program(&program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_c6_2_{tag}_{}", std::process::id()));
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

/// The C6.2a invariant, asserted directly rather than only through the linkage preflight: for every
/// `Callee::Instance` reference, the body named by its canonical symbol carries the SAME
/// `(symbol, item, type_args)` triple.
fn assert_reference_identity_matches_bodies(tag: &str, program: &MirProgram) {
    let bodies: std::collections::BTreeMap<&str, &Instance> = program
        .bodies
        .iter()
        .map(|b| (b.instance.symbol.as_str(), &b.instance))
        .collect();
    for body in &program.bodies {
        linkage::visit_instance_refs(body, &mut |referenced, _kind| {
            let defining = bodies.get(referenced.symbol.as_str()).unwrap_or_else(|| {
                panic!("{tag}: reference to `{}` names no body", referenced.symbol)
            });
            assert_eq!(
                (&referenced.item, &referenced.type_args),
                (&defining.item, &defining.type_args),
                "{tag}: `{}` referenced with a different identity than its body",
                referenced.symbol
            );
        });
    }
}

// ------------------------------------------------------- the unblocked shapes --

#[test]
fn c62a_inherent_method() {
    agree(
        "inherent",
        "struct P { v: Int32 }\nimpl P { fn get(&self) -> Int32 { self.v } }\n\
         fn main() { let p = P { v: 7 }; assert_eq(p.get(), 7); }",
    );
}

#[test]
fn c62a_associated_function() {
    agree(
        "assoc_fn",
        "struct P { v: Int32 }\nimpl P { fn new(v: Int32) -> P { P { v: v } } }\n\
         fn main() { let p = P::new(9); assert_eq(p.v, 9); }",
    );
}

#[test]
fn c62a_user_trait_implementation_method() {
    agree(
        "user_trait",
        "trait Shape { fn area(&self) -> Int32; }\nstruct Sq { s: Int32 }\n\
         impl Shape for Sq { fn area(&self) -> Int32 { self.s * self.s } }\n\
         fn main() { let q = Sq { s: 3 }; assert_eq(q.area(), 9); }",
    );
}

#[test]
fn c62a_default_trait_method() {
    agree(
        "default_method",
        "trait Greet { fn base(&self) -> Int32; fn twice(&self) -> Int32 { self.base() * 2 } }\n\
         struct G { v: Int32 }\nimpl Greet for G { fn base(&self) -> Int32 { self.v } }\n\
         fn main() { let g = G { v: 5 }; assert_eq(g.twice(), 10); }",
    );
}

#[test]
fn c62a_bounded_generic_calls_the_bound_method() {
    agree(
        "bounded_generic",
        "trait Shape { fn area(&self) -> Int32; }\nstruct Sq { s: Int32 }\n\
         impl Shape for Sq { fn area(&self) -> Int32 { self.s * self.s } }\n\
         fn total<T: Shape>(t: T) -> Int32 { t.area() }\n\
         fn main() { assert_eq(total(Sq { s: 4 }), 16); }",
    );
}

#[test]
fn c62a_generic_nominal_method() {
    agree(
        "generic_nominal_method",
        "struct Box2<T> { v: T }\nimpl<T> Box2<T> { fn get(self) -> T { self.v } }\n\
         fn main() { let b = Box2 { v: 7 }; assert_eq(b.get(), 7); }",
    );
}

#[test]
fn c62a_method_level_generic_method() {
    agree(
        "method_generic",
        "struct C { v: Int32 }\nimpl C { fn pick<T>(self, x: T) -> T { x } }\n\
         fn main() { let c = C { v: 1 }; let n: Int32 = 5; assert_eq(c.pick(n), 5); }",
    );
}

#[test]
fn c62a_associated_type() {
    agree(
        "assoc_type",
        "trait Holder { type Item; fn get(&self) -> Self::Item; }\nstruct H { v: Int32 }\n\
         impl Holder for H { type Item = Int32; fn get(&self) -> Int32 { self.v } }\n\
         fn main() { let h = H { v: 6 }; assert_eq(h.get(), 6); }",
    );
}

// --- operator / CoreTrait dispatch must invoke the STARK impl, not a Rust equivalent ---

#[test]
fn c62a_eq_operator_dispatch_uses_the_user_impl() {
    // Adversarial: `eq` is ALWAYS true. Rust's own `==` on the generated struct would say false.
    agree(
        "adversarial_eq",
        "struct Odd { v: Int32 }\nimpl Eq for Odd { fn eq(&self, other: &Odd) -> Bool { true } }\n\
         fn main() { let a = Odd { v: 1 }; let b = Odd { v: 2 }; assert(a == b); }",
    );
}

#[test]
fn c62a_ord_operator_dispatch_uses_the_user_impl() {
    // Adversarial: `cmp` is REVERSED, so `1 > 2` must hold.
    agree(
        "adversarial_ord",
        "struct Rev { v: Int32 }\n\
         impl Eq for Rev { fn eq(&self, other: &Rev) -> Bool { self.v == other.v } }\n\
         impl Ord for Rev { fn cmp(&self, other: &Rev) -> Ordering { if self.v < other.v { Ordering::Greater } else { Ordering::Less } } }\n\
         fn main() { let a = Rev { v: 1 }; let b = Rev { v: 2 }; assert(a > b); }",
    );
}

// ------------------------------------------------------------- cross-package --

#[test]
fn c62a_cross_package_trait_method_call() {
    let root = std::env::temp_dir().join(format!("stark_c6_2_xpkg_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for (name, dep, src) in [
        (
            "app",
            "model",
            "use model::Sq;\nuse model::Shape;\nfn main() { let q = Sq { s: 5 }; assert_eq(q.area(), 25); }",
        ),
        (
            "model",
            "",
            "pub trait Shape { fn area(&self) -> Int32; }\npub struct Sq { pub s: Int32 }\n\
             impl Shape for Sq { fn area(&self) -> Int32 { self.s * self.s } }\n",
        ),
    ] {
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
    assert!(pd.is_empty(), "parse: {pd:?}");
    let src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        src,
    ));
    let (hir, rd) = resolve(&ast, root_file.clone());
    assert!(rd.is_empty(), "resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, root_file.clone());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "typecheck: {errs:?}");
    let program = lower_program(&hir, &checked.tables, root_file)
        .unwrap_or_else(|e| panic!("lower: {}", e.what));

    assert_reference_identity_matches_bodies("xpkg", &program);
    linkage::build(&program).expect("cross-package trait call must link");

    if rustc_available() {
        let verified = verify_program(&program).expect("verify");
        let out = root.join("out");
        let artifact = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: out,
                target_contract: "stark-64-v1".to_string(),
            },
        )
        .expect("cross-package trait call must build");
        let run = std::process::Command::new(&artifact.binary_path)
            .output()
            .unwrap();
        assert!(run.status.success(), "cross-package trait call must exit 0");
    }
    let _ = std::fs::remove_dir_all(&root);
}

// ------------------------------------------------- the check is NOT weakened --

#[test]
fn a_mismatched_item_is_still_rejected() {
    // The linkage consistency check must keep firing: a reference carrying a different `item` than
    // the body that defines the symbol is still refused before rustc.
    fn info() -> SourceInfo {
        SourceInfo {
            file: FileId(0),
            span: Span::new(0, 0),
            origin: Origin::UserCode,
        }
    }
    fn body(symbol: &str, item: u32, blocks: Vec<BasicBlock>) -> MirBody {
        MirBody {
            instance: Instance {
                item: ItemId(item),
                type_args: Vec::new(),
                symbol: symbol.to_string(),
            },
            params: Vec::new(),
            ret: MirTy::Unit,
            locals: Vec::new(),
            blocks,
            entry: BlockId(0),
        }
    }
    let main = body(
        "main@[]",
        0,
        vec![
            BasicBlock {
                statements: Vec::new(),
                terminator: (
                    Terminator::Call {
                        // referenced with item 1 …
                        callee: Callee::Instance(Instance {
                            item: ItemId(1),
                            type_args: Vec::new(),
                            symbol: "z_callee@[]".to_string(),
                        }),
                        args: Vec::new(),
                        dest: Place::local(LocalId(0)),
                        target: BlockId(1),
                    },
                    info(),
                ),
            },
            BasicBlock {
                statements: Vec::new(),
                terminator: (Terminator::Return, info()),
            },
        ],
    );
    // … but defined with item 2.
    let callee = body(
        "z_callee@[]",
        2,
        vec![BasicBlock {
            statements: Vec::new(),
            terminator: (Terminator::Return, info()),
        }],
    );
    let program = MirProgram {
        files: Vec::new(),
        bodies: vec![main, callee],
        types: TypeContext::default(),
        mir_version: "test".to_string(),
        runtime_surface: "test".to_string(),
    };
    let msg = match linkage::build(&program) {
        Ok(_) => panic!("a mismatched item must still be refused by the linkage check"),
        Err(e) => format!("{e:?}"),
    };
    assert!(msg.contains("two identities"), "{msg}");
}

fn _unused(_: &Path) {}
