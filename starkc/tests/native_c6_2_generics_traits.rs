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

// ============================================================ WP-C6.2b / DEV-102 ==
//
// Fully qualified trait calls. 03-Type-System TYPE-METHOD-001: "Fully qualified
// `Trait::method(receiver, ...)` bypasses trait-name lookup but still requires a unique coherent
// impl", and "Trait methods can always be called in fully-qualified function form". Before C6.2b
// the front end and the HIR oracle accepted these while MIR lowering refused them
// (`LOWER: callee form (C4.5)`), so no such program could be built natively.
//
// The disambiguation pair below is the load-bearing case: it is the spec's own remedy for the
// §18 ambiguity error (E0203), so `A::go` and `B::go` MUST select different impls.

#[test]
fn c62b_fully_qualified_trait_call() {
    agree(
        "fq_user_trait",
        "trait Shape { fn area(&self) -> Int32; }\nstruct Sq { s: Int32 }\n\
         impl Shape for Sq { fn area(&self) -> Int32 { self.s * self.s } }\n\
         fn main() { let q = Sq { s: 3 }; assert_eq(Shape::area(&q), 9); }",
    );
}

#[test]
fn c62b_fully_qualified_selects_the_named_trait() {
    // Both traits supply `go`; `s.go()` is E0203. Each qualified form must pick its own impl.
    const DECLS: &str = "trait A { fn go(&self) -> Int32; }\ntrait B { fn go(&self) -> Int32; }\n\
         struct S { v: Int32 }\n\
         impl A for S { fn go(&self) -> Int32 { 1 } }\n\
         impl B for S { fn go(&self) -> Int32 { 2 } }\n";
    agree(
        "fq_pick_a",
        &format!("{DECLS}fn main() {{ let s = S {{ v: 0 }}; assert_eq(A::go(&s), 1); }}"),
    );
    agree(
        "fq_pick_b",
        &format!("{DECLS}fn main() {{ let s = S {{ v: 0 }}; assert_eq(B::go(&s), 2); }}"),
    );
}

#[test]
fn c62b_fully_qualified_ignores_an_inherent_method_of_the_same_name() {
    // `s.go()` prefers the inherent method (TYPE-METHOD-001); `A::go(&s)` must not.
    agree(
        "fq_ignores_inherent",
        "trait A { fn go(&self) -> Int32; }\nstruct S { v: Int32 }\n\
         impl A for S { fn go(&self) -> Int32 { 2 } }\nimpl S { fn go(&self) -> Int32 { 1 } }\n\
         fn main() { let s = S { v: 0 }; assert_eq(A::go(&s), 2); assert_eq(s.go(), 1); }",
    );
}

#[test]
fn c62b_fully_qualified_reaches_a_trait_default_body() {
    agree(
        "fq_default_body",
        "trait G { fn base(&self) -> Int32; fn twice(&self) -> Int32 { self.base() * 2 } }\n\
         struct S { v: Int32 }\nimpl G for S { fn base(&self) -> Int32 { self.v } }\n\
         fn main() { let s = S { v: 5 }; assert_eq(G::twice(&s), 10); }",
    );
}

#[test]
fn c62b_fully_qualified_passes_further_arguments_and_a_mut_receiver() {
    agree(
        "fq_with_args",
        "trait Add2 { fn add(&self, o: Int32) -> Int32; }\nstruct S { v: Int32 }\n\
         impl Add2 for S { fn add(&self, o: Int32) -> Int32 { self.v + o } }\n\
         fn main() { let s = S { v: 5 }; assert_eq(Add2::add(&s, 3), 8); }",
    );
    agree(
        "fq_mut_receiver",
        "trait Bump { fn bump(&mut self); fn get(&self) -> Int32; }\nstruct S { v: Int32 }\n\
         impl Bump for S { fn bump(&mut self) { self.v = self.v + 1; } fn get(&self) -> Int32 { self.v } }\n\
         fn main() { let mut s = S { v: 1 }; Bump::bump(&mut s); assert_eq(Bump::get(&s), 2); }",
    );
}

#[test]
fn c62b_fully_qualified_on_a_drop_bearing_receiver() {
    agree(
        "fq_drop_receiver",
        "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) { } }\n\
         trait Take { fn peek(&self) -> Int32; }\nimpl Take for D { fn peek(&self) -> Int32 { self.v } }\n\
         fn main() { let d = D { v: 4 }; assert_eq(Take::peek(&d), 4); }",
    );
}

#[test]
fn c62b_ambiguous_unqualified_call_is_still_rejected() {
    // The qualified form is the spec's remedy for this error, so the error must remain.
    let src = "trait A { fn go(&self) -> Int32; }\ntrait B { fn go(&self) -> Int32; }\n\
         struct S { v: Int32 }\n\
         impl A for S { fn go(&self) -> Int32 { 1 } }\nimpl B for S { fn go(&self) -> Int32 { 2 } }\n\
         fn main() { let s = S { v: 0 }; assert_eq(s.go(), 1); }";
    let file = Arc::new(SourceFile::new("ambig.stark".to_string(), src.to_string()));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file);
    let codes: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .filter_map(|d| d.code.clone())
        .collect();
    assert!(
        codes.iter().any(|c| c == "E0203"),
        "expected E0203, got {codes:?}"
    );
}

#[test]
fn c62b_receiverless_qualified_call_is_rejected_by_the_checker() {
    // `Mk::make()` has no receiver argument, so the implementing type is unrecoverable. The
    // checker refuses it (E0005) — lowering's matching guard is defensive, never reached here.
    let src = "trait Mk { fn make() -> Self; fn val(&self) -> Int32; }\nstruct G { v: Int32 }\n\
         impl Mk for G { fn make() -> Self { G { v: 8 } } fn val(&self) -> Int32 { self.v } }\n\
         fn main() { let g = Mk::make(); assert_eq(g.val(), 8); }";
    let file = Arc::new(SourceFile::new(
        "recvless.stark".to_string(),
        src.to_string(),
    ));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file);
    let codes: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .filter_map(|d| d.code.clone())
        .collect();
    assert!(
        codes.iter().any(|c| c == "E0005"),
        "expected E0005, got {codes:?}"
    );
}
