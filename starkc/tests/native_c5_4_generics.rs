//! WP-C5.4b — generated-source structural proofs for concrete generic instances.
//!
//! `three_engine_differential.rs` proves generic bodies COMPUTE correctly at multiple type
//! arguments. This file proves the structure a value comparison structurally cannot see (§13.2):
//!   * distinct type arguments of one generic item become distinct concrete Rust definitions
//!     (not deduplicated by item — mutation guard §13.5 #2);
//!   * one concrete instance reached through several paths is emitted exactly once (§10.3/§10.4 #5);
//!   * NO Rust semantic generic (`fn name<...>`) ever represents a STARK generic (§10.2, mutation
//!     guard §13.5 #9).
//!
//! Structure only: these build generated source directly through `emit_program::emit` (no rustc),
//! since the claims are about emitted text, and value/behaviour is the three-engine harness's job.

use starkc::backend::generated_rust::{emit_program, mangle};
use starkc::backend::version::build_versions;
use starkc::diag::Severity;
use starkc::layout::TargetLayout;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::mir::MirProgram;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn compile(source: &str, tag: &str) -> (MirProgram, String) {
    let file = Arc::new(SourceFile::new(
        format!("c5_4b_{tag}.stark"),
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
    let program = lower_program(&hir, &checked.tables, file)
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    // Verify for well-formedness (and to keep the structural test on the same precondition as a
    // real build), then emit source directly — no cargo/rustc needed for a structural claim.
    let _verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let versions = build_versions("0.0.0-test".to_string(), "test-triple".to_string());
    let layout = TargetLayout::default();
    let generated = emit_program::emit(&program, &versions, &layout)
        .unwrap_or_else(|e| panic!("{tag} emit: {e:?}"))
        .main_rs;
    (program, generated)
}

/// Count non-overlapping occurrences of `needle` in `hay`.
fn count(hay: &str, needle: &str) -> usize {
    hay.matches(needle).count()
}

/// §10.3/§10.4 #5: every body in the verified set is defined exactly once under its canonical
/// generated name. This is the "exactly once" claim stated over the whole program, so it catches
/// both a dropped body and a doubly-emitted one.
fn assert_each_body_defined_once(program: &MirProgram, generated: &str) {
    for body in &program.bodies {
        let name = mangle::function_name_for_symbol(&body.instance.symbol);
        let defs = count(generated, &format!("fn {name}("));
        assert_eq!(
            defs, 1,
            "body `{}` (Rust `{name}`) should be defined exactly once, found {defs}",
            body.instance.symbol
        );
    }
}

/// Mutation guard §13.5 #9: no generated function may carry a Rust generic parameter list. Scans
/// every `fn ` definition and asserts there is no `<` between the `fn` keyword and its `(` (a `<`
/// in a return type, which comes after `(`, is fine — e.g. `-> ValueSlot<...>`).
fn assert_no_rust_generic_definitions(generated: &str) {
    for line in generated.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("fn ") {
            let header = rest.split('(').next().unwrap_or(rest);
            assert!(
                !header.contains('<'),
                "generated fn carries a Rust generic parameter list: {line:?}"
            );
        }
    }
}

#[test]
fn distinct_type_arguments_emit_distinct_concrete_definitions() {
    let (program, generated) = compile(
        r#"fn identity<T>(x: T) -> T { x }
fn main() {
    let a: Int32 = 21;
    let b: Int64 = 9000000000;
    assert_eq(identity(a), 21);
    assert_eq(identity(b), 9000000000);
}
"#,
        "identity_two_types",
    );

    // Two concrete instances of one generic item: same `item`, different `type_args`. A backend
    // deduplicating by item alone would find one.
    let identities: Vec<_> = program
        .bodies
        .iter()
        .filter(|b| b.instance.symbol.contains("identity"))
        .collect();
    assert_eq!(
        identities.len(),
        2,
        "expected two concrete identity instances, found: {:?}",
        identities
            .iter()
            .map(|b| &b.instance.symbol)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        identities[0].instance.item, identities[1].instance.item,
        "same generic item"
    );
    assert_ne!(
        identities[0].instance.type_args, identities[1].instance.type_args,
        "distinct type arguments"
    );

    // Each is a distinct concrete Rust definition, emitted once; no Rust `identity` fn at all
    // (names are sanitized canonical symbols), and no Rust generic anywhere.
    assert_each_body_defined_once(&program, &generated);
    assert_no_rust_generic_definitions(&generated);
    assert!(
        !generated.contains("fn identity"),
        "a STARK generic must not appear as a Rust fn named `identity`"
    );
}

#[test]
fn one_instance_reached_by_two_paths_is_emitted_once() {
    // `pick(1)` and `pick(2)` are both `pick@[Int32]` — one instance, two call paths.
    let (program, generated) = compile(
        r#"fn pick<T>(x: T) -> T { x }
fn via_g() -> Int32 { pick(1) }
fn via_h() -> Int32 { pick(2) }
fn main() {
    assert_eq(via_g() + via_h(), 3);
}
"#,
        "pick_shared",
    );

    let picks: Vec<_> = program
        .bodies
        .iter()
        .filter(|b| b.instance.symbol.contains("pick"))
        .collect();
    assert_eq!(
        picks.len(),
        1,
        "one shared instance, not one per path: {:?}",
        picks.iter().map(|b| &b.instance.symbol).collect::<Vec<_>>()
    );

    let name = mangle::function_name_for_symbol(&picks[0].instance.symbol);
    assert_eq!(
        count(&generated, &format!("fn {name}(")),
        1,
        "the shared instance must be defined exactly once"
    );
    assert_each_body_defined_once(&program, &generated);
}

#[test]
fn a_recursive_generic_instance_is_one_self_calling_definition() {
    // `depth@[Int32]` calls itself; it must be one definition, not an unrolled chain, and it must
    // reference its own generated name inside its body (the recursive call).
    let (program, generated) = compile(
        r#"fn depth<T>(x: T, n: Int32) -> Int32 {
    if n <= 0 { 0 } else { depth::<T>(x, n - 1) + 1 }
}
fn main() {
    let seed: Int32 = 3;
    assert_eq(depth::<Int32>(seed, 4), 4);
}
"#,
        "depth_recursive",
    );

    let depths: Vec<_> = program
        .bodies
        .iter()
        .filter(|b| b.instance.symbol.contains("depth"))
        .collect();
    assert_eq!(depths.len(), 1, "one recursive instance");
    let name = mangle::function_name_for_symbol(&depths[0].instance.symbol);
    // Exactly one definition; and the name appears more than once overall (definition + the
    // recursive call site), proving the recursion targets the same concrete instance.
    assert_eq!(count(&generated, &format!("fn {name}(")), 1);
    assert!(
        count(&generated, &name) >= 2,
        "recursive call should reference the same generated name as the definition"
    );
    assert_each_body_defined_once(&program, &generated);
    assert_no_rust_generic_definitions(&generated);
}
