//! **Campaign A final audit, §10-F — deleting checker metadata must fail loudly.**
//!
//! Every wired boundary reads a checker-published table. The claim is that absence of that
//! metadata for a reachable language construct is an `InternalInvariant`, never a skip. That claim
//! is only testable by *removing* an entry: a table that is always populated proves nothing about
//! what happens when it is not, and "missing metadata means skip validation" is the single pattern
//! this campaign has spent most of its findings deleting.
//!
//! These mutations touch the TABLES, not the interpreter — the published metadata is exactly what
//! the boundaries claim to depend on, so removing it is the sharpest available test of that
//! dependency. A program that still produces the right answer with its metadata deleted was never
//! really reading it.

use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck::{self, TypeTables};
use std::sync::Arc;

struct Program {
    hir: starkc::hir::Hir,
    file: Arc<SourceFile>,
    tables: TypeTables,
}

fn compile(source: &str) -> Program {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "parse: {parse_diags:?}");
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {resolve_diags:?}");
    let checked = typecheck::analyze(&hir);
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "typecheck: {errors:?}");
    Program {
        hir,
        file,
        tables: checked.tables,
    }
}

impl Program {
    fn run(&self) -> interp::ExecutionOutcome {
        interp::run_capturing(
            &self.hir,
            self.hir.source_named(&self.file.name).expect("registered"),
            &self.tables,
        )
    }
}

/// Compile `source`, assert it runs clean, then apply `remove` to the tables and require the run
/// to fail as a compiler defect. The clean run matters: a mutation "detected" on an
/// already-failing program proves nothing.
fn removing_metadata_must_fail(
    what: &str,
    source: &str,
    remove: impl FnOnce(&mut TypeTables) -> bool,
) -> starkc::interp::RuntimeError {
    let mut program = compile(source);
    let clean = program.run();
    assert!(
        clean.result.is_ok(),
        "{what}: the witness must run clean before its metadata is removed: {:?}",
        clean.result
    );

    assert!(
        remove(&mut program.tables),
        "{what}: nothing was removed, so this control tests nothing — the entry the boundary is \
         claimed to read was not where this test looked for it"
    );

    let error = program.run().result.err().unwrap_or_else(|| {
        panic!(
            "{what}: the metadata was deleted and the program still ran — \
                                   the boundary that claims to read it does not"
        )
    });
    assert_eq!(
        error.class,
        starkc::interp::FailureClass::InternalInvariant,
        "{what}: absent compiler metadata is a compiler defect, never a language trap: {}",
        error.message
    );
    error
}

/// A body's published signature backs `Receiver`, `Parameter`, `Return` and `Propagation`.
#[test]
fn removing_a_callable_signature_fails_loudly() {
    removing_metadata_must_fail(
        "callable_types",
        "fn add(a: Int32, b: Int32) -> Int32 { a + b } fn main() { println(add(1, 2)); }",
        |tables| {
            let key = *tables
                .callable_types
                .keys()
                .max_by_key(|body| body.0)
                .expect("some body has a signature");
            tables.callable_types.remove(&key).is_some()
        },
    );
}

/// `expr_types` backs `ExpressionResult`, `Assignment`, `FieldWrite` and `ElementWrite`.
#[test]
fn removing_an_expression_type_fails_loudly() {
    removing_metadata_must_fail(
        "expr_types",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(7); println(v.len()); }",
        |tables| {
            let before = tables.expr_types.len();
            tables.expr_types.clear();
            before > 0
        },
    );
}

/// `local_types` backs `LetBinding`, `MatchBinding` and `LoopBinding` — the funnel that until this
/// audit's predecessor inherited a permissive lookup.
#[test]
fn removing_a_local_type_fails_loudly() {
    removing_metadata_must_fail(
        "local_types",
        "fn main() { let n = 41; println(n + 1); }",
        |tables| {
            let before = tables.local_types.len();
            tables.local_types.clear();
            before > 0
        },
    );
}

/// `aggregate_field_types` backs `AggregateField`, and is the newest of the published tables.
#[test]
fn removing_aggregate_field_types_fails_loudly() {
    removing_metadata_must_fail(
        "aggregate_field_types",
        "struct Pair { a: Int32, b: Int32 } \
         fn main() { let p = Pair { a: 1, b: 2 }; println(p.a + p.b); }",
        |tables| {
            let before = tables.aggregate_field_types.len();
            tables.aggregate_field_types.clear();
            before > 0
        },
    );
}

/// **The one this audit was most suspicious of.** `push_callable_env` treats an absent
/// `callable_instantiations` entry as "nothing to push" — which is correct for a non-generic call
/// and is DEV-197 for a generic one. The witness is generic and its answer depends on `T`, so if
/// absence were silently tolerated the program would either produce the wrong answer or run with
/// `T` unbound.
#[test]
fn removing_a_generic_instantiation_fails_loudly() {
    removing_metadata_must_fail(
        "callable_instantiations",
        "fn width<T>(x: T) -> Int32 { size_of::<T>() as Int32 } \
         fn main() { println(width(1.5)); }",
        |tables| {
            let before = tables.callable_instantiations.len();
            tables.callable_instantiations.clear();
            before > 0
        },
    );
}
