//! AS3 Boundary 1 — the keying model, proven before any consumer migrates.
//!
//! `WP-CALLABLE-USE-TOTAL.md` §3.2 was rewritten after owner review rejected the first draft's
//! assumption that one `ExprId` is one callable use. It is not:
//!
//! - `display_deep` recurses through tuples, arrays, `Option`, `Result` and slots, reaching a
//!   nominal's `Display::fmt` at any depth, so `println((a, b))` is **one argument expression and
//!   two `fmt` use sites**, and `println(vec)` is one use site executed once per element;
//! - `language_equal` dispatches `Eq::eq` from collection lookup, reached with runtime values and a
//!   span rather than a unique invoking expression.
//!
//! So a use is a **static semantic use site**: one expression may carry zero, one or many, and a
//! use may execute zero, one or thousands of times. This file pins that shape *now*, because if the
//! keying is wrong every later boundary encodes it.
//!
//! Boundary 1 publishes two families — free calls and function values. The function-value case is
//! here deliberately rather than later: it is the only `CalleeSelection::FunctionValue` +
//! `GenericEnvironment::FromFunctionValue` pair, so the dynamic half of the model is exercised at
//! the start rather than discovered at Boundary 4.

use starkc::options::LanguageOptions;
use starkc::session::CompilerSession;
use starkc::typecheck::{
    CalleeSelection, DispatchProvenance, GenericEnvironment, ReceiverAdjustment, ReceiverBinding,
};
use std::sync::Arc;

fn analyse(source: &str) -> starkc::session::CheckedProgram {
    let file = Arc::new(starkc::source::SourceFile::new("keying.stark", source));
    CompilerSession::for_source(file, LanguageOptions::CORE)
        .check()
        .unwrap_or_else(|failure| panic!("fixture must compile:\n{}", failure.render()))
}

/// The structural claim: the tables can represent zero, one and many uses per expression, and the
/// index and the store agree.
#[test]
fn an_expression_may_carry_zero_one_or_many_uses() {
    let program = analyse(
        "fn one() -> Int32 {\n    1\n}\n\
         fn two(a: Int32) -> Int32 {\n    a\n}\n\
         fn main() {\n    let a: Int32 = one();\n    let b: Int32 = two(one());\n    let c: Int32 = a + b;\n}\n",
    );
    let tables = program.tables();

    // Every id in the index resolves in the store, and nothing is orphaned.
    let mut indexed = 0usize;
    for (expr, ids) in tables.callable_uses_by_expr.iter() {
        assert!(!ids.is_empty(), "expr {expr:?} indexed with an empty list");
        for id in ids {
            assert!(
                (id.0 as usize) < tables.callable_uses.len(),
                "expr {expr:?} names use {id:?}, which the store does not contain"
            );
            indexed += 1;
        }
    }
    assert_eq!(
        indexed,
        tables.callable_uses.len(),
        "every published use must be reachable from exactly one expression"
    );

    // Non-vacuity: this program really does call things.
    assert!(
        tables.callable_uses.len() >= 3,
        "expected at least three calls, found {}",
        tables.callable_uses.len()
    );

    // `let b = two(one())` is two calls in two expressions — the ordinary one-to-one case, which
    // the model must still express now that it can express many-to-one.
    let single = tables
        .callable_uses_by_expr
        .values()
        .filter(|ids: &&Vec<starkc::typecheck::CallableUseId>| ids.len() == 1)
        .count();
    assert!(single >= 3, "expected several single-use expressions");
}

/// A free call publishes a `Static` selection with a real declaration and an **explicitly empty**
/// environment — not an absent one.
#[test]
fn a_free_call_publishes_a_static_selection_with_an_empty_environment() {
    let program = analyse("fn f() -> Int32 {\n    7\n}\nfn main() {\n    let v: Int32 = f();\n}\n");
    let tables = program.tables();

    let statics: Vec<_> = tables
        .callable_uses
        .iter()
        .filter(|u| matches!(u.selection, CalleeSelection::Static { .. }))
        .collect();
    assert_eq!(statics.len(), 1, "one call to `f`, one static use");

    let use_ = statics[0];
    assert_eq!(use_.provenance, DispatchProvenance::Direct);
    assert_eq!(use_.receiver_adjustment, ReceiverAdjustment::None);
    assert_eq!(use_.receiver_binding, ReceiverBinding::None);
    match &use_.environment {
        GenericEnvironment::Static(bindings) => assert!(
            bindings.is_empty(),
            "a non-generic call binds nothing, and says so explicitly: {bindings:?}"
        ),
        other => panic!("a free call has a static environment, got {other:?}"),
    }
}

/// A generic free call publishes its selected environment, not an empty one.
#[test]
fn a_generic_free_call_publishes_its_environment() {
    let program = analyse(
        "fn identity<T>(value: T) -> T {\n    value\n}\n\
         fn main() {\n    let a: Int32 = identity(1);\n    let b: Bool = identity(true);\n}\n",
    );
    let tables = program.tables();

    let environments: Vec<usize> = tables
        .callable_uses
        .iter()
        .filter_map(|u| match &u.environment {
            GenericEnvironment::Static(b) if !b.is_empty() => Some(b.len()),
            _ => None,
        })
        .collect();
    assert_eq!(
        environments.len(),
        2,
        "two instantiations of `identity` publish two non-empty environments, got {environments:?}"
    );

    // The two uses bind DIFFERENT types — the reason an environment cannot be keyed by body.
    let bound: Vec<String> = tables
        .callable_uses
        .iter()
        .filter_map(|u| match &u.environment {
            GenericEnvironment::Static(b) => b.first().map(|(_, ty)| format!("{ty:?}")),
            _ => None,
        })
        .collect();
    assert_eq!(bound.len(), 2);
    assert_ne!(
        bound[0], bound[1],
        "one generic body invoked at two types must publish two environments: {bound:?}"
    );
}

/// The dynamic half: calling through a function value publishes `FunctionValue` on **both** axes.
///
/// DEV-178 is why. The value carries the item and the bindings it was created with; the call site's
/// `Ty::Fn` cannot reconstruct which instantiation produced it. A model that demanded a `BlockId`
/// here would have to invent one.
#[test]
fn a_function_value_call_defers_both_selection_and_environment() {
    let program = analyse(
        "fn double(n: Int32) -> Int32 {\n    n * 2\n}\n\
         fn main() {\n    let f: fn(Int32) -> Int32 = double;\n    let v: Int32 = f(21);\n}\n",
    );
    let tables = program.tables();

    let dynamic: Vec<_> = tables
        .callable_uses
        .iter()
        .filter(|u| matches!(u.selection, CalleeSelection::FunctionValue))
        .collect();
    assert_eq!(dynamic.len(), 1, "one call through a function value");

    let use_ = dynamic[0];
    assert_eq!(use_.provenance, DispatchProvenance::FunctionValue);
    assert_eq!(
        use_.environment,
        GenericEnvironment::FromFunctionValue,
        "the environment comes from the value, not from this site (DEV-178)"
    );
    assert_eq!(
        use_.signature.params.len(),
        1,
        "the checker still knows the signature even when it does not know the body"
    );
}

/// §3.4's invariant: a use's instantiated signature must agree with the body's parametric one, so
/// `callable_uses` and `callable_types` cannot become competing signature authorities.
#[test]
fn a_static_uses_signature_agrees_with_its_bodys_arity() {
    let program = analyse(
        "fn add(a: Int32, b: Int32) -> Int32 {\n    a + b\n}\n\
         fn none() {}\n\
         fn main() {\n    let v: Int32 = add(1, 2);\n    none();\n}\n",
    );
    let tables = program.tables();

    let mut checked = 0usize;
    for use_ in &tables.callable_uses {
        let CalleeSelection::Static { body, .. } = &use_.selection else {
            continue;
        };
        let Some(body_sig) = tables.callable_types.get(body) else {
            panic!("a static use names body {body:?}, which callable_types does not carry");
        };
        assert_eq!(
            use_.signature.params.len(),
            body_sig.params.len(),
            "the use's signature and the body's disagree about arity — two signature authorities"
        );
        assert_eq!(
            use_.signature.receiver.is_some(),
            body_sig.receiver.is_some(),
            "the use's signature and the body's disagree about a receiver"
        );
        checked += 1;
    }
    assert!(
        checked >= 2,
        "only {checked} static uses cross-checked; the invariant is not being exercised"
    );
}
