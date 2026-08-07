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

/// §3.4's invariant, on the COMPLETE signature, for **every** static use including generic ones.
///
/// The first version compared only arity and receiver presence. The second compared full signatures
/// but **skipped every generic use** — while the commit message said it compared them. That is the
/// area DEV-176 and A3c-S came from, so skipping it left the invariant unproved exactly where it
/// had failed before.
///
/// The substitution is the compiler's own `substitute_ty`, applied to the same name→type view
/// `CallableInstantiation` publishes. Writing a second instantiation algorithm to check the first
/// would be the defect this packet removes, in a test.
#[test]
fn a_static_uses_signature_equals_its_substituted_body_signature() {
    let program = analyse(
        "struct Wrapper<T> { value: T }\n\
         impl<T> Wrapper<T> {\n\
         \x20   pub fn peek(&self) -> Int32 {\n        1\n    }\n\
         \x20   pub fn swap<U>(&self, other: U) -> U {\n        other\n    }\n\
         \x20   pub fn take(self) -> Int32 {\n        2\n    }\n}\n\
         struct Plain { v: Int32 }\n\
         impl Plain {\n    pub fn get(&self) -> Int32 {\n        self.v\n    }\n}\n\
         fn identity<T>(value: T) -> T {\n    value\n}\n\
         fn add(a: Int32, b: Int32) -> Int32 {\n    a + b\n}\n\
         fn main() {\n\
         \x20   let w: Wrapper<Int32> = Wrapper { value: 5 };\n\
         \x20   let a: Int32 = w.peek();\n\
         \x20   let b: Bool = w.swap(true);\n\
         \x20   let p: Plain = Plain { v: 1 };\n\
         \x20   let c: Int32 = p.get();\n\
         \x20   let d: Int32 = identity(3);\n\
         \x20   let e: Int32 = add(1, 2);\n\
         \x20   let f: Int32 = w.take();\n}\n",
    );
    let tables = program.tables();

    let mut checked = 0usize;
    let mut generic_checked = 0usize;
    let mut with_receiver = 0usize;
    for use_ in &tables.callable_uses {
        let CalleeSelection::Static { body, .. } = &use_.selection else {
            continue;
        };
        let Some(body_sig) = tables.callable_types.get(body) else {
            panic!("static use names body {body:?}, which callable_types does not carry");
        };

        // The compiler's own substitution, over the environment this use published.
        let subs = use_.environment.substitutions();
        let expected_receiver = body_sig
            .receiver
            .as_ref()
            .map(|ty| starkc::typecheck::substitute_ty(ty, &subs));
        let expected_params: Vec<_> = body_sig
            .params
            .iter()
            .map(|ty| starkc::typecheck::substitute_ty(ty, &subs))
            .collect();
        let expected_ret = starkc::typecheck::substitute_ty(&body_sig.ret, &subs);

        assert_eq!(
            format!("{:?}", use_.signature.receiver),
            format!("{expected_receiver:?}"),
            "receiver differs from the substituted body signature for {use_:?}"
        );
        assert_eq!(
            format!("{:?}", use_.signature.params),
            format!("{expected_params:?}"),
            "parameters differ from the substituted body signature for {use_:?}"
        );
        assert_eq!(
            format!("{:?}", use_.signature.ret),
            format!("{expected_ret:?}"),
            "result differs from the substituted body signature for {use_:?}"
        );

        if use_.signature.receiver.is_some() {
            with_receiver += 1;
        }
        if !subs.is_empty() {
            generic_checked += 1;
        }
        checked += 1;
    }

    assert!(
        checked >= 6,
        "only {checked} static uses cross-checked; the invariant is barely exercised"
    );
    assert!(
        with_receiver >= 4,
        "only {with_receiver} uses carried a receiver — the method half is not being tested, \
         which is how the receiver gap survived"
    );
    // The load-bearing non-vacuity guard. If this is zero the test has silently regressed to the
    // non-generic-only version it replaced, which is the failure being repaired.
    assert!(
        generic_checked >= 3,
        "only {generic_checked} GENERIC uses cross-checked — a `Wrapper<T>` method, a method-own \
         generic and a generic free function must all be present, or the invariant is unproved \
         exactly where DEV-176 and A3c-S came from"
    );
}

// ---------------------------------------------------------------------------------------------
// Boundary 2 — named dispatch: methods, associated functions, qualified calls, trait defaults
// ---------------------------------------------------------------------------------------------

use starkc::typecheck::CallableDeclId;

/// A method call publishes an `ImplMember` declaration — built from ids the HIR possesses.
///
/// The first design draft assumed methods have their own `ItemId`. They do not: `ImplItem::Fn`
/// embeds a `FnDef` positionally inside the impl's `items`, which is why A3b chose `BlockId` for
/// executable identity. This asserts the corrected model rather than the assumed one.
#[test]
fn a_method_call_publishes_an_impl_member_declaration() {
    let program = analyse(
        "struct Counter { value: Int32 }\n\
         impl Counter {\n    pub fn get(&self) -> Int32 {\n        self.value\n    }\n}\n\
         fn main() {\n    let c: Counter = Counter { value: 3 };\n    let v: Int32 = c.get();\n}\n",
    );
    let tables = program.tables();

    let members: Vec<_> = tables
        .callable_uses
        .iter()
        .filter(|u| {
            matches!(
                u.selection,
                CalleeSelection::Static {
                    declaration: CallableDeclId::ImplMember { .. },
                    ..
                }
            )
        })
        .collect();
    assert_eq!(members.len(), 1, "one method call, one impl-member use");
    assert_eq!(
        members[0].receiver_binding,
        ReceiverBinding::Shared,
        "`&self` binds a shared receiver, and the binding is published separately from the \
         call site's adjustment"
    );
}

/// `&mut self` and `self` publish different bindings — the field is carrying information, not a
/// constant.
#[test]
fn the_receiver_binding_distinguishes_self_forms() {
    let program = analyse(
        "struct Cell { value: Int32 }\n\
         impl Cell {\n\
         \x20   pub fn read(&self) -> Int32 {\n        self.value\n    }\n\
         \x20   pub fn bump(&mut self) {\n        self.value = self.value + 1;\n    }\n\
         \x20   pub fn consume(self) -> Int32 {\n        self.value\n    }\n}\n\
         fn main() {\n\
         \x20   let mut c: Cell = Cell { value: 1 };\n\
         \x20   c.bump();\n\
         \x20   let a: Int32 = c.read();\n\
         \x20   let b: Int32 = c.consume();\n}\n",
    );
    let tables = program.tables();

    let mut bindings: Vec<ReceiverBinding> = tables
        .callable_uses
        .iter()
        .filter(|u| {
            matches!(
                u.selection,
                CalleeSelection::Static {
                    declaration: CallableDeclId::ImplMember { .. },
                    ..
                }
            )
        })
        .map(|u| u.receiver_binding)
        .collect();
    bindings.sort_by_key(|b| format!("{b:?}"));
    assert_eq!(
        bindings,
        vec![
            ReceiverBinding::ByValue,
            ReceiverBinding::Exclusive,
            ReceiverBinding::Shared
        ],
        "three self forms must publish three distinct bindings"
    );
}

/// Every published static use names a declaration the HIR can produce, over a program that
/// exercises free calls, methods and associated functions together.
#[test]
fn every_static_use_names_a_real_declaration() {
    let program = analyse(
        "struct Point { x: Int32 }\n\
         impl Point {\n\
         \x20   pub fn origin() -> Point {\n        Point { x: 0 }\n    }\n\
         \x20   pub fn x(&self) -> Int32 {\n        self.x\n    }\n}\n\
         fn helper(n: Int32) -> Int32 {\n    n\n}\n\
         fn main() {\n\
         \x20   let p: Point = Point::origin();\n\
         \x20   let a: Int32 = p.x();\n\
         \x20   let b: Int32 = helper(a);\n}\n",
    );
    let tables = program.tables();

    let mut kinds = std::collections::BTreeSet::new();
    let mut statics = 0usize;
    for use_ in &tables.callable_uses {
        if let CalleeSelection::Static { declaration, .. } = &use_.selection {
            statics += 1;
            kinds.insert(match declaration {
                CallableDeclId::Item(_) => "Item",
                CallableDeclId::ImplMember { .. } => "ImplMember",
                CallableDeclId::TraitMember { .. } => "TraitMember",
            });
        }
    }
    assert!(
        statics >= 3,
        "expected at least three static uses, found {statics}"
    );
    assert!(
        kinds.contains("Item") && kinds.contains("ImplMember"),
        "the fixture must exercise more than one declaration kind, got {kinds:?}"
    );
}

/// TYPE-METHOD-002's auto-dereference, published rather than discarded.
///
/// Method resolution repeatedly removes leading `&`/`&mut` before matching a receiver form. That
/// peel count is a decision the CALL SITE made, and every named-dispatch publication originally
/// passed `ReceiverAdjustment::None` — so a consumer would have received `binding = Shared,
/// adjustment = None` and still had to reconstruct the receiver semantics itself, which is exactly
/// what AS3 removes.
///
/// The six cases the hardening review named.
#[test]
fn the_receiver_adjustment_publishes_the_deref_count() {
    let program = analyse(
        "struct T2 { v: Int32 }\n\
         impl T2 {\n\
         \x20   pub fn shared(&self) -> Int32 {\n        self.v\n    }\n\
         \x20   pub fn exclusive(&mut self) {\n        self.v = self.v + 1;\n    }\n\
         \x20   pub fn owned(self) -> Int32 {\n        self.v\n    }\n}\n\
         fn via_ref(t: &T2) -> Int32 {\n    t.shared()\n}\n\
         fn via_ref_ref(t: &&T2) -> Int32 {\n    t.shared()\n}\n\
         fn main() {\n\
         \x20   let mut a: T2 = T2 { v: 1 };\n\
         \x20   let direct: Int32 = a.shared();\n\
         \x20   a.exclusive();\n\
         \x20   let r: Int32 = via_ref(&a);\n\
         \x20   let b: T2 = T2 { v: 2 };\n\
         \x20   let rr: Int32 = via_ref_ref(&&b);\n\
         \x20   let o: Int32 = b.owned();\n}\n",
    );
    let tables = program.tables();

    let mut observed: Vec<(ReceiverBinding, ReceiverAdjustment)> = tables
        .callable_uses
        .iter()
        .filter(|u| {
            matches!(
                u.selection,
                CalleeSelection::Static {
                    declaration: CallableDeclId::ImplMember { .. },
                    ..
                }
            )
        })
        .map(|u| (u.receiver_binding, u.receiver_adjustment))
        .collect();
    observed.sort_by_key(|pair| format!("{pair:?}"));
    observed.dedup();

    // `T -> self` is by value, with no peeling to record.
    assert!(
        observed.contains(&(ReceiverBinding::ByValue, ReceiverAdjustment::ByValue)),
        "an owned receiver must publish a by-value adjustment: {observed:?}"
    );
    // `T -> &self` and `T -> &mut self` peel nothing; the adjustment is the borrow form.
    assert!(
        observed.contains(&(
            ReceiverBinding::Shared,
            ReceiverAdjustment::Shared { derefs: 0 }
        )),
        "a direct shared call must publish zero derefs: {observed:?}"
    );
    assert!(
        observed.contains(&(
            ReceiverBinding::Exclusive,
            ReceiverAdjustment::Exclusive { derefs: 0 }
        )),
        "a direct exclusive call must publish zero derefs: {observed:?}"
    );
    // `&T -> &self` peels one, `&&T -> &self` peels two. The count is the information that was
    // being thrown away.
    assert!(
        observed.contains(&(
            ReceiverBinding::Shared,
            ReceiverAdjustment::Shared { derefs: 1 }
        )),
        "a call through `&T` must publish one deref: {observed:?}"
    );
    assert!(
        observed.contains(&(
            ReceiverBinding::Shared,
            ReceiverAdjustment::Shared { derefs: 2 }
        )),
        "a call through `&&T` must publish two derefs — the case that proves the count is real \
         rather than a constant: {observed:?}"
    );

    // Non-vacuity: the adjustment must not be uniformly `None`, which is the defect this repairs.
    assert!(
        !observed
            .iter()
            .all(|(_, adj)| *adj == ReceiverAdjustment::None),
        "every adjustment was None; the field is publishing nothing"
    );
}

// ---------------------------------------------------------------------------------------------
// Boundary 3 — operator dispatch: equality and ordering
// ---------------------------------------------------------------------------------------------

use starkc::typecheck::DispatchProvenance as DP;

/// `==` and `<` on a user nominal publish a `CoreTrait` use.
///
/// These are dispatched by the LANGUAGE, not by a written call, and both engines currently
/// re-select them with no trait filter in MIR at all. This is the checker stating which body runs.
#[test]
fn operators_on_a_user_nominal_publish_core_trait_uses() {
    let program = analyse(
        "struct Id { v: Int32 }\n\
         impl Eq for Id {\n\
         \x20   fn eq(&self, other: &Id) -> Bool {\n        self.v == other.v\n    }\n}\n\
         impl Ord for Id {\n\
         \x20   fn cmp(&self, other: &Id) -> Ordering {\n        self.v.cmp(&other.v)\n    }\n}\n\
         fn main() {\n\
         \x20   let a: Id = Id { v: 1 };\n\
         \x20   let b: Id = Id { v: 2 };\n\
         \x20   let same: Bool = a == b;\n\
         \x20   let less: Bool = a < b;\n}\n",
    );
    let tables = program.tables();

    let cores: Vec<_> = tables
        .callable_uses
        .iter()
        .filter(|u| matches!(u.provenance, DP::CoreTrait { .. }))
        .collect();
    assert_eq!(
        cores.len(),
        2,
        "one `==` and one `<` on a user nominal are two core-trait uses, got {cores:?}"
    );

    for use_ in &cores {
        assert!(
            matches!(
                use_.selection,
                CalleeSelection::Static {
                    declaration: CallableDeclId::ImplMember { .. },
                    ..
                }
            ),
            "an operator use must name the impl member that runs: {use_:?}"
        );
        assert_eq!(
            use_.receiver_binding,
            ReceiverBinding::Shared,
            "`Eq::eq` and `Ord::cmp` both borrow their receiver"
        );
        assert!(
            use_.signature.receiver.is_some(),
            "the published signature must carry the receiver it read from the declaration"
        );
    }

    // The two must name DIFFERENT bodies — otherwise the publication is not distinguishing the
    // traits, which is the defect DEV-BOUND-TRAIT-IDENTITY was.
    let bodies: Vec<String> = cores.iter().map(|u| format!("{:?}", u.selection)).collect();
    assert_ne!(
        bodies[0], bodies[1],
        "`==` and `<` must select different bodies"
    );
}

/// Operators on primitives publish nothing: they have built-in meaning (DEV-075) and reach no user
/// body, so they are not callable uses. Publishing one would make totality claim something false.
#[test]
fn operators_on_primitives_publish_no_use() {
    let program = analyse(
        "fn main() {\n\
         \x20   let a: Int32 = 1;\n\
         \x20   let b: Int32 = 2;\n\
         \x20   let same: Bool = a == b;\n\
         \x20   let less: Bool = a < b;\n}\n",
    );
    let tables = program.tables();
    let cores = tables
        .callable_uses
        .iter()
        .filter(|u| matches!(u.provenance, DP::CoreTrait { .. }))
        .count();
    assert_eq!(
        cores, 0,
        "primitive operators reach no user body and must publish no core-trait use"
    );
}

/// Boundary 3 consumption: the interpreter runs the body the CHECKER selected for `==` and `<`.
///
/// Behavioural, not structural — it exercises a user `Eq`/`Ord` through the HIR engine and checks
/// the answer. The value is that this is the first AS3 commit where a wrong consumption would
/// change a result rather than merely publish something unread.
#[test]
fn operator_dispatch_runs_the_selected_body() {
    let program = analyse(
        "struct Even { v: Int32 }\n\
         impl Eq for Even {\n\
         \x20   fn eq(&self, other: &Even) -> Bool {\n\
         \x20       (self.v % 2) == (other.v % 2)\n    }\n}\n\
         fn main() {\n\
         \x20   let a: Even = Even { v: 2 };\n\
         \x20   let b: Even = Even { v: 4 };\n\
         \x20   let c: Even = Even { v: 3 };\n\
         \x20   println(a == b);\n\
         \x20   println(a == c);\n}\n",
    );
    let execution = program.execute_hir().expect("the fixture must run");
    assert_eq!(
        execution.output, "true\nfalse\n",
        "the user's `Eq::eq` decides equality — 2 and 4 are equal by parity, 2 and 3 are not"
    );
}

// ---------------------------------------------------------------------------------------------
// Boundary 4 — iteration
// ---------------------------------------------------------------------------------------------

/// A `for` loop over a GENERIC user iterator publishes the impl's generic environment.
///
/// The hardening negative. The first iteration attempt added a second Iterator selector that
/// discarded `match_impl_type`'s substitution and published `Static(vec![])`, so
/// `impl<T> Iterator for Repeat<T>` lost its `T` binding while the element-type calculation kept
/// it. Behaviour was unaffected, so no functional test could see it — this one asserts the
/// environment directly, and fails if it is emptied.
#[test]
fn a_generic_user_iterator_publishes_its_impl_environment() {
    let program = analyse(
        "struct Repeat<T> { value: T, left: Int32 }\n\
         impl<T> Iterator for Repeat<T> {\n\
         \x20   type Item = T;\n\
         \x20   fn next(&mut self) -> Option<T> {\n\
         \x20       if self.left <= 0 {\n            return None;\n        }\n\
         \x20       self.left = self.left - 1;\n\
         \x20       Some(self.value)\n    }\n}\n\
         fn main() {\n\
         \x20   let mut r: Repeat<Int32> = Repeat { value: 7, left: 2 };\n\
         \x20   let mut total: Int32 = 0;\n\
         \x20   for v in r {\n        total = total + v;\n    }\n\
         \x20   println(total);\n}\n",
    );
    let tables = program.tables();

    let iterator_uses: Vec<_> = tables
        .callable_uses
        .iter()
        .filter(|u| {
            matches!(
                u.provenance,
                DP::CoreTrait {
                    core: starkc::hir::CoreTrait::Iterator
                }
            )
        })
        .collect();
    assert_eq!(iterator_uses.len(), 1, "one `for` over a user iterator");

    let use_ = iterator_uses[0];
    match &use_.environment {
        GenericEnvironment::Static(bindings) => {
            assert!(
                !bindings.is_empty(),
                "`impl<T> Iterator for Repeat<T>` binds `T`; an empty environment is the defect \
                 this test exists for"
            );
            let bound: Vec<String> = bindings.iter().map(|(_, ty)| format!("{ty:?}")).collect();
            assert!(
                bound.iter().any(|ty| ty.contains("Int32")),
                "`T` must be bound to the CONCRETE argument for `Repeat<Int32>`, got {bound:?}"
            );
        }
        other => panic!("a user iterator has a static environment, got {other:?}"),
    }

    assert_eq!(
        use_.receiver_binding,
        ReceiverBinding::Exclusive,
        "`next(&mut self)` binds exclusively"
    );

    // And it still runs correctly: 7 twice.
    let execution = program.execute_hir().expect("the fixture must run");
    assert_eq!(execution.output, "14\n");
}

/// The trait is identified by RESOLVED identity, not spelling.
///
/// A user trait **named `Iterator`** is a different trait. The first iteration attempt matched on
/// `item_text(..) == "Iterator"`, which is DEV-BOUND-TRAIT-IDENTITY's class.
///
/// The impl below is written to match *structurally* — same name, a `type Item`, a `next` — so that
/// a spelling-based selector would accept it. Mutation-verified: switching `resolve_user_iterator`
/// back to a text comparison makes this test fail. An earlier version used a trait named
/// `Iterator2` and passed under that mutation, proving nothing.
#[test]
fn a_user_trait_named_iterator_does_not_make_a_type_iterable() {
    let file = Arc::new(starkc::source::SourceFile::new(
        "shadow.stark",
        "trait Iterator {\n    fn next(&mut self) -> Option<Int32>;\n}\n\
         struct Fake { v: Int32 }\n\
         impl Iterator for Fake {\n\
         \x20   type Item = Int32;\n\
         \x20   fn next(&mut self) -> Option<Int32> {\n        None\n    }\n}\n\
         fn main() {\n\
         \x20   let f: Fake = Fake { v: 1 };\n\
         \x20   for x in f {\n        println(x);\n    }\n}\n",
    ));
    let failure = CompilerSession::for_source(file, LanguageOptions::CORE)
        .check()
        .err()
        .expect("a user trait named `Iterator` is not the core `Iterator`");
    let rendered = failure.render();
    assert!(
        rendered.contains("iterable"),
        "expected a not-iterable diagnostic, got:\n{rendered}"
    );
}
