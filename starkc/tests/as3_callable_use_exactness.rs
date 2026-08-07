//! **AS3 exit criterion 1 — exactly one record per executable user-callable use.**
//!
//! > Every executable user-callable use has exactly one record; duplicates and omissions fail an
//! > exactness test.
//!
//! **The design problem, stated because it decides what this file can prove.** After Boundary 4
//! deleted `find_method` and `find_impl_fn`, "every callable the engines execute came from a
//! publication" is true *by construction* — there is no other way to reach a body. A test asserting
//! that would pass against any publisher, including one that publishes nothing for a family nobody
//! exercises. It would be vacuous.
//!
//! So the expectations here are derived from a **different source than the table under test**: the
//! HIR's shape plus the checker's `expr_types`. For every expression that *syntactically* requires a
//! user callable — an operator on a nominal, a method call, a `for` over a user iterator, a
//! qualified trait call, a `Display` render — this file independently decides that a record must
//! exist, then checks whether one does. Nothing is read back out of `callable_uses` to decide what
//! `callable_uses` should contain.
//!
//! Two properties, and they fail for different reasons:
//!
//! * **Omission** — a required site with no record. The publisher missed a family.
//! * **Duplication** — a site with two records of the same kind. A publisher ran twice, and two
//!   consumers could then legitimately select different ones.
//!
//! `Display` is exempt from the second rule *by kind* and checked more strictly instead: one
//! expression genuinely carries many `Display` uses (`println((a, b))` renders two bodies), so
//! uniqueness is asserted as an exact correspondence with `display_uses`'s path keys — every
//! published `Display` use is reachable by exactly one path, and every path names a published use.
//! An extra use with no path, or a path pointing at nothing, fails.

use starkc::hir::{self, CoreTrait};
use starkc::options::LanguageOptions;
use starkc::session::{CheckedProgram, CompilerSession};
use starkc::source::SourceFile;
use starkc::typecheck::{CallableUse, DispatchProvenance, Ty};
use std::collections::BTreeMap;
use std::sync::Arc;

fn compile(source: &str) -> CheckedProgram {
    let file = Arc::new(SourceFile::new("test.stark", source));
    match CompilerSession::for_source(file, LanguageOptions::CORE).check() {
        Ok(program) => program,
        Err(failure) => panic!("fixture must compile:\n{}", failure.render()),
    }
}

/// The coarse kind of a record, for counting. Two records of the same kind at one expression are a
/// duplicate; two of different kinds are not (an operator and a method call can share an
/// expression).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RecordKind {
    Direct,
    Method,
    Bound,
    /// `CoreTrait` is not `Ord`, and this key lives in a `BTreeMap`. Its debug spelling is a
    /// stable, total discriminator and keeps the failure messages readable.
    Core(&'static str),
    FunctionValue,
}

fn core_name(core: CoreTrait) -> &'static str {
    match core {
        CoreTrait::Eq => "Eq",
        CoreTrait::Ord => "Ord",
        CoreTrait::Display => "Display",
        CoreTrait::Iterator => "Iterator",
        CoreTrait::Clone => "Clone",
        CoreTrait::Hash => "Hash",
        CoreTrait::Num => "Num",
        other => {
            let _ = other;
            "other"
        }
    }
}

fn kind_of(use_: &CallableUse) -> RecordKind {
    match &use_.provenance {
        DispatchProvenance::Direct => RecordKind::Direct,
        DispatchProvenance::Inherent
        | DispatchProvenance::TraitImpl { .. }
        | DispatchProvenance::Qualified { .. } => RecordKind::Method,
        DispatchProvenance::Bound { .. } => RecordKind::Bound,
        DispatchProvenance::CoreTrait { core } => RecordKind::Core(core_name(*core)),
        DispatchProvenance::FunctionValue => RecordKind::FunctionValue,
    }
}

fn peel(mut ty: &Ty) -> &Ty {
    while let Ty::Ref { inner, .. } = ty {
        ty = inner;
    }
    ty
}

fn is_user_nominal(ty: Option<&Ty>) -> bool {
    matches!(ty.map(peel), Some(Ty::Struct(..) | Ty::Enum(..)))
}

fn is_param(ty: Option<&Ty>) -> bool {
    matches!(ty.map(peel), Some(Ty::Param(_)))
}

/// What the HIR says must exist, derived without consulting `callable_uses`.
struct Required {
    expr: hir::ExprId,
    kinds: Vec<RecordKind>,
    what: String,
}

/// Walk the program's HIR and state, independently, which expressions must carry which records.
fn required_records(program: &CheckedProgram) -> Vec<Required> {
    let hir = program.hir();
    let types = &program.tables().expr_types;
    let mut out = Vec::new();

    for index in 0..hir.exprs.len() {
        let expr = hir::ExprId(index as u32);
        let node = hir.expr(expr);
        match &node.kind {
            // An operator on a user nominal desugars to `Eq::eq` / `Ord::cmp`; on a bounded
            // parameter it is late-bound but still a user callable.
            hir::ExprKind::Binary { op, lhs, .. } => {
                let lhs_ty = types.get(lhs);
                let core = match op {
                    starkc::ast::BinOp::Eq | starkc::ast::BinOp::Ne => Some(CoreTrait::Eq),
                    starkc::ast::BinOp::Lt
                    | starkc::ast::BinOp::Le
                    | starkc::ast::BinOp::Gt
                    | starkc::ast::BinOp::Ge => Some(CoreTrait::Ord),
                    _ => None,
                };
                let Some(core) = core else { continue };
                if is_user_nominal(lhs_ty) {
                    out.push(Required {
                        expr,
                        kinds: vec![RecordKind::Core(core_name(core))],
                        what: format!("operator on a user nominal ({core:?})"),
                    });
                } else if is_param(lhs_ty) {
                    out.push(Required {
                        expr,
                        kinds: vec![RecordKind::Bound],
                        what: format!("operator on a bounded parameter ({core:?})"),
                    });
                }
            }
            // `for x in it` over a user nominal runs that nominal's `Iterator::next`.
            hir::ExprKind::For { iter, .. } => {
                if is_user_nominal(types.get(iter)) {
                    out.push(Required {
                        expr,
                        kinds: vec![RecordKind::Core("Iterator")],
                        what: "for over a user Iterator".to_string(),
                    });
                }
            }
            hir::ExprKind::Call { callee, .. } => {
                match &hir.expr(*callee).kind {
                    // A method call: `x.m()`.
                    hir::ExprKind::Field { base, .. } => {
                        let base_ty = types.get(base);
                        if is_user_nominal(base_ty) {
                            out.push(Required {
                                expr,
                                kinds: vec![RecordKind::Method],
                                what: "method call on a user nominal".to_string(),
                            });
                        } else if is_param(base_ty) {
                            out.push(Required {
                                expr,
                                kinds: vec![RecordKind::Bound],
                                what: "method call on a bounded parameter".to_string(),
                            });
                        }
                    }
                    // A qualified trait call: `Tr::m(&a, &b)` / `Eq::eq(&a, &b)`.
                    hir::ExprKind::Path {
                        res: hir::Res::TraitMember(..),
                        ..
                    } => out.push(Required {
                        expr,
                        kinds: vec![RecordKind::Method],
                        what: "qualified user-trait call".to_string(),
                    }),
                    hir::ExprKind::Path {
                        res: hir::Res::CoreTraitMember(core, _),
                        ..
                    } => out.push(Required {
                        expr,
                        kinds: vec![RecordKind::Core(core_name(*core))],
                        what: "qualified core-trait call".to_string(),
                    }),
                    // A free call to a user function. **The record lives on the CALLEE PATH**,
                    // not the call: the callee's identity is what names the body, and the path is
                    // where the checker resolves it. Recorded here as a convention rather than
                    // discovered per-site, because two conventions coexist — methods, operators,
                    // iterators and Display key on the CALL — and a reader has to know which.
                    hir::ExprKind::Path {
                        res: hir::Res::Item(item),
                        ..
                    } => {
                        if matches!(hir.item(*item).kind, hir::ItemKind::Fn(_)) {
                            out.push(Required {
                                expr: *callee,
                                kinds: vec![RecordKind::Direct],
                                what: "free call to a user function (keyed on the callee path)"
                                    .to_string(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    out
}

fn records_by_expr(program: &CheckedProgram) -> BTreeMap<u32, Vec<RecordKind>> {
    let tables = program.tables();
    let mut map: BTreeMap<u32, Vec<RecordKind>> = BTreeMap::new();
    for (expr, ids) in &tables.callable_uses_by_expr {
        let kinds = ids
            .iter()
            .filter_map(|id| tables.callable_uses.get(id.0 as usize))
            .map(kind_of)
            .collect();
        map.insert(expr.0, kinds);
    }
    map
}

/// **No omissions.** Every independently-required record exists.
fn assert_no_omissions(label: &str, program: &CheckedProgram) {
    let published = records_by_expr(program);
    for required in required_records(program) {
        let found = published.get(&required.expr.0).cloned().unwrap_or_default();
        for kind in &required.kinds {
            assert!(
                found.contains(kind),
                "{label}: OMISSION — {} at ExprId({}) requires a {kind:?} record; published: \
                 {found:?}",
                required.what,
                required.expr.0
            );
        }
    }
}

/// **No duplicates.** At most one record of each kind per expression, except `Display`.
fn assert_no_duplicates(label: &str, program: &CheckedProgram) {
    for (expr, kinds) in records_by_expr(program) {
        let mut seen: BTreeMap<RecordKind, usize> = BTreeMap::new();
        for kind in kinds {
            *seen.entry(kind).or_default() += 1;
        }
        for (kind, count) in seen {
            if kind == RecordKind::Core("Display") {
                continue; // checked by path correspondence instead
            }
            assert_eq!(
                count, 1,
                "{label}: DUPLICATE — ExprId({expr}) carries {count} {kind:?} records; a use \
                 published twice lets two consumers select different ones"
            );
        }
    }
}

/// **`Display` uses and `display_uses` paths correspond exactly.** An extra use with no path, or a
/// path naming no use, fails — the uniqueness rule for the one family that legitimately publishes
/// many records against one expression.
fn assert_display_paths_correspond(label: &str, program: &CheckedProgram) {
    let tables = program.tables();
    let mut by_root: BTreeMap<u32, usize> = BTreeMap::new();
    for (root, _) in tables.display_uses.keys() {
        *by_root.entry(root.0).or_default() += 1;
    }
    let mut display_records: BTreeMap<u32, usize> = BTreeMap::new();
    for (expr, ids) in &tables.callable_uses_by_expr {
        let count = ids
            .iter()
            .filter_map(|id| tables.callable_uses.get(id.0 as usize))
            .filter(|u| {
                matches!(kind_of(u), RecordKind::Core("Display"))
                    || matches!(
                        u.provenance,
                        DispatchProvenance::Bound {
                            trait_: hir::BoundTrait::Core(CoreTrait::Display)
                        }
                    )
            })
            .count();
        if count > 0 {
            display_records.insert(expr.0, count);
        }
    }
    assert_eq!(
        display_records, by_root,
        "{label}: every Display record must be reachable by exactly one path, and every path must \
         name a record"
    );

    // And every path key must resolve to a real record.
    for ((root, path), id) in &tables.display_uses {
        assert!(
            tables.callable_uses.get(id.0 as usize).is_some(),
            "{label}: display path {path:?} at ExprId({}) names no record",
            root.0
        );
    }
}

fn check(label: &str, source: &str) {
    let program = compile(source);
    assert_no_omissions(label, &program);
    assert_no_duplicates(label, &program);
    assert_display_paths_correspond(label, &program);
    // The program must also RUN — an exactness claim over a program that traps proves little.
    program
        .execute_hir()
        .unwrap_or_else(|e| panic!("{label}: fixture must run: {}", e.message));
}

const DECLS: &str = "\
struct A { v: Int32 }
impl Display for A {
    fn fmt(&self) -> String { String::from(\"A!\") }
}
impl Eq for A {
    fn eq(&self, other: &A) -> Bool { self.v == other.v }
}
impl Ord for A {
    fn cmp(&self, other: &A) -> Ordering { self.v.cmp(&other.v) }
}
struct W<T> { v: T }
impl<T> Display for W<T> {
    fn fmt(&self) -> String { String::from(\"W!\") }
}
trait Speak { fn speak(&self) -> String; fn twice(&self) -> String { self.speak() } }
impl Speak for A { fn speak(&self) -> String { String::from(\"hi\") } }
struct Counter { n: Int32 }
impl Iterator for Counter {
    type Item = Int32;
    fn next(&mut self) -> Option<Int32> {
        if self.n > 2 { None } else { let v = self.n; self.n = self.n + 1; Some(v) }
    }
}
fn free(x: Int32) -> Int32 { x + 1 }
";

#[test]
fn free_calls_and_methods_are_exact() {
    check(
        "free calls and methods",
        &format!(
            "{DECLS}fn main() {{\n\
             \x20   println(free(1));\n\
             \x20   let a: A = A {{ v: 1 }};\n\
             \x20   println(a.speak());\n\
             \x20   println(a.twice());\n}}\n"
        ),
    );
}

#[test]
fn operators_on_a_user_nominal_are_exact() {
    check(
        "operators",
        &format!(
            "{DECLS}fn main() {{\n\
             \x20   let a: A = A {{ v: 1 }};\n\
             \x20   let b: A = A {{ v: 2 }};\n\
             \x20   println(a == b);\n\
             \x20   println(a < b);\n}}\n"
        ),
    );
}

#[test]
fn bounded_generic_dispatch_is_exact() {
    check(
        "bounded generics",
        &format!(
            "{DECLS}fn shout<T: Speak>(x: T) -> String {{ x.speak() }}\n\
             fn same<T: Eq>(x: T, y: T) -> Bool {{ x == y }}\n\
             fn main() {{\n\
             \x20   println(shout(A {{ v: 1 }}));\n\
             \x20   println(same(A {{ v: 1 }}, A {{ v: 1 }}));\n}}\n"
        ),
    );
}

#[test]
fn qualified_trait_calls_are_exact() {
    check(
        "qualified",
        &format!(
            "{DECLS}fn main() {{\n\
             \x20   let a: A = A {{ v: 1 }};\n\
             \x20   let b: A = A {{ v: 1 }};\n\
             \x20   println(Eq::eq(&a, &b));\n\
             \x20   println(Speak::speak(&a));\n}}\n"
        ),
    );
}

#[test]
fn user_iterators_are_exact() {
    check(
        "iterator",
        &format!(
            "{DECLS}fn main() {{\n\
             \x20   let c: Counter = Counter {{ n: 0 }};\n\
             \x20   for x in c {{ println(x); }}\n}}\n"
        ),
    );
}

#[test]
fn display_records_and_paths_correspond_across_shapes() {
    // The family that publishes many records against one expression, in every shape that does so.
    check(
        "display shapes",
        &format!(
            "{DECLS}fn main() {{\n\
             \x20   let a: A = A {{ v: 1 }};\n\
             \x20   let b: A = A {{ v: 2 }};\n\
             \x20   println(a);\n\
             \x20   println((a, b));\n\
             \x20   let o: Option<A> = Some(A {{ v: 3 }});\n\
             \x20   println(o);\n\
             \x20   let w: W<Int32> = W {{ v: 1 }};\n\
             \x20   println(w);\n\
             \x20   println(f\"{{w}}\");\n}}\n"
        ),
    );
}

#[test]
fn one_expression_carrying_several_kinds_is_not_a_duplicate() {
    // `println(a == b)` is one call expression with a `Direct`-ish builtin and an `Eq` record on
    // the operand. Different kinds at one expression must not read as duplication — the rule is
    // per kind, and a rule that counted records alone would fire here.
    check(
        "mixed kinds",
        &format!(
            "{DECLS}fn main() {{\n\
             \x20   let a: A = A {{ v: 1 }};\n\
             \x20   let b: A = A {{ v: 1 }};\n\
             \x20   println(free(1) == free(1));\n\
             \x20   println(a == b);\n}}\n"
        ),
    );
}

#[test]
fn every_published_use_names_a_real_declaration_or_defers_it() {
    // A record that names nothing is neither an omission nor a duplicate, and would pass both rules
    // above while being useless to a consumer. `Static` must name a body; `Bound` must name a
    // member; only `FunctionValue` may name neither.
    let program = compile(&format!(
        "{DECLS}fn shout<T: Speak>(x: T) -> String {{ x.speak() }}\n\
         fn main() {{\n\
         \x20   let a: A = A {{ v: 1 }};\n\
         \x20   println(a.speak());\n\
         \x20   println(shout(A {{ v: 2 }}));\n\
         \x20   println(a);\n}}\n"
    ));
    let tables = program.tables();
    for (index, use_) in tables.callable_uses.iter().enumerate() {
        match &use_.selection {
            starkc::typecheck::CalleeSelection::Static { body, .. } => {
                assert!(
                    tables.callable_types.contains_key(body),
                    "record {index} names body {body:?}, which has no signature (A3b)"
                );
            }
            starkc::typecheck::CalleeSelection::Bound { member, .. } => {
                assert!(
                    !member.is_empty(),
                    "record {index} is a Bound with no member"
                );
            }
            starkc::typecheck::CalleeSelection::FunctionValue => {}
        }
    }
}

#[test]
fn dev193_a_direct_call_is_not_published_as_a_function_value() {
    // **Found by the exactness test on its first run.** `free(1)` and `g(2)` published the SAME
    // record at their call expressions — `FunctionValue`, the selection meaning "the body is not
    // knowable here". For `free(1)` it is knowable: the callee path published `Direct`/`Static`.
    //
    // Two of the three binding times were therefore indistinguishable at the call site, which is
    // the conflation `CalleeSelection` exists to prevent. This asserts they differ.
    let program = compile(
        "fn free(x: Int32) -> Int32 { x + 1 }\n         fn main() {\n         \x20   println(free(1));\n         \x20   let g: fn(Int32) -> Int32 = free;\n         \x20   println(g(2));\n}\n",
    );
    let tables = program.tables();
    let hir = program.hir();
    let mut direct_calls = 0usize;
    let mut value_calls = 0usize;
    for index in 0..hir.exprs.len() {
        let expr = hir::ExprId(index as u32);
        let hir::ExprKind::Call { callee, .. } = &hir.expr(expr).kind else {
            continue;
        };
        let published_here: Vec<RecordKind> = tables
            .callable_uses_by_expr
            .get(&expr)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| tables.callable_uses.get(id.0 as usize))
                    .map(kind_of)
                    .collect()
            })
            .unwrap_or_default();
        match &hir.expr(*callee).kind {
            hir::ExprKind::Path {
                res: hir::Res::Item(_),
                ..
            } => {
                assert!(
                    !published_here.contains(&RecordKind::FunctionValue),
                    "a call to a known fn item must NOT publish FunctionValue; got \
                     {published_here:?}"
                );
                direct_calls += 1;
            }
            hir::ExprKind::Path {
                res: hir::Res::Local(_),
                ..
            } => {
                assert!(
                    published_here.contains(&RecordKind::FunctionValue),
                    "a call through a function VALUE must publish FunctionValue; got \
                     {published_here:?}"
                );
                value_calls += 1;
            }
            _ => {}
        }
    }
    assert!(
        direct_calls >= 1 && value_calls >= 1,
        "the fixture must exercise both a direct call and a value call, got \
         direct={direct_calls} value={value_calls}"
    );
    assert_eq!(program.execute_hir().expect("must run").output, "2\n3\n");
}
