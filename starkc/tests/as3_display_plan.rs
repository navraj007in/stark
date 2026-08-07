//! **AS3 Boundary 4 — the `Display` dispatch plan, published.**
//!
//! `println(x)` may render zero, one or many user bodies, at positions the argument's STATIC type
//! determines. The checker now publishes that plan as `display_uses: (root expr, DisplayPath) ->
//! CallableUseId`, so neither engine has to scan a nominal's impls for a member named `fmt`.
//!
//! **Why the key is a path and not a nominal.** `println((W<Int32>, W<Bool>))` renders the SAME
//! body at two instantiations. A runtime `Value::Struct { item, fields }` carries no type
//! arguments, so a nominal-keyed lookup cannot tell the two apart; the static position can. That
//! case is `two_instantiations_of_one_generic_are_distinct_positions` below, and it is the one that
//! rules out the cheaper design.
//!
//! **The STOP rule.** `println(W<A>)` prints `W!`, not a `W!` containing an `A!` — the outer
//! nominal's own `fmt` runs and the renderer does not descend. So the plan must have exactly one
//! entry, at the root. Publishing the inner `A` would claim a call no engine makes.
//!
//! Measured against `AS3-DISPLAY-CHARACTERIZATION.md`, whose table these fixtures mirror.
//!
//! **Scope: publication only.** The engines do not consume this yet — `find_method`/`find_impl_fn`
//! still answer Display at run time. Consumption is the next step, and until it lands these tests
//! guard the plan's shape rather than its use.

use starkc::options::LanguageOptions;
use starkc::session::CompilerSession;
use starkc::source::SourceFile;
use starkc::typecheck::{CalleeSelection, DisplayStep};
use std::sync::Arc;

const DECLS: &str = "\
struct A { v: Int32 }
impl Display for A {
    fn fmt(&self) -> String { String::from(\"A!\") }
}
struct B { v: Int32 }
impl Display for B {
    fn fmt(&self) -> String { String::from(\"B!\") }
}
struct W<T> { v: T }
impl<T> Display for W<T> {
    fn fmt(&self) -> String { String::from(\"W!\") }
}
";

fn compile(source: &str) -> starkc::session::CheckedProgram {
    let file = Arc::new(SourceFile::new("test.stark", source));
    match CompilerSession::for_source(file, LanguageOptions::CORE).check() {
        Ok(program) => program,
        Err(failure) => panic!("must compile:\n{}", failure.render()),
    }
}

/// The published plan as `(path, "Static"|"Bound")`, in path order.
fn plan(source: &str) -> Vec<(Vec<DisplayStep>, &'static str)> {
    let program = compile(source);
    let tables = program.tables();
    tables
        .display_uses
        .iter()
        .map(|((_, path), id)| {
            let kind = match &tables.callable_uses[id.0 as usize].selection {
                CalleeSelection::Static { .. } => "Static",
                CalleeSelection::Bound { .. } => "Bound",
                CalleeSelection::FunctionValue => "FunctionValue",
            };
            (path.0.clone(), kind)
        })
        .collect()
}

/// Plan plus the rendered output, so a plan that is structurally right but names the wrong body
/// still fails.
fn plan_and_output(source: &str) -> (Vec<(Vec<DisplayStep>, &'static str)>, String) {
    let out = compile(source).execute_hir().expect("must run").output;
    (plan(source), out)
}

fn main_with(body: &str) -> String {
    format!("{DECLS}fn main() {{\n{body}\n}}\n")
}

#[test]
fn a_top_level_nominal_publishes_one_use_at_the_root() {
    let (plan, out) = plan_and_output(&main_with("    let a: A = A { v: 1 };\n    println(a);"));
    assert_eq!(plan, vec![(vec![], "Static")]);
    assert_eq!(out, "A!\n");
}

#[test]
fn a_tuple_of_two_nominals_publishes_two_positions() {
    let (plan, out) = plan_and_output(&main_with(
        "    let a: A = A { v: 1 };\n    let b: B = B { v: 2 };\n    println((a, b));",
    ));
    assert_eq!(
        plan,
        vec![
            (vec![DisplayStep::TupleField(0)], "Static"),
            (vec![DisplayStep::TupleField(1)], "Static"),
        ],
        "one argument expression, two distinct `fmt` bodies"
    );
    assert_eq!(out, "(A!, B!)\n");
}

#[test]
fn the_same_nominal_twice_is_two_positions_not_one() {
    // One body, one environment, reached at two positions. The plan has two entries because the
    // POSITIONS are two — collapsing them would make the count depend on the type rather than the
    // structure, and the renderer visits both.
    let (plan, out) = plan_and_output(&main_with(
        "    let a1: A = A { v: 1 };\n    let a2: A = A { v: 2 };\n    println((a1, a2));",
    ));
    assert_eq!(plan.len(), 2);
    assert_eq!(out, "(A!, A!)\n");
}

#[test]
fn two_instantiations_of_one_generic_are_distinct_positions() {
    // **The case that rules out keying by nominal.** `W<Int32>` and `W<Bool>` are the same
    // `ItemId`, and the runtime value carries no type arguments, so only the static position tells
    // them apart. Their environments differ, which is why this matters beyond bookkeeping.
    let source = main_with(
        "    let x: W<Int32> = W { v: 1 };\n\
         \x20   let y: W<Bool> = W { v: true };\n\
         \x20   println((x, y));",
    );
    let (plan, out) = plan_and_output(&source);
    assert_eq!(
        plan,
        vec![
            (vec![DisplayStep::TupleField(0)], "Static"),
            (vec![DisplayStep::TupleField(1)], "Static"),
        ]
    );
    assert_eq!(out, "(W!, W!)\n");

    // The two uses must carry DIFFERENT environments — `T = Int32` and `T = Bool`. A plan that
    // published one shared entry, or two entries with the same (or empty) environment, would pass
    // the shape assertion above and still be wrong.
    let program = compile(&source);
    let tables = program.tables();
    let envs: Vec<String> = tables
        .display_uses
        .values()
        .map(|id| format!("{:?}", tables.callable_uses[id.0 as usize].environment))
        .collect();
    assert_eq!(envs.len(), 2);
    assert_ne!(
        envs[0], envs[1],
        "the two instantiations must publish different environments, got {envs:?}"
    );
}

#[test]
fn a_container_publishes_one_position_for_its_element() {
    // Executed once per element, but ONE position: the plan is static, the loop is the renderer's.
    let (plan, out) = plan_and_output(&main_with(
        "    let mut v: Vec<A> = Vec::new();\n\
         \x20   v.push(A { v: 1 });\n\
         \x20   v.push(A { v: 2 });\n\
         \x20   println(v);",
    ));
    assert_eq!(plan, vec![(vec![DisplayStep::VecElement], "Static")]);
    assert_eq!(out, "[A!, A!]\n");
}

#[test]
fn option_and_result_publish_their_payload_positions() {
    let (opt_plan, opt_out) = plan_and_output(&main_with(
        "    let o: Option<A> = Some(A { v: 1 });\n    println(o);",
    ));
    assert_eq!(opt_plan, vec![(vec![DisplayStep::OptionSome], "Static")]);
    assert_eq!(opt_out, "Some(A!)\n");

    // `Result` publishes BOTH arms: the plan is static, so it cannot know which one runs.
    let (res_plan, res_out) = plan_and_output(&main_with(
        "    let r: Result<A, B> = Ok(A { v: 1 });\n    println(r);",
    ));
    assert_eq!(
        res_plan,
        vec![
            (vec![DisplayStep::ResultOk], "Static"),
            (vec![DisplayStep::ResultErr], "Static"),
        ]
    );
    assert_eq!(res_out, "Ok(A!)\n");
}

#[test]
fn the_walk_stops_at_the_first_nominal_with_a_display_impl() {
    // **The STOP rule.** `W<A>` renders as `W!` — `W`'s own `fmt` runs and the renderer never
    // reaches the `A` inside. Exactly one entry, at the root. A walk that descended would publish
    // a use no engine executes, and any totality claim over the plan would then be false.
    let (plan, out) = plan_and_output(&main_with(
        "    let w: W<A> = W { v: A { v: 1 } };\n    println(w);",
    ));
    assert_eq!(
        plan,
        vec![(vec![], "Static")],
        "the inner `A` must NOT be published: it is never rendered"
    );
    assert_eq!(out, "W!\n");
}

#[test]
fn a_generic_body_publishes_a_bound_position() {
    // The design gate from `AS3-DISPLAY-CHARACTERIZATION.md` §3. Inside `show`, `T` is unbound and
    // no body can be named — but the OBLIGATION is fixed, and that is what `Bound` records.
    let (plan, out) = plan_and_output(&format!(
        "{DECLS}fn show<T: Display>(x: T) {{ println(x); }}\n\
         fn main() {{ show(A {{ v: 1 }}); }}\n"
    ));
    assert_eq!(plan, vec![(vec![], "Bound")]);
    assert_eq!(out, "A!\n");
}

#[test]
fn a_generic_body_publishes_bound_positions_inside_a_tuple() {
    // Two parameters, two positions, both late-bound. Proves the path walk and the Bound branch
    // compose rather than being two special cases.
    let (plan, out) = plan_and_output(&format!(
        "{DECLS}fn show2<P: Display, Q: Display>(x: P, y: Q) {{ println((x, y)); }}\n\
         fn main() {{ show2(A {{ v: 1 }}, B {{ v: 2 }}); }}\n"
    ));
    assert_eq!(
        plan,
        vec![
            (vec![DisplayStep::TupleField(0)], "Bound"),
            (vec![DisplayStep::TupleField(1)], "Bound"),
        ]
    );
    assert_eq!(out, "(A!, B!)\n");
}

#[test]
fn primitives_publish_no_position() {
    // The engines render these themselves; there is no user callable to name. A plan entry here
    // would be an invention, and `Int32`/`String`/`Bool` are the overwhelming majority of real
    // `println` arguments, so a walk that over-published would be noisy as well as wrong.
    for body in [
        "    println(7);",
        "    println(true);",
        "    println(String::from(\"hi\"));",
        "    println((1, true));",
    ] {
        assert!(
            plan(&main_with(body)).is_empty(),
            "no user callable renders {body}"
        );
    }
}

#[test]
fn a_nested_container_publishes_the_full_path() {
    // Depth beyond one step, so the path is genuinely a path rather than a single tag.
    let (plan, out) = plan_and_output(&main_with(
        "    let mut v: Vec<A> = Vec::new();\n\
         \x20   v.push(A { v: 1 });\n\
         \x20   let o: Option<Vec<A>> = Some(v);\n\
         \x20   println(o);",
    ));
    assert_eq!(
        plan,
        vec![(
            vec![DisplayStep::OptionSome, DisplayStep::VecElement],
            "Static"
        )]
    );
    assert_eq!(out, "Some([A!])\n");
}
