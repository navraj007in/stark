//! **Campaign A final audit §8 — function values keep their instantiation on every reachable route.**
//!
//! Function values are the highest-risk carrier in the language: the generic instantiation is fixed
//! at the coercion and **cannot be recovered from the later `Ty::Fn`**, which records only the
//! signature. DEV-178 put the bindings on the value for that reason; DEV-197 then found the call
//! site discarding them, and neither defect changed an answer, because the bodies involved returned
//! their argument unchanged.
//!
//! So every witness here answers `size_of::<T>()` — 8 for `Float64`, 4 for `Int32`. An answer of
//! `8` is only producible by a body running under the instantiation the coercion captured. A test
//! using an identity-shaped function would reproduce exactly the blindness that let DEV-197
//! accumulate nine sites.
//!
//! The mutation half lives with the interpreter
//! (`audit_10d_a_function_value_stripped_of_its_bindings_is_refused`), because stripping the
//! bindings requires a `#[cfg(test)]` seam at the capture site. This file is the reachability half:
//! it enumerates the routes a function value can actually take through Core v1 and shows the
//! instantiation surviving each.

use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Runs `source` and returns its output, insisting it neither trapped nor tripped an invariant.
fn output_of(source: &str) -> String {
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
    let outcome = interp::run_capturing(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    );
    assert!(outcome.result.is_ok(), "{:?}", outcome.result);
    outcome.output
}

/// `width<T>` is the probe: its answer is decided entirely by the instantiation, so `8` cannot be
/// produced by a body running with `T` unbound or bound to anything else.
const WIDTH: &str = "fn width<T>(x: T) -> Int32 { size_of::<T>() as Int32 } ";

/// **The creator's frame is gone by the time the value is called.** This is the route that makes
/// reconstruction impossible in principle rather than merely inconvenient.
#[test]
fn a_function_value_outlives_the_frame_that_created_it() {
    let output = output_of(&format!(
        "{WIDTH}fn make() -> fn(Float64) -> Int32 {{ width }} \
         fn main() {{ let f = make(); println(f(1.5)); }}"
    ));
    assert_eq!(
        output, "8\n",
        "the instantiation must travel with the value"
    );
}

/// Passed as an argument to another function, which calls it.
#[test]
fn a_function_value_survives_being_passed_through_another_function() {
    let output = output_of(&format!(
        "{WIDTH}fn apply(g: fn(Float64) -> Int32) -> Int32 {{ g(1.5) }} \
         fn main() {{ println(apply(width)); }}"
    ));
    assert_eq!(output, "8\n");
}

/// Created INSIDE a generic body, where a caller-frame environment is live and could plausibly be
/// mistaken for the value's own.
#[test]
fn a_function_value_created_inside_a_generic_body_keeps_its_own_instantiation() {
    let output = output_of(&format!(
        "{WIDTH}fn outer<U>(u: U) -> Int32 {{ let f: fn(Float64) -> Int32 = width; f(1.5) }} \
         fn main() {{ println(outer(1)); }}"
    ));
    assert_eq!(
        output, "8\n",
        "the value's captured `T = Float64` must win over the enclosing `U = Int32`"
    );
}

/// `Option::map` — one of the combinator paths DEV-197's third discovery event found executing a
/// body with no environment at all.
#[test]
fn option_map_preserves_the_captured_instantiation() {
    let output = output_of(&format!(
        "{WIDTH}fn main() {{ let o: Option<Float64> = Some(1.5); \
         match o.map(width) {{ Some(n) => println(n), None => println(0) }} }}"
    ));
    assert_eq!(output, "8\n");
}

/// `Result::map` — the same combinator family, the other carrier.
#[test]
fn result_map_preserves_the_captured_instantiation() {
    let output = output_of(&format!(
        "{WIDTH}fn main() {{ let r: Result<Float64, Int32> = Ok(1.5); \
         match r.map(width) {{ Ok(n) => println(n), Err(_e) => println(0) }} }}"
    ));
    assert_eq!(output, "8\n");
}

/// Stored in a struct field, read back out, then called — the route that makes the value outlive
/// any expression the checker could key an environment on.
#[test]
fn a_function_value_stored_in_an_aggregate_keeps_its_bindings() {
    let output = output_of(&format!(
        "{WIDTH}struct H {{ f: fn(Float64) -> Int32 }} \
         fn main() {{ let h = H {{ f: width }}; let g = h.f; println(g(1.5)); }}"
    ));
    assert_eq!(output, "8\n");
}

/// **The control that makes the six above mean something.** If `width` answered the same value at
/// every instantiation, every test in this file would pass with the bindings discarded — which is
/// precisely how DEV-197's first two defects stayed invisible.
#[test]
fn the_probe_actually_depends_on_its_instantiation() {
    let wide = output_of(&format!("{WIDTH}fn main() {{ println(width(1.5)); }}"));
    let narrow = output_of(&format!("{WIDTH}fn main() {{ println(width(1)); }}"));
    assert_eq!(wide, "8\n");
    assert_eq!(narrow, "4\n");
    assert_ne!(
        wide, narrow,
        "the probe must distinguish instantiations, or none of this file's evidence holds"
    );
}
