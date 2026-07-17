//! WP8.3 test framework test suite: exercises `test_runner` discovery and
//! execution directly (no `stark` binary / on-disk package needed).

use starkc::diag::Severity;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::test_runner::{discover_tests, filter_by_name, run_test, Outcome};
use starkc::typecheck;
use std::sync::Arc;

/// Parse + resolve + typecheck `source` as a standalone program, returning
/// the compiled pieces `test_runner` needs. Panics on any compile error —
/// every fixture below is expected to compile cleanly.
fn compile(
    source: &str,
) -> (
    starkc::hir::Hir,
    Arc<SourceFile>,
    typecheck::TypeCheckResult,
) {
    let file = Arc::new(SourceFile::new("test.stark", source.to_string()));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "parse failed: {:?}", parse_diags);

    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(
        resolve_diags.is_empty(),
        "resolve failed: {:?}",
        resolve_diags
    );

    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "typecheck failed: {:?}", errors);

    (hir, file, checked)
}

#[test]
fn discovers_top_level_test_functions_only() {
    let (hir, file, _) = compile(
        "
        fn add(a: Int32, b: Int32) -> Int32 { a + b }
        fn test_one() { assert(true); }
        fn test_two() { assert(true); }
        fn not_a_test() { assert(true); }
        fn main() {}
        ",
    );
    let tests = discover_tests(&hir, &file);
    let mut names: Vec<_> = tests.iter().map(|t| t.name.as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["test_one", "test_two"]);
}

#[test]
fn discovers_nested_mod_tests_with_qualified_names() {
    let (hir, file, _) = compile(
        "
        mod geometry {
            fn area(w: Int32, h: Int32) -> Int32 { w * h }
            fn test_area() { assert_eq(area(3, 4), 12); }
        }
        fn main() {}
        ",
    );
    let tests = discover_tests(&hir, &file);
    assert_eq!(tests.len(), 1);
    assert_eq!(tests[0].name, "geometry::test_area");
}

#[test]
fn functions_with_parameters_are_not_discovered_as_tests() {
    // `fn test_x(n: Int32)` can't be called as a zero-arg entry point, so
    // it must not be discovered even though the name matches.
    let (hir, file, _) = compile(
        "
        fn test_takes_arg(n: Int32) { assert(n == n); }
        fn main() {}
        ",
    );
    assert!(discover_tests(&hir, &file).is_empty());
}

#[test]
fn test_prefix_alone_is_not_a_test() {
    // `fn test()` has nothing after the marker prefix; not discovered.
    let (hir, file, _) = compile(
        "
        fn test() { assert(true); }
        fn main() {}
        ",
    );
    assert!(discover_tests(&hir, &file).is_empty());
}

#[test]
fn ignored_convention_marks_tests_ignored_but_still_discovered() {
    let (hir, file, _) = compile(
        "
        fn test_normal() { assert(true); }
        fn test_ignored_slow() { assert(false); }
        fn main() {}
        ",
    );
    let tests = discover_tests(&hir, &file);
    assert_eq!(tests.len(), 2);
    let normal = tests.iter().find(|t| t.name == "test_normal").unwrap();
    let ignored = tests
        .iter()
        .find(|t| t.name == "test_ignored_slow")
        .unwrap();
    assert!(!normal.ignored);
    assert!(ignored.ignored);
}

#[test]
fn passing_test_reports_passed_with_captured_output() {
    let (hir, file, checked) = compile(
        "
        fn test_prints() {
            println(\"hello from test\");
            assert(true);
        }
        fn main() {}
        ",
    );
    let tests = discover_tests(&hir, &file);
    let result = run_test(&hir, file, &checked.tables, &tests[0]);
    assert!(matches!(result.outcome, Outcome::Passed));
    assert_eq!(result.output, "hello from test\n");
}

#[test]
fn failing_assert_reports_failed_not_a_crash() {
    let (hir, file, checked) = compile(
        "
        fn test_fails() { assert(false); }
        fn main() {}
        ",
    );
    let tests = discover_tests(&hir, &file);
    let result = run_test(&hir, file, &checked.tables, &tests[0]);
    match result.outcome {
        Outcome::Failed { message } => assert!(message.contains("assertion failed")),
        _ => panic!("expected Failed outcome"),
    }
}

#[test]
fn assert_eq_failure_message_shows_both_sides() {
    let (hir, file, checked) = compile(
        "
        fn test_fails() { assert_eq(2 + 2, 5); }
        fn main() {}
        ",
    );
    let tests = discover_tests(&hir, &file);
    let result = run_test(&hir, file, &checked.tables, &tests[0]);
    match result.outcome {
        Outcome::Failed { message } => {
            assert!(message.contains('4'), "message was: {message}");
            assert!(message.contains('5'), "message was: {message}");
        }
        _ => panic!("expected Failed outcome"),
    }
}

#[test]
fn assert_ne_passes_when_values_differ_fails_when_equal() {
    let (hir, file, checked) = compile(
        "
        fn test_ne_pass() { assert_ne(1, 2); }
        fn test_ne_fail() { assert_ne(1, 1); }
        fn main() {}
        ",
    );
    let tests = discover_tests(&hir, &file);
    let pass = tests.iter().find(|t| t.name == "test_ne_pass").unwrap();
    let fail = tests.iter().find(|t| t.name == "test_ne_fail").unwrap();

    let pass_result = run_test(&hir, file.clone(), &checked.tables, pass);
    assert!(matches!(pass_result.outcome, Outcome::Passed));

    let fail_result = run_test(&hir, file, &checked.tables, fail);
    assert!(matches!(fail_result.outcome, Outcome::Failed { .. }));
}

#[test]
fn one_test_failing_does_not_affect_another_tests_outcome() {
    let (hir, file, checked) = compile(
        "
        fn test_a_fails() { assert(false); }
        fn test_b_passes() { assert(true); }
        fn main() {}
        ",
    );
    let tests = discover_tests(&hir, &file);
    for t in &tests {
        let result = run_test(&hir, file.clone(), &checked.tables, t);
        match t.name.as_str() {
            "test_a_fails" => assert!(matches!(result.outcome, Outcome::Failed { .. })),
            "test_b_passes" => assert!(matches!(result.outcome, Outcome::Passed)),
            other => panic!("unexpected test: {other}"),
        }
    }
}

#[test]
fn filter_by_name_matches_substring() {
    let (hir, file, _) = compile(
        "
        fn test_add_basic() { assert(true); }
        fn test_add_overflow() { assert(true); }
        fn test_sub_basic() { assert(true); }
        fn main() {}
        ",
    );
    let all = discover_tests(&hir, &file);
    let filtered = filter_by_name(&all, Some("add"));
    let mut names: Vec<_> = filtered.iter().map(|t| t.name.as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["test_add_basic", "test_add_overflow"]);
}

#[test]
fn filter_by_name_none_returns_everything() {
    let (hir, file, _) = compile(
        "
        fn test_one() { assert(true); }
        fn test_two() { assert(true); }
        fn main() {}
        ",
    );
    let all = discover_tests(&hir, &file);
    assert_eq!(filter_by_name(&all, None).len(), 2);
}
