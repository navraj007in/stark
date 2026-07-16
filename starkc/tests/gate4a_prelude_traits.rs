use starkc::diag::Severity;
use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn execute_snippet(source: &str) -> String {
    let file = Arc::new(SourceFile::new("snippet-test.stark", source.to_string()));
    let (ast, parse_diagnostics) = parse(&file, ParseMode::Program);
    assert!(
        parse_diagnostics.is_empty(),
        "parse failed: {:?}",
        parse_diagnostics
    );

    let (hir, resolve_diagnostics) = resolve(&ast, file.clone());
    assert!(
        resolve_diagnostics.is_empty(),
        "resolve failed: {:?}",
        resolve_diagnostics
    );

    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "typecheck failed: {:?}", errors);

    interp::run(&hir, file, &checked.tables).unwrap().output
}

#[test]
fn test_size_of_and_align_of() {
    let source = "
        fn main() {
            println(size_of::<Int32>());
            println(align_of::<Float64>());
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "8\n8\n");
}

#[test]
fn test_swap() {
    let source = "
        fn main() {
            let mut a = 10;
            let mut b = 20;
            swap(&mut a, &mut b);
            println(a);
            println(b);
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "20\n10\n");
}

#[test]
fn test_replace() {
    let source = "
        fn main() {
            let mut a = 10;
            let old = replace(&mut a, 99);
            println(a);
            println(old);
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "99\n10\n");
}

#[test]
fn test_take() {
    let source = "
        fn main() {
            let mut a = String::from(\"hello\");
            let old = take(&mut a);
            println(a.as_str());
            println(old.as_str());
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "\nhello\n");
}

#[test]
fn test_trait_bounds_clone_default_display() {
    let source = "
        fn require_clone<T: Clone>(val: T) {
            println(\"has clone\");
        }
        fn require_default<T: Default>() {
            println(\"has default\");
        }
        fn require_display<T: Display>(val: T) {
            println(\"has display\");
        }
        fn main() {
            require_clone::<Int32>(5);
            require_default::<Vec<Int32>>();
            require_display::<String>(String::from(\"yes\"));
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "has clone\nhas default\nhas display\n");
}

#[test]
fn test_float_hash_bound_rejected() {
    let source = "
        fn require_hash<T: Hash>(val: T) {}
        fn main() {
            require_hash::<Float32>(3.14f32);
        }
    ";
    let file = Arc::new(SourceFile::new(
        "snippet-test-fail.stark",
        source.to_string(),
    ));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();

    assert!(
        !errors.is_empty(),
        "expected typecheck to fail for Float32 with Hash bound"
    );
    let err_msg = format!("{:?}", errors[0]);
    assert!(err_msg.contains("does not satisfy trait bound") || err_msg.contains("E0500"));
}
