use starkc::diag::Severity;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn compile_program(source: &str) -> Vec<starkc::diag::Diagnostic> {
    let file = Arc::new(SourceFile::new("test.stark", source.to_string()));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    if !parse_diags.is_empty() {
        return parse_diags;
    }
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    if !resolve_diags.is_empty() {
        return resolve_diags;
    }
    let checked = typecheck::analyze(&hir, file.clone());
    checked.diagnostics
}

#[test]
fn test_trait_default_method_body_checking() {
    // 1. Valid body checks successfully
    let valid_source = "
        trait Foo {
            fn do_something(val: Int32) -> Int32 {
                val + 1
            }
        }
    ";
    let diags = compile_program(valid_source);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "expected valid trait default method body to typecheck cleanly"
    );

    // 2. Invalid body gets typecheck error
    let invalid_source = "
        trait Foo {
            fn do_something(val: Int32) -> String {
                val + 1
            }
        }
    ";
    let diags = compile_program(invalid_source);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        !errors.is_empty(),
        "expected invalid trait default method body to fail typechecking"
    );
}

#[test]
fn test_re_export_visibility_and_unresolved() {
    // 1. Unresolved import E0205
    let unresolved_source = "
        use non_existent::Something;
    ";
    let diags = compile_program(unresolved_source);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(!errors.is_empty(), "expected unresolved import to fail");
    assert!(errors[0].code.as_deref() == Some("E0205"));

    // 2. Multi-level private re-export visibility check E0207
    let private_source = "
        mod a {
            struct PrivateStruct {}
        }
        mod b {
            // Private import inside b
            use crate::a::PrivateStruct;
        }
        fn main() {
            // Should fail because PrivateStruct is private to a, or not public in b
            let x = b::PrivateStruct {};
        }
    ";
    let diags = compile_program(private_source);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(!errors.is_empty(), "expected private import to fail");
    assert!(errors.iter().any(|d| d.code.as_deref() == Some("E0207")));
}

#[test]
fn test_orphan_rules_and_overlapping_coherence() {
    // 1. Inherent implementation requires a local type
    let non_local_inherent = "
        impl Int32 {
            fn extra() {}
        }
    ";
    let diags = compile_program(non_local_inherent);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        !errors.is_empty(),
        "expected inherent impl of non-local type to fail"
    );
    assert!(errors[0].code.as_deref() == Some("E0500"));

    // 2. Overlapping implementations conflict
    let overlap_source = "
        struct MyStruct {}
        trait MyTrait {
            fn perform() {}
        }
        impl MyTrait for MyStruct {
            fn perform() {}
        }
        impl MyTrait for MyStruct {
            fn perform() {}
        }
    ";
    let diags = compile_program(overlap_source);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        !errors.is_empty(),
        "expected overlapping implementations to fail"
    );
    assert!(errors[0].code.as_deref() == Some("E0500"));
    let formatted = format!("{:?}", errors[0]);
    assert!(
        formatted.contains("conflicting implementation found in test.stark"),
        "diagnostic should identify both conflicting implementations"
    );
}

#[test]
fn test_unreachable_match_arms() {
    let unreachable_source = "
        fn test_match(x: Int32) {
            match x {
                1 => println(\"one\"),
                1 => println(\"two\"),
                _ => println(\"other\"),
            }
        }
    ";
    let diags = compile_program(unreachable_source);
    let warnings: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Warning)
        .collect();
    assert!(
        !warnings.is_empty(),
        "expected unreachable match arm warning"
    );
    assert!(warnings[0].message.contains("unreachable match arm"));

    // Nested pattern usefulness unreachability
    let nested_unreachable = "
        enum Custom {
            One(Int32),
            Two,
        }
        fn test_nested(c: Custom) {
            match c {
                Custom::One(_) => println(\"any one\"),
                Custom::One(5) => println(\"specific one\"),
                Custom::Two => println(\"two\"),
            }
        }
    ";
    let diags = compile_program(nested_unreachable);
    let warnings: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Warning)
        .collect();
    assert!(
        !warnings.is_empty(),
        "expected nested redundant arm warning"
    );
    assert!(warnings[0].message.contains("unreachable match arm"));
}
