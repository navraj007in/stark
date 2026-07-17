//! Phase 4E — Math, Random, and I/O (`stark-spec-parity-roadmap.md`,
//! `06-Standard-Library.md`). Exercises each new stdlib surface end to end
//! (parse -> resolve -> typecheck -> execute) rather than unit-testing
//! individual compiler passes.

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
fn math_constants() {
    let output = execute_snippet("fn main() { println(PI); println(E); }");
    assert_eq!(output, "3.141592653589793\n2.718281828459045\n");
}

#[test]
fn math_abs_int_and_float() {
    let output =
        execute_snippet("fn main() { println(abs(-5)); println(abs(5)); println(abs(-2.5)); }");
    assert_eq!(output, "5\n5\n2.5\n");
}

#[test]
fn math_min_max_are_qualified_not_bare() {
    // Bare `min`/`max` are already claimed by the `tensor` extension
    // (`resolve.rs`); Math's versions are `math::min`/`math::max`.
    let output =
        execute_snippet("fn main() { println(math::min(3, 7)); println(math::max(3, 7)); }");
    assert_eq!(output, "3\n7\n");
}

#[test]
fn math_clamp() {
    let output = execute_snippet(
        "fn main() { println(clamp(15, 0, 10)); println(clamp(-5, 0, 10)); println(clamp(5, 0, 10)); }",
    );
    assert_eq!(output, "10\n0\n5\n");
}

#[test]
fn math_transcendental_functions() {
    let output = execute_snippet(
        "fn main() {
            println(pow(2.0, 10.0));
            println(log10(100.0));
            println(sin(0.0));
            println(cos(0.0));
            println(floor(3.7));
            println(ceil(3.2));
            println(round(3.5));
            println(trunc(3.9));
        }",
    );
    assert_eq!(output, "1024\n2\n0\n1\n3\n4\n4\n3\n");
}

#[test]
fn random_next_int_is_deterministic_for_a_seed() {
    // Same seed -> same sequence (LCG is a pure function of its state).
    let source = "
        fn main() {
            let mut r = Random::new(42u64);
            println(r.next_int());
            println(r.next_int());
        }
    ";
    assert_eq!(execute_snippet(source), execute_snippet(source));
}

#[test]
fn random_next_float_is_in_unit_range() {
    let output = execute_snippet(
        "fn main() {
            let mut r = Random::new(1u64);
            let f = r.next_float();
            println(f >= 0.0 && f < 1.0);
        }",
    );
    assert_eq!(output, "true\n");
}

#[test]
fn random_range_stays_within_bounds() {
    let output = execute_snippet(
        "fn main() {
            let mut r = Random::new(7u64);
            let mut i = 0;
            let mut ok = true;
            while i < 20 {
                let v = r.range(5, 15);
                if v < 5 || v >= 15 { ok = false; }
                i = i + 1;
            }
            println(ok);
        }",
    );
    assert_eq!(output, "true\n");
}

#[test]
fn eprintln_does_not_pollute_captured_stdout() {
    let output = execute_snippet("fn main() { eprintln(\"to stderr\"); println(\"to stdout\"); }");
    assert_eq!(output, "to stdout\n");
}

#[test]
fn read_file_missing_path_returns_io_error_not_found() {
    let output = execute_snippet(
        "fn main() {
            match read_file(\"/this/path/should/not/exist/on/any/machine\") {
                Ok(_) => println(\"unexpected ok\"),
                Err(IOError::NotFound) => println(\"not found as expected\"),
                Err(_) => println(\"wrong error kind\"),
            }
        }",
    );
    assert_eq!(output, "not found as expected\n");
}

#[test]
fn write_file_then_read_file_round_trips() {
    let path = std::env::temp_dir().join(format!("stark-phase4e-{}.txt", std::process::id()));
    let escaped = path.to_string_lossy().replace('\\', "\\\\");
    let source = format!(
        "fn main() {{
            write_file(\"{escaped}\", \"hello phase 4e\").unwrap();
            match read_file(\"{escaped}\") {{
                Ok(content) => println(content),
                Err(_) => println(\"read failed\"),
            }}
        }}"
    );
    assert_eq!(execute_snippet(&source), "hello phase 4e\n");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn io_error_other_constructs_and_matches() {
    let output = execute_snippet(
        "fn main() {
            let e = IOError::Other(String::from(\"custom\"));
            match e {
                IOError::Other(msg) => println(msg),
                _ => println(\"wrong variant\"),
            }
        }",
    );
    assert_eq!(output, "custom\n");
}

#[test]
fn io_error_unit_variants_all_construct_and_match() {
    // Each constructor gets its own binding and match (IOError isn't
    // Copy, since Other(String) carries non-Copy data — moving several
    // out of a shared array, as an earlier version of this test tried,
    // correctly hits the borrow checker's E0100).
    let output = execute_snippet(
        "fn main() {
            let a = IOError::NotFound;
            let b = IOError::PermissionDenied;
            let c = IOError::AlreadyExists;
            let d = IOError::InvalidInput;
            match a {
                IOError::NotFound => println(\"NotFound\"),
                _ => println(\"wrong\"),
            }
            match b {
                IOError::PermissionDenied => println(\"PermissionDenied\"),
                _ => println(\"wrong\"),
            }
            match c {
                IOError::AlreadyExists => println(\"AlreadyExists\"),
                _ => println(\"wrong\"),
            }
            match d {
                IOError::InvalidInput => println(\"InvalidInput\"),
                _ => println(\"wrong\"),
            }
        }",
    );
    assert_eq!(
        output,
        "NotFound\nPermissionDenied\nAlreadyExists\nInvalidInput\n"
    );
}
