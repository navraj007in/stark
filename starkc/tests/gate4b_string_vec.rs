use std::sync::Arc;
use starkc::diag::Severity;
use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;

fn execute_snippet(source: &str) -> String {
    let file = Arc::new(SourceFile::new("snippet-test.stark", source.to_string()));
    let (ast, parse_diagnostics) = parse(&file, ParseMode::Program);
    assert!(parse_diagnostics.is_empty(), "parse failed: {:?}", parse_diagnostics);
    
    let (hir, resolve_diagnostics) = resolve(&ast, file.clone());
    assert!(resolve_diagnostics.is_empty(), "resolve failed: {:?}", resolve_diagnostics);
    
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked.diagnostics.iter().filter(|d| d.severity == Severity::Error).collect();
    assert!(errors.is_empty(), "typecheck failed: {:?}", errors);
    
    interp::run(&hir, file, &checked.tables).unwrap().output
}

#[test]
fn test_string_methods() {
    let source = "
        fn main() {
            let s = String::from(\"  STARK-lang  \");
            println(s.contains(\"lang\"));
            println(s.starts_with(\"  ST\"));
            println(s.ends_with(\"  \"));
            
            let trimmed = s.trim();
            println(trimmed.as_str());
            
            let lower = trimmed.to_lowercase();
            println(lower.as_str());
            
            let upper = trimmed.to_uppercase();
            println(upper.as_str());
            
            let replaced = trimmed.replace(\"lang\", \"compiler\");
            println(replaced.as_str());
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(
        output,
        "true\ntrue\ntrue\nSTARK-lang\nstark-lang\nSTARK-LANG\nSTARK-compiler\n"
    );
}

#[test]
fn test_string_iterators_and_bytes() {
    let source = "
        fn main() {
            let s = String::from(\"abc\");
            let mut it = s.chars();
            while true {
                match it.next() {
                    Some(ch) => {
                        println(ch);
                    }
                    None => {
                        break;
                    }
                }
            }
            
            let bytes = s.bytes();
            println(bytes.len());
            
            let into_b = s.into_bytes();
            println(into_b.len());
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "a\nb\nc\n3\n3\n");
}

#[test]
fn test_string_split() {
    let source = "
        fn main() {
            let s = String::from(\"one,two,three\");
            let mut it = s.split(\",\");
            while true {
                match it.next() {
                    Some(part) => {
                        println(part.as_str());
                    }
                    None => {
                        break;
                    }
                }
            }
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "one\ntwo\nthree\n");
}

#[test]
fn test_vec_get_mut_and_as_slice() {
    let source = "
        fn main() {
            let mut v = Vec::new();
            v.push(10);
            v.push(20);
            
            match v.get_mut(1u64) {
                Some(r) => {
                    *r = 99;
                }
                None => {}
            }
            
            println(v[1u64]);
            
            let slice = v.as_slice();
            println(slice.len());
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "99\n2\n");
}

#[test]
fn test_vec_iter_and_extend() {
    let source = "
        fn main() {
            let mut v = Vec::new();
            v.push(1);
            v.push(2);
            
            let mut it = v.iter();
            while true {
                match it.next() {
                    Some(r) => {
                        println(*r);
                    }
                    None => {
                        break;
                    }
                }
            }
            
            let mut other = Vec::new();
            other.push(3);
            other.push(4);
            
            let mut it2 = other.iter();
            v.extend(&mut it2);
            println(v.len());
            println(v[2u64]);
            println(v[3u64]);
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "1\n2\n4\n3\n4\n");
}
