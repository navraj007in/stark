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

    interp::run(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    )
    .unwrap()
    .output
}

#[test]
fn test_hashmap_basic() {
    let source = "
        fn main() {
            let mut map: HashMap<Int32, String> = HashMap::new();
            println(map.len());
            println(map.is_empty());
            
            println(map.insert(42, String::from(\"hello\")).is_none());
            println(map.len());
            println(map.is_empty());
            
            // Check lookup
            match map.get(&42) {
                Some(val) => println(val.as_str()),
                None => println(\"not found\"),
            }
            
            // Check contains
            println(map.contains_key(&42));
            println(map.contains_key(&99));
            
            // Re-insert to get old value
            match map.insert(42, String::from(\"world\")) {
                Some(old) => println(old.as_str()),
                None => println(\"none\"),
            }
            
            // Get mutated value
            match map.get_mut(&42) {
                Some(val) => {
                    println(val.as_str());
                }
                None => {}
            }
            
            // Remove key
            match map.remove(&42) {
                Some(removed) => println(removed.as_str()),
                None => println(\"not found\"),
            }
            println(map.len());
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(
        output,
        "0\ntrue\ntrue\n1\nfalse\nhello\ntrue\nfalse\nhello\nworld\nworld\n0\n"
    );
}

#[test]
fn test_hashset_basic() {
    let source = "
        fn main() {
            let mut set: HashSet<String> = HashSet::new();
            println(set.len());
            println(set.is_empty());
            
            println(set.insert(String::from(\"apple\")));
            println(set.insert(String::from(\"apple\"))); // duplicate insert returns false
            println(set.len());
            
            println(set.contains(&String::from(\"apple\")));
            println(set.contains(&String::from(\"banana\")));
            
            println(set.remove(&String::from(\"apple\")));
            println(set.contains(&String::from(\"apple\")));
            println(set.len());
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(
        output,
        "0\ntrue\ntrue\nfalse\n1\ntrue\nfalse\ntrue\nfalse\n0\n"
    );
}

#[test]
fn test_hashmap_iterators() {
    let source = "
        fn main() {
            let mut map: HashMap<Int32, String> = HashMap::new();
            map.insert(1, String::from(\"one\"));
            map.insert(2, String::from(\"two\"));
            
            // Test keys iteration
            let mut keys = map.keys();
            while true {
                match keys.next() {
                    Some(k) => {
                        println(k);
                    }
                    None => {
                        break;
                    }
                }
            }
            
            // Test values iteration
            let mut values = map.values();
            while true {
                match values.next() {
                    Some(v) => {
                        println(v.as_str());
                    }
                    None => {
                        break;
                    }
                }
            }
            
            // Test key-value pairs iteration
            let mut items = map.iter();
            while true {
                match items.next() {
                    Some(pair) => {
                        println(pair.0);
                        println(pair.1.as_str());
                    }
                    None => {
                        break;
                    }
                }
            }
        }
    ";
    let output = execute_snippet(source);
    // BTreeMap keeps elements sorted by key, so output order is deterministic!
    assert_eq!(output, "1\n2\none\ntwo\n1\none\n2\ntwo\n");
}

#[test]
fn test_hashset_iterators() {
    let source = "
        fn main() {
            let mut set: HashSet<Int32> = HashSet::new();
            set.insert(10);
            set.insert(20);
            
            let mut it = set.iter();
            while true {
                match it.next() {
                    Some(val) => {
                        println(val);
                    }
                    None => {
                        break;
                    }
                }
            }
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "10\n20\n");
}

#[test]
fn test_collections_extend() {
    let source = "
        fn main() {
            let mut map: HashMap<Int32, String> = HashMap::new();
            map.insert(10, String::from(\"ten\"));
            
            let mut other_map: HashMap<Int32, String> = HashMap::new();
            other_map.insert(20, String::from(\"twenty\"));
            other_map.insert(30, String::from(\"thirty\"));
            
            let mut it = other_map.iter();
            map.extend(&mut it);
            
            println(map.len());
            println(map.contains_key(&20));
            println(map.contains_key(&30));
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "3\ntrue\ntrue\n");
}

/// **The `Iterator` combinator surface is refused by the front end (WP-C7.9 Packet E, `E0105`).**
///
/// This test used to execute `count`, `map`, `filter`, `fold`, `any`, `all`, `find` and `collect`
/// through the HIR interpreter, and it passed. It passed because the HIR interpreter is the only
/// engine that implements them: none has a MIR lowering, so every program here type-checked and
/// ran in one engine while no compiler could build it.
///
/// That split is the defect Packet E closed, so the assertion is inverted rather than deleted. The
/// combinators are not gone from the language's intent — implementing them needs MIR adapter types
/// and is its own work package (`WP-C7.9-ACCEPTED-SURFACE-AUDIT.md`). When that lands, this test
/// fails and becomes a three-engine case again, which is the right way for it to be noticed.
#[test]
fn test_iterator_combinators_are_refused_by_the_front_end() {
    for (name, snippet) in [
        ("count", "let mut it = set.iter(); println(it.count());"),
        (
            "map",
            "let mut it = set.iter(); let mut m = it.map(double); println(m.count());",
        ),
        (
            "filter",
            "let mut it = set.iter(); let mut f = it.filter(is_odd); println(f.count());",
        ),
        ("fold", "let mut it = set.iter(); println(it.fold(0, add));"),
        ("any", "let mut it = set.iter(); println(it.any(is_odd));"),
        ("all", "let mut it = set.iter(); println(it.all(is_odd));"),
        (
            "collect",
            "let mut it = set.iter(); let v: Vec<Int32> = it.collect(); println(v.len());",
        ),
    ] {
        let source = format!(
            "fn double(x: &Int32) -> Int32 {{ let val = *x; val * 2 }}\n\
             fn add(acc: Int32, x: Int32) -> Int32 {{ acc + x }}\n\
             fn is_odd(x: &Int32) -> Bool {{ let val = *x; val % 2 != 0 }}\n\
             fn main() {{ let mut set: HashSet<Int32> = HashSet::new(); set.insert(1); {snippet} }}\n"
        );
        let file = Arc::new(SourceFile::new(format!("{name}.stark"), source));
        let (ast, pd) = parse(&file, ParseMode::Program);
        assert!(pd.is_empty(), "{name}: parse: {pd:?}");
        let (hir, rd) = resolve(&ast, file.clone());
        assert!(rd.is_empty(), "{name}: resolve: {rd:?}");
        let checked = typecheck::analyze(&hir, file);
        assert!(
            checked
                .diagnostics
                .iter()
                .any(|d| d.code.as_deref() == Some("E0105")),
            "{name}: expected an E0105 refusal, got {:?}",
            checked.diagnostics
        );
    }
}
