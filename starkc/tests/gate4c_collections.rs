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

#[test]
fn test_iterator_combinators() {
    let source = "
        fn is_even(x: &Int32) -> Bool {
            let val = *x;
            val % 2 == 0
        }
        fn double(x: &Int32) -> Int32 {
            let val = *x;
            val * 2
        }
        fn add(acc: Int32, x: Int32) -> Int32 {
            acc + x
        }
        fn is_greater_than_five(x: &Int32) -> Bool {
            let val = *x;
            val > 5
        }
        fn is_b(ch: &Char) -> Bool {
            let val = *ch;
            val == 'b'
        }
        fn is_odd(x: &Int32) -> Bool {
            let val = *x;
            val % 2 != 0
        }

        fn main() {
            let mut set: HashSet<Int32> = HashSet::new();
            set.insert(1);
            set.insert(2);
            set.insert(3);
            set.insert(4);
            set.insert(5);

            // count
            let mut it = set.iter();
            println(it.count());

            // map, filter, fold
            let mut it2 = set.iter();
            let mut mapped = it2.map(double);
            let mut filtered = mapped.filter(is_greater_than_five);
            let result = filtered.fold(0, add);
            println(result); // 6 + 8 + 10 = 24

            // any, all
            let mut it3 = set.iter();
            println(it3.any(is_odd)); // true
            
            let mut it4 = set.iter();
            println(it4.all(is_odd)); // false

            // find
            let mut s = String::from(\"abc\");
            let mut it5 = s.chars();
            match it5.find(is_b) {
                Some(val) => println(val),
                None => println('x'),
            } // b

            // collect
            let mut it6 = set.iter();
            let mut vec: Vec<Int32> = it6.collect();
            println(vec.len());
        }
    ";
    let output = execute_snippet(source);
    assert_eq!(output, "5\n24\ntrue\nfalse\nb\n5\n");
}
