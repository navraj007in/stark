//! WP-C6.3d — HashMap/HashSet KEY IDENTITY (STD-HASH-001), proven across engines.
//!
//! The spec determines key identity "exclusively with lawful `Eq`", using `Hash` only to select
//! candidate buckets. Before CD-133 the MIR interpreter compared keys STRUCTURALLY, so a user `Eq`
//! impl was ignored and MIR silently disagreed with HIR — a divergence on a program that
//! type-checks and runs in both engines, invisible because `HashMap` is absent from the differential
//! corpus. These are the adversarial cases §27 requires, run against every engine that can execute
//! them.
//!
//! CE4 (CD-132, owner): identity is decided by an `Eq`-only scan in first-insertion order; `Hash` is
//! never consulted for map operations, so a `Hash` that violates its law cannot desynchronise the
//! engines from one another.

use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::run_program;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// HIR and MIR agree, and both print `expect`. (Native lands with the C6.3d backend slice; these
/// cases are the semantic contract both interpreters must already satisfy.)
fn agree(tag: &str, src: &str, expect: &str) {
    let file = Arc::new(SourceFile::new(
        format!("c63d_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "{tag} typecheck: {errs:?}");

    let hir_exec = interp::run_with_partial_output(&hir, file.clone(), &checked.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag} HIR: {}", e.message));
    assert_eq!(hir_exec.output.trim(), expect, "{tag}: HIR output");

    let program = lower_program(&hir, &checked.tables, file)
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(
        mir_exec.output.trim(),
        expect,
        "{tag}: MIR output must equal the HIR oracle"
    );
}

/// A key whose `Eq` deliberately ignores field `b`, so `K{1,1}` and `K{1,2}` are the SAME key. The
/// map must hold ONE entry. Structural comparison would say two — this is the case that caught the
/// MIR divergence.
#[test]
fn custom_eq_decides_key_identity() {
    agree(
        "customeq",
        "struct K { a: Int32, b: Int32 }\n\
         impl Eq for K { fn eq(&self, other: &K) -> Bool { self.a == other.a } }\n\
         impl Hash for K { fn hash(&self) -> UInt64 { 7u64 } }\n\
         fn main() {\n\
           let mut m: HashMap<K, Int32> = HashMap::new();\n\
           m.insert(K { a: 1, b: 1 }, 10);\n\
           m.insert(K { a: 1, b: 2 }, 20);\n\
           println(m.len());\n\
         }",
        "1",
    );
}

/// STD-HASH-001: replacing an equal key updates the VALUE and keeps the position; the originally
/// stored key is retained. `b` stays 1 even though the second insert supplied `b = 2`.
#[test]
fn replacement_retains_the_first_stored_key() {
    agree(
        "retainkey",
        "struct K { a: Int32, b: Int32 }\n\
         impl Eq for K { fn eq(&self, other: &K) -> Bool { self.a == other.a } }\n\
         impl Hash for K { fn hash(&self) -> UInt64 { 7u64 } }\n\
         fn main() {\n\
           let mut m: HashMap<K, Int32> = HashMap::new();\n\
           m.insert(K { a: 1, b: 1 }, 10);\n\
           m.insert(K { a: 1, b: 2 }, 20);\n\
           for k in m.keys() { print(k.b); }\n\
           println(\"\");\n\
         }",
        "1",
    );
}

/// TOTAL COLLISION: every key hashes to the same value, so only `Eq` can keep them apart. Unequal
/// keys with equal hashes must remain distinct entries.
#[test]
fn total_hash_collision_keeps_unequal_keys_distinct() {
    agree(
        "collide",
        "struct K { a: Int32 }\n\
         impl Eq for K { fn eq(&self, other: &K) -> Bool { self.a == other.a } }\n\
         impl Hash for K { fn hash(&self) -> UInt64 { 1u64 } }\n\
         fn main() {\n\
           let mut m: HashMap<K, Int32> = HashMap::new();\n\
           m.insert(K { a: 1 }, 10);\n\
           m.insert(K { a: 2 }, 20);\n\
           m.insert(K { a: 3 }, 30);\n\
           println(m.len());\n\
         }",
        "3",
    );
}

/// A custom `Eq` also decides `get` and `contains_key`, not just `insert`.
#[test]
fn custom_eq_decides_lookup() {
    agree(
        "lookup",
        "struct K { a: Int32, b: Int32 }\n\
         impl Eq for K { fn eq(&self, other: &K) -> Bool { self.a == other.a } }\n\
         impl Hash for K { fn hash(&self) -> UInt64 { 7u64 } }\n\
         fn main() {\n\
           let mut m: HashMap<K, Int32> = HashMap::new();\n\
           m.insert(K { a: 1, b: 1 }, 10);\n\
           let q1 = K { a: 1, b: 99 }; print(m.contains_key(&q1));\n\
           let q2 = K { a: 2, b: 1 }; print(m.contains_key(&q2));\n\
           println(\"\");\n\
         }",
        "truefalse",
    );
}

/// CD-009 iteration order survives a custom `Eq`: first insertion appends, replacement keeps the
/// position, and remove-then-reinsert moves the key to the END.
#[test]
fn insertion_order_with_custom_eq() {
    agree(
        "order",
        "struct K { a: Int32 }\n\
         impl Eq for K { fn eq(&self, other: &K) -> Bool { self.a == other.a } }\n\
         impl Hash for K { fn hash(&self) -> UInt64 { 3u64 } }\n\
         fn main() {\n\
           let mut m: HashMap<K, Int32> = HashMap::new();\n\
           m.insert(K { a: 1 }, 1);\n\
           m.insert(K { a: 2 }, 2);\n\
           m.insert(K { a: 3 }, 3);\n\
           m.insert(K { a: 1 }, 9);\n\
           for k in m.keys() { print(k.a); }\n\
           println(\"\");\n\
         }",
        "123",
    );
}

/// A primitive key keeps working: it has no user impl, and its structural comparison IS its lawful
/// `Eq`, so the dispatching path must not disturb it.
#[test]
fn primitive_keys_are_unaffected() {
    agree(
        "prim",
        "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); \
         m.insert(1, 10); m.insert(2, 20); m.insert(1, 30); \
         print(m.len()); let q: Int32 = 2; print(m.contains_key(&q)); println(\"\"); }",
        "2true",
    );
}
