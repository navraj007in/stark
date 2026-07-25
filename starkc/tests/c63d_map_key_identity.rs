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

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions};
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

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// All three engines agree and print `expect`.
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

    if rustc_available() {
        let verified = verify_program(&program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_c63d_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let artifact = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: dir.clone(),
                target_contract: "stark-64-v1".to_string(),
            },
        )
        .unwrap_or_else(|e| panic!("{tag} native build: {e:?}"));
        let run = std::process::Command::new(&artifact.binary_path)
            .output()
            .expect("run");
        assert!(run.status.success(), "{tag}: native must exit 0");
        assert_eq!(
            String::from_utf8_lossy(&run.stdout).trim(),
            expect,
            "{tag}: native output must equal the oracle"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
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

// ---- The rest of the §27 matrix: what is native, and what is NOT a native gap ----

/// A `String` key: no user impl, so identity is the structural comparator — and it must work
/// natively alongside the dispatched-`Eq` path.
#[test]
fn string_keys_work() {
    agree(
        "stringkey",
        "fn main() { let mut m: HashMap<String, Int32> = HashMap::new(); \
         m.insert(String::from(\"a\"), 1); m.insert(String::from(\"b\"), 2); \
         m.insert(String::from(\"a\"), 3); println(m.len()); }",
        "2",
    );
}

/// The program type-checks and the HIR interpreter runs it, but LOWERING refuses it — an HIR-only
/// shape, i.e. a lowering gap rather than a native one.
fn hir_only(tag: &str, src: &str) {
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
    assert!(
        errs.is_empty(),
        "{tag}: expected it to type-check, got {errs:?}"
    );
    interp::run_with_partial_output(&hir, file.clone(), &checked.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag}: HIR should run it: {}", e.message));
    assert!(
        lower_program(&hir, &checked.tables, file).is_err(),
        "{tag}: lowering is expected to refuse this; if it now lowers, promote it to a three-engine case"
    );
}

/// `HashSet` has NO MIR representation at all (`Core(HashSet, …)` is refused at lowering), so it is
/// HIR-only. Implementing it — even as "HashMap to Unit" — is a LOWERING feature, not a backend one:
/// neither MIR nor native can represent it, so there is no native divergence for this gate to close.
/// Same category, and the same precedent, as C6.3c's adapter iterators.
#[test]
fn hashset_is_hir_only() {
    hir_only(
        "hashset",
        "fn main() { let mut s: HashSet<Int32> = HashSet::new(); s.insert(1); s.insert(1); \
         println(s.len()); }",
    );
}

/// CE4 (CD-132): user-`Drop` keys and values stay REFUSED before MIR, which is what keeps entry
/// Drop order unobservable and therefore legitimately unspecified. Both positions are refused.
#[test]
fn drop_bearing_keys_and_values_are_refused() {
    for (tag, src) in [
        (
            "dropvalue",
            "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) {} }\n\
             fn main() { let mut m: HashMap<Int32, D> = HashMap::new(); \
             m.insert(1, D { v: 1 }); println(m.len()); }",
        ),
        (
            "dropkey",
            "struct K { v: Int32 }\nimpl Drop for K { fn drop(&mut self) {} }\n\
             impl Eq for K { fn eq(&self, other: &K) -> Bool { self.v == other.v } }\n\
             impl Hash for K { fn hash(&self) -> UInt64 { 1u64 } }\n\
             fn main() { let mut m: HashMap<K, Int32> = HashMap::new(); \
             m.insert(K { v: 1 }, 1); println(m.len()); }",
        ),
    ] {
        hir_only(tag, src);
    }
}
