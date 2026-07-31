//! WP-C7.9 Packet I — `HashMap`/`HashSet` key bounds (DEV-118).
//!
//! **Why this omission was the dangerous kind.** `06-Standard-Library.md` declares
//! `HashMap<K: Hash + Eq, V>` and `HashSet<T: Hash + Eq>`, and neither half was enforced. Nothing
//! caught it, and nothing *could*: the current storage scans by `Eq` and never consults a hash, so
//! all three engines accepted the same invalid instantiations and produced the same answers. A
//! differential comparator sees agreement and reports success — consistency is not conformance, and
//! this is the case that proves the difference matters.
//!
//! It also has a fuse on it. The moment any one implementation starts using the hash — a real hash
//! table in the native runtime, say — a program whose key type has no `Hash` becomes a live
//! divergence between engines, in code that compiled cleanly for months.
//!
//! The enforcement point is **type instantiation**, not `insert`. `HashMap<Float64, Int32>` is
//! ill-typed wherever it is written, including in a signature nobody ever calls.

mod support;

use support::differential::rejects_at_typecheck;

fn reject(name: &str, source: &str) {
    rejects_at_typecheck(&format!("{name}.stark"), source, "E0500");
}

fn accept(name: &str, source: &str) {
    let file = std::sync::Arc::new(starkc::source::SourceFile::new(
        format!("{name}.stark"),
        source.to_string(),
    ));
    let (ast, pd) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    assert!(pd.is_empty(), "{name}: parse: {pd:?}");
    let (hir, rd) = starkc::resolve::resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{name}: resolve: {rd:?}");
    let checked = starkc::typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "{name}: a valid key type was rejected: {errors:?}"
    );
}

const HASH_ONLY: &str = "struct HashOnly { v: Int32 }\n\
                         impl Hash for HashOnly { fn hash(&self) -> UInt64 { 1u64 } }\n";

const EQ_ONLY: &str = "struct EqOnly { v: Int32 }\n\
                       impl Eq for EqOnly { fn eq(&self, o: &EqOnly) -> Bool { self.v == o.v } }\n";

const BOTH: &str = "struct Key { v: Int32 }\n\
                    impl Eq for Key { fn eq(&self, o: &Key) -> Bool { self.v == o.v } }\n\
                    impl Hash for Key { fn hash(&self) -> UInt64 { 1u64 } }\n";

// ------------------------------------------------------------------------- rejections --

/// A key with `Eq` but no `Hash`. The half DEV-118 named explicitly.
#[test]
fn a_key_without_hash_is_rejected() {
    reject(
        "eq_no_hash",
        &format!("{EQ_ONLY}fn main() {{ let m: HashMap<EqOnly, Int32> = HashMap::new(); }}\n"),
    );
    reject(
        "set_eq_no_hash",
        &format!("{EQ_ONLY}fn main() {{ let s: HashSet<EqOnly> = HashSet::new(); }}\n"),
    );
}

/// A key with `Hash` but no `Eq`. Hashing alone cannot decide identity — two distinct keys may
/// collide — so `Eq` is not optional.
#[test]
fn a_key_without_eq_is_rejected() {
    reject(
        "hash_no_eq",
        &format!("{HASH_ONLY}fn main() {{ let m: HashMap<HashOnly, Int32> = HashMap::new(); }}\n"),
    );
}

/// Floating-point keys. They have no `Eq` in this language for the reason that makes them bad keys:
/// `NaN != NaN`, so a float key could never be found again.
#[test]
fn floating_point_keys_are_rejected() {
    reject(
        "float_key",
        "fn main() { let m: HashMap<Float64, Int32> = HashMap::new(); }\n",
    );
    reject(
        "float32_set",
        "fn main() { let s: HashSet<Float32> = HashSet::new(); }\n",
    );
}

/// The obligation is checked where the type is WRITTEN, so a signature that never runs is still
/// ill-typed. An enforcement bolted onto `insert` would accept this file entirely.
#[test]
fn an_invalid_key_type_in_an_uncalled_signature_is_rejected() {
    reject(
        "uncalled_signature",
        "fn never_called(m: HashMap<Float64, Int32>) -> UInt64 { m.len() }\nfn main() { }\n",
    );
}

/// A generic function whose own bounds are insufficient cannot instantiate the map: `T: Eq` alone
/// does not discharge `Hash`.
#[test]
fn a_generic_function_with_insufficient_bounds_is_rejected() {
    reject(
        "insufficient_generic_bounds",
        "fn build<T: Eq>() -> HashMap<T, Int32> { HashMap::new() }\nfn main() { }\n",
    );
}

/// A nested invalid key: the obligation applies to the type as written, at every position where a
/// map or set is instantiated.
#[test]
fn a_nested_invalid_key_type_is_rejected() {
    reject(
        "nested_invalid",
        "fn main() { let outer: Vec<HashMap<Float64, Int32>> = Vec::new(); }\n",
    );
}

// -------------------------------------------------------------------------- acceptances --

/// Primitives that normatively satisfy both bounds keep working — this is most of the corpus, and a
/// rule that broke them would be worse than the omission it replaced.
#[test]
fn primitive_keys_are_accepted() {
    accept(
        "int_key",
        "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 2); \
         println(m.len()); }\n",
    );
    accept(
        "string_key",
        "fn main() { let mut m: HashMap<String, Int32> = HashMap::new(); \
         m.insert(String::from(\"a\"), 1); println(m.len()); }\n",
    );
    accept(
        "char_set",
        "fn main() { let mut s: HashSet<Char> = HashSet::new(); s.insert('a'); \
         println(s.len()); }\n",
    );
}

/// A user nominal with both impls is a valid key — the case the corpus's own identity sentinels
/// depend on.
#[test]
fn a_user_key_with_both_impls_is_accepted() {
    accept(
        "user_key",
        &format!(
            "{BOTH}fn main() {{ let mut m: HashMap<Key, Int32> = HashMap::new(); \
             m.insert(Key {{ v: 1 }}, 2); println(m.len()); }}\n"
        ),
    );
}

/// A generic function that declares both bounds discharges the obligation — the same way a written
/// bound on any other generic type is discharged.
#[test]
fn a_generic_function_with_both_bounds_is_accepted() {
    accept(
        "sufficient_generic_bounds",
        "fn build<T: Hash + Eq>() -> HashMap<T, Int32> { HashMap::new() }\n\
         fn main() { let m: HashMap<Int32, Int32> = build(); println(m.len()); }\n",
    );
}

/// The VALUE position carries no obligation: only keys are hashed and compared.
#[test]
fn the_value_position_is_unconstrained() {
    accept(
        "float_value",
        "fn main() { let mut m: HashMap<Int32, Float64> = HashMap::new(); m.insert(1, 0.5f64); \
         println(m.len()); }\n",
    );
}
