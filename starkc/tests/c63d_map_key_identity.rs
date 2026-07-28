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

mod support;

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

/// Delegates to the shared comparator (R-02), keeping this suite's independent stdout pin: three
/// engines agreeing on the wrong map contents still fails.
fn agree(tag: &str, src: &str, expect: &str) {
    support::differential::agree_completing_with_stdout(tag, src, &format!("{}\n", expect.trim()));
}

/// All three engines agree and print `expect`.
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

/// **DEV-116 is FIXED (CD-176), so this no longer asserts "HIR only".**
///
/// It asserted that `HashSet` ran in the oracle and was refused by MIR — which was true, and was a
/// MIR/native lowering gap rather than a Core exclusion, so §4.3 forbade recording it as one. V19
/// is now covered by the `dev116__hashset_*` corpus cases, replayed through all three engines.
///
/// What survives here is the part that would otherwise be lost: proof the set agrees with the MAP
/// on identity, in the suite that owns key-identity semantics. Same element type, same user `Eq`,
/// same expectation.
#[test]
fn hashset_agrees_with_hashmap_on_user_defined_identity() {
    agree(
        "hashset_identity",
        "struct K { v: Int32 }\n\
         impl Eq for K { fn eq(&self, other: &K) -> Bool { self.v == other.v } }\n\
         impl Hash for K { fn hash(&self) -> UInt64 { 1u64 } }\n\
         fn main() {\n\
             let mut s: HashSet<K> = HashSet::new();\n\
             s.insert(K { v: 1 });\n\
             s.insert(K { v: 1 });\n\
             let mut m: HashMap<K, Int32> = HashMap::new();\n\
             m.insert(K { v: 1 }, 10);\n\
             m.insert(K { v: 1 }, 20);\n\
             print(s.len());\n\
             print(\"|\");\n\
             println(m.len());\n\
         }\n",
        "1|1",
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

// ---------------------------------------------------------------------------------------------
// CD-138 — the ADVERSARIAL `Eq` cases. CD-133 proved user `Eq` is CONSULTED; these pin HOW.
//
// An `Eq` impl is arbitrary user code, so a map operation that calls it has three observable
// properties beyond its answer: which operand is `self`, how many times and in what order it is
// called, and what happens when it diverges. Each is a real behaviour the engines must share, and
// none is implied by "identity is decided by `Eq`" — an implementation could satisfy that sentence
// with either argument order, any scan order, and any panic provenance.
//
// These use ILLEGAL `Eq` impls on purpose (asymmetric, effectful, diverging). STD-HASH-001 requires
// a LAWFUL `Eq` for the map's own guarantees, and a program breaking that law forfeits those. It
// does NOT forfeit engine agreement: the three engines must still agree with one another, because
// disagreement is a COMPILER defect regardless of how ill-behaved the program is. That is what is
// being tested — determinism under adversarial input, not lawfulness.
// ---------------------------------------------------------------------------------------------

/// **Argument direction.** `eq(&self, other: &K)` is asymmetric here, so the answer reveals which
/// operand the map passes as `self`: the STORED key or the PROBE.
///
/// `self.a == other.a + 1` holds for `stored.eq(probe)` with stored `a = 1` and probe `a = 0`, and
/// fails for `probe.eq(stored)`. The engines must not merely agree by accident — a comparator is a
/// two-argument call, and MIR passing `(stored, probe)` while native passes `(probe, stored)` is
/// invisible for every lawful (symmetric) `Eq` and silently divergent for this one.
#[test]
fn eq_receives_the_stored_key_as_self() {
    agree(
        "eqdirection",
        "struct K { a: Int32 }\n\
         impl Eq for K { fn eq(&self, other: &K) -> Bool { self.a == other.a + 1 } }\n\
         impl Hash for K { fn hash(&self) -> UInt64 { 1u64 } }\n\
         fn main() {\n\
           let mut m: HashMap<K, Int32> = HashMap::new();\n\
           m.insert(K { a: 1 }, 10);\n\
           let probe = K { a: 0 };\n\
           println(m.contains_key(&probe));\n\
         }",
        "true",
    );
}

/// **Call order and call count.** The comparator PRINTS, so the output IS the scan trace.
///
/// Every key collides on `Hash`, so `Eq` alone separates them and the scan cannot be short-circuited
/// by bucketing. CD-132 fixed the scan as first-insertion order over all entries; that makes the
/// number of calls and their sequence observable, and therefore something the engines must share.
/// An engine scanning in reverse, or caching a comparison, or comparing an entry twice, prints a
/// different trace while still reporting the same `true`.
#[test]
fn eq_call_order_and_count_are_observable() {
    agree(
        "eqtrace",
        "struct K { a: Int32 }\n\
         impl Eq for K { fn eq(&self, other: &K) -> Bool { \
           print(self.a); print(\":\"); print(other.a); print(\" \"); self.a == other.a } }\n\
         impl Hash for K { fn hash(&self) -> UInt64 { 1u64 } }\n\
         fn main() {\n\
           let mut m: HashMap<K, Int32> = HashMap::new();\n\
           m.insert(K { a: 1 }, 10);\n\
           m.insert(K { a: 2 }, 20);\n\
           m.insert(K { a: 3 }, 30);\n\
           let probe = K { a: 3 };\n\
           println(m.contains_key(&probe));\n\
         }",
        "1:2 1:3 2:3 1:3 2:3 3:3 true",
    );
}

/// **Equal keys, INCONSISTENT hashes.** `Eq` says equal; `Hash` says different. A hash-bucketed map
/// would place them in different buckets and report two entries — the classic silent-corruption
/// shape. CD-132 makes `Hash` unreachable for map operations, so the map must report ONE entry.
///
/// This is the converse of `total_hash_collision_keeps_unequal_keys_distinct`: together they show
/// `Hash` cannot affect the answer in either direction.
#[test]
fn equal_keys_with_differing_hashes_are_one_entry() {
    agree(
        "eqbadhash",
        "struct K { a: Int32, h: UInt64 }\n\
         impl Eq for K { fn eq(&self, other: &K) -> Bool { self.a == other.a } }\n\
         impl Hash for K { fn hash(&self) -> UInt64 { self.h } }\n\
         fn main() {\n\
           let mut m: HashMap<K, Int32> = HashMap::new();\n\
           m.insert(K { a: 1, h: 1u64 }, 10);\n\
           m.insert(K { a: 1, h: 99999u64 }, 20);\n\
           println(m.len());\n\
           let probe = K { a: 1, h: 7u64 };\n\
           println(m.contains_key(&probe));\n\
         }",
        "1\ntrue",
    );
}

/// **A diverging comparator.** `panic` inside `Eq` aborts from inside a runtime map operation — the
/// one place where user code runs UNDER the runtime rather than above it. The abort must report the
/// user's `panic` message and the user's source location, not a runtime frame: the comparator is a
/// normal generated function that the runtime happens to call, so it must not acquire different
/// provenance by being reached that way (DEV-107's requirement, applied to this path).
///
/// Output printed before the panic must still reach stdout (CD-120 Contract B), which also proves
/// the abort came from INSIDE the second insert — the first cannot compare anything.
#[test]
fn an_eq_that_panics_aborts_with_the_users_provenance() {
    let src = "struct K { a: Int32 }\n\
         impl Eq for K { fn eq(&self, other: &K) -> Bool { panic(\"eq exploded\") } }\n\
         impl Hash for K { fn hash(&self) -> UInt64 { 1u64 } }\n\
         fn main() {\n\
           let mut m: HashMap<K, Int32> = HashMap::new();\n\
           m.insert(K { a: 1 }, 10);\n\
           println(\"before\");\n\
           m.insert(K { a: 2 }, 20);\n\
           println(\"unreachable\");\n\
         }";
    let file = Arc::new(SourceFile::new(
        "c63d_eqpanic.stark".to_string(),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "typecheck: {errs:?}");

    let (hir_err, hir_out) = interp::run_with_partial_output(&hir, file.clone(), &checked.tables)
        .expect_err("HIR: the comparator must abort the program");
    assert!(
        hir_err.message.contains("eq exploded"),
        "HIR: the user's panic message must survive: {}",
        hir_err.message
    );
    assert_eq!(hir_out, "before\n", "HIR: output before the panic");

    let program =
        lower_program(&hir, &checked.tables, file).unwrap_or_else(|e| panic!("lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let mir_fail = match run_program(verified) {
        Err(f) => f,
        Ok(_) => panic!("MIR: the comparator must abort the program"),
    };
    assert!(
        matches!(
            &mir_fail.error,
            starkc::mir::interp::MirRunError::Trap { message: Some(m), .. } if m.contains("eq exploded")
        ),
        "MIR: the user's panic message must survive: {:?}",
        mir_fail.error
    );
    assert_eq!(mir_fail.output, "before\n", "MIR: output before the panic");

    if support::differential::rustc_available() {
        let verified = verify_program(&program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_c63d_eqpanic_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let artifact = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: dir.clone(),
                target_contract: "stark-64-v1".to_string(),
            },
        )
        .unwrap_or_else(|e| panic!("native build: {e:?}"));
        let run = std::process::Command::new(&artifact.binary_path)
            .output()
            .expect("run");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(run.status.code(), Some(101), "native: abort exit code");
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(
            stderr.contains("eq exploded"),
            "native: the user's panic message must survive: {stderr}"
        );
        assert!(
            stderr.contains("c63d_eqpanic.stark:2:"),
            "native: the abort must name the `panic` inside the user's `eq`, line 2: {stderr}"
        );
        assert_eq!(
            String::from_utf8_lossy(&run.stdout),
            "before\n",
            "native: output before the abort (CD-120 Contract B)"
        );
    }
}

// ---- MUTATION EVIDENCE: `TypeContext::eq_impls` is load-bearing, and its absence is caught. ----

/// Compile `src` to unverified MIR.
fn lower_only(tag: &str, src: &str) -> starkc::mir::MirProgram {
    let file = Arc::new(SourceFile::new(
        format!("c63d_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    lower_program(&hir, &checked.tables, file).unwrap_or_else(|e| panic!("{tag} lower: {}", e.what))
}

/// Two key types, each USED as a map key so lowering records both `Eq` instances. `J`'s comparator
/// is unconditionally `true`; `K`'s is honest. The `K` map holds two distinct keys.
const TWO_KEYS: &str = "struct K { a: Int32 }\n\
     impl Eq for K { fn eq(&self, other: &K) -> Bool { self.a == other.a } }\n\
     impl Hash for K { fn hash(&self) -> UInt64 { 1u64 } }\n\
     struct J { a: Int32 }\n\
     impl Eq for J { fn eq(&self, other: &J) -> Bool { true } }\n\
     impl Hash for J { fn hash(&self) -> UInt64 { 1u64 } }\n\
     fn main() {\n\
       let mut m: HashMap<K, Int32> = HashMap::new();\n\
       m.insert(K { a: 1 }, 10);\n\
       m.insert(K { a: 2 }, 20);\n\
       let mut j: HashMap<J, Int32> = HashMap::new();\n\
       j.insert(J { a: 1 }, 10);\n\
       println(m.len());\n\
     }";

/// CD-138: REMOVING the entry must be caught by VERIFICATION, ahead of either engine.
///
/// This is the mutation that motivated the check. Before it, the two engines reacted to a missing
/// entry differently — the MIR interpreter fell back to STRUCTURAL equality and produced an answer,
/// while the backend refused to emit. A defect that yields a plausible-looking number in one engine
/// and a build failure in the other is worse than either, and the fallback was the dangerous half.
/// Now the program is rejected before it reaches either.
#[test]
fn a_missing_eq_instance_fails_verification() {
    let mut program = lower_only("mutclear", TWO_KEYS);
    assert!(
        verify_program(&program).is_ok(),
        "the unmutated program must verify, or this mutation proves nothing"
    );
    program.types.eq_impls.clear();
    let errors = match verify_program(&program) {
        Err(e) => e,
        Ok(_) => panic!("a missing `Eq` instance must be rejected"),
    };
    assert!(
        errors.iter().any(|e| e.code == "MIR-0018"),
        "expected MIR-0018, got: {errors:?}"
    );
}

/// CD-138: MISROUTING the entry must change the ANSWER.
///
/// `J`'s `eq` returns `true` unconditionally. Pointing `K`'s entry at it collapses two distinct keys
/// into one, so `len()` reads 1 instead of 2. This is the half the verifier CANNOT catch — a wrong
/// symbol is still a symbol — and it is why the table has to be the single source both engines read
/// rather than something each reconstructs: if the MIR interpreter re-derived the comparator instead
/// of consuming this table, this mutation would move native and leave MIR behind, which is exactly
/// the divergence CD-133 found in the field.
#[test]
fn misrouting_the_eq_instance_changes_the_answer() {
    let program = lower_only("mutswap", TWO_KEYS);
    let baseline = run_program(verify_program(&program).unwrap()).expect("baseline runs");
    assert_eq!(
        baseline.output.trim(),
        "2",
        "baseline: two distinct `K` keys"
    );

    // EXCHANGE the two recorded comparators. Swapping symmetrically avoids having to identify `K`
    // by its mangled name: whichever entry the `K` map reads now holds the other type's `eq`, and
    // `J`'s is unconditionally `true`, so two distinct keys collapse into one.
    let mut mutated = program.clone();
    let keys: Vec<_> = mutated.types.eq_impls.keys().cloned().collect();
    assert_eq!(keys.len(), 2, "both `Eq` impls must be recorded: {keys:?}");
    let a = mutated.types.eq_impls[&keys[0]].clone();
    let b = mutated.types.eq_impls[&keys[1]].clone();
    assert_ne!(a, b, "the two impls must be distinct symbols");
    mutated.types.eq_impls.insert(keys[0].clone(), b);
    mutated.types.eq_impls.insert(keys[1].clone(), a);

    let mutant = run_program(verify_program(&mutated).unwrap()).expect("mutant runs");
    assert_eq!(
        mutant.output.trim(),
        "1",
        "the substituted comparator must be the one that decided identity"
    );
}
