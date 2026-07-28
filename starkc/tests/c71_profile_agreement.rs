//! WP-C7.1 §3.2/§3.6 — debug and release must observe identically.
//!
//! Release is a **fourth execution mode**, not a faster third one. The order of authority is
//! specification → HIR oracle → unoptimised MIR → optimised/native → measured performance, and an
//! optimisation that changes an observation is invalid however fast it is. So the same corpus that
//! establishes three-engine agreement is run again with the native engine built `--release`, and
//! the same §39 observation fields are compared.
//!
//! **What this is really guarding.** The generated crate's `[profile.release]` overrides Cargo's
//! defaults — `panic = "abort"` most importantly, since Cargo's release default is `"unwind"` and
//! unwinding runs destructors, which DROP-ABORT-001 forbids after a trap. An override written into
//! a manifest is a *claim* until two profiles are compared, and this is that comparison.
//!
//! The C7.0 baseline established two invariants that make the comparison narrower than it looks,
//! and both are properties of the code rather than of any build setting:
//!
//! * overflow trapping does not depend on the profile — arithmetic lowers to explicit `checked_*`,
//!   so `overflow-checks = false` could not have made it wrap;
//! * trap paths end in `process::exit(101)`, which does not unwind.
//!
//! Those are why release was expected to agree. The point of a differential suite is that
//! "expected to agree" is not evidence.

mod support;

use starkc::backend::generated_rust::Profile;
use support::differential::{
    canonical_form, first_difference, front_end, run_native_with_profile, rustc_available,
    Observation,
};

/// Build one source under BOTH native profiles and require identical observations.
fn profiles_agree(tag: &str, source: &str) -> Observation {
    let name = format!("c71_{tag}.stark");
    let front = front_end(&name, source);
    let program = starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone())
        .unwrap_or_else(|e| panic!("{name}: lowering failed: {}", e.what));
    let debug = run_native_with_profile(&name, &format!("{tag}_d"), &program, Profile::Debug);
    let release = run_native_with_profile(&name, &format!("{tag}_r"), &program, Profile::Release);
    if let Some(field) = first_difference(&debug, &release) {
        panic!(
            "{name}: DEBUG/RELEASE DISAGREEMENT on {field}\n--- debug ---\n{}\n--- release ---\n{}",
            canonical_form(&debug),
            canonical_form(&release)
        );
    }
    debug
}

fn check(tag: &str, source: &str) {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    profiles_agree(tag, source);
}

// --- the semantic dimensions §3.2 enumerates ---

#[test]
fn arithmetic_and_control_flow_agree() {
    check(
        "arith",
        "fn main() {\n    let mut total: Int32 = 0;\n    let mut i: Int32 = 0;\n\
         while i < 50 { if i % 3 == 0 { total = total + i; } i = i + 1; }\n\
         print(total);\n}\n",
    );
}

/// The one an unchecked-arithmetic release build would break. `overflow-checks = false` is Cargo's
/// release default; STARK traps here regardless because the lowering is explicit.
#[test]
fn integer_overflow_traps_in_release_too() {
    check(
        "overflow",
        "fn main() {\n    print(\"before\");\n    let a: Int32 = 2147483647;\n\
         let b: Int32 = a + 1;\n    print(b);\n}\n",
    );
}

#[test]
fn division_by_zero_traps_in_release_too() {
    check(
        "divzero",
        "fn main() {\n    let z: Int32 = 0;\n    print(1 / z);\n}\n",
    );
}

#[test]
fn index_out_of_bounds_traps_in_release_too() {
    check(
        "indexoob",
        "fn main() {\n    let xs: [Int32; 2] = [1, 2];\n    let i: Int32 = 5;\n    print(xs[i]);\n}\n",
    );
}

/// Trap PROVENANCE, not just the category: an optimiser that moved or merged the trapping
/// expression would change the reported line while still trapping.
#[test]
fn trap_source_location_agrees() {
    check(
        "provenance",
        "fn main() {\n    print(\"a\");\n    let z: Int32 = 0;\n    let q: Int32 = 7 / z;\n\
         print(q);\n}\n",
    );
}

#[test]
fn panic_message_agrees() {
    check(
        "panicmsg",
        "fn main() {\n    print(\"before\");\n    panic(\"deliberate\");\n}\n",
    );
}

/// DROP-ABORT-001 under release: destructors must NOT run after a trap. This is the case that
/// `panic = "unwind"` would break if the manifest had inherited Cargo's default.
#[test]
fn no_destructor_runs_after_a_trap_in_release() {
    check(
        "dropabort",
        "struct Loud { id: Int32 }\n\
         impl Drop for Loud {\n    fn drop(&mut self) {\n        print(\"@@stark-drop:Loud#\");\n\
         print(self.id);\n        println(\"@@\");\n    }\n}\n\
         fn main() {\n    let held: Loud = Loud { id: 1 };\n    print(\"before\");\n\
         let z: Int32 = 0;\n    print(1 / z);\n}\n",
    );
}

/// Drop ORDER on the normal path — reverse declaration order, in both profiles.
#[test]
fn drop_order_agrees() {
    check(
        "droporder",
        "struct Loud { id: Int32 }\n\
         impl Drop for Loud {\n    fn drop(&mut self) {\n        print(\"@@stark-drop:Loud#\");\n\
         print(self.id);\n        println(\"@@\");\n    }\n}\n\
         fn main() {\n    let a: Loud = Loud { id: 1 };\n    let b: Loud = Loud { id: 2 };\n\
         let c: Loud = Loud { id: 3 };\n    println(\"end\");\n}\n",
    );
}

#[test]
fn generics_and_trait_dispatch_agree() {
    check(
        "generics",
        "trait Shape { fn area(&self) -> Int32; }\nstruct Sq { s: Int32 }\n\
         impl Shape for Sq { fn area(&self) -> Int32 { self.s * self.s } }\n\
         fn total<T: Shape>(t: T) -> Int32 { t.area() }\n\
         fn main() { print(total(Sq { s: 7 })); }\n",
    );
}

/// User-defined `Eq` must remain authoritative in release — a release build that let Rust's
/// structural equality stand in would give a different length.
#[test]
fn user_defined_equality_stays_authoritative_in_release() {
    check(
        "usereq",
        "struct Tag { id: Int32, note: Int32 }\n\
         impl Eq for Tag { fn eq(&self, other: &Tag) -> Bool { self.id == other.id } }\n\
         impl Hash for Tag { fn hash(&self) -> UInt64 { 7u64 } }\n\
         fn main() {\n    let mut s: HashSet<Tag> = HashSet::new();\n\
         s.insert(Tag { id: 1, note: 100 });\n    s.insert(Tag { id: 1, note: 999 });\n\
         println(s.len());\n}\n",
    );
}

/// Collection ITERATION ORDER is normative (first-insertion), and must survive optimisation.
#[test]
fn collection_iteration_order_agrees() {
    check(
        "iterorder",
        "fn main() {\n    let mut s: HashSet<Int32> = HashSet::new();\n\
         s.insert(30); s.insert(10); s.insert(20); s.insert(10);\n\
         for v in s.iter() { print(*v); print(\",\"); }\n    println(\"\");\n}\n",
    );
}

/// DEV-119's lifecycle shape, carried into release per §7: the cursor's borrow must end with the
/// loop in both profiles.
#[test]
fn dev119_post_loop_mutation_agrees_in_release() {
    check(
        "dev119",
        "fn main() {\n    let mut v: Vec<Int32> = Vec::new();\n    v.push(1);\n\
         for x in v.iter() { print(*x); }\n    println(\"\");\n    v.push(2);\n\
         println(v.len());\n}\n",
    );
}

/// DEV-117's reinitialisation shape, also carried into release.
#[test]
fn dev117_reinitialisation_agrees_in_release() {
    check(
        "dev117",
        "fn take(s: String) -> UInt64 { s.len() }\n\
         fn main() {\n    let mut slot: String = String::from(\"ab\");\n\
         assert_eq(take(slot), 2u64);\n    slot = String::from(\"cde\");\n\
         println(take(slot));\n}\n",
    );
}

/// PROC-EXIT-001: the entry's return value becomes the exit status, in both profiles.
#[test]
fn entry_exit_status_agrees() {
    check("exitstatus", "fn main() -> Int32 {\n    3\n}\n");
}
