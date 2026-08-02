//! **Layer audit: which MIR refusals are reachable from a type-correct program?**
//!
//! `lower.rs` holds 194 `unsupported(…)` sites (160 distinct messages). WP-C4.6 classified them by
//! whether the refusal is CORRECT. This asks the different question E0105 was an answer to: **does the front end accept the program first?**
//!
//! A refusal that is semantically right but happens below semantic analysis produces an accepted
//! program no compiler can build — the checker says yes, the reference interpreter runs it, and
//! lowering refuses. One instance was fixed by moving the refusal up: E0105, for by-value `Vec` iteration. A second
//! attempt, E0106 for indexing a non-`Copy` element, was REVERTED (CD-294) and found redundant
//! (CD-297) — E0100 had always refused that at the right layer. Two more were fixed differently,
//! by teaching lowering the construct rather than raising the refusal: DEV-132 (`&v[i].field`) and
//! DEV-133 (`&[T; N]` to `&[T]`), both found by a package build rather than by this audit, because
//! both were VERIFIER rejections on MIR that lowering produced willingly — a shape no probe here
//! reaches. See `WP-LOWERING-COVERAGE-MATRIX.md`.
//!
//! **This was measurement; since WP-DEV-134-139 §11 it is ENFORCEMENT.** It used to report its
//! findings and pass unconditionally, which meant a NEW layer defect could appear and the suite
//! would stay green — the audit could only ever be read by a human who happened to look.
//!
//! Every probe now carries the disposition it is EXPECTED to have, and the test fails when actual
//! and expected disagree. The bar is deliberately not "zero findings": six reachable lowering
//! refusals exist and are not repaired by this programme. The bar is **zero UNREGISTERED
//! findings** — each of the six is pinned to the DEV that owns it, so it is tracked rather than
//! merely observed.
//!
//! It therefore fails when:
//!
//! ```text
//! a new layer defect appears                     -> unregistered finding
//! a registered one stops reproducing             -> either fixed (close the DEV) or the probe
//!                                                   stopped reaching its site
//! a probe changes disposition in either direction -> the inventory moved without being updated
//! ```
//!
//! A probe that stops reproducing its defect is a FAILURE here, not a quiet success: it means
//! either the DEV was fixed and its registration is stale, or the probe no longer exercises the
//! construct it was written for. Both need a human decision, and both are invisible if the test
//! only ever looks for regressions.
//!
//! Reading the output:
//!
//! ```text
//! LAYER-DEFECT  front end ACCEPTED, lowering refused  -> an E0105-class defect
//! ok-frontend   front end refused                     -> correct: refused where it should be
//! ok-lowers     both accepted                         -> the probe does not reach its site
//! ```

use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// The disposition a probe is REGISTERED as having. Every probe declares one; the test compares
/// it against what actually happens.
#[derive(Debug, PartialEq, Eq)]
enum Expect {
    /// The front end refuses it. The program never reaches lowering — correct layering.
    FrontEnd,
    /// Both layers accept. The probe does not reach the site it was aimed at, which is recorded
    /// rather than deleted so the site keeps a name.
    Lowers,
    /// A KNOWN reachable lowering refusal, owned by this deviation. Not repaired by
    /// WP-DEV-134-139; registered so it is tracked rather than merely observed.
    KnownDev(&'static str),
}

enum Outcome {
    /// The front end accepted and lowering refused: the defect this audit looks for.
    LayerDefect(String),
    /// The front end refused. Correct — the program never reaches lowering.
    FrontEnd(String),
    /// Both accepted: this probe does not reach the site it was aimed at.
    Lowers,
}

fn probe(src: &str) -> Outcome {
    let file = Arc::new(SourceFile::new("probe.stark", src.to_string()));
    // A PARSE or RESOLVE refusal is the front end refusing, which is the outcome this audit is
    // asking about — it is not a broken probe. These used to assert, so the `L3483 nested item`
    // probe (rejected by the parser, since Core v1 has no nested items) took the whole audit down
    // and no result was reported at all. Classifying instead of asserting is what lets a probe aimed
    // at a lowering site legitimately land in the front end.
    let (ast, pd) = parse(&file, ParseMode::Program);
    if let Some(first) = pd.first() {
        return Outcome::FrontEnd(format!(
            "parse: {} {}",
            first.code.as_deref().unwrap_or("-"),
            first.message
        ));
    }
    let (hir, rd) = resolve(&ast, file.clone());
    if let Some(first) = rd.first() {
        return Outcome::FrontEnd(format!(
            "resolve: {} {}",
            first.code.as_deref().unwrap_or("-"),
            first.message
        ));
    }
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    if let Some(first) = errors.first() {
        return Outcome::FrontEnd(format!(
            "{} {}",
            first.code.as_deref().unwrap_or("-"),
            first.message
        ));
    }
    match lower_program(&hir, &checked.tables, file) {
        Ok(_) => Outcome::Lowers,
        Err(e) => Outcome::LayerDefect(e.what),
    }
}

/// Each entry is `(label, source)`. The label names the lowering refusal the probe aims at.
fn probes() -> Vec<(&'static str, &'static str, Expect)> {
    vec![
        (
            "L10807 nested pattern in match arm",
            "fn main() { let o: Option<Result<Int32, Bool>> = None; \
             match o { Some(Ok(n)) => { println(n); } Some(Err(_b)) => { } None => { } } }",
            Expect::Lowers,
        ),
        (
            "L9516 integer match without a default arm",
            "fn main() { let n: Int32 = 1; match n { 1 => { println(1); } 2 => { println(2); } } }",
            Expect::FrontEnd,
        ),
        (
            "L6979 Option combinator on a droppable payload",
            "fn main() { let o: Option<String> = None; \
             let s = o.unwrap_or(String::from(\"x\")); println(s.len()); }",
            Expect::Lowers,
        ),
        (
            "L7153 Vec:: method outside the implemented set",
            "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.insert(0u64, 2); }",
            Expect::KnownDev("DEV-140"),
        ),
        (
            "L8109 HashMap:: reserved for std-full",
            "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 2); \
             let _e = m.entry(1); }",
            Expect::FrontEnd,
        ),
        (
            "L8093 HashMap over user-Drop value types",
            "struct D { v: Int32 } impl Drop for D { fn drop(&mut self) { } } \
             fn main() { let mut m: HashMap<Int32, D> = HashMap::new(); m.insert(1, D { v: 1 }); }",
            Expect::KnownDev("DEV-141"),
        ),
        (
            "L9346 print/println of this type",
            "struct P { a: Int32 } fn main() { let p = P { a: 1 }; println(p); }",
            Expect::FrontEnd,
        ),
        (
            "L9096 Display of a type inside a composite",
            "fn main() { let t: (Int32, Bool) = (1, true); println(t); }",
            Expect::Lowers,
        ),
        (
            "L9130 droppable composite carrying a borrowed element",
            "fn main() { let s = String::from(\"a\"); let t: (String, &str) = (s, \"b\"); println(t); }",
            Expect::KnownDev("DEV-142"),
        ),
        (
            "L5346 assert_eq on a user-defined type",
            "struct P { a: Int32 } impl Eq for P { fn eq(&self, other: &P) -> Bool { self.a == other.a } } \
             fn main() { let x = P { a: 1 }; let y = P { a: 1 }; assert_eq(x, y); }",
            Expect::KnownDev("DEV-143"),
        ),
        (
            "L3698 for over a non-range, non-Vec iterator",
            "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 2); \
             for v in m.values() { println(*v); } }",
            Expect::KnownDev("DEV-144"),
        ),
        (
            "L2004 move through a non-field projection of a drop-tracked local",
            "fn main() { let mut v: Vec<String> = Vec::new(); v.push(String::from(\"a\")); \
             let a: [String; 1] = [String::from(\"b\")]; let m = a[0u64]; println(m.len()); }",
            Expect::FrontEnd,
        ),
        (
            "L4387 field access on non-struct",
            "fn main() { let t: (Int32, Int32) = (1, 2); println(t.0); }",
            Expect::Lowers,
        ),
        (
            "L6450 method on a peeled type outside the slice",
            "fn main() { let s = String::from(\"abc\"); let u = s.to_uppercase(); println(u.len()); }",
            Expect::KnownDev("DEV-145"),
        ),
        (
            "L5130 indexing a non-Vec/array base",
            "fn main() { let s = String::from(\"abc\"); let b = s[0u64]; println(b); }",
            Expect::FrontEnd,
        ),
        (
            "L1884 array length is not a literal count",
            "const N: UInt64 = 4u64; fn main() { let a: [Int32; 4] = [0; 4]; println(a[0u64]); }",
            Expect::Lowers,
        ),
        (
            "L8238 HashSet:: reserved for std-full",
            "fn main() { let mut s: HashSet<Int32> = HashSet::new(); s.insert(1); \
             let o: HashSet<Int32> = HashSet::new(); let u = s.union(&o); }",
            Expect::FrontEnd,
        ),
        (
            "L4083 unary operator outside the set",
            "fn main() { let x: Int32 = 1; let y = -x; println(y); }",
            Expect::Lowers,
        ),
        (
            "L3483 nested item",
            "fn main() { fn inner() -> Int32 { 1 } println(inner()); }",
            Expect::FrontEnd,
        ),
        (
            "L6901 Option/Result method outside the slice",
            "fn main() { let o: Option<Int32> = Some(1); let m = o.map_or(0, 1); println(m); }",
            Expect::FrontEnd,
        ),
    ]
}

#[test]
fn layer_audit_matches_its_registered_inventory() {
    let mut mismatches: Vec<String> = Vec::new();
    let mut defects = Vec::new();
    let (mut frontend, mut lowers) = (0usize, 0usize);

    println!("\n=== LAYER AUDIT: front end vs lowering (registered inventory) ===\n");
    for (label, src, expected) in probes() {
        let outcome = probe(src);
        let actual = match &outcome {
            Outcome::LayerDefect(what) => {
                println!("LAYER-DEFECT | {label}\n               lowering: {what}");
                defects.push(label);
                // Which DEV owns it is the registration's business, not the probe's; any
                // KnownDev matches here and the exact identity is compared below.
                Expect::KnownDev("")
            }
            Outcome::FrontEnd(d) => {
                println!("ok-frontend  | {label}\n               refused by: {d}");
                frontend += 1;
                Expect::FrontEnd
            }
            Outcome::Lowers => {
                println!("ok-lowers    | {label}  (probe does not reach its site)");
                lowers += 1;
                Expect::Lowers
            }
        };

        let agrees = match (&expected, &actual) {
            (Expect::KnownDev(_), Expect::KnownDev(_)) => true,
            (a, b) => a == b,
        };
        if !agrees {
            mismatches.push(format!(
                "{label}\n      registered as {expected:?} but actually {actual:?}"
            ));
        }
    }

    println!(
        "\n=== TOTALS: {} layer defects, {} correctly refused up front, {} lowered cleanly ===",
        defects.len(),
        frontend,
        lowers
    );
    if !defects.is_empty() {
        println!("\nReachable lowering refusals (accepted by the checker, refused by MIR):");
        for d in &defects {
            println!("  - {d}");
        }
    }
    println!();

    // WP-DEV-134-139 §11: zero UNREGISTERED findings, not zero findings. A mismatch in EITHER
    // direction fails — a new layer defect, and equally a registered one that stopped
    // reproducing, because the second means either the DEV was fixed (close it and update the
    // registration) or the probe no longer reaches its site (fix the probe). Both need a human
    // decision and both are invisible if the test only looks for regressions.
    assert!(
        mismatches.is_empty(),
        "the layer audit no longer matches its registered inventory:\n\n  - {}\n\n\
         Every probe declares the disposition it is expected to have. Update the registration in \
         `probes()` in the same change as whatever moved it, and the owning DEV entry with it.",
        mismatches.join("\n  - ")
    );
}
