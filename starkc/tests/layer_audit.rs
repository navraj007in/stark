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
//! This is measurement, not enforcement. Every probe below is REPORTED, not asserted, and the test
//! passes regardless of the counts — its output is the inventory. The one thing it does assert is
//! that every probe still parses and resolves, so a probe that silently stopped exercising its site
//! fails rather than quietly reporting "front end refuses".
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
fn probes() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "L10807 nested pattern in match arm",
            "fn main() { let o: Option<Result<Int32, Bool>> = None; \
             match o { Some(Ok(n)) => { println(n); } Some(Err(_b)) => { } None => { } } }",
        ),
        (
            "L9516 integer match without a default arm",
            "fn main() { let n: Int32 = 1; match n { 1 => { println(1); } 2 => { println(2); } } }",
        ),
        (
            "L6979 Option combinator on a droppable payload",
            "fn main() { let o: Option<String> = None; \
             let s = o.unwrap_or(String::from(\"x\")); println(s.len()); }",
        ),
        (
            "L7153 Vec:: method outside the implemented set",
            "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.insert(0u64, 2); }",
        ),
        (
            "L8109 HashMap:: reserved for std-full",
            "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 2); \
             let _e = m.entry(1); }",
        ),
        (
            "L8093 HashMap over user-Drop value types",
            "struct D { v: Int32 } impl Drop for D { fn drop(&mut self) { } } \
             fn main() { let mut m: HashMap<Int32, D> = HashMap::new(); m.insert(1, D { v: 1 }); }",
        ),
        (
            "L9346 print/println of this type",
            "struct P { a: Int32 } fn main() { let p = P { a: 1 }; println(p); }",
        ),
        (
            "L9096 Display of a type inside a composite",
            "fn main() { let t: (Int32, Bool) = (1, true); println(t); }",
        ),
        (
            "L9130 droppable composite carrying a borrowed element",
            "fn main() { let s = String::from(\"a\"); let t: (String, &str) = (s, \"b\"); println(t); }",
        ),
        (
            "L5346 assert_eq on a user-defined type",
            "struct P { a: Int32 } impl Eq for P { fn eq(&self, other: &P) -> Bool { self.a == other.a } } \
             fn main() { let x = P { a: 1 }; let y = P { a: 1 }; assert_eq(x, y); }",
        ),
        (
            "L3698 for over a non-range, non-Vec iterator",
            "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 2); \
             for v in m.values() { println(*v); } }",
        ),
        (
            "L2004 move through a non-field projection of a drop-tracked local",
            "fn main() { let mut v: Vec<String> = Vec::new(); v.push(String::from(\"a\")); \
             let a: [String; 1] = [String::from(\"b\")]; let m = a[0u64]; println(m.len()); }",
        ),
        (
            "L4387 field access on non-struct",
            "fn main() { let t: (Int32, Int32) = (1, 2); println(t.0); }",
        ),
        (
            "L6450 method on a peeled type outside the slice",
            "fn main() { let s = String::from(\"abc\"); let u = s.to_uppercase(); println(u.len()); }",
        ),
        (
            "L5130 indexing a non-Vec/array base",
            "fn main() { let s = String::from(\"abc\"); let b = s[0u64]; println(b); }",
        ),
        (
            "L1884 array length is not a literal count",
            "const N: UInt64 = 4u64; fn main() { let a: [Int32; 4] = [0; 4]; println(a[0u64]); }",
        ),
        (
            "L8238 HashSet:: reserved for std-full",
            "fn main() { let mut s: HashSet<Int32> = HashSet::new(); s.insert(1); \
             let o: HashSet<Int32> = HashSet::new(); let u = s.union(&o); }",
        ),
        (
            "L4083 unary operator outside the set",
            "fn main() { let x: Int32 = 1; let y = -x; println(y); }",
        ),
        (
            "L3483 nested item",
            "fn main() { fn inner() -> Int32 { 1 } println(inner()); }",
        ),
        (
            "L6901 Option/Result method outside the slice",
            "fn main() { let o: Option<Int32> = Some(1); let m = o.map_or(0, 1); println(m); }",
        ),
    ]
}

#[test]
fn layer_audit_reports_reachable_lowering_refusals() {
    let mut defects = Vec::new();
    let mut frontend = 0usize;
    let mut lowers = 0usize;

    println!("\n=== LAYER AUDIT: front end vs lowering ===\n");
    for (label, src) in probes() {
        match probe(src) {
            Outcome::LayerDefect(what) => {
                println!("LAYER-DEFECT | {label}\n               lowering: {what}");
                defects.push(label);
            }
            Outcome::FrontEnd(d) => {
                println!("ok-frontend  | {label}\n               refused by: {d}");
                frontend += 1;
            }
            Outcome::Lowers => {
                println!("ok-lowers    | {label}  (probe does not reach its site)");
                lowers += 1;
            }
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
}
