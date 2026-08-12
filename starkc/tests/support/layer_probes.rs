//! **The layer-probe inventory: one table, two consumers.**
//!
//! Each probe is a small type-correct program aimed at a specific lowering refusal. The table
//! answers one question — *does the front end accept the program before lowering refuses it?* —
//! and two suites read the answer for different purposes:
//!
//! ```text
//! layer_audit.rs                 ENFORCES the inventory: a new layer defect, or a registered one
//!                                that stopped reproducing, fails the audit
//! native_conformance_matrix.rs   GENERATES the published native conformance contract from the
//!                                same measurements, so the matrix cannot drift from behaviour
//! ```
//!
//! **The table lives here so the two cannot disagree.** Before WP-ARCH-CLOSE AC2 the inventory was
//! private to the audit; a matrix built beside it would have been a second classifier for one
//! question, which is the duplicate-authority shape AC5 exists to find. The matrix therefore has no
//! opinion of its own about what a probe does — it renders what this table measures.

use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// The disposition a probe is REGISTERED as having. Every probe declares one; `layer_audit`
/// compares it against what actually happens.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Expect {
    /// The front end refuses it. The program never reaches lowering — correct layering.
    FrontEnd,
    /// Both layers accept. The probe does not reach the site it was aimed at, which is recorded
    /// rather than deleted so the site keeps a name.
    Lowers,
    /// A KNOWN reachable lowering refusal, owned by this deviation. Registered so it is tracked
    /// rather than merely observed.
    KnownDev(&'static str),
}

#[derive(Debug)]
pub enum Outcome {
    /// The front end accepted and lowering refused: the defect this audit looks for.
    LayerDefect(String),
    /// The front end refused. Correct — the program never reaches lowering.
    FrontEnd(String),
    /// Both accepted: this probe does not reach the site it was aimed at.
    Lowers,
}

pub struct LayerProbe {
    /// Names the lowering refusal the probe aims at, by source line. Implementation-facing.
    pub label: &'static str,
    /// **The same probe in the words of someone writing STARK**, for the published matrix. An
    /// external developer cannot act on "L7153"; they can act on "`Vec::insert` and its
    /// neighbours". Where a deviation names a working alternative spelling, it is stated here —
    /// a boundary a developer can route around is a different thing from a dead end.
    pub construct: &'static str,
    pub source: &'static str,
    pub expect: Expect,
    /// Whether the matrix generator runs this probe through the execution engines. Only meaningful
    /// for probes that lower; a refused program has nothing to execute.
    pub execute: bool,
}

/// Where a program stopped, stage by stage. The matrix's front-end columns are read from this;
/// [`probe`] is a projection of it, so the audit and the matrix cannot measure differently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage {
    /// Reached this stage and passed it.
    Passed,
    /// Refused here, with the diagnostic that did it. **Always a STARK diagnostic** — every stage
    /// below is STARK's own, so a refusal recorded here is by construction not a rustc message
    /// leaking through the backend. AC2 requires a refused construct to name its owner, and this
    /// is where that is established rather than asserted.
    Refused(String),
    /// An earlier stage refused, so this one never ran. Distinct from `Passed`: a stage that was
    /// never reached has not been shown to accept anything.
    NotReached,
}

impl Stage {
    pub fn cell(&self) -> &'static str {
        match self {
            Stage::Passed => "pass",
            Stage::Refused(_) => "REFUSES",
            Stage::NotReached => "—",
        }
    }
}

/// Every front-end stage's verdict for one probe, plus the lowering verdict.
#[derive(Debug, Clone)]
pub struct StageMeasurement {
    pub parse: Stage,
    pub resolve: Stage,
    pub typecheck: Stage,
    pub mir_lower: Stage,
}

impl StageMeasurement {
    /// The first refusal, whichever stage produced it.
    pub fn refusal(&self) -> Option<&str> {
        for stage in [&self.parse, &self.resolve, &self.typecheck, &self.mir_lower] {
            if let Stage::Refused(diagnostic) = stage {
                return Some(diagnostic);
            }
        }
        None
    }
}

/// Run a probe through the front end and lowering, recording each stage separately.
pub fn measure(src: &str) -> StageMeasurement {
    let file = Arc::new(SourceFile::new("probe.stark", src.to_string()));
    let mut m = StageMeasurement {
        parse: Stage::Passed,
        resolve: Stage::NotReached,
        typecheck: Stage::NotReached,
        mir_lower: Stage::NotReached,
    };

    // A PARSE or RESOLVE refusal is the front end refusing, which is the outcome this audit is
    // asking about — it is not a broken probe. These used to assert, so the `L3483 nested item`
    // probe (rejected by the parser, since Core v1 has no nested items) took the whole audit down
    // and no result was reported at all. Classifying instead of asserting is what lets a probe aimed
    // at a lowering site legitimately land in the front end.
    let (ast, pd) = parse(&file, ParseMode::Program);
    if let Some(first) = pd.first() {
        m.parse = Stage::Refused(format!(
            "parse: {} {}",
            first.code.as_deref().unwrap_or("-"),
            first.message
        ));
        return m;
    }

    let (hir, rd) = resolve(&ast, file.clone());
    if let Some(first) = rd.first() {
        m.resolve = Stage::Refused(format!(
            "resolve: {} {}",
            first.code.as_deref().unwrap_or("-"),
            first.message
        ));
        return m;
    }
    m.resolve = Stage::Passed;

    let checked = typecheck::analyze(&hir);
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    if let Some(first) = errors.first() {
        m.typecheck = Stage::Refused(format!(
            "{} {}",
            first.code.as_deref().unwrap_or("-"),
            first.message
        ));
        return m;
    }
    m.typecheck = Stage::Passed;

    m.mir_lower = match lower_program(
        &hir,
        &checked.tables,
        hir.source_named(&file.name).expect("registered"),
    ) {
        Ok(_) => Stage::Passed,
        Err(e) => Stage::Refused(e.what),
    };
    m
}

/// The audit's three-way classification, **projected from [`measure`]** rather than measured
/// again. One traversal of the compiler, two readings of it.
pub fn probe(src: &str) -> Outcome {
    let m = measure(src);
    match (&m.parse, &m.resolve, &m.typecheck, &m.mir_lower) {
        (Stage::Refused(d), _, _, _)
        | (_, Stage::Refused(d), _, _)
        | (_, _, Stage::Refused(d), _) => Outcome::FrontEnd(d.clone()),
        (_, _, _, Stage::Refused(what)) => Outcome::LayerDefect(what.clone()),
        _ => Outcome::Lowers,
    }
}

/// The inventory. Order is the published matrix's row order, so it is stable rather than incidental.
pub fn probes() -> Vec<LayerProbe> {
    vec![
        LayerProbe {
            label: "L10807 nested pattern in match arm",
            construct: "A nested pattern in a match arm — `Some(Ok(n))`",
            source: "fn main() { let o: Option<Result<Int32, Bool>> = None; \
                     match o { Some(Ok(n)) => { println(n); } Some(Err(_b)) => { } None => { } } }",
            expect: Expect::Lowers,
            execute: true,
        },
        LayerProbe {
            label: "L9516 integer match without a default arm",
            construct: "A `match` over an integer with no arm covering the remaining values",
            source: "fn main() { let n: Int32 = 1; match n { 1 => { println(1); } 2 => { println(2); } } }",
            expect: Expect::FrontEnd,
            execute: false,
        },
        LayerProbe {
            label: "L6979 Option combinator on a droppable payload",
            construct: "`Option::unwrap_or` where the payload owns a destructor — `Option<String>`",
            source: "fn main() { let o: Option<String> = None; \
                     let s = o.unwrap_or(String::from(\"x\")); println(s.len()); }",
            expect: Expect::Lowers,
            execute: true,
        },
        LayerProbe {
            label: "L7153 Vec:: method outside the implemented set",
            construct: "`Vec::insert` — and equally extend/truncate/sort/reverse/contains/dedup/\
                        split_off/drain/retain. `push`, `pop`, `len` and indexing are supported",
            source: "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.insert(0u64, 2); }",
            expect: Expect::KnownDev("DEV-140"),
            execute: false,
        },
        LayerProbe {
            label: "L8109 HashMap:: reserved for std-full",
            construct: "`HashMap::entry` — reserved for the `std-full` profile this build does not carry",
            source: "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 2); \
                     let _e = m.entry(1); }",
            expect: Expect::FrontEnd,
            execute: false,
        },
        LayerProbe {
            label: "L8093 HashMap over user-Drop value types",
            construct: "`HashMap<K, V>` where `V` implements `Drop`. A `HashMap` of values without \
                        destructors is unaffected",
            source: "struct D { v: Int32 } impl Drop for D { fn drop(&mut self) { } } \
                     fn main() { let mut m: HashMap<Int32, D> = HashMap::new(); m.insert(1, D { v: 1 }); }",
            expect: Expect::KnownDev("DEV-141"),
            execute: false,
        },
        LayerProbe {
            label: "L9346 print/println of this type",
            construct: "`println` of a user struct that implements no `Display`",
            source: "struct P { a: Int32 } fn main() { let p = P { a: 1 }; println(p); }",
            expect: Expect::FrontEnd,
            execute: false,
        },
        LayerProbe {
            label: "L9096 Display of a type inside a composite",
            construct: "`println` of a tuple of primitives — `(Int32, Bool)`",
            source: "fn main() { let t: (Int32, Bool) = (1, true); println(t); }",
            expect: Expect::Lowers,
            execute: true,
        },
        LayerProbe {
            label: "L9130 droppable composite carrying a borrowed element",
            construct: "A composite mixing an owned droppable and a borrow — `(String, &str)`. \
                        Printing the parts separately works",
            source: "fn main() { let s = String::from(\"a\"); let t: (String, &str) = (s, \"b\"); println(t); }",
            expect: Expect::KnownDev("DEV-142"),
            execute: false,
        },
        LayerProbe {
            label: "L5346 assert_eq on a user-defined type",
            construct: "`assert_eq` on a user type implementing `Eq`. `a == b` on the same type \
                        works in every engine",
            source: "struct P { a: Int32 } impl Eq for P { fn eq(&self, other: &P) -> Bool { self.a == other.a } } \
                     fn main() { let x = P { a: 1 }; let y = P { a: 1 }; assert_eq(x, y); }",
            expect: Expect::KnownDev("DEV-143"),
            execute: false,
        },
        LayerProbe {
            label: "L3698 for over a non-range, non-Vec iterator",
            construct: "`for` over an iterator that is neither a range nor a `Vec` cursor — \
                        `HashMap::values()`",
            source: "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 2); \
                     for v in m.values() { println(*v); } }",
            expect: Expect::KnownDev("DEV-144"),
            execute: false,
        },
        LayerProbe {
            label: "L2004 move through a non-field projection of a drop-tracked local",
            construct: "Moving out of an indexed place holding a droppable — `let m = a[0u64];`",
            source: "fn main() { let mut v: Vec<String> = Vec::new(); v.push(String::from(\"a\")); \
                     let a: [String; 1] = [String::from(\"b\")]; let m = a[0u64]; println(m.len()); }",
            expect: Expect::FrontEnd,
            execute: false,
        },
        LayerProbe {
            label: "L4387 field access on non-struct",
            construct: "Tuple element access — `t.0`",
            source: "fn main() { let t: (Int32, Int32) = (1, 2); println(t.0); }",
            expect: Expect::Lowers,
            execute: true,
        },
        LayerProbe {
            label: "L6450 method on a peeled type outside the slice",
            construct: "`String::to_uppercase` — and equally to_lowercase/trim/replace/starts_with/\
                        ends_with/find/split_at/repeat. `len`, `as_str` and `push_str` are supported",
            source: "fn main() { let s = String::from(\"abc\"); let u = s.to_uppercase(); println(u.len()); }",
            expect: Expect::KnownDev("DEV-145"),
            execute: false,
        },
        LayerProbe {
            label: "L5130 indexing a non-Vec/array base",
            construct: "Indexing a `String` — `s[0u64]`",
            source: "fn main() { let s = String::from(\"abc\"); let b = s[0u64]; println(b); }",
            expect: Expect::FrontEnd,
            execute: false,
        },
        LayerProbe {
            label: "L1884 array length is not a literal count",
            construct: "A fixed-size array with a literal length — `[Int32; 4]`",
            source: "const N: UInt64 = 4u64; fn main() { let a: [Int32; 4] = [0; 4]; println(a[0u64]); }",
            expect: Expect::Lowers,
            execute: true,
        },
        LayerProbe {
            label: "L8238 HashSet:: reserved for std-full",
            construct: "`HashSet::union` — reserved for the `std-full` profile this build does not carry",
            source: "fn main() { let mut s: HashSet<Int32> = HashSet::new(); s.insert(1); \
                     let o: HashSet<Int32> = HashSet::new(); let u = s.union(&o); }",
            expect: Expect::FrontEnd,
            execute: false,
        },
        LayerProbe {
            label: "L4083 unary operator outside the set",
            construct: "Unary negation of an integer — `-x`",
            source: "fn main() { let x: Int32 = 1; let y = -x; println(y); }",
            expect: Expect::Lowers,
            execute: true,
        },
        LayerProbe {
            label: "L3483 nested item",
            construct: "A function declared inside another function",
            source: "fn main() { fn inner() -> Int32 { 1 } println(inner()); }",
            expect: Expect::FrontEnd,
            execute: false,
        },
        LayerProbe {
            label: "L6901 Option/Result method outside the slice",
            construct: "`Option::map_or`",
            source: "fn main() { let o: Option<Int32> = Some(1); let m = o.map_or(0, 1); println(m); }",
            expect: Expect::FrontEnd,
            execute: false,
        },
    ]
}
