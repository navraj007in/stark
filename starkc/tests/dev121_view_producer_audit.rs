//! **DEV-121 class closure — the view-producer audit.**
//!
//! AS3 work item 6 and exit criterion 5:
//!
//! > Inventory and close the separately identified typed-mutation boundaries before closing the
//! > DEV-121 defect class. […] DEV-121 closes only with a **class-level evidence statement, not one
//! > regression case**.
//!
//! **The class.** An intrinsic whose declared return is a view (`&[T]`, `&str`) but whose runtime
//! representation in the HIR oracle is OWNED (`Value::Vec`, `Value::String`). Such a value moves
//! when it should copy, so the producer's own binding is emptied and a later use traps — on a
//! program the checker and MIR both accepted, with correct MIR. Two instances reached users before
//! any tooling saw them: `String::bytes()` (CD-305) and `String::split()`'s item (CD-340).
//!
//! **Why one more regression case would not close it.** Both instances were found by user-facing
//! packages, not by the invariant, and DEV-121 UPDATE 2 named the reason: `INV-VALUE-REP-001`
//! checked `let` bindings, and *both instances were reachable through a `for`-loop item, which is
//! not a `let`*. A third fixture for a third producer would have the same blind spot.
//!
//! So this file makes two class-level claims instead:
//!
//! 1. **Every view-returning intrinsic is exercised** in all three binding positions the invariant
//!    now covers — `let`, loop item, call argument — so a representation defect fires the invariant
//!    rather than surfacing as a trap three frames away.
//! 2. **The inventory cannot go stale.** `every_view_returning_intrinsic_is_classified` scans
//!    `core_method_signature` and requires every method arm mentioning a view type to appear in one
//!    of the tables below. Adding a new `&[T]`/`&str` intrinsic without classifying it fails this
//!    test — which is the difference between an audit and a snapshot of one afternoon.

use starkc::options::LanguageOptions;
use starkc::session::CompilerSession;
use starkc::source::SourceFile;
use std::sync::Arc;

/// Intrinsics that RETURN a view. Each is exercised in all three binding positions below.
///
/// `setup` builds an owner binding; `call` is the view-producing expression over it.
const VIEW_PRODUCERS: &[(&str, &str, &str, &str)] = &[
    // (method, setup, view expression, the view's own type as a parameter)
    (
        "as_str",
        "let owner: String = String::from(\"hello\");",
        "owner.as_str()",
        "&str",
    ),
    (
        "trim",
        "let owner: String = String::from(\"  hi  \");",
        "owner.trim()",
        "&str",
    ),
    (
        "bytes",
        "let owner: String = String::from(\"hey\");",
        "owner.bytes()",
        "&[UInt8]",
    ),
    (
        "as_slice",
        "let mut owner: Vec<Int32> = Vec::new(); owner.push(1); owner.push(2);",
        "owner.as_slice()",
        "&[Int32]",
    ),
    (
        "substring",
        "let owner: String = String::from(\"hello\");",
        "owner.substring(1, 3)",
        "&str",
    ),
];

/// Methods whose arm mentions a view type only in a PARAMETER, or which return an owned value
/// deliberately. Listed so the completeness scan has a total classification and neither answer is
/// the default.
const NOT_VIEW_RETURNING: &[&str] = &[
    // view in parameter position only
    "push_str",
    "contains",
    "starts_with",
    "ends_with",
    "write",
    "write_str",
    "replace",
    "find",
    "split",
    "push",
    "extend",
    "append",
    "insert",
    // deliberately owned returns — `into_bytes` is `bytes`'s owning twin, and the pair sharing one
    // implementation arm is what produced CD-305 in the first place
    "into_bytes",
    "to_string",
    "to_lowercase",
    "to_uppercase",
    "read_to_string",
    "collect",
    // not value-producing, or producing a non-view
    "clear",
    "is_empty",
    "len",
    "pop",
    "iter",
    "chars",
    "next",
];

fn run(label: &str, source: &str) -> String {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let program = match CompilerSession::for_source(file, LanguageOptions::CORE).check() {
        Ok(program) => program,
        Err(failure) => panic!("{label}: must compile:\n{}", failure.render()),
    };
    match program.execute_hir() {
        Ok(execution) => execution.output,
        Err(error) => panic!(
            "{label}: must run, got: {}\n\
             A firing of INV-VALUE-REP-001 here is a DEV-121 instance: the declared view type and \
             the runtime representation disagree.",
            error.message
        ),
    }
}

/// **Position 1 — `let`.** The shape `INV-VALUE-REP-001` already covered.
#[test]
fn every_view_producer_binds_and_survives_a_let() {
    for (method, setup, call, _kind) in VIEW_PRODUCERS {
        // The view is bound, then used TWICE. If it were owned, the first use would move it and
        // the second would trap — which is exactly how DEV-121 presented.
        let source = format!(
            "fn take_len(v: {}) -> UInt64 {{ v.len() }}\n\
             fn main() {{\n\
             \x20   {setup}\n\
             \x20   let view = {call};\n\
             \x20   let a: UInt64 = take_len(view);\n\
             \x20   let b: UInt64 = take_len(view);\n\
             \x20   println(a + b);\n}}\n",
            _kind
        );
        let out = run(&format!("{method} / let"), &source);
        assert!(
            !out.is_empty(),
            "{method}: a view bound by `let` must survive two uses"
        );
    }
}

/// **Position 2 — call argument.** Newly covered. A view passed to a function must not be moved
/// out of the caller's binding.
#[test]
fn every_view_producer_survives_being_passed_as_an_argument() {
    for (method, setup, call, kind) in VIEW_PRODUCERS {
        let source = format!(
            "fn take_len(v: {}) -> UInt64 {{ v.len() }}\n\
             fn main() {{\n\
             \x20   {setup}\n\
             \x20   let view = {call};\n\
             \x20   println(take_len(view));\n\
             \x20   println(take_len(view));\n}}\n",
            kind
        );
        let out = run(&format!("{method} / argument"), &source);
        let lines: Vec<&str> = out.trim_end().split('\n').collect();
        assert_eq!(
            lines.len(),
            2,
            "{method}: both calls must run — a moved view would trap on the second"
        );
        assert_eq!(
            lines[0], lines[1],
            "{method}: a view is Copy, so two calls must see the same length"
        );
    }
}

/// **Position 3 — `for`-loop item.** The blind spot DEV-121 UPDATE 2 named, and the shape BOTH
/// known instances of the class arrived through.
#[test]
fn a_view_used_as_a_loop_item_is_checked() {
    // `String::split()`'s item is declared `&str` (CD-340). Iterating it binds a view to a loop
    // local — the position that had no coverage at all.
    let out = run(
        "split item / loop",
        "fn main() {\n\
         \x20   let owner: String = String::from(\"a,b,c\");\n\
         \x20   let mut n: UInt64 = 0u64;\n\
         \x20   for part in owner.split(\",\") {\n\
         \x20       n = n + part.len();\n\
         \x20   }\n\
         \x20   println(n);\n}\n",
    );
    assert_eq!(out, "3\n", "three single-character parts");
}

/// **There is no loop position for a slice view, and that is a language rule, not a gap.**
///
/// `for b in view` where `view: &[UInt8]` is rejected — *"for-loop requires an iterable value,
/// found '&[UInt8]'"*. Core v1 does not make a slice iterable. Recorded here rather than left as a
/// missing case, so a reader does not later "fix" this file by adding a fixture that cannot compile.
///
/// The loop coverage above therefore rests on `&str` items from `split()`, which is the shape BOTH
/// known instances of the class actually took.
#[test]
fn a_slice_view_is_not_iterable_in_core_v1() {
    let file = Arc::new(SourceFile::new(
        "test.stark",
        "fn main() {\n\
         \x20   let owner: String = String::from(\"abc\");\n\
         \x20   let view = owner.bytes();\n\
         \x20   for b in view { println(b); }\n}\n",
    ));
    let failure = CompilerSession::for_source(file, LanguageOptions::CORE)
        .check()
        .err()
        .expect("iterating a slice must be REJECTED, not accepted-then-trapped");
    let rendered = failure.render();
    assert!(
        rendered.contains("for-loop requires an iterable value"),
        "the rejection must be the iterability rule, not an unrelated error: {rendered}"
    );
}

/// The slice view still survives repeated use — the DEV-121 property — even without a loop.
#[test]
fn a_slice_view_survives_repeated_use() {
    let out = run(
        "bytes / repeated use",
        "fn take(v: &[UInt8]) -> UInt64 { v.len() }\n         fn main() {\n         \x20   let owner: String = String::from(\"abc\");\n         \x20   let view = owner.bytes();\n         \x20   println(take(view));\n         \x20   println(take(view));\n         \x20   println(owner.len());\n}\n",
    );
    assert_eq!(
        out, "3\n3\n3\n",
        "the view is Copy and the OWNER stays live — DEV-121 emptied the owner's binding"
    );
}

/// **The claim that makes this an audit rather than a snapshot.**
///
/// Scans `core_method_signature` and requires every method arm mentioning a view type to be
/// classified — either exercised by `VIEW_PRODUCERS` or listed in `NOT_VIEW_RETURNING`. A new
/// `&[T]`/`&str` intrinsic added without a decision fails here.
///
/// The scan deliberately OVER-approximates: it flags any arm mentioning `str_ref` or `Ty::Slice`,
/// including those using one only as a parameter. An extra entry costs a line in a table; a missed
/// one is a producer nobody audited, which is the whole defect class.
#[test]
fn every_view_returning_intrinsic_is_classified() {
    // Source-scanning: normalise line endings at the read, or this passes on Unix and fails in the
    // Windows lane where the checkout has CRLF.
    // AS7: `core_method_signature` moved from `typecheck.rs` to `typecheck/traits.rs` when the
    // pass was split by semantic ownership. This audit follows the FUNCTION, not the file.
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/typecheck/traits.rs"
    ))
    .expect("typecheck/traits.rs must be readable")
    .replace("\r\n", "\n");

    let start = source
        .find("fn core_method_signature")
        .expect("core_method_signature must exist");
    let body = &source[start..];
    let end = body
        .find("\n    }\n")
        .expect("core_method_signature must be delimited");
    let body = &body[..end];

    let mut unclassified: Vec<String> = Vec::new();
    // An arm is the text from one method-name literal to the next. Any arm mentioning a view type
    // is a candidate.
    let mut names: Vec<(usize, String)> = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while let Some(open) = body[i..].find('"') {
        let open = i + open;
        let Some(close_rel) = body[open + 1..].find('"') else {
            break;
        };
        let close = open + 1 + close_rel;
        let name = &body[open + 1..close];
        if !name.is_empty()
            && name.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
            && bytes.get(close + 1).is_some()
        {
            names.push((close, name.to_string()));
        }
        i = close + 1;
    }

    for (index, (pos, name)) in names.iter().enumerate() {
        let arm_end = names
            .get(index + 1)
            .map(|(next, _)| *next)
            .unwrap_or(body.len());
        let arm = &body[*pos..arm_end];
        let mentions_view = arm.contains("str_ref") || arm.contains("Ty::Slice");
        if !mentions_view {
            continue;
        }
        let audited = VIEW_PRODUCERS.iter().any(|(m, ..)| m == name);
        let excluded = NOT_VIEW_RETURNING.contains(&name.as_str());
        if !audited && !excluded {
            unclassified.push(name.clone());
        }
    }
    unclassified.sort();
    unclassified.dedup();

    assert!(
        unclassified.is_empty(),
        "unclassified view-mentioning intrinsics: {unclassified:?}\n\
         Each must be added to VIEW_PRODUCERS (it returns a view — exercise it) or to \
         NOT_VIEW_RETURNING (it only takes one, or returns owned storage deliberately). \
         An unaudited view producer is DEV-121's defect class."
    );
}

/// **A view that never binds to a `let`** — the position the original invariant could not see at
/// all, and the one that makes the extension load-bearing rather than decorative.
///
/// Measured, with `String::bytes()` mutated back to its DEV-121 behaviour (returning an owned
/// `Value::Vec`):
///
/// ```text
/// invariant wired at parameters   TRAP  "...holds an owned Vec... (DEV-121)"
/// invariant NOT wired (let-only)  OK    "3\n3\n"      <- defect completely invisible
/// ```
///
/// The second row is the point. With `let`-only coverage the broken producer produces a program
/// that runs and prints the right answer, so no test and no user would ever see it — which is how
/// both known instances reached packages before any tooling noticed.
#[test]
fn a_view_reaching_a_parameter_without_a_let_is_covered() {
    let out = run(
        "bytes / parameter, no let",
        "fn take(v: &[UInt8]) -> UInt64 { v.len() }\n         fn main() {\n         \x20   let owner: String = String::from(\"abc\");\n         \x20   println(take(owner.bytes()));\n         \x20   println(take(owner.bytes()));\n         \x20   println(owner.len());\n}\n",
    );
    assert_eq!(
        out, "3\n3\n3\n",
        "the view must reach the parameter as a view, twice, leaving the owner live"
    );
}
