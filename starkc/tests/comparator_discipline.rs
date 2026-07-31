//! WP-C7.9 G.2/G.3 — guards on the comparator itself.
//!
//! These do not test the language. They test the properties the comparator must keep having, and
//! they are here because both were lost once already:
//!
//! - **Trap identity must be structural.** The comparator used to recover a trap's category by
//!   searching the oracle's message for phrases like `"integer overflow"`. That made diagnostic
//!   wording load-bearing for semantic classification: rewording a message would silently
//!   reclassify a trap, and any error whose text happened to contain a phrase would be classified
//!   as that trap. Every language trap now states its category where it is raised.
//! - **No test may pass by running nothing.** Every arm of `three_engine_test!` used to check for
//!   `rustc` and return early if it was missing — so in an environment without a Rust toolchain,
//!   the whole suite passed while comparing nothing at all.
//!
//! Both guards read the comparator's own source. That is deliberate: the properties are about what
//! the code is allowed to contain, and a behavioural test cannot observe the absence of a fallback
//! that only fires in an environment the test is not running in.

use std::path::PathBuf;

fn comparator_source() -> String {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "support",
        "differential.rs",
    ]
    .iter()
    .collect();
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read the comparator at {}: {e}", path.display()))
}

/// The banned pattern: classifying a trap by matching prose.
///
/// The phrases themselves are fine in comments and in failure messages — what is banned is a
/// `contains(...)` test on a trap message, which is how the old normaliser worked. The guard looks
/// for that shape rather than for the words.
#[test]
fn trap_identity_is_not_recovered_from_prose() {
    let source = comparator_source();
    for phrase in [
        "integer overflow",
        "division by zero",
        "invalid shift",
        "out of bounds",
        "assertion failed",
        "numeric cast",
    ] {
        let banned = format!("contains(\"{phrase}\")");
        assert!(
            !source.contains(&banned),
            "the comparator classifies a trap by matching the phrase {phrase:?}. Trap identity \
             must come from the category stated at the raise site (WP-C7.9 G.3); prose is \
             diagnostic content, and rewording a message must not be able to change what a \
             program is observed to have done."
        );
    }
}

/// No arm of the three-engine macro may return without comparing something.
#[test]
fn the_three_engine_macro_never_returns_without_comparing() {
    let source = comparator_source();
    let start = source
        .find("macro_rules! three_engine_test")
        .expect("the macro must exist");
    let macro_body = &source[start..];
    let end = macro_body
        .find("\n}\n")
        .expect("the macro must be terminated");
    let macro_body = &macro_body[..end];
    assert!(
        !macro_body.contains("return;"),
        "an arm of three_engine_test! returns early. Every arm must delegate to an \
         available-engines comparator, so a missing native toolchain removes an ENGINE rather \
         than the whole comparison (WP-C7.9 G.2)."
    );
    assert!(
        !macro_body.contains("rustc_available"),
        "an arm of three_engine_test! decides for itself what to do when rustc is missing. That \
         decision belongs to the available-engines comparators, which report which engines ran."
    );
}

/// The oracle's category reader must fail loudly on an unclassified trap rather than defaulting.
#[test]
fn an_unclassified_oracle_trap_is_a_failure_not_a_default() {
    let source = comparator_source();
    let start = source
        .find("pub fn oracle_category")
        .expect("oracle_category must exist");
    let body = &source[start..start + 1200];
    assert!(
        body.contains("panic!"),
        "oracle_category must fail on a trap with no stated category. Defaulting would hand it \
         whatever category the other engines reported, which is precisely the agreement-without-\
         conformance failure mode this work package closed."
    );
}
