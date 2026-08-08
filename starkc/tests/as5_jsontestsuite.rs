//! AS5-d — the **independent** JSON conformance oracle.
//!
//! `as5_json_conformance.rs` is project-authored: derived from RFC 8259 and from the twelve
//! constructs on which STARK's two previous parsers diverged. It proves the historical defects stay
//! fixed. It cannot prove the parser and its tests do not share a misreading of the grammar — and
//! AS5 exists precisely *because* two in-tree implementations disagreed about what JSON is.
//!
//! This runs Nicolas Seriot's **JSONTestSuite** (MIT), vendored unmodified under
//! `tests/fixtures/jsontestsuite/`. It was fetched, run once, and committed; the parser had never
//! seen it.
//!
//! | Prefix | Meaning | Count |
//! | --- | --- | ---: |
//! | `y_` | a conforming parser **must accept** | 95 |
//! | `n_` | a conforming parser **must reject** | 188 |
//! | `i_` | implementation-defined — either verdict conforms | 35 |
//!
//! The `i_` verdicts are pinned below rather than skipped. They are where a parser's real character
//! shows, and leaving them unasserted would mean a change in what STARK accepts passed in silence.

use std::path::{Path, PathBuf};

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("jsontestsuite")
}

/// Whether the compiler's JSON authority accepts a file.
///
/// A JSON text is UTF-8 (RFC 8259 §8.1), so input that is not UTF-8 is not JSON. `crate::json`
/// takes `&str`, so that check happens here rather than inside the parser — and the suite has
/// several UTF-16 and invalid-UTF-8 cases that exercise it.
fn accepts(path: &Path) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        panic!("corpus file must be readable: {}", path.display());
    };
    match std::str::from_utf8(&bytes) {
        Ok(text) => starkc::json::parse(text).is_ok(),
        Err(_) => false,
    }
}

fn files_with_prefix(prefix: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(corpus())
        .expect("the vendored corpus must be present")
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|e| e == "json")
                && path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with(prefix))
        })
        .collect();
    out.sort();
    out
}

#[test]
fn every_must_accept_case_is_accepted() {
    let files = files_with_prefix("y_");
    assert_eq!(
        files.len(),
        95,
        "the vendored corpus should carry 95 must-accept cases; the fixture set has changed"
    );
    let rejected: Vec<String> = files
        .iter()
        .filter(|path| !accepts(path))
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert!(
        rejected.is_empty(),
        "these are valid JSON and were refused:\n  {}",
        rejected.join("\n  ")
    );
}

#[test]
fn every_must_reject_case_is_rejected() {
    let files = files_with_prefix("n_");
    assert_eq!(
        files.len(),
        188,
        "the vendored corpus should carry 188 must-reject cases; the fixture set has changed"
    );
    let accepted: Vec<String> = files
        .iter()
        .filter(|path| accepts(path))
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert!(
        accepted.is_empty(),
        "these are not JSON and were accepted:\n  {}",
        accepted.join("\n  ")
    );
}

/// The implementation-defined group, with STARK's verdict for each pinned.
///
/// The pattern in this table *is* the design, and it is the same one DEV-185 settled:
///
/// - **every huge-number case is accepted.** `1e400`, a 700-digit integer and the overflow cases are
///   all syntactically valid JSON numbers, and RFC 8259 §6 sets no range limit. `JsonNumber` keeps
///   the text; whether a value fits `i64` or a finite `f64` is a question each consumer answers, and
///   can refuse.
/// - **every surrogate and UTF-8 case is rejected.** DEV-182 was the LSP parser silently
///   substituting for a malformed escape. Refusing is the whole lesson.
/// - **the 500-nested-array case is rejected**, by `json::MAX_DEPTH` (128). RFC 8259 §9 permits the
///   limit, and a recursive-descent parser reading from a socket without one turns depth into a
///   stack overflow.
/// - **byte-order marks are rejected.** A BOM is not JSON text; §8.1 says an implementation "MAY
///   ignore" one, and ignoring input is how DEV-182 shaped errors get in.
#[test]
fn implementation_defined_verdicts_are_pinned() {
    const PINNED: &[(&str, bool)] = &[
        ("i_number_double_huge_neg_exp.json", true),
        ("i_number_huge_exp.json", true),
        ("i_number_neg_int_huge_exp.json", true),
        ("i_number_pos_double_huge_exp.json", true),
        ("i_number_real_neg_overflow.json", true),
        ("i_number_real_pos_overflow.json", true),
        ("i_number_real_underflow.json", true),
        ("i_number_too_big_neg_int.json", true),
        ("i_number_too_big_pos_int.json", true),
        ("i_number_very_big_negative_int.json", true),
        ("i_object_key_lone_2nd_surrogate.json", false),
        ("i_string_1st_surrogate_but_2nd_missing.json", false),
        ("i_string_1st_valid_surrogate_2nd_invalid.json", false),
        ("i_string_UTF-16LE_with_BOM.json", false),
        ("i_string_UTF-8_invalid_sequence.json", false),
        ("i_string_UTF8_surrogate_U+D800.json", false),
        ("i_string_incomplete_surrogate_and_escape_valid.json", false),
        ("i_string_incomplete_surrogate_pair.json", false),
        ("i_string_incomplete_surrogates_escape_valid.json", false),
        ("i_string_invalid_lonely_surrogate.json", false),
        ("i_string_invalid_surrogate.json", false),
        ("i_string_invalid_utf-8.json", false),
        ("i_string_inverted_surrogates_U+1D11E.json", false),
        ("i_string_iso_latin_1.json", false),
        ("i_string_lone_second_surrogate.json", false),
        ("i_string_lone_utf8_continuation_byte.json", false),
        ("i_string_not_in_unicode_range.json", false),
        ("i_string_overlong_sequence_2_bytes.json", false),
        ("i_string_overlong_sequence_6_bytes.json", false),
        ("i_string_overlong_sequence_6_bytes_null.json", false),
        ("i_string_truncated-utf-8.json", false),
        ("i_string_utf16BE_no_BOM.json", false),
        ("i_string_utf16LE_no_BOM.json", false),
        ("i_structure_500_nested_arrays.json", false),
        ("i_structure_UTF-8_BOM_empty_object.json", false),
    ];

    let files = files_with_prefix("i_");
    assert_eq!(
        files.len(),
        PINNED.len(),
        "the implementation-defined group changed size; every case needs a recorded verdict"
    );

    let mut changed = Vec::new();
    for (name, expected) in PINNED {
        let path = corpus().join(name);
        assert!(
            path.exists(),
            "pinned case {name} is missing from the corpus"
        );
        let actual = accepts(&path);
        if actual != *expected {
            changed.push(format!(
                "{name}: was {}, now {}",
                if *expected { "accepted" } else { "rejected" },
                if actual { "accepted" } else { "rejected" }
            ));
        }
    }
    assert!(
        changed.is_empty(),
        "STARK's verdict changed on implementation-defined input:\n  {}\n\n\
         Either verdict conforms, so this is not automatically a defect — but it is a decision. \
         Record why it changed and update the table.",
        changed.join("\n  ")
    );

    // Non-vacuity: the table must actually contain both verdicts, or it is asserting nothing about
    // where the line falls.
    assert!(PINNED.iter().any(|(_, v)| *v), "no accepted case pinned");
    assert!(PINNED.iter().any(|(_, v)| !*v), "no rejected case pinned");
}

/// The corpus is present and whole. A silently empty fixture directory would make all three tests
/// above pass while checking nothing.
#[test]
fn the_vendored_corpus_is_intact() {
    let total = std::fs::read_dir(corpus())
        .expect("fixture directory must exist")
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "json"))
        .count();
    assert_eq!(total, 318, "the vendored corpus should carry 318 cases");
    assert!(
        corpus().join("LICENSE").exists(),
        "the MIT licence must travel with the vendored corpus"
    );
    assert!(
        corpus().join("README.md").exists(),
        "the corpus's provenance note must be present"
    );
}
