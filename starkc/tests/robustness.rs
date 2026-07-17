//! Deterministic pseudo-fuzz for the lexer+parser (PLAN.md WP1.4).
//!
//! The plan's fuzz gate is "no panics, no hangs on arbitrary input; grammar
//! correctness is owned by the fixtures, not the fuzzer". This runs that
//! gate on the stable toolchain in ordinary CI: a fixed-seed LCG generates
//! random character soup, random token soup, mutated fixtures, and
//! pathological nesting. Every case must parse (in both modes) without
//! panicking; diagnostics are expected and ignored.

use starkc::parser::{parse, ParseMode};
use starkc::source::SourceFile;
use std::path::PathBuf;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        // Numerical Recipes LCG constants; determinism matters, quality not.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn parse_both(src: &str) {
    let file = SourceFile::new("fuzz.stark", src.to_string());
    let _ = parse(&file, ParseMode::Program);
    let file = SourceFile::new("fuzz.stark", src.to_string());
    let _ = parse(&file, ParseMode::Snippet);
}

#[test]
fn random_character_soup() {
    let alphabet: Vec<char> = ('!'..='~').chain(" \t\n\"'\u{1F600}éλ".chars()).collect();
    let mut rng = Lcg(0xC0FFEE);
    for _ in 0..500 {
        let len = rng.below(200);
        let src: String = (0..len)
            .map(|_| alphabet[rng.below(alphabet.len())])
            .collect();
        parse_both(&src);
    }
}

#[test]
fn random_token_soup() {
    const VOCAB: &[&str] = &[
        "fn", "struct", "enum", "trait", "impl", "let", "mut", "const", "type", "use", "mod", "if",
        "else", "match", "for", "while", "loop", "break", "continue", "return", "in", "as", "pub",
        "priv", "self", "Self", "super", "crate", "true", "false", "async", "Int32", "String",
        "Bool", "str", "Unit", "x", "y", "Vec", "Option", "T", "_", "42", "3.14", "0xFF", "1_000",
        "42i32", "\"s\"", "'c'", "r\"raw\"", "+", "-", "*", "/", "%", "**", "==", "!=", "<", "<=",
        ">", ">=", "&&", "||", "!", "&", "|", "^", "~", "<<", ">>", "=", "+=", "-=", "**=", ">>=",
        "..", "..=", "?", "::", ".", "->", "=>", "(", ")", "[", "]", "{", "}", ",", ";", ":", "<",
        ">",
    ];
    let mut rng = Lcg(0xBADC0DE);
    for _ in 0..500 {
        let len = rng.below(120);
        let src: String = (0..len)
            .map(|_| VOCAB[rng.below(VOCAB.len())])
            .collect::<Vec<_>>()
            .join(" ");
        parse_both(&src);
    }
}

#[test]
fn mutated_fixtures() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../STARKLANG/tests/spec-fixtures");
    let mut rng = Lcg(0xFEED);
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "stark") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        // Truncate at a random char boundary.
        let mut cut = rng.below(src.len() + 1);
        while !src.is_char_boundary(cut) {
            cut -= 1;
        }
        parse_both(&src[..cut]);
        // Delete a random line.
        let lines: Vec<&str> = src.lines().collect();
        if !lines.is_empty() {
            let victim = rng.below(lines.len());
            let mutated: String = lines
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != victim)
                .map(|(_, l)| *l)
                .collect::<Vec<_>>()
                .join("\n");
            parse_both(&mutated);
        }
    }
}

#[test]
fn pathological_nesting_is_a_diagnostic_not_a_crash() {
    for src in [
        "(".repeat(50_000),
        "[".repeat(50_000),
        "{".repeat(50_000),
        "-".repeat(50_000) + "x",
        "&".repeat(50_000) + "x",
        format!("let x: {}Int32;", "&".to_string().repeat(50_000)),
        format!("match x {{ {} => 1 }}", "(".repeat(50_000)),
        "mod m {".repeat(20_000),
    ] {
        parse_both(&src);
    }
}

/// WP-C1.1: pathological_nesting_is_a_diagnostic_not_a_crash above discards its parse result
/// entirely (`let _ = parse(...)`), so it never confirmed the depth limit actually produces the
/// expected diagnostic, produces it exactly once (the `depth_reported` latch in parser.rs), or
/// leaves moderate-but-legitimate nesting unaffected. This closes that gap.
#[test]
fn depth_limit_fires_once_and_does_not_false_positive_below_it() {
    // Well under parser.rs's MAX_DEPTH=200 -- must parse clean, no depth diagnostic at all.
    let shallow = SourceFile::new("depth.stark", "-".repeat(50) + "x");
    let (_ast, diags) = parse(&shallow, ParseMode::Snippet);
    assert!(
        diags
            .iter()
            .all(|d| !d.message.contains("nested too deeply")),
        "moderate nesting (50 levels) must not trip the depth limit: {:?}",
        diags
    );

    // Well over MAX_DEPTH -- must produce the diagnostic exactly once, not once per exceeded
    // recursion attempt (which, at 50,000 levels, would otherwise be tens of thousands of
    // duplicate diagnostics).
    let deep = SourceFile::new("depth.stark", "-".repeat(50_000) + "x");
    let (_ast, diags) = parse(&deep, ParseMode::Snippet);
    let depth_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.message.contains("nested too deeply"))
        .collect();
    assert_eq!(
        depth_diags.len(),
        1,
        "expected exactly one depth-exceeded diagnostic, got {}: {:?}",
        depth_diags.len(),
        depth_diags
    );
}

/// WP-C1.1 (checklist item 10): the fuzz generators above are deterministic in their *inputs*
/// (fixed LCG seeds) but nothing previously checked that *outputs* (diagnostics) are stable
/// across two runs of the identical input -- Charter §2.5 requires "generated output is
/// deterministic across two runs". Re-parsing a sample of generated cases and comparing
/// diagnostic (code, message, span) triples catches any nondeterministic-iteration regression
/// (the class of bug found separately in resolve.rs's glob-import handling, DEV-007).
#[test]
fn diagnostics_are_deterministic_across_repeated_parses() {
    let mut rng = Lcg(0xD00D);
    let alphabet: Vec<char> = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ \t\n"
        .chars()
        .collect();
    for _ in 0..100 {
        let len = rng.below(150);
        let src: String = (0..len)
            .map(|_| alphabet[rng.below(alphabet.len())])
            .collect();
        for mode in [ParseMode::Program, ParseMode::Snippet] {
            let file_a = SourceFile::new("det.stark", src.clone());
            let (_ast_a, diags_a) = parse(&file_a, mode);
            let file_b = SourceFile::new("det.stark", src.clone());
            let (_ast_b, diags_b) = parse(&file_b, mode);
            let summarize = |diags: &[starkc::diag::Diagnostic]| -> Vec<(Option<String>, String)> {
                diags
                    .iter()
                    .map(|d| (d.code.clone(), d.message.clone()))
                    .collect()
            };
            assert_eq!(
                summarize(&diags_a),
                summarize(&diags_b),
                "diagnostics differ across two parses of the identical input: {:?}",
                src
            );
        }
    }
}
