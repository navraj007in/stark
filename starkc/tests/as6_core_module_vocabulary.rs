//! **AS6 — the forcing function: Core front-end modules do not carry the extension's vocabulary.**
//!
//! AS6's work list names a deliverable the packet's four implementation commits did not produce:
//!
//! > Add dependency/lint tests preventing new tensor imports in designated Core-only modules.
//!
//! Exit qualification found it missing and this file is it. It exists because exit criterion 2 —
//! *"central Core modules do not contain open-ended tensor spelling tables or method catalogues"* —
//! is the one criterion that **decays silently**. Criteria 1 and 3 are behavioural and the
//! two-directional suite in `as6_core_session_isolation.rs` catches a regression in either. A
//! spelling table, by contrast, comes back one arm at a time: somebody adds `"Float8"` to a match
//! in `parse_type` because that is where the surrounding code already is, every test still passes,
//! and the boundary AS6 paid for erodes without a single failure.
//!
//! ## What it checks, and why that is the right check
//!
//! Not `grep Tensor == 0`. Packet 4C demonstrated why: moving 21 spellings out of `parser.rs`
//! *raised* its match count from 225 to 227, because `tensor_syntax::` is itself a match. A test
//! written against reference counts would have scored that commit as a regression.
//!
//! The check is instead: **no Core front-end module may contain a string literal that is exactly
//! one of the extension's owned or reserved names.** Referring to the extension is fine — calling
//! `tensor_syntax::reserved_type_note(name)` is the boundary working. Spelling `"QInt8"` is the
//! thing that must live in one place.
//!
//! The vocabulary is not restated here. It is read back out of the extension's own tables, so a
//! name added to `extensions/tensor/syntax.rs` is automatically a name Core may not spell.

use std::path::{Path, PathBuf};

/// Core front-end modules, in the sense AS6 uses: the passes and data structures a Core-only
/// session runs through. Deliberately not the CLI, backend or provider surfaces, where words like
/// `"output"` are ordinary English.
const CORE_FRONT_END: &[&str] = &[
    "src/lexer.rs",
    "src/parser.rs",
    "src/resolve.rs",
    "src/typecheck.rs",
    "src/hir.rs",
    "src/ast.rs",
    "src/diag.rs",
    "src/borrowck.rs",
    "src/flow.rs",
    "src/format_syntax.rs",
    "src/formatter/printer.rs",
    "src/interp.rs",
    "src/mir/lower.rs",
    "src/deploy/lower.rs",
    "src/deploy/ir.rs",
    "src/deploy/emit.rs",
];

/// Spellings a Core module may keep, each with the reason. The assertion below is **set equality**,
/// not a skip-list: a new violation fails, and so does *removing* one of these without updating the
/// list. A loose exemption list decays into a place to hide things; a tight one is a ledger.
///
/// `ast::Primitive::name` is the exhaustive rendering of Core's primitive enum, two of whose
/// variants the `tensor` extension added (`Float16`, `BFloat16`). This is residue of a different
/// kind from the tables AS6 moved: it is not an *open-ended table* that grows an arm each time the
/// extension gains a name — it is a `match` over a closed Core enum, and adding a dtype means
/// adding a `Primitive` variant, which the compiler forces to be handled everywhere. Sealing it
/// properly is the same cut fe80129 made for `hir::Builtin`'s thirty-three tensor variants, and it
/// is a larger change than AS6 scoped: every `match` on `Primitive` in the checker, interpreter,
/// MIR and backends. **Recorded, not hidden.**
const ACCEPTED_RESIDUE: &[(&str, &str, &str)] = &[
    (
        "src/ast.rs",
        "Float16",
        "ast::Primitive::name — exhaustive rendering of a closed Core enum, not an open table",
    ),
    (
        "src/ast.rs",
        "BFloat16",
        "ast::Primitive::name — exhaustive rendering of a closed Core enum, not an open table",
    ),
    (
        "src/deploy/ir.rs",
        "TensorAny",
        "Display for DeployTy, a closed enum of the deployment IR's own types",
    ),
    (
        "src/deploy/emit.rs",
        "Tensor",
        "the GENERATED RUST host's type name, not a STARK spelling — it happens to coincide",
    ),
];

/// The surfaces AS6's inventory found already extension-free. They are pinned at **zero** rather
/// than at "no spellings", because for these the stronger property already holds and a first
/// tensor reference in any of them is a design question, not a detail.
const MUST_STAY_TENSOR_FREE: &[&str] = &["src/lexer.rs", "src/diag.rs", "src/format_syntax.rs"];

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Read a source file with line endings normalised. A checkout with CRLF endings must not change
/// what a source-scanning test sees.
fn read_normalised(path: &Path) -> String {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{} unreadable: {error}", path.display()));
    text.replace("\r\n", "\n")
}

/// Strip `//` line comments so a doc comment explaining the boundary is not itself a violation.
/// Deliberately crude: it does not parse strings containing `//`, which would make the test
/// stricter, never weaker.
fn code_only(source: &str) -> String {
    source
        .lines()
        .map(|line| match line.find("//") {
            Some(index) => &line[..index],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every name the extension claims, read from the extension's own tables rather than restated
/// here. `starkc`'s tensor syntax module is `pub(crate)`, so an integration test cannot call it —
/// the list is therefore mirrored from the module's source text, which keeps the mirror honest by
/// failing loudly if the module is renamed or restructured.
fn extension_vocabulary() -> Vec<String> {
    let source = read_normalised(&crate_root().join("src/extensions/tensor/syntax.rs"));
    // Production code only. The module's own tests write *example* names — Core primitives like
    // `"Int32"`, and `"String"` as a negative case — which are not the extension's vocabulary.
    // Reading them would make the lint claim Core may not spell its own primitive types.
    let source = match source.find("#[cfg(test)]") {
        Some(index) => source[..index].to_string(),
        None => source,
    };
    let mut names: Vec<String> = Vec::new();
    let mut rest = source.as_str();
    while let Some(open) = rest.find('"') {
        rest = &rest[open + 1..];
        let Some(close) = rest.find('"') else { break };
        let literal = &rest[..close];
        rest = &rest[close + 1..];
        // Identifier-shaped literals only: the file's diagnostic sentences are not vocabulary.
        if !literal.is_empty()
            && literal
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
            && literal
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
        {
            names.push(literal.to_string());
        }
    }
    names.sort();
    names.dedup();
    assert!(
        names.len() >= 20,
        "expected the extension's syntax module to name at least twenty spellings, found {}: {names:?}",
        names.len()
    );
    for expected in ["Tensor", "QInt8", "Float16", "model", "input"] {
        assert!(
            names.iter().any(|n| n == expected),
            "`{expected}` should be among the extension's spellings; the mirror is stale"
        );
    }
    names
}

/// **Exit criterion 2.** No Core front-end module spells a name the extension owns.
#[test]
fn core_front_end_modules_do_not_spell_extension_names() {
    let vocabulary = extension_vocabulary();
    let mut violations: Vec<String> = Vec::new();

    for relative in CORE_FRONT_END {
        let path = crate_root().join(relative);
        let source = code_only(&read_normalised(&path));
        // The test module of a Core file legitimately writes tensor programs as fixtures.
        let production = match source.find("#[cfg(test)]") {
            Some(index) => &source[..index],
            None => source.as_str(),
        };
        for name in &vocabulary {
            let literal = format!("\"{name}\"");
            if production.contains(&literal) {
                violations.push(format!("{relative} spells {literal}"));
            }
        }
    }

    let mut accepted: Vec<String> = ACCEPTED_RESIDUE
        .iter()
        .map(|(file, name, _)| format!("{file} spells \"{name}\""))
        .collect();
    accepted.sort();
    violations.sort();

    let unexpected: Vec<&String> = violations
        .iter()
        .filter(|v| !accepted.contains(v))
        .collect();
    assert!(
        unexpected.is_empty(),
        "AS6 exit criterion 2: the extension's vocabulary belongs in `extensions/tensor/syntax.rs`, \
         and these Core modules restate it.\n  {}\n\
         Add the spelling to the extension's table and call it, as `parser.rs` and `resolve.rs` do.",
        unexpected
            .iter()
            .map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );

    let stale: Vec<&String> = accepted
        .iter()
        .filter(|a| !violations.contains(a))
        .collect();
    assert!(
        stale.is_empty(),
        "these accepted-residue entries no longer occur — the residue was cleaned up, so remove \
         them from ACCEPTED_RESIDUE rather than leaving an exemption nothing needs:\n  {}",
        stale
            .iter()
            .map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// **Exit criterion 1, structural half.** The three surfaces AS6's inventory found clean stay
/// clean. The behavioural half is `as6_core_session_isolation.rs`.
#[test]
fn surfaces_without_tensor_references_keep_none() {
    let mut violations: Vec<String> = Vec::new();
    for relative in MUST_STAY_TENSOR_FREE {
        let path = crate_root().join(relative);
        let source = read_normalised(&path).to_lowercase();
        for term in ["tensor", "dtype", "model"] {
            if source.contains(term) {
                violations.push(format!("{relative} mentions `{term}`"));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "these surfaces were extension-free at AS6's inventory and a first reference is a design \
         question, not a detail:\n  {}",
        violations.join("\n  ")
    );
}

/// **Exit criterion 4.** The extension boundary is internal. `pub mod extensions` predates AS6, but
/// nothing AS6 added may widen the supported surface — a `pub` item in the modules the packet
/// created would make the quarantine's internals part of the compiler's public API.
#[test]
fn as6_added_no_public_extension_api() {
    let mut violations: Vec<String> = Vec::new();
    for relative in [
        "src/extensions/tensor/check.rs",
        "src/extensions/tensor/syntax.rs",
    ] {
        let path = crate_root().join(relative);
        for (number, line) in read_normalised(&path).lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub ") && !trimmed.starts_with("pub(crate)") {
                violations.push(format!("{relative}:{} {}", number + 1, trimmed));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "AS6 exit criterion 4 forbids a public extension API; these items are `pub`:\n  {}",
        violations.join("\n  ")
    );
}
