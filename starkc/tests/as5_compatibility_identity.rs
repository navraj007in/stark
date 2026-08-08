//! AS5-g — a MIR shape change cannot pass tests without updating its compatibility identity.
//!
//! Exit criterion 4. `MIR_VERSION` and `MIR_RUNTIME_SURFACE` are the constants a consumer checks
//! before touching a body, and the build key hashes them, so a stale one serves a cached artifact
//! produced under different rules. Both were maintained by memory, and the record shows memory
//! failing: `MIR_RUNTIME_SURFACE`'s own doc comment states that CD-378 and CD-380 each added
//! `RuntimeFn` members **without advancing it**, and A14 corrected both omissions at once.
//!
//! # Why this is not an exact set of variant names
//!
//! The obvious cheap version — pin the `MirTy`, `Statement` and `Terminator` variant names — would
//! have missed the change that caused the most recent bump. AS1b-iii took MIR 0.3 → 0.4 by:
//!
//! ```text
//! SourceInfo.file      removed
//! MirProgram.files     removed
//! MirProgram.sources   introduced
//! FileId               eliminated
//! ```
//!
//! Not one variant name changed. A guard that cannot see its own most recent trigger is theatre.
//!
//! So the two constants get two different mechanisms, matching what each one means:
//!
//! ```text
//! MIR_RUNTIME_SURFACE   the set of runtime OPERATIONS      → exact canonical set of RuntimeFn
//! MIR_VERSION           the structural SHAPE of the model  → schema fingerprint over every
//!                                                            public type: variant names, payload
//!                                                            shapes, field names, field types
//! ```
//!
//! # Why the fingerprint is not a hash of the source text
//!
//! Hashing `mir/mod.rs` would make a comment edit or a `rustfmt` run look like a compatibility
//! change, and this file's types carry long doc comments that are edited often. The extractor below
//! reads the **declarations** and discards everything else, so the fingerprint moves when the shape
//! moves and at no other time.

use std::collections::BTreeSet;

/// Pinned alongside the constants they describe. Both must be updated in the same change as the
/// shape they identify — that is the whole mechanism.
const EXPECTED_MIR_VERSION: &str = "0.4";
const EXPECTED_MIR_SCHEMA_FINGERPRINT: &str = "ca001b20768a0ef8";
const EXPECTED_RUNTIME_SURFACE: &str = "0.1-A14";

fn mir_source() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("mir")
        .join("mod.rs");
    std::fs::read_to_string(path)
        .expect("mir/mod.rs must be readable")
        .replace("\r\n", "\n")
}

/// One declaration's canonical shape, e.g.
/// `struct SourceInfo{span:Span,origin:Origin}` or `enum Origin{UserCode,Synthetic(SyntheticKind)}`.
///
/// Comments, attributes, blank lines, visibility and formatting are all discarded. What survives is
/// the name, the kind, and — for every member — its own name and type.
fn schema(source: &str) -> Vec<String> {
    let lines: Vec<&str> = source.split('\n').collect();
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < lines.len() {
        let line = lines[index].trim();
        let (kind, rest) = if let Some(rest) = line.strip_prefix("pub struct ") {
            ("struct", rest)
        } else if let Some(rest) = line.strip_prefix("pub enum ") {
            ("enum", rest)
        } else {
            index += 1;
            continue;
        };

        // A one-line tuple struct: `pub struct LocalId(pub u32);`
        if rest.trim_end().ends_with(");") {
            if let (Some(open), Some(close)) = (rest.find('('), rest.rfind(')')) {
                let name = rest[..open].trim();
                out.push(format!(
                    "{kind} {name}({})",
                    normalise_members(&rest[open + 1..close])
                ));
                index += 1;
                continue;
            }
        }

        let Some(open) = rest.find('{') else {
            index += 1;
            continue;
        };
        let name = rest[..open].trim().to_string();

        // Accumulate the body until the brace that opened it closes.
        let mut body = String::new();
        let mut depth: i32 = 1;
        let mut cursor = index;
        let mut segment = rest[open + 1..].to_string();
        loop {
            let text = strip_noise(&segment);
            let mut kept = String::new();
            for character in text.chars() {
                match character {
                    '{' | '(' | '[' => depth += 1,
                    '}' | ')' | ']' => depth -= 1,
                    _ => {}
                }
                if depth == 0 {
                    break;
                }
                kept.push(character);
            }
            body.push_str(&kept);
            body.push(' ');
            if depth == 0 {
                break;
            }
            cursor += 1;
            if cursor >= lines.len() {
                break;
            }
            segment = lines[cursor].to_string();
        }
        out.push(format!("{kind} {name}{{{}}}", normalise_members(&body)));
        index = cursor + 1;
    }
    out
}

/// Drop line comments and attributes; keep code.
fn strip_noise(line: &str) -> String {
    let trimmed = line.trim();
    if trimmed.starts_with("//") || trimmed.starts_with("#[") || trimmed.starts_with("#!") {
        return String::new();
    }
    match trimmed.find("//") {
        Some(at) => trimmed[..at].trim_end().to_string(),
        None => trimmed.to_string(),
    }
}

/// Collapse whitespace so formatting cannot move the fingerprint, and drop `pub`, which is
/// visibility rather than shape.
fn normalise_members(body: &str) -> String {
    let mut out = String::new();
    let mut last_was_space = true;
    for character in body.replace("pub ", "").chars() {
        if character.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(character);
            last_was_space = false;
        }
    }
    // Spaces adjacent to punctuation are formatting, not shape.
    let mut canonical = out.trim().trim_end_matches(',').to_string();
    for punctuation in [':', ',', '<', '>', '(', ')', '{', '}'] {
        canonical = canonical
            .replace(&format!(" {punctuation}"), &punctuation.to_string())
            .replace(&format!("{punctuation} "), &punctuation.to_string());
    }
    canonical
}

/// A stable 64-bit digest. FNV-1a, written out rather than pulled in: the value only has to be
/// deterministic across runs and platforms, and a dependency for sixteen hex digits is not worth it.
fn fingerprint(entries: &[String]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for entry in entries {
        for byte in entry.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The load-bearing test: MIR's structural schema, fingerprinted.
///
/// If this fails, the MIR data model changed. Decide whether that is a shape change under contract
/// §11 — it almost always is, since this only sees declarations — advance `MIR_VERSION`, and update
/// both constants here in the same commit.
#[test]
fn the_mir_schema_fingerprint_matches_the_declared_version() {
    let entries = schema(&mir_source());

    // Non-vacuity: the extractor really did find the model, not an empty list.
    assert!(
        entries.len() >= 30,
        "the schema extractor found only {} declarations; it is not reading mir/mod.rs correctly:\n{}",
        entries.len(),
        entries.join("\n")
    );
    for required in [
        "struct SourceInfo{span:Span,origin:Origin}",
        "struct MirProgram{",
        "enum Terminator{",
        "enum Statement{",
        "enum MirTy{",
    ] {
        assert!(
            entries.iter().any(|entry| entry.starts_with(required)),
            "the extractor did not find `{required}`; it is not reading the shapes it claims to.\n\
             Found:\n{}",
            entries.join("\n")
        );
    }

    // And it must be sensitive to a field, not only to a type name — the AS1b-iii lesson.
    let source_info = entries
        .iter()
        .find(|entry| entry.starts_with("struct SourceInfo"))
        .expect("SourceInfo is in the schema");
    assert!(
        !source_info.contains("file:"),
        "SourceInfo still declares a `file` field: {source_info}"
    );

    assert_eq!(
        starkc::mir::MIR_VERSION,
        EXPECTED_MIR_VERSION,
        "MIR_VERSION moved without updating this test"
    );
    assert_eq!(
        fingerprint(&entries),
        EXPECTED_MIR_SCHEMA_FINGERPRINT,
        "\n\nMIR's structural schema changed.\n\n\
         This is a shape change under the MIR contract §11 unless you can say why it is not. \
         A consumer that accepted {EXPECTED_MIR_VERSION} may not be able to represent the new \
         model, and the build key hashes MIR_VERSION — a stale one serves a cached artifact \
         produced under the old shape.\n\n\
         Advance MIR_VERSION with a history entry saying what changed and why the increment is \
         load-bearing, then update EXPECTED_MIR_VERSION and EXPECTED_MIR_SCHEMA_FINGERPRINT here \
         in the SAME commit.\n\n\
         Current schema:\n{}\n",
        entries.join("\n")
    );
}

/// `MIR_RUNTIME_SURFACE` identifies the set of runtime OPERATIONS, so an exact set is the right
/// shape of check — and unlike the schema fingerprint, the members are worth reading in a diff.
///
/// A14's own doc comment records the failure this prevents: CD-378 added seven `Fmt*` members and
/// CD-380 added five `Fmt*Spec` members, **neither advancing the constant**, and A14 corrected both
/// at once. A consumer built against A13 cannot represent any of the twelve.
#[test]
fn the_runtime_surface_is_an_exact_set() {
    let source = mir_source();
    let start = source
        .find("pub enum RuntimeFn {")
        .expect("RuntimeFn is declared in mir/mod.rs");
    let body = &source[start..];
    let end = body.find("\n}").expect("RuntimeFn closes");
    let mut members = BTreeSet::new();
    for line in body[..end].split('\n').skip(1) {
        let text = strip_noise(line);
        let trimmed = text.trim().trim_end_matches(',');
        if trimmed.is_empty() {
            continue;
        }
        // A bare variant name, or the head of one carrying a payload.
        let name: String = trimmed
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() && name.chars().next().is_some_and(char::is_uppercase) {
            members.insert(name);
        }
    }

    assert!(
        members.len() > 50,
        "only {} runtime members found; the extractor is not reading RuntimeFn",
        members.len()
    );
    // The twelve A14 added, named explicitly: these are the ones that shipped without a bump.
    for formatting in [
        "FmtInt64",
        "FmtUInt64",
        "FmtBool",
        "FmtFloat64",
        "FmtFloat32",
        "FmtChar",
        "FmtUnit",
        "FmtPad",
        "FmtIntSpec",
        "FmtUIntSpec",
        "FmtFloat64Spec",
        "FmtFloat32Spec",
    ] {
        assert!(
            members.contains(formatting),
            "A14 declares {formatting}, which is not in RuntimeFn"
        );
    }

    assert_eq!(
        starkc::mir::MIR_RUNTIME_SURFACE,
        EXPECTED_RUNTIME_SURFACE,
        "MIR_RUNTIME_SURFACE moved without updating this test"
    );
    assert_eq!(
        fingerprint(&members.iter().cloned().collect::<Vec<_>>()),
        RUNTIME_SURFACE_FINGERPRINT,
        "\n\nThe set of runtime operations changed.\n\n\
         Adding, removing or renaming a `RuntimeFn` member changes what a consumer must be able to \
         represent; one that cannot must reject the program before consuming a body (V-SURFACE-1). \
         Advance MIR_RUNTIME_SURFACE and update this test in the SAME commit.\n\n\
         Current members ({}):\n{}\n",
        members.len(),
        members.iter().cloned().collect::<Vec<_>>().join("\n")
    );
}

const RUNTIME_SURFACE_FINGERPRINT: &str = "6e29d36352118e60";

/// The two constants describe different things, and the schema fingerprint must not silently
/// stand in for the runtime one. Stated as a test so the distinction survives a refactor.
#[test]
fn the_two_identities_are_independent() {
    let entries = schema(&mir_source());
    assert_ne!(
        fingerprint(&entries),
        RUNTIME_SURFACE_FINGERPRINT,
        "the schema and runtime-surface fingerprints must be computed over different things"
    );
}

/// The guard's own mutation test.
///
/// Mutating `mir/mod.rs` on disk cannot demonstrate this: adding a field or a variant stops the
/// crate compiling, so the test binary never runs and a broken guard looks identical to a working
/// one. The extractor reads text, so the mutations are applied to text.
///
/// Each case is a real historical shape change or a near-miss of one.
#[test]
fn the_schema_fingerprint_notices_a_shape_change() {
    let original = mir_source();
    let baseline = fingerprint(&schema(&original));
    assert_eq!(baseline, EXPECTED_MIR_SCHEMA_FINGERPRINT);

    let mutations: &[(&str, &str, &str)] = &[
        (
            "a field added to SourceInfo — AS1b-iii in reverse, and the case a variant-name set misses",
            "pub struct SourceInfo {\n    pub span: Span,",
            "pub struct SourceInfo {\n    pub span: Span,\n    pub something: u32,",
        ),
        (
            "a field removed from SourceInfo — AS1b-iii itself",
            "pub struct SourceInfo {\n    pub span: Span,\n    pub origin: Origin,\n}",
            "pub struct SourceInfo {\n    pub span: Span,\n}",
        ),
        (
            "a field RENAMED, with no variant anywhere touched",
            "    pub sources: crate::source::SourceTable,",
            "    pub source_table: crate::source::SourceTable,",
        ),
        (
            "a field's TYPE changed — MirProgram going back to a mutable registry",
            "pub sources: crate::source::SourceTable,",
            "pub sources: crate::source::SourceRegistry,",
        ),
        (
            "an enum variant's payload changed",
            "    Trap {\n        info: TrapInfo,",
            "    Trap {\n        info: TrapInfo,\n        extra: u8,",
        ),
        (
            "a whole type removed",
            "pub struct TrapInfo {",
            "pub struct TrapInfoRenamed {",
        ),
    ];

    for (what, from, to) in mutations {
        let mutated = original.replacen(from, to, 1);
        assert_ne!(
            mutated, original,
            "the mutation `{what}` did not apply; its anchor text has moved and the case is vacuous"
        );
        assert_ne!(
            fingerprint(&schema(&mutated)),
            baseline,
            "the schema fingerprint did NOT move for: {what}"
        );
    }
}

/// And the converse: things that are not shape must not move it, or the guard becomes noise that
/// people learn to re-pin without reading.
#[test]
fn the_schema_fingerprint_ignores_everything_that_is_not_shape() {
    let original = mir_source();
    let baseline = fingerprint(&schema(&original));

    let benign: &[(&str, &str, &str)] = &[
        (
            "a doc comment reworded",
            "/// Where a statement, terminator or trap came from.",
            "/// Where a statement, terminator or trap came from. Reworded entirely.",
        ),
        (
            "a line comment added inside a declaration",
            "pub struct SourceInfo {\n    pub span: Span,",
            "pub struct SourceInfo {\n    // an explanatory note\n    pub span: Span,",
        ),
        (
            "extra whitespace around a field",
            "pub struct TrapInfo {\n    pub category: TrapCategory,",
            "pub struct TrapInfo {\n\n    pub category:    TrapCategory,",
        ),
    ];

    for (what, from, to) in benign {
        let mutated = original.replacen(from, to, 1);
        assert_ne!(
            mutated, original,
            "the case `{what}` did not apply; its anchor text has moved and the case is vacuous"
        );
        assert_eq!(
            fingerprint(&schema(&mutated)),
            baseline,
            "the schema fingerprint moved for something that is not a shape change: {what}"
        );
    }
}
