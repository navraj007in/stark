//! AS5-a — DEV-184: three of the compiler's four JSON escapers emit invalid JSON.
//!
//! RFC 8259 §7: *"All Unicode characters may be placed within the quotation marks, except for the
//! characters that MUST be escaped: quotation mark, reverse solidus, and the control characters
//! (U+0000 through U+001F)."*
//!
//! Written to FAIL before the repair, per the packet's rule that a value-divergence finding is
//! repaired under its own DEV record with a fails-before-the-repair test rather than absorbed into
//! the consolidation commit. The consolidation onto one authority is AS5-c; this is the defect.
//!
//! The escaper this file does *not* exercise is `diag.rs::escape_json`, which was already correct
//! and is what the shared authority is built from.

/// Every C0 control character, plus the two that must always be escaped.
fn adversarial() -> String {
    let mut s = String::from("start");
    for byte in 0u8..0x20 {
        s.push(char::from(byte));
    }
    s.push('"');
    s.push('\\');
    s.push_str("end");
    s
}

/// A JSON string body is well-formed only if it contains no raw C0 control character.
fn raw_controls(escaped: &str) -> Vec<u32> {
    escaped
        .chars()
        .map(|c| c as u32)
        .filter(|&c| c < 0x20)
        .collect()
}

#[test]
fn the_lsp_transport_escapes_every_control_character() {
    // This is the wire protocol. GATE-C8-CLOSURE.md §4 records that C8's protocol validation
    // compared verdicts, not values — which is how DEV-182 passed it — so "the LSP suite is green"
    // does not establish that what goes on the wire is valid JSON.
    let rendered = starkc::lsp::protocol::JsonValue::String(adversarial()).to_string();
    assert!(
        raw_controls(&rendered).is_empty(),
        "the LSP transport emitted raw control characters {:?} inside a JSON string",
        raw_controls(&rendered)
    );
}

#[test]
fn the_onnx_report_escapes_every_control_character() {
    let escaped = starkc::onnx::escape_json(&adversarial());
    assert!(
        raw_controls(&escaped).is_empty(),
        "the ONNX report emitted raw control characters {:?} inside a JSON string",
        raw_controls(&escaped)
    );
}

/// `stark doctor --json` must emit a document a conforming parser accepts, even when a string it
/// reports carries a control character.
///
/// **The control character arrives through the manifest, not through the path.** The first version
/// of this test put a TAB in a directory name — legal on POSIX, and `InvalidFilename` on Windows,
/// where it failed the CI lane and proved nothing. Carrying it in the manifest's `version` field is
/// portable and tests more: the escape is decoded on the way in by the strict parser and must be
/// re-escaped on the way out, so this exercises the round trip through the real binary.
#[test]
fn doctor_json_is_parseable_when_a_reported_string_contains_a_control_character() {
    let exe = env!("CARGO_BIN_EXE_stark");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let root = std::env::temp_dir().join(format!("as5_doctor_{}_{nanos}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();

    // `\u0001` is a legal JSON escape; the value it denotes is a raw control character, which the
    // emitter must escape again rather than copy through.
    let backslash = char::from(92u8);
    std::fs::write(
        root.join("manifest.json"),
        format!(
            r#"{{"stark_version":"1.2.3{backslash}u0001dev","host_target":"x86_64-unknown-linux-gnu","files":[]}}"#
        ),
    )
    .unwrap();

    let output = std::process::Command::new(exe)
        .args(["doctor", "--json", "--root"])
        .arg(&root)
        .output()
        .expect("running stark doctor failed");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let _ = std::fs::remove_dir_all(&root);

    assert!(
        stdout.contains("\"version\""),
        "the probe must reach the JSON writer with a manifest in hand; got:\n{stdout}"
    );
    let offending: Vec<u32> = stdout
        .lines()
        .flat_map(|line| line.chars())
        .map(|c| c as u32)
        .filter(|&c| c < 0x20)
        .collect();
    assert!(
        offending.is_empty(),
        "stark doctor --json emitted raw control characters {offending:?}; \
         no conforming JSON parser accepts the document:\n{stdout}"
    );
}
