use starkc::lsp::protocol::parse_json;
use std::process::Command;

#[test]
fn cli_json_diagnostics_are_structured_and_deterministic() {
    let unique = format!("stark-diagnostic-transport-{}", std::process::id());
    let directory = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("main.stark");
    std::fs::write(&source, "fn main() { missing; }\n").unwrap();

    let run = || {
        Command::new(env!("CARGO_BIN_EXE_starkc"))
            .args([
                "check",
                "--message-format",
                "json",
                source.to_str().unwrap(),
            ])
            .output()
            .unwrap()
    };
    let first = run();
    let second = run();
    assert!(!first.status.success());
    assert_eq!(first.stdout, second.stdout);
    assert!(first.stderr.is_empty());

    let json = String::from_utf8(first.stdout).unwrap();
    let parsed = parse_json(&json).expect("CLI diagnostic output must be valid JSON");
    assert_eq!(
        parsed.get("schemaVersion").and_then(|value| value.as_i64()),
        Some(1)
    );
    let diagnostic = &parsed
        .get("diagnostics")
        .and_then(|value| value.as_array())
        .unwrap()[0];
    assert_eq!(
        diagnostic.get("code").and_then(|value| value.as_str()),
        Some("E0200")
    );
    assert_eq!(
        diagnostic
            .get("primary")
            .and_then(|value| value.get("file"))
            .and_then(|value| value.as_str()),
        source.to_str()
    );
    assert!(diagnostic.get("related").is_some());
    assert!(diagnostic.get("sourceVersion").is_some());
    assert!(diagnostic.get("ruleId").is_some());
    assert!(diagnostic.get("deviationId").is_some());

    std::fs::remove_dir_all(directory).unwrap();
}
