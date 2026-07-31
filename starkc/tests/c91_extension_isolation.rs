use std::process::Command;

use starkc::analysis::{analyze_project, ProjectInput};
use starkc::diag::Severity;
use starkc::options::LanguageOptions;
use starkc::parser::ParseMode;
use starkc::source::SourceFile;

const TENSOR_DECL: &str = "\
model Resnet50V17<N: Dim> {
    input data: Tensor<Float32, [N, 3, 224, 224]>;
    output scores: Tensor<Float32, [N, 1000]>;
}
";

fn analyze(source: &str, options: LanguageOptions) -> starkc::analysis::ProjectAnalysis {
    analyze_project(
        ProjectInput::Source {
            file: SourceFile::new("c91.stark", source.to_string()).into(),
            mode: ParseMode::Program,
        },
        options,
    )
}

fn messages(analysis: &starkc::analysis::ProjectAnalysis) -> Vec<String> {
    analysis
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == Severity::Error)
        .map(|diagnostic| diagnostic.message.clone())
        .collect()
}

#[test]
fn c91_core_default_rejects_tensor_constructs_and_tensor_accepts_them() {
    let core = analyze(TENSOR_DECL, LanguageOptions::CORE);
    let core_messages = messages(&core);
    assert!(
        core_messages
            .iter()
            .any(|message| message.contains("`model` declarations require extension `tensor`")),
        "{core_messages:?}"
    );

    let tensor = analyze(TENSOR_DECL, LanguageOptions::with_tensor());
    assert!(
        messages(&tensor).is_empty(),
        "tensor-enabled analysis must accept the declaration: {:?}",
        messages(&tensor)
    );
}

#[test]
fn c91_sequential_and_parallel_analyses_do_not_share_extension_state() {
    let tensor_first = analyze(TENSOR_DECL, LanguageOptions::with_tensor());
    assert!(messages(&tensor_first).is_empty());

    let core_second = analyze(TENSOR_DECL, LanguageOptions::CORE);
    assert!(
        messages(&core_second)
            .iter()
            .any(|message| message.contains("extension `tensor`")),
        "{:?}",
        messages(&core_second)
    );

    let core_first = analyze(TENSOR_DECL, LanguageOptions::CORE);
    assert!(!messages(&core_first).is_empty());

    let tensor_second = analyze(TENSOR_DECL, LanguageOptions::with_tensor());
    assert!(messages(&tensor_second).is_empty());

    let core_handle = std::thread::spawn(|| messages(&analyze(TENSOR_DECL, LanguageOptions::CORE)));
    let tensor_handle =
        std::thread::spawn(|| messages(&analyze(TENSOR_DECL, LanguageOptions::with_tensor())));
    assert!(core_handle
        .join()
        .expect("core analysis thread must finish")
        .iter()
        .any(|message| message.contains("extension `tensor`")));
    assert!(tensor_handle
        .join()
        .expect("tensor analysis thread must finish")
        .is_empty());
}

#[test]
fn c91_cli_extension_config_rejects_unknown_and_duplicates() {
    for args in [
        vec![
            "check",
            "--extension",
            "unknown",
            "--stdin",
            "--filename",
            "c91.stark",
        ],
        vec![
            "check",
            "--extension",
            "tensor",
            "--extension",
            "tensor",
            "--stdin",
            "--filename",
            "c91.stark",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_starkc"))
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("run starkc");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("unknown extension") || stderr.contains("enabled more than once"),
            "{stderr}"
        );
    }
}
