//! Gate C9.1 — extension-isolation conformance.
//!
//! **Normative rule: `EXT-ISOLATION-001`** — *"How are post-v1 extensions prevented from silently
//! changing Core v1 behavior?"* (`spec/CORE-V1-FUTURE-BOUNDARIES.md` §Extension Isolation).
//!
//! The rule id is cited here because of C10-A2. `CORE-V1-COMPLETENESS.md` records this rule's
//! evidence as `none; none` while these five tests have run in CI on every push since C9.1 closed
//! — and C10-A2's resolver, which keys on normative rule ids rather than symbol names, could not
//! find them either, because nothing in this file named the rule it pins.
//!
//! **That is the whole lesson in one file.** A control that does not name its rule is invisible to
//! every mechanical audit, so it gets recorded as absent and then re-litigated. One line of
//! provenance is the difference between "no evidence exists" and "here is the evidence".

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

fn temp_package(name: &str, main_source: &str) -> std::path::PathBuf {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(format!(
            "temp_c91_{name}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time must be after epoch")
                .as_nanos()
        ));
    let src = root.join("src");
    std::fs::create_dir_all(&src).expect("create temp package src");
    std::fs::write(
        root.join("starkpkg.json"),
        r#"{"name":"c91_app","version":"0.1.0","entry":"src/main.stark"}"#,
    )
    .expect("write manifest");
    std::fs::write(src.join("main.stark"), main_source).expect("write main");
    root
}

fn command_output(bin: &str, args: &[&str], cwd: &std::path::Path) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap_or_else(|error| panic!("run {bin} {args:?}: {error}"))
}

/// EXT-ISOLATION-001, positive AND negative: extension syntax is REJECTED in Core-only and
/// ACCEPTED only when the extension is enabled.
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

/// EXT-ISOLATION-001: one session's extension set cannot leak into another's — the clause that
/// makes isolation a property of the compiler rather than of call order.
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

/// EXT-ISOLATION-001: unknown and duplicate extension configuration behaves consistently instead
/// of silently enabling or ignoring a surface.
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

/// EXT-ISOLATION-001: every entry point — package build, formatter, doc generator, test runner —
/// is Core-only by default, not merely the one the compiler happens to be invoked through.
#[test]
fn c91_package_and_tool_entry_points_are_core_only_without_extension_surface() {
    let root = temp_package("tool_core", TENSOR_DECL);

    let stark = env!("CARGO_BIN_EXE_stark");
    for (args, expected_code, expected_message) in [
        (
            vec!["check"],
            Some(1),
            "`model` declarations require extension `tensor`",
        ),
        (
            vec!["run"],
            Some(1),
            "`model` declarations require extension `tensor`",
        ),
        (
            vec!["fmt", "--check"],
            Some(1),
            "`model` declarations require extension `tensor`",
        ),
        (
            vec!["doc"],
            Some(1),
            "`model` declarations require extension `tensor`",
        ),
        (vec!["check", "--extension", "tensor"], Some(2), "Usage:"),
        (vec!["run", "--extension", "tensor"], Some(2), "Usage:"),
        (vec!["build", "--extension", "tensor"], Some(2), "Usage:"),
    ] {
        let output = command_output(stark, &args, &root);
        assert_eq!(
            output.status.code(),
            expected_code,
            "stark {args:?} stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            combined.contains(expected_message),
            "stark {args:?} did not contain {expected_message:?}: {combined}"
        );
    }

    let _ = std::fs::remove_dir_all(root);
}

/// EXT-ISOLATION-001: a package module cannot inherit enablement from an unrelated session.
#[test]
fn c91_package_modules_do_not_leak_tensor_enablement_from_other_sessions() {
    let root = temp_package("module_core", "mod tensor_mod;\nfn main() { }\n");
    std::fs::write(root.join("src/tensor_mod.stark"), TENSOR_DECL).expect("write module");
    let manifest = starkc::package::find_package_root(&root).expect("find manifest");
    let graph = starkc::package::PackageGraph::load_from_root(&manifest).expect("load package");

    let tensor_single = analyze(TENSOR_DECL, LanguageOptions::with_tensor());
    assert!(messages(&tensor_single).is_empty());

    let package_core = analyze_project(ProjectInput::package(graph), LanguageOptions::CORE);
    let package_messages = messages(&package_core);
    assert!(
        package_messages
            .iter()
            .any(|message| message.contains("`model` declarations require extension `tensor`")),
        "{package_messages:?}"
    );

    let _ = std::fs::remove_dir_all(root);
}
