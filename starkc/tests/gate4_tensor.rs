use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn example(name: &str) -> PathBuf {
    PathBuf::from("examples/gate4").join(name)
}

fn check(path: &Path, tensor: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_starkc"));
    command.current_dir(env!("CARGO_MANIFEST_DIR")).arg("check");
    if tensor {
        command.args(["--extension", "tensor"]);
    }
    command.arg(path).output().expect("run starkc check")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn normalize_diagnostic(diagnostic: &str) -> String {
    diagnostic.replace("\r\n", "\n").replace('\\', "/")
}

#[test]
fn diagnostic_golden_normalization_is_host_independent() {
    assert_eq!(
        normalize_diagnostic("  --> examples\\gate4\\bad_shape.stark:8:49\r\n"),
        "  --> examples/gate4/bad_shape.stark:8:49\n"
    );
}

#[test]
fn valid_gate4_declarations_and_pipeline_check_cleanly() {
    for name in ["reference_resnet50.stark", "valid_pipeline.stark"] {
        let output = check(&example(name), true);
        assert!(output.status.success(), "{name}: {}", stderr(&output));
        assert!(
            String::from_utf8_lossy(&output.stdout).contains(": OK"),
            "{name}: {:?}",
            output
        );
    }
}

#[test]
fn headline_defect_examples_fail_with_actionable_origins() {
    for (name, expected) in [
        (
            "bad_shape.stark",
            include_str!("snapshots/gate4_bad_shape.stderr"),
        ),
        (
            "bad_dtype.stark",
            include_str!("snapshots/gate4_bad_dtype.stderr"),
        ),
        (
            "bad_device.stark",
            include_str!("snapshots/gate4_bad_device.stderr"),
        ),
    ] {
        let output = check(&example(name), true);
        assert_eq!(output.status.code(), Some(1), "{name}");
        let diagnostic = stderr(&output);
        assert_eq!(
            normalize_diagnostic(&diagnostic),
            normalize_diagnostic(expected),
            "{name}"
        );
    }
}

#[test]
fn tensor_example_remains_isolated_from_core_mode() {
    let output = check(&example("valid_pipeline.stark"), false);
    assert_eq!(output.status.code(), Some(1));
    let diagnostic = stderr(&output);
    assert!(
        diagnostic.contains("`model` declarations require extension `tensor`"),
        "{diagnostic}"
    );
    assert!(
        diagnostic.contains("shape arguments require extension `tensor`"),
        "{diagnostic}"
    );
}
