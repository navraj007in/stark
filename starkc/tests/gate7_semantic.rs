//! Gate 7 (G7-04) — image value-range semantics.
//!
//! Hermetic: drives `starkc check --extension tensor` over the valid value-range
//! progression and the six required defect cases, asserting each defect is
//! rejected during checking with a diagnostic that names the value-range state.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn check_src(src: &str) -> Output {
    let dir = std::env::temp_dir().join(format!(
        "stark-g7sem-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("case.stark");
    std::fs::write(&file, src).unwrap();
    check_path(&file)
}

static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn check_path(path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_starkc"))
        .args(["check", "--extension", "tensor"])
        .arg(path)
        .output()
        .expect("run starkc check")
}

fn diagnostics(o: &Output) -> String {
    let mut s = String::from_utf8_lossy(&o.stderr).into_owned();
    s.push_str(&String::from_utf8_lossy(&o.stdout));
    s
}

/// A model whose input contract is `Normalized`, plus a boilerplate preamble
/// used by several defect cases.
const NORM_MODEL: &str = "model NormNet<B: Dim> {\n\
    \x20   input data: Tensor<Float32, [B, 3, 4, 4], range = Normalized>;\n\
    \x20   output logits: Tensor<Float32, [B, 10]>;\n\
    }\n";

fn assert_rejected(label: &str, src: &str, must_contain: &[&str]) {
    let out = check_src(src);
    let diag = diagnostics(&out);
    assert_eq!(
        out.status.code(),
        Some(1),
        "{label}: expected a checking failure, got:\n{diag}"
    );
    assert!(diag.contains("E0212"), "{label}: expected E0212:\n{diag}");
    for fragment in must_contain {
        assert!(
            diag.contains(fragment),
            "{label}: missing `{fragment}`:\n{diag}"
        );
    }
}

// ---- valid progression -------------------------------------------------------

#[test]
fn valid_value_range_progression_checks() {
    let out = check_path(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/gate7/range_pipeline.stark"),
    );
    let diag = diagnostics(&out);
    assert!(
        out.status.success(),
        "valid range pipeline must check:\n{diag}"
    );
}

// ---- the six required defects ------------------------------------------------

#[test]
fn defect1_omit_scaling_is_rejected() {
    // Stops at UnitRange but the model contract requires Normalized.
    let src = format!(
        "{NORM_MODEL}\
         fn infer<B: Dim>(model: NormNet, raw: TensorAny) -> Result<Tensor<Float32, [B, 10]>, String> {{\n\
         \x20 let bytes = raw.refine::<UInt8, [B, 3, 4, 4], range = ByteRange>()?;\n\
         \x20 Ok(model.predict(&bytes.cast::<Float32>().scale_255()))\n\
         }}"
    );
    assert_rejected(
        "omit normalization",
        &src,
        &["value-range mismatch", "Normalized", "UnitRange"],
    );
}

#[test]
fn defect2_double_scaling_is_rejected() {
    let src = "fn f(x: Tensor<Float32, [3, 4], range = ByteRange>) \
         -> Tensor<Float32, [3, 4], range = UnitRange> { x.scale_255().scale_255() }";
    assert_rejected(
        "scale_255 twice",
        src,
        &["`scale_255`", "requires a `ByteRange`", "UnitRange"],
    );
}

#[test]
fn defect3_normalize_byterange_is_rejected() {
    let src = "fn f(x: Tensor<Float32, [3, 4], range = ByteRange>) \
         -> Tensor<Float32, [3, 4], range = Normalized> { x.normalize() }";
    assert_rejected(
        "normalize a ByteRange",
        src,
        &["`normalize`", "requires a `UnitRange`", "ByteRange"],
    );
}

#[test]
fn defect4_unitrange_to_normalized_model_is_rejected() {
    let src = format!(
        "{NORM_MODEL}\
         fn infer<B: Dim>(model: NormNet, raw: TensorAny) -> Result<Tensor<Float32, [B, 10]>, String> {{\n\
         \x20 let unit = raw.refine::<Float32, [B, 3, 4, 4], range = UnitRange>()?;\n\
         \x20 Ok(model.predict(&unit))\n\
         }}"
    );
    assert_rejected(
        "UnitRange to Normalized model",
        &src,
        &["value-range mismatch", "Normalized", "UnitRange"],
    );
}

#[test]
fn defect5_merge_incompatible_ranges_is_rejected() {
    let src =
        "fn f(a: Tensor<Float32, [4], range = ByteRange>, b: Tensor<Float32, [4], range = UnitRange>) \
         -> Tensor<Float32, [4], range = ByteRange> { add(&a, &b) }";
    assert_rejected(
        "merge incompatible",
        src,
        &["cannot merge", "ByteRange", "UnitRange"],
    );
}

#[test]
fn defect6_erasing_range_without_boundary_is_rejected() {
    // Passing a ByteRange tensor where an Unspecified one is expected would
    // silently erase the semantic state.
    let src = "fn helper(x: Tensor<Float32, [4]>) -> Tensor<Float32, [4]> { x }\n\
               fn f(x: Tensor<Float32, [4], range = ByteRange>) -> Tensor<Float32, [4]> { helper(x) }";
    assert_rejected(
        "erase without boundary",
        src,
        &["value-range mismatch", "Unspecified", "ByteRange"],
    );
}

// ---- ordinary arithmetic does not claim a transition -------------------------

#[test]
fn ordinary_division_does_not_transition_range() {
    // Dividing a ByteRange tensor by a constant does NOT make it UnitRange; the
    // result stays ByteRange (only `scale_255` transitions), so feeding it to a
    // UnitRange contract is still an error.
    let src = "fn f(x: Tensor<Float32, [3, 4], range = ByteRange>, c: Tensor<Float32, [3, 4]>) \
         -> Tensor<Float32, [3, 4], range = UnitRange> { div(&x, &c) }";
    assert_rejected(
        "ordinary div keeps ByteRange",
        src,
        &["value-range mismatch", "UnitRange", "ByteRange"],
    );
}
