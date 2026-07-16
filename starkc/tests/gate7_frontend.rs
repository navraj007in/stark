//! Gate 7 (G7-02) — frontend symbolic-shape coverage.
//!
//! Hermetic: drives `starkc check --extension tensor` over the frozen Gate 7
//! pipeline plus focused positive/negative snippets. Every negative case
//! asserts that the defect is rejected with a source-level diagnostic (named
//! operation, source-level dimension names) and that **no internal dimension id
//! (`d<n>`) ever leaks** into user-visible output — the core G7-02 requirement.

mod common;

use common::TempDir;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn check_src(src: &str) -> Output {
    let dir = TempDir::new();
    let file = dir.write("case.stark", src.as_bytes());
    check_path(&file)
}

fn check_path(path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_starkc"))
        .args(["check", "--extension", "tensor"])
        .arg(path)
        .output()
        .expect("run starkc check")
}

fn diagnostics(o: &Output) -> String {
    // Diagnostics render to stderr; include stdout so a leak anywhere is caught.
    let mut s = String::from_utf8_lossy(&o.stderr).into_owned();
    s.push_str(&String::from_utf8_lossy(&o.stdout));
    s
}

/// True if `s` contains an internal dimension id such as `d3`, `?d3`, or
/// `21125*d3`: a lowercase `d` directly followed by ASCII digits and not
/// preceded by an alphanumeric (so `Float32`, `Int64`, `add`, `grid` never
/// match). The Gate 7 corpus never uses source dims spelled `d<n>`.
fn leaks_internal_id(s: &str) -> bool {
    let b = s.as_bytes();
    (0..b.len()).any(|i| {
        b[i] == b'd'
            && b.get(i + 1).is_some_and(u8::is_ascii_digit)
            && (i == 0 || !b[i - 1].is_ascii_alphanumeric())
    })
}

fn gate7_example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/gate7")
        .join(name)
}

fn assert_ok(label: &str, src: &str) {
    let out = check_src(src);
    let diag = diagnostics(&out);
    assert!(
        out.status.success(),
        "{label}: expected success, got:\n{diag}"
    );
    assert!(
        !diag.contains("Error"),
        "{label}: unexpected error:\n{diag}"
    );
}

/// Assert a negative case: checking fails with the given code, the operation is
/// named, every `must_contain` (source-level) fragment appears, and no internal
/// id leaks.
fn assert_rejected(label: &str, src: &str, code: &str, op: &str, must_contain: &[&str]) {
    let out = check_src(src);
    let diag = diagnostics(&out);
    assert_eq!(
        out.status.code(),
        Some(1),
        "{label}: expected checking to fail, got:\n{diag}"
    );
    assert!(diag.contains(code), "{label}: missing code {code}:\n{diag}");
    assert!(
        diag.contains(op),
        "{label}: operation `{op}` not named:\n{diag}"
    );
    for fragment in must_contain {
        assert!(
            diag.contains(fragment),
            "{label}: missing `{fragment}`:\n{diag}"
        );
    }
    assert!(
        !leaks_internal_id(&diag),
        "{label}: internal dimension id leaked:\n{diag}"
    );
}

// ---------------------------------------------------------------------------
// Positive cases
// ---------------------------------------------------------------------------

#[test]
fn frozen_valid_pipeline_checks() {
    let out = check_path(&gate7_example("valid_pipeline.stark"));
    let diag = diagnostics(&out);
    assert!(out.status.success(), "frozen pipeline must check:\n{diag}");
}

#[test]
fn frozen_model_signature_checks() {
    let out = check_path(&gate7_example("model.stark"));
    assert!(out.status.success(), "frozen model must check");
}

#[test]
fn symbolic_batch_preserved_across_helpers() {
    assert_ok(
        "batch across helpers",
        "fn helper<B: Dim>(x: Tensor<Float32, [B, 4]>) -> Tensor<Float32, [B, 4]> { x }\n\
         fn top<B: Dim>(x: Tensor<Float32, [B, 4]>) -> Tensor<Float32, [B, 4]> { helper(x) }",
    );
}

#[test]
fn polynomial_normalization_identity() {
    // A * (5 + C) == A*5 + A*C
    assert_ok(
        "poly normalization",
        "fn f<A: Dim, C: Dim>(x: Tensor<Float32, [A * (5 + C)]>) \
         -> Tensor<Float32, [A * 5 + A * C]> { x.reshape::<[A * 5 + A * C]>() }",
    );
}

#[test]
fn valid_reshape_product_equality() {
    assert_ok(
        "valid reshape",
        "fn f<B: Dim>(x: Tensor<Float32, [B, 6]>) -> Tensor<Float32, [B, 2, 3]> \
         { x.reshape::<[B, 2, 3]>() }",
    );
}

#[test]
fn valid_explicit_broadcast_to() {
    assert_ok(
        "valid broadcast_to",
        "fn f(x: Tensor<Float32, [3]>) -> Tensor<Float32, [2, 3]> { broadcast_to::<[2, 3]>(&x) }",
    );
}

#[test]
fn valid_elementwise_add_broadcast() {
    assert_ok(
        "valid add broadcast",
        "fn f<B: Dim>(a: Tensor<Float32, [B, 3, 4]>, b: Tensor<Float32, [3, 4]>) \
         -> Tensor<Float32, [B, 3, 4]> { add(&a, &b) }",
    );
}

#[test]
fn valid_elementwise_mul_broadcast() {
    assert_ok(
        "valid mul broadcast",
        "fn f<B: Dim>(a: Tensor<Float32, [B, 3, 4]>, b: Tensor<Float32, [1, 4]>) \
         -> Tensor<Float32, [B, 3, 4]> { mul(&a, &b) }",
    );
}

#[test]
fn literal_one_broadcast_leading_and_trailing() {
    // Leading 1 (a) and trailing 1 (b) both expand.
    assert_ok(
        "literal-one broadcast",
        "fn f(a: Tensor<Float32, [1, 3, 4]>, b: Tensor<Float32, [2, 3, 1]>) \
         -> Tensor<Float32, [2, 3, 4]> { add(&a, &b) }",
    );
}

#[test]
fn existing_symbolic_matmul_still_valid() {
    // Representative pre-Gate-7 symbolic tensor program (the full Gate 4/5
    // suites also run and must stay green).
    assert_ok(
        "symbolic matmul",
        "fn f<M: Dim, K: Dim, N: Dim>(a: Tensor<Float32, [M, K]>, b: Tensor<Float32, [K, N]>) \
         -> Tensor<Float32, [M, N]> { matmul(&a, &b) }",
    );
}

// ---------------------------------------------------------------------------
// Negative cases — each must fail with a source-level, id-free diagnostic
// ---------------------------------------------------------------------------

#[test]
fn invalid_reshape_product_is_rejected() {
    // 5 * 26 != 125
    assert_rejected(
        "invalid reshape product",
        "fn f<B: Dim>(x: Tensor<Float32, [B, 125]>) -> Tensor<Float32, [B, 5, 26]> \
         { x.reshape::<[B, 5, 26]>() }",
        "E0212",
        "reshape cannot preserve element count",
        &[
            "source shape: [B, 125]",
            "B * 125",
            "B * 5 * 26",
            "originates",
        ],
    );
}

#[test]
fn cross_symbol_dimensions_do_not_unify() {
    // B and B2 must not unify merely by spelling / possible runtime value.
    assert_rejected(
        "cross-symbol",
        "fn f<B: Dim, B2: Dim>(a: Tensor<Float32, [B, 5]>, b: Tensor<Float32, [B2, 5]>) \
         { let x = add(&a, &b); }",
        "E0212",
        "broadcast",
        &["[B, 5]", "[B2, 5]", "`B`", "`B2`"],
    );
}

#[test]
fn invalid_explicit_broadcast_is_rejected() {
    assert_rejected(
        "invalid broadcast_to",
        "fn f(x: Tensor<Float32, [2, 3]>) -> Tensor<Float32, [4, 3]> { broadcast_to::<[4, 3]>(&x) }",
        "E0212",
        "broadcast_to",
        &["source shape: [2, 3]", "target shape: [4, 3]", "axis 0"],
    );
}

#[test]
fn invalid_elementwise_add_is_rejected() {
    assert_rejected(
        "invalid add",
        "fn f<B: Dim, C: Dim>(a: Tensor<Float32, [B, 5]>, b: Tensor<Float32, [C, 5]>) \
         { let x = add(&a, &b); }",
        "E0212",
        "broadcast",
        &["axis 0", "`B`", "`C`"],
    );
}

#[test]
fn invalid_elementwise_mul_is_rejected() {
    assert_rejected(
        "invalid mul",
        "fn f(a: Tensor<Float32, [2, 5]>, b: Tensor<Float32, [3, 5]>) { let x = mul(&a, &b); }",
        "E0212",
        "broadcast",
        &["[2, 5]", "[3, 5]", "axis 0"],
    );
}

#[test]
fn unprovable_dimension_equality_is_rejected() {
    // matmul K vs K2: two distinct symbolic dims cannot be proven equal.
    assert_rejected(
        "unprovable equality",
        "fn f<K: Dim, K2: Dim>(a: Tensor<Float32, [2, K]>, b: Tensor<Float32, [K2, 3]>) \
         -> Tensor<Float32, [2, 3]> { matmul(&a, &b) }",
        "E0212",
        "matmul",
        &["`K`", "`K2`"],
    );
}

#[test]
fn possibly_negative_dimension_subtraction_is_rejected() {
    assert_rejected(
        "negative subtraction",
        "fn f<B: Dim>(x: Tensor<Float32, [B]>) -> Tensor<Float32, [B - 1]> { x.reshape::<[B - 1]>() }",
        "E0211",
        "subtraction may be negative",
        &[],
    );
}

#[test]
fn rank_mismatch_broadcast_is_rejected() {
    assert_rejected(
        "rank mismatch",
        "fn f(a: Tensor<Float32, [2, 3, 4]>) { let x = broadcast_to::<[3]>(&a); }",
        "E0212",
        "broadcast_to",
        &["rank mismatch", "source rank 3", "target rank 1"],
    );
}

#[test]
fn helper_return_losing_symbolic_batch_is_rejected() {
    // `top` claims to return [B2, 4] but forwards a [B, 4] value.
    assert_rejected(
        "helper drops batch",
        "fn helper<B: Dim>(x: Tensor<Float32, [B, 4]>) -> Tensor<Float32, [B, 4]> { x }\n\
         fn top<B: Dim, B2: Dim>(x: Tensor<Float32, [B, 4]>) -> Tensor<Float32, [B2, 4]> \
         { helper(x) }",
        "E0212",
        "dimension mismatch",
        &["`B`", "`B2`"],
    );
}

// ---------------------------------------------------------------------------
// Flagship: the reshape diagnostic is fully source-level (the known G7-01 gap)
// ---------------------------------------------------------------------------

#[test]
fn reshape_diagnostic_is_source_level_and_id_free() {
    let out = check_src(
        "fn f<B: Dim>(x: Tensor<Float32, [B, 125, 13, 13]>) \
         -> Tensor<Float32, [B, 5, 26, 13, 13]> { x.reshape::<[B, 5, 26, 13, 13]>() }",
    );
    let diag = diagnostics(&out);
    assert_eq!(out.status.code(), Some(1), "{diag}");
    for expected in [
        "reshape cannot preserve element count",
        "source shape: [B, 125, 13, 13]",
        "target shape: [B, 5, 26, 13, 13]",
        "required: B * 125 * 13 * 13 == B * 5 * 26 * 13 * 13",
        "dimension `B` originates from a function generic parameter",
    ] {
        assert!(diag.contains(expected), "missing `{expected}`:\n{diag}");
    }
    assert!(!leaks_internal_id(&diag), "internal id leaked:\n{diag}");
}
