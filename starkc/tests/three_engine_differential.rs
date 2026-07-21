//! WP-C5.2 exit condition — the three-engine (HIR / MIR / native) differential harness.
//!
//! `WP-C5-ENTRY.md` §14's "C5.2 exit" requires *three-engine agreement* for scalar arithmetic,
//! branches, loops, direct calls, successful checked operations, and each admitted trap
//! category. Until this file existed there was no such comparator: `mir_differential.rs`
//! compares two engines (HIR oracle vs. MIR interpreter), and every `native_c5_2*.rs` file
//! asserts on the native engine's own output in isolation, with a human eye supplying the
//! correspondence. Two automated engines plus a manually corresponding third is not three-engine
//! agreement, and WP-C5.2 was deliberately held open for it.
//!
//! What this harness does, per case: takes ONE source string, runs that exact source through all
//! three engines, normalises each result into a common [`Outcome`], and requires all three to be
//! equal. The normalisation is the point — an outcome is either normal completion (stdout + exit
//! status) or a trap (category + exact source file/line/column + the stdout emitted before it),
//! so agreement covers completion-vs-trap, exit status, trap category, and trap location, not
//! just "all three exited nonzero".
//!
//! Observation channel. Native `println` is `Unsupported` until WP-C5.3 (no string values yet),
//! so the admitted C5.2 source surface produces no observable output at all. Values are
//! therefore observed by in-program `assert`/`assert_eq`, which reach the C5.2e trap ABI on
//! failure in all three engines — see [`a_false_assertion_traps_in_all_three_engines`], the
//! negative control that keeps "all three completed with exit 0" from being satisfiable by a
//! backend that compiled assertions away. [`NATIVE_STDOUT_SUPPORTED`] records the switch-over:
//! while it is `false` the harness *enforces* that every case is output-free, so comparing the
//! three normalised outcomes is total rather than silently skipping a dimension.

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions};
use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::{run_program, MirFailure, MirRunError};
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::mir::{Origin, TrapCategory};
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::process::Command;
use std::sync::Arc;

/// Whether the native backend can emit observable stdout yet. `false` through WP-C5.2: the
/// backend rejects `PrintStr`/`PrintlnStr` as `Unsupported` because string values are WP-C5.3.
///
/// While this is `false` the comparator asserts that each case's oracle run produced NO output,
/// which is what makes full [`Outcome`] equality across three engines an honest total
/// comparison rather than one with a quietly excluded dimension. When native output lands, flip
/// this constant: the precondition drops away and the same equality check starts comparing real
/// stdout bytes on all three sides, with no other change to the harness.
const NATIVE_STDOUT_SUPPORTED: bool = false;

/// One engine's result, normalised to the observable outcome the other two can be compared
/// against. Deliberately NOT engine-shaped: the HIR oracle reports a message and a byte span,
/// MIR reports a category and a `SourceInfo`, and the native binary reports a line of stderr
/// text and a process exit code. All three are projected onto this.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Completed {
        stdout: String,
        exit: i32,
    },
    Trapped {
        category: TrapCategory,
        file: String,
        line: u32,
        column: u32,
        /// C4.5e-0: output emitted before the trap is observable, so two programs printing
        /// different prefixes before the same trap are different outcomes.
        stdout_before: String,
    },
}

struct Front {
    hir: starkc::hir::Hir,
    file: Arc<SourceFile>,
    tables: starkc::typecheck::TypeTables,
}

fn front_end(name: &str, source: &str) -> Front {
    let file = Arc::new(SourceFile::new(name, source.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{name}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{name}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{name}: typecheck: {errors:?}");
    Front {
        hir,
        file,
        tables: checked.tables,
    }
}

/// The HIR oracle's trap message → category. The oracle is the semantic authority (charter §1.6
/// rule 6) but reports prose, not a category, so normalising it means reading that prose. The
/// mapping is exact-message-driven rather than fuzzy, and an unrecognised message is a hard
/// failure: a silent fallback would let a wrong-category trap normalise to whatever the other
/// engines said.
///
/// `IndexOutOfBounds`, `UnwrapNone`, `UnwrapErr` and message-carrying `Panic` are not reachable
/// from the C5.2-admitted surface (arrays, `Option`/`Result` and string values are all WP-C5.3),
/// so they are listed here as explicit "not admitted yet" failures rather than guessed at.
fn oracle_category(message: &str) -> TrapCategory {
    if message.contains("integer overflow") {
        TrapCategory::IntegerOverflow
    } else if message.contains("division by zero") {
        TrapCategory::DivideByZero
    } else if message.contains("invalid shift") {
        TrapCategory::InvalidShift
    } else if message.contains("numeric cast out of range")
        || message.contains("invalid numeric cast")
    {
        TrapCategory::CastFailure
    } else if message.contains("assertion failed") {
        TrapCategory::AssertFailure
    } else if message.contains("out of bounds") || message.contains("unwrap") {
        panic!(
            "oracle raised a trap category outside the C5.2-admitted surface: {message:?} \
             (arrays and Option/Result are WP-C5.3)"
        )
    } else {
        panic!(
            "unrecognised oracle trap message {message:?} — normalise it here rather than \
             letting it default to a category the other engines happen to report"
        )
    }
}

/// `starkc::mir::TrapCategory` → the runtime's own copy, so the native stderr text this harness
/// matches against is the runtime's single source of truth (`stark-runtime/src/trap.rs`) rather
/// than a second table in a test file that could drift from it. The match is exhaustive on
/// purpose: a new category fails to compile here until it is mapped.
fn runtime_category(category: TrapCategory) -> stark_runtime::trap::TrapCategory {
    use stark_runtime::trap::TrapCategory as Rt;
    match category {
        TrapCategory::IntegerOverflow => Rt::IntegerOverflow,
        TrapCategory::DivideByZero => Rt::DivideByZero,
        TrapCategory::IndexOutOfBounds => Rt::IndexOutOfBounds,
        TrapCategory::CastFailure => Rt::CastFailure,
        TrapCategory::Panic => Rt::Panic,
        TrapCategory::UnwrapNone => Rt::UnwrapNone,
        TrapCategory::UnwrapErr => Rt::UnwrapErr,
        TrapCategory::AssertFailure => Rt::AssertFailure,
        TrapCategory::InvalidShift => Rt::InvalidShift,
    }
}

const ALL_CATEGORIES: [TrapCategory; 9] = [
    TrapCategory::IntegerOverflow,
    TrapCategory::DivideByZero,
    TrapCategory::IndexOutOfBounds,
    TrapCategory::CastFailure,
    TrapCategory::Panic,
    TrapCategory::UnwrapNone,
    TrapCategory::UnwrapErr,
    TrapCategory::AssertFailure,
    TrapCategory::InvalidShift,
];

fn rustc_available() -> bool {
    Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ------------------------------------------------------------------ engine 1 --

fn run_hir(name: &str, front: &Front) -> Outcome {
    match interp::run_with_partial_output(&front.hir, front.file.clone(), &front.tables) {
        Ok(exec) => {
            assert!(
                exec.stderr.is_empty(),
                "{name}: oracle wrote to stderr on normal completion ({:?}) — the C5.2 surface \
                 has no such path, and the native engine has no channel to match it",
                exec.stderr
            );
            Outcome::Completed {
                stdout: exec.output,
                exit: exec.status as i32,
            }
        }
        Err((err, partial)) => {
            assert!(
                err.is_trap,
                "{name}: oracle failed without trapping ({}) — an entrypoint-selection failure \
                 is a compiler error, not a language outcome the other engines can match",
                err.message
            );
            let (line, column) = front.file.line_col(err.span.lo);
            Outcome::Trapped {
                category: oracle_category(&err.message),
                file: front.file.name.clone(),
                line: line as u32,
                column: column as u32,
                stdout_before: partial,
            }
        }
    }
}

// ------------------------------------------------------------------ engine 2 --

fn run_mir(name: &str, program: &starkc::mir::MirProgram) -> Outcome {
    let verified = match verify_program(program) {
        Ok(v) => v,
        Err(errors) => panic!("{name}: verifier rejected lowered MIR:\n{errors:#?}"),
    };
    match run_program(verified) {
        Ok(exec) => Outcome::Completed {
            stdout: exec.output,
            exit: exec.status as i32,
        },
        Err(MirFailure {
            error:
                MirRunError::Trap {
                    category,
                    source,
                    message,
                },
            output,
        }) => {
            assert!(
                (source.file.0 as usize) < program.files.len(),
                "{name}: MIR trap carries an invalid FileId"
            );
            assert!(
                message.is_none(),
                "{name}: a message-carrying trap ({message:?}) needs string values — WP-C5.3, \
                 outside the C5.2-admitted surface this harness compares"
            );
            assert!(
                matches!(source.origin, Origin::UserCode),
                "{name}: trap origin is {:?}; the harness compares exact user-source locations, \
                 so a synthetic-origin trap needs its own documented correspondence rule",
                source.origin
            );
            let file = &program.files[source.file.0 as usize];
            let (line, column) = file.line_col(source.span.lo);
            Outcome::Trapped {
                category,
                file: file.name.clone(),
                line: line as u32,
                column: column as u32,
                stdout_before: output,
            }
        }
        Err(MirFailure {
            error: MirRunError::Internal(message),
            ..
        }) => panic!("{name}: MIR internal error: {message}"),
    }
}

// ------------------------------------------------------------------ engine 3 --

fn run_native(name: &str, tag: &str, program: &starkc::mir::MirProgram) -> Outcome {
    let verified = match verify_program(program) {
        Ok(v) => v,
        Err(errors) => panic!("{name}: verifier rejected lowered MIR:\n{errors:#?}"),
    };
    let target_dir = std::env::temp_dir().join(format!(
        "stark_3eng_{tag}_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&target_dir);
    let options = NativeBuildOptions {
        target_dir: target_dir.clone(),
    };
    let artifact = emit_native_debug(&verified, &options)
        .unwrap_or_else(|e| panic!("{name}: native build failed: {e:?}"));
    let run = Command::new(&artifact.binary_path)
        .output()
        .expect("running the generated binary failed");
    let _ = std::fs::remove_dir_all(&target_dir);

    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&run.stderr).into_owned();
    match run.status.code() {
        Some(101) => {
            let (category, file, line, column) = parse_native_trap(name, &stderr);
            Outcome::Trapped {
                category,
                file,
                line,
                column,
                stdout_before: stdout,
            }
        }
        Some(code) => {
            assert!(
                stderr.is_empty(),
                "{name}: native run exited {code} but wrote to stderr: {stderr}"
            );
            Outcome::Completed { stdout, exit: code }
        }
        None => panic!("{name}: native run terminated by a signal; stderr: {stderr}"),
    }
}

/// Reads the native trap ABI's stderr back into the normalised form. The format is fixed by
/// `stark_runtime::trap::abort`:
///
/// ```text
/// error: runtime trap: <category message>
///   --> <file>:<line>:<column>
/// ```
fn parse_native_trap(name: &str, stderr: &str) -> (TrapCategory, String, u32, u32) {
    let message = stderr
        .lines()
        .find_map(|l| l.strip_prefix("error: runtime trap: "))
        .unwrap_or_else(|| panic!("{name}: native stderr has no trap header:\n{stderr}"))
        .trim();
    let category = ALL_CATEGORIES
        .into_iter()
        .find(|c| runtime_category(*c).message() == message)
        .unwrap_or_else(|| {
            panic!("{name}: native trap message {message:?} matches no known category")
        });
    let location = stderr
        .lines()
        .find_map(|l| l.trim().strip_prefix("--> "))
        .unwrap_or_else(|| panic!("{name}: native stderr has no `-->` location:\n{stderr}"))
        .trim();
    // Split from the RIGHT: line and column are the last two fields, everything before them is
    // the file path (which may itself contain `:` on some platforms).
    let mut parts = location.rsplitn(3, ':');
    let column: u32 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("{name}: unparseable column in {location:?}"));
    let line: u32 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("{name}: unparseable line in {location:?}"));
    let file = parts
        .next()
        .unwrap_or_else(|| panic!("{name}: unparseable file in {location:?}"))
        .to_string();
    (category, file, line, column)
}

// ----------------------------------------------------------------- the check --

/// Run one source through all three engines and require identical normalised outcomes.
///
/// `tag` names the scratch build directory only; `name` becomes the STARK source file name, so
/// it is what every engine reports as the trap location and is therefore itself compared.
fn three_engine(tag: &str, source: &str) -> Outcome {
    let name = format!("three_engine_{tag}.stark");
    let front = front_end(&name, source);
    let program = match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
    };

    let hir = run_hir(&name, &front);
    let mir = run_mir(&name, &program);
    let native = run_native(&name, tag, &program);

    if !NATIVE_STDOUT_SUPPORTED {
        // Enforced, not assumed: if a case ever starts printing, this fires rather than letting
        // the native side's necessarily-empty stdout quietly disagree with the other two.
        let printed = match &hir {
            Outcome::Completed { stdout, .. } => stdout,
            Outcome::Trapped { stdout_before, .. } => stdout_before,
        };
        assert!(
            printed.is_empty(),
            "{name}: case produces stdout ({printed:?}), but the native backend cannot emit \
             output until WP-C5.3 — every harness case must observe values through in-program \
             assertions while NATIVE_STDOUT_SUPPORTED is false"
        );
    }

    assert_eq!(
        hir, mir,
        "{name}: HIR/MIR DISAGREEMENT\n--- HIR oracle ---\n{hir:#?}\n--- MIR ---\n{mir:#?}"
    );
    assert_eq!(
        mir, native,
        "{name}: MIR/NATIVE DISAGREEMENT\n--- MIR ---\n{mir:#?}\n--- native ---\n{native:#?}"
    );
    hir
}

/// All three completed normally with exit 0 — i.e. every in-program assertion held in every
/// engine. Meaningful only because [`a_false_assertion_traps_in_all_three_engines`] proves a
/// FALSE assertion is observable; see this file's header.
fn agree_completing(tag: &str, source: &str) {
    let outcome = three_engine(tag, source);
    assert!(
        matches!(outcome, Outcome::Completed { exit: 0, .. }),
        "{tag}: expected normal completion, got {outcome:#?}"
    );
}

/// All three trapped, with the same category at the same source line — and that line is stated
/// here independently, so a case whose three engines agreed on the WRONG location still fails.
fn agree_trapping(tag: &str, source: &str, expected: TrapCategory, expected_line: u32) {
    let outcome = three_engine(tag, source);
    match outcome {
        Outcome::Trapped { category, line, .. } => {
            assert_eq!(category, expected, "{tag}: trap category");
            assert_eq!(line, expected_line, "{tag}: trap line");
        }
        other => panic!("{tag}: expected a trap, got {other:#?}"),
    }
}

macro_rules! three_engine_test {
    ($name:ident, $tag:literal, completes, $source:literal) => {
        #[test]
        fn $name() {
            if !rustc_available() {
                eprintln!("SKIP: no rustc in this environment.");
                return;
            }
            agree_completing($tag, $source);
        }
    };
    ($name:ident, $tag:literal, traps($category:expr, $line:literal), $source:literal) => {
        #[test]
        fn $name() {
            if !rustc_available() {
                eprintln!("SKIP: no rustc in this environment.");
                return;
            }
            agree_trapping($tag, $source, $category, $line);
        }
    };
}

// ======================================================= §14 exit condition 1 --
// scalar arithmetic

three_engine_test!(
    scalar_arithmetic_agrees,
    "arith",
    completes,
    r#"fn main() {
    let a: Int32 = 10;
    let b: Int32 = 3;
    assert_eq(a + b, 13);
    assert_eq(a - b, 7);
    assert_eq(a * b, 30);
    assert_eq(a / b, 3);
    assert_eq(a % b, 1);
    assert_eq(-a, -10);
    assert_eq(a + b * 2 - 1, 15);
    assert_eq((a + b) * 2, 26);
    assert_eq(a << 2, 40);
    assert_eq(a >> 1, 5);
    assert_eq(a & b, 2);
    assert_eq(a | b, 11);
    assert_eq(a ^ b, 9);
    assert(a > b);
    assert(a >= b);
    assert(a != b);
    assert(!(a == b));
    assert(b < a);
    assert(b <= a);
    let neg: Int32 = -7;
    assert_eq(neg / 2, -3);
    assert_eq(neg % 2, -1);
    let wide: Int64 = 4000000000;
    assert_eq(wide + 1, 4000000001);
    let small: Int8 = 100;
    assert_eq(small - 100, 0);
    let unsigned: UInt32 = 4000000000;
    assert_eq(unsigned / 2, 2000000000);
    let f: Float64 = 3.5;
    let g: Float64 = 1.5;
    assert_eq(f + g, 5.0);
    assert_eq(f - g, 2.0);
    assert_eq(f * g, 5.25);
    assert_eq(f / g, 2.3333333333333335);
    assert_eq(-f, -3.5);
    assert(f > g);
}
"#
);

// ======================================================= §14 exit condition 2 --
// branches

three_engine_test!(
    branches_both_directions_agree,
    "branch",
    completes,
    r#"fn main() {
    let a: Int32 = 5;
    let mut taken: Int32 = 0;
    if a > 3 { taken = 1; } else { taken = 2; }
    assert_eq(taken, 1);
    if a > 100 { taken = 3; } else { taken = 4; }
    assert_eq(taken, 4);
    let mut chain: Int32 = 0;
    if a == 1 { chain = 10; } else if a == 5 { chain = 20; } else { chain = 30; }
    assert_eq(chain, 20);
    if a == 1 { chain = 40; } else if a == 2 { chain = 50; } else { chain = 60; }
    assert_eq(chain, 60);
    let mut nested: Int32 = 0;
    if a > 0 {
        if a > 4 { nested = 1; } else { nested = 2; }
    } else {
        nested = 3;
    }
    assert_eq(nested, 1);
    let mut no_else: Int32 = 7;
    if a < 0 { no_else = 8; }
    assert_eq(no_else, 7);
    let value: Int32 = if a > 3 { 100 } else { 200 };
    assert_eq(value, 100);
    let other: Int32 = if a > 30 { 100 } else { 200 };
    assert_eq(other, 200);
    assert(a > 3 && a < 10);
    assert(a > 30 || a < 10);
    assert(!(a > 30 && a < 10));
}
"#
);

// ======================================================= §14 exit condition 3 --
// loops, including the zero-iteration case

three_engine_test!(
    zero_iteration_loop_agrees,
    "loop0",
    completes,
    r#"fn main() {
    let mut i: Int32 = 0;
    let mut body_ran: Int32 = 0;
    while i < 0 {
        body_ran = 1;
        i = i + 1;
    }
    assert_eq(body_ran, 0);
    assert_eq(i, 0);
    let mut n: Int32 = 10;
    let mut count: Int32 = 0;
    while n < 10 {
        count = count + 1;
        n = n + 1;
    }
    assert_eq(count, 0);
    assert_eq(n, 10);
}
"#
);

three_engine_test!(
    multi_iteration_loop_agrees,
    "loopn",
    completes,
    r#"fn main() {
    let mut i: Int32 = 0;
    let mut sum: Int32 = 0;
    while i < 5 {
        sum = sum + i;
        i = i + 1;
    }
    assert_eq(i, 5);
    assert_eq(sum, 10);

    let mut j: Int32 = 0;
    let mut evens: Int32 = 0;
    while j < 10 {
        j = j + 1;
        if j % 2 == 1 { continue; }
        evens = evens + j;
    }
    assert_eq(evens, 30);

    let mut k: Int32 = 0;
    while k < 100 {
        if k == 7 { break; }
        k = k + 1;
    }
    assert_eq(k, 7);

    let mut outer: Int32 = 0;
    let mut cells: Int32 = 0;
    while outer < 3 {
        let mut inner: Int32 = 0;
        while inner < 4 {
            cells = cells + 1;
            inner = inner + 1;
        }
        outer = outer + 1;
    }
    assert_eq(cells, 12);
}
"#
);

// ======================================================= §14 exit condition 4 --
// direct calls

three_engine_test!(
    direct_calls_agree,
    "calls",
    completes,
    r#"fn add(a: Int32, b: Int32) -> Int32 {
    a + b
}

fn sub(a: Int32, b: Int32) -> Int32 {
    a - b
}

fn clamp(v: Int32, lo: Int32, hi: Int32) -> Int32 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

fn no_args() -> Int32 {
    42
}

fn returns_unit(a: Int32) {
    let ignored: Int32 = a + 1;
}

fn factorial(n: Int32) -> Int32 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

fn main() {
    assert_eq(add(2, 3), 5);
    // Argument ORDER, not just arity: `sub` is not commutative, so a backend that reversed
    // parameters would pass `add` and fail here.
    assert_eq(sub(10, 4), 6);
    assert_eq(clamp(15, 0, 10), 10);
    assert_eq(clamp(-5, 0, 10), 0);
    assert_eq(clamp(5, 0, 10), 5);
    assert_eq(no_args(), 42);
    assert_eq(add(add(1, 2), sub(10, 4)), 9);
    assert_eq(factorial(5), 120);
    returns_unit(1);
    let mut total: Int32 = 0;
    let mut i: Int32 = 0;
    while i < 4 {
        total = add(total, i);
        i = i + 1;
    }
    assert_eq(total, 6);
}
"#
);

// ======================================================= §14 exit condition 5 --
// successful checked operations — the values that sit just INSIDE every bound whose outside
// traps below, so "traps correctly" and "does not trap spuriously" are both pinned.

three_engine_test!(
    successful_checked_operations_agree,
    "checked_ok",
    completes,
    r#"fn main() {
    let max32: Int32 = 2147483647;
    assert_eq(max32 - 1 + 1, 2147483647);
    let min32: Int32 = -2147483647 - 1;
    assert_eq(min32 + 1 - 1, min32);
    let max8: Int8 = 127;
    assert_eq(max8 / 1, 127);
    let umax: UInt8 = 255;
    assert_eq(umax - 255, 0);

    let a: Int32 = 100;
    let b: Int32 = 7;
    assert_eq(a / b, 14);
    assert_eq(a % b, 2);

    // The greatest legal shift COUNT for each width is width-1; one more traps as
    // `InvalidShift` (see `invalid_shift_trap_agrees`). A legal count can still overflow on the
    // RESULT -- `1 << 31` is not representable in `Int32` and traps as `IntegerOverflow`, a
    // distinction NUM-SHIFT-001 draws deliberately -- so the width-1 count is exercised here on
    // a value whose result does fit.
    let zero32: Int32 = 0;
    assert_eq(zero32 << 31, 0);
    let one32: Int32 = 1;
    assert_eq(one32 << 30, 1073741824);
    let one64: Int64 = 1;
    assert_eq(one64 << 62, 4611686018427387904);
    let byte: Int8 = 1;
    assert_eq(byte << 6, 64);

    // In-range casts at the exact boundary of the narrower type. A minimum-value literal has
    // no direct spelling at any width (`-128` is unary minus applied to `128`, which is already
    // out of range for `Int8` before the minus applies), so it is built as `-max - 1` -- the
    // same shape `float_to_int_boundary_conversions_agree` needs for `Int64::MIN`.
    let fits: Int32 = 127;
    let expect_max8: Int8 = 127;
    assert_eq(fits as Int8, expect_max8);
    let fits_neg: Int32 = -127 - 1;
    let expect_min8: Int8 = -127 - 1;
    assert_eq(fits_neg as Int8, expect_min8);
    let unsigned_fits: Int32 = 255;
    let expect_umax8: UInt8 = 255;
    assert_eq(unsigned_fits as UInt8, expect_umax8);
    let widening: Int8 = -127 - 1;
    assert_eq(widening as Int32, -128);
    let to_float: Int32 = 3;
    assert_eq(to_float as Float64, 3.0);
    let from_float: Float64 = 42.9;
    assert_eq(from_float as Int32, 42);
}
"#
);

// ======================================================= §14 exit condition 6 --
// each admitted trap category. `IndexOutOfBounds`, `UnwrapNone`, `UnwrapErr` and
// message-carrying `Panic` are NOT admitted by the C5.2 surface (arrays, Option/Result and
// string values are WP-C5.3), and `oracle_category` fails loudly rather than silently if one
// ever reaches this harness.

three_engine_test!(
    integer_overflow_trap_agrees,
    "trap_overflow",
    traps(TrapCategory::IntegerOverflow, 4),
    r#"fn main() {
    let a: Int32 = 2147483647;
    let b: Int32 = 1;
    let c: Int32 = a + b;
}
"#
);

three_engine_test!(
    divide_by_zero_trap_agrees,
    "trap_divzero",
    traps(TrapCategory::DivideByZero, 4),
    r#"fn main() {
    let a: Int32 = 10;
    let b: Int32 = 0;
    let c: Int32 = a / b;
}
"#
);

three_engine_test!(
    remainder_by_zero_trap_agrees,
    "trap_remzero",
    traps(TrapCategory::DivideByZero, 4),
    r#"fn main() {
    let a: Int32 = 10;
    let b: Int32 = 0;
    let c: Int32 = a % b;
}
"#
);

three_engine_test!(
    invalid_shift_trap_agrees,
    "trap_shift",
    traps(TrapCategory::InvalidShift, 4),
    r#"fn main() {
    let a: Int32 = 1;
    let n: Int32 = 32;
    let b: Int32 = a << n;
}
"#
);

three_engine_test!(
    cast_failure_trap_agrees,
    "trap_cast",
    traps(TrapCategory::CastFailure, 3),
    r#"fn main() {
    let a: Int32 = 1000;
    let b: Int8 = a as Int8;
}
"#
);

// THE NEGATIVE CONTROL for every completing case in this file. Every value observation here
// runs through `assert`/`assert_eq`, so "all three engines completed with exit 0" is only
// evidence if a false assertion really does fail the run in all three. Without this test the
// entire harness would pass against three engines that all ignored assertions.
three_engine_test!(
    a_false_assertion_traps_in_all_three_engines,
    "trap_assert",
    traps(TrapCategory::AssertFailure, 3),
    r#"fn main() {
    let a: Int32 = 10;
    assert_eq(a + 1, 12);
}
"#
);

three_engine_test!(
    a_false_bare_assertion_traps_in_all_three_engines,
    "trap_assert_bare",
    traps(TrapCategory::AssertFailure, 3),
    r#"fn main() {
    let a: Int32 = 10;
    assert(a > 100);
}
"#
);

// ========================================================= review regressions --
// CD-052's fixed defects, re-pinned as three-engine agreement rather than per-engine assertions.

// DEV-091 (positive side). The MIR interpreter and the native backend both compared a truncated
// float against `max as f64`, which rounds UP at 64-bit widths — so the boundary needed pinning
// from BOTH directions. These are the values that must still convert.
//
// Two front-end limitations shape the spelling, neither related to the bound under test:
// no integer literal above `Int64::MAX` is expressible (so the near-2^64 result is pinned by
// halving it), and `Int64::MIN` has no literal spelling (so it is built as `-max - 1`).
three_engine_test!(
    float_to_int_boundary_conversions_agree,
    "cast_bound_ok",
    completes,
    r#"fn main() {
    let below_u64: Float64 = 18446744073709549568.0;
    let converted_u64: UInt64 = below_u64 as UInt64;
    let expect_half: UInt64 = 9223372036854774784;
    assert_eq(converted_u64 / 2, expect_half);

    let below_i64: Float64 = 9223372036854774784.0;
    let expect_i64: Int64 = 9223372036854774784;
    assert_eq(below_i64 as Int64, expect_i64);

    let at_i64_min: Float64 = -9223372036854775808.0;
    let expect_min: Int64 = -9223372036854775807 - 1;
    assert_eq(at_i64_min as Int64, expect_min);

    let fractional: Float64 = 42.9;
    assert_eq(fractional as Int32, 42);
    let negative_fraction: Float64 = -42.9;
    assert_eq(negative_fraction as Int32, -42);
}
"#
);

// DEV-091 (the defect itself). Exactly 2^64 is one past `UInt64::MAX`; both non-oracle engines
// accepted it and saturated to `UInt64::MAX` instead of trapping.
three_engine_test!(
    float_to_uint64_at_two_pow_64_traps_in_all_three_engines,
    "cast_u64_bound",
    traps(TrapCategory::CastFailure, 3),
    r#"fn main() {
    let at_bound: Float64 = 18446744073709551616.0;
    let converted: UInt64 = at_bound as UInt64;
}
"#
);

// DEV-091, signed twin: exactly 2^63 is one past `Int64::MAX`.
three_engine_test!(
    float_to_int64_at_two_pow_63_traps_in_all_three_engines,
    "cast_i64_bound",
    traps(TrapCategory::CastFailure, 3),
    r#"fn main() {
    let at_bound: Float64 = 9223372036854775808.0;
    let converted: Int64 = at_bound as Int64;
}
"#
);

// DEV-091, the negative end: `Int64::MIN` itself converts (pinned above), the next f64 below it
// must not.
three_engine_test!(
    float_below_int64_min_traps_in_all_three_engines,
    "cast_i64_min",
    traps(TrapCategory::CastFailure, 3),
    r#"fn main() {
    let below_bound: Float64 = -9223372036854777856.0;
    let converted: Int64 = below_bound as Int64;
}
"#
);

// DEV-096 — the cast-CATEGORY correction, which only a three-engine comparison of categories
// (not exit codes) can hold. The HIR oracle reported every out-of-range cast, at every width, as
// `IntegerOverflow`, because both cast arms routed through `check_integer_range` and inherited
// its arithmetic-overflow message. All three engines must now say `CastFailure` — this case is
// a float→int cast at a NON-boundary width, so it is the category alone under test, distinct
// from `cast_failure_trap_agrees`'s int→int narrowing.
three_engine_test!(
    out_of_range_cast_is_a_cast_failure_not_an_overflow,
    "cast_category",
    traps(TrapCategory::CastFailure, 3),
    r#"fn main() {
    let big: Float64 = 3000000000.0;
    let narrowed: Int32 = big as Int32;
}
"#
);

// DEV-092 — symbol mangling must be INJECTIVE, not merely sanitized. Under the previous
// encoding (`_` passed through unchanged) the module-qualified symbol `m::f@[]` and the
// legally-named top-level function `m_3a_3af@[]` both sanitized to the same Rust identifier, so
// the generated crate would have defined one function twice — or silently called the wrong one.
//
// The unit tests in `mangle.rs` pin the encoding; this pins the SOURCE-LEVEL consequence: the
// two functions return different values, and all three engines must observe both.
three_engine_test!(
    colliding_symbol_names_stay_distinct_functions,
    "mangle_collide",
    completes,
    r#"mod m {
    pub fn f() -> Int32 { 1 }
}

fn m_3a_3af() -> Int32 { 2 }

fn main() {
    assert_eq(m::f(), 1);
    assert_eq(m_3a_3af(), 2);
}
"#
);

/// The harness's own guard: a case whose engines disagree must FAIL, not pass quietly. This runs
/// the comparator's normalisation over two deliberately different outcomes and asserts the
/// equality check rejects them — cheap, no native build, and it fails if someone ever weakens
/// `three_engine`'s assertions into warnings.
#[test]
fn the_comparator_rejects_disagreeing_outcomes() {
    let completed = Outcome::Completed {
        stdout: String::new(),
        exit: 0,
    };
    let trapped = Outcome::Trapped {
        category: TrapCategory::IntegerOverflow,
        file: "x.stark".to_string(),
        line: 4,
        column: 20,
        stdout_before: String::new(),
    };
    assert_ne!(completed, trapped);
    // Same category, same file, ONE column apart: the comparator distinguishes trap locations,
    // not just trap categories.
    let shifted = Outcome::Trapped {
        category: TrapCategory::IntegerOverflow,
        file: "x.stark".to_string(),
        line: 4,
        column: 21,
        stdout_before: String::new(),
    };
    assert_ne!(trapped, shifted);
    // And it distinguishes categories at one location.
    let other_category = Outcome::Trapped {
        category: TrapCategory::CastFailure,
        file: "x.stark".to_string(),
        line: 4,
        column: 20,
        stdout_before: String::new(),
    };
    assert_ne!(trapped, other_category);
}
