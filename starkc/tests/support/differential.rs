//! **The three-engine comparator authority** (WP-C6.5 §8, finding C65-F1).
//!
//! This module is the single definition of "the HIR oracle, the MIR interpreter and the native
//! binary agree" for the C6 differential suites. Before WP-C6.5 the same shape was written out
//! independently in 23 test files — each with its own local `run_case` helper, each checking
//! whatever its own work package needed — so the union of those definitions was not a definition.
//! Every rule now lives here and nowhere else.
//!
//! Extracted MECHANICALLY from `tests/three_engine_differential.rs` at WP-C6.5 commit 2, with no
//! behaviour change: the runners, the normalisation, the comparator and its helpers are the code
//! that file carried since WP-C5.2 (CD-053), made `pub` and moved. The observation model is
//! extended to the §39 shape in the FOLLOWING commit, deliberately separated so that a later
//! disagreement can be attributed to the extension rather than to the move.
//!
//! What a case does, per WP-C5.2's exit condition: takes ONE source string, runs that exact source
//! through all three engines, normalises each result into a common [`Outcome`], and requires all
//! three to be equal. The normalisation is the point — an outcome is either normal completion
//! (stdout + exit status) or a trap (category + exact source file/line/column + the stdout emitted
//! before it + the user message where one is normative), so agreement covers completion-vs-trap,
//! exit status, trap category, trap location and trap text, not just "all three exited nonzero".
//!
//! Traps are compared in **normalised** form. Raw stderr byte equality is NOT compared, because
//! the HIR oracle has no canonical stderr format to compare against — its trap text is a set of ad
//! hoc per-call-site strings, which `stark_runtime::trap`'s own doc comment records it does not
//! attempt to match byte for byte. What is compared is what those bytes mean.
//!
//! Every agreement rule lives in [`compare_outcomes`] and nowhere else, so the rules can be — and
//! are — tested directly against disagreeing inputs, rather than only being exercised by cases
//! expected to agree.

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
pub const NATIVE_STDOUT_SUPPORTED: bool = true;

/// One engine's result, normalised to the observable outcome the other two can be compared
/// against. Deliberately NOT engine-shaped: the HIR oracle reports a message and a byte span,
/// MIR reports a category and a `SourceInfo`, and the native binary reports a line of stderr
/// text and a process exit code. All three are projected onto this.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
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
        /// DEV-106 (CD-136): the USER-supplied text of a message-carrying trap — `panic(msg)`.
        /// `None` for a category-only trap (overflow, index, cast, …), where each engine words its
        /// own prose and there is no canonical string to compare.
        ///
        /// Category and location were always compared; the MESSAGE was not, because the harness
        /// REFUSED message-carrying traps outright ("needs string values — outside the C5.2-admitted
        /// surface"). Strings landed in C6.3a, so that refusal was stale and the text is now
        /// comparable across all three engines.
        message: Option<String>,
    },
}

pub struct Front {
    pub hir: starkc::hir::Hir,
    pub file: Arc<SourceFile>,
    pub tables: starkc::typecheck::TypeTables,
}

pub fn front_end(name: &str, source: &str) -> Front {
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
/// `UnwrapNone`, `UnwrapErr` and message-carrying `Panic` are not reachable from the currently
/// admitted surface (`Option`/`Result` are WP-C5.3c, string values WP-C5.3), so they are listed
/// here as explicit "not admitted yet" failures rather than guessed at. `IndexOutOfBounds`
/// joined the admitted set with WP-C5.3a's arrays.
pub fn oracle_category(message: &str) -> TrapCategory {
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
    } else if message.contains("out of bounds") || message.contains("negative index") {
        // Admitted as of WP-C5.3a (arrays). The oracle words the two ends of the range
        // differently ("index out of bounds" vs "negative index") while MIR and the native
        // engine use one category for both -- normalised here rather than by loosening the
        // match, so a genuinely unknown message still fails loudly.
        TrapCategory::IndexOutOfBounds
    } else if message.contains("unwrap") {
        panic!(
            "oracle raised a trap category outside the admitted surface: {message:?} \
             (Option/Result are WP-C5.3c)"
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
pub fn runtime_category(category: TrapCategory) -> stark_runtime::trap::TrapCategory {
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

pub const ALL_CATEGORIES: [TrapCategory; 9] = [
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

pub fn rustc_available() -> bool {
    Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ------------------------------------------------------------------ engine 1 --

pub fn run_hir(name: &str, front: &Front) -> Outcome {
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
            // CD-141: the STATED category wins when the oracle supplies one. `panic(msg)`
            // raises arbitrary USER text, so prose matching cannot classify it — that is the
            // whole reason DEV-106 added `RuntimeError::trap_category`. This harness kept
            // classifying by prose regardless, so the two `panic` cases hit
            // `oracle_category`'s unrecognised-message failure. Prose matching remains the
            // fallback for every trap the interpreter raises without a category.
            let category = err
                .trap_category
                .unwrap_or_else(|| oracle_category(&err.message));
            Outcome::Trapped {
                category,
                file: front.file.name.clone(),
                line: line as u32,
                column: column as u32,
                stdout_before: partial,
                // `panic(msg)` raises the rendered message verbatim as its `RuntimeError`, so for
                // that category the error text IS the user string.
                message: (category == TrapCategory::Panic).then(|| err.message.clone()),
            }
        }
    }
}

// ------------------------------------------------------------------ engine 2 --

pub fn run_mir(name: &str, program: &starkc::mir::MirProgram) -> Outcome {
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
                message,
            }
        }
        Err(MirFailure {
            error: MirRunError::Internal(message),
            ..
        }) => panic!("{name}: MIR internal error: {message}"),
    }
}

// ------------------------------------------------------------------ engine 3 --

pub fn run_native(name: &str, tag: &str, program: &starkc::mir::MirProgram) -> Outcome {
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
        target_contract: "stark-64-v1".to_string(),
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
            let (category, file, line, column, message) = parse_native_trap(name, &stderr);
            Outcome::Trapped {
                category,
                file,
                line,
                column,
                stdout_before: stdout,
                message,
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
pub fn parse_native_trap(
    name: &str,
    stderr: &str,
) -> (TrapCategory, String, u32, u32, Option<String>) {
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
    // DEV-106: `trap::abort_with_message` prints the user's text on its own line AFTER the `-->`
    // location, indented. A category-only trap prints no such line, which is exactly the `None`
    // case — so the shape of the stderr distinguishes the two without a second parse mode.
    let user_message = stderr
        .lines()
        .skip_while(|l| !l.trim().starts_with("--> "))
        .nth(1)
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty());
    (category, file, line, column, user_message)
}

// ----------------------------------------------------------------- the check --

/// **The comparator.** Every agreement rule this harness enforces lives here and nowhere else,
/// as a pure function of three already-normalised outcomes — deliberately returning `Err(reason)`
/// rather than asserting, so the rules themselves are testable against deliberately disagreeing
/// inputs (`the_comparator_rejects_disagreeing_outcomes`) instead of only being exercised by
/// cases that are expected to agree. A comparator whose only coverage is passing cases is a
/// comparator nobody has watched fail.
///
/// `three_engine` turns an `Err` into the test failure; it adds no rule of its own.
pub fn compare_outcomes(
    name: &str,
    hir: &Outcome,
    mir: &Outcome,
    native: &Outcome,
) -> Result<(), String> {
    if !NATIVE_STDOUT_SUPPORTED {
        // Enforced, not assumed: if a case ever starts printing, this fires rather than letting
        // the native side's necessarily-empty stdout quietly disagree with the other two.
        let printed = match hir {
            Outcome::Completed { stdout, .. } => stdout,
            Outcome::Trapped { stdout_before, .. } => stdout_before,
        };
        if !printed.is_empty() {
            return Err(format!(
                "{name}: case produces stdout ({printed:?}), but the native backend cannot emit \
                 output until WP-C5.3 — every harness case must observe values through in-program \
                 assertions while NATIVE_STDOUT_SUPPORTED is false"
            ));
        }
    }

    if hir != mir {
        return Err(format!(
            "{name}: HIR/MIR DISAGREEMENT\n--- HIR oracle ---\n{hir:#?}\n--- MIR ---\n{mir:#?}"
        ));
    }
    if mir != native {
        return Err(format!(
            "{name}: MIR/NATIVE DISAGREEMENT\n--- MIR ---\n{mir:#?}\n--- native ---\n{native:#?}"
        ));
    }
    Ok(())
}

/// Run one source through all three engines and require identical normalised outcomes.
///
/// `tag` names the scratch build directory only; `name` becomes the STARK source file name, so
/// it is what every engine reports as the trap location and is therefore itself compared.
pub fn three_engine(tag: &str, source: &str) -> Outcome {
    let name = format!("three_engine_{tag}.stark");
    let front = front_end(&name, source);
    let program = match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
    };

    let hir = run_hir(&name, &front);
    let mir = run_mir(&name, &program);
    let native = run_native(&name, tag, &program);

    if let Err(disagreement) = compare_outcomes(&name, &hir, &mir, &native) {
        panic!("{disagreement}");
    }
    hir
}

/// All three completed normally with exit 0 — i.e. every in-program assertion held in every
/// engine. Meaningful only because `a_false_assertion_traps_in_all_three_engines` proves a
/// FALSE assertion is observable; see `three_engine_differential.rs`.
pub fn agree_completing(tag: &str, source: &str) {
    let outcome = three_engine(tag, source);
    assert!(
        matches!(outcome, Outcome::Completed { exit: 0, .. }),
        "{tag}: expected normal completion, got {outcome:#?}"
    );
}

/// All three trapped, with the same category at the same source line — and that line is stated
/// here independently, so a case whose three engines agreed on the WRONG location still fails.
pub fn agree_trapping(tag: &str, source: &str, expected: TrapCategory, expected_line: u32) {
    let outcome = three_engine(tag, source);
    match outcome {
        Outcome::Trapped { category, line, .. } => {
            assert_eq!(category, expected, "{tag}: trap category");
            assert_eq!(line, expected_line, "{tag}: trap line");
        }
        other => panic!("{tag}: expected a trap, got {other:#?}"),
    }
}

/// DEV-106 (CD-136): all three trapped with the same category, line AND the same user-supplied
/// MESSAGE. `three_engine` already requires the three outcomes to be identical, so the message is
/// compared engine-to-engine by construction; `expected_message` additionally pins it to the text
/// the source actually wrote, so three engines agreeing on the WRONG string still fails.
pub fn agree_trapping_with_message(
    tag: &str,
    source: &str,
    expected: TrapCategory,
    expected_line: u32,
    expected_message: &str,
) {
    let outcome = three_engine(tag, source);
    match outcome {
        Outcome::Trapped {
            category,
            line,
            ref message,
            ..
        } => {
            assert_eq!(category, expected, "{tag}: trap category");
            assert_eq!(line, expected_line, "{tag}: trap line");
            let message = message.as_deref().unwrap_or_else(|| {
                panic!("{tag}: expected a message-carrying trap, got {outcome:#?}")
            });
            assert_eq!(message, expected_message, "{tag}: trap message");
        }
        other => panic!("{tag}: expected a trap, got {other:#?}"),
    }
}

/// Declares one three-engine case as a `#[test]`.
///
/// The expansion refers to this module by absolute path (`$crate::support::differential::…`), so a
/// consuming binary needs only `mod support;` at its root — it does not have to import the runners
/// the macro happens to call.
#[macro_export]
macro_rules! three_engine_test {
    ($name:ident, $tag:literal, completes, $source:literal) => {
        #[test]
        fn $name() {
            if !$crate::support::differential::rustc_available() {
                eprintln!("SKIP: no rustc in this environment.");
                return;
            }
            $crate::support::differential::agree_completing($tag, $source);
        }
    };
    ($name:ident, $tag:literal, traps($category:expr, $line:literal), $source:literal) => {
        #[test]
        fn $name() {
            if !$crate::support::differential::rustc_available() {
                eprintln!("SKIP: no rustc in this environment.");
                return;
            }
            $crate::support::differential::agree_trapping($tag, $source, $category, $line);
        }
    };
}
