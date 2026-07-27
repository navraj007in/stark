//! **The three-engine comparator authority** (WP-C6.5 §8, finding C65-F1).
//!
//! This module is the single definition of "the HIR oracle, the MIR interpreter and the native
//! binary agree" for the C6 differential suites. Before WP-C6.5 the same shape was written out
//! independently in 23 test files — each with its own local `run_case` helper, each checking
//! whatever its own work package needed — so the union of those definitions was not a definition.
//! Every rule now lives here and nowhere else.
//!
//! Commit 2 extracted the runners and the comparator mechanically. **Commit 3 (this) extends the
//! observation to the §39 shape**, which is what the required claim actually needs: the engines
//! produce the same *normative observations*, not merely the same stdout and exit code.
//!
//! ```text
//! Completed { stdout_bytes, stderr_bytes, exit_status, returned_observation, drop_log }
//! Trapped   { category, source_file, line, column, message_class, stdout_before_trap,
//!             stderr_observation, exit_status, drop_log_before_trap }
//! ```
//!
//! Four rules govern how those fields are produced, because each is a place where a comparator can
//! quietly stop comparing:
//!
//! 1. **Bytes, not host strings** (§8.4). Native output is kept as `Vec<u8>`; the interpreters'
//!    `String` channels convert without platform line-ending translation. Nothing is lossily
//!    decoded for equality — only the reserved protocol frames, which are ASCII by construction, are
//!    read as text.
//! 2. **Trap stderr is normalised, not byte-matched** (§8.5). The native engine has real stderr; the
//!    interpreters have none. So the comparator *constructs* the normative
//!    [`TrapStderrObservation`] for them from the category and location, taking the category text
//!    from `stark_runtime::trap`'s own table — the same source the native ABI prints from — and
//!    *parses* it out of the native engine's stderr. An unrecognised native rendering is a hard
//!    failure, never a silent pass.
//! 3. **Drop events come from the program, not from the host** (§8.8). A drop-observing case emits a
//!    reserved frame from its own `Drop` impl; the harness extracts those frames from stdout, in
//!    order, and removes them before comparing stdout. Inferring Drop order from generated Rust
//!    destructors or host traces would make the native engine's Drop schedule unfalsifiable.
//! 4. **Returned values go through a framed probe** (§8.7), so a case can observe a function's
//!    result without that result being indistinguishable from ordinary output.
//!
//! Every agreement rule lives in [`compare_observations`] and nowhere else, so the rules can be —
//! and are — tested directly against deliberately disagreeing inputs rather than only being
//! exercised by cases expected to agree. A comparator whose only coverage is passing cases is a
//! comparator nobody has watched fail.

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
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

/// TRAP-ABORT-001: a language trap terminates with status 101. The interpreters are not processes
/// and have no status of their own, so this is the **normative constant** they report — not a
/// default standing in for an unknown. It is compared against the native process's real exit code,
/// so a backend that aborted with any other status fails here rather than being accommodated.
pub const TRAP_EXIT_STATUS: i32 = 101;

/// Whether the native backend can emit observable stdout. `true` since WP-C5.3; retained because
/// [`compare_observations`] still enforces the output-free precondition while it is `false`, and
/// that guard is the reason full observation equality was an honest comparison during C5.2 rather
/// than one with a quietly excluded dimension.
pub const NATIVE_STDOUT_SUPPORTED: bool = true;

// ------------------------------------------------------------------ §39 shape --

/// One engine's result, normalised to the observable outcome the other two are compared against.
/// Deliberately NOT engine-shaped: the HIR oracle reports a message and a byte span, MIR reports a
/// category and a `SourceInfo`, and the native binary reports a line of stderr text and a process
/// exit code. All three are projected onto this.
#[derive(Debug, PartialEq, Eq)]
pub enum Observation {
    Completed(CompletionObservation),
    Trapped(TrapObservation),
}

#[derive(Debug, PartialEq, Eq)]
pub struct CompletionObservation {
    /// Normative stdout, with every protocol frame removed (§8.7/§8.8).
    pub stdout_bytes: Vec<u8>,
    /// PROC-EXIT-001's `Err(message)` write is the only stderr a completing Core program produces
    /// today; `eprint`/`eprintln` reach no engine's captured channel (DEV-111's recorded gap), so
    /// this compares empty-to-empty for every case that does not return `Err` from `main`.
    pub stderr_bytes: Vec<u8>,
    pub exit_status: i32,
    /// `None` for ordinary program-level cases; `Some` only for a framed probe (§8.7).
    pub returned_observation: Option<CanonicalReturnedValue>,
    pub drop_log: Vec<DropEvent>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TrapObservation {
    pub category: TrapCategory,
    pub source_file: String,
    pub line: u32,
    pub column: u32,
    pub message_class: TrapMessageClass,
    /// C4.5e-0: output emitted before the trap is observable, so two programs printing different
    /// prefixes before the same trap are different outcomes.
    pub stdout_before_trap: Vec<u8>,
    pub stderr_observation: TrapStderrObservation,
    pub exit_status: i32,
    /// §8.8: only the events that happened *before* the trap. TRAP-ABORT-001 aborts without
    /// running destructors, so this field is what makes "no Drop after a trap" falsifiable rather
    /// than assumed.
    pub drop_log_before_trap: Vec<DropEvent>,
}

/// §8.6. Which part of a trap's text is normative, and therefore comparable.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TrapMessageClass {
    /// A compiler-generated trap: each engine words its own prose, so only the category and
    /// provenance are compared.
    CategoryOnly,
    /// `panic(msg)`: the user's text is normative and compared byte for byte.
    UserMessageExact(String),
    /// A runtime-compatibility mismatch — a pre-user-code build/runtime observation, never a
    /// program trap. Constructed only so the normalizer can recognise it and fail loudly instead of
    /// classifying it as a language outcome.
    RuntimeCompatibility,
}

/// The normative content of a trap's stderr: parsed from the native engine, constructed for the
/// interpreters (§8.5). Raw engine diagnostics are kept out of equality entirely — they are
/// reported in failure messages for debugging, not compared.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TrapStderrObservation {
    pub category_text: String,
    pub user_message: Option<String>,
    pub source_file: String,
    pub line: u32,
    pub column: u32,
}

/// One user-`Drop` execution, as the program itself reported it. `sequence` is assigned by the
/// harness from the order the frames appear, so a reordered Drop schedule changes the log even when
/// the same identities appear.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct DropEvent {
    pub sequence: u32,
    pub identity: String,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct CanonicalReturnedValue {
    pub type_tag: String,
    pub rendered: Vec<u8>,
}

// --------------------------------------------------------------- protocols --

/// §8.8. A `Drop` impl emits `println("@@stark-drop:<identity>@@")`; the harness turns each frame
/// into a [`DropEvent`] and removes the line from normative stdout.
pub const DROP_FRAME_PREFIX: &str = "@@stark-drop:";
/// §8.7. The generated probe wrapper emits `@@stark-ret:<type_tag>:<rendered>@@`.
///
/// `@@` rather than the `##` the plan sketches: a case source is a Rust raw string in the test
/// file, and `"##` terminates `r#"…"#`, so every drop-observing case would have had to remember to
/// write `r###"`. The sentinel is arbitrary; the friction was not.
pub const RET_FRAME_PREFIX: &str = "@@stark-ret:";
const FRAME_SUFFIX: &str = "@@";

/// What a scan of one engine's raw stdout yields.
struct ProtocolScan {
    stdout: Vec<u8>,
    drop_log: Vec<DropEvent>,
    returned: Option<CanonicalReturnedValue>,
}

/// Splits raw stdout into normative stdout plus protocol events (§8.7/§8.8).
///
/// Line-oriented, and strict about it: a frame must occupy a whole line. A case that emits a frame
/// after an unterminated `print` produces a line that *contains* the reserved prefix without
/// starting at it, and that is a hard failure rather than being passed through as ordinary output —
/// otherwise a Drop event could silently vanish into stdout and the drop_log would under-report.
/// The prefixes are reserved: a program printing one for any other reason fails here too.
///
/// Only frame lines are decoded as text. Everything else is copied through as bytes, so a case
/// emitting non-UTF-8 output is compared byte for byte (§8.4).
fn scan_protocol(raw: &[u8], engine: &str) -> Result<ProtocolScan, String> {
    let mut stdout: Vec<u8> = Vec::new();
    let mut drop_log: Vec<DropEvent> = Vec::new();
    let mut returned: Option<CanonicalReturnedValue> = None;
    let mut seen_identities: Vec<String> = Vec::new();

    // `split_inclusive` keeps each line's terminator, so a trailing unterminated chunk (an
    // unflushed `print`) round-trips exactly rather than gaining a newline.
    for line in raw.split_inclusive(|&b| b == b'\n') {
        let text = std::str::from_utf8(line).ok();
        let trimmed = text.map(|t| t.trim_end_matches(['\n', '\r'])).unwrap_or("");

        let is_frame =
            trimmed.starts_with(DROP_FRAME_PREFIX) || trimmed.starts_with(RET_FRAME_PREFIX);
        if !is_frame {
            // A reserved prefix anywhere else in the line means the frame was not emitted on a line
            // of its own.
            if let Some(t) = text {
                for prefix in [DROP_FRAME_PREFIX, RET_FRAME_PREFIX] {
                    if t.contains(prefix) {
                        return Err(format!(
                            "{engine}: malformed protocol frame — {prefix:?} appears mid-line in \
                             {t:?}. A frame must occupy a whole line: emit it with `println`, and \
                             make sure the preceding output ended with a newline"
                        ));
                    }
                }
            }
            stdout.extend_from_slice(line);
            continue;
        }

        let body = trimmed.strip_suffix(FRAME_SUFFIX).ok_or_else(|| {
            format!("{engine}: protocol frame is not terminated by `{FRAME_SUFFIX}`: {trimmed:?}")
        })?;

        if let Some(identity) = body.strip_prefix(DROP_FRAME_PREFIX) {
            if identity.is_empty() {
                return Err(format!(
                    "{engine}: Drop frame carries no identity: {trimmed:?}"
                ));
            }
            if seen_identities.iter().any(|seen| seen == identity) {
                // §8.8 "malformed or duplicate sequence framing fails". A repeated identity makes
                // the log ambiguous exactly where it matters most — a double Drop and two
                // same-named values would be indistinguishable — so a case must give each
                // droppable value a distinct identity.
                return Err(format!(
                    "{engine}: Drop identity {identity:?} was emitted twice. Identities must be \
                     unique within a case, or a double-Drop cannot be told from two values sharing \
                     a name"
                ));
            }
            seen_identities.push(identity.to_string());
            drop_log.push(DropEvent {
                sequence: drop_log.len() as u32 + 1,
                identity: identity.to_string(),
            });
            continue;
        }

        let payload = body
            .strip_prefix(RET_FRAME_PREFIX)
            .expect("checked by is_frame");
        let (type_tag, rendered) = payload.split_once(':').ok_or_else(|| {
            format!("{engine}: return frame has no `<type_tag>:<rendered>` split: {trimmed:?}")
        })?;
        if returned.is_some() {
            return Err(format!("{engine}: more than one return frame in one run"));
        }
        returned = Some(CanonicalReturnedValue {
            type_tag: type_tag.to_string(),
            rendered: rendered.as_bytes().to_vec(),
        });
    }

    Ok(ProtocolScan {
        stdout,
        drop_log,
        returned,
    })
}

/// §8.7 steps 2–3: the wrapper that calls a case's zero-argument `probe` and frames its result.
/// Appended AFTER the case source so every user line number is unchanged — a trap inside the case
/// must not be attributed to generated code.
fn probe_wrapper(type_tag: &str) -> String {
    format!(
        "fn main() {{\n    print(\"{RET_FRAME_PREFIX}{type_tag}:\");\n    print(probe());\n    \
         println(\"{FRAME_SUFFIX}\");\n}}\n"
    )
}

// ----------------------------------------------------------------- front end --

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
/// `UnwrapNone`, `UnwrapErr`, `InvalidExitStatus` and message-carrying `Panic` traps state their
/// category outright (`RuntimeError::with_category`), so they never reach this prose path. The
/// `unwrap` arm used to REFUSE outright — "Option/Result are WP-C5.3c" — a refusal that outlived its
/// reason by three work packages and blocked the corpus's `UnwrapNone`/`UnwrapErr` cases (R-01).
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
    } else {
        panic!(
            "unrecognised oracle trap message {message:?} — normalise it here rather than \
             letting it default to a category the other engines happen to report"
        )
    }
}

/// `starkc::mir::TrapCategory` → the runtime's own copy, so the native stderr text this harness
/// matches against — and the text it CONSTRUCTS for the interpreters (§8.5) — is the runtime's
/// single source of truth (`stark-runtime/src/trap.rs`) rather than a second table in a test file
/// that could drift from it. The match is exhaustive on purpose: a new category fails to compile
/// here until it is mapped, which is what makes the normalizer total over `TrapCategory`.
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
        TrapCategory::InvalidExitStatus => Rt::InvalidExitStatus,
    }
}

pub const ALL_CATEGORIES: [TrapCategory; 10] = [
    TrapCategory::IntegerOverflow,
    TrapCategory::DivideByZero,
    TrapCategory::IndexOutOfBounds,
    TrapCategory::CastFailure,
    TrapCategory::Panic,
    TrapCategory::UnwrapNone,
    TrapCategory::UnwrapErr,
    TrapCategory::AssertFailure,
    TrapCategory::InvalidShift,
    TrapCategory::InvalidExitStatus,
];

pub fn rustc_available() -> bool {
    Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Builds the interpreters' [`TrapStderrObservation`]: they have no stderr stream, so the normative
/// rendering is CONSTRUCTED from the same runtime table the native ABI prints from (§8.5).
fn constructed_trap_stderr(
    category: TrapCategory,
    file: &str,
    line: u32,
    column: u32,
    user_message: Option<String>,
) -> TrapStderrObservation {
    TrapStderrObservation {
        category_text: runtime_category(category).message().to_string(),
        user_message,
        source_file: file.to_string(),
        line,
        column,
    }
}

fn message_class(message: Option<&String>) -> TrapMessageClass {
    match message {
        Some(text) => TrapMessageClass::UserMessageExact(text.clone()),
        None => TrapMessageClass::CategoryOnly,
    }
}

/// A package or workspace case, compiled from a package graph rather than from one source string
/// (§15.1).
///
/// **Always compiled from a COPY.** `PackageGraph` resolution writes `stark.lock` into the root
/// package, so pointing it at the checked-in corpus would both dirty the tree and break
/// `corpus.lock`'s hashes — and, running under `cargo test`'s thread pool, two cases sharing a root
/// would race on the same lock file (fatal on Windows). The copy is the caller's; this function
/// takes the path it should use.
pub fn front_end_package(root_package: &Path) -> (Front, starkc::mir::MirProgram) {
    use starkc::options::LanguageOptions;
    use starkc::package::{find_package_root, PackageGraph};
    use starkc::parser::parse_package_graph;

    let manifest = find_package_root(root_package)
        .unwrap_or_else(|e| panic!("{}: no starkpkg.json: {e:?}", root_package.display()));
    let graph = PackageGraph::load_from_root(&manifest)
        .unwrap_or_else(|e| panic!("{}: package graph: {e:?}", root_package.display()));
    let (ast, parse_diags) = parse_package_graph(&graph, LanguageOptions::CORE);
    assert!(
        parse_diags.is_empty(),
        "{}: parse: {parse_diags:?}",
        root_package.display()
    );
    // The root file is named by its REAL path, as `parse_package_graph` named it: entry-point
    // discovery matches the root file against the parsed graph, and a logical name produces
    // "program without a `main` function". §15.2's "trap source names remain logical source paths"
    // is therefore a property of the COMPILER here, not something this harness can impose — see
    // `c6_package.rs`, which measures it rather than assuming either answer.
    let entry = root_package.join("src/main.stark");
    let entry_text =
        std::fs::read_to_string(&entry).unwrap_or_else(|e| panic!("{}: {e}", entry.display()));
    let root_file = Arc::new(SourceFile::new(
        entry.to_string_lossy().into_owned(),
        entry_text,
    ));
    let (hir, resolve_diags) = resolve(&ast, root_file.clone());
    assert!(
        resolve_diags.is_empty(),
        "{}: resolve: {resolve_diags:?}",
        root_package.display()
    );
    let checked = typecheck::analyze(&hir, root_file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "{}: typecheck: {errors:?}",
        root_package.display()
    );
    let front = Front {
        hir,
        file: root_file,
        tables: checked.tables,
    };
    let program = match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(e) => panic!(
            "{}: lowering failed: {} @ {:?}",
            root_package.display(),
            e.what,
            e.span
        ),
    };
    (front, program)
}

/// Copies a corpus package case into a scratch directory and returns the staged ROOT package.
///
/// The whole CASE DIRECTORY is copied, not just the root package: a workspace's root depends on its
/// siblings by relative path (`"model": { "path": "../model" }`), so staging `app/` alone leaves the
/// dependency dangling. The case directory is `cases/<kind>/<case>`; `package_root` names the root
/// package inside it, which for a workspace is a subdirectory and for a single package is the case
/// directory itself.
pub fn stage_package(case_id: &str, corpus_root: &Path, package_root: &str) -> PathBuf {
    let parts: Vec<&str> = package_root.split('/').collect();
    assert!(
        parts.len() >= 3,
        "package_root `{package_root}` is not under cases/<kind>/<case>"
    );
    let case_dir = parts[..3].join("/");
    let remainder = parts[3..].join("/");
    let (scratch, _) = stage_dir(case_id, &corpus_root.join(&case_dir));
    if remainder.is_empty() {
        scratch
    } else {
        scratch.join(remainder)
    }
}

/// Copies a directory tree into a scratch location. Returns the copy's root twice for callers that
/// want to keep the handle and the path separately.
pub fn stage_dir(case_id: &str, corpus_case_root: &Path) -> (PathBuf, PathBuf) {
    fn copy_tree(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).expect("scratch package dir");
        for entry in std::fs::read_dir(from).expect("read package dir") {
            let path = entry.expect("entry").path();
            let name = path.file_name().expect("name").to_owned();
            if path.is_dir() {
                copy_tree(&path, &to.join(name));
            } else {
                std::fs::copy(&path, to.join(name)).expect("copy package file");
            }
        }
    }
    let scratch = std::env::temp_dir().join(format!(
        "stark_c6pkg_{case_id}_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&scratch);
    copy_tree(corpus_case_root, &scratch);
    (scratch.clone(), scratch)
}

// ------------------------------------------------------------------ engine 1 --

pub fn run_hir(name: &str, front: &Front) -> Observation {
    match interp::run_with_partial_output(&front.hir, front.file.clone(), &front.tables) {
        Ok(exec) => {
            let scan = scan_protocol(exec.output.as_bytes(), "HIR oracle")
                .unwrap_or_else(|reason| panic!("{name}: {reason}"));
            Observation::Completed(CompletionObservation {
                stdout_bytes: scan.stdout,
                stderr_bytes: exec.stderr.into_bytes(),
                exit_status: i32::from(exec.status),
                returned_observation: scan.returned,
                drop_log: scan.drop_log,
            })
        }
        Err((err, partial)) => {
            assert!(
                err.is_trap,
                "{name}: oracle failed without trapping ({}) — an entrypoint-selection failure \
                 is a compiler error, not a language outcome the other engines can match",
                err.message
            );
            // DEV-113-B: the trap's OWN file when the oracle supplies one — for a multi-file or
            // package program the raising file is not the entry file, and using the entry file made
            // the oracle disagree with MIR about which file trapped.
            let raised_in = err.file.clone().unwrap_or_else(|| front.file.clone());
            let (line, column) = raised_in.line_col(err.span.lo);
            // CD-141: the STATED category wins when the oracle supplies one. `panic(msg)`
            // raises arbitrary USER text, so prose matching cannot classify it — that is the
            // whole reason DEV-106 added `RuntimeError::trap_category`. Prose matching remains the
            // fallback for every trap the interpreter raises without a category.
            let category = err
                .trap_category
                .unwrap_or_else(|| oracle_category(&err.message));
            // `panic(msg)` raises the rendered message verbatim as its `RuntimeError`, so for that
            // category the error text IS the user string.
            let message = (category == TrapCategory::Panic).then(|| err.message.clone());
            let scan = scan_protocol(partial.as_bytes(), "HIR oracle")
                .unwrap_or_else(|reason| panic!("{name}: {reason}"));
            let (line, column) = (line as u32, column as u32);
            Observation::Trapped(TrapObservation {
                category,
                source_file: raised_in.name.clone(),
                line,
                column,
                message_class: message_class(message.as_ref()),
                stdout_before_trap: scan.stdout,
                stderr_observation: constructed_trap_stderr(
                    category,
                    &raised_in.name,
                    line,
                    column,
                    message,
                ),
                exit_status: TRAP_EXIT_STATUS,
                drop_log_before_trap: scan.drop_log,
            })
        }
    }
}

// ------------------------------------------------------------------ engine 2 --

pub fn run_mir(name: &str, program: &starkc::mir::MirProgram) -> Observation {
    let verified = match verify_program(program) {
        Ok(v) => v,
        Err(errors) => panic!("{name}: verifier rejected lowered MIR:\n{errors:#?}"),
    };
    match run_program(verified) {
        Ok(exec) => {
            let scan = scan_protocol(exec.output.as_bytes(), "MIR")
                .unwrap_or_else(|reason| panic!("{name}: {reason}"));
            Observation::Completed(CompletionObservation {
                stdout_bytes: scan.stdout,
                stderr_bytes: exec.stderr.into_bytes(),
                exit_status: i32::from(exec.status),
                returned_observation: scan.returned,
                drop_log: scan.drop_log,
            })
        }
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
            let (line, column) = (line as u32, column as u32);
            let scan =
                scan_protocol(output.as_bytes(), "MIR").unwrap_or_else(|r| panic!("{name}: {r}"));
            Observation::Trapped(TrapObservation {
                category,
                source_file: file.name.clone(),
                line,
                column,
                message_class: message_class(message.as_ref()),
                stdout_before_trap: scan.stdout,
                stderr_observation: constructed_trap_stderr(
                    category, &file.name, line, column, message,
                ),
                exit_status: TRAP_EXIT_STATUS,
                drop_log_before_trap: scan.drop_log,
            })
        }
        Err(MirFailure {
            error: MirRunError::Internal(message),
            ..
        }) => panic!("{name}: MIR internal error: {message}"),
    }
}

// ------------------------------------------------------------------ engine 3 --

pub fn run_native(name: &str, tag: &str, program: &starkc::mir::MirProgram) -> Observation {
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

    // §8.4: the native engine's bytes stay bytes. Only the trap rendering is read as text, and
    // only after the process is known to have trapped.
    let scan =
        scan_protocol(&run.stdout, "native").unwrap_or_else(|reason| panic!("{name}: {reason}"));
    match run.status.code() {
        Some(TRAP_EXIT_STATUS) => {
            let stderr = String::from_utf8_lossy(&run.stderr).into_owned();
            let observed = parse_native_trap(name, &stderr);
            Observation::Trapped(TrapObservation {
                category: observed.category,
                source_file: observed.stderr_observation.source_file.clone(),
                line: observed.stderr_observation.line,
                column: observed.stderr_observation.column,
                message_class: observed.message_class,
                stdout_before_trap: scan.stdout,
                stderr_observation: observed.stderr_observation,
                exit_status: TRAP_EXIT_STATUS,
                drop_log_before_trap: scan.drop_log,
            })
        }
        Some(code) => Observation::Completed(CompletionObservation {
            stdout_bytes: scan.stdout,
            stderr_bytes: run.stderr,
            exit_status: code,
            returned_observation: scan.returned,
            drop_log: scan.drop_log,
        }),
        None => panic!(
            "{name}: native run terminated by a signal; stderr: {}",
            String::from_utf8_lossy(&run.stderr)
        ),
    }
}

/// What the native trap ABI's stderr says, normalised (§8.5).
pub struct NativeTrap {
    pub category: TrapCategory,
    pub message_class: TrapMessageClass,
    pub stderr_observation: TrapStderrObservation,
}

/// Reads the native trap ABI's stderr back into the normalised form. The format is fixed by
/// `stark_runtime::trap::abort`:
///
/// ```text
/// error: runtime trap: <category message>
///   --> <file>:<line>:<column>
/// ```
///
/// Cargo text and host backtraces are ignored (§8.5); an unrecognised rendering fails.
pub fn parse_native_trap(name: &str, stderr: &str) -> NativeTrap {
    if stderr.contains("stark-runtime version mismatch") {
        // §8.6: a pre-user-code runtime-compatibility failure is not a language trap. Surfaced as a
        // harness failure rather than compared, so it can never be mistaken for one.
        panic!(
            "{name}: runtime-compatibility mismatch, which is a build/runtime observation and not \
             a program trap ({:?}):\n{stderr}",
            TrapMessageClass::RuntimeCompatibility
        );
    }
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
    // location, indented. A category-only trap prints no such line, which is exactly the
    // `CategoryOnly` case — so the shape of the stderr distinguishes the two without a second
    // parse mode.
    let user_message = stderr
        .lines()
        .skip_while(|l| !l.trim().starts_with("--> "))
        .nth(1)
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty());
    NativeTrap {
        category,
        message_class: message_class(user_message.as_ref()),
        stderr_observation: TrapStderrObservation {
            category_text: message.to_string(),
            user_message,
            source_file: file,
            line,
            column,
        },
    }
}

// ----------------------------------------------------------------- the check --

/// A canonical, explicit rendering of an observation — the input to the evidence `observation_hash`
/// (§21.1).
///
/// Written out field by field rather than derived from `Debug`: `Debug` output is stable in practice
/// but is not a contract, and an evidence hash that changed with a Rust release would invalidate
/// every stored record for no semantic reason. Bytes are rendered as hex so a non-UTF-8 observation
/// hashes exactly as it was observed.
pub fn canonical_form(observation: &Observation) -> String {
    fn hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
        out
    }
    fn drops(log: &[DropEvent]) -> String {
        log.iter()
            .map(|event| format!("{}:{}", event.sequence, event.identity))
            .collect::<Vec<_>>()
            .join(",")
    }
    match observation {
        Observation::Completed(done) => format!(
            "completed\nstdout={}\nstderr={}\nexit={}\nreturned={}\ndrops={}\n",
            hex(&done.stdout_bytes),
            hex(&done.stderr_bytes),
            done.exit_status,
            match &done.returned_observation {
                Some(value) => format!("{}:{}", value.type_tag, hex(&value.rendered)),
                None => "none".to_string(),
            },
            drops(&done.drop_log),
        ),
        Observation::Trapped(trap) => {
            format!(
            "trapped\ncategory={:?}\nfile={}\nline={}\ncolumn={}\nmessage={}\nstdout_before={}\n\
             stderr_category={}\nstderr_message={}\nstderr_location={}:{}:{}\nexit={}\ndrops={}\n",
            trap.category,
            trap.source_file,
            trap.line,
            trap.column,
            match &trap.message_class {
                TrapMessageClass::CategoryOnly => "category-only".to_string(),
                TrapMessageClass::UserMessageExact(text) => format!("exact:{text}"),
                TrapMessageClass::RuntimeCompatibility => "runtime-compatibility".to_string(),
            },
            hex(&trap.stdout_before_trap),
            trap.stderr_observation.category_text,
            trap.stderr_observation.user_message.as_deref().unwrap_or("none"),
            trap.stderr_observation.source_file,
            trap.stderr_observation.line,
            trap.stderr_observation.column,
            trap.exit_status,
            drops(&trap.drop_log_before_trap),
        )
        }
    }
}

/// The first field on which two observations differ, named. Field-by-field rather than a derived
/// `!=` so a failure says WHICH normative dimension disagreed — with nine fields on a trap, "these
/// two structs differ" is not a useful answer.
pub fn first_difference(a: &Observation, b: &Observation) -> Option<&'static str> {
    match (a, b) {
        (Observation::Completed(x), Observation::Completed(y)) => {
            if x.stdout_bytes != y.stdout_bytes {
                Some("stdout_bytes")
            } else if x.stderr_bytes != y.stderr_bytes {
                Some("stderr_bytes")
            } else if x.exit_status != y.exit_status {
                Some("exit_status")
            } else if x.returned_observation != y.returned_observation {
                Some("returned_observation")
            } else if x.drop_log != y.drop_log {
                Some("drop_log")
            } else {
                None
            }
        }
        (Observation::Trapped(x), Observation::Trapped(y)) => {
            if x.category != y.category {
                Some("trap category")
            } else if x.source_file != y.source_file {
                Some("trap source_file")
            } else if x.line != y.line {
                Some("trap line")
            } else if x.column != y.column {
                Some("trap column")
            } else if x.message_class != y.message_class {
                Some("trap message_class")
            } else if x.stdout_before_trap != y.stdout_before_trap {
                Some("stdout_before_trap")
            } else if x.stderr_observation != y.stderr_observation {
                Some("stderr_observation")
            } else if x.exit_status != y.exit_status {
                Some("trap exit_status")
            } else if x.drop_log_before_trap != y.drop_log_before_trap {
                Some("drop_log_before_trap")
            } else {
                None
            }
        }
        _ => Some("completion versus trap"),
    }
}

/// **The comparator.** Every agreement rule this harness enforces lives here and nowhere else, as a
/// pure function of three already-normalised observations — deliberately returning `Err(reason)`
/// rather than asserting, so the rules themselves are testable against deliberately disagreeing
/// inputs instead of only being exercised by cases that are expected to agree.
///
/// `three_engine` turns an `Err` into the test failure; it adds no rule of its own.
pub fn compare_observations(
    name: &str,
    hir: &Observation,
    mir: &Observation,
    native: &Observation,
) -> Result<(), String> {
    if !NATIVE_STDOUT_SUPPORTED {
        // Enforced, not assumed: if a case ever starts printing, this fires rather than letting
        // the native side's necessarily-empty stdout quietly disagree with the other two.
        let printed = match hir {
            Observation::Completed(c) => &c.stdout_bytes,
            Observation::Trapped(t) => &t.stdout_before_trap,
        };
        if !printed.is_empty() {
            return Err(format!(
                "{name}: case produces stdout ({:?}), but the native backend cannot emit output \
                 while NATIVE_STDOUT_SUPPORTED is false — every harness case must observe values \
                 through in-program assertions until it flips",
                String::from_utf8_lossy(printed)
            ));
        }
    }

    if let Some(field) = first_difference(hir, mir) {
        return Err(format!(
            "{name}: HIR/MIR DISAGREEMENT on {field}\n--- HIR oracle ---\n{hir:#?}\n--- MIR ---\n{mir:#?}"
        ));
    }
    if let Some(field) = first_difference(mir, native) {
        return Err(format!(
            "{name}: MIR/NATIVE DISAGREEMENT on {field}\n--- MIR ---\n{mir:#?}\n--- native ---\n{native:#?}"
        ));
    }
    Ok(())
}

/// Run one source through all three engines and require identical normalised observations.
///
/// `tag` names the scratch build directory only; `name` becomes the STARK source file name, so
/// it is what every engine reports as the trap location and is therefore itself compared.
pub fn three_engine(tag: &str, source: &str) -> Observation {
    let name = format!("three_engine_{tag}.stark");
    let front = front_end(&name, source);
    let program = match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
    };

    let hir = run_hir(&name, &front);
    let mir = run_mir(&name, &program);
    let native = run_native(&name, tag, &program);

    if let Err(disagreement) = compare_observations(&name, &hir, &mir, &native) {
        panic!("{disagreement}");
    }
    hir
}

/// Two engines only: the HIR oracle and MIR, compared by the same comparator.
///
/// Not a weaker standard applied for convenience — it is for cases the native backend REFUSES to
/// build, where running two engines is the whole truth available and the third is a recorded
/// blocker (DEV-111's entry-contract cases are the current population). The corpus manifest states
/// which engines a case requires, so a case cannot quietly opt out of native: dropping
/// `native-debug` from `required_engines` is a visible edit that needs a `deviation` beside it.
pub fn two_engine(tag: &str, source: &str) -> Observation {
    let name = format!("two_engine_{tag}.stark");
    let front = front_end(&name, source);
    let program = match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
    };
    let hir = run_hir(&name, &front);
    let mir = run_mir(&name, &program);
    if let Some(field) = first_difference(&hir, &mir) {
        panic!(
            "{name}: HIR/MIR DISAGREEMENT on {field}\n--- HIR oracle ---\n{hir:#?}\n--- MIR ---\n{mir:#?}"
        );
    }
    hir
}

/// All three completed normally with exit 0 — i.e. every in-program assertion held in every
/// engine. Meaningful only because `a_false_assertion_traps_in_all_three_engines` proves a FALSE
/// assertion is observable; see `three_engine_differential.rs`.
pub fn agree_completing(tag: &str, source: &str) -> CompletionObservation {
    match three_engine(tag, source) {
        Observation::Completed(done) if done.exit_status == 0 => done,
        other => panic!("{tag}: expected normal completion with status 0, got {other:#?}"),
    }
}

/// §8.8. A drop-observing case: all three engines agreed, AND the Drop log is the one stated here —
/// so three engines agreeing on a wrong Drop schedule still fails. `expected` is a list of
/// identities in the order they must be destroyed; sequence numbers are checked implicitly by
/// position.
pub fn agree_completing_with_drops(tag: &str, source: &str, expected: &[&str]) {
    let done = agree_completing(tag, source);
    let observed: Vec<&str> = done.drop_log.iter().map(|e| e.identity.as_str()).collect();
    assert_eq!(observed, expected, "{tag}: Drop log");
    for (index, event) in done.drop_log.iter().enumerate() {
        assert_eq!(
            event.sequence,
            index as u32 + 1,
            "{tag}: Drop sequence numbering"
        );
    }
}

/// §8.7. Runs a case whose `probe()` returns a value, and requires all three engines to agree on
/// the framed result — then pins it to `expected`, so unanimity on a wrong value still fails.
pub fn agree_returning(tag: &str, source: &str, type_tag: &str, expected: &str) {
    let full = format!("{source}{}", probe_wrapper(type_tag));
    let done = agree_completing(tag, &full);
    let returned = done
        .returned_observation
        .as_ref()
        .unwrap_or_else(|| panic!("{tag}: no return frame was observed; got {done:#?}"));
    assert_eq!(returned.type_tag, type_tag, "{tag}: returned type tag");
    assert_eq!(
        String::from_utf8_lossy(&returned.rendered),
        expected,
        "{tag}: returned value"
    );
    assert!(
        done.stdout_bytes.is_empty(),
        "{tag}: a probe case must emit no ordinary stdout — the frame is the observation ({:?})",
        String::from_utf8_lossy(&done.stdout_bytes)
    );
}

/// All three trapped, with the same category at the same source line — and that line is stated
/// here independently, so a case whose three engines agreed on the WRONG location still fails.
pub fn agree_trapping(tag: &str, source: &str, expected: TrapCategory, expected_line: u32) {
    match three_engine(tag, source) {
        Observation::Trapped(trap) => {
            assert_eq!(trap.category, expected, "{tag}: trap category");
            assert_eq!(trap.line, expected_line, "{tag}: trap line");
        }
        other => panic!("{tag}: expected a trap, got {other:#?}"),
    }
}

/// DEV-106 (CD-136): all three trapped with the same category, line AND the same user-supplied
/// MESSAGE. `three_engine` already requires the three observations to be identical, so the message
/// is compared engine-to-engine by construction; `expected_message` additionally pins it to the text
/// the source actually wrote, so three engines agreeing on the WRONG string still fails.
pub fn agree_trapping_with_message(
    tag: &str,
    source: &str,
    expected: TrapCategory,
    expected_line: u32,
    expected_message: &str,
) {
    match three_engine(tag, source) {
        Observation::Trapped(trap) => {
            assert_eq!(trap.category, expected, "{tag}: trap category");
            assert_eq!(trap.line, expected_line, "{tag}: trap line");
            assert_eq!(
                trap.message_class,
                TrapMessageClass::UserMessageExact(expected_message.to_string()),
                "{tag}: trap message class"
            );
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
    ($name:ident, $tag:literal, drops($expected:expr), $source:literal) => {
        #[test]
        fn $name() {
            if !$crate::support::differential::rustc_available() {
                eprintln!("SKIP: no rustc in this environment.");
                return;
            }
            $crate::support::differential::agree_completing_with_drops($tag, $source, $expected);
        }
    };
}
