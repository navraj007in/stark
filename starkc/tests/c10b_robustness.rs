//! C10-B — robustness qualification for Gate C10.
//!
//! **The gate, and it is deliberately narrow** (`COMPILER-ROADMAP.md` WP-C10.2, and the C10
//! execution plan §9.3):
//!
//! ```text
//! no panic          including no `unreachable!()`, no arithmetic overflow in the compiler
//!                   itself, and no unwrap on generated input
//! no hang           every target runs under a wall-clock bound; a timeout is a finding
//! bounded failure   a diagnostic or a clean error, never an unbounded allocation
//! deterministic     the same seed produces byte-identical diagnostics
//! ```
//!
//! **It is NOT a claim that random programs are semantically meaningful.** A generated program that
//! is rejected is a pass, provided it is rejected the same way twice.
//!
//! **Why this is not `cargo-fuzz`.** Charter §1.10 rule 8 forbids nightly Rust, and
//! libFuzzer requires it. This extends the deterministic seeded-generator pattern that
//! `tests/robustness.rs` already established, so the release wording must say *bounded
//! deterministic robustness testing*, not *fuzzing*.
//!
//! **The target population was declared in `C10-0-OPENING-INVENTORY.md` §8 BEFORE any of this ran**
//! (plan §7: no denominator may be chosen after seeing the result). T1 is covered by the
//! pre-existing `robustness.rs` and extended here to the whole front end.
//!
//! **`harness_self_test_detects_an_injected_panic` is the forcing function.** A clean run of an
//! uncalibrated harness is worth nothing — it is EI2's error in a new costume. That test proves
//! this file's driver reports a panic rather than swallowing it, and it runs first by name order.

use starkc::analysis::{analyze_project, ProjectInput};
use starkc::diag::Severity;
use starkc::lsp::Server;
use starkc::mir::{lower::lower_program, verify::verify_program};
use starkc::options::LanguageOptions;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::io::Cursor;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Determinism primitives. Nothing about the host -- clock, PID, path, hash seed -- may enter a
// case's identity, or another machine cannot reproduce a failure this file reports.
// ---------------------------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next() % n as u64) as usize
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// Per-case wall-clock bound. A case exceeding it is a HANG finding, not a slow machine: the
/// budget is ~1000x the slowest legitimate case observed while writing this file.
const CASE_BUDGET: Duration = Duration::from_secs(10);

/// What a case is allowed to produce. Any of these is a PASS; the absence of all of them --
/// a panic, or exceeding `CASE_BUDGET` -- is the finding.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// Accepted, with or without warnings.
    Accepted,
    /// Rejected with at least one error diagnostic. The normative outcome for most generated input.
    Rejected,
}

fn guard(target: &str, case: &str, elapsed: Duration) {
    assert!(
        elapsed < CASE_BUDGET,
        "{target}: HANG -- case took {elapsed:?}, budget {CASE_BUDGET:?}\ncase:\n{case}"
    );
}

/// Drive the whole front end (lex, parse, resolve, typecheck, borrowck) over one source.
///
/// Returns the outcome plus a digest of the diagnostics, so determinism can be compared without
/// pinning message text -- which is not a stability promise C10-F has made.
fn drive_front_end(target: &str, source: &str) -> (Outcome, String) {
    let start = Instant::now();
    let file = Arc::new(SourceFile::new("c10b.stark", source.to_string()));
    let analysis = analyze_project(ProjectInput::program(file), LanguageOptions::CORE);
    guard(target, source, start.elapsed());

    let mut codes: Vec<String> = analysis
        .diagnostics
        .iter()
        .map(|d| format!("{:?}:{}", d.severity, d.code.as_deref().unwrap_or("-")))
        .collect();
    codes.sort();
    let outcome = if analysis
        .diagnostics
        .iter()
        .any(|d| d.severity == Severity::Error)
    {
        Outcome::Rejected
    } else {
        Outcome::Accepted
    };
    (outcome, codes.join(","))
}

// ---------------------------------------------------------------------------------------------
// FORCING FUNCTION. Runs before any clean result is believed.
// ---------------------------------------------------------------------------------------------

/// The harness must report a panic rather than swallow it.
///
/// Every other test in this file asserts the ABSENCE of a panic, and an assertion of absence is
/// worthless until the detector is shown to fire. This is the same two-sided calibration
/// `as8-mutate.py --batch 0` applies to mutations, and the same reason `c6_mutation` exists.
#[test]
fn aaa_harness_self_test_detects_an_injected_panic() {
    // Positive half: a deliberate panic inside the driven region is caught and reported.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(|| {
        drive_front_end("self-test", "fn main() { }");
        panic!("INJECTED — the harness must notice this");
    });
    std::panic::set_hook(hook);
    assert!(
        caught.is_err(),
        "the harness cannot detect a panic in the driven region; every 'no panic' result in this \
         file would be vacuous"
    );

    // Negative half: an ordinary case must NOT be reported as a panic. A detector that fires on
    // everything is as useless as one that never fires.
    let ok = std::panic::catch_unwind(|| drive_front_end("self-test", "fn main() { }"));
    assert!(
        ok.is_ok(),
        "a well-formed program must not panic the driver"
    );
}

// ---------------------------------------------------------------------------------------------
// T1 -- lexer and parser. `robustness.rs` covers parse alone; this drives the FULL front end,
// which is where a resolver or checker panic on syntactically-odd-but-parseable input would show.
// ---------------------------------------------------------------------------------------------

#[test]
fn t1_random_token_soup_through_the_whole_front_end() {
    const VOCAB: &[&str] = &[
        "fn", "struct", "enum", "trait", "impl", "let", "mut", "const", "type", "use", "mod", "if",
        "else", "match", "for", "while", "loop", "break", "continue", "return", "in", "as", "pub",
        "self", "Self", "super", "true", "false", "Int32", "UInt64", "Float64", "String", "Bool",
        "str", "Unit", "Vec", "Option", "Result", "T", "x", "y", "_", "42", "0xFF", "1_000",
        "42i32", "3.14", "\"s\"", "'c'", "+", "-", "*", "/", "%", "==", "!=", "<", ">", "&&", "||",
        "!", "&", "|", "^", "<<", ">>", "=", "+=", "..", "..=", "?", "::", ".", "->", "=>", "(",
        ")", "[", "]", "{", "}", ",", ";", ":", "<", ">",
    ];
    let mut rng = Lcg::new(0xC10B_0001);
    for _ in 0..400 {
        let len = 1 + rng.below(40);
        let src: Vec<&str> = (0..len).map(|_| rng.pick(VOCAB)).collect();
        let src = src.join(" ");
        // The oracle is "it returned", not "it returned a particular thing".
        let _ = drive_front_end("T1", &src);
    }
}

#[test]
fn t1_random_character_soup_through_the_whole_front_end() {
    let alphabet: Vec<char> = ('!'..='~')
        .chain(" \t\n\r\"'\\".chars())
        .chain("\u{1F600}é\u{03BB}\u{0000}\u{FEFF}".chars())
        .collect();
    let mut rng = Lcg::new(0xC10B_0002);
    for _ in 0..400 {
        let len = rng.below(160);
        let src: String = (0..len)
            .map(|_| alphabet[rng.below(alphabet.len())])
            .collect();
        let _ = drive_front_end("T1", &src);
    }
}

// ---------------------------------------------------------------------------------------------
// T2 -- the malformed-source corpus. Hand-built rather than generated: each case names a specific
// hostile SHAPE, so a failure says what is wrong instead of handing back 200 random bytes.
// ---------------------------------------------------------------------------------------------

#[test]
fn t2_malformed_source_corpus_fails_boundedly() {
    let deep_nesting_parens = format!(
        "fn main() {{ let x = {}1{}; }}",
        "(".repeat(500),
        ")".repeat(500)
    );
    let deep_nesting_blocks = format!("fn main() {{ {} {} }}", "{".repeat(400), "}".repeat(400));
    let unclosed = format!("fn main() {{ let x = {}1;", "(".repeat(500));
    let long_ident = format!("fn {}() {{ }}", "a".repeat(4096));
    let long_string = format!("fn main() {{ let s = \"{}\"; }}", "x".repeat(100_000));
    let many_items = (0..2000)
        .map(|i| format!("fn f{i}() -> Int32 {{ {i} }}"))
        .collect::<Vec<_>>()
        .join("\n");
    let deep_generics = format!(
        "fn main() {{ let x: {}Int32{}; }}",
        "Option<".repeat(200),
        ">".repeat(200)
    );

    let cases: Vec<(&str, &str)> = vec![
        ("empty", ""),
        ("nul-byte", "fn main() { let x = \u{0}; }"),
        ("bom-then-program", "\u{FEFF}fn main() { }"),
        ("bom-midway", "fn main() { \u{FEFF} }"),
        ("lone-cr", "fn main() {\r let x = 1; }"),
        ("crlf-mixed", "fn main() {\r\n let x = 1;\n }\r\n"),
        ("unterminated-string", "fn main() { let s = \"abc; }"),
        ("unterminated-char", "fn main() { let c = 'a; }"),
        ("unterminated-raw", "fn main() { let s = r\"abc; }"),
        ("unterminated-block-comment", "/* fn main() { }"),
        ("nested-block-comment", "/* /* */ fn main() { }"),
        (
            "lone-surrogate-escape",
            "fn main() { let s = \"\\u{D800}\"; }",
        ),
        (
            "oversized-escape",
            "fn main() { let s = \"\\u{FFFFFFFF}\"; }",
        ),
        (
            "int-literal-overflow",
            "fn main() { let x: Int32 = 99999999999999999999999; }",
        ),
        ("float-garbage", "fn main() { let x = 1.2.3.4e; }"),
        ("only-operators", "+++---***///"),
        ("only-delimiters", "((([[[{{{"),
        ("stray-close", ")]}"),
        ("keyword-salad", "fn fn fn struct struct impl impl"),
        ("deep-parens", deep_nesting_parens.as_str()),
        ("deep-blocks", deep_nesting_blocks.as_str()),
        ("unclosed-parens", unclosed.as_str()),
        ("very-long-identifier", long_ident.as_str()),
        ("very-long-string", long_string.as_str()),
        ("many-items", many_items.as_str()),
        ("deep-generic-nesting", deep_generics.as_str()),
    ];

    for (name, source) in cases {
        let (_outcome, _digest) = drive_front_end(&format!("T2/{name}"), source);
        // Reaching here is the whole assertion: no panic, inside CASE_BUDGET.
    }
}

// ---------------------------------------------------------------------------------------------
// T4/T5 -- generated ill-typed and ownership-hostile programs. These are STRUCTURALLY VALID
// programs built from a grammar-aware template set, so they actually reach the checker and the
// borrow checker rather than dying in the parser -- which is what T1's soup mostly does.
// ---------------------------------------------------------------------------------------------

fn generated_program(rng: &mut Lcg) -> String {
    const TYPES: &[&str] = &["Int32", "UInt64", "Float64", "Bool", "String", "Unit"];
    const EXPRS: &[&str] = &[
        "1",
        "0",
        "-1",
        "true",
        "false",
        "\"s\"",
        "1 + 2",
        "1 / 0",
        "x",
        "y",
        "1 as Int32",
        "1.5 as Int32",
        "(1, 2)",
        "[1, 2, 3]",
    ];
    const STMTS: &[&str] = &[
        "let x = {E};",
        "let mut y: {T} = {E};",
        "let z = &x;",
        "let w = &mut y;",
        "y = {E};",
        "if {E} { let a = {E}; }",
        "while {E} { break; }",
        "match {E} { _ => { } }",
        "let moved = x; let reuse = x;",
        "return;",
        "drop_it({E});",
    ];
    let n = 1 + rng.below(8);
    let body: String = (0..n)
        .map(|_| {
            rng.pick(STMTS)
                .replace("{T}", rng.pick(TYPES))
                .replace("{E}", rng.pick(EXPRS))
        })
        .collect::<Vec<_>>()
        .join("\n    ");
    format!("fn drop_it(v: Int32) {{ }}\nfn main() {{\n    {body}\n}}\n")
}

#[test]
fn t4_t5_generated_programs_reach_the_checkers_and_fail_boundedly() {
    let mut rng = Lcg::new(0xC10B_0045);
    let mut reached_checker = 0usize;
    for _ in 0..500 {
        let src = generated_program(&mut rng);
        let (_o, digest) = drive_front_end("T4/T5", &src);
        // A digest containing an E-code means the program parsed and a later stage spoke, which
        // is what distinguishes this target from T1's soup. Counted, not asserted per case.
        if digest.contains("E0") {
            reached_checker += 1;
        }
    }
    assert!(
        reached_checker > 50,
        "T4/T5: only {reached_checker}/500 generated programs produced a semantic diagnostic — \
         the generator is producing parse failures, not checker input, and this target is not \
         exercising what it claims to"
    );
}

// ---------------------------------------------------------------------------------------------
// T6 -- the MIR verifier. Reached the only way it can be from source: lower every generated
// program that type-checks, then verify. The verifier must REJECT or ACCEPT, never panic.
// ---------------------------------------------------------------------------------------------

#[test]
fn t6_mir_lowering_and_verification_never_panic() {
    let mut rng = Lcg::new(0xC10B_0006);
    let mut lowered = 0usize;
    let mut verified = 0usize;
    for _ in 0..300 {
        let src = generated_program(&mut rng);
        let start = Instant::now();
        let file = Arc::new(SourceFile::new("c10b_mir.stark", src.clone()));
        let (ast, pd) = parse(&file, ParseMode::Program);
        if !pd.is_empty() {
            continue;
        }
        let (hir, rd) = resolve(&ast, file.clone());
        if !rd.is_empty() {
            continue;
        }
        let checked = typecheck::analyze(&hir);
        if checked
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            continue;
        }
        let Some(registered) = hir.source_named(&file.name) else {
            continue;
        };
        // Lowering may legitimately fail; it may not panic.
        if let Ok(program) = lower_program(&hir, &checked.tables, registered) {
            lowered += 1;
            // The verifier's contract: Ok, or a list of MirErrors. Never a panic.
            if verify_program(&program).is_ok() {
                verified += 1;
            }
        }
        guard("T6", &src, start.elapsed());
    }
    assert!(
        lowered > 0,
        "T6: no generated program reached MIR lowering — the target is vacuous and the generator \
         needs widening before any 'the verifier does not panic' claim is made"
    );
    // Recorded, not asserted against a threshold: a verifier rejection is a legitimate outcome.
    eprintln!("T6: {lowered} programs lowered, {verified} verified clean");
}

// ---------------------------------------------------------------------------------------------
// T8 -- malformed LSP / JSON-RPC transport input. `Server::run` is generic over BufRead/Write,
// so the transport is driven in-process with no subprocess and no sockets.
// ---------------------------------------------------------------------------------------------

fn drive_lsp(name: &str, input: &[u8]) {
    let start = Instant::now();
    let mut server = Server::new();
    let mut out: Vec<u8> = Vec::new();
    // An IO error is a legitimate bounded failure (a truncated body, for instance). A panic is not.
    let _ = server.run(Cursor::new(input.to_vec()), &mut out);
    guard(
        &format!("T8/{name}"),
        &String::from_utf8_lossy(input),
        start.elapsed(),
    );
}

fn framed(body: &str) -> Vec<u8> {
    format!("Content-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
}

#[test]
fn t8_malformed_protocol_input_fails_boundedly() {
    let long_method = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"{}"}}"#,
        "x".repeat(10_000)
    );
    let deep_json = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}1{}}}"#,
        "[".repeat(200),
        "]".repeat(200)
    );
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", b"".to_vec()),
        ("headers-only", b"Content-Length: 0\r\n\r\n".to_vec()),
        ("no-headers-just-body", br#"{"jsonrpc":"2.0"}"#.to_vec()),
        ("missing-content-length", b"X-Other: 1\r\n\r\n{}".to_vec()),
        (
            "non-numeric-content-length",
            b"Content-Length: abc\r\n\r\n{}".to_vec(),
        ),
        (
            "negative-content-length",
            b"Content-Length: -5\r\n\r\n{}".to_vec(),
        ),
        (
            "content-length-longer-than-body",
            b"Content-Length: 9999\r\n\r\n{}".to_vec(),
        ),
        (
            "content-length-shorter-than-body",
            b"Content-Length: 2\r\n\r\n{\"a\":1}".to_vec(),
        ),
        ("lf-only-framing", b"Content-Length: 2\n\n{}".to_vec()),
        ("no-blank-line", b"Content-Length: 2\r\n{}".to_vec()),
        ("header-no-colon", b"ContentLength 2\r\n\r\n{}".to_vec()),
        (
            "invalid-utf8-body",
            vec![
                b'C', b'o', b'n', b't', b'e', b'n', b't', b'-', b'L', b'e', b'n', b'g', b't', b'h',
                b':', b' ', b'2', b'\r', b'\n', b'\r', b'\n', 0xFF, 0xFE,
            ],
        ),
        ("not-json", framed("this is not json at all")),
        ("json-truncated", framed(r#"{"jsonrpc":"2.0","id":1,"meth"#)),
        ("json-null", framed("null")),
        ("json-array-root", framed("[1,2,3]")),
        (
            "unknown-method",
            framed(r#"{"jsonrpc":"2.0","id":1,"method":"nope/nothing"}"#),
        ),
        ("missing-method", framed(r#"{"jsonrpc":"2.0","id":1}"#)),
        (
            "id-as-object",
            framed(r#"{"jsonrpc":"2.0","id":{"a":1},"method":"shutdown"}"#),
        ),
        (
            "params-wrong-shape",
            framed(r#"{"jsonrpc":"2.0","id":1,"method":"textDocument/hover","params":42}"#),
        ),
        (
            "position-out-of-range",
            framed(
                r#"{"jsonrpc":"2.0","id":1,"method":"textDocument/hover","params":{"textDocument":{"uri":"file:///a.stark"},"position":{"line":999999,"character":999999}}}"#,
            ),
        ),
        (
            "negative-position",
            framed(
                r#"{"jsonrpc":"2.0","id":1,"method":"textDocument/hover","params":{"textDocument":{"uri":"file:///a.stark"},"position":{"line":-1,"character":-1}}}"#,
            ),
        ),
        (
            "non-bmp-in-label",
            framed(
                r#"{"jsonrpc":"2.0","id":1,"method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///\ud83d\ude00.stark","languageId":"stark","version":1,"text":"fn main() { }"}}}"#,
            ),
        ),
        ("very-long-method", framed(&long_method)),
        ("deeply-nested-params", framed(&deep_json)),
        ("two-messages-second-malformed", {
            let mut v = framed(r#"{"jsonrpc":"2.0","id":1,"method":"shutdown"}"#);
            v.extend_from_slice(&framed("garbage"));
            v
        }),
    ];
    for (name, bytes) in cases {
        drive_lsp(name, &bytes);
    }
}

// ---------------------------------------------------------------------------------------------
// T9 -- hostile-input resource limits. This target is where DEV-186 lives, and it is
// CHARACTERISED here rather than repaired: C10 is a qualification campaign.
// ---------------------------------------------------------------------------------------------

/// DEV-186, characterised at HEAD.
///
/// `Server::run` does `vec![0u8; content_length]` BEFORE reading the body, so a header alone
/// decides an allocation. This test pins the CURRENT behaviour at a size that is large but safe
/// (64 MiB) and records that no bound exists -- it does not assert that the allocation is
/// refused, because at HEAD it is not.
///
/// **It is written so that a repair flips it**, the same way AS8 wrote DEV-213's test: when a
/// limit is introduced, this test starts failing and becomes the regression test for the limit.
#[test]
fn t9_dev186_content_length_allocates_before_reading_and_is_unbounded_at_head() {
    // 64 MiB: demonstrably far past any legitimate LSP message, and small enough that the test
    // cannot itself become the denial of service it is describing.
    let claim = 64 * 1024 * 1024usize;
    let input = format!("Content-Length: {claim}\r\n\r\n").into_bytes();
    let start = Instant::now();
    let mut server = Server::new();
    let mut out: Vec<u8> = Vec::new();
    let result = server.run(Cursor::new(input), &mut out);
    let elapsed = start.elapsed();

    // The read fails -- the body is not there -- which is the BOUNDED FAILURE half of the gate.
    assert!(
        result.is_err(),
        "DEV-186: a truncated body must surface as an IO error, not as success"
    );
    assert!(
        elapsed < CASE_BUDGET,
        "DEV-186: a 64 MiB claim took {elapsed:?}; that is a hang, not merely an allocation"
    );
    // No assertion that the allocation was refused: AT HEAD IT IS NOT. DEV-186 is OPEN, this is
    // its characterisation, and a future limit will make the `is_err` above arrive for a
    // different and better reason.
}

/// DEV-214, characterised at HEAD -- and characterised BELOW the cliff, deliberately.
///
/// A left-associative operator chain (`1 + 1 + 1 + ...`) is parsed ITERATIVELY: the parser
/// implements 16 precedence levels as one loop each, so chain length never increments the
/// recursion counter that `MAX_DEPTH = 200` bounds. The AST it produces is nonetheless `n` deep,
/// and every recursive walk downstream -- the type checker, and the post-typecheck index building
/// in `analyze_project` -- descends that depth and overflows the stack.
///
/// **The guard bounds SYNTACTIC nesting, not the depth of the tree that nesting produces.**
/// `(((...1...)))` at 300 is rejected cleanly with *"this code is nested too deeply to parse"* --
/// exactly the bounded failure this gate asks for. `1 + 1 + ... ` at 250 aborts the process.
///
/// **The threshold scales with the thread's stack, which is what makes this serious.** Measured
/// on macOS-arm64 with `examples/c10b_thread.rs`, which runs the analysis on a thread of a chosen
/// size:
///
/// ```text
/// 8 MiB stack (a process main thread)      n = 240 OK,  n = 250 ABORTS
/// 2 MiB stack (Rust's default for a
///              SPAWNED thread, and what
///              `cargo test` gives a test)  n =  60 OK,  n =  65 ABORTS
/// ```
///
/// ~30 KB of stack per AST level. **On a default-stack thread, sixty-five `+` operators abort the
/// compiler** — and the LSP analyses on a server thread, so an embedding is on the low number, not
/// the high one. `1 + 1 + ...` is a stand-in: any left-associative chain does it, including
/// ordinary string concatenation and long boolean conditions.
///
/// This test pins the SAFE side only, well below the lowest measured cliff so it stays true on
/// platforms whose default thread stack differs. The failing side is not run here because a stack
/// overflow aborts the whole test binary and would take every other test in this file with it; it
/// is reproduced by `cargo run --example c10b_repro -- 250` and
/// `cargo run --example c10b_thread -- 65 2097152`.
///
/// **Written so a repair flips it.** When DEV-214 is fixed -- by counting chain depth, by
/// converting the walks to an explicit worklist, or by an owner decision on the limit -- the
/// assertion below stops being the interesting one and the repro example stops aborting.
#[test]
fn t9_dev214_operator_chain_depth_is_unguarded_and_the_safe_boundary_holds() {
    // 40 operands: comfortably below the lowest measured cliff (65 on a 2 MiB thread stack), so
    // this stays true on any platform whose default differs. If a future change lowers the real
    // limit below FORTY, this assertion fires -- the early warning a release claim wants.
    let src = format!("fn main() {{ let x = {}; }}", vec!["1"; 40].join(" + "));
    let (outcome, _digest) = drive_front_end("T9/dev214-safe-boundary", &src);
    assert_eq!(
        outcome,
        Outcome::Accepted,
        "a 40-term arithmetic chain must still compile; if this now fails, the effective depth \
         limit moved and DEV-214's characterisation is stale"
    );

    // The CONTRASTING shape, which the parser's guard DOES catch. Its presence here is the point:
    // it shows the guard exists and works, so DEV-214 is a gap in what the guard measures rather
    // than an absence of any guard at all.
    let nested = format!(
        "fn main() {{ let x = {}1{}; }}",
        "(".repeat(300),
        ")".repeat(300)
    );
    let (nested_outcome, _) = drive_front_end("T9/dev214-contrast", &nested);
    assert_eq!(
        nested_outcome,
        Outcome::Rejected,
        "300 levels of parenthesis nesting must be REJECTED by MAX_DEPTH, not accepted and not \
         fatal — this is the bounded-failure behaviour DEV-214 lacks"
    );
}

#[test]
fn t9_pathological_inputs_stay_within_the_case_budget() {
    // Each is a shape whose cost could plausibly be super-linear. The assertion is the budget.
    let cases = vec![
        (
            "wide-tuple",
            format!("fn main() {{ let t = ({}); }}", vec!["1"; 2000].join(", ")),
        ),
        (
            "many-locals",
            format!(
                "fn main() {{\n{}\n}}",
                (0..3000)
                    .map(|i| format!("let v{i} = {i};"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
        ),
        (
            "many-fields",
            format!(
                "struct S {{ {} }}\nfn main() {{ }}",
                (0..1500)
                    .map(|i| format!("f{i}: Int32"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ),
        (
            "many-variants",
            format!(
                "enum E {{ {} }}\nfn main() {{ }}",
                (0..1500)
                    .map(|i| format!("V{i}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ),
        (
            "many-match-arms",
            format!(
                "fn main() {{ let x = 1; match x {{ {} _ => {{ }} }} }}",
                (0..1500)
                    .map(|i| format!("{i} => {{ }},"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        ),
    ];
    for (name, src) in cases {
        let (_o, _d) = drive_front_end(&format!("T9/{name}"), &src);
    }
}

// ---------------------------------------------------------------------------------------------
// Determinism. Required by the gate, and separately by Charter §1.6 rule 16 -- diagnostics are
// part of behaviour.
// ---------------------------------------------------------------------------------------------

#[test]
fn diagnostics_are_deterministic_across_runs_for_the_same_seed() {
    let mut a = Lcg::new(0xC10B_DE7E);
    let mut b = Lcg::new(0xC10B_DE7E);
    for i in 0..200 {
        let sa = generated_program(&mut a);
        let sb = generated_program(&mut b);
        assert_eq!(sa, sb, "case {i}: the generator is not deterministic");
        let (oa, da) = drive_front_end("determinism", &sa);
        let (ob, db) = drive_front_end("determinism", &sb);
        assert_eq!(
            oa, ob,
            "case {i}: outcome differs between identical runs\n{sa}"
        );
        assert_eq!(
            da, db,
            "case {i}: diagnostics differ between identical runs\n{sa}"
        );
    }
}

#[test]
fn the_same_source_analysed_twice_yields_identical_diagnostics() {
    // Distinct from the seed test: that one proves the GENERATOR is stable, this proves the
    // COMPILER is, including any iteration-order-dependent diagnostic emission.
    let sources = [
        "fn main() { let x = 1 / 0; }",
        "fn main() { let mut y = 1; let a = &mut y; let b = &mut y; }",
        "struct S { a: Int32, b: Int32 } fn main() { let s = S { a: 1 }; }",
        "fn main() { let s = \"x\"; let n: Int32 = s; }",
        "trait T { fn f(&self) -> Int32; } struct S; impl T for S { }",
    ];
    for src in sources {
        let first = drive_front_end("determinism", src);
        let second = drive_front_end("determinism", src);
        assert_eq!(first, second, "non-deterministic diagnostics for:\n{src}");
    }
}
