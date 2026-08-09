//! DEV-214 (OD-9) — the AST structural-depth guard.
//!
//! **The defect.** The parser bounds recursion with `MAX_DEPTH = 200`, which is what makes
//! `(((((...1...)))))` a clean diagnostic. A left-associative chain never recurses in the parser —
//! the precedence table is one `loop` per level — so `1 + 1 + 1 + ...` folded iteratively, the
//! counter never moved, and the *tree* it built was as deep as the chain was long. Every recursive
//! pass downstream then descended it and the process died: SIGABRT at ~65 terms on a 2 MiB thread
//! stack, ~250 on 8 MiB.
//!
//! **The repair (OD-9).** The same limit, measured over the AST the parser produced, enforced
//! after parsing and before any recursive pass consumes it. Not a new limit, not a bigger stack,
//! and not a conversion of the downstream walks to worklists.
//!
//! **Why the failing side runs on a 2 MiB thread.** C10-B established that the cliff scales with
//! the stack: a test on the main thread would pass at sizes that kill a real analysis thread, and
//! would have reported a threshold four times too generous. Every over-limit case below therefore
//! runs on a thread sized like the one an editor or a test harness actually gives the compiler.

use starkc::analysis::{analyze_project, ProjectInput};
use starkc::options::LanguageOptions;
use starkc::parser::MAX_DEPTH;
use starkc::source::SourceFile;
use std::sync::Arc;

/// Rust's default for a SPAWNED thread, and what `cargo test` gives each test. The LSP analyses on
/// a server thread, so this is the stack a real embedding has — not the process main thread's.
const EMBEDDING_STACK: usize = 2 * 1024 * 1024;

/// A process main thread's stack, which is what `stark build` actually runs on.
const MAIN_THREAD_STACK: usize = 8 * 1024 * 1024;

fn chain(n: usize) -> String {
    format!("fn main() {{ let x = {}; }}", vec!["1"; n].join(" + "))
}

/// Analyse on a small-stack thread and return the diagnostic messages.
///
/// If the guard fails to fire, this thread overflows and `join` returns `Err` — so a regression
/// surfaces as a test failure rather than as a SIGABRT that takes the whole binary down.
fn analyse_on_small_stack(src: String) -> Vec<String> {
    analyse_on_stack(src, EMBEDDING_STACK)
}

fn analyse_on_stack(src: String, stack: usize) -> Vec<String> {
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || {
            let file = Arc::new(SourceFile::new("dev214.stark", src));
            let analysis = analyze_project(ProjectInput::program(file), LanguageOptions::CORE);
            analysis
                .diagnostics
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        })
        .expect("spawn")
        .join()
        .expect(
            "DEV-214 REGRESSION: analysis overflowed a 2 MiB stack. The depth guard did not fire \
             before the recursive passes ran",
        )
}

fn is_depth_diagnostic(messages: &[String]) -> bool {
    messages
        .iter()
        .any(|m| m.contains("nested too deeply to analyse"))
}

// ---------------------------------------------------------------------------------------------
// The four controls OD-9 requires, plus the contrast it requires preserved.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_40_term_chain_is_still_accepted() {
    let diags = analyse_on_small_stack(chain(40));
    assert!(
        diags.is_empty(),
        "40 terms must compile cleanly; got {diags:?}"
    );
}

#[test]
fn a_chain_at_exactly_the_limit_is_accepted_on_a_main_thread_stack() {
    // The boundary is the EXISTING limit. If this fails, the repair moved a limit it was told not
    // to move.
    //
    // **Run on an 8 MiB stack, and that is a finding rather than a convenience.** OD-9 asked for
    // both "200 is still accepted" and "no stack overflow even on a 2 MiB thread", and those two
    // cannot both hold at MAX_DEPTH = 200: a 200-deep expression needs roughly 6 MiB of stack in
    // the downstream walks. See `dev214_residual_the_limit_permits_more_than_a_2mib_stack_carries`.
    let n = MAX_DEPTH as usize;
    let diags = analyse_on_stack(chain(n), MAIN_THREAD_STACK);
    assert!(
        diags.is_empty(),
        "a chain at exactly MAX_DEPTH ({n}) must be accepted — OD-9 forbids reducing the limit; \
         got {diags:?}"
    );
}

/// **The residual OD-9's two criteria cannot both satisfy, measured rather than argued.**
///
/// The repair closed the UNBOUNDED hole completely: at any depth over the limit, on either stack,
/// the answer is one diagnostic and no crash. What it did not close — and could not, without
/// reducing `MAX_DEPTH`, which OD-9 forbids — is the window the limit PERMITS but a small stack
/// cannot carry:
///
/// ```text
/// depth        8 MiB (main thread)        2 MiB (spawned / LSP / cargo test)
/// <= 60        accepted                   accepted
/// 61..=200     accepted                   ABORTS  <- the residual
/// > 200        E0209 diagnostic           E0209 diagnostic
/// ```
///
/// This test pins the part of that table it can pin in-process: the safe side on 2 MiB, and the
/// full acceptance range on 8 MiB. The aborting cell is reproduced by
/// `cargo run --example c10b_thread -- 100 2097152`, and is not run here because a stack overflow
/// aborts the whole test binary.
///
/// **Written so a decision flips it.** If the owner later lowers the effective limit for small
/// stacks, or the walks become iterative, the 2 MiB assertion below can be widened to 200 and this
/// residual disappears.
#[test]
fn dev214_residual_the_limit_permits_more_than_a_2mib_stack_carries() {
    // Safe on a 2 MiB stack today.
    assert!(
        analyse_on_small_stack(chain(60)).is_empty(),
        "60 terms must be accepted on a 2 MiB stack"
    );
    // Accepted on a main-thread stack at a depth a 2 MiB stack cannot carry. If this ever starts
    // producing a diagnostic, the effective limit moved and the table above is stale.
    assert!(
        analyse_on_stack(chain(150), MAIN_THREAD_STACK).is_empty(),
        "150 terms must still be accepted on an 8 MiB stack"
    );
}

#[test]
fn one_term_past_the_limit_is_a_deterministic_diagnostic() {
    let n = MAX_DEPTH as usize + 1;
    let diags = analyse_on_small_stack(chain(n));
    assert!(
        is_depth_diagnostic(&diags),
        "{n} terms must produce the depth diagnostic; got {diags:?}"
    );
    // Deterministic: the same source twice gives the same answer, byte for byte.
    assert_eq!(diags, analyse_on_small_stack(chain(n)), "non-deterministic");
}

#[test]
fn far_past_the_limit_is_the_same_bounded_diagnostic_and_never_overflows() {
    // 1_000 and 10_000 both killed the process before the repair — 10_000 by two orders of
    // magnitude. Both must now be ordinary rejections on a 2 MiB stack.
    for n in [1_000usize, 10_000] {
        let diags = analyse_on_small_stack(chain(n));
        assert!(
            is_depth_diagnostic(&diags),
            "{n} terms must produce the depth diagnostic, not a crash and not silence; got \
             {diags:?}"
        );
    }
}

#[test]
fn deep_parenthesis_nesting_is_still_rejected_by_the_parser() {
    // The behaviour the repair must PRESERVE. This was already correct, and a guard that broke it
    // would have replaced one bounded rejection with another for the wrong reason.
    let src = format!(
        "fn main() {{ let x = {}1{}; }}",
        "(".repeat(300),
        ")".repeat(300)
    );
    let diags = analyse_on_small_stack(src);
    assert!(
        diags
            .iter()
            .any(|m| m.contains("nested too deeply to parse")),
        "300 nested parentheses must still be rejected BY THE PARSER; got {diags:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// The guard's own properties.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_rejection_is_not_followed_by_partial_semantic_analysis() {
    // "no partial semantic analysis" from OD-9. A program rejected for depth must not then be
    // half-resolved by the very walks the rejection exists to protect: the depth diagnostic must
    // be the ONLY one, not the first of a cascade.
    let src = chain(MAX_DEPTH as usize + 50);
    let diags = analyse_on_small_stack(src);
    assert!(
        is_depth_diagnostic(&diags),
        "expected the depth diagnostic; got {diags:?}"
    );
    assert_eq!(
        diags.len(),
        1,
        "the depth rejection must short-circuit the pipeline, not lead a cascade; got {diags:?}"
    );
}

#[test]
fn the_diagnostic_blames_the_expression_rather_than_the_file() {
    // A span pointing at nothing in particular makes the user hunt for the expression. The guard
    // reports the deepest node's own span.
    let src = chain(MAX_DEPTH as usize + 1);
    let file = Arc::new(SourceFile::new("dev214.stark", src.clone()));
    let analysis = analyze_project(ProjectInput::program(file), LanguageOptions::CORE);
    let d = analysis
        .diagnostics
        .iter()
        .find(|d| d.message.contains("nested too deeply to analyse"))
        .expect("depth diagnostic");
    assert!(
        d.span.hi > d.span.lo,
        "the diagnostic must carry a real span, not an empty one: {:?}",
        d.span
    );
    assert!(
        (d.span.hi as usize) <= src.len(),
        "span must lie inside the source: {:?} vs len {}",
        d.span,
        src.len()
    );
    assert_eq!(d.code.as_deref(), Some("E0209"));
}

#[test]
fn other_deep_shapes_that_previously_survived_are_unaffected() {
    // The guard must not start rejecting shapes that were always fine. Each of these is WIDE
    // rather than DEEP: a 2,000-element tuple is depth 2, not depth 2,000.
    let cases = [
        format!("fn main() {{ let t = ({}); }}", vec!["1"; 2000].join(", ")),
        format!(
            "fn main() {{\n{}\n}}",
            (0..2000)
                .map(|i| format!("let v{i} = {i};"))
                .collect::<Vec<_>>()
                .join("\n")
        ),
        format!(
            "struct S {{ {} }}\nfn main() {{ }}",
            (0..1500)
                .map(|i| format!("f{i}: Int32"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    ];
    for src in cases {
        let diags = analyse_on_small_stack(src);
        assert!(
            !is_depth_diagnostic(&diags),
            "a WIDE construct was rejected as too DEEP — the guard is measuring the wrong thing; \
             got {diags:?}"
        );
    }
}
