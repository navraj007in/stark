//! C10-E — per-phase compiler timing, in-process.
//!
//! Plan §12.2 requires lex / parse / resolve / check split out. `c7-baseline.py` measures the
//! STARK-vs-cargo split of a whole `stark build`, which is a different and coarser question: it
//! answers "how much of a build is Cargo" and cannot answer "where does the front end spend its
//! time".
//!
//! Timed IN-PROCESS and cumulatively, because the phases are a pipeline: parse needs the tokens,
//! resolve needs the AST, check needs the HIR. Each figure is the marginal cost of that stage over
//! the previous one, so they sum to the front-end total by construction rather than by hope.
//!
//! **Not a benchmark harness.** No warmup, no statistical machinery — the driver takes the median
//! of repeated runs. Reporting one number from one run would be the kind of measurement §12.2's
//! own harness header warns about, where a method error produced a -0.3% host share and only its
//! impossibility revealed it.
//!
//! ```text
//! cargo run --example c10e_phases -- <file.stark> [reps]
//! ```
//! Emits one JSON object on stdout: nanoseconds per phase, plus sizes.

use starkc::lexer;
use starkc::options::LanguageOptions;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve_with_options;
use starkc::source::{SourceFile, SourceRegistry};
use starkc::typecheck;
use std::sync::Arc;
use std::time::Instant;

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: c10e_phases <file.stark> [reps]");
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(9);
    let src = std::fs::read_to_string(&path).expect("read source");

    let (mut lex_ns, mut parse_ns, mut resolve_ns, mut check_ns) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut tokens = 0usize;

    for _ in 0..reps {
        let file = Arc::new(SourceFile::new(&path, src.clone()));

        // A `SourceId` is registry-local by design and has no public constructor, so the
        // registry is built OUTSIDE the timed region — interning is not lexing.
        let mut registry = SourceRegistry::default();
        let registered = registry.intern(file.clone());
        let sid = registered.id();

        let t = Instant::now();
        let (toks, _) = lexer::tokenize(&file, sid);
        lex_ns.push(t.elapsed().as_nanos());
        tokens = toks.len();

        let t = Instant::now();
        let (ast, pd) = parse(&file, ParseMode::Program);
        parse_ns.push(t.elapsed().as_nanos());
        if !pd.is_empty() {
            eprintln!("parse diagnostics: {}", pd.len());
        }

        let t = Instant::now();
        let (hir, _) = resolve_with_options(&ast, file.clone(), LanguageOptions::CORE);
        resolve_ns.push(t.elapsed().as_nanos());

        let t = Instant::now();
        let _ = typecheck::analyze_with_options(&hir, LanguageOptions::CORE);
        check_ns.push(t.elapsed().as_nanos());
    }

    println!(
        "{{\"file\":{:?},\"bytes\":{},\"tokens\":{},\"reps\":{},\
         \"lex_ns\":{},\"parse_ns\":{},\"resolve_ns\":{},\"check_ns\":{}}}",
        path,
        src.len(),
        tokens,
        reps,
        median(lex_ns),
        median(parse_ns),
        median(resolve_ns),
        median(check_ns)
    );
}
