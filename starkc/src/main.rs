use starkc::diag::Diagnostic;
use starkc::source::{SourceFile, Span};
use std::process::ExitCode;

const USAGE: &str = "\
starkc — compiler for the STARK Core v1 language

Usage:
  starkc parse <file.stark>   Parse a source file and report diagnostics
  starkc --help               Show this help
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, path] if cmd == "parse" => cmd_parse(path),
        [flag] if flag == "--help" || flag == "-h" => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        _ => {
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn cmd_parse(path: &str) -> ExitCode {
    let src = match std::fs::read_to_string(path) {
        Ok(src) => src,
        Err(err) => {
            eprintln!("Error: cannot read '{path}': {err}");
            return ExitCode::FAILURE;
        }
    };
    let file = SourceFile::new(path, src);

    // WP1.1 stub: demonstrate span/diagnostic plumbing end to end.
    // WP1.2 replaces this with lexing; WP1.4 with parsing.
    let diag = Diagnostic::error("parsing is not yet implemented", Span::point(0))
        .with_label("starkc is at WP1.1 (scaffold)")
        .with_note("see STARKLANG/docs/PLAN.md for the delivery sequence");
    eprint!("{}", diag.render(&file));
    ExitCode::FAILURE
}
