use starkc::diag::Diagnostic;
use starkc::lexer::{tokenize, TokenKind};
use starkc::source::{SourceFile, Span};
use std::process::ExitCode;

const USAGE: &str = "\
starkc — compiler for the STARK Core v1 language

Usage:
  starkc parse <file.stark>   Parse a source file and report diagnostics
  starkc lex <file.stark>     Dump the token stream (debugging aid)
  starkc --help               Show this help
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, path] if cmd == "parse" => cmd_parse(path),
        [cmd, path] if cmd == "lex" => cmd_lex(path),
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

fn load(path: &str) -> Result<SourceFile, ExitCode> {
    match std::fs::read_to_string(path) {
        Ok(src) => Ok(SourceFile::new(path, src)),
        Err(err) => {
            eprintln!("Error: cannot read '{path}': {err}");
            Err(ExitCode::FAILURE)
        }
    }
}

fn cmd_lex(path: &str) -> ExitCode {
    let file = match load(path) {
        Ok(f) => f,
        Err(code) => return code,
    };
    let (tokens, diags) = tokenize(&file);
    for token in &tokens {
        let (line, col) = file.line_col(token.span.lo);
        let text = &file.src[token.span.lo as usize..token.span.hi as usize];
        println!("{line}:{col}\t{:?}\t{text:?}", token.kind);
    }
    for diag in &diags {
        eprint!("{}", diag.render(&file));
    }
    if diags.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn cmd_parse(path: &str) -> ExitCode {
    let file = match load(path) {
        Ok(f) => f,
        Err(code) => return code,
    };
    let (tokens, diags) = tokenize(&file);
    for diag in &diags {
        eprint!("{}", diag.render(&file));
    }
    if !diags.is_empty() {
        return ExitCode::FAILURE;
    }

    // WP1.2 state: lexing is real, parsing is not. WP1.4 replaces this stub.
    let first_real = tokens
        .iter()
        .find(|t| t.kind != TokenKind::Eof)
        .map_or(Span::point(0), |t| t.span);
    let diag = Diagnostic::error("parsing is not yet implemented", first_real)
        .with_label("lexed successfully; the parser lands in WP1.4")
        .with_note("see STARKLANG/docs/PLAN.md for the delivery sequence");
    eprint!("{}", diag.render(&file));
    ExitCode::FAILURE
}
