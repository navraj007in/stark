use starkc::ast;
use starkc::lexer::tokenize;
use starkc::parser::{parse, ParseMode};
use starkc::source::SourceFile;
use std::process::ExitCode;

const USAGE: &str = "\
starkc — compiler for the STARK Core v1 language

Usage:
  starkc parse [--snippet] [--dump] <file.stark>
                              Parse a source file and report diagnostics.
                              --snippet parses the harness block-body form
                              (items + statements) instead of Program.
                              --dump prints the AST on success.
  starkc lex <file.stark>     Dump the token stream (debugging aid)
  starkc --help               Show this help
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((cmd, rest)) if cmd == "parse" => {
            let mut mode = ParseMode::Program;
            let mut dump = false;
            let mut path = None;
            for arg in rest {
                match arg.as_str() {
                    "--snippet" => mode = ParseMode::Snippet,
                    "--dump" => dump = true,
                    _ if path.is_none() => path = Some(arg.clone()),
                    _ => {
                        eprint!("{USAGE}");
                        return ExitCode::from(2);
                    }
                }
            }
            let Some(path) = path else {
                eprint!("{USAGE}");
                return ExitCode::from(2);
            };
            cmd_parse(&path, mode, dump)
        }
        Some((cmd, [path])) if cmd == "lex" => cmd_lex(path),
        Some((flag, [])) if flag == "--help" || flag == "-h" => {
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

fn cmd_parse(path: &str, mode: ParseMode, dump: bool) -> ExitCode {
    let file = match load(path) {
        Ok(f) => f,
        Err(code) => return code,
    };
    let (tree, diags) = parse(&file, mode);
    for diag in &diags {
        eprint!("{}", diag.render(&file));
    }
    if !diags.is_empty() {
        eprintln!("{}: {} error(s)", file.name, diags.len());
        return ExitCode::FAILURE;
    }
    if dump {
        print!("{}", ast::dump(&tree, &file));
    } else {
        println!("{}: OK", file.name);
    }
    ExitCode::SUCCESS
}
