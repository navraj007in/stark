use starkc::ast_dump;
use starkc::lexer::tokenize;
use starkc::parser::{parse, ParseMode};
use starkc::source::SourceFile;
use std::process::ExitCode;

const USAGE: &str = "\
starkc — compiler for the STARK Core v1 language

Usage:
  starkc check [--snippet] <file.stark> Check a source file and report semantic diagnostics.
  starkc run <file.stark>               Check and execute a Core program.
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
        Some((cmd, rest)) if cmd == "check" => {
            let mut mode = ParseMode::Program;
            let mut path = None;
            for arg in rest {
                match arg.as_str() {
                    "--snippet" => mode = ParseMode::Snippet,
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
            cmd_check(&path, mode)
        }
        Some((cmd, [path])) if cmd == "run" => cmd_run(path),
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
        print!("{}", ast_dump::dump(&tree, &file));
    } else {
        println!("{}: OK", file.name);
    }
    ExitCode::SUCCESS
}

fn cmd_check(path: &str, mode: ParseMode) -> ExitCode {
    let file = match load(path) {
        Ok(f) => f,
        Err(code) => return code,
    };
    let (tree, mut diags) = parse(&file, mode);
    if !diags.is_empty() {
        for diag in &diags {
            eprint!("{}", diag.render(&file));
        }
        eprintln!("{}: {} error(s)", file.name, diags.len());
        return ExitCode::FAILURE;
    }

    let file_arc = std::sync::Arc::new(file);
    let (hir, mut sem_diags) = starkc::resolve::resolve(&tree, file_arc.clone());
    diags.append(&mut sem_diags);

    if diags.is_empty() {
        let mut type_diags = starkc::typecheck::check(&hir, file_arc.clone());
        diags.append(&mut type_diags);
    }

    for diag in &diags {
        eprint!("{}", diag.render(&file_arc));
    }
    let error_count = diags
        .iter()
        .filter(|diag| diag.severity == starkc::diag::Severity::Error)
        .count();
    if error_count > 0 {
        eprintln!("{}: {} error(s)", file_arc.name, error_count);
        return ExitCode::FAILURE;
    }
    println!("{}: OK", file_arc.name);
    ExitCode::SUCCESS
}

fn cmd_run(path: &str) -> ExitCode {
    let file = match load(path) {
        Ok(file) => file,
        Err(code) => return code,
    };
    let (tree, mut diagnostics) = parse(&file, ParseMode::Program);
    let file = std::sync::Arc::new(file);
    if diagnostics.is_empty() {
        let (hir, mut resolution) = starkc::resolve::resolve(&tree, file.clone());
        diagnostics.append(&mut resolution);
        if diagnostics.is_empty() {
            let checked = starkc::typecheck::analyze(&hir, file.clone());
            diagnostics.extend(checked.diagnostics);
            for diagnostic in &diagnostics {
                eprint!("{}", diagnostic.render(&file));
            }
            if diagnostics
                .iter()
                .any(|diagnostic| diagnostic.severity == starkc::diag::Severity::Error)
            {
                return ExitCode::FAILURE;
            }
            return match starkc::interp::run(&hir, file.clone(), &checked.tables) {
                Ok(execution) => {
                    print!("{}", execution.output);
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    let diagnostic = starkc::diag::Diagnostic::error(
                        format!("runtime error: {}", error.message),
                        error.span,
                    );
                    eprint!("{}", diagnostic.render(&file));
                    ExitCode::FAILURE
                }
            };
        }
    }
    for diagnostic in &diagnostics {
        eprint!("{}", diagnostic.render(&file));
    }
    ExitCode::FAILURE
}
