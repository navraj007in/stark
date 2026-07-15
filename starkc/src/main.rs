use starkc::ast_dump;
use starkc::lexer::tokenize;
use starkc::options::{options_from_extension_flags, LanguageOptions};
use starkc::parser::{parse_with_options, ParseMode};
use starkc::source::SourceFile;
use std::process::ExitCode;

const USAGE: &str = "\
starkc — compiler for the STARK Core v1 language

Usage:
  starkc check [--snippet] [--extension <name>] <file.stark>
                              Check a source file and report semantic diagnostics.
  starkc run <file.stark>               Check and execute a Core program.
  starkc parse [--snippet] [--dump] [--extension <name>] <file.stark>
                              Parse a source file and report diagnostics.
                              --snippet parses the harness block-body form
                              (items + statements) instead of Program.
                              --dump prints the AST on success.
                              --extension <name> enables an optional language
                              extension (Gate 4+): tensor.
  starkc lex <file.stark>     Dump the token stream (debugging aid)
  starkc import <model.onnx> --out <model.stark> [--force]
                              Generate a deterministic STARK model declaration.
  starkc verify <model.onnx> --declaration <model.stark> [--model <Name>]
                              Verify an artifact against a model declaration.
  starkc --help               Show this help
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((cmd, rest)) if cmd == "parse" => {
            let mut mode = ParseMode::Program;
            let mut dump = false;
            let mut path = None;
            let mut extensions = Vec::new();
            let mut args = rest.iter();
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--snippet" => mode = ParseMode::Snippet,
                    "--dump" => dump = true,
                    "--extension" => match args.next() {
                        Some(name) => extensions.push(name.clone()),
                        None => {
                            eprint!("{USAGE}");
                            return ExitCode::from(2);
                        }
                    },
                    _ if path.is_none() => path = Some(arg.clone()),
                    _ => {
                        eprint!("{USAGE}");
                        return ExitCode::from(2);
                    }
                }
            }
            let (Some(path), Some(options)) = (path, extension_options(&extensions)) else {
                eprint!("{USAGE}");
                return ExitCode::from(2);
            };
            cmd_parse(&path, mode, dump, options)
        }
        Some((cmd, rest)) if cmd == "check" => {
            let mut mode = ParseMode::Program;
            let mut path = None;
            let mut extensions = Vec::new();
            let mut args = rest.iter();
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--snippet" => mode = ParseMode::Snippet,
                    "--extension" => match args.next() {
                        Some(name) => extensions.push(name.clone()),
                        None => {
                            eprint!("{USAGE}");
                            return ExitCode::from(2);
                        }
                    },
                    _ if path.is_none() => path = Some(arg.clone()),
                    _ => {
                        eprint!("{USAGE}");
                        return ExitCode::from(2);
                    }
                }
            }
            let (Some(path), Some(options)) = (path, extension_options(&extensions)) else {
                eprint!("{USAGE}");
                return ExitCode::from(2);
            };
            cmd_check(&path, mode, options)
        }
        Some((cmd, rest)) if cmd == "import" => cmd_import(rest),
        Some((cmd, rest)) if cmd == "verify" => cmd_verify(rest),
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

fn cmd_import(args: &[String]) -> ExitCode {
    let mut input = None;
    let mut output = None;
    let mut force = false;
    let mut arguments = args.iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--out" if output.is_none() => {
                let Some(value) = arguments.next().filter(|value| !value.starts_with('-')) else {
                    eprint!("{USAGE}");
                    return ExitCode::from(2);
                };
                output = Some(value.clone());
            }
            "--force" if !force => force = true,
            value if !value.starts_with('-') && input.is_none() => input = Some(value.to_string()),
            _ => {
                eprint!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(input), Some(output)) = (input, output) else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    match starkc::onnx::import_file(
        std::path::Path::new(&input),
        std::path::Path::new(&output),
        force,
    ) {
        Ok(_) => {
            println!("{}: imported ONNX declaration", output);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_verify(args: &[String]) -> ExitCode {
    let mut artifact = None;
    let mut declaration = None;
    let mut model = None;
    let mut arguments = args.iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--declaration" if declaration.is_none() => {
                let Some(value) = arguments.next().filter(|value| !value.starts_with('-')) else {
                    eprint!("{USAGE}");
                    return ExitCode::from(2);
                };
                declaration = Some(value.clone());
            }
            "--model" if model.is_none() => {
                let Some(value) = arguments.next().filter(|value| !value.starts_with('-')) else {
                    eprint!("{USAGE}");
                    return ExitCode::from(2);
                };
                model = Some(value.clone());
            }
            value if !value.starts_with('-') && artifact.is_none() => {
                artifact = Some(value.to_string());
            }
            _ => {
                eprint!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(artifact), Some(declaration)) = (artifact, declaration) else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    match starkc::onnx::verify_declaration_file(
        std::path::Path::new(&artifact),
        std::path::Path::new(&declaration),
        model.as_deref(),
    ) {
        Ok(report) if report.is_match() => {
            println!("{}: ONNX signature matches", declaration);
            ExitCode::SUCCESS
        }
        Ok(report) => {
            eprintln!("Error: ONNX signature mismatch");
            for difference in report.differences {
                eprintln!("  - {difference}");
            }
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Build [`LanguageOptions`] from collected `--extension` values, printing a
/// precise usage error and returning `None` if any id is unknown or duplicated.
fn extension_options(names: &[String]) -> Option<LanguageOptions> {
    match options_from_extension_flags(names) {
        Ok(options) => Some(options),
        Err(err) => {
            eprintln!("Error: {err}");
            None
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

fn cmd_parse(path: &str, mode: ParseMode, dump: bool, options: LanguageOptions) -> ExitCode {
    let file = match load(path) {
        Ok(f) => f,
        Err(code) => return code,
    };
    let (tree, diags) = parse_with_options(&file, mode, options);
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

fn cmd_check(path: &str, mode: ParseMode, options: LanguageOptions) -> ExitCode {
    let file = match load(path) {
        Ok(f) => f,
        Err(code) => return code,
    };
    let (tree, mut diags) = parse_with_options(&file, mode, options);
    if !diags.is_empty() {
        for diag in &diags {
            eprint!("{}", diag.render(&file));
        }
        eprintln!("{}: {} error(s)", file.name, diags.len());
        return ExitCode::FAILURE;
    }

    let file_arc = std::sync::Arc::new(file);
    let (hir, mut sem_diags) =
        starkc::resolve::resolve_with_options(&tree, file_arc.clone(), options);
    diags.append(&mut sem_diags);

    if diags.is_empty() {
        let mut type_diags = starkc::typecheck::check_with_options(&hir, file_arc.clone(), options);
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
    let (tree, mut diagnostics) =
        parse_with_options(&file, ParseMode::Program, LanguageOptions::CORE);
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
