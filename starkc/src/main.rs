use starkc::ast_dump;
use starkc::lexer::tokenize;
use starkc::options::{options_from_extension_flags, LanguageOptions};
use starkc::parser::{parse_with_options, ParseMode};
use starkc::source::SourceFile;
use std::process::ExitCode;

const USAGE: &str = "\
starkc — compiler for the STARK Core v1 language

Usage:
  starkc check [--snippet] [--extension <name>] [--message-format <text|json>] [--stdin --filename <path>] [<file.stark>]
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
  starkc verify <model.onnx> --declaration <model.stark> [--model <Name>] [--message-format <text|json>]
                              Verify an artifact against a model declaration.
  starkc deploy <pipeline.stark> --model <model.onnx> --entry <fn> --out <dir> [--force]
                              Generate a native ONNX Runtime host crate for a
                              checked tensor inference pipeline (Gate 5).
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
            let mut message_format = "text";
            let mut stdin = false;
            let mut filename = None;
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
                    "--message-format" => match args.next() {
                        Some(value) => {
                            if value != "text" && value != "json" {
                                eprintln!("Error: invalid message format `{value}`; expected `text` or `json`");
                                return ExitCode::from(2);
                            }
                            if value == "json" {
                                message_format = "json";
                            } else {
                                message_format = "text";
                            }
                        }
                        None => {
                            eprint!("{USAGE}");
                            return ExitCode::from(2);
                        }
                    },
                    "--stdin" => stdin = true,
                    "--filename" => match args.next() {
                        Some(value) => filename = Some(value.clone()),
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
            let Some(options) = extension_options(&extensions) else {
                eprint!("{USAGE}");
                return ExitCode::from(2);
            };
            
            let file = if stdin {
                if path.is_some() {
                    eprintln!("Error: cannot specify input path when `--stdin` is passed");
                    return ExitCode::from(2);
                }
                let Some(fname) = filename else {
                    eprintln!("Error: `--filename <path>` is required when `--stdin` is passed");
                    return ExitCode::from(2);
                };
                
                use std::io::Read;
                let mut buffer = String::new();
                if let Err(err) = std::io::stdin().read_to_string(&mut buffer) {
                    eprintln!("Error: cannot read from stdin: {err}");
                    return ExitCode::from(3);
                }
                SourceFile::new(fname, buffer)
            } else {
                if filename.is_some() {
                    eprintln!("Error: `--filename` is only valid when `--stdin` is passed");
                    return ExitCode::from(2);
                }
                let Some(p) = path else {
                    eprintln!("Error: missing input file path");
                    eprint!("{USAGE}");
                    return ExitCode::from(2);
                };
                match load(&p) {
                    Ok(f) => f,
                    Err(code) => return code,
                }
            };
            
            cmd_check(file, mode, options, message_format)
        }
        Some((cmd, rest)) if cmd == "import" => cmd_import(rest),
        Some((cmd, rest)) if cmd == "verify" => cmd_verify(rest),
        Some((cmd, rest)) if cmd == "deploy" => cmd_deploy(rest),
        Some((cmd, rest)) if cmd == "run" => {
            let mut path = None;
            let mut extensions = Vec::new();
            let mut args = rest.iter();
            while let Some(arg) = args.next() {
                match arg.as_str() {
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
            let Some(p) = path else {
                eprintln!("Error: missing input file path");
                eprint!("{USAGE}");
                return ExitCode::from(2);
            };
            let Some(options) = extension_options(&extensions) else {
                eprint!("{USAGE}");
                return ExitCode::from(2);
            };
            cmd_run(&p, options)
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
    let mut message_format = "text";
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
            "--message-format" => {
                let Some(value) = arguments.next().filter(|value| !value.starts_with('-')) else {
                    eprint!("{USAGE}");
                    return ExitCode::from(2);
                };
                if value != "text" && value != "json" {
                    eprintln!("Error: invalid message format `{value}`; expected `text` or `json`");
                    return ExitCode::from(2);
                }
                if value == "json" {
                    message_format = "json";
                } else {
                    message_format = "text";
                }
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

    let sig_and_hash = match starkc::onnx::read_signature(std::path::Path::new(&artifact)) {
        Ok((sig, bytes)) => {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let digest = hasher.finalize();
            let mut hash = String::new();
            for &byte in digest.iter() {
                use std::fmt::Write;
                write!(&mut hash, "{:02x}", byte).unwrap();
            }
            Some((sig, hash))
        }
        Err(error) => {
            if message_format == "json" {
                println!(
                    "{{\
                      \"schema_version\":1,\
                      \"status\":\"error\",\
                      \"artifact\":{{\"path\":\"{}\",\"sha256\":null}},\
                      \"declaration\":{{\"path\":\"{}\",\"model\":null}},\
                      \"differences\":[]\
                    }}",
                    starkc::onnx::escape_json(&artifact),
                    starkc::onnx::escape_json(&declaration)
                );
            }
            eprintln!("Error: {error}");
            return ExitCode::from(3);
        }
    };

    let (signature, sha256) = sig_and_hash.unwrap();

    let source = match std::fs::read_to_string(&declaration) {
        Ok(src) => src,
        Err(error) => {
            if message_format == "json" {
                println!(
                    "{{\
                      \"schema_version\":1,\
                      \"status\":\"error\",\
                      \"artifact\":{{\"path\":\"{}\",\"sha256\":\"{}\"}},\
                      \"declaration\":{{\"path\":\"{}\",\"model\":null}},\
                      \"differences\":[]\
                    }}",
                    starkc::onnx::escape_json(&artifact),
                    starkc::onnx::escape_json(&sha256),
                    starkc::onnx::escape_json(&declaration)
                );
            }
            eprintln!("Error: cannot read declaration file `{declaration}`: {error}");
            return ExitCode::from(3);
        }
    };

    match starkc::onnx::verify_declaration_source(
        &signature,
        &source,
        &declaration,
        model.as_deref(),
    ) {
        Ok(report) => {
            if message_format == "json" {
                let status_str = if report.is_match() { "match" } else { "mismatch" };
                let model_str = model.clone().unwrap_or_else(|| {
                    find_first_model_name(&source).unwrap_or_else(|| "Unknown".to_string())
                });
                let diffs_json = report.differences.iter()
                    .map(|d| d.to_json())
                    .collect::<Vec<_>>()
                    .join(",");
                println!(
                    "{{\
                      \"schema_version\":1,\
                      \"status\":\"{status_str}\",\
                      \"artifact\":{{\"path\":\"{artifact_path}\",\"sha256\":\"{sha256}\"}},\
                      \"declaration\":{{\"path\":\"{dec_path}\",\"model\":\"{model_name}\"}},\
                      \"differences\":[{diffs_json}]\
                    }}",
                    artifact_path = starkc::onnx::escape_json(&artifact),
                    sha256 = starkc::onnx::escape_json(&sha256),
                    dec_path = starkc::onnx::escape_json(&declaration),
                    model_name = starkc::onnx::escape_json(&model_str),
                    diffs_json = diffs_json
                );
            } else {
                if report.is_match() {
                    println!("{}: ONNX signature matches", declaration);
                } else {
                    eprintln!("Error: ONNX signature mismatch");
                    for difference in &report.differences {
                        eprintln!("  - {difference}");
                    }
                }
            }
            if report.is_match() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            if message_format == "json" {
                println!(
                    "{{\
                      \"schema_version\":1,\
                      \"status\":\"error\",\
                      \"artifact\":{{\"path\":\"{}\",\"sha256\":\"{}\"}},\
                      \"declaration\":{{\"path\":\"{}\",\"model\":null}},\
                      \"differences\":[]\
                    }}",
                    starkc::onnx::escape_json(&artifact),
                    starkc::onnx::escape_json(&sha256),
                    starkc::onnx::escape_json(&declaration)
                );
            }
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_deploy(args: &[String]) -> ExitCode {
    let mut pipeline = None;
    let mut model = None;
    let mut entry = None;
    let mut out = None;
    let mut force = false;
    let mut arguments = args.iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model" if model.is_none() => match arguments.next() {
                Some(value) if !value.starts_with('-') => model = Some(value.clone()),
                _ => return usage_exit(),
            },
            "--entry" if entry.is_none() => match arguments.next() {
                Some(value) if !value.starts_with('-') => entry = Some(value.clone()),
                _ => return usage_exit(),
            },
            "--out" if out.is_none() => match arguments.next() {
                Some(value) if !value.starts_with('-') => out = Some(value.clone()),
                _ => return usage_exit(),
            },
            "--force" if !force => force = true,
            value if !value.starts_with('-') && pipeline.is_none() => {
                pipeline = Some(value.to_string())
            }
            _ => return usage_exit(),
        }
    }
    let (Some(pipeline), Some(model), Some(entry), Some(out)) = (pipeline, model, entry, out)
    else {
        return usage_exit();
    };

    match starkc::deploy::deploy(
        std::path::Path::new(&pipeline),
        std::path::Path::new(&model),
        &entry,
        std::path::Path::new(&out),
        force,
    ) {
        Ok(summary) => {
            println!("Generated deployment host: {}", summary.out_dir.display());
            println!("  model SHA-256: {}", summary.model_sha256);
            println!("  entry:         {}", summary.entry);
            println!(
                "  next:          cargo build --release --locked --manifest-path {}/Cargo.toml",
                summary.out_dir.display()
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprint!("{}", error.render());
            ExitCode::FAILURE
        }
    }
}

fn usage_exit() -> ExitCode {
    eprint!("{USAGE}");
    ExitCode::from(2)
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

fn cmd_check(file: SourceFile, mode: ParseMode, options: LanguageOptions, message_format: &str) -> ExitCode {
    let mut diags = Vec::new();
    let (tree, parse_diags) = parse_with_options(&file, mode, options);
    diags.extend(parse_diags);

    let file_arc = std::sync::Arc::new(file);
    if diags.iter().all(|d| d.severity != starkc::diag::Severity::Error) {
        let (hir, sem_diags) = starkc::resolve::resolve_with_options(&tree, file_arc.clone(), options);
        diags.extend(sem_diags);

        if diags.iter().all(|d| d.severity != starkc::diag::Severity::Error) {
            let type_diags = starkc::typecheck::check_with_options(&hir, file_arc.clone(), options);
            diags.extend(type_diags);
        }
    }

    if message_format == "json" {
        print_json_diagnostics(&file_arc.name, &diags);
    } else {
        for diag in &diags {
            eprint!("{}", diag.render(&file_arc));
        }
    }

    let error_count = diags
        .iter()
        .filter(|diag| diag.severity == starkc::diag::Severity::Error)
        .count();
    if error_count > 0 {
        if message_format != "json" {
            eprintln!("{}: {} error(s)", file_arc.name, error_count);
        }
        ExitCode::FAILURE
    } else {
        if message_format != "json" {
            println!("{}: OK", file_arc.name);
        }
        ExitCode::SUCCESS
    }
}

fn print_json_diagnostics(file_name: &str, diagnostics: &[starkc::diag::Diagnostic]) {
    let mut diags_json = Vec::new();
    for diag in diagnostics {
        let severity_str = match diag.severity {
            starkc::diag::Severity::Error => "error",
            starkc::diag::Severity::Warning => "warning",
        };
        let code_str = match &diag.code {
            Some(c) => format!("\"code\":\"{}\"", starkc::onnx::escape_json(c)),
            None => "\"code\":null".to_string(),
        };
        
        let label_json = if !diag.label.is_empty() {
            format!(
                "[{{\"message\":\"{}\",\"file\":\"{}\",\"range\":{{\"startByte\":{},\"endByte\":{}}}}}]",
                starkc::onnx::escape_json(&diag.label),
                starkc::onnx::escape_json(file_name),
                diag.span.lo,
                diag.span.hi
            )
        } else {
            "[]".to_string()
        };

        let notes_json = diag.notes.iter()
            .map(|n| format!("\"{}\"", starkc::onnx::escape_json(n)))
            .collect::<Vec<_>>()
            .join(",");

        let help_str = if diag.helps.is_empty() {
            "\"help\":null".to_string()
        } else {
            format!("\"help\":\"{}\"", starkc::onnx::escape_json(&diag.helps.join("\n")))
        };

        diags_json.push(format!(
            "{{\
              \"severity\":\"{severity_str}\",\
              {code_str},\
              \"message\":\"{message}\",\
              \"file\":\"{file}\",\
              \"range\":{{\"startByte\":{start},\"endByte\":{end}}},\
              \"labels\":{labels},\
              \"notes\":[{notes}],\
              {help}\
            }}",
            message = starkc::onnx::escape_json(&diag.message),
            file = starkc::onnx::escape_json(file_name),
            start = diag.span.lo,
            end = diag.span.hi,
            labels = label_json,
            notes = notes_json,
            help = help_str
        ));
    }

    println!(
        "{{\
          \"schemaVersion\":1,\
          \"tool\":\"starkc\",\
          \"toolVersion\":\"{}\",\
          \"diagnostics\":[{}]\
        }}",
        env!("CARGO_PKG_VERSION"),
        diags_json.join(",")
    );
}

fn find_first_model_name(source: &str) -> Option<String> {
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("model ") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                let name = parts[1];
                let clean_name = name.split('<').next().unwrap_or(name);
                return Some(clean_name.to_string());
            }
        }
    }
    None
}

fn cmd_run(path: &str, options: LanguageOptions) -> ExitCode {
    let file = match load(path) {
        Ok(file) => file,
        Err(code) => return code,
    };
    let (tree, mut diagnostics) =
        parse_with_options(&file, ParseMode::Program, options);
    let file = std::sync::Arc::new(file);
    if diagnostics.is_empty() {
        let (hir, mut resolution) = starkc::resolve::resolve_with_options(&tree, file.clone(), options);
        diagnostics.append(&mut resolution);
        if diagnostics.is_empty() {
            let checked = starkc::typecheck::analyze_with_options(&hir, file.clone(), options);
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
