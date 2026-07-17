use starkc::diag::Severity;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::parse_package_graph;
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

const USAGE: &str = "\
stark — package manager and builder for the STARK Core v1 language

Usage:
  stark check                   Check the current package and dependencies.
  stark build                   Compile the current package and dependencies.
  stark run                     Compile and execute the package main entry point.
  stark test                    Compile and execute package tests.
  stark fmt [--check] [<file.stark>]
                                 Format the current package, or a single file.
                                 --check reports non-canonical files without
                                 modifying them (exit 1 if any differ).
  stark --help                  Show this help.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = match args.first() {
        Some(c) => c.as_str(),
        None => {
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    if cmd == "--help" || cmd == "-h" {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    if cmd == "fmt" {
        return cmd_fmt(&args[1..]);
    }

    if cmd != "check" && cmd != "build" && cmd != "run" && cmd != "test" {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    }

    let mut locked = false;
    let mut offline = false;
    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "--locked" => locked = true,
            "--offline" => offline = true,
            _ => {
                eprint!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }

    let current_dir = match std::env::current_dir() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("Error: failed to get current working directory: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let manifest_path = match find_package_root(&current_dir) {
        Ok(path) => path,
        Err(e) => {
            eprintln!("Error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let graph = match PackageGraph::load_from_root_with_modes(&manifest_path, locked, offline) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let options = LanguageOptions::CORE;
    let (ast, mut diags) = parse_package_graph(&graph, options);

    let root_pkg = graph.packages.get(&graph.root_package_name).unwrap();
    let entry_src = match std::fs::read_to_string(&root_pkg.entry) {
        Ok(src) => src,
        Err(e) => {
            eprintln!(
                "Error: failed to read entry file '{}': {}",
                root_pkg.entry.display(),
                e
            );
            return ExitCode::FAILURE;
        }
    };
    let root_file = Arc::new(SourceFile::new(
        root_pkg.entry.to_string_lossy().into_owned(),
        entry_src,
    ));

    if diags.iter().all(|d| d.severity != Severity::Error) {
        let (hir, mut resolution) = resolve(&ast, root_file.clone());
        diags.append(&mut resolution);

        if diags.iter().all(|d| d.severity != Severity::Error) {
            let checked = typecheck::analyze_with_options(&hir, root_file.clone(), options);
            diags.extend(checked.diagnostics);

            for diag in &diags {
                eprint!("{}", diag.render(&root_file));
            }

            let has_errors = diags.iter().any(|d| d.severity == Severity::Error);
            if has_errors {
                eprintln!("{}: package compilation failed", root_pkg.name);
                return ExitCode::FAILURE;
            }

            if cmd == "check" || cmd == "build" {
                println!("{}: OK", root_pkg.name);
                return ExitCode::SUCCESS;
            }

            if cmd == "test" {
                println!("{}: test OK", root_pkg.name);
                return ExitCode::SUCCESS;
            }

            if cmd == "run" {
                return match starkc::interp::run(&hir, root_file.clone(), &checked.tables) {
                    Ok(execution) => {
                        print!("{}", execution.output);
                        ExitCode::SUCCESS
                    }
                    Err(error) => {
                        let diagnostic = starkc::diag::Diagnostic::error(
                            format!("runtime error: {}", error.message),
                            error.span,
                        );
                        eprint!("{}", diagnostic.render(&root_file));
                        ExitCode::FAILURE
                    }
                };
            }
        }
    }

    for diag in &diags {
        eprint!("{}", diag.render(&root_file));
    }
    eprintln!("{}: package compilation failed", root_pkg.name);
    ExitCode::FAILURE
}

fn cmd_fmt(args: &[String]) -> ExitCode {
    let mut check = false;
    let mut path: Option<String> = None;
    for arg in args {
        match arg.as_str() {
            "--check" => check = true,
            value if !value.starts_with('-') && path.is_none() => path = Some(value.to_string()),
            _ => {
                eprint!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }

    let files: Vec<PathBuf> = match path {
        Some(p) => vec![PathBuf::from(p)],
        None => {
            let current_dir = match std::env::current_dir() {
                Ok(dir) => dir,
                Err(e) => {
                    eprintln!("Error: failed to get current working directory: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            let manifest_path = match find_package_root(&current_dir) {
                Ok(path) => path,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            let package_root = manifest_path
                .parent()
                .expect("manifest path has a parent directory")
                .to_path_buf();
            let mut found = Vec::new();
            collect_stark_files(&package_root, &mut found);
            found.sort();
            found
        }
    };

    if files.is_empty() {
        eprintln!("Error: no `.stark` files found");
        return ExitCode::FAILURE;
    }

    let mut any_non_canonical = false;
    let mut any_error = false;

    for file_path in &files {
        let src = match std::fs::read_to_string(file_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: cannot read '{}': {}", file_path.display(), e);
                any_error = true;
                continue;
            }
        };
        let source = SourceFile::new(file_path.to_string_lossy().into_owned(), src.clone());
        let formatted = match starkc::formatter::format_file(&source, LanguageOptions::CORE) {
            Ok(f) => f,
            Err(diags) => {
                for diag in &diags {
                    eprint!("{}", diag.render(&source));
                }
                eprintln!("Error: {}: formatting failed", file_path.display());
                any_error = true;
                continue;
            }
        };

        if formatted == src {
            continue;
        }

        if check {
            println!("{}: not formatted", file_path.display());
            any_non_canonical = true;
        } else if let Err(e) = std::fs::write(file_path, &formatted) {
            eprintln!("Error: cannot write '{}': {}", file_path.display(), e);
            any_error = true;
        } else {
            println!("{}: formatted", file_path.display());
        }
    }

    if any_error {
        ExitCode::FAILURE
    } else if check && any_non_canonical {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn collect_stark_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(name, "target" | "node_modules" | ".git") {
                continue;
            }
            collect_stark_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("stark") {
            out.push(path);
        }
    }
}
