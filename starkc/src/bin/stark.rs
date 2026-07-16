use std::process::ExitCode;
use std::sync::Arc;
use starkc::diag::Severity;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::parse_package_graph;
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;

const USAGE: &str = "\
stark — package manager and builder for the STARK Core v1 language

Usage:
  stark check                   Check the current package and dependencies.
  stark build                   Compile the current package and dependencies.
  stark run                     Compile and execute the package main entry point.
  stark test                    Compile and execute package tests.
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

    if cmd != "check" && cmd != "build" && cmd != "run" && cmd != "test" {
        eprint!("{USAGE}");
        return ExitCode::from(2);
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

    let graph = match PackageGraph::load_from_root(&manifest_path) {
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
            eprintln!("Error: failed to read entry file '{}': {}", root_pkg.entry.display(), e);
            return ExitCode::FAILURE;
        }
    };
    let root_file = Arc::new(SourceFile::new(root_pkg.entry.to_string_lossy().into_owned(), entry_src));

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
