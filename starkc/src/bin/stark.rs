use starkc::diag::Severity;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::{parse_package_graph, parse_with_options, ParseMode};
use starkc::resolve::{resolve, resolve_with_options};
use starkc::source::SourceFile;
use starkc::test_runner::{self, Outcome};
use starkc::typecheck;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "\
stark — package manager and builder for the STARK Core v1 language

Usage:
  stark check                   Check the current package and dependencies.
  stark build                   Compile the current package and dependencies.
  stark run                     Compile and execute the package main entry point.
  stark test [name] [--ignored] [--show-output]
                                 Run `fn test_*()` functions in the package,
                                 tests/*.stark integration programs, and
                                 examples/*.stark. [name] filters by
                                 substring. --ignored also runs
                                 `test_ignored_*` functions (skipped by
                                 default). --show-output prints captured
                                 stdout even for passing tests.
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

    if cmd == "test" {
        return cmd_test(&args[1..]);
    }

    if cmd != "check" && cmd != "build" && cmd != "run" {
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

    if any_error || (check && any_non_canonical) {
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

fn cmd_test(args: &[String]) -> ExitCode {
    let mut name_filter: Option<String> = None;
    let mut run_ignored = false;
    let mut show_output = false;
    for arg in args {
        match arg.as_str() {
            "--ignored" => run_ignored = true,
            "--show-output" => show_output = true,
            value if !value.starts_with('-') && name_filter.is_none() => {
                name_filter = Some(value.to_string())
            }
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
    let graph = match PackageGraph::load_from_root_with_modes(&manifest_path, false, false) {
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

    let mut overall_failed = false;

    if diags.iter().any(|d| d.severity == Severity::Error) {
        for diag in &diags {
            eprint!("{}", diag.render(&root_file));
        }
        eprintln!("{}: package compilation failed", root_pkg.name);
        return ExitCode::FAILURE;
    }
    let (hir, mut resolution) = resolve(&ast, root_file.clone());
    diags.append(&mut resolution);
    if diags.iter().any(|d| d.severity == Severity::Error) {
        for diag in &diags {
            eprint!("{}", diag.render(&root_file));
        }
        eprintln!("{}: package compilation failed", root_pkg.name);
        return ExitCode::FAILURE;
    }
    let checked = typecheck::analyze_with_options(&hir, root_file.clone(), options);
    if checked
        .diagnostics
        .iter()
        .any(|d| d.severity == Severity::Error)
    {
        for diag in checked.diagnostics.iter().chain(diags.iter()) {
            eprint!("{}", diag.render(&root_file));
        }
        eprintln!("{}: package compilation failed", root_pkg.name);
        return ExitCode::FAILURE;
    }

    // ---- unit tests: fn test_*() discovered in the package's own module tree ----
    let all_tests = test_runner::discover_tests(&hir, &root_file);
    let selected = test_runner::filter_by_name(&all_tests, name_filter.as_deref());

    println!("running {} tests", selected.len());
    println!();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut ignored = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new();
    let start_all = Instant::now();
    for test in &selected {
        if test.ignored && !run_ignored {
            println!("test {} ... ignored", test.name);
            ignored += 1;
            continue;
        }
        let result = test_runner::run_test(&hir, root_file.clone(), &checked.tables, test);
        match &result.outcome {
            Outcome::Passed => {
                passed += 1;
                let ms = result.duration.as_millis();
                let timing = if ms > 10 {
                    format!(" ({ms}ms)")
                } else {
                    String::new()
                };
                println!("test {} ... ok{timing}", test.name);
                if show_output && !result.output.is_empty() {
                    println!("---- {} stdout ----", test.name);
                    print!("{}", result.output);
                }
            }
            Outcome::Failed { message } => {
                failed += 1;
                println!("test {} ... FAILED", test.name);
                failures.push((test.name.clone(), message.clone()));
            }
            Outcome::Ignored => {
                ignored += 1;
                println!("test {} ... ignored", test.name);
            }
        }
    }
    let total_ms = start_all.elapsed().as_millis();

    if !failures.is_empty() {
        println!();
        println!("failures:");
        println!();
        for (name, message) in &failures {
            println!("---- {name} ----");
            println!("{message}");
            println!();
        }
        println!("failures:");
        for (name, _) in &failures {
            println!("    {name}");
        }
    }

    println!();
    println!(
        "test result: {}. {passed} passed; {failed} failed; {ignored} ignored; {total_ms}ms total",
        if failed == 0 { "ok" } else { "FAILED" }
    );
    if failed > 0 {
        overall_failed = true;
    }

    // ---- integration tests: tests/*.stark, each a standalone program ----
    let package_root = manifest_path
        .parent()
        .expect("manifest path has a parent directory")
        .to_path_buf();
    if let Some(more_failed) = run_standalone_suite(&package_root.join("tests"), "test", options) {
        overall_failed = overall_failed || more_failed;
    }

    // ---- examples: examples/*.stark, each compiled and run ----
    if let Some(more_failed) =
        run_standalone_suite(&package_root.join("examples"), "example", options)
    {
        overall_failed = overall_failed || more_failed;
    }

    if overall_failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Run every `.stark` file under `dir` as a standalone program (its own
/// `fn main()`). Returns `None` if `dir` doesn't exist or is empty (nothing
/// to report), else `Some(any_failed)`.
fn run_standalone_suite(dir: &Path, label: &str, options: LanguageOptions) -> Option<bool> {
    if !dir.is_dir() {
        return None;
    }
    let mut files = Vec::new();
    collect_stark_files(dir, &mut files);
    files.sort();
    if files.is_empty() {
        return None;
    }

    println!();
    println!("running {} {label}s", files.len());
    println!();
    let mut passed = 0usize;
    let mut failed = 0usize;
    for file_path in &files {
        let display_name = file_path
            .strip_prefix(dir)
            .unwrap_or(file_path)
            .display()
            .to_string();
        match run_standalone_program(file_path, options) {
            Ok(()) => {
                passed += 1;
                println!("{label} {display_name} ... ok");
            }
            Err(msg) => {
                failed += 1;
                println!("{label} {display_name} ... FAILED");
                println!("  {msg}");
            }
        }
    }
    println!();
    println!(
        "{label} result: {}. {passed} passed; {failed} failed",
        if failed == 0 { "ok" } else { "FAILED" }
    );

    Some(failed > 0)
}

fn run_standalone_program(path: &Path, options: LanguageOptions) -> Result<(), String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("cannot read: {e}"))?;
    let file = SourceFile::new(path.to_string_lossy().into_owned(), src);
    let (tree, diagnostics) = parse_with_options(&file, ParseMode::Program, options);
    let file = Arc::new(file);
    if diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return Err(format!("{} parse error(s)", diagnostics.len()));
    }
    let (hir, mut resolution) = resolve_with_options(&tree, file.clone(), options);
    if resolution.iter().any(|d| d.severity == Severity::Error) {
        return Err(format!("{} resolve error(s)", resolution.len()));
    }
    resolution.clear();
    let checked = typecheck::analyze_with_options(&hir, file.clone(), options);
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    if !errors.is_empty() {
        return Err(format!("{} typecheck error(s)", errors.len()));
    }
    starkc::interp::run(&hir, file, &checked.tables)
        .map(|_| ())
        .map_err(|e| e.message)
}
