use starkc::diag::Severity;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::{parse_package_graph, parse_with_options, ParseMode};
use starkc::resolve::{resolve, resolve_with_options};
use starkc::source::SourceFile;
use starkc::test_runner::{self, Outcome};
use starkc::typecheck;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "\
stark — package manager and builder for the STARK Core v1 language

Usage:
  stark check                   Check the current package and dependencies.
  stark build [--locked] [--offline] [--keep-generated] [--emit-rust] [--verbose]
                                 Compile a native debug executable.
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
  stark doc [--open] [--output <dir>]
                                 Generate API documentation for the current
                                 package's public items into <dir> (default:
                                 docs/). --open opens index.html afterward.
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

    if cmd == "doc" {
        return cmd_doc(&args[1..]);
    }

    if cmd == "build" {
        return cmd_build(&args[1..]);
    }

    if cmd != "check" && cmd != "run" {
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

    let root_package_name = graph.root_package_name.clone();
    let options = LanguageOptions::CORE;
    let analysis =
        starkc::analysis::analyze_project(starkc::analysis::ProjectInput::package(graph), options);
    let root_file = analysis.root_file.clone();
    let diagnostic_batch = analysis.diagnostic_batch(&HashMap::new());
    eprint!("{}", diagnostic_batch.render(&analysis.source_map));
    if analysis.has_errors() {
        eprintln!("{}: package compilation failed", root_package_name);
        return ExitCode::FAILURE;
    }
    if cmd == "check" {
        println!("{}: OK", root_package_name);
        return ExitCode::SUCCESS;
    }
    if cmd == "run" {
        let hir = analysis.hir.as_ref().expect("successful analysis has HIR");
        let tables = analysis
            .type_tables
            .as_ref()
            .expect("successful analysis has type tables");
        return match starkc::interp::run(hir, root_file.clone(), tables) {
            Ok(execution) => {
                print!("{}", execution.output);
                eprint!("{}", execution.stderr);
                ExitCode::from(execution.status)
            }
            Err(error) => {
                let mut diagnostic = starkc::diag::Diagnostic::error(
                    if error.is_trap {
                        format!("runtime error: {}", error.message)
                    } else {
                        format!("executable target error: {}", error.message)
                    },
                    error.span,
                );
                if !error.is_trap {
                    diagnostic.code = Some("E0214".to_string());
                }
                eprint!("{}", diagnostic.render(&root_file));
                ExitCode::from(if error.is_trap { 101 } else { 1 })
            }
        };
    }
    eprintln!("{}: package compilation failed", root_package_name);
    ExitCode::FAILURE
}

fn cmd_build(args: &[String]) -> ExitCode {
    let mut options = starkc::native_build::BuildCommandOptions::default();
    for arg in args {
        match arg.as_str() {
            "--locked" => options.locked = true,
            "--offline" => options.offline = true,
            "--keep-generated" => options.keep_generated = true,
            "--emit-rust" => {
                options.emit_rust = true;
                options.keep_generated = true;
            }
            "--verbose" => options.verbose = true,
            "--help" | "-h" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            _ => {
                eprint!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let current_dir = match std::env::current_dir() {
        Ok(dir) => dir,
        Err(error) => {
            eprintln!("error: failed to get current working directory: {error}");
            return ExitCode::FAILURE;
        }
    };
    match starkc::native_build::build_current_package(&current_dir, &options) {
        Ok(result) => {
            if options.verbose {
                println!(
                    "[stark build] package root: {}",
                    result.package_root.display()
                );
                println!("[stark build] package: {}", result.package_name);
                println!("[stark build] analysis: complete");
                println!("[stark build] MIR bodies: {}", result.mir_bodies);
                println!("[stark build] MIR verification: complete");
                println!(
                    "[stark build] rustc: {} ({})",
                    result.toolchain.rustc.display(),
                    result.toolchain.rustc_release
                );
                println!(
                    "[stark build] cargo: {} ({})",
                    result.toolchain.cargo.display(),
                    result.toolchain.cargo_release
                );
                println!("[stark build] host: {}", result.toolchain.host_triple);
                println!(
                    "[stark build] runtime: {}",
                    result.toolchain.runtime_crate.display()
                );
                if let Some(path) = &result.generated_dir {
                    println!("[stark build] generated crate: {}", path.display());
                }
                println!(
                    "[stark build] backend binary: {}",
                    result.backend_artifact.display()
                );
                println!(
                    "[stark build] final artifact: {}",
                    result.artifact_path.display()
                );
            }
            if let Some(path) = result.generated_dir {
                println!("Generated crate -> {}", path.display());
            }
            if let Some(path) = result.generated_rust {
                println!("Generated Rust -> {}", path.display());
            }
            println!(
                "Built {} [debug] -> {}",
                result.package_name,
                result.artifact_path.display()
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            render_build_error(&error, options.verbose);
            ExitCode::FAILURE
        }
    }
}

fn render_build_error(error: &starkc::native_build::BuildCommandError, verbose: bool) {
    use starkc::native_build::BuildCommandError;
    use starkc::native_toolchain::ToolchainError;
    match error {
        BuildCommandError::Package(message) => eprintln!("error: {message}"),
        BuildCommandError::Analysis {
            rendered,
            package_name,
        } => {
            eprint!("{rendered}");
            eprintln!("{package_name}: package compilation failed");
        }
        BuildCommandError::Lowering(message) => {
            eprintln!("error: native build does not yet support this program: {message}")
        }
        BuildCommandError::MirVerification(detail) => {
            eprintln!("error: internal compiler error: generated MIR failed verification");
            if verbose {
                eprintln!("{detail}");
            }
        }
        BuildCommandError::Toolchain(ToolchainError::Missing {
            tool,
            attempted,
            detail,
        }) => {
            eprintln!("error: Rust toolchain component '{tool}' not found");
            eprintln!(
                "help: install a supported Rust toolchain or set STARK_RUSTC and STARK_CARGO"
            );
            if verbose {
                eprintln!("attempted {}: {detail}", attempted.display());
            }
        }
        BuildCommandError::Toolchain(ToolchainError::InvalidVersion { tool, output }) => {
            eprintln!("error: could not determine {tool} version");
            if verbose {
                eprintln!("probe output: {output}");
            }
        }
        BuildCommandError::Toolchain(ToolchainError::TooOld { found, required }) => {
            eprintln!("error: Rust compiler {found} is too old; STARK native builds require {required} or newer");
        }
        BuildCommandError::Toolchain(ToolchainError::RuntimeMissing { attempted }) => {
            eprintln!("error: STARK native runtime installation is missing");
            eprintln!("help: install stark-runtime with STARK or set STARK_RUNTIME_DIR");
            if verbose {
                for path in attempted {
                    eprintln!("attempted: {}", path.display());
                }
            }
        }
        BuildCommandError::UnsupportedNative(message) => {
            eprintln!("error: native build does not yet support this program: {message}")
        }
        BuildCommandError::BackendBuild(error) => {
            let failure = &error.failure;
            eprintln!(
                "error: the STARK native backend generated a crate that Cargo could not build"
            );
            eprintln!(
                "note: generated crate retained at {}",
                failure.build_dir.display()
            );
            if verbose {
                eprintln!(
                    "rustc: {} ({})",
                    error.toolchain.rustc.display(),
                    error.toolchain.rustc_release
                );
                eprintln!(
                    "cargo: {} ({})",
                    error.toolchain.cargo.display(),
                    error.toolchain.cargo_release
                );
                eprintln!("summary: {}", failure.summary);
                eprintln!("command: {}", failure.command.join(" "));
                eprintln!(
                    "exit status: {}",
                    failure
                        .status
                        .map_or_else(|| "not started".to_string(), |code| code.to_string())
                );
                if !failure.stdout.is_empty() {
                    eprintln!("--- Cargo stdout ---\n{}", failure.stdout);
                }
                if !failure.stderr.is_empty() {
                    eprintln!("--- Cargo stderr ---\n{}", failure.stderr);
                }
            }
        }
        BuildCommandError::ArtifactMissing(path) => eprintln!(
            "error: native backend artifact is missing at {}",
            path.display()
        ),
        BuildCommandError::ArtifactInstall { from, to, detail } => eprintln!(
            "error: could not install native artifact from {} to {}: {detail}",
            from.display(),
            to.display()
        ),
        BuildCommandError::Io {
            action,
            path,
            detail,
        } => {
            if let Some(path) = path {
                eprintln!("error: {action} at {}: {detail}", path.display());
            } else {
                eprintln!("error: {action}: {detail}");
            }
        }
    }
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

fn cmd_doc(args: &[String]) -> ExitCode {
    let mut open = false;
    let mut output: Option<String> = None;
    let mut arguments = args.iter();
    while let Some(arg) = arguments.next() {
        match arg.as_str() {
            "--open" => open = true,
            "--output" if output.is_none() => match arguments.next() {
                Some(value) => output = Some(value.clone()),
                None => {
                    eprint!("{USAGE}");
                    return ExitCode::from(2);
                }
            },
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
    let package_root = manifest_path
        .parent()
        .expect("manifest path has a parent directory")
        .to_path_buf();

    let graph = match PackageGraph::load_from_root_with_modes(&manifest_path, false, false) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let package_name = graph.root_package_name.clone();

    let mut files = Vec::new();
    collect_stark_files(&package_root, &mut files);
    files.sort();
    if files.is_empty() {
        eprintln!(
            "Error: no `.stark` files found under {}",
            package_root.display()
        );
        return ExitCode::FAILURE;
    }

    let options = LanguageOptions::CORE;
    let mut all_items = Vec::new();
    let mut all_failed_examples: Vec<(String, String)> = Vec::new();
    let mut had_errors = false;
    for file_path in &files {
        let src = match std::fs::read_to_string(file_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: cannot read '{}': {}", file_path.display(), e);
                had_errors = true;
                continue;
            }
        };
        let source = SourceFile::new(file_path.to_string_lossy().into_owned(), src.clone());
        let (ast, diagnostics) = parse_with_options(&source, ParseMode::Program, options);
        if diagnostics.iter().any(|d| d.severity == Severity::Error) {
            for diag in &diagnostics {
                eprint!("{}", diag.render(&source));
            }
            eprintln!("Error: {}: parse failed", file_path.display());
            had_errors = true;
            continue;
        }
        let (_, comments, _) = starkc::lexer::tokenize_with_comments(&source);
        let items = starkc::doc_gen::extract::extract(&ast, &source, &comments);
        // Validate this file's examples with its own source in scope: an
        // example commonly calls the very item it documents (the plan's
        // own `assert_eq(add(2, 3), 5)` on `fn add`), so it must see that
        // file's other definitions, not compile in isolation.
        let examples = starkc::doc_gen::extract::collect_examples(&items);
        all_failed_examples.extend(starkc::doc_gen::validate_examples(&examples, &src));
        all_items.extend(items);
    }
    if had_errors {
        eprintln!("Error: doc generation aborted: one or more files failed to parse");
        return ExitCode::FAILURE;
    }

    let output_dir = package_root.join(output.unwrap_or_else(|| "docs".to_string()));
    let items_documented =
        match starkc::doc_gen::generate_from_items(&all_items, &package_name, &output_dir) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("Error: failed to write documentation site: {e}");
                return ExitCode::FAILURE;
            }
        };

    println!(
        "{}: generated docs for {} item(s) into {}",
        package_name,
        items_documented,
        output_dir.display()
    );

    if !all_failed_examples.is_empty() {
        eprintln!(
            "Error: {} doc example(s) failed:",
            all_failed_examples.len()
        );
        for (owner, message) in &all_failed_examples {
            eprintln!("  {owner}: {message}");
        }
        return ExitCode::FAILURE;
    }

    if open {
        let index_path = output_dir.join("index.html");
        if let Err(e) = open_in_browser(&index_path) {
            eprintln!("Warning: could not open browser: {e}");
        }
    }

    ExitCode::SUCCESS
}

fn open_in_browser(path: &Path) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(path).status()?;
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open").arg(path).status()?;
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", ""])
            .arg(path)
            .status()?;
    }
    Ok(())
}
