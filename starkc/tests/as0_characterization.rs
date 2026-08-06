//! AS0 — entry-point characterization matrix.
//!
//! **This test does not say which behaviour is right.** It records what each production entry point
//! *currently does* with the same inputs, as a committed baseline. AS2 consolidates these
//! assemblies onto one driver, and consolidation necessarily picks a winner: without this baseline,
//! "the entry points now agree" is satisfied equally well by every one of them having silently
//! changed. Divergences here are findings for AS2 to resolve consciously, not failures.
//!
//! The exact set of production assemblies is recorded in
//! `STARKLANG/docs/compiler/audits/AS0-BASELINE-AND-INVENTORY.md` §3: eleven, six of which bypass
//! the driver.
//!
//! Regenerate after a deliberate change:
//!
//! ```text
//! STARK_UPDATE_CHARACTERIZATION=1 cargo test --test as0_characterization
//! ```
//!
//! Committing a regenerated baseline is an assertion that every diff in it was intended.

use std::path::{Path, PathBuf};
use std::process::Command;

const BASELINE: &str = "tests/as0-characterization/BASELINE.txt";

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

const VALID_MAIN: &str = "mod helper;\n\nfn main() {\n    let value: Int32 = helper::seven();\n}\n";
const VALID_HELPER: &str = "pub fn seven() -> Int32 {\n    7\n}\n";
/// Two errors in a known textual order, so diagnostic ORDER is part of what is pinned.
const INVALID_MAIN: &str =
    "mod helper;\n\nfn main() {\n    let a: Int32 = undefined_one();\n    let b: Bool = 5;\n}\n";
/// The error lives in the DEPENDENCY, which is what proves source attribution and provenance.
const INVALID_HELPER: &str = "pub fn seven() -> Int32 {\n    undefined_two()\n}\n";

fn write(path: &Path, contents: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, contents).unwrap();
}

/// A single package at `root`.
fn stage_package(root: &Path, name: &str, main: &str, helper: &str) {
    write(
        &root.join("starkpkg.json"),
        &format!(r#"{{"name":"{name}","version":"0.1.0","entry":"src/main.stark"}}"#),
    );
    write(&root.join("src").join("main.stark"), main);
    write(&root.join("src").join("helper.stark"), helper);
}

/// `app` depending on a sibling `lib`, because dependency paths must stay siblings.
fn stage_workspace(base: &Path, lib_main: &str) -> PathBuf {
    let lib = base.join("lib");
    write(
        &lib.join("starkpkg.json"),
        r#"{"name":"lib","version":"0.1.0","entry":"src/lib.stark","dependencies":{}}"#,
    );
    write(&lib.join("src").join("lib.stark"), lib_main);

    let app = base.join("app");
    write(
        &app.join("starkpkg.json"),
        r#"{"name":"app","version":"0.1.0","entry":"src/main.stark","dependencies":{"lib":{"package":"lib","path":"../lib","version":"0.1.0"}}}"#,
    );
    write(
        &app.join("src").join("main.stark"),
        "fn main() {\n    let value: Int32 = lib::seven();\n}\n",
    );
    app
}

fn unique_base(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let base = std::env::temp_dir().join(format!("as0_char_{tag}_{}_{nanos}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

// ---------------------------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------------------------

/// Everything about a run that AS2 must not change by accident.
struct Observation {
    status: String,
    stdout: String,
    stderr: String,
}

fn run(bin: &str, args: &[&str], cwd: &Path, scrub: &[&Path]) -> Observation {
    let output = Command::new(bin)
        .args(args)
        .current_dir(cwd)
        // Keep the child's own diagnostics deterministic and locale-independent.
        .env("NO_COLOR", "1")
        .output()
        .unwrap_or_else(|e| panic!("failed to run {bin} {args:?}: {e}"));

    Observation {
        status: match output.status.code() {
            Some(code) => format!("exit {code}"),
            None => "signalled".to_string(),
        },
        stdout: normalise(&String::from_utf8_lossy(&output.stdout), scrub),
        stderr: normalise(&String::from_utf8_lossy(&output.stderr), scrub),
    }
}

/// Path separators inside an already-scrubbed `<TMP>` path, forced to `/`.
///
/// **Found by the Windows lane, not locally.** Scrubbing the temp directory prefix is not enough:
/// what follows it is still a real path, and on Windows that means `<TMP>\single_ok.stark` against
/// a baseline recorded on macOS as `<TMP>/single_ok.stark`. Four rows differed, all of them
/// single-file ones.
///
/// That difference is AS0 finding D4 showing up in CI — package sources are named logically with
/// `/`, single-file sources carry the platform's own separator, and that split is deliberate. The
/// baseline should therefore record one canonical spelling rather than the host's.
///
/// Only separators *within* a scrubbed path are touched, and only up to the next whitespace, so a
/// backslash anywhere else in the output — an escape in a diagnostic, say — is left alone and stays
/// visible as behaviour.
fn normalise_scrubbed_separators(text: &str) -> String {
    const MARK: &str = "<TMP>";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(MARK) {
        out.push_str(&rest[..at + MARK.len()]);
        rest = &rest[at + MARK.len()..];
        let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
        out.push_str(&rest[..end].replace('\\', "/"));
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// Replace anything that varies between machines or runs. What SURVIVES normalisation is the
/// behaviour being pinned — so a surviving absolute path is itself a finding.
fn normalise(text: &str, scrub: &[&Path]) -> String {
    let mut out = text.replace("\r\n", "\n");
    // Longest paths first, so a base directory does not mask the package directories inside it.
    let mut paths: Vec<String> = scrub
        .iter()
        .flat_map(|p| {
            let raw = p.to_string_lossy().into_owned();
            let canonical = p
                .canonicalize()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_else(|_| raw.clone());
            [canonical, raw]
        })
        .collect();
    paths.sort_by_key(|p| std::cmp::Reverse(p.len()));
    for path in paths {
        out = out.replace(&path, "<TMP>");
    }
    out = normalise_scrubbed_separators(&out);
    // Timings and durations.
    let mut cleaned = String::new();
    for line in out.lines() {
        let line = if line.contains("finished in") || line.contains("elapsed") {
            let cut = line.find("finished in").or_else(|| line.find("elapsed"));
            match cut {
                Some(i) => format!("{}<TIMING>", &line[..i]),
                None => line.to_string(),
            }
        } else {
            line.to_string()
        };
        cleaned.push_str(line.trim_end());
        cleaned.push('\n');
    }
    cleaned
}

fn section(report: &mut String, assembly: &str, row: &str, obs: &Observation) {
    report.push_str(&format!("=== {assembly} | {row}\n"));
    report.push_str(&format!("--- status\n{}\n", obs.status));
    for (label, body) in [("stdout", &obs.stdout), ("stderr", &obs.stderr)] {
        let body = body.trim_end();
        if body.is_empty() {
            report.push_str(&format!("--- {label}\n(empty)\n"));
        } else {
            report.push_str(&format!("--- {label}\n{body}\n"));
        }
    }
    report.push('\n');
}

fn not_applicable(report: &mut String, assembly: &str, row: &str, why: &str) {
    report.push_str(&format!(
        "=== {assembly} | {row}\nNOT-APPLICABLE: {why}\n\n"
    ));
}

// ---------------------------------------------------------------------------------------------
// The matrix
// ---------------------------------------------------------------------------------------------

#[test]
fn entry_point_characterization_matrix() {
    let stark = env!("CARGO_BIN_EXE_stark");
    let starkc = env!("CARGO_BIN_EXE_starkc");

    let mut report = String::new();
    report.push_str(
        "AS0 entry-point characterization matrix\n\
         Generated by tests/as0_characterization.rs. Records CURRENT behaviour, not intended\n\
         behaviour. AS2 must resolve each divergence consciously. Absolute paths are scrubbed to\n\
         <TMP>; a surviving absolute path is a finding.\n\n",
    );

    // --- Package entry points: stark check / run / test -----------------------------------
    let base = unique_base("pkg");
    let valid = base.join("valid");
    let invalid_root = base.join("invalid_root");
    stage_package(&valid, "probe", VALID_MAIN, VALID_HELPER);
    stage_package(&invalid_root, "probe", INVALID_MAIN, VALID_HELPER);
    let invalid_dep_app = stage_workspace(&base.join("dep"), INVALID_HELPER);
    let scrub: Vec<&Path> = vec![&valid, &invalid_root, &invalid_dep_app, &base];

    for (cmd, assembly) in [
        ("check", "stark check [bypasses driver]"),
        ("run", "stark run [bypasses driver]"),
        ("test", "stark test [bypasses driver]"),
    ] {
        section(
            &mut report,
            assembly,
            "valid package",
            &run(stark, &[cmd], &valid, &scrub),
        );
        section(
            &mut report,
            assembly,
            "invalid root source",
            &run(stark, &[cmd], &invalid_root, &scrub),
        );
        section(
            &mut report,
            assembly,
            "invalid dependency source",
            &run(stark, &[cmd], &invalid_dep_app, &scrub),
        );
    }

    // --- Single-file entry points: starkc check / run --------------------------------------
    let single_ok = base.join("single_ok.stark");
    let single_bad = base.join("single_bad.stark");
    write(&single_ok, "fn main() {\n    let value: Int32 = 7;\n}\n");
    write(
        &single_bad,
        "fn main() {\n    let a: Int32 = undefined_one();\n    let b: Bool = 5;\n}\n",
    );

    for (cmd, assembly) in [
        ("check", "starkc check [uses driver]"),
        ("run", "starkc run [bypasses driver]"),
    ] {
        section(
            &mut report,
            assembly,
            "valid single file",
            &run(starkc, &[cmd, single_ok.to_str().unwrap()], &base, &scrub),
        );
        section(
            &mut report,
            assembly,
            "invalid single file",
            &run(starkc, &[cmd, single_bad.to_str().unwrap()], &base, &scrub),
        );
        not_applicable(
            &mut report,
            assembly,
            "package rows",
            "this entry point takes a file, not a package",
        );
    }

    // --- Assemblies not characterised here, with the reason ---------------------------------
    not_applicable(
        &mut report,
        "stark build [uses driver]",
        "all rows",
        "requires a host rustc toolchain and a native link; characterised by native_build_cli",
    );
    not_applicable(
        &mut report,
        "starkide [bypasses driver]",
        "all rows",
        "interactive terminal UI driven by stty and ANSI sequences; no non-interactive surface",
    );
    not_applicable(
        &mut report,
        "LSP package analysis [uses driver]",
        "all rows",
        "stdio JSON-RPC session; its diagnostics path is covered by the lsp suites",
    );
    not_applicable(
        &mut report,
        "deploy / doc_gen [use driver]",
        "all rows",
        "single-file driver callers with specialised inputs; no package assembly of their own",
    );
    not_applicable(
        &mut report,
        "ONNX signature verification [bypasses driver]",
        "all rows",
        "PARTIAL assembly: resolve without typecheck, over a tensor-extension declaration only",
    );

    let _ = std::fs::remove_dir_all(&base);

    // --- Compare or update ------------------------------------------------------------------
    let baseline_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BASELINE);
    if std::env::var("STARK_UPDATE_CHARACTERIZATION").is_ok() {
        std::fs::create_dir_all(baseline_path.parent().unwrap()).unwrap();
        std::fs::write(&baseline_path, &report).unwrap();
        eprintln!("baseline updated: {}", baseline_path.display());
        return;
    }

    let expected = std::fs::read_to_string(&baseline_path).unwrap_or_else(|e| {
        panic!(
            "missing characterization baseline at {}: {e}\n\
             regenerate with STARK_UPDATE_CHARACTERIZATION=1",
            baseline_path.display()
        )
    });

    if expected.replace("\r\n", "\n") != report {
        let mut diff = String::new();
        let expected_lines: Vec<&str> = expected.lines().collect();
        let actual_lines: Vec<&str> = report.lines().collect();
        for i in 0..expected_lines.len().max(actual_lines.len()) {
            let e = expected_lines.get(i).copied().unwrap_or("<missing>");
            let a = actual_lines.get(i).copied().unwrap_or("<missing>");
            if e != a {
                diff.push_str(&format!(
                    "line {}\n  baseline {e:?}\n  actual   {a:?}\n",
                    i + 1
                ));
                if diff.lines().count() > 60 {
                    diff.push_str("...\n");
                    break;
                }
            }
        }
        panic!(
            "entry-point behaviour drifted from the AS0 baseline.\n\n{diff}\n\
             If the change was intended, regenerate with STARK_UPDATE_CHARACTERIZATION=1 and say\n\
             in the commit which entry point changed and why."
        );
    }
}
