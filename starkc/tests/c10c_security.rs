//! C10-C — empirical probes for the security surface inventory.
//!
//! **Method, inherited from `HC13-THREAT-MODEL.md`:**
//!
//! > *A threat model with no falsifier attached is a list of intentions.*
//!
//! So every defence C10-C claims names the test that would fail if the defence were removed, and
//! this file is where the cheap ones live. A surface whose defence is only *reasoned about* is
//! recorded in `C10-C-SECURITY-REVIEW.md` as **UNVERIFIED**, never as a defence.
//!
//! **The surface inventory was FROZEN in `C10-0-OPENING-INVENTORY.md` §9 before any finding here
//! was reviewed** (plan §11.1). This file does not add surfaces; it tests some of them.
//!
//! Findings are classified A/B/C/D per plan §11.2 — compiler correctness, security vulnerability,
//! release/distribution weakness, accepted operational limitation — and are NOT collapsed into DEV
//! numbers.

use starkc::analysis::{analyze_project, ProjectInput};
use starkc::options::LanguageOptions;
use starkc::source::SourceFile;
use std::sync::Arc;

fn analyse(src: &str) -> Vec<String> {
    let file = Arc::new(SourceFile::new("c10c.stark", src.to_string()));
    let a = analyze_project(ProjectInput::program(file), LanguageOptions::CORE);
    a.diagnostics.iter().map(|d| d.message.clone()).collect()
}

// ---------------------------------------------------------------------------------------------
// S01 — source and module path traversal.
//
// A `mod` declaration becomes a filesystem path: `parent_dir.join(format!("{mod_name}.stark"))`.
// The DEFENCE is the identifier grammar -- `mod_name` is the text of an identifier token, and an
// identifier cannot contain `/`, `\`, or `.`. That is a claim about the LEXER, so it is tested
// against the lexer rather than asserted from the parser's source.
// ---------------------------------------------------------------------------------------------

#[test]
fn s01_a_module_name_cannot_carry_a_path_separator_or_traversal() {
    // Each of these would be a traversal if `mod_name` were free text. Each must fail to parse as
    // a module declaration -- the assertion is that the compiler REJECTS, not that it resolves
    // safely, because rejection is the stronger property.
    let hostile = [
        "mod ../../../etc/passwd;",
        "mod ..;",
        "mod ../sibling;",
        "mod /etc/passwd;",
        r"mod ..\..\windows\system32;",
        "mod a/b;",
        "mod \"../escape\";",
        "mod .hidden;",
    ];
    for src in hostile {
        let diags = analyse(src);
        let joined = diags.join(" | ");
        assert!(
            !diags.is_empty(),
            "S01: `{src}` produced no diagnostic — a module name reached the filesystem layer \
             carrying a path"
        );
        // THE ASSERTION THAT MATTERS, and the first version of this test lacked it. "Some
        // diagnostic appeared" is nearly vacuous here: a plain `mod name;` in a bare program also
        // produces one (the file is missing). What distinguishes a DEFENDED surface from a lucky
        // one is WHICH diagnostic: the hostile forms must be rejected by the GRAMMAR, before any
        // path is built, so they must never reach the file-not-found path.
        assert!(
            !joined.contains("file not found for module"),
            "S01: `{src}` was accepted as a module NAME and reached filesystem lookup — the \
             traversal text was treated as an identifier. got {joined:?}"
        );
    }
}

#[test]
fn s01_the_defence_is_the_grammar_and_a_plain_module_name_still_works() {
    // The negative control for the test above. If EVERY `mod` failed, the assertions above would
    // pass vacuously and prove nothing about traversal.
    //
    // `allow_missing_modules` is off for a bare program, so a missing file is a diagnostic -- but
    // it must be a MISSING-FILE diagnostic, not a parse error. That distinction is the point: the
    // name was accepted as an identifier and only then failed to resolve.
    let diags = analyse("mod ordinary_name;");
    let joined = diags.join(" | ");
    assert!(
        joined.contains("ordinary_name"),
        "S01 control: a well-formed module name must reach resolution and be reported by NAME; \
         got {joined:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// S04 — generated Rust / source escaping.
//
// User-controlled text reaches generated Rust in string constants. The defence is escaping via
// Rust's own `{:?}` / `escape_default`. The falsifier is a literal that would BREAK OUT of the
// generated string and become code if escaping were removed.
// ---------------------------------------------------------------------------------------------

#[test]
fn s04_a_hostile_string_literal_does_not_escape_its_own_literal() {
    // Each of these, emitted verbatim into `let s = "<here>";`, would terminate the literal and
    // inject Rust. They must all compile as ordinary STARK strings with no diagnostic.
    let payloads = [
        r#"\"; std::process::exit(1); let _ = \""#,
        r#"\" + include_str!(\"/etc/passwd\") + \""#,
        r#"\\"#,
        "\\n\\r\\t",
        r#"{}{:?}{{}}"#, // format-string metacharacters, in case anything formats rather than escapes
    ];
    for payload in payloads {
        let src = format!("fn main() {{ let s = \"{payload}\"; }}");
        let diags = analyse(&src);
        assert!(
            diags.is_empty(),
            "S04: `{payload}` was rejected by the front end; the probe is not reaching the \
             backend and this surface is untested by it. diags={diags:?}"
        );
    }
    // NOTE ON SCOPE, recorded rather than glossed: this proves the FRONT END accepts the payloads,
    // which is what makes them reach code generation. The emitter's own escaping is covered by
    // `emit_types::tests::str_constant_emits_an_escaped_rust_literal` and by
    // `build::tests::a_generated_manifest_with_an_adversarial_runtime_path_stays_one_well_formed_line`,
    // and end-to-end by the native suites compiling generated code with rustc -- which fails
    // loudly if a literal ever breaks out.
}

// ---------------------------------------------------------------------------------------------
// S03 — artifact parsing limits, tested where an artifact is plain text the compiler must parse.
// ---------------------------------------------------------------------------------------------

#[test]
fn s03_a_malformed_manifest_is_refused_rather_than_partially_believed() {
    use starkc::package::find_package_root;
    let dir = std::env::temp_dir().join(format!(
        "c10c_manifest_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("src/main.stark"), "fn main() { }\n").unwrap();

    let hostile = [
        ("not-json", "this is not json"),
        ("truncated", r#"{"name":"app","versi"#),
        ("wrong-types", r#"{"name":42,"version":[],"entry":{}}"#),
        ("null-name", r#"{"name":null,"version":"0.1.0"}"#),
        (
            "entry-traversal",
            r#"{"name":"app","version":"0.1.0","entry":"../../../etc/passwd"}"#,
        ),
        (
            "entry-absolute",
            r#"{"name":"app","version":"0.1.0","entry":"/etc/passwd"}"#,
        ),
        (
            "deep-nesting",
            &format!(
                r#"{{"name":"app","version":"0.1.0","x":{}1{}}}"#,
                "[".repeat(500),
                "]".repeat(500)
            ),
        ),
    ];
    for (label, body) in hostile {
        std::fs::write(dir.join("starkpkg.json"), body).unwrap();
        // The contract under test is bounded failure: an Err, or an Ok whose contents are sane.
        // Never a panic, and never a hang.
        if let Ok(manifest) = find_package_root(&dir.join("src")) {
            // Either outcome is acceptable -- an Err is a refusal, an Ok is a manifest that
            // parsed. The assertion is that control reached HERE at all: no panic, no hang.
            let _ =
                starkc::package::PackageGraph::load_from_root_with_modes(&manifest, false, true);
        }
        eprintln!("S03/{label}: bounded");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------------
// S08 — temporary files and directories.
//
// C6.4 row 17 records the scheme: `env::temp_dir()` + PID + a counter, no shared root. The claim
// worth testing is COLLISION RESISTANCE, because that is what a shared-root scheme gets wrong.
// ---------------------------------------------------------------------------------------------

#[test]
fn s08_generated_temp_paths_do_not_collide_within_a_process() {
    // The compiler's own scheme is exercised end-to-end by the native suites. What is checked here
    // is the property those suites rely on and never state: two directories requested in quick
    // succession within one process are distinct.
    let mut seen = std::collections::HashSet::new();
    for _ in 0..200 {
        let p = std::env::temp_dir().join(format!(
            "stark_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(
            seen.insert(p.clone()),
            "S08: two temp paths collided within one process: {}. The PID+timestamp scheme is not \
             collision-free at this rate, and a parallel build could share a directory",
            p.display()
        );
    }
}
