use sha2::{Digest, Sha256};
use starkc::diag::Severity;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::{parse_with_options, ParseMode};
use starkc::session::CompilerSession;
use starkc::source::SourceFile;
use starkc::source_extensions::is_stark_source;
use starkc::test_runner::{self, Outcome};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "\
stark — package manager and builder for the STARK Core v1 language

Usage:
  stark check [--target-native]
                                 Check the current package and dependencies.
                                 Native-surface gaps warn by default; --target-native
                                 makes them errors.
  stark build [--release] [--target <triple>] [--no-build-cache] [--no-mir-opt]
              [--locked] [--offline] [--keep-generated] [--emit-rust] [--verbose]
                                 Compile a native executable. Debug by default;
                                 --release builds the optimised profile with the
                                 same STARK-observable semantics. --target names
                                 a triple; cross-compilation is validated but not
                                 yet supported, and is refused with its reason.
                                 --no-build-cache deletes the generated crate
                                 afterwards, which is the qualification path.
                                 --no-mir-opt compiles MIR exactly as lowered, so
                                 a suspected optimiser defect can be bisected
                                 against the unoptimised program.
  stark cache status | clean     Report or clear the bounded build cache. It
                                 reuses whole content-addressed generated crates
                                 and their Cargo artefacts; it is NOT fine-grained
                                 incremental compilation.
  stark run                     Compile and execute the package main entry point.
  stark test [name] [--ignored] [--show-output]
                                 Run `fn test_*()` functions in the package,
                                 tests/*.stark|*.st integration programs, and
                                 examples/*.stark|*.st. [name] filters by
                                 substring. --ignored also runs
                                 `test_ignored_*` functions (skipped by
                                 default). --show-output prints captured
                                 stdout even for passing tests.
  stark fmt [--check] [<file.stark|file.st>]
                                 Format the current package, or a single file.
                                 --check reports non-canonical files without
                                 modifying them (exit 1 if any differ).
  stark doc [--open] [--output <dir>]
                                 Generate API documentation for the current
                                 package's public items into <dir> (default:
                                 docs/). --open opens index.html afterward.
  stark doctor [--root <dir>] [--json]
                                 Inspect the installed toolchain layout and
                                 verify manifest-listed files.
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

    if cmd == "cache" {
        return cmd_cache(&args[1..]);
    }

    if cmd == "doctor" {
        return cmd_doctor(&args[1..]);
    }

    if cmd != "check" && cmd != "run" {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    }

    let mut locked = false;
    let mut offline = false;
    let mut target_native = false;
    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "--locked" => locked = true,
            "--offline" => offline = true,
            "--target-native" if cmd == "check" => target_native = true,
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

    let overlays = match starkc::native_build::provider_overlays_for_analysis(&graph) {
        Ok(overlays) => overlays,
        Err(e) => {
            eprintln!("Error: {:?}", e);
            return ExitCode::FAILURE;
        }
    };
    // AS2: parse → resolve → typecheck through the ONE pipeline. This command used to assemble it
    // by hand, decide for itself what counted as failure, and render diagnostics in its own order.
    // Command-line parsing above and presentation below stay here; the pipeline does not.
    let package_name = graph.root_package_name.clone();
    let session = if overlays.is_empty() {
        CompilerSession::for_package(graph, options)
    } else {
        CompilerSession::for_package_with_overlays(graph, overlays, options)
    };
    let program = match session.check() {
        Ok(program) => program,
        Err(failure) => {
            eprint!("{}", failure.render());
            eprintln!("{package_name}: package compilation failed");
            return ExitCode::FAILURE;
        }
    };

    for diagnostic in program.diagnostics() {
        eprint!("{}", diagnostic.render(program.sources()));
    }

    if cmd == "check" {
        if let Ok(mir) = program.lower_mir() {
            let exclusions =
                starkc::backend::generated_rust::emit_runtime::exclusions_in_program(&mir);
            for (_, span, message) in &exclusions {
                let diagnostic = if target_native {
                    starkc::diag::Diagnostic::error(message.clone(), *span).with_code("E0106")
                } else {
                    starkc::diag::Diagnostic::warning(message.clone(), *span).with_code("W0106")
                };
                eprint!("{}", diagnostic.render(program.sources()));
            }
            if target_native && !exclusions.is_empty() {
                eprintln!("{package_name}: native-target check failed");
                return ExitCode::FAILURE;
            }
        }
        println!("{package_name}: OK");
        return ExitCode::SUCCESS;
    }
    if cmd == "run" {
        return match program.execute_hir() {
            Ok(execution) => {
                print!("{}", execution.output);
                eprint!("{}", execution.stderr);
                ExitCode::from(execution.status)
            }
            Err(error) => {
                // WP-C7.9 Packet F: three outcomes, three renderings and three statuses. A
                // host/process resource limit (`LIMIT-RESOURCE-001`) is neither a language trap nor
                // a compiler rejection: the program was valid and the machine ran out, so it must
                // not be reported with a trap category and must not exit 101, which is the status
                // TRAP-ABORT-001 reserves for traps.
                let (headline, code, status) = match error.class {
                    starkc::interp::FailureClass::Trap => {
                        (format!("runtime error: {}", error.message), None, 101u8)
                    }
                    starkc::interp::FailureClass::Entry => (
                        format!("executable target error: {}", error.message),
                        Some("E0214"),
                        1,
                    ),
                    starkc::interp::FailureClass::HostResource => (
                        format!("resource limit reached: {}", error.message),
                        None,
                        2,
                    ),
                    starkc::interp::FailureClass::InternalInvariant => (
                        format!("internal compiler error: {}", error.message),
                        None,
                        70,
                    ),
                };
                let mut diagnostic = starkc::diag::Diagnostic::error(headline, error.span);
                diagnostic.code = code.map(str::to_string);
                // **Rendered against the file the trap was RAISED in, not the entry file.**
                //
                // This renderer indexed every span against the root, so a fault inside a dependency
                // was reported at a line number in the CONSUMER's source. `line_col` clamps an
                // out-of-range offset to end-of-file, so the result was not a visible error but a
                // plausible, wrong location: a 21-line consumer was told the fault was at line 31
                // of itself. It cost real time on the `String::bytes()` defect (CD-305) — the first
                // characterisation described the consumer's use of a match binding, and the
                // reproducer built from it passed while the fault sat three frames away in
                // `stark-mime`.
                //
                // DEV-113-B fixed it by stamping the raising file onto the error. AS1b-ii-d deleted
                // the stamp: `error.span` names the source, and the program's registry resolves it.
                eprint!("{}", diagnostic.render(program.sources()));
                ExitCode::from(status)
            }
        };
    }
    eprintln!("{package_name}: package compilation failed");
    ExitCode::FAILURE
}

#[derive(Debug)]
struct ManifestFile {
    path: String,
    size: u64,
    sha256: String,
}

#[derive(Debug)]
struct InstallManifest {
    version: String,
    target: String,
    files: Vec<ManifestFile>,
}

#[derive(Debug)]
struct DoctorCheck {
    name: String,
    ok: bool,
    detail: String,
}

/// Where the runtime crate sits in an installed tree, mirror layout first.
///
/// The mirror layout puts it under `starkc/`, matching the repository; a package built before that
/// move puts it flat. Both are installable and both are found by
/// `native_toolchain::discover_runtime`, so doctor accepts either — checking only the flat path
/// reported `runtime: fail` against a mirror-layout installation that builds programs correctly.
const RUNTIME_LAYOUTS: &[&str] = &[
    "lib/stark/starkc/stark-runtime/Cargo.toml",
    "lib/stark/stark-runtime/Cargo.toml",
];

/// Where the provider ABI crate sits, in the same two layouts and the same order.
const PROVIDER_ABI_LAYOUTS: &[&str] = &[
    "lib/stark/starkc/stark-provider-abi/Cargo.toml",
    "lib/stark/stark-provider-abi/Cargo.toml",
];

/// The first candidate that exists under `root`, reported by name.
///
/// On failure the detail names the FIRST candidate rather than listing all of them: that is the
/// layout a current package writes, so the message points at where the file should have been
/// rather than at wherever the search happened to end.
fn layout_check(name: &str, root: &Path, candidates: &[&str]) -> DoctorCheck {
    let found = candidates
        .iter()
        .map(|relative| root.join(relative))
        .find(|path| path.is_file());
    DoctorCheck {
        name: name.to_string(),
        ok: found.is_some(),
        detail: found
            .unwrap_or_else(|| root.join(candidates[0]))
            .display()
            .to_string(),
    }
}

fn cmd_doctor(args: &[String]) -> ExitCode {
    let mut explicit_root = None;
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => {
                json = true;
                i += 1;
            }
            "--root" => {
                if i + 1 >= args.len() {
                    eprint!("{USAGE}");
                    return ExitCode::from(2);
                }
                explicit_root = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            _ => {
                eprint!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }

    let root = match explicit_root {
        Some(root) => root,
        None => match discover_install_root() {
            Some(root) => root,
            None => {
                let detail = "could not locate manifest.json beside the running stark executable";
                if json {
                    println!("{{\"ok\":false,\"error\":\"{}\"}}", json_escape(detail));
                } else {
                    eprintln!("STARK doctor: FAIL");
                    eprintln!("  install root: unresolved");
                    eprintln!("  manifest: {detail}");
                }
                return ExitCode::FAILURE;
            }
        },
    };
    let mut checks = Vec::new();
    let manifest_path = root.join("manifest.json");
    let manifest = match read_install_manifest(&manifest_path) {
        Ok(manifest) => {
            checks.push(DoctorCheck {
                name: "manifest".to_string(),
                ok: true,
                detail: manifest_path.display().to_string(),
            });
            Some(manifest)
        }
        Err(detail) => {
            checks.push(DoctorCheck {
                name: "manifest".to_string(),
                ok: false,
                detail,
            });
            None
        }
    };

    // **The executable name comes from the MANIFEST, not from this host.** `bin/stark` was
    // hardcoded, so on Windows -- where the payload is `bin\stark.exe` -- `install.ps1` ran
    // `stark.exe doctor --root ...` during staging, got a failure for a perfectly good package,
    // and threw "staged STARK installation failed manifest verification". Install-blocking, and
    // invisible on Unix.
    //
    // Reading the target also makes `doctor --root` work when INSPECTING a package built for
    // another platform, which is how this was reproduced without a Windows machine.
    let windows_payload = manifest
        .as_ref()
        .map(|m| m.target.contains("windows"))
        .unwrap_or(cfg!(windows));
    let stark_binary = if windows_payload {
        "bin/stark.exe"
    } else {
        "bin/stark"
    };
    checks.push({
        let path = root.join(stark_binary);
        DoctorCheck {
            name: "bin".to_string(),
            ok: path.is_file(),
            detail: path.display().to_string(),
        }
    });

    for (name, candidates) in [
        ("runtime", RUNTIME_LAYOUTS),
        ("provider_abi", PROVIDER_ABI_LAYOUTS),
    ] {
        checks.push(layout_check(name, &root, candidates));
    }

    if let Some(manifest) = &manifest {
        let mut verified = 0usize;
        for file in &manifest.files {
            let path = root.join(&file.path);
            match verify_manifest_file(&path, file) {
                Ok(()) => verified += 1,
                Err(detail) => checks.push(DoctorCheck {
                    name: format!("file:{}", file.path),
                    ok: false,
                    detail,
                }),
            }
        }
        checks.push(DoctorCheck {
            name: "manifest_files".to_string(),
            ok: verified == manifest.files.len(),
            detail: format!("{verified}/{} files verified", manifest.files.len()),
        });
    }

    let ok = checks.iter().all(|check| check.ok);
    if json {
        print_doctor_json(&root, manifest.as_ref(), &checks, ok);
    } else {
        print_doctor_human(&root, manifest.as_ref(), &checks, ok);
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn discover_install_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let bin = exe.parent()?;
    // Two shapes, in order: an extracted tarball run in place (`<root>/bin/stark`), and an
    // installed prefix whose `bin/` is a symlink farm into `lib/stark/current`. The manifest is
    // what identifies a root, so probing for it distinguishes them without either path needing to
    // know which one it is.
    [
        bin.parent().map(Path::to_path_buf),
        Some(bin.join("../lib/stark/current")),
    ]
    .into_iter()
    .flatten()
    .find(|candidate| candidate.join("manifest.json").is_file())
}

/// A minimal, complete JSON reader for the install manifest.
///
/// # Why this exists rather than `serde_json`
///
/// `starkc` has **three** dependencies (`sha2` and two path crates). Pulling in `serde` +
/// `serde_json` + their proc-macro chain to read one file is a supply-chain decision, not a code
/// decision, and this project pins its crypto dependencies exactly because it takes that seriously
/// (CD-361). So the dependency question is the owner's; the *defect* is fixed here either way.
///
/// # What was wrong with what this replaces
///
/// The previous reader searched for `"key"` anywhere in the document and split the file array on
/// the literal `"\n    {"`. That made it depend on `build-release.py`'s exact pretty-printing:
/// a semantically identical manifest with different indentation, or compact output, parsed as
/// **zero files** — and "0/0 files verified" is a PASS. A reader whose failure mode is silently
/// verifying nothing is worse than one that errors.
///
/// It also could not see string escapes, so a path containing `\"` truncated the value, and
/// key lookup was unscoped: a value appearing earlier in the document could answer for a key.
///
/// This is a real recursive-descent parser: escapes (including `\uXXXX` with surrogate pairs),
/// nesting, and duplicate keys, which are rejected rather than last-wins.
mod json {
    use std::collections::BTreeMap;

    #[derive(Debug, Clone, PartialEq)]
    pub enum Json {
        Null,
        Bool(bool),
        Number(f64),
        String(String),
        Array(Vec<Json>),
        Object(BTreeMap<String, Json>),
    }

    impl Json {
        pub fn object(&self) -> Option<&BTreeMap<String, Json>> {
            match self {
                Json::Object(map) => Some(map),
                _ => None,
            }
        }
        pub fn array(&self) -> Option<&Vec<Json>> {
            match self {
                Json::Array(items) => Some(items),
                _ => None,
            }
        }
        pub fn string(&self) -> Option<&str> {
            match self {
                Json::String(text) => Some(text.as_str()),
                _ => None,
            }
        }
        /// A size is a count of bytes. Rejecting negative, fractional and out-of-range values here
        /// means the manifest cannot express a size the verifier would then compare against.
        pub fn u64(&self) -> Option<u64> {
            match self {
                Json::Number(n) if n.is_finite() && *n >= 0.0 && n.fract() == 0.0 => {
                    let rounded = *n as u64;
                    (rounded as f64 == *n).then_some(rounded)
                }
                _ => None,
            }
        }
    }

    pub fn parse(text: &str) -> Result<Json, String> {
        let bytes: Vec<char> = text.chars().collect();
        let mut parser = Parser { bytes, at: 0 };
        parser.skip_whitespace();
        let value = parser.value(0)?;
        parser.skip_whitespace();
        if parser.at != parser.bytes.len() {
            return Err(format!("trailing input at character {}", parser.at));
        }
        Ok(value)
    }

    struct Parser {
        bytes: Vec<char>,
        at: usize,
    }

    /// Bounded so a hostile manifest cannot exhaust the stack through nesting alone. The real
    /// document is two levels deep.
    const MAX_DEPTH: usize = 32;

    impl Parser {
        fn peek(&self) -> Option<char> {
            self.bytes.get(self.at).copied()
        }

        fn skip_whitespace(&mut self) {
            while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
                self.at += 1;
            }
        }

        fn expect(&mut self, ch: char) -> Result<(), String> {
            if self.peek() == Some(ch) {
                self.at += 1;
                Ok(())
            } else {
                Err(format!("expected `{ch}` at character {}", self.at))
            }
        }

        fn literal(&mut self, word: &str) -> Result<(), String> {
            for ch in word.chars() {
                self.expect(ch)?;
            }
            Ok(())
        }

        fn value(&mut self, depth: usize) -> Result<Json, String> {
            if depth > MAX_DEPTH {
                return Err("manifest nesting is too deep".to_string());
            }
            self.skip_whitespace();
            match self.peek() {
                Some('{') => self.object(depth),
                Some('[') => self.array(depth),
                Some('"') => Ok(Json::String(self.string()?)),
                Some('t') => self.literal("true").map(|()| Json::Bool(true)),
                Some('f') => self.literal("false").map(|()| Json::Bool(false)),
                Some('n') => self.literal("null").map(|()| Json::Null),
                Some(_) => self.number(),
                None => Err("unexpected end of manifest".to_string()),
            }
        }

        fn object(&mut self, depth: usize) -> Result<Json, String> {
            self.expect('{')?;
            let mut map = BTreeMap::new();
            self.skip_whitespace();
            if self.peek() == Some('}') {
                self.at += 1;
                return Ok(Json::Object(map));
            }
            loop {
                self.skip_whitespace();
                let key = self.string()?;
                self.skip_whitespace();
                self.expect(':')?;
                let value = self.value(depth + 1)?;
                // Last-wins would let a manifest carry two `sha256` values and leave which one is
                // checked up to the reader's implementation.
                if map.insert(key.clone(), value).is_some() {
                    return Err(format!("duplicate key `{key}` in manifest object"));
                }
                self.skip_whitespace();
                match self.peek() {
                    Some(',') => self.at += 1,
                    Some('}') => {
                        self.at += 1;
                        return Ok(Json::Object(map));
                    }
                    _ => return Err(format!("expected `,` or `}}` at character {}", self.at)),
                }
            }
        }

        fn array(&mut self, depth: usize) -> Result<Json, String> {
            self.expect('[')?;
            let mut items = Vec::new();
            self.skip_whitespace();
            if self.peek() == Some(']') {
                self.at += 1;
                return Ok(Json::Array(items));
            }
            loop {
                items.push(self.value(depth + 1)?);
                self.skip_whitespace();
                match self.peek() {
                    Some(',') => self.at += 1,
                    Some(']') => {
                        self.at += 1;
                        return Ok(Json::Array(items));
                    }
                    _ => return Err(format!("expected `,` or `]` at character {}", self.at)),
                }
            }
        }

        fn string(&mut self) -> Result<String, String> {
            self.expect('"')?;
            let mut out = String::new();
            loop {
                let ch = self
                    .peek()
                    .ok_or_else(|| "unterminated string in manifest".to_string())?;
                self.at += 1;
                match ch {
                    '"' => return Ok(out),
                    '\\' => {
                        let escape = self
                            .peek()
                            .ok_or_else(|| "unterminated escape in manifest".to_string())?;
                        self.at += 1;
                        match escape {
                            '"' => out.push('"'),
                            '\\' => out.push('\\'),
                            '/' => out.push('/'),
                            'b' => out.push('\u{8}'),
                            'f' => out.push('\u{c}'),
                            'n' => out.push('\n'),
                            'r' => out.push('\r'),
                            't' => out.push('\t'),
                            'u' => out.push(self.unicode_escape()?),
                            other => {
                                return Err(format!("unknown escape `\\{other}` in manifest"));
                            }
                        }
                    }
                    // A control character must be escaped; accepting a raw one would let a
                    // manifest path carry a newline and misalign anything that logs it.
                    c if (c as u32) < 0x20 => {
                        return Err("unescaped control character in manifest string".to_string());
                    }
                    c => out.push(c),
                }
            }
        }

        fn unicode_escape(&mut self) -> Result<char, String> {
            let first = self.hex4()?;
            // A surrogate is only meaningful as a PAIR; a lone one is not a character and
            // `from_u32` would reject it with a less useful message.
            if (0xD800..0xDC00).contains(&first) {
                self.expect('\\')?;
                self.expect('u')?;
                let second = self.hex4()?;
                if !(0xDC00..0xE000).contains(&second) {
                    return Err("unpaired surrogate escape in manifest".to_string());
                }
                let combined = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
                return char::from_u32(combined)
                    .ok_or_else(|| "invalid surrogate pair in manifest".to_string());
            }
            char::from_u32(first).ok_or_else(|| "invalid \\u escape in manifest".to_string())
        }

        fn hex4(&mut self) -> Result<u32, String> {
            let mut value = 0u32;
            for _ in 0..4 {
                let ch = self
                    .peek()
                    .ok_or_else(|| "truncated \\u escape in manifest".to_string())?;
                let digit = ch
                    .to_digit(16)
                    .ok_or_else(|| format!("`{ch}` is not a hex digit in a \\u escape"))?;
                value = value * 16 + digit;
                self.at += 1;
            }
            Ok(value)
        }

        fn number(&mut self) -> Result<Json, String> {
            let start = self.at;
            if self.peek() == Some('-') {
                self.at += 1;
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))
            {
                self.at += 1;
            }
            let text: String = self.bytes[start..self.at].iter().collect();
            text.parse::<f64>()
                .map(Json::Number)
                .map_err(|_| format!("`{text}` is not a number"))
        }
    }
}

use json::Json;

fn read_install_manifest(path: &Path) -> Result<InstallManifest, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let value = json::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let root = value.object().ok_or("manifest is not a JSON object")?;

    let version = root
        .get("stark_version")
        .and_then(Json::string)
        .ok_or("manifest is missing stark_version")?
        .to_string();
    let target = root
        .get("host_target")
        .and_then(Json::string)
        .ok_or("manifest is missing host_target")?
        .to_string();

    let entries = root
        .get("files")
        .and_then(Json::array)
        .ok_or("manifest is missing files")?;
    let mut files = Vec::with_capacity(entries.len());
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for entry in entries {
        let object = entry
            .object()
            .ok_or("manifest file entry is not an object")?;
        let file_path = object
            .get("path")
            .and_then(Json::string)
            .ok_or("manifest file entry is missing path")?
            .to_string();
        // **A manifest path decides what gets hashed, so it decides what gets TRUSTED.** A path
        // escaping the install root would let a manifest certify a file the package never
        // installed -- verifying, say, the system `/bin/sh` and reporting the installation sound.
        check_manifest_path(&file_path)?;
        // Case-INSENSITIVE, because Windows and macOS default filesystems are: two entries
        // differing only in case name one file there, and the second silently certifies whatever
        // the first wrote.
        if !seen.insert(file_path.to_lowercase()) {
            return Err(format!(
                "manifest lists {file_path} twice (case-insensitively); one entry would certify \
                 the other's bytes"
            ));
        }
        let sha256 = object
            .get("sha256")
            .and_then(Json::string)
            .ok_or_else(|| format!("manifest file entry {file_path} is missing sha256"))?
            .to_string();
        let size = object
            .get("size")
            .and_then(Json::u64)
            .ok_or_else(|| format!("manifest file entry {file_path} is missing size"))?;
        files.push(ManifestFile {
            path: file_path,
            size,
            sha256,
        });
    }
    Ok(InstallManifest {
        version,
        target,
        files,
    })
}

/// A manifest path must name something INSIDE the installation and nothing else.
fn check_manifest_path(relative: &str) -> Result<(), String> {
    if relative.is_empty() {
        return Err("manifest file entry has an empty path".to_string());
    }
    let candidate = Path::new(relative);
    if candidate.is_absolute() || relative.starts_with('/') || relative.starts_with('\\') {
        return Err(format!("manifest path {relative} is absolute"));
    }
    // Windows accepts `C:\..` and treats `\` as a separator; a check written only against `/`
    // would pass a path that escapes on the platform where the installer runs.
    if relative.contains(':') {
        return Err(format!("manifest path {relative} names a drive or stream"));
    }
    for component in relative.split(['/', '\\']) {
        if component == ".." {
            return Err(format!("manifest path {relative} escapes the install root"));
        }
    }
    if candidate.components().any(|c| {
        matches!(
            c,
            std::path::Component::ParentDir | std::path::Component::RootDir
        )
    }) {
        return Err(format!("manifest path {relative} escapes the install root"));
    }
    Ok(())
}

fn verify_manifest_file(path: &Path, file: &ManifestFile) -> Result<(), String> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("{}: missing or unreadable: {e}", path.display()))?;
    if metadata.len() != file.size {
        return Err(format!(
            "{}: size mismatch: expected {}, actual {}",
            path.display(),
            file.size,
            metadata.len()
        ));
    }
    let digest = sha256_path(path)?;
    if digest != file.sha256 {
        return Err(format!(
            "{}: sha256 mismatch: expected {}, actual {}",
            path.display(),
            file.sha256,
            digest
        ));
    }
    Ok(())
}

fn sha256_path(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(hex)
}

fn print_doctor_human(
    root: &Path,
    manifest: Option<&InstallManifest>,
    checks: &[DoctorCheck],
    ok: bool,
) {
    println!("STARK doctor: {}", if ok { "OK" } else { "FAIL" });
    println!("  install root: {}", root.display());
    if let Some(manifest) = manifest {
        println!("  version: {}", manifest.version);
        println!("  host target: {}", manifest.target);
    }
    for check in checks {
        println!(
            "  {}: {} ({})",
            check.name,
            if check.ok { "ok" } else { "fail" },
            check.detail
        );
    }
}

fn print_doctor_json(
    root: &Path,
    manifest: Option<&InstallManifest>,
    checks: &[DoctorCheck],
    ok: bool,
) {
    println!("{{");
    println!("  \"ok\": {ok},");
    println!(
        "  \"install_root\": \"{}\",",
        json_escape(&root.display().to_string())
    );
    if let Some(manifest) = manifest {
        println!("  \"version\": \"{}\",", json_escape(&manifest.version));
        println!("  \"host_target\": \"{}\",", json_escape(&manifest.target));
    }
    println!("  \"checks\": [");
    for (index, check) in checks.iter().enumerate() {
        let comma = if index + 1 == checks.len() { "" } else { "," };
        println!(
            "    {{\"name\":\"{}\",\"ok\":{},\"detail\":\"{}\"}}{}",
            json_escape(&check.name),
            check.ok,
            json_escape(&check.detail),
            comma
        );
    }
    println!("  ]");
    println!("}}");
}

/// AS5-f: delegates to the compiler's one escaping authority (DEV-184).
fn json_escape(input: &str) -> String {
    starkc::json::escape(input)
}

/// WP-C7.3. One `cache` command with two verbs rather than two top-level commands — the smallest
/// routing that covers §5.5's "provide a cache clear command" without widening the CLI surface.
fn cmd_cache(args: &[String]) -> ExitCode {
    let current_dir = match std::env::current_dir() {
        Ok(dir) => dir,
        Err(error) => {
            eprintln!("error: failed to get current working directory: {error}");
            return ExitCode::FAILURE;
        }
    };
    let manifest = match starkc::package::find_package_root(&current_dir) {
        Ok(path) => path,
        Err(message) => {
            eprintln!("error: {message}");
            return ExitCode::FAILURE;
        }
    };
    let Some(package_root) = manifest.parent() else {
        eprintln!("error: package manifest has no parent directory");
        return ExitCode::FAILURE;
    };
    let roots = [
        package_root.join("target/stark/debug"),
        package_root.join("target/stark/release"),
    ];
    match args.first().map(String::as_str) {
        Some("status") | None => {
            let mut total = 0u64;
            for root in &roots {
                let status = starkc::build_cache::status(root);
                if status.entries.is_empty() {
                    continue;
                }
                println!("{}", root.display());
                for entry in &status.entries {
                    let name = entry
                        .path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    println!(
                        "  {name}  {:.1} MB{}",
                        entry.bytes as f64 / 1e6,
                        if entry.pinned { "  (pinned)" } else { "" }
                    );
                }
                total += status.total_bytes();
            }
            println!(
                "{:.1} MB cached, cap {:.0} MB",
                total as f64 / 1e6,
                starkc::build_cache::DEFAULT_MAX_BYTES as f64 / 1e6
            );
            ExitCode::SUCCESS
        }
        Some("clean") => {
            let mut freed = 0u64;
            let mut removed = 0usize;
            for root in &roots {
                let report = starkc::build_cache::clean(root);
                freed += report.freed_bytes;
                removed += report.removed.len();
            }
            println!(
                "removed {removed} cache entries, freed {:.1} MB",
                freed as f64 / 1e6
            );
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("error: unknown `stark cache` verb `{other}` (expected `status` or `clean`)");
            ExitCode::from(2)
        }
    }
}

fn cmd_build(args: &[String]) -> ExitCode {
    let mut options = starkc::native_build::BuildCommandOptions::default();
    let mut pending_target = false;
    for arg in args {
        // `--target` takes a value, so the previous iteration may have claimed this argument.
        if pending_target {
            options.target = Some(arg.clone());
            pending_target = false;
            continue;
        }
        match arg.as_str() {
            "--release" => options.release = true,
            "--no-build-cache" => options.no_build_cache = true,
            "--no-mir-opt" => options.no_mir_opt = true,
            "--target" => pending_target = true,
            a if a.starts_with("--target=") => {
                options.target = Some(a["--target=".len()..].to_string());
            }
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
    if pending_target {
        eprintln!("error: `--target` requires a target triple");
        return ExitCode::from(2);
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
                match &result.mir_opt {
                    Some(stats) => println!(
                        "[stark build] MIR optimisation: {} changes ({} rvalues, {} checked \
                         folded, {} checked proven trapping, {} branches, {} constants, {} dead \
                         blocks)",
                        stats.total(),
                        stats.rvalues_folded,
                        stats.checked_folded,
                        stats.checked_trapped,
                        stats.branches_folded,
                        stats.constants_propagated,
                        stats.blocks_removed
                    ),
                    None => println!("[stark build] MIR optimisation: disabled (--no-mir-opt)"),
                }
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
                if let Some(path) = &result.backend_artifact {
                    println!("[stark build] backend binary: {}", path.display());
                }
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
                "Built {} [{}] -> {}",
                result.package_name,
                result.profile.as_str(),
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
        BuildCommandError::Target(message) => eprintln!("error: {message}"),
        // WP-C7.8 (CD-212, Packet 5): a capability requirement that cannot be satisfied is its own
        // failure class, reported on its own terms rather than as a missing feature, an
        // unsupported target, or a downstream linker error.
        BuildCommandError::Capability(message) => eprintln!("error: {message}"),
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
        // WP-C6.4a (§8.1): the diagnostic names the supported tier-1 targets, and separates
        // "STARK does not build for that machine" from "that machine's Rust toolchain is not
        // installed" -- the remedies are different, and neither is a program error.
        BuildCommandError::TargetRejected(error) => {
            eprintln!("error: {error}");
            match &error {
                starkc::target::TargetError::UnsupportedByStark { .. } => eprintln!(
                    "help: STARK qualifies {} for native builds",
                    starkc::target::tier1_triples().join(" and ")
                ),
                starkc::target::TargetError::SupportedButToolchainMissing { .. } => {
                    eprintln!("help: install the Rust toolchain support for that target")
                }
                starkc::target::TargetError::HostOrTargetMetadataMismatch { .. }
                | starkc::target::TargetError::LayoutContractMismatch { .. } => {
                    eprintln!("help: this is a compiler-internal target-metadata inconsistency")
                }
            }
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
        eprintln!("Error: no `.stark` or `.st` files found");
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
            Err(failure) => {
                for diag in failure.diagnostics() {
                    eprint!("{}", diag.render(failure.sources()));
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
        } else if is_stark_source(&path) {
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

    // **`provider_api` must be synthesized before the front end runs, here as in a native build.**
    //
    // Without it every generated `*_raw` function is E0200 and a package declaring `provider_api`
    // fails to compile before a single test is discovered — which is what `stark test` did to
    // `stark-io` and `stark-random`, both of which ship their own unit tests.
    //
    // The triple is derived from this build rather than probed from `rustc`: testing runs through
    // the interpreter and compiles nothing, so requiring a Rust toolchain to run a package's tests
    // would be a toolchain dependency bought for nothing.
    let overlays = match starkc::native_build::provider_overlays_for_analysis(&graph) {
        Ok(overlays) => overlays,
        Err(e) => {
            eprintln!("Error: {:?}", e);
            return ExitCode::FAILURE;
        }
    };
    // AS2: the ONE pipeline. The overlaid entry source — synthesis appends generated `provider_api`
    // items past the end of the on-disk text, and rendering a diagnostic in that region against the
    // on-disk file panics with "byte index N is out of bounds" — is now the session's concern
    // rather than something this command reconstructs.
    //
    // It also removes a latent options split: this command resolved with `resolve()`, which is
    // hard-wired to `LanguageOptions::CORE`, while typechecking with `options`. Both are CORE here,
    // so nothing observable changed, but the session threads ONE options value through every phase
    // and the split can no longer be reintroduced by editing one line.
    let package_name = graph.root_package_name.clone();
    let session = if overlays.is_empty() {
        CompilerSession::for_package(graph, options)
    } else {
        CompilerSession::for_package_with_overlays(graph, overlays, options)
    };
    let program = match session.check() {
        Ok(program) => program,
        Err(failure) => {
            eprint!("{}", failure.render());
            eprintln!("{package_name}: package compilation failed");
            return ExitCode::FAILURE;
        }
    };
    let root_file = program.root_file().clone();
    let hir = program.hir();
    let tables = program.tables();

    let mut overall_failed = false;

    // ---- unit tests: fn test_*() discovered in the package's own module tree ----
    let all_tests = test_runner::discover_tests(hir, &root_file);
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
        let result = test_runner::run_test(hir, root_file.clone(), tables, test);
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

    // ---- integration tests: tests/*.stark|*.st, each a standalone program ----
    let package_root = manifest_path
        .parent()
        .expect("manifest path has a parent directory")
        .to_path_buf();
    if let Some(more_failed) = run_standalone_suite(&package_root.join("tests"), "test", options) {
        overall_failed = overall_failed || more_failed;
    }

    // ---- examples: examples/*.stark|*.st, each compiled and run ----
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

/// Run every `.stark` or `.st` file under `dir` as a standalone program (its own
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
    // AS2: a single-file session. The name stays the PATH — `SourceFile::name`'s contract for a
    // compile with no package around it, and AS0 finding D4 says not to consolidate that away.
    let file = Arc::new(SourceFile::new(path.to_string_lossy().into_owned(), src));
    // The `<n> <phase> error(s)` format is documented in
    // `docs/WP8_3_TEST_FRAMEWORK_IMPLEMENTATION.md`, so the session supplies the failing phase
    // rather than this function keeping its own pipeline to know which one it was. The count now
    // counts ERRORS; the parse and resolve arms previously reported the whole diagnostic list, so a
    // program with warnings alongside a parse error over-reported.
    match CompilerSession::for_source(file, options).check() {
        Err(failure) => Err(failure.summary()),
        Ok(program) => program.execute_hir().map(|_| ()).map_err(|e| e.message),
    }
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
            "Error: no `.stark` or `.st` files found under {}",
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
                // AS1b-ii-d: the parse registered this file; its own registry resolves the spans.
                eprint!("{}", diag.render(&ast.sources));
            }
            eprintln!("Error: {}: parse failed", file_path.display());
            had_errors = true;
            continue;
        }
        let doc_source = ast
            .sources
            .id_for_name(&source.name)
            .expect("the parse registered this file");
        let (_, comments, _) = starkc::lexer::tokenize_with_comments(&source, doc_source);
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

#[cfg(test)]
mod doctor_tests {
    use super::*;

    fn manifest(files: &str) -> String {
        format!(
            r#"{{"stark_version":"0.1.0","host_target":"x86_64-unknown-linux-gnu","files":[{files}]}}"#
        )
    }

    fn entry(path: &str) -> String {
        format!(r#"{{"path":"{path}","size":1,"sha256":"ab"}}"#)
    }

    /// **The defect that mattered most: the old reader did not FAIL on a reformatted manifest, it
    /// reported `manifest_files: ok (0/0 files verified)`.**
    ///
    /// It split the file array on the literal `"\n    {"`, so compact JSON yielded no entries — and
    /// zero of zero verified is a pass. A verifier whose failure mode is silently checking nothing
    /// is worse than one that errors, because the operator is told the installation is sound.
    #[test]
    fn compact_and_pretty_manifests_parse_identically() {
        let compact = manifest(&entry("bin/stark"));
        let pretty = "{\n  \"stark_version\" : \"0.1.0\",\n  \"host_target\":\n\
                      \t\"x86_64-unknown-linux-gnu\",\n  \"files\" : [\n        {\n\
                      \"path\":\"bin/stark\", \"size\" : 1, \"sha256\":\"ab\"\n  }\n ]\n}";
        let a = json::parse(&compact).expect("compact");
        let b = json::parse(pretty).expect("pretty");
        assert_eq!(a, b, "whitespace must not change the value");
        assert_eq!(
            a.object()
                .unwrap()
                .get("files")
                .unwrap()
                .array()
                .unwrap()
                .len(),
            1
        );
    }

    /// Doctor must pass on BOTH installed layouts.
    ///
    /// The packager moved the runtime under `starkc/` so that
    /// `native_build::provider_root_beside_runtime` can derive the provider root from it. Doctor
    /// still probed only the flat path, so a correct mirror-layout installation — one that builds
    /// capability-using programs successfully — reported `runtime: fail`. A tree is checked here
    /// rather than the constant, because the defect was in what the probe LOOKED AT.
    #[test]
    fn doctor_accepts_both_installed_layouts() {
        let base = std::env::temp_dir().join(format!("stark-doctor-layout-{}", std::process::id()));
        for (label, relative) in [
            ("mirror", "lib/stark/starkc/stark-runtime/Cargo.toml"),
            ("flat", "lib/stark/stark-runtime/Cargo.toml"),
        ] {
            let root = base.join(label);
            let file = root.join(relative);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, "[package]\nname='stark-runtime'\n").unwrap();

            let check = layout_check("runtime", &root, RUNTIME_LAYOUTS);
            assert!(check.ok, "{label} layout must satisfy doctor");
            assert_eq!(
                check.detail,
                file.display().to_string(),
                "{label} layout must be reported at the path actually found"
            );
        }

        // An empty root fails, and names the layout a current package writes.
        let empty = base.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        let missing = layout_check("runtime", &empty, RUNTIME_LAYOUTS);
        assert!(!missing.ok, "an empty root must not satisfy doctor");
        assert!(
            missing
                .detail
                .ends_with("lib/stark/starkc/stark-runtime/Cargo.toml"),
            "must point at the mirror layout, got {}",
            missing.detail
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    /// A path containing an escaped quote truncated at the escape under the old reader.
    #[test]
    fn string_escapes_survive() {
        let value = json::parse(r#"{"a":"x\"y\\z\u0041\n","b":"\uD83D\uDE00"}"#).expect("parse");
        let object = value.object().unwrap();
        assert_eq!(object.get("a").unwrap().string().unwrap(), "x\"y\\zA\n");
        assert_eq!(object.get("b").unwrap().string().unwrap(), "😀");
    }

    /// Last-wins would leave it to the reader which `sha256` is checked.
    #[test]
    fn duplicate_keys_are_refused() {
        let error = json::parse(r#"{"sha256":"aa","sha256":"bb"}"#).unwrap_err();
        assert!(error.contains("duplicate key"), "{error}");
    }

    #[test]
    fn malformed_input_is_refused_rather_than_partially_read() {
        for bad in [
            r#"{"a":1"#,
            r#"{"a" 1}"#,
            "{\"a\":\"unterminated",
            r#"{"a":"raw
control"}"#,
            r#"{"a":"\uD800"}"#,
            r#"{"a":"\q"}"#,
            r#"{} trailing"#,
        ] {
            assert!(json::parse(bad).is_err(), "must refuse: {bad}");
        }
    }

    #[test]
    fn nesting_is_bounded() {
        let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
        assert!(
            json::parse(&deep).is_err(),
            "unbounded nesting must be refused"
        );
    }

    /// A size is a byte count. Anything else must not reach the comparison.
    #[test]
    fn only_whole_non_negative_sizes_are_accepted() {
        let value = json::parse(r#"{"a":-1,"b":1.5,"c":7,"d":"7"}"#).unwrap();
        let object = value.object().unwrap();
        assert_eq!(object.get("c").unwrap().u64(), Some(7));
        for key in ["a", "b", "d"] {
            assert_eq!(
                object.get(key).unwrap().u64(),
                None,
                "{key} must not be a size"
            );
        }
    }

    /// **A manifest path decides what gets hashed, so it decides what gets trusted.** One escaping
    /// the root would let a manifest certify a file the package never installed.
    #[test]
    fn manifest_paths_may_not_escape_the_install_root() {
        for bad in [
            "../outside",
            "bin/../../outside",
            "/etc/passwd",
            "\\windows\\system32",
            "C:/windows",
            "bin\\..\\..\\outside",
            "",
        ] {
            assert!(
                check_manifest_path(bad).is_err(),
                "must refuse manifest path: {bad:?}"
            );
        }
        for good in [
            "bin/stark",
            "lib/stark/stark-runtime/Cargo.toml",
            "README.md",
        ] {
            assert!(check_manifest_path(good).is_ok(), "must accept: {good}");
        }
    }

    /// Windows and macOS default filesystems are case-insensitive, so two entries differing only in
    /// case name ONE file — and the second silently certifies whatever the first wrote.
    #[test]
    fn case_colliding_paths_are_refused() {
        let text = manifest(&format!("{},{}", entry("bin/stark"), entry("bin/STARK")));
        let temp =
            std::env::temp_dir().join(format!("stark-doctor-case-{}.json", std::process::id()));
        std::fs::write(&temp, text).unwrap();
        let error = read_install_manifest(&temp).unwrap_err();
        let _ = std::fs::remove_file(&temp);
        assert!(error.contains("twice"), "{error}");
    }

    /// A manifest with no `files` key must be an ERROR, never an empty verification that passes.
    #[test]
    fn a_manifest_without_files_is_an_error_not_an_empty_pass() {
        let temp =
            std::env::temp_dir().join(format!("stark-doctor-nofiles-{}.json", std::process::id()));
        std::fs::write(&temp, r#"{"stark_version":"0.1.0","host_target":"t"}"#).unwrap();
        let error = read_install_manifest(&temp).unwrap_err();
        let _ = std::fs::remove_file(&temp);
        assert!(error.contains("missing files"), "{error}");
    }
}
