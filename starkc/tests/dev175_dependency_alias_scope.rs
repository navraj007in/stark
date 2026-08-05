//! **DEV-175 — a dependency alias is visible from every module of the package that declared it.**
//!
//! The parser attaches each package's direct dependencies as synthetic module wrappers under that
//! package's ROOT module. First-segment resolution searched the lexical scope, the current module's
//! items, primitives, built-ins and Core traits — and never the containing package's dependencies.
//! So `use stark_http_core::Header;` resolved in `src/main.stark`, where the wrapper is physically
//! present, and failed with E0205 from any sibling module; the fully-qualified path failed the same
//! way with E0202. Every package in the repo happened to be a single `lib.stark`, so nothing had
//! hit it until an application wanted more than one file.
//!
//! The repair is a package-scoped alias table consulted after the current module's own items. The
//! risk in it is over-reach, so half the matrix below is negative: the fix must not make ordinary
//! root items visible from child modules, and must not expose a transitive dependency the package
//! never declared.

use starkc::diag::Severity;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::parse_package_graph;
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::PathBuf;
use std::sync::Arc;

/// A package in a temporary workspace: its name, the dependencies it DECLARES, and its files.
///
/// `deps` are `(manifest alias, package directory)` so a manifest alias that differs from the
/// package name is expressible — that is case 5, and it is the case that would break if the lookup
/// keyed on anything but the alias as written.
struct Pkg<'a> {
    name: &'a str,
    deps: &'a [(&'a str, &'a str)],
    /// `(path under src/, source)`. The first entry must be `main.stark`.
    files: &'a [(&'a str, &'a str)],
}

fn workspace(tag: &str, packages: &[Pkg]) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "stark_dev175_{tag}_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    for pkg in packages {
        let dir = root.join(pkg.name);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let deps = if pkg.deps.is_empty() {
            String::new()
        } else {
            let entries: Vec<String> = pkg
                .deps
                .iter()
                .map(|(alias, path)| {
                    format!(r#""{alias}": {{ "package": "{path}", "path": "../{path}" }}"#)
                })
                .collect();
            format!(r#", "dependencies": {{ {} }}"#, entries.join(", "))
        };
        std::fs::write(
            dir.join("starkpkg.json"),
            format!(
                r#"{{ "name": "{}", "version": "0.1.0", "entry": "src/main.stark"{deps} }}"#,
                pkg.name
            ),
        )
        .unwrap();
        for (path, src) in pkg.files {
            let file = dir.join("src").join(path);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, src).unwrap();
        }
    }
    root
}

/// Resolve the first package and return its error diagnostics as `(code, message)`.
fn resolve_errors(tag: &str, packages: &[Pkg]) -> Vec<(String, String)> {
    let root = workspace(tag, packages);
    let app_dir = root.join(packages[0].name);
    let manifest = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest).unwrap();
    let (ast, parse_diags) = parse_package_graph(&graph, LanguageOptions::CORE);
    let mut out: Vec<(String, String)> = parse_diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| (d.code.clone().unwrap_or_default(), d.message.clone()))
        .collect();
    let entry = app_dir.join("src/main.stark");
    let src = std::fs::read_to_string(&entry).unwrap();
    let file = Arc::new(SourceFile::new(entry.to_string_lossy().into_owned(), src));
    let (_, diags) = resolve(&ast, file);
    out.extend(
        diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| (d.code.clone().unwrap_or_default(), d.message.clone())),
    );
    out
}

/// Accepting cases are TYPE-CHECKED, not merely resolved.
///
/// Resolution alone cannot distinguish which `lib` a name bound to, so case 11 would pass either
/// way. The two `make`s have different return types, so the type checker is what makes the answer
/// observable.
fn assert_resolves(tag: &str, packages: &[Pkg]) {
    let root = workspace(tag, packages);
    let app_dir = root.join(packages[0].name);
    let manifest = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest).unwrap();
    let (ast, parse_diags) = parse_package_graph(&graph, LanguageOptions::CORE);
    assert!(parse_diags.is_empty(), "{tag} parse: {parse_diags:?}");
    let entry = app_dir.join("src/main.stark");
    let src = std::fs::read_to_string(&entry).unwrap();
    let file = Arc::new(SourceFile::new(entry.to_string_lossy().into_owned(), src));
    let (hir, diags) = resolve(&ast, file.clone());
    let errors: Vec<String> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    assert!(errors.is_empty(), "{tag} must resolve, got {errors:?}");
    let checked = typecheck::analyze(&hir, file);
    let type_errors: Vec<String> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    assert!(
        type_errors.is_empty(),
        "{tag} must type-check, got {type_errors:?}"
    );
}

fn assert_rejected(tag: &str, packages: &[Pkg], code: &str, needle: &str) {
    let errors = resolve_errors(tag, packages);
    assert!(
        errors.iter().any(|(c, m)| c == code && m.contains(needle)),
        "{tag} must be rejected with {code} mentioning {needle:?}, got {errors:?}"
    );
}

/// The dependency every case below imports from. `Secret` is deliberately private: it is what
/// case 8 reaches for.
const LIB: &str = "\
pub struct Widget { pub size: Int32 }

struct Secret { hidden: Int32 }

pub fn make(size: Int32) -> Widget {
    Widget { size: size }
}
";

fn lib(name: &str) -> Pkg<'static> {
    // Leaked so the &'static str the struct wants can be built from a runtime name.
    Pkg {
        name: Box::leak(name.to_string().into_boxed_str()),
        deps: &[],
        files: Box::leak(Box::new([("main.stark", LIB)])),
    }
}

// ------------------------------------------------------------------------------------- ACCEPT --

/// 1. The entry file. This always worked — it is the control that says the repair did not break
///    the one case that was already fine.
#[test]
fn the_entry_file_uses_a_direct_dependency() {
    assert_resolves(
        "entry",
        &[
            Pkg {
                name: "app",
                deps: &[("lib", "lib")],
                files: &[(
                    "main.stark",
                    "\
use lib::make;
use lib::Widget;

fn main() {
    let w: Widget = make(3);
    println(w.size);
}
",
                )],
            },
            lib("lib"),
        ],
    );
}

/// 2. **The reported defect.** A sibling module importing the package's own dependency.
#[test]
fn a_child_module_imports_a_direct_dependency() {
    assert_resolves(
        "child",
        &[
            Pkg {
                name: "app",
                deps: &[("lib", "lib")],
                files: &[
                    (
                        "main.stark",
                        "mod child;\n\nfn main() {\n    println(child::size_of(4));\n}\n",
                    ),
                    (
                        "child.stark",
                        "\
use lib::make;
use lib::Widget;

pub fn size_of(n: Int32) -> Int32 {
    let w: Widget = make(n);
    w.size
}
",
                    ),
                ],
            },
            lib("lib"),
        ],
    );
}

/// 3. Nested one level deeper. The alias is scoped to the PACKAGE, not to the root's direct
///    children, so depth must not matter.
#[test]
fn a_nested_child_module_imports_a_direct_dependency() {
    assert_resolves(
        "nested",
        &[
            Pkg {
                name: "app",
                deps: &[("lib", "lib")],
                files: &[
                    (
                        "main.stark",
                        "mod outer;\n\nfn main() {\n    println(outer::inner::size_of(5));\n}\n",
                    ),
                    ("outer/mod.stark", "pub mod inner;\n"),
                    (
                        "outer/inner.stark",
                        "\
use lib::make;

pub fn size_of(n: Int32) -> Int32 {
    make(n).size
}
",
                    ),
                ],
            },
            lib("lib"),
        ],
    );
}

/// 4. The fully-qualified spelling from a child, with no `use` at all. This failed with E0202
///    rather than E0205, which is why fixing only the import path would have been half a repair.
#[test]
fn a_child_module_uses_a_fully_qualified_dependency_alias() {
    assert_resolves(
        "qualified",
        &[
            Pkg {
                name: "app",
                deps: &[("lib", "lib")],
                files: &[
                    (
                        "main.stark",
                        "mod child;\n\nfn main() {\n    println(child::size_of(6));\n}\n",
                    ),
                    (
                        "child.stark",
                        "\
pub fn size_of(n: Int32) -> Int32 {
    let w: lib::Widget = lib::make(n);
    w.size
}
",
                    ),
                ],
            },
            lib("lib"),
        ],
    );
}

/// 5. A manifest alias that is not the package's name. The lookup keys on the alias as the manifest
///    spells it; keying on the package name would fail here and nowhere else.
#[test]
fn a_manifest_alias_may_differ_from_the_package_name() {
    assert_resolves(
        "alias",
        &[
            Pkg {
                name: "app",
                deps: &[("renamed", "lib")],
                files: &[
                    (
                        "main.stark",
                        "mod child;\n\nfn main() {\n    println(child::size_of(7));\n}\n",
                    ),
                    (
                        "child.stark",
                        "\
use renamed::make;

pub fn size_of(n: Int32) -> Int32 {
    make(n).size
}
",
                    ),
                ],
            },
            lib("lib"),
        ],
    );
}

/// 6. The same rule one level down: a DEPENDENCY's own child module reaching the dependency's own
///    dependency. Registration is keyed by the declaring package's root, so this must work without
///    the alias becoming visible to the app — case 7 is the other half of that claim.
#[test]
fn a_dependencys_child_module_uses_its_own_direct_dependency() {
    assert_resolves(
        "transitive_ok",
        &[
            Pkg {
                name: "app",
                deps: &[("mid", "mid")],
                files: &[(
                    "main.stark",
                    "use mid::relay;\n\nfn main() {\n    println(relay(8));\n}\n",
                )],
            },
            Pkg {
                name: "mid",
                deps: &[("lib", "lib")],
                files: &[
                    ("main.stark", "mod helper;\n\npub fn relay(n: Int32) -> Int32 {\n    helper::size_of(n)\n}\n"),
                    (
                        "helper.stark",
                        "\
use lib::make;

pub fn size_of(n: Int32) -> Int32 {
    make(n).size
}
",
                    ),
                ],
            },
            lib("lib"),
        ],
    );
}

// ------------------------------------------------------------------------------------- REJECT --

/// 7. **The containment control.** `app -> mid -> lib`; `app` never declared `lib`. Registering
///    aliases against a global package graph instead of the declaring package's root would leak
///    every transitive dependency into every package, and this is what catches that.
#[test]
fn an_undeclared_transitive_dependency_stays_out_of_reach() {
    assert_rejected(
        "transitive_reject",
        &[
            Pkg {
                name: "app",
                deps: &[("mid", "mid")],
                files: &[
                    (
                        "main.stark",
                        "mod child;\n\nfn main() {\n    println(child::size_of(9));\n}\n",
                    ),
                    (
                        "child.stark",
                        "\
use lib::make;

pub fn size_of(n: Int32) -> Int32 {
    make(n).size
}
",
                    ),
                ],
            },
            Pkg {
                name: "mid",
                deps: &[("lib", "lib")],
                files: &[(
                    "main.stark",
                    "pub fn relay(n: Int32) -> Int32 {\n    n\n}\n",
                )],
            },
            lib("lib"),
        ],
        "E0205",
        "lib",
    );
}

/// 8. Reaching a dependency's PRIVATE item from a child module. Making the alias visible must not
///    also make the dependency's internals visible: visibility is applied on the later segments,
///    and this proves that path still runs.
#[test]
fn a_private_dependency_item_stays_private_from_a_child_module() {
    let errors = resolve_errors(
        "private",
        &[
            Pkg {
                name: "app",
                deps: &[("lib", "lib")],
                files: &[
                    (
                        "main.stark",
                        "mod child;\n\nfn main() {\n    println(child::peek());\n}\n",
                    ),
                    (
                        "child.stark",
                        "\
use lib::Secret;

pub fn peek() -> Int32 {
    0
}
",
                    ),
                ],
            },
            lib("lib"),
        ],
    );
    assert!(
        !errors.is_empty(),
        "a private dependency item must not be importable from a child module"
    );
}

/// 9. **The leakage control, and the most important negative in this file.** An ordinary root-level
///    function must remain invisible unqualified from a child module. A fix that fell back to
///    searching the package root's ITEMS — rather than a table holding only dependency aliases —
///    would make every root helper implicitly global, which the spec forbids: an unqualified name
///    does not search parent or crate scopes.
#[test]
fn an_ordinary_root_item_does_not_leak_into_a_child_module() {
    assert_rejected(
        "leak",
        &[
            Pkg {
                name: "app",
                deps: &[("lib", "lib")],
                files: &[
                    (
                        "main.stark",
                        "mod child;\n\nfn root_only() -> Int32 {\n    1\n}\n\nfn main() {\n    println(child::call());\n}\n",
                    ),
                    ("child.stark", "pub fn call() -> Int32 {\n    root_only()\n}\n"),
                ],
            },
            lib("lib"),
        ],
        "E0200",
        "root_only",
    );
}

/// 12. An alias the manifest never declared is still an ordinary unresolved name.
#[test]
fn an_unknown_dependency_alias_is_still_unresolved() {
    assert_rejected(
        "unknown",
        &[
            Pkg {
                name: "app",
                deps: &[("lib", "lib")],
                files: &[
                    (
                        "main.stark",
                        "mod child;\n\nfn main() {\n    println(child::call());\n}\n",
                    ),
                    (
                        "child.stark",
                        "\
use nonexistent::thing;

pub fn call() -> Int32 {
    0
}
",
                    ),
                ],
            },
            lib("lib"),
        ],
        "E0205",
        "nonexistent",
    );
}

// ------------------------------------------------------------------------------- name ordering --

/// 10. `crate::root_item` still reaches the root explicitly and obeys ordinary visibility — the
///     explicit path is unaffected by the alias table, which only changes UNqualified first
///     segments.
#[test]
fn an_explicit_crate_path_still_reaches_a_root_item() {
    assert_resolves(
        "crate_path",
        &[
            Pkg {
                name: "app",
                deps: &[("lib", "lib")],
                files: &[
                    (
                        "main.stark",
                        "mod child;\n\npub fn root_item() -> Int32 {\n    2\n}\n\nfn main() {\n    println(child::call());\n}\n",
                    ),
                    ("child.stark", "pub fn call() -> Int32 {\n    crate::root_item()\n}\n"),
                ],
            },
            lib("lib"),
        ],
    );
}

/// 11. **A local name wins over a dependency alias.** The alias lookup is placed AFTER the current
///     module's own items deliberately: a module that defines `lib` means its own `lib`, and a
///     dependency must not be able to capture a name the module already bound. The assertion is on
///     the VALUE — `20` is the local's answer and `4` the dependency's — because both spellings
///     compile and only the result distinguishes them.
#[test]
fn a_local_name_shadows_a_dependency_alias() {
    assert_resolves(
        "shadow",
        &[
            Pkg {
                name: "app",
                deps: &[("lib", "lib")],
                files: &[
                    (
                        "main.stark",
                        "mod child;\n\nfn main() {\n    println(child::call());\n}\n",
                    ),
                    (
                        "child.stark",
                        "\
pub mod lib {
    pub fn make(n: Int32) -> Int32 {
        n * 10
    }
}

pub fn call() -> Int32 {
    lib::make(2)
}
",
                    ),
                ],
            },
            lib("lib"),
        ],
    );
}
