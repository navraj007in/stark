//! **AS7 Packet 1 — the dependency-direction forcing test.**
//!
//! AS6's exit qualification found that criterion 2 was the only one of its five with **no
//! behavioural signature**, and that it therefore failed twice after its surfaces had been declared
//! clean. AS7 has the same asymmetry:
//!
//! ```text
//! criterion 1  "no semantic behaviour or diagnostic structure changes"   pinned by existing suites
//! criterion 2  "dependency direction documented and cycle-free"          pinned by NOTHING
//! ```
//!
//! This file is criterion 2's executable check, and it exists **before** any code moves — the
//! ordering AS6 paid to learn.
//!
//! # Why it does not simply grep `use super::`
//!
//! Because that would be another proxy that resembles the question. Rust's inherent-method
//! resolution hides dependencies from the import graph entirely: a method defined in `traits.rs`
//! and called as `self.select_impl(..)` from `body.rs` is a real `body -> traits` dependency that
//! produces **no `use` statement at all**. So this test derives edges from two sources:
//!
//! ```text
//! 1  explicit references   use super::<m>,  crate::typecheck::<m>,  super::<m>::
//! 2  METHOD OWNERSHIP      `fn X` defined in module A, `self.X(` called in module B  =>  B -> A
//! ```
//!
//! `use super::*` is forbidden outright: a glob import would make source 1 blind by construction.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::PathBuf;

/// The approved decomposition (owner decision, 2026-08-08). Frozen for the packet.
const MODULES: &[&str] = &[
    "mod", "types", "state", "infer", "convert", "traits", "patterns", "body", "items",
];

/// The approved dependency DAG, as direct `depends-on` edges. A module may depend on anything
/// **reachable below it**, which the reachability closure below computes; it may depend on nothing
/// else.
///
/// ```text
///   types <- state <- infer <- convert <- traits, patterns <- body <- items <- mod
/// ```
const DECLARED_EDGES: &[(&str, &str)] = &[
    ("mod", "items"),
    ("items", "body"),
    ("body", "traits"),
    ("body", "patterns"),
    ("traits", "convert"),
    ("patterns", "convert"),
    ("convert", "infer"),
    ("infer", "state"),
    ("state", "types"),
];

fn typecheck_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/typecheck")
}

fn read_normalised(path: &std::path::Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} unreadable: {e}", path.display()))
        .replace("\r\n", "\n")
}

/// Everything reachable from `from` in the declared DAG, i.e. everything it may depend on.
fn permitted(from: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::from([from.to_string()]);
    while let Some(m) = queue.pop_front() {
        for (a, b) in DECLARED_EDGES {
            if *a == m && seen.insert(b.to_string()) {
                queue.push_back(b.to_string());
            }
        }
    }
    seen
}

/// **The declared graph must itself be a DAG.** Checked unconditionally — it is the one assertion
/// that is meaningful before the split and would be embarrassing to discover after it.
#[test]
fn declared_dependency_graph_is_acyclic() {
    for m in MODULES {
        assert!(
            !permitted(m).contains(*m),
            "declared graph has a cycle through `{m}`"
        );
    }
    for (a, b) in DECLARED_EDGES {
        assert!(MODULES.contains(a), "edge names unknown module `{a}`");
        assert!(MODULES.contains(b), "edge names unknown module `{b}`");
    }
}

/// Modules present on disk today. Empty before Packet 3 splits the file, which is not a failure —
/// the checks below are written to be exact whenever they can run at all.
fn present_modules() -> BTreeMap<String, String> {
    let dir = typecheck_dir();
    let mut found = BTreeMap::new();
    if !dir.is_dir() {
        return found;
    }
    for name in MODULES {
        let path = dir.join(format!("{name}.rs"));
        if path.is_file() {
            found.insert((*name).to_string(), read_normalised(&path));
        }
    }
    found
}

/// Strip `//` comments and `#[cfg(test)]` bodies: a doc comment naming a module is not a
/// dependency, and a unit test may legitimately reach anywhere.
fn production_code(source: &str) -> String {
    let source = match source.find("#[cfg(test)]") {
        Some(i) => &source[..i],
        None => source,
    };
    source
        .lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A glob import would make the explicit-reference half of this test blind, so it is banned.
#[test]
fn split_modules_do_not_glob_import_siblings() {
    let violations: Vec<String> = present_modules()
        .iter()
        .filter(|(_, src)| production_code(src).contains("use super::*"))
        .map(|(m, _)| format!("typecheck/{m}.rs uses `use super::*`"))
        .collect();
    assert!(
        violations.is_empty(),
        "a glob import hides the dependency graph this packet exists to enforce:\n  {}",
        violations.join("\n  ")
    );
}

/// Inherent `fn` names defined in each module — the ownership map that makes method-call
/// dependencies visible.
fn method_owners(modules: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut owner = BTreeMap::new();
    for (m, src) in modules {
        for line in production_code(src).lines() {
            let t = line.trim_start();
            let Some(rest) = t
                .strip_prefix("pub(crate) fn ")
                .or_else(|| t.strip_prefix("pub fn "))
                .or_else(|| t.strip_prefix("fn "))
            else {
                continue;
            };
            if let Some(name) = rest.split('(').next() {
                let name = name.trim();
                if !name.is_empty() && !name.contains(char::is_whitespace) {
                    owner.insert(name.to_string(), m.clone());
                }
            }
        }
    }
    owner
}

/// **The check.** Every observed edge must be permitted by the declared DAG.
#[test]
fn observed_dependencies_respect_the_declared_direction() {
    let modules = present_modules();
    if modules.len() < 2 {
        // Before Packet 3, there is nothing to constrain. `declared_dependency_graph_is_acyclic`
        // still runs, and this test becomes enforcing the moment the split begins.
        return;
    }
    let owners = method_owners(&modules);
    let mut violations = Vec::new();

    for (m, src) in &modules {
        let code = production_code(src);
        let allowed = permitted(m);
        let mut observed: BTreeSet<String> = BTreeSet::new();

        // 1. explicit references
        for other in MODULES {
            if other == m {
                continue;
            }
            for pattern in [
                format!("use super::{other}"),
                format!("super::{other}::"),
                format!("crate::typecheck::{other}::"),
            ] {
                if code.contains(&pattern) {
                    observed.insert((*other).to_string());
                }
            }
        }

        // 2. method ownership — the half a `use`-only test cannot see
        for (method, owner) in &owners {
            if owner != m && code.contains(&format!("self.{method}(")) {
                observed.insert(owner.clone());
            }
        }

        for dep in observed {
            if !allowed.contains(&dep) {
                violations.push(format!(
                    "typecheck/{m}.rs -> typecheck/{dep}.rs is not permitted \
                     (`{m}` may depend on: {})",
                    allowed.iter().cloned().collect::<Vec<_>>().join(", ")
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "AS7 exit criterion 2 — dependency direction:\n  {}\n\
         Either the code is wrong, or the decomposition is. Per the owner's rule, change the \
         decomposition ONLY if real code proves the direction cannot express an established \
         semantic interaction without a cycle — not because a move is inconvenient.",
        violations.join("\n  ")
    );
}

/// **The AS6 boundary is frozen and AS7 must not reopen it.** The tensor extension may consume
/// checked types; it may not regain control over Core expression checking.
#[test]
fn as7_does_not_reopen_the_as6_extension_boundary() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/extensions/tensor/check.rs");
    let code = production_code(&read_normalised(&path));
    assert!(
        !code.contains("check_expr"),
        "extensions/tensor/check.rs references `check_expr`: AS6's one-directional semantic \
         control has been reopened. The extension consumes checked expression types; it does not \
         cause expression checking."
    );
    assert!(
        !code.contains("fn check_expr"),
        "extensions/tensor/check.rs defines an expression checker"
    );
}
