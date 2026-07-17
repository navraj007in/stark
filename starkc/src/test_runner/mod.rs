//! `stark test` — unit test discovery and execution (WP8.3).
//!
//! Core v1 has no attribute syntax (`#[test]`/`#[ignore]` don't exist in
//! the grammar — see `docs/PHASE8_GRAMMAR_GAPS.md`), so test discovery uses
//! a naming convention instead, per the Phase 8 plan's own fallback:
//!
//! - `fn test_*()` — a unit test. Must take no parameters and have no
//!   receiver (a plain top-level or `mod`-nested function).
//! - `fn test_ignored_*()` — a unit test that is discovered (counted,
//!   listed) but not run unless `--ignored` is passed.
//!
//! Each test runs as its own interpreter entry point
//! (`interp::run_item`), independent of `main` and of every other test —
//! one test panicking never aborts the run.

use crate::hir::{Hir, ItemId, ItemKind, Root};
use crate::interp;
use crate::source::SourceFile;
use crate::typecheck::TypeTables;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A discovered test function.
pub struct TestCase {
    pub item: ItemId,
    /// Fully-qualified-ish display name: `mod` path joined with `::`, then
    /// the function name (e.g. `geometry::tests::test_area`). Matches the
    /// plan's `module1::test_basic` output format.
    pub name: String,
    pub ignored: bool,
}

pub enum Outcome {
    Passed,
    Failed { message: String },
    Ignored,
}

pub struct TestResult {
    pub name: String,
    pub outcome: Outcome,
    pub duration: Duration,
    /// Captured stdout (`println`/`print` output) from the test's run.
    /// Only surfaced to the user when `--show-output` is set or the test
    /// failed.
    pub output: String,
}

/// Discover every `test_*` function in `hir`, recursing into `mod` bodies.
/// Order matches source order (depth-first, same order `mod` items appear
/// in their parent's item list) so `--show-output`-free runs are
/// deterministic without needing the `--seed` flag the plan mentions —
/// Core v1 has no randomness primitives yet (`docs/PHASE8_GRAMMAR_GAPS.md`),
/// so there is nothing to seed and test order is always stable.
pub fn discover_tests(hir: &Hir, root_file: &SourceFile) -> Vec<TestCase> {
    let mut tests = Vec::new();
    if let Root::Program(items) = &hir.root {
        collect(hir, root_file, items, &mut Vec::new(), &mut tests);
    }
    tests
}

fn collect(
    hir: &Hir,
    root_file: &SourceFile,
    items: &[ItemId],
    mod_path: &mut Vec<String>,
    out: &mut Vec<TestCase>,
) {
    for &id in items {
        let node = hir.item(id);
        match &node.kind {
            ItemKind::Fn(def) => {
                let name = item_text(hir, root_file, id, def.sig.name);
                if let Some(suffix) = test_name(name) {
                    if def.sig.params.is_empty() && def.sig.receiver.is_none() {
                        let mut full = mod_path.clone();
                        // Display name keeps the full function name
                        // (including the `test_` marker), matching the
                        // plan's `module1::test_basic` output convention —
                        // only discovery strips the prefix.
                        full.push(name.to_string());
                        out.push(TestCase {
                            item: id,
                            name: full.join("::"),
                            ignored: suffix.starts_with("ignored_"),
                        });
                    }
                }
            }
            ItemKind::Mod {
                name,
                items: Some(inner),
            } => {
                mod_path.push(item_text(hir, root_file, id, *name).to_string());
                collect(hir, root_file, inner, mod_path, out);
                mod_path.pop();
            }
            _ => {}
        }
    }
}

/// Strips the `test_` prefix that marks a function as a test, or `None` if
/// `name` isn't one (so e.g. `test` itself, with nothing after the
/// underscore, is not a test — matches the plan's `test_*` convention
/// literally).
fn test_name(name: &str) -> Option<&str> {
    name.strip_prefix("test_").filter(|rest| !rest.is_empty())
}

/// `hir.item_files` only covers items loaded from `mod` submodule files
/// (`resolve.rs` populates it during AST->HIR lowering, carried over from
/// the parser's per-submodule tracking); an item declared directly in the
/// package's entry file has no entry there, so `root_file` is the fallback.
fn item_text<'a>(
    hir: &'a Hir,
    root_file: &'a SourceFile,
    item: ItemId,
    span: crate::source::Span,
) -> &'a str {
    let src = hir
        .item_files
        .get(&item)
        .map(|f| f.src.as_str())
        .unwrap_or(&root_file.src);
    &src[span.lo as usize..span.hi as usize]
}

/// Run a single discovered test as its own interpreter entry point.
/// Any panic/trap (`RuntimeError`) is a failure, not a propagated error —
/// exactly one test's outcome, never the whole run.
pub fn run_test(
    hir: &Hir,
    root_file: Arc<SourceFile>,
    tables: &TypeTables,
    test: &TestCase,
) -> TestResult {
    if test.ignored {
        return TestResult {
            name: test.name.clone(),
            outcome: Outcome::Ignored,
            duration: Duration::ZERO,
            output: String::new(),
        };
    }
    let start = Instant::now();
    let result = interp::run_item(hir, root_file, tables, test.item);
    let duration = start.elapsed();
    match result {
        Ok(execution) => TestResult {
            name: test.name.clone(),
            outcome: Outcome::Passed,
            duration,
            output: execution.output,
        },
        Err(error) => TestResult {
            name: test.name.clone(),
            outcome: Outcome::Failed {
                message: error.message,
            },
            duration,
            output: String::new(),
        },
    }
}

/// Filter tests by a name substring (the plan's `stark test test_name`
/// single-test form; substrings match multiple tests, as in `cargo test`).
pub fn filter_by_name<'a>(tests: &'a [TestCase], name_filter: Option<&str>) -> Vec<&'a TestCase> {
    match name_filter {
        Some(f) => tests.iter().filter(|t| t.name.contains(f)).collect(),
        None => tests.iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_requires_nonempty_suffix() {
        assert_eq!(test_name("test_foo"), Some("foo"));
        assert_eq!(test_name("test_"), None);
        assert_eq!(test_name("test"), None);
        assert_eq!(test_name("nottest_foo"), None);
    }

    #[test]
    fn ignored_prefix_detection() {
        assert_eq!(test_name("test_ignored_slow"), Some("ignored_slow"));
    }
}
