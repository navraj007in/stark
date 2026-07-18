//! WP-C1.1 (checklist item 8, "AST span integrity"): `ast.rs` has no span-validation helper of
//! any kind, and nothing previously checked programmatically that a child node's `Span` is
//! contained within its parent's -- `starkc/tests/snapshots.rs` renders span positions as text
//! and catches *regressions* against 15 golden fixtures, but a systematically-wrong-but-stable
//! span (e.g. every node's span silently pointing at its parent's) would produce a stable
//! snapshot and pass forever. This file walks the AST arena directly (every `ExprNode`/
//! `BlockNode` is stored flat in `Ast.exprs`/`Ast.blocks`, so no separate visitor is needed --
//! each node already names its own children's IDs) and asserts real containment invariants
//! across the full spec-fixture corpus.
//!
//! Scope note: this checks `Expr`/`Block` containment only, not `Type`/`Pat`/`Item` nodes or
//! cross-kind containment (e.g. a `Stmt`'s span against its parent `Block`). A fully exhaustive,
//! generic AST-position walker (useful beyond testing -- e.g. "node at byte offset" for LSP
//! hover, DEV-010) is properly scoped to WP-C2.4 ("position and symbol query infrastructure"),
//! not this WP; building it in full here would import a later-gate mechanism. See
//! COMPILER-STATE.md Follow-ups.

use starkc::ast::{Ast, ExprId, ExprKind};
use starkc::parser::{parse, ParseMode};
use starkc::source::{SourceFile, Span};
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../STARKLANG/tests/spec-fixtures")
}

/// Fixture names whose manifest verdict is `notation` -- grammar-snippet fragments extracted
/// from the spec prose (e.g. `&T // Immutable reference to T`), not parseable STARK programs.
/// `conformance.rs`'s own harness skips these ("30 skipped (notation)"); this test follows the
/// same convention rather than treating parser error-recovery output on deliberately-non-STARK
/// input as a span-integrity finding. `parse-fail` fixtures are excluded for the same reason --
/// they are expected not to parse, so whatever partial AST recovery produces is not meaningful
/// to check for containment either.
fn skip_verdicts() -> std::collections::HashSet<String> {
    let manifest_path = fixture_dir().join("manifest.toml");
    let content = std::fs::read_to_string(&manifest_path).expect("manifest.toml is readable");
    let mut skip = std::collections::HashSet::new();
    let mut current: Option<String> = None;
    for line in content.lines() {
        let line = line.trim();
        if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            current = Some(name.trim_matches('"').to_string());
        } else if let Some(verdict) = line.strip_prefix("verdict = \"") {
            let verdict = verdict.trim_end_matches('"');
            if (verdict == "notation" || verdict == "parse-fail") && current.is_some() {
                skip.insert(current.clone().unwrap());
            }
        }
    }
    skip
}

fn contains(outer: Span, inner: Span) -> bool {
    inner.lo >= outer.lo && inner.hi <= outer.hi
}

/// Checks every expr node's directly-named ExprId/BlockId children (per ExprKind) are spanned
/// within the node's own span. Returns human-readable failure descriptions, not a bool, so a
/// failing corpus run names every offending node rather than just the first.
fn check_expr_containment(ast: &Ast, id: ExprId, failures: &mut Vec<String>) {
    let node = ast.expr(id);
    let parent_span = node.span;

    fn check_child(
        ast: &Ast,
        id: ExprId,
        parent_span: Span,
        child: ExprId,
        label: &str,
        failures: &mut Vec<String>,
    ) {
        let child_span = ast.expr(child).span;
        if !contains(parent_span, child_span) {
            failures.push(format!(
                "expr {:?} ({label}): child span {:?} not contained in parent span {:?}",
                id, child_span, parent_span
            ));
        }
    }
    fn check_block(
        ast: &Ast,
        id: ExprId,
        parent_span: Span,
        block: starkc::ast::BlockId,
        label: &str,
        failures: &mut Vec<String>,
    ) {
        let block_span = ast.block(block).span;
        if !contains(parent_span, block_span) {
            failures.push(format!(
                "expr {:?} ({label}): block span {:?} not contained in parent span {:?}",
                id, block_span, parent_span
            ));
        }
    }
    macro_rules! check_child {
        ($child:expr, $label:expr) => {
            check_child(ast, id, parent_span, $child, $label, failures)
        };
    }
    macro_rules! check_block {
        ($block:expr, $label:expr) => {
            check_block(ast, id, parent_span, $block, $label, failures)
        };
    }

    match &node.kind {
        ExprKind::Unary { operand, .. } => check_child!(*operand, "unary operand"),
        ExprKind::Binary { lhs, rhs, .. } => {
            check_child!(*lhs, "binary lhs");
            check_child!(*rhs, "binary rhs");
        }
        ExprKind::Assign { lhs, rhs, .. } => {
            check_child!(*lhs, "assign lhs");
            check_child!(*rhs, "assign rhs");
        }
        ExprKind::Range { lo, hi, .. } => {
            check_child!(*lo, "range lo");
            check_child!(*hi, "range hi");
        }
        ExprKind::Cast { expr, .. } => check_child!(*expr, "cast operand"),
        ExprKind::Call { callee, args } => {
            check_child!(*callee, "call callee");
            for a in args {
                check_child!(*a, "call arg");
            }
        }
        ExprKind::Field { base, .. } => check_child!(*base, "field base"),
        ExprKind::TupleField { base, .. } => check_child!(*base, "tuple-field base"),
        ExprKind::Index { base, index } => {
            check_child!(*base, "index base");
            check_child!(*index, "index index");
        }
        ExprKind::Try(inner) => check_child!(*inner, "try operand"),
        ExprKind::Tuple(elems) | ExprKind::Array(elems) => {
            for e in elems {
                check_child!(*e, "tuple/array element");
            }
        }
        ExprKind::Repeat { value, count } => {
            check_child!(*value, "repeat value");
            check_child!(*count, "repeat count");
        }
        ExprKind::StructLit { fields, .. } => {
            for f in fields {
                if let Some(e) = f.expr {
                    check_child!(e, "struct-lit field value");
                }
            }
        }
        ExprKind::If {
            cond,
            then_block,
            else_,
        } => {
            check_child!(*cond, "if cond");
            check_block!(*then_block, "if then-block");
            if let Some(e) = else_ {
                check_child!(*e, "if else");
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            check_child!(*scrutinee, "match scrutinee");
            for arm in arms {
                check_child!(arm.body, "match arm body");
            }
        }
        ExprKind::Loop { body } => check_block!(*body, "loop body"),
        ExprKind::While { cond, body } => {
            check_child!(*cond, "while cond");
            check_block!(*body, "while body");
        }
        ExprKind::For { iter, body, .. } => {
            check_child!(*iter, "for iter");
            check_block!(*body, "for body");
        }
        ExprKind::Block(block) => check_block!(*block, "block expr"),
        ExprKind::Lit(_) | ExprKind::Path { .. } | ExprKind::Error => {}
    }
}

fn check_block_containment(ast: &Ast, block: starkc::ast::BlockId, failures: &mut Vec<String>) {
    let node = ast.block(block);
    for &stmt_id in &node.stmts {
        let stmt_span = ast.stmt(stmt_id).span;
        if !contains(node.span, stmt_span) {
            failures.push(format!(
                "block {:?}: stmt span {:?} not contained in block span {:?}",
                block, stmt_span, node.span
            ));
        }
    }
    if let Some(tail) = node.tail {
        let tail_span = ast.expr(tail).span;
        if !contains(node.span, tail_span) {
            failures.push(format!(
                "block {:?}: tail expr span {:?} not contained in block span {:?}",
                block, tail_span, node.span
            ));
        }
    }
}

#[test]
fn expr_and_block_spans_are_contained_across_the_fixture_corpus() {
    let mut paths: Vec<_> = std::fs::read_dir(fixture_dir())
        .expect("spec-fixtures directory exists")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "stark"))
        .collect();
    paths.sort();
    assert!(paths.len() >= 100, "expected the full fixture corpus");
    let skip = skip_verdicts();
    let expected_checked = paths.len() - skip.len();
    let mut checked = 0usize;

    let mut failures = Vec::new();
    for path in &paths {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if skip.contains(&name) {
            continue;
        }
        checked += 1;
        let src = std::fs::read_to_string(path).expect("fixture is readable");
        for (mode, mode_name) in [
            (ParseMode::Program, "Program"),
            (ParseMode::Snippet, "Snippet"),
        ] {
            let file = SourceFile::new(name.clone(), src.clone());
            let (ast, _diags) = parse(&file, mode);

            let mut node_failures = Vec::new();
            for i in 0..ast.exprs.len() {
                check_expr_containment(&ast, ExprId(i as u32), &mut node_failures);
            }
            for i in 0..ast.blocks.len() {
                check_block_containment(&ast, starkc::ast::BlockId(i as u32), &mut node_failures);
            }
            if !node_failures.is_empty() {
                failures.push(format!(
                    "{name} ({mode_name}): {} span-containment violation(s):\n  {}",
                    node_failures.len(),
                    node_failures.join("\n  ")
                ));
            }
        }
    }
    assert_eq!(
        checked, expected_checked,
        "every eligible manifest fixture must receive span-integrity checks"
    );
    assert!(
        failures.is_empty(),
        "span-containment violations found:\n{}",
        failures.join("\n\n")
    );
}

/// Narrower, hand-picked regression cases for constructs most likely to get span-splitting
/// wrong: the `>>` generic-close split (checklist item 4) and deeply right-associative chains,
/// where an off-by-one in `eat_gt`'s token mutation or `span_from`'s fallback (parser.rs:556-563
/// falls back to `lo` for `hi` when `self.pos == 0`) would show up as a parent span that doesn't
/// actually cover its child.
#[test]
fn span_containment_survives_generic_close_splitting() {
    for src in [
        "let m: Vec<Vec<Int32>> = Vec::new();",
        "let m: Vec<Vec<Vec<Int32>>> = Vec::new();",
        "let x: Vec<Vec<Int32>>=Vec::new();",
        "let z = a + b * c - d / e;",
        "let z = -----x;",
        "let z = f(g(h(x)));",
    ] {
        let file = SourceFile::new("span.stark", src.to_string());
        let (ast, diags) = parse(&file, ParseMode::Snippet);
        assert!(diags.is_empty(), "{src}: unexpected diagnostics {diags:?}");
        let mut failures = Vec::new();
        for i in 0..ast.exprs.len() {
            check_expr_containment(&ast, ExprId(i as u32), &mut failures);
        }
        assert!(
            failures.is_empty(),
            "{src}: span-containment violations:\n{}",
            failures.join("\n")
        );
    }
}
