//! Definite-assignment and assignment-mutability data-flow analysis.
//!
//! This pass is intentionally separate from type inference. It propagates a
//! flow state through expressions and intersects branch exits, which avoids
//! treating every nested block as a fresh function-local environment.

use crate::ast::{AssignOp, UnOp};
use crate::diag::Diagnostic;
use crate::hir::{self, BlockId, ExprId, Hir, LocalId, PatId, Res, StmtId};
use crate::source::SourceFile;
use crate::typecheck::Ty;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Clone, Default)]
struct FlowState {
    initialized: HashSet<LocalId>,
    mutable: HashMap<LocalId, bool>,
}

pub fn check(
    hir: &Hir,
    _file: Arc<SourceFile>,
    expr_types: &HashMap<ExprId, Ty>,
) -> Vec<Diagnostic> {
    let mut checker = FlowChecker {
        hir,
        expr_types,
        diagnostics: Vec::new(),
    };
    checker.check_root();
    checker.diagnostics
}

struct FlowChecker<'a> {
    hir: &'a Hir,
    expr_types: &'a HashMap<ExprId, Ty>,
    diagnostics: Vec<Diagnostic>,
}

impl FlowChecker<'_> {
    fn check_root(&mut self) {
        for item in &self.hir.items {
            match &item.kind {
                hir::ItemKind::Fn(def) => self.check_fn(def),
                hir::ItemKind::Impl { items, .. } => {
                    for item in items {
                        if let hir::ImplItem::Fn { def, .. } = item {
                            self.check_fn(def);
                        }
                    }
                }
                hir::ItemKind::Trait { items, .. } => {
                    for item in items {
                        if let hir::TraitItem::Method {
                            sig,
                            body: Some(body),
                        } = item
                        {
                            self.check_body(sig, *body);
                        }
                    }
                }
                _ => {}
            }
        }

        if let hir::Root::Snippet { stmts, tail } = &self.hir.root {
            let mut state = FlowState::default();
            for &stmt in stmts {
                self.visit_stmt(stmt, &mut state);
            }
            if let Some(tail) = tail {
                self.visit_expr(*tail, &mut state);
            }
        }
    }

    fn check_fn(&mut self, def: &hir::FnDef) {
        self.check_body(&def.sig, def.body);
    }

    fn check_body(&mut self, sig: &hir::FnSig, body: BlockId) {
        let mut state = FlowState::default();
        if let Some(local) = sig.receiver_local {
            state.initialized.insert(local);
            state
                .mutable
                .insert(local, matches!(sig.receiver, Some(hir::Receiver::RefMut)));
        }
        for param in &sig.params {
            state.initialized.insert(param.local);
            state.mutable.insert(param.local, param.mutable);
        }
        self.visit_block(body, &mut state);
    }

    fn visit_block(&mut self, block_id: BlockId, state: &mut FlowState) {
        let block = self.hir.block(block_id);
        let mut declared = Vec::new();
        for &stmt in &block.stmts {
            if let hir::StmtKind::Let { local, .. } = self.hir.stmt(stmt).kind {
                declared.push(local);
            }
            self.visit_stmt(stmt, state);
        }
        if let Some(tail) = block.tail {
            self.visit_expr(tail, state);
        }
        for local in declared {
            state.initialized.remove(&local);
            state.mutable.remove(&local);
        }
    }

    fn visit_stmt(&mut self, stmt_id: StmtId, state: &mut FlowState) {
        let stmt = self.hir.stmt(stmt_id);
        match &stmt.kind {
            hir::StmtKind::Empty | hir::StmtKind::Continue | hir::StmtKind::Error => {}
            hir::StmtKind::Expr { expr, .. } => self.visit_expr(*expr, state),
            hir::StmtKind::Let {
                mutable,
                local,
                init,
                ..
            } => {
                if let Some(init) = init {
                    self.visit_expr(*init, state);
                    state.initialized.insert(*local);
                } else {
                    state.initialized.remove(local);
                }
                state.mutable.insert(*local, *mutable);
            }
            hir::StmtKind::Return(expr) | hir::StmtKind::Break(expr) => {
                if let Some(expr) = expr {
                    self.visit_expr(*expr, state);
                }
            }
            hir::StmtKind::Item(_) => {}
        }
    }

    fn visit_expr(&mut self, expr_id: ExprId, state: &mut FlowState) {
        let expr = self.hir.expr(expr_id);
        match &expr.kind {
            hir::ExprKind::Lit(_) | hir::ExprKind::Error => {}
            hir::ExprKind::Path { res, .. } => {
                if let Res::Local(local) | Res::SelfValue(local) = res {
                    self.require_initialized(*local, expr.span, state);
                }
            }
            hir::ExprKind::Unary { operand, .. }
            | hir::ExprKind::Try(operand)
            | hir::ExprKind::Cast { expr: operand, .. } => self.visit_expr(*operand, state),
            hir::ExprKind::Binary { lhs, rhs, .. }
            | hir::ExprKind::Range {
                lo: lhs, hi: rhs, ..
            } => {
                self.visit_expr(*lhs, state);
                self.visit_expr(*rhs, state);
            }
            hir::ExprKind::Assign { op, lhs, rhs } => {
                if *op != AssignOp::Assign {
                    self.visit_expr(*lhs, state);
                } else {
                    self.visit_place_inputs(*lhs, state);
                }
                self.visit_expr(*rhs, state);
                self.assign_place(*lhs, expr.span, state);
            }
            hir::ExprKind::Call { callee, args } => {
                self.visit_expr(*callee, state);
                for &arg in args {
                    self.visit_expr(arg, state);
                }
            }
            hir::ExprKind::Field { base, .. } | hir::ExprKind::TupleField { base, .. } => {
                self.visit_expr(*base, state);
            }
            hir::ExprKind::Index { base, index }
            | hir::ExprKind::Repeat {
                value: base,
                count: index,
            } => {
                self.visit_expr(*base, state);
                self.visit_expr(*index, state);
            }
            hir::ExprKind::Tuple(elements) | hir::ExprKind::Array(elements) => {
                for &element in elements {
                    self.visit_expr(element, state);
                }
            }
            hir::ExprKind::StructLit { fields, .. } => {
                for field in fields {
                    if let Some(value) = field.expr {
                        self.visit_expr(value, state);
                    }
                }
            }
            hir::ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                self.visit_expr(*cond, state);
                let before = state.clone();
                let mut then_state = before.clone();
                self.visit_block(*then_block, &mut then_state);
                let mut else_state = before.clone();
                if let Some(else_expr) = else_ {
                    self.visit_expr(*else_expr, &mut else_state);
                }
                state.initialized = then_state
                    .initialized
                    .intersection(&else_state.initialized)
                    .copied()
                    .collect();
            }
            hir::ExprKind::Match { scrutinee, arms } => {
                self.visit_expr(*scrutinee, state);
                let before = state.clone();
                let mut exits = Vec::new();
                for arm in arms {
                    let mut arm_state = before.clone();
                    let bindings = self.bind_pattern(arm.pat, &mut arm_state);
                    self.visit_expr(arm.body, &mut arm_state);
                    for local in bindings {
                        arm_state.initialized.remove(&local);
                        arm_state.mutable.remove(&local);
                    }
                    exits.push(arm_state.initialized);
                }
                if let Some(first) = exits
                    .into_iter()
                    .reduce(|left, right| left.intersection(&right).copied().collect())
                {
                    state.initialized = first;
                }
            }
            hir::ExprKind::Loop { body } => {
                let mut body_state = state.clone();
                self.visit_block(*body, &mut body_state);
            }
            hir::ExprKind::While { cond, body } => {
                self.visit_expr(*cond, state);
                let mut body_state = state.clone();
                self.visit_block(*body, &mut body_state);
            }
            hir::ExprKind::For {
                local, iter, body, ..
            } => {
                self.visit_expr(*iter, state);
                let mut body_state = state.clone();
                body_state.initialized.insert(*local);
                body_state.mutable.insert(*local, false);
                self.visit_block(*body, &mut body_state);
            }
            hir::ExprKind::Block(block) => self.visit_block(*block, state),
        }
    }

    fn visit_place_inputs(&mut self, expr_id: ExprId, state: &mut FlowState) {
        match &self.hir.expr(expr_id).kind {
            hir::ExprKind::Path { .. } => {}
            hir::ExprKind::Field { base, .. } | hir::ExprKind::TupleField { base, .. } => {
                self.visit_place_inputs(*base, state);
            }
            hir::ExprKind::Index { base, index } => {
                self.visit_place_inputs(*base, state);
                self.visit_expr(*index, state);
            }
            hir::ExprKind::Unary {
                op: UnOp::Deref,
                operand,
            } => self.visit_expr(*operand, state),
            _ => self.visit_expr(expr_id, state),
        }
    }

    fn assign_place(
        &mut self,
        place: ExprId,
        assignment_span: crate::source::Span,
        state: &mut FlowState,
    ) {
        let Some(root) = self.root_local(place) else {
            if !matches!(
                self.hir.expr(place).kind,
                hir::ExprKind::Unary {
                    op: UnOp::Deref,
                    ..
                }
            ) {
                self.assignment_error(assignment_span);
            }
            return;
        };

        let direct = matches!(self.hir.expr(place).kind, hir::ExprKind::Path { .. });
        let initialized = state.initialized.contains(&root);
        let mutable =
            state.mutable.get(&root).copied().unwrap_or(false) || self.place_through_mut_ref(place);

        if initialized && !mutable {
            self.assignment_error(assignment_span);
        } else if !initialized && !direct {
            self.require_initialized(root, self.hir.expr(place).span, state);
        } else if direct {
            state.initialized.insert(root);
        }
    }

    fn assignment_error(&mut self, span: crate::source::Span) {
        self.diagnostics.push(
            Diagnostic::error("assignment to immutable place", span)
                .with_code("E0400")
                .with_label("cannot assign to this immutable variable or field"),
        );
    }

    fn root_local(&self, expr_id: ExprId) -> Option<LocalId> {
        match &self.hir.expr(expr_id).kind {
            hir::ExprKind::Path {
                res: Res::Local(local) | Res::SelfValue(local),
                ..
            } => Some(*local),
            hir::ExprKind::Field { base, .. }
            | hir::ExprKind::TupleField { base, .. }
            | hir::ExprKind::Index { base, .. } => self.root_local(*base),
            _ => None,
        }
    }

    fn place_through_mut_ref(&self, expr_id: ExprId) -> bool {
        match &self.hir.expr(expr_id).kind {
            hir::ExprKind::Unary {
                op: UnOp::Deref,
                operand,
            } => matches!(
                self.expr_types.get(operand),
                Some(Ty::Ref { mutable: true, .. })
            ),
            hir::ExprKind::Field { base, .. }
            | hir::ExprKind::TupleField { base, .. }
            | hir::ExprKind::Index { base, .. } => {
                matches!(
                    self.expr_types.get(base),
                    Some(Ty::Ref { mutable: true, .. })
                ) || self.place_through_mut_ref(*base)
            }
            _ => false,
        }
    }

    fn require_initialized(
        &mut self,
        local: LocalId,
        span: crate::source::Span,
        state: &FlowState,
    ) {
        if !state.initialized.contains(&local) {
            self.diagnostics.push(
                Diagnostic::error("use of possibly-uninitialized variable", span)
                    .with_code("E0401")
                    .with_label("variable read before it is definitely assigned"),
            );
        }
    }

    fn bind_pattern(&self, pat_id: PatId, state: &mut FlowState) -> Vec<LocalId> {
        let mut locals = Vec::new();
        self.collect_pattern_bindings(pat_id, &mut locals);
        for &local in &locals {
            state.initialized.insert(local);
            state.mutable.insert(local, false);
        }
        locals
    }

    fn collect_pattern_bindings(&self, pat_id: PatId, locals: &mut Vec<LocalId>) {
        match &self.hir.pat(pat_id).kind {
            hir::PatKind::Binding { local, .. } => locals.push(*local),
            hir::PatKind::TupleVariant { pats, .. }
            | hir::PatKind::Tuple(pats)
            | hir::PatKind::Array(pats) => {
                for &pat in pats {
                    self.collect_pattern_bindings(pat, locals);
                }
            }
            hir::PatKind::Struct { fields, .. } => {
                for field in fields {
                    if let Some(pat) = field.pat {
                        self.collect_pattern_bindings(pat, locals);
                    } else if let Some(local) = field.local {
                        locals.push(local);
                    }
                }
            }
            _ => {}
        }
    }
}
