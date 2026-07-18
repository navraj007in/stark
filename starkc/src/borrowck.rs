//! Borrow checker and ownership pass for STARK (PLAN.md M2.4).

use crate::ast::UnOp;
use crate::diag::Diagnostic;
use crate::hir::{
    self, BlockId, Builtin, CoreTrait, CoreType, ExprId, Hir, ItemId, LocalId, Res, StmtId,
};
use crate::source::{SourceFile, Span};
use crate::typecheck::Ty;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Clone, Debug)]
struct Borrow {
    local: LocalId,
    mutable: bool,
    _span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Projection {
    Field(u32, u32),
    TupleField(u32, u32),
    Index,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Place {
    local: LocalId,
    projections: Vec<Projection>,
}

pub struct BorrowChecker<'a> {
    hir: &'a Hir,
    file: Arc<SourceFile>,
    diags: Vec<Diagnostic>,
    expr_types: &'a HashMap<ExprId, Ty>,
    local_types: &'a HashMap<LocalId, Ty>,
    copy_types: HashSet<ItemId>,

    // Active borrow tracking
    active_borrows: Vec<Borrow>,
    // Moved variables tracking
    moved_places: HashSet<Place>,
}

pub fn check(
    hir: &Hir,
    file: Arc<SourceFile>,
    expr_types: &HashMap<ExprId, Ty>,
    local_types: &HashMap<LocalId, Ty>,
) -> Vec<Diagnostic> {
    let mut checker = BorrowChecker {
        hir,
        file,
        diags: Vec::new(),
        expr_types,
        local_types,
        copy_types: collect_copy_types(hir),
        active_borrows: Vec::new(),
        moved_places: HashSet::new(),
    };

    checker.check_crate();
    checker.diags
}

pub fn check_fn(
    hir: &Hir,
    file: Arc<SourceFile>,
    expr_types: &HashMap<ExprId, Ty>,
    local_types: &HashMap<LocalId, Ty>,
    def: &hir::FnDef,
) -> Vec<Diagnostic> {
    let mut checker = BorrowChecker {
        hir,
        file,
        diags: Vec::new(),
        expr_types,
        local_types,
        copy_types: collect_copy_types(hir),
        active_borrows: Vec::new(),
        moved_places: HashSet::new(),
    };
    checker.check_fn_def(def);
    checker.diags
}

pub fn check_snippet(
    hir: &Hir,
    file: Arc<SourceFile>,
    expr_types: &HashMap<ExprId, Ty>,
    local_types: &HashMap<LocalId, Ty>,
    stmts: &[StmtId],
    tail: Option<ExprId>,
) -> Vec<Diagnostic> {
    let mut checker = BorrowChecker {
        hir,
        file,
        diags: Vec::new(),
        expr_types,
        local_types,
        copy_types: collect_copy_types(hir),
        active_borrows: Vec::new(),
        moved_places: HashSet::new(),
    };
    for &stmt_id in stmts {
        checker.check_stmt(stmt_id);
    }
    if let Some(tail_id) = tail {
        checker.check_expr(tail_id);
        checker.check_return_escape(tail_id);
    }
    checker.diags
}

impl<'a> BorrowChecker<'a> {
    fn text(&self, span: Span) -> &str {
        &self.file.src[span.lo as usize..span.hi as usize]
    }

    /// WP-C1.4 (2026-07-17): backfill `.with_file()` when a diagnostic doesn't already carry one,
    /// mirroring resolve.rs's `push_diag` (WP-C1.2, DEV-006's resolve half) and typecheck.rs's
    /// per-item backfill. Every raw `self.diags.push(...)` call site in this file was converted
    /// to go through this method instead. See COMPILER-STATE.md DEV-006.
    fn push_diag(&mut self, diag: Diagnostic) {
        let diag = if diag.file.is_none() {
            diag.with_file(self.file.clone())
        } else {
            diag
        };
        self.diags.push(diag);
    }

    /// WP-C1.4: check an expression that is about to be consumed as an OWNED value -- bound to
    /// a `let`, returned, or produced as a block's tail value -- as opposed to `check_expr`'s
    /// general recursive traversal, which is also reached from read-only contexts (e.g. binary-
    /// operator operands via `check_read_expr`, or field/index access through a deref, which
    /// already have their own correct, narrower Copy checks) that must NOT be treated as moves.
    ///
    /// Before this fix, `place_of`/`consume_place` had no `Deref` case at all, so `*r` for
    /// `r: &T` was never checked as a move source in any context. For non-`Copy` `T` this meant
    /// `let owned = *r;` (or a function whose tail/return expression is `*r`) type-checked and
    /// then silently deep-cloned `T` out of borrowed storage at runtime (`interp.rs`'s `UnOp::
    /// Deref` evaluation uses `clone_place_value`, not move semantics) -- for a `Drop`-
    /// implementing `T`, this duplicates the value without an explicit `.clone()`/`impl Clone`
    /// ever being written, so its destructor then runs twice against what was conceptually one
    /// logical resource. Confirmed empirically before this fix, both at compile time (accepted)
    /// and at runtime (double-drop observed). This matches Rust's own rule (moving out of `*r`
    /// behind a shared OR mutable reference is rejected unless the pointee is `Copy`) rather
    /// than the narrower "Drop types only" reading of the checklist text, since the general
    /// rule is what the spec's Copy/reference model implies and is what makes the fix actually
    /// sound (a non-Copy, non-Drop type moved out from behind a reference is just as unsound --
    /// two owners now alias the same conceptual value -- even without a destructor to
    /// double-run). See COMPILER-STATE.md's WP-C1.4 findings.
    ///
    /// Also wired into every other position that builds a new owned value out of a
    /// sub-expression: call arguments (`take(*r)`), tuple/array elements (`(*r, 1)`), and
    /// struct-literal field values (`S { field: *r }`) -- each confirmed empirically to exhibit
    /// the identical double-drop pattern before being covered here. Free-function call arguments
    /// need no callee-signature awareness to apply this safely: STARK has no argument-position
    /// auto-ref/deref-coercion (only method *receivers* auto-borrow, per 03-Type-System.md), so
    /// `*r` in argument position only ever type-checks when the callee's parameter type already
    /// equals the pointee type by value -- confirmed empirically that passing `*r` where a `&T`
    /// parameter is expected is already a type error independent of this check.
    fn check_owned_value(&mut self, expr_id: ExprId) {
        if let hir::ExprKind::Unary {
            op: UnOp::Deref,
            operand,
        } = &self.hir.expr(expr_id).kind
        {
            let operand = *operand;
            let pointee_ty = self.expr_types.get(&expr_id).cloned().unwrap_or(Ty::Error);
            if !self.is_copy_type(&pointee_ty) {
                self.push_diag(
                    Diagnostic::error(
                        "cannot move a non-Copy value out of a reference",
                        self.hir.expr(expr_id).span,
                    )
                    .with_code("E0100")
                    .with_label("borrow this value instead of moving out of the reference"),
                );
            }
            self.check_expr(operand);
            return;
        }
        self.check_expr(expr_id);
    }

    fn is_copy_type(&self, ty: &Ty) -> bool {
        match ty {
            Ty::Primitive(primitive) => !matches!(
                primitive,
                crate::ast::Primitive::String | crate::ast::Primitive::Str
            ),
            Ty::Error | Ty::Ref { mutable: false, .. } => true,
            Ty::Struct(item, _) | Ty::Enum(item, _) => self.copy_types.contains(item),
            Ty::Core(CoreType::Option | CoreType::Result, args) => {
                args.iter().all(|arg| self.is_copy_type(arg))
            }
            Ty::Tuple(elements) => elements.iter().all(|element| self.is_copy_type(element)),
            Ty::Array(element, _) => self.is_copy_type(element),
            _ => false,
        }
    }

    fn check_crate(&mut self) {
        // WP-C1.4 (2026-07-17): swap `self.file` to each item's own file (from
        // `hir.item_files`, the same side table resolve.rs/typecheck.rs already use) before
        // checking it, so `push_diag`'s backfill attributes diagnostics to the file the item
        // actually came from rather than always the crate's top-level file -- for multi-file
        // packages, every borrow-check diagnostic for a non-root-file item was previously
        // misattributed to the root file. See COMPILER-STATE.md DEV-006.
        let root_file = self.file.clone();
        for (index, item) in self.hir.items.iter().enumerate() {
            let item_id = ItemId(index as u32);
            self.file = self
                .hir
                .item_files
                .get(&item_id)
                .cloned()
                .unwrap_or_else(|| root_file.clone());
            match &item.kind {
                hir::ItemKind::Fn(def) => {
                    self.check_fn_def(def);
                }
                hir::ItemKind::Impl { items, .. } => {
                    for impl_item in items {
                        if let hir::ImplItem::Fn { def, .. } = impl_item {
                            self.check_fn_def(def);
                        }
                    }
                }
                _ => {}
            }
        }
        self.file = root_file;

        // Snippet mode check
        if let hir::Root::Snippet { stmts, tail } = &self.hir.root {
            self.moved_places.clear();
            self.active_borrows.clear();
            for &stmt_id in stmts {
                self.check_stmt(stmt_id);
            }
            if let Some(tail_id) = tail {
                self.check_expr(*tail_id);
            }
        }
    }

    fn check_fn_def(&mut self, def: &hir::FnDef) {
        self.moved_places.clear();
        self.active_borrows.clear();

        // Parameters are initially owned/borrowed (not moved)
        self.check_block(def.body);
    }

    fn check_block(&mut self, block_id: BlockId) {
        let block = self.hir.block(block_id);

        // Record borrows count to pop block-local borrows at the end
        let borrows_before = self.active_borrows.len();

        for &stmt_id in &block.stmts {
            self.check_stmt(stmt_id);
        }

        if let Some(tail_expr) = block.tail {
            self.check_owned_value(tail_expr);
            self.check_return_escape(tail_expr);
        }

        // Pop block-scoped borrows
        self.active_borrows.truncate(borrows_before);
    }

    fn check_stmt(&mut self, stmt_id: StmtId) {
        let stmt = self.hir.stmt(stmt_id);
        match &stmt.kind {
            hir::StmtKind::Expr { expr, .. } => {
                let borrows_before = self.active_borrows.len();
                self.check_expr(*expr);
                self.active_borrows.truncate(borrows_before);
            }
            hir::StmtKind::Let { local, init, .. } => {
                let borrows_before = self.active_borrows.len();
                if let Some(init_expr) = init {
                    self.check_owned_value(*init_expr);
                    self.reinitialize(&Place {
                        local: *local,
                        projections: Vec::new(),
                    });
                    let carries_borrow = self.expr_carries_borrow(*init_expr);
                    if !carries_borrow {
                        self.active_borrows.truncate(borrows_before);
                    }
                } else {
                    self.active_borrows.truncate(borrows_before);
                }
            }
            hir::StmtKind::Return(Some(expr)) => {
                self.check_owned_value(*expr);
                self.check_return_escape(*expr);
            }
            hir::StmtKind::Break(Some(expr)) => {
                self.check_expr(*expr);
            }
            hir::StmtKind::Return(None) | hir::StmtKind::Break(None) => {}
            hir::StmtKind::Continue => {}
            _ => {}
        }
    }

    fn check_expr(&mut self, expr_id: ExprId) {
        let expr = self.hir.expr(expr_id);
        match &expr.kind {
            hir::ExprKind::Path { res, .. } => {
                if matches!(res, Res::Local(_) | Res::SelfValue(_)) {
                    self.consume_place(expr_id);
                }
            }
            hir::ExprKind::Unary { op, operand } => {
                if let UnOp::Ref { mutable } = op {
                    // Borrow creation!
                    if let Some(place) = self.place_of(*operand) {
                        let local_id = place.local;
                        self.check_place_available(&place, self.hir.expr(*operand).span);
                        // Check conflicts
                        let mut has_conflict = false;
                        for b in &self.active_borrows {
                            if b.local == local_id && (*mutable || b.mutable) {
                                self.push_diag(
                                    Diagnostic::error(format!("cannot borrow variable '{}' because it is already borrowed", self.text(expr.span)), expr.span)
                                        .with_code("E0101")
                                );
                                has_conflict = true;
                                break;
                            }
                        }
                        if !has_conflict {
                            self.active_borrows.push(Borrow {
                                local: local_id,
                                mutable: *mutable,
                                _span: expr.span,
                            });
                        }
                    } else {
                        self.check_expr(*operand);
                    }
                } else {
                    self.check_expr(*operand);
                }
            }
            hir::ExprKind::Binary { lhs, rhs, .. } => {
                self.check_read_expr(*lhs);
                self.check_read_expr(*rhs);
            }
            hir::ExprKind::Range {
                lo: lhs, hi: rhs, ..
            } => {
                self.check_expr(*lhs);
                self.check_expr(*rhs);
            }
            hir::ExprKind::Assign { lhs, rhs, .. } => {
                self.check_expr(*rhs);

                // Write check: verify no active borrows on the place
                if let Some(local_id) = self.get_root_local(*lhs) {
                    for b in &self.active_borrows {
                        if b.local == local_id {
                            self.push_diag(
                                Diagnostic::error(
                                    format!(
                                        "cannot assign to variable '{}' because it is borrowed",
                                        self.text(self.hir.expr(*lhs).span)
                                    ),
                                    self.hir.expr(*lhs).span,
                                )
                                .with_code("E0101")
                                .with_label("assignment conflict: variable is currently borrowed"),
                            );
                            break;
                        }
                    }
                    if let Some(place) = self.place_of(*lhs) {
                        self.reinitialize(&place);
                    }
                } else {
                    self.check_expr(*lhs);
                }
            }
            hir::ExprKind::Call { callee, args } => {
                if let hir::ExprKind::Field { base, name, .. } = &self.hir.expr(*callee).kind {
                    if self.text(*name) == "refine" {
                        self.consume_place(*base);
                    } else {
                        match self.method_receiver(*base, *name) {
                            Some(hir::Receiver::Value) => self.consume_place(*base),
                            Some(hir::Receiver::Ref) => self.borrow_method_receiver(*base, false),
                            Some(hir::Receiver::RefMut) => self.borrow_method_receiver(*base, true),
                            None => self.check_expr(*base),
                        }
                    }
                } else {
                    self.check_expr(*callee);
                }
                for &arg in args {
                    self.check_owned_value(arg);
                }
            }
            hir::ExprKind::Field { .. } | hir::ExprKind::TupleField { .. } => {
                self.consume_place(expr_id);
            }
            hir::ExprKind::Index { base, index } => {
                if let Some(place) = self.place_of(*base) {
                    self.check_place_available(&place, self.hir.expr(*base).span);
                } else {
                    self.check_expr(*base);
                }
                self.check_expr(*index);
                let ty = self.expr_types.get(&expr_id).cloned().unwrap_or(Ty::Error);
                if !matches!(ty, Ty::Slice(_)) && !self.is_copy_type(&ty) {
                    self.push_diag(
                        Diagnostic::error(
                            "cannot move a non-Copy value out of an indexed place",
                            expr.span,
                        )
                        .with_code("E0100")
                        .with_label("use an ownership-transferring collection method instead"),
                    );
                }
            }
            hir::ExprKind::Tuple(elems) | hir::ExprKind::Array(elems) => {
                for &e in elems {
                    self.check_owned_value(e);
                }
            }
            hir::ExprKind::StructLit { fields, .. } => {
                for f in fields {
                    if let Some(val) = f.expr {
                        self.check_owned_value(val);
                    }
                }
            }
            hir::ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                self.check_expr(*cond);

                let moved_before = self.moved_places.clone();
                self.check_block(*then_block);
                let moved_then = self.moved_places.clone();

                if let Some(else_expr) = else_ {
                    self.moved_places = moved_before;
                    self.check_expr(*else_expr);
                    let moved_else = self.moved_places.clone();

                    // Merged moves: union of both branches
                    self.moved_places = moved_then.union(&moved_else).cloned().collect();
                } else {
                    // No else branch: variables moved in then branch might be used if then branch is skipped,
                    // so we restore state before if (moved_before), but wait, in STARK, if a variable is moved
                    // in one branch but not the other, it is considered moved after the block.
                    self.moved_places = moved_then;
                }
            }
            hir::ExprKind::Match { scrutinee, arms } => {
                self.check_expr(*scrutinee);
                let moved_before = self.moved_places.clone();
                let mut merged_moved = HashSet::new();
                for arm in arms {
                    self.moved_places = moved_before.clone();
                    self.check_expr(arm.body);
                    merged_moved.extend(self.moved_places.iter().cloned());
                }
                self.moved_places = merged_moved;
            }
            hir::ExprKind::Block(b) => {
                self.check_block(*b);
            }
            hir::ExprKind::Loop { body } => {
                self.check_block(*body);
            }
            hir::ExprKind::While { cond, body } => {
                self.check_expr(*cond);
                self.check_block(*body);
            }
            hir::ExprKind::For { iter, body, .. } => {
                self.check_expr(*iter);
                self.check_block(*body);
            }
            hir::ExprKind::Try(expr) => {
                self.check_expr(*expr);
            }
            hir::ExprKind::Cast {
                expr: cast_expr, ..
            } => {
                self.check_expr(*cast_expr);
            }
            hir::ExprKind::Repeat { value, count } => {
                self.check_expr(*value);
                self.check_expr(*count);
            }
            _ => {}
        }
    }

    fn get_root_local(&self, expr_id: ExprId) -> Option<LocalId> {
        let expr = self.hir.expr(expr_id);
        match &expr.kind {
            hir::ExprKind::Path {
                res: Res::Local(local_id) | Res::SelfValue(local_id),
                ..
            } => Some(*local_id),
            hir::ExprKind::Field { base, .. }
            | hir::ExprKind::TupleField { base, .. }
            | hir::ExprKind::Index { base, .. } => self.get_root_local(*base),
            _ => None,
        }
    }

    fn method_receiver(&self, base: ExprId, name: Span) -> Option<hir::Receiver> {
        let mut base_ty = self.expr_types.get(&base)?.clone();
        while let Ty::Ref { inner, .. } = base_ty {
            base_ty = *inner;
        }
        let method_name = self.text(name);
        if matches!(
            base_ty,
            Ty::Primitive(crate::ast::Primitive::String | crate::ast::Primitive::Str)
                | Ty::Core(..)
        ) {
            return Some(
                if matches!(
                    method_name,
                    "push"
                        | "push_str"
                        | "pop"
                        | "clear"
                        | "insert"
                        | "remove"
                        | "append"
                        | "get_mut"
                        | "next"
                        | "read_to_string"
                        | "write"
                        | "write_str"
                ) {
                    hir::Receiver::RefMut
                } else if matches!(method_name, "unwrap" | "unwrap_or" | "into_inner" | "close") {
                    hir::Receiver::Value
                } else {
                    hir::Receiver::Ref
                },
            );
        }
        if let Ty::Extension(ext) = &base_ty {
            if matches!(ext.as_ref(), crate::typecheck::ExtensionTy::Model(_))
                && method_name == "predict"
            {
                return Some(hir::Receiver::Ref);
            }
            if matches!(ext.as_ref(), crate::typecheck::ExtensionTy::Tensor(_))
                && matches!(
                    method_name,
                    "add"
                        | "sub"
                        | "mul"
                        | "div"
                        | "min"
                        | "max"
                        | "eq"
                        | "ne"
                        | "lt"
                        | "le"
                        | "gt"
                        | "ge"
                        | "broadcast_to"
                        | "matmul"
                        | "batch_matmul"
                        | "concat"
                        | "permute"
                        | "reshape"
                        | "slice_axis"
                        | "transpose"
                        | "sum_axis"
                        | "mean_axis"
                        | "argmax"
                        | "sum"
                        | "softmax"
                        | "cast"
                        | "to_device"
                )
            {
                return Some(hir::Receiver::Ref);
            }
        }

        let item_id = match base_ty {
            Ty::Struct(item, _) | Ty::Enum(item, _) => item,
            _ => return None,
        };
        self.hir.items.iter().find_map(|item| {
            let hir::ItemKind::Impl { self_ty, items, .. } = &item.kind else {
                return None;
            };
            let matches_type = matches!(
                self.hir.ty(*self_ty).kind,
                hir::TypeKind::Path {
                    res: Res::Item(impl_item),
                    ..
                } if impl_item == item_id
            );
            if !matches_type {
                return None;
            }
            items.iter().find_map(|item| match item {
                hir::ImplItem::Fn { def, .. } if self.text(def.sig.name) == method_name => {
                    def.sig.receiver
                }
                _ => None,
            })
        })
    }

    fn borrow_method_receiver(&mut self, base: ExprId, mutable: bool) {
        let Some(place) = self.place_of(base) else {
            self.check_expr(base);
            return;
        };
        let span = self.hir.expr(base).span;
        self.check_place_available(&place, span);
        if self
            .active_borrows
            .iter()
            .any(|borrow| borrow.local == place.local && (mutable || borrow.mutable))
        {
            self.push_diag(
                Diagnostic::error("method receiver conflicts with an active borrow", span)
                    .with_code("E0101"),
            );
        } else {
            self.active_borrows.push(Borrow {
                local: place.local,
                mutable,
                _span: span,
            });
        }
    }

    fn place_of(&self, expr_id: ExprId) -> Option<Place> {
        let expr = self.hir.expr(expr_id);
        match &expr.kind {
            hir::ExprKind::Path {
                res: Res::Local(local) | Res::SelfValue(local),
                ..
            } => Some(Place {
                local: *local,
                projections: Vec::new(),
            }),
            hir::ExprKind::Field { base, name, .. } => {
                let mut place = self.place_of(*base)?;
                place.projections.push(Projection::Field(name.lo, name.hi));
                Some(place)
            }
            hir::ExprKind::TupleField { base, index } => {
                let mut place = self.place_of(*base)?;
                place
                    .projections
                    .push(Projection::TupleField(index.lo, index.hi));
                Some(place)
            }
            hir::ExprKind::Index { base, .. } => {
                let mut place = self.place_of(*base)?;
                place.projections.push(Projection::Index);
                Some(place)
            }
            _ => None,
        }
    }

    fn consume_place(&mut self, expr_id: ExprId) {
        let expr = self.hir.expr(expr_id);
        let Some(place) = self.place_of(expr_id) else {
            match &expr.kind {
                hir::ExprKind::Field { base, .. } | hir::ExprKind::TupleField { base, .. } => {
                    self.check_expr(*base);
                }
                _ => {}
            }
            return;
        };

        if !self.check_place_available(&place, expr.span) {
            return;
        }
        self.check_read_borrow_conflict(place.local, expr.span);

        let ty = self.expr_types.get(&expr_id).cloned().unwrap_or(Ty::Error);
        if self.is_copy_type(&ty) {
            return;
        }
        // WP-C1.4: `check_read_borrow_conflict` above only rejects reads against a *mutable*
        // borrow (shared reads under a shared borrow are sound). A move is different: moving a
        // non-Copy place invalidates its storage, so it must be rejected under *any* active
        // borrow of that local, mutable or shared -- e.g. `let it = v.iter(); consume(v);` must
        // not compile, since `it` is a live shared view into `v`'s storage. Confirmed empirically
        // that this previously compiled and crashed at runtime; see COMPILER-STATE.md's WP-C1.4
        // findings.
        if self.active_borrows.iter().any(|b| b.local == place.local) {
            self.push_diag(
                Diagnostic::error(
                    format!(
                        "cannot move variable '{}' because it is borrowed",
                        self.text(expr.span)
                    ),
                    expr.span,
                )
                .with_code("E0101")
                .with_label("move conflict: variable is currently borrowed"),
            );
            return;
        }
        if place
            .projections
            .iter()
            .any(|projection| matches!(projection, Projection::Index))
        {
            self.push_diag(
                Diagnostic::error(
                    "cannot move a non-Copy value out of an indexed place",
                    expr.span,
                )
                .with_code("E0100"),
            );
            return;
        }
        if !place.projections.is_empty() && self.local_has_drop(place.local) {
            self.push_diag(
                Diagnostic::error(
                    "cannot partially move a value whose type implements Drop",
                    expr.span,
                )
                .with_code("E0100")
                .with_label("move the whole value or borrow this field"),
            );
            return;
        }
        self.moved_places.insert(place);
    }

    fn check_read_expr(&mut self, expr_id: ExprId) {
        if let Some(place) = self.place_of(expr_id) {
            if self.check_place_available(&place, self.hir.expr(expr_id).span) {
                self.check_read_borrow_conflict(place.local, self.hir.expr(expr_id).span);
            }
        } else {
            self.check_expr(expr_id);
        }
    }

    fn check_place_available(&mut self, place: &Place, span: Span) -> bool {
        if self
            .moved_places
            .iter()
            .any(|moved| places_overlap(moved, place))
        {
            self.push_diag(
                Diagnostic::error(format!("use of moved value '{}'", self.text(span)), span)
                    .with_code("E0100")
                    .with_label("value used here after move"),
            );
            false
        } else {
            true
        }
    }

    fn check_read_borrow_conflict(&mut self, local: LocalId, span: Span) {
        if self
            .active_borrows
            .iter()
            .any(|borrow| borrow.local == local && borrow.mutable)
        {
            self.push_diag(
                Diagnostic::error(
                    format!(
                        "cannot read variable '{}' because it is mutably borrowed",
                        self.text(span)
                    ),
                    span,
                )
                .with_code("E0101")
                .with_label("read conflict: variable is currently mutably borrowed"),
            );
        }
    }

    fn reinitialize(&mut self, place: &Place) {
        if place.projections.is_empty() {
            self.moved_places.retain(|moved| moved.local != place.local);
        } else {
            self.moved_places.retain(|moved| {
                moved.local != place.local || !is_prefix(&place.projections, &moved.projections)
            });
        }
    }

    fn local_has_drop(&self, local: LocalId) -> bool {
        let Some(ty) = self.local_types.get(&local) else {
            return false;
        };
        let item_id = match ty {
            Ty::Struct(id, _) | Ty::Enum(id, _) => *id,
            _ => return false,
        };
        self.hir.items.iter().any(|item| {
            let hir::ItemKind::Impl {
                trait_: Some(trait_ref),
                self_ty,
                ..
            } = &item.kind
            else {
                return false;
            };
            let is_drop = self.text(trait_ref.path.span).ends_with("Drop");
            let matches_type = matches!(
                &self.hir.ty(*self_ty).kind,
                hir::TypeKind::Path {
                    res: Res::Item(id),
                    ..
                } if *id == item_id
            );
            is_drop && matches_type
        })
    }

    fn check_return_escape(&mut self, expr_id: ExprId) {
        if let Some(local_id) = self.borrowed_local(expr_id) {
            let local_ty = self
                .local_types
                .get(&local_id)
                .cloned()
                .unwrap_or(Ty::Error);
            if !matches!(local_ty, Ty::Ref { .. }) {
                self.push_diag(
                    Diagnostic::error(
                        "cannot return reference to local stack variable",
                        self.hir.expr(expr_id).span,
                    )
                    .with_code("E0103")
                    .with_label("reference to stack memory escapes function"),
                );
            }
        }
    }

    fn expr_carries_borrow(&self, expr_id: ExprId) -> bool {
        if self
            .expr_types
            .get(&expr_id)
            .is_some_and(Self::type_carries_borrow)
        {
            return true;
        }
        let expr = self.hir.expr(expr_id);
        match &expr.kind {
            hir::ExprKind::Tuple(elems) | hir::ExprKind::Array(elems) => {
                elems.iter().any(|expr| self.expr_carries_borrow(*expr))
            }
            hir::ExprKind::Repeat { value, .. }
            | hir::ExprKind::Try(value)
            | hir::ExprKind::Cast { expr: value, .. } => self.expr_carries_borrow(*value),
            hir::ExprKind::StructLit { fields, .. } => fields
                .iter()
                .filter_map(|field| field.expr)
                .any(|expr| self.expr_carries_borrow(expr)),
            hir::ExprKind::Call { callee, args }
                if matches!(
                    self.hir.expr(*callee).kind,
                    hir::ExprKind::Path {
                        res: Res::Variant(..)
                            | Res::Builtin(Builtin::Some | Builtin::Ok | Builtin::Err),
                        ..
                    }
                ) =>
            {
                args.iter().any(|expr| self.expr_carries_borrow(*expr))
            }
            hir::ExprKind::If {
                then_block, else_, ..
            } => {
                self.block_carries_borrow(*then_block)
                    || else_.is_some_and(|expr| self.expr_carries_borrow(expr))
            }
            hir::ExprKind::Match { arms, .. } => {
                arms.iter().any(|arm| self.expr_carries_borrow(arm.body))
            }
            hir::ExprKind::Block(block) => self.block_carries_borrow(*block),
            _ => false,
        }
    }

    fn block_carries_borrow(&self, block_id: BlockId) -> bool {
        self.hir
            .block(block_id)
            .tail
            .is_some_and(|expr| self.expr_carries_borrow(expr))
    }

    fn type_carries_borrow(ty: &Ty) -> bool {
        match ty {
            Ty::Ref { .. } => true,
            Ty::Tuple(elements) => elements.iter().any(Self::type_carries_borrow),
            Ty::Array(element, _) | Ty::Slice(element) | Ty::Range(element) => {
                Self::type_carries_borrow(element)
            }
            // WP-C1.4: iterator CoreTypes are borrow-carrying VIEWS of their source collection
            // regardless of their element type argument -- `VecIter<Int32>`'s only generic arg
            // is the *element* type (Int32, not a reference), so the generic "recurse into args
            // looking for a Ty::Ref" rule below never recognized these as borrow-carrying at
            // all. Before this fix, `let it = v.iter();` computed `expr_carries_borrow` as false
            // for the init expression, so the shared borrow `borrow_method_receiver` registers
            // while evaluating `v.iter()` was immediately truncated at end of that `let`
            // statement -- the checker believed `v` was no longer borrowed for the rest of the
            // block, even though `it` is a live, aliasing view into `v`'s storage. Confirmed
            // empirically: moving `v` into another function while `it` was still live compiled
            // and then crashed at runtime ("use of unavailable value"). See COMPILER-STATE.md's
            // WP-C1.4 findings.
            Ty::Core(
                CoreType::VecIter
                | CoreType::CharsIter
                | CoreType::SplitIter
                | CoreType::KeysIter
                | CoreType::ValuesIter
                | CoreType::Iter
                | CoreType::MapIter
                | CoreType::FilterIter,
                _,
            ) => true,
            Ty::Struct(_, args) | Ty::Enum(_, args) | Ty::Core(_, args) => {
                args.iter().any(Self::type_carries_borrow)
            }
            Ty::Fn { params, ret } => {
                params.iter().any(Self::type_carries_borrow) || Self::type_carries_borrow(ret)
            }
            _ => false,
        }
    }

    fn borrowed_local(&self, expr_id: ExprId) -> Option<LocalId> {
        let expr = self.hir.expr(expr_id);
        match &expr.kind {
            hir::ExprKind::Unary {
                op: UnOp::Ref { .. },
                operand,
            } => self.get_root_local(*operand),
            hir::ExprKind::Tuple(elems) | hir::ExprKind::Array(elems) => {
                elems.iter().find_map(|expr| self.borrowed_local(*expr))
            }
            hir::ExprKind::Repeat { value, .. }
            | hir::ExprKind::Try(value)
            | hir::ExprKind::Cast { expr: value, .. } => self.borrowed_local(*value),
            hir::ExprKind::StructLit { fields, .. } => fields
                .iter()
                .filter_map(|field| field.expr)
                .find_map(|expr| self.borrowed_local(expr)),
            hir::ExprKind::Call { args, .. }
                if self
                    .expr_types
                    .get(&expr_id)
                    .is_some_and(Self::type_carries_borrow) =>
            {
                // A call returning a borrow-carrying aggregate is conservatively
                // tied to its borrowed arguments.  This closes the escape hole in
                // wrappers such as `Some(identity(&local))` and user functions
                // returning `Option<&T>`.
                args.iter().find_map(|expr| self.borrowed_local(*expr))
            }
            hir::ExprKind::If {
                then_block, else_, ..
            } => self
                .borrowed_local_from_block(*then_block)
                .or_else(|| else_.and_then(|expr| self.borrowed_local(expr))),
            hir::ExprKind::Match { arms, .. } => {
                arms.iter().find_map(|arm| self.borrowed_local(arm.body))
            }
            hir::ExprKind::Block(block) => self.borrowed_local_from_block(*block),
            _ => None,
        }
    }

    fn borrowed_local_from_block(&self, block_id: BlockId) -> Option<LocalId> {
        self.hir
            .block(block_id)
            .tail
            .and_then(|expr| self.borrowed_local(expr))
    }
}

fn collect_copy_types(hir: &Hir) -> HashSet<ItemId> {
    hir.items
        .iter()
        .filter_map(|item| match &item.kind {
            hir::ItemKind::Impl {
                trait_:
                    Some(hir::TraitRef {
                        res: Res::CoreTrait(CoreTrait::Copy),
                        ..
                    }),
                self_ty,
                ..
            } => match &hir.ty(*self_ty).kind {
                hir::TypeKind::Path {
                    res: Res::Item(item),
                    ..
                } => Some(*item),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

fn is_prefix(prefix: &[Projection], value: &[Projection]) -> bool {
    prefix.len() <= value.len() && prefix.iter().zip(value).all(|(a, b)| a == b)
}

fn places_overlap(left: &Place, right: &Place) -> bool {
    left.local == right.local
        && (is_prefix(&left.projections, &right.projections)
            || is_prefix(&right.projections, &left.projections))
}
