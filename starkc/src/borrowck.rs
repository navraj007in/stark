//! Borrow checker and ownership pass for STARK (PLAN.md M2.4).

use crate::ast::UnOp;
use crate::diag::Diagnostic;
use crate::hir::{self, BlockId, Builtin, CoreType, ExprId, Hir, ItemId, LocalId, Res, StmtId};
use crate::source::Span;
use crate::typecheck::Ty;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
struct Borrow {
    /// DEV-154: the full borrowed PLACE, not merely its root local.
    ///
    /// OWN-BORROW-001 says "Disjoint field projections do not overlap", and every comparison here
    /// used to test `b.local == local` — so a borrow of `p.a` blocked a read of `p.b`, refusing a
    /// valid program and contradicting the spec. `places_overlap` has been field-precise since
    /// DEV-135; only these comparisons never used it.
    place: Place,
    mutable: bool,
    _span: Span,
}

/// DEV-135: a projection identifies a field by its NAME, not by the span the name was written at.
///
/// This used to be `Field(u32, u32)` — the span's byte offsets. Two mentions of the same field sit
/// at different offsets, so `owner.handle` on one line and `owner.handle` on the next produced two
/// DIFFERENT projections that `places_overlap` then correctly reported as disjoint.
///
/// The move set was ALREADY field-precise — `places_overlap` does prefix matching, so moving
/// `pair.left` correctly left `pair.right` live and moving the parent afterwards was correctly
/// refused. What was broken was field IDENTITY alone, so a field could be moved out twice and the
/// second move was invisible to the front end. The HIR oracle then failed at run time with
/// "internal compiler error: use of moved or invalid field" — the wrong category for a
/// user-authored program, and several layers late.
///
/// Same class as DEV-122: identity taken from a span rather than from what the span denotes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Projection {
    Field(String),
    TupleField(String),
    Index,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Place {
    local: LocalId,
    projections: Vec<Projection>,
}

pub struct BorrowChecker<'a> {
    hir: &'a Hir,
    diags: Vec<Diagnostic>,
    expr_types: &'a HashMap<ExprId, Ty>,
    local_types: &'a HashMap<LocalId, Ty>,
    copy_types: HashSet<ItemId>,
    /// Nominals with a user destructor, by RESOLVED identity (DEV-210). Read from the checker's
    /// authority rather than rescanned here.
    drop_items: HashSet<ItemId>,

    // Active borrow tracking
    active_borrows: Vec<Borrow>,
    // Moved variables tracking
    moved_places: HashSet<Place>,
    /// DEV-150: spans already reported as a call-argument overlap, so the left-to-right walk that
    /// follows does not report the same read a second time in different words.
    overlap_reported: HashSet<(u32, u32)>,
    /// DEV-DISPLAY-DISPATCH: the generic parameters in scope for the body being checked — the
    /// function's own, plus the enclosing impl's (WP-C6.2b-F5). `method_receiver` needs them to
    /// answer "what receiver does `x.fmt()` take" when `x`'s type is a bounded parameter: the
    /// answer is on the BOUND, and without it the receiver defaulted to by-value and every
    /// `&self` trait method silently MOVED its receiver.
    current_generics: Vec<hir::GenericParam>,
    /// The enclosing impl's generic parameters, set while its methods are checked.
    enclosing_generics: Vec<hir::GenericParam>,
}

pub fn check(
    hir: &Hir,
    expr_types: &HashMap<ExprId, Ty>,
    local_types: &HashMap<LocalId, Ty>,
) -> Vec<Diagnostic> {
    let mut checker = BorrowChecker {
        hir,
        diags: Vec::new(),
        expr_types,
        local_types,
        copy_types: collect_copy_types(hir),
        drop_items: crate::typecheck::nominals_with_destructor(hir),
        active_borrows: Vec::new(),
        moved_places: HashSet::new(),
        overlap_reported: HashSet::new(),
        current_generics: Vec::new(),
        enclosing_generics: Vec::new(),
    };

    checker.check_crate();
    checker.diags
}

pub fn check_fn(
    hir: &Hir,
    expr_types: &HashMap<ExprId, Ty>,
    local_types: &HashMap<LocalId, Ty>,
    def: &hir::FnDef,
) -> Vec<Diagnostic> {
    let mut checker = BorrowChecker {
        hir,
        diags: Vec::new(),
        expr_types,
        local_types,
        copy_types: collect_copy_types(hir),
        drop_items: crate::typecheck::nominals_with_destructor(hir),
        active_borrows: Vec::new(),
        moved_places: HashSet::new(),
        overlap_reported: HashSet::new(),
        current_generics: Vec::new(),
        enclosing_generics: Vec::new(),
    };
    checker.check_fn_def(def);
    checker.diags
}

pub fn check_snippet(
    hir: &Hir,
    expr_types: &HashMap<ExprId, Ty>,
    local_types: &HashMap<LocalId, Ty>,
    stmts: &[StmtId],
    tail: Option<ExprId>,
) -> Vec<Diagnostic> {
    let mut checker = BorrowChecker {
        hir,
        diags: Vec::new(),
        expr_types,
        local_types,
        copy_types: collect_copy_types(hir),
        drop_items: crate::typecheck::nominals_with_destructor(hir),
        active_borrows: Vec::new(),
        moved_places: HashSet::new(),
        overlap_reported: HashSet::new(),
        current_generics: Vec::new(),
        enclosing_generics: Vec::new(),
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
    /// Read a span, against the source the SPAN NAMES.
    ///
    /// AS1b-ii-d: this used to slice `self.file`, "the file of the item being checked", which
    /// `check_crate` re-aimed per item (WP-C1.4, DEV-069) — right for that item's spans and wrong
    /// for every other item's, hence the separate `item_text`. Both are one read now.
    /// Non-panicking since WP-C4.7-4: an out-of-range span was a compiler crash before.
    fn text(&self, span: Span) -> &str {
        self.hir
            .sources
            .get(span.source)
            .and_then(|file| file.src.get(span.lo as usize..span.hi as usize))
            .unwrap_or("?")
    }

    /// AS1b-ii-d: the item is no longer consulted — `span` names its own source.
    fn item_text(&self, _item: hir::ItemId, span: Span) -> &str {
        self.text(span)
    }

    /// WP-C1.4 (2026-07-17) routed every `self.diags.push(...)` in this file through here so a
    /// diagnostic could be stamped with the file being checked (DEV-006).
    ///
    /// AS1b-ii-d removed the stamp: a span names its own source, so attribution is no longer a
    /// second thing that can be right or wrong. The funnel stays.
    fn push_diag(&mut self, diag: Diagnostic) {
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
            // DEV-066: reading through the reference must not consume it.
            if let Some(place) = self.place_of(operand) {
                self.check_place_available(&place, self.hir.expr(operand).span);
            } else {
                self.check_expr(operand);
            }
            return;
        }
        // DEV-232. The arm above covers a bare `*r`. A FIELD read through the same reference --
        // `t.v` for `t: &T` -- produces an owned non-`Copy` value just as surely, and was not
        // covered: `stark check` accepted it, the interpreter raised `internal compiler error:
        // use of moved or invalid field`, and native leaked an internal `Place` description. A
        // function that only borrows its argument destroyed the caller's value and nothing said
        // so.
        //
        // The rule was already implemented for the PATTERN case by DEV-072
        // (`reject_moves_out_of_borrow`); this is the same prohibition in expression position, and
        // it reuses DEV-072's own classifier so the two cannot drift apart.
        //
        // A `Copy` field is untouched, because a `Copy` read moves nothing -- `fn peek(t: &T) ->
        // Int64 { t.v }` stays legal, which is why the check is on the VALUE's type rather than on
        // the shape alone.
        if matches!(
            &self.hir.expr(expr_id).kind,
            hir::ExprKind::Field { .. } | hir::ExprKind::TupleField { .. }
        ) {
            let value_ty = self.expr_types.get(&expr_id).cloned().unwrap_or(Ty::Error);
            if !self.is_copy_type(&value_ty) && self.scrutinee_reads_through_ref(expr_id) {
                self.push_diag(
                    Diagnostic::error(
                        "cannot move a non-Copy value out of a reference",
                        self.hir.expr(expr_id).span,
                    )
                    .with_code("E0100")
                    .with_label("borrow this field instead of moving it out of the reference"),
                );
            }
        }
        // WP-C6.1f-b2: an argument of type `&mut T` **re-borrows** rather than moving.
        //
        // A parameter is an expected-type boundary, and 03-Type-System's reference coercions make
        // `&mut T -> &T` normative there — but even passing `&mut T` unchanged must not consume the
        // caller's reference, or `f(m); f(m);` would be rejected. 03 "References and Lifetimes"
        // rule 4 settles the duration: a borrow not bound to a variable "ends at the end of its
        // enclosing statement, so `f(&x); g(&mut x);` is legal". The re-borrow therefore lives for
        // the call and no longer, which is exactly the property b1 relied on for receivers.
        //
        // Exclusivity is unchanged: the source stays frozen for the statement, and
        // `check_place_available` still rejects a re-borrow of an already-moved place. Only the
        // spurious *move* is removed.
        if matches!(
            self.expr_types.get(&expr_id),
            Some(Ty::Ref { mutable: true, .. })
        ) {
            if let Some(place) = self.place_of(expr_id) {
                self.check_place_available(&place, self.hir.expr(expr_id).span);
                return;
            }
        }
        self.check_expr(expr_id);
    }

    /// Whether a match scrutinee is read through a reference. Deliberately IDENTICAL to MIR
    /// lowering's `scrutinee_reads_through_ref` (`mir/lower.rs`), so the two engines classify
    /// by-reference matching the same way by construction rather than by coincidence — the
    /// disagreement between them is exactly what DEV-072 was.
    ///
    /// Both shared and mutable derefs count: ownership cannot be moved out of either.
    fn scrutinee_reads_through_ref(&self, expr: ExprId) -> bool {
        match &self.hir.expr(expr).kind {
            hir::ExprKind::Unary {
                op: crate::ast::UnOp::Deref,
                ..
            } => true,
            hir::ExprKind::Field { base, .. } | hir::ExprKind::TupleField { base, .. } => {
                matches!(self.expr_types.get(base), Some(Ty::Ref { .. }))
                    || self.scrutinee_reads_through_ref(*base)
            }
            _ => false,
        }
    }

    /// DEV-072: report every binding in `pat` that would move a non-`Copy` value out of a
    /// borrowed scrutinee. Wildcards, literals, and path (unit-variant) patterns bind nothing
    /// and stay legal — matching by reference is fine, it is only *taking ownership* that is
    /// not — as does any `Copy` binding, which copies rather than moves.
    /// Whether the scrutinee's own type is a nominal carrying a user destructor.
    ///
    /// Reads the published authority (DEV-210), not a private scan.
    fn scrutinee_nominal_has_drop(&self, scrutinee: hir::ExprId) -> bool {
        match self.expr_types.get(&scrutinee) {
            Some(Ty::Struct(id, _) | Ty::Enum(id, _)) => self.drop_items.contains(id),
            _ => false,
        }
    }

    /// DEV-211's walk. Structurally the same as [`Self::reject_moves_out_of_borrow`] — a different
    /// reason for the same prohibition, so it is a sibling rather than a flag on that one: the
    /// diagnostics must say different things, and a shared walk with a mode parameter would make
    /// both messages harder to get right than either is separately.
    fn reject_moves_out_of_drop_scrutinee(&mut self, pat: hir::PatId) {
        let node = self.hir.pat(pat);
        let span = node.span;
        match &node.kind {
            hir::PatKind::Wild | hir::PatKind::Lit(_) | hir::PatKind::Path { .. } => {}
            hir::PatKind::Binding { local, .. } => {
                let is_non_copy = self
                    .local_types
                    .get(local)
                    .is_some_and(|ty| !self.is_copy_type(ty));
                if is_non_copy {
                    self.push_diag(
                        Diagnostic::error(
                            format!(
                                "cannot move '{}' out of a value whose type implements Drop: the \
                                 destructor requires the complete value (OWN-PARTIAL-001)",
                                self.text(span)
                            ),
                            span,
                        )
                        .with_code("E0100")
                        .with_label("match through a reference, or bind a Copy component"),
                    );
                }
            }
            hir::PatKind::TupleVariant { pats, .. }
            | hir::PatKind::Tuple(pats)
            | hir::PatKind::Array(pats) => {
                for pat in pats {
                    self.reject_moves_out_of_drop_scrutinee(*pat);
                }
            }
            hir::PatKind::Struct { fields, .. } => {
                let fields: Vec<(Option<hir::PatId>, Option<LocalId>, Span)> = fields
                    .iter()
                    .map(|field| (field.pat, field.local, field.name))
                    .collect();
                for (pat, local, name) in fields {
                    match (pat, local) {
                        (Some(pat), _) => self.reject_moves_out_of_drop_scrutinee(pat),
                        (None, Some(local)) => {
                            let is_non_copy = self
                                .local_types
                                .get(&local)
                                .is_some_and(|ty| !self.is_copy_type(ty));
                            if is_non_copy {
                                self.push_diag(
                                    Diagnostic::error(
                                        "cannot move a field out of a value whose type implements \
                                         Drop: the destructor requires the complete value \
                                         (OWN-PARTIAL-001)",
                                        name,
                                    )
                                    .with_code("E0100"),
                                );
                            }
                        }
                        (None, None) => {}
                    }
                }
            }
            hir::PatKind::Error => {}
        }
    }

    fn reject_moves_out_of_borrow(&mut self, pat: hir::PatId) {
        let node = self.hir.pat(pat);
        let span = node.span;
        match &node.kind {
            hir::PatKind::Wild | hir::PatKind::Lit(_) | hir::PatKind::Path { .. } => {}
            hir::PatKind::Binding { local, .. } => {
                let is_non_copy = self
                    .local_types
                    .get(local)
                    .is_some_and(|ty| !self.is_copy_type(ty));
                if is_non_copy {
                    self.push_diag(
                        Diagnostic::error(
                            format!(
                                "cannot move out of a borrow: binding '{}' would take ownership \
                                 of a non-Copy value read through a reference",
                                self.text(span)
                            ),
                            span,
                        )
                        .with_code("E0101")
                        .with_label("bind by reference or match a Copy field instead"),
                    );
                }
            }
            hir::PatKind::TupleVariant { pats, .. }
            | hir::PatKind::Tuple(pats)
            | hir::PatKind::Array(pats) => {
                for pat in pats {
                    self.reject_moves_out_of_borrow(*pat);
                }
            }
            hir::PatKind::Struct { fields, .. } => {
                for field in fields {
                    match (field.pat, field.local) {
                        (Some(pat), _) => self.reject_moves_out_of_borrow(pat),
                        // Shorthand `Point { x }` binds without a sub-pattern node.
                        (None, Some(local)) => {
                            let is_non_copy = self
                                .local_types
                                .get(&local)
                                .is_some_and(|ty| !self.is_copy_type(ty));
                            if is_non_copy {
                                self.push_diag(
                                    Diagnostic::error(
                                        format!(
                                            "cannot move out of a borrow: binding '{}' would \
                                             take ownership of a non-Copy field read through a \
                                             reference",
                                            self.text(field.name)
                                        ),
                                        field.name,
                                    )
                                    .with_code("E0101")
                                    .with_label("bind by reference or match a Copy field instead"),
                                );
                            }
                        }
                        (None, None) => {}
                    }
                }
            }
            hir::PatKind::Error => {}
        }
    }

    /// **AS4: one Copy authority over `Ty`.** This was a second implementation of
    /// `typecheck::is_copy_with_impls`, kept aligned by hand — its own comment said so — and it had
    /// drifted: a `_ => false` wildcard swallowed `Ty::Never`, which 03-Type-System.md calls `Copy`
    /// ("reference values, function values, `Unit`, and `!` are `Copy`"), and `Ty::Extension`,
    /// where the checker consults the tensor's own answer.
    ///
    /// Measured before merging (`as4_copy_rule_inventory`): the two agreed on every sample except
    /// `Never`. Delegating adopts the checker's answer, and the public entry point it needs
    /// (`is_copy_type_with`) already existed — this duplicate never had to be written.
    fn is_copy_type(&self, ty: &Ty) -> bool {
        crate::typecheck::is_copy_type_with(ty, &self.copy_types)
    }

    fn check_crate(&mut self) {
        // WP-C1.4 (2026-07-17) pointed `self.file` at each item's own file before checking it, so
        // that a borrow-check diagnostic for a non-root-file item was not misattributed to the root
        // (DEV-006). AS1b-ii-d: spans carry their source, so there is no file to aim.
        for item in self.hir.items.iter() {
            match &item.kind {
                hir::ItemKind::Fn(def) => {
                    self.check_fn_def(def);
                }
                hir::ItemKind::Impl {
                    items,
                    generics: impl_generics,
                    ..
                } => {
                    for impl_item in items {
                        if let hir::ImplItem::Fn { def, .. } = impl_item {
                            // WP-C6.2b-F5: a bound written on the impl head (`impl<T: Sh> W<T>`)
                            // is in scope for every method body inside it.
                            self.enclosing_generics = impl_generics.clone();
                            self.check_fn_def(def);
                            self.enclosing_generics = Vec::new();
                        }
                    }
                }
                _ => {}
            }
        }

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
        // DEV-DISPLAY-DISPATCH: the parameters a method call's receiver type may name.
        self.current_generics = def.sig.generics.clone();

        // Parameters are initially owned/borrowed (not moved)
        self.check_block(def.body);
        // WP-C6.1f-b5: the return-escape check belongs to the FUNCTION BODY's tail, not to every
        // nested block's. `borrowed_local` recurses through `Block`, `If` and `Match`, so applying
        // it once here still finds a reference to a local that reaches the return through any of
        // them -- while a nested block's tail on its own is not a return at all.
        if let Some(tail_expr) = self.hir.block(def.body).tail {
            self.check_return_escape(tail_expr);
        }
    }

    /// DEV-137: check a CONDITION expression, ending the borrows it creates before the branch it
    /// guards. Used by `while` and `if`, which are the only two positions whose operand is
    /// consumed by a branch and cannot outlive it.
    ///
    /// `active_borrows` is scoped by exactly two mechanisms — `check_block` truncates at block
    /// end, and `check_stmt` truncates after each expression statement. A condition is NEITHER:
    /// it is an expression evaluated outside any statement of its own. So the receiver auto-borrow
    /// that `while i < v.len()` takes was pushed and then still on the stack when
    /// `check_block(body)` ran — and `check_block` records its own entry depth AFTER that push, so
    /// it could never pop it. Every mutation of that receiver inside the branch was an E0101
    /// against a borrow that had already ended.
    ///
    /// `03-Type-System.md` "References and Lifetimes": a temporary borrow ends with its statement.
    /// A condition's value is consumed by the branch, so its temporaries end at the branch
    /// boundary.
    ///
    /// The truncate is DEPTH-BASED, which is what keeps the negative cases negative: a borrow
    /// created before the loop (`let view = &values;`) lives at a shallower depth than this
    /// snapshot, is untouched, and a mutation through its owner is still refused.
    ///
    /// **Deliberately NOT applied to `match` scrutinees or `for` iterators.** Those are not
    /// conditions: PAT-BIND-001 binds arm payloads BY REFERENCE into the scrutinee, and
    /// `for x in &v` yields references into the iterated value, so in both cases the borrow must
    /// span the body. Truncating them would hand out references to storage the checker had
    /// stopped tracking.
    /// DEV-136: whether `block` definitely does NOT reach the statement after it.
    ///
    /// Ownership state is merged at a control-flow join from its predecessors. A predecessor that
    /// terminates — `return`, `break`, `continue`, `panic`, a trap — is not a predecessor of that
    /// join at all, so the moves it performed must not appear in the merged state. Before this,
    /// every branch contributed unconditionally, so
    ///
    /// ```stark
    /// if flag { return out; }
    /// out.push('a');
    /// ```
    ///
    /// reported "use of moved value" on a path where the move provably did not happen.
    ///
    /// **This predicate is deliberately conservative in one direction only.** Answering `true`
    /// wrongly would DROP a real move from the join and accept a use-after-move — unsound.
    /// Answering `false` wrongly only preserves the pre-existing false positive. So every arm
    /// below reports `true` solely on evidence of divergence, and anything unrecognised
    /// (including `loop` without a reachable `break`, which would need reachability analysis to
    /// judge) falls through to `false`.
    fn block_diverges(&self, block_id: BlockId) -> bool {
        let block = self.hir.block(block_id);
        // Statements execute in order, so a `return`/`break`/`continue` anywhere in the sequence
        // makes everything after it — and the join — unreachable.
        for &stmt_id in &block.stmts {
            match &self.hir.stmt(stmt_id).kind {
                hir::StmtKind::Return(_) | hir::StmtKind::Break(_) | hir::StmtKind::Continue => {
                    return true;
                }
                hir::StmtKind::Expr { expr, .. } if self.expr_diverges(*expr) => {
                    return true;
                }
                _ => {}
            }
        }
        block.tail.is_some_and(|tail| self.expr_diverges(tail))
    }

    /// DEV-136: whether evaluating `expr` definitely does not fall through. See
    /// [`Self::block_diverges`] for why this errs toward `false`.
    fn expr_diverges(&self, expr_id: ExprId) -> bool {
        // The type checker already proves divergence for anything of type `!` — `panic(..)`, and
        // any call to a function returning `!`. Reusing that answer keeps one authority for the
        // question rather than re-deriving it from syntax.
        if matches!(self.expr_types.get(&expr_id), Some(Ty::Never)) {
            return true;
        }
        match &self.hir.expr(expr_id).kind {
            hir::ExprKind::Block(block) => self.block_diverges(*block),
            // An `if` escapes only when BOTH sides do; without an `else` the fall-through path
            // always exists.
            hir::ExprKind::If {
                then_block, else_, ..
            } => {
                else_.is_some_and(|else_expr| self.expr_diverges(else_expr))
                    && self.block_diverges(*then_block)
            }
            // A `match` escapes only when every arm does. Exhaustiveness is checked elsewhere, so
            // an empty arm list cannot reach a join either way.
            hir::ExprKind::Match { arms, .. } => {
                !arms.is_empty() && arms.iter().all(|arm| self.expr_diverges(arm.body))
            }
            _ => false,
        }
    }

    /// Whether a value of this type can carry a borrow — directly, or nested inside an aggregate.
    ///
    /// DEV-181 uses it to decide whether an assignment's right-hand side temporaries may be
    /// released. An owned value's borrows end with the temporaries that produced it; a
    /// borrow-carrying one keeps them, because the reference IS the value being stored.
    ///
    /// Conservative by construction: an unknown type answers `true`, so an unrecognised shape keeps
    /// its borrows rather than silently releasing them. A false positive here is a refusal; a false
    /// negative would be an untracked reference.
    fn ty_carries_borrow(&self, ty: Option<&Ty>) -> bool {
        let Some(ty) = ty else {
            return true;
        };
        match ty {
            Ty::Ref { .. } | Ty::Slice(_) => true,
            Ty::Tuple(elements) => elements.iter().any(|t| self.ty_carries_borrow(Some(t))),
            Ty::Array(element, _) => self.ty_carries_borrow(Some(element)),
            // A nominal's or core type's ARGUMENTS may be references — `Option<&T>` is a
            // borrow-carrying value per 03-Type-System.md's references-and-lifetimes rules.
            Ty::Struct(_, args) | Ty::Enum(_, args) | Ty::Core(_, args) => {
                args.iter().any(|t| self.ty_carries_borrow(Some(t)))
            }
            Ty::Range(inner) => self.ty_carries_borrow(Some(inner)),
            Ty::Primitive(_) | Ty::Fn { .. } | Ty::Never => false,
            // Anything else — parameters, inference variables, errors, extensions — is treated as
            // carrying a borrow, which is the safe direction.
            _ => true,
        }
    }

    fn check_condition(&mut self, cond: ExprId) {
        let borrows_before = self.active_borrows.len();
        self.check_expr(cond);
        self.active_borrows.truncate(borrows_before);
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
            // WP-C6.1f-b5: the escape check is NOT applied here -- it is applied once to the
            // FUNCTION BODY's tail in `check_fn_def`, and to every `return` statement. A nested
            // block's tail is not a return: `let r = if c { &p } else { &q };` has `&p` as an
            // `if`-branch tail, and reporting E0103 "cannot return reference to local stack
            // variable" for it was both a wrong diagnosis (nothing is returned) and an
            // over-rejection -- OWN-CARRY-001 explicitly contemplates a control-flow merge, which
            // "carries the union of possible source referents" with a region no larger than the
            // intersection of theirs. It even fired when both branches borrowed the SAME owner.
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
                // DEV-066: `*r` USES the reference `r` by read — it must never consume it.
                // Both `println(*r)`-style reads and `*r = v` writes were marking `r` moved
                // (mutable references are non-Copy, so the generic path treated the use as a
                // move), wrongly rejecting the canonical `*r = *r + 1` mutation pattern.
                if matches!(op, UnOp::Deref) {
                    if let Some(place) = self.place_of(*operand) {
                        self.check_place_available(&place, self.hir.expr(*operand).span);
                    } else {
                        self.check_expr(*operand);
                    }
                    return;
                }
                if let UnOp::Ref { mutable } = op {
                    // Borrow creation!
                    if let Some(place) = self.place_of(*operand) {
                        let local_id = place.local;
                        self.check_place_available(&place, self.hir.expr(*operand).span);
                        // Check conflicts
                        let mut has_conflict = false;
                        for b in &self.active_borrows {
                            if places_overlap(&b.place, &place) && (*mutable || b.mutable) {
                                self.push_diag(
                                    Diagnostic::error(format!("cannot borrow variable '{}' because it is already borrowed", self.text(expr.span)), expr.span)
                                        .with_code("E0101")
                                );
                                has_conflict = true;
                                break;
                            }
                        }
                        if !has_conflict {
                            let _ = local_id;
                            self.active_borrows.push(Borrow {
                                place: place.clone(),
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
                // **DEV-181: a borrow taken by the assignment's OWN right-hand side must not block
                // the assignment.** `n = n.deeper()` pushed the receiver auto-borrow from
                // `n.deeper()` and then found it still on the stack in the write check below —
                // refusing an everyday idiom with no hoisting workaround.
                //
                // Same mechanism as DEV-137, and the same shape of repair: snapshot, check,
                // truncate. But NOT unconditionally, because the RHS's borrow is sometimes the
                // assigned VALUE. `n.deeper()` yields an owned `Node`, so its temporary's borrow
                // dies with it; `r = &v.field` yields a reference whose borrow must survive.
                // Dropping that would hand out a reference the checker had stopped tracking, so the
                // truncation is gated on the assigned type carrying no borrow — the same kind of
                // boundary DEV-137 drew when it excluded `match` scrutinees and `for` iterators.
                let borrows_before = self.active_borrows.len();
                self.check_expr(*rhs);
                if !self.ty_carries_borrow(self.expr_types.get(rhs)) {
                    self.active_borrows.truncate(borrows_before);
                }

                // Write check: verify no active borrows on the place
                let assigned = self.place_of(*lhs);
                if let Some(local_id) = self.get_root_local(*lhs) {
                    for b in &self.active_borrows {
                        // With a place in hand compare precisely; otherwise fall back to the root
                        // local, which is the conservative answer.
                        let conflicts = match &assigned {
                            Some(place) => places_overlap(&b.place, place),
                            None => b.place.local == local_id,
                        };
                        if conflicts {
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
                // DEV-150 / CD-357: the overlap rule is checked over the WHOLE argument list, and
                // before any of it is walked. A method's RECEIVER is an argument for this purpose —
                // `buffer.fill(buffer.len())` is the same conflict as `fill(&mut buffer, len)` —
                // so it joins the list when the call has one.
                let mut parts: Vec<ExprId> = Vec::with_capacity(args.len() + 1);
                if let hir::ExprKind::Field { base, name, .. } = &self.hir.expr(*callee).kind {
                    if self.text(*name) != "refine"
                        && matches!(
                            self.method_receiver(*base, *name),
                            Some(hir::Receiver::Ref | hir::Receiver::RefMut)
                        )
                    {
                        parts.push(*base);
                    }
                }
                parts.extend(args.iter().copied());
                self.check_argument_overlap(&parts);

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
                self.check_condition(*cond);

                let moved_before = self.moved_places.clone();
                self.check_block(*then_block);
                let moved_then = self.moved_places.clone();
                // DEV-136: a branch that terminates is not a predecessor of the join, so the
                // moves it performed must not be merged into the state after the `if`.
                let then_diverges = self.block_diverges(*then_block);

                if let Some(else_expr) = else_ {
                    self.moved_places = moved_before.clone();
                    self.check_expr(*else_expr);
                    let moved_else = self.moved_places.clone();
                    let else_diverges = self.expr_diverges(*else_expr);

                    self.moved_places = match (then_diverges, else_diverges) {
                        // Neither reaches the join, so nothing after the `if` is reachable and
                        // the merged value cannot be observed. The union is kept rather than
                        // something emptier so that an unreachable tail still sees a
                        // conservative state if reachability is ever judged differently.
                        (true, true) => moved_then.union(&moved_else).cloned().collect(),
                        (true, false) => moved_else,
                        (false, true) => moved_then,
                        // The pre-existing rule, and still the right one when both arrive:
                        // moved on EITHER path means maybe-moved, which is treated as moved.
                        (false, false) => moved_then.union(&moved_else).cloned().collect(),
                    };
                } else {
                    // With no `else`, the fall-through path is the one that skipped the branch.
                    // If the branch terminates, reaching this point PROVES it did not run, so the
                    // state is exactly what it was before. Otherwise a move inside it is a
                    // maybe-move, which stays treated as moved.
                    self.moved_places = if then_diverges {
                        moved_before
                    } else {
                        moved_then
                    };
                }
            }
            hir::ExprKind::Match { scrutinee, arms } => {
                self.check_expr(*scrutinee);
                // DEV-072 (WP-C4.7-5): a scrutinee read THROUGH a reference is matched by
                // reference — binding a non-`Copy` payload out of it would move ownership out of
                // a borrow, which the ownership rules forbid (a borrow never transfers
                // ownership). Nothing checked this before: patterns were not inspected here at
                // all, so `match *self { Holder::Val(s) => … }` in a `&self` method passed the
                // front end, and only MIR lowering refused it — the two engines disagreed about
                // whether the program was legal. The oracle's legacy clone semantics masked the
                // unsoundness at runtime (the CLONE was consumed, not the referent).
                if self.scrutinee_reads_through_ref(*scrutinee) {
                    for arm in arms {
                        self.reject_moves_out_of_borrow(arm.pat);
                    }
                } else if self.scrutinee_nominal_has_drop(*scrutinee) {
                    // **DEV-211: OWN-PARTIAL-001 applies to a matched component too.**
                    //
                    // "Moving a field from a type that implements `Drop` is prohibited, because its
                    // destructor requires the complete value." A `match` that binds a non-`Copy`
                    // payload out of an owned `Drop` nominal is that move — the destructor cannot
                    // then run on a complete value, and PAT-DROP-001 destroys only the *unbound*
                    // components, so nothing runs the type's own destructor at all.
                    //
                    // Measured before repairing: `impl Drop for E` with `match e { E::A(s) => … }`
                    // printed the arm and never the destructor, in BOTH the HIR oracle and MIR.
                    // The engines agreed, so this was a front-end conformance defect rather than an
                    // engine divergence — the checker had the rule for struct fields
                    // (`local_has_drop`) and never applied it to a matched component.
                    for arm in arms {
                        self.reject_moves_out_of_drop_scrutinee(arm.pat);
                    }
                }
                let moved_before = self.moved_places.clone();
                let mut merged_moved = HashSet::new();
                // DEV-136: only arms that REACH the join contribute to it. An arm ending in
                // `return`/`break`/`continue`/`panic` is not a predecessor.
                let mut any_arm_reaches_join = false;
                for arm in arms {
                    self.moved_places = moved_before.clone();
                    self.check_expr(arm.body);
                    if !self.expr_diverges(arm.body) {
                        any_arm_reaches_join = true;
                        merged_moved.extend(self.moved_places.iter().cloned());
                    }
                }
                // When no arm reaches the join, `merged_moved` is empty — which would WIDEN the
                // live set by discarding moves that happened before the `match`. The join is
                // unreachable, so the state cannot be observed, but it must still not claim a
                // previously moved value is live again.
                self.moved_places = if any_arm_reaches_join {
                    merged_moved
                } else {
                    moved_before
                };
            }
            hir::ExprKind::Block(b) => {
                self.check_block(*b);
            }
            hir::ExprKind::Loop { body } => {
                self.check_block(*body);
            }
            hir::ExprKind::While { cond, body } => {
                self.check_condition(*cond);
                self.check_block(*body);
            }
            hir::ExprKind::For { iter, body, .. } => {
                // WP-C6.1d closed DEV-090: by-value iteration over a fixed-length non-`Copy` array
                // is now lowered by unrolling (each element moves via `ConstIndex(i)`), and the HIR
                // and MIR engines agree, so the deterministic E0104 rejection that used to forecast
                // an oracle/MIR divergence here is removed. `Copy` arrays still iterate by copy.
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

    /// DEV-DISPLAY-DISPATCH: the receiver form `method` is declared with, by whichever trait a
    /// bound on generic parameter `param_name` supplies it.
    ///
    /// **DEV-BOUND-TRAIT-IDENTITY: the bound's identity comes from the resolver.** This pass used
    /// to take the bound's SPELLING and scan every HIR item for a trait declared with that name,
    /// which meant a qualified bound matched nothing, an unrelated same-named trait could win,
    /// and — worst — with two same-named traits the receiver came from whichever appeared first
    /// in HIR order. Two identical programs differing only in the order their traits were
    /// declared then disagreed about whether `x.act()` moved `x`. Both passes now read
    /// `hir::resolved_bound_trait`, so the trait the type checker selected a method FROM is the
    /// trait this reads the receiver form OF.
    ///
    /// A user trait's declaration and a Core trait's contract are both consulted, and neither is
    /// preferred — if two bounds supply the same name the type checker has already reported
    /// E0203, and reading either signature here only affects which already-rejected program gets
    /// a second diagnostic.
    fn bound_method_receiver(&self, param_name: &str, method: &str) -> Option<hir::Receiver> {
        let mut generics = self.current_generics.clone();
        generics.extend(self.enclosing_generics.iter().cloned());
        for param in &generics {
            if self.text(param.name) != param_name {
                continue;
            }
            for bound in &param.bounds {
                let Some(bound_trait) = hir::resolved_bound_trait(self.hir, bound) else {
                    continue;
                };
                let receiver = match bound_trait {
                    // DEV-069: the trait's method names belong to the TRAIT's declaring file.
                    hir::BoundTrait::User(trait_id) => {
                        hir::trait_method_receiver(self.hir, trait_id, method, |item, span| {
                            self.item_text(item, span).to_string()
                        })
                    }
                    hir::BoundTrait::Core(core_trait) => {
                        crate::typecheck::core_trait_method_receiver(core_trait, method)
                    }
                };
                if let Some(receiver) = receiver {
                    return Some(receiver);
                }
            }
        }
        None
    }

    fn method_receiver(&self, base: ExprId, name: Span) -> Option<hir::Receiver> {
        let mut base_ty = self.expr_types.get(&base)?.clone();
        while let Ty::Ref { inner, .. } = base_ty {
            base_ty = *inner;
        }
        let method_name = self.text(name);
        // DEV-082 (WP-C4.7-8.6): SLICE and ARRAY receivers were absent here, so a method call on
        // one returned `None` and the caller's fallback CONSUMED the receiver. For a `&[T]` local
        // that is harmless (shared references are `Copy`), which is why shared slices shipped
        // without anyone noticing; for a `&mut [T]` local it is a move, so `s.len()` followed by
        // any second use failed E0100 "use of moved value". Slice methods (`len`/`is_empty`) only
        // ever read, so a shared borrow is the right receiver kind.
        if matches!(base_ty, Ty::Slice(..) | Ty::Array(..)) {
            return Some(hir::Receiver::Ref);
        }
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

        // DEV-DISPLAY-DISPATCH: a BOUNDED GENERIC receiver. The receiver kind is declared by the
        // bound, and there was no branch for it here at all — `Ty::Param` fell into the `None`
        // below, whose caller CONSUMES the receiver. So every `&self` method reached through a
        // bound moved its receiver, and `let a = x.fmt(); let b = x.fmt();` failed E0100 "use of
        // moved value" for a `T: Display` parameter exactly as it did for a `T: Named` one. The
        // signature comes from the same two sources the type checker selects from: a user trait's
        // own declaration, or the Core trait's implementation contract.
        if let Ty::Param(param_name) = &base_ty {
            if let Some(receiver) = self.bound_method_receiver(param_name, method_name) {
                return Some(receiver);
            }
        }

        let item_id = match base_ty {
            Ty::Struct(item, _) | Ty::Enum(item, _) => item,
            _ => return None,
        };
        // DEV-069: scans every impl in the program, so each impl's method names are read
        // against the file that declares that impl.
        if let Some(receiver) = self.hir.items.iter().enumerate().find_map(|(idx, item)| {
            let impl_id = hir::ItemId(idx as u32);
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
                hir::ImplItem::Fn { def, .. }
                    if self.item_text(impl_id, def.sig.name) == method_name =>
                {
                    def.sig.receiver
                }
                _ => None,
            })
        }) {
            return Some(receiver);
        }

        // DEV-060: mirror typecheck/body.rs::resolve_method's `default_fallback` (WP-C1.3) -- a
        // trait method declared with a real body and never overridden by any impl is a legal
        // call, but the search above only ever looks at `ImplItem::Fn` overrides (exactly like
        // typecheck/body.rs's own override-only `candidates` collection, considered alone). Without
        // this fallback, an un-overridden trait default method returns `None` here, and the
        // `Call` handler's `None => self.check_expr(*base)` arm unconditionally moves the
        // receiver (`check_expr`'s `Path` arm consumes any `Local`/`SelfValue` place)
        // regardless of the method's real `&self`/`&mut self`/`self` kind -- so a second call
        // on the same receiver was wrongly flagged as a use of a moved value.
        self.hir.items.iter().find_map(|item| {
            let hir::ItemKind::Impl {
                self_ty: impl_self_ty_id,
                trait_: Some(trait_ref),
                ..
            } = &item.kind
            else {
                return None;
            };
            let matches_type = matches!(
                self.hir.ty(*impl_self_ty_id).kind,
                hir::TypeKind::Path {
                    res: Res::Item(impl_item),
                    ..
                } if impl_item == item_id
            );
            if !matches_type {
                return None;
            }
            let Res::Item(trait_id) = trait_ref.res else {
                return None;
            };
            let hir::ItemKind::Trait {
                items: trait_items, ..
            } = &self.hir.item(trait_id).kind
            else {
                return None;
            };
            // DEV-069: the trait's method names belong to the TRAIT's declaring file.
            trait_items.iter().find_map(|trait_item| match trait_item {
                hir::TraitItem::Method { sig, body: Some(_) }
                    if self.item_text(trait_id, sig.name) == method_name =>
                {
                    sig.receiver
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
            .any(|borrow| places_overlap(&borrow.place, &place) && (mutable || borrow.mutable))
        {
            self.push_diag(
                Diagnostic::error("method receiver conflicts with an active borrow", span)
                    .with_code("E0101"),
            );
        } else {
            self.active_borrows.push(Borrow {
                place: place.clone(),
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
                // The NAME, not the span: see `Projection`. Every expression reaching here
                // belongs to the item currently being checked, and `self.file` tracks that item
                // (DEV-069), so `self.text` reads against the right source.
                place
                    .projections
                    .push(Projection::Field(self.text(*name).to_string()));
                Some(place)
            }
            hir::ExprKind::TupleField { base, index } => {
                let mut place = self.place_of(*base)?;
                place
                    .projections
                    .push(Projection::TupleField(self.text(*index).to_string()));
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
        self.check_read_borrow_conflict(&place, expr.span);

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
        if self
            .active_borrows
            .iter()
            .any(|b| places_overlap(&b.place, &place))
        {
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

    /// **DEV-150 / CD-357 — argument evaluation does NOT provide two-phase borrow semantics.**
    ///
    /// > A call may not create an exclusive borrow of a place while another argument in the same
    /// > call reads from or borrows an overlapping place. Such reads must be evaluated into locals
    /// > before the exclusive borrow is created.
    ///
    /// ```stark
    /// f(&mut x, x.field);            // refused
    /// let field = x.field;           // hoist
    /// f(&mut x, field);              // accepted
    /// ```
    ///
    /// The rule is UNIFORM in the base it applies to — a local, a place reached through `&mut`, a
    /// field projection, an index, a free function or a method receiver — and it is
    /// ORDER-INDEPENDENT: `f(x.field, &mut x)` is refused exactly as `f(&mut x, x.field)` is. That
    /// is why this runs as its own pass over the whole argument list rather than falling out of the
    /// left-to-right walk, which by construction can only see a conflict when the borrow comes
    /// first.
    ///
    /// # What was wrong before
    ///
    /// The local case was refused and the indirect case ACCEPTED:
    ///
    /// ```stark
    /// fn forward(h: &mut Holder) { bump(h, h.limit); }   // accepted, ran, did not build
    /// ```
    ///
    /// Passing a `&mut`-typed place REBORROWS, which registers no `active_borrows` entry, so the
    /// read that followed saw nothing to conflict with. The HIR oracle then executed it correctly
    /// and the native backend emitted Rust that rustc refused with E0503 — accepted-but-unbuildable,
    /// and worse, a rule that changed meaning one indirection away from where it was written.
    ///
    /// # Why rejection rather than sequencing
    ///
    /// Blessing the accepted case would have required accepting the LOCAL case too, which widens
    /// the borrow rule into two-phase borrows — evaluation-order machinery, and a semantics
    /// commitment. Uniform rejection keeps one backend-neutral rule that every engine can agree on
    /// by construction, and stays reversible if STARK later adopts two-phase borrows deliberately.
    fn check_argument_overlap(&mut self, parts: &[ExprId]) {
        let mut exclusive: Vec<(usize, Place)> = Vec::new();
        for (index, &part) in parts.iter().enumerate() {
            if let Some(place) = self.exclusive_borrow_of(part) {
                exclusive.push((index, place));
            }
        }
        if exclusive.is_empty() {
            return;
        }
        for (index, &part) in parts.iter().enumerate() {
            let mut reads = Vec::new();
            self.collect_read_places(part, &mut reads);
            for (place, span) in reads {
                for (owner, exclusive_place) in &exclusive {
                    if *owner != index && places_overlap(exclusive_place, &place) {
                        self.push_diag(
                            Diagnostic::error(
                                format!(
                                    "cannot read '{}' in the same call that borrows it exclusively",
                                    self.text(span)
                                ),
                                span,
                            )
                            .with_code("E0101")
                            .with_label(
                                "bind this to a local before the call: argument evaluation does                                  not provide two-phase borrows",
                            ),
                        );
                        // One mistake, one diagnostic. The left-to-right walk that follows would
                        // otherwise report the SAME read again through `check_read_borrow_conflict`
                        // — with different wording, which reads like two separate problems.
                        self.overlap_reported.insert((span.lo, span.hi));
                        return;
                    }
                }
            }
        }
    }

    /// The place an argument borrows EXCLUSIVELY, if any.
    ///
    /// Two forms produce one, and treating them alike is the whole point of the repair: an explicit
    /// `&mut place`, and a place whose type is already `&mut T` — which reborrows. The second is the
    /// one that was invisible.
    fn exclusive_borrow_of(&self, expr_id: ExprId) -> Option<Place> {
        if let hir::ExprKind::Unary {
            op: UnOp::Ref { mutable: true },
            operand,
        } = &self.hir.expr(expr_id).kind
        {
            return self.place_of(*operand);
        }
        if matches!(
            self.expr_types.get(&expr_id),
            Some(Ty::Ref { mutable: true, .. })
        ) {
            return self.place_of(expr_id);
        }
        None
    }

    /// Every place an argument expression reads or borrows.
    ///
    /// A sub-expression that IS a place is recorded whole and not descended into — `x.a.b` is one
    /// read of `x.a.b`, and `places_overlap`'s prefix rule already relates it to a borrow of `x` or
    /// of `x.a`. Anything else recurses, so a place buried inside arithmetic or a nested call is
    /// still found.
    fn collect_read_places(&self, expr_id: ExprId, out: &mut Vec<(Place, Span)>) {
        let expr = self.hir.expr(expr_id);
        if let Some(place) = self.place_of(expr_id) {
            out.push((place, expr.span));
            return;
        }
        match &expr.kind {
            hir::ExprKind::Unary { operand, .. } => self.collect_read_places(*operand, out),
            hir::ExprKind::Binary { lhs, rhs, .. }
            | hir::ExprKind::Assign { lhs, rhs, .. }
            | hir::ExprKind::Range {
                lo: lhs, hi: rhs, ..
            } => {
                self.collect_read_places(*lhs, out);
                self.collect_read_places(*rhs, out);
            }
            hir::ExprKind::Cast { expr, .. } | hir::ExprKind::Try(expr) => {
                self.collect_read_places(*expr, out)
            }
            hir::ExprKind::Call { callee, args } => {
                self.collect_read_places(*callee, out);
                for &arg in args {
                    self.collect_read_places(arg, out);
                }
            }
            hir::ExprKind::Field { base, .. } | hir::ExprKind::TupleField { base, .. } => {
                self.collect_read_places(*base, out)
            }
            hir::ExprKind::Index { base, index } => {
                self.collect_read_places(*base, out);
                self.collect_read_places(*index, out);
            }
            hir::ExprKind::Tuple(items) | hir::ExprKind::Array(items) => {
                for &item in items {
                    self.collect_read_places(item, out);
                }
            }
            hir::ExprKind::Repeat { value, count } => {
                self.collect_read_places(*value, out);
                self.collect_read_places(*count, out);
            }
            hir::ExprKind::StructLit { fields, .. } => {
                for field in fields {
                    // A shorthand field init (`Point { x, y }`) carries no expression; the name
                    // itself is the read, and it resolves to a local the caller can see.
                    if let Some(value) = field.expr {
                        self.collect_read_places(value, out);
                    }
                }
            }
            // A borrow reads the place it borrows: `f(&mut x, &x)` overlaps, and so does
            // `f(&mut x, &mut x)`.
            _ => {}
        }
    }

    fn check_read_expr(&mut self, expr_id: ExprId) {
        if let Some(place) = self.place_of(expr_id) {
            if self.check_place_available(&place, self.hir.expr(expr_id).span) {
                self.check_read_borrow_conflict(&place, self.hir.expr(expr_id).span);
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

    fn check_read_borrow_conflict(&mut self, place: &Place, span: Span) {
        // DEV-150: already reported as a call-argument overlap, with the message that says what to
        // do about it. One mistake gets one diagnostic.
        if self.overlap_reported.contains(&(span.lo, span.hi)) {
            return;
        }
        if self
            .active_borrows
            .iter()
            .any(|borrow| places_overlap(&borrow.place, place) && borrow.mutable)
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

    /// **DEV-210: answered by identity, not by spelling.**
    ///
    /// This scanned the impl set itself and asked whether the written trait name
    /// `.ends_with("Drop")` — so `impl MyDrop for S` made `S` "implement `Drop`", and a legal
    /// partial move out of one of its fields was refused with E0100. Valid Core, rejected, because
    /// a user trait's name happened to end in four particular letters. CD-379 settled the identity
    /// rule for `Display`; this is the same defect, and the fix is to stop having a private answer
    /// at all.
    fn local_has_drop(&self, local: LocalId) -> bool {
        let Some(ty) = self.local_types.get(&local) else {
            return false;
        };
        match ty {
            Ty::Struct(id, _) | Ty::Enum(id, _) => self.drop_items.contains(id),
            _ => false,
        }
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
    crate::typecheck::copy_eligible_types(hir)
}

fn is_prefix(prefix: &[Projection], value: &[Projection]) -> bool {
    prefix.len() <= value.len() && prefix.iter().zip(value).all(|(a, b)| a == b)
}

fn places_overlap(left: &Place, right: &Place) -> bool {
    left.local == right.local
        && (is_prefix(&left.projections, &right.projections)
            || is_prefix(&right.projections, &left.projections))
}

/// **AS4 — the Copy rule over `Ty`: two implementations, measured before consolidation.**
///
/// `borrowck::is_copy_type` and `typecheck::is_copy_with_impls` both answer "is this `Ty` `Copy`?"
/// — the same question in the same type language, unlike the MIR/checker split, where different
/// type languages justify separate code. `borrowck`'s own comment says it exists to stay "aligned
/// with the type checker's `is_copy_with_impls`", which is an alignment maintained by hand.
///
/// RB0's method: measure before merging, and record what differs.
#[cfg(test)]
mod as4_copy_rule_inventory {
    use crate::hir::{CoreType, ItemId};
    use crate::typecheck::{is_copy_type_with, Ty};
    use std::collections::HashSet;

    /// The borrowck implementation, lifted verbatim so the matrix compares the RULES rather than
    /// the surrounding machinery. Deleted along with the original once they are shown equivalent.
    fn borrowck_rule(ty: &Ty, copy_types: &HashSet<ItemId>) -> bool {
        match ty {
            Ty::Primitive(primitive) => !matches!(
                primitive,
                crate::ast::Primitive::String | crate::ast::Primitive::Str
            ),
            Ty::Error | Ty::Ref { mutable: false, .. } => true,
            Ty::Struct(item, args) | Ty::Enum(item, args) => {
                copy_types.contains(item) && args.iter().all(|arg| borrowck_rule(arg, copy_types))
            }
            Ty::Core(CoreType::Option | CoreType::Result, args) => {
                args.iter().all(|arg| borrowck_rule(arg, copy_types))
            }
            Ty::Tuple(elements) => elements.iter().all(|e| borrowck_rule(e, copy_types)),
            Ty::Array(element, _) => borrowck_rule(element, copy_types),
            Ty::Fn { .. } => true,
            _ => false,
        }
    }

    fn samples() -> Vec<(&'static str, Ty)> {
        let i32t = || Ty::Primitive(crate::ast::Primitive::Int32);
        let string = || Ty::Primitive(crate::ast::Primitive::String);
        vec![
            ("Int32", i32t()),
            ("String", string()),
            ("Str", Ty::Primitive(crate::ast::Primitive::Str)),
            ("Unit", Ty::Primitive(crate::ast::Primitive::Unit)),
            ("Error", Ty::Error),
            ("Never", Ty::Never),
            ("Param", Ty::Param("T".to_string())),
            (
                "&T",
                Ty::Ref {
                    mutable: false,
                    inner: Box::new(i32t()),
                },
            ),
            (
                "&mut T",
                Ty::Ref {
                    mutable: true,
                    inner: Box::new(i32t()),
                },
            ),
            ("Slice", Ty::Slice(Box::new(i32t()))),
            ("Range", Ty::Range(Box::new(i32t()))),
            (
                "Fn",
                Ty::Fn {
                    params: vec![i32t()],
                    ret: Box::new(i32t()),
                },
            ),
            ("Tuple(Int32)", Ty::Tuple(vec![i32t()])),
            ("Tuple(String)", Ty::Tuple(vec![string()])),
            ("Array(Int32)", Ty::Array(Box::new(i32t()), 2)),
            ("Array(String)", Ty::Array(Box::new(string()), 2)),
            ("Option(Int32)", Ty::Core(CoreType::Option, vec![i32t()])),
            ("Option(String)", Ty::Core(CoreType::Option, vec![string()])),
            (
                "Result(Int32,Int32)",
                Ty::Core(CoreType::Result, vec![i32t(), i32t()]),
            ),
            ("Vec(Int32)", Ty::Core(CoreType::Vec, vec![i32t()])),
            ("HashMap", Ty::Core(CoreType::HashMap, vec![i32t(), i32t()])),
            ("Struct(eligible)", Ty::Struct(ItemId(0), Vec::new())),
            ("Struct(ineligible)", Ty::Struct(ItemId(1), Vec::new())),
            (
                "Struct(eligible,<String>)",
                Ty::Struct(ItemId(0), vec![string()]),
            ),
            ("Enum(eligible)", Ty::Enum(ItemId(0), Vec::new())),
        ]
    }

    /// **The finding, pinned.** The two disagree on exactly one sample, and it is the wildcard's
    /// doing: `borrowck`'s `_ => false` swallows `Ty::Never`, which the checker calls `Copy` (03:
    /// "reference values, function values, `Unit`, and `!` are `Copy`").
    ///
    /// `Ty::Extension` would be a second, but a tensor type cannot be built here without the
    /// extension's machinery; it is recorded in the audit rather than sampled.
    #[test]
    fn the_two_copy_rules_over_ty_are_measured_against_each_other() {
        let mut copy_types = HashSet::new();
        copy_types.insert(ItemId(0));
        let mut disagree = Vec::new();
        for (name, ty) in samples() {
            let checker = is_copy_type_with(&ty, &copy_types);
            let borrow = borrowck_rule(&ty, &copy_types);
            if checker != borrow {
                disagree.push(format!("{name}: checker={checker} borrowck={borrow}"));
            }
        }
        assert_eq!(
            disagree,
            vec!["Never: checker=true borrowck=false"],
            "the Copy rule over `Ty` exists twice; this pins where the copies differ so \
             consolidation is evidence-led"
        );
    }
}
