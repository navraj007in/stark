//! HIR → Deployment IR lowering (Gate 5 M5.1).
//!
//! Lowers *only* the entry function's acyclic reachable call graph, and only
//! the bounded set of straight-line tensor/model operations the prototype
//! needs. Every other reachable construct — recursion, loops, `if`/`match`,
//! mutation, unsupported tensor ops, non-static shapes — is rejected with an
//! `E06xx` diagnostic carrying the offending span. Nothing is silently skipped
//! and nothing falls back to the interpreter (plan §6).

use crate::ast::{DimBinOp, UnOp};
use crate::diag::Diagnostic;
use crate::extensions::tensor::types::{DType, TensorKind};
use crate::hir::{
    self, Builtin, DimExpr, ExprId, ExprKind, Hir, ItemId, ItemKind, LocalId, Res, TypeId,
};
use crate::source::{SourceFile, Span};
use crate::typecheck::{ExtensionTy, Ty, TypeTables};
use std::collections::HashMap;

use super::ir::{
    DeployDim, DeployOp, DeployParam, DeployStmt, DeployTy, DeploymentFunction, ScalarLit,
    TensorShape, ValueId,
};

/// Error codes for deployment lowering. `E06xx` is unused by the normative
/// front end (highest existing code is `E0500`).
mod code {
    pub const ENTRY_NOT_FOUND: &str = "E0601";
    pub const ENTRY_ABI: &str = "E0602";
    pub const MODEL_SELECTION: &str = "E0603";
    pub const RECURSION: &str = "E0604";
    pub const UNSUPPORTED: &str = "E0605";
    /// A dimension expression the deployment backend cannot represent (only
    /// literals, symbols, and their sums/products are supported).
    pub const UNSUPPORTED_DIM: &str = "E0606";
}

/// Result of the pure HIR→IR lowering step (before the model/manifest is
/// attached in [`super`]).
pub(crate) struct LoweredGraph {
    pub functions: Vec<DeploymentFunction>,
    /// The selected model's declaration name and its HIR item.
    pub model_type_name: String,
}

pub(crate) fn lower_reachable(
    hir: &Hir,
    tables: &TypeTables,
    file: &SourceFile,
    entry: &str,
) -> Result<LoweredGraph, Vec<Diagnostic>> {
    // Locate the entry function item.
    let Some(entry_id) = find_fn(hir, file, entry) else {
        return Err(vec![Diagnostic::error(
            format!("deployment entry function `{entry}` was not found"),
            Span::new(0, 0),
        )
        .with_code(code::ENTRY_NOT_FOUND)]);
    };

    // Discover the acyclic reachable call graph, entry first.
    let mut order = Vec::new();
    let mut stack = Vec::new();
    let mut state: HashMap<ItemId, DiscoverState> = HashMap::new();
    let mut diags = Vec::new();
    discover(
        hir, file, entry_id, &mut order, &mut stack, &mut state, &mut diags,
    );
    if !diags.is_empty() {
        return Err(diags);
    }

    let fn_index: HashMap<ItemId, usize> =
        order.iter().enumerate().map(|(i, id)| (*id, i)).collect();

    let model_names: std::collections::HashSet<String> = hir
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Model(def) => Some(text(file, def.name).to_string()),
            _ => None,
        })
        .collect();
    let mut lowerer = Lowerer {
        hir,
        tables,
        file,
        fn_index: &fn_index,
        model_names,
        diags: Vec::new(),
    };

    let mut functions = Vec::with_capacity(order.len());
    for &id in &order {
        if let Some(f) = lowerer.lower_fn(id) {
            functions.push(f);
        }
    }
    if !lowerer.diags.is_empty() {
        return Err(std::mem::take(&mut lowerer.diags));
    }

    // Validate the entry ABI now that it is lowered (plan §6.1).
    let mut abi_diags = Vec::new();
    let model_type_name = validate_entry_abi(hir, file, &functions[0], &mut abi_diags);
    if !abi_diags.is_empty() {
        return Err(abi_diags);
    }

    Ok(LoweredGraph {
        functions,
        model_type_name: model_type_name.expect("ABI validated"),
    })
}

fn find_fn(hir: &Hir, file: &SourceFile, name: &str) -> Option<ItemId> {
    for (i, item) in hir.items.iter().enumerate() {
        if let ItemKind::Fn(def) = &item.kind {
            if text(file, def.sig.name) == name {
                return Some(ItemId(i as u32));
            }
        }
    }
    None
}

// ------------------------------------------------------ reachability --

#[derive(Clone, Copy, PartialEq)]
enum DiscoverState {
    Active,
    Done,
}

/// DFS pre-order collection of reachable functions with cycle detection.
fn discover(
    hir: &Hir,
    file: &SourceFile,
    id: ItemId,
    order: &mut Vec<ItemId>,
    stack: &mut Vec<ItemId>,
    state: &mut HashMap<ItemId, DiscoverState>,
    diags: &mut Vec<Diagnostic>,
) {
    if state.get(&id) == Some(&DiscoverState::Done) {
        return;
    }
    order.push(id);
    state.insert(id, DiscoverState::Active);
    stack.push(id);

    let ItemKind::Fn(def) = &hir.item(id).kind else {
        stack.pop();
        state.insert(id, DiscoverState::Done);
        return;
    };
    // Collect user-function callees reachable through this body, in source order.
    let mut body_calls = Vec::new();
    walk_block_calls(hir, def.body, &mut body_calls);
    for callee in body_calls {
        if state.get(&callee) == Some(&DiscoverState::Active) {
            diags.push(
                Diagnostic::error(
                    format!(
                        "recursion is not supported in a deployment pipeline: `{}` is reachable from itself",
                        fn_name(hir, file, callee)
                    ),
                    hir_fn_name_span(hir, callee),
                )
                .with_code(code::RECURSION),
            );
            continue;
        }
        discover(hir, file, callee, order, stack, state, diags);
    }
    stack.pop();
    state.insert(id, DiscoverState::Done);
}

/// Collect user-function callees (`Res::Item` pointing at an `fn`) reachable
/// through the expressions of a block.
fn walk_block_calls(hir: &Hir, block: hir::BlockId, out: &mut Vec<ItemId>) {
    let b = hir.block(block);
    for &sid in &b.stmts {
        walk_stmt_calls(hir, sid, out);
    }
    if let Some(t) = b.tail {
        walk_expr_calls(hir, t, out);
    }
}

fn walk_stmt_calls(hir: &Hir, sid: hir::StmtId, out: &mut Vec<ItemId>) {
    match &hir.stmt(sid).kind {
        hir::StmtKind::Let { init: Some(e), .. } => walk_expr_calls(hir, *e, out),
        hir::StmtKind::Expr { expr, .. } => walk_expr_calls(hir, *expr, out),
        hir::StmtKind::Return(Some(e)) | hir::StmtKind::Break(Some(e)) => {
            walk_expr_calls(hir, *e, out)
        }
        _ => {}
    }
}

fn walk_expr_calls(hir: &Hir, eid: ExprId, out: &mut Vec<ItemId>) {
    match &hir.expr(eid).kind {
        ExprKind::Call { callee, args } => {
            if let ExprKind::Path {
                res: Res::Item(fn_id),
                ..
            } = &hir.expr(*callee).kind
            {
                if matches!(hir.item(*fn_id).kind, ItemKind::Fn(_)) && !out.contains(fn_id) {
                    out.push(*fn_id);
                }
            }
            walk_expr_calls(hir, *callee, out);
            for &a in args {
                walk_expr_calls(hir, a, out);
            }
        }
        ExprKind::Unary { operand, .. } => walk_expr_calls(hir, *operand, out),
        ExprKind::Binary { lhs, rhs, .. } | ExprKind::Assign { lhs, rhs, .. } => {
            walk_expr_calls(hir, *lhs, out);
            walk_expr_calls(hir, *rhs, out);
        }
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => {
            walk_expr_calls(hir, *base, out)
        }
        ExprKind::Try(e) => walk_expr_calls(hir, *e, out),
        ExprKind::Cast { expr, .. } => walk_expr_calls(hir, *expr, out),
        ExprKind::Index { base, index } => {
            walk_expr_calls(hir, *base, out);
            walk_expr_calls(hir, *index, out);
        }
        ExprKind::Tuple(items) | ExprKind::Array(items) => {
            for &e in items {
                walk_expr_calls(hir, e, out);
            }
        }
        ExprKind::Block(b) => walk_block_calls(hir, *b, out),
        _ => {}
    }
}

fn fn_name(hir: &Hir, file: &SourceFile, id: ItemId) -> String {
    if let ItemKind::Fn(def) = &hir.item(id).kind {
        text(file, def.sig.name).to_string()
    } else {
        "<item>".to_string()
    }
}

fn hir_fn_name_span(hir: &Hir, id: ItemId) -> Span {
    if let ItemKind::Fn(def) = &hir.item(id).kind {
        def.sig.name
    } else {
        hir.item(id).span
    }
}

// -------------------------------------------------------------- lowering --

struct Lowerer<'a> {
    hir: &'a Hir,
    tables: &'a TypeTables,
    file: &'a SourceFile,
    fn_index: &'a HashMap<ItemId, usize>,
    /// Every `model` declaration's type name, used to recognise a model
    /// parameter type from its source annotation. Using the full set (not just
    /// the unique model) keeps the single-model selection error (`E0603`) as the
    /// diagnostic when a pipeline declares more than one model.
    model_names: std::collections::HashSet<String>,
    diags: Vec<Diagnostic>,
}

/// Per-function lowering state.
struct FnCtx {
    env: HashMap<LocalId, ValueId>,
    body: Vec<DeployStmt>,
    next: u32,
}

impl FnCtx {
    fn fresh(&mut self) -> ValueId {
        let v = ValueId(self.next);
        self.next += 1;
        v
    }
}

impl<'a> Lowerer<'a> {
    fn error(&mut self, msg: impl Into<String>, span: Span) {
        self.diags
            .push(Diagnostic::error(msg, span).with_code(code::UNSUPPORTED));
    }

    fn lower_fn(&mut self, id: ItemId) -> Option<DeploymentFunction> {
        let ItemKind::Fn(def) = &self.hir.item(id).kind else {
            return None;
        };
        let sig = &def.sig;
        if sig.receiver.is_some() {
            self.diags.push(
                Diagnostic::error(
                    "deployment functions must be free functions (no `self` receiver)",
                    sig.name,
                )
                .with_code(code::UNSUPPORTED),
            );
            return None;
        }

        let mut ctx = FnCtx {
            env: HashMap::new(),
            body: Vec::new(),
            next: 0,
        };

        let mut params = Vec::with_capacity(sig.params.len());
        for p in &sig.params {
            let v = ctx.fresh();
            ctx.env.insert(p.local, v);
            // Parameter types come from the source annotation so symbolic
            // dimensions keep their source names (needed to bind them from the
            // argument shapes in the generated host).
            let ty = self.deploy_ty_from_ast(p.ty, p.name)?;
            params.push(DeployParam {
                name: text(self.file, p.name).to_string(),
                ty,
                value: v,
            });
        }

        let result = self.lower_block(def.body, &mut ctx)?;
        let ret = match sig.ret {
            hir::RetTy::Ty(tid) => self.deploy_ty_from_ast(tid, sig.name)?,
            hir::RetTy::Unit | hir::RetTy::Never(_) => {
                self.error(
                    "a deployment function must return a tensor or `Result` value",
                    sig.name,
                );
                return None;
            }
        };

        Some(DeploymentFunction {
            name: text(self.file, sig.name).to_string(),
            params,
            ret,
            body: ctx.body,
            result,
            span: sig.name,
        })
    }

    fn lower_block(&mut self, block: hir::BlockId, ctx: &mut FnCtx) -> Option<ValueId> {
        let b = self.hir.block(block);
        for &sid in &b.stmts {
            match &self.hir.stmt(sid).kind {
                hir::StmtKind::Empty => {}
                hir::StmtKind::Let {
                    local,
                    init: Some(init),
                    mutable: false,
                    ..
                } => {
                    let v = self.lower_expr(*init, ctx)?;
                    ctx.env.insert(*local, v);
                }
                hir::StmtKind::Let { mutable: true, .. } => {
                    self.error(
                        "mutable `let` is not supported in a deployment pipeline",
                        self.hir.stmt(sid).span,
                    );
                    return None;
                }
                hir::StmtKind::Let { init: None, .. } => {
                    self.error(
                        "uninitialized `let` is not supported in a deployment pipeline",
                        self.hir.stmt(sid).span,
                    );
                    return None;
                }
                other => {
                    self.error(
                        format!(
                            "unsupported statement in a deployment pipeline: {}",
                            stmt_kind_name(other)
                        ),
                        self.hir.stmt(sid).span,
                    );
                    return None;
                }
            }
        }
        let Some(tail) = b.tail else {
            self.error(
                "a deployment function must end in a tail expression that produces its result",
                b.span,
            );
            return None;
        };
        self.lower_expr(tail, ctx)
    }

    /// Lower an expression, emitting any needed op and returning its value.
    /// References are transparent at this IR level.
    fn lower_expr(&mut self, eid: ExprId, ctx: &mut FnCtx) -> Option<ValueId> {
        let node = self.hir.expr(eid);
        match &node.kind {
            // `&x` / `&mut x` — borrow is transparent for lowering.
            ExprKind::Unary {
                op: UnOp::Ref { .. },
                operand,
            } => self.lower_expr(*operand, ctx),

            ExprKind::Path { res, .. } => match res {
                Res::Local(id) | Res::SelfValue(id) => match ctx.env.get(id) {
                    Some(v) => Some(*v),
                    None => {
                        self.error("reference to an unbound value", node.span);
                        None
                    }
                },
                _ => {
                    self.error(
                        "unsupported name reference in a deployment pipeline",
                        node.span,
                    );
                    None
                }
            },

            ExprKind::Try(inner) => {
                let src = self.lower_expr(*inner, ctx)?;
                self.emit(ctx, DeployOp::Try { src }, eid, node.span)
            }

            ExprKind::Call { callee, args } => self.lower_call(*callee, args, eid, node.span, ctx),

            _ => {
                self.error(
                    format!(
                        "unsupported expression in a deployment pipeline: {}",
                        expr_kind_name(&node.kind)
                    ),
                    node.span,
                );
                None
            }
        }
    }

    fn lower_call(
        &mut self,
        callee: ExprId,
        args: &[ExprId],
        call_eid: ExprId,
        span: Span,
        ctx: &mut FnCtx,
    ) -> Option<ValueId> {
        match &self.hir.expr(callee).kind {
            // Method-style: `base.name::<turbofish>(args)`.
            ExprKind::Field {
                base,
                name,
                turbofish,
            } => {
                let method = text(self.file, *name).to_string();
                self.lower_method(
                    &method,
                    *base,
                    turbofish.as_ref(),
                    args,
                    call_eid,
                    *name,
                    ctx,
                )
            }
            // Bare builtin: `full::<..>(v)`, `concat::<axis>(a, b)`, `Ok(x)`.
            ExprKind::Path {
                res: Res::Builtin(b),
                turbofish,
                ..
            } => self.lower_builtin(*b, turbofish.as_ref(), args, call_eid, span, ctx),
            // User function call.
            ExprKind::Path {
                res: Res::Item(fn_id),
                ..
            } if matches!(self.hir.item(*fn_id).kind, ItemKind::Fn(_)) => {
                let Some(&idx) = self.fn_index.get(fn_id) else {
                    self.error("call to a function outside the deployment graph", span);
                    return None;
                };
                let mut lowered = Vec::with_capacity(args.len());
                for &a in args {
                    lowered.push(self.lower_expr(a, ctx)?);
                }
                self.emit(
                    ctx,
                    DeployOp::Call {
                        callee: idx,
                        args: lowered,
                    },
                    call_eid,
                    span,
                )
            }
            _ => {
                self.error("unsupported call target in a deployment pipeline", span);
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_method(
        &mut self,
        method: &str,
        base: ExprId,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        call_eid: ExprId,
        name_span: Span,
        ctx: &mut FnCtx,
    ) -> Option<ValueId> {
        match method {
            "refine" => {
                let src = self.lower_expr(base, ctx)?;
                let (dtype, dims) = self.type_and_shape_args(turbofish, name_span)?;
                self.emit(
                    ctx,
                    DeployOp::Refine { src, dtype, dims },
                    call_eid,
                    name_span,
                )
            }
            "permute" => {
                let src = self.lower_expr(base, ctx)?;
                let perm = self.index_list_arg(turbofish, name_span)?;
                self.emit(ctx, DeployOp::Permute { src, perm }, call_eid, name_span)
            }
            "reshape" => {
                let src = self.lower_expr(base, ctx)?;
                let g = self.require_turbofish(turbofish, name_span)?;
                if g.args.len() != 1 {
                    self.error("reshape expects a single shape generic argument", g.span);
                    return None;
                }
                let dims = self.shape_from_arg(&g.args[0], g.span)?;
                self.emit(ctx, DeployOp::Reshape { src, dims }, call_eid, name_span)
            }
            "cast" => {
                let src = self.lower_expr(base, ctx)?;
                let dtype = self.single_type_arg(turbofish, name_span)?;
                self.emit(ctx, DeployOp::Cast { src, dtype }, call_eid, name_span)
            }
            "sub" | "div" => {
                let lhs = self.lower_expr(base, ctx)?;
                let rhs = self.one_arg(args, name_span, ctx)?;
                let op = if method == "sub" {
                    DeployOp::Sub { lhs, rhs }
                } else {
                    DeployOp::Div { lhs, rhs }
                };
                self.emit(ctx, op, call_eid, name_span)
            }
            "softmax" | "argmax" => {
                let src = self.lower_expr(base, ctx)?;
                let axis = self.axis_arg(turbofish, name_span)?;
                let op = if method == "softmax" {
                    DeployOp::Softmax { src, axis }
                } else {
                    DeployOp::ArgMax { src, axis }
                };
                self.emit(ctx, op, call_eid, name_span)
            }
            "predict" => {
                let model = self.lower_expr(base, ctx)?;
                let input = self.one_arg(args, name_span, ctx)?;
                self.emit(ctx, DeployOp::Predict { model, input }, call_eid, name_span)
            }
            other => {
                self.error(
                    format!("tensor operation `{other}` is not supported in a deployment pipeline"),
                    name_span,
                );
                None
            }
        }
    }

    fn lower_builtin(
        &mut self,
        b: Builtin,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        call_eid: ExprId,
        span: Span,
        ctx: &mut FnCtx,
    ) -> Option<ValueId> {
        match b {
            Builtin::Ok => {
                let src = self.one_arg(args, span, ctx)?;
                self.emit(ctx, DeployOp::WrapOk { src }, call_eid, span)
            }
            Builtin::TensorFull => {
                let (dtype, dims) = self.type_and_shape_args(turbofish, span)?;
                let scalar = self.scalar_arg(args, dtype, span)?;
                self.emit(
                    ctx,
                    DeployOp::Full {
                        dtype,
                        dims,
                        scalar,
                    },
                    call_eid,
                    span,
                )
            }
            Builtin::TensorConcat => {
                let axis = self.axis_arg(turbofish, span)?;
                if args.len() != 2 {
                    self.error("`concat` expects two tensor arguments", span);
                    return None;
                }
                let lhs = self.lower_expr(args[0], ctx)?;
                let rhs = self.lower_expr(args[1], ctx)?;
                self.emit(ctx, DeployOp::Concat { axis, lhs, rhs }, call_eid, span)
            }
            other => {
                self.error(
                    format!(
                        "builtin `{}` is not supported in a deployment pipeline",
                        builtin_name(other)
                    ),
                    span,
                );
                None
            }
        }
    }

    // ---- operand helpers ----

    fn one_arg(&mut self, args: &[ExprId], span: Span, ctx: &mut FnCtx) -> Option<ValueId> {
        if args.len() != 1 {
            self.error("expected exactly one argument", span);
            return None;
        }
        self.lower_expr(args[0], ctx)
    }

    fn scalar_arg(&mut self, args: &[ExprId], dtype: DType, span: Span) -> Option<ScalarLit> {
        if args.len() != 1 {
            self.error("`full` expects exactly one scalar argument", span);
            return None;
        }
        let arg = self.hir.expr(args[0]);
        if !matches!(arg.kind, ExprKind::Lit(_)) {
            self.error(
                "`full` requires a numeric literal argument in a deployment pipeline",
                arg.span,
            );
            return None;
        }
        Some(ScalarLit {
            text: strip_num_suffix(text(self.file, arg.span)),
            dtype,
        })
    }

    // ---- generic-argument parsing ----

    fn type_and_shape_args(
        &mut self,
        turbofish: Option<&hir::GenericArgs>,
        span: Span,
    ) -> Option<(DType, Vec<DeployDim>)> {
        let g = self.require_turbofish(turbofish, span)?;
        if g.args.len() != 2 {
            self.error("expected a dtype and a shape generic argument", g.span);
            return None;
        }
        let dtype = self.dtype_from_arg(&g.args[0], g.span)?;
        let dims = self.shape_from_arg(&g.args[1], g.span)?;
        Some((dtype, dims))
    }

    fn single_type_arg(
        &mut self,
        turbofish: Option<&hir::GenericArgs>,
        span: Span,
    ) -> Option<DType> {
        let g = self.require_turbofish(turbofish, span)?;
        if g.args.len() != 1 {
            self.error("expected a single dtype generic argument", g.span);
            return None;
        }
        self.dtype_from_arg(&g.args[0], g.span)
    }

    fn index_list_arg(
        &mut self,
        turbofish: Option<&hir::GenericArgs>,
        span: Span,
    ) -> Option<Vec<u32>> {
        let g = self.require_turbofish(turbofish, span)?;
        if g.args.len() != 1 {
            self.error("expected a single index-list generic argument", g.span);
            return None;
        }
        let hir::GenericArg::Shape(shape) = &g.args[0] else {
            self.error("expected an index list `[..]`", g.span);
            return None;
        };
        let dims = self.literal_index_list(shape, g.span)?;
        Some(dims.into_iter().map(|d| d as u32).collect())
    }

    fn axis_arg(&mut self, turbofish: Option<&hir::GenericArgs>, span: Span) -> Option<u32> {
        let g = self.require_turbofish(turbofish, span)?;
        if g.args.len() != 1 {
            self.error("expected a single axis generic argument", g.span);
            return None;
        }
        match &g.args[0] {
            hir::GenericArg::Const(s) => self.parse_u64(*s).map(|n| n as u32),
            hir::GenericArg::Shape(shape) if shape.dims.len() == 1 => {
                if let hir::DimExpr::Lit(s) = &shape.dims[0] {
                    self.parse_u64(*s).map(|n| n as u32)
                } else {
                    self.error("axis must be a constant integer", g.span);
                    None
                }
            }
            _ => {
                self.error("axis must be a constant integer", g.span);
                None
            }
        }
    }

    fn require_turbofish<'g>(
        &mut self,
        turbofish: Option<&'g hir::GenericArgs>,
        span: Span,
    ) -> Option<&'g hir::GenericArgs> {
        match turbofish {
            Some(g) => Some(g),
            None => {
                self.error("this operation requires explicit generic arguments", span);
                None
            }
        }
    }

    fn dtype_from_arg(&mut self, arg: &hir::GenericArg, span: Span) -> Option<DType> {
        let hir::GenericArg::Type(tid) = arg else {
            self.error("expected a dtype", span);
            return None;
        };
        match &self.hir.ty(*tid).kind {
            hir::TypeKind::Primitive(p) => primitive_dtype(*p).or_else(|| {
                self.error("unsupported element type", span);
                None
            }),
            hir::TypeKind::Path { path, .. } => {
                dtype_by_name(text(self.file, path.span)).or_else(|| {
                    self.error("unsupported element type", span);
                    None
                })
            }
            _ => {
                self.error("expected a dtype", span);
                None
            }
        }
    }

    /// Lower a turbofish shape argument (`[..]`) to deployment dimensions, each
    /// of which may be a literal, a symbol, or a checked sum/product.
    fn shape_from_arg(&mut self, arg: &hir::GenericArg, span: Span) -> Option<Vec<DeployDim>> {
        let hir::GenericArg::Shape(shape) = arg else {
            self.error("expected a shape `[..]`", span);
            return None;
        };
        let mut dims = Vec::with_capacity(shape.dims.len());
        for d in &shape.dims {
            dims.push(self.deploy_dim_from_expr(d, span)?);
        }
        Some(dims)
    }

    /// Lower one source dimension expression to a [`DeployDim`]. Only literals,
    /// dimension variables, and their sums/products are representable in the
    /// deployment backend; subtraction and any other form fail here (`E0606`)
    /// rather than being silently erased.
    fn deploy_dim_from_expr(&mut self, expr: &DimExpr, span: Span) -> Option<DeployDim> {
        match expr {
            DimExpr::Lit(s) => Some(DeployDim::Literal(self.parse_u64(*s)?)),
            DimExpr::Var(s) => Some(DeployDim::Symbol(text(self.file, *s).to_string())),
            DimExpr::Binary { op, lhs, rhs } => {
                let l = self.deploy_dim_from_expr(lhs, span)?;
                let r = self.deploy_dim_from_expr(rhs, span)?;
                match op {
                    DimBinOp::Add => Some(DeployDim::Add(Box::new(l), Box::new(r))),
                    DimBinOp::Mul => Some(DeployDim::Mul(Box::new(l), Box::new(r))),
                    DimBinOp::Sub => {
                        self.diags.push(
                            Diagnostic::error(
                                "dimension subtraction is not supported in a deployment shape",
                                span,
                            )
                            .with_code(code::UNSUPPORTED_DIM),
                        );
                        None
                    }
                }
            }
            DimExpr::Error => None,
        }
    }

    /// A perm/index list must be literal integers.
    fn literal_index_list(&mut self, shape: &hir::ShapeArg, span: Span) -> Option<Vec<u64>> {
        let mut out = Vec::with_capacity(shape.dims.len());
        for d in &shape.dims {
            match d {
                DimExpr::Lit(s) => out.push(self.parse_u64(*s)?),
                _ => {
                    self.error("index lists must be constant integers", span);
                    return None;
                }
            }
        }
        Some(out)
    }

    fn parse_u64(&mut self, span: Span) -> Option<u64> {
        match text(self.file, span).trim().parse::<u64>() {
            Ok(n) => Some(n),
            Err(_) => {
                self.error("expected a non-negative integer literal", span);
                None
            }
        }
    }

    // ---- typing ----

    fn emit(
        &mut self,
        ctx: &mut FnCtx,
        op: DeployOp,
        ty_of: ExprId,
        span: Span,
    ) -> Option<ValueId> {
        let ty = self.deploy_ty(ty_of, span)?;
        let result = ctx.fresh();
        ctx.body.push(DeployStmt {
            result,
            op,
            ty,
            span,
        });
        Some(result)
    }

    fn deploy_ty(&mut self, eid: ExprId, span: Span) -> Option<DeployTy> {
        match self.tables.expr_types.get(&eid) {
            Some(ty) => self.convert_ty(ty, span),
            None => {
                self.error("internal: missing type for a deployment expression", span);
                None
            }
        }
    }

    /// Build a deployment type from a source type annotation, preserving
    /// symbolic dimension names. References are transparent at this IR level.
    fn deploy_ty_from_ast(&mut self, ty: TypeId, span: Span) -> Option<DeployTy> {
        match &self.hir.ty(ty).kind {
            hir::TypeKind::Ref { inner, .. } => self.deploy_ty_from_ast(*inner, span),
            hir::TypeKind::Path { path, args, .. } => {
                let name = text(self.file, path.span).trim().to_string();
                match name.as_str() {
                    "TensorAny" => Some(DeployTy::TensorAny),
                    "Tensor" => {
                        let args = args.as_ref()?;
                        let dtype_arg = args.args.first()?;
                        let dtype = self.dtype_from_arg(dtype_arg, span)?;
                        let shape = args.args.iter().find_map(|a| match a {
                            hir::GenericArg::Shape(s) => Some(s),
                            _ => None,
                        })?;
                        let mut dims = Vec::with_capacity(shape.dims.len());
                        for d in &shape.dims {
                            dims.push(self.deploy_dim_from_expr(d, span)?);
                        }
                        Some(DeployTy::Tensor(TensorShape { dtype, dims }))
                    }
                    "Result" => {
                        let args = args.as_ref()?;
                        let hir::GenericArg::Type(ok) = args.args.first()? else {
                            self.error("unsupported `Result` argument in a deployment type", span);
                            return None;
                        };
                        Some(DeployTy::Result(Box::new(
                            self.deploy_ty_from_ast(*ok, span)?,
                        )))
                    }
                    other if self.model_names.contains(other) => Some(DeployTy::Model),
                    _ => {
                        self.error(
                            format!("unsupported type `{name}` in a deployment pipeline"),
                            span,
                        );
                        None
                    }
                }
            }
            _ => {
                self.error("unsupported type in a deployment pipeline", span);
                None
            }
        }
    }

    fn convert_ty(&mut self, ty: &Ty, span: Span) -> Option<DeployTy> {
        match ty {
            Ty::Extension(ext) => match &**ext {
                ExtensionTy::Tensor(TensorKind::Tensor(t)) => {
                    // This shape (an intermediate value's checked type) is not
                    // emitted; the emitted shapes come from source turbofish and
                    // parameter annotations. Build a symbolic form tolerant of
                    // non-constant dims so lowering does not reject the pipeline.
                    let dims = t.shape.dims.iter().map(deploy_dim_from_poly).collect();
                    Some(DeployTy::Tensor(TensorShape {
                        dtype: t.dtype,
                        dims,
                    }))
                }
                ExtensionTy::Tensor(TensorKind::TensorAny) => Some(DeployTy::TensorAny),
                ExtensionTy::Tensor(TensorKind::TensorDyn(_)) => {
                    self.error(
                        "`TensorDyn` values are not supported in a deployment pipeline",
                        span,
                    );
                    None
                }
                ExtensionTy::Model(_) => Some(DeployTy::Model),
                ExtensionTy::ModelError => {
                    self.error("unresolved model type in a deployment pipeline", span);
                    None
                }
            },
            Ty::Core(hir::CoreType::Result, args) if args.len() == 2 => {
                let inner = self.convert_ty(&args[0], span)?;
                Some(DeployTy::Result(Box::new(inner)))
            }
            _ => {
                self.error("unsupported value type in a deployment pipeline", span);
                None
            }
        }
    }
}

/// Validate the entry signature against the deployment ABI (plan §6.1):
/// `fn(entry_model: Model, input: TensorAny) -> Result<Tensor<Int64, [1]>, String>`.
/// Build a symbolic [`DeployDim`] from a checked dimension polynomial. Used only
/// for intermediate value types, which are never emitted, so variable names are
/// synthesized from ids rather than provenance labels.
fn deploy_dim_from_poly(poly: &crate::extensions::tensor::dim::Poly) -> DeployDim {
    if let Some(n) = poly.as_constant() {
        return DeployDim::Literal(n.max(0) as u64);
    }
    let mut sum: Option<DeployDim> = None;
    for (vars, coeff) in poly.iter_terms() {
        let mut term: Option<DeployDim> = if coeff == 1 && !vars.is_empty() {
            None
        } else {
            Some(DeployDim::Literal(coeff.max(0) as u64))
        };
        for v in vars {
            let sym = DeployDim::Symbol(format!("d{}", v.0));
            term = Some(match term.take() {
                Some(t) => DeployDim::Mul(Box::new(t), Box::new(sym)),
                None => sym,
            });
        }
        let term = term.unwrap_or(DeployDim::Literal(0));
        sum = Some(match sum.take() {
            Some(s) => DeployDim::Add(Box::new(s), Box::new(term)),
            None => term,
        });
    }
    sum.unwrap_or(DeployDim::Literal(0))
}

/// Returns the selected model's declared type name on success.
fn validate_entry_abi(
    hir: &Hir,
    file: &SourceFile,
    entry: &DeploymentFunction,
    diags: &mut Vec<Diagnostic>,
) -> Option<String> {
    let abi_err = |diags: &mut Vec<Diagnostic>, msg: String| {
        diags.push(Diagnostic::error(msg, entry.span).with_code(code::ENTRY_ABI));
    };

    if entry.params.len() != 2 {
        abi_err(
            diags,
            format!(
                "deployment entry must take exactly two parameters (a model and a `TensorAny` input), found {}",
                entry.params.len()
            ),
        );
        return None;
    }
    if entry.params[0].ty != DeployTy::Model {
        abi_err(
            diags,
            "the first entry parameter must be a model".to_string(),
        );
        return None;
    }
    if entry.params[1].ty != DeployTy::TensorAny {
        abi_err(
            diags,
            "the second entry parameter must be a `TensorAny` input".to_string(),
        );
        return None;
    }
    // The entry must return `Result<Tensor<..>, String>`; the tensor's dtype and
    // (possibly symbolic) shape are the pipeline's, not fixed by the ABI.
    if !matches!(&entry.ret, DeployTy::Result(inner) if matches!(**inner, DeployTy::Tensor(_))) {
        abi_err(
            diags,
            format!(
                "deployment entry must return `Result<Tensor<..>, String>`, found `{}`",
                entry.ret
            ),
        );
        return None;
    }

    // Recover the selected model's declaration name from the front end.
    let model_name = model_decl_name(hir, file);
    if model_name.is_none() {
        diags.push(
            Diagnostic::error(
                "exactly one `model` declaration must be reachable from the entry",
                entry.span,
            )
            .with_code(code::MODEL_SELECTION),
        );
    }
    model_name
}

/// Find the single `model` declaration in the program. The prototype supports
/// exactly one; zero or many is a selection error.
fn model_decl_name(hir: &Hir, file: &SourceFile) -> Option<String> {
    let mut found = None;
    for item in &hir.items {
        if let ItemKind::Model(def) = &item.kind {
            if found.is_some() {
                return None;
            }
            found = Some(def.name);
        }
    }
    found.map(|span| text(file, span).to_string())
}

// ------------------------------------------------------------- helpers --

fn text(file: &SourceFile, span: Span) -> &str {
    &file.src[span.lo as usize..span.hi as usize]
}

fn strip_num_suffix(s: &str) -> String {
    let s = s.trim();
    for suffix in [
        "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
    ] {
        if let Some(stripped) = s.strip_suffix(suffix) {
            return stripped.to_string();
        }
    }
    s.to_string()
}

fn primitive_dtype(p: crate::ast::Primitive) -> Option<DType> {
    use crate::ast::Primitive::*;
    Some(match p {
        Int8 => DType::Int8,
        Int16 => DType::Int16,
        Int32 => DType::Int32,
        Int64 => DType::Int64,
        UInt8 => DType::UInt8,
        UInt16 => DType::UInt16,
        UInt32 => DType::UInt32,
        UInt64 => DType::UInt64,
        Float32 => DType::Float32,
        Float64 => DType::Float64,
        Bool => DType::Bool,
        _ => return None,
    })
}

fn dtype_by_name(name: &str) -> Option<DType> {
    Some(match name {
        "Int8" => DType::Int8,
        "Int16" => DType::Int16,
        "Int32" => DType::Int32,
        "Int64" => DType::Int64,
        "UInt8" => DType::UInt8,
        "UInt16" => DType::UInt16,
        "UInt32" => DType::UInt32,
        "UInt64" => DType::UInt64,
        "Float16" => DType::Float16,
        "BFloat16" => DType::BFloat16,
        "Float32" => DType::Float32,
        "Float64" => DType::Float64,
        "Bool" => DType::Bool,
        _ => return None,
    })
}

fn builtin_name(b: Builtin) -> &'static str {
    // Best-effort label for diagnostics.
    match b {
        Builtin::TensorZeros => "zeros",
        Builtin::TensorOnes => "ones",
        Builtin::TensorFromVec => "from_vec",
        Builtin::TensorAdd => "add",
        Builtin::TensorMul => "mul",
        Builtin::TensorMatMul => "matmul",
        Builtin::TensorReshape => "reshape",
        _ => "operation",
    }
}

fn expr_kind_name(k: &ExprKind) -> &'static str {
    match k {
        ExprKind::If { .. } => "an `if` expression",
        ExprKind::Match { .. } => "a `match` expression",
        ExprKind::Loop { .. } => "a `loop`",
        ExprKind::While { .. } => "a `while` loop",
        ExprKind::For { .. } => "a `for` loop",
        ExprKind::Index { .. } => "an index expression",
        ExprKind::Binary { .. } => "a binary operator expression",
        ExprKind::Assign { .. } => "an assignment",
        ExprKind::StructLit { .. } => "a struct literal",
        ExprKind::Lit(_) => "a bare literal",
        ExprKind::Tuple(_) => "a tuple",
        ExprKind::Array(_) => "an array literal",
        _ => "this construct",
    }
}

fn stmt_kind_name(k: &hir::StmtKind) -> &'static str {
    match k {
        hir::StmtKind::Return(_) => "`return`",
        hir::StmtKind::Break(_) => "`break`",
        hir::StmtKind::Continue => "`continue`",
        hir::StmtKind::Item(_) => "a nested item",
        hir::StmtKind::Expr { .. } => "a bare expression statement",
        _ => "this statement",
    }
}
