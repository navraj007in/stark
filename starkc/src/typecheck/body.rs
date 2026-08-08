//! **AS7 Packet 9b — expression, statement and block checking.**
//!
//! The top of the checking hierarchy below `items`: `body` may use `patterns`, `bounds`,
//! `convert`, `traits`, `infer`, `state` and `types`. Nothing below may reach back into it.
//!
//! Everything that gives an *expression* a type lives here — calls, operators, field and index
//! access, control-flow typing, builtin call-form validation, and the publication of what a call
//! resolved to.
//!
//! **`resolve_method` is here, not in `traits`, and that is a rule rather than an accident.**
//! Method *selection* is a trait question, but this function also evaluates the receiver and every
//! argument — five `check_expr` calls. The owner's constraint on the decomposition is explicit:
//! any trait-method invocation path that evaluates arguments stays in `body`, because putting it
//! in `traits` makes `traits <-> body` a cycle. Packet 9a measured every selection function for
//! `check_expr` before moving any of them; this is the one that failed the test.
//!
//! **The tensor entry points are the Core half of AS6's boundary.** `check_tensor_op` and friends
//! locate the operation, validate the call form and evaluate arguments, then hand already-typed
//! operands to `extensions::tensor::check`. They must not regain the semantic decisions AS6 moved
//! out, and `as7_module_dependencies` asserts the extension never calls back into `check_expr`.

use super::state::TypeChecker;
use super::state::{PublishedEnv, SelfScope};
use super::traits::core_trait_source_name;
use super::types::{
    bound_receiver_ty, receiver_adjustment_for, BindMode, CallableDeclId, CallableSigTy,
    CalleeSelection, ControlSummary, DispatchProvenance, DisplayPath, DisplayStep, ExtensionTy,
    FnSigTy, GenericBinder, GenericEnvironment, LoopContext, MethodCandidate, ModelTy,
    ReceiverAdjustment, ReceiverBinding, Ty, VariantFields,
};
use super::types::{
    convert_float_suffix, convert_int_suffix, is_float_primitive, is_integer, is_numeric,
    strip_ref, substitute_ty, ty_contains_infer, unit_or_tuple, CallableUse,
};

use super::types::standard_display_type;
use super::types::BoundMethod;
use super::types::{is_cast_numeric, type_is_sized};
use crate::ast::{AssignOp, BinOp, Lit, Primitive, UnOp};
use crate::diag::Diagnostic;
use crate::extensions::tensor::check as tensor_check;
use crate::extensions::tensor::rules::TENSOR_OPS;
use crate::hir::{self, BlockId, Builtin, CoreType, ExprId, ItemId, LocalId, Res, StmtId};
use crate::literal;
use crate::source::Span;
use std::collections::{HashMap, HashSet};

impl TypeChecker<'_> {
    pub(super) fn const_eval_i128(
        &self,
        expr_id: ExprId,
        visiting: &mut HashSet<ItemId>,
    ) -> Option<i128> {
        let expr = self.hir.expr(expr_id);
        match &expr.kind {
            hir::ExprKind::Lit(Lit::Int { base, suffix }) => {
                literal::parse_int_literal(self.text(expr.span), *base, *suffix)
            }
            hir::ExprKind::Path {
                res: Res::Item(item_id),
                ..
            } => match &self.hir.item(*item_id).kind {
                hir::ItemKind::Const { value, .. } => {
                    if !visiting.insert(*item_id) {
                        return None;
                    }
                    let result = self.const_eval_i128(*value, visiting);
                    visiting.remove(item_id);
                    result
                }
                _ => None,
            },
            hir::ExprKind::Unary { op, operand } => {
                let value = self.const_eval_i128(*operand, visiting)?;
                match op {
                    UnOp::Neg => value.checked_neg(),
                    UnOp::BitNot => Some(!value),
                    _ => None,
                }
            }
            hir::ExprKind::Binary { op, lhs, rhs } => {
                let lhs = self.const_eval_i128(*lhs, visiting)?;
                let rhs = self.const_eval_i128(*rhs, visiting)?;
                match op {
                    BinOp::Add => lhs.checked_add(rhs),
                    BinOp::Sub => lhs.checked_sub(rhs),
                    BinOp::Mul => lhs.checked_mul(rhs),
                    BinOp::Div => lhs.checked_div(rhs),
                    BinOp::Rem => lhs.checked_rem(rhs),
                    BinOp::Pow => u32::try_from(rhs).ok().and_then(|rhs| lhs.checked_pow(rhs)),
                    BinOp::BitAnd => Some(lhs & rhs),
                    BinOp::BitOr => Some(lhs | rhs),
                    BinOp::BitXor => Some(lhs ^ rhs),
                    BinOp::Shl => u32::try_from(rhs).ok().and_then(|rhs| lhs.checked_shl(rhs)),
                    BinOp::Shr => u32::try_from(rhs).ok().and_then(|rhs| lhs.checked_shr(rhs)),
                    _ => None,
                }
            }
            hir::ExprKind::Cast { expr, .. } => self.const_eval_i128(*expr, visiting),
            hir::ExprKind::Block(block) => {
                let block = self.hir.block(*block);
                if block.stmts.iter().any(|statement| {
                    !matches!(
                        &self.hir.stmt(*statement).kind,
                        hir::StmtKind::Empty | hir::StmtKind::Expr { .. }
                    )
                }) {
                    return None;
                }
                for statement in &block.stmts {
                    if let hir::StmtKind::Expr { expr, .. } = &self.hir.stmt(*statement).kind {
                        self.const_eval_i128(*expr, visiting)?;
                    }
                }
                block
                    .tail
                    .and_then(|tail| self.const_eval_i128(tail, visiting))
            }
            _ => None,
        }
    }

    pub(super) fn builtin_type(&mut self, builtin: Builtin) -> Ty {
        let unit = Ty::Primitive(Primitive::Unit);
        match builtin {
            Builtin::Print | Builtin::Println => Ty::Fn {
                params: vec![self.new_type_var()],
                ret: Box::new(unit),
            },
            Builtin::Panic => Ty::Fn {
                params: vec![self.new_type_var()],
                ret: Box::new(Ty::Never),
            },
            Builtin::Assert => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::Bool)],
                ret: Box::new(unit),
            },
            Builtin::AssertEq | Builtin::AssertNe => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![value.clone(), value],
                    ret: Box::new(unit),
                }
            }
            Builtin::Sqrt => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::Float64)],
                ret: Box::new(Ty::Primitive(Primitive::Float64)),
            },
            Builtin::Drop => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![value],
                    ret: Box::new(unit),
                }
            }
            Builtin::StringFrom => Ty::Fn {
                params: vec![Ty::Ref {
                    mutable: false,
                    inner: Box::new(Ty::Primitive(Primitive::Str)),
                }],
                ret: Box::new(Ty::Primitive(Primitive::String)),
            },
            Builtin::StringNew => Ty::Fn {
                params: Vec::new(),
                ret: Box::new(Ty::Primitive(Primitive::String)),
            },
            Builtin::StringWithCapacity => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::UInt64)],
                ret: Box::new(Ty::Primitive(Primitive::String)),
            },
            Builtin::CharFromU32 => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::UInt32)],
                ret: Box::new(Ty::Core(
                    CoreType::Option,
                    vec![Ty::Primitive(Primitive::Char)],
                )),
            },
            Builtin::VecNew => Ty::Fn {
                params: Vec::new(),
                ret: Box::new(Ty::Core(CoreType::Vec, vec![self.new_type_var()])),
            },
            Builtin::VecWithCapacity => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::UInt64)],
                ret: Box::new(Ty::Core(CoreType::Vec, vec![self.new_type_var()])),
            },
            Builtin::HashMapNew => {
                let key = self.new_type_var();
                let val = self.new_type_var();
                Ty::Fn {
                    params: Vec::new(),
                    ret: Box::new(Ty::Core(CoreType::HashMap, vec![key, val])),
                }
            }
            Builtin::HashMapWithCapacity => {
                let key = self.new_type_var();
                let val = self.new_type_var();
                Ty::Fn {
                    params: vec![Ty::Primitive(Primitive::UInt64)],
                    ret: Box::new(Ty::Core(CoreType::HashMap, vec![key, val])),
                }
            }
            Builtin::HashSetNew => {
                let val = self.new_type_var();
                Ty::Fn {
                    params: Vec::new(),
                    ret: Box::new(Ty::Core(CoreType::HashSet, vec![val])),
                }
            }
            Builtin::BoxNew => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![value.clone()],
                    ret: Box::new(Ty::Core(CoreType::Box, vec![value])),
                }
            }
            Builtin::BoxIntoInner => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![Ty::Core(CoreType::Box, vec![value.clone()])],
                    ret: Box::new(value),
                }
            }
            Builtin::ReadFile => Ty::Fn {
                params: vec![Ty::Ref {
                    mutable: false,
                    inner: Box::new(Ty::Primitive(Primitive::Str)),
                }],
                ret: Box::new(Ty::Core(
                    CoreType::Result,
                    vec![
                        Ty::Primitive(Primitive::String),
                        Ty::Core(CoreType::IOError, Vec::new()),
                    ],
                )),
            },
            Builtin::WriteFile => Ty::Fn {
                params: vec![
                    Ty::Ref {
                        mutable: false,
                        inner: Box::new(Ty::Primitive(Primitive::Str)),
                    },
                    Ty::Ref {
                        mutable: false,
                        inner: Box::new(Ty::Primitive(Primitive::Str)),
                    },
                ],
                ret: Box::new(Ty::Core(
                    CoreType::Result,
                    vec![
                        Ty::Primitive(Primitive::Unit),
                        Ty::Core(CoreType::IOError, Vec::new()),
                    ],
                )),
            },
            Builtin::FileOpen | Builtin::FileCreate => Ty::Fn {
                params: vec![Ty::Ref {
                    mutable: false,
                    inner: Box::new(Ty::Primitive(Primitive::Str)),
                }],
                ret: Box::new(Ty::Core(
                    CoreType::Result,
                    vec![
                        Ty::Core(CoreType::File, Vec::new()),
                        Ty::Core(CoreType::IOError, Vec::new()),
                    ],
                )),
            },
            Builtin::Some => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![value.clone()],
                    ret: Box::new(Ty::Core(CoreType::Option, vec![value])),
                }
            }
            Builtin::None => Ty::Core(CoreType::Option, vec![self.new_type_var()]),
            Builtin::Ok => {
                let value = self.new_type_var();
                let error = self.new_type_var();
                Ty::Fn {
                    params: vec![value.clone()],
                    ret: Box::new(Ty::Core(CoreType::Result, vec![value, error])),
                }
            }
            Builtin::Err => {
                let value = self.new_type_var();
                let error = self.new_type_var();
                Ty::Fn {
                    params: vec![error.clone()],
                    ret: Box::new(Ty::Core(CoreType::Result, vec![value, error])),
                }
            }
            // AS6: one arm, not thirty-three patterns for one behaviour. Every tensor
            // operation's *signature* is refined by the extension's own rules
            // (`check_tensor_op`); Core only needs to know a call is a call.
            Builtin::Tensor(_) => Ty::Fn {
                params: vec![],
                ret: Box::new(self.new_type_var()),
            },
            Builtin::SizeOf | Builtin::AlignOf => Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Primitive(Primitive::UInt64)),
            },
            Builtin::Swap => {
                let value = self.new_type_var();
                let ref_ty = Ty::Ref {
                    mutable: true,
                    inner: Box::new(value),
                };
                Ty::Fn {
                    params: vec![ref_ty.clone(), ref_ty],
                    ret: Box::new(unit),
                }
            }
            Builtin::Replace => {
                let value = self.new_type_var();
                let ref_ty = Ty::Ref {
                    mutable: true,
                    inner: Box::new(value.clone()),
                };
                Ty::Fn {
                    params: vec![ref_ty, value.clone()],
                    ret: Box::new(value),
                }
            }
            Builtin::Take => {
                let value = self.new_type_var();
                let ref_ty = Ty::Ref {
                    mutable: true,
                    inner: Box::new(value.clone()),
                };
                Ty::Fn {
                    params: vec![ref_ty],
                    ret: Box::new(value),
                }
            }
            // -- Phase 4E: Math constants and functions --
            Builtin::MathPi | Builtin::MathE => Ty::Primitive(Primitive::Float64),
            Builtin::MathAbs => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![value.clone()],
                    ret: Box::new(value),
                }
            }
            Builtin::MathMin | Builtin::MathMax => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![value.clone(), value.clone()],
                    ret: Box::new(value),
                }
            }
            Builtin::MathClamp => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![value.clone(), value.clone(), value.clone()],
                    ret: Box::new(value),
                }
            }
            Builtin::Pow | Builtin::Atan2 => Ty::Fn {
                params: vec![
                    Ty::Primitive(Primitive::Float64),
                    Ty::Primitive(Primitive::Float64),
                ],
                ret: Box::new(Ty::Primitive(Primitive::Float64)),
            },
            Builtin::Log
            | Builtin::Log10
            | Builtin::Exp
            | Builtin::Sin
            | Builtin::Cos
            | Builtin::Tan
            | Builtin::Asin
            | Builtin::Acos
            | Builtin::Atan
            | Builtin::Floor
            | Builtin::Ceil
            | Builtin::Round
            | Builtin::Trunc => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::Float64)],
                ret: Box::new(Ty::Primitive(Primitive::Float64)),
            },
            // -- Phase 4E: stderr --
            //
            // DEV-174: typed as a fresh variable, exactly like `print`/`println`.
            //
            // 06-Standard-Library declares `fn eprint<T: Display>(value: T)` and the
            // `eprintln`/`eprint` analogues, and PRINT-DISPLAY-001 covers all four by name. This
            // took `&str` instead, so `eprintln(s)` with an owned `String` — let alone any other
            // `Display` type — was rejected while `println(s)` was accepted. The stderr half of the
            // runtime surface has carried the full display family since 0.1-A13
            // (`EprintlnInt64`, `EprintBool`, …); only the signature lagged.
            Builtin::Eprint | Builtin::Eprintln => Ty::Fn {
                params: vec![self.new_type_var()],
                ret: Box::new(unit),
            },
            // -- Phase 4E: Random (simple LCG per `06-Standard-Library.md`) --
            Builtin::RandomNew => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::UInt64)],
                ret: Box::new(Ty::Core(CoreType::Random, Vec::new())),
            },
            // WP-C2.2 (DEV-027): Ordering's unit variants.
            Builtin::OrderingLess | Builtin::OrderingEqual | Builtin::OrderingGreater => {
                Ty::Core(CoreType::Ordering, Vec::new())
            }
            // -- Phase 4E: IOError variant constructors --
            Builtin::IOErrorNotFound
            | Builtin::IOErrorPermissionDenied
            | Builtin::IOErrorAlreadyExists
            | Builtin::IOErrorInvalidInput => Ty::Core(CoreType::IOError, Vec::new()),
            Builtin::IOErrorOther => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::String)],
                ret: Box::new(Ty::Core(CoreType::IOError, Vec::new())),
            },
        }
    }

    pub(super) fn check_field_initializers(
        &mut self,
        owner: Option<ItemId>,
        expected_fields: &HashMap<String, Ty>,
        map: &HashMap<String, Ty>,
        fields: &[hir::FieldInit],
        span: Span,
    ) {
        let mut provided = HashSet::new();
        for field in fields {
            let name = self.text(field.name).to_string();
            provided.insert(name.clone());
            if let Some(expected) = expected_fields.get(&name) {
                // WP-C6.2b-F1: constructing with a private field is inaccessible outside its module.
                if let Some(struct_id) = owner {
                    let is_pub = self.struct_field_is_pub(struct_id, &name);
                    self.check_member_visible(is_pub, struct_id, "field", &name, field.name);
                }
                if let Some(value) = field.expr {
                    let actual = self.check_expr(value);
                    let expected = self.instantiate_ty(expected, map);
                    let _ = self.unify(expected, actual, field.name);
                }
            } else {
                self.diags.push(
                    Diagnostic::error(format!("field '{name}' does not exist"), field.name)
                        .with_code("E0001"),
                );
            }
        }
        for missing in expected_fields
            .keys()
            .filter(|name| !provided.contains(*name))
        {
            self.diags.push(
                Diagnostic::error(format!("missing field '{missing}'"), span).with_code("E0001"),
            );
        }
    }

    /// Build the ordered binder list for one callable use.
    ///
    /// AS3 extracted this from `publish_callable_env` so the instantiation table and the
    /// `CallableUse` record are built by the SAME code. Two constructions of "what generic
    /// environment did this use select" is the shape of defect this packet exists to remove, and
    /// duplicating it here to publish a second table would have been an immediate instance.
    pub(super) fn env_bindings(
        self_ty: &Option<Ty>,
        impl_names: &[String],
        own_names: &[String],
        own_is_method: bool,
        map: &HashMap<String, Ty>,
    ) -> Vec<(GenericBinder, Ty)> {
        let mut bindings: Vec<(GenericBinder, Ty)> = Vec::new();
        if let Some(self_ty) = self_ty {
            // **`Self` is substituted through the WHOLE map here, not only through the binders
            // named below.** A trait default invoked on a generic impl publishes
            // `Self = Tagged<T>` while the environment carries the TRAIT's generics — which for
            // `trait Describe` are none — so `T` would have nothing to resolve against and the
            // install would refuse a correct program. `map` holds every binding the checker
            // selected, including the impl's, so resolving here uses all of them regardless of
            // which are individually named as binders.
            bindings.push((GenericBinder::SelfType, substitute_ty(self_ty, map)));
        }
        for (index, name) in impl_names.iter().enumerate() {
            if let Some(ty) = map.get(name) {
                bindings.push((
                    GenericBinder::ImplParam {
                        index,
                        name: name.clone(),
                    },
                    ty.clone(),
                ));
            }
        }
        for (index, name) in own_names.iter().enumerate() {
            if let Some(ty) = map.get(name) {
                let binder = if own_is_method {
                    GenericBinder::MethodParam {
                        index,
                        name: name.clone(),
                    }
                } else {
                    GenericBinder::FunctionParam {
                        index,
                        name: name.clone(),
                    }
                };
                bindings.push((binder, ty.clone()));
            }
        }
        bindings
    }

    pub(super) fn check_block(&mut self, block_id: BlockId, state: &mut HashSet<LocalId>) -> Ty {
        let block = self.hir.block(block_id);
        // Refinement-introduced existential dimensions live through the rest
        // of this block and do not escape it.
        let saved_dim_scope = self.dim_scope.clone();

        // Scope state for block variables
        let mut reachable = true;
        for &stmt_id in &block.stmts {
            if !reachable {
                self.diags.push(
                    Diagnostic::warning("unreachable code", self.hir.stmt(stmt_id).span)
                        .with_code("W0005"),
                );
            }
            self.check_stmt(stmt_id, state);
            if reachable && !self.control_summary_stmt(stmt_id).can_complete {
                reachable = false;
            }
        }

        let result = if let Some(tail_expr) = block.tail {
            self.check_expr(tail_expr)
        } else {
            Ty::Primitive(Primitive::Unit)
        };
        self.dim_scope = saved_dim_scope;
        result
    }

    pub(super) fn check_stmt(&mut self, stmt_id: StmtId, state: &mut HashSet<LocalId>) {
        let stmt = self.hir.stmt(stmt_id);
        match &stmt.kind {
            hir::StmtKind::Empty => {}
            hir::StmtKind::Expr { expr, .. } => {
                let _ = self.check_expr(*expr);
            }
            hir::StmtKind::Let {
                mutable,
                name: _,
                local,
                ty,
                init,
            } => {
                let mut expected_ty = self.new_type_var();
                if let Some(ty_id) = ty {
                    expected_ty = self.convert_hir_type(*ty_id);
                }

                self.local_mutability.insert(*local, *mutable);
                self.local_types.insert(*local, expected_ty.clone());

                if let Some(init_expr) = init {
                    let init_ty = self.check_expr(*init_expr);
                    let _ = self.unify(expected_ty, init_ty, stmt.span);
                    state.insert(*local); // Initialized
                } else {
                    // Uninitialized
                    state.remove(local);
                }
                if self.is_unsized_value_type(
                    &self.resolve(self.local_types.get(local).unwrap_or(&Ty::Error)),
                ) {
                    self.diags.push(
                        Diagnostic::error(
                            "unsized local types must be behind a reference",
                            stmt.span,
                        )
                        .with_code("E0001"),
                    );
                }
            }
            hir::StmtKind::Return(expr) => {
                let val_ty = if let Some(e) = expr {
                    self.check_expr(*e)
                } else {
                    Ty::Primitive(Primitive::Unit)
                };

                if let Some(expected) = &self.current_fn_ret {
                    let _ = self.unify(expected.clone(), val_ty, stmt.span);
                } else {
                    self.diags.push(
                        Diagnostic::error("return outside function body", stmt.span)
                            .with_code("E0301"),
                    );
                }
            }
            hir::StmtKind::Break(expr) => {
                if self.loop_nesting == 0 {
                    self.diags.push(
                        Diagnostic::error("break outside loop", stmt.span).with_code("E0302"),
                    );
                    if let Some(e) = expr {
                        let _ = self.check_expr(*e);
                    }
                } else {
                    let break_ty =
                        expr.map_or(Ty::Primitive(Primitive::Unit), |e| self.check_expr(e));
                    let (allows_value, expected) = self
                        .loop_contexts
                        .last()
                        .map(|context| (context.allows_value, context.break_ty.clone()))
                        .unwrap_or((false, Ty::Error));
                    if expr.is_some() && !allows_value {
                        self.diags.push(
                            Diagnostic::error(
                                "break values are allowed only in loop expressions",
                                stmt.span,
                            )
                            .with_code("E0001"),
                        );
                    } else {
                        let _ = self.unify(expected, break_ty, stmt.span);
                    }
                    if let Some(context) = self.loop_contexts.last_mut() {
                        context.has_break = true;
                    }
                }
            }
            hir::StmtKind::Continue => {
                if self.loop_nesting == 0 {
                    self.diags.push(
                        Diagnostic::error("continue outside loop", stmt.span).with_code("E0302"),
                    );
                }
            }
            hir::StmtKind::Item(item_id) => {
                // Snippet-level items are ignored in the checker's execution flow
                let item = self.hir.item(*item_id);
                if let hir::ItemKind::Fn(def) = &item.kind {
                    let params = def
                        .sig
                        .params
                        .iter()
                        .map(|p| self.convert_hir_type(p.ty))
                        .collect();
                    let ret = match def.sig.ret {
                        hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
                        hir::RetTy::Ty(t) => self.convert_hir_type(t),
                        hir::RetTy::Never(_) => Ty::Never,
                    };
                    self.fn_sigs.insert(*item_id, FnSigTy { params, ret });
                }
            }
            hir::StmtKind::Error => {}
        }
    }

    /// WP-FMT-001: check one interpolation field.
    ///
    /// The specification decides what the type must be. A padding-only field (or none at all) asks
    /// for `Display` and nothing more; a numeric mode asks for a concrete integer or float. Every
    /// rejection happens HERE, at type checking — §6.7's requirement that no bad type/spec pairing
    /// reaches run time.
    pub(super) fn check_format_field(
        &mut self,
        expr: ExprId,
        spec: &crate::ast::FormatSpec,
        expr_span: Span,
    ) {
        use crate::ast::FormatKind;
        let ty = self.check_expr(expr);
        let ty = self.default_int_literals_deep(&ty);
        if matches!(ty, Ty::Error) {
            return;
        }
        let spec_span = spec.span.unwrap_or(expr_span);
        // **DEV-206: do not strip the reference that MAKES the value.**
        //
        // Stripping is right for `fn render<T: Display>(v: &T)` — `Display::fmt` borrows anyway
        // (STD-FORMAT-001), so a reference to a displayable type is displayable. It is wrong for
        // `&[T]`: the pointee is UNSIZED, the reference is not incidental, and stripping it turns
        // the one displayable spelling into the one that is not a value at all.
        //
        // Found by the value-context property, which required every context to accept the
        // reference form and caught interpolation still rejecting `&[Int32]` after `println`
        // had been repaired.
        let stripped = match &ty {
            Ty::Ref { inner, .. } if !type_is_sized(inner) => ty.clone(),
            other => strip_ref(other).clone(),
        };

        // A numeric mode requires a numeric type. `Display` does NOT imply integer formatting
        // (§11.5), so a generic `T: Display` is refused here rather than given a meaning it has
        // not proved — inventing a numeric bound to make it compile is out of scope.
        //
        // The guards carry the type requirement, and the final arm enumerates every `FormatKind`
        // explicitly rather than using `_`: a new format type must force a decision here about
        // which types accept it.
        match spec.kind {
            Some(
                FormatKind::Bin | FormatKind::Oct | FormatKind::LowerHex | FormatKind::UpperHex,
            ) if !matches!(&stripped, Ty::Primitive(p) if is_integer(*p)) => {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "type '{}' cannot be formatted in another base",
                            self.ty_to_string(&ty)
                        ),
                        spec_span,
                    )
                    .with_code("E0306")
                    .with_label("'b', 'o', 'x' and 'X' require an integer type"),
                );
                return;
            }
            Some(FormatKind::Fixed) if !matches!(&stripped, Ty::Primitive(p) if is_float_primitive(*p)) =>
            {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "type '{}' cannot be formatted with fixed precision",
                            self.ty_to_string(&ty)
                        ),
                        spec_span,
                    )
                    .with_code("E0306")
                    .with_label("'f' requires 'Float32' or 'Float64'"),
                );
                return;
            }
            Some(
                FormatKind::Bin
                | FormatKind::Oct
                | FormatKind::LowerHex
                | FormatKind::UpperHex
                | FormatKind::Fixed,
            )
            | None => {}
        }

        if spec.precision.is_some() && spec.kind.is_none() {
            // A bare `.N` is fixed-point on a float. On a string it would have to mean truncation,
            // which WP-FMT-001 deliberately does not define (§7): cutting Unicode text needs a
            // scalar/grapheme/byte ruling nobody has made.
            if !matches!(&stripped, Ty::Primitive(p) if is_float_primitive(*p)) {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "type '{}' cannot be formatted with a precision",
                            self.ty_to_string(&ty)
                        ),
                        spec_span,
                    )
                    .with_code("E0306")
                    .with_label("precision applies to 'Float32' and 'Float64'")
                    .with_note(
                        "string truncation is not a format specification in Core v1; slice the \
                         value explicitly if you need a shorter one"
                            .to_string(),
                    ),
                );
                return;
            }
        }

        if (spec.sign.is_some() || spec.alternate || spec.zero_pad)
            && spec.kind.is_none()
            && spec.precision.is_none()
            && !matches!(&stripped, Ty::Primitive(p) if is_numeric(*p))
        {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "type '{}' does not accept a numeric format specification",
                        self.ty_to_string(&ty)
                    ),
                    spec_span,
                )
                .with_code("E0306")
                .with_label("sign, '#' and zero-padding require a numeric type"),
            );
            return;
        }

        // Everything reaching here renders through `Display`. This is the SAME predicate
        // `print`/`println` use, so it routes through bound identity (CD-379) and a user trait
        // merely NAMED `Display` cannot satisfy it.
        //
        // The check is on the STRIPPED type. `println` takes its argument by value and so never
        // sees a reference, but a field routinely does — `fn render<T: Display>(v: &T)` formats a
        // `&T`, and `Display::fmt` borrows anyway (STD-FORMAT-001), so a reference to a
        // displayable type is displayable.
        // A generic parameter is displayable only if one of ITS OWN bounds supplies `fmt`.
        // `type_is_displayable` answers `true` for any `Ty::Param` — correct for `println`, whose
        // caller discharges the bound at the call site, and wrong here: an interpolation inside
        // `fn render<T>(v: &T)` has no such caller obligation, and must be refused where it is
        // written. The check goes through `bound_method_candidates`, so it is CD-379's identity
        // path — a user trait merely NAMED `Display` does not satisfy it.
        if let Ty::Param(param_name) = &stripped {
            let param_name = param_name.clone();
            // Queued before the guard: a parameter WITH the bound is a real late-bound render
            // position, and this branch returns early.
            self.record_display_plan(expr, stripped.clone());
            if self.bound_method_candidates(&param_name, "fmt").is_empty() {
                self.diags.push(
                    Diagnostic::error(
                        format!("'{param_name}' has no bound that provides 'Display'"),
                        expr_span,
                    )
                    .with_code("E0306")
                    .with_label(format!("add the bound '{param_name}: Display'")),
                );
            }
            return;
        }
        // **AS3 Boundary 4: interpolation is the SECOND `Display` entry point**, and it renders the
        // same way — `"{w}"` on a `W<A>` runs `W`'s own `fmt` and stops, exactly as `println(w)`
        // does. So it queues the same walk rather than getting a dispatch mechanism of its own,
        // which is what left `find_impl_fn(nominal, "fmt", ..)` serving two callers.
        self.record_display_plan(expr, stripped.clone());
        if !self.type_is_displayable(&stripped) {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "type '{}' does not implement 'Display' and cannot be interpolated",
                        self.ty_to_string(&ty)
                    ),
                    expr_span,
                )
                .with_code("E0306")
                .with_label("write an 'impl Display for ...' for this type"),
            );
        }
    }

    pub(super) fn check_expr(&mut self, expr_id: ExprId) -> Ty {
        let expr = self.hir.expr(expr_id);
        let ty = match &expr.kind {
            // WP-FMT-001: every field is checked, in source order, and the whole literal is a
            // `String`. Checking in order matters for diagnostics: the first bad field is reported
            // first, which is where a reader looks.
            hir::ExprKind::FormatString { segments } => {
                let fields: Vec<(ExprId, crate::ast::FormatSpec, Span)> = segments
                    .iter()
                    .filter_map(|segment| match segment {
                        hir::FormatSegment::Field {
                            expr,
                            spec,
                            expr_span,
                            ..
                        } => Some((*expr, *spec, *expr_span)),
                        hir::FormatSegment::Literal { .. } => None,
                    })
                    .collect();
                for (expr, spec, expr_span) in fields {
                    self.check_format_field(expr, &spec, expr_span);
                }
                Ty::Primitive(Primitive::String)
            }
            hir::ExprKind::Lit(lit) => match lit {
                // WP-C1.5 (DEV-015): no stage previously checked a literal's magnitude against
                // its suffix's (or, for unsuffixed literals, its default-inferred) representable
                // range -- `let x: UInt8 = 300u8;` compiled clean, and `let x = 99999999999;`
                // silently became a broken Int32 instead of the spec's "Int32 if it fits, else
                // Int64" (03-Type-System.md:28). Checked here, at typecheck time, since an
                // unsuffixed literal's fit-check depends on the type it's being inferred into
                // (Int32 vs Int64) -- the lexer sees only token shape, never a target type.
                Lit::Int { base, suffix } => {
                    let value = literal::parse_int_literal(self.text(expr.span), *base, *suffix);
                    if let Some(s) = suffix {
                        if let Some(value) = value {
                            if !literal::int_suffix_range_contains(*s, value) {
                                self.diags.push(
                                    Diagnostic::error(
                                        format!(
                                            "integer literal out of range for '{}'",
                                            self.ty_to_string(&Ty::Primitive(convert_int_suffix(
                                                *s
                                            )))
                                        ),
                                        expr.span,
                                    )
                                    .with_code("E0008"),
                                );
                            }
                        }
                        Ty::Primitive(convert_int_suffix(*s))
                    } else {
                        // WP-C4.7-6.3: an UNSUFFIXED literal takes a fresh integer-kinded
                        // inference variable instead of committing to `Int32` here. Expected
                        // types flow inward from annotations, parameters, fields and assignment
                        // destinations (03-Type-System), and only a literal still unconstrained
                        // after that is defaulted — step 5, applied in
                        // `default_unconstrained_int_literals`. Committing at this point was the
                        // whole defect: it made `takes_u64(0)` "expected 'UInt64', found 'Int32'".
                        match value {
                            Some(value) if i64::try_from(value).is_ok() => {
                                let var = self.new_type_var();
                                if let Ty::Infer(id) = var {
                                    self.int_literal_vars.insert(id, (value, expr.span));
                                }
                                var
                            }
                            Some(_) => {
                                // Beyond `Int64`'s range there is no representable type to adopt,
                                // so this is an error here rather than at binding time.
                                self.diags.push(
                                    Diagnostic::error(
                                        "integer literal out of range for 'Int64'",
                                        expr.span,
                                    )
                                    .with_code("E0008"),
                                );
                                Ty::Primitive(Primitive::Int64)
                            }
                            None => Ty::Primitive(Primitive::Int32),
                        }
                    }
                }
                Lit::Float { suffix, .. } => {
                    if let Some(s) = suffix {
                        Ty::Primitive(convert_float_suffix(*s))
                    } else {
                        Ty::Primitive(Primitive::Float64)
                    }
                }
                Lit::Str { .. } => Ty::Ref {
                    mutable: false,
                    inner: Box::new(Ty::Primitive(Primitive::Str)),
                },
                Lit::Char => Ty::Primitive(Primitive::Char),
                Lit::Bool(_) => Ty::Primitive(Primitive::Bool),
            },
            hir::ExprKind::Path { res, turbofish, .. } => match res {
                Res::Local(local_id) => {
                    self.local_types.get(local_id).cloned().unwrap_or(Ty::Error)
                }
                Res::Item(item_id) => {
                    if let Some(sig) = self.fn_sigs.get(item_id) {
                        let instantiated_sig = self.instantiate_sig(
                            *item_id,
                            sig.clone(),
                            turbofish.as_ref(),
                            Some(expr_id),
                            expr.span,
                        );
                        Ty::Fn {
                            params: instantiated_sig.params,
                            ret: Box::new(instantiated_sig.ret),
                        }
                    } else if let Some(const_ty) = self.const_types.get(item_id) {
                        // DEV-088 (WP-C4.7 close-out §7): USING a `const` declared in a different
                        // file is not yet supported and is rejected HERE, deterministically,
                        // before either engine runs. The oracle would evaluate the initializer's
                        // literal against the USE site's file (wrong text → "invalid literal" at
                        // runtime) while MIR does not lower a const in value position at all; a
                        // static rejection forecloses that inconsistency. Same-file `const` use is
                        // unaffected. Ownership-transferring cross-file constant use is deferred to
                        // the front-end/multi-file completion package (recorded in
                        // KNOWN-DEVIATIONS.md alongside DEV-083).
                        // AS1b-ii-d: identity, not name equality against an ambient file.
                        let cross_file = self
                            .hir
                            .item_sources
                            .get(item_id)
                            .is_some_and(|declaring| *declaring != expr.span.source);
                        if cross_file {
                            self.diags.push(
                                Diagnostic::error(
                                    "using a `const` declared in another file is not yet supported",
                                    expr.span,
                                )
                                .with_code("E0215")
                                .with_label(
                                    "move the constant into this file, or inline its value, until \
                                     cross-file constant use is implemented",
                                ),
                            );
                        }
                        const_ty.clone()
                    } else {
                        // Struct or Enum as expression (error in E02xx, but Ty::Error here)
                        Ty::Error
                    }
                }
                Res::Variant(enum_id, variant_idx) => {
                    let args = self.nominal_use_args(*enum_id, turbofish.as_ref(), expr.span);
                    let map = self.nominal_param_map(*enum_id, &args);
                    let variant = self
                        .enum_variants
                        .get(enum_id)
                        .and_then(|variants| variants.get(*variant_idx as usize))
                        .cloned();
                    match variant.map(|variant| variant.fields) {
                        Some(VariantFields::Unit) => Ty::Enum(*enum_id, args),
                        Some(VariantFields::Tuple(tys)) => Ty::Fn {
                            params: tys.iter().map(|ty| self.instantiate_ty(ty, &map)).collect(),
                            ret: Box::new(Ty::Enum(*enum_id, args)),
                        },
                        Some(VariantFields::Struct(_)) | None => Ty::Error,
                    }
                }
                Res::Primitive(p) => Ty::Primitive(*p),
                Res::AssociatedFn(item_id, name) => {
                    self.associated_fn_type(*item_id, *name, turbofish.as_ref(), expr.span, expr_id)
                }
                Res::ModelLoad(item_id) => {
                    self.validate_generic_arity(
                        0,
                        turbofish.as_ref().map_or(0, |args| args.args.len()),
                        expr.span,
                    );
                    let model_ty =
                        Ty::Extension(Box::new(ExtensionTy::Model(ModelTy { item_id: *item_id })));
                    let ret_ty = Ty::Core(
                        CoreType::Result,
                        vec![model_ty, Ty::Extension(Box::new(ExtensionTy::ModelError))],
                    );
                    Ty::Fn {
                        params: vec![Ty::Ref {
                            mutable: false,
                            inner: Box::new(Ty::Primitive(Primitive::Str)),
                        }],
                        ret: Box::new(ret_ty),
                    }
                }
                Res::SelfType => self.current_self_ty.clone().unwrap_or(Ty::Error),
                Res::SelfValue(local) => self.local_types.get(local).cloned().unwrap_or(Ty::Error),
                Res::Builtin(builtin) => {
                    if *builtin == Builtin::SizeOf || *builtin == Builtin::AlignOf {
                        self.validate_generic_arity(
                            1,
                            turbofish.as_ref().map_or(0, |args| args.args.len()),
                            expr.span,
                        );
                        if let Some(ref args) = turbofish {
                            for arg in &args.args {
                                if let hir::GenericArg::Type(type_id) = arg {
                                    // WP-C5.3e: the resolved type's CONTRACT LAYOUT is recorded
                                    // now. It was previously computed and discarded, which is
                                    // why the HIR oracle had no way to answer per type. A type
                                    // the contract does not describe records nothing, and every
                                    // engine then refuses the query rather than inventing a
                                    // number.
                                    // WP-C5.3e: the FULL conversion, not
                                    // `type_from_hir_without_diagnostics` -- that helper handles
                                    // only primitives, bare nominals and references, dropping
                                    // generic arguments and mapping tuples/arrays to `Ty::Error`.
                                    // It was adequate when the result was discarded; a layout
                                    // answer needs the real type.
                                    let ty = self.convert_hir_type(*type_id);
                                    let ty = self.ground(&ty);
                                    self.layout_queries.insert(expr_id, ty);
                                }
                            }
                        }
                    }
                    self.builtin_type(*builtin)
                }
                Res::TraitMember(_, _) => Ty::Error,
                Res::CoreTraitMember(_, _) => Ty::Error,
                Res::Err
                | Res::TypeParam
                | Res::CoreTrait(_)
                | Res::CoreType(_)
                | Res::SelfAssoc(_)
                | Res::ParamAssoc(..) => Ty::Error,
            },
            hir::ExprKind::Unary { op, operand } => {
                let op_ty = self.check_expr(*operand);
                match op {
                    UnOp::Neg => {
                        match self.resolve(&op_ty) {
                            Ty::Primitive(p) if is_numeric(p) => {}
                            Ty::Param(_) => self.require_operator_bound(&op_ty, "Num", expr.span),
                            Ty::Infer(_) | Ty::Error => {}
                            _ => self.diags.push(
                                Diagnostic::error("negation targets non-numeric type", expr.span)
                                    .with_code("E0001"),
                            ),
                        }
                        op_ty
                    }
                    UnOp::Not => {
                        let _ = self.unify(Ty::Primitive(Primitive::Bool), op_ty, expr.span);
                        Ty::Primitive(Primitive::Bool)
                    }
                    UnOp::BitNot => {
                        match self.resolve(&op_ty) {
                            Ty::Primitive(p) if is_integer(p) => {}
                            Ty::Param(_) => self.require_operator_bound(&op_ty, "Num", expr.span),
                            Ty::Infer(_) | Ty::Error => {}
                            _ => self.diags.push(
                                Diagnostic::error(
                                    "bitwise not targets non-integer type",
                                    expr.span,
                                )
                                .with_code("E0001"),
                            ),
                        }
                        op_ty
                    }
                    UnOp::Ref { mutable } => Ty::Ref {
                        mutable: *mutable,
                        inner: Box::new(op_ty),
                    },
                    UnOp::Deref => match self.resolve(&op_ty) {
                        Ty::Ref { inner, .. } => *inner,
                        Ty::Error => Ty::Error,
                        other => {
                            self.diags.push(
                                Diagnostic::error(
                                    format!(
                                        "cannot dereference non-reference type '{}'",
                                        self.ty_to_string(&other)
                                    ),
                                    expr.span,
                                )
                                .with_code("E0001"),
                            );
                            Ty::Error
                        }
                    },
                }
            }
            hir::ExprKind::Binary { op, lhs, rhs } => {
                let lhs_ty = self.check_expr(*lhs);
                let rhs_ty = self.check_expr(*rhs);

                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                        let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        self.require_operator_bound(&lhs_ty, "Num", expr.span);
                        lhs_ty
                    }
                    BinOp::Pow => {
                        let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        match self.resolve(&lhs_ty) {
                            Ty::Primitive(p) if is_integer(p) => {}
                            Ty::Infer(_) | Ty::Error => {}
                            _ => self.diags.push(
                                Diagnostic::error(
                                    "`**` is defined only for integer primitive types",
                                    expr.span,
                                )
                                .with_code("E0001")
                                .with_note(
                                    "use `std::math::pow` for floating-point exponentiation",
                                ),
                            ),
                        }
                        lhs_ty
                    }
                    BinOp::Eq | BinOp::Ne => {
                        if !self.string_types_comparable(&lhs_ty, &rhs_ty) {
                            let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        }
                        self.require_operator_bound(&lhs_ty, "Eq", expr.span);
                        self.publish_operator_use(expr_id, &lhs_ty, "Eq", "eq", hir::CoreTrait::Eq);
                        Ty::Primitive(Primitive::Bool)
                    }
                    BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        if !self.string_types_comparable(&lhs_ty, &rhs_ty) {
                            let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        }
                        self.require_operator_bound(&lhs_ty, "Ord", expr.span);
                        self.publish_operator_use(
                            expr_id,
                            &lhs_ty,
                            "Ord",
                            "cmp",
                            hir::CoreTrait::Ord,
                        );
                        Ty::Primitive(Primitive::Bool)
                    }
                    BinOp::And | BinOp::Or => {
                        let _ = self.unify(Ty::Primitive(Primitive::Bool), lhs_ty, expr.span);
                        let _ = self.unify(Ty::Primitive(Primitive::Bool), rhs_ty, expr.span);
                        Ty::Primitive(Primitive::Bool)
                    }
                    BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                        let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        self.require_operator_bound(&lhs_ty, "Num", expr.span);
                        lhs_ty
                    }
                }
            }
            hir::ExprKind::Assign { op, lhs, rhs } => {
                let lhs_ty = self.check_expr(*lhs);
                let rhs_ty = self.check_expr(*rhs);

                match op {
                    AssignOp::Assign => {
                        let _ = self.unify(lhs_ty, rhs_ty, expr.span);
                    }
                    AssignOp::PowAssign => {
                        let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        match self.resolve(&lhs_ty) {
                            Ty::Primitive(p) if is_integer(p) => {}
                            Ty::Infer(_) | Ty::Error => {}
                            _ => self.diags.push(
                                Diagnostic::error(
                                    "`**=` is defined only for integer primitive types",
                                    expr.span,
                                )
                                .with_code("E0001")
                                .with_note(
                                    "use `std::math::pow` for floating-point exponentiation",
                                ),
                            ),
                        }
                    }
                    _ => {
                        let _ = self.unify(lhs_ty, rhs_ty, expr.span);
                    }
                }
                Ty::Primitive(Primitive::Unit)
            }
            hir::ExprKind::Range {
                lo,
                hi,
                inclusive: _,
            } => {
                let lo_ty = self.check_expr(*lo);
                let hi_ty = self.check_expr(*hi);
                let _ = self.unify(lo_ty.clone(), hi_ty, expr.span);
                Ty::Range(Box::new(lo_ty))
            }
            hir::ExprKind::Cast {
                expr: cast_expr,
                ty,
            } => {
                let source = self.check_expr(*cast_expr);
                let saved = self.allow_half_type;
                self.allow_half_type = true;
                let target = self.convert_hir_type(*ty);
                self.allow_half_type = saved;
                // WP-C4.7-6.3: `5 as UInt8` — the cast's SOURCE must be concrete to classify as
                // numeric. A literal operand has no other constraint (a cast does not propagate
                // its target inward: per 03, casts are explicit conversions, not expectations),
                // so settle it to its default width here.
                let source_resolved = self.default_int_literal_now(&source);
                let target_resolved = self.resolve(&target);
                if !matches!(source_resolved, Ty::Error)
                    && !matches!(target_resolved, Ty::Error)
                    && (!matches!(&source_resolved, Ty::Primitive(p) if is_cast_numeric(*p))
                        || !matches!(&target_resolved, Ty::Primitive(p) if is_cast_numeric(*p)))
                {
                    self.diags.push(
                        Diagnostic::error(
                            "casts are permitted only between numeric types",
                            expr.span,
                        )
                        .with_code("E0001"),
                    );
                }
                target
            }
            hir::ExprKind::Call { callee, args } => {
                if let hir::ExprKind::Field {
                    base,
                    name,
                    turbofish,
                } = &self.hir.expr(*callee).kind
                {
                    self.resolve_method(*base, *name, turbofish.as_ref(), args, expr.span, expr_id)
                } else if let hir::ExprKind::Path {
                    res: Res::TraitMember(trait_id, member),
                    ..
                } = &self.hir.expr(*callee).kind
                {
                    self.check_qualified_trait_call(expr_id, *trait_id, *member, args, expr.span)
                } else if let hir::ExprKind::Path {
                    res: Res::CoreTraitMember(core_trait, method_span),
                    ..
                } = &self.hir.expr(*callee).kind
                {
                    self.check_qualified_core_trait_call(
                        expr_id,
                        *core_trait,
                        *method_span,
                        args,
                        expr.span,
                    )
                } else if let hir::ExprKind::Path {
                    res: Res::Builtin(builtin),
                    turbofish,
                    ..
                } = &self.hir.expr(*callee).kind
                {
                    if crate::resolve::is_tensor_builtin(*builtin) {
                        self.check_tensor_builtin_call(
                            *builtin,
                            turbofish.as_ref(),
                            args,
                            expr.span,
                        )
                    } else {
                        let callee_ty = self.check_expr(*callee);
                        let arg_tys: Vec<Ty> = args.iter().map(|&a| self.check_expr(a)).collect();
                        // WP-C4.7-9 audit: `print`/`println` type their argument as a fresh
                        // inference variable, so they accepted ANY type — including a user struct
                        // with no `Display` impl. 06-Standard-Library says `Display` is not a
                        // syntax hook and user types must implement it, so that was an
                        // over-acceptance: the checker admitted a program the oracle then
                        // rendered in an unspecified debug-ish form and MIR refused outright.
                        // Deferred to Pass 3 so inference has settled first.
                        if matches!(
                            builtin,
                            Builtin::Print | Builtin::Println | Builtin::Eprint | Builtin::Eprintln
                        ) {
                            if let (Some(ty), Some(arg)) = (arg_tys.first(), args.first()) {
                                self.display_checks
                                    .push((ty.clone(), self.hir.expr(*arg).span));
                                self.record_display_plan(*arg, ty.clone());
                            }
                        }
                        match self.resolve(&callee_ty) {
                            Ty::Fn { params, ret } => {
                                if params.len() != arg_tys.len() {
                                    self.diags.push(
                                        Diagnostic::error(
                                            format!(
                                                "wrong number of arguments: expected {}, found {}",
                                                params.len(),
                                                arg_tys.len()
                                            ),
                                            expr.span,
                                        )
                                        .with_code("E0005"),
                                    );
                                }
                                for ((param, arg), arg_expr) in
                                    params.into_iter().zip(arg_tys).zip(args)
                                {
                                    let _ = self.unify(param, arg, self.hir.expr(*arg_expr).span);
                                }
                                // WP-C6.2c: arguments have fixed the base type parameters, so a
                                // deferred projection in the return can be resolved before use.
                                self.discharge_ready_projections();
                                *ret
                            }
                            Ty::Error => Ty::Error,
                            other => {
                                self.diags.push(
                                    Diagnostic::error(
                                        format!(
                                            "called expression has non-function type '{}'",
                                            self.ty_to_string(&other)
                                        ),
                                        expr.span,
                                    )
                                    .with_code("E0001"),
                                );
                                Ty::Error
                            }
                        }
                    }
                } else {
                    let callee_ty = self.check_expr(*callee);
                    let arg_tys: Vec<Ty> = args.iter().map(|&a| self.check_expr(a)).collect();
                    match self.resolve(&callee_ty) {
                        Ty::Fn { params, ret } => {
                            let param_snapshot = params.clone();
                            if params.len() != arg_tys.len() {
                                self.diags.push(
                                    Diagnostic::error(
                                        format!(
                                            "wrong number of arguments: expected {}, found {}",
                                            params.len(),
                                            arg_tys.len()
                                        ),
                                        expr.span,
                                    )
                                    .with_code("E0005"),
                                );
                            }
                            for ((param, arg), arg_expr) in
                                params.into_iter().zip(arg_tys).zip(args)
                            {
                                let _ = self.unify(param, arg, self.hir.expr(*arg_expr).span);
                            }
                            // WP-C6.2c: resolve any deferred return projection now the arguments
                            // have fixed the base type parameters.
                            self.discharge_ready_projections();
                            // AS3 Boundary 1: the DYNAMIC half of the model, published at the same
                            // time as the static half so it is exercised from the start.
                            //
                            // The checker knows this is a call and knows its signature. It does NOT
                            // know the body: DEV-178 established that the value carries the item and
                            // the bindings it was created with, because `Ty::Fn` cannot say which
                            // instantiation produced it. `FunctionValue` states that rather than
                            // pretending a `BlockId` exists here.
                            // **DEV-193: not every call reaching here is a function-VALUE call.**
                            //
                            // `free(1)`, where `free` names a known `fn` item, falls into this
                            // branch too — and published `FunctionValue`, the selection that means
                            // "the body is not knowable here". It is knowable: the callee path
                            // published `Direct`/`Static(body)` a moment earlier. So `free(1)` and
                            // `g(2)` produced IDENTICAL records at their call expressions, and a
                            // consumer reading the call site could not tell a direct call from a
                            // call through a value — the exact conflation three binding times exist
                            // to prevent.
                            //
                            // The record for a direct call is the path's; publishing a second,
                            // weaker one here would be a duplicate that contradicts it.
                            let callee_is_known_fn = match &self.hir.expr(*callee).kind {
                                hir::ExprKind::Path {
                                    res: Res::Item(item),
                                    ..
                                } => matches!(self.hir.item(*item).kind, hir::ItemKind::Fn(_)),
                                _ => false,
                            };
                            let use_ = CallableUse {
                                selection: CalleeSelection::FunctionValue,
                                environment: GenericEnvironment::FromFunctionValue,
                                receiver_adjustment: ReceiverAdjustment::None,
                                receiver_binding: ReceiverBinding::None,
                                signature: CallableSigTy {
                                    receiver: None,
                                    params: param_snapshot,
                                    ret: (*ret).clone(),
                                },
                                provenance: DispatchProvenance::FunctionValue,
                            };
                            if !callee_is_known_fn {
                                self.publish_callable_use(expr_id, use_);
                            }
                            *ret
                        }
                        Ty::Error => Ty::Error,
                        other => {
                            self.diags.push(
                                Diagnostic::error(
                                    format!(
                                        "called expression has non-function type '{}'",
                                        self.ty_to_string(&other)
                                    ),
                                    expr.span,
                                )
                                .with_code("E0001"),
                            );
                            Ty::Error
                        }
                    }
                }
            }
            hir::ExprKind::Field { base, name, .. } => {
                let mut base_ty = self.check_expr(*base);
                while let Ty::Ref { inner, .. } = self.resolve(&base_ty) {
                    base_ty = *inner;
                }

                let name_str = self.text(*name);
                match self.resolve(&base_ty) {
                    Ty::Struct(struct_id, args) => {
                        let field_ty = self
                            .struct_fields
                            .get(&struct_id)
                            .and_then(|fields| fields.get(name_str))
                            .cloned();
                        if let Some(field_ty) = field_ty {
                            // WP-C6.2b-F1: a private field is inaccessible outside its module.
                            let name_owned = name_str.to_string();
                            let is_pub = self.struct_field_is_pub(struct_id, &name_owned);
                            self.check_member_visible(
                                is_pub,
                                struct_id,
                                "field",
                                &name_owned,
                                *name,
                            );
                            let map = self.nominal_param_map(struct_id, &args);
                            self.instantiate_ty(&field_ty, &map)
                        } else if self.struct_fields.contains_key(&struct_id) {
                            self.diags.push(
                                Diagnostic::error(
                                    format!("struct field '{}' not found", name_str),
                                    *name,
                                )
                                .with_code("E0001"),
                            );
                            Ty::Error
                        } else {
                            Ty::Error
                        }
                    }
                    Ty::Error => Ty::Error,
                    other => {
                        self.diags.push(
                            Diagnostic::error(
                                format!(
                                    "cannot access field '{}' on non-struct type '{}'",
                                    name_str,
                                    self.ty_to_string(&other)
                                ),
                                expr.span,
                            )
                            .with_code("E0001"),
                        );
                        Ty::Error
                    }
                }
            }
            hir::ExprKind::TupleField { base, index } => {
                let mut base_ty = self.check_expr(*base);
                while let Ty::Ref { inner, .. } = self.resolve(&base_ty) {
                    base_ty = *inner;
                }

                match self.resolve(&base_ty) {
                    Ty::Tuple(elems) => {
                        let idx_str = self.text(*index);
                        let idx = idx_str.parse::<usize>().unwrap_or(0);
                        if idx < elems.len() {
                            elems[idx].clone()
                        } else {
                            self.diags.push(
                                Diagnostic::error(
                                    format!(
                                        "tuple index out of bounds: length is {}, but index is {}",
                                        elems.len(),
                                        idx
                                    ),
                                    *index,
                                )
                                .with_code("E0007"),
                            );
                            Ty::Error
                        }
                    }
                    Ty::Error => Ty::Error,
                    other => {
                        self.diags.push(
                            Diagnostic::error(
                                format!(
                                    "cannot access tuple field on non-tuple type '{}'",
                                    self.ty_to_string(&other)
                                ),
                                expr.span,
                            )
                            .with_code("E0001"),
                        );
                        Ty::Error
                    }
                }
            }
            hir::ExprKind::Index { base, index } => {
                let mut base_ty = self.check_expr(*base);
                while let Ty::Ref { inner, .. } = self.resolve(&base_ty) {
                    base_ty = *inner;
                }

                let index_ty = self.check_expr(*index);
                let resolved_index_ty = self.resolve(&index_ty);
                let is_range = matches!(resolved_index_ty, Ty::Range(_));
                let is_integer = matches!(
                    resolved_index_ty,
                    Ty::Primitive(Primitive::Int8)
                        | Ty::Primitive(Primitive::Int16)
                        | Ty::Primitive(Primitive::Int32)
                        | Ty::Primitive(Primitive::Int64)
                        | Ty::Primitive(Primitive::UInt8)
                        | Ty::Primitive(Primitive::UInt16)
                        | Ty::Primitive(Primitive::UInt32)
                        | Ty::Primitive(Primitive::UInt64)
                        | Ty::Error
                );
                if !is_integer && !is_range {
                    if let Ty::Infer(_) = resolved_index_ty {
                        let _ = self.unify(
                            Ty::Primitive(Primitive::Int32),
                            index_ty.clone(),
                            self.hir.expr(*index).span,
                        );
                    } else {
                        self.diags.push(
                            Diagnostic::error(
                                "array index must be an integer type",
                                self.hir.expr(*index).span,
                            )
                            .with_code("E0001"),
                        );
                    }
                }

                // Static bounds checking if index is a literal
                let idx_val = if let hir::ExprKind::Lit(Lit::Int { base: _, suffix: _ }) =
                    &self.hir.expr(*index).kind
                {
                    let idx_str = self.text(self.hir.expr(*index).span);
                    idx_str.parse::<u64>().ok()
                } else {
                    None
                };

                match self.resolve(&base_ty) {
                    Ty::Array(elem, len) => {
                        if is_range {
                            Ty::Slice(elem)
                        } else {
                            if let Some(idx) = idx_val {
                                if idx >= len {
                                    self.diags.push(
                                        Diagnostic::error(
                                            format!("index out of bounds: the length is {} but the index is {}", len, idx),
                                            expr.span,
                                        )
                                        .with_code("E0007")
                                    );
                                }
                            }
                            *elem
                        }
                    }
                    Ty::Slice(elem) => {
                        if is_range {
                            Ty::Slice(elem)
                        } else {
                            *elem
                        }
                    }
                    Ty::Core(CoreType::Vec, mut args) => {
                        let elem = args.pop().unwrap_or(Ty::Error);
                        if is_range {
                            Ty::Slice(Box::new(elem))
                        } else {
                            elem
                        }
                    }
                    Ty::Error => Ty::Error,
                    other => {
                        self.diags.push(
                            Diagnostic::error(
                                format!(
                                    "indexing requires array or slice, found '{}'",
                                    self.ty_to_string(&other)
                                ),
                                expr.span,
                            )
                            .with_code("E0001"),
                        );
                        Ty::Error
                    }
                }
            }
            hir::ExprKind::Try(try_expr) => {
                let expr_ty = self.check_expr(*try_expr);

                // 1. Check enclosing function return type
                let mut ret_ok = false;
                if let Some(fn_ret) = &self.current_fn_ret {
                    let fn_ret = self.resolve(fn_ret);
                    match fn_ret {
                        // WP-C1.5: `Option`/`Result` are always `Ty::Core(CoreType::Option|
                        // Result, _)` (see `hir::CoreType`), never `Ty::Enum` -- a `Ty::Enum`
                        // arm here previously did a substring search over the enum's entire
                        // declaration source text for "Result"/"Option", which let any
                        // unrelated user enum with a matching substring anywhere in its
                        // declaration (e.g. a variant literally named `ResultVariant`) satisfy
                        // this check. 03-Type-System.md:590 defines `?` exclusively for
                        // `Result<T, E>`/`Option<T>`; there is no user-extensible Try trait in
                        // Core v1, so no `Ty::Enum` should ever satisfy this.
                        Ty::Core(CoreType::Result | CoreType::Option, _) => ret_ok = true,
                        Ty::Error => {
                            ret_ok = true; // suppress
                        }
                        _ => {}
                    }
                } else {
                    // Snippet mode: enclosing is snippet root
                    ret_ok = true;
                }

                if !ret_ok {
                    self.diags.push(
                        Diagnostic::error("try operator '?' cannot be used in a function that does not return Result or Option", expr.span)
                            .with_code("E0006")
                    );
                }

                // DEV-134: relate the OPERAND to the enclosing return type. Before this, the two
                // were checked only for being Result-or-Option INDEPENDENTLY, and never against
                // each other -- so `Result<_, Low>` propagated out of a function returning
                // `Result<_, High>` (no `From` impl required or applied), and `Option<_>`
                // propagated out of a function returning `Result<_, _>`. Both produced a value
                // whose variant tag belonged to a different type: type confusion, not a
                // diagnostic gap. Deferred to `check_try_compatibility` so inference has settled.
                if let Some(fn_ret) = self.current_fn_ret.clone() {
                    self.try_checks.push((expr_ty.clone(), fn_ret, expr.span));
                }

                // 2. Check try expression type
                match self.resolve(&expr_ty) {
                    // WP-C1.5: same fix as above -- Option/Result never resolve to `Ty::Enum`,
                    // so this used to be exploitable via any user enum with a "Result"/"Option"
                    // substring anywhere in its declaration text. No `Ty::Enum` arm here at all
                    // now; it falls through to the `_` rejection below, correctly.
                    Ty::Core(CoreType::Result | CoreType::Option, args) => {
                        args.first().cloned().unwrap_or(Ty::Error)
                    }
                    Ty::Error => Ty::Error,
                    _ => {
                        if expr_ty != Ty::Error {
                            self.diags.push(
                                Diagnostic::error(
                                    "try operator '?' requires Result or Option",
                                    expr.span,
                                )
                                .with_code("E0006"),
                            );
                        }
                        Ty::Error
                    }
                }
            }
            hir::ExprKind::Tuple(elems) => {
                let tys: Vec<Ty> = elems.iter().map(|&e| self.check_expr(e)).collect();
                unit_or_tuple(tys)
            }
            hir::ExprKind::Array(elems) => {
                let elem_var = self.new_type_var();
                for &e in elems {
                    let ety = self.check_expr(e);
                    let _ = self.unify(elem_var.clone(), ety, expr.span);
                }
                Ty::Array(Box::new(elem_var), elems.len() as u64)
            }
            hir::ExprKind::Repeat { value, count } => {
                let val_ty = self.check_expr(*value);
                let count_ty = self.check_expr(*count);
                let count_ty = self.resolve(&count_ty);
                // WP-C4.7-6.3: an unsuffixed literal count is an integer-kinded inference var
                // here, not yet a concrete `Int32`. It is integer BY CONSTRUCTION (only integer
                // literals get these vars), so accept it and let defaulting settle the width.
                let count_is_int_literal =
                    matches!(&count_ty, Ty::Infer(id) if self.int_literal_vars.contains_key(id));
                if !matches!(&count_ty, Ty::Primitive(p) if is_integer(*p))
                    && !count_is_int_literal
                    && !matches!(count_ty, Ty::Error)
                {
                    self.diags.push(
                        Diagnostic::error("array repeat count must be an integer", expr.span)
                            .with_code("E0001"),
                    );
                }

                // WP-C1.5: `count` (02-Syntax-Grammar.md:330: "must be a compile-time constant
                // expression") was previously computed by parsing the *raw source text* of the
                // count expression as a bare unsuffixed decimal (`text.parse::<u64>()`) --
                // anything else (a suffixed literal like `5u32`, an underscore-grouped literal
                // like `1_0`, or a `const` item reference) silently failed to parse and fell
                // back to length 0, which then falsely rejected every subsequent valid index
                // into the array with E0007. `const_eval_u64` handles the confirmed-common
                // shapes (a literal, or a reference to a `const` item); anything else is
                // reported directly rather than silently defaulting to a wrong length.
                let len = match self.const_eval_u64(*count) {
                    Some(len) => len,
                    None => {
                        if !matches!(count_ty, Ty::Error) {
                            self.diags.push(
                                Diagnostic::error(
                                    "array repeat count must be a compile-time constant \
                                     expression",
                                    self.hir.expr(*count).span,
                                )
                                .with_code("E0009"),
                            );
                        }
                        0
                    }
                };
                Ty::Array(Box::new(val_ty), len)
            }
            hir::ExprKind::StructLit { res, fields, .. } => match res {
                Res::Item(struct_id) => {
                    let args = self.nominal_use_args(*struct_id, None, expr.span);
                    let map = self.nominal_param_map(*struct_id, &args);
                    let expected = self
                        .struct_fields
                        .get(struct_id)
                        .cloned()
                        .unwrap_or_default();
                    self.publish_aggregate_field_types(expr_id, &expected, &map);
                    self.check_field_initializers(
                        Some(*struct_id),
                        &expected,
                        &map,
                        fields,
                        expr.span,
                    );
                    Ty::Struct(*struct_id, args)
                }
                Res::Variant(enum_id, variant) => {
                    let args = self.nominal_use_args(*enum_id, None, expr.span);
                    let map = self.nominal_param_map(*enum_id, &args);
                    let expected = self
                        .enum_variants
                        .get(enum_id)
                        .and_then(|variants| variants.get(*variant as usize))
                        .and_then(|variant| match &variant.fields {
                            VariantFields::Struct(fields) => Some(fields.clone()),
                            _ => None,
                        });
                    if let Some(expected) = expected {
                        self.publish_aggregate_field_types(expr_id, &expected, &map);
                        self.check_field_initializers(None, &expected, &map, fields, expr.span);
                        Ty::Enum(*enum_id, args)
                    } else {
                        self.diags.push(
                            Diagnostic::error(
                                "struct literal syntax requires a struct-like variant",
                                expr.span,
                            )
                            .with_code("E0001"),
                        );
                        Ty::Error
                    }
                }
                _ => Ty::Error,
            },
            hir::ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                let cond_ty = self.check_expr(*cond);
                let _ = self.unify(
                    Ty::Primitive(Primitive::Bool),
                    cond_ty,
                    self.hir.expr(*cond).span,
                );

                // For snippet blocks where variables may leak/define:
                let mut dummy_state = HashSet::new();
                let then_ty = self.check_block(*then_block, &mut dummy_state);

                if let Some(else_expr) = else_ {
                    let else_ty = self.check_expr(*else_expr);
                    let _ = self.unify(then_ty.clone(), else_ty, expr.span);
                    then_ty
                } else {
                    let _ = self.unify(Ty::Primitive(Primitive::Unit), then_ty.clone(), expr.span);
                    Ty::Primitive(Primitive::Unit)
                }
            }
            hir::ExprKind::Match { scrutinee, arms } => {
                let scr_expr_ty = self.check_expr(*scrutinee);
                let ret_ty = self.new_type_var();

                // **A REFERENCE-TYPED SCRUTINEE IS REJECTED, PER PAT-BIND-001.**
                //
                // The spec states it directly: "a struct/variant path must name the scrutinee's
                // normalized nominal type, and `&T` is not a nominal type, so `match r { E::V(x) =>
                // .. }` for `r: &E` is a type error. This is why the rule is stated over the place
                // read, not over the scrutinee's type." The program writes `match *r`, which IS a
                // read through a reference and binds by PAT-BIND-001.
                //
                // It was not rejected. `Ty::Ref` simply fell through every classifier, and the
                // result was the worst available combination:
                //
                //   - the exhaustiveness check saw a domain it did not know and demanded a wildcard,
                //     reporting E0303 on a match that already covered every variant; and then
                //   - the `_` arm added to satisfy that ABSORBED EVERY CASE at run time, because
                //     the constructor arms were typed against a reference and matched nothing.
                //
                // So the diagnostic pointed at the wrong problem, and the obvious response to it
                // produced a function that silently returned the wildcard's answer for every input.
                // `stark-percent`'s `is_incomplete_escape` reported "not an incomplete escape" for
                // an incomplete escape it had just been handed, and no test could see it, because
                // the helper was reporting on itself.
                //
                // Rejecting names the fix rather than the symptom. A `match *r` in the same place
                // compiles and behaves.
                let scr_resolved = self.resolve(&scr_expr_ty);
                if matches!(scr_resolved, Ty::Ref { .. })
                    && arms.iter().any(|arm| self.pat_is_constructor(arm.pat))
                {
                    self.diags.push(
                        Diagnostic::error(
                            format!(
                                "cannot match a reference-typed scrutinee '{}' against constructor patterns",
                                self.ty_to_string(&scr_resolved)
                            ),
                            self.hir.expr(*scrutinee).span,
                        )
                        .with_code("E0001")
                        .with_help(
                            "dereference the scrutinee first: `match *r { .. }`. A binding to a \
                             non-Copy component then has reference type (PAT-BIND-001) and nothing \
                             is moved out of the referent",
                        ),
                    );
                }

                let scr_ty = scr_expr_ty.clone();
                let bind_non_copy_by_ref = if self.scrutinee_reads_through_ref(*scrutinee) {
                    BindMode::ThroughRef
                } else {
                    BindMode::ByValue
                };

                let mut matched_variants = HashSet::new();
                let mut matched_bools = HashSet::new();
                let mut has_wildcard = false;
                // WP-C1.5: `Option`/`Result` resolve to `Ty::Core(CoreType::Option|Result, _)`,
                // never `Ty::Enum` (see `hir::CoreType`), and their `Some`/`None`/`Ok`/`Err`
                // patterns resolve via `Res::Builtin`, never `Res::Variant` -- so the existing
                // `matched_variants`/`Ty::Enum` machinery below never covered them at all.
                // `match opt { Some(v) => .. }` (missing `None`) compiled clean before this fix.
                let (mut matched_some, mut matched_none) = (false, false);
                let (mut matched_ok, mut matched_err) = (false, false);
                // DEV-071 (WP-C4.7-7): the prelude `Ordering` is `Ty::Core(CoreType::Ordering)`
                // with `Res::Builtin` variants — exactly like `Option`/`Result`, and for exactly
                // the same reason it was invisible to the `Ty::Enum`/`matched_variants`
                // machinery. Unlike those two, though, `Ordering` fell through to the
                // "unknown domain, require a wildcard" default, so an all-three-variant match
                // was reported NON-exhaustive (E0303) and every `Ordering` match needed a
                // pointless `_` arm.
                let (mut matched_less, mut matched_equal, mut matched_greater) =
                    (false, false, false);

                let mut preceding_patterns = Vec::new();

                for arm in arms {
                    let pat_ty =
                        self.check_pat_with_mode(arm.pat, scr_ty.clone(), bind_non_copy_by_ref);
                    let _ = self.unify(scr_ty.clone(), pat_ty, arm.pat.span(self.hir));

                    let pat = self.hir.pat(arm.pat);

                    let mut is_unreachable = false;
                    for prev_pat in &preceding_patterns {
                        #[allow(clippy::explicit_auto_deref)]
                        if self.pat_subsumes(*prev_pat, pat) {
                            is_unreachable = true;
                            break;
                        }
                    }
                    if is_unreachable {
                        self.diags.push(
                            Diagnostic::warning("unreachable match arm", arm.pat.span(self.hir))
                                .with_code("W0006")
                                .with_label(
                                    "this pattern is redundant and covered by a preceding arm",
                                ),
                        );
                    } else {
                        preceding_patterns.push(pat);
                    }

                    if self.is_irrefutable(pat) {
                        has_wildcard = true;
                    }
                    match &pat.kind {
                        hir::PatKind::Wild | hir::PatKind::Binding { .. } => {}
                        hir::PatKind::Path { res, .. }
                        | hir::PatKind::TupleVariant { res, .. }
                        | hir::PatKind::Struct { res, .. } => match res {
                            Res::Variant(_, variant_idx) => {
                                matched_variants.insert(*variant_idx);
                            }
                            Res::Builtin(Builtin::Some) => matched_some = true,
                            Res::Builtin(Builtin::None) => matched_none = true,
                            Res::Builtin(Builtin::Ok) => matched_ok = true,
                            Res::Builtin(Builtin::Err) => matched_err = true,
                            Res::Builtin(Builtin::OrderingLess) => matched_less = true,
                            Res::Builtin(Builtin::OrderingEqual) => matched_equal = true,
                            Res::Builtin(Builtin::OrderingGreater) => matched_greater = true,
                            _ => {}
                        },
                        hir::PatKind::Lit(Lit::Bool(value)) => {
                            matched_bools.insert(*value);
                        }
                        _ => {}
                    }

                    let body_ty = self.check_expr(arm.body);
                    let _ = self.unify(ret_ty.clone(), body_ty, self.hir.expr(arm.body).span);
                }

                if !has_wildcard {
                    let non_exhaustive = match self.resolve(&scr_ty) {
                        Ty::Enum(enum_id, _) => self
                            .enum_variants
                            .get(&enum_id)
                            .is_some_and(|variants| matched_variants.len() < variants.len()),
                        Ty::Primitive(Primitive::Bool) => matched_bools.len() < 2,
                        Ty::Core(CoreType::Option, _) => !(matched_some && matched_none),
                        Ty::Core(CoreType::Result, _) => !(matched_ok && matched_err),
                        // DEV-071: `Ordering` has exactly three fieldless variants, so matching
                        // all three IS exhaustive and needs no wildcard.
                        Ty::Core(CoreType::Ordering, _) => {
                            !(matched_less && matched_equal && matched_greater)
                        }
                        // WP-C1.5: every other scrutinee type previously fell through here
                        // silently, regardless of arm coverage -- `match x: Int32 { 1 => ..,
                        // 2 => .. }` (missing every other Int32 value) compiled clean and only
                        // trapped at runtime ("non-exhaustive match reached", interp.rs) if an
                        // unmatched value actually occurred. 04-Semantic-Analysis.md is explicit:
                        // "If a match is not exhaustive, it is a compile-time error." A real
                        // usefulness/coverage algorithm (tracking which literal values or ranges
                        // are covered) is out of this WP's scope; instead, any scrutinee type
                        // that isn't one of the small, exactly-enumerable domains above now
                        // requires an explicit wildcard/binding arm to be considered exhaustive
                        // -- sound (never accepts a genuinely non-exhaustive match), and matches
                        // this codebase's existing "reject some safe programs is intentional"
                        // philosophy (03-Type-System.md's own framing for the analogous borrow-
                        // checking tradeoff). `Ty::Struct` is exempted: a struct type has exactly
                        // one shape, so any single struct-pattern arm is exhaustive over it by
                        // construction (sub-pattern-level literal restrictions, e.g. `Point{x: 0,
                        // ..}`, are not yet analyzed here -- same pre-existing imprecision as
                        // before this fix, backstopped by the same runtime trap).
                        Ty::Error | Ty::Struct(..) => false,
                        _ => true,
                    };
                    if non_exhaustive {
                        self.diags.push(
                            Diagnostic::error("non-exhaustive pattern match", expr.span)
                                .with_code("E0303"),
                        );
                    }
                }

                ret_ty
            }
            hir::ExprKind::Loop { body } => {
                let break_ty = self.new_type_var();
                self.loop_contexts.push(LoopContext {
                    allows_value: true,
                    break_ty: break_ty.clone(),
                    has_break: false,
                });
                self.loop_nesting += 1;
                let mut dummy_state = HashSet::new();
                let _ = self.check_block(*body, &mut dummy_state);
                self.loop_nesting -= 1;
                let context = self.loop_contexts.pop().expect("loop context exists");
                if context.has_break {
                    self.resolve(&break_ty)
                } else {
                    Ty::Never
                }
            }
            hir::ExprKind::While { cond, body } => {
                let cond_ty = self.check_expr(*cond);
                let _ = self.unify(
                    Ty::Primitive(Primitive::Bool),
                    cond_ty,
                    self.hir.expr(*cond).span,
                );
                self.loop_contexts.push(LoopContext {
                    allows_value: false,
                    break_ty: Ty::Primitive(Primitive::Unit),
                    has_break: false,
                });
                self.loop_nesting += 1;
                let mut dummy_state = HashSet::new();
                let _ = self.check_block(*body, &mut dummy_state);
                self.loop_nesting -= 1;
                self.loop_contexts.pop();
                Ty::Primitive(Primitive::Unit)
            }
            hir::ExprKind::For {
                local, iter, body, ..
            } => {
                let iter_ty = self.check_expr(*iter);
                let resolved_iter_ty = self.resolve(&iter_ty);
                // WP-C7.9 Packet E: by-VALUE `Vec` iteration is refused here rather than left to
                // be accepted and then refused by lowering. It type-checked and ran in the
                // reference interpreter while no compiler could build it — an accepted program no
                // engine below HIR could execute. Iterating a borrow (`v.iter()`) is supported and
                // is what the diagnostic points at.
                if matches!(resolved_iter_ty, Ty::Core(CoreType::Vec, _)) {
                    self.diags.push(
                        Diagnostic::error(
                            "by-value iteration over Vec<T> is not supported by this compiler; \
                             iterate over a borrow with 'v.iter()'",
                            self.hir.expr(*iter).span,
                        )
                        .with_code("E0105"),
                    );
                }
                let elem_ty = match resolved_iter_ty.clone() {
                    Ty::Range(elem) | Ty::Array(elem, _) | Ty::Slice(elem) => *elem,
                    Ty::Core(CoreType::Vec, args) => args.first().cloned().unwrap_or(Ty::Error),
                    // **`for x in &v` — the spelling everyone reaches for first.**
                    //
                    // It used to be E0001 "requires an iterable value, found '&Vec<T>'", which is
                    // an unhelpful refusal: the value IS iterable, and the borrow is exactly what
                    // Vec iteration wants. Combined with by-value `for x in v` being refused
                    // (E0105), two of the three natural spellings failed and only `v.iter()`
                    // worked — with the practical effect that a `Vec` of non-`Copy` elements
                    // looked unreadable, since indexing it is refused too.
                    //
                    // This is the same borrowed cursor `v.iter()` builds, so the item is `&T`.
                    // `&mut Vec<T>` iterates the same way: the cursor is shared regardless, and
                    // accepting it avoids a second confusing refusal for a caller who happens to
                    // hold a mutable borrow.
                    Ty::Ref { inner, .. }
                        if matches!(inner.as_ref(), Ty::Core(CoreType::Vec, _)) =>
                    {
                        match inner.as_ref() {
                            Ty::Core(CoreType::Vec, args) => Ty::Ref {
                                mutable: false,
                                inner: Box::new(args.first().cloned().unwrap_or(Ty::Error)),
                            },
                            _ => Ty::Error,
                        }
                    }
                    other if self.is_iterator_type(&other) => self.iterator_item_type(&other),
                    Ty::Struct(..) | Ty::Enum(..) => self
                        .user_iterator_item_type(&resolved_iter_ty)
                        .unwrap_or_else(|| {
                            self.diags.push(
                                Diagnostic::error(
                                    format!(
                                        "for-loop requires an iterable value, found '{}'",
                                        self.ty_to_string(&resolved_iter_ty)
                                    ),
                                    self.hir.expr(*iter).span,
                                )
                                .with_code("E0001"),
                            );
                            Ty::Error
                        }),
                    Ty::Error => Ty::Error,
                    other => {
                        self.diags.push(
                            Diagnostic::error(
                                format!(
                                    "for-loop requires an iterable value, found '{}'",
                                    self.ty_to_string(&other)
                                ),
                                self.hir.expr(*iter).span,
                            )
                            .with_code("E0001"),
                        );
                        Ty::Error
                    }
                };

                // AS3 Boundary 4: publish the `Iterator::next` this loop selected. Placed after
                // the element type is resolved, because that resolution is what proves an
                // `Iterator` impl matched.
                self.publish_iterator_use(expr_id, &resolved_iter_ty);
                self.local_types.insert(*local, elem_ty);
                self.local_mutability.insert(*local, false);

                self.loop_contexts.push(LoopContext {
                    allows_value: false,
                    break_ty: Ty::Primitive(Primitive::Unit),
                    has_break: false,
                });
                self.loop_nesting += 1;
                let mut dummy_state = HashSet::new();
                dummy_state.insert(*local);
                let _ = self.check_block(*body, &mut dummy_state);
                self.loop_nesting -= 1;
                self.loop_contexts.pop();
                Ty::Primitive(Primitive::Unit)
            }
            hir::ExprKind::Block(b) => {
                let mut dummy_state = HashSet::new();
                self.check_block(*b, &mut dummy_state)
            }
            hir::ExprKind::Error => Ty::Error,
        };

        self.expr_types.insert(expr_id, ty.clone());
        ty
    }

    /// DEV-052: `Eq::eq(&a, &b)`-style qualified calls to a compiler-known `CoreTrait`'s method.
    /// Unlike `check_qualified_trait_call` (a user-declared trait, which has an
    /// `hir::ItemKind::Trait` item whose declared signature is authoritative for every
    /// implementor), a `CoreTrait` has no such declaration item -- each `impl <CoreTrait> for T`
    /// writes its own method signature directly, so the *matching impl's own* signature is used
    /// instead of one inherited from a shared trait declaration. `receiver_ty`'s own `impl`
    /// search matches by source-text trait name (`self.text(trait_ref.path.span)`), mirroring
    /// `ty_satisfies_operator_bound`'s existing approach for the same compiler-known traits.
    pub(super) fn check_qualified_core_trait_call(
        &mut self,
        // AS3 Boundary 4: the call expression, so the selected impl member can be published.
        call_expr: ExprId,
        core_trait: hir::CoreTrait,
        method_span: Span,
        args: &[ExprId],
        span: Span,
    ) -> Ty {
        let method_name = self.text(method_span).to_string();
        let core_trait_name = core_trait_source_name(core_trait);

        let actual_args: Vec<Ty> = args.iter().map(|arg| self.check_expr(*arg)).collect();
        let Some(first_actual) = actual_args.first() else {
            self.diags.push(
                Diagnostic::error("qualified trait method requires a receiver", span)
                    .with_code("E0005"),
            );
            return Ty::Error;
        };
        let mut receiver_type = self.resolve(first_actual);
        while let Ty::Ref { inner, .. } = receiver_type {
            receiver_type = self.resolve(&inner);
        }
        // **AS3 Boundary 4: publish the selection.** `Eq::eq(&a, &b)` is the explicit spelling of
        // the same dispatch `a == b` performs, so it publishes through the same publisher — one
        // statement of what a qualified core-trait call means, not two.
        let receiver_for_publication = receiver_type.clone();
        self.publish_operator_use(
            call_expr,
            &receiver_for_publication,
            core_trait_name,
            &method_name,
            core_trait,
        );

        let mut selected: Option<hir::FnSig> = None;
        for item in &self.hir.items {
            let hir::ItemKind::Impl {
                trait_: Some(trait_ref),
                self_ty,
                items,
                generics,
            } = &item.kind
            else {
                continue;
            };
            if self.text(trait_ref.path.span) != core_trait_name {
                continue;
            }
            let implementation_type = self.convert_hir_type(*self_ty);
            if self
                .match_impl_type(&implementation_type, &receiver_type, generics)
                .is_none()
            {
                continue;
            }
            selected = items.iter().find_map(|impl_item| match impl_item {
                hir::ImplItem::Fn { def, .. } if self.text(def.sig.name) == method_name => {
                    Some(def.sig.clone())
                }
                _ => None,
            });
            if selected.is_some() {
                break;
            }
        }

        let Some(sig) = selected else {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "type '{}' does not implement '{core_trait_name}'",
                        self.ty_to_string(&receiver_type)
                    ),
                    span,
                )
                .with_code("E0500"),
            );
            return Ty::Error;
        };

        let mut expected = Vec::new();
        if let Some(receiver) = sig.receiver {
            expected.push(match receiver {
                hir::Receiver::Value => receiver_type.clone(),
                hir::Receiver::Ref => Ty::Ref {
                    mutable: false,
                    inner: Box::new(receiver_type.clone()),
                },
                hir::Receiver::RefMut => Ty::Ref {
                    mutable: true,
                    inner: Box::new(receiver_type.clone()),
                },
            });
        }
        expected.extend(
            sig.params
                .iter()
                .map(|param| self.convert_hir_type(param.ty)),
        );
        let result = match sig.ret {
            hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
            hir::RetTy::Ty(ty) => self.convert_hir_type(ty),
            hir::RetTy::Never(_) => Ty::Never,
        };

        if expected.len() != actual_args.len() {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "wrong number of arguments: expected {}, found {}",
                        expected.len(),
                        actual_args.len()
                    ),
                    span,
                )
                .with_code("E0005"),
            );
        }
        for ((expected, actual), arg) in expected.into_iter().zip(actual_args).zip(args) {
            let _ = self.unify(expected, actual, self.hir.expr(*arg).span);
        }
        result
    }

    pub(super) fn check_qualified_trait_call(
        &mut self,
        call_expr: ExprId,
        trait_id: ItemId,
        member: u32,
        args: &[ExprId],
        span: Span,
    ) -> Ty {
        let signature = match &self.hir.item(trait_id).kind {
            hir::ItemKind::Trait { items, .. } => match items.get(member as usize) {
                Some(hir::TraitItem::Method { sig, .. }) => sig.clone(),
                _ => {
                    self.diags.push(
                        Diagnostic::error("trait member is not callable", span).with_code("E0001"),
                    );
                    return Ty::Error;
                }
            },
            _ => return Ty::Error,
        };

        let actual_args: Vec<Ty> = args.iter().map(|arg| self.check_expr(*arg)).collect();
        let Some(first_actual) = actual_args.first() else {
            self.diags.push(
                Diagnostic::error("qualified trait method requires a receiver", span)
                    .with_code("E0005"),
            );
            return Ty::Error;
        };
        let mut receiver_type = self.resolve(first_actual);
        while let Ty::Ref { inner, .. } = receiver_type {
            receiver_type = self.resolve(&inner);
        }
        // AS3 Boundary 4: publish the selection, so the interpreter reads the body instead of
        // scanning the receiver's nominal for the member name.
        let trait_name = match &self.hir.item(trait_id).kind {
            hir::ItemKind::Trait { name, .. } => self.item_text(trait_id, *name).to_string(),
            _ => String::new(),
        };
        let member_name = match &self.hir.item(trait_id).kind {
            hir::ItemKind::Trait { items, .. } => match items.get(member as usize) {
                Some(hir::TraitItem::Method { sig, .. }) => {
                    self.item_text(trait_id, sig.name).to_string()
                }
                _ => String::new(),
            },
            _ => String::new(),
        };
        // **The impl member if the implementor overrides it, otherwise the trait's DEFAULT body.**
        //
        // `operator_impl_member` finds written impl members only. `<T as Tr>::m(&x)` where `T`
        // accepts the default has no impl member to find, and publishing nothing there left the
        // interpreter — which no longer has a name scan — with nothing to select. Third instance of
        // one shape in this packet: a trait default reached by a route other than a `Static` method
        // call.
        let selected = self
            .operator_impl_member(&receiver_type, &trait_name, &member_name)
            .map(|(owner, owner_member, body, _)| (owner, owner_member, body))
            .or_else(|| self.trait_default_member(trait_id, member));
        if let Some((owner, owner_member, body)) = selected {
            // The signature comes from whichever declaration owns the body — an impl member, or
            // the trait itself when the implementor accepts the default.
            let signature = if owner == trait_id {
                self.trait_member_signature(trait_id, owner_member, &receiver_type)
            } else {
                self.declared_member_signature(owner, owner_member)
            };
            if let Some((receiver, params, ret)) = signature {
                let use_ = CallableUse {
                    selection: CalleeSelection::Static {
                        declaration: CallableDeclId::ImplMember {
                            impl_item: owner,
                            member: owner_member,
                        },
                        body,
                    },
                    // **`Self`, published.** A trait DEFAULT body reached this way runs with
                    // `Ty::Param("Self")` throughout, so without this binding a `self.other()`
                    // inside it resolves nothing. The checker knows the receiver here — unlike the
                    // bound path, where the body is only chosen at run time.
                    environment: GenericEnvironment::Static(vec![(
                        GenericBinder::SelfType,
                        receiver_type.clone(),
                    )]),
                    receiver_adjustment: ReceiverAdjustment::Shared { derefs: 0 },
                    receiver_binding: ReceiverBinding::Shared,
                    signature: CallableSigTy {
                        receiver,
                        params,
                        ret,
                    },
                    provenance: DispatchProvenance::Qualified {
                        trait_item: Some(trait_id),
                    },
                };
                self.publish_callable_use(call_expr, use_);
            }
        }

        let impl_infos: Vec<_> = self
            .hir
            .items
            .iter()
            .filter_map(|item| {
                let hir::ItemKind::Impl {
                    generics,
                    trait_: Some(trait_ref),
                    self_ty,
                    items,
                } = &item.kind
                else {
                    return None;
                };
                if trait_ref.res != Res::Item(trait_id) {
                    return None;
                }
                let associated = items
                    .iter()
                    .filter_map(|item| match item {
                        hir::ImplItem::AssocType { name, ty } => Some((*name, *ty)),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                Some((*self_ty, generics.clone(), associated))
            })
            .collect();
        let mut selected = None;
        for (self_type_id, generics, associated) in impl_infos {
            let implementation_type = self.convert_hir_type(self_type_id);
            if let Some(map) = self.match_impl_type(&implementation_type, &receiver_type, &generics)
            {
                selected = Some((associated, map));
                break;
            }
        }

        let Some((associated, map)) = selected else {
            self.diags.push(
                Diagnostic::error("trait is not implemented for receiver type", span)
                    .with_code("E0500"),
            );
            return Ty::Error;
        };

        let previous_self = self.enter_self_scope(receiver_type.clone());
        let previous_assoc = std::mem::take(&mut self.current_assoc_types);
        for (name, ty) in associated {
            let ty = self.convert_hir_type(ty);
            self.current_assoc_types
                .insert(self.text(name).to_string(), self.instantiate_ty(&ty, &map));
        }

        let mut expected = Vec::new();
        if let Some(receiver) = signature.receiver {
            expected.push(match receiver {
                hir::Receiver::Value => receiver_type.clone(),
                hir::Receiver::Ref => Ty::Ref {
                    mutable: false,
                    inner: Box::new(receiver_type.clone()),
                },
                hir::Receiver::RefMut => Ty::Ref {
                    mutable: true,
                    inner: Box::new(receiver_type.clone()),
                },
            });
        }
        expected.extend(signature.params.iter().map(|param| {
            let ty = self.convert_hir_type(param.ty);
            self.instantiate_ty(&ty, &map)
        }));
        let result = match signature.ret {
            hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
            hir::RetTy::Ty(ty) => {
                let ty = self.convert_hir_type(ty);
                self.instantiate_ty(&ty, &map)
            }
            hir::RetTy::Never(_) => Ty::Never,
        };
        self.exit_self_scope(SelfScope {
            assoc_types: Some(previous_assoc),
            ..previous_self
        });

        if expected.len() != actual_args.len() {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "wrong number of arguments: expected {}, found {}",
                        expected.len(),
                        actual_args.len()
                    ),
                    span,
                )
                .with_code("E0005"),
            );
        }
        for ((expected, actual), arg) in expected.into_iter().zip(actual_args).zip(args) {
            let _ = self.unify(expected, actual, self.hir.expr(*arg).span);
        }
        result
    }

    pub(super) fn check_tensor_refine(
        &mut self,
        base: Ty,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        name_span: Span,
        call_span: Span,
    ) -> Ty {
        for arg in args {
            self.check_expr(*arg);
        }
        if !args.is_empty() {
            self.tensor_error("`refine` takes no value arguments", call_span);
        }
        // AS6 packet 4B group 2C: what a refinement produces is tensor semantics.
        tensor_check::eval_tensor_refine(self, base, turbofish, name_span)
    }

    /// The argument half of a call against an already-resolved parameter list. Extracted so the
    /// Core-trait bound path and `check_trait_member_call` cannot drift in how they report an
    /// arity mismatch or unify an argument.
    pub(super) fn check_call_arguments(
        &mut self,
        params_ty: Vec<Ty>,
        args: &[ExprId],
        call_span: Span,
    ) {
        if args.len() != params_ty.len() {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "wrong number of arguments: expected {}, found {}",
                        params_ty.len(),
                        args.len()
                    ),
                    call_span,
                )
                .with_code("E0005"),
            );
        }
        for (arg, param_t) in args.iter().zip(params_ty) {
            let arg_t = self.check_expr(*arg);
            let _ = self.unify(param_t, arg_t, self.hir.expr(*arg).span);
        }
    }

    /// Checks a call's arguments against an already-resolved trait method signature (see
    /// `find_trait_method_sig`) and returns its return type.
    ///
    /// `trait_id` was the declaring trait, carried here for DEV-101 provenance: the signature's
    /// types — including `Self::Item` associated-type spans — had to be read against the trait's
    /// file, which differs from the caller's for a cross-package trait. AS1b-ii-d: those spans
    /// name the trait's file themselves. The parameter is kept so the call sites still say which
    /// trait they resolved.
    /// Returns the method's return type **and** this call site's binding of the method's own
    /// generic parameters, in declaration order — the `method_args` a `CalleeSelection::Bound`
    /// publishes.
    pub(super) fn check_trait_member_call(
        &mut self,
        _trait_id: ItemId,
        sig: &hir::FnSig,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        call_span: Span,
    ) -> (Ty, Vec<Ty>) {
        // **AS3 Boundary 4 (DEV-188): bind the method's OWN generic parameters.**
        //
        // This ignored `sig.generics` entirely, so `U` stayed rigid and *any* trait method that
        // mentioned its own generic parameter was uncallable through a bound — the turbofish was
        // dropped on the floor. The concrete-receiver path (WP-C4.7-8.4) and the trait-default
        // path already do exactly this; only the bound and `Self`-receiver paths did not.
        let mut map: HashMap<String, Ty> = HashMap::new();
        let mut method_args: Vec<Ty> = Vec::new();
        if let Some(generic_args) = turbofish {
            self.validate_generic_arity(sig.generics.len(), generic_args.args.len(), call_span);
        }
        for (index, param) in sig.generics.iter().enumerate() {
            let ty = match turbofish.and_then(|g| g.args.get(index)) {
                Some(hir::GenericArg::Type(t)) => self.convert_hir_type(*t),
                Some(_) => Ty::Error,
                // No turbofish (or too few): infer it from the arguments, as an ordinary generic
                // call does. `t.to(1)` must work without a turbofish for the same reason `f(1)`
                // does.
                None => self.new_type_var(),
            };
            map.insert(self.decl_text(param.name).to_string(), ty.clone());
            method_args.push(ty);
        }

        // AS1b-ii-d: this used to swap `self.file` to the trait's file to convert the signature
        // and swap back for the arguments. The signature's spans name the trait's file and the
        // arguments' name the caller's, so both convert correctly with no swap at all.
        let params_ty: Vec<Ty> = sig
            .params
            .iter()
            .map(|p| {
                let ty = self.convert_hir_type(p.ty);
                self.instantiate_ty(&ty, &map)
            })
            .collect();
        let ret_ty = match sig.ret {
            hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
            hir::RetTy::Ty(t) => {
                let ty = self.convert_hir_type(t);
                self.instantiate_ty(&ty, &map)
            }
            hir::RetTy::Never(_) => Ty::Never,
        };
        self.check_call_arguments(params_ty, args, call_span);
        // Resolve after the arguments have constrained any inference variable introduced above, so
        // an omitted turbofish still publishes the type the call site actually settled on rather
        // than an unresolved `_infer_N`.
        let method_args = method_args.iter().map(|ty| self.resolve(ty)).collect();
        (self.resolve(&ret_ty), method_args)
    }

    pub(super) fn resolve_method(
        &mut self,
        base_expr: ExprId,
        name_span: Span,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        call_span: Span,
        // WP-C4.7-8.4: the call expression itself, used to key this call site's METHOD-level
        // generic instantiation for MIR monomorphisation.
        call_expr: ExprId,
    ) -> Ty {
        let base_ty = self.check_expr(base_expr);
        // WP-C4.7-6.3: method resolution must branch on a CONCRETE receiver type, and a literal
        // receiver (`3.cmp(&5)`) has no other constraint to wait for — settle it here rather than
        // failing with "method call on non-struct/enum type '_infer_N'".
        // WP-C6.2b-F2: default int literals inside the receiver too, so a concrete-instance
        // impl (`impl Get for W<Int32>`) matches `let w = W { v: 7 }; w.get()`.
        let resolved_base = self.default_int_literals_deep(&base_ty);
        let name_str = self.text(name_span).to_string();

        if self.options.tensor() && name_str == "refine" {
            return self.check_tensor_refine(resolved_base, turbofish, args, name_span, call_span);
        }

        // AS3 Boundary 2 hardening: TYPE-METHOD-002's auto-dereference is a decision the CALL SITE
        // makes, and it was being discarded. Counting the peels here is what lets
        // `ReceiverAdjustment` publish it instead of every consumer re-deriving it from the
        // receiver's type — which is precisely the reconstruction this packet exists to remove.
        let mut receiver_ty = resolved_base.clone();
        let mut receiver_derefs: u8 = 0;
        let mut outermost_ref_is_mut = false;
        if let Ty::Ref { mutable, .. } = &receiver_ty {
            outermost_ref_is_mut = *mutable;
        }
        while let Ty::Ref { inner, .. } = receiver_ty {
            receiver_ty = self.resolve(&inner);
            receiver_derefs = receiver_derefs.saturating_add(1);
        }

        // DEV-067(b) (WP-C4.7-7): a method call on a BOUNDED generic parameter resolves through
        // the parameter's declared bounds. This tested `resolved_base` — the UNPEELED receiver —
        // so it matched `t: T` but never `t: &T`, and `fn f<T: Speak>(t: &T) { t.speak() }`
        // failed E0302 "method 'speak' not found for type '&T'". TYPE-METHOD-002 requires
        // auto-dereference to peel leading `&`/`&mut` before receiver matching, exactly as the
        // concrete-type path below already did with `receiver_ty`; using the same peeled type
        // here makes the bounded-parameter path obey the same rule.
        // DEV-DISPLAY-DISPATCH: candidate collection over the bounds is ADDITIVE across both
        // kinds of trait, and there is ONE selection step afterwards. Before this, the loop
        // returned on the first bound that supplied the name, and only a bound naming a
        // `hir::ItemKind::Trait` was ever consulted at all — so a compiler-known trait
        // (`Display`, `Ord`, `Clone`, ...) contributed nothing, and two bounds supplying the same
        // name were resolved by declaration order rather than reported as ambiguous.
        if let Ty::Param(p_name) = &receiver_ty {
            let p_name = p_name.clone();
            let candidates = self.bound_method_candidates(&p_name, &name_str);
            if candidates.len() > 1 {
                // Same rule the concrete-receiver path applies when two impls supply the name.
                // Order of the bounds is deliberately not a tie-breaker, and being
                // compiler-known is deliberately not a preference.
                self.diags.push(
                    Diagnostic::error("ambiguous trait method call", call_span)
                        .with_code("E0203")
                        .with_label(format!(
                            "'{}' is declared by more than one trait bound on '{}': {}",
                            name_str,
                            p_name,
                            self.bound_trait_list(&candidates)
                        )),
                );
                for &arg in args {
                    let _ = self.check_expr(arg);
                }
                return Ty::Error;
            }
            if let Some(candidate) = candidates.into_iter().next() {
                // DEV-BOUND-TRAIT-IDENTITY: record WHICH trait supplied the method, so the
                // engines below select the same implementation rather than the first impl on the
                // receiver's nominal that happens to declare the name.
                self.bound_trait_calls.insert(
                    call_expr,
                    match &candidate {
                        BoundMethod::User { trait_id, .. } => Res::Item(*trait_id),
                        BoundMethod::Core { core_trait, .. } => Res::CoreTrait(*core_trait),
                    },
                );
                // WP-C6.2c: a trait method returning `Self::Item` yields the receiver's
                // projection (`T::Item`), which is then pinned by any explicit
                // `T: Trait<Item = ..>` binding in scope.
                let (ret, method_args) =
                    self.check_bound_method_call(&candidate, &p_name, turbofish, args, call_span);
                let ret = Self::subst_self(&ret, &p_name);
                let binding_map = self.assoc_binding_map();
                let ret = self.normalize_projections(&ret, &binding_map);
                // **AS3 Boundary 4 step 2: publish the LATE-BOUND obligation.**
                //
                // This branch previously returned here, so a call on a bounded generic parameter
                // published no `CallableUse` at all — the missing third category. The body cannot
                // be named: `Self` is `Ty::Param(p_name)` and stays parametric until the enclosing
                // function is instantiated. What IS fixed is the obligation, and that is what a
                // `Bound` selection records.
                self.publish_bound_use(
                    call_expr,
                    &candidate,
                    &p_name,
                    &name_str,
                    &ret,
                    method_args,
                );
                return ret;
            }
        }

        // DEV-051: `self.other_method()` called from inside a trait's own default-method body
        // has `current_self_ty == Ty::Param("Self")` (set alongside `current_trait_id` while
        // checking `hir::ItemKind::Trait`'s default bodies), so `self`'s dereferenced type here
        // is `Ty::Param("Self")` -- there's no concrete `impl` to match against yet, since the
        // default body is checked once, generically, at the trait declaration site rather than
        // once per implementor. The trait's own declared signature for `name_str` (required or
        // another default) is authoritative regardless: every real implementor is separately
        // checked elsewhere to provide a matching method, so calling it through `self` from a
        // sibling default body is always legal. (Checked after the deref loop above, unlike the
        // bounded-generic-parameter case just above, because a generic parameter received by
        // value has no reference to peel off, but `self` is always received by reference.)
        if let Ty::Param(p_name) = &receiver_ty {
            if p_name == "Self" {
                if let Some(trait_id) = self.current_trait_id {
                    if let Some(sig) = self.find_trait_method_sig(trait_id, &name_str) {
                        // Same DEV-188 repair: a sibling default body calling another generic
                        // trait method through `self` had `U` rigid for the same reason.
                        let (ret, method_args) = self
                            .check_trait_member_call(trait_id, &sig, turbofish, args, call_span);
                        // **AS3 Boundary 4 (DEV-190): publish this call too.**
                        //
                        // Like the bounded-parameter branch before step 2, this returned without
                        // publishing anything — so `self.id()` inside `fn twice(&self)` had no
                        // `CallableUse`, and both engines had to fall back to a name scan. It is a
                        // `Bound` selection by the same argument: `Self` is a parameter, the trait
                        // is known, and the body is fixed only once an implementor is chosen.
                        let candidate = BoundMethod::User { trait_id, sig };
                        self.publish_bound_use(
                            call_expr,
                            &candidate,
                            "Self",
                            &name_str,
                            &ret,
                            method_args,
                        );
                        return ret;
                    }
                }
            }
        }

        if self.options.tensor() {
            if let Ty::Extension(ext) = &receiver_ty {
                if let ExtensionTy::Tensor(_) = &**ext {
                    return self.check_tensor_method_call(
                        &receiver_ty,
                        &name_str,
                        turbofish,
                        args,
                        name_span,
                        call_span,
                    );
                }
                if let ExtensionTy::Model(model) = &**ext {
                    return self
                        .check_model_method_call(model, &name_str, args, name_span, call_span);
                }
            }
        }

        let mut candidates = Vec::new();

        // DEV-069: this scans EVERY impl in the program, including impls declared in other
        // files, so method names are read against each impl's OWN file — not `self.file`, which
        // is the file of the item being checked.
        for (impl_index, item) in self.hir.items.iter().enumerate() {
            let impl_item_id = ItemId(impl_index as u32);
            if let hir::ItemKind::Impl {
                self_ty: impl_self_ty_id,
                items,
                trait_,
                generics,
                ..
            } = &item.kind
            {
                // CD-358: both the self-type conversion and the generic-name keying read spans
                // from the IMPL's file. Without this, `impl<T> Wrap<T>` resolved through a module
                // boundary produced a parameter named from the caller's file — `Wrap<T>::get`
                // returned `&S` — and no substitution could ever fire.
                let impl_self_ty = self.convert_hir_type(*impl_self_ty_id);
                let matched = self.match_impl_type(&impl_self_ty, &receiver_ty, generics);
                let Some(map) = matched else {
                    continue;
                };

                for impl_item in items {
                    if let hir::ImplItem::Fn { vis, def } = impl_item {
                        let method_name_str = self.item_text(impl_item_id, def.sig.name);
                        if method_name_str == name_str {
                            candidates.push((
                                def,
                                trait_.is_some(),
                                map.clone(),
                                impl_self_ty.clone(),
                                matches!(vis, Some(crate::ast::Vis::Pub)),
                                impl_item_id,
                            ));
                        }
                    }
                }
            }
        }

        let inherent: Vec<_> = candidates
            .iter()
            .filter(|(_, is_trait, _, _, _, _)| !is_trait)
            .collect();
        // WP-C6.2b-F1: pick the chosen candidate, enforce its visibility, then hand the same
        // 4-tuple downstream. A trait method is visible via its trait's own path rules; the
        // private-impl-member check applies to inherent (and inherent-selected) methods.
        let chosen: Option<MethodCandidate> = if let Some(candidate) = inherent.first() {
            Some((**candidate).clone())
        } else if candidates.len() == 1 {
            candidates.first().cloned()
        } else if candidates.len() > 1 {
            self.diags.push(
                Diagnostic::error("ambiguous trait method call", call_span).with_code("E0203"),
            );
            None
        } else {
            None
        };
        if let Some((_, is_trait, _, _, is_pub, impl_item_id)) = &chosen {
            if !is_trait {
                self.check_member_visible(*is_pub, *impl_item_id, "method", &name_str, call_span);
            }
            // DEV-169: `Drop::drop` MUST NOT be called explicitly (03-Type-System.md, "Copy and
            // Drop"). Accepting it was a DOUBLE DESTRUCTION, not merely an over-acceptance:
            // `r.drop()` ran the destructor once for the call and again when the value went out of
            // scope. Confirmed empirically before the fix — `dropped / after / dropped`.
            //
            // Checked at IMPL-MEMBER SELECTION rather than on the method's name, so it fires
            // exactly when a call resolves into an `impl Drop for T` block and never for an
            // unrelated method that happens to be called `drop`.
            self.reject_explicit_drop(*impl_item_id, &name_str, call_span);
        }
        // CD-358: the impl's ItemId is carried through, because the signature conversion below
        // must read its spans against the impl's own file.
        let selected = chosen.map(|(def, is_trait, map, self_ty, _, impl_item_id)| {
            (def, is_trait, map, self_ty, impl_item_id)
        });

        // WP-C1.3 (2026-07-17): fall back to a trait's own default method body when no impl
        // overrides it. `candidates` above only ever collects `ImplItem::Fn` overrides -- a
        // trait method declared with a real body (03-Type-System.md trait defaults) was never
        // consulted at all, so calling an un-overridden default method failed to type-check
        // with E0302 "method not found" even though the interpreter (once its own matching gap
        // is fixed) has a real body to run. Confirmed empirically before this fix. See
        // COMPILER-STATE.md DEV-013.
        let default_fallback = if selected.is_none() {
            self.hir.items.iter().find_map(|item| {
                let hir::ItemKind::Impl {
                    self_ty: impl_self_ty_id,
                    trait_: Some(trait_ref),
                    generics,
                    ..
                } = &item.kind
                else {
                    return None;
                };
                let impl_self_ty = self.convert_hir_type(*impl_self_ty_id);
                let map = self.match_impl_type(&impl_self_ty, &receiver_ty, generics)?;
                let Res::Item(trait_id) = trait_ref.res else {
                    return None;
                };
                let hir::ItemKind::Trait {
                    items: trait_items, ..
                } = &self.hir.item(trait_id).kind
                else {
                    return None;
                };
                // DEV-069: a trait default's name belongs to the trait's own file, which may
                // differ from both the impl's file and the file being checked.
                trait_items.iter().find_map(|trait_item| match trait_item {
                    hir::TraitItem::Method {
                        sig,
                        body: Some(body),
                    } if self.item_text(trait_id, sig.name) == name_str => Some((
                        sig.clone(),
                        map.clone(),
                        impl_self_ty.clone(),
                        trait_id,
                        *body,
                    )),
                    _ => None,
                })
            })
        } else {
            None
        };

        if let Some((sig, mut map, impl_self_ty, trait_id, trait_body)) = default_fallback {
            // CD-358: a trait default's signature is declared in the TRAIT's file, which may
            // differ from the impl's file and from the file under check. DEV-069 already applied
            // that rule to the default's NAME; its parameter and return types need it too.
            // WP-C4.7-9 audit: a TRAIT-DEFAULT method may declare its own generic parameters
            // too (`02:64`). WP-C4.7-8.4 gave the selected-impl path fresh per-call-site
            // variables for those; this path had the same gap, so `d.say(5)` on an
            // un-overridden `fn say<U>(&self, x: U) -> U` still failed with `U` rigid.
            if let Some(args) = turbofish {
                self.validate_generic_arity(sig.generics.len(), args.args.len(), call_span);
                for (param, arg) in sig.generics.iter().zip(&args.args) {
                    let ty = match arg {
                        hir::GenericArg::Type(ty) => self.convert_hir_type(*ty),
                        _ => Ty::Error,
                    };
                    map.insert(self.text(param.name).to_string(), ty);
                }
            } else {
                for param in &sig.generics {
                    let infer = self.new_type_var();
                    map.insert(self.text(param.name).to_string(), infer);
                }
            }
            // Record this call site's method-level instantiation for MIR monomorphisation, as
            // the selected-impl path does.
            // A3c-S: the full environment, including `Self` and the trait's own parameters, which
            // the positional record above cannot express. A trait default body carries
            // `Ty::Param("Self")` from the checker, so without this the oracle has no binding for
            // it at all (DEV-176).
            let trait_generics = match &self.hir.item(trait_id).kind {
                hir::ItemKind::Trait { generics, .. } => generics.clone(),
                _ => Vec::new(),
            };
            let env_map = map.clone();
            let trait_names: Vec<String> = trait_generics
                .iter()
                .map(|param| self.item_text(trait_id, param.name).to_string())
                .collect();
            let own_names: Vec<String> = sig
                .generics
                .iter()
                .map(|param| self.decl_text(param.name).to_string())
                .collect();
            let use_self_ty = Some(impl_self_ty.clone());
            // The receiver this use binds, instantiated. `None` when the declaration takes no
            // receiver, which keeps the published signature comparable with A3b's body signature.
            let receiver_self_ty = bound_receiver_ty(
                sig.receiver.as_ref(),
                self.instantiate_ty(&impl_self_ty, &map),
            );
            self.publish_callable_env(PublishedEnv {
                call_expr,
                body: trait_body,
                self_ty: Some(impl_self_ty.clone()),
                impl_names: &trait_names,
                own_names: &own_names,
                own_is_method: true,
                map: &env_map,
            });
            if matches!(sig.receiver, Some(hir::Receiver::RefMut))
                && !self.is_mutable_place(base_expr)
            {
                self.diags.push(
                    Diagnostic::error(
                        "mutable method receiver requires a mutable place",
                        name_span,
                    )
                    .with_code("E0400"),
                );
            }
            let previous_self = self.enter_self_scope(impl_self_ty);
            let params_ty: Vec<Ty> = sig
                .params
                .iter()
                .map(|p| {
                    let ty = self.convert_hir_type(p.ty);
                    self.instantiate_ty(&ty, &map)
                })
                .collect();
            let ret_ty = match sig.ret {
                hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
                hir::RetTy::Ty(t) => {
                    let ty = self.convert_hir_type(t);
                    self.instantiate_ty(&ty, &map)
                }
                hir::RetTy::Never(_) => Ty::Never,
            };
            self.exit_self_scope(previous_self);

            // AS3 Boundary 2: the same selection, published so an engine can CONSUME it rather
            // than re-derive it. Receiver ADJUSTMENT (what the call site did) and receiver BINDING
            // (what the callable binds) are separate fields: they correlate here, but they are
            // different authorities and AS4 asks about the binding side.
            let receiver_binding = match sig.receiver {
                Some(hir::Receiver::Value) => ReceiverBinding::ByValue,
                Some(hir::Receiver::Ref) => ReceiverBinding::Shared,
                Some(hir::Receiver::RefMut) => ReceiverBinding::Exclusive,
                None => ReceiverBinding::None,
            };
            let use_bindings =
                Self::env_bindings(&use_self_ty, &trait_names, &own_names, true, &env_map);
            self.publish_named_use(
                call_expr,
                trait_body,
                use_bindings,
                receiver_adjustment_for(receiver_derefs, outermost_ref_is_mut, receiver_binding),
                receiver_binding,
                CallableSigTy {
                    // AS3 Boundary 2 hardening: a real method's A3b body signature carries its
                    // receiver, so publishing `None` here made the §3.4 invariant unenforceable —
                    // the two signatures would disagree on every method. The instantiated `Self`
                    // is the receiver this use binds.
                    receiver: receiver_self_ty.clone(),
                    params: params_ty.clone(),
                    ret: ret_ty.clone(),
                },
                DispatchProvenance::Qualified { trait_item: None },
            );

            if args.len() != params_ty.len() {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "wrong number of arguments: expected {}, found {}",
                            params_ty.len(),
                            args.len()
                        ),
                        call_span,
                    )
                    .with_code("E0005"),
                );
            }
            for (arg, param_t) in args.iter().zip(params_ty) {
                let arg_t = self.check_expr(*arg);
                let _ = self.unify(param_t, arg_t, self.hir.expr(*arg).span);
            }
            return ret_ty;
        }

        if let Some((def, _, mut map, impl_self_ty, impl_item_id)) = selected {
            // CD-358: every name below — the method's own generic parameters, and the parameter
            // and return TYPES — is a span into the impl's file.
            // WP-C4.7-8.4: the candidate's `map` carries only the IMPL's generic parameters. A
            // method may declare its OWN (`02:64` puts `GenericParams?` on every `FunctionSig`,
            // and `02:120` makes an impl item a `Function`), and those need a fresh inference
            // variable PER CALL SITE — otherwise the signature is used with `U` still a rigid
            // `Ty::Param` and every argument fails to unify against it ("expected 'U', found …").
            // The associated-function path already did exactly this; only the method path did not.
            if let Some(args) = turbofish {
                self.validate_generic_arity(def.sig.generics.len(), args.args.len(), call_span);
                for (param, arg) in def.sig.generics.iter().zip(&args.args) {
                    let ty = match arg {
                        hir::GenericArg::Type(ty) => self.convert_hir_type(*ty),
                        _ => Ty::Error,
                    };
                    map.insert(self.decl_text(param.name).to_string(), ty);
                }
            } else {
                for param in &def.sig.generics {
                    let infer = self.new_type_var();
                    map.insert(self.decl_text(param.name).to_string(), infer);
                }
            }
            // WP-C4.7-8.4: record this call site's METHOD-level instantiation for MIR
            // monomorphisation, keyed by the method-call expression — the same mechanism C4.5c
            // uses for top-level generic fns, which had no method equivalent. Recorded in the
            // method's own declaration order, and only when the method actually declares
            // parameters, so non-generic methods add no entries.
            // A3c-S: the full environment. `map` already carries the IMPL's parameters from
            // candidate selection plus the method's own — everything DEV-176 needs was computed
            // here and thrown away, except for the positional slice above.
            let impl_generics = match &self.hir.item(impl_item_id).kind {
                hir::ItemKind::Impl { generics, .. } => generics.clone(),
                _ => Vec::new(),
            };
            let env_map = map.clone();
            let env_self = impl_self_ty.clone();
            let env_generics = def.sig.generics.clone();
            let impl_names: Vec<String> = impl_generics
                .iter()
                .map(|param| self.item_text(impl_item_id, param.name).to_string())
                .collect();
            let own_names: Vec<String> = env_generics
                .iter()
                .map(|param| self.decl_text(param.name).to_string())
                .collect();
            let use_self_ty = Some(env_self.clone());
            let receiver_self_ty = bound_receiver_ty(
                def.sig.receiver.as_ref(),
                self.instantiate_ty(&env_self, &map),
            );
            self.publish_callable_env(PublishedEnv {
                call_expr,
                body: def.body,
                self_ty: Some(env_self),
                impl_names: &impl_names,
                own_names: &own_names,
                own_is_method: true,
                map: &env_map,
            });
            if matches!(def.sig.receiver, Some(hir::Receiver::RefMut))
                && !self.is_mutable_place(base_expr)
            {
                self.diags.push(
                    Diagnostic::error(
                        "mutable method receiver requires a mutable place",
                        name_span,
                    )
                    .with_code("E0400"),
                );
            }
            let previous_self = self.enter_self_scope(impl_self_ty);
            let params_ty: Vec<Ty> = def
                .sig
                .params
                .iter()
                .map(|p| {
                    let ty = self.convert_hir_type(p.ty);
                    self.instantiate_ty(&ty, &map)
                })
                .collect();
            let ret_ty = match def.sig.ret {
                hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
                hir::RetTy::Ty(t) => {
                    let ty = self.convert_hir_type(t);
                    self.instantiate_ty(&ty, &map)
                }
                hir::RetTy::Never(_) => Ty::Never,
            };
            self.exit_self_scope(previous_self);

            // AS3 Boundary 2: the same selection, published so an engine can CONSUME it rather
            // than re-derive it. Receiver ADJUSTMENT (what the call site did) and receiver BINDING
            // (what the callable binds) are separate fields: they correlate here, but they are
            // different authorities and AS4 asks about the binding side.
            let receiver_binding = match def.sig.receiver {
                Some(hir::Receiver::Value) => ReceiverBinding::ByValue,
                Some(hir::Receiver::Ref) => ReceiverBinding::Shared,
                Some(hir::Receiver::RefMut) => ReceiverBinding::Exclusive,
                None => ReceiverBinding::None,
            };
            let use_bindings =
                Self::env_bindings(&use_self_ty, &impl_names, &own_names, true, &env_map);
            self.publish_named_use(
                call_expr,
                def.body,
                use_bindings,
                receiver_adjustment_for(receiver_derefs, outermost_ref_is_mut, receiver_binding),
                receiver_binding,
                CallableSigTy {
                    // AS3 Boundary 2 hardening: a real method's A3b body signature carries its
                    // receiver, so publishing `None` here made the §3.4 invariant unenforceable —
                    // the two signatures would disagree on every method. The instantiated `Self`
                    // is the receiver this use binds.
                    receiver: receiver_self_ty.clone(),
                    params: params_ty.clone(),
                    ret: ret_ty.clone(),
                },
                DispatchProvenance::Inherent,
            );

            if args.len() != params_ty.len() {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "wrong number of arguments: expected {}, found {}",
                            params_ty.len(),
                            args.len()
                        ),
                        call_span,
                    )
                    .with_code("E0005"),
                );
            }

            for (arg, param_t) in args.iter().zip(params_ty) {
                let arg_t = self.check_expr(*arg);
                let _ = self.unify(param_t, arg_t, self.hir.expr(*arg).span);
            }

            ret_ty
        } else if let Some((params_ty, ret_ty, needs_mut)) =
            self.core_method_signature(&receiver_ty, &name_str, name_span)
        {
            // **AS3 Boundary 4: a core container that compares its elements.**
            //
            // `vec.contains(&x)`, `set.insert(v)`, `map.get(k)` and friends run `Eq::eq` on the
            // ELEMENT when it is a user nominal — the interpreter's `language_equal`. That site had
            // no expression id and so scanned for a member named `eq`; publishing here gives it
            // one, keyed on the container call itself.
            self.publish_core_element_eq_use(call_expr, &receiver_ty, &name_str);
            if needs_mut && !self.is_mutable_place(base_expr) {
                self.diags.push(
                    Diagnostic::error(
                        "mutable method receiver requires a mutable place",
                        name_span,
                    )
                    .with_code("E0400"),
                );
            }
            if args.len() != params_ty.len() {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "wrong number of arguments: expected {}, found {}",
                            params_ty.len(),
                            args.len()
                        ),
                        call_span,
                    )
                    .with_code("E0005"),
                );
            }
            for (arg, param_ty) in args.iter().zip(params_ty) {
                let arg_ty = self.check_expr(*arg);
                let _ = self.unify(param_ty, arg_ty, self.hir.expr(*arg).span);
            }
            ret_ty
        } else {
            let is_ok_type = matches!(
                resolved_base,
                Ty::Struct(..) | Ty::Enum(..) | Ty::Ref { .. } | Ty::Param(_) | Ty::Error
            );
            if !is_ok_type {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "method call on non-struct/enum type '{}'",
                            self.ty_to_string(&resolved_base)
                        ),
                        call_span,
                    )
                    .with_code("E0304"),
                );
            } else if let Ty::Param(p_name) = &receiver_ty {
                // DEV-DISPLAY-DISPATCH: on a generic parameter, "not found" is the wrong story.
                // The method exists; the parameter is simply not bounded by the trait that
                // declares it, and the fix is to write that bound. Naming the trait is derived
                // from the traits actually in scope — nothing here keys on a method name.
                let providers = self.traits_declaring_method(&name_str);
                let mut diagnostic = if providers.is_empty() {
                    Diagnostic::error(
                        format!("method '{name_str}' not found for type '{p_name}'"),
                        call_span,
                    )
                    .with_code("E0302")
                    .with_label(format!(
                        "no trait in scope declares a method named '{name_str}'"
                    ))
                } else {
                    Diagnostic::error(
                        format!(
                            "method '{}' requires the bound '{}: {}'",
                            name_str, p_name, providers[0]
                        ),
                        call_span,
                    )
                    .with_code("E0302")
                    .with_label(format!(
                        "'{p_name}' has no bound that declares '{name_str}'"
                    ))
                };
                if providers.len() > 1 {
                    diagnostic = diagnostic.with_note(format!(
                        "'{}' is declared by: {}. Bound '{}' by the one this call means.",
                        name_str,
                        providers.join(", "),
                        p_name
                    ));
                }
                self.diags.push(diagnostic);
            } else {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "method '{}' not found for type '{}'",
                            name_str,
                            self.ty_to_string(&resolved_base)
                        ),
                        call_span,
                    )
                    .with_code("E0302"),
                );
            }
            Ty::Error
        }
    }

    /// DEV-139: whether the type parameter `param_name` declares `required`, anywhere in the
    /// generic environment the current body is checked in.
    ///
    /// **That environment is the impl's parameters PLUS the function's own**, and this is the one
    /// place that assembles it. Both bound questions — operator desugaring
    /// (`ty_satisfies_operator_bound`) and trait-bound satisfaction (`satisfies_bound`) — read it
    /// through here, so they cannot drift apart; before this each kept its own copy of the lookup
    /// and both consulted `current_fn_generics` alone. WP-C6.2b-F5 had already brought impl-head
    /// generics into scope for method bodies via `current_impl_generics`; the bound lookups simply
    /// never asked. So
    ///
    /// ```stark
    /// impl<T: Ord> Pair<T> {
    ///     fn larger(&self) -> &T { if self.a > self.b { &self.a } else { &self.b } }
    /// }
    /// ```
    ///
    /// was refused E0500 "type 'T' does not satisfy operator trait 'Ord'" while the identical
    /// comparison in a free `fn largest<T: Ord>` was accepted — the bound was declared, just not
    /// looked at.
    ///
    /// Impl generics are searched FIRST only for readability; a method may not redeclare an
    /// impl-level parameter name, so the two sets are disjoint and order cannot change the answer.
    /// Whether generic parameter `param_name` carries a bound denoting the CORE trait `required`.
    ///
    /// **DEV-171: by resolved identity, not by spelling.** The operator path compared
    /// `text(bound.path.span)` against `"Eq"`, so an unrelated trait imported under that name
    /// authorised `==`:
    ///
    /// ```text
    /// mod fake { pub trait Eq { fn unrelated(&self) -> Int32; } }
    /// use fake::Eq;
    /// fn compare<T: Eq>(a: T, b: T) -> Bool { a == b }   // was ACCEPTED
    /// ```
    ///
    /// Written qualified (`T: fake::Eq`) the same program was rejected — the tell that the answer
    /// depended on how the bound was spelled. Operators dispatch to the CANONICAL Core trait
    /// (03-Type-System.md, "Operators and Traits"), so only that trait discharges the obligation.
    ///
    /// **This is deliberately separate from [`Self::param_declares_bound`].** That one answers a
    /// different question — "does this parameter carry the bound being discharged", where the
    /// bound may be any user trait — and folding the two together made every qualified user-trait
    /// bound stop satisfying anything, because a user trait is not a Core trait. Caught by
    /// `dev_bound_trait_identity::a_qualified_bound_forwards_through_nested_generics` on CI.
    pub(super) fn param_declares_core_bound(&self, param_name: &str, required: &str) -> bool {
        let Some(required) = crate::resolve::resolve_core_trait(required) else {
            return false;
        };
        self.current_impl_generics
            .iter()
            .flatten()
            .chain(self.current_fn_generics.iter().flatten())
            .any(|param| {
                self.text(param.name) == param_name
                    && param.bounds.iter().any(|bound| {
                        hir::resolved_bound_trait(self.hir, bound)
                            == Some(hir::BoundTrait::Core(required))
                    })
            })
    }

    pub(super) fn walk_display_ty(
        &mut self,
        root: ExprId,
        ty: &Ty,
        path: DisplayPath,
        span: Span,
        depth: u32,
    ) {
        // A displayable type is a finite tree, but `Ty` is produced by inference and a defect
        // elsewhere should not become a stack overflow here.
        if depth > 64 {
            return;
        }
        let ty = self.resolve(ty);
        match &ty {
            // A reference renders as its referent: `Display::fmt` borrows anyway.
            Ty::Ref { inner, .. } => {
                let inner = (**inner).clone();
                self.walk_display_ty(root, &inner, path, span, depth + 1);
            }
            Ty::Tuple(elems) => {
                for (index, elem) in elems.clone().into_iter().enumerate() {
                    let step = DisplayStep::TupleField(index as u32);
                    self.walk_display_ty(root, &elem, path.child(step), span, depth + 1);
                }
            }
            Ty::Array(elem, _) => {
                let elem = (**elem).clone();
                let next = path.child(DisplayStep::ArrayElement);
                self.walk_display_ty(root, &elem, next, span, depth + 1);
            }
            Ty::Slice(elem) => {
                let elem = (**elem).clone();
                let next = path.child(DisplayStep::SliceElement);
                self.walk_display_ty(root, &elem, next, span, depth + 1);
            }
            Ty::Core(CoreType::Vec, args) => {
                if let Some(elem) = args.first().cloned() {
                    let next = path.child(DisplayStep::VecElement);
                    self.walk_display_ty(root, &elem, next, span, depth + 1);
                }
            }
            Ty::Core(CoreType::Option, args) => {
                if let Some(inner) = args.first().cloned() {
                    let next = path.child(DisplayStep::OptionSome);
                    self.walk_display_ty(root, &inner, next, span, depth + 1);
                }
            }
            Ty::Core(CoreType::Result, args) => {
                let args = args.clone();
                if let Some(ok) = args.first().cloned() {
                    let next = path.child(DisplayStep::ResultOk);
                    self.walk_display_ty(root, &ok, next, span, depth + 1);
                }
                if let Some(err) = args.get(1).cloned() {
                    let next = path.child(DisplayStep::ResultErr);
                    self.walk_display_ty(root, &err, next, span, depth + 1);
                }
            }
            // **STOP.** A user nominal with a `Display` impl renders through it and no further.
            Ty::Struct(..) | Ty::Enum(..) => self.publish_display_static(root, &ty, path, span),
            // A generic parameter's body is not knowable here — `show<T: Display>` is checked once
            // with `T` unbound, and one `show` may be instantiated at several types. The obligation
            // is fixed, and that is what `Bound` records (§3).
            Ty::Param(name) => {
                let name = name.clone();
                self.publish_display_bound(root, &name, path);
            }
            // Primitives, `String`, `Ordering`, `IOError` — rendered by the engines themselves,
            // with no user callable to name.
            _ => {}
        }
    }

    pub(super) fn publish_bound_operator_use(
        &mut self,
        expr_id: ExprId,
        param_name: &str,
        method: &str,
        core: hir::CoreTrait,
    ) {
        let candidates = self.bound_method_candidates(param_name, method);
        let Some(BoundMethod::Core {
            core_trait,
            method: contract,
            trait_args,
        }) = candidates
            .into_iter()
            .find(|c| matches!(c, BoundMethod::Core { core_trait, .. } if *core_trait == core))
        else {
            // No such bound in scope. Arithmetic on a `T: Num` reaches here and correctly
            // publishes nothing: `Num` is compiler-known and primitives-only, so there is no
            // user body for a call site to name.
            return;
        };
        let self_ty = Ty::Param(param_name.to_string());
        let trait_arg_tys: Vec<Ty> = trait_args
            .iter()
            .map(|ty| self.convert_hir_type(*ty))
            .collect();
        let params: Vec<Ty> = contract
            .params
            .iter()
            .map(|term| self.contract_ty_to_ty(*term, &self_ty, &trait_arg_tys))
            .collect();
        let ret = match contract.ret {
            None => Ty::Primitive(Primitive::Unit),
            Some(term) => self.contract_ty_to_ty(term, &self_ty, &trait_arg_tys),
        };
        let use_ = CallableUse {
            selection: CalleeSelection::Bound {
                trait_: hir::BoundTrait::Core(core_trait),
                member: contract.name.to_string(),
                self_ty: self_ty.clone(),
                trait_args: trait_arg_tys,
                // A core trait's contract cannot declare method-level generics (DEV-188).
                method_args: Vec::new(),
            },
            environment: GenericEnvironment::FromBoundSelection,
            receiver_adjustment: ReceiverAdjustment::Shared { derefs: 0 },
            receiver_binding: ReceiverBinding::Shared,
            signature: CallableSigTy {
                receiver: bound_receiver_ty(contract.receiver.as_ref(), self_ty),
                params,
                ret,
            },
            provenance: DispatchProvenance::Bound {
                trait_: hir::BoundTrait::Core(core_trait),
            },
        };
        self.publish_callable_use(expr_id, use_);
    }

    pub(super) fn publish_operator_use(
        &mut self,
        expr_id: ExprId,
        operand: &Ty,
        trait_name: &str,
        method: &str,
        core: hir::CoreTrait,
    ) {
        let operand = self.resolve(operand);
        // **AS3 Boundary 4 (DEV-191): an operator on a BOUNDED GENERIC PARAMETER.**
        //
        // `a == b` inside `fn same<T: Eq>(a: T, b: T)` published nothing at all — this guard
        // returned on `Ty::Param`. So MIR, which sees the monomorphised `P` and lowers a user
        // `Eq::eq` call, had no published record to consume and fell back to scanning impls by
        // name. It is the same missing binding time step 2 found for method calls, on the operator
        // path: the trait is fixed here, the body only once `T` is instantiated.
        if let Ty::Param(param_name) = &operand {
            let param_name = param_name.clone();
            self.publish_bound_operator_use(expr_id, &param_name, method, core);
            return;
        }
        if !matches!(operand, Ty::Struct(..) | Ty::Enum(..)) {
            return;
        }
        let Some((impl_item, member, body, substitution)) =
            self.operator_impl_member(&operand, trait_name, method)
        else {
            return;
        };
        // **The signature is READ from the declaration, not assumed.** `Eq::eq` returns `Bool` and
        // `Ord::cmp` returns `Ordering`, but writing those in would be this packet's own defect —
        // a second answer to what the callable's signature is, which §3.4's invariant then has to
        // reconcile against the body's.
        let Some((receiver, params, ret)) = self.declared_member_signature(impl_item, member)
        else {
            return;
        };
        // **DEV-201: an operator on a GENERIC impl published an empty environment.**
        //
        // `Static(Vec::new())` was written here unconditionally. For `impl Eq for Point` that is
        // correct — there is nothing to bind. For `impl<T> Eq for W<T>` it is a body running with
        // `T` unbound, which is AS3 criterion 2's exact prohibition. Nothing observed it until
        // DEV-121's receiver boundary read `callable_types[body].receiver` and found
        // `&W<Param(\"T\")>` with no `T` in scope.
        let environment = self.impl_dispatch_bindings(impl_item, &operand);
        let use_ = CallableUse {
            selection: CalleeSelection::Static {
                declaration: CallableDeclId::ImplMember { impl_item, member },
                body,
            },
            environment: GenericEnvironment::Static(environment),
            // `Eq::eq(&self, &other)` and `Ord::cmp(&self, &other)` both borrow: the receiver binds
            // shared, and the call site takes a shared borrow of an owned operand — zero derefs.
            receiver_adjustment: ReceiverAdjustment::Shared { derefs: 0 },
            receiver_binding: ReceiverBinding::Shared,
            // AS3 Boundary 2 §3.4: the publication records the INSTANTIATED signature, so a
            // consumer reading it sees `&W<Int32>` rather than the declaration's `&W<T>`.
            signature: CallableSigTy {
                receiver: receiver
                    .as_ref()
                    .map(|ty| self.instantiate_ty(ty, &substitution)),
                params: params
                    .iter()
                    .map(|ty| self.instantiate_ty(ty, &substitution))
                    .collect(),
                ret: self.instantiate_ty(&ret, &substitution),
            },
            provenance: DispatchProvenance::CoreTrait { core },
        };
        self.publish_callable_use(expr_id, use_);
    }

    /// Publish a `CallableUse::Bound` for a call resolved through a generic parameter's bound.
    ///
    /// AS3 Boundary 4 step 2, deliberately landed **before** Display so the late-bound mechanism is
    /// proved on an ordinary `fn f<T: Speak>(x: T) { x.speak(); }` rather than tangled with
    /// recursive formatting.
    pub(super) fn publish_bound_use(
        &mut self,
        call_expr: ExprId,
        candidate: &BoundMethod,
        param_name: &str,
        method: &str,
        ret: &Ty,
        method_args: Vec<Ty>,
    ) {
        let (trait_, receiver_form, params) = match candidate {
            BoundMethod::User { trait_id, sig } => (
                hir::BoundTrait::User(*trait_id),
                sig.receiver,
                sig.params.iter().map(|p| p.ty).collect::<Vec<_>>(),
            ),
            BoundMethod::Core {
                core_trait, method, ..
            } => {
                // A core trait's contract is declared, not written in HIR, so its parameter types
                // are not `TypeId`s. The signature is published with the receiver and result only;
                // the specialiser produces the full instantiated signature from the impl it picks.
                let receiver_self = Ty::Param(param_name.to_string());
                let use_ = CallableUse {
                    selection: CalleeSelection::Bound {
                        trait_: hir::BoundTrait::Core(*core_trait),
                        member: method.name.to_string(),
                        self_ty: receiver_self.clone(),
                        trait_args: Vec::new(),
                        // Always empty, and that is the answer rather than a gap: a core trait's
                        // contract is `ContractTy`, which cannot declare method-level generics.
                        method_args,
                    },
                    environment: GenericEnvironment::FromBoundSelection,
                    receiver_adjustment: ReceiverAdjustment::None,
                    receiver_binding: match method.receiver {
                        Some(hir::Receiver::Value) => ReceiverBinding::ByValue,
                        Some(hir::Receiver::Ref) => ReceiverBinding::Shared,
                        Some(hir::Receiver::RefMut) => ReceiverBinding::Exclusive,
                        None => ReceiverBinding::None,
                    },
                    signature: CallableSigTy {
                        receiver: bound_receiver_ty(method.receiver.as_ref(), receiver_self),
                        params: Vec::new(),
                        ret: ret.clone(),
                    },
                    provenance: DispatchProvenance::Bound {
                        trait_: hir::BoundTrait::Core(*core_trait),
                    },
                };
                self.publish_callable_use(call_expr, use_);
                return;
            }
        };
        let receiver_self = Ty::Param(param_name.to_string());
        let params: Vec<Ty> = params
            .into_iter()
            .map(|id| self.convert_hir_type(id))
            .collect();
        let use_ = CallableUse {
            selection: CalleeSelection::Bound {
                trait_,
                member: method.to_string(),
                self_ty: receiver_self.clone(),
                trait_args: Vec::new(),
                // DEV-188: this call site's binding of the METHOD's own generics, from the
                // turbofish or inferred from the arguments.
                method_args,
            },
            environment: GenericEnvironment::FromBoundSelection,
            receiver_adjustment: ReceiverAdjustment::None,
            receiver_binding: match receiver_form {
                Some(hir::Receiver::Value) => ReceiverBinding::ByValue,
                Some(hir::Receiver::Ref) => ReceiverBinding::Shared,
                Some(hir::Receiver::RefMut) => ReceiverBinding::Exclusive,
                None => ReceiverBinding::None,
            },
            signature: CallableSigTy {
                receiver: bound_receiver_ty(receiver_form.as_ref(), receiver_self),
                params,
                ret: ret.clone(),
            },
            provenance: DispatchProvenance::Bound { trait_ },
        };
        self.publish_callable_use(call_expr, use_);
    }

    /// The impl that supplies operator trait `required` for a user nominal, and the member index
    /// and body of the method that implements it.
    ///
    /// **AS3 Boundary 3.** `ty_satisfies_operator_bound` already performs this scan — it walks every
    /// impl looking for one whose trait path reads `"Eq"`/`"Ord"` and whose self type matches — and
    /// then returns a `bool`, discarding the impl it just found. So the checker *does* select for
    /// `==` and `<`; it throws the selection away and both engines find it again.
    ///
    /// That makes this a **fourth** scan of the same shape, after `Interpreter::find_method` and
    /// `FnLowerer::find_impl_fn`. `AS0-CALLABLE-EXECUTION-SITE-INVENTORY.md` counted three because
    /// it looked for algorithms that *return* a callable; this one answers a narrower question and
    /// drops the answer, which is how it escaped the count.
    pub(super) fn operator_impl_member(
        &self,
        ty: &Ty,
        required: &str,
        method: &str,
    ) -> Option<(ItemId, u32, BlockId, HashMap<String, Ty>)> {
        for (idx, item) in self.hir.items.iter().enumerate() {
            let impl_id = ItemId(idx as u32);
            let hir::ItemKind::Impl {
                trait_: Some(trait_ref),
                self_ty,
                generics,
                items,
            } = &item.kind
            else {
                continue;
            };
            if self.item_text(impl_id, trait_ref.path.span) != required {
                continue;
            }
            // **The substitution is RETURNED, not discarded.** It was computed here to decide
            // whether the impl applies at all, and thrown away — so the operator publication had
            // no way to say what `impl<T> Eq for W<T>` binds `T` to, and published an empty
            // environment for a generic impl.
            let Some(substitution) = self.match_impl_type(
                &self.impl_self_ty_with_args(impl_id, *self_ty),
                ty,
                generics,
            ) else {
                continue;
            };
            for (member, impl_item) in items.iter().enumerate() {
                if let hir::ImplItem::Fn { def, .. } = impl_item {
                    if self.item_text(impl_id, def.sig.name) == method {
                        return Some((impl_id, member as u32, def.body, substitution));
                    }
                }
            }
        }
        None
    }

    pub(super) fn ty_satisfies_operator_bound(&self, ty: &Ty, required: &str) -> bool {
        match ty {
            // DEV-075 (owner specification decision, 2026-07-20). This gate is about the
            // OPERATOR, not the trait, and on primitives operators have built-in meaning
            // (03-Type-System, "Operators and Traits"). So primitive FLOATS keep `==` and `<`
            // here — IEEE comparison per CD-006 — even though CD-015 denies them the `Eq`/`Ord`
            // *traits*; that distinction lives in `satisfies_bound`, which gates generic bounds.
            // What DOES change: `Bool` loses ordering. `false < true` is definable, but Core v1
            // has no meaningful use for ordering truth values, and rejecting it is clearer than
            // inventing an order merely because one is technically available. `Char` is ordered,
            // by Unicode scalar value.
            Ty::Primitive(primitive) => match required {
                "Num" => is_numeric(*primitive),
                "Eq" => !matches!(primitive, Primitive::Unit),
                "Ord" => !matches!(primitive, Primitive::Unit | Primitive::Bool),
                _ => false,
            },
            Ty::Ref {
                mutable: false,
                inner,
            } if required == "Eq" || required == "Ord" => {
                let inner = self.resolve(inner);
                self.ty_satisfies_operator_bound(&inner, required)
            }
            Ty::Param(name) => self.param_declares_core_bound(name, required),
            // DEV-073 (WP-C4.7-5): a GENERIC impl satisfies a concrete instantiation's bound —
            // `impl<T> Eq for W<T>` satisfies `W<Int32>: Eq`. This used to demand
            // `types_equal(impl_self_ty, ty)`, an EXACT match, so the impl's written self type
            // `W<T>` never equalled `W<Int32>` and every operator on a generic nominal was
            // rejected E0500. The fix reuses `match_impl_type` — the same one-way unification
            // method resolution already uses for exactly this question, so operator bounds and
            // method calls now agree by construction instead of by coincidence.
            // DEV-069: the trait name written on each impl is read against that impl's own file.
            Ty::Struct(..) | Ty::Enum(..) => {
                self.hir.items.iter().enumerate().any(|(idx, item)| {
                    let impl_id = ItemId(idx as u32);
                    let hir::ItemKind::Impl {
                        trait_: Some(trait_ref),
                        self_ty,
                        generics,
                        ..
                    } = &item.kind
                    else {
                        return false;
                    };
                    self.item_text(impl_id, trait_ref.path.span) == required
                        && self
                            .match_impl_type(
                                &self.impl_self_ty_with_args(impl_id, *self_ty),
                                ty,
                                generics,
                            )
                            .is_some()
                })
            }
            Ty::Core(core_type, args) if required == "Eq" || required == "Ord" => {
                matches!(
                    core_type,
                    CoreType::Option | CoreType::Result | CoreType::Vec | CoreType::Box
                ) && args.iter().all(|arg| {
                    let arg = self.resolve(arg);
                    self.ty_satisfies_operator_bound(&arg, required)
                })
            }
            Ty::Infer(_) | Ty::Error => true,
            _ => false,
        }
    }

    /// WP-C4.7-9 audit: whether a value of this type can be given to `print`/`println` — a
    /// standard-library `Display` type, or a user nominal with its own `Display` impl.
    /// DEV-134: `?` may propagate only into a return type that can actually receive it.
    ///
    /// `03-Type-System.md` defines `?` for `Result<T, E>`/`Option<T>` and Core v1 has no
    /// user-extensible `Try` trait and no conversion step at the propagation site. The rule is
    /// therefore EXACT compatibility, deliberately and conservatively:
    ///
    /// - `Result<_, E_in>?` in a function returning `Result<_, E_out>` requires `E_in == E_out`
    ///   under `types_equal`, the compiler's canonical equivalence;
    /// - `Option<_>?` in a function returning `Option<_>` is always fine (there is no payload on
    ///   `None` to relate);
    /// - mixing the two constructors in either direction is refused.
    ///
    /// An implicit `From` conversion is NOT introduced here. The specification does not scope
    /// one, so adding it would be new semantics rather than a repair; that question is recorded
    /// separately (see the DEV-134 ledger entry). Rejection is the conservative half and is what
    /// this implements.
    pub(super) fn check_try_compatibility(&mut self, operand_ty: &Ty, ret_ty: &Ty, span: Span) {
        let operand = self.resolve(operand_ty);
        let ret = self.resolve(ret_ty);

        // Never cascade: an already-failed or still-undetermined type says nothing about `?`.
        if matches!(operand, Ty::Error) || matches!(ret, Ty::Error) {
            return;
        }
        if ty_contains_infer(&operand) || ty_contains_infer(&ret) {
            return;
        }

        let (Ty::Core(operand_ctor, operand_args), Ty::Core(ret_ctor, ret_args)) = (&operand, &ret)
        else {
            // Not a `?`-capable pair at all. The pre-existing E0006 checks in the `Try` arm
            // already reported that; adding a second diagnostic here would double-report.
            return;
        };
        if !matches!(operand_ctor, CoreType::Result | CoreType::Option)
            || !matches!(ret_ctor, CoreType::Result | CoreType::Option)
        {
            return;
        }

        // Constructor mismatch. This is the half that is easy to overlook: it produces exactly
        // the same type confusion as an error-type mismatch, because the propagated value's
        // variant tag (`None`) belongs to a different enum than the one the caller matches on.
        if operand_ctor != ret_ctor {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "'?' cannot propagate '{}' out of a function returning '{}'",
                        self.ty_to_string(&operand),
                        self.ty_to_string(&ret)
                    ),
                    span,
                )
                .with_code("E0006")
                .with_label("the propagated value and the return type are different types")
                .with_note(
                    "'?' performs no conversion in Core v1. Match on the operand and construct \
                     the returned type explicitly."
                        .to_string(),
                ),
            );
            return;
        }

        // Same constructor. Only `Result` carries an error type to relate.
        if *operand_ctor != CoreType::Result {
            return;
        }
        let (Some(err_in), Some(err_out)) = (operand_args.get(1), ret_args.get(1)) else {
            return;
        };
        let err_in = self.resolve(err_in);
        let err_out = self.resolve(err_out);
        if matches!(err_in, Ty::Error) || matches!(err_out, Ty::Error) {
            return;
        }
        if ty_contains_infer(&err_in) || ty_contains_infer(&err_out) {
            return;
        }
        if self.types_equal_inner(&err_in, &err_out, true) {
            return;
        }

        self.diags.push(
            Diagnostic::error(
                format!(
                    "'?' cannot propagate error type '{}' out of a function returning '{}'",
                    self.ty_to_string(&err_in),
                    self.ty_to_string(&ret)
                ),
                span,
            )
            .with_code("E0006")
            .with_label("error types must match exactly")
            .with_note(format!(
                "'?' performs no conversion in Core v1: it does not apply 'From', and an \
                 'impl From<{}> for {}' would not change this. Match on the operand and \
                 construct '{}' explicitly.",
                self.ty_to_string(&err_in),
                self.ty_to_string(&err_out),
                self.ty_to_string(&err_out)
            )),
        );
    }

    pub(super) fn check_tensor_builtin_call(
        &mut self,
        builtin: Builtin,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        span: Span,
    ) -> Ty {
        // AS6: the spelling table belonged to the extension, not to Core's checker — the same
        // criterion-2 shape the resolver's table had. `TensorBuiltin::op_name` is exhaustive, so a
        // new operation cannot reach here unnamed.
        let Builtin::Tensor(op) = builtin else {
            return Ty::Error;
        };
        let op_name = op.op_name();
        self.check_tensor_op(op_name, None, turbofish, args, span)
    }

    pub(super) fn check_tensor_method_call(
        &mut self,
        receiver: &Ty,
        name: &str,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        _name_span: Span,
        call_span: Span,
    ) -> Ty {
        self.check_tensor_op(name, Some(receiver), turbofish, args, call_span)
    }

    pub(super) fn check_tensor_op(
        &mut self,
        op_name: &str,
        receiver: Option<&Ty>,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        span: Span,
    ) -> Ty {
        let Some(descriptor) = TENSOR_OPS
            .iter()
            .find(|candidate| candidate.name == op_name)
        else {
            self.diags.push(Diagnostic::error(
                format!("unknown tensor operation `{op_name}`"),
                span,
            ));
            return Ty::Error;
        };
        if receiver.is_some() && !descriptor.method {
            self.diags.push(Diagnostic::error(
                format!("tensor operation `{op_name}` is not a method"),
                span,
            ));
            return Ty::Error;
        }
        if receiver.is_none() && !descriptor.standalone {
            self.diags.push(Diagnostic::error(
                format!("tensor operation `{op_name}` requires a receiver"),
                span,
            ));
            return Ty::Error;
        }

        let mut actual_ops = Vec::new();
        if let Some(r) = receiver {
            actual_ops.push(r.clone());
        }
        for arg in args {
            actual_ops.push(self.check_expr(*arg));
        }

        // AS6 packet 4B group 2C: Core's half is done — the operation is located, the call form
        // is validated, and every argument expression has been evaluated. Every dtype, shape,
        // device, schema and broadcasting decision from here on is the extension's.
        tensor_check::eval_tensor_op(
            self,
            op_name,
            descriptor,
            receiver.is_some(),
            turbofish,
            actual_ops,
            span,
        )
    }

    pub(super) fn check_model_method_call(
        &mut self,
        model: &ModelTy,
        name: &str,
        args: &[ExprId],
        name_span: Span,
        call_span: Span,
    ) -> Ty {
        // AS6 packet 4B group 2C: a model's method surface, its `.predict(...)` calling
        // convention and its result shape are model semantics; Core keeps the HIR walk, the
        // declaration scope, the freshening and the argument evaluation.
        if !tensor_check::check_model_method_name(self, name, name_span) {
            return Ty::Error;
        }

        let item = self.hir.item(model.item_id);
        let def = match &item.kind {
            hir::ItemKind::Model(def) => def,
            _ => return Ty::Error,
        };

        // Extract input and output ports
        let inputs: Vec<&hir::ModelPort> = def
            .ports
            .iter()
            .filter(|p| p.dir == crate::ast::PortDir::Input)
            .collect();
        let outputs: Vec<&hir::ModelPort> = def
            .ports
            .iter()
            .filter(|p| p.dir == crate::ast::PortDir::Output)
            .collect();

        if !tensor_check::check_model_predict_arity(self, inputs.len(), args.len(), call_span) {
            return Ty::Error;
        }

        let mut fresh_dims = HashMap::new();
        let mut fresh_dtypes = HashMap::new();
        let mut fresh_devices = HashMap::new();

        // Convert every port in one declaration scope so repeated model
        // dimensions (for example `B` across two inputs and an output) share
        // one rigid identity before the whole signature is freshened per call.
        let saved = self.enter_tensor_param_scope(&def.generics);
        let declared_inputs = inputs
            .iter()
            .map(|port| (self.convert_hir_type(port.ty), port.span))
            .collect::<Vec<_>>();
        let declared_outputs = outputs
            .iter()
            .map(|port| self.convert_hir_type(port.ty))
            .collect::<Vec<_>>();
        self.exit_tensor_param_scope(saved);

        let instantiated_inputs = declared_inputs
            .into_iter()
            .map(|(ty, port_span)| {
                (
                    self.freshen_call_ty(
                        ty,
                        &mut fresh_dims,
                        &mut fresh_dtypes,
                        &mut fresh_devices,
                        call_span,
                    ),
                    port_span,
                )
            })
            .collect::<Vec<_>>();
        let instantiated_outputs = declared_outputs
            .into_iter()
            .map(|ty| {
                self.freshen_call_ty(
                    ty,
                    &mut fresh_dims,
                    &mut fresh_dtypes,
                    &mut fresh_devices,
                    call_span,
                )
            })
            .collect::<Vec<_>>();

        // Argument evaluation stays here and stays interleaved: the extension rule runs once per
        // argument, immediately after that argument is checked, so diagnostic order is unchanged.
        for (arg_expr_id, (expected_port_ty, port_decl_span)) in
            args.iter().zip(instantiated_inputs)
        {
            let arg_ty = self.check_expr(*arg_expr_id);
            let arg_span = self.hir.expr(*arg_expr_id).span;
            let port_note = self.hir.sources.get(port_decl_span.source).map(|source| {
                let (line, column) = source.line_col(port_decl_span.lo);
                format!(
                    "corresponding model port declared at {}:{line}:{column}",
                    source.name
                )
            });
            tensor_check::check_model_predict_arg(
                self,
                arg_ty,
                expected_port_ty,
                arg_span,
                port_note,
            );
        }

        tensor_check::model_predict_result(instantiated_outputs)
    }
}

impl TypeChecker<'_> {
    /// DEV-DISPLAY-DISPATCH: check a call against the single selected bound candidate.
    ///
    /// Both arms end in the same argument-checking loop; they differ only in where the declared
    /// parameter and return types come from — an HIR signature for a user trait, the Core trait's
    /// implementation contract for a compiler-known one.
    /// Returns the return type and the method's own generic arguments at this call site.
    pub(super) fn check_bound_method_call(
        &mut self,
        candidate: &BoundMethod,
        p_name: &str,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        call_span: Span,
    ) -> (Ty, Vec<Ty>) {
        match candidate {
            BoundMethod::User { trait_id, sig } => {
                self.check_trait_member_call(*trait_id, sig, turbofish, args, call_span)
            }
            BoundMethod::Core {
                method, trait_args, ..
            } => {
                let self_ty = Ty::Param(p_name.to_string());
                let trait_arg_tys: Vec<Ty> = trait_args
                    .iter()
                    .map(|ty| self.convert_hir_type(*ty))
                    .collect();
                let params_ty: Vec<Ty> = method
                    .params
                    .iter()
                    .map(|term| self.contract_ty_to_ty(*term, &self_ty, &trait_arg_tys))
                    .collect();
                let ret_ty = match method.ret {
                    None => Ty::Primitive(Primitive::Unit),
                    Some(term) => self.contract_ty_to_ty(term, &self_ty, &trait_arg_tys),
                };
                self.check_call_arguments(params_ty, args, call_span);
                // A core trait's contract is fixed (`ContractTy`) and declares no method-level
                // generics, so this list is empty as a FACT about core traits, not as a gap.
                (ret_ty, Vec::new())
            }
        }
    }
    pub(super) fn require_operator_bound(&mut self, ty: &Ty, required: &str, span: Span) {
        let ty = self.resolve(ty);
        let satisfied = self.ty_satisfies_operator_bound(&ty, required);
        if !satisfied {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "type '{}' does not satisfy operator trait '{required}'",
                        self.ty_to_string(&ty)
                    ),
                    span,
                )
                .with_code("E0500"),
            );
        }
    }
    pub(super) fn is_mutable_place(&self, expr: ExprId) -> bool {
        let node = self.hir.expr(expr);
        match &node.kind {
            hir::ExprKind::Path {
                res: Res::Local(local) | Res::SelfValue(local),
                ..
            } => {
                self.local_mutability.get(local).copied().unwrap_or(false)
                    || matches!(
                        self.resolve(self.local_types.get(local).unwrap_or(&Ty::Error)),
                        Ty::Ref { mutable: true, .. }
                    )
            }
            hir::ExprKind::Field { base, .. }
            | hir::ExprKind::TupleField { base, .. }
            | hir::ExprKind::Index { base, .. } => self.is_mutable_place(*base),
            hir::ExprKind::Unary {
                op: UnOp::Deref,
                operand,
            } => matches!(
                self.resolve(self.expr_types.get(operand).unwrap_or(&Ty::Error)),
                Ty::Ref { mutable: true, .. }
            ),
            _ => false,
        }
    }
    pub(super) fn is_unsized_value_type(&self, ty: &Ty) -> bool {
        matches!(
            self.resolve(ty),
            Ty::Slice(_) | Ty::Primitive(Primitive::Str)
        )
    }
    /// DEV-DISPLAY-DISPATCH: every candidate the bounds on generic parameter `p_name` contribute
    /// for method `name`, from both kinds of trait, one per distinct trait identity.
    ///
    /// This is candidate COLLECTION only. Selection, ambiguity and argument checking happen once,
    /// at the call site, over whatever this returns — which is the whole point: a compiler-known
    /// bound and a user bound reach the same selection through the same list.
    /// DEV-169: refuse an explicit call to a `Drop` implementation's `drop`.
    ///
    /// 03-Type-System.md, "Copy and Drop": "`Drop::drop` MUST NOT be called explicitly; use the
    /// free function `drop(value)`." The free function is a different thing — it MOVES its
    /// argument, so the destructor still runs exactly once.
    pub(super) fn reject_explicit_drop(&mut self, impl_item: ItemId, name: &str, span: Span) {
        if name != "drop" {
            return;
        }
        let hir::ItemKind::Impl {
            trait_: Some(trait_ref),
            ..
        } = &self.hir.item(impl_item).kind
        else {
            return;
        };
        if !matches!(
            hir::resolved_bound_trait(self.hir, trait_ref),
            Some(hir::BoundTrait::Core(hir::CoreTrait::Drop))
        ) {
            return;
        }
        self.diags.push(
            Diagnostic::error("'Drop::drop' cannot be called explicitly", span)
                .with_code("E0307")
                .with_label("the destructor runs automatically when the value goes out of scope")
                .with_note(
                    "to destroy a value early, move it into the free function 'drop(value)'; \
                     calling the method here would run the destructor twice"
                        .to_string(),
                ),
        );
    }
    pub(super) fn string_types_comparable(&self, left: &Ty, right: &Ty) -> bool {
        fn is_string_like(ty: &Ty) -> bool {
            match ty {
                Ty::Primitive(Primitive::String | Primitive::Str)
                | Ty::Core(CoreType::String, _) => true,
                Ty::Ref { inner, .. } => is_string_like(inner),
                _ => false,
            }
        }
        is_string_like(&self.resolve(left)) && is_string_like(&self.resolve(right))
    }
    pub(super) fn type_is_displayable(&self, ty: &Ty) -> bool {
        if standard_display_type(ty) {
            return true;
        }
        match ty {
            // Containers print elementwise in the reference implementation.
            Ty::Core(CoreType::Option | CoreType::Result | CoreType::Vec, args) => {
                args.iter().all(|a| self.type_is_displayable(a))
            }
            Ty::Tuple(elems) => elems.iter().all(|e| self.type_is_displayable(e)),
            // **DEV-206: an array is a value; a bare slice is not.**
            //
            // These shared an arm, which accepted the UNSIZED `[T]` — a type §6.6 says is never a
            // standalone value, and which the representation relation refuses at every boundary
            // for exactly that reason. So `println(v[0..2])` type-checked and then had no valid
            // runtime representation, while `println(&v[0..2])` — the form that *can* exist — was
            // rejected because no arm below matched a reference to a slice. The polarity was
            // reversed.
            Ty::Array(elem, _) => self.type_is_displayable(elem),
            // A slice is observed THROUGH a reference. `&[T]` is displayable exactly when `T` is,
            // which is the same elementwise rule the other containers use; nothing is invented for
            // a non-`Display` element. Deliberately shared references only — `&mut [T]` is not
            // broadened here, because DEV-206 is about the `[T]`/`&[T]` contradiction and nothing
            // in the standard rules currently implies the exclusive form.
            Ty::Ref {
                mutable: false,
                inner,
            } if matches!(inner.as_ref(), Ty::Slice(_)) => match inner.as_ref() {
                Ty::Slice(elem) => self.type_is_displayable(elem),
                _ => false,
            },
            Ty::Struct(..) | Ty::Enum(..) => self.ty_satisfies_operator_bound(ty, "Display"),
            Ty::Param(_) => true, // discharged by the caller's own bound
            _ => false,
        }
    }
    pub(super) fn control_summary_stmt(&self, stmt_id: StmtId) -> ControlSummary {
        match &self.hir.stmt(stmt_id).kind {
            hir::StmtKind::Return(Some(expr)) => {
                if self.resolve(self.expr_types.get(expr).unwrap_or(&Ty::Error)) == Ty::Never {
                    ControlSummary {
                        can_complete: false,
                        may_return: false,
                    }
                } else {
                    ControlSummary {
                        can_complete: false,
                        may_return: true,
                    }
                }
            }
            hir::StmtKind::Return(None) => ControlSummary {
                can_complete: false,
                may_return: true,
            },
            hir::StmtKind::Break(_) | hir::StmtKind::Continue => ControlSummary {
                can_complete: false,
                may_return: false,
            },
            hir::StmtKind::Expr { expr, .. } => self.control_summary_expr(*expr),
            _ => ControlSummary {
                can_complete: true,
                may_return: false,
            },
        }
    }
    /// WP-C1.5: minimal constant evaluator for array-repeat-expression counts (`[value; count]`,
    /// 02-Syntax-Grammar.md:330). Handles the two confirmed-common shapes -- a literal, or a
    /// reference to a `const` item (recursing into its initializer) -- rather than a full
    /// general constant-folding pass, which is out of this WP's scope.
    pub(super) fn const_eval_u64(&self, expr_id: ExprId) -> Option<u64> {
        self.const_eval_i128(expr_id, &mut HashSet::new())
            .and_then(|value| u64::try_from(value).ok())
    }
}

impl TypeChecker<'_> {
    pub(super) fn control_summary_expr(&self, expr_id: ExprId) -> ControlSummary {
        let expr = self.hir.expr(expr_id);
        match &expr.kind {
            hir::ExprKind::If {
                then_block, else_, ..
            } => {
                let then_summary = self.control_summary_block(*then_block);
                let else_summary = else_.map_or(
                    ControlSummary {
                        can_complete: true,
                        may_return: false,
                    },
                    |expr| self.control_summary_expr(expr),
                );
                ControlSummary {
                    can_complete: then_summary.can_complete || else_summary.can_complete,
                    may_return: then_summary.may_return || else_summary.may_return,
                }
            }
            hir::ExprKind::Match { arms, .. } => ControlSummary {
                can_complete: arms
                    .iter()
                    .any(|arm| self.control_summary_expr(arm.body).can_complete),
                may_return: arms
                    .iter()
                    .any(|arm| self.control_summary_expr(arm.body).may_return),
            },
            hir::ExprKind::Block(block) => self.control_summary_block(*block),
            hir::ExprKind::Loop { body } => {
                let body_summary = self.control_summary_block(*body);
                ControlSummary {
                    can_complete: self.resolve(self.expr_types.get(&expr_id).unwrap_or(&Ty::Error))
                        != Ty::Never,
                    may_return: body_summary.may_return,
                }
            }
            hir::ExprKind::While { body, .. } | hir::ExprKind::For { body, .. } => ControlSummary {
                can_complete: true,
                may_return: self.control_summary_block(*body).may_return,
            },
            _ if self.resolve(self.expr_types.get(&expr_id).unwrap_or(&Ty::Error)) == Ty::Never => {
                ControlSummary {
                    can_complete: false,
                    may_return: false,
                }
            }
            _ => ControlSummary {
                can_complete: true,
                may_return: false,
            },
        }
    }
}

impl TypeChecker<'_> {
    pub(super) fn control_summary_block(&self, block_id: BlockId) -> ControlSummary {
        let block = self.hir.block(block_id);
        let mut summary = ControlSummary {
            can_complete: true,
            may_return: false,
        };
        for stmt in &block.stmts {
            if !summary.can_complete {
                break;
            }
            let stmt_summary = self.control_summary_stmt(*stmt);
            summary.can_complete = stmt_summary.can_complete;
            summary.may_return |= stmt_summary.may_return;
        }
        if summary.can_complete {
            if let Some(tail) = block.tail {
                let tail_summary = self.control_summary_expr(tail);
                summary.can_complete = tail_summary.can_complete;
                summary.may_return |= tail_summary.may_return;
            }
        }
        summary
    }
}
