//! AS6 packet 4B group 2C: the tensor **semantic authority**.
//!
//! Core owns the *mechanism* of checking a call — locating the operation, validating its call
//! form, evaluating argument expressions, converting written type syntax — and then hands the
//! already-typed operands to this module, which owns every dtype, shape, device, schema and
//! broadcasting *decision*. The result flows back for Core to publish.
//!
//! ```text
//! CORE       identify the tensor call, validate its FORM, evaluate argument expressions
//!               ↓ already-typed operands
//! EXTENSION  dtype / shape / device / schema / broadcasting decisions
//!               ↓ Ty
//! CORE       unify, publish, emit
//! ```
//!
//! The invariant this file exists to enforce is **one-directional semantic control**: the tensor
//! checker may consume checked expression types; it may not cause Core expression checking.
//! `check_expr` is therefore deliberately absent from [`TensorCheckCtx`], and the compiler proves
//! its absence — `TypeChecker`'s own members are private to the `typecheck` module, so nothing
//! here can reach anything the trait does not name.

use crate::ast::Primitive;
use crate::diag::Diagnostic;
use crate::extensions::tensor::dim::Poly;
use crate::extensions::tensor::rules::{
    BroadcastError, TensorDTypeRule, TensorDeviceRule, TensorGenericSchema, TensorOpDescriptor,
    TensorResultRule, TensorShapeRule,
};
use crate::extensions::tensor::types::{
    DType, Device, Shape, TensorKind, TensorTy, UnifyCtx, ValueRange,
};
use crate::hir::{self, CoreType, TypeId};
use crate::source::Span;
use crate::typecheck::{ExtensionTy, Ty};
use std::collections::HashSet;

/// The Core services the tensor semantic rules are allowed to use.
///
/// Fifteen capabilities, in four groups. Every one of them is a *capability* — it reads Core
/// state, converts written syntax, or emits — and none of them re-enters Core expression
/// checking. That is the whole point: a narrow surface is necessary but not sufficient, and the
/// property AS6 requires is the absent `check_expr`, not the member count.
pub(crate) trait TensorCheckCtx {
    // --- diagnostics -------------------------------------------------------------------------
    /// The diagnostic sink, in Core's ordering.
    fn diags(&mut self) -> &mut Vec<Diagnostic>;
    /// An `E0211` tensor diagnostic, honouring Core's speculative-checking suppression.
    fn tensor_error(&mut self, message: &str, span: Span);

    // --- Core type machinery -----------------------------------------------------------------
    fn resolve(&self, ty: &Ty) -> Ty;
    fn unify(&mut self, a: Ty, b: Ty, span: Span) -> Result<(), ()>;
    fn ty_to_string(&self, ty: &Ty) -> String;

    // --- constant and range extraction -------------------------------------------------------
    fn extract_const_int(&self, arg: &hir::GenericArg) -> Option<i64>;
    fn extract_const_int_list(&mut self, arg: &hir::GenericArg) -> Option<Vec<i64>>;
    fn extract_dim_generic(&mut self, arg: &hir::GenericArg, label: &str) -> Option<Poly>;
    fn combine_value_range(&self, a: ValueRange, b: ValueRange) -> Option<ValueRange>;
    /// The `range = R` binding of a generic argument list, if present.
    fn value_range_of(&mut self, generic_args: &hir::GenericArgs) -> ValueRange;

    // --- written tensor syntax -> tensor objects ---------------------------------------------
    // Converting written type syntax is Core machinery by nature (it walks HIR types and the
    // generic-parameter scopes), so Core keeps it and offers it as a service.
    fn build_shape(&mut self, shape: &hir::ShapeArg) -> Shape;
    fn build_refine_shape(&mut self, shape: &hir::ShapeArg) -> Shape;
    fn build_device(&mut self, arg: Option<&hir::GenericArg>, span: Span) -> Device;
    fn tensor_dtype(&mut self, ty_id: TypeId, span: Span) -> DType;

    // --- extension-owned inference state, held by the host -----------------------------------
    fn tensor_state(&mut self) -> &mut UnifyCtx;
}

/// The tensor kind behind a (possibly borrowed) operand type, or `None` if it is not a tensor.
fn tensor_kind_of(cx: &dyn TensorCheckCtx, ty: &Ty) -> Option<TensorKind> {
    let resolved = cx.resolve(ty);
    let tensor_ty = match resolved {
        Ty::Ref { inner, .. } => cx.resolve(&inner),
        other => other,
    };
    match tensor_ty {
        Ty::Extension(ext) => match &*ext {
            ExtensionTy::Tensor(kind) => Some(kind.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// The Core value type a tensor element of `dtype` produces.
fn dtype_to_ty(dtype: DType) -> Ty {
    match dtype {
        DType::Int8 => Ty::Primitive(Primitive::Int8),
        DType::Int16 => Ty::Primitive(Primitive::Int16),
        DType::Int32 => Ty::Primitive(Primitive::Int32),
        DType::Int64 => Ty::Primitive(Primitive::Int64),
        DType::UInt8 => Ty::Primitive(Primitive::UInt8),
        DType::UInt16 => Ty::Primitive(Primitive::UInt16),
        DType::UInt32 => Ty::Primitive(Primitive::UInt32),
        DType::UInt64 => Ty::Primitive(Primitive::UInt64),
        DType::Float16 => Ty::Primitive(Primitive::Float16),
        DType::Float32 => Ty::Primitive(Primitive::Float32),
        DType::Float64 => Ty::Primitive(Primitive::Float64),
        DType::BFloat16 => Ty::Primitive(Primitive::BFloat16),
        DType::Bool => Ty::Primitive(Primitive::Bool),
        DType::Var(_) => Ty::Error,
    }
}

/// Right-aligned NumPy-style broadcast of two shapes. On success returns the result shape; on
/// failure returns the result-aligned axis at which the two dimensions are neither provably equal
/// nor a literal `1`.
fn broadcast_shapes(
    cx: &mut dyn TensorCheckCtx,
    sa: &Shape,
    sb: &Shape,
    span: Span,
) -> Result<Shape, usize> {
    let rank_a = sa.rank();
    let rank_b = sb.rank();
    let rank_out = std::cmp::max(rank_a, rank_b);
    let mut dims_out = Vec::with_capacity(rank_out);
    let mut spans_out = Vec::with_capacity(rank_out);

    for trailing in 0..rank_out {
        let index_a = rank_a.checked_sub(trailing + 1);
        let index_b = rank_b.checked_sub(trailing + 1);
        let dim_a = index_a.map(|index| &sa.dims[index]);
        let dim_b = index_b.map(|index| &sb.dims[index]);
        let span_a = index_a
            .and_then(|index| sa.spans.get(index).copied())
            .unwrap_or(span);
        let span_b = index_b
            .and_then(|index| sb.spans.get(index).copied())
            .unwrap_or(span);

        match (dim_a, dim_b) {
            (Some(da), Some(db)) => {
                let resolved_a = cx
                    .tensor_state()
                    .resolve_dim(da)
                    .unwrap_or_else(|_| da.clone());
                let resolved_b = cx
                    .tensor_state()
                    .resolve_dim(db)
                    .unwrap_or_else(|_| db.clone());

                if resolved_a == resolved_b {
                    dims_out.push(resolved_a);
                    spans_out.push(span_a);
                } else if resolved_a.as_constant() == Some(1) {
                    dims_out.push(resolved_b);
                    spans_out.push(span_b);
                } else if resolved_b.as_constant() == Some(1) {
                    dims_out.push(resolved_a);
                    spans_out.push(span_a);
                } else {
                    // Broadcasting is proof-based: unrelated variables do not become equal merely
                    // because an operation wants them to. Only equality already established by the
                    // surrounding type constraints is accepted here. Report the axis aligned to
                    // the result shape.
                    return Err(rank_out - 1 - trailing);
                }
            }
            (Some(da), None) => {
                dims_out.push(da.clone());
                spans_out.push(span_a);
            }
            (None, Some(db)) => {
                dims_out.push(db.clone());
                spans_out.push(span_b);
            }
            (None, None) => unreachable!(),
        }
    }

    dims_out.reverse();
    spans_out.reverse();
    Ok(Shape::with_spans(dims_out, spans_out))
}

/// Whether `source` can be explicitly broadcast to `target`. On failure distinguishes a rank
/// mismatch from a specific target-aligned axis that cannot be expanded.
fn broadcast_to_check(
    cx: &mut dyn TensorCheckCtx,
    source: &Shape,
    target: &Shape,
) -> Result<(), BroadcastError> {
    if source.rank() > target.rank() {
        return Err(BroadcastError::Rank {
            source: source.rank(),
            target: target.rank(),
        });
    }
    for trailing in 0..source.rank() {
        let source_index = source.rank() - 1 - trailing;
        let target_index = target.rank() - 1 - trailing;
        let source_dim = cx
            .tensor_state()
            .resolve_dim(&source.dims[source_index])
            .unwrap_or_else(|_| source.dims[source_index].clone());
        let target_dim = cx
            .tensor_state()
            .resolve_dim(&target.dims[target_index])
            .unwrap_or_else(|_| target.dims[target_index].clone());
        if source_dim != target_dim && source_dim.as_constant() != Some(1) {
            return Err(BroadcastError::Axis {
                result_axis: target_index,
            });
        }
    }
    Ok(())
}

/// Boolean form for callers that only need the yes/no answer (e.g. fix suggestions).
fn can_broadcast_to(cx: &mut dyn TensorCheckCtx, source: &Shape, target: &Shape) -> bool {
    broadcast_to_check(cx, source, target).is_ok()
}

fn shape_volume(cx: &mut dyn TensorCheckCtx, shape: &Shape) -> Result<Poly, ()> {
    let mut volume = Poly::constant(1);
    for dimension in &shape.dims {
        let resolved = cx.tensor_state().resolve_dim(dimension).map_err(|_| ())?;
        volume = volume.mul(&resolved).map_err(|_| ())?;
    }
    Ok(volume)
}

/// The `.cast`/`.to_device`/`.broadcast_to` repair a mismatched pair of tensor types would need.
fn get_fix_suggestion(
    cx: &mut dyn TensorCheckCtx,
    expected: &TensorKind,
    found: &TensorKind,
) -> Option<String> {
    let (TensorKind::Tensor(expected), TensorKind::Tensor(found)) = (expected, found) else {
        return None;
    };
    let dtype_differs = match (expected.dtype, found.dtype) {
        (DType::Var(_), _) | (_, DType::Var(_)) => false,
        (left, right) => left != right,
    };
    let device_differs = match (expected.device, found.device) {
        (Device::Var(_), _) | (_, Device::Var(_)) => false,
        (left, right) => left != right,
    };
    let shape_differs = expected.shape.dims != found.shape.dims;

    match (dtype_differs, device_differs, shape_differs) {
        (true, false, false) => Some(format!(
            "cast the second tensor with `.cast::<{}>()`",
            expected.dtype.name()
        )),
        (false, true, false) => Some(format!(
            "move the second tensor with `.to_device::<{}>()`",
            expected.device
        )),
        (false, false, true) if can_broadcast_to(cx, &found.shape, &expected.shape) => {
            let target = cx.tensor_state().display_shape(&expected.shape);
            Some(format!(
                "broadcast the second tensor with `.broadcast_to::<{target}>()`"
            ))
        }
        _ => None,
    }
}

/// Evaluate a tensor operation's dtype/shape/device/schema rules over already-typed operands.
///
/// Core has, before this point: located `descriptor`, rejected an unknown operation, rejected a
/// standalone/method form mismatch, and evaluated every argument expression. `actual_ops` holds
/// the receiver (when there is one) followed by the argument types, in source order.
pub(crate) fn eval_tensor_op(
    cx: &mut dyn TensorCheckCtx,
    op_name: &str,
    descriptor: &TensorOpDescriptor,
    has_receiver: bool,
    turbofish: Option<&hir::GenericArgs>,
    actual_ops: Vec<Ty>,
    span: Span,
) -> Ty {
    if actual_ops.len() != descriptor.arity {
        cx.diags().push(
            Diagnostic::error(
                format!(
                    "wrong number of arguments to `{op_name}`: expected {}, found {}",
                    descriptor.arity,
                    actual_ops.len()
                ),
                span,
            )
            .with_code("E0005"),
        );
        return Ty::Error;
    }

    let generic_arity = turbofish.map_or(0, |generic_args| generic_args.args.len());
    if generic_arity != descriptor.generics.arity() {
        cx.diags().push(
            Diagnostic::error(
                format!(
                    "wrong number of generic arguments to `{op_name}`: expected {}, found {generic_arity}",
                    descriptor.generics.arity()
                ),
                turbofish.map_or(span, |generic_args| generic_args.span),
            )
            .with_code("E0213"),
        );
        return Ty::Error;
    }

    debug_assert!(match descriptor.device {
        TensorDeviceRule::Fresh => matches!(
            descriptor.shape,
            TensorShapeRule::Construct | TensorShapeRule::FromVec
        ),
        TensorDeviceRule::Match => descriptor.arity == 2,
        TensorDeviceRule::Preserve | TensorDeviceRule::Target => descriptor.arity == 1,
    });
    debug_assert!(match descriptor.dtype {
        TensorDTypeRule::Construct => matches!(
            descriptor.generics,
            TensorGenericSchema::DTypeAndShape | TensorGenericSchema::DTypeAndDim
        ),
        TensorDTypeRule::Cast => descriptor.generics == TensorGenericSchema::DType,
        TensorDTypeRule::ArgMax
        | TensorDTypeRule::Compare
        | TensorDTypeRule::Match
        | TensorDTypeRule::Preserve => true,
    });

    if !matches!(
        descriptor.shape,
        TensorShapeRule::Construct | TensorShapeRule::FromVec
    ) {
        for (index, operand) in actual_ops.iter().enumerate() {
            if has_receiver && index == 0 {
                continue;
            }
            if !matches!(cx.resolve(operand), Ty::Ref { mutable: false, .. }) {
                cx.diags().push(
                    Diagnostic::error(
                        format!(
                            "tensor operand {} of `{op_name}` must be borrowed (for example `&tensor`)",
                            index + 1
                        ),
                        span,
                    )
                    .with_code("E0005"),
                );
                return Ty::Error;
            }
        }
    }

    match descriptor.shape {
        TensorShapeRule::Construct => {
            let g_args = match turbofish {
                Some(g) => g,
                None => {
                    cx.diags().push(
                        Diagnostic::error(
                            format!(
                                "`{}` requires explicit type and shape generic arguments",
                                op_name
                            ),
                            span,
                        )
                        .with_code("E0213"),
                    );
                    return Ty::Error;
                }
            };
            if g_args.args.len() != 2 {
                cx.diags().push(
                    Diagnostic::error(
                        format!(
                            "`{}` expects 2 generic arguments, found {}",
                            op_name,
                            g_args.args.len()
                        ),
                        g_args.span,
                    )
                    .with_code("E0213"),
                );
                return Ty::Error;
            }
            let dtype = match &g_args.args[0] {
                hir::GenericArg::Type(t) => cx.tensor_dtype(*t, g_args.span),
                _ => {
                    cx.diags().push(Diagnostic::error(
                        "first generic argument must be a type",
                        g_args.span,
                    ));
                    DType::Float32
                }
            };
            let shape = match &g_args.args[1] {
                hir::GenericArg::Shape(s) => cx.build_shape(s),
                _ => {
                    cx.diags().push(Diagnostic::error(
                        "second generic argument must be a shape",
                        g_args.span,
                    ));
                    Shape::default()
                }
            };

            if op_name == "full" {
                let val_ty = actual_ops[0].clone();
                let expected_val_ty = dtype_to_ty(dtype);
                let _ = cx.unify(expected_val_ty, val_ty, span);
            }

            Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                TensorTy {
                    dtype,
                    shape,
                    device: cx.tensor_state().fresh_device(),
                    range: ValueRange::Unspecified,
                },
            ))))
        }
        TensorShapeRule::FromVec => {
            let g_args = match turbofish {
                Some(g) => g,
                None => {
                    cx.diags().push(
                        Diagnostic::error(
                            "`from_vec` requires explicit type and dimension generic arguments",
                            span,
                        )
                        .with_code("E0213"),
                    );
                    return Ty::Error;
                }
            };
            if g_args.args.len() != 2 {
                cx.diags().push(
                    Diagnostic::error(
                        format!(
                            "`from_vec` expects 2 generic arguments, found {}",
                            g_args.args.len()
                        ),
                        g_args.span,
                    )
                    .with_code("E0213"),
                );
                return Ty::Error;
            }
            let dtype = match &g_args.args[0] {
                hir::GenericArg::Type(t) => cx.tensor_dtype(*t, g_args.span),
                _ => {
                    cx.diags().push(Diagnostic::error(
                        "first generic argument must be a type",
                        g_args.span,
                    ));
                    DType::Float32
                }
            };
            let dim_poly = match &g_args.args[1] {
                hir::GenericArg::Shape(s) => {
                    let shape = cx.build_shape(s);
                    if shape.dims.len() != 1 {
                        cx.diags().push(Diagnostic::error(
                            "from_vec dimension argument must have rank 1",
                            s.span,
                        ));
                        Poly::constant(1)
                    } else {
                        shape.dims[0].clone()
                    }
                }
                _ => Poly::constant(1),
            };

            let val_ty = actual_ops[0].clone();
            let expected_val_ty = Ty::Core(CoreType::Vec, vec![dtype_to_ty(dtype)]);
            let _ = cx.unify(expected_val_ty, val_ty, span);

            let tensor = Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                TensorTy {
                    dtype,
                    shape: Shape::new(vec![dim_poly]),
                    device: cx.tensor_state().fresh_device(),
                    range: ValueRange::Unspecified,
                },
            ))));
            Ty::Core(CoreType::Result, vec![tensor, Ty::Error])
        }
        TensorShapeRule::Elementwise => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("first argument must be a tensor", span));
                return Ty::Error;
            };
            let Some(kb) = tensor_kind_of(cx, &actual_ops[1]) else {
                cx.diags()
                    .push(Diagnostic::error("second argument must be a tensor", span));
                return Ty::Error;
            };

            match (&ka, &kb) {
                (TensorKind::Tensor(ta), TensorKind::Tensor(tb)) => {
                    if cx.tensor_state().unify_dtype(ta.dtype, tb.dtype).is_err() {
                        let mut diag = Diagnostic::error(
                            format!(
                                "tensor element type mismatch: expected `{}`, found `{}`",
                                ta.dtype.name(),
                                tb.dtype.name()
                            ),
                            span,
                        )
                        .with_code("E0212");
                        if let Some(fix) = get_fix_suggestion(cx, &ka, &kb) {
                            diag = diag.with_note(fix);
                        }
                        cx.diags().push(diag);
                        return Ty::Error;
                    }
                    if cx
                        .tensor_state()
                        .unify_device(ta.device, tb.device)
                        .is_err()
                    {
                        let mut diag = Diagnostic::error(
                            format!(
                                "tensor device mismatch: expected `{:?}`, found `{:?}`",
                                ta.device, tb.device
                            ),
                            span,
                        )
                        .with_code("E0212");
                        if let Some(fix) = get_fix_suggestion(cx, &ka, &kb) {
                            diag = diag.with_note(fix);
                        }
                        cx.diags().push(diag);
                        return Ty::Error;
                    }
                    let out_shape = match broadcast_shapes(cx, &ta.shape, &tb.shape, span) {
                        Ok(s) => s,
                        Err(result_axis) => {
                            let lhs = cx.tensor_state().display_shape(&ta.shape);
                            let rhs = cx.tensor_state().display_shape(&tb.shape);
                            let mut diag = Diagnostic::error(
                                "tensor shapes cannot be broadcast together",
                                span,
                            )
                            .with_code("E0212")
                            .with_note(format!("left shape: {lhs}"))
                            .with_note(format!("right shape: {rhs}"))
                            .with_note(format!(
                                "axis {result_axis} (aligned to the result) is neither equal nor `1`"
                            ));
                            for origin in
                                cx.tensor_state().dim_origin_notes(&[&ta.shape, &tb.shape])
                            {
                                diag = diag.with_note(origin);
                            }
                            if let Some(fix) = get_fix_suggestion(cx, &ka, &kb) {
                                diag = diag.with_note(fix);
                            }
                            cx.diags().push(diag);
                            return Ty::Error;
                        }
                    };

                    let out_dtype = if descriptor.result == TensorResultRule::BoolTensor {
                        DType::Bool
                    } else {
                        ta.dtype
                    };

                    // Elementwise ops must not merge incompatible value-range
                    // states. An `Unspecified` operand is neutral (a bare
                    // constant); two different *specified* ranges are an error.
                    let out_range = match cx.combine_value_range(ta.range, tb.range) {
                        Some(r) => r,
                        None => {
                            cx.diags().push(
                                Diagnostic::error(
                                    format!(
                                        "`{}` cannot merge tensors with value ranges `{}` and `{}`",
                                        descriptor.name, ta.range, tb.range
                                    ),
                                    span,
                                )
                                .with_code("E0212"),
                            );
                            return Ty::Error;
                        }
                    };

                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: out_dtype,
                            shape: out_shape,
                            device: ta.device,
                            range: out_range,
                        },
                    ))))
                }
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::BroadcastTo => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("argument must be a tensor", span));
                return Ty::Error;
            };
            let g_args = match turbofish {
                Some(g) => g,
                None => {
                    cx.diags().push(
                        Diagnostic::error(
                            "`broadcast_to` requires explicit shape generic argument",
                            span,
                        )
                        .with_code("E0213"),
                    );
                    return Ty::Error;
                }
            };
            if g_args.args.len() != 1 {
                cx.diags().push(Diagnostic::error(
                    "`broadcast_to` expects exactly 1 generic argument",
                    g_args.span,
                ));
                return Ty::Error;
            }
            let target_shape = match &g_args.args[0] {
                hir::GenericArg::Shape(s) => cx.build_shape(s),
                _ => {
                    cx.diags().push(Diagnostic::error(
                        "generic argument must be a shape",
                        g_args.span,
                    ));
                    Shape::default()
                }
            };

            match &ka {
                TensorKind::Tensor(t) => {
                    if let Err(err) = broadcast_to_check(cx, &t.shape, &target_shape) {
                        let source = cx.tensor_state().display_shape(&t.shape);
                        let target = cx.tensor_state().display_shape(&target_shape);
                        let mut diag =
                            Diagnostic::error("cannot `broadcast_to` the target shape", span)
                                .with_code("E0212")
                                .with_note(format!("source shape: {source}"))
                                .with_note(format!("target shape: {target}"));
                        diag = match err {
                            BroadcastError::Rank {
                                source: s,
                                target: t,
                            } => diag.with_note(format!(
                                "rank mismatch: source rank {s} exceeds target rank {t}"
                            )),
                            BroadcastError::Axis { result_axis } => diag.with_note(format!(
                                "axis {result_axis} (aligned to the result) is neither equal nor `1`"
                            )),
                        };
                        for origin in cx
                            .tensor_state()
                            .dim_origin_notes(&[&t.shape, &target_shape])
                        {
                            diag = diag.with_note(origin);
                        }
                        cx.diags().push(diag);
                        return Ty::Error;
                    }
                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: t.dtype,
                            shape: target_shape,
                            device: t.device,
                            range: t.range,
                        },
                    ))))
                }
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::MatMul => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("first argument must be a tensor", span));
                return Ty::Error;
            };
            let Some(kb) = tensor_kind_of(cx, &actual_ops[1]) else {
                cx.diags()
                    .push(Diagnostic::error("second argument must be a tensor", span));
                return Ty::Error;
            };

            match (&ka, &kb) {
                (TensorKind::Tensor(ta), TensorKind::Tensor(tb)) => {
                    if ta.shape.rank() != 2 {
                        cx.diags().push(Diagnostic::error(
                            format!(
                                "matmul first argument must be rank 2, found rank {}",
                                ta.shape.rank()
                            ),
                            span,
                        ));
                        return Ty::Error;
                    }
                    if tb.shape.rank() != 2 {
                        cx.diags().push(Diagnostic::error(
                            format!(
                                "matmul second argument must be rank 2, found rank {}",
                                tb.shape.rank()
                            ),
                            span,
                        ));
                        return Ty::Error;
                    }
                    if cx.tensor_state().unify_dtype(ta.dtype, tb.dtype).is_err() {
                        cx.diags()
                            .push(Diagnostic::error("matmul dtype mismatch", span));
                        return Ty::Error;
                    }
                    if cx
                        .tensor_state()
                        .unify_device(ta.device, tb.device)
                        .is_err()
                    {
                        cx.diags()
                            .push(Diagnostic::error("matmul device mismatch", span));
                        return Ty::Error;
                    }
                    if cx
                        .tensor_state()
                        .unify_dim(&ta.shape.dims[1], &tb.shape.dims[0], 0)
                        .is_err()
                    {
                        let lhs = cx.tensor_state().display_dim(&ta.shape.dims[1]);
                        let rhs = cx.tensor_state().display_dim(&tb.shape.dims[0]);
                        cx.diags().push(
                            Diagnostic::error(
                                format!("matmul inner dimensions mismatch: `{lhs}` and `{rhs}`"),
                                span,
                            )
                            .with_code("E0212"),
                        );
                        return Ty::Error;
                    }

                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: ta.dtype,
                            shape: Shape::new(vec![
                                ta.shape.dims[0].clone(),
                                tb.shape.dims[1].clone(),
                            ]),
                            device: ta.device,
                            // matmul mixes values across the contracted
                            // axis, so any input value range is no longer
                            // meaningful: the result is Unspecified.
                            range: ValueRange::Unspecified,
                        },
                    ))))
                }
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::BatchMatMul => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("first argument must be a tensor", span));
                return Ty::Error;
            };
            let Some(kb) = tensor_kind_of(cx, &actual_ops[1]) else {
                cx.diags()
                    .push(Diagnostic::error("second argument must be a tensor", span));
                return Ty::Error;
            };

            match (&ka, &kb) {
                (TensorKind::Tensor(ta), TensorKind::Tensor(tb)) => {
                    if ta.shape.rank() != 3 {
                        cx.diags().push(Diagnostic::error(
                            format!(
                                "batch_matmul first argument must be rank 3, found rank {}",
                                ta.shape.rank()
                            ),
                            span,
                        ));
                        return Ty::Error;
                    }
                    if tb.shape.rank() != 3 {
                        cx.diags().push(Diagnostic::error(
                            format!(
                                "batch_matmul second argument must be rank 3, found rank {}",
                                tb.shape.rank()
                            ),
                            span,
                        ));
                        return Ty::Error;
                    }
                    if cx.tensor_state().unify_dtype(ta.dtype, tb.dtype).is_err() {
                        cx.diags()
                            .push(Diagnostic::error("batch_matmul dtype mismatch", span));
                        return Ty::Error;
                    }
                    if cx
                        .tensor_state()
                        .unify_device(ta.device, tb.device)
                        .is_err()
                    {
                        cx.diags()
                            .push(Diagnostic::error("batch_matmul device mismatch", span));
                        return Ty::Error;
                    }
                    if cx
                        .tensor_state()
                        .unify_dim(&ta.shape.dims[0], &tb.shape.dims[0], 0)
                        .is_err()
                    {
                        let lhs = cx.tensor_state().display_dim(&ta.shape.dims[0]);
                        let rhs = cx.tensor_state().display_dim(&tb.shape.dims[0]);
                        cx.diags().push(
                            Diagnostic::error(
                                format!(
                                    "batch_matmul batch dimension mismatch: `{lhs}` and `{rhs}`"
                                ),
                                span,
                            )
                            .with_code("E0212"),
                        );
                        return Ty::Error;
                    }
                    if cx
                        .tensor_state()
                        .unify_dim(&ta.shape.dims[2], &tb.shape.dims[1], 1)
                        .is_err()
                    {
                        let lhs = cx.tensor_state().display_dim(&ta.shape.dims[2]);
                        let rhs = cx.tensor_state().display_dim(&tb.shape.dims[1]);
                        cx.diags().push(
                            Diagnostic::error(
                                format!(
                                    "batch_matmul inner dimensions mismatch: `{lhs}` and `{rhs}`"
                                ),
                                span,
                            )
                            .with_code("E0212"),
                        );
                        return Ty::Error;
                    }

                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: ta.dtype,
                            shape: Shape::new(vec![
                                ta.shape.dims[0].clone(),
                                ta.shape.dims[1].clone(),
                                tb.shape.dims[2].clone(),
                            ]),
                            device: ta.device,
                            // See matmul: the contracted product is not a
                            // value-range-preserving operation.
                            range: ValueRange::Unspecified,
                        },
                    ))))
                }
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::Concat => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("first argument must be a tensor", span));
                return Ty::Error;
            };
            let Some(kb) = tensor_kind_of(cx, &actual_ops[1]) else {
                cx.diags()
                    .push(Diagnostic::error("second argument must be a tensor", span));
                return Ty::Error;
            };

            let g_args = match turbofish {
                Some(g) => g,
                None => {
                    cx.diags().push(
                        Diagnostic::error("concat requires explicit axis generic argument", span)
                            .with_code("E0213"),
                    );
                    return Ty::Error;
                }
            };
            if g_args.args.len() != 1 {
                cx.diags().push(Diagnostic::error(
                    "concat expects exactly 1 generic argument",
                    g_args.span,
                ));
                return Ty::Error;
            }
            let axis = match cx.extract_const_int(&g_args.args[0]) {
                Some(a) => a,
                None => {
                    cx.diags().push(Diagnostic::error(
                        "concat axis must be a constant integer",
                        g_args.span,
                    ));
                    return Ty::Error;
                }
            };

            match (&ka, &kb) {
                (TensorKind::Tensor(ta), TensorKind::Tensor(tb)) => {
                    let rank = ta.shape.rank();
                    if tb.shape.rank() != rank {
                        cx.diags().push(Diagnostic::error(
                            "concat tensors must have equal rank",
                            span,
                        ));
                        return Ty::Error;
                    }
                    if axis < 0 || axis >= rank as i64 {
                        cx.diags().push(Diagnostic::error(
                            format!("concat axis {} is out of range for rank {}", axis, rank),
                            g_args.span,
                        ));
                        return Ty::Error;
                    }
                    if cx.tensor_state().unify_dtype(ta.dtype, tb.dtype).is_err() {
                        cx.diags()
                            .push(Diagnostic::error("concat dtype mismatch", span));
                        return Ty::Error;
                    }
                    if cx
                        .tensor_state()
                        .unify_device(ta.device, tb.device)
                        .is_err()
                    {
                        cx.diags()
                            .push(Diagnostic::error("concat device mismatch", span));
                        return Ty::Error;
                    }
                    let mut out_dims = Vec::new();
                    for i in 0..rank {
                        if i as i64 == axis {
                            let sum_dim = match ta.shape.dims[i].add(&tb.shape.dims[i]) {
                                Ok(d) => d,
                                Err(_) => {
                                    cx.diags()
                                        .push(Diagnostic::error("concat dimension overflow", span));
                                    return Ty::Error;
                                }
                            };
                            out_dims.push(sum_dim);
                        } else {
                            if cx
                                .tensor_state()
                                .unify_dim(&ta.shape.dims[i], &tb.shape.dims[i], i)
                                .is_err()
                            {
                                cx.diags().push(Diagnostic::error(
                                    format!(
                                        "concat dimension mismatch at axis {}: {} and {}",
                                        i, ta.shape.dims[i], tb.shape.dims[i]
                                    ),
                                    span,
                                ));
                                return Ty::Error;
                            }
                            out_dims.push(ta.shape.dims[i].clone());
                        }
                    }

                    // Concat joins two tensors, so their value ranges must
                    // combine like an elementwise op (Unspecified neutral).
                    let out_range = match cx.combine_value_range(ta.range, tb.range) {
                        Some(r) => r,
                        None => {
                            cx.diags().push(
                                Diagnostic::error(
                                    format!(
                                        "`concat` cannot merge tensors with value ranges `{}` and `{}`",
                                        ta.range, tb.range
                                    ),
                                    span,
                                )
                                .with_code("E0212"),
                            );
                            return Ty::Error;
                        }
                    };
                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: ta.dtype,
                            shape: Shape::new(out_dims),
                            device: ta.device,
                            range: out_range,
                        },
                    ))))
                }
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::Permute => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("receiver must be a tensor", span));
                return Ty::Error;
            };
            let g_args = match turbofish {
                Some(g) => g,
                None => {
                    cx.diags().push(
                        Diagnostic::error("permute requires explicit target index list", span)
                            .with_code("E0213"),
                    );
                    return Ty::Error;
                }
            };
            if g_args.args.len() != 1 {
                cx.diags().push(Diagnostic::error(
                    "permute expects exactly 1 generic argument",
                    g_args.span,
                ));
                return Ty::Error;
            }
            let permutation = match cx.extract_const_int_list(&g_args.args[0]) {
                Some(p) => p,
                None => {
                    cx.diags().push(Diagnostic::error(
                        "permute argument must be a constant integer list",
                        g_args.span,
                    ));
                    return Ty::Error;
                }
            };

            match &ka {
                TensorKind::Tensor(t) => {
                    let rank = t.shape.rank();
                    if permutation.len() != rank {
                        cx.diags().push(Diagnostic::error(
                            format!(
                                "permute length mismatch: expected list of length {}, found {}",
                                rank,
                                permutation.len()
                            ),
                            g_args.span,
                        ));
                        return Ty::Error;
                    }
                    let mut seen = HashSet::new();
                    for &idx in &permutation {
                        if idx < 0 || idx >= rank as i64 {
                            cx.diags().push(Diagnostic::error(
                                format!("index {} is out of range for rank {}", idx, rank),
                                g_args.span,
                            ));
                            return Ty::Error;
                        }
                        if !seen.insert(idx) {
                            cx.diags().push(Diagnostic::error(
                                format!("duplicate index {} in permute list", idx),
                                g_args.span,
                            ));
                            return Ty::Error;
                        }
                    }

                    let mut out_dims = Vec::new();
                    for &idx in &permutation {
                        out_dims.push(t.shape.dims[idx as usize].clone());
                    }

                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: t.dtype,
                            shape: Shape::new(out_dims),
                            device: t.device,
                            range: t.range,
                        },
                    ))))
                }
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::Reshape => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("receiver must be a tensor", span));
                return Ty::Error;
            };
            let g_args = match turbofish {
                Some(g) => g,
                None => {
                    cx.diags().push(
                        Diagnostic::error("reshape requires explicit target shape", span)
                            .with_code("E0213"),
                    );
                    return Ty::Error;
                }
            };
            if g_args.args.len() != 1 {
                cx.diags().push(Diagnostic::error(
                    "reshape expects exactly 1 generic argument",
                    g_args.span,
                ));
                return Ty::Error;
            }
            let target_shape = match &g_args.args[0] {
                hir::GenericArg::Shape(s) => cx.build_shape(s),
                _ => {
                    cx.diags().push(Diagnostic::error(
                        "generic argument must be a shape",
                        g_args.span,
                    ));
                    Shape::default()
                }
            };

            match &ka {
                TensorKind::Tensor(t) => {
                    let (source_volume, target_volume) =
                        match (shape_volume(cx, &t.shape), shape_volume(cx, &target_shape)) {
                            (Ok(source), Ok(target)) => (source, target),
                            _ => {
                                cx.diags().push(
                                    Diagnostic::error(
                                        "reshape element-count calculation overflowed",
                                        span,
                                    )
                                    .with_code("E0212"),
                                );
                                return Ty::Error;
                            }
                        };
                    if source_volume != target_volume {
                        let source_shape = cx.tensor_state().display_shape(&t.shape);
                        let target_display = cx.tensor_state().display_shape(&target_shape);
                        let source_product = cx.tensor_state().shape_product_display(&t.shape);
                        let target_product = cx.tensor_state().shape_product_display(&target_shape);
                        let mut diag =
                            Diagnostic::error("reshape cannot preserve element count", span)
                                .with_code("E0212")
                                .with_note(format!("source shape: {source_shape}"))
                                .with_note(format!("target shape: {target_display}"))
                                .with_note(format!(
                                    "required: {source_product} == {target_product}"
                                ));
                        for origin in cx
                            .tensor_state()
                            .dim_origin_notes(&[&t.shape, &target_shape])
                        {
                            diag = diag.with_note(origin);
                        }
                        cx.diags().push(diag);
                        return Ty::Error;
                    }
                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: t.dtype,
                            shape: target_shape,
                            device: t.device,
                            range: t.range,
                        },
                    ))))
                }
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::SliceAxis => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("receiver must be a tensor", span));
                return Ty::Error;
            };
            let g_args = match turbofish {
                Some(g) => g,
                None => {
                    cx.diags().push(
                        Diagnostic::error(
                            "slice_axis requires AXIS, START, LEN generic arguments",
                            span,
                        )
                        .with_code("E0213"),
                    );
                    return Ty::Error;
                }
            };
            if g_args.args.len() != 3 {
                cx.diags().push(
                    Diagnostic::error(
                        format!(
                            "slice_axis expects 3 generic arguments, found {}",
                            g_args.args.len()
                        ),
                        g_args.span,
                    )
                    .with_code("E0213"),
                );
                return Ty::Error;
            }
            let axis = match cx.extract_const_int(&g_args.args[0]) {
                Some(a) => a,
                None => {
                    cx.diags().push(Diagnostic::error(
                        "AXIS must be a constant integer",
                        g_args.span,
                    ));
                    return Ty::Error;
                }
            };
            let Some(start) = cx.extract_dim_generic(&g_args.args[1], "START") else {
                return Ty::Error;
            };
            let Some(len) = cx.extract_dim_generic(&g_args.args[2], "LEN") else {
                return Ty::Error;
            };

            match &ka {
                TensorKind::Tensor(t) => {
                    let rank = t.shape.rank();
                    if axis < 0 || axis >= rank as i64 {
                        cx.diags().push(Diagnostic::error(
                            format!("axis {} out of range for rank {}", axis, rank),
                            g_args.span,
                        ));
                        return Ty::Error;
                    }
                    let axis_len = cx
                        .tensor_state()
                        .resolve_dim(&t.shape.dims[axis as usize])
                        .unwrap_or_else(|_| t.shape.dims[axis as usize].clone());
                    let start = cx.tensor_state().resolve_dim(&start).unwrap_or(start);
                    let len = cx.tensor_state().resolve_dim(&len).unwrap_or(len);
                    let end = match start.add(&len) {
                        Ok(end) => end,
                        Err(_) => {
                            cx.diags().push(
                                Diagnostic::error(
                                    "slice dimension arithmetic overflowed",
                                    g_args.span,
                                )
                                .with_code("E0212"),
                            );
                            return Ty::Error;
                        }
                    };
                    let exact = end == axis_len;
                    let literal_within_bounds = match (
                        start.as_constant(),
                        len.as_constant(),
                        axis_len.as_constant(),
                        end.as_constant(),
                    ) {
                        (Some(start), Some(len), Some(axis_len), Some(end)) => {
                            start >= 0 && len >= 0 && end <= axis_len
                        }
                        _ => false,
                    };
                    if !exact && !literal_within_bounds {
                        cx.diags().push(
                            Diagnostic::error(
                                format!(
                                    "cannot prove slice constraint `{start} + {len} == {axis_len}`"
                                ),
                                g_args.span,
                            )
                            .with_code("E0212"),
                        );
                        return Ty::Error;
                    }

                    let mut out_dims = t.shape.dims.clone();
                    out_dims[axis as usize] = len;

                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: t.dtype,
                            shape: Shape::new(out_dims),
                            device: t.device,
                            range: t.range,
                        },
                    ))))
                }
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::ReduceAxis | TensorShapeRule::Softmax => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("receiver must be a tensor", span));
                return Ty::Error;
            };
            let g_args = match turbofish {
                Some(g) => g,
                None => {
                    cx.diags().push(
                        Diagnostic::error(
                            format!("`{}` requires explicit axis generic argument", op_name),
                            span,
                        )
                        .with_code("E0213"),
                    );
                    return Ty::Error;
                }
            };
            if g_args.args.len() != 1 {
                cx.diags().push(Diagnostic::error(
                    format!("`{}` expects exactly 1 generic argument", op_name),
                    g_args.span,
                ));
                return Ty::Error;
            }
            let axis = match cx.extract_const_int(&g_args.args[0]) {
                Some(a) => a,
                None => {
                    cx.diags().push(Diagnostic::error(
                        "AXIS must be a constant integer",
                        g_args.span,
                    ));
                    return Ty::Error;
                }
            };

            match &ka {
                TensorKind::Tensor(t) => {
                    let rank = t.shape.rank();
                    if axis < 0 || axis >= rank as i64 {
                        cx.diags().push(Diagnostic::error(
                            format!("axis {} is out of range for rank {}", axis, rank),
                            g_args.span,
                        ));
                        return Ty::Error;
                    }

                    if descriptor.shape == TensorShapeRule::Softmax {
                        // Softmax preserves shape/dtype/device but produces
                        // probabilities, not the input's image values, so the
                        // value range does not carry through.
                        Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                            TensorTy {
                                dtype: t.dtype,
                                shape: t.shape.clone(),
                                device: t.device,
                                range: ValueRange::Unspecified,
                            },
                        ))))
                    } else {
                        let mut out_dims = t.shape.dims.clone();
                        out_dims.remove(axis as usize);

                        let out_dtype = if descriptor.result == TensorResultRule::Int64Tensor {
                            DType::Int64
                        } else {
                            t.dtype
                        };

                        Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                            TensorTy {
                                dtype: out_dtype,
                                shape: Shape::new(out_dims),
                                device: t.device,
                                // Reductions (incl. softmax/argmax) change
                                // the meaning of the values, so the input
                                // value range does not carry through.
                                range: ValueRange::Unspecified,
                            },
                        ))))
                    }
                }
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::FullReduce => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("receiver must be a tensor", span));
                return Ty::Error;
            };
            match &ka {
                TensorKind::Tensor(t) => Ty::Extension(Box::new(ExtensionTy::Tensor(
                    TensorKind::Tensor(TensorTy {
                        dtype: t.dtype,
                        shape: Shape::new(Vec::new()),
                        device: t.device,
                        // A full reduction to a scalar drops the value range.
                        range: ValueRange::Unspecified,
                    }),
                ))),
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::Cast => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("receiver must be a tensor", span));
                return Ty::Error;
            };
            let g_args = match turbofish {
                Some(g) => g,
                None => {
                    cx.diags().push(
                        Diagnostic::error("cast requires explicit target type", span)
                            .with_code("E0213"),
                    );
                    return Ty::Error;
                }
            };
            if g_args.args.len() != 1 {
                cx.diags().push(Diagnostic::error(
                    "cast expects exactly 1 generic argument",
                    g_args.span,
                ));
                return Ty::Error;
            }
            let target_dtype = match &g_args.args[0] {
                hir::GenericArg::Type(t) => cx.tensor_dtype(*t, g_args.span),
                _ => {
                    cx.diags().push(Diagnostic::error(
                        "cast argument must be a type",
                        g_args.span,
                    ));
                    DType::Float32
                }
            };

            match &ka {
                TensorKind::Tensor(t) => Ty::Extension(Box::new(ExtensionTy::Tensor(
                    TensorKind::Tensor(TensorTy {
                        dtype: target_dtype,
                        shape: t.shape.clone(),
                        device: t.device,
                        range: t.range,
                    }),
                ))),
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::RangeTransition { from, to } => {
            use crate::extensions::tensor::types::DType as TDType;
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("receiver must be a tensor", span));
                return Ty::Error;
            };
            match &ka {
                TensorKind::Tensor(t) => {
                    // The transition operations are defined on Float32 values.
                    if !matches!(t.dtype, TDType::Float32 | TDType::Var(_)) {
                        cx.diags().push(
                            Diagnostic::error(
                                format!(
                                    "`{}` requires a Float32 tensor, found {}",
                                    descriptor.name,
                                    t.dtype.name()
                                ),
                                span,
                            )
                            .with_code("E0212"),
                        );
                        return Ty::Error;
                    }
                    // The receiver must already carry the source value range.
                    if t.range != from {
                        cx.diags().push(
                            Diagnostic::error(
                                format!(
                                    "`{}` requires a `{from}` tensor, found `{}`",
                                    descriptor.name, t.range
                                ),
                                span,
                            )
                            .with_code("E0212")
                            .with_note(format!(
                                "`{}` transitions the value range `{from}` -> `{to}`",
                                descriptor.name
                            )),
                        );
                        return Ty::Error;
                    }
                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: t.dtype,
                            shape: t.shape.clone(),
                            device: t.device,
                            range: to,
                        },
                    ))))
                }
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::ToDevice => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("receiver must be a tensor", span));
                return Ty::Error;
            };
            let g_args = match turbofish {
                Some(g) => g,
                None => {
                    cx.diags().push(
                        Diagnostic::error("to_device requires explicit target device", span)
                            .with_code("E0213"),
                    );
                    return Ty::Error;
                }
            };
            if g_args.args.len() != 1 {
                cx.diags().push(Diagnostic::error(
                    "to_device expects exactly 1 generic argument",
                    g_args.span,
                ));
                return Ty::Error;
            }
            let target_device = cx.build_device(Some(&g_args.args[0]), g_args.span);

            match &ka {
                TensorKind::Tensor(t) => Ty::Extension(Box::new(ExtensionTy::Tensor(
                    TensorKind::Tensor(TensorTy {
                        dtype: t.dtype,
                        shape: t.shape.clone(),
                        device: target_device,
                        range: t.range,
                    }),
                ))),
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
        TensorShapeRule::Transpose => {
            let Some(ka) = tensor_kind_of(cx, &actual_ops[0]) else {
                cx.diags()
                    .push(Diagnostic::error("receiver must be a tensor", span));
                return Ty::Error;
            };

            match &ka {
                TensorKind::Tensor(t) => {
                    let rank = t.shape.rank();
                    if rank != 2 {
                        cx.diags().push(Diagnostic::error(
                            format!("transpose expects a rank-2 tensor, found rank {}", rank),
                            span,
                        ));
                        return Ty::Error;
                    }
                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: t.dtype,
                            shape: Shape::new(vec![
                                t.shape.dims[1].clone(),
                                t.shape.dims[0].clone(),
                            ]),
                            device: t.device,
                            range: t.range,
                        },
                    ))))
                }
                _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
            }
        }
    }
}

/// The `TensorDyn`/`TensorAny` -> `Tensor` refinement boundary.
///
/// Core has already evaluated (and rejected) any value arguments; what remains is the decision
/// about what a refinement produces, which is tensor semantics.
pub(crate) fn eval_tensor_refine(
    cx: &mut dyn TensorCheckCtx,
    base: Ty,
    turbofish: Option<&hir::GenericArgs>,
    name_span: Span,
) -> Ty {
    let Some(generic_args) = turbofish else {
        cx.tensor_error("`refine` requires an explicit target shape", name_span);
        return Ty::Error;
    };

    // A `refine` boundary may also assign the initial value range with an optional `range = R`
    // binding; the remaining args are positional.
    let range = cx.value_range_of(generic_args);
    let positional: Vec<hir::GenericArg> = generic_args
        .args
        .iter()
        .filter(|a| !matches!(a, hir::GenericArg::Binding { .. }))
        .cloned()
        .collect();

    let (dtype, shape) = match base {
        Ty::Extension(ext) => match &*ext {
            ExtensionTy::Tensor(TensorKind::TensorDyn(dtype)) => match positional.as_slice() {
                [hir::GenericArg::Shape(shape)] => (*dtype, cx.build_refine_shape(shape)),
                _ => {
                    cx.tensor_error(
                        "`TensorDyn<T>::refine` expects exactly one shape argument",
                        generic_args.span,
                    );
                    return Ty::Error;
                }
            },
            ExtensionTy::Tensor(TensorKind::TensorAny) => match positional.as_slice() {
                [hir::GenericArg::Type(dtype), hir::GenericArg::Shape(shape)] => (
                    cx.tensor_dtype(*dtype, generic_args.span),
                    cx.build_refine_shape(shape),
                ),
                _ => {
                    cx.tensor_error(
                        "`TensorAny::refine` expects a dtype and a shape",
                        generic_args.span,
                    );
                    return Ty::Error;
                }
            },
            ExtensionTy::Tensor(TensorKind::Tensor(_)) => {
                cx.tensor_error(
                    "`refine` is valid only on `TensorDyn` or `TensorAny`",
                    name_span,
                );
                return Ty::Error;
            }
            _ => {
                cx.tensor_error(
                    "`refine` receiver must be `TensorDyn` or `TensorAny`",
                    name_span,
                );
                return Ty::Error;
            }
        },
        Ty::Error => return Ty::Error,
        _ => {
            cx.tensor_error(
                "`refine` receiver must be `TensorDyn` or `TensorAny`",
                name_span,
            );
            return Ty::Error;
        }
    };

    let tensor = Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
        TensorTy {
            dtype,
            shape,
            device: cx.tensor_state().fresh_device(),
            range,
        },
    ))));
    Ty::Core(CoreType::Result, vec![tensor, Ty::Error])
}

/// A model type's method surface: `predict`, and nothing else. Returns whether checking should
/// continue.
pub(crate) fn check_model_method_name(
    cx: &mut dyn TensorCheckCtx,
    name: &str,
    name_span: Span,
) -> bool {
    if name == "predict" {
        return true;
    }
    cx.diags().push(Diagnostic::error(
        format!("model type has no method named `{}`", name),
        name_span,
    ));
    false
}

/// `.predict(...)` takes exactly one argument per declared input port. Returns whether checking
/// should continue.
pub(crate) fn check_model_predict_arity(
    cx: &mut dyn TensorCheckCtx,
    expected: usize,
    found: usize,
    call_span: Span,
) -> bool {
    if expected == found {
        return true;
    }
    cx.diags().push(
        Diagnostic::error(
            format!(
                "wrong number of arguments for `.predict(...)`: expected {expected}, found {found}"
            ),
            call_span,
        )
        .with_code("E0005"),
    );
    false
}

/// One `.predict(...)` argument against its instantiated input port.
///
/// Core evaluates the argument expression and passes its type in, so this runs once per argument
/// in source order and the interleaving of argument diagnostics with port diagnostics is
/// unchanged.
pub(crate) fn check_model_predict_arg(
    cx: &mut dyn TensorCheckCtx,
    arg_ty: Ty,
    expected_port_ty: Ty,
    arg_span: Span,
    port_note: Option<String>,
) {
    match cx.resolve(&arg_ty) {
        Ty::Ref { inner, .. } => {
            let diagnostic_count = cx.diags().len();
            if cx.unify(expected_port_ty, *inner, arg_span).is_err() {
                if let (Some(note), Some(diagnostic)) =
                    (port_note, cx.diags().get_mut(diagnostic_count))
                {
                    diagnostic.notes.push(note);
                }
            }
        }
        _ => {
            let found = cx.ty_to_string(&arg_ty);
            cx.diags().push(
                Diagnostic::error(
                    format!(
                        "mismatched types: expected a borrowed tensor (e.g. `&tensor`), found `{found}`"
                    ),
                    arg_span,
                )
                .with_code("E0005"),
            );
        }
    }
}

/// A model call yields its single output port's type, or a tuple of all of them.
pub(crate) fn model_predict_result(outputs: Vec<Ty>) -> Ty {
    if outputs.len() == 1 {
        outputs.into_iter().next().expect("length checked")
    } else {
        Ty::Tuple(outputs)
    }
}
