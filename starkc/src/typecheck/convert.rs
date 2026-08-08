//! **AS7 Packet 7 — written type syntax becomes a `Ty`.**
//!
//! One level above `infer`: `convert` may depend on `infer`, `state` and `types`.
//!
//! Everything here answers one question — *what type did the programmer write?* HIR `TypeId` to
//! `Ty`, generic arguments, the generic-parameter scopes those arguments are read against, and the
//! tensor written-type bridge: shapes, devices, dtypes, dimension expressions and value ranges.
//!
//! **Why the tensor conversions are Core machinery and not extension semantics.** AS6 settled
//! this: converting written type syntax walks HIR and the generic-parameter scope tables, which
//! are Core's, so Core keeps the conversion and offers it to the extension as a service. The
//! extension owns the *vocabulary* — which spellings name a dtype, a device constructor or a
//! kind — and that lives in `extensions::tensor::syntax`. This module calls it; it does not
//! restate it. The AS6 vocabulary lint enforces that, and `mod.rs` keeps the `TensorCheckCtx`
//! impl that publishes these as services.
//!
//! **`enter_tensor_param_scope` lives here, and Packet 6 is why.** It was extracted into `state`
//! with the other scoped operations, and the dependency checker then reported a path that would
//! have become `state -> convert -> infer -> state` — a genuine cycle, the only admissible reason
//! to revisit an assignment. It classifies written generic parameters, which is conversion, not
//! storage. `exit_tensor_param_scope` follows it: a scope's two halves belong together.

use super::state::single_segment_name;
use super::state::{TensorParamScopes, TypeChecker};
use super::types::{unit_or_tuple, ExtensionTy, GenericKind, ModelTy, Ty};
use crate::ast::Primitive;
use crate::diag::Diagnostic;
use crate::extensions::tensor::dim::Poly;
use crate::extensions::tensor::syntax as tensor_syntax;
use crate::extensions::tensor::types::{
    DType, Device, DimProvenance, OriginKind, Shape, TensorKind, TensorTy,
};
use crate::hir::{self, CoreType, Res, TypeId};
use crate::source::Span;
use std::collections::HashMap;

impl TypeChecker<'_> {
    /// Register tensor extension generic kinds for an item scope.
    pub(super) fn enter_tensor_param_scope(
        &mut self,
        generics: &[hir::GenericParam],
    ) -> TensorParamScopes {
        let saved = TensorParamScopes {
            dims: std::mem::take(&mut self.dim_scope),
            dtypes: std::mem::take(&mut self.dtype_scope),
            devices: std::mem::take(&mut self.device_scope),
            kinds: std::mem::take(&mut self.generic_kinds),
        };
        for g in generics {
            let name = self.text(g.name).to_string();
            let kind = self.generic_kind(g);
            self.generic_kinds.insert(name.clone(), kind);
            match kind {
                GenericKind::Dim => {
                    let var = self.tensor_ctx.rigid_dim(DimProvenance {
                        span: g.name,
                        origin: OriginKind::Param,
                        label: name.clone(),
                    });
                    self.dim_scope.insert(name, var);
                }
                GenericKind::DType => {
                    let dtype = self.tensor_ctx.rigid_dtype();
                    self.dtype_scope.insert(name, dtype);
                }
                GenericKind::Device => {
                    let device = self.tensor_ctx.rigid_device();
                    self.device_scope.insert(name, device);
                }
                GenericKind::Type => {}
            }
        }
        saved
    }

    pub(super) fn convert_generic_type_args(&mut self, args: Option<&hir::GenericArgs>) -> Vec<Ty> {
        args.map_or_else(Vec::new, |args| {
            args.args
                .iter()
                .filter_map(|arg| match arg {
                    hir::GenericArg::Type(ty) => Some(self.convert_hir_type(*ty)),
                    hir::GenericArg::Const(_) => None,
                    hir::GenericArg::Binding { .. } => None,
                    // Shape arguments are not Core type arguments; the tensor
                    // extension checker (M4.2+) interprets them.
                    hir::GenericArg::Shape(_) => None,
                })
                .collect()
        })
    }

    pub(super) fn convert_hir_type(&mut self, id: TypeId) -> Ty {
        let node = self.hir.ty(id);
        match &node.kind {
            hir::TypeKind::Primitive(p) => {
                if matches!(p, Primitive::Float16 | Primitive::BFloat16) && !self.allow_half_type {
                    self.tensor_error(
                        "`Float16` and `BFloat16` are valid only as tensor element types or explicit cast targets",
                        node.span,
                    );
                    Ty::Error
                } else {
                    Ty::Primitive(*p)
                }
            }
            hir::TypeKind::Path { path, res, args } => {
                // `tensor` extension types resolve to `Res::Err` in the Core
                // resolver; build them here when the extension is enabled.
                if self.options.tensor() {
                    if let Some(name) = single_segment_name(path, self).map(str::to_string) {
                        if let Some(ty) = self.build_tensor_type(&name, args.as_ref(), node.span) {
                            return ty;
                        }
                    }
                }
                match res {
                    Res::Item(item_id) => {
                        let item = self.hir.item(*item_id);
                        match &item.kind {
                            hir::ItemKind::Struct { generics, .. } => {
                                let type_args = self.convert_generic_type_args(args.as_ref());
                                self.validate_generic_arity(
                                    generics.len(),
                                    type_args.len(),
                                    node.span,
                                );
                                Ty::Struct(*item_id, type_args)
                            }
                            hir::ItemKind::Enum { generics, .. } => {
                                let type_args = self.convert_generic_type_args(args.as_ref());
                                self.validate_generic_arity(
                                    generics.len(),
                                    type_args.len(),
                                    node.span,
                                );
                                Ty::Enum(*item_id, type_args)
                            }
                            hir::ItemKind::TypeAlias {
                                generics,
                                ty: target,
                                ..
                            } => {
                                let generics = generics.clone();
                                let target = *target;
                                let type_args = self.convert_generic_type_args(args.as_ref());
                                self.validate_generic_arity(
                                    generics.len(),
                                    type_args.len(),
                                    node.span,
                                );
                                if self.alias_stack.contains(item_id) {
                                    self.diags.push(
                                        Diagnostic::error("recursive type-alias cycle", node.span)
                                            .with_code("E0216"),
                                    );
                                    Ty::Error
                                } else {
                                    self.alias_stack.push(*item_id);
                                    let expanded = self.convert_hir_type(target);
                                    self.alias_stack.pop();
                                    let substitutions: HashMap<String, Ty> = generics
                                        .iter()
                                        .zip(type_args)
                                        .map(|(parameter, argument)| {
                                            (self.text(parameter.name).to_string(), argument)
                                        })
                                        .collect();
                                    self.instantiate_ty(&expanded, &substitutions)
                                }
                            }
                            hir::ItemKind::Model(_def) => {
                                if !self.options.tensor() {
                                    self.diags.push(Diagnostic::error(
                                        "model types require `--extension tensor` to be enabled",
                                        node.span,
                                    ));
                                    Ty::Error
                                } else {
                                    self.validate_generic_arity(
                                        0,
                                        args.as_ref()
                                            .map_or(0, |generic_args| generic_args.args.len()),
                                        node.span,
                                    );
                                    Ty::Extension(Box::new(ExtensionTy::Model(ModelTy {
                                        item_id: *item_id,
                                    })))
                                }
                            }
                            _ => Ty::Error,
                        }
                    }
                    Res::Primitive(p) => Ty::Primitive(*p),
                    Res::SelfType => {
                        if let Some(self_ty) = &self.current_self_ty {
                            self_ty.clone()
                        } else {
                            self.diags.push(
                                Diagnostic::error("use of 'Self' outside impl or trait", node.span)
                                    .with_code("E0202"),
                            );
                            Ty::Error
                        }
                    }
                    Res::SelfAssoc(name) => self
                        .current_assoc_types
                        .get(self.text(*name))
                        .cloned()
                        .unwrap_or_else(|| Ty::Param(format!("Self::{}", self.text(*name)))),
                    Res::TypeParam => {
                        // DEV-148: a type parameter's NAME is a span into the file that declared
                        // the signature being converted, which is not the file being checked when
                        // the call crosses a module boundary. AS1b-ii-d: the span says which file
                        // that is, so no foreign-signature item has to be carried here.
                        let name_str = self.text(node.span);
                        match self.generic_kinds.get(name_str).copied() {
                            Some(GenericKind::Dim) => {
                                self.tensor_error(
                                    "a `Dim` parameter cannot be used in type position",
                                    node.span,
                                );
                                Ty::Error
                            }
                            Some(GenericKind::DType) => {
                                self.tensor_error(
                                    "a `DType` parameter is valid only as a tensor element type",
                                    node.span,
                                );
                                Ty::Error
                            }
                            Some(GenericKind::Device) => {
                                self.tensor_error(
                                    "a `Device` parameter is valid only in `device = ...`",
                                    node.span,
                                );
                                Ty::Error
                            }
                            _ => Ty::Param(name_str.to_string()),
                        }
                    }
                    Res::ParamAssoc(param, assoc) => {
                        Ty::Param(format!("{}::{}", self.text(*param), self.text(*assoc)))
                    }
                    Res::CoreType(core) => {
                        let args = self.convert_generic_type_args(args.as_ref());
                        let expected = match core {
                            CoreType::String
                            | CoreType::CharsIter
                            | CoreType::SplitIter
                            | CoreType::Random
                            | CoreType::IOError
                            | CoreType::File
                            | CoreType::Ordering => 0,
                            CoreType::Vec
                            | CoreType::Box
                            | CoreType::Option
                            | CoreType::Range
                            | CoreType::RangeInclusive
                            | CoreType::VecIter
                            | CoreType::HashSet
                            | CoreType::KeysIter
                            | CoreType::ValuesIter
                            | CoreType::FilterIter => 1,
                            CoreType::Result | CoreType::HashMap | CoreType::MapIter => 2,
                            CoreType::Iter => {
                                if args.len() != 1 && args.len() != 2 {
                                    self.diags.push(
                                        Diagnostic::error(
                                            format!(
                                                "generic type 'Iter' expects 1 or 2 generic arguments, found {}",
                                                args.len()
                                            ),
                                            node.span,
                                        )
                                        .with_code("E0107"),
                                    );
                                }
                                args.len()
                            }
                        };
                        self.validate_generic_arity(expected, args.len(), node.span);
                        // WP-C7.9 Packet I (DEV-118): the obligations the STANDARD LIBRARY imposes
                        // on its own generic parameters, checked at the point of instantiation —
                        // where a written bound would be checked. This is the general mechanism,
                        // not a check bolted onto `insert`: a `HashMap<Float64, Int32>` is
                        // ill-typed wherever it is written, including in a signature it is never
                        // called through.
                        self.check_builtin_type_bounds(*core, &args, node.span);
                        Ty::Core(*core, args)
                    }
                    _ => Ty::Error,
                }
            }
            hir::TypeKind::Array { elem, len } => {
                let elem_ty = self.convert_hir_type(*elem);
                let len_str = self.text(*len);
                let len_val = len_str.parse::<u64>().unwrap_or(0);
                Ty::Array(Box::new(elem_ty), len_val)
            }
            hir::TypeKind::Slice(elem) => {
                let elem_ty = self.convert_hir_type(*elem);
                Ty::Slice(Box::new(elem_ty))
            }
            hir::TypeKind::Tuple(types) => {
                let elems: Vec<Ty> = types.iter().map(|&t| self.convert_hir_type(t)).collect();
                unit_or_tuple(elems)
            }
            hir::TypeKind::Ref { mutable, inner } => {
                let inner_ty = self.convert_hir_type(*inner);
                Ty::Ref {
                    mutable: *mutable,
                    inner: Box::new(inner_ty),
                }
            }
            hir::TypeKind::Fn { params, ret } => {
                let params_ty = params.iter().map(|&p| self.convert_hir_type(p)).collect();
                let ret_ty = ret
                    .map(|r| self.convert_hir_type(r))
                    .unwrap_or(Ty::Primitive(Primitive::Unit));
                Ty::Fn {
                    params: params_ty,
                    ret: Box::new(ret_ty),
                }
            }
            hir::TypeKind::Never => Ty::Never,
            hir::TypeKind::Error => Ty::Error,
        }
    }

    /// Build a `tensor` extension type from a path name and generic arguments,
    /// or `None` if the name is not an extension tensor type. Emits diagnostics
    /// for malformed shapes, undeclared dimensions, and unsupported dtypes.
    pub(super) fn build_tensor_type(
        &mut self,
        name: &str,
        args: Option<&hir::GenericArgs>,
        span: Span,
    ) -> Option<Ty> {
        let empty: &[hir::GenericArg] = &[];
        let arg_list = args.map_or(empty, |a| a.args.as_slice());
        let constructor = tensor_syntax::tensor_type_constructor(name)?;
        match constructor {
            tensor_syntax::TensorTypeConstructor::TensorAny => {
                self.tensor_arity(constructor.name(), 0, arg_list.len(), span);
                Some(Ty::Extension(Box::new(ExtensionTy::Tensor(
                    TensorKind::TensorAny,
                ))))
            }
            tensor_syntax::TensorTypeConstructor::TensorDyn => {
                self.tensor_arity(constructor.name(), 1, arg_list.len(), span);
                let dtype = match arg_list.first() {
                    Some(hir::GenericArg::Type(t)) => self.tensor_dtype(*t, span),
                    _ => {
                        self.tensor_error("`TensorDyn` requires an element type argument", span);
                        DType::Float32
                    }
                };
                Some(Ty::Extension(Box::new(ExtensionTy::Tensor(
                    TensorKind::TensorDyn(dtype),
                ))))
            }
            tensor_syntax::TensorTypeConstructor::Tensor => {
                if !(2..=4).contains(&arg_list.len()) {
                    self.tensor_error(
                        &format!(
                            "`Tensor` expects two to four arguments, found {}",
                            arg_list.len()
                        ),
                        span,
                    );
                }
                let dtype = match arg_list.first() {
                    Some(hir::GenericArg::Type(t)) => self.tensor_dtype(*t, span),
                    _ => {
                        self.tensor_error("`Tensor` requires an element type argument", span);
                        DType::Float32
                    }
                };
                let shape = match arg_list.get(1) {
                    Some(hir::GenericArg::Shape(s)) => self.build_shape(s),
                    _ => {
                        self.tensor_error("`Tensor` requires a shape argument", span);
                        Shape::default()
                    }
                };
                // The `device = D` and `range = R` bindings may appear after the
                // shape in either order; each is optional.
                let mut device_arg = None;
                let mut range_arg = None;
                for arg in arg_list.iter().skip(2) {
                    match arg {
                        hir::GenericArg::Binding { name, .. } => match self.text(*name) {
                            "device" => device_arg = Some(arg),
                            "range" => range_arg = Some(arg),
                            other => self.tensor_error(
                                &format!(
                                    "unknown `Tensor` binding `{other} = ...`; expected `device` or `range`"
                                ),
                                span,
                            ),
                        },
                        _ => self.tensor_error(
                            "a `Tensor` argument after the shape must be `device = D` or `range = R`",
                            span,
                        ),
                    }
                }
                let device = self.build_device(device_arg, span);
                let range = self.build_value_range(range_arg, span);
                Some(Ty::Extension(Box::new(ExtensionTy::Tensor(
                    TensorKind::Tensor(TensorTy {
                        dtype,
                        shape,
                        device,
                        range,
                    }),
                ))))
            }
            tensor_syntax::TensorTypeConstructor::ModelError => {
                self.tensor_arity(constructor.name(), 0, arg_list.len(), span);
                Some(Ty::Extension(Box::new(ExtensionTy::ModelError)))
            }
        }
    }

    pub(super) fn tensor_arity(&mut self, name: &str, expected: usize, actual: usize, span: Span) {
        if expected != actual {
            self.tensor_error(
                &format!("`{name}` expects {expected} argument(s), found {actual}"),
                span,
            );
        }
    }

    /// Convert a type argument to a concrete or generic `DType`.
    pub(super) fn tensor_dtype(&mut self, ty_id: TypeId, span: Span) -> DType {
        if let hir::TypeKind::Path {
            res: Res::TypeParam,
            ..
        } = &self.hir.ty(ty_id).kind
        {
            let name = self.text(self.hir.ty(ty_id).span);
            if let Some(dtype) = self.dtype_scope.get(name) {
                return *dtype;
            }
            self.tensor_error(
                &format!("type parameter `{name}` does not have kind `DType`"),
                span,
            );
            return DType::Float32;
        }
        let saved = self.allow_half_type;
        self.allow_half_type = true;
        let ty = self.convert_hir_type(ty_id);
        self.allow_half_type = saved;
        match ty {
            // AS7 Packet 7: AS6 already established the extension-owned authority for this
            // mapping; Core does not keep a second copy.
            Ty::Primitive(p) => match tensor_syntax::dtype_of_primitive(p) {
                Some(d) => d,
                None => {
                    self.tensor_error(
                        &format!("`{}` is not a valid tensor element type", p.name()),
                        span,
                    );
                    DType::Float32
                }
            },
            _ => {
                self.tensor_error("tensor element type must be a dtype", span);
                DType::Float32
            }
        }
    }

    pub(super) fn build_shape(&mut self, shape: &hir::ShapeArg) -> Shape {
        let span = shape.span;
        let dims = shape
            .dims
            .iter()
            .map(|d| self.dim_expr_to_poly(d, span))
            .collect();
        let spans = shape
            .dims
            .iter()
            .map(|dim| match dim {
                hir::DimExpr::Lit(span) | hir::DimExpr::Var(span) => *span,
                hir::DimExpr::Binary { .. } | hir::DimExpr::Error => shape.span,
            })
            .collect();
        Shape::with_spans(dims, spans)
    }

    pub(super) fn build_refine_shape(&mut self, shape: &hir::ShapeArg) -> Shape {
        let dims = shape
            .dims
            .iter()
            .map(|dim| self.dim_expr_to_poly_mode(dim, shape.span, true))
            .collect();
        let spans = shape
            .dims
            .iter()
            .map(|dim| match dim {
                hir::DimExpr::Lit(span) | hir::DimExpr::Var(span) => *span,
                hir::DimExpr::Binary { .. } | hir::DimExpr::Error => shape.span,
            })
            .collect();
        Shape::with_spans(dims, spans)
    }

    /// Convert a HIR dimension expression to a polynomial, resolving variables
    /// against the current dim scope and enforcing non-negativity (§3.3).
    /// `fallback` is used for diagnostics on nodes (binaries) without a span.
    pub(super) fn dim_expr_to_poly(&mut self, dim: &hir::DimExpr, fallback: Span) -> Poly {
        self.dim_expr_to_poly_mode(dim, fallback, false)
    }

    pub(super) fn dim_expr_to_poly_mode(
        &mut self,
        dim: &hir::DimExpr,
        fallback: Span,
        bind_unbound: bool,
    ) -> Poly {
        match dim {
            hir::DimExpr::Lit(s) => {
                let text = self.text(*s);
                match text.parse::<i64>() {
                    Ok(v) => Poly::constant(v),
                    Err(_) => {
                        self.tensor_error(
                            &format!("dimension literal `{text}` is out of range"),
                            *s,
                        );
                        Poly::constant(0)
                    }
                }
            }
            hir::DimExpr::Var(s) => {
                let name = self.text(*s).to_string();
                match self.dim_scope.get(&name) {
                    Some(&var) => Poly::var(var),
                    None if bind_unbound => {
                        let var = self.tensor_ctx.rigid_dim(DimProvenance {
                            span: *s,
                            origin: OriginKind::Refine,
                            label: name.clone(),
                        });
                        self.dim_scope.insert(name, var);
                        Poly::var(var)
                    }
                    None => {
                        self.tensor_error(&format!("undeclared dimension variable `{name}`"), *s);
                        Poly::constant(0)
                    }
                }
            }
            hir::DimExpr::Binary { op, lhs, rhs } => {
                let l = self.dim_expr_to_poly_mode(lhs, fallback, bind_unbound);
                let r = self.dim_expr_to_poly_mode(rhs, fallback, bind_unbound);
                let result = match op {
                    crate::ast::DimBinOp::Add => l.add(&r),
                    crate::ast::DimBinOp::Sub => l.sub(&r),
                    crate::ast::DimBinOp::Mul => l.mul(&r),
                };
                match result {
                    Ok(p) => {
                        if matches!(op, crate::ast::DimBinOp::Sub) && !p.is_provably_nonnegative() {
                            self.tensor_error(
                                "dimension subtraction may be negative; \
                                 non-negativity must follow from literal constants (§3.3)",
                                fallback,
                            );
                        }
                        p
                    }
                    Err(_) => {
                        self.tensor_error("dimension arithmetic overflowed", fallback);
                        Poly::constant(0)
                    }
                }
            }
            hir::DimExpr::Error => Poly::constant(0),
        }
    }

    /// Resolve an optional `device = D` argument. `Cpu` is concrete; a type
    /// parameter or omission yields a fresh device variable (device-polymorphic
    /// by default, §8).
    pub(super) fn build_device(&mut self, arg: Option<&hir::GenericArg>, span: Span) -> Device {
        match arg {
            None => self.tensor_ctx.fresh_device(),
            Some(hir::GenericArg::Binding { name, ty }) if self.text(*name) == "device" => {
                if let hir::TypeKind::Path { path, res, args } = &self.hir.ty(*ty).kind {
                    let spelling = single_segment_name(path, self);
                    if *res == Res::TypeParam {
                        if let Some(device) = spelling.and_then(|n| self.device_scope.get(n)) {
                            return *device;
                        }
                        self.tensor_error(
                            "device parameter must have kind `Device`",
                            self.hir.ty(*ty).span,
                        );
                        return self.tensor_ctx.fresh_device();
                    }
                    match spelling.and_then(tensor_syntax::device_constructor) {
                        Some(tensor_syntax::DeviceConstructor::Cpu) => {
                            if args.as_ref().is_some_and(|a| !a.args.is_empty()) {
                                self.tensor_error(
                                    "`Cpu` does not take arguments",
                                    self.hir.ty(*ty).span,
                                );
                            }
                            Device::Cpu
                        }
                        Some(tensor_syntax::DeviceConstructor::Cuda) => {
                            self.build_cuda_device(args.as_ref(), self.hir.ty(*ty).span)
                        }
                        None => {
                            self.tensor_error(
                                tensor_syntax::DEVICE_EXPECTATION,
                                self.hir.ty(*ty).span,
                            );
                            self.tensor_ctx.fresh_device()
                        }
                    }
                } else {
                    self.tensor_error("tensor device must be a device type", self.hir.ty(*ty).span);
                    self.tensor_ctx.fresh_device()
                }
            }
            Some(_) => {
                self.tensor_error(
                    "unexpected third `Tensor` argument; expected `device = D`",
                    span,
                );
                self.tensor_ctx.fresh_device()
            }
        }
    }

    /// Resolve an optional `range = R` argument to a value-range state. An
    /// omitted `range` is `Unspecified` (no claim). The states are a fixed,
    /// closed set; unknown names are a tensor error.
    pub(super) fn build_value_range(
        &mut self,
        arg: Option<&hir::GenericArg>,
        _span: Span,
    ) -> crate::extensions::tensor::types::ValueRange {
        use crate::extensions::tensor::types::ValueRange;
        match arg {
            None => ValueRange::Unspecified,
            Some(hir::GenericArg::Binding { ty, .. }) => {
                if let hir::TypeKind::Path { path, .. } = &self.hir.ty(*ty).kind {
                    match single_segment_name(path, self).and_then(tensor_syntax::value_range_state)
                    {
                        Some(state) => state,
                        None => {
                            self.tensor_error(
                                tensor_syntax::VALUE_RANGE_EXPECTATION,
                                self.hir.ty(*ty).span,
                            );
                            ValueRange::Unspecified
                        }
                    }
                } else {
                    self.tensor_error(
                        "tensor range must be a range-state name",
                        self.hir.ty(*ty).span,
                    );
                    ValueRange::Unspecified
                }
            }
            Some(_) => ValueRange::Unspecified,
        }
    }

    pub(super) fn build_cuda_device(
        &mut self,
        args: Option<&hir::GenericArgs>,
        span: Span,
    ) -> Device {
        let Some(args) = args else {
            self.tensor_error(
                "`Cuda` requires one non-negative integer device index",
                span,
            );
            return Device::Cuda(0);
        };
        if args.args.len() != 1 {
            self.tensor_error("`Cuda` requires exactly one device index", span);
            return Device::Cuda(0);
        }
        let hir::GenericArg::Const(index) = args.args[0] else {
            self.tensor_error("`Cuda` device index must be an integer constant", span);
            return Device::Cuda(0);
        };
        match self.text(index).parse::<u32>() {
            Ok(index) => Device::Cuda(index),
            Err(_) => {
                self.tensor_error("`Cuda` device index is out of range", index);
                Device::Cuda(0)
            }
        }
    }

    pub(super) fn generic_kind(&mut self, generic: &hir::GenericParam) -> GenericKind {
        let extension_bounds = generic
            .bounds
            .iter()
            .filter(|bound| bound.res == Res::Err)
            .filter_map(|bound| single_segment_name(&bound.path, self))
            .filter_map(|name| tensor_syntax::tensor_param_kind(name).map(GenericKind::from))
            .collect::<Vec<_>>();
        if extension_bounds.is_empty() {
            return GenericKind::Type;
        }
        if generic.bounds.len() != 1 || extension_bounds.len() != 1 {
            self.tensor_error(tensor_syntax::TENSOR_PARAM_KIND_EXPECTATION, generic.name);
        }
        extension_bounds[0]
    }

    /// Emit a tensor extension diagnostic (error code `E0211`).
    pub(super) fn tensor_error(&mut self, message: &str, span: Span) {
        if !self.suppress_tensor_diagnostics {
            self.diags
                .push(Diagnostic::error(message.to_string(), span).with_code("E0211"));
        }
    }

    pub(super) fn extract_const_int(&self, arg: &hir::GenericArg) -> Option<i64> {
        match arg {
            hir::GenericArg::Const(span) => self.text(*span).parse::<i64>().ok(),
            _ => None,
        }
    }

    pub(super) fn extract_dim_generic(
        &mut self,
        arg: &hir::GenericArg,
        label: &str,
    ) -> Option<Poly> {
        let dimension = match arg {
            hir::GenericArg::Const(span) => {
                self.text(*span).parse::<i64>().ok().map(Poly::constant)
            }
            hir::GenericArg::Type(type_id) => {
                let node = self.hir.ty(*type_id);
                match &node.kind {
                    hir::TypeKind::Path { path, .. } => single_segment_name(path, self)
                        .and_then(|name| self.dim_scope.get(name).copied())
                        .map(Poly::var),
                    _ => None,
                }
            }
            hir::GenericArg::Shape(shape) if shape.dims.len() == 1 => {
                Some(self.build_shape(shape).dims[0].clone())
            }
            _ => None,
        };
        match dimension {
            Some(poly) if poly.is_provably_nonnegative() => Some(poly),
            _ => {
                self.diags.push(
                    Diagnostic::error(
                        format!("{label} must be a non-negative dimension expression"),
                        match arg {
                            hir::GenericArg::Const(span) => *span,
                            hir::GenericArg::Type(type_id) => self.hir.ty(*type_id).span,
                            hir::GenericArg::Binding { name, .. } => *name,
                            hir::GenericArg::Shape(shape) => shape.span,
                        },
                    )
                    .with_code("E0213"),
                );
                None
            }
        }
    }

    pub(super) fn extract_const_int_list(&mut self, arg: &hir::GenericArg) -> Option<Vec<i64>> {
        match arg {
            hir::GenericArg::Shape(s) => {
                let shape = self.build_shape(s);
                let mut list = Vec::new();
                for dim in &shape.dims {
                    let c = dim.as_constant()?;
                    list.push(c);
                }
                Some(list)
            }
            _ => None,
        }
    }

    /// The `range = R` binding of a generic argument list, if any.
    pub(super) fn value_range_of(
        &mut self,
        generic_args: &hir::GenericArgs,
    ) -> crate::extensions::tensor::types::ValueRange {
        let range_arg = generic_args.args.iter().find(
            |a| matches!(a, hir::GenericArg::Binding { name, .. } if self.text(*name) == "range"),
        );
        self.build_value_range(range_arg, generic_args.span)
    }

    pub(super) fn exit_tensor_param_scope(&mut self, saved: TensorParamScopes) {
        self.dim_scope = saved.dims;
        self.dtype_scope = saved.dtypes;
        self.device_scope = saved.devices;
        self.generic_kinds = saved.kinds;
    }

    // AS7 Packet 7: validating a written type application belongs with the conversion that
    // reads it. `check_builtin_type_bounds` stays at the SAME point during conversion, so the
    // diagnostic order and inference timing are unchanged; it asks `traits` for identity only.
    pub(super) fn check_builtin_type_bounds(&mut self, core: CoreType, args: &[Ty], span: Span) {
        for (position, required) in Self::builtin_type_bounds(core) {
            let Some(arg) = args.get(*position) else {
                continue;
            };
            // An inference variable is not yet a type; requiring a bound of it here would reject
            // programs whose key type is perfectly valid but not yet known.
            let resolved = self.resolve(arg);
            if matches!(resolved, Ty::Error | Ty::Infer(_)) {
                continue;
            }
            for bound in *required {
                // AS7 Packet 7: a builtin's bounds carry no associated-type bindings, so this is a
                // trait-IDENTITY question. Calling `bounds` here would close the cycle that
                // stopped the packet.
                if !self.satisfies_bound_identity(&resolved, bound, None) {
                    self.diags.push(
                        Diagnostic::error(
                            format!(
                                "type '{}' does not satisfy the bound '{bound}' required by \
                                 '{}' parameter {}",
                                self.ty_to_string(&resolved),
                                self.ty_to_string(&Ty::Core(core, Vec::new())),
                                position + 1
                            ),
                            span,
                        )
                        .with_code("E0500"),
                    );
                }
            }
        }
    }
    pub(super) fn validate_generic_arity(&mut self, expected: usize, actual: usize, span: Span) {
        if expected != actual {
            self.diags.push(
                Diagnostic::error(
                    format!("generic argument count mismatch: expected {expected}, found {actual}"),
                    span,
                )
                .with_code("E0001"),
            );
        }
    }
}
