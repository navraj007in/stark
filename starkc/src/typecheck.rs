//! Type checking, mutability, and definite assignment validation pass for STARK (PLAN.md M2.2).

use crate::ast::{AssignOp, BinOp, Lit, Primitive, UnOp};
use crate::diag::Diagnostic;
use crate::extensions::tensor::dim::{DimVar, Poly};
use crate::extensions::tensor::types::{
    DType, Device, DeviceVar, DimProvenance, OriginKind, Shape, TensorKind, TensorTy, UnifyCtx,
    UnifyError,
};
use crate::hir::{
    self, BlockId, Builtin, CoreType, ExprId, Hir, ItemId, LocalId, PatId, Res, StmtId, TypeId,
};
use crate::options::LanguageOptions;
use crate::source::{SourceFile, Span};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVarId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Primitive(Primitive),
    Struct(ItemId, Vec<Ty>),
    Enum(ItemId, Vec<Ty>),
    Core(CoreType, Vec<Ty>),
    Ref { mutable: bool, inner: Box<Ty> },
    Tuple(Vec<Ty>),
    Array(Box<Ty>, u64),
    Slice(Box<Ty>),
    Fn { params: Vec<Ty>, ret: Box<Ty> },
    Range(Box<Ty>),
    Never,
    Param(String),
    Infer(TypeVarId),
    Extension(Box<ExtensionTy>),
    Error,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExtensionTy {
    Tensor(TensorKind),
    Model(ModelTy),
    ModelError,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ModelTy {
    pub item_id: ItemId,
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum VariantFields {
    Unit,
    Tuple(Vec<Ty>),
    Struct(HashMap<String, Ty>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GenericKind {
    Type,
    Dim,
    DType,
    Device,
}

struct TensorParamScopes {
    dims: HashMap<String, DimVar>,
    dtypes: HashMap<String, DType>,
    devices: HashMap<String, Device>,
    kinds: HashMap<String, GenericKind>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct VariantTy {
    name: String,
    fields: VariantFields,
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct FnSigTy {
    params: Vec<Ty>,
    ret: Ty,
}

struct LoopContext {
    allows_value: bool,
    break_ty: Ty,
    has_break: bool,
}

#[derive(Clone, Copy)]
struct ControlSummary {
    can_complete: bool,
    may_return: bool,
}

pub struct TypeChecker<'a> {
    hir: &'a Hir,
    file: Arc<SourceFile>,
    diags: Vec<Diagnostic>,
    subst: HashMap<TypeVarId, Ty>,
    var_count: u32,

    // Side tables
    expr_types: HashMap<ExprId, Ty>,
    local_types: HashMap<LocalId, Ty>,
    local_mutability: HashMap<LocalId, bool>,
    struct_fields: HashMap<ItemId, HashMap<String, Ty>>,
    enum_variants: HashMap<ItemId, Vec<VariantTy>>,
    fn_sigs: HashMap<ItemId, FnSigTy>,
    const_types: HashMap<ItemId, Ty>,

    // Scopes context
    current_self_ty: Option<Ty>,
    current_assoc_types: HashMap<String, Ty>,
    current_fn_ret: Option<Ty>,
    loop_nesting: u32,
    loop_contexts: Vec<LoopContext>,
    current_fn_generics: Option<Vec<hir::GenericParam>>,

    // Bounds checks to run at the end of checking
    bounds_checks: Vec<(Ty, Vec<hir::TraitRef>, Span)>,

    /// Enabled language extensions, threaded from the CLI through the whole
    /// front end (parse → resolve → typecheck).
    options: LanguageOptions,

    /// Dimension/device unification state and provenance for the `tensor`
    /// extension (§5). Empty and unused for Core-only programs.
    tensor_ctx: UnifyCtx,

    /// Dimension variables in scope for the item being checked, keyed by name
    /// (the `Dim` generic parameters of the enclosing function or model, §3.1).
    /// A dimension identifier not found here is an undeclared-dimension error.
    dim_scope: HashMap<String, DimVar>,
    dtype_scope: HashMap<String, DType>,
    device_scope: HashMap<String, Device>,
    generic_kinds: HashMap<String, GenericKind>,
    suppress_tensor_diagnostics: bool,
    allow_half_type: bool,
}

#[derive(Debug, Clone, Default)]
pub struct TypeTables {
    pub expr_types: HashMap<ExprId, Ty>,
    pub local_types: HashMap<LocalId, Ty>,
    pub local_mutability: HashMap<LocalId, bool>,
}

#[derive(Debug, Clone)]
pub struct TypeCheckResult {
    pub diagnostics: Vec<Diagnostic>,
    pub tables: TypeTables,
}

pub fn check(hir: &Hir, file: Arc<SourceFile>) -> Vec<Diagnostic> {
    analyze(hir, file).diagnostics
}

/// Core-only [`check`], with the option-aware pipeline (Gate 4+).
pub fn check_with_options(
    hir: &Hir,
    file: Arc<SourceFile>,
    options: LanguageOptions,
) -> Vec<Diagnostic> {
    analyze_with_options(hir, file, options).diagnostics
}

pub fn analyze(hir: &Hir, file: Arc<SourceFile>) -> TypeCheckResult {
    analyze_with_options(hir, file, LanguageOptions::CORE)
}

pub fn analyze_with_options(
    hir: &Hir,
    file: Arc<SourceFile>,
    options: LanguageOptions,
) -> TypeCheckResult {
    let mut checker = TypeChecker {
        hir,
        file: file.clone(),
        options,
        diags: Vec::new(),
        subst: HashMap::new(),
        var_count: 0,
        expr_types: HashMap::new(),
        local_types: HashMap::new(),
        local_mutability: HashMap::new(),
        struct_fields: HashMap::new(),
        enum_variants: HashMap::new(),
        fn_sigs: HashMap::new(),
        const_types: HashMap::new(),
        current_self_ty: None,
        current_assoc_types: HashMap::new(),
        current_fn_ret: None,
        loop_nesting: 0,
        loop_contexts: Vec::new(),
        current_fn_generics: None,
        bounds_checks: Vec::new(),
        tensor_ctx: UnifyCtx::new(),
        dim_scope: HashMap::new(),
        dtype_scope: HashMap::new(),
        device_scope: HashMap::new(),
        generic_kinds: HashMap::new(),
        suppress_tensor_diagnostics: false,
        allow_half_type: false,
    };

    checker.check_crate();
    let expr_types = checker
        .expr_types
        .iter()
        .map(|(&id, ty)| (id, checker.ground(ty)))
        .collect();
    let local_types = checker
        .local_types
        .iter()
        .map(|(&id, ty)| (id, checker.ground(ty)))
        .collect();
    let mut diagnostics = checker.diags;
    diagnostics.extend(crate::flow::check(hir, file.clone(), &expr_types));
    diagnostics.extend(crate::borrowck::check(hir, file, &expr_types, &local_types));
    TypeCheckResult {
        diagnostics,
        tables: TypeTables {
            expr_types,
            local_types,
            local_mutability: checker.local_mutability,
        },
    }
}

impl<'a> TypeChecker<'a> {
    fn text(&self, span: Span) -> &str {
        &self.file.src[span.lo as usize..span.hi as usize]
    }

    fn new_type_var(&mut self) -> Ty {
        let id = TypeVarId(self.var_count);
        self.var_count += 1;
        Ty::Infer(id)
    }

    fn builtin_type(&mut self, builtin: Builtin) -> Ty {
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
            Builtin::VecNew => Ty::Fn {
                params: Vec::new(),
                ret: Box::new(Ty::Core(CoreType::Vec, vec![self.new_type_var()])),
            },
            Builtin::VecWithCapacity => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::UInt64)],
                ret: Box::new(Ty::Core(CoreType::Vec, vec![self.new_type_var()])),
            },
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
                        Ty::Primitive(Primitive::String),
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
                        Ty::Primitive(Primitive::String),
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
            Builtin::TensorZeros
            | Builtin::TensorOnes
            | Builtin::TensorFull
            | Builtin::TensorFromVec
            | Builtin::TensorAdd
            | Builtin::TensorSub
            | Builtin::TensorMul
            | Builtin::TensorDiv
            | Builtin::TensorMin
            | Builtin::TensorMax
            | Builtin::TensorEq
            | Builtin::TensorNe
            | Builtin::TensorLt
            | Builtin::TensorLe
            | Builtin::TensorGt
            | Builtin::TensorGe
            | Builtin::TensorBroadcastTo
            | Builtin::TensorMatMul
            | Builtin::TensorBatchMatMul
            | Builtin::TensorConcat
            | Builtin::TensorPermute
            | Builtin::TensorReshape
            | Builtin::TensorSliceAxis
            | Builtin::TensorTranspose
            | Builtin::TensorSumAxis
            | Builtin::TensorMeanAxis
            | Builtin::TensorArgMax
            | Builtin::TensorSum
            | Builtin::TensorSoftmax
            | Builtin::TensorCast
            | Builtin::TensorScale255
            | Builtin::TensorNormalize
            | Builtin::TensorToDevice => Ty::Fn {
                params: vec![],
                ret: Box::new(self.new_type_var()),
            },
        }
    }

    fn resolve(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Infer(id) => {
                if let Some(target) = self.subst.get(id) {
                    self.resolve(target)
                } else {
                    ty.clone()
                }
            }
            Ty::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.resolve(inner)),
            },
            Ty::Struct(item, args) => {
                Ty::Struct(*item, args.iter().map(|arg| self.resolve(arg)).collect())
            }
            Ty::Enum(item, args) => {
                Ty::Enum(*item, args.iter().map(|arg| self.resolve(arg)).collect())
            }
            Ty::Core(core, args) => {
                Ty::Core(*core, args.iter().map(|arg| self.resolve(arg)).collect())
            }
            Ty::Tuple(elems) => Ty::Tuple(elems.iter().map(|e| self.resolve(e)).collect()),
            Ty::Array(elem, len) => Ty::Array(Box::new(self.resolve(elem)), *len),
            Ty::Slice(elem) => Ty::Slice(Box::new(self.resolve(elem))),
            Ty::Fn { params, ret } => Ty::Fn {
                params: params.iter().map(|p| self.resolve(p)).collect(),
                ret: Box::new(self.resolve(ret)),
            },
            Ty::Range(elem) => Ty::Range(Box::new(self.resolve(elem))),
            Ty::Extension(ext) => Ty::Extension(ext.clone()),
            _ => ty.clone(),
        }
    }

    /// Deep-resolve a type for publication in [`TypeTables`], additionally
    /// grounding tensor shape dimensions through the tensor unification context
    /// (e.g. a model's fresh output dim `N` bound to `1` by a `predict` call).
    /// Unlike [`Self::resolve`] this is *not* used on the unification hot path,
    /// so backend consumers see concrete shapes wherever they are determined.
    fn ground(&self, ty: &Ty) -> Ty {
        let ty = self.resolve(ty);
        self.ground_tensor_dims(&ty)
    }

    fn ground_tensor_dims(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Extension(ext) => match &**ext {
                ExtensionTy::Tensor(TensorKind::Tensor(t)) => {
                    let dims: Vec<_> = t
                        .shape
                        .dims
                        .iter()
                        .map(|d| self.tensor_ctx.resolve_dim(d).unwrap_or_else(|_| d.clone()))
                        .collect();
                    // Grounding preserves rank; keep spans only if they still align.
                    let spans = if t.shape.spans.len() == dims.len() {
                        t.shape.spans.clone()
                    } else {
                        Vec::new()
                    };
                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: t.dtype,
                            shape: Shape { dims, spans },
                            device: t.device,
                            range: t.range,
                        },
                    ))))
                }
                _ => ty.clone(),
            },
            Ty::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.ground_tensor_dims(inner)),
            },
            Ty::Struct(item, args) => Ty::Struct(
                *item,
                args.iter().map(|a| self.ground_tensor_dims(a)).collect(),
            ),
            Ty::Enum(item, args) => Ty::Enum(
                *item,
                args.iter().map(|a| self.ground_tensor_dims(a)).collect(),
            ),
            Ty::Core(core, args) => Ty::Core(
                *core,
                args.iter().map(|a| self.ground_tensor_dims(a)).collect(),
            ),
            Ty::Tuple(elems) => {
                Ty::Tuple(elems.iter().map(|e| self.ground_tensor_dims(e)).collect())
            }
            Ty::Array(elem, len) => Ty::Array(Box::new(self.ground_tensor_dims(elem)), *len),
            Ty::Slice(elem) => Ty::Slice(Box::new(self.ground_tensor_dims(elem))),
            Ty::Range(elem) => Ty::Range(Box::new(self.ground_tensor_dims(elem))),
            Ty::Fn { params, ret } => Ty::Fn {
                params: params.iter().map(|p| self.ground_tensor_dims(p)).collect(),
                ret: Box::new(self.ground_tensor_dims(ret)),
            },
            _ => ty.clone(),
        }
    }

    fn occurs_in(&self, id: TypeVarId, ty: &Ty) -> bool {
        match ty {
            Ty::Infer(other_id) => id == *other_id,
            Ty::Ref { inner, .. } => self.occurs_in(id, inner),
            Ty::Struct(_, args) | Ty::Enum(_, args) | Ty::Core(_, args) => {
                args.iter().any(|arg| self.occurs_in(id, arg))
            }
            Ty::Tuple(elems) => elems.iter().any(|e| self.occurs_in(id, e)),
            Ty::Array(elem, _) => self.occurs_in(id, elem),
            Ty::Slice(elem) => self.occurs_in(id, elem),
            Ty::Fn { params, ret } => {
                params.iter().any(|p| self.occurs_in(id, p)) || self.occurs_in(id, ret)
            }
            Ty::Range(elem) => self.occurs_in(id, elem),
            Ty::Extension(ext) => match &**ext {
                ExtensionTy::Tensor(_) | ExtensionTy::Model(_) | ExtensionTy::ModelError => false,
            },
            _ => false,
        }
    }

    fn unify(&mut self, t1: Ty, t2: Ty, span: Span) -> Result<(), ()> {
        let t1 = self.resolve(&t1);
        let t2 = self.resolve(&t2);

        match (t1, t2) {
            (Ty::Infer(id1), Ty::Infer(id2)) if id1 == id2 => Ok(()),
            (Ty::Infer(id), other) => {
                if self.occurs_in(id, &other) {
                    self.diags.push(
                        Diagnostic::error("recursive type inference mismatch", span)
                            .with_code("E0001"),
                    );
                    return Err(());
                }
                self.subst.insert(id, other);
                Ok(())
            }
            (other, Ty::Infer(id)) => {
                if self.occurs_in(id, &other) {
                    self.diags.push(
                        Diagnostic::error("recursive type inference mismatch", span)
                            .with_code("E0001"),
                    );
                    return Err(());
                }
                self.subst.insert(id, other);
                Ok(())
            }
            (Ty::Primitive(p1), Ty::Primitive(p2)) if p1 == p2 => Ok(()),
            (Ty::Struct(s1, args1), Ty::Struct(s2, args2)) if s1 == s2 => {
                self.unify_type_lists(args1, args2, span)
            }
            (Ty::Enum(e1, args1), Ty::Enum(e2, args2)) if e1 == e2 => {
                self.unify_type_lists(args1, args2, span)
            }
            (Ty::Core(c1, args1), Ty::Core(c2, args2)) if c1 == c2 => {
                self.unify_type_lists(args1, args2, span)
            }
            (
                Ty::Ref {
                    mutable: false,
                    inner: expected,
                },
                Ty::Ref {
                    mutable: true,
                    inner: actual,
                },
            ) => self.unify(*expected, *actual, span),
            (
                Ty::Ref {
                    mutable: m1,
                    inner: i1,
                },
                Ty::Ref {
                    mutable: m2,
                    inner: i2,
                },
            ) => {
                if m1 == m2 {
                    self.unify(*i1, *i2, span)
                } else {
                    self.diags.push(
                        Diagnostic::error("reference mutability mismatch", span).with_code("E0001"),
                    );
                    Err(())
                }
            }
            (Ty::Tuple(elems1), Ty::Tuple(elems2)) => {
                if elems1.len() == elems2.len() {
                    for (e1, e2) in elems1.into_iter().zip(elems2) {
                        self.unify(e1, e2, span)?;
                    }
                    Ok(())
                } else {
                    self.diags
                        .push(Diagnostic::error("tuple size mismatch", span).with_code("E0001"));
                    Err(())
                }
            }
            (Ty::Array(e1, len1), Ty::Array(e2, len2)) => {
                if len1 == len2 {
                    self.unify(*e1, *e2, span)
                } else {
                    self.diags
                        .push(Diagnostic::error("array length mismatch", span).with_code("E0001"));
                    Err(())
                }
            }
            (Ty::Slice(e1), Ty::Slice(e2)) => self.unify(*e1, *e2, span),
            (Ty::Slice(expected), Ty::Array(actual, _)) => self.unify(*expected, *actual, span),
            (
                Ty::Fn {
                    params: p1,
                    ret: r1,
                },
                Ty::Fn {
                    params: p2,
                    ret: r2,
                },
            ) => {
                if p1.len() == p2.len() {
                    for (param1, param2) in p1.into_iter().zip(p2) {
                        self.unify(param1, param2, span)?;
                    }
                    self.unify(*r1, *r2, span)
                } else {
                    self.diags.push(
                        Diagnostic::error("function signature parameters mismatch", span)
                            .with_code("E0005"),
                    );
                    Err(())
                }
            }
            (Ty::Range(e1), Ty::Range(e2)) => self.unify(*e1, *e2, span),
            (Ty::Param(p1), Ty::Param(p2)) if p1 == p2 => Ok(()),
            (Ty::Extension(a), Ty::Extension(b)) => match (a.as_ref(), b.as_ref()) {
                (ExtensionTy::Tensor(ta), ExtensionTy::Tensor(tb)) => {
                    self.unify_tensor_types(ta, tb, span)
                }
                (ExtensionTy::Model(ma), ExtensionTy::Model(mb)) => {
                    if ma.item_id == mb.item_id {
                        Ok(())
                    } else {
                        let name_a =
                            if let hir::ItemKind::Model(def) = &self.hir.item(ma.item_id).kind {
                                self.text(def.name).to_string()
                            } else {
                                "Model".to_string()
                            };
                        let name_b =
                            if let hir::ItemKind::Model(def) = &self.hir.item(mb.item_id).kind {
                                self.text(def.name).to_string()
                            } else {
                                "Model".to_string()
                            };
                        self.diags.push(
                            Diagnostic::error(
                                format!("type mismatch: model `{name_a}` and model `{name_b}`"),
                                span,
                            )
                            .with_code("E0005"),
                        );
                        Err(())
                    }
                }
                (ExtensionTy::ModelError, ExtensionTy::ModelError) => Ok(()),
                _ => {
                    self.diags.push(
                        Diagnostic::error(
                            format!(
                                "type mismatch: `{}` and `{}`",
                                self.ty_to_string(&Ty::Extension(a.clone())),
                                self.ty_to_string(&Ty::Extension(b.clone()))
                            ),
                            span,
                        )
                        .with_code("E0005"),
                    );
                    Err(())
                }
            },
            (Ty::Never, _) | (_, Ty::Never) => Ok(()),
            (Ty::Error, _) | (_, Ty::Error) => Ok(()),
            (t1_resolved, t2_resolved) => {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "type mismatch: expected '{}', found '{}'",
                            self.ty_to_string(&t1_resolved),
                            self.ty_to_string(&t2_resolved)
                        ),
                        span,
                    )
                    .with_code("E0001"),
                );
                Err(())
            }
        }
    }

    fn unify_type_lists(&mut self, left: Vec<Ty>, right: Vec<Ty>, span: Span) -> Result<(), ()> {
        if left.len() != right.len() {
            self.diags.push(
                Diagnostic::error("generic argument count mismatch", span).with_code("E0001"),
            );
            return Err(());
        }
        for (left, right) in left.into_iter().zip(right) {
            self.unify(left, right, span)?;
        }
        Ok(())
    }

    fn format_nominal(&self, name: Span, args: &[Ty]) -> String {
        let name = self.text(name);
        if args.is_empty() {
            name.to_string()
        } else {
            format!(
                "{}<{}>",
                name,
                args.iter()
                    .map(|arg| self.ty_to_string(arg))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }

    fn convert_generic_type_args(&mut self, args: Option<&hir::GenericArgs>) -> Vec<Ty> {
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

    /// A deterministic key for a shape argument, used only to keep signature
    /// keys total. The tensor extension checker owns real shape equality.
    fn dim_key(&self, dim: &hir::DimExpr) -> String {
        match dim {
            hir::DimExpr::Lit(s) | hir::DimExpr::Var(s) => self.text(*s).to_string(),
            hir::DimExpr::Binary { op, lhs, rhs } => {
                format!(
                    "({} {} {})",
                    self.dim_key(lhs),
                    op.symbol(),
                    self.dim_key(rhs)
                )
            }
            hir::DimExpr::Error => "<err>".to_string(),
        }
    }

    fn validate_generic_arity(&mut self, expected: usize, actual: usize, span: Span) {
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

    fn item_generic_params(&self, item_id: ItemId) -> &[hir::GenericParam] {
        match &self.hir.item(item_id).kind {
            hir::ItemKind::Struct { generics, .. }
            | hir::ItemKind::Enum { generics, .. }
            | hir::ItemKind::Trait { generics, .. }
            | hir::ItemKind::TypeAlias { generics, .. } => generics,
            _ => &[],
        }
    }

    fn nominal_use_args(
        &mut self,
        item_id: ItemId,
        explicit: Option<&hir::GenericArgs>,
        span: Span,
    ) -> Vec<Ty> {
        let expected = self.item_generic_params(item_id).len();
        if let Some(explicit) = explicit {
            let args = self.convert_generic_type_args(Some(explicit));
            self.validate_generic_arity(expected, args.len(), span);
            args
        } else {
            (0..expected).map(|_| self.new_type_var()).collect()
        }
    }

    fn nominal_param_map(&self, item_id: ItemId, args: &[Ty]) -> HashMap<String, Ty> {
        self.item_generic_params(item_id)
            .iter()
            .zip(args)
            .map(|(param, arg)| (self.text(param.name).to_string(), arg.clone()))
            .collect()
    }

    fn is_unsized_value_type(&self, ty: &Ty) -> bool {
        matches!(
            self.resolve(ty),
            Ty::Slice(_) | Ty::Primitive(Primitive::Str)
        )
    }

    fn check_field_initializers(
        &mut self,
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

    fn ty_to_string(&self, ty: &Ty) -> String {
        let ty = self.resolve(ty);
        match ty {
            Ty::Primitive(p) => p.name().to_string(),
            Ty::Struct(id, args) => {
                let item = self.hir.item(id);
                if let hir::ItemKind::Struct { name, .. } = &item.kind {
                    self.format_nominal(*name, &args)
                } else {
                    "Struct".to_string()
                }
            }
            Ty::Enum(id, args) => {
                let item = self.hir.item(id);
                if let hir::ItemKind::Enum { name, .. } = &item.kind {
                    self.format_nominal(*name, &args)
                } else {
                    "Enum".to_string()
                }
            }
            Ty::Core(core, args) => {
                let name = match core {
                    CoreType::String => "String",
                    CoreType::Vec => "Vec",
                    CoreType::Box => "Box",
                    CoreType::Option => "Option",
                    CoreType::Result => "Result",
                    CoreType::Range => "Range",
                    CoreType::RangeInclusive => "RangeInclusive",
                };
                if args.is_empty() {
                    name.to_string()
                } else {
                    format!(
                        "{}<{}>",
                        name,
                        args.iter()
                            .map(|arg| self.ty_to_string(arg))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            Ty::Ref { mutable, inner } => {
                let prefix = if mutable { "&mut " } else { "&" };
                format!("{}{}", prefix, self.ty_to_string(&inner))
            }
            Ty::Tuple(elems) => {
                let el_strs: Vec<String> = elems.iter().map(|e| self.ty_to_string(e)).collect();
                format!("({})", el_strs.join(", "))
            }
            Ty::Array(elem, len) => {
                format!("[{}; {}]", self.ty_to_string(&elem), len)
            }
            Ty::Slice(elem) => {
                format!("[{}]", self.ty_to_string(&elem))
            }
            Ty::Fn { params, ret } => {
                let p_strs: Vec<String> = params.iter().map(|p| self.ty_to_string(p)).collect();
                format!("fn({}) -> {}", p_strs.join(", "), self.ty_to_string(&ret))
            }
            Ty::Range(elem) => format!("Range<{}>", self.ty_to_string(&elem)),
            Ty::Param(name) => name.clone(),
            Ty::Never => "!".to_string(),
            Ty::Infer(id) => format!("_infer_{}", id.0),
            Ty::Extension(ext) => match ext.as_ref() {
                ExtensionTy::Tensor(tensor) => self.tensor_ctx.display_tensor(tensor),
                ExtensionTy::Model(model) => {
                    let item = self.hir.item(model.item_id);
                    if let hir::ItemKind::Model(def) = &item.kind {
                        self.text(def.name).to_string()
                    } else {
                        "Model".to_string()
                    }
                }
                ExtensionTy::ModelError => "ModelError".to_string(),
            },
            Ty::Error => "{error}".to_string(),
        }
    }

    fn convert_hir_type(&mut self, id: TypeId) -> Ty {
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
                            CoreType::String => 0,
                            CoreType::Vec
                            | CoreType::Box
                            | CoreType::Option
                            | CoreType::Range
                            | CoreType::RangeInclusive => 1,
                            CoreType::Result => 2,
                        };
                        self.validate_generic_arity(expected, args.len(), node.span);
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
                let elems = types.iter().map(|&t| self.convert_hir_type(t)).collect();
                Ty::Tuple(elems)
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
    fn build_tensor_type(
        &mut self,
        name: &str,
        args: Option<&hir::GenericArgs>,
        span: Span,
    ) -> Option<Ty> {
        let empty: &[hir::GenericArg] = &[];
        let arg_list = args.map_or(empty, |a| a.args.as_slice());
        match name {
            "TensorAny" => {
                self.tensor_arity("TensorAny", 0, arg_list.len(), span);
                Some(Ty::Extension(Box::new(ExtensionTy::Tensor(
                    TensorKind::TensorAny,
                ))))
            }
            "TensorDyn" => {
                self.tensor_arity("TensorDyn", 1, arg_list.len(), span);
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
            "Tensor" => {
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
            "ModelError" => {
                self.tensor_arity("ModelError", 0, arg_list.len(), span);
                Some(Ty::Extension(Box::new(ExtensionTy::ModelError)))
            }
            _ => None,
        }
    }

    fn tensor_arity(&mut self, name: &str, expected: usize, actual: usize, span: Span) {
        if expected != actual {
            self.tensor_error(
                &format!("`{name}` expects {expected} argument(s), found {actual}"),
                span,
            );
        }
    }

    /// Convert a type argument to a concrete or generic `DType`.
    fn tensor_dtype(&mut self, ty_id: TypeId, span: Span) -> DType {
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
            Ty::Primitive(p) => match dtype_from_primitive(p) {
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

    fn build_shape(&mut self, shape: &hir::ShapeArg) -> Shape {
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

    fn build_refine_shape(&mut self, shape: &hir::ShapeArg) -> Shape {
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
    fn dim_expr_to_poly(&mut self, dim: &hir::DimExpr, fallback: Span) -> Poly {
        self.dim_expr_to_poly_mode(dim, fallback, false)
    }

    fn dim_expr_to_poly_mode(
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
    fn build_device(&mut self, arg: Option<&hir::GenericArg>, span: Span) -> Device {
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
                    match spelling {
                        Some("Cpu") => {
                            if args.as_ref().is_some_and(|a| !a.args.is_empty()) {
                                self.tensor_error(
                                    "`Cpu` does not take arguments",
                                    self.hir.ty(*ty).span,
                                );
                            }
                            Device::Cpu
                        }
                        Some("Cuda") => {
                            self.build_cuda_device(args.as_ref(), self.hir.ty(*ty).span)
                        }
                        _ => {
                            self.tensor_error("unknown tensor device; expected `Cpu`, `Cuda<N>`, or a `Device` parameter", self.hir.ty(*ty).span);
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
    fn build_value_range(
        &mut self,
        arg: Option<&hir::GenericArg>,
        _span: Span,
    ) -> crate::extensions::tensor::types::ValueRange {
        use crate::extensions::tensor::types::ValueRange;
        match arg {
            None => ValueRange::Unspecified,
            Some(hir::GenericArg::Binding { ty, .. }) => {
                if let hir::TypeKind::Path { path, .. } = &self.hir.ty(*ty).kind {
                    match single_segment_name(path, self) {
                        Some("Unspecified") => ValueRange::Unspecified,
                        Some("ByteRange") => ValueRange::ByteRange,
                        Some("UnitRange") => ValueRange::UnitRange,
                        Some("Normalized") => ValueRange::Normalized,
                        _ => {
                            self.tensor_error(
                                "unknown value range; expected `ByteRange`, `UnitRange`, \
                                 `Normalized`, or `Unspecified`",
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

    /// Combine two operand value ranges for an elementwise op. `Unspecified` is
    /// neutral (absorbs the other side); two different specified ranges cannot be
    /// merged and yield `None`.
    fn combine_value_range(
        &self,
        a: crate::extensions::tensor::types::ValueRange,
        b: crate::extensions::tensor::types::ValueRange,
    ) -> Option<crate::extensions::tensor::types::ValueRange> {
        use crate::extensions::tensor::types::ValueRange::Unspecified;
        match (a, b) {
            (Unspecified, r) | (r, Unspecified) => Some(r),
            (x, y) if x == y => Some(x),
            _ => None,
        }
    }

    fn build_cuda_device(&mut self, args: Option<&hir::GenericArgs>, span: Span) -> Device {
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

    /// Register tensor extension generic kinds for an item scope.
    fn enter_tensor_param_scope(&mut self, generics: &[hir::GenericParam]) -> TensorParamScopes {
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

    fn generic_kind(&mut self, generic: &hir::GenericParam) -> GenericKind {
        let extension_bounds = generic
            .bounds
            .iter()
            .filter(|bound| bound.res == Res::Err)
            .filter_map(|bound| single_segment_name(&bound.path, self))
            .filter_map(|name| match name {
                "Dim" => Some(GenericKind::Dim),
                "DType" => Some(GenericKind::DType),
                "Device" => Some(GenericKind::Device),
                _ => None,
            })
            .collect::<Vec<_>>();
        if extension_bounds.is_empty() {
            return GenericKind::Type;
        }
        if generic.bounds.len() != 1 || extension_bounds.len() != 1 {
            self.tensor_error(
                "tensor kind parameters must have exactly one of `Dim`, `DType`, or `Device` and no trait bounds",
                generic.name,
            );
        }
        extension_bounds[0]
    }

    fn exit_tensor_param_scope(&mut self, saved: TensorParamScopes) {
        self.dim_scope = saved.dims;
        self.dtype_scope = saved.dtypes;
        self.device_scope = saved.devices;
        self.generic_kinds = saved.kinds;
    }

    /// Emit a tensor extension diagnostic (error code `E0211`).
    fn tensor_error(&mut self, message: &str, span: Span) {
        if !self.suppress_tensor_diagnostics {
            self.diags
                .push(Diagnostic::error(message.to_string(), span).with_code("E0211"));
        }
    }

    /// Unify two tensor types, delegating shape/device unification to the
    /// extension and rendering a provenance-rich diagnostic on mismatch (§9).
    fn unify_tensor_types(&mut self, a: &TensorKind, b: &TensorKind, span: Span) -> Result<(), ()> {
        match (a, b) {
            (TensorKind::Tensor(ta), TensorKind::Tensor(tb)) => {
                match self.tensor_ctx.unify_tensor(ta, tb) {
                    Ok(()) => Ok(()),
                    Err(err) => {
                        self.emit_tensor_unify_error(&err, span);
                        Err(())
                    }
                }
            }
            (TensorKind::TensorDyn(da), TensorKind::TensorDyn(db)) if da == db => Ok(()),
            (TensorKind::TensorAny, TensorKind::TensorAny) => Ok(()),
            _ => {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "tensor type mismatch: expected `{}`, found `{}`",
                            self.tensor_ctx.display_tensor(a),
                            self.tensor_ctx.display_tensor(b)
                        ),
                        span,
                    )
                    .with_code("E0212"),
                );
                Err(())
            }
        }
    }

    fn emit_tensor_unify_error(&mut self, err: &UnifyError, span: Span) {
        let msg = match err {
            UnifyError::DTypeMismatch { expected, found } => format!(
                "tensor element type mismatch: expected `{}`, found `{}`",
                expected.name(),
                found.name()
            ),
            UnifyError::RankMismatch { expected, found } => {
                format!("tensor rank mismatch: expected rank {expected}, found rank {found}")
            }
            UnifyError::DimMismatch {
                axis,
                expected,
                found,
                expected_origin,
                found_origin,
                ..
            } => format!(
                "tensor dimension mismatch at axis {axis}: expected `{}` from {expected_origin}, found `{}` from {found_origin}",
                self.tensor_ctx.display_dim(expected),
                self.tensor_ctx.display_dim(found)
            ),
            UnifyError::DeviceMismatch { expected, found } => {
                format!("tensor device mismatch: expected `{expected}`, found `{found}`")
            }
            UnifyError::RangeMismatch { expected, found } => {
                format!(
                    "tensor value-range mismatch: expected `{expected}`, found `{found}`"
                )
            }
            UnifyError::Arithmetic => "tensor dimension arithmetic overflowed".to_string(),
        };
        let mut diagnostic = Diagnostic::error(msg, span).with_code("E0212");
        if let UnifyError::DimMismatch {
            expected_span,
            found_span,
            ..
        } = err
        {
            if let Some(found) = found_span {
                diagnostic.span = *found;
            }
            if let Some(expected) = expected_span {
                let (line, column) = self.file.line_col(expected.lo);
                diagnostic = diagnostic
                    .with_note(format!("expected dimension originates at {line}:{column}"));
            }
            if let Some(found) = found_span {
                let (line, column) = self.file.line_col(found.lo);
                diagnostic =
                    diagnostic.with_note(format!("found dimension originates at {line}:{column}"));
            }
        }
        self.diags.push(diagnostic);
    }

    fn check_crate(&mut self) {
        let root_file = self.file.clone();
        // Pass 1: Populate item signatures (structs, enums, functions)
        for item in &self.hir.items {
            let item_id = hir::ItemId(
                self.hir
                    .items
                    .iter()
                    .position(|i| std::ptr::eq(i, item))
                    .unwrap() as u32,
            );
            if let Some(item_file) = self.hir.item_files.get(&item_id) {
                self.file = item_file.clone();
            } else {
                self.file = root_file.clone();
            }
            let start_len = self.diags.len();

            match &item.kind {
                hir::ItemKind::Struct { fields, .. } => {
                    let mut fields_ty = HashMap::new();
                    for field in fields {
                        if matches!(self.hir.ty(field.ty).kind, hir::TypeKind::Ref { .. }) {
                            self.diags.push(
                                Diagnostic::error(
                                    "Core v1 does not permit declared reference fields",
                                    field.name,
                                )
                                .with_code("E0001"),
                            );
                        }
                        let ty = self.convert_hir_type(field.ty);
                        fields_ty.insert(self.text(field.name).to_string(), ty);
                    }
                    self.struct_fields.insert(item_id, fields_ty);
                }
                hir::ItemKind::Enum { variants, .. } => {
                    let mut variants_ty = Vec::new();
                    for variant in variants {
                        let fields = match &variant.kind {
                            hir::VariantKind::Unit => VariantFields::Unit,
                            hir::VariantKind::Tuple(types) => {
                                for ty in types {
                                    if matches!(self.hir.ty(*ty).kind, hir::TypeKind::Ref { .. }) {
                                        self.diags.push(
                                            Diagnostic::error(
                                                "Core v1 does not permit declared reference fields",
                                                self.hir.ty(*ty).span,
                                            )
                                            .with_code("E0001"),
                                        );
                                    }
                                }
                                VariantFields::Tuple(
                                    types.iter().map(|&t| self.convert_hir_type(t)).collect(),
                                )
                            }
                            hir::VariantKind::Struct(fields) => {
                                let mut fields_map = HashMap::new();
                                for f in fields {
                                    if matches!(self.hir.ty(f.ty).kind, hir::TypeKind::Ref { .. }) {
                                        self.diags.push(
                                            Diagnostic::error(
                                                "Core v1 does not permit declared reference fields",
                                                f.name,
                                            )
                                            .with_code("E0001"),
                                        );
                                    }
                                    fields_map.insert(
                                        self.text(f.name).to_string(),
                                        self.convert_hir_type(f.ty),
                                    );
                                }
                                VariantFields::Struct(fields_map)
                            }
                        };
                        variants_ty.push(VariantTy {
                            name: self.text(variant.name).to_string(),
                            fields,
                        });
                    }
                    self.enum_variants.insert(item_id, variants_ty);
                }
                hir::ItemKind::Fn(def) => {
                    self.suppress_tensor_diagnostics = true;
                    let saved = self.enter_tensor_param_scope(&def.sig.generics);
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
                    self.exit_tensor_param_scope(saved);
                    self.suppress_tensor_diagnostics = false;
                    self.fn_sigs.insert(item_id, FnSigTy { params, ret });
                }
                hir::ItemKind::Const { ty, .. } => {
                    let const_ty = self.convert_hir_type(*ty);
                    self.const_types.insert(item_id, const_ty);
                }
                hir::ItemKind::Impl { self_ty, items, .. } => {
                    let impl_self_ty = self.convert_hir_type(*self_ty);
                    let previous_self = self.current_self_ty.replace(impl_self_ty);
                    // Register methods of the impl
                    for impl_item in items {
                        if let hir::ImplItem::Fn { def, .. } = impl_item {
                            self.suppress_tensor_diagnostics = true;
                            let saved = self.enter_tensor_param_scope(&def.sig.generics);
                            let _params: Vec<Ty> = def
                                .sig
                                .params
                                .iter()
                                .map(|p| self.convert_hir_type(p.ty))
                                .collect();
                            let _ret = match def.sig.ret {
                                hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
                                hir::RetTy::Ty(t) => self.convert_hir_type(t),
                                hir::RetTy::Never(_) => Ty::Never,
                            };
                            self.exit_tensor_param_scope(saved);
                            self.suppress_tensor_diagnostics = false;
                        }
                    }
                    self.current_self_ty = previous_self;
                }
                _ => {}
            }

            let end_len = self.diags.len();
            for i in start_len..end_len {
                if self.diags[i].file.is_none() {
                    self.diags[i].file = Some(self.file.clone());
                }
            }
        }

        let start_len = self.diags.len();
        self.validate_impl_rules();
        let end_len = self.diags.len();
        for i in start_len..end_len {
            if self.diags[i].file.is_none() {
                self.diags[i].file = Some(root_file.clone());
            }
        }

        // Pass 2: Typecheck bodies & run semantic checks
        for item in &self.hir.items {
            let item_id = hir::ItemId(
                self.hir
                    .items
                    .iter()
                    .position(|i| std::ptr::eq(i, item))
                    .unwrap() as u32,
            );
            if let Some(item_file) = self.hir.item_files.get(&item_id) {
                self.file = item_file.clone();
            } else {
                self.file = root_file.clone();
            }
            let start_len = self.diags.len();

            match &item.kind {
                hir::ItemKind::Fn(def) => {
                    self.check_fn_def(item_id, def);
                }
                hir::ItemKind::Model(def) => {
                    self.check_model_def(item_id, def);
                }
                hir::ItemKind::Impl { self_ty, items, .. } => {
                    let prev_self = self.current_self_ty.take();
                    let prev_assoc = std::mem::take(&mut self.current_assoc_types);
                    self.current_self_ty = Some(self.convert_hir_type(*self_ty));
                    for impl_item in items {
                        if let hir::ImplItem::AssocType { name, ty } = impl_item {
                            let ty = self.convert_hir_type(*ty);
                            self.current_assoc_types
                                .insert(self.text(*name).to_string(), ty);
                        }
                    }
                    for impl_item in items {
                        if let hir::ImplItem::Fn { def, .. } = impl_item {
                            self.check_fn_def(item_id, def);
                        }
                    }
                    self.current_self_ty = prev_self;
                    self.current_assoc_types = prev_assoc;
                }
                hir::ItemKind::Const { value, ty, .. } => {
                    let expected_ty = self.convert_hir_type(*ty);
                    let val_ty = self.check_expr(*value);
                    let _ = self.unify(expected_ty, val_ty, item.span);
                }
                _ => {}
            }

            let end_len = self.diags.len();
            for i in start_len..end_len {
                if self.diags[i].file.is_none() {
                    self.diags[i].file = Some(self.file.clone());
                }
            }
        }

        // Snippet mode check
        if let hir::Root::Snippet { stmts, tail } = &self.hir.root {
            self.file = root_file.clone();
            let mut state = HashSet::new();
            for &stmt_id in stmts {
                self.check_stmt(stmt_id, &mut state);
            }
            if let Some(tail_id) = tail {
                let _tail_ty = self.check_expr(*tail_id);
            }
        }

        self.file = root_file;

        // Pass 3: Check trait bounds
        let bounds = std::mem::take(&mut self.bounds_checks);
        for (concrete_ty, bounds_list, span) in bounds {
            for bound in bounds_list {
                if !self.satisfies_bound(&concrete_ty, &bound) {
                    self.diags.push(
                        Diagnostic::error(
                            format!(
                                "type '{}' does not satisfy trait bound '{}'",
                                self.ty_to_string(&concrete_ty),
                                self.text(bound.path.span)
                            ),
                            span,
                        )
                        .with_code("E0500"),
                    );
                }
            }
        }
    }

    fn validate_impl_rules(&mut self) {
        let mut impls: Vec<(Option<Res>, Ty, HashSet<String>)> = Vec::new();
        let mut copy_types = HashSet::new();
        let mut drop_types = HashSet::new();
        let root_file = self.file.clone();

        for item in &self.hir.items {
            let item_id = hir::ItemId(
                self.hir
                    .items
                    .iter()
                    .position(|i| std::ptr::eq(i, item))
                    .unwrap() as u32,
            );
            if let Some(item_file) = self.hir.item_files.get(&item_id) {
                self.file = item_file.clone();
            } else {
                self.file = root_file.clone();
            }
            let start_len = self.diags.len();

            let hir::ItemKind::Impl {
                trait_,
                self_ty,
                items,
                ..
            } = &item.kind
            else {
                continue;
            };
            let self_ty = self.convert_hir_type(*self_ty);
            let trait_res = trait_.as_ref().map(|trait_ref| trait_ref.res);
            let method_names: HashSet<String> = items
                .iter()
                .filter_map(|item| match item {
                    hir::ImplItem::Fn { def, .. } => Some(self.text(def.sig.name).to_string()),
                    _ => None,
                })
                .collect();

            let self_type_is_local = matches!(&self_ty, Ty::Struct(..) | Ty::Enum(..));
            let trait_is_local = matches!(trait_res, Some(Res::Item(_)));
            if trait_.is_some() && !self_type_is_local && !trait_is_local {
                self.diags.push(
                    Diagnostic::error(
                        "implementation violates the orphan rule: neither trait nor type is local",
                        item.span,
                    )
                    .with_code("E0500"),
                );
            } else if trait_.is_none() && !self_type_is_local {
                self.diags.push(
                    Diagnostic::error("inherent implementations require a local type", item.span)
                        .with_code("E0500"),
                );
            }

            if impls
                .iter()
                .any(|(previous_trait, previous_ty, previous_methods)| {
                    if *previous_trait != trait_res
                        || !self.types_may_overlap(previous_ty, &self_ty)
                    {
                        return false;
                    }
                    trait_res.is_some() || !previous_methods.is_disjoint(&method_names)
                })
            {
                self.diags.push(
                    Diagnostic::error("overlapping implementation for the same type", item.span)
                        .with_code("E0500")
                        .with_label("another applicable impl already exists"),
                );
            }
            impls.push((trait_res, self_ty.clone(), method_names));

            let trait_name = trait_
                .as_ref()
                .map(|trait_ref| self.text(trait_ref.path.span).to_owned());
            if trait_name.as_deref() == Some("Num") {
                self.diags.push(
                    Diagnostic::error(
                        "user types cannot implement compiler-known trait Num",
                        item.span,
                    )
                    .with_code("E0500"),
                );
            }
            if let Ty::Struct(id, _) | Ty::Enum(id, _) = &self_ty {
                match trait_name.as_deref() {
                    Some("Copy") => {
                        copy_types.insert(*id);
                    }
                    Some("Drop") => {
                        drop_types.insert(*id);
                    }
                    _ => {}
                }
            }

            if let Some(hir::TraitRef {
                res: Res::Item(trait_id),
                ..
            }) = trait_
            {
                if let hir::ItemKind::Trait {
                    items: trait_items, ..
                } = &self.hir.item(*trait_id).kind
                {
                    let required: HashSet<String> = trait_items
                        .iter()
                        .filter_map(|item| match item {
                            hir::TraitItem::AssocType { name } => {
                                Some(self.text(*name).to_string())
                            }
                            _ => None,
                        })
                        .collect();
                    let provided: HashSet<String> = items
                        .iter()
                        .filter_map(|item| match item {
                            hir::ImplItem::AssocType { name, .. } => {
                                Some(self.text(*name).to_string())
                            }
                            _ => None,
                        })
                        .collect();
                    let required_methods: HashSet<String> = trait_items
                        .iter()
                        .filter_map(|item| match item {
                            hir::TraitItem::Method { sig, body: None } => {
                                Some(self.text(sig.name).to_string())
                            }
                            _ => None,
                        })
                        .collect();
                    let declared_methods: HashSet<String> = trait_items
                        .iter()
                        .filter_map(|item| match item {
                            hir::TraitItem::Method { sig, .. } => {
                                Some(self.text(sig.name).to_string())
                            }
                            _ => None,
                        })
                        .collect();
                    let provided_methods: HashSet<String> = items
                        .iter()
                        .filter_map(|item| match item {
                            hir::ImplItem::Fn { def, .. } => {
                                Some(self.text(def.sig.name).to_string())
                            }
                            _ => None,
                        })
                        .collect();
                    for missing in required.difference(&provided) {
                        self.diags.push(
                            Diagnostic::error(
                                format!("implementation is missing associated type '{missing}'"),
                                item.span,
                            )
                            .with_code("E0500"),
                        );
                    }
                    for extra in provided.difference(&required) {
                        self.diags.push(
                            Diagnostic::error(
                                format!("associated type '{extra}' is not declared by the trait"),
                                item.span,
                            )
                            .with_code("E0500"),
                        );
                    }
                    for missing in required_methods.difference(&provided_methods) {
                        self.diags.push(
                            Diagnostic::error(
                                format!("implementation is missing method '{missing}'"),
                                item.span,
                            )
                            .with_code("E0500"),
                        );
                    }
                    for extra in provided_methods.difference(&declared_methods) {
                        self.diags.push(
                            Diagnostic::error(
                                format!("method '{extra}' is not declared by the trait"),
                                item.span,
                            )
                            .with_code("E0500"),
                        );
                    }

                    let associated: HashMap<String, TypeId> = items
                        .iter()
                        .filter_map(|item| match item {
                            hir::ImplItem::AssocType { name, ty } => {
                                Some((self.text(*name).to_string(), *ty))
                            }
                            _ => None,
                        })
                        .collect();
                    for trait_item in trait_items {
                        let hir::TraitItem::Method { sig: trait_sig, .. } = trait_item else {
                            continue;
                        };
                        let Some(impl_sig) = items.iter().find_map(|item| match item {
                            hir::ImplItem::Fn { def, .. }
                                if self.text(def.sig.name) == self.text(trait_sig.name) =>
                            {
                                Some(&def.sig)
                            }
                            _ => None,
                        }) else {
                            continue;
                        };
                        if !self.trait_method_signature_matches(
                            trait_sig,
                            impl_sig,
                            &self_ty,
                            &associated,
                        ) {
                            self.diags.push(
                                Diagnostic::error(
                                    format!(
                                        "method '{}' has a signature incompatible with its trait declaration",
                                        self.text(impl_sig.name)
                                    ),
                                    impl_sig.span,
                                )
                                .with_code("E0500"),
                            );
                        }
                    }
                }
            }

            let end_len = self.diags.len();
            for i in start_len..end_len {
                if self.diags[i].file.is_none() {
                    self.diags[i].file = Some(self.file.clone());
                }
            }
        }

        for item_id in copy_types.intersection(&drop_types) {
            let file = self.hir.item_files.get(item_id).cloned().unwrap_or(root_file.clone());
            self.diags.push(
                Diagnostic::error(
                    "a type cannot implement both Copy and Drop",
                    self.hir.item(*item_id).span,
                )
                .with_file(file)
                .with_code("E0500"),
            );
        }

        for item_id in copy_types.iter().copied() {
            let file = self.hir.item_files.get(&item_id).cloned().unwrap_or(root_file.clone());
            let fields: Vec<Ty> = match &self.hir.item(item_id).kind {
                hir::ItemKind::Struct { .. } => self
                    .struct_fields
                    .get(&item_id)
                    .map(|fields| fields.values().cloned().collect())
                    .unwrap_or_default(),
                hir::ItemKind::Enum { .. } => self
                    .enum_variants
                    .get(&item_id)
                    .map(|variants| {
                        variants
                            .iter()
                            .flat_map(|variant| match &variant.fields {
                                VariantFields::Unit => Vec::new(),
                                VariantFields::Tuple(fields) => fields.clone(),
                                VariantFields::Struct(fields) => fields.values().cloned().collect(),
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                _ => Vec::new(),
            };
            if fields
                .iter()
                .any(|field| !is_copy_with_impls(field, &copy_types))
            {
                self.diags.push(
                    Diagnostic::error(
                        "Copy may only be implemented when every field is Copy",
                        self.hir.item(item_id).span,
                    )
                    .with_file(file)
                    .with_code("E0500"),
                );
            }
        }
        self.file = root_file;
    }

    fn trait_method_signature_matches(
        &self,
        trait_sig: &hir::FnSig,
        impl_sig: &hir::FnSig,
        self_ty: &Ty,
        associated: &HashMap<String, TypeId>,
    ) -> bool {
        if trait_sig.receiver != impl_sig.receiver
            || trait_sig.params.len() != impl_sig.params.len()
            || trait_sig.generics.len() != impl_sig.generics.len()
        {
            return false;
        }
        let trait_generics: HashMap<String, usize> = trait_sig
            .generics
            .iter()
            .enumerate()
            .map(|(index, param)| (self.text(param.name).to_string(), index))
            .collect();
        let impl_generics: HashMap<String, usize> = impl_sig
            .generics
            .iter()
            .enumerate()
            .map(|(index, param)| (self.text(param.name).to_string(), index))
            .collect();
        let self_key = format!("{self_ty:?}");
        let params_match =
            trait_sig
                .params
                .iter()
                .zip(&impl_sig.params)
                .all(|(trait_param, impl_param)| {
                    self.signature_type_key(trait_param.ty, &self_key, associated, &trait_generics)
                        == self.signature_type_key(
                            impl_param.ty,
                            &self_key,
                            associated,
                            &impl_generics,
                        )
                });
        params_match
            && match (trait_sig.ret, impl_sig.ret) {
                (hir::RetTy::Unit, hir::RetTy::Unit)
                | (hir::RetTy::Never(_), hir::RetTy::Never(_)) => true,
                (hir::RetTy::Ty(left), hir::RetTy::Ty(right)) => {
                    self.signature_type_key(left, &self_key, associated, &trait_generics)
                        == self.signature_type_key(right, &self_key, associated, &impl_generics)
                }
                _ => false,
            }
    }

    fn signature_type_key(
        &self,
        id: TypeId,
        self_key: &str,
        associated: &HashMap<String, TypeId>,
        generics: &HashMap<String, usize>,
    ) -> String {
        match &self.hir.ty(id).kind {
            hir::TypeKind::Primitive(primitive) => format!("p:{primitive:?}"),
            hir::TypeKind::Path { res, args, .. } => {
                let base = match res {
                    Res::SelfType => self_key.to_string(),
                    Res::SelfAssoc(name) => {
                        let name = self.text(*name);
                        return associated.get(name).map_or_else(
                            || format!("assoc:{name}"),
                            |ty| self.signature_type_key(*ty, self_key, associated, generics),
                        );
                    }
                    Res::TypeParam => generics
                        .get(self.text(self.hir.ty(id).span))
                        .map_or_else(|| "generic:?".to_string(), |index| format!("g:{index}")),
                    Res::Item(item) => format!("item:{}", item.0),
                    Res::Primitive(primitive) => format!("p:{primitive:?}"),
                    Res::CoreType(core) => format!("core:{core:?}"),
                    _ => "error".to_string(),
                };
                let args = args
                    .as_ref()
                    .map(|args| {
                        args.args
                            .iter()
                            .map(|arg| match arg {
                                hir::GenericArg::Type(ty) => {
                                    self.signature_type_key(*ty, self_key, associated, generics)
                                }
                                hir::GenericArg::Const(span) => self.text(*span).to_string(),
                                hir::GenericArg::Shape(shape) => {
                                    let dims: Vec<String> =
                                        shape.dims.iter().map(|d| self.dim_key(d)).collect();
                                    format!("shape[{}]", dims.join(","))
                                }
                                hir::GenericArg::Binding { name, ty } => format!(
                                    "{}={}",
                                    self.text(*name),
                                    self.signature_type_key(*ty, self_key, associated, generics)
                                ),
                            })
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .unwrap_or_default();
                format!("{base}<{args}>")
            }
            hir::TypeKind::Array { elem, len } => format!(
                "array:{}:{}",
                self.signature_type_key(*elem, self_key, associated, generics),
                self.text(*len)
            ),
            hir::TypeKind::Slice(elem) => format!(
                "slice:{}",
                self.signature_type_key(*elem, self_key, associated, generics)
            ),
            hir::TypeKind::Tuple(elems) => format!(
                "tuple:{}",
                elems
                    .iter()
                    .map(|ty| self.signature_type_key(*ty, self_key, associated, generics))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            hir::TypeKind::Ref { mutable, inner } => format!(
                "ref:{mutable}:{}",
                self.signature_type_key(*inner, self_key, associated, generics)
            ),
            hir::TypeKind::Fn { params, ret } => format!(
                "fn:{}->{}",
                params
                    .iter()
                    .map(|ty| self.signature_type_key(*ty, self_key, associated, generics))
                    .collect::<Vec<_>>()
                    .join(","),
                ret.map_or_else(
                    || "unit".to_string(),
                    |ty| self.signature_type_key(ty, self_key, associated, generics)
                )
            ),
            hir::TypeKind::Never => "never".to_string(),
            hir::TypeKind::Error => "error".to_string(),
        }
    }

    fn check_fn_def(&mut self, _item_id: ItemId, def: &hir::FnDef) {
        let sig = &def.sig;

        // `Dim` generic parameters are in scope for every signature type and
        // the body (tensor extension §3.1). No-op for Core-only functions.
        let saved_dims = self.enter_tensor_param_scope(&sig.generics);

        let expected_ret = match sig.ret {
            hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
            hir::RetTy::Ty(t) => self.convert_hir_type(t),
            hir::RetTy::Never(_) => Ty::Never,
        };
        if self.is_unsized_value_type(&expected_ret) {
            self.diags.push(
                Diagnostic::error("unsized return types must be behind a reference", sig.span)
                    .with_code("E0001"),
            );
        }
        self.current_fn_ret = Some(expected_ret.clone());
        self.current_fn_generics = Some(sig.generics.clone());

        // Parameters in local_types
        let mut state = HashSet::new();
        if let Some(receiver) = &sig.receiver {
            let local = sig.receiver_local.expect("lowered receiver has a local ID");
            let self_ty = self.current_self_ty.clone().unwrap_or(Ty::Error);
            let receiver_ty = match receiver {
                hir::Receiver::Value => self_ty,
                hir::Receiver::Ref => Ty::Ref {
                    mutable: false,
                    inner: Box::new(self_ty),
                },
                hir::Receiver::RefMut => Ty::Ref {
                    mutable: true,
                    inner: Box::new(self_ty),
                },
            };
            self.local_types.insert(local, receiver_ty);
            self.local_mutability
                .insert(local, matches!(receiver, hir::Receiver::RefMut));
            state.insert(local);
        }

        for param in &sig.params {
            let ty = self.convert_hir_type(param.ty);
            if self.is_unsized_value_type(&ty) {
                self.diags.push(
                    Diagnostic::error(
                        "unsized parameter types must be behind a reference",
                        param.name,
                    )
                    .with_code("E0001"),
                );
            }
            self.local_types.insert(param.local, ty);
            self.local_mutability.insert(param.local, param.mutable);
            state.insert(param.local);
        }

        let ret_ty = self.check_block(def.body, &mut state);

        // Verify function return paths.
        let resolved_expected_ret = self.resolve(&expected_ret);
        let block = self.hir.block(def.body);
        let control = self.control_summary_block(def.body);
        if resolved_expected_ret == Ty::Never {
            if control.can_complete || control.may_return {
                self.diags.push(
                    Diagnostic::error("function returning '!' may return normally", block.span)
                        .with_code("E0301"),
                );
            }
        } else if resolved_expected_ret != Ty::Primitive(Primitive::Unit)
            && resolved_expected_ret != Ty::Error
            && block.tail.is_none()
            && control.can_complete
        {
            self.diags
                .push(Diagnostic::error("missing return value", block.span).with_code("E0301"));
        }

        if resolved_expected_ret == Ty::Never {
            // Never is a coercion source, not a target that accepts normal completion.
            if ret_ty != Ty::Never && !control.can_complete && !control.may_return {
                // A diverging statement such as `panic();` gives the block a syntactic Unit tail.
                // Its control summary is authoritative, so no unification diagnostic is needed.
            } else if control.can_complete || control.may_return {
                let _ = self.unify(Ty::Error, ret_ty, sig.span);
            }
        } else {
            let _ = self.unify(expected_ret, ret_ty, sig.span);
        }
        self.current_fn_ret = None;
        self.current_fn_generics = None;
        self.exit_tensor_param_scope(saved_dims);
    }

    fn check_block(&mut self, block_id: BlockId, state: &mut HashSet<LocalId>) -> Ty {
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

    fn control_summary_block(&self, block_id: BlockId) -> ControlSummary {
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

    fn control_summary_stmt(&self, stmt_id: StmtId) -> ControlSummary {
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

    fn control_summary_expr(&self, expr_id: ExprId) -> ControlSummary {
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

    fn check_stmt(&mut self, stmt_id: StmtId, state: &mut HashSet<LocalId>) {
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

    fn check_expr(&mut self, expr_id: ExprId) -> Ty {
        let expr = self.hir.expr(expr_id);
        let ty = match &expr.kind {
            hir::ExprKind::Lit(lit) => match lit {
                Lit::Int { suffix, .. } => {
                    if let Some(s) = suffix {
                        Ty::Primitive(convert_int_suffix(*s))
                    } else {
                        Ty::Primitive(Primitive::Int32)
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
                            expr.span,
                        );
                        Ty::Fn {
                            params: instantiated_sig.params,
                            ret: Box::new(instantiated_sig.ret),
                        }
                    } else if let Some(const_ty) = self.const_types.get(item_id) {
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
                    self.associated_fn_type(*item_id, *name, turbofish.as_ref(), expr.span)
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
                Res::Builtin(builtin) => self.builtin_type(*builtin),
                Res::TraitMember(_, _) => Ty::Error,
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
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow => {
                        let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        self.require_operator_bound(&lhs_ty, "Num", expr.span);
                        lhs_ty
                    }
                    BinOp::Eq | BinOp::Ne => {
                        if !self.string_types_comparable(&lhs_ty, &rhs_ty) {
                            let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        }
                        self.require_operator_bound(&lhs_ty, "Eq", expr.span);
                        Ty::Primitive(Primitive::Bool)
                    }
                    BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        if !self.string_types_comparable(&lhs_ty, &rhs_ty) {
                            let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        }
                        self.require_operator_bound(&lhs_ty, "Ord", expr.span);
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
                let source_resolved = self.resolve(&source);
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
                    self.resolve_method(*base, *name, turbofish.as_ref(), args, expr.span)
                } else if let hir::ExprKind::Path {
                    res: Res::TraitMember(trait_id, member),
                    ..
                } = &self.hir.expr(*callee).kind
                {
                    self.check_qualified_trait_call(*trait_id, *member, args, expr.span)
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
                        if let Some(fields) = self.struct_fields.get(&struct_id) {
                            if let Some(field_ty) = fields.get(name_str) {
                                let field_ty = field_ty.clone();
                                let map = self.nominal_param_map(struct_id, &args);
                                self.instantiate_ty(&field_ty, &map)
                            } else {
                                self.diags.push(
                                    Diagnostic::error(
                                        format!("struct field '{}' not found", name_str),
                                        *name,
                                    )
                                    .with_code("E0001"),
                                );
                                Ty::Error
                            }
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
                        Ty::Enum(enum_id, _) => {
                            let name_str = self.text(self.hir.item(enum_id).span);
                            if name_str.contains("Result") || name_str.contains("Option") {
                                ret_ok = true;
                            }
                        }
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

                // 2. Check try expression type
                match self.resolve(&expr_ty) {
                    Ty::Enum(enum_id, args) => {
                        let name_str = self.text(self.hir.item(enum_id).span);
                        if name_str.contains("Result") || name_str.contains("Option") {
                            args.first().cloned().unwrap_or(Ty::Error)
                        } else {
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
                let tys = elems.iter().map(|&e| self.check_expr(e)).collect();
                Ty::Tuple(tys)
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
                if !matches!(&count_ty, Ty::Primitive(p) if is_integer(*p))
                    && !matches!(count_ty, Ty::Error)
                {
                    self.diags.push(
                        Diagnostic::error("array repeat count must be an integer", expr.span)
                            .with_code("E0001"),
                    );
                }

                let count_str = self.text(self.hir.expr(*count).span);
                let len = count_str.parse::<u64>().unwrap_or(0);
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
                    self.check_field_initializers(&expected, &map, fields, expr.span);
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
                        self.check_field_initializers(&expected, &map, fields, expr.span);
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
                let scr_ty = self.check_expr(*scrutinee);
                let ret_ty = self.new_type_var();

                let mut matched_variants = HashSet::new();
                let mut matched_bools = HashSet::new();
                let mut has_wildcard = false;

                for arm in arms {
                    let pat_ty = self.check_pat(arm.pat, scr_ty.clone());
                    let _ = self.unify(scr_ty.clone(), pat_ty, arm.pat.span(self.hir));

                    let pat = self.hir.pat(arm.pat);
                    match &pat.kind {
                        hir::PatKind::Wild | hir::PatKind::Binding { .. } => {
                            has_wildcard = true;
                        }
                        hir::PatKind::Path { res, .. }
                        | hir::PatKind::TupleVariant { res, .. }
                        | hir::PatKind::Struct { res, .. } => {
                            if let Res::Variant(_, variant_idx) = res {
                                matched_variants.insert(*variant_idx);
                            }
                        }
                        hir::PatKind::Lit(Lit::Bool(value)) => {
                            matched_bools.insert(*value);
                        }
                        _ => {}
                    }

                    let body_ty = self.check_expr(arm.body);
                    let _ = self.unify(ret_ty.clone(), body_ty, self.hir.expr(arm.body).span);
                }

                if !has_wildcard {
                    match self.resolve(&scr_ty) {
                        Ty::Enum(enum_id, _) => {
                            if let Some(variants) = self.enum_variants.get(&enum_id) {
                                if matched_variants.len() < variants.len() {
                                    self.diags.push(
                                        Diagnostic::error(
                                            "non-exhaustive pattern match",
                                            expr.span,
                                        )
                                        .with_code("E0303"),
                                    );
                                }
                            }
                        }
                        Ty::Primitive(Primitive::Bool) if matched_bools.len() < 2 => {
                            self.diags.push(
                                Diagnostic::error("non-exhaustive pattern match", expr.span)
                                    .with_code("E0303"),
                            );
                        }
                        _ => {}
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
                let elem_ty = match self.resolve(&iter_ty) {
                    Ty::Range(elem) | Ty::Array(elem, _) | Ty::Slice(elem) => *elem,
                    Ty::Core(CoreType::Vec, args) => args.first().cloned().unwrap_or(Ty::Error),
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

    fn check_pat(&mut self, pat_id: PatId, expected: Ty) -> Ty {
        let pat = self.hir.pat(pat_id);
        match &pat.kind {
            hir::PatKind::Lit(lit) => match lit {
                Lit::Int { suffix, .. } => {
                    if let Some(s) = suffix {
                        Ty::Primitive(convert_int_suffix(*s))
                    } else {
                        Ty::Primitive(Primitive::Int32)
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
            hir::PatKind::Wild => expected,
            hir::PatKind::Binding { local, .. } => {
                self.local_types.insert(*local, expected.clone());
                expected
            }
            hir::PatKind::Path { res, .. } => match res {
                Res::Item(item_id) => {
                    if let Some(const_ty) = self.const_types.get(item_id) {
                        const_ty.clone()
                    } else {
                        Ty::Error
                    }
                }
                Res::Variant(enum_id, _) => {
                    let args = self.nominal_use_args(*enum_id, None, pat.span);
                    Ty::Enum(*enum_id, args)
                }
                _ => Ty::Error,
            },
            hir::PatKind::TupleVariant { res, pats, .. } => {
                if let Res::Variant(enum_id, variant_idx) = res {
                    let args = match self.resolve(&expected) {
                        Ty::Enum(expected_id, args) if expected_id == *enum_id => args,
                        _ => self.nominal_use_args(*enum_id, None, pat.span),
                    };
                    let map = self.nominal_param_map(*enum_id, &args);
                    let tys_opt = self.enum_variants.get(enum_id).and_then(|variants| {
                        let variant = &variants[*variant_idx as usize];
                        if let VariantFields::Tuple(tys) = &variant.fields {
                            Some(tys.clone())
                        } else {
                            None
                        }
                    });
                    if let Some(tys) = tys_opt {
                        for (p, expected_t) in pats.iter().zip(tys) {
                            let expected_t = self.instantiate_ty(&expected_t, &map);
                            let p_ty = self.check_pat(*p, expected_t.clone());
                            let _ = self.unify(expected_t, p_ty, p.span(self.hir));
                        }
                    }
                    Ty::Enum(*enum_id, args)
                } else if let Res::Builtin(builtin) = res {
                    let resolved = self.resolve(&expected);
                    let payload = match (builtin, &resolved) {
                        (Builtin::Some, Ty::Core(CoreType::Option, args)) => args.first().cloned(),
                        (Builtin::Ok, Ty::Core(CoreType::Result, args)) => args.first().cloned(),
                        (Builtin::Err, Ty::Core(CoreType::Result, args)) => args.get(1).cloned(),
                        _ => None,
                    };
                    if let (Some(subpat), Some(payload)) = (pats.first(), payload) {
                        let p_ty = self.check_pat(*subpat, payload.clone());
                        let _ = self.unify(payload, p_ty, subpat.span(self.hir));
                    }
                    resolved
                } else {
                    Ty::Error
                }
            }
            hir::PatKind::Struct { res, fields, .. } => {
                if let Res::Item(struct_id) = res {
                    let args = self.nominal_use_args(*struct_id, None, pat.span);
                    let map = self.nominal_param_map(*struct_id, &args);
                    let expected_fields = self
                        .struct_fields
                        .get(struct_id)
                        .cloned()
                        .unwrap_or_default();
                    for field in fields {
                        let f_name = self.text(field.name);
                        if let Some(expected_f_ty) = expected_fields.get(f_name) {
                            if let Some(sub_pat) = field.pat {
                                let expected_f_ty = self.instantiate_ty(expected_f_ty, &map);
                                let p_ty = self.check_pat(sub_pat, expected_f_ty.clone());
                                let _ = self.unify(expected_f_ty, p_ty, field.name);
                            } else if let Some(local) = field.local {
                                let expected_f_ty = self.instantiate_ty(expected_f_ty, &map);
                                self.local_types.insert(local, expected_f_ty);
                            }
                        }
                    }
                    Ty::Struct(*struct_id, args)
                } else if let Res::Variant(enum_id, variant_idx) = res {
                    let args = match self.resolve(&expected) {
                        Ty::Enum(expected_id, args) if expected_id == *enum_id => args,
                        _ => self.nominal_use_args(*enum_id, None, pat.span),
                    };
                    let map = self.nominal_param_map(*enum_id, &args);
                    let expected_fields = self
                        .enum_variants
                        .get(enum_id)
                        .and_then(|variants| variants.get(*variant_idx as usize))
                        .and_then(|variant| match &variant.fields {
                            VariantFields::Struct(fields) => Some(fields.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    for field in fields {
                        let name = self.text(field.name);
                        if let Some(field_ty) = expected_fields.get(name) {
                            let field_ty = self.instantiate_ty(field_ty, &map);
                            if let Some(subpat) = field.pat {
                                let pat_ty = self.check_pat(subpat, field_ty.clone());
                                let _ = self.unify(field_ty, pat_ty, field.name);
                            } else if let Some(local) = field.local {
                                self.local_types.insert(local, field_ty);
                            }
                        }
                    }
                    Ty::Enum(*enum_id, args)
                } else {
                    Ty::Error
                }
            }
            hir::PatKind::Tuple(elems) => {
                let expected_elems = match self.resolve(&expected) {
                    Ty::Tuple(tys) if tys.len() == elems.len() => tys,
                    _ => (0..elems.len()).map(|_| self.new_type_var()).collect(),
                };
                let tys = elems
                    .iter()
                    .zip(expected_elems)
                    .map(|(&p, ty)| self.check_pat(p, ty))
                    .collect();
                Ty::Tuple(tys)
            }
            hir::PatKind::Array(elems) => {
                let elem_ty = match self.resolve(&expected) {
                    Ty::Array(elem, _) | Ty::Slice(elem) => *elem,
                    _ => self.new_type_var(),
                };
                for &e in elems {
                    let ety = self.check_pat(e, elem_ty.clone());
                    let _ = self.unify(elem_ty.clone(), ety, pat.span);
                }
                Ty::Array(Box::new(elem_ty), elems.len() as u64)
            }
            hir::PatKind::Error => Ty::Error,
        }
    }

    fn instantiate_ty(&self, ty: &Ty, map: &HashMap<String, Ty>) -> Ty {
        match ty {
            Ty::Param(name) => {
                if let Some(target) = map.get(name) {
                    target.clone()
                } else {
                    ty.clone()
                }
            }
            Ty::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.instantiate_ty(inner, map)),
            },
            Ty::Struct(item, args) => Ty::Struct(
                *item,
                args.iter()
                    .map(|arg| self.instantiate_ty(arg, map))
                    .collect(),
            ),
            Ty::Enum(item, args) => Ty::Enum(
                *item,
                args.iter()
                    .map(|arg| self.instantiate_ty(arg, map))
                    .collect(),
            ),
            Ty::Core(core, args) => Ty::Core(
                *core,
                args.iter()
                    .map(|arg| self.instantiate_ty(arg, map))
                    .collect(),
            ),
            Ty::Tuple(elems) => {
                Ty::Tuple(elems.iter().map(|e| self.instantiate_ty(e, map)).collect())
            }
            Ty::Array(elem, len) => Ty::Array(Box::new(self.instantiate_ty(elem, map)), *len),
            Ty::Slice(elem) => Ty::Slice(Box::new(self.instantiate_ty(elem, map))),
            Ty::Fn { params, ret } => Ty::Fn {
                params: params.iter().map(|p| self.instantiate_ty(p, map)).collect(),
                ret: Box::new(self.instantiate_ty(ret, map)),
            },
            Ty::Range(elem) => Ty::Range(Box::new(self.instantiate_ty(elem, map))),
            Ty::Extension(ext) => match &**ext {
                ExtensionTy::Tensor(_) | ExtensionTy::Model(_) | ExtensionTy::ModelError => {
                    ty.clone()
                }
            },
            _ => ty.clone(),
        }
    }

    fn freshen_call_ty(
        &mut self,
        ty: Ty,
        dims: &mut HashMap<DimVar, DimVar>,
        dtypes: &mut HashMap<u32, DType>,
        devices: &mut HashMap<DeviceVar, Device>,
        span: Span,
    ) -> Ty {
        match ty {
            Ty::Extension(ext) => match &*ext {
                ExtensionTy::Tensor(kind) => {
                    match self.tensor_ctx.freshen_tensor(kind, dims, dtypes, devices) {
                        Ok(kind) => Ty::Extension(Box::new(ExtensionTy::Tensor(kind))),
                        Err(error) => {
                            self.emit_tensor_unify_error(&error, span);
                            Ty::Error
                        }
                    }
                }
                ExtensionTy::Model(model) => {
                    Ty::Extension(Box::new(ExtensionTy::Model(model.clone())))
                }
                ExtensionTy::ModelError => Ty::Extension(Box::new(ExtensionTy::ModelError)),
            },
            Ty::Ref { mutable, inner } => Ty::Ref {
                mutable,
                inner: Box::new(self.freshen_call_ty(*inner, dims, dtypes, devices, span)),
            },
            Ty::Tuple(items) => Ty::Tuple(
                items
                    .into_iter()
                    .map(|item| self.freshen_call_ty(item, dims, dtypes, devices, span))
                    .collect(),
            ),
            Ty::Core(core, items) => Ty::Core(
                core,
                items
                    .into_iter()
                    .map(|item| self.freshen_call_ty(item, dims, dtypes, devices, span))
                    .collect(),
            ),
            Ty::Array(item, len) => Ty::Array(
                Box::new(self.freshen_call_ty(*item, dims, dtypes, devices, span)),
                len,
            ),
            Ty::Slice(item) => Ty::Slice(Box::new(
                self.freshen_call_ty(*item, dims, dtypes, devices, span),
            )),
            Ty::Fn { params, ret } => Ty::Fn {
                params: params
                    .into_iter()
                    .map(|param| self.freshen_call_ty(param, dims, dtypes, devices, span))
                    .collect(),
                ret: Box::new(self.freshen_call_ty(*ret, dims, dtypes, devices, span)),
            },
            Ty::Range(item) => Ty::Range(Box::new(
                self.freshen_call_ty(*item, dims, dtypes, devices, span),
            )),
            other => other,
        }
    }

    fn instantiate_sig(
        &mut self,
        item_id: ItemId,
        sig: FnSigTy,
        turbofish: Option<&hir::GenericArgs>,
        span: Span,
    ) -> FnSigTy {
        let item = self.hir.item(item_id);
        let generics = match &item.kind {
            hir::ItemKind::Fn(def) => &def.sig.generics,
            _ => return sig,
        };

        if generics.is_empty() {
            if turbofish.is_some() {
                self.diags.push(
                    Diagnostic::error("generic arguments provided for non-generic function", span)
                        .with_code("E0101"),
                );
            }
            return self.freshen_call_sig(sig, span);
        }

        let mut map = HashMap::new();
        if let Some(args) = turbofish {
            let has_tensor_kind = generics.iter().any(|param| {
                param.bounds.iter().any(|bound| {
                    bound.res == Res::Err
                        && matches!(
                            single_segment_name(&bound.path, self),
                            Some("Dim" | "DType" | "Device")
                        )
                })
            });
            if has_tensor_kind {
                self.tensor_error(
                    "explicit tensor-kind function arguments are reserved for tensor operation typing; use inference here",
                    span,
                );
            }
            if args.args.len() != generics.len() {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "generic parameters mismatch: expected {} generic arguments, found {}",
                            generics.len(),
                            args.args.len()
                        ),
                        span,
                    )
                    .with_code("E0101"),
                );
            }

            for (param, arg) in generics.iter().zip(&args.args) {
                let param_name = self.text(param.name).to_string();
                let arg_ty = match arg {
                    hir::GenericArg::Type(t) => self.convert_hir_type(*t),
                    _ => Ty::Error,
                };

                let trait_bounds = param
                    .bounds
                    .iter()
                    .filter(|bound| bound.res != Res::Err)
                    .cloned()
                    .collect();
                self.bounds_checks
                    .push((arg_ty.clone(), trait_bounds, span));
                map.insert(param_name, arg_ty);
            }
        } else {
            for param in generics {
                let param_name = self.text(param.name).to_string();
                let var = self.new_type_var();
                let trait_bounds = param
                    .bounds
                    .iter()
                    .filter(|bound| bound.res != Res::Err)
                    .cloned()
                    .collect();
                self.bounds_checks.push((var.clone(), trait_bounds, span));
                map.insert(param_name, var);
            }
        }

        // Associated-type equality bindings participate in substitution, so a
        // return such as `I::Item` becomes concrete at each instantiation of
        // `fn first<I: Iterator<Item = Int32>>(...)`.
        for param in generics {
            let param_name = self.text(param.name).to_string();
            for bound in &param.bounds {
                if let Some(args) = &bound.args {
                    for arg in &args.args {
                        if let hir::GenericArg::Binding { name, ty } = arg {
                            let binding = self.convert_hir_type(*ty);
                            let binding = self.instantiate_ty(&binding, &map);
                            map.insert(format!("{param_name}::{}", self.text(*name)), binding);
                        }
                    }
                }
            }
        }

        let params: Vec<Ty> = sig
            .params
            .iter()
            .map(|p| self.instantiate_ty(p, &map))
            .collect();
        let ret = self.instantiate_ty(&sig.ret, &map);

        self.freshen_call_sig(FnSigTy { params, ret }, span)
    }

    fn freshen_call_sig(&mut self, sig: FnSigTy, span: Span) -> FnSigTy {
        let mut dims = HashMap::new();
        let mut dtypes = HashMap::new();
        let mut devices = HashMap::new();
        let params = sig
            .params
            .into_iter()
            .map(|param| self.freshen_call_ty(param, &mut dims, &mut dtypes, &mut devices, span))
            .collect();
        let ret = self.freshen_call_ty(sig.ret, &mut dims, &mut dtypes, &mut devices, span);
        FnSigTy { params, ret }
    }

    fn check_qualified_trait_call(
        &mut self,
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

        let previous_self = self.current_self_ty.replace(receiver_type.clone());
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
        self.current_self_ty = previous_self;
        self.current_assoc_types = previous_assoc;

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

    fn check_tensor_refine(
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
        let Some(generic_args) = turbofish else {
            self.tensor_error("`refine` requires an explicit target shape", name_span);
            return Ty::Error;
        };

        // A `refine` boundary may also assign the initial value range with an
        // optional `range = R` binding; the remaining args are positional.
        let range_arg = generic_args.args.iter().find(
            |a| matches!(a, hir::GenericArg::Binding { name, .. } if self.text(*name) == "range"),
        );
        let range = self.build_value_range(range_arg, generic_args.span);
        let positional: Vec<hir::GenericArg> = generic_args
            .args
            .iter()
            .filter(|a| !matches!(a, hir::GenericArg::Binding { .. }))
            .cloned()
            .collect();

        let (dtype, shape) = match base {
            Ty::Extension(ext) => match &*ext {
                ExtensionTy::Tensor(TensorKind::TensorDyn(dtype)) => match positional.as_slice() {
                    [hir::GenericArg::Shape(shape)] => (*dtype, self.build_refine_shape(shape)),
                    _ => {
                        self.tensor_error(
                            "`TensorDyn<T>::refine` expects exactly one shape argument",
                            generic_args.span,
                        );
                        return Ty::Error;
                    }
                },
                ExtensionTy::Tensor(TensorKind::TensorAny) => match positional.as_slice() {
                    [hir::GenericArg::Type(dtype), hir::GenericArg::Shape(shape)] => (
                        self.tensor_dtype(*dtype, generic_args.span),
                        self.build_refine_shape(shape),
                    ),
                    _ => {
                        self.tensor_error(
                            "`TensorAny::refine` expects a dtype and a shape",
                            generic_args.span,
                        );
                        return Ty::Error;
                    }
                },
                ExtensionTy::Tensor(TensorKind::Tensor(_)) => {
                    self.tensor_error(
                        "`refine` is valid only on `TensorDyn` or `TensorAny`",
                        name_span,
                    );
                    return Ty::Error;
                }
                _ => {
                    self.tensor_error(
                        "`refine` receiver must be `TensorDyn` or `TensorAny`",
                        name_span,
                    );
                    return Ty::Error;
                }
            },
            Ty::Error => return Ty::Error,
            _ => {
                self.tensor_error(
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
                device: self.tensor_ctx.fresh_device(),
                range,
            },
        ))));
        Ty::Core(CoreType::Result, vec![tensor, Ty::Error])
    }

    fn associated_fn_type(
        &mut self,
        nominal: ItemId,
        name_span: Span,
        turbofish: Option<&hir::GenericArgs>,
        call_span: Span,
    ) -> Ty {
        let name = self.text(name_span).to_string();
        let selected = self.hir.items.iter().find_map(|item| {
            let hir::ItemKind::Impl {
                trait_: None,
                self_ty,
                items,
                generics,
            } = &item.kind
            else {
                return None;
            };
            if !matches!(
                &self.hir.ty(*self_ty).kind,
                hir::TypeKind::Path { res: Res::Item(item), .. } if *item == nominal
            ) {
                return None;
            }
            items.iter().find_map(|item| match item {
                hir::ImplItem::Fn { def, .. }
                    if def.sig.receiver.is_none() && self.text(def.sig.name) == name =>
                {
                    Some((def.sig.clone(), *self_ty, generics.clone()))
                }
                _ => None,
            })
        });
        let Some((sig, self_ty_id, impl_generics)) = selected else {
            self.diags.push(
                Diagnostic::error(format!("associated function '{name}' not found"), name_span)
                    .with_code("E0200"),
            );
            return Ty::Error;
        };

        let self_ty = self.convert_hir_type(self_ty_id);
        let previous_self = self.current_self_ty.replace(self_ty);
        let mut params: Vec<Ty> = sig
            .params
            .iter()
            .map(|param| self.convert_hir_type(param.ty))
            .collect();
        let mut ret = match sig.ret {
            hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
            hir::RetTy::Ty(ty) => self.convert_hir_type(ty),
            hir::RetTy::Never(_) => Ty::Never,
        };
        self.current_self_ty = previous_self;

        let mut map = HashMap::new();
        for param in &impl_generics {
            let infer = self.new_type_var();
            map.insert(self.text(param.name).to_string(), infer);
        }
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
        params = params
            .iter()
            .map(|ty| self.instantiate_ty(ty, &map))
            .collect();
        ret = self.instantiate_ty(&ret, &map);
        Ty::Fn {
            params,
            ret: Box::new(ret),
        }
    }

    fn resolve_method(
        &mut self,
        base_expr: ExprId,
        name_span: Span,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        call_span: Span,
    ) -> Ty {
        let base_ty = self.check_expr(base_expr);
        let resolved_base = self.resolve(&base_ty);
        let name_str = self.text(name_span).to_string();

        if self.options.tensor() && name_str == "refine" {
            return self.check_tensor_refine(resolved_base, turbofish, args, name_span, call_span);
        }

        if let Ty::Param(p_name) = &resolved_base {
            if let Some(generics) = &self.current_fn_generics {
                for param in generics {
                    if self.text(param.name) == p_name {
                        for bound in &param.bounds {
                            let bound_trait_name = self.text(bound.path.span);
                            for item in &self.hir.items {
                                if let hir::ItemKind::Trait { name, items, .. } = &item.kind {
                                    if self.text(*name) == bound_trait_name {
                                        for trait_item in items {
                                            if let hir::TraitItem::Method { sig, .. } = trait_item {
                                                if self.text(sig.name) == name_str {
                                                    let params_ty: Vec<Ty> = sig
                                                        .params
                                                        .iter()
                                                        .map(|p| self.convert_hir_type(p.ty))
                                                        .collect();
                                                    let ret_ty = match sig.ret {
                                                        hir::RetTy::Unit => {
                                                            Ty::Primitive(Primitive::Unit)
                                                        }
                                                        hir::RetTy::Ty(t) => {
                                                            self.convert_hir_type(t)
                                                        }
                                                        hir::RetTy::Never(_) => Ty::Never,
                                                    };
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
                                                    for (arg, param_t) in args.iter().zip(params_ty)
                                                    {
                                                        let arg_t = self.check_expr(*arg);
                                                        let _ = self.unify(
                                                            param_t,
                                                            arg_t,
                                                            self.hir.expr(*arg).span,
                                                        );
                                                    }
                                                    return ret_ty;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut receiver_ty = resolved_base.clone();
        while let Ty::Ref { inner, .. } = receiver_ty {
            receiver_ty = self.resolve(&inner);
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

        for item in &self.hir.items {
            if let hir::ItemKind::Impl {
                self_ty: impl_self_ty_id,
                items,
                trait_,
                generics,
                ..
            } = &item.kind
            {
                let impl_self_ty = self.convert_hir_type(*impl_self_ty_id);
                let Some(map) = self.match_impl_type(&impl_self_ty, &receiver_ty, generics) else {
                    continue;
                };

                for impl_item in items {
                    if let hir::ImplItem::Fn { def, .. } = impl_item {
                        let method_name_str = self.text(def.sig.name);
                        if method_name_str == name_str {
                            candidates.push((
                                def,
                                trait_.is_some(),
                                map.clone(),
                                impl_self_ty.clone(),
                            ));
                        }
                    }
                }
            }
        }

        let inherent: Vec<_> = candidates
            .iter()
            .filter(|(_, is_trait, _, _)| !is_trait)
            .collect();
        let selected = if let Some(candidate) = inherent.first() {
            Some((
                candidate.0,
                candidate.1,
                candidate.2.clone(),
                candidate.3.clone(),
            ))
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

        if let Some((def, _, map, impl_self_ty)) = selected {
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
            let previous_self = self.current_self_ty.replace(impl_self_ty);
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
            self.current_self_ty = previous_self;

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
            self.core_method_signature(&receiver_ty, &name_str)
        {
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
                    .with_code("E0303"),
                );
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

    fn core_method_signature(&self, receiver: &Ty, name: &str) -> Option<(Vec<Ty>, Ty, bool)> {
        let unit = Ty::Primitive(Primitive::Unit);
        let bool_ty = Ty::Primitive(Primitive::Bool);
        let u64_ty = Ty::Primitive(Primitive::UInt64);
        let str_ref = Ty::Ref {
            mutable: false,
            inner: Box::new(Ty::Primitive(Primitive::Str)),
        };
        match receiver {
            Ty::Primitive(Primitive::String | Primitive::Str) => match name {
                "len" => Some((Vec::new(), u64_ty, false)),
                "is_empty" => Some((Vec::new(), bool_ty, false)),
                "push" => Some((vec![Ty::Primitive(Primitive::Char)], unit, true)),
                "push_str" => Some((vec![str_ref.clone()], unit, true)),
                "pop" => Some((
                    Vec::new(),
                    Ty::Core(CoreType::Option, vec![Ty::Primitive(Primitive::Char)]),
                    true,
                )),
                "clear" => Some((Vec::new(), unit, true)),
                "as_str" | "trim" => Some((Vec::new(), str_ref, false)),
                "contains" | "starts_with" | "ends_with" => Some((vec![str_ref], bool_ty, false)),
                "find" => Some((
                    vec![str_ref],
                    Ty::Core(CoreType::Option, vec![u64_ty]),
                    false,
                )),
                "replace" => Some((
                    vec![str_ref.clone(), str_ref],
                    Ty::Primitive(Primitive::String),
                    false,
                )),
                "substring" => Some((
                    vec![u64_ty.clone(), u64_ty],
                    Ty::Ref {
                        mutable: false,
                        inner: Box::new(Ty::Primitive(Primitive::Str)),
                    },
                    false,
                )),
                "to_string" | "to_lowercase" | "to_uppercase" => {
                    Some((Vec::new(), Ty::Primitive(Primitive::String), false))
                }
                _ => None,
            },
            Ty::Core(CoreType::Vec, args) => {
                let elem = args.first().cloned().unwrap_or(Ty::Error);
                match name {
                    "push" => Some((vec![elem], unit, true)),
                    "pop" => Some((Vec::new(), Ty::Core(CoreType::Option, vec![elem]), true)),
                    "len" | "capacity" => Some((Vec::new(), u64_ty, false)),
                    "is_empty" => Some((Vec::new(), bool_ty, false)),
                    "get" => Some((
                        vec![u64_ty],
                        Ty::Core(
                            CoreType::Option,
                            vec![Ty::Ref {
                                mutable: false,
                                inner: Box::new(elem),
                            }],
                        ),
                        false,
                    )),
                    "insert" => Some((vec![u64_ty, elem], unit, true)),
                    "remove" => Some((vec![u64_ty], elem, true)),
                    "clear" => Some((Vec::new(), unit, true)),
                    "append" => Some((
                        vec![Ty::Ref {
                            mutable: true,
                            inner: Box::new(receiver.clone()),
                        }],
                        unit,
                        true,
                    )),
                    "as_slice" => Some((
                        Vec::new(),
                        Ty::Ref {
                            mutable: false,
                            inner: Box::new(Ty::Slice(Box::new(elem))),
                        },
                        false,
                    )),
                    _ => None,
                }
            }
            Ty::Core(CoreType::Option, args) => {
                let value = args.first().cloned().unwrap_or(Ty::Error);
                match name {
                    "is_some" | "is_none" => Some((Vec::new(), bool_ty, false)),
                    "unwrap" => Some((Vec::new(), value.clone(), false)),
                    "unwrap_or" => Some((vec![value.clone()], value, false)),
                    _ => None,
                }
            }
            Ty::Core(CoreType::Result, args) => {
                let value = args.first().cloned().unwrap_or(Ty::Error);
                match name {
                    "is_ok" | "is_err" => Some((Vec::new(), bool_ty, false)),
                    "unwrap" => Some((Vec::new(), value.clone(), false)),
                    "unwrap_or" => Some((vec![value.clone()], value, false)),
                    _ => None,
                }
            }
            Ty::Core(CoreType::Box, args) if name == "into_inner" => Some((
                Vec::new(),
                args.first().cloned().unwrap_or(Ty::Error),
                false,
            )),
            _ => None,
        }
    }

    fn match_impl_type(
        &self,
        implementation: &Ty,
        receiver: &Ty,
        generics: &[hir::GenericParam],
    ) -> Option<HashMap<String, Ty>> {
        let mut map = HashMap::new();
        let matches = match (self.resolve(implementation), self.resolve(receiver)) {
            (Ty::Param(name), receiver) => {
                map.insert(name, receiver);
                true
            }
            (Ty::Struct(left, left_args), Ty::Struct(right, right_args))
            | (Ty::Enum(left, left_args), Ty::Enum(right, right_args))
                if left == right && left_args.len() == right_args.len() =>
            {
                left_args.iter().zip(right_args).all(|(left, right)| {
                    if let Ty::Param(name) = left {
                        map.insert(name.clone(), right);
                        true
                    } else {
                        self.types_equal(left, &right)
                    }
                })
            }
            (left, right) => self.types_equal(&left, &right),
        };
        if matches {
            for generic in generics {
                map.entry(self.text(generic.name).to_string())
                    .or_insert_with(|| Ty::Param(self.text(generic.name).to_string()));
            }
            Some(map)
        } else {
            None
        }
    }

    fn is_mutable_place(&self, expr: ExprId) -> bool {
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

    fn types_equal(&self, t1: &Ty, t2: &Ty) -> bool {
        let t1 = self.resolve(t1);
        let t2 = self.resolve(t2);
        match (&t1, &t2) {
            (Ty::Primitive(p1), Ty::Primitive(p2)) => p1 == p2,
            (Ty::Struct(s1, args1), Ty::Struct(s2, args2)) => {
                s1 == s2
                    && args1.len() == args2.len()
                    && args1
                        .iter()
                        .zip(args2)
                        .all(|(left, right)| self.types_equal(left, right))
            }
            (Ty::Enum(e1, args1), Ty::Enum(e2, args2)) => {
                e1 == e2
                    && args1.len() == args2.len()
                    && args1
                        .iter()
                        .zip(args2)
                        .all(|(left, right)| self.types_equal(left, right))
            }
            (Ty::Core(c1, args1), Ty::Core(c2, args2)) => {
                c1 == c2
                    && args1.len() == args2.len()
                    && args1
                        .iter()
                        .zip(args2)
                        .all(|(left, right)| self.types_equal(left, right))
            }
            (
                Ty::Ref {
                    mutable: m1,
                    inner: i1,
                },
                Ty::Ref {
                    mutable: m2,
                    inner: i2,
                },
            ) => m1 == m2 && self.types_equal(i1, i2),
            _ => false,
        }
    }

    fn types_may_overlap(&self, left: &Ty, right: &Ty) -> bool {
        match (self.resolve(left), self.resolve(right)) {
            (Ty::Param(_), _) | (_, Ty::Param(_)) | (Ty::Infer(_), _) | (_, Ty::Infer(_)) => true,
            (Ty::Struct(a, aa), Ty::Struct(b, ba)) | (Ty::Enum(a, aa), Ty::Enum(b, ba)) => {
                a == b
                    && aa.len() == ba.len()
                    && aa
                        .iter()
                        .zip(&ba)
                        .all(|(left, right)| self.types_may_overlap(left, right))
            }
            (Ty::Core(a, aa), Ty::Core(b, ba)) => {
                a == b
                    && aa.len() == ba.len()
                    && aa
                        .iter()
                        .zip(&ba)
                        .all(|(left, right)| self.types_may_overlap(left, right))
            }
            (
                Ty::Ref {
                    mutable: am,
                    inner: ai,
                },
                Ty::Ref {
                    mutable: bm,
                    inner: bi,
                },
            ) => am == bm && self.types_may_overlap(&ai, &bi),
            (left, right) => self.types_equal(&left, &right),
        }
    }

    fn require_operator_bound(&mut self, ty: &Ty, required: &str, span: Span) {
        let ty = self.resolve(ty);
        let satisfied = match &ty {
            Ty::Primitive(primitive) => match required {
                "Num" => is_numeric(*primitive),
                "Eq" | "Ord" => !matches!(primitive, Primitive::Unit),
                _ => false,
            },
            Ty::Ref {
                mutable: false,
                inner,
            } if required == "Eq" || required == "Ord" => {
                matches!(self.resolve(inner), Ty::Primitive(p) if !matches!(p, Primitive::Unit))
            }
            Ty::Param(name) => self.current_fn_generics.as_ref().is_some_and(|params| {
                params.iter().any(|param| {
                    self.text(param.name) == name
                        && param
                            .bounds
                            .iter()
                            .any(|bound| self.text(bound.path.span) == required)
                })
            }),
            Ty::Struct(..) | Ty::Enum(..) => self.hir.items.iter().any(|item| {
                let hir::ItemKind::Impl {
                    trait_: Some(trait_ref),
                    self_ty,
                    ..
                } = &item.kind
                else {
                    return false;
                };
                self.text(trait_ref.path.span) == required
                    && self.types_equal(&self.type_from_hir_without_diagnostics(*self_ty), &ty)
            }),
            Ty::Infer(_) | Ty::Error => true,
            _ => false,
        };
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

    fn type_from_hir_without_diagnostics(&self, id: TypeId) -> Ty {
        match &self.hir.ty(id).kind {
            hir::TypeKind::Primitive(primitive) => Ty::Primitive(*primitive),
            hir::TypeKind::Path {
                res: Res::Item(item),
                ..
            } => match &self.hir.item(*item).kind {
                hir::ItemKind::Struct { .. } => Ty::Struct(*item, Vec::new()),
                hir::ItemKind::Enum { .. } => Ty::Enum(*item, Vec::new()),
                _ => Ty::Error,
            },
            hir::TypeKind::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.type_from_hir_without_diagnostics(*inner)),
            },
            _ => Ty::Error,
        }
    }

    fn satisfies_bound(&mut self, ty: &Ty, bound: &hir::TraitRef) -> bool {
        let ty = self.resolve(ty);
        let bound_name = self.text(bound.path.span).to_string();

        match &ty {
            Ty::Ref {
                mutable: false,
                inner,
            } if bound_name == "Eq" || bound_name == "Ord" => self.satisfies_bound(inner, bound),
            Ty::Primitive(p) => {
                if bound_name == "Num" {
                    is_numeric(*p)
                } else if bound_name == "Eq" || bound_name == "Ord" {
                    !matches!(p, Primitive::Unit)
                } else {
                    false
                }
            }
            Ty::Struct(struct_id, _) | Ty::Enum(struct_id, _) => {
                let associated = self.hir.items.iter().find_map(|item| {
                    let hir::ItemKind::Impl {
                        self_ty: impl_self_ty_id,
                        trait_: Some(trait_ref),
                        items,
                        ..
                    } = &item.kind
                    else {
                        return None;
                    };
                    let same_nominal = matches!(
                        &self.hir.ty(*impl_self_ty_id).kind,
                        hir::TypeKind::Path { res: Res::Item(id), .. } if id == struct_id
                    );
                    if !same_nominal
                        || (trait_ref.res != bound.res
                            && self.text(trait_ref.path.span) != bound_name)
                    {
                        return None;
                    }
                    Some(
                        items
                            .iter()
                            .filter_map(|item| match item {
                                hir::ImplItem::AssocType { name, ty } => {
                                    Some((self.text(*name).to_string(), *ty))
                                }
                                _ => None,
                            })
                            .collect::<HashMap<_, _>>(),
                    )
                });
                let Some(associated) = associated else {
                    return false;
                };
                let bindings_match = bound.args.as_ref().is_none_or(|args| {
                    args.args.iter().all(|arg| match arg {
                        hir::GenericArg::Type(_) => true,
                        hir::GenericArg::Const(_) => true,
                        // Shape args do not appear in Core trait-bound bindings.
                        hir::GenericArg::Shape(_) => true,
                        hir::GenericArg::Binding { name, ty: expected } => {
                            let Some(actual) = associated.get(self.text(*name)).copied() else {
                                return false;
                            };
                            let actual = self.convert_hir_type(actual);
                            let expected = self.convert_hir_type(*expected);
                            self.types_equal(&actual, &expected)
                        }
                    })
                });
                bindings_match
            }
            Ty::Error => true,
            _ => false,
        }
    }

    fn string_types_comparable(&self, left: &Ty, right: &Ty) -> bool {
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
}

impl PatId {
    fn span(&self, hir: &Hir) -> Span {
        hir.pat(*self).span
    }
}

fn convert_int_suffix(suffix: crate::lexer::IntSuffix) -> Primitive {
    match suffix {
        crate::lexer::IntSuffix::I8 => Primitive::Int8,
        crate::lexer::IntSuffix::I16 => Primitive::Int16,
        crate::lexer::IntSuffix::I32 => Primitive::Int32,
        crate::lexer::IntSuffix::I64 => Primitive::Int64,
        crate::lexer::IntSuffix::U8 => Primitive::UInt8,
        crate::lexer::IntSuffix::U16 => Primitive::UInt16,
        crate::lexer::IntSuffix::U32 => Primitive::UInt32,
        crate::lexer::IntSuffix::U64 => Primitive::UInt64,
    }
}

fn convert_float_suffix(suffix: crate::lexer::FloatSuffix) -> Primitive {
    match suffix {
        crate::lexer::FloatSuffix::F32 => Primitive::Float32,
        crate::lexer::FloatSuffix::F64 => Primitive::Float64,
    }
}

fn is_numeric(p: Primitive) -> bool {
    is_integer(p) || matches!(p, Primitive::Float32 | Primitive::Float64)
}

fn is_cast_numeric(p: Primitive) -> bool {
    is_numeric(p) || matches!(p, Primitive::Float16 | Primitive::BFloat16)
}

fn is_integer(p: Primitive) -> bool {
    matches!(
        p,
        Primitive::Int8
            | Primitive::Int16
            | Primitive::Int32
            | Primitive::Int64
            | Primitive::UInt8
            | Primitive::UInt16
            | Primitive::UInt32
            | Primitive::UInt64
    )
}

fn is_copy_primitive(primitive: Primitive) -> bool {
    !matches!(primitive, Primitive::String | Primitive::Str)
}

/// The single-segment name of a path, if it has exactly one segment.
fn single_segment_name<'t>(path: &crate::ast::Path, checker: &'t TypeChecker) -> Option<&'t str> {
    match path.segments.as_slice() {
        [seg] => Some(checker.text(seg.span)),
        _ => None,
    }
}

/// Map a Core primitive to a tensor `DType`, if it is a valid element type.
fn dtype_from_primitive(p: Primitive) -> Option<DType> {
    Some(match p {
        Primitive::Int8 => DType::Int8,
        Primitive::Int16 => DType::Int16,
        Primitive::Int32 => DType::Int32,
        Primitive::Int64 => DType::Int64,
        Primitive::UInt8 => DType::UInt8,
        Primitive::UInt16 => DType::UInt16,
        Primitive::UInt32 => DType::UInt32,
        Primitive::UInt64 => DType::UInt64,
        Primitive::Float32 => DType::Float32,
        Primitive::Float64 => DType::Float64,
        Primitive::Float16 => DType::Float16,
        Primitive::BFloat16 => DType::BFloat16,
        Primitive::Bool => DType::Bool,
        Primitive::Char | Primitive::String | Primitive::Str | Primitive::Unit => return None,
    })
}

fn is_copy_with_impls(ty: &Ty, copy_types: &HashSet<ItemId>) -> bool {
    match ty {
        Ty::Primitive(primitive) => is_copy_primitive(*primitive),
        Ty::Ref { mutable: false, .. } | Ty::Never | Ty::Error => true,
        Ty::Struct(id, args) | Ty::Enum(id, args) => {
            copy_types.contains(id) && args.iter().all(|arg| is_copy_with_impls(arg, copy_types))
        }
        Ty::Core(CoreType::Option | CoreType::Result, args) => {
            args.iter().all(|arg| is_copy_with_impls(arg, copy_types))
        }
        Ty::Core(_, _) => false,
        Ty::Tuple(elements) => elements
            .iter()
            .all(|element| is_copy_with_impls(element, copy_types)),
        Ty::Array(element, _) => is_copy_with_impls(element, copy_types),
        Ty::Infer(_) | Ty::Param(_) => false,
        Ty::Ref { mutable: true, .. } | Ty::Slice(_) | Ty::Fn { .. } | Ty::Range(_) => false,
        Ty::Extension(ext) => match &**ext {
            ExtensionTy::Tensor(tensor) => tensor.is_copy(),
            ExtensionTy::Model(_) => false,
            ExtensionTy::ModelError => false,
        },
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TensorGenericSchema {
    None,
    DTypeAndShape,
    DTypeAndDim,
    Shape,
    Axis,
    AxisStartLen,
    DType,
    Device,
    IndexList,
}

impl TensorGenericSchema {
    const fn arity(self) -> usize {
        match self {
            TensorGenericSchema::None => 0,
            TensorGenericSchema::DTypeAndShape | TensorGenericSchema::DTypeAndDim => 2,
            TensorGenericSchema::Shape
            | TensorGenericSchema::Axis
            | TensorGenericSchema::DType
            | TensorGenericSchema::Device
            | TensorGenericSchema::IndexList => 1,
            TensorGenericSchema::AxisStartLen => 3,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TensorDTypeRule {
    Construct,
    Match,
    Compare,
    Preserve,
    ArgMax,
    Cast,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TensorDeviceRule {
    Fresh,
    Match,
    Preserve,
    Target,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TensorShapeRule {
    Construct,
    FromVec,
    Elementwise,
    BroadcastTo,
    MatMul,
    BatchMatMul,
    Concat,
    Permute,
    Reshape,
    SliceAxis,
    Transpose,
    ReduceAxis,
    FullReduce,
    Softmax,
    Cast,
    ToDevice,
    /// A value-range transition (Gate 7): identity shape/dtype, requires the
    /// receiver to already be in `from`, and produces `to`.
    RangeTransition {
        from: crate::extensions::tensor::types::ValueRange,
        to: crate::extensions::tensor::types::ValueRange,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TensorResultRule {
    Tensor,
    BoolTensor,
    Int64Tensor,
    FallibleTensor,
}

#[derive(Clone, Copy)]
struct TensorOpDescriptor {
    name: &'static str,
    arity: usize,
    standalone: bool,
    method: bool,
    generics: TensorGenericSchema,
    dtype: TensorDTypeRule,
    device: TensorDeviceRule,
    shape: TensorShapeRule,
    result: TensorResultRule,
}

macro_rules! tensor_op {
    ($name:literal, $arity:literal, $method:literal, $generics:expr, $dtype:expr, $device:expr, $shape:expr, $result:expr $(,)?) => {
        TensorOpDescriptor {
            name: $name,
            arity: $arity,
            standalone: true,
            method: $method,
            generics: $generics,
            dtype: $dtype,
            device: $device,
            shape: $shape,
            result: $result,
        }
    };
}

static TENSOR_OPS: &[TensorOpDescriptor] = &[
    tensor_op!(
        "zeros",
        0,
        false,
        TensorGenericSchema::DTypeAndShape,
        TensorDTypeRule::Construct,
        TensorDeviceRule::Fresh,
        TensorShapeRule::Construct,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "ones",
        0,
        false,
        TensorGenericSchema::DTypeAndShape,
        TensorDTypeRule::Construct,
        TensorDeviceRule::Fresh,
        TensorShapeRule::Construct,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "full",
        1,
        false,
        TensorGenericSchema::DTypeAndShape,
        TensorDTypeRule::Construct,
        TensorDeviceRule::Fresh,
        TensorShapeRule::Construct,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "from_vec",
        1,
        false,
        TensorGenericSchema::DTypeAndDim,
        TensorDTypeRule::Construct,
        TensorDeviceRule::Fresh,
        TensorShapeRule::FromVec,
        TensorResultRule::FallibleTensor,
    ),
    tensor_op!(
        "add",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "sub",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "mul",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "div",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "min",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "max",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "eq",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "ne",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "lt",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "le",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "gt",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "ge",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "broadcast_to",
        1,
        true,
        TensorGenericSchema::Shape,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::BroadcastTo,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "matmul",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::MatMul,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "batch_matmul",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::BatchMatMul,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "concat",
        2,
        true,
        TensorGenericSchema::Axis,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Concat,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "permute",
        1,
        true,
        TensorGenericSchema::IndexList,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::Permute,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "reshape",
        1,
        true,
        TensorGenericSchema::Shape,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::Reshape,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "slice_axis",
        1,
        true,
        TensorGenericSchema::AxisStartLen,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::SliceAxis,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "transpose",
        1,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::Transpose,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "sum_axis",
        1,
        true,
        TensorGenericSchema::Axis,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::ReduceAxis,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "mean_axis",
        1,
        true,
        TensorGenericSchema::Axis,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::ReduceAxis,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "argmax",
        1,
        true,
        TensorGenericSchema::Axis,
        TensorDTypeRule::ArgMax,
        TensorDeviceRule::Preserve,
        TensorShapeRule::ReduceAxis,
        TensorResultRule::Int64Tensor,
    ),
    tensor_op!(
        "sum",
        1,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::FullReduce,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "softmax",
        1,
        true,
        TensorGenericSchema::Axis,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::Softmax,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "cast",
        1,
        true,
        TensorGenericSchema::DType,
        TensorDTypeRule::Cast,
        TensorDeviceRule::Preserve,
        TensorShapeRule::Cast,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "to_device",
        1,
        true,
        TensorGenericSchema::Device,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Target,
        TensorShapeRule::ToDevice,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "scale_255",
        1,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::RangeTransition {
            from: crate::extensions::tensor::types::ValueRange::ByteRange,
            to: crate::extensions::tensor::types::ValueRange::UnitRange,
        },
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "normalize",
        1,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::RangeTransition {
            from: crate::extensions::tensor::types::ValueRange::UnitRange,
            to: crate::extensions::tensor::types::ValueRange::Normalized,
        },
        TensorResultRule::Tensor,
    ),
];

/// Why an explicit `broadcast_to` failed: a rank mismatch, or a specific
/// target-aligned axis that cannot be expanded to the target dimension.
enum BroadcastError {
    Rank { source: usize, target: usize },
    Axis { result_axis: usize },
}

impl TypeChecker<'_> {
    fn check_tensor_builtin_call(
        &mut self,
        builtin: Builtin,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        span: Span,
    ) -> Ty {
        let op_name = match builtin {
            Builtin::TensorZeros => "zeros",
            Builtin::TensorOnes => "ones",
            Builtin::TensorFull => "full",
            Builtin::TensorFromVec => "from_vec",
            Builtin::TensorAdd => "add",
            Builtin::TensorSub => "sub",
            Builtin::TensorMul => "mul",
            Builtin::TensorDiv => "div",
            Builtin::TensorMin => "min",
            Builtin::TensorMax => "max",
            Builtin::TensorEq => "eq",
            Builtin::TensorNe => "ne",
            Builtin::TensorLt => "lt",
            Builtin::TensorLe => "le",
            Builtin::TensorGt => "gt",
            Builtin::TensorGe => "ge",
            Builtin::TensorBroadcastTo => "broadcast_to",
            Builtin::TensorMatMul => "matmul",
            Builtin::TensorBatchMatMul => "batch_matmul",
            Builtin::TensorConcat => "concat",
            Builtin::TensorPermute => "permute",
            Builtin::TensorReshape => "reshape",
            Builtin::TensorSliceAxis => "slice_axis",
            Builtin::TensorTranspose => "transpose",
            Builtin::TensorSumAxis => "sum_axis",
            Builtin::TensorMeanAxis => "mean_axis",
            Builtin::TensorArgMax => "argmax",
            Builtin::TensorSum => "sum",
            Builtin::TensorSoftmax => "softmax",
            Builtin::TensorCast => "cast",
            Builtin::TensorToDevice => "to_device",
            Builtin::TensorScale255 => "scale_255",
            Builtin::TensorNormalize => "normalize",
            _ => return Ty::Error,
        };
        self.check_tensor_op(op_name, None, turbofish, args, span)
    }

    fn check_tensor_method_call(
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

    fn get_fix_suggestion(&mut self, expected: &TensorKind, found: &TensorKind) -> Option<String> {
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
            (false, false, true) if self.can_broadcast_to(&found.shape, &expected.shape) => {
                let target = self.tensor_ctx.display_shape(&expected.shape);
                Some(format!(
                    "broadcast the second tensor with `.broadcast_to::<{target}>()`"
                ))
            }
            _ => None,
        }
    }

    fn dtype_to_ty(&self, dtype: DType) -> Ty {
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

    fn extract_const_int(&self, arg: &hir::GenericArg) -> Option<i64> {
        match arg {
            hir::GenericArg::Const(span) => self.text(*span).parse::<i64>().ok(),
            _ => None,
        }
    }

    fn extract_dim_generic(&mut self, arg: &hir::GenericArg, label: &str) -> Option<Poly> {
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

    fn extract_const_int_list(&mut self, arg: &hir::GenericArg) -> Option<Vec<i64>> {
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

    /// Right-aligned NumPy-style broadcast of two shapes. On success returns the
    /// result shape; on failure returns the result-aligned axis at which the two
    /// dimensions are neither provably equal nor a literal `1`.
    fn broadcast_shapes(&mut self, sa: &Shape, sb: &Shape, span: Span) -> Result<Shape, usize> {
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
                    let resolved_a = self
                        .tensor_ctx
                        .resolve_dim(da)
                        .unwrap_or_else(|_| da.clone());
                    let resolved_b = self
                        .tensor_ctx
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
                        // Broadcasting is proof-based: unrelated variables do
                        // not become equal merely because an operation wants
                        // them to. Only equality already established by the
                        // surrounding type constraints is accepted here. Report
                        // the axis aligned to the result shape.
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

    /// Whether `source` can be explicitly broadcast to `target`. On failure
    /// distinguishes a rank mismatch from a specific target-aligned axis that
    /// cannot be expanded.
    fn broadcast_to_check(&mut self, source: &Shape, target: &Shape) -> Result<(), BroadcastError> {
        if source.rank() > target.rank() {
            return Err(BroadcastError::Rank {
                source: source.rank(),
                target: target.rank(),
            });
        }
        for trailing in 0..source.rank() {
            let source_index = source.rank() - 1 - trailing;
            let target_index = target.rank() - 1 - trailing;
            let source_dim = self
                .tensor_ctx
                .resolve_dim(&source.dims[source_index])
                .unwrap_or_else(|_| source.dims[source_index].clone());
            let target_dim = self
                .tensor_ctx
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

    /// Boolean form for callers that only need the yes/no answer (e.g. fix
    /// suggestions).
    fn can_broadcast_to(&mut self, source: &Shape, target: &Shape) -> bool {
        self.broadcast_to_check(source, target).is_ok()
    }

    fn shape_volume(&mut self, shape: &Shape) -> Result<Poly, ()> {
        let mut volume = Poly::constant(1);
        for dimension in &shape.dims {
            let resolved = self.tensor_ctx.resolve_dim(dimension).map_err(|_| ())?;
            volume = volume.mul(&resolved).map_err(|_| ())?;
        }
        Ok(volume)
    }

    fn check_tensor_op(
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

        let get_tensor_kind = |ty: &Ty| -> Option<TensorKind> {
            let resolved = self.resolve(ty);
            let tensor_ty = match resolved {
                Ty::Ref { inner, .. } => self.resolve(&inner),
                other => other,
            };
            match tensor_ty {
                Ty::Extension(ext) => match &*ext {
                    ExtensionTy::Tensor(kind) => Some(kind.clone()),
                    _ => None,
                },
                _ => None,
            }
        };

        if actual_ops.len() != descriptor.arity {
            self.diags.push(
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
            self.diags.push(
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
                if receiver.is_some() && index == 0 {
                    continue;
                }
                if !matches!(self.resolve(operand), Ty::Ref { mutable: false, .. }) {
                    self.diags.push(
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
                        self.diags.push(
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
                    self.diags.push(
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
                    hir::GenericArg::Type(t) => self.tensor_dtype(*t, g_args.span),
                    _ => {
                        self.diags.push(Diagnostic::error(
                            "first generic argument must be a type",
                            g_args.span,
                        ));
                        DType::Float32
                    }
                };
                let shape = match &g_args.args[1] {
                    hir::GenericArg::Shape(s) => self.build_shape(s),
                    _ => {
                        self.diags.push(Diagnostic::error(
                            "second generic argument must be a shape",
                            g_args.span,
                        ));
                        Shape::default()
                    }
                };

                if op_name == "full" {
                    let val_ty = actual_ops[0].clone();
                    let expected_val_ty = self.dtype_to_ty(dtype);
                    let _ = self.unify(expected_val_ty, val_ty, span);
                }

                Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                    TensorTy {
                        dtype,
                        shape,
                        device: self.tensor_ctx.fresh_device(),
                        range: crate::extensions::tensor::types::ValueRange::Unspecified,
                    },
                ))))
            }
            TensorShapeRule::FromVec => {
                let g_args = match turbofish {
                    Some(g) => g,
                    None => {
                        self.diags.push(
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
                    self.diags.push(
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
                    hir::GenericArg::Type(t) => self.tensor_dtype(*t, g_args.span),
                    _ => {
                        self.diags.push(Diagnostic::error(
                            "first generic argument must be a type",
                            g_args.span,
                        ));
                        DType::Float32
                    }
                };
                let dim_poly = match &g_args.args[1] {
                    hir::GenericArg::Shape(s) => {
                        let shape = self.build_shape(s);
                        if shape.dims.len() != 1 {
                            self.diags.push(Diagnostic::error(
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
                let expected_val_ty = Ty::Core(CoreType::Vec, vec![self.dtype_to_ty(dtype)]);
                let _ = self.unify(expected_val_ty, val_ty, span);

                let tensor = Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                    TensorTy {
                        dtype,
                        shape: Shape::new(vec![dim_poly]),
                        device: self.tensor_ctx.fresh_device(),
                        range: crate::extensions::tensor::types::ValueRange::Unspecified,
                    },
                ))));
                Ty::Core(CoreType::Result, vec![tensor, Ty::Error])
            }
            TensorShapeRule::Elementwise => {
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("first argument must be a tensor", span));
                    return Ty::Error;
                };
                let Some(kb) = get_tensor_kind(&actual_ops[1]) else {
                    self.diags
                        .push(Diagnostic::error("second argument must be a tensor", span));
                    return Ty::Error;
                };

                match (&ka, &kb) {
                    (TensorKind::Tensor(ta), TensorKind::Tensor(tb)) => {
                        if self.tensor_ctx.unify_dtype(ta.dtype, tb.dtype).is_err() {
                            let mut diag = Diagnostic::error(
                                format!(
                                    "tensor element type mismatch: expected `{}`, found `{}`",
                                    ta.dtype.name(),
                                    tb.dtype.name()
                                ),
                                span,
                            )
                            .with_code("E0212");
                            if let Some(fix) = self.get_fix_suggestion(&ka, &kb) {
                                diag = diag.with_note(fix);
                            }
                            self.diags.push(diag);
                            return Ty::Error;
                        }
                        if self.tensor_ctx.unify_device(ta.device, tb.device).is_err() {
                            let mut diag = Diagnostic::error(
                                format!(
                                    "tensor device mismatch: expected `{:?}`, found `{:?}`",
                                    ta.device, tb.device
                                ),
                                span,
                            )
                            .with_code("E0212");
                            if let Some(fix) = self.get_fix_suggestion(&ka, &kb) {
                                diag = diag.with_note(fix);
                            }
                            self.diags.push(diag);
                            return Ty::Error;
                        }
                        let out_shape = match self.broadcast_shapes(&ta.shape, &tb.shape, span) {
                            Ok(s) => s,
                            Err(result_axis) => {
                                let lhs = self.tensor_ctx.display_shape(&ta.shape);
                                let rhs = self.tensor_ctx.display_shape(&tb.shape);
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
                                    self.tensor_ctx.dim_origin_notes(&[&ta.shape, &tb.shape])
                                {
                                    diag = diag.with_note(origin);
                                }
                                if let Some(fix) = self.get_fix_suggestion(&ka, &kb) {
                                    diag = diag.with_note(fix);
                                }
                                self.diags.push(diag);
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
                        let out_range = match self.combine_value_range(ta.range, tb.range) {
                            Some(r) => r,
                            None => {
                                self.diags.push(
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
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("argument must be a tensor", span));
                    return Ty::Error;
                };
                let g_args = match turbofish {
                    Some(g) => g,
                    None => {
                        self.diags.push(
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
                    self.diags.push(Diagnostic::error(
                        "`broadcast_to` expects exactly 1 generic argument",
                        g_args.span,
                    ));
                    return Ty::Error;
                }
                let target_shape = match &g_args.args[0] {
                    hir::GenericArg::Shape(s) => self.build_shape(s),
                    _ => {
                        self.diags.push(Diagnostic::error(
                            "generic argument must be a shape",
                            g_args.span,
                        ));
                        Shape::default()
                    }
                };

                match &ka {
                    TensorKind::Tensor(t) => {
                        if let Err(err) = self.broadcast_to_check(&t.shape, &target_shape) {
                            let source = self.tensor_ctx.display_shape(&t.shape);
                            let target = self.tensor_ctx.display_shape(&target_shape);
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
                            for origin in
                                self.tensor_ctx.dim_origin_notes(&[&t.shape, &target_shape])
                            {
                                diag = diag.with_note(origin);
                            }
                            self.diags.push(diag);
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
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("first argument must be a tensor", span));
                    return Ty::Error;
                };
                let Some(kb) = get_tensor_kind(&actual_ops[1]) else {
                    self.diags
                        .push(Diagnostic::error("second argument must be a tensor", span));
                    return Ty::Error;
                };

                match (&ka, &kb) {
                    (TensorKind::Tensor(ta), TensorKind::Tensor(tb)) => {
                        if ta.shape.rank() != 2 {
                            self.diags.push(Diagnostic::error(
                                format!(
                                    "matmul first argument must be rank 2, found rank {}",
                                    ta.shape.rank()
                                ),
                                span,
                            ));
                            return Ty::Error;
                        }
                        if tb.shape.rank() != 2 {
                            self.diags.push(Diagnostic::error(
                                format!(
                                    "matmul second argument must be rank 2, found rank {}",
                                    tb.shape.rank()
                                ),
                                span,
                            ));
                            return Ty::Error;
                        }
                        if self.tensor_ctx.unify_dtype(ta.dtype, tb.dtype).is_err() {
                            self.diags
                                .push(Diagnostic::error("matmul dtype mismatch", span));
                            return Ty::Error;
                        }
                        if self.tensor_ctx.unify_device(ta.device, tb.device).is_err() {
                            self.diags
                                .push(Diagnostic::error("matmul device mismatch", span));
                            return Ty::Error;
                        }
                        if self
                            .tensor_ctx
                            .unify_dim(&ta.shape.dims[1], &tb.shape.dims[0], 0)
                            .is_err()
                        {
                            let lhs = self.tensor_ctx.display_dim(&ta.shape.dims[1]);
                            let rhs = self.tensor_ctx.display_dim(&tb.shape.dims[0]);
                            self.diags.push(
                                Diagnostic::error(
                                    format!(
                                        "matmul inner dimensions mismatch: `{lhs}` and `{rhs}`"
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
                                    tb.shape.dims[1].clone(),
                                ]),
                                device: ta.device,
                                // matmul mixes values across the contracted
                                // axis, so any input value range is no longer
                                // meaningful: the result is Unspecified.
                                range: crate::extensions::tensor::types::ValueRange::Unspecified,
                            },
                        ))))
                    }
                    _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
                }
            }
            TensorShapeRule::BatchMatMul => {
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("first argument must be a tensor", span));
                    return Ty::Error;
                };
                let Some(kb) = get_tensor_kind(&actual_ops[1]) else {
                    self.diags
                        .push(Diagnostic::error("second argument must be a tensor", span));
                    return Ty::Error;
                };

                match (&ka, &kb) {
                    (TensorKind::Tensor(ta), TensorKind::Tensor(tb)) => {
                        if ta.shape.rank() != 3 {
                            self.diags.push(Diagnostic::error(
                                format!(
                                    "batch_matmul first argument must be rank 3, found rank {}",
                                    ta.shape.rank()
                                ),
                                span,
                            ));
                            return Ty::Error;
                        }
                        if tb.shape.rank() != 3 {
                            self.diags.push(Diagnostic::error(
                                format!(
                                    "batch_matmul second argument must be rank 3, found rank {}",
                                    tb.shape.rank()
                                ),
                                span,
                            ));
                            return Ty::Error;
                        }
                        if self.tensor_ctx.unify_dtype(ta.dtype, tb.dtype).is_err() {
                            self.diags
                                .push(Diagnostic::error("batch_matmul dtype mismatch", span));
                            return Ty::Error;
                        }
                        if self.tensor_ctx.unify_device(ta.device, tb.device).is_err() {
                            self.diags
                                .push(Diagnostic::error("batch_matmul device mismatch", span));
                            return Ty::Error;
                        }
                        if self
                            .tensor_ctx
                            .unify_dim(&ta.shape.dims[0], &tb.shape.dims[0], 0)
                            .is_err()
                        {
                            let lhs = self.tensor_ctx.display_dim(&ta.shape.dims[0]);
                            let rhs = self.tensor_ctx.display_dim(&tb.shape.dims[0]);
                            self.diags.push(
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
                        if self
                            .tensor_ctx
                            .unify_dim(&ta.shape.dims[2], &tb.shape.dims[1], 1)
                            .is_err()
                        {
                            let lhs = self.tensor_ctx.display_dim(&ta.shape.dims[2]);
                            let rhs = self.tensor_ctx.display_dim(&tb.shape.dims[1]);
                            self.diags.push(
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
                                range: crate::extensions::tensor::types::ValueRange::Unspecified,
                            },
                        ))))
                    }
                    _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
                }
            }
            TensorShapeRule::Concat => {
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("first argument must be a tensor", span));
                    return Ty::Error;
                };
                let Some(kb) = get_tensor_kind(&actual_ops[1]) else {
                    self.diags
                        .push(Diagnostic::error("second argument must be a tensor", span));
                    return Ty::Error;
                };

                let g_args = match turbofish {
                    Some(g) => g,
                    None => {
                        self.diags.push(
                            Diagnostic::error(
                                "concat requires explicit axis generic argument",
                                span,
                            )
                            .with_code("E0213"),
                        );
                        return Ty::Error;
                    }
                };
                if g_args.args.len() != 1 {
                    self.diags.push(Diagnostic::error(
                        "concat expects exactly 1 generic argument",
                        g_args.span,
                    ));
                    return Ty::Error;
                }
                let axis = match self.extract_const_int(&g_args.args[0]) {
                    Some(a) => a,
                    None => {
                        self.diags.push(Diagnostic::error(
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
                            self.diags.push(Diagnostic::error(
                                "concat tensors must have equal rank",
                                span,
                            ));
                            return Ty::Error;
                        }
                        if axis < 0 || axis >= rank as i64 {
                            self.diags.push(Diagnostic::error(
                                format!("concat axis {} is out of range for rank {}", axis, rank),
                                g_args.span,
                            ));
                            return Ty::Error;
                        }
                        if self.tensor_ctx.unify_dtype(ta.dtype, tb.dtype).is_err() {
                            self.diags
                                .push(Diagnostic::error("concat dtype mismatch", span));
                            return Ty::Error;
                        }
                        if self.tensor_ctx.unify_device(ta.device, tb.device).is_err() {
                            self.diags
                                .push(Diagnostic::error("concat device mismatch", span));
                            return Ty::Error;
                        }
                        let mut out_dims = Vec::new();
                        for i in 0..rank {
                            if i as i64 == axis {
                                let sum_dim = match ta.shape.dims[i].add(&tb.shape.dims[i]) {
                                    Ok(d) => d,
                                    Err(_) => {
                                        self.diags.push(Diagnostic::error(
                                            "concat dimension overflow",
                                            span,
                                        ));
                                        return Ty::Error;
                                    }
                                };
                                out_dims.push(sum_dim);
                            } else {
                                if self
                                    .tensor_ctx
                                    .unify_dim(&ta.shape.dims[i], &tb.shape.dims[i], i)
                                    .is_err()
                                {
                                    self.diags.push(Diagnostic::error(
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
                        let out_range = match self.combine_value_range(ta.range, tb.range) {
                            Some(r) => r,
                            None => {
                                self.diags.push(
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
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("receiver must be a tensor", span));
                    return Ty::Error;
                };
                let g_args = match turbofish {
                    Some(g) => g,
                    None => {
                        self.diags.push(
                            Diagnostic::error("permute requires explicit target index list", span)
                                .with_code("E0213"),
                        );
                        return Ty::Error;
                    }
                };
                if g_args.args.len() != 1 {
                    self.diags.push(Diagnostic::error(
                        "permute expects exactly 1 generic argument",
                        g_args.span,
                    ));
                    return Ty::Error;
                }
                let permutation = match self.extract_const_int_list(&g_args.args[0]) {
                    Some(p) => p,
                    None => {
                        self.diags.push(Diagnostic::error(
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
                            self.diags.push(Diagnostic::error(
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
                                self.diags.push(Diagnostic::error(
                                    format!("index {} is out of range for rank {}", idx, rank),
                                    g_args.span,
                                ));
                                return Ty::Error;
                            }
                            if !seen.insert(idx) {
                                self.diags.push(Diagnostic::error(
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
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("receiver must be a tensor", span));
                    return Ty::Error;
                };
                let g_args = match turbofish {
                    Some(g) => g,
                    None => {
                        self.diags.push(
                            Diagnostic::error("reshape requires explicit target shape", span)
                                .with_code("E0213"),
                        );
                        return Ty::Error;
                    }
                };
                if g_args.args.len() != 1 {
                    self.diags.push(Diagnostic::error(
                        "reshape expects exactly 1 generic argument",
                        g_args.span,
                    ));
                    return Ty::Error;
                }
                let target_shape = match &g_args.args[0] {
                    hir::GenericArg::Shape(s) => self.build_shape(s),
                    _ => {
                        self.diags.push(Diagnostic::error(
                            "generic argument must be a shape",
                            g_args.span,
                        ));
                        Shape::default()
                    }
                };

                match &ka {
                    TensorKind::Tensor(t) => {
                        let (source_volume, target_volume) = match (
                            self.shape_volume(&t.shape),
                            self.shape_volume(&target_shape),
                        ) {
                            (Ok(source), Ok(target)) => (source, target),
                            _ => {
                                self.diags.push(
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
                            let source_shape = self.tensor_ctx.display_shape(&t.shape);
                            let target_display = self.tensor_ctx.display_shape(&target_shape);
                            let source_product = self.tensor_ctx.shape_product_display(&t.shape);
                            let target_product =
                                self.tensor_ctx.shape_product_display(&target_shape);
                            let mut diag =
                                Diagnostic::error("reshape cannot preserve element count", span)
                                    .with_code("E0212")
                                    .with_note(format!("source shape: {source_shape}"))
                                    .with_note(format!("target shape: {target_display}"))
                                    .with_note(format!(
                                        "required: {source_product} == {target_product}"
                                    ));
                            for origin in
                                self.tensor_ctx.dim_origin_notes(&[&t.shape, &target_shape])
                            {
                                diag = diag.with_note(origin);
                            }
                            self.diags.push(diag);
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
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("receiver must be a tensor", span));
                    return Ty::Error;
                };
                let g_args = match turbofish {
                    Some(g) => g,
                    None => {
                        self.diags.push(
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
                    self.diags.push(
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
                let axis = match self.extract_const_int(&g_args.args[0]) {
                    Some(a) => a,
                    None => {
                        self.diags.push(Diagnostic::error(
                            "AXIS must be a constant integer",
                            g_args.span,
                        ));
                        return Ty::Error;
                    }
                };
                let Some(start) = self.extract_dim_generic(&g_args.args[1], "START") else {
                    return Ty::Error;
                };
                let Some(len) = self.extract_dim_generic(&g_args.args[2], "LEN") else {
                    return Ty::Error;
                };

                match &ka {
                    TensorKind::Tensor(t) => {
                        let rank = t.shape.rank();
                        if axis < 0 || axis >= rank as i64 {
                            self.diags.push(Diagnostic::error(
                                format!("axis {} out of range for rank {}", axis, rank),
                                g_args.span,
                            ));
                            return Ty::Error;
                        }
                        let axis_len = self
                            .tensor_ctx
                            .resolve_dim(&t.shape.dims[axis as usize])
                            .unwrap_or_else(|_| t.shape.dims[axis as usize].clone());
                        let start = self.tensor_ctx.resolve_dim(&start).unwrap_or(start);
                        let len = self.tensor_ctx.resolve_dim(&len).unwrap_or(len);
                        let end = match start.add(&len) {
                            Ok(end) => end,
                            Err(_) => {
                                self.diags.push(
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
                            self.diags.push(
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
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("receiver must be a tensor", span));
                    return Ty::Error;
                };
                let g_args = match turbofish {
                    Some(g) => g,
                    None => {
                        self.diags.push(
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
                    self.diags.push(Diagnostic::error(
                        format!("`{}` expects exactly 1 generic argument", op_name),
                        g_args.span,
                    ));
                    return Ty::Error;
                }
                let axis = match self.extract_const_int(&g_args.args[0]) {
                    Some(a) => a,
                    None => {
                        self.diags.push(Diagnostic::error(
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
                            self.diags.push(Diagnostic::error(
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
                                    range:
                                        crate::extensions::tensor::types::ValueRange::Unspecified,
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
                                    range:
                                        crate::extensions::tensor::types::ValueRange::Unspecified,
                                },
                            ))))
                        }
                    }
                    _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
                }
            }
            TensorShapeRule::FullReduce => {
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
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
                            range: crate::extensions::tensor::types::ValueRange::Unspecified,
                        }),
                    ))),
                    _ => Ty::Extension(Box::new(ExtensionTy::Tensor(ka.clone()))),
                }
            }
            TensorShapeRule::Cast => {
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("receiver must be a tensor", span));
                    return Ty::Error;
                };
                let g_args = match turbofish {
                    Some(g) => g,
                    None => {
                        self.diags.push(
                            Diagnostic::error("cast requires explicit target type", span)
                                .with_code("E0213"),
                        );
                        return Ty::Error;
                    }
                };
                if g_args.args.len() != 1 {
                    self.diags.push(Diagnostic::error(
                        "cast expects exactly 1 generic argument",
                        g_args.span,
                    ));
                    return Ty::Error;
                }
                let target_dtype = match &g_args.args[0] {
                    hir::GenericArg::Type(t) => self.tensor_dtype(*t, g_args.span),
                    _ => {
                        self.diags.push(Diagnostic::error(
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
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("receiver must be a tensor", span));
                    return Ty::Error;
                };
                match &ka {
                    TensorKind::Tensor(t) => {
                        // The transition operations are defined on Float32 values.
                        if !matches!(t.dtype, TDType::Float32 | TDType::Var(_)) {
                            self.diags.push(
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
                            self.diags.push(
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
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("receiver must be a tensor", span));
                    return Ty::Error;
                };
                let g_args = match turbofish {
                    Some(g) => g,
                    None => {
                        self.diags.push(
                            Diagnostic::error("to_device requires explicit target device", span)
                                .with_code("E0213"),
                        );
                        return Ty::Error;
                    }
                };
                if g_args.args.len() != 1 {
                    self.diags.push(Diagnostic::error(
                        "to_device expects exactly 1 generic argument",
                        g_args.span,
                    ));
                    return Ty::Error;
                }
                let target_device = self.build_device(Some(&g_args.args[0]), g_args.span);

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
                let Some(ka) = get_tensor_kind(&actual_ops[0]) else {
                    self.diags
                        .push(Diagnostic::error("receiver must be a tensor", span));
                    return Ty::Error;
                };

                match &ka {
                    TensorKind::Tensor(t) => {
                        let rank = t.shape.rank();
                        if rank != 2 {
                            self.diags.push(Diagnostic::error(
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

    fn check_model_def(&mut self, _item_id: ItemId, def: &hir::ModelDef) {
        if !self.options.tensor() {
            self.diags.push(Diagnostic::error(
                "model declarations require `--extension tensor` to be enabled",
                def.name,
            ));
            return;
        }

        let mut inputs_count = 0;
        let mut outputs_count = 0;
        let mut port_names = HashSet::new();

        let saved = self.enter_tensor_param_scope(&def.generics);

        // Verify generic parameter kinds
        for g in &def.generics {
            let kind = self.generic_kind(g);
            if kind != GenericKind::Dim {
                self.diags.push(
                    Diagnostic::error(
                        "model generic parameters must have kind `Dim` (e.g. `<N: Dim>`)",
                        g.name,
                    )
                    .with_code("E0211"),
                );
            }
        }

        for port in &def.ports {
            let name = self.text(port.name).to_string();
            if !port_names.insert(name.clone()) {
                self.diags.push(
                    Diagnostic::error(format!("duplicate port name `{}`", name), port.name)
                        .with_code("E0211"),
                );
            }

            match port.dir {
                crate::ast::PortDir::Input => inputs_count += 1,
                crate::ast::PortDir::Output => outputs_count += 1,
            }

            let ty = self.convert_hir_type(port.ty);
            match self.resolve(&ty) {
                Ty::Extension(ext) => match ext.as_ref() {
                    ExtensionTy::Tensor(TensorKind::Tensor(_) | TensorKind::TensorDyn(_)) => {
                        // Valid
                    }
                    _ => {
                        self.diags.push(
                            Diagnostic::error(
                                format!("invalid port type `{}`: models allow only `Tensor` and `TensorDyn` ports", self.ty_to_string(&ty)),
                                port.span,
                            )
                            .with_code("E0211"),
                        );
                    }
                },
                Ty::Error => {}
                _ => {
                    self.diags.push(
                        Diagnostic::error(
                            format!("invalid port type `{}`: models allow only `Tensor` and `TensorDyn` ports", self.ty_to_string(&ty)),
                            port.span,
                        )
                        .with_code("E0211"),
                    );
                }
            }
        }

        if inputs_count == 0 {
            self.diags.push(
                Diagnostic::error("model must declare at least one input port", def.name)
                    .with_code("E0211"),
            );
        }
        if outputs_count == 0 {
            self.diags.push(
                Diagnostic::error("model must declare at least one output port", def.name)
                    .with_code("E0211"),
            );
        }

        self.exit_tensor_param_scope(saved);
    }

    fn check_model_method_call(
        &mut self,
        model: &ModelTy,
        name: &str,
        args: &[ExprId],
        name_span: Span,
        call_span: Span,
    ) -> Ty {
        if name != "predict" {
            self.diags.push(Diagnostic::error(
                format!("model type has no method named `{}`", name),
                name_span,
            ));
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

        if args.len() != inputs.len() {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "wrong number of arguments for `.predict(...)`: expected {}, found {}",
                        inputs.len(),
                        args.len()
                    ),
                    call_span,
                )
                .with_code("E0005"),
            );
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

        for (arg_expr_id, (expected_port_ty, port_decl_span)) in
            args.iter().zip(instantiated_inputs)
        {
            let arg_ty = self.check_expr(*arg_expr_id);
            match self.resolve(&arg_ty) {
                Ty::Ref { inner, .. } => {
                    let diagnostic_count = self.diags.len();
                    if self
                        .unify(
                            expected_port_ty.clone(),
                            *inner.clone(),
                            self.hir.expr(*arg_expr_id).span,
                        )
                        .is_err()
                    {
                        let (line, column) = self.file.line_col(port_decl_span.lo);
                        if let Some(diagnostic) = self.diags.get_mut(diagnostic_count) {
                            diagnostic.notes.push(format!(
                                "corresponding model port declared at {}:{line}:{column}",
                                self.file.name
                            ));
                        }
                    }
                }
                _ => {
                    self.diags.push(
                        Diagnostic::error(
                            format!("mismatched types: expected a borrowed tensor (e.g. `&tensor`), found `{}`", self.ty_to_string(&arg_ty)),
                            self.hir.expr(*arg_expr_id).span,
                        )
                        .with_code("E0005"),
                    );
                }
            }
        }

        if instantiated_outputs.len() == 1 {
            instantiated_outputs[0].clone()
        } else {
            Ty::Tuple(instantiated_outputs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse, ParseMode};
    use crate::resolve::resolve;

    fn check_src(src: &str) -> Vec<Diagnostic> {
        let file = Arc::new(SourceFile::new("test.stark".to_string(), src.to_string()));
        let (tree, diags) = parse(&file, ParseMode::Program);
        assert!(diags.is_empty(), "parse failed: {:?}", diags);
        let (hir, sem_diags) = resolve(&tree, file.clone());
        let mut all_diags = sem_diags.clone();
        let mut type_diags = check(&hir, file);
        all_diags.append(&mut type_diags);
        all_diags
    }

    /// Parse, resolve, and type-check a program with the `tensor` extension.
    fn check_tensor(src: &str) -> Vec<Diagnostic> {
        use crate::options::LanguageOptions;
        let opts = LanguageOptions::with_tensor();
        let file = Arc::new(SourceFile::new("test.stark".to_string(), src.to_string()));
        let (tree, diags) = crate::parser::parse_with_options(&file, ParseMode::Program, opts);
        assert!(diags.is_empty(), "parse failed: {:?}", diags);
        let (hir, sem) = crate::resolve::resolve_with_options(&tree, file.clone(), opts);
        let mut all = sem;
        all.extend(check_with_options(&hir, file, opts));
        all
    }

    fn tensor_msgs(src: &str) -> Vec<String> {
        check_tensor(src)
            .iter()
            .map(|d| d.message.clone())
            .collect()
    }

    #[test]
    fn tensor_signature_checks_clean() {
        // A symbolic-batch signature returning its argument type-checks.
        let d = check_tensor(
            "fn scale<N: Dim>(x: Tensor<Float32, [N, 3]>) -> Tensor<Float32, [N, 3]> { x }",
        );
        assert!(
            d.is_empty(),
            "unexpected: {:?}",
            tensor_msgs(
                "fn scale<N: Dim>(x: Tensor<Float32, [N, 3]>) -> Tensor<Float32, [N, 3]> { x }"
            )
        );
    }

    #[test]
    fn tensor_generic_kinds_and_cuda_check_clean() {
        let src = "fn identity<T: DType, N: Dim, D: Device>(x: Tensor<T, [N], device = D>) -> Tensor<T, [N], device = D> { x }\nfn gpu(x: Tensor<Float32, [1], device = Cuda<0>>) { }";
        let diagnostics = check_tensor(src);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
    }

    #[test]
    fn tensor_kind_device_and_arity_errors_are_reported_once() {
        let cases = [
            "fn bad<B: Dim>(x: B) { }",
            "fn bad<B: Dim + Copy>(x: Int32) { }",
            "fn bad(x: Tensor<Float32, [1], device = String>) { }",
            "fn bad(x: TensorDyn<Float32, Int32>) { }",
            "fn bad(x: TensorAny<Int32>) { }",
        ];
        for src in cases {
            let diagnostics = check_tensor(src);
            let tensor_errors = diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.code.as_deref() == Some("E0211"))
                .count();
            assert_eq!(tensor_errors, 1, "{src}: {diagnostics:?}");
        }
    }

    #[test]
    fn scalar_half_types_are_restricted_but_casts_are_allowed() {
        assert!(!check_tensor("fn bad(x: Float16) { }").is_empty());
        let diagnostics =
            check_tensor("fn cast(x: Float32) -> Float32 { let y = x as Float16; y as Float32 }");
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
    }

    #[test]
    fn refine_binds_existential_dims_for_the_rest_of_the_block() {
        let source = "fn accept<N: Dim>(x: Tensor<UInt8, [N, 3]>) { }\nfn handle(request: TensorAny) -> Result<Int32, String> { let images = request.refine::<UInt8, [B, 3]>()?; accept(images); Ok(0) }";
        let diagnostics = check_tensor(source);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
    }

    #[test]
    fn refine_existential_dims_do_not_escape_their_block() {
        let source = "fn handle(request: TensorAny) -> Result<Int32, String> { { let images = request.refine::<UInt8, [B]>()?; } let outside: Tensor<UInt8, [B]>; Ok(0) }";
        let messages = tensor_msgs(source);
        assert!(
            messages
                .iter()
                .any(|message| message.contains("undeclared dimension variable `B`")),
            "{messages:?}"
        );
    }

    #[test]
    fn generic_tensor_calls_are_fresh_and_infer_independently() {
        let source = "fn identity<T: DType, N: Dim>(x: Tensor<T, [N]>) -> Tensor<T, [N]> { x }\nfn calls(a: Tensor<Float32, [4]>, b: Tensor<UInt8, [7]>) { let x: Tensor<Float32, [4]> = identity(a); let y: Tensor<UInt8, [7]> = identity(b); }";
        let diagnostics = check_tensor(source);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
    }

    #[test]
    fn refine_consumes_its_dynamic_tensor_receiver() {
        let source = "fn handle(request: TensorAny) -> Result<Int32, String> { let first = request.refine::<UInt8, [B]>()?; let second = request.refine::<UInt8, [C]>()?; Ok(0) }";
        let messages = tensor_msgs(source);
        assert!(
            messages.iter().any(|message| message.contains("moved")),
            "{messages:?}"
        );
    }

    #[test]
    fn refine_reuses_bound_symbols_and_distinguishes_new_ones() {
        let same = "fn pair<N: Dim>(a: Tensor<UInt8, [N]>, b: Tensor<UInt8, [N]>) { } fn handle(first: TensorAny, second: TensorAny) -> Result<Int32, String> { let a = first.refine::<UInt8, [B]>()?; let b = second.refine::<UInt8, [B]>()?; pair(a, b); Ok(0) }";
        let diagnostics = check_tensor(same);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");

        let distinct = "fn pair<N: Dim>(a: Tensor<UInt8, [N]>, b: Tensor<UInt8, [N]>) { } fn handle(first: TensorAny, second: TensorAny) -> Result<Int32, String> { let a = first.refine::<UInt8, [B]>()?; let b = second.refine::<UInt8, [C]>()?; pair(a, b); Ok(0) }";
        let messages = tensor_msgs(distinct);
        assert!(
            messages
                .iter()
                .any(|message| message.contains("dimension mismatch")),
            "{messages:?}"
        );
    }

    #[test]
    fn tensor_dtype_mismatch_is_reported() {
        let msgs = tensor_msgs(
            "fn f(a: Tensor<Float32, [4, 4]>) -> Unit { let b: Tensor<Float16, [4, 4]> = a; }",
        );
        assert!(
            msgs.iter().any(|m| m.contains("element type mismatch")),
            "{msgs:?}"
        );
    }

    #[test]
    fn tensor_dimension_mismatch_reports_axis_and_values() {
        let source =
            "fn f(a: Tensor<Float32, [4, 8]>) -> Unit { let b: Tensor<Float32, [4, 16]> = a; }";
        let diagnostics = check_tensor(source);
        let msgs = diagnostics
            .iter()
            .map(|diagnostic| diagnostic.message.clone())
            .collect::<Vec<_>>();
        assert!(
            msgs.iter()
                .any(|m| m.contains("dimension mismatch at axis 1")
                    && m.contains("16")
                    && m.contains('8')
                    && m.contains("literal dimension")),
            "{msgs:?}"
        );
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.notes.len() == 2),
            "{diagnostics:?}"
        );
    }

    #[test]
    fn tensor_rank_mismatch_is_reported() {
        let msgs = tensor_msgs(
            "fn f(a: Tensor<Float32, [4, 4]>) -> Unit { let b: Tensor<Float32, [4]> = a; }",
        );
        assert!(msgs.iter().any(|m| m.contains("rank mismatch")), "{msgs:?}");
    }

    #[test]
    fn undeclared_dimension_is_reported() {
        let msgs = tensor_msgs("fn f(x: Tensor<Float32, [B, 3]>) -> Unit {}");
        assert!(
            msgs.iter()
                .any(|m| m.contains("undeclared dimension variable `B`")),
            "{msgs:?}"
        );
    }

    #[test]
    fn negative_dimension_is_rejected() {
        let msgs = tensor_msgs("fn f<N: Dim>(x: Tensor<Float32, [N - 1]>) -> Unit {}");
        assert!(
            msgs.iter().any(|m| m.contains("may be negative")),
            "{msgs:?}"
        );
    }

    #[test]
    fn tensor_is_not_copy() {
        // Moving a tensor twice is a use-after-move: tensors are Move (§4.2).
        let msgs =
            tensor_msgs("fn use2(a: Tensor<Float32, [4]>) -> Unit { let b = a; let c = a; }");
        assert!(
            msgs.iter().any(|m| m.to_lowercase().contains("move")),
            "expected a move error, got {msgs:?}"
        );
    }

    #[test]
    fn tensor_op_elementwise_checks() {
        let diagnostics = check_tensor(
            "fn f(a: Tensor<Float32, [4, 4]>, b: Tensor<Float16, [4, 4]>) -> Unit {
                let c = add(&a, &b);
            }",
        );
        let msgs = diagnostics
            .iter()
            .map(|d| d.message.clone())
            .collect::<Vec<_>>();
        assert!(
            msgs.iter().any(|m| m.contains("element type mismatch")),
            "{msgs:?}"
        );
        assert!(
            diagnostics
                .iter()
                .any(|d| d.notes.iter().any(|n| n.contains(".cast::<Float32>()"))),
            "{diagnostics:?}"
        );
    }

    #[test]
    fn tensor_mixed_rank_broadcasting_is_safe_and_directional() {
        let valid = check_tensor(
            "fn f(a: Tensor<Float32, [2, 3]>, b: Tensor<Float32, [3]>) -> Tensor<Float32, [2, 3]> { add(&a, &b) }",
        );
        assert!(valid.is_empty(), "{valid:?}");

        let invalid =
            tensor_msgs("fn f(a: Tensor<Float32, [2, 3]>) { let x = broadcast_to::<[3]>(&a); }");
        assert!(
            invalid
                .iter()
                .any(|message| message.contains("cannot `broadcast_to` the target shape")),
            "{invalid:?}"
        );
    }

    #[test]
    fn tensor_broadcasting_does_not_unify_unrelated_symbols() {
        let messages = tensor_msgs(
            "fn f<B: Dim, C: Dim>(a: Tensor<Float32, [B]>, b: Tensor<Float32, [C]>) { let x = add(&a, &b); }",
        );
        assert!(
            messages
                .iter()
                .any(|message| message.contains("tensor shapes cannot be broadcast together")),
            "{messages:?}"
        );
    }

    #[test]
    fn tensor_reshape_requires_polynomially_equal_volume() {
        let valid = check_tensor(
            "fn f<B: Dim, H: Dim>(x: Tensor<Float32, [B, H + 1]>) -> Tensor<Float32, [B * H + B]> { x.reshape::<[B * H + B]>() }",
        );
        assert!(valid.is_empty(), "{valid:?}");

        let invalid = tensor_msgs(
            "fn f<B: Dim, C: Dim, D: Dim>(x: Tensor<Float32, [B, C]>) { let y = x.reshape::<[B, D]>(); }",
        );
        assert!(
            invalid
                .iter()
                .any(|message| message.contains("reshape cannot preserve element count")),
            "{invalid:?}"
        );
    }

    #[test]
    fn tensor_slice_axis_proves_symbolic_constraints_and_allows_zero() {
        let symbolic = check_tensor(
            "fn f<S: Dim, L: Dim>(x: Tensor<Float32, [S + L]>) -> Tensor<Float32, [L]> { x.slice_axis::<0, S, L>() }",
        );
        assert!(symbolic.is_empty(), "{symbolic:?}");

        let zero = check_tensor(
            "fn f(x: Tensor<Float32, [0]>) -> Tensor<Float32, [0]> { x.slice_axis::<0, 0, 0>() }",
        );
        assert!(zero.is_empty(), "{zero:?}");

        let invalid = tensor_msgs(
            "fn f<N: Dim, S: Dim, L: Dim>(x: Tensor<Float32, [N]>) { let y = x.slice_axis::<0, S, L>(); }",
        );
        assert!(
            invalid
                .iter()
                .any(|message| message.contains("cannot prove slice constraint")),
            "{invalid:?}"
        );
    }

    #[test]
    fn standalone_tensor_functions_require_borrowed_operands() {
        let valid = check_tensor(
            "fn f(a: Tensor<Float32, [4]>, b: Tensor<Float32, [4]>) { let x = add(&a, &b); let y = add(&a, &b); }",
        );
        assert!(valid.is_empty(), "{valid:?}");

        let invalid = tensor_msgs(
            "fn f(a: Tensor<Float32, [4]>, b: Tensor<Float32, [4]>) { let x = add(a, b); }",
        );
        assert!(
            invalid
                .iter()
                .any(|message| message.contains("must be borrowed")),
            "{invalid:?}"
        );
    }

    #[test]
    fn tensor_suggestions_are_only_emitted_when_one_fix_is_proven() {
        let diagnostics = check_tensor(
            "fn f(a: Tensor<Float32, [4]>, b: Tensor<Float16, [3]>) { let x = add(&a, &b); }",
        );
        assert!(
            diagnostics.iter().all(|diagnostic| diagnostic
                .notes
                .iter()
                .all(|note| !note.contains(".cast::<")
                    && !note.contains(".broadcast_to::<")
                    && !note.contains(".to_device::<"))),
            "{diagnostics:?}"
        );
    }

    #[test]
    fn model_load_is_nominal_and_predict_dims_are_fresh_per_call() {
        let valid = check_tensor(
            "model Classifier<B: Dim> { input image: Tensor<Float32, [B, 3]>; output class: Tensor<Float32, [B, 10]>; } fn load_it() -> Result<Classifier, ModelError> { Classifier::load(\"model.onnx\") } fn run(model: Classifier, one: Tensor<Float32, [1, 3]>, eight: Tensor<Float32, [8, 3]>) { let a: Tensor<Float32, [1, 10]> = model.predict(&one); let b: Tensor<Float32, [8, 10]> = model.predict(&eight); }",
        );
        assert!(valid.is_empty(), "{valid:?}");
    }

    #[test]
    fn model_predict_preserves_shared_dimensions_across_ports() {
        let diagnostics = check_tensor(
            "model Pair<B: Dim> { input left: Tensor<Float32, [B, 3]>; input right: Tensor<Float32, [B, 4]>; output result: Tensor<Float32, [B, 7]>; } fn run(model: Pair, left: Tensor<Float32, [2, 3]>, right: Tensor<Float32, [5, 4]>) { let result = model.predict(&left, &right); }",
        );
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code.as_deref() == Some("E0212")),
            "{diagnostics:?}"
        );
        assert!(
            diagnostics.iter().any(|diagnostic| diagnostic
                .notes
                .iter()
                .any(|note| note.contains("model port declared"))),
            "{diagnostics:?}"
        );
    }

    fn check_snippet(src: &str) -> Vec<Diagnostic> {
        let file = Arc::new(SourceFile::new("test.stark".to_string(), src.to_string()));
        let (tree, diags) = parse(&file, ParseMode::Snippet);
        assert!(diags.is_empty(), "parse failed: {:?}", diags);
        let (hir, sem_diags) = resolve(&tree, file.clone());
        let mut all_diags = sem_diags.clone();
        let mut type_diags = check(&hir, file);
        all_diags.append(&mut type_diags);
        all_diags
    }

    #[test]
    fn test_type_mismatch() {
        let diags = check_snippet("let x: Int32 = \"hello\";");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code.as_deref(), Some("E0001"));
    }

    #[test]
    fn test_immutable_reassignment() {
        let diags = check_snippet("let x = 42; x = 43;");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code.as_deref(), Some("E0400"));
    }

    #[test]
    fn test_uninitialized_use() {
        let diags = check_snippet("let x: Int32; let y = x;");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code.as_deref(), Some("E0401"));
    }

    #[test]
    fn test_deferred_initialization() {
        let diags = check_snippet("let x: Int32; x = 42; let y = x;");
        assert!(diags.is_empty(), "unexpected diagnostics: {:?}", diags);
    }

    #[test]
    fn test_array_bounds_check() {
        let diags = check_snippet("let arr: [Int32; 3] = [1, 2, 3]; let x = arr[5];");
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0007")));
    }

    #[test]
    fn test_try_non_result_function() {
        let diags = check_src("fn foo() -> Int32 { let x = 42; x? }");
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0006")));
    }

    #[test]
    fn test_borrow_conflicts() {
        let diags = check_src("fn foo() { let mut x = 42; let r1 = &x; let r2 = &mut x; }");
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0101")));
    }

    #[test]
    fn test_return_escape() {
        let diags = check_src("fn foo() -> &Int32 { let x = 42; &x }");
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0103")));
    }

    #[test]
    fn test_non_exhaustive_match() {
        let diags = check_src(
            "enum Color { Red, Green } fn test(c: Color) { match c { Color::Red => {} } }",
        );
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0303")));
    }

    #[test]
    fn test_break_outside_loop() {
        let diags = check_src("fn test() { break; }");
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0302")));
    }

    #[test]
    fn builtin_name_in_type_position_is_diagnostic_not_panic() {
        let diags = check_src("fn main() { let x: print; }");
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0202")));
    }

    #[test]
    fn generic_item_parameters_are_in_scope() {
        let diags = check_src("struct Box<T> { value: T } fn id<T>(x: T) -> T { x }");
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }

    #[test]
    fn integer_literal_is_a_valid_array_index() {
        let diags = check_src("fn main() { let a: [Int32; 3] = [1, 2, 3]; let x = a[2]; }");
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }

    #[test]
    fn branch_initialization_is_intersected() {
        let diags = check_src(
            "fn choose(c: Bool) -> Int32 { let x: Int32; if c { x = 1; } else { x = 2; } x }",
        );
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }

    #[test]
    fn initialized_outer_local_is_visible_in_loop() {
        let diags = check_src("fn f(c: Bool) { let x = 1; while c { let y = x; } }");
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }

    #[test]
    fn temporary_borrow_ends_with_statement() {
        let diags = check_src("fn f() { let mut x = 1; &x; x = 2; }");
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }

    #[test]
    fn inferred_copy_value_can_be_reused() {
        let diags = check_src("fn f() { let x = 1; let y = x; let z = x; }");
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }

    #[test]
    fn user_copy_type_can_be_reused() {
        let diags = check_src(
            "struct S { x: Int32 } impl Copy for S {} fn f() { let s = S { x: 1 }; let a = s; let b = s; }",
        );
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }

    #[test]
    fn method_receivers_obey_ownership_and_borrowing() {
        let moved = check_src(
            "struct S { text: String } impl S { fn consume(self) {} } fn f() { let s = S { text: String::from(\"x\") }; s.consume(); s.consume(); }",
        );
        assert!(moved
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0100")));

        let conflict = check_src(
            "struct S { value: Int32 } impl S { fn update(&mut self) {} } fn f() { let mut s = S { value: 1 }; let shared = &s; s.update(); }",
        );
        assert!(conflict
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0101")));

        let immutable = check_src(
            "struct S { value: Int32 } impl S { fn update(&mut self) {} } fn f() { let s = S { value: 1 }; s.update(); }",
        );
        assert!(immutable
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0400")));
    }

    #[test]
    fn borrow_stored_in_local_lasts_for_block() {
        let diags = check_src("fn f() { let mut x = 1; let r = &x; x = 2; }");
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0101")));
    }

    #[test]
    fn borrow_nested_in_tuple_lasts_for_block() {
        let diags = check_src("fn f() { let mut x = 1; let r = (&x,); x = 2; }");
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0101")));
    }

    #[test]
    fn returning_nested_reference_to_local_is_rejected() {
        let diags = check_src("fn f() -> (&Int32,) { let x = 1; (&x,) }");
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0103")));
    }

    #[test]
    fn ownership_checker_visits_loop_bodies() {
        let diags = check_src(
            "struct S { x: Int32 } fn take(v: S) {} fn f(c: Bool) { let s = S { x: 1 }; while c { take(s); take(s); } }",
        );
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0100")));
    }

    #[test]
    fn partial_move_allows_sibling_but_not_whole_value() {
        let valid = check_src(
            "struct S { x: Int32 } struct Pair { a: S, b: S } fn take(v: S) {} fn f() { let p = Pair { a: S { x: 1 }, b: S { x: 2 } }; take(p.a); take(p.b); }",
        );
        assert!(valid.is_empty(), "unexpected diagnostics: {valid:?}");

        let invalid = check_src(
            "struct S { x: Int32 } struct Pair { a: S, b: S } fn take(v: S) {} fn take_pair(v: Pair) {} fn f() { let p = Pair { a: S { x: 1 }, b: S { x: 2 } }; take(p.a); take_pair(p); }",
        );
        assert!(invalid.iter().any(|d| d.code.as_deref() == Some("E0100")));
    }

    #[test]
    fn shorthand_struct_field_consumes_its_source() {
        let diags = check_src(
            "struct S { x: Int32 } struct W { s: S } fn take(v: S) {} fn f() { let s = S { x: 1 }; let w = W { s }; take(s); }",
        );
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0100")));
    }

    #[test]
    fn generic_operator_requires_and_accepts_trait_bound() {
        let valid = check_src("fn max<T: Ord>(a: T, b: T) -> T { if a > b { a } else { b } }");
        assert!(valid.is_empty(), "unexpected diagnostics: {valid:?}");

        let invalid = check_src("fn max<T>(a: T, b: T) -> T { if a > b { a } else { b } }");
        assert!(invalid
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0500")));
    }

    #[test]
    fn associated_types_are_required_by_trait_impls() {
        let diags = check_src(
            "trait Iterator { type Item; } struct Counter { n: Int32 } impl Iterator for Counter {}",
        );
        assert!(diags
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0500")));
    }

    #[test]
    fn associated_type_bindings_are_checked_at_instantiation() {
        let diags = check_src(
            "trait Source { type Item; } struct Number { n: Int32 } impl Source for Number { type Item = Int32; } fn need<I: Source<Item = String>>(value: I) -> I { value } fn main() { let n = Number { n: 1 }; need::<Number>(n); }",
        );
        assert!(diags
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0500")));
    }

    #[test]
    fn required_trait_methods_and_orphan_rules_are_enforced() {
        let missing_method = check_src(
            "trait T { fn apply(&self) -> Int32; } struct S { x: Int32 } impl T for S {}",
        );
        assert!(missing_method
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0500")));

        let orphan = check_src("impl Copy for Int32 {}");
        assert!(orphan
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0500")));
    }

    #[test]
    fn trait_method_signatures_must_match() {
        let wrong_receiver = check_src(
            "trait T { fn get(&self) -> Int32; } struct S { x: Int32 } impl T for S { fn get(self) -> Int32 { self.x } }",
        );
        assert!(wrong_receiver
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0500")));

        let wrong_return = check_src(
            "trait T { fn get(&self) -> Int32; } struct S { x: Int32 } impl T for S { fn get(&self) -> Bool { true } }",
        );
        assert!(wrong_return
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0500")));
    }

    #[test]
    fn borrowed_local_cannot_escape_through_user_function() {
        let diags = check_src(
            "fn wrap<T>(x: &T) -> Option<&T> { Some(x) } fn bad() -> Option<&Int32> { let x = 1; wrap(&x) }",
        );
        assert!(diags
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0103")));
    }

    #[test]
    fn overlapping_impls_are_rejected() {
        let diags = check_src(
            "struct S { x: Int32 } impl S { fn value(&self) -> Int32 { self.x } } impl S { fn value(&self) -> Int32 { self.x } }",
        );
        assert!(diags
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0500")));
    }

    #[test]
    fn copy_and_drop_soundness_rules_are_enforced() {
        let both = check_src("struct S { x: Int32 } impl Copy for S {} impl Drop for S {}");
        assert!(both
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0500")));

        let non_copy_field = check_src(
            "struct Inner { x: Int32 } struct Outer { inner: Inner } impl Copy for Outer {}",
        );
        assert!(non_copy_field
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0500")));
    }
}
