//! Type checking, mutability, and definite assignment validation pass for STARK (PLAN.md M2.2).

use crate::ast::{AssignOp, BinOp, Lit, Primitive, UnOp};
use crate::diag::Diagnostic;
use crate::hir::{
    self, BlockId, Builtin, CoreType, ExprId, Hir, ItemId, LocalId, PatId, Res, StmtId, TypeId,
};
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
    Error,
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum VariantFields {
    Unit,
    Tuple(Vec<Ty>),
    Struct(HashMap<String, Ty>),
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

pub fn analyze(hir: &Hir, file: Arc<SourceFile>) -> TypeCheckResult {
    let mut checker = TypeChecker {
        hir,
        file: file.clone(),
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
    };

    checker.check_crate();
    let expr_types = checker
        .expr_types
        .iter()
        .map(|(&id, ty)| (id, checker.resolve(ty)))
        .collect();
    let local_types = checker
        .local_types
        .iter()
        .map(|(&id, ty)| (id, checker.resolve(ty)))
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
                    hir::GenericArg::Binding { .. } => None,
                })
                .collect()
        })
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
            Ty::Error => "{error}".to_string(),
        }
    }

    fn convert_hir_type(&mut self, id: TypeId) -> Ty {
        let node = self.hir.ty(id);
        match &node.kind {
            hir::TypeKind::Primitive(p) => Ty::Primitive(*p),
            hir::TypeKind::Path { res, args, .. } => match res {
                Res::Item(item_id) => {
                    let item = self.hir.item(*item_id);
                    let type_args = self.convert_generic_type_args(args.as_ref());
                    match &item.kind {
                        hir::ItemKind::Struct { generics, .. } => {
                            self.validate_generic_arity(generics.len(), type_args.len(), node.span);
                            Ty::Struct(*item_id, type_args)
                        }
                        hir::ItemKind::Enum { generics, .. } => {
                            self.validate_generic_arity(generics.len(), type_args.len(), node.span);
                            Ty::Enum(*item_id, type_args)
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
                    Ty::Param(name_str.to_string())
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
            },
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

    fn check_crate(&mut self) {
        // Pass 1: Populate item signatures (structs, enums, functions)
        for item in &self.hir.items {
            let item_id = hir::ItemId(
                self.hir
                    .items
                    .iter()
                    .position(|i| std::ptr::eq(i, item))
                    .unwrap() as u32,
            );
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
                    self.fn_sigs.insert(item_id, FnSigTy { params, ret });
                }
                hir::ItemKind::Const { ty, .. } => {
                    let const_ty = self.convert_hir_type(*ty);
                    self.const_types.insert(item_id, const_ty);
                }
                hir::ItemKind::Impl { self_ty, items, .. } => {
                    let _self_ty = self.convert_hir_type(*self_ty);
                    // Register methods of the impl
                    for impl_item in items {
                        if let hir::ImplItem::Fn { def, .. } = impl_item {
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
                            // We need to associate this method signature in fn_sigs.
                            // Find the ItemId of the containing ImplItemFn.
                            // Since impl_item is inside the Impl, we don't have its direct ItemId in the top-level items,
                            // but when we check the impl, we will check each method. Let's register it.
                            // In resolve.rs, we allocated the method def as part of ImplItem.
                        }
                    }
                }
                _ => {}
            }
        }

        self.validate_impl_rules();

        // Pass 2: Typecheck bodies & run semantic checks
        for item in &self.hir.items {
            let item_id = hir::ItemId(
                self.hir
                    .items
                    .iter()
                    .position(|i| std::ptr::eq(i, item))
                    .unwrap() as u32,
            );
            match &item.kind {
                hir::ItemKind::Fn(def) => {
                    self.check_fn_def(item_id, def);
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
        }

        // Snippet mode check
        if let hir::Root::Snippet { stmts, tail } = &self.hir.root {
            let mut state = HashSet::new();
            for &stmt_id in stmts {
                self.check_stmt(stmt_id, &mut state);
            }
            if let Some(tail_id) = tail {
                let _tail_ty = self.check_expr(*tail_id);
            }
        }

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

        for item in &self.hir.items {
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
        }

        for item_id in copy_types.intersection(&drop_types) {
            self.diags.push(
                Diagnostic::error(
                    "a type cannot implement both Copy and Drop",
                    self.hir.item(*item_id).span,
                )
                .with_code("E0500"),
            );
        }

        for item_id in copy_types.iter().copied() {
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
                    .with_code("E0500"),
                );
            }
        }
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
    }

    fn check_block(&mut self, block_id: BlockId, state: &mut HashSet<LocalId>) -> Ty {
        let block = self.hir.block(block_id);

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

        if let Some(tail_expr) = block.tail {
            self.check_expr(tail_expr)
        } else {
            Ty::Primitive(Primitive::Unit)
        }
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
                        let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        self.require_operator_bound(&lhs_ty, "Eq", expr.span);
                        Ty::Primitive(Primitive::Bool)
                    }
                    BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
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
                let target = self.convert_hir_type(*ty);
                let source_resolved = self.resolve(&source);
                let target_resolved = self.resolve(&target);
                if !matches!(source_resolved, Ty::Error)
                    && !matches!(target_resolved, Ty::Error)
                    && (!matches!(&source_resolved, Ty::Primitive(p) if is_numeric(*p))
                        || !matches!(&target_resolved, Ty::Primitive(p) if is_numeric(*p)))
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
                if let hir::ExprKind::Field { base, name } = &self.hir.expr(*callee).kind {
                    self.resolve_method(*base, *name, args, expr.span)
                } else if let hir::ExprKind::Path {
                    res: Res::TraitMember(trait_id, member),
                    ..
                } = &self.hir.expr(*callee).kind
                {
                    self.check_qualified_trait_call(*trait_id, *member, args, expr.span)
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
            hir::ExprKind::Field { base, name } => {
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
                if !is_integer {
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
                    Ty::Slice(elem) => *elem,
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
                    let pat_ty = self.check_pat(arm.pat);
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
                let elem_ty = self.new_type_var();
                let _ = self.unify(
                    Ty::Range(Box::new(elem_ty.clone())),
                    iter_ty,
                    self.hir.expr(*iter).span,
                );

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

    fn check_pat(&mut self, pat_id: PatId) -> Ty {
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
            hir::PatKind::Wild => self.new_type_var(),
            hir::PatKind::Binding { local, .. } => {
                self.local_types.get(local).cloned().unwrap_or(Ty::Error)
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
                    let args = self.nominal_use_args(*enum_id, None, pat.span);
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
                            let p_ty = self.check_pat(*p);
                            let expected_t = self.instantiate_ty(&expected_t, &map);
                            let _ = self.unify(expected_t, p_ty, p.span(self.hir));
                        }
                    }
                    Ty::Enum(*enum_id, args)
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
                                let p_ty = self.check_pat(sub_pat);
                                let expected_f_ty = self.instantiate_ty(expected_f_ty, &map);
                                let _ = self.unify(expected_f_ty, p_ty, field.name);
                            }
                        }
                    }
                    Ty::Struct(*struct_id, args)
                } else {
                    Ty::Error
                }
            }
            hir::PatKind::Tuple(elems) => {
                let tys = elems.iter().map(|&p| self.check_pat(p)).collect();
                Ty::Tuple(tys)
            }
            hir::PatKind::Array(elems) => {
                let elem_ty = self.new_type_var();
                for &e in elems {
                    let ety = self.check_pat(e);
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
            _ => ty.clone(),
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
            return sig;
        }

        let mut map = HashMap::new();
        if let Some(args) = turbofish {
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

                self.bounds_checks
                    .push((arg_ty.clone(), param.bounds.clone(), span));
                map.insert(param_name, arg_ty);
            }
        } else {
            for param in generics {
                let param_name = self.text(param.name).to_string();
                let var = self.new_type_var();
                self.bounds_checks
                    .push((var.clone(), param.bounds.clone(), span));
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

        let params = sig
            .params
            .iter()
            .map(|p| self.instantiate_ty(p, &map))
            .collect();
        let ret = self.instantiate_ty(&sig.ret, &map);

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

    fn resolve_method(
        &mut self,
        base_expr: ExprId,
        name_span: Span,
        args: &[ExprId],
        call_span: Span,
    ) -> Ty {
        let base_ty = self.check_expr(base_expr);
        let resolved_base = self.resolve(&base_ty);
        let name_str = self.text(name_span).to_string();

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
