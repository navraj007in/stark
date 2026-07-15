//! Type checking, mutability, and definite assignment validation pass for STARK (PLAN.md M2.2).

use crate::ast::{AssignOp, BinOp, Lit, Primitive, UnOp};
use crate::diag::Diagnostic;
use crate::hir::{
    self, BlockId, Builtin, ExprId, Hir, ItemId, LocalId, PatId, Res, StmtId, TypeId,
};
use crate::source::{SourceFile, Span};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVarId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Primitive(Primitive),
    Struct(ItemId),
    Enum(ItemId),
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
    current_fn_ret: Option<Ty>,
    loop_nesting: u32,
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
        current_fn_ret: None,
        loop_nesting: 0,
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
            (Ty::Struct(s1), Ty::Struct(s2)) if s1 == s2 => Ok(()),
            (Ty::Enum(e1), Ty::Enum(e2)) if e1 == e2 => Ok(()),
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

    fn ty_to_string(&self, ty: &Ty) -> String {
        let ty = self.resolve(ty);
        match ty {
            Ty::Primitive(p) => p.name().to_string(),
            Ty::Struct(id) => {
                let item = self.hir.item(id);
                if let hir::ItemKind::Struct { name, .. } = &item.kind {
                    self.text(*name).to_string()
                } else {
                    "Struct".to_string()
                }
            }
            Ty::Enum(id) => {
                let item = self.hir.item(id);
                if let hir::ItemKind::Enum { name, .. } = &item.kind {
                    self.text(*name).to_string()
                } else {
                    "Enum".to_string()
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
            hir::TypeKind::Path { res, .. } => match res {
                Res::Item(item_id) => {
                    if item_id.0 == 999999 {
                        self.diags.push(
                            Diagnostic::error(
                                format!("undefined type '{}'", self.text(node.span)),
                                node.span,
                            )
                            .with_code("E0202"),
                        );
                        Ty::Error
                    } else {
                        let item = self.hir.item(*item_id);
                        match &item.kind {
                            hir::ItemKind::Struct { .. } => Ty::Struct(*item_id),
                            hir::ItemKind::Enum { .. } => Ty::Enum(*item_id),
                            _ => Ty::Error,
                        }
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
                Res::TypeParam => {
                    let name_str = self.text(node.span);
                    Ty::Param(name_str.to_string())
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
                            hir::VariantKind::Tuple(types) => VariantFields::Tuple(
                                types.iter().map(|&t| self.convert_hir_type(t)).collect(),
                            ),
                            hir::VariantKind::Struct(fields) => {
                                let mut fields_map = HashMap::new();
                                for f in fields {
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
                    self.current_self_ty = Some(self.convert_hir_type(*self_ty));
                    for impl_item in items {
                        if let hir::ImplItem::Fn { def, .. } = impl_item {
                            self.check_fn_def(item_id, def);
                        }
                    }
                    self.current_self_ty = prev_self;
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
        let mut impls: Vec<(Option<Res>, Ty, Span)> = Vec::new();
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

            let self_type_is_local = matches!(&self_ty, Ty::Struct(_) | Ty::Enum(_));
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

            if impls.iter().any(|(previous_trait, previous_ty, _)| {
                *previous_trait == trait_res && self.types_equal(previous_ty, &self_ty)
            }) {
                self.diags.push(
                    Diagnostic::error("overlapping implementation for the same type", item.span)
                        .with_code("E0500")
                        .with_label("another applicable impl already exists"),
                );
            }
            impls.push((trait_res, self_ty.clone(), item.span));

            let trait_name = trait_
                .as_ref()
                .map(|trait_ref| self.text(trait_ref.path.span));
            if let Ty::Struct(id) | Ty::Enum(id) = self_ty {
                match trait_name {
                    Some("Copy") => {
                        copy_types.insert(id);
                    }
                    Some("Drop") => {
                        drop_types.insert(id);
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

    fn check_fn_def(&mut self, _item_id: ItemId, def: &hir::FnDef) {
        let sig = &def.sig;

        let expected_ret = match sig.ret {
            hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
            hir::RetTy::Ty(t) => self.convert_hir_type(t),
            hir::RetTy::Never(_) => Ty::Never,
        };
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
            self.local_types.insert(param.local, ty);
            self.local_mutability.insert(param.local, param.mutable);
            state.insert(param.local);
        }

        let ret_ty = self.check_block(def.body, &mut state);

        // Verify function return type
        let resolved_expected_ret = self.resolve(&expected_ret);
        if resolved_expected_ret != Ty::Primitive(Primitive::Unit)
            && resolved_expected_ret != Ty::Never
            && resolved_expected_ret != Ty::Error
        {
            let block = self.hir.block(def.body);
            if block.tail.is_none() {
                self.diags
                    .push(Diagnostic::error("missing return value", block.span).with_code("E0301"));
            }
        }

        let _ = self.unify(expected_ret, ret_ty, sig.span);
        self.current_fn_ret = None;
        self.current_fn_generics = None;
    }

    fn check_block(&mut self, block_id: BlockId, state: &mut HashSet<LocalId>) -> Ty {
        let block = self.hir.block(block_id);

        // Scope state for block variables
        for &stmt_id in &block.stmts {
            self.check_stmt(stmt_id, state);
        }

        if let Some(tail_expr) = block.tail {
            self.check_expr(tail_expr)
        } else {
            Ty::Primitive(Primitive::Unit)
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
                }
                if let Some(e) = expr {
                    let _ = self.check_expr(*e);
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
                    if let Some(variants) = self.enum_variants.get(enum_id) {
                        let variant = &variants[*variant_idx as usize];
                        match &variant.fields {
                            VariantFields::Unit => Ty::Enum(*enum_id),
                            VariantFields::Tuple(tys) => Ty::Fn {
                                params: tys.clone(),
                                ret: Box::new(Ty::Enum(*enum_id)),
                            },
                            VariantFields::Struct(_) => {
                                // Struct variant construct is struct lit, not path
                                Ty::Error
                            }
                        }
                    } else {
                        Ty::Error
                    }
                }
                Res::Primitive(p) => Ty::Primitive(*p),
                Res::SelfType => self.current_self_ty.clone().unwrap_or(Ty::Error),
                Res::SelfValue(local) => self.local_types.get(local).cloned().unwrap_or(Ty::Error),
                Res::Builtin(builtin) => self.builtin_type(*builtin),
                Res::Err | Res::TypeParam | Res::CoreTrait(_) => Ty::Error,
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
                let _ = self.check_expr(*cast_expr);
                self.convert_hir_type(*ty)
            }
            hir::ExprKind::Call { callee, args } => {
                if let hir::ExprKind::Field { base, name } = &self.hir.expr(*callee).kind {
                    self.resolve_method(*base, *name, args, expr.span)
                } else {
                    let callee_ty = self.check_expr(*callee);
                    let arg_tys: Vec<Ty> = args.iter().map(|&a| self.check_expr(a)).collect();

                    let ret_var = self.new_type_var();
                    let expected_fn = Ty::Fn {
                        params: arg_tys,
                        ret: Box::new(ret_var.clone()),
                    };

                    let _ = self.unify(expected_fn, callee_ty, expr.span);
                    ret_var
                }
            }
            hir::ExprKind::Field { base, name } => {
                let mut base_ty = self.check_expr(*base);
                while let Ty::Ref { inner, .. } = self.resolve(&base_ty) {
                    base_ty = *inner;
                }

                let name_str = self.text(*name);
                match self.resolve(&base_ty) {
                    Ty::Struct(struct_id) => {
                        if let Some(fields) = self.struct_fields.get(&struct_id) {
                            if let Some(field_ty) = fields.get(name_str) {
                                field_ty.clone()
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
                        Ty::Enum(enum_id) => {
                            let name_str = self.text(self.hir.item(enum_id).span);
                            if name_str.contains("Result") || name_str.contains("Option") {
                                ret_ok = true;
                            }
                        }
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
                    Ty::Enum(enum_id) => {
                        let name_str = self.text(self.hir.item(enum_id).span);
                        if name_str.contains("Result") || name_str.contains("Option") {
                            self.new_type_var()
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
                let _ = self.unify(
                    Ty::Primitive(Primitive::UInt64),
                    count_ty,
                    self.hir.expr(*count).span,
                );

                let count_str = self.text(self.hir.expr(*count).span);
                let len = count_str.parse::<u64>().unwrap_or(0);
                Ty::Array(Box::new(val_ty), len)
            }
            hir::ExprKind::StructLit { res, fields, .. } => {
                if let Res::Item(struct_id) = res {
                    let expected_fields = self
                        .struct_fields
                        .get(struct_id)
                        .cloned()
                        .unwrap_or_default();
                    for field in fields {
                        let f_name = self.text(field.name);
                        if let Some(expected_f_ty) = expected_fields.get(f_name) {
                            if let Some(val_expr) = field.expr {
                                let val_ty = self.check_expr(val_expr);
                                let _ = self.unify(expected_f_ty.clone(), val_ty, field.name);
                            }
                        } else {
                            self.diags.push(
                                Diagnostic::error(
                                    format!("field '{}' does not exist in struct", f_name),
                                    field.name,
                                )
                                .with_code("E0001"),
                            );
                        }
                    }
                    Ty::Struct(*struct_id)
                } else {
                    Ty::Error
                }
            }
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
                        _ => {}
                    }

                    let body_ty = self.check_expr(arm.body);
                    let _ = self.unify(ret_ty.clone(), body_ty, self.hir.expr(arm.body).span);
                }

                if !has_wildcard {
                    if let Ty::Enum(enum_id) = self.resolve(&scr_ty) {
                        if let Some(variants) = self.enum_variants.get(&enum_id) {
                            if matched_variants.len() < variants.len() {
                                self.diags.push(
                                    Diagnostic::error("non-exhaustive pattern match", expr.span)
                                        .with_code("E0303"),
                                );
                            }
                        }
                    }
                }

                ret_ty
            }
            hir::ExprKind::Loop { body } => {
                self.loop_nesting += 1;
                let mut dummy_state = HashSet::new();
                let _ = self.check_block(*body, &mut dummy_state);
                self.loop_nesting -= 1;
                Ty::Never
            }
            hir::ExprKind::While { cond, body } => {
                let cond_ty = self.check_expr(*cond);
                let _ = self.unify(
                    Ty::Primitive(Primitive::Bool),
                    cond_ty,
                    self.hir.expr(*cond).span,
                );
                self.loop_nesting += 1;
                let mut dummy_state = HashSet::new();
                let _ = self.check_block(*body, &mut dummy_state);
                self.loop_nesting -= 1;
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

                self.loop_nesting += 1;
                let mut dummy_state = HashSet::new();
                dummy_state.insert(*local);
                let _ = self.check_block(*body, &mut dummy_state);
                self.loop_nesting -= 1;
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
                Res::Variant(enum_id, _) => Ty::Enum(*enum_id),
                _ => Ty::Error,
            },
            hir::PatKind::TupleVariant { res, pats, .. } => {
                if let Res::Variant(enum_id, variant_idx) = res {
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
                            let _ = self.unify(expected_t, p_ty, p.span(self.hir));
                        }
                    }
                    Ty::Enum(*enum_id)
                } else {
                    Ty::Error
                }
            }
            hir::PatKind::Struct { res, fields, .. } => {
                if let Res::Item(struct_id) = res {
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
                                let _ = self.unify(expected_f_ty.clone(), p_ty, field.name);
                            }
                        }
                    }
                    Ty::Struct(*struct_id)
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

        let params = sig
            .params
            .iter()
            .map(|p| self.instantiate_ty(p, &map))
            .collect();
        let ret = self.instantiate_ty(&sig.ret, &map);

        FnSigTy { params, ret }
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

        let mut try_types = vec![
            (resolved_base.clone(), false, false),
            (
                Ty::Ref {
                    mutable: false,
                    inner: Box::new(resolved_base.clone()),
                },
                true,
                false,
            ),
            (
                Ty::Ref {
                    mutable: true,
                    inner: Box::new(resolved_base.clone()),
                },
                true,
                true,
            ),
        ];

        if let Ty::Ref { mutable: _, inner } = &resolved_base {
            let inner = self.resolve(inner);
            try_types.push((inner.clone(), false, false));
            try_types.push((
                Ty::Ref {
                    mutable: false,
                    inner: Box::new(inner.clone()),
                },
                true,
                false,
            ));
            try_types.push((
                Ty::Ref {
                    mutable: true,
                    inner: Box::new(inner.clone()),
                },
                true,
                true,
            ));
        }

        let mut candidates = Vec::new();

        for item in &self.hir.items {
            if let hir::ItemKind::Impl {
                self_ty: impl_self_ty_id,
                items,
                trait_,
                ..
            } = &item.kind
            {
                let impl_self_ty = self.convert_hir_type(*impl_self_ty_id);

                for impl_item in items {
                    if let hir::ImplItem::Fn { def, .. } = impl_item {
                        let method_name_str = self.text(def.sig.name);
                        if method_name_str == name_str {
                            for (try_ty, _, _) in &try_types {
                                if self.types_equal(&impl_self_ty, try_ty) {
                                    candidates.push((def, impl_self_ty.clone(), trait_.is_some()));
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some((def, _, _)) = candidates.first() {
            let params_ty: Vec<Ty> = def
                .sig
                .params
                .iter()
                .map(|p| self.convert_hir_type(p.ty))
                .collect();
            let ret_ty = match def.sig.ret {
                hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
                hir::RetTy::Ty(t) => self.convert_hir_type(t),
                hir::RetTy::Never(_) => Ty::Never,
            };

            for (arg, param_t) in args.iter().zip(params_ty) {
                let arg_t = self.check_expr(*arg);
                let _ = self.unify(param_t, arg_t, self.hir.expr(*arg).span);
            }

            ret_ty
        } else {
            let is_ok_type = matches!(
                resolved_base,
                Ty::Struct(_) | Ty::Enum(_) | Ty::Ref { .. } | Ty::Param(_) | Ty::Error
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

    fn types_equal(&self, t1: &Ty, t2: &Ty) -> bool {
        let t1 = self.resolve(t1);
        let t2 = self.resolve(t2);
        match (&t1, &t2) {
            (Ty::Primitive(p1), Ty::Primitive(p2)) => p1 == p2,
            (Ty::Struct(s1), Ty::Struct(s2)) => s1 == s2,
            (Ty::Enum(e1), Ty::Enum(e2)) => e1 == e2,
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
            Ty::Struct(_) | Ty::Enum(_) => self.hir.items.iter().any(|item| {
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
                hir::ItemKind::Struct { .. } => Ty::Struct(*item),
                hir::ItemKind::Enum { .. } => Ty::Enum(*item),
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
            Ty::Struct(struct_id) | Ty::Enum(struct_id) => {
                for item in &self.hir.items {
                    if let hir::ItemKind::Impl {
                        self_ty: impl_self_ty_id,
                        trait_: Some(trait_ref),
                        ..
                    } = &item.kind
                    {
                        let raw_ty = self.convert_hir_type(*impl_self_ty_id);
                        let impl_self_ty = self.resolve(&raw_ty);
                        if let Ty::Struct(s_id) = impl_self_ty {
                            if s_id == *struct_id {
                                let trait_name = self.text(trait_ref.path.span);
                                if trait_name == bound_name {
                                    return true;
                                }
                            }
                        } else if let Ty::Enum(e_id) = impl_self_ty {
                            if e_id == *struct_id {
                                let trait_name = self.text(trait_ref.path.span);
                                if trait_name == bound_name {
                                    return true;
                                }
                            }
                        }
                    }
                }
                false
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

fn is_copy_with_impls(ty: &Ty, copy_types: &HashSet<ItemId>) -> bool {
    match ty {
        Ty::Primitive(_) | Ty::Ref { mutable: false, .. } | Ty::Never | Ty::Error => true,
        Ty::Struct(id) | Ty::Enum(id) => copy_types.contains(id),
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
    fn overlapping_impls_are_rejected() {
        let diags = check_src("struct S { x: Int32 } impl S {} impl S {}");
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
