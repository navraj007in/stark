//! **AS7 Packet 10 — the item passes.**
//!
//! The top of the DAG below the facade: `items` may use every checking module beneath it.
//!
//! `check_crate` drives the two passes over `hir.items` — signatures first, then bodies — and the
//! per-item checks that only make sense once for a whole declaration: generic-parameter shadowing,
//! type well-formedness, public-API reachability, associated-type projections and the layout
//! tables.
//!
//! **Model declarations are here, not in the tensor extension, and AS6 decided that deliberately.**
//! Declaring a model means converting written type syntax and managing a declaration scope, which
//! is Core machinery; AS6 packet 4D split the function so Core normalises the declaration and
//! `extensions::tensor::check` owns every validity rule — generic kinds, duplicate ports, allowed
//! port types, at least one input and one output.

use super::state::SelfScope;
use super::state::TypeChecker;
use super::types::{
    ty_contains_infer, type_is_sized, CallableSigTy, DeferredDisplayCheck, DeferredDisplayPlan,
    DisplayPath, FnSigTy, LayoutTables, Ty, VariantFields, VariantTy,
};
use crate::ast::Primitive;
use crate::diag::Diagnostic;
use crate::extensions::tensor::check as tensor_check;
use crate::hir::{self, CoreType, ExprId, ItemId, Res};
use crate::source::Span;
use std::collections::{HashMap, HashSet};

impl TypeChecker<'_> {
    /// Which package a source belongs to, for the orphan rule.
    ///
    /// AS1a gives every package source the logical name `<package>/<path within the package>`, so
    /// the package is the leading segment. `None` means "not a package build" — a single-file or
    /// path-named compile, where every source belongs to the one program and everything is local.
    ///
    /// This replaced `find_package_root`, which walked the file's PATH upwards looking for a
    /// `starkpkg.json` on disk. That only ever worked here by an asymmetry: the root file carried
    /// an absolute disk path while every other item's file carried a logical name, so the root
    /// probe found a manifest and the dependency probe found nothing, and "different package" fell
    /// out of the difference. Reading identity off the names makes the comparison say what it
    /// means, and stops it depending on the filesystem at type-check time.
    pub(super) fn source_package<'s>(&self, name: &'s str) -> Option<&'s str> {
        if std::path::Path::new(name).is_absolute() {
            return None;
        }
        name.split_once('/').map(|(package, _)| package)
    }

    pub(super) fn check_crate(&mut self) {
        // AS1b-ii-d: each of this function's three item walks used to open by pointing `self.file`
        // at the item's declaring file and close by restoring the root — DEV-069's mechanism for
        // getting span reads and diagnostic attribution right. Reads go through the span's own
        // source now, so there is no ambient file to aim.
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
                    self.suppress_tensor_diagnostics = true;
                    let saved = self.enter_tensor_param_scope(&def.sig.generics);
                    // WP-C7.9 Packet I: the function's generics are in scope for its own signature
                    // types here too, so a bound check reached during conversion sees the bounds
                    // the declaration actually wrote.
                    let saved_generics = self.current_fn_generics.replace(def.sig.generics.clone());
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
                    self.current_fn_generics = saved_generics;
                    self.suppress_tensor_diagnostics = false;
                    self.fn_sigs.insert(item_id, FnSigTy { params, ret });
                }
                hir::ItemKind::Const { ty, .. } => {
                    let const_ty = self.convert_hir_type(*ty);
                    self.const_types.insert(item_id, const_ty);
                }
                hir::ItemKind::TypeAlias { ty, .. } => {
                    self.alias_stack.push(item_id);
                    let _ = self.convert_hir_type(*ty);
                    self.alias_stack.pop();
                }
                hir::ItemKind::Impl { self_ty, items, .. } => {
                    let impl_self_ty = self.convert_hir_type(*self_ty);
                    let previous_self = self.enter_self_scope(impl_self_ty);
                    // Register methods of the impl
                    for impl_item in items {
                        if let hir::ImplItem::Fn { def, .. } = impl_item {
                            self.suppress_tensor_diagnostics = true;
                            let saved = self.enter_tensor_param_scope(&def.sig.generics);
                            let saved_generics =
                                self.current_fn_generics.replace(def.sig.generics.clone());
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
                            self.current_fn_generics = saved_generics;
                            self.suppress_tensor_diagnostics = false;
                        }
                    }
                    self.exit_self_scope(previous_self);
                }
                _ => {}
            }

            // AS1b-ii-d: the diagnostics this item produced used to be stamped with `self.file`
            // here, because a span could not say which file it indexed. It can now, so there is
            // nothing to stamp and `start_len` has no reader.
        }

        self.check_public_api_reachability();
        self.check_type_well_formedness();

        self.validate_impl_rules();

        // WP-C6.2c: precompute concrete associated-type bindings before checking bodies, so a
        // projection carried through generic instantiation (`Ty::Param("H::Item")`) can be
        // normalised to the impl's bound type at any call site.
        self.build_assoc_projections();

        // Pass 2: Typecheck bodies & run semantic checks
        for item in &self.hir.items {
            let item_id = hir::ItemId(
                self.hir
                    .items
                    .iter()
                    .position(|i| std::ptr::eq(i, item))
                    .unwrap() as u32,
            );
            let saved_item_scope = self.enter_item_scope(item_id);

            match &item.kind {
                hir::ItemKind::Fn(def) => {
                    self.check_fn_def(item_id, def);
                }
                hir::ItemKind::Model(def) => {
                    self.check_model_def(item_id, def);
                }
                hir::ItemKind::Impl {
                    self_ty,
                    items,
                    generics,
                    ..
                } => {
                    // WP-C6.2b-F5: bring the impl-head generics/bounds into scope for the bodies.
                    let converted_self = self.convert_hir_type(*self_ty);
                    let saved_scope = SelfScope {
                        self_ty: self.current_self_ty.replace(converted_self),
                        assoc_types: Some(std::mem::take(&mut self.current_assoc_types)),
                        impl_generics: Some(self.current_impl_generics.replace(generics.clone())),
                        trait_id: None,
                    };
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
                    self.exit_self_scope(saved_scope);
                }
                hir::ItemKind::Trait { items, .. } => {
                    let saved_scope = SelfScope {
                        self_ty: self.current_self_ty.replace(Ty::Param("Self".to_string())),
                        assoc_types: None,
                        impl_generics: None,
                        trait_id: Some(self.current_trait_id.replace(item_id)),
                    };
                    for trait_item in items {
                        if let hir::TraitItem::Method {
                            sig,
                            body: Some(body_id),
                        } = trait_item
                        {
                            let def = hir::FnDef {
                                sig: sig.clone(),
                                body: *body_id,
                            };
                            self.check_fn_def(item_id, &def);
                        }
                    }
                    self.exit_self_scope(saved_scope);
                }
                hir::ItemKind::Const { value, ty, .. } => {
                    let expected_ty = self.convert_hir_type(*ty);
                    let val_ty = self.check_expr(*value);
                    let _ = self.unify(expected_ty, val_ty, item.span);
                }
                _ => {}
            }

            self.exit_item_scope(saved_item_scope);

            // AS1b-ii-d: the diagnostics this item produced used to be stamped with `self.file`
            // here, because a span could not say which file it indexed. It can now, so there is
            // nothing to stamp and `start_len` has no reader.
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

        // WP-C6.2c: resolve deferred associated-type projections (`T::Item` where the base was an
        // inference variable) now that every argument has unified — before int-literal defaulting,
        // so a projection that grounds to `Int32` can still constrain a literal argument.
        self.discharge_ready_projections();

        // WP-C4.7-6.3: 03's solving step 5 — default any still-unconstrained integer literal —
        // runs HERE: after every body has been checked (so every expected type has had its
        // chance to constrain a literal) but BEFORE the deferred bound checks below, which must
        // see a concrete type rather than an open variable.
        self.default_unconstrained_int_literals();
        self.default_never_coerced_vars();

        // WP-C4.7-9 audit: `print`/`println` require a `Display`-able argument.
        let display = std::mem::take(&mut self.display_checks);
        for check in display {
            let DeferredDisplayCheck {
                ty,
                span,
                generic_scope: (fn_generics, impl_generics),
            } = check;
            let resolved = self.resolve(&ty);
            if matches!(resolved, Ty::Error) || ty_contains_infer(&resolved) {
                continue; // already failed, or undetermined — no cascade
            }
            // **DEV-236: restore the scope this print was WRITTEN in**, exactly as the plan loop
            // below does and for the same reason. The obligation now answers `Ty::Param` from the
            // parameter's declared bounds, which is a question about a scope Pass 3 has torn down;
            // asking it here without restoring saw no generics and refused
            // `fn show<T: Display>(x: T) { println(x); }`, a bound plainly written.
            let saved_fn = std::mem::replace(&mut self.current_fn_generics, fn_generics);
            let saved_impl = std::mem::replace(&mut self.current_impl_generics, impl_generics);
            let displayable = self.type_is_displayable(&resolved);
            self.current_fn_generics = saved_fn;
            self.current_impl_generics = saved_impl;
            if !displayable {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "type '{}' cannot be printed: it does not implement 'Display'",
                            self.ty_to_string(&resolved)
                        ),
                        span,
                    )
                    .with_code("E0500"),
                );
            }
        }

        // **AS3 Boundary 4: build the `Display` dispatch plan.**
        //
        // Here, not at the call sites: the walk keys positions off the RESOLVED type, and this is
        // the first point where every one of them is settled. Publishing earlier would key
        // positions off inference variables.
        let plans = std::mem::take(&mut self.display_plans);
        for plan in plans {
            let DeferredDisplayPlan {
                root,
                ty,
                generic_scope: (fn_generics, impl_generics),
            } = plan;
            let resolved = self.resolve(&ty);
            if matches!(resolved, Ty::Error) || ty_contains_infer(&resolved) {
                continue;
            }
            // Restore the scope this expression was WRITTEN in, so a `T: Display` bound is visible
            // to the walk exactly as it was where the programmer wrote it.
            let saved_fn = std::mem::replace(&mut self.current_fn_generics, fn_generics);
            let saved_impl = std::mem::replace(&mut self.current_impl_generics, impl_generics);
            if self.type_is_displayable(&resolved) {
                self.publish_display_uses(root, &resolved, self.hir.expr(root).span);
            }
            self.current_fn_generics = saved_fn;
            self.current_impl_generics = saved_impl;
        }

        // DEV-134: `?` propagation compatibility, for the same reason and at the same point.
        let tries = std::mem::take(&mut self.try_checks);
        for (operand_ty, ret_ty, span) in tries {
            self.check_try_compatibility(&operand_ty, &ret_ty, span);
        }

        // Pass 3: Check trait bounds
        let bounds = std::mem::take(&mut self.bounds_checks);
        for (concrete_ty, bounds_list, span, enclosing) in bounds {
            // DEV-067(a): restore the generic environment this obligation was recorded in, so a
            // caller's own `T: Ord` can discharge a callee's `T: Ord` (TYPE-GENERIC-001).
            let saved_generics = self.current_fn_generics.replace(enclosing);
            // DEV-101 also swapped `self.file` to the declaring file around these reads, because
            // `satisfies_bound` identifies the trait by the bound path's TEXT and the checker had
            // long since returned to the root file. The swap is gone: `bound.path.span` names the
            // callee's file, and the diagnostic's `span` names the caller's call site — the two
            // no longer have to take turns owning one ambient file.
            let mut violations = Vec::new();
            for bound in bounds_list {
                if !self.satisfies_bound(&concrete_ty, &bound) {
                    violations.push((
                        self.ty_to_string(&concrete_ty),
                        self.text(bound.path.span).to_string(),
                    ));
                }
            }
            self.current_fn_generics = saved_generics;
            for (ty_str, bound_str) in violations {
                self.diags.push(
                    Diagnostic::error(
                        format!("type '{ty_str}' does not satisfy trait bound '{bound_str}'"),
                        span,
                    )
                    .with_code("E0500"),
                );
            }
        }
    }

    pub(super) fn check_public_api_reachability(&mut self) {
        let mut exposures = Vec::new();
        for (index, item) in self.hir.items.iter().enumerate() {
            if item.vis != Some(crate::ast::Vis::Pub) {
                continue;
            }
            let item_id = ItemId(index as u32);
            let mut types = Vec::new();
            match &item.kind {
                hir::ItemKind::Fn(def) => {
                    types.extend(def.sig.params.iter().map(|param| param.ty));
                    if let hir::RetTy::Ty(ty) = def.sig.ret {
                        types.push(ty);
                    }
                }
                hir::ItemKind::Struct { fields, .. } => {
                    types.extend(
                        fields
                            .iter()
                            .filter(|field| field.is_pub)
                            .map(|field| field.ty),
                    );
                }
                hir::ItemKind::Enum { variants, .. } => {
                    for variant in variants {
                        match &variant.kind {
                            hir::VariantKind::Unit => {}
                            hir::VariantKind::Tuple(fields) => types.extend(fields.iter().copied()),
                            hir::VariantKind::Struct(fields) => {
                                types.extend(fields.iter().map(|field| field.ty));
                            }
                        }
                    }
                }
                hir::ItemKind::Trait { items, .. } => {
                    for trait_item in items {
                        if let hir::TraitItem::Method { sig, .. } = trait_item {
                            types.extend(sig.params.iter().map(|param| param.ty));
                            if let hir::RetTy::Ty(ty) = sig.ret {
                                types.push(ty);
                            }
                        }
                    }
                }
                hir::ItemKind::Const { ty, .. } | hir::ItemKind::TypeAlias { ty, .. } => {
                    types.push(*ty);
                }
                _ => {}
            }
            for ty in types {
                if let Some(private) = self.private_type_in(ty) {
                    exposures.push((item_id, private, self.hir.ty(ty).span));
                }
            }
        }

        for (public_item, private_item, span) in exposures {
            let private_name = self.item_name(private_item);
            let public_name = self.item_name(public_item);
            let diagnostic = Diagnostic::error(
                format!("public item '{public_name}' exposes non-public type '{private_name}'"),
                span,
            )
            .with_code("E0209")
            .with_note("make the type publicly nameable or remove it from the public signature");
            self.diags.push(diagnostic);
        }
    }

    pub(super) fn check_type_well_formedness(&mut self) {
        let mut reported_unsized = HashSet::new();
        for (item, fields) in &self.struct_fields {
            for ty in fields.values() {
                if !type_is_sized(ty) && reported_unsized.insert(*item) {
                    self.diags.push(
                        Diagnostic::error(
                            "unsized types may occur only immediately behind a reference",
                            self.hir.item(*item).span,
                        )
                        .with_code("E0217"),
                    );
                }
            }
        }
        for (item, variants) in &self.enum_variants {
            for variant in variants {
                let types: Vec<&Ty> = match &variant.fields {
                    VariantFields::Unit => Vec::new(),
                    VariantFields::Tuple(types) => types.iter().collect(),
                    VariantFields::Struct(fields) => fields.values().collect(),
                };
                if types.iter().any(|ty| !type_is_sized(ty)) && reported_unsized.insert(*item) {
                    self.diags.push(
                        Diagnostic::error(
                            "unsized types may occur only immediately behind a reference",
                            self.hir.item(*item).span,
                        )
                        .with_code("E0217"),
                    );
                }
            }
        }

        let mut edges: HashMap<ItemId, HashSet<ItemId>> = HashMap::new();
        for (item, fields) in &self.struct_fields {
            let entry = edges.entry(*item).or_default();
            for ty in fields.values() {
                collect_direct_value_edges(ty, entry);
            }
        }
        for (item, variants) in &self.enum_variants {
            let entry = edges.entry(*item).or_default();
            for variant in variants {
                match &variant.fields {
                    VariantFields::Unit => {}
                    VariantFields::Tuple(types) => {
                        for ty in types {
                            collect_direct_value_edges(ty, entry);
                        }
                    }
                    VariantFields::Struct(fields) => {
                        for ty in fields.values() {
                            collect_direct_value_edges(ty, entry);
                        }
                    }
                }
            }
        }

        let mut reported = HashSet::new();
        for &item in edges.keys() {
            let mut active = HashSet::new();
            if direct_value_cycle(item, item, &edges, &mut active) && reported.insert(item) {
                self.diags.push(
                    Diagnostic::error(
                        "type has infinite size through a direct value cycle",
                        self.hir.item(item).span,
                    )
                    .with_code("E0217"),
                );
            }
        }
    }

    pub(super) fn private_type_in(&self, ty: hir::TypeId) -> Option<ItemId> {
        let node = self.hir.ty(ty);
        match &node.kind {
            hir::TypeKind::Path { res, args, .. } => {
                if let Res::Item(item) = res {
                    if self.hir.item(*item).vis != Some(crate::ast::Vis::Pub)
                        && !self.hir.publicly_nameable_items.contains(item)
                    {
                        return Some(*item);
                    }
                }
                args.as_ref().and_then(|args| {
                    args.args.iter().find_map(|arg| match arg {
                        hir::GenericArg::Type(ty) | hir::GenericArg::Binding { ty, .. } => {
                            self.private_type_in(*ty)
                        }
                        _ => None,
                    })
                })
            }
            hir::TypeKind::Array { elem, .. }
            | hir::TypeKind::Slice(elem)
            | hir::TypeKind::Ref { inner: elem, .. } => self.private_type_in(*elem),
            hir::TypeKind::Tuple(types) => types.iter().find_map(|ty| self.private_type_in(*ty)),
            hir::TypeKind::Fn { params, ret } => params
                .iter()
                .find_map(|ty| self.private_type_in(*ty))
                .or_else(|| ret.and_then(|ty| self.private_type_in(ty))),
            _ => None,
        }
    }

    /// DEV-069: an item's own name is read against the file that declares it, which is not
    /// necessarily the file being checked.
    pub(super) fn item_name(&self, item: ItemId) -> String {
        match &self.hir.item(item).kind {
            hir::ItemKind::Fn(def) => self.item_text(item, def.sig.name).to_string(),
            hir::ItemKind::Struct { name, .. }
            | hir::ItemKind::Enum { name, .. }
            | hir::ItemKind::Trait { name, .. }
            | hir::ItemKind::Const { name, .. }
            | hir::ItemKind::TypeAlias { name, .. }
            | hir::ItemKind::Mod { name, .. } => self.item_text(item, *name).to_string(),
            hir::ItemKind::Model(def) => self.item_text(item, def.name).to_string(),
            hir::ItemKind::Impl { .. } | hir::ItemKind::Use(_) => format!("item#{}", item.0),
        }
    }

    /// **NAME-SHADOW-001 (DEV-177): a generic parameter may not duplicate another one in scope.**
    ///
    /// 04-Semantic-Analysis.md: "Generic parameters may not duplicate another generic parameter or
    /// an item-level `Self`; a nested item introduces fresh item scopes."
    ///
    /// The rule existed and was unenforced, which let `impl<T> W<T> { fn choose<T>(..) }` both
    /// check and RUN — binding two distinct types to one name in one signature. That is not merely
    /// untidy: `Ty::Param` identifies a parameter by its `String`, so while duplicates are legal a
    /// name-keyed substitution environment could bind one concrete type to two different binders,
    /// and every available tie-break is a guess at semantics the type system does not carry.
    /// Enforcing the rule is what makes `Ty::Param(String)` unambiguous by construction.
    ///
    /// `owners` are the generic lists **normatively in scope** for this declaration — the enclosing
    /// impl's or trait's, never a lexically enclosing function's. Scope here means INHERITED, not
    /// nested: Core v1 rejects items inside blocks outright ("items are not allowed inside blocks"),
    /// so the specification's fresh-item-scope case cannot currently be written, and comparing only
    /// against inherited owners is what would keep it correct if it ever could be.
    ///
    /// A generic named `Self` needs no check here: the parser already refuses it with "expected a
    /// generic parameter name, found `Self`". Duplicating that as a type-check would be a second
    /// answer to a question already settled.
    pub(super) fn check_generic_shadowing(
        &mut self,
        generics: &[hir::GenericParam],
        owners: &[&[hir::GenericParam]],
        what: &str,
    ) {
        let mut seen: Vec<(String, Span)> = Vec::new();
        for owner in owners {
            for param in *owner {
                seen.push((self.text(param.name).to_string(), param.name));
            }
        }
        for param in generics {
            let name = self.text(param.name).to_string();
            if let Some((_, first)) = seen.iter().find(|(seen_name, _)| *seen_name == name) {
                let first = *first;
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "generic parameter '{name}' duplicates another generic parameter in \
                             scope"
                        ),
                        param.name,
                    )
                    .with_code("E0204")
                    .with_label(format!("'{name}' is already declared by {what}"))
                    .with_related(first, format!("'{name}' first declared here")),
                );
            }
            seen.push((name, param.name));
        }
    }

    pub(super) fn check_fn_def(&mut self, _item_id: ItemId, def: &hir::FnDef) {
        let sig = &def.sig;

        // `Dim` generic parameters are in scope for every signature type and
        // the body (tensor extension §3.1). No-op for Core-only functions.
        let saved_dims = self.enter_tensor_param_scope(&sig.generics);

        // WP-C7.9 Packet I: the function's own generics are in scope for its SIGNATURE types, not
        // only for its body. This used to be installed after the return type was converted, which
        // was invisible until a check needed to ask whether a type parameter satisfied a bound
        // while converting a signature: `fn build<T: Hash + Eq>() -> HashMap<T, Int32>` would then
        // see `T` with no declared bounds at all and reject its own return type.
        // NAME-SHADOW-001: check BEFORE installing this signature's generics, so the comparison is
        // against what was already in scope rather than against itself.
        let impl_owned = self.current_impl_generics.clone().unwrap_or_default();
        let trait_owned = match self.current_trait_id {
            Some(trait_id) => match &self.hir.item(trait_id).kind {
                hir::ItemKind::Trait { generics, .. } => generics.clone(),
                _ => Vec::new(),
            },
            None => Vec::new(),
        };
        let owner_label = if !impl_owned.is_empty() {
            "the enclosing impl"
        } else {
            "the enclosing trait"
        };
        self.check_generic_shadowing(
            &sig.generics,
            &[impl_owned.as_slice(), trait_owned.as_slice()],
            owner_label,
        );

        let saved_fn_scope = self.enter_fn_scope(sig.generics.clone());

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
        self.set_fn_return(expected_ret.clone());

        // Parameters in local_types
        let mut state = HashSet::new();
        let mut published_receiver: Option<Ty> = None;
        let mut published_params: Vec<Ty> = Vec::new();
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
            published_receiver = Some(receiver_ty.clone());
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
            published_params.push(ty.clone());
            self.local_types.insert(param.local, ty);
            self.local_mutability.insert(param.local, param.mutable);
            state.insert(param.local);
        }

        // **A3b: publish this body's signature.** `check_fn_def` is the single entry point for all
        // six executable callable classes, so publishing here covers free functions, inherent
        // methods, trait implementation methods, associated functions, `Drop::drop` and trait
        // default bodies — and cannot reach a bodyless trait declaration, which has no body to key
        // on. Publishing from the types the checker JUST established, rather than reconverting the
        // HIR signature later, is what keeps this from becoming a second answer to what the
        // signature is.
        let previous = self.callable_sigs.insert(
            def.body,
            CallableSigTy {
                receiver: published_receiver,
                params: published_params,
                ret: expected_ret.clone(),
            },
        );
        if previous.is_some() {
            // One HIR body belongs to exactly one callable. Two signatures for one body would mean
            // the arena is shared, and a later reader would silently get whichever landed last.
            self.diags.push(
                Diagnostic::error(
                    "internal: one HIR body was assigned two callable signatures",
                    sig.span,
                )
                .with_code("E0001"),
            );
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
        // AS7 Packet 2: RESTORE the enclosing function's scope rather than clearing it. Identical
        // while item checking does not nest; correct by construction if AS7's splitting makes it.
        self.exit_fn_scope(saved_fn_scope);
        self.exit_tensor_param_scope(saved_dims);
    }

    /// WP-C5.3e / DEV-100: publish the tables a layout walk needs so the walk itself can live in
    /// ONE place ([`LayoutTables::layout_of`]) and outlive the checker.
    ///
    /// Declaration ORDER is read from the HIR items, not from the checker's own `struct_fields`
    /// map: layout depends on field order and that map is name-keyed. A struct-shaped enum variant
    /// is omitted rather than laid out in an arbitrary order — its fields live in a `HashMap` too,
    /// and a wrong order is a wrong observable answer.
    pub(super) fn build_layout_tables(&self) -> LayoutTables {
        let mut struct_fields: HashMap<ItemId, Vec<Ty>> = HashMap::new();
        let mut enum_variants: HashMap<ItemId, Vec<Vec<Ty>>> = HashMap::new();
        let mut nominal_params: HashMap<ItemId, Vec<String>> = HashMap::new();

        for (&item, table) in &self.struct_fields {
            let hir::ItemKind::Struct { fields, .. } = &self.hir.item(item).kind else {
                continue;
            };
            let mut ordered = Vec::with_capacity(fields.len());
            let mut complete = true;
            for field in fields {
                let name: String = self.item_text(item, field.name).to_string();
                match table.get(name.as_str()) {
                    Some(ty) => ordered.push(ty.clone()),
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            if complete {
                struct_fields.insert(item, ordered);
            }
        }

        for (&item, variants) in &self.enum_variants {
            let mut ordered = Vec::with_capacity(variants.len());
            let mut complete = true;
            for variant in variants {
                match &variant.fields {
                    VariantFields::Unit => ordered.push(Vec::new()),
                    VariantFields::Tuple(tys) => ordered.push(tys.clone()),
                    VariantFields::Struct(_) => {
                        complete = false;
                        break;
                    }
                }
            }
            if complete {
                enum_variants.insert(item, ordered);
            }
        }

        for item in struct_fields.keys().chain(enum_variants.keys()) {
            let names: Vec<String> = self
                .item_generic_params(*item)
                .iter()
                .map(|param| self.item_text(*item, param.name).to_string())
                .collect();
            nominal_params.insert(*item, names);
        }

        LayoutTables {
            contract: crate::layout::TargetLayout::default(),
            struct_fields,
            enum_variants,
            nominal_params,
        }
    }

    /// WP-C6.2c: populate `assoc_projections` from every impl's associated-type bindings, keyed by
    /// the implementing nominal's `ItemId` and the associated-type name.
    pub(super) fn build_assoc_projections(&mut self) {
        let count = self.hir.items.len();
        for index in 0..count {
            let item_id = ItemId(index as u32);
            let hir::ItemKind::Impl { self_ty, items, .. } = &self.hir.item(item_id).kind else {
                continue;
            };
            let self_ty = *self_ty;
            // Convert the associated-type bindings against the impl's own file (types name items
            // relative to their declaring file).
            let bindings: Vec<(String, hir::TypeId)> = items
                .iter()
                .filter_map(|impl_item| match impl_item {
                    hir::ImplItem::AssocType { name, ty } => Some((*name, *ty)),
                    _ => None,
                })
                .map(|(name, ty)| (self.text(name).to_string(), ty))
                .collect();
            if bindings.is_empty() {
                continue;
            }
            let nominal = match self.convert_hir_type(self_ty) {
                Ty::Struct(id, _) | Ty::Enum(id, _) => Some(id),
                _ => None,
            };
            if let Some(nominal) = nominal {
                for (name, ty) in bindings {
                    let ty = self.convert_hir_type(ty);
                    self.assoc_projections.insert((nominal, name), ty);
                }
            }
        }
    }

    pub(super) fn publish_display_uses(&mut self, root: ExprId, ty: &Ty, span: Span) {
        self.walk_display_ty(root, ty, DisplayPath::default(), span, 0);
    }

    /// AS6 packet 4D-A: Core normalises the declaration — enters the generic scope, classifies
    /// each parameter, converts each written port type — and the extension decides whether what
    /// the declaration says is *allowed*. Staged rather than hoisted so that a conversion
    /// diagnostic cannot overtake the duplicate-name diagnostic for the same port.
    pub(super) fn check_model_def(&mut self, _item_id: ItemId, def: &hir::ModelDef) {
        if !self.options.tensor() {
            self.diags.push(Diagnostic::error(
                "model declarations require `--extension tensor` to be enabled",
                def.name,
            ));
            return;
        }

        let saved = self.enter_tensor_param_scope(&def.generics);

        for g in &def.generics {
            let kind = self.generic_kind(g);
            tensor_check::ModelDeclCheck::check_generic_kind(self, kind.as_tensor_param(), g.name);
        }

        let mut declaration = tensor_check::ModelDeclCheck::new();
        for port in &def.ports {
            let name = self.text(port.name).to_string();
            declaration.declare_port(self, &name, port.name, port.dir);
            let ty = self.convert_hir_type(port.ty);
            declaration.check_port_type(self, &ty, port.span);
        }
        declaration.finish(self, def.name);

        self.exit_tensor_param_scope(saved);
    }
}

fn collect_direct_value_edges(ty: &Ty, output: &mut HashSet<ItemId>) {
    match ty {
        Ty::Struct(item, arguments) | Ty::Enum(item, arguments) => {
            output.insert(*item);
            for argument in arguments {
                collect_direct_value_edges(argument, output);
            }
        }
        Ty::Ref { .. } | Ty::Core(CoreType::Box | CoreType::Vec, _) => {}
        Ty::Tuple(types) | Ty::Core(_, types) => {
            for ty in types {
                collect_direct_value_edges(ty, output);
            }
        }
        Ty::Array(element, _) | Ty::Slice(element) | Ty::Range(element) => {
            collect_direct_value_edges(element, output);
        }
        Ty::Fn { params, ret } => {
            for ty in params {
                collect_direct_value_edges(ty, output);
            }
            collect_direct_value_edges(ret, output);
        }
        _ => {}
    }
}

fn direct_value_cycle(
    origin: ItemId,
    current: ItemId,
    edges: &HashMap<ItemId, HashSet<ItemId>>,
    active: &mut HashSet<ItemId>,
) -> bool {
    if !active.insert(current) {
        return false;
    }
    let found = edges.get(&current).is_some_and(|targets| {
        targets
            .iter()
            .any(|target| *target == origin || direct_value_cycle(origin, *target, edges, active))
    });
    active.remove(&current);
    found
}
