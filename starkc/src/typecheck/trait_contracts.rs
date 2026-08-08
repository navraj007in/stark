//! **AS7 correction (2026-08-09) — trait machinery that must convert written types.**
//!
//! Sits ABOVE `convert`: may depend on `convert`, `traits`, `infer`, `state`, `types`.
//!
//! Packet 7 split trait IDENTITY below `convert` and stopped on a `convert <-> traits` cycle, and
//! the owner ruling added `bounds` to break it. That was correct as far as it went, but the
//! conversion-dependent trait machinery — impl-rule validation, Core trait contract checking,
//! associated-function typing, the trait-impl index — stayed in `traits` and kept calling
//! `convert_hir_type`, so the cycle was never actually gone. The forcing test could not see it:
//! its ownership map recognised an enumerated list of visibility prefixes that omitted
//! `pub(super)`, which is what every extracted method uses, so it observed 36 of 234 methods.
//!
//! The rule this module makes structural:
//!
//! ```text
//! traits            does this type stand in this trait relation, and which impl says so?
//!                   NO HIR type conversion
//! trait_contracts   questions that require knowing what a WRITTEN type means
//! ```

use super::state::PublishedEnv;
use super::state::TypeChecker;
use super::traits::{
    contract_ty_source, core_trait_contract, core_trait_source_name, is_copy_with_impls,
};
use super::types::{
    bound_receiver_ty, receiver_source, CallableDeclId, CallableSigTy, ContractTy,
    DispatchProvenance, GenericBinder, ReceiverAdjustment, ReceiverBinding, Ty, VariantFields,
};
use crate::ast::Primitive;
use crate::diag::Diagnostic;
use crate::hir::{self, BlockId, CoreType, ExprId, ItemId, Res, TypeId};
use crate::source::Span;
use std::collections::{HashMap, HashSet};

impl TypeChecker<'_> {
    pub(super) fn validate_impl_rules(&mut self) {
        type ImplRecord = (Option<Res>, Ty, HashSet<String>, Span);
        let mut impls: Vec<ImplRecord> = Vec::new();
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

            let impl_pkg = self.source_package(self.source_name(item.span));

            let trait_is_local = if let Some(Res::Item(trait_item_id)) = trait_res {
                if let Some(trait_file) = self.hir.item_file(trait_item_id) {
                    self.source_package(&trait_file.name) == impl_pkg
                } else {
                    false
                }
            } else {
                false
            };

            let self_type_is_local = match &self_ty {
                Ty::Struct(struct_item_id, _) | Ty::Enum(struct_item_id, _) => {
                    if let Some(type_file) = self.hir.item_file(*struct_item_id) {
                        self.source_package(&type_file.name) == impl_pkg
                    } else {
                        false
                    }
                }
                _ => false,
            };

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

            let mut conflicting = None;
            for (previous_trait, previous_ty, previous_methods, prev_span) in &impls {
                if *previous_trait == trait_res
                    && self.types_may_overlap(previous_ty, &self_ty)
                    && (trait_res.is_some() || !previous_methods.is_disjoint(&method_names))
                {
                    conflicting = Some(*prev_span);
                    break;
                }
            }

            if let Some(prev_span) = conflicting {
                // AS1b-ii-d: the record used to carry the impl's file alongside its span so this
                // note could name it. The span names it.
                let note = format!(
                    "conflicting implementation found in {} at {:?}",
                    self.source_name(prev_span),
                    prev_span
                );
                self.diags.push(
                    Diagnostic::error("overlapping implementation for the same type", item.span)
                        .with_code("E0500")
                        .with_label("another applicable impl already exists")
                        .with_note(note),
                );
            }
            impls.push((trait_res, self_ty.clone(), method_names, item.span));

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

                    // WP-C7.9 Packet B: duplicates. The membership checks above are set
                    // differences, and a set cannot see that the same name was implemented twice —
                    // so two `fn eq` bodies in one impl block reached name resolution with the
                    // second silently shadowing or colliding with the first.
                    let mut counts: HashMap<String, usize> = HashMap::new();
                    for impl_item in items {
                        if let hir::ImplItem::Fn { def, .. } = impl_item {
                            *counts
                                .entry(self.text(def.sig.name).to_string())
                                .or_insert(0) += 1;
                        }
                    }
                    for impl_item in items {
                        if let hir::ImplItem::Fn { def, .. } = impl_item {
                            let name = self.text(def.sig.name).to_string();
                            if counts.get(&name).copied().unwrap_or(0) > 1 {
                                self.diags.push(
                                    Diagnostic::error(
                                        format!(
                                            "method '{name}' is implemented more than once in this \
                                             implementation block"
                                        ),
                                        def.sig.span,
                                    )
                                    .with_code("E0500"),
                                );
                            }
                        }
                    }
                }
            }

            // WP-C7.9 Packet B: the same conformance obligation for a compiler-known trait, which
            // has no HIR declaration item for the block above to compare against.
            if let Some(trait_ref) = trait_ {
                if let Res::CoreTrait(core_trait) = trait_ref.res {
                    self.check_core_trait_impl(core_trait, trait_ref, &self_ty, items, item.span);
                }
            }

            // AS1b-ii-d: the diagnostics this item produced used to be stamped with `self.file`
            // here, because a span could not say which file it indexed. It can now, so there is
            // nothing to stamp and `start_len` has no reader.
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

    /// WP-C7.9 Packet B: a Core trait's implementation must conform before any body is executable.
    ///
    /// **What was missing.** A user-declared trait is an HIR item, so its declaration is available
    /// to compare an impl against, and `trait_method_signature_matches` does exactly that. A
    /// `CoreTrait` has no such item — every `impl Ord for T` writes its own signature and nothing
    /// checked it. `fn cmp(&self, other: &Self) -> Bool` therefore type-checked, lowered, and only
    /// failed at execution, differently in each engine.
    ///
    /// The contract now comes from one canonical table ([`core_trait_contract`]) rather than from
    /// checks scattered through the operator paths, and it is compared with the same key machinery
    /// the user-trait path uses — so `Self`, the written self type, associated types and the
    /// trait's own arguments normalise identically for both trait kinds.
    pub(super) fn check_core_trait_impl(
        &mut self,
        core_trait: hir::CoreTrait,
        trait_ref: &hir::TraitRef,
        self_ty: &Ty,
        items: &[hir::ImplItem],
        impl_span: Span,
    ) {
        let Some(contract) = core_trait_contract(core_trait) else {
            return;
        };
        let trait_name = core_trait_source_name(core_trait);
        // `Self` in a signature resolves through `current_self_ty`, and converting a type without
        // it both fails and reports a spurious "use of 'Self' outside impl or trait". The impl's
        // own self type is exactly what it should be here.
        let saved_self_ty = self.enter_self_scope(self_ty.clone());

        let associated: HashMap<String, TypeId> = items
            .iter()
            .filter_map(|item| match item {
                hir::ImplItem::AssocType { name, ty } => Some((self.text(*name).to_string(), *ty)),
                _ => None,
            })
            .collect();
        // The trait's own arguments, as written in `impl From<Celsius> for F`. A contract term
        // `TraitArg(n)` is compared against the n-th of these.
        let trait_args: Vec<TypeId> = trait_ref
            .args
            .as_ref()
            .map(|args| {
                args.args
                    .iter()
                    .filter_map(|arg| match arg {
                        hir::GenericArg::Type(ty) => Some(*ty),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();

        // ---- item membership: missing, extra, duplicate ----

        let mut seen_methods: HashMap<String, usize> = HashMap::new();
        for item in items {
            if let hir::ImplItem::Fn { def, .. } = item {
                *seen_methods
                    .entry(self.text(def.sig.name).to_string())
                    .or_insert(0) += 1;
            }
        }
        let mut seen_assoc: HashMap<String, usize> = HashMap::new();
        for item in items {
            if let hir::ImplItem::AssocType { name, .. } = item {
                *seen_assoc.entry(self.text(*name).to_string()).or_insert(0) += 1;
            }
        }

        for method in contract.methods {
            if !seen_methods.contains_key(method.name) {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "implementation of '{trait_name}' is missing method '{}'",
                            method.name
                        ),
                        impl_span,
                    )
                    .with_code("E0500")
                    .with_note(format!(
                        "'{trait_name}' declares {}",
                        self.core_method_source(trait_name, method)
                    )),
                );
            }
        }
        for assoc in contract.assoc_types {
            if !seen_assoc.contains_key(*assoc) {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "implementation of '{trait_name}' is missing associated type '{assoc}'"
                        ),
                        impl_span,
                    )
                    .with_code("E0500"),
                );
            }
        }
        for item in items {
            match item {
                hir::ImplItem::Fn { def, .. } => {
                    let name = self.text(def.sig.name).to_string();
                    if !contract.methods.iter().any(|m| m.name == name) {
                        self.diags.push(
                            Diagnostic::error(
                                format!("method '{name}' is not declared by '{trait_name}'"),
                                def.sig.span,
                            )
                            .with_code("E0500"),
                        );
                    } else if seen_methods.get(&name).copied().unwrap_or(0) > 1 {
                        self.diags.push(
                            Diagnostic::error(
                                format!(
                                    "method '{name}' is implemented more than once for \
                                     '{trait_name}'"
                                ),
                                def.sig.span,
                            )
                            .with_code("E0500"),
                        );
                    }
                }
                hir::ImplItem::AssocType { name, .. } => {
                    let text = self.text(*name).to_string();
                    if !contract.assoc_types.contains(&text.as_str()) {
                        self.diags.push(
                            Diagnostic::error(
                                format!(
                                    "associated type '{text}' is not declared by '{trait_name}'"
                                ),
                                *name,
                            )
                            .with_code("E0500"),
                        );
                    } else if seen_assoc.get(&text).copied().unwrap_or(0) > 1 {
                        self.diags.push(
                            Diagnostic::error(
                                format!(
                                    "associated type '{text}' is declared more than once for \
                                     '{trait_name}'"
                                ),
                                *name,
                            )
                            .with_code("E0500"),
                        );
                    }
                }
            }
        }

        // ---- signature conformance, per declared method ----

        for method in contract.methods {
            let Some(sig) = items.iter().find_map(|item| match item {
                hir::ImplItem::Fn { def, .. } if self.text(def.sig.name) == method.name => {
                    Some(&def.sig)
                }
                _ => None,
            }) else {
                continue; // already reported as missing
            };

            if sig.receiver != method.receiver {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "method '{}' of '{trait_name}' must take {}, but this implementation \
                             takes {}",
                            method.name,
                            receiver_source(method.receiver),
                            receiver_source(sig.receiver)
                        ),
                        sig.span,
                    )
                    .with_code("E0500"),
                );
            }

            if !sig.generics.is_empty() {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "method '{}' of '{trait_name}' declares no type parameters, but this \
                             implementation declares {}",
                            method.name,
                            sig.generics.len()
                        ),
                        sig.span,
                    )
                    .with_code("E0500"),
                );
            }

            if sig.params.len() != method.params.len() {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "method '{}' of '{trait_name}' takes {} parameter(s) after the \
                             receiver, but this implementation takes {}",
                            method.name,
                            method.params.len(),
                            sig.params.len()
                        ),
                        sig.span,
                    )
                    .with_code("E0500")
                    .with_note(format!(
                        "'{trait_name}' declares {}",
                        self.core_method_source(trait_name, method)
                    )),
                );
            } else {
                for (position, (expected, param)) in
                    method.params.iter().zip(&sig.params).enumerate()
                {
                    let expected_ty =
                        self.contract_ty(*expected, self_ty, &associated, &trait_args);
                    let actual_ty = self.convert_hir_type(param.ty);
                    // `Ty::Error` on either side means something else already failed; blaming the
                    // signature too would be a second diagnostic for one cause.
                    if !matches!(expected_ty, Ty::Error)
                        && !matches!(actual_ty, Ty::Error)
                        && self.ty_signature_key(&expected_ty) != self.ty_signature_key(&actual_ty)
                    {
                        self.diags.push(
                            Diagnostic::error(
                                format!(
                                    "parameter {} of '{}' must have type '{}', but this \
                                     implementation writes '{}'",
                                    position + 1,
                                    method.name,
                                    contract_ty_source(*expected),
                                    self.text(self.hir.ty(param.ty).span)
                                ),
                                self.hir.ty(param.ty).span,
                            )
                            .with_code("E0500"),
                        );
                    }
                }
            }

            match (method.ret, sig.ret) {
                (None, hir::RetTy::Unit) => {}
                (None, _) => {
                    self.diags.push(
                        Diagnostic::error(
                            format!(
                                "method '{}' of '{trait_name}' returns Unit, but this \
                                 implementation declares a return type",
                                method.name
                            ),
                            sig.span,
                        )
                        .with_code("E0500"),
                    );
                }
                (Some(expected), hir::RetTy::Ty(actual)) => {
                    let expected_ty = self.contract_ty(expected, self_ty, &associated, &trait_args);
                    let actual_ty = self.convert_hir_type(actual);
                    // Two spellings, two normalisations. `Self` and the written self type are
                    // reconciled by converting both to a `Ty`; `Self::Item` is not — it is resolved
                    // by `signature_type_key`, through the impl's own associated declarations,
                    // which is how the user-trait path has always compared it. A signature is
                    // conformant if either normalisation says so, because the two spellings mean
                    // the same thing and an impl may write either (WP-C6.2b-F6's rule, extended).
                    let self_key = self.ty_signature_key(self_ty);
                    let written_key =
                        self.signature_type_key(actual, &self_key, &associated, &HashMap::new());
                    let contract_key = match expected {
                        ContractTy::OptionAssoc(name) => {
                            let inner = associated.get(name).map_or_else(
                                || format!("assoc:{name}"),
                                |ty| {
                                    self.signature_type_key(
                                        *ty,
                                        &self_key,
                                        &associated,
                                        &HashMap::new(),
                                    )
                                },
                            );
                            format!("core:{:?}<{inner}>", CoreType::Option)
                        }
                        _ => String::new(),
                    };
                    let assoc_spelling_matches =
                        !contract_key.is_empty() && contract_key == written_key;
                    if !assoc_spelling_matches
                        && !matches!(expected_ty, Ty::Error)
                        && !matches!(actual_ty, Ty::Error)
                        && self.ty_signature_key(&expected_ty) != self.ty_signature_key(&actual_ty)
                    {
                        self.diags.push(
                            Diagnostic::error(
                                format!(
                                    "method '{}' of '{trait_name}' must return '{}', but this \
                                     implementation returns '{}'",
                                    method.name,
                                    contract_ty_source(expected),
                                    self.text(self.hir.ty(actual).span)
                                ),
                                self.hir.ty(actual).span,
                            )
                            .with_code("E0500"),
                        );
                    }
                }
                (Some(expected), _) => {
                    self.diags.push(
                        Diagnostic::error(
                            format!(
                                "method '{}' of '{trait_name}' must return '{}', but this \
                                 implementation returns Unit",
                                method.name,
                                contract_ty_source(expected)
                            ),
                            sig.span,
                        )
                        .with_code("E0500"),
                    );
                }
            }
        }
        self.exit_self_scope(saved_self_ty);
    }

    pub(super) fn associated_fn_type(
        &mut self,
        nominal: ItemId,
        name_span: Span,
        turbofish: Option<&hir::GenericArgs>,
        call_span: Span,
        use_expr: ExprId,
    ) -> Ty {
        let name = self.text(name_span).to_string();
        let mut inherent = Vec::new();
        let mut trait_candidates = Vec::new();
        for (impl_idx, item) in self.hir.items.iter().enumerate() {
            let hir::ItemKind::Impl {
                trait_,
                self_ty,
                items,
                generics,
            } = &item.kind
            else {
                continue;
            };
            if !matches!(
                &self.hir.ty(*self_ty).kind,
                hir::TypeKind::Path { res: Res::Item(item), .. } if *item == nominal
            ) {
                continue;
            }
            let impl_item_id = ItemId(impl_idx as u32);
            let candidate = items.iter().find_map(|item| match item {
                // WP-C6.2b-F1: capture visibility + defining impl for the private-member check.
                //
                // DEV-148: `item_text`, NOT `text`. A member's name is a span into the file that
                // DECLARED the impl, and `self.text` slices whichever file is currently being
                // checked. Across a module boundary those differ, so the comparison sliced the
                // wrong file and matched garbage — `make` came back as `"rap:"`, and a name that
                // ran past the shorter file's end came back as `"?"`. No candidate ever matched,
                // and the caller got "associated function not found" for a function that plainly
                // exists.
                //
                // METHODS were unaffected because method lookup selects on the receiver's TYPE
                // rather than by slicing a name, which is exactly why the two diverged and why
                // this looked like a visibility or coherence rule rather than a text bug.
                hir::ImplItem::Fn { vis, def }
                    if def.sig.receiver.is_none()
                        && self.item_text(impl_item_id, def.sig.name) == name =>
                {
                    Some((
                        def.sig.clone(),
                        *self_ty,
                        generics.clone(),
                        matches!(vis, Some(crate::ast::Vis::Pub)),
                        impl_item_id,
                    ))
                }
                _ => None,
            });
            if let Some(candidate) = candidate {
                if trait_.is_none() {
                    inherent.push(candidate);
                } else {
                    trait_candidates.push(candidate);
                }
            }
        }
        let candidates = if inherent.is_empty() {
            trait_candidates
        } else {
            inherent
        };
        if candidates.len() > 1 {
            self.diags.push(
                Diagnostic::error(
                    format!("associated function '{name}' is ambiguous"),
                    name_span,
                )
                .with_code("E0204"),
            );
            return Ty::Error;
        }
        let selected = candidates.into_iter().next();
        let Some((sig, self_ty_id, impl_generics, is_pub, impl_item_id)) = selected else {
            self.diags.push(
                Diagnostic::error(format!("associated function '{name}' not found"), name_span)
                    .with_code("E0200"),
            );
            return Ty::Error;
        };
        // WP-C6.2b-F1: a private associated function is inaccessible outside its defining module.
        self.check_member_visible(
            is_pub,
            impl_item_id,
            "associated function",
            &name,
            call_span,
        );

        // DEV-148: everything from here until the context is restored reads spans belonging to
        // the IMPL's file, not the caller's. The names must be sliced consistently on both sides —
        // the map's keys and the `Ty::Param`s they substitute into — or substitution silently
        // fails to fire and the caller sees a stray parameter type like `'r'`.
        let self_ty = self.convert_hir_type(self_ty_id);
        let previous_self = self.enter_self_scope(self_ty);
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
        self.exit_self_scope(previous_self);

        let mut map = HashMap::new();
        for param in &impl_generics {
            let infer = self.new_type_var();
            map.insert(self.item_text(impl_item_id, param.name).to_string(), infer);
        }
        if let Some(args) = turbofish {
            self.validate_generic_arity(sig.generics.len(), args.args.len(), call_span);
            for (param, arg) in sig.generics.iter().zip(&args.args) {
                let ty = match arg {
                    hir::GenericArg::Type(ty) => self.convert_hir_type(*ty),
                    _ => Ty::Error,
                };
                map.insert(self.item_text(impl_item_id, param.name).to_string(), ty);
            }
        } else {
            for param in &sig.generics {
                let infer = self.new_type_var();
                map.insert(self.item_text(impl_item_id, param.name).to_string(), infer);
            }
        }
        // A3c-S2/A4: an associated call has a generic environment like any other callable use.
        // It was the one publication site A3c-S missed, invisible until A4 resolved signatures:
        // the BODY worked because nothing needed the frame, and only the signature could tell.
        // Names are read against the IMPL's file, matching this `map`'s keys (DEV-101).
        let mut published_use: Option<(BlockId, Vec<(GenericBinder, Ty)>)> = None;
        if let Some((body, own_generics)) =
            self.hir
                .items
                .get(impl_item_id.0 as usize)
                .and_then(|item| match &item.kind {
                    hir::ItemKind::Impl { items, .. } => items.iter().find_map(|it| match it {
                        hir::ImplItem::Fn { def, .. }
                            if self.item_text(impl_item_id, def.sig.name) == name =>
                        {
                            Some((def.body, def.sig.generics.clone()))
                        }
                        _ => None,
                    }),
                    _ => None,
                })
        {
            let impl_names: Vec<String> = impl_generics
                .iter()
                .map(|param| self.item_text(impl_item_id, param.name).to_string())
                .collect();
            let own_names: Vec<String> = own_generics
                .iter()
                .map(|param| self.item_text(impl_item_id, param.name).to_string())
                .collect();
            let env_map = map.clone();
            self.publish_callable_env(PublishedEnv {
                call_expr: use_expr,
                body,
                self_ty: None,
                impl_names: &impl_names,
                own_names: &own_names,
                own_is_method: true,
                map: &env_map,
            });
            published_use = Some((
                body,
                Self::env_bindings(&None, &impl_names, &own_names, true, &env_map),
            ));
        }
        params = params
            .iter()
            .map(|ty| self.instantiate_ty(ty, &map))
            .collect();
        ret = self.instantiate_ty(&ret, &map);
        // AS3 Boundary 2: an associated function takes no receiver, so both receiver fields are
        // `None` — recorded rather than left absent, which is the same distinction Boundary 1 drew
        // for a non-generic call's empty environment.
        if let Some((body, bindings)) = published_use {
            self.publish_named_use(
                use_expr,
                body,
                bindings,
                ReceiverAdjustment::None,
                ReceiverBinding::None,
                CallableSigTy {
                    receiver: None,
                    params: params.clone(),
                    ret: ret.clone(),
                },
                DispatchProvenance::Qualified { trait_item: None },
            );
        }
        Ty::Fn {
            params,
            ret: Box::new(ret),
        }
    }

    /// Build the program's coherent dispatch index (AS3 Boundary 4a).
    ///
    /// Built in the CHECKER, which already has converted self-types and knows every declaration,
    /// and frozen into `TypeTables`. Building it in each engine would be two indexes of one fact —
    /// which is the shape of `find_method` and `find_impl_fn`.
    ///
    /// Records the **effective** target per member: the impl's override where one exists, otherwise
    /// the trait's default body (G1), together with the binder namespace that body owns.
    pub(super) fn build_trait_impl_index(&mut self) -> crate::bound_dispatch::TraitImplIndex {
        let mut impls = Vec::new();
        for (idx, item) in self.hir.items.iter().enumerate() {
            let impl_item = ItemId(idx as u32);
            let hir::ItemKind::Impl {
                trait_,
                self_ty,
                items,
                generics,
            } = &item.kind
            else {
                continue;
            };
            let trait_ref = trait_.clone();
            let self_ty_id = *self_ty;
            let generic_names: Vec<String> = generics
                .iter()
                .map(|param| self.item_text(impl_item, param.name).to_string())
                .collect();
            // Members written in the impl.
            let mut effective_members: Vec<crate::bound_dispatch::IndexedTarget> = Vec::new();
            let mut written: Vec<String> = Vec::new();
            for (member, impl_item_node) in items.iter().enumerate() {
                let hir::ImplItem::Fn { def, .. } = impl_item_node else {
                    continue;
                };
                let name = self.item_text(impl_item, def.sig.name).to_string();
                written.push(name.clone());
                let mut binders = vec![GenericBinder::SelfType];
                for (position, impl_name) in generic_names.iter().enumerate() {
                    binders.push(GenericBinder::ImplParam {
                        index: position,
                        name: impl_name.clone(),
                    });
                }
                for (position, param) in def.sig.generics.iter().enumerate() {
                    binders.push(GenericBinder::MethodParam {
                        index: position,
                        name: self.item_text(impl_item, param.name).to_string(),
                    });
                }
                effective_members.push(crate::bound_dispatch::IndexedTarget {
                    member: name,
                    declaration: CallableDeclId::ImplMember {
                        impl_item,
                        member: member as u32,
                    },
                    body: def.body,
                    binders,
                });
            }
            // G1: trait defaults the impl did NOT override are still executable targets, and their
            // bodies own the TRAIT's binder namespace rather than the impl's.
            let bound_trait = trait_ref.as_ref().map(|reference| match reference.res {
                Res::Item(trait_id) => hir::BoundTrait::User(trait_id),
                Res::CoreTrait(core) => hir::BoundTrait::Core(core),
                _ => hir::BoundTrait::User(impl_item),
            });
            if let Some(hir::BoundTrait::User(trait_id)) = bound_trait {
                if let hir::ItemKind::Trait {
                    items: trait_items,
                    generics: trait_generics,
                    ..
                } = &self.hir.item(trait_id).kind
                {
                    let trait_generics = trait_generics.to_vec();
                    for (member, trait_item) in trait_items.iter().enumerate() {
                        let hir::TraitItem::Method {
                            sig,
                            body: Some(body),
                        } = trait_item
                        else {
                            continue;
                        };
                        let name = self.item_text(trait_id, sig.name).to_string();
                        if written.contains(&name) {
                            continue;
                        }
                        let mut binders = vec![GenericBinder::SelfType];
                        for (position, param) in trait_generics.iter().enumerate() {
                            binders.push(GenericBinder::TraitParam {
                                index: position,
                                name: self.item_text(trait_id, param.name).to_string(),
                            });
                        }
                        for (position, param) in sig.generics.iter().enumerate() {
                            binders.push(GenericBinder::MethodParam {
                                index: position,
                                name: self.item_text(trait_id, param.name).to_string(),
                            });
                        }
                        effective_members.push(crate::bound_dispatch::IndexedTarget {
                            member: name,
                            declaration: CallableDeclId::TraitMember {
                                trait_item: trait_id,
                                member: member as u32,
                            },
                            body: *body,
                            binders,
                        });
                    }
                }
            }
            let converted_self = self.convert_hir_type(self_ty_id);
            let trait_args: Vec<Ty> = trait_ref
                .as_ref()
                .and_then(|reference| reference.args.as_ref())
                .map(|args| {
                    args.args
                        .iter()
                        .filter_map(|arg| match arg {
                            hir::GenericArg::Type(id) => Some(*id),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
                .into_iter()
                .map(|id| self.convert_hir_type(id))
                .collect();
            impls.push(crate::bound_dispatch::IndexedImpl {
                impl_item,
                trait_: bound_trait,
                trait_args,
                self_ty: converted_self,
                generic_names,
                effective_members,
            });
        }
        crate::bound_dispatch::TraitImplIndex::from_parts(impls)
    }

    /// The `Ty` a contract term denotes for this implementation.
    ///
    /// Both sides of every comparison are converted to a `Ty` and keyed with the SAME function
    /// (`ty_signature_key`). Keying the expected side as a string and the actual side through
    /// `signature_type_key` looked equivalent and was not: a generic impl (`impl<T> Eq for W<T>`)
    /// renders its own parameter as `param:T` on one path and `g:0` on the other, so a correct
    /// `fn eq(&self, other: &W<T>) -> Bool` was rejected with "must have type '&Self', but this
    /// implementation writes '&W<T>'" — the two spellings the rule exists to treat as one.
    pub(super) fn contract_ty(
        &mut self,
        ty: ContractTy,
        self_ty: &Ty,
        associated: &HashMap<String, TypeId>,
        trait_args: &[TypeId],
    ) -> Ty {
        match ty {
            ContractTy::SelfTy => self_ty.clone(),
            ContractTy::RefSelf => Ty::Ref {
                mutable: false,
                inner: Box::new(self_ty.clone()),
            },
            ContractTy::Bool => Ty::Primitive(Primitive::Bool),
            ContractTy::UInt64 => Ty::Primitive(Primitive::UInt64),
            ContractTy::StringTy => Ty::Primitive(Primitive::String),
            ContractTy::Ordering => Ty::Core(CoreType::Ordering, Vec::new()),
            ContractTy::OptionAssoc(name) => {
                let item = associated
                    .get(name)
                    .map(|ty| self.convert_hir_type(*ty))
                    .unwrap_or(Ty::Error);
                Ty::Core(CoreType::Option, vec![item])
            }
            ContractTy::TraitArg(index) => trait_args
                .get(index)
                .map(|ty| self.convert_hir_type(*ty))
                .unwrap_or(Ty::Error),
        }
    }

    /// The declared signature of an impl member, converted — receiver, parameters and result.
    ///
    /// AS3 Boundary 3: the operator publication reads the signature it publishes rather than
    /// asserting one, so `callable_uses` and `callable_types` describe the same declaration.
    #[allow(clippy::type_complexity)]
    pub(super) fn declared_member_signature(
        &mut self,
        impl_item: ItemId,
        member: u32,
    ) -> Option<(Option<Ty>, Vec<Ty>, Ty)> {
        let hir::ItemKind::Impl { items, self_ty, .. } = &self.hir.item(impl_item).kind else {
            return None;
        };
        let hir::ImplItem::Fn { def, .. } = items.get(member as usize)? else {
            return None;
        };
        // Take the ids out of the borrow before converting: `convert_hir_type` needs `&mut self`,
        // and `FnDef` is not `Clone`, so the borrow has to end rather than be copied.
        let receiver_form = def.sig.receiver;
        let param_ids: Vec<hir::TypeId> = def.sig.params.iter().map(|p| p.ty).collect();
        let ret_form = def.sig.ret;
        let self_ty_id = *self_ty;

        let self_ty = self.convert_hir_type(self_ty_id);
        let receiver = bound_receiver_ty(receiver_form.as_ref(), self_ty);
        let params = param_ids
            .into_iter()
            .map(|id| self.convert_hir_type(id))
            .collect();
        let ret = match ret_form {
            hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
            hir::RetTy::Ty(id) => self.convert_hir_type(id),
            hir::RetTy::Never(_) => Ty::Never,
        };
        Some((receiver, params, ret))
    }

    /// The trait's own default body for `member`, as an (owner, member, body) triple shaped like
    /// [`Self::operator_impl_member`]'s. Used when an implementor accepts the default and there is
    /// therefore no impl member to find.
    /// A TRAIT method's declared signature, with `Self` bound to the concrete receiver. The
    /// trait-default counterpart of [`Self::declared_member_signature`], which reads an impl.
    pub(super) fn trait_member_signature(
        &mut self,
        trait_id: ItemId,
        member: u32,
        self_ty: &Ty,
    ) -> Option<(Option<Ty>, Vec<Ty>, Ty)> {
        let hir::ItemKind::Trait { items, .. } = &self.hir.item(trait_id).kind else {
            return None;
        };
        let hir::TraitItem::Method { sig, .. } = items.get(member as usize)? else {
            return None;
        };
        let receiver_form = sig.receiver;
        let param_ids: Vec<hir::TypeId> = sig.params.iter().map(|p| p.ty).collect();
        let ret_form = sig.ret;
        let receiver = bound_receiver_ty(receiver_form.as_ref(), self_ty.clone());
        let params = param_ids
            .into_iter()
            .map(|id| self.convert_hir_type(id))
            .collect();
        let ret = match ret_form {
            hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
            hir::RetTy::Ty(id) => Self::subst_self_ty(self.convert_hir_type(id), self_ty),
            hir::RetTy::Never(_) => Ty::Never,
        };
        Some((receiver, params, ret))
    }

    /// WP-C6.2c: associated-type projections pinned by explicit binding constraints in scope
    /// (`T: Holder<Item = Int32>` yields `"T::Item" -> Int32`), gathered from the current function's
    /// and enclosing impl's generic parameters.
    pub(super) fn assoc_binding_map(&mut self) -> HashMap<String, Ty> {
        let mut generics = self.current_fn_generics.clone().unwrap_or_default();
        if let Some(impl_generics) = &self.current_impl_generics {
            generics.extend(impl_generics.iter().cloned());
        }
        let mut map = HashMap::new();
        for param in &generics {
            let pname = self.text(param.name).to_string();
            for bound in &param.bounds {
                let Some(bound_args) = &bound.args else {
                    continue;
                };
                for arg in &bound_args.args {
                    if let hir::GenericArg::Binding { name, ty } = arg {
                        let key = format!("{}::{}", pname, self.text(*name));
                        let bty = self.convert_hir_type(*ty);
                        map.insert(key, bty);
                    }
                }
            }
        }
        map
    }
}
