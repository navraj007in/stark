//! **AS7 Packet 7 — trait identity and impl selection.**
//!
//! Depends on `types`, `state` and `infer`. **It may not depend on `convert` or `bounds`, and the
//! dependency checker enforces that** — which is the whole reason this module exists as a layer
//! separate from `bounds`.
//!
//! This layer answers one question: *does this type stand in this trait relation, and which impl
//! makes that true?* It deliberately knows nothing about written associated-type bindings
//! (`Iterator<Item = Foo>`), because answering those requires converting HIR types, and a module
//! that converted HIR types here would close the `convert <-> traits` cycle that stopped Packet 7.

use super::state::PublishedEnv;
use super::state::TypeChecker;
use super::types::ordered_primitive;
use super::types::receiver_source;
use super::types::strip_ref;
use super::types::UserIteratorSelection;
use super::types::{is_float_primitive, is_numeric, standard_display_type, standard_hash_type};
use super::types::{
    CallableDeclId, CallableSigTy, DispatchProvenance, ExtensionTy, GenericBinder,
    ReceiverAdjustment, ReceiverBinding, Ty, VariantFields,
};
use crate::diag::Diagnostic;
use crate::source::Span;

use crate::ast::Primitive;
use crate::hir::{self, BlockId, CoreType, ExprId, Hir, ItemId, Res, TypeId};
use std::collections::{HashMap, HashSet};

/// Why a trait relation holds **for a nominal type**. `Impl` carries the selected impl so that the
/// layer above can look up *that impl's* associated types — the same impl this layer chose, which
/// is what preserves the existing behaviour when several traits expose the same associated name.
///
/// The owner ruling sketched a third variant, `Yes`, for "holds with no impl to point at". It is
/// deliberately absent: this witness is only ever produced for `Ty::Struct`/`Ty::Enum`, and every
/// non-nominal case — primitives, Core containers, references, generic parameters — is answered
/// as a `bool` by `satisfies_bound_identity` before a witness is ever requested. A `Yes` variant
/// would be unconstructible, and CI's `-D warnings` says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BoundWitness {
    /// No impl discharges the bound.
    No,
    /// It holds because of this `impl <Trait> for <Ty>` item.
    Impl(ItemId),
}

impl TypeChecker<'_> {
    /// Select the `impl <bound> for <nominal>` that discharges a bound, if one exists.
    ///
    /// Extracted verbatim from `satisfies_bound_parts`' `Ty::Struct | Ty::Enum` arm: same scan,
    /// same nominal test, same `res`-or-name comparison. It stops one step earlier than the
    /// original, returning the impl rather than its associated-type map, so that no HIR type is
    /// converted at this layer.
    pub(super) fn bound_impl_witness(
        &mut self,
        struct_id: &ItemId,
        bound_name: &str,
        bound_res: Option<Res>,
    ) -> BoundWitness {
        let found = self.hir.items.iter().enumerate().find_map(|(index, item)| {
            let hir::ItemKind::Impl {
                self_ty: impl_self_ty_id,
                trait_: Some(trait_ref),
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
                || (Some(trait_ref.res) != bound_res
                    && self.text(trait_ref.path.span) != bound_name)
            {
                return None;
            }
            Some(ItemId(index as u32))
        });
        match found {
            Some(id) => BoundWitness::Impl(id),
            None => BoundWitness::No,
        }
    }

    /// The associated types an impl declares, by name. Pure HIR reading — no conversion.
    pub(super) fn impl_assoc_type_ids(&self, impl_item: ItemId) -> HashMap<String, TypeId> {
        let hir::ItemKind::Impl { items, .. } = &self.hir.item(impl_item).kind else {
            return HashMap::new();
        };
        items
            .iter()
            .filter_map(|item| match item {
                hir::ImplItem::AssocType { name, ty } => Some((self.text(*name).to_string(), *ty)),
                _ => None,
            })
            .collect()
    }

    /// WP-C7.9 Packet I: bound satisfaction, addressable by NAME rather than only by a written
    /// `TraitRef`.
    ///
    /// The obligations the *implementation itself* imposes — `HashMap<K, V>` requiring
    /// `K: Hash + Eq`, for instance — have no trait reference in the source to check against:
    /// nobody wrote them, the standard library declares them. Before this, that meant they were not
    /// checked at all (DEV-118). Splitting the name out of the reference lets one mechanism serve
    /// both: a written bound passes its own name, and a built-in obligation passes the name the
    /// specification states.
    pub(super) fn satisfies_bound_identity(
        &mut self,
        ty: &Ty,
        bound_name: &str,
        bound_res: Option<Res>,
    ) -> bool {
        let ty = self.resolve(ty);
        let bound_name = bound_name.to_string();

        match &ty {
            Ty::Ref { mutable: _, inner } => {
                if bound_name == "Eq"
                    || bound_name == "Ord"
                    || bound_name == "Clone"
                    || bound_name == "Hash"
                    || bound_name == "Display"
                {
                    self.satisfies_bound_identity(inner, &bound_name, bound_res)
                } else {
                    false
                }
            }
            Ty::Primitive(p) => {
                if bound_name == "Num" {
                    is_numeric(*p)
                } else if bound_name == "Eq" {
                    // DEV-075 matrix: every primitive except `Unit` and the floats (CD-015).
                    !matches!(p, Primitive::Unit) && !is_float_primitive(*p)
                } else if bound_name == "Ord" {
                    // DEV-075 matrix: as `Eq`, and additionally NOT `Bool`. `Char` is ordered.
                    !matches!(p, Primitive::Unit | Primitive::Bool) && !is_float_primitive(*p)
                } else if bound_name == "Display" {
                    standard_display_type(&ty)
                } else if bound_name == "Clone" || bound_name == "Default" {
                    true
                } else if bound_name == "Hash" {
                    standard_hash_type(&ty)
                } else {
                    false
                }
            }
            Ty::Core(core_type, args) => {
                if bound_name == "Clone" {
                    args.clone()
                        .iter()
                        .all(|arg| self.satisfies_bound_identity(arg, &bound_name, bound_res))
                } else if bound_name == "Display" {
                    standard_display_type(&ty)
                } else if bound_name == "Hash" {
                    standard_hash_type(&ty)
                } else if bound_name == "Eq" || bound_name == "Ord" {
                    args.clone()
                        .iter()
                        .all(|arg| self.satisfies_bound_identity(arg, &bound_name, bound_res))
                } else if bound_name == "Default" {
                    *core_type == CoreType::Vec
                        || *core_type == CoreType::Option
                        || *core_type == CoreType::HashMap
                        || *core_type == CoreType::HashSet
                } else if bound_name == "Iterator" {
                    *core_type == CoreType::CharsIter
                        || *core_type == CoreType::SplitIter
                        || *core_type == CoreType::VecIter
                        || *core_type == CoreType::KeysIter
                        || *core_type == CoreType::ValuesIter
                        || *core_type == CoreType::Iter
                        || *core_type == CoreType::MapIter
                        || *core_type == CoreType::FilterIter
                } else {
                    false
                }
            }
            Ty::Struct(struct_id, _) | Ty::Enum(struct_id, _) => {
                // AS7 Packet 7: trait IDENTITY is `traits`' answer; the written associated-type
                // bindings are this layer's. Same impl, same actual/expected conversion ordering
                // as before the split — see AS7-MODULE-DAG.md's correction section.
                // Identity stops here: an impl exists, or it does not. Whether the WRITTEN
                // associated-type bindings match is `bounds`' question, because answering it
                // requires converting HIR types — which this layer must never do.
                matches!(
                    self.bound_impl_witness(struct_id, &bound_name, bound_res),
                    BoundWitness::Impl(_)
                )
            }
            // DEV-067(a) (WP-C4.7-7): a bound on a generic parameter is discharged by the
            // ENCLOSING function's own declared bounds. There was no `Ty::Param` arm at all, so
            // this fell to `_ => false` and any generic fn calling another generic fn with a
            // bounded parameter — including simple recursion — failed E0500 "type 'T' does not
            // satisfy trait bound 'Ord'", even though `T: Ord` was declared right there
            // (TYPE-GENERIC-001: the caller's own bound discharges the callee's obligation).
            // This mirrors the `Ty::Param` arm `ty_satisfies_operator_bound` already had for the
            // operator-desugaring bounds, so the two bound checks now agree about parameters.
            Ty::Param(param_name) => self.param_declares_bound(param_name, &bound_name, bound_res),
            Ty::Error => true,
            _ => false,
        }
    }

    // AS7 Packet 7: moved to the layer that owns the question.
    /// Whether generic parameter `param_name` carries the bound being discharged.
    ///
    /// **By resolved identity when there is one.** `required_res` is the obligation's own
    /// resolution, so two bounds naming the same trait match however each was SPELLED:
    ///
    /// ```text
    /// use traits::Render;
    /// fn inner<U: traits::Render>(v: &U) { }
    /// fn outer<T: Render>(v: &T) { inner(v) }      // the same trait, two spellings
    /// ```
    ///
    /// Comparing spellings rejected that — an over-refusal, and the reason the first version of
    /// this split was only half a repair. It also could not have distinguished `left::Render` from
    /// `right::Render` had both been reachable unqualified, which is the same defect pointing the
    /// other way.
    ///
    /// **Spelling remains the fallback, and only for obligations with no resolution.** DEV-118's
    /// built-in obligations — `HashMap<K, V>` requiring `K: Hash + Eq` — have no `TraitRef` in any
    /// source, because nobody wrote them; the standard library states them. Those arrive with
    /// `required_res == None` and are matched by name, which is the only handle they have.
    pub(super) fn param_declares_bound(
        &self,
        param_name: &str,
        required: &str,
        required_res: Option<Res>,
    ) -> bool {
        let required_identity = required_res.and_then(|res| hir::bound_trait_of_res(self.hir, res));
        self.current_impl_generics
            .iter()
            .flatten()
            .chain(self.current_fn_generics.iter().flatten())
            .any(|param| {
                self.text(param.name) == param_name
                    && param.bounds.iter().any(|bound| match required_identity {
                        Some(wanted) => hir::resolved_bound_trait(self.hir, bound) == Some(wanted),
                        None => self.text(bound.path.span) == required,
                    })
            })
    }

    // AS7 Packet 8: moved to the layer that owns the question.
    pub(super) fn is_copy_ty(&mut self, ty: &Ty) -> bool {
        let resolved = self.resolve(ty);
        let copy_types = copy_eligible_types(self.hir);
        is_copy_with_impls(&resolved, &copy_types)
    }
}

// AS7 Packet 8: Copy is a trait relation, so its eligibility set belongs with identity.
pub fn copy_eligible_types(hir: &Hir) -> HashSet<ItemId> {
    // One authority, consulted rather than repeated: this scan used to compute `drop_items` inline.
    let drop_items = nominals_with_destructor(hir);
    let mut eligible: HashSet<ItemId> = HashSet::new();
    for item in hir.items.iter() {
        if let hir::ItemKind::Impl {
            trait_: Some(trait_ref),
            self_ty,
            ..
        } = &item.kind
        {
            if let hir::TypeKind::Path {
                res: Res::Item(target),
                ..
            } = &hir.ty(*self_ty).kind
            {
                // An explicit `impl Copy` seeds the set; its field validity is checked separately
                // (a `Copy`+non-`Copy`-field type is a reported error).
                if trait_ref.res == Res::CoreTrait(hir::CoreTrait::Copy) {
                    eligible.insert(*target);
                }
            }
        }
    }
    // Fixpoint: a nominal joins the set once all its fields are eligible under the current set.
    // Terminates because the set only grows and is bounded by the item count.
    loop {
        let mut changed = false;
        for (idx, item) in hir.items.iter().enumerate() {
            let id = ItemId(idx as u32);
            if eligible.contains(&id) || drop_items.contains(&id) {
                continue;
            }
            let field_tys: Vec<TypeId> = match &item.kind {
                hir::ItemKind::Struct { fields, .. } => fields.iter().map(|f| f.ty).collect(),
                // **OWN-COPY-001, amended (CD-251): a ZERO-VARIANT enum is never structurally
                // `Copy`.**
                //
                // The unamended rule reached the wrong answer by vacuous truth: "every payload of
                // every variant is `Copy`" is trivially true when there are no variants. That
                // reasoning silently assumes a value of the type arose from one of those variants.
                //
                // CD-234 makes that assumption false. A host-resource nominal is deliberately a
                // zero-variant enum -- opaque because nothing in source can construct one -- but its
                // values enter from an external provider. Vacuous `Copy` then made those values
                // freely duplicable, so `MatchDesugar` extracted a payload with `copy` and
                // exactly-once close was broken in the FRONT END, before MIR existed. (`MIR-0026`
                // rejected the result, which is how this was found.)
                //
                // General rule, not a provider marker: an enum is structurally `Copy` only if it has
                // at least one variant AND every payload of every variant is `Copy`. No existing
                // program can be affected, because no existing program could obtain a value of an
                // uninhabited type to copy.
                hir::ItemKind::Enum { variants, .. } if variants.is_empty() => continue,
                hir::ItemKind::Enum { variants, .. } => variants
                    .iter()
                    .flat_map(|v| match &v.kind {
                        hir::VariantKind::Unit => Vec::new(),
                        hir::VariantKind::Tuple(tys) => tys.clone(),
                        hir::VariantKind::Struct(fields) => fields.iter().map(|f| f.ty).collect(),
                    })
                    .collect(),
                _ => continue,
            };
            if field_tys
                .iter()
                .all(|t| field_ty_copy_eligible(hir, *t, &eligible))
            {
                eligible.insert(id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    eligible
}
pub(super) fn is_copy_with_impls(ty: &Ty, copy_types: &HashSet<ItemId>) -> bool {
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
        // DEV-062: function values are `Copy` per 03-Type-System.md §Copy and Drop ("reference
        // values, function values, `Unit`, and `!` are `Copy`") / TYPE-FN-001. This arm
        // previously listed `Ty::Fn` alongside `&mut`/slices as non-Copy, contradicting the
        // spec.
        Ty::Fn { .. } => true,
        Ty::Ref { mutable: true, .. } | Ty::Slice(_) | Ty::Range(_) => false,
        Ty::Extension(ext) => match &**ext {
            ExtensionTy::Tensor(tensor) => tensor.is_copy(),
            ExtensionTy::Model(_) => false,
            ExtensionTy::ModelError => false,
        },
    }
}

/// WP-C6.1g-a (OWN-COPY-001, amended): the set of nominal items that are `Copy` — the union of
/// items with an explicit `impl Copy` and items **structurally** eligible: every stored
/// field/payload recursively `Copy`, no `Drop` implementation, no owned non-`Copy` resource, no
/// `&mut` field. Computed once and shared by the type checker (`is_copy_with_impls`) and the move
/// checker (`borrowck`) so the two cannot disagree — a divergence there is the DEV-072 class.
///
/// Per-instance genericity is handled at the query, not here: this set answers "is `struct H` ever
/// `Copy`", and `is_copy_with_impls`/`is_copy_type` additionally require every type argument to be
/// `Copy` (`args.all(is_copy)`), so `H<&P>` is `Copy` while `H<String>` is not, from one set.
/// **AS4: the single authority for "does this nominal have a user destructor?"**
///
/// Answered by RESOLVED IDENTITY — `Res::CoreTrait(CoreTrait::Drop)` — never by the trait's
/// spelling. CD-379 settled that rule for `Display`; DEV-210 is the same defect found in the borrow
/// checker, which asked whether the written trait name `.ends_with("Drop")` and so refused a legal
/// partial move on any type implementing a user trait called `MyDrop`.
///
/// Extracted from `copy_eligible_types`, which already computed exactly this set for its own use
/// and kept it private. Publishing it costs nothing and removes the incentive to write a third
/// scan: every consumer of "has a destructor" now reads one answer.
pub fn nominals_with_destructor(hir: &Hir) -> HashSet<ItemId> {
    let mut drop_items: HashSet<ItemId> = HashSet::new();
    for item in hir.items.iter() {
        let hir::ItemKind::Impl {
            trait_: Some(trait_ref),
            self_ty,
            ..
        } = &item.kind
        else {
            continue;
        };
        if trait_ref.res != Res::CoreTrait(hir::CoreTrait::Drop) {
            continue;
        }
        if let hir::TypeKind::Path {
            res: Res::Item(target),
            ..
        } = &hir.ty(*self_ty).kind
        {
            drop_items.insert(*target);
        }
    }
    drop_items
}
/// Whether a written field type is `Copy`-eligible, treating a bare type parameter as `Copy`
/// (per-instance genericity is enforced at the query by requiring the actual argument `Copy`).
/// Conservative: any form not provably `Copy` returns `false`, so a value stays `Move`.
pub(super) fn field_ty_copy_eligible(hir: &Hir, ty: TypeId, eligible: &HashSet<ItemId>) -> bool {
    match &hir.ty(ty).kind {
        hir::TypeKind::Primitive(p) => is_copy_primitive(*p),
        hir::TypeKind::Ref { mutable, .. } => !*mutable,
        hir::TypeKind::Array { elem, .. } => field_ty_copy_eligible(hir, *elem, eligible),
        hir::TypeKind::Slice(_) => false,
        hir::TypeKind::Tuple(elems) => elems
            .iter()
            .all(|e| field_ty_copy_eligible(hir, *e, eligible)),
        hir::TypeKind::Fn { .. } | hir::TypeKind::Never => true,
        hir::TypeKind::Error => false,
        hir::TypeKind::Path { res, args, .. } => {
            let args_copy = |eligible: &HashSet<ItemId>| {
                args.as_ref().map(|a| &a.args).is_none_or(|list| {
                    list.iter().all(|arg| match arg {
                        hir::GenericArg::Type(t) => field_ty_copy_eligible(hir, *t, eligible),
                        // Non-type args (const, shape) carry no ownership.
                        _ => true,
                    })
                })
            };
            match res {
                // A bare type parameter is assumed `Copy`; the actual argument's copy-ness is
                // checked at instantiation (`is_copy_with_impls`'s `args.all(is_copy)`).
                Res::TypeParam => true,
                Res::Primitive(p) => is_copy_primitive(*p),
                Res::Item(id) => eligible.contains(id) && args_copy(eligible),
                // Option/Result are `Copy` when their arguments are; every other core nominal
                // (`Box`, `Vec`, `String`, maps, sets, iterators, ranges) is an owned resource.
                Res::CoreType(CoreType::Option | CoreType::Result) => args_copy(eligible),
                _ => false,
            }
        }
    }
}
pub(super) fn is_copy_primitive(primitive: Primitive) -> bool {
    !matches!(primitive, Primitive::String | Primitive::Str)
}

impl TypeChecker<'_> {
    // ---------------------------------------------------------------------------------------
    // AS7 Packet 9a — trait and impl selection, Core trait contracts, coherence.
    //
    // `resolve_method` is deliberately NOT here. It calls `check_expr` five times — once for the
    // receiver and once per argument — so it is a method INVOCATION path that evaluates
    // expressions, and the owner's constraint on this module is explicit: any such path stays in
    // `body`, or `traits <-> body` becomes the next cycle. Every function below was measured and
    // calls `check_expr` zero times.
    // ---------------------------------------------------------------------------------------

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

    pub(super) fn signature_type_key(
        &self,
        id: TypeId,
        self_key: &str,
        associated: &HashMap<String, TypeId>,
        generics: &HashMap<String, usize>,
    ) -> String {
        match &self.hir.ty(id).kind {
            hir::TypeKind::Primitive(primitive) => format!("p:{primitive:?}"),
            hir::TypeKind::Path { res, args, .. } => {
                if matches!(res, Res::SelfType) {
                    return self_key.to_string();
                }
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

    /// **The single Iterator selection.** One scan answers every question a `for` loop asks:
    /// which impl, which `next` body, what the substitution is, and what `Item` becomes.
    ///
    /// AS3 Boundary 4 hardening. The first attempt added a *second* selector beside
    /// `user_iterator_item_type`, which reintroduced two defects the programme exists to remove:
    ///
    /// * it identified the trait by **spelling** (`item_text(..) == "Iterator"`) while this one
    ///   uses resolved identity — DEV-BOUND-TRAIT-IDENTITY's exact class;
    /// * it discarded `match_impl_type`'s substitution and published an EMPTY generic environment,
    ///   so `impl<T> Iterator for Repeat<T>` lost its `T` binding while the element-type
    ///   calculation kept it.
    ///
    /// Both were invisible to behavioural tests, which is why there is one selector now rather than
    /// two agreeing ones.
    pub(super) fn resolve_user_iterator(&mut self, iter_ty: &Ty) -> Option<UserIteratorSelection> {
        for (idx, item) in self.hir.items.iter().enumerate() {
            let impl_item = ItemId(idx as u32);
            let hir::ItemKind::Impl {
                trait_: Some(trait_ref),
                self_ty,
                items,
                generics,
            } = &item.kind
            else {
                continue;
            };
            // Resolved identity, never the spelling.
            if !matches!(trait_ref.res, Res::CoreTrait(hir::CoreTrait::Iterator)) {
                continue;
            }
            let Some(substitutions) = self.match_impl_type(
                &self.impl_self_ty_with_args(impl_item, *self_ty),
                iter_ty,
                generics,
            ) else {
                continue;
            };
            let mut associated_item = None;
            let mut next_member = None;
            for (member, impl_item_node) in items.iter().enumerate() {
                match impl_item_node {
                    hir::ImplItem::AssocType { name, ty }
                        if self.item_text(impl_item, *name) == "Item" =>
                    {
                        associated_item = Some(*ty);
                    }
                    hir::ImplItem::Fn { def, .. }
                        if self.item_text(impl_item, def.sig.name) == "next" =>
                    {
                        next_member = Some((member as u32, def.body));
                    }
                    _ => {}
                }
            }
            let associated_item = associated_item?;
            let (member, body) = next_member?;
            // The impl's own generic parameters, in declaration order, bound to what
            // `match_impl_type` resolved — so `impl<T> Iterator for Repeat<T>` publishes `T`.
            let impl_names: Vec<String> = generics
                .iter()
                .map(|param| self.item_text(impl_item, param.name).to_string())
                .collect();
            let bindings = Self::env_bindings(&None, &impl_names, &[], true, &substitutions);
            return Some(UserIteratorSelection {
                impl_item,
                member,
                body,
                associated_item,
                substitutions,
                bindings,
            });
        }
        None
    }

    pub(super) fn core_method_signature(
        &mut self,
        receiver: &Ty,
        name: &str,
        span: Span,
    ) -> Option<(Vec<Ty>, Ty, bool)> {
        let unit = Ty::Primitive(Primitive::Unit);
        let bool_ty = Ty::Primitive(Primitive::Bool);
        let u64_ty = Ty::Primitive(Primitive::UInt64);
        let str_ref = Ty::Ref {
            mutable: false,
            inner: Box::new(Ty::Primitive(Primitive::Str)),
        };
        // WP-C1.3 (2026-07-17): `.clone()` had no method-signature entry for ANY compiler-
        // builtin type -- `Clone` as a *bound* (satisfies_bound) already recognized String/Vec/
        // Option/Result/etc., but calling `.clone()` on a value of one of these types
        // unconditionally failed with "method call on non-struct/enum type" (confirmed
        // empirically -- struct types with a hand-written `impl Clone for T` worked fine, since
        // those go through ordinary impl-block method resolution; every compiler-builtin type
        // did not). Scoped to genuinely value-like core types; iterator/cursor CoreTypes
        // (CharsIter/SplitIter/VecIter/KeysIter/ValuesIter/Iter/MapIter/FilterIter) and `Random`
        // are deliberately excluded -- cloning cursor/stateful-stream semantics is not requested
        // or normatively specified, and adding it would be new semantics, not a bug fix (Charter
        // rule 4). See COMPILER-STATE.md DEV-013.
        if name == "clone" {
            let clonable = matches!(receiver, Ty::Primitive(Primitive::String | Primitive::Str))
                || matches!(
                    receiver,
                    Ty::Core(
                        CoreType::Vec
                            | CoreType::Box
                            | CoreType::Option
                            | CoreType::Result
                            | CoreType::Range
                            | CoreType::RangeInclusive
                            | CoreType::HashMap
                            | CoreType::HashSet
                            | CoreType::IOError,
                        _
                    )
                );
            if clonable {
                return Some((Vec::new(), receiver.clone(), false));
            }
        }
        if name == "fmt" && standard_display_type(receiver) {
            return Some((Vec::new(), Ty::Primitive(Primitive::String), false));
        }
        if name == "hash" && standard_hash_type(receiver) {
            return Some((Vec::new(), u64_ty, false));
        }
        // WP-C4.7-6.2: `Ord::cmp` on a PRIMITIVE receiver. 06-Standard-Library specifies
        // `impl Ord for Int32 { fn cmp(&self, other: &Int32) -> Ordering }` "and similar for
        // other types", and `Ordering` is `core-min` prelude, but calling `3.cmp(&5)` failed
        // E0304 "method call on non-struct/enum type" — primitives had no `cmp` entry at all,
        // so the ONLY way to obtain an `Ordering` was a user-defined `Ord` impl.
        //
        // Scope: types with a total order. FLOATS ARE EXCLUDED deliberately — CD-015 (WP-C2.9)
        // froze that primitive floats do not implement `Eq`/`Ord`/`Hash`, so `1.0.cmp(&2.0)`
        // must stay rejected. `Unit` has no ordering to report.
        if name == "cmp" && ordered_primitive(receiver) {
            let self_ref = Ty::Ref {
                mutable: false,
                inner: Box::new(strip_ref(receiver).clone()),
            };
            return Some((
                vec![self_ref],
                Ty::Core(CoreType::Ordering, Vec::new()),
                false,
            ));
        }
        if matches!(receiver, Ty::Core(CoreType::File, args) if args.is_empty()) {
            let io_error = Ty::Core(CoreType::IOError, Vec::new());
            return match name {
                "read_to_string" => Some((
                    Vec::new(),
                    Ty::Core(
                        CoreType::Result,
                        vec![Ty::Primitive(Primitive::String), io_error],
                    ),
                    true,
                )),
                "write" => Some((
                    vec![Ty::Ref {
                        mutable: false,
                        inner: Box::new(Ty::Slice(Box::new(Ty::Primitive(Primitive::UInt8)))),
                    }],
                    Ty::Core(CoreType::Result, vec![u64_ty, io_error]),
                    true,
                )),
                "write_str" => Some((
                    vec![str_ref.clone()],
                    Ty::Core(CoreType::Result, vec![u64_ty, io_error]),
                    true,
                )),
                "close" => Some((
                    Vec::new(),
                    Ty::Core(CoreType::Result, vec![unit.clone(), io_error]),
                    false,
                )),
                _ => None,
            };
        }
        if self.is_iterator_type(receiver) {
            let item_ty = self.iterator_item_type(receiver);
            // WP-C7.9 Packet E: the `Iterator` COMBINATOR surface is refused by the front end.
            //
            // The audit that this list came from is the whole block below: every one of these
            // type-checked and ran in the reference interpreter, and NONE of them has a MIR
            // lowering — `map` and `filter` have no MIR representation for their adapter types,
            // and the rest are method calls on a non-nominal receiver that lowering does not
            // perform. So each was an accepted program that no compiler could build, which is the
            // split this packet closes. `next` is unaffected and is what `for` loops use, so
            // ordinary iteration over a borrow keeps working.
            //
            // Refusal rather than implementation is a scope decision (D3), not a judgement that
            // these should not exist: implementing them needs MIR adapter types and is its own
            // work package.
            if matches!(
                name,
                "count" | "collect" | "map" | "filter" | "fold" | "reduce" | "any" | "all" | "find"
            ) {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "iterator method '{name}' is not supported by this compiler; use a \
                             'for' loop over the iterator instead"
                        ),
                        span,
                    )
                    .with_code("E0105"),
                );
            }
            match name {
                "count" => return Some((Vec::new(), u64_ty, true)),
                "collect" => {
                    let c_ty = self.new_type_var();
                    return Some((Vec::new(), c_ty, true));
                }
                "map" => {
                    let u_ty = self.new_type_var();
                    let map_fn = Ty::Fn {
                        params: vec![item_ty.clone()],
                        ret: Box::new(u_ty.clone()),
                    };
                    return Some((
                        vec![map_fn],
                        Ty::Core(CoreType::MapIter, vec![receiver.clone(), u_ty]),
                        true,
                    ));
                }
                "filter" => {
                    let pred_fn = Ty::Fn {
                        params: vec![Ty::Ref {
                            mutable: false,
                            inner: Box::new(item_ty.clone()),
                        }],
                        ret: Box::new(bool_ty.clone()),
                    };
                    return Some((
                        vec![pred_fn],
                        Ty::Core(CoreType::FilterIter, vec![receiver.clone()]),
                        true,
                    ));
                }
                "fold" => {
                    let b_ty = self.new_type_var();
                    let fold_fn = Ty::Fn {
                        params: vec![b_ty.clone(), item_ty.clone()],
                        ret: Box::new(b_ty.clone()),
                    };
                    return Some((vec![b_ty.clone(), fold_fn], b_ty, true));
                }
                "reduce" => {
                    let red_fn = Ty::Fn {
                        params: vec![item_ty.clone(), item_ty.clone()],
                        ret: Box::new(item_ty.clone()),
                    };
                    return Some((
                        vec![red_fn],
                        Ty::Core(CoreType::Option, vec![item_ty.clone()]),
                        true,
                    ));
                }
                "any" => {
                    let pred_fn = Ty::Fn {
                        params: vec![item_ty.clone()],
                        ret: Box::new(bool_ty.clone()),
                    };
                    return Some((vec![pred_fn], bool_ty, true));
                }
                "all" => {
                    let pred_fn = Ty::Fn {
                        params: vec![item_ty.clone()],
                        ret: Box::new(bool_ty.clone()),
                    };
                    return Some((vec![pred_fn], bool_ty, true));
                }
                "find" => {
                    let pred_fn = Ty::Fn {
                        params: vec![Ty::Ref {
                            mutable: false,
                            inner: Box::new(item_ty.clone()),
                        }],
                        ret: Box::new(bool_ty.clone()),
                    };
                    return Some((
                        vec![pred_fn],
                        Ty::Core(CoreType::Option, vec![item_ty.clone()]),
                        true,
                    ));
                }
                _ => {}
            }
        }
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
                "chars" => Some((Vec::new(), Ty::Core(CoreType::CharsIter, Vec::new()), false)),
                "bytes" => Some((
                    Vec::new(),
                    Ty::Ref {
                        mutable: false,
                        inner: Box::new(Ty::Slice(Box::new(Ty::Primitive(Primitive::UInt8)))),
                    },
                    false,
                )),
                "into_bytes" => Some((
                    Vec::new(),
                    Ty::Core(CoreType::Vec, vec![Ty::Primitive(Primitive::UInt8)]),
                    false,
                )),
                "split" => Some((
                    vec![str_ref.clone()],
                    Ty::Core(CoreType::SplitIter, Vec::new()),
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
                    "get_mut" => Some((
                        vec![u64_ty],
                        Ty::Core(
                            CoreType::Option,
                            vec![Ty::Ref {
                                mutable: true,
                                inner: Box::new(elem.clone()),
                            }],
                        ),
                        true,
                    )),
                    "extend" => {
                        let iter_ty = self.new_type_var();
                        Some((vec![iter_ty], unit, true))
                    }
                    "iter" => Some((
                        Vec::new(),
                        Ty::Core(CoreType::VecIter, vec![elem.clone()]),
                        false,
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
                    // DEV-063: the fn-value-consuming combinators from 06-Standard-Library.md
                    // §Option. `U` is a fresh inference variable determined by unifying the
                    // declared `fn(T) -> U` parameter against the argument -- the same pattern
                    // the iterator `.map`/`.filter` signatures below already use.
                    "map" => {
                        let u_ty = self.new_type_var();
                        let map_fn = Ty::Fn {
                            params: vec![value.clone()],
                            ret: Box::new(u_ty.clone()),
                        };
                        Some((vec![map_fn], Ty::Core(CoreType::Option, vec![u_ty]), false))
                    }
                    "and_then" => {
                        let u_ty = self.new_type_var();
                        let then_fn = Ty::Fn {
                            params: vec![value.clone()],
                            ret: Box::new(Ty::Core(CoreType::Option, vec![u_ty.clone()])),
                        };
                        Some((vec![then_fn], Ty::Core(CoreType::Option, vec![u_ty]), false))
                    }
                    _ => None,
                }
            }
            Ty::Core(CoreType::Result, args) => {
                let value = args.first().cloned().unwrap_or(Ty::Error);
                let error = args.get(1).cloned().unwrap_or(Ty::Error);
                match name {
                    "is_ok" | "is_err" => Some((Vec::new(), bool_ty, false)),
                    "unwrap" => Some((Vec::new(), value.clone(), false)),
                    "unwrap_or" => Some((vec![value.clone()], value, false)),
                    // DEV-063: 06-Standard-Library.md §Result combinators.
                    "map" => {
                        let u_ty = self.new_type_var();
                        let map_fn = Ty::Fn {
                            params: vec![value.clone()],
                            ret: Box::new(u_ty.clone()),
                        };
                        Some((
                            vec![map_fn],
                            Ty::Core(CoreType::Result, vec![u_ty, error]),
                            false,
                        ))
                    }
                    "map_err" => {
                        let f_ty = self.new_type_var();
                        let map_fn = Ty::Fn {
                            params: vec![error.clone()],
                            ret: Box::new(f_ty.clone()),
                        };
                        Some((
                            vec![map_fn],
                            Ty::Core(CoreType::Result, vec![value, f_ty]),
                            false,
                        ))
                    }
                    "and_then" => {
                        let u_ty = self.new_type_var();
                        let then_fn = Ty::Fn {
                            params: vec![value.clone()],
                            ret: Box::new(Ty::Core(
                                CoreType::Result,
                                vec![u_ty.clone(), error.clone()],
                            )),
                        };
                        Some((
                            vec![then_fn],
                            Ty::Core(CoreType::Result, vec![u_ty, error]),
                            false,
                        ))
                    }
                    _ => None,
                }
            }
            Ty::Core(CoreType::Box, args) if name == "into_inner" => Some((
                Vec::new(),
                args.first().cloned().unwrap_or(Ty::Error),
                false,
            )),
            Ty::Core(CoreType::CharsIter, _) if name == "next" => Some((
                Vec::new(),
                Ty::Core(CoreType::Option, vec![Ty::Primitive(Primitive::Char)]),
                true,
            )),
            Ty::Core(CoreType::SplitIter, _) if name == "next" => Some((
                Vec::new(),
                Ty::Core(CoreType::Option, vec![str_ref.clone()]),
                true,
            )),
            Ty::Core(CoreType::VecIter, args) if name == "next" => {
                let elem = args.first().cloned().unwrap_or(Ty::Error);
                Some((
                    Vec::new(),
                    Ty::Core(
                        CoreType::Option,
                        vec![Ty::Ref {
                            mutable: false,
                            inner: Box::new(elem),
                        }],
                    ),
                    true,
                ))
            }
            Ty::Core(CoreType::HashMap, args) => {
                let k = args.first().cloned().unwrap_or(Ty::Error);
                let v = args.get(1).cloned().unwrap_or(Ty::Error);
                let k_ref = Ty::Ref {
                    mutable: false,
                    inner: Box::new(k.clone()),
                };
                match name {
                    "insert" => Some((
                        vec![k, v.clone()],
                        Ty::Core(CoreType::Option, vec![v]),
                        true,
                    )),
                    "get" => Some((
                        vec![k_ref.clone()],
                        Ty::Core(
                            CoreType::Option,
                            vec![Ty::Ref {
                                mutable: false,
                                inner: Box::new(v.clone()),
                            }],
                        ),
                        false,
                    )),
                    "get_mut" => Some((
                        vec![k_ref.clone()],
                        Ty::Core(
                            CoreType::Option,
                            vec![Ty::Ref {
                                mutable: true,
                                inner: Box::new(v.clone()),
                            }],
                        ),
                        true,
                    )),
                    "remove" => Some((
                        vec![k_ref.clone()],
                        Ty::Core(CoreType::Option, vec![v]),
                        true,
                    )),
                    "contains_key" => Some((vec![k_ref], bool_ty, false)),
                    "len" => Some((Vec::new(), u64_ty, false)),
                    "is_empty" => Some((Vec::new(), bool_ty, false)),
                    "clear" => Some((Vec::new(), unit, true)),
                    "keys" => Some((Vec::new(), Ty::Core(CoreType::KeysIter, vec![k]), false)),
                    "values" => Some((Vec::new(), Ty::Core(CoreType::ValuesIter, vec![v]), false)),
                    "iter" => Some((Vec::new(), Ty::Core(CoreType::Iter, vec![k, v]), false)),
                    "extend" => {
                        let iter_ty = self.new_type_var();
                        Some((vec![iter_ty], unit, true))
                    }
                    _ => None,
                }
            }
            Ty::Core(CoreType::HashSet, args) => {
                let t = args.first().cloned().unwrap_or(Ty::Error);
                let t_ref = Ty::Ref {
                    mutable: false,
                    inner: Box::new(t.clone()),
                };
                match name {
                    "insert" => Some((vec![t.clone()], bool_ty, true)),
                    "remove" => Some((vec![t_ref.clone()], bool_ty, true)),
                    "contains" => Some((vec![t_ref], bool_ty, false)),
                    "len" => Some((Vec::new(), u64_ty, false)),
                    "is_empty" => Some((Vec::new(), bool_ty, false)),
                    "clear" => Some((Vec::new(), unit, true)),
                    "iter" => Some((Vec::new(), Ty::Core(CoreType::Iter, vec![t]), false)),
                    "extend" => {
                        let iter_ty = self.new_type_var();
                        Some((vec![iter_ty], unit, true))
                    }
                    _ => None,
                }
            }
            Ty::Core(CoreType::KeysIter, args) if name == "next" => {
                let k = args.first().cloned().unwrap_or(Ty::Error);
                Some((
                    Vec::new(),
                    Ty::Core(
                        CoreType::Option,
                        vec![Ty::Ref {
                            mutable: false,
                            inner: Box::new(k),
                        }],
                    ),
                    true,
                ))
            }
            Ty::Core(CoreType::ValuesIter, args) if name == "next" => {
                let v = args.first().cloned().unwrap_or(Ty::Error);
                Some((
                    Vec::new(),
                    Ty::Core(
                        CoreType::Option,
                        vec![Ty::Ref {
                            mutable: false,
                            inner: Box::new(v),
                        }],
                    ),
                    true,
                ))
            }
            Ty::Core(CoreType::Iter, args) if name == "next" => {
                if args.len() == 2 {
                    let k = args.first().cloned().unwrap_or(Ty::Error);
                    let v = args.get(1).cloned().unwrap_or(Ty::Error);
                    let tuple_ty = Ty::Tuple(vec![
                        Ty::Ref {
                            mutable: false,
                            inner: Box::new(k),
                        },
                        Ty::Ref {
                            mutable: false,
                            inner: Box::new(v),
                        },
                    ]);
                    Some((Vec::new(), Ty::Core(CoreType::Option, vec![tuple_ty]), true))
                } else {
                    let t = args.first().cloned().unwrap_or(Ty::Error);
                    Some((
                        Vec::new(),
                        Ty::Core(
                            CoreType::Option,
                            vec![Ty::Ref {
                                mutable: false,
                                inner: Box::new(t),
                            }],
                        ),
                        true,
                    ))
                }
            }
            Ty::Core(CoreType::MapIter, args) if name == "next" => {
                let u = args.get(1).cloned().unwrap_or(Ty::Error);
                Some((Vec::new(), Ty::Core(CoreType::Option, vec![u]), true))
            }
            Ty::Core(CoreType::FilterIter, args) if name == "next" => {
                let inner = args.first().cloned().unwrap_or(Ty::Error);
                let item = self.iterator_item_type(&inner);
                Some((Vec::new(), Ty::Core(CoreType::Option, vec![item]), true))
            }
            Ty::Slice(_) => match name {
                "len" => Some((Vec::new(), u64_ty, false)),
                "is_empty" => Some((Vec::new(), bool_ty, false)),
                _ => None,
            },
            Ty::Ref { inner, .. } => match &**inner {
                Ty::Slice(_) => match name {
                    "len" => Some((Vec::new(), u64_ty, false)),
                    "is_empty" => Some((Vec::new(), bool_ty, false)),
                    _ => None,
                },
                _ => None,
            },
            // Phase 4E: `Random` (simple LCG, `06-Standard-Library.md`
            // "Random numbers" — `&mut self`, matching the spec exactly).
            Ty::Core(CoreType::Random, _) => match name {
                "next_int" => Some((Vec::new(), u64_ty, true)),
                "next_float" => Some((Vec::new(), Ty::Primitive(Primitive::Float64), true)),
                "range" => Some((
                    vec![
                        Ty::Primitive(Primitive::Int32),
                        Ty::Primitive(Primitive::Int32),
                    ],
                    Ty::Primitive(Primitive::Int32),
                    true,
                )),
                _ => None,
            },
            _ => None,
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
}

impl TypeChecker<'_> {
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
    /// The trait's declaration of `method`, as a source-shaped line for a diagnostic note.
    pub(super) fn core_method_source(&self, trait_name: &str, method: &CoreTraitMethod) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(receiver) = method.receiver {
            parts.push(receiver_source(Some(receiver)).to_string());
        }
        for param in method.params {
            parts.push(contract_ty_source(*param));
        }
        let ret = match method.ret {
            None => String::new(),
            Some(ty) => format!(" -> {}", contract_ty_source(ty)),
        };
        format!("'{trait_name}::{}({}){ret}'", method.name, parts.join(", "))
    }
    /// DEV-073 (WP-C4.7-5): convert an impl's WRITTEN self type while PRESERVING its generic
    /// arguments, with type parameters kept as `Ty::Param` so `match_impl_type` can unify them
    /// against a concrete instantiation.
    ///
    /// This exists because `type_from_hir_without_diagnostics` deliberately drops generic
    /// arguments (`Ty::Struct(item, Vec::new())`). That was invisible while the only consumers
    /// compared NON-generic nominals — `struct P` converts to `Struct(id, [])` either way — and
    /// was the actual reason generic impls failed bound checks: the impl's `W<T>` converted to
    /// `W<>`, whose argument count never matched `W<Int32>`'s.
    ///
    /// `item` is the impl whose self type this is; its spans (parameter names) belong to that
    /// impl's own file (DEV-069).
    pub(super) fn impl_self_ty_with_args(&self, item: ItemId, id: TypeId) -> Ty {
        match &self.hir.ty(id).kind {
            hir::TypeKind::Primitive(primitive) => Ty::Primitive(*primitive),
            hir::TypeKind::Path { res, args, .. } => {
                let converted: Vec<Ty> = args.as_ref().map_or_else(Vec::new, |list| {
                    list.args
                        .iter()
                        .map(|arg| match arg {
                            hir::GenericArg::Type(ty) => self.impl_self_ty_with_args(item, *ty),
                            _ => Ty::Error,
                        })
                        .collect()
                });
                match res {
                    Res::Item(nominal) => match &self.hir.item(*nominal).kind {
                        hir::ItemKind::Struct { .. } => Ty::Struct(*nominal, converted),
                        hir::ItemKind::Enum { .. } => Ty::Enum(*nominal, converted),
                        _ => Ty::Error,
                    },
                    Res::TypeParam => {
                        Ty::Param(self.item_text(item, self.hir.ty(id).span).to_string())
                    }
                    _ => Ty::Error,
                }
            }
            hir::TypeKind::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.impl_self_ty_with_args(item, *inner)),
            },
            _ => Ty::Error,
        }
    }
    pub(super) fn is_iterator_type(&self, receiver: &Ty) -> bool {
        if let Ty::Core(core, _) = receiver {
            matches!(
                core,
                CoreType::CharsIter
                    | CoreType::SplitIter
                    | CoreType::VecIter
                    | CoreType::KeysIter
                    | CoreType::ValuesIter
                    | CoreType::Iter
                    | CoreType::MapIter
                    | CoreType::FilterIter
            )
        } else {
            false
        }
    }
    pub(super) fn iterator_item_type(&self, iter_ty: &Ty) -> Ty {
        match iter_ty {
            Ty::Core(CoreType::CharsIter, _) => Ty::Primitive(Primitive::Char),
            Ty::Core(CoreType::SplitIter, _) => Ty::Ref {
                mutable: false,
                inner: Box::new(Ty::Primitive(Primitive::Str)),
            },
            Ty::Core(CoreType::VecIter, args) => Ty::Ref {
                mutable: false,
                inner: Box::new(args.first().cloned().unwrap_or(Ty::Error)),
            },
            Ty::Core(CoreType::KeysIter, args) => Ty::Ref {
                mutable: false,
                inner: Box::new(args.first().cloned().unwrap_or(Ty::Error)),
            },
            Ty::Core(CoreType::ValuesIter, args) => Ty::Ref {
                mutable: false,
                inner: Box::new(args.first().cloned().unwrap_or(Ty::Error)),
            },
            Ty::Core(CoreType::Iter, args) => {
                if args.len() == 2 {
                    let k = args.first().cloned().unwrap_or(Ty::Error);
                    let v = args.get(1).cloned().unwrap_or(Ty::Error);
                    Ty::Tuple(vec![
                        Ty::Ref {
                            mutable: false,
                            inner: Box::new(k),
                        },
                        Ty::Ref {
                            mutable: false,
                            inner: Box::new(v),
                        },
                    ])
                } else {
                    let t = args.first().cloned().unwrap_or(Ty::Error);
                    Ty::Ref {
                        mutable: false,
                        inner: Box::new(t),
                    }
                }
            }
            Ty::Core(CoreType::MapIter, args) => args.get(1).cloned().unwrap_or(Ty::Error),
            Ty::Core(CoreType::FilterIter, args) => {
                let inner = args.first().cloned().unwrap_or(Ty::Error);
                self.iterator_item_type(&inner)
            }
            _ => Ty::Error,
        }
    }
    pub(super) fn match_impl_type(
        &self,
        implementation: &Ty,
        receiver: &Ty,
        generics: &[hir::GenericParam],
    ) -> Option<HashMap<String, Ty>> {
        let mut map = HashMap::new();
        let matches = self.unify_impl_ty(implementation, receiver, &mut map);
        if matches {
            for generic in generics {
                // CD-358: the impl's own parameter names, read against the impl's file.
                let name = self.decl_text(generic.name).to_string();
                map.entry(name.clone())
                    .or_insert_with(|| Ty::Param(name.clone()));
            }
            Some(map)
        } else {
            None
        }
    }
    pub(super) fn trait_method_signature_matches(
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
        // WP-C6.2b-F6: the self type's key in the SAME format `signature_type_key`
        // produces for a path, so `Self` and the written concrete self type (`G`, `W<Int32>`)
        // compare equal — an impl may spell either.
        let self_key = self.ty_signature_key(self_ty);
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
    /// WP-C6.2b-F6: a `Ty`'s key in the exact format `signature_type_key` produces for the same
    /// type written as a path, so the impl's self type and a `Self` mention share one key.
    pub(super) fn ty_signature_key(&self, ty: &Ty) -> String {
        let ty = self.resolve(ty);
        let keyed = |items: &[Ty]| {
            items
                .iter()
                .map(|t| self.ty_signature_key(t))
                .collect::<Vec<_>>()
                .join(",")
        };
        match &ty {
            Ty::Primitive(p) => format!("p:{p:?}"),
            Ty::Struct(id, args) | Ty::Enum(id, args) => {
                format!("item:{}<{}>", id.0, keyed(args))
            }
            Ty::Core(core, args) => format!("core:{core:?}<{}>", keyed(args)),
            Ty::Ref { mutable, inner } => format!("ref:{mutable}:{}", self.ty_signature_key(inner)),
            Ty::Tuple(elems) => format!("tuple:{}", keyed(elems)),
            Ty::Array(elem, n) => format!("array:{}:{n}", self.ty_signature_key(elem)),
            Ty::Slice(elem) => format!("slice:{}", self.ty_signature_key(elem)),
            Ty::Param(name) => format!("param:{name}"),
            other => format!("{other:?}"),
        }
    }
    pub(super) fn types_may_overlap(&self, left: &Ty, right: &Ty) -> bool {
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
    pub(super) fn check_member_visible(
        &mut self,
        is_pub: bool,
        defining_item: ItemId,
        kind: &str,
        name: &str,
        span: Span,
    ) -> bool {
        if is_pub {
            return true;
        }
        let member_module = self.hir.item_modules.get(&defining_item).copied();
        if member_module == self.current_module {
            return true;
        }
        self.diags.push(
            Diagnostic::error(format!("{kind} '{name}' is private"), span)
                .with_code("E0207")
                .with_label("private to its defining module"),
        );
        false
    }
    /// A deterministic key for a shape argument, used only to keep signature
    /// keys total. The tensor extension checker owns real shape equality.
    pub(super) fn dim_key(&self, dim: &hir::DimExpr) -> String {
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
    /// The logical name of the file `span` belongs to.
    pub(super) fn source_name(&self, span: Span) -> &str {
        self.hir
            .sources
            .get(span.source)
            .map(|file| file.name.as_str())
            .unwrap_or("<unknown>")
    }
}

/// How a contract term reads in a diagnostic — the *expected* half of "expected X, found Y".
pub(super) fn contract_ty_source(ty: ContractTy) -> String {
    match ty {
        ContractTy::SelfTy => "Self".to_string(),
        ContractTy::RefSelf => "&Self".to_string(),
        ContractTy::Bool => "Bool".to_string(),
        ContractTy::UInt64 => "UInt64".to_string(),
        ContractTy::StringTy => "String".to_string(),
        ContractTy::Ordering => "Ordering".to_string(),
        ContractTy::OptionAssoc(name) => format!("Option<Self::{name}>"),
        ContractTy::TraitArg(index) => format!("the trait's type argument #{}", index + 1),
    }
}

/// A Core trait's complete implementation contract.
pub(super) struct CoreTraitContract {
    /// Every method the trait declares. All are required — no Core trait has a defaulted method.
    pub(super) methods: &'static [CoreTraitMethod],
    /// Associated types the implementation must declare.
    pub(super) assoc_types: &'static [&'static str],
}

/// A Core trait method's required shape. Core v1 declares no method-level generics on any of
/// these, so an impl that introduces one is malformed by construction.
pub(super) struct CoreTraitMethod {
    pub(super) name: &'static str,
    pub(super) receiver: Option<hir::Receiver>,
    pub(super) params: &'static [ContractTy],
    /// `None` is a `Unit` return (`06-Standard-Library.md`: `fn drop(&mut self);`).
    pub(super) ret: Option<ContractTy>,
}

/// One type position in a Core trait's declared signature.
///
/// These are the *contract's* terms, not the implementation's. Each is rendered into the same key
/// format `signature_type_key` produces for a written type, so one comparison serves both a
/// user-declared trait (whose declaration is an HIR item) and a Core trait (which has none).
#[derive(Clone, Copy)]
pub(super) enum ContractTy {
    /// `Self` — the implementing type.
    SelfTy,
    /// `&Self`.
    RefSelf,
    Bool,
    UInt64,
    /// The prelude `String`.
    StringTy,
    /// The prelude `Ordering`.
    Ordering,
    /// `Option<Self::Name>` — the associated type the impl declared.
    OptionAssoc(&'static str),
    /// The trait's own generic argument at this position, as written in `impl Trait<..> for T`.
    TraitArg(usize),
}

/// The contract for `core_trait`, or `None` when this trait's implementation shape is not modelled.
///
/// **`None` is a scope statement, not an oversight.** `Index`/`IndexMut`/`TryFrom`/`Error`/
/// `FromIterator` declare signatures over associated types *and* method-level generics
/// (`fn from_iter<I: Iterator<Item = T>>(iter: I) -> Self`), and no user implementation of them is
/// supported anywhere in the compiler today. Writing a contract for them here would assert a
/// support level that does not exist, and would be checked against nothing. `Num` is excluded for
/// the opposite reason: implementing it at all is already rejected outright, before this check.
///
/// Every trait a user can implement in practice — the seven fixed-signature traits, `Iterator`,
/// `From` and `Into` — is modelled.
pub(super) fn core_trait_contract(core_trait: hir::CoreTrait) -> Option<CoreTraitContract> {
    use hir::CoreTrait as CT;
    use hir::Receiver::{Ref, RefMut, Value};
    let contract = match core_trait {
        // Markers: no items at all, so any item in the block is an extra one.
        CT::Copy => CoreTraitContract {
            methods: &[],
            assoc_types: &[],
        },
        // `fn drop(&mut self);`
        CT::Drop => CoreTraitContract {
            methods: &[CoreTraitMethod {
                name: "drop",
                receiver: Some(RefMut),
                params: &[],
                ret: None,
            }],
            assoc_types: &[],
        },
        // `fn eq(&self, other: &Self) -> Bool;`
        CT::Eq => CoreTraitContract {
            methods: &[CoreTraitMethod {
                name: "eq",
                receiver: Some(Ref),
                params: &[ContractTy::RefSelf],
                ret: Some(ContractTy::Bool),
            }],
            assoc_types: &[],
        },
        // `fn cmp(&self, other: &Self) -> Ordering;`
        CT::Ord => CoreTraitContract {
            methods: &[CoreTraitMethod {
                name: "cmp",
                receiver: Some(Ref),
                params: &[ContractTy::RefSelf],
                ret: Some(ContractTy::Ordering),
            }],
            assoc_types: &[],
        },
        // `fn clone(&self) -> Self;`
        CT::Clone => CoreTraitContract {
            methods: &[CoreTraitMethod {
                name: "clone",
                receiver: Some(Ref),
                params: &[],
                ret: Some(ContractTy::SelfTy),
            }],
            assoc_types: &[],
        },
        // `fn hash(&self) -> UInt64;`
        CT::Hash => CoreTraitContract {
            methods: &[CoreTraitMethod {
                name: "hash",
                receiver: Some(Ref),
                params: &[],
                ret: Some(ContractTy::UInt64),
            }],
            assoc_types: &[],
        },
        // `fn default() -> Self;` — no receiver.
        CT::Default => CoreTraitContract {
            methods: &[CoreTraitMethod {
                name: "default",
                receiver: None,
                params: &[],
                ret: Some(ContractTy::SelfTy),
            }],
            assoc_types: &[],
        },
        // `fn fmt(&self) -> String;`
        CT::Display => CoreTraitContract {
            methods: &[CoreTraitMethod {
                name: "fmt",
                receiver: Some(Ref),
                params: &[],
                ret: Some(ContractTy::StringTy),
            }],
            assoc_types: &[],
        },
        // `type Item; fn next(&mut self) -> Option<Self::Item>;`
        CT::Iterator => CoreTraitContract {
            methods: &[CoreTraitMethod {
                name: "next",
                receiver: Some(RefMut),
                params: &[],
                ret: Some(ContractTy::OptionAssoc("Item")),
            }],
            assoc_types: &["Item"],
        },
        // `fn from(value: T) -> Self;` — `T` is the trait's own argument.
        CT::From => CoreTraitContract {
            methods: &[CoreTraitMethod {
                name: "from",
                receiver: None,
                params: &[ContractTy::TraitArg(0)],
                ret: Some(ContractTy::SelfTy),
            }],
            assoc_types: &[],
        },
        // `fn into(self) -> T;`
        CT::Into => CoreTraitContract {
            methods: &[CoreTraitMethod {
                name: "into",
                receiver: Some(Value),
                params: &[],
                ret: Some(ContractTy::TraitArg(0)),
            }],
            assoc_types: &[],
        },
        CT::Num | CT::Error | CT::TryFrom | CT::Index | CT::IndexMut | CT::FromIterator => {
            return None
        }
    };
    Some(contract)
}

/// DEV-052: reverse of `resolve.rs`'s private `resolve_core_trait` -- the source spelling of a
/// `CoreTrait`, used to match an `impl <name> for T` block by its trait-ref source text, the
/// same way `ty_satisfies_operator_bound` already does for these compiler-known traits.
pub(super) fn core_trait_source_name(core_trait: hir::CoreTrait) -> &'static str {
    match core_trait {
        hir::CoreTrait::Copy => "Copy",
        hir::CoreTrait::Drop => "Drop",
        hir::CoreTrait::Eq => "Eq",
        hir::CoreTrait::Ord => "Ord",
        hir::CoreTrait::Num => "Num",
        hir::CoreTrait::Clone => "Clone",
        hir::CoreTrait::Hash => "Hash",
        hir::CoreTrait::Default => "Default",
        hir::CoreTrait::Display => "Display",
        hir::CoreTrait::Error => "Error",
        hir::CoreTrait::From => "From",
        hir::CoreTrait::Into => "Into",
        hir::CoreTrait::TryFrom => "TryFrom",
        hir::CoreTrait::Index => "Index",
        hir::CoreTrait::IndexMut => "IndexMut",
        hir::CoreTrait::Iterator => "Iterator",
        hir::CoreTrait::FromIterator => "FromIterator",
    }
}

impl TypeChecker<'_> {
    /// AS1b-ii-d: kept as a name, not a mechanism. CD-358 introduced this because a name belonging
    /// to a DECLARATION had to be read against the declaring file while `self.text` read the file
    /// being checked; across a module boundary those differ, and getting it wrong compared garbage
    /// rather than erroring. A declaration's span names its own file, so this is `text`.
    pub(super) fn decl_text(&self, span: Span) -> &str {
        self.text(span)
    }
    /// WP-C4.7-8.5: ONE-WAY structural unification of an impl's written self type against a
    /// concrete receiver, binding the impl's parameters along the way.
    ///
    /// Recursion is what admits NON-BARE impl heads: `impl<T> Holder<Option<T>>` must match
    /// `Holder<Option<Int32>>`. The previous version only bound a parameter when it stood ALONE
    /// as a type argument and otherwise demanded `types_equal`, so `Option<T>` versus
    /// `Option<Int32>` failed and every non-bare head was invisible to method resolution
    /// (E0302 "method not found").
    ///
    /// One-way: parameters are bound only from the IMPLEMENTATION side. A `Ty::Param` on the
    /// receiver side is an ordinary type to match against, never a hole to fill — otherwise an
    /// impl for a concrete type would spuriously match a generic receiver.
    /// AS3 Boundary 4 step 3: delegates to the shared structural matcher so the checker and the
    /// bound specialiser use **one** algorithm. Two matchers that must agree is the pattern this
    /// packet removes; the only difference between the callers is how they resolve inference
    /// variables, which is why that is a parameter rather than a fork.
    pub(super) fn unify_impl_ty(
        &self,
        implementation: &Ty,
        receiver: &Ty,
        map: &mut HashMap<String, Ty>,
    ) -> bool {
        crate::bound_dispatch::unify_impl_ty_with(implementation, receiver, map, &|ty| {
            self.resolve(ty)
        })
    }
}
