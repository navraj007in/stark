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

use super::state::TypeChecker;
use super::types::{is_float_primitive, is_numeric, standard_display_type, standard_hash_type};
use super::types::{ExtensionTy, Ty};

use crate::ast::Primitive;
use crate::hir::{self, CoreType, Hir, ItemId, Res, TypeId};
use std::collections::{HashMap, HashSet};

/// Why a trait relation holds. `Impl` carries the selected impl so that the layer above can look
/// up *that impl's* associated types — the same impl this layer chose, which is what preserves
/// the existing behaviour when several traits expose the same associated name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BoundWitness {
    /// The relation does not hold.
    No,
    /// It holds, with no impl to point at (a primitive or built-in rule).
    Yes,
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
