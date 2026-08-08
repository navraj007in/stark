//! **AS7 Packet 5 — the checker's storage and its scoped ambient contexts.**
//!
//! One level above `types` in the DAG: `state` may depend on `types` and on nothing else inside
//! `typecheck`.
//!
//! What lives here is the `TypeChecker` value itself and **every operation that enters or leaves
//! an ambient scope**. Packet 2 established the rule this module now enforces by construction:
//! every scope SAVES AND RESTORES, and none clears to a default. `check_fn_def` used to clear
//! `current_fn_ret` to `None` on exit, which was correct only while item checking never nests —
//! and the file splitting this packet is part of is the change most likely to break that.
//!
//! **Why `state` owns the mutation rather than merely the storage.** The dependency checker sees
//! method calls and imports; it cannot see raw field access. If `body` later writes
//! `self.current_self_ty` directly, that is a real `body -> state` edge criterion 2 would miss
//! entirely. Routing every ambient write through a named operation here is what keeps that check
//! honest for the rest of the marathon.
//!
//! Fields are `pub(super)` — visible throughout `typecheck` and its descendants, invisible
//! outside it — because the passes still being extracted read them directly. Narrowing that is a
//! later packet's work, not this one's.

use super::types::{
    BoundsCheck, CallableDeclId, CallableInstantiation, CallableSigTy, CallableUse, CallableUseId,
    DeferredDisplayPlan, DisplayPath, FnSigTy, GenericKind, LoopContext, Ty, TypeVarId, VariantTy,
};
use crate::diag::Diagnostic;
use crate::extensions::tensor::dim::DimVar;
use crate::extensions::tensor::types::{DType, Device, UnifyCtx};
use crate::extensions::tensor::types::{DimProvenance, OriginKind};
use crate::hir::Res;
use crate::hir::{self, BlockId, ExprId, Hir, ItemId, LocalId};
use crate::options::LanguageOptions;
use crate::source::Span;
use std::collections::{BTreeMap, HashMap};

/// AS7 Packet 2 — outer values displaced by entering a `Self`-carrying scope: an impl, a trait's
/// default bodies, or a method whose receiver fixes `Self`.
///
/// Each field is `Option`al *about whether this scope establishes it at all*, so a caller that
/// only installs `Self` does not have to state an opinion about the enclosing trait id. That is
/// why this is not one flat `AmbientContext`: the four fields share a scope shape, not a lifetime.
pub(super) struct SelfScope {
    pub(super) self_ty: Option<Ty>,
    pub(super) assoc_types: Option<HashMap<String, Ty>>,
    pub(super) impl_generics: Option<Option<Vec<hir::GenericParam>>>,
    pub(super) trait_id: Option<Option<ItemId>>,
}

pub(super) struct TensorParamScopes {
    pub(super) dims: HashMap<String, DimVar>,
    pub(super) dtypes: HashMap<String, DType>,
    pub(super) devices: HashMap<String, Device>,
    pub(super) kinds: HashMap<String, GenericKind>,
}

pub struct TypeChecker<'a> {
    pub(super) hir: &'a Hir,
    pub(super) diags: Vec<Diagnostic>,
    pub(super) subst: HashMap<TypeVarId, Ty>,
    /// WP-C4.7-6.3: inference variables introduced for UNSUFFIXED integer literals, with the
    /// literal's value and span. 03-Type-System's solver defaults "an **unconstrained** integer
    /// literal" to `Int32`/`Int64` — step 5, *after* expected types have flowed inward from
    /// annotations, parameters, fields and so on (the paragraph above the numbered steps). The
    /// checker used to skip straight to the default, committing every literal to `Int32` before
    /// any expectation could apply, so `takes_u64(0)` was rejected "expected 'UInt64', found
    /// 'Int32'". These vars are integer-KINDED: they unify only with primitive integer types,
    /// and binding one range-checks the value.
    pub(super) int_literal_vars: HashMap<TypeVarId, (i128, Span)>,
    /// WP-C4.7-9 audit: deferred `print`/`println` argument types, checked for `Display` after
    /// inference settles (the argument may still be a variable while the body is being checked).
    pub(super) display_checks: Vec<(Ty, Span)>,
    /// **AS3 Boundary 4: the queue the `Display` dispatch plan is built from.**
    ///
    /// Separate from `display_checks`, which exists to emit E0500. One queue per job: a queue that
    /// both reports errors and publishes a plan is the one a fourth concern gets added to.
    ///
    /// Both `println`-family arguments and interpolation fields land here, so the plan is built by
    /// ONE walk regardless of which syntax reached `Display`.
    ///
    pub(super) display_plans: Vec<DeferredDisplayPlan>,
    /// DEV-134: deferred `?` propagation compatibility — (operand type, enclosing return type,
    /// span). Deferred for the same reason as `display_checks`: the operand's error type is
    /// routinely an inference variable while the body is being checked (`Err(make())?`), so
    /// comparing it eagerly would either reject valid code or force a premature binding.
    pub(super) try_checks: Vec<(Ty, Ty, Span)>,
    pub(super) var_count: u32,

    // Side tables
    pub(super) expr_types: HashMap<ExprId, Ty>,
    pub(super) local_types: HashMap<LocalId, Ty>,
    pub(super) local_mutability: HashMap<LocalId, bool>,
    pub(super) struct_fields: HashMap<ItemId, HashMap<String, Ty>>,
    /// AS3 Packet 5: the INSTANTIATED declared type of each field of an aggregate literal, keyed
    /// by the literal expression. Publication only; consumed by the `AggregateField` boundary.
    pub(super) aggregate_field_types: HashMap<ExprId, HashMap<String, Ty>>,
    pub(super) enum_variants: HashMap<ItemId, Vec<VariantTy>>,
    pub(super) fn_sigs: HashMap<ItemId, FnSigTy>,
    /// A3b: raw (pre-grounding) callable signatures, keyed by body.
    pub(super) callable_sigs: HashMap<BlockId, CallableSigTy>,
    /// A3c-S: raw (pre-grounding) callable environments, keyed by the call expression.
    pub(super) callable_envs: HashMap<ExprId, CallableInstantiation>,
    /// AS3: the published uses, in publication order. `CallableUseId` is the index.
    pub(super) callable_uses: Vec<CallableUse>,
    pub(super) callable_uses_by_expr: HashMap<ExprId, Vec<CallableUseId>>,
    pub(super) display_uses: BTreeMap<(ExprId, DisplayPath), CallableUseId>,
    /// AS3: body → declaration, built on first use.
    #[allow(dead_code)] // read by `decl_for_body`, which Boundary 2 consumes.
    pub(super) body_decls: Option<HashMap<BlockId, CallableDeclId>>,
    pub(super) const_types: HashMap<ItemId, Ty>,
    pub(super) alias_stack: Vec<ItemId>,
    /// WP-C4.5c / A3c-S: ordered generic-argument types for every use of a *generic* fn item, keyed
    /// by the referencing path expression. Grounded and published as
    /// `TypeTables::callable_instantiations` for MIR monomorphisation; an instantiation still
    /// containing `Ty::Infer` once inference completes is rejected with E0004
    /// (TYPE-GENERIC-001 / TYPE-FN-002 — the DEV-064 fix).
    /// WP-C5.3e: the queried type of each `size_of::<T>()` / `align_of::<T>()`, keyed by the
    /// builtin's own path expression. Kept OUT of `callable_instantiations` deliberately: that table drives
    /// MIR monomorphisation of generic fn instances, and a layout query is not one.
    pub(super) layout_queries: HashMap<ExprId, Ty>,
    /// DEV-BOUND-TRAIT-IDENTITY: for each method call resolved through a generic parameter's
    /// BOUND, the trait that bound denotes. Keyed by the call expression.
    ///
    /// The identity the checker selected a signature from must be the identity execution selects
    /// an implementation from. Without it, both engines fell back to "first impl on this nominal
    /// declaring a method with that name", so `use_left(&item)` and `use_right(&item)` — bounded
    /// on two different `Render` traits, each implemented for `Item` — both ran `left::Render`'s.
    /// Type checking was right and every engine below it was wrong in the same way.
    pub(super) bound_trait_calls: HashMap<ExprId, Res>,

    // Scopes context
    pub(super) current_self_ty: Option<Ty>,
    /// DEV-148: the item whose FILE the signature currently being converted belongs to, or `None`
    /// when the signature is local. Every name sliced out of a foreign signature — type-parameter
    /// names above all — must be read from that file, not from the file under check.
    pub(super) current_assoc_types: HashMap<String, Ty>,
    /// WP-C6.2c: resolved associated-type bindings across the whole program, keyed by
    /// `(implementing nominal, associated-type name)`. Lets a concrete projection
    /// `<H as Holder>::Item` (carried through generic instantiation as `Ty::Param("H::Item")`)
    /// be normalised to the impl's bound type. Built once in Pass 1 (`build_assoc_projections`).
    pub(super) assoc_projections: HashMap<(ItemId, String), Ty>,
    /// WP-C6.2c: deferred associated-type projections whose base is still an inference variable at
    /// the call site — `fn first<T: Holder>(t: T) -> T::Item` called on a value whose type is only
    /// determined by unifying the argument. Each entry is `(projection var, base var, assoc name,
    /// span)`; resolved after all bodies are checked, once the base var has grounded to a nominal.
    pub(super) projection_obligations: Vec<(TypeVarId, TypeVarId, String, Span)>,
    pub(super) current_fn_ret: Option<Ty>,
    pub(super) loop_nesting: u32,
    pub(super) loop_contexts: Vec<LoopContext>,
    pub(super) current_fn_generics: Option<Vec<hir::GenericParam>>,
    /// WP-C6.2b-F5: the ENCLOSING impl's generic parameters (with their bounds), in scope while an
    /// impl method body is checked so a bounded impl-head parameter's methods resolve — the impl
    /// analog of `current_fn_generics`.
    pub(super) current_impl_generics: Option<Vec<hir::GenericParam>>,
    /// DEV-051: set while type-checking a trait's own default-method bodies (alongside
    /// `current_self_ty = Ty::Param("Self")`) so `resolve_method` can look up a sibling trait
    /// method called through `self` directly against *this* trait's item list, the same way it
    /// already looks up a bounded generic type parameter's trait methods. `None` everywhere
    /// else (ordinary functions, `impl` method bodies, where `self`'s type is already concrete).
    pub(super) current_trait_id: Option<ItemId>,
    /// WP-C6.2b-F1: the module of the item whose body is being checked (the use-site module for
    /// member/field visibility). `None` before Pass 2.
    pub(super) current_module: Option<u32>,

    // Bounds checks to run at the end of checking
    /// Deferred trait-bound obligations. The 4th element is the generic environment ACTIVE
    /// WHERE THE OBLIGATION AROSE (DEV-067(a)): bounds are checked in a pass that runs after
    /// every body, by which time `current_fn_generics` belongs to whatever was checked last, so
    /// an obligation on a caller's own type parameter cannot be discharged unless the enclosing
    /// bounds travel with it.
    // DEV-101 made a deferred obligation carry the file that DECLARES the bounds, because a
    // bound's path name is only meaningful against its own file and these are discharged long
    // after the checker has moved on. AS1b-ii-d dropped it: `bound.path.span` names that file.
    pub(super) bounds_checks: Vec<BoundsCheck>,

    /// Enabled language extensions, threaded from the CLI through the whole
    /// front end (parse → resolve → typecheck).
    pub(super) options: LanguageOptions,

    /// Dimension/device unification state and provenance for the `tensor`
    /// extension (§5). Empty and unused for Core-only programs.
    pub(super) tensor_ctx: UnifyCtx,

    /// Dimension variables in scope for the item being checked, keyed by name
    /// (the `Dim` generic parameters of the enclosing function or model, §3.1).
    /// A dimension identifier not found here is an undeclared-dimension error.
    pub(super) dim_scope: HashMap<String, DimVar>,
    pub(super) dtype_scope: HashMap<String, DType>,
    pub(super) device_scope: HashMap<String, Device>,
    pub(super) generic_kinds: HashMap<String, GenericKind>,
    pub(super) suppress_tensor_diagnostics: bool,
    pub(super) allow_half_type: bool,
}

impl<'a> TypeChecker<'a> {
    pub(super) fn new(hir: &'a Hir, options: LanguageOptions) -> Self {
        TypeChecker {
            hir,
            options,
            diags: Vec::new(),
            subst: HashMap::new(),
            int_literal_vars: HashMap::new(),
            display_checks: Vec::new(),
            display_plans: Vec::new(),
            try_checks: Vec::new(),
            var_count: 0,
            expr_types: HashMap::new(),
            local_types: HashMap::new(),
            local_mutability: HashMap::new(),
            struct_fields: HashMap::new(),
            aggregate_field_types: HashMap::new(),
            enum_variants: HashMap::new(),
            fn_sigs: HashMap::new(),
            callable_sigs: HashMap::new(),
            callable_envs: HashMap::new(),
            callable_uses: Vec::new(),
            callable_uses_by_expr: HashMap::new(),
            display_uses: BTreeMap::new(),
            body_decls: None,
            const_types: HashMap::new(),
            alias_stack: Vec::new(),
            layout_queries: HashMap::new(),
            bound_trait_calls: HashMap::new(),
            current_self_ty: None,
            current_assoc_types: HashMap::new(),
            assoc_projections: HashMap::new(),
            projection_obligations: Vec::new(),
            current_fn_ret: None,
            loop_nesting: 0,
            loop_contexts: Vec::new(),
            current_fn_generics: None,
            current_impl_generics: None,
            current_trait_id: None,
            current_module: None,
            bounds_checks: Vec::new(),
            tensor_ctx: UnifyCtx::new(),
            dim_scope: HashMap::new(),
            dtype_scope: HashMap::new(),
            device_scope: HashMap::new(),
            generic_kinds: HashMap::new(),
            suppress_tensor_diagnostics: false,
            allow_half_type: false,
        }
    }

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

    /// Outer values displaced by entering an item.
    pub(super) fn enter_item_scope(&mut self, item_id: ItemId) -> Option<u32> {
        let saved = self.current_module;
        // WP-C6.2b-F1: the use-site module for visibility checks inside this item's body.
        self.current_module = self.hir.item_modules.get(&item_id).copied();
        saved
    }

    pub(super) fn exit_item_scope(&mut self, saved: Option<u32>) {
        self.current_module = saved;
    }

    /// Outer values displaced by entering a `Self`-carrying scope: an impl, a trait's default
    /// bodies, or a method whose receiver fixes `Self`.
    ///
    /// Narrow on purpose — the four fields of this family are not always installed together, so
    /// the caller passes only what it establishes and the rest is untouched (see [`SelfScope`]).
    /// Enter a scope in which `Self` is `self_ty`. Restores through [`exit_self_scope`].
    pub(super) fn enter_self_scope(&mut self, self_ty: Ty) -> SelfScope {
        SelfScope {
            self_ty: self.current_self_ty.replace(self_ty),
            assoc_types: None,
            impl_generics: None,
            trait_id: None,
        }
    }

    pub(super) fn exit_self_scope(&mut self, saved: SelfScope) {
        self.current_self_ty = saved.self_ty;
        if let Some(assoc) = saved.assoc_types {
            self.current_assoc_types = assoc;
        }
        if let Some(generics) = saved.impl_generics {
            self.current_impl_generics = generics;
        }
        if let Some(trait_id) = saved.trait_id {
            self.current_trait_id = trait_id;
        }
    }

    /// Outer values displaced by entering a function signature and body.
    pub(super) fn enter_fn_scope(
        &mut self,
        generics: Vec<hir::GenericParam>,
    ) -> (Option<Ty>, Option<Vec<hir::GenericParam>>) {
        let saved_ret = self.current_fn_ret.take();
        let saved_generics = self.current_fn_generics.replace(generics);
        (saved_ret, saved_generics)
    }

    /// Publish the function's return type once it has been converted. Separate from
    /// `enter_fn_scope` because the return type is converted *with the signature's generics
    /// already in scope* (WP-C7.9 Packet I).
    pub(super) fn set_fn_return(&mut self, ret: Ty) {
        self.current_fn_ret = Some(ret);
    }

    pub(super) fn exit_fn_scope(&mut self, saved: (Option<Ty>, Option<Vec<hir::GenericParam>>)) {
        self.current_fn_ret = saved.0;
        self.current_fn_generics = saved.1;
    }

    pub(super) fn exit_tensor_param_scope(&mut self, saved: TensorParamScopes) {
        self.dim_scope = saved.dims;
        self.dtype_scope = saved.dtypes;
        self.device_scope = saved.devices;
        self.generic_kinds = saved.kinds;
    }
}
