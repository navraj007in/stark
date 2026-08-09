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
    CalleeSelection, DeferredDisplayPlan, DispatchProvenance, DisplayPath, FnSigTy, GenericBinder,
    GenericEnvironment, GenericKind, LoopContext, ReceiverAdjustment, ReceiverBinding, Ty,
    TypeVarId, VariantTy,
};
use crate::diag::Diagnostic;
use crate::extensions::tensor::dim::DimVar;
use crate::extensions::tensor::types::{DType, Device, UnifyCtx};
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

    /// DEV-172 — the integer literal that is the direct operand of a unary `-`, if the checker is
    /// currently descending into one.
    ///
    /// A negative literal is parsed as negation applied to a POSITIVE literal, so the magnitude
    /// used to reach the range check is `128`, not `-128`. Checked against `Int8` that magnitude is
    /// out of range, and the same argument refuses the minimum of every signed width. The operand
    /// is recorded here before it is checked so the `Lit::Int` arm can range-check the value the
    /// program actually denotes.
    pub(super) negated_int_literal: Option<ExprId>,
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
            negated_int_literal: None,
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

    // AS7 Packet 6: source text and type rendering read the checker's own storage, so they are
    // state services. The dependency checker found `infer` and `state` reaching into `mod` for
    // them once its ownership map was repaired.
    /// Read a span, against the source the SPAN NAMES.
    ///
    /// AS1b-ii-d. This used to slice `self.file` — "the file currently being checked" — which was
    /// right for the item under check and wrong for every span belonging to another item. Four
    /// separate repairs of that one mistake are recorded (DEV-069, DEV-101, DEV-148 and its
    /// generic second site), each adding another way to carry the declaring file to the read:
    /// `item_text`, `item_src`, `item_file`, `decl_text`, a foreign-signature item stack, and a
    /// per-item `self.file` swap. All of it existed to answer a question the span now answers.
    ///
    /// `"?"` remains for an unresolvable span rather than a panic (WP-C4.7-4): a wrong read should
    /// be visible in a diagnostic, not a compiler crash.
    pub(super) fn text(&self, span: Span) -> &str {
        self.hir
            .sources
            .get(span.source)
            .and_then(|file| file.src.get(span.lo as usize..span.hi as usize))
            .unwrap_or("?")
    }

    // AS7 Packet 6: `item_text` is the file-provenance counterpart of `text` — the pair this
    // repo has already paid for once, when a span was read against the wrong file. They belong
    // in one module.
    /// AS1b-ii-d: the item is no longer consulted — `span` names its own source.
    pub(super) fn item_text(&self, _item: ItemId, span: Span) -> &str {
        self.text(span)
    }
}

// AS7 Packet 7: moved to the layer that owns the question.
/// The single-segment name of a path, if it has exactly one segment.
pub(super) fn single_segment_name<'t>(
    path: &crate::ast::Path,
    checker: &'t TypeChecker,
) -> Option<&'t str> {
    match path.segments.as_slice() {
        [seg] => Some(checker.text(seg.span)),
        _ => None,
    }
}

impl TypeChecker<'_> {
    pub(super) fn publish_callable_env(&mut self, published: PublishedEnv<'_>) {
        let PublishedEnv {
            call_expr,
            body,
            self_ty,
            impl_names,
            own_names,
            own_is_method,
            map,
        } = published;
        let bindings = Self::env_bindings(&self_ty, impl_names, own_names, own_is_method, map);
        self.callable_envs
            .insert(call_expr, CallableInstantiation { body, bindings });
    }
    /// Publish a `CallableUse` for a named-dispatch site (AS3 Boundary 2).
    ///
    /// Takes the same inputs the instantiation table already receives, so the two are built from
    /// one decision rather than two.
    #[allow(clippy::too_many_arguments, dead_code)] // consumed by Boundary 2.
    pub(super) fn publish_named_use(
        &mut self,
        call_expr: ExprId,
        body: BlockId,
        bindings: Vec<(GenericBinder, Ty)>,
        receiver_adjustment: ReceiverAdjustment,
        receiver_binding: ReceiverBinding,
        signature: CallableSigTy,
        provenance: DispatchProvenance,
    ) {
        let Some(declaration) = self.decl_for_body(body) else {
            // **A body with no declaration is an internal inconsistency, and is reported as one.**
            //
            // This used to `return`, which turned the inconsistency into MISSING METADATA — the
            // precise thing AS3's rule forbids, since an execution site finding no record must be
            // an internal compiler error rather than a licence to fall through and scan. Silently
            // omitting a use would have made the totality invariant unenforceable by construction.
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "internal compiler error: callable body {body:?} belongs to no item, impl                          member or trait member, so no CallableUse can be published for it"
                    ),
                    self.hir.expr(call_expr).span,
                )
                .with_code("E9001"),
            );
            return;
        };
        let use_ = CallableUse {
            selection: CalleeSelection::Static { declaration, body },
            environment: GenericEnvironment::Static(bindings),
            receiver_adjustment,
            receiver_binding,
            signature,
            provenance,
        };
        self.publish_callable_use(call_expr, use_);
    }
}

impl TypeChecker<'_> {
    /// **A3c-S: record the generic environment the checker selected for one callable use.**
    ///
    /// `map` is the substitution the caller already built while selecting the callable — impl
    /// parameters from candidate selection, then the callable's own parameters. `self_ty` is the
    /// impl's or trait's `Self`. All of it is already computed at every call site; before this it
    /// was discarded except for one positional slice, which is why impl generics, trait generics
    /// and `Self` never reached execution (DEV-176).
    ///
    /// Binders are recorded with their origin so consumers can derive an ordered view (MIR) or a
    /// name view (the oracle) from one stored answer rather than from separate tables.
    /// **Names arrive already resolved, and that is DEV-101's rule, not a convenience.** A generic
    /// parameter's name is a span into the file that DECLARED it, so a cross-package callee's
    /// parameters must be read with `item_text` against the callee's file — reading them with the
    /// caller's `decl_text` yields a different string, every `map` lookup misses, and the
    /// environment silently publishes as empty. Resolving names inside this helper is exactly how
    /// that regression happened, so the caller supplies them.
    /// The declaration a body belongs to, built once and cached.
    ///
    /// AS3: `CallableDeclId` needs the impl/trait member position, which the HIR expresses only by
    /// index into the owner's `items`. Deriving it per publication site would mean four scans
    /// written four ways; deriving it once means one.
    ///
    /// **This is a lookup, not a selection.** The body is already the checker's own answer — this
    /// only says which declaration that answer came from.
    #[allow(dead_code)] // consumed by Boundary 2, which is the next commit.
    pub(super) fn decl_for_body(&mut self, body: BlockId) -> Option<CallableDeclId> {
        if self.body_decls.is_none() {
            let mut map: HashMap<BlockId, CallableDeclId> = HashMap::new();
            for (index, item) in self.hir.items.iter().enumerate() {
                let owner = ItemId(index as u32);
                match &item.kind {
                    hir::ItemKind::Fn(def) => {
                        map.insert(def.body, CallableDeclId::Item(owner));
                    }
                    hir::ItemKind::Impl { items, .. } => {
                        for (member, impl_item) in items.iter().enumerate() {
                            if let hir::ImplItem::Fn { def, .. } = impl_item {
                                map.insert(
                                    def.body,
                                    CallableDeclId::ImplMember {
                                        impl_item: owner,
                                        member: member as u32,
                                    },
                                );
                            }
                        }
                    }
                    hir::ItemKind::Trait { items, .. } => {
                        for (member, trait_item) in items.iter().enumerate() {
                            if let hir::TraitItem::Method { body: Some(b), .. } = trait_item {
                                map.insert(
                                    *b,
                                    CallableDeclId::TraitMember {
                                        trait_item: owner,
                                        member: member as u32,
                                    },
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
            self.body_decls = Some(map);
        }
        self.body_decls
            .as_ref()
            .and_then(|map| map.get(&body))
            .copied()
    }
    /// **AS3: publish one callable use.** The single point at which the checker's selection becomes
    /// something an engine may consume.
    ///
    /// Returns the id so a caller that needs to refer to the use later can. Ungrounded types are
    /// fine here — `analyze` grounds every published use once, at the end, the same way it grounds
    /// `callable_instantiations`.
    pub(super) fn publish_callable_use(
        &mut self,
        expr: ExprId,
        use_: CallableUse,
    ) -> CallableUseId {
        let id = CallableUseId(self.callable_uses.len() as u32);
        self.callable_uses.push(use_);
        self.callable_uses_by_expr.entry(expr).or_default().push(id);
        id
    }
}

/// One call site's inputs to [`Checker::publish_callable_env`].
///
/// A struct rather than seven positional parameters: the two name slices are the same type and
/// differ only in which declaration they came from, so an argument-order slip would compile and
/// publish an environment with impl and method binders exchanged.
pub(super) struct PublishedEnv<'a> {
    pub(super) call_expr: ExprId,
    pub(super) body: BlockId,
    pub(super) self_ty: Option<Ty>,
    pub(super) impl_names: &'a [String],
    pub(super) own_names: &'a [String],
    pub(super) own_is_method: bool,
    pub(super) map: &'a HashMap<String, Ty>,
}

impl TypeChecker<'_> {
    /// WP-C1.3: whether `ty` satisfies the compiler-known operator-desugaring bound `required`
    /// ("Num" | "Eq" | "Ord"). Recurses into `Ty::Core` container type arguments (`Option<T>`,
    /// `Result<T, E>`, `Vec<T>`, `Box<T>`) so e.g. `Option<Int32> == Option<Int32>` type-checks
    /// -- container types have no `Ty::Core` arm at all before this WP, so every `==`/`<` on any
    /// of these normatively "essential" standard-library types (06-Standard-Library.md) was
    /// unconditionally rejected. `HashMap`/`HashSet`/iterator/`Random`/`IOError` CoreTypes are
    /// deliberately excluded: they are not normatively specified as Eq/Ord-comparable, and
    /// giving them one now would be new semantics, not a bug fix (Charter rule 4).
    /// Publish the `CallableUse` for an operator that dispatches to a user body (AS3 Boundary 3).
    ///
    /// Only user nominals reach a user body: primitives have built-in operator meaning (DEV-075),
    /// and a `Ty::Core` composite compares element-wise through the runtime rather than through one
    /// selected callable. Publishing nothing for those is correct — they are not callable uses.
    /// The `Bound` half of [`Self::publish_operator_use`]: `a == b` where `a: T` and `T: Eq`.
    ///
    /// The signature comes from the Core trait's own contract — the same `CoreTraitMethod` table a
    /// user `impl` is checked against — rather than from `Eq::eq -> Bool` written in by hand. There
    /// is deliberately no second statement of what these operators mean.
    /// **AS3 Boundary 4: the `Display` dispatch plan for one `println` argument.**
    ///
    /// Walks the argument's STATIC type in the shape `display_deep` and `emit_display_value`
    /// already render (`AS3-DISPLAY-CHARACTERIZATION.md` §2.3), publishing one use per position
    /// that reaches a user body.
    ///
    /// **The STOP rule is the load-bearing part.** `println(W<A>)` prints `W!`, not a `W!`
    /// containing an `A!`: the outer nominal's own `fmt` runs and the renderer does not descend
    /// into its fields. So the walk stops at the first nominal with a `Display` impl. Descending
    /// further would publish uses no engine executes, and a totality claim over those would be
    /// false.
    /// Queue one expression that renders through `Display`, with the generic scope it was
    /// written in. Called by `println`-family arguments and by interpolation fields alike.
    pub(super) fn record_display_plan(&mut self, root: ExprId, ty: Ty) {
        self.display_plans.push(DeferredDisplayPlan {
            root,
            ty,
            generic_scope: (
                self.current_fn_generics.clone(),
                self.current_impl_generics.clone(),
            ),
        });
    }
}

impl TypeChecker<'_> {
    /// The declared name of a nominal (struct/enum) item, read against its declaring file.
    pub(super) fn nominal_name(&self, item: ItemId) -> String {
        match &self.hir.item(item).kind {
            hir::ItemKind::Struct { name, .. } | hir::ItemKind::Enum { name, .. } => {
                self.item_text(item, *name).to_string()
            }
            _ => String::new(),
        }
    }
}
