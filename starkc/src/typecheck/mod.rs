//! Type checking, mutability, and definite assignment validation pass for STARK (PLAN.md M2.2).

// AS6 packet 4B group 2C: the tensor semantic authority lives in
// `extensions::tensor::check`. What remains here is the integration boundary — locating an
// operation in `TENSOR_OPS`, validating the call form, evaluating arguments, and publishing the
// type the extension decided — plus the `TensorCheckCtx` impl that names, exhaustively, the Core
// services the extension is allowed to use.
use crate::diag::Diagnostic;
use crate::extensions::tensor::check::TensorCheckCtx;
use crate::extensions::tensor::dim::Poly;
use crate::extensions::tensor::syntax as tensor_syntax;
use crate::extensions::tensor::types::{DType, Device, Shape, UnifyCtx};
use crate::hir::{self, CoreType, ExprId, Hir, ItemId, PatId, TypeId};
use crate::options::LanguageOptions;
use crate::source::Span;
use std::collections::{HashMap, HashSet};

mod body;
mod bounds;
/// WP-C6.2b-F1: a selected inherent/trait method candidate carried through visibility enforcement:
/// (signature def, is-trait-method, impl substitution, impl self type, member is `pub`, impl item).
mod convert;
mod infer;
mod items;
mod patterns;
mod state;
mod traits;
use state::TypeChecker;
use traits::core_trait_bound_method;
use traits::is_copy_with_impls;
pub use traits::{copy_eligible_types, nominals_with_destructor};

mod types;
use types::GenericKind;
pub use types::{
    collect_ty_params, substitute_ty, ty_contains_param, CallableDeclId, CallableInstantiation,
    CallableSigTy, CallableUse, CallableUseId, CalleeSelection, DispatchProvenance, DisplayPath,
    DisplayStep, ExtensionTy, GenericBinder, GenericEnvironment, LayoutTables, ModelTy,
    ReceiverAdjustment, ReceiverBinding, Ty, TypeCheckResult, TypeTables, TypeVarId,
};

// Crate-internal pure helpers of the representation, used by the passes still in `mod.rs`.
use types::{ty_contains_error, ty_contains_infer};

impl From<tensor_syntax::TensorParamKind> for GenericKind {
    fn from(kind: tensor_syntax::TensorParamKind) -> Self {
        match kind {
            tensor_syntax::TensorParamKind::Dim => GenericKind::Dim,
            tensor_syntax::TensorParamKind::DType => GenericKind::DType,
            tensor_syntax::TensorParamKind::Device => GenericKind::Device,
        }
    }
}

#[cfg(test)]
mod layout_substitution_tests {
    use super::*;
    use crate::ast::Primitive;

    fn map(pairs: &[(&str, Ty)]) -> HashMap<String, Ty> {
        pairs
            .iter()
            .map(|(n, t)| ((*n).to_string(), t.clone()))
            .collect()
    }

    fn param(name: &str) -> Ty {
        Ty::Param(name.to_string())
    }

    /// DEV-100's mutation case: substitution must RECURSE. A version that only swapped a bare
    /// `Ty::Param` would leave every one of these unchanged, and each is a shape a real query
    /// takes — `size_of::<[T; 4]>()`, `size_of::<Pair<T>>()`, `size_of::<Option<T>>()`.
    #[test]
    fn substitution_reaches_every_position_a_parameter_can_hide_in() {
        let m = map(&[("T", Ty::Primitive(Primitive::Int32))]);
        let int32 = Ty::Primitive(Primitive::Int32);

        let cases: Vec<(Ty, Ty)> = vec![
            (param("T"), int32.clone()),
            (
                Ty::Array(Box::new(param("T")), 4),
                Ty::Array(Box::new(int32.clone()), 4),
            ),
            (
                Ty::Tuple(vec![param("T"), Ty::Primitive(Primitive::Int8)]),
                Ty::Tuple(vec![int32.clone(), Ty::Primitive(Primitive::Int8)]),
            ),
            (
                Ty::Struct(ItemId(7), vec![param("T")]),
                Ty::Struct(ItemId(7), vec![int32.clone()]),
            ),
            (
                Ty::Enum(ItemId(9), vec![param("T")]),
                Ty::Enum(ItemId(9), vec![int32.clone()]),
            ),
            (
                Ty::Core(CoreType::Option, vec![param("T")]),
                Ty::Core(CoreType::Option, vec![int32.clone()]),
            ),
            (
                Ty::Core(CoreType::Result, vec![param("T"), param("T")]),
                Ty::Core(CoreType::Result, vec![int32.clone(), int32.clone()]),
            ),
            (
                Ty::Ref {
                    mutable: false,
                    inner: Box::new(param("T")),
                },
                Ty::Ref {
                    mutable: false,
                    inner: Box::new(int32.clone()),
                },
            ),
            (
                Ty::Fn {
                    params: vec![param("T")],
                    ret: Box::new(param("T")),
                },
                Ty::Fn {
                    params: vec![int32.clone()],
                    ret: Box::new(int32.clone()),
                },
            ),
            // Nested two deep: `[(T, Int8); 2]`.
            (
                Ty::Array(
                    Box::new(Ty::Tuple(vec![param("T"), Ty::Primitive(Primitive::Int8)])),
                    2,
                ),
                Ty::Array(
                    Box::new(Ty::Tuple(vec![
                        int32.clone(),
                        Ty::Primitive(Primitive::Int8),
                    ])),
                    2,
                ),
            ),
        ];

        for (before, expected) in cases {
            let after = substitute_ty(&before, &m);
            assert_eq!(after, expected, "substitution missed {before:?}");
            assert!(
                ty_contains_param(&before),
                "the case must actually contain a parameter: {before:?}"
            );
            assert!(
                !ty_contains_param(&after),
                "a parameter survived substitution in {before:?}"
            );
        }
    }

    /// The mutation the directive names: remove the PUSH (substitute against an empty frame). The
    /// parameter survives and `ty_contains_param` catches it, which is what turns a missing frame
    /// into a visible oracle defect instead of a wrong observable answer.
    #[test]
    fn without_a_substitution_frame_the_parameter_survives_and_is_detected() {
        let empty = HashMap::new();
        let queried = Ty::Array(Box::new(param("T")), 4);
        let after = substitute_ty(&queried, &empty);
        assert_eq!(after, queried, "an empty frame must change nothing");
        assert!(
            ty_contains_param(&after),
            "an unsubstituted parameter must be detectable, not silently laid out"
        );
    }

    /// A frame that binds a DIFFERENT parameter is not a partial match to fall back on.
    #[test]
    fn an_unrelated_frame_leaves_the_parameter_in_place() {
        let m = map(&[("U", Ty::Primitive(Primitive::Int64))]);
        let after = substitute_ty(&param("T"), &m);
        assert!(ty_contains_param(&after));
    }
}

// ------------------------------------------------------- AS3 / WP-CALLABLE-USE-TOTAL --

// AS3 Boundary 4 uses `hir::BoundTrait` — `User(ItemId)` or `Core(CoreTrait)` — which the compiler
// already carries because user traits and compiler-known traits both occur as bounds.
//
// This resolves a model bug found by the Display characterization: `DispatchProvenance::Bound
// { trait_item: ItemId }` could not represent `T: Display`, since `Display` is a `CoreTrait` with
// no trait `ItemId`. Selection and provenance now speak the same identity language, and it is the
// language the rest of the compiler already speaks.

/// AS1b-ii-d: no `file` parameter. The checker used to be handed the root file and re-aim it at
/// each item's declaring file as it walked; every span it reads now names its own source, which the
/// `Hir`'s registry resolves.
pub fn check(hir: &Hir) -> Vec<Diagnostic> {
    analyze(hir).diagnostics
}

/// Core-only [`check`], with the option-aware pipeline (Gate 4+).
pub fn check_with_options(hir: &Hir, options: LanguageOptions) -> Vec<Diagnostic> {
    analyze_with_options(hir, options).diagnostics
}

pub fn analyze(hir: &Hir) -> TypeCheckResult {
    analyze_with_options(hir, LanguageOptions::CORE)
}

/// AS7 Packet 1: the constructor extracted from `analyze_with_options`, unchanged.
/// Behaviour-identical — it exists so the ambient-state harness can build a checker without
/// duplicating sixty field initialisers, and because `state.rs` will own it.
impl<'a> TypeChecker<'a> {}

pub fn analyze_with_options(hir: &Hir, options: LanguageOptions) -> TypeCheckResult {
    let mut checker = TypeChecker::new(hir, options);

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
    let callable_uses: Vec<CallableUse> = checker
        .callable_uses
        .iter()
        .map(|use_| CallableUse {
            // **Grounded like every other published field.** It was the ONE field copied verbatim,
            // and a `Bound` selection carries types: `self_ty`, `trait_args`, and the method's own
            // `method_args`. An inferred method argument (`t.to(1)` with no turbofish) is resolved
            // when `check_trait_member_call` returns, but the integer literal that determines it is
            // not defaulted until later — so the selection published `Infer(N)`, the specialiser
            // built an environment binding `U -> Infer(N)`, and the return boundary compared a
            // value against an inference variable. Found by DEV-121's return boundary; nothing
            // observed it before because nothing read the environment.
            selection: ground_selection(&checker, &use_.selection),
            environment: match &use_.environment {
                GenericEnvironment::Static(bindings) => GenericEnvironment::Static(
                    bindings
                        .iter()
                        .map(|(binder, ty)| (binder.clone(), checker.ground(ty)))
                        .collect(),
                ),
                GenericEnvironment::FromBoundSelection => GenericEnvironment::FromBoundSelection,
                GenericEnvironment::FromFunctionValue => GenericEnvironment::FromFunctionValue,
            },
            receiver_adjustment: use_.receiver_adjustment,
            receiver_binding: use_.receiver_binding,
            signature: CallableSigTy {
                receiver: use_
                    .signature
                    .receiver
                    .as_ref()
                    .map(|ty| checker.ground(ty)),
                params: use_
                    .signature
                    .params
                    .iter()
                    .map(|ty| checker.ground(ty))
                    .collect(),
                ret: checker.ground(&use_.signature.ret),
            },
            provenance: use_.provenance.clone(),
        })
        .collect();
    let callable_uses_by_expr = checker.callable_uses_by_expr.clone();
    let trait_impls = checker.build_trait_impl_index();
    let callable_instantiations = checker
        .callable_envs
        .iter()
        .map(|(&expr, env)| {
            (
                expr,
                CallableInstantiation {
                    body: env.body,
                    bindings: env
                        .bindings
                        .iter()
                        .map(|(binder, ty)| (binder.clone(), checker.ground(ty)))
                        .collect(),
                },
            )
        })
        .collect();
    let callable_types = checker
        .callable_sigs
        .iter()
        .map(|(&body, sig)| {
            (
                body,
                CallableSigTy {
                    receiver: sig.receiver.as_ref().map(|ty| checker.ground(ty)),
                    params: sig.params.iter().map(|ty| checker.ground(ty)).collect(),
                    ret: checker.ground(&sig.ret),
                },
            )
        })
        .collect();
    let fn_types = checker
        .fn_sigs
        .iter()
        .map(|(&id, sig)| {
            (
                id,
                (
                    sig.params.iter().map(|ty| checker.ground(ty)).collect(),
                    checker.ground(&sig.ret),
                ),
            )
        })
        .collect();
    // WP-C4.5c (DEV-064): every use of a generic fn must have fully determined generic
    // arguments once inference completes — an undetermined instantiation cannot be
    // monomorphised, so it is rejected here (TYPE-GENERIC-001: "if any parameter remains
    // unconstrained, the call requires explicit arguments"; TYPE-FN-002 for the fn-value
    // coercion form), never left for a backend to trip over. `Ty::Param` entries are fine:
    // inside a generic body they are determined by the enclosing instantiation.
    // A3c-S: the undetermined-instantiation check moved onto the single table, but deliberately
    // over the SAME subset it always covered — a callable's OWN arguments. The environment also
    // carries impl, trait and `Self` bindings, and validating those here would reject programs that
    // previously passed, turning a table migration into a language change.
    let mut undetermined: Vec<Span> = Vec::new();
    for (&expr_id, env) in &checker.callable_envs {
        let grounded: Vec<Ty> = env
            .own_arguments()
            .iter()
            .map(|ty| checker.ground(ty))
            .collect();
        if grounded.iter().any(ty_contains_error) {
            continue; // the use site already failed checking; avoid a cascade
        }
        if grounded.iter().any(ty_contains_infer) {
            undetermined.push(hir.expr(expr_id).span);
        }
    }
    let layout_queries: HashMap<ExprId, Ty> = checker
        .layout_queries
        .iter()
        .map(|(&expr_id, ty)| (expr_id, checker.ground(ty)))
        .collect();
    let layout = checker.build_layout_tables();

    undetermined.sort_by_key(|span| (span.lo, span.hi));
    for span in undetermined {
        checker.diags.push(
            Diagnostic::error(
                "cannot infer the generic arguments for this use of a generic function; \
                 supply them explicitly with `::<...>`",
                span,
            )
            .with_code("E0004"),
        );
    }
    let assoc_projections: HashMap<(ItemId, String), Ty> = checker
        .assoc_projections
        .iter()
        .map(|((nominal, name), ty)| ((*nominal, name.clone()), checker.ground(ty)))
        .collect();
    let aggregate_field_types: HashMap<ExprId, HashMap<String, Ty>> = checker
        .aggregate_field_types
        .iter()
        .map(|(expr, fields)| {
            (
                *expr,
                fields
                    .iter()
                    .map(|(name, ty)| (name.clone(), checker.ground(ty)))
                    .collect(),
            )
        })
        .collect();
    let mut diagnostics = checker.diags;
    diagnostics.extend(crate::flow::check(hir, &expr_types));
    diagnostics.extend(crate::borrowck::check(hir, &expr_types, &local_types));
    let tables = TypeTables {
        expr_types,
        local_types,
        local_mutability: checker.local_mutability,
        fn_types,
        assoc_projections,
        aggregate_field_types,
        callable_types,
        callable_instantiations,
        callable_uses,
        callable_uses_by_expr,
        display_uses: checker.display_uses,
        trait_impls,
        layout_queries,
        layout,
        bound_trait_calls: checker.bound_trait_calls,
    };
    diagnostics.extend(crate::interp::check_constants(hir, &tables));
    TypeCheckResult {
        diagnostics,
        tables,
    }
}

/// Ground every type a [`CalleeSelection`] carries.
///
/// `Static` and `FunctionValue` carry none — they name a declaration and a body, not types — so
/// they pass through. Written as an exhaustive match rather than a `if let Bound` so a future
/// variant that carries a type cannot be published ungrounded by omission.
fn ground_selection(checker: &TypeChecker<'_>, selection: &CalleeSelection) -> CalleeSelection {
    match selection {
        CalleeSelection::Static { .. } | CalleeSelection::FunctionValue => selection.clone(),
        CalleeSelection::Bound {
            trait_,
            member,
            self_ty,
            trait_args,
            method_args,
        } => CalleeSelection::Bound {
            trait_: *trait_,
            member: member.clone(),
            self_ty: checker.ground(self_ty),
            trait_args: trait_args.iter().map(|ty| checker.ground(ty)).collect(),
            method_args: method_args.iter().map(|ty| checker.ground(ty)).collect(),
        },
    }
}

impl<'a> TypeChecker<'a> {
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

    // -----------------------------------------------------------------------------------------
    // AS7 Packet 2 — scoped ambient contexts.
    //
    // The eight ambient fields are NOT one context: they have different dynamic scopes, and
    // collapsing them into a single `AmbientContext` would make it possible to restore a
    // function's return type while leaving an impl's `Self` behind. They are grouped by the scope
    // they actually belong to, and each group has a matching enter/exit pair.
    //
    // Every pair SAVES AND RESTORES. None of them clears to a default. That distinction is the
    // whole point of the packet: `check_fn_def` used to clear `current_fn_ret` to `None` on exit,
    // which is correct only while item checking never nests — and AS7's own splitting is the
    // change most likely to break that invariant. Restoring is identical in behaviour today and
    // correct by construction tomorrow.
    // -----------------------------------------------------------------------------------------

    /// WP-C2.2 (DEV-031): recover the `Iterator::Item` associated type for a nominal user
    /// iterator so a `for` loop can type its binding from the trait implementation.
    fn user_iterator_item_type(&mut self, iter_ty: &Ty) -> Option<Ty> {
        // DEV-073 (WP-C4.7-5): a GENERIC `Iterator` impl must be recognized for a concrete
        // instantiation — `impl<T> Iterator for Repeat<T>` makes `Repeat<Int32>` iterable. Like
        // the operator-bound check, this used to demand an EXACT type match against an impl self
        // type whose generic arguments had been dropped, so `for x in r` on any generic user
        // iterator was rejected E0001. Matching now goes through `match_impl_type`, and the
        // resulting substitution is applied to the associated `Item` — `type Item = T` on
        // `Repeat<Int32>` must yield `Int32`, not a dangling `Ty::Param`.
        // DEV-069: impl/assoc-type names are read against the declaring impl's own file.
        let selection = self.resolve_user_iterator(iter_ty)?;
        let item_ty = self.convert_hir_type(selection.associated_item);
        Some(self.instantiate_ty(&item_ty, &selection.substitutions))
    }

    /// The bounds the standard library declares on its own generic types (WP-C7.9 Packet I).
    ///
    /// `06-Standard-Library.md` writes `HashMap<K: Hash + Eq, V>` and `HashSet<T: Hash + Eq>`.
    /// **DEV-118 was that neither half was enforced** — and specifically that all three engines
    /// accepted the same invalid instantiations, because the current storage scans by `Eq` and
    /// never consults a hash. That is the most dangerous shape of omission this project has: the
    /// engines agree, so no differential can see it, and it becomes a live divergence the moment
    /// one implementation starts hashing. Consistency is not conformance.
    ///
    /// Returns, per parameter position, the trait names that position must satisfy.
    fn builtin_type_bounds(core: CoreType) -> &'static [(usize, &'static [&'static str])] {
        match core {
            // `K` is hashed and compared; `V` is neither.
            CoreType::HashMap => &[(0, &["Hash", "Eq"])],
            CoreType::HashSet => &[(0, &["Hash", "Eq"])],
            _ => &[],
        }
    }
}

impl PatId {
    fn span(&self, hir: &Hir) -> Span {
        hir.pat(*self).span
    }
}

// ------------------------------------------- WP-C7.9 Packet B: Core-trait implementation contracts --

// ------------------------------------------------- DEV-DISPLAY-DISPATCH: bounds as one surface --

/// DEV-DISPLAY-DISPATCH: the receiver form a compiler-known trait declares for `name`, for the
/// passes outside the type checker that need it — the move checker most of all, which must know
/// that `Display::fmt` BORROWS before it decides whether `x.fmt()` consumed `x`.
///
/// Reads [`core_trait_contract`], the same table everything else does; there is one source for a
/// Core trait's signatures, not one per consumer.
pub fn core_trait_method_receiver(core_trait: hir::CoreTrait, name: &str) -> Option<hir::Receiver> {
    core_trait_bound_method(core_trait, name).and_then(|method| method.receiver)
}

impl<'a> TypeChecker<'a> {}

/// Whether `ty` is `Copy`, given the set of `Copy`-eligible nominals.
///
/// **Published for WP-VALUE-REP-TOTAL A2.** The representation relation permits `&T` to be
/// represented by a bare `T` only when the POINTEE is `Copy`, and that must be the same answer the
/// checker uses — a second Copy predicate in the interpreter would be a second definition of move
/// behaviour, which is the disagreement WP-COPY-CANON exists to prevent.
pub fn is_copy_type_with(ty: &Ty, copy_types: &HashSet<ItemId>) -> bool {
    is_copy_with_impls(ty, copy_types)
}

impl TypeChecker<'_> {}

/// AS6 packet 4B group 2C: the Core services the tensor semantic rules may use, and the whole of
/// what they may use. `check_expr` is deliberately not among them — the tensor checker consumes
/// checked expression types, it does not cause expression checking.
impl TensorCheckCtx for TypeChecker<'_> {
    fn diags(&mut self) -> &mut Vec<Diagnostic> {
        &mut self.diags
    }

    fn tensor_error(&mut self, message: &str, span: Span) {
        TypeChecker::tensor_error(self, message, span)
    }

    fn resolve(&self, ty: &Ty) -> Ty {
        TypeChecker::resolve(self, ty)
    }

    fn unify(&mut self, a: Ty, b: Ty, span: Span) -> Result<(), ()> {
        TypeChecker::unify(self, a, b, span)
    }

    fn ty_to_string(&self, ty: &Ty) -> String {
        TypeChecker::ty_to_string(self, ty)
    }

    fn extract_const_int(&self, arg: &hir::GenericArg) -> Option<i64> {
        TypeChecker::extract_const_int(self, arg)
    }

    fn extract_const_int_list(&mut self, arg: &hir::GenericArg) -> Option<Vec<i64>> {
        TypeChecker::extract_const_int_list(self, arg)
    }

    fn extract_dim_generic(&mut self, arg: &hir::GenericArg, label: &str) -> Option<Poly> {
        TypeChecker::extract_dim_generic(self, arg, label)
    }

    fn combine_value_range(
        &self,
        a: crate::extensions::tensor::types::ValueRange,
        b: crate::extensions::tensor::types::ValueRange,
    ) -> Option<crate::extensions::tensor::types::ValueRange> {
        TypeChecker::combine_value_range(self, a, b)
    }

    fn value_range_of(
        &mut self,
        generic_args: &hir::GenericArgs,
    ) -> crate::extensions::tensor::types::ValueRange {
        TypeChecker::value_range_of(self, generic_args)
    }

    fn build_shape(&mut self, shape: &hir::ShapeArg) -> Shape {
        TypeChecker::build_shape(self, shape)
    }

    fn build_refine_shape(&mut self, shape: &hir::ShapeArg) -> Shape {
        TypeChecker::build_refine_shape(self, shape)
    }

    fn build_device(&mut self, arg: Option<&hir::GenericArg>, span: Span) -> Device {
        TypeChecker::build_device(self, arg, span)
    }

    fn tensor_dtype(&mut self, ty_id: TypeId, span: Span) -> DType {
        TypeChecker::tensor_dtype(self, ty_id, span)
    }

    fn tensor_state(&mut self) -> &mut UnifyCtx {
        &mut self.tensor_ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // AS7 Packet 10: the test module needs names the production facade no longer does. Imported
    // here rather than at the top, because `cargo check --lib` does not compile `#[cfg(test)]`
    // code and would report an unused import that `cargo test` then needs.
    use crate::ast::Primitive;
    use crate::parser::{parse, ParseMode};
    use crate::resolve::resolve;
    use crate::source::SourceFile;
    use std::sync::Arc;

    fn check_src(src: &str) -> Vec<Diagnostic> {
        let file = Arc::new(SourceFile::new("test.stark".to_string(), src.to_string()));
        let (tree, diags) = parse(&file, ParseMode::Program);
        assert!(diags.is_empty(), "parse failed: {:?}", diags);
        let (hir, sem_diags) = resolve(&tree, file.clone());
        let mut all_diags = sem_diags.clone();
        let mut type_diags = check(&hir);
        all_diags.append(&mut type_diags);
        all_diags
    }

    // ---------------------------------------------------------------------------------------
    // AS7 Packet 1 — the ambient-state harness.
    //
    // The AS7 opening inventory (3f18e49) found no ambient defect and two LATENT sites: the
    // ambient state is correct today, and `current_fn_ret`/`current_fn_generics` are correct only
    // because item checking never nests. AS7's file splitting is the change most likely to break
    // that invariant.
    //
    // This harness exists BEFORE the conversion, for the reason AS6 paid to learn: a quarantine
    // — or here, a scoped context — that silently restores the wrong value produces PLAUSIBLE
    // wrong answers, not crashes, and has no behavioural signature at all.
    //
    // Two of the assertions below deliberately pin the CURRENT, invariant-dependent behaviour.
    // They are written to FAIL when AS7 Packet 2 converts those fields to save/restore, and that
    // failure is the evidence the conversion happened. This is the pattern AS0 used against
    // AS1a: commit the assertion that describes today, and let the fix flip it.
    // ---------------------------------------------------------------------------------------

    /// The six ambient fields that already save and restore must **nest**: an inner scope must
    /// see its own value, and leaving it must restore the outer one — not clear it, and not leak.
    #[test]
    fn as7_saved_ambient_fields_nest_correctly() {
        let file = Arc::new(SourceFile::new("t.stark".to_string(), String::new()));
        let (tree, _) = parse(&file, ParseMode::Program);
        let (hir, _) = resolve(&tree, file.clone());
        let mut tc = TypeChecker::new(&hir, LanguageOptions::default());

        let outer = Ty::Primitive(Primitive::Int32);
        let inner = Ty::Primitive(Primitive::Bool);

        // current_self_ty — the `.replace()` / restore pattern used at eight sites.
        assert_eq!(tc.current_self_ty, None, "entry state");
        let save_outer = tc.current_self_ty.replace(outer.clone());
        assert_eq!(
            tc.current_self_ty,
            Some(outer.clone()),
            "outer scope installed"
        );
        let save_inner = tc.current_self_ty.replace(inner.clone());
        assert_eq!(tc.current_self_ty, Some(inner), "inner scope installed");
        tc.current_self_ty = save_inner;
        assert_eq!(
            tc.current_self_ty,
            Some(outer),
            "leaving the inner scope must RESTORE the outer value, not clear it"
        );
        tc.current_self_ty = save_outer;
        assert_eq!(tc.current_self_ty, None, "original state restored");

        // current_trait_id and current_impl_generics use the same shape.
        let a = tc.current_trait_id.replace(ItemId(1));
        let b = tc.current_trait_id.replace(ItemId(2));
        assert_eq!(tc.current_trait_id, Some(ItemId(2)));
        tc.current_trait_id = b;
        assert_eq!(tc.current_trait_id, Some(ItemId(1)), "outer trait restored");
        tc.current_trait_id = a;
        assert_eq!(tc.current_trait_id, None);
    }

    /// **The first latent site, now converted.** `check_fn_def` used to set `current_fn_ret` and
    /// then CLEAR it to `None`, which was correct only while item checking never nests. It now
    /// saves and restores, so nesting is correct by construction.
    ///
    /// This test exercises the real `enter_fn_scope`/`exit_fn_scope` helpers. Packet 1's version
    /// asserted the same property by assigning the fields by hand — it therefore pinned a
    /// *pattern* rather than the implementation, and did **not** fail when the implementation
    /// changed. That was a defect in the harness, and this is the correction: a test that does not
    /// call the code it describes cannot detect the code changing.
    #[test]
    fn as7_fn_scope_saves_and_restores_the_enclosing_function() {
        let file = Arc::new(SourceFile::new("t.stark".to_string(), String::new()));
        let (tree, _) = parse(&file, ParseMode::Program);
        let (hir, _) = resolve(&tree, file.clone());
        let mut tc = TypeChecker::new(&hir, LanguageOptions::default());

        let outer_ret = Ty::Primitive(Primitive::Int32);
        let inner_ret = Ty::Primitive(Primitive::Bool);

        let outer = tc.enter_fn_scope(Vec::new());
        tc.set_fn_return(outer_ret.clone());
        assert_eq!(tc.current_fn_ret, Some(outer_ret.clone()));

        let inner = tc.enter_fn_scope(Vec::new());
        tc.set_fn_return(inner_ret.clone());
        assert_eq!(
            tc.current_fn_ret,
            Some(inner_ret),
            "inner function installed"
        );

        tc.exit_fn_scope(inner);
        assert_eq!(
            tc.current_fn_ret,
            Some(outer_ret),
            "leaving the inner function must RESTORE the enclosing return type. Clearing it to \
             None — the pre-Packet-2 behaviour — loses it, and is detectable only under nesting."
        );

        tc.exit_fn_scope(outer);
        assert_eq!(tc.current_fn_ret, None, "original state restored");
        assert!(tc.current_fn_generics.is_none());
    }

    /// **The second latent site, now converted.** `current_module` was assigned once per item and
    /// never restored — correct only because that single write dominated every item branch.
    #[test]
    fn as7_item_scope_saves_and_restores_the_enclosing_module() {
        let file = Arc::new(SourceFile::new("t.stark".to_string(), String::new()));
        let (tree, _) = parse(&file, ParseMode::Program);
        let (hir, _) = resolve(&tree, file.clone());
        let mut tc = TypeChecker::new(&hir, LanguageOptions::default());

        tc.current_module = Some(7);
        let saved = tc.enter_item_scope(ItemId(0));
        tc.exit_item_scope(saved);
        assert_eq!(
            tc.current_module,
            Some(7),
            "leaving an item must restore the enclosing module rather than leaving the item's own"
        );
    }

    /// The structural half: no ambient field may be *cleared* on scope exit anywhere in production
    /// code. Clearing is the pre-Packet-2 shape, and it is invisible until something nests.
    #[test]
    fn as7_no_ambient_field_is_cleared_on_scope_exit() {
        let source =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/typecheck/mod.rs"))
                .expect("own source")
                .replace("\r\n", "\n");
        // `rfind`, not `find`: this file has an earlier inline test module, so the first marker
        // would truncate almost the whole file and the check would pass vacuously.
        let production = match source.rfind("#[cfg(test)]") {
            Some(i) => &source[..i],
            None => source.as_str(),
        };
        let mut cleared = Vec::new();
        for field in [
            "current_fn_ret",
            "current_fn_generics",
            "current_self_ty",
            "current_trait_id",
            "current_impl_generics",
            "current_module",
        ] {
            if production.contains(&format!("self.{field} = None;")) {
                cleared.push(field);
            }
        }
        assert!(
            cleared.is_empty(),
            "these ambient fields are CLEARED rather than restored on exit: {cleared:?}. \
             Clearing is correct only while the scope never nests; save the outer value and put \
             it back instead."
        );
    }

    /// DEV-051: a trait default method body calling another method of the same trait through
    /// `self` used to fail with `E0302 method 'name' not found for type '&Self'`. Root cause:
    /// `resolve_method`'s only mechanism for a receiver with no concrete `impl` to match (an
    /// abstract `Ty::Param` receiver) was scoped to a bounded *generic function* type parameter
    /// (`fn f<T: Greet>(x: T)`), never to `self` inside a trait's own default-method body
    /// (`current_self_ty == Ty::Param("Self")`while type-checking that body generically, once,
    /// at the trait declaration site). Fixed by adding `current_trait_id` (set alongside
    /// `current_self_ty` for a trait's default bodies) and checking it the same way, after the
    /// reference-deref loop since `self` is always received by reference unlike a by-value
    /// generic parameter. Confirmed empirically via `starkc check` before writing this test.
    #[test]
    fn trait_default_method_calling_sibling_trait_method_through_self_type_checks() {
        let src = "trait Greet { \
                       fn name(&self) -> String; \
                       fn greeting(&self) -> String { self.name() } \
                   } \
                   struct Person { label: String } \
                   impl Greet for Person { \
                       fn name(&self) -> String { self.label.clone() } \
                   } \
                   fn main() { let p = Person { label: String::from(\"Ada\") }; let _ = p.greeting(); }";
        let diags = check_src(src);
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }

    /// Companion: a default method calling *another default* sibling method (neither has been
    /// overridden) must also type-check -- `find_trait_method_sig` matches on name alone,
    /// regardless of whether the found method has a body.
    #[test]
    fn trait_default_method_calling_another_default_method_type_checks() {
        let src = "trait Greet { \
                       fn name(&self) -> String; \
                       fn shout(&self) -> String { self.greeting() } \
                       fn greeting(&self) -> String { self.name() } \
                   } \
                   struct Person { label: String } \
                   impl Greet for Person { \
                       fn name(&self) -> String { self.label.clone() } \
                   } \
                   fn main() { let p = Person { label: String::from(\"Ada\") }; let _ = p.shout(); }";
        let diags = check_src(src);
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }

    /// The DEV-051 fix must not silently swallow a genuine arity mismatch -- calling a sibling
    /// trait method with the wrong number of arguments from inside a default body must still
    /// raise `E0005`, proving `check_trait_member_call`'s argument check still runs on this path.
    #[test]
    fn trait_default_method_wrong_arg_count_to_sibling_trait_method_still_errors() {
        let src = "trait Greet { \
                       fn name(&self, suffix: String) -> String; \
                       fn greeting(&self) -> String { self.name() } \
                   } \
                   struct Person { label: String } \
                   impl Greet for Person { \
                       fn name(&self, suffix: String) -> String { self.label.clone() } \
                   } \
                   fn main() { let p = Person { label: String::from(\"Ada\") }; let _ = p.greeting(); }";
        let diags = check_src(src);
        assert!(
            diags.iter().any(|d| d.code.as_deref() == Some("E0005")),
            "expected E0005 for the missing argument, got: {diags:?}"
        );
    }

    /// DEV-060 [CLOSED]: calling the same un-overridden trait *default* method twice on the
    /// same receiver used to incorrectly report `E0100 use of moved value` on the second call,
    /// even though the method only takes `&self`. Root cause: `borrowck.rs`'s `method_receiver`
    /// (used by the `Call` handler to decide whether a method receiver is moved, borrowed, or
    /// mutably borrowed) only ever searched `ImplItem::Fn` overrides -- it had no equivalent to
    /// `typecheck.rs::resolve_method`'s `default_fallback` (WP-C1.3/DEV-013), so an
    /// un-overridden default method returned `None`, and the `None` arm unconditionally
    /// consumed (moved) the receiver via `check_expr`'s `Path` arm, regardless of the method's
    /// real receiver kind. Fixed by adding the matching trait-default-body fallback to
    /// `method_receiver` itself. Confirmed narrow both before and after the fix: two calls to
    /// an *overridden* trait method, or two calls to an ordinary inherent method, were always
    /// unaffected (`interp.rs`'s `repeated_call_to_overridden_trait_method_is_unaffected_by_
    /// dev060`/`::repeated_call_to_inherent_method_is_unaffected_by_dev060`).
    #[test]
    fn repeated_call_to_unoverridden_default_trait_method_is_no_longer_flagged_as_move() {
        let src = "trait Greet { \
                       fn name(&self) -> String; \
                       fn greeting(&self) -> String { self.name() } \
                   } \
                   struct Person { label: String } \
                   impl Greet for Person { \
                       fn name(&self) -> String { self.label.clone() } \
                   } \
                   fn main() { \
                       let p = Person { label: String::from(\"Ada\") }; \
                       println(p.greeting()); \
                       println(p.greeting()); \
                   }";
        let diags = check_src(src);
        assert!(
            diags.iter().all(|d| d.code.as_deref() != Some("E0100")),
            "DEV-060 regressed: unexpected 'use of moved value' on a repeated call to an \
             un-overridden trait default method: {diags:?}"
        );
    }

    /// DEV-060 companion: the same defect for a `&mut self` un-overridden trait default -- the
    /// fallback must propagate `RefMut`, not just `Ref`, so two calls correctly register two
    /// non-conflicting mutable borrows (sequential, not simultaneous) rather than a move.
    #[test]
    fn repeated_call_to_unoverridden_mut_default_trait_method_is_no_longer_flagged_as_move() {
        let src = "trait Counter { \
                       fn bump_inner(&mut self); \
                       fn bump(&mut self) { self.bump_inner(); } \
                   } \
                   struct Count { value: Int32 } \
                   impl Counter for Count { \
                       fn bump_inner(&mut self) { self.value = self.value + 1; } \
                   } \
                   fn main() { \
                       let mut c = Count { value: 0 }; \
                       c.bump(); \
                       c.bump(); \
                   }";
        let diags = check_src(src);
        assert!(
            diags.iter().all(|d| d.code.as_deref() != Some("E0100")),
            "DEV-060 (mut receiver variant) regressed: {diags:?}"
        );
    }

    /// TYPE-FN-001 (CD-027): function values do not participate in `Eq`/`Ord` — comparing them
    /// is a compile-time E0500, exactly like the float primitives. Pins the pre-existing
    /// rejection now that it is normative rather than incidental.
    #[test]
    fn fn_values_do_not_satisfy_eq_or_ord() {
        for op in ["==", "<"] {
            let src = format!(
                "fn double(x: Int32) -> Int32 {{ x * 2 }} \
                 fn triple(x: Int32) -> Int32 {{ x * 3 }} \
                 fn main() {{ \
                     let f: fn(Int32) -> Int32 = double; \
                     let g: fn(Int32) -> Int32 = triple; \
                     println(f {op} g); \
                 }}"
            );
            let diags = check_src(&src);
            assert!(
                diags.iter().any(|d| d.code.as_deref() == Some("E0500")),
                "fn-value `{op}` must be rejected with E0500 (TYPE-FN-001): {diags:?}"
            );
        }
    }

    /// DEV-062 [CLOSED]: function values are `Copy` (03-Type-System.md §Copy and Drop /
    /// TYPE-FN-001), so repeated use of a fn-typed local — including `f(f(x))`, CD-021 workload
    /// item 22's exact shape — must not raise E0100.
    #[test]
    fn fn_typed_local_is_copy_and_reusable() {
        let src = "fn double(x: Int32) -> Int32 { x * 2 } \
                   fn apply(f: fn(Int32) -> Int32, v: Int32) -> Int32 { f(v) } \
                   fn main() { \
                       let f: fn(Int32) -> Int32 = double; \
                       println(f(f(10))); \
                       println(apply(f, 7)); \
                       println(f(1)); \
                   }";
        let diags = check_src(src);
        assert!(
            diags.iter().all(|d| d.code.as_deref() != Some("E0100")),
            "DEV-062 regressed — fn-typed local wrongly moved: {diags:?}"
        );
    }

    /// DEV-052: a qualified call to a `CoreTrait` the receiver type does not actually implement
    /// must still be rejected -- confirms the fix doesn't accidentally accept a genuinely
    /// invalid program just because the qualified-call *syntax* now resolves.
    #[test]
    fn qualified_call_to_unimplemented_core_trait_is_rejected() {
        let src = "struct Point { x: Int32 } \
                   fn main() { \
                       let a = Point { x: 1 }; \
                       let b = Point { x: 1 }; \
                       let _ = Eq::eq(&a, &b); \
                   }";
        let diags = check_src(src);
        assert!(
            diags
                .iter()
                .any(|d| d.code.as_deref() == Some("E0500")
                    && d.message.contains("does not implement")),
            "expected an E0500 rejection for a Point with no impl Eq: {diags:?}"
        );
    }

    /// WP-C1.3: `Ty::Core` (Option/Result/Vec/Box) had no arm in `require_operator_bound` at
    /// all before this WP, so `==`/`<` on any of these normatively "essential" standard-library
    /// types (06-Standard-Library.md) was unconditionally rejected with E0500, even when their
    /// type arguments are obviously comparable primitives. Confirmed empirically via
    /// `starkc check` before writing this test (not merely inferred from source reading).
    #[test]
    fn option_result_vec_box_satisfy_eq_when_their_type_args_do() {
        for src in [
            "fn main() { let a: Option<Int32> = Some(1); let b: Option<Int32> = Some(1); let _c = a == b; }",
            "fn main() { let a: Result<Int32, String> = Ok(1); let b: Result<Int32, String> = Ok(1); let _c = a == b; }",
            "fn main() { let a: Vec<Int32> = Vec::new(); let b: Vec<Int32> = Vec::new(); let _c = a == b; }",
            // Nested: Option<Option<Int32>> should recurse correctly.
            "fn main() { let a: Option<Option<Int32>> = Some(Some(1)); let b: Option<Option<Int32>> = Some(Some(1)); let _c = a == b; }",
        ] {
            let diags = check_src(src);
            assert!(diags.is_empty(), "{src}: unexpected diagnostics {diags:?}");
        }
    }

    /// WP-C1.3: the recursive `Ty::Core` bound check must still correctly *reject* a container
    /// whose type argument does not itself satisfy Eq -- confirms the fix isn't overly
    /// permissive (e.g. accidentally treating every `Option<T>` as Eq regardless of `T`).
    #[test]
    fn option_of_non_eq_type_is_rejected() {
        let diags = check_src(
            "struct NoEq { x: Int32 } \
             fn main() { \
                 let a: Option<NoEq> = Some(NoEq { x: 1 }); \
                 let b: Option<NoEq> = Some(NoEq { x: 1 }); \
                 let _c = a == b; \
             }",
        );
        assert!(
            diags.iter().any(|d| d.code.as_deref() == Some("E0500")),
            "expected E0500 for Option<NoEq> == Option<NoEq>, got {:?}",
            diags
        );
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
        all.extend(check_with_options(&hir, opts));
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
        let mut type_diags = check(&hir);
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
    // WP-C6.1g-a: S must be a genuinely-Move type -- an all-Copy-field struct is now Copy
    // (OWN-COPY-001, amended), so a `String` field keeps this test's Move vehicle intact.
    fn ownership_checker_visits_loop_bodies() {
        let diags = check_src(
            "struct S { x: String } fn take(v: S) {} fn f(c: Bool) { let s = S { x: String::new() }; while c { take(s); take(s); } }",
        );
        assert!(diags.iter().any(|d| d.code.as_deref() == Some("E0100")));
    }

    #[test]
    fn partial_move_allows_sibling_but_not_whole_value() {
        let valid = check_src(
            "struct S { x: String } struct Pair { a: S, b: S } fn take(v: S) {} fn f() { let p = Pair { a: S { x: String::new() }, b: S { x: String::new() } }; take(p.a); take(p.b); }",
        );
        assert!(valid.is_empty(), "unexpected diagnostics: {valid:?}");

        let invalid = check_src(
            "struct S { x: String } struct Pair { a: S, b: S } fn take(v: S) {} fn take_pair(v: Pair) {} fn f() { let p = Pair { a: S { x: String::new() }, b: S { x: String::new() } }; take(p.a); take_pair(p); }",
        );
        assert!(invalid.iter().any(|d| d.code.as_deref() == Some("E0100")));
    }

    #[test]
    fn shorthand_struct_field_consumes_its_source() {
        let diags = check_src(
            "struct S { x: String } struct W { s: S } fn take(v: S) {} fn f() { let s = S { x: String::new() }; let w = W { s }; take(s); }",
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

    /// WP-C4.5c / DEV-064 (TYPE-FN-002): coercing a generic fn whose generic arguments the
    /// expected fn type does not determine must be rejected — `T` appears nowhere in
    /// `count`'s signature, so no instantiation is nameable.
    #[test]
    fn undetermined_generic_fn_coercion_is_rejected() {
        let diags = check_src(
            "fn count<T>() -> Int32 { 0 } fn main() { let f: fn() -> Int32 = count; f(); }",
        );
        assert!(
            diags.iter().any(|d| d.code.as_deref() == Some("E0004")),
            "expected E0004, got: {diags:?}"
        );
    }

    /// WP-C4.5c (TYPE-GENERIC-001): a direct call that leaves a generic parameter
    /// unconstrained requires explicit arguments; without turbofish it is rejected, with
    /// turbofish it is accepted.
    #[test]
    fn undetermined_generic_call_requires_turbofish() {
        let invalid = check_src("fn count<T>() -> Int32 { 0 } fn main() { count(); }");
        assert!(
            invalid.iter().any(|d| d.code.as_deref() == Some("E0004")),
            "expected E0004, got: {invalid:?}"
        );

        let valid = check_src("fn count<T>() -> Int32 { 0 } fn main() { count::<Bool>(); }");
        assert!(valid.is_empty(), "unexpected diagnostics: {valid:?}");
    }

    /// WP-C4.5c: a coercion the expected type fully determines stays accepted, and the
    /// grounded instantiation is published for monomorphisation.
    #[test]
    fn determined_generic_fn_coercion_publishes_instantiation() {
        let file = Arc::new(SourceFile::new(
            "test.stark",
            "fn id<T>(x: T) -> T { x } fn main() { let f: fn(Int32) -> Int32 = id; f(1); }"
                .to_string(),
        ));
        let (ast, parse_diags) = crate::parser::parse(&file, crate::parser::ParseMode::Program);
        assert!(parse_diags.is_empty());
        let (hir, resolve_diags) = crate::resolve::resolve(&ast, file.clone());
        assert!(resolve_diags.is_empty());
        let result = analyze(&hir);
        assert!(
            result.diagnostics.is_empty(),
            "unexpected diagnostics: {:?}",
            result.diagnostics
        );
        assert!(
            result
                .tables
                .callable_instantiations
                .values()
                .any(|env| env.own_arguments() == vec![Ty::Primitive(Primitive::Int32)]),
            "expected a published [Int32] instantiation, got: {:?}",
            result.tables.callable_instantiations
        );
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
    fn positive_bounds_do_not_make_unifying_impl_heads_disjoint() {
        let diagnostics = check_src(
            "trait A {} trait B {} trait Marker {} \
             struct Wrapper<T> { value: T } \
             impl<T: A> Marker for Wrapper<T> {} \
             impl<T: B> Marker for Wrapper<T> {}",
        );
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code.as_deref() == Some("E0500")
                && diagnostic.message.contains("overlapping implementation")
        }));
    }

    #[test]
    fn generic_reference_fields_propagate_borrows_without_becoming_illegal_fields() {
        let diagnostics = check_src(
            "struct Holder<T> { value: T } \
             fn hold(value: &Int32) -> Holder<&Int32> { Holder { value: value } }",
        );
        assert!(
            diagnostics.is_empty(),
            "unexpected diagnostics: {diagnostics:?}"
        );

        let escaping = check_src(
            "struct Holder<T> { value: T } \
             fn bad() -> Holder<&Int32> { let local = 1; Holder { value: &local } }",
        );
        assert!(escaping
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0103")));
    }

    #[test]
    fn constant_patterns_reject_nonprimitive_values() {
        let diagnostics = check_src(
            "struct Key { value: Int32 } \
             impl Eq for Key { fn eq(&self, other: &Key) -> Bool { self.value == other.value } } \
             const FIRST: Key = Key { value: 1 }; \
             fn classify(value: Key) -> Int32 { match value { FIRST => 1, _ => 0 } }",
        );
        assert!(diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0305")));
    }

    #[test]
    fn floating_exponent_operator_is_rejected() {
        let diagnostics =
            check_src("fn main() { let x: Float64 = 2.0; let y: Float64 = 3.0; let _z = x ** y; }");
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code.as_deref() == Some("E0001")
                && diagnostic.message.contains("integer primitive")
        }));
    }

    #[test]
    fn trait_associated_conversion_function_resolves() {
        let diagnostics = check_src(
            "struct Celsius { value: Int32 } \
             struct Fahrenheit { value: Int32 } \
             impl From<Celsius> for Fahrenheit { \
                 fn from(value: Celsius) -> Fahrenheit { \
                     Fahrenheit { value: value.value } \
                 } \
             } \
             fn main() { \
                 let c = Celsius { value: 10 }; \
                 let _f: Fahrenheit = Fahrenheit::from(c); \
             }",
        );
        assert!(
            diagnostics.is_empty(),
            "unexpected diagnostics: {diagnostics:?}"
        );
    }

    #[test]
    fn ambiguous_trait_associated_functions_require_qualification() {
        let diagnostics = check_src(
            "struct Value { raw: Int32 } \
             trait First { fn make() -> Value; } \
             trait Second { fn make() -> Value; } \
             impl First for Value { fn make() -> Value { Value { raw: 1 } } } \
             impl Second for Value { fn make() -> Value { Value { raw: 2 } } } \
             fn main() { let value = Value::make(); }",
        );
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code.as_deref() == Some("E0204") && diagnostic.message.contains("ambiguous")
        }));
    }

    #[test]
    fn public_api_rejects_private_signature_types() {
        let diagnostics = check_src(
            "struct Secret { value: Int32 } \
             pub fn reveal(value: Secret) -> Secret { value }",
        );
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code.as_deref() == Some("E0209") && diagnostic.message.contains("Secret")
        }));
    }

    #[test]
    fn public_api_accepts_publicly_nameable_signature_types() {
        let diagnostics = check_src(
            "pub struct PublicValue { pub value: Int32 } \
             pub fn identity(value: PublicValue) -> PublicValue { value }",
        );
        assert!(
            diagnostics.is_empty(),
            "unexpected diagnostics: {diagnostics:?}"
        );
    }

    #[test]
    fn public_api_accepts_a_type_made_nameable_by_public_reexport() {
        let diags = check_src(
            "mod hidden { pub struct Token { pub value: Int32 } } \
             pub use hidden::Token; \
             pub fn make() -> Token { Token { value: 1 } }",
        );
        assert!(
            diags
                .iter()
                .all(|diagnostic| diagnostic.code.as_deref() != Some("E0209")),
            "{diags:?}"
        );
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
