//! Type checking, mutability, and definite assignment validation pass for STARK (PLAN.md M2.2).

// AS6 packet 4B group 2C: the tensor semantic authority lives in
// `extensions::tensor::check`. What remains here is the integration boundary — locating an
// operation in `TENSOR_OPS`, validating the call form, evaluating arguments, and publishing the
// type the extension decided — plus the `TensorCheckCtx` impl that names, exhaustively, the Core
// services the extension is allowed to use.
use crate::ast::{AssignOp, BinOp, Lit, Primitive, UnOp};
use crate::diag::Diagnostic;
use crate::extensions::tensor::check as tensor_check;
use crate::extensions::tensor::check::TensorCheckCtx;
use crate::extensions::tensor::dim::{DimVar, Poly};
use crate::extensions::tensor::rules::TENSOR_OPS;
use crate::extensions::tensor::syntax as tensor_syntax;
use crate::extensions::tensor::types::{DType, Device, DeviceVar, Shape, UnifyCtx};
use crate::hir::{
    self, BlockId, Builtin, CoreType, ExprId, Hir, ItemId, LocalId, PatId, Res, StmtId, TypeId,
};
use crate::literal;
use crate::options::LanguageOptions;
use crate::source::Span;
use std::collections::{HashMap, HashSet};

mod bounds;
/// WP-C6.2b-F1: a selected inherent/trait method candidate carried through visibility enforcement:
/// (signature def, is-trait-method, impl substitution, impl self type, member is `pub`, impl item).
mod convert;
mod infer;
mod patterns;
mod state;
mod traits;
use state::{single_segment_name, PublishedEnv, SelfScope, TypeChecker};
use traits::is_copy_with_impls;
pub use traits::{copy_eligible_types, nominals_with_destructor};
use traits::{core_trait_contract, core_trait_source_name, ContractTy, CoreTraitMethod};

mod types;
pub use types::{
    collect_ty_params, substitute_ty, ty_contains_param, CallableDeclId, CallableInstantiation,
    CallableSigTy, CallableUse, CallableUseId, CalleeSelection, DispatchProvenance, DisplayPath,
    DisplayStep, ExtensionTy, GenericBinder, GenericEnvironment, LayoutTables, ModelTy,
    ReceiverAdjustment, ReceiverBinding, Ty, TypeCheckResult, TypeTables, TypeVarId,
};
use types::{
    BindMode, ControlSummary, DeferredDisplayPlan, FnSigTy, GenericKind, LoopContext,
    VariantFields, VariantTy,
};

// Crate-internal pure helpers of the representation, used by the passes still in `mod.rs`.
use types::{
    convert_float_suffix, convert_int_suffix, is_float_primitive, is_integer, is_numeric,
    standard_display_type, strip_ref, ty_contains_error, ty_contains_infer, unit_or_tuple,
};

type MethodCandidate<'a> = (&'a hir::FnDef, bool, HashMap<String, Ty>, Ty, bool, ItemId);

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

/// A deferred callable-use publication: the call expression, the body it selected, and the generic
/// environment — held until the instantiated signature exists (AS3 Boundary 2).
type PendingUse = (ExprId, BlockId, Vec<(GenericBinder, Ty)>);

// AS3 Boundary 4 uses `hir::BoundTrait` — `User(ItemId)` or `Core(CoreTrait)` — which the compiler
// already carries because user traits and compiler-known traits both occur as bounds.
//
// This resolves a model bug found by the Display characterization: `DispatchProvenance::Bound
// { trait_item: ItemId }` could not represent `T: Display`, since `Display` is a `CoreTrait` with
// no trait `ItemId`. Selection and provenance now speak the same identity language, and it is the
// language the rest of the compiler already speaks.

/// The receiver type a declaration BINDS, formed the way A3b forms it.
///
/// AS3 Boundary 2 hardening, second correction. The publication recorded the instantiated `Self`
/// bare, while `callable_types` records the receiver **as the body binds it** — `&Self` for
/// `&self`, `&mut Self` for `&mut self`. So the two disagreed on every borrowing method, and §3.4's
/// invariant would have failed on all of them. It did, the moment the test stopped skipping
/// generics.
///
/// This mirrors `check_fn_def`'s construction rather than re-deriving it, so the two cannot drift.
fn bound_receiver_ty(receiver: Option<&hir::Receiver>, self_ty: Ty) -> Option<Ty> {
    match receiver? {
        hir::Receiver::Value => Some(self_ty),
        hir::Receiver::Ref => Some(Ty::Ref {
            mutable: false,
            inner: Box::new(self_ty),
        }),
        hir::Receiver::RefMut => Some(Ty::Ref {
            mutable: true,
            inner: Box::new(self_ty),
        }),
    }
}

/// What the CALL SITE did to the receiver, from TYPE-METHOD-002's peel count and the form the
/// selected declaration binds.
///
/// AS3 Boundary 2 hardening. Every named-dispatch publication passed `ReceiverAdjustment::None`,
/// so the field existed and published nothing — a consumer would have received
/// `binding = Shared, adjustment = None` and still had to work out the receiver semantics itself.
///
/// `derefs` is how many leading `&`/`&mut` method resolution removed before matching. A receiver
/// of `&&mut T` calling a `&self` method is two derefs and a shared adjustment; `T` calling `self`
/// is zero derefs by value.
fn receiver_adjustment_for(
    derefs: u8,
    outermost_ref_is_mut: bool,
    binding: ReceiverBinding,
) -> ReceiverAdjustment {
    match binding {
        ReceiverBinding::None => ReceiverAdjustment::None,
        ReceiverBinding::ByValue => ReceiverAdjustment::ByValue,
        ReceiverBinding::Shared => ReceiverAdjustment::Shared { derefs },
        ReceiverBinding::Exclusive => {
            // An exclusive binding reached through a shared reference would be a borrow error the
            // checker has already rejected; recording the outermost form keeps the two answers
            // consistent rather than asserting one from the other.
            let _ = outermost_ref_is_mut;
            ReceiverAdjustment::Exclusive { derefs }
        }
    }
}

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
    fn source_package<'s>(&self, name: &'s str) -> Option<&'s str> {
        if std::path::Path::new(name).is_absolute() {
            return None;
        }
        name.split_once('/').map(|(package, _)| package)
    }

    /// WP-C1.5: minimal constant evaluator for array-repeat-expression counts (`[value; count]`,
    /// 02-Syntax-Grammar.md:330). Handles the two confirmed-common shapes -- a literal, or a
    /// reference to a `const` item (recursing into its initializer) -- rather than a full
    /// general constant-folding pass, which is out of this WP's scope.
    fn const_eval_u64(&self, expr_id: ExprId) -> Option<u64> {
        self.const_eval_i128(expr_id, &mut HashSet::new())
            .and_then(|value| u64::try_from(value).ok())
    }

    fn const_eval_i128(&self, expr_id: ExprId, visiting: &mut HashSet<ItemId>) -> Option<i128> {
        let expr = self.hir.expr(expr_id);
        match &expr.kind {
            hir::ExprKind::Lit(Lit::Int { base, suffix }) => {
                literal::parse_int_literal(self.text(expr.span), *base, *suffix)
            }
            hir::ExprKind::Path {
                res: Res::Item(item_id),
                ..
            } => match &self.hir.item(*item_id).kind {
                hir::ItemKind::Const { value, .. } => {
                    if !visiting.insert(*item_id) {
                        return None;
                    }
                    let result = self.const_eval_i128(*value, visiting);
                    visiting.remove(item_id);
                    result
                }
                _ => None,
            },
            hir::ExprKind::Unary { op, operand } => {
                let value = self.const_eval_i128(*operand, visiting)?;
                match op {
                    UnOp::Neg => value.checked_neg(),
                    UnOp::BitNot => Some(!value),
                    _ => None,
                }
            }
            hir::ExprKind::Binary { op, lhs, rhs } => {
                let lhs = self.const_eval_i128(*lhs, visiting)?;
                let rhs = self.const_eval_i128(*rhs, visiting)?;
                match op {
                    BinOp::Add => lhs.checked_add(rhs),
                    BinOp::Sub => lhs.checked_sub(rhs),
                    BinOp::Mul => lhs.checked_mul(rhs),
                    BinOp::Div => lhs.checked_div(rhs),
                    BinOp::Rem => lhs.checked_rem(rhs),
                    BinOp::Pow => u32::try_from(rhs).ok().and_then(|rhs| lhs.checked_pow(rhs)),
                    BinOp::BitAnd => Some(lhs & rhs),
                    BinOp::BitOr => Some(lhs | rhs),
                    BinOp::BitXor => Some(lhs ^ rhs),
                    BinOp::Shl => u32::try_from(rhs).ok().and_then(|rhs| lhs.checked_shl(rhs)),
                    BinOp::Shr => u32::try_from(rhs).ok().and_then(|rhs| lhs.checked_shr(rhs)),
                    _ => None,
                }
            }
            hir::ExprKind::Cast { expr, .. } => self.const_eval_i128(*expr, visiting),
            hir::ExprKind::Block(block) => {
                let block = self.hir.block(*block);
                if block.stmts.iter().any(|statement| {
                    !matches!(
                        &self.hir.stmt(*statement).kind,
                        hir::StmtKind::Empty | hir::StmtKind::Expr { .. }
                    )
                }) {
                    return None;
                }
                for statement in &block.stmts {
                    if let hir::StmtKind::Expr { expr, .. } = &self.hir.stmt(*statement).kind {
                        self.const_eval_i128(*expr, visiting)?;
                    }
                }
                block
                    .tail
                    .and_then(|tail| self.const_eval_i128(tail, visiting))
            }
            _ => None,
        }
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
            Builtin::AssertEq | Builtin::AssertNe => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![value.clone(), value],
                    ret: Box::new(unit),
                }
            }
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
            Builtin::CharFromU32 => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::UInt32)],
                ret: Box::new(Ty::Core(
                    CoreType::Option,
                    vec![Ty::Primitive(Primitive::Char)],
                )),
            },
            Builtin::VecNew => Ty::Fn {
                params: Vec::new(),
                ret: Box::new(Ty::Core(CoreType::Vec, vec![self.new_type_var()])),
            },
            Builtin::VecWithCapacity => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::UInt64)],
                ret: Box::new(Ty::Core(CoreType::Vec, vec![self.new_type_var()])),
            },
            Builtin::HashMapNew => {
                let key = self.new_type_var();
                let val = self.new_type_var();
                Ty::Fn {
                    params: Vec::new(),
                    ret: Box::new(Ty::Core(CoreType::HashMap, vec![key, val])),
                }
            }
            Builtin::HashMapWithCapacity => {
                let key = self.new_type_var();
                let val = self.new_type_var();
                Ty::Fn {
                    params: vec![Ty::Primitive(Primitive::UInt64)],
                    ret: Box::new(Ty::Core(CoreType::HashMap, vec![key, val])),
                }
            }
            Builtin::HashSetNew => {
                let val = self.new_type_var();
                Ty::Fn {
                    params: Vec::new(),
                    ret: Box::new(Ty::Core(CoreType::HashSet, vec![val])),
                }
            }
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
                        Ty::Core(CoreType::IOError, Vec::new()),
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
                        Ty::Core(CoreType::IOError, Vec::new()),
                    ],
                )),
            },
            Builtin::FileOpen | Builtin::FileCreate => Ty::Fn {
                params: vec![Ty::Ref {
                    mutable: false,
                    inner: Box::new(Ty::Primitive(Primitive::Str)),
                }],
                ret: Box::new(Ty::Core(
                    CoreType::Result,
                    vec![
                        Ty::Core(CoreType::File, Vec::new()),
                        Ty::Core(CoreType::IOError, Vec::new()),
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
            // AS6: one arm, not thirty-three patterns for one behaviour. Every tensor
            // operation's *signature* is refined by the extension's own rules
            // (`check_tensor_op`); Core only needs to know a call is a call.
            Builtin::Tensor(_) => Ty::Fn {
                params: vec![],
                ret: Box::new(self.new_type_var()),
            },
            Builtin::SizeOf | Builtin::AlignOf => Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Primitive(Primitive::UInt64)),
            },
            Builtin::Swap => {
                let value = self.new_type_var();
                let ref_ty = Ty::Ref {
                    mutable: true,
                    inner: Box::new(value),
                };
                Ty::Fn {
                    params: vec![ref_ty.clone(), ref_ty],
                    ret: Box::new(unit),
                }
            }
            Builtin::Replace => {
                let value = self.new_type_var();
                let ref_ty = Ty::Ref {
                    mutable: true,
                    inner: Box::new(value.clone()),
                };
                Ty::Fn {
                    params: vec![ref_ty, value.clone()],
                    ret: Box::new(value),
                }
            }
            Builtin::Take => {
                let value = self.new_type_var();
                let ref_ty = Ty::Ref {
                    mutable: true,
                    inner: Box::new(value.clone()),
                };
                Ty::Fn {
                    params: vec![ref_ty],
                    ret: Box::new(value),
                }
            }
            // -- Phase 4E: Math constants and functions --
            Builtin::MathPi | Builtin::MathE => Ty::Primitive(Primitive::Float64),
            Builtin::MathAbs => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![value.clone()],
                    ret: Box::new(value),
                }
            }
            Builtin::MathMin | Builtin::MathMax => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![value.clone(), value.clone()],
                    ret: Box::new(value),
                }
            }
            Builtin::MathClamp => {
                let value = self.new_type_var();
                Ty::Fn {
                    params: vec![value.clone(), value.clone(), value.clone()],
                    ret: Box::new(value),
                }
            }
            Builtin::Pow | Builtin::Atan2 => Ty::Fn {
                params: vec![
                    Ty::Primitive(Primitive::Float64),
                    Ty::Primitive(Primitive::Float64),
                ],
                ret: Box::new(Ty::Primitive(Primitive::Float64)),
            },
            Builtin::Log
            | Builtin::Log10
            | Builtin::Exp
            | Builtin::Sin
            | Builtin::Cos
            | Builtin::Tan
            | Builtin::Asin
            | Builtin::Acos
            | Builtin::Atan
            | Builtin::Floor
            | Builtin::Ceil
            | Builtin::Round
            | Builtin::Trunc => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::Float64)],
                ret: Box::new(Ty::Primitive(Primitive::Float64)),
            },
            // -- Phase 4E: stderr --
            //
            // DEV-174: typed as a fresh variable, exactly like `print`/`println`.
            //
            // 06-Standard-Library declares `fn eprint<T: Display>(value: T)` and the
            // `eprintln`/`eprint` analogues, and PRINT-DISPLAY-001 covers all four by name. This
            // took `&str` instead, so `eprintln(s)` with an owned `String` — let alone any other
            // `Display` type — was rejected while `println(s)` was accepted. The stderr half of the
            // runtime surface has carried the full display family since 0.1-A13
            // (`EprintlnInt64`, `EprintBool`, …); only the signature lagged.
            Builtin::Eprint | Builtin::Eprintln => Ty::Fn {
                params: vec![self.new_type_var()],
                ret: Box::new(unit),
            },
            // -- Phase 4E: Random (simple LCG per `06-Standard-Library.md`) --
            Builtin::RandomNew => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::UInt64)],
                ret: Box::new(Ty::Core(CoreType::Random, Vec::new())),
            },
            // WP-C2.2 (DEV-027): Ordering's unit variants.
            Builtin::OrderingLess | Builtin::OrderingEqual | Builtin::OrderingGreater => {
                Ty::Core(CoreType::Ordering, Vec::new())
            }
            // -- Phase 4E: IOError variant constructors --
            Builtin::IOErrorNotFound
            | Builtin::IOErrorPermissionDenied
            | Builtin::IOErrorAlreadyExists
            | Builtin::IOErrorInvalidInput => Ty::Core(CoreType::IOError, Vec::new()),
            Builtin::IOErrorOther => Ty::Fn {
                params: vec![Ty::Primitive(Primitive::String)],
                ret: Box::new(Ty::Core(CoreType::IOError, Vec::new())),
            },
        }
    }

    fn is_unsized_value_type(&self, ty: &Ty) -> bool {
        matches!(
            self.resolve(ty),
            Ty::Slice(_) | Ty::Primitive(Primitive::Str)
        )
    }

    fn check_field_initializers(
        &mut self,
        owner: Option<ItemId>,
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
                // WP-C6.2b-F1: constructing with a private field is inaccessible outside its module.
                if let Some(struct_id) = owner {
                    let is_pub = self.struct_field_is_pub(struct_id, &name);
                    self.check_member_visible(is_pub, struct_id, "field", &name, field.name);
                }
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

    fn check_crate(&mut self) {
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

        // WP-C4.7-9 audit: `print`/`println` require a `Display`-able argument.
        let display = std::mem::take(&mut self.display_checks);
        for (ty, span) in display {
            let resolved = self.resolve(&ty);
            if matches!(resolved, Ty::Error) || ty_contains_infer(&resolved) {
                continue; // already failed, or undetermined — no cascade
            }
            if !self.type_is_displayable(&resolved) {
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

    fn check_public_api_reachability(&mut self) {
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

    fn check_type_well_formedness(&mut self) {
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

    fn private_type_in(&self, ty: hir::TypeId) -> Option<ItemId> {
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
    fn item_name(&self, item: ItemId) -> String {
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

    /// Record each field's DECLARED type, instantiated for this literal's type arguments.
    ///
    /// AS3 Packet 5. The `AggregateField` boundary needs the type the *nominal declares* for the
    /// field, not the type of the expression that produced the value — the second would compare a
    /// value against its own producer and assert nothing. `expected` is the nominal's parametric
    /// field map and `map` is the substitution the literal's arguments determine, which is exactly
    /// what `check_field_initializers` unifies against; publishing here rather than re-deriving
    /// keeps the boundary reading the same answer the checker enforced.
    ///
    /// Shorthand initialisers (`W { v }`) are covered for free: the key is the FIELD NAME, so a
    /// field with no initialiser expression still has a published type.
    fn publish_aggregate_field_types(
        &mut self,
        expr_id: ExprId,
        expected: &HashMap<String, Ty>,
        map: &HashMap<String, Ty>,
    ) {
        let concrete = expected
            .iter()
            .map(|(name, ty)| (name.clone(), self.instantiate_ty(ty, map)))
            .collect();
        self.aggregate_field_types.insert(expr_id, concrete);
    }

    /// Build the ordered binder list for one callable use.
    ///
    /// AS3 extracted this from `publish_callable_env` so the instantiation table and the
    /// `CallableUse` record are built by the SAME code. Two constructions of "what generic
    /// environment did this use select" is the shape of defect this packet exists to remove, and
    /// duplicating it here to publish a second table would have been an immediate instance.
    fn env_bindings(
        self_ty: &Option<Ty>,
        impl_names: &[String],
        own_names: &[String],
        own_is_method: bool,
        map: &HashMap<String, Ty>,
    ) -> Vec<(GenericBinder, Ty)> {
        let mut bindings: Vec<(GenericBinder, Ty)> = Vec::new();
        if let Some(self_ty) = self_ty {
            // **`Self` is substituted through the WHOLE map here, not only through the binders
            // named below.** A trait default invoked on a generic impl publishes
            // `Self = Tagged<T>` while the environment carries the TRAIT's generics — which for
            // `trait Describe` are none — so `T` would have nothing to resolve against and the
            // install would refuse a correct program. `map` holds every binding the checker
            // selected, including the impl's, so resolving here uses all of them regardless of
            // which are individually named as binders.
            bindings.push((GenericBinder::SelfType, substitute_ty(self_ty, map)));
        }
        for (index, name) in impl_names.iter().enumerate() {
            if let Some(ty) = map.get(name) {
                bindings.push((
                    GenericBinder::ImplParam {
                        index,
                        name: name.clone(),
                    },
                    ty.clone(),
                ));
            }
        }
        for (index, name) in own_names.iter().enumerate() {
            if let Some(ty) = map.get(name) {
                let binder = if own_is_method {
                    GenericBinder::MethodParam {
                        index,
                        name: name.clone(),
                    }
                } else {
                    GenericBinder::FunctionParam {
                        index,
                        name: name.clone(),
                    }
                };
                bindings.push((binder, ty.clone()));
            }
        }
        bindings
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
    fn check_generic_shadowing(
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

    fn check_fn_def(&mut self, _item_id: ItemId, def: &hir::FnDef) {
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

    /// WP-FMT-001: check one interpolation field.
    ///
    /// The specification decides what the type must be. A padding-only field (or none at all) asks
    /// for `Display` and nothing more; a numeric mode asks for a concrete integer or float. Every
    /// rejection happens HERE, at type checking — §6.7's requirement that no bad type/spec pairing
    /// reaches run time.
    fn check_format_field(&mut self, expr: ExprId, spec: &crate::ast::FormatSpec, expr_span: Span) {
        use crate::ast::FormatKind;
        let ty = self.check_expr(expr);
        let ty = self.default_int_literals_deep(&ty);
        if matches!(ty, Ty::Error) {
            return;
        }
        let spec_span = spec.span.unwrap_or(expr_span);
        // **DEV-206: do not strip the reference that MAKES the value.**
        //
        // Stripping is right for `fn render<T: Display>(v: &T)` — `Display::fmt` borrows anyway
        // (STD-FORMAT-001), so a reference to a displayable type is displayable. It is wrong for
        // `&[T]`: the pointee is UNSIZED, the reference is not incidental, and stripping it turns
        // the one displayable spelling into the one that is not a value at all.
        //
        // Found by the value-context property, which required every context to accept the
        // reference form and caught interpolation still rejecting `&[Int32]` after `println`
        // had been repaired.
        let stripped = match &ty {
            Ty::Ref { inner, .. } if !type_is_sized(inner) => ty.clone(),
            other => strip_ref(other).clone(),
        };

        // A numeric mode requires a numeric type. `Display` does NOT imply integer formatting
        // (§11.5), so a generic `T: Display` is refused here rather than given a meaning it has
        // not proved — inventing a numeric bound to make it compile is out of scope.
        //
        // The guards carry the type requirement, and the final arm enumerates every `FormatKind`
        // explicitly rather than using `_`: a new format type must force a decision here about
        // which types accept it.
        match spec.kind {
            Some(
                FormatKind::Bin | FormatKind::Oct | FormatKind::LowerHex | FormatKind::UpperHex,
            ) if !matches!(&stripped, Ty::Primitive(p) if is_integer(*p)) => {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "type '{}' cannot be formatted in another base",
                            self.ty_to_string(&ty)
                        ),
                        spec_span,
                    )
                    .with_code("E0306")
                    .with_label("'b', 'o', 'x' and 'X' require an integer type"),
                );
                return;
            }
            Some(FormatKind::Fixed) if !matches!(&stripped, Ty::Primitive(p) if is_float_primitive(*p)) =>
            {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "type '{}' cannot be formatted with fixed precision",
                            self.ty_to_string(&ty)
                        ),
                        spec_span,
                    )
                    .with_code("E0306")
                    .with_label("'f' requires 'Float32' or 'Float64'"),
                );
                return;
            }
            Some(
                FormatKind::Bin
                | FormatKind::Oct
                | FormatKind::LowerHex
                | FormatKind::UpperHex
                | FormatKind::Fixed,
            )
            | None => {}
        }

        if spec.precision.is_some() && spec.kind.is_none() {
            // A bare `.N` is fixed-point on a float. On a string it would have to mean truncation,
            // which WP-FMT-001 deliberately does not define (§7): cutting Unicode text needs a
            // scalar/grapheme/byte ruling nobody has made.
            if !matches!(&stripped, Ty::Primitive(p) if is_float_primitive(*p)) {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "type '{}' cannot be formatted with a precision",
                            self.ty_to_string(&ty)
                        ),
                        spec_span,
                    )
                    .with_code("E0306")
                    .with_label("precision applies to 'Float32' and 'Float64'")
                    .with_note(
                        "string truncation is not a format specification in Core v1; slice the \
                         value explicitly if you need a shorter one"
                            .to_string(),
                    ),
                );
                return;
            }
        }

        if (spec.sign.is_some() || spec.alternate || spec.zero_pad)
            && spec.kind.is_none()
            && spec.precision.is_none()
            && !matches!(&stripped, Ty::Primitive(p) if is_numeric(*p))
        {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "type '{}' does not accept a numeric format specification",
                        self.ty_to_string(&ty)
                    ),
                    spec_span,
                )
                .with_code("E0306")
                .with_label("sign, '#' and zero-padding require a numeric type"),
            );
            return;
        }

        // Everything reaching here renders through `Display`. This is the SAME predicate
        // `print`/`println` use, so it routes through bound identity (CD-379) and a user trait
        // merely NAMED `Display` cannot satisfy it.
        //
        // The check is on the STRIPPED type. `println` takes its argument by value and so never
        // sees a reference, but a field routinely does — `fn render<T: Display>(v: &T)` formats a
        // `&T`, and `Display::fmt` borrows anyway (STD-FORMAT-001), so a reference to a
        // displayable type is displayable.
        // A generic parameter is displayable only if one of ITS OWN bounds supplies `fmt`.
        // `type_is_displayable` answers `true` for any `Ty::Param` — correct for `println`, whose
        // caller discharges the bound at the call site, and wrong here: an interpolation inside
        // `fn render<T>(v: &T)` has no such caller obligation, and must be refused where it is
        // written. The check goes through `bound_method_candidates`, so it is CD-379's identity
        // path — a user trait merely NAMED `Display` does not satisfy it.
        if let Ty::Param(param_name) = &stripped {
            let param_name = param_name.clone();
            // Queued before the guard: a parameter WITH the bound is a real late-bound render
            // position, and this branch returns early.
            self.record_display_plan(expr, stripped.clone());
            if self.bound_method_candidates(&param_name, "fmt").is_empty() {
                self.diags.push(
                    Diagnostic::error(
                        format!("'{param_name}' has no bound that provides 'Display'"),
                        expr_span,
                    )
                    .with_code("E0306")
                    .with_label(format!("add the bound '{param_name}: Display'")),
                );
            }
            return;
        }
        // **AS3 Boundary 4: interpolation is the SECOND `Display` entry point**, and it renders the
        // same way — `"{w}"` on a `W<A>` runs `W`'s own `fmt` and stops, exactly as `println(w)`
        // does. So it queues the same walk rather than getting a dispatch mechanism of its own,
        // which is what left `find_impl_fn(nominal, "fmt", ..)` serving two callers.
        self.record_display_plan(expr, stripped.clone());
        if !self.type_is_displayable(&stripped) {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "type '{}' does not implement 'Display' and cannot be interpolated",
                        self.ty_to_string(&ty)
                    ),
                    expr_span,
                )
                .with_code("E0306")
                .with_label("write an 'impl Display for ...' for this type"),
            );
        }
    }

    fn check_expr(&mut self, expr_id: ExprId) -> Ty {
        let expr = self.hir.expr(expr_id);
        let ty = match &expr.kind {
            // WP-FMT-001: every field is checked, in source order, and the whole literal is a
            // `String`. Checking in order matters for diagnostics: the first bad field is reported
            // first, which is where a reader looks.
            hir::ExprKind::FormatString { segments } => {
                let fields: Vec<(ExprId, crate::ast::FormatSpec, Span)> = segments
                    .iter()
                    .filter_map(|segment| match segment {
                        hir::FormatSegment::Field {
                            expr,
                            spec,
                            expr_span,
                            ..
                        } => Some((*expr, *spec, *expr_span)),
                        hir::FormatSegment::Literal { .. } => None,
                    })
                    .collect();
                for (expr, spec, expr_span) in fields {
                    self.check_format_field(expr, &spec, expr_span);
                }
                Ty::Primitive(Primitive::String)
            }
            hir::ExprKind::Lit(lit) => match lit {
                // WP-C1.5 (DEV-015): no stage previously checked a literal's magnitude against
                // its suffix's (or, for unsuffixed literals, its default-inferred) representable
                // range -- `let x: UInt8 = 300u8;` compiled clean, and `let x = 99999999999;`
                // silently became a broken Int32 instead of the spec's "Int32 if it fits, else
                // Int64" (03-Type-System.md:28). Checked here, at typecheck time, since an
                // unsuffixed literal's fit-check depends on the type it's being inferred into
                // (Int32 vs Int64) -- the lexer sees only token shape, never a target type.
                Lit::Int { base, suffix } => {
                    let value = literal::parse_int_literal(self.text(expr.span), *base, *suffix);
                    if let Some(s) = suffix {
                        if let Some(value) = value {
                            if !literal::int_suffix_range_contains(*s, value) {
                                self.diags.push(
                                    Diagnostic::error(
                                        format!(
                                            "integer literal out of range for '{}'",
                                            self.ty_to_string(&Ty::Primitive(convert_int_suffix(
                                                *s
                                            )))
                                        ),
                                        expr.span,
                                    )
                                    .with_code("E0008"),
                                );
                            }
                        }
                        Ty::Primitive(convert_int_suffix(*s))
                    } else {
                        // WP-C4.7-6.3: an UNSUFFIXED literal takes a fresh integer-kinded
                        // inference variable instead of committing to `Int32` here. Expected
                        // types flow inward from annotations, parameters, fields and assignment
                        // destinations (03-Type-System), and only a literal still unconstrained
                        // after that is defaulted — step 5, applied in
                        // `default_unconstrained_int_literals`. Committing at this point was the
                        // whole defect: it made `takes_u64(0)` "expected 'UInt64', found 'Int32'".
                        match value {
                            Some(value) if i64::try_from(value).is_ok() => {
                                let var = self.new_type_var();
                                if let Ty::Infer(id) = var {
                                    self.int_literal_vars.insert(id, (value, expr.span));
                                }
                                var
                            }
                            Some(_) => {
                                // Beyond `Int64`'s range there is no representable type to adopt,
                                // so this is an error here rather than at binding time.
                                self.diags.push(
                                    Diagnostic::error(
                                        "integer literal out of range for 'Int64'",
                                        expr.span,
                                    )
                                    .with_code("E0008"),
                                );
                                Ty::Primitive(Primitive::Int64)
                            }
                            None => Ty::Primitive(Primitive::Int32),
                        }
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
                            Some(expr_id),
                            expr.span,
                        );
                        Ty::Fn {
                            params: instantiated_sig.params,
                            ret: Box::new(instantiated_sig.ret),
                        }
                    } else if let Some(const_ty) = self.const_types.get(item_id) {
                        // DEV-088 (WP-C4.7 close-out §7): USING a `const` declared in a different
                        // file is not yet supported and is rejected HERE, deterministically,
                        // before either engine runs. The oracle would evaluate the initializer's
                        // literal against the USE site's file (wrong text → "invalid literal" at
                        // runtime) while MIR does not lower a const in value position at all; a
                        // static rejection forecloses that inconsistency. Same-file `const` use is
                        // unaffected. Ownership-transferring cross-file constant use is deferred to
                        // the front-end/multi-file completion package (recorded in
                        // KNOWN-DEVIATIONS.md alongside DEV-083).
                        // AS1b-ii-d: identity, not name equality against an ambient file.
                        let cross_file = self
                            .hir
                            .item_sources
                            .get(item_id)
                            .is_some_and(|declaring| *declaring != expr.span.source);
                        if cross_file {
                            self.diags.push(
                                Diagnostic::error(
                                    "using a `const` declared in another file is not yet supported",
                                    expr.span,
                                )
                                .with_code("E0215")
                                .with_label(
                                    "move the constant into this file, or inline its value, until \
                                     cross-file constant use is implemented",
                                ),
                            );
                        }
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
                    self.associated_fn_type(*item_id, *name, turbofish.as_ref(), expr.span, expr_id)
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
                Res::Builtin(builtin) => {
                    if *builtin == Builtin::SizeOf || *builtin == Builtin::AlignOf {
                        self.validate_generic_arity(
                            1,
                            turbofish.as_ref().map_or(0, |args| args.args.len()),
                            expr.span,
                        );
                        if let Some(ref args) = turbofish {
                            for arg in &args.args {
                                if let hir::GenericArg::Type(type_id) = arg {
                                    // WP-C5.3e: the resolved type's CONTRACT LAYOUT is recorded
                                    // now. It was previously computed and discarded, which is
                                    // why the HIR oracle had no way to answer per type. A type
                                    // the contract does not describe records nothing, and every
                                    // engine then refuses the query rather than inventing a
                                    // number.
                                    // WP-C5.3e: the FULL conversion, not
                                    // `type_from_hir_without_diagnostics` -- that helper handles
                                    // only primitives, bare nominals and references, dropping
                                    // generic arguments and mapping tuples/arrays to `Ty::Error`.
                                    // It was adequate when the result was discarded; a layout
                                    // answer needs the real type.
                                    let ty = self.convert_hir_type(*type_id);
                                    let ty = self.ground(&ty);
                                    self.layout_queries.insert(expr_id, ty);
                                }
                            }
                        }
                    }
                    self.builtin_type(*builtin)
                }
                Res::TraitMember(_, _) => Ty::Error,
                Res::CoreTraitMember(_, _) => Ty::Error,
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
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                        let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        self.require_operator_bound(&lhs_ty, "Num", expr.span);
                        lhs_ty
                    }
                    BinOp::Pow => {
                        let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        match self.resolve(&lhs_ty) {
                            Ty::Primitive(p) if is_integer(p) => {}
                            Ty::Infer(_) | Ty::Error => {}
                            _ => self.diags.push(
                                Diagnostic::error(
                                    "`**` is defined only for integer primitive types",
                                    expr.span,
                                )
                                .with_code("E0001")
                                .with_note(
                                    "use `std::math::pow` for floating-point exponentiation",
                                ),
                            ),
                        }
                        lhs_ty
                    }
                    BinOp::Eq | BinOp::Ne => {
                        if !self.string_types_comparable(&lhs_ty, &rhs_ty) {
                            let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        }
                        self.require_operator_bound(&lhs_ty, "Eq", expr.span);
                        self.publish_operator_use(expr_id, &lhs_ty, "Eq", "eq", hir::CoreTrait::Eq);
                        Ty::Primitive(Primitive::Bool)
                    }
                    BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        if !self.string_types_comparable(&lhs_ty, &rhs_ty) {
                            let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        }
                        self.require_operator_bound(&lhs_ty, "Ord", expr.span);
                        self.publish_operator_use(
                            expr_id,
                            &lhs_ty,
                            "Ord",
                            "cmp",
                            hir::CoreTrait::Ord,
                        );
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
                    AssignOp::PowAssign => {
                        let _ = self.unify(lhs_ty.clone(), rhs_ty, expr.span);
                        match self.resolve(&lhs_ty) {
                            Ty::Primitive(p) if is_integer(p) => {}
                            Ty::Infer(_) | Ty::Error => {}
                            _ => self.diags.push(
                                Diagnostic::error(
                                    "`**=` is defined only for integer primitive types",
                                    expr.span,
                                )
                                .with_code("E0001")
                                .with_note(
                                    "use `std::math::pow` for floating-point exponentiation",
                                ),
                            ),
                        }
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
                // WP-C4.7-6.3: `5 as UInt8` — the cast's SOURCE must be concrete to classify as
                // numeric. A literal operand has no other constraint (a cast does not propagate
                // its target inward: per 03, casts are explicit conversions, not expectations),
                // so settle it to its default width here.
                let source_resolved = self.default_int_literal_now(&source);
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
                    self.resolve_method(*base, *name, turbofish.as_ref(), args, expr.span, expr_id)
                } else if let hir::ExprKind::Path {
                    res: Res::TraitMember(trait_id, member),
                    ..
                } = &self.hir.expr(*callee).kind
                {
                    self.check_qualified_trait_call(expr_id, *trait_id, *member, args, expr.span)
                } else if let hir::ExprKind::Path {
                    res: Res::CoreTraitMember(core_trait, method_span),
                    ..
                } = &self.hir.expr(*callee).kind
                {
                    self.check_qualified_core_trait_call(
                        expr_id,
                        *core_trait,
                        *method_span,
                        args,
                        expr.span,
                    )
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
                        // WP-C4.7-9 audit: `print`/`println` type their argument as a fresh
                        // inference variable, so they accepted ANY type — including a user struct
                        // with no `Display` impl. 06-Standard-Library says `Display` is not a
                        // syntax hook and user types must implement it, so that was an
                        // over-acceptance: the checker admitted a program the oracle then
                        // rendered in an unspecified debug-ish form and MIR refused outright.
                        // Deferred to Pass 3 so inference has settled first.
                        if matches!(
                            builtin,
                            Builtin::Print | Builtin::Println | Builtin::Eprint | Builtin::Eprintln
                        ) {
                            if let (Some(ty), Some(arg)) = (arg_tys.first(), args.first()) {
                                self.display_checks
                                    .push((ty.clone(), self.hir.expr(*arg).span));
                                self.record_display_plan(*arg, ty.clone());
                            }
                        }
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
                                // WP-C6.2c: arguments have fixed the base type parameters, so a
                                // deferred projection in the return can be resolved before use.
                                self.discharge_ready_projections();
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
                            let param_snapshot = params.clone();
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
                            // WP-C6.2c: resolve any deferred return projection now the arguments
                            // have fixed the base type parameters.
                            self.discharge_ready_projections();
                            // AS3 Boundary 1: the DYNAMIC half of the model, published at the same
                            // time as the static half so it is exercised from the start.
                            //
                            // The checker knows this is a call and knows its signature. It does NOT
                            // know the body: DEV-178 established that the value carries the item and
                            // the bindings it was created with, because `Ty::Fn` cannot say which
                            // instantiation produced it. `FunctionValue` states that rather than
                            // pretending a `BlockId` exists here.
                            // **DEV-193: not every call reaching here is a function-VALUE call.**
                            //
                            // `free(1)`, where `free` names a known `fn` item, falls into this
                            // branch too — and published `FunctionValue`, the selection that means
                            // "the body is not knowable here". It is knowable: the callee path
                            // published `Direct`/`Static(body)` a moment earlier. So `free(1)` and
                            // `g(2)` produced IDENTICAL records at their call expressions, and a
                            // consumer reading the call site could not tell a direct call from a
                            // call through a value — the exact conflation three binding times exist
                            // to prevent.
                            //
                            // The record for a direct call is the path's; publishing a second,
                            // weaker one here would be a duplicate that contradicts it.
                            let callee_is_known_fn = match &self.hir.expr(*callee).kind {
                                hir::ExprKind::Path {
                                    res: Res::Item(item),
                                    ..
                                } => matches!(self.hir.item(*item).kind, hir::ItemKind::Fn(_)),
                                _ => false,
                            };
                            let use_ = CallableUse {
                                selection: CalleeSelection::FunctionValue,
                                environment: GenericEnvironment::FromFunctionValue,
                                receiver_adjustment: ReceiverAdjustment::None,
                                receiver_binding: ReceiverBinding::None,
                                signature: CallableSigTy {
                                    receiver: None,
                                    params: param_snapshot,
                                    ret: (*ret).clone(),
                                },
                                provenance: DispatchProvenance::FunctionValue,
                            };
                            if !callee_is_known_fn {
                                self.publish_callable_use(expr_id, use_);
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
                        let field_ty = self
                            .struct_fields
                            .get(&struct_id)
                            .and_then(|fields| fields.get(name_str))
                            .cloned();
                        if let Some(field_ty) = field_ty {
                            // WP-C6.2b-F1: a private field is inaccessible outside its module.
                            let name_owned = name_str.to_string();
                            let is_pub = self.struct_field_is_pub(struct_id, &name_owned);
                            self.check_member_visible(
                                is_pub,
                                struct_id,
                                "field",
                                &name_owned,
                                *name,
                            );
                            let map = self.nominal_param_map(struct_id, &args);
                            self.instantiate_ty(&field_ty, &map)
                        } else if self.struct_fields.contains_key(&struct_id) {
                            self.diags.push(
                                Diagnostic::error(
                                    format!("struct field '{}' not found", name_str),
                                    *name,
                                )
                                .with_code("E0001"),
                            );
                            Ty::Error
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
                        // WP-C1.5: `Option`/`Result` are always `Ty::Core(CoreType::Option|
                        // Result, _)` (see `hir::CoreType`), never `Ty::Enum` -- a `Ty::Enum`
                        // arm here previously did a substring search over the enum's entire
                        // declaration source text for "Result"/"Option", which let any
                        // unrelated user enum with a matching substring anywhere in its
                        // declaration (e.g. a variant literally named `ResultVariant`) satisfy
                        // this check. 03-Type-System.md:590 defines `?` exclusively for
                        // `Result<T, E>`/`Option<T>`; there is no user-extensible Try trait in
                        // Core v1, so no `Ty::Enum` should ever satisfy this.
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

                // DEV-134: relate the OPERAND to the enclosing return type. Before this, the two
                // were checked only for being Result-or-Option INDEPENDENTLY, and never against
                // each other -- so `Result<_, Low>` propagated out of a function returning
                // `Result<_, High>` (no `From` impl required or applied), and `Option<_>`
                // propagated out of a function returning `Result<_, _>`. Both produced a value
                // whose variant tag belonged to a different type: type confusion, not a
                // diagnostic gap. Deferred to `check_try_compatibility` so inference has settled.
                if let Some(fn_ret) = self.current_fn_ret.clone() {
                    self.try_checks.push((expr_ty.clone(), fn_ret, expr.span));
                }

                // 2. Check try expression type
                match self.resolve(&expr_ty) {
                    // WP-C1.5: same fix as above -- Option/Result never resolve to `Ty::Enum`,
                    // so this used to be exploitable via any user enum with a "Result"/"Option"
                    // substring anywhere in its declaration text. No `Ty::Enum` arm here at all
                    // now; it falls through to the `_` rejection below, correctly.
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
                let tys: Vec<Ty> = elems.iter().map(|&e| self.check_expr(e)).collect();
                unit_or_tuple(tys)
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
                // WP-C4.7-6.3: an unsuffixed literal count is an integer-kinded inference var
                // here, not yet a concrete `Int32`. It is integer BY CONSTRUCTION (only integer
                // literals get these vars), so accept it and let defaulting settle the width.
                let count_is_int_literal =
                    matches!(&count_ty, Ty::Infer(id) if self.int_literal_vars.contains_key(id));
                if !matches!(&count_ty, Ty::Primitive(p) if is_integer(*p))
                    && !count_is_int_literal
                    && !matches!(count_ty, Ty::Error)
                {
                    self.diags.push(
                        Diagnostic::error("array repeat count must be an integer", expr.span)
                            .with_code("E0001"),
                    );
                }

                // WP-C1.5: `count` (02-Syntax-Grammar.md:330: "must be a compile-time constant
                // expression") was previously computed by parsing the *raw source text* of the
                // count expression as a bare unsuffixed decimal (`text.parse::<u64>()`) --
                // anything else (a suffixed literal like `5u32`, an underscore-grouped literal
                // like `1_0`, or a `const` item reference) silently failed to parse and fell
                // back to length 0, which then falsely rejected every subsequent valid index
                // into the array with E0007. `const_eval_u64` handles the confirmed-common
                // shapes (a literal, or a reference to a `const` item); anything else is
                // reported directly rather than silently defaulting to a wrong length.
                let len = match self.const_eval_u64(*count) {
                    Some(len) => len,
                    None => {
                        if !matches!(count_ty, Ty::Error) {
                            self.diags.push(
                                Diagnostic::error(
                                    "array repeat count must be a compile-time constant \
                                     expression",
                                    self.hir.expr(*count).span,
                                )
                                .with_code("E0009"),
                            );
                        }
                        0
                    }
                };
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
                    self.publish_aggregate_field_types(expr_id, &expected, &map);
                    self.check_field_initializers(
                        Some(*struct_id),
                        &expected,
                        &map,
                        fields,
                        expr.span,
                    );
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
                        self.publish_aggregate_field_types(expr_id, &expected, &map);
                        self.check_field_initializers(None, &expected, &map, fields, expr.span);
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
                let scr_expr_ty = self.check_expr(*scrutinee);
                let ret_ty = self.new_type_var();

                // **A REFERENCE-TYPED SCRUTINEE IS REJECTED, PER PAT-BIND-001.**
                //
                // The spec states it directly: "a struct/variant path must name the scrutinee's
                // normalized nominal type, and `&T` is not a nominal type, so `match r { E::V(x) =>
                // .. }` for `r: &E` is a type error. This is why the rule is stated over the place
                // read, not over the scrutinee's type." The program writes `match *r`, which IS a
                // read through a reference and binds by PAT-BIND-001.
                //
                // It was not rejected. `Ty::Ref` simply fell through every classifier, and the
                // result was the worst available combination:
                //
                //   - the exhaustiveness check saw a domain it did not know and demanded a wildcard,
                //     reporting E0303 on a match that already covered every variant; and then
                //   - the `_` arm added to satisfy that ABSORBED EVERY CASE at run time, because
                //     the constructor arms were typed against a reference and matched nothing.
                //
                // So the diagnostic pointed at the wrong problem, and the obvious response to it
                // produced a function that silently returned the wildcard's answer for every input.
                // `stark-percent`'s `is_incomplete_escape` reported "not an incomplete escape" for
                // an incomplete escape it had just been handed, and no test could see it, because
                // the helper was reporting on itself.
                //
                // Rejecting names the fix rather than the symptom. A `match *r` in the same place
                // compiles and behaves.
                let scr_resolved = self.resolve(&scr_expr_ty);
                if matches!(scr_resolved, Ty::Ref { .. })
                    && arms.iter().any(|arm| self.pat_is_constructor(arm.pat))
                {
                    self.diags.push(
                        Diagnostic::error(
                            format!(
                                "cannot match a reference-typed scrutinee '{}' against constructor patterns",
                                self.ty_to_string(&scr_resolved)
                            ),
                            self.hir.expr(*scrutinee).span,
                        )
                        .with_code("E0001")
                        .with_help(
                            "dereference the scrutinee first: `match *r { .. }`. A binding to a \
                             non-Copy component then has reference type (PAT-BIND-001) and nothing \
                             is moved out of the referent",
                        ),
                    );
                }

                let scr_ty = scr_expr_ty.clone();
                let bind_non_copy_by_ref = if self.scrutinee_reads_through_ref(*scrutinee) {
                    BindMode::ThroughRef
                } else {
                    BindMode::ByValue
                };

                let mut matched_variants = HashSet::new();
                let mut matched_bools = HashSet::new();
                let mut has_wildcard = false;
                // WP-C1.5: `Option`/`Result` resolve to `Ty::Core(CoreType::Option|Result, _)`,
                // never `Ty::Enum` (see `hir::CoreType`), and their `Some`/`None`/`Ok`/`Err`
                // patterns resolve via `Res::Builtin`, never `Res::Variant` -- so the existing
                // `matched_variants`/`Ty::Enum` machinery below never covered them at all.
                // `match opt { Some(v) => .. }` (missing `None`) compiled clean before this fix.
                let (mut matched_some, mut matched_none) = (false, false);
                let (mut matched_ok, mut matched_err) = (false, false);
                // DEV-071 (WP-C4.7-7): the prelude `Ordering` is `Ty::Core(CoreType::Ordering)`
                // with `Res::Builtin` variants — exactly like `Option`/`Result`, and for exactly
                // the same reason it was invisible to the `Ty::Enum`/`matched_variants`
                // machinery. Unlike those two, though, `Ordering` fell through to the
                // "unknown domain, require a wildcard" default, so an all-three-variant match
                // was reported NON-exhaustive (E0303) and every `Ordering` match needed a
                // pointless `_` arm.
                let (mut matched_less, mut matched_equal, mut matched_greater) =
                    (false, false, false);

                let mut preceding_patterns = Vec::new();

                for arm in arms {
                    let pat_ty =
                        self.check_pat_with_mode(arm.pat, scr_ty.clone(), bind_non_copy_by_ref);
                    let _ = self.unify(scr_ty.clone(), pat_ty, arm.pat.span(self.hir));

                    let pat = self.hir.pat(arm.pat);

                    let mut is_unreachable = false;
                    for prev_pat in &preceding_patterns {
                        #[allow(clippy::explicit_auto_deref)]
                        if self.pat_subsumes(*prev_pat, pat) {
                            is_unreachable = true;
                            break;
                        }
                    }
                    if is_unreachable {
                        self.diags.push(
                            Diagnostic::warning("unreachable match arm", arm.pat.span(self.hir))
                                .with_code("W0006")
                                .with_label(
                                    "this pattern is redundant and covered by a preceding arm",
                                ),
                        );
                    } else {
                        preceding_patterns.push(pat);
                    }

                    if self.is_irrefutable(pat) {
                        has_wildcard = true;
                    }
                    match &pat.kind {
                        hir::PatKind::Wild | hir::PatKind::Binding { .. } => {}
                        hir::PatKind::Path { res, .. }
                        | hir::PatKind::TupleVariant { res, .. }
                        | hir::PatKind::Struct { res, .. } => match res {
                            Res::Variant(_, variant_idx) => {
                                matched_variants.insert(*variant_idx);
                            }
                            Res::Builtin(Builtin::Some) => matched_some = true,
                            Res::Builtin(Builtin::None) => matched_none = true,
                            Res::Builtin(Builtin::Ok) => matched_ok = true,
                            Res::Builtin(Builtin::Err) => matched_err = true,
                            Res::Builtin(Builtin::OrderingLess) => matched_less = true,
                            Res::Builtin(Builtin::OrderingEqual) => matched_equal = true,
                            Res::Builtin(Builtin::OrderingGreater) => matched_greater = true,
                            _ => {}
                        },
                        hir::PatKind::Lit(Lit::Bool(value)) => {
                            matched_bools.insert(*value);
                        }
                        _ => {}
                    }

                    let body_ty = self.check_expr(arm.body);
                    let _ = self.unify(ret_ty.clone(), body_ty, self.hir.expr(arm.body).span);
                }

                if !has_wildcard {
                    let non_exhaustive = match self.resolve(&scr_ty) {
                        Ty::Enum(enum_id, _) => self
                            .enum_variants
                            .get(&enum_id)
                            .is_some_and(|variants| matched_variants.len() < variants.len()),
                        Ty::Primitive(Primitive::Bool) => matched_bools.len() < 2,
                        Ty::Core(CoreType::Option, _) => !(matched_some && matched_none),
                        Ty::Core(CoreType::Result, _) => !(matched_ok && matched_err),
                        // DEV-071: `Ordering` has exactly three fieldless variants, so matching
                        // all three IS exhaustive and needs no wildcard.
                        Ty::Core(CoreType::Ordering, _) => {
                            !(matched_less && matched_equal && matched_greater)
                        }
                        // WP-C1.5: every other scrutinee type previously fell through here
                        // silently, regardless of arm coverage -- `match x: Int32 { 1 => ..,
                        // 2 => .. }` (missing every other Int32 value) compiled clean and only
                        // trapped at runtime ("non-exhaustive match reached", interp.rs) if an
                        // unmatched value actually occurred. 04-Semantic-Analysis.md is explicit:
                        // "If a match is not exhaustive, it is a compile-time error." A real
                        // usefulness/coverage algorithm (tracking which literal values or ranges
                        // are covered) is out of this WP's scope; instead, any scrutinee type
                        // that isn't one of the small, exactly-enumerable domains above now
                        // requires an explicit wildcard/binding arm to be considered exhaustive
                        // -- sound (never accepts a genuinely non-exhaustive match), and matches
                        // this codebase's existing "reject some safe programs is intentional"
                        // philosophy (03-Type-System.md's own framing for the analogous borrow-
                        // checking tradeoff). `Ty::Struct` is exempted: a struct type has exactly
                        // one shape, so any single struct-pattern arm is exhaustive over it by
                        // construction (sub-pattern-level literal restrictions, e.g. `Point{x: 0,
                        // ..}`, are not yet analyzed here -- same pre-existing imprecision as
                        // before this fix, backstopped by the same runtime trap).
                        Ty::Error | Ty::Struct(..) => false,
                        _ => true,
                    };
                    if non_exhaustive {
                        self.diags.push(
                            Diagnostic::error("non-exhaustive pattern match", expr.span)
                                .with_code("E0303"),
                        );
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
                let resolved_iter_ty = self.resolve(&iter_ty);
                // WP-C7.9 Packet E: by-VALUE `Vec` iteration is refused here rather than left to
                // be accepted and then refused by lowering. It type-checked and ran in the
                // reference interpreter while no compiler could build it — an accepted program no
                // engine below HIR could execute. Iterating a borrow (`v.iter()`) is supported and
                // is what the diagnostic points at.
                if matches!(resolved_iter_ty, Ty::Core(CoreType::Vec, _)) {
                    self.diags.push(
                        Diagnostic::error(
                            "by-value iteration over Vec<T> is not supported by this compiler; \
                             iterate over a borrow with 'v.iter()'",
                            self.hir.expr(*iter).span,
                        )
                        .with_code("E0105"),
                    );
                }
                let elem_ty = match resolved_iter_ty.clone() {
                    Ty::Range(elem) | Ty::Array(elem, _) | Ty::Slice(elem) => *elem,
                    Ty::Core(CoreType::Vec, args) => args.first().cloned().unwrap_or(Ty::Error),
                    // **`for x in &v` — the spelling everyone reaches for first.**
                    //
                    // It used to be E0001 "requires an iterable value, found '&Vec<T>'", which is
                    // an unhelpful refusal: the value IS iterable, and the borrow is exactly what
                    // Vec iteration wants. Combined with by-value `for x in v` being refused
                    // (E0105), two of the three natural spellings failed and only `v.iter()`
                    // worked — with the practical effect that a `Vec` of non-`Copy` elements
                    // looked unreadable, since indexing it is refused too.
                    //
                    // This is the same borrowed cursor `v.iter()` builds, so the item is `&T`.
                    // `&mut Vec<T>` iterates the same way: the cursor is shared regardless, and
                    // accepting it avoids a second confusing refusal for a caller who happens to
                    // hold a mutable borrow.
                    Ty::Ref { inner, .. }
                        if matches!(inner.as_ref(), Ty::Core(CoreType::Vec, _)) =>
                    {
                        match inner.as_ref() {
                            Ty::Core(CoreType::Vec, args) => Ty::Ref {
                                mutable: false,
                                inner: Box::new(args.first().cloned().unwrap_or(Ty::Error)),
                            },
                            _ => Ty::Error,
                        }
                    }
                    other if self.is_iterator_type(&other) => self.iterator_item_type(&other),
                    Ty::Struct(..) | Ty::Enum(..) => self
                        .user_iterator_item_type(&resolved_iter_ty)
                        .unwrap_or_else(|| {
                            self.diags.push(
                                Diagnostic::error(
                                    format!(
                                        "for-loop requires an iterable value, found '{}'",
                                        self.ty_to_string(&resolved_iter_ty)
                                    ),
                                    self.hir.expr(*iter).span,
                                )
                                .with_code("E0001"),
                            );
                            Ty::Error
                        }),
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

                // AS3 Boundary 4: publish the `Iterator::next` this loop selected. Placed after
                // the element type is resolved, because that resolution is what proves an
                // `Iterator` impl matched.
                self.publish_iterator_use(expr_id, &resolved_iter_ty);
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

    /// WP-C5.3e / DEV-100: publish the tables a layout walk needs so the walk itself can live in
    /// ONE place ([`LayoutTables::layout_of`]) and outlive the checker.
    ///
    /// Declaration ORDER is read from the HIR items, not from the checker's own `struct_fields`
    /// map: layout depends on field order and that map is name-keyed. A struct-shaped enum variant
    /// is omitted rather than laid out in an arbitrary order — its fields live in a `HashMap` too,
    /// and a wrong order is a wrong observable answer.
    fn build_layout_tables(&self) -> LayoutTables {
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
                    match self
                        .tensor_ctx
                        .freshen_tensor(kind, dims, dtypes, devices, span)
                    {
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
        use_expr: Option<ExprId>,
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
            let fresh = self.freshen_call_sig(sig, span);
            // **AS3: a non-generic call is published too.**
            //
            // `callable_instantiations` records nothing here — there is no environment to record —
            // which is why `push_callable_env` reports "not pushed" and the interpreter falls
            // through. Under totality that absence is indistinguishable from "no record exists",
            // and the whole point is that an execution site finding no record is an internal
            // compiler error rather than a licence to scan. So the environment is an explicitly
            // EMPTY `Static(vec![])`.
            if let (Some(expr_id), hir::ItemKind::Fn(def)) =
                (use_expr, &self.hir.item(item_id).kind)
            {
                let body = def.body;
                let use_ = CallableUse {
                    selection: CalleeSelection::Static {
                        declaration: CallableDeclId::Item(item_id),
                        body,
                    },
                    environment: GenericEnvironment::Static(Vec::new()),
                    receiver_adjustment: ReceiverAdjustment::None,
                    receiver_binding: ReceiverBinding::None,
                    signature: CallableSigTy {
                        receiver: None,
                        params: fresh.params.clone(),
                        ret: fresh.ret.clone(),
                    },
                    provenance: DispatchProvenance::Direct,
                };
                self.publish_callable_use(expr_id, use_);
            }
            return fresh;
        }

        let mut pending_use: Option<PendingUse> = None;
        let mut map = HashMap::new();
        if let Some(args) = turbofish {
            let has_tensor_kind = generics.iter().any(|param| {
                param.bounds.iter().any(|bound| {
                    bound.res == Res::Err
                        && single_segment_name(&bound.path, self)
                            .and_then(tensor_syntax::tensor_param_kind)
                            .is_some()
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
                // DEV-101: the generic parameter NAME is declared by the callee, so its span is
                // only meaningful against the callee's file (`item_text`) — reading it with
                // `self.text` (the CALLER's file) produced a wrong string for a cross-file/
                // cross-package callee, so this key never matched the `Ty::Param(name)` recorded in
                // `fn_sigs` (built under the callee's file) and the parameter stayed unsubstituted.
                // The turbofish ARGUMENT below is the caller's, so it stays on `self.file`.
                let param_name = self.item_text(item_id, param.name).to_string();
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
                let enclosing = self.current_generic_env();
                self.bounds_checks
                    .push((arg_ty.clone(), trait_bounds, span, enclosing));
                map.insert(param_name, arg_ty);
            }
        } else {
            for param in generics {
                // DEV-101: callee-declared name → callee's file.
                let param_name = self.item_text(item_id, param.name).to_string();
                let var = self.new_type_var();
                let trait_bounds = param
                    .bounds
                    .iter()
                    .filter(|bound| bound.res != Res::Err)
                    .cloned()
                    .collect();
                let enclosing = self.current_generic_env();
                self.bounds_checks
                    .push((var.clone(), trait_bounds, span, enclosing));
                map.insert(param_name, var);
            }
        }

        // WP-C4.5c: record the ordered instantiation for MIR monomorphisation, keyed by the
        // referencing path expression. Fresh inference variables recorded here resolve through
        // `subst` by the time `analyze` grounds and publishes the table; any that remain
        // undetermined are rejected there (E0004 — TYPE-GENERIC-001 / TYPE-FN-002, DEV-064).
        // Tensor-kinded parameters (`Dim`/`DType`/`Device` bounds) unify through the tensor
        // context, not value-type substitution — those functions are extension territory and
        // are neither recorded nor subject to the undetermined-instantiation rejection.
        let has_tensor_kinded_param = generics.iter().any(|param| {
            param.bounds.iter().any(|bound| {
                bound.res == Res::Err
                    && single_segment_name(&bound.path, self)
                        .and_then(tensor_syntax::tensor_param_kind)
                        .is_some()
            })
        });
        if let Some(expr_id) = use_expr.filter(|_| !has_tensor_kinded_param) {
            // A3c-S: the same instantiation as a provenance-carrying environment. A free function
            // has no impl, no trait and no `Self`, so this is the degenerate case of the table —
            // recorded through the same path so there is one answer to what a generic call means
            // rather than one table for free functions and another for methods.
            if let hir::ItemKind::Fn(def) = &self.hir.item(item_id).kind {
                let body = def.body;
                let own_names: Vec<String> = generics
                    .iter()
                    .map(|param| self.item_text(item_id, param.name).to_string())
                    .collect();
                let env_map = map.clone();
                self.publish_callable_env(PublishedEnv {
                    call_expr: expr_id,
                    body,
                    self_ty: None,
                    impl_names: &[],
                    own_names: &own_names,
                    own_is_method: false,
                    map: &env_map,
                });
                // AS3 Boundary 1: the same decision, published in the form an engine can CONSUME
                // rather than verify. **Deferred to the end of this function**, because the
                // signature must be the INSTANTIATED one: publishing `sig` here recorded
                // `params: [Param("T")]` against an environment saying `T = Int32`, so §3.4's
                // invariant failed the moment the test stopped skipping generic uses.
                let bindings = Self::env_bindings(&None, &[], &own_names, false, &env_map);
                pending_use = Some((expr_id, body, bindings));
            }
        }

        // Associated-type equality bindings participate in substitution, so a
        // return such as `I::Item` becomes concrete at each instantiation of
        // `fn first<I: Iterator<Item = Int32>>(...)`.
        for param in generics {
            // DEV-101: the parameter name and the associated-binding name are both callee-declared,
            // so they read against the callee's file. (The binding TYPE `ty` is converted below;
            // for a cross-file callee whose binding type names a callee-local type, that conversion
            // still reads `self.file` — covered by the cross-package associated-type work, not the
            // core unbounded/inference fix.)
            let param_name = self.item_text(item_id, param.name).to_string();
            for bound in &param.bounds {
                if let Some(args) = &bound.args {
                    for arg in &args.args {
                        if let hir::GenericArg::Binding { name, ty } = arg {
                            let binding = self.convert_hir_type(*ty);
                            let binding = self.instantiate_ty(&binding, &map);
                            map.insert(
                                format!("{param_name}::{}", self.item_text(item_id, *name)),
                                binding,
                            );
                        }
                    }
                }
            }
        }

        let params: Vec<Ty> = sig
            .params
            .iter()
            .map(|p| self.instantiate_ty_deferring_projections(p, &map, span))
            .collect();
        let ret = self.instantiate_ty_deferring_projections(&sig.ret, &map, span);

        let instantiated = self.freshen_call_sig(FnSigTy { params, ret }, span);
        if let Some((expr_id, body, bindings)) = pending_use {
            self.publish_named_use(
                expr_id,
                body,
                bindings,
                ReceiverAdjustment::None,
                ReceiverBinding::None,
                CallableSigTy {
                    receiver: None,
                    params: instantiated.params.clone(),
                    ret: instantiated.ret.clone(),
                },
                DispatchProvenance::Direct,
            );
        }
        instantiated
    }

    /// WP-C6.2c: like [`Self::instantiate_ty`], but a projection `T::Item` whose base substitutes
    /// to an inference variable (the concrete type is fixed only by unifying a call argument) is
    /// replaced with a fresh variable and a deferred obligation, resolved once the base grounds.
    fn instantiate_ty_deferring_projections(
        &mut self,
        ty: &Ty,
        map: &HashMap<String, Ty>,
        span: Span,
    ) -> Ty {
        match ty {
            Ty::Param(name) => {
                if let Some(target) = map.get(name) {
                    return target.clone();
                }
                if let Some((base, assoc)) = name.split_once("::") {
                    match map.get(base) {
                        Some(Ty::Struct(id, _) | Ty::Enum(id, _)) => {
                            if let Some(bound) =
                                self.assoc_projections.get(&(*id, assoc.to_string()))
                            {
                                return bound.clone();
                            }
                        }
                        Some(Ty::Infer(base_var)) => {
                            let base_var = *base_var;
                            let assoc = assoc.to_string();
                            let proj = self.new_type_var();
                            if let Ty::Infer(pid) = proj {
                                self.projection_obligations
                                    .push((pid, base_var, assoc, span));
                            }
                            return proj;
                        }
                        _ => {}
                    }
                }
                ty.clone()
            }
            Ty::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.instantiate_ty_deferring_projections(inner, map, span)),
            },
            Ty::Struct(item, args) => Ty::Struct(
                *item,
                args.iter()
                    .map(|a| self.instantiate_ty_deferring_projections(a, map, span))
                    .collect(),
            ),
            Ty::Enum(item, args) => Ty::Enum(
                *item,
                args.iter()
                    .map(|a| self.instantiate_ty_deferring_projections(a, map, span))
                    .collect(),
            ),
            Ty::Core(core, args) => Ty::Core(
                *core,
                args.iter()
                    .map(|a| self.instantiate_ty_deferring_projections(a, map, span))
                    .collect(),
            ),
            Ty::Tuple(elems) => Ty::Tuple(
                elems
                    .iter()
                    .map(|e| self.instantiate_ty_deferring_projections(e, map, span))
                    .collect(),
            ),
            Ty::Array(elem, len) => Ty::Array(
                Box::new(self.instantiate_ty_deferring_projections(elem, map, span)),
                *len,
            ),
            Ty::Slice(elem) => Ty::Slice(Box::new(
                self.instantiate_ty_deferring_projections(elem, map, span),
            )),
            Ty::Range(elem) => Ty::Range(Box::new(
                self.instantiate_ty_deferring_projections(elem, map, span),
            )),
            // Fn types and everything else fall back to the non-deferring instantiation.
            _ => self.instantiate_ty(ty, map),
        }
    }

    /// WP-C6.2c: discharge every deferred projection obligation whose base variable has grounded to
    /// a nominal, binding its placeholder to the impl's associated-type binding. Obligations whose
    /// base is still open are retained. Called eagerly after each call's arguments unify (so an
    /// immediate use like `build(H {}).v` sees a concrete type) and once more at the end of
    /// checking to catch bases that only ground later.
    fn discharge_ready_projections(&mut self) {
        if self.projection_obligations.is_empty() {
            return;
        }
        let obligations = std::mem::take(&mut self.projection_obligations);
        let mut retained = Vec::new();
        for (proj_var, base_var, assoc, span) in obligations {
            let nominal = match self.resolve(&Ty::Infer(base_var)) {
                Ty::Struct(id, _) | Ty::Enum(id, _) => Some(id),
                _ => None,
            };
            match nominal.and_then(|n| self.assoc_projections.get(&(n, assoc.clone())).cloned()) {
                Some(bound) => {
                    let _ = self.unify(Ty::Infer(proj_var), bound, span);
                }
                None => retained.push((proj_var, base_var, assoc, span)),
            }
        }
        self.projection_obligations = retained;
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

    /// DEV-052: `Eq::eq(&a, &b)`-style qualified calls to a compiler-known `CoreTrait`'s method.
    /// Unlike `check_qualified_trait_call` (a user-declared trait, which has an
    /// `hir::ItemKind::Trait` item whose declared signature is authoritative for every
    /// implementor), a `CoreTrait` has no such declaration item -- each `impl <CoreTrait> for T`
    /// writes its own method signature directly, so the *matching impl's own* signature is used
    /// instead of one inherited from a shared trait declaration. `receiver_ty`'s own `impl`
    /// search matches by source-text trait name (`self.text(trait_ref.path.span)`), mirroring
    /// `ty_satisfies_operator_bound`'s existing approach for the same compiler-known traits.
    fn check_qualified_core_trait_call(
        &mut self,
        // AS3 Boundary 4: the call expression, so the selected impl member can be published.
        call_expr: ExprId,
        core_trait: hir::CoreTrait,
        method_span: Span,
        args: &[ExprId],
        span: Span,
    ) -> Ty {
        let method_name = self.text(method_span).to_string();
        let core_trait_name = core_trait_source_name(core_trait);

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
        // **AS3 Boundary 4: publish the selection.** `Eq::eq(&a, &b)` is the explicit spelling of
        // the same dispatch `a == b` performs, so it publishes through the same publisher — one
        // statement of what a qualified core-trait call means, not two.
        let receiver_for_publication = receiver_type.clone();
        self.publish_operator_use(
            call_expr,
            &receiver_for_publication,
            core_trait_name,
            &method_name,
            core_trait,
        );

        let mut selected: Option<hir::FnSig> = None;
        for item in &self.hir.items {
            let hir::ItemKind::Impl {
                trait_: Some(trait_ref),
                self_ty,
                items,
                generics,
            } = &item.kind
            else {
                continue;
            };
            if self.text(trait_ref.path.span) != core_trait_name {
                continue;
            }
            let implementation_type = self.convert_hir_type(*self_ty);
            if self
                .match_impl_type(&implementation_type, &receiver_type, generics)
                .is_none()
            {
                continue;
            }
            selected = items.iter().find_map(|impl_item| match impl_item {
                hir::ImplItem::Fn { def, .. } if self.text(def.sig.name) == method_name => {
                    Some(def.sig.clone())
                }
                _ => None,
            });
            if selected.is_some() {
                break;
            }
        }

        let Some(sig) = selected else {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "type '{}' does not implement '{core_trait_name}'",
                        self.ty_to_string(&receiver_type)
                    ),
                    span,
                )
                .with_code("E0500"),
            );
            return Ty::Error;
        };

        let mut expected = Vec::new();
        if let Some(receiver) = sig.receiver {
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
        expected.extend(
            sig.params
                .iter()
                .map(|param| self.convert_hir_type(param.ty)),
        );
        let result = match sig.ret {
            hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
            hir::RetTy::Ty(ty) => self.convert_hir_type(ty),
            hir::RetTy::Never(_) => Ty::Never,
        };

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

    /// The trait's own default body for `member`, as an (owner, member, body) triple shaped like
    /// [`Self::operator_impl_member`]'s. Used when an implementor accepts the default and there is
    /// therefore no impl member to find.
    /// A TRAIT method's declared signature, with `Self` bound to the concrete receiver. The
    /// trait-default counterpart of [`Self::declared_member_signature`], which reads an impl.
    fn trait_member_signature(
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

    fn trait_default_member(
        &self,
        trait_id: ItemId,
        member: u32,
    ) -> Option<(ItemId, u32, BlockId)> {
        let hir::ItemKind::Trait { items, .. } = &self.hir.item(trait_id).kind else {
            return None;
        };
        let hir::TraitItem::Method {
            body: Some(body), ..
        } = items.get(member as usize)?
        else {
            return None;
        };
        Some((trait_id, member, *body))
    }

    fn check_qualified_trait_call(
        &mut self,
        call_expr: ExprId,
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
        // AS3 Boundary 4: publish the selection, so the interpreter reads the body instead of
        // scanning the receiver's nominal for the member name.
        let trait_name = match &self.hir.item(trait_id).kind {
            hir::ItemKind::Trait { name, .. } => self.item_text(trait_id, *name).to_string(),
            _ => String::new(),
        };
        let member_name = match &self.hir.item(trait_id).kind {
            hir::ItemKind::Trait { items, .. } => match items.get(member as usize) {
                Some(hir::TraitItem::Method { sig, .. }) => {
                    self.item_text(trait_id, sig.name).to_string()
                }
                _ => String::new(),
            },
            _ => String::new(),
        };
        // **The impl member if the implementor overrides it, otherwise the trait's DEFAULT body.**
        //
        // `operator_impl_member` finds written impl members only. `<T as Tr>::m(&x)` where `T`
        // accepts the default has no impl member to find, and publishing nothing there left the
        // interpreter — which no longer has a name scan — with nothing to select. Third instance of
        // one shape in this packet: a trait default reached by a route other than a `Static` method
        // call.
        let selected = self
            .operator_impl_member(&receiver_type, &trait_name, &member_name)
            .map(|(owner, owner_member, body, _)| (owner, owner_member, body))
            .or_else(|| self.trait_default_member(trait_id, member));
        if let Some((owner, owner_member, body)) = selected {
            // The signature comes from whichever declaration owns the body — an impl member, or
            // the trait itself when the implementor accepts the default.
            let signature = if owner == trait_id {
                self.trait_member_signature(trait_id, owner_member, &receiver_type)
            } else {
                self.declared_member_signature(owner, owner_member)
            };
            if let Some((receiver, params, ret)) = signature {
                let use_ = CallableUse {
                    selection: CalleeSelection::Static {
                        declaration: CallableDeclId::ImplMember {
                            impl_item: owner,
                            member: owner_member,
                        },
                        body,
                    },
                    // **`Self`, published.** A trait DEFAULT body reached this way runs with
                    // `Ty::Param("Self")` throughout, so without this binding a `self.other()`
                    // inside it resolves nothing. The checker knows the receiver here — unlike the
                    // bound path, where the body is only chosen at run time.
                    environment: GenericEnvironment::Static(vec![(
                        GenericBinder::SelfType,
                        receiver_type.clone(),
                    )]),
                    receiver_adjustment: ReceiverAdjustment::Shared { derefs: 0 },
                    receiver_binding: ReceiverBinding::Shared,
                    signature: CallableSigTy {
                        receiver,
                        params,
                        ret,
                    },
                    provenance: DispatchProvenance::Qualified {
                        trait_item: Some(trait_id),
                    },
                };
                self.publish_callable_use(call_expr, use_);
            }
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

        let previous_self = self.enter_self_scope(receiver_type.clone());
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
        self.exit_self_scope(SelfScope {
            assoc_types: Some(previous_assoc),
            ..previous_self
        });

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
        // AS6 packet 4B group 2C: what a refinement produces is tensor semantics.
        tensor_check::eval_tensor_refine(self, base, turbofish, name_span)
    }

    /// DEV-DISPLAY-DISPATCH: every candidate the bounds on generic parameter `p_name` contribute
    /// for method `name`, from both kinds of trait, one per distinct trait identity.
    ///
    /// This is candidate COLLECTION only. Selection, ambiguity and argument checking happen once,
    /// at the call site, over whatever this returns — which is the whole point: a compiler-known
    /// bound and a user bound reach the same selection through the same list.
    /// DEV-169: refuse an explicit call to a `Drop` implementation's `drop`.
    ///
    /// 03-Type-System.md, "Copy and Drop": "`Drop::drop` MUST NOT be called explicitly; use the
    /// free function `drop(value)`." The free function is a different thing — it MOVES its
    /// argument, so the destructor still runs exactly once.
    fn reject_explicit_drop(&mut self, impl_item: ItemId, name: &str, span: Span) {
        if name != "drop" {
            return;
        }
        let hir::ItemKind::Impl {
            trait_: Some(trait_ref),
            ..
        } = &self.hir.item(impl_item).kind
        else {
            return;
        };
        if !matches!(
            hir::resolved_bound_trait(self.hir, trait_ref),
            Some(hir::BoundTrait::Core(hir::CoreTrait::Drop))
        ) {
            return;
        }
        self.diags.push(
            Diagnostic::error("'Drop::drop' cannot be called explicitly", span)
                .with_code("E0307")
                .with_label("the destructor runs automatically when the value goes out of scope")
                .with_note(
                    "to destroy a value early, move it into the free function 'drop(value)'; \
                     calling the method here would run the destructor twice"
                        .to_string(),
                ),
        );
    }

    fn bound_method_candidates(&self, p_name: &str, name: &str) -> Vec<BoundMethod> {
        // WP-C6.2b-F5: consult the method's own generics AND the enclosing impl's generics, so
        // a bound written on the impl head (`impl<T: Sh> W<T>`) is visible in the method body.
        let mut generics = self.current_fn_generics.clone().unwrap_or_default();
        if let Some(impl_generics) = &self.current_impl_generics {
            generics.extend(impl_generics.iter().cloned());
        }
        let mut seen: Vec<BoundTrait> = Vec::new();
        let mut candidates: Vec<BoundMethod> = Vec::new();
        for param in &generics {
            if self.text(param.name) != p_name {
                continue;
            }
            for bound in &param.bounds {
                // DEV-BOUND-TRAIT-IDENTITY: the identity comes from the RESOLVER, never from
                // how the bound was spelled. See `hir::resolved_bound_trait` for the three
                // failures the previous spelling-based lookup produced.
                let Some(bound_trait) = hir::resolved_bound_trait(self.hir, bound) else {
                    continue;
                };
                // `T: Display + Display` names one trait, not two: a repeated bound must not
                // manufacture an ambiguity.
                if seen.contains(&bound_trait) {
                    continue;
                }
                seen.push(bound_trait);
                match bound_trait {
                    BoundTrait::User(trait_id) => {
                        if let Some(sig) = self.find_trait_method_sig(trait_id, name) {
                            candidates.push(BoundMethod::User { trait_id, sig });
                        }
                    }
                    BoundTrait::Core(core_trait) => {
                        if let Some(method) = core_trait_bound_method(core_trait, name) {
                            candidates.push(BoundMethod::Core {
                                core_trait,
                                method,
                                trait_args: trait_ref_type_args(bound),
                            });
                        }
                    }
                }
            }
        }
        candidates
    }

    /// The traits behind a set of candidates, as they read in a diagnostic.
    fn bound_trait_list(&self, candidates: &[BoundMethod]) -> String {
        let names: Vec<String> = candidates
            .iter()
            .map(|candidate| match candidate {
                BoundMethod::User { trait_id, .. } => {
                    let hir::ItemKind::Trait { name, .. } = &self.hir.item(*trait_id).kind else {
                        return "<trait>".to_string();
                    };
                    self.item_text(*trait_id, *name).to_string()
                }
                BoundMethod::Core { core_trait, .. } => {
                    core_trait_source_name(*core_trait).to_string()
                }
            })
            .collect();
        names.join(" and ")
    }

    /// DEV-DISPLAY-DISPATCH: check a call against the single selected bound candidate.
    ///
    /// Both arms end in the same argument-checking loop; they differ only in where the declared
    /// parameter and return types come from — an HIR signature for a user trait, the Core trait's
    /// implementation contract for a compiler-known one.
    /// Returns the return type and the method's own generic arguments at this call site.
    fn check_bound_method_call(
        &mut self,
        candidate: &BoundMethod,
        p_name: &str,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        call_span: Span,
    ) -> (Ty, Vec<Ty>) {
        match candidate {
            BoundMethod::User { trait_id, sig } => {
                self.check_trait_member_call(*trait_id, sig, turbofish, args, call_span)
            }
            BoundMethod::Core {
                method, trait_args, ..
            } => {
                let self_ty = Ty::Param(p_name.to_string());
                let trait_arg_tys: Vec<Ty> = trait_args
                    .iter()
                    .map(|ty| self.convert_hir_type(*ty))
                    .collect();
                let params_ty: Vec<Ty> = method
                    .params
                    .iter()
                    .map(|term| self.contract_ty_to_ty(*term, &self_ty, &trait_arg_tys))
                    .collect();
                let ret_ty = match method.ret {
                    None => Ty::Primitive(Primitive::Unit),
                    Some(term) => self.contract_ty_to_ty(term, &self_ty, &trait_arg_tys),
                };
                self.check_call_arguments(params_ty, args, call_span);
                // A core trait's contract is fixed (`ContractTy`) and declares no method-level
                // generics, so this list is empty as a FACT about core traits, not as a gap.
                (ret_ty, Vec::new())
            }
        }
    }

    /// DEV-DISPLAY-DISPATCH: a Core trait contract term as a checker type, with `Self` bound to
    /// the receiver. `Self::Item` becomes the projection `T::Item`, which the caller normalises
    /// against the bindings in scope — the same treatment a user trait's `Self::Item` gets.
    fn contract_ty_to_ty(&mut self, term: ContractTy, self_ty: &Ty, trait_args: &[Ty]) -> Ty {
        match term {
            ContractTy::SelfTy => self_ty.clone(),
            ContractTy::RefSelf => Ty::Ref {
                mutable: false,
                inner: Box::new(self_ty.clone()),
            },
            ContractTy::Bool => Ty::Primitive(Primitive::Bool),
            ContractTy::UInt64 => Ty::Primitive(Primitive::UInt64),
            ContractTy::StringTy => Ty::Primitive(Primitive::String),
            ContractTy::Ordering => Ty::Core(CoreType::Ordering, Vec::new()),
            ContractTy::OptionAssoc(assoc) => {
                let base = match self_ty {
                    Ty::Param(name) => name.clone(),
                    other => self.ty_to_string(other),
                };
                Ty::Core(
                    CoreType::Option,
                    vec![Ty::Param(format!("{base}::{assoc}"))],
                )
            }
            // An unwritten argument (`T: Into` with no `<..>`) has no type to name. `Ty::Error`
            // is the checker's "already reported / do not cascade" type, and the missing argument
            // is reported where the bound is written, not here.
            ContractTy::TraitArg(index) => trait_args.get(index).cloned().unwrap_or(Ty::Error),
        }
    }

    /// The argument half of a call against an already-resolved parameter list. Extracted so the
    /// Core-trait bound path and `check_trait_member_call` cannot drift in how they report an
    /// arity mismatch or unify an argument.
    fn check_call_arguments(&mut self, params_ty: Vec<Ty>, args: &[ExprId], call_span: Span) {
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
    }

    /// Looks up trait `trait_id`'s own declared signature for method `name_str` (a required
    /// method or another default), without needing any concrete `impl` -- used both for a
    /// bounded generic type parameter's method call and for `self.other_method()` called from
    /// inside a trait's own default-method body (DEV-051), where in either case the receiver's
    /// type is an abstract placeholder (`Ty::Param`), not a real struct/enum, so there is no
    /// `impl` to match against yet.
    fn find_trait_method_sig(&self, trait_id: ItemId, name_str: &str) -> Option<hir::FnSig> {
        let hir::ItemKind::Trait { items, .. } = &self.hir.item(trait_id).kind else {
            return None;
        };
        // DEV-069: the trait's method names belong to the TRAIT's declaring file.
        items.iter().find_map(|trait_item| match trait_item {
            hir::TraitItem::Method { sig, .. }
                if self.item_text(trait_id, sig.name) == name_str =>
            {
                Some(sig.clone())
            }
            _ => None,
        })
    }

    /// WP-C6.2c: populate `assoc_projections` from every impl's associated-type bindings, keyed by
    /// the implementing nominal's `ItemId` and the associated-type name.
    fn build_assoc_projections(&mut self) {
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

    /// WP-C6.2c: associated-type projections pinned by explicit binding constraints in scope
    /// (`T: Holder<Item = Int32>` yields `"T::Item" -> Int32`), gathered from the current function's
    /// and enclosing impl's generic parameters.
    fn assoc_binding_map(&mut self) -> HashMap<String, Ty> {
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

    /// WP-C6.2c: resolve any associated-type projection reachable in `ty`. A `Ty::Param("X::Item")`
    /// whose base `X` names a bound param with an explicit binding is replaced from `binding_map`;
    /// one whose base resolves to a concrete nominal is replaced from the program-wide
    /// `assoc_projections` table. Recurses so projections nested inside aggregates are resolved.
    fn normalize_projections(&self, ty: &Ty, binding_map: &HashMap<String, Ty>) -> Ty {
        match ty {
            Ty::Param(name) if name.contains("::") => {
                if let Some(bound) = binding_map.get(name) {
                    return bound.clone();
                }
                if let Some((base, assoc)) = name.split_once("::") {
                    // The base may itself be a bound param carrying a concrete binding: normalise it
                    // first (e.g. `Self` already rewritten to a nominal-bearing param upstream).
                    for ((nominal, aname), bound) in &self.assoc_projections {
                        if aname == assoc && self.nominal_name(*nominal) == base {
                            return bound.clone();
                        }
                    }
                }
                ty.clone()
            }
            Ty::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.normalize_projections(inner, binding_map)),
            },
            Ty::Struct(item, args) => Ty::Struct(
                *item,
                args.iter()
                    .map(|a| self.normalize_projections(a, binding_map))
                    .collect(),
            ),
            Ty::Enum(item, args) => Ty::Enum(
                *item,
                args.iter()
                    .map(|a| self.normalize_projections(a, binding_map))
                    .collect(),
            ),
            Ty::Core(core, args) => Ty::Core(
                *core,
                args.iter()
                    .map(|a| self.normalize_projections(a, binding_map))
                    .collect(),
            ),
            Ty::Tuple(elems) => Ty::Tuple(
                elems
                    .iter()
                    .map(|e| self.normalize_projections(e, binding_map))
                    .collect(),
            ),
            Ty::Array(elem, len) => Ty::Array(
                Box::new(self.normalize_projections(elem, binding_map)),
                *len,
            ),
            Ty::Slice(elem) => Ty::Slice(Box::new(self.normalize_projections(elem, binding_map))),
            Ty::Range(elem) => Ty::Range(Box::new(self.normalize_projections(elem, binding_map))),
            other => other.clone(),
        }
    }

    /// The declared name of a nominal (struct/enum) item, read against its declaring file.
    fn nominal_name(&self, item: ItemId) -> String {
        match &self.hir.item(item).kind {
            hir::ItemKind::Struct { name, .. } | hir::ItemKind::Enum { name, .. } => {
                self.item_text(item, *name).to_string()
            }
            _ => String::new(),
        }
    }

    /// Checks a call's arguments against an already-resolved trait method signature (see
    /// `find_trait_method_sig`) and returns its return type.
    ///
    /// `trait_id` was the declaring trait, carried here for DEV-101 provenance: the signature's
    /// types — including `Self::Item` associated-type spans — had to be read against the trait's
    /// file, which differs from the caller's for a cross-package trait. AS1b-ii-d: those spans
    /// name the trait's file themselves. The parameter is kept so the call sites still say which
    /// trait they resolved.
    /// Returns the method's return type **and** this call site's binding of the method's own
    /// generic parameters, in declaration order — the `method_args` a `CalleeSelection::Bound`
    /// publishes.
    fn check_trait_member_call(
        &mut self,
        _trait_id: ItemId,
        sig: &hir::FnSig,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        call_span: Span,
    ) -> (Ty, Vec<Ty>) {
        // **AS3 Boundary 4 (DEV-188): bind the method's OWN generic parameters.**
        //
        // This ignored `sig.generics` entirely, so `U` stayed rigid and *any* trait method that
        // mentioned its own generic parameter was uncallable through a bound — the turbofish was
        // dropped on the floor. The concrete-receiver path (WP-C4.7-8.4) and the trait-default
        // path already do exactly this; only the bound and `Self`-receiver paths did not.
        let mut map: HashMap<String, Ty> = HashMap::new();
        let mut method_args: Vec<Ty> = Vec::new();
        if let Some(generic_args) = turbofish {
            self.validate_generic_arity(sig.generics.len(), generic_args.args.len(), call_span);
        }
        for (index, param) in sig.generics.iter().enumerate() {
            let ty = match turbofish.and_then(|g| g.args.get(index)) {
                Some(hir::GenericArg::Type(t)) => self.convert_hir_type(*t),
                Some(_) => Ty::Error,
                // No turbofish (or too few): infer it from the arguments, as an ordinary generic
                // call does. `t.to(1)` must work without a turbofish for the same reason `f(1)`
                // does.
                None => self.new_type_var(),
            };
            map.insert(self.decl_text(param.name).to_string(), ty.clone());
            method_args.push(ty);
        }

        // AS1b-ii-d: this used to swap `self.file` to the trait's file to convert the signature
        // and swap back for the arguments. The signature's spans name the trait's file and the
        // arguments' name the caller's, so both convert correctly with no swap at all.
        let params_ty: Vec<Ty> = sig
            .params
            .iter()
            .map(|p| {
                let ty = self.convert_hir_type(p.ty);
                self.instantiate_ty(&ty, &map)
            })
            .collect();
        let ret_ty = match sig.ret {
            hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
            hir::RetTy::Ty(t) => {
                let ty = self.convert_hir_type(t);
                self.instantiate_ty(&ty, &map)
            }
            hir::RetTy::Never(_) => Ty::Never,
        };
        self.check_call_arguments(params_ty, args, call_span);
        // Resolve after the arguments have constrained any inference variable introduced above, so
        // an omitted turbofish still publishes the type the call site actually settled on rather
        // than an unresolved `_infer_N`.
        let method_args = method_args.iter().map(|ty| self.resolve(ty)).collect();
        (self.resolve(&ret_ty), method_args)
    }

    fn resolve_method(
        &mut self,
        base_expr: ExprId,
        name_span: Span,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        call_span: Span,
        // WP-C4.7-8.4: the call expression itself, used to key this call site's METHOD-level
        // generic instantiation for MIR monomorphisation.
        call_expr: ExprId,
    ) -> Ty {
        let base_ty = self.check_expr(base_expr);
        // WP-C4.7-6.3: method resolution must branch on a CONCRETE receiver type, and a literal
        // receiver (`3.cmp(&5)`) has no other constraint to wait for — settle it here rather than
        // failing with "method call on non-struct/enum type '_infer_N'".
        // WP-C6.2b-F2: default int literals inside the receiver too, so a concrete-instance
        // impl (`impl Get for W<Int32>`) matches `let w = W { v: 7 }; w.get()`.
        let resolved_base = self.default_int_literals_deep(&base_ty);
        let name_str = self.text(name_span).to_string();

        if self.options.tensor() && name_str == "refine" {
            return self.check_tensor_refine(resolved_base, turbofish, args, name_span, call_span);
        }

        // AS3 Boundary 2 hardening: TYPE-METHOD-002's auto-dereference is a decision the CALL SITE
        // makes, and it was being discarded. Counting the peels here is what lets
        // `ReceiverAdjustment` publish it instead of every consumer re-deriving it from the
        // receiver's type — which is precisely the reconstruction this packet exists to remove.
        let mut receiver_ty = resolved_base.clone();
        let mut receiver_derefs: u8 = 0;
        let mut outermost_ref_is_mut = false;
        if let Ty::Ref { mutable, .. } = &receiver_ty {
            outermost_ref_is_mut = *mutable;
        }
        while let Ty::Ref { inner, .. } = receiver_ty {
            receiver_ty = self.resolve(&inner);
            receiver_derefs = receiver_derefs.saturating_add(1);
        }

        // DEV-067(b) (WP-C4.7-7): a method call on a BOUNDED generic parameter resolves through
        // the parameter's declared bounds. This tested `resolved_base` — the UNPEELED receiver —
        // so it matched `t: T` but never `t: &T`, and `fn f<T: Speak>(t: &T) { t.speak() }`
        // failed E0302 "method 'speak' not found for type '&T'". TYPE-METHOD-002 requires
        // auto-dereference to peel leading `&`/`&mut` before receiver matching, exactly as the
        // concrete-type path below already did with `receiver_ty`; using the same peeled type
        // here makes the bounded-parameter path obey the same rule.
        // DEV-DISPLAY-DISPATCH: candidate collection over the bounds is ADDITIVE across both
        // kinds of trait, and there is ONE selection step afterwards. Before this, the loop
        // returned on the first bound that supplied the name, and only a bound naming a
        // `hir::ItemKind::Trait` was ever consulted at all — so a compiler-known trait
        // (`Display`, `Ord`, `Clone`, ...) contributed nothing, and two bounds supplying the same
        // name were resolved by declaration order rather than reported as ambiguous.
        if let Ty::Param(p_name) = &receiver_ty {
            let p_name = p_name.clone();
            let candidates = self.bound_method_candidates(&p_name, &name_str);
            if candidates.len() > 1 {
                // Same rule the concrete-receiver path applies when two impls supply the name.
                // Order of the bounds is deliberately not a tie-breaker, and being
                // compiler-known is deliberately not a preference.
                self.diags.push(
                    Diagnostic::error("ambiguous trait method call", call_span)
                        .with_code("E0203")
                        .with_label(format!(
                            "'{}' is declared by more than one trait bound on '{}': {}",
                            name_str,
                            p_name,
                            self.bound_trait_list(&candidates)
                        )),
                );
                for &arg in args {
                    let _ = self.check_expr(arg);
                }
                return Ty::Error;
            }
            if let Some(candidate) = candidates.into_iter().next() {
                // DEV-BOUND-TRAIT-IDENTITY: record WHICH trait supplied the method, so the
                // engines below select the same implementation rather than the first impl on the
                // receiver's nominal that happens to declare the name.
                self.bound_trait_calls.insert(
                    call_expr,
                    match &candidate {
                        BoundMethod::User { trait_id, .. } => Res::Item(*trait_id),
                        BoundMethod::Core { core_trait, .. } => Res::CoreTrait(*core_trait),
                    },
                );
                // WP-C6.2c: a trait method returning `Self::Item` yields the receiver's
                // projection (`T::Item`), which is then pinned by any explicit
                // `T: Trait<Item = ..>` binding in scope.
                let (ret, method_args) =
                    self.check_bound_method_call(&candidate, &p_name, turbofish, args, call_span);
                let ret = Self::subst_self(&ret, &p_name);
                let binding_map = self.assoc_binding_map();
                let ret = self.normalize_projections(&ret, &binding_map);
                // **AS3 Boundary 4 step 2: publish the LATE-BOUND obligation.**
                //
                // This branch previously returned here, so a call on a bounded generic parameter
                // published no `CallableUse` at all — the missing third category. The body cannot
                // be named: `Self` is `Ty::Param(p_name)` and stays parametric until the enclosing
                // function is instantiated. What IS fixed is the obligation, and that is what a
                // `Bound` selection records.
                self.publish_bound_use(
                    call_expr,
                    &candidate,
                    &p_name,
                    &name_str,
                    &ret,
                    method_args,
                );
                return ret;
            }
        }

        // DEV-051: `self.other_method()` called from inside a trait's own default-method body
        // has `current_self_ty == Ty::Param("Self")` (set alongside `current_trait_id` while
        // checking `hir::ItemKind::Trait`'s default bodies), so `self`'s dereferenced type here
        // is `Ty::Param("Self")` -- there's no concrete `impl` to match against yet, since the
        // default body is checked once, generically, at the trait declaration site rather than
        // once per implementor. The trait's own declared signature for `name_str` (required or
        // another default) is authoritative regardless: every real implementor is separately
        // checked elsewhere to provide a matching method, so calling it through `self` from a
        // sibling default body is always legal. (Checked after the deref loop above, unlike the
        // bounded-generic-parameter case just above, because a generic parameter received by
        // value has no reference to peel off, but `self` is always received by reference.)
        if let Ty::Param(p_name) = &receiver_ty {
            if p_name == "Self" {
                if let Some(trait_id) = self.current_trait_id {
                    if let Some(sig) = self.find_trait_method_sig(trait_id, &name_str) {
                        // Same DEV-188 repair: a sibling default body calling another generic
                        // trait method through `self` had `U` rigid for the same reason.
                        let (ret, method_args) = self
                            .check_trait_member_call(trait_id, &sig, turbofish, args, call_span);
                        // **AS3 Boundary 4 (DEV-190): publish this call too.**
                        //
                        // Like the bounded-parameter branch before step 2, this returned without
                        // publishing anything — so `self.id()` inside `fn twice(&self)` had no
                        // `CallableUse`, and both engines had to fall back to a name scan. It is a
                        // `Bound` selection by the same argument: `Self` is a parameter, the trait
                        // is known, and the body is fixed only once an implementor is chosen.
                        let candidate = BoundMethod::User { trait_id, sig };
                        self.publish_bound_use(
                            call_expr,
                            &candidate,
                            "Self",
                            &name_str,
                            &ret,
                            method_args,
                        );
                        return ret;
                    }
                }
            }
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

        // DEV-069: this scans EVERY impl in the program, including impls declared in other
        // files, so method names are read against each impl's OWN file — not `self.file`, which
        // is the file of the item being checked.
        for (impl_index, item) in self.hir.items.iter().enumerate() {
            let impl_item_id = ItemId(impl_index as u32);
            if let hir::ItemKind::Impl {
                self_ty: impl_self_ty_id,
                items,
                trait_,
                generics,
                ..
            } = &item.kind
            {
                // CD-358: both the self-type conversion and the generic-name keying read spans
                // from the IMPL's file. Without this, `impl<T> Wrap<T>` resolved through a module
                // boundary produced a parameter named from the caller's file — `Wrap<T>::get`
                // returned `&S` — and no substitution could ever fire.
                let impl_self_ty = self.convert_hir_type(*impl_self_ty_id);
                let matched = self.match_impl_type(&impl_self_ty, &receiver_ty, generics);
                let Some(map) = matched else {
                    continue;
                };

                for impl_item in items {
                    if let hir::ImplItem::Fn { vis, def } = impl_item {
                        let method_name_str = self.item_text(impl_item_id, def.sig.name);
                        if method_name_str == name_str {
                            candidates.push((
                                def,
                                trait_.is_some(),
                                map.clone(),
                                impl_self_ty.clone(),
                                matches!(vis, Some(crate::ast::Vis::Pub)),
                                impl_item_id,
                            ));
                        }
                    }
                }
            }
        }

        let inherent: Vec<_> = candidates
            .iter()
            .filter(|(_, is_trait, _, _, _, _)| !is_trait)
            .collect();
        // WP-C6.2b-F1: pick the chosen candidate, enforce its visibility, then hand the same
        // 4-tuple downstream. A trait method is visible via its trait's own path rules; the
        // private-impl-member check applies to inherent (and inherent-selected) methods.
        let chosen: Option<MethodCandidate> = if let Some(candidate) = inherent.first() {
            Some((**candidate).clone())
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
        if let Some((_, is_trait, _, _, is_pub, impl_item_id)) = &chosen {
            if !is_trait {
                self.check_member_visible(*is_pub, *impl_item_id, "method", &name_str, call_span);
            }
            // DEV-169: `Drop::drop` MUST NOT be called explicitly (03-Type-System.md, "Copy and
            // Drop"). Accepting it was a DOUBLE DESTRUCTION, not merely an over-acceptance:
            // `r.drop()` ran the destructor once for the call and again when the value went out of
            // scope. Confirmed empirically before the fix — `dropped / after / dropped`.
            //
            // Checked at IMPL-MEMBER SELECTION rather than on the method's name, so it fires
            // exactly when a call resolves into an `impl Drop for T` block and never for an
            // unrelated method that happens to be called `drop`.
            self.reject_explicit_drop(*impl_item_id, &name_str, call_span);
        }
        // CD-358: the impl's ItemId is carried through, because the signature conversion below
        // must read its spans against the impl's own file.
        let selected = chosen.map(|(def, is_trait, map, self_ty, _, impl_item_id)| {
            (def, is_trait, map, self_ty, impl_item_id)
        });

        // WP-C1.3 (2026-07-17): fall back to a trait's own default method body when no impl
        // overrides it. `candidates` above only ever collects `ImplItem::Fn` overrides -- a
        // trait method declared with a real body (03-Type-System.md trait defaults) was never
        // consulted at all, so calling an un-overridden default method failed to type-check
        // with E0302 "method not found" even though the interpreter (once its own matching gap
        // is fixed) has a real body to run. Confirmed empirically before this fix. See
        // COMPILER-STATE.md DEV-013.
        let default_fallback = if selected.is_none() {
            self.hir.items.iter().find_map(|item| {
                let hir::ItemKind::Impl {
                    self_ty: impl_self_ty_id,
                    trait_: Some(trait_ref),
                    generics,
                    ..
                } = &item.kind
                else {
                    return None;
                };
                let impl_self_ty = self.convert_hir_type(*impl_self_ty_id);
                let map = self.match_impl_type(&impl_self_ty, &receiver_ty, generics)?;
                let Res::Item(trait_id) = trait_ref.res else {
                    return None;
                };
                let hir::ItemKind::Trait {
                    items: trait_items, ..
                } = &self.hir.item(trait_id).kind
                else {
                    return None;
                };
                // DEV-069: a trait default's name belongs to the trait's own file, which may
                // differ from both the impl's file and the file being checked.
                trait_items.iter().find_map(|trait_item| match trait_item {
                    hir::TraitItem::Method {
                        sig,
                        body: Some(body),
                    } if self.item_text(trait_id, sig.name) == name_str => Some((
                        sig.clone(),
                        map.clone(),
                        impl_self_ty.clone(),
                        trait_id,
                        *body,
                    )),
                    _ => None,
                })
            })
        } else {
            None
        };

        if let Some((sig, mut map, impl_self_ty, trait_id, trait_body)) = default_fallback {
            // CD-358: a trait default's signature is declared in the TRAIT's file, which may
            // differ from the impl's file and from the file under check. DEV-069 already applied
            // that rule to the default's NAME; its parameter and return types need it too.
            // WP-C4.7-9 audit: a TRAIT-DEFAULT method may declare its own generic parameters
            // too (`02:64`). WP-C4.7-8.4 gave the selected-impl path fresh per-call-site
            // variables for those; this path had the same gap, so `d.say(5)` on an
            // un-overridden `fn say<U>(&self, x: U) -> U` still failed with `U` rigid.
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
            // Record this call site's method-level instantiation for MIR monomorphisation, as
            // the selected-impl path does.
            // A3c-S: the full environment, including `Self` and the trait's own parameters, which
            // the positional record above cannot express. A trait default body carries
            // `Ty::Param("Self")` from the checker, so without this the oracle has no binding for
            // it at all (DEV-176).
            let trait_generics = match &self.hir.item(trait_id).kind {
                hir::ItemKind::Trait { generics, .. } => generics.clone(),
                _ => Vec::new(),
            };
            let env_map = map.clone();
            let trait_names: Vec<String> = trait_generics
                .iter()
                .map(|param| self.item_text(trait_id, param.name).to_string())
                .collect();
            let own_names: Vec<String> = sig
                .generics
                .iter()
                .map(|param| self.decl_text(param.name).to_string())
                .collect();
            let use_self_ty = Some(impl_self_ty.clone());
            // The receiver this use binds, instantiated. `None` when the declaration takes no
            // receiver, which keeps the published signature comparable with A3b's body signature.
            let receiver_self_ty = bound_receiver_ty(
                sig.receiver.as_ref(),
                self.instantiate_ty(&impl_self_ty, &map),
            );
            self.publish_callable_env(PublishedEnv {
                call_expr,
                body: trait_body,
                self_ty: Some(impl_self_ty.clone()),
                impl_names: &trait_names,
                own_names: &own_names,
                own_is_method: true,
                map: &env_map,
            });
            if matches!(sig.receiver, Some(hir::Receiver::RefMut))
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
            let previous_self = self.enter_self_scope(impl_self_ty);
            let params_ty: Vec<Ty> = sig
                .params
                .iter()
                .map(|p| {
                    let ty = self.convert_hir_type(p.ty);
                    self.instantiate_ty(&ty, &map)
                })
                .collect();
            let ret_ty = match sig.ret {
                hir::RetTy::Unit => Ty::Primitive(Primitive::Unit),
                hir::RetTy::Ty(t) => {
                    let ty = self.convert_hir_type(t);
                    self.instantiate_ty(&ty, &map)
                }
                hir::RetTy::Never(_) => Ty::Never,
            };
            self.exit_self_scope(previous_self);

            // AS3 Boundary 2: the same selection, published so an engine can CONSUME it rather
            // than re-derive it. Receiver ADJUSTMENT (what the call site did) and receiver BINDING
            // (what the callable binds) are separate fields: they correlate here, but they are
            // different authorities and AS4 asks about the binding side.
            let receiver_binding = match sig.receiver {
                Some(hir::Receiver::Value) => ReceiverBinding::ByValue,
                Some(hir::Receiver::Ref) => ReceiverBinding::Shared,
                Some(hir::Receiver::RefMut) => ReceiverBinding::Exclusive,
                None => ReceiverBinding::None,
            };
            let use_bindings =
                Self::env_bindings(&use_self_ty, &trait_names, &own_names, true, &env_map);
            self.publish_named_use(
                call_expr,
                trait_body,
                use_bindings,
                receiver_adjustment_for(receiver_derefs, outermost_ref_is_mut, receiver_binding),
                receiver_binding,
                CallableSigTy {
                    // AS3 Boundary 2 hardening: a real method's A3b body signature carries its
                    // receiver, so publishing `None` here made the §3.4 invariant unenforceable —
                    // the two signatures would disagree on every method. The instantiated `Self`
                    // is the receiver this use binds.
                    receiver: receiver_self_ty.clone(),
                    params: params_ty.clone(),
                    ret: ret_ty.clone(),
                },
                DispatchProvenance::Qualified { trait_item: None },
            );

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
            return ret_ty;
        }

        if let Some((def, _, mut map, impl_self_ty, impl_item_id)) = selected {
            // CD-358: every name below — the method's own generic parameters, and the parameter
            // and return TYPES — is a span into the impl's file.
            // WP-C4.7-8.4: the candidate's `map` carries only the IMPL's generic parameters. A
            // method may declare its OWN (`02:64` puts `GenericParams?` on every `FunctionSig`,
            // and `02:120` makes an impl item a `Function`), and those need a fresh inference
            // variable PER CALL SITE — otherwise the signature is used with `U` still a rigid
            // `Ty::Param` and every argument fails to unify against it ("expected 'U', found …").
            // The associated-function path already did exactly this; only the method path did not.
            if let Some(args) = turbofish {
                self.validate_generic_arity(def.sig.generics.len(), args.args.len(), call_span);
                for (param, arg) in def.sig.generics.iter().zip(&args.args) {
                    let ty = match arg {
                        hir::GenericArg::Type(ty) => self.convert_hir_type(*ty),
                        _ => Ty::Error,
                    };
                    map.insert(self.decl_text(param.name).to_string(), ty);
                }
            } else {
                for param in &def.sig.generics {
                    let infer = self.new_type_var();
                    map.insert(self.decl_text(param.name).to_string(), infer);
                }
            }
            // WP-C4.7-8.4: record this call site's METHOD-level instantiation for MIR
            // monomorphisation, keyed by the method-call expression — the same mechanism C4.5c
            // uses for top-level generic fns, which had no method equivalent. Recorded in the
            // method's own declaration order, and only when the method actually declares
            // parameters, so non-generic methods add no entries.
            // A3c-S: the full environment. `map` already carries the IMPL's parameters from
            // candidate selection plus the method's own — everything DEV-176 needs was computed
            // here and thrown away, except for the positional slice above.
            let impl_generics = match &self.hir.item(impl_item_id).kind {
                hir::ItemKind::Impl { generics, .. } => generics.clone(),
                _ => Vec::new(),
            };
            let env_map = map.clone();
            let env_self = impl_self_ty.clone();
            let env_generics = def.sig.generics.clone();
            let impl_names: Vec<String> = impl_generics
                .iter()
                .map(|param| self.item_text(impl_item_id, param.name).to_string())
                .collect();
            let own_names: Vec<String> = env_generics
                .iter()
                .map(|param| self.decl_text(param.name).to_string())
                .collect();
            let use_self_ty = Some(env_self.clone());
            let receiver_self_ty = bound_receiver_ty(
                def.sig.receiver.as_ref(),
                self.instantiate_ty(&env_self, &map),
            );
            self.publish_callable_env(PublishedEnv {
                call_expr,
                body: def.body,
                self_ty: Some(env_self),
                impl_names: &impl_names,
                own_names: &own_names,
                own_is_method: true,
                map: &env_map,
            });
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
            let previous_self = self.enter_self_scope(impl_self_ty);
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
            self.exit_self_scope(previous_self);

            // AS3 Boundary 2: the same selection, published so an engine can CONSUME it rather
            // than re-derive it. Receiver ADJUSTMENT (what the call site did) and receiver BINDING
            // (what the callable binds) are separate fields: they correlate here, but they are
            // different authorities and AS4 asks about the binding side.
            let receiver_binding = match def.sig.receiver {
                Some(hir::Receiver::Value) => ReceiverBinding::ByValue,
                Some(hir::Receiver::Ref) => ReceiverBinding::Shared,
                Some(hir::Receiver::RefMut) => ReceiverBinding::Exclusive,
                None => ReceiverBinding::None,
            };
            let use_bindings =
                Self::env_bindings(&use_self_ty, &impl_names, &own_names, true, &env_map);
            self.publish_named_use(
                call_expr,
                def.body,
                use_bindings,
                receiver_adjustment_for(receiver_derefs, outermost_ref_is_mut, receiver_binding),
                receiver_binding,
                CallableSigTy {
                    // AS3 Boundary 2 hardening: a real method's A3b body signature carries its
                    // receiver, so publishing `None` here made the §3.4 invariant unenforceable —
                    // the two signatures would disagree on every method. The instantiated `Self`
                    // is the receiver this use binds.
                    receiver: receiver_self_ty.clone(),
                    params: params_ty.clone(),
                    ret: ret_ty.clone(),
                },
                DispatchProvenance::Inherent,
            );

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
            self.core_method_signature(&receiver_ty, &name_str, name_span)
        {
            // **AS3 Boundary 4: a core container that compares its elements.**
            //
            // `vec.contains(&x)`, `set.insert(v)`, `map.get(k)` and friends run `Eq::eq` on the
            // ELEMENT when it is a user nominal — the interpreter's `language_equal`. That site had
            // no expression id and so scanned for a member named `eq`; publishing here gives it
            // one, keyed on the container call itself.
            self.publish_core_element_eq_use(call_expr, &receiver_ty, &name_str);
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
                    .with_code("E0304"),
                );
            } else if let Ty::Param(p_name) = &receiver_ty {
                // DEV-DISPLAY-DISPATCH: on a generic parameter, "not found" is the wrong story.
                // The method exists; the parameter is simply not bounded by the trait that
                // declares it, and the fix is to write that bound. Naming the trait is derived
                // from the traits actually in scope — nothing here keys on a method name.
                let providers = self.traits_declaring_method(&name_str);
                let mut diagnostic = if providers.is_empty() {
                    Diagnostic::error(
                        format!("method '{name_str}' not found for type '{p_name}'"),
                        call_span,
                    )
                    .with_code("E0302")
                    .with_label(format!(
                        "no trait in scope declares a method named '{name_str}'"
                    ))
                } else {
                    Diagnostic::error(
                        format!(
                            "method '{}' requires the bound '{}: {}'",
                            name_str, p_name, providers[0]
                        ),
                        call_span,
                    )
                    .with_code("E0302")
                    .with_label(format!(
                        "'{p_name}' has no bound that declares '{name_str}'"
                    ))
                };
                if providers.len() > 1 {
                    diagnostic = diagnostic.with_note(format!(
                        "'{}' is declared by: {}. Bound '{}' by the one this call means.",
                        name_str,
                        providers.join(", "),
                        p_name
                    ));
                }
                self.diags.push(diagnostic);
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

    /// DEV-DISPLAY-DISPATCH: every trait in scope — user-declared or compiler-known — that
    /// declares a method-callable `name`, named as it would be written in a bound.
    ///
    /// Both kinds are consulted, and neither is preferred: the point of the missing-bound
    /// diagnostic is to name the trait the programmer must write, and a compiler-known trait is
    /// written in a bound exactly like any other.
    fn traits_declaring_method(&self, name: &str) -> Vec<String> {
        let mut providers = Vec::new();
        for (index, item) in self.hir.items.iter().enumerate() {
            let trait_id = ItemId(index as u32);
            let hir::ItemKind::Trait {
                name: trait_name, ..
            } = &item.kind
            else {
                continue;
            };
            let declares = self.find_trait_method_sig(trait_id, name).is_some();
            if declares {
                let spelled = self.item_text(trait_id, *trait_name).to_string();
                if !providers.contains(&spelled) {
                    providers.push(spelled);
                }
            }
        }
        for core_trait in all_core_traits() {
            if core_trait_bound_method(core_trait, name).is_some() {
                let spelled = core_trait_source_name(core_trait).to_string();
                if !providers.contains(&spelled) {
                    providers.push(spelled);
                }
            }
        }
        providers
    }

    /// WP-C6.2b-F1: enforce member/field visibility. `defining_item` is the item that owns the
    /// member (the impl block for a method/associated fn, the struct/enum for a field/variant).
    /// A non-`pub` member is accessible only from its own defining module (private is exact-module,
    /// matching `resolve::item_is_visible_from`). Returns true if accessible; otherwise emits E0207
    /// and returns false.
    /// WP-C6.2b-F1: whether a struct field is declared `pub`. Missing struct/field → treat as
    /// public (an unrelated error already fired, or it is a tuple/core type).
    fn struct_field_is_pub(&self, struct_id: ItemId, field: &str) -> bool {
        if let hir::ItemKind::Struct { fields, .. } = &self.hir.item(struct_id).kind {
            for f in fields {
                if self.item_text(struct_id, f.name) == field {
                    return f.is_pub;
                }
            }
        }
        true
    }

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

    /// DEV-139: the full generic environment the current body is checked in — the impl's
    /// parameters followed by the function's own.
    ///
    /// Deferred trait-bound obligations capture this and replay it at drain time. DEV-067(a)
    /// introduced that capture so a caller's own `T: Ord` could discharge a callee's `T: Ord`,
    /// but it captured `current_fn_generics` ALONE, so an obligation raised inside
    /// `impl<T: Ord> Pair<T>` replayed against half its environment and failed. Capturing the
    /// combined list here means the drain needs no second field to restore.
    fn current_generic_env(&self) -> Vec<hir::GenericParam> {
        self.current_impl_generics
            .iter()
            .flatten()
            .chain(self.current_fn_generics.iter().flatten())
            .cloned()
            .collect()
    }

    /// DEV-139: whether the type parameter `param_name` declares `required`, anywhere in the
    /// generic environment the current body is checked in.
    ///
    /// **That environment is the impl's parameters PLUS the function's own**, and this is the one
    /// place that assembles it. Both bound questions — operator desugaring
    /// (`ty_satisfies_operator_bound`) and trait-bound satisfaction (`satisfies_bound`) — read it
    /// through here, so they cannot drift apart; before this each kept its own copy of the lookup
    /// and both consulted `current_fn_generics` alone. WP-C6.2b-F5 had already brought impl-head
    /// generics into scope for method bodies via `current_impl_generics`; the bound lookups simply
    /// never asked. So
    ///
    /// ```stark
    /// impl<T: Ord> Pair<T> {
    ///     fn larger(&self) -> &T { if self.a > self.b { &self.a } else { &self.b } }
    /// }
    /// ```
    ///
    /// was refused E0500 "type 'T' does not satisfy operator trait 'Ord'" while the identical
    /// comparison in a free `fn largest<T: Ord>` was accepted — the bound was declared, just not
    /// looked at.
    ///
    /// Impl generics are searched FIRST only for readability; a method may not redeclare an
    /// impl-level parameter name, so the two sets are disjoint and order cannot change the answer.
    /// Whether generic parameter `param_name` carries a bound denoting the CORE trait `required`.
    ///
    /// **DEV-171: by resolved identity, not by spelling.** The operator path compared
    /// `text(bound.path.span)` against `"Eq"`, so an unrelated trait imported under that name
    /// authorised `==`:
    ///
    /// ```text
    /// mod fake { pub trait Eq { fn unrelated(&self) -> Int32; } }
    /// use fake::Eq;
    /// fn compare<T: Eq>(a: T, b: T) -> Bool { a == b }   // was ACCEPTED
    /// ```
    ///
    /// Written qualified (`T: fake::Eq`) the same program was rejected — the tell that the answer
    /// depended on how the bound was spelled. Operators dispatch to the CANONICAL Core trait
    /// (03-Type-System.md, "Operators and Traits"), so only that trait discharges the obligation.
    ///
    /// **This is deliberately separate from [`Self::param_declares_bound`].** That one answers a
    /// different question — "does this parameter carry the bound being discharged", where the
    /// bound may be any user trait — and folding the two together made every qualified user-trait
    /// bound stop satisfying anything, because a user trait is not a Core trait. Caught by
    /// `dev_bound_trait_identity::a_qualified_bound_forwards_through_nested_generics` on CI.
    fn param_declares_core_bound(&self, param_name: &str, required: &str) -> bool {
        let Some(required) = crate::resolve::resolve_core_trait(required) else {
            return false;
        };
        self.current_impl_generics
            .iter()
            .flatten()
            .chain(self.current_fn_generics.iter().flatten())
            .any(|param| {
                self.text(param.name) == param_name
                    && param.bounds.iter().any(|bound| {
                        hir::resolved_bound_trait(self.hir, bound)
                            == Some(hir::BoundTrait::Core(required))
                    })
            })
    }

    fn require_operator_bound(&mut self, ty: &Ty, required: &str, span: Span) {
        let ty = self.resolve(ty);
        let satisfied = self.ty_satisfies_operator_bound(&ty, required);
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
    fn record_display_plan(&mut self, root: ExprId, ty: Ty) {
        self.display_plans.push(DeferredDisplayPlan {
            root,
            ty,
            generic_scope: (
                self.current_fn_generics.clone(),
                self.current_impl_generics.clone(),
            ),
        });
    }

    fn publish_display_uses(&mut self, root: ExprId, ty: &Ty, span: Span) {
        self.walk_display_ty(root, ty, DisplayPath::default(), span, 0);
    }

    fn walk_display_ty(
        &mut self,
        root: ExprId,
        ty: &Ty,
        path: DisplayPath,
        span: Span,
        depth: u32,
    ) {
        // A displayable type is a finite tree, but `Ty` is produced by inference and a defect
        // elsewhere should not become a stack overflow here.
        if depth > 64 {
            return;
        }
        let ty = self.resolve(ty);
        match &ty {
            // A reference renders as its referent: `Display::fmt` borrows anyway.
            Ty::Ref { inner, .. } => {
                let inner = (**inner).clone();
                self.walk_display_ty(root, &inner, path, span, depth + 1);
            }
            Ty::Tuple(elems) => {
                for (index, elem) in elems.clone().into_iter().enumerate() {
                    let step = DisplayStep::TupleField(index as u32);
                    self.walk_display_ty(root, &elem, path.child(step), span, depth + 1);
                }
            }
            Ty::Array(elem, _) => {
                let elem = (**elem).clone();
                let next = path.child(DisplayStep::ArrayElement);
                self.walk_display_ty(root, &elem, next, span, depth + 1);
            }
            Ty::Slice(elem) => {
                let elem = (**elem).clone();
                let next = path.child(DisplayStep::SliceElement);
                self.walk_display_ty(root, &elem, next, span, depth + 1);
            }
            Ty::Core(CoreType::Vec, args) => {
                if let Some(elem) = args.first().cloned() {
                    let next = path.child(DisplayStep::VecElement);
                    self.walk_display_ty(root, &elem, next, span, depth + 1);
                }
            }
            Ty::Core(CoreType::Option, args) => {
                if let Some(inner) = args.first().cloned() {
                    let next = path.child(DisplayStep::OptionSome);
                    self.walk_display_ty(root, &inner, next, span, depth + 1);
                }
            }
            Ty::Core(CoreType::Result, args) => {
                let args = args.clone();
                if let Some(ok) = args.first().cloned() {
                    let next = path.child(DisplayStep::ResultOk);
                    self.walk_display_ty(root, &ok, next, span, depth + 1);
                }
                if let Some(err) = args.get(1).cloned() {
                    let next = path.child(DisplayStep::ResultErr);
                    self.walk_display_ty(root, &err, next, span, depth + 1);
                }
            }
            // **STOP.** A user nominal with a `Display` impl renders through it and no further.
            Ty::Struct(..) | Ty::Enum(..) => self.publish_display_static(root, &ty, path, span),
            // A generic parameter's body is not knowable here — `show<T: Display>` is checked once
            // with `T` unbound, and one `show` may be instantiated at several types. The obligation
            // is fixed, and that is what `Bound` records (§3).
            Ty::Param(name) => {
                let name = name.clone();
                self.publish_display_bound(root, &name, path);
            }
            // Primitives, `String`, `Ordering`, `IOError` — rendered by the engines themselves,
            // with no user callable to name.
            _ => {}
        }
    }

    fn publish_display_static(&mut self, root: ExprId, ty: &Ty, path: DisplayPath, span: Span) {
        let Some((impl_item, member, body, substitution)) =
            self.operator_impl_member(ty, "Display", "fmt")
        else {
            return;
        };
        let Some((receiver, params, ret)) = self.declared_member_signature(impl_item, member)
        else {
            return;
        };
        let use_ = CallableUse {
            selection: CalleeSelection::Static {
                declaration: CallableDeclId::ImplMember { impl_item, member },
                body,
            },
            environment: GenericEnvironment::Static(self.impl_dispatch_bindings(impl_item, ty)),
            // `Display::fmt(&self)` borrows; the renderer holds the value and lends it.
            receiver_adjustment: ReceiverAdjustment::Shared { derefs: 0 },
            receiver_binding: ReceiverBinding::Shared,
            // §3.4: the INSTANTIATED signature, so `impl<T> Display for W<T>` publishes `&W<Int32>`
            // rather than the declaration's `&W<T>`.
            signature: CallableSigTy {
                receiver: receiver
                    .as_ref()
                    .map(|ty| self.instantiate_ty(ty, &substitution)),
                params: params
                    .iter()
                    .map(|ty| self.instantiate_ty(ty, &substitution))
                    .collect(),
                ret: self.instantiate_ty(&ret, &substitution),
            },
            provenance: DispatchProvenance::CoreTrait {
                core: hir::CoreTrait::Display,
            },
        };
        let _ = span;
        let id = self.publish_callable_use(root, use_);
        self.display_uses.insert((root, path), id);
    }

    fn publish_display_bound(&mut self, root: ExprId, param_name: &str, path: DisplayPath) {
        let candidates = self.bound_method_candidates(param_name, "fmt");
        let Some(BoundMethod::Core {
            core_trait: core @ hir::CoreTrait::Display,
            method: contract,
            ..
        }) = candidates.into_iter().find(|c| {
            matches!(
                c,
                BoundMethod::Core {
                    core_trait: hir::CoreTrait::Display,
                    ..
                }
            )
        })
        else {
            return;
        };
        let self_ty = Ty::Param(param_name.to_string());
        let ret = match contract.ret {
            None => Ty::Primitive(Primitive::Unit),
            Some(term) => self.contract_ty_to_ty(term, &self_ty, &[]),
        };
        let use_ = CallableUse {
            selection: CalleeSelection::Bound {
                trait_: hir::BoundTrait::Core(core),
                member: contract.name.to_string(),
                self_ty: self_ty.clone(),
                trait_args: Vec::new(),
                method_args: Vec::new(),
            },
            environment: GenericEnvironment::FromBoundSelection,
            receiver_adjustment: ReceiverAdjustment::Shared { derefs: 0 },
            receiver_binding: ReceiverBinding::Shared,
            signature: CallableSigTy {
                receiver: bound_receiver_ty(contract.receiver.as_ref(), self_ty),
                params: Vec::new(),
                ret,
            },
            provenance: DispatchProvenance::Bound {
                trait_: hir::BoundTrait::Core(core),
            },
        };
        let id = self.publish_callable_use(root, use_);
        self.display_uses.insert((root, path), id);
    }

    /// The impl's generic parameters bound to the rendered type's arguments — the "instantiated
    /// environment" requirement the Iterator hardening established after publishing an empty one.
    fn impl_dispatch_bindings(&mut self, impl_item: ItemId, ty: &Ty) -> Vec<(GenericBinder, Ty)> {
        let hir::ItemKind::Impl {
            self_ty, generics, ..
        } = &self.hir.item(impl_item).kind
        else {
            return Vec::new();
        };
        let (self_ty, generics) = (*self_ty, generics.clone());
        let parametric = self.impl_self_ty_with_args(impl_item, self_ty);
        let Some(map) = self.match_impl_type(&parametric, ty, &generics) else {
            return Vec::new();
        };
        let impl_names: Vec<String> = generics
            .iter()
            .map(|param| self.item_text(impl_item, param.name).to_string())
            .collect();
        // `env_bindings`, not a second construction of the same list: `Display::fmt` declares no
        // generics of its own, so the method half is empty and `Self` is the rendered type.
        Self::env_bindings(&Some(ty.clone()), &impl_names, &[], true, &map)
    }

    /// The `Eq::eq` a core container method runs on its elements, published against the container
    /// call.
    ///
    /// The method list is explicit rather than "anything that might compare": publishing a use the
    /// renderer never executes would make the totality claim false, which is the same discipline
    /// the `Display` walk's STOP rule follows.
    fn publish_core_element_eq_use(&mut self, call_expr: ExprId, receiver: &Ty, method: &str) {
        // **Publish for every container method, not a hand-listed subset.**
        //
        // The first version listed the methods believed to compare elements and omitted `get_mut`,
        // so `map.get_mut(&k)` fell back to STRUCTURAL comparison and silently retrieved the wrong
        // entry — caught by `hash_collections_use_language_eq_for_keys`, whose whole purpose is a
        // user `Eq` that disagrees with structural equality.
        //
        // The asymmetry decides it: an unused entry costs a table slot, while a missing one is a
        // wrong answer. That is the opposite of the Display walk's STOP rule, and deliberately so —
        // there, over-publishing would falsify a claim about what the renderer executes; here the
        // claim is only "if this call compares elements, this is the body", which stays true
        // whether or not it does.
        let _ = method;
        let mut receiver = self.resolve(receiver);
        while let Ty::Ref { inner, .. } = receiver {
            receiver = self.resolve(&inner);
        }
        // The compared type: a map compares KEYS, a set and a Vec compare elements.
        let element = match &receiver {
            Ty::Core(CoreType::HashMap, args) => args.first().cloned(),
            Ty::Core(CoreType::HashSet | CoreType::Vec, args) => args.first().cloned(),
            _ => None,
        };
        let Some(element) = element else { return };
        // `publish_operator_use` already handles "not a user nominal" and "no `Eq` impl" by
        // publishing nothing, and handles a bounded parameter through DEV-191's branch.
        self.publish_operator_use(call_expr, &element, "Eq", "eq", hir::CoreTrait::Eq);
    }

    fn publish_bound_operator_use(
        &mut self,
        expr_id: ExprId,
        param_name: &str,
        method: &str,
        core: hir::CoreTrait,
    ) {
        let candidates = self.bound_method_candidates(param_name, method);
        let Some(BoundMethod::Core {
            core_trait,
            method: contract,
            trait_args,
        }) = candidates
            .into_iter()
            .find(|c| matches!(c, BoundMethod::Core { core_trait, .. } if *core_trait == core))
        else {
            // No such bound in scope. Arithmetic on a `T: Num` reaches here and correctly
            // publishes nothing: `Num` is compiler-known and primitives-only, so there is no
            // user body for a call site to name.
            return;
        };
        let self_ty = Ty::Param(param_name.to_string());
        let trait_arg_tys: Vec<Ty> = trait_args
            .iter()
            .map(|ty| self.convert_hir_type(*ty))
            .collect();
        let params: Vec<Ty> = contract
            .params
            .iter()
            .map(|term| self.contract_ty_to_ty(*term, &self_ty, &trait_arg_tys))
            .collect();
        let ret = match contract.ret {
            None => Ty::Primitive(Primitive::Unit),
            Some(term) => self.contract_ty_to_ty(term, &self_ty, &trait_arg_tys),
        };
        let use_ = CallableUse {
            selection: CalleeSelection::Bound {
                trait_: hir::BoundTrait::Core(core_trait),
                member: contract.name.to_string(),
                self_ty: self_ty.clone(),
                trait_args: trait_arg_tys,
                // A core trait's contract cannot declare method-level generics (DEV-188).
                method_args: Vec::new(),
            },
            environment: GenericEnvironment::FromBoundSelection,
            receiver_adjustment: ReceiverAdjustment::Shared { derefs: 0 },
            receiver_binding: ReceiverBinding::Shared,
            signature: CallableSigTy {
                receiver: bound_receiver_ty(contract.receiver.as_ref(), self_ty),
                params,
                ret,
            },
            provenance: DispatchProvenance::Bound {
                trait_: hir::BoundTrait::Core(core_trait),
            },
        };
        self.publish_callable_use(expr_id, use_);
    }

    fn publish_operator_use(
        &mut self,
        expr_id: ExprId,
        operand: &Ty,
        trait_name: &str,
        method: &str,
        core: hir::CoreTrait,
    ) {
        let operand = self.resolve(operand);
        // **AS3 Boundary 4 (DEV-191): an operator on a BOUNDED GENERIC PARAMETER.**
        //
        // `a == b` inside `fn same<T: Eq>(a: T, b: T)` published nothing at all — this guard
        // returned on `Ty::Param`. So MIR, which sees the monomorphised `P` and lowers a user
        // `Eq::eq` call, had no published record to consume and fell back to scanning impls by
        // name. It is the same missing binding time step 2 found for method calls, on the operator
        // path: the trait is fixed here, the body only once `T` is instantiated.
        if let Ty::Param(param_name) = &operand {
            let param_name = param_name.clone();
            self.publish_bound_operator_use(expr_id, &param_name, method, core);
            return;
        }
        if !matches!(operand, Ty::Struct(..) | Ty::Enum(..)) {
            return;
        }
        let Some((impl_item, member, body, substitution)) =
            self.operator_impl_member(&operand, trait_name, method)
        else {
            return;
        };
        // **The signature is READ from the declaration, not assumed.** `Eq::eq` returns `Bool` and
        // `Ord::cmp` returns `Ordering`, but writing those in would be this packet's own defect —
        // a second answer to what the callable's signature is, which §3.4's invariant then has to
        // reconcile against the body's.
        let Some((receiver, params, ret)) = self.declared_member_signature(impl_item, member)
        else {
            return;
        };
        // **DEV-201: an operator on a GENERIC impl published an empty environment.**
        //
        // `Static(Vec::new())` was written here unconditionally. For `impl Eq for Point` that is
        // correct — there is nothing to bind. For `impl<T> Eq for W<T>` it is a body running with
        // `T` unbound, which is AS3 criterion 2's exact prohibition. Nothing observed it until
        // DEV-121's receiver boundary read `callable_types[body].receiver` and found
        // `&W<Param(\"T\")>` with no `T` in scope.
        let environment = self.impl_dispatch_bindings(impl_item, &operand);
        let use_ = CallableUse {
            selection: CalleeSelection::Static {
                declaration: CallableDeclId::ImplMember { impl_item, member },
                body,
            },
            environment: GenericEnvironment::Static(environment),
            // `Eq::eq(&self, &other)` and `Ord::cmp(&self, &other)` both borrow: the receiver binds
            // shared, and the call site takes a shared borrow of an owned operand — zero derefs.
            receiver_adjustment: ReceiverAdjustment::Shared { derefs: 0 },
            receiver_binding: ReceiverBinding::Shared,
            // AS3 Boundary 2 §3.4: the publication records the INSTANTIATED signature, so a
            // consumer reading it sees `&W<Int32>` rather than the declaration's `&W<T>`.
            signature: CallableSigTy {
                receiver: receiver
                    .as_ref()
                    .map(|ty| self.instantiate_ty(ty, &substitution)),
                params: params
                    .iter()
                    .map(|ty| self.instantiate_ty(ty, &substitution))
                    .collect(),
                ret: self.instantiate_ty(&ret, &substitution),
            },
            provenance: DispatchProvenance::CoreTrait { core },
        };
        self.publish_callable_use(expr_id, use_);
    }

    /// The declared signature of an impl member, converted — receiver, parameters and result.
    ///
    /// AS3 Boundary 3: the operator publication reads the signature it publishes rather than
    /// asserting one, so `callable_uses` and `callable_types` describe the same declaration.
    #[allow(clippy::type_complexity)]
    fn declared_member_signature(
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

    /// Publish a `CallableUse::Bound` for a call resolved through a generic parameter's bound.
    ///
    /// AS3 Boundary 4 step 2, deliberately landed **before** Display so the late-bound mechanism is
    /// proved on an ordinary `fn f<T: Speak>(x: T) { x.speak(); }` rather than tangled with
    /// recursive formatting.
    fn publish_bound_use(
        &mut self,
        call_expr: ExprId,
        candidate: &BoundMethod,
        param_name: &str,
        method: &str,
        ret: &Ty,
        method_args: Vec<Ty>,
    ) {
        let (trait_, receiver_form, params) = match candidate {
            BoundMethod::User { trait_id, sig } => (
                hir::BoundTrait::User(*trait_id),
                sig.receiver,
                sig.params.iter().map(|p| p.ty).collect::<Vec<_>>(),
            ),
            BoundMethod::Core {
                core_trait, method, ..
            } => {
                // A core trait's contract is declared, not written in HIR, so its parameter types
                // are not `TypeId`s. The signature is published with the receiver and result only;
                // the specialiser produces the full instantiated signature from the impl it picks.
                let receiver_self = Ty::Param(param_name.to_string());
                let use_ = CallableUse {
                    selection: CalleeSelection::Bound {
                        trait_: hir::BoundTrait::Core(*core_trait),
                        member: method.name.to_string(),
                        self_ty: receiver_self.clone(),
                        trait_args: Vec::new(),
                        // Always empty, and that is the answer rather than a gap: a core trait's
                        // contract is `ContractTy`, which cannot declare method-level generics.
                        method_args,
                    },
                    environment: GenericEnvironment::FromBoundSelection,
                    receiver_adjustment: ReceiverAdjustment::None,
                    receiver_binding: match method.receiver {
                        Some(hir::Receiver::Value) => ReceiverBinding::ByValue,
                        Some(hir::Receiver::Ref) => ReceiverBinding::Shared,
                        Some(hir::Receiver::RefMut) => ReceiverBinding::Exclusive,
                        None => ReceiverBinding::None,
                    },
                    signature: CallableSigTy {
                        receiver: bound_receiver_ty(method.receiver.as_ref(), receiver_self),
                        params: Vec::new(),
                        ret: ret.clone(),
                    },
                    provenance: DispatchProvenance::Bound {
                        trait_: hir::BoundTrait::Core(*core_trait),
                    },
                };
                self.publish_callable_use(call_expr, use_);
                return;
            }
        };
        let receiver_self = Ty::Param(param_name.to_string());
        let params: Vec<Ty> = params
            .into_iter()
            .map(|id| self.convert_hir_type(id))
            .collect();
        let use_ = CallableUse {
            selection: CalleeSelection::Bound {
                trait_,
                member: method.to_string(),
                self_ty: receiver_self.clone(),
                trait_args: Vec::new(),
                // DEV-188: this call site's binding of the METHOD's own generics, from the
                // turbofish or inferred from the arguments.
                method_args,
            },
            environment: GenericEnvironment::FromBoundSelection,
            receiver_adjustment: ReceiverAdjustment::None,
            receiver_binding: match receiver_form {
                Some(hir::Receiver::Value) => ReceiverBinding::ByValue,
                Some(hir::Receiver::Ref) => ReceiverBinding::Shared,
                Some(hir::Receiver::RefMut) => ReceiverBinding::Exclusive,
                None => ReceiverBinding::None,
            },
            signature: CallableSigTy {
                receiver: bound_receiver_ty(receiver_form.as_ref(), receiver_self),
                params,
                ret: ret.clone(),
            },
            provenance: DispatchProvenance::Bound { trait_ },
        };
        self.publish_callable_use(call_expr, use_);
    }

    /// Publish the `Iterator::next` use a `for` loop selects.
    ///
    /// Uses `resolve_user_iterator` — the SAME selection the element type came from — so there is
    /// one answer to "which `next` does this loop run", not two that must agree.
    fn publish_iterator_use(&mut self, for_expr: ExprId, iter_ty: &Ty) {
        let iter_ty = self.resolve(iter_ty);
        if !matches!(iter_ty, Ty::Struct(..) | Ty::Enum(..)) {
            return;
        }
        let Some(selection) = self.resolve_user_iterator(&iter_ty) else {
            return;
        };
        let (impl_item, member, body) = (selection.impl_item, selection.member, selection.body);
        let Some((receiver, params, ret)) = self.declared_member_signature(impl_item, member)
        else {
            return;
        };
        // Instantiated against the impl's substitution: `impl<T> Iterator for Repeat<T>` publishes
        // a `next` returning `Option<Int32>` for `Repeat<Int32>`, not `Option<T>`.
        let receiver = receiver.map(|ty| self.instantiate_ty(&ty, &selection.substitutions));
        let params: Vec<Ty> = params
            .iter()
            .map(|ty| self.instantiate_ty(ty, &selection.substitutions))
            .collect();
        let ret = self.instantiate_ty(&ret, &selection.substitutions);
        let use_ = CallableUse {
            selection: CalleeSelection::Static {
                declaration: CallableDeclId::ImplMember { impl_item, member },
                body,
            },
            // The impl's generic environment, RETAINED. Publishing an empty one here was the
            // second defect the hardening review found.
            environment: GenericEnvironment::Static(selection.bindings),
            // `Iterator::next(&mut self)` advances the iterator, so the loop takes an exclusive
            // borrow of the iterator place — no dereferencing.
            receiver_adjustment: ReceiverAdjustment::Exclusive { derefs: 0 },
            receiver_binding: ReceiverBinding::Exclusive,
            signature: CallableSigTy {
                receiver,
                params,
                ret,
            },
            provenance: DispatchProvenance::CoreTrait {
                core: hir::CoreTrait::Iterator,
            },
        };
        self.publish_callable_use(for_expr, use_);
    }

    /// The impl that supplies operator trait `required` for a user nominal, and the member index
    /// and body of the method that implements it.
    ///
    /// **AS3 Boundary 3.** `ty_satisfies_operator_bound` already performs this scan — it walks every
    /// impl looking for one whose trait path reads `"Eq"`/`"Ord"` and whose self type matches — and
    /// then returns a `bool`, discarding the impl it just found. So the checker *does* select for
    /// `==` and `<`; it throws the selection away and both engines find it again.
    ///
    /// That makes this a **fourth** scan of the same shape, after `Interpreter::find_method` and
    /// `FnLowerer::find_impl_fn`. `AS0-CALLABLE-EXECUTION-SITE-INVENTORY.md` counted three because
    /// it looked for algorithms that *return* a callable; this one answers a narrower question and
    /// drops the answer, which is how it escaped the count.
    fn operator_impl_member(
        &self,
        ty: &Ty,
        required: &str,
        method: &str,
    ) -> Option<(ItemId, u32, BlockId, HashMap<String, Ty>)> {
        for (idx, item) in self.hir.items.iter().enumerate() {
            let impl_id = ItemId(idx as u32);
            let hir::ItemKind::Impl {
                trait_: Some(trait_ref),
                self_ty,
                generics,
                items,
            } = &item.kind
            else {
                continue;
            };
            if self.item_text(impl_id, trait_ref.path.span) != required {
                continue;
            }
            // **The substitution is RETURNED, not discarded.** It was computed here to decide
            // whether the impl applies at all, and thrown away — so the operator publication had
            // no way to say what `impl<T> Eq for W<T>` binds `T` to, and published an empty
            // environment for a generic impl.
            let Some(substitution) = self.match_impl_type(
                &self.impl_self_ty_with_args(impl_id, *self_ty),
                ty,
                generics,
            ) else {
                continue;
            };
            for (member, impl_item) in items.iter().enumerate() {
                if let hir::ImplItem::Fn { def, .. } = impl_item {
                    if self.item_text(impl_id, def.sig.name) == method {
                        return Some((impl_id, member as u32, def.body, substitution));
                    }
                }
            }
        }
        None
    }

    fn ty_satisfies_operator_bound(&self, ty: &Ty, required: &str) -> bool {
        match ty {
            // DEV-075 (owner specification decision, 2026-07-20). This gate is about the
            // OPERATOR, not the trait, and on primitives operators have built-in meaning
            // (03-Type-System, "Operators and Traits"). So primitive FLOATS keep `==` and `<`
            // here — IEEE comparison per CD-006 — even though CD-015 denies them the `Eq`/`Ord`
            // *traits*; that distinction lives in `satisfies_bound`, which gates generic bounds.
            // What DOES change: `Bool` loses ordering. `false < true` is definable, but Core v1
            // has no meaningful use for ordering truth values, and rejecting it is clearer than
            // inventing an order merely because one is technically available. `Char` is ordered,
            // by Unicode scalar value.
            Ty::Primitive(primitive) => match required {
                "Num" => is_numeric(*primitive),
                "Eq" => !matches!(primitive, Primitive::Unit),
                "Ord" => !matches!(primitive, Primitive::Unit | Primitive::Bool),
                _ => false,
            },
            Ty::Ref {
                mutable: false,
                inner,
            } if required == "Eq" || required == "Ord" => {
                let inner = self.resolve(inner);
                self.ty_satisfies_operator_bound(&inner, required)
            }
            Ty::Param(name) => self.param_declares_core_bound(name, required),
            // DEV-073 (WP-C4.7-5): a GENERIC impl satisfies a concrete instantiation's bound —
            // `impl<T> Eq for W<T>` satisfies `W<Int32>: Eq`. This used to demand
            // `types_equal(impl_self_ty, ty)`, an EXACT match, so the impl's written self type
            // `W<T>` never equalled `W<Int32>` and every operator on a generic nominal was
            // rejected E0500. The fix reuses `match_impl_type` — the same one-way unification
            // method resolution already uses for exactly this question, so operator bounds and
            // method calls now agree by construction instead of by coincidence.
            // DEV-069: the trait name written on each impl is read against that impl's own file.
            Ty::Struct(..) | Ty::Enum(..) => {
                self.hir.items.iter().enumerate().any(|(idx, item)| {
                    let impl_id = ItemId(idx as u32);
                    let hir::ItemKind::Impl {
                        trait_: Some(trait_ref),
                        self_ty,
                        generics,
                        ..
                    } = &item.kind
                    else {
                        return false;
                    };
                    self.item_text(impl_id, trait_ref.path.span) == required
                        && self
                            .match_impl_type(
                                &self.impl_self_ty_with_args(impl_id, *self_ty),
                                ty,
                                generics,
                            )
                            .is_some()
                })
            }
            Ty::Core(core_type, args) if required == "Eq" || required == "Ord" => {
                matches!(
                    core_type,
                    CoreType::Option | CoreType::Result | CoreType::Vec | CoreType::Box
                ) && args.iter().all(|arg| {
                    let arg = self.resolve(arg);
                    self.ty_satisfies_operator_bound(&arg, required)
                })
            }
            Ty::Infer(_) | Ty::Error => true,
            _ => false,
        }
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

    fn satisfies_bound(&mut self, ty: &Ty, bound: &hir::TraitRef) -> bool {
        let bound_name = self.text(bound.path.span).to_string();
        self.satisfies_bound_parts(ty, &bound_name, Some(bound.res), bound.args.clone())
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

// ------------------------------------------- WP-C7.9 Packet B: Core-trait implementation contracts --

// ------------------------------------------------- DEV-DISPLAY-DISPATCH: bounds as one surface --

/// The `CoreTrait` following `current` in declaration order, or `None` at the end of the enum.
///
/// This exists so [`all_core_traits`] can enumerate the compiler-known traits **without a list
/// that can fall behind the enum**. The match is total: adding a `CoreTrait` variant is a
/// compile error here, which is the no-wildcard discipline applied to an enumeration rather than
/// to a dispatch.
fn next_core_trait(current: hir::CoreTrait) -> Option<hir::CoreTrait> {
    use hir::CoreTrait as CT;
    let next = match current {
        CT::Copy => CT::Drop,
        CT::Drop => CT::Eq,
        CT::Eq => CT::Ord,
        CT::Ord => CT::Num,
        CT::Num => CT::Clone,
        CT::Clone => CT::Hash,
        CT::Hash => CT::Default,
        CT::Default => CT::Display,
        CT::Display => CT::Error,
        CT::Error => CT::From,
        CT::From => CT::Into,
        CT::Into => CT::TryFrom,
        CT::TryFrom => CT::Index,
        CT::Index => CT::IndexMut,
        CT::IndexMut => CT::Iterator,
        CT::Iterator => CT::FromIterator,
        CT::FromIterator => return None,
    };
    Some(next)
}

/// Every compiler-known trait, in declaration order.
fn all_core_traits() -> Vec<hir::CoreTrait> {
    let mut traits = vec![hir::CoreTrait::Copy];
    while let Some(next) = next_core_trait(*traits.last().expect("non-empty")) {
        traits.push(next);
    }
    traits
}

/// The ordinary trait identity a generic bound resolves to.
///
/// Core v1 has two kinds of trait, and until DEV-DISPLAY-DISPATCH only one of them was reachable
/// from a bound. A user trait has a `hir::ItemKind::Trait` declaration, and method resolution
/// found it by matching that item's name. A compiler-known trait has no declaration item at all,
/// so the same lookup silently produced nothing: `fn show<T: Display>(x: &T) { x.fmt() }` failed
/// with "method 'fmt' not found for type 'T'" even though `T: Display` had been *checked* as a
/// bound. Method visibility, in other words, depended on whether the trait happened to be
/// compiler-known — a second trait model rather than one.
///
/// DEV-BOUND-TRAIT-IDENTITY then moved the type itself into `hir`, alongside
/// `hir::resolved_bound_trait`, so the borrow checker consumes the SAME identity this pass does
/// rather than deriving its own from the bound's spelling.
use hir::BoundTrait;

/// One method a bound contributes, with enough of its declaration to check a call.
enum BoundMethod {
    User {
        trait_id: ItemId,
        sig: hir::FnSig,
    },
    Core {
        core_trait: hir::CoreTrait,
        /// Borrowed from [`core_trait_contract`], the same table user `impl` blocks are checked
        /// against. There is deliberately no second registry of Core-trait signatures: what a
        /// bound makes callable is by construction what an implementation must provide.
        method: &'static CoreTraitMethod,
        /// The bound's own written type arguments (`T: Into<Int32>`), for `ContractTy::TraitArg`.
        trait_args: Vec<hir::TypeId>,
    },
}

/// The method a compiler-known trait contributes to a bound, if it declares one by this name and
/// that method is callable with method syntax.
///
/// **`receiver.is_some()` is the whole filter.** `Default::default()` and `From::from(v)` have no
/// receiver, so no `x.default()` spelling exists to resolve; they are reached through their
/// qualified paths, which DEV-052 already handles. Nothing here keys on a method NAME — the
/// contract's own shape decides.
fn core_trait_bound_method(
    core_trait: hir::CoreTrait,
    name: &str,
) -> Option<&'static CoreTraitMethod> {
    let contract = core_trait_contract(core_trait)?;
    let methods: &'static [CoreTraitMethod] = contract.methods;
    methods
        .iter()
        .find(|method| method.name == name && method.receiver.is_some())
}

/// DEV-DISPLAY-DISPATCH: the receiver form a compiler-known trait declares for `name`, for the
/// passes outside the type checker that need it — the move checker most of all, which must know
/// that `Display::fmt` BORROWS before it decides whether `x.fmt()` consumed `x`.
///
/// Reads [`core_trait_contract`], the same table everything else does; there is one source for a
/// Core trait's signatures, not one per consumer.
pub fn core_trait_method_receiver(core_trait: hir::CoreTrait, name: &str) -> Option<hir::Receiver> {
    core_trait_bound_method(core_trait, name).and_then(|method| method.receiver)
}

/// The written type arguments of a trait reference (`T: Into<Int32>` → `[Int32]`), skipping the
/// argument forms that do not name a type.
fn trait_ref_type_args(bound: &hir::TraitRef) -> Vec<hir::TypeId> {
    let Some(args) = &bound.args else {
        return Vec::new();
    };
    args.args
        .iter()
        .filter_map(|arg| match arg {
            hir::GenericArg::Type(ty) => Some(*ty),
            // An associated-type binding constrains a projection, it does not fill an argument
            // position; a const or a tensor shape argument is not a type at all.
            hir::GenericArg::Const(_)
            | hir::GenericArg::Binding { .. }
            | hir::GenericArg::Shape(_) => None,
        })
        .collect()
}

fn is_cast_numeric(p: Primitive) -> bool {
    is_numeric(p) || matches!(p, Primitive::Float16 | Primitive::BFloat16)
}

fn type_is_sized(ty: &Ty) -> bool {
    match ty {
        Ty::Primitive(Primitive::Str) | Ty::Slice(_) => false,
        Ty::Ref { .. } => true,
        Ty::Tuple(types) => types.iter().all(type_is_sized),
        Ty::Array(element, _) => type_is_sized(element),
        Ty::Struct(_, arguments) | Ty::Enum(_, arguments) => arguments.iter().all(type_is_sized),
        Ty::Core(CoreType::Box, arguments) => arguments.first().is_some_and(type_is_sized),
        Ty::Core(_, arguments) => arguments.iter().all(type_is_sized),
        Ty::Fn { params, ret } => params.iter().all(type_is_sized) && type_is_sized(ret),
        Ty::Range(element) => type_is_sized(element),
        Ty::Extension(_) | Ty::Primitive(_) | Ty::Never | Ty::Param(_) | Ty::Infer(_) => true,
        Ty::Error => true,
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

impl<'a> TypeChecker<'a> {
    /// WP-C4.7-9 audit: whether a value of this type can be given to `print`/`println` — a
    /// standard-library `Display` type, or a user nominal with its own `Display` impl.
    /// DEV-134: `?` may propagate only into a return type that can actually receive it.
    ///
    /// `03-Type-System.md` defines `?` for `Result<T, E>`/`Option<T>` and Core v1 has no
    /// user-extensible `Try` trait and no conversion step at the propagation site. The rule is
    /// therefore EXACT compatibility, deliberately and conservatively:
    ///
    /// - `Result<_, E_in>?` in a function returning `Result<_, E_out>` requires `E_in == E_out`
    ///   under `types_equal`, the compiler's canonical equivalence;
    /// - `Option<_>?` in a function returning `Option<_>` is always fine (there is no payload on
    ///   `None` to relate);
    /// - mixing the two constructors in either direction is refused.
    ///
    /// An implicit `From` conversion is NOT introduced here. The specification does not scope
    /// one, so adding it would be new semantics rather than a repair; that question is recorded
    /// separately (see the DEV-134 ledger entry). Rejection is the conservative half and is what
    /// this implements.
    fn check_try_compatibility(&mut self, operand_ty: &Ty, ret_ty: &Ty, span: Span) {
        let operand = self.resolve(operand_ty);
        let ret = self.resolve(ret_ty);

        // Never cascade: an already-failed or still-undetermined type says nothing about `?`.
        if matches!(operand, Ty::Error) || matches!(ret, Ty::Error) {
            return;
        }
        if ty_contains_infer(&operand) || ty_contains_infer(&ret) {
            return;
        }

        let (Ty::Core(operand_ctor, operand_args), Ty::Core(ret_ctor, ret_args)) = (&operand, &ret)
        else {
            // Not a `?`-capable pair at all. The pre-existing E0006 checks in the `Try` arm
            // already reported that; adding a second diagnostic here would double-report.
            return;
        };
        if !matches!(operand_ctor, CoreType::Result | CoreType::Option)
            || !matches!(ret_ctor, CoreType::Result | CoreType::Option)
        {
            return;
        }

        // Constructor mismatch. This is the half that is easy to overlook: it produces exactly
        // the same type confusion as an error-type mismatch, because the propagated value's
        // variant tag (`None`) belongs to a different enum than the one the caller matches on.
        if operand_ctor != ret_ctor {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "'?' cannot propagate '{}' out of a function returning '{}'",
                        self.ty_to_string(&operand),
                        self.ty_to_string(&ret)
                    ),
                    span,
                )
                .with_code("E0006")
                .with_label("the propagated value and the return type are different types")
                .with_note(
                    "'?' performs no conversion in Core v1. Match on the operand and construct \
                     the returned type explicitly."
                        .to_string(),
                ),
            );
            return;
        }

        // Same constructor. Only `Result` carries an error type to relate.
        if *operand_ctor != CoreType::Result {
            return;
        }
        let (Some(err_in), Some(err_out)) = (operand_args.get(1), ret_args.get(1)) else {
            return;
        };
        let err_in = self.resolve(err_in);
        let err_out = self.resolve(err_out);
        if matches!(err_in, Ty::Error) || matches!(err_out, Ty::Error) {
            return;
        }
        if ty_contains_infer(&err_in) || ty_contains_infer(&err_out) {
            return;
        }
        if self.types_equal_inner(&err_in, &err_out, true) {
            return;
        }

        self.diags.push(
            Diagnostic::error(
                format!(
                    "'?' cannot propagate error type '{}' out of a function returning '{}'",
                    self.ty_to_string(&err_in),
                    self.ty_to_string(&ret)
                ),
                span,
            )
            .with_code("E0006")
            .with_label("error types must match exactly")
            .with_note(format!(
                "'?' performs no conversion in Core v1: it does not apply 'From', and an \
                 'impl From<{}> for {}' would not change this. Match on the operand and \
                 construct '{}' explicitly.",
                self.ty_to_string(&err_in),
                self.ty_to_string(&err_out),
                self.ty_to_string(&err_out)
            )),
        );
    }

    fn type_is_displayable(&self, ty: &Ty) -> bool {
        if standard_display_type(ty) {
            return true;
        }
        match ty {
            // Containers print elementwise in the reference implementation.
            Ty::Core(CoreType::Option | CoreType::Result | CoreType::Vec, args) => {
                args.iter().all(|a| self.type_is_displayable(a))
            }
            Ty::Tuple(elems) => elems.iter().all(|e| self.type_is_displayable(e)),
            // **DEV-206: an array is a value; a bare slice is not.**
            //
            // These shared an arm, which accepted the UNSIZED `[T]` — a type §6.6 says is never a
            // standalone value, and which the representation relation refuses at every boundary
            // for exactly that reason. So `println(v[0..2])` type-checked and then had no valid
            // runtime representation, while `println(&v[0..2])` — the form that *can* exist — was
            // rejected because no arm below matched a reference to a slice. The polarity was
            // reversed.
            Ty::Array(elem, _) => self.type_is_displayable(elem),
            // A slice is observed THROUGH a reference. `&[T]` is displayable exactly when `T` is,
            // which is the same elementwise rule the other containers use; nothing is invented for
            // a non-`Display` element. Deliberately shared references only — `&mut [T]` is not
            // broadened here, because DEV-206 is about the `[T]`/`&[T]` contradiction and nothing
            // in the standard rules currently implies the exclusive form.
            Ty::Ref {
                mutable: false,
                inner,
            } if matches!(inner.as_ref(), Ty::Slice(_)) => match inner.as_ref() {
                Ty::Slice(elem) => self.type_is_displayable(elem),
                _ => false,
            },
            Ty::Struct(..) | Ty::Enum(..) => self.ty_satisfies_operator_bound(ty, "Display"),
            Ty::Param(_) => true, // discharged by the caller's own bound
            _ => false,
        }
    }
}

/// Whether `ty` is `Copy`, given the set of `Copy`-eligible nominals.
///
/// **Published for WP-VALUE-REP-TOTAL A2.** The representation relation permits `&T` to be
/// represented by a bare `T` only when the POINTEE is `Copy`, and that must be the same answer the
/// checker uses — a second Copy predicate in the interpreter would be a second definition of move
/// behaviour, which is the disagreement WP-COPY-CANON exists to prevent.
pub fn is_copy_type_with(ty: &Ty, copy_types: &HashSet<ItemId>) -> bool {
    is_copy_with_impls(ty, copy_types)
}

impl TypeChecker<'_> {
    fn check_tensor_builtin_call(
        &mut self,
        builtin: Builtin,
        turbofish: Option<&hir::GenericArgs>,
        args: &[ExprId],
        span: Span,
    ) -> Ty {
        // AS6: the spelling table belonged to the extension, not to Core's checker — the same
        // criterion-2 shape the resolver's table had. `TensorBuiltin::op_name` is exhaustive, so a
        // new operation cannot reach here unnamed.
        let Builtin::Tensor(op) = builtin else {
            return Ty::Error;
        };
        let op_name = op.op_name();
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

        // AS6 packet 4B group 2C: Core's half is done — the operation is located, the call form
        // is validated, and every argument expression has been evaluated. Every dtype, shape,
        // device, schema and broadcasting decision from here on is the extension's.
        tensor_check::eval_tensor_op(
            self,
            op_name,
            descriptor,
            receiver.is_some(),
            turbofish,
            actual_ops,
            span,
        )
    }

    /// AS6 packet 4D-A: Core normalises the declaration — enters the generic scope, classifies
    /// each parameter, converts each written port type — and the extension decides whether what
    /// the declaration says is *allowed*. Staged rather than hoisted so that a conversion
    /// diagnostic cannot overtake the duplicate-name diagnostic for the same port.
    fn check_model_def(&mut self, _item_id: ItemId, def: &hir::ModelDef) {
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

    fn check_model_method_call(
        &mut self,
        model: &ModelTy,
        name: &str,
        args: &[ExprId],
        name_span: Span,
        call_span: Span,
    ) -> Ty {
        // AS6 packet 4B group 2C: a model's method surface, its `.predict(...)` calling
        // convention and its result shape are model semantics; Core keeps the HIR walk, the
        // declaration scope, the freshening and the argument evaluation.
        if !tensor_check::check_model_method_name(self, name, name_span) {
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

        if !tensor_check::check_model_predict_arity(self, inputs.len(), args.len(), call_span) {
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

        // Argument evaluation stays here and stays interleaved: the extension rule runs once per
        // argument, immediately after that argument is checked, so diagnostic order is unchanged.
        for (arg_expr_id, (expected_port_ty, port_decl_span)) in
            args.iter().zip(instantiated_inputs)
        {
            let arg_ty = self.check_expr(*arg_expr_id);
            let arg_span = self.hir.expr(*arg_expr_id).span;
            let port_note = self.hir.sources.get(port_decl_span.source).map(|source| {
                let (line, column) = source.line_col(port_decl_span.lo);
                format!(
                    "corresponding model port declared at {}:{line}:{column}",
                    source.name
                )
            });
            tensor_check::check_model_predict_arg(
                self,
                arg_ty,
                expected_port_ty,
                arg_span,
                port_note,
            );
        }

        tensor_check::model_predict_result(instantiated_outputs)
    }
}

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
