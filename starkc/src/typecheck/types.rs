//! **AS7 Packet 4 — the type representation.**
//!
//! The bottom of the checker's dependency DAG: `types` may depend on **nothing** inside
//! `typecheck`. That is what makes it the right first extraction — any edge the dependency checker
//! reports out of this module is unambiguously a real violation rather than an artefact of
//! extraction order.
//!
//! What lives here is the *representation* and the operations that are pure functions of it.
//! Anything needing the checker's state, its substitution map or its diagnostics belongs further
//! up: `state`, `infer` and above.
//!
//! Re-exported unchanged through `typecheck/mod.rs`, so `crate::typecheck::Ty` and
//! `crate::typecheck::substitute_ty` keep resolving for every existing caller. AS7 is not a
//! repository-wide import migration.

use crate::ast::Primitive;
use crate::diag::Diagnostic;
use crate::extensions::tensor::syntax as tensor_syntax;
use crate::extensions::tensor::types::TensorKind;
use crate::hir::Res;
use crate::hir::{self, BlockId, CoreType, ExprId, ItemId, LocalId};
use crate::source::Span;
use std::collections::{BTreeMap, HashMap};

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
    Extension(Box<ExtensionTy>),
    Error,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExtensionTy {
    Tensor(TensorKind),
    Model(ModelTy),
    ModelError,
}

/// **TYPE-PRIM-001**, applied where a tuple type is built: *"`Unit` and `()` are two spellings of
/// the same single-inhabitant type"* (03-Type-System also states it directly — "`()` is `Unit`").
///
/// DEV-112: the checker used to give the empty tuple its own type, `Ty::Tuple([])`, which unified
/// with nothing — so `let x: Unit = ();` failed E0001 with the memorable *"expected 'Unit', found
/// '()'"*, and there was **no way to write a value of type `Unit` at all**. That is not cosmetic:
/// PROC-EXIT-001 gives `Ok(Unit)` its own exit-status rule and PROC-MAIN-001 admits
/// `Result<Unit, String>` entries, so the success branch of a legal entry signature was unreachable
/// from source. Found by DEV-111's entry-contract cases.
///
/// Canonicalising at construction — rather than teaching `unify` that two representations are
/// interchangeable — is what makes them *the same type* as the rule says, instead of two types with
/// a special case. `Ty::Tuple([])` is therefore not constructible from source, and no comparison
/// site has to know about the equivalence.
pub(super) fn unit_or_tuple(elems: Vec<Ty>) -> Ty {
    if elems.is_empty() {
        Ty::Primitive(Primitive::Unit)
    } else {
        Ty::Tuple(elems)
    }
}

/// Structural search for a type constructor anywhere inside `ty` (WP-C4.5c helpers for
/// auditing grounded generic instantiations before publication).
pub(super) fn ty_contains(ty: &Ty, pred: &dyn Fn(&Ty) -> bool) -> bool {
    if pred(ty) {
        return true;
    }
    match ty {
        Ty::Ref { inner, .. } => ty_contains(inner, pred),
        Ty::Struct(_, args) | Ty::Enum(_, args) | Ty::Core(_, args) => {
            args.iter().any(|arg| ty_contains(arg, pred))
        }
        Ty::Tuple(elems) => elems.iter().any(|e| ty_contains(e, pred)),
        Ty::Array(elem, _) | Ty::Slice(elem) | Ty::Range(elem) => ty_contains(elem, pred),
        Ty::Fn { params, ret } => {
            params.iter().any(|p| ty_contains(p, pred)) || ty_contains(ret, pred)
        }
        _ => false,
    }
}

pub(super) fn ty_contains_infer(ty: &Ty) -> bool {
    ty_contains(ty, &|t| matches!(t, Ty::Infer(_)))
}

pub(super) fn ty_contains_error(ty: &Ty) -> bool {
    ty_contains(ty, &|t| matches!(t, Ty::Error))
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ModelTy {
    pub item_id: ItemId,
}

/// Substitute generic parameters throughout a type.
///
/// DEV-100 requires this to recurse everywhere a parameter can hide, not just to swap a bare
/// `Ty::Param`: `size_of::<[T; 4]>()` and `size_of::<Pair<T>>()` are the immediate next holes if
/// it does not. Mirrors `TypeChecker::instantiate_ty`, which does the same job while the checker
/// is still alive.
pub fn substitute_ty(ty: &Ty, map: &HashMap<String, Ty>) -> Ty {
    match ty {
        Ty::Param(name) => map.get(name).cloned().unwrap_or_else(|| ty.clone()),
        Ty::Ref { mutable, inner } => Ty::Ref {
            mutable: *mutable,
            inner: Box::new(substitute_ty(inner, map)),
        },
        Ty::Struct(item, args) => {
            Ty::Struct(*item, args.iter().map(|a| substitute_ty(a, map)).collect())
        }
        Ty::Enum(item, args) => {
            Ty::Enum(*item, args.iter().map(|a| substitute_ty(a, map)).collect())
        }
        Ty::Core(core, args) => {
            Ty::Core(*core, args.iter().map(|a| substitute_ty(a, map)).collect())
        }
        Ty::Tuple(elems) => Ty::Tuple(elems.iter().map(|e| substitute_ty(e, map)).collect()),
        Ty::Array(elem, len) => Ty::Array(Box::new(substitute_ty(elem, map)), *len),
        Ty::Slice(elem) => Ty::Slice(Box::new(substitute_ty(elem, map))),
        Ty::Fn { params, ret } => Ty::Fn {
            params: params.iter().map(|p| substitute_ty(p, map)).collect(),
            ret: Box::new(substitute_ty(ret, map)),
        },
        Ty::Range(elem) => Ty::Range(Box::new(substitute_ty(elem, map))),
        other => other.clone(),
    }
}

/// Whether any generic parameter survives anywhere in `ty`. DEV-100 requires an unsubstituted
/// parameter to be an oracle DEFECT rather than a fallback layout.
pub fn ty_contains_param(ty: &Ty) -> bool {
    ty_contains(ty, &|t| matches!(t, Ty::Param(_)))
}

/// Collect every `Ty::Param` NAME reachable in `ty`, including associated-type projections
/// (`"T::Item"`), which the type system encodes as a param whose name contains `::`.
///
/// Published because the interpreter must discharge those projections at a value boundary and
/// needs to know which ones a type mentions before it can look them up. Traverses through the same
/// structure as [`substitute_ty`], so the two cannot disagree about where a parameter can hide.
pub fn collect_ty_params(ty: &Ty, out: &mut std::collections::BTreeSet<String>) {
    match ty {
        Ty::Param(name) => {
            out.insert(name.clone());
        }
        Ty::Ref { inner, .. } => collect_ty_params(inner, out),
        Ty::Struct(_, args) | Ty::Enum(_, args) | Ty::Core(_, args) => {
            for arg in args {
                collect_ty_params(arg, out);
            }
        }
        Ty::Tuple(elems) => {
            for elem in elems {
                collect_ty_params(elem, out);
            }
        }
        Ty::Array(elem, _) | Ty::Slice(elem) | Ty::Range(elem) => collect_ty_params(elem, out),
        Ty::Fn { params, ret } => {
            for param in params {
                collect_ty_params(param, out);
            }
            collect_ty_params(ret, out);
        }
        _ => {}
    }
}

// -----------------------------------------------------------------------------------------------
// AS7 Packet 5 — the result and metadata types.
//
// Deferred in Packet 4 as "assess whether they move cleanly or drag behaviour with them". The
// dependency checker answered it: extracting `state.rs` produced `state -> mod`, which the DAG
// forbids, because `state`'s storage is typed in these very types. They are pure data plus
// operations that are pure functions of it, so they belong at the bottom with the representation,
// exactly as the approved decomposition says.
// -----------------------------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum VariantFields {
    Unit,
    Tuple(Vec<Ty>),
    Struct(HashMap<String, Ty>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum GenericKind {
    Type,
    Dim,
    DType,
    Device,
}

impl GenericKind {
    /// The tensor kind this parameter carries, if any. `Type` is the ordinary Core case and has
    /// no tensor kind.
    pub(super) fn as_tensor_param(self) -> Option<tensor_syntax::TensorParamKind> {
        match self {
            GenericKind::Type => None,
            GenericKind::Dim => Some(tensor_syntax::TensorParamKind::Dim),
            GenericKind::DType => Some(tensor_syntax::TensorParamKind::DType),
            GenericKind::Device => Some(tensor_syntax::TensorParamKind::Device),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct VariantTy {
    pub(super) name: String,
    pub(super) fields: VariantFields,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct FnSigTy {
    pub(super) params: Vec<Ty>,
    pub(super) ret: Ty,
}

pub(super) struct LoopContext {
    pub(super) allows_value: bool,
    pub(super) break_ty: Ty,
    pub(super) has_break: bool,
}

#[derive(Clone, Copy)]
pub(super) struct ControlSummary {
    pub(super) can_complete: bool,
    pub(super) may_return: bool,
}

/// WP-C5.3e / DEV-100: everything needed to answer a layout query over a CHECKER type, in one
/// place that outlives the checker.
///
/// It exists because the layout walk needs three things the raw `Ty` does not carry — declaration
/// order for a struct's fields (the checker's own map is name-keyed and has none), a nominal's
/// variant payloads, and a nominal's generic parameter NAMES so `Ty::Param` can be substituted.
/// The checker owns all three during analysis; the HIR oracle needs them afterwards. Publishing
/// them is what lets there be ONE checker-side walker rather than a second one in the oracle.
#[derive(Clone, Debug, Default)]
pub struct LayoutTables {
    /// The named target contract layout answers come from (CD-067).
    pub contract: crate::layout::TargetLayout,
    /// Field types in DECLARATION order — layout depends on order, so a name-keyed map will not do.
    pub struct_fields: HashMap<ItemId, Vec<Ty>>,
    /// Variant payloads in declaration order, each payload in declaration order.
    pub enum_variants: HashMap<ItemId, Vec<Vec<Ty>>>,
    /// Generic parameter names in declaration order, per nominal item.
    pub nominal_params: HashMap<ItemId, Vec<String>>,
}

/// How a pattern binding takes its value, decided once per `match` from its scrutinee.
///
/// **STARK differs from Rust here, deliberately.** Rust's match ergonomics bind EVERY component by
/// reference under a reference scrutinee, so `E::A(n)` gives `n: &u64` and the arm writes `*n`.
/// STARK copies a `Copy` component and binds a non-`Copy` one by reference: `n: UInt64`, written
/// `n`. The rule is "take what cannot be taken from the referent by reference, copy the rest",
/// which is the rule both runtime engines already implement for `match *r`.
///
/// The alternative was tried and rejected mid-change: matching Rust would mean threading the mode
/// through `PatternSource` in the interpreter and through MIR pattern lowering as well, to keep
/// three engines agreeing about the type of a binding. That is a language-semantics change worth
/// making deliberately, not as a side effect of fixing a match that did not work at all. The
/// consequence for callers is a `*` that is not written, and the checker says so precisely
/// ("cannot dereference non-reference type 'UInt64'").
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum BindMode {
    /// An owned scrutinee: every binding takes its value.
    ByValue,
    /// A scrutinee reached through a reference — either read through one (`*r`, or a field of one)
    /// or reference-typed itself (`match r` where `r: &E`). Non-`Copy` components bind by
    /// reference so the match cannot move out of borrowed storage; a `Copy` component is copied,
    /// because copying it takes nothing from the referent.
    ThroughRef,
}

impl BindMode {
    pub(super) fn binds_by_ref(self, is_copy: bool) -> bool {
        match self {
            BindMode::ByValue => false,
            BindMode::ThroughRef => !is_copy,
        }
    }
}

impl LayoutTables {
    /// The CHECKER-type walker into the shared layout combinators (`crate::layout`).
    ///
    /// One of the contract's two adapters; the other walks `MirTy` for the MIR interpreter and the
    /// native backend. They cannot share a traversal — the representations genuinely differ — so
    /// the *algorithm* is shared and only the walk is duplicated.
    pub fn layout_of(&self, ty: &Ty) -> Result<crate::layout::Layout, crate::layout::LayoutError> {
        use crate::layout::{LayoutError, Scalar};
        let t = &self.contract;
        let unsupported = |what: String| {
            LayoutError(format!(
                "the {} layout contract does not describe {what}",
                t.identity.target_contract
            ))
        };
        Ok(match ty {
            Ty::Primitive(p) => t.scalar(match p {
                Primitive::Int8 => Scalar::Int8,
                Primitive::Int16 => Scalar::Int16,
                Primitive::Int32 => Scalar::Int32,
                Primitive::Int64 => Scalar::Int64,
                Primitive::UInt8 => Scalar::UInt8,
                Primitive::UInt16 => Scalar::UInt16,
                Primitive::UInt32 => Scalar::UInt32,
                Primitive::UInt64 => Scalar::UInt64,
                Primitive::Float32 => Scalar::Float32,
                Primitive::Float64 => Scalar::Float64,
                Primitive::Bool => Scalar::Bool,
                Primitive::Char => Scalar::Char,
                Primitive::Unit => Scalar::Unit,
                other => return Err(unsupported(format!("the primitive {other:?}"))),
            }),
            Ty::Ref { .. } => t.scalar(Scalar::Reference),
            Ty::Fn { .. } => t.scalar(Scalar::FnValue),
            Ty::Tuple(elems) => {
                let mut fields = Vec::with_capacity(elems.len());
                for elem in elems {
                    fields.push(self.layout_of(elem)?);
                }
                t.aggregate(fields)
            }
            Ty::Array(elem, len) => {
                let elem = self.layout_of(elem)?;
                t.array(elem, *len)
            }
            Ty::Struct(item, args) => {
                let field_tys = self
                    .struct_fields
                    .get(item)
                    .ok_or_else(|| unsupported(format!("struct item {item:?}")))?;
                let map = self.param_map(*item, args);
                let mut fields = Vec::with_capacity(field_tys.len());
                for field_ty in field_tys {
                    fields.push(self.layout_of(&substitute_ty(field_ty, &map))?);
                }
                t.aggregate(fields)
            }
            Ty::Enum(item, args) => {
                let variants = self
                    .enum_variants
                    .get(item)
                    .ok_or_else(|| unsupported(format!("enum item {item:?}")))?;
                let map = self.param_map(*item, args);
                let mut laid_out = Vec::with_capacity(variants.len());
                for payload in variants {
                    let mut fields = Vec::with_capacity(payload.len());
                    for field_ty in payload {
                        fields.push(self.layout_of(&substitute_ty(field_ty, &map))?);
                    }
                    laid_out.push(t.aggregate(fields));
                }
                t.sum(laid_out)
            }
            // The core enums' payloads are derived from their type arguments, exactly as
            // `mir::drop_plan::variant_payloads` derives them for the other adapter.
            Ty::Core(CoreType::Option, args) => {
                let inner = args
                    .first()
                    .ok_or_else(|| unsupported("Option without a type argument".to_string()))?;
                let inner = self.layout_of(inner)?;
                t.sum([t.aggregate([]), t.aggregate([inner])])
            }
            Ty::Core(CoreType::Result, args) => {
                let ok = args
                    .first()
                    .ok_or_else(|| unsupported("Result without an Ok argument".to_string()))?;
                let err = args
                    .get(1)
                    .ok_or_else(|| unsupported("Result without an Err argument".to_string()))?;
                let ok = self.layout_of(ok)?;
                let err = self.layout_of(err)?;
                t.sum([t.aggregate([ok]), t.aggregate([err])])
            }
            Ty::Core(CoreType::Ordering, _) => {
                t.sum([t.aggregate([]), t.aggregate([]), t.aggregate([])])
            }
            other => {
                return Err(unsupported(format!(
                    "{other:?}: owning runtime types, unsized types and unsubstituted generic \
                     parameters have no contract entry"
                )))
            }
        })
    }

    fn param_map(&self, item: ItemId, args: &[Ty]) -> HashMap<String, Ty> {
        self.nominal_params
            .get(&item)
            .map(|names| {
                names
                    .iter()
                    .cloned()
                    .zip(args.iter().cloned())
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default()
    }
}

/// Where a generic parameter was DECLARED. (WP-VALUE-REP-TOTAL, A3c-S.)
///
/// Provenance is kept because the consumers need different views of one answer: the HIR oracle
/// wants a name→type map, MIR wants the method's own arguments in declaration order, and a
/// diagnostic wants to say which declaration a binding came from. Storing one table with origins
/// lets each derive its view; storing several tables would be several authorities on what a generic
/// call means.
///
/// **Names are unique across every binder simultaneously in scope**, guaranteed by NAME-SHADOW-001
/// (enforced for DEV-177). That is what makes the derived `HashMap<String, Ty>` sound — the
/// provenance here is for ordering and diagnostics, not to disambiguate colliding names.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericBinder {
    /// The implicit `Self` of an impl or trait.
    SelfType,
    /// A free function's own parameter.
    FunctionParam { index: usize, name: String },
    /// The enclosing impl's parameter.
    ImplParam { index: usize, name: String },
    /// The enclosing trait's parameter.
    TraitParam { index: usize, name: String },
    /// The callable's own parameter, when the callable is a method or associated function.
    MethodParam { index: usize, name: String },
}

impl GenericBinder {
    /// The name this binder introduces. `Self` is spelled as the type keyword because that is how
    /// `Ty::Param` carries it inside a trait default body.
    pub fn name(&self) -> &str {
        match self {
            GenericBinder::SelfType => "Self",
            GenericBinder::FunctionParam { name, .. }
            | GenericBinder::ImplParam { name, .. }
            | GenericBinder::TraitParam { name, .. }
            | GenericBinder::MethodParam { name, .. } => name,
        }
    }
}

/// The generic environment the checker selected for ONE callable use. (A3c-S.)
///
/// Keyed by the call expression, never by the body: one generic body is legitimately invoked as
/// `Wrapper<Int32>` and `Wrapper<String>` in the same program, so an environment attached to the
/// body could only hold one of them.
///
/// Bindings may themselves contain `Ty::Param` when the CALLER is generic — `fn outer<T>(w:
/// Wrapper<T>)` publishes `impl T -> T`. The interpreter concretises against its active frame
/// before installing, which is what composes the two.
#[derive(Debug, Clone)]
pub struct CallableInstantiation {
    /// The body this use selected, so a consumer can pair the environment with A3b's signature.
    pub body: BlockId,
    pub bindings: Vec<(GenericBinder, Ty)>,
}

impl CallableInstantiation {
    /// The callable's OWN parameters in declaration order — the view MIR monomorphisation needs,
    /// and exactly what the deleted `generic_insts` stored on its own.
    ///
    /// Impl, trait and `Self` bindings are excluded: a monomorphisation key is per callable, and
    /// including the enclosing impl's arguments would change the key's arity.
    pub fn own_arguments(&self) -> Vec<Ty> {
        self.bindings
            .iter()
            .filter(|(binder, _)| {
                matches!(
                    binder,
                    GenericBinder::FunctionParam { .. } | GenericBinder::MethodParam { .. }
                )
            })
            .map(|(_, ty)| ty.clone())
            .collect()
    }

    /// The name→type view the runtime substitutes with. Sound because NAME-SHADOW-001 forbids two
    /// binders in scope sharing a name.
    pub fn substitutions(&self) -> HashMap<String, Ty> {
        self.bindings
            .iter()
            .map(|(binder, ty)| (binder.name().to_string(), ty.clone()))
            .collect()
    }
}

/// The checker-established signature of one executable callable body. (WP-VALUE-REP-TOTAL, A3b.)
///
/// **Keyed by `BlockId`, because that is the identity execution has.** `fn_types` is keyed by
/// `ItemId` and therefore covers free functions only: `hir::FnDef` carries no `ItemId`, so an
/// inherent method, a trait implementation method, an associated function, `Drop::drop` and a trait
/// default body all have signatures the checker computed and nothing could look up. A `Callable`
/// already carries the selected body, so no name lookup or reconstructed identity is needed.
///
/// Bodyless trait declarations are excluded structurally rather than by a filter — they have no
/// `BlockId` to key on.
#[derive(Debug, Clone, PartialEq)]
pub struct CallableSigTy {
    pub receiver: Option<Ty>,
    pub params: Vec<Ty>,
    pub ret: Ty,
}

/// What one scan of the impl set establishes about a user iterator (AS3 Boundary 4).
pub(super) struct UserIteratorSelection {
    pub(super) impl_item: ItemId,
    pub(super) member: u32,
    pub(super) body: BlockId,
    /// The `type Item = ...` declaration, still parametric.
    pub(super) associated_item: hir::TypeId,
    /// `match_impl_type`'s result — what makes `Item` concrete.
    pub(super) substitutions: HashMap<String, Ty>,
    /// The same substitution as ordered binders, for the published environment.
    pub(super) bindings: Vec<(GenericBinder, Ty)>,
}

/// A published callable use. Indexes [`TypeTables::callable_uses`].
///
/// **A use is a STATIC SEMANTIC USE SITE, not an expression and not a runtime invocation.** One
/// expression may give rise to zero, one or many: `println((a, b))` is one argument expression and
/// two `Display::fmt` use sites, and `println(vec)` is one use site executed once per element. A
/// map keyed by `ExprId` cannot represent either, which is why this id exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallableUseId(pub u32);

/// WHICH declaration the checker selected, expressed in ids the HIR actually possesses.
///
/// Methods, associated functions and trait defaults have **no `ItemId`**: `ImplItem::Fn` embeds a
/// `FnDef` and `TraitItem::Method` embeds a signature, both positional inside their owner's `Vec`.
/// That is why A3b chose `BlockId` for executable identity. This is the *declaration* identity,
/// which provenance and diagnostics need and which a `BlockId` cannot express — and it is built
/// from real ids rather than from fabricated ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallableDeclId {
    /// A free function: it has its own item.
    Item(ItemId),
    /// A member of an impl, by position in that impl's `items`.
    ImplMember { impl_item: ItemId, member: u32 },
    /// A member of a trait — the declaration site of a default body.
    TraitMember { trait_item: ItemId, member: u32 },
}

/// What runs. Static for an ordinary call; deferred to the value for a function value.
///
/// DEV-178: a function value carries the item and the bindings it was created with, because the
/// call site's `Ty::Fn` cannot reconstruct which instantiation produced it. Pretending every use
/// has a statically known body would erase that.
/// **Three binding times, not two.**
///
/// AS3 Boundary 4's Display characterization found a category the two-variant model could not
/// represent, and it is not a Display corner: a call on a bounded generic parameter
/// (`fn f<T: Speak>(x: T) { x.speak(); }`) has the same shape. `resolve_method`'s bound branch
/// records `bound_trait_calls` and returns before Boundary 2's publication, so nothing was ever
/// published for it.
///
/// ```text
/// Static          body known during typecheck
/// Bound           trait/member known during typecheck;
///                 body known when `Self` becomes concrete
/// FunctionValue   body and environment carried by the runtime value
/// ```
///
/// **`Bound` is not a licence for an engine to select later.** It is a checker-published dispatch
/// obligation whose *declaration identity* is fixed now, and whose *executable target* is resolved
/// by one shared bound-specialisation authority when `Self` becomes concrete. Both engines consume
/// that authority's result; neither implements matching. Defining it the other way would simply
/// give the old scans a respectable name.
#[derive(Debug, Clone, PartialEq)]
pub enum CalleeSelection {
    Static {
        declaration: CallableDeclId,
        body: BlockId,
    },
    /// Late-bound: the obligation is fixed, the body is not.
    Bound {
        trait_: hir::BoundTrait,
        /// The trait member invoked, by name — traits declare members positionally like impls, and
        /// the specialiser resolves the position against whichever impl `Self` selects.
        member: String,
        /// The receiver type, which may still contain caller parameters (`T`, `W<T>`).
        self_ty: Ty,
        /// Trait arguments, for parameterised traits.
        trait_args: Vec<Ty>,
        /// The METHOD's own generic arguments. G2 characterization: `x.to::<Int32>()` through a
        /// bound is accepted, so without these the specialiser would bind the impl's parameters
        /// and silently drop the method's — the Iterator empty-environment class again.
        method_args: Vec<Ty>,
    },
    FunctionValue,
}

/// The generic environment, on the same footing as the selection.
///
/// A non-generic static call is `Static(vec![])` — an empty environment, never an absent one.
#[derive(Debug, Clone, PartialEq)]
pub enum GenericEnvironment {
    Static(Vec<(GenericBinder, Ty)>),
    /// **The callee's environment does not exist yet**, because the callee is not selected yet.
    /// The bound specialiser produces body and environment *atomically* — reconstructing the
    /// environment separately from the body is how DEV-176 happened, in a new place.
    FromBoundSelection,
    /// Fixed at coercion and carried by the value (DEV-178).
    FromFunctionValue,
}

impl GenericEnvironment {
    /// The name→type view to substitute with — **the same view `CallableInstantiation` publishes**,
    /// so a consumer or a test never has to build a second one.
    ///
    /// Sound because NAME-SHADOW-001 forbids two binders in scope sharing a name. Empty for
    /// `FromFunctionValue`: the environment is on the value, not here.
    pub fn substitutions(&self) -> HashMap<String, Ty> {
        match self {
            GenericEnvironment::Static(bindings) => bindings
                .iter()
                .map(|(binder, ty)| (binder.name().to_string(), ty.clone()))
                .collect(),
            // Neither carries a callee environment HERE: the function value has it, and a bound
            // use has none until specialisation produces body and environment together.
            GenericEnvironment::FromBoundSelection | GenericEnvironment::FromFunctionValue => {
                HashMap::new()
            }
        }
    }
}

/// What the CALL SITE did to the receiver (TYPE-METHOD-002 auto-borrow / auto-deref).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverAdjustment {
    None,
    ByValue,
    Shared { derefs: u8 },
    Exclusive { derefs: u8 },
}

/// What the SELECTED CALLABLE binds.
///
/// Deliberately separate from [`ReceiverAdjustment`]. They correlate in ordinary code, but they are
/// different questions with different authorities — the call site's adjustment versus the
/// declaration's `self` form — and AS4 asks about the binding side specifically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverBinding {
    None,
    ByValue,
    Shared,
    Exclusive,
}

/// **A `Display` render position queued for Pass 3.**
///
/// A named structure rather than a tuple because the third field is the reason this type exists,
/// and a tuple hides that: `generic_scope` is the function/impl generic environment *as it was
/// where the expression was written*.
///
/// The walk runs in Pass 3, since it keys positions off RESOLVED types. But
/// `bound_method_candidates` reads LIVE scope, which Pass 3 has already torn down — so without
/// carrying it, the walk reached `Ty::Param("T")` inside `fn show<T: Display>` and found no
/// candidates for a bound that is plainly written, publishing nothing and saying nothing.
///
/// The general rule, which is not about `Display`: a deferred obligation may read resolved types
/// freely, but any **scope-sensitive** question it asks is a question about a scope that no longer
/// exists. Capture the scope with the obligation.
pub(super) struct DeferredDisplayPlan {
    /// The expression that renders — a `println`-family argument or an interpolation field. Both
    /// are roots in their own right; an interpolation field has its own `ExprId`.
    pub(super) root: ExprId,
    pub(super) ty: Ty,
    /// `(current_fn_generics, current_impl_generics)` at the point of writing.
    pub(super) generic_scope: (
        Option<Vec<hir::GenericParam>>,
        Option<Vec<hir::GenericParam>>,
    ),
}

/// **One structural position inside a `println` argument's STATIC type.**
///
/// `println((a, b))` is one expression and two `Display::fmt` bodies; `println(vec)` is one body
/// executed once per element; `println((W<Int32>, W<Bool>))` is the SAME body at two different
/// instantiations. A nominal-keyed lookup cannot tell the last pair apart — a runtime
/// `Value::Struct { item, fields }` carries no type arguments — so the key is the static position,
/// which distinguishes all three. See `AS3-DISPLAY-CHARACTERIZATION.md` §2.
///
/// Array, slice and `Vec` elements are deliberately three steps rather than one. At any given path
/// the static type already says which it is, so collapsing them would lose nothing for lookup — but
/// it would also let an engine walking the wrong container shape find a use anyway, and agreeing by
/// accident is what this packet keeps finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DisplayStep {
    TupleField(u32),
    ArrayElement,
    SliceElement,
    VecElement,
    OptionSome,
    ResultOk,
    ResultErr,
}

/// A path from a `println` argument to one nominal that renders through its own `Display::fmt`.
/// Empty means the argument itself.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DisplayPath(pub Vec<DisplayStep>);

impl DisplayPath {
    /// This path with one more step. Used by the checker to BUILD the plan and by both engines to
    /// walk it, so the two constructions cannot drift.
    pub fn child(&self, step: DisplayStep) -> Self {
        let mut steps = self.0.clone();
        steps.push(step);
        DisplayPath(steps)
    }
}

/// Why this callable and not another.
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchProvenance {
    /// `f(x)` — a path resolved to an item.
    Direct,
    /// `x.m()` — inherent method resolution.
    Inherent,
    /// `x.m()` where `m` came from a trait impl.
    TraitImpl { trait_item: ItemId },
    /// `T::m()` / `<T as Tr>::m()`.
    Qualified { trait_item: Option<ItemId> },
    /// A generic parameter's BOUND supplied the signature — what `bound_trait_calls` carries today.
    /// Uses [`BoundTrait`] because a bound may name a `CoreTrait`, which has no `ItemId`.
    Bound { trait_: hir::BoundTrait },
    /// A compiler-known trait operation: `==`, `<`, `for`, `{}` formatting. The four families both
    /// engines currently re-select with no filter at all.
    CoreTrait { core: hir::CoreTrait },
    /// Calling a function value.
    FunctionValue,
}

/// Everything the checker decided about one callable use.
///
/// The rule this exists to serve: **the checker publishes, execution consumes, neither engine
/// reconstructs selection.** An engine may CHOOSE among published records using runtime or static
/// structure; it may not scan the HIR and re-run method selection.
#[derive(Debug, Clone, PartialEq)]
pub struct CallableUse {
    pub selection: CalleeSelection,
    pub environment: GenericEnvironment,
    pub receiver_adjustment: ReceiverAdjustment,
    pub receiver_binding: ReceiverBinding,
    /// This use's signature.
    ///
    /// **Inference-grounded, not fully concrete**: no surviving `Ty::Infer` or `Ty::Error`. A
    /// caller's own `Ty::Param` may remain and is concretised against the active caller
    /// environment — the same rule `CallableInstantiation` documents, and why `callable_types` is
    /// body-parametric.
    pub signature: CallableSigTy,
    pub provenance: DispatchProvenance,
}

#[derive(Debug, Clone, Default)]
pub struct TypeTables {
    pub expr_types: HashMap<ExprId, Ty>,
    pub local_types: HashMap<LocalId, Ty>,
    pub local_mutability: HashMap<LocalId, bool>,
    /// Grounded signatures for top-level functions.  Executable-target
    /// selection consumes this table after ordinary package analysis so a
    /// package can remain library-importable without imposing a `main`
    /// requirement during type checking.
    pub fn_types: HashMap<ItemId, (Vec<Ty>, Ty)>,
    /// **Associated-type bindings, keyed by (implementing nominal, associated name).**
    ///
    /// Published so the interpreter can discharge a projection like `T::Item` at a value boundary
    /// once `T` is concrete. Without it the oracle would need its own scan of the impl set — a
    /// third authority for a question the checker already answered, beside `normalize_projections`
    /// here and `ProgramMeta::assoc_projections` in MIR lowering.
    pub assoc_projections: HashMap<(ItemId, String), Ty>,
    /// AS3 Packet 5: each aggregate literal's fields, by name, at their DECLARED type instantiated
    /// for that literal. The `AggregateField` boundary's expected type — never the initialiser
    /// expression's own type, which would make the check compare a value against its producer.
    pub aggregate_field_types: HashMap<ExprId, HashMap<String, Ty>>,
    /// WP-VALUE-REP-TOTAL A3b: every executable callable body's signature, keyed by its body.
    ///
    /// Covers all six classes `check_fn_def` sees — free functions, inherent methods, trait
    /// implementation methods, associated functions, `Drop::drop`, and trait default bodies with
    /// a body. Publication only: nothing consumes it until A4 wires the boundaries.
    ///
    /// Entries may still contain `Ty::Param`. A generic body is checked ONCE, so its signature is
    /// parametric by nature; concretising it per invocation is A3c's job, not this table's.
    pub callable_types: HashMap<BlockId, CallableSigTy>,
    /// WP-VALUE-REP-TOTAL A3c-S: the generic environment selected for each callable USE.
    ///
    /// Replaced `generic_insts`, which recorded only a free function's or a method's OWN
    /// parameters positionally and therefore could not express impl generics, trait generics or
    /// `Self` — the reason DEV-176 exists.
    pub callable_instantiations: HashMap<ExprId, CallableInstantiation>,
    /// **AS3 / WP-CALLABLE-USE-TOTAL: every published callable use, indexed by `CallableUseId`.**
    ///
    /// One record per STATIC SEMANTIC USE SITE. See `callable_uses_by_expr` for why this is not a
    /// map keyed by expression.
    pub callable_uses: Vec<CallableUse>,
    /// The uses an expression gives rise to — **zero, one or many**.
    ///
    /// Many is not hypothetical: `display_deep` recurses through tuples, arrays, `Option`, `Result`
    /// and slots, reaching a nominal's `Display::fmt` at any depth, so `println((a, b))` is one
    /// argument expression and two use sites. Zero is ordinary — most expressions call nothing.
    pub callable_uses_by_expr: HashMap<ExprId, Vec<CallableUseId>>,

    /// **AS3 Boundary 4: the `Display` dispatch plan, keyed by static position.**
    ///
    /// `(root argument expression, path) -> the use that renders there`. Both engines recurse
    /// value-and-static-type together and look the position up, instead of scanning a nominal's
    /// impls for a member named `fmt`.
    pub display_uses: BTreeMap<(ExprId, DisplayPath), CallableUseId>,
    /// **AS3 Boundary 4a: the program's coherent dispatch index**, frozen for execution.
    ///
    /// Answers "given trait identity, trait arguments, concrete `Self` and a member, which
    /// executable body does the checked program mean" — the authority `find_method` and
    /// `find_impl_fn` currently duplicate. Carries no signatures: `callable_types[body]` is the
    /// sole signature authority (A3b).
    pub trait_impls: crate::bound_dispatch::TraitImplIndex,
    /// WP-C4.5c / A3c-S: grounded, ordered generic-argument types for each use of a generic fn item,
    /// keyed by the referencing path expression (the call callee or fn-value use). Inside a
    /// generic body the entries may themselves be `Ty::Param`; they are fully concrete after
    /// the enclosing instantiation substitutes its own arguments. Entries never contain
    /// `Ty::Infer` — undetermined instantiations are rejected during checking (E0004).
    /// WP-C5.3e: the QUERIED TYPE of each layout query, keyed by the builtin's path expression.
    /// Before it existed the oracle returned a hardcoded `8` without even looking at the type.
    ///
    /// DEV-100: the type rather than a precomputed layout, because inside a generic body it still
    /// contains `Ty::Param` — the checker sees a generic body ONCE, so there is no
    /// per-instantiation answer for it to precompute. The oracle substitutes from its call-time
    /// substitution stack and then resolves through [`LayoutTables`].
    pub layout_queries: HashMap<ExprId, Ty>,
    /// DEV-BOUND-TRAIT-IDENTITY: the trait a bounded-generic method call resolved through,
    /// keyed by the call expression. `Res::Item` for a user trait, `Res::CoreTrait` for a
    /// compiler-known one. Consumed by the HIR interpreter's `find_method` trait filter and by
    /// MIR lowering's implementation lookup, so all three engines select the impl the checker
    /// selected the signature from.
    pub bound_trait_calls: HashMap<ExprId, Res>,
    /// WP-C5.3e: the tables and contract a layout query is resolved against.
    pub layout: LayoutTables,
}

#[derive(Debug, Clone)]
pub struct TypeCheckResult {
    pub diagnostics: Vec<Diagnostic>,
    pub tables: TypeTables,
}

/// A deferred trait-bound obligation (DEV-067): the concrete type, the bounds it must satisfy,
/// the call span to report against, and the caller's enclosing generic environment.
///
/// DEV-101 added a fifth element — the file declaring the bounds — so a bound path's name could be
/// read correctly once the obligation was discharged. AS1b-ii-d removed it: the bound path's span
/// names that file.
pub(super) type BoundsCheck = (Ty, Vec<hir::TraitRef>, Span, Vec<hir::GenericParam>);

// AS7 Packet 6: a pure predicate on the primitive set — the bottom of the DAG.
/// WP-C4.7-6.3: the primitive integer types an unsuffixed integer literal may adopt.
pub(super) fn is_integer_primitive(p: Primitive) -> bool {
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

// AS7 Packet 7: moved to the layer that owns the question.
/// DEV-075: the primitive float types. CD-015 (WP-C2.9) froze that primitive floats implement
/// none of `Eq`/`Ord`/`Hash`; ordered float COMPARISON operators remain available as built-in
/// primitive operations (IEEE), which is a separate thing from the trait.
pub(super) fn is_float_primitive(p: Primitive) -> bool {
    matches!(
        p,
        Primitive::Float16 | Primitive::BFloat16 | Primitive::Float32 | Primitive::Float64
    )
}
pub(super) fn is_numeric(p: Primitive) -> bool {
    is_integer(p) || matches!(p, Primitive::Float32 | Primitive::Float64)
}
pub(super) fn standard_display_type(ty: &Ty) -> bool {
    match ty {
        Ty::Primitive(primitive) => matches!(
            primitive,
            Primitive::Int8
                | Primitive::Int16
                | Primitive::Int32
                | Primitive::Int64
                | Primitive::UInt8
                | Primitive::UInt16
                | Primitive::UInt32
                | Primitive::UInt64
                | Primitive::Float32
                | Primitive::Float64
                | Primitive::Bool
                | Primitive::Char
                | Primitive::Unit
                | Primitive::String
                | Primitive::Str
        ),
        Ty::Core(CoreType::Ordering | CoreType::IOError, args) => args.is_empty(),
        Ty::Ref { inner, .. } => standard_display_type(inner),
        _ => false,
    }
}
pub(super) fn standard_hash_type(ty: &Ty) -> bool {
    match ty {
        Ty::Primitive(primitive) => !matches!(
            primitive,
            Primitive::Float16 | Primitive::BFloat16 | Primitive::Float32 | Primitive::Float64
        ),
        Ty::Tuple(elements) => elements.iter().all(standard_hash_type),
        Ty::Array(element, _) => standard_hash_type(element),
        Ty::Core(CoreType::Vec | CoreType::Option, args) => {
            args.len() == 1 && args.iter().all(standard_hash_type)
        }
        Ty::Core(CoreType::Result, args) => args.len() == 2 && args.iter().all(standard_hash_type),
        Ty::Ref { inner, .. } => standard_hash_type(inner),
        _ => false,
    }
}
pub(super) fn is_integer(p: Primitive) -> bool {
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

// AS7 Packet 8: literal suffix -> primitive is a pure fact about the token.
pub(super) fn convert_int_suffix(suffix: crate::lexer::IntSuffix) -> Primitive {
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
pub(super) fn convert_float_suffix(suffix: crate::lexer::FloatSuffix) -> Primitive {
    match suffix {
        crate::lexer::FloatSuffix::F32 => Primitive::Float32,
        crate::lexer::FloatSuffix::F64 => Primitive::Float64,
    }
}
/// WP-C4.7-6.2: primitive types that have a total order AND a working ordered comparison in
/// both execution engines, so `a.cmp(&b)` can never disagree with `a < b`.
///
/// Excluded, deliberately:
/// - **Floats** — CD-015 (WP-C2.9) froze that primitive floats do not implement `Eq`/`Ord`/
///   `Hash`, so `1.0.cmp(&2.0)` must stay rejected.
/// - **`Unit`** — nothing to order.
/// - **`Bool`** — per DEV-075's owner specification decision, `Bool` implements `Eq` and `Hash`
///   but NOT `Ord`: its ordered operators and `Bool::cmp` are compile-time errors.
///
/// `Char` IS included (DEV-075): it is totally ordered by Unicode scalar value.
pub(super) fn ordered_primitive(ty: &Ty) -> bool {
    matches!(
        strip_ref(ty),
        Ty::Primitive(
            Primitive::Int8
                | Primitive::Int16
                | Primitive::Int32
                | Primitive::Int64
                | Primitive::UInt8
                | Primitive::UInt16
                | Primitive::UInt32
                | Primitive::UInt64
                | Primitive::Char
                | Primitive::String
                | Primitive::Str
        )
    )
}

/// The receiver type with any leading references removed (method receivers auto-deref).
pub(super) fn strip_ref(ty: &Ty) -> &Ty {
    let mut current = ty;
    while let Ty::Ref { inner, .. } = current {
        current = inner;
    }
    current
}

/// How a receiver form reads in a diagnostic.
pub(super) fn receiver_source(receiver: Option<hir::Receiver>) -> &'static str {
    match receiver {
        None => "no receiver",
        Some(hir::Receiver::Value) => "self",
        Some(hir::Receiver::Ref) => "&self",
        Some(hir::Receiver::RefMut) => "&mut self",
    }
}
