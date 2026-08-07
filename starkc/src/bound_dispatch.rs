//! **The one bound-specialisation authority** (AS3 Boundary 4, step 3).
//!
//! `CalleeSelection::Bound` records an obligation the checker fixed — trait, member, and a `Self`
//! that may still be parametric — without a body, because there is none until `Self` becomes
//! concrete. This module is what makes it concrete, and it is deliberately in **neither engine**:
//!
//! ```text
//!                CallableUse::Bound
//!         trait + member + parametric Self
//!                        │
//!            ┌───────────┴───────────┐
//!      HIR generic frame       MIR monomorphisation
//!            └───────────┬───────────┘
//!                        ▼
//!             specialize_bound_callable
//!                        ▼
//!         declaration + body + environment + signature
//! ```
//!
//! The two engines call it at different *times* — the interpreter when its generic frame makes
//! `Self` concrete, MIR when monomorphisation does — but neither implements matching. That is the
//! whole point: `Bound` must not become a respectable name for the scans AS3 is removing.
//!
//! **Body and environment are produced atomically.** Choosing a body in one place and
//! reconstructing its environment in another is how DEV-176 happened; a specialiser that returned
//! only a body would invite exactly that.

use crate::hir::{self, BlockId, ItemId};
use crate::typecheck::{CallableDeclId, CallableSigTy, GenericBinder, Ty};
use std::collections::HashMap;

/// The structural impl matcher, shared with `typecheck`.
///
/// The checker and the specialiser differ only in how they resolve inference variables — the
/// checker through its substitution, the specialiser not at all, since specialisation types are
/// grounded. That difference is a parameter rather than a fork, so there is one matcher.
pub fn unify_impl_ty_with(
    implementation: &Ty,
    receiver: &Ty,
    map: &mut HashMap<String, Ty>,
    resolve: &dyn Fn(&Ty) -> Ty,
) -> bool {
    let (imp, recv) = (resolve(implementation), resolve(receiver));
    match (imp, recv) {
        // A parameter absorbs whatever it is matched against, but must stay CONSISTENT: the same
        // parameter appearing twice (`Pair<T, T>`) has to see the same type both times.
        (Ty::Param(name), recv) => match map.get(&name) {
            Some(bound) if !matches!(bound, Ty::Param(p) if *p == name) => bound == &recv,
            _ => {
                map.insert(name, recv);
                true
            }
        },
        (Ty::Struct(l, l_args), Ty::Struct(r, r_args))
        | (Ty::Enum(l, l_args), Ty::Enum(r, r_args))
            if l == r && l_args.len() == r_args.len() =>
        {
            l_args
                .iter()
                .zip(&r_args)
                .all(|(l, r)| unify_impl_ty_with(l, r, map, resolve))
        }
        (Ty::Core(l, l_args), Ty::Core(r, r_args)) if l == r && l_args.len() == r_args.len() => {
            l_args
                .iter()
                .zip(&r_args)
                .all(|(l, r)| unify_impl_ty_with(l, r, map, resolve))
        }
        (Ty::Tuple(l), Ty::Tuple(r)) if l.len() == r.len() => l
            .iter()
            .zip(&r)
            .all(|(l, r)| unify_impl_ty_with(l, r, map, resolve)),
        (
            Ty::Ref {
                mutable: lm,
                inner: li,
            },
            Ty::Ref {
                mutable: rm,
                inner: ri,
            },
        ) if lm == rm => unify_impl_ty_with(&li, &ri, map, resolve),
        (Ty::Array(l, ln), Ty::Array(r, rn)) if ln == rn => {
            unify_impl_ty_with(&l, &r, map, resolve)
        }
        (Ty::Slice(l), Ty::Slice(r)) => unify_impl_ty_with(&l, &r, map, resolve),
        (left, right) => left == right,
    }
}

/// Where an effective callable target lives, and which binder namespace owns its body.
///
/// G1 (`AS3-DISPLAY-CHARACTERIZATION.md` §5): `impl Describe for A2 {}` with no override runs the
/// **trait default**. So the index records the *effective* target, not members physically written
/// inside an impl — and the two targets own **different binder namespaces**: an impl override's
/// body owns `ImplParam`, a trait default's owns `TraitParam`. Deciding that in the specialiser
/// would mean bespoke per-kind environment building, one edit away from populating a default body's
/// environment with impl binders.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexedTarget {
    pub member: String,
    pub declaration: CallableDeclId,
    pub body: BlockId,
    /// The binders **this executable body** owns, named by the checker, which is the only party
    /// that knows the declaration. The specialiser supplies their values and branches on nothing.
    ///
    /// Method generics are positional, not nominal: `trait_method_signature_matches` builds
    /// separate trait-side and impl-side name→index maps, so `fn to<U>` may be implemented as
    /// `fn to<V>`. The binder carries the index the body actually uses.
    pub binders: Vec<GenericBinder>,
}

/// One impl head, reduced to what specialisation needs. **No signature** — `callable_types[body]`
/// is the sole signature authority (A3b), and a copy here would be a second one.
#[derive(Debug, Clone)]
pub struct IndexedImpl {
    pub impl_item: ItemId,
    /// Which trait this impl provides — `None` for an inherent impl.
    pub trait_: Option<hir::BoundTrait>,
    /// The trait's arguments, still parametric. Matched against the obligation's concrete arguments
    /// through the **same** substitution map as `self_ty`, so `impl<T> Convert<T> for W<T>` binds
    /// `T` once from both sides rather than twice inconsistently.
    pub trait_args: Vec<Ty>,
    /// The impl's `Self` type, still parametric.
    pub self_ty: Ty,
    /// The impl's own generic parameter names, in declaration order.
    pub generic_names: Vec<String>,
    /// Effective targets, one per member name.
    pub effective_members: Vec<IndexedTarget>,
}

/// The program's coherent impl set, built **once** by the analysis phase.
///
/// Built at check time because that is where impl self-types are converted and named; consulted at
/// execution time by both engines. Building it in each engine would be two indexes of one fact.
#[derive(Debug, Clone, Default)]
pub struct TraitImplIndex {
    impls: Vec<IndexedImpl>,
}

impl TraitImplIndex {
    pub fn from_parts(impls: Vec<IndexedImpl>) -> Self {
        TraitImplIndex { impls }
    }

    /// Every indexed impl, in declaration order.
    pub fn impls(&self) -> &[IndexedImpl] {
        &self.impls
    }

    pub fn len(&self) -> usize {
        self.impls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.impls.is_empty()
    }
}

/// What specialisation produces — the four facts, together.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedCallable {
    pub declaration: CallableDeclId,
    pub body: BlockId,
    pub environment: Vec<(GenericBinder, Ty)>,
    pub signature: CallableSigTy,
}

/// Resolve a `Bound` obligation against a concrete `Self`.
///
/// **4b builds this.** 4a lands the frozen index shape only; wiring the specialiser to
/// `callable_types[body]` and the binder schema is the next sub-packet, kept separate so a failure
/// is attributable to construction or to resolution but not to both.
pub fn specialize_bound_callable(
    _index: &TraitImplIndex,
    _trait_: hir::BoundTrait,
    _member: &str,
    _self_ty: &Ty,
) -> Option<ResolvedCallable> {
    None
}
