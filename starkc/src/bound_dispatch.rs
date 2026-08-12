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
/// What the specialiser owns, and nothing more.
///
/// **`signature` was removed by AC4-F2 (owner decision, 2026-08-12).** The specialiser used to
/// return a fourth fact — the parametric signature with the environment substituted in — and no
/// execution path consumed it: production reads `body` (six sites across the HIR interpreter and
/// the MIR lowerer) and `environment` (one), and nothing read `signature`. AC4 proved the
/// construction observationally irrelevant by corrupting it, and the repair is deletion rather than
/// a manufactured consumer: **architecture documentation should describe what execution needs, not
/// make execution consume something because the documentation promised it.**
///
/// `declaration` is retained on different grounds, verified rather than assumed by association. It
/// is a field COPY of an already-made selection — no computation — and it is the witness that
/// distinguishes an impl override from a trait default, which `as3_callable_use_keying` asserts and
/// which is a real semantic distinction (G1's case).
///
/// The contract this leaves is DEV-176's reason for existing, exactly:
///
/// ```text
/// bound obligation -> selected executable body + the environment required to execute it
/// ```
pub struct ResolvedCallable {
    pub declaration: CallableDeclId,
    pub body: BlockId,
    pub environment: Vec<(GenericBinder, Ty)>,
}

/// Resolve a `Bound` obligation against a concrete `Self` (AS3 Boundary 4b).
///
/// The caller substitutes its **own** generic environment first and passes concrete types, so this
/// function has no notion of a caller frame — the interpreter calls it when its generic frame makes
/// `Self` concrete, MIR when monomorphisation does, and neither implements matching.
///
/// Four authorities, one each:
///
/// ```text
/// body selection      TraitImplIndex
/// generic matching    unify_impl_ty_with
/// signature           callable_types[body]      (A3b)
/// substitution        substitute_ty
/// ```
pub fn specialize_bound_callable(
    index: &TraitImplIndex,
    callable_types: &HashMap<BlockId, CallableSigTy>,
    trait_: hir::BoundTrait,
    member: &str,
    self_ty: &Ty,
    trait_args: &[Ty],
    method_args: &[Ty],
) -> Option<ResolvedCallable> {
    for candidate in &index.impls {
        if candidate.trait_ != Some(trait_) {
            continue;
        }
        // **One substitution map across `Self` AND the trait arguments.** `impl<T> Convert<T> for
        // W<T>` must bind `T` once from both sides; two maps could bind it inconsistently and the
        // index would be authoritative only for non-parameterised traits.
        let mut map: HashMap<String, Ty> = HashMap::new();
        let identity = |ty: &Ty| ty.clone();
        if !unify_impl_ty_with(&candidate.self_ty, self_ty, &mut map, &identity) {
            continue;
        }
        if candidate.trait_args.len() != trait_args.len() {
            continue;
        }
        if !candidate
            .trait_args
            .iter()
            .zip(trait_args)
            .all(|(parametric, concrete)| {
                unify_impl_ty_with(parametric, concrete, &mut map, &identity)
            })
        {
            continue;
        }
        // `continue`, not `?`. A `?` here abandoned the whole search when the FIRST impl whose
        // head unified did not declare the member — so a later impl that does would never be
        // reached. Wrong control flow independently of DEV-187, and the kind that only shows up
        // once a program has two impls of one trait.
        let Some(target) = candidate
            .effective_members
            .iter()
            .find(|t| t.member == member)
        else {
            continue;
        };

        // The environment is read off the TARGET's binder schema, so an impl override and a trait
        // default are built by the same code from different declarations. Branching on target kind
        // here is what would let a default body inherit impl binders.
        let mut environment: Vec<(GenericBinder, Ty)> = Vec::new();
        for binder in &target.binders {
            let value = match binder {
                GenericBinder::SelfType => Some(self_ty.clone()),
                GenericBinder::ImplParam { name, .. } => map.get(name).cloned(),
                GenericBinder::TraitParam { index, .. } => trait_args.get(*index).cloned(),
                GenericBinder::MethodParam { index, .. } => method_args.get(*index).cloned(),
                // A free function's own parameter cannot appear on a trait member's body.
                GenericBinder::FunctionParam { .. } => None,
            };
            if let Some(value) = value {
                environment.push((binder.clone(), value));
            }
        }

        // A3b's exact-set test guarantees every executable body has a signature, trait defaults
        // included. A missing one is an internal inconsistency, not a case to tolerate — the same
        // rule `publish_named_use` learned.
        //
        // **This lookup is a VALIDITY CHECK, and that is now its whole job** (AC4-F2). It used to
        // also supply the parametric signature that this function specialised and returned. That
        // specialised signature was dead: AC4 emptied its substitution map — so every
        // bound-specialised callable kept `T` instead of its instantiated type — and ~850 tests
        // passed, including the three-engine differential, while a probe showed the map being
        // built with real bindings six times in two suites. Constructed, never consumed.
        //
        // The `continue` is kept deliberately. Removing the lookup with the signature would have
        // let a candidate whose body has NO registered signature be returned instead of skipped,
        // which is a behaviour change wearing a deletion's clothes.
        if callable_types.get(&target.body).is_none() {
            continue;
        }

        return Some(ResolvedCallable {
            declaration: target.declaration,
            body: target.body,
            environment,
        });
    }
    None
}
