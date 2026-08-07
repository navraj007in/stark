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

use crate::hir::{self, BlockId, Hir, ItemId};
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

/// One impl, reduced to what specialisation needs.
#[derive(Debug, Clone)]
pub struct IndexedImpl {
    pub impl_item: ItemId,
    /// Which trait this impl provides — `None` for an inherent impl.
    pub trait_: Option<hir::BoundTrait>,
    /// The impl's `Self` type, still parametric.
    pub self_ty: Ty,
    /// The impl's own generic parameter names, in declaration order.
    pub generic_names: Vec<String>,
    /// `method name -> (member index, body)`.
    pub members: Vec<(String, u32, BlockId)>,
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
/// `self_ty` is the obligation's `Self` **already substituted** through the caller's environment —
/// the caller knows its own instantiation, and asking it to pass the result keeps this function
/// free of any notion of a caller frame.
///
/// **Trait identity is matched, never spelling.** DEV-BOUND-TRAIT-IDENTITY is the reason, and the
/// Iterator hardening is the reason it is worth repeating.
pub fn specialize_bound_callable(
    index: &TraitImplIndex,
    hir: &Hir,
    trait_: hir::BoundTrait,
    member: &str,
    self_ty: &Ty,
) -> Option<ResolvedCallable> {
    for candidate in &index.impls {
        if candidate.trait_ != Some(trait_) {
            continue;
        }
        let mut map: HashMap<String, Ty> = HashMap::new();
        if !unify_impl_ty_with(&candidate.self_ty, self_ty, &mut map, &|ty| ty.clone()) {
            continue;
        }
        let Some((_, index_in_impl, body)) = candidate
            .members
            .iter()
            .find(|(name, _, _)| name == member)
            .cloned()
        else {
            continue;
        };
        // The impl's own parameters, bound to what the match resolved — so `impl<T> Display for
        // W<T>` specialised at `W<Int32>` carries `T = Int32` rather than an empty environment.
        let mut environment: Vec<(GenericBinder, Ty)> = Vec::new();
        environment.push((GenericBinder::SelfType, self_ty.clone()));
        for (position, name) in candidate.generic_names.iter().enumerate() {
            if let Some(ty) = map.get(name) {
                environment.push((
                    GenericBinder::ImplParam {
                        index: position,
                        name: name.clone(),
                    },
                    ty.clone(),
                ));
            }
        }
        let signature = declared_signature(hir, candidate.impl_item, index_in_impl, self_ty)?;
        return Some(ResolvedCallable {
            declaration: CallableDeclId::ImplMember {
                impl_item: candidate.impl_item,
                member: index_in_impl,
            },
            body,
            environment,
            signature,
        });
    }
    None
}

/// The selected member's signature, with the receiver formed as the declaration binds it.
///
/// **INCOMPLETE, and stated as such.** Parameter and result types require the checker's
/// `convert_hir_type`, which is not reachable from here — so they are left empty and `Ty::Error`
/// rather than guessed. Step 4 either threads the conversion in or has the checker precompute each
/// member's parametric signature into `IndexedImpl`, at which point §3.4's invariant applies to a
/// specialised use exactly as it does to a static one.
///
/// Publishing a fabricated signature here would be the packet's own defect class: a second answer
/// to what a callable's signature is.
fn declared_signature(
    hir: &Hir,
    impl_item: ItemId,
    member: u32,
    self_ty: &Ty,
) -> Option<CallableSigTy> {
    let hir::ItemKind::Impl { items, .. } = &hir.item(impl_item).kind else {
        return None;
    };
    let hir::ImplItem::Fn { def, .. } = items.get(member as usize)? else {
        return None;
    };
    let receiver = match def.sig.receiver? {
        hir::Receiver::Value => self_ty.clone(),
        hir::Receiver::Ref => Ty::Ref {
            mutable: false,
            inner: Box::new(self_ty.clone()),
        },
        hir::Receiver::RefMut => Ty::Ref {
            mutable: true,
            inner: Box::new(self_ty.clone()),
        },
    };
    Some(CallableSigTy {
        receiver: Some(receiver),
        params: Vec::new(),
        ret: Ty::Error,
    })
}
