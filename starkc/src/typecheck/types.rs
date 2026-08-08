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
use crate::extensions::tensor::types::TensorKind;
use crate::hir::{CoreType, ItemId};
use std::collections::HashMap;

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
