//! **AS7 Packet 7 — complete written-bound satisfaction.**
//!
//! The layer the original decomposition was missing. It may depend on `convert` and `traits`;
//! neither may depend on it, and the dependency checker enforces both directions.
//!
//! ```text
//! traits    does this type satisfy this trait IDENTITY, and which impl makes that true?
//! bounds    does it satisfy the COMPLETE written constraint, including `Item = Foo`?
//! ```
//!
//! Packet 7 stopped here because `convert` and `traits` formed a strongly connected pair:
//! converting `HashMap<K, V>` must prove `K: Hash + Eq`, and proving `Iterator<Item = Foo>` must
//! convert the written `Item`. Both directions are real, but the modules do not need a cycle —
//! what was missing was the orchestration of the two operations. See AS7-MODULE-DAG.md's
//! correction section.
//!
//! # Why the wrapper structure is behaviour-preserving, not merely plausible
//!
//! The original `satisfies_bound_parts` threaded `bound_args` through its whole recursion, so
//! splitting identity from bindings is only sound if bindings can be *consumed* in one place.
//! They can, and the trait matrices in this checker are what prove it:
//!
//! ```text
//! Ty::Ref arm       recurses only for Eq, Ord, Clone, Hash, Display — none has an associated type
//! Ty::Core arm      recurses only for Clone, Eq, Ord — likewise
//! Ty::Core Iterator returns a membership test and IGNORES bound_args entirely
//! Ty::Struct/Enum   the ONLY arm that reads a binding
//! ```
//!
//! So the binding path below reproduces exactly one thing from the original recursion — the `Ref`
//! peeling, for the same five bound names — and then consumes bindings at `Struct`/`Enum`, against
//! **the same impl `traits` selected**. Every other shape ignored bindings before and ignores them
//! now.

use super::state::TypeChecker;
use super::traits::BoundWitness;
use super::types::Ty;
use crate::hir;

impl TypeChecker<'_> {
    /// Whether `ty` satisfies a written bound, including any associated-type bindings.
    ///
    /// Identity first — if the trait relation does not hold, no binding can rescue it — then the
    /// bindings, if the constraint carries any.
    pub(super) fn satisfies_bound_parts(
        &mut self,
        ty: &Ty,
        bound_name: &str,
        bound_res: Option<hir::Res>,
        bound_args: Option<hir::GenericArgs>,
    ) -> bool {
        if !self.satisfies_bound_identity(ty, bound_name, bound_res) {
            return false;
        }
        let Some(args) = bound_args else {
            return true;
        };
        if !args
            .args
            .iter()
            .any(|arg| matches!(arg, hir::GenericArg::Binding { .. }))
        {
            return true;
        }
        self.written_bindings_match(ty, bound_name, bound_res, &args)
    }

    /// Compare each written `Name = Ty` binding against the selected impl's associated type.
    fn written_bindings_match(
        &mut self,
        ty: &Ty,
        bound_name: &str,
        bound_res: Option<hir::Res>,
        args: &hir::GenericArgs,
    ) -> bool {
        // Mirror the original recursion's ONE binding-relevant step: a reference is transparent
        // for exactly these five bounds, so `&MyStruct: Eq<..>` reads MyStruct's impl.
        let mut current = self.resolve(ty);
        while let Ty::Ref { inner, .. } = &current {
            if !matches!(bound_name, "Eq" | "Ord" | "Clone" | "Hash" | "Display") {
                return true;
            }
            current = self.resolve(inner);
        }
        let (Ty::Struct(struct_id, _) | Ty::Enum(struct_id, _)) = &current else {
            // Every other shape ignored bindings before the split and ignores them now.
            return true;
        };
        let BoundWitness::Impl(impl_item) =
            self.bound_impl_witness(struct_id, bound_name, bound_res)
        else {
            return false;
        };
        let associated = self.impl_assoc_type_ids(impl_item);
        args.args.iter().all(|arg| match arg {
            hir::GenericArg::Type(_) => true,
            hir::GenericArg::Const(_) => true,
            // Shape args do not appear in Core trait-bound bindings.
            hir::GenericArg::Shape(_) => true,
            hir::GenericArg::Binding { name, ty: expected } => {
                let Some(actual) = associated.get(self.text(*name)).copied() else {
                    return false;
                };
                let actual = self.convert_hir_type(actual);
                let expected = self.convert_hir_type(*expected);
                self.types_equal(&actual, &expected)
            }
        })
    }
}
