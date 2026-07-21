//! WP-C5.3d-1b — the canonical destruction plan (CD-062).
//!
//! Before this module, destruction order existed twice: `mir::interp::drop_in_place` walked the
//! type context recursively at run time, and `backend::generated_rust::emit_bodies::
//! emit_drop_glue` walked it again at emit time. The two walks agreed only because they were
//! written to agree, and they had already drifted once — the emitter's enum arm walked
//! `struct_fields` and dropped no payload at all (CD-060). That is a defect *class*: two
//! independent reconstructions of one semantic rule.
//!
//! [`DropPlan`] is the single derivation. It is **representation-neutral**: it names components
//! by MIR index and carries MIR types, and says nothing about frames, slots, Rust syntax, or
//! generated names. The MIR interpreter and the native emitter each consume the same plan and
//! differ only in how they *apply* a step.
//!
//! # The invariants the shape enforces
//!
//! - **The type's own destructor runs first.** [`DropPlan::Destructor`] nests the component plan
//!   inside it, so a consumer cannot run components first without deliberately taking the plan
//!   apart and reassembling it in the wrong order.
//! - **Components run in reverse declaration order.** The derivation reverses; consumers iterate
//!   forward. Order is not a consumer decision.
//! - **Every variant of an enum is covered.** [`DropPlan::Variants`] is indexed by variant number
//!   and always has one entry per variant, so a `match` built from it is exhaustive without a
//!   catch-all and a newly added variant cannot silently acquire a no-op arm.
//! - **A component that needs no action is absent.** Any type whose plan is [`DropPlan::Noop`] is
//!   omitted from its parent's field list, and a parent all of whose components are `Noop` (and
//!   which has no destructor of its own) is itself `Noop`. "Do not drop a `Copy` field" is
//!   therefore a property of the plan, not a filter each consumer has to remember to apply.
//!
//! MIR v0.1 is unchanged: this derives from [`MirTy`] and [`TypeContext`], both of which already
//! exist, and adds no node, terminator, or type to the IR.

use super::{EnumRef, MirTy, TypeContext};
use crate::hir::CoreType;

/// Derivation refused: the type context does not describe the type. Always a compiler bug or a
/// malformed program, never a user error — verified MIR carries a table for every nominal a
/// `Drop` can reach.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DropPlanError(pub String);

impl std::fmt::Display for DropPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One component of an aggregate that needs destruction, in the parent's own index space
/// (`Projection::Field(index)` for a struct or tuple, `Projection::VariantField(v, index)` for a
/// variant payload).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedField {
    pub index: u32,
    pub plan: DropPlan,
}

/// One variant's arm of an enum plan.
///
/// `arity` is the variant's FULL payload length, not just the droppable part: a consumer that
/// binds payload slots positionally (the native emitter builds `E::V1(_, __p1)`) needs the shape
/// of the variant, while `fields` says which of those slots actually get destroyed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariantPlan {
    pub arity: usize,
    pub fields: Vec<PlannedField>,
}

/// The canonical destruction plan for a MIR type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DropPlan {
    /// Nothing to destroy anywhere in this type.
    Noop,
    /// Run `symbol` — the type's own `Drop::drop` instance, receiving `&mut self` — and then
    /// `then`. Mutations the destructor makes are visible to `then`, which is why the destructor
    /// takes a reference rather than the value.
    Destructor { symbol: String, then: Box<DropPlan> },
    /// A struct's fields or a tuple's elements, already in destruction order. `base` is the
    /// aggregate's own type, which consumers need to spell a component access.
    Fields {
        base: MirTy,
        fields: Vec<PlannedField>,
    },
    /// An enum's variants, indexed by variant number; one entry per variant, always.
    Variants {
        base: MirTy,
        variants: Vec<VariantPlan>,
    },
    /// `len` elements of one type, destroyed at the indices [`array_order`] yields.
    Array { len: u64, elem: Box<DropPlan> },
    /// `Vec<T>`: elements back to front, then the buffer (unobservable — LAYOUT-QUERY-001).
    ///
    /// The element plan is named by TYPE and derived on demand rather than inlined, and so is
    /// [`DropPlan::BoxInner`]. `Vec` and `Box` are Core v1's only indirection and therefore its
    /// only route to a recursive type (`enum List { Nil, Cons(Int32, Box<List>) }`); inlining
    /// their element plans would not terminate. Every other component is stored inline, is
    /// necessarily finite, and is planned eagerly.
    VecElements { elem: MirTy },
    /// `Box<T>`: the contained value, then the allocation (unobservable).
    BoxInner { inner: MirTy },
}

impl DropPlan {
    /// Whether this plan has any effect. Used by the derivation to collapse empty aggregates and
    /// by consumers that want to skip a whole `Drop` cheaply.
    pub fn is_noop(&self) -> bool {
        matches!(self, DropPlan::Noop)
    }
}

/// The canonical destruction order for an array of `len` elements: back to front.
///
/// A free function rather than an implicit convention so that both consumers get the order from
/// the same place; reversing it in one consumer is then a visible edit to this call, not a
/// plausible-looking `for i in 0..len`.
pub fn array_order(len: u64) -> impl Iterator<Item = u64> {
    (0..len).rev()
}

/// The variant payload table for ANY enum, core or user — the one derivation the plan, the
/// verifier, the interpreter's place typing, and the backend's type emission all read.
///
/// `Option` is `None = 0` / `Some(T) = 1`; `Result` is `Ok(T) = 0` / `Err(E) = 1`; `Ordering` is
/// three fieldless variants `Less`/`Equal`/`Greater`.
pub fn variant_payloads(
    enum_ref: &EnumRef,
    args: &[MirTy],
    types: &TypeContext,
) -> Option<Vec<Vec<MirTy>>> {
    match enum_ref {
        EnumRef::CoreOption => Some(vec![vec![], vec![args.first()?.clone()]]),
        EnumRef::CoreResult => Some(vec![
            vec![args.first()?.clone()],
            vec![args.get(1)?.clone()],
        ]),
        EnumRef::CoreOrdering => Some(vec![vec![], vec![], vec![]]),
        EnumRef::User(item) => types.enum_variants.get(&(item.0, args.to_vec())).cloned(),
    }
}

/// Derive the destruction plan for `ty`.
pub fn plan_for(ty: &MirTy, types: &TypeContext) -> Result<DropPlan, DropPlanError> {
    match ty {
        MirTy::Struct(item, args) => {
            let fields = types
                .struct_fields
                .get(&(item.0, args.clone()))
                .cloned()
                .ok_or_else(|| DropPlanError(format!("no field table for struct #{}", item.0)))?;
            let planned = plan_components(&fields, types)?;
            let body = if planned.is_empty() {
                DropPlan::Noop
            } else {
                DropPlan::Fields {
                    base: ty.clone(),
                    fields: planned,
                }
            };
            Ok(with_destructor(item.0, args, types, body))
        }
        MirTy::Enum(enum_ref, args) => {
            let table = variant_payloads(enum_ref, args, types)
                .ok_or_else(|| DropPlanError(format!("no variant table for {ty:?}")))?;
            let mut variants = Vec::with_capacity(table.len());
            let mut any = false;
            for payload in &table {
                let fields = plan_components(payload, types)?;
                any |= !fields.is_empty();
                variants.push(VariantPlan {
                    arity: payload.len(),
                    fields,
                });
            }
            let body = if any {
                DropPlan::Variants {
                    base: ty.clone(),
                    variants,
                }
            } else {
                DropPlan::Noop
            };
            // Only a USER enum can carry a `Drop` impl; `Option`/`Result`/`Ordering` are core
            // types the program cannot write an impl for.
            Ok(match enum_ref {
                EnumRef::User(item) => with_destructor(item.0, args, types, body),
                _ => body,
            })
        }
        MirTy::Tuple(elems) => {
            let planned = plan_components(elems, types)?;
            Ok(if planned.is_empty() {
                DropPlan::Noop
            } else {
                DropPlan::Fields {
                    base: ty.clone(),
                    fields: planned,
                }
            })
        }
        MirTy::Array(elem, len) => {
            let plan = plan_for(elem, types)?;
            Ok(if plan.is_noop() || *len == 0 {
                DropPlan::Noop
            } else {
                DropPlan::Array {
                    len: *len,
                    elem: Box::new(plan),
                }
            })
        }
        // `Vec`/`Box` are never collapsed to `Noop` even when their element plan is: they own an
        // allocation, and a consumer that models the buffer needs the step to exist. The
        // element plan is deliberately not derived here (see `VecElements`).
        MirTy::Core(CoreType::Vec, args) => Ok(DropPlan::VecElements {
            elem: args.first().cloned().unwrap_or(MirTy::Unit),
        }),
        MirTy::Core(CoreType::Box, args) => Ok(DropPlan::BoxInner {
            inner: args.first().cloned().unwrap_or(MirTy::Unit),
        }),
        // Everything else destroys nothing through STARK glue.
        //
        // FLAGGED, and deliberately a faithful copy rather than a correction. The remaining
        // `Core` types — `String`, `HashMap`, `HashSet`, the iterators, `File` — reclaim their
        // own storage, and `mir::interp::drop_in_place` runs no element glue for any of them.
        // For `HashMap<K, V>` with a `V` that has a destructor that is arguably wrong, but it is
        // the reference semantics as they stand; changing it here would move the oracle without
        // an owner decision. Recorded so the question is answerable, not lost.
        _ => Ok(DropPlan::Noop),
    }
}

/// Plan a component list and drop the entries that need no action, preserving the parent's own
/// index for the ones that remain and reversing into destruction order.
fn plan_components(tys: &[MirTy], types: &TypeContext) -> Result<Vec<PlannedField>, DropPlanError> {
    let mut out = Vec::new();
    for (i, ty) in tys.iter().enumerate().rev() {
        let plan = plan_for(ty, types)?;
        if plan.is_noop() {
            continue;
        }
        out.push(PlannedField {
            index: i as u32,
            plan,
        });
    }
    Ok(out)
}

/// Wrap `body` in the nominal instance's own destructor, if it has one.
fn with_destructor(item: u32, args: &[MirTy], types: &TypeContext, body: DropPlan) -> DropPlan {
    match types.drop_impls.get(&(item, args.to_vec())) {
        Some(symbol) => DropPlan::Destructor {
            symbol: symbol.clone(),
            then: Box::new(body),
        },
        None => body,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::ItemId;

    fn ctx() -> TypeContext {
        TypeContext::default()
    }

    fn user_struct(id: u32) -> MirTy {
        MirTy::Struct(ItemId(id), Vec::new())
    }

    fn user_enum(id: u32) -> MirTy {
        MirTy::Enum(EnumRef::User(ItemId(id)), Vec::new())
    }

    /// A droppable leaf: a struct with its own destructor and no fields.
    fn droppable_leaf(types: &mut TypeContext, id: u32, symbol: &str) -> MirTy {
        types.struct_fields.insert((id, Vec::new()), Vec::new());
        types
            .drop_impls
            .insert((id, Vec::new()), symbol.to_string());
        user_struct(id)
    }

    #[test]
    fn primitives_and_all_copy_aggregates_plan_to_noop() {
        let mut types = ctx();
        types
            .struct_fields
            .insert((1, Vec::new()), vec![MirTy::Int32, MirTy::Bool]);
        assert!(plan_for(&MirTy::Int32, &types).unwrap().is_noop());
        assert!(plan_for(&MirTy::Unit, &types).unwrap().is_noop());
        assert!(plan_for(&user_struct(1), &types).unwrap().is_noop());
        assert!(
            plan_for(&MirTy::Tuple(vec![MirTy::Int32, MirTy::Char]), &types)
                .unwrap()
                .is_noop()
        );
        assert!(plan_for(&MirTy::Array(Box::new(MirTy::Int32), 4), &types)
            .unwrap()
            .is_noop());
    }

    /// The whole point of the shape: the destructor is structurally outside the components, so
    /// "user destructor first" is not something a consumer can get wrong by iterating in the
    /// order it happens to have written.
    #[test]
    fn own_destructor_nests_outside_the_component_plan() {
        let mut types = ctx();
        let leaf = droppable_leaf(&mut types, 2, "dtor_leaf");
        types
            .struct_fields
            .insert((1, Vec::new()), vec![leaf.clone()]);
        types
            .drop_impls
            .insert((1, Vec::new()), "dtor_outer".to_string());

        let plan = plan_for(&user_struct(1), &types).unwrap();
        let DropPlan::Destructor { symbol, then } = plan else {
            panic!("expected the outer destructor at the root, got {plan:?}");
        };
        assert_eq!(symbol, "dtor_outer");
        let DropPlan::Fields { fields, .. } = *then else {
            panic!("expected fields under the destructor");
        };
        assert_eq!(fields.len(), 1);
        assert!(matches!(
            fields[0].plan,
            DropPlan::Destructor { .. } // the field's own destructor, nested one level down
        ));
    }

    #[test]
    fn struct_fields_are_listed_in_reverse_declaration_order() {
        let mut types = ctx();
        let a = droppable_leaf(&mut types, 10, "dtor_a");
        let b = droppable_leaf(&mut types, 11, "dtor_b");
        let c = droppable_leaf(&mut types, 12, "dtor_c");
        types.struct_fields.insert((1, Vec::new()), vec![a, b, c]);

        let DropPlan::Fields { fields, .. } = plan_for(&user_struct(1), &types).unwrap() else {
            panic!("expected a field plan");
        };
        assert_eq!(
            fields.iter().map(|f| f.index).collect::<Vec<_>>(),
            vec![2, 1, 0],
            "components must be planned back to front"
        );
    }

    /// Copy fields are absent, and the surviving fields keep the DECLARATION index — a consumer
    /// projects with `Projection::Field(index)`, so a compacted index would address the wrong
    /// field.
    #[test]
    fn copy_fields_are_omitted_and_survivors_keep_their_declaration_index() {
        let mut types = ctx();
        let d = droppable_leaf(&mut types, 20, "dtor_d");
        types.struct_fields.insert(
            (1, Vec::new()),
            vec![MirTy::Int32, d, MirTy::Bool, MirTy::Char],
        );

        let DropPlan::Fields { fields, .. } = plan_for(&user_struct(1), &types).unwrap() else {
            panic!("expected a field plan");
        };
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].index, 1);
    }

    #[test]
    fn every_variant_gets_an_entry_even_when_its_payload_needs_nothing() {
        let mut types = ctx();
        let d = droppable_leaf(&mut types, 30, "dtor_e");
        types.enum_variants.insert(
            (1, Vec::new()),
            vec![
                vec![],                       // V0: fieldless
                vec![MirTy::Int32],           // V1: Copy payload
                vec![d.clone(), MirTy::Bool], // V2: one droppable, one Copy
            ],
        );

        let DropPlan::Variants { variants, .. } = plan_for(&user_enum(1), &types).unwrap() else {
            panic!("expected a variant plan");
        };
        assert_eq!(variants.len(), 3, "coverage must be complete");
        assert_eq!(variants[0].arity, 0);
        assert!(variants[0].fields.is_empty());
        assert_eq!(variants[1].arity, 1);
        assert!(variants[1].fields.is_empty());
        assert_eq!(variants[2].arity, 2, "arity is the FULL payload length");
        assert_eq!(variants[2].fields.len(), 1);
        assert_eq!(variants[2].fields[0].index, 0);
    }

    #[test]
    fn variant_payloads_are_listed_in_reverse_declaration_order() {
        let mut types = ctx();
        let a = droppable_leaf(&mut types, 40, "dtor_pa");
        let b = droppable_leaf(&mut types, 41, "dtor_pb");
        types
            .enum_variants
            .insert((1, Vec::new()), vec![vec![a, b]]);

        let DropPlan::Variants { variants, .. } = plan_for(&user_enum(1), &types).unwrap() else {
            panic!("expected a variant plan");
        };
        assert_eq!(
            variants[0]
                .fields
                .iter()
                .map(|f| f.index)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
    }

    #[test]
    fn an_enum_with_no_droppable_payload_anywhere_is_noop() {
        let mut types = ctx();
        types.enum_variants.insert(
            (1, Vec::new()),
            vec![vec![], vec![MirTy::Int32], vec![MirTy::Bool, MirTy::Char]],
        );
        assert!(plan_for(&user_enum(1), &types).unwrap().is_noop());
    }

    /// A core enum cannot carry a user `Drop` impl, but its type ARGUMENT can be droppable.
    #[test]
    fn core_option_plans_its_payload_without_looking_for_a_destructor() {
        let mut types = ctx();
        let d = droppable_leaf(&mut types, 50, "dtor_o");
        let ty = MirTy::Enum(EnumRef::CoreOption, vec![d]);

        let DropPlan::Variants { variants, .. } = plan_for(&ty, &types).unwrap() else {
            panic!("expected a variant plan");
        };
        assert_eq!(variants.len(), 2);
        assert!(variants[0].fields.is_empty(), "None carries nothing");
        assert_eq!(variants[1].fields.len(), 1);
    }

    #[test]
    fn core_result_covers_both_arms_independently() {
        let mut types = ctx();
        let d = droppable_leaf(&mut types, 51, "dtor_r");
        let ty = MirTy::Enum(EnumRef::CoreResult, vec![MirTy::Int32, d]);

        let DropPlan::Variants { variants, .. } = plan_for(&ty, &types).unwrap() else {
            panic!("expected a variant plan");
        };
        assert!(variants[0].fields.is_empty(), "Ok(Int32) needs no glue");
        assert_eq!(variants[1].fields.len(), 1, "Err(D) does");
    }

    #[test]
    fn array_of_droppable_carries_its_length_and_a_single_element_plan() {
        let mut types = ctx();
        let d = droppable_leaf(&mut types, 60, "dtor_arr");
        let plan = plan_for(&MirTy::Array(Box::new(d), 3), &types).unwrap();
        let DropPlan::Array { len, elem } = plan else {
            panic!("expected an array plan");
        };
        assert_eq!(len, 3);
        assert!(matches!(*elem, DropPlan::Destructor { .. }));
        assert_eq!(array_order(len).collect::<Vec<_>>(), vec![2, 1, 0]);
    }

    #[test]
    fn zero_length_array_is_noop() {
        let mut types = ctx();
        let d = droppable_leaf(&mut types, 61, "dtor_z");
        assert!(plan_for(&MirTy::Array(Box::new(d), 0), &types)
            .unwrap()
            .is_noop());
    }

    /// Vec and Box name their element by TYPE. This is what makes a recursive STARK type
    /// planable at all: deriving `Box<List>`'s inner plan eagerly would re-enter `List`.
    #[test]
    fn vec_and_box_defer_their_element_plan() {
        let types = ctx();
        assert_eq!(
            plan_for(&MirTy::Core(CoreType::Vec, vec![MirTy::Int32]), &types).unwrap(),
            DropPlan::VecElements { elem: MirTy::Int32 }
        );
        assert_eq!(
            plan_for(&MirTy::Core(CoreType::Box, vec![MirTy::Int32]), &types).unwrap(),
            DropPlan::BoxInner {
                inner: MirTy::Int32
            }
        );
    }

    #[test]
    fn a_recursive_type_through_box_terminates() {
        let mut types = ctx();
        // enum List { Nil, Cons(Int32, Box<List>) }
        let list = user_enum(70);
        types.enum_variants.insert(
            (70, Vec::new()),
            vec![
                vec![],
                vec![MirTy::Int32, MirTy::Core(CoreType::Box, vec![list.clone()])],
            ],
        );
        let DropPlan::Variants { variants, .. } = plan_for(&list, &types).unwrap() else {
            panic!("expected a variant plan");
        };
        assert_eq!(variants[1].fields.len(), 1);
        assert_eq!(variants[1].fields[0].index, 1);
    }

    #[test]
    fn a_missing_table_is_an_error_not_a_silent_noop() {
        let types = ctx();
        assert!(plan_for(&user_struct(99), &types).is_err());
        assert!(plan_for(&user_enum(99), &types).is_err());
    }
}
