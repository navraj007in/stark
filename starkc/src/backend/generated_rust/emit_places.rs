//! WP-C5.2b/C5.3a — place emission and the type environment projections resolve against.
//!
//! C5.2b supported bare locals only. C5.3a adds the aggregate projections: struct/tuple `Field`,
//! `ConstIndex`, and proof-backed `Index`. `VariantField` (enum payloads) is WP-C5.3b and
//! `Deref` needs references, which are not in the C5 subset yet.
//!
//! **Why a type environment is needed at all.** MIR's `Projection::Field(i)` is one variant
//! covering both struct fields and tuple elements, but the generated Rust differs (`.f0` on a
//! named struct, `.0` on a Rust tuple). Choosing between them requires knowing the *type of the
//! place being projected*, which means walking the projection chain from the local's declared
//! type through the nominal type context. [`TyEnv`] is that walk.

use super::{emit_types, BackendDiagnostic};
use crate::mir::{MirBody, MirTy, Place, Projection, TypeContext};

/// Matches MIR's own dump format (`_0`, `_1`, ...) one-to-one, which is also a valid Rust
/// identifier and (as a bonus, not the reason it was chosen) already suppresses Rust's
/// `unused_variables` lint on its own, since any leading-underscore identifier does.
pub fn local_name(local: u32) -> String {
    format!("_{local}")
}

/// The information a place needs to be emitted: the body's local declarations (where a
/// projection chain starts) and the program's nominal type context (how it continues).
#[derive(Clone, Copy)]
pub struct TyEnv<'a> {
    pub body: &'a MirBody,
    pub types: &'a TypeContext,
    /// WP-C5.3e: the named target layout contract a `LayoutQuery` is answered from. Threaded
    /// rather than defaulted at the use site, so a build cannot silently answer from a contract
    /// it was not given (CD-067).
    pub layout: &'a crate::layout::TargetLayout,
}

impl<'a> TyEnv<'a> {
    pub fn new(
        body: &'a MirBody,
        types: &'a TypeContext,
        layout: &'a crate::layout::TargetLayout,
    ) -> Self {
        TyEnv {
            body,
            types,
            layout,
        }
    }

    pub(super) fn local_ty(&self, local: u32) -> Result<MirTy, BackendDiagnostic> {
        self.body
            .locals
            .get(local as usize)
            .map(|d| d.ty.clone())
            .ok_or_else(|| {
                BackendDiagnostic::Unsupported(format!(
                    "place names local _{local} which the body does not declare"
                ))
            })
    }

    /// The type of `place`, resolved by walking its projection chain. Every error here is a
    /// backend limitation or a malformed place, never a user-source error — verified MIR
    /// guarantees projections are well-typed, so a failure means either an unsupported
    /// projection kind or a compiler defect upstream.
    pub fn place_ty(&self, place: &Place) -> Result<MirTy, BackendDiagnostic> {
        let mut ty = self.local_ty(place.local.0)?;
        for projection in &place.projection {
            ty = self.project_once(&ty, projection)?;
        }
        Ok(ty)
    }

    fn project_once(
        &self,
        base: &MirTy,
        projection: &Projection,
    ) -> Result<MirTy, BackendDiagnostic> {
        match (projection, base) {
            (Projection::Field(i), MirTy::Tuple(elems)) => {
                elems.get(*i as usize).cloned().ok_or_else(|| {
                    BackendDiagnostic::Unsupported(format!(
                        "tuple field {i} out of range for {base:?}"
                    ))
                })
            }
            (Projection::Field(i), MirTy::Struct(item, args)) => {
                let fields = self
                    .types
                    .struct_fields
                    .get(&(item.0, args.clone()))
                    .ok_or_else(|| {
                        BackendDiagnostic::Unsupported(format!(
                            "no type-context entry for struct instance {base:?} -- the nominal \
                             type context does not reach this instance"
                        ))
                    })?;
                fields.get(*i as usize).cloned().ok_or_else(|| {
                    BackendDiagnostic::Unsupported(format!(
                        "struct field {i} out of range for {base:?}"
                    ))
                })
            }
            (Projection::Index(_) | Projection::ConstIndex(_), MirTy::Array(elem, _)) => {
                Ok((**elem).clone())
            }
            (Projection::Deref, MirTy::Ref { inner, .. }) => Ok((**inner).clone()),
            (Projection::VariantField(v, i), MirTy::Enum(enum_ref, args)) => {
                let variants = emit_types::variant_payloads(enum_ref, args, self.types)
                    .ok_or_else(|| {
                        BackendDiagnostic::Unsupported(format!(
                            "no variant table for enum instance {base:?}"
                        ))
                    })?;
                variants
                    .get(*v as usize)
                    .and_then(|payload| payload.get(*i as usize))
                    .cloned()
                    .ok_or_else(|| {
                        BackendDiagnostic::Unsupported(format!(
                            "variant field v{v}.{i} out of range for {base:?}"
                        ))
                    })
            }
            (projection, base) => Err(BackendDiagnostic::Unsupported(format!(
                "projection {projection:?} on {base:?} has no WP-C5.3a representation yet -- \
                 VariantField lands in WP-C5.3b, Deref needs references (not in the C5 subset), \
                 and indexing a Slice/Vec needs their runtime representations"
            ))),
        }
    }
}

/// Whether a local is backed by a `ValueSlot` (WP-C5.3d-0) rather than a bare Rust value.
/// Exactly the non-`Copy` locals: `Copy` values need no liveness tracking, because MIR never
/// moves out of them and Rust never destroys them.
pub fn is_slot_local(local: u32, env: &TyEnv) -> Result<bool, BackendDiagnostic> {
    let ty = env.local_ty(local)?;
    Ok(emit_types::is_slot_backed(&ty, env.types))
}

/// A place in READ position. A slot-backed local reads through `get()`, which borrows rather
/// than moves — the whole point of the slot is that reading does not disturb liveness.
pub fn emit_place(place: &Place, env: &TyEnv) -> Result<String, BackendDiagnostic> {
    emit_place_from(place, env, PlaceMode::Read)
}

/// A place in WRITE position: the base is accessed mutably. Only valid for a PROJECTED place —
/// a whole-local assignment to a slot goes through `write()` instead, which is a statement, not
/// a place expression. `emit_bodies` branches on that before calling here.
pub fn emit_place_mut(place: &Place, env: &TyEnv) -> Result<String, BackendDiagnostic> {
    emit_place_from(place, env, PlaceMode::Write)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaceMode {
    Read,
    Write,
}

fn emit_place_from(
    place: &Place,
    env: &TyEnv,
    mode: PlaceMode,
) -> Result<String, BackendDiagnostic> {
    // WP-C5.3d-0. A single-level READ of a `Copy` field out of a slot goes through a raw
    // projection helper rather than `get()`, because it must keep working after a SIBLING drop
    // unit has been moved out -- at which point the storage no longer holds a valid complete
    // value and `get()` is correctly refused. Enum payload reads are excluded: they have no raw
    // projection syntax, and an enum's payload is read while the value is still whole.
    if mode == PlaceMode::Read && place.projection.len() == 1 && is_slot_local(place.local.0, env)?
    {
        let base_ty = env.local_ty(place.local.0)?;
        let field_ty = env.place_ty(place)?;
        let raw_base = matches!(
            (&place.projection[0], &base_ty),
            (Projection::Field(_), MirTy::Struct(..) | MirTy::Tuple(_))
                | (Projection::ConstIndex(_), MirTy::Array(..))
        );
        if raw_base && emit_types::mir_ty_is_copy(&field_ty, env.types) {
            // Mirrors `emit_projections::collect_operand` case 2 exactly; the two must agree, or
            // emission would name a helper the collector never generated.
            let helper = super::emit_projections::collect_for_place(
                place,
                env,
                super::emit_projections::HelperOp::Copy,
            )?;
            return Ok(format!(
                "stark_proj::{helper}(&{})",
                local_name(place.local.0)
            ));
        }
    }
    let mut rendered = local_name(place.local.0);
    if is_slot_local(place.local.0, env)? {
        rendered = match mode {
            PlaceMode::Read => format!("{rendered}.get()"),
            PlaceMode::Write => format!("{rendered}.get_mut()"),
        };
    }
    let mut ty = env.local_ty(place.local.0)?;

    for projection in &place.projection {
        match (projection, &ty) {
            // A Rust tuple element is `.0`; a generated named struct's field is `.f0`. One MIR
            // variant, two Rust syntaxes -- the reason `TyEnv` exists.
            (Projection::Field(i), MirTy::Tuple(_)) => rendered.push_str(&format!(".{i}")),
            (Projection::Field(i), MirTy::Struct(..)) => {
                rendered.push_str(&format!(".{}", emit_types::field_name(*i)))
            }
            // A5/CD-038: statically known and verifier-checked against the array length, so it
            // needs no bounds check of its own.
            (Projection::ConstIndex(n), MirTy::Array(..)) => rendered.push_str(&format!("[{n}]")),
            // WP-C5.3d-1a: reading through an ephemeral reference. Parenthesised because the
            // projection chain may continue (`(*_6).f0`).
            (Projection::Deref, MirTy::Ref { .. }) => rendered = format!("(*{rendered})"),
            // A proof-backed index: `CheckedOp::CheckIndex` already validated it and trapped
            // otherwise, so this is a plain index expression rather than a second check.
            (Projection::Index(proof), MirTy::Array(..)) => {
                rendered.push_str(&format!("[{} as usize]", local_name(proof.0)))
            }
            // WP-C5.3b. Rust has NO way to project into an enum variant's field outside a
            // `match`, so this is the one projection that cannot be a suffix: it wraps what came
            // before. The `_` arm is provably dead -- V-DISC-1 makes a variant-field projection
            // legal only after a discriminant test -- so it takes the same `unreachable!()` the
            // verifier-proved dead-block path already uses.
            //
            // `match &base` (not `match base`) keeps the read non-consuming; the field is then
            // dereferenced, which requires the FIELD type to be Copy. A non-Copy payload would
            // have to be moved out, which needs WP-C5.3d's controlled storage for the same
            // reason C5.3a's cross-block moves do.
            (Projection::VariantField(v, i), MirTy::Enum(enum_ref, args)) => {
                let field_ty = env.project_once(&ty, projection)?;
                if !emit_types::mir_ty_is_copy(&field_ty, env.types) {
                    return Err(BackendDiagnostic::Unsupported(format!(
                        "reading the non-Copy payload field v{v}.{i} of {ty:?} needs WP-C5.3d's \
                         controlled storage: `match &e` yields a reference, and moving out of it \
                         runs into the same block-dispatch limit as C5.3a's cross-block moves"
                    )));
                }
                let enum_name = emit_types::nominal_type_name(&ty).ok_or_else(|| {
                    BackendDiagnostic::Unsupported(format!("no generated name for {ty:?}"))
                })?;
                let arity = emit_types::variant_payloads(enum_ref, args, env.types)
                    .and_then(|variants| variants.get(*v as usize).cloned())
                    .map(|payload| payload.len())
                    .ok_or_else(|| {
                        BackendDiagnostic::Unsupported(format!(
                            "variant v{v} unresolvable for {ty:?}"
                        ))
                    })?;
                let mut binders: Vec<String> = (0..arity).map(|_| "_".to_string()).collect();
                binders[*i as usize] = "__payload".to_string();
                rendered = format!(
                    "(match &{rendered} {{ {enum_name}::{}({}) => *__payload, \
                     _ => unreachable!(\"V-DISC-1: variant-field projection without a \
                     discriminant test\") }})",
                    emit_types::variant_name(*v),
                    binders.join(", ")
                );
            }
            _ => {
                return Err(BackendDiagnostic::Unsupported(format!(
                    "projection {projection:?} on {ty:?} has no WP-C5.3b representation yet"
                )))
            }
        }
        ty = env.project_once(&ty, projection)?;
    }
    Ok(rendered)
}

/// A move OUT of a non-`Copy` place (WP-C5.3d-0).
///
/// **Whole-local moves only in this increment.** A sub-place move is sound in `ValueSlot`
/// (`move_field`), but it requires a per-type, per-field *projection helper* the backend must
/// generate — partial access is defined only through raw pointers, never through `&mut T`, so it
/// cannot be expressed by splicing a closure over the slot's contents at the call site. Emitting
/// those helpers is the next increment; until then a sub-place move is refused here, before
/// rustc, which is what deliverable 6 requires of any partial-move form not yet implemented.
///
/// Refusing is not merely conservative. The owner review of the first `ValueSlot` design found
/// that a partially moved value cannot be touched through `&T`/`&mut T` at all: a whole-value
/// read, take, or drop over it is undefined behaviour, and Miri reproduces the drop case as a
/// use-after-free. A sub-place move emitted through the whole-value path would therefore be
/// silently unsound rather than merely unsupported.
pub fn emit_move_out(place: &Place, env: &TyEnv) -> Result<String, BackendDiagnostic> {
    let local = local_name(place.local.0);
    if !is_slot_local(place.local.0, env)? {
        return Err(BackendDiagnostic::Unsupported(format!(
            "move out of the non-slot place {place:?}"
        )));
    }
    if !place.projection.is_empty() {
        // A sub-place move goes through a GENERATED projection helper, so the emitted body
        // contains no `unsafe` of its own. Which slot operation applies depends on whether the
        // sub-place has raw-pointer syntax: struct/tuple/array projections do and therefore work
        // on already-partial storage; an enum payload does not, and is legal only while the slot
        // is still whole.
        let helper = super::emit_projections::collect_for_place(
            place,
            env,
            super::emit_projections::HelperOp::Move,
        )?;
        return Ok(format!("stark_proj::{helper}(&mut {local})"));
    }
    Ok(format!("{local}.take()"))
}

/// Whether `place` reads through an enum variant's payload. Such a place emits as a `match`
/// EXPRESSION (see [`emit_place`]), which is not a Rust place expression and therefore cannot be
/// an assignment destination. Callers that splice a place on the left of `=` must refuse it.
///
/// Reachability check, so this is a guard and not a limitation: MIR lowering emits
/// `VariantField` through `read_place` and pattern tests only, never as an assignment
/// destination, and STARK source has no syntax for assigning into an enum payload.
pub fn reads_through_variant_field(place: &Place) -> bool {
    place
        .projection
        .iter()
        .any(|p| matches!(p, Projection::VariantField(..)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{BlockId, Instance, LocalDecl, LocalId, LocalKind};

    fn body_with(locals: Vec<MirTy>) -> MirBody {
        MirBody {
            instance: Instance {
                item: crate::hir::ItemId(0),
                symbol: "t@[]".to_string(),
                type_args: vec![],
            },
            params: vec![],
            ret: MirTy::Unit,
            locals: locals
                .into_iter()
                .map(|ty| LocalDecl {
                    ty,
                    kind: LocalKind::Temp,
                })
                .collect(),
            blocks: vec![],
            entry: BlockId(0),
        }
    }

    #[test]
    fn bare_local_emits_its_dump_name() {
        let body = body_with(vec![MirTy::Unit, MirTy::Int32, MirTy::Int32, MirTy::Int32]);
        let types = TypeContext::default();
        let layout = crate::layout::TargetLayout::default();
        let env = TyEnv::new(&body, &types, &layout);
        assert_eq!(emit_place(&Place::local(LocalId(3)), &env).unwrap(), "_3");
    }

    /// One MIR `Field` projection, two Rust syntaxes, chosen by the base type -- the property
    /// that makes `TyEnv` necessary rather than merely convenient.
    ///
    /// Also pins the WP-C5.3d-0 slot distinction: the tuple of primitives is `Copy` and reads
    /// directly, while the struct (no `impl Copy`, so non-`Copy` in MIR) is slot-backed and reads
    /// through `get()`. Both syntaxes appear in one test so a change to either is visible.
    #[test]
    fn field_syntax_differs_between_tuples_and_structs() {
        let body = body_with(vec![
            MirTy::Tuple(vec![MirTy::Int32, MirTy::Bool]),
            MirTy::Struct(crate::hir::ItemId(0), vec![]),
        ]);
        let mut types = TypeContext::default();
        types
            .struct_fields
            .insert((0, vec![]), vec![MirTy::Int32, MirTy::Bool]);
        let layout = crate::layout::TargetLayout::default();
        let env = TyEnv::new(&body, &types, &layout);

        let tuple_field = Place {
            local: LocalId(0),
            projection: vec![Projection::Field(1)],
        };
        assert_eq!(emit_place(&tuple_field, &env).unwrap(), "_0.1");

        let struct_field = Place {
            local: LocalId(1),
            projection: vec![Projection::Field(1)],
        };
        // WP-C5.3d-0: a single-level Copy field read on a slot local goes through a raw
        // projection helper, so it keeps working after a SIBLING field has been moved out.
        assert_eq!(
            emit_place(&struct_field, &env).unwrap(),
            format!(
                "stark_proj::{}(&_1)",
                super::super::emit_projections::helper_name(
                    &MirTy::Struct(crate::hir::ItemId(0), vec![]),
                    &Projection::Field(1),
                    super::super::emit_projections::HelperOp::Copy
                )
            )
        );
    }

    #[test]
    fn nested_projections_resolve_through_the_type_context() {
        // A struct whose second field is a tuple: `_0.f1.0` cannot be emitted without first
        // resolving what `.f1` is.
        let body = body_with(vec![MirTy::Struct(crate::hir::ItemId(0), vec![])]);
        let mut types = TypeContext::default();
        types.struct_fields.insert(
            (0, vec![]),
            vec![MirTy::Int32, MirTy::Tuple(vec![MirTy::Bool, MirTy::Int64])],
        );
        let layout = crate::layout::TargetLayout::default();
        let env = TyEnv::new(&body, &types, &layout);
        let place = Place {
            local: LocalId(0),
            projection: vec![Projection::Field(1), Projection::Field(0)],
        };
        // `_0` is a non-Copy struct, hence slot-backed; the projection chain continues from
        // the borrowed value.
        assert_eq!(emit_place(&place, &env).unwrap(), "_0.get().f1.0");
        assert_eq!(env.place_ty(&place).unwrap(), MirTy::Bool);
    }

    #[test]
    fn array_indices_emit_both_forms() {
        let body = body_with(vec![MirTy::Array(Box::new(MirTy::Int32), 3), MirTy::Int64]);
        let types = TypeContext::default();
        let layout = crate::layout::TargetLayout::default();
        let env = TyEnv::new(&body, &types, &layout);

        let constant = Place {
            local: LocalId(0),
            projection: vec![Projection::ConstIndex(2)],
        };
        assert_eq!(emit_place(&constant, &env).unwrap(), "_0[2]");

        let proved = Place {
            local: LocalId(0),
            projection: vec![Projection::Index(LocalId(1))],
        };
        assert_eq!(emit_place(&proved, &env).unwrap(), "_0[_1 as usize]");
    }

    /// Enum payloads are WP-C5.3b: rejected loudly rather than guessed at.
    #[test]
    fn variant_field_is_still_unsupported() {
        let body = body_with(vec![MirTy::Int32]);
        let types = TypeContext::default();
        let layout = crate::layout::TargetLayout::default();
        let env = TyEnv::new(&body, &types, &layout);
        let place = Place {
            local: LocalId(0),
            projection: vec![Projection::VariantField(0, 0)],
        };
        assert!(matches!(
            emit_place(&place, &env),
            Err(BackendDiagnostic::Unsupported(_))
        ));
    }
}
