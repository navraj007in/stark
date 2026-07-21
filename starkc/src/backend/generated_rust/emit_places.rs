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
}

impl<'a> TyEnv<'a> {
    pub fn new(body: &'a MirBody, types: &'a TypeContext) -> Self {
        TyEnv { body, types }
    }

    fn local_ty(&self, local: u32) -> Result<MirTy, BackendDiagnostic> {
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
            (
                Projection::VariantField(v, i),
                MirTy::Enum(crate::mir::EnumRef::User(item), args),
            ) => {
                let variants = self
                    .types
                    .enum_variants
                    .get(&(item.0, args.clone()))
                    .ok_or_else(|| {
                        BackendDiagnostic::Unsupported(format!(
                            "no type-context entry for enum instance {base:?}"
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

pub fn emit_place(place: &Place, env: &TyEnv) -> Result<String, BackendDiagnostic> {
    let mut rendered = local_name(place.local.0);
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
            (
                Projection::VariantField(v, i),
                MirTy::Enum(crate::mir::EnumRef::User(item), args),
            ) => {
                let field_ty = env.project_once(&ty, projection)?;
                if !emit_types::mir_ty_is_copy(&field_ty, env.types) {
                    return Err(BackendDiagnostic::Unsupported(format!(
                        "reading the non-Copy payload field v{v}.{i} of {ty:?} needs WP-C5.3d's \
                         controlled storage: `match &e` yields a reference, and moving out of it \
                         runs into the same block-dispatch limit as C5.3a's cross-block moves"
                    )));
                }
                let enum_name = super::mangle::type_name_for_nominal(item.0, args);
                let arity = env
                    .types
                    .enum_variants
                    .get(&(item.0, args.clone()))
                    .and_then(|variants| variants.get(*v as usize))
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
        let env = TyEnv::new(&body, &types);
        assert_eq!(emit_place(&Place::local(LocalId(3)), &env).unwrap(), "_3");
    }

    /// One MIR `Field` projection, two Rust syntaxes, chosen by the base type -- the property
    /// that makes `TyEnv` necessary rather than merely convenient.
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
        let env = TyEnv::new(&body, &types);

        let tuple_field = Place {
            local: LocalId(0),
            projection: vec![Projection::Field(1)],
        };
        assert_eq!(emit_place(&tuple_field, &env).unwrap(), "_0.1");

        let struct_field = Place {
            local: LocalId(1),
            projection: vec![Projection::Field(1)],
        };
        assert_eq!(emit_place(&struct_field, &env).unwrap(), "_1.f1");
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
        let env = TyEnv::new(&body, &types);
        let place = Place {
            local: LocalId(0),
            projection: vec![Projection::Field(1), Projection::Field(0)],
        };
        assert_eq!(emit_place(&place, &env).unwrap(), "_0.f1.0");
        assert_eq!(env.place_ty(&place).unwrap(), MirTy::Bool);
    }

    #[test]
    fn array_indices_emit_both_forms() {
        let body = body_with(vec![MirTy::Array(Box::new(MirTy::Int32), 3), MirTy::Int64]);
        let types = TypeContext::default();
        let env = TyEnv::new(&body, &types);

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
        let env = TyEnv::new(&body, &types);
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
