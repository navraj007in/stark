//! WP-C5.3d-0 — generated per-type field projection helpers.
//!
//! A partially moved `ValueSlot` may only be reached through raw pointers, so a sub-place move
//! needs a function of type `fn(*mut T) -> *mut F`. That function's body must dereference a raw
//! pointer, which is `unsafe` — and deliverable 2 forbids `unsafe` inside emitted MIR bodies.
//! The resolution is this module: one small generated helper per (type, sub-place) pair actually
//! used, collected up front, with the `unsafe` confined here and the call sites safe.
//!
//! Two projection forms exist because two are genuinely needed:
//!
//! - **Raw** (`fn(*mut T) -> *mut F`) via `addr_of_mut!`, which computes a field address without
//!   dereferencing and is therefore valid over partially moved storage. Used for struct fields,
//!   tuple elements, and constant array indices.
//! - **Whole** (`fn(&mut T) -> &mut F`) for an **enum variant payload**, which Rust offers no way
//!   to address without a `match` — and a `match` needs a reference. Legal only while the slot is
//!   `Whole`, which `ValueSlot::move_field_whole` enforces.
//!
//! Helpers are collected by walking every body before emission, so the generated module contains
//! exactly the projections the program uses, in a deterministic (`BTreeMap`) order.

use super::{emit_places, emit_types, mangle, BackendDiagnostic};
use crate::mir::{
    MirProgram, MirTy, Operand, Place, Projection, Rvalue, Statement, Terminator, TypeContext,
};
use std::collections::BTreeMap;

/// Which projection syntax a sub-place needs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProjectionForm {
    /// `fn(*mut T) -> *mut F` — valid on partially moved storage.
    Raw,
    /// `fn(&mut T) -> &mut F` — an enum payload; requires a `Whole` slot.
    Whole,
}

/// Which slot primitive a helper wraps. One wrapper per (type, sub-place, OPERATION), so each
/// generated function calls exactly one `unsafe` primitive with exactly one fixed projection —
/// which is what discharges those primitives' safety obligation by construction.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum HelperOp {
    /// Moves the drop unit out: `ValueSlot::move_field` / `move_field_whole`.
    Move,
    /// Reads a `Copy` field without disturbing liveness: `ValueSlot::copy_field`.
    Copy,
}

/// One helper to generate.
#[derive(Clone)]
pub struct ProjectionHelper {
    pub name: String,
    pub op: HelperOp,
    pub base_ty: MirTy,
    pub field_ty: MirTy,
    pub projection: Projection,
    pub form: ProjectionForm,
}

/// The generated helper name for one (base type, projection) pair. Deterministic and injective:
/// it reuses [`mangle::sanitize_symbol`]'s encoding over a canonical key, so two distinct
/// projections cannot collide any more than two distinct instances can.
pub fn helper_name(base_ty: &MirTy, projection: &Projection, op: HelperOp) -> String {
    let verb = match op {
        HelperOp::Move => "move",
        HelperOp::Copy => "copy",
    };
    let selector = match projection {
        Projection::Field(i) => format!("f{i}"),
        Projection::ConstIndex(n) => format!("i{n}"),
        Projection::VariantField(v, i) => format!("v{v}f{i}"),
        other => format!("{other:?}"),
    };
    mangle::sanitize_symbol(&format!(
        "{verb}#{}#{selector}",
        crate::mir::dump_ty(base_ty)
    ))
}

/// Walk every body and collect the projections that need helpers: sub-place moves, and sub-place
/// drops. Deduplicated and ordered by helper name.
pub fn collect(program: &MirProgram) -> Result<Vec<ProjectionHelper>, BackendDiagnostic> {
    let mut found: BTreeMap<String, ProjectionHelper> = BTreeMap::new();
    for body in &program.bodies {
        let env = emit_places::TyEnv::new(body, &program.types);
        for block in &body.blocks {
            for (statement, _) in &block.statements {
                if let Statement::Assign(_, rvalue) = statement {
                    for operand in rvalue_operands(rvalue) {
                        collect_operand(operand, &env, &program.types, &mut found)?;
                    }
                }
            }
            match &block.terminator.0 {
                Terminator::Call { args, .. } | Terminator::Checked { args, .. } => {
                    for arg in args {
                        collect_operand(arg, &env, &program.types, &mut found)?;
                    }
                }
                Terminator::SwitchInt { scrut, .. } => {
                    collect_operand(scrut, &env, &program.types, &mut found)?
                }
                Terminator::Drop { place, .. } if !place.projection.is_empty() => {
                    collect_place(place, &env, HelperOp::Move, &mut found)?
                }
                _ => {}
            }
        }
    }
    Ok(found.into_values().collect())
}

fn rvalue_operands(rvalue: &Rvalue) -> Vec<&Operand> {
    match rvalue {
        Rvalue::Use(operand) | Rvalue::UnOp(_, operand) => vec![operand],
        Rvalue::BinOp(_, lhs, rhs) => vec![lhs, rhs],
        Rvalue::Aggregate(_, operands) => operands.iter().collect(),
        Rvalue::Discriminant(_) | Rvalue::RefOf { .. } | Rvalue::LayoutQuery { .. } => vec![],
    }
}

/// A helper is needed in exactly two situations, and they are different enough to state
/// separately rather than fold into one condition:
///
/// 1. **a non-`Copy` sub-place is MOVED out** — the move itself must go through `move_field` or
///    `move_field_whole`, since it changes which drop units remain live;
/// 2. **a `Copy` field is READ out of a raw-projectable base** — `emit_places` routes these
///    through `copy_field` so they keep working after a SIBLING unit has been moved, at which
///    point `get()` is correctly refused.
///
/// Everything else (`Copy` reads through the `get()` path, enum payload reads, non-slot locals)
/// needs no helper.
fn collect_operand(
    operand: &Operand,
    env: &emit_places::TyEnv,
    types: &TypeContext,
    found: &mut BTreeMap<String, ProjectionHelper>,
) -> Result<(), BackendDiagnostic> {
    let (Operand::Move(place) | Operand::Copy(place)) = operand else {
        return Ok(());
    };
    if place.projection.is_empty() || !emit_places::is_slot_local(place.local.0, env)? {
        return Ok(());
    }
    let field_ty = env.place_ty(place)?;
    let field_is_copy = emit_types::mir_ty_is_copy(&field_ty, types);

    if !field_is_copy && matches!(operand, Operand::Move(_)) {
        return collect_place(place, env, HelperOp::Move, found); // case 1
    }
    if field_is_copy && place.projection.len() == 1 && raw_projectable(place, env)? {
        return collect_place(place, env, HelperOp::Copy, found); // case 2
    }
    Ok(())
}

/// Whether a single-level projection has raw-pointer syntax (`addr_of_mut!`). Enum payloads do
/// not, which is why they take the `&mut T` path instead.
fn raw_projectable(place: &Place, env: &emit_places::TyEnv) -> Result<bool, BackendDiagnostic> {
    let base_ty = env.local_ty(place.local.0)?;
    Ok(matches!(
        (&place.projection[0], &base_ty),
        (Projection::Field(_), MirTy::Struct(..) | MirTy::Tuple(_))
            | (Projection::ConstIndex(_), MirTy::Array(..))
    ))
}

fn collect_place(
    place: &Place,
    env: &emit_places::TyEnv,
    op: HelperOp,
    found: &mut BTreeMap<String, ProjectionHelper>,
) -> Result<(), BackendDiagnostic> {
    if place.projection.len() != 1 {
        return Err(BackendDiagnostic::Unsupported(format!(
            "a sub-place move/drop through a projection chain of length {} ({place:?}) is not in \
             the WP-C5.3d-0 subset: only ONE level is implemented, because each additional level \
             needs its own helper over an intermediate type whose validity after a partial move \
             has to be established separately. Refused before rustc rather than approximated",
            place.projection.len()
        )));
    }
    let base_ty = env.local_ty(place.local.0)?;
    let projection = place.projection[0].clone();
    let field_ty = env.place_ty(place)?;
    let form = match (&projection, &base_ty) {
        (Projection::Field(_), MirTy::Struct(..) | MirTy::Tuple(_))
        | (Projection::ConstIndex(_), MirTy::Array(..)) => ProjectionForm::Raw,
        (Projection::VariantField(..), MirTy::Enum(..)) => ProjectionForm::Whole,
        _ => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "sub-place move/drop through {projection:?} on {base_ty:?} is not in the \
                 WP-C5.3d-0 subset"
            )))
        }
    };
    let name = helper_name(&base_ty, &projection, op);
    found.entry(name.clone()).or_insert(ProjectionHelper {
        name,
        op,
        base_ty,
        field_ty,
        projection,
        form,
    });
    Ok(())
}

/// The helper name and form for ONE place, used at emission time. Shares
/// [`collect_place`]'s rules exactly, so a place the collector refused cannot be emitted, and a
/// place it accepted always has a helper.
pub fn collect_for_place(
    place: &Place,
    env: &emit_places::TyEnv,
    op: HelperOp,
) -> Result<String, BackendDiagnostic> {
    let mut found = BTreeMap::new();
    collect_place(place, env, op, &mut found)?;
    let helper = found
        .into_values()
        .next()
        .expect("collect_place inserts exactly one helper on success");
    Ok(helper.name)
}

/// Emit the generated helper module. Empty when the program performs no sub-place moves.
pub fn emit(
    helpers: &[ProjectionHelper],
    types: &TypeContext,
) -> Result<String, BackendDiagnostic> {
    if helpers.is_empty() {
        return Ok(String::new());
    }
    // The explanatory banner lives INSIDE the module: a comment above it would put the word
    // "unsafe" outside the module's braces, where the generated program is supposed to contain
    // none -- and the tests that check that boundary would have to special-case a comment.
    let mut out = String::from(
        "mod stark_proj {\n\
         \x20   //! Generated field projections (WP-C5.3d-0). The ONLY place `unsafe` appears in\n\
         \x20   //! a generated program: emitted MIR bodies contain none of their own.\n\
         \x20   #![allow(unused)]\n\
         \x20   use super::*;\n",
    );
    for helper in helpers {
        let base = emit_types::emit_ty(&helper.base_ty)?;
        let field = emit_types::emit_ty(&helper.field_ty)?;
        // The projection expression, inlined into the wrapper rather than passed around: each
        // wrapper therefore pairs ONE primitive with ONE fixed projection over ONE slot type,
        // which is exactly the pairing the primitives' safety contract requires.
        let projection = match helper.form {
            ProjectionForm::Raw => {
                let selector = raw_selector(&helper.projection, &helper.base_ty)?;
                format!("|p: *mut {base}| unsafe {{ core::ptr::addr_of_mut!((*p){selector}) }}")
            }
            ProjectionForm::Whole => {
                let Projection::VariantField(variant, index) = &helper.projection else {
                    return Err(BackendDiagnostic::Unsupported(
                        "whole-form projection outside an enum payload".to_string(),
                    ));
                };
                let MirTy::Enum(enum_ref, args) = &helper.base_ty else {
                    return Err(BackendDiagnostic::Unsupported(
                        "whole-form projection on a non-enum base".to_string(),
                    ));
                };
                let arity = emit_types::variant_payloads(enum_ref, args, types)
                    .and_then(|variants| variants.get(*variant as usize).cloned())
                    .map(|payload| payload.len())
                    .ok_or_else(|| {
                        BackendDiagnostic::Unsupported(format!(
                            "variant v{variant} unresolvable for {:?}",
                            helper.base_ty
                        ))
                    })?;
                let mut binders: Vec<String> = (0..arity).map(|_| "_".to_string()).collect();
                binders[*index as usize] = "__payload".to_string();
                format!(
                    "|v: &mut {base}| match v {{ {base}::{}({}) => __payload, \
                     _ => unreachable!(\"V-DISC-1: payload projection without a discriminant \
                     test\") }}",
                    emit_types::variant_name(*variant),
                    binders.join(", ")
                )
            }
        };
        match (helper.op, helper.form) {
            (HelperOp::Move, ProjectionForm::Raw) => out.push_str(&format!(
                "\n    /// Moves one drop unit out. Raw projection: `addr_of_mut!` computes a \
                 field\n    /// address WITHOUT dereferencing, so it stays valid over partially \
                 moved storage.\n                 \x20   pub fn {}(slot: &mut stark_runtime::slot::ValueSlot<{base}>) -> {field} \
                 {{\n                 \x20       // SAFETY: one fixed projection into THIS slot's storage; MIR's drop \
                 flags\n                 \x20       // guarantee the unit is live and moved at most once.\n                 \x20       unsafe {{ slot.move_field({projection}) }}\n                 \x20   }}\n",
                helper.name
            )),
            (HelperOp::Move, ProjectionForm::Whole) => out.push_str(&format!(
                "\n    /// Moves an enum payload out. Rust cannot address a variant's field \
                 without a\n    /// `match`, and a `match` needs a reference -- so this form is \
                 valid only while the\n    /// slot is WHOLE, which `move_field_whole` enforces. \
                 The `_` arm is dead under V-DISC-1.\n                 \x20   pub fn {}(slot: &mut stark_runtime::slot::ValueSlot<{base}>) -> {field} \
                 {{\n                 \x20       // SAFETY: one fixed projection into THIS slot's storage; the active \
                 variant\n                 \x20       // was established by a discriminant test (V-DISC-1).\n                 \x20       unsafe {{ slot.move_field_whole({projection}) }}\n                 \x20   }}\n",
                helper.name
            )),
            (HelperOp::Copy, ProjectionForm::Raw) => out.push_str(&format!(
                "\n    /// Reads a `Copy` field without disturbing liveness. Raw projection, so \
                 it keeps\n    /// working after a SIBLING unit has been moved out.\n                 \x20   pub fn {}(slot: &stark_runtime::slot::ValueSlot<{base}>) -> {field} {{\n                 \x20       // SAFETY: one fixed projection into THIS slot's storage; a `Copy` \
                 field\n                 \x20       // carries no drop obligation, so reading it cannot double-free.\n                 \x20       unsafe {{ slot.copy_field({projection}) }}\n                 \x20   }}\n",
                helper.name
            )),
            (HelperOp::Copy, ProjectionForm::Whole) => {
                return Err(BackendDiagnostic::Unsupported(
                    "a Copy read of an enum payload uses the `get()` path, not a helper"
                        .to_string(),
                ))
            }
        }
    }
    out.push_str("}\n\n");
    Ok(out)
}

fn raw_selector(projection: &Projection, base_ty: &MirTy) -> Result<String, BackendDiagnostic> {
    Ok(match (projection, base_ty) {
        (Projection::Field(i), MirTy::Tuple(_)) => format!(".{i}"),
        (Projection::Field(i), MirTy::Struct(..)) => format!(".{}", emit_types::field_name(*i)),
        (Projection::ConstIndex(n), MirTy::Array(..)) => format!("[{n}]"),
        _ => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "no raw selector for {projection:?} on {base_ty:?}"
            )))
        }
    })
}
