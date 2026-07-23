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
    AggKind, MirProgram, MirTy, Operand, Place, Projection, Rvalue, Statement, Terminator,
    TypeContext,
};
use std::collections::BTreeMap;

/// WP-C6.1c: whether `operands` are exactly one enum slot's payload fields `VariantField(v, 0..n)`
/// in order — the canonical decomposition aggregate that `mir::lower::
/// materialize_consumed_variant_payload` emits. Such an aggregate emits as ONE destructuring
/// `take()` match, so it needs no per-field projection helper (`collect` skips its operands) and
/// `emit_bodies` special-cases it. Returns `Some((base local, variant))` when it matches. The
/// structural test is the single source of truth shared by `collect` and `emit_bodies`, so the two
/// cannot disagree about which aggregate is a decomposition.
pub(super) fn variant_payload_decomposition(
    operands: &[Operand],
    env: &emit_places::TyEnv,
) -> Result<Option<(u32, u32)>, BackendDiagnostic> {
    if operands.len() < 2 {
        return Ok(None);
    }
    let mut base: Option<(u32, u32)> = None; // (base local, variant)
    for (i, op) in operands.iter().enumerate() {
        let place = match op {
            Operand::Move(p) | Operand::Copy(p) => p,
            Operand::Const(_) => return Ok(None),
        };
        let [Projection::VariantField(v, k)] = place.projection.as_slice() else {
            return Ok(None);
        };
        if *k as usize != i {
            return Ok(None); // fields must be exactly 0..n in order
        }
        match base {
            None => base = Some((place.local.0, *v)),
            Some((l, bv)) => {
                if place.local.0 != l || bv != *v {
                    return Ok(None); // all fields from ONE enum slot and variant
                }
            }
        }
    }
    let (local, variant) = base.expect("operands is non-empty");
    if !emit_places::is_slot_local(local, env)? {
        return Ok(None);
    }
    Ok(Some((local, variant)))
}

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
    /// Destroys ONE drop unit in place: `ValueSlot::drop_field_with` (WP-C5.3d-1c). The unit's
    /// destruction plan is baked into the wrapper, because a wrapper is already per-(base type,
    /// projection) and that fixes the field type.
    Drop,
}

/// One helper to generate. WP-C6.1b: `projection` is the FULL chain (depth ≥1). A pure-`Raw` chain
/// of any depth is valid over partially-moved storage; a `Whole` (enum-payload) form is depth-1
/// only.
#[derive(Clone)]
pub struct ProjectionHelper {
    pub name: String,
    pub op: HelperOp,
    pub base_ty: MirTy,
    pub field_ty: MirTy,
    pub projection: Vec<Projection>,
    pub form: ProjectionForm,
}

/// The canonical selector token for one projection level, used to key a helper name.
fn projection_token(projection: &Projection) -> String {
    match projection {
        Projection::Field(i) => format!("f{i}"),
        Projection::ConstIndex(n) => format!("i{n}"),
        Projection::VariantField(v, i) => format!("v{v}f{i}"),
        other => format!("{other:?}"),
    }
}

/// The generated helper name for one (base type, projection CHAIN) pair. Deterministic and
/// injective: it reuses [`mangle::sanitize_symbol`]'s encoding over a canonical key joining every
/// level of the chain, so two distinct projection chains cannot collide any more than two distinct
/// instances can.
pub fn helper_name(base_ty: &MirTy, projection: &[Projection], op: HelperOp) -> String {
    let verb = match op {
        HelperOp::Move => "move",
        HelperOp::Copy => "copy",
        HelperOp::Drop => "drop",
    };
    let selector: Vec<String> = projection.iter().map(projection_token).collect();
    mangle::sanitize_symbol(&format!(
        "{verb}#{}#{}",
        crate::mir::dump_ty(base_ty),
        selector.join(".")
    ))
}

/// WP-C6.1b: whether every level of `chain` has raw-pointer (`addr_of_mut!`) syntax over `base_ty`
/// — a `Field` on a struct/tuple, or a `ConstIndex` on an array, at every level. A raw chain is
/// valid over partially-moved storage at ANY depth, because `addr_of_mut!` computes a nested field
/// address without dereferencing intermediates. A `VariantField` (enum) or a dynamic `Index` at any
/// level is NOT raw.
pub(super) fn chain_is_raw(
    base_ty: &MirTy,
    chain: &[Projection],
    types: &TypeContext,
) -> Result<bool, BackendDiagnostic> {
    if chain.is_empty() {
        return Ok(false);
    }
    let mut ty = base_ty.clone();
    for projection in chain {
        let raw = matches!(
            (projection, &ty),
            (Projection::Field(_), MirTy::Struct(..) | MirTy::Tuple(_))
                | (Projection::ConstIndex(_), MirTy::Array(..))
        );
        if !raw {
            return Ok(false);
        }
        ty = emit_places::project_ty_once(&ty, projection, types)?;
    }
    Ok(true)
}

/// The chained Rust selector for a raw chain, e.g. `.f0.f0` for two struct fields or `.f1.0[2]` for
/// struct→tuple→array. Walks `types` to choose `.f{i}` (struct) vs `.{i}` (tuple) vs `[{n}]` (array)
/// at each level. Precondition: [`chain_is_raw`] holds.
fn raw_selector_chain(
    base_ty: &MirTy,
    chain: &[Projection],
    types: &TypeContext,
) -> Result<String, BackendDiagnostic> {
    let mut out = String::new();
    let mut ty = base_ty.clone();
    for projection in chain {
        out.push_str(&raw_selector(projection, &ty)?);
        ty = emit_places::project_ty_once(&ty, projection, types)?;
    }
    Ok(out)
}

/// Walk every body and collect the projections that need helpers: sub-place moves, and sub-place
/// drops. Deduplicated and ordered by helper name.
pub fn collect(
    program: &MirProgram,
    layout: &crate::layout::TargetLayout,
) -> Result<Vec<ProjectionHelper>, BackendDiagnostic> {
    let mut found: BTreeMap<String, ProjectionHelper> = BTreeMap::new();
    for body in &program.bodies {
        let env = emit_places::TyEnv::new(body, &program.types, layout);
        for block in &body.blocks {
            for (statement, _) in &block.statements {
                if let Statement::Assign(_, rvalue) = statement {
                    // WP-C6.1c: a variant-payload decomposition aggregate emits as one
                    // `take()`+destructure match, needing no per-field projection helper — its
                    // `VariantField` operands must NOT be collected (they cannot be raw-projected,
                    // and collecting them would re-trip the multi-unit refusal here).
                    if let Rvalue::Aggregate(AggKind::Tuple, operands) = rvalue {
                        if variant_payload_decomposition(operands, &env)?.is_some() {
                            continue;
                        }
                    }
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
                // WP-C5.3d-1c: MIR's drop elaboration decomposes an aggregate with more than
                // one drop unit into per-unit, flag-guarded `Drop`s on PROJECTED places, so this
                // is the ordinary shape for any struct with two droppable fields -- not an edge
                // case. Each one needs a wrapper around `drop_field_with`.
                Terminator::Drop { place, .. } if !place.projection.is_empty() => {
                    collect_place(place, &env, HelperOp::Drop, &mut found)?
                }
                _ => {}
            }
        }
    }
    Ok(found.into_values().collect())
}

/// Every operand an rvalue reads. Shared with `emit_bodies`' reference validator so the two
/// cannot disagree about what counts as a use.
pub fn rvalue_operands(rvalue: &Rvalue) -> Vec<&Operand> {
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
    // WP-C6.1b: a `Copy` field read out of a slot uses a raw helper (so it survives a sibling
    // move) for a raw chain of ANY depth, not just depth 1.
    let base_ty = env.local_ty(place.local.0)?;
    if field_is_copy && chain_is_raw(&base_ty, &place.projection, env.types)? {
        return collect_place(place, env, HelperOp::Copy, found); // case 2
    }
    Ok(())
}

fn collect_place(
    place: &Place,
    env: &emit_places::TyEnv,
    op: HelperOp,
    found: &mut BTreeMap<String, ProjectionHelper>,
) -> Result<(), BackendDiagnostic> {
    let base_ty = env.local_ty(place.local.0)?;
    let projection = place.projection.clone();
    let field_ty = env.place_ty(place)?;

    // WP-C6.1b: a raw chain (all `Field`/`ConstIndex` over struct/tuple/array) of ANY depth is
    // valid over partially-moved storage, so a multi-level partial move/drop like `o.a.x` is a
    // single raw helper whose `addr_of_mut!` selector chains through every level. This closes G3:
    // C5.3d-0 only implemented depth 1.
    if chain_is_raw(&base_ty, &projection, env.types)? {
        let name = helper_name(&base_ty, &projection, op);
        found.entry(name.clone()).or_insert(ProjectionHelper {
            name,
            op,
            base_ty,
            field_ty,
            projection,
            form: ProjectionForm::Raw,
        });
        return Ok(());
    }

    // Enum payload: a `Whole` (`&mut T` + `match`) form, valid ONLY at depth 1 and ONLY while the
    // slot is whole. A multi-unit payload (arity > 1) is refused — the CD-070 boundary, C6.1c's
    // work — and a `VariantField` nested inside a longer chain is likewise out of C6.1b's raw
    // subset.
    let form = match projection.as_slice() {
        [Projection::VariantField(variant, _)] if matches!(base_ty, MirTy::Enum(..)) => {
            let MirTy::Enum(enum_ref, args) = &base_ty else {
                unreachable!("guarded by the match arm")
            };
            // CD-070: an enum payload has no raw projection, so moving one unit out goes through
            // `move_field_whole`, which REQUIRES a complete value and leaves the slot `Partial`.
            // With more than one payload unit, the second move — or the whole-enum drop of a
            // surviving sibling — then hits a `Whole` requirement over partial storage.
            //
            // Found by an adversarial fixture: `enum E { V(A, B) }` with `match e { E::V(a, b) =>
            // take_a(a) }` COMPILED and aborted at run time inside `slot_violation`, whose own
            // message says "STARK compiler defect, not a program fault". Refusing here restores
            // the promise that an out-of-subset shape is rejected deterministically before rustc.
            //
            // Single-unit payloads are unaffected, which is what `Option`/`Result` and the
            // approved consuming-match shapes use.
            let arity = emit_types::variant_payloads(enum_ref, args, env.types)
                .and_then(|variants: Vec<Vec<MirTy>>| {
                    variants.get(*variant as usize).map(|p| p.len())
                })
                .ok_or_else(|| {
                    BackendDiagnostic::Unsupported(format!(
                        "variant v{variant} unresolvable for {base_ty:?}"
                    ))
                })?;
            if arity > 1 {
                return Err(BackendDiagnostic::Unsupported(format!(
                    "moving one unit out of a MULTI-UNIT enum payload ({place:?}, variant \
                     v{variant} has {arity} fields) is outside the C5 subset: an enum \
                     payload has no raw projection, so the move goes through \
                     `move_field_whole`, which requires a complete value and leaves the \
                     slot partial -- the sibling unit could then be neither moved nor \
                     destroyed. C5 supports whole enum payload movement and single-unit \
                     consuming-match shapes; partial movement of one field from a \
                     multi-unit payload is deferred to broad ownership/reference \
                     completion (C6)"
                )));
            }
            ProjectionForm::Whole
        }
        _ => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "sub-place move/drop through {projection:?} on {base_ty:?} is not in the C6.1b \
                 subset: only a pure raw chain (struct/tuple `Field`, array `ConstIndex`) at any \
                 depth, or a single-level single-unit enum payload, is supported. A `VariantField` \
                 nested in a longer chain is enum-payload work (C6.1c)"
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
                // WP-C6.1b: chained selector over every level, e.g. `.f0.f0`, so `addr_of_mut!`
                // computes the deep field's address without dereferencing intermediates.
                let selector = raw_selector_chain(&helper.base_ty, &helper.projection, types)?;
                format!("|p: *mut {base}| unsafe {{ core::ptr::addr_of_mut!((*p){selector}) }}")
            }
            ProjectionForm::Whole => {
                let [Projection::VariantField(variant, index)] = helper.projection.as_slice()
                else {
                    return Err(BackendDiagnostic::Unsupported(
                        "whole-form projection is a single enum-payload level only".to_string(),
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
            // WP-C5.3d-1c. The destruction plan is baked in here rather than passed from the call
            // site: a wrapper is already per-(base type, projection), which fixes the field type,
            // which fixes the plan. It also keeps the call site free of glue, so an emitted MIR
            // body still contains no `unsafe` and no destruction logic of its own.
            (HelperOp::Drop, ProjectionForm::Raw) => {
                let plan = crate::mir::drop_plan::plan_for(&helper.field_ty, types)
                    .map_err(|e| BackendDiagnostic::Unsupported(e.to_string()))?;
                let glue = super::emit_bodies::emit_drop_plan(&plan, "__v")?;
                out.push_str(&format!(
                    "\n    /// Destroys ONE drop unit in place, leaving its siblings alone. Raw \
                     projection, so\n    /// it is valid over storage a sibling has already been \
                     moved out of.\n    pub fn {}(slot: &mut stark_runtime::slot::ValueSlot<{base}>) {{\n         \
                     \x20       // SAFETY: one fixed projection into THIS slot's storage; MIR's drop \
                     flags\n         \x20       // guarantee the unit is live and destroyed at most once.\n         \
                     \x20       unsafe {{\n             \x20           slot.drop_field_with({projection}, |__p: *mut {field}| {{\n                 \
                     \x20               // SAFETY: the unit is live, so its bytes are a valid `{field}`.\n                 \
                     \x20               let __v: &mut {field} = unsafe {{ &mut *__p }};\n                 \
                     \x20               {glue}\n             \x20           }})\n         \x20       }}\n    }}\n",
                    helper.name
                ))
            }
            // A variant payload has no raw projection, so its drop unit cannot be destroyed
            // through this path. MIR does not ask it to: an enum's payload is destroyed by the
            // WHOLE-enum plan, whose `Variants` arm matches the live variant. If a projected
            // `Drop` on a payload ever appears, it needs its own design rather than the `Whole`
            // form, which requires a complete value the drop is in the middle of dismantling.
            (HelperOp::Drop, ProjectionForm::Whole) => {
                return Err(BackendDiagnostic::Unsupported(format!(
                    "a projected Drop of the enum payload {:?} is not in the C5.3d-1c subset: an \
                     enum's payload is destroyed by the whole-enum plan's variant match, and the \
                     `&mut T` projection form needs a complete value",
                    helper.base_ty
                )))
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
