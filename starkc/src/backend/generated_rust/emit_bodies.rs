//! WP-C5.2d — concrete instances, parameters, return destinations, and direct call
//! continuations, on top of WP-C5.2c's arbitrary control flow.
//!
//! Arbitrary MIR control flow (any CFG shape, including cycles for loops) is emitted as a
//! block-index dispatch loop: `let mut __bb: u32 = <entry>; loop { match __bb { 0 => { ... },
//! 1 => { ... }, ... } }`. Rust has no `goto`, so this is the standard technique for emitting a
//! basic-block graph without first recovering structured `if`/`while` shapes -- the same
//! approach rustc's own backends use at the LLVM-IR level (basic blocks + branches, no
//! structure recovery). It composes uniformly for `Goto`/`SwitchInt` without needing to detect
//! "this is a loop" versus "this is a branch": both are just `__bb = target; continue;`.
//!
//! `Rvalue::Aggregate`/`Discriminant` (WP-C5.3), `Rvalue::RefOf` (not yet scheduled to a
//! specific sub-WP), indirect (`Callee::FnValue`) / runtime (`Callee::Runtime`) calls
//! (WP-C5.4c / wherever `RuntimeFn` support lands), the `Drop` terminator (WP-C5.3), and
//! `Terminator::Trap` carrying a user MESSAGE (needs `&str` values -- WP-C5.3) remain
//! `Unsupported`. Message-less `Terminator::Trap` is supported as of WP-C5.2e.
//!
//! WP-C5.2e: every checked-operation trap now carries a real source location. The location is
//! resolved at COMPILE TIME (`SourceFile::line_col` against `MirProgram::files`, both already
//! available to the backend) and baked into the generated call site as literals -- see
//! `stark-runtime/src/trap.rs`'s doc comment for why this is a deliberate, documented
//! alternative to §13.1's compact-span-ID-plus-runtime-lookup design, not an oversight.

use super::emit_places::TyEnv;
use super::{emit_places, emit_projections, emit_types, mangle, BackendDiagnostic};
use crate::mir::drop_plan::{self, DropPlan};
use crate::mir::{
    AggKind, Callee, CheckedOp, Constant, LocalKind, MirBinOp, MirBody, MirTy, MirUnOp, Operand,
    Rvalue, SourceInfo, Statement, Terminator, TrapCategory, TrapInfo, TypeContext,
};
use crate::source::SourceFile;
use std::sync::Arc;

/// A complete `fn name(params) -> ret { ... }` for an ordinary (non-entry) function. The entry
/// instance is emitted separately, by `emit_program.rs`, as Rust's literal `fn main()` with the
/// version-check prologue prepended -- `emit_block_body` is the shared piece both use.
pub fn emit_function(
    body: &MirBody,
    name: &str,
    files: &[Arc<SourceFile>],
    types: &TypeContext,
    layout: &crate::layout::TargetLayout,
) -> Result<String, BackendDiagnostic> {
    let params = emit_param_list(body, types)?;
    let ret_ty = emit_types::emit_ty(&body.ret)?;
    let block = emit_block_body(body, files, types, layout)?;
    Ok(format!("fn {name}({params}) -> {ret_ty} {block}"))
}

/// A non-`Copy` parameter arrives as an ordinary Rust value (the caller moved it in), but the
/// body needs it in a `ValueSlot` like every other non-`Copy` local. It is therefore bound under
/// a distinct incoming name `__pN` and written into the slot in the prologue -- binding it as
/// `_N` directly would collide with the slot the body declares.
fn emit_param_list(body: &MirBody, types: &TypeContext) -> Result<String, BackendDiagnostic> {
    let mut param_locals: Vec<Option<u32>> = vec![None; body.params.len()];
    for (i, decl) in body.locals.iter().enumerate() {
        if let LocalKind::Param(j) = decl.kind {
            let slot = param_locals.get_mut(j as usize).ok_or_else(|| {
                BackendDiagnostic::Unsupported(format!(
                    "Param index {j} out of range for a {}-parameter body",
                    body.params.len()
                ))
            })?;
            *slot = Some(i as u32);
        }
    }
    let mut parts = Vec::with_capacity(body.params.len());
    for (j, ty) in body.params.iter().enumerate() {
        let local = param_locals[j].ok_or_else(|| {
            BackendDiagnostic::Unsupported(format!("no local declares LocalKind::Param({j})"))
        })?;
        let rust_ty = emit_types::emit_ty(ty)?;
        // `mut`: a parameter binding is immutable by default in Rust; MIR does not distinguish
        // a reassignable parameter from any other local, so this matches every other local's
        // uniform `let mut` treatment rather than trying to prove a parameter is never
        // reassigned.
        let decl_ty = &body.locals[local as usize].ty;
        let name = if emit_types::is_slot_backed(decl_ty, types) {
            incoming_param_name(local)
        } else {
            emit_places::local_name(local)
        };
        parts.push(format!("mut {name}: {rust_ty}"));
    }
    Ok(parts.join(", "))
}

/// Emits a function body for the WP-C5.2d-supported shape: every local `Return`/`Param`/`User`/
/// `Temp`-kinded, every statement a `Nop` or an assignment whose `Rvalue` is `Use`/`UnOp`/
/// `BinOp`/`LayoutQuery`, and every terminator `Goto`/`SwitchInt`/`Checked`/`Call` (direct only)/
/// `Return`/`Unreachable`.
/// The Rust parameter name a non-`Copy` argument arrives under, before it is moved into its
/// slot. Distinct from `local_name` so the two cannot collide.
fn incoming_param_name(local: u32) -> String {
    format!("__p{local}")
}

pub fn emit_block_body(
    body: &MirBody,
    files: &[Arc<SourceFile>],
    types: &TypeContext,
    layout: &crate::layout::TargetLayout,
) -> Result<String, BackendDiagnostic> {
    let env = &TyEnv::new(body, types, layout);
    validate_ephemeral_references(body, env)?;
    let mut out = String::from("{\n");

    // Every non-parameter local is declared `let mut`, DEFAULT-INITIALISED, up front (WP-C5.2c's
    // record explains why default-initialisation, not left uninitialised, is required once a
    // body has more than one block). `Param`-kinded locals are NOT re-declared here: they are
    // already bound by the Rust function signature (`emit_param_list`), under the exact same
    // `_N` name (`emit_places::local_name`); declaring them again here would shadow the actual
    // argument with a fabricated default, so every parameter would read as its default value
    // rather than the value passed in.
    for (i, decl) in body.locals.iter().enumerate() {
        // One rule for both the parameter and the ordinary-local branch: a value is slot-backed
        // when it is non-Copy AND not a reference. References are ephemeral by the C5.3d-1a lane
        // and must never be wrapped -- a slot-backed `&mut Self` receiver would make the
        // destructor body's `Deref` project through a `ValueSlot` instead of the reference.
        let slotted = emit_types::is_slot_backed(&decl.ty, env.types);
        match &decl.kind {
            // `IndexProof` (WP-C5.3a) is an ordinary integer local in generated Rust: MIR
            // keeps it opaque so only `Projection::Index` can consume it, but that opacity is a
            // MIR-level property the verifier enforces, not something the backend re-creates.
            // `DropFlag` (WP-C5.3d) is an ordinary `Bool` local: MIR's drop elaboration produces
            // it, and the backend simply carries it -- per-drop-unit liveness lives there, not in
            // `ValueSlot`.
            LocalKind::Return
            | LocalKind::User(_)
            | LocalKind::Temp
            | LocalKind::IndexProof
            | LocalKind::DropFlag => {}
            // A Copy parameter is already bound by the signature under its `_N` name. A non-Copy
            // one is bound as `__pN` and moved into its slot below.
            LocalKind::Param(_) if !slotted => continue,
            LocalKind::Param(_) => {
                let slot_ty = emit_types::emit_slot_ty(&decl.ty)?;
                let name = emit_places::local_name(i as u32);
                out.push_str(&format!(
                    "    let mut {name}: {slot_ty} = stark_runtime::slot::ValueSlot::dead();\n"
                ));
                out.push_str(&format!(
                    "    {name}.write({});\n",
                    incoming_param_name(i as u32)
                ));
                continue;
            } // No catch-all: every `LocalKind` is handled as of WP-C5.3d-1a, and keeping the
              // match exhaustive means a NEW kind stops compilation rather than silently becoming
              // an `Unsupported` diagnostic nobody reads.
        }
        // WP-C5.3d-0: a non-Copy local is backed by a `ValueSlot`, which starts DEAD -- so
        // generated code stops fabricating a default value it would immediately overwrite, and
        // MIR liveness is represented explicitly instead of being approximated by Rust's.
        // A REFERENCE local is declared UNINITIALISED. It has no valid default value, and the
        // C5.3d-1a lane guarantees it is assigned and consumed within one basic block, so Rust's
        // own definite-assignment analysis can see the assignment precedes the use. That makes
        // rustc a second check on the lane: a reference that escaped its block would fail to
        // compile as "possibly uninitialized" rather than silently reading a fabricated value.
        if matches!(decl.ty, MirTy::Ref { .. }) {
            out.push_str(&format!(
                "    let mut {}: {};\n",
                emit_places::local_name(i as u32),
                emit_types::emit_ty(&decl.ty)?
            ));
            continue;
        }
        if !slotted {
            let ty = emit_types::emit_ty(&decl.ty)?;
            let default = emit_types::default_value_expr(&decl.ty, env.types)?;
            out.push_str(&format!(
                "    let mut {}: {ty} = {default};\n",
                emit_places::local_name(i as u32)
            ));
        } else {
            let slot_ty = emit_types::emit_slot_ty(&decl.ty)?;
            out.push_str(&format!(
                "    let mut {}: {slot_ty} = stark_runtime::slot::ValueSlot::dead();\n",
                emit_places::local_name(i as u32)
            ));
        }
    }

    out.push_str(&format!("    let mut __bb: u32 = {};\n", body.entry.0));
    out.push_str("    let __stark_ret = loop {\n");
    out.push_str("        match __bb {\n");
    for (bi, block) in body.blocks.iter().enumerate() {
        out.push_str(&format!("            {bi} => {{\n"));
        for (stmt, _) in &block.statements {
            match stmt {
                Statement::Nop => {}
                Statement::Assign(place, rvalue) => {
                    let dest_ty = env.place_ty(place)?;
                    let value = emit_rvalue(rvalue, &dest_ty, env)?;
                    out.push_str(&format!(
                        "                {}\n",
                        emit_assignment(place, &value, env)?
                    ));
                }
            }
        }
        emit_terminator(body, files, &mut out, &block.terminator.0, env)?;
        out.push_str("            }\n");
    }
    out.push_str("            _ => unreachable!(\"invalid block index\"),\n");
    out.push_str("        }\n");
    out.push_str("    };\n");
    out.push_str("    __stark_ret\n");
    out.push_str("}\n");
    Ok(out)
}

/// WP-C5.3d-1a (CD-062): the **ephemeral borrowed-call reference lane** validator.
///
/// References are admitted into C5 only in a shape narrow enough that they need no storage and no
/// liveness tracking. Everything outside it is refused HERE, before rustc, so a broader reference
/// use fails as a named STARK backend limitation rather than as a borrow-check error in generated
/// code.
///
/// Enforced:
///
/// - a `RefOf` result must land in a compiler-generated temporary, never a user local (the lane
///   forbids writing a reference into a user-visible binding);
/// - every use of that temporary must be in the SAME basic block that created it (no reference
///   crosses a block boundary, which is what makes "ephemeral" true rather than aspirational);
/// - it must not be returned, stored into an aggregate, or written into a non-temporary place.
///
/// **One deviation from CD-062's wording, reported rather than absorbed.** The lane says the
/// reference must be "consumed by a statically resolved direct call". That is the shape a user
/// destructor takes, but it is NOT what `a.cmp(&b)` lowers to: for primitives, lowering inlines
/// the comparison, so the reference is consumed by a `Deref` READ inside a `BinOp` — and it is
/// first copied into a second temporary. Both remain ephemeral, same-block and unstored, so the
/// lane's *purpose* holds; its stated consumption form does not. This validator therefore accepts
/// same-block consumption by read as well as by call, and the difference is recorded in CD-063
/// rather than quietly widened.
fn validate_ephemeral_references(body: &MirBody, env: &TyEnv) -> Result<(), BackendDiagnostic> {
    for (bi, block) in body.blocks.iter().enumerate() {
        let mut created_here: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for (statement, _) in &block.statements {
            let Statement::Assign(place, rvalue) = statement else {
                continue;
            };
            if let Rvalue::RefOf { .. } = rvalue {
                if !place.projection.is_empty() {
                    return Err(BackendDiagnostic::Unsupported(format!(
                        "a borrow written through a projection ({place:?}) is outside the C5 \
                         ephemeral reference lane"
                    )));
                }
                let kind = &body.locals[place.local.0 as usize].kind;
                if !matches!(kind, LocalKind::Temp) {
                    return Err(BackendDiagnostic::Unsupported(format!(
                        "a borrow stored in a {kind:?} local is outside the C5 ephemeral \
                         reference lane: references may live only in compiler-generated \
                         temporaries, never in user bindings"
                    )));
                }
                created_here.insert(place.local.0);
            }
            // A reference VALUE may only be copied into another temporary in the same block
            // (which is how `cmp` lowers); it may not be written into a user local or an
            // aggregate.
            for operand in super::emit_projections::rvalue_operands(rvalue) {
                let (Operand::Copy(src) | Operand::Move(src)) = operand else {
                    continue;
                };
                if !created_here.contains(&src.local.0) {
                    continue;
                }
                let dest_is_temp = place.projection.is_empty()
                    && matches!(body.locals[place.local.0 as usize].kind, LocalKind::Temp);
                if matches!(rvalue, Rvalue::Aggregate(..)) || !dest_is_temp {
                    return Err(BackendDiagnostic::Unsupported(format!(
                        "a reference flowing into {place:?} is outside the C5 ephemeral \
                         reference lane: it may not be stored in an aggregate, written to a user \
                         local, or returned"
                    )));
                }
                created_here.insert(place.local.0);
            }
        }
        // Nothing created in this block may be read in any other one.
        for (other_i, other) in body.blocks.iter().enumerate() {
            if other_i == bi {
                continue;
            }
            for (statement, _) in &other.statements {
                if let Statement::Assign(_, rvalue) = statement {
                    for operand in super::emit_projections::rvalue_operands(rvalue) {
                        if let Operand::Copy(p) | Operand::Move(p) = operand {
                            if created_here.contains(&p.local.0) {
                                return Err(BackendDiagnostic::Unsupported(format!(
                                    "reference temporary _{} is created in block {bi} and used in \
                                     block {other_i}: the C5 ephemeral reference lane requires \
                                     creation and consumption in the SAME basic block",
                                    p.local.0
                                )));
                            }
                        }
                    }
                }
            }
        }
    }
    // A reference may never be returned.
    if matches!(env.local_ty(0)?, MirTy::Ref { .. }) {
        return Err(BackendDiagnostic::Unsupported(
            "returning a reference is outside the C5 ephemeral reference lane".to_string(),
        ));
    }
    Ok(())
}

fn emit_terminator(
    body: &MirBody,
    files: &[Arc<SourceFile>],
    out: &mut String,
    terminator: &Terminator,
    env: &TyEnv,
) -> Result<(), BackendDiagnostic> {
    match terminator {
        Terminator::Goto { target } => {
            out.push_str(&format!("                __bb = {}; continue;\n", target.0));
        }
        Terminator::SwitchInt {
            scrut,
            arms,
            otherwise,
        } => {
            let key = switch_key_expr(scrut, env)?;
            out.push_str(&format!("                match {key} {{\n"));
            for (value, target) in arms {
                out.push_str(&format!(
                    "                    {value}u128 => {{ __bb = {}; continue; }}\n",
                    target.0
                ));
            }
            out.push_str(&format!(
                "                    _ => {{ __bb = {}; continue; }}\n",
                otherwise.0
            ));
            out.push_str("                }\n");
        }
        Terminator::Checked {
            op,
            args,
            dest,
            target,
            trap,
        } => {
            let dest_ty = &body.locals[dest.0 as usize].ty;
            let expr = emit_checked_expr(files, *op, args, dest_ty, trap, env)?;
            out.push_str(&format!(
                "                {}\n",
                emit_assignment(&crate::mir::Place::local(*dest), &expr, env)?
            ));
            out.push_str(&format!("                __bb = {}; continue;\n", target.0));
        }
        Terminator::Call {
            callee,
            args,
            dest,
            target,
        } => {
            let call_expr = emit_call(callee, args, env)?;
            out.push_str(&format!(
                "                {}\n",
                emit_assignment(dest, &call_expr, env)?
            ));
            out.push_str(&format!("                __bb = {}; continue;\n", target.0));
        }
        Terminator::Return => {
            // A non-Copy return value is MOVED out of its slot: the callee's local is dead
            // afterwards, which is exactly what `take` records.
            let ret = if emit_places::is_slot_local(0, env)? {
                format!("{}.take()", emit_places::local_name(0))
            } else {
                emit_places::local_name(0)
            };
            out.push_str(&format!("                break {ret};\n"));
        }
        Terminator::Unreachable => {
            // WP-C4's verifier proves this block is dead (e.g. the synthetic trailer WP-C4.5's
            // return-slot elaboration appends after every straight-line function); Rust's own
            // `unreachable!()` aborts (the generated crate's `panic = "abort"` profile turns the
            // panic into an abort, matching §7.7's no-unwind requirement) if that proof is ever
            // wrong, rather than silently falling through.
            out.push_str(
                "                unreachable!(\"WP-C4 verifier proved this block is dead\");\n",
            );
        }
        Terminator::Trap { info, message } => {
            // WP-C5.2e: an UNCONDITIONAL trap -- `panic(msg)` and a failed `assert*`, as opposed
            // to `Terminator::Checked`'s conditional one. It shares the same abort call and the
            // same compile-time-resolved location, so a native `assert_eq` failure and a native
            // overflow report through one path and one stderr format.
            if message.is_some() {
                // `panic("...")`/`assert*` with a user message needs a `&str` VALUE crossing into
                // the runtime, which needs string representation -- WP-C5.3. Message-less traps
                // (every compiler-generated trap, and `assert`/`assert_eq`/`assert_ne`, which
                // `mir::lower` emits with `message: None`) are fully supported now.
                return Err(BackendDiagnostic::Unsupported(
                    "Terminator::Trap with a user message needs `&str` values (WP-C5.3); \
                     message-less traps are supported"
                        .to_string(),
                ));
            }
            let abort = emit_abort_call(files, info.category, &info.source);
            out.push_str(&format!("                {abort};\n"));
        }
        // WP-C5.3d-0. Mirrors `mir::interp`'s `drop_in_place` exactly, which is the semantic
        // authority: run the type's own destructor instance if `TypeContext::drop_impls` names
        // one, then its fields in REVERSE declaration order.
        //
        // The unit is marked dead BEFORE any glue runs (§7.5, via `ValueSlot::drop_with`), so a
        // destructor that itself traps cannot leave a live value the abort path might re-enter.
        Terminator::Drop { place, target } => {
            out.push_str(&format!("                {}\n", emit_drop(place, env)?));
            out.push_str(&format!("                __bb = {}; continue;\n", target.0));
        } // No catch-all: as of WP-C5.3d-0 every `Terminator` variant is handled, and keeping the
          // match exhaustive means a NEW variant stops this compiling rather than silently becoming
          // an `Unsupported` diagnostic nobody notices.
    }
    Ok(())
}

/// Direct calls (`Callee::Instance`) and WP-C5.4c indirect calls through a function value
/// (`Callee::FnValue`). Runtime calls (`Callee::Runtime`) land alongside whichever `RuntimeFn`
/// group first gets lowered.
fn emit_call(callee: &Callee, args: &[Operand], env: &TyEnv) -> Result<String, BackendDiagnostic> {
    // Argument emission is IDENTICAL for direct and indirect calls (§9.2): the same left-to-right
    // move/copy operand handling MIR already sequenced. Only the callee expression differs.
    let mut arg_exprs = Vec::with_capacity(args.len());
    for arg in args {
        arg_exprs.push(emit_operand(arg, env)?);
    }
    match callee {
        Callee::Instance(instance) => {
            let name = mangle::function_name_for_symbol(&instance.symbol);
            Ok(format!("{name}({})", arg_exprs.join(", ")))
        }
        // §9.1: the target is a typed Rust `fn` pointer read from the operand. No runtime signature
        // switch and no reconstruction of source call order -- MIR verification already proved the
        // operand has `MirTy::FnPtr` with matching arity/parameter/return types (§9.3), so the
        // parenthesised operand is applied directly.
        Callee::FnValue(operand) => {
            let f = emit_operand(operand, env)?;
            Ok(format!("({f})({})", arg_exprs.join(", ")))
        }
        Callee::Runtime(_) => Err(BackendDiagnostic::Unsupported(format!(
            "Callee {callee:?} has no representation yet -- runtime calls land alongside their \
             RuntimeFn support"
        ))),
    }
}

/// `dest_ty` is the type of the place being assigned. It is needed because
/// `Rvalue::Aggregate(AggKind::Struct(item), ..)` names the nominal ITEM but not its type
/// arguments — a monomorphised instance is `(item, args)`, and only the destination knows the
/// args. Reading them from the destination is not inference: verified MIR guarantees the
/// assignment is well-typed, so the destination's type IS the aggregate's type.
/// A MIR `Drop(place)`, lowered to `slot.drop_with(|v| <glue>)`.
///
/// WP-C5.3d-1b (CD-062): the glue is an APPLICATION of `mir::drop_plan`, the canonical plan the
/// MIR interpreter also consumes. This function decides only how a step is spelled in Rust —
/// which destructor symbol to call, how to reach a component, how to bind a variant payload. It
/// decides nothing about order, coverage, or which components carry an obligation. Those were
/// previously reconstructed here independently of the interpreter, and the reconstruction had
/// already drifted (CD-060: the enum arm dropped no payload at all).
/// WP-C5.3d-1c: a **sub-place** `Drop` goes through a generated `stark_proj` wrapper instead.
///
/// This is not an edge case. MIR's drop elaboration decomposes any aggregate with more than one
/// drop unit into per-unit, flag-guarded `Drop`s on projected places — `drop _1.1` then
/// `drop _1.0` — so a plain two-droppable-field struct arrives here projected. Collapsing that
/// into a whole-local drop would violate §7.6 (it would destroy a unit MIR's flags say is already
/// gone), which is why the previous refusal was correct and why the fix is a real per-unit
/// operation rather than a relaxation.
fn emit_drop(place: &crate::mir::Place, env: &TyEnv) -> Result<String, BackendDiagnostic> {
    let ty = env.place_ty(place)?;
    if emit_types::mir_ty_is_copy(&ty, env.types) {
        return Err(BackendDiagnostic::Unsupported(format!(
            "MIR emitted Drop on the Copy type {ty:?}, which has no destructor and no slot"
        )));
    }
    if !place.projection.is_empty() {
        if !emit_places::is_slot_local(place.local.0, env)? {
            return Err(BackendDiagnostic::Unsupported(format!(
                "sub-place Drop of {place:?} whose base local is not slot-backed: there is no \
                 per-unit liveness to update"
            )));
        }
        // The wrapper carries the plan; the call site carries none of it, so an emitted body
        // still contains no destruction logic and no `unsafe`.
        let helper =
            emit_projections::collect_for_place(place, env, emit_projections::HelperOp::Drop)?;
        return Ok(format!(
            "stark_proj::{helper}(&mut {});",
            emit_places::local_name(place.local.0)
        ));
    }
    let glue = emit_drop_glue(&ty, "__v", env)?;
    Ok(format!(
        "{}.drop_with(|__v| {{ {glue} }});",
        emit_places::local_name(place.local.0)
    ))
}

/// The destructor sequence for `ty`, applied to the Rust expression `value` (an `&mut`).
fn emit_drop_glue(ty: &MirTy, value: &str, env: &TyEnv) -> Result<String, BackendDiagnostic> {
    let plan = drop_plan::plan_for(ty, env.types)
        .map_err(|e| BackendDiagnostic::Unsupported(e.to_string()))?;
    emit_drop_plan(&plan, value)
}

/// Apply one [`DropPlan`] to the Rust `&mut` expression `value`.
///
/// Every ordering question — destructor before components, components back to front, one arm per
/// variant, `Copy` components absent — is already answered by the plan's SHAPE. This function
/// walks it in the order given.
pub(super) fn emit_drop_plan(plan: &DropPlan, value: &str) -> Result<String, BackendDiagnostic> {
    let mut out = String::new();
    match plan {
        DropPlan::Noop => {}
        // The type's own destructor, taking `&mut self` -- the same receiver shape
        // `mir::interp` passes. `then` is emitted after it because the plan nests it there.
        DropPlan::Destructor { symbol, then } => {
            let name = mangle::function_name_for_symbol(symbol);
            out.push_str(&format!("{name}({value}); "));
            out.push_str(&emit_drop_plan(then, value)?);
        }
        DropPlan::Fields { base, fields } => {
            for field in fields {
                let access = component_access(base, field.index, value)?;
                out.push_str(&emit_drop_plan(&field.plan, &access)?);
            }
        }
        DropPlan::Variants { base, variants } => {
            let name = emit_types::nominal_type_name(base).ok_or_else(|| {
                BackendDiagnostic::Unsupported(format!("no generated name for {base:?}"))
            })?;
            // One arm per variant -- including the ones whose payloads need no glue -- so the
            // match is exhaustive without a catch-all, and a variant added to the plan cannot
            // silently acquire a no-op arm here.
            let mut arms = Vec::with_capacity(variants.len());
            for (v, variant) in variants.iter().enumerate() {
                let mut binders: Vec<String> =
                    (0..variant.arity).map(|_| "_".to_string()).collect();
                let mut body = String::new();
                for field in &variant.fields {
                    let binder = format!("__p{}", field.index);
                    binders[field.index as usize] = binder.clone();
                    body.push_str(&emit_drop_plan(&field.plan, &binder)?);
                }
                arms.push(format!(
                    "{name}::{}({}) => {{ {body} }}",
                    emit_types::variant_name(v as u32),
                    binders.join(", ")
                ));
            }
            out.push_str(&format!("match {value} {{ {} }}; ", arms.join(", ")));
        }
        DropPlan::Array { len, elem } => {
            for i in drop_plan::array_order(*len) {
                out.push_str(&emit_drop_plan(elem, &format!("(&mut {value}[{i}])"))?);
            }
        }
        // `Vec`/`Box` own an allocation the generated crate has no representation for yet; the
        // plan names the step, and refusing it is the honest outcome rather than emitting glue
        // that destroys the elements and leaks the buffer.
        DropPlan::VecElements { .. } | DropPlan::BoxInner { .. } => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "drop glue for {plan:?} needs the owning-runtime-type representation, which is \
                 outside the C5.3 subset"
            )))
        }
    }
    Ok(out)
}

/// How to reach component `index` of an aggregate whose `&mut` expression is `value`.
///
/// The two spellings exist because §6.2 maps a MIR tuple to a Rust tuple (`.0`) and a MIR struct
/// to a generated named type (`.f0`) — the same split `emit_places` resolves for reads.
fn component_access(base: &MirTy, index: u32, value: &str) -> Result<String, BackendDiagnostic> {
    match base {
        MirTy::Struct(..) => Ok(format!("(&mut {value}.{})", emit_types::field_name(index))),
        MirTy::Tuple(..) => Ok(format!("(&mut {value}.{index})")),
        other => Err(BackendDiagnostic::Unsupported(format!(
            "drop plan named a field of {other:?}, which has no component syntax"
        ))),
    }
}

/// Emit one assignment statement, choosing the form the destination requires.
///
/// Three cases, and the split is forced rather than stylistic:
///
/// - **a slot-backed whole local** → `_1.write(value);`. `write` is a method call, not a place
///   expression, so this cannot be spliced as `dest = value`. It also asserts the slot is dead,
///   which is MIR's own invariant: MIR emits an explicit `Drop` before reassigning a live place.
/// - **a projected place through a slot** → `_1.get_mut().f0 = value;`, mutable access to a
///   still-live value.
/// - **an ordinary Copy local** → `_1 = value;` as before.
///
/// A read through an enum variant field is refused: its read form is a `match` EXPRESSION, which
/// is not a Rust place. MIR lowering never produces such a destination — `VariantField` appears
/// only under `read_place` and pattern tests, and STARK has no syntax for assigning into a
/// payload — so this is a guard against a silently wrong splice, not a limitation.
fn emit_assignment(
    place: &crate::mir::Place,
    value: &str,
    env: &TyEnv,
) -> Result<String, BackendDiagnostic> {
    if emit_places::reads_through_variant_field(place) {
        return Err(BackendDiagnostic::Unsupported(format!(
            "assigning THROUGH an enum variant field ({place:?}) is not representable: the read \
             form is a `match` expression, which is not a Rust place. No MIR lowering path emits \
             this, so reaching it means either a new lowering shape or a compiler defect"
        )));
    }
    if emit_places::is_slot_local(place.local.0, env)? && place.projection.is_empty() {
        return Ok(format!(
            "{}.write({value});",
            emit_places::local_name(place.local.0)
        ));
    }
    Ok(format!(
        "{} = {value};",
        emit_places::emit_place_mut(place, env)?
    ))
}

fn emit_rvalue(rvalue: &Rvalue, dest_ty: &MirTy, env: &TyEnv) -> Result<String, BackendDiagnostic> {
    match rvalue {
        Rvalue::Use(operand) => emit_operand(operand, env),
        Rvalue::UnOp(MirUnOp::Not, operand) => Ok(format!("(!({}))", emit_operand(operand, env)?)),
        Rvalue::UnOp(MirUnOp::FloatNeg, operand) => {
            Ok(format!("(-({}))", emit_operand(operand, env)?))
        }
        Rvalue::BinOp(op, lhs, rhs) => {
            emit_binop(*op, emit_operand(lhs, env)?, emit_operand(rhs, env)?)
        }
        // WP-C5.3b: the variant index of an enum value. Same shape as a variant-field read and
        // for the same reason -- Rust exposes no integer discriminant for an enum with payloads,
        // so the index is recovered by matching. Every variant is listed, so there is no
        // catch-all arm and adding a variant cannot silently fall through to a wrong index.
        // WP-C5.3d-1a: an ephemeral borrow. The lane forbids storing, returning or carrying it
        // across blocks, so this never needs reference STORAGE -- it is a borrow expression the
        // validator has already proved is consumed in the same block.
        Rvalue::RefOf { mutable, place } => {
            // DEV-098: two corrections, both of which stayed hidden because only the destructor
            // path exercised `&mut` before, and that one is emitted by the drop glue rather than
            // through here.
            //
            // 1. A MUTABLE borrow must reach the place mutably. `emit_place`'s read mode spells a
            //    slot-backed local as `_1.get()`, an `&T`, so `&mut _1.get()` is "cannot borrow
            //    data in a `&` reference as mutable".
            // 2. A WHOLE slot-backed local's accessor already RETURNS the reference -- `get_mut()`
            //    is `&mut T` -- so wrapping it again builds `&mut &mut T` over a temporary and
            //    fails with "temporary value dropped while borrowed". Only a projected place
            //    (`_1.get_mut().f0`) is a Rust place expression needing the borrow operator.
            let whole_slot =
                place.projection.is_empty() && emit_places::is_slot_local(place.local.0, env)?;
            Ok(if *mutable {
                let base = emit_places::emit_place_mut(place, env)?;
                if whole_slot {
                    base
                } else {
                    format!("(&mut {base})")
                }
            } else {
                let base = emit_places::emit_place(place, env)?;
                if whole_slot {
                    base
                } else {
                    format!("(&{base})")
                }
            })
        }
        Rvalue::Discriminant(place) => {
            let base = emit_places::emit_place(place, env)?;
            let ty = env.place_ty(place)?;
            let MirTy::Enum(enum_ref, args) = &ty else {
                return Err(BackendDiagnostic::Unsupported(format!(
                    "Discriminant of the non-enum type {ty:?}"
                )));
            };
            let variants =
                emit_types::variant_payloads(enum_ref, args, env.types).ok_or_else(|| {
                    BackendDiagnostic::Unsupported(format!("no variant table for {ty:?}"))
                })?;
            let name = emit_types::nominal_type_name(&ty).ok_or_else(|| {
                BackendDiagnostic::Unsupported(format!("no generated name for {ty:?}"))
            })?;
            // The arms are typed by the DESTINATION, not by a fixed width: MIR types the
            // discriminant local itself (Int64 today), and a hardcoded `i128` literal made the
            // generated crate fail to compile with "expected i64, found i128".
            let discriminant_ty = emit_types::emit_ty(dest_ty)?;
            let arms: Vec<String> = variants
                .iter()
                .enumerate()
                .map(|(v, _)| {
                    format!(
                        "{name}::{}(..) => {v}{discriminant_ty}",
                        emit_types::variant_name(v as u32)
                    )
                })
                .collect();
            Ok(format!("(match &{base} {{ {} }})", arms.join(", ")))
        }
        // WP-C5.3a: aggregate construction. Each kind has a direct Rust constructor, so nothing
        // here reconstructs source syntax -- the operand order IS the MIR field/element order.
        Rvalue::Aggregate(kind, operands) => {
            let mut parts = Vec::with_capacity(operands.len());
            for operand in operands {
                parts.push(emit_operand(operand, env)?);
            }
            Ok(match kind {
                AggKind::Tuple => match parts.len() {
                    0 => "()".to_string(),
                    1 => format!("({},)", parts[0]),
                    _ => format!("({})", parts.join(", ")),
                },
                AggKind::Array(_) => format!("[{}]", parts.join(", ")),
                AggKind::Struct(item) => {
                    let MirTy::Struct(dest_item, args) = dest_ty else {
                        return Err(BackendDiagnostic::Unsupported(format!(
                            "struct aggregate assigned to a non-struct destination {dest_ty:?}"
                        )));
                    };
                    if dest_item.0 != item.0 {
                        return Err(BackendDiagnostic::Unsupported(format!(
                            "struct aggregate names item {} but its destination is item {}",
                            item.0, dest_item.0
                        )));
                    }
                    let name = mangle::type_name_for_nominal(item.0, args);
                    let fields: Vec<String> = parts
                        .iter()
                        .enumerate()
                        .map(|(i, value)| format!("{}: {value}", emit_types::field_name(i as u32)))
                        .collect();
                    format!("{name} {{ {} }}", fields.join(", "))
                }
                AggKind::EnumVariant(_, variant) => {
                    // Like the struct case, the type ARGUMENTS come from the destination:
                    // `AggKind::EnumVariant` names the enum and the variant, not the instance.
                    let MirTy::Enum(..) = dest_ty else {
                        return Err(BackendDiagnostic::Unsupported(format!(
                            "enum aggregate assigned to a non-enum destination {dest_ty:?}"
                        )));
                    };
                    let name = emit_types::nominal_type_name(dest_ty).ok_or_else(|| {
                        BackendDiagnostic::Unsupported(format!("no generated name for {dest_ty:?}"))
                    })?;
                    format!(
                        "{name}::{}({})",
                        emit_types::variant_name(*variant),
                        parts.join(", ")
                    )
                }
            })
        }
        // WP-C5.3e (CD-067): a layout query answers from the selected named target CONTRACT,
        // emitted as a CONSTANT. It must NOT be `core::mem::size_of::<T>()`: that would report
        // this backend's private physical representation, making the observable answer depend on
        // a transitional backend and on `repr(Rust)`'s deliberately unspecified field ordering.
        // The generated crate is free to lay its types out however rustc likes -- nothing here
        // asserts the two agree, because no STARK program can observe that they do.
        Rvalue::LayoutQuery { kind, ty } => {
            let layout = env
                .layout
                .layout_of(ty, env.types)
                .map_err(|e| BackendDiagnostic::Unsupported(format!("layout query: {}", e.0)))?;
            Ok(match kind {
                crate::mir::LayoutKind::SizeOf => format!("{}u64", layout.size),
                crate::mir::LayoutKind::AlignOf => format!("{}u64", layout.align),
            })
        } // No catch-all: every `Rvalue` variant is handled as of WP-C5.3d-1a (`RefOf` was the
          // last), so a new one stops compilation instead of silently degrading.
    }
}

/// Exhaustive over `MirBinOp` on purpose (no `other` arm) -- every binary operator MIR can
/// produce today has a Rust token; if a new `MirBinOp` variant is ever added, this stops
/// compiling instead of silently falling through to `Unsupported`.
fn emit_binop(op: MirBinOp, l: String, r: String) -> Result<String, BackendDiagnostic> {
    use MirBinOp::*;
    let rust_op = match op {
        Eq => "==",
        Ne => "!=",
        Lt => "<",
        Le => "<=",
        Gt => ">",
        Ge => ">=",
        FloatAdd => "+",
        FloatSub => "-",
        FloatMul => "*",
        BitAnd => "&",
        BitOr => "|",
        BitXor => "^",
    };
    Ok(format!("(({l}) {rust_op} ({r}))"))
}

/// A `Copy`-classified scalar's `Move` is value-identical to its `Copy` -- WP-C5.2c only ever
/// declares `Copy` locals (`emit_types::emit_ty` admits primitives only, and every primitive
/// `MirTy` is `Copy` by construction, CLAUDE.md: "Copy requires all-Copy fields"), so both
/// operand kinds emit the same bare place reference here. Real non-`Copy` move/liveness tracking
/// (`WP-C5-ENTRY.md` §7.2's `MaybeUninit<ManuallyDrop<T>>` strategy) is deferred to whichever WP
/// first admits a non-`Copy` `MirTy` (WP-C5.3+).
fn emit_operand(operand: &Operand, env: &TyEnv) -> Result<String, BackendDiagnostic> {
    match operand {
        Operand::Const(c) => emit_types::emit_constant(c),
        // DEV-098 (CD-070): `Operand::Copy` means "read WITHOUT consuming". For an exclusive
        // reference that is a REBORROW, not a bare read -- a bare read of a Rust `&mut` is a
        // move, so a second use of the same reference in the same block would fail to compile.
        //
        // The previous record claimed the ephemeral-reference lane contained this by enforcing
        // single use. It does not: `validate_ephemeral_references` checks where a reference is
        // created, that it stays in one block, and that it is never stored or returned -- it
        // never counts uses. So the shape reached rustc, which is exactly what the lane promises
        // cannot happen. Emitting the reborrow states the MIR meaning directly and needs no
        // artificial one-use restriction.
        Operand::Copy(place) => {
            if matches!(env.place_ty(place)?, MirTy::Ref { mutable: true, .. }) {
                return Ok(format!("(&mut *{})", emit_places::emit_place(place, env)?));
            }
            emit_places::emit_place(place, env)
        }
        // WP-C5.3d-0. A move out of a Copy place is a read like any other -- MIR only marks it
        // `Move` for uniformity, and Rust copies. A move out of a NON-Copy place must go through
        // the slot, which records that the source unit is now dead; that record is the whole
        // reason the block-dispatch loop no longer defeats the borrow checker.
        Operand::Move(place) => {
            let ty = env.place_ty(place)?;
            if emit_types::mir_ty_is_copy(&ty, env.types) {
                return emit_places::emit_place(place, env);
            }
            // DEV-098: a reference is non-`Copy` at MIR level but is never slot-backed, so it has
            // no per-unit liveness to update and `emit_move_out` would refuse it outright. Moving
            // a reference IS a plain Rust move of the reference value. Without this, passing
            // `&mut x` to a user function failed with "move out of the non-slot place" -- the
            // lane admitted the shape and the emitter then could not spell it.
            if matches!(ty, MirTy::Ref { .. }) {
                return emit_places::emit_place(place, env);
            }
            emit_places::emit_move_out(place, env)
        }
    }
}

/// Resolves an operand's MIR type -- needed for `SwitchInt`'s bit-pattern key and `Cast`'s
/// source type, neither of which `Operand` itself carries. WP-C5.3a: projected places resolve
/// through `TyEnv`, so a switch on a struct field or an array element is no longer a special
/// case that has to be refused.
fn operand_mir_ty(operand: &Operand, env: &TyEnv) -> Result<MirTy, BackendDiagnostic> {
    match operand {
        Operand::Const(Constant::Bool(_)) => Ok(MirTy::Bool),
        Operand::Const(Constant::Int(_, ty)) => Ok(ty.clone()),
        Operand::Const(Constant::Float(_, ty)) => Ok(ty.clone()),
        Operand::Const(Constant::Unit) => Ok(MirTy::Unit),
        Operand::Copy(place) | Operand::Move(place) => env.place_ty(place),
        other => Err(BackendDiagnostic::Unsupported(format!(
            "operand {other:?} has no WP-C5.3a type resolution yet"
        ))),
    }
}

/// `SwitchInt`'s arm keys are `u128`, computed by `mir::interp::run` (the oracle this must
/// match) as: `Bool -> 0/1`, `Int (any width, including Char's codepoint encoding) -> the value
/// reinterpreted as `i128` then `u128`` (`v as u128` on the interpreter's `i128`-carrier value
/// -- i.e. sign-extension THEN bit-reinterpretation, not a same-width truncation). Reproduced
/// here by casting through `i128` explicitly so the same sign-extension happens regardless of
/// the Rust-side value's actual declared width.
fn switch_key_expr(operand: &Operand, env: &TyEnv) -> Result<String, BackendDiagnostic> {
    let ty = operand_mir_ty(operand, env)?;
    let value = emit_operand(operand, env)?;
    match ty {
        MirTy::Bool => Ok(format!("(if {value} {{ 1u128 }} else {{ 0u128 }})")),
        MirTy::Char => Ok(format!("(({value} as u32) as i128 as u128)")),
        MirTy::Int8
        | MirTy::Int16
        | MirTy::Int32
        | MirTy::Int64
        | MirTy::UInt8
        | MirTy::UInt16
        | MirTy::UInt32
        | MirTy::UInt64 => Ok(format!("(({value}) as i128 as u128)")),
        other => Err(BackendDiagnostic::Unsupported(format!(
            "SwitchInt on {other:?} has no WP-C5.2c representation yet"
        ))),
    }
}

/// `starkc::mir::TrapCategory` and `stark_runtime::trap::TrapCategory` share identical variant
/// names by design (`stark-runtime/src/trap.rs`'s own doc comment states the coupling); this
/// reuses `{:?}` rather than a hand-written match so the two cannot silently drift without a
/// compile error surfacing somewhere (adding a variant to one and not the other is still not
/// caught HERE, but is caught the moment generated code fails to name a real
/// `stark_runtime::trap::TrapCategory` variant).
fn trap_category_token(category: TrapCategory) -> String {
    format!("stark_runtime::trap::TrapCategory::{category:?}")
}

/// WP-C5.2e: resolves a `SourceInfo` to (file path, 1-based line, 1-based column) at COMPILE
/// TIME, using the same `SourceFile::line_col` the rest of the compiler's diagnostics already
/// use (`04-Semantic-Analysis.md`'s 1-based convention) -- not a new position-mapping scheme.
fn resolve_source_location(files: &[Arc<SourceFile>], info: &SourceInfo) -> (String, u32, u32) {
    let file = &files[info.file.0 as usize];
    let (line, col) = file.line_col(info.span.lo);
    (file.name.clone(), line as u32, col as u32)
}

/// A Rust string-literal token. File paths are compiler-controlled (from `MirProgram::files`,
/// not raw user input), but escaped defensively rather than trusted verbatim regardless.
fn rust_str_lit(s: &str) -> String {
    format!("\"{}\"", s.escape_default())
}

/// The full `stark_runtime::trap::abort(...)` call expression for a given category and source
/// location -- the one place that assembles a trap abort call, so every trap site (the
/// terminator's own default category, and the `Shl`/`Shr` `InvalidShift` override) goes through
/// the same construction.
fn emit_abort_call(
    files: &[Arc<SourceFile>],
    category: TrapCategory,
    source: &SourceInfo,
) -> String {
    let (file, line, col) = resolve_source_location(files, source);
    format!(
        "stark_runtime::trap::abort({}, {}, {line}, {col})",
        trap_category_token(category),
        rust_str_lit(&file),
    )
}

fn int_bounds_tokens(ty: &MirTy) -> Result<(&'static str, &'static str), BackendDiagnostic> {
    Ok(match ty {
        MirTy::Int8 => ("(i8::MIN as i128)", "(i8::MAX as i128)"),
        MirTy::Int16 => ("(i16::MIN as i128)", "(i16::MAX as i128)"),
        MirTy::Int32 => ("(i32::MIN as i128)", "(i32::MAX as i128)"),
        MirTy::Int64 => ("(i64::MIN as i128)", "(i64::MAX as i128)"),
        MirTy::UInt8 => ("0i128", "(u8::MAX as i128)"),
        MirTy::UInt16 => ("0i128", "(u16::MAX as i128)"),
        MirTy::UInt32 => ("0i128", "(u32::MAX as i128)"),
        MirTy::UInt64 => ("0i128", "(u64::MAX as i128)"),
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "integer bounds requested for non-integer MirTy {other:?}"
            )))
        }
    })
}

/// The float→int cast range test's bounds, as EXACT `f64` literals: `(min, upper_exclusive)`,
/// where the accept condition is `min <= truncated < upper_exclusive`.
///
/// This is deliberately separate from [`int_bounds_tokens`], whose `(min, max)` pair is compared
/// in exact `i128` arithmetic by the checked-arithmetic path and is correct there. Reusing it for
/// a FLOAT comparison is not: `u64::MAX as f64` rounds up to 2^64 and `i64::MAX as f64` rounds up
/// to 2^63, so an inclusive `truncated > (max as f64)` test accepts exactly 2^64 / 2^63 and Rust's
/// saturating `as` then silently clamps them into range -- admitting a value 03-Type-System.md
/// requires to trap. Every `max + 1` here is a power of two, hence exactly representable as `f64`,
/// so the half-open form is exact at every width. Mirrors `mir::interp`'s `Cast` arm; the two
/// engines must agree, and the differential comparator checks that they do.
fn int_float_bounds_tokens(ty: &MirTy) -> Result<(&'static str, &'static str), BackendDiagnostic> {
    Ok(match ty {
        MirTy::Int8 => ("-128.0f64", "128.0f64"),
        MirTy::Int16 => ("-32768.0f64", "32768.0f64"),
        MirTy::Int32 => ("-2147483648.0f64", "2147483648.0f64"),
        MirTy::Int64 => ("-9223372036854775808.0f64", "9223372036854775808.0f64"),
        MirTy::UInt8 => ("0.0f64", "256.0f64"),
        MirTy::UInt16 => ("0.0f64", "65536.0f64"),
        MirTy::UInt32 => ("0.0f64", "4294967296.0f64"),
        MirTy::UInt64 => ("0.0f64", "18446744073709551616.0f64"),
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "float cast bounds requested for non-integer MirTy {other:?}"
            )))
        }
    })
}

fn int_width_tokens(ty: &MirTy) -> Result<&'static str, BackendDiagnostic> {
    Ok(match ty {
        MirTy::Int8 | MirTy::UInt8 => "8i128",
        MirTy::Int16 | MirTy::UInt16 => "16i128",
        MirTy::Int32 | MirTy::UInt32 => "32i128",
        MirTy::Int64 | MirTy::UInt64 => "64i128",
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "integer width requested for non-integer MirTy {other:?}"
            )))
        }
    })
}

/// Mirrors `mir::interp::eval_checked` exactly (that function is the semantic oracle): every
/// integer op is computed at `i128` width, then range-filtered against the DESTINATION type,
/// rather than using the narrower Rust type's own `checked_*` directly. For `Add`/`Sub`/`Mul`/
/// `Div`/`Rem`/`Neg`/`Pow` this happens to be provably equivalent to native narrow-width checked
/// arithmetic (the true mathematical result either fits the narrow range or it doesn't, and
/// `i128` can never itself overflow for these widths) -- but for `Shl` it is NOT equivalent:
/// Rust's native `checked_shl` only validates the *shift count*, silently dropping overflowed
/// bits within the narrow type, whereas STARK traps `IntegerOverflow` when a left shift's TRUE
/// result does not fit. The `i128`-widen-then-filter approach is used uniformly for all of
/// these rather than optimizing the ones where it isn't required, to stay a provable match
/// against the oracle rather than a "should be equivalent" argument re-derived per operator.
fn emit_checked_expr(
    files: &[Arc<SourceFile>],
    op: CheckedOp,
    args: &[Operand],
    dest_ty: &MirTy,
    trap: &TrapInfo,
    env: &TyEnv,
) -> Result<String, BackendDiagnostic> {
    use CheckedOp::*;
    let default_abort = emit_abort_call(files, trap.category, &trap.source);
    match op {
        Add | Sub | Mul | Div | Rem | Neg | Pow => {
            let dest_rust_ty = emit_types::emit_ty(dest_ty)?;
            let (min, max) = int_bounds_tokens(dest_ty)?;
            let a = emit_operand(&args[0], env)?;
            let checked = match op {
                Add => format!(
                    "(({a}) as i128).checked_add(({}) as i128)",
                    emit_operand(&args[1], env)?
                ),
                Sub => format!(
                    "(({a}) as i128).checked_sub(({}) as i128)",
                    emit_operand(&args[1], env)?
                ),
                Mul => format!(
                    "(({a}) as i128).checked_mul(({}) as i128)",
                    emit_operand(&args[1], env)?
                ),
                Div => format!(
                    "(({a}) as i128).checked_div(({}) as i128)",
                    emit_operand(&args[1], env)?
                ),
                Rem => format!(
                    "(({a}) as i128).checked_rem(({}) as i128)",
                    emit_operand(&args[1], env)?
                ),
                Neg => format!("(({a}) as i128).checked_neg()"),
                // A5: exponent must be nonnegative and fit in u32 (NUM-INT-ARITH-001);
                // `u32::try_from` rejects both a negative exponent and one wider than u32.
                Pow => format!(
                    "u32::try_from(({}) as i128).ok().and_then(|__e| (({a}) as i128).checked_pow(__e))",
                    emit_operand(&args[1], env)?
                ),
                _ => unreachable!(),
            };
            Ok(format!(
                "match {checked} {{ Some(__v) if __v >= {min} && __v <= {max} => __v as \
                 {dest_rust_ty}, _ => {default_abort} }}"
            ))
        }
        Shl | Shr => {
            let dest_rust_ty = emit_types::emit_ty(dest_ty)?;
            let (min, max) = int_bounds_tokens(dest_ty)?;
            let width = int_width_tokens(dest_ty)?;
            let l = emit_operand(&args[0], env)?;
            let count = emit_operand(&args[1], env)?;
            let method = if matches!(op, Shl) {
                "checked_shl"
            } else {
                "checked_shr"
            };
            let invalid_shift_abort =
                emit_abort_call(files, TrapCategory::InvalidShift, &trap.source);
            Ok(format!(
                "{{ let __count = ({count}) as i128; if __count < 0 || __count >= {width} {{ \
                 {invalid_shift_abort} }} else {{ match (({l}) as \
                 i128).{method}(__count as u32) {{ Some(__v) if __v >= {min} && __v <= {max} => \
                 __v as {dest_rust_ty}, _ => {default_abort} \
                 }} }} }}"
            ))
        }
        FloatDiv | FloatRem => {
            let dest_rust_ty = emit_types::emit_ty(dest_ty)?;
            let l = emit_operand(&args[0], env)?;
            let r = emit_operand(&args[1], env)?;
            let rust_op = if matches!(op, FloatDiv) { "/" } else { "%" };
            Ok(format!(
                "{{ let __l: {dest_rust_ty} = {l}; let __r: {dest_rust_ty} = {r}; if __r == 0.0 \
                 {{ {default_abort} }} else {{ __l {rust_op} \
                 __r }} }}"
            ))
        }
        Cast => emit_cast_expr(files, &args[0], dest_ty, trap, env),
        // WP-C5.3a: the bounds check that DEFINES an index proof for `Projection::Index`. The
        // proof value is the validated index itself, so the projection can then index without a
        // second check. Compared at `i128` for the same reason the arithmetic ops are: the index
        // operand may be any signed integer width, and a negative index must trap rather than
        // wrap into a huge `usize`.
        CheckIndex => {
            let array = emit_operand(&args[0], env)?;
            let index = emit_operand(&args[1], env)?;
            let dest_rust_ty = emit_types::emit_ty(dest_ty)?;
            Ok(format!(
                "{{ let __i = ({index}) as i128; \
                 let __n = ({array}).len() as i128; \
                 if __i < 0 || __i >= __n {{ {default_abort} }} else {{ __i as {dest_rust_ty} }} }}"
            ))
        }
    }
}

fn emit_cast_expr(
    files: &[Arc<SourceFile>],
    source: &Operand,
    dest_ty: &MirTy,
    trap: &TrapInfo,
    env: &TyEnv,
) -> Result<String, BackendDiagnostic> {
    let src_ty = operand_mir_ty(source, env)?;
    let a = emit_operand(source, env)?;
    let dest_rust_ty = emit_types::emit_ty(dest_ty)?;
    let src_is_int = int_bounds_tokens(&src_ty).is_ok();
    let dest_is_int = int_bounds_tokens(dest_ty).is_ok();
    let src_is_float = matches!(src_ty, MirTy::Float32 | MirTy::Float64);
    let dest_is_float = matches!(dest_ty, MirTy::Float32 | MirTy::Float64);
    let default_abort = emit_abort_call(files, trap.category, &trap.source);

    if src_is_int && dest_is_int {
        let (min, max) = int_bounds_tokens(dest_ty)?;
        return Ok(format!(
            "{{ let __v = ({a}) as i128; if __v >= {min} && __v <= {max} {{ __v as \
             {dest_rust_ty} }} else {{ {default_abort} }} }}"
        ));
    }
    if src_is_int && dest_is_float {
        // Always succeeds (interp: `(MirValue::Int(v), Float32|64) => Some(Float(*v as f64))`).
        return Ok(format!("(({a}) as {dest_rust_ty})"));
    }
    if src_is_float && dest_is_float {
        // Always succeeds; Rust's `as` between f32/f64 matches interp's rounding.
        return Ok(format!("(({a}) as {dest_rust_ty})"));
    }
    if src_is_float && dest_is_int {
        // Half-open, on EXACT f64 bounds -- see `int_float_bounds_tokens` for why the inclusive
        // `int_bounds_tokens` pair is wrong here. Widening an f32 source to f64 is exact, so the
        // comparison is performed at one width for every source type.
        let (min, upper_exclusive) = int_float_bounds_tokens(dest_ty)?;
        return Ok(format!(
            "{{ let __f = ({a}) as f64; let __t = __f.trunc(); if __f.is_nan() || __t < {min} \
             || __t >= {upper_exclusive} {{ {default_abort} \
             }} else {{ __t as {dest_rust_ty} }} }}"
        ));
    }
    Err(BackendDiagnostic::Unsupported(format!(
        "cast {src_ty:?} -> {dest_ty:?} has no WP-C5.2c representation yet"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{BlockId, EnumRef, Instance, LocalDecl, LocalKind};

    fn env_body() -> MirBody {
        MirBody {
            instance: Instance {
                item: crate::hir::ItemId(0),
                symbol: "t@[]".to_string(),
                type_args: vec![],
            },
            params: vec![],
            ret: MirTy::Unit,
            locals: vec![LocalDecl {
                ty: MirTy::Unit,
                kind: LocalKind::Temp,
            }],
            blocks: vec![],
            entry: BlockId(0),
        }
    }

    /// **The correction this test exists for.** An earlier `emit_drop_glue` walked only
    /// `struct_fields`, so dropping a whole non-`Copy` ENUM located a possible user destructor
    /// and then stopped — never matching the active variant, never dropping its payload. The
    /// value leaked, and Miri could not report it because the slot tests ignore leaks by design.
    ///
    /// Correct glue matches EVERY variant (no catch-all, so a new variant cannot silently
    /// acquire a no-op drop) and drops each droppable payload field in reverse order, mirroring
    /// `mir::interp::drop_in_place`.
    #[test]
    fn enum_drop_glue_matches_every_variant_and_drops_its_payload() {
        let mut types = TypeContext::default();
        // A droppable inner struct: it has its own destructor instance, so it needs glue.
        types.struct_fields.insert((1, vec![]), vec![MirTy::Int32]);
        types
            .drop_impls
            .insert((1, vec![]), "Inner::drop@[]".to_string());
        // An enum whose variant 1 carries two droppable fields, and whose variant 0 carries none.
        types.enum_variants.insert(
            (2, vec![]),
            vec![
                vec![],
                vec![
                    MirTy::Struct(crate::hir::ItemId(1), vec![]),
                    MirTy::Struct(crate::hir::ItemId(1), vec![]),
                ],
            ],
        );

        let body = env_body();
        let layout = crate::layout::TargetLayout::default();
        let env = TyEnv::new(&body, &types, &layout);
        let ty = MirTy::Enum(EnumRef::User(crate::hir::ItemId(2)), vec![]);
        let glue = emit_drop_glue(&ty, "__v", &env).expect("enum glue must emit");

        // Both variants appear: the payload-free one too, so the match needs no catch-all.
        assert!(
            glue.contains("::V0()"),
            "every variant must have an arm: {glue}"
        );
        assert!(
            glue.contains("::V1("),
            "the payload variant must have an arm: {glue}"
        );
        // Reverse declaration order, checked on the arm BODY -- the pattern binders necessarily
        // list `__p0` first, so searching the whole string would compare the wrong occurrences.
        let body = glue
            .split_once("__p1) => {")
            .expect("the payload arm must bind both fields")
            .1;
        let first = body.find("__p1").expect("field 1 must be dropped");
        let second = body.find("__p0").expect("field 0 must be dropped");
        assert!(
            first < second,
            "payload fields must drop in REVERSE declaration order: {glue}"
        );
        // And each one dispatches the concrete destructor the type context names.
        assert_eq!(
            glue.matches(&mangle::function_name_for_symbol("Inner::drop@[]"))
                .count(),
            2,
            "both droppable payload fields must dispatch their destructor: {glue}"
        );
    }

    /// A `Copy` payload field carries no drop obligation, so it must NOT be bound or dropped --
    /// binding it would be harmless but dropping it would be wrong.
    ///
    /// WP-C5.3d-1b sharpened this: since the plan omits `Noop` components entirely, an enum whose
    /// payloads are ALL `Copy` produces no glue at all, and the mixed case below is what proves
    /// the `_` binder.
    #[test]
    fn enum_drop_glue_skips_copy_payload_fields() {
        let mut types = TypeContext::default();
        types.struct_fields.insert((1, vec![]), vec![MirTy::Int32]);
        types
            .drop_impls
            .insert((1, vec![]), "Inner::drop@[]".to_string());
        // V0 mixes a Copy field with a droppable one; V1 is entirely Copy.
        types.enum_variants.insert(
            (3, vec![]),
            vec![
                vec![MirTy::Int32, MirTy::Struct(crate::hir::ItemId(1), vec![])],
                vec![MirTy::Bool],
            ],
        );
        let body = env_body();
        let layout = crate::layout::TargetLayout::default();
        let env = TyEnv::new(&body, &types, &layout);
        let ty = MirTy::Enum(EnumRef::User(crate::hir::ItemId(3)), vec![]);
        let glue = emit_drop_glue(&ty, "__v", &env).expect("glue must emit");
        assert!(
            glue.contains("::V0(_, __p1)"),
            "a Copy payload must be ignored, not bound: {glue}"
        );
        assert!(
            glue.contains("::V1(_)"),
            "an all-Copy variant still needs an arm, with nothing bound: {glue}"
        );
    }

    /// The `Copy`-only case, now that the plan collapses it: no glue, no match, nothing to run.
    #[test]
    fn an_enum_with_no_drop_obligation_anywhere_emits_no_glue() {
        let mut types = TypeContext::default();
        types
            .enum_variants
            .insert((3, vec![]), vec![vec![MirTy::Int32], vec![]]);
        let body = env_body();
        let layout = crate::layout::TargetLayout::default();
        let env = TyEnv::new(&body, &types, &layout);
        let ty = MirTy::Enum(EnumRef::User(crate::hir::ItemId(3)), vec![]);
        assert_eq!(
            emit_drop_glue(&ty, "__v", &env).expect("glue must emit"),
            "",
            "a plan with no obligation must emit nothing, not an empty match"
        );
    }

    // ---- WP-C5.3d-1b: the plan and its application are one mechanism ----
    //
    // These are the mutation cases CD-062 requires. Each one takes the shared plan, corrupts it in
    // exactly the way a hand-written reconstruction used to get wrong, and shows the corruption
    // reaches the generated Rust. Together they establish that the emitter APPLIES the plan
    // rather than re-deriving it -- if it re-derived, a corrupted plan would leave the output
    // unchanged and every one of these would fail.

    fn plan_fixture() -> (TypeContext, MirTy) {
        let mut types = TypeContext::default();
        types.struct_fields.insert((1, vec![]), vec![MirTy::Int32]);
        types
            .drop_impls
            .insert((1, vec![]), "Inner::drop@[]".to_string());
        let inner = MirTy::Struct(crate::hir::ItemId(1), vec![]);
        types
            .enum_variants
            .insert((2, vec![]), vec![vec![], vec![inner.clone(), inner]]);
        (
            types,
            MirTy::Enum(EnumRef::User(crate::hir::ItemId(2)), vec![]),
        )
    }

    /// Mutation: delete a variant from the plan. The generated `match` must lose its arm, which
    /// is what makes the emitted code stop compiling rather than silently skip a variant.
    #[test]
    fn mutating_the_plan_to_omit_a_variant_removes_its_arm() {
        let (types, ty) = plan_fixture();
        let mut plan = drop_plan::plan_for(&ty, &types).unwrap();
        let DropPlan::Variants { variants, .. } = &mut plan else {
            panic!("expected a variant plan");
        };
        variants.pop();
        let glue = emit_drop_plan(&plan, "__v").unwrap();
        assert!(
            !glue.contains("::V1("),
            "the omitted variant must not survive into the output: {glue}"
        );
    }

    /// Mutation: delete a payload field from a variant. Its destructor call must disappear.
    #[test]
    fn mutating_the_plan_to_omit_a_payload_field_removes_its_destructor_call() {
        let (types, ty) = plan_fixture();
        let mut plan = drop_plan::plan_for(&ty, &types).unwrap();
        let DropPlan::Variants { variants, .. } = &mut plan else {
            panic!("expected a variant plan");
        };
        variants[1].fields.pop();
        let glue = emit_drop_plan(&plan, "__v").unwrap();
        assert_eq!(
            glue.matches(&mangle::function_name_for_symbol("Inner::drop@[]"))
                .count(),
            1,
            "one of the two payload destructor calls must be gone: {glue}"
        );
    }

    /// Mutation: reverse the plan's field order. The emitted order must follow it, which proves
    /// the emitter is not imposing an order of its own.
    #[test]
    fn mutating_the_plan_field_order_changes_the_emitted_order() {
        let (types, ty) = plan_fixture();
        let mut plan = drop_plan::plan_for(&ty, &types).unwrap();
        let DropPlan::Variants { variants, .. } = &mut plan else {
            panic!("expected a variant plan");
        };
        variants[1].fields.reverse();
        let glue = emit_drop_plan(&plan, "__v").unwrap();
        let arm = glue
            .split_once("__p1) => {")
            .expect("the payload arm must bind both fields")
            .1;
        assert!(
            arm.find("__p0").unwrap() < arm.find("__p1").unwrap(),
            "the emitted order must follow the mutated plan: {glue}"
        );
    }

    /// Mutation: try to make the components run BEFORE the type's own destructor.
    ///
    /// The plan's nesting makes that unrepresentable -- `Destructor { then }` contains its
    /// components, so there is no rearrangement that sequences them ahead of it *at the same
    /// value*. The nearest thing the shape allows is to push the destructor down onto a
    /// component, and the assertion below is that doing so is not a quiet reordering: the
    /// destructor is then applied to the FIELD, so the generated crate fails to compile
    /// (`Outer::drop` receiving `&mut Inner`). A misordering here cannot reach a running program.
    #[test]
    fn the_plan_shape_makes_fields_before_the_destructor_unrepresentable() {
        let mut types = TypeContext::default();
        types.struct_fields.insert((1, vec![]), vec![MirTy::Int32]);
        types
            .drop_impls
            .insert((1, vec![]), "Inner::drop@[]".to_string());
        types.struct_fields.insert(
            (2, vec![]),
            vec![MirTy::Struct(crate::hir::ItemId(1), vec![])],
        );
        types
            .drop_impls
            .insert((2, vec![]), "Outer::drop@[]".to_string());
        let ty = MirTy::Struct(crate::hir::ItemId(2), vec![]);

        let plan = drop_plan::plan_for(&ty, &types).unwrap();
        let outer = mangle::function_name_for_symbol("Outer::drop@[]");
        let inner = mangle::function_name_for_symbol("Inner::drop@[]");
        let glue = emit_drop_plan(&plan, "__v").unwrap();
        assert!(
            glue.find(&outer).unwrap() < glue.find(&inner).unwrap(),
            "the canonical plan runs the type's own destructor first: {glue}"
        );

        // The only rearrangement the shape permits: push the destructor down onto the field.
        let DropPlan::Destructor { symbol, then } = plan else {
            panic!("expected a destructor at the root");
        };
        let pushed_down = match *then {
            DropPlan::Fields { base, mut fields } => {
                fields[0].plan = DropPlan::Destructor {
                    symbol,
                    then: Box::new(fields[0].plan.clone()),
                };
                DropPlan::Fields { base, fields }
            }
            other => panic!("expected fields under the destructor, got {other:?}"),
        };
        let glue = emit_drop_plan(&pushed_down, "__v").unwrap();
        assert!(
            glue.contains(&format!("{outer}((&mut __v.f0))")),
            "the displaced destructor must land on the FIELD, not the value -- which is a type \
             error in the generated crate rather than a silent reordering: {glue}"
        );
    }

    /// Mutation: put a `Copy` component back into the plan. A destructor call appears for it --
    /// so "do not drop a Copy field" is enforced by the derivation, in one place, and not by a
    /// filter each consumer has to remember.
    #[test]
    fn mutating_the_plan_to_include_a_copy_field_emits_a_drop_for_it() {
        let mut types = TypeContext::default();
        types.struct_fields.insert((1, vec![]), vec![MirTy::Int32]);
        types
            .drop_impls
            .insert((1, vec![]), "Inner::drop@[]".to_string());
        types.struct_fields.insert(
            (2, vec![]),
            vec![MirTy::Int32, MirTy::Struct(crate::hir::ItemId(1), vec![])],
        );
        let ty = MirTy::Struct(crate::hir::ItemId(2), vec![]);

        let mut plan = drop_plan::plan_for(&ty, &types).unwrap();
        let DropPlan::Fields { fields, .. } = &mut plan else {
            panic!("expected a field plan");
        };
        assert_eq!(
            fields.len(),
            1,
            "the Copy field must be absent to begin with"
        );
        fields.push(crate::mir::drop_plan::PlannedField {
            index: 0,
            plan: DropPlan::Destructor {
                symbol: "Inner::drop@[]".to_string(),
                then: Box::new(DropPlan::Noop),
            },
        });
        let glue = emit_drop_plan(&plan, "__v").unwrap();
        assert!(
            glue.contains("__v.f0"),
            "a plan that names the Copy field makes the emitter drop it: {glue}"
        );
    }

    /// Tuples and arrays reach the emitter only through the plan; before it, `emit_drop_glue`
    /// refused everything that was not a struct or a user enum.
    #[test]
    fn tuple_and_array_components_get_their_rust_spellings_from_the_plan() {
        let mut types = TypeContext::default();
        types.struct_fields.insert((1, vec![]), vec![MirTy::Int32]);
        types
            .drop_impls
            .insert((1, vec![]), "Inner::drop@[]".to_string());
        let inner = MirTy::Struct(crate::hir::ItemId(1), vec![]);
        let body = env_body();
        let layout = crate::layout::TargetLayout::default();
        let env = TyEnv::new(&body, &types, &layout);

        let tuple = MirTy::Tuple(vec![MirTy::Int32, inner.clone()]);
        let glue = emit_drop_glue(&tuple, "__v", &env).expect("tuple glue must emit");
        assert!(
            glue.contains("(&mut __v.1)"),
            "a tuple component is `.1`, not `.f1`: {glue}"
        );

        let array = MirTy::Array(Box::new(inner), 3);
        let glue = emit_drop_glue(&array, "__v", &env).expect("array glue must emit");
        let at = |i: usize| {
            glue.find(&format!("__v[{i}]"))
                .expect("every element drops")
        };
        assert!(
            at(2) < at(1) && at(1) < at(0),
            "array elements destroy back to front: {glue}"
        );
    }
}
