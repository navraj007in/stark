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
use super::{emit_places, emit_types, mangle, BackendDiagnostic};
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
) -> Result<String, BackendDiagnostic> {
    let params = emit_param_list(body, types)?;
    let ret_ty = emit_types::emit_ty(&body.ret)?;
    let block = emit_block_body(body, files, types)?;
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
        let name = if emit_types::mir_ty_is_copy(decl_ty, types) {
            emit_places::local_name(local)
        } else {
            incoming_param_name(local)
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
) -> Result<String, BackendDiagnostic> {
    let env = &TyEnv::new(body, types);
    let mut out = String::from("{\n");

    // Every non-parameter local is declared `let mut`, DEFAULT-INITIALISED, up front (WP-C5.2c's
    // record explains why default-initialisation, not left uninitialised, is required once a
    // body has more than one block). `Param`-kinded locals are NOT re-declared here: they are
    // already bound by the Rust function signature (`emit_param_list`), under the exact same
    // `_N` name (`emit_places::local_name`); declaring them again here would shadow the actual
    // argument with a fabricated default, so every parameter would read as its default value
    // rather than the value passed in.
    for (i, decl) in body.locals.iter().enumerate() {
        let is_copy = emit_types::mir_ty_is_copy(&decl.ty, env.types);
        match &decl.kind {
            // `IndexProof` (WP-C5.3a) is an ordinary integer local in generated Rust: MIR
            // keeps it opaque so only `Projection::Index` can consume it, but that opacity is a
            // MIR-level property the verifier enforces, not something the backend re-creates.
            LocalKind::Return | LocalKind::User(_) | LocalKind::Temp | LocalKind::IndexProof => {}
            // A Copy parameter is already bound by the signature under its `_N` name. A non-Copy
            // one is bound as `__pN` and moved into its slot below.
            LocalKind::Param(_) if is_copy => continue,
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
            }
            other => {
                return Err(BackendDiagnostic::Unsupported(format!(
                    "WP-C5.3a supports Return/Param/User/Temp/IndexProof locals; {other:?} \
                     lands alongside Drop elaboration (WP-C5.3d)"
                )))
            }
        }
        // WP-C5.3d-0: a non-Copy local is backed by a `ValueSlot`, which starts DEAD -- so
        // generated code stops fabricating a default value it would immediately overwrite, and
        // MIR liveness is represented explicitly instead of being approximated by Rust's.
        if is_copy {
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

/// Direct calls only (`Callee::Instance`). Indirect calls through a function value
/// (`Callee::FnValue`) are WP-C5.4c's job (function values as first-class values need the same
/// representation decisions as everywhere else they appear, not a special case here); runtime
/// calls (`Callee::Runtime`) land alongside whichever `RuntimeFn` group first gets lowered.
fn emit_call(callee: &Callee, args: &[Operand], env: &TyEnv) -> Result<String, BackendDiagnostic> {
    match callee {
        Callee::Instance(instance) => {
            let name = mangle::function_name_for_symbol(&instance.symbol);
            let mut arg_exprs = Vec::with_capacity(args.len());
            for arg in args {
                arg_exprs.push(emit_operand(arg, env)?);
            }
            Ok(format!("{name}({})", arg_exprs.join(", ")))
        }
        other => Err(BackendDiagnostic::Unsupported(format!(
            "Callee {other:?} has no WP-C5.2d representation yet -- indirect calls land in \
             WP-C5.4c, runtime calls land alongside their RuntimeFn support"
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
/// The glue is built from the type context, not reconstructed from source: the destructor
/// instance named by `drop_impls`, then fields in reverse declaration order — the same order
/// `mir::interp::drop_in_place` uses, because that function is the semantic authority.
fn emit_drop(place: &crate::mir::Place, env: &TyEnv) -> Result<String, BackendDiagnostic> {
    if !place.projection.is_empty() {
        return Err(BackendDiagnostic::Unsupported(format!(
            "dropping a SUB-PLACE ({place:?}) needs per-drop-unit liveness in the emitted body; \
             WP-C5.3d-1. Collapsing it into a whole-local drop would violate §7.6, so it is \
             refused rather than approximated"
        )));
    }
    let ty = env.place_ty(place)?;
    if emit_types::mir_ty_is_copy(&ty, env.types) {
        return Err(BackendDiagnostic::Unsupported(format!(
            "MIR emitted Drop on the Copy type {ty:?}, which has no destructor and no slot"
        )));
    }
    let glue = emit_drop_glue(&ty, "__v", env)?;
    Ok(format!(
        "{}.drop_with(|__v| {{ {glue} }});",
        emit_places::local_name(place.local.0)
    ))
}

/// The destructor sequence for `ty`, applied to the Rust expression `value` (an `&mut`).
fn emit_drop_glue(ty: &MirTy, value: &str, env: &TyEnv) -> Result<String, BackendDiagnostic> {
    let mut out = String::new();
    let (item, args) = match ty {
        MirTy::Struct(item, args) => (item.0, args.clone()),
        MirTy::Enum(crate::mir::EnumRef::User(item), args) => (item.0, args.clone()),
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "drop glue for {other:?} is not in the WP-C5.3d-0 subset"
            )))
        }
    };
    // The type's own destructor first, taking `&mut self` -- the same receiver shape
    // `mir::interp` passes.
    if let Some(symbol) = env.types.drop_impls.get(&(item, args.clone())) {
        let name = mangle::function_name_for_symbol(symbol);
        out.push_str(&format!("{name}({value}); "));
    }
    // Then fields, in REVERSE declaration order. Only non-Copy fields have glue; a Copy field
    // has no destructor by construction (Copy and Drop are mutually exclusive in STARK).
    if let Some(fields) = env.types.struct_fields.get(&(item, args.clone())) {
        for (i, field_ty) in fields.iter().enumerate().rev() {
            if emit_types::mir_ty_is_copy(field_ty, env.types) {
                continue;
            }
            let field = format!("(&mut {value}.{})", emit_types::field_name(i as u32));
            out.push_str(&emit_drop_glue(field_ty, &field, env)?);
        }
    }
    Ok(out)
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
        Rvalue::Discriminant(place) => {
            let base = emit_places::emit_place(place, env)?;
            let ty = env.place_ty(place)?;
            let MirTy::Enum(crate::mir::EnumRef::User(item), args) = &ty else {
                return Err(BackendDiagnostic::Unsupported(format!(
                    "Discriminant of {ty:?} lands in WP-C5.3c (Option/Result/Ordering)"
                )));
            };
            let variants = env
                .types
                .enum_variants
                .get(&(item.0, args.clone()))
                .ok_or_else(|| {
                    BackendDiagnostic::Unsupported(format!(
                        "no type-context entry for enum instance {ty:?}"
                    ))
                })?;
            let name = mangle::type_name_for_nominal(item.0, args);
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
                AggKind::EnumVariant(enum_ref, variant) => {
                    let crate::mir::EnumRef::User(item) = enum_ref else {
                        return Err(BackendDiagnostic::Unsupported(format!(
                            "aggregate for {enum_ref:?} lands in WP-C5.3c (Option/Result/Ordering)"
                        )));
                    };
                    // Like the struct case, the type ARGUMENTS come from the destination:
                    // `AggKind::EnumVariant` names the enum and the variant, not the instance.
                    let MirTy::Enum(_, args) = dest_ty else {
                        return Err(BackendDiagnostic::Unsupported(format!(
                            "enum aggregate assigned to a non-enum destination {dest_ty:?}"
                        )));
                    };
                    let name = mangle::type_name_for_nominal(item.0, args);
                    format!(
                        "{name}::{}({})",
                        emit_types::variant_name(*variant),
                        parts.join(", ")
                    )
                }
            })
        }
        Rvalue::LayoutQuery { kind, ty } => {
            let rust_ty = emit_types::emit_ty(ty)?;
            Ok(match kind {
                crate::mir::LayoutKind::SizeOf => {
                    format!("(core::mem::size_of::<{rust_ty}>() as u64)")
                }
                crate::mir::LayoutKind::AlignOf => {
                    format!("(core::mem::align_of::<{rust_ty}>() as u64)")
                }
            })
        }
        other => Err(BackendDiagnostic::Unsupported(format!(
            "Rvalue {other:?} has no WP-C5.2c representation yet -- Aggregate/Discriminant land \
             in WP-C5.3, RefOf is not yet scheduled to a specific sub-WP"
        ))),
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
        Operand::Copy(place) => emit_places::emit_place(place, env),
        // WP-C5.3d-0. A move out of a Copy place is a read like any other -- MIR only marks it
        // `Move` for uniformity, and Rust copies. A move out of a NON-Copy place must go through
        // the slot, which records that the source unit is now dead; that record is the whole
        // reason the block-dispatch loop no longer defeats the borrow checker.
        Operand::Move(place) => {
            let ty = env.place_ty(place)?;
            if emit_types::mir_ty_is_copy(&ty, env.types) {
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
