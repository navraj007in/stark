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
//! specific sub-WP), and indirect (`Callee::FnValue`) / runtime (`Callee::Runtime`) calls and
//! `Drop`/`Trap` terminators (WP-C5.4c / WP-C5.2e / wherever `RuntimeFn` support lands) remain
//! `Unsupported`.

use super::{emit_places, emit_types, mangle, BackendDiagnostic};
use crate::mir::{
    Callee, CheckedOp, Constant, LocalKind, MirBinOp, MirBody, MirTy, MirUnOp, Operand, Rvalue,
    Statement, Terminator, TrapCategory,
};

/// A complete `fn name(params) -> ret { ... }` for an ordinary (non-entry) function. The entry
/// instance is emitted separately, by `emit_program.rs`, as Rust's literal `fn main()` with the
/// version-check prologue prepended -- `emit_block_body` is the shared piece both use.
pub fn emit_function(body: &MirBody, name: &str) -> Result<String, BackendDiagnostic> {
    let params = emit_param_list(body)?;
    let ret_ty = emit_types::emit_ty(&body.ret)?;
    let block = emit_block_body(body)?;
    Ok(format!("fn {name}({params}) -> {ret_ty} {block}"))
}

fn emit_param_list(body: &MirBody) -> Result<String, BackendDiagnostic> {
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
        parts.push(format!("mut {}: {rust_ty}", emit_places::local_name(local)));
    }
    Ok(parts.join(", "))
}

/// Emits a function body for the WP-C5.2d-supported shape: every local `Return`/`Param`/`User`/
/// `Temp`-kinded, every statement a `Nop` or an assignment whose `Rvalue` is `Use`/`UnOp`/
/// `BinOp`/`LayoutQuery`, and every terminator `Goto`/`SwitchInt`/`Checked`/`Call` (direct only)/
/// `Return`/`Unreachable`.
pub fn emit_block_body(body: &MirBody) -> Result<String, BackendDiagnostic> {
    let mut out = String::from("{\n");

    // Every non-parameter local is declared `let mut`, DEFAULT-INITIALISED, up front (WP-C5.2c's
    // record explains why default-initialisation, not left uninitialised, is required once a
    // body has more than one block). `Param`-kinded locals are NOT re-declared here: they are
    // already bound by the Rust function signature (`emit_param_list`), under the exact same
    // `_N` name (`emit_places::local_name`); declaring them again here would shadow the actual
    // argument with a fabricated default, so every parameter would read as its default value
    // rather than the value passed in.
    for (i, decl) in body.locals.iter().enumerate() {
        match &decl.kind {
            LocalKind::Return | LocalKind::User(_) | LocalKind::Temp => {}
            LocalKind::Param(_) => continue,
            other => {
                return Err(BackendDiagnostic::Unsupported(format!(
                    "WP-C5.2d supports only Return/Param/User/Temp locals; {other:?} lands \
                     alongside indexing/Drop lowering (WP-C5.3)"
                )))
            }
        }
        let ty = emit_types::emit_ty(&decl.ty)?;
        let default = emit_types::default_value_expr(&decl.ty)?;
        out.push_str(&format!(
            "    let mut {}: {ty} = {default};\n",
            emit_places::local_name(i as u32)
        ));
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
                    let dest = emit_places::emit_place(place)?;
                    let value = emit_rvalue(rvalue)?;
                    out.push_str(&format!("                {dest} = {value};\n"));
                }
            }
        }
        emit_terminator(body, &mut out, &block.terminator.0)?;
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
    out: &mut String,
    terminator: &Terminator,
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
            let key = switch_key_expr(body, scrut)?;
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
            let dest_place = emit_places::local_name(dest.0);
            let dest_ty = &body.locals[dest.0 as usize].ty;
            let expr = emit_checked_expr(body, *op, args, dest_ty, trap.category)?;
            out.push_str(&format!("                {dest_place} = {expr};\n"));
            out.push_str(&format!("                __bb = {}; continue;\n", target.0));
        }
        Terminator::Call {
            callee,
            args,
            dest,
            target,
        } => {
            let dest_place = emit_places::emit_place(dest)?;
            let call_expr = emit_call(callee, args)?;
            out.push_str(&format!("                {dest_place} = {call_expr};\n"));
            out.push_str(&format!("                __bb = {}; continue;\n", target.0));
        }
        Terminator::Return => {
            out.push_str(&format!(
                "                break {};\n",
                emit_places::local_name(0)
            ));
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
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "Terminator {other:?} has no WP-C5.2d representation yet -- Drop/Trap land in \
                 WP-C5.2e/C5.3"
            )))
        }
    }
    Ok(())
}

/// Direct calls only (`Callee::Instance`). Indirect calls through a function value
/// (`Callee::FnValue`) are WP-C5.4c's job (function values as first-class values need the same
/// representation decisions as everywhere else they appear, not a special case here); runtime
/// calls (`Callee::Runtime`) land alongside whichever `RuntimeFn` group first gets lowered.
fn emit_call(callee: &Callee, args: &[Operand]) -> Result<String, BackendDiagnostic> {
    match callee {
        Callee::Instance(instance) => {
            let name = mangle::function_name_for_symbol(&instance.symbol);
            let mut arg_exprs = Vec::with_capacity(args.len());
            for arg in args {
                arg_exprs.push(emit_operand(arg)?);
            }
            Ok(format!("{name}({})", arg_exprs.join(", ")))
        }
        other => Err(BackendDiagnostic::Unsupported(format!(
            "Callee {other:?} has no WP-C5.2d representation yet -- indirect calls land in \
             WP-C5.4c, runtime calls land alongside their RuntimeFn support"
        ))),
    }
}

fn emit_rvalue(rvalue: &Rvalue) -> Result<String, BackendDiagnostic> {
    match rvalue {
        Rvalue::Use(operand) => emit_operand(operand),
        Rvalue::UnOp(MirUnOp::Not, operand) => Ok(format!("(!({}))", emit_operand(operand)?)),
        Rvalue::UnOp(MirUnOp::FloatNeg, operand) => Ok(format!("(-({}))", emit_operand(operand)?)),
        Rvalue::BinOp(op, lhs, rhs) => emit_binop(*op, emit_operand(lhs)?, emit_operand(rhs)?),
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
fn emit_operand(operand: &Operand) -> Result<String, BackendDiagnostic> {
    match operand {
        Operand::Const(c) => emit_types::emit_constant(c),
        Operand::Copy(place) | Operand::Move(place) => emit_places::emit_place(place),
    }
}

/// Resolves an operand's MIR type -- needed for `SwitchInt`'s bit-pattern key and `Cast`'s
/// source type, neither of which `Operand` itself carries.
fn operand_mir_ty(body: &MirBody, operand: &Operand) -> Result<MirTy, BackendDiagnostic> {
    match operand {
        Operand::Const(Constant::Bool(_)) => Ok(MirTy::Bool),
        Operand::Const(Constant::Int(_, ty)) => Ok(ty.clone()),
        Operand::Const(Constant::Float(_, ty)) => Ok(ty.clone()),
        Operand::Const(Constant::Unit) => Ok(MirTy::Unit),
        Operand::Copy(place) | Operand::Move(place) => {
            if !place.projection.is_empty() {
                return Err(BackendDiagnostic::Unsupported(
                    "WP-C5.2c supports only bare-local operand type lookups; projections land \
                     in WP-C5.2c's place work / WP-C5.3"
                        .into(),
                ));
            }
            Ok(body.locals[place.local.0 as usize].ty.clone())
        }
        other => Err(BackendDiagnostic::Unsupported(format!(
            "operand type lookup for {other:?} has no WP-C5.2c representation yet"
        ))),
    }
}

/// `SwitchInt`'s arm keys are `u128`, computed by `mir::interp::run` (the oracle this must
/// match) as: `Bool -> 0/1`, `Int (any width, including Char's codepoint encoding) -> the value
/// reinterpreted as `i128` then `u128`` (`v as u128` on the interpreter's `i128`-carrier value
/// -- i.e. sign-extension THEN bit-reinterpretation, not a same-width truncation). Reproduced
/// here by casting through `i128` explicitly so the same sign-extension happens regardless of
/// the Rust-side value's actual declared width.
fn switch_key_expr(body: &MirBody, operand: &Operand) -> Result<String, BackendDiagnostic> {
    let ty = operand_mir_ty(body, operand)?;
    let value = emit_operand(operand)?;
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
    body: &MirBody,
    op: CheckedOp,
    args: &[Operand],
    dest_ty: &MirTy,
    default_category: TrapCategory,
) -> Result<String, BackendDiagnostic> {
    use CheckedOp::*;
    let default_trap = trap_category_token(default_category);
    match op {
        Add | Sub | Mul | Div | Rem | Neg | Pow => {
            let dest_rust_ty = emit_types::emit_ty(dest_ty)?;
            let (min, max) = int_bounds_tokens(dest_ty)?;
            let a = emit_operand(&args[0])?;
            let checked = match op {
                Add => format!(
                    "(({a}) as i128).checked_add(({}) as i128)",
                    emit_operand(&args[1])?
                ),
                Sub => format!(
                    "(({a}) as i128).checked_sub(({}) as i128)",
                    emit_operand(&args[1])?
                ),
                Mul => format!(
                    "(({a}) as i128).checked_mul(({}) as i128)",
                    emit_operand(&args[1])?
                ),
                Div => format!(
                    "(({a}) as i128).checked_div(({}) as i128)",
                    emit_operand(&args[1])?
                ),
                Rem => format!(
                    "(({a}) as i128).checked_rem(({}) as i128)",
                    emit_operand(&args[1])?
                ),
                Neg => format!("(({a}) as i128).checked_neg()"),
                // A5: exponent must be nonnegative and fit in u32 (NUM-INT-ARITH-001);
                // `u32::try_from` rejects both a negative exponent and one wider than u32.
                Pow => format!(
                    "u32::try_from(({}) as i128).ok().and_then(|__e| (({a}) as i128).checked_pow(__e))",
                    emit_operand(&args[1])?
                ),
                _ => unreachable!(),
            };
            Ok(format!(
                "match {checked} {{ Some(__v) if __v >= {min} && __v <= {max} => __v as \
                 {dest_rust_ty}, _ => stark_runtime::trap::abort_minimal({default_trap}) }}"
            ))
        }
        Shl | Shr => {
            let dest_rust_ty = emit_types::emit_ty(dest_ty)?;
            let (min, max) = int_bounds_tokens(dest_ty)?;
            let width = int_width_tokens(dest_ty)?;
            let l = emit_operand(&args[0])?;
            let count = emit_operand(&args[1])?;
            let method = if matches!(op, Shl) {
                "checked_shl"
            } else {
                "checked_shr"
            };
            let invalid_shift = trap_category_token(TrapCategory::InvalidShift);
            Ok(format!(
                "{{ let __count = ({count}) as i128; if __count < 0 || __count >= {width} {{ \
                 stark_runtime::trap::abort_minimal({invalid_shift}) }} else {{ match (({l}) as \
                 i128).{method}(__count as u32) {{ Some(__v) if __v >= {min} && __v <= {max} => \
                 __v as {dest_rust_ty}, _ => stark_runtime::trap::abort_minimal({default_trap}) \
                 }} }} }}"
            ))
        }
        FloatDiv | FloatRem => {
            let dest_rust_ty = emit_types::emit_ty(dest_ty)?;
            let l = emit_operand(&args[0])?;
            let r = emit_operand(&args[1])?;
            let rust_op = if matches!(op, FloatDiv) { "/" } else { "%" };
            Ok(format!(
                "{{ let __l: {dest_rust_ty} = {l}; let __r: {dest_rust_ty} = {r}; if __r == 0.0 \
                 {{ stark_runtime::trap::abort_minimal({default_trap}) }} else {{ __l {rust_op} \
                 __r }} }}"
            ))
        }
        Cast => emit_cast_expr(body, &args[0], dest_ty, &default_trap),
        other => Err(BackendDiagnostic::Unsupported(format!(
            "CheckedOp {other:?} has no WP-C5.2c representation yet -- CheckIndex lands \
             alongside array/slice indexing (WP-C5.3)"
        ))),
    }
}

fn emit_cast_expr(
    body: &MirBody,
    source: &Operand,
    dest_ty: &MirTy,
    default_trap: &str,
) -> Result<String, BackendDiagnostic> {
    let src_ty = operand_mir_ty(body, source)?;
    let a = emit_operand(source)?;
    let dest_rust_ty = emit_types::emit_ty(dest_ty)?;
    let src_is_int = int_bounds_tokens(&src_ty).is_ok();
    let dest_is_int = int_bounds_tokens(dest_ty).is_ok();
    let src_is_float = matches!(src_ty, MirTy::Float32 | MirTy::Float64);
    let dest_is_float = matches!(dest_ty, MirTy::Float32 | MirTy::Float64);

    if src_is_int && dest_is_int {
        let (min, max) = int_bounds_tokens(dest_ty)?;
        return Ok(format!(
            "{{ let __v = ({a}) as i128; if __v >= {min} && __v <= {max} {{ __v as \
             {dest_rust_ty} }} else {{ stark_runtime::trap::abort_minimal({default_trap}) }} }}"
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
        let (min, max) = int_bounds_tokens(dest_ty)?;
        return Ok(format!(
            "{{ let __f = ({a}) as f64; let __t = __f.trunc(); if __f.is_nan() || __t < ({min} \
             as f64) || __t > ({max} as f64) {{ stark_runtime::trap::abort_minimal({default_trap}) \
             }} else {{ __t as {dest_rust_ty} }} }}"
        ));
    }
    Err(BackendDiagnostic::Unsupported(format!(
        "cast {src_ty:?} -> {dest_ty:?} has no WP-C5.2c representation yet"
    )))
}
