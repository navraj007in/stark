//! WP-C4.4 — MIR interpreter.
//!
//! Executes a **verified** `MirProgram` for differential validation against the HIR
//! interpreter (the semantic oracle, charter §1.6 rule 6). This is NOT a user-facing VM
//! (charter §1.6 rule 11) — its sole purpose is the contract's observable comparator:
//! for each frozen workload, `HIR interpreter output/failure == MIR interpreter
//! output/failure`.
//!
//! Design points:
//! - Executes only what the verifier accepted; internal inconsistencies (which verified MIR
//!   cannot exhibit) surface as loud `InternalError`s, never silent misbehavior.
//! - `Checked` terminators implement STARK's trap semantics per integer width (overflow,
//!   divide-by-zero incl. `MIN / -1`, float div/rem by zero per CD-006, checked numeric
//!   casts). A trap ABORTS execution with its `TrapCategory` — no unwinding, matching the
//!   abstract machine.
//! - `Move` operands *take* the value (leaving a poisoned slot) so any verifier-missed
//!   use-after-move explodes loudly here instead of silently reading stale data.
//! - Float printing calls `crate::interp::canonical_float` — the oracle's own formatter, one
//!   algorithm for both engines by construction.
//! - A step-limit fuel guard turns lowering bugs that produce infinite loops into a clean
//!   `InternalError` rather than a hung differential harness.

use super::*;
use std::collections::HashMap;
use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub enum MirValue {
    Int(i128),
    Float(f64),
    Bool(bool),
    Unit,
    /// Struct/tuple/array payloads (field order = declaration/element order).
    Aggregate(Vec<MirValue>),
    Enum {
        variant: u32,
        fields: Vec<MirValue>,
    },
    /// Index into `MirProgram::bodies`.
    FnPtr(usize),
}

#[derive(Debug)]
pub enum MirRunError {
    /// A language trap: category AND provenance (review correction: discarding SourceInfo
    /// made the differential blind to wrong-location traps — a right-category trap at the
    /// wrong operand would have passed C4.4).
    Trap {
        category: TrapCategory,
        source: SourceInfo,
    },
    /// A bug in lowering/verification/interpretation — never a language-level outcome.
    Internal(String),
}

pub struct MirExecution {
    pub output: String,
    pub status: u8,
}

const FUEL: u64 = 50_000_000;

pub fn run_program(
    verified: crate::mir::verify::VerifiedMirProgram<'_>,
) -> Result<MirExecution, MirRunError> {
    let program = verified.program();
    let by_symbol: HashMap<&str, usize> = program
        .bodies
        .iter()
        .enumerate()
        .map(|(i, b)| (b.instance.symbol.as_str(), i))
        .collect();
    let Some(&main_index) = by_symbol.get("main@[]") else {
        return Err(MirRunError::Internal("no main@[] instance".to_string()));
    };
    let mut cx = Interp {
        program,
        by_symbol,
        output: String::new(),
        fuel: FUEL,
    };
    cx.call(main_index, Vec::new())?;
    Ok(MirExecution {
        output: cx.output,
        status: 0,
    })
}

struct Interp<'a> {
    program: &'a MirProgram,
    by_symbol: HashMap<&'a str, usize>,
    output: String,
    fuel: u64,
}

impl<'a> Interp<'a> {
    fn internal<T>(&self, message: impl Into<String>) -> Result<T, MirRunError> {
        Err(MirRunError::Internal(message.into()))
    }

    fn call(&mut self, body_index: usize, args: Vec<MirValue>) -> Result<MirValue, MirRunError> {
        let body = &self.program.bodies[body_index];
        let mut locals: Vec<Option<MirValue>> = vec![None; body.locals.len()];
        // Params occupy locals 1..=n by construction (Return is 0).
        for (i, value) in args.into_iter().enumerate() {
            locals[1 + i] = Some(value);
        }
        // The return place starts uninitialized; Unit-returning bodies assign it explicitly.
        let mut block = body.entry;
        loop {
            if self.fuel == 0 {
                return self.internal("fuel exhausted (runaway loop in lowered MIR?)");
            }
            self.fuel -= 1;

            let bb = &body.blocks[block.0 as usize];
            for (stmt, _) in &bb.statements {
                match stmt {
                    Statement::Assign(place, rvalue) => {
                        let value = self.eval_rvalue(body_index, &mut locals, rvalue)?;
                        self.write_place(&mut locals, place, value)?;
                    }
                    Statement::Nop => {}
                }
            }
            match &bb.terminator.0 {
                Terminator::Goto { target } => block = *target,
                Terminator::SwitchInt {
                    scrut,
                    arms,
                    otherwise,
                } => {
                    let value = self.eval_operand(&mut locals, scrut)?;
                    let key: u128 = match value {
                        MirValue::Bool(b) => u128::from(b),
                        MirValue::Int(v) => v as u128, // same wrap as lowering's arm keys
                        other => {
                            return self
                                .internal(format!("SwitchInt on non-integer value {other:?}"))
                        }
                    };
                    block = arms
                        .iter()
                        .find(|(arm, _)| *arm == key)
                        .map(|(_, target)| *target)
                        .unwrap_or(*otherwise);
                }
                Terminator::Call {
                    callee,
                    args,
                    dest,
                    target,
                } => {
                    let mut values = Vec::new();
                    for arg in args {
                        values.push(self.eval_operand(&mut locals, arg)?);
                    }
                    let result = match callee {
                        Callee::Instance(instance) => {
                            let Some(&idx) = self.by_symbol.get(instance.symbol.as_str()) else {
                                return self.internal(format!(
                                    "call to unknown instance {}",
                                    instance.symbol
                                ));
                            };
                            self.call(idx, values)?
                        }
                        Callee::FnValue(op) => {
                            let value = self.eval_operand(&mut locals, op)?;
                            let MirValue::FnPtr(idx) = value else {
                                return self.internal(format!("indirect call through {value:?}"));
                            };
                            self.call(idx, values)?
                        }
                        Callee::Runtime(rt) => self.run_runtime(*rt, values)?,
                    };
                    self.write_place(&mut locals, dest, result)?;
                    block = *target;
                }
                Terminator::Drop { place: _, target } => {
                    // Scalar core never emits Drop; executing one on drop-free types is a nop.
                    // C4.5's elaboration will give this real destructor dispatch.
                    block = *target;
                }
                Terminator::Checked {
                    op,
                    args,
                    dest,
                    target,
                    trap,
                } => {
                    let mut values = Vec::new();
                    for arg in args {
                        values.push(self.eval_operand(&mut locals, arg)?);
                    }
                    let dest_ty = &body.locals[dest.0 as usize].ty;
                    match self.eval_checked(*op, &values, dest_ty)? {
                        Some(value) => {
                            locals[dest.0 as usize] = Some(value);
                            block = *target;
                        }
                        None => {
                            return Err(MirRunError::Trap {
                                category: trap.category,
                                source: trap.source,
                            })
                        }
                    }
                }
                Terminator::Trap { info } => {
                    return Err(MirRunError::Trap {
                        category: info.category,
                        source: info.source,
                    })
                }
                Terminator::Return => {
                    return match locals[0].take() {
                        Some(value) => Ok(value),
                        None => self.internal("Return with uninitialized return place"),
                    };
                }
                Terminator::Unreachable => {
                    return self.internal("reached an Unreachable terminator");
                }
            }
        }
    }

    // ---- values ----

    fn eval_rvalue(
        &mut self,
        body_index: usize,
        locals: &mut [Option<MirValue>],
        rvalue: &Rvalue,
    ) -> Result<MirValue, MirRunError> {
        let _ = body_index;
        Ok(match rvalue {
            Rvalue::Use(op) => self.eval_operand(locals, op)?,
            Rvalue::UnOp(op, operand) => {
                let value = self.eval_operand(locals, operand)?;
                match (op, value) {
                    (MirUnOp::Not, MirValue::Bool(b)) => MirValue::Bool(!b),
                    (MirUnOp::FloatNeg, MirValue::Float(f)) => MirValue::Float(-f),
                    (op, value) => {
                        return self.internal(format!("UnOp {op:?} on {value:?}"));
                    }
                }
            }
            Rvalue::BinOp(op, lhs, rhs) => {
                let l = self.eval_operand(locals, lhs)?;
                let r = self.eval_operand(locals, rhs)?;
                self.eval_binop(*op, l, r)?
            }
            Rvalue::Aggregate(kind, operands) => {
                let mut values = Vec::new();
                for op in operands {
                    values.push(self.eval_operand(locals, op)?);
                }
                match kind {
                    AggKind::EnumVariant(_, variant) => MirValue::Enum {
                        variant: *variant,
                        fields: values,
                    },
                    _ => MirValue::Aggregate(values),
                }
            }
            Rvalue::Discriminant(place) => {
                let value = self.read_place(locals, place)?;
                let MirValue::Enum { variant, .. } = value else {
                    return self.internal(format!("Discriminant of {value:?}"));
                };
                MirValue::Int(i128::from(variant))
            }
            Rvalue::RefOf { .. } => {
                return self.internal("RefOf is not executable in the v0.1 MIR interpreter");
            }
        })
    }

    fn eval_binop(&self, op: MirBinOp, l: MirValue, r: MirValue) -> Result<MirValue, MirRunError> {
        use MirBinOp::*;
        Ok(match (op, l, r) {
            (Eq, l, r) => MirValue::Bool(l == r),
            (Ne, l, r) => MirValue::Bool(l != r),
            (Lt, MirValue::Int(a), MirValue::Int(b)) => MirValue::Bool(a < b),
            (Le, MirValue::Int(a), MirValue::Int(b)) => MirValue::Bool(a <= b),
            (Gt, MirValue::Int(a), MirValue::Int(b)) => MirValue::Bool(a > b),
            (Ge, MirValue::Int(a), MirValue::Int(b)) => MirValue::Bool(a >= b),
            (Lt, MirValue::Float(a), MirValue::Float(b)) => MirValue::Bool(a < b),
            (Le, MirValue::Float(a), MirValue::Float(b)) => MirValue::Bool(a <= b),
            (Gt, MirValue::Float(a), MirValue::Float(b)) => MirValue::Bool(a > b),
            (Ge, MirValue::Float(a), MirValue::Float(b)) => MirValue::Bool(a >= b),
            (FloatAdd, MirValue::Float(a), MirValue::Float(b)) => MirValue::Float(a + b),
            (FloatSub, MirValue::Float(a), MirValue::Float(b)) => MirValue::Float(a - b),
            (FloatMul, MirValue::Float(a), MirValue::Float(b)) => MirValue::Float(a * b),
            (op, l, r) => {
                return self.internal(format!("BinOp {op:?} on {l:?}, {r:?}"));
            }
        })
    }

    /// Checked/trapping primitives. `Ok(None)` = trap.
    fn eval_checked(
        &self,
        op: CheckedOp,
        args: &[MirValue],
        dest_ty: &MirTy,
    ) -> Result<Option<MirValue>, MirRunError> {
        use CheckedOp::*;
        match op {
            Add | Sub | Mul | Div | Rem | Neg => {
                let (min, max) = int_range(dest_ty)
                    .ok_or_else(|| MirRunError::Internal("checked int op on non-int".into()))?;
                let int = |v: &MirValue| -> Result<i128, MirRunError> {
                    match v {
                        MirValue::Int(i) => Ok(*i),
                        other => Err(MirRunError::Internal(format!(
                            "checked op operand {other:?}"
                        ))),
                    }
                };
                let result: Option<i128> = match op {
                    Add => int(&args[0])?.checked_add(int(&args[1])?),
                    Sub => int(&args[0])?.checked_sub(int(&args[1])?),
                    Mul => int(&args[0])?.checked_mul(int(&args[1])?),
                    Div => {
                        let (a, b) = (int(&args[0])?, int(&args[1])?);
                        if b == 0 {
                            None
                        } else {
                            a.checked_div(b)
                        }
                    }
                    Rem => {
                        let (a, b) = (int(&args[0])?, int(&args[1])?);
                        if b == 0 {
                            None
                        } else {
                            a.checked_rem(b)
                        }
                    }
                    Neg => int(&args[0])?.checked_neg(),
                    _ => unreachable!(),
                };
                Ok(result.filter(|v| *v >= min && *v <= max).map(MirValue::Int))
            }
            Shl | Shr => Err(MirRunError::Internal(
                "shifts are not lowered in the scalar core".into(),
            )),
            FloatDiv | FloatRem => {
                let (a, b) = match (&args[0], &args[1]) {
                    (MirValue::Float(a), MirValue::Float(b)) => (*a, *b),
                    other => {
                        return Err(MirRunError::Internal(format!(
                            "checked float op operands {other:?}"
                        )))
                    }
                };
                // CD-006: division/modulo by zero traps for floats too.
                if b == 0.0 {
                    return Ok(None);
                }
                Ok(Some(MirValue::Float(if matches!(op, FloatDiv) {
                    a / b
                } else {
                    a % b
                })))
            }
            Cast => {
                let value = &args[0];
                Ok(match (value, dest_ty) {
                    (MirValue::Int(v), ty) if int_range(ty).is_some() => {
                        let (min, max) = int_range(ty).unwrap();
                        if *v >= min && *v <= max {
                            Some(MirValue::Int(*v))
                        } else {
                            None // CastFailure trap
                        }
                    }
                    (MirValue::Int(v), MirTy::Float32 | MirTy::Float64) => {
                        Some(MirValue::Float(*v as f64))
                    }
                    (MirValue::Float(f), MirTy::Float64) => Some(MirValue::Float(*f)),
                    (MirValue::Float(f), MirTy::Float32) => {
                        Some(MirValue::Float(f64::from(*f as f32)))
                    }
                    (MirValue::Float(f), ty) if int_range(ty).is_some() => {
                        let truncated = f.trunc();
                        let (min, max) = int_range(ty).unwrap();
                        if f.is_nan() || truncated < min as f64 || truncated > max as f64 {
                            None
                        } else {
                            Some(MirValue::Int(truncated as i128))
                        }
                    }
                    (value, ty) => {
                        return Err(MirRunError::Internal(format!("cast {value:?} to {ty:?}")))
                    }
                })
            }
            CheckIndex => Err(MirRunError::Internal(
                "CheckIndex is not lowered in the scalar core".into(),
            )),
        }
    }

    // ---- places ----

    fn eval_operand(
        &mut self,
        locals: &mut [Option<MirValue>],
        op: &Operand,
    ) -> Result<MirValue, MirRunError> {
        match op {
            Operand::Copy(place) => self.read_place(locals, place),
            Operand::Move(place) => {
                if place.projection.is_empty() {
                    match locals[place.local.0 as usize].take() {
                        Some(value) => Ok(value),
                        None => self.internal(format!(
                            "move from uninitialized/moved local _{}",
                            place.local.0
                        )),
                    }
                } else {
                    // Projected move: read the sub-value (the verifier's whole-local
                    // conservatism already guards double-use).
                    self.read_place(locals, place)
                }
            }
            Operand::Const(constant) => Ok(match constant {
                Constant::Int(v, _) => MirValue::Int(*v),
                Constant::Float(f, _) => MirValue::Float(*f),
                Constant::Bool(b) => MirValue::Bool(*b),
                Constant::Unit => MirValue::Unit,
                Constant::FnPtr(instance) => match self.by_symbol.get(instance.symbol.as_str()) {
                    Some(&idx) => MirValue::FnPtr(idx),
                    None => {
                        return self
                            .internal(format!("FnPtr to unknown instance {}", instance.symbol))
                    }
                },
            }),
        }
    }

    fn read_place(
        &self,
        locals: &[Option<MirValue>],
        place: &Place,
    ) -> Result<MirValue, MirRunError> {
        let Some(mut value) = locals[place.local.0 as usize].clone() else {
            return self.internal(format!(
                "read of uninitialized/moved local _{}",
                place.local.0
            ));
        };
        for projection in &place.projection {
            value = match (projection, value) {
                (Projection::Field(i), MirValue::Aggregate(fields)) => {
                    match fields.into_iter().nth(*i as usize) {
                        Some(f) => f,
                        None => return self.internal("field projection out of bounds"),
                    }
                }
                (Projection::VariantField(v, i), MirValue::Enum { variant, fields }) => {
                    if variant != *v {
                        return self.internal("VariantField read from a different active variant");
                    }
                    match fields.into_iter().nth(*i as usize) {
                        Some(f) => f,
                        None => return self.internal("variant field out of bounds"),
                    }
                }
                (projection, value) => {
                    return self.internal(format!("projection {projection:?} on value {value:?}"));
                }
            };
        }
        Ok(value)
    }

    fn write_place(
        &self,
        locals: &mut [Option<MirValue>],
        place: &Place,
        value: MirValue,
    ) -> Result<(), MirRunError> {
        if place.projection.is_empty() {
            locals[place.local.0 as usize] = Some(value);
            return Ok(());
        }
        let slot = locals
            .get_mut(place.local.0 as usize)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| {
                MirRunError::Internal(format!(
                    "write through uninitialized local _{}",
                    place.local.0
                ))
            })?;
        let mut target = slot;
        for projection in &place.projection {
            target = match (projection, target) {
                (Projection::Field(i), MirValue::Aggregate(fields)) => fields
                    .get_mut(*i as usize)
                    .ok_or_else(|| MirRunError::Internal("field write out of bounds".into()))?,
                (Projection::VariantField(v, i), MirValue::Enum { variant, fields }) => {
                    if *variant != *v {
                        return Err(MirRunError::Internal(
                            "VariantField write to a different active variant".into(),
                        ));
                    }
                    fields
                        .get_mut(*i as usize)
                        .ok_or_else(|| MirRunError::Internal("variant write oob".into()))?
                }
                (projection, target) => {
                    return Err(MirRunError::Internal(format!(
                        "write projection {projection:?} on {target:?}"
                    )));
                }
            };
        }
        *target = value;
        Ok(())
    }

    // ---- runtime surface ----

    fn run_runtime(&mut self, rt: RuntimeFn, args: Vec<MirValue>) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        let arg = args.into_iter().next();
        match (rt, arg) {
            (PrintlnInt64 | PrintlnUInt64, Some(MirValue::Int(v))) => {
                if matches!(rt, PrintlnUInt64) {
                    let _ = writeln!(self.output, "{}", v as u128);
                } else {
                    let _ = writeln!(self.output, "{v}");
                }
                Ok(MirValue::Unit)
            }
            (PrintInt64 | PrintUInt64, Some(MirValue::Int(v))) => {
                if matches!(rt, PrintUInt64) {
                    let _ = write!(self.output, "{}", v as u128);
                } else {
                    let _ = write!(self.output, "{v}");
                }
                Ok(MirValue::Unit)
            }
            (PrintlnBool, Some(MirValue::Bool(b))) => {
                let _ = writeln!(self.output, "{b}");
                Ok(MirValue::Unit)
            }
            (PrintBool, Some(MirValue::Bool(b))) => {
                let _ = write!(self.output, "{b}");
                Ok(MirValue::Unit)
            }
            (PrintlnFloat64, Some(MirValue::Float(f))) => {
                // The oracle's own formatter — identical output by construction.
                let _ = writeln!(self.output, "{}", crate::interp::canonical_float(f));
                Ok(MirValue::Unit)
            }
            (PrintFloat64, Some(MirValue::Float(f))) => {
                let _ = write!(self.output, "{}", crate::interp::canonical_float(f));
                Ok(MirValue::Unit)
            }
            (rt, arg) => self.internal(format!("runtime {rt:?} with argument {arg:?}")),
        }
    }
}

fn int_range(ty: &MirTy) -> Option<(i128, i128)> {
    Some(match ty {
        MirTy::Int8 => (i128::from(i8::MIN), i128::from(i8::MAX)),
        MirTy::Int16 => (i128::from(i16::MIN), i128::from(i16::MAX)),
        MirTy::Int32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        MirTy::Int64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
        MirTy::UInt8 => (0, i128::from(u8::MAX)),
        MirTy::UInt16 => (0, i128::from(u16::MAX)),
        MirTy::UInt32 => (0, i128::from(u32::MAX)),
        MirTy::UInt64 => (0, i128::from(u64::MAX)),
        _ => return None,
    })
}
