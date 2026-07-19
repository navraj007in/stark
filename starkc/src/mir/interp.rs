//! WP-C4.4/C4.5b-2 — MIR interpreter.
//!
//! Executes a **verified** `MirProgram` for differential validation against the HIR
//! interpreter (the semantic oracle, charter §1.6 rule 6). This is NOT a user-facing VM
//! (charter §1.6 rule 11) — its sole purpose is the contract's observable comparator:
//! for each frozen workload, `HIR interpreter output/failure == MIR interpreter
//! output/failure`.
//!
//! Execution model (C4.5b-2 frame restructure):
//! - An explicit frame stack (`frames: Vec<Frame>`) replaces call-local storage, so a
//!   **reference value can point into a caller's frame** — the semantic requirement behind
//!   `&mut self` and `&x` arguments. A reference is a resolved path
//!   `(frame index, local, concrete projections)`; `Deref` re-anchors place resolution to the
//!   referent. Index proofs always resolve in the frame *evaluating* the place, before any
//!   re-anchoring.
//! - Borrow checking upstream guarantees no reference outlives its frame in legal programs;
//!   the interpreter still guards every frame access and reports a loud `Internal` error on a
//!   dangling path (defense in depth, never silent misbehavior).
//! - `Checked` terminators implement STARK trap semantics per integer width (overflow,
//!   divide-by-zero incl. `MIN / -1`, float div/rem by zero per CD-006, checked casts,
//!   CheckIndex bounds). A trap ABORTS with category AND provenance (CD-029) — no unwinding.
//! - `Move` operands *take* whole-local values (poisoning the slot) so verifier-missed
//!   use-after-move explodes loudly; projected/deref reads copy.
//! - Float printing calls `crate::interp::canonical_float` — the oracle's own formatter
//!   (shared by design; compensating spec tests live in `tests/canonical_float.rs`).
//! - A fuel guard turns runaway loops from lowering bugs into clean internal errors.

use super::*;
use std::collections::HashMap;
use std::fmt::Write as _;

/// A fully resolved projection step (no `Deref`, no proof locals — both are resolved away
/// during place resolution).
#[derive(Clone, Debug, PartialEq)]
pub enum ConcreteProj {
    Field(u32),
    Variant(u32, u32),
    Index(usize),
}

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
    /// A reference: a resolved path into some live frame (C4.5b-2).
    Ref {
        frame: usize,
        local: u32,
        path: Vec<ConcreteProj>,
    },
}

#[derive(Debug)]
pub enum MirRunError {
    /// A language trap: category AND provenance (CD-029 — category alone made the
    /// differential blind to wrong-location traps).
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

struct Frame {
    locals: Vec<Option<MirValue>>,
}

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
        frames: Vec::new(),
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
    frames: Vec<Frame>,
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
        for (i, value) in args.into_iter().enumerate() {
            locals[1 + i] = Some(value);
        }
        self.frames.push(Frame { locals });
        let result = self.run(body_index);
        self.frames.pop();
        result
    }

    fn run(&mut self, body_index: usize) -> Result<MirValue, MirRunError> {
        let body = &self.program.bodies[body_index];
        let here = self.frames.len() - 1;
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
                        let value = self.eval_rvalue(here, rvalue)?;
                        self.write_place(here, place, value)?;
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
                    let value = self.eval_operand(here, scrut)?;
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
                        values.push(self.eval_operand(here, arg)?);
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
                            let value = self.eval_operand(here, op)?;
                            let MirValue::FnPtr(idx) = value else {
                                return self.internal(format!("indirect call through {value:?}"));
                            };
                            self.call(idx, values)?
                        }
                        Callee::Runtime(rt) => self.run_runtime(*rt, values)?,
                    };
                    self.write_place(here, dest, result)?;
                    block = *target;
                }
                Terminator::Drop { place: _, target } => {
                    // No Drop elaboration is emitted before C4.5d; executing one on the
                    // drop-free subset is a nop. C4.5d gives this destructor dispatch.
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
                        values.push(self.eval_operand(here, arg)?);
                    }
                    let dest_ty = &body.locals[dest.0 as usize].ty;
                    match self.eval_checked(*op, &values, dest_ty)? {
                        Some(value) => {
                            self.frames[here].locals[dest.0 as usize] = Some(value);
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
                    return match self.frames[here].locals[0].take() {
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

    // ---- place resolution (C4.5b-2) ----

    /// Resolve a syntactic place (evaluated in `eval_frame`) to a concrete
    /// (frame, local, path). `Deref` re-anchors through `Ref` values; `Index(proof)` reads the
    /// proof from `eval_frame` (proof locals always belong to the evaluating body).
    fn resolve_place(
        &self,
        eval_frame: usize,
        place: &Place,
    ) -> Result<(usize, u32, Vec<ConcreteProj>), MirRunError> {
        let mut frame = eval_frame;
        let mut local = place.local.0;
        let mut path: Vec<ConcreteProj> = Vec::new();
        for projection in &place.projection {
            match projection {
                Projection::Field(i) => path.push(ConcreteProj::Field(*i)),
                Projection::VariantField(v, i) => path.push(ConcreteProj::Variant(*v, *i)),
                Projection::Index(proof) => {
                    let proof_value = self
                        .frames
                        .get(eval_frame)
                        .and_then(|f| f.locals.get(proof.0 as usize))
                        .and_then(|s| s.as_ref());
                    match proof_value {
                        Some(MirValue::Int(i)) => path.push(ConcreteProj::Index(*i as usize)),
                        other => {
                            return self.internal(format!(
                                "index proof _{} unavailable: {other:?}",
                                proof.0
                            ))
                        }
                    }
                }
                Projection::Deref => {
                    let current = self.read_resolved(frame, local, &path)?;
                    match current {
                        MirValue::Ref {
                            frame: f,
                            local: l,
                            path: p,
                        } => {
                            if f >= self.frames.len() {
                                return self.internal("dangling reference (frame no longer live)");
                            }
                            frame = f;
                            local = l;
                            path = p;
                        }
                        other => return self.internal(format!("Deref of non-reference {other:?}")),
                    }
                }
            }
        }
        Ok((frame, local, path))
    }

    fn read_resolved(
        &self,
        frame: usize,
        local: u32,
        path: &[ConcreteProj],
    ) -> Result<MirValue, MirRunError> {
        let Some(mut value) = self
            .frames
            .get(frame)
            .and_then(|f| f.locals.get(local as usize))
            .and_then(|s| s.as_ref())
        else {
            return self.internal(format!(
                "read of uninitialized/moved local _{local} (frame {frame})"
            ));
        };
        for step in path {
            value = match (step, value) {
                (ConcreteProj::Field(i), MirValue::Aggregate(fields)) => {
                    match fields.get(*i as usize) {
                        Some(f) => f,
                        None => return self.internal("field projection out of bounds"),
                    }
                }
                (ConcreteProj::Variant(v, i), MirValue::Enum { variant, fields }) => {
                    if variant != v {
                        return self.internal("VariantField read from a different active variant");
                    }
                    match fields.get(*i as usize) {
                        Some(f) => f,
                        None => return self.internal("variant field out of bounds"),
                    }
                }
                (ConcreteProj::Index(i), MirValue::Aggregate(elems)) => match elems.get(*i) {
                    Some(e) => e,
                    None => {
                        return self.internal("proven index out of bounds (verifier/lowering bug)")
                    }
                },
                (step, value) => {
                    return self.internal(format!("projection {step:?} on value {value:?}"))
                }
            };
        }
        Ok(value.clone())
    }

    fn write_resolved(
        &mut self,
        frame: usize,
        local: u32,
        path: &[ConcreteProj],
        value: MirValue,
    ) -> Result<(), MirRunError> {
        if path.is_empty() {
            let slot = self
                .frames
                .get_mut(frame)
                .and_then(|f| f.locals.get_mut(local as usize))
                .ok_or_else(|| MirRunError::Internal("write to invalid local".into()))?;
            *slot = Some(value);
            return Ok(());
        }
        let slot = self
            .frames
            .get_mut(frame)
            .and_then(|f| f.locals.get_mut(local as usize))
            .and_then(|s| s.as_mut())
            .ok_or_else(|| {
                MirRunError::Internal(format!(
                    "write through uninitialized local _{local} (frame {frame})"
                ))
            })?;
        let mut target = slot;
        for step in path {
            target = match (step, target) {
                (ConcreteProj::Field(i), MirValue::Aggregate(fields)) => fields
                    .get_mut(*i as usize)
                    .ok_or_else(|| MirRunError::Internal("field write out of bounds".into()))?,
                (ConcreteProj::Variant(v, i), MirValue::Enum { variant, fields }) => {
                    if *variant != *v {
                        return Err(MirRunError::Internal(
                            "VariantField write to a different active variant".into(),
                        ));
                    }
                    fields
                        .get_mut(*i as usize)
                        .ok_or_else(|| MirRunError::Internal("variant write oob".into()))?
                }
                (ConcreteProj::Index(i), MirValue::Aggregate(elems)) => {
                    elems.get_mut(*i).ok_or_else(|| {
                        MirRunError::Internal("proven index write out of bounds".into())
                    })?
                }
                (step, target) => {
                    return Err(MirRunError::Internal(format!(
                        "write projection {step:?} on {target:?}"
                    )));
                }
            };
        }
        *target = value;
        Ok(())
    }

    fn write_place(
        &mut self,
        eval_frame: usize,
        place: &Place,
        value: MirValue,
    ) -> Result<(), MirRunError> {
        let (frame, local, path) = self.resolve_place(eval_frame, place)?;
        self.write_resolved(frame, local, &path, value)
    }

    // ---- values ----

    fn eval_rvalue(&mut self, here: usize, rvalue: &Rvalue) -> Result<MirValue, MirRunError> {
        Ok(match rvalue {
            Rvalue::Use(op) => self.eval_operand(here, op)?,
            Rvalue::UnOp(op, operand) => {
                let value = self.eval_operand(here, operand)?;
                match (op, value) {
                    (MirUnOp::Not, MirValue::Bool(b)) => MirValue::Bool(!b),
                    (MirUnOp::FloatNeg, MirValue::Float(f)) => MirValue::Float(-f),
                    (op, value) => {
                        return self.internal(format!("UnOp {op:?} on {value:?}"));
                    }
                }
            }
            Rvalue::BinOp(op, lhs, rhs) => {
                let l = self.eval_operand(here, lhs)?;
                let r = self.eval_operand(here, rhs)?;
                self.eval_binop(*op, l, r)?
            }
            Rvalue::Aggregate(kind, operands) => {
                let mut values = Vec::new();
                for op in operands {
                    values.push(self.eval_operand(here, op)?);
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
                let (f, l, p) = self.resolve_place(here, place)?;
                let value = self.read_resolved(f, l, &p)?;
                let MirValue::Enum { variant, .. } = value else {
                    return self.internal(format!("Discriminant of {value:?}"));
                };
                MirValue::Int(i128::from(variant))
            }
            // C4.5b-2: real reference creation.
            Rvalue::RefOf { place, .. } => {
                let (frame, local, path) = self.resolve_place(here, place)?;
                MirValue::Ref { frame, local, path }
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
                "shifts are not lowered yet (C4.5)".into(),
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
            CheckIndex => {
                let len = match &args[0] {
                    MirValue::Aggregate(elems) => elems.len() as i128,
                    other => {
                        return Err(MirRunError::Internal(format!(
                            "CheckIndex base is not an aggregate: {other:?}"
                        )))
                    }
                };
                let index = match &args[1] {
                    MirValue::Int(i) => *i,
                    other => {
                        return Err(MirRunError::Internal(format!(
                            "CheckIndex index is not an integer: {other:?}"
                        )))
                    }
                };
                if index >= 0 && index < len {
                    // The proof VALUE is the checked index (interp-internal representation of
                    // the opaque token; MIR-level opacity is the verifier's concern).
                    Ok(Some(MirValue::Int(index)))
                } else {
                    Ok(None) // IndexOutOfBounds trap
                }
            }
        }
    }

    // ---- operands ----

    fn eval_operand(&mut self, here: usize, op: &Operand) -> Result<MirValue, MirRunError> {
        match op {
            Operand::Copy(place) => {
                let (f, l, p) = self.resolve_place(here, place)?;
                self.read_resolved(f, l, &p)
            }
            Operand::Move(place) => {
                if place.projection.is_empty() {
                    match self.frames[here].locals[place.local.0 as usize].take() {
                        Some(value) => Ok(value),
                        None => self.internal(format!(
                            "move from uninitialized/moved local _{}",
                            place.local.0
                        )),
                    }
                } else {
                    // Projected move: read the sub-value (the verifier's whole-local
                    // conservatism already guards double-use).
                    let (f, l, p) = self.resolve_place(here, place)?;
                    self.read_resolved(f, l, &p)
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
