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
//! - `Drop` terminators (C4.5d) run recursive drop glue: the type's own destructor instance
//!   through an `&mut` reference (mutations stay visible to what follows), then fields or the
//!   runtime-discriminant variant payload in reverse declaration order; whole-local drops
//!   poison the slot afterwards.
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
    /// 0.1-A6 (A4 slicing): a sub-range view over an Array/Vec referent — `start` is the
    /// absolute element offset in the underlying container, `len` the view length. Appears
    /// only in `MirValue::Ref` paths (a `&[T]` value); re-slicing composes into one step.
    Slice {
        start: usize,
        len: usize,
    },
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
    /// A reference: a resolved path into some live frame (C4.5b-2). C4.5f-1: carries the
    /// pointee frame's **generation** — frame slots are reused across calls, and a slot
    /// index alone would let a stale reference silently alias a later frame (CD-030 review
    /// finding). Every deref validates `frames[frame].generation == generation` and fails
    /// loudly on mismatch.
    Ref {
        frame: usize,
        generation: u64,
        local: u32,
        path: Vec<ConcreteProj>,
    },
    /// A `&str` value (A1). Self-contained: a literal, or a read-only snapshot produced by
    /// `StringAsStr` (§5b — sound because the view is read-only and str identity is
    /// unobservable). `Rc` is an unobservable cheap-copy convenience.
    Str(std::rc::Rc<str>),
    /// An owned `String` value (A1). Non-Copy; drop-elaborated (buffer reclaim is a no-op).
    String(String),
    /// An owned `Vec<T>` value (A1/C4.5e-2). Non-Copy; drop-elaborated (elements dropped in
    /// reverse index order, then buffer reclaimed — §5a). Opaque to projections: manipulated
    /// only through the Vec `RuntimeFn` surface.
    Vec(Vec<MirValue>),
    /// C4.5f-1 (CD-030 deferral): the poison a projected `Move` leaves behind. Any read of a
    /// `Moved` hole is a loud internal error — verifier-missed use-after-partial-move can
    /// never silently observe the retained value (previously projected moves cloned, relying
    /// on the verifier alone).
    Moved,
}

#[derive(Debug)]
pub enum MirRunError {
    /// A language trap: category AND provenance (CD-029 — category alone made the
    /// differential blind to wrong-location traps) AND the resolved user message when one
    /// exists (A1/CD-031 — `panic(msg)` carries it; compiler-generated traps carry `None`).
    Trap {
        category: TrapCategory,
        source: SourceInfo,
        message: Option<String>,
    },
    /// A bug in lowering/verification/interpretation — never a language-level outcome.
    Internal(String),
}

pub struct MirExecution {
    pub output: String,
    pub status: u8,
}

/// A failed run, still carrying the stdout accumulated before the failure (C4.5e-0): the
/// roadmap's comparator is output AND failure equality, so pre-trap output is observable.
#[derive(Debug)]
pub struct MirFailure {
    pub error: MirRunError,
    pub output: String,
}

const FUEL: u64 = 50_000_000;

struct Frame {
    locals: Vec<Option<MirValue>>,
    /// C4.5f-1: unique, monotonically assigned identity — never reused, unlike the slot index.
    generation: u64,
}

pub fn run_program(
    verified: crate::mir::verify::VerifiedMirProgram<'_>,
) -> Result<MirExecution, MirFailure> {
    let program = verified.program();
    let by_symbol: HashMap<&str, usize> = program
        .bodies
        .iter()
        .enumerate()
        .map(|(i, b)| (b.instance.symbol.as_str(), i))
        .collect();
    let Some(&main_index) = by_symbol.get("main@[]") else {
        return Err(MirFailure {
            error: MirRunError::Internal("no main@[] instance".to_string()),
            output: String::new(),
        });
    };
    let mut cx = Interp {
        program,
        by_symbol,
        frames: Vec::new(),
        next_generation: 0,
        output: String::new(),
        fuel: FUEL,
    };
    match cx.call(main_index, Vec::new()) {
        Ok(_) => Ok(MirExecution {
            output: cx.output,
            status: 0,
        }),
        Err(error) => Err(MirFailure {
            error,
            output: cx.output,
        }),
    }
}

struct Interp<'a> {
    program: &'a MirProgram,
    by_symbol: HashMap<&'a str, usize>,
    frames: Vec<Frame>,
    /// C4.5f-1: monotonic frame-generation counter (never reset, never reused).
    next_generation: u64,
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
        // Bind arguments by their declared `Param(i)` kind — param locals are NOT
        // contiguous when drop-flag locals are interleaved between them (C4.5d).
        let mut args = args.into_iter();
        for param in 0.. {
            let Some(slot) = body
                .locals
                .iter()
                .position(|decl| decl.kind == LocalKind::Param(param))
            else {
                break;
            };
            let Some(value) = args.next() else {
                return self.internal("argument count does not match callee param locals");
            };
            locals[slot] = Some(value);
        }
        if args.next().is_some() {
            return self.internal("argument count does not match callee param locals");
        }
        let generation = self.next_generation;
        self.next_generation += 1;
        self.frames.push(Frame { locals, generation });
        let result = self.run(body_index);
        self.frames.pop();
        result
    }

    /// C4.5f-1: a live-frame check by (slot, generation) — the defense-in-depth guarantee
    /// that a stale reference fails loudly instead of silently aliasing a reused slot.
    fn check_ref_live(&self, frame: usize, generation: u64) -> Result<(), MirRunError> {
        match self.frames.get(frame) {
            Some(f) if f.generation == generation => Ok(()),
            Some(f) => self.internal(format!(
                "dangling reference: frame slot {frame} was reused (generation {} != {})",
                f.generation, generation
            )),
            None => self.internal(format!("dangling reference: frame {frame} no longer live")),
        }
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
                        // A1: trap-capable runtime ops abort with the CALL SITE's SourceInfo
                        // as provenance (runtime ops carry no TrapInfo of their own, §5).
                        Callee::Runtime(rt) => self.run_runtime(*rt, values, bb.terminator.1)?,
                    };
                    self.write_place(here, dest, result)?;
                    block = *target;
                }
                Terminator::Drop { place, target } => {
                    // C4.5d: run the place's drop glue — its own destructor instance (if
                    // the type has a Drop impl), then its fields/payload in reverse
                    // declaration order. A destructor that traps aborts (no unwind edge).
                    let ty = self.place_ty(body, place)?;
                    let (f, l, p) = self.resolve_place(here, place)?;
                    self.drop_in_place(f, l, p, &ty)?;
                    if place.projection.is_empty() {
                        // Whole-local drop: poison the slot so verifier-missed
                        // use-after-drop explodes loudly (same discipline as Move).
                        self.frames[here].locals[place.local.0 as usize] = None;
                    }
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
                        CheckedOutcome::Value(value) => {
                            self.frames[here].locals[dest.0 as usize] = Some(value);
                            block = *target;
                        }
                        // A5: a shift with a bad count overrides the terminator's category with
                        // `InvalidShift`; every other trap uses the terminator's own category.
                        CheckedOutcome::Trap(override_category) => {
                            return Err(MirRunError::Trap {
                                category: override_category.unwrap_or(trap.category),
                                source: trap.source,
                                message: None,
                            })
                        }
                    }
                }
                Terminator::Trap { info, message } => {
                    // A1: resolve the optional `&str` message before aborting (it participates
                    // in evaluation like any operand).
                    let resolved = match message {
                        Some(op) => Some(self.eval_str_operand(here, op)?),
                        None => None,
                    };
                    return Err(MirRunError::Trap {
                        category: info.category,
                        source: info.source,
                        message: resolved,
                    });
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

    // ---- drop glue (C4.5d) ----

    /// Syntactic type of a place: local type refined through the projections, resolved via
    /// the program's type context (the same derivation the verifier types places with).
    fn place_ty(&self, body: &MirBody, place: &Place) -> Result<MirTy, MirRunError> {
        let mut ty = body
            .locals
            .get(place.local.0 as usize)
            .map(|d| d.ty.clone())
            .ok_or_else(|| MirRunError::Internal("place local out of bounds".into()))?;
        for proj in &place.projection {
            ty = match (proj, ty) {
                (Projection::Field(i), MirTy::Struct(item, args)) => self
                    .program
                    .types
                    .struct_fields
                    .get(&(item.0, args))
                    .and_then(|fields| fields.get(*i as usize))
                    .cloned()
                    .ok_or_else(|| {
                        MirRunError::Internal(format!("no field type for struct #{}", item.0))
                    })?,
                (Projection::Field(i), MirTy::Tuple(elems)) => elems
                    .get(*i as usize)
                    .cloned()
                    .ok_or_else(|| MirRunError::Internal("tuple field out of bounds".into()))?,
                (Projection::VariantField(v, i), MirTy::Enum(enum_ref, args)) => self
                    .variant_payload_tys(&enum_ref, &args, *v)?
                    .get(*i as usize)
                    .cloned()
                    .ok_or_else(|| MirRunError::Internal("variant field out of bounds".into()))?,
                (Projection::Deref, MirTy::Ref { inner, .. }) => *inner,
                (Projection::Index(_), MirTy::Array(elem, _))
                | (Projection::Index(_), MirTy::Slice(elem)) => *elem,
                (proj, ty) => {
                    return self.internal(format!("place typing: {proj:?} on {ty:?}"));
                }
            };
        }
        Ok(ty)
    }

    fn variant_payload_tys(
        &self,
        enum_ref: &EnumRef,
        args: &[MirTy],
        variant: u32,
    ) -> Result<Vec<MirTy>, MirRunError> {
        match enum_ref {
            EnumRef::CoreOption => Ok(if variant == 1 {
                vec![args
                    .first()
                    .cloned()
                    .ok_or_else(|| MirRunError::Internal("Option without argument".into()))?]
            } else {
                Vec::new()
            }),
            EnumRef::CoreResult => Ok(vec![args
                .get(variant as usize)
                .cloned()
                .ok_or_else(|| MirRunError::Internal("Result variant out of range".into()))?]),
            // A2 (CE3): Ordering's three variants are all fieldless.
            EnumRef::CoreOrdering => Ok(Vec::new()),
            EnumRef::User(item) => self
                .program
                .types
                .enum_variants
                .get(&(item.0, args.to_vec()))
                .and_then(|variants| variants.get(variant as usize))
                .cloned()
                .ok_or_else(|| {
                    MirRunError::Internal(format!("no variant table for enum #{}", item.0))
                }),
        }
    }

    /// Recursive drop glue for the value at (frame, local, path): call the type's own
    /// destructor instance through an `&mut` reference (mutations stay visible to the field
    /// destruction that follows, matching the oracle), then destroy fields/payload in
    /// reverse declaration order.
    fn drop_in_place(
        &mut self,
        frame: usize,
        local: u32,
        path: Vec<ConcreteProj>,
        ty: &MirTy,
    ) -> Result<(), MirRunError> {
        match ty {
            MirTy::Struct(item, args) => {
                if let Some(symbol) = self.program.types.drop_impls.get(&(item.0, args.clone())) {
                    let Some(&idx) = self.by_symbol.get(symbol.as_str()) else {
                        return self.internal(format!("dtor instance {symbol} not lowered"));
                    };
                    let receiver = MirValue::Ref {
                        frame,
                        generation: self.frames[frame].generation,
                        local,
                        path: path.clone(),
                    };
                    self.call(idx, vec![receiver])?;
                }
                let fields = self
                    .program
                    .types
                    .struct_fields
                    .get(&(item.0, args.clone()))
                    .cloned()
                    .ok_or_else(|| {
                        MirRunError::Internal(format!("no field table for struct #{}", item.0))
                    })?;
                for (i, fty) in fields.iter().enumerate().rev() {
                    let mut p = path.clone();
                    p.push(ConcreteProj::Field(i as u32));
                    self.drop_in_place(frame, local, p, fty)?;
                }
            }
            MirTy::Enum(enum_ref, args) => {
                if let EnumRef::User(item) = enum_ref {
                    if let Some(symbol) = self.program.types.drop_impls.get(&(item.0, args.clone()))
                    {
                        let Some(&idx) = self.by_symbol.get(symbol.as_str()) else {
                            return self.internal(format!("dtor instance {symbol} not lowered"));
                        };
                        let receiver = MirValue::Ref {
                            frame,
                            generation: self.frames[frame].generation,
                            local,
                            path: path.clone(),
                        };
                        self.call(idx, vec![receiver])?;
                    }
                }
                let value = self.read_resolved(frame, local, &path)?;
                let MirValue::Enum { variant, .. } = value else {
                    return self.internal("Drop glue: enum-typed place holds a non-enum value");
                };
                let payload = self.variant_payload_tys(enum_ref, args, variant)?;
                for (i, fty) in payload.iter().enumerate().rev() {
                    let mut p = path.clone();
                    p.push(ConcreteProj::Variant(variant, i as u32));
                    self.drop_in_place(frame, local, p, fty)?;
                }
            }
            MirTy::Tuple(elems) => {
                for (i, ety) in elems.iter().enumerate().rev() {
                    let mut p = path.clone();
                    p.push(ConcreteProj::Field(i as u32));
                    self.drop_in_place(frame, local, p, ety)?;
                }
            }
            MirTy::Array(elem, len) => {
                for i in (0..*len).rev() {
                    let mut p = path.clone();
                    p.push(ConcreteProj::Index(i as usize));
                    self.drop_in_place(frame, local, p, elem)?;
                }
            }
            // A1 (CD-031) §5a: Vec<T> drops its elements through STARK glue in REVERSE index
            // order (matched to the oracle), then reclaims the buffer (unobservable). Elements
            // are opaque to projections, so they drop from the read-out value via a scratch
            // slot rather than a place path.
            MirTy::Core(crate::hir::CoreType::Vec, args) => {
                let elem_ty = args.first().cloned().unwrap_or(MirTy::Unit);
                if let MirValue::Vec(mut elems) = self.read_resolved(frame, local, &path)? {
                    while let Some(e) = elems.pop() {
                        self.drop_owned_value(frame, e, &elem_ty)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Drop a standalone owned value (a Vec element, §5a): place it in a scratch slot of
    /// `frame` so its glue can take an `&mut` reference, run the glue, then remove the slot.
    fn drop_owned_value(
        &mut self,
        frame: usize,
        value: MirValue,
        ty: &MirTy,
    ) -> Result<(), MirRunError> {
        let scratch = self.frames[frame].locals.len() as u32;
        self.frames[frame].locals.push(Some(value));
        let result = self.drop_in_place(frame, scratch, Vec::new(), ty);
        self.frames[frame].locals.truncate(scratch as usize);
        result
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
                            generation,
                            local: l,
                            path: p,
                        } => {
                            // C4.5f-1: slot AND generation must match — a reused slot is a
                            // dangling reference, reported loudly, never silently aliased.
                            self.check_ref_live(f, generation)?;
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
        let mut k = 0;
        while k < path.len() {
            let step = &path[k];
            value = match (step, value) {
                // 0.1-A6 (A4 slicing): a Slice window either COMPOSES with a following Index
                // (view-relative i becomes absolute start+i) or, when it ends the path, yields
                // a cloned sub-view value (read-only — used for CheckIndex length reads).
                (ConcreteProj::Slice { start, len }, MirValue::Aggregate(elems))
                | (ConcreteProj::Slice { start, len }, MirValue::Vec(elems)) => {
                    if let Some(ConcreteProj::Index(i)) = path.get(k + 1) {
                        if *i >= *len {
                            return self
                                .internal("proven slice index out of bounds (verifier bug)");
                        }
                        k += 1; // consume the composed Index as well
                        match elems.get(start + i) {
                            Some(e) => e,
                            None => return self.internal("slice window exceeds its base"),
                        }
                    } else {
                        let sub = elems
                            .get(*start..start + len)
                            .ok_or_else(|| {
                                MirRunError::Internal("slice window exceeds its base".into())
                            })?
                            .to_vec();
                        return Ok(MirValue::Vec(sub));
                    }
                }
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
                // 0.1-A2: `Index` also resolves into a runtime Vec snapshot (iterator
                // interior references, C4.5f-2).
                (ConcreteProj::Index(i), MirValue::Aggregate(elems))
                | (ConcreteProj::Index(i), MirValue::Vec(elems)) => match elems.get(*i) {
                    Some(e) => e,
                    None => {
                        return self.internal("proven index out of bounds (verifier/lowering bug)")
                    }
                },
                (step, value) => {
                    return self.internal(format!("projection {step:?} on value {value:?}"))
                }
            };
            k += 1;
        }
        if matches!(value, MirValue::Moved) {
            return self.internal(format!(
                "read of a moved-out place _{local}{path:?} (C4.5f-1 poison)"
            ));
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
                (ConcreteProj::Index(i), MirValue::Aggregate(elems))
                | (ConcreteProj::Index(i), MirValue::Vec(elems)) => {
                    elems.get_mut(*i).ok_or_else(|| {
                        MirRunError::Internal("proven index write out of bounds".into())
                    })?
                }
                // 0.1-A6: slice views are SHARED — no writes route through them (the front end
                // rejects assignment through an immutable slice; loud failure if one slips by).
                (ConcreteProj::Slice { .. }, _) => {
                    return Err(MirRunError::Internal(
                        "write through a shared slice view".into(),
                    ))
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
            // C4.5b-2: real reference creation (C4.5f-1: stamped with the pointee frame's
            // generation).
            Rvalue::RefOf { place, .. } => {
                let (frame, local, path) = self.resolve_place(here, place)?;
                let generation = self.frames[frame].generation;
                MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }
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
            // A5: bitwise on the sign-extended i128 carrier — for same-width operands the low
            // bits agree with the true-width result and the value stays in range (no trap).
            (BitAnd, MirValue::Int(a), MirValue::Int(b)) => MirValue::Int(a & b),
            (BitOr, MirValue::Int(a), MirValue::Int(b)) => MirValue::Int(a | b),
            (BitXor, MirValue::Int(a), MirValue::Int(b)) => MirValue::Int(a ^ b),
            (op, l, r) => {
                return self.internal(format!("BinOp {op:?} on {l:?}, {r:?}"));
            }
        })
    }

    // (see `CheckedOutcome` below)
    /// Checked/trapping primitives. `Trap(None)` traps with the terminator's own category;
    /// `Trap(Some(cat))` overrides it (A5 shifts: a bad count is `InvalidShift`).
    fn eval_checked(
        &self,
        op: CheckedOp,
        args: &[MirValue],
        dest_ty: &MirTy,
    ) -> Result<CheckedOutcome, MirRunError> {
        use CheckedOp::*;
        match op {
            Add | Sub | Mul | Div | Rem | Neg | Pow => {
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
                    // A5: exponent must be nonnegative (u32::try_from rejects negatives,
                    // NUM-INT-ARITH-001); each intermediate multiply is checked by checked_pow.
                    Pow => {
                        let base = int(&args[0])?;
                        u32::try_from(int(&args[1])?)
                            .ok()
                            .and_then(|exp| base.checked_pow(exp))
                    }
                    _ => unreachable!(),
                };
                Ok(result
                    .filter(|v| *v >= min && *v <= max)
                    .map(MirValue::Int)
                    .into())
            }
            Shl | Shr => {
                // A5 / NUM-SHIFT-001: the count must be nonnegative and strictly less than the
                // bit width of the LEFT operand (= the dest/result type); otherwise trap. No
                // masking or reduction. Left shift traps when the result is not representable
                // (the post-hoc range filter); right shift on the i128 carrier is arithmetic
                // for signed and — since unsigned values are stored nonnegative — logical for
                // unsigned, matching the abstract machine.
                let (min, max) = int_range(dest_ty)
                    .ok_or_else(|| MirRunError::Internal("shift on non-int".into()))?;
                let width = int_width(dest_ty)
                    .ok_or_else(|| MirRunError::Internal("shift width on non-int".into()))?;
                let int = |v: &MirValue| -> Result<i128, MirRunError> {
                    match v {
                        MirValue::Int(i) => Ok(*i),
                        other => Err(MirRunError::Internal(format!("shift operand {other:?}"))),
                    }
                };
                let (left, count) = (int(&args[0])?, int(&args[1])?);
                if count < 0 || count >= i128::from(width) {
                    return Ok(CheckedOutcome::Trap(Some(TrapCategory::InvalidShift)));
                }
                let result = if matches!(op, Shl) {
                    left.checked_shl(count as u32)
                } else {
                    left.checked_shr(count as u32)
                };
                Ok(result
                    .filter(|v| *v >= min && *v <= max)
                    .map(MirValue::Int)
                    .into())
            }
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
                    return Ok(CheckedOutcome::Trap(None));
                }
                Ok(CheckedOutcome::Value(MirValue::Float(
                    if matches!(op, FloatDiv) { a / b } else { a % b },
                )))
            }
            Cast => {
                let value = &args[0];
                Ok(CheckedOutcome::from(match (value, dest_ty) {
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
                }))
            }
            CheckIndex => {
                let len = match &args[0] {
                    MirValue::Aggregate(elems) => elems.len() as i128,
                    // 0.1-A6: a slice-view base reads as a Vec sub-view; its len is the VIEW
                    // length, so the proof bounds i against the view, not the base container.
                    MirValue::Vec(elems) => elems.len() as i128,
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
                    Ok(CheckedOutcome::Value(MirValue::Int(index)))
                } else {
                    Ok(CheckedOutcome::Trap(None)) // IndexOutOfBounds
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
                    // C4.5f-1: projected move TAKES — the sub-value is replaced with a
                    // `Moved` poison so a verifier-missed later read explodes loudly instead
                    // of silently observing a retained clone (CD-030 review warning; the
                    // field-precise V-MOVE-1 is the primary guard, this is defense in depth).
                    let (f, l, p) = self.resolve_place(here, place)?;
                    let value = self.read_resolved(f, l, &p)?;
                    if matches!(value, MirValue::Moved) {
                        return self.internal(format!(
                            "move from an already-moved place _{}{:?}",
                            place.local.0, place.projection
                        ));
                    }
                    self.write_resolved(f, l, &p, MirValue::Moved)?;
                    Ok(value)
                }
            }
            Operand::Const(constant) => Ok(match constant {
                Constant::Int(v, _) => MirValue::Int(*v),
                Constant::Float(f, _) => MirValue::Float(*f),
                Constant::Bool(b) => MirValue::Bool(*b),
                Constant::Unit => MirValue::Unit,
                Constant::Str(s) => MirValue::Str(std::rc::Rc::from(s.as_str())),
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

    fn run_runtime(
        &mut self,
        rt: RuntimeFn,
        args: Vec<MirValue>,
        call_info: SourceInfo,
    ) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        if is_vec_runtime(rt) {
            return self.run_vec_runtime(rt, args, call_info);
        }
        if is_map_runtime(rt) {
            return self.run_map_runtime(rt, args);
        }
        if is_slice_runtime(rt) {
            return self.run_slice_runtime(rt, args, call_info);
        }
        let mut iter = args.into_iter();
        let arg = iter.next();
        let rest: Vec<MirValue> = iter.collect();
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
            // --- A1 str/String ops. `arg` holds the reconstructed first argument; the closure
            // below re-materializes the full list when an op needs more than one. ---
            (rt, arg) => self.run_string_runtime(rt, arg, rest),
        }
    }

    /// A1 String/str runtime ops. `first` is the (already-popped) first argument; `rest` is the
    /// remainder. `&str` operands arrive as `Str` values (lowering inserts `StringAsStr` for
    /// `String` sources, §5b); `&String`/`&mut String` operands arrive as `Ref`s into a live
    /// frame, read/mutated in place here.
    fn run_string_runtime(
        &mut self,
        rt: RuntimeFn,
        first: Option<MirValue>,
        rest: Vec<MirValue>,
    ) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        let mut rest = rest.into_iter();
        match rt {
            PrintlnStr | PrintStr => {
                let s = self.as_str(&first)?;
                if matches!(rt, PrintlnStr) {
                    let _ = writeln!(self.output, "{s}");
                } else {
                    let _ = write!(self.output, "{s}");
                }
                Ok(MirValue::Unit)
            }
            StringNew => Ok(MirValue::String(String::new())),
            StringFromStr => Ok(MirValue::String(self.as_str(&first)?.to_string())),
            StrToString => Ok(MirValue::String(self.as_str(&first)?.to_string())),
            StringClone => {
                let s = self.read_string_ref(&first)?;
                Ok(MirValue::String(s))
            }
            StringAsStr => {
                // Interior reference → read-only snapshot (§5b).
                let s = self.read_string_ref(&first)?;
                Ok(MirValue::Str(std::rc::Rc::from(s.as_str())))
            }
            StringLen => {
                let s = self.read_string_ref(&first)?;
                Ok(MirValue::Int(s.len() as i128))
            }
            StringIsEmpty => {
                let s = self.read_string_ref(&first)?;
                Ok(MirValue::Bool(s.is_empty()))
            }
            StringContains => {
                let s = self.read_string_ref(&first)?;
                let pat = self.as_str(&rest.next())?.to_string();
                Ok(MirValue::Bool(s.contains(&pat)))
            }
            StringPushStr => {
                let suffix = self.as_str(&rest.next())?.to_string();
                self.mutate_string_ref(&first, |s| s.push_str(&suffix))?;
                Ok(MirValue::Unit)
            }
            StringClear => {
                self.mutate_string_ref(&first, |s| s.clear())?;
                Ok(MirValue::Unit)
            }
            // 0.1-A3 (f-3b): Char ops. Char values are Unicode scalar codepoints in
            // MirValue::Int.
            PrintlnChar | PrintChar => {
                let c = char_of(&first)?;
                if matches!(rt, PrintlnChar) {
                    let _ = writeln!(self.output, "{c}");
                } else {
                    let _ = write!(self.output, "{c}");
                }
                Ok(MirValue::Unit)
            }
            StringPushChar => {
                let c = char_of(&rest.next())?;
                self.mutate_string_ref(&first, |s| s.push(c))?;
                Ok(MirValue::Unit)
            }
            StringPopChar => {
                let mut popped: Option<char> = None;
                self.mutate_string_ref(&first, |s| popped = s.pop())?;
                Ok(match popped {
                    Some(c) => MirValue::Enum {
                        variant: 1,
                        fields: vec![MirValue::Int(i128::from(u32::from(c)))],
                    },
                    None => MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    },
                })
            }
            StrLen => Ok(MirValue::Int(self.as_str(&first)?.len() as i128)),
            StrIsEmpty => Ok(MirValue::Bool(self.as_str(&first)?.is_empty())),
            StrEq => {
                let a = self.as_str(&first)?.to_string();
                let b = self.as_str(&rest.next())?.to_string();
                Ok(MirValue::Bool(a == b))
            }
            StrCmp => {
                let a = self.as_str(&first)?.to_string();
                let b = self.as_str(&rest.next())?.to_string();
                let ord = match a.cmp(&b) {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                };
                Ok(MirValue::Int(ord))
            }
            // 0.1-A5 (A4-2d): `chars()` iteration. The iterator snapshots the string's chars
            // (Char is Copy, so a snapshot matches the oracle's borrowed `CharsIter`); it lives
            // as `Aggregate([Str(snapshot), Int(cursor)])`. `Next` yields `Option<Char>`.
            CharsIterNew => {
                let s = self.as_str(&first)?.to_string();
                Ok(MirValue::Aggregate(vec![
                    MirValue::Str(std::rc::Rc::from(s.as_str())),
                    MirValue::Int(0),
                ]))
            }
            CharsIterNext => {
                let Some(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }) = first
                else {
                    return self.internal("CharsIterNext expects a &mut iterator reference");
                };
                self.check_ref_live(frame, generation)?;
                let iter_value = self.read_resolved(frame, local, &path)?;
                let MirValue::Aggregate(fields) = &iter_value else {
                    return self.internal(format!("chars iterator referent is {iter_value:?}"));
                };
                let (snapshot, cursor) = match (fields.first(), fields.get(1)) {
                    (Some(MirValue::Str(s)), Some(MirValue::Int(c))) => (s.clone(), *c as usize),
                    other => {
                        return self.internal(format!("malformed chars-iterator state {other:?}"))
                    }
                };
                match snapshot.chars().nth(cursor) {
                    Some(ch) => {
                        let mut cursor_path = path;
                        cursor_path.push(ConcreteProj::Field(1));
                        self.write_resolved(
                            frame,
                            local,
                            &cursor_path,
                            MirValue::Int(cursor as i128 + 1),
                        )?;
                        Ok(MirValue::Enum {
                            variant: 1,
                            fields: vec![MirValue::Int(i128::from(u32::from(ch)))],
                        })
                    }
                    None => Ok(MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    }),
                }
            }
            other => self.internal(format!("runtime {other:?} (string group) unhandled")),
        }
    }

    /// A1 (CD-031) Vec data-surface ops. `&Vec`/`&mut Vec` operands arrive as `Ref`s and are
    /// read/mutated in place; index/replace/remove trap `IndexOutOfBounds` with the call
    /// site's provenance (§5). Iteration is not here (deferred — see the enum note).
    fn run_vec_runtime(
        &mut self,
        rt: RuntimeFn,
        args: Vec<MirValue>,
        call_info: SourceInfo,
    ) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        let oob = || MirRunError::Trap {
            category: TrapCategory::IndexOutOfBounds,
            source: call_info,
            message: None,
        };
        let mut args = args.into_iter();
        match rt {
            VecNew | VecWithCapacity => Ok(MirValue::Vec(Vec::new())),
            VecPush => {
                let recv = args.next();
                let item = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("VecPush missing item".into()))?;
                self.mutate_vec_ref(&recv, |v| v.push(item))?;
                Ok(MirValue::Unit)
            }
            VecPop => {
                let recv = args.next();
                let popped = self.mutate_vec_ref(&recv, |v| v.pop())?;
                Ok(option_value(popped))
            }
            VecLen => {
                let recv = args.next();
                Ok(MirValue::Int(self.read_vec_ref(&recv)?.len() as i128))
            }
            VecIsEmpty => {
                let recv = args.next();
                Ok(MirValue::Bool(self.read_vec_ref(&recv)?.is_empty()))
            }
            VecClear => {
                let recv = args.next();
                self.mutate_vec_ref(&recv, |v| v.clear())?;
                Ok(MirValue::Unit)
            }
            VecIndexGet => {
                let recv = args.next();
                let i = int_arg(args.next())? as usize;
                let v = self.read_vec_ref(&recv)?;
                v.get(i).cloned().ok_or_else(oob)
            }
            VecReplace => {
                let recv = args.next();
                let i = int_arg(args.next())? as usize;
                let item = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("VecReplace missing item".into()))?;
                let len = self.read_vec_ref(&recv)?.len();
                if i >= len {
                    return Err(oob());
                }
                self.mutate_vec_ref(&recv, |v| std::mem::replace(&mut v[i], item))
            }
            VecRemove => {
                let recv = args.next();
                let i = int_arg(args.next())? as usize;
                let len = self.read_vec_ref(&recv)?.len();
                if i >= len {
                    return Err(oob());
                }
                self.mutate_vec_ref(&recv, |v| v.remove(i))
            }
            // 0.1-A4 (A4-2b): `get`/`get_mut` — checked interior access, NEVER traps. Return an
            // interior reference into the live Vec at `i` as `Some(&v[i])`, or `None` when out
            // of bounds. Mirrors HashMapGet; the mutability is a static property of the ref type.
            VecGetRef | VecGetMutRef => {
                let Some(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }) = args.next()
                else {
                    return self.internal("VecGet(Mut)Ref expects a &Vec reference");
                };
                self.check_ref_live(frame, generation)?;
                let i = int_arg(args.next())? as usize;
                let len = match self.read_resolved(frame, local, &path)? {
                    MirValue::Vec(elems) => elems.len(),
                    other => return self.internal(format!("Vec referent is {other:?}")),
                };
                if i < len {
                    let mut elem_path = path;
                    elem_path.push(ConcreteProj::Index(i));
                    Ok(MirValue::Enum {
                        variant: 1,
                        fields: vec![MirValue::Ref {
                            frame,
                            generation,
                            local,
                            path: elem_path,
                        }],
                    })
                } else {
                    Ok(MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    })
                }
            }
            // --- 0.1-A2 (C4.5f-2): by-reference iteration. The iterator value is an opaque
            // two-field aggregate [snapshot Vec, Int cursor] living in a frame local; `Next`
            // hands out interior references into THAT local (base + [Field(0), Index(i)]),
            // which the f-1 frame-generation guard protects once the iterator dies. The
            // snapshot is sound because iteration is `T: Copy` and borrowck forbids mutating
            // the source Vec while the iterator lives (A1 §5e carry-forward).
            VecIterNew => {
                // A6: a TRUE borrowed cursor `[vec-ref, cursor]` — Next indexes the LIVE Vec
                // through the reference and hands out an interior `&T`. No snapshot, so the
                // element type need not be `Copy` (the borrow checker already forbids mutating
                // the source Vec while the iterator lives).
                let vec_ref = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("VecIterNew missing receiver".into()))?;
                Ok(MirValue::Aggregate(vec![vec_ref, MirValue::Int(0)]))
            }
            VecIterNext => {
                let Some(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }) = args.next()
                else {
                    return self.internal("VecIterNext expects a &mut iterator reference");
                };
                self.check_ref_live(frame, generation)?;
                let iter_value = self.read_resolved(frame, local, &path)?;
                let MirValue::Aggregate(fields) = &iter_value else {
                    return self.internal(format!("iterator referent is {iter_value:?}"));
                };
                let (vec_ref, cursor) = match (fields.first(), fields.get(1)) {
                    (Some(r @ MirValue::Ref { .. }), Some(MirValue::Int(c))) => (r.clone(), *c),
                    other => return self.internal(format!("malformed iterator state {other:?}")),
                };
                let MirValue::Ref {
                    frame: vf,
                    generation: vg,
                    local: vl,
                    path: vp,
                } = vec_ref
                else {
                    unreachable!("matched Ref above");
                };
                self.check_ref_live(vf, vg)?;
                let len = match self.read_resolved(vf, vl, &vp)? {
                    MirValue::Vec(elems) => elems.len(),
                    other => return self.internal(format!("Vec referent is {other:?}")),
                };
                if (cursor as usize) < len {
                    // Bump the cursor in place, then hand out an interior reference into the
                    // live Vec (base + [Index(cursor)]), protected by the f-1 generation guard.
                    let mut cursor_path = path;
                    cursor_path.push(ConcreteProj::Field(1));
                    self.write_resolved(frame, local, &cursor_path, MirValue::Int(cursor + 1))?;
                    let mut elem_path = vp;
                    elem_path.push(ConcreteProj::Index(cursor as usize));
                    Ok(MirValue::Enum {
                        variant: 1,
                        fields: vec![MirValue::Ref {
                            frame: vf,
                            generation: vg,
                            local: vl,
                            path: elem_path,
                        }],
                    })
                } else {
                    Ok(MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    })
                }
            }
            other => self.internal(format!("runtime {other:?} is not a Vec op")),
        }
    }

    /// 0.1-A3 (C4.5f-3a): HashMap ops. The map value is an insertion-ordered pair vector
    /// (`MirValue::Vec` of `Aggregate([k, v])`) per CD-009 — re-inserting an existing key
    /// keeps its position; lookups are structural key comparison. `Get` and the keys
    /// iterator hand out interior references (entry `[Index(i), Field(0|1)]`), guarded by
    /// the f-1 frame generations.
    fn run_map_runtime(
        &mut self,
        rt: RuntimeFn,
        args: Vec<MirValue>,
    ) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        let mut args = args.into_iter();
        match rt {
            HashMapNew => Ok(MirValue::Vec(Vec::new())),
            HashMapInsert => {
                let recv = args.next();
                let key = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("HashMapInsert missing key".into()))?;
                let value = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("HashMapInsert missing value".into()))?;
                self.mutate_vec_ref(&recv, |entries| {
                    for entry in entries.iter_mut() {
                        if let MirValue::Aggregate(kv) = entry {
                            if kv[0] == key {
                                // CD-009: existing key keeps its position; old value returned.
                                let old = std::mem::replace(&mut kv[1], value);
                                return MirValue::Enum {
                                    variant: 1,
                                    fields: vec![old],
                                };
                            }
                        }
                    }
                    entries.push(MirValue::Aggregate(vec![key, value]));
                    MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    }
                })
            }
            HashMapGet => {
                let Some(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }) = args.next()
                else {
                    return self.internal("HashMapGet expects a &HashMap reference");
                };
                self.check_ref_live(frame, generation)?;
                let key = self.deref_key_arg(args.next())?;
                let entries = match self.read_resolved(frame, local, &path)? {
                    MirValue::Vec(entries) => entries,
                    other => return self.internal(format!("HashMap referent is {other:?}")),
                };
                for (i, entry) in entries.iter().enumerate() {
                    if let MirValue::Aggregate(kv) = entry {
                        if kv[0] == key {
                            let mut elem_path = path;
                            elem_path.push(ConcreteProj::Index(i));
                            elem_path.push(ConcreteProj::Field(1));
                            return Ok(MirValue::Enum {
                                variant: 1,
                                fields: vec![MirValue::Ref {
                                    frame,
                                    generation,
                                    local,
                                    path: elem_path,
                                }],
                            });
                        }
                    }
                }
                Ok(MirValue::Enum {
                    variant: 0,
                    fields: Vec::new(),
                })
            }
            HashMapLen => {
                let recv = args.next();
                Ok(MirValue::Int(self.read_vec_ref(&recv)?.len() as i128))
            }
            HashMapIsEmpty => {
                let recv = args.next();
                Ok(MirValue::Bool(self.read_vec_ref(&recv)?.is_empty()))
            }
            HashMapContainsKey => {
                let recv = args.next();
                let key = self.deref_key_arg(args.next())?;
                let entries = self.read_vec_ref(&recv)?;
                let found = entries
                    .iter()
                    .any(|e| matches!(e, MirValue::Aggregate(kv) if kv[0] == key));
                Ok(MirValue::Bool(found))
            }
            HashMapKeysIterNew => {
                // A TRUE borrowed cursor: [map-ref, cursor] — Next indexes the live map.
                let map_ref = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("KeysIterNew missing receiver".into()))?;
                Ok(MirValue::Aggregate(vec![map_ref, MirValue::Int(0)]))
            }
            HashMapKeysIterNext => {
                let Some(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }) = args.next()
                else {
                    return self.internal("KeysIterNext expects a &mut iterator reference");
                };
                self.check_ref_live(frame, generation)?;
                let iter_value = self.read_resolved(frame, local, &path)?;
                let MirValue::Aggregate(fields) = &iter_value else {
                    return self.internal(format!("keys iterator referent is {iter_value:?}"));
                };
                let (map_ref, cursor) = match (fields.first(), fields.get(1)) {
                    (Some(r @ MirValue::Ref { .. }), Some(MirValue::Int(c))) => (r.clone(), *c),
                    other => {
                        return self.internal(format!("malformed keys-iterator state {other:?}"))
                    }
                };
                let MirValue::Ref {
                    frame: mf,
                    generation: mg,
                    local: ml,
                    path: mp,
                } = map_ref
                else {
                    unreachable!("matched Ref above");
                };
                self.check_ref_live(mf, mg)?;
                let len = match self.read_resolved(mf, ml, &mp)? {
                    MirValue::Vec(entries) => entries.len(),
                    other => return self.internal(format!("HashMap referent is {other:?}")),
                };
                if (cursor as usize) < len {
                    let mut cursor_path = path;
                    cursor_path.push(ConcreteProj::Field(1));
                    self.write_resolved(frame, local, &cursor_path, MirValue::Int(cursor + 1))?;
                    let mut key_path = mp;
                    key_path.push(ConcreteProj::Index(cursor as usize));
                    key_path.push(ConcreteProj::Field(0));
                    Ok(MirValue::Enum {
                        variant: 1,
                        fields: vec![MirValue::Ref {
                            frame: mf,
                            generation: mg,
                            local: ml,
                            path: key_path,
                        }],
                    })
                } else {
                    Ok(MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    })
                }
            }
            other => self.internal(format!("runtime {other:?} is not a HashMap op")),
        }
    }

    /// A `&K` key argument, dereferenced to the key value for structural comparison.
    /// 0.1-A6 (A4 slicing): slice ops. A `&[T]` value is a `Ref` whose path ends with a
    /// `ConcreteProj::Slice` window; `SliceNew` composes windows (re-slicing never stacks two
    /// Slice steps) and TRAPS IndexOutOfBounds on a negative, inverted, or out-of-range bound
    /// with the CALL SITE's provenance, per the 06-Standard-Library behavioral requirement.
    fn run_slice_runtime(
        &mut self,
        rt: RuntimeFn,
        args: Vec<MirValue>,
        call_info: SourceInfo,
    ) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        let mut args = args.into_iter();
        let Some(MirValue::Ref {
            frame,
            generation,
            local,
            path,
        }) = args.next()
        else {
            return self.internal("slice op expects a reference receiver");
        };
        self.check_ref_live(frame, generation)?;
        match rt {
            SliceNew => {
                let oob = || MirRunError::Trap {
                    category: TrapCategory::IndexOutOfBounds,
                    source: call_info,
                    message: None,
                };
                let int = |v: Option<MirValue>| -> Result<i128, MirRunError> {
                    match v {
                        Some(MirValue::Int(i)) => Ok(i),
                        other => Err(MirRunError::Internal(format!(
                            "SliceNew bound is not an integer: {other:?}"
                        ))),
                    }
                };
                let lo = int(args.next())?;
                let hi = int(args.next())?;
                let inclusive = match args.next() {
                    Some(MirValue::Bool(b)) => b,
                    other => {
                        return self
                            .internal(format!("SliceNew inclusive flag is not Bool: {other:?}"))
                    }
                };
                if lo < 0 || hi < 0 {
                    return Err(oob());
                }
                let start = lo as usize;
                let end = if inclusive {
                    hi as usize + 1
                } else {
                    hi as usize
                };
                // Window base: an existing Slice tail composes; otherwise the referent's length.
                let (parent_path, window_start, base_len) = match path.last() {
                    Some(ConcreteProj::Slice { start: s0, len: l0 }) => {
                        (path[..path.len() - 1].to_vec(), *s0, *l0)
                    }
                    _ => {
                        let len = match self.read_resolved(frame, local, &path)? {
                            MirValue::Vec(elems) => elems.len(),
                            MirValue::Aggregate(elems) => elems.len(),
                            other => return self.internal(format!("sliced referent is {other:?}")),
                        };
                        (path, 0, len)
                    }
                };
                if start > end || end > base_len {
                    return Err(oob());
                }
                let mut new_path = parent_path;
                new_path.push(ConcreteProj::Slice {
                    start: window_start + start,
                    len: end - start,
                });
                Ok(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path: new_path,
                })
            }
            SliceLen | SliceIsEmpty => {
                let len = match path.last() {
                    Some(ConcreteProj::Slice { len, .. }) => *len,
                    other => {
                        return self.internal(format!(
                            "slice receiver has no view window (path tail {other:?})"
                        ))
                    }
                };
                Ok(if matches!(rt, SliceLen) {
                    MirValue::Int(len as i128)
                } else {
                    MirValue::Bool(len == 0)
                })
            }
            other => self.internal(format!("runtime {other:?} is not a slice op")),
        }
    }

    fn deref_key_arg(&self, v: Option<MirValue>) -> Result<MirValue, MirRunError> {
        match v {
            Some(MirValue::Ref {
                frame,
                generation,
                local,
                path,
            }) => {
                self.check_ref_live(frame, generation)?;
                self.read_resolved(frame, local, &path)
            }
            Some(other) => Ok(other),
            None => self.internal("missing key argument"),
        }
    }

    fn read_vec_ref(&self, v: &Option<MirValue>) -> Result<Vec<MirValue>, MirRunError> {
        match v {
            Some(MirValue::Ref {
                frame,
                generation,
                local,
                path,
            }) => {
                self.check_ref_live(*frame, *generation)?;
                match self.read_resolved(*frame, *local, path)? {
                    MirValue::Vec(elems) => Ok(elems),
                    other => self.internal(format!("Vec ref referent is {other:?}")),
                }
            }
            Some(MirValue::Vec(elems)) => Ok(elems.clone()),
            other => self.internal(format!("expected a &Vec argument, got {other:?}")),
        }
    }

    /// Mutate the `Vec` behind a `&mut Vec` reference argument in place, returning the
    /// closure's result.
    fn mutate_vec_ref<R>(
        &mut self,
        v: &Option<MirValue>,
        f: impl FnOnce(&mut Vec<MirValue>) -> R,
    ) -> Result<R, MirRunError> {
        let Some(MirValue::Ref {
            frame,
            generation,
            local,
            path,
        }) = v
        else {
            return self.internal(format!("expected a &mut Vec argument, got {v:?}"));
        };
        self.check_ref_live(*frame, *generation)?;
        let (frame, local, path) = (*frame, *local, path.clone());
        let mut vec = match self.read_resolved(frame, local, &path)? {
            MirValue::Vec(elems) => elems,
            other => return self.internal(format!("&mut Vec referent is {other:?}")),
        };
        let out = f(&mut vec);
        self.write_resolved(frame, local, &path, MirValue::Vec(vec))?;
        Ok(out)
    }

    /// The content of a `&str` argument (a `Str` value; a `String`/`Ref` is a lowering bug).
    fn as_str(&self, v: &Option<MirValue>) -> Result<std::rc::Rc<str>, MirRunError> {
        match v {
            Some(MirValue::Str(s)) => Ok(s.clone()),
            other => self.internal(format!("expected a &str argument, got {other:?}")),
        }
    }

    /// Resolve a `&String`/`&mut String` reference argument to a snapshot of the referent.
    fn read_string_ref(&self, v: &Option<MirValue>) -> Result<String, MirRunError> {
        match v {
            Some(MirValue::Ref {
                frame,
                generation,
                local,
                path,
            }) => {
                self.check_ref_live(*frame, *generation)?;
                match self.read_resolved(*frame, *local, path)? {
                    MirValue::String(s) => Ok(s),
                    MirValue::Str(s) => Ok(s.to_string()),
                    other => self.internal(format!("String ref referent is {other:?}")),
                }
            }
            Some(MirValue::String(s)) => Ok(s.clone()),
            other => self.internal(format!("expected a &String argument, got {other:?}")),
        }
    }

    /// Mutate the `String` behind a `&mut String` reference argument in place.
    fn mutate_string_ref(
        &mut self,
        v: &Option<MirValue>,
        f: impl FnOnce(&mut String),
    ) -> Result<(), MirRunError> {
        let Some(MirValue::Ref {
            frame,
            generation,
            local,
            path,
        }) = v
        else {
            return self.internal(format!("expected a &mut String argument, got {v:?}"));
        };
        self.check_ref_live(*frame, *generation)?;
        let (frame, local, path) = (*frame, *local, path.clone());
        let mut s = match self.read_resolved(frame, local, &path)? {
            MirValue::String(s) => s,
            other => return self.internal(format!("&mut String referent is {other:?}")),
        };
        f(&mut s);
        self.write_resolved(frame, local, &path, MirValue::String(s))
    }

    /// Resolve a `&str` operand (a `Str` value) to its content — used for `Trap.message`.
    fn eval_str_operand(&mut self, here: usize, op: &Operand) -> Result<String, MirRunError> {
        match self.eval_operand(here, op)? {
            MirValue::Str(s) => Ok(s.to_string()),
            MirValue::String(s) => Ok(s),
            other => self.internal(format!("trap message operand is {other:?}")),
        }
    }
}

fn is_vec_runtime(rt: RuntimeFn) -> bool {
    use RuntimeFn::*;
    matches!(
        rt,
        VecNew
            | VecWithCapacity
            | VecPush
            | VecPop
            | VecLen
            | VecIsEmpty
            | VecIndexGet
            | VecReplace
            | VecRemove
            | VecClear
            | VecIterNew
            | VecIterNext
            | VecGetRef
            | VecGetMutRef
    )
}

fn is_slice_runtime(rt: RuntimeFn) -> bool {
    use RuntimeFn::*;
    matches!(rt, SliceNew | SliceLen | SliceIsEmpty)
}

fn is_map_runtime(rt: RuntimeFn) -> bool {
    use RuntimeFn::*;
    matches!(
        rt,
        HashMapNew
            | HashMapInsert
            | HashMapGet
            | HashMapLen
            | HashMapIsEmpty
            | HashMapContainsKey
            | HashMapKeysIterNew
            | HashMapKeysIterNext
    )
}

/// A Char argument (a Unicode scalar codepoint carried as `MirValue::Int`).
fn char_of(v: &Option<MirValue>) -> Result<char, MirRunError> {
    match v {
        Some(MirValue::Int(cp)) => u32::try_from(*cp)
            .ok()
            .and_then(char::from_u32)
            .ok_or_else(|| MirRunError::Internal(format!("invalid Char codepoint {cp}"))),
        other => Err(MirRunError::Internal(format!(
            "expected a Char argument, got {other:?}"
        ))),
    }
}

fn int_arg(v: Option<MirValue>) -> Result<i128, MirRunError> {
    match v {
        Some(MirValue::Int(i)) => Ok(i),
        other => Err(MirRunError::Internal(format!(
            "expected an integer argument, got {other:?}"
        ))),
    }
}

/// Wrap an optional element as a `MirValue` `Option` enum (CoreOption: v0 = None, v1 = Some).
fn option_value(v: Option<MirValue>) -> MirValue {
    match v {
        Some(inner) => MirValue::Enum {
            variant: 1,
            fields: vec![inner],
        },
        None => MirValue::Enum {
            variant: 0,
            fields: Vec::new(),
        },
    }
}

/// Outcome of a checked/trapping primitive (A5). `Trap(None)` traps with the terminator's own
/// category; `Trap(Some(cat))` overrides it — a shift with a bad count is `InvalidShift` even
/// though the terminator's default category is `IntegerOverflow`.
enum CheckedOutcome {
    Value(MirValue),
    Trap(Option<TrapCategory>),
}

impl From<Option<MirValue>> for CheckedOutcome {
    fn from(opt: Option<MirValue>) -> Self {
        match opt {
            Some(v) => CheckedOutcome::Value(v),
            None => CheckedOutcome::Trap(None),
        }
    }
}

/// Bit width of an integer MIR type (A5, for the NUM-SHIFT-001 count bound).
fn int_width(ty: &MirTy) -> Option<u32> {
    Some(match ty {
        MirTy::Int8 | MirTy::UInt8 => 8,
        MirTy::Int16 | MirTy::UInt16 => 16,
        MirTy::Int32 | MirTy::UInt32 => 32,
        MirTy::Int64 | MirTy::UInt64 => 64,
        _ => return None,
    })
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
