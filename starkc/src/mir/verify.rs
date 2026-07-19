//! WP-C4.3 — MIR verifier.
//!
//! Implements the contract's §10 obligations (mir.md, APPROVED CD-028) over a `MirProgram`.
//! Invalid MIR produces `MIR-xxxx` **compiler-internal** diagnostics (charter §5.1 namespace —
//! these are never user-source errors; reaching this point with invalid MIR means the lowering
//! has a bug) and fails safely: the verifier returns every violation it finds, and no backend
//! may consume a program that failed verification.
//!
//! Obligation → code map (first allocation of the MIR-xxxx namespace, recorded in
//! COMPILER-STATE.md per charter §5.1):
//!
//! ```text
//! V-CFG-1/2  MIR-0001 (block target out of bounds), MIR-0002 (local/place out of bounds)
//! V-TY-1     MIR-0004 (assignment type mismatch), MIR-0005 (call/checked signature mismatch)
//! V-TY-2     structural: MirTy has no Param/Infer variants — nothing to check at runtime
//! V-TY-3     MIR-0006 (bare unsized type outside Ref)
//! V-MOVE-1   MIR-0007 (use of a possibly-moved place; conservative whole-local dataflow)
//! V-DISC-1   MIR-0008 (discriminant/variant-field/switch misuse)
//! V-DROP-1/2 MIR-0009 (drop on non-droppable / drop-flag discipline violation)
//! V-IDX-1/2  MIR-0010 (index-proof discipline violation)
//! V-FN-1     MIR-0011 (arithmetic/comparison on a FnPtr operand)
//! V-RT-1     MIR-0012 (runtime callee signature mismatch)
//! V-SRC-1    MIR-0013 (SourceInfo missing a valid FileId)
//! MIR-0003   projection type mismatch (V-CFG-2's "projections type-correct step by step")
//! ```
//!
//! Scope notes (v0.1, honest limitations — refinements, not silent gaps):
//! - V-MOVE-1 uses whole-local granularity with any-path (union) joins to a fixpoint: a `Move`
//!   out of a projected place conservatively marks the whole local moved. Reinitialization by
//!   whole-local assignment clears it. This can reject over-clever (but legal) MIR; it cannot
//!   accept a genuinely moved-from read. Field-precise tracking is a later refinement.
//! - Drop-flag "read only by SwitchInt" is enforced as: DropFlag locals never appear in any
//!   statement rvalue other than their own `Const(true/false)` initialization, and never as
//!   call/checked arguments; SwitchInt reads are permitted.

use super::*;
use std::collections::{BTreeSet, VecDeque};

#[derive(Debug, Clone)]
pub struct MirError {
    pub code: &'static str,
    pub message: String,
    pub symbol: String,
    pub block: u32,
}

/// Proof-of-verification token (review correction: "no backend bypasses MIR validation" as an
/// API property, not just a roadmap rule). The only way to obtain one is `verify_program`
/// succeeding; `mir::interp::run_program` — and eventually the generated-Rust backend —
/// consume this wrapper, never a raw `&MirProgram`.
#[derive(Clone, Copy)]
pub struct VerifiedMirProgram<'a> {
    program: &'a MirProgram,
}

impl<'a> VerifiedMirProgram<'a> {
    pub fn program(&self) -> &'a MirProgram {
        self.program
    }
}

pub fn verify_program(program: &MirProgram) -> Result<VerifiedMirProgram<'_>, Vec<MirError>> {
    let mut errors = Vec::new();
    for body in &program.bodies {
        let mut cx = BodyCx {
            program,
            body,
            errors: &mut errors,
        };
        cx.verify();
    }
    if errors.is_empty() {
        Ok(VerifiedMirProgram { program })
    } else {
        Err(errors)
    }
}

struct BodyCx<'a> {
    program: &'a MirProgram,
    body: &'a MirBody,
    errors: &'a mut Vec<MirError>,
}

impl<'a> BodyCx<'a> {
    fn err(&mut self, code: &'static str, block: u32, message: impl Into<String>) {
        self.errors.push(MirError {
            code,
            message: message.into(),
            symbol: self.body.instance.symbol.clone(),
            block,
        });
    }

    fn verify(&mut self) {
        self.verify_locals_wellformed();
        for bi in 0..self.body.blocks.len() {
            self.verify_block(bi as u32);
        }
        self.verify_moves();
        self.verify_index_proofs();
        self.verify_index_bindings();
    }

    // ---- V-TY-3 + local sanity ----

    fn verify_locals_wellformed(&mut self) {
        for (i, decl) in self.body.locals.iter().enumerate() {
            if bare_unsized(&decl.ty) {
                self.err(
                    "MIR-0006",
                    0,
                    format!("local _{i} has a bare unsized type {:?}", decl.ty),
                );
            }
            if matches!(decl.kind, LocalKind::DropFlag) && decl.ty != MirTy::Bool {
                self.err("MIR-0009", 0, format!("drop-flag local _{i} is not Bool"));
            }
        }
        if (self.body.entry.0 as usize) >= self.body.blocks.len() {
            self.err("MIR-0001", 0, "entry block out of bounds");
        }
    }

    // ---- per-block checks ----

    fn verify_block(&mut self, bi: u32) {
        let block = &self.body.blocks[bi as usize];

        for (stmt, info) in &block.statements {
            self.verify_source(info, bi);
            if let Statement::Assign(place, rvalue) = stmt {
                self.verify_assign(place, rvalue, bi);
            }
        }
        let (term, info) = &block.terminator;
        self.verify_source(info, bi);
        self.verify_terminator(term, bi);
    }

    fn verify_source(&mut self, info: &SourceInfo, bi: u32) {
        if (info.file.0 as usize) >= self.program.files.len() {
            self.err("MIR-0013", bi, "SourceInfo carries an invalid FileId");
        }
    }

    fn check_target(&mut self, target: BlockId, bi: u32) {
        if (target.0 as usize) >= self.body.blocks.len() {
            self.err(
                "MIR-0001",
                bi,
                format!("target bb{} out of bounds", target.0),
            );
        }
    }

    // ---- typing ----

    fn local_ty(&mut self, local: LocalId, bi: u32) -> Option<MirTy> {
        match self.body.locals.get(local.0 as usize) {
            Some(decl) => Some(decl.ty.clone()),
            None => {
                self.err("MIR-0002", bi, format!("local _{} out of bounds", local.0));
                None
            }
        }
    }

    fn place_ty(&mut self, place: &Place, bi: u32) -> Option<MirTy> {
        let mut ty = self.local_ty(place.local, bi)?;
        for projection in &place.projection {
            ty = match (projection, ty) {
                (Projection::Field(i), MirTy::Struct(item, _)) => {
                    match self
                        .program
                        .types
                        .struct_fields
                        .get(&item.0)
                        .and_then(|fields| fields.get(*i as usize))
                    {
                        Some(f) => f.clone(),
                        None => {
                            self.err(
                                "MIR-0003",
                                bi,
                                format!("struct field .{i} unresolvable for item #{}", item.0),
                            );
                            return None;
                        }
                    }
                }
                (Projection::Field(i), MirTy::Tuple(elems)) => match elems.get(*i as usize) {
                    Some(e) => e.clone(),
                    None => {
                        self.err("MIR-0003", bi, format!("tuple field .{i} out of bounds"));
                        return None;
                    }
                },
                (Projection::VariantField(v, i), MirTy::Enum(enum_ref, args)) => {
                    match variant_payload(self.program, &enum_ref, &args, *v)
                        .and_then(|payload| payload.get(*i as usize).cloned())
                    {
                        Some(f) => f,
                        None => {
                            self.err(
                                "MIR-0008",
                                bi,
                                format!("variant field v{v}.{i} unresolvable"),
                            );
                            return None;
                        }
                    }
                }
                (Projection::VariantField(..), other) => {
                    self.err(
                        "MIR-0008",
                        bi,
                        format!("variant-field projection on non-enum type {other:?}"),
                    );
                    return None;
                }
                (Projection::Deref, MirTy::Ref { inner, .. }) => *inner,
                (Projection::Index(proof), MirTy::Array(elem, _))
                | (Projection::Index(proof), MirTy::Slice(elem)) => {
                    match self.body.locals.get(proof.0 as usize) {
                        Some(decl) if matches!(decl.kind, LocalKind::IndexProof) => *elem,
                        Some(_) => {
                            self.err(
                                "MIR-0010",
                                bi,
                                format!("Index projection consumes _{} which is not an IndexProof local", proof.0),
                            );
                            return None;
                        }
                        None => {
                            self.err(
                                "MIR-0002",
                                bi,
                                format!("proof local _{} out of bounds", proof.0),
                            );
                            return None;
                        }
                    }
                }
                (projection, other) => {
                    self.err(
                        "MIR-0003",
                        bi,
                        format!("projection {projection:?} does not apply to type {other:?}"),
                    );
                    return None;
                }
            };
        }
        Some(ty)
    }

    fn operand_ty(&mut self, operand: &Operand, bi: u32) -> Option<MirTy> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.place_ty(place, bi),
            Operand::Const(constant) => Some(match constant {
                Constant::Int(_, ty) | Constant::Float(_, ty) => ty.clone(),
                Constant::Bool(_) => MirTy::Bool,
                Constant::Unit => MirTy::Unit,
                Constant::FnPtr(instance) => match self.instance_sig(&instance.symbol) {
                    Some((params, ret)) => MirTy::FnPtr {
                        params,
                        ret: Box::new(ret),
                    },
                    None => {
                        self.err(
                            "MIR-0005",
                            bi,
                            format!(
                                "FnPtr constant references undiscovered instance {}",
                                instance.symbol
                            ),
                        );
                        return None;
                    }
                },
            }),
        }
    }

    fn instance_sig(&self, symbol: &str) -> Option<(Vec<MirTy>, MirTy)> {
        self.program
            .bodies
            .iter()
            .find(|b| b.instance.symbol == symbol)
            .map(|b| (b.params.clone(), b.ret.clone()))
    }

    fn verify_assign(&mut self, place: &Place, rvalue: &Rvalue, bi: u32) {
        // Drop-flag discipline (V-DROP-2 write half): only Const(true/false) may be written.
        if place.projection.is_empty() {
            if let Some(decl) = self.body.locals.get(place.local.0 as usize) {
                if matches!(decl.kind, LocalKind::DropFlag)
                    && !matches!(rvalue, Rvalue::Use(Operand::Const(Constant::Bool(_))))
                {
                    self.err(
                        "MIR-0009",
                        bi,
                        format!("drop-flag _{} written with a non-constant", place.local.0),
                    );
                }
                if matches!(decl.kind, LocalKind::IndexProof) {
                    self.err(
                        "MIR-0010",
                        bi,
                        format!(
                            "IndexProof local _{} assigned by a statement",
                            place.local.0
                        ),
                    );
                }
            }
        }
        let Some(lhs_ty) = self.place_ty(place, bi) else {
            return;
        };
        self.check_rvalue_against(&lhs_ty, rvalue, bi);
    }

    fn check_rvalue_against(&mut self, expected: &MirTy, rvalue: &Rvalue, bi: u32) {
        match rvalue {
            Rvalue::Use(op) => {
                self.check_no_fn_misuse_ok(op);
                if let Some(ty) = self.operand_ty(op, bi) {
                    self.expect_ty(expected, &ty, "assignment", bi);
                }
            }
            Rvalue::UnOp(op, operand) => {
                let Some(ty) = self.operand_ty(operand, bi) else {
                    return;
                };
                if matches!(ty, MirTy::FnPtr { .. }) {
                    self.err("MIR-0011", bi, "unary operation on a FnPtr operand");
                    return;
                }
                match op {
                    MirUnOp::Not => {
                        if ty != MirTy::Bool {
                            self.err("MIR-0004", bi, format!("Not on non-Bool {ty:?}"));
                        }
                        self.expect_ty(expected, &MirTy::Bool, "Not result", bi);
                    }
                    MirUnOp::FloatNeg => {
                        if !matches!(ty, MirTy::Float32 | MirTy::Float64) {
                            self.err("MIR-0004", bi, format!("FloatNeg on non-float {ty:?}"));
                        }
                        self.expect_ty(expected, &ty, "FloatNeg result", bi);
                    }
                }
            }
            Rvalue::BinOp(op, lhs, rhs) => {
                let (Some(lt), Some(rt)) = (self.operand_ty(lhs, bi), self.operand_ty(rhs, bi))
                else {
                    return;
                };
                if matches!(lt, MirTy::FnPtr { .. }) || matches!(rt, MirTy::FnPtr { .. }) {
                    self.err(
                        "MIR-0011",
                        bi,
                        "binary operation on a FnPtr operand (TYPE-FN-001)",
                    );
                    return;
                }
                if lt != rt {
                    self.err(
                        "MIR-0004",
                        bi,
                        format!("BinOp operand types differ: {lt:?} vs {rt:?}"),
                    );
                }
                match op {
                    MirBinOp::Eq
                    | MirBinOp::Ne
                    | MirBinOp::Lt
                    | MirBinOp::Le
                    | MirBinOp::Gt
                    | MirBinOp::Ge => self.expect_ty(expected, &MirTy::Bool, "comparison", bi),
                    MirBinOp::FloatAdd | MirBinOp::FloatSub | MirBinOp::FloatMul => {
                        if !matches!(lt, MirTy::Float32 | MirTy::Float64) {
                            self.err("MIR-0004", bi, format!("float BinOp on {lt:?}"));
                        }
                        self.expect_ty(expected, &lt, "float arithmetic", bi);
                    }
                }
            }
            Rvalue::Aggregate(kind, operands) => self.check_aggregate(expected, kind, operands, bi),
            Rvalue::Discriminant(place) => {
                match self.place_ty(place, bi) {
                    Some(MirTy::Enum(..)) => {}
                    Some(other) => {
                        self.err(
                            "MIR-0008",
                            bi,
                            format!("Discriminant of non-enum type {other:?}"),
                        );
                    }
                    None => {}
                }
                self.expect_ty(expected, &MirTy::Int64, "discriminant", bi);
            }
            Rvalue::RefOf { mutable, place } => {
                if let Some(inner) = self.place_ty(place, bi) {
                    let ref_ty = MirTy::Ref {
                        mutable: *mutable,
                        inner: Box::new(inner),
                    };
                    self.expect_ty(expected, &ref_ty, "reference", bi);
                }
            }
        }
    }

    fn check_aggregate(&mut self, expected: &MirTy, kind: &AggKind, operands: &[Operand], bi: u32) {
        let operand_tys: Vec<Option<MirTy>> =
            operands.iter().map(|op| self.operand_ty(op, bi)).collect();
        match (kind, expected) {
            (AggKind::Tuple, MirTy::Tuple(elems)) => {
                if elems.len() != operands.len() {
                    self.err("MIR-0004", bi, "tuple aggregate arity mismatch");
                    return;
                }
                for (e, t) in elems.iter().zip(&operand_tys) {
                    if let Some(t) = t {
                        self.expect_ty(e, t, "tuple field", bi);
                    }
                }
            }
            (AggKind::Array(elem), MirTy::Array(expected_elem, len)) => {
                if **expected_elem != *elem {
                    self.err("MIR-0004", bi, "array aggregate element type mismatch");
                }
                if *len as usize != operands.len() {
                    self.err("MIR-0004", bi, "array aggregate length mismatch");
                }
                for t in operand_tys.iter().flatten() {
                    self.expect_ty(elem, t, "array element", bi);
                }
            }
            (AggKind::Struct(item), MirTy::Struct(expected_item, _)) => {
                if item != expected_item {
                    self.err("MIR-0004", bi, "struct aggregate item mismatch");
                    return;
                }
                match self.program.types.struct_fields.get(&item.0) {
                    Some(fields) => {
                        if fields.len() != operands.len() {
                            self.err("MIR-0004", bi, "struct aggregate field-count mismatch");
                            return;
                        }
                        for (f, t) in fields.clone().iter().zip(&operand_tys) {
                            if let Some(t) = t {
                                self.expect_ty(f, t, "struct field", bi);
                            }
                        }
                    }
                    None => self.err(
                        "MIR-0003",
                        bi,
                        format!("struct #{} has no type-context entry", item.0),
                    ),
                }
            }
            (AggKind::EnumVariant(enum_ref, variant), MirTy::Enum(expected_ref, args)) => {
                if enum_ref != expected_ref {
                    self.err("MIR-0008", bi, "enum aggregate nominal mismatch");
                    return;
                }
                match variant_payload(self.program, enum_ref, args, *variant) {
                    Some(payload) => {
                        if payload.len() != operands.len() {
                            self.err("MIR-0008", bi, "enum variant payload arity mismatch");
                            return;
                        }
                        for (f, t) in payload.iter().zip(&operand_tys) {
                            if let Some(t) = t {
                                self.expect_ty(f, t, "variant payload", bi);
                            }
                        }
                    }
                    None => self.err(
                        "MIR-0008",
                        bi,
                        format!("variant v{variant} unresolvable for {enum_ref:?}"),
                    ),
                }
            }
            (kind, other) => self.err(
                "MIR-0004",
                bi,
                format!("aggregate {kind:?} assigned to incompatible type {other:?}"),
            ),
        }
    }

    fn expect_ty(&mut self, expected: &MirTy, actual: &MirTy, what: &str, bi: u32) {
        if expected != actual && *actual != MirTy::Never {
            self.err(
                "MIR-0004",
                bi,
                format!("{what}: expected {expected:?}, found {actual:?}"),
            );
        }
    }

    fn check_no_fn_misuse_ok(&self, _op: &Operand) {
        // FnPtr operands are fine in Use/Call positions; misuse checks live in UnOp/BinOp.
    }

    // ---- terminators ----

    fn verify_terminator(&mut self, term: &Terminator, bi: u32) {
        match term {
            Terminator::Goto { target } => self.check_target(*target, bi),
            Terminator::SwitchInt {
                scrut,
                arms,
                otherwise,
            } => {
                for (_, target) in arms {
                    self.check_target(*target, bi);
                }
                self.check_target(*otherwise, bi);
                let Some(ty) = self.operand_ty(scrut, bi) else {
                    return;
                };
                match ty {
                    MirTy::Bool
                    | MirTy::Int8
                    | MirTy::Int16
                    | MirTy::Int32
                    | MirTy::Int64
                    | MirTy::UInt8
                    | MirTy::UInt16
                    | MirTy::UInt32
                    | MirTy::UInt64 => {}
                    other => self.err(
                        "MIR-0004",
                        bi,
                        format!("SwitchInt scrutinee is non-integer {other:?}"),
                    ),
                }
            }
            Terminator::Call {
                callee,
                args,
                dest,
                target,
            } => {
                self.check_target(*target, bi);
                let arg_tys: Vec<Option<MirTy>> =
                    args.iter().map(|a| self.operand_ty(a, bi)).collect();
                let dest_ty = self.place_ty(dest, bi);
                let sig: Option<(Vec<MirTy>, MirTy)> = match callee {
                    Callee::Instance(instance) => {
                        let sig = self.instance_sig(&instance.symbol);
                        if sig.is_none() {
                            self.err(
                                "MIR-0005",
                                bi,
                                format!("call to undiscovered instance {}", instance.symbol),
                            );
                        }
                        sig
                    }
                    Callee::FnValue(op) => match self.operand_ty(op, bi) {
                        Some(MirTy::FnPtr { params, ret }) => Some((params, *ret)),
                        Some(other) => {
                            self.err(
                                "MIR-0005",
                                bi,
                                format!("indirect call through non-FnPtr type {other:?}"),
                            );
                            None
                        }
                        None => None,
                    },
                    Callee::Runtime(rt) => Some(runtime_sig(*rt)),
                };
                if let Some((params, ret)) = sig {
                    if params.len() != args.len() {
                        self.err("MIR-0005", bi, "call arity mismatch");
                    } else {
                        for (p, t) in params.iter().zip(&arg_tys) {
                            if let Some(t) = t {
                                if p != t && *t != MirTy::Never {
                                    self.err(
                                        "MIR-0005",
                                        bi,
                                        format!("call argument: expected {p:?}, found {t:?}"),
                                    );
                                }
                            }
                        }
                    }
                    if let Some(dest_ty) = dest_ty {
                        if dest_ty != ret && ret != MirTy::Never {
                            self.err(
                                "MIR-0005",
                                bi,
                                format!("call dest {dest_ty:?} vs return {ret:?}"),
                            );
                        }
                    }
                }
            }
            Terminator::Drop { place, target } => {
                self.check_target(*target, bi);
                if let Some(ty) = self.place_ty(place, bi) {
                    if !may_need_drop(&ty) {
                        self.err(
                            "MIR-0009",
                            bi,
                            format!("Drop of a type that can never need dropping: {ty:?}"),
                        );
                    }
                }
            }
            Terminator::Checked {
                op,
                args,
                dest,
                target,
                ..
            } => {
                self.check_target(*target, bi);
                let arg_tys: Vec<Option<MirTy>> =
                    args.iter().map(|a| self.operand_ty(a, bi)).collect();
                let Some(dest_ty) = self.local_ty(*dest, bi) else {
                    return;
                };
                let dest_is_proof = matches!(
                    self.body.locals.get(dest.0 as usize).map(|d| &d.kind),
                    Some(LocalKind::IndexProof)
                );
                match op {
                    CheckedOp::CheckIndex => {
                        if !dest_is_proof {
                            self.err(
                                "MIR-0010",
                                bi,
                                "CheckIndex dest must be an IndexProof local",
                            );
                        }
                        if args.len() != 2 {
                            self.err("MIR-0005", bi, "CheckIndex expects (base, index)");
                        } else {
                            match &args[0] {
                                Operand::Copy(base_place) => {
                                    match self.place_ty(base_place, bi) {
                                        Some(MirTy::Array(..) | MirTy::Slice(_)) | None => {}
                                        Some(other) => self.err(
                                            "MIR-0010",
                                            bi,
                                            format!("CheckIndex base has non-indexable type {other:?}"),
                                        ),
                                    }
                                }
                                _ => self.err(
                                    "MIR-0010",
                                    bi,
                                    "CheckIndex base must be Copy(place) so the proof can bind base identity",
                                ),
                            }
                            if let Some(t) = &arg_tys[1] {
                                if !is_integer(t) {
                                    self.err("MIR-0010", bi, "CheckIndex index must be an integer");
                                }
                            }
                        }
                    }
                    CheckedOp::Cast => {
                        if dest_is_proof {
                            self.err("MIR-0010", bi, "Cast may not define an IndexProof local");
                        }
                        if args.len() != 1 {
                            self.err("MIR-0005", bi, "Cast expects one operand");
                        }
                        for t in arg_tys.iter().flatten() {
                            if !is_numeric(t) {
                                self.err("MIR-0004", bi, format!("Cast of non-numeric {t:?}"));
                            }
                        }
                        if !is_numeric(&dest_ty) {
                            self.err("MIR-0004", bi, "Cast to non-numeric type");
                        }
                    }
                    CheckedOp::FloatDiv | CheckedOp::FloatRem => {
                        if dest_is_proof {
                            self.err("MIR-0010", bi, "Checked op may not define an IndexProof");
                        }
                        for t in arg_tys.iter().flatten() {
                            self.expect_ty(&dest_ty, t, "checked float operand", bi);
                        }
                        if !matches!(dest_ty, MirTy::Float32 | MirTy::Float64) {
                            self.err("MIR-0004", bi, "checked float op on non-float dest");
                        }
                    }
                    CheckedOp::Neg => {
                        if dest_is_proof {
                            self.err("MIR-0010", bi, "Checked op may not define an IndexProof");
                        }
                        if args.len() != 1 {
                            self.err("MIR-0005", bi, "Neg expects one operand");
                        }
                        for t in arg_tys.iter().flatten() {
                            self.expect_ty(&dest_ty, t, "checked neg operand", bi);
                        }
                        if !is_integer(&dest_ty) {
                            self.err("MIR-0004", bi, "checked Neg on non-integer dest");
                        }
                    }
                    _ => {
                        if dest_is_proof {
                            self.err("MIR-0010", bi, "Checked op may not define an IndexProof");
                        }
                        if args.len() != 2 {
                            self.err("MIR-0005", bi, "checked binary op expects two operands");
                        }
                        for t in arg_tys.iter().flatten() {
                            if matches!(t, MirTy::FnPtr { .. }) {
                                self.err("MIR-0011", bi, "checked op on a FnPtr operand");
                                continue;
                            }
                            self.expect_ty(&dest_ty, t, "checked operand", bi);
                        }
                        if !is_integer(&dest_ty) {
                            self.err("MIR-0004", bi, "checked integer op on non-integer dest");
                        }
                    }
                }
            }
            Terminator::Trap { .. } | Terminator::Return | Terminator::Unreachable => {}
        }
    }

    // ---- V-MOVE-1: conservative any-path moved-local dataflow to a fixpoint ----

    fn verify_moves(&mut self) {
        let block_count = self.body.blocks.len();
        // moved_in[b]: set of locals possibly moved on entry to b.
        let mut moved_in: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); block_count];
        let mut work: VecDeque<usize> = VecDeque::new();
        work.push_back(self.body.entry.0 as usize);
        let mut visited = vec![false; block_count];
        visited[self.body.entry.0 as usize] = true;

        // Fixpoint over the union lattice.
        while let Some(bi) = work.pop_front() {
            let (moved_out, successors) = self.flow_block(bi as u32, moved_in[bi].clone(), false);
            for succ in successors {
                let s = succ.0 as usize;
                if s >= block_count {
                    // Broken edge — already reported as MIR-0001 by the CFG check. The verifier
                    // must fail safely on invalid MIR, never panic (contract §10).
                    continue;
                }
                let before = moved_in[s].len();
                moved_in[s].extend(moved_out.iter().copied());
                if !visited[s] || moved_in[s].len() != before {
                    visited[s] = true;
                    work.push_back(s);
                }
            }
        }
        // Second pass: report uses of possibly-moved locals.
        for bi in 0..block_count {
            if visited[bi] {
                let entry = moved_in[bi].clone();
                self.flow_block(bi as u32, entry, true);
            }
        }
    }

    /// Simulate one block. Returns (moved-out set, successors). When `report`, emit MIR-0007
    /// on any read of a possibly-moved local.
    fn flow_block(
        &mut self,
        bi: u32,
        mut moved: BTreeSet<u32>,
        report: bool,
    ) -> (BTreeSet<u32>, Vec<BlockId>) {
        // Collect the reads/writes/moves per statement in order.
        let block = self.body.blocks[bi as usize].clone();
        for (stmt, _) in &block.statements {
            if let Statement::Assign(place, rvalue) = stmt {
                self.flow_rvalue(rvalue, &mut moved, bi, report);
                if place.projection.is_empty() {
                    moved.remove(&place.local.0); // whole-local reinitialization
                } else if report && moved.contains(&place.local.0) {
                    self.err(
                        "MIR-0007",
                        bi,
                        format!("write through possibly-moved local _{}", place.local.0),
                    );
                }
            }
        }
        let mut successors = Vec::new();
        match &block.terminator.0 {
            Terminator::Goto { target } => successors.push(*target),
            Terminator::SwitchInt {
                scrut,
                arms,
                otherwise,
            } => {
                self.flow_operand(scrut, &mut moved, bi, report);
                successors.extend(arms.iter().map(|(_, b)| *b));
                successors.push(*otherwise);
            }
            Terminator::Call {
                callee,
                args,
                dest,
                target,
            } => {
                if let Callee::FnValue(op) = callee {
                    self.flow_operand(op, &mut moved, bi, report);
                }
                for arg in args {
                    self.flow_operand(arg, &mut moved, bi, report);
                }
                if dest.projection.is_empty() {
                    moved.remove(&dest.local.0);
                }
                successors.push(*target);
            }
            Terminator::Drop { place, target } => {
                if report && moved.contains(&place.local.0) {
                    self.err(
                        "MIR-0007",
                        bi,
                        format!("Drop of possibly-moved local _{}", place.local.0),
                    );
                }
                moved.insert(place.local.0);
                successors.push(*target);
            }
            Terminator::Checked {
                args, dest, target, ..
            } => {
                for arg in args {
                    self.flow_operand(arg, &mut moved, bi, report);
                }
                moved.remove(&dest.0);
                successors.push(*target);
            }
            Terminator::Trap { .. } | Terminator::Return | Terminator::Unreachable => {}
        }
        (moved, successors)
    }

    fn flow_rvalue(&mut self, rvalue: &Rvalue, moved: &mut BTreeSet<u32>, bi: u32, report: bool) {
        match rvalue {
            Rvalue::Use(op) => self.flow_operand(op, moved, bi, report),
            Rvalue::UnOp(_, op) => self.flow_operand(op, moved, bi, report),
            Rvalue::BinOp(_, a, b) => {
                self.flow_operand(a, moved, bi, report);
                self.flow_operand(b, moved, bi, report);
            }
            Rvalue::Aggregate(_, ops) => {
                for op in ops {
                    self.flow_operand(op, moved, bi, report);
                }
            }
            Rvalue::Discriminant(place) | Rvalue::RefOf { place, .. } => {
                if report && moved.contains(&place.local.0) {
                    self.err(
                        "MIR-0007",
                        bi,
                        format!("read of possibly-moved local _{}", place.local.0),
                    );
                }
            }
        }
    }

    fn flow_operand(&mut self, op: &Operand, moved: &mut BTreeSet<u32>, bi: u32, report: bool) {
        match op {
            Operand::Copy(place) => {
                if report && moved.contains(&place.local.0) {
                    self.err(
                        "MIR-0007",
                        bi,
                        format!("copy from possibly-moved local _{}", place.local.0),
                    );
                }
            }
            Operand::Move(place) => {
                if report && moved.contains(&place.local.0) {
                    self.err(
                        "MIR-0007",
                        bi,
                        format!("move from possibly-moved local _{}", place.local.0),
                    );
                }
                // Whole-local granularity: a projected move conservatively moves the local.
                moved.insert(place.local.0);
            }
            Operand::Const(_) => {}
        }
    }

    // ---- V-IDX-2: proof locals appear only as CheckIndex dests / Index projections ----

    /// V-IDX-1 same-base rule (CE3-revised design): every `Index(proof)` projection's place
    /// prefix must equal the base place its `CheckIndex` bound. Dominance of an ordinary local
    /// is insufficient; the binding is to base identity.
    fn verify_index_bindings(&mut self) {
        use std::collections::HashMap;
        // proof local -> bound base place (from its CheckIndex definition).
        let mut bound: HashMap<u32, Place> = HashMap::new();
        for block in &self.body.blocks {
            if let Terminator::Checked {
                op: CheckedOp::CheckIndex,
                args,
                dest,
                ..
            } = &block.terminator.0
            {
                if let Some(Operand::Copy(base)) = args.first() {
                    bound.insert(dest.0, base.clone());
                }
            }
        }
        // Every place used anywhere: check each Index projection against the binding.
        let mut places: Vec<Place> = Vec::new();
        for block in &self.body.blocks {
            for (stmt, _) in &block.statements {
                if let Statement::Assign(place, rvalue) = stmt {
                    places.push(place.clone());
                    collect_rvalue_places(rvalue, &mut places);
                }
            }
            match &block.terminator.0 {
                Terminator::SwitchInt { scrut, .. } => collect_operand_place(scrut, &mut places),
                Terminator::Call {
                    callee, args, dest, ..
                } => {
                    if let Callee::FnValue(op) = callee {
                        collect_operand_place(op, &mut places);
                    }
                    for arg in args {
                        collect_operand_place(arg, &mut places);
                    }
                    places.push(dest.clone());
                }
                Terminator::Drop { place, .. } => places.push(place.clone()),
                Terminator::Checked { args, .. } => {
                    for arg in args {
                        collect_operand_place(arg, &mut places);
                    }
                }
                _ => {}
            }
        }
        let mut violations: Vec<String> = Vec::new();
        for place in &places {
            for (k, projection) in place.projection.iter().enumerate() {
                if let Projection::Index(proof) = projection {
                    let prefix = Place {
                        local: place.local,
                        projection: place.projection[..k].to_vec(),
                    };
                    match bound.get(&proof.0) {
                        Some(base) if *base == prefix => {}
                        Some(base) => violations.push(format!(
                            "Index(proof _{}) used on {:?} but the proof binds base {:?}",
                            proof.0, prefix, base
                        )),
                        None => violations.push(format!(
                            "Index(proof _{}) has no defining CheckIndex in this body",
                            proof.0
                        )),
                    }
                }
            }
        }
        for message in violations {
            self.err("MIR-0010", 0, message);
        }
    }

    fn verify_index_proofs(&mut self) {
        for (bi, block) in self.body.blocks.clone().iter().enumerate() {
            for (stmt, _) in &block.statements {
                if let Statement::Assign(_, rvalue) = stmt {
                    self.scan_rvalue_for_proof_misuse(rvalue, bi as u32);
                }
            }
            match &block.terminator.0 {
                Terminator::SwitchInt { scrut, .. } => {
                    self.scan_operand_for_proof_misuse(scrut, bi as u32)
                }
                Terminator::Call { callee, args, .. } => {
                    if let Callee::FnValue(op) = callee {
                        self.scan_operand_for_proof_misuse(op, bi as u32);
                    }
                    for arg in args {
                        self.scan_operand_for_proof_misuse(arg, bi as u32);
                    }
                }
                Terminator::Checked { op, args, .. } => {
                    if !matches!(op, CheckedOp::CheckIndex) {
                        for arg in args {
                            self.scan_operand_for_proof_misuse(arg, bi as u32);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn scan_rvalue_for_proof_misuse(&mut self, rvalue: &Rvalue, bi: u32) {
        let ops: Vec<&Operand> = match rvalue {
            Rvalue::Use(op) | Rvalue::UnOp(_, op) => vec![op],
            Rvalue::BinOp(_, a, b) => vec![a, b],
            Rvalue::Aggregate(_, ops) => ops.iter().collect(),
            Rvalue::Discriminant(_) | Rvalue::RefOf { .. } => Vec::new(),
        };
        for op in ops {
            self.scan_operand_for_proof_misuse(op, bi);
        }
    }

    fn scan_operand_for_proof_misuse(&mut self, op: &Operand, bi: u32) {
        if let Operand::Copy(place) | Operand::Move(place) = op {
            if let Some(decl) = self.body.locals.get(place.local.0 as usize) {
                if matches!(decl.kind, LocalKind::IndexProof) {
                    self.err(
                        "MIR-0010",
                        bi,
                        format!(
                            "IndexProof local _{} used as an ordinary operand",
                            place.local.0
                        ),
                    );
                }
            }
        }
    }
}

// ---- helpers ----

fn bare_unsized(ty: &MirTy) -> bool {
    matches!(ty, MirTy::Str | MirTy::Slice(_))
}

fn is_integer(ty: &MirTy) -> bool {
    matches!(
        ty,
        MirTy::Int8
            | MirTy::Int16
            | MirTy::Int32
            | MirTy::Int64
            | MirTy::UInt8
            | MirTy::UInt16
            | MirTy::UInt32
            | MirTy::UInt64
    )
}

fn is_numeric(ty: &MirTy) -> bool {
    is_integer(ty) || matches!(ty, MirTy::Float32 | MirTy::Float64)
}

/// Can a value of this type ever require dropping? (Conservative: user nominals yes — the
/// scalar core never emits Drop, and C4.5's elaboration will refine this with real Drop-impl
/// knowledge; primitives/fn-values/refs never need dropping.)
fn may_need_drop(ty: &MirTy) -> bool {
    match ty {
        MirTy::Struct(..) | MirTy::Enum(..) | MirTy::String | MirTy::Core(..) => true,
        MirTy::Tuple(elems) => elems.iter().any(may_need_drop),
        MirTy::Array(elem, _) => may_need_drop(elem),
        _ => false,
    }
}

fn variant_payload(
    program: &MirProgram,
    enum_ref: &EnumRef,
    args: &[MirTy],
    variant: u32,
) -> Option<Vec<MirTy>> {
    match enum_ref {
        EnumRef::CoreOption => match variant {
            0 => Some(Vec::new()),
            1 => Some(vec![args.first()?.clone()]),
            _ => None,
        },
        EnumRef::CoreResult => match variant {
            0 => Some(vec![args.first()?.clone()]),
            1 => Some(vec![args.get(1)?.clone()]),
            _ => None,
        },
        EnumRef::User(item) => program
            .types
            .enum_variants
            .get(&item.0)
            .and_then(|variants| variants.get(variant as usize).cloned()),
    }
}

fn runtime_sig(rt: RuntimeFn) -> (Vec<MirTy>, MirTy) {
    match rt {
        RuntimeFn::PrintlnInt64 | RuntimeFn::PrintInt64 => (vec![MirTy::Int64], MirTy::Unit),
        RuntimeFn::PrintlnUInt64 | RuntimeFn::PrintUInt64 => (vec![MirTy::UInt64], MirTy::Unit),
        RuntimeFn::PrintlnBool | RuntimeFn::PrintBool => (vec![MirTy::Bool], MirTy::Unit),
        RuntimeFn::PrintlnFloat64 | RuntimeFn::PrintFloat64 => (vec![MirTy::Float64], MirTy::Unit),
    }
}

fn collect_operand_place(op: &Operand, out: &mut Vec<Place>) {
    if let Operand::Copy(place) | Operand::Move(place) = op {
        out.push(place.clone());
    }
}

fn collect_rvalue_places(rvalue: &Rvalue, out: &mut Vec<Place>) {
    match rvalue {
        Rvalue::Use(op) | Rvalue::UnOp(_, op) => collect_operand_place(op, out),
        Rvalue::BinOp(_, a, b) => {
            collect_operand_place(a, out);
            collect_operand_place(b, out);
        }
        Rvalue::Aggregate(_, ops) => {
            for op in ops {
                collect_operand_place(op, out);
            }
        }
        Rvalue::Discriminant(place) | Rvalue::RefOf { place, .. } => out.push(place.clone()),
    }
}
