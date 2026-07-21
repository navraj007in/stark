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
//! V-REF-1    MIR-0014 (write crossing a Deref of a shared reference — C4.5e-0)
//! V-STR-1/2  MIR-0015 (invalid Str constant / String|str in a structural op / bad Trap msg — A1)
//! V-COPY-1   MIR-0016 (Copy-only runtime op on a non-Copy element type — A1; Vec ops land e-2)
//! V-SURFACE-1 MIR-0017 (unsupported mir_version/runtime_surface — A1, program-level gate)
//! MIR-0003   projection type mismatch (V-CFG-2's "projections type-correct step by step")
//! ```
//!
//! Scope notes (v0.1, honest limitations — refinements, not silent gaps):
//! - V-MOVE-1 (field-precise under C4.5d; VariantField-precise under DEV-079; ConstIndex-precise
//!   under A5/DEV-086): entries are (local, path) where a path component is a TYPED
//!   `MovePathStep` — struct/tuple field, variant-then-field, or a constant array index — so
//!   distinct projection kinds cannot compare equal. Places with `Deref` or the proof-backed
//!   `Index` are their whole local, because neither names a statically-known sub-place. Any-path
//!   (union) joins to a fixpoint. A read conflicts with a moved entry when the paths are
//!   prefix-related either way; assignment reinitializes the subtree it covers. `Drop` of a
//!   possibly-moved place is NOT an error: flag-guarded conditional drops (contract §8) are
//!   exactly that state by design, and the dataflow is flag-blind — the runtime flag guard
//!   plus the differential corpus enforce exactly-once there.
//! - Drop-flag "read only by SwitchInt" (V-DROP-2 read half) is enforced as: DropFlag locals
//!   never appear in any statement rvalue other than their own `Const(true/false)`
//!   initialization, and never as call/checked/drop operands; `SwitchInt` reads are
//!   permitted.

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
    // A1 (CD-031) surface-version gate: reject an unsupported MIR version or runtime surface
    // BEFORE consuming any body. The per-program fields are authoritative (not the global
    // constants, which are only what a same-build producer stamped).
    if program.mir_version != MIR_VERSION || program.runtime_surface != MIR_RUNTIME_SURFACE {
        return Err(vec![MirError {
            code: "MIR-0017",
            message: format!(
                "unsupported MIR surface: program is v{}/{}, this build supports v{}/{}",
                program.mir_version, program.runtime_surface, MIR_VERSION, MIR_RUNTIME_SURFACE
            ),
            symbol: String::new(),
            block: 0,
        }]);
    }
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
        self.verify_drop_flags();
        self.verify_index_proofs();
        self.verify_index_bindings();
        self.verify_proof_flow();
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
        self.place_ty_impl(place, bi, false)
    }

    /// Type a place used as a *write destination*. V-REF-1 (C4.5e-0): a write crossing a
    /// `Deref` requires that dereferenced layer to be `&mut` — mutation through a shared
    /// reference is structurally invalid MIR regardless of what borrowck upstream promised.
    fn place_ty_for_write(&mut self, place: &Place, bi: u32) -> Option<MirTy> {
        self.place_ty_impl(place, bi, true)
    }

    fn place_ty_impl(&mut self, place: &Place, bi: u32, writing: bool) -> Option<MirTy> {
        let mut ty = self.local_ty(place.local, bi)?;
        for projection in &place.projection {
            if writing {
                if let (Projection::Deref, MirTy::Ref { mutable: false, .. }) = (projection, &ty) {
                    self.err(
                        "MIR-0014",
                        bi,
                        format!(
                            "write through a shared reference (place rooted at _{})",
                            place.local.0
                        ),
                    );
                    return None;
                }
            }
            ty = match (projection, ty) {
                (Projection::Field(i), MirTy::Struct(item, args)) => {
                    match self
                        .program
                        .types
                        .struct_fields
                        .get(&(item.0, args))
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
                // A5 (CD-038): a statically known array element. The verifier checks the bound
                // itself — that is the entire justification for the form carrying no proof.
                // Deliberately NOT accepted on `Slice` or `Vec`: their lengths are not statically
                // known, so nothing here could check the index.
                (Projection::ConstIndex(i), MirTy::Array(elem, len)) => {
                    if *i >= len {
                        self.err(
                            "MIR-0010",
                            bi,
                            format!("ConstIndex {i} out of bounds for Array<_, {len}>"),
                        );
                        return None;
                    }
                    *elem
                }
                (Projection::ConstIndex(_), other) => {
                    self.err(
                        "MIR-0010",
                        bi,
                        format!(
                            "ConstIndex projection on {other:?}; valid only on a fixed-length Array"
                        ),
                    );
                    return None;
                }
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
                // A1: a string literal is a `&str`.
                Constant::Str(_) => MirTy::Ref {
                    mutable: false,
                    inner: Box::new(MirTy::Str),
                },
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
        let Some(lhs_ty) = self.place_ty_for_write(place, bi) else {
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
                if is_stringish(&ty) {
                    self.err(
                        "MIR-0015",
                        bi,
                        "unary operation on a String/str operand (V-STR-2)",
                    );
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
                // A1 V-STR-2: String/str equality/ordering routes through StrEq/StrCmp, never
                // a structural BinOp.
                if is_stringish(&lt) || is_stringish(&rt) {
                    self.err(
                        "MIR-0015",
                        bi,
                        "binary operation on a String/str operand (V-STR-2)",
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
                    // A5: bitwise ops are integer-only and result-typed as the operands.
                    MirBinOp::BitAnd | MirBinOp::BitOr | MirBinOp::BitXor => {
                        if !is_integer(&lt) {
                            self.err("MIR-0004", bi, format!("bitwise BinOp on {lt:?}"));
                        }
                        self.expect_ty(expected, &lt, "bitwise op", bi);
                    }
                }
            }
            // A4 (CD-036): a layout query answers in `UInt64` (06: `fn size_of<T>() -> UInt64`).
            // The QUERIED type is deliberately unconstrained — every `MirTy` is a legal question,
            // and `Sized`-ness is the checked front end's property, not a verifier rule
            // (amendment §5).
            Rvalue::LayoutQuery { .. } => {
                self.expect_ty(expected, &MirTy::UInt64, "layout query", bi);
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
            (AggKind::Struct(item), MirTy::Struct(expected_item, args)) => {
                if item != expected_item {
                    self.err("MIR-0004", bi, "struct aggregate item mismatch");
                    return;
                }
                match self
                    .program
                    .types
                    .struct_fields
                    .get(&(item.0, args.clone()))
                {
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
                    | MirTy::UInt64
                    // A2: Char literal patterns switch on the Unicode scalar codepoint.
                    | MirTy::Char => {}
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
                let dest_ty = self.place_ty_for_write(dest, bi);
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
                    // A1: Vec ops have schematic-in-T signatures and V-COPY-1 constraints;
                    // string/print ops use the fixed table.
                    Callee::Runtime(rt) if is_vec_runtime_fn(*rt) => {
                        self.vec_runtime_sig(*rt, &arg_tys, dest_ty.as_ref(), bi)
                    }
                    // 0.1-A3: HashMap ops are schematic in (K, V).
                    Callee::Runtime(rt) if is_map_runtime_fn(*rt) => {
                        self.map_runtime_sig(*rt, &arg_tys, dest_ty.as_ref(), bi)
                    }
                    // 0.1-A6: slice ops are schematic in T (from the reference receiver).
                    Callee::Runtime(rt) if is_slice_runtime_fn(*rt) => {
                        self.slice_runtime_sig(*rt, &arg_tys, bi)
                    }
                    // 0.1-A7: Box ops are schematic in T.
                    Callee::Runtime(rt) if is_box_runtime_fn(*rt) => {
                        self.box_runtime_sig(*rt, &arg_tys, bi)
                    }
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
                if let Some(ty) = self.place_ty_for_write(place, bi) {
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
            // A1: a `Trap` message, when present, must type as `&str` (MIR-0015).
            Terminator::Trap {
                message: Some(op), ..
            } => {
                if let Some(ty) = self.operand_ty(op, bi) {
                    let expected = MirTy::Ref {
                        mutable: false,
                        inner: Box::new(MirTy::Str),
                    };
                    if ty != expected {
                        self.err(
                            "MIR-0015",
                            bi,
                            format!("Trap message must be &str, found {ty:?}"),
                        );
                    }
                }
            }
            Terminator::Trap { message: None, .. }
            | Terminator::Return
            | Terminator::Unreachable => {}
        }
    }

    // ---- V-MOVE-1: any-path moved-place dataflow to a fixpoint (field-precise, C4.5d) ----
    //
    // Entries are (local, pure-Field path); a place with any non-Field projection is
    // conservatively its whole local (path []). A read of place P conflicts with a moved
    // entry M when the paths are prefix-related in either direction; assignment to D
    // reinitializes every entry D covers. `Drop` consumes its place WITHOUT a
    // possibly-moved error: flag-guarded conditional drops (contract §8) are exactly the
    // designed possibly-moved-at-Drop state, invisible to a flag-blind dataflow — the
    // runtime guard plus the differential corpus carry that obligation instead.

    fn verify_moves(&mut self) {
        let block_count = self.body.blocks.len();
        // moved_in[b]: set of places possibly moved on entry to b.
        let mut moved_in: Vec<BTreeSet<(u32, Vec<MovePathStep>)>> =
            vec![BTreeSet::new(); block_count];
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
                moved_in[s].extend(moved_out.iter().cloned());
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
    /// on any read of a possibly-moved place.
    fn flow_block(
        &mut self,
        bi: u32,
        mut moved: BTreeSet<(u32, Vec<MovePathStep>)>,
        report: bool,
    ) -> (BTreeSet<(u32, Vec<MovePathStep>)>, Vec<BlockId>) {
        // Collect the reads/writes/moves per statement in order.
        let block = self.body.blocks[bi as usize].clone();
        for (stmt, _) in &block.statements {
            if let Statement::Assign(place, rvalue) = stmt {
                self.flow_rvalue(rvalue, &mut moved, bi, report);
                self.flow_reinit(place, &mut moved, bi, report);
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
                self.flow_reinit(dest, &mut moved, bi, report);
                successors.push(*target);
            }
            Terminator::Drop { place, target } => {
                // No possibly-moved error (see the pass header note); the drop consumes.
                moved.insert(moved_key(place));
                successors.push(*target);
            }
            Terminator::Checked {
                args, dest, target, ..
            } => {
                for arg in args {
                    self.flow_operand(arg, &mut moved, bi, report);
                }
                let dest_local = dest.0;
                moved.retain(|(l, _)| *l != dest_local);
                successors.push(*target);
            }
            // A1: the Trap message participates in move dataflow (V-MOVE-1).
            Terminator::Trap {
                message: Some(op), ..
            } => self.flow_operand(op, &mut moved, bi, report),
            Terminator::Trap { message: None, .. }
            | Terminator::Return
            | Terminator::Unreachable => {}
        }
        (moved, successors)
    }

    /// A read of `place` conflicts with a moved entry when the field paths are
    /// prefix-related in either direction (reading a moved value, a field of a moved value,
    /// or a container with a moved field).
    fn flow_read(
        &mut self,
        place: &Place,
        moved: &BTreeSet<(u32, Vec<MovePathStep>)>,
        bi: u32,
        report: bool,
        what: &str,
    ) {
        if !report {
            return;
        }
        let (local, path) = moved_key(place);
        let conflict = moved
            .iter()
            .any(|(l, m)| *l == local && paths_prefix_related(m, &path));
        if conflict {
            self.err(
                "MIR-0007",
                bi,
                format!("{what} possibly-moved place _{local}{path:?}"),
            );
        }
    }

    /// Assignment to `place` reinitializes every moved entry it covers; writing a field
    /// inside a still-moved container is an error.
    fn flow_reinit(
        &mut self,
        place: &Place,
        moved: &mut BTreeSet<(u32, Vec<MovePathStep>)>,
        bi: u32,
        report: bool,
    ) {
        let (local, path) = moved_key(place);
        if report {
            let inside_moved = moved
                .iter()
                .any(|(l, m)| *l == local && m.len() < path.len() && path[..m.len()] == m[..]);
            if inside_moved {
                self.err(
                    "MIR-0007",
                    bi,
                    format!("write through possibly-moved place _{local}"),
                );
            }
        }
        moved.retain(|(l, m)| {
            *l != local || !(m.len() >= path.len() && m[..path.len()] == path[..])
        });
    }

    fn flow_rvalue(
        &mut self,
        rvalue: &Rvalue,
        moved: &mut BTreeSet<(u32, Vec<MovePathStep>)>,
        bi: u32,
        report: bool,
    ) {
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
                self.flow_read(place, moved, bi, report, "read of");
            }
            // A4: a layout query reads no place, so it neither moves nor requires initialization.
            Rvalue::LayoutQuery { .. } => {}
        }
    }

    fn flow_operand(
        &mut self,
        op: &Operand,
        moved: &mut BTreeSet<(u32, Vec<MovePathStep>)>,
        bi: u32,
        report: bool,
    ) {
        match op {
            Operand::Copy(place) => {
                self.flow_read(place, moved, bi, report, "copy from");
            }
            Operand::Move(place) => {
                self.flow_read(place, moved, bi, report, "move from");
                moved.insert(moved_key(place));
            }
            Operand::Const(_) => {}
        }
    }

    // ---- A1 (CD-031) Vec runtime ops: schematic T + V-COPY-1 ----

    /// Resolve a Vec op's schematic signature (A1 §6: constructors take `T` from the
    /// destination element, methods from the first `Vec` operand) and enforce V-COPY-1
    /// (MIR-0016): `VecIndexGet` requires `T: Copy`; `VecClear` requires non-droppable `T`.
    fn vec_runtime_sig(
        &mut self,
        rt: RuntimeFn,
        arg_tys: &[Option<MirTy>],
        dest_ty: Option<&MirTy>,
        bi: u32,
    ) -> Option<(Vec<MirTy>, MirTy)> {
        use RuntimeFn::*;
        let vec = |t: &MirTy| MirTy::Core(crate::hir::CoreType::Vec, vec![t.clone()]);
        let vec_ref = |t: &MirTy, mutable| MirTy::Ref {
            mutable,
            inner: Box::new(vec(t)),
        };
        let opt = |t: &MirTy| MirTy::Enum(EnumRef::CoreOption, vec![t.clone()]);

        // Resolve T.
        let t = match rt {
            VecNew | VecWithCapacity => match dest_ty {
                Some(MirTy::Core(crate::hir::CoreType::Vec, args)) => args.first().cloned(),
                _ => {
                    self.err("MIR-0012", bi, "Vec constructor dest is not a Vec type");
                    None
                }
            },
            // 0.1-A2: iterator ops resolve T from the Core(VecIter,[T]) operand.
            VecIterNext => match arg_tys.first().and_then(|o| o.as_ref()) {
                Some(MirTy::Ref { inner, .. }) => match inner.as_ref() {
                    MirTy::Core(crate::hir::CoreType::VecIter, args) => args.first().cloned(),
                    other => {
                        self.err(
                            "MIR-0012",
                            bi,
                            format!("VecIterNext operand is &{other:?}, not &mut VecIter"),
                        );
                        None
                    }
                },
                _ => {
                    self.err("MIR-0012", bi, "VecIterNext operand is not a &mut VecIter");
                    None
                }
            },
            _ => match arg_tys.first().and_then(|o| o.as_ref()) {
                Some(MirTy::Ref { inner, .. }) => match inner.as_ref() {
                    MirTy::Core(crate::hir::CoreType::Vec, args) => args.first().cloned(),
                    other => {
                        self.err(
                            "MIR-0012",
                            bi,
                            format!("Vec op first operand is &{other:?}, not &Vec"),
                        );
                        None
                    }
                },
                _ => {
                    self.err("MIR-0012", bi, "Vec op first operand is not a &Vec");
                    None
                }
            },
        }?;

        // V-COPY-1.
        match rt {
            VecIndexGet if !self.mir_is_copy(&t) => {
                self.err(
                    "MIR-0016",
                    bi,
                    format!("VecIndexGet requires a Copy element type, found {t:?}"),
                );
            }
            // A6: Vec iteration is a borrowed cursor (no snapshot), so `T` need NOT be Copy.
            VecClear if self.mir_needs_drop(&t) => {
                self.err(
                    "MIR-0016",
                    bi,
                    format!("VecClear requires a non-droppable element type, found {t:?}"),
                );
            }
            _ => {}
        }

        let uint64 = MirTy::UInt64;
        Some(match rt {
            VecNew => (vec![], vec(&t)),
            VecWithCapacity => (vec![uint64], vec(&t)),
            VecPush => (vec![vec_ref(&t, true), t.clone()], MirTy::Unit),
            VecPop => (vec![vec_ref(&t, true)], opt(&t)),
            VecLen => (vec![vec_ref(&t, false)], uint64),
            VecIsEmpty => (vec![vec_ref(&t, false)], MirTy::Bool),
            VecIndexGet => (vec![vec_ref(&t, false), uint64], t.clone()),
            // 0.1-A4: checked interior access → Option<&T>/Option<&mut T> (no Copy requirement).
            VecGetRef => (
                vec![vec_ref(&t, false), uint64],
                opt(&MirTy::Ref {
                    mutable: false,
                    inner: Box::new(t.clone()),
                }),
            ),
            VecGetMutRef => (
                vec![vec_ref(&t, true), uint64],
                opt(&MirTy::Ref {
                    mutable: true,
                    inner: Box::new(t.clone()),
                }),
            ),
            VecReplace => (vec![vec_ref(&t, true), uint64, t.clone()], t.clone()),
            VecRemove => (vec![vec_ref(&t, true), uint64], t.clone()),
            VecClear => (vec![vec_ref(&t, true)], MirTy::Unit),
            // 0.1-A2 iteration: New borrows the Vec; Next yields Option<&T>.
            VecIterNew => (
                vec![vec_ref(&t, false)],
                MirTy::Core(crate::hir::CoreType::VecIter, vec![t.clone()]),
            ),
            VecIterNext => {
                let iter_ref = MirTy::Ref {
                    mutable: true,
                    inner: Box::new(MirTy::Core(crate::hir::CoreType::VecIter, vec![t.clone()])),
                };
                let elem_ref = MirTy::Ref {
                    mutable: false,
                    inner: Box::new(t.clone()),
                };
                (vec![iter_ref], opt(&elem_ref))
            }
            _ => unreachable!("non-Vec op in vec_runtime_sig"),
        })
    }

    /// 0.1-A3: HashMap ops, schematic in (K, V). Constructors resolve from the destination;
    /// methods from the first `&HashMap` operand; the keys iterator from `Core(KeysIter,[K])`.
    /// User-`Drop` K/V is a lowering-side exclusion; the verifier checks shapes only.
    fn map_runtime_sig(
        &mut self,
        rt: RuntimeFn,
        arg_tys: &[Option<MirTy>],
        dest_ty: Option<&MirTy>,
        bi: u32,
    ) -> Option<(Vec<MirTy>, MirTy)> {
        use RuntimeFn::*;
        let map = |k: &MirTy, v: &MirTy| {
            MirTy::Core(crate::hir::CoreType::HashMap, vec![k.clone(), v.clone()])
        };
        let map_ref = |k: &MirTy, v: &MirTy, mutable| MirTy::Ref {
            mutable,
            inner: Box::new(map(k, v)),
        };
        let sref = |t: &MirTy| MirTy::Ref {
            mutable: false,
            inner: Box::new(t.clone()),
        };
        let opt = |t: MirTy| MirTy::Enum(EnumRef::CoreOption, vec![t]);

        // Resolve (K, V).
        let kv = match rt {
            HashMapNew => match dest_ty {
                Some(MirTy::Core(crate::hir::CoreType::HashMap, args)) if args.len() == 2 => {
                    Some((args[0].clone(), args[1].clone()))
                }
                _ => {
                    self.err("MIR-0012", bi, "HashMap constructor dest is not a HashMap");
                    None
                }
            },
            HashMapKeysIterNext => match arg_tys.first().and_then(|o| o.as_ref()) {
                Some(MirTy::Ref { inner, .. }) => match inner.as_ref() {
                    MirTy::Core(crate::hir::CoreType::KeysIter, args) => args
                        .first()
                        .map(|k| (k.clone(), MirTy::Unit /* V unused */)),
                    other => {
                        self.err(
                            "MIR-0012",
                            bi,
                            format!("KeysIterNext operand is &{other:?}, not &mut KeysIter"),
                        );
                        None
                    }
                },
                _ => {
                    self.err(
                        "MIR-0012",
                        bi,
                        "KeysIterNext operand is not a &mut KeysIter",
                    );
                    None
                }
            },
            _ => match arg_tys.first().and_then(|o| o.as_ref()) {
                Some(MirTy::Ref { inner, .. }) => match inner.as_ref() {
                    MirTy::Core(crate::hir::CoreType::HashMap, args) if args.len() == 2 => {
                        Some((args[0].clone(), args[1].clone()))
                    }
                    other => {
                        self.err(
                            "MIR-0012",
                            bi,
                            format!("HashMap op first operand is &{other:?}, not &HashMap"),
                        );
                        None
                    }
                },
                _ => {
                    self.err("MIR-0012", bi, "HashMap op first operand is not a &HashMap");
                    None
                }
            },
        };
        let (k, v) = kv?;

        Some(match rt {
            HashMapNew => (vec![], map(&k, &v)),
            HashMapInsert => (
                vec![map_ref(&k, &v, true), k.clone(), v.clone()],
                opt(v.clone()),
            ),
            HashMapGet => (vec![map_ref(&k, &v, false), sref(&k)], opt(sref(&v))),
            HashMapLen => (vec![map_ref(&k, &v, false)], MirTy::UInt64),
            HashMapIsEmpty => (vec![map_ref(&k, &v, false)], MirTy::Bool),
            HashMapContainsKey => (vec![map_ref(&k, &v, false), sref(&k)], MirTy::Bool),
            HashMapKeysIterNew => (
                vec![map_ref(&k, &v, false)],
                MirTy::Core(crate::hir::CoreType::KeysIter, vec![k.clone()]),
            ),
            HashMapKeysIterNext => {
                let iter_ref = MirTy::Ref {
                    mutable: true,
                    inner: Box::new(MirTy::Core(crate::hir::CoreType::KeysIter, vec![k.clone()])),
                };
                (vec![iter_ref], opt(sref(&k)))
            }
            _ => unreachable!("non-HashMap op in map_runtime_sig"),
        })
    }

    /// 0.1-A6 (A4 slicing): slice ops, schematic in T. `SliceNew`'s receiver is a shared
    /// reference to an Array/Vec/slice with element T; the bounds are a matching integer pair
    /// plus the inclusive Bool; the result is `&[T]`. `SliceLen`/`SliceIsEmpty` read a `&[T]`.
    /// 0.1-A7 (WP-C4.7-6.1): `BoxNew(T) -> Box<T>` and `BoxIntoInner(Box<T>) -> T`, schematic
    /// in `T`. Both take their argument BY VALUE — `BoxNew` consumes the value it stores and
    /// `BoxIntoInner` consumes the box, transferring the contained value out without dropping
    /// it. `T` is read from the argument, so a mismatched destination is caught by the ordinary
    /// call-dest check (MIR-0005) in the caller.
    fn box_runtime_sig(
        &mut self,
        rt: RuntimeFn,
        arg_tys: &[Option<MirTy>],
        bi: u32,
    ) -> Option<(Vec<MirTy>, MirTy)> {
        use RuntimeFn::*;
        let arg = arg_tys.first().and_then(|o| o.as_ref())?;
        let boxed = |t: &MirTy| MirTy::Core(crate::hir::CoreType::Box, vec![t.clone()]);
        match rt {
            BoxNew => Some((vec![arg.clone()], boxed(arg))),
            BoxIntoInner => match arg {
                MirTy::Core(crate::hir::CoreType::Box, args) => {
                    let inner = args.first().cloned().unwrap_or(MirTy::Unit);
                    Some((vec![arg.clone()], inner))
                }
                other => {
                    self.err(
                        "MIR-0005",
                        bi,
                        format!("BoxIntoInner expects a Box receiver, found {other:?}"),
                    );
                    None
                }
            },
            _ => unreachable!("non-Box op in box_runtime_sig"),
        }
    }

    fn slice_runtime_sig(
        &mut self,
        rt: RuntimeFn,
        arg_tys: &[Option<MirTy>],
        bi: u32,
    ) -> Option<(Vec<MirTy>, MirTy)> {
        use RuntimeFn::*;
        let recv_ty = arg_tys.first().and_then(|o| o.as_ref());
        match rt {
            SliceNew | SliceNewMut => {
                let (recv, t) = match recv_ty {
                    Some(r @ MirTy::Ref { inner, .. }) => match inner.as_ref() {
                        MirTy::Array(elem, _) | MirTy::Slice(elem) => (r.clone(), (**elem).clone()),
                        MirTy::Core(crate::hir::CoreType::Vec, args) => {
                            (r.clone(), args.first().cloned().unwrap_or(MirTy::Unit))
                        }
                        other => {
                            self.err(
                                "MIR-0012",
                                bi,
                                format!("SliceNew receiver is &{other:?}, not a sliceable"),
                            );
                            return None;
                        }
                    },
                    _ => {
                        self.err("MIR-0012", bi, "SliceNew receiver is not a reference");
                        return None;
                    }
                };
                // Bounds: a matching integer pair (the range's element type).
                let bound = match arg_tys.get(1).and_then(|o| o.as_ref()) {
                    Some(b) if is_integer(b) => b.clone(),
                    Some(other) => {
                        self.err(
                            "MIR-0012",
                            bi,
                            format!("SliceNew bound is {other:?}, not an integer"),
                        );
                        return None;
                    }
                    None => return None,
                };
                // 0.1-A8: the EXCLUSIVE form yields `&mut [T]` and requires an exclusive
                // receiver borrow — a shared base cannot produce a writable view.
                let exclusive = matches!(rt, SliceNewMut);
                if exclusive && !matches!(recv, MirTy::Ref { mutable: true, .. }) {
                    self.err(
                        "MIR-0012",
                        bi,
                        "SliceNewMut receiver must be an exclusive reference",
                    );
                    return None;
                }
                let result = MirTy::Ref {
                    mutable: exclusive,
                    inner: Box::new(MirTy::Slice(Box::new(t))),
                };
                Some((vec![recv, bound.clone(), bound, MirTy::Bool], result))
            }
            // `len`/`is_empty` only read, so either a shared or an exclusive view is a valid
            // receiver; the signature is stated in terms of the receiver's own mutability.
            SliceLen | SliceIsEmpty => {
                let recv_mutable = matches!(recv_ty, Some(MirTy::Ref { mutable: true, .. }));
                let t = match recv_ty {
                    Some(MirTy::Ref { inner, .. }) => match inner.as_ref() {
                        MirTy::Slice(elem) => (**elem).clone(),
                        other => {
                            self.err(
                                "MIR-0012",
                                bi,
                                format!("slice op receiver is &{other:?}, not &[T]"),
                            );
                            return None;
                        }
                    },
                    _ => {
                        self.err("MIR-0012", bi, "slice op receiver is not a reference");
                        return None;
                    }
                };
                let ret = if matches!(rt, SliceLen) {
                    MirTy::UInt64
                } else {
                    MirTy::Bool
                };
                let recv = MirTy::Ref {
                    mutable: recv_mutable,
                    inner: Box::new(MirTy::Slice(Box::new(t))),
                };
                Some((vec![recv], ret))
            }
            _ => unreachable!("non-slice op in slice_runtime_sig"),
        }
    }

    /// A1: precise droppability, mirroring lowering's `ty_needs_drop` so the verifier never
    /// rejects a valid lowering (String/Vec always; a nominal with an own `Drop` impl or a
    /// droppable field; recursion through aggregates). Distinct from the conservative
    /// `may_need_drop` used by the Drop-terminator sanity check.
    fn mir_needs_drop(&self, ty: &MirTy) -> bool {
        match ty {
            MirTy::String | MirTy::Core(..) => true,
            MirTy::Struct(item, args) => {
                let key = (item.0, args.clone());
                self.program.types.drop_impls.contains_key(&key)
                    || self
                        .program
                        .types
                        .struct_fields
                        .get(&key)
                        .is_some_and(|fs| fs.iter().any(|f| self.mir_needs_drop(f)))
            }
            MirTy::Enum(EnumRef::User(item), args) => {
                let key = (item.0, args.clone());
                self.program.types.drop_impls.contains_key(&key)
                    || self
                        .program
                        .types
                        .enum_variants
                        .get(&key)
                        .is_some_and(|vs| {
                            vs.iter().any(|v| v.iter().any(|f| self.mir_needs_drop(f)))
                        })
            }
            MirTy::Enum(_, args) => args.iter().any(|a| self.mir_needs_drop(a)),
            MirTy::Tuple(elems) => elems.iter().any(|e| self.mir_needs_drop(e)),
            MirTy::Array(elem, _) => self.mir_needs_drop(elem),
            _ => false,
        }
    }

    /// A1: is `ty` `Copy` at the MIR level? Primitives/refs/fn-values/all-Copy aggregates are
    /// Copy; user nominals are Copy iff `TypeContext::copy_types` records an `impl Copy`;
    /// String/Vec and mutable refs are not.
    fn mir_is_copy(&self, ty: &MirTy) -> bool {
        match ty {
            MirTy::Struct(item, args) | MirTy::Enum(EnumRef::User(item), args) => self
                .program
                .types
                .copy_types
                .contains(&(item.0, args.clone())),
            MirTy::Enum(_, args) => args.iter().all(|a| self.mir_is_copy(a)),
            MirTy::Tuple(elems) => elems.iter().all(|e| self.mir_is_copy(e)),
            MirTy::Array(elem, _) => self.mir_is_copy(elem),
            MirTy::Ref { mutable, .. } => !*mutable,
            MirTy::Slice(_) | MirTy::Core(..) | MirTy::String => false,
            _ => true,
        }
    }

    // ---- V-DROP-2 read half: drop flags are read only by SwitchInt ----

    /// A `DropFlag` local may appear only as (a) the destination of its own
    /// `Const(true/false)` assignment (write half, enforced in `verify_assign`) and (b) a
    /// `SwitchInt` scrutinee `Copy`. Any other appearance — statement rvalue operand,
    /// call/checked argument, drop place, projection base — is MIR-0009.
    fn verify_drop_flags(&mut self) {
        let is_flag = |body: &MirBody, local: u32| {
            matches!(
                body.locals.get(local as usize).map(|d| &d.kind),
                Some(LocalKind::DropFlag)
            )
        };
        for (bi, block) in self.body.blocks.clone().iter().enumerate() {
            let bi = bi as u32;
            let flag_operand = |this: &mut Self, op: &Operand, what: &str| {
                if let Operand::Copy(place) | Operand::Move(place) = op {
                    if is_flag(this.body, place.local.0) {
                        this.err(
                            "MIR-0009",
                            bi,
                            format!("drop-flag _{} read as {what}", place.local.0),
                        );
                    }
                }
            };
            for (stmt, _) in &block.statements {
                if let Statement::Assign(_, rvalue) = stmt {
                    match rvalue {
                        Rvalue::Use(op) | Rvalue::UnOp(_, op) => {
                            flag_operand(self, op, "a statement operand")
                        }
                        Rvalue::BinOp(_, a, b) => {
                            flag_operand(self, a, "a statement operand");
                            flag_operand(self, b, "a statement operand");
                        }
                        Rvalue::Aggregate(_, ops) => {
                            for op in ops {
                                flag_operand(self, op, "an aggregate operand");
                            }
                        }
                        Rvalue::Discriminant(place) | Rvalue::RefOf { place, .. } => {
                            if is_flag(self.body, place.local.0) {
                                self.err(
                                    "MIR-0009",
                                    bi,
                                    format!("drop-flag _{} used as a place", place.local.0),
                                );
                            }
                        }
                        // A4: no operand and no place — a drop flag cannot appear here.
                        Rvalue::LayoutQuery { .. } => {}
                    }
                }
            }
            match &block.terminator.0 {
                Terminator::SwitchInt {
                    scrut: Operand::Move(place),
                    ..
                } if is_flag(self.body, place.local.0) => {
                    // The one permitted read form is a Copy of the bare flag local.
                    let local = place.local.0;
                    self.err(
                        "MIR-0009",
                        bi,
                        format!("drop-flag _{local} consumed by Move"),
                    );
                }
                Terminator::Call { callee, args, .. } => {
                    if let Callee::FnValue(op) = callee {
                        flag_operand(self, op, "an indirect callee");
                    }
                    for arg in args {
                        flag_operand(self, arg, "a call argument");
                    }
                }
                Terminator::Checked { args, .. } => {
                    for arg in args {
                        flag_operand(self, arg, "a checked argument");
                    }
                }
                // A1: the Trap message participates in drop-flag discipline (V-DROP-2).
                Terminator::Trap {
                    message: Some(op), ..
                } => flag_operand(self, op, "a trap message"),
                Terminator::Drop { place, .. } if is_flag(self.body, place.local.0) => {
                    let local = place.local.0;
                    self.err(
                        "MIR-0009",
                        bi,
                        format!("drop-flag _{local} used as a Drop place"),
                    );
                }
                _ => {}
            }
        }
    }

    // ---- V-IDX-1 definite initialization (C4.5e-0, review Finding 1) ----

    /// Every `Index(proof)` use must be *definitely preceded* by its defining `CheckIndex`
    /// on every execution path. The global name→base map (`verify_index_bindings`) alone
    /// accepts MIR whose check runs on only one branch of a join. Must-analysis over the
    /// CFG: `defined_in[b] = ⋂ preds' defined_out` (unvisited = top); a `CheckIndex`
    /// terminator defines its dest for its (single) normal successor, so uses in the
    /// defining block itself — which precede the terminator — correctly do not see it.
    /// Also enforces exactly one `CheckIndex` definition site per proof local (lowering
    /// mints a fresh proof local per check; a second site would let the same token witness
    /// two different checks).
    fn verify_proof_flow(&mut self) {
        let has_proofs = self
            .body
            .locals
            .iter()
            .any(|d| matches!(d.kind, LocalKind::IndexProof));
        if !has_proofs {
            return;
        }
        // Unique-definition rule.
        let mut def_counts: std::collections::HashMap<u32, usize> = Default::default();
        for block in &self.body.blocks {
            if let Terminator::Checked {
                op: CheckedOp::CheckIndex,
                dest,
                ..
            } = &block.terminator.0
            {
                *def_counts.entry(dest.0).or_insert(0) += 1;
            }
        }
        for (local, count) in &def_counts {
            if *count > 1 {
                self.err(
                    "MIR-0010",
                    0,
                    format!("proof local _{local} defined by {count} CheckIndex sites"),
                );
            }
        }
        // Must-dataflow to a fixpoint. `None` = top (unvisited).
        let block_count = self.body.blocks.len();
        let mut defined_in: Vec<Option<BTreeSet<u32>>> = vec![None; block_count];
        let entry = self.body.entry.0 as usize;
        if entry >= block_count {
            return; // already reported by the CFG check
        }
        defined_in[entry] = Some(BTreeSet::new());
        let mut work: VecDeque<usize> = VecDeque::new();
        work.push_back(entry);
        while let Some(bi) = work.pop_front() {
            let Some(in_set) = defined_in[bi].clone() else {
                continue;
            };
            let (successors, defines) = match &self.body.blocks[bi].terminator.0 {
                Terminator::Goto { target } => (vec![*target], None),
                Terminator::SwitchInt {
                    arms, otherwise, ..
                } => {
                    let mut s: Vec<BlockId> = arms.iter().map(|(_, b)| *b).collect();
                    s.push(*otherwise);
                    (s, None)
                }
                Terminator::Call { target, .. } | Terminator::Drop { target, .. } => {
                    (vec![*target], None)
                }
                Terminator::Checked {
                    op, dest, target, ..
                } => (
                    vec![*target],
                    matches!(op, CheckedOp::CheckIndex).then_some(dest.0),
                ),
                Terminator::Trap { .. } | Terminator::Return | Terminator::Unreachable => {
                    (Vec::new(), None)
                }
            };
            let mut out_set = in_set;
            if let Some(d) = defines {
                out_set.insert(d);
            }
            for succ in successors {
                let s = succ.0 as usize;
                if s >= block_count {
                    continue; // broken edge, reported elsewhere
                }
                let updated = match &defined_in[s] {
                    None => Some(out_set.clone()),
                    Some(current) => {
                        let met: BTreeSet<u32> = current.intersection(&out_set).copied().collect();
                        (met != *current).then_some(met)
                    }
                };
                if let Some(new_in) = updated {
                    defined_in[s] = Some(new_in);
                    work.push_back(s);
                }
            }
        }
        // Reporting pass: every Index(proof) used in a reachable block requires the proof
        // definitely defined at block entry.
        for (bi, block) in self.body.blocks.clone().iter().enumerate() {
            let Some(in_set) = &defined_in[bi] else {
                continue; // unreachable block
            };
            let in_set = in_set.clone();
            let mut places: Vec<Place> = Vec::new();
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
                Terminator::Trap {
                    message: Some(op), ..
                } => collect_operand_place(op, &mut places),
                _ => {}
            }
            for place in &places {
                for projection in &place.projection {
                    if let Projection::Index(proof) = projection {
                        if !in_set.contains(&proof.0) {
                            self.err(
                                "MIR-0010",
                                bi as u32,
                                format!(
                                    "Index(proof _{}) is not definitely preceded by its \
                                     CheckIndex on every path",
                                    proof.0
                                ),
                            );
                        }
                    }
                }
            }
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
                Terminator::Trap {
                    message: Some(op), ..
                } => collect_operand_place(op, &mut places),
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
                Terminator::Checked { op, args, .. } if !matches!(op, CheckedOp::CheckIndex) => {
                    for arg in args {
                        self.scan_operand_for_proof_misuse(arg, bi as u32);
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
            // A4: a layout query has no operands at all — its only input is a type.
            Rvalue::Discriminant(_) | Rvalue::RefOf { .. } | Rvalue::LayoutQuery { .. } => {
                Vec::new()
            }
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
/// Dataflow key of a place: its pure-Field path, or the whole local (path `[]`) when any
/// non-Field projection is involved (conservative).
/// A single step of a move-dataflow path (A5 / CD-038).
///
/// TYPED rather than a raw integer, deliberately. The earlier encoding flattened everything into
/// `Vec<u32>`, which meant distinct projection kinds could in principle produce the same integer
/// sequence — `VariantField(0, 1)` and a struct path `.0.1`, say. Nothing exploited that
/// (a local has exactly one type, so its projections are all one kind), but adding constant array
/// indices made a third kind share the space, and "provably safe today by an argument about
/// types" is a poor foundation for an analysis this load-bearing. Distinct kinds now cannot
/// compare equal, by construction.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum MovePathStep {
    /// Struct/tuple field by declaration index.
    Field(u32),
    /// Enum variant payload field: variant, then field index.
    VariantField(u32, u32),
    /// A statically known array element (`Projection::ConstIndex`).
    ConstIndex(u64),
}

/// The move-dataflow key for a place: its local plus a precise path.
///
/// Sub-place precision is what lets sibling components move independently. `VariantField`
/// distinguishes `_6.v0.0` from `_6.v0.1` (DEV-079 — collapsing them made every enum variant with
/// two or more droppable payload fields fail V-MOVE-1, producing MIR that lowering accepted and
/// verification rejected). `ConstIndex` does the same for array elements (DEV-086), which is what
/// makes consuming array patterns and by-value array iteration expressible at all.
///
/// `Deref` and the proof-backed `Index` still collapse to the whole local. That is not
/// conservatism to be removed later: neither denotes a statically-known sub-place — a dynamic
/// proof's value is not known here — so there is nothing precise to say about them.
fn moved_key(place: &Place) -> (u32, Vec<MovePathStep>) {
    let mut path = Vec::new();
    for proj in &place.projection {
        match proj {
            Projection::Field(i) => path.push(MovePathStep::Field(*i)),
            Projection::VariantField(v, i) => path.push(MovePathStep::VariantField(*v, *i)),
            Projection::ConstIndex(i) => path.push(MovePathStep::ConstIndex(*i)),
            Projection::Deref | Projection::Index(_) => return (place.local.0, Vec::new()),
        }
    }
    (place.local.0, path)
}

fn paths_prefix_related(a: &[MovePathStep], b: &[MovePathStep]) -> bool {
    let n = a.len().min(b.len());
    a[..n] == b[..n]
}

fn may_need_drop(ty: &MirTy) -> bool {
    match ty {
        MirTy::Struct(..) | MirTy::Enum(..) | MirTy::String | MirTy::Core(..) => true,
        MirTy::Tuple(elems) => elems.iter().any(may_need_drop),
        MirTy::Array(elem, _) => may_need_drop(elem),
        _ => false,
    }
}

/// WP-C5.3d-1b: the table is `drop_plan::variant_payloads`, the one derivation the verifier, the
/// interpreter's place typing, the drop plan, and the backend's type emission all read. It had
/// been written out three times — here, in `interp`, and in `emit_types` — with the variant
/// indices agreeing only by inspection.
///
/// A2 (CE3): `Ordering` has exactly three fieldless variants (`Less` = 0, `Equal` = 1,
/// `Greater` = 2); any other index is invalid and fails verification, which falls out of indexing
/// the table.
fn variant_payload(
    program: &MirProgram,
    enum_ref: &EnumRef,
    args: &[MirTy],
    variant: u32,
) -> Option<Vec<MirTy>> {
    super::drop_plan::variant_payloads(enum_ref, args, &program.types)
        .and_then(|variants| variants.get(variant as usize).cloned())
}

fn is_vec_runtime_fn(rt: RuntimeFn) -> bool {
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

/// 0.1-A7 (WP-C4.7-6.1): the `Box<T>` group. Schematic in `T`, resolved from the argument.
fn is_box_runtime_fn(rt: RuntimeFn) -> bool {
    use RuntimeFn::*;
    matches!(rt, BoxNew | BoxIntoInner)
}

fn is_slice_runtime_fn(rt: RuntimeFn) -> bool {
    use RuntimeFn::*;
    matches!(rt, SliceNew | SliceNewMut | SliceLen | SliceIsEmpty)
}

fn is_map_runtime_fn(rt: RuntimeFn) -> bool {
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

/// A1 V-STR-2: is `ty` a `String`, or a `str` behind any depth of reference? Such operands
/// must not appear in structural `BinOp`/`UnOp` — comparisons route through `StrEq`/`StrCmp`.
fn is_stringish(ty: &MirTy) -> bool {
    match ty {
        MirTy::String | MirTy::Str => true,
        MirTy::Ref { inner, .. } => is_stringish(inner),
        _ => false,
    }
}

fn runtime_sig(rt: RuntimeFn) -> (Vec<MirTy>, MirTy) {
    use RuntimeFn::*;
    let str_ref = || MirTy::Ref {
        mutable: false,
        inner: Box::new(MirTy::Str),
    };
    let string_ref = |mutable| MirTy::Ref {
        mutable,
        inner: Box::new(MirTy::String),
    };
    match rt {
        PrintlnInt64 | PrintInt64 => (vec![MirTy::Int64], MirTy::Unit),
        PrintlnUInt64 | PrintUInt64 => (vec![MirTy::UInt64], MirTy::Unit),
        PrintlnBool | PrintBool => (vec![MirTy::Bool], MirTy::Unit),
        PrintlnFloat64 | PrintFloat64 => (vec![MirTy::Float64], MirTy::Unit),
        // --- A1 (CD-031) String/str surface ---
        PrintlnStr | PrintStr => (vec![str_ref()], MirTy::Unit),
        StringNew => (vec![], MirTy::String),
        StringFromStr => (vec![str_ref()], MirTy::String),
        StringLen => (vec![string_ref(false)], MirTy::UInt64),
        StringIsEmpty => (vec![string_ref(false)], MirTy::Bool),
        StringPushStr => (vec![string_ref(true), str_ref()], MirTy::Unit),
        StringClear => (vec![string_ref(true)], MirTy::Unit),
        StringAsStr => (vec![string_ref(false)], str_ref()),
        StringClone => (vec![string_ref(false)], MirTy::String),
        StringContains => (vec![string_ref(false), str_ref()], MirTy::Bool),
        StrLen => (vec![str_ref()], MirTy::UInt64),
        StrIsEmpty => (vec![str_ref()], MirTy::Bool),
        StrToString => (vec![str_ref()], MirTy::String),
        StrEq => (vec![str_ref(), str_ref()], MirTy::Bool),
        StrCmp => (vec![str_ref(), str_ref()], MirTy::Int64),
        // 0.1-A3 (f-3b): Char ops.
        PrintlnChar | PrintChar => (vec![MirTy::Char], MirTy::Unit),
        StringPushChar => (vec![string_ref(true), MirTy::Char], MirTy::Unit),
        StringPopChar => (
            vec![string_ref(true)],
            MirTy::Enum(EnumRef::CoreOption, vec![MirTy::Char]),
        ),
        // 0.1-A5 (A4-2d): string chars iteration (fixed types — no schematic param).
        CharsIterNew => (
            vec![str_ref()],
            MirTy::Core(crate::hir::CoreType::CharsIter, Vec::new()),
        ),
        CharsIterNext => (
            vec![MirTy::Ref {
                mutable: true,
                inner: Box::new(MirTy::Core(crate::hir::CoreType::CharsIter, Vec::new())),
            }],
            MirTy::Enum(EnumRef::CoreOption, vec![MirTy::Char]),
        ),
        // Box ops are schematic in T — resolved by `box_runtime_sig`, never this fixed table.
        BoxNew | BoxIntoInner => {
            unreachable!("Box ops resolve through box_runtime_sig, not runtime_sig")
        }
        // Vec ops are schematic in T — resolved by `vec_runtime_sig`, never this fixed table.
        VecNew | VecWithCapacity | VecPush | VecPop | VecLen | VecIsEmpty | VecIndexGet
        | VecReplace | VecRemove | VecClear | VecIterNew | VecIterNext | VecGetRef
        | VecGetMutRef => {
            unreachable!("Vec ops resolve through vec_runtime_sig, not runtime_sig")
        }
        HashMapNew | HashMapInsert | HashMapGet | HashMapLen | HashMapIsEmpty
        | HashMapContainsKey | HashMapKeysIterNew | HashMapKeysIterNext => {
            unreachable!("HashMap ops resolve through map_runtime_sig, not runtime_sig")
        }
        SliceNew | SliceNewMut | SliceLen | SliceIsEmpty => {
            unreachable!("slice ops resolve through slice_runtime_sig, not runtime_sig")
        }
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
        // A4: no places are read by a layout query.
        Rvalue::LayoutQuery { .. } => {}
    }
}
