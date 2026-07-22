//! WP-C5.4a: the linkage preflight. ONE read-only validation/indexing pass over verified MIR,
//! run before any generated source is assembled (§6). It is a backend validation structure, not a
//! second instance-discovery graph, resolver, or monomorphiser (§2.2, §6): it consumes the
//! concrete, transitively-reachable body set it is given and proves that set is internally
//! complete and consistently referenced, refusing deterministically (`BackendDiagnostic`) BEFORE
//! rustc if it is not.
//!
//! The governing principle (§0): `Instance.symbol` is the canonical callable identity;
//! `MirProgram.bodies` is the authoritative concrete-body set. This pass validates and indexes
//! those artifacts; it does not reconstruct, repair, or extend them (§6.4).

use super::{emit_types, mangle, BackendDiagnostic};
use crate::mir::{
    Callee, Constant, Instance, MirBody, MirProgram, Operand, Rvalue, Statement, Terminator,
};
use std::collections::BTreeMap;

/// One linked body plus the Rust name it will emit under. Borrows the body from the program; the
/// index never owns or clones bodies (§6.4 forbids the pass from materialising a body).
pub struct LinkedInstance<'a> {
    pub body: &'a MirBody,
    pub rust_name: String,
}

/// The deterministic read-only linkage index (§6.1). Keyed by canonical symbol, so lookup order
/// is the same canonical order `MirProgram.bodies` is sorted in.
pub struct LinkageIndex<'a> {
    pub by_symbol: BTreeMap<String, LinkedInstance<'a>>,
}

impl<'a> LinkageIndex<'a> {
    /// Resolve a referenced instance to its defining body, proving §6.2.4 consistency: the symbol
    /// must name exactly one body, and that body's `item`/`type_args` must match the reference.
    /// A missing symbol is a naming-no-body defect; a present symbol with a different identity is
    /// the "one canonical symbol, different item/type-argument identities" defect (§5.3). Neither
    /// is repaired here (§6.4) — both are refused so rustc never sees them.
    pub fn resolve(&self, instance: &Instance) -> Result<&LinkedInstance<'a>, BackendDiagnostic> {
        let linked = self.by_symbol.get(&instance.symbol).ok_or_else(|| {
            BackendDiagnostic::Unsupported(format!(
                "linkage: reference to instance `{}` names no body in the linked program \
                 (backend/verifier contract defect: the concrete body set is incomplete)",
                instance.symbol
            ))
        })?;
        if linked.body.instance.item != instance.item
            || linked.body.instance.type_args != instance.type_args
        {
            return Err(BackendDiagnostic::Unsupported(format!(
                "linkage: canonical symbol `{}` is referenced with item/type-args \
                 ({:?}, {:?}) but its body is defined with ({:?}, {:?}) \
                 (canonical-identity defect: one symbol, two identities)",
                instance.symbol,
                instance.item,
                instance.type_args,
                linked.body.instance.item,
                linked.body.instance.type_args,
            )));
        }
        Ok(linked)
    }
}

/// How an instance is referenced. Direct calls and function values resolve through the SAME path
/// (§6.2), but the kind is carried so diagnostics and the C5.4b/c reachability tests can tell a
/// function referenced only through `Constant::FnPtr` from one that is also directly called.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RefKind {
    DirectCall,
    FnValue,
}

/// Build and validate the linkage index (§6.2). This is the deterministic refusal boundary: every
/// check here fails before generated source exists, so no out-of-subset or internally-inconsistent
/// program reaches rustc.
pub fn build(program: &MirProgram) -> Result<LinkageIndex<'_>, BackendDiagnostic> {
    // §6.2.1 + §6.2.2 + §6.2.3: strictly-sorted, unique symbols; unique generated Rust names.
    let mut by_symbol: BTreeMap<String, LinkedInstance<'_>> = BTreeMap::new();
    let mut by_rust_name: BTreeMap<String, String> = BTreeMap::new();
    let mut prev_symbol: Option<&str> = None;
    let mut entry_bodies = 0usize;

    for body in &program.bodies {
        let symbol = &body.instance.symbol;

        // §6.2.1: canonical symbols strictly sorted. The MIR contract promises this
        // (`MirProgram.bodies` doc, produced by `lower.rs`'s sort); the backend VALIDATES it
        // rather than re-sorting and silently normalising a producer defect (§6.4).
        if let Some(prev) = prev_symbol {
            if prev >= symbol.as_str() {
                return Err(BackendDiagnostic::Unsupported(format!(
                    "linkage: bodies are not strictly sorted by canonical symbol \
                     (`{prev}` before `{symbol}`) — MIR contract violation, not repaired here"
                )));
            }
        }
        prev_symbol = Some(symbol);

        // §6.2.7: every body's parameter and return types must be C5-representable. `emit_ty` is
        // the representability oracle emission itself uses, so this can only refuse what emission
        // would already refuse — just earlier, with a linkage-facing message.
        for (i, param) in body.params.iter().enumerate() {
            emit_types::emit_ty(param).map_err(|e| annotate(symbol, &format!("param {i}"), e))?;
        }
        emit_types::emit_ty(&body.ret).map_err(|e| annotate(symbol, "return type", e))?;

        let rust_name = mangle::function_name_for_symbol(symbol);

        // §6.2.5: exactly one entry body; §6.2.6: no non-entry body maps to Rust `main`.
        if symbol == mangle::ENTRY_SYMBOL {
            entry_bodies += 1;
        } else if rust_name == "main" {
            return Err(BackendDiagnostic::Unsupported(format!(
                "linkage: non-entry body `{symbol}` maps to Rust `main` — generated-name collision \
                 with the entry wrapper"
            )));
        }

        // §6.2.3: generated Rust name uniqueness. Two canonical symbols mapping to one Rust name
        // (§5.3) would silently become one function in the crate — the exact failure `mangle`'s
        // injectivity exists to prevent, re-checked here over the actual body set.
        if let Some(other) = by_rust_name.insert(rust_name.clone(), symbol.clone()) {
            return Err(BackendDiagnostic::Unsupported(format!(
                "linkage: canonical symbols `{other}` and `{symbol}` both generate Rust function \
                 name `{rust_name}` (sanitiser injectivity defect or duplicate body)"
            )));
        }

        // §6.2.2: symbol uniqueness (a duplicate body under one canonical symbol, §5.3).
        if by_symbol
            .insert(symbol.clone(), LinkedInstance { body, rust_name })
            .is_some()
        {
            return Err(BackendDiagnostic::Unsupported(format!(
                "linkage: duplicate body for canonical symbol `{symbol}`"
            )));
        }
    }

    // §6.2.5: exactly one entry body. (`emit_program` already refuses zero entries, but the
    // linkage contract states the invariant in one place and also rejects a second entry.)
    if entry_bodies != 1 {
        return Err(BackendDiagnostic::Unsupported(format!(
            "linkage: expected exactly one `{}` entry body, found {entry_bodies}",
            mangle::ENTRY_SYMBOL
        )));
    }

    let index = LinkageIndex { by_symbol };

    // §6.2: every referenced instance resolves to exactly one body with matching identity. Direct
    // callees and function constants go through the SAME resolution path (`index.resolve`), so a
    // function reachable only through `Constant::FnPtr` is validated identically to a called one.
    for body in &program.bodies {
        let mut result = Ok(());
        visit_instance_refs(body, &mut |instance, _kind| {
            if result.is_ok() {
                if let Err(e) = index.resolve(instance) {
                    result = Err(e);
                }
            }
        });
        result?;
    }

    Ok(index)
}

fn annotate(symbol: &str, where_: &str, e: BackendDiagnostic) -> BackendDiagnostic {
    match e {
        BackendDiagnostic::Unsupported(msg) => BackendDiagnostic::Unsupported(format!(
            "linkage: body `{symbol}` {where_} is not C5-representable: {msg}"
        )),
        other => other,
    }
}

// --------------------------------------------------------------- operand walker --

/// §6.3: the ONE exhaustive instance-reference walker. Every operand-bearing MIR location routes
/// through here, so linkage, and any future liveness/reachability check, agree on which locations
/// carry an operand. Every `Rvalue`/`Terminator`/`Statement`/`Callee`/`Operand`/`Constant` variant
/// is matched WITHOUT a wildcard: a new variant fails to compile until this walker is updated
/// (§6.3), rather than silently escaping linkage validation.
pub fn visit_instance_refs<F: FnMut(&Instance, RefKind)>(body: &MirBody, f: &mut F) {
    for block in &body.blocks {
        for (stmt, _info) in &block.statements {
            match stmt {
                Statement::Assign(_place, rvalue) => visit_rvalue(rvalue, f),
                Statement::Nop => {}
            }
        }
        let (term, _info) = &block.terminator;
        visit_terminator(term, f);
    }
}

fn visit_rvalue<F: FnMut(&Instance, RefKind)>(rvalue: &Rvalue, f: &mut F) {
    match rvalue {
        Rvalue::Use(op) => visit_operand(op, f),
        Rvalue::UnOp(_, op) => visit_operand(op, f),
        Rvalue::BinOp(_, a, b) => {
            visit_operand(a, f);
            visit_operand(b, f);
        }
        Rvalue::Aggregate(_kind, ops) => {
            for op in ops {
                visit_operand(op, f);
            }
        }
        // Places and pure layout queries carry no operand and thus no instance reference.
        Rvalue::Discriminant(_) | Rvalue::RefOf { .. } | Rvalue::LayoutQuery { .. } => {}
    }
}

fn visit_terminator<F: FnMut(&Instance, RefKind)>(term: &Terminator, f: &mut F) {
    match term {
        Terminator::Call { callee, args, .. } => {
            visit_callee(callee, f);
            for arg in args {
                visit_operand(arg, f);
            }
        }
        Terminator::SwitchInt { scrut, .. } => visit_operand(scrut, f),
        Terminator::Checked { args, .. } => {
            for arg in args {
                visit_operand(arg, f);
            }
        }
        Terminator::Trap { message, .. } => {
            if let Some(op) = message {
                visit_operand(op, f);
            }
        }
        // No operands: `Drop` carries a place, and these three carry only control flow.
        Terminator::Drop { .. }
        | Terminator::Goto { .. }
        | Terminator::Return
        | Terminator::Unreachable => {}
    }
}

fn visit_callee<F: FnMut(&Instance, RefKind)>(callee: &Callee, f: &mut F) {
    match callee {
        Callee::Instance(instance) => f(instance, RefKind::DirectCall),
        // The indirect-call target is an operand: if it is a `Constant::FnPtr` it names an
        // instance (validated as a function value below), otherwise it is a local read.
        Callee::FnValue(op) => visit_operand(op, f),
        Callee::Runtime(_) => {}
    }
}

fn visit_operand<F: FnMut(&Instance, RefKind)>(op: &Operand, f: &mut F) {
    match op {
        Operand::Const(constant) => visit_constant(constant, f),
        // A place read references no instance identity of its own.
        Operand::Copy(_) | Operand::Move(_) => {}
    }
}

fn visit_constant<F: FnMut(&Instance, RefKind)>(constant: &Constant, f: &mut F) {
    match constant {
        Constant::FnPtr(instance) => f(instance, RefKind::FnValue),
        Constant::Int(_, _)
        | Constant::Float(_, _)
        | Constant::Bool(_)
        | Constant::Unit
        | Constant::Str(_) => {}
    }
}
