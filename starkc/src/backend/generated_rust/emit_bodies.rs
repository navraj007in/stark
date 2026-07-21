//! Function-body emission. C5.1b implements only the trivial `Unit`-returning, single-block,
//! no-real-work shape needed to prove the backend/runtime/build pipeline end-to-end on an
//! empty `fn main() { }` -- general place/rvalue/control-flow lowering is WP-C5.2's job
//! (§14 C5.2a-e), and this function is replaced then, not extended piecemeal.

use super::BackendDiagnostic;
use crate::mir::{Constant, MirBody, MirTy, Operand, Place, Rvalue, Statement, Terminator};

/// Emits a function body for the C5.1b-supported shape only: a single block, a `Unit`-typed
/// return slot, and statements that only ever (re)establish that the return slot is `Unit` --
/// nothing else is lowered here yet.
pub fn emit_trivial_unit_body(body: &MirBody) -> Result<String, BackendDiagnostic> {
    if !matches!(body.ret, MirTy::Unit) {
        return Err(BackendDiagnostic::Unsupported(
            "WP-C5.1b's trivial-body path only proves a Unit-returning entry; general return \
             values land in WP-C5.2a"
                .into(),
        ));
    }
    // WP-C4.5's synthetic-return-slot lowering appends a trailing `Unreachable` block after a
    // straight-line function's real entry block (see `bb1` in `fn main() { }`'s own dump) --
    // dead by construction, since nothing ever jumps to it. C5.1b reads only the entry block
    // and defensively confirms every other block is exactly that shape, rather than silently
    // ignoring blocks it does not understand.
    let entry_index = body.entry.0 as usize;
    let entry_block = body.blocks.get(entry_index).ok_or_else(|| {
        BackendDiagnostic::Unsupported("entry block index out of range".to_string())
    })?;
    for (bi, block) in body.blocks.iter().enumerate() {
        if bi == entry_index {
            continue;
        }
        let trivially_dead =
            block.statements.is_empty() && matches!(block.terminator.0, Terminator::Unreachable);
        if !trivially_dead {
            return Err(BackendDiagnostic::Unsupported(format!(
                "WP-C5.1b supports only an entry block plus trivially-dead `Unreachable` \
                 blocks; real multi-block control flow lands in WP-C5.2c (bb{bi} is not dead)"
            )));
        }
    }
    for (stmt, _) in &entry_block.statements {
        match stmt {
            Statement::Nop => {}
            Statement::Assign(place, Rvalue::Use(Operand::Const(Constant::Unit)))
                if is_return_slot(place) => {}
            other => {
                return Err(BackendDiagnostic::Unsupported(format!(
                    "WP-C5.1b supports only Nop/return-slot-Unit statements; {other:?} lands \
                     in WP-C5.2b"
                )))
            }
        }
    }
    match &entry_block.terminator.0 {
        Terminator::Return => {}
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "WP-C5.1b supports only a Return terminator; {other:?} lands in WP-C5.2"
            )))
        }
    }
    Ok("{\n}\n".to_string())
}

fn is_return_slot(place: &Place) -> bool {
    place.local.0 == 0 && place.projection.is_empty()
}
