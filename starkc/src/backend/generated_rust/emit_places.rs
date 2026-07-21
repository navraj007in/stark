//! WP-C5.2b — place emission. Scope is bare locals only (no projections): struct/tuple field,
//! variant field, `Deref`, `Index`, `ConstIndex` all require aggregates/references, which
//! `emit_types::emit_ty` does not admit yet (WP-C5.2c/C5.3).

use super::BackendDiagnostic;
use crate::mir::Place;

/// Matches MIR's own dump format (`_0`, `_1`, ...) one-to-one, which is also a valid Rust
/// identifier and (as a bonus, not the reason it was chosen) already suppresses Rust's
/// `unused_variables` lint on its own, since any leading-underscore identifier does.
pub fn local_name(local: u32) -> String {
    format!("_{local}")
}

pub fn emit_place(place: &Place) -> Result<String, BackendDiagnostic> {
    if !place.projection.is_empty() {
        return Err(BackendDiagnostic::Unsupported(format!(
            "WP-C5.2b supports only bare locals, no projections; place {place:?} lands in \
             WP-C5.2c/C5.3 (field/variant/index/deref projections)"
        )));
    }
    Ok(local_name(place.local.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{LocalId, Projection};

    #[test]
    fn bare_local_emits_its_dump_name() {
        let place = Place::local(LocalId(3));
        assert_eq!(emit_place(&place).unwrap(), "_3");
    }

    #[test]
    fn projected_place_is_unsupported() {
        let place = Place {
            local: LocalId(0),
            projection: vec![Projection::Field(1)],
        };
        assert!(matches!(
            emit_place(&place),
            Err(BackendDiagnostic::Unsupported(_))
        ));
    }
}
