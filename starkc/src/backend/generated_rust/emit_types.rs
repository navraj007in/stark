//! §6.2 type mapping. C5.1b implements the primitive rows only (what an empty `fn main()`
//! needs); `WP-C5.1.md`'s `MirTy` matrix records the full IN/OUT split -- aggregates, enums,
//! references, and every `Core(CoreType, _)` payload return `Unsupported` here until WP-C5.2/
//! C5.3 implement their §6.2 rows.

use super::BackendDiagnostic;
use crate::mir::MirTy;

pub fn emit_ty(ty: &MirTy) -> Result<String, BackendDiagnostic> {
    Ok(match ty {
        MirTy::Int8 => "i8".to_string(),
        MirTy::Int16 => "i16".to_string(),
        MirTy::Int32 => "i32".to_string(),
        MirTy::Int64 => "i64".to_string(),
        MirTy::UInt8 => "u8".to_string(),
        MirTy::UInt16 => "u16".to_string(),
        MirTy::UInt32 => "u32".to_string(),
        MirTy::UInt64 => "u64".to_string(),
        MirTy::Float32 => "f32".to_string(),
        MirTy::Float64 => "f64".to_string(),
        MirTy::Bool => "bool".to_string(),
        MirTy::Char => "char".to_string(),
        MirTy::Unit => "()".to_string(),
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "MirTy {other:?} has no C5.1b generated-Rust representation yet -- lands in \
                 WP-C5.2 (scalars/aggregates) or WP-C5.3 (enums); see WP-C5.1.md's MirTy matrix"
            )))
        }
    })
}
