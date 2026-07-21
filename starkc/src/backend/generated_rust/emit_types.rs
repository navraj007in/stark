//! WP-C5.2a — §6.2 type mapping and constant emission ("primitive values and constants").
//! `WP-C5.1.md`'s `MirTy` matrix records the full IN/OUT split -- aggregates, enums, references,
//! and every `Core(CoreType, _)` payload return `Unsupported` from [`emit_ty`] until WP-C5.2b/c
//! (aggregates/control flow) or WP-C5.3 (enums) implement their §6.2 rows. `String`/`str` are
//! deferred alongside output support (WP-C5.2c/d, wherever `RuntimeFn` calls first lower), since
//! a constant string with nothing to do with it yet is not independently useful to prove.

use super::BackendDiagnostic;
use crate::mir::{Constant, MirTy};

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
                "MirTy {other:?} has no C5.2 generated-Rust representation yet -- aggregates \
                 land in WP-C5.2b/c, enums in WP-C5.3; see WP-C5.1.md's MirTy matrix"
            )))
        }
    })
}

/// A Rust expression, not necessarily a single literal token (e.g. a `Char` constant becomes
/// `char::from_u32(...).unwrap()`, and a negative float becomes a unary-negation expression) --
/// callers must not assume the result can be used only where a bare literal is legal, only that
/// it is valid wherever a Rust expression is.
pub fn emit_constant(c: &Constant) -> Result<String, BackendDiagnostic> {
    Ok(match c {
        Constant::Bool(b) => b.to_string(),
        Constant::Unit => "()".to_string(),
        Constant::Int(value, MirTy::Char) => emit_char_constant(*value)?,
        Constant::Int(value, ty) => {
            let suffix = int_suffix(ty)?;
            format!("{value}{suffix}")
        }
        Constant::Float(value, ty) => emit_float_constant(*value, ty)?,
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "Constant {other:?} has no C5.2a generated-Rust representation yet -- `FnPtr` \
                 lands in WP-C5.2d/C5.4c (function values), `Str` lands alongside String/output \
                 support"
            )))
        }
    })
}

fn int_suffix(ty: &MirTy) -> Result<&'static str, BackendDiagnostic> {
    Ok(match ty {
        MirTy::Int8 => "i8",
        MirTy::Int16 => "i16",
        MirTy::Int32 => "i32",
        MirTy::Int64 => "i64",
        MirTy::UInt8 => "u8",
        MirTy::UInt16 => "u16",
        MirTy::UInt32 => "u32",
        MirTy::UInt64 => "u64",
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "integer constant with non-integer MirTy {other:?}"
            )))
        }
    })
}

/// A `Char` constant is `Constant::Int(codepoint, MirTy::Char)` (`mir::lower`'s own encoding,
/// f-3b: "a Char literal is its Unicode scalar codepoint, typed Char") -- there is no Rust
/// numeric-literal suffix for `char`, so this reconstructs the value via `char::from_u32`. The
/// `.unwrap()` cannot fail for a program that reached verified MIR: the front end already
/// guarantees every `Char` constant is a valid Unicode scalar value: a failure here would mean a
/// STARK compiler defect upstream of this backend, not a reachable user-facing condition.
fn emit_char_constant(codepoint: i128) -> Result<String, BackendDiagnostic> {
    let codepoint = u32::try_from(codepoint).map_err(|_| {
        BackendDiagnostic::Unsupported(format!(
            "Char constant codepoint {codepoint} does not fit in u32 -- verified MIR should be \
             unreachable here"
        ))
    })?;
    Ok(format!("(char::from_u32({codepoint}u32).unwrap())"))
}

fn emit_float_constant(value: f64, ty: &MirTy) -> Result<String, BackendDiagnostic> {
    let typed_f64 = format_f64_literal_typed(value);
    match ty {
        MirTy::Float64 => Ok(typed_f64),
        // No f32 round-trip literal formatter is implemented here; casting an already-typed f64
        // expression preserves the value without one, at the cost of one extra cast expression.
        MirTy::Float32 => Ok(format!("(({typed_f64}) as f32)")),
        other => Err(BackendDiagnostic::Unsupported(format!(
            "float constant with non-float MirTy {other:?}"
        ))),
    }
}

/// Rust's `Debug` formatting for `f64` (unlike `Display`) always includes a decimal point or
/// exponent, so the result is guaranteed to parse back as a Rust float literal once suffixed --
/// and, per `std`, is already the shortest string that round-trips to the same bit pattern.
/// NaN/infinity have no Rust literal syntax at all, so those branches return an already-typed
/// `f64::NAN`/`f64::INFINITY`/`f64::NEG_INFINITY` expression instead of a bare literal --
/// callers must not append a further type suffix to this function's result.
fn format_f64_literal_typed(value: f64) -> String {
    if value.is_nan() {
        "f64::NAN".to_string()
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            "f64::INFINITY".to_string()
        } else {
            "f64::NEG_INFINITY".to_string()
        }
    } else {
        format!("{value:?}f64")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compiles_as_a_rust_expression(src: &str) -> bool {
        // A cheap, dependency-free "is this syntactically a Rust expression" check: parenthesize
        // it inside a const-eval-free position and let `rustc --edition 2021 --crate-type lib`
        // parse-check a throwaway file. Skipped if rustc is unavailable, matching the project's
        // existing `rustc_available()` convention (`tests/spike_genrust.rs`,
        // `tests/native_c5_1b_skeleton.rs`) rather than failing the unit test suite in an
        // environment with no Rust toolchain.
        let Ok(status) = std::process::Command::new("rustc")
            .arg("--version")
            .output()
        else {
            return true;
        };
        if !status.status.success() {
            return true;
        }
        let dir = std::env::temp_dir().join(format!(
            "stark_c5_2a_lit_check_{}_{}",
            std::process::id(),
            src.len()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("check.rs");
        let _ = std::fs::write(
            &file,
            format!("#[allow(dead_code)]\nfn f() {{ let _ = {src}; }}\n"),
        );
        let ok = std::process::Command::new("rustc")
            .arg("--edition")
            .arg("2021")
            .arg("--crate-type")
            .arg("lib")
            .arg("-o")
            .arg(dir.join("out.rlib"))
            .arg(&file)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        let _ = std::fs::remove_dir_all(&dir);
        ok
    }

    #[test]
    fn int_constants_round_trip_through_rustc() {
        for (value, ty, expected_suffix) in [
            (0i128, MirTy::Int32, "i32"),
            (-5i128, MirTy::Int8, "i8"),
            (255i128, MirTy::UInt8, "u8"),
            (u64::MAX as i128, MirTy::UInt64, "u64"),
        ] {
            let out = emit_constant(&Constant::Int(value, ty)).unwrap();
            assert!(out.ends_with(expected_suffix), "{out}");
            assert!(compiles_as_a_rust_expression(&out), "{out}");
        }
    }

    #[test]
    fn bool_and_unit_constants() {
        assert_eq!(emit_constant(&Constant::Bool(true)).unwrap(), "true");
        assert_eq!(emit_constant(&Constant::Bool(false)).unwrap(), "false");
        assert_eq!(emit_constant(&Constant::Unit).unwrap(), "()");
    }

    #[test]
    fn char_constant_reconstructs_via_from_u32() {
        let out = emit_constant(&Constant::Int(0x41, MirTy::Char)).unwrap(); // 'A'
        assert!(out.contains("char::from_u32"), "{out}");
        assert!(compiles_as_a_rust_expression(&out), "{out}");
    }

    #[test]
    fn float_constants_including_edge_cases_compile() {
        for (value, ty) in [
            (0.0f64, MirTy::Float64),
            (-0.0f64, MirTy::Float64),
            (3.5f64, MirTy::Float32),
            (f64::NAN, MirTy::Float64),
            (f64::INFINITY, MirTy::Float64),
            (f64::NEG_INFINITY, MirTy::Float32),
        ] {
            let out = emit_constant(&Constant::Float(value, ty)).unwrap();
            assert!(compiles_as_a_rust_expression(&out), "{out}");
        }
    }

    #[test]
    fn unsupported_constants_are_reported_not_guessed() {
        assert!(matches!(
            emit_constant(&Constant::Str("x".to_string())),
            Err(BackendDiagnostic::Unsupported(_))
        ));
    }
}
