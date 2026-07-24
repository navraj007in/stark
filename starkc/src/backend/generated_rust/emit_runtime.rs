//! `RuntimeFn`/`Callee::Runtime` call emission — bridges MIR runtime calls to `stark_runtime`
//! functions. Each arm renders the runtime call as a Rust expression from the already-emitted
//! argument expressions; the assignment/slot wrapping around it is `emit_bodies`' concern.
//!
//! WP-C6.3a activated the String/str and str-output surface. The remaining `RuntimeFn` groups
//! (Vec, Box, slices, iterators, HashMap, formatting of non-string values) land with their
//! sub-packages and stay `Unsupported` until then.

use super::{emit_types, BackendDiagnostic};
use crate::mir::{MirTy, RuntimeFn};

/// Render `rt(args...)` as a Rust expression. `args` are the argument operands already emitted by
/// `emit_operand`; a String/str receiver arrives as `&String`/`&str` (deref-coercing to the `&str`
/// the `stark_runtime::string` helpers take) and a `&mut self` receiver as `&mut String`.
/// `dest_ty` is the type of the place being assigned — an `Option`-returning runtime fn wraps its
/// Rust `Option` into the program's generated Option enum, whose name `dest_ty` supplies.
pub fn emit_runtime_call(
    rt: RuntimeFn,
    args: &[String],
    dest_ty: &MirTy,
) -> Result<String, BackendDiagnostic> {
    use RuntimeFn::*;
    // A small helper: the argument at `i`, wrapped so method/`.as_bytes()` suffixes bind correctly.
    let arg = |i: usize| format!("({})", args[i]);
    Ok(match rt {
        // --- str output (06 PRINT-DISPLAY-001): bytes submitted through the runtime sink. ---
        PrintlnStr => format!("stark_runtime::output::stdout_line({}.as_bytes())", arg(0)),
        PrintStr => format!("stark_runtime::output::stdout_bytes({}.as_bytes())", arg(0)),

        // --- String construction / conversion ---
        StringNew => "stark_runtime::string::new()".to_string(),
        StringFromStr => format!("stark_runtime::string::from_str({})", arg(0)),
        StrToString => format!("stark_runtime::string::to_string({})", arg(0)),
        StringClone => format!("stark_runtime::string::clone_string({})", arg(0)),
        StringAsStr => format!("stark_runtime::string::as_str({})", arg(0)),

        // --- String / str queries ---
        StringLen | StrLen => format!("stark_runtime::string::len({})", arg(0)),
        StringIsEmpty | StrIsEmpty => format!("stark_runtime::string::is_empty({})", arg(0)),
        StringContains => format!("stark_runtime::string::contains({}, {})", arg(0), arg(1)),
        StrEq => format!("stark_runtime::string::eq({}, {})", arg(0), arg(1)),
        StrCmp => format!("stark_runtime::string::cmp({}, {})", arg(0), arg(1)),

        // --- String mutation ---
        StringPushStr => format!("stark_runtime::string::push_str({}, {})", arg(0), arg(1)),
        StringClear => format!("stark_runtime::string::clear({})", arg(0)),
        StringPushChar => format!("stark_runtime::string::push_char({}, {})", arg(0), arg(1)),
        // Returns `Option<Char>` — wrapped into the generated Option enum.
        StringPopChar => wrap_option(
            &format!("stark_runtime::string::pop_char({})", arg(0)),
            dest_ty,
        )?,

        // --- Char output (the Char is a Copy `char` value) ---
        PrintlnChar => format!("stark_runtime::string::println_char({})", arg(0)),
        PrintChar => format!("stark_runtime::string::print_char({})", arg(0)),

        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "RuntimeFn {other:?} has no generated-Rust representation yet -- it lands with its \
                 WP-C6.3 sub-package (Vec/Box/slices/iterators/HashMap/formatting)"
            )))
        }
    })
}

/// Wrap a Rust `Option<T>`-producing expression into the program's generated Option enum named by
/// `dest_ty`. The generated enum uses the shared variant layout `V0 = None`, `V1(T) = Some(T)`
/// (`variant_payloads`), so both engines agree on discriminants.
fn wrap_option(inner: &str, dest_ty: &MirTy) -> Result<String, BackendDiagnostic> {
    let name = emit_types::nominal_type_name(dest_ty).ok_or_else(|| {
        BackendDiagnostic::Unsupported(format!(
            "Option-returning runtime fn assigned to non-Option dest {dest_ty:?}"
        ))
    })?;
    // The generated enum's variants are tuple variants (`V0()`, `V1(T)`), so the fieldless None
    // arm is constructed with empty parentheses.
    Ok(format!(
        "match {inner} {{ Some(__v) => {name}::V1(__v), None => {name}::V0() }}"
    ))
}
