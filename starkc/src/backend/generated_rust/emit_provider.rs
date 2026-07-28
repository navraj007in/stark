//! WP-C7.8.2d-2 — static `extern "C"` provider declarations and call emission for the
//! non-resource parameter forms.
//!
//! This is Packet 1's option B in generated code: the provider is an ordinary Rust crate linked
//! into the produced binary, reached by a direct symbol reference. There is no dynamic loading and
//! no symbol table indirection — the declared name *is* the linkage name.
//!
//! **What the backend must not do** (A10 §6, restated where it is enforced): rename or sanitise the
//! symbol; catch panics; reinterpret unknown status codes as recoverable; read output slots before
//! a successful status; create Rust-owned destruction semantics for ABI resources; or bypass the
//! approved boundary helpers.
//!
//! **C7.8.2d-3 adds output-slot discipline and status dispatch**, closing invariants 8 and 9 for
//! non-resource calls. Output slots are `MaybeUninit` and are read only on success; the status is
//! dispatched into ABI §12's three channels, with an undeclared code aborting through the
//! contract-violation channel rather than becoming any STARK value.

use super::emit_bodies::{emit_assignment, emit_operand};
use super::emit_places::TyEnv;
use super::BackendDiagnostic;
use crate::mir::{MirProgram, Operand, Place, ValidatedProviderCall};
use crate::provider_abi::{AbiParam, ScalarTy};
use std::collections::BTreeMap;

/// The runtime path every ABI boundary type is named through. Written once so a rename cannot
/// leave half the emitter pointing at a stale path.
const ABI: &str = "stark_runtime::provider_abi";

fn scalar_rust_ty(t: ScalarTy) -> &'static str {
    match t {
        ScalarTy::U8 => "u8",
        ScalarTy::U16 => "u16",
        ScalarTy::U32 => "u32",
        ScalarTy::U64 => "u64",
        ScalarTy::I8 => "i8",
        ScalarTy::I16 => "i16",
        ScalarTy::I32 => "i32",
        ScalarTy::I64 => "i64",
        ScalarTy::Bool => "bool",
        ScalarTy::F32 => "f32",
        ScalarTy::F64 => "f64",
    }
}

/// One `AbiParam`'s C parameter type, following ABI §6.1's table exactly.
///
/// `ScalarOut` and `ScalarInOut` are both `*mut T` — §6.1 says so explicitly and records that the
/// difference between them is an *initialisation contract* (§11.1), not a type. The C signature
/// cannot carry that difference; the declaration does, and C7.8.2d-3 enforces it.
fn extern_param_ty(param: &AbiParam) -> Result<String, BackendDiagnostic> {
    Ok(match param {
        AbiParam::ScalarIn(t) => scalar_rust_ty(*t).to_string(),
        AbiParam::ScalarOut(t) | AbiParam::ScalarInOut(t) => {
            format!("*mut {}", scalar_rust_ty(*t))
        }
        AbiParam::BufferIn => format!("{ABI}::BorrowedBuffer"),
        AbiParam::BufferInOut => format!("{ABI}::BorrowedBufferMut"),
        AbiParam::HandleBorrowed { resource_type }
        | AbiParam::HandleConsumed { resource_type }
        | AbiParam::HandleOut { resource_type } => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "provider resource type `{resource_type}` is not admitted by this build \
                 (WP-C7.8.2d-4 / C7.8.4)"
            )));
        }
    })
}

/// Emits one `extern "C"` block declaring every provider function the program calls.
///
/// Deduplicated by symbol and emitted in sorted order, because two call sites of the same function
/// must produce one declaration and the generated source must not depend on body iteration order —
/// the same determinism requirement Gate C7.2's reproducibility classification rests on.
///
/// Symbols are emitted **verbatim**. They never pass through `mangle::sanitize_symbol`: that
/// encodes MIR canonical symbols into legal Rust identifiers, and a provider symbol is not a MIR
/// instance. MIR verification already proved each one is a legal C identifier (V-PROV-4), so no
/// repair is needed here — and a repair would make the linkage name differ from the metadata name.
pub fn emit_extern_declarations(program: &MirProgram) -> Result<String, BackendDiagnostic> {
    if program.provider_calls.is_empty() {
        return Ok(String::new());
    }

    let mut by_symbol: BTreeMap<&str, &ValidatedProviderCall> = BTreeMap::new();
    for call in &program.provider_calls {
        by_symbol.insert(call.symbol(), call);
    }

    let mut out = String::new();
    out.push_str(
        "// Native Provider ABI v0.1 (A10, CD-200): statically linked first-party providers.\n\
         // Symbols are the declared names VERBATIM -- never mangled, never sanitised.\n",
    );
    out.push_str("extern \"C\" {\n");
    for (symbol, call) in &by_symbol {
        let mut params = Vec::with_capacity(call.function.params.len());
        for (i, p) in call.function.params.iter().enumerate() {
            params.push(format!("a{i}: {}", extern_param_ty(p)?));
        }
        out.push_str(&format!(
            "    // provider `{}` capability `{}`\n    fn {symbol}({}) -> {ABI}::ProviderStatus;\n",
            call.provider.name,
            call.capability,
            params.join(", ")
        ));
    }
    out.push_str("}\n\n");
    Ok(out)
}

/// Emits the statement sequence for one provider call.
///
/// **Invariant 6 is enforced by construction here.** Every borrowed argument is bound to a *named*
/// local before the call, and the pointer is taken from that named local. The shape a naive
/// emitter would produce —
///
/// ```text
/// stark_provider_fn(make_bytes(value).as_ptr(), ...)
/// ```
///
/// — creates a pointer into a temporary that may die before or during the call expression. Naming
/// the binding makes its storage live for the whole statement sequence, which is exactly the
/// "materialised from a named live place, lifetime covering the complete call" rule.
///
/// The MIR operand for each argument is already a place-derived reference (`provider_sig` maps
/// `ScalarOut(T)` to `&mut T`, `BufferIn` to `&[UInt8]`, and so on), so the named binding preserves
/// a property MIR established rather than inventing one.
/// Emits the statement sequence for one provider call.
///
/// **Invariant 6 (borrow validity)** is enforced by construction: every argument is bound to a
/// *named* local before the call, and pointers are taken from that binding. The shape this rules
/// out —
///
/// ```text
/// stark_provider_fn(make_bytes(value).as_ptr(), ...)
/// ```
///
/// — creates a pointer into a temporary that may die before or during the call expression, which
/// compiles, links, and is undefined behaviour at runtime.
///
/// **Invariant 8 (output-slot discipline)**: a `ScalarOut` slot is emitter-owned `MaybeUninit`
/// storage, and its value reaches the MIR-visible location **only** on the success arm. That is
/// stronger than a rule saying "do not read on failure": the MIR local is never written at all
/// unless the provider reported success, so there is no failure path on which a read could occur.
/// `ScalarInOut` and `BufferInOut` deliberately keep d-2's direct pointer — ABI §11.1 makes them
/// caller-initialised and caller-owned, so `MaybeUninit` semantics would be wrong for them.
///
/// **Invariant 9 (channel discipline)**: the status is dispatched into ABI §12's three channels,
/// which stay structurally distinct. There is no `_ => SomeError::Other` arm; the wildcard is the
/// contract-violation channel, so an undeclared status **never becomes a STARK value at all** —
/// it aborts before the destination is written.
pub fn emit_provider_call(
    call: &ValidatedProviderCall,
    args: &[Operand],
    dest: &Place,
    env: &TyEnv,
    indent: &str,
) -> Result<String, BackendDiagnostic> {
    if args.len() != call.function.params.len() {
        return Err(BackendDiagnostic::Unsupported(format!(
            "provider call `{}`: {} MIR argument(s) for {} declared parameter(s)",
            call.symbol(),
            args.len(),
            call.function.params.len()
        )));
    }

    let mut out = String::new();
    let mut call_args = Vec::with_capacity(args.len());
    // (named `&mut T` binding, MaybeUninit slot, Rust scalar type) per ScalarOut, for the
    // success-arm copy-back.
    let mut out_slots: Vec<(String, String, &'static str)> = Vec::new();

    for (i, (param, arg)) in call.function.params.iter().zip(args).enumerate() {
        let value = emit_operand(arg, env)?;
        let binding = format!("__prov_a{i}");
        out.push_str(&format!("{indent}let {binding} = {value};\n"));

        match param {
            AbiParam::ScalarIn(_) => call_args.push(binding),
            AbiParam::ScalarOut(t) => {
                // Emitter-owned uninitialised storage. The provider writes HERE, not into the MIR
                // local, so a failure status leaves the MIR local untouched by construction.
                let slot = format!("__prov_o{i}");
                let ty = scalar_rust_ty(*t);
                out.push_str(&format!(
                    "{indent}let mut {slot} = std::mem::MaybeUninit::<{ty}>::uninit();\n"
                ));
                call_args.push(format!("{slot}.as_mut_ptr()"));
                out_slots.push((binding, slot, ty));
            }
            AbiParam::ScalarInOut(t) => {
                // §11.1: caller-initialised and caller-owned across the call. Its validity does
                // not depend on the status, so it is NOT MaybeUninit and needs no copy-back.
                call_args.push(format!("{binding} as *mut {}", scalar_rust_ty(*t)));
            }
            AbiParam::BufferIn => {
                let buf = format!("__prov_b{i}");
                out.push_str(&format!(
                    "{indent}let {buf} = {ABI}::BorrowedBuffer {{ ptr: {binding}.as_ptr(), \
                     len: {binding}.len() }};\n"
                ));
                call_args.push(buf);
            }
            AbiParam::BufferInOut => {
                let buf = format!("__prov_b{i}");
                out.push_str(&format!(
                    "{indent}let mut {buf} = {ABI}::BorrowedBufferMut {{ \
                     ptr: {binding}.as_mut_ptr(), len: {binding}.len() }};\n"
                ));
                call_args.push(buf);
            }
            AbiParam::HandleBorrowed { resource_type }
            | AbiParam::HandleConsumed { resource_type }
            | AbiParam::HandleOut { resource_type } => {
                return Err(BackendDiagnostic::Unsupported(format!(
                    "provider resource type `{resource_type}` is not admitted by this build \
                     (WP-C7.8.2d-4 / C7.8.4)"
                )));
            }
        }
    }

    // The one `unsafe` in a provider call, and it is exactly the FFI call -- no wider block, so a
    // reviewer sees precisely what is unchecked. No `catch_unwind`: the generated workspace builds
    // with `panic = "abort"` in both profiles, so a provider panic aborts rather than unwinding
    // into generated code, and wrapping it here would misclassify a provider defect as recoverable
    // (Packet 1 §1.1).
    out.push_str(&format!(
        "{indent}let __prov_status: u32 = unsafe {{ {}({}) }}.code;\n",
        call.symbol(),
        call_args.join(", ")
    ));

    // ABI §12's three channels, structurally distinct.
    out.push_str(&format!("{indent}match __prov_status {{\n"));

    // Channel 0 -- success. Every declared output is read here and nowhere else, so a STARK value
    // can only be built from outputs the provider reported as valid.
    out.push_str(&format!("{indent}    0u32 => {{\n"));
    if out_slots.is_empty() {
        out.push_str(&format!("{indent}        // no declared output slots\n"));
    }
    for (binding, slot, ty) in &out_slots {
        out.push_str(&format!(
            "{indent}        // §11.1: valid ONLY because the status reported success.\n\
             {indent}        *{binding} = unsafe {{ {slot}.assume_init() }};\n"
        ));
        let _ = ty;
    }
    out.push_str(&format!("{indent}    }}\n"));

    // Channel 1 -- a code the package declared recoverable. Outputs are NOT read.
    for (code, package_error) in call.status_binding.declared_codes() {
        out.push_str(&format!(
            "{indent}    {code}u32 => {{\n\
             {indent}        // declared recoverable: `{package_error}`. Output slots are NOT read.\n\
             {indent}    }}\n"
        ));
    }

    // Channel 2 -- contract violation. The wildcard is deliberately NOT a generic package error:
    // an undeclared status aborts rather than becoming a STARK value, so it never reaches the
    // destination below.
    out.push_str(&format!(
        "{indent}    unknown => {ABI}::contract_violation_unknown_status(\n\
         {indent}        {},\n{indent}        {},\n{indent}        unknown,\n{indent}    ),\n",
        rust_str_lit(&call.provider.name),
        rust_str_lit(call.symbol())
    ));
    out.push_str(&format!("{indent}}}\n"));

    // Only success or a declared code can reach this point; the contract-violation arm diverges.
    out.push_str(&format!(
        "{indent}{}\n",
        emit_assignment(dest, "__prov_status", env)?
    ));
    Ok(out)
}

/// A Rust string literal with the escaping a provider name or symbol could conceivably need.
/// Symbols are already validated C identifiers, but provider *names* are free text.
fn rust_str_lit(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}
