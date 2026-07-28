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
//! Status *dispatch* is C7.8.2d-3. This slice emits the call and hands the raw
//! `ProviderStatus.code` to the MIR destination, which is exactly what MIR says the destination
//! holds (`provider_sig::PROVIDER_STATUS_TY`). It deliberately does not interpret the code.

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

    for (i, (param, arg)) in call.function.params.iter().zip(args).enumerate() {
        let value = emit_operand(arg, env)?;
        // The named borrow temporary. One per argument, including copied scalars: uniformity keeps
        // the emitted sequence readable and costs nothing, and it means a later slice cannot
        // accidentally special-case one form into an inline temporary.
        let binding = format!("__prov_a{i}");
        out.push_str(&format!("{indent}let {binding} = {value};\n"));

        match param {
            AbiParam::ScalarIn(_) => call_args.push(binding),
            AbiParam::ScalarOut(t) | AbiParam::ScalarInOut(t) => {
                // `binding` is `&mut T`; the C parameter is `*mut T`.
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
    let status = "__prov_status";
    out.push_str(&format!(
        "{indent}let {status}: u32 = unsafe {{ {}({}) }}.code;\n",
        call.symbol(),
        call_args.join(", ")
    ));

    // C7.8.2d-2 hands the RAW status to the MIR destination, which `provider_sig` types as
    // `UInt32`. Interpreting it -- success/declared-error/contract-violation dispatch -- is
    // C7.8.2d-3's job, and doing it here would put channel policy in the emitter rather than in the
    // binding plan that C7.8.2d-1 built for it.
    out.push_str(&format!(
        "{indent}{}\n",
        emit_assignment(dest, status, env)?
    ));
    Ok(out)
}
