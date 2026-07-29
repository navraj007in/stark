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
use super::emit_places::emit_place_to_borrow;
use super::emit_places::TyEnv;
use super::BackendDiagnostic;
use crate::mir::{MirProgram, Operand, Place, ValidatedProviderCall};
use crate::provider_abi::{AbiParam, ScalarTy};
use crate::provider_bind::{ProviderBindingPlan, ProviderInputPlan, ProviderOutputPlan};
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
        // §6.1: borrowed and consumed handles cross by value as `RawResourceHandle`; an out
        // handle is a pointer to one. The resource TYPE does not appear in the C signature at all
        // -- it is carried in the handle's own `resource_type` field and validated on return.
        AbiParam::HandleBorrowed { .. } | AbiParam::HandleConsumed { .. } => {
            format!("{ABI}::RawResourceHandle")
        }
        AbiParam::HandleOut { .. } => format!("*mut {ABI}::RawResourceHandle"),
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

    // `extern crate` for every provider crate, and it is load-bearing rather than stylistic.
    //
    // An `extern "C"` block creates no Rust-level dependency edge, so rustc has no reason to link
    // a path dependency whose only exports are `#[no_mangle]` symbols -- it drops the rlib and the
    // link fails with an undefined symbol. Naming the crate is what pulls it in.
    //
    // Crate names are Cargo package names, so `-` becomes `_` exactly as Cargo does it. The alias
    // keeps the binding unused-warning-free without an attribute.
    let mut crates: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for call in &program.provider_calls {
        crates.insert(call.provider_crate.as_str());
    }
    for name in &crates {
        let ident = name.replace('-', "_");
        out.push_str(&format!("extern crate {ident} as _{ident};\n"));
    }
    if !crates.is_empty() {
        out.push('\n');
    }
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

/// Emits the statement sequence for one provider call, driven by its **binding plan**.
///
/// Emission walks the plan rather than re-reading `AbiParam`s. That is what C7.8.2d-1 built the
/// plan for: parameter classification, resource-type resolution and the status vocabulary are
/// decided once, in one place, and the emitter cannot quietly disagree with the verifier about
/// which parameters are outputs.
///
/// **Invariant 6 (borrow validity)**: every argument is bound to a *named* local before the call,
/// and pointers are taken from that binding — never from a temporary that could die during the
/// call expression.
///
/// **Invariant 7 (consumed-resource invalidation)**: a `HandleConsumed` argument is taken from its
/// MIR place and converted with `into_raw`, which consumes the owning handle **before** the call.
/// There is no restore-on-failure path, because ABI §8 has none: ownership returning on failure
/// would make a handle's liveness a runtime property, and exactly-once close would stop being
/// statically verifiable.
///
/// **Invariant 8 (output-slot discipline)**: `ScalarOut` and `HandleOut` are emitter-owned
/// `MaybeUninit` storage, read only on the success arm. `ScalarInOut`/`BufferInOut` keep the
/// argument binding — §11.1 makes them caller-owned, so `MaybeUninit` would be wrong for them.
///
/// **Invariant 9 (channel discipline)**: the wildcard arm is the contract-violation channel, so an
/// undeclared status never becomes a STARK value.
pub fn emit_provider_call(
    call: &ValidatedProviderCall,
    plan: &ProviderBindingPlan,
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
    if !plan.covers(call.function.params.len()) {
        return Err(BackendDiagnostic::Unsupported(format!(
            "provider call `{}`: binding plan does not cover every declared parameter",
            call.symbol()
        )));
    }

    let provider_lit = rust_str_lit(&call.provider.name);
    let symbol_lit = rust_str_lit(call.symbol());

    let mut out = String::new();
    // Argument expressions by declared index, so inputs and outputs can be emitted in plan order
    // while the call still receives them in DECLARATION order.
    let mut call_args: Vec<Option<String>> = vec![None; args.len()];
    // Success-arm writebacks, in declared order.
    let mut writebacks: Vec<String> = Vec::new();

    // Named bindings for every argument, first, so every borrow outlives the call statement.
    //
    // A `&mut`-shaped argument is REBORROWED rather than read out of its place. `&mut T` is not
    // `Copy`, so emitting the ordinary operand form moves out of the local and rustc rejects it
    // (E0507) -- the defect a real build found, and one no text-level assertion could have. `&mut
    // *place` produces the same pointer without disturbing the local, which is also what the
    // caller-owned contract requires: §11.1's in/out storage stays the caller's across the call.
    let mut_shaped: std::collections::BTreeSet<usize> = call
        .function
        .params
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            matches!(
                p,
                AbiParam::ScalarOut(_) | AbiParam::ScalarInOut(_) | AbiParam::BufferInOut
            )
        })
        .map(|(i, _)| i)
        .collect();

    let handle_out: std::collections::BTreeSet<usize> = call
        .function
        .params
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p, AbiParam::HandleOut { .. }))
        .map(|(i, _)| i)
        .collect();

    for (i, arg) in args.iter().enumerate() {
        // A HandleOut argument names the destination; binding it would read a slot that is still
        // dead, which is the whole reason it is a place rather than a reference.
        if handle_out.contains(&i) {
            continue;
        }
        let value = match arg {
            Operand::Copy(place) | Operand::Move(place) if mut_shaped.contains(&i) => {
                format!("&mut *{}", emit_place_to_borrow(place, env)?)
            }
            other => emit_operand(other, env)?,
        };
        out.push_str(&format!("{indent}let __prov_a{i} = {value};\n"));
    }

    for input in &plan.inputs {
        let i = input.index();
        let binding = format!("__prov_a{i}");
        match input {
            ProviderInputPlan::Scalar { .. } => call_args[i] = Some(binding),
            ProviderInputPlan::ScalarInOut { ty, .. } => {
                call_args[i] = Some(format!("{binding} as *mut {}", mir_scalar_rust_ty(ty)?));
            }
            ProviderInputPlan::BufferIn { .. } => {
                out.push_str(&format!(
                    "{indent}let __prov_b{i} = {ABI}::BorrowedBuffer {{ ptr: {binding}.as_ptr(), \
                     len: {binding}.len() }};\n"
                ));
                call_args[i] = Some(format!("__prov_b{i}"));
            }
            ProviderInputPlan::BufferInOut { .. } => {
                out.push_str(&format!(
                    "{indent}let mut __prov_b{i} = {ABI}::BorrowedBufferMut {{ \
                     ptr: {binding}.as_mut_ptr(), len: {binding}.len() }};\n"
                ));
                call_args[i] = Some(format!("__prov_b{i}"));
            }
            // §8: the caller retains ownership, so the raw form is BORROWED for the call only.
            ProviderInputPlan::HandleBorrowed { .. } => {
                call_args[i] = Some(format!("{binding}.as_raw()"));
            }
            // §8: ownership transfers AT CALL ENTRY. `into_raw` consumes the owning handle here,
            // before the call, so the source is dead on every path out -- success, declared error
            // and contract violation alike. Nothing below restores it, and nothing can: the value
            // was moved.
            ProviderInputPlan::HandleConsumed { .. } => {
                out.push_str(&format!(
                    "{indent}// §8: ownership transfers at call entry, regardless of status.\n\
                     {indent}let __prov_c{i} = {binding}.into_raw();\n"
                ));
                call_args[i] = Some(format!("__prov_c{i}"));
            }
        }
    }

    for output in &plan.outputs {
        let i = output.index();
        match output {
            ProviderOutputPlan::Scalar { ty, .. } => {
                let rust_ty = mir_scalar_rust_ty(ty)?;
                out.push_str(&format!(
                    "{indent}let mut __prov_o{i} = std::mem::MaybeUninit::<{rust_ty}>::uninit();\n"
                ));
                call_args[i] = Some(format!("__prov_o{i}.as_mut_ptr()"));
                writebacks.push(format!(
                    "{indent}        // §11.1: valid ONLY because the status reported success.\n\
                     {indent}        *__prov_a{i} = unsafe {{ __prov_o{i}.assume_init() }};\n"
                ));
            }
            ProviderOutputPlan::Handle { type_id, .. } => {
                // The argument is the destination PLACE (see `provider_sig`), so the handle is
                // written with an ordinary assignment -- the write that makes a slot live. Anything
                // else would have to borrow a slot nothing has written yet.
                let Some(Operand::Move(dest_place) | Operand::Copy(dest_place)) = args.get(i)
                else {
                    return Err(BackendDiagnostic::Unsupported(format!(
                        "provider call `{}`: a HandleOut argument must be a place",
                        call.symbol()
                    )));
                };
                out.push_str(&format!(
                    "{indent}let mut __prov_o{i} = \
                     std::mem::MaybeUninit::<{ABI}::RawResourceHandle>::uninit();\n"
                ));
                call_args[i] = Some(format!("__prov_o{i}.as_mut_ptr()"));
                // §11.1: validate the resource type BEFORE constructing the owning value. The
                // check lives in `from_raw_checked` so it cannot be skipped, and a mismatch is a
                // contract violation rather than a recoverable error -- wrapping a mistyped handle
                // would hand generated code an owning value for a resource of unknown kind.
                let checked = format!(
                    "match {ABI}::OwnedResourceHandle::from_raw_checked(__prov_raw{i}, \
                     {type_id}u32) {{ Ok(handle) => handle, Err(mismatch) => \
                     {ABI}::contract_violation_resource_type({provider_lit}, {symbol_lit}, \
                     mismatch.expected, mismatch.found) }}"
                );
                writebacks.push(format!(
                    "{indent}        let __prov_raw{i} = unsafe {{ __prov_o{i}.assume_init() }};\n\
                     {indent}        {}\n",
                    emit_assignment(dest_place, &checked, env)?
                ));
            }
        }
    }

    let mut ordered = Vec::with_capacity(call_args.len());
    for (i, a) in call_args.into_iter().enumerate() {
        ordered.push(a.ok_or_else(|| {
            BackendDiagnostic::Unsupported(format!(
                "provider call `{}`: parameter {i} produced no argument",
                call.symbol()
            ))
        })?);
    }

    // The one `unsafe` in a provider call, and it is exactly the FFI call -- no wider block, so a
    // reviewer sees precisely what is unchecked. No `catch_unwind`: the generated workspace builds
    // with `panic = "abort"` in both profiles, so a provider panic aborts rather than unwinding
    // into generated code, and wrapping it here would misclassify a provider defect as recoverable
    // (Packet 1 §1.1).
    out.push_str(&format!(
        "{indent}let __prov_status: u32 = unsafe {{ {}({}) }}.code;\n",
        call.symbol(),
        ordered.join(", ")
    ));

    // ABI §12's three channels, structurally distinct.
    out.push_str(&format!("{indent}match __prov_status {{\n"));
    out.push_str(&format!("{indent}    0u32 => {{\n"));
    if writebacks.is_empty() {
        out.push_str(&format!("{indent}        // no declared output slots\n"));
    }
    for w in &writebacks {
        out.push_str(w);
    }
    out.push_str(&format!("{indent}    }}\n"));

    for (code, package_error) in call.status_binding.declared_codes() {
        out.push_str(&format!(
            "{indent}    {code}u32 => {{\n\
             {indent}        // declared recoverable: `{package_error}`. Output slots are NOT read.\n\
             {indent}    }}\n"
        ));
    }

    out.push_str(&format!(
        "{indent}    unknown => {ABI}::contract_violation_unknown_status(\n\
         {indent}        {provider_lit},\n{indent}        {symbol_lit},\n\
         {indent}        unknown,\n{indent}    ),\n"
    ));
    out.push_str(&format!("{indent}}}\n"));

    out.push_str(&format!(
        "{indent}{}\n",
        emit_assignment(dest, "__prov_status", env)?
    ));
    Ok(out)
}

/// The Rust scalar name for a plan's `MirTy`. The plan resolved these from `ScalarTy`, so anything
/// else here is a compiler defect rather than a rejectable program.
fn mir_scalar_rust_ty(ty: &crate::mir::MirTy) -> Result<&'static str, BackendDiagnostic> {
    use crate::mir::MirTy as T;
    Ok(match ty {
        T::UInt8 => "u8",
        T::UInt16 => "u16",
        T::UInt32 => "u32",
        T::UInt64 => "u64",
        T::Int8 => "i8",
        T::Int16 => "i16",
        T::Int32 => "i32",
        T::Int64 => "i64",
        T::Bool => "bool",
        T::Float32 => "f32",
        T::Float64 => "f64",
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "provider scalar parameter has non-scalar MIR type {other:?}"
            )))
        }
    })
}

/// A Rust string literal with the escaping a provider name could need. Symbols are already
/// validated C identifiers; provider *names* are free text.
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
