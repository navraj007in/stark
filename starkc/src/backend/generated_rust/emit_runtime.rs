//! `RuntimeFn`/`Callee::Runtime` call emission — bridges MIR runtime calls to `stark_runtime`
//! functions. Each arm renders the runtime call as a Rust expression from the already-emitted
//! argument expressions; the assignment/slot wrapping around it is `emit_bodies`' concern.
//!
//! WP-C6.3a activated the String/str and str-output surface. The remaining `RuntimeFn` groups
//! (Vec, Box, slices, iterators, HashMap, formatting of non-string values) land with their
//! sub-packages and stay `Unsupported` until then.

use super::{emit_types, BackendDiagnostic};
use crate::mir::{MirTy, RuntimeFn};

/// WP-C6.3b (DEV-107): the user source location of the call being emitted, resolved at COMPILE time
/// from the terminator's own `SourceInfo` (every terminator carries one, `Call` included).
///
/// A trapping runtime op — `v[i]`, `remove`, slicing — must report the location of the STARK
/// expression that trapped. Before this, those ops aborted with a location internal to the runtime
/// (`"<vec index>":0:0`), which was correct in category and exit code but useless for provenance and
/// inconsistent with `Terminator::Checked`'s array/arithmetic traps. Non-trapping ops ignore it.
pub struct CallSite {
    pub file: String,
    pub line: u32,
    pub col: u32,
}

impl CallSite {
    /// The `file, line, column` argument triple appended to a trapping runtime call.
    fn trap_args(&self) -> String {
        format!("{:?}, {}, {}", self.file, self.line, self.col)
    }
}

/// Render `rt(args...)` as a Rust expression. `args` are the argument operands already emitted by
/// `emit_operand`; a String/str receiver arrives as `&String`/`&str` (deref-coercing to the `&str`
/// the `stark_runtime::string` helpers take) and a `&mut self` receiver as `&mut String`.
/// `dest_ty` is the type of the place being assigned — an `Option`-returning runtime fn wraps its
/// Rust `Option` into the program's generated Option enum, whose name `dest_ty` supplies.
pub fn emit_runtime_call(
    rt: RuntimeFn,
    args: &[String],
    dest_ty: &MirTy,
    site: &CallSite,
    // WP-C6.3d: how a map op decides key identity (`None` for every non-map op).
    key_eq: Option<&str>,
) -> Result<String, BackendDiagnostic> {
    use RuntimeFn::*;
    let eq = || {
        key_eq.ok_or_else(|| {
            BackendDiagnostic::Unsupported(
                "a HashMap operation reached the backend without a resolved key `Eq` (CD-133)"
                    .to_string(),
            )
        })
    };
    // A small helper: the argument at `i`, wrapped so method/`.as_bytes()` suffixes bind correctly.
    let arg = |i: usize| format!("({})", args[i]);
    Ok(match rt {
        // --- str output (06 PRINT-DISPLAY-001): bytes submitted through the runtime sink. ---
        PrintlnStr => format!("stark_runtime::output::stdout_line({}.as_bytes())", arg(0)),
        PrintStr => format!("stark_runtime::output::stdout_bytes({}.as_bytes())", arg(0)),

        // --- Primitive output (WP-C6.3e). The value is already widened to i64/u64/f64 by lowering
        // (`widen_for_print`); the `as` casts make that width-safe regardless and are no-ops when it
        // already holds. Rendering matches the HIR oracle via `stark_runtime::format`. ---
        PrintlnInt64 => format!("stark_runtime::format::println_i64({} as i64)", arg(0)),
        PrintInt64 => format!("stark_runtime::format::print_i64({} as i64)", arg(0)),
        PrintlnUInt64 => format!("stark_runtime::format::println_u64({} as u64)", arg(0)),
        PrintUInt64 => format!("stark_runtime::format::print_u64({} as u64)", arg(0)),
        PrintlnBool => format!("stark_runtime::format::println_bool({})", arg(0)),
        PrintBool => format!("stark_runtime::format::print_bool({})", arg(0)),
        PrintlnFloat64 => format!("stark_runtime::format::println_f64({} as f64)", arg(0)),
        // 0.1-A9 (DEV-105): `as f32` is a no-op here — the verifier already requires a `Float32`
        // operand — but it pins the width at the call site, so a future widening upstream is a
        // compile error in the generated crate rather than silently wrong digits.
        PrintlnFloat32 => format!("stark_runtime::format::println_f32({} as f32)", arg(0)),
        PrintFloat32 => format!("stark_runtime::format::print_f32({} as f32)", arg(0)),
        PrintFloat64 => format!("stark_runtime::format::print_f64({} as f64)", arg(0)),

        // --- String construction / conversion ---
        StringNew => "stark_runtime::string::new()".to_string(),
        StringFromStr => format!("stark_runtime::string::from_str({})", arg(0)),
        StrToString => format!("stark_runtime::string::to_string({})", arg(0)),
        StringClone => format!("stark_runtime::string::clone_string({})", arg(0)),
        StringAsStr => format!("stark_runtime::string::as_str({})", arg(0)),

        // --- String / str queries ---
        StringLen | StrLen => format!("stark_runtime::string::len({})", arg(0)),
        StringIsEmpty | StrIsEmpty => format!("stark_runtime::string::is_empty({})", arg(0)),
        StrBytes => format!("{}.as_bytes()", arg(0)),
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

        // --- Vec value surface (WP-C6.3b). Owning, slot-backed; receivers arrive as
        // `&Vec`/`&mut Vec`. Trapping index/replace/remove and interior-ref get/iter/slice are
        // later slices. ---
        VecNew => "stark_runtime::vec::new()".to_string(),
        VecWithCapacity => format!("stark_runtime::vec::with_capacity({})", arg(0)),
        VecPush => format!("stark_runtime::vec::push({}, {})", arg(0), arg(1)),
        VecPop => wrap_option(&format!("stark_runtime::vec::pop({})", arg(0)), dest_ty)?,
        VecLen => format!("stark_runtime::vec::len({})", arg(0)),
        VecIsEmpty => format!("stark_runtime::vec::is_empty({})", arg(0)),
        // `v[i]` by Copy (V-COPY-1); traps IndexOutOfBounds at the USER's location (DEV-107 closed).
        VecIndexGet => format!(
            "stark_runtime::vec::index_get({}, {}, {})",
            arg(0),
            arg(1),
            site.trap_args()
        ),
        VecClear => format!("stark_runtime::vec::clear({})", arg(0)),
        // Trapping removal, and the CHECKED interior accessors that never trap (0.1-A4): `get`
        // yields `Option<&T>`, `get_mut` `Option<&mut T>`, both wrapped into the generated Option.
        VecRemove => format!(
            "stark_runtime::vec::remove({}, {}, {})",
            arg(0),
            arg(1),
            site.trap_args()
        ),
        VecGetRef => wrap_option(
            &format!("stark_runtime::vec::get_ref({}, {})", arg(0), arg(1)),
            dest_ty,
        )?,
        VecGetMutRef => wrap_option(
            &format!("stark_runtime::vec::get_mut_ref({}, {})", arg(0), arg(1)),
            dest_ty,
        )?,

        // --- Slice views (WP-C6.3b, 0.1-A6/A8). `SliceNew(&base, lo, hi, inclusive)` traps
        // IndexOutOfBounds on an inverted or out-of-range window (a trap, never a clamp); the `Mut`
        // form yields `&mut [T]` so writes reach the base (REF-SLICE-001). `&[T; N]`/`&Vec<T>` both
        // coerce to `&[T]`, so no per-base variants are needed. ---
        SliceNew => format!(
            "stark_runtime::vec::slice_new({}, {} as i64, {} as i64, {}, {})",
            arg(0),
            arg(1),
            arg(2),
            arg(3),
            site.trap_args()
        ),
        SliceNewMut => format!(
            "stark_runtime::vec::slice_new_mut({}, {} as i64, {} as i64, {}, {})",
            arg(0),
            arg(1),
            arg(2),
            arg(3),
            site.trap_args()
        ),
        SliceLen => format!("stark_runtime::vec::slice_len({})", arg(0)),
        SliceIsEmpty => format!("stark_runtime::vec::slice_is_empty({})", arg(0)),

        // --- Box (WP-C6.3b): construction and consuming extraction. No `Deref` in Core v1. ---
        BoxNew => format!("stark_runtime::boxed::new({})", arg(0)),
        BoxIntoInner => format!("stark_runtime::boxed::into_inner({})", arg(0)),

        // --- Iterator cursors (WP-C6.3c). `New` borrows the source; `Next` takes `&mut cursor` and
        // returns an `Option` wrapped into the program's generated Option enum. `VecIterNext` lends
        // `&T` out of the SOURCE (so the loop variable outlives the call); `CharsIterNext` yields a
        // `Char` by value. ---
        VecIterNew => format!("stark_runtime::vec::iter_new({})", arg(0)),
        VecIterNext => wrap_option(
            &format!("stark_runtime::vec::iter_next({})", arg(0)),
            dest_ty,
        )?,
        CharsIterNew => format!("stark_runtime::string::chars_new({})", arg(0)),
        CharsIterNext => wrap_option(
            &format!("stark_runtime::string::chars_next({})", arg(0)),
            dest_ty,
        )?,

        // --- HashMap (WP-C6.3d). CE4: an insertion-ordered vector; identity by the key type's
        // lawful `Eq`, passed in as a comparator so the map never decides it and `Hash` is never
        // consulted (CD-132). `keys()` is a borrowed cursor over the keys in insertion order. ---
        HashMapNew => "stark_runtime::map::new()".to_string(),
        HashMapInsert => wrap_option(
            &format!(
                "stark_runtime::map::insert({}, {}, {}, {})",
                arg(0),
                arg(1),
                arg(2),
                eq()?
            ),
            dest_ty,
        )?,
        HashMapGet => wrap_option(
            &format!("stark_runtime::map::get({}, {}, {})", arg(0), arg(1), eq()?),
            dest_ty,
        )?,
        HashMapContainsKey => format!(
            "stark_runtime::map::contains_key({}, {}, {})",
            arg(0),
            arg(1),
            eq()?
        ),
        HashMapLen => format!("stark_runtime::map::len({})", arg(0)),
        HashMapIsEmpty => format!("stark_runtime::map::is_empty({})", arg(0)),
        // DEV-116: HashSet. The runtime models it as `StarkMap<T, ()>`, so `eq()` is the SAME
        // comparator the map uses — the user's selected `Eq::eq` for a nominal element, and
        // `structural_eq` only where that IS the lawful `Eq` (primitives, `String`). No structural
        // host fallback substitutes for a user implementation.
        HashSetNew => "stark_runtime::map::new()".to_string(),
        HashSetInsert => format!(
            "stark_runtime::map::set_insert({}, {}, {})",
            arg(0),
            arg(1),
            eq()?
        ),
        HashSetRemove => format!(
            "stark_runtime::map::set_remove({}, {}, {})",
            arg(0),
            arg(1),
            eq()?
        ),
        HashSetContains => format!(
            "stark_runtime::map::contains_key({}, {}, {})",
            arg(0),
            arg(1),
            eq()?
        ),
        HashSetLen => format!("stark_runtime::map::len({})", arg(0)),
        HashSetIsEmpty => format!("stark_runtime::map::is_empty({})", arg(0)),
        HashSetClear => format!("stark_runtime::map::clear({})", arg(0)),
        HashMapKeysIterNew => format!("stark_runtime::map::keys_iter_new({})", arg(0)),
        HashMapKeysIterNext => wrap_option(
            &format!("stark_runtime::map::keys_iter_next({})", arg(0)),
            dest_ty,
        )?,

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
