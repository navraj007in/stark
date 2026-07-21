//! §6.4: every MIR instance gets one deterministic, injective-within-one-linked-program Rust
//! identifier. MIR lowering (WP-C4.5c's `key_symbol`) already produces a deterministic,
//! injective canonical symbol per instance (e.g. `main@[]`, `foo@[Int32]`) -- this module does
//! not re-derive that identity, it only sanitizes it into valid Rust syntax, since `@`, `[`,
//! `]`, `,`, and spaces are not legal in a Rust identifier.

/// The entry instance's canonical MIR symbol. `mir::interp::run_program` already looks up this
/// exact string (`starkc/src/mir/interp.rs`); the backend uses the same convention rather than
/// inventing a second one, per §5.2 ("no backend semantic reconstruction").
pub const ENTRY_SYMBOL: &str = "main@[]";

/// Sanitizes a canonical MIR symbol into a valid Rust identifier. Injectivity is preserved by
/// hex-encoding every byte outside `[A-Za-z0-9_]` individually (never collapsing runs), so two
/// distinct disallowed characters can never sanitize to the same output.
pub fn sanitize_symbol(symbol: &str) -> String {
    let mut out = String::with_capacity(symbol.len() + 8);
    out.push_str("stark_");
    for b in symbol.bytes() {
        let c = b as char;
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push_str(&format!("_{b:02x}"));
        }
    }
    out
}

/// WP-C5.2d: the entry instance is always emitted as Rust's literal `fn main()` (required by
/// Rust, not a choice), never its sanitized symbol -- used both when *defining* the entry
/// function (`emit_program.rs`) and when *calling* it (`emit_bodies::emit_call`'s `Callee::
/// Instance` case), so a direct call that happened to target the entry symbol (not exercised by
/// any STARK source today, but not structurally impossible) would still resolve to the same
/// function it was defined as, rather than a second, differently-named, never-defined one.
pub fn function_name_for_symbol(symbol: &str) -> String {
    if symbol == ENTRY_SYMBOL {
        "main".to_string()
    } else {
        sanitize_symbol(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_symbols_sanitize_distinctly() {
        let a = sanitize_symbol("foo@[Int32]");
        let b = sanitize_symbol("foo@[Int64]");
        assert_ne!(a, b);
    }

    #[test]
    fn entry_symbol_names_to_main() {
        assert_eq!(function_name_for_symbol(ENTRY_SYMBOL), "main");
    }

    #[test]
    fn non_entry_symbol_names_to_its_sanitized_form() {
        assert_eq!(
            function_name_for_symbol("foo@[]"),
            sanitize_symbol("foo@[]")
        );
    }

    #[test]
    fn sanitized_symbol_is_a_valid_rust_identifier_shape() {
        let s = sanitize_symbol("main@[]");
        assert!(s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
        assert!(s.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'));
    }
}
