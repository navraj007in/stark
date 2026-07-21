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
    fn sanitized_symbol_is_a_valid_rust_identifier_shape() {
        let s = sanitize_symbol("main@[]");
        assert!(s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
        assert!(s.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'));
    }
}
