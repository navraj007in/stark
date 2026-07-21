//! §6.4: every MIR instance gets one deterministic, injective-within-one-linked-program Rust
//! identifier. MIR lowering (WP-C4.5c's `key_symbol`) already produces a deterministic,
//! injective canonical symbol per instance (e.g. `main@[]`, `foo@[Int32]`) -- this module does
//! not re-derive that identity, it only sanitizes it into valid Rust syntax, since `@`, `[`,
//! `]`, `,`, and spaces are not legal in a Rust identifier.

/// The entry instance's canonical MIR symbol. `mir::interp::run_program` already looks up this
/// exact string (`starkc/src/mir/interp.rs`); the backend uses the same convention rather than
/// inventing a second one, per §5.2 ("no backend semantic reconstruction").
pub const ENTRY_SYMBOL: &str = "main@[]";

/// Sanitizes a canonical MIR symbol into a valid Rust identifier.
///
/// The encoding is injective, and injectivity is the whole point: two distinct MIR instances that
/// sanitized to one Rust identifier would silently become one function in the generated crate.
/// `_` is the escape introducer, so it must itself be escaped -- an earlier version passed `_`
/// through unchanged, which made encoded output indistinguishable from source text that already
/// looked like an escape, and `pkg::f` (encoding to `pkg_3a_3af`) collided with a legally-named
/// STARK function `pkg_3a_3af`. `key_symbol` (`starkc/src/mir/lower.rs`) puts a `::`-joined
/// module/package path in every symbol, so that collision was reachable from ordinary source.
///
/// Each source byte maps to exactly one of:
/// - `[A-Za-z0-9]`  → itself (never a `_`, so it can never start an escape),
/// - `_`            → `__`,
/// - anything else  → `_hh`, lowercase hex, where `h` is never `_`.
///
/// Decoding is therefore unambiguous -- on reading `_`, the next byte is either `_` (a literal
/// underscore) or the first of exactly two hex digits -- which is precisely injectivity.
pub fn sanitize_symbol(symbol: &str) -> String {
    let mut out = String::with_capacity(symbol.len() + 8);
    out.push_str("stark_");
    for b in symbol.bytes() {
        let c = b as char;
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if c == '_' {
            out.push_str("__");
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

    /// The regression this encoding exists for: a canonical symbol containing `::` must not
    /// collide with a legally-named STARK function whose own name spells the escape.
    #[test]
    fn escape_lookalike_identifier_does_not_collide_with_a_module_path() {
        assert_ne!(
            sanitize_symbol("pkg::f@[]"),
            sanitize_symbol("pkg_3a_3af@[]")
        );
    }

    /// Every character class the encoder treats specially, each paired with source text that
    /// spells its own escape. A single pairwise-distinctness check over the whole set is a
    /// stronger statement than a list of hand-picked `assert_ne!`s, and fails loudly with the
    /// colliding pair named.
    #[test]
    fn adversarial_symbols_are_pairwise_distinct() {
        let symbols = [
            // `::` (0x3a 0x3a) vs. text spelling its escape, at both a package and a module
            // boundary -- `key_symbol` emits `⟨package/module path⟩::name@[args]`.
            "pkg::f@[]",
            "pkg_3a_3af@[]",
            "pkg::mod::f@[]",
            "pkg::mod_3a_3af@[]",
            "pkg_3a_3amod::f@[]",
            // `@` (0x40) and the bracket pair (0x5b/0x5d) that delimit type arguments.
            "f@[]",
            "f_40_5b_5d@[]",
            "f@[Int32]",
            "f_40_5b_49nt32_5d@[]",
            // A literal `_` must survive as distinct from an escape that produces one.
            "f_g@[]",
            "f__g@[]",
            "f_5fg@[]",
            // Type-argument separator (`, ` -- comma plus space) vs. its spelling.
            "f@[Int32, Int64]",
            "f@[Int32_2c_20Int64]",
            // Non-ASCII identifiers encode per UTF-8 byte; distinct code points stay distinct,
            // and neither collides with the ASCII text spelling its bytes.
            "café@[]",
            "cafe@[]",
            "caf_c3_a9@[]",
        ];
        for (i, a) in symbols.iter().enumerate() {
            for b in &symbols[i + 1..] {
                assert_ne!(
                    sanitize_symbol(a),
                    sanitize_symbol(b),
                    "distinct MIR symbols {a:?} and {b:?} sanitized to one Rust identifier"
                );
            }
        }
    }

    /// Injectivity stated directly: the encoding is decodable, so no two inputs can share an
    /// output. Checked by round-tripping rather than by asserting non-collision on samples.
    #[test]
    fn sanitize_round_trips_through_a_decoder() {
        fn decode(encoded: &str) -> String {
            let body = encoded.strip_prefix("stark_").expect("missing prefix");
            let bytes = body.as_bytes();
            let mut out: Vec<u8> = Vec::new();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'_' {
                    if bytes[i + 1] == b'_' {
                        out.push(b'_');
                        i += 2;
                    } else {
                        let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap();
                        out.push(u8::from_str_radix(hex, 16).unwrap());
                        i += 3;
                    }
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            String::from_utf8(out).unwrap()
        }

        for symbol in [
            "main@[]",
            "pkg::f@[]",
            "pkg_3a_3af@[]",
            "f__g@[Int32, Int64]",
            "café@[]",
        ] {
            assert_eq!(decode(&sanitize_symbol(symbol)), symbol);
        }
    }

    #[test]
    fn sanitized_symbol_is_a_valid_rust_identifier_shape() {
        let s = sanitize_symbol("main@[]");
        assert!(s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
        assert!(s.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'));
    }
}
