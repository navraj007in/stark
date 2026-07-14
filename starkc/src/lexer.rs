//! Lexer for `01-Lexical-Grammar.md`.
//!
//! WP1.2 (PLAN.md): hand-written; maximal munch; keywords vs reserved words
//! vs identifiers; all literal forms with suffixes and the strict underscore
//! rule; nested block comments; raw strings; `>>` lexed as one `Shr` token
//! that the parser can split into two `>` in generic-argument position.

// Implemented in WP1.2.
