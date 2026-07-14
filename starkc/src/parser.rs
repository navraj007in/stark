//! Parser for `02-Syntax-Grammar.md`.
//!
//! WP1.4 (PLAN.md): hand-written recursive descent with a Pratt expression
//! core using the normative 16-level precedence table. Notable rules the
//! implementation must honor:
//! - non-associative comparisons and ranges (structural, not semantic);
//! - struct-literal restriction at block-position expression heads;
//! - `>>` re-tokenized as two `>` in generic-argument position;
//! - trailing block expressions (statement-first, expression-before-`}`);
//! - panic-mode recovery at `;` and item keywords.

// Implemented in WP1.4.
