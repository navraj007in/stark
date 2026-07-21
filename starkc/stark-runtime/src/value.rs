//! Move/Drop helpers for non-`Copy` MIR locals (`WP-C5-ENTRY.md` §7.2/§7.8: `place_read`,
//! `place_write`, `value_move`, `value_drop`, routed through this one reviewed module rather
//! than ad hoc `unsafe` at each call site). Empty until WP-C5.2/C5.3 need real non-`Copy`
//! storage -- C5.1b's proof program has no non-`Copy` locals.
