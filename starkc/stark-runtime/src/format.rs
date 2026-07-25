//! WP-C6.3e — STARK canonical value formatting.
//!
//! Shared by the HIR interpreter (the format ORACLE) and the native backend, so native `println`
//! output is identical to the oracle's BY CONSTRUCTION — not Rust's `Debug`/`Display`. `starkc`'s
//! `interp::canonical_float` delegates here, so there is one implementation, not two that can drift.
//!
//! This slice covers the primitives (`Int*`/`UInt*` widened to `i64`/`u64` by lowering, `Bool`,
//! `Float32`/`Float64`). The composite renderers (tuple/struct/enum/Option/Result/Vec, user
//! `Display`) land in later C6.3e slices.

// ---- Primitive output (bytes submitted through the runtime sink) ----

/// `println` of a signed integer (all `Int*` widen to `Int64` at lowering).
pub fn println_i64(v: i64) {
    crate::output::stdout_line(itoa_i64(v).as_bytes());
}
pub fn print_i64(v: i64) {
    crate::output::stdout_bytes(itoa_i64(v).as_bytes());
}

/// `println` of an unsigned integer (all `UInt*` widen to `UInt64`).
pub fn println_u64(v: u64) {
    crate::output::stdout_line(itoa_u64(v).as_bytes());
}
pub fn print_u64(v: u64) {
    crate::output::stdout_bytes(itoa_u64(v).as_bytes());
}

/// `println` of a `Bool` — `true`/`false`, matching the oracle's `{b}`.
pub fn println_bool(b: bool) {
    crate::output::stdout_line(bool_bytes(b));
}
pub fn print_bool(b: bool) {
    crate::output::stdout_bytes(bool_bytes(b));
}

/// `println` of a `Float64` in STARK's canonical form.
pub fn println_f64(v: f64) {
    crate::output::stdout_line(canonical_float(v).as_bytes());
}
pub fn print_f64(v: f64) {
    crate::output::stdout_bytes(canonical_float(v).as_bytes());
}

fn itoa_i64(v: i64) -> String {
    v.to_string()
}
fn itoa_u64(v: u64) -> String {
    v.to_string()
}
fn bool_bytes(b: bool) -> &'static [u8] {
    if b {
        b"true"
    } else {
        b"false"
    }
}

// ---- Canonical float formatting (moved from starkc::interp; single source) ----

/// STARK's canonical `Float64` rendering: `NaN`, `inf`/`-inf`, signed zero, otherwise the shortest
/// round-tripping decimal in fixed or scientific form (see `canonical_float_digits`).
pub fn canonical_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f64::INFINITY {
        return "inf".to_string();
    }
    if value == f64::NEG_INFINITY {
        return "-inf".to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    canonical_float_digits(&value.to_string())
}

/// STARK's canonical `Float32` rendering.
pub fn canonical_float32(value: f32) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f32::INFINITY {
        return "inf".to_string();
    }
    if value == f32::NEG_INFINITY {
        return "-inf".to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    canonical_float_digits(&value.to_string())
}

fn canonical_float_digits(shortest: &str) -> String {
    let (sign, unsigned) = shortest
        .strip_prefix('-')
        .map_or(("", shortest), |rest| ("-", rest));
    let (mantissa, explicit_exponent) = unsigned
        .split_once(['e', 'E'])
        .map_or((unsigned, 0_i32), |(mantissa, exponent)| {
            (mantissa, exponent.parse::<i32>().unwrap())
        });
    let decimal_position = mantissa
        .find('.')
        .map_or(mantissa.len() as i32, |position| position as i32)
        + explicit_exponent;
    let raw_digits: String = mantissa
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let leading_zeroes = raw_digits
        .bytes()
        .take_while(|digit| *digit == b'0')
        .count() as i32;
    let scientific_exponent = decimal_position - leading_zeroes - 1;
    let significant = raw_digits.trim_start_matches('0').trim_end_matches('0');
    let significant = if significant.is_empty() {
        "0"
    } else {
        significant
    };

    if (-4..=15).contains(&scientific_exponent) {
        let point = scientific_exponent + 1;
        let mut rendered = String::from(sign);
        if point <= 0 {
            rendered.push_str("0.");
            rendered.extend(std::iter::repeat_n('0', (-point) as usize));
            rendered.push_str(significant);
        } else if point as usize >= significant.len() {
            rendered.push_str(significant);
            rendered.extend(std::iter::repeat_n('0', point as usize - significant.len()));
            rendered.push_str(".0");
        } else {
            rendered.push_str(&significant[..point as usize]);
            rendered.push('.');
            rendered.push_str(&significant[point as usize..]);
        }
        rendered
    } else {
        let mut rendered = String::from(sign);
        rendered.push_str(&significant[..1]);
        if significant.len() > 1 {
            rendered.push('.');
            rendered.push_str(&significant[1..]);
        }
        rendered.push('e');
        rendered.push_str(&scientific_exponent.to_string());
        rendered
    }
}
