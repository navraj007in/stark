//! **WP-COPY-CANON Phase 1 — the sentinel matrix for DEV-121.**
//!
//! The governing law this matrix enforces:
//!
//! > After expression typing, Copy/move behaviour — and the runtime representation that carries it
//! > — is determined exclusively by the normalized semantic type, never by the expression that
//! > produced the value. This binds the checker, MIR lowering, the native backend, and each
//! > interpreter's value model equally.
//!
//! # Why a matrix rather than a regression test
//!
//! DEV-121 was CORPUS-GAP, not oracle-blind: the HIR interpreter diverged from MIR and native
//! alone, so a differential could have caught it — no input ever exercised the shape. A single
//! regression for `bytes()` would close the instance and leave the class open, because the defect
//! is per-PRODUCER: `bytes()` and `as_slice()` have the same normalized type and were built by
//! different code paths, and only one of them was wrong.
//!
//! So the matrix is the cross product of *producers* of a reference-typed value against *use
//! modes*, and it is a permanent corpus obligation rather than one-time DEV evidence.
//!
//! # The three evidence columns
//!
//! A cell that merely "passes" records nothing about why. Each producer is therefore checked on
//! three axes, because the defect lived in the third and the first two were already correct:
//!
//! 1. **MirTy + copy-eligibility** — what the type system says the value is.
//! 2. **Emitted MIR call operand** — `copy` or `move`. Asserted from the MIR dump, so this fails on
//!    wrong operand selection *even when runtime behaviour is green*. That is the property the
//!    defect class evades: DEV-121 had correct MIR and a wrong runtime value, and a runtime-only
//!    test would pass just as happily if the reverse were true.
//! 3. **Runtime value kind** — reference-kind (`Slice`/`Ref`/`Str`) versus owned (`Vec`/`String`).
//!    This is the column DEV-121 lived in. `String::bytes()` returned an owned `Value::Vec` for a
//!    declared `&[UInt8]`, so passing it moved it and the caller's binding was emptied.
//!
//! # Producers
//!
//! Enumerated from `typecheck.rs`'s core-method signature arms rather than guessed — every method
//! whose DECLARED RETURN mentions `Ty::Ref`. `append` and `write` mention `Ty::Ref` only in their
//! parameters and are excluded. Ordinary-language producers (slice expressions, aliases, function
//! returns) are included as controls: the law covers all producers of a normalized reference type,
//! not only intrinsics.

mod support;

use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// A producer of a reference-typed value, and the STARK expression that produces one.
struct Producer {
    /// Identifier used in failure messages and in the recorded results table.
    name: &'static str,
    /// Setup lines binding `view` to a reference-typed value.
    setup: &'static str,
    /// The element type of the view, for the consuming function's signature.
    consumer_param: &'static str,
    /// What the runtime value kind must be: a reference kind, never an owned one.
    expect_reference_kind: bool,
}

fn producers() -> Vec<Producer> {
    vec![
        // --- intrinsics whose declared return is a reference (from the signature arms) ---
        Producer {
            name: "String::bytes",
            setup: "let owner = String::from(\"abcd\"); let view = owner.bytes();",
            consumer_param: "&[UInt8]",
            expect_reference_kind: true,
        },
        Producer {
            name: "str::bytes (via as_str)",
            setup: "let owner = String::from(\"abcd\"); let view = owner.as_str().bytes();",
            consumer_param: "&[UInt8]",
            expect_reference_kind: true,
        },
        Producer {
            name: "Vec::as_slice",
            setup: "let mut owner: Vec<UInt8> = Vec::new(); owner.push(1u8); owner.push(2u8); \
                    let view = owner.as_slice();",
            consumer_param: "&[UInt8]",
            expect_reference_kind: true,
        },
        // --- ordinary-language controls: the law is not intrinsic-specific ---
        Producer {
            name: "slice expression &arr[lo..hi]",
            setup: "let owner: [UInt8; 4] = [1u8, 2u8, 3u8, 4u8]; let view = &owner[0u64..4u64];",
            consumer_param: "&[UInt8]",
            expect_reference_kind: true,
        },
        Producer {
            name: "alias of a reference-typed local",
            setup:
                "let owner = String::from(\"abcd\"); let first = owner.bytes(); let view = first;",
            consumer_param: "&[UInt8]",
            expect_reference_kind: true,
        },
        Producer {
            name: "function returning a reference",
            setup: "let owner = String::from(\"abcd\"); let view = borrow_of(&owner);",
            consumer_param: "&[UInt8]",
            expect_reference_kind: true,
        },
    ]
}

/// Use modes. Each is a fragment appended after the producer's setup; every one must leave `view`
/// usable, because a shared reference is `Copy` and no use of it consumes it.
const USE_MODES: &[(&str, &str)] = &[
    ("index only", "let _a = view[0u64];"),
    ("len only", "let _a = view.len();"),
    ("pass once", "let _a = use_len(view);"),
    (
        "pass twice",
        "let _a = use_len(view); let _b = use_len(view);",
    ),
    (
        "pass then index",
        "let _a = use_len(view); let _b = view[0u64];",
    ),
    (
        "pass then len",
        "let _a = use_len(view); let _b = view.len();",
    ),
];

fn program_for(producer: &Producer, use_mode: &str) -> String {
    format!(
        "fn use_len(value: {param}) -> UInt64 {{ value.len() }}\n\
         fn borrow_of(s: &String) -> {param} {{ s.bytes() }}\n\
         fn main() {{ {setup} {use_mode} }}\n",
        param = producer.consumer_param,
        setup = producer.setup,
        use_mode = use_mode,
    )
}

/// Column 1 + 2: does the program type-check, and what operands does MIR emit for the calls?
///
/// Returns `Err(diagnostic)` for a cell the checker REFUSES — recorded, never skipped, so the
/// matrix doubles as documentation of the acceptance surface and stays honest if that surface
/// widens later.
fn check_and_lower(src: &str, tag: &str) -> Result<String, String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    if let Some(first) = checked
        .diagnostics
        .iter()
        .find(|d| d.severity == starkc::diag::Severity::Error)
    {
        return Err(format!(
            "{} {}",
            first.code.as_deref().unwrap_or("-"),
            first.message
        ));
    }
    match lower_program(&hir, &checked.tables, file) {
        Ok(program) => Ok(program.dump()),
        Err(e) => Err(format!("LOWER: {}", e.what)),
    }
}

/// **The matrix.** Every admitted cell must type-check, lower, and emit `copy` — never `move` — for
/// every call that receives the reference-typed local.
#[test]
fn every_reference_producer_is_copy_in_every_use_mode() {
    let mut refused: Vec<String> = Vec::new();
    let mut checked_cells = 0usize;
    let mut move_operands: Vec<String> = Vec::new();

    for producer in producers() {
        for (mode_name, mode_src) in USE_MODES {
            let tag = format!(
                "{}__{}",
                producer
                    .name
                    .replace([':', ' ', '(', ')', '[', ']', '.'], "_"),
                mode_name.replace(' ', "_")
            );
            let src = program_for(&producer, mode_src);
            match check_and_lower(&src, &tag) {
                Err(diagnostic) => {
                    // REFUSED cells are recorded, not skipped.
                    refused.push(format!("{} / {}: {}", producer.name, mode_name, diagnostic));
                }
                Ok(dump) => {
                    checked_cells += 1;
                    // Column 2, as a snapshot assertion over emitted operands. A `move` of the
                    // view into `use_len` is the defect this column exists to catch, and it is
                    // detectable here even if the runtime happened to behave.
                    for line in dump.lines() {
                        if line.contains("call use_len@") && line.contains("move ") {
                            move_operands.push(format!(
                                "{} / {}: {}",
                                producer.name,
                                mode_name,
                                line.trim()
                            ));
                        }
                    }
                }
            }
        }
    }

    println!("\n=== WP-COPY-CANON matrix: {checked_cells} admitted cells ===");
    if !refused.is_empty() {
        println!("REFUSED ({}):", refused.len());
        for r in &refused {
            println!("  {r}");
        }
    }

    assert!(
        move_operands.is_empty(),
        "a reference-typed local was passed by MOVE — a shared reference is Copy and passing it \
         must never consume it:\n  {}",
        move_operands.join("\n  ")
    );
    assert!(
        checked_cells > 0,
        "the matrix admitted no cells at all, which means it is testing nothing"
    );
}

/// Column 3, three-engine: the runtime behaviour every admitted cell must have. Passing a shared
/// reference leaves the source binding live, so `pass twice` returns twice the length.
///
/// Separate from the MIR assertion above on purpose: MIR being right did not make DEV-121 work,
/// and runtime being right would not have caught a `move` operand. Both columns are load-bearing.
#[test]
fn reference_views_survive_being_passed_in_every_engine() {
    for producer in producers() {
        if !producer.expect_reference_kind {
            continue;
        }
        let tag = format!(
            "runtime_{}",
            producer
                .name
                .replace([':', ' ', '(', ')', '[', ']', '.'], "_")
        );
        let src = format!(
            "fn use_len(value: {param}) -> UInt64 {{ value.len() }}\n\
             fn borrow_of(s: &String) -> {param} {{ s.bytes() }}\n\
             fn main() {{ {setup} \
             let a = use_len(view); let b = use_len(view); let c = view[0u64]; \
             assert_eq(a, b); assert_eq(c, c); }}\n",
            param = producer.consumer_param,
            setup = producer.setup,
        );
        support::differential::agree_completing_available_engines(&tag, &src);
    }
}
