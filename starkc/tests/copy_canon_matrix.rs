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
use starkc::typecheck;

/// AS1b-ii: a real registered source for a hand-built MIR program.
/// The one registry a hand-built `MirProgram` in this file is measured against.
///
/// AS1b-iii: a fixture used to state its source twice — a `RegisteredSource` for the spans and an
/// unrelated `Arc<SourceFile>` in `MirProgram::files`, often under a different name. Nothing
/// checked that they agreed, which is the duplication the amendment removes. Now the program
/// carries the registry the handle came from, so there is nothing to keep in step.
fn test_sources() -> starkc::source::SourceTable {
    let mut registry = starkc::source::SourceRegistry::default();
    registry.intern(std::sync::Arc::new(starkc::source::SourceFile::new(
        "test.stark",
        "",
    )));
    registry.freeze()
}

fn test_source() -> starkc::source::RegisteredSource {
    test_sources()
        .entry()
        .expect("the registry was just populated")
        .clone()
}

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
    let file = std::sync::Arc::new(starkc::source::SourceFile::new(
        format!("{tag}.stark"),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
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
    match lower_program(
        &hir,
        &checked.tables,
        hir.source_named(&file.name).expect("registered"),
    ) {
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

// ------------------------------------------------- chained producers (DEV-126) --

/// A reference-producing expression written over an owner named `owner`, together with the owner's
/// declaration. Chains are the second axis the original matrix lacked.
struct Chain {
    name: &'static str,
    /// The owner's type as a borrowed parameter, e.g. `&String`.
    owner_param: &'static str,
    /// How `main` builds the owner.
    owner_init: &'static str,
    /// The expression producing a `&[UInt8]` from `owner`. This is the whole subject of the test.
    expr: &'static str,
}

/// **The chains, each a producer applied to the result of another producer.**
///
/// # Why this axis exists
///
/// The original matrix crossed producers with USE MODES and never with each other, and it tested
/// escape only for a direct `bytes()`. DEV-126 lived in the cell that crossing would have covered:
/// `owner.as_str().bytes()` returned from a function dangled, while `owner.bytes()` — same type,
/// same declared lifetime — worked. `as_str` returned a detached copy of the string, so the bytes
/// materialised from it had no owner to anchor to and were promoted into the frame about to pop.
///
/// The failure was found by CI, in `stark-json`, whose hot loop is exactly
/// `cursor.input.as_str().bytes()`. A matrix with this axis would have found it first, which is the
/// entire argument for adding it.
///
/// Provenance, not type, is what varies here: every chain has the identical declared type
/// `&[UInt8]`, so anything that distinguishes them is by definition a representation defect.
fn chains() -> Vec<Chain> {
    vec![
        // The control: one step, already covered, included so a failure of the whole group is
        // distinguishable from a failure of chaining specifically.
        Chain {
            name: "bytes (direct, control)",
            owner_param: "&String",
            owner_init: "let owner = String::from(\"abcd\");",
            expr: "owner.bytes()",
        },
        // DEV-126 itself.
        Chain {
            name: "as_str then bytes",
            owner_param: "&String",
            owner_init: "let owner = String::from(\"abcd\");",
            expr: "owner.as_str().bytes()",
        },
        // A view re-sliced: the second step consumes the first step's reference as a place.
        Chain {
            name: "bytes then re-slice",
            owner_param: "&String",
            owner_init: "let owner = String::from(\"abcd\");",
            expr: "&owner.bytes()[1u64..3u64]",
        },
        Chain {
            name: "as_str then bytes then re-slice",
            owner_param: "&String",
            owner_init: "let owner = String::from(\"abcd\");",
            expr: "&owner.as_str().bytes()[1u64..3u64]",
        },
        // The Vec side of the same shape.
        Chain {
            name: "as_slice (direct, control)",
            owner_param: "&Vec<UInt8>",
            owner_init: "let mut owner: Vec<UInt8> = Vec::new(); owner.push(1u8); \
                         owner.push(2u8); owner.push(3u8); owner.push(4u8);",
            expr: "owner.as_slice()",
        },
        Chain {
            name: "as_slice then re-slice",
            owner_param: "&Vec<UInt8>",
            owner_init: "let mut owner: Vec<UInt8> = Vec::new(); owner.push(1u8); \
                         owner.push(2u8); owner.push(3u8); owner.push(4u8);",
            expr: "&owner.as_slice()[1u64..3u64]",
        },
    ]
}

/// The escaping form: the chain is evaluated inside a function and RETURNED. Core v1 admits this —
/// a returned reference deriving from a reference parameter — so the program is valid and any
/// failure is a representation defect.
///
/// This is the position DEV-126 failed in and the position the original matrix never put a chained
/// producer in.
#[test]
fn chained_producers_survive_escaping_their_defining_function() {
    for chain in chains() {
        let tag = format!(
            "chain_escape_{}",
            chain.name.replace([' ', '(', ')', ',', '.'], "_")
        );
        let src = format!(
            "fn use_len(value: &[UInt8]) -> UInt64 {{ value.len() }}\n\
             fn make(owner: {owner_param}) -> &[UInt8] {{ {expr} }}\n\
             fn main() {{ {owner_init} \
             let view = make(&owner); \
             let a = use_len(view); let b = use_len(view); let c = view[0u64]; \
             assert_eq(a, b); assert_eq(c, c); }}\n",
            owner_param = chain.owner_param,
            owner_init = chain.owner_init,
            expr = chain.expr,
        );
        support::differential::agree_completing_available_engines(&tag, &src);
    }
}

/// The same chains used locally, without crossing a function boundary.
///
/// Kept as the contrast: DEV-126 PASSED in this position and failed in the escaping one, so a
/// matrix holding only this form reports success for a broken representation. Its value is
/// entirely in being compared against the test above.
#[test]
fn chained_producers_are_usable_in_place() {
    for chain in chains() {
        let tag = format!(
            "chain_local_{}",
            chain.name.replace([' ', '(', ')', ',', '.'], "_")
        );
        let src = format!(
            "fn use_len(value: &[UInt8]) -> UInt64 {{ value.len() }}\n\
             fn main() {{ {owner_init} \
             let view = {expr}; \
             let a = use_len(view); let b = use_len(view); let c = view[0u64]; \
             assert_eq(a, b); assert_eq(c, c); }}\n",
            owner_init = chain.owner_init,
            expr = chain.expr,
        );
        support::differential::agree_completing_available_engines(&tag, &src);
    }
}

// ---------------------------------------------------------- INV-MOVE-001 --

/// Liveness fixtures for INV-MOVE-001 (MIR-0036).
///
/// **Why these exist at all.** Once the desugar defect (DEV-124) is fixed, no STARK program
/// reaches the invariant — which is the goal, and also means a broken invariant would look exactly
/// like a working one. An invariant nothing can trip is indistinguishable from `if false`. So the
/// rule is exercised on hand-built MIR: one body that must be rejected, and two that must not.
mod inv_move_001 {
    use starkc::mir::{
        self, BasicBlock, Constant, LocalDecl, LocalKind, MirBody, MirProgram, MirTy, Operand,
        Place, Rvalue, SourceInfo, Statement, Terminator, TypeContext,
    };

    fn info() -> SourceInfo {
        SourceInfo {
            span: super::test_source().synthetic_span(),
            origin: mir::Origin::UserCode,
        }
    }

    fn place(index: u32) -> Place {
        Place {
            local: mir::LocalId(index),
            projection: Vec::new(),
        }
    }

    /// `_1: ty = <init>; _2: ty = <read> _1; return`. The operand is the only variable, so a
    /// rejection can only be about the operand.
    fn program_reading(ty: MirTy, init: Rvalue, read: Operand) -> MirProgram {
        let body = MirBody {
            instance: mir::Instance {
                item: starkc::hir::ItemId(0),
                type_args: Vec::new(),
                symbol: "main@[]".to_string(),
            },
            params: Vec::new(),
            ret: MirTy::Unit,
            locals: vec![
                LocalDecl {
                    ty: MirTy::Unit,
                    kind: LocalKind::Return,
                },
                LocalDecl {
                    ty: ty.clone(),
                    kind: LocalKind::Temp,
                },
                LocalDecl {
                    ty,
                    kind: LocalKind::Temp,
                },
                // `_3: UInt64` — a referent for the non-Copy fixture's `RefOf`. Without it that
                // fixture had to point at `_2`, which is itself `&mut UInt64`, so the rvalue typed
                // as `&mut &mut UInt64` against a `&mut UInt64` local and the body was rejected
                // MIR-0004 before the operand rule was ever consulted. A control that fails for
                // the wrong reason proves nothing about the rule it is controlling for.
                LocalDecl {
                    ty: MirTy::UInt64,
                    kind: LocalKind::Temp,
                },
            ],
            blocks: vec![BasicBlock {
                statements: vec![
                    (Statement::Assign(place(1), init), info()),
                    (Statement::Assign(place(2), Rvalue::Use(read)), info()),
                ],
                terminator: (Terminator::Return, info()),
            }],
            entry: mir::BlockId(0),
        };
        MirProgram {
            entry_source: super::test_source().id(),
            sources: super::test_sources(),
            bodies: vec![body],
            types: TypeContext::default(),
            mir_version: mir::MIR_VERSION.to_string(),
            runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
            provider_calls: Vec::new(),
            resource_bindings: Vec::new(),
            provider_closes: Vec::new(),
        }
    }

    fn scalar_init() -> Rvalue {
        Rvalue::Use(Operand::Const(Constant::Int(0, MirTy::UInt64)))
    }

    /// The rule. A `Copy` type's contract is that reading leaves the source intact; `Move` empties
    /// it. MIR must not assert both about one value.
    #[test]
    fn a_move_from_a_copy_place_is_rejected() {
        let program = program_reading(MirTy::UInt64, scalar_init(), Operand::Move(place(1)));
        match mir::verify::verify_program(&program) {
            Ok(_) => panic!("INV-MOVE-001 is dead: a move from a Copy UInt64 place verified"),
            Err(errors) => assert!(
                errors.iter().any(|e| e.code == "MIR-0036"),
                "expected MIR-0036, got: {errors:#?}"
            ),
        }
    }

    /// Control one: the same body, `copy` instead of `move`, must verify. Without this the test
    /// above proves only that the fixture is malformed somehow.
    #[test]
    fn a_copy_from_a_copy_place_verifies() {
        let program = program_reading(MirTy::UInt64, scalar_init(), Operand::Copy(place(1)));
        if let Err(errors) = mir::verify::verify_program(&program) {
            panic!("the copy form must verify: {errors:#?}");
        }
    }

    /// Control two: the rule is *not* "never move". A `&mut` is not `Copy`, and moving one is both
    /// legal and necessary — an over-broad invariant would reject this and would have been the
    /// wrong fix for DEV-124.
    #[test]
    fn a_move_from_a_non_copy_place_verifies() {
        let ty = MirTy::Ref {
            mutable: true,
            inner: Box::new(MirTy::UInt64),
        };
        let init = Rvalue::RefOf {
            mutable: true,
            place: place(3),
        };
        let program = program_reading(ty, init, Operand::Move(place(1)));
        if let Err(errors) = mir::verify::verify_program(&program) {
            panic!("moving a non-Copy place must verify: {errors:#?}");
        }
    }
}
