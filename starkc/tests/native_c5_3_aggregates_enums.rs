//! WP-C5.3a/C5.3b bring-up proofs that belong to the native engine alone: the generated-source
//! shape (nominal definitions for structs and enums), and the SCOPE BOUNDARY — what the backend
//! refuses, and how.
//!
//! Value agreement for aggregates lives in `three_engine_differential.rs`, which is where §14's
//! C5.3 exit condition is discharged. This file covers what a three-engine comparator
//! structurally cannot: a program that one engine must reject.

use starkc::backend::generated_rust::{emit_native_debug, BackendDiagnostic, NativeBuildOptions};
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn build(source: &str, tag: &str) -> Result<(String, std::process::Output), BackendDiagnostic> {
    let file = Arc::new(SourceFile::new(
        format!("c5_3_{tag}.stark"),
        source.to_string(),
    ));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "{tag} parse: {parse_diags:?}");
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "{tag} resolve: {resolve_diags:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let type_errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(type_errors.is_empty(), "{tag} typecheck: {type_errors:?}");

    let program = match lower_program(&hir, &checked.tables, file) {
        Ok(program) => program,
        Err(e) => panic!("{tag} must lower: {} @ {:?}", e.what, e.span),
    };
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} must verify: {e:?}"));

    let target_dir = std::env::temp_dir().join(format!("stark_c5_3_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);
    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: target_dir.clone(),
            target_contract: "stark-64-v1".to_string(),
        },
    )?;
    let generated = std::fs::read_to_string(artifact.build_dir.join("src/main.rs")).unwrap();
    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("running the generated binary failed");
    let _ = std::fs::remove_dir_all(&target_dir);
    Ok((generated, run))
}

/// Split generated source into (projection module, everything else) by brace counting from
/// `mod stark_proj {`. Deliverable 2's check is about where `unsafe` may appear, so it needs an
/// exact boundary rather than a substring guess.
fn split_projection_module(generated: &str) -> (String, String) {
    let Some(start) = generated.find("mod stark_proj {") else {
        return (String::new(), generated.to_string());
    };
    let mut depth = 0usize;
    let mut end = start;
    for (offset, ch) in generated[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = start + offset + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    let module = generated[start..end].to_string();
    let mut rest = generated[..start].to_string();
    rest.push_str(&generated[end..]);
    (module, rest)
}

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// §6.3: one Rust definition per reachable concrete nominal instance, and no derived traits
/// beyond what MIR's own `Copy` classification calls for.
#[test]
fn each_nominal_instance_gets_exactly_one_generated_definition() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Point { x: Int32, y: Int32 }

struct Wrapper { p: Point }

fn main() {
    let a: Point = Point { x: 1, y: 2 };
    let b: Point = Point { x: 3, y: 4 };
    let w: Wrapper = Wrapper { p: Point { x: 5, y: 6 } };
    assert_eq(a.x + b.y + w.p.x, 10);
}
"#;
    let (generated, run) = build(source, "nominals").expect("must build");
    assert_eq!(
        run.status.code(),
        Some(0),
        "in-program assertions must hold"
    );

    // TWO nominals, TWO definitions -- three `Point` values do not produce three definitions.
    assert_eq!(
        generated.matches("struct stark_ty_").count(),
        2,
        "expected exactly one definition per nominal instance:\n{generated}"
    );
    // Neither struct has an `impl Copy` in STARK, so neither generated type derives anything.
    assert!(
        !generated.contains("#[derive("),
        "no STARK type here is Copy, so nothing should be derived:\n{generated}"
    );
}

/// The flagged §6.3-vs-§7.4 reading (CD-056): a STARK `impl Copy` — and ONLY that — makes the
/// generated type derive `Clone, Copy`. If the owner overrules the reading, this test is what
/// changes.
#[test]
fn a_stark_impl_copy_is_what_makes_a_generated_type_copy() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Marked { v: Int32 }

impl Copy for Marked {}

struct Unmarked { v: Int32 }

fn main() {
    let a: Marked = Marked { v: 1 };
    let b: Unmarked = Unmarked { v: 2 };
    assert_eq(a.v + b.v, 3);
}
"#;
    let (generated, run) = build(source, "copyimpl").expect("must build");
    assert_eq!(run.status.code(), Some(0));
    assert_eq!(
        generated.matches("#[derive(Clone, Copy)]").count(),
        1,
        "exactly the marked type should derive Copy:\n{generated}"
    );
}

/// **WP-C5.3d-0's payoff.** This program was refused before the slot foundation existed: passing
/// a non-`Copy` struct by value puts the move in a different basic block from the construction,
/// and the block-dispatch loop made Rust's borrow checker reject a move verified MIR proves
/// sound. With `ValueSlot` carrying liveness explicitly, it compiles and runs.
#[test]
fn a_cross_block_non_copy_move_now_compiles_and_runs() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Point { x: Int32, y: Int32 }

fn sum(p: Point) -> Int32 {
    p.x + p.y
}

fn main() {
    let p: Point = Point { x: 3, y: 4 };
    assert_eq(p.x, 3);
    assert_eq(sum(p), 7);
}
"#;
    let (generated, run) = build(source, "crossblock").expect("slots must make this expressible");
    assert_eq!(
        run.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        generated.contains("ValueSlot"),
        "the non-Copy local must be slot-backed:\n{generated}"
    );
    // Deliverable 2: no ad hoc unsafe in emitted MIR bodies. `unsafe` may appear only inside the
    // generated projection module, which this program does use -- its `Copy` field reads go
    // through raw projections so they survive a sibling move.
    let (projections, _) = split_projection_module(&generated);
    assert_eq!(
        generated.matches("unsafe").count(),
        projections.matches("unsafe").count(),
        "every `unsafe` must be inside `mod stark_proj`:\n{generated}"
    );
}

/// The guard must not over-reject: moving a non-`Copy` value WITHIN one block is how ordinary
/// aggregate construction lowers (`_2 = aggregate ..; _1 = move _2;`) and must keep working.
#[test]
fn a_same_block_non_copy_move_still_builds() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Point { x: Int32, y: Int32 }

fn main() {
    let p: Point = Point { x: 3, y: 4 };
    assert_eq(p.x + p.y, 7);
}
"#;
    let (_, run) = build(source, "sameblock").expect("same-block moves must still build");
    assert_eq!(run.status.code(), Some(0));
}

// ------------------------------------------------------------ WP-C5.3b: enums --

/// §6.3 again, for enums: one Rust definition per instance, with uniformly TUPLE variants so an
/// empty payload needs no unit-variant special case in construction, patterns, or the
/// discriminant match.
#[test]
fn an_enum_instance_gets_one_definition_with_uniform_tuple_variants() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"enum Shape { Point, Circle(Int32), Rect(Int32, Int32) }

fn main() {
    let s: Shape = Shape::Rect(3, 4);
    let a: Int32 = match s {
        Shape::Point => 0,
        Shape::Circle(r) => r,
        Shape::Rect(w, h) => w * h,
    };
    assert_eq(a, 12);
}
"#;
    let (generated, run) = build(source, "enumdef").expect("must build");
    assert_eq!(run.status.code(), Some(0));

    assert_eq!(
        generated.matches("enum stark_ty_").count(),
        1,
        "expected exactly one enum definition:\n{generated}"
    );
    // The fieldless variant is still a tuple variant -- `V0()`, not `V0`.
    assert!(
        generated.contains("V0(),"),
        "an empty payload must still be a tuple variant:\n{generated}"
    );
    assert!(
        generated.contains("V1(i32),") && generated.contains("V2(i32, i32),"),
        "payload arity must follow the type context:\n{generated}"
    );
}

/// The discriminant is recovered by MATCHING, not by a Rust integer cast: an enum with payloads
/// has no `as` conversion. Every variant is listed with no catch-all, so adding a variant cannot
/// silently fall through to a wrong index.
#[test]
fn a_discriminant_read_lists_every_variant_with_no_catch_all() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"enum Three { A, B, C(Int32) }

fn main() {
    let t: Three = Three::B;
    let v: Int32 = match t {
        Three::A => 1,
        Three::B => 2,
        Three::C(n) => n,
    };
    assert_eq(v, 2);
}
"#;
    let (generated, run) = build(source, "enumdisc").expect("must build");
    assert_eq!(run.status.code(), Some(0));
    for variant in ["V0(..) =>", "V1(..) =>", "V2(..) =>"] {
        assert!(
            generated.contains(variant),
            "the discriminant match must name {variant}:\n{generated}"
        );
    }
}

/// V-DISC-1 makes a variant-field projection legal only after a discriminant test, so the `_`
/// arm of a payload read is provably dead. It is emitted as `unreachable!()` naming the rule --
/// the same treatment the verifier-proved dead-block path gets -- rather than as a fabricated
/// value that would silently paper over a lowering bug.
#[test]
fn a_payload_read_marks_its_impossible_arm_unreachable() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"enum Holder { Has(Int32), Empty }

fn main() {
    let h: Holder = Holder::Has(41);
    let v: Int32 = match h {
        Holder::Has(n) => n + 1,
        Holder::Empty => 0,
    };
    assert_eq(v, 42);
}
"#;
    let (generated, run) = build(source, "enumpayload").expect("must build");
    assert_eq!(run.status.code(), Some(0));
    assert!(
        generated.contains("V-DISC-1"),
        "the impossible arm should name the rule that makes it impossible:\n{generated}"
    );
}

// --------------------------------------------- WP-C5.3d-0: storage and movement --

/// §7.7: an aborting trap must not run pending Drop glue.
///
/// **This is a structural proof, not an observable one, and the difference matters.** The
/// observable version — a destructor with a side effect that must not appear after a trap —
/// cannot be built natively yet: a user `Drop` impl's receiver is `&mut Self`, and references
/// are outside the C5 subset entirely, so `impl Drop` does not compile natively at all (see
/// CD-059). What can be proven now is that the emitted trap path reaches the abort with no drop
/// call before it, which is the mechanism the observable test would exercise.
#[test]
fn a_trapping_block_emits_no_drop_before_its_abort() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Held { v: Int32 }

fn main() {
    let h: Held = Held { v: 1 };
    let a: Int32 = 2147483647;
    let b: Int32 = a + 1;
    assert_eq(h.v, 1);
}
"#;
    let (generated, run) = build(source, "droptrap").expect("must build");
    assert_eq!(run.status.code(), Some(101), "the overflow must trap");

    // Every abort call site must be the first statement of its block: nothing -- least of all a
    // `drop_with` -- may precede it.
    for block in generated.split("trap::abort") {
        let tail: String = block.chars().rev().take(200).collect();
        let preceding: String = tail.chars().rev().collect();
        assert!(
            !preceding.contains("drop_with"),
            "a drop ran on the path to a trap:\n{preceding}"
        );
    }
}

/// Deliverable 6: a partial move must NOT be collapsed into whole-local liveness. Moving one
/// field out and then reading a sibling is the observable consequence -- under a whole-local
/// approximation the sibling read would find a dead slot and abort.
#[test]
fn a_field_move_does_not_kill_its_siblings() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Inner { v: Int32 }

struct Outer { a: Inner, b: Int32 }

fn take_inner(i: Inner) -> Int32 {
    i.v
}

fn main() {
    let o: Outer = Outer { a: Inner { v: 5 }, b: 9 };
    assert_eq(take_inner(o.a), 5);
    assert_eq(o.b, 9);
}
"#;
    let (generated, run) = build(source, "partial").expect("must build");
    assert_eq!(
        run.status.code(),
        Some(0),
        "the sibling read must still work; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    // The move went through a generated wrapper, not a whole-local take -- and the wrapper, not
    // the emitted body, is what calls the unsafe primitive.
    let (projections, bodies) = split_projection_module(&generated);
    assert!(
        projections.contains("slot.move_field("),
        "the move wrapper must call the primitive:\n{projections}"
    );
    assert!(
        projections.contains("slot.copy_field("),
        "the sibling read must use raw projection so it survives the sibling move:\n{projections}"
    );
    assert!(
        !bodies.contains("move_field") && !bodies.contains("copy_field"),
        "emitted bodies must call WRAPPERS, never the unsafe primitives:\n{bodies}"
    );
}

/// Deliverable 2, checked on a program that actually exercises partial moves: the ONLY `unsafe`
/// in a generated program is inside the generated projection module.
#[test]
fn unsafe_appears_only_in_the_generated_projection_module() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Inner { v: Int32 }

struct Outer { a: Inner, b: Int32 }

fn take_inner(i: Inner) -> Int32 {
    i.v
}

fn main() {
    let o: Outer = Outer { a: Inner { v: 5 }, b: 9 };
    assert_eq(take_inner(o.a), 5);
}
"#;
    let (generated, _) = build(source, "scopecheck").expect("must build");
    // NOTE: the tag must not contain "unsafe" -- the source file name is baked into every trap
    // call site, so a tag like "unsafescope" would make this check fail on its own file name.
    let (projections, outside) = split_projection_module(&generated);
    assert!(
        projections.contains("unsafe"),
        "the projection module is where unsafe belongs:\n{projections}"
    );
    let stray: Vec<&str> = outside
        .lines()
        .filter(|line| line.contains("unsafe"))
        .collect();
    assert!(
        stray.is_empty(),
        "every `unsafe` in a generated program must be inside `mod stark_proj`; found outside: \
         {stray:?}"
    );
}

// -------------------------------------- WP-C5.3c: Option, Result, and `?` --

/// Core enums take the SAME generated representation as user enums, from one shared variant
/// table. §6.2 preferred ordinary Rust `Option`/`Result` "if all observable semantics match";
/// generated enums are used instead so one mechanism covers every enum and no Rust drop glue
/// exists for a type MIR is responsible for destroying (CD-060).
#[test]
fn option_and_result_generate_their_own_enums_with_mir_discriminant_order() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"fn half(n: Int32) -> Option<Int32> {
    if n % 2 == 0 { Some(n / 2) } else { None }
}

fn checked(n: Int32) -> Result<Int32, Bool> {
    if n > 0 { Ok(n) } else { Err(true) }
}

fn main() {
    match half(4) {
        Some(v) => assert_eq(v, 2),
        None => assert(false),
    }
    match checked(1) {
        Ok(v) => assert_eq(v, 1),
        Err(e) => assert(false),
    }
}
"#;
    let (generated, run) = build(source, "coreenums").expect("must build");
    assert_eq!(run.status.code(), Some(0));

    // No STARK local is declared at Rust's prelude `Option`/`Result`. The check is on LOCAL
    // DECLARATIONS specifically: Rust's `Option` does legitimately appear elsewhere in the
    // output as an implementation detail of checked arithmetic (`checked_add` returns one), and
    // that has nothing to do with how STARK's `Option` is represented.
    let rust_typed_locals: Vec<&str> = generated
        .lines()
        .filter(|line| line.trim_start().starts_with("let mut _"))
        .filter(|line| line.contains(": Option<") || line.contains(": Result<"))
        .collect();
    assert!(
        rust_typed_locals.is_empty(),
        "core enums must use generated definitions, not Rust's: {rust_typed_locals:?}"
    );
    // Two generated core enums, each with its MIR variant order: Option is None=0/Some=1, so V1
    // carries the payload; Result is Ok=0/Err=1, so V0 does.
    assert!(
        generated.contains("V0(),\n    V1(i32),"),
        "Option must be None=0 then Some(T)=1:\n{generated}"
    );
    assert!(
        generated.contains("V0(i32),\n    V1(bool),"),
        "Result must be Ok(T)=0 then Err(E)=1:\n{generated}"
    );
}

/// `?` is already ordinary control flow by the time the backend sees it: §14 requires emitting
/// the lowered flow "without reconstructing source patterns". The check is that no Rust `?`
/// appears in the output -- the propagation is branches and returns, not a borrowed Rust
/// operator whose semantics would then have to be argued equivalent.
#[test]
fn question_mark_emits_control_flow_not_a_rust_question_mark() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"fn parse(n: Int32) -> Result<Int32, Bool> {
    if n > 0 { Ok(n * 2) } else { Err(true) }
}

fn twice(n: Int32) -> Result<Int32, Bool> {
    let first: Int32 = parse(n)?;
    Ok(first)
}

fn main() {
    match twice(2) {
        Ok(v) => assert_eq(v, 4),
        Err(e) => assert(false),
    }
}
"#;
    let (generated, run) = build(source, "qmark").expect("must build");
    assert_eq!(run.status.code(), Some(0));
    assert!(
        !generated.contains(")?;") && !generated.contains(")?\n"),
        "`?` must be emitted as MIR's own branches, not Rust's operator:\n{generated}"
    );
}

// ------------------------------ WP-C5.3d-1a: the ephemeral reference lane --

/// The lane's negative cases: a reference outside the admitted shape must be refused **before
/// rustc**, as a named STARK limitation, not as a borrow-check error in generated code.
///
/// Each source here is rejected by the front end or the backend; what the test pins is that the
/// rejection happens on OUR side of the boundary. A reference that reached rustc and failed there
/// would be a diagnostic defect even though the program is correctly not compiled.
///
/// **WP-C6.1f-b3:** the "store a reference in a user binding that outlives its block" case has been
/// MOVED OUT of this list to `native_c61f_b3_stored_refs.rs` as a positive test — stored references
/// are now supported, so this test followed its own instruction ("if it is now legitimately
/// supported, move it to a positive test"). Returning a reference is still refused.
#[test]
fn references_outside_the_lane_are_refused_before_rustc() {
    for (tag, source) in [
        // Returning a reference.
        (
            "ret",
            r#"fn pick(a: &Int32) -> &Int32 {
    a
}

fn main() {
    let x: Int32 = 1;
    let r: &Int32 = pick(&x);
}
"#,
        ),
    ] {
        let file = Arc::new(SourceFile::new(
            format!("c5_3_ref_{tag}.stark"),
            source.to_string(),
        ));
        let (ast, parse_diags) = parse(&file, ParseMode::Program);
        if !parse_diags.is_empty() {
            continue; // rejected at parse: still before rustc
        }
        let (hir, resolve_diags) = resolve(&ast, file.clone());
        if !resolve_diags.is_empty() {
            continue;
        }
        let checked = typecheck::analyze(&hir, file.clone());
        if checked
            .diagnostics
            .iter()
            .any(|d| d.severity == starkc::diag::Severity::Error)
        {
            continue; // rejected by the front end: before rustc
        }
        let Ok(program) = lower_program(&hir, &checked.tables, file) else {
            continue; // rejected by lowering: before rustc
        };
        let Ok(verified) = verify_program(&program) else {
            continue; // rejected by the verifier: before rustc
        };
        let target_dir =
            std::env::temp_dir().join(format!("stark_c5_3_ref_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&target_dir);
        let result = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: target_dir.clone(),
                target_contract: "stark-64-v1".to_string(),
            },
        );
        let _ = std::fs::remove_dir_all(&target_dir);
        match result {
            Err(BackendDiagnostic::Unsupported(_)) => {}
            Err(BackendDiagnostic::BuildFailed(failure)) => panic!(
                "{tag}: a reference outside the lane reached rustc and failed THERE; the backend \
                 must refuse it first:\n{}",
                failure.stderr
            ),
            Err(other) => panic!("{tag}: expected an Unsupported refusal, got {other:?}"),
            Ok(_) => panic!(
                "{tag}: this reference shape is outside the C5 lane and must be refused; if it \
                 is now legitimately supported, move it to a positive test"
            ),
        }
    }
}

/// `Ordering` and a user destructor both compile now, and both go through the lane. The
/// generated destructor takes a real Rust reference receiver rather than a slot.
#[test]
fn the_lane_admits_orderings_and_destructor_receivers() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Held { v: Int32 }

impl Drop for Held {
    fn drop(&mut self) {
        let read: Int32 = self.v;
    }
}

fn main() {
    let a: Int32 = 1;
    let b: Int32 = 2;
    let o: Ordering = a.cmp(&b);
    let v: Int32 = match o {
        Ordering::Less => 10,
        Ordering::Equal => 20,
        Ordering::Greater => 30,
    };
    assert_eq(v, 10);
    let h: Held = Held { v: 3 };
    assert_eq(h.v, 3);
}
"#;
    let (generated, run) = build(source, "lane").expect("both lane cases must build");
    assert_eq!(run.status.code(), Some(0));
    // The destructor receiver is a plain Rust reference, NOT a ValueSlot: a slot-backed receiver
    // would make the body's Deref project through the slot instead of the reference.
    assert!(
        generated.contains("mut _1: &mut stark_ty_"),
        "the destructor receiver must be a bare reference parameter:\n{generated}"
    );
    // And the borrow for `cmp` is inlined at its use rather than stored.
    assert!(
        generated.contains("(&_"),
        "the shared borrow should appear as a borrow expression:\n{generated}"
    );
}

// ------------------------------- WP-C5.3d-1c: per-unit (sub-place) destruction --

/// **The shape C5.3d-1c's partial-move fixture exposed.**
///
/// MIR's drop elaboration does not emit one whole-local `Drop` for an aggregate with several drop
/// units. It emits one flag-guarded `Drop` per unit, on a PROJECTED place — `drop _1.1` then
/// `drop _1.0` — so a plain two-droppable-field struct needs sub-place destruction. The backend
/// used to refuse that outright, which meant the C5 subset could not compile a struct with two
/// droppable fields at all.
///
/// The refusal was right, not merely conservative: collapsing per-unit drops into a whole-local
/// one would destroy a unit MIR's flags say is already gone (§7.6). The fix is a real per-unit
/// operation — a generated wrapper over `ValueSlot::drop_field_with` — not a relaxation.
#[test]
fn sub_place_drops_go_through_generated_wrappers_in_reverse_unit_order() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct First { x: Int32 }
impl Drop for First {
    fn drop(&mut self) { let r: Int32 = self.x; }
}
struct Second { x: Int32 }
impl Drop for Second {
    fn drop(&mut self) { let r: Int32 = self.x; }
}
struct Pair { a: First, b: Second }

fn main() {
    let p: Pair = Pair { a: First { x: 1 }, b: Second { x: 2 } };
    assert_eq(p.a.x + p.b.x, 3);
}
"#;
    let (generated, run) = build(source, "subdrop").expect("per-unit drops must build");
    assert_eq!(run.status.code(), Some(0));

    let (projections, bodies) = split_projection_module(&generated);
    // Each unit gets its own wrapper, and the wrapper -- not the body -- carries the plan.
    assert!(
        projections.contains("drop_field_with"),
        "sub-place destruction must go through the slot's per-unit primitive:\n{projections}"
    );
    // The call sites are plain safe calls: no `unsafe`, no destruction logic inlined into a body.
    let f0 = bodies
        .find("_23f0(&mut _1)")
        .expect("field 0's drop wrapper must be called");
    let f1 = bodies
        .find("_23f1(&mut _1)")
        .expect("field 1's drop wrapper must be called");
    assert!(
        f1 < f0,
        "MIR sequences the drop units back to front, and the emitter must follow:\n{bodies}"
    );
    let stray: Vec<&str> = bodies
        .lines()
        .filter(|line| line.contains("unsafe"))
        .collect();
    assert!(
        stray.is_empty(),
        "sub-place destruction must not put `unsafe` in a body; found: {stray:?}"
    );
}

/// The partial-move case: one unit is moved out, its sibling still owes a destructor. The moved
/// unit's drop is skipped at run time by MIR's own flag, and the emitter has no say in it — which
/// is the point, since per-unit liveness is MIR's to track (§7.6).
#[test]
fn a_partially_moved_aggregate_still_destroys_its_surviving_unit() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct First { x: Int32 }
impl Drop for First {
    fn drop(&mut self) { let r: Int32 = self.x; }
}
struct Second { x: Int32 }
impl Drop for Second {
    fn drop(&mut self) { let r: Int32 = self.x; }
}
struct Pair { a: First, b: Second }

fn consume(v: First) -> Int32 {
    v.x
}

fn main() {
    let p: Pair = Pair { a: First { x: 1 }, b: Second { x: 2 } };
    assert_eq(consume(p.a), 1);
}
"#;
    let (generated, run) = build(source, "partialdrop").expect("partial moves must build");
    // Exit 0 is the load-bearing assertion. A double destruction of the moved-out unit would hit
    // `ValueSlot`'s state machine and abort through `slot_violation`, not pass quietly.
    assert_eq!(
        run.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    // Both the move wrapper and the surviving unit's drop wrapper are generated over the SAME
    // slot: that pairing is what makes partial liveness expressible at all.
    assert!(
        generated.contains("move_field") && generated.contains("drop_field_with"),
        "the partial-move path needs both a move and a per-unit drop wrapper:\n{generated}"
    );
}

// ----------------------------- WP-C5.3e: the target-layout contract in the backend --

/// **CD-067.** A layout query answers from the named target CONTRACT, emitted as a constant. It
/// must not be `core::mem::size_of::<T>()`: that reports this backend's private physical
/// representation, which would make an observable language answer depend on a transitional
/// backend and on `repr(Rust)`'s deliberately unspecified field ordering.
///
/// The generated crate also asserts NOTHING about its own layout, and its nominals are not
/// `#[repr(C)]`. A host-layout assertion would enforce a rule Core v1 does not have — that the
/// contract equal the backend's representation — and would obstruct a later backend implementing
/// the same contract over a different one.
#[test]
fn layout_queries_emit_contract_constants_and_assert_nothing_about_rust_layout() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Padded { a: Int8, b: Int64 }

fn main() {
    assert_eq(size_of::<Int32>(), 4);
    assert_eq(align_of::<Int64>(), 8);
    assert_eq(size_of::<Padded>(), 16);
    assert_eq(size_of::<[Int32; 4]>(), 16);
    assert_eq(size_of::<Option<Int32>>(), 8);
}
"#;
    let (generated, run) = build(source, "layoutcontract").expect("layout queries must build");
    assert_eq!(
        run.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );

    assert!(
        !generated.contains("core::mem::size_of") && !generated.contains("core::mem::align_of"),
        "a layout query must not read the host representation:\n{generated}"
    );
    // The contract's values appear as plain constants.
    for constant in ["4u64", "8u64", "16u64"] {
        assert!(
            generated.contains(constant),
            "expected the contract constant {constant} in the generated source:\n{generated}"
        );
    }
    // No host-layout cross-check, and no repr(C) added for one.
    assert!(
        !generated.contains("repr(C)"),
        "generated nominals must stay repr(Rust); CD-067 rejected repr(C)-for-cross-check:\n{generated}"
    );
    assert!(
        !generated.contains("assert!(core::mem::"),
        "the generated crate must assert nothing about its own physical layout:\n{generated}"
    );
}

/// The contract's identity reaches the build report, so a build's observable layout answers can be
/// attributed to a named contract at a stated version.
///
/// Built inline rather than through `build()`, which deletes its target directory on the way out.
#[test]
fn the_build_report_records_the_layout_contract_identity() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = "fn main() { assert_eq(size_of::<Int32>(), 4); }\n";
    let file = Arc::new(SourceFile::new("c5_3_report.stark", source.to_string()));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file.clone());
    let Ok(program) = lower_program(&hir, &checked.tables, file) else {
        panic!("must lower");
    };
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("must verify: {e:?}"));
    let target_dir = std::env::temp_dir().join(format!("stark_c5_3_report_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);
    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: target_dir.clone(),
            target_contract: "stark-64-v1".to_string(),
        },
    )
    .expect("must build");
    let manifest =
        std::fs::read_to_string(artifact.build_dir.join("build.json")).expect("build.json");
    let _ = std::fs::remove_dir_all(&target_dir);
    for field in [
        "\"target_contract\": \"stark-64-v1\"",
        "\"layout_contract_version\": 1",
        "\"compiler_layout_revision\": 1",
    ] {
        assert!(
            manifest.contains(field),
            "the build report must carry {field}:\n{manifest}"
        );
    }
}

/// An unknown target contract is REJECTED before emission, not silently defaulted: a layout answer
/// is observable and target-specific (LAYOUT-ABI-001).
#[test]
fn an_unknown_target_contract_is_rejected_before_emission() {
    let source = "fn main() { assert_eq(size_of::<Int32>(), 4); }\n";
    let file = Arc::new(SourceFile::new("c5_3_badtarget.stark", source.to_string()));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file.clone());
    let Ok(program) = lower_program(&hir, &checked.tables, file) else {
        panic!("must lower");
    };
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("must verify: {e:?}"));
    let result = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: std::env::temp_dir().join("stark_c5_3_badtarget"),
            target_contract: "stark-128-v9".to_string(),
        },
    );
    let Err(err) = result else {
        panic!("an unknown contract must be refused");
    };
    match err {
        BackendDiagnostic::Unsupported(message) => assert!(
            message.contains("no layout contract named"),
            "unexpected refusal: {message}"
        ),
        other => panic!("expected an Unsupported refusal, got {other:?}"),
    }
}

// ------------------ CD-070 → WP-C6.1c: the multi-unit enum payload boundary, now CLOSED --

/// The CD-070 adversarial fixture, **now a positive test**: WP-C6.1c closed the multi-unit
/// enum-payload boundary. `enum E { V(A, B) }` with `match e { E::V(a, b) => … }` moves one
/// non-`Copy` payload unit out while a droppable sibling remains — which the C5 backend refused
/// (an enum payload has no raw projection, so the second `VariantField` move hit a partial slot).
///
/// C6.1c lowers the active-variant payload into ONE canonical tuple aggregate, which the backend
/// emits as a single destructuring `match e.take() { E::V(f0, f1) => (f0, f1), … }`; after that,
/// per-field movement is ordinary raw-projectable tuple-field machinery. The whole enum is moved
/// once, so no partial-slot access ever occurs. These build AND run to a successful exit (the
/// `assert_eq` observations hold, and no `slot_violation` fires from a mis-drop).
#[test]
fn a_partial_move_out_of_a_multi_unit_enum_payload_builds_and_runs() {
    let sources = [
        // One unit consumed, the sibling unbound but still owing a destructor (dropped at arm end).
        r#"struct A { x: Int32 }
impl Drop for A { fn drop(&mut self) { let r: Int32 = self.x; } }
struct B { x: Int32 }
impl Drop for B { fn drop(&mut self) { let r: Int32 = self.x; } }
enum E { V(A, B) }
fn take_a(a: A) -> Int32 { a.x }
fn main() {
    let e: E = E::V(A { x: 1 }, B { x: 2 });
    let n: Int32 = match e { E::V(a, b) => take_a(a) };
    assert_eq(n, 1);
}
"#,
        // Both units bound and both used: the second access follows the first move.
        r#"struct A { x: Int32 }
impl Drop for A { fn drop(&mut self) { let r: Int32 = self.x; } }
struct B { x: Int32 }
impl Drop for B { fn drop(&mut self) { let r: Int32 = self.x; } }
enum E { V(A, B) }
fn take_a(a: A) -> Int32 { a.x }
fn main() {
    let e: E = E::V(A { x: 1 }, B { x: 2 });
    let n: Int32 = match e { E::V(a, b) => take_a(a) + b.x };
    assert_eq(n, 3);
}
"#,
    ];
    for (i, source) in sources.iter().enumerate() {
        let (generated, run) = build(source, &format!("multiunit{i}"))
            .unwrap_or_else(|e| panic!("case {i} must build now (C6.1c): {e:?}"));
        assert!(
            run.status.success(),
            "case {i} must exit 0 (asserts hold, no slot_violation); stderr: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        // Structural: exactly one destructuring extraction match for the payload.
        assert_eq!(
            generated.matches(".take() {").count(),
            1,
            "case {i}: the payload decomposition must emit exactly one `take()` destructure"
        );
    }
}

/// The negative control for the refusal above: a SINGLE-unit payload move is the approved
/// consuming-match shape and must keep working. A refusal that rejected every payload move would
/// pass the test above while breaking `Option`/`Result` entirely.
#[test]
fn a_single_unit_enum_payload_move_still_works() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct A { x: Int32 }
impl Drop for A { fn drop(&mut self) { let r: Int32 = self.x; } }
enum E { V(A) }
fn take_a(a: A) -> Int32 { a.x }
fn main() {
    let e: E = E::V(A { x: 7 });
    let n: Int32 = match e { E::V(a) => take_a(a) };
    assert_eq(n, 7);
}
"#;
    let (_, run) = build(source, "singleunit").expect("a single-unit payload move must build");
    assert_eq!(
        run.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}
