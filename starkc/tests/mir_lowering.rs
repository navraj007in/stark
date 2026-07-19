//! WP-C4.2 — scalar HIR→MIR lowering tests.
//!
//! Verifies against real programs (including frozen exec_snapshots corpus cases,
//! `corpus_version = 1.0.0`): that the scalar subset lowers, that the dump is deterministic,
//! that structural contract invariants hold on every produced body (single sealed terminator
//! per block, `SourceInfo` present with a valid `FileId` on every statement and terminator —
//! V-SRC-1's data prerequisite), and that out-of-subset constructs are reported as clean
//! `Unsupported` errors naming C4.5 rather than mislowered (charter: nothing unsupported
//! reaches a backend silently). Execution-level differential validation is WP-C4.4 (the MIR
//! interpreter), not this file.

use starkc::diag::Severity;
use starkc::mir::{self, lower::lower_program};
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::PathBuf;
use std::sync::Arc;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/exec_snapshots")
}

struct Front {
    hir: starkc::hir::Hir,
    file: Arc<SourceFile>,
    tables: starkc::typecheck::TypeTables,
}

fn front_end_src(name: &str, source: String) -> Front {
    let file = Arc::new(SourceFile::new(name, source));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{name}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{name}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{name}: typecheck: {errors:?}");
    Front {
        hir,
        file,
        tables: checked.tables,
    }
}

fn front_end_corpus(name: &str) -> Front {
    let path = corpus_dir().join(format!("{name}.stark"));
    let source = std::fs::read_to_string(&path).unwrap();
    front_end_src(&path.to_string_lossy(), source)
}

/// Contract structural invariants every lowered program must satisfy (pre-verifier, WP-C4.3
/// builds the real verifier on top of these).
fn assert_structural_invariants(program: &mir::MirProgram) {
    assert!(!program.files.is_empty(), "file table must not be empty");
    for body in &program.bodies {
        assert!(
            !body.blocks.is_empty(),
            "{}: body must have at least one block",
            body.instance.symbol
        );
        assert!(
            (body.entry.0 as usize) < body.blocks.len(),
            "{}: entry block in bounds",
            body.instance.symbol
        );
        for (bi, block) in body.blocks.iter().enumerate() {
            // Every statement and the terminator carry SourceInfo with a valid FileId.
            for (_, info) in &block.statements {
                assert!(
                    (info.file.0 as usize) < program.files.len(),
                    "{} bb{bi}: statement SourceInfo has invalid FileId",
                    body.instance.symbol
                );
            }
            let (_, term_info) = &block.terminator;
            assert!(
                (term_info.file.0 as usize) < program.files.len(),
                "{} bb{bi}: terminator SourceInfo has invalid FileId",
                body.instance.symbol
            );
            // Terminator targets are in bounds.
            let mut targets: Vec<mir::BlockId> = Vec::new();
            match &block.terminator.0 {
                mir::Terminator::Goto { target } => targets.push(*target),
                mir::Terminator::SwitchInt {
                    arms, otherwise, ..
                } => {
                    targets.extend(arms.iter().map(|(_, b)| *b));
                    targets.push(*otherwise);
                }
                mir::Terminator::Call { target, .. }
                | mir::Terminator::Drop { target, .. }
                | mir::Terminator::Checked { target, .. } => targets.push(*target),
                mir::Terminator::Trap { .. }
                | mir::Terminator::Return
                | mir::Terminator::Unreachable => {}
            }
            for target in targets {
                assert!(
                    (target.0 as usize) < body.blocks.len(),
                    "{} bb{bi}: terminator target bb{} out of bounds",
                    body.instance.symbol,
                    target.0
                );
            }
        }
        // No Param/Infer types can exist by construction (MirTy has no such variants) —
        // monomorphised-only holds structurally (V-TY-2).
    }
}

fn lower_ok(front: &Front) -> mir::MirProgram {
    match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(e) => panic!("lowering failed: {} @ {:?}", e.what, e.span),
    }
}

/// Frozen-corpus scalar cases that must lower. (Cases using methods, strings, `?`, casts, or
/// collections are C4.5 scope and are covered by the unsupported test below.)
const LOWERABLE_CORPUS: &[&str] = &[
    "expr_stmt__01_arithmetic_and_precedence",
    "expr_stmt__03_loops_break_continue",
    "primitive__01_integer_widths_and_overflow_traps",
    "primitive__02_integer_overflow_traps",
    "struct_enum_trait__01_struct_construction_and_methods",
    "struct_enum_trait__02_enum_and_pattern_match",
];

#[test]
fn scalar_corpus_cases_lower_with_contract_invariants() {
    for name in LOWERABLE_CORPUS {
        let front = front_end_corpus(name);
        let program = lower_ok(&front);
        assert_structural_invariants(&program);
        assert!(
            program
                .bodies
                .iter()
                .any(|b| b.instance.symbol.starts_with("main@")),
            "{name}: lowered program must contain main"
        );
    }
}

#[test]
fn dump_is_deterministic_and_versioned() {
    let front = front_end_corpus("expr_stmt__03_loops_break_continue");
    let first = lower_ok(&front).dump();
    let second = lower_ok(&front).dump();
    assert_eq!(first, second, "dump must be deterministic across runs");
    assert!(
        first.starts_with(&format!(
            "// STARK MIR v{} (runtime-surface {})\n",
            mir::MIR_VERSION,
            mir::MIR_RUNTIME_SURFACE
        )),
        "dump must carry the MIR version + runtime-surface header"
    );
}

/// Golden end-to-end mini-dump: pins the concrete shape reviewers approved (Checked terminator
/// with trap category, runtime println call, return-place convention). Deliberately tiny so
/// the golden stays reviewable; broad coverage is structural, not golden-based.
#[test]
fn golden_mini_dump() {
    let front = front_end_src("mini.stark", "fn main() { println(2 + 3); }".to_string());
    let program = lower_ok(&front);
    let dump = program.dump();
    // Note the real shapes this pins: integer literals default to Int32 (Core inference), the
    // Add is a Checked terminator on Int32, and println's Int64 runtime signature forces an
    // explicit (infallible, still Checked) widening Cast -- uniform checked casts per contract.
    let expected = "\
// STARK MIR v0.1 (runtime-surface 0.1-A1)

fn main@[] {
  locals: _0: Unit [ret], _1: Unit [tmp], _2: Int32 [tmp], _3: Int64 [tmp]
  bb0:
    _2 = checked Add(const 2Int32, const 3Int32) -> bb1 trap:IntegerOverflow  // mini.stark:1:21
  bb1:
    _3 = checked Cast(copy _2) -> bb2 trap:CastFailure  // mini.stark:1:13
  bb2:
    _1 = call runtime:PrintlnInt64(copy _3) -> bb3  // mini.stark:1:13
  bb3:
    _0 = const ()  // mini.stark:1:11 synthetic:ReturnSlot
    return  // mini.stark:1:11 synthetic:ReturnSlot
  bb4:
    unreachable  // mini.stark:1:11 synthetic:ReturnSlot
}
";
    assert_eq!(dump, expected, "golden mini-dump changed:\n{dump}");
}

/// Function values and indirect calls (CD-021 items 16/17) lower to FnPtr constants and
/// FnValue callees.
#[test]
fn function_values_lower_to_fnptr_and_indirect_calls() {
    let front = front_end_src(
        "fnval.stark",
        "fn double(x: Int32) -> Int32 { x * 2 } \
         fn main() { let f: fn(Int32) -> Int32 = double; println(f(21)); }"
            .to_string(),
    );
    let program = lower_ok(&front);
    assert_structural_invariants(&program);
    let dump = program.dump();
    assert!(
        dump.contains("const fnptr double@[]"),
        "expected a FnPtr constant in:\n{dump}"
    );
    assert!(
        dump.contains("call fnvalue("),
        "expected an indirect FnValue call in:\n{dump}"
    );
    assert!(
        program
            .bodies
            .iter()
            .any(|b| b.instance.symbol == "double@[]"),
        "instance discovery must include the fn-value target"
    );
}

/// Option/Result lower as logical enums (CD-028 required change #2): construction is an
/// EnumVariant aggregate and matching goes through Discriminant + SwitchInt — no runtime call.
#[test]
fn option_lowers_as_logical_enum() {
    let front = front_end_src(
        "opt.stark",
        "fn pick(flag: Bool) -> Option<Int32> { if flag { Some(7) } else { None } } \
         fn main() { \
             match pick(true) { \
                 Some(v) => println(v), \
                 None => println(0), \
             } \
         }"
        .to_string(),
    );
    let program = lower_ok(&front);
    assert_structural_invariants(&program);
    let dump = program.dump();
    assert!(
        dump.contains("aggregate Option::v1") && dump.contains("aggregate Option::v0"),
        "Option construction must be EnumVariant aggregates:\n{dump}"
    );
    assert!(
        dump.contains("discriminant("),
        "Option matching must read a discriminant:\n{dump}"
    );
}

/// Out-of-subset constructs are clean Unsupported errors naming the owning follow-up — never
/// silent mislowering.
#[test]
fn unsupported_constructs_report_cleanly() {
    let cases = [
        // C4.5c lowers generic fns; comparisons on user nominal types still dispatch through
        // the user's Eq/Ord impl, which C4.5e owns — the guard must reject, never emit a
        // structural BinOp that would silently diverge from the oracle.
        (
            "user_eq.stark",
            "struct P { x: Int32 } \
             impl Eq for P { fn eq(&self, other: &P) -> Bool { self.x == other.x } } \
             fn main() { let a = P { x: 1 }; let b = P { x: 1 }; println(a == b); }",
            "C4.5e",
        ),
        (
            // Vec is a later C4.5e sub-slice (e-2); strings (e-1) now lower.
            "vec.stark",
            "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); println(v.len()); }",
            "C4.5",
        ),
        (
            "tryop.stark",
            "fn half(n: Int32) -> Option<Int32> { \
                 if n % 2 == 0 { Some(n / 2) } else { None } \
             } \
             fn chain(n: Int32) -> Option<Int32> { \
                 let h = half(n)?; \
                 Some(h + 1) \
             } \
             fn main() { \
                 match chain(10) { \
                     Some(v) => println(v), \
                     None => println(0), \
                 } \
             }",
            "C4.5",
        ),
    ];
    for (name, src, needle) in cases {
        let front = front_end_src(name, src.to_string());
        match lower_program(&front.hir, &front.tables, front.file.clone()) {
            Ok(_) => panic!("{name}: expected Unsupported, but lowering succeeded"),
            Err(e) => assert!(
                e.what.contains(needle),
                "{name}: unsupported reason should mention {needle:?}, got: {}",
                e.what
            ),
        }
    }
}

/// WP-C4.5c: generic fns lower once per concrete instantiation with deterministic, injective
/// symbols; identical instantiations reached through multiple call sites deduplicate to one
/// body.
#[test]
fn generic_instances_deduplicate_with_deterministic_symbols() {
    let src = "fn id<T>(x: T) -> T { x } \
               fn main() { println(id(1)); println(id(2)); println(id(true)); }";
    let front = front_end_src("mono.stark", src.to_string());
    let program = lower_ok(&front);
    assert_structural_invariants(&program);
    let symbols: Vec<&str> = program
        .bodies
        .iter()
        .map(|b| b.instance.symbol.as_str())
        .collect();
    assert_eq!(
        symbols,
        vec!["id@[Bool]", "id@[Int32]", "main@[]"],
        "one body per instantiation, sorted by canonical symbol"
    );
    // The Int32 instance carries its type argument on the Instance record too.
    let int_instance = &program.bodies[1].instance;
    assert_eq!(int_instance.type_args, vec![mir::MirTy::Int32]);
    // Determinism: an independent second lowering produces an identical dump.
    let again = lower_ok(&front_end_src("mono.stark", src.to_string()));
    assert_eq!(program.dump(), again.dump());
}

/// WP-C4.5c: a generic nominal instantiation reachable from the bodies gets a type-context
/// entry keyed by its concrete type arguments, so the verifier can resolve its projections.
#[test]
fn generic_struct_instantiations_register_in_type_context() {
    let src = "struct Pair<T> { a: T, b: T } \
               fn main() { \
                   let p = Pair { a: 1, b: 2 }; \
                   let q = Pair { a: true, b: false }; \
                   println(p.a); \
                   if q.b { println(1); } \
               }";
    let front = front_end_src("genstruct.stark", src.to_string());
    let program = lower_ok(&front);
    let entries: Vec<&(u32, Vec<mir::MirTy>)> = program.types.struct_fields.keys().collect();
    let int_entry = entries
        .iter()
        .find(|(_, args)| args == &vec![mir::MirTy::Int32]);
    let bool_entry = entries
        .iter()
        .find(|(_, args)| args == &vec![mir::MirTy::Bool]);
    assert!(
        int_entry.is_some() && bool_entry.is_some(),
        "expected Pair<Int32> and Pair<Bool> context entries, got keys: {entries:?}"
    );
    // And the verifier accepts the program end to end.
    starkc::mir::verify::verify_program(&program).expect("verifier accepts generic nominals");
}

/// WP-C4.5c: polymorphic recursion cannot be monomorphised; it must fail through the named
/// compiler-resource limit — deterministically, never by memory exhaustion or stack overflow.
#[test]
fn polymorphic_recursion_trips_the_named_instance_limit() {
    let src = "fn f<T>(x: T) { f((x,)); } fn main() { f(0); }";
    let front = front_end_src("polyrec.stark", src.to_string());
    match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(_) => panic!("polymorphic recursion must not lower"),
        Err(e) => assert!(
            e.what.contains("LIMIT-MIR-MONO-INSTANCES"),
            "limit failure must name the resource limit, got: {}",
            e.what
        ),
    }
}

/// WP-C4.5d: drop elaboration structure — DropFlag locals exist for droppable locals, Drop
/// terminators are emitted, the dtor instance is discovered and lowered, and the dtor
/// symbol is registered in the type context for glue dispatch.
#[test]
fn drop_elaboration_emits_flags_drops_and_dtor_instances() {
    let src = "struct Loud { id: Int32 } \
               impl Drop for Loud { fn drop(&mut self) { println(self.id); } } \
               fn main() { let a = Loud { id: 1 }; println(0); }";
    let front = front_end_src("dropstruct.stark", src.to_string());
    let program = lower_ok(&front);
    assert_structural_invariants(&program);
    starkc::mir::verify::verify_program(&program).expect("drop elaboration verifies");
    let main = program
        .bodies
        .iter()
        .find(|b| b.instance.symbol == "main@[]")
        .expect("main body");
    assert!(
        main.locals
            .iter()
            .any(|d| matches!(d.kind, mir::LocalKind::DropFlag)),
        "droppable local must get a DropFlag"
    );
    let has_drop_terminator = main
        .blocks
        .iter()
        .any(|b| matches!(b.terminator.0, mir::Terminator::Drop { .. }));
    assert!(
        has_drop_terminator,
        "scope exit must emit a Drop terminator"
    );
    assert!(
        program
            .bodies
            .iter()
            .any(|b| b.instance.symbol == "Loud::Drop::drop@[]"),
        "the dtor instance must be discovered and lowered; symbols: {:?}",
        program
            .bodies
            .iter()
            .map(|b| &b.instance.symbol)
            .collect::<Vec<_>>()
    );
    assert!(
        program
            .types
            .drop_impls
            .values()
            .any(|s| s == "Loud::Drop::drop@[]"),
        "the dtor symbol must be registered for glue dispatch"
    );
}

/// WP-C4.5d boundary: an owned Drop-bearing match scrutinee needs partial-drop of the
/// unbound remainder — clean Unsupported until a later increment, never mislowered.
#[test]
fn match_on_droppable_scrutinee_reports_cleanly() {
    let src = "struct Loud { id: Int32 } \
               impl Drop for Loud { fn drop(&mut self) { println(self.id); } } \
               enum Holder { Empty, Full(Loud) } \
               fn main() { \
                   let h = Holder::Full(Loud { id: 1 }); \
                   match h { \
                       Holder::Full(v) => println(1), \
                       Holder::Empty => println(0), \
                   } \
               }";
    let front = front_end_src("dropmatch.stark", src.to_string());
    match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(_) => panic!("match on droppable scrutinee must be Unsupported"),
        Err(e) => assert!(
            e.what.contains("C4.5"),
            "unsupported reason should name the owner, got: {}",
            e.what
        ),
    }
}

/// WP-C4.5e-1 (A1): a string literal lowers to `const "…"` with escapes round-tripped through
/// the dump, and the dump carries the runtime-surface header.
#[test]
fn string_literal_dump_round_trips_escapes() {
    let src = "fn main() { println(\"a\\\"b\\\\c\\n\"); }";
    let front = front_end_src("strdump.stark", src.to_string());
    let program = lower_ok(&front);
    let dump = program.dump();
    assert!(
        dump.contains("(runtime-surface 0.1-A1)"),
        "dump header must carry the A1 runtime surface, got:\n{dump}"
    );
    assert!(
        dump.contains("const \"a\\\"b\\\\c\\n\""),
        "string literal must render with round-tripped escapes, got:\n{dump}"
    );
}
