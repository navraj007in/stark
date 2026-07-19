//! WP-C4.3 — MIR verifier tests.
//!
//! Positive: every program the WP-C4.2 scalar lowering produces must pass verification — the
//! lowering and the verifier are two independent readings of the same approved contract
//! (mir.md, CD-028), and their agreement over real programs is the evidence both are faithful.
//! Negative: hand-crafted invalid MIR must be rejected with the specific `MIR-xxxx` code the
//! contract's §10 obligation map allocates — the verifier fails safely and loudly, never
//! passing malformed MIR toward a backend.

use starkc::diag::Severity;
use starkc::mir::{self, lower::lower_program, verify::verify_program};
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::PathBuf;
use std::sync::Arc;

// ------------------------------------------------------------------ positive --

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/exec_snapshots")
}

fn lower_source(name: &str, source: String) -> mir::MirProgram {
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
    match lower_program(&hir, &checked.tables, file) {
        Ok(program) => program,
        Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
    }
}

#[test]
fn every_lowerable_program_verifies_clean() {
    let corpus = [
        "expr_stmt__01_arithmetic_and_precedence",
        "expr_stmt__03_loops_break_continue",
        "primitive__01_integer_widths_and_overflow_traps",
        "primitive__02_integer_overflow_traps",
        "struct_enum_trait__02_enum_and_pattern_match",
    ];
    for name in corpus {
        let path = corpus_dir().join(format!("{name}.stark"));
        let source = std::fs::read_to_string(&path).unwrap();
        let program = lower_source(&path.to_string_lossy(), source);
        if let Err(errors) = verify_program(&program) {
            panic!("{name}: verifier rejected lowered MIR:\n{errors:#?}");
        }
    }

    let inline = [
        (
            "fnval.stark",
            "fn double(x: Int32) -> Int32 { x * 2 } \
             fn main() { let f: fn(Int32) -> Int32 = double; println(f(21)); }",
        ),
        (
            "opt.stark",
            "fn pick(flag: Bool) -> Option<Int32> { if flag { Some(7) } else { None } } \
             fn main() { \
                 match pick(true) { \
                     Some(v) => println(v), \
                     None => println(0), \
                 } \
             }",
        ),
        (
            "structs.stark",
            "struct Point { x: Int32, y: Int32 } \
             fn main() { \
                 let p = Point { x: 3, y: 4 }; \
                 println(p.x + p.y); \
             }",
        ),
    ];
    for (name, src) in inline {
        let program = lower_source(name, src.to_string());
        if let Err(errors) = verify_program(&program) {
            panic!("{name}: verifier rejected lowered MIR:\n{errors:#?}");
        }
    }
}

// ------------------------------------------------------------------ negative --

use mir::{
    AggKind, BasicBlock, BlockId, Callee, Constant, EnumRef, FileId, Instance, LocalDecl, LocalId,
    LocalKind, MirBody, MirProgram, MirTy, Operand, Origin, Place, Projection, RuntimeFn, Rvalue,
    SourceInfo, Statement, Terminator, TypeContext,
};
use starkc::source::Span;

fn info() -> SourceInfo {
    SourceInfo {
        file: FileId(0),
        span: Span { lo: 0, hi: 0 },
        origin: Origin::UserCode,
    }
}

fn body(locals: Vec<LocalDecl>, blocks: Vec<BasicBlock>) -> MirBody {
    MirBody {
        instance: Instance {
            item: starkc::hir::ItemId(0),
            type_args: Vec::new(),
            symbol: "test@[]".to_string(),
        },
        params: Vec::new(),
        ret: MirTy::Unit,
        locals,
        blocks,
        entry: BlockId(0),
    }
}

fn program_with(bodies: Vec<MirBody>) -> MirProgram {
    MirProgram {
        files: vec![Arc::new(SourceFile::new("hand.stark", ""))],
        bodies,
        types: TypeContext::default(),
    }
}

fn local(ty: MirTy) -> LocalDecl {
    LocalDecl {
        ty,
        kind: LocalKind::Temp,
    }
}

fn ret_local() -> LocalDecl {
    LocalDecl {
        ty: MirTy::Unit,
        kind: LocalKind::Return,
    }
}

fn block(statements: Vec<Statement>, terminator: Terminator) -> BasicBlock {
    BasicBlock {
        statements: statements.into_iter().map(|s| (s, info())).collect(),
        terminator: (terminator, info()),
    }
}

fn expect_code(program: &MirProgram, code: &str) {
    match verify_program(program) {
        Ok(()) => panic!("expected verifier rejection with {code}, got clean pass"),
        Err(errors) => assert!(
            errors.iter().any(|e| e.code == code),
            "expected {code}, got: {errors:#?}"
        ),
    }
}

#[test]
fn rejects_out_of_bounds_target() {
    let b = body(
        vec![ret_local()],
        vec![block(vec![], Terminator::Goto { target: BlockId(9) })],
    );
    expect_code(&program_with(vec![b]), "MIR-0001");
}

#[test]
fn rejects_out_of_bounds_local() {
    let b = body(
        vec![ret_local()],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(7)),
                Rvalue::Use(Operand::Const(Constant::Unit)),
            )],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0002");
}

#[test]
fn rejects_assignment_type_mismatch() {
    let b = body(
        vec![ret_local(), local(MirTy::Bool)],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(1)),
                Rvalue::Use(Operand::Const(Constant::Int(3, MirTy::Int32))),
            )],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0004");
}

#[test]
fn rejects_runtime_call_signature_mismatch() {
    // PrintlnInt64 called with a Bool argument.
    let b = body(
        vec![ret_local(), local(MirTy::Unit)],
        vec![
            block(
                vec![],
                Terminator::Call {
                    callee: Callee::Runtime(RuntimeFn::PrintlnInt64),
                    args: vec![Operand::Const(Constant::Bool(true))],
                    dest: Place::local(LocalId(1)),
                    target: BlockId(1),
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    expect_code(&program_with(vec![b]), "MIR-0005");
}

#[test]
fn rejects_use_after_move() {
    let b = body(
        vec![
            ret_local(),
            local(MirTy::Int32),
            local(MirTy::Int32),
            local(MirTy::Int32),
        ],
        vec![block(
            vec![
                Statement::Assign(
                    Place::local(LocalId(1)),
                    Rvalue::Use(Operand::Const(Constant::Int(1, MirTy::Int32))),
                ),
                Statement::Assign(
                    Place::local(LocalId(2)),
                    Rvalue::Use(Operand::Move(Place::local(LocalId(1)))),
                ),
                Statement::Assign(
                    Place::local(LocalId(3)),
                    Rvalue::Use(Operand::Move(Place::local(LocalId(1)))),
                ),
            ],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0007");
}

#[test]
fn rejects_discriminant_of_non_enum() {
    let b = body(
        vec![ret_local(), local(MirTy::Int32), local(MirTy::Int64)],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(2)),
                Rvalue::Discriminant(Place::local(LocalId(1))),
            )],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0008");
}

#[test]
fn rejects_drop_flag_written_non_constant() {
    let b = body(
        vec![
            ret_local(),
            LocalDecl {
                ty: MirTy::Bool,
                kind: LocalKind::DropFlag,
            },
            local(MirTy::Bool),
        ],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(1)),
                Rvalue::Use(Operand::Copy(Place::local(LocalId(2)))),
            )],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0009");
}

#[test]
fn rejects_drop_of_undroppable_type() {
    let b = body(
        vec![ret_local(), local(MirTy::Int32)],
        vec![
            block(
                vec![],
                Terminator::Drop {
                    place: Place::local(LocalId(1)),
                    target: BlockId(1),
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    expect_code(&program_with(vec![b]), "MIR-0009");
}

#[test]
fn rejects_index_without_proof_token() {
    // Index projection consuming an ordinary integer local (the CE3-revised rule).
    let b = body(
        vec![
            ret_local(),
            local(MirTy::Array(Box::new(MirTy::Int32), 3)),
            local(MirTy::Int64), // ordinary Temp, not IndexProof
            local(MirTy::Int32),
        ],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(3)),
                Rvalue::Use(Operand::Copy(Place {
                    local: LocalId(1),
                    projection: vec![Projection::Index(LocalId(2))],
                })),
            )],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0010");
}

#[test]
fn rejects_comparison_on_fn_values() {
    // TYPE-FN-001 at the MIR level: Eq on FnPtr operands.
    let fn_ty = MirTy::FnPtr {
        params: vec![],
        ret: Box::new(MirTy::Unit),
    };
    let b = body(
        vec![ret_local(), local(fn_ty), local(MirTy::Bool)],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(2)),
                Rvalue::BinOp(
                    mir::MirBinOp::Eq,
                    Operand::Copy(Place::local(LocalId(1))),
                    Operand::Copy(Place::local(LocalId(1))),
                ),
            )],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0011");
}

#[test]
fn rejects_invalid_file_id() {
    let bad_info = SourceInfo {
        file: FileId(42),
        span: Span { lo: 0, hi: 0 },
        origin: Origin::UserCode,
    };
    let b = MirBody {
        instance: Instance {
            item: starkc::hir::ItemId(0),
            type_args: Vec::new(),
            symbol: "test@[]".to_string(),
        },
        params: Vec::new(),
        ret: MirTy::Unit,
        locals: vec![ret_local()],
        blocks: vec![BasicBlock {
            statements: Vec::new(),
            terminator: (Terminator::Return, bad_info),
        }],
        entry: BlockId(0),
    };
    expect_code(&program_with(vec![b]), "MIR-0013");
}

#[test]
fn rejects_bare_unsized_local() {
    let b = body(
        vec![ret_local(), local(MirTy::Str)],
        vec![block(vec![], Terminator::Return)],
    );
    expect_code(&program_with(vec![b]), "MIR-0006");
}

#[test]
fn rejects_enum_aggregate_arity_mismatch() {
    // Some(x, y) shape: CoreOption v1 with two payload operands.
    let b = body(
        vec![
            ret_local(),
            local(MirTy::Enum(EnumRef::CoreOption, vec![MirTy::Int32])),
        ],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(1)),
                Rvalue::Aggregate(
                    AggKind::EnumVariant(EnumRef::CoreOption, 1),
                    vec![
                        Operand::Const(Constant::Int(1, MirTy::Int32)),
                        Operand::Const(Constant::Int(2, MirTy::Int32)),
                    ],
                ),
            )],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0008");
}
