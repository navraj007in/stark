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
        "struct_enum_trait__01_struct_construction_and_methods",
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
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
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
        Ok(_) => panic!("expected verifier rejection with {code}, got clean pass"),
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

#[test]
fn rejects_invalid_core_ordering_variant() {
    // A2 (CE3): CoreOrdering has exactly three fieldless variants (0/1/2); a v3 aggregate is an
    // invalid variant and must fail verification (MIR-0008).
    let b = body(
        vec![
            ret_local(),
            local(MirTy::Enum(EnumRef::CoreOrdering, Vec::new())),
        ],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(1)),
                Rvalue::Aggregate(AggKind::EnumVariant(EnumRef::CoreOrdering, 3), Vec::new()),
            )],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0008");
}

#[test]
fn accepts_valid_core_ordering_variants() {
    // The three legal Ordering variants (Less=0, Equal=1, Greater=2) verify cleanly.
    for v in 0..=2u32 {
        let b = body(
            vec![
                ret_local(),
                local(MirTy::Enum(EnumRef::CoreOrdering, Vec::new())),
            ],
            vec![block(
                vec![Statement::Assign(
                    Place::local(LocalId(1)),
                    Rvalue::Aggregate(AggKind::EnumVariant(EnumRef::CoreOrdering, v), Vec::new()),
                )],
                Terminator::Return,
            )],
        );
        assert!(
            verify_program(&program_with(vec![b])).is_ok(),
            "Ordering variant v{v} should verify"
        );
    }
}

#[test]
fn rejects_index_proof_bound_to_a_different_base() {
    // The CE3 same-base rule: a proof produced by CheckIndex on _1 may not index _2.
    use mir::{CheckedOp, TrapCategory, TrapInfo};
    let arr = MirTy::Array(Box::new(MirTy::Int32), 3);
    let b = MirBody {
        instance: Instance {
            item: starkc::hir::ItemId(0),
            type_args: Vec::new(),
            symbol: "test@[]".to_string(),
        },
        params: Vec::new(),
        ret: MirTy::Unit,
        locals: vec![
            ret_local(),
            local(arr.clone()), // _1: the checked base
            local(arr),         // _2: a DIFFERENT array
            LocalDecl {
                ty: MirTy::Int64,
                kind: LocalKind::IndexProof,
            }, // _3: proof bound to _1
            local(MirTy::Int32), // _4: read dest
        ],
        blocks: vec![
            BasicBlock {
                statements: Vec::new(),
                terminator: (
                    Terminator::Checked {
                        op: CheckedOp::CheckIndex,
                        args: vec![
                            Operand::Copy(Place::local(LocalId(1))),
                            Operand::Const(Constant::Int(0, MirTy::Int64)),
                        ],
                        dest: LocalId(3),
                        target: BlockId(1),
                        trap: TrapInfo {
                            category: TrapCategory::IndexOutOfBounds,
                            source: info(),
                        },
                    },
                    info(),
                ),
            },
            block(
                vec![Statement::Assign(
                    Place::local(LocalId(4)),
                    Rvalue::Use(Operand::Copy(Place {
                        local: LocalId(2), // WRONG base: proof binds _1
                        projection: vec![Projection::Index(LocalId(3))],
                    })),
                )],
                Terminator::Return,
            ),
        ],
        entry: BlockId(0),
    };
    expect_code(&program_with(vec![b]), "MIR-0010");
}

/// WP-C4.5d (V-DROP-2 read half): a drop flag may be read only as a SwitchInt scrutinee —
/// any other read (here: a call argument) is rejected.
#[test]
fn rejects_drop_flag_read_outside_switchint() {
    let b = body(
        vec![
            ret_local(),
            LocalDecl {
                ty: MirTy::Bool,
                kind: LocalKind::DropFlag,
            },
            local(MirTy::Unit),
        ],
        vec![
            block(
                vec![Statement::Assign(
                    Place::local(LocalId(1)),
                    Rvalue::Use(Operand::Const(Constant::Bool(true))),
                )],
                Terminator::Call {
                    callee: Callee::Runtime(RuntimeFn::PrintlnBool),
                    args: vec![Operand::Copy(Place::local(LocalId(1)))],
                    dest: Place::local(LocalId(2)),
                    target: BlockId(1),
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    expect_code(&program_with(vec![b]), "MIR-0009");
}

/// WP-C4.5d (V-MOVE-1 field precision): moving one field leaves sibling fields readable —
/// the partial-move MIR the drop elaboration emits must verify clean.
#[test]
fn partial_move_of_one_field_leaves_sibling_readable() {
    let item = starkc::hir::ItemId(0);
    let struct_ty = MirTy::Struct(item, Vec::new());
    let b = body(
        vec![
            ret_local(),
            local(struct_ty.clone()),
            local(MirTy::Int32),
            local(MirTy::Int32),
        ],
        vec![block(
            vec![
                Statement::Assign(
                    Place::local(LocalId(1)),
                    Rvalue::Aggregate(
                        starkc::mir::AggKind::Struct(item),
                        vec![
                            Operand::Const(Constant::Int(1, MirTy::Int32)),
                            Operand::Const(Constant::Int(2, MirTy::Int32)),
                        ],
                    ),
                ),
                // move .0 out, then read .1 — legal with field precision.
                Statement::Assign(
                    Place::local(LocalId(2)),
                    Rvalue::Use(Operand::Move(Place {
                        local: LocalId(1),
                        projection: vec![Projection::Field(0)],
                    })),
                ),
                Statement::Assign(
                    Place::local(LocalId(3)),
                    Rvalue::Use(Operand::Copy(Place {
                        local: LocalId(1),
                        projection: vec![Projection::Field(1)],
                    })),
                ),
            ],
            Terminator::Return,
        )],
    );
    let mut program = program_with(vec![b]);
    program
        .types
        .struct_fields
        .insert((0, Vec::new()), vec![MirTy::Int32, MirTy::Int32]);
    if let Err(errors) = verify_program(&program) {
        panic!("partial move should verify clean, got:\n{errors:#?}");
    }
}

/// WP-C4.5d (V-MOVE-1 field precision): reading the moved field itself — or the whole
/// containing value — is still rejected.
#[test]
fn rejects_read_of_moved_field_and_moved_container() {
    let item = starkc::hir::ItemId(0);
    let struct_ty = MirTy::Struct(item, Vec::new());
    let b = body(
        vec![
            ret_local(),
            local(struct_ty.clone()),
            local(MirTy::Int32),
            local(MirTy::Int32),
        ],
        vec![block(
            vec![
                Statement::Assign(
                    Place::local(LocalId(1)),
                    Rvalue::Aggregate(
                        starkc::mir::AggKind::Struct(item),
                        vec![
                            Operand::Const(Constant::Int(1, MirTy::Int32)),
                            Operand::Const(Constant::Int(2, MirTy::Int32)),
                        ],
                    ),
                ),
                Statement::Assign(
                    Place::local(LocalId(2)),
                    Rvalue::Use(Operand::Move(Place {
                        local: LocalId(1),
                        projection: vec![Projection::Field(0)],
                    })),
                ),
                // read of the moved field: conflict.
                Statement::Assign(
                    Place::local(LocalId(3)),
                    Rvalue::Use(Operand::Copy(Place {
                        local: LocalId(1),
                        projection: vec![Projection::Field(0)],
                    })),
                ),
            ],
            Terminator::Return,
        )],
    );
    let mut program = program_with(vec![b]);
    program
        .types
        .struct_fields
        .insert((0, Vec::new()), vec![MirTy::Int32, MirTy::Int32]);
    expect_code(&program, "MIR-0007");
}

// ---- WP-C4.5e-0: IndexProof definite-initialization (review Finding 1) ----

fn proof_flow_body(blocks: Vec<BasicBlock>, extra_locals: Vec<LocalDecl>) -> MirBody {
    use starkc::mir::LocalKind;
    let mut locals = vec![
        ret_local(),
        local(MirTy::Array(Box::new(MirTy::Int32), 3)), // _1: base array
        LocalDecl {
            ty: MirTy::Int64,
            kind: LocalKind::IndexProof,
        }, // _2: proof
        local(MirTy::Int32),                            // _3: read dest
    ];
    locals.extend(extra_locals);
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

fn check_index_terminator(target: BlockId) -> Terminator {
    use mir::{CheckedOp, TrapCategory, TrapInfo};
    Terminator::Checked {
        op: CheckedOp::CheckIndex,
        args: vec![
            Operand::Copy(Place::local(LocalId(1))),
            Operand::Const(Constant::Int(0, MirTy::Int64)),
        ],
        dest: LocalId(2),
        target,
        trap: TrapInfo {
            category: TrapCategory::IndexOutOfBounds,
            source: info(),
        },
    }
}

fn indexed_read() -> Statement {
    Statement::Assign(
        Place::local(LocalId(3)),
        Rvalue::Use(Operand::Copy(Place {
            local: LocalId(1),
            projection: vec![Projection::Index(LocalId(2))],
        })),
    )
}

/// Use precedes the defining CheckIndex within the very block that defines it (the
/// definition is the terminator; statements run first).
#[test]
fn rejects_index_proof_use_before_its_check() {
    let b = proof_flow_body(
        vec![
            block(vec![indexed_read()], check_index_terminator(BlockId(1))),
            block(vec![], Terminator::Return),
        ],
        vec![],
    );
    expect_code(&program_with(vec![b]), "MIR-0010");
}

/// The check runs on only one branch of a join; the use after the join is not definitely
/// preceded by it (the review's exact malformed-MIR example).
#[test]
fn rejects_index_proof_defined_on_one_branch_only() {
    let b = proof_flow_body(
        vec![
            // bb0: branch
            block(
                vec![],
                Terminator::SwitchInt {
                    scrut: Operand::Copy(Place::local(LocalId(4))),
                    arms: vec![(1, BlockId(1))],
                    otherwise: BlockId(2),
                },
            ),
            // bb1: the only path with the check
            block(vec![], check_index_terminator(BlockId(3))),
            // bb2: skips the check
            block(vec![], Terminator::Goto { target: BlockId(3) }),
            // bb3: join, then use
            block(vec![indexed_read()], Terminator::Return),
        ],
        vec![local(MirTy::Bool)], // _4: condition
    );
    expect_code(&program_with(vec![b]), "MIR-0010");
}

/// Two CheckIndex sites defining the same proof local: one token must witness exactly one
/// check.
#[test]
fn rejects_index_proof_with_two_definition_sites() {
    let b = proof_flow_body(
        vec![
            block(vec![], check_index_terminator(BlockId(1))),
            block(vec![], check_index_terminator(BlockId(2))),
            block(vec![indexed_read()], Terminator::Return),
        ],
        vec![],
    );
    expect_code(&program_with(vec![b]), "MIR-0010");
}

/// A skip edge reaches the use block without passing the check (non-dominated use through
/// a longer defining path).
#[test]
fn rejects_index_proof_use_reachable_by_a_skip_edge() {
    let b = proof_flow_body(
        vec![
            // bb0: either into the checking chain or straight to the use
            block(
                vec![],
                Terminator::SwitchInt {
                    scrut: Operand::Copy(Place::local(LocalId(4))),
                    arms: vec![(1, BlockId(1))],
                    otherwise: BlockId(3),
                },
            ),
            // bb1 → bb2: the checking chain
            block(vec![], check_index_terminator(BlockId(2))),
            block(vec![], Terminator::Goto { target: BlockId(3) }),
            // bb3: use
            block(vec![indexed_read()], Terminator::Return),
        ],
        vec![local(MirTy::Bool)], // _4: condition
    );
    expect_code(&program_with(vec![b]), "MIR-0010");
}

/// WP-C4.5e-0 (V-REF-1): assignment through a `&T` deref is structurally invalid MIR —
/// mutation requires the dereferenced layer to be `&mut`, independent of upstream borrowck.
#[test]
fn rejects_write_through_shared_reference() {
    let shared_ref = MirTy::Ref {
        mutable: false,
        inner: Box::new(MirTy::Int32),
    };
    let b = body(
        vec![ret_local(), local(MirTy::Int32), local(shared_ref)],
        vec![block(
            vec![
                Statement::Assign(
                    Place::local(LocalId(1)),
                    Rvalue::Use(Operand::Const(Constant::Int(1, MirTy::Int32))),
                ),
                Statement::Assign(
                    Place::local(LocalId(2)),
                    Rvalue::RefOf {
                        mutable: false,
                        place: Place::local(LocalId(1)),
                    },
                ),
                // *shared = 2 — must be rejected.
                Statement::Assign(
                    Place {
                        local: LocalId(2),
                        projection: vec![Projection::Deref],
                    },
                    Rvalue::Use(Operand::Const(Constant::Int(2, MirTy::Int32))),
                ),
            ],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0014");
}

// ---- WP-C4.5e-1: A1 string/surface verifier rules ----

/// A1 V-STR-2 (MIR-0015): a structural BinOp on `String`/`str` operands is invalid — string
/// comparison must route through StrEq/StrCmp.
#[test]
fn rejects_binop_on_string_operands() {
    let b = body(
        vec![ret_local(), local(MirTy::String), local(MirTy::Bool)],
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
    expect_code(&program_with(vec![b]), "MIR-0015");
}

/// A1 (MIR-0015): a `Trap` message operand must be `&str`.
#[test]
fn rejects_non_str_trap_message() {
    use mir::{TrapCategory, TrapInfo};
    let b = body(
        vec![ret_local(), local(MirTy::Int32)],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(1)),
                Rvalue::Use(Operand::Const(Constant::Int(7, MirTy::Int32))),
            )],
            Terminator::Trap {
                info: TrapInfo {
                    category: TrapCategory::Panic,
                    source: info(),
                },
                message: Some(Operand::Copy(Place::local(LocalId(1)))),
            },
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0015");
}

/// A1 V-SURFACE-1 (MIR-0017): a program stamped with an unsupported runtime surface is
/// rejected before any body is verified.
#[test]
fn rejects_unsupported_runtime_surface() {
    let b = body(vec![ret_local()], vec![block(vec![], Terminator::Return)]);
    let mut program = program_with(vec![b]);
    program.runtime_surface = "0.1-A999".to_string();
    expect_code(&program, "MIR-0017");
}

// ---- WP-C4.5e-2: A1 Vec V-COPY-1 verifier rules ----

fn vec_ty(elem: MirTy) -> MirTy {
    MirTy::Core(starkc::hir::CoreType::Vec, vec![elem])
}

/// A1 V-COPY-1 (MIR-0016): `VecIndexGet` requires a Copy element type; a non-Copy element
/// (here a user struct with no `impl Copy`) is rejected.
#[test]
fn rejects_vec_index_get_on_non_copy_element() {
    let item = starkc::hir::ItemId(0);
    let struct_ty = MirTy::Struct(item, Vec::new());
    let vref = MirTy::Ref {
        mutable: false,
        inner: Box::new(vec_ty(struct_ty.clone())),
    };
    let b = body(
        vec![
            ret_local(),
            local(vec_ty(struct_ty.clone())),
            local(vref.clone()),
            local(struct_ty.clone()),
        ],
        vec![
            block(
                vec![Statement::Assign(
                    Place::local(LocalId(2)),
                    Rvalue::RefOf {
                        mutable: false,
                        place: Place::local(LocalId(1)),
                    },
                )],
                Terminator::Call {
                    callee: Callee::Runtime(RuntimeFn::VecIndexGet),
                    args: vec![
                        Operand::Copy(Place::local(LocalId(2))),
                        Operand::Const(Constant::Int(0, MirTy::UInt64)),
                    ],
                    dest: Place::local(LocalId(3)),
                    target: BlockId(1),
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    let mut program = program_with(vec![b]);
    // Struct has no impl Copy (copy_types empty) and no fields → non-Copy, non-droppable.
    program
        .types
        .struct_fields
        .insert((0, Vec::new()), Vec::new());
    expect_code(&program, "MIR-0016");
}

/// A1 V-COPY-1 (MIR-0016): `VecClear` requires a non-droppable element type; a Drop-bearing
/// element must instead lower to a pop-and-drop loop, so a `VecClear` on it is invalid MIR.
#[test]
fn rejects_vec_clear_on_droppable_element() {
    let item = starkc::hir::ItemId(0);
    let struct_ty = MirTy::Struct(item, Vec::new());
    let vref = MirTy::Ref {
        mutable: true,
        inner: Box::new(vec_ty(struct_ty.clone())),
    };
    let b = body(
        vec![
            ret_local(),
            local(vec_ty(struct_ty.clone())),
            local(vref),
            local(MirTy::Unit),
        ],
        vec![
            block(
                vec![Statement::Assign(
                    Place::local(LocalId(2)),
                    Rvalue::RefOf {
                        mutable: true,
                        place: Place::local(LocalId(1)),
                    },
                )],
                Terminator::Call {
                    callee: Callee::Runtime(RuntimeFn::VecClear),
                    args: vec![Operand::Copy(Place::local(LocalId(2)))],
                    dest: Place::local(LocalId(3)),
                    target: BlockId(1),
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    let mut program = program_with(vec![b]);
    // Give the struct a Drop impl so it is droppable.
    program
        .types
        .struct_fields
        .insert((0, Vec::new()), Vec::new());
    program
        .types
        .drop_impls
        .insert((0, Vec::new()), "Loud::Drop::drop@[]".to_string());
    expect_code(&program, "MIR-0016");
}

// ---- WP-C4.5f-1: frame-generation defense (CD-030 Finding 2) ----

/// A verifier-passing but semantically bogus program: a function returns a reference to its
/// own local (the verifier has no escape analysis — by design, borrowck upstream owns that),
/// a second call then reuses the same frame slot, and the stale reference is dereferenced.
/// Pre-C4.5f-1 the interpreter silently read the SECOND call's local (slot aliasing); with
/// frame generations it must fail loudly, never alias.
#[test]
fn stale_reference_to_reused_frame_slot_fails_loudly() {
    use starkc::mir::interp::{run_program, MirRunError};
    use starkc::mir::{Rvalue, Terminator};

    let ref_int = MirTy::Ref {
        mutable: false,
        inner: Box::new(MirTy::Int32),
    };
    // leak@[]: fn() -> &Int32 { let x = 7; &x }
    let leak = MirBody {
        instance: Instance {
            item: starkc::hir::ItemId(1),
            type_args: Vec::new(),
            symbol: "leak@[]".to_string(),
        },
        params: Vec::new(),
        ret: ref_int.clone(),
        locals: vec![
            LocalDecl {
                ty: ref_int.clone(),
                kind: LocalKind::Return,
            },
            local(MirTy::Int32),
        ],
        blocks: vec![block(
            vec![
                Statement::Assign(
                    Place::local(LocalId(1)),
                    Rvalue::Use(Operand::Const(Constant::Int(7, MirTy::Int32))),
                ),
                Statement::Assign(
                    Place::local(LocalId(0)),
                    Rvalue::RefOf {
                        mutable: false,
                        place: Place::local(LocalId(1)),
                    },
                ),
            ],
            Terminator::Return,
        )],
        entry: BlockId(0),
    };
    // clobber@[]: fn() { let y = 99; } — reuses leak's frame slot.
    let clobber = MirBody {
        instance: Instance {
            item: starkc::hir::ItemId(2),
            type_args: Vec::new(),
            symbol: "clobber@[]".to_string(),
        },
        params: Vec::new(),
        ret: MirTy::Unit,
        locals: vec![ret_local(), local(MirTy::Int32)],
        blocks: vec![block(
            vec![
                Statement::Assign(
                    Place::local(LocalId(1)),
                    Rvalue::Use(Operand::Const(Constant::Int(99, MirTy::Int32))),
                ),
                Statement::Assign(
                    Place::local(LocalId(0)),
                    Rvalue::Use(Operand::Const(Constant::Unit)),
                ),
            ],
            Terminator::Return,
        )],
        entry: BlockId(0),
    };
    // main@[]: r = leak(); clobber(); read *r.
    let main = MirBody {
        instance: Instance {
            item: starkc::hir::ItemId(0),
            type_args: Vec::new(),
            symbol: "main@[]".to_string(),
        },
        params: Vec::new(),
        ret: MirTy::Unit,
        locals: vec![
            ret_local(),
            local(ref_int.clone()),
            local(MirTy::Unit),
            local(MirTy::Int32),
        ],
        blocks: vec![
            block(
                vec![],
                Terminator::Call {
                    callee: Callee::Instance(Instance {
                        item: starkc::hir::ItemId(1),
                        type_args: Vec::new(),
                        symbol: "leak@[]".to_string(),
                    }),
                    args: vec![],
                    dest: Place::local(LocalId(1)),
                    target: BlockId(1),
                },
            ),
            block(
                vec![],
                Terminator::Call {
                    callee: Callee::Instance(Instance {
                        item: starkc::hir::ItemId(2),
                        type_args: Vec::new(),
                        symbol: "clobber@[]".to_string(),
                    }),
                    args: vec![],
                    dest: Place::local(LocalId(2)),
                    target: BlockId(2),
                },
            ),
            block(
                vec![
                    Statement::Assign(
                        Place::local(LocalId(3)),
                        Rvalue::Use(Operand::Copy(Place {
                            local: LocalId(1),
                            projection: vec![Projection::Deref],
                        })),
                    ),
                    Statement::Assign(
                        Place::local(LocalId(0)),
                        Rvalue::Use(Operand::Const(Constant::Unit)),
                    ),
                ],
                Terminator::Return,
            ),
        ],
        entry: BlockId(0),
    };
    let program = program_with(vec![main, leak, clobber]);
    let verified = verify_program(&program).expect("no escape analysis at MIR level — verifies");
    match run_program(verified) {
        Ok(exec) => panic!(
            "stale reference silently aliased a reused frame (stdout {:?})",
            exec.output
        ),
        Err(failure) => match failure.error {
            MirRunError::Internal(message) => assert!(
                message.contains("reused") || message.contains("dangling"),
                "expected a dangling/reused-frame error, got: {message}"
            ),
            other => panic!("expected a loud Internal error, got {other:?}"),
        },
    }
}

#[test]
fn rejects_slice_new_with_bad_dest_type() {
    // 0.1-A6: SliceNew's dest must be &[T] with T matching the receiver's element type.
    use mir::RuntimeFn;
    let arr = MirTy::Array(Box::new(MirTy::Int32), 3);
    let arr_ref = MirTy::Ref {
        mutable: false,
        inner: Box::new(arr.clone()),
    };
    let wrong_slice = MirTy::Ref {
        mutable: false,
        inner: Box::new(MirTy::Slice(Box::new(MirTy::Bool))),
    };
    let b = body(
        vec![
            ret_local(),
            local(arr),
            local(arr_ref),
            local(wrong_slice),
            local(MirTy::Unit),
        ],
        vec![
            BasicBlock {
                statements: vec![(
                    Statement::Assign(
                        Place::local(LocalId(2)),
                        Rvalue::RefOf {
                            mutable: false,
                            place: Place::local(LocalId(1)),
                        },
                    ),
                    info(),
                )],
                terminator: (
                    Terminator::Call {
                        callee: Callee::Runtime(RuntimeFn::SliceNew),
                        args: vec![
                            Operand::Copy(Place::local(LocalId(2))),
                            Operand::Const(Constant::Int(0, MirTy::Int32)),
                            Operand::Const(Constant::Int(2, MirTy::Int32)),
                            Operand::Const(Constant::Bool(false)),
                        ],
                        dest: Place::local(LocalId(3)),
                        target: BlockId(1),
                    },
                    info(),
                ),
            },
            block(vec![], Terminator::Return),
        ],
    );
    expect_code(&program_with(vec![b]), "MIR-0005");
}

#[test]
fn rejects_slice_len_on_non_slice_receiver() {
    use mir::RuntimeFn;
    let b = body(
        vec![ret_local(), local(MirTy::Int32), local(MirTy::UInt64)],
        vec![
            block(
                vec![],
                Terminator::Call {
                    callee: Callee::Runtime(RuntimeFn::SliceLen),
                    args: vec![Operand::Copy(Place::local(LocalId(1)))],
                    dest: Place::local(LocalId(2)),
                    target: BlockId(1),
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    expect_code(&program_with(vec![b]), "MIR-0012");
}

// ------------------------------------------- WP-C4.7-2: Class-A verifier negatives --
//
// CD-033's evidence rule: every Class-A class carries hand-built negatives proving the verifier
// rejects malformed MIR of that class loudly, with the contract's allocated code — the lowering
// and the verifier stay two independent readings of mir.md. These complement the differential
// tests (which only ever see MIR the lowering actually produced).

#[test]
fn rejects_bitwise_binop_on_floats() {
    // A5/amendment A3: bitwise ops are INTEGER-only. A float operand is MIR-0004 — the guard
    // that keeps `&`/`|`/`^` from being read as a generic bit-pattern operation on any type.
    let b = body(
        vec![
            ret_local(),
            local(MirTy::Float64),
            local(MirTy::Float64),
            local(MirTy::Float64),
        ],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(3)),
                Rvalue::BinOp(
                    mir::MirBinOp::BitAnd,
                    Operand::Copy(Place::local(LocalId(1))),
                    Operand::Copy(Place::local(LocalId(2))),
                ),
            )],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0004");
}

#[test]
fn rejects_pow_on_non_integer_dest() {
    // A5/amendment A3: `Pow` is integer exponentiation (NUM-INT-ARITH-001). A Float64 dest must
    // not silently become a float power operation, which has different trapping behavior.
    use mir::CheckedOp;
    let b = body(
        vec![ret_local(), local(MirTy::Float64)],
        vec![
            block(
                vec![],
                Terminator::Checked {
                    op: CheckedOp::Pow,
                    args: vec![
                        Operand::Const(Constant::Float(2.0, MirTy::Float64)),
                        Operand::Const(Constant::Float(3.0, MirTy::Float64)),
                    ],
                    dest: LocalId(1),
                    target: BlockId(1),
                    trap: mir::TrapInfo {
                        category: mir::TrapCategory::IntegerOverflow,
                        source: info(),
                    },
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    expect_code(&program_with(vec![b]), "MIR-0004");
}

#[test]
fn rejects_vec_get_ref_with_wrong_dest() {
    // 0.1-A4: `VecGetRef` is schematic in T — its dest is `Option<&T>` for the RECEIVER's T.
    // `Vec<Int32>` receiver with an `Option<&Bool>` dest is MIR-0005; the schematic signature
    // must not degrade to "any Option of any reference".
    let vec_int = MirTy::Core(starkc::hir::CoreType::Vec, vec![MirTy::Int32]);
    let vec_ref = MirTy::Ref {
        mutable: false,
        inner: Box::new(vec_int.clone()),
    };
    let wrong_dest = MirTy::Enum(
        EnumRef::CoreOption,
        vec![MirTy::Ref {
            mutable: false,
            inner: Box::new(MirTy::Bool),
        }],
    );
    let b = body(
        vec![
            ret_local(),
            local(vec_int),
            local(vec_ref),
            local(wrong_dest),
        ],
        vec![
            BasicBlock {
                statements: vec![(
                    Statement::Assign(
                        Place::local(LocalId(2)),
                        Rvalue::RefOf {
                            mutable: false,
                            place: Place::local(LocalId(1)),
                        },
                    ),
                    info(),
                )],
                terminator: (
                    Terminator::Call {
                        callee: Callee::Runtime(RuntimeFn::VecGetRef),
                        args: vec![
                            Operand::Copy(Place::local(LocalId(2))),
                            Operand::Const(Constant::Int(0, MirTy::UInt64)),
                        ],
                        dest: Place::local(LocalId(3)),
                        target: BlockId(1),
                    },
                    info(),
                ),
            },
            block(vec![], Terminator::Return),
        ],
    );
    expect_code(&program_with(vec![b]), "MIR-0005");
}

#[test]
fn rejects_chars_iter_next_on_non_iterator() {
    // 0.1-A5: `CharsIterNext` takes `&mut CharsIter` from the FIXED signature table. An Int32
    // operand is MIR-0005 — the fixed table is checked, not assumed.
    let b = body(
        vec![
            ret_local(),
            local(MirTy::Int32),
            local(MirTy::Enum(EnumRef::CoreOption, vec![MirTy::Char])),
        ],
        vec![
            block(
                vec![],
                Terminator::Call {
                    callee: Callee::Runtime(RuntimeFn::CharsIterNext),
                    args: vec![Operand::Copy(Place::local(LocalId(1)))],
                    dest: Place::local(LocalId(2)),
                    target: BlockId(1),
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    expect_code(&program_with(vec![b]), "MIR-0005");
}

#[test]
fn rejects_runtime_call_arity_mismatch() {
    // The runtime surface is a CLOSED, signed contract: a `CharsIterNext` called with two
    // operands is MIR-0005 on arity, before any per-argument typing. (WP-C4.7-2 checked for an
    // existing `rejects_call_arity_against_instance`; there was none, so the arity path against
    // a runtime signature is pinned here.)
    let b = body(
        vec![
            ret_local(),
            local(MirTy::Enum(EnumRef::CoreOption, vec![MirTy::Char])),
        ],
        vec![
            block(
                vec![],
                Terminator::Call {
                    callee: Callee::Runtime(RuntimeFn::CharsIterNext),
                    args: vec![
                        Operand::Const(Constant::Int(0, MirTy::Int32)),
                        Operand::Const(Constant::Int(1, MirTy::Int32)),
                    ],
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
fn rejects_switch_on_float() {
    // A2 widened `SwitchInt`'s scrutinee to `Char` (literal Char patterns switch on the Unicode
    // scalar). This pins that the widening stopped there: a Float64 scrutinee is MIR-0004. Float
    // equality is not a discriminant test, and a jump table over floats has no meaning.
    let b = body(
        vec![ret_local(), local(MirTy::Float64)],
        vec![
            block(
                vec![],
                Terminator::SwitchInt {
                    scrut: Operand::Copy(Place::local(LocalId(1))),
                    arms: vec![(0, BlockId(1))],
                    otherwise: BlockId(1),
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    expect_code(&program_with(vec![b]), "MIR-0004");
}

#[test]
fn rejects_layout_query_with_non_uint64_dest() {
    // A4 (CD-036): a layout query answers in `UInt64` (06: `fn size_of<T>() -> UInt64`). An
    // Int32 dest is MIR-0004. The QUERIED type is deliberately not constrained — every MirTy is
    // a legal question, so this test pins only the half the verifier owns.
    let b = body(
        vec![ret_local(), local(MirTy::Int32)],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(1)),
                Rvalue::LayoutQuery {
                    kind: mir::LayoutKind::SizeOf,
                    ty: MirTy::Bool,
                },
            )],
            Terminator::Return,
        )],
    );
    expect_code(&program_with(vec![b]), "MIR-0004");
}

#[test]
fn accepts_layout_query_of_any_type_into_uint64() {
    // The complement: an unsized queried type (`str`) is a legal QUESTION at the MIR level —
    // `Sized`-ness is the checked front end's property (amendment A4 §5), not a verifier rule.
    let b = body(
        vec![ret_local(), local(MirTy::UInt64)],
        vec![block(
            vec![Statement::Assign(
                Place::local(LocalId(1)),
                Rvalue::LayoutQuery {
                    kind: mir::LayoutKind::AlignOf,
                    ty: MirTy::Str,
                },
            )],
            Terminator::Return,
        )],
    );
    assert!(
        verify_program(&program_with(vec![b])).is_ok(),
        "a layout query into a UInt64 dest must verify cleanly"
    );
}

#[test]
fn rejects_box_into_inner_on_non_box() {
    // 0.1-A7: `BoxIntoInner(Box<T>) -> T` is schematic in T, read from the argument. An Int32
    // argument has no contained type, so the signature cannot be formed: MIR-0005.
    let b = body(
        vec![ret_local(), local(MirTy::Int32), local(MirTy::Int32)],
        vec![
            block(
                vec![],
                Terminator::Call {
                    callee: Callee::Runtime(RuntimeFn::BoxIntoInner),
                    args: vec![Operand::Const(Constant::Int(1, MirTy::Int32))],
                    dest: Place::local(LocalId(2)),
                    target: BlockId(1),
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    expect_code(&program_with(vec![b]), "MIR-0005");
}

#[test]
fn rejects_box_new_with_mismatched_dest() {
    // 0.1-A7: `BoxNew(T) -> Box<T>` — the destination's contained type must be the argument's
    // type. `BoxNew(Int32)` into a `Box<Bool>` is MIR-0005; the schematic signature must not
    // degrade to "any Box".
    let boxed_bool = MirTy::Core(starkc::hir::CoreType::Box, vec![MirTy::Bool]);
    let b = body(
        vec![ret_local(), local(boxed_bool)],
        vec![
            block(
                vec![],
                Terminator::Call {
                    callee: Callee::Runtime(RuntimeFn::BoxNew),
                    args: vec![Operand::Const(Constant::Int(1, MirTy::Int32))],
                    dest: Place::local(LocalId(1)),
                    target: BlockId(1),
                },
            ),
            block(vec![], Terminator::Return),
        ],
    );
    expect_code(&program_with(vec![b]), "MIR-0005");
}
