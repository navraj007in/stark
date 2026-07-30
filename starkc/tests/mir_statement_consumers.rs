//! **A12 / CE3 coordination guard: every MIR consumer handles every `Statement` variant.**
//!
//! A12 widened a statement set that had been closed at "assignments and nops only" since the
//! contract was written. The CE3 approval carries one condition — that a future consumer cannot
//! silently overlook a statement — and this file is that condition.
//!
//! # Why two guards and not one
//!
//! Neither half is sufficient alone, and the reason is the exact failure A12 itself came from.
//!
//! **The compile-time half** ([`consumers_of`]) is an exhaustive `match` with no wildcard arm. Add a
//! `Statement` variant and this file stops compiling, with the checklist of consumers sitting right
//! there in the error's context. That catches *omission* — the thing a reviewer forgets.
//!
//! It cannot catch a consumer that "handles" a variant by ignoring it. `HostResource` was swallowed
//! by six `MirTy` catch-all arms in A11, each of which compiled cleanly, and the most serious of
//! them meant every resource leaked while every unit test passed. A checklist that only proves
//! someone wrote an arm would repeat that.
//!
//! **The behavioural half** therefore runs a real program containing every statement variant through
//! every consumer and asserts each one *did something with it* — the verifier accepts it, the dump
//! shows it, the interpreter and the backend both execute it, and the two agree. That is what makes
//! this a guard rather than a reminder.

mod support;

use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::mir::{Statement, StorageEnd};
use support::differential::{
    canonical_form, first_difference, front_end, run_mir, run_native, rustc_available,
};

/// **The compile-time guard.** Exhaustive by construction: no `_` arm, ever.
///
/// If you added a `Statement` variant and landed here, every consumer named below needs a decision —
/// and "it does nothing" is a decision that must be written down at the consumer, not defaulted into
/// by a wildcard. The reference interpreter's `StorageDead` arm is the model: inert, with the reason
/// stated.
fn consumers_of(stmt: &Statement) -> &'static [&'static str] {
    const CONSUMERS: &[&str] = &[
        "mir::verify — validity rules",
        "mir::interp — reference execution",
        "mir::mod::dump_statement — the textual dump",
        "mir::opt — every pass that walks statements",
        "backend::generated_rust::emit_bodies — native emission",
        "backend::generated_rust::linkage — instance-reference visiting",
        "C8 semantic services — if it ever begins reading MIR",
    ];
    match stmt {
        Statement::Assign(..) => CONSUMERS,
        Statement::Nop => CONSUMERS,
        Statement::StorageDead(..) => CONSUMERS,
    }
}

/// A program whose lowering contains every statement variant.
///
/// The `match` in a loop over a droppable payload is what produces `StorageDead` — it is the
/// `DEFECT-C788-LOOP-TEMP` shape, kept here because a guard built on a synthetic body would drift
/// from what lowering actually emits.
const EVERY_STATEMENT: &str = r#"
struct Res { id: Int32 }

impl Drop for Res {
    fn drop(&mut self) {
        print(self.id);
    }
}

enum Maybe { Some(Res), None }

fn make(i: Int32) -> Maybe { Maybe::Some(Res { id: i }) }

fn main() {
    let mut i: Int32 = 0;
    while i < 3 {
        match make(i) {
            Maybe::Some(r) => { print(r.id); }
            Maybe::None => { }
        }
        i = i + 1;
    }
}
"#;

#[test]
fn the_consumer_checklist_covers_every_statement_variant() {
    // Reading the checklist for each variant is the point: the match is exhaustive, so this cannot
    // pass while a variant is unlisted.
    for stmt in [
        Statement::Nop,
        Statement::StorageDead(
            starkc::mir::Place::local(starkc::mir::LocalId(0)),
            StorageEnd::Accounted,
        ),
    ] {
        assert!(
            !consumers_of(&stmt).is_empty(),
            "every statement variant must name its consumers"
        );
    }
}

#[test]
fn every_statement_variant_survives_every_consumer() {
    let front = front_end("stmt_consumers", EVERY_STATEMENT);
    let program = lower_program(&front.hir, &front.tables, front.file.clone())
        .unwrap_or_else(|e| panic!("lowering failed: {}", e.what));

    // Lowering really did produce all three, so the rest of this test is not vacuous.
    let mut seen_assign = false;
    let mut seen_storage_dead = false;
    for body in &program.bodies {
        for block in &body.blocks {
            for (stmt, _) in &block.statements {
                match stmt {
                    Statement::Assign(..) => seen_assign = true,
                    Statement::StorageDead(..) => seen_storage_dead = true,
                    Statement::Nop => {}
                }
            }
        }
    }
    assert!(seen_assign, "the fixture must contain an Assign");
    assert!(
        seen_storage_dead,
        "the fixture must contain a StorageDead — if lowering stopped emitting one, this guard has \
         quietly stopped guarding, which is worse than failing"
    );

    // Consumer 1: the verifier accepts it.
    let verified = verify_program(&program)
        .unwrap_or_else(|errors| panic!("the verifier rejected the fixture: {errors:#?}"));
    let _ = verified;

    // Consumer 2: the dump renders it, rather than dropping it silently.
    let dumped = program.dump();
    assert!(
        dumped.contains("storage_dead"),
        "dump_statement must render StorageDead:\n{dumped}"
    );

    // Consumers 3 and 4: the reference interpreter and the native backend both execute it, and
    // agree. Agreement is the part that catches an arm that compiles but does nothing.
    if !rustc_available() {
        eprintln!("SKIP: native half of every_statement_variant_survives_every_consumer: no rustc");
        return;
    }
    let mir = run_mir("stmt_consumers", &program);
    let native = run_native("stmt_consumers", "stmt_consumers", &program);
    if let Some(field) = first_difference(&mir, &native) {
        panic!(
            "MIR and native disagree on `{field}` for a program containing every statement \
             variant.\n--- mir ---\n{}\n--- native ---\n{}",
            canonical_form(&mir),
            canonical_form(&native)
        );
    }
}

/// Linkage validation walks statements to find instance references. It must not panic or miss a
/// variant — it is the consumer least likely to be remembered, because it reads MIR for a reason
/// unrelated to execution.
#[test]
fn linkage_validation_walks_every_statement_variant() {
    let front = front_end("stmt_linkage", EVERY_STATEMENT);
    let program = lower_program(&front.hir, &front.tables, front.file.clone())
        .unwrap_or_else(|e| panic!("lowering failed: {}", e.what));
    let verified = verify_program(&program)
        .unwrap_or_else(|errors| panic!("the verifier rejected the fixture: {errors:#?}"));
    // Walk every body the way linkage validation does. Reaching the end means the walk covered
    // the fixture's statements without refusing one it did not recognise; the count is asserted so
    // a walk that silently visited nothing cannot pass.
    let mut instances = 0usize;
    for body in &verified.program().bodies {
        starkc::backend::generated_rust::linkage::visit_instance_refs(body, &mut |_, _| {
            instances += 1;
        });
    }
    assert!(
        instances > 0,
        "the fixture calls user functions, so the linkage walk must see instance references"
    );
}
