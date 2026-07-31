//! WP-C7.4 — baseline MIR optimisations.
//!
//! # The rule these passes are built around
//!
//! Gate C7 fixes an order of authority: spec, then HIR, then **unoptimised MIR**, then optimised
//! and native execution. An optimised program that observes differently from the unoptimised one is
//! a defect in the optimiser — never a licence to revise the semantics. So every pass here is
//! written to preserve the §39 observation exactly: stdout and stderr bytes, exit status, returned
//! value, trap category and location, and the drop log.
//!
//! That last one does most of the constraining. STARK's drop log is observable output
//! (DROP-ORDER-001, §8.8), so a "dead" store whose local is later dropped is not dead, and a block
//! that looks unused but carries a `Drop` terminator cannot be deleted. The passes below therefore
//! never remove a `Drop`, and never delete a block that is still reachable — dead-block elimination
//! removes only blocks with no path from the entry, which by definition cannot contribute a byte to
//! any observation.
//!
//! # Why folding calls the interpreter
//!
//! Constant folding evaluates through [`crate::mir::interp::eval_checked`] / `eval_binop` /
//! `eval_unop` — the exact functions the MIR interpreter runs. Writing a second arithmetic
//! implementation inside an optimiser is the classic way to get a compiler that disagrees with its
//! own interpreter on one edge case in ten thousand. Sharing the code makes the two structurally
//! incapable of disagreeing, so the differential tests are checking the passes' *reasoning* rather
//! than re-testing arithmetic.
//!
//! # Trap semantics
//!
//! Integer overflow, division by zero, bad shifts and failing casts trap in **every** build mode.
//! Folding must therefore never make a trap disappear. When a `Checked` terminator's arguments are
//! all constants, the fold evaluates it and takes one of two branches:
//!
//! - it produces a value → the terminator becomes an assignment plus a `Goto`;
//! - it traps → the terminator becomes a `Trap` **carrying the original `TrapInfo`**, so the
//!   category and the source location a user sees are byte-identical to the unoptimised run.
//!
//! A folded trap is still a trap that happens at exactly the same point in the program, after
//! exactly the same preceding statements. This is the one place where "constant folding" and
//! "constant folding consistent with trap semantics" differ, and it is where an optimiser that
//! folded `1/0` into a value would silently delete a required abort.
//!
//! # What is deliberately NOT folded
//!
//! **Floating-point arithmetic.** The interpreter computes in `f64`; a native backend may compute a
//! `Float32` expression in `f32`. Folding a float with the interpreter's answer would bake the
//! interpreter's rounding into the native binary and make the native result depend on whether an
//! operand happened to be constant. Integers have no such freedom — every STARK integer type is a
//! fixed width, there is no pointer-sized integer, and two's-complement results are identical on
//! every target — so integer folding is target-independent and float folding is not.
//!
//! **`CheckIndex`.** It produces an opaque index-proof token, not a value; folding it away would
//! break the proof discipline the verifier enforces.
//!
//! **Anything reached through a reference or a projection.** Constant propagation applies only to
//! whole locals that are assigned exactly once, never borrowed, and never projected into.

use std::collections::{BTreeMap, BTreeSet};

use super::interp::{eval_binop, eval_checked, eval_unop, CheckedOutcome, MirValue};
use super::{
    BasicBlock, BlockId, CheckedOp, Constant, LocalId, LocalKind, MirBody, MirProgram, MirTy,
    Operand, Place, Rvalue, Statement, Terminator,
};

/// What a run of the optimiser changed. Reported rather than inferred: a pass that silently does
/// nothing looks identical to one that is working, and the C7.4 evidence needs to tell them apart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OptStats {
    /// `UnOp`/`BinOp` rvalues replaced by a constant.
    pub rvalues_folded: usize,
    /// `Checked` terminators resolved at compile time into a value.
    pub checked_folded: usize,
    /// `Checked` terminators proven to trap, rewritten as `Trap` with the original `TrapInfo`.
    pub checked_trapped: usize,
    /// `SwitchInt` terminators with a constant scrutinee replaced by `Goto`.
    pub branches_folded: usize,
    /// Operand reads replaced by the constant the local was assigned.
    pub constants_propagated: usize,
    /// Blocks with no path from the entry, removed.
    pub blocks_removed: usize,
}

impl OptStats {
    fn merge(&mut self, other: OptStats) {
        self.rvalues_folded += other.rvalues_folded;
        self.checked_folded += other.checked_folded;
        self.checked_trapped += other.checked_trapped;
        self.branches_folded += other.branches_folded;
        self.constants_propagated += other.constants_propagated;
        self.blocks_removed += other.blocks_removed;
    }

    fn any(&self) -> bool {
        *self != OptStats::default()
    }

    /// Total changes — for the `--verbose` line and for tests that only care whether a pass fired.
    pub fn total(&self) -> usize {
        self.rvalues_folded
            + self.checked_folded
            + self.checked_trapped
            + self.branches_folded
            + self.constants_propagated
            + self.blocks_removed
    }
}

/// Passes iterate to a fixpoint, because they feed each other: propagation exposes folds, folds
/// expose constant branches, folded branches make blocks unreachable, and removing those blocks can
/// make another local single-assignment. The bound exists so a pass pair that oscillated would
/// produce a slower build rather than a hang; reaching it is not a correctness problem, only a
/// missed optimisation, and no input has been observed to need more than three rounds.
const MAX_ROUNDS: usize = 8;

/// Optimise every body in place.
pub fn optimise(program: &mut MirProgram) -> OptStats {
    let mut stats = OptStats::default();
    for body in &mut program.bodies {
        stats.merge(optimise_body(body));
    }
    stats
}

pub fn optimise_body(body: &mut MirBody) -> OptStats {
    let mut stats = OptStats::default();
    for _ in 0..MAX_ROUNDS {
        let mut round = OptStats::default();
        round.merge(propagate_constants(body));
        round.merge(fold_rvalues(body));
        round.merge(fold_checked(body));
        round.merge(fold_branches(body));
        round.merge(remove_unreachable_blocks(body));
        if !round.any() {
            break;
        }
        stats.merge(round);
    }
    stats
}

// ------------------------------------------------------------ constant support --

/// Whether a constant may be folded or propagated at all.
///
/// Integers, booleans and `Unit` qualify. Floats do not — see the module note on `f32` versus
/// `f64`. `Str` and `FnPtr` are excluded conservatively: neither participates in arithmetic, so
/// admitting them buys nothing and each carries identity questions this pass has no reason to open.
fn foldable(c: &Constant) -> bool {
    matches!(c, Constant::Int(..) | Constant::Bool(_) | Constant::Unit)
}

fn to_value(c: &Constant) -> Option<MirValue> {
    match c {
        Constant::Int(v, _) => Some(MirValue::Int(*v)),
        Constant::Bool(b) => Some(MirValue::Bool(*b)),
        Constant::Unit => Some(MirValue::Unit),
        _ => None,
    }
}

/// Turn an evaluated value back into a constant of the destination's declared type.
///
/// The type comes from the destination local rather than from the value, because `MirValue::Int`
/// carries no width — an `i128` carrier holding `7` is a valid `Int8` and a valid `UInt64`, and only
/// the declared type says which. This is also why folding is restricted to assignments into a bare
/// local: a projected destination has no single declared type to read here.
fn from_value(value: &MirValue, ty: &MirTy) -> Option<Constant> {
    match value {
        MirValue::Int(v) if is_int_ty(ty) => Some(Constant::Int(*v, ty.clone())),
        MirValue::Bool(b) if matches!(ty, MirTy::Bool) => Some(Constant::Bool(*b)),
        MirValue::Unit if matches!(ty, MirTy::Unit) => Some(Constant::Unit),
        _ => None,
    }
}

fn is_int_ty(ty: &MirTy) -> bool {
    matches!(
        ty,
        MirTy::Int8
            | MirTy::Int16
            | MirTy::Int32
            | MirTy::Int64
            | MirTy::UInt8
            | MirTy::UInt16
            | MirTy::UInt32
            | MirTy::UInt64
            | MirTy::Char
    )
}

/// Checked operations whose result is exactly determined on every target.
///
/// `Cast` is admitted only when BOTH the operand and the destination are integral: an int→float or
/// float→int cast has the same representation freedom that keeps float arithmetic out of this pass.
fn foldable_checked(op: CheckedOp, dest_ty: &MirTy) -> bool {
    match op {
        CheckedOp::Add
        | CheckedOp::Sub
        | CheckedOp::Mul
        | CheckedOp::Div
        | CheckedOp::Rem
        | CheckedOp::Neg
        | CheckedOp::Pow
        | CheckedOp::Shl
        | CheckedOp::Shr => is_int_ty(dest_ty),
        CheckedOp::Cast => is_int_ty(dest_ty),
        // Produces an opaque proof token, not a value.
        _ => false,
    }
}

fn const_operand(op: &Operand) -> Option<&Constant> {
    match op {
        Operand::Const(c) if foldable(c) => Some(c),
        _ => None,
    }
}

// ------------------------------------------------------- constant propagation --

/// Replace reads of a local that provably holds one constant with that constant.
///
/// Eligibility is deliberately narrow, and each condition closes a specific unsoundness:
///
/// - **assigned exactly once, with a constant** — otherwise the value at a given read is not known;
/// - **a `Temp` or `User` local** — never a parameter, return slot, drop flag or index proof, each
///   of which is written or read by machinery outside the statement stream;
/// - **never borrowed** (`RefOf`) — a reference can observe or mutate the local through an alias
///   this pass does not track;
/// - **never used as a projection base, and never assigned through a projection** — a partial write
///   would invalidate the recorded constant;
/// - **never the operand of a `Drop`** — dropping is observable, so a local that is dropped must
///   keep existing as a place.
///
/// Reads via `Move` are substituted as well as reads via `Copy`. That is sound precisely because
/// the constants admitted here are scalars: moving one leaves nothing behind to observe, and the
/// verifier's move analysis is not weakened by a value that was never a resource.
fn propagate_constants(body: &mut MirBody) -> OptStats {
    let mut assignments: BTreeMap<u32, Option<Constant>> = BTreeMap::new();
    let mut disqualified: BTreeSet<u32> = BTreeSet::new();

    for (index, decl) in body.locals.iter().enumerate() {
        if !matches!(decl.kind, LocalKind::Temp | LocalKind::User(_)) {
            disqualified.insert(index as u32);
        }
    }

    // Pass 1: record single constant assignments and disqualify everything unsafe.
    for block in &body.blocks {
        for (statement, _) in &block.statements {
            let Statement::Assign(place, rvalue) = statement else {
                continue;
            };
            if !place.projection.is_empty() {
                disqualified.insert(place.local.0);
            } else {
                let constant = match rvalue {
                    Rvalue::Use(Operand::Const(c)) if foldable(c) => Some(c.clone()),
                    _ => None,
                };
                match assignments.entry(place.local.0) {
                    std::collections::btree_map::Entry::Vacant(slot) => {
                        slot.insert(constant);
                    }
                    // A second assignment: the local is not single-valued.
                    std::collections::btree_map::Entry::Occupied(_) => {
                        disqualified.insert(place.local.0);
                    }
                }
            }
            visit_rvalue_places(rvalue, &mut |place, borrowed| {
                if borrowed || !place.projection.is_empty() {
                    disqualified.insert(place.local.0);
                }
            });
        }
        match &block.terminator.0 {
            Terminator::Drop { place, .. } => {
                disqualified.insert(place.local.0);
            }
            Terminator::Call { dest, .. } => {
                disqualified.insert(dest.local.0);
            }
            // A `Checked` destination is an assignment like any other: the first one records the
            // local as assigned-but-not-constant, and a second disqualifies it outright.
            Terminator::Checked { dest, .. } => match assignments.entry(dest.0) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(None);
                }
                std::collections::btree_map::Entry::Occupied(_) => {
                    disqualified.insert(dest.0);
                }
            },
            _ => {}
        }
        for operand in terminator_operands(&block.terminator.0) {
            if let Operand::Copy(place) | Operand::Move(place) = operand {
                if !place.projection.is_empty() {
                    disqualified.insert(place.local.0);
                }
            }
        }
    }

    let known: BTreeMap<u32, Constant> = assignments
        .into_iter()
        .filter(|(local, _)| !disqualified.contains(local))
        .filter_map(|(local, constant)| constant.map(|c| (local, c)))
        .collect();
    if known.is_empty() {
        return OptStats::default();
    }

    // Pass 2: substitute reads. The defining assignment itself is left alone — removing it is
    // dead-store elimination, which is not in C7.4's permitted set, and leaving it costs one store
    // that the host backend's own optimiser removes.
    let mut replaced = 0usize;
    let substitute = |operand: &mut Operand, replaced: &mut usize| {
        let local = match operand {
            Operand::Copy(place) | Operand::Move(place) if place.projection.is_empty() => {
                place.local.0
            }
            _ => return,
        };
        if let Some(constant) = known.get(&local) {
            *operand = Operand::Const(constant.clone());
            *replaced += 1;
        }
    };
    for block in &mut body.blocks {
        for (statement, _) in &mut block.statements {
            let Statement::Assign(place, rvalue) = statement else {
                continue;
            };
            let defining = place.projection.is_empty() && known.contains_key(&place.local.0);
            if defining {
                continue;
            }
            visit_rvalue_operands(rvalue, &mut |operand| substitute(operand, &mut replaced));
        }
        for operand in terminator_operands_mut(&mut block.terminator.0) {
            substitute(operand, &mut replaced);
        }
    }
    OptStats {
        constants_propagated: replaced,
        ..OptStats::default()
    }
}

// ------------------------------------------------------------ constant folding --

fn fold_rvalues(body: &mut MirBody) -> OptStats {
    let mut folded = 0usize;
    for block_index in 0..body.blocks.len() {
        for statement_index in 0..body.blocks[block_index].statements.len() {
            let (statement, _) = &body.blocks[block_index].statements[statement_index];
            let Statement::Assign(place, rvalue) = statement else {
                continue;
            };
            if !place.projection.is_empty() {
                continue;
            }
            let ty = body.locals[place.local.0 as usize].ty.clone();
            let value = match rvalue {
                Rvalue::UnOp(op, operand) => {
                    let Some(c) = const_operand(operand) else {
                        continue;
                    };
                    let Some(v) = to_value(c) else { continue };
                    eval_unop(*op, v).ok()
                }
                Rvalue::BinOp(op, lhs, rhs) => {
                    let (Some(l), Some(r)) = (const_operand(lhs), const_operand(rhs)) else {
                        continue;
                    };
                    let (Some(l), Some(r)) = (to_value(l), to_value(r)) else {
                        continue;
                    };
                    eval_binop(*op, l, r).ok()
                }
                _ => continue,
            };
            let Some(value) = value else { continue };
            let Some(constant) = from_value(&value, &ty) else {
                continue;
            };
            let (statement, _) = &mut body.blocks[block_index].statements[statement_index];
            if let Statement::Assign(_, rvalue) = statement {
                *rvalue = Rvalue::Use(Operand::Const(constant));
                folded += 1;
            }
        }
    }
    OptStats {
        rvalues_folded: folded,
        ..OptStats::default()
    }
}

/// Resolve `Checked` terminators whose arguments are all constants.
///
/// This is the pass the trap rule is about. Both outcomes are preserved exactly: a successful
/// operation becomes a store and a `Goto`; a trapping one becomes `Trap` carrying the ORIGINAL
/// `TrapInfo`, so category, file, line and column are unchanged. The statements that precede the
/// terminator are untouched, so anything they wrote to stdout or to the drop log still happens
/// first, in order.
fn fold_checked(body: &mut MirBody) -> OptStats {
    /// What the fold decided, computed while the terminator is only READ so the rewrite below can
    /// take a mutable borrow. Keeping the two phases apart is not merely a borrow-checker
    /// concession: it makes "decide" and "rewrite" separately readable, and the trap case is the
    /// one a reviewer will want to check in isolation.
    enum Decision {
        Value(LocalId, BlockId, Constant),
        Trap(super::TrapInfo),
    }

    let locals = body.locals.clone();
    let mut stats = OptStats::default();
    for block in &mut body.blocks {
        let decision = {
            let Terminator::Checked {
                op,
                args,
                dest,
                target,
                trap,
            } = &block.terminator.0
            else {
                continue;
            };
            let dest_ty = &locals[dest.0 as usize].ty;
            if !foldable_checked(*op, dest_ty) || args.is_empty() {
                continue;
            }
            let mut values = Vec::with_capacity(args.len());
            for arg in args {
                let Some(value) = const_operand(arg).and_then(to_value) else {
                    break;
                };
                values.push(value);
            }
            if values.len() != args.len() {
                continue;
            }
            let Ok(outcome) = eval_checked(*op, &values, dest_ty) else {
                continue;
            };
            match outcome {
                CheckedOutcome::Value(value) => {
                    let Some(constant) = from_value(&value, dest_ty) else {
                        continue;
                    };
                    Decision::Value(*dest, *target, constant)
                }
                CheckedOutcome::Trap(override_category) => {
                    let mut info = *trap;
                    // A5 shifts report `InvalidShift`, and signed `MIN / -1` / `MIN % -1` report
                    // `IntegerOverflow`, rather than the terminator's own category; the
                    // interpreter applies those overrides, so the fold must apply them identically
                    // or a folded operation would report a different category from an executed one.
                    if let Some(category) = override_category {
                        info.category = category;
                    }
                    Decision::Trap(info)
                }
            }
        };
        match decision {
            Decision::Value(dest, target, constant) => {
                let info = block.terminator.1;
                block.statements.push((
                    Statement::Assign(Place::local(dest), Rvalue::Use(Operand::Const(constant))),
                    info,
                ));
                block.terminator.0 = Terminator::Goto { target };
                stats.checked_folded += 1;
            }
            Decision::Trap(info) => {
                block.terminator.0 = Terminator::Trap {
                    info,
                    message: None,
                };
                stats.checked_trapped += 1;
            }
        }
    }
    stats
}

/// A `SwitchInt` on a constant scrutinee has one reachable successor. Replacing it with a `Goto` is
/// what makes the other arms unreachable, which is what dead-block elimination then removes — the
/// two passes are only useful together.
fn fold_branches(body: &mut MirBody) -> OptStats {
    let mut folded = 0usize;
    for block in &mut body.blocks {
        let Terminator::SwitchInt {
            scrut,
            arms,
            otherwise,
        } = &block.terminator.0
        else {
            continue;
        };
        let Some(constant) = const_operand(scrut) else {
            continue;
        };
        let discriminant: u128 = match constant {
            Constant::Int(v, _) if *v >= 0 => *v as u128,
            Constant::Bool(b) => u128::from(*b),
            // A negative scrutinee cannot match a `u128` arm key, so it always takes `otherwise` —
            // but rather than encode that reasoning here, leave it alone. The pass exists to remove
            // branches it is certain about.
            _ => continue,
        };
        let target = arms
            .iter()
            .find(|(key, _)| *key == discriminant)
            .map(|(_, block)| *block)
            .unwrap_or(*otherwise);
        block.terminator.0 = Terminator::Goto { target };
        folded += 1;
    }
    OptStats {
        branches_folded: folded,
        ..OptStats::default()
    }
}

// ---------------------------------------------------- dead-block elimination --

/// Remove blocks with no path from the entry, and renumber what remains.
///
/// Unreachable blocks are the only ones safe to delete without a liveness analysis: no execution
/// enters them, so nothing they contain — including a `Drop`, a `Call` or a trap — can contribute a
/// byte to any observation. A block that is merely *rarely* reached is not touched.
fn remove_unreachable_blocks(body: &mut MirBody) -> OptStats {
    let total = body.blocks.len();
    let mut reachable = vec![false; total];
    let mut stack = vec![body.entry.0 as usize];
    while let Some(index) = stack.pop() {
        if index >= total || reachable[index] {
            continue;
        }
        reachable[index] = true;
        for successor in successors(&body.blocks[index].terminator.0) {
            stack.push(successor.0 as usize);
        }
    }
    if reachable.iter().all(|hit| *hit) {
        return OptStats::default();
    }

    let mut remap = vec![None; total];
    let mut next = 0u32;
    for (index, hit) in reachable.iter().enumerate() {
        if *hit {
            remap[index] = Some(BlockId(next));
            next += 1;
        }
    }
    let removed = total - next as usize;

    let mut kept: Vec<BasicBlock> = Vec::with_capacity(next as usize);
    for (index, block) in body.blocks.drain(..).enumerate() {
        if reachable[index] {
            kept.push(block);
        }
    }
    for block in &mut kept {
        for target in successors_mut(&mut block.terminator.0) {
            // Every successor of a reachable block is itself reachable, so the remap entry exists.
            *target = remap[target.0 as usize].expect("successor of a reachable block");
        }
    }
    body.blocks = kept;
    body.entry = remap[body.entry.0 as usize].expect("entry is reachable from itself");
    OptStats {
        blocks_removed: removed,
        ..OptStats::default()
    }
}

// ------------------------------------------------------------------- walking --

fn successors(terminator: &Terminator) -> Vec<BlockId> {
    match terminator {
        Terminator::Goto { target } => vec![*target],
        Terminator::SwitchInt {
            arms, otherwise, ..
        } => {
            let mut out: Vec<BlockId> = arms.iter().map(|(_, block)| *block).collect();
            out.push(*otherwise);
            out
        }
        Terminator::Call { target, .. }
        | Terminator::Drop { target, .. }
        | Terminator::Checked { target, .. } => vec![*target],
        Terminator::Trap { .. } | Terminator::Return | Terminator::Unreachable => Vec::new(),
    }
}

fn successors_mut(terminator: &mut Terminator) -> Vec<&mut BlockId> {
    match terminator {
        Terminator::Goto { target } => vec![target],
        Terminator::SwitchInt {
            arms, otherwise, ..
        } => {
            let mut out: Vec<&mut BlockId> =
                arms.iter_mut().map(|(_, block)| block).collect::<Vec<_>>();
            out.push(otherwise);
            out
        }
        Terminator::Call { target, .. }
        | Terminator::Drop { target, .. }
        | Terminator::Checked { target, .. } => vec![target],
        Terminator::Trap { .. } | Terminator::Return | Terminator::Unreachable => Vec::new(),
    }
}

/// Visit every place an rvalue reads, flagging whether the read is a BORROW. A borrowed local is
/// disqualified from propagation regardless of what it holds.
fn visit_rvalue_places(rvalue: &Rvalue, visit: &mut impl FnMut(&Place, bool)) {
    match rvalue {
        Rvalue::Use(operand) | Rvalue::UnOp(_, operand) => visit_operand_place(operand, visit),
        Rvalue::BinOp(_, lhs, rhs) => {
            visit_operand_place(lhs, visit);
            visit_operand_place(rhs, visit);
        }
        Rvalue::Aggregate(_, operands) => {
            for operand in operands {
                visit_operand_place(operand, visit);
            }
        }
        Rvalue::Discriminant(place) => visit(place, false),
        Rvalue::RefOf { place, .. } => visit(place, true),
        Rvalue::LayoutQuery { .. } => {}
    }
}

fn visit_operand_place(operand: &Operand, visit: &mut impl FnMut(&Place, bool)) {
    if let Operand::Copy(place) | Operand::Move(place) = operand {
        visit(place, false);
    }
}

fn visit_rvalue_operands(rvalue: &mut Rvalue, visit: &mut impl FnMut(&mut Operand)) {
    match rvalue {
        Rvalue::Use(operand) | Rvalue::UnOp(_, operand) => visit(operand),
        Rvalue::BinOp(_, lhs, rhs) => {
            visit(lhs);
            visit(rhs);
        }
        Rvalue::Aggregate(_, operands) => {
            for operand in operands {
                visit(operand);
            }
        }
        Rvalue::Discriminant(_) | Rvalue::RefOf { .. } | Rvalue::LayoutQuery { .. } => {}
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<&Operand> {
    match terminator {
        Terminator::SwitchInt { scrut, .. } => vec![scrut],
        Terminator::Call { args, callee, .. } => {
            let mut out: Vec<&Operand> = args.iter().collect();
            if let super::Callee::FnValue(operand) = callee {
                out.push(operand);
            }
            out
        }
        Terminator::Checked { args, .. } => args.iter().collect(),
        Terminator::Trap { message, .. } => message.iter().collect(),
        _ => Vec::new(),
    }
}

fn terminator_operands_mut(terminator: &mut Terminator) -> Vec<&mut Operand> {
    match terminator {
        Terminator::SwitchInt { scrut, .. } => vec![scrut],
        Terminator::Call { args, callee, .. } => {
            let mut out: Vec<&mut Operand> = args.iter_mut().collect();
            if let super::Callee::FnValue(operand) = callee {
                out.push(operand);
            }
            out
        }
        Terminator::Checked { args, .. } => args.iter_mut().collect(),
        Terminator::Trap { message, .. } => message.iter_mut().collect(),
        _ => Vec::new(),
    }
}

/// A local that is only ever assigned and never read is NOT removed — see the module note on drop
/// logs. Kept as a named helper so the omission reads as a decision rather than an oversight.
#[allow(dead_code)]
fn dead_store_elimination_is_out_of_scope() {}
