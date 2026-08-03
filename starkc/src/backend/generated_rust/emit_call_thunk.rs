//! **DEV-160 — the call-site thunk plan.**
//!
//! # The defect
//!
//! STARK's borrow checker is place-granular (DEV-154) and correctly admits disjoint accesses to one
//! aggregate in one call:
//!
//! ```text
//! consume(p.url.as_str(), p.headers, p.body)
//! ```
//!
//! The generated projections each borrow the WHOLE slot — `stark_copy_…(&_2)`,
//! `stark_move_…(&mut _2)` — so rustc sees an `&`/`&mut` conflict and rejects code the compiler
//! proved sound. A correct program refused by the backend.
//!
//! # Why a thunk, and why it owns the whole call
//!
//! The fix cannot be to reorder arguments: CD-007 freezes strict left-to-right evaluation, and a
//! generated-Rust limitation may not change language semantics. It cannot be to hoist each argument
//! into a local either — a hoisted shared borrow stays live until the call consumes it, so every
//! later `&mut` still conflicts.
//!
//! So a generated thunk takes each participating slot ONCE through a real lifetime-bearing
//! `&'a mut ValueSlot<T>`, derives raw pointers from it, performs every access through those, and
//! invokes the callee itself.
//!
//! **The thunk owns the COMPLETE MIR-significant evaluation, not just the conflicting slot.**
//! Pre-evaluating another argument at the call site would place it before the projections performed
//! inside the thunk, splitting the ordered MIR argument sequence — the CD-007 violation the design
//! exists to avoid. Only observationally inert reads (constants, unprojected non-slot `Copy`
//! locals) may be evaluated at the call site and passed by value.
//!
//! # Why a plan
//!
//! Detection, helper collection and emission all consume ONE [`CallThunkPlan`]. They must not
//! independently reconstruct what a call requires: that is exactly how DEV-162 shipped an `E0425` —
//! the emitter named a helper the collector never generated, surfacing as a name error inside code
//! nobody wrote. A shared structure replaces the comment that used to say "the two must agree".

use super::{emit_places, emit_projections, emit_types, BackendDiagnostic};
use crate::mir::{Callee, MirTy, Operand, Place};
use emit_projections::{HelperOp, ProjectionHelper};
use std::collections::BTreeMap;

/// One MIR-significant argument evaluation, in MIR order.
#[derive(Clone, Debug)]
pub enum ThunkArg {
    /// A field of a participating slot, through a raw-pointer projection wrapper.
    Projection {
        slot: usize,
        /// The `stark_proj` wrapper's name — the raw twin, taking `*mut ValueSlot<T>`.
        helper: String,
        /// True for a `Ref` twin, whose result is a reference and needs the thunk's lifetime.
        borrows: bool,
    },
    /// A whole slot-backed local moved out: `take_raw`.
    Take { slot: usize },
    /// Evaluated at the CALL SITE and passed by value. Permitted only where the read is
    /// observationally inert, so moving it ahead of the thunk's projections changes nothing.
    ByValue { index: usize },
}

#[derive(Clone, Debug)]
pub struct ThunkSlot {
    pub local: u32,
    pub ty: MirTy,
}

#[derive(Clone, Debug)]
pub struct ThunkByValue {
    pub ty: MirTy,
    /// The expression the ordinary emitter would have produced, rendered at the call site.
    pub expr: String,
}

/// Everything detection, collection and emission need. Built once, consumed by all three.
#[derive(Clone)]
pub struct CallThunkPlan {
    pub name: String,
    /// The call site this plan belongs to. A `Call` is a TERMINATOR, so a block holds at most one —
    /// `(function symbol, block index)` is unique, and it is how the emitter finds the plan the
    /// collector already built rather than building a second one.
    pub body_symbol: String,
    pub block_index: u32,
    /// Participating slot-backed locals, in first-use order. Each becomes one `&'a mut ValueSlot`
    /// parameter.
    pub slots: Vec<ThunkSlot>,
    pub by_value: Vec<ThunkByValue>,
    /// Arguments in EXACT MIR order.
    pub args: Vec<ThunkArg>,
    /// The callee's generated Rust function name. Direct `Instance` calls only, for now.
    pub callee: String,
    pub ret_ty: MirTy,
    /// The raw-twin wrappers this thunk calls. Emitted because the plan says so, never rediscovered.
    pub helpers: Vec<ProjectionHelper>,
    /// Indices, within this call's block, of the statements the thunk performs instead --- the
    /// `RefOf` that creates a borrow and any copy carrying it to the argument. `emit_bodies`
    /// suppresses exactly these; emitting them as well would leave a live borrow beside the
    /// thunk's `&mut`, which is the conflict itself.
    pub absorbed: Vec<usize>,
}

/// A stable per-body call-site identity. A `Call` is a TERMINATOR, so a block holds at most one —
/// `(function symbol, block index)` is therefore unique and stable across runs.
pub fn thunk_name(body_symbol: &str, block_index: u32) -> String {
    super::mangle::sanitize_symbol(&format!("thunk#{body_symbol}#bb{block_index}"))
}

/// Every call in the program that needs a thunk, planned ONCE.
///
/// This is the single authority the addendum requires: `emit_projections` skips the argument lists
/// these plans cover, `emit_projections::emit` renders the helpers each plan names, and
/// `emit_bodies::emit_call` looks its plan up here instead of deciding again. Three consumers, one
/// structure — the arrangement DEV-162 lacked when the emitter named a helper the collector had
/// never been asked for.
pub fn collect_plans(
    program: &crate::mir::MirProgram,
    layout: &crate::layout::TargetLayout,
) -> Result<Vec<CallThunkPlan>, BackendDiagnostic> {
    let mut plans = Vec::new();
    for body in &program.bodies {
        let env = emit_places::TyEnv::new(body, &program.types, layout)
            .with_provider_calls(&program.provider_calls);
        for block_index in 0..body.blocks.len() as u32 {
            if let Some(plan) = plan_for_call(body, block_index, &env)? {
                plans.push(plan);
            }
        }
    }
    Ok(plans)
}

/// The plan for one call site, if it has one.
pub fn plan_at<'p>(
    plans: &'p [CallThunkPlan],
    body_symbol: &str,
    block_index: u32,
) -> Option<&'p CallThunkPlan> {
    plans
        .iter()
        .find(|p| p.block_index == block_index && p.body_symbol == body_symbol)
}

/// Build the plan for a call, or `None` when ordinary emission is correct.
///
/// **The condition is narrow on purpose.** A thunk is needed only when the same slot-backed local
/// is accessed two or more ways by one call AND at least one of those accesses needs `&mut` — a
/// move. Two `Copy` reads of one slot are two shared borrows and compile fine; a call touching a
/// slot once is fine. Ordinary calls keep byte-identical emission, which is what bounds the blast
/// radius of this change on the compiler's most-used path.
///
/// It takes the whole BLOCK, not just the argument list, because one of the accesses is usually not
/// in the argument list at all: `f(&p.name, p.body)` lowers to a `RefOf` STATEMENT that fills a
/// reference temporary, and only the temporary appears as an argument. That statement is the `&`
/// side of the conflict, and a plan that could not see it would miss the exact shape the defect was
/// reported as.
pub fn plan_for_call(
    body: &crate::mir::MirBody,
    block_index: u32,
    env: &emit_places::TyEnv,
) -> Result<Option<CallThunkPlan>, BackendDiagnostic> {
    let block = &body.blocks[block_index as usize];
    let crate::mir::Terminator::Call {
        callee, args, dest, ..
    } = &block.terminator.0
    else {
        return Ok(None);
    };
    let (borrows, escaping) = absorbable_borrows(body, block_index, args, env)?;
    if !conflicts(args, &borrows, &escaping, env)? {
        return Ok(None);
    }

    // **The provider audit (A10/CD-200).** A provider call never reaches `emit_call`: it is emitted
    // as a statement SEQUENCE — one `let __prov_aN = ...;` per argument, then the `extern "C"` call
    // — because it needs named borrow temporaries the expression form has nowhere to put.
    //
    // That shape has the SAME conflict. `__prov_a0` holding a shared borrow is a live local when
    // `__prov_a1` moves a sibling out through `&mut`, and rustc rejects it identically. The thunk
    // does not apply (there is no single expression to replace, and the ABI's out-parameters and
    // handle transfers are not arguments a thunk could carry), so this is a refusal rather than a
    // gap — and a refusal is the point: without it the case would reach rustc as `E0502` in a
    // generated statement sequence, which is how this defect family stays invisible.
    if let Callee::Provider(id) = callee {
        return Err(BackendDiagnostic::Unsupported(format!(
            "provider call #{} accesses one slot-backed local several ways in one argument list. \
             A provider call is emitted as a statement sequence with named borrow temporaries \
             (A10/CD-200), so the DEV-160 call thunk does not apply to it. Bind the fields to \
             locals before the call",
            id.0
        )));
    }

    // A borrow of a field that must OUTLIVE this call, beside a move out of a sibling. The thunk
    // cannot absorb it (the value is needed later, so its definition must stay where it is) and
    // cannot avoid it (its `&mut` conflicts with the live borrow). Refused by name, before rustc:
    // the alternative is `E0502` pointing at a line of generated code, which tells the user
    // nothing about their program.
    if let Some((local, borrow)) = escaping.iter().next() {
        return Err(BackendDiagnostic::Unsupported(format!(
            "a borrow of _{}{:?} outlives a call that moves out of a sibling field of the same \
             local (_{local} is read again after the call). STARK accepts this -- the accesses are \
             disjoint -- but the native backend cannot yet emit it: the thunk that resolves the \
             conflict (DEV-160) can only take over a borrow whose every use is an argument of the \
             one call",
            borrow.source.local.0, borrow.source.projection
        )));
    }

    // Direct calls only, for now. A non-direct call that NEEDS a thunk is refused by name here,
    // before rustc, rather than emitted in a form that will fail with a borrow error in generated
    // code.
    let Callee::Instance(instance) = callee else {
        return Err(BackendDiagnostic::Unsupported(format!(
            "this call needs a DEV-160 call-site thunk (it accesses one slot-backed local several \
             ways at once), and thunks are implemented for direct calls only. Callee: {callee:?}"
        )));
    };

    let ret_ty = body.locals[dest.local.0 as usize].ty.clone();
    let mut slots: Vec<ThunkSlot> = Vec::new();
    let mut by_value: Vec<ThunkByValue> = Vec::new();
    let mut plan_args: Vec<ThunkArg> = Vec::new();
    let mut helpers: BTreeMap<String, ProjectionHelper> = BTreeMap::new();
    let mut absorbed: Vec<usize> = Vec::new();
    // Every by-value argument's source local, so provenance can be checked against the
    // participating slots once they are all known.
    let mut by_value_provenance: Vec<(u32, usize)> = Vec::new();

    for arg in args {
        let (Operand::Move(place) | Operand::Copy(place)) = arg else {
            // A constant: inert by construction.
            by_value.push(ThunkByValue {
                ty: super::emit_bodies::operand_mir_ty(arg, env)?,
                expr: super::emit_bodies::emit_operand(arg, env)?,
            });
            plan_args.push(ThunkArg::ByValue {
                index: by_value.len() - 1,
            });
            continue;
        };

        // A reference temporary this thunk takes over. The `RefOf` statement is suppressed at the
        // call site and re-performed HERE, through the raw twin, so the borrow it produces is
        // derived from the same `&'a mut` the sibling move goes through instead of racing it.
        if let Some(borrow) = borrows.get(&place.local.0) {
            let slot = slot_index(&mut slots, &borrow.source, env)?;
            let helper = collect_raw_helper(&borrow.source, env, HelperOp::RefRaw, &mut helpers)?;
            plan_args.push(ThunkArg::Projection {
                slot,
                helper,
                borrows: true,
            });
            for index in &borrow.statements {
                if !absorbed.contains(index) {
                    absorbed.push(*index);
                }
            }
            continue;
        }

        if !emit_places::is_slot_local(place.local.0, env)? {
            // A non-slot local: a `Copy` scalar or a stored reference. Its read has no observable
            // order, so evaluating it at the call site is inert. Non-slot locals are NOT turned
            // into slots to make them thunk parameters.
            let ty = env.place_ty(place)?;
            by_value_provenance.push((place.local.0, by_value.len()));
            by_value.push(ThunkByValue {
                ty,
                expr: super::emit_bodies::emit_operand(arg, env)?,
            });
            plan_args.push(ThunkArg::ByValue {
                index: by_value.len() - 1,
            });
            continue;
        }

        let slot = slot_index(&mut slots, place, env)?;

        if place.projection.is_empty() {
            match arg {
                Operand::Move(_) => plan_args.push(ThunkArg::Take { slot }),
                // A whole-value shared read of a slot inside a conflicting call would need a
                // `&ValueSlot`, which is exactly the reference this design must not reconstruct.
                Operand::Copy(_) => {
                    return Err(BackendDiagnostic::Unsupported(format!(
                        "a DEV-160 thunk cannot pass the whole slot-backed local _{} by shared \
                         reference: forming `&ValueSlot` beside a raw field access is the aliasing \
                         the thunk exists to avoid",
                        place.local.0
                    )))
                }
                Operand::Const(_) => unreachable!("handled above"),
            }
            continue;
        }

        let (op, borrows) = projection_op(arg, place, env)?;
        let helper = collect_raw_helper(place, env, op, &mut helpers)?;
        plan_args.push(ThunkArg::Projection {
            slot,
            helper,
            borrows,
        });
    }

    // **A by-value argument must not BORROW a slot the thunk takes `&mut`.**
    //
    // This is the shape DEV-160 was reported as: `send_once(builder.url.as_str(), builder.headers,
    // builder.body)`. `as_str` runs in an earlier block and returns a `&str` that borrows the very
    // local the thunk is about to take exclusively. Nothing about the argument says so -- it is an
    // ordinary non-slot local -- which is why provenance is traced rather than assumed.
    //
    // Absorbing it would mean absorbing the intermediate CALL as well, across a block boundary.
    // That is a larger mechanism than this increment carries, so the case is refused by name.
    let provenance = borrow_provenance(body, env)?;
    let participating: std::collections::BTreeSet<u32> = slots.iter().map(|s| s.local).collect();
    for (local, index) in &by_value_provenance {
        // A value that CANNOT carry a borrow is not a conflict however it was computed. Without
        // this, `consume(p.taken, p.kept.len())` would be refused: `len` takes `&p.kept`, so the
        // provenance relation propagates `_1` into its result -- but that result is a `UInt64` and
        // borrows nothing. Provenance over-approximates by design; the type is what makes it
        // actionable.
        if !may_carry_borrow(&by_value[*index].ty) {
            continue;
        }
        let Some(borrowed) = provenance.get(local) else {
            continue;
        };
        if let Some(slot) = borrowed.intersection(&participating).next() {
            return Err(BackendDiagnostic::Unsupported(format!(
                "the call in bb{block_index} of `{}` passes a reference (_{local}) that borrows \
                 _{slot} while also moving out of _{slot}'s fields. STARK accepts this -- the \
                 accesses are disjoint -- but the reference reaches the call through an earlier \
                 block (typically an intermediate call such as `.as_str()`), and the DEV-160 thunk \
                 can only take over evaluation within the call's OWN block. Bind the fields to \
                 locals before the call as a workaround",
                body.instance.symbol
            )));
        }
    }

    Ok(Some(CallThunkPlan {
        name: thunk_name(&body.instance.symbol, block_index),
        body_symbol: body.instance.symbol.clone(),
        block_index,
        slots,
        by_value,
        args: plan_args,
        callee: super::mangle::function_name_for_symbol(&instance.symbol),
        ret_ty,
        helpers: helpers.into_values().collect(),
        absorbed,
    }))
}

/// The `RefOf` statements in this block that the thunk may take over: reference temporary → the
/// slot-backed place it borrows.
///
/// Three conditions, and all three are needed:
///
/// 1. the borrow is created in **this** block, so moving it inside the thunk does not move it past
///    a branch;
/// 2. its base is a **slot-backed local with a projection** — a whole-slot borrow has no raw twin,
///    and STARK's own borrow checker rejects it beside a sibling move anyway;
/// 3. **every** read of the temporary in the whole body is an argument of THIS call. Suppressing
///    the definition is only sound if nothing else needs the value; a borrow read once here and
///    once three blocks later must still be created where it was.
///
/// Condition 3 counts reads rather than requiring exactly one, so `f(r, r, p.body)` — one borrow
/// used twice in one argument list — absorbs as two shared raw projections, which is what it is.
///
/// # Delaying the borrow is sound, and it is a real delay
///
/// The `RefOf` moves from where MIR put it to inside the thunk, i.e. after any statement that sat
/// between it and the call. That is only safe because the front end has already proved nothing in
/// between can mutate what is borrowed: a shared borrow of `p.name` forbids writing `p.name` or
/// `p` while it lives (03 "References and Lifetimes"). A *disjoint* sibling may be moved in that
/// gap, and re-deriving through a raw projection reads the untouched field either way — which is
/// precisely what a whole-value accessor could not do.
///
/// A candidate failing any condition is returned separately as an ESCAPING borrow, not silently
/// dropped: see [`plan_for_call`], which refuses it by name rather than leaving rustc to report
/// `E0502` inside code the user never wrote.
// `(absorbable, escaping)`. A named type for a local pair would hide which half is which at the one
// call site that reads it.
#[allow(clippy::type_complexity)]
fn absorbable_borrows(
    body: &crate::mir::MirBody,
    block_index: u32,
    args: &[Operand],
    env: &emit_places::TyEnv,
) -> Result<(BTreeMap<u32, Borrow>, BTreeMap<u32, Borrow>), BackendDiagnostic> {
    use crate::mir::{Rvalue, Statement};
    // A borrow does not always reach the call in one step. `let r = &p.name;` lowers to a PAIR --
    // `_8 = &_1.0` then `_7 = copy _8` -- and only `_7` appears as an argument, so a search for
    // `RefOf` destinations alone finds nothing to absorb and the conflict survives. The chain is
    // followed, and every statement along it is suppressed together: leaving the copy behind would
    // leave a read of a temporary whose definition had just been removed.
    let mut chains: BTreeMap<u32, Borrow> = BTreeMap::new();
    for (index, (statement, _)) in body.blocks[block_index as usize]
        .statements
        .iter()
        .enumerate()
    {
        let Statement::Assign(dest, rvalue) = statement else {
            continue;
        };
        if !dest.projection.is_empty() {
            continue;
        }
        match rvalue {
            Rvalue::RefOf { place, .. } => {
                // A WHOLE-slot borrow has no raw twin, and STARK's own borrow checker rejects one
                // beside a move out of the same value anyway -- so there is nothing to absorb.
                if place.projection.is_empty() || !emit_places::is_slot_local(place.local.0, env)? {
                    continue;
                }
                chains.insert(
                    dest.local.0,
                    Borrow {
                        source: place.clone(),
                        statements: vec![index],
                    },
                );
            }
            Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) if src.projection.is_empty() => {
                if let Some(previous) = chains.get(&src.local.0).cloned() {
                    // The intermediate must exist ONLY to feed this copy. If it is read anywhere
                    // else, removing its definition would break that other read.
                    if reads_of(body, src.local.0) != 1 {
                        continue;
                    }
                    let mut statements = previous.statements.clone();
                    statements.push(index);
                    chains.insert(
                        dest.local.0,
                        Borrow {
                            source: previous.source,
                            statements,
                        },
                    );
                }
            }
            _ => {}
        }
    }

    let mut absorbable = BTreeMap::new();
    let mut escaping = BTreeMap::new();
    for (local, borrow) in chains {
        let here = args
            .iter()
            .filter(|a| {
                matches!(a, Operand::Copy(p) | Operand::Move(p)
                    if p.local.0 == local && p.projection.is_empty())
            })
            .count() as u32;
        if here == 0 {
            continue;
        }
        if here == reads_of(body, local) {
            absorbable.insert(local, borrow);
        } else {
            escaping.insert(local, borrow);
        }
    }
    Ok((absorbable, escaping))
}

/// Whether a value of this type could carry a borrow at all.
///
/// A declared struct or enum FIELD may not be a reference (03 rule 1), so nothing needs to be looked
/// up: a borrow reaches a composite only through a generic ARGUMENT (`Option<&T>`, `(&T, U)`) or a
/// structural component, and both are present in the `MirTy` itself. `FnPtr` is a code address and
/// borrows nothing.
fn may_carry_borrow(ty: &MirTy) -> bool {
    match ty {
        MirTy::Ref { .. } => true,
        MirTy::Tuple(parts)
        | MirTy::Struct(_, parts)
        | MirTy::Enum(_, parts)
        | MirTy::Core(_, parts) => parts.iter().any(may_carry_borrow),
        MirTy::Array(element, _) | MirTy::Slice(element) => may_carry_borrow(element),
        _ => false,
    }
}

/// Which slot-backed locals each local's value may BORROW, over-approximated.
///
/// A reference does not have to arrive at a call as a `RefOf` in the same block. The idiom this
/// defect was reported as sends it through an intermediate call:
///
/// ```text
/// _7 = &_2.0                       // &builder.url
/// _8 = call String::as_str(_7)     // a &str borrowing _2, produced in an EARLIER block
/// _9 = call send_once(_8, move _2.2, move _2.3)
/// ```
///
/// `_8` is an ordinary non-slot local, so nothing about it says it borrows `_2` — but it does, and
/// a thunk taking `&mut _2` beside it is `E0502` again. This traces where a reference could have
/// come from: `RefOf` seeds it, copies and borrow-carrying aggregates propagate it (OWN-CARRY-001
/// makes provenance structural), and a call's result inherits the provenance of its arguments,
/// which is exactly STARK's own shortest-input rule read as a may-alias relation.
///
/// Over-approximate on purpose. Being wrong in this direction refuses a program that might have
/// compiled; being wrong in the other emits code rustc rejects for reasons the user cannot act on.
fn borrow_provenance(
    body: &crate::mir::MirBody,
    env: &emit_places::TyEnv,
) -> Result<BTreeMap<u32, std::collections::BTreeSet<u32>>, BackendDiagnostic> {
    use crate::mir::{Rvalue, Statement, Terminator};
    use std::collections::BTreeSet;

    let mut prov: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    let mut slot_seed: Vec<(u32, u32)> = Vec::new();
    for block in &body.blocks {
        for (statement, _) in &block.statements {
            if let Statement::Assign(dest, Rvalue::RefOf { place, .. }) = statement {
                if dest.projection.is_empty() && emit_places::is_slot_local(place.local.0, env)? {
                    slot_seed.push((dest.local.0, place.local.0));
                }
            }
        }
    }
    for (dest, slot) in slot_seed {
        prov.entry(dest).or_default().insert(slot);
    }

    // A tiny fixpoint. Bodies are small and the relation only grows, so this terminates in a couple
    // of rounds; iterating to a fixpoint rather than once matters because MIR block order is not
    // definition order.
    let mut changed = true;
    while changed {
        changed = false;
        let mut merge = |into: u32, from: &[u32], prov: &mut BTreeMap<u32, BTreeSet<u32>>| {
            let mut union: BTreeSet<u32> = BTreeSet::new();
            for local in from {
                if let Some(set) = prov.get(local) {
                    union.extend(set.iter().copied());
                }
            }
            if union.is_empty() {
                return;
            }
            let target = prov.entry(into).or_default();
            let before = target.len();
            target.extend(union);
            if target.len() != before {
                changed = true;
            }
        };
        for block in &body.blocks {
            for (statement, _) in &block.statements {
                let Statement::Assign(dest, rvalue) = statement else {
                    continue;
                };
                if !dest.projection.is_empty() {
                    continue;
                }
                let sources: Vec<u32> = match rvalue {
                    Rvalue::RefOf { place, .. } => vec![place.local.0],
                    Rvalue::Use(Operand::Copy(p) | Operand::Move(p)) => vec![p.local.0],
                    // A borrow-carrying aggregate: `Option<&T>`, a tuple of references. The value
                    // carries every borrow its components carried.
                    Rvalue::Aggregate(_, operands) => operands
                        .iter()
                        .filter_map(|o| match o {
                            Operand::Copy(p) | Operand::Move(p) => Some(p.local.0),
                            Operand::Const(_) => None,
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                merge(dest.local.0, &sources, &mut prov);
            }
            // A call's result may borrow anything its arguments borrowed -- STARK's shortest-input
            // rule (03 rule 3) read as a may-alias relation. `Checked` is arithmetic and returns a
            // scalar, so it carries nothing.
            if let Terminator::Call { args, dest, .. } = &block.terminator.0 {
                let sources: Vec<u32> = args
                    .iter()
                    .filter_map(|o| match o {
                        Operand::Copy(p) | Operand::Move(p) => Some(p.local.0),
                        Operand::Const(_) => None,
                    })
                    .collect();
                merge(dest.local.0, &sources, &mut prov);
            }
        }
    }
    Ok(prov)
}

/// A borrow of a slot-backed field on its way to a call argument, and the statements that build it.
#[derive(Clone, Debug)]
pub struct Borrow {
    /// The place actually borrowed — always a projection of a slot-backed local.
    pub source: Place,
    /// Indices, within the call's block, of the statements the thunk takes over.
    pub statements: Vec<usize>,
}

/// How many times a local is READ across the whole body. Definitions, `StorageDead` and
/// `StorageWhole` do not count: they are bookkeeping over the local, not uses of its value.
fn reads_of(body: &crate::mir::MirBody, local: u32) -> u32 {
    use crate::mir::{Rvalue, Statement, Terminator};
    let mut count = 0;
    fn reads(operand: &Operand, local: u32) -> u32 {
        match operand {
            Operand::Copy(p) | Operand::Move(p) if p.local.0 == local => 1,
            _ => 0,
        }
    }
    for block in &body.blocks {
        for (statement, _) in &block.statements {
            let Statement::Assign(dest, rvalue) = statement else {
                continue;
            };
            // A projected destination READS its base to reach the field.
            if dest.local.0 == local && !dest.projection.is_empty() {
                count += 1;
            }
            for operand in emit_projections::rvalue_operands(rvalue) {
                count += reads(operand, local);
            }
            if let Rvalue::RefOf { place, .. } | Rvalue::Discriminant(place) = rvalue {
                if place.local.0 == local {
                    count += 1;
                }
            }
        }
        match &block.terminator.0 {
            Terminator::Call { args, .. } | Terminator::Checked { args, .. } => {
                for arg in args {
                    count += reads(arg, local);
                }
            }
            Terminator::SwitchInt { scrut, .. } => count += reads(scrut, local),
            Terminator::Drop { place, .. } => {
                if place.local.0 == local {
                    count += 1;
                }
            }
            Terminator::Trap {
                message: Some(message),
                ..
            } => count += reads(message, local),
            _ => {}
        }
    }
    count
}

/// Whether ordinary emission would produce an `&`/`&mut` conflict on one slot.
fn conflicts(
    args: &[Operand],
    borrows: &BTreeMap<u32, Borrow>,
    escaping: &BTreeMap<u32, Borrow>,
    env: &emit_places::TyEnv,
) -> Result<bool, BackendDiagnostic> {
    let mut seen: BTreeMap<u32, (u32, bool)> = BTreeMap::new();
    for arg in args {
        let (Operand::Move(place) | Operand::Copy(place)) = arg else {
            continue;
        };
        // An absorbable reference temporary counts as a SHARED access to the slot it borrows. This
        // is the `&` half of `f(&p.name, p.body)`, and without it the argument list looks like a
        // single access to `p` and no thunk is planned — while rustc still sees the live borrow.
        if let Some(borrow) = borrows
            .get(&place.local.0)
            .or_else(|| escaping.get(&place.local.0))
        {
            let entry = seen.entry(borrow.source.local.0).or_insert((0, false));
            entry.0 += 1;
            continue;
        }
        if !emit_places::is_slot_local(place.local.0, env)? {
            continue;
        }
        // A MOVE takes `&mut` in the reference form, whether it is a whole-local `take()` or a
        // projected `move_field`. Everything else reads through `&`.
        let entry = seen.entry(place.local.0).or_insert((0, false));
        entry.0 += 1;
        entry.1 |= matches!(arg, Operand::Move(_));
    }
    Ok(seen
        .values()
        .any(|(count, needs_mut)| *count >= 2 && *needs_mut))
}

fn slot_index(
    slots: &mut Vec<ThunkSlot>,
    place: &Place,
    env: &emit_places::TyEnv,
) -> Result<usize, BackendDiagnostic> {
    if let Some(index) = slots.iter().position(|s| s.local == place.local.0) {
        return Ok(index);
    }
    slots.push(ThunkSlot {
        local: place.local.0,
        ty: env.local_ty(place.local.0)?,
    });
    Ok(slots.len() - 1)
}

/// Which raw twin an operand needs, and whether its result is a reference.
fn projection_op(
    arg: &Operand,
    place: &Place,
    env: &emit_places::TyEnv,
) -> Result<(HelperOp, bool), BackendDiagnostic> {
    let field_ty = env.place_ty(place)?;
    let field_is_copy = emit_types::mir_ty_is_copy(&field_ty, env.types);
    Ok(match (arg, field_is_copy) {
        // Moving a `Copy` field is a read: nothing changes liveness.
        (Operand::Move(_), true) | (Operand::Copy(_), true) => (HelperOp::CopyRaw, false),
        (Operand::Move(_), false) => (HelperOp::MoveRaw, false),
        (Operand::Copy(_), false) => (HelperOp::RefRaw, true),
        (Operand::Const(_), _) => unreachable!("constants are handled before projection"),
    })
}

/// Register the raw twin the plan needs, and return its name. The plan carries the helper, so
/// nothing downstream rediscovers it.
fn collect_raw_helper(
    place: &Place,
    env: &emit_places::TyEnv,
    op: HelperOp,
    found: &mut BTreeMap<String, ProjectionHelper>,
) -> Result<String, BackendDiagnostic> {
    let before = found.len();
    emit_projections::collect_place_pub(place, env, op, found)?;
    debug_assert!(found.len() >= before);
    // `collect_place` keys by helper name, so the one just inserted is the one for this place.
    Ok(emit_projections::helper_name(
        &env.local_ty(place.local.0)?,
        &place.projection,
        op,
    ))
}

/// Render the thunk into `mod stark_proj`.
///
/// The single lifetime `'a` is what makes every reference the thunk hands on sound: each slot
/// arrives through a real `&'a mut ValueSlot<T>`, the raw pointer is derived from it, and any
/// reference produced by a `Ref` twin borrows for `'a` — which outlives the thunk's return, so a
/// reference-carrying result keeps its provenance.
pub fn emit_thunk(plan: &CallThunkPlan) -> Result<String, BackendDiagnostic> {
    let mut params: Vec<String> = Vec::new();
    for (index, slot) in plan.slots.iter().enumerate() {
        params.push(format!(
            "s{index}: &'a mut stark_runtime::slot::ValueSlot<{}>",
            emit_types::emit_ty(&slot.ty)?
        ));
    }
    for (index, value) in plan.by_value.iter().enumerate() {
        params.push(format!("v{index}: {}", emit_types::emit_ty(&value.ty)?));
    }

    let mut body = String::new();
    // Every raw pointer is derived BEFORE any field reference exists, and no `&ValueSlot` or
    // `&mut ValueSlot` is reconstructed afterwards — the aliasing rule the primitives require.
    for index in 0..plan.slots.len() {
        body.push_str(&format!(
            "            let p{index}: *mut stark_runtime::slot::ValueSlot<{}> = s{index};\n",
            emit_types::emit_ty(&plan.slots[index].ty)?
        ));
    }

    let mut arg_names: Vec<String> = Vec::new();
    let mut evaluated = String::new();
    for (index, arg) in plan.args.iter().enumerate() {
        let name = format!("a{index}");
        let expr = match arg {
            ThunkArg::Projection {
                slot,
                helper,
                borrows,
            } => {
                if *borrows {
                    format!("{helper}::<'a>(p{slot})")
                } else {
                    format!("{helper}(p{slot})")
                }
            }
            ThunkArg::Take { slot } => format!("stark_runtime::slot::ValueSlot::take_raw(p{slot})"),
            ThunkArg::ByValue { index } => format!("v{index}"),
        };
        evaluated.push_str(&format!("            let {name} = {expr};\n"));
        arg_names.push(name);
    }

    Ok(format!(
        "\n\
         \x20   /// DEV-160: one call, several disjoint accesses to one slot.\n\
         \x20   ///\n\
         \x20   /// Ordinary borrowing cannot express this -- each projection wrapper borrows the\n\
         \x20   /// WHOLE slot, so a read that lives into the call and a sibling move conflict.\n\
         \x20   /// Here the slot arrives once through `&'a mut`, the raw pointer is derived from\n\
         \x20   /// it, and every operand is evaluated IN MIR ORDER inside. Argument order is\n\
         \x20   /// language semantics (CD-007) and is not rearranged to suit the borrow checker.\n\
         \x20   pub fn {name}<'a>({params}) -> {ret} {{\n\
         {derived}\
         \x20       // SAFETY: every projection is fixed in generated code, and MIR proved the\n\
         \x20       // accesses disjoint and each moved unit live. No reference to a whole slot is\n\
         \x20       // formed after a field reference exists, and `'a` comes from the slot\n\
         \x20       // parameters, so every borrow handed on outlives the call.\n\
         \x20       unsafe {{\n\
         {evaluated}\
         \x20           {callee}({args})\n\
         \x20       }}\n\
         \x20   }}\n",
        name = plan.name,
        params = params.join(", "),
        ret = emit_types::emit_ty(&plan.ret_ty)?,
        derived = body,
        evaluated = evaluated,
        callee = plan.callee,
        args = arg_names.join(", "),
    ))
}

/// The call-site expression: one safe call, no `unsafe` in the generated MIR body.
pub fn emit_call_site(plan: &CallThunkPlan) -> String {
    let mut args: Vec<String> = plan
        .slots
        .iter()
        .map(|slot| format!("&mut {}", emit_places::local_name(slot.local)))
        .collect();
    args.extend(plan.by_value.iter().map(|v| v.expr.clone()));
    format!("stark_proj::{}({})", plan.name, args.join(", "))
}
