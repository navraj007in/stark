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
//! # Scope (owner ruling, 2026-08-03; b closed 2026-08-12)
//!
//! ```text
//! DEV-160a  same-block direct-call disjoint projections      CLOSED -- this module
//! DEV-160b  borrow returned by an EARLIER call               CLOSED -- absorbed producer,
//!                                                            WP-ARCH-CLOSE AC1 step 2
//! DEV-160c  conflicting provider-call argument sequence      refused by name; DEFERRED
//! DEV-160d  borrow surviving beyond the sibling move/call    refused by name; DEFERRED
//! ```
//!
//! c and d remain **over-refusals, not unsound execution**. Each is named before rustc sees the
//! program, because the alternative is `E0502` inside this generated module — a correct compiler
//! error about code the user never wrote.
//!
//! # How b was closed, and why the cheaper repair was rejected
//!
//! The borrow reaches the call from an earlier block because a call is a MIR terminator:
//! `r.url.as_str()` ends one block and `send(…)` begins the next. The small repair would leave the
//! producing call where it is and launder its result through a raw pointer at the call site.
//!
//! **That is unsound**, and `slot.rs` states the reason directly: the thunk's `&'a mut` is *"what
//! anchors every reference it hands on"*. Under Stacked Borrows, taking that `&mut` invalidates
//! tags derived from any earlier borrow of the slot, so a reference created before the thunk was
//! entered is dead inside it however it travelled. It must be created INSIDE — which brings the
//! producing call with it. See [`absorbable_producer`] for the six admission conditions.
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
    /// **DEV-160b: a value produced by a call the thunk absorbed from an EARLIER block.**
    ///
    /// `send(r.url.as_str(), r.body)` lowers with `as_str` as the terminator of the preceding
    /// block, so its `&str` result is a live borrow of `r` when the thunk takes `&mut`. It cannot
    /// be laundered through a raw pointer at the call site: under Stacked Borrows the thunk's
    /// `&'a mut` invalidates tags derived from any earlier borrow of the same slot, so reading it
    /// inside would be UB. The reference has to be CREATED inside, anchored by the thunk's own
    /// `&'a mut` — which means the producing call moves in with it.
    ///
    /// `expr` is rendered at plan time and already names the thunk's `p{slot}` pointers.
    Produced { expr: String },
}

/// A call in an earlier block that a thunk performs itself (DEV-160b).
///
/// `emit_bodies` consumes this from the PRODUCER's side: it suppresses the named statements and
/// emits a plain `goto` in place of the call terminator the thunk has taken over.
#[derive(Clone, Debug)]
pub struct AbsorbedProducer {
    /// The block whose call terminator this thunk performs instead.
    pub block_index: u32,
    /// Statement indices in THAT block the thunk performs instead — the `RefOf` seeding the
    /// producer's reference argument, and any copy carrying it.
    pub statements: Vec<usize>,
    /// What that block's terminator becomes. It is the consuming call's own block, because
    /// absorption is admitted only across a single straight-line edge.
    pub goto_target: u32,
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
    /// DEV-160b: a producer call in an earlier block this thunk performs itself.
    pub absorbed_producer: Option<AbsorbedProducer>,
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
            if let Some(plan) = plan_for_call(body, block_index, &env, &program.sources)? {
                plans.push(plan);
            }
        }
    }
    Ok(plans)
}

/// The producer-side view: is THIS block's call terminator absorbed by some thunk (DEV-160b)?
///
/// Looked up by the producer's own block index, because `emit_bodies` asks the question from there
/// — it is emitting the block whose call has been taken over and needs to know to emit a `goto`
/// instead. The plan that owns the absorption lives on the CONSUMING block.
pub fn producer_absorbed_at<'p>(
    plans: &'p [CallThunkPlan],
    body_symbol: &str,
    block_index: u32,
) -> Option<&'p AbsorbedProducer> {
    plans.iter().find_map(|p| {
        p.absorbed_producer
            .as_ref()
            .filter(|a| a.block_index == block_index && p.body_symbol == body_symbol)
    })
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
    sources: &crate::source::SourceTable,
) -> Result<Option<CallThunkPlan>, BackendDiagnostic> {
    let block = &body.blocks[block_index as usize];
    let crate::mir::Terminator::Call {
        callee, args, dest, ..
    } = &block.terminator.0
    else {
        return Ok(None);
    };
    let (borrows, escaping) = absorbable_borrows(body, block_index, args, env)?;
    // **The borrow-origin relation comes from MIR, not from here** (CE3, 2026-08-12). This module
    // used to compute a `may derive from` over-approximation of its own; the analysis now lives at
    // `crate::mir::borrows`, which owns lowered form and can state the relation exactly.
    let call_provenance = crate::mir::borrows::origins(body);
    if !conflicts(args, &borrows, &escaping, &call_provenance, env)? {
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
             (A10/CD-200), so the call thunk does not apply to it. Bind the fields to locals \
             before the call. DEV-160c, deferred to its own work package",
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
             conflict can only take over a borrow whose every use is an argument of the one call. \
             DEV-160d, deferred to its own work package",
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
    let mut by_value_provenance: Vec<(u32, usize, usize)> = Vec::new();

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
            by_value_provenance.push((place.local.0, plan_args.len(), by_value.len()));
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
    let provenance = &call_provenance;
    let participating: std::collections::BTreeSet<u32> = slots.iter().map(|s| s.local).collect();
    let mut absorbed_producer: Option<AbsorbedProducer> = None;
    for (local, plan_arg_index, by_value_index) in &by_value_provenance {
        // **No type check here any more, and its absence is the point.** This used to read
        // `if !may_carry_borrow(&by_value[index].ty) { continue; }` — a consumer compensating for
        // an authority that over-approximated, because the heuristic recorded
        // `consume(p.taken, p.kept.len())`'s `UInt64` result as borrowing `_1`.
        // `crate::mir::borrows` now states that invariant itself: a value that cannot store a
        // reference has no origins at all, so there is nothing left to filter.
        let Some(borrowed) = provenance.of(*local) else {
            continue;
        };
        if let Some(slot) = borrowed.intersection(&participating).next() {
            // **DEV-160b: try to absorb the call that produced this reference.** The reference
            // cannot stay where it is — it is a live borrow of the slot the thunk is about to take
            // exclusively — and it cannot be laundered through a raw pointer, because Stacked
            // Borrows invalidates tags derived from an earlier borrow once the `&'a mut` exists.
            // Creating it inside, anchored by the thunk's own `&mut`, is the only sound placement,
            // and that means the producing call comes with it.
            //
            // Only ONE producer may be absorbed per thunk: two would each need their own edge from
            // their own predecessor, and a block has one predecessor here by condition 3.
            if absorbed_producer.is_none() {
                if let Some((expr, producer)) = absorbable_producer(
                    body,
                    block_index,
                    *local,
                    &participating,
                    &mut slots,
                    &mut helpers,
                    env,
                    sources,
                )? {
                    plan_args[*plan_arg_index] = ThunkArg::Produced { expr };
                    // **The by-value entry must go with it.** The thunk's parameter list and its
                    // call site are both built from `by_value`; leaving the entry behind emits a
                    // thunk that still TAKES the reference as an argument and a call site that
                    // still passes it — which is the borrow this absorption exists to remove, and
                    // it read as `_9.unwrap()` on a local nothing assigns any more.
                    by_value.remove(*by_value_index);
                    for arg in plan_args.iter_mut() {
                        if let ThunkArg::ByValue { index } = arg {
                            if *index > *by_value_index {
                                *index -= 1;
                            }
                        }
                    }
                    absorbed_producer = Some(producer);
                    continue;
                }
            }
            // **The message describes why absorption DECLINED, not a missing mechanism.**
            // Before DEV-160b closed it said the thunk "can only take over evaluation within the
            // call's OWN block" and that the case was deferred. Both became false the moment
            // absorption landed, and a refusal that misdescribes its own cause sends the reader
            // looking for the wrong thing.
            return Err(BackendDiagnostic::Unsupported(format!(
                "the call in bb{block_index} of `{}` passes a reference (_{local}) that borrows \
                 _{slot} while also moving out of _{slot}'s fields. STARK accepts this -- the \
                 accesses are disjoint -- and the backend absorbs the producing call where it \
                 can, but this one does not qualify: absorption needs the reference to come from \
                 a call in the single immediately-preceding block, to be read exactly once, and \
                 for every argument of that call to be a borrow of a slot this thunk already \
                 holds. Only one such producer may be absorbed per call. Binding the fields to \
                 locals before the call avoids the conflict entirely. See DEV-160's admission \
                 conditions",
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
        absorbed_producer,
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

/// **DEV-160b's admission test: can this thunk take over the call that produced `local`?**
///
/// The conditions are deliberately all-or-nothing. Every one of them is a way the absorbed call
/// could observe something different from where it was, and a partial absorption that got any of
/// them wrong would move a side effect rather than a borrow.
///
/// ```text
/// 1  `local` is defined by a Call TERMINATOR, in some other block
/// 2  that block's call targets THIS block -- one edge, no intervening blocks
/// 3  this block has exactly ONE predecessor, so the producer always runs immediately before
/// 4  `local` is read exactly once: by the consuming call. A second read would still need the
///    value where it was
/// 5  every producer argument is either inert (a constant) or an absorbable borrow of a
///    PARTICIPATING slot, read exactly once
/// 6  the producer's callee is Runtime or a direct Instance -- the two this can render
/// ```
///
/// Condition 3 is what makes "straight-line" true rather than assumed: with one predecessor there
/// is no path that reaches the consumer without the producer, and none that reaches the producer
/// twice per consumer.
///
/// Returns the inner argument plan and the statements to suppress in the producer's block.
#[allow(clippy::too_many_arguments)]
fn absorbable_producer(
    body: &crate::mir::MirBody,
    block_index: u32,
    local: u32,
    participating: &std::collections::BTreeSet<u32>,
    slots: &mut Vec<ThunkSlot>,
    helpers: &mut BTreeMap<String, ProjectionHelper>,
    env: &emit_places::TyEnv,
    sources: &crate::source::SourceTable,
) -> Result<Option<(String, AbsorbedProducer)>, BackendDiagnostic> {
    use crate::mir::Terminator;

    // 4. Read exactly once — by the call we are planning.
    if reads_of(body, local) != 1 {
        return Ok(None);
    }

    // 1 + 2. The defining call terminator, whose target is this block.
    let mut producer: Option<u32> = None;
    for (index, block) in body.blocks.iter().enumerate() {
        if let Terminator::Call { dest, target, .. } = &block.terminator.0 {
            if dest.local.0 == local && dest.projection.is_empty() && target.0 == block_index {
                producer = Some(index as u32);
            }
        }
    }
    let Some(producer_index) = producer else {
        return Ok(None);
    };

    // 3. Exactly one predecessor.
    let predecessors = body
        .blocks
        .iter()
        .filter(|b| terminator_targets(&b.terminator.0).contains(&block_index))
        .count();
    if predecessors != 1 {
        return Ok(None);
    }

    let producer_block = &body.blocks[producer_index as usize];
    let Terminator::Call { callee, args, .. } = &producer_block.terminator.0 else {
        return Ok(None);
    };

    // 5. Trace each producer argument. A borrow must resolve to a projection of a PARTICIPATING
    //    slot, or the thunk has no pointer to rebuild it from.
    let (borrows, escaping) = absorbable_borrows(body, producer_index, args, env)?;
    if !escaping.is_empty() {
        return Ok(None);
    }
    let mut inner: Vec<String> = Vec::new();
    let mut statements: Vec<usize> = Vec::new();
    for arg in args {
        let (Operand::Move(place) | Operand::Copy(place)) = arg else {
            inner.push(super::emit_bodies::emit_operand(arg, env)?);
            continue;
        };
        let Some(borrow) = borrows.get(&place.local.0) else {
            // Not a borrow this thunk can rebuild. A by-value operand here would be evaluated at
            // the wrong point relative to the absorbed projections, so absorption is declined
            // rather than approximated.
            return Ok(None);
        };
        if !participating.contains(&borrow.source.local.0) {
            return Ok(None);
        }
        let slot = slot_index(slots, &borrow.source, env)?;
        let helper = collect_raw_helper(&borrow.source, env, HelperOp::RefRaw, helpers)?;
        inner.push(format!("{helper}::<'a>(p{slot})"));
        for index in &borrow.statements {
            if !statements.contains(index) {
                statements.push(*index);
            }
        }
    }

    // 6. Render the producer call itself.
    let dest_ty = body.locals[local as usize].ty.clone();
    let (file, line, col) =
        super::emit_bodies::source_location_of(sources, &producer_block.terminator.1);
    let site = super::emit_runtime::CallSite { file, line, col };
    let expr = match callee {
        Callee::Runtime(rt) => {
            super::emit_runtime::emit_runtime_call(*rt, &inner, &dest_ty, &site, None)?
        }
        Callee::Instance(instance) => {
            let name = super::mangle::function_name_for_symbol(&instance.symbol);
            format!("{name}({})", inner.join(", "))
        }
        _ => return Ok(None),
    };

    Ok(Some((
        expr,
        AbsorbedProducer {
            block_index: producer_index,
            statements,
            goto_target: block_index,
        },
    )))
}

/// Successor block indices of a terminator. A local twin of `emit_bodies::terminator_successors`,
/// which is private there; kept tiny and total rather than widening that module's surface.
fn terminator_targets(t: &crate::mir::Terminator) -> Vec<u32> {
    use crate::mir::Terminator;
    match t {
        Terminator::Goto { target }
        | Terminator::Call { target, .. }
        | Terminator::Drop { target, .. }
        | Terminator::Checked { target, .. } => vec![target.0],
        Terminator::SwitchInt {
            arms, otherwise, ..
        } => {
            let mut v: Vec<u32> = arms.iter().map(|(_, b)| b.0).collect();
            v.push(otherwise.0);
            v
        }
        Terminator::Return | Terminator::Unreachable | Terminator::Trap { .. } => Vec::new(),
    }
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
    provenance: &crate::mir::borrows::BorrowOrigins,
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
            // As above: the type filter this used to carry is now the analysis's own invariant.
            if let Some(sources) = provenance.of(place.local.0) {
                for source in sources {
                    seen.entry(*source).or_insert((0, false)).0 += 1;
                }
            }
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
            // DEV-160b. Rendered at plan time and already naming this thunk's `p{slot}` pointers,
            // so the reference it derives is anchored by the `&'a mut` above rather than by a
            // borrow that existed before the thunk was entered.
            ThunkArg::Produced { expr } => expr.clone(),
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
