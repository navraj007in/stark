//! WP-C4.4/C4.5b-2 — MIR interpreter.
//!
//! Executes a **verified** `MirProgram` for differential validation against the HIR
//! interpreter (the semantic oracle, charter §1.6 rule 6). This is NOT a user-facing VM
//! (charter §1.6 rule 11) — its sole purpose is the contract's observable comparator:
//! for each frozen workload, `HIR interpreter output/failure == MIR interpreter
//! output/failure`.
//!
//! Execution model (C4.5b-2 frame restructure):
//! - An explicit frame stack (`frames: Vec<Frame>`) replaces call-local storage, so a
//!   **reference value can point into a caller's frame** — the semantic requirement behind
//!   `&mut self` and `&x` arguments. A reference is a resolved path
//!   `(frame index, local, concrete projections)`; `Deref` re-anchors place resolution to the
//!   referent. Index proofs always resolve in the frame *evaluating* the place, before any
//!   re-anchoring.
//! - Borrow checking upstream guarantees no reference outlives its frame in legal programs;
//!   the interpreter still guards every frame access and reports a loud `Internal` error on a
//!   dangling path (defense in depth, never silent misbehavior).
//! - `Checked` terminators implement STARK trap semantics per integer width (overflow,
//!   divide-by-zero incl. `MIN / -1`, float div/rem by zero per CD-006, checked casts,
//!   CheckIndex bounds). A trap ABORTS with category AND provenance (CD-029) — no unwinding.
//! - `Move` operands *take* whole-local values (poisoning the slot) so verifier-missed
//!   use-after-move explodes loudly; projected/deref reads copy.
//! - `Drop` terminators (C4.5d) run recursive drop glue: the type's own destructor instance
//!   through an `&mut` reference (mutations stay visible to what follows), then fields or the
//!   runtime-discriminant variant payload in reverse declaration order; whole-local drops
//!   poison the slot afterwards.
//! - Float printing calls `crate::interp::canonical_float` — the oracle's own formatter
//!   (shared by design; compensating spec tests live in `tests/canonical_float.rs`).
//! - A fuel guard turns runaway loops from lowering bugs into clean internal errors.

use super::drop_plan::{self, DropPlan};
use super::*;
use std::collections::HashMap;
use std::fmt::Write as _;

/// A fully resolved projection step (no `Deref`, no proof locals — both are resolved away
/// during place resolution).
#[derive(Clone, Debug, PartialEq)]
pub enum ConcreteProj {
    Field(u32),
    Variant(u32, u32),
    Index(usize),
    /// 0.1-A6 (A4 slicing): a sub-range view over an Array/Vec referent — `start` is the
    /// absolute element offset in the underlying container, `len` the view length. Appears
    /// only in `MirValue::Ref` paths (a `&[T]` value); re-slicing composes into one step.
    Slice {
        start: usize,
        len: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum MirValue {
    Int(i128),
    Float(f64),
    Bool(bool),
    Unit,
    /// Struct/tuple/array payloads (field order = declaration/element order).
    Aggregate(Vec<MirValue>),
    Enum {
        variant: u32,
        fields: Vec<MirValue>,
    },
    /// Index into `MirProgram::bodies`.
    FnPtr(usize),
    /// A reference: a resolved path into some live frame (C4.5b-2). C4.5f-1: carries the
    /// pointee frame's **generation** — frame slots are reused across calls, and a slot
    /// index alone would let a stale reference silently alias a later frame (CD-030 review
    /// finding). Every deref validates `frames[frame].generation == generation` and fails
    /// loudly on mismatch.
    Ref {
        frame: usize,
        generation: u64,
        local: u32,
        path: Vec<ConcreteProj>,
    },
    /// A `&str` value (A1). Self-contained: a literal, or a read-only snapshot produced by
    /// `StringAsStr` (§5b — sound because the view is read-only and str identity is
    /// unobservable). `Rc` is an unobservable cheap-copy convenience.
    Str(std::rc::Rc<str>),
    /// A read-only `&[UInt8]` snapshot produced by `str.bytes()`. General slices remain
    /// place windows (`Ref` + `ConcreteProj::Slice`); string bytes have no UInt8 element place in
    /// MIR, so they use a self-contained immutable value.
    ByteSlice(std::rc::Rc<[u8]>),
    /// An owned `String` value (A1). Non-Copy; drop-elaborated (buffer reclaim is a no-op).
    String(String),
    /// An owned `Vec<T>` value (A1/C4.5e-2). Non-Copy; drop-elaborated (elements dropped in
    /// reverse index order, then buffer reclaimed — §5a). Opaque to projections: manipulated
    /// only through the Vec `RuntimeFn` surface.
    Vec(Vec<MirValue>),
    /// C4.5f-1 (CD-030 deferral): the poison a projected `Move` leaves behind. Any read of a
    /// `Moved` hole is a loud internal error — verifier-missed use-after-partial-move can
    /// never silently observe the retained value (previously projected moves cloned, relying
    /// on the verifier alone).
    Moved,
}

#[derive(Debug)]
pub enum MirRunError {
    /// A language trap: category AND provenance (CD-029 — category alone made the
    /// differential blind to wrong-location traps) AND the resolved user message when one
    /// exists (A1/CD-031 — `panic(msg)` carries it; compiler-generated traps carry `None`).
    Trap {
        category: TrapCategory,
        source: SourceInfo,
        message: Option<String>,
    },
    /// A bug in lowering/verification/interpretation — never a language-level outcome.
    Internal(String),
}

pub struct MirExecution {
    pub output: String,
    pub status: u8,
    /// DEV-111: bytes destined for the Core stderr stream on normal `Err` completion —
    /// PROC-EXIT-001's *"`Err(message)` writes `message` plus LF to stderr and returns status 1"*.
    /// Until DEV-111 this engine had no stderr channel at all, so the whole `Err` half of the
    /// entry contract was silently unobservable here while the HIR oracle implemented it. The
    /// oracle's `Execution` has carried the same field since Phase 4E; this is the MIR counterpart,
    /// deliberately named and shaped identically so the comparator can compare them directly.
    ///
    /// `eprint`/`eprintln` do NOT flow here — they have no MIR lowering at all (their oracle
    /// implementation writes to the host process's stderr), which is recorded as an open channel
    /// gap in `C6-CORPUS-COVERAGE-MATRIX.md` §8 rather than papered over.
    pub stderr: String,
}

/// A failed run, still carrying the stdout accumulated before the failure (C4.5e-0): the
/// roadmap's comparator is output AND failure equality, so pre-trap output is observable.
#[derive(Debug)]
pub struct MirFailure {
    pub error: MirRunError,
    pub output: String,
}

const FUEL: u64 = 50_000_000;

struct Frame {
    locals: Vec<Option<MirValue>>,
    /// C4.5f-1: unique, monotonically assigned identity — never reused, unlike the slot index.
    generation: u64,
}

pub fn run_program(
    verified: crate::mir::verify::VerifiedMirProgram<'_>,
) -> Result<MirExecution, MirFailure> {
    let program = verified.program();
    let by_symbol: HashMap<&str, usize> = program
        .bodies
        .iter()
        .enumerate()
        .map(|(i, b)| (b.instance.symbol.as_str(), i))
        .collect();
    let Some(&main_index) = by_symbol.get("main@[]") else {
        return Err(MirFailure {
            error: MirRunError::Internal("no main@[] instance".to_string()),
            output: String::new(),
        });
    };
    let mut cx = Interp {
        program,
        by_symbol,
        frames: Vec::new(),
        next_generation: 0,
        output: String::new(),
        fuel: FUEL,
        layout: crate::layout::TargetLayout::default(),
        drop_plans: std::collections::BTreeMap::new(),
    };
    match cx.call(main_index, Vec::new()) {
        Ok(value) => match entry_termination(value) {
            Ok((status, stderr)) => Ok(MirExecution {
                output: cx.output,
                status,
                stderr,
            }),
            Err(error) => Err(MirFailure {
                error,
                output: cx.output,
            }),
        },
        Err(error) => Err(MirFailure {
            error,
            output: cx.output,
        }),
    }
}

/// PROC-EXIT-001, on the value `main` returned. **DEV-111**: this engine used to match `Ok(_)` and
/// report status 0 unconditionally, discarding the entry's return value — so `fn main() -> Int32 { 3 }`
/// completed with status 0 here and status 3 in the HIR oracle, and an `Err` return lost both its
/// status and its stderr write. Three engines, three answers, on a signature PROC-MAIN-001 declares
/// a legal executable target.
///
/// The rule, quoted: *"Normal `Unit` and `Ok(Unit)` return status 0. `Int32` and `Ok(Int32)` must be
/// in `0..=255` and return that status; an out-of-range value traps as `invalid-exit-status`.
/// `Err(message)` writes `message` plus LF to stderr and returns status 1."* The mapping below is
/// the same one `interp::main_result_to_status` applies, so the two engines derive their termination
/// from one reading of the rule rather than two.
///
/// `Result` is `EnumRef::CoreResult` with `Ok` = variant 0 and `Err` = variant 1 (lowering, the
/// `Builtin::Ok`/`Builtin::Err` constructors). `MirValue::Enum` does not carry its `EnumRef`, but it
/// does not need to: PROC-MAIN-001 admits exactly `Unit`, `Int32`, `Result<Unit, String>` and
/// `Result<Int32, String>` as entry types, and the checker rejects everything else before lowering.
fn entry_termination(value: MirValue) -> Result<(u8, String), MirRunError> {
    fn status_of(value: MirValue) -> Result<u8, MirRunError> {
        match value {
            MirValue::Unit => Ok(0),
            MirValue::Int(status) => u8::try_from(status).map_err(|_| {
                // CD-150 CE3, MIR amendment A7: PROC-EXIT-001 makes this a language TRAP, and the
                // category now exists. Provenance is the entry file at 1:1 — the contract is
                // violated by the entry's RESULT, not by an expression, so this is the one location
                // all three engines can report identically. DEV-111's `Internal` stopgap is gone.
                MirRunError::Trap {
                    category: TrapCategory::InvalidExitStatus,
                    source: SourceInfo {
                        file: FileId(0),
                        span: crate::source::Span::point(0),
                        origin: Origin::UserCode,
                    },
                    message: None,
                }
            }),
            other => Err(MirRunError::Internal(format!(
                "entry returned {other:?}, which PROC-MAIN-001 does not admit as an entry type"
            ))),
        }
    }

    match value {
        MirValue::Enum { variant: 0, fields } => Ok((
            status_of(fields.into_iter().next().unwrap_or(MirValue::Unit))?,
            String::new(),
        )),
        MirValue::Enum { variant: 1, fields } => match fields.into_iter().next() {
            Some(MirValue::String(message)) => Ok((1, format!("{message}\n"))),
            Some(MirValue::Str(message)) => Ok((1, format!("{message}\n"))),
            other => Err(MirRunError::Internal(format!(
                "entry error payload is {other:?}, not the String PROC-MAIN-001 requires"
            ))),
        },
        other => Ok((status_of(other)?, String::new())),
    }
}

struct Interp<'a> {
    program: &'a MirProgram,
    by_symbol: HashMap<&'a str, usize>,
    frames: Vec<Frame>,
    /// C4.5f-1: monotonic frame-generation counter (never reset, never reused).
    next_generation: u64,
    output: String,
    fuel: u64,
    /// WP-C5.3e: the named target layout contract this run answers `size_of`/`align_of` from.
    /// Replaces the C4 `reference_layout` stub, whose own doc recorded that it reported one
    /// machine word for every type until a real target contract existed.
    layout: crate::layout::TargetLayout,
    /// WP-C5.3d-1b: derived destruction plans, memoised per type.
    ///
    /// The plan is derived from the type context, not from the value, so it is the same on every
    /// execution of the same `Drop` — and a `Drop` inside a loop executes once per iteration. The
    /// walk this replaced was lazy and did its table lookups per drop; caching keeps the eager
    /// derivation from being a per-iteration cost. `Rc` so a cached plan can be handed to
    /// `run_drop_plan` without holding a borrow of `self`.
    drop_plans: std::collections::BTreeMap<MirTy, std::rc::Rc<DropPlan>>,
}

impl<'a> Interp<'a> {
    fn internal<T>(&self, message: impl Into<String>) -> Result<T, MirRunError> {
        Err(MirRunError::Internal(message.into()))
    }

    fn call(&mut self, body_index: usize, args: Vec<MirValue>) -> Result<MirValue, MirRunError> {
        let body = &self.program.bodies[body_index];
        let mut locals: Vec<Option<MirValue>> = vec![None; body.locals.len()];
        // Bind arguments by their declared `Param(i)` kind — param locals are NOT
        // contiguous when drop-flag locals are interleaved between them (C4.5d).
        let mut args = args.into_iter();
        for param in 0.. {
            let Some(slot) = body
                .locals
                .iter()
                .position(|decl| decl.kind == LocalKind::Param(param))
            else {
                break;
            };
            let Some(value) = args.next() else {
                return self.internal("argument count does not match callee param locals");
            };
            locals[slot] = Some(value);
        }
        if args.next().is_some() {
            return self.internal("argument count does not match callee param locals");
        }
        let generation = self.next_generation;
        self.next_generation += 1;
        self.frames.push(Frame { locals, generation });
        let result = self.run(body_index);
        self.frames.pop();
        result
    }

    /// C4.5f-1: a live-frame check by (slot, generation) — the defense-in-depth guarantee
    /// that a stale reference fails loudly instead of silently aliasing a reused slot.
    fn check_ref_live(&self, frame: usize, generation: u64) -> Result<(), MirRunError> {
        match self.frames.get(frame) {
            Some(f) if f.generation == generation => Ok(()),
            Some(f) => self.internal(format!(
                "dangling reference: frame slot {frame} was reused (generation {} != {})",
                f.generation, generation
            )),
            None => self.internal(format!("dangling reference: frame {frame} no longer live")),
        }
    }

    fn run(&mut self, body_index: usize) -> Result<MirValue, MirRunError> {
        let body = &self.program.bodies[body_index];
        let here = self.frames.len() - 1;
        let mut block = body.entry;
        loop {
            if self.fuel == 0 {
                return self.internal("fuel exhausted (runaway loop in lowered MIR?)");
            }
            self.fuel -= 1;

            let bb = &body.blocks[block.0 as usize];
            for (stmt, _) in &bb.statements {
                match stmt {
                    Statement::Assign(place, rvalue) => {
                        let value = self.eval_rvalue(here, rvalue)?;
                        // CD-140: round a `Float32` destination to binary32. This engine carries
                        // every float in an f64, so without this an f64-precision result would be
                        // stored under a `Float32` type and only rounded when DISPLAYED —
                        // observable precision, not a rendering detail. The HIR oracle does the
                        // same thing at the same point (`normalize_numeric`, keyed on the
                        // expression's static type); here the destination's declared type is the
                        // equivalent authority. Every float rvalue reaches a typed destination, so
                        // this one site covers arithmetic, negation, and operand reads alike.
                        let ty = self.place_ty(body, place)?;
                        let value = narrow_to_declared_width(value, &ty);
                        self.write_place(here, place, value)?;
                    }
                    Statement::Nop => {}
                    // A12: this engine holds values in a map keyed by place, with no notion of
                    // partially moved storage, so there is no state for a storage end to correct.
                    // Deliberately inert rather than unimplemented — see `Statement::StorageDead`.
                    Statement::StorageDead(_) => {}
                }
            }
            match &bb.terminator.0 {
                Terminator::Goto { target } => block = *target,
                Terminator::SwitchInt {
                    scrut,
                    arms,
                    otherwise,
                } => {
                    let value = self.eval_operand(here, scrut)?;
                    let key: u128 = match value {
                        MirValue::Bool(b) => u128::from(b),
                        MirValue::Int(v) => v as u128, // same wrap as lowering's arm keys
                        other => {
                            return self
                                .internal(format!("SwitchInt on non-integer value {other:?}"))
                        }
                    };
                    block = arms
                        .iter()
                        .find(|(arm, _)| *arm == key)
                        .map(|(_, target)| *target)
                        .unwrap_or(*otherwise);
                }
                Terminator::Call {
                    callee,
                    args,
                    dest,
                    target,
                } => {
                    let mut values = Vec::new();
                    for arg in args {
                        values.push(self.eval_operand(here, arg)?);
                    }
                    let result = match callee {
                        // A10 (CD-200): provider calls do not execute here, and this is a
                        // permanent property rather than a C7.8.2a gap. The MIR interpreter is a
                        // pure semantic oracle with no provider linked into it; a host call has no
                        // meaning it could reproduce. Native execution is where providers run
                        // (Packet 1's statically linked path), so differential comparison for
                        // provider-backed programs is native-only by construction.
                        Callee::Provider(id) => {
                            let name = self
                                .program
                                .provider_call(*id)
                                .map(|c| c.symbol().to_string())
                                .unwrap_or_else(|| format!("<unresolved #{}>", id.0));
                            return self.internal(format!(
                                "provider call `{name}` cannot execute in the MIR interpreter: no \
                                 provider is linked into it (A10)"
                            ));
                        }
                        Callee::Instance(instance) => {
                            let Some(&idx) = self.by_symbol.get(instance.symbol.as_str()) else {
                                return self.internal(format!(
                                    "call to unknown instance {}",
                                    instance.symbol
                                ));
                            };
                            self.call(idx, values)?
                        }
                        Callee::FnValue(op) => {
                            let value = self.eval_operand(here, op)?;
                            let MirValue::FnPtr(idx) = value else {
                                return self.internal(format!("indirect call through {value:?}"));
                            };
                            self.call(idx, values)?
                        }
                        // A1: trap-capable runtime ops abort with the CALL SITE's SourceInfo
                        // as provenance (runtime ops carry no TrapInfo of their own, §5).
                        Callee::Runtime(rt) => {
                            // WP-C6.3d: a map op decides key identity with the KEY TYPE'S lawful
                            // `Eq` (STD-HASH-001). The instance is resolved HERE, where the call's
                            // operands and the enclosing body's local types are still in scope.
                            let key_eq = self.map_key_eq(body, args);
                            self.run_runtime(*rt, values, bb.terminator.1, key_eq)?
                        }
                    };
                    self.write_place(here, dest, result)?;
                    block = *target;
                }
                Terminator::Drop { place, target } => {
                    // C4.5d: run the place's drop glue — its own destructor instance (if
                    // the type has a Drop impl), then its fields/payload in reverse
                    // declaration order. A destructor that traps aborts (no unwind edge).
                    let ty = self.place_ty(body, place)?;
                    let (f, l, p) = self.resolve_place(here, place)?;
                    self.drop_in_place(f, l, p, &ty)?;
                    if place.projection.is_empty() {
                        // Whole-local drop: poison the slot so verifier-missed
                        // use-after-drop explodes loudly (same discipline as Move).
                        self.frames[here].locals[place.local.0 as usize] = None;
                    }
                    block = *target;
                }
                Terminator::Checked {
                    op,
                    args,
                    dest,
                    target,
                    trap,
                } => {
                    let mut values = Vec::new();
                    for arg in args {
                        values.push(self.eval_operand(here, arg)?);
                    }
                    let dest_ty = &body.locals[dest.0 as usize].ty;
                    match eval_checked(*op, &values, dest_ty)? {
                        CheckedOutcome::Value(value) => {
                            self.frames[here].locals[dest.0 as usize] = Some(value);
                            block = *target;
                        }
                        // A5: a shift with a bad count overrides the terminator's category with
                        // `InvalidShift`; every other trap uses the terminator's own category.
                        CheckedOutcome::Trap(override_category) => {
                            return Err(MirRunError::Trap {
                                category: override_category.unwrap_or(trap.category),
                                source: trap.source,
                                message: None,
                            })
                        }
                    }
                }
                Terminator::Trap { info, message } => {
                    // A1: resolve the optional `&str` message before aborting (it participates
                    // in evaluation like any operand).
                    let resolved = match message {
                        Some(op) => Some(self.eval_str_operand(here, op)?),
                        None => None,
                    };
                    return Err(MirRunError::Trap {
                        category: info.category,
                        source: info.source,
                        message: resolved,
                    });
                }
                Terminator::Return => {
                    return match self.frames[here].locals[0].take() {
                        Some(value) => Ok(value),
                        None => self.internal("Return with uninitialized return place"),
                    };
                }
                Terminator::Unreachable => {
                    return self.internal("reached an Unreachable terminator");
                }
            }
        }
    }

    // ---- drop glue (C4.5d) ----

    /// Syntactic type of a place: local type refined through the projections, resolved via
    /// the program's type context (the same derivation the verifier types places with).
    /// WP-C6.3d: the body index of the selected `Eq::eq` for a map operation's KEY type, or `None`
    /// when the key needs no dispatch (a primitive/`String` key has no user impl, and its structural
    /// comparison IS its lawful `Eq`).
    fn map_key_eq(&self, body: &MirBody, args: &[Operand]) -> KeyEqMode {
        let Some(Operand::Copy(place) | Operand::Move(place)) = args.first() else {
            return KeyEqMode::Structural;
        };
        let Ok(mut ty) = self.place_ty(body, place) else {
            return KeyEqMode::Structural;
        };
        while let MirTy::Ref { inner, .. } = ty {
            ty = *inner;
        }
        let MirTy::Core(crate::hir::CoreType::HashMap | crate::hir::CoreType::HashSet, type_args) =
            ty
        else {
            return KeyEqMode::Structural;
        };
        let (item, key_args) = match type_args.first() {
            Some(MirTy::Struct(item, a) | MirTy::Enum(EnumRef::User(item), a)) => {
                (*item, a.clone())
            }
            // A primitive/`String` key: no user impl exists, and its structural comparison IS its
            // lawful `Eq`.
            _ => return KeyEqMode::Structural,
        };
        match self
            .program
            .types
            .eq_impls
            .get(&(item.0, key_args))
            .and_then(|symbol| self.by_symbol.get(symbol.as_str()).copied())
        {
            Some(index) => KeyEqMode::UserEq(index),
            // A nominal key ALWAYS has an entry — it needs an `impl Eq` to satisfy the key bound at
            // all — so a missing one is a compiler defect, reported as such rather than silently
            // becoming structural comparison (which the backend would refuse: CD-138).
            None => KeyEqMode::MissingForNominal,
        }
    }

    /// WP-C6.3d: the index of the entry whose key equals `query` under the key type's lawful `Eq`,
    /// scanning in first-insertion order so the FIRST match wins (STD-HASH-001: an equal key retains
    /// the originally stored key and its position).
    fn find_entry(
        &mut self,
        key_eq: KeyEqMode,
        recv: &Option<MirValue>,
        query: &MirValue,
    ) -> Result<Option<usize>, MirRunError> {
        let Some(MirValue::Ref {
            frame,
            generation,
            local,
            path,
        }) = recv.clone()
        else {
            return self.internal("HashMap op expects a map reference");
        };
        self.check_ref_live(frame, generation)?;
        let entries = match self.read_resolved(frame, local, &path)? {
            MirValue::Vec(entries) => entries,
            other => return self.internal(format!("HashMap referent is {other:?}")),
        };
        for (index, entry) in entries.iter().enumerate() {
            let MirValue::Aggregate(kv) = entry else {
                continue;
            };
            if self.entry_key_matches(
                key_eq, frame, generation, local, &path, index, &kv[0], query,
            )? {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    /// WP-C6.3d: does the entry at `index` hold `query` as its key?
    ///
    /// With no selected `Eq` this is the structural comparison map ops have always used, which is
    /// correct for primitive and `String` keys. With one, it CALLS the user's `Eq::eq` — so a user
    /// impl decides identity, its panics are observable, and HIR/MIR/native agree (before this, MIR
    /// compared structurally and silently disagreed with HIR on any custom `Eq`).
    #[allow(clippy::too_many_arguments)]
    fn entry_key_matches(
        &mut self,
        key_eq: KeyEqMode,
        frame: usize,
        generation: u64,
        local: u32,
        path: &[ConcreteProj],
        index: usize,
        stored_key: &MirValue,
        query: &MirValue,
    ) -> Result<bool, MirRunError> {
        let body_index = match key_eq {
            KeyEqMode::Structural => return Ok(stored_key == query),
            KeyEqMode::UserEq(index) => index,
            KeyEqMode::MissingForNominal => {
                return self.internal(
                    "a nominal HashMap key reached execution with no `eq_impls` entry — key \
                     identity would silently fall back to structural comparison while the native \
                     backend refuses the same program (CD-138)",
                )
            }
        };
        let mut key_path = path.to_vec();
        key_path.push(ConcreteProj::Index(index));
        key_path.push(ConcreteProj::Field(0));
        let stored_ref = MirValue::Ref {
            frame,
            generation,
            local,
            path: key_path,
        };
        // `Eq::eq(&self, other: &K)` needs a PLACE for the query too; it is a value here, so it is
        // parked in a scratch frame for exactly the duration of the call.
        let scratch_generation = self.next_generation;
        self.next_generation += 1;
        self.frames.push(Frame {
            locals: vec![Some(query.clone())],
            generation: scratch_generation,
        });
        let scratch = self.frames.len() - 1;
        let query_ref = MirValue::Ref {
            frame: scratch,
            generation: scratch_generation,
            local: 0,
            path: Vec::new(),
        };
        let result = self.call(body_index, vec![stored_ref, query_ref]);
        self.frames.truncate(scratch);
        match result? {
            MirValue::Bool(equal) => Ok(equal),
            other => self.internal(format!("Eq::eq returned {other:?}, expected Bool")),
        }
    }

    fn place_ty(&self, body: &MirBody, place: &Place) -> Result<MirTy, MirRunError> {
        let mut ty = body
            .locals
            .get(place.local.0 as usize)
            .map(|d| d.ty.clone())
            .ok_or_else(|| MirRunError::Internal("place local out of bounds".into()))?;
        for proj in &place.projection {
            ty = match (proj, ty) {
                (Projection::Field(i), MirTy::Struct(item, args)) => self
                    .program
                    .types
                    .struct_fields
                    .get(&(item.0, args))
                    .and_then(|fields| fields.get(*i as usize))
                    .cloned()
                    .ok_or_else(|| {
                        MirRunError::Internal(format!("no field type for struct #{}", item.0))
                    })?,
                (Projection::Field(i), MirTy::Tuple(elems)) => elems
                    .get(*i as usize)
                    .cloned()
                    .ok_or_else(|| MirRunError::Internal("tuple field out of bounds".into()))?,
                (Projection::VariantField(v, i), MirTy::Enum(enum_ref, args)) => self
                    .variant_payload_tys(&enum_ref, &args, *v)?
                    .get(*i as usize)
                    .cloned()
                    .ok_or_else(|| MirRunError::Internal("variant field out of bounds".into()))?,
                (Projection::Deref, MirTy::Ref { inner, .. }) => *inner,
                (Projection::Index(_), MirTy::Array(elem, _))
                | (Projection::Index(_), MirTy::Slice(elem)) => *elem,
                // A5 (CD-038): statically known array element; the verifier proved the bound.
                (Projection::ConstIndex(_), MirTy::Array(elem, _)) => *elem,
                (proj, ty) => {
                    return self.internal(format!("place typing: {proj:?} on {ty:?}"));
                }
            };
        }
        Ok(ty)
    }

    fn variant_payload_tys(
        &self,
        enum_ref: &EnumRef,
        args: &[MirTy],
        variant: u32,
    ) -> Result<Vec<MirTy>, MirRunError> {
        // WP-C5.3d-1b: the table itself comes from `drop_plan::variant_payloads`, the single
        // derivation shared with the drop plan and the backend's type emission.
        drop_plan::variant_payloads(enum_ref, args, &self.program.types)
            .and_then(|variants| variants.get(variant as usize).cloned())
            .ok_or_else(|| {
                MirRunError::Internal(format!("no payload for variant {variant} of {enum_ref:?}"))
            })
    }

    /// Drop glue for the value at (frame, local, path).
    ///
    /// WP-C5.3d-1b (CD-062): the ORDER is no longer decided here. `mir::drop_plan` derives the
    /// canonical plan from the type and the type context, and this function only applies it —
    /// projecting a component becomes a path push, running a destructor becomes a call. The
    /// native emitter applies the same plan with its own two operations. Before this split, the
    /// two consumers reconstructed the order independently and had already disagreed (CD-060).
    fn drop_in_place(
        &mut self,
        frame: usize,
        local: u32,
        path: Vec<ConcreteProj>,
        ty: &MirTy,
    ) -> Result<(), MirRunError> {
        let plan = self.drop_plan_for(ty)?;
        self.run_drop_plan(frame, local, path, &plan)
    }

    /// The memoised plan for `ty`.
    fn drop_plan_for(&mut self, ty: &MirTy) -> Result<std::rc::Rc<DropPlan>, MirRunError> {
        if let Some(plan) = self.drop_plans.get(ty) {
            return Ok(plan.clone());
        }
        let plan = std::rc::Rc::new(
            drop_plan::plan_for(ty, &self.program.types).map_err(|e| MirRunError::Internal(e.0))?,
        );
        self.drop_plans.insert(ty.clone(), plan.clone());
        Ok(plan)
    }

    /// Apply a [`DropPlan`] to the value at (frame, local, path).
    ///
    /// The destructor call takes an `&mut` receiver so that mutations it makes stay visible to
    /// the component destruction that follows — matching the oracle. That ordering is the plan's
    /// (`Destructor { then }` nests the components inside), not this function's.
    fn run_drop_plan(
        &mut self,
        frame: usize,
        local: u32,
        path: Vec<ConcreteProj>,
        plan: &DropPlan,
    ) -> Result<(), MirRunError> {
        match plan {
            DropPlan::Noop => {}
            // A11 §5: closing a host resource means calling into a native provider, which the
            // reference interpreter cannot do -- it has no linked provider and no ABI boundary. A
            // program using a host resource is a NATIVE-only program, and saying so is better than
            // pretending the close happened.
            DropPlan::HostResourceClose { close } => {
                return self.internal(format!(
                    "host-resource close (provider call {}) cannot run in the reference \
                     interpreter: closing requires a linked native provider",
                    close.0
                ))
            }
            DropPlan::Destructor { symbol, then } => {
                let Some(&idx) = self.by_symbol.get(symbol.as_str()) else {
                    return self.internal(format!("dtor instance {symbol} not lowered"));
                };
                let receiver = MirValue::Ref {
                    frame,
                    generation: self.frames[frame].generation,
                    local,
                    path: path.clone(),
                };
                self.call(idx, vec![receiver])?;
                self.run_drop_plan(frame, local, path, then)?;
            }
            DropPlan::Fields { fields, .. } => {
                for field in fields {
                    let mut p = path.clone();
                    p.push(ConcreteProj::Field(field.index));
                    self.run_drop_plan(frame, local, p, &field.plan)?;
                }
            }
            DropPlan::Variants { variants, .. } => {
                let value = self.read_resolved(frame, local, &path)?;
                let MirValue::Enum { variant, .. } = value else {
                    return self.internal("Drop glue: enum-typed place holds a non-enum value");
                };
                let Some(arm) = variants.get(variant as usize) else {
                    return self.internal(format!("drop plan has no arm for variant {variant}"));
                };
                // `arm.fields` is cloned so the borrow of `self.program` ends before the
                // recursive calls; the plan is small and this runs once per enum drop.
                for field in arm.fields.clone() {
                    let mut p = path.clone();
                    p.push(ConcreteProj::Variant(variant, field.index));
                    self.run_drop_plan(frame, local, p, &field.plan)?;
                }
            }
            DropPlan::Array { len, elem } => {
                for i in drop_plan::array_order(*len) {
                    let mut p = path.clone();
                    p.push(ConcreteProj::Index(i as usize));
                    self.run_drop_plan(frame, local, p, elem)?;
                }
            }
            // A1 (CD-031) §5a: Vec<T> drops its elements through STARK glue in REVERSE index
            // order (matched to the oracle), then reclaims the buffer (unobservable). Elements
            // are opaque to projections, so they drop from the read-out value via a scratch
            // slot rather than a place path.
            DropPlan::VecElements { elem } => {
                let elem_plan = self.drop_plan_for(elem)?;
                if let MirValue::Vec(mut elems) = self.read_resolved(frame, local, &path)? {
                    while let Some(e) = elems.pop() {
                        self.drop_owned_value(frame, e, &elem_plan)?;
                    }
                }
            }
            // 0.1-A7: dropping a `Box<T>` drops the contained `T` exactly once, then releases
            // the allocation (unobservable). A box consumed by `into_inner` no longer holds the
            // value — ownership moved to the caller — so nothing is dropped twice.
            DropPlan::BoxInner { inner } => {
                let inner_plan = self.drop_plan_for(inner)?;
                if let MirValue::Aggregate(mut fields) = self.read_resolved(frame, local, &path)? {
                    if let Some(value) = fields.pop() {
                        self.drop_owned_value(frame, value, &inner_plan)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// 0.1-A7 (WP-C4.7-6.1): the `Box<T>` runtime group.
    ///
    /// A box is represented as a one-element `Aggregate`. The allocation itself is unobservable
    /// in Core v1 (LAYOUT-QUERY-001 makes addresses unobservable), so the reference interpreter
    /// models only what a program can observe: that the box OWNS its value. `BoxIntoInner` moves
    /// that value out and lets the box go — it must NOT run the value's destructor, because
    /// ownership transfers to the caller.
    fn run_box_runtime(
        &mut self,
        rt: RuntimeFn,
        args: Vec<MirValue>,
    ) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        let mut iter = args.into_iter();
        match rt {
            BoxNew => {
                let value = iter
                    .next()
                    .ok_or_else(|| MirRunError::Internal("BoxNew expects one argument".into()))?;
                Ok(MirValue::Aggregate(vec![value]))
            }
            BoxIntoInner => match iter.next() {
                Some(MirValue::Aggregate(mut fields)) if fields.len() == 1 => Ok(fields.remove(0)),
                other => self.internal(format!("BoxIntoInner on {other:?}")),
            },
            _ => self.internal(format!("non-Box runtime op {rt:?} in run_box_runtime")),
        }
    }

    /// Drop a standalone owned value (a Vec element, §5a): place it in a scratch slot of
    /// `frame` so its glue can take an `&mut` reference, run the glue, then remove the slot.
    fn drop_owned_value(
        &mut self,
        frame: usize,
        value: MirValue,
        plan: &DropPlan,
    ) -> Result<(), MirRunError> {
        let scratch = self.frames[frame].locals.len() as u32;
        self.frames[frame].locals.push(Some(value));
        let result = self.run_drop_plan(frame, scratch, Vec::new(), plan);
        self.frames[frame].locals.truncate(scratch as usize);
        result
    }

    // ---- place resolution (C4.5b-2) ----

    /// Resolve a syntactic place (evaluated in `eval_frame`) to a concrete
    /// (frame, local, path). `Deref` re-anchors through `Ref` values; `Index(proof)` reads the
    /// proof from `eval_frame` (proof locals always belong to the evaluating body).
    fn resolve_place(
        &self,
        eval_frame: usize,
        place: &Place,
    ) -> Result<(usize, u32, Vec<ConcreteProj>), MirRunError> {
        let mut frame = eval_frame;
        let mut local = place.local.0;
        let mut path: Vec<ConcreteProj> = Vec::new();
        for projection in &place.projection {
            match projection {
                Projection::Field(i) => path.push(ConcreteProj::Field(*i)),
                Projection::VariantField(v, i) => path.push(ConcreteProj::Variant(*v, *i)),
                // A5 (CD-038): the index is already known; the verifier proved it in range.
                Projection::ConstIndex(i) => path.push(ConcreteProj::Index(*i as usize)),
                Projection::Index(proof) => {
                    let proof_value = self
                        .frames
                        .get(eval_frame)
                        .and_then(|f| f.locals.get(proof.0 as usize))
                        .and_then(|s| s.as_ref());
                    match proof_value {
                        Some(MirValue::Int(i)) => path.push(ConcreteProj::Index(*i as usize)),
                        other => {
                            return self.internal(format!(
                                "index proof _{} unavailable: {other:?}",
                                proof.0
                            ))
                        }
                    }
                }
                Projection::Deref => {
                    let current = self.read_resolved(frame, local, &path)?;
                    match current {
                        MirValue::Ref {
                            frame: f,
                            generation,
                            local: l,
                            path: p,
                        } => {
                            // C4.5f-1: slot AND generation must match — a reused slot is a
                            // dangling reference, reported loudly, never silently aliased.
                            self.check_ref_live(f, generation)?;
                            frame = f;
                            local = l;
                            path = p;
                        }
                        MirValue::ByteSlice(_) => {
                            // `str.bytes()` stores an immutable `&[UInt8]` snapshot directly in
                            // the local. Dereferencing it keeps the same value; following index
                            // projections are resolved by `read_resolved` on `ByteSlice`.
                        }
                        other => return self.internal(format!("Deref of non-reference {other:?}")),
                    }
                }
            }
        }
        Ok((frame, local, path))
    }

    fn read_resolved(
        &self,
        frame: usize,
        local: u32,
        path: &[ConcreteProj],
    ) -> Result<MirValue, MirRunError> {
        let Some(mut value) = self
            .frames
            .get(frame)
            .and_then(|f| f.locals.get(local as usize))
            .and_then(|s| s.as_ref())
        else {
            return self.internal(format!(
                "read of uninitialized/moved local _{local} (frame {frame})"
            ));
        };
        let mut k = 0;
        while k < path.len() {
            let step = &path[k];
            value = match (step, value) {
                // 0.1-A6 (A4 slicing): a Slice window either COMPOSES with a following Index
                // (view-relative i becomes absolute start+i) or, when it ends the path, yields
                // a cloned sub-view value (read-only — used for CheckIndex length reads).
                (ConcreteProj::Slice { start, len }, MirValue::Aggregate(elems))
                | (ConcreteProj::Slice { start, len }, MirValue::Vec(elems)) => {
                    if let Some(ConcreteProj::Index(i)) = path.get(k + 1) {
                        if *i >= *len {
                            return self
                                .internal("proven slice index out of bounds (verifier bug)");
                        }
                        k += 1; // consume the composed Index as well
                        match elems.get(start + i) {
                            Some(e) => e,
                            None => return self.internal("slice window exceeds its base"),
                        }
                    } else {
                        let sub = elems
                            .get(*start..start + len)
                            .ok_or_else(|| {
                                MirRunError::Internal("slice window exceeds its base".into())
                            })?
                            .to_vec();
                        return Ok(MirValue::Vec(sub));
                    }
                }
                (ConcreteProj::Field(i), MirValue::Aggregate(fields)) => {
                    match fields.get(*i as usize) {
                        Some(f) => f,
                        None => return self.internal("field projection out of bounds"),
                    }
                }
                (ConcreteProj::Variant(v, i), MirValue::Enum { variant, fields }) => {
                    if variant != v {
                        return self.internal("VariantField read from a different active variant");
                    }
                    match fields.get(*i as usize) {
                        Some(f) => f,
                        None => return self.internal("variant field out of bounds"),
                    }
                }
                // 0.1-A2: `Index` also resolves into a runtime Vec snapshot (iterator
                // interior references, C4.5f-2).
                (ConcreteProj::Index(i), MirValue::Aggregate(elems))
                | (ConcreteProj::Index(i), MirValue::Vec(elems)) => match elems.get(*i) {
                    Some(e) => e,
                    None => {
                        return self.internal("proven index out of bounds (verifier/lowering bug)")
                    }
                },
                (ConcreteProj::Index(i), MirValue::ByteSlice(bytes)) => {
                    return bytes
                        .get(*i)
                        .map(|b| MirValue::Int(*b as i128))
                        .ok_or_else(|| {
                            MirRunError::Internal(
                                "proven byte-slice index out of bounds (verifier/lowering bug)"
                                    .into(),
                            )
                        });
                }
                (step, value) => {
                    return self.internal(format!("projection {step:?} on value {value:?}"))
                }
            };
            k += 1;
        }
        if matches!(value, MirValue::Moved) {
            return self.internal(format!(
                "read of a moved-out place _{local}{path:?} (C4.5f-1 poison)"
            ));
        }
        Ok(value.clone())
    }

    fn write_resolved(
        &mut self,
        frame: usize,
        local: u32,
        path: &[ConcreteProj],
        value: MirValue,
    ) -> Result<(), MirRunError> {
        if path.is_empty() {
            let slot = self
                .frames
                .get_mut(frame)
                .and_then(|f| f.locals.get_mut(local as usize))
                .ok_or_else(|| MirRunError::Internal("write to invalid local".into()))?;
            *slot = Some(value);
            return Ok(());
        }
        let slot = self
            .frames
            .get_mut(frame)
            .and_then(|f| f.locals.get_mut(local as usize))
            .and_then(|s| s.as_mut())
            .ok_or_else(|| {
                MirRunError::Internal(format!(
                    "write through uninitialized local _{local} (frame {frame})"
                ))
            })?;
        let mut target = slot;
        // 0.1-A8 (WP-C4.7-8.6): a `Slice` window followed by an `Index` composes to the absolute
        // element, exactly as the READ path does — that composition is what makes a write through
        // an exclusive view reach the original object (REF-SLICE-001). Normalizing here keeps the
        // walk below a simple one-step-at-a-time loop.
        let path: Vec<ConcreteProj> = {
            let mut out: Vec<ConcreteProj> = Vec::with_capacity(path.len());
            let mut k = 0;
            while k < path.len() {
                match (&path[k], path.get(k + 1)) {
                    (ConcreteProj::Slice { start, len }, Some(ConcreteProj::Index(i))) => {
                        if *i >= *len {
                            return Err(MirRunError::Internal(
                                "proven slice index out of bounds (verifier bug)".into(),
                            ));
                        }
                        out.push(ConcreteProj::Index(start + i));
                        k += 2;
                    }
                    (step, _) => {
                        out.push(step.clone());
                        k += 1;
                    }
                }
            }
            out
        };
        for step in &path {
            target = match (step, target) {
                (ConcreteProj::Field(i), MirValue::Aggregate(fields)) => fields
                    .get_mut(*i as usize)
                    .ok_or_else(|| MirRunError::Internal("field write out of bounds".into()))?,
                (ConcreteProj::Variant(v, i), MirValue::Enum { variant, fields }) => {
                    if *variant != *v {
                        return Err(MirRunError::Internal(
                            "VariantField write to a different active variant".into(),
                        ));
                    }
                    fields
                        .get_mut(*i as usize)
                        .ok_or_else(|| MirRunError::Internal("variant write oob".into()))?
                }
                (ConcreteProj::Index(i), MirValue::Aggregate(elems))
                | (ConcreteProj::Index(i), MirValue::Vec(elems)) => {
                    elems.get_mut(*i).ok_or_else(|| {
                        MirRunError::Internal("proven index write out of bounds".into())
                    })?
                }
                // A bare `Slice` window with no following `Index` is not a writable place: it
                // denotes the sub-view as a value, and the normalization above has already
                // folded away every composed form. Reaching here means malformed MIR.
                (ConcreteProj::Slice { .. }, _) => {
                    return Err(MirRunError::Internal(
                        "write to a slice view as a whole (not an element)".into(),
                    ))
                }
                (step, target) => {
                    return Err(MirRunError::Internal(format!(
                        "write projection {step:?} on {target:?}"
                    )));
                }
            };
        }
        *target = value;
        Ok(())
    }

    fn write_place(
        &mut self,
        eval_frame: usize,
        place: &Place,
        value: MirValue,
    ) -> Result<(), MirRunError> {
        let (frame, local, path) = self.resolve_place(eval_frame, place)?;
        self.write_resolved(frame, local, &path, value)
    }

    // ---- values ----

    fn eval_rvalue(&mut self, here: usize, rvalue: &Rvalue) -> Result<MirValue, MirRunError> {
        Ok(match rvalue {
            Rvalue::Use(op) => self.eval_operand(here, op)?,
            Rvalue::UnOp(op, operand) => {
                let value = self.eval_operand(here, operand)?;
                eval_unop(*op, value)?
            }
            Rvalue::BinOp(op, lhs, rhs) => {
                let l = self.eval_operand(here, lhs)?;
                let r = self.eval_operand(here, rhs)?;
                eval_binop(*op, l, r)?
            }
            Rvalue::Aggregate(kind, operands) => {
                let mut values = Vec::new();
                for op in operands {
                    values.push(self.eval_operand(here, op)?);
                }
                match kind {
                    AggKind::EnumVariant(_, variant) => MirValue::Enum {
                        variant: *variant,
                        fields: values,
                    },
                    _ => MirValue::Aggregate(values),
                }
            }
            Rvalue::Discriminant(place) => {
                let (f, l, p) = self.resolve_place(here, place)?;
                let value = self.read_resolved(f, l, &p)?;
                let MirValue::Enum { variant, .. } = value else {
                    return self.internal(format!("Discriminant of {value:?}"));
                };
                MirValue::Int(i128::from(variant))
            }
            // A4 (CD-036) established that a layout query is answered by ONE layout service.
            // WP-C5.3e (CD-067): answered from the selected named target CONTRACT, not from any
            // representation this interpreter happens to use for values. A type the contract does
            // not describe is refused rather than guessed, so no engine can answer a query
            // another engine must refuse.
            Rvalue::LayoutQuery { kind, ty } => {
                let layout = self
                    .layout
                    .layout_of(ty, &self.program.types)
                    .map_err(|e| MirRunError::Internal(e.0))?;
                MirValue::Int(i128::from(match kind {
                    LayoutKind::SizeOf => layout.size,
                    LayoutKind::AlignOf => layout.align,
                }))
            }
            // C4.5b-2: real reference creation (C4.5f-1: stamped with the pointee frame's
            // generation).
            Rvalue::RefOf { place, .. } => {
                let (frame, local, path) = self.resolve_place(here, place)?;
                let generation = self.frames[frame].generation;
                MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }
            }
        })
    }
}

/// Pure unary operators — free for the same reason [`eval_binop`] is.
pub(crate) fn eval_unop(op: MirUnOp, value: MirValue) -> Result<MirValue, MirRunError> {
    match (op, value) {
        (MirUnOp::Not, MirValue::Bool(b)) => Ok(MirValue::Bool(!b)),
        (MirUnOp::FloatNeg, MirValue::Float(f)) => Ok(MirValue::Float(-f)),
        (op, value) => Err(MirRunError::Internal(format!("UnOp {op:?} on {value:?}"))),
    }
}

/// Pure binary operators, as a FREE function so `mir::opt` folds constants with the exact code the
/// interpreter executes. A second implementation of these tables is precisely how an optimiser and
/// an interpreter come to disagree, and §39 makes that disagreement observable.
pub(crate) fn eval_binop(op: MirBinOp, l: MirValue, r: MirValue) -> Result<MirValue, MirRunError> {
    use MirBinOp::*;
    Ok(match (op, l, r) {
        (Eq, l, r) => MirValue::Bool(l == r),
        (Ne, l, r) => MirValue::Bool(l != r),
        (Lt, MirValue::Int(a), MirValue::Int(b)) => MirValue::Bool(a < b),
        (Le, MirValue::Int(a), MirValue::Int(b)) => MirValue::Bool(a <= b),
        (Gt, MirValue::Int(a), MirValue::Int(b)) => MirValue::Bool(a > b),
        (Ge, MirValue::Int(a), MirValue::Int(b)) => MirValue::Bool(a >= b),
        (Lt, MirValue::Float(a), MirValue::Float(b)) => MirValue::Bool(a < b),
        (Le, MirValue::Float(a), MirValue::Float(b)) => MirValue::Bool(a <= b),
        (Gt, MirValue::Float(a), MirValue::Float(b)) => MirValue::Bool(a > b),
        (Ge, MirValue::Float(a), MirValue::Float(b)) => MirValue::Bool(a >= b),
        (FloatAdd, MirValue::Float(a), MirValue::Float(b)) => MirValue::Float(a + b),
        (FloatSub, MirValue::Float(a), MirValue::Float(b)) => MirValue::Float(a - b),
        (FloatMul, MirValue::Float(a), MirValue::Float(b)) => MirValue::Float(a * b),
        // CD-139: IEEE division and remainder, TOTAL. Rust's `f64` `/` and `%` are IEEE 754,
        // so a zero divisor yields the signed infinity or NaN NUM-FLOAT-OP-001 requires
        // without any special case here — which is exactly why no check is owed.
        (FloatDiv, MirValue::Float(a), MirValue::Float(b)) => MirValue::Float(a / b),
        (FloatRem, MirValue::Float(a), MirValue::Float(b)) => MirValue::Float(a % b),
        // A5: bitwise on the sign-extended i128 carrier — for same-width operands the low
        // bits agree with the true-width result and the value stays in range (no trap).
        (BitAnd, MirValue::Int(a), MirValue::Int(b)) => MirValue::Int(a & b),
        (BitOr, MirValue::Int(a), MirValue::Int(b)) => MirValue::Int(a | b),
        (BitXor, MirValue::Int(a), MirValue::Int(b)) => MirValue::Int(a ^ b),
        (op, l, r) => {
            return Err(MirRunError::Internal(format!(
                "BinOp {op:?} on {l:?}, {r:?}"
            )));
        }
    })
}

// (see `CheckedOutcome` below)
/// Checked/trapping primitives. `Trap(None)` traps with the terminator's own category;
/// `Trap(Some(cat))` overrides it (A5 shifts: a bad count is `InvalidShift`).
pub(crate) fn eval_checked(
    op: CheckedOp,
    args: &[MirValue],
    dest_ty: &MirTy,
) -> Result<CheckedOutcome, MirRunError> {
    use CheckedOp::*;
    match op {
        Add | Sub | Mul | Div | Rem | Neg | Pow => {
            let (min, max) = int_range(dest_ty)
                .ok_or_else(|| MirRunError::Internal("checked int op on non-int".into()))?;
            let int = |v: &MirValue| -> Result<i128, MirRunError> {
                match v {
                    MirValue::Int(i) => Ok(*i),
                    other => Err(MirRunError::Internal(format!(
                        "checked op operand {other:?}"
                    ))),
                }
            };
            let result: Option<i128> = match op {
                Add => int(&args[0])?.checked_add(int(&args[1])?),
                Sub => int(&args[0])?.checked_sub(int(&args[1])?),
                Mul => int(&args[0])?.checked_mul(int(&args[1])?),
                Div => {
                    let (a, b) = (int(&args[0])?, int(&args[1])?);
                    if b == 0 {
                        None
                    } else {
                        a.checked_div(b)
                    }
                }
                Rem => {
                    let (a, b) = (int(&args[0])?, int(&args[1])?);
                    if b == 0 {
                        None
                    } else {
                        a.checked_rem(b)
                    }
                }
                Neg => int(&args[0])?.checked_neg(),
                // A5: exponent must be nonnegative (u32::try_from rejects negatives,
                // NUM-INT-ARITH-001); each intermediate multiply is checked by checked_pow.
                Pow => {
                    let base = int(&args[0])?;
                    u32::try_from(int(&args[1])?)
                        .ok()
                        .and_then(|exp| base.checked_pow(exp))
                }
                _ => unreachable!(),
            };
            Ok(result
                .filter(|v| *v >= min && *v <= max)
                .map(MirValue::Int)
                .into())
        }
        Shl | Shr => {
            // A5 / NUM-SHIFT-001: the count must be nonnegative and strictly less than the
            // bit width of the LEFT operand (= the dest/result type); otherwise trap. No
            // masking or reduction. Left shift traps when the result is not representable
            // (the post-hoc range filter); right shift on the i128 carrier is arithmetic
            // for signed and — since unsigned values are stored nonnegative — logical for
            // unsigned, matching the abstract machine.
            let (min, max) = int_range(dest_ty)
                .ok_or_else(|| MirRunError::Internal("shift on non-int".into()))?;
            let width = int_width(dest_ty)
                .ok_or_else(|| MirRunError::Internal("shift width on non-int".into()))?;
            let int = |v: &MirValue| -> Result<i128, MirRunError> {
                match v {
                    MirValue::Int(i) => Ok(*i),
                    other => Err(MirRunError::Internal(format!("shift operand {other:?}"))),
                }
            };
            let (left, count) = (int(&args[0])?, int(&args[1])?);
            if count < 0 || count >= i128::from(width) {
                return Ok(CheckedOutcome::Trap(Some(TrapCategory::InvalidShift)));
            }
            let result = if matches!(op, Shl) {
                left.checked_shl(count as u32)
            } else {
                left.checked_shr(count as u32)
            };
            Ok(result
                .filter(|v| *v >= min && *v <= max)
                .map(MirValue::Int)
                .into())
        }
        FloatDiv | FloatRem => {
            let (a, b) = match (&args[0], &args[1]) {
                (MirValue::Float(a), MirValue::Float(b)) => (*a, *b),
                other => {
                    return Err(MirRunError::Internal(format!(
                        "checked float op operands {other:?}"
                    )))
                }
            };
            // CD-006: division/modulo by zero traps for floats too.
            if b == 0.0 {
                return Ok(CheckedOutcome::Trap(None));
            }
            Ok(CheckedOutcome::Value(MirValue::Float(
                if matches!(op, FloatDiv) { a / b } else { a % b },
            )))
        }
        Cast => {
            let value = &args[0];
            Ok(CheckedOutcome::from(match (value, dest_ty) {
                (MirValue::Int(v), ty) if int_range(ty).is_some() => {
                    let (min, max) = int_range(ty).unwrap();
                    if *v >= min && *v <= max {
                        Some(MirValue::Int(*v))
                    } else {
                        None // CastFailure trap
                    }
                }
                // CD-140: an integer-to-`Float32` cast rounds ONCE to binary32
                // (NUM-FLOAT-CONV-001), so it must narrow like the float-to-`Float32` arm
                // below. Sharing the `Float64` arm silently produced an f64-precision result
                // for a `Float32` destination.
                (MirValue::Int(v), MirTy::Float32) => Some(MirValue::Float(f64::from(*v as f32))),
                (MirValue::Int(v), MirTy::Float64) => Some(MirValue::Float(*v as f64)),
                (MirValue::Float(f), MirTy::Float64) => Some(MirValue::Float(*f)),
                (MirValue::Float(f), MirTy::Float32) => Some(MirValue::Float(f64::from(*f as f32))),
                (MirValue::Float(f), ty) if int_range(ty).is_some() => {
                    let truncated = f.trunc();
                    let (min, max) = int_range(ty).unwrap();
                    // `max as f64` ROUNDS at 64-bit widths -- `u64::MAX as f64` is 2^64 and
                    // `i64::MAX as f64` is 2^63, both one past the real maximum. An
                    // inclusive `truncated > max as f64` test therefore ACCEPTS 2^64/2^63,
                    // which 03-Type-System.md requires to trap. `max + 1` is always an exact
                    // power of two and so exactly representable as f64, making the half-open
                    // `>= (max + 1) as f64` test exact at every width. `min` needs no such
                    // care: every min is 0 or -2^(n-1), already exact as f64. `+inf`/`-inf`
                    // fall out of the same two comparisons; NaN is rejected explicitly.
                    if f.is_nan() || truncated < min as f64 || truncated >= (max + 1) as f64 {
                        None
                    } else {
                        Some(MirValue::Int(truncated as i128))
                    }
                }
                (value, ty) => {
                    return Err(MirRunError::Internal(format!("cast {value:?} to {ty:?}")))
                }
            }))
        }
        CheckIndex => {
            let len = match &args[0] {
                MirValue::Aggregate(elems) => elems.len() as i128,
                // 0.1-A6: a slice-view base reads as a Vec sub-view; its len is the VIEW
                // length, so the proof bounds i against the view, not the base container.
                MirValue::Vec(elems) => elems.len() as i128,
                MirValue::ByteSlice(bytes) => bytes.len() as i128,
                other => {
                    return Err(MirRunError::Internal(format!(
                        "CheckIndex base is not an aggregate: {other:?}"
                    )))
                }
            };
            let index = match &args[1] {
                MirValue::Int(i) => *i,
                other => {
                    return Err(MirRunError::Internal(format!(
                        "CheckIndex index is not an integer: {other:?}"
                    )))
                }
            };
            if index >= 0 && index < len {
                // The proof VALUE is the checked index (interp-internal representation of
                // the opaque token; MIR-level opacity is the verifier's concern).
                Ok(CheckedOutcome::Value(MirValue::Int(index)))
            } else {
                Ok(CheckedOutcome::Trap(None)) // IndexOutOfBounds
            }
        }
    }
}

impl<'a> Interp<'a> {
    // ---- operands ----

    fn eval_operand(&mut self, here: usize, op: &Operand) -> Result<MirValue, MirRunError> {
        match op {
            Operand::Copy(place) => {
                let (f, l, p) = self.resolve_place(here, place)?;
                self.read_resolved(f, l, &p)
            }
            Operand::Move(place) => {
                if place.projection.is_empty() {
                    match self.frames[here].locals[place.local.0 as usize].take() {
                        Some(value) => Ok(value),
                        None => self.internal(format!(
                            "move from uninitialized/moved local _{}",
                            place.local.0
                        )),
                    }
                } else {
                    // C4.5f-1: projected move TAKES — the sub-value is replaced with a
                    // `Moved` poison so a verifier-missed later read explodes loudly instead
                    // of silently observing a retained clone (CD-030 review warning; the
                    // field-precise V-MOVE-1 is the primary guard, this is defense in depth).
                    let (f, l, p) = self.resolve_place(here, place)?;
                    let value = self.read_resolved(f, l, &p)?;
                    if matches!(value, MirValue::Moved) {
                        return self.internal(format!(
                            "move from an already-moved place _{}{:?}",
                            place.local.0, place.projection
                        ));
                    }
                    self.write_resolved(f, l, &p, MirValue::Moved)?;
                    Ok(value)
                }
            }
            Operand::Const(constant) => Ok(match constant {
                Constant::Int(v, _) => MirValue::Int(*v),
                Constant::Float(f, _) => MirValue::Float(*f),
                Constant::Bool(b) => MirValue::Bool(*b),
                Constant::Unit => MirValue::Unit,
                Constant::Str(s) => MirValue::Str(std::rc::Rc::from(s.as_str())),
                Constant::FnPtr(instance) => match self.by_symbol.get(instance.symbol.as_str()) {
                    Some(&idx) => MirValue::FnPtr(idx),
                    None => {
                        return self
                            .internal(format!("FnPtr to unknown instance {}", instance.symbol))
                    }
                },
            }),
        }
    }

    // ---- runtime surface ----

    fn run_runtime(
        &mut self,
        rt: RuntimeFn,
        args: Vec<MirValue>,
        call_info: SourceInfo,
        key_eq: KeyEqMode,
    ) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        if is_vec_runtime(rt) {
            return self.run_vec_runtime(rt, args, call_info);
        }
        if is_map_runtime(rt) {
            return self.run_map_runtime(rt, args, key_eq);
        }
        if is_slice_runtime(rt) {
            return self.run_slice_runtime(rt, args, call_info);
        }
        if is_box_runtime(rt) {
            return self.run_box_runtime(rt, args);
        }
        let mut iter = args.into_iter();
        let arg = iter.next();
        let rest: Vec<MirValue> = iter.collect();
        match (rt, arg) {
            (PrintlnInt64 | PrintlnUInt64, Some(MirValue::Int(v))) => {
                if matches!(rt, PrintlnUInt64) {
                    let _ = writeln!(self.output, "{}", v as u128);
                } else {
                    let _ = writeln!(self.output, "{v}");
                }
                Ok(MirValue::Unit)
            }
            (PrintInt64 | PrintUInt64, Some(MirValue::Int(v))) => {
                if matches!(rt, PrintUInt64) {
                    let _ = write!(self.output, "{}", v as u128);
                } else {
                    let _ = write!(self.output, "{v}");
                }
                Ok(MirValue::Unit)
            }
            (PrintlnBool, Some(MirValue::Bool(b))) => {
                let _ = writeln!(self.output, "{b}");
                Ok(MirValue::Unit)
            }
            (PrintBool, Some(MirValue::Bool(b))) => {
                let _ = write!(self.output, "{b}");
                Ok(MirValue::Unit)
            }
            (PrintlnFloat64, Some(MirValue::Float(f))) => {
                // The oracle's own formatter — identical output by construction.
                let _ = writeln!(self.output, "{}", crate::interp::canonical_float(f));
                Ok(MirValue::Unit)
            }
            (PrintFloat64, Some(MirValue::Float(f))) => {
                let _ = write!(self.output, "{}", crate::interp::canonical_float(f));
                Ok(MirValue::Unit)
            }
            // DEV-105 (0.1-A9): the OPERATION carries the declared width, so the interpreter
            // narrows its internal f64 storage back to f32 at this boundary and formats there —
            // `canonical_float32`, the same renderer the HIR oracle and the runtime use.
            (PrintlnFloat32, Some(MirValue::Float(f))) => {
                let _ = writeln!(
                    self.output,
                    "{}",
                    crate::interp::canonical_float32(f as f32)
                );
                Ok(MirValue::Unit)
            }
            (PrintFloat32, Some(MirValue::Float(f))) => {
                let _ = write!(
                    self.output,
                    "{}",
                    crate::interp::canonical_float32(f as f32)
                );
                Ok(MirValue::Unit)
            }
            // --- A1 str/String ops. `arg` holds the reconstructed first argument; the closure
            // below re-materializes the full list when an op needs more than one. ---
            (rt, arg) => self.run_string_runtime(rt, arg, rest),
        }
    }

    /// A1 String/str runtime ops. `first` is the (already-popped) first argument; `rest` is the
    /// remainder. `&str` operands arrive as `Str` values (lowering inserts `StringAsStr` for
    /// `String` sources, §5b); `&String`/`&mut String` operands arrive as `Ref`s into a live
    /// frame, read/mutated in place here.
    fn run_string_runtime(
        &mut self,
        rt: RuntimeFn,
        first: Option<MirValue>,
        rest: Vec<MirValue>,
    ) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        let mut rest = rest.into_iter();
        match rt {
            PrintlnStr | PrintStr => {
                let s = self.as_str(&first)?;
                if matches!(rt, PrintlnStr) {
                    let _ = writeln!(self.output, "{s}");
                } else {
                    let _ = write!(self.output, "{s}");
                }
                Ok(MirValue::Unit)
            }
            StringNew => Ok(MirValue::String(String::new())),
            StringFromStr => Ok(MirValue::String(self.as_str(&first)?.to_string())),
            StrToString => Ok(MirValue::String(self.as_str(&first)?.to_string())),
            StrBytes => {
                let s = self.as_str(&first)?;
                Ok(MirValue::ByteSlice(std::rc::Rc::from(s.as_bytes())))
            }
            StrSubstring => {
                let s = self.as_str(&first)?;
                let start = usize_of(&rest.next())?;
                let end = usize_of(&rest.next())?;
                let Some(slice) = s.get(start..end) else {
                    return self
                        .internal("String::substring range is not on valid UTF-8 boundaries");
                };
                Ok(MirValue::Str(std::rc::Rc::from(slice)))
            }
            StringClone => {
                let s = self.read_string_ref(&first)?;
                Ok(MirValue::String(s))
            }
            StringAsStr => {
                // Interior reference → read-only snapshot (§5b).
                let s = self.read_string_ref(&first)?;
                Ok(MirValue::Str(std::rc::Rc::from(s.as_str())))
            }
            StringLen => {
                let s = self.read_string_ref(&first)?;
                Ok(MirValue::Int(s.len() as i128))
            }
            StringIsEmpty => {
                let s = self.read_string_ref(&first)?;
                Ok(MirValue::Bool(s.is_empty()))
            }
            StringContains => {
                let s = self.read_string_ref(&first)?;
                let pat = self.as_str(&rest.next())?.to_string();
                Ok(MirValue::Bool(s.contains(&pat)))
            }
            StringPushStr => {
                let suffix = self.as_str(&rest.next())?.to_string();
                self.mutate_string_ref(&first, |s| s.push_str(&suffix))?;
                Ok(MirValue::Unit)
            }
            StringClear => {
                self.mutate_string_ref(&first, |s| s.clear())?;
                Ok(MirValue::Unit)
            }
            // 0.1-A3 (f-3b): Char ops. Char values are Unicode scalar codepoints in
            // MirValue::Int.
            PrintlnChar | PrintChar => {
                let c = char_of(&first)?;
                if matches!(rt, PrintlnChar) {
                    let _ = writeln!(self.output, "{c}");
                } else {
                    let _ = write!(self.output, "{c}");
                }
                Ok(MirValue::Unit)
            }
            CharFromU32 => {
                let code = u32_of(&first)?;
                Ok(match char::from_u32(code) {
                    Some(ch) => MirValue::Enum {
                        variant: 1,
                        fields: vec![MirValue::Int(i128::from(u32::from(ch)))],
                    },
                    None => MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    },
                })
            }
            StringPushChar => {
                let c = char_of(&rest.next())?;
                self.mutate_string_ref(&first, |s| s.push(c))?;
                Ok(MirValue::Unit)
            }
            StringPopChar => {
                let mut popped: Option<char> = None;
                self.mutate_string_ref(&first, |s| popped = s.pop())?;
                Ok(match popped {
                    Some(c) => MirValue::Enum {
                        variant: 1,
                        fields: vec![MirValue::Int(i128::from(u32::from(c)))],
                    },
                    None => MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    },
                })
            }
            StrLen => Ok(MirValue::Int(self.as_str(&first)?.len() as i128)),
            StrIsEmpty => Ok(MirValue::Bool(self.as_str(&first)?.is_empty())),
            StrEq => {
                let a = self.as_str(&first)?.to_string();
                let b = self.as_str(&rest.next())?.to_string();
                Ok(MirValue::Bool(a == b))
            }
            StrCmp => {
                let a = self.as_str(&first)?.to_string();
                let b = self.as_str(&rest.next())?.to_string();
                let ord = match a.cmp(&b) {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                };
                Ok(MirValue::Int(ord))
            }
            // 0.1-A5 (A4-2d): `chars()` iteration. The iterator snapshots the string's chars
            // (Char is Copy, so a snapshot matches the oracle's borrowed `CharsIter`); it lives
            // as `Aggregate([Str(snapshot), Int(cursor)])`. `Next` yields `Option<Char>`.
            CharsIterNew => {
                let s = self.as_str(&first)?.to_string();
                Ok(MirValue::Aggregate(vec![
                    MirValue::Str(std::rc::Rc::from(s.as_str())),
                    MirValue::Int(0),
                ]))
            }
            CharsIterNext => {
                let Some(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }) = first
                else {
                    return self.internal("CharsIterNext expects a &mut iterator reference");
                };
                self.check_ref_live(frame, generation)?;
                let iter_value = self.read_resolved(frame, local, &path)?;
                let MirValue::Aggregate(fields) = &iter_value else {
                    return self.internal(format!("chars iterator referent is {iter_value:?}"));
                };
                let (snapshot, cursor) = match (fields.first(), fields.get(1)) {
                    (Some(MirValue::Str(s)), Some(MirValue::Int(c))) => (s.clone(), *c as usize),
                    other => {
                        return self.internal(format!("malformed chars-iterator state {other:?}"))
                    }
                };
                match snapshot.chars().nth(cursor) {
                    Some(ch) => {
                        let mut cursor_path = path;
                        cursor_path.push(ConcreteProj::Field(1));
                        self.write_resolved(
                            frame,
                            local,
                            &cursor_path,
                            MirValue::Int(cursor as i128 + 1),
                        )?;
                        Ok(MirValue::Enum {
                            variant: 1,
                            fields: vec![MirValue::Int(i128::from(u32::from(ch)))],
                        })
                    }
                    None => Ok(MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    }),
                }
            }
            other => self.internal(format!("runtime {other:?} (string group) unhandled")),
        }
    }

    /// A1 (CD-031) Vec data-surface ops. `&Vec`/`&mut Vec` operands arrive as `Ref`s and are
    /// read/mutated in place; index/replace/remove trap `IndexOutOfBounds` with the call
    /// site's provenance (§5). Iteration is not here (deferred — see the enum note).
    fn run_vec_runtime(
        &mut self,
        rt: RuntimeFn,
        args: Vec<MirValue>,
        call_info: SourceInfo,
    ) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        let oob = || MirRunError::Trap {
            category: TrapCategory::IndexOutOfBounds,
            source: call_info,
            message: None,
        };
        let mut args = args.into_iter();
        match rt {
            VecNew | VecWithCapacity => Ok(MirValue::Vec(Vec::new())),
            VecPush => {
                let recv = args.next();
                let item = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("VecPush missing item".into()))?;
                self.mutate_vec_ref(&recv, |v| v.push(item))?;
                Ok(MirValue::Unit)
            }
            VecPop => {
                let recv = args.next();
                let popped = self.mutate_vec_ref(&recv, |v| v.pop())?;
                Ok(option_value(popped))
            }
            VecLen => {
                let recv = args.next();
                Ok(MirValue::Int(self.read_vec_ref(&recv)?.len() as i128))
            }
            VecIsEmpty => {
                let recv = args.next();
                Ok(MirValue::Bool(self.read_vec_ref(&recv)?.is_empty()))
            }
            VecClear => {
                let recv = args.next();
                self.mutate_vec_ref(&recv, |v| v.clear())?;
                Ok(MirValue::Unit)
            }
            VecIndexGet => {
                let recv = args.next();
                let i = int_arg(args.next())? as usize;
                let v = self.read_vec_ref(&recv)?;
                v.get(i).cloned().ok_or_else(oob)
            }
            VecReplace => {
                let recv = args.next();
                let i = int_arg(args.next())? as usize;
                let item = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("VecReplace missing item".into()))?;
                let len = self.read_vec_ref(&recv)?.len();
                if i >= len {
                    return Err(oob());
                }
                self.mutate_vec_ref(&recv, |v| std::mem::replace(&mut v[i], item))
            }
            VecRemove => {
                let recv = args.next();
                let i = int_arg(args.next())? as usize;
                let len = self.read_vec_ref(&recv)?.len();
                if i >= len {
                    return Err(oob());
                }
                self.mutate_vec_ref(&recv, |v| v.remove(i))
            }
            // 0.1-A4 (A4-2b): `get`/`get_mut` — checked interior access, NEVER traps. Return an
            // interior reference into the live Vec at `i` as `Some(&v[i])`, or `None` when out
            // of bounds. Mirrors HashMapGet; the mutability is a static property of the ref type.
            VecGetRef | VecGetMutRef => {
                let Some(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }) = args.next()
                else {
                    return self.internal("VecGet(Mut)Ref expects a &Vec reference");
                };
                self.check_ref_live(frame, generation)?;
                let i = int_arg(args.next())? as usize;
                let len = match self.read_resolved(frame, local, &path)? {
                    MirValue::Vec(elems) => elems.len(),
                    other => return self.internal(format!("Vec referent is {other:?}")),
                };
                if i < len {
                    let mut elem_path = path;
                    elem_path.push(ConcreteProj::Index(i));
                    Ok(MirValue::Enum {
                        variant: 1,
                        fields: vec![MirValue::Ref {
                            frame,
                            generation,
                            local,
                            path: elem_path,
                        }],
                    })
                } else {
                    Ok(MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    })
                }
            }
            // --- 0.1-A2 (C4.5f-2): by-reference iteration. The iterator value is an opaque
            // two-field aggregate [snapshot Vec, Int cursor] living in a frame local; `Next`
            // hands out interior references into THAT local (base + [Field(0), Index(i)]),
            // which the f-1 frame-generation guard protects once the iterator dies. The
            // snapshot is sound because iteration is `T: Copy` and borrowck forbids mutating
            // the source Vec while the iterator lives (A1 §5e carry-forward).
            VecIterNew => {
                // A6: a TRUE borrowed cursor `[vec-ref, cursor]` — Next indexes the LIVE Vec
                // through the reference and hands out an interior `&T`. No snapshot, so the
                // element type need not be `Copy` (the borrow checker already forbids mutating
                // the source Vec while the iterator lives).
                let vec_ref = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("VecIterNew missing receiver".into()))?;
                Ok(MirValue::Aggregate(vec![vec_ref, MirValue::Int(0)]))
            }
            VecIterNext => {
                let Some(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }) = args.next()
                else {
                    return self.internal("VecIterNext expects a &mut iterator reference");
                };
                self.check_ref_live(frame, generation)?;
                let iter_value = self.read_resolved(frame, local, &path)?;
                let MirValue::Aggregate(fields) = &iter_value else {
                    return self.internal(format!("iterator referent is {iter_value:?}"));
                };
                let (vec_ref, cursor) = match (fields.first(), fields.get(1)) {
                    (Some(r @ MirValue::Ref { .. }), Some(MirValue::Int(c))) => (r.clone(), *c),
                    other => return self.internal(format!("malformed iterator state {other:?}")),
                };
                let MirValue::Ref {
                    frame: vf,
                    generation: vg,
                    local: vl,
                    path: vp,
                } = vec_ref
                else {
                    unreachable!("matched Ref above");
                };
                self.check_ref_live(vf, vg)?;
                let len = match self.read_resolved(vf, vl, &vp)? {
                    MirValue::Vec(elems) => elems.len(),
                    other => return self.internal(format!("Vec referent is {other:?}")),
                };
                if (cursor as usize) < len {
                    // Bump the cursor in place, then hand out an interior reference into the
                    // live Vec (base + [Index(cursor)]), protected by the f-1 generation guard.
                    let mut cursor_path = path;
                    cursor_path.push(ConcreteProj::Field(1));
                    self.write_resolved(frame, local, &cursor_path, MirValue::Int(cursor + 1))?;
                    let mut elem_path = vp;
                    elem_path.push(ConcreteProj::Index(cursor as usize));
                    Ok(MirValue::Enum {
                        variant: 1,
                        fields: vec![MirValue::Ref {
                            frame: vf,
                            generation: vg,
                            local: vl,
                            path: elem_path,
                        }],
                    })
                } else {
                    Ok(MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    })
                }
            }
            other => self.internal(format!("runtime {other:?} is not a Vec op")),
        }
    }

    /// 0.1-A3 (C4.5f-3a): HashMap ops. The map value is an insertion-ordered pair vector
    /// (`MirValue::Vec` of `Aggregate([k, v])`) per CD-009 — re-inserting an existing key
    /// keeps its position; lookups are structural key comparison. `Get` and the keys
    /// iterator hand out interior references (entry `[Index(i), Field(0|1)]`), guarded by
    /// the f-1 frame generations.
    fn run_map_runtime(
        &mut self,
        rt: RuntimeFn,
        args: Vec<MirValue>,
        // WP-C6.3d: how the key type decides identity, resolved at the call site.
        key_eq: KeyEqMode,
    ) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        let mut args = args.into_iter();
        match rt {
            HashMapNew => Ok(MirValue::Vec(Vec::new())),
            HashMapInsert => {
                let recv = args.next();
                let key = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("HashMapInsert missing key".into()))?;
                let value = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("HashMapInsert missing value".into()))?;
                // WP-C6.3d: the matching entry is found FIRST, through the key type's lawful
                // `Eq` (which may run user code and so cannot happen inside a `&mut` closure over
                // the entries); only then is the map mutated by index.
                let existing = self.find_entry(key_eq, &recv, &key)?;
                self.mutate_vec_ref(&recv, |entries| match existing {
                    // CD-009: an existing key keeps its position AND its originally stored key
                    // (STD-HASH-001); only the value is replaced, and the old one is returned.
                    Some(index) => {
                        let MirValue::Aggregate(kv) = &mut entries[index] else {
                            unreachable!("HashMap entry is not a key/value pair")
                        };
                        let old = std::mem::replace(&mut kv[1], value);
                        MirValue::Enum {
                            variant: 1,
                            fields: vec![old],
                        }
                    }
                    None => {
                        entries.push(MirValue::Aggregate(vec![key, value]));
                        MirValue::Enum {
                            variant: 0,
                            fields: Vec::new(),
                        }
                    }
                })
            }
            HashMapGet => {
                let Some(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }) = args.next()
                else {
                    return self.internal("HashMapGet expects a &HashMap reference");
                };
                self.check_ref_live(frame, generation)?;
                let key = self.deref_key_arg(args.next())?;
                let entries = match self.read_resolved(frame, local, &path)? {
                    MirValue::Vec(entries) => entries,
                    other => return self.internal(format!("HashMap referent is {other:?}")),
                };
                for (i, entry) in entries.iter().enumerate() {
                    let MirValue::Aggregate(kv) = entry else {
                        continue;
                    };
                    if self.entry_key_matches(
                        key_eq, frame, generation, local, &path, i, &kv[0], &key,
                    )? {
                        let mut elem_path = path;
                        elem_path.push(ConcreteProj::Index(i));
                        elem_path.push(ConcreteProj::Field(1));
                        return Ok(MirValue::Enum {
                            variant: 1,
                            fields: vec![MirValue::Ref {
                                frame,
                                generation,
                                local,
                                path: elem_path,
                            }],
                        });
                    }
                }
                Ok(MirValue::Enum {
                    variant: 0,
                    fields: Vec::new(),
                })
            }
            HashMapLen => {
                let recv = args.next();
                Ok(MirValue::Int(self.read_vec_ref(&recv)?.len() as i128))
            }
            HashMapIsEmpty => {
                let recv = args.next();
                Ok(MirValue::Bool(self.read_vec_ref(&recv)?.is_empty()))
            }
            HashMapContainsKey => {
                let recv = args.next();
                let key = self.deref_key_arg(args.next())?;
                Ok(MirValue::Bool(
                    self.find_entry(key_eq, &recv, &key)?.is_some(),
                ))
            }
            // --- DEV-116: HashSet. Stored EXACTLY as a map is — a `Vec` of `Aggregate([elem,
            // Unit])` — so `find_entry` decides membership through the same lawful-`Eq` dispatch
            // and first-insertion order is inherited rather than reimplemented. A second
            // representation would be a second place for STD-HASH-001 to drift.
            HashSetNew => Ok(MirValue::Vec(Vec::new())),
            HashSetInsert => {
                let recv = args.next();
                let value = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("HashSetInsert missing value".into()))?;
                // Found FIRST, because a user `Eq` runs arbitrary code and cannot execute inside a
                // `&mut` borrow of the entries.
                let existing = self.find_entry(key_eq, &recv, &value)?;
                self.mutate_vec_ref(&recv, |entries| match existing {
                    // Already present: the ORIGINALLY STORED element and its position are kept
                    // (STD-HASH-001), which is observable when two elements are equal by a user
                    // `Eq` but structurally different. `insert` reports "not newly added".
                    Some(_) => MirValue::Bool(false),
                    None => {
                        entries.push(MirValue::Aggregate(vec![value, MirValue::Unit]));
                        MirValue::Bool(true)
                    }
                })
            }
            HashSetRemove => {
                let recv = args.next();
                let value = self.deref_key_arg(args.next())?;
                let existing = self.find_entry(key_eq, &recv, &value)?;
                self.mutate_vec_ref(&recv, |entries| match existing {
                    Some(index) => {
                        // `remove`, not `swap_remove`: the surviving entries keep their relative
                        // order, which the normative iteration rule requires.
                        entries.remove(index);
                        MirValue::Bool(true)
                    }
                    None => MirValue::Bool(false),
                })
            }
            HashSetContains => {
                let recv = args.next();
                let value = self.deref_key_arg(args.next())?;
                Ok(MirValue::Bool(
                    self.find_entry(key_eq, &recv, &value)?.is_some(),
                ))
            }
            HashSetLen => {
                let recv = args.next();
                Ok(MirValue::Int(self.read_vec_ref(&recv)?.len() as i128))
            }
            HashSetIsEmpty => {
                let recv = args.next();
                Ok(MirValue::Bool(self.read_vec_ref(&recv)?.is_empty()))
            }
            HashSetClear => {
                let recv = args.next();
                self.mutate_vec_ref(&recv, |entries| {
                    entries.clear();
                    MirValue::Unit
                })
            }
            HashMapRemove => {
                let recv = args.next();
                let key = self.deref_key_arg(args.next())?;
                let existing = self.find_entry(key_eq, &recv, &key)?;
                self.mutate_vec_ref(&recv, |entries| match existing {
                    Some(index) => {
                        // `remove`, not `swap_remove`: the surviving entries keep their order,
                        // which the normative first-insertion iteration rule requires.
                        let MirValue::Aggregate(kv) = entries.remove(index) else {
                            unreachable!("HashMap entry is not a key/value pair")
                        };
                        let mut kv = kv;
                        MirValue::Enum {
                            variant: 1,
                            fields: vec![kv.remove(1)],
                        }
                    }
                    None => MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    },
                })
            }
            HashMapClear => {
                let recv = args.next();
                self.mutate_vec_ref(&recv, |entries| {
                    entries.clear();
                    MirValue::Unit
                })
            }
            HashMapKeysIterNew | HashSetIterNew => {
                // A TRUE borrowed cursor: [map-ref, cursor] — Next indexes the live map.
                let map_ref = args
                    .next()
                    .ok_or_else(|| MirRunError::Internal("KeysIterNew missing receiver".into()))?;
                Ok(MirValue::Aggregate(vec![map_ref, MirValue::Int(0)]))
            }
            HashMapKeysIterNext | HashSetIterNext => {
                let Some(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path,
                }) = args.next()
                else {
                    return self.internal("KeysIterNext expects a &mut iterator reference");
                };
                self.check_ref_live(frame, generation)?;
                let iter_value = self.read_resolved(frame, local, &path)?;
                let MirValue::Aggregate(fields) = &iter_value else {
                    return self.internal(format!("keys iterator referent is {iter_value:?}"));
                };
                let (map_ref, cursor) = match (fields.first(), fields.get(1)) {
                    (Some(r @ MirValue::Ref { .. }), Some(MirValue::Int(c))) => (r.clone(), *c),
                    other => {
                        return self.internal(format!("malformed keys-iterator state {other:?}"))
                    }
                };
                let MirValue::Ref {
                    frame: mf,
                    generation: mg,
                    local: ml,
                    path: mp,
                } = map_ref
                else {
                    unreachable!("matched Ref above");
                };
                self.check_ref_live(mf, mg)?;
                let len = match self.read_resolved(mf, ml, &mp)? {
                    MirValue::Vec(entries) => entries.len(),
                    other => return self.internal(format!("HashMap referent is {other:?}")),
                };
                if (cursor as usize) < len {
                    let mut cursor_path = path;
                    cursor_path.push(ConcreteProj::Field(1));
                    self.write_resolved(frame, local, &cursor_path, MirValue::Int(cursor + 1))?;
                    let mut key_path = mp;
                    key_path.push(ConcreteProj::Index(cursor as usize));
                    key_path.push(ConcreteProj::Field(0));
                    Ok(MirValue::Enum {
                        variant: 1,
                        fields: vec![MirValue::Ref {
                            frame: mf,
                            generation: mg,
                            local: ml,
                            path: key_path,
                        }],
                    })
                } else {
                    Ok(MirValue::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    })
                }
            }
            other => self.internal(format!("runtime {other:?} is not a HashMap op")),
        }
    }

    /// A `&K` key argument, dereferenced to the key value for structural comparison.
    /// 0.1-A6 (A4 slicing): slice ops. A `&[T]` value is a `Ref` whose path ends with a
    /// `ConcreteProj::Slice` window; `SliceNew` composes windows (re-slicing never stacks two
    /// Slice steps) and TRAPS IndexOutOfBounds on a negative, inverted, or out-of-range bound
    /// with the CALL SITE's provenance, per the 06-Standard-Library behavioral requirement.
    fn run_slice_runtime(
        &mut self,
        rt: RuntimeFn,
        args: Vec<MirValue>,
        call_info: SourceInfo,
    ) -> Result<MirValue, MirRunError> {
        use RuntimeFn::*;
        let mut args = args.into_iter();
        let Some(receiver) = args.next() else {
            return self.internal("slice op expects a receiver");
        };
        if let MirValue::ByteSlice(bytes) = receiver {
            return match rt {
                SliceLen => Ok(MirValue::Int(bytes.len() as i128)),
                SliceIsEmpty => Ok(MirValue::Bool(bytes.is_empty())),
                SliceNew => {
                    let oob = || MirRunError::Trap {
                        category: TrapCategory::IndexOutOfBounds,
                        source: call_info,
                        message: None,
                    };
                    let int = |v: Option<MirValue>| -> Result<i128, MirRunError> {
                        match v {
                            Some(MirValue::Int(i)) => Ok(i),
                            other => Err(MirRunError::Internal(format!(
                                "SliceNew bound is not an integer: {other:?}"
                            ))),
                        }
                    };
                    let lo = int(args.next())?;
                    let hi = int(args.next())?;
                    let inclusive = match args.next() {
                        Some(MirValue::Bool(b)) => b,
                        other => {
                            return self.internal(format!(
                                "SliceNew inclusive flag is not Bool: {other:?}"
                            ))
                        }
                    };
                    if lo < 0 || hi < 0 {
                        return Err(oob());
                    }
                    let start = lo as usize;
                    let end = if inclusive {
                        hi as usize + 1
                    } else {
                        hi as usize
                    };
                    if start > end || end > bytes.len() {
                        return Err(oob());
                    }
                    Ok(MirValue::ByteSlice(std::rc::Rc::from(&bytes[start..end])))
                }
                SliceNewMut => self.internal("cannot take a mutable slice of str.bytes()"),
                other => self.internal(format!("runtime {other:?} is not a slice op")),
            };
        };
        let MirValue::Ref {
            frame,
            generation,
            local,
            path,
        } = receiver
        else {
            return self.internal("slice op expects a reference receiver");
        };
        self.check_ref_live(frame, generation)?;
        match rt {
            // 0.1-A8: the shared and exclusive constructors compute the SAME window; they
            // differ only in the reference they yield, and the interpreter's `Ref` already
            // carries no mutability of its own — write permission is a static property the
            // verifier enforces (`SliceNewMut` requires an exclusive receiver). So one arm.
            SliceNew | SliceNewMut => {
                let oob = || MirRunError::Trap {
                    category: TrapCategory::IndexOutOfBounds,
                    source: call_info,
                    message: None,
                };
                let int = |v: Option<MirValue>| -> Result<i128, MirRunError> {
                    match v {
                        Some(MirValue::Int(i)) => Ok(i),
                        other => Err(MirRunError::Internal(format!(
                            "SliceNew bound is not an integer: {other:?}"
                        ))),
                    }
                };
                let lo = int(args.next())?;
                let hi = int(args.next())?;
                let inclusive = match args.next() {
                    Some(MirValue::Bool(b)) => b,
                    other => {
                        return self
                            .internal(format!("SliceNew inclusive flag is not Bool: {other:?}"))
                    }
                };
                if lo < 0 || hi < 0 {
                    return Err(oob());
                }
                let start = lo as usize;
                let end = if inclusive {
                    hi as usize + 1
                } else {
                    hi as usize
                };
                // Window base: an existing Slice tail composes; otherwise the referent's length.
                let (parent_path, window_start, base_len) = match path.last() {
                    Some(ConcreteProj::Slice { start: s0, len: l0 }) => {
                        (path[..path.len() - 1].to_vec(), *s0, *l0)
                    }
                    _ => {
                        let len = match self.read_resolved(frame, local, &path)? {
                            MirValue::Vec(elems) => elems.len(),
                            MirValue::Aggregate(elems) => elems.len(),
                            other => return self.internal(format!("sliced referent is {other:?}")),
                        };
                        (path, 0, len)
                    }
                };
                if start > end || end > base_len {
                    return Err(oob());
                }
                let mut new_path = parent_path;
                new_path.push(ConcreteProj::Slice {
                    start: window_start + start,
                    len: end - start,
                });
                Ok(MirValue::Ref {
                    frame,
                    generation,
                    local,
                    path: new_path,
                })
            }
            SliceLen | SliceIsEmpty => {
                let len = match path.last() {
                    Some(ConcreteProj::Slice { len, .. }) => *len,
                    other => {
                        return self.internal(format!(
                            "slice receiver has no view window (path tail {other:?})"
                        ))
                    }
                };
                Ok(if matches!(rt, SliceLen) {
                    MirValue::Int(len as i128)
                } else {
                    MirValue::Bool(len == 0)
                })
            }
            other => self.internal(format!("runtime {other:?} is not a slice op")),
        }
    }

    fn deref_key_arg(&self, v: Option<MirValue>) -> Result<MirValue, MirRunError> {
        match v {
            Some(MirValue::Ref {
                frame,
                generation,
                local,
                path,
            }) => {
                self.check_ref_live(frame, generation)?;
                self.read_resolved(frame, local, &path)
            }
            Some(other) => Ok(other),
            None => self.internal("missing key argument"),
        }
    }

    fn read_vec_ref(&self, v: &Option<MirValue>) -> Result<Vec<MirValue>, MirRunError> {
        match v {
            Some(MirValue::Ref {
                frame,
                generation,
                local,
                path,
            }) => {
                self.check_ref_live(*frame, *generation)?;
                match self.read_resolved(*frame, *local, path)? {
                    MirValue::Vec(elems) => Ok(elems),
                    other => self.internal(format!("Vec ref referent is {other:?}")),
                }
            }
            Some(MirValue::Vec(elems)) => Ok(elems.clone()),
            other => self.internal(format!("expected a &Vec argument, got {other:?}")),
        }
    }

    /// Mutate the `Vec` behind a `&mut Vec` reference argument in place, returning the
    /// closure's result.
    fn mutate_vec_ref<R>(
        &mut self,
        v: &Option<MirValue>,
        f: impl FnOnce(&mut Vec<MirValue>) -> R,
    ) -> Result<R, MirRunError> {
        let Some(MirValue::Ref {
            frame,
            generation,
            local,
            path,
        }) = v
        else {
            return self.internal(format!("expected a &mut Vec argument, got {v:?}"));
        };
        self.check_ref_live(*frame, *generation)?;
        let (frame, local, path) = (*frame, *local, path.clone());
        let mut vec = match self.read_resolved(frame, local, &path)? {
            MirValue::Vec(elems) => elems,
            other => return self.internal(format!("&mut Vec referent is {other:?}")),
        };
        let out = f(&mut vec);
        self.write_resolved(frame, local, &path, MirValue::Vec(vec))?;
        Ok(out)
    }

    /// The content of a `&str` argument (a `Str` value; a `String`/`Ref` is a lowering bug).
    fn as_str(&self, v: &Option<MirValue>) -> Result<std::rc::Rc<str>, MirRunError> {
        match v {
            Some(MirValue::Str(s)) => Ok(s.clone()),
            other => self.internal(format!("expected a &str argument, got {other:?}")),
        }
    }

    /// Resolve a `&String`/`&mut String` reference argument to a snapshot of the referent.
    fn read_string_ref(&self, v: &Option<MirValue>) -> Result<String, MirRunError> {
        match v {
            Some(MirValue::Ref {
                frame,
                generation,
                local,
                path,
            }) => {
                self.check_ref_live(*frame, *generation)?;
                match self.read_resolved(*frame, *local, path)? {
                    MirValue::String(s) => Ok(s),
                    MirValue::Str(s) => Ok(s.to_string()),
                    other => self.internal(format!("String ref referent is {other:?}")),
                }
            }
            Some(MirValue::String(s)) => Ok(s.clone()),
            other => self.internal(format!("expected a &String argument, got {other:?}")),
        }
    }

    /// Mutate the `String` behind a `&mut String` reference argument in place.
    fn mutate_string_ref(
        &mut self,
        v: &Option<MirValue>,
        f: impl FnOnce(&mut String),
    ) -> Result<(), MirRunError> {
        let Some(MirValue::Ref {
            frame,
            generation,
            local,
            path,
        }) = v
        else {
            return self.internal(format!("expected a &mut String argument, got {v:?}"));
        };
        self.check_ref_live(*frame, *generation)?;
        let (frame, local, path) = (*frame, *local, path.clone());
        let mut s = match self.read_resolved(frame, local, &path)? {
            MirValue::String(s) => s,
            other => return self.internal(format!("&mut String referent is {other:?}")),
        };
        f(&mut s);
        self.write_resolved(frame, local, &path, MirValue::String(s))
    }

    /// Resolve a `&str` operand (a `Str` value) to its content — used for `Trap.message`.
    fn eval_str_operand(&mut self, here: usize, op: &Operand) -> Result<String, MirRunError> {
        match self.eval_operand(here, op)? {
            MirValue::Str(s) => Ok(s.to_string()),
            MirValue::String(s) => Ok(s),
            other => self.internal(format!("trap message operand is {other:?}")),
        }
    }
}

/// WP-C6.3d/CD-138: how a map operation decides key identity — see `map_key_eq`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KeyEqMode {
    /// A primitive or `String` key: Rust equality IS its lawful STARK `Eq`.
    Structural,
    /// A nominal key: the body index of its selected `Eq::eq`.
    UserEq(usize),
    /// A nominal key with NO recorded `Eq` instance — a compiler defect, never a fallback.
    MissingForNominal,
}

/// CD-140 (NUM-FLOAT-FORMAT-001): force a scalar float into the precision of its DECLARED type.
///
/// `MirValue::Float` is an f64 whatever the STARK type says, so a `Float32` local could hold a
/// value no binary32 can represent. That is not a representation detail — it is observable: an
/// overflowing `Float32` product stayed finite instead of becoming `inf`, so `inf - inf` gave
/// `0.0` rather than `NaN`, and `0.1f32 as Float64` widened a value that had never been narrowed.
///
/// Non-float values and non-`Float32` types pass through untouched. `Float64` needs no rounding —
/// the carrier already IS binary64.
fn narrow_to_declared_width(value: MirValue, ty: &MirTy) -> MirValue {
    match (value, ty) {
        (MirValue::Float(f), MirTy::Float32) => MirValue::Float(f64::from(f as f32)),
        (value, _) => value,
    }
}

fn is_vec_runtime(rt: RuntimeFn) -> bool {
    use RuntimeFn::*;
    matches!(
        rt,
        VecNew
            | VecWithCapacity
            | VecPush
            | VecPop
            | VecLen
            | VecIsEmpty
            | VecIndexGet
            | VecReplace
            | VecRemove
            | VecClear
            | VecIterNew
            | VecIterNext
            | VecGetRef
            | VecGetMutRef
    )
}

fn is_box_runtime(rt: RuntimeFn) -> bool {
    use RuntimeFn::*;
    matches!(rt, BoxNew | BoxIntoInner)
}

fn is_slice_runtime(rt: RuntimeFn) -> bool {
    use RuntimeFn::*;
    matches!(rt, SliceNew | SliceNewMut | SliceLen | SliceIsEmpty)
}

fn is_map_runtime(rt: RuntimeFn) -> bool {
    use RuntimeFn::*;
    matches!(
        rt,
        HashMapNew
            | HashMapInsert
            | HashMapGet
            | HashMapLen
            | HashMapIsEmpty
            | HashMapContainsKey
            | HashMapKeysIterNew
            | HashMapKeysIterNext
            | HashMapRemove
            | HashMapClear
            // DEV-116: the set family shares the map group because it shares the representation
            // and, crucially, the `key_eq` dispatch this group threads through every operation.
            | HashSetNew
            | HashSetInsert
            | HashSetRemove
            | HashSetContains
            | HashSetLen
            | HashSetIsEmpty
            | HashSetClear
            | HashSetIterNew
            | HashSetIterNext
    )
}

/// A Char argument (a Unicode scalar codepoint carried as `MirValue::Int`).
fn char_of(v: &Option<MirValue>) -> Result<char, MirRunError> {
    match v {
        Some(MirValue::Int(cp)) => u32::try_from(*cp)
            .ok()
            .and_then(char::from_u32)
            .ok_or_else(|| MirRunError::Internal(format!("invalid Char codepoint {cp}"))),
        other => Err(MirRunError::Internal(format!(
            "expected a Char argument, got {other:?}"
        ))),
    }
}

fn u32_of(v: &Option<MirValue>) -> Result<u32, MirRunError> {
    match v {
        Some(MirValue::Int(value)) => u32::try_from(*value)
            .map_err(|_| MirRunError::Internal(format!("invalid UInt32 value {value}"))),
        other => Err(MirRunError::Internal(format!(
            "expected a UInt32 argument, got {other:?}"
        ))),
    }
}

fn usize_of(v: &Option<MirValue>) -> Result<usize, MirRunError> {
    match v {
        Some(MirValue::Int(value)) => usize::try_from(*value)
            .map_err(|_| MirRunError::Internal(format!("invalid usize value {value}"))),
        other => Err(MirRunError::Internal(format!(
            "expected an integer argument, got {other:?}"
        ))),
    }
}

fn int_arg(v: Option<MirValue>) -> Result<i128, MirRunError> {
    match v {
        Some(MirValue::Int(i)) => Ok(i),
        other => Err(MirRunError::Internal(format!(
            "expected an integer argument, got {other:?}"
        ))),
    }
}

/// Wrap an optional element as a `MirValue` `Option` enum (CoreOption: v0 = None, v1 = Some).
fn option_value(v: Option<MirValue>) -> MirValue {
    match v {
        Some(inner) => MirValue::Enum {
            variant: 1,
            fields: vec![inner],
        },
        None => MirValue::Enum {
            variant: 0,
            fields: Vec::new(),
        },
    }
}

/// Outcome of a checked/trapping primitive (A5). `Trap(None)` traps with the terminator's own
/// category; `Trap(Some(cat))` overrides it — a shift with a bad count is `InvalidShift` even
/// though the terminator's default category is `IntegerOverflow`.
pub(crate) enum CheckedOutcome {
    Value(MirValue),
    Trap(Option<TrapCategory>),
}

impl From<Option<MirValue>> for CheckedOutcome {
    fn from(opt: Option<MirValue>) -> Self {
        match opt {
            Some(v) => CheckedOutcome::Value(v),
            None => CheckedOutcome::Trap(None),
        }
    }
}

/// Bit width of an integer MIR type (A5, for the NUM-SHIFT-001 count bound).
fn int_width(ty: &MirTy) -> Option<u32> {
    Some(match ty {
        MirTy::Int8 | MirTy::UInt8 => 8,
        MirTy::Int16 | MirTy::UInt16 => 16,
        MirTy::Int32 | MirTy::UInt32 => 32,
        MirTy::Int64 | MirTy::UInt64 => 64,
        _ => return None,
    })
}

fn int_range(ty: &MirTy) -> Option<(i128, i128)> {
    Some(match ty {
        MirTy::Int8 => (i128::from(i8::MIN), i128::from(i8::MAX)),
        MirTy::Int16 => (i128::from(i16::MIN), i128::from(i16::MAX)),
        MirTy::Int32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        MirTy::Int64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
        MirTy::UInt8 => (0, i128::from(u8::MAX)),
        MirTy::UInt16 => (0, i128::from(u16::MAX)),
        MirTy::UInt32 => (0, i128::from(u32::MAX)),
        MirTy::UInt64 => (0, i128::from(u64::MAX)),
        _ => return None,
    })
}
