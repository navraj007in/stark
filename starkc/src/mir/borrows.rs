//! **WP-ARCH-CLOSE AC1 — the single authority for "which storage does this value borrow?"**
//!
//! # Why this module exists, and why it is here rather than in the backend
//!
//! DEV-160's call-site thunk has to know whether a reference reaching a call borrows the same slot
//! a sibling argument moves out of. It answered that with a `borrow_provenance` heuristic living in
//! `backend/generated_rust/emit_call_thunk.rs` — a *may-derive-from* over-approximation, computed
//! in the emitter, downstream of every phase with authority over ownership.
//!
//! The deviation's own entry named the replacement and why the interim was temporary:
//!
//! > *"Provenance answers 'this value may derive from that slot', not 'a live borrow of that slot
//! > reaches this call'. Those differ, and the first attempt is proof that the gap has teeth. …
//! > absence of a counterexample is not a proof of precision."*
//!
//! Owner decision (CE3, 2026-08-12) placed the analysis **here**, in MIR, for a reason worth
//! stating: the question is about MIR places and MIR dataflow. The HIR borrow checker answers a
//! different question — *is this program legal* — and returns only diagnostics. This one answers
//! *what does this value borrow, in the lowered form*, and MIR is the phase that owns lowered form.
//! A backend computing it was reconstructing downstream what no phase had ever published.
//!
//! # What "precise" buys, concretely
//!
//! The heuristic propagated a call's arguments into its result unconditionally. So in
//!
//! ```text
//! _9 = call send(_8, move _2.2)      // send(u: &str, b: String) -> UInt64
//! ```
//!
//! `_9` — a `UInt64` — was recorded as possibly borrowing `_2`. **A scalar cannot borrow
//! anything.** Two rules remove that whole class:
//!
//! ```text
//! a result that cannot STORE a reference borrows nothing, whatever its arguments were
//! only REFERENCE arguments can pass a borrow into a result (03 rule 3, shortest input)
//! ```
//!
//! # What this module does NOT yet answer
//!
//! Origins are a *dataflow* relation: which storage a value borrows. Absorption additionally needs
//! *liveness* — whether the producing statement dominates the consuming call on one straight-line
//! path with nothing observable between — which is DEV-160b's second half and is not implemented
//! here. This module deliberately stops at the relation, so that what it does claim is exact.
//!
//! # Direction of error
//!
//! Still conservative where it is unsure: an unrecognised producer contributes nothing to the
//! result, but any *recognised* borrow-carrying path is recorded. Being wrong by refusing a program
//! that could have compiled is recoverable; being wrong the other way emits code rustc rejects for
//! reasons the user cannot act on.

use super::{MirBody, MirTy, Operand, Rvalue, Statement, Terminator};
use std::collections::{BTreeMap, BTreeSet};

/// Which locals' storage each local's value borrows.
#[derive(Debug, Default, Clone)]
pub(crate) struct BorrowOrigins {
    by_local: BTreeMap<u32, BTreeSet<u32>>,
}

impl BorrowOrigins {
    /// The locals whose storage `local`'s value borrows, or `None` when it borrows nothing.
    pub(crate) fn of(&self, local: u32) -> Option<&BTreeSet<u32>> {
        self.by_local.get(&local)
    }
}

/// The type of a local, or `Unit` for an out-of-range index — which lowering does not produce, and
/// which a defensive default here must not turn into a silent "borrows nothing".
fn local_ty(body: &MirBody, local: u32) -> Option<&MirTy> {
    body.locals.get(local as usize).map(|l| &l.ty)
}

/// Whether an operand is a reference-typed value, and therefore able to carry a borrow into a
/// call's result under 03 rule 3.
fn operand_carries(body: &MirBody, operand: &Operand) -> Option<u32> {
    let place = match operand {
        Operand::Copy(p) | Operand::Move(p) => p,
        Operand::Const(_) => return None,
    };
    let ty = local_ty(body, place.local.0)?;
    // **The authority, not a fourth copy of it.** `reference_rule::stores_a_reference` is AS4's
    // consolidated answer to "does this type STORE a reference?"; the backend carried a private
    // re-implementation (`may_carry_borrow`) that agreed on every arm but re-asserted the property
    // behind a wildcard the authority deliberately refuses.
    if super::reference_rule::stores_a_reference(ty) {
        Some(place.local.0)
    } else {
        None
    }
}

/// Compute the borrow-origin relation for one body.
pub(crate) fn origins(body: &MirBody) -> BorrowOrigins {
    let mut by_local: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();

    // `&place` seeds the relation: the reference borrows the storage of `place`'s base local.
    for block in &body.blocks {
        for (statement, _) in &block.statements {
            if let Statement::Assign(dest, Rvalue::RefOf { place, .. }) = statement {
                if dest.projection.is_empty()
                    && local_ty(body, dest.local.0)
                        .is_some_and(super::reference_rule::stores_a_reference)
                {
                    by_local
                        .entry(dest.local.0)
                        .or_default()
                        .insert(place.local.0);
                }
            }
        }
    }

    // A small fixpoint. Bodies are small and the relation only grows, so this settles in a couple of
    // rounds; iterating rather than making one pass matters because MIR block order is not
    // definition order — a value can be defined in a later block than the one that reads it.
    let mut changed = true;
    while changed {
        changed = false;
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

                    // A copy propagates whatever the source borrowed.
                    Rvalue::Use(Operand::Copy(p)) => vec![p.local.0],

                    // **A move SEVERS the relation.** After `let url = builder.url;` the value owns
                    // its own storage, and borrowing `url` does not borrow `builder`. Propagating
                    // through moves is precisely the rule that over-refused
                    // `stark_http_client::follow` on the first repair attempt.
                    Rvalue::Use(Operand::Move(_)) => Vec::new(),

                    // A borrow-carrying aggregate carries what its borrow-carrying components
                    // carried. A component that cannot store a reference contributes nothing —
                    // `(String, &str)` borrows through its second element only.
                    Rvalue::Aggregate(_, operands) => operands
                        .iter()
                        .filter_map(|o| operand_carries(body, o))
                        .collect(),

                    _ => Vec::new(),
                };
                // **A destination that cannot STORE a reference borrows nothing, by construction.**
                //
                // This is the invariant that lets consumers stop compensating. The heuristic this
                // module replaces recorded scalars as borrowing aggregates, and every consumer
                // carried its own type check to undo that — `emit_call_thunk` had two. A consumer
                // patching an authority's output is the shape §4 calls out; stating the rule once,
                // here, removes the need for it.
                if local_ty(body, dest.local.0)
                    .is_some_and(super::reference_rule::stores_a_reference)
                {
                    merge(&mut by_local, dest.local.0, &sources, &mut changed);
                }
            }

            // A call's result may borrow what its arguments borrowed — STARK's shortest-input rule
            // (03 rule 3) read as a may-alias relation. Two precision rules apply, and together
            // they are the difference between this module and the heuristic it replaces.
            if let Terminator::Call { args, dest, .. } = &block.terminator.0 {
                // 1. A result that cannot STORE a reference borrows nothing, whatever came in.
                //    `send(u: &str, b: String) -> UInt64` returns a scalar; a scalar borrows
                //    nothing, and the heuristic recorded it as borrowing the aggregate.
                let returns_borrow = local_ty(body, dest.local.0)
                    .is_some_and(super::reference_rule::stores_a_reference);
                if returns_borrow {
                    // 2. Only REFERENCE arguments can pass a borrow through. A `String` moved in
                    //    by value cannot leak its storage into the result.
                    let sources: Vec<u32> = args
                        .iter()
                        .filter_map(|o| operand_carries(body, o))
                        .collect();
                    merge(&mut by_local, dest.local.0, &sources, &mut changed);
                }
            }
        }
    }

    BorrowOrigins { by_local }
}

fn merge(by_local: &mut BTreeMap<u32, BTreeSet<u32>>, into: u32, from: &[u32], changed: &mut bool) {
    let mut union: BTreeSet<u32> = BTreeSet::new();
    for local in from {
        if let Some(set) = by_local.get(local) {
            union.extend(set.iter().copied());
        }
        // A reference's own seed is itself a source: `_8 = copy _7` where `_7 = &_2.0` reaches
        // `_2` through `_7`'s set, which the seeding pass above already established.
    }
    if union.is_empty() {
        return;
    }
    let target = by_local.entry(into).or_default();
    let before = target.len();
    target.extend(union);
    if target.len() != before {
        *changed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::lower::lower_program;
    use crate::parser::{parse, ParseMode};
    use crate::resolve::resolve;
    use crate::source::SourceFile;
    use crate::typecheck;
    use std::sync::Arc;

    /// Lowers a program and returns `main`'s origin relation.
    fn origins_of_main(source: &str) -> (BorrowOrigins, crate::mir::MirBody) {
        let file = Arc::new(SourceFile::new("borrows_test.stark", source.to_string()));
        let (ast, pd) = parse(&file, ParseMode::Program);
        assert!(pd.is_empty(), "parse: {pd:?}");
        let (hir, rd) = resolve(&ast, file.clone());
        assert!(rd.is_empty(), "resolve: {rd:?}");
        let checked = typecheck::analyze(&hir);
        let errors: Vec<_> = checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == crate::diag::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "typecheck: {errors:?}");
        let registered = hir.source_named(&file.name).expect("registered");
        let program = lower_program(&hir, &checked.tables, registered)
            .unwrap_or_else(|e| panic!("lowering: {}", e.what));
        let body = program
            .bodies
            .iter()
            .find(|b| format!("{:?}", b.instance).contains("main"))
            .expect("main was lowered")
            .clone();
        (origins(&body), body)
    }

    /// Renders the whole relation as `local:ty -> {slots}` lines, sorted. A characterization of
    /// the ENTIRE result, not a spot check.
    fn render(origins: &BorrowOrigins, body: &crate::mir::MirBody) -> String {
        let mut lines: Vec<String> = Vec::new();
        for (local, slots) in &origins.by_local {
            let ty = &body.locals[*local as usize].ty;
            let carries = super::super::reference_rule::stores_a_reference(ty);
            let slots: Vec<String> = slots.iter().map(|s| format!("_{s}")).collect();
            lines.push(format!(
                "_{local} carries={carries} -> {{{}}}",
                slots.join(", ")
            ));
        }
        lines.join("\n")
    }

    /// **The characterization control.**
    ///
    /// The three rule-specific tests below were each written first and each turned out NOT to kill
    /// a mutation of the rule it named: in source-derived programs the seed guard, the statement
    /// guard and the call guard mask one another, so removing any one of them changed nothing
    /// observable. That is worth recording rather than hiding — a test that cannot fail is not a
    /// control, and three of them looked like coverage.
    ///
    /// This pins the ENTIRE relation instead. Any rule change that alters what is recorded, for any
    /// local, fails here and has to be explained. It is a characterization test and makes no claim
    /// that the pinned relation is *correct* — the tests below argue that separately.
    #[test]
    fn the_whole_relation_is_pinned_for_the_reported_shape() {
        let (origins, body) = origins_of_main(
            r#"
struct Req { url: String, body: String }
fn send(u: &str, b: String) -> UInt64 { u.len() + b.len() }
fn main() {
    let r = Req { url: String::from("abc"), body: String::from("de") };
    let n = send(r.url.as_str(), r.body);
    if n != 5u64 { panic("bad"); }
}
"#,
        );
        let rendered = render(&origins, &body);
        assert_eq!(
            rendered, "_9 carries=true -> {_1}\n_10 carries=true -> {_1}",
            "the borrow-origin relation changed. Every entry must be a value that CAN store a \
             reference, and the two here are `&r.url` and the `&str` `as_str` returns from it. \
             If a rule moved, say which and why."
        );
    }

    /// **The precision rule that motivated this module.**
    ///
    /// `send(u: &str, b: String) -> UInt64` takes a reference and returns a scalar. The heuristic
    /// this replaced propagated a call's arguments into its result unconditionally, so the `UInt64`
    /// was recorded as borrowing the aggregate — and every consumer carried its own type check to
    /// undo that. **No local whose type cannot store a reference may have an origin.**
    ///
    /// Asserted over the whole body rather than one local, so it cannot pass by naming the one case
    /// that happens to work.
    #[test]
    fn no_value_that_cannot_store_a_reference_has_an_origin() {
        let (origins, body) = origins_of_main(
            r#"
struct Req { url: String, body: String }
fn send(u: &str, b: String) -> UInt64 { u.len() + b.len() }
fn main() {
    let r = Req { url: String::from("abc"), body: String::from("de") };
    let n = send(r.url.as_str(), r.body);
    if n != 5u64 { panic("bad"); }
}
"#,
        );
        for (local, slots) in &origins.by_local {
            let ty = &body.locals[*local as usize].ty;
            assert!(
                super::super::reference_rule::stores_a_reference(ty),
                "_{local}: {ty:?} cannot store a reference, yet it is recorded as borrowing \
                 {slots:?}. This is the exact over-approximation the heuristic made, and the \
                 reason its consumers had to compensate."
            );
        }
    }

    /// **Which of this module's rules any program actually reaches — measured, not assumed.**
    ///
    /// Written after the mutation trials below returned an uncomfortable answer. Three rules were
    /// mutated; only one was killed:
    ///
    /// ```text
    /// FIRST TRIAL, simple program only
    ///   call-result guard    (`returns_borrow`)     KILLED by 2
    ///   move severs          (Move -> no sources)   SURVIVED
    ///   statement dest guard (stores_a_reference)   SURVIVED
    ///
    /// AFTER pinning the borrow-carrying-tuple relation below
    ///   call-result guard                           KILLED by 3   CONTROLLED
    ///   move severs                                 KILLED by 1   CONTROLLED
    ///   statement dest guard                        SURVIVED      uncontrolled
    ///   aggregate component filter                  SURVIVED      uncontrolled
    /// ```
    ///
    /// The first trial's survivors were not weak rules but an unreaching program: in the simple
    /// shape a `String` moved out of an aggregate is not borrow-carrying, so the move rule and the
    /// dest guard both decline for the same value and removing either changes nothing. Adding the
    /// `(String, &str)` shape — non-`Copy` and borrow-carrying — reached the move rule and made it
    /// falsifiable.
    ///
    /// **Two rules remain uncontrolled and are recorded as such**: the statement dest guard and the
    /// aggregate component filter. Both survived a mutation that was verified to apply, so they are
    /// precautionary rather than verified. That is the honest reading of AC4's standard — an
    /// authority may be called verified only where a control exists — and it is recorded here
    /// rather than left for a reviewer to find.
    ///
    /// This test asserts only what it can: that the census runs and reports. It is a measurement,
    /// not a control, and is labelled so it is never counted as one.
    #[test]
    fn census_which_rules_a_program_reaches() {
        let (origins, body) = origins_of_main(
            r#"
struct Req { url: String, body: String }
fn main() {
    let r = Req { url: String::from("abc"), body: String::from("de") };
    let t: (String, &str) = (String::from("x"), r.url.as_str());
    let u = t;
    if u.0.len() != 1u64 { panic("bad"); }
}
"#,
        );
        println!("CENSUS relation: {}", render(&origins, &body));
        // **Pinned, which is what makes the move and aggregate rules falsifiable.** The census
        // above found that this program DOES reach them — `_11` and `_12` are the borrow-carrying
        // tuple, and `_16` borrows a different local — where the simpler program reached only the
        // seed and the call rule. Pinning the richer relation is what turns those rules from
        // precautionary into controlled.
        assert_eq!(
            render(&origins, &body),
            "_10 carries=true -> {_1}\n_11 carries=true -> {_1}\n_12 carries=true -> {_1}\n\
             _16 carries=true -> {_13}",
            "the borrow-origin relation changed for the borrow-carrying-tuple shape. If a rule \
             moved, say which and why."
        );
        let borrow_carrying_tuples = body
            .locals
            .iter()
            .filter(|l| {
                matches!(&l.ty, MirTy::Tuple(_))
                    && super::super::reference_rule::stores_a_reference(&l.ty)
            })
            .count();
        println!("CENSUS borrow-carrying tuple locals lowered: {borrow_carrying_tuples}");
        println!(
            "CENSUS locals with an origin: {} of {}",
            origins.by_local.len(),
            body.locals.len()
        );
    }

    /// **A move severs the relation** — the rule whose absence over-refused
    /// `stark_http_client::follow` on the first repair attempt.
    ///
    /// After `let url = r.url;` the value owns its own storage. A reference to `url` borrows `url`,
    /// not `r`, so nothing may record it as borrowing `r`'s local.
    #[test]
    fn a_move_severs_the_origin() {
        let (origins, body) = origins_of_main(
            r#"
struct Req { url: String, body: String }
fn send(u: &str, b: String) -> UInt64 { u.len() + b.len() }
fn main() {
    let r = Req { url: String::from("abc"), body: String::from("de") };
    let url = r.url;
    let body = r.body;
    let n = send(url.as_str(), body);
    if n != 5u64 { panic("bad"); }
}
"#,
        );
        // `r`'s local is the one holding the aggregate. Whichever index it is, nothing may claim to
        // borrow it: every reference here derives from a local the aggregate no longer owns.
        let aggregate: Vec<u32> = body
            .locals
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l.ty, MirTy::Struct(_, _)))
            .map(|(i, _)| i as u32)
            .collect();
        assert!(
            !aggregate.is_empty(),
            "the test program declares a struct local"
        );
        for (local, slots) in &origins.by_local {
            for owner in &aggregate {
                assert!(
                    !slots.contains(owner),
                    "_{local} is recorded as borrowing the aggregate _{owner}, but every field was \
                     MOVED to a local first. Propagating through moves is what broke \
                     `stark_http_client::follow`."
                );
            }
        }
    }

    /// A reference really does get an origin — the control that stops the two tests above from
    /// passing on an analysis that simply records nothing at all.
    #[test]
    fn a_reference_records_the_storage_it_borrows() {
        let (origins, _) = origins_of_main(
            r#"
struct Req { url: String, body: String }
fn read(u: &String) -> UInt64 { u.len() }
fn main() {
    let r = Req { url: String::from("abc"), body: String::from("de") };
    if read(&r.url) != 3u64 { panic("bad"); }
}
"#,
        );
        assert!(
            !origins.by_local.is_empty(),
            "`&r.url` must be recorded as borrowing `r`'s storage. An analysis that records \
             nothing would satisfy every other test in this module."
        );
    }
}
