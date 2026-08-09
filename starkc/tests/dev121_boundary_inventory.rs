//! **DEV-121 / AS3 #3 — the exact inventory of value boundaries.**
//!
//! The owner ruling on the reopened DEV-121 is explicit about the order:
//!
//! > **Inventory every value boundary first**, with exact-set evidence like AS3's callable-use
//! > work. […] Don't simply replace the current five calls with `check_value_for_ty` and declare
//! > victory. **The inventory of where values cross typed boundaries is as important as the
//! > relation itself.**
//!
//! This file is that inventory, and it is executable rather than prose: [`classify`] is an
//! **exhaustive match** over `RepBoundary`, so adding a variant fails to compile until someone
//! records where it occurs, what supplies its expected `Ty`, and whether it is wired. That is the
//! mechanism closure condition 7 asks for — a new value-storage or transfer form cannot silently
//! bypass validation.
//!
//! # Why the expected type must be checker-published
//!
//! Every row names a source in the checker's tables — `callable_types[body]` for signatures,
//! `local_types` for bindings, the nominal's declared fields for aggregates. **None reconstructs an
//! expected type from the runtime value**, which would make the relation agree with whatever it was
//! handed and turn the invariant into a tautology.
//!
//! # Status at the time of writing
//!
//! **All twelve `RepBoundary` variants are wired.** — `Return` (`9e5dc3b`), then `Receiver`, `Parameter` and
//! `Propagation` together (AS3 Packet 2). All four live in the invocation authority and all four
//! read one lookup of `callable_types[body]`, so a body cannot be checked on the way out but
//! unchecked on the way in. Wiring `Return` immediately exposed DEV-197 (bodies running with no
//! generic environment); wiring `Receiver` exposed the destructor representation collision, which
//! receiver materialization repaired rather than exempted. Packet 3 then added `LetBinding`,
//! `MatchBinding` and `LoopBinding` through one `bind_typed_local` funnel — and found that the
//! USER-iterator `for` form checked nothing at all, a second spelling of a boundary only one of
//! whose spellings was covered. The remaining four are `Unwired`, each with its site and type
//! source identified so the work is bounded rather than exploratory. Packet 4 then wired the three
//! write boundaries through `write_place` — one path, and *which* write it is follows from the
//! place's last projection rather than from the caller. Only `AggregateField` remains.
//!
//! **The earlier "no local to key on" framing was wrong, and this is where it broke.** A field and
//! an indexed slot both have a checker-published type, because both are named by an EXPRESSION:
//! `expr_types[lhs]` answers for the target whatever the projection depth.
//!
//! Packet 5 closed the last one, `AggregateField`, on `aggregate_field_types[lit][field]` — the
//! field's DECLARED type instantiated for that literal, published where the checker unified the
//! initialisers against it. Deliberately not `expr_types[init]`: that is the type of the expression
//! that produced the value, so it would assert nothing, and a shorthand field (`W { v }`) has no
//! initialiser expression to read at all.
//!
//! **The boundary that had no `RepBoundary` variant is closed.** Inline values entering builtins and
//! runtime operations — named explicitly by the ruling — are covered by
//! `RepBoundary::ExpressionResult` at `expect_value`, the producer-side funnel. It was recorded as a
//! gap rather than folded into `Parameter` precisely so that closing it would be a visible change
//! rather than a redefinition.
//!
//! **No boundary is structurally impossible.** That is a finding, not an omission: the earlier
//! framing assumed struct fields and indexed slots could not be reached because they have no local
//! to key on, but both have a checker-published declared type — the field's, from the nominal — and
//! are reachable through the single `write_place` site.

use starkc::interp::RepBoundary;

/// **The three-way classification the ruling requires.** Every runtime value-transfer or storage
/// funnel gets exactly one, and "not currently seen" is not among them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    /// A boundary requiring the total relation, and it runs here today.
    Wired,
    /// A boundary requiring the total relation; the call is not there yet.
    ///
    /// **Currently unconstructed, and that is the result rather than dead code** — every variant of
    /// `RepBoundary` is `Wired`. Kept because it is the vocabulary the next boundary needs: a new
    /// variant must be classifiable as unwired before it is wired, and deleting this would force
    /// whoever adds one to either wire it in the same commit or invent a word for "not yet".
    #[allow(dead_code)]
    Unwired,
    /// Cannot carry a mis-represented value, with a written reason.
    #[allow(dead_code)]
    StructurallyImpossible,
    /// Not a boundary: the value does not come to rest here, with a written reason.
    #[allow(dead_code)]
    NotABoundary,
}

struct Row {
    /// Where in `interp.rs` the value crosses the boundary.
    site: &'static str,
    /// What supplies the expected `Ty`. Never the runtime value.
    expected_ty_from: &'static str,
    class: Class,
}

/// **EXHAUSTIVE ON PURPOSE.** A new `RepBoundary` variant breaks this match, and the only way to
/// fix it is to state the variant's site, its type source and its status.
fn classify(boundary: RepBoundary) -> Row {
    match boundary {
        RepBoundary::Return => Row {
            site: "execute_body — Flow::Value | Flow::Return, after the frame pop",
            expected_ty_from: "callable_types[callable.body].ret",
            class: Class::Wired,
        },
        RepBoundary::Propagation => Row {
            site: "execute_body — Flow::Propagate, after the frame pop",
            expected_ty_from: "callable_types[body].ret — a `?` that leaves the body IS its return",
            class: Class::Wired,
        },
        RepBoundary::Parameter => Row {
            site: "execute_body — the params/args zip, before the frame is pushed",
            expected_ty_from: "callable_types[body].params, positionally",
            class: Class::Wired,
        },
        RepBoundary::Receiver => Row {
            site: "execute_body — the receiver insert, after materialization",
            expected_ty_from: "callable_types[body].receiver — as the body BINDS it",
            class: Class::Wired,
        },
        RepBoundary::LetBinding => Row {
            site: "bind_typed_local, from eval_stmt — hir::StmtKind::Let",
            expected_ty_from: "local_types[local]",
            class: Class::Wired,
        },
        RepBoundary::MatchBinding => Row {
            site: "bind_typed_local, from the match arm's pattern bindings",
            expected_ty_from: "local_types[local] for each bound local",
            class: Class::Wired,
        },
        RepBoundary::LoopBinding => Row {
            site: "bind_typed_local, from BOTH `for` forms — built-in iterable and user iterator",
            expected_ty_from: "local_types[local]",
            class: Class::Wired,
        },
        RepBoundary::Assignment => Row {
            site: "write_place — the single place-write path, with an empty projection",
            expected_ty_from: "expr_types[the Assign lhs] — the checker's type for the target",
            class: Class::Wired,
        },
        RepBoundary::FieldWrite => Row {
            site: "write_place — projection ending in Projection::Field",
            expected_ty_from: "expr_types[the Assign lhs] — the field expression's declared type",
            class: Class::Wired,
        },
        RepBoundary::ElementWrite => Row {
            site: "write_place — projection ending in Projection::Index or ::MapIndex",
            expected_ty_from: "expr_types[the Assign lhs] — the element expression's declared type",
            class: Class::Wired,
        },
        RepBoundary::ExpressionResult => Row {
            site: "expect_value — every expression result, before the consumer sees it",
            expected_ty_from: "expr_types[expr] — the checker's type for that expression",
            class: Class::Wired,
        },
        RepBoundary::AggregateField => Row {
            site: "eval_struct_lit — before the per-field `values.insert`",
            expected_ty_from: "aggregate_field_types[lit][field] — the nominal's declared field \
                               type, instantiated for this literal",
            class: Class::Wired,
        },
    }
}

/// Every `RepBoundary` this inventory knows about. Kept beside `classify` so the two can disagree
/// and be caught, rather than one list serving both purposes.
const ALL: &[RepBoundary] = &[
    RepBoundary::LetBinding,
    RepBoundary::Parameter,
    RepBoundary::Receiver,
    RepBoundary::Return,
    RepBoundary::Propagation,
    RepBoundary::MatchBinding,
    RepBoundary::LoopBinding,
    RepBoundary::Assignment,
    RepBoundary::FieldWrite,
    RepBoundary::ElementWrite,
    RepBoundary::AggregateField,
    RepBoundary::ExpressionResult,
];

#[test]
fn every_boundary_has_a_site_and_a_checker_published_type_source() {
    for boundary in ALL {
        let row = classify(*boundary);
        assert!(
            !row.site.is_empty(),
            "{boundary:?}: no site recorded — where does the value cross?"
        );
        assert!(
            !row.expected_ty_from.is_empty(),
            "{boundary:?}: no expected-type source recorded"
        );
        assert!(
            matches!(row.class, Class::Wired | Class::Unwired),
            "{boundary:?}: every `RepBoundary` variant is a real boundary; \
             StructurallyImpossible/NotABoundary are for funnels that are NOT RepBoundary variants"
        );
        // The rule that keeps the relation from becoming a tautology.
        let src = row.expected_ty_from;
        assert!(
            src.contains("callable_types")
                || src.contains("local_types")
                || src.contains("declared")
                || src.contains("expr_types"),
            "{boundary:?}: the expected type must come from the CHECKER's tables, not from the \
             runtime value; got {src:?}"
        );
    }
}

/// The list and the exhaustive match must cover the same set. If a variant is added to
/// `RepBoundary`, `classify` fails to compile; if it is added to `classify` but not to `ALL`, this
/// catches it.
#[test]
fn the_inventory_list_and_the_classifier_agree() {
    assert_eq!(
        ALL.len(),
        12,
        "RepBoundary has 12 variants; update ALL and `classify` together"
    );
    let mut seen: Vec<String> = ALL.iter().map(|b| format!("{b:?}")).collect();
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), ALL.len(), "ALL contains a duplicate");
}

/// **The producer-side funnel, measured.**
///
/// The ruling asks whether a single interpreter funnel exists through which evaluated expression
/// results reliably pass, so inline values cannot bypass validation merely by never binding to a
/// local. There is one: `expect_value(expr)`, which carries the `ExprId` and therefore has
/// `expr_types[expr]` — the checker's answer for that expression.
///
/// Census of every expression-result path (counts asserted below so the claim cannot rot):
///
/// ```text
/// expect_value               28 callers   ← the funnel; has ExprId
/// expect_bool / expect_int     8 callers   ← delegate to expect_value
/// direct eval_expr             6 sites     ← 1 is expect_value itself; 5 are NOT boundaries
/// ```
///
/// The five direct sites are `eval_block`'s tail, `eval_stmt`'s expression statement, the `else`
/// branch, a match arm's body, and one nested expression case. **None is a boundary**: the value
/// flows to the *enclosing* expression rather than coming to rest, and lands at one of the eleven
/// boundaries when it finally does. Classified `NotABoundary`, with that reason, rather than left
/// as an unexplained gap.
///
/// The producer-side assertion at `expect_value` (Packet 6) does not REPLACE the destination
/// checks — both consume the same `check_value_for_ty`, so it adds defence, not a second
/// authority. It is what covers the values that never reach a destination boundary at all.
#[test]
fn the_producer_side_funnel_is_expect_value() {
    let source = interp_source();
    let funnel = source.matches("self.expect_value(").count();
    let direct = source.matches("self.eval_expr(").count();
    assert!(
        funnel >= 25,
        "expect_value should remain the dominant expression-result path; found {funnel}"
    );
    // **An exact count, not a bound.** The final audit required every direct `eval_expr` consumer
    // to be classified, and a `<=` bound let a new one appear without review. It did: the
    // interpolation field's non-place branch consumed an expression result with no producer check
    // at all (DEV-203), and a bound of eight had room for it.
    assert_eq!(
        direct, 6,
        "every direct `eval_expr` consumer must be classified. Found {direct}, expected 6:\n\
         \x20 1. `expect_value` itself — the funnel\n\
         \x20 2. `eval_block`'s tail — flows to the block's value, NotABoundary\n\
         \x20 3. `StmtKind::Expr` — the value is dropped, not stored\n\
         \x20 4. an interpolation field — a CHECKED consumer, `check_expr_value` (DEV-203)\n\
         \x20 5. an `else` branch — flows to the enclosing `if`\n\
         \x20 6. a match arm body — flows to the enclosing `match`\n\
         A new one is either a typed consumer that must call `check_expr_value`, or a \
         flow-through that must be classified here before it is added."
    );
}

/// **The storage-route forcing pin.** AS3 final audit §9.
///
/// `bind_typed_local` is claimed to be the only way a value comes to rest in a LANGUAGE local. The
/// claim needs a pin, because a new `frame_mut().insert(local, Some(value))` elsewhere would
/// reintroduce exactly the per-site convention Packet 3 replaced — and no behavioural test would
/// notice, since the inserted value is usually correct.
///
/// The two raw `values.insert` calls are the temp-promotion helpers, which deliberately bypass
/// `Frame::insert`: a promoted temp is a view's backing storage, not a value the frame owns and
/// destroys, and registering it in `order` made promoted temps participate in destruction. Those
/// locals are `LocalId(1000000 + …)`, outside the checker's local space, and have no `local_types`
/// entry by construction — which is why `bind_typed_local` can treat a missing entry as an
/// invariant failure.
#[test]
fn typed_local_storage_has_one_funnel() {
    let source = interp_source();
    let typed = source.matches("self.frame_mut().insert(").count();
    assert_eq!(
        typed, 3,
        "found {typed} `frame_mut().insert` sites, expected 3:\n\
         \x20 1. `bind_typed_local` — the checked funnel for let/match/loop bindings\n\
         \x20 2. a `let` with no initialiser — inserts `None`, so there is no value to check\n\
         \x20 3. `promote_to_owned_temp_place` — interpreter-internal backing storage\n\
         A new site puts a value into a local without the `LetBinding`/`MatchBinding`/\
         `LoopBinding` check. Route it through `bind_typed_local`, or classify it here."
    );
    let raw = source.matches(".values.insert(").count();
    assert_eq!(
        raw, 3,
        "found {raw} raw `values.insert` sites, expected 3: `Frame::insert` itself and the two \
         temp-promotion helpers, which must bypass `Frame::order` so promoted temps do not \
         participate in destruction. A new raw insert is untracked storage."
    );
}

/// The CODE of `interp.rs` — comment lines removed, CRLF normalised at the read.
///
/// **Both filters are load-bearing.** The raw-insert census below first counted 4 because a doc
/// comment *describing* `.values.insert(...)` matched the pattern; a census that counts prose is a
/// census that fails for reasons unrelated to what it measures, and worse, one that could be
/// satisfied by editing a comment. CRLF normalisation is the same discipline in the other
/// direction: without it these counts fail on a Windows checkout and nowhere else.
fn interp_source() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/interp.rs"))
        .expect("interp.rs must be readable")
        .replace("\r\n", "\n")
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// **The boundary the ruling named that `RepBoundary` had no variant for — now closed.**
///
/// The closure conditions list "inline values entering builtins/runtime operations". A value passed
/// to `call_builtin` or a `RuntimeFn` never binds to a local, so none of the eleven DESTINATION
/// boundaries could see it: it is not a `Parameter` (that is a user callable's declared parameter,
/// typed by `callable_types`), and folding it into one would have claimed coverage the inventory
/// did not have.
///
/// `RepBoundary::ExpressionResult` names it, and `expect_value` — the funnel every such value
/// passes through, and the one place that still has the `ExprId` — enforces it against
/// `expr_types[expr]`. A builtin argument is an expression result, so it is covered at the producer
/// rather than needing a `BuiltinArgument` variant whose expected type would have been the same
/// table anyway.
///
/// This test now asserts the closure rather than the gap: the variant exists, it is classified, and
/// it is `Wired`.
#[test]
fn inline_values_entering_builtins_are_covered_by_the_producer_boundary() {
    let row = classify(RepBoundary::ExpressionResult);
    assert_eq!(
        row.class,
        Class::Wired,
        "the producer-side boundary is what covers a value that never binds to a local"
    );
    assert!(
        row.site.contains("expect_value"),
        "the funnel is `expect_value`, because it is the one producer path carrying the ExprId; \
         got {:?}",
        row.site
    );
    assert!(
        row.expected_ty_from.contains("expr_types"),
        "the expected type must be the checker's answer for the expression; got {:?}",
        row.expected_ty_from
    );
}

/// **The progress pin.** It records how much of DEV-121's closure has actually landed, so "nearly
/// done" cannot accumulate around it again — which is precisely how the premature 2026-08-07
/// closure happened.
#[test]
fn dev121_wiring_progress_is_recorded_exactly() {
    let wired: Vec<String> = ALL
        .iter()
        .filter(|b| classify(**b).class == Class::Wired)
        .map(|b| format!("{b:?}"))
        .collect();
    assert_eq!(
        wired,
        vec![
            "LetBinding",
            "Parameter",
            "Receiver",
            "Return",
            "Propagation",
            "MatchBinding",
            "LoopBinding",
            "Assignment",
            "FieldWrite",
            "ElementWrite",
            "AggregateField",
            "ExpressionResult",
        ],
        "DEV-121 closes when every boundary is Wired. Update this pin in the same change that \
         wires one, and update AS3 #3/#5 in CAMPAIGN-A-EXIT-REPORT.md when the list is complete."
    );
}
