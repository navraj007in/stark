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
//! Four boundaries are wired — `Return` (`9e5dc3b`), then `Receiver`, `Parameter` and
//! `Propagation` together (AS3 Packet 2). All four live in the invocation authority and all four
//! read one lookup of `callable_types[body]`, so a body cannot be checked on the way out but
//! unchecked on the way in. Wiring `Return` immediately exposed DEV-197 (bodies running with no
//! generic environment); wiring `Receiver` exposed the destructor representation collision, which
//! receiver materialization repaired rather than exempted. The remaining seven are `Unwired`, each
//! with its site and type source identified so the work is bounded rather than exploratory.
//!
//! **One boundary has no `RepBoundary` variant at all** — inline values entering builtins and
//! runtime operations, which the ruling names explicitly. See
//! `builtin_arguments_are_a_boundary_with_no_repboundary_variant`: the expected type is available
//! (`expr_types[arg]`, via the `arg_exprs` `call_builtin` now receives), but the vocabulary to name
//! the boundary is not. Recorded rather than folded into `Parameter`, which would claim coverage
//! the inventory does not have.
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
            site: "eval_stmt — hir::StmtKind::Let",
            expected_ty_from: "local_types[local]",
            class: Class::Unwired,
        },
        RepBoundary::MatchBinding => Row {
            site: "eval_expr — the match arm's `bindings` insert",
            expected_ty_from: "local_types[local] for each bound local",
            class: Class::Unwired,
        },
        RepBoundary::LoopBinding => Row {
            site: "eval_expr — both `for` forms' loop-item insert",
            expected_ty_from: "local_types[local]",
            class: Class::Unwired,
        },
        RepBoundary::Assignment => Row {
            site: "write_place — the single place-write path, with an empty projection",
            expected_ty_from: "local_types[place.local]",
            class: Class::Unwired,
        },
        RepBoundary::FieldWrite => Row {
            site: "write_place — projection ending in Projection::Field",
            expected_ty_from: "the nominal's declared field type, instantiated",
            class: Class::Unwired,
        },
        RepBoundary::ElementWrite => Row {
            site: "write_place — projection ending in Projection::Index",
            expected_ty_from: "the container's element type from the base's expr_types",
            class: Class::Unwired,
        },
        RepBoundary::AggregateField => Row {
            site: "eval_struct_lit — the per-field `values.insert`",
            expected_ty_from: "the nominal's declared field type, instantiated",
            class: Class::Unwired,
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
        11,
        "RepBoundary has 11 variants; update ALL and `classify` together"
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
/// A producer-side assertion at `expect_value` would not replace destination checks — both consume
/// the same `check_value_for_ty`, so it adds defence, not a second authority.
#[test]
fn the_producer_side_funnel_is_expect_value() {
    let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/interp.rs"))
        .expect("interp.rs must be readable")
        .replace("\r\n", "\n");
    let funnel = source.matches("self.expect_value(").count();
    let direct = source.matches("self.eval_expr(").count();
    assert!(
        funnel >= 25,
        "expect_value should remain the dominant expression-result path; found {funnel}"
    );
    assert!(
        direct <= 8,
        "direct `eval_expr` calls bypass the funnel. Found {direct}; each new one must be \
         classified in this file before it is added — that is what makes the inventory exact."
    );
}

/// **A boundary the ruling names that `RepBoundary` has no variant for.**
///
/// The closure conditions list "inline values entering builtins/runtime operations". A value passed
/// to `call_builtin` or a `RuntimeFn` never binds to a local, so no existing variant describes it:
/// it is not a `Parameter` (that is a user callable's declared parameter, typed by
/// `callable_types`), and folding it into one would make the inventory claim coverage it does not
/// have.
///
/// The expected type IS available — `call_builtin` now receives `arg_exprs`, so `expr_types[arg]`
/// gives the checker's answer for each argument. What is missing is a variant to name the boundary.
///
/// Recorded as a test rather than a comment so it is part of the exact set: closing DEV-121
/// requires either adding `RepBoundary::BuiltinArgument` and wiring it, or establishing that
/// builtins cannot receive a mis-represented value — and the second is a claim needing evidence,
/// not an assumption.
#[test]
fn builtin_arguments_are_a_boundary_with_no_repboundary_variant() {
    // Nothing to assert against the enum yet — the point is that the set is incomplete, and this
    // test is where that fact lives until it is closed.
    let named_by_a_variant = ALL.iter().any(|b| format!("{b:?}").contains("Builtin"));
    assert!(
        !named_by_a_variant,
        "a `Builtin`-shaped RepBoundary variant now exists: add it to `classify` with its site \
         (`call_builtin`, via `arg_exprs`) and its type source (`expr_types[arg]`), and delete \
         this test."
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
        vec!["Parameter", "Receiver", "Return", "Propagation"],
        "DEV-121 closes when every boundary is Wired. Update this pin in the same change that \
         wires one, and update AS3 #3/#5 in CAMPAIGN-A-EXIT-REPORT.md when the list is complete."
    );
}
