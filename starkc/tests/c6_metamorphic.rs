//! WP-C6.5 §13 — metamorphic pairs: two sources the specification makes equivalent must be
//! observed identically.
//!
//! What a pair adds over a case: a case shows an engine produces the right answer for one source
//! form. A pair shows the answer does not depend on the *form* — that renaming a local, inserting a
//! scope, reordering non-overlapping arms, spelling a generic argument explicitly, or extracting a
//! helper changes nothing observable. Compilers fail this in ways single cases cannot detect,
//! because both members are individually plausible.
//!
//! §13.4's comparison is per engine and then across engines:
//!
//! ```text
//! HIR(base) == HIR(transformed)      MIR(base) == MIR(transformed)      native(base) == native(transformed)
//! ```
//!
//! then three-engine agreement for both members — which the §12 replay already establishes for every
//! corpus case, so this suite adds the pair dimension rather than repeating it.

mod support;

use std::collections::BTreeMap;

use support::corpus::{corpus_root, load, Case};
use support::differential::{
    first_difference, front_end, run_hir, run_mir, run_native, Observation,
};

/// §13.1's twelve families. M08/M09 are absent by construction — both transform a package graph and
/// every corpus case is single-file until §15 — so they are listed here to keep the gap visible
/// rather than letting a ten-family corpus read as complete.
const REQUIRED_FAMILIES: [&str; 12] = [
    "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M11", "M12",
];
const FAMILIES_BLOCKED_ON_PACKAGE_GRAPHS: [&str; 2] = ["M08", "M09"];

struct Pair<'a> {
    group: String,
    base: &'a Case,
    transformed: &'a Case,
    precondition: String,
}

fn pairs(cases: &[Case]) -> Vec<Pair<'_>> {
    let mut by_group: BTreeMap<String, Vec<&Case>> = BTreeMap::new();
    for case in cases.iter().filter(|c| c.metamorphic_group.is_some()) {
        by_group
            .entry(case.metamorphic_group.clone().unwrap())
            .or_default()
            .push(case);
    }
    let mut out = Vec::new();
    for (group, members) in by_group {
        let base = members
            .iter()
            .find(|c| c.metamorphic_role.as_deref() == Some("base"))
            .unwrap_or_else(|| panic!("{group}: no base member"));
        let transformed = members
            .iter()
            .find(|c| c.metamorphic_role.as_deref() == Some("transformed"))
            .unwrap_or_else(|| panic!("{group}: no transformed member"));
        assert_eq!(
            members.len(),
            2,
            "{group}: a metamorphic group is exactly one base and one transformed member, found {}",
            members.len()
        );
        out.push(Pair {
            precondition: base.metamorphic_precondition.clone().unwrap_or_default(),
            group,
            base,
            transformed,
        });
    }
    out
}

fn observe(case: &Case) -> (Observation, Observation, Option<Observation>) {
    let source = std::fs::read_to_string(corpus_root().join(&case.sources[0]))
        .unwrap_or_else(|e| panic!("{}: {e}", case.sources[0]));
    let name = format!("{}.stark", case.case_id);
    let front = front_end(&name, &source);
    let program =
        match starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone()) {
            Ok(program) => program,
            Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
        };
    let hir = run_hir(&name, &front);
    let mir = run_mir(&name, &program);
    let native = case
        .required_engines
        .iter()
        .any(|e| e == "native-debug")
        .then(|| run_native(&name, &case.case_id, &program));
    (hir, mir, native)
}

/// §13.4. Each engine must preserve each pair, and the failure names the engine, the field and the
/// precondition — because a diverging pair is either a compiler defect or an invalid transformation,
/// and §13.7 forbids deciding which without normative analysis. The precondition is what that
/// analysis starts from.
#[test]
fn every_metamorphic_pair_is_preserved_by_every_engine() {
    if !support::differential::rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let (cases, _) = load();
    let groups = pairs(&cases);
    assert!(!groups.is_empty(), "no metamorphic groups in the corpus");

    let mut failures = Vec::new();
    for pair in &groups {
        let (base_hir, base_mir, base_native) = observe(pair.base);
        let (other_hir, other_mir, other_native) = observe(pair.transformed);
        let mut check = |engine: &str, left: &Observation, right: &Observation| {
            if let Some(field) = first_difference(left, right) {
                failures.push(format!(
                    "\n=== {group} ===\nengine        {engine}\nfirst differing field  {field}\n\
                     base          {}\ntransformed   {}\nprecondition  {}\n\
                     §13.7         retain both sources, open a defect, and do NOT rewrite the pair \
                     to make it pass\n--- base ---\n{left:#?}\n--- transformed ---\n{right:#?}\n",
                    pair.base.case_id,
                    pair.transformed.case_id,
                    pair.precondition,
                    group = pair.group,
                ));
            }
        };
        check("hir", &base_hir, &other_hir);
        check("mir", &base_mir, &other_mir);
        match (&base_native, &other_native) {
            (Some(left), Some(right)) => check("native-debug", left, right),
            (None, None) => {}
            _ => failures.push(format!(
                "{}: one member requires native and the other does not — a pair must be compared on \
                 the same engines",
                pair.group
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} metamorphic pairs diverged:\n{}",
        failures.len(),
        groups.len(),
        failures.join("")
    );
}

/// §13.2's acceptance floor, and the two families that cannot meet it yet. Stated as a test so the
/// shortfall is a visible, failing-if-forgotten fact rather than a paragraph someone has to read.
#[test]
fn the_metamorphic_floor_is_reported_honestly() {
    let (cases, _) = load();
    let groups = pairs(&cases);
    let mut per_family: BTreeMap<&str, usize> = BTreeMap::new();
    for pair in &groups {
        let family = pair.base.metamorphic_family.as_deref().unwrap_or("?");
        *per_family.entry(family).or_default() += 1;
    }

    // Every family that is present must have at least two independent groups (§13.2).
    for (family, count) in &per_family {
        assert!(
            *count >= 2,
            "{family} has {count} group(s); §13.2 requires at least two independent groups"
        );
    }
    // Every family that is absent must be one of the two with a recorded package-graph reason.
    for family in REQUIRED_FAMILIES {
        if !per_family.contains_key(family) {
            assert!(
                FAMILIES_BLOCKED_ON_PACKAGE_GRAPHS.contains(&family),
                "{family} has no groups and no recorded reason — §13.1 requires all twelve families"
            );
        }
    }
    let members = groups.len() * 2;
    // The floor itself is NOT met, and this is the assertion that says so out loud: 24 groups / 48
    // members with all twelve families. Ten families at two groups each is 20/40. When M08/M09
    // become buildable this expectation moves up rather than the shortfall being forgotten.
    assert_eq!(groups.len(), 20, "unexpected group count");
    assert_eq!(members, 40, "unexpected member count");
    assert!(
        groups.len() < 24,
        "the floor is now met — raise this test to require 24 groups / 48 members and remove the \
         M08/M09 exemption"
    );
}

/// A pair whose members are byte-identical would agree trivially. The generator asserts this too;
/// asserting it here as well means a hand-added pair cannot skip the check.
#[test]
fn no_pair_compares_a_source_to_itself() {
    let (cases, _) = load();
    for pair in pairs(&cases) {
        let base =
            std::fs::read_to_string(corpus_root().join(&pair.base.sources[0])).expect("base");
        let transformed = std::fs::read_to_string(corpus_root().join(&pair.transformed.sources[0]))
            .expect("transformed");
        assert_ne!(
            base, transformed,
            "{}: the two members are byte-identical, so the pair proves nothing",
            pair.group
        );
    }
}
