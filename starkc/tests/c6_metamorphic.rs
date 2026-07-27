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
    first_difference, front_end, front_end_package, run_hir, run_mir, run_native, stage_package,
    Observation,
};

/// §13.1's twelve families, all now built (R-04/R-05, CD-167).
const REQUIRED_FAMILIES: [&str; 12] = [
    "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M11", "M12",
];
/// M08 and M09 used to live here: "both transform a package graph and every corpus case is
/// single-file until §15". That stopped being true when DEV-113/DEV-114 put package cases in the
/// corpus, and DEV-114's fix is what made M09 comparable at all — before it, a diamond graph
/// produced different canonical symbols run to run, so a reorder pair would have disagreed for a
/// reason having nothing to do with the reorder. The list stays, empty, because
/// `the_metamorphic_floor_is_reported_honestly` reads it: a silently deleted structure is how an
/// unmet floor stops being reported.
const FAMILIES_BLOCKED_ON_PACKAGE_GRAPHS: [&str; 0] = [];

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

/// What a member observed, plus the two things a package pair's SUBJECT can be but its observation
/// cannot witness: the canonical symbol set, and the physical root it was staged at.
struct Observed {
    hir: Observation,
    mir: Observation,
    native: Option<Observation>,
    symbols: Vec<String>,
    staged_root: Option<std::path::PathBuf>,
}

fn observe(case: &Case) -> Observed {
    // A package member is STAGED first: resolution writes `stark.lock` into the root package, which
    // would dirty the corpus and break its lock. For M08 the staging is not incidental — it is the
    // transformation, and `relocation_members_are_staged_at_different_roots` proves the two members
    // really do land in different directories.
    let staged = case
        .package_root
        .as_ref()
        .map(|root| stage_package(&case.case_id, &corpus_root(), root));
    let name = format!("{}.stark", case.case_id);
    let (front, program) = match &staged {
        Some(root) => front_end_package(root),
        None => {
            let source = std::fs::read_to_string(corpus_root().join(&case.sources[0]))
                .unwrap_or_else(|e| panic!("{}: {e}", case.sources[0]));
            let front = front_end(&name, &source);
            let program = match starkc::mir::lower::lower_program(
                &front.hir,
                &front.tables,
                front.file.clone(),
            ) {
                Ok(program) => program,
                Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
            };
            (front, program)
        }
    };
    let hir = run_hir(&name, &front);
    let mir = run_mir(&name, &program);
    let native = case
        .required_engines
        .iter()
        .any(|e| e == "native-debug")
        .then(|| run_native(&name, &case.case_id, &program));
    let symbols = program
        .bodies
        .iter()
        .map(|body| body.instance.symbol.clone())
        .collect();
    Observed {
        hir,
        mir,
        native,
        symbols,
        staged_root: staged,
    }
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
        let base = observe(pair.base);
        let other = observe(pair.transformed);
        let (base_hir, base_mir, base_native) = (&base.hir, &base.mir, &base.native);
        let (other_hir, other_mir, other_native) = (&other.hir, &other.mir, &other.native);

        // R-04/R-05. A package pair's SUBJECT can be something its observation cannot witness, so
        // where that is the case the harness pins it directly. Two members can print identical
        // bytes while disagreeing about what their symbols are called — which is exactly what
        // DEV-114 did, and why a bytes-only comparison would have called that pair equal.
        if pair.base.metamorphic_pin_canonical_symbols {
            // A pin over an empty symbol set would pass no matter what the compiler did. Both
            // members must actually have produced bodies before their equality means anything.
            assert!(
                !base.symbols.is_empty() && !other.symbols.is_empty(),
                "{}: a canonical-symbol pin over an empty symbol set is vacuous ({} / {})",
                pair.group,
                base.symbols.len(),
                other.symbols.len()
            );
        }
        if pair.base.metamorphic_pin_canonical_symbols && base.symbols != other.symbols {
            let only_base: Vec<&String> = base
                .symbols
                .iter()
                .filter(|s| !other.symbols.contains(s))
                .collect();
            let only_other: Vec<&String> = other
                .symbols
                .iter()
                .filter(|s| !base.symbols.contains(s))
                .collect();
            failures.push(format!(
                "\n=== {} ===\nCANONICAL SYMBOLS differ across the transformation.\n\
                 TYPE-NOMINAL-001 makes identity `canonical package instance + module path + item \
                 name`, so neither a staging directory nor a dependency declaration ORDER may \
                 appear in it.\nonly in base        {only_base:?}\nonly in transformed {only_other:?}\n\
                 precondition        {}\n",
                pair.group, pair.precondition,
            ));
        }

        // PKG-IDENTITY-001: a package token is "never an absolute checkout path". The observation
        // carries the trap's source file, but a COMPLETING member never reports a file at all, so
        // for relocation pairs the provenance is pinned from the symbol/staging side too.
        if pair.base.metamorphic_pin_logical_provenance {
            // Same vacuity problem: the leak check below only runs for a member that was staged, so
            // an unstaged member would make the whole pin a no-op.
            assert!(
                base.staged_root.is_some() && other.staged_root.is_some(),
                "{}: a provenance pin needs both members staged, or the leak check never runs",
                pair.group
            );
            for (role, observed) in [("base", &base), ("transformed", &other)] {
                if let Some(root) = &observed.staged_root {
                    let leaked: Vec<&String> = observed
                        .symbols
                        .iter()
                        .filter(|s| s.contains(&root.to_string_lossy().to_string()))
                        .collect();
                    assert!(
                        leaked.is_empty(),
                        "{}/{role}: canonical symbols carry the staging path {root:?}: {leaked:?} \
                         — PKG-IDENTITY-001 forbids an absolute checkout path in identity",
                        pair.group
                    );
                }
                if let Observation::Trapped(trap) = &observed.hir {
                    assert!(
                        !trap.source_file.contains(std::path::MAIN_SEPARATOR)
                            || !std::path::Path::new(&trap.source_file).is_absolute(),
                        "{}/{role}: trap provenance {:?} is an absolute path — DEV-113",
                        pair.group,
                        trap.source_file
                    );
                }
            }
        }

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
        check("hir", base_hir, other_hir);
        check("mir", base_mir, other_mir);
        match (base_native, other_native) {
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
    // §13.2's floor, now MET (R-04, CD-167). This test previously asserted the shortfall — 20
    // groups / 40 members — and carried its own instruction: "the floor is now met — raise this
    // test to require 24 groups / 48 members and remove the M08/M09 exemption". Adding M08 and M09
    // made it fail exactly as designed, which is the only reason a shortfall assertion is worth
    // writing: it fails when the shortfall ends, rather than quietly continuing to pass.
    assert_eq!(
        groups.len(),
        24,
        "§13.2 requires 24 groups (twelve families × two)"
    );
    assert_eq!(members, 48, "§13.2 requires 48 members");
    assert!(
        FAMILIES_BLOCKED_ON_PACKAGE_GRAPHS.is_empty(),
        "a family is recorded as blocked, so the floor claim above is not the whole story"
    );
}

/// A pair whose members are byte-identical would agree trivially. The generator asserts this too;
/// asserting it here as well means a hand-added pair cannot skip the check.
#[test]
fn no_pair_compares_a_source_to_itself() {
    let (cases, _) = load();
    for pair in pairs(&cases) {
        // R-04/R-05: the protection is kind-aware, NOT relaxed. An ordinary transformation must
        // change the logical source. A RELOCATION must not — identical files are its precondition,
        // and requiring a textual difference would be the wrong check rather than a stricter one.
        // What must differ instead is the physical root, which
        // `relocation_members_are_staged_at_different_roots` proves.
        match pair.base.metamorphic_kind.as_deref() {
            Some("relocation") => {
                // Identical files are the PRECONDITION, not a defect. What must differ is the
                // physical root — `relocation_members_are_staged_at_different_roots`.
                assert_eq!(
                    read_tree(pair.base),
                    read_tree(pair.transformed),
                    "{}: a relocation pair must keep every logical file identical — a differing \
                     tree is some other transformation wearing its name",
                    pair.group
                );
            }
            Some("dependency-reorder") => {
                // The sources must be identical and the MANIFESTS must differ. Comparing
                // `sources[0]` would have compared two `.stark` files that are equal by design and
                // called the pair fake: manifests are not in `sources` at all.
                let base = read_tree(pair.base);
                let other = read_tree(pair.transformed);
                let manifest = |path: &String| path.ends_with("starkpkg.json");
                let differing: Vec<&String> = base
                    .iter()
                    .filter(|(path, text)| other.get(*path) != Some(*text))
                    .map(|(path, _)| path)
                    .collect();
                assert!(
                    !differing.is_empty(),
                    "{}: the two members are identical, so the reorder reordered nothing",
                    pair.group
                );
                let sources: Vec<&&String> =
                    differing.iter().filter(|path| !manifest(path)).collect();
                assert!(
                    sources.is_empty(),
                    "{}: a dependency reorder must change only manifests; these sources differ too: \
                     {sources:?}",
                    pair.group
                );
            }
            _ => {
                let base = std::fs::read_to_string(corpus_root().join(&pair.base.sources[0]))
                    .expect("base");
                let transformed =
                    std::fs::read_to_string(corpus_root().join(&pair.transformed.sources[0]))
                        .expect("transformed");
                assert_ne!(
                    base, transformed,
                    "{}: the two members are byte-identical, so the pair proves nothing",
                    pair.group
                );
            }
        }
    }
}

/// Every file of a package member, keyed by its path RELATIVE to that member's own tree.
///
/// Walks the directory rather than reading `case.sources`, because `sources` lists only `.stark`
/// files and a dependency reorder changes a `starkpkg.json` — the very file a sources-only read
/// cannot see.
fn read_tree(case: &Case) -> BTreeMap<String, String> {
    fn walk(dir: &std::path::Path, base: &std::path::Path, out: &mut BTreeMap<String, String>) {
        for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{dir:?}: {e}")) {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                walk(&path, base, out);
            } else if path.file_name().is_some_and(|n| n != "stark.lock") {
                // `stark.lock` is written by resolution into whatever root it ran in, so it is a
                // product of staging rather than part of the authored tree.
                let relative = path
                    .strip_prefix(base)
                    .expect("under the member root")
                    .to_string_lossy()
                    .replace('\\', "/");
                out.insert(
                    relative,
                    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}")),
                );
            }
        }
    }
    let root = corpus_root().join(format!("metamorphic/pkg/{}", case.case_id));
    let mut out = BTreeMap::new();
    walk(&root, &root, &mut out);
    out
}

/// **R-04/R-05.** A relocation pair's members are staged at two genuinely different physical roots.
///
/// This is M08's identity-transform guard, and it is the whole content of the family. Two
/// references to one staged workspace would agree trivially while being labelled a relocation —
/// the relocation-shaped version of the fake pairs the source-level guard already catches. The
/// members' logical files are identical *on purpose*, so nothing else in this suite would notice.
#[test]
fn relocation_members_are_staged_at_different_roots() {
    let (cases, _) = load();
    let mut checked = 0;
    for pair in pairs(&cases) {
        if pair.base.metamorphic_kind.as_deref() != Some("relocation") {
            continue;
        }
        let base_root = pair
            .base
            .package_root
            .as_ref()
            .map(|root| stage_package(&pair.base.case_id, &corpus_root(), root))
            .unwrap_or_else(|| panic!("{}: a relocation member needs a package_root", pair.group));
        let other_root = pair
            .transformed
            .package_root
            .as_ref()
            .map(|root| stage_package(&pair.transformed.case_id, &corpus_root(), root))
            .unwrap_or_else(|| panic!("{}: a relocation member needs a package_root", pair.group));
        assert_ne!(
            base_root, other_root,
            "{}: both members staged to the SAME root, so nothing was relocated",
            pair.group
        );
        // Not merely different strings: different directories. A pair that differed only in a
        // trailing component of one shared parent would still be one workspace.
        assert!(
            !base_root.starts_with(&other_root) && !other_root.starts_with(&base_root),
            "{}: one staged root contains the other ({base_root:?} / {other_root:?}) — that is one \
             workspace, not two",
            pair.group
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no relocation pairs were checked — M08 is declared in REQUIRED_FAMILIES, so a corpus \
         with none would make this control vacuous"
    );
}
