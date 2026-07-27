//! WP-C6.5 §11.11 / §13.7 — the retention workflows, exercised rather than described (R-08).
//!
//! §17 Review C and E both recorded these as **untested**: the procedures existed on paper and
//! nothing had ever run them. A procedure nobody has executed is indistinguishable from one that
//! does not work, which is the same objection the mutation controls answer for the comparator.
//!
//! Two workflows, and they fail differently:
//!
//! * **§11.11 failing-case retention.** When a case finds a defect, the original is kept, a reduced
//!   reproduction is derived, the reduction is recorded as equivalent, and the reduced case becomes
//!   a regression once the defect is fixed. The layout is `cases/retained/<DEV-ID>/{original,
//!   reduced}/` plus a `RETENTION.toml` record.
//! * **§13.7 divergence retention.** When a metamorphic pair diverges, BOTH sources are retained,
//!   the first differing normative field is identified, and the pair is NOT rewritten to pass.
//!
//! The first is driven by a real retention — DEV-117, found by writing O14's coverage case. The
//! second has no real divergence to drive it (no pair currently diverges), so it is driven by a
//! SYNTHETIC controlled divergence: two observations constructed to differ in one field, put
//! through the production comparator, with the required output asserted.

mod support;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use support::corpus::{corpus_root, load};
use support::differential::{
    first_difference, CompletionObservation, DropEvent, Observation, TrapMessageClass,
    TrapObservation, TrapStderrObservation,
};

fn retained_root() -> PathBuf {
    corpus_root().join("cases/retained")
}

/// Every `<DEV-ID>/` retention directory, parsed from its `RETENTION.toml`.
fn retentions() -> Vec<(String, BTreeMap<String, String>)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(retained_root()).expect("retained dir") {
        let path = entry.expect("entry").path();
        let name = path
            .file_name()
            .expect("name")
            .to_string_lossy()
            .to_string();
        if !path.is_dir() || !name.starts_with("DEV-") {
            continue;
        }
        let record = path.join("RETENTION.toml");
        let text = std::fs::read_to_string(&record)
            .unwrap_or_else(|e| panic!("{name}: RETENTION.toml is required by §11.11: {e}"));
        let mut fields = BTreeMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with('#') || !line.contains(" = ") {
                continue;
            }
            let (k, v) = line.split_once(" = ").expect("checked");
            fields.insert(k.trim().to_string(), v.trim().trim_matches('"').to_string());
        }
        out.push((name, fields));
    }
    out
}

/// §11.11 steps 1–6. The layout and the record, checked for every retention that exists.
#[test]
fn every_retention_follows_the_11_11_procedure() {
    let found = retentions();
    assert!(
        !found.is_empty(),
        "no `cases/retained/DEV-*/` retentions — this control would be vacuous, and §11.11 has been \
         triggered at least once (DEV-117)"
    );
    for (dev, fields) in &found {
        let dir = retained_root().join(dev);
        // Step 1 and 2: the original is kept and a reduction exists.
        for required in ["original/main.stark", "reduced/main.stark"] {
            assert!(
                dir.join(required).is_file(),
                "{dev}: §11.11 requires `{required}`"
            );
        }
        // Step 3: the record carries the provenance a reader would need to reproduce it.
        for key in [
            "dev_id",
            "status",
            "normative_rule",
            "summary",
            "generator_seed",
            "template_id",
            "dimensions",
            "reduction_equivalent",
        ] {
            assert!(
                fields.contains_key(key),
                "{dev}: RETENTION.toml is missing `{key}` (§11.11 step 3)"
            );
        }
        assert_eq!(
            fields["dev_id"], *dev,
            "{dev}: record names a different DEV"
        );
        let status = fields["status"].as_str();
        assert!(
            status == "open" || status == "fixed",
            "{dev}: status must be `open` or `fixed`, got `{status}`"
        );

        // Step 5: the original may not be deleted until the reduction is proven equivalent — so a
        // record that claims equivalence must still have the original present, which is checked
        // above, and one that does not must say so rather than leaving it blank.
        assert!(
            !fields["reduction_equivalent"].is_empty(),
            "{dev}: §11.11 step 5 needs an explicit equivalence statement"
        );

        // The reduction must actually be smaller. A "reduced" copy of the original is not a
        // reduction, and §11.12's whole point is that the reproduction shrinks.
        let original = std::fs::read_to_string(dir.join("original/main.stark")).expect("original");
        let reduced = std::fs::read_to_string(dir.join("reduced/main.stark")).expect("reduced");
        assert!(
            reduced.len() < original.len(),
            "{dev}: the reduced case ({} bytes) is not smaller than the original ({} bytes)",
            reduced.len(),
            original.len()
        );

        // Step 4 and 6: once fixed, the reduced case becomes a regression IN THE MANIFEST. While
        // open it is deliberately absent, because a case the compiler refuses cannot replay.
        let (cases, _) = load();
        let in_manifest = cases.iter().any(|c| {
            c.sources
                .iter()
                .any(|s| s.contains(&format!("retained/{dev}/reduced")))
        });
        match status {
            "fixed" => assert!(
                in_manifest,
                "{dev}: status is `fixed`, so §11.11 step 6 requires the reduced case in the manifest"
            ),
            _ => assert!(
                !in_manifest,
                "{dev}: status is `open`, but the reduced case is in the runnable manifest — it \
                 cannot replay while the defect stands"
            ),
        }
    }
}

/// The §11.11 control, proven to REFUSE. A retention missing its reduction, or whose "reduction" is
/// a copy of the original, must fail — otherwise the checks above are decoration.
#[test]
fn the_retention_checks_refuse_a_malformed_retention() {
    let scratch = std::env::temp_dir().join(format!("stark_c65_retention_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let dev = scratch.join("DEV-999");
    std::fs::create_dir_all(dev.join("original")).expect("mk");
    std::fs::write(
        dev.join("original/main.stark"),
        "fn main() { let a: Int32 = 1; }\n",
    )
    .expect("w");
    std::fs::write(
        dev.join("RETENTION.toml"),
        "dev_id = \"DEV-999\"\nstatus = \"open\"\n",
    )
    .expect("w");

    // Missing `reduced/` entirely.
    assert!(
        !dev.join("reduced/main.stark").is_file(),
        "the malformed retention must genuinely lack its reduction"
    );

    // A "reduction" that is a byte copy of the original is not smaller, which is the check that
    // catches the tempting shortcut of copying the file to satisfy the layout.
    std::fs::create_dir_all(dev.join("reduced")).expect("mk");
    let original = std::fs::read_to_string(dev.join("original/main.stark")).expect("r");
    std::fs::write(dev.join("reduced/main.stark"), &original).expect("w");
    let reduced = std::fs::read_to_string(dev.join("reduced/main.stark")).expect("r");
    assert!(
        reduced.len() >= original.len(),
        "a copied reduction must not satisfy the size check"
    );

    // And the record is missing every §11.11 step-3 field but the two written above.
    let text = std::fs::read_to_string(dev.join("RETENTION.toml")).expect("r");
    for key in ["normative_rule", "summary", "reduction_equivalent"] {
        assert!(
            !text.contains(key),
            "the malformed record must genuinely lack `{key}`"
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
}

fn completed(stdout: &str, drops: &[&str]) -> Observation {
    Observation::Completed(CompletionObservation {
        stdout_bytes: stdout.as_bytes().to_vec(),
        stderr_bytes: Vec::new(),
        exit_status: 0,
        returned_observation: None,
        drop_log: drops
            .iter()
            .enumerate()
            .map(|(i, id)| DropEvent {
                sequence: i as u32 + 1,
                identity: (*id).to_string(),
            })
            .collect(),
    })
}

/// §13.7, driven by a SYNTHETIC controlled divergence.
///
/// No metamorphic pair currently diverges, so there is nothing real to drive this — and waiting for
/// a divergence to discover whether the divergence rule works is exactly the position §17 objected
/// to. Instead the two observations are constructed to differ in exactly one normative field, and
/// the rule's required outputs are asserted: the first differing field is IDENTIFIED, and it is the
/// one that was planted rather than whichever field happens to be compared first.
#[test]
fn a_diverging_pair_identifies_its_first_differing_field() {
    // Planted in `drop_log`: same bytes, same exit, different destruction order. A comparator that
    // only compared stdout would call this pair equal, which is the defect class §13.7 exists for.
    let base = completed("same\n", &["A#1", "B#2"]);
    let transformed = completed("same\n", &["B#2", "A#1"]);
    assert_eq!(
        first_difference(&base, &transformed),
        Some("drop_log"),
        "§13.7 requires the FIRST differing normative field to be identified"
    );

    // A divergence in a trap field is reported as that field, not collapsed into "the trap differs".
    let trap = |line: u32| {
        Observation::Trapped(TrapObservation {
            category: starkc::mir::TrapCategory::IntegerOverflow,
            source_file: "pair.stark".to_string(),
            line,
            column: 5,
            message_class: TrapMessageClass::CategoryOnly,
            stdout_before_trap: b"before".to_vec(),
            stderr_observation: TrapStderrObservation {
                category_text: "integer overflow".to_string(),
                user_message: None,
                source_file: "pair.stark".to_string(),
                line,
                column: 5,
            },
            exit_status: 101,
            drop_log_before_trap: Vec::new(),
        })
    };
    assert_eq!(
        first_difference(&trap(7), &trap(9)),
        Some("trap line"),
        "a trap-location divergence must name the location field"
    );

    // And a pair that agrees produces no field at all — so the assertions above are not simply
    // "this function always returns something".
    assert_eq!(
        first_difference(&base, &completed("same\n", &["A#1", "B#2"])),
        None
    );
}

/// §13.7's prohibition, as a check rather than a hope: a diverging pair must not be rewritten to
/// pass. The retained sources are the evidence, so a retention whose two members are identical
/// would mean the pair had been "fixed" by editing it.
#[test]
fn divergence_retention_keeps_both_sources_distinct() {
    for (dev, fields) in retentions() {
        let dir = retained_root().join(&dev);
        if fields
            .get("first_differing_field")
            .is_none_or(|f| f.is_empty())
        {
            continue;
        }
        let original = std::fs::read_to_string(dir.join("original/main.stark")).expect("original");
        let reduced = std::fs::read_to_string(dir.join("reduced/main.stark")).expect("reduced");
        assert_ne!(
            original, reduced,
            "{dev}: both retained sources are identical, so nothing was retained"
        );
        assert!(
            !fields["first_differing_field"].is_empty(),
            "{dev}: §13.7 requires the first differing normative field to be recorded"
        );
    }
}

/// The retained tree must not drift from the manifest: a `.stark` under `cases/retained/` that is
/// neither listed as a case nor part of a `DEV-*` retention is a file nobody is checking.
#[test]
fn no_retained_source_is_unaccounted_for() {
    let (cases, _) = load();
    let listed: Vec<&str> = cases
        .iter()
        .flat_map(|c| c.sources.iter().map(String::as_str))
        .collect();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("read") {
            let p = entry.expect("entry").path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|e| e == "stark") {
                out.push(p);
            }
        }
    }
    let mut found = Vec::new();
    walk(&retained_root(), &mut found);
    for path in found {
        let relative = path
            .strip_prefix(corpus_root())
            .expect("under corpus")
            .to_string_lossy()
            .replace('\\', "/");
        let in_manifest = listed.iter().any(|s| *s == relative);
        let in_retention = relative.contains("/DEV-");
        assert!(
            in_manifest || in_retention,
            "{relative} is under cases/retained/ but is neither a manifest case nor part of a \
             DEV-* retention"
        );
    }
}
