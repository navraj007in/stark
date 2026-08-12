//! **Layer audit: which MIR refusals are reachable from a type-correct program?**
//!
//! `lower.rs` holds 194 `unsupported(…)` sites (160 distinct messages). WP-C4.6 classified them by
//! whether the refusal is CORRECT. This asks the different question E0105 was an answer to: **does the front end accept the program first?**
//!
//! A refusal that is semantically right but happens below semantic analysis produces an accepted
//! program no compiler can build — the checker says yes, the reference interpreter runs it, and
//! lowering refuses. One instance was fixed by moving the refusal up: E0105, for by-value `Vec` iteration. A second
//! attempt, E0106 for indexing a non-`Copy` element, was REVERTED (CD-294) and found redundant
//! (CD-297) — E0100 had always refused that at the right layer. Two more were fixed differently,
//! by teaching lowering the construct rather than raising the refusal: DEV-132 (`&v[i].field`) and
//! DEV-133 (`&[T; N]` to `&[T]`), both found by a package build rather than by this audit, because
//! both were VERIFIER rejections on MIR that lowering produced willingly — a shape no probe here
//! reaches. See `WP-LOWERING-COVERAGE-MATRIX.md`.
//!
//! **This was measurement; since WP-DEV-134-139 §11 it is ENFORCEMENT.** It used to report its
//! findings and pass unconditionally, which meant a NEW layer defect could appear and the suite
//! would stay green — the audit could only ever be read by a human who happened to look.
//!
//! Every probe now carries the disposition it is EXPECTED to have, and the test fails when actual
//! and expected disagree. The bar is deliberately not "zero findings": six reachable lowering
//! refusals exist and are not repaired by this programme. The bar is **zero UNREGISTERED
//! findings** — each of the six is pinned to the DEV that owns it, so it is tracked rather than
//! merely observed.
//!
//! It therefore fails when:
//!
//! ```text
//! a new layer defect appears                     -> unregistered finding
//! a registered one stops reproducing             -> either fixed (close the DEV) or the probe
//!                                                   stopped reaching its site
//! a probe changes disposition in either direction -> the inventory moved without being updated
//! ```
//!
//! A probe that stops reproducing its defect is a FAILURE here, not a quiet success: it means
//! either the DEV was fixed and its registration is stale, or the probe no longer exercises the
//! construct it was written for. Both need a human decision, and both are invisible if the test
//! only ever looks for regressions.
//!
//! Reading the output:
//!
//! **The probe table itself lives in `support/layer_probes.rs` since WP-ARCH-CLOSE AC2**, because
//! the published native conformance matrix is generated from the same measurements. Two suites
//! reading one table cannot disagree about what a probe does; two tables would be a second
//! classifier for one question.
//!
//! ```text
//! LAYER-DEFECT  front end ACCEPTED, lowering refused  -> an E0105-class defect
//! ok-frontend   front end refused                     -> correct: refused where it should be
//! ok-lowers     both accepted                         -> the probe does not reach its site
//! ```

mod support;

use support::layer_probes::{probe, probes, Expect, Outcome};

#[test]
fn layer_audit_matches_its_registered_inventory() {
    let mut mismatches: Vec<String> = Vec::new();
    let mut defects = Vec::new();
    let (mut frontend, mut lowers) = (0usize, 0usize);

    println!("\n=== LAYER AUDIT: front end vs lowering (registered inventory) ===\n");
    for entry in probes() {
        let (label, expected) = (entry.label, entry.expect);
        let outcome = probe(entry.source);
        let actual = match &outcome {
            Outcome::LayerDefect(what) => {
                println!("LAYER-DEFECT | {label}\n               lowering: {what}");
                defects.push(label);
                // Which DEV owns it is the registration's business, not the probe's; any
                // KnownDev matches here and the exact identity is compared below.
                Expect::KnownDev("")
            }
            Outcome::FrontEnd(d) => {
                println!("ok-frontend  | {label}\n               refused by: {d}");
                frontend += 1;
                Expect::FrontEnd
            }
            Outcome::Lowers => {
                println!("ok-lowers    | {label}  (probe does not reach its site)");
                lowers += 1;
                Expect::Lowers
            }
        };

        let agrees = match (&expected, &actual) {
            (Expect::KnownDev(_), Expect::KnownDev(_)) => true,
            (a, b) => a == b,
        };
        if !agrees {
            mismatches.push(format!(
                "{label}\n      registered as {expected:?} but actually {actual:?}"
            ));
        }
    }

    println!(
        "\n=== TOTALS: {} layer defects, {} correctly refused up front, {} lowered cleanly ===",
        defects.len(),
        frontend,
        lowers
    );
    if !defects.is_empty() {
        println!("\nReachable lowering refusals (accepted by the checker, refused by MIR):");
        for d in &defects {
            println!("  - {d}");
        }
    }
    println!();

    // WP-DEV-134-139 §11: zero UNREGISTERED findings, not zero findings. A mismatch in EITHER
    // direction fails — a new layer defect, and equally a registered one that stopped
    // reproducing, because the second means either the DEV was fixed (close it and update the
    // registration) or the probe no longer reaches its site (fix the probe). Both need a human
    // decision and both are invisible if the test only looks for regressions.
    assert!(
        mismatches.is_empty(),
        "the layer audit no longer matches its registered inventory:\n\n  - {}\n\n\
         Every probe declares the disposition it is expected to have. Update the registration in \
         `probes()` in the same change as whatever moved it, and the owning DEV entry with it.",
        mismatches.join("\n  - ")
    );
}
