# WP-C2.6 — Core Completeness Inventory and Specification Authority

Gate: C2 (Reference Execution Semantics and Compiler Service Foundation).

Status: **Completed 2026-07-18.**

## Scope delivered

- a granular inventory of independently observable and independently rejectable Core v1
  questions across every required semantic-freeze domain;
- stable rule IDs, completeness and behavior classification, one normative home, positive and
  negative evidence state, compatibility cost, owner, decision state, and deviation links;
- one authoritative specification map separating type legality, abstract execution, static
  analysis, safety guarantees, library contracts, package/target contracts, and future
  boundaries;
- a checked 59-entry transition map from every legacy broad conformance rule to granular IDs,
  without copying broad `implemented` status onto the new questions;
- a complete open-question register assigning semantic decisions to C2.7–C2.10 and evidence/
  diagnostic alignment to C2.11;
- removal of stale maturity, phase, success-checklist, compiler-algorithm, physical-layout, and
  performance-planning prose from the normative generated source surface; future sketches that
  remain are explicitly informative and non-normative;
- explicit ownership for DEV-009, DEV-017, DEV-018, DEV-022, DEV-023, and DEV-024.

## Decisions

Only the governance authority split in `CORE-V1-COMPLETENESS.md` is approved by this work
package. It decides where later approved language rules must live; it does not approve those
rules. `CORE-Q-002` through `CORE-Q-024` remain pending, and current compiler behavior is not
promoted merely because it exists.

## Evidence

The inventory was cross-checked against normative chapters 00–07, the reference-execution
contract, all 59 legacy coverage entries, the generated conformance report, the execution plan,
the preflight audit, and the deviation ledger. Validation checks that every legacy rule occurs
exactly once in the split map, every mapped granular ID exists in the inventory, every inventory
ID is unique, and every question/deviation reference is resolvable.

Generated Markdown, HTML, and PDF specifications are regenerated from the cleaned normative
sources. C2.6 changes no compiler semantics and adds no C2.7+ implementation.

## Scope control

The adjacent one-line Rust Clippy compatibility correction in `starkc/src/lsp/protocol.rs`
predates C2.6 and is not semantic-freeze work. It is preserved and validated separately.
