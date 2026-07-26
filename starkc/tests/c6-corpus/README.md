# WP-C6.5 differential corpus

The corpus WP-C6.5 replays through all three engines (HIR oracle, MIR interpreter, native debug
binary) on both Tier-1 targets. Governed by `WP-C6.5.md` and the execution plan's §9; the coverage
it must reach is `C6-CORPUS-COVERAGE-MATRIX.md`.

**This is not the frozen execution corpus.** `starkc/tests/exec_snapshots/` (corpus v1.4.0) is a
separate, inherited artifact with its own lock, preserved rather than absorbed (§3.2). Neither lock
is valid for the other tree, and neither version number moves with the other.

## Layout

```text
c6-corpus/
  README.md              this file
  manifest.toml          one [[case]] per case — the schema is §9.2, documented below
  corpus.lock            per-source hashes + counts (§9.5); generated, never hand-edited
  generator-version.txt  version of generate.py's case-generation behaviour
  generate.py            lock generation today; case generation from C6.5-4
  cases/
    handwritten/         C6.5-3: focused witnesses for categories the matrix names
    generated/           C6.5-4: deterministic template output, regenerable from seed
    retained/            cases retained from real defects (§18.3) — permanent regressions
  metamorphic/           C6.5-6: transformation pairs that must observe identically
  mutations/             C6.5-7: controls proving the corpus can detect a broken compiler
```

A case is a single `.stark` file or a directory holding a complete package/workspace graph.

## What is here today

Six `retained` cases, from DEV-111 and DEV-112 — the entry-contract defects WP-C6.5 found while
building the observation model. They are here rather than only in a test file because §18.3 requires
a retained case to remain a permanent regression, and because a corpus whose machinery has never
locked a real file proves nothing about the machinery.

`c65_entry_exit_contract.rs` reads these same files with `include_str!`, so the corpus source and
the assertions cannot drift apart: editing a case changes both its hash and the test that pins its
expected observation.

Two entry-contract programs are deliberately **not** corpus cases:

- `fn main() -> Int32 { 300 }` — PROC-EXIT-001 requires an `invalid-exit-status` trap, and that trap
  has no `TrapCategory` yet (a CE3 bundled with the native entry work at CD-150). It has no
  replayable observation until that lands, so it stays a boundary probe in the test file.
- `let x: Unit = ()` **before** DEV-112 — the pre-fix rejection is history, not a case.

## Manifest schema

| Field | Meaning |
| --- | --- |
| `case_id` | unique; the manifest is sorted by it, and enumeration order follows it |
| `kind` | `handwritten` / `generated` / `retained` |
| `category` | one of the eight coverage-matrix groups |
| `subcategories` | matrix row IDs this case is evidence for (`K16`, `O13`, …) |
| `sources` | corpus-root-relative, `/`-separated; every source is owned by exactly one case |
| `package_graph` | `single-file` / `package` / `workspace` |
| `language_options` | non-default options the case needs |
| `expected_outcome` | `completion` / `trap` |
| `expected_trap_category` | required when the outcome is `trap` |
| `required_engines` | subset of `hir`, `mir`, `native-debug` |
| `required_targets` | subset of the Tier-1 triples |
| `metamorphic_family` / `metamorphic_group` | both or neither |
| `generator_seed` / `generator_version` / `template_id` | all three required for `generated` |
| `normative_rules` | at least one, for every non-quarantined case |
| `return_probe` | the probe function name, for §8.7 framed-return cases |
| `drop_protocol` | `true` when the case emits §8.8 Drop frames |
| `deviation` | a recorded, non-blocking difference — e.g. an engine that cannot run this case yet |
| `quarantine` | reason class + `CD-###` authority; only §4.4's three allowed classes parse |

`deviation` and `quarantine` are not synonyms. A deviation records something known and accepted
while the case still runs on the engines it lists; a quarantine removes the case from the required
set entirely and is only available for a confirmed non-Core feature, an unavailable external
artifact, or an approved environment condition. **A semantic quarantine does not exist**: engine
disagreement, wrong output, a wrong trap, a wrong Drop order, or a native refusal of an accepted
Core program is a C6 blocker that keeps the gate open (§4.4), and the validator rejects any attempt
to write one.

## Changing the corpus

```bash
python3 tests/c6-corpus/generate.py --lock    # regenerate after any corpus change
python3 tests/c6-corpus/generate.py --check   # what CI asks: is the lock current?
cargo test --test c6_corpus_manifest          # strict manifest + lock integrity
```

Then bump `corpus_version` in `generate.py` and the assertion in `c6_corpus_manifest.rs`, per §9.6:
**patch** for metadata or evidence corrections with no source or expected-observation change,
**minor** for new cases, templates or families, **major** for an incompatible manifest, protocol or
expectation model. The generator version moves independently.

The version assertion in the test is a deliberate speed bump. Regenerating the lock is easy; doing
it *without* a version bump would let a corpus edit quietly redefine the baseline every later claim
is measured against, so the test fails until both are done.
