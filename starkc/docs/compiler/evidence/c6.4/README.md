# WP-C6.4 platform evidence

This directory holds the Tier-1 qualification records for `WP-C6.4`.

**There are no platform records here right now, and that is deliberate.** CI run 30191381334 at
`61008f6` produced two passing, agreeing ones; the owner's second review round then strengthened
the comparator, which now refuses them for missing `target_pointer_width`,
`layout_contract_version`, `compiler_layout_revision` and `required_steps`. A record the gate
rejects cannot support a claim, so they were deleted rather than carried forward. The replacements
come from the corrected commit's run.

**No record here is ever written by hand.** `WP-C6-ENTRY.md` §35 says "no real platform run
means no platform claim", and the execution plan's §2 forbids `CLOSED` "while a required
real-platform or generated-corpus run is absent". A hand-written or locally-simulated record here
would defeat the only thing these files are for. The records arrive from CI, produced by the
qualification harness on the runner that ran them:

| File | Produced by | Contains |
| --- | --- | --- |
| `macos-arm64.json` / `.md` | `c64-qualification (macos-arm64)` | the Tier-1 record for `aarch64-apple-darwin` |
| `linux-x64.json` / `.md` | `c64-qualification (linux-x64)` | the Tier-1 record for `x86_64-unknown-linux-gnu` |
| `qualification-summary.md` | `c64-tier1-agreement` | the comparison of the two records |
| `windows-x64-gap-report.md` | maintained by hand, from the `c64-windows-gap` probe | §36's Tier-2 disposition |

## How a record is produced

```bash
cd starkc
python3 scripts/run-c64-qualification.py \
  --expected-target <triple> \
  --commit "$(git rev-parse HEAD)" \
  --output-dir docs/compiler/evidence/c6.4
```

The harness refuses to describe a run it did not make: `--expected-target` is compared with what
`rustc -vV` reports, `--commit` with `git rev-parse HEAD`, a dirty tracked worktree is a deviation,
and a self-skipped test (`SKIP:` — printed when no rustc is present) fails the command that
contained it rather than counting as a pass.

## How the records get here

They are produced on the runner and downloaded, never regenerated locally — a locally regenerated
"copy" of a CI record is a different run wearing the same filename.

```bash
gh run download <run-id> --pattern 'c64-evidence-*' --dir /tmp/c64
gh run download <run-id> --name c64-qualification-summary --dir /tmp/c64
cp /tmp/c64/{macos-arm64,linux-x64}.{json,md} /tmp/c64/qualification-summary.md \
   starkc/docs/compiler/evidence/c6.4/
```

Check before committing them: both records must name the same `commit_sha`, that commit must be the
one being claimed, `dirty_worktree` must be `false` and `quick_mode` must be absent or `false` in
both, and `unclassified_ignores` must be empty. The comparison job asserts all of this — the manual
check is for the case where someone downloads from the wrong run.

## How the two records become one claim

```bash
python3 starkc/scripts/compare-c64-evidence.py \
  macos-arm64.json linux-x64.json --out qualification-summary.md
```

Two green jobs are not agreement. The comparison requires the two records to be for the two
*different* Tier-1 targets, to name the same commit and the same compiler/MIR/runtime/backend/
layout versions, and to make the same per-command observations. Anything else exits non-zero.

## What is deliberately not compared

Host triple, OS, architecture, rustc/Cargo version, runner identity, timestamps, durations. These
are supposed to differ; comparing them would fail every honest run. They are recorded so a reader
can see what the two platforms were.

## Generated corpus

`generated_corpus_status` reads `BLOCKED-BY-C6.5` in every record. The deterministic generated
corpus belongs to the `WP-C6.5` chapter of `WP-C6-ENTRY.md` (§§38–45) and does not exist yet; the frozen
execution corpus (`tests/exec_snapshots/corpus.lock`) is a different artifact and is recorded under
its own command. See `WP-C6.4.md` §1.2 for the sequencing disposition.
