# WP-C6.4 platform evidence

This directory holds the Tier-1 qualification records for `WP-C6.4`.

**It is empty of platform records on purpose.** `WP-C6-ENTRY.md` §35 says "no real platform run
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
