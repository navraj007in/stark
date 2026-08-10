# stark-content-id evidence

Baseline SHA: `49a43a470bea8de1352f3b70408e2ff046542ebb`

Baseline branch: `codex/stark-sha256-package`

Worktree clean: false (`docs/STARK-Language-Overview.docx`, `stark-http-get/` were pre-existing untracked files)

Final SHA: not committed in this working tree

Files added:

- `packages/stark-content-id/`
- `packages/stark-content-id-consumer/`

Files modified:

- `starkc/scripts/qualify-first-party-packages.py`

Package test command:
- `../../starkc/target/release/stark test` from `packages/stark-content-id`

Test count: 27 passed, 0 failed, 0 ignored

Results: PASS

Format/check result:

- `../../starkc/target/release/stark fmt --check` from `packages/stark-content-id`: PASS
- `../../starkc/target/release/stark check` from `packages/stark-content-id`: PASS
- `../../starkc/target/release/stark check` from `packages/stark-content-id-consumer`: PASS

Consumer check: PASS

Consumer run:

- `../../starkc/target/release/stark run` from `packages/stark-content-id-consumer`: PASS
- stdout: `content-id:ok`

HIR result: package tests PASS

MIR result: native build MIR verification PASS in debug and release builds

Native debug result:

- `../../starkc/target/release/stark build --no-build-cache` from `packages/stark-content-id-consumer`: PASS
- `./target/stark/debug/stark-content-id-consumer`: PASS
- stdout: `content-id:ok`

Native release result:

- `../../starkc/target/release/stark build --release --no-build-cache` from `packages/stark-content-id-consumer`: PASS
- `./target/stark/release/stark-content-id-consumer`: PASS
- stdout: `content-id:ok`

First-party qualification result:

- Full command: `python3 starkc/scripts/qualify-first-party-packages.py --stark starkc/target/release/stark --repo-root /Users/nexper/Documents/GitHub/stark`
- `stark-content-id` segment: PASS
- Observed: check PASS, tests PASS, docs PASS, surface check PASS with 5 public callables all called, fmt PASS, consumer check PASS, consumer run PASS with `content-id:ok`, native debug consumer build/run PASS with `content-id:ok`.
- Full run later stopped at pre-existing `stark-net` provider resolution: `UnknownFunction { capability: "network-client", function: "stark_tcp_stream_connect_timeout", provider: "stark-std-net" }`.

CI workflow: not run

CI run ID: not run

CI conclusion: not run

Compiler changes: none

Provider changes: none

Host capabilities: none

Residuals:

- The packet's requested `git switch develop`/`git pull --ff-only` preflight was not used for this
  implementation because `stark-digest` is required and is present on the current stacked branch
  baseline, not on the earlier observed baseline.
- Direct `Digest` return-type annotations in this package's test module collided with the public
  `digest` function name in current name resolution. Tests avoid that annotation while preserving
  the required public API, including `digest(id) -> &Digest`.

New deviations: none
