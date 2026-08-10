# stark-http-auth evidence

Baseline SHA: `49a43a470bea8de1352f3b70408e2ff046542ebb`

Baseline branch: `codex/stark-sha256-package`

Worktree clean: false (`stark-content-id` work, `docs/STARK-Language-Overview.docx`, and `stark-http-get/` were already present)

Final SHA: not committed in this working tree

Package tests:

- command: `../../starkc/target/release/stark test` from `packages/stark-http-auth`
- count: 19 passed, 0 failed, 0 ignored
- result: PASS

Consumer:

- check: PASS
- run: `../../starkc/target/release/stark run` from `packages/stark-http-auth-consumer`
- result: PASS, stdout `http-auth:ok`

Engine evidence:

- HIR: package tests PASS
- MIR: native build MIR verification PASS in debug and release builds
- native debug: build PASS, `./target/stark/debug/stark-http-auth-consumer` PASS with `http-auth:ok`
- native release: build PASS, `./target/stark/release/stark-http-auth-consumer` PASS with `http-auth:ok`

First-party qualification:

- command: `python3 starkc/scripts/qualify-first-party-packages.py --stark starkc/target/release/stark --repo-root /Users/nexper/Documents/GitHub/stark`
- result: `stark-http-auth` segment PASS: check PASS, tests PASS, docs PASS, surface check PASS with 4 public callables all called, fmt PASS, consumer check PASS, consumer run PASS with `http-auth:ok`, native debug consumer build/run PASS with `http-auth:ok`
- full run later stopped at pre-existing `stark-net` provider resolution: `UnknownFunction { capability: "network-client", function: "stark_tcp_stream_connect_timeout", provider: "stark-std-net" }`

CI:

- run ID: not run
- conclusion: not run

Compiler changes: none

Provider changes: none

Host capabilities: none

Dependencies: `stark-base64 0.1.0`

New deviations: none

Residuals:

- The packet's requested `git switch develop`/`git pull --ff-only` preflight was not used because
  this checkout is currently carrying stacked package work and untracked files. The implementation
  uses the current branch baseline and records it above.
- `basic(username, password)` originally kept a `String` return type and trapped when `username`
  contained `:`. That was recorded here as a residual and has since been repaired: it returns
  `Result<String, AuthError>` and rejects a colon in the username, and any control byte in either
  argument, with `InvalidBasicCredentials`. A trap aborts under Core v1 semantics, so on
  attacker-influenced credentials the old form was a denial of service; and construction now
  rejects exactly what `parse` rejects, which it previously did not.
