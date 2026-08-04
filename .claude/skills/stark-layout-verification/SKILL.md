---
name: stark-layout-verification
description: Use when a change touches file layout, paths, the installer, provider crates, runtime/provider discovery, or anything the compiler resolves by relative path — moving directories, editing Cargo.toml path dependencies, changing install scripts or build-release.py, or altering discover_runtime/provider_repo_root. Encodes the verification sequence that three separate platform-divergent defects got past, and forbids the "verified" claims that hid them.
tools: Read, Grep, Glob, Bash, Edit
---

# STARK layout and path verification

**The failure this exists to prevent: a path change that works on the developer's machine, passes
the tests they thought to run, and cannot build at all on another platform.** It has happened three
times.

| when | what | why it got through |
| --- | --- | --- |
| CD-377 | Installed toolchain could not build on macOS or Windows | Linux resolves `current_exe()` through `/proc/self/exe`, so the flat lookup matched there and only there. The author's "verified end to end" never set `STARK_REQUIRE_INSTALLED_RUNTIME=1`, so every build had silently fallen back to the source checkout. |
| DEV-163 | Socket read deadline reported as a connection failure on Unix only | Same shape: one platform's behaviour taken for the contract. |
| packages/ move | Four `include_str!` paths broke | Cargo.toml path dependencies were audited; `include_str!` was not. It resolves relative to the **source file**, and the provider crates' own test targets — where those macros live — are never compiled by the `starkc` suite. |

The pattern in all three: **the verification that was run did not exercise the thing that changed.**

---

## Before you start

Write down, in one line each, the answers to these. If you cannot, you do not yet understand the
change well enough to verify it.

1. **What resolves this path, and relative to what?** A Cargo.toml path dependency resolves from
   the manifest's directory. `include_str!`/`include_bytes!`/`#[path]` resolve from the **source
   file**. `current_exe()`-based discovery resolves from the installed binary, whose location is
   platform-dependent. A working-directory-relative path in a script resolves from wherever CI
   invoked it.
2. **Does the same file get used at two different depths?** Provider crates are the standing
   example: the *same* `Cargo.toml` is read in a checkout and in an installed tree. Its relative
   paths must be correct in both, which means the two trees must have the same shape. If you change
   one depth you must change the other.
3. **Which platform would notice first, and why not the others?**

## The invariants that keep breaking

- **One `stark-provider-abi`, reachable from two directions.** The runtime says `../`; a provider
  crate says `../../../`. Both must land on the same directory. Cargo refuses a lockfile naming one
  package at two paths, **and a symlink does not help** — Cargo does not canonicalise symlinked
  path dependencies.
- **The provider root is discovered, not configured.** `provider_repo_root` walks up from the
  package looking for `stark-time/native/Cargo.toml`, then falls back to roots beside the runtime.
  So `crate_path` in `starkc/providers/*.json` is relative to *that discovered root*, not to the
  repository. Changing both at once double-counts.
- **Older installations must keep working.** Discovery keeps candidates for previous layouts
  because an installation made before a move carries crates whose relative paths were correct at
  *that* depth. Adding a new candidate is right; replacing the old ones strands users.
- **Discovery is environment-free by design** (Packet 5). Do not "fix" a path problem by consulting
  a new environment variable.

## The verification sequence — all of it, in order

Do not skip a step because the previous one passed. Each covers something the others cannot see.

```bash
# 1. Compiler and its tests still build.
cd starkc && CARGO_TARGET_DIR=<scratch> cargo check --tests

# 2. The provider crates' OWN test targets. The starkc suite never compiles these,
#    and include_str!/metadata tests live here.
for p in stark-time stark-env stark-file stark-net stark-random stark-tls; do
  CARGO_TARGET_DIR=<scratch> cargo test --manifest-path packages/$p/native/Cargo.toml
done

# 3. The provider-backed integration suites.
cd starkc && CARGO_TARGET_DIR=<scratch> cargo test \
  --test c78_capability_declaration --test a10_stark_time_e2e --test c783_env_e2e \
  --test c784_file_e2e --test c785_time_closeout --test c788_source_time_e2e \
  --test c78_buffer_e2e --test c788_starkc_build

# 4. A real native build of a capability package — the end-to-end proof.
cd packages/stark-env-consumer && stark build && ./target/stark/debug/stark-env-consumer
# and the hardest case, which depends on the ABI, on starkc, AND on another provider:
cd packages/stark-tls-consumer && stark build

# 5. If the INSTALLED layout changed, prove it against an installed tree, not a checkout:
STARK_REQUIRE_INSTALLED_RUNTIME=1 <prefix>/bin/stark build
# Without that variable the compiler falls back to a source checkout and you are
# proving the checkout works. That variable is the whole experiment.

# 6. Lint, because CI denies warnings.
cd starkc && cargo clippy --all-targets --all-features -- -D warnings && cargo fmt --check
```

Then **push and read CI**. macOS and Windows lanes are not optional for a path change — they are
the only lanes that would have caught two of the three defects above. `C7.8 Native Capabilities`
runs the provider crates on all three platforms; that is the run to watch.

## Sweep for the whole family, not just the reported symptom

When one path breaks, the others of its kind usually broke too and are merely unreported:

```bash
grep -rn 'include_str!\|include_bytes!\|include!' packages/*/native/src/*.rs
grep -rn '#\[path' packages/*/native/src/*.rs
grep -rn 'path = "\.\.' packages/*/native/Cargo.toml
ls packages/*/native/build.rs 2>/dev/null
grep -rn 'CARGO_MANIFEST_DIR' packages/*/native/src/*.rs
grep -rn 'repo_root\|join("packages"' starkc/tests/*.rs   # must go through support/paths.rs
grep -rn 'stark-[a-z-]*/native' .github/workflows/*.yml starkc/scripts/*.py
```

## What you may not claim

- Not **"verified end to end"** unless step 5 ran with `STARK_REQUIRE_INSTALLED_RUNTIME=1`, when an
  installed layout is in scope.
- Not **"all tests pass"** when you ran one crate's suite. Name what you ran.
- Not **"works on all platforms"** from a macOS run. Say "green on macOS locally; Linux and Windows
  pending CI."
- If CI finds what you missed, say so plainly and say *why the local run could not have seen it* —
  that sentence is what stops the fourth occurrence.
