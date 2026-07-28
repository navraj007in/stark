# WP-C7.2 — the reproducibility contract

**Status:** `PARTIAL` — classification, measurement and the path-remapping change are done. The
machine-readable build manifest (§4.6) and cross-machine/CI-runner comparison (§4.3) remain.
**Measured at:** `31c3aac` + this change, macOS arm64, seven frozen workloads.

## 1. Classification, per artefact AND per profile

§4.1 forbids one global "reproducible" label. The measurements show why: a single build produces
artefacts in three different classes, and the executable's class depends on the profile.

| artefact | class | basis |
| --- | --- | --- |
| generated Rust | `BYTE-REPRODUCIBLE` | two paths × both profiles, 7 workloads |
| generated `Cargo.toml` | `SEMANTICALLY-REPRODUCIBLE` | byte-identical per machine; embeds the compiler's own runtime path, so it varies across installations |
| `stark.lock` | `BYTE-REPRODUCIBLE` | two paths, 7 workloads |
| executable — **release** | `BYTE-REPRODUCIBLE` | two paths, byte-identical, zero embedded build paths |
| executable — **debug**, macOS | `NOT-YET-REPRODUCIBLE` | 31 embedded build-directory strings; sizes differ by the path-length delta |
| executable — **debug**, other platforms | `UNMEASURED` | see the note below — this was wrongly generalised from the macOS measurement |
| debug symbols | `PLATFORM-METADATA-EXCLUDED` | not produced as a separate artefact today |

## 2. What was changed, and what it did not achieve

`--remap-path-prefix=…` is now set for the generated crate's build, mapping the crate directory to
`/stark/crate` and the runtime crate to `/stark/runtime`. Each prefix is remapped in **both** its
literal and canonicalised form, because on macOS `/var` resolves to `/private/var` and a remap
written with the unresolved path matches nothing — which is precisely what the first attempt did.

The flags are passed through **`CARGO_ENCODED_RUSTFLAGS`**, not `RUSTFLAGS`, and the distinction is
load-bearing. `RUSTFLAGS` is space-separated, so a build directory containing a space splits one
`--remap-path-prefix=FROM=TO` into two arguments and rustc rejects the fragment outright — every
build under such a path fails, rather than merely going un-remapped. `CARGO_ENCODED_RUSTFLAGS`
separates on `\x1f`, which no path contains. This was shipped as `RUSTFLAGS` first and broke the
spaces-in-paths portability test on all three platforms; corrected in CD-190, with a dedicated
regression test in `c72_reproducibility.rs` that asserts both that such a build succeeds and that
the remap actually applied.

It removes 31 of the 62 embedded strings from a debug binary, including **every reference to the
compiler's own installation directory**. That is worth having on its own: a debug binary should not
name the machine that built it.

**It is not what makes release reproducible, and it does not make debug reproducible.** Both were
measured rather than assumed:

- reverting the remap and rebuilding, release remained byte-identical across paths and embedded zero
  remapped markers — so release reproduced before the change and after it;
- the 31 strings that survive in debug are recorded by the **linker**, not by rustc. macOS writes
  object-file paths into the debug map for `dsymutil`, and no rustc path flag reaches them.

Claiming the remap "fixed reproducibility" would credit it for an outcome it did not cause.

## 3. Reproducibility input identity (§4.2)

A reproducibility claim binds: STARK source contents, package manifests, `stark.lock`, dependency
graph identity, compiler version, runtime version, backend version, MIR version, language/extension
options, target triple, **profile**, host compiler version, and the runtime crate location. The
first eleven are already hashed into the generated crate's build key (`BuildVersions`), and WP-C7.1
added the profile to it. The runtime location is the one input that is bound but not reproducible
across machines — see the `Cargo.toml` row above.

Explicitly EXCLUDED, and therefore outside any byte claim: absolute checkout path (release only —
debug still carries it), build timestamp, host user, and temporary-directory names. C7.0 verified
that PID and thread-ID temporary paths do not reach any artefact.

## 4. What remains

1. **The build manifest (§4.6)** — a deterministic machine-readable record of the identity above,
   emitted per build. Not yet implemented.
2. **Cross-machine and CI-runner comparison (§4.3)** — everything here is same-machine. The
   `Cargo.toml` classification predicts a cross-machine byte difference; that prediction is
   untested.
3. **Debug reproducibility** — would need linker-level path control (`-oso_prefix` on macOS, and the
   equivalent elsewhere). Recorded rather than attempted, because it is platform-specific work whose
   benefit is smaller than release reproducibility, which already holds.

## Correction (CD-190): debug non-reproducibility is a macOS finding, not a STARK property

The table above originally carried one global `NOT-YET-REPRODUCIBLE` row for debug executables, and
the conformance test asserted it on every platform. That was a measurement taken on macOS and stated
as if it were universal. CI refuted it: `c72_reproducibility` failed on **linux-x64 only**, which is
what you would expect if the residual paths come from the macOS linker's debug map rather than from
anything rustc or STARK emits — remove that mechanism and remapping is sufficient.

The test now asserts only on macOS, where the mechanism has been measured, and prints a
`C72-DEBUG-REPRO platform=… identical=…` line on every platform so the remaining rows can be filled
in from evidence rather than inference. Generalising one platform's measurement is exactly the error
that produced the failure; the fix is not a looser assertion but a narrower claim.
