# C10 — compiler threat model and attack-surface inventory

**FROZEN 2026-08-09, before any finding was reviewed** (plan §11.1). The sixteen surfaces below
were fixed in `C10-0-OPENING-INVENTORY.md` §9; this file adds the adversary, the defences, and —
for each defence — **the test that would fail if the defence were removed.**

> **Method, inherited verbatim from `HC13-THREAT-MODEL.md`:**
> *A threat model with no falsifier attached is a list of intentions.*

A defence whose falsifier is `none` is recorded **UNVERIFIED**. It is not a defence yet.

---

# 1. The adversary, and what is out of scope

**In scope.** Someone who controls **source the compiler is asked to compile**, or **an artifact it
is asked to read**, or **bytes sent to the language server**. Concretely: a hostile dependency, a
pull request, a generated file, a repository opened in an editor.

**Out of scope**, and stated so the model is not read as claiming more than it does:

```text
an attacker who already controls the invoking user's account or PATH
an attacker who controls the machine's Rust toolchain or its sysroot
an attacker with the private key of a trusted release signer (there is no signing today — S16)
supply-chain compromise of a crates.io dependency (S14 records the surface, not a defence)
denial of service against the host machine as a whole
```

**The compiler is a build tool and trusts what it is told to build.** The interesting question is
not "can hostile source do anything" but **"can hostile source do something the user did not ask
for by asking for a build"** — read outside the package, execute something other than the
toolchain, or inject into generated code.

---

# 2. Surfaces, defences, falsifiers

| # | Surface | Defence | Falsifier | Status |
| --- | --- | --- | --- | --- |
| **S01** | source / module path traversal | A `mod` name is an **identifier token**; the grammar cannot express `/`, `\`, `.` or `..`, so a traversal never reaches `parent_dir.join(...)` | `c10c_security::s01_a_module_name_cannot_carry_a_path_separator_or_traversal` — asserts the hostile forms are rejected **by the grammar** and specifically never reach the file-not-found path. Its control, `s01_the_defence_is_the_grammar_and_a_plain_module_name_still_works`, keeps it from passing vacuously | **VERIFIED** |
| **S02** | package / cache filesystem access | Resolution follows the file's real directory; `logical_source_name` refuses to let a logical name become absolute and falls back to the bare file name if a path escapes the package root | `package.rs` identity tests; AS1a source-identity suites | **VERIFIED (inherited)** |
| **S03** | artifact parsing and limits | Manifests are parsed, not `eval`ed; a malformed manifest yields an error rather than a partial belief | `c10c_security::s03_a_malformed_manifest_is_refused_rather_than_partially_believed` — 7 hostile manifests incl. traversal and absolute `entry`, and 500-deep nesting | **VERIFIED for bounded failure**; see §3 residual R-S03 |
| **S04** | generated Rust / source escaping | Every user string reaching generated Rust is escaped by Rust's own `{:?}` / `escape_default`; manifests are escaped to **TOML** rules, not Rust `Debug` rules | `emit_types::str_constant_emits_an_escaped_rust_literal`; `build::manifest_paths_are_escaped_to_toml_rules_not_rust_debug_rules`; `build::a_generated_manifest_with_an_adversarial_runtime_path_stays_one_well_formed_line`; `c10c_security::s04_…` proves the payloads reach the backend; **and rustc itself rejects generated code that does not compile** | **VERIFIED** |
| **S05** | process execution | The compiler spawns exactly `cargo` and `rustc` (plus `rustc` probes). Arguments are constructed, never shell-interpolated — there is no shell | source-verified: 3 `Command::new` sites in `backend/generated_rust/build.rs`, 1 in `native_toolchain.rs`. No falsifier test | **UNVERIFIED** — R-S05 |
| **S06** | linker arguments | Delegated wholly to Cargo; STARK passes no `-C link-arg` | source-verified; no falsifier test | **UNVERIFIED** — R-S06 |
| **S07** | environment propagation | Reads are enumerable: `STARK_CARGO`/`CARGO`, `STARK_RUSTC`/`RUSTC`, `STARK_RUNTIME_DIR`, `STARK_REQUIRE_INSTALLED_RUNTIME`, `STARK_DUMP_MIR_ON_VERIFY_FAIL` | `c163_*`/`dev161_ambient_cargo_target_dir` covers the `CARGO_TARGET_DIR` interaction (DEV-161) | **PARTIAL** |
| **S08** | temporary files / directories | `env::temp_dir()` + PID + counter; **no shared root** (C6.4 row 17) | `c10c_security::s08_generated_temp_paths_do_not_collide_within_a_process`; C6.4 row 17's platform evidence | **VERIFIED**, with one known survivor outside the matrix using `/tmp` (a gate-7 fixture) |
| **S09** | archive extraction | Installer-side. Release archives are produced by `build-release.py`; the compiler extracts nothing | `test_build_release.py`; `release package smoke` on three platforms | **VERIFIED for production**, not for extraction of a hostile archive — R-S09 |
| **S10** | dependency / package provenance | **CORRECTED 2026-08-09 — see §2a.** `stark.lock` records, per dependency, the **canonical absolute `source` directory** and a verified **`sha256` content hash**; a mismatch is a hard error. The lockfile also carries a `capability_vocabulary` version | the content-hash-mismatch path (`"content hash mismatch for cached package"`); `qualify-first-party-packages.py` | **VERIFIED — by a different mechanism than this row first claimed** |
| **S11** | LSP workspace trust | The server reads files of the package containing an opened URI. **There is no trust prompt** — opening a folder analyses it | none | **UNVERIFIED, and see DEV-186** — R-S11 |
| **S12** | executable / tool paths | `command_path` prefers `STARK_CARGO`/`CARGO`, else the **bare name** `cargo` — i.e. **PATH lookup** | source-verified; no falsifier test | **ACCEPTED LIMITATION (class D)** — §4 |
| **S13** | denial-of-service inputs | The parser's `MAX_DEPTH = 200` bounds syntactic nesting; the LSP bounds nothing | `c10b_robustness` T1/T2/T9 — **and this surface is where C10-B's two findings live** | **FAILS** — DEV-214, DEV-186 |
| **S14** | dependency vulnerabilities | The compiler's own dependency set is deliberately tiny: `sha2` only, `default-features = false`. Charter §1.10 requires a necessity/maintenance/licence/security note per new dependency | `Cargo.toml` inspection; **no automated advisory scan runs** | **PARTIAL** — R-S14 |
| **S15** | licences | Project MIT. Dependency licences unaudited in CI | none | **UNVERIFIED** — R-S15 |
| **S16** | installer / release authenticity | `stark doctor` re-hashes every payload file against `manifest.json` — **integrity** | `release package smoke`; `stark doctor` tests | **INTEGRITY VERIFIED. AUTHENTICITY ABSENT** — class C, §4 |

---

# 2a. S10 CORRECTION — I described a defence that does not exist, and missed the one that does

**This row originally read:** *"the workspace root is the package's parent, so a dependency outside
`packages/` is refused by name."* **That is false at HEAD.** It was written from a remembered
constraint rather than from the code.

Measured:

```text
is_within_workspace   called at exactly ONE site, and it checks the ROOT MANIFEST:
                          "root package is outside the permitted workspace"
dependency paths      parent_dir.join(dep_path) -> canonicalize() -> DependencySource::Path
                      NO containment check. An external path dependency is ACCEPTED
```

**External path dependencies are permitted** — which is what Cargo does, and is a **class D accepted
operational limitation** rather than a vulnerability. It must be stated, because a reader of the
original row would have assumed a boundary that is not there.

**The real defence is stronger and more specific than the one I invented.** `LockfilePackage`
records, per dependency:

```text
source   Option<String>  "Auditable acquisition origin. Path dependencies use the canonical
                          absolute directory; registry dependencies use `registry`"
sha256   String          content hash, VERIFIED on use — a mismatch is an error, not a warning
```

plus a `capability_vocabulary` version on the lockfile itself. **Provenance is established by
recording and hashing what was actually used, not by restricting where it may live.** That is a
better design than containment, and one I would have missed entirely had I not checked.

**How this was found, because the method is the transferable part.** `CLAUDE.md` changed
mid-session and asserted that external path dependencies are supported — contradicting this row.
The charter puts `CLAUDE.md` at level 6 of the source-of-truth hierarchy and the implementation at
level 3, so **neither the old row nor the new summary was trusted; the code was read.** The summary
was right and my security document was wrong.

---

# 3. Residuals — surfaces with a defence but no falsifier

These are **not** claimed as defended. Naming them is the point.

```text
R-S03   the manifest probe proves BOUNDED FAILURE, not that a traversing `entry` is refused.
        `{"entry":"../../../etc/passwd"}` is accepted as a string; whether it is then USED is
        not tested here
R-S05   no test asserts the compiler spawns only cargo/rustc. A future change adding a spawn
        would not be noticed by anything
R-S06   no test asserts the absence of STARK-supplied linker arguments
R-S09   no test extracts a HOSTILE archive (path traversal in an archive entry, symlink escape)
R-S11   no LSP workspace-trust boundary exists, and none is tested
R-S14   no advisory scan (`cargo audit` or equivalent) runs in CI
R-S15   no licence audit runs in CI
```

**R-S05, R-S06 and R-S09 are cheap to close and were not closed here** — C10-C is a review packet
and building seven new suites would have made it an implementation packet.

---

# 4. Findings, classified — never collapsed into DEV numbers

Per plan §11.2, and mapped onto OD-3's populations.

## Class A — compiler correctness

*(none new; DEV-214 and DEV-186 are class A **and** class C, and are counted once, in population A)*

## Class B — security vulnerability

**None found.** No `SEC-C10-*` was allocated. Stated plainly because a security review that finds
nothing must say so rather than padding: the surfaces where an adversary would **gain** something —
injection into generated code (S04), path escape (S01, S02), execution of an unexpected binary
(S05, S12) — are each defended, and S01/S04/S08 now have falsifiers.

## Class C — release / distribution weakness (population B)

```text
S16   AUTHENTICITY ABSENT. `stark doctor` establishes INTEGRITY: it detects corruption and a
      partial extraction. It does not establish that the manifest came from a STARK release —
      anyone who can replace the payload can replace the manifest and its sidecar with it.
      A public distribution needs a signed manifest, a trusted release key, verification BEFORE
      installation, and platform notarisation. None exists
S13   DEV-214 and DEV-186 are denial-of-service surfaces as well as robustness defects
```

## Class D — accepted operational limitation

```text
S12   The compiler executes whichever `cargo` and `rustc` PATH resolves, and honours
      STARK_CARGO/CARGO/STARK_RUSTC/RUSTC overrides. This is what every build tool does, cargo
      included, and the trust boundary is the invoking user's environment. ACCEPTED — and it must
      be STATED in the release notes rather than left implicit, because it is the boundary a
      reader would otherwise assume differently
S11   Opening a folder in an editor analyses it. No workspace-trust prompt exists. ACCEPTED for
      a pre-alpha language server, and it becomes class B the moment analysis can execute
      anything — which today it cannot, because the interpreters have no host access
S02   `stark build` reads and writes inside the package and its target directory, and reads the
      sibling packages a manifest names. ACCEPTED: it is what a build is
```

---

# 5. What this model does not cover

```text
the runtime and the provider crates       their threat model is HC13-THREAT-MODEL.md; this file
                                          is the COMPILER's surface
the packages under packages/              first-party package security is the package track's
the tensor/ONNX artifact path             S03 covers artifact PARSING limits generically; the
                                          ONNX reader's own hardening is Gate 5/7 territory and
                                          the track is deferred research
a hostile TOOLCHAIN                       out of scope by §1
```
