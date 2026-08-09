# C10-C — Security review

**Packet:** C10-C, WP-C10.4. **Date:** 2026-08-09.
**Baseline:** `f12ececca6d4bdabf828d657c4a4f719a7f9c39a`.
**Frozen surface:** `C10-0-OPENING-INVENTORY.md` §9 (16 surfaces, frozen before review).
**Model:** `STARKLANG/docs/compiler/C10-THREAT-MODEL.md`.
**Probes:** `starkc/tests/c10c_security.rs` (5 tests).

**Verdict: no class-B security vulnerability found. One class-C weakness governs the release
wording. Seven surfaces carry a defence with no falsifier and are recorded UNVERIFIED rather than
claimed.**

---

# 1. What was done, in the order the plan requires

```text
1  the surface inventory was FROZEN in C10-0 §9, before any finding was reviewed   (plan §11.1)
2  the threat model was written, naming a FALSIFIER for every claimed defence
3  probes were built for the surfaces where a falsifier is cheap: S01, S03, S04, S08
4  findings were classified A/B/C/D and mapped onto OD-3's populations
```

Order matters and is the reason this packet is short: a surface list assembled *after* looking at
findings is a list of the things that happened to be found.

---

# 2. The headline: no class-B finding, and why that is a real result rather than a shrug

**No `SEC-C10-*` was allocated.** A security review that finds nothing must say so plainly instead
of padding the register, and the reason is structural rather than lucky:

- **The compiler has no shell.** All four `Command::new` sites construct argument vectors. There is
  no string a hostile source could be interpolated into.
- **The compiler has almost no dependencies.** One: `sha2`, `default-features = false`.
- **Generated Rust is escaped by Rust's own escaper, and then compiled by rustc.** A literal that
  broke out of its string would not be a silent injection — it would be a build failure. rustc is a
  genuine external control here, narrowed only by `RA-LINTS` (two deny-by-default lints suppressed).
- **The interpreters have no host access at all.** `src/interp.rs` contains the string `provider`
  zero times. Analysis cannot execute anything, which is what keeps S11 (no LSP workspace trust) in
  class D rather than class B.

**The surfaces where an adversary would actually gain something are the defended ones.** Injection
into generated code (S04), path escape (S01, S02), and execution of an unexpected binary (S05, S12)
are each defended, and three of those now have falsifiers that did not exist this morning.

---

# 3. S01 — the probe that had to be strengthened, and why it is worth recording

The first version asserted only that a hostile `mod` name produced *some* diagnostic. It passed.

**It was nearly vacuous**: `mod ordinary_name;` in a bare program also produces a diagnostic — the
file is missing. "Some diagnostic appeared" therefore does not distinguish a defended surface from
an undefended one.

Reading the actual messages is what exposed it:

```text
mod ordinary_name;         ->  "file not found for module 'ordinary_name'"     <- reached the FS
mod ../../../etc/passwd;   ->  "expected a module name, found `..`"            <- rejected by GRAMMAR
mod a/b;                   ->  "expected `{` or `;`, found `/`"
mod .hidden;               ->  "expected a module name, found `.`"
```

The defence is that the hostile forms **never reach the filesystem layer at all**, so the assertion
is now that they never produce the file-not-found diagnostic. That is a claim about *which* code
path ran, not merely that something failed.

> **A passing test is not evidence until you know what it would have to see to fail.** This is the
> AS8 lesson arriving in a security packet, and it cost one `println!` to find.

---

# 4. Findings

| Surface | Finding | Class | Population | Disposition |
| --- | --- | --- | --- | --- |
| S16 | **Integrity, not authenticity.** `stark doctor` re-hashes the payload against `manifest.json`; anyone who can replace the payload can replace the manifest and sidecar with it. No signed manifest, no release key, no verification before installation, no notarisation | **C** | B | **Governs the release wording.** C10-Q may not describe the distribution as verified or trusted |
| S13 | DEV-214 (operator-chain stack overflow) and DEV-186 (unbounded `Content-Length`) are denial-of-service surfaces as well as robustness defects | A + C | A | Counted once, in A. DEV-214 needs an owner call |
| S12 | The compiler executes whichever `cargo`/`rustc` PATH resolves, honouring `STARK_CARGO`/`CARGO`/`STARK_RUSTC`/`RUSTC` | **D** | — | **ACCEPTED — and must be STATED in the release notes.** It is what every build tool does; it is also the boundary a reader would otherwise assume differently |
| S11 | No LSP workspace-trust boundary: opening a folder analyses it | **D** | — | ACCEPTED for a pre-alpha server. **Becomes class B the moment analysis can execute anything** — today it cannot |
| S02 | `stark build` reads/writes inside the package and target dir, and reads sibling packages a manifest names | **D** | — | ACCEPTED: it is what a build is |
| S01, S04, S08 | Defended, and now falsifiable | — | — | Newly VERIFIED by `c10c_security.rs` |
| **S10** | **This review described a defence that does not exist.** It claimed a dependency outside the workspace is "refused by name". `is_within_workspace` is called at ONE site and checks the ROOT manifest only, so external path dependencies are accepted | **D** | — | **CORRECTED** — `C10-THREAT-MODEL.md` §2a. The real control is `stark.lock`'s per-dependency canonical `source` plus verified `sha256`, which is stronger than the containment I imagined |
| R-S03/05/06/09/11/14/15 | Seven surfaces with a defence but **no falsifier** | — | C | **UNVERIFIED.** Named, not claimed |

---

# 5. The residuals, and an honest note about why they are residuals

```text
R-S03   the manifest probe proves BOUNDED FAILURE. It does NOT prove that a traversing `entry`
        (`{"entry":"../../../etc/passwd"}`) is refused — it is accepted as a string, and whether
        it is subsequently USED as a path is untested. THE MOST WORTH CLOSING OF THE SEVEN
R-S05   nothing asserts the compiler spawns only cargo and rustc. A future change adding a spawn
        would be noticed by no test
R-S06   nothing asserts the absence of STARK-supplied linker arguments
R-S09   no hostile ARCHIVE is extracted anywhere (entry-path traversal, symlink escape)
R-S11   no workspace-trust boundary exists, so none is tested
R-S14   no advisory scan (`cargo audit` or equivalent) runs in CI
R-S15   no licence audit runs in CI
```

**R-S05, R-S06 and R-S09 are each roughly one test.** They were not written because C10-C is a
review packet, and building seven suites would have turned it into an implementation packet — the
scope drift plan §3.2 forbids. They are recorded as population C residuals with an owner, not
quietly dropped.

**R-S03 is the one I would close first**, because it is the only residual where the *defence itself*
is unknown rather than merely untested.

---

# 6. What C10-C does NOT claim

```text
NOT "the compiler is secure"        16 surfaces reviewed, 7 with no falsifier. The correct
                                    statement is: no class-B finding, on this surface list,
                                    at this baseline
NOT a runtime/provider review       that is HC13-THREAT-MODEL.md's subject. This file is the
                                    COMPILER's surface
NOT a dependency audit              S14 records the surface. No scan ran
NOT a licence audit                 S15 likewise
NOT a distribution security claim   S16 is exactly the opposite of one
NOT an ONNX artifact-reader audit   S03 covers artifact PARSING limits generically; the ONNX
                                    reader's own hardening belongs to a deferred track
```

**CE9 note.** Charter §2.3 makes archive extraction, process execution, code generation and native
linking owner-escalated decisions. This packet **reviewed** those surfaces and **changed none of
them**. No CE9 decision is requested, because no finding required a behaviour change. Had a class-B
finding appeared, C10-C would have stopped here.
