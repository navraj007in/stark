# HC13 — release checklist

**Status:** current as of 2026-08-03

This is the list to work through before tagging a release of the HTTP client stack. It is written as
checks with **falsifiers**, not as intentions: each line says what to run and what a failure looks
like, so it can be executed by someone who did not write the code.

---

## 0. Classification

```text
HTTP client FEATURE track      COMPLETE   HC0-HC13
PUBLIC RELEASE readiness       BLOCKED
```

The two are not the same thing and this document exists because they were briefly conflated. What
blocks a release:

| blocker | state |
| --- | --- |
| SEC-HTTP-001, SEC-HTTP-002 | **fixed** in CD-376, with falsifiers; they were remote aborts |
| DEV-165 — `connect_timeout` accepted and ignored | open; either enforce it or remove the field |
| installer / distribution | **Phase I implemented** — see §0.1. Not yet a standalone signed distribution. |

## 0.1 What "release" means here — installer Phase I

An installable toolchain now exists (CD-377): `.tar.gz` plus a native installer per platform, a
versioned install tree with a `current` symlink, uninstall scripts, and `stark doctor` verifying
every payload file against a SHA-256 manifest. Verified on a clean prefix with
`STARK_REQUIRE_INSTALLED_RUNTIME=1`, which is what makes the check mean anything — without it the
compiler silently falls back to a source checkout and the installation proves nothing.

**It is Phase I, and the remaining distance is not small:**

| | |
| --- | --- |
| **Packages are not distributed** | the payload carries the compiler, `stark-runtime` and `stark-provider-abi`. It does not carry the first-party packages or their provider crates, so a clean machine cannot build an HTTP/TLS program without obtaining `stark-http-client`, `stark-tls`, `stark-net` and the rest separately. Verified: a fresh install plus a package with a `path` dependency to something absent fails at manifest resolution, as it should. |
| **Unsigned** | `manifest.json` establishes **integrity**, not **authenticity**. It detects corruption and partial extraction. It does not establish that the manifest came from a STARK release — an attacker replacing the payload replaces the manifest and the sidecar with it. A public distribution needs a signed manifest, a trusted release key, verification *before* installation, and platform notarisation. |
| **No release workflow** | nothing publishes these artefacts; they are built on demand. |

---

## 1. Gate

| # | check | command | failure looks like |
| --- | --- | --- | --- |
| 1.1 | 16-package qualification, all three Tier-1 lanes | CI: `first-party package qualification` | any lane red; `fail-fast: false` means the others still report |
| 1.2 | full workspace test | CI: `fmt, clippy, test` | any of the three platforms red |
| 1.3 | `ci-complete` green | CI | it gates on an explicit `needs:` list — **check the new job is in it** |
| 1.4 | Miri over the raw slot primitives | CI: `DEV-160 raw slot primitives under Miri` | a Stacked Borrows violation, or a toolchain that silently ran stable |
| 1.5 | spec fixture conformance | CI | grammar and examples out of step |

**On 1.3 and 1.4 together.** Both have failed for reasons that had nothing to do with the code:
`ci-complete` gates on a hand-maintained list, so a new job is outside it until someone adds it; and
a bare `cargo` inside `starkc/` honours `rust-toolchain.toml` over whatever a CI action installed,
so a Miri job can run stable and report a component error. **Adding a job is not the same as gating
on it, and installing a toolchain is not the same as using it.**

---

## 2. Evidence documents

| # | document | current |
| --- | --- | --- |
| 2.1 | HC13-QUALIFICATION-REPORT.md | ✅ |
| 2.2 | HC13-PLATFORM-MATRIX.md | ✅ |
| 2.3 | HC13-THREAT-MODEL.md | ✅ |
| 2.4 | HC13-KNOWN-LIMITATIONS.md | ✅ |
| 2.5 | HC13-RELEASE-CHECKLIST.md | ✅ (this) |

**2.6 — cross-reference sweep.** Before tagging, confirm: every count stated in prose matches its
list; no table contradicts the paragraph next to it; every "see §N" points at a section that exists;
and every limitation named in the report also appears in the limitations document. Counts and
cross-references are the first thing an external reviewer finds wrong.

---

## 3. Security

| # | check | falsifier |
| --- | --- | --- |
| 3.1 | header injection refused at the **serializer**, not only at construction | HC12.1's regression test IS the working exploit — it produced a real injected header line before the fix |
| 3.2 | credentials stripped on origin change, **asserted from the wire** | `/echo` reflects what the peer received; `GET|-|-|` proves absence, a boolean does not |
| 3.3 | downgrade refused before DNS and before connect | no SYN reaches the http target |
| 3.4 | hostname verification independent of chain validity | the mismatch peer's chain is **valid** |
| 3.5 | every smuggling primitive refused on the wire | HC13-THREAT-MODEL.md §2, table T4 |
| 3.6 | no `unsafe` outside the provider crates and `mod stark_proj` | generated MIR bodies contain none |
| 3.7 | parser arithmetic cannot trap on hostile input | `/bad-length-overflow`, `/bad-chunk-cumulative-overflow`; reverting either fix reproduces `integer overflow` (SEC-HTTP-001/002) |

---

## 4. Supply chain

| # | check | current |
| --- | --- | --- |
| 4.1 | one cryptographic backend, not two | `rustls` with `default-features = false` + `aws_lc_rs`; `ring` not compiled in |
| 4.2 | exact pins in the manifest, not ranges | `=0.23.43`, `=2.2.0`, `=1.15.1`, `=0.8.2` |
| 4.3 | transitive crypto pinned by lockfile | `aws-lc-rs 1.17.3`, `aws-lc-sys 0.43.0` |
| 4.4 | `Cargo.lock` committed for every provider crate | ✅ |
| 4.5 | provider versions recorded in the platform matrix | HC13-PLATFORM-MATRIX.md §4 |
| 4.6 | audit for advisories against the pinned set | **not automated — do this by hand before tagging** |

---

## 5. Compatibility

| # | check | note |
| --- | --- | --- |
| 5.1 | `Header` / `HeaderMap.entries` still public | a known weakness; making them private is a **breaking** change and must not slip into a patch release |
| 5.2 | dot-segment resolution still absent | belongs in `stark-url`; adding it changes redirect targets, so it is behaviour-visible |
| 5.3 | DEV-160b workaround still present in `send()` | remove only when cross-block absorption lands, and only after the original source form builds natively |
| 5.4 | `ClientConfig.connect_timeout` still accepted and ignored | DEV-165. Either enforce it or remove the field — shipping a deadline that silently does nothing is the worse of the three options |

---

## 6. The two things most likely to be got wrong

**6.1 — Do not mark phase-specific timeouts as fully proved, and do not count the routes as
phases.** TWO of five phases are proved (`ReadResponse`, `TlsHandshake`) by three routes — both
read stalls report `ReadResponse`. An earlier draft of the report said three by counting routes.
`WriteRequest` is unproven, `Connect` is **not implemented** (DEV-165) and `Resolve` is **absent**.
Filing the last two as "unproven" invites a reader to assume they merely lack a test.

**6.2 — Do not treat a green single-platform run as a matrix.** DEV-163 was green on Windows and
wrong on Unix, from identical source, and no amount of re-running one platform would have shown it.
If a lane is skipped, the release is not qualified — which is why every peer in the harness asserts
its bind rather than attempting it.

---

## 7. Sign-off

```text
[ ] 1.1-1.5   gate green on all three Tier-1 platforms
[ ] 2.1-2.6   evidence documents present and cross-referenced
[ ] 3.1-3.6   security falsifiers pass
[ ] 4.1-4.6   supply chain pinned; advisories checked BY HAND
[ ] 5.1-5.3   known breaking changes deliberately deferred or taken
[ ] 6.1-6.2   partial claims still reported as partial
[ ] 0         installer Phase I verified on all three platforms; the release states plainly
              that it is unsigned and that packages are obtained separately
```
