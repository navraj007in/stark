# STARK Installer and Distribution Work Package

## Status

**Proposed**

## Target window

**Ecosystem consolidation phase — January to February 2027**

## Purpose

Deliver a repeatable, secure, cross-platform installation and update mechanism for STARK.

The installer must make STARK usable by a developer who does not know the repository layout, compiler internals, provider architecture, Cargo workspace structure, or first-party package qualification process.

The installer is not merely a wrapper around the compiler binary. A complete STARK installation includes:

- the `stark` command;
- the `starkc` compiler executable where it remains separately exposed;
- first-party standard packages;
- built-in provider manifests;
- native provider crates and their required assets;
- target metadata;
- templates and examples where included;
- version and build provenance;
- uninstall and update metadata;
- the files needed for supported offline native builds.

The distribution mechanism must preserve STARK's existing guarantees:

- explicit provider identity;
- pinned provider versions;
- deterministic package and build inputs;
- no silent fallback between capabilities or security profiles;
- no dependence on a developer's warm Cargo cache;
- no hidden network access during an offline build;
- no compiler-source edits merely to add or locate a provider.

---

# 1. User-facing outcome

A new user should be able to install STARK and run:

```bash
stark --version
stark doctor
stark new hello
cd hello
stark check
stark run
stark build --release
```

A program using first-party packages should build without requiring the user to clone the STARK repository:

```stark
use stark_json;
use stark_http_client;
use stark_tls;
```

On supported systems, the installation should include everything required to compile an offline first-party native consumer after the installation payload has been downloaded.

---

# 2. Supported platforms

Initial Tier-1 installer targets:

| Platform | Architecture | Preferred package |
|---|---:|---|
| Linux | x86_64 | `.tar.gz` plus optional `.deb` |
| macOS | arm64 | signed `.pkg` plus `.tar.gz` |
| Windows | x86_64 | signed `.msi` plus `.zip` |

Possible later targets:

- Linux arm64;
- macOS x86_64, if retained;
- Windows arm64;
- Homebrew;
- Winget;
- Scoop;
- apt repository;
- container image;
- Nix package.

The portable archive remains the canonical fallback on every platform. Native installers may add path integration and uninstall registration, but must install the same logical payload.

---

# 3. Installation layout

Use a versioned installation root.

## Unix-like systems

System installation:

```text
/usr/local/stark/
    versions/
        0.1.0/
            bin/
            lib/
            packages/
            providers/
            targets/
            share/
            manifest.json
    current -> versions/0.1.0
```

User installation:

```text
~/.local/stark/
    versions/
    current
```

Commands exposed through:

```text
/usr/local/bin/stark
/usr/local/bin/starkc
```

or:

```text
~/.local/bin/stark
~/.local/bin/starkc
```

## Windows

```text
%LOCALAPPDATA%\Stark\
    versions\
        0.1.0\
            bin\
            lib\
            packages\
            providers\
            targets\
            share\
            manifest.json
    current\
```

The installer may update the user's `PATH`, with explicit consent where the platform requires it.

## Rule

The compiler must not infer repository-relative locations in an installed environment.

All installed resources must be resolved through one installation-root service or explicit command-line configuration.

---

# 4. Distribution contents

Each release payload must contain:

## 4.1 Executables

- `stark`;
- `starkc`, if still separately supported;
- optional helper executable for updates;
- optional diagnostic/support utility if not integrated into `stark doctor`.

## 4.2 Standard packages

All release-qualified first-party packages, including their:

- STARK source;
- package manifests;
- package version;
- content digest;
- public API documentation;
- native consumer fixtures where retained for support diagnostics.

Packages must be installed read-only under the installation root.

User packages and build outputs must never modify the installed package bundle.

## 4.3 Provider definitions

- provider manifests;
- provider identity;
- provider version;
- provider content hash;
- capability declarations;
- resource declarations;
- target compatibility;
- provider crate path relative to the installed provider root;
- profile information, including normal and optional security profiles.

Provider location must continue to come from the provider manifest and an explicit installation root. Do not reintroduce a hardcoded compiler match table.

## 4.4 Native provider crates

Include the source and metadata needed to build the qualified native providers.

Where a provider depends on registry crates, the release must choose and document one distribution strategy:

### Strategy A — vendored Cargo source

Include a Cargo vendor directory and generated source replacement configuration.

Advantages:

- strongest offline story;
- exact source reviewed and shipped;
- no dependence on registry state.

### Strategy B — release-local Cargo registry/cache seed

Include a release-owned registry source/cache populated with the exact pinned dependencies.

Advantages:

- closer to normal Cargo layout;
- may be easier for dependencies with build scripts.

### Strategy C — prebuilt static provider libraries

Include prebuilt provider libraries per target and profile.

Advantages:

- fastest installation and builds.

Costs:

- larger release matrix;
- compiler and linker compatibility constraints;
- provenance and ABI obligations become stricter.

The first release should prefer **vendored Cargo source** unless qualification proves it unworkable.

A developer's global Cargo registry is not part of the installation contract.

## 4.5 Target metadata

For each supported target:

- target triple;
- data layout contract;
- provider availability;
- supported profiles;
- linker requirements;
- minimum operating-system version;
- required host tools;
- release qualification status.

## 4.6 Documentation and templates

Include:

- quick start;
- command reference;
- package index;
- provider/capability reference;
- examples;
- project template;
- troubleshooting guide;
- release notes;
- license notices;
- third-party notices.

---

# 5. Installation manifest

Every installed version must have one canonical machine-readable manifest.

Suggested file:

```text
manifest.json
```

Required fields:

```json
{
  "schema_version": 1,
  "stark_version": "0.1.0",
  "release_channel": "stable",
  "build_commit": "<git-sha>",
  "build_timestamp": "<reproducible-or-declared-timestamp>",
  "host_target": "aarch64-apple-darwin",
  "compiler": {
    "version": "0.1.0",
    "sha256": "<digest>"
  },
  "mir_version": "0.3",
  "runtime_version": "0.1",
  "backend_version": "0.1",
  "packages": [],
  "providers": [],
  "files": [],
  "signing": {
    "scheme": "<scheme>",
    "key_id": "<key-id>"
  }
}
```

Each file entry should record:

- relative path;
- size;
- SHA-256 digest;
- logical component;
- executable bit where relevant.

The installer, updater, `stark doctor`, and uninstaller must all read the same manifest format.

Do not maintain separate hand-written component inventories in multiple scripts.

---

# 6. Installer commands

Recommended command surface:

```bash
stark install
stark update
stark uninstall
stark doctor
stark env
```

The native platform installer may perform the first installation, but the installed CLI should provide consistent lifecycle commands afterward.

## `stark doctor`

Must report:

- installed STARK version;
- installation root;
- resolved package root;
- resolved provider root;
- host and target triple;
- provider availability;
- missing build prerequisites;
- Cargo/Rust toolchain state;
- C/C++ compiler state where needed;
- CMake and Go state for optional profiles;
- write permissions for cache/build directories;
- offline build readiness;
- installation-manifest integrity;
- PATH conflicts;
- multiple installed versions;
- current selected version.

Output should have:

- human-readable form;
- `--json` form;
- non-zero exit status for actionable failures.

---

# 7. Toolchain policy

The installation must define whether STARK:

1. bundles the Rust toolchain;
2. installs a pinned toolchain through `rustup`;
3. requires a compatible preinstalled toolchain.

For the first public installer, the preferred model is:

> Install or use a STARK-owned pinned Rust toolchain isolated from the user's default Rust toolchain.

Suggested location:

```text
<stark-home>/toolchains/rust/<version>/
```

Benefits:

- reproducible native builds;
- no dependence on the user's `rustup default`;
- no breakage when the user's Rust version changes;
- exact compiler behaviour for qualified releases.

Requirements:

- record exact Rust version;
- record Cargo version;
- record installed components;
- never modify the user's default toolchain;
- allow `STARK_RUST_TOOLCHAIN` override for advanced users;
- include the selected toolchain in `stark doctor`;
- make unsupported overrides explicit and non-qualified.

If bundling is initially too large, the bootstrap installer may fetch the pinned toolchain, but the full offline installer must include it or provide a separately downloadable offline toolchain bundle.

---

# 8. Cache and generated-build layout

Separate immutable installation data from mutable user data.

Suggested user-state root:

## Unix-like

```text
~/.stark/
    cache/
    builds/
    registry/
    logs/
    config/
    toolchains/
```

## Windows

```text
%LOCALAPPDATA%\Stark\UserData\
```

Required properties:

- content-addressed build directories;
- target triple included in identity;
- debug/release profile included in identity;
- compiler/runtime/backend versions included in identity;
- provider identity/version/hash included in identity;
- source and package graph included in identity;
- no collision between security profiles;
- atomic cache population;
- process-safe concurrent builds;
- failed builds do not become cache hits;
- uninstall does not delete user projects;
- cache removal is an explicit command.

Suggested commands:

```bash
stark cache status
stark cache clean
stark cache verify
```

---

# 9. Offline installation and builds

Define two distinct guarantees.

## 9.1 Offline installation

A complete offline installer can be transferred to a machine and installed without internet access.

It must include:

- executables;
- standard packages;
- provider sources or binaries;
- pinned build dependencies;
- required toolchain, or a matching offline toolchain bundle;
- target metadata;
- signatures and checksums.

## 9.2 Offline native build

After installation, a qualified first-party program can run:

```bash
stark build --offline
```

without contacting:

- crates.io;
- GitHub;
- package mirrors;
- certificate download services;
- toolchain servers.

The build must fail with a named diagnostic if an external package or provider dependency was not included in the offline set.

It must not silently switch from offline to online.

---

# 10. Security model

## 10.1 Release signing

Release artifacts must be signed.

At minimum:

- publish SHA-256 checksums;
- sign the checksum manifest;
- verify the signature before installation;
- pin or distribute the release public key through a documented trust path;
- record key ID in the installation manifest.

Platform packages should additionally use:

- Apple Developer ID signing and notarisation for macOS;
- Authenticode signing for Windows;
- repository/package signing for Linux package feeds.

## 10.2 Archive extraction safety

The installer must reject:

- absolute archive paths;
- `..` path traversal;
- symlink escape from installation root;
- duplicate paths with conflicting contents;
- case-folding collisions on case-insensitive filesystems;
- unexpected executable files;
- manifest/file mismatches.

Extraction should happen into a staging directory, followed by verification, then atomic activation.

## 10.3 Privilege boundary

Default to user-local installation where practical.

System-wide installation must:

- request elevation only for the final installation step;
- never run downloaded project code as administrator/root;
- verify all files before privileged copying;
- keep caches and generated builds outside privileged installation directories.

## 10.4 Provider integrity

Before a native build, verify installed provider content against the release manifest or a trusted external-provider approval record.

A modified built-in provider must not be used silently.

## 10.5 No silent profile substitution

When a requested provider profile is unavailable:

```text
ProfileUnavailable
```

or equivalent must be returned.

Never silently replace:

- FIPS profile with normal profile;
- secure randomness with deterministic randomness;
- bundled roots with system roots;
- one provider version with another.

---

# 11. Installation process

Recommended transaction:

```text
download/open installer
→ verify release signature
→ inspect host compatibility
→ choose user/system installation
→ extract into staging
→ verify every file against manifest
→ check required toolchain/prerequisites
→ atomically activate version
→ update current-version pointer
→ update PATH integration
→ run stark doctor
→ record installation receipt
```

If any step fails:

- current installation remains usable;
- staging directory is removable;
- no half-active version;
- error identifies the failed component;
- logs are retained in the user-state directory.

---

# 12. Side-by-side versions

Support multiple installed versions.

Commands:

```bash
stark versions
stark use 0.1.0
stark use stable
stark use nightly
```

Project-level version pinning should later be possible through a file such as:

```text
.stark-version
```

Resolution order should be explicit:

1. command-line version override;
2. project version file;
3. environment override;
4. selected global/user version.

The selected compiler and standard-package bundle must come from the same release unless an explicit compatibility contract says otherwise.

---

# 13. Update model

## 13.1 Channels

Initial channels:

- `stable`;
- `preview`;
- optional `nightly`.

A channel identifies an update stream, not an implicit trust reduction.

## 13.2 Update sequence

```text
fetch signed release index
→ select compatible release
→ download complete artifact or verified delta
→ verify
→ install side-by-side
→ run doctor/smoke tests
→ switch current pointer atomically
```

Do not overwrite the active installation in place.

## 13.3 Rollback

Support:

```bash
stark use <previous-version>
```

Automatic rollback should occur when post-install smoke tests fail before activation.

## 13.4 Delta updates

Defer delta updates until full-package updates are stable.

Correctness and rollback matter more than bandwidth initially.

---

# 14. Uninstall

The uninstaller must distinguish:

## Installation-owned data

Safe to remove:

- selected installed versions;
- installation manifests;
- installed provider/package payload;
- PATH integration;
- installer receipts.

## User-owned data

Do not remove by default:

- projects;
- source files;
- user configuration;
- downloaded external packages;
- build cache;
- logs;
- toolchains shared with other STARK versions.

Offer explicit flags:

```bash
stark uninstall --version 0.1.0
stark uninstall --all
stark uninstall --all --purge-cache
```

Before removal, report what will be deleted.

---

# 15. External packages and providers

The built-in installer must not automatically trust external native providers.

An external provider installation flow must require:

- explicit opt-in;
- provider identity;
- exact version;
- checksum;
- trust tier;
- release/development scope;
- target compatibility;
- provider manifest validation.

Recommended future command:

```bash
stark provider add ./provider-package \
  --version 1.2.0 \
  --sha256 <digest> \
  --trust development
```

Release builds must reject providers admitted only for development where the existing trust policy requires that distinction.

---

# 16. Platform-specific requirements

## 16.1 Linux

Portable archive:

- install under user-selected root;
- shell script only as a thin verified bootstrap;
- no dependency on Bash-specific behaviour unless documented.

Optional `.deb`:

- install under `/opt/stark` or `/usr/lib/stark`;
- expose commands under `/usr/bin`;
- register package ownership cleanly;
- avoid modifying shell startup files.

Later package feeds must be signed.

## 16.2 macOS

`.pkg` installer:

- signed with Developer ID Installer;
- notarised;
- no quarantine-breaking workaround;
- install versioned payload;
- provide uninstall instructions because macOS packages do not provide a universal uninstall UI.

Portable archive remains supported for CI and advanced users.

## 16.3 Windows

`.msi` installer:

- per-user installation by default;
- optional per-machine installation;
- PATH component;
- Apps & Features registration;
- upgrade code and product versioning;
- signed binaries and MSI;
- long-path handling;
- no PowerShell execution-policy weakening.

Portable `.zip` remains supported.

---

# 17. Bootstrap installer

A small bootstrap installer may be provided for convenience.

It may:

- detect platform;
- download the correct signed full installer;
- verify it;
- launch or extract it.

It must not:

- execute an unverified download;
- infer trust from HTTPS alone;
- silently install prerequisites with elevated privilege;
- make the bootstrap path the only installation method.

The full offline artifacts remain first-class release products.

---

# 18. CI and release pipeline

The release pipeline must produce installers from a tagged commit.

Required stages:

```text
source/tag verification
→ compiler build
→ first-party package qualification
→ provider qualification
→ installer payload assembly
→ manifest generation
→ file hashing
→ platform packaging
→ signing/notarisation
→ clean-machine install
→ offline smoke build
→ update/rollback test
→ uninstall test
→ release publication
```

## 18.1 Clean-machine tests

For each Tier-1 platform:

1. start from a clean VM/runner;
2. install using the produced installer;
3. verify `PATH`;
4. run `stark doctor`;
5. create a project;
6. check it;
7. run it;
8. build debug;
9. build release;
10. build a first-party HTTP/TLS consumer offline;
11. verify build provenance;
12. install a second version;
13. switch versions;
14. roll back;
15. uninstall one version;
16. confirm the remaining version still works.

## 18.2 Hostile-environment tests

Test with:

- pre-existing `CARGO_TARGET_DIR`;
- unusual `HOME`;
- paths containing spaces;
- non-ASCII user profile path;
- read-only installation directory;
- no network;
- missing optional tools;
- multiple Rust toolchains;
- conflicting `stark` executable earlier on `PATH`;
- simultaneous builds;
- interrupted installation;
- interrupted update.

## 18.3 Reproducibility

At minimum, prove that:

- manifest contents are deterministic from declared inputs;
- package/provider file digests are stable;
- release metadata records any non-reproducible fields;
- two builds from the same source identify all differences.

Full bit-for-bit installer reproducibility may be a later gate where platform signing introduces timestamps.

---

# 19. Diagnostics

Installer errors must be classified.

Suggested categories:

```text
UnsupportedPlatform
UnsupportedArchitecture
SignatureVerificationFailed
ManifestMismatch
ArchiveTraversalRefused
InstallationPermissionDenied
ToolchainUnavailable
ToolchainInstallFailed
ProviderPayloadMissing
OfflineDependencyMissing
PathConflict
VersionAlreadyInstalled
UpdateIndexInvalid
RollbackUnavailable
UninstallIncomplete
```

Errors must identify:

- operation;
- component;
- relevant path;
- expected and actual version/digest where safe;
- corrective action.

Do not collapse failures into a generic "installation failed".

---

# 20. Telemetry and privacy

Initial releases should not require telemetry.

If update checks are enabled:

- make the behaviour visible;
- allow disabling them;
- do not upload project names, source, package graph, build errors, or machine identifiers;
- document exactly what request is sent;
- keep manual update commands available.

Crash reports must be explicit opt-in.

---

# 21. Documentation deliverables

Create:

- installation guide;
- offline installation guide;
- upgrade guide;
- rollback guide;
- uninstall guide;
- enterprise/offline deployment guide;
- proxy/firewall guide;
- toolchain policy;
- release signing and verification guide;
- troubleshooting matrix;
- provider installation guide;
- version-selection guide.

---

# 22. Implementation phases

## Phase I — Canonical portable bundle

Deliver:

- versioned directory layout;
- canonical manifest;
- portable archives;
- installation-root resolution;
- user-state/cache separation;
- `stark doctor`;
- clean-machine smoke test.

This is the minimum viable distribution.

## Phase II — Offline native-build bundle

Deliver:

- pinned Rust toolchain policy;
- vendored native dependencies;
- offline first-party provider builds;
- offline HTTP/TLS consumer test;
- cache and target isolation.

## Phase III — Native platform installers

Deliver:

- macOS `.pkg`;
- Windows `.msi`;
- Linux `.deb`;
- signing;
- PATH integration;
- uninstall registration/instructions.

## Phase IV — Updates and side-by-side versions

Deliver:

- signed release index;
- `stark update`;
- atomic activation;
- rollback;
- project version pinning.

## Phase V — External provider installation

Deliver:

- explicit trust workflow;
- version/checksum pinning;
- provider installation records;
- development-versus-release enforcement.

---

# 23. Acceptance criteria

The installer programme is complete when all of the following are true:

## Packaging

- one canonical payload definition;
- every installed file appears in the manifest;
- all files verify after installation;
- compiler, packages and providers resolve without repository-relative paths.

## Installation

- clean user-local install works on all Tier-1 platforms;
- system install works where offered;
- paths with spaces and non-ASCII characters work;
- installation is transactional;
- a failed install leaves the prior version intact.

## Build readiness

- `stark doctor` reports a qualified environment;
- a generated native hello-world builds in debug and release;
- a first-party HTTP/TLS consumer builds and runs;
- an offline build succeeds without a warm global Cargo cache;
- hostile `CARGO_TARGET_DIR` does not redirect or break output discovery.

## Security

- release signature is verified;
- archive traversal is refused;
- built-in provider tampering is detected;
- no silent provider/profile substitution;
- platform installers are signed where applicable.

## Lifecycle

- two versions install side-by-side;
- version switching is atomic;
- rollback works;
- uninstall removes only selected installation-owned files;
- user projects remain untouched.

## Evidence

- clean-machine CI rows are green for Linux x64, macOS arm64 and Windows x64;
- installer, update, rollback and uninstall logs are retained;
- exact qualifying commit and artifact digests are recorded;
- release documentation is complete.

---

# 24. Non-goals for the first installer release

Defer unless required:

- background auto-update service;
- silent unattended system-wide updates;
- delta patches;
- package registry client;
- IDE installation;
- container orchestration;
- remote build service;
- compiler self-update while a build is running;
- transparent migration of arbitrary old caches;
- automatic trust of third-party providers.

---

# 25. Final deliverables

```text
Installer/distribution specification
Canonical installation manifest schema
Portable Linux/macOS/Windows archives
macOS signed and notarised package
Windows signed MSI
Linux package
stark doctor
offline native-build bundle
signed release index
update and rollback commands
uninstaller
clean-machine installer CI
offline build CI
release signing documentation
installation and troubleshooting documentation
```

---

# 26. Strategic outcome

The installer programme is successful when a new developer can obtain STARK as a normal language toolchain rather than as a repository.

The expected experience is:

```text
download
→ verify
→ install
→ stark doctor
→ stark new
→ stark run
```

The installation must preserve the architecture already proven by the compiler and provider programmes. Distribution must not reintroduce hidden paths, ambient caches, unpinned native code, silent substitutions, or parallel sources of authority.
