# Package Fixture Designs

These fixtures are owner-side package-graph examples for a future C6.5-8 package breadth pass.
They are intentionally not wired into the replay harness, manifest lock, generator, or comparator
infrastructure.

Each fixture includes:

- package manifests;
- STARK source files;
- `expected.stdout`;
- `expected.status`;
- `FIXTURE.md` with intended matrix rows, normative citations, and metamorphic relation.

Claude retains ownership of corpus integration, manifest-schema changes, package replay,
lock/version updates, and C6.5 evidence.

## Smoke Results At Creation

Validated at repository head `0c59c8044d8e1301c85f127aba619f5075b758bb` with
`starkc/target/debug/stark`.

| Fixture | Command shape | Result |
|---|---|---|
| `multifile/app` | `stark check`; `stark run` | `pkg-multifile-app: OK`; `multifile:42` |
| `dependency/app` | `stark check`; `stark run` | `pkg-dependency-app: OK`; `dependency:ok` |
| `reexport/app` | `stark check`; `stark run` | `pkg-reexport-app: OK`; `reexport:core` |
| `dependency-reorder/app-a` | `stark check`; `stark run` | `pkg-reorder-app-a: OK`; `alpha|beta` |
| `dependency-reorder/app-b` | `stark check`; `stark run` | `pkg-reorder-app-b: OK`; `alpha|beta` |
| `relocated/app` | `stark check`; `stark run` | `pkg-relocated-app: OK`; `relocated:stable` |
| `offline/app` | `stark check --locked --offline`; `stark run` | `pkg-offline-app: OK`; `offline:path` |
| `unicode-path/πακέτο` | `stark check`; `stark run` | `pkg-unicode-path: OK`; `unicode:path` |
| `spaced-path/package with spaces` | `stark check`; `stark run` | `pkg-spaced-path: OK`; `spaced:path` |
