# reexport

- Intended rows: dependency chain; facade package import; public API forwarding.
- Normative citations: `STARKLANG/docs/spec/07-Modules-and-Packages.md`.
- Expected stdout: `reexport:core`
- Expected exit status: `0`
- Metamorphic relation: importing from the facade or directly from the core package should produce the same stdout when both expose the same value.

