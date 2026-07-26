# dependency-reorder

- Intended rows: dependency declaration order determinism; canonical package identity.
- Normative citations: `STARKLANG/docs/spec/07-Modules-and-Packages.md`; `COMPILER-STATE.md` C6.2e deterministic identity note.
- Expected stdout: `alpha|beta`
- Expected exit status: `0`
- Metamorphic relation: `app-a` and `app-b` differ only in manifest dependency declaration order and must produce identical stdout.

