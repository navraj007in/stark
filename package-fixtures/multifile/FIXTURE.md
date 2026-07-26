# multifile

- Intended rows: package module resolution; multi-file package entry.
- Normative citations: `STARKLANG/docs/spec/07-Modules-and-Packages.md`; `STARKLANG/docs/spec/02-Syntax-Grammar.md`.
- Expected stdout: `multifile:42`
- Expected exit status: `0`
- Metamorphic relation: moving `answer()` between sibling module files without changing its public symbol should preserve stdout.

