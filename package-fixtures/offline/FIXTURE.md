# offline

- Intended rows: offline path dependency; lock-respecting package check.
- Normative citations: `README.md` project CLI `--offline` and `--locked` modes; `STARKLANG/docs/spec/07-Modules-and-Packages.md`.
- Expected stdout: `offline:path`
- Expected exit status: `0`
- Metamorphic relation: running with online access available or unavailable should preserve stdout because all dependencies are local path dependencies.

