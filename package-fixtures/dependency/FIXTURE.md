# dependency

- Intended rows: path dependency resolution; public function import across package boundary.
- Normative citations: `STARKLANG/docs/spec/07-Modules-and-Packages.md`; `README.md` project workflow package dependency examples.
- Expected stdout: `dependency:ok`
- Expected exit status: `0`
- Metamorphic relation: relocating the dependency together with the app and updating only the relative path should preserve stdout.

