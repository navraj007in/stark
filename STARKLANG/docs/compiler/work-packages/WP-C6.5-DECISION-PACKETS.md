# WP-C6.5 — decision packets for DEV-113, DEV-114 and the CD-150 CE3

Three dispositions the owner asked for before any semantic fix is implemented (2026-07-27). Each
states root cause, normative requirement, choices, recommendation, compatibility impact,
implementation surface and required regression evidence.

All three touch the qualified path, so all three land inside the single consolidated batch — not
before it, and not one at a time.

---

# Packet 1 — DEV-113: dependency-package trap provenance

## Root cause

Two independent defects share one symptom, and the packet keeps them separate because they have
different fixes.

**113-A — file identity in a package build is the filesystem path.** `parse_package_graph` names
every `SourceFile` by the path it read (`src/parser.rs:173` for the entry, `:340` for submodules,
`:401` for the flattened graph). The name reaches the trap observation directly: MIR's `SourceInfo`
resolves through `program.files[...]`, whose names came from those constructors. So a workspace
compiled at `/tmp/a/app` and the same workspace at `/tmp/b — ünïcode/app` produce **different trap
provenance for the same program**.

**113-B — the HIR oracle attributes every trap to the root file.** `RuntimeError` carries a `span`
and no file (`src/interp.rs:35–58`), so `run_hir` names the trap with `front.file` — the *entry*
file — whatever file actually trapped. The interpreter *does* know: `self.file` is swapped per
callable (`src/interp.rs:1258`, DEV-069's per-item file resolution). The information exists at the
raise site and is discarded.

Consequence today: MIR reports `dep/src/main.stark`, the oracle reports `app/src/main.stark`, and a
dependency-trap case would fail as an engine divergence that is really a harness-visible defect.

## Normative requirement

- **PKG-IDENTITY-001** — a resolved package-instance token is relocation-stable and is "never an
  absolute checkout path".
- **TRAP-CATEGORY-001** and the C6.4 trap-provenance rows require exact `file:line:column`.
- **§15.2** — "no absolute path in semantic identity", "trap source names remain logical source
  paths".

## Choices

| | Option | Consequence |
| --- | --- | --- |
| A | **Logical file names in the package graph**: name each `SourceFile` `<package>/<path-relative-to-package-root>`, e.g. `dep/src/main.stark` | Provenance becomes relocation-stable everywhere it is observed. Diagnostics lose the absolute path a user may want in a terminal — mitigated by keeping the absolute path in a separate non-identity field |
| B | Keep absolute names; **normalise in the harness** | Cheapest, and wrong: it hides the leak from the corpus while leaving it in every user-visible diagnostic and in any future evidence that records provenance |
| C | Keep absolute names; **declare §15.2 unmet** and exclude dependency-trap cases permanently | Honest but concedes a normative requirement for a fixable defect |

For 113-B: attach the raising file to `RuntimeError` (the only real option; the alternative is to keep
the oracle unable to attribute a trap, which makes multi-file trap parity untestable).

## Recommendation

**A + attach the file to `RuntimeError`.** A is the only option that satisfies PKG-IDENTITY-001 rather
than concealing its violation, and 113-B has no defensible alternative — an oracle that cannot say
which file trapped cannot be the semantic authority for a multi-file program.

## Compatibility impact

- **Diagnostics change shape**: user-visible paths in a package build become package-relative. This is
  a *diagnostic text* change, not a semantic one; no Core rule constrains it.
- **Snapshots**: any `.snap` or expected-output file recording an absolute path must be re-pinned. The
  frozen `exec_snapshots` corpus is single-file and names its own case files, so it is unaffected —
  to be confirmed by running it, not assumed.
- **No MIR/ABI change**: `SourceFile.name` is not part of `mir.md`'s contract; `FileId` indices are
  unchanged.
- **`RuntimeError` gains a field** — an internal type, not a published contract.

## Implementation surface

```text
src/parser.rs:173,340,401   name SourceFiles <package>/<relative path>, carrying the absolute
                            path in a separate field for diagnostics only
src/package.rs              expose the package root so the relative path can be computed
src/interp.rs:35–58         RuntimeError gains `file: Option<Arc<SourceFile>>`
src/interp.rs (raise sites) attach `self.file` at the point the error is constructed
tests/support/differential.rs  run_hir uses the error's file when present, falling back to the root
tests/c6_package.rs         the two DEV-113 pins invert: they currently assert the defect
tests/c6-corpus/            add the dependency-trap case §15.1 requires
```

## Required regression evidence

1. The two `dev_113` pins in `c6_package.rs` **must fail** and be replaced by positive assertions.
2. A new corpus case: a trap raised inside a dependency, observed by all three engines, asserting
   provenance `dep/src/main.stark:<line>:<col>` — with the case relocated to a second directory and
   observing identically.
3. `mir_differential`, `exec_snapshots`, `conformance`, `multi_file_spans`, and every `native_*`
   suite, because diagnostic paths appear in their expectations.
4. Windows: the mixed-separator behaviour recorded at CD-160 disappears if names become logical —
   assert forward slashes on every platform.

---

# Packet 2 — DEV-114: deterministic canonical symbols for diamond package graphs

## Root cause

`src/parser.rs:200`:

```rust
for dep_name in pkg.dependencies.keys() {
```

`Package::dependencies` is a `HashMap<String, Dependency>` (`src/package.rs:437`). Rust seeds each
process's `HashMap` randomly, so **dependency visit order varies run to run**. Each dependency is
wrapped in a synthetic `Mod` named after the alias, and `parsed_packages` memoises a package's items
by name — so for a package reachable by two paths, whichever path is walked *first* determines the
module nesting its items are seen under. `model::leaf@[]` on one run, `logic::model::leaf@[]` on the
next, from identical inputs.

Two smaller order-dependencies ride along: synthetic span allocation
(`0x8000_0000 + ast.synthetic_spans.len()`) and the order dependencies are pushed into `root_items`.

## Normative requirement

- **TYPE-NOMINAL-001** — identity is "canonical package instance + module path + item name +
  normalized generic arguments". The **package instance** is the root of identity; the module path is
  the path *within* that package. A dependency edge is not a module-path segment.
- **PKG-IDENTITY-001** — "Aliases and re-exports preserve it", and the token is relocation-stable.
- **CD-108** — deterministic package identity.
- **§15.3** — no package-order leak.

Read together, these say the symbol for `model::leaf` must be the same however `model` is reached.
The current implementation contradicts all three, and does so nondeterministically.

## Choices

| | Option | Consequence |
| --- | --- | --- |
| A | **Sort the dependency iteration only** | Removes the nondeterminism; keeps path-dependent nesting, so a diamond still names one package two ways depending on which alias sorts first. Fixes the flakiness, not the identity violation |
| B | **Canonical prefix = the package's own name**, independent of the path taken | Matches TYPE-NOMINAL-001 directly: `model::leaf@[]` always. Requires the memo to wrap a package's items once, at its canonical name, with other reachers referring to it |
| C | Shortest-path nesting, ties broken by sorted alias | Deterministic and reorder-invariant, but still encodes reachability in identity — a re-export could change a symbol, which PKG-IDENTITY-001 forbids |

## Recommendation

**B, with A's sorted iteration as part of it.** B is what the specification says identity *is*; A
alone would leave a spec violation that is merely reproducible. Sorted iteration remains worth doing
because it removes hash-order dependence from span allocation and item order as well.

## Compatibility impact

- **Symbols change for multiply-reachable packages** — `logic::model::x` becomes `model::x`. Symbols
  are explicitly non-ABI (`mir.md` §2), and no released artifact depends on them.
- **The frozen reference workspace** (`native_c5_4_workspace`) freezes a symbol set in
  `EXPECTED-SYMBOLS.txt`. Its graph is a chain (`app → logic → model`), so B should leave it
  unchanged — **to be verified by running it, not asserted**; if it does change, that file is
  re-pinned in the same commit with the reason recorded.
- **No MIR shape or runtime-surface change.** `Instance.symbol` content changes; its type does not.

## Implementation surface

```text
src/parser.rs:200           iterate dependencies in sorted alias order
src/parser.rs:202–222       wrap a package's items under its OWN canonical name once; a second
                            reacher references the existing module rather than re-wrapping
src/parser.rs:203           synthetic span allocation becomes order-independent
tests/c6_package.rs         the DEV-114 pin inverts to an equality assertion
tests/fixtures/c5-native-workspace/EXPECTED-SYMBOLS.txt   re-pin only if the chain graph moves
```

## Required regression evidence

1. `diamond_package_symbols_are_nondeterministic_dev_114` **must fail**, replaced by: the same graph
   compiled in six separate processes yields one symbol set.
2. Dependency-declaration reorder yields **identical symbols** — the assertion CD-159 had to remove.
3. `native_c5_4_workspace` symbol freeze green, or re-pinned with the diff explained.
4. M08 and M09 metamorphic groups become buildable — this is the finding's real payoff (R-05).

---

# Packet 3 — CD-150 CE3: `invalid-exit-status` bundled with native entry signatures

## Root cause

Two halves of one feature, bundled at CD-150 because the backend increment that emits a non-`Unit`
entry must also emit the trap that entry can raise.

- **The trap has no category.** PROC-EXIT-001 requires an out-of-range status to trap; the nine
  `TrapCategory` values contain nothing for it. The oracle raises it as an uncategorised
  `RuntimeError`, MIR raises a loud `Internal` error (DEV-111's stopgap), and the comparator cannot
  normalise it.
- **Native refuses every non-`Unit` entry**: `Unsupported("the entry instance must return Unit to
  become Rust's fn main()")`, while PROC-MAIN-001 admits four entry types.

## Normative requirements, quoted

- **PROC-MAIN-001** — an executable target requires exactly one non-generic root `main` with no
  parameters returning `Unit`, `Int32`, `Result<Unit, String>` or `Result<Int32, String>`.
- **PROC-EXIT-001** — "Normal `Unit` and `Ok(Unit)` return status 0. `Int32` and `Ok(Int32)` must be
  in `0..=255` and return that status; an out-of-range value traps as `invalid-exit-status`.
  `Err(message)` writes `message` plus LF to stderr and returns status 1. A language trap returns
  status 101 after its specified diagnostic."
- **TRAP-CATEGORY-001** — a language trap is a failure required by a normative rule; trap identity is
  a WP-C6.0-frozen contract, which is what makes this a CE3.

## Precise semantics proposed

| Aspect | Proposal |
| --- | --- |
| Category name | `TrapCategory::InvalidExitStatus` |
| Runtime message | `invalid exit status` (lower case, matching the existing table's style) |
| Message class | `CategoryOnly` — the offending value is *not* part of the normative text, so it is not compared across engines |
| Raise point | after the entry returns and its value is converted, before the process exits |
| Status range | `0..=255` inclusive; `Ok(Int32)` unwraps first and applies the same range |
| Exit status of the trap | 101, as every language trap (TRAP-ABORT-001) |
| Provenance | the `main` signature's span — the entry is where the contract is violated, and there is no expression to blame |
| Destructors | none run; the trap aborts (DROP-ABORT-001) |
| `Err(message)` | writes `message` + LF to **stderr**, status 1, no trap |

## Allowed entry signatures after the change

All four of PROC-MAIN-001's, on all three engines: `Unit`, `Int32`, `Result<Unit, String>`,
`Result<Int32, String>`. The `Ok(Unit)` branch became writable at DEV-112.

## Choices

| | Option | Consequence |
| --- | --- | --- |
| A | **Add the tenth `TrapCategory` and implement the native entry together** | Satisfies both rules; one amendment, one implementation, one set of three-engine evidence. What CD-150 intended |
| B | Add the category now, implement native later | The category would exist with no engine able to exercise it end to end — a contract change with no evidence |
| C | Reword PROC-EXIT-001 so the range violation is not a trap | A specification change to avoid an implementation cost, and it would make the oracle's current behaviour wrong |

## Recommendation

**A.** It is also the only option under which the comparator's exhaustive `runtime_category` match —
which will not compile until the new category is mapped — becomes a help rather than an obstacle.

## Compatibility impact

- **`mir.md` amendment required** (trap identity is frozen): a new category, recorded as an amendment
  in the A-series, with `MIR_VERSION` unchanged and `MIR_RUNTIME_SURFACE` bumped only if the runtime
  gains an operation — it does not; `abort` already takes a category.
- **`stark-runtime` gains an enum variant and a message string.** Runtime version bump per its own
  policy; the generated-crate version check will then require the matching runtime, which
  `c63_closure_evidence` already tests.
- **Generated `fn main()` shape changes** for non-`Unit` entries: it computes the status, writes the
  `Err` message to stderr, and calls `std::process::exit`. Existing `Unit` entries are unchanged.
- **No language-surface change.** Nothing new is expressible; two already-legal signatures start
  working natively.

## Implementation surface

```text
src/mir/mod.rs                     TrapCategory::InvalidExitStatus
stark-runtime/src/trap.rs          matching variant + message()
STARKLANG/docs/compiler/mir.md     amendment record (trap identity)
src/interp.rs (main_result_to_status)   raise with_category instead of an uncategorised error
src/mir/interp.rs (entry_termination)   raise the trap instead of MirRunError::Internal
src/backend/generated_rust/emit_program.rs   non-Unit entry: status computation, stderr write,
                                             process::exit, and the range trap
tests/support/differential.rs      ALL_CATEGORIES, runtime_category (exhaustive match forces this)
tests/c65_entry_exit_contract.rs   the two escalation pins invert to three-engine assertions
tests/c6-corpus/                   the four entry shapes become three-engine corpus cases
```

## Required regression evidence

1. `an_out_of_range_exit_status_is_refused_by_both_engines_pending_a_trap_category` and
   `native_refuses_every_non_unit_entry_signature` **must fail**, replaced by three-engine agreement.
2. Corpus: `entry_exit__02/03/04/05` move from `required_engines = ["hir","mir"]` to all three, and
   `main -> Int32 { 300 }` joins as a trap case with category `InvalidExitStatus`.
3. A mutation control for the new category (MU03's shape) and for `stderr_bytes`, which the `Err`
   entry case makes observable for the first time — this closes part of R-03.
4. `c63_closure_evidence` for the runtime-version interaction.
5. Trap-category coverage becomes **10 of 10** admitted categories once R-01's work lands alongside.

---

## Sequencing note

Packets 1 and 2 are independent of each other. Packet 3 depends on neither, but its corpus cases
depend on **Packet 1** if any of them is a package case — they are single-file, so they are not
blocked.

**R-04/R-05's metamorphic floor depends on Packet 2**: M08 and M09 cannot be built while a diamond
graph's symbols are nondeterministic.
