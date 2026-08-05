---
name: stark-doc-sweep
description: Run before declaring any multi-document edit done in this repo — specs, ADRs, roadmaps, gate reports, READMEs, the website, or the agent context files. Catches the drift external reviewers reliably find: stale counts, tables contradicting prose, pointers to sections that do not exist, downstream docs running ahead of canonical sources, and retired decisions still quoted as current policy. Contains this repo's actual grep commands, not generic advice.
tools: Read, Grep, Glob, Bash, Edit
---

# STARK cross-document sweep

Individually sound edits make a bundle incoherent in aggregate. This repo has a lot of
cross-citing documents — normative spec, generated spec, gate reports, two live gate tracks, a
consolidated roadmap, six archived roadmaps, per-package docs, two agent context files and a
website — so the drift is not hypothetical. Every item below has actually happened here.

**This is a mandatory final step, not polish.** If it finds something, fix it in the same turn.

---

## 1. Counts and enumerations

Adding one item to a list orphans every sentence that counted the old list.

```bash
# package count — count MANIFESTS, not directories. `packages/stark-file/` is a provider crate
# with no starkpkg.json, so counting directories yields 25 and every doc says 24.
ls -d packages/stark-*/ | grep -v -- '-consumer' \
  | while read -r d; do [ -f "$d/starkpkg.json" ] && echo x; done | wc -l
grep -rn '25 first-party\|25 packages\|Twenty-five' README.md CLAUDE.md AGENTS.md website/src/content.ts

# spec fixture count
grep -c '^\[' STARKLANG/tests/spec-fixtures/manifest.toml
grep -rn '[0-9]\{3\}[- ]fixture\|[0-9]\{3\} extracted' README.md CLAUDE.md AGENTS.md

# any other stated count near a list you touched
grep -rnE '\b(two|three|four|five|six|seven|eight|nine|ten|[0-9]+) (gates?|commands?|packages?|capabilities|providers?|decisions?|steps?|phases)\b' \
  README.md ROADMAP.md CLAUDE.md AGENTS.md starkc/README.md
```

Known live counts to keep honest: 25 packages, 22 consumers, 6 provider crates, 8 capability names,
116 spec fixtures, 87 audited stdlib methods / 59 verified, four engine configurations.

## 2. Table vs. prose

When prose changes, check the nearby table row, and vice versa. Both must agree — implementers who
follow the table otherwise diverge from those who follow the paragraph, and whoever ships first
picks the winner arbitrarily.

Specifically re-read, whenever the compiler's position moves: README's gate tables **and** the
"Where the compiler actually is" prose above them; `starkc/README.md`'s module table; the status
lists in `website/src/content.ts`.

## 3. Cross-document pointers must resolve

Every `see X §N` needs X to exist and §N to say that.

```bash
# Do referenced files exist? Resolve each link RELATIVE TO THE FILE CONTAINING IT —
# starkc/README.md's links are relative to starkc/, so a repo-root check reports
# seven false positives and gets ignored thereafter.
for f in README.md ROADMAP.md CLAUDE.md AGENTS.md starkc/README.md; do
  d=$(dirname "$f")
  grep -ohE '\]\([^)#]+\.md[^)]*\)' "$f" | sed 's/](\(.*\))/\1/' | sed 's/#.*//' | sort -u \
    | while read -r t; do [ -e "$d/$t" ] || echo "DANGLING in $f -> $t"; done
done

# do referenced sections exist? (check by hand — open the target and read the heading)
grep -rn '§[0-9]' <files you edited>
```

**This bites here.** Gate 7's memo cited "§13 of the proposal" for the external-developer protocol;
§13 is that proposal's work-package list, and the protocol is §8 with the bar in §11. The wrong
pointer survived from 2026-07-16 until someone opened the target.

Also confirm markdown anchors: a link to `#installing-the-toolchain` requires a heading whose text
slugs to exactly that.

## 4. Canonical source first, then downstream

A new command, field or concept lands in the canonical document *before* anything downstream
describes it. In this repo the canonical sources are:

| subject | canonical |
| --- | --- |
| language semantics | `STARKLANG/docs/spec/` source documents 00–07 + `CORE-V1-*.md` |
| compiler position | `COMPILER-STATE.md` (repo root) |
| forward plan | `ROADMAP.md` (repo root) |
| compiler governance | `STARKLANG/docs/compiler/COMPILER-CHARTER.md` |

README, `starkc/README.md`, the website, `CLAUDE.md` and `AGENTS.md` are all **downstream**. They
summarise and may lag; they must never describe behaviour the canonical source has not scoped.

Editing anything under `docs/spec/` means regenerating in the same change:

```bash
python3 STARKLANG/tools/build-core-spec.py          # never edit STARK-Core-v1.md directly
STARKLANG/tools/extract-spec-examples.sh            # fails if fixtures diverge from the manifest
```

## 5. Retired decisions must not read as current

This repo records decisions and supersedes them rather than deleting them, so a retired policy
keeps existing in text. Check that no *live* document still asserts one.

```bash
grep -rn "RETAIN AS RESEARCH" --include="*.md" . | grep -v node_modules | grep -v docs/archive
# every hit must be either (a) inside the preserved memo or dated evidence,
# or (b) phrased as superseded. None may read as current policy.
```

Same test for any other retired label you find: does the sentence tell a first-time reader this is
*current*? Historical records (`gate*-decision.md`, `C0-exit-report.md`, `COMPILER-STATE.md`
entries, spikes) are preserved as written and must **not** be rewritten — banner them instead.

## 6. Field/check symmetry

If a document defines a check that reads a field, the canonical type or schema must expose that
field. Otherwise implementers invent ad-hoc client-side tables that drift from the backend.

---

## Report honestly

Say which checks you ran and what they found. "Swept counts, pointers and the retired-policy grep;
fixed two stale counts" is useful. "Updated the docs" is not.
