#!/usr/bin/env python3
"""C10-A2 — resolve each normative rule against the TEST TREE, not against the inventory.

WHY THIS EXISTS. C10-A1 measured the inventory's CITATION state and found three buckets that look
like evidence states and are not:

    PRECISE    36    positive AND negative evidence at test-function precision
    AGGREGATE  85    cited only through a file or the aggregate runner (DEV-017)
    ABSENT     42    the inventory's evidence column records `none`

A1-F3 proved `ABSENT` does not mean untested: `EXT-ISOLATION-001` records `none; none` while
`c91_extension_isolation.rs` runs nine tests in CI on every push. The inventory froze a snapshot in
July 2026 and was never maintained as the tree grew. **So the buckets may not be copied into the
dashboard.** Each row is resolved against the tree, or it says UNRESOLVED.

THE KEY IS THE NORMATIVE RULE ID, NOT THE SYMBOL. This is `as8-control-census.py`'s method,
generalised from 11 shared-fate authorities to all 168 granular rules. `c61f_structural_copy.rs`
never names `copy_eligible_types`; it cites `OWN-COPY-001`. A symbol search finds tests that touch
the IMPLEMENTATION and misses every test that pins the RULE — and only the second kind can act as a
control, because a control must be able to contradict the implementation.

THE SAME STATED LIMITATION, inherited deliberately. This finds tests that CITE a rule ID. A test
that pins a rule without naming it is invisible here, and the spec-fixture corpus carries no rule
IDs at all. **Absence of a hit is NOT proof of absence of a control** — it is a prompt to look, and
the honest output for a rule with no hit is UNRESOLVED, never `none`.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
INVENTORY = ROOT / "STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md"
C211 = ROOT / "STARKLANG/conformance/core-v1-c2.11-evidence.toml"
LEGACY_MAP = ROOT / "STARKLANG/conformance/core-v1-rule-id-map.toml"

ID = re.compile(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+-\d{3}")
ROW = re.compile(r"^\|\s*(" + ID.pattern + r")\s*\|(.*)$")

#: Where a citation can live. `src` is included because many controls are `#[cfg(test)]` units --
#: `resolve.rs`'s own tests killed AS8-MUT-038 when no integration suite could.
SEARCH_ROOTS = ("starkc/tests", "starkc/src", "STARKLANG/tests")
SEARCH_EXT = (".rs", ".toml", ".stark", ".md")


def granular_ids() -> list[str]:
    out, seen = [], set()
    for line in INVENTORY.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n"):
        m = ROW.match(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    return out


def corpus() -> dict[pathlib.Path, str]:
    files = {}
    for base in SEARCH_ROOTS:
        for root, _, names in os.walk(ROOT / base):
            for n in names:
                if n.endswith(SEARCH_EXT):
                    p = pathlib.Path(root) / n
                    try:
                        files[p] = p.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        pass
    return files


def enclosing_fn(src: str, index: int) -> str | None:
    """The `fn` a citation belongs to — the precision the dashboard wants.

    Two cases, and conflating them produces names that look right and are wrong by one:

    * a citation inside a `///` DOC COMMENT documents the function BELOW it, so scan forward;
    * any other citation sits inside a body, so scan backward for the enclosing `fn`.

    The first version scanned backward unconditionally. Applied to a doc comment above a `#[test]`,
    that attributed every citation to the PRECEDING function — so five attributions in
    `c91_extension_isolation.rs` came back shifted by one, naming a helper and four wrong tests.
    Every name was a real function in the right file, which is exactly why it would have survived
    review: a dashboard citing the wrong test is worse than one citing none, because it looks
    checked.

    A citation in a file or module header still has no owning function and is reported as
    file-level — the AGGREGATE precision A1 says must never be promoted to PRECISE.
    """
    line_start = src.rfind("\n", 0, index) + 1
    line = src[line_start : src.find("\n", index) if src.find("\n", index) != -1 else len(src)]
    if line.lstrip().startswith("//"):
        # A comment. If a `fn` follows before the next blank-line-separated item, it owns this.
        tail = src[index:]
        m = re.search(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(<]", tail)
        if m:
            # Only claim it when nothing but attributes/comments intervene -- otherwise a module
            # header would capture the first function in the file.
            between = tail[: m.start()]
            if not re.search(r"[;}]\s*$", between.strip()) and between.count("\n") <= 6:
                return m.group(1)
        return None
    head = src[:index]
    matches = list(re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(<]", head))
    return matches[-1].group(1) if matches else None


def in_test_context(path: pathlib.Path, src: str, index: int) -> bool:
    """Is this citation inside TEST code, rather than inside the implementation?

    **This distinction is the whole point, and the first version of this script did not make it.**
    It reported `interp.rs::eval_expr` and `lower.rs::lower_expr_to_operand` as evidence for
    `TYPE-PRIM-001`. Those are the IMPLEMENTATION naming the rule it implements. An implementation
    cannot be its own control — that is `AS8-R4` exactly (`copy_canon_matrix` is a transcription of
    `core_method_signature`, so it passes just as happily if the rule is wrong), arriving in a new
    disguise.

    A file under `tests/` is test code throughout. Inside `src/`, only a `#[cfg(test)]` module
    counts, and the test is whether the citation sits after the LAST `#[cfg(test)]` marker —
    an approximation, because test modules are conventionally last in a file, and one that can
    only ever be WRONG IN THE CONSERVATIVE DIRECTION for a file whose test module is not last.
    """
    posix = str(path).replace(os.sep, "/")
    if "/tests/" in posix:
        return True
    if path.suffix != ".rs":
        return False
    marker = src.rfind("#[cfg(test)]")
    return marker != -1 and index > marker


def classify_control(path: pathlib.Path, src: str) -> str:
    """Can this file's evidence DISAGREE with the implementation?

    `as8-control-census.py`'s distinction, and the reason it matters: a differential suite compares
    engines to each other and inherits every front-end decision, so it cannot contradict a shared
    authority however many engines agree (EI0's frozen rule). A front-end test asserting a
    diagnostic can.
    """
    name = path.name
    posix = str(path).replace(os.sep, "/")
    if path.suffix in (".stark", ".toml"):
        return "corpus"
    if path.suffix == ".md":
        return "doc"
    if "differential" in name or "three_engine" in name:
        return "ENGINE(correlated)"
    if "spec-fixtures" in posix:
        return "SPEC"
    if "/src/" in posix:
        return "UNIT(cfg-test)"
    return "INTEGRATION"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--unresolved-only", action="store_true")
    args = ap.parse_args()

    ids = granular_ids()
    files = corpus()
    precise = set(re.findall(r'^\s*id\s*=\s*"([^"]+)"', C211.read_text(encoding="utf-8"), re.M))

    rows = {}
    for rid in ids:
        hits = []
        for p, src in files.items():
            # A rule ID is a whole token; `NUM-INT-001` must not match `NUM-INT-0012`.
            for m in re.finditer(re.escape(rid) + r"(?![0-9A-Za-z-])", src):
                hits.append({
                    "file": str(p.relative_to(ROOT)),
                    "fn": enclosing_fn(src, m.start()),
                    "class": classify_control(p, src),
                    # An implementation citing the rule it implements is NOT evidence for it.
                    "test": in_test_context(p, src, m.start()),
                })
        # Dedupe by (file, fn) — one citation per site is what the dashboard needs.
        seen, uniq = set(), []
        for h in hits:
            k = (h["file"], h["fn"])
            if k not in seen:
                seen.add(k)
                uniq.append(h)
        # Only TEST-context citations with a named RUST function can be function-precision
        # evidence. Three exclusions, each learned by reading the names rather than the counts:
        #
        #   doc      a citation in a .md file is prose, not a control
        #   corpus   a .stark case has an `fn main`, and attributing a rule to "main" is precision
        #            theatre -- corpus evidence is real, and it belongs in the corpus bucket
        #   impl     an implementation citing the rule it implements cannot be its own control
        with_fn = [
            h for h in uniq
            if h["fn"] and h["test"] and h["class"] not in ("doc", "corpus")
        ]
        impl_only = [h for h in uniq if not h["test"] and h["class"] not in ("doc", "corpus")]
        rows[rid] = {
            "in_c211": rid in precise,
            "citations": uniq,
            "function_level": with_fn,
            "implementation_only": [h["file"] for h in impl_only],
            "state": (
                "PRECISE-C211" if rid in precise
                else "RESOLVED-BY-TREE" if with_fn
                else "CORPUS-OR-FILE-LEVEL"
                if [h for h in uniq if h["test"] and h["class"] != "doc"]
                or [h for h in uniq if h["class"] == "corpus"]
                else "IMPLEMENTATION-ONLY" if impl_only
                else "UNRESOLVED"
            ),
        }

    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        print()
        return 0

    counts = {}
    for r in rows.values():
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    print(f"POPULATION: {len(ids)} granular rules\n")
    for k in ("PRECISE-C211", "RESOLVED-BY-TREE", "CORPUS-OR-FILE-LEVEL",
              "IMPLEMENTATION-ONLY", "UNRESOLVED"):
        print(f"  {k:<18} {counts.get(k,0):>4}")
    print()
    print("RESOLVED-BY-TREE means a test FUNCTION cites the rule id and the inventory did not say so.")
    print("UNRESOLVED means NO citation was found. Per this script's stated limitation that is a")
    print("prompt to look by hand -- it is NOT evidence that no control exists.")
    print()
    for rid, r in rows.items():
        if args.unresolved_only and r["state"] != "UNRESOLVED":
            continue
        if r["state"] == "RESOLVED-BY-TREE":
            sites = ", ".join(f"{h['file'].split('/')[-1]}::{h['fn']}" for h in r["function_level"][:3])
            print(f"  {rid:<26} {r['state']:<18} {sites}")
        elif args.unresolved_only:
            print(f"  {rid:<26} {r['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
