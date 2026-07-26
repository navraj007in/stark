"""The one Python reader for `starkc/target-matrix.json` (WP-C6.4).

Every Python consumer — packaging, qualification, comparison — asks this module instead of
carrying its own triple table. §8.2 forbids duplicating triple matching across CLI, builder,
backend, tests and scripts; the Rust side is centralised in `src/target.rs`, and this is how the
Python side joins it rather than paralleling it.

The JSON is pinned to `src/target.rs` in both directions by
`tests/c64_platform_matrix.rs::target_matrix_json_matches_the_compiler`, so a lookup here answers
with the compiler's own classification. Nothing in this module infers anything from the *shape* of
a triple: `classify` is an exact-match lookup and `require` raises on an unknown triple. A
substring test like `"windows" in target` is exactly what this module exists to replace — it
misclassifies any unknown triple containing the word and silently accepts targets the compiler
will not build for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MATRIX_PATH = Path(__file__).resolve().parent.parent / "target-matrix.json"
SCHEMA = "stark-target-matrix-1"


class UnknownTarget(Exception):
    """Raised for a triple the compiler does not name. Never a fallback, always an error."""


@dataclass(frozen=True)
class TargetEntry:
    triple: str
    tier: str
    layout_contract: str
    executable_suffix: str
    pointer_width: int
    archive: str
    installers: tuple[str, ...]

    @property
    def is_tier1(self) -> bool:
        return self.tier == "tier-1"


def _load() -> dict[str, TargetEntry]:
    data = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise UnknownTarget(
            f"{MATRIX_PATH} declares schema {data.get('schema')!r}, expected {SCHEMA!r}"
        )
    entries = {}
    for raw in data["targets"]:
        entry = TargetEntry(
            triple=raw["triple"],
            tier=raw["tier"],
            layout_contract=raw["layout_contract"],
            executable_suffix=raw["executable_suffix"],
            pointer_width=int(raw["pointer_width"]),
            archive=raw["archive"],
            installers=tuple(raw["installers"]),
        )
        entries[entry.triple] = entry
    if not entries:
        raise UnknownTarget(f"{MATRIX_PATH} declares no targets")
    return entries


_ENTRIES = _load()


def classify(triple: str) -> TargetEntry | None:
    """Exact match, or `None`. Never a prefix or substring match."""
    return _ENTRIES.get(triple)


def require(triple: str) -> TargetEntry:
    """Exact match, or raise. The packaging path uses this: producing an archive for a target the
    compiler does not name would ship an artifact nothing can qualify."""
    entry = _ENTRIES.get(triple)
    if entry is None:
        raise UnknownTarget(
            f"`{triple}` is not a target STARK names. Known targets: "
            f"{', '.join(sorted(_ENTRIES))}"
        )
    return entry


def all_targets() -> tuple[TargetEntry, ...]:
    return tuple(_ENTRIES.values())


def tier1_triples() -> tuple[str, ...]:
    return tuple(e.triple for e in _ENTRIES.values() if e.is_tier1)


def tier_of(triple: str) -> str:
    entry = _ENTRIES.get(triple)
    return entry.tier if entry else "unsupported"
