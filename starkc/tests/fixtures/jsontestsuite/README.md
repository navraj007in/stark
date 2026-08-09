# JSONTestSuite — vendored, unmodified

**Source:** https://github.com/nst/JSONTestSuite (Nicolas Seriot), `test_parsing/`.
**Licence:** MIT — see `LICENSE` in this directory. Copyright (c) 2016 Nicolas Seriot.
**Vendored:** 2026-08-07, for AS5 exit criterion 2.

## Why it is here

AS5's own corpus (`tests/as5_json_conformance.rs`) is **project-authored**. It is derived from
RFC 8259 and from the twelve constructs on which STARK's two previous parsers diverged, so it proves
the historical defects stay fixed — but it cannot prove that the implementation and its tests do not
share a misreading of the grammar. AS5 exists precisely because two in-tree implementations
disagreed about what JSON is; a corpus written by the same people who wrote the parser is
self-confirming evidence.

This is the independent oracle. It was fetched, run once, and vendored — the parser had never seen
it.

## Naming convention (the suite's own)

| Prefix | Meaning | Count |
| --- | --- | ---: |
| `y_` | **must** be accepted by a conforming parser | 95 |
| `n_` | **must** be rejected | 188 |
| `i_` | implementation-defined; either verdict conforms | 35 |

`tests/as5_jsontestsuite.rs` runs all three groups. The `y_`/`n_` groups are pass/fail. The `i_`
group's verdicts are **pinned in this repository**, one line per file, so that a change in what
STARK accepts shows up as a diff and a decision rather than as silence.

## Not modified

The files are byte-for-byte as fetched, including the deliberately malformed and non-UTF-8 ones.
Do not "fix" a file here — if a case looks wrong, the finding belongs in the test or in the parser.
