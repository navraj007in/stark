# stark-csv

Strict deterministic CSV derived from RFC 4180 with explicitly frozen v0.1 edge semantics.

## Current status: PARTIAL — IMPLEMENTATION COMPLETE, WAITING_C6.3_NATIVE_EVIDENCE

The pure-STARK parser and writer are complete, tested (107 `stark test` cases, all passing),
proven usable across files and across packages, and cross-checked against Python's standard
`csv` module over 1,000 deterministic cases (`tools/csv_oracle.py`). Real native execution is
unavailable: the compiler's generated-Rust backend does not yet represent nested
`Vec<Vec<String>>` (this package's actual data model) or `str::bytes()`. See `BLOCKERS.md` for
the exact failure and `EVIDENCE.md` for the full requirement ledger.

## Dialect

Comma-delimited, double-quote-quoted, doubled-quote-escaped. Input accepts LF or CRLF record
separators; output always uses CRLF, with no trailing separator after the final record. A quote
opens a quoted field only as the field's first byte; anything else after unquoted content, or any
byte other than comma/LF/CRLF/EOF immediately after a closing quote, is rejected.

## Quotes and doubled quotes

Inside a quoted field, `""` decodes to one `"`; comma, CR, LF and CRLF are ordinary content, not
separators. The writer quotes a field if and only if it contains a comma, quote, CR, or LF — never
merely because a field is empty, begins/ends with whitespace, contains a tab, contains Unicode, or
contains NUL. When a field is quoted, every quote inside it is doubled.

## Unicode

`String`/`str` content is exact UTF-8; parsing and writing never normalize, transcode, or replace
any Unicode content, including combining marks and supplementary-plane scalars.

## Empty input versus blank line

An entirely empty document parses to zero records (`"" -> []`). A blank line (`"\n"` or
`"\r\n"` alone) parses to one record with one empty field (`[[""]]`) — these are different
things. A trailing separator terminates the record it follows but does not add another record.

One representational exception this package cannot avoid: a record consisting of exactly one
empty field writes to the same empty text as a zero-field record, since CSV without quoting-for-
disambiguation has no other way to represent it, and this package's writer never quotes solely
for emptiness (see "Quotes and doubled quotes" above, and the note in `tools/csv_oracle.py`'s
docstring — this is also where this package's canonical output differs from Python's `csv`
module, which quotes that one case to work around the same ambiguity).

## Unequal widths

Records in one document may have different field counts; this is accepted, not an error. This
package does not enforce a uniform record width.

## Limits and errors

Every parse/write operation takes explicit `CsvLimits` (`default_limits()` or `limits(...)`);
every limit that would be exceeded returns a typed `CsvError` carrying byte offsets and
record/field indexes — never a trap, never a partial result. All limit and length arithmetic is
overflow-guarded.

## Free-function API

Public entry points (`parse`, `parse_with_limits`, `write`, `write_with_limits`,
`default_limits`, `limits`) are free functions, not associated functions (`Csv::parse(...)`):
the repository currently has a cross-file associated-function resolution gap — a
`Type::function()` path only resolves within the file that declares the `impl` block — so an
associated-function API would not resolve from this package's own `tests.stark`, let alone from
another package. `CsvLimits`/`CsvError` fields and variants are constructed and inspected
directly (both are plain data types with no invariants an associated constructor would need to
protect beyond what `limits(...)`'s explicit parameters already state).

## What this package does not do

No custom delimiters, dialect detection, TSV/semicolon variants, comments, `sep=` handling,
whitespace trimming/normalization, backslash or single-quote escaping, numeric/boolean/null
conversion, headers/maps/schema inference/typed rows, record-width enforcement, streaming/lazy
APIs, file I/O, native code, or formula-injection sanitization.

## Testing

- `src/tests.stark` — 107 tests covering §19 (parser) and §20 (writer) requirements, run with
  `stark test`.
- `fixtures/` — an external path-dependency consumer proving cross-package use with only the
  public API.
- `tools/csv_oracle.py` — a Python host-oracle differential over 1,000 deterministic cases
  against Python's `csv` module (see `EVIDENCE.md` for the run).
