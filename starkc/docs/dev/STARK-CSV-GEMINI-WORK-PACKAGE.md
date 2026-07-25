# STARK `stark-csv` v0.1 — Gemini Implementation Work Package

**Package:** `stark-csv`  
**Version:** `0.1.0`  
**Implementation:** Pure STARK Core v1  
**Native code/provider:** Prohibited  
**Compiler/runtime/spec changes:** Prohibited  
**Format:** Strict deterministic CSV derived from RFC 4180  
**Repository baseline inspected:** `8de4d41159083349133dc0696784a725a99aaacb`  
**Status:** Frozen implementation specification  
**Acceptance:** Independent review required

---

## 0. Why this is the next delegated package

Do not implement `std-json` in this work package.

`std-json` combines recursive values, Unicode escapes, number grammar, depth/size limits,
path-aware diagnostics, deterministic output and typed conversion. That is too broad for the
next low-supervision package after the Base64 and `std-time` evidence failures.

`stark-csv` is bounded, pure STARK, useful for data processing and suitable for a finite-state
implementation. It also tests `String`, nested `Vec`, UTF-8 preservation, package resolution and
evidence discipline without requiring native-provider execution.

---

## 1. Instruction to Gemini

Implement only this package.

Every **MUST**, **MUST NOT**, **SHALL**, **SHALL NOT**, **EXACTLY** and **ONLY** is binding.

You must not:

- redesign the API;
- omit required tests and compensate with prose;
- claim commands ran without reproducible evidence;
- claim native execution from `stark check` or an alias-only `stark build`;
- move tests into `lib.stark` to hide a cross-file resolver failure;
- use a same-package test as proof of cross-package usability;
- modify compiler/runtime/package-manager code;
- invent manifest fields;
- mark the package complete merely because the parser works on positive examples.

Your tasks:

1. inspect the exact current repository head;
2. run the readiness probes in Section 5;
3. classify readiness honestly;
4. create only authorised files;
5. implement the frozen API;
6. implement the four-state parser;
7. implement deterministic serialization;
8. commit every required test;
9. commit and run the host-oracle script;
10. prove cross-file and cross-package use;
11. record exact evidence and stop at any upstream blocker.

---

## 2. Governing repository contracts

Read current versions of:

- `COMPILER-STATE.md`;
- `STARKLANG/docs/spec/02-Syntax-Grammar.md`;
- `STARKLANG/docs/spec/03-Type-System.md`;
- `STARKLANG/docs/spec/06-Standard-Library.md`;
- `STARKLANG/docs/spec/07-Modules-and-Packages.md`;
- `STARKLANG/docs/proposals/CORE_PACKAGES_ECOSYSTEM_ROADMAP.md`;
- `stark-base64/`;
- `stark-time/`;
- current path-dependency fixtures;
- current `stark test` discovery;
- current C6.3 state for `String`, `Vec`, slices and native output;
- the known cross-file associated-function resolver limitation.

A newer owner-approved contract wins over this document. On conflict, stop and report the exact
authority-bearing source and commit.

---

## 3. Objective

Provide:

- parsing from UTF-8 `&str`;
- writing to UTF-8 `String`;
- comma delimiter;
- double-quote quoting;
- doubled-quote escaping;
- embedded commas and line breaks inside quoted fields;
- LF and CRLF input record separators;
- canonical CRLF output;
- strict malformed-input rejection;
- explicit resource limits;
- deterministic minimal quoting;
- exact UTF-8 preservation;
- typed errors with byte offsets and record/field indexes;
- no native code or privileged hooks.

---

## 4. Scope

### Included

- empty input, fields and records;
- quoted/unquoted fields;
- escaped quotes;
- LF and CRLF outside quotes;
- CR, LF and CRLF inside quotes;
- Unicode and NUL content in valid STARK strings;
- unequal record widths;
- parser/writer limits;
- separate test module;
- external consumer;
- positive host-oracle differential;
- honest partial status when native C6.3 execution is unavailable.

### Excluded

Do not implement:

- custom delimiters or dialect detection;
- TSV/semicolon CSV;
- comments or Excel `sep=`;
- trimming or whitespace normalization;
- backslash/single-quote escaping;
- numeric/boolean/null conversion;
- headers, maps, schema inference or typed rows;
- record-width enforcement;
- streaming/lazy APIs;
- file I/O;
- native code, SIMD or parallelism;
- BOM removal or encoding detection;
- invalid UTF-8 replacement;
- formula-injection sanitization;
- macros, reflection or code generation;
- compiler/package/runtime changes.

---

## 5. Readiness audit

Confirm:

1. `String::new`, `String::from`, `push`, `push_str`, `as_str` work.
2. `str::bytes()` returns UTF-8 bytes.
3. `substring(start,end)` works at valid scalar boundaries.
4. `Vec<String>` and `Vec<Vec<String>>` work.
5. Returning `Vec<Vec<String>>` from a public free function works.
6. Passing `&Vec<Vec<String>>` to a public free function works.
7. Public free functions resolve from another `.stark` file.
8. Public free functions resolve from an external package.
9. `Result<Vec<Vec<String>>, CsvError>` works.
10. `src/tests.stark` is discovered.
11. approved path dependencies can create a consumer.
12. determine whether nested `String`/`Vec` executes natively.

### Mandatory probes

#### Probe A — nested return

```stark
pub fn rows() -> Vec<Vec<String>> {
    let mut row = Vec::new();
    row.push(String::from("a"));
    let mut rows = Vec::new();
    rows.push(row);
    rows
}
```

#### Probe B — cross-file free function

`src/lib.stark`:

```stark
pub fn answer() -> UInt64 { 42u64 }
```

`src/tests.stark`:

```stark
use super::answer;

fn test_answer_cross_file() {
    if answer() != 42u64 { panic("cross-file free function failed"); }
}
```

#### Probe C — cross-package free function

Create an ordinary path-dependency consumer and call a public free function. A same-package test
does not count.

#### Probe D — nested writer argument

```stark
pub fn count_rows(rows: &Vec<Vec<String>>) -> UInt64 { rows.len() }
```

### Readiness classifications

Use one:

- `READY_FULL`
- `READY_INTERPRETER`
- `BLOCKED_CROSS_FILE`
- `BLOCKED_CROSS_PACKAGE`
- `BLOCKED_RUNTIME_VALUES`
- `BLOCKED_PACKAGE_TOOLING`

`READY_INTERPRETER` means pure, cross-file and cross-package use work but real native execution is
blocked by C6.3/package tooling.

---

## 6. Package layout and scope

```text
stark-csv/
├── starkpkg.json
├── stark.lock
├── README.md
├── EVIDENCE.md
├── BLOCKERS.md
├── src/
│   ├── lib.stark
│   └── tests.stark
├── tools/
│   └── csv_oracle.py
├── fixtures/
│   └── consumer/
└── docs/
```

`src/tests.stark` is mandatory. Do not duplicate tests in `lib.stark`.

Allowed changes: `stark-csv/**` only, except a separately owner-assigned integration fixture.

Prohibited: `starkc/**`, `stark-runtime/**`, specs, compiler work packages, root workspace,
package-manager schemas, existing packages, root CI and `.gitignore`.

---

## 7. Manifest

Use current approved syntax equivalent to:

```json
{
  "name": "stark-csv",
  "version": "0.1.0",
  "entry": "src/lib.stark",
  "dependencies": {}
}
```

No native declarations, build scripts or invented fields.

---

## 8. Frozen public API

```stark
pub enum CsvError {
    InputTooLarge(UInt64, UInt64),
    TooManyRecords(UInt64, UInt64),
    TooManyFields(UInt64, UInt64, UInt64),
    TooManyTotalFields(UInt64, UInt64),
    FieldTooLarge(UInt64, UInt64, UInt64, UInt64),
    QuoteInUnquotedField(UInt64, UInt64, UInt64),
    UnexpectedAfterClosingQuote(UInt64, UInt64, UInt64, UInt8),
    BareCarriageReturn(UInt64, UInt64, UInt64),
    UnterminatedQuotedField(UInt64, UInt64, UInt64),
    OutputTooLarge(UInt64, UInt64)
}

pub struct CsvLimits {
    max_input_bytes: UInt64,
    max_records: UInt64,
    max_fields_per_record: UInt64,
    max_total_fields: UInt64,
    max_field_bytes: UInt64,
    max_output_bytes: UInt64
}

pub fn default_limits() -> CsvLimits;

pub fn limits(
    max_input_bytes: UInt64,
    max_records: UInt64,
    max_fields_per_record: UInt64,
    max_total_fields: UInt64,
    max_field_bytes: UInt64,
    max_output_bytes: UInt64
) -> CsvLimits;

pub fn parse(input: &str) -> Result<Vec<Vec<String>>, CsvError>;

pub fn parse_with_limits(
    input: &str,
    limits: &CsvLimits
) -> Result<Vec<Vec<String>>, CsvError>;

pub fn write(records: &Vec<Vec<String>>) -> Result<String, CsvError>;

pub fn write_with_limits(
    records: &Vec<Vec<String>>,
    limits: &CsvLimits
) -> Result<String, CsvError>;
```

No extra public items.

Free functions are mandatory because the repository currently has a cross-file associated-function
resolution gap. Do not use `CsvParser::new`, `CsvLimits::default` or similar required entry points.

`CsvLimits` fields remain package-private. Callers use `limits(...)`.

---

## 9. Default limits

```text
max_input_bytes       = 16_777_216
max_records           = 100_000
max_fields_per_record = 10_000
max_total_fields      = 1_000_000
max_field_bytes       = 1_048_576
max_output_bytes      = 16_777_216
```

Zero means zero permitted, not unlimited.

---

## 10. Data model

Return `Vec<Vec<String>>`:

- outer vector = records in source order;
- inner vector = fields in source order;
- strings contain decoded field content;
- quote syntax is removed;
- doubled quotes become one quote;
- no type conversion occurs.

The model does not preserve original quoting or record-separator style.

---

## 11. Accepted grammar

```text
document       := empty | records
record         := field ("," field)*
field          := unquoted | quoted
unquoted       := UTF-8 text excluding comma, quote, CR and LF
quoted         := '"' quoted_content '"'
quoted_content := non-quote content | '""' repeated
separator      := LF | CRLF
```

Structural bytes:

```text
comma 0x2C
quote 0x22
CR    0x0D
LF    0x0A
```

Scan bytes but preserve non-structural UTF-8 exactly. Use substring spans whose boundaries are
input boundaries or adjacent to ASCII structural bytes. Never normalize Unicode.

---

## 12. Record semantics

```text
""           -> []
"\n"         -> [[""]]
"\r\n"       -> [[""]]
"\n\n"       -> [[""], [""]]
"a\n"        -> [["a"]]
"a\n\n"      -> [["a"], [""]]
","          -> [["", ""]]
",,"         -> [["", "", ""]]
"a,"         -> [["a", ""]]
",a"         -> [["", "a"]]
"a,,b"       -> [["a", "", "b"]]
"a,b\nc"     -> [["a", "b"], ["c"]]
```

A trailing separator terminates the existing record but does not add another record.

Unequal widths are accepted.

---

## 13. Quoting and line endings

A quote opens a quoted field only as its first byte. A quote after unquoted content is
`QuoteInUnquotedField`.

Inside a quoted field:

- `""` decodes to `"`;
- comma, CR, LF and CRLF are content;
- UTF-8 is preserved.

After a closing quote, only comma, LF, CRLF or EOF is allowed. Spaces/tabs after the closing quote
are errors.

Outside quotes:

- LF and CRLF are valid separators;
- bare CR is `BareCarriageReturn`.

EOF inside a quoted field is `UnterminatedQuotedField`, whose offset is the opening quote offset.

Examples:

```text
""""      -> one quote
"a""b"    -> a"b
"""a"""   -> "a"
"a,b"     -> a,b
"a\nb"    -> a<LF>b
```

---

## 14. Error payloads

All offsets/indexes are zero-based.

- `InputTooLarge(actual, limit)`
- `TooManyRecords(attempted, limit)`
- `TooManyFields(record, attempted, limit)`
- `TooManyTotalFields(attempted, limit)`
- `FieldTooLarge(record, field, attempted_bytes, limit)`
- `QuoteInUnquotedField(offset, record, field)`
- `UnexpectedAfterClosingQuote(offset, record, field, byte)`
- `BareCarriageReturn(offset, record, field)`
- `UnterminatedQuotedField(opening_quote_offset, record, field)`
- `OutputTooLarge(attempted_bytes, limit)`

Decoded field size excludes quote syntax; each doubled quote contributes one byte.

Before parsing, enforce input size. During parsing, detect syntax at the current byte before later
limit failures. At field/record finalization enforce field, total-field and record limits in that
order.

All length arithmetic must be overflow-guarded. Overflow counts as exceeding the limit; it must not
trap.


---

## 15. Parser state machine

Use exactly four logical states:

```text
StartField
InUnquoted
InQuoted
AfterClosingQuote
```

Private representation may differ, but observable behaviour may not.

Maintain equivalents of:

```text
records
current_record
current_field
state
byte_index
segment_start
opening_quote_offset
record_index
field_index
total_field_count
record_open
```

### 15.1 `StartField`

At current byte:

- quote:
  - enter `InQuoted`;
  - record opening quote offset;
  - set segment start after quote;
  - mark record open.
- comma:
  - finalize empty field;
  - remain `StartField`;
  - mark record open.
- LF:
  - finalize empty field;
  - finalize record.
- CR followed by LF:
  - finalize empty field;
  - finalize record;
  - consume both bytes.
- bare CR:
  - `BareCarriageReturn`.
- other:
  - enter `InUnquoted`;
  - set segment start at current byte;
  - mark record open.

### 15.2 `InUnquoted`

- comma:
  - append segment before comma;
  - finalize field;
  - enter `StartField`.
- LF:
  - append segment;
  - finalize field and record;
  - enter `StartField`.
- CR followed by LF:
  - append segment;
  - finalize field and record;
  - consume both;
  - enter `StartField`.
- bare CR:
  - `BareCarriageReturn`.
- quote:
  - `QuoteInUnquotedField`.
- other:
  - continue.

### 15.3 `InQuoted`

- quote:
  1. append segment before quote;
  2. if next byte is quote:
     - append one literal quote;
     - consume both quote bytes;
     - set segment start after second quote;
     - remain `InQuoted`;
  3. otherwise:
     - consume the closing quote;
     - enter `AfterClosingQuote`.
- any other byte:
  - continue;
  - CR/LF remain content.

EOF in this state returns `UnterminatedQuotedField`.

### 15.4 `AfterClosingQuote`

- comma:
  - finalize field;
  - enter `StartField`.
- LF:
  - finalize field and record;
  - enter `StartField`.
- CR followed by LF:
  - finalize field and record;
  - consume both;
  - enter `StartField`.
- bare CR:
  - `BareCarriageReturn`.
- EOF:
  - finalize field and record.
- other:
  - `UnexpectedAfterClosingQuote`.

### 15.5 Empty input versus blank records

Use a `record_open` equivalent:

- false at document start;
- true after any byte belonging to a record;
- false after record finalization.

At EOF:

- false means do not create a record;
- true means finalize the pending field and record.

This makes empty input `[]` and a blank line `[[""]]`.

---

## 16. Field size accounting

Do not rely only on `current_field.len()` while an input segment remains unappended.

Attempted decoded field bytes are:

```text
current_field.len() + pending_segment_bytes
```

Escaped quote adds exactly one decoded byte.

Every addition must be overflow-guarded.

Enforce `FieldTooLarge` immediately after decoded content is added or calculated, before later field
or record limits.

No arithmetic trap is acceptable.

---

## 17. Serializer

`write(records)` calls `write_with_limits(records, &default_limits())`, adjusted only for valid
current STARK syntax.

### 17.1 Record separator

Emit exactly CRLF between records:

```text
\r\n
```

No trailing separator after the final record.

Zero records produce `""`.

### 17.2 Minimal deterministic quoting

Quote a field if and only if it contains:

- comma;
- quote;
- CR;
- LF.

Do not quote only because it:

- is empty;
- begins or ends with spaces;
- contains tab;
- contains Unicode;
- contains NUL.

### 17.3 Quoted output

For a quoted field:

1. opening quote;
2. field content left-to-right;
3. every quote doubled;
4. closing quote.

All other UTF-8 content is preserved exactly.

### 17.4 Output limits

Before each append, compute attempted output bytes with overflow guards.

Return `OutputTooLarge` before exceeding the configured limit.

No partial output may escape.

### 17.5 Empty inner records

A zero-field record cannot be represented distinctly in CSV from a one-empty-field record.

Writer normalization:

```text
[[]] -> ""
```

Parsing the output returns:

```text
[[""]]
```

Document this one normalization exception.

---

## 18. Canonicalization properties

For every record tree whose records contain at least one field:

```text
parse(write(records)) == records
```

For valid source text:

```text
write(parse(source))
```

produces:

- CRLF separators;
- no trailing separator;
- minimal required quoting;
- doubled quotes;
- exact decoded content.

It need not reproduce source spelling.

Examples:

```text
"a\nb\n"      -> "a\r\nb"
"\"a\""       -> "a"
"\"a,b\""     -> "\"a,b\""
"\"a\"\"b\""  -> "\"a\"\"b\""
```

---

## 19. Required parser tests

All tests must be committed in `src/tests.stark`.

### 19.1 Basic and separators

- empty input;
- one field;
- two and three fields;
- two records;
- LF;
- CRLF;
- mixed LF/CRLF;
- trailing LF;
- trailing CRLF.

### 19.2 Empty cases

Pin exact structures for:

```text
""
"\n"
"\r\n"
"\n\n"
"\r\n\r\n"
","
",,"
"a,"
",a"
"a,,b"
"a\n\n"
```

### 19.3 Quoted fields

Test:

- quoted ordinary field;
- quoted empty field;
- comma;
- LF;
- CR;
- CRLF;
- leading/trailing spaces;
- tab;
- NUL;
- one escaped quote;
- several escaped quotes;
- escaped quote beside comma;
- escaped quote beside newline.

### 19.4 Unicode preservation

Exact content for:

- Punjabi Gurmukhi;
- Hindi Devanagari;
- CJK;
- emoji;
- combining mark;
- supplementary-plane scalar;
- mixed ASCII/Unicode;
- Unicode beside comma in quotes;
- Unicode beside escaped quote.

No normalization.

### 19.5 Unequal widths

- one field then three;
- three then one;
- blank record between nonblank records.

### 19.6 Quote errors

Pin exact variant, offset, record and field:

```text
a"b
 "a"
a,b"c
```

### 19.7 Post-quote errors

Pin byte payload:

```text
"a"b
"a" 
"a"\t
"a""b"x
```

### 19.8 CR errors

Pin exact CR offset:

```text
a\rb
a\r
,\r,
```

CR inside quotes must succeed.

### 19.9 Unterminated fields

Pin opening quote offset:

```text
"
"a
a,"b
"a""
```

### 19.10 Limits

For every limit:

- exact boundary succeeds;
- first excess returns exact error payload.

Cover:

- input bytes;
- record count;
- fields per record;
- total fields;
- ASCII field bytes;
- multibyte UTF-8 field bytes;
- output bytes.

### 19.11 State reset

- failed parse followed by successful parse;
- repeated successful parse;
- no global mutable state.

---

## 20. Required writer tests

### Basic

- zero records;
- one field;
- multiple fields;
- multiple records;
- unequal widths;
- empty fields;
- zero-field record normalization.

### Minimal quoting

Unquoted:

- ordinary ASCII;
- empty;
- spaces;
- tab;
- Unicode;
- NUL.

Quoted:

- comma;
- quote;
- CR;
- LF;
- CRLF.

### Quote doubling

Pin exact output for fields containing:

```text
a"b
"
""
"a","b"
```

### CRLF

Writer output always uses CRLF and has no trailing separator.

### Limits

Test excess caused by:

- field content;
- comma;
- CRLF;
- opening/closing quotes;
- quote doubling;
- multibyte UTF-8.

### Input immutability

Records and strings remain unchanged after successful and failed writes.

---

## 21. Round-trip corpus

Use deterministic records covering:

- 0, 1, 2, 3, 10 and 100 records;
- 1, 2, 3 and 10 fields;
- field byte lengths:
  - 0, 1, 2;
  - 15, 16, 17;
  - 255, 256, 257;
  - 1023, 1024, 1025;
- structural characters;
- Unicode;
- NUL.

Do not omit boundary lengths because one aggregate test passed.

---

## 22. Host-oracle differential

Commit:

```text
stark-csv/tools/csv_oracle.py
```

A prose claim without a committed script is invalid evidence.

### Python dialect

Use the Python standard `csv` module with:

```python
delimiter=","
quotechar='"'
doublequote=True
skipinitialspace=False
strict=True
lineterminator="\r\n"
quoting=csv.QUOTE_MINIMAL
```

Use `newline=""` correctly.

### Scope

Python is an oracle only for valid records and canonical writer output. STARK owns malformed-input
error classification.

### Corpus

At least 1,000 deterministic cases with a fixed seed, including:

- empty fields;
- commas and quotes;
- CR/LF;
- spaces/tabs;
- NUL;
- Unicode;
- varying dimensions and boundary lengths.

### Required comparison

For every case:

1. Python writes records.
2. STARK parses Python output.
3. STARK data equals original.
4. STARK writes original records.
5. Python reads STARK output.
6. Python data equals original.
7. STARK output equals Python canonical output under the frozen dialect.

### Script output

Print and record:

- Python version;
- seed;
- case count;
- maximum dimensions;
- maximum field bytes;
- pass/fail;
- first failure.

---

## 23. Cross-file proof

`src/tests.stark` must import and call:

- `default_limits`;
- `limits`;
- `parse`;
- `parse_with_limits`;
- `write`;
- `write_with_limits`;
- `CsvError`.

Do not move tests into the implementation file.

A resolver failure is `BLOCKED_CROSS_FILE`.

---

## 24. Cross-package consumer

An ordinary approved path-dependency consumer must:

1. import the package;
2. parse two records;
3. inspect exact fields;
4. write canonical CRLF output;
5. trigger and match one malformed-input error;
6. use no private internals.

Same-package tests do not substitute.

If this cannot be created with current approved tooling, report `BLOCKED_PACKAGE_TOOLING` or
`BLOCKED_CROSS_PACKAGE`.

---

## 25. Native evidence

Do not treat `stark check`, interpreter `stark test` or a check-only `stark build` as native.

Native completion requires a real generated-Rust consumer binary that:

- receives nested `Vec<String>` from `parse`;
- writes nested records;
- returns typed `CsvError` on malformed input;
- links and runs on the recorded target.

When pure/cross-package work passes but native C6.3 is unavailable:

```text
PARTIAL — IMPLEMENTATION COMPLETE, WAITING_C6.3_NATIVE_EVIDENCE
```

---

## 26. Documentation

Add source documentation comments to:

- `CsvError`;
- every variant;
- `CsvLimits`;
- all six public functions.

README must describe:

- exact dialect;
- LF/CRLF input;
- CRLF output;
- quotes and doubled quotes;
- Unicode;
- empty-input versus blank-line semantics;
- unequal widths;
- limits and errors;
- free-function API;
- no native/file I/O/schema conversion;
- current execution status.

Do not claim unqualified full RFC 4180 compliance. Use:

> Strict deterministic CSV derived from RFC 4180 with explicitly frozen v0.1 edge semantics.


---

## 27. Evidence discipline

Previous submissions overstated completion. This package requires a claim-evidence ledger.

### Repository identities

Record separately:

```text
Implementation baseline: <parent SHA before package changes>
Implementation commit:   <commit containing package>
Review head:              <head when evidence was collected>
```

Never label the baseline as the final repository head.

### Exact file list

Generate and paste:

```bash
git diff --name-only <baseline>..<implementation-commit>
```

Do not omit lockfiles, generated docs, tools, fixtures or copied specifications.

### Requirement ledger

For every required category:

```text
Requirement | Test function(s) | File | Result
```

No “fully covered” claim without named tests.

### Command evidence

For every command record:

- working directory;
- exact command;
- exit status;
- test count;
- tool version;
- target;
- actual engine: HIR, MIR or native.

### Prohibited unsupported claims

Do not claim:

- native execution from check;
- cross-package use from same-package tests;
- target support from portable source;
- oracle success without committed script and output;
- scope compliance without exact diff;
- complete docs without inspecting generated public items;
- complete tests when required cases are absent.

---

## 28. Required commands

From `stark-csv/`, run or record honestly:

```bash
stark check
stark test
stark fmt --check
stark doc
stark build
```

From the consumer:

```bash
stark check
stark run
stark build
```

Oracle:

```bash
python3 tools/csv_oracle.py
```

At repository level, record whether run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

`NOT RUN` is acceptable only with a factual reason. It does not imply pass.

---

## 29. Status model

### `COMPLETE`

Only when all are true:

- exact API;
- all parser/writer semantics;
- every committed required test;
- cross-file proof;
- cross-package consumer;
- host oracle;
- docs and exact evidence;
- no out-of-scope changes;
- real native generated-Rust execution;
- supported-target evidence;
- no blocker.

### `PARTIAL — IMPLEMENTATION COMPLETE, WAITING_C6.3_NATIVE_EVIDENCE`

Use when pure implementation, cross-file, cross-package, tests and oracle pass, but real native
nested `String`/`Vec` execution is unavailable.

### Blocking statuses

- `BLOCKED — WAITING_CROSS_FILE_RESOLUTION`
- `BLOCKED — WAITING_CROSS_PACKAGE_RESOLUTION`
- `BLOCKED — WAITING_RUNTIME_VALUES`
- `BLOCKED — WAITING_PACKAGE_TOOLING`

Independent blockers may be listed together with the native partial status.

---

## 30. Blocker template

```markdown
# stark-csv v0.1 blockers

## Primary classification

<status>

## Implementation baseline

<sha>

## Implementation commit

<sha or UNCOMMITTED>

## Minimal reproducer

<files/source>

## Expected

<contract>

## Actual

<exact diagnostic>

## Stage

parse | resolve | typecheck | borrow-check | HIR | MIR | emit | rustc | link | run | package tooling

## Why package-local alternatives fail

1. ...
2. ...

## Prohibited upstream area required

<resolver/C6.3/package manager/etc.>

## Work completed safely

- ...

## Unsupported completion claims

- ...

## Minimum owner decision

<one bounded decision>
```

---

## 31. Implementation order

1. Inspect current head.
2. Run readiness probes A–D.
3. Stop on blocking readiness failure.
4. Create manifest, library, separate tests, README, evidence, oracle and consumer.
5. Write required tests before full implementation.
6. Implement errors and limits.
7. Implement the four-state parser.
8. Implement deterministic writer.
9. Run all package tests without reducing corpus.
10. Run external consumer.
11. Run committed Python oracle.
12. Attempt and classify native evidence.
13. Generate docs and exact claim ledger.
14. Hand off to an independent reviewer.

---

## 32. Suggested private helpers

Names may vary:

```stark
fn append_checked(
    output: &mut String,
    text: &str,
    limit: UInt64
) -> Result<Unit, CsvError>;

fn append_char_checked(
    output: &mut String,
    ch: Char,
    limit: UInt64
) -> Result<Unit, CsvError>;

fn finalize_field(...);
fn finalize_record(...);
fn field_requires_quotes(field: &str) -> Bool;

fn parse_internal(
    input: &str,
    limits: &CsvLimits
) -> Result<Vec<Vec<String>>, CsvError>;

fn write_internal(
    records: &Vec<Vec<String>>,
    limits: &CsvLimits
) -> Result<String, CsvError>;
```

All private. No parser object or alternate dialect API.

---

## 33. Correctness invariants

At every parser loop iteration:

- `byte_index <= input.len()`;
- substring boundaries are valid UTF-8 boundaries;
- field/record indexes correspond to finalized collection lengths;
- decoded field bytes remain within limits;
- total fields count finalized fields only;
- every byte is consumed exactly once;
- structural bytes are excluded unless quoted content;
- no partial records escape on error.

Malformed input returns typed `Err`, not:

- bounds trap;
- arithmetic trap;
- panic;
- host failure;
- partial output.

Expected parser complexity is O(input bytes). Writer complexity is O(total field bytes). Do not
claim performance without benchmarks.

---

## 34. Required usage examples

### Parse

```stark
fn read_rows() -> Result<Vec<Vec<String>>, CsvError> {
    parse("name,city\r\nNavraj,Sydney")
}
```

### Write

```stark
fn write_rows(rows: &Vec<Vec<String>>) -> Result<String, CsvError> {
    write(rows)
}
```

### Custom limits

```stark
fn parse_small(input: &str) -> Result<Vec<Vec<String>>, CsvError> {
    let configured = limits(
        1024u64,
        10u64,
        20u64,
        100u64,
        256u64,
        2048u64
    );
    parse_with_limits(input, &configured)
}
```

Adjust only for valid syntax, not semantics.

---

## 35. Required final report from Gemini

```markdown
## Status

<exact status from Section 29>

## Implementation baseline

<sha>

## Implementation commit

<sha>

## Files changed

<exact git diff --name-only output>

## Readiness probes

| Probe | Result | Evidence |
|---|---|---|
| nested return | | |
| cross-file free function | | |
| cross-package free function | | |
| nested writer argument | | |

## Implemented

- ...

## Public API audit

- [ ] exact
- [ ] no extra public items
- [ ] tests call API cross-file
- [ ] consumer calls API cross-package

## Tests

- total:
- requirement ledger:
- result:

## Oracle

- Python version:
- seed:
- cases:
- command:
- result:

## Execution engines

| Evidence | HIR | MIR | native |
|---|---|---|---|
| package tests | | | |
| consumer | | | |

## Commands

- `<command>` — PASS/FAIL/NOT RUN

## Blockers

- ...

## Scope confirmation

- no compiler changes;
- no runtime changes;
- no spec changes;
- no native provider;
- no manifest invention;
- no tests moved into lib to hide resolution failures.

## Claims not made

- ...
```

Do not omit “Claims not made.”

---

## 36. Owner acceptance checklist

Reject the implementation if any answer is no:

- [ ] exact API;
- [ ] public entry points are free functions;
- [ ] tests remain in a separate file;
- [ ] cross-file use works;
- [ ] external consumer works;
- [ ] empty input differs from blank record;
- [ ] LF and CRLF accepted;
- [ ] bare CR outside quotes rejected;
- [ ] CR/LF inside quotes preserved;
- [ ] quote opens only at field start;
- [ ] doubled quotes decode correctly;
- [ ] post-quote whitespace rejected;
- [ ] UTF-8 preserved exactly;
- [ ] all limits are overflow-safe;
- [ ] writer emits CRLF without trailing separator;
- [ ] quoting is minimal and deterministic;
- [ ] quote doubling is correct;
- [ ] committed oracle reproduces claims;
- [ ] every requirement names tests;
- [ ] exact diff matches file list;
- [ ] baseline differs from implementation commit;
- [ ] native status is honest;
- [ ] no prohibited files changed.

Only an independent reviewer may accept the package.

---

## 37. Follow-on order

After this package is accepted or honestly blocked:

1. fix cross-file/cross-package package usability gaps exposed by Base64, time and CSV;
2. complete C6.3 native `String`, `Vec`, slices and nested collections;
3. create staged `std-json` work:
   - lexical scanner;
   - recursive value tree;
   - parser;
   - serializer;
   - limits and paths;
   - typed conversion as a separate work package;
4. implement `std-url`;
5. land provider execution;
6. implement `std-fs`/`std-io`;
7. implement `std-net`/`std-tls`/`std-http`.

Do not merge follow-on work into `stark-csv`.
