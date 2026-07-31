# STARK JSON v0.1 — Codex Implementation Work Package

**Package:** `stark-json`  
**Version:** `0.1.0`  
**Implementation language:** STARK Core v1 only  
**Native code:** Prohibited  
**Compiler changes:** Prohibited  
**Provider access:** Prohibited  
**Third-party dependencies:** Prohibited  
**Primary standard:** RFC 8259 / ECMA-404 compatible JSON text syntax  
**Object member policy:** Preserve input/insertion order  
**Duplicate object member policy:** Reject duplicates in v0.1  
**Number representation:** Lossless lexical number representation  
**Status:** Implementation specification — public API and behaviour frozen for this work package  
**Required first action:** Re-pin to the actual repository head before implementation

---

## 1. Instruction to Codex

Implement only the package specified here. Treat every statement containing **MUST**, **MUST NOT**, **SHALL**, **SHALL NOT**, **REQUIRED**, or **EXACTLY** as binding.

Do not redesign the API. Do not add compiler intrinsics, runtime builtins, native providers, macros, reflection, code generation, package-manager changes, or convenience APIs outside the frozen surface.

Responsibilities:

1. inspect the current repository and verify prerequisites;
2. create the package in the owner-assigned directory;
3. implement the frozen public API in pure STARK;
4. create the complete positive, negative, Unicode, boundary, position, determinism, and round-trip test matrix;
5. validate through every execution engine currently able to support it;
6. reduce and report compiler/runtime blockers without modifying compiler-owned files;
7. record exact evidence and final status;
8. stop without widening scope.

If a compiler or runtime capability is unavailable, do not fix it inside this package. Use the blocker protocol in Section 20.

---

## 2. Objective

Create a deterministic JSON package that can:

- parse a complete UTF-8 JSON text into a recursive STARK value;
- encode that value into compact valid JSON;
- preserve object member order;
- reject malformed input deterministically;
- report stable error categories and source positions;
- enforce explicit resource limits;
- run without native code, privileged hooks, OS access, or third-party dependencies;
- serve later as the JSON component of the P1 Native Systems Baseline.

The package should exercise real STARK capabilities: `String`, `str`, `Vec`, recursive enums, `Option`, `Result`, pattern matching, loops, checked arithmetic, ownership, Drop, modules, packages, and cross-engine execution.

---

## 3. Scope

### 3.1 Included

The package MUST support:

- `null`;
- booleans;
- strings;
- numbers;
- arrays;
- objects;
- arbitrary nesting within configured limits;
- JSON whitespace;
- all required string escapes;
- UTF-8 input;
- UTF-16 `\uXXXX` decoding, including valid surrogate pairs;
- compact encoding;
- deterministic object order;
- duplicate-key rejection;
- stable byte offset, line, and column reporting;
- explicit parser limits;
- full-input consumption;
- deterministic error selection;
- parse/encode/parse round trips;
- package documentation;
- HIR evidence;
- MIR/native evidence when the current compiler supports the full surface.

### 3.2 Explicitly excluded

Do not implement:

- JSON5;
- comments;
- trailing commas;
- unquoted keys;
- single-quoted strings;
- hexadecimal, binary, octal, `NaN`, or infinity numbers;
- automatic conversion of all numbers to machine integers/floats;
- arbitrary-precision arithmetic;
- JSON Pointer;
- JSON Patch;
- JSON Schema;
- streaming parser or encoder;
- event/SAX API;
- reader/writer/file APIs;
- network APIs;
- reflection or derived serialization;
- pretty printing;
- RFC 8785 canonical JSON;
- sorted object keys;
- query/path helpers;
- mutation helpers;
- native Rust/C implementation;
- compiler/runtime builtins;
- registry publication;
- concurrency or async parsing;
- performance claims beyond measured evidence.

---

## 4. Frozen design decisions

### 4.1 Object representation

Objects MUST use an ordered vector:

```stark
pub struct JsonMember {
    pub key: String,
    pub value: JsonValue,
}

pub enum JsonValue {
    Null,
    Bool(Bool),
    Number(JsonNumber),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<JsonMember>),
}
```

Do not use `HashMap` in v0.1.

### 4.2 Duplicate keys

Duplicate decoded keys MUST be rejected. Thus:

```json
{"a":1,"\u0061":2}
```

is invalid. Report the second key.

### 4.3 Numbers

Numbers MUST retain the validated original lexeme:

```stark
pub struct JsonNumber {
    raw: String,
}
```

Do not parse all numbers into `Float64`. The encoder emits the stored lexeme exactly.

### 4.4 Strings

Parsed strings store decoded Unicode scalar values. Original escape spelling is not preserved.

### 4.5 Object order

Parsing preserves source order. Encoding emits stored order. Keys MUST NOT be sorted.

### 4.6 Error selection

Return the first error encountered under the deterministic precedence defined below. Do not collect multiple errors.

---

## 5. Preconditions and repository inspection

Before editing, inspect current `main` and record:

```text
compiler SHA
package format
CLI commands
C6.3 String/Vec/slice/iterator status
HIR support
MIR support
native support
cross-package support
```

Confirm:

1. recursive enums;
2. `Vec<JsonValue>` and `Vec<JsonMember>`;
3. String construction and append;
4. `&str` or equivalent string view;
5. UTF-8 byte/scalar iteration;
6. byte-offset tracking;
7. safe indexing or iterator equivalent;
8. `Result<T,E>`;
9. `Option<T>`;
10. recursive pattern matching;
11. checked arithmetic;
12. cross-module package compilation;
13. string equality;
14. recursive Drop;
15. package-local tests;
16. cross-package imports;
17. deterministic execution.

Allowed classifications:

- `READY` — HIR, MIR, and native support all required capabilities.
- `INTERPRETER_READY` — HIR implementation/tests can proceed; MIR/native incomplete.
- `DESIGN_READY` — API, tests, docs, and skeleton can be completed but execution is blocked.
- `BLOCKED` — implementation cannot proceed without changing frozen semantics or prohibited infrastructure.

HIR-only success MUST remain partial.

---

## 6. Package structure

Preferred layout:

```text
stark-json/
├── starkpkg.json
├── README.md
├── EVIDENCE.md
├── TEST-MATRIX.md
├── src/
│   ├── lib.stark
│   ├── value.stark
│   ├── number.stark
│   ├── error.stark
│   ├── limits.stark
│   ├── cursor.stark
│   ├── parser.stark
│   ├── encoder.stark
│   └── tests.stark
└── fixtures/
    ├── valid/
    └── invalid/
```

Allowed modifications:

- package directory only;
- one owner-approved consumer fixture/package.

Prohibited modifications:

- `starkc/`;
- `stark-runtime/`;
- specs;
- compiler work packages;
- `COMPILER-STATE.md`;
- package-manager implementation;
- unrelated packages;
- shared compiler tests or evidence.

---

## 7. Manifest

Use the current approved package format. Expected form:

```json
{
  "name": "stark-json",
  "version": "0.1.0",
  "entry": "src/lib.stark",
  "dependencies": {}
}
```

The package MUST have no dependencies, provider declarations, build scripts, install scripts, network access, compiler extensions, or generated host-language source.

---

## 8. Frozen public API

Implement exactly:

```stark
pub enum JsonValue {
    Null,
    Bool(Bool),
    Number(JsonNumber),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<JsonMember>),
}

pub struct JsonMember {
    pub key: String,
    pub value: JsonValue,
}

pub struct JsonNumber {
    raw: String,
}

pub struct JsonLimits {
    pub max_input_bytes: UInt64,
    pub max_depth: UInt32,
    pub max_string_bytes: UInt64,
    pub max_array_items: UInt64,
    pub max_object_members: UInt64,
    pub max_number_bytes: UInt64,
}

pub enum JsonErrorKind {
    InputTooLarge,
    UnexpectedEnd,
    UnexpectedByte,
    ExpectedValue,
    ExpectedColon,
    ExpectedCommaOrEnd,
    TrailingCharacters,
    InvalidLiteral,
    InvalidNumber,
    NumberTooLong,
    InvalidEscape,
    InvalidUnicodeEscape,
    LoneHighSurrogate,
    LoneLowSurrogate,
    UnescapedControlCharacter,
    InvalidUtf8,
    DuplicateObjectKey,
    MaximumDepthExceeded,
    StringTooLong,
    TooManyArrayItems,
    TooManyObjectMembers,
    PositionOverflow,
}

pub struct JsonError {
    pub kind: JsonErrorKind,
    pub offset: UInt64,
    pub line: UInt64,
    pub column: UInt64,
}

pub fn default_limits() -> JsonLimits;

pub fn parse(input: &str) -> Result<JsonValue, JsonError>;

pub fn parse_with_limits(
    input: &str,
    limits: &JsonLimits,
) -> Result<JsonValue, JsonError>;

pub fn encode(value: &JsonValue) -> String;

pub fn number_raw(number: &JsonNumber) -> &str;

pub fn member_key(member: &JsonMember) -> &str;

pub fn member_value(member: &JsonMember) -> &JsonValue;
```

Do not add public parser, cursor, token, builder, mutation, query, file, pretty-printing, or number-conversion APIs.

---

## 9. Default limits

`default_limits()` MUST return:

```text
max_input_bytes      = 16,777,216
max_depth            = 128
max_string_bytes     = 4,194,304
max_array_items      = 1,000,000
max_object_members   = 1,000,000
max_number_bytes     = 1,024
```

Interpretation:

- byte limits count UTF-8 bytes;
- depth counts currently open arrays/objects;
- string size counts decoded UTF-8 bytes;
- item/member limits apply per container;
- number size counts token bytes.

All counters MUST use checked arithmetic. Counter overflow returns `PositionOverflow` rather than trapping or wrapping.

---

## 10. Position model

- `offset`: zero-based UTF-8 byte offset.
- `line`: one-based.
- `column`: one-based Unicode scalar column.
- LF increments line and resets column to 1.
- CR is whitespace but does not independently create a new line.
- CRLF counts as one line break at LF.
- tab advances column by one.

Errors identify the first byte that demonstrates the error.

---

## 11. Grammar

```text
json-text = ws value ws EOF

value = object | array | string | number | "true" | "false" | "null"

object =
    "{" ws ("}" | member (ws "," ws member)* ws "}")

member = string ws ":" ws value

array =
    "[" ws ("]" | value (ws "," ws value)* ws "]")

string = '"' character* '"'

character = unescaped | escape

escape =
    '\"' | '\\' | '\/' | '\b' | '\f' | '\n' | '\r' | '\t'
  | '\u' hex hex hex hex

number = minus? int frac? exp?
minus = "-"
int = "0" | digit1-9 digit*
frac = "." digit+
exp = ("e" | "E") ("+" | "-")? digit+

ws = *(U+0020 | U+0009 | U+000A | U+000D)
```

No other whitespace is accepted.

---

## 12. Parser architecture

Preferred architecture: cursor-driven recursive descent without a complete token-vector allocation.

Private components:

```text
Cursor
parse_value
parse_object
parse_array
parse_string
parse_number
parse_literal
skip_whitespace
peek_byte
consume_byte
position
```

Cursor invariants:

- never advances beyond input;
- lookahead does not consume;
- every consuming operation updates position exactly once;
- parser loops consume input or return;
- malformed UTF-8 cannot be silently ignored;
- ASCII punctuation may be processed by byte;
- scalar decoding happens only at valid boundaries.

After one value, skip whitespace and reject any remaining byte as `TrailingCharacters`.

---

## 13. Deterministic error precedence

### 13.1 Global

1. input-size check;
2. parse root value;
3. skip trailing whitespace;
4. reject trailing characters.

### 13.2 Value dispatch

- `{` → object;
- `[` → array;
- `"` → string;
- `-` or digit → number;
- `t`, `f`, `n` → literal;
- EOF → `UnexpectedEnd`;
- otherwise → `ExpectedValue`.

### 13.3 Object

- after `{` and whitespace, `}` means empty;
- otherwise a quoted key is required;
- after key, colon is required;
- after value, comma or `}` is required;
- trailing comma fails at the next `}` because a key is required.

### 13.4 Array

- after `[` and whitespace, `]` means empty;
- otherwise parse a value;
- after item, comma or `]` is required;
- trailing comma fails at the next `]` as `ExpectedValue`.

### 13.5 Literals

Compare expected bytes left-to-right. Premature EOF is `UnexpectedEnd`; mismatch is `InvalidLiteral`. A valid literal followed by extra text produces `TrailingCharacters` or an enclosing delimiter error.

---

## 14. Number semantics

Accepted examples:

```text
0
-0
1
-1
1234567890
0.0
-0.0
1.25
1e10
1E10
1e+10
1e-10
-12.34e+56
```

Rejected examples:

```text
+1
01
-01
.
1.
.1
1e
1e+
1e-
--1
NaN
Infinity
0x10
1_000
```

Validation:

1. optional `-`;
2. integer component: `0` or nonzero digit plus digits;
3. optional fraction with at least one digit;
4. optional exponent with optional sign and at least one digit;
5. stop at valid delimiter/whitespace;
6. invalid suffix produces `InvalidNumber`;
7. enforce `max_number_bytes`;
8. store exact lexeme.

Valid delimiters:

```text
space tab CR LF , ] } EOF
```

`01` fails at the second byte. Number validation MUST NOT depend on host float parsing.

---

## 15. String and Unicode semantics

Inside a string:

- `"` ends the string;
- `\` starts an escape;
- raw U+0000..U+001F are invalid;
- all other valid Unicode scalars are accepted.

Simple escapes decode exactly:

```text
\" \\ \/ \b \f \n \r \t
```

Unicode escapes:

- `\u` must have exactly four ASCII hex digits;
- malformed digits → `InvalidUnicodeEscape`;
- premature EOF → `UnexpectedEnd`.

Surrogates:

- non-surrogate values become scalars directly;
- high surrogate `D800..DBFF` must be followed immediately by a low-surrogate escape;
- lone high → `LoneHighSurrogate`;
- lone low → `LoneLowSurrogate`;
- valid pair computes:

```text
0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
```

Track decoded UTF-8 byte length. Exceeding the limit returns `StringTooLong` at the source byte beginning the excess scalar/escape.

Encoder escaping:

- quote → `\"`;
- backslash → `\\`;
- standard controls → short escapes;
- other controls → uppercase `\u00XX`;
- slash remains `/`;
- other Unicode scalars emit direct UTF-8.

---

## 16. Array semantics

- preserve order;
- enforce `max_array_items` before appending excess item;
- entering an array increments depth;
- if new depth exceeds limit, report `MaximumDepthExceeded` at `[`;
- empty array encodes as `[]`;
- nonempty array encodes compactly with commas and no spaces.

---

## 17. Object semantics

- preserve member order;
- enforce `max_object_members` before appending excess member;
- duplicate detection compares decoded key strings;
- duplicate error points to the new duplicate key;
- exact decoded-string equality is used;
- Unicode normalization is not performed;
- duplicate scan may be O(n²) in v0.1;
- encoding is compact and preserves stored order.

---

## 18. Encoder semantics

`encode` MUST:

- always produce valid JSON for a valid `JsonValue`;
- be deterministic;
- preserve array and object order;
- emit number lexemes exactly;
- escape strings deterministically;
- not use debug formatting;
- not mutate or consume the input.

Exact primitive output:

```text
Null        → null
Bool(true)  → true
Bool(false) → false
```

If recursive encoding risks stack limits, an explicit-stack implementation is preferred. Do not change the public API to return an error merely to avoid internal recursion work.

---

## 19. Cross-package consumer

Create an owner-approved consumer proving:

- package resolution;
- public API visibility;
- recursive value return across package boundary;
- encoding of a parsed value;
- Drop at program exit;
- native build where available;
- no privileged compiler/runtime hook.

---

## 20. Compiler blocker protocol

Codex MUST NOT modify compiler/runtime code.

For each blocker, report:

```text
BLOCKER ID
PACKAGE COMMIT
COMPILER COMMIT
EXECUTION ENGINE
MINIMISED STARK SOURCE
COMMAND
EXPECTED RESULT
ACTUAL RESULT
DIAGNOSTICS
REQUIRED CAPABILITY
NORMATIVE BASIS
WORKAROUND CONSIDERED
WHY WORKAROUND WAS REJECTED OR TEMPORARY
```

Classify as:

```text
FRONTEND_REJECTION
BORROW_OR_LIFETIME
MIR_LOWERING
MIR_VERIFIER
NATIVE_BACKEND
RUNTIME_STRING
RUNTIME_VEC
RUNTIME_RECURSIVE_DROP
PACKAGE_SYSTEM
TEST_RUNNER
FORMATTER
UNKNOWN
```

Forbidden permanent workarounds include flattening all JSON into strings/tokens, converting all numbers to `Float64`, omitting Unicode, accepting duplicates, sorting keys, using host-language parsing, or removing source positions.

---

## 21. Test strategy

Required groups:

```text
API
PRIMITIVES
WHITESPACE
NUMBERS_VALID
NUMBERS_INVALID
STRINGS_VALID
STRINGS_INVALID
UNICODE
SURROGATES
ARRAYS
OBJECTS
DUPLICATE_KEYS
NESTING
LIMITS
POSITIONS
ENCODING
ROUND_TRIP
DETERMINISM
OWNERSHIP_DROP
CROSS_PACKAGE
THREE_ENGINE
```

Every `JsonErrorKind` requires at least one exact test for:

```text
kind
offset
line
column
```

### 21.1 Minimum valid cases

Primitives:

```text
null true false 0 -0 1 -1 0.0 1.25 1e10 1E-10 "" "hello" [] {}
```

Strings:

- ASCII;
- every simple escape;
- BMP escape;
- lowercase/uppercase hex;
- direct UTF-8;
- valid surrogate pair;
- embedded NUL via escape.

Arrays/objects:

- empty and nonempty;
- mixed types;
- nested arrays/objects;
- Unicode keys;
- order preservation.

Numbers:

- huge valid integer lexeme;
- long fraction;
- exponent variants;
- negative zero;
- maximum allowed token length.

### 21.2 Minimum invalid cases

General:

```text
(empty)
(space only)
x
true false
nullx
```

Arrays:

```text
[
[1
[1,
[1,]
[,1]
[1 2]
```

Objects:

```text
{
{"a"
{"a":
{"a":1
{"a":1,
{"a":1,}
{a:1}
{"a" 1}
{"a":1 "b":2}
```

Strings:

- unclosed quote;
- raw LF/control;
- invalid escape;
- short/non-hex Unicode escape;
- lone surrogates;
- string-limit overflow.

Numbers:

```text
-
-.
00
-00
1..0
1e1.0
1a
```

Duplicate keys:

```json
{"a":1,"a":2}
{"a":1,"\u0061":2}
{"\uD834\uDD1E":1,"𝄞":2}
```

### 21.3 Position tests

Include errors after:

- ASCII;
- multibyte UTF-8;
- LF;
- CRLF;
- tab;
- nested arrays/objects;
- string escapes;
- duplicate key.

### 21.4 Limits

For every limit:

- exact boundary succeeds;
- boundary + 1 fails;
- error position exact;
- precedence over unrelated limits tested.

### 21.5 Round trips

For representative values:

```text
parse(encode(value)) == value
```

For valid text:

```text
value1 = parse(input)
text2 = encode(value1)
value2 = parse(text2)
value1 == value2
```

Do not require encoded text to match original whitespace/escape spelling.

For valid numbers:

```text
encode(parse(number)) == original_number
```

### 21.6 Determinism

- same input twice → equal value;
- same value twice → identical bytes;
- stable errors;
- stable object order;
- HIR/MIR/native agreement where available;
- relocation does not alter output.

### 21.7 Ownership/Drop

Provide package workloads proving:

- nested values drop normally;
- partial values are cleaned on `Result::Err` return;
- returned parsed value outlives parser locals;
- encoder borrows rather than consumes;
- cross-package return works;
- no clone is added merely to bypass ownership.

---

## 22. External corpus

A checked-in licence-compatible JSON corpus may be used.

Requirements:

- record provenance and licence;
- never fetch during tests;
- classify accepted/rejected/interoperability-sensitive fixtures;
- include STARK-specific position, limit, and determinism tests separately;
- do not blindly import redundant cases.

---

## 23. Module responsibilities

### `lib.stark`
Public re-exports and top-level API only.

### `value.stark`
`JsonValue`, `JsonMember`, accessors, test-only structural comparison helper if needed.

### `number.stark`
`JsonNumber`, private validated construction, `number_raw`.

### `error.stark`
`JsonErrorKind`, `JsonError`, private constructors.

### `limits.stark`
`JsonLimits`, `default_limits`.

### `cursor.stark`
Private byte/scalar cursor and position tracking.

### `parser.stark`
Recursive descent, strings, Unicode, numbers, limits, duplicates.

### `encoder.stark`
Compact deterministic encoding.

### `tests.stark`
Package tests and helpers only; no production logic hidden in tests.

---

## 24. Milestones

### J0 — Capability audit
Record current SHA, prerequisites, package path, and readiness status.

### J1 — Data model and errors
Implement frozen types, limits, accessors, and API compile tests.

### J2 — Cursor and positions
Implement byte/scalar navigation, whitespace, line/column, checked counters.

### J3 — Primitives and numbers
Implement null, booleans, number grammar, full-input check, number limits.

### J4 — Strings and Unicode
Implement direct UTF-8, escapes, surrogate pairs, controls, positions, limits.

### J5 — Arrays
Implement recursion, delimiters, item limits, depth, partial cleanup.

### J6 — Objects
Implement keys, members, duplicates, order, member limits, depth.

### J7 — Encoder
Implement compact deterministic output and round trips.

### J8 — Breadth corpus
Complete grammar, malformed-input, boundary, deep, large, no-hang tests.

### J9 — Cross-package and engines
Run consumer, HIR, MIR, native; reduce/report blockers.

### J10 — Documentation and closure
Complete README, EVIDENCE, TEST-MATRIX, exact commands/counts, final status.

---

## 25. Required commands

Adapt to the current CLI and record exact commands. Expected forms:

```bash
stark check
stark test
stark fmt --check
stark build
stark run
```

Run from both package and consumer where relevant.

Do not represent interpreter success as native success.

---

## 26. Complexity expectations

```text
parse overall       O(input bytes + duplicate checks)
array construction  O(items)
object construction O(members²) worst case
encode              O(output bytes)
memory              O(parsed value + parser stack)
```

Do not claim linear object parsing while duplicate detection scans prior members.

No unsafe code, host parser, hash-map substitution, or unchecked preallocation.

---

## 27. Security and robustness

The package MUST:

- check input size before parsing;
- check nesting depth;
- check string/number/container sizes;
- use checked counters;
- make parser progress or return;
- reject malformed escapes and raw controls;
- reject duplicate keys;
- avoid network/filesystem access;
- avoid executing input;
- return deterministic bounded errors.

Each parser loop iteration MUST consume input or return.

Add no-hang tests for:

```text
[
{
"
\
\u
-
1e
{"a"
```

---

## 28. Documentation

### README.md

Include:

- purpose;
- supported scope;
- API;
- number model;
- duplicate-key policy;
- object-order policy;
- limits;
- examples;
- exclusions;
- engine support;
- complexity.

### EVIDENCE.md

Include:

```text
package commit
compiler commit
toolchain
platform
readiness status
commands
test counts
HIR result
MIR result
native result
cross-package result
formatter result
external corpus provenance
blockers
final status
```

### TEST-MATRIX.md

Each row:

```text
ID
category
input/fixture
expected result
expected error
offset
line
column
engines
status
```

---

## 29. Acceptance criteria

### API

- [ ] exact public API;
- [ ] no extra public API;
- [ ] no dependencies;
- [ ] no compiler/runtime changes;
- [ ] cross-package import works.

### Parsing

- [ ] every grammar production;
- [ ] exact whitespace;
- [ ] full input consumed;
- [ ] recursive arrays/objects;
- [ ] duplicate rejection;
- [ ] exact number grammar and lexeme preservation;
- [ ] string/escape/Unicode correctness;
- [ ] surrogate correctness;
- [ ] controls rejected;
- [ ] all limits;
- [ ] checked positions.

### Encoding

- [ ] valid compact JSON;
- [ ] deterministic bytes;
- [ ] order preserved;
- [ ] number lexeme preserved;
- [ ] string escaping deterministic;
- [ ] no debug-format substitution;
- [ ] input not consumed.

### Testing

- [ ] valid/invalid matrix complete;
- [ ] every error kind tested;
- [ ] exact positions;
- [ ] limits;
- [ ] Unicode/surrogates;
- [ ] duplicate equivalence cases;
- [ ] round trips;
- [ ] determinism;
- [ ] deep/large bounded cases;
- [ ] no-hang progress corpus.

### Engines

For `READY — COMPLETE`:

- [ ] HIR passes;
- [ ] MIR passes;
- [ ] native debug passes;
- [ ] observations agree;
- [ ] cross-package native consumer passes;
- [ ] supported Tier-1 evidence recorded.

For partial readiness:

- [ ] HIR requirements pass;
- [ ] missing engine capabilities reduced/reported;
- [ ] status remains partial.

---

## 30. Status vocabulary

Use one:

```text
READY — COMPLETE
PARTIAL — WAITING_COMPILER_RUNTIME
PARTIAL — IMPLEMENTATION_IN_PROGRESS
BLOCKED — COMPILER_CAPABILITY
BLOCKED — DESIGN_AUTHORITY
```

Do not use “complete” for HIR-only success.

---

## 31. Adversarial review checklist

### Parsing

- Can malformed input create a non-progress loop?
- Can lookahead consume?
- Can unclosed containers succeed?
- Can trailing content be ignored?
- Can trailing commas pass?
- Can non-JSON whitespace pass?
- Can literal prefixes pass?
- Can invalid number suffixes pass?
- Can leading zero pass?

### Unicode

- Can short/non-hex `\u` pass?
- Can lone surrogates pass?
- Can invalid pairs create scalars?
- Can decoded byte length be undercounted?
- Are offsets bytes and columns scalars?
- Is CRLF consistent?
- Can raw controls pass?

### Objects

- Can direct duplicates pass?
- Can escape-equivalent duplicates pass?
- Is Unicode normalization incorrectly applied?
- Is order preserved?
- Can missing colon be misclassified?
- Can a member append before limit rejection?

### Numbers

- Is `-0` preserved?
- Are exponent case/sign preserved?
- Are huge numbers accepted?
- Is host float parsing used?
- Can callers construct invalid `JsonNumber`?
- Can encoder emit invalid numbers?

### Ownership

- Are values cloned to evade moves?
- Are parser temporaries retained?
- Are partial structures cleaned on error?
- Does cross-package return work?
- Does encoder borrow?
- Does recursive native Drop work?

### Evidence

- Are skips visible?
- Is HIR success mislabeled native?
- Are exact commits recorded?
- Are fixture licences recorded?
- Are blockers reduced?
- Is every claim executable?

---

## 32. Final report format

```text
STATUS
PACKAGE PATH
PACKAGE HEAD
COMPILER HEAD
PLATFORM
TOOLCHAIN
FILES CREATED
FILES MODIFIED OUTSIDE PACKAGE
PUBLIC API
PRECONDITION RESULT
PARSER IMPLEMENTATION
ENCODER IMPLEMENTATION
NUMBER MODEL
DUPLICATE-KEY POLICY
OBJECT-ORDER POLICY
DEFAULT LIMITS
VALID TEST COUNT
INVALID TEST COUNT
POSITION TEST COUNT
LIMIT TEST COUNT
ROUND-TRIP TEST COUNT
EXTERNAL CORPUS RESULT
HIR RESULT
MIR RESULT
NATIVE RESULT
CROSS-PACKAGE RESULT
FORMAT RESULT
PERFORMANCE OBSERVATIONS
COMPILER BLOCKERS
PACKAGE LIMITATIONS
DEVIATIONS FROM SPEC
RECOMMENDED STATUS
```

Expected:

```text
FILES MODIFIED OUTSIDE PACKAGE: NONE
```

---

## 33. Direct execution prompt

Implement `stark-json` v0.1 as a pure-STARK package using this work package as binding authority.

Re-pin the repository and classify prerequisites as `READY`, `INTERPRETER_READY`, `DESIGN_READY`, or `BLOCKED`. Work only inside the owner-assigned package directory. Do not modify compiler, runtime, specs, package manager, shared test infrastructure, or compiler state documents.

Use ordered `Vec<JsonMember>` objects. Preserve insertion/source order. Reject duplicate decoded keys. Represent numbers as validated original lexemes; do not convert all numbers to `Float64`. Parse complete RFC 8259-style JSON with deterministic first-error reporting, byte offsets, one-based lines, one-based Unicode-scalar columns, strict escapes, valid surrogate pairs, full-input consumption, and explicit resource limits.

Implement compact deterministic encoding. Emit stored object order, exact number lexemes, deterministic string escapes, and direct UTF-8 for ordinary Unicode scalars.

Build the complete positive, negative, number, Unicode, surrogate, duplicate-key, position, limit, nesting, round-trip, determinism, ownership, no-hang, and cross-package matrix. Every error kind requires an exact position test.

When a compiler/runtime issue appears, minimize and report it through the blocker protocol. Do not redesign JSON semantics or add host-language parsing. HIR-only success remains `PARTIAL — WAITING_COMPILER_RUNTIME` until required MIR/native execution works.

Finish with README.md, EVIDENCE.md, TEST-MATRIX.md, exact commands/counts, exact package/compiler commits, and the final report in Section 32.
