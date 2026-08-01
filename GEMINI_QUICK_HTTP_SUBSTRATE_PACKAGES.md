# Gemini Work Package — Quick HTTP Substrate Packages

**Status:** APPROVED FOR GEMINI EXECUTION  
**Execution lane:** Low-complexity, pure STARK package implementation  
**Primary purpose:** Build reusable substrate for the existing P1 REST server and HTTP client Track A  
**Authority level:** Package implementation only  
**Expected landing state:** `IMPLEMENTED_LOCAL`

---

## 1. Scope

Gemini is assigned the following package batch:

### Start immediately, in parallel

1. `stark-ascii`
2. `stark-percent`
3. `stark-mime`

### Start after `stark-percent`

4. `stark-query`

### Optional, only if capacity remains

5. `stark-form`

These packages are not a separate campaign. They exist only to support:

- P1 REST server request parsing;
- HTTP request-target parsing;
- HTTP header validation and case-insensitive comparison;
- `Content-Type` parsing;
- query-string handling;
- HTTP client Track A.

---

## 2. Hard execution constraints

Gemini must not:

- modify the compiler;
- modify MIR;
- modify provider ABI contracts;
- modify native providers;
- introduce new generic frameworks;
- add concurrency;
- add async;
- broaden package scope;
- redefine standing roadmap priorities;
- change package semantics to hide compiler defects;
- invent alternate spellings for already-frozen APIs;
- add host dependencies to these packages.

If a compiler, MIR, package-manager, interpreter, or backend defect blocks correct implementation:

1. stop that affected path;
2. create a minimal reproducer;
3. record expected and actual behaviour;
4. identify the failing layer;
5. report the blocker;
6. do not weaken the package API or semantics to work around it.

General project law:

> **Do not modify package semantics to hide a compiler defect.**

---

## 3. Delivery order

```text
stark-ascii ───────────────┬── stark-percent ── stark-query ── stark-form
                           │
                           └── stark-mime
```

Execution order:

```text
Wave 1:
- stark-ascii
- stark-percent
- stark-mime

Wave 2:
- stark-query

Wave 3, optional:
- stark-form
```

`stark-query` must not begin until the `stark-percent` public API is frozen locally.

`stark-form` must not begin until both `stark-percent` and `stark-query` are stable locally.

---

# 4. Package 1 — `stark-ascii`

## Objective

Provide byte-first ASCII primitives for HTTP, URL, MIME, and parser packages.

## Design rule

The primary API must operate on `UInt8` and byte slices.

String APIs, if added, are convenience wrappers only.

HTTP parsing operates on untrusted bytes before UTF-8 validation. The package must not force callers to create `String` values before validating protocol bytes.

## Required public API

```stark
fn is_ascii(byte: UInt8) -> Bool;

fn is_ascii_alpha(byte: UInt8) -> Bool;
fn is_ascii_uppercase(byte: UInt8) -> Bool;
fn is_ascii_lowercase(byte: UInt8) -> Bool;
fn is_ascii_digit(byte: UInt8) -> Bool;
fn is_ascii_hex_digit(byte: UInt8) -> Bool;
fn is_ascii_whitespace(byte: UInt8) -> Bool;
fn is_ascii_control(byte: UInt8) -> Bool;

fn is_tchar(byte: UInt8) -> Bool;

fn to_ascii_lowercase(byte: UInt8) -> UInt8;
fn to_ascii_uppercase(byte: UInt8) -> UInt8;

fn eq_ignore_ascii_case(
    left: &[UInt8],
    right: &[UInt8],
) -> Bool;
```

Optional wrappers:

```stark
fn string_eq_ignore_ascii_case(
    left: &String,
    right: &String,
) -> Bool;
```

The wrapper must delegate to the byte implementation.

## `tchar` definition

`is_tchar` must implement the HTTP token-character set:

```text
! # $ % & ' * + - . ^ _ ` | ~
0-9
A-Z
a-z
```

It must reject:

- spaces;
- tabs;
- separators not listed above;
- control bytes;
- non-ASCII bytes.

## Required tests

### Classification

- all ASCII letters;
- all digits;
- hexadecimal digits;
- ASCII whitespace;
- control bytes;
- bytes `0x80`–`0xFF`;
- all valid `tchar` punctuation;
- invalid token punctuation.

### Case conversion

- uppercase to lowercase;
- lowercase to uppercase;
- digits unchanged;
- punctuation unchanged;
- non-ASCII unchanged.

### Comparison

- equal same-case bytes;
- equal mixed-case bytes;
- unequal content;
- unequal length;
- empty slices;
- non-ASCII bytes compared exactly, not Unicode-folded.

## Exclusions

Do not add:

- Unicode case folding;
- locale-aware behaviour;
- UTF-8 validation;
- Unicode categories;
- normalization;
- collation.

## Exit criteria

- API is byte-first;
- all 256 byte values are covered by classification tests;
- HIR/MIR/native behaviour matches where supported;
- no host dependency;
- deterministic on every platform.

---

# 5. Package 2 — `stark-percent`

## Objective

Provide strict RFC-style percent encoding and decoding for URLs and query components.

## Scope rule

v0.1 must use a closed enum of named encode sets.

Do not implement user-defined 128-entry tables, callback predicates, or arbitrary custom sets.

## Required public types

```stark
enum PercentEncodeSet {
    PathSegment,
    Path,
    QueryComponent,
}
```

Only add another named set if an existing frozen HTTP or URL requirement proves it is necessary.

## Required public API

```stark
fn encode(
    input: &[UInt8],
    set: PercentEncodeSet,
) -> String;

fn decode(
    input: &String,
) -> Result<Vec<UInt8>, PercentError>;
```

## Error model

```stark
enum PercentError {
    IncompleteEscape {
        offset: UInt64,
    },

    InvalidHexDigit {
        offset: UInt64,
        byte: UInt8,
    },

    OutputTooLarge,
}
```

Use the exact project-supported enum/field syntax. If named fields are not admitted, preserve the same information using the nearest accepted representation.

## Required semantics

- encode using uppercase hexadecimal digits;
- decode both uppercase and lowercase hexadecimal digits;
- malformed `%` sequences fail;
- exact failure offset is reported;
- decoding returns bytes;
- no implicit UTF-8 reconstruction;
- no implicit `+`-as-space;
- `+` remains a literal plus byte;
- output growth is checked;
- behaviour is deterministic.

## Required tests

### Encoding

- unreserved bytes;
- reserved bytes under each closed encode set;
- space;
- slash;
- question mark;
- percent;
- non-ASCII bytes;
- empty input.

### Decoding

- uppercase hex;
- lowercase hex;
- mixed hex;
- encoded UTF-8 bytes;
- literal plus;
- incomplete `%`;
- one-digit escape;
- invalid first hex digit;
- invalid second hex digit;
- exact offset checks;
- empty input.

### Round trips

- fixed vectors;
- all byte values;
- encode/decode round trip under suitable sets.

## Exclusions

Do not add:

- form encoding;
- `+` conversion;
- user-defined encode tables;
- Unicode normalization;
- URL parsing;
- query parsing.

## Exit criteria

- closed encode-set enum only;
- strict decoding;
- exact error offsets;
- no duplicated ASCII predicates;
- depends on `stark-ascii` where appropriate;
- pure and deterministic.

---

# 6. Package 3 — `stark-mime`

## Objective

Provide a bounded parser and formatter for HTTP media types.

## Initial supported shape

```text
type/subtype
type/subtype; parameter=value
type/subtype; parameter="quoted value"
```

## Required public types

```stark
struct MediaTypeParameter {
    name: String,
    value: String,
}

struct MediaType {
    type_name: String,
    subtype: String,
    parameters: Vec<MediaTypeParameter>,
}
```

## Required public API

```stark
impl MediaType {
    fn parse(
        input: &String,
        limits: MediaTypeLimits,
    ) -> Result<MediaType, MediaTypeError>;

    fn format(&self) -> String;

    fn is(
        &self,
        type_name: &String,
        subtype: &String,
    ) -> Bool;

    fn parameter(
        &self,
        name: &String,
    ) -> Option<&String>;
}
```

Adjust only where current STARK syntax requires a different accepted form.

## Required limits

```stark
struct MediaTypeLimits {
    max_total_bytes: UInt64,
    max_parameter_count: UInt64,
    max_parameter_name_bytes: UInt64,
    max_parameter_value_bytes: UInt64,
}
```

## Required semantics

- type and subtype use ASCII token validation;
- parameter names use ASCII token validation;
- type, subtype, and parameter-name comparison is ASCII case-insensitive;
- parameter values preserve value content;
- optional whitespace around separators is handled by one frozen rule;
- quoted strings support only the escapes explicitly admitted by v0.1;
- duplicate parameter names are either:
  - preserved in order; or
  - rejected;

The choice must be recorded before implementation. Do not silently collapse duplicates.

- deterministic formatting;
- common constants may be included:
  - `application/json`;
  - `application/octet-stream`;
  - `text/plain`;
  - `application/x-www-form-urlencoded`.

## Required tests

- simple type/subtype;
- uppercase and mixed case;
- parameters;
- quoted values;
- whitespace;
- empty type;
- empty subtype;
- missing slash;
- invalid token byte;
- malformed quote;
- excessive parameters;
- excessive total length;
- case-insensitive `is`;
- parameter lookup;
- deterministic formatting.

## Exclusions

Do not add:

- multipart parsing;
- MIME message parsing;
- content negotiation;
- charset conversion;
- email MIME;
- boundary parsing;
- media-range matching;
- quality values.

## Exit criteria

- strict bounded parser;
- ASCII behaviour delegated to `stark-ascii`;
- no HTTP dependency;
- deterministic formatting;
- pure and cross-engine testable.

---

# 7. Package 4 — `stark-query`

## Objective

Provide strict RFC-style query-string parsing and serialization.

## Semantic rule

`stark-query` does not implement form semantics.

`+` is a literal plus character.

There is no option to reinterpret `+` as space.

## Required public types

```stark
struct QueryPair {
    name: String,
    value: String,
}

struct QueryLimits {
    max_total_bytes: UInt64,
    max_pair_count: UInt64,
    max_name_bytes: UInt64,
    max_value_bytes: UInt64,
}
```

## Required public API

```stark
fn parse(
    input: &String,
    limits: QueryLimits,
) -> Result<Vec<QueryPair>, QueryError>;

fn serialize(
    pairs: &[QueryPair],
    limits: QueryLimits,
) -> Result<String, QueryError>;
```

## Required semantics

- preserve pair order;
- preserve duplicate names;
- split pairs on `&`;
- split name/value at the first `=`;
- missing `=` yields an empty value;
- empty names are either admitted or rejected by one frozen rule;
- percent decoding uses `stark-percent`;
- percent encoding uses `PercentEncodeSet::QueryComponent`;
- `+` remains literal;
- no map-only representation;
- deterministic serialization;
- bounded input and output;
- exact malformed-percent errors are preserved or wrapped without losing offsets.

## Required tests

- one pair;
- multiple pairs;
- duplicate names;
- empty value;
- missing equals;
- empty name;
- empty query;
- percent-encoded names;
- percent-encoded values;
- literal plus;
- malformed escape;
- excessive pair count;
- excessive name/value size;
- deterministic round trip;
- ordering preserved.

## Exclusions

Do not add:

- `+`-as-space;
- form encoding;
- nested object syntax;
- array conventions;
- framework-specific query binding;
- direct `HashMap` conversion as the primary model.

## Exit criteria

- depends on `stark-percent`;
- duplicate keys and ordering preserved;
- no form semantics;
- pure and deterministic.

---

# 8. Optional Package 5 — `stark-form`

## Objective

Implement `application/x-www-form-urlencoded`.

## Start condition

Begin only when:

- `stark-percent` is stable locally;
- `stark-query` is stable locally;
- capacity remains;
- no main-line P1 work is displaced.

## Required semantic distinction

This is the only package in this lane that owns:

```text
+ means space
```

## Required API

```stark
fn parse(
    input: &String,
    limits: FormLimits,
) -> Result<Vec<FormPair>, FormError>;

fn serialize(
    pairs: &[FormPair],
    limits: FormLimits,
) -> Result<String, FormError>;
```

## Required behaviour

- preserve order;
- preserve duplicate names;
- decode `+` to space;
- encode space as `+`;
- use percent encoding for other required bytes;
- strict malformed escape handling;
- bounded pair count and total size;
- deterministic serialization.

## Exclusions

Do not add:

- multipart forms;
- file uploads;
- nested framework conventions;
- typed object binding.

---

# 9. Explicitly excluded from this batch

The following are not part of this quick Gemini lane:

- DNS;
- TLS;
- TCP;
- HTTP client orchestration;
- HTTP server orchestration;
- concurrency;
- async;
- compression;
- ZIP/TAR;
- regex;
- full Unicode;
- timezone database;
- process spawning;
- database drivers;
- logging framework;
- cryptographic provider work.

This list does not override standing roadmap priorities.

Specifically:

- `stark-sha2` retains its existing OPS P0 position;
- bounded `stark-log` v0.1 retains its existing OPS P1 position;
- only a broad logging framework is excluded here.

---

# 10. Deferred packages

## `stark-bounded`

Do not create now.

Use local checked-size and budget logic while implementing actual packages.

Extract a shared package only after the same stable abstraction appears in at least three real packages.

## `stark-result-ext`

Do not implement.

Generic error wrappers conflict with the established package-specific closed error-enum discipline and provide no direct value to P1 or HTTP Track A.

---

# 11. Required package acceptance template

Each package must include:

```text
README.md
SPEC.md
TEST-MATRIX.md
BLOCKERS.md
src/lib.stark
src/tests.stark
examples/ or consumer fixture
```

Use existing repository conventions where paths differ.

## `SPEC.md`

Must record:

- purpose;
- exact public API;
- semantics;
- limits;
- errors;
- determinism;
- exclusions;
- dependencies;
- unsupported behaviour.

## `TEST-MATRIX.md`

Must distinguish:

- implemented;
- locally tested;
- HIR qualified;
- MIR qualified;
- native qualified;
- Tier-1 platform qualified;
- pending;
- blocked.

Do not mark a test qualified merely because source exists.

## `BLOCKERS.md`

For every blocker include:

```text
ID
summary
minimal reproducer
expected behaviour
actual behaviour
failing layer
package impact
workaround, if any
closure requirement
```

---

# 12. Testing rules

## Minimum local checks

Run the repository-equivalent commands for:

- formatting;
- type checking;
- package compilation;
- package tests where supported;
- consumer fixture;
- HIR execution;
- MIR execution;
- native execution where available.

Use the exact CI commands where known.

Do not report a stale result after later edits.

## Full-byte testing

For byte-classification packages:

- test all 256 byte values;
- use table-driven expected results;
- include boundary values explicitly.

## Malformed-input testing

For every parser:

- incomplete input;
- invalid byte;
- oversized input;
- overflow attempt;
- empty components;
- duplicate components;
- exact error position where applicable.

## Cross-engine rule

Any disagreement between HIR, MIR, and native behaviour is a defect.

Do not choose one engine as correct without checking the specification.

---

# 13. Status and landing rules

Each package lands initially as:

```text
IMPLEMENTED_LOCAL
```

Do not claim:

```text
QUALIFIED
TIER1_QUALIFIED
RELEASED
```

until the required Q1/Tier-1 CI evidence exists.

Commit messages must state:

- package implemented;
- tests actually run;
- tests not run;
- known blockers;
- qualification status;
- no compiler/provider changes made.

---

# 14. Escalation triggers

Stop and escalate if implementation requires any of the following:

- compiler modification;
- MIR modification;
- provider synthesis change;
- ABI change;
- new ownership rule;
- new generic mechanism;
- new string representation rule;
- new Unicode rule;
- package-manager change;
- new cross-package visibility mechanism;
- reinterpretation of frozen HTTP semantics.

Also escalate if:

- the required API is not expressible;
- a package compiles only by weakening semantics;
- HIR/MIR/native disagree;
- package tests cannot execute due to tooling;
- a supposedly pure package unexpectedly requires host authority.

---

# 15. HTTP roadmap governance update

These dependencies must be recorded in `HC0-DECISIONS.md`.

Required decision:

```text
DECISION: HTTP Track A consumes shared pure substrate packages.

HC1 stark-url depends on:
- stark-ascii
- stark-percent

HC5 stark-http-core depends on:
- stark-ascii
- stark-mime

HC7 stark-http-parser depends on:
- stark-ascii
- stark-http-core

Query handling depends on:
- stark-percent
- stark-query

Form encoding, when admitted, depends on:
- stark-percent
- stark-query
- stark-form
```

The frozen HTTP dependency graph must be amended accordingly.

Gemini must not independently edit the frozen HTTP roadmap unless explicitly assigned. Report the required graph change for the owning agent to apply.

---

# 16. Assignment boundaries

## Gemini owns

- package code;
- package-local specifications;
- package tests;
- fixed vectors;
- malformed-input corpus;
- consumer examples;
- local status documentation;
- blocker reproduction.

## Gemini does not own

- P1 implementation;
- C7 closure;
- compiler fixes;
- provider APIs;
- DNS;
- TLS;
- CRYPTO0 decisions;
- OPS reprioritisation;
- HTTP roadmap governance;
- release qualification rulings.

The guard that matters:

> This batch keeps Gemini productively occupied with pure substrate. It changes nothing about who owns or implements P1.

---

# 17. Final definition of done

The Gemini lane is complete when:

1. `stark-ascii` is implemented with byte-first APIs;
2. all 256 byte values are tested;
3. `stark-percent` uses a closed encode-set enum;
4. percent decoding is strict and reports exact offsets;
5. `stark-mime` parses and formats bounded media types;
6. `stark-query` preserves order and duplicate names;
7. `stark-query` treats `+` literally;
8. `stark-form`, if implemented, exclusively owns `+`-as-space;
9. no package introduces a host dependency;
10. no compiler, MIR, or provider code is modified;
11. every discovered compiler defect has a reproducer;
12. each package has specification, test matrix, blockers, and consumer evidence;
13. each package is marked `IMPLEMENTED_LOCAL`;
14. no global roadmap priority is changed;
15. HTTP roadmap dependency changes are reported for HC0 governance recording.

---

## Immediate instruction to Gemini

Execute in this order:

```text
1. Implement stark-ascii.
2. Implement stark-percent.
3. Implement stark-mime in parallel where safe.
4. Freeze stark-percent locally.
5. Implement stark-query.
6. Implement stark-form only if capacity remains.
7. Report all blockers without modifying compiler or package semantics.
```
