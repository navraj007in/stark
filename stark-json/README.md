# stark-json v0.1

`stark-json` is a pure-STARK JSON package implementation for the `STARK JSON v0.1`
work package, qualified locally on macOS arm64.

## Scope

Implemented and checked under the current compiler:

- package manifest with no dependencies;
- frozen public type/function names, including `JsonValue::Bool` and `JsonValue::String`;
- recursive `JsonValue` shape using ordered `Vec<JsonMember>` objects;
- parser entry points `parse` and `parse_with_limits(input, limits: JsonLimits)`;
- byte-oriented recursive descent for primitives, numbers, arrays, objects, whitespace, duplicate
  checks, position tracking, and selected limits;
- decoded string preservation for direct UTF-8 input, simple escapes, BMP `\uXXXX` escapes, and
  valid surrogate pairs;
- deterministic compact encoder with borrowed, non-consuming traversal;
- 17 valid fixtures and 32 invalid fixtures;
- 10 package-local tests and a sibling cross-package native consumer.

Compiler/runtime work completed for this package:

- `JsonValue::Bool(Bool)` and `JsonValue::String(String)` now parse and check after the compiler
  accepts primitive type keywords as enum variant declaration names and path segments.
- Borrowed matches over non-Copy enum payloads now bind payloads by reference, allowing
  `encode(value: &JsonValue)` to inspect recursive strings, arrays, objects, and numbers without
  consuming the input.
- `Char::from_u32` is available for validated Unicode scalar construction, so non-ASCII
  `\uXXXX` escapes and valid surrogate pairs decode to UTF-8 strings.

## API

Public items:

- `JsonValue`
- `JsonMember`
- `JsonNumber`
- `JsonLimits`
- `JsonErrorKind`
- `JsonError`
- `default_limits`
- `parse`
- `parse_with_limits`
- `encode`
- `number_raw`
- `member_key`
- `member_value`

## Defaults

- `max_input_bytes`: 16,777,216
- `max_depth`: 128
- `max_string_bytes`: 4,194,304
- `max_array_items`: 1,000,000
- `max_object_members`: 1,000,000
- `max_number_bytes`: 1,024

## Status

Recommended status: `PARTIAL - PLATFORM_QUALIFICATION_PENDING`.

Local macOS arm64 package, formatter, compiler, and native consumer checks pass. Linux x64 and
Windows x64 Tier-1 qualification has not been run in this workspace.
