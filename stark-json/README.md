# stark-json v0.1

`stark-json` is a pure-STARK JSON package partial implementation for the `STARK JSON v0.1`
work package.

## Scope

Implemented and checked under the current compiler:

- package manifest with no dependencies;
- frozen public type/function names, including `JsonValue::Bool` and `JsonValue::String`;
- recursive `JsonValue` shape using ordered `Vec<JsonMember>` objects;
- parser entry points `parse` and `parse_with_limits`;
- byte-oriented recursive descent for primitives, numbers, arrays, objects, whitespace, duplicate
  checks, position tracking, and selected limits;
- decoded string preservation for direct UTF-8 input, simple escapes, and ASCII `\u00XX` escapes;
- 17 valid fixtures and 32 invalid fixtures;
- package-local tests and a sibling cross-package consumer.

Current compiler/runtime blockers prevent a complete compliant implementation:

- `JsonValue::Bool(Bool)` and `JsonValue::String(String)` now parse and check after the compiler
  accepts primitive type keywords as enum variant declaration names and path segments.
- The compiler has no supported ref-binding pattern for non-Copy enum payloads, so
  `encode(value: &JsonValue)` cannot inspect recursive `String`, `Vec`, or object payloads without
  move errors. The function currently preserves the required signature but returns `null`.
- There is no public scalar-from-codepoint constructor for decoding non-ASCII `\uXXXX` escapes into
  actual Unicode scalar values. Non-ASCII Unicode escapes currently return `InvalidUtf8` rather than
  silently producing the wrong string.

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

Recommended status: `PARTIAL - WAITING_COMPILER_RUNTIME`.
