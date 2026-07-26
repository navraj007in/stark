# stark-json v0.1

`stark-json` is a pure-STARK JSON package scaffold and partial implementation for the
`STARK JSON v0.1` work package.

## Scope

Implemented and checked under the current compiler:

- package manifest with no dependencies;
- frozen public type/function names except the `JsonValue` payload variant spelling noted below;
- recursive `JsonValue` shape using ordered `Vec<JsonMember>` objects;
- parser entry points `parse` and `parse_with_limits`;
- byte-oriented recursive descent for primitives, numbers, arrays, objects, whitespace, duplicate
  checks, position tracking, and selected limits;
- package-local tests and a sibling cross-package consumer.

Current compiler/runtime blockers prevent a complete compliant implementation:

- `JsonValue::Bool(Bool)` and `JsonValue::String(String)` are rejected by the frontend because
  variant names collide with built-in type names. This package uses `BoolValue` and `StringValue`.
- The compiler has no supported ref-binding pattern for non-Copy enum payloads, so
  `encode(value: &JsonValue)` cannot inspect recursive `String`, `Vec`, or object payloads without
  move errors. The function currently preserves the required signature but returns `null`.
- There is no public scalar-from-codepoint constructor for decoding non-ASCII `\uXXXX` escapes into
  actual Unicode scalar values.
- To keep package tests executable, parsed string contents currently validate syntax and limits but
  return an empty decoded string.

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
