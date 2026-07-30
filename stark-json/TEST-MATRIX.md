# stark-json v0.1 Test Matrix

| ID | Category | Input/Fixture | Expected | Engines | Status |
| --- | --- | --- | --- | --- | --- |
| API-001 | API | package check | frozen public package resolves | HIR/check | PASS |
| API-002 | API | `JsonValue::Bool`, `JsonValue::String` | primitive-named variants parse and check | parser/HIR/check | PASS |
| PRIM-001 | PRIMITIVES | `null true false` as separate cases | parse success | test runner | PASS |
| NUM-001 | NUMBERS_VALID | `0`, `-12.34e+56` | parse success, lexeme retained through `number_raw` | test runner/consumer | PASS |
| NUM-002 | NUMBERS_INVALID | `01`, `1.`, `1e+` | exact error positions | test runner | PASS |
| ARR-001 | ARRAYS | `[1,true,null]` | parse success | test runner | PASS |
| OBJ-001 | OBJECTS | `{"a":1}` | parse success | test runner | PASS |
| DUP-001 | DUPLICATE_KEYS | `{"a":1,"a":2}` | duplicate key position | test runner | PASS |
| STR-001 | UNICODE | `"\u0061"`, `"\u00E9"`, `"\u20AC"` | escapes accepted and decoded as UTF-8 | test runner | PASS |
| STR-002 | SURROGATES | `"\uD83D\uDE00"`, lone high/low surrogate escapes | valid pair decodes; lone surrogates error | test runner/consumer | PASS |
| LIM-001 | LIMITS | `max_input_bytes = 2`, input `123` | limit error | test runner | PASS |
| ENC-001 | ENCODING | consumer `encode(&value)` | compact canonical JSON without consuming input | interpreter/native | PASS |
| CROSS-001 | CROSS_PACKAGE | `stark-json-consumer` | dependency import and run | check/run | PASS |
| FMT-001 | FORMATTER | `stark fmt --check` | canonical formatting | formatter | PASS |
| FIX-001 | FIXTURES | `fixtures/valid` | 17 classified valid cases | checked in | PASS |
| FIX-002 | FIXTURES | `fixtures/invalid` | 32 classified invalid cases | checked in | PASS |
| NATIVE-001 | THREE_ENGINE | `stark build --no-build-cache` on consumer | native evidence | native build | PASS |
| TIER1-001 | PLATFORM | macOS arm64 | check/test/fmt/native consumer | local | PASS |
| TIER1-002 | PLATFORM | Linux x64 | check/test/fmt/native consumer | not run | PENDING |
| TIER1-003 | PLATFORM | Windows x64 | check/test/fmt/native consumer | not run | PENDING |

The frozen package behavior is implemented and locally qualified on macOS arm64. The complete
work-package matrix remains open until Linux x64 and Windows x64 Tier-1 runs are recorded.
