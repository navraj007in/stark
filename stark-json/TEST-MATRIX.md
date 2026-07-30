# stark-json v0.1 Test Matrix

| ID | Category | Input/Fixture | Expected | Engines | Status |
| --- | --- | --- | --- | --- | --- |
| API-001 | API | package check | frozen public package resolves | HIR/check | PASS |
| API-002 | API | `JsonValue::Bool`, `JsonValue::String` | primitive-named variants parse and check | parser/HIR/check | PASS |
| PRIM-001 | PRIMITIVES | `null true false` as separate cases | parse success | test runner | PASS |
| NUM-001 | NUMBERS_VALID | `0`, `-12.34e+56` | parse success, lexeme retained internally | test runner | PARTIAL |
| NUM-002 | NUMBERS_INVALID | `01`, `1.`, `1e+` | exact error positions | test runner | PASS |
| ARR-001 | ARRAYS | `[1,true,null]` | parse success | test runner | PASS |
| OBJ-001 | OBJECTS | `{"a":1}` | parse success | test runner | PASS |
| DUP-001 | DUPLICATE_KEYS | `{"a":1,"a":2}` | duplicate key position | test runner | PASS |
| STR-001 | UNICODE | `"\u0061"` | escape accepted and decoded as `a` | test runner | PASS |
| STR-002 | SURROGATES | lone high/low surrogate escapes | exact error positions | test runner | PASS |
| LIM-001 | LIMITS | `max_input_bytes = 2`, input `123` | limit error | test runner | PASS |
| ENC-001 | ENCODING | consumer `encode(&value)` | compact JSON | interpreter | BLOCKED: borrowed non-Copy enum payload binding |
| CROSS-001 | CROSS_PACKAGE | `stark-json-consumer` | dependency import and run | check/run | PASS |
| FMT-001 | FORMATTER | `stark fmt --check` | canonical formatting | formatter | PASS |
| FIX-001 | FIXTURES | `fixtures/valid` | 17 classified valid cases | checked in | PASS |
| FIX-002 | FIXTURES | `fixtures/invalid` | 32 classified invalid cases | checked in | PASS |
| NATIVE-001 | THREE_ENGINE | `stark build` on consumer | native evidence | native build | BLOCKED |

The complete work-package matrix remains open because the current compiler/runtime blockers prevent
compliant borrowed recursive encoding and decoded Unicode scalar storage. The previous public API
spelling blocker is resolved.
