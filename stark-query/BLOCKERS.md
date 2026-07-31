# stark-query Blockers

## BLOCKER-STARKC-INTERP-VEC-STRUCT-DROP

- **ID**: `BLOCKER-STARKC-INTERP-VEC-STRUCT-DROP`
- **Summary**: Interpreter traps with `use of unavailable value` when unwinding a stack frame containing a `Vec<CustomStruct>` (`Vec<QueryPair>`).
- **Failing Layer**: Interpreter (`starkc/src/interp.rs`)
- **Package Impact**: Package type checking (`stark check`) passes clean. Execution of programs holding `Vec<QueryPair>` traps at scope end due to interpreter struct-in-vector drop tracking.
- **Expected Behaviour**: Interpreter drops `Vec<QueryPair>` clean.
- **Actual Behaviour**: Interpreter raises `use of unavailable value` during local place teardown.
- **Workaround**: None (gemini is prohibited from weakening package struct definitions or modifying the compiler).
- **Closure Requirement**: Fix `interp.rs` value drop handler for `Vec` containing custom struct elements.
