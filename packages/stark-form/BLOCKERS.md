# stark-form Blockers

## BLOCKER-STARKC-INTERP-VEC-STRUCT-DROP

- **ID**: `BLOCKER-STARKC-INTERP-VEC-STRUCT-DROP`
- **Summary**: Interpreter traps with `use of unavailable value` when unwinding a stack frame containing `Vec<FormPair>`.
- **Failing Layer**: Interpreter (`starkc/src/interp.rs`)
- **Package Impact**: Package type checking (`stark check`) passes clean. Execution of programs holding `Vec<FormPair>` traps at scope end due to interpreter struct-in-vector drop tracking.
- **Expected Behaviour**: Interpreter drops `Vec<FormPair>` clean.
- **Actual Behaviour**: Interpreter raises `use of unavailable value` during local place teardown.
- **Workaround**: None.
- **Closure Requirement**: Fix `interp.rs` value drop handler for `Vec` containing custom struct elements.
