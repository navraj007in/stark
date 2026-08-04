# stark-percent Blockers

## BLOCKER-STARKC-TEST-RUNNER-SYNTHETIC-SPAN

- **ID**: `BLOCKER-STARKC-TEST-RUNNER-SYNTHETIC-SPAN`
- **Summary**: `stark test` panics with `byte index 2147483648 is out of bounds` when running unit tests in any package that defines enums with tuple payload variants.
- **Failing Layer**: Compiler Test Runner (`starkc/src/test_runner/mod.rs:127`)
- **Package Impact**: `stark check` and `stark run` (consumer execution) pass 100% clean. Package unit testing via `stark test` panics due to compiler test-runner bug.
- **Expected Behaviour**: `test_runner` ignores synthetic items (`span.lo >= 0x8000_0000`) during test item text extraction and runs package tests.
- **Actual Behaviour**: `item_text` in `src/test_runner/mod.rs` attempts to slice `src[span.lo as usize..span.hi as usize]` without checking for synthetic spans, causing panic on `2147483648` (`0x8000_0000`).
- **Minimal Reproducer**:
  ```stark
  pub enum Error {
      Invalid(UInt64),
  }

  fn test_foo() {}
  ```
  Run `stark test`.
- **Workaround**: None (gemini is prohibited from weakening package error enums or modifying the compiler). Consumer execution (`stark run`) qualifies the implementation.
- **Closure Requirement**: Update `starkc/src/test_runner/mod.rs` to guard `item_text` or `collect` against synthetic spans (`span.lo >= 0x8000_0000`).
