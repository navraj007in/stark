# STARK Phase 8 — Production Tooling Implementation Plan

**Status:** Planning  
**Prerequisite:** Phases 0–5 complete (multi-file, packages, semantic analysis)  
**Objective:** Deliver professional-grade developer tooling making STARK usable without compiler internals knowledge

---

## Phase 8 Overview

Phase 8 transforms STARK from a compiler-centric tool into a complete development platform by delivering:

1. **Language Server (LSP)** — IDE-agnostic completions, diagnostics, navigation
2. **Formatter** — Consistent, specification-driven code style
3. **Test Framework** — Unit and integration testing infrastructure
4. **IDE Integration** — VS Code, JetBrains, and Neovim extensions
5. **Documentation Generator** — Auto-generated API documentation

### Exit Criteria

A new developer can:

```bash
stark new myapp                    # Create project
stark fmt --check                  # Format check
stark test                         # Run tests
stark doc                          # Generate docs
```

And use their editor with full language support (completion, hover, diagnostics, rename, go-to-definition).

---

## Roadmap Context

**Why Phase 8 before Phase 6?**

- **Independence:** Tooling doesn't require native compilation; works equally with interpreter
- **Developer Experience:** Phase 6 (native compilation) work benefits from immediate LSP/formatter/test support
- **Parallel Work:** Language-server engineers can work while compiler engineers build MIR/Cranelift
- **Test Framework Essential:** Phase 6 needs `stark test` for comprehensive validation

**Estimated Timeline:** 12–16 weeks (assuming 2–3 engineers)

---

## Phase 8 Work Packages

### WP8.1 — LSP Server Foundation (4 weeks)

**Objective:** Implement a production-grade Language Server Protocol server supporting Core v1 diagnostics and basic navigation.

**Deliverables:**

1. **LSP Server Core** (`starkc/src/lsp/`)
   - Stdio-based server (VS Code integration)
   - Message routing and error handling
   - Document synchronization (full and incremental)
   - Workspace folder support
   - Configuration and capability negotiation

2. **Basic Features**
   - `textDocument/didOpen` — compile on open
   - `textDocument/didChange` — incremental compilation
   - `textDocument/didClose` — cleanup
   - `textDocument/didSave` — on-save check
   - `textDocument/publishDiagnostics` — all E/W codes with source spans

3. **Hover Support** (`textDocument/hover`)
   - Hover over identifier → show inferred type
   - Hover over function → show signature
   - Hover over trait method → show documentation

4. **Go-to-Definition** (`textDocument/definition`)
   - Jump to struct/enum/fn/trait definitions
   - Cross-file navigation
   - Cross-module navigation (respecting visibility)

5. **References** (`textDocument/references`)
   - Find all uses of a name
   - Exclude private references outside defining module

**Implementation Strategy:**

- Reuse existing `starkc` compiler pipeline (no forking)
- Store compilation state in LSP server (caches between edits)
- Lazy evaluation for expensive operations (hover on first access only)

**Testing:**

- LSP conformance tests (synthetic client)
- Editor integration tests (VS Code driver)
- Stress tests (large files, rapid edits)

**Exit Criteria:**

- All diagnostic codes render correctly
- Hover shows correct types (unit tests verify type inference)
- Go-to-definition works across files and modules
- References correctly filter private names
- Server handles 100+ edits/second without crashing

---

### WP8.2 — Formatter Implementation (3 weeks)

**Objective:** Implement a specification-driven, idempotent formatter matching STARK's style guide.

**Deliverables:**

1. **AST-Based Formatter** (`starkc/src/formatter/`)
   - Walks the parsed AST (not source text)
   - Produces canonical output
   - Guarantees idempotency (format(format(x)) == format(x))
   - No configuration (deterministic by design)

2. **Formatting Rules**
   - Indentation: 4 spaces (tabs disallowed)
   - Line length: soft-wrap at 100 chars, hard-break at 120
   - Operator spacing: binary operators have space on both sides
   - Type annotations: space after `:` in function sigs
   - Imports: sort by path, one per line
   - Trailing commas in multi-line structures
   - Space after `if`/`while`/`for`/`fn` keywords

3. **CLI Command** (`stark fmt`)
   ```bash
   stark fmt                      # Format current package
   stark fmt --check              # Check without modifying
   stark fmt src/lib.stark        # Format single file
   ```

4. **Integration with LSP**
   - `textDocument/formatting` — full-document format
   - `textDocument/rangeFormatting` — range format
   - `textDocument/onTypeFormatting` — format-on-type (after `;`, `}`)

5. **Generated Code Support**
   - Phase 6 (native compilation) generates Rust/C
   - Formatter applies to generated files before output
   - Ensures deterministic, readable generated code

**Implementation Strategy:**

- Traverse HIR, not AST (easier to handle desugared forms)
- Build a document with spacing annotations
- Render to source with proper indentation
- No external dependencies (format manually)

**Testing:**

- Round-trip tests (parse → format → parse → compare AST)
- Idempotency tests (format(format(x)) == format(x))
- Golden file tests (before/after snapshots)
- Large file tests (stress test performance)

**Exit Criteria:**

- `cargo test --test formatter` passes 100 cases
- `stark fmt --check` rejects non-canonical code
- Formatting is deterministic and idempotent
- Handles all Core v1 syntax correctly
- Runs in < 100ms for typical project

---

### WP8.3 — Test Framework (4 weeks)

**Objective:** Deliver `stark test` with unit test, integration test, and example discovery.

**Deliverables:**

1. **Test Discovery** (`starkc/src/test_runner/`)
   - Scan package for test functions (convention: `#[test] fn test_*()`)
   - Scan `tests/` directory for integration tests
   - Scan `examples/` directory for example programs
   - Collect into a test suite manifest

2. **Test Annotations**
   ```stark
   #[test]
   fn test_addition() {
       assert_eq(2 + 2, 4);
   }

   #[test]
   #[ignore]
   fn test_slow_operation() {
       // skipped by default
   }
   ```

3. **Assertion Macros** (in stdlib `std::test`)
   ```stark
   assert(condition)                // Panic if false
   assert_eq(left, right)           // Panic if not equal
   assert_ne(left, right)           // Panic if equal
   assert_matches(value, pattern)   // Panic if pattern doesn't match
   ```

4. **CLI Commands**
   ```bash
   stark test                       # Run all tests
   stark test test_name             # Run single test
   stark test --ignored             # Run only #[ignore] tests
   stark test --show-output         // Show println! output
   stark test -- --seed 42          // Deterministic random seed
   ```

5. **Test Output Format**
   ```
   running 42 tests

   test module1::test_basic ... ok
   test module2::test_edge_case ... ok (15ms)
   test module3::test_slow ... ignored

   test result: ok. 40 passed; 0 failed; 2 ignored; 15ms total
   ```

6. **Integration Tests** (in `tests/` directory)
   - Each file is a standalone STARK program
   - Exit code 0 = pass, non-zero = fail
   - Capture stdout/stderr for comparison

7. **Example Validation** (in `examples/` directory)
   - Each example must compile and run
   - Capture output to validate correctness
   - Allows `#[ignore]` for expensive examples

**Implementation Strategy:**

- Reuse compiler pipeline for each test
- Run tests sequentially (no parallelism yet)
- Cache compilation results within test run
- Interpreter executes; Phase 6 will enable native execution

**Testing:**

- Self-tests: test runner tests itself
- Golden output tests: verify output format
- Determinism tests: same seed produces same results

**Exit Criteria:**

- `stark test` discovers and runs all test functions
- Test discovery finds all test annotations
- Assertions panic with clear messages
- Output format matches specification
- `cargo test --test test_framework` passes

---

### WP8.4 — VS Code Extension (3 weeks)

**Objective:** Deliver a production-ready VS Code extension with full language support.

**Deliverables:**

1. **Extension Skeleton** (`editors/vscode/`)
   - `package.json` with language definition
   - `extension.ts` — extension entrypoint
   - LSP client initialization
   - Command registration

2. **Language Definition**
   - TextMate grammar for syntax highlighting
   - Bracket pair colorization
   - Indentation rules
   - Comment rules

3. **Commands**
   - `stark.check` — run `stark check`
   - `stark.run` — run `stark run`
   - `stark.format` — run `stark fmt`
   - `stark.generateDocs` — run `stark doc`
   - `stark.toggleTensorMode` — enable/disable tensor extension

4. **Status Bar**
   - Display LSP server status
   - Show current project name
   - Compilation status indicator

5. **Keybindings**
   ```
   Ctrl+Shift+F → Format document
   F12 → Go to definition
   Shift+F12 → Find references
   Ctrl+K Ctrl+I → Hover
   ```

6. **Settings**
   ```json
   {
     "stark.extensionEnabled": true,
     "stark.lspLogLevel": "info",
     "stark.formatOnSave": true,
     "stark.tensorExtensionEnabled": false,
     "stark.testOnSave": false
   }
   ```

7. **Troubleshooting**
   - Command palette: "STARK: Show LSP Output"
   - Command palette: "STARK: Restart Language Server"
   - Help panel with common issues

**Testing:**

- Install and run against test projects
- Verify diagnostics render inline
- Verify hover popups show correct types
- Verify go-to-definition navigates correctly
- Test on small, medium, and large projects

**Exit Criteria:**

- Extension publishes to VS Code Marketplace
- 10,000+ downloads
- Rating ≥ 4.5 stars
- LSP responds in < 500ms
- No memory leaks in long sessions

---

### WP8.5 — Documentation Generator (3 weeks)

**Objective:** Generate searchable, cross-linked API documentation from source code.

**Deliverables:**

1. **Doc Comments** (in language)
   ```stark
   /// Add two numbers.
   /// 
   /// # Arguments
   /// * `a` — first number
   /// * `b` — second number
   ///
   /// # Returns
   /// The sum of a and b.
   ///
   /// # Example
   /// ```stark
   /// assert_eq(add(2, 3), 5);
   /// ```
   pub fn add(a: Int32, b: Int32) -> Int32 {
       a + b
   }
   ```

2. **Doc Extractor** (`starkc/src/doc_gen/`)
   - Extract doc comments from all public items
   - Parse markdown + STARK code blocks
   - Validate example code compiles

3. **HTML Generator**
   - Generate static HTML site
   - One page per module/type
   - Search index (JSON)
   - Cross-references (hyperlinks)
   - Syntax highlighting for code examples

4. **CLI Command**
   ```bash
   stark doc                       # Generate docs/
   stark doc --open                # Open in browser
   stark doc --output ./target/doc # Custom location
   ```

5. **Documentation Layout**
   ```
   docs/
     index.html                    # Home page
     search.html                   # Search interface
     search.json                   # Search index
     std/
       option/
         index.html                # Option type docs
       vec/
         index.html                # Vec type docs
     mymodule/
       index.html                  # Module docs
   ```

6. **Search**
   - Full-text search over API names
   - Fuzzy matching
   - Namespace filtering

**Testing:**

- Generate docs for std library
- Validate all cross-references work
- Verify examples in docs compile
- Check HTML validity

**Exit Criteria:**

- `stark doc` generates valid HTML
- All public items documented
- Cross-references are correct
- Search works with 1000+ items
- Examples in docs all compile

---

### WP8.6 — IDE Extensions (Neovim, JetBrains) (2 weeks)

**Objective:** Extend LSP support to other popular editors.

**Deliverables:**

1. **Neovim Integration** (`editors/nvim/`)
   - `init.lua` configuration
   - Diagnostic display
   - Go-to-definition keybindings
   - Hover support

2. **JetBrains Plugin** (`editors/jetbrains/`)
   - LSP client in plugin SDK
   - Syntax highlighting
   - Plugin marketplace submission

3. **Generic LSP Instructions**
   - Documentation for other editors (Sublime, Emacs, etc.)
   - LSP connection instructions
   - Troubleshooting guide

**Testing:**

- Test in Neovim and IntelliJ IDEA
- Verify diagnostics appear
- Verify navigation works

**Exit Criteria:**

- Neovim and JetBrains both support STARK
- Documentation guides users through setup
- Basic features work in each editor

---

## Phase 8 Testing Strategy

### Test Suites

| Suite | Scope | Coverage |
|---|---|---|
| **LSP Conformance** | Protocol messages | 100% of implemented features |
| **Formatter** | AST → source | Idempotency + all syntax forms |
| **Test Framework** | Discovery + execution | All assertion types + annotations |
| **VS Code** | Extension lifecycle | Install, configure, use |
| **Doc Generator** | HTML output | All item types + cross-refs |

### CI/CD Pipeline

```bash
# Formatter
cargo test --test formatter

# Test framework
cargo test --test test_framework

# LSP
cargo test --test lsp_conformance

# Integration (VS Code)
npm test (in editors/vscode)
```

### Manual Testing Checklist

- [ ] Create new project with `stark new`
- [ ] Open in VS Code, verify syntax highlighting
- [ ] Hover over types, verify popup
- [ ] Go to definition on imports, verify navigation
- [ ] Make intentional error, verify diagnostic appears
- [ ] Run `stark fmt`, verify output is canonical
- [ ] Run `stark test`, verify test discovery
- [ ] Generate docs with `stark doc`, view in browser
- [ ] Search docs, verify search works
- [ ] Test on file with 1000+ lines (performance)
- [ ] Rapid edits (100+/sec), verify no crashes
- [ ] Close and reopen file, verify state restored

---

## Phase 8 Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| LSP performance regresses under load | Medium | Profile regularly; cache compilation results; limit re-compilation |
| Formatter breaks existing code | Low | Idempotency tests + golden file tests + real project validation |
| VS Code extension marketplace rejection | Low | Follow submission guidelines; test thoroughly before submission |
| Test framework discovers unwanted items | Medium | Clear naming convention (`test_*`) + allowlist in manifest |
| Documentation generator missing items | Medium | Audit public API; enforce that all public items have docs |
| Dependency on external LSP library | High | **Mitigation:** Hand-write LSP (no external deps) or use well-maintained `tower-lsp` |

---

## Phase 8 Exit Criteria (Comprehensive)

### Functionality

- ✅ LSP server implements all Core v1 language features
- ✅ Formatter is deterministic and idempotent
- ✅ Test framework discovers and runs all tests
- ✅ VS Code extension is production-ready
- ✅ Documentation generator produces valid HTML
- ✅ All Phases 0–5 code compiles without warnings

### Quality

- ✅ 95%+ test coverage for all tooling components
- ✅ Zero memory leaks in long-running processes (LSP server)
- ✅ Performance: LSP responds in < 500ms, formatter in < 100ms
- ✅ All public APIs documented
- ✅ Security review: no arbitrary code execution, no file traversal escapes

### Usability

- ✅ New developer can complete `stark new → fmt → test → doc` workflow
- ✅ VS Code extension installs from Marketplace with one click
- ✅ All features work offline (no network calls)
- ✅ Error messages are actionable and point to fixes
- ✅ Documentation is complete and searchable

### Release

- ✅ VS Code extension published to Marketplace
- ✅ Neovim and JetBrains extensions available
- ✅ Release notes document all features
- ✅ Installation guide included in main README
- ✅ Changelog entries for all user-visible changes

---

## Phase 8 to Phase 6 Handoff

When Phase 8 is complete and Phase 6 begins:

**From Phase 8:**

- Formatter can validate generated Rust/C code
- Test framework can run Phase 6 native-compiled binaries
- LSP can provide diagnostics for Phase 6 MIR generation errors

**Phase 6 Additions:**

- Extend formatter to handle generated code style
- Extend test framework to support native binary execution
- Extend LSP with MIR-level diagnostics (if applicable)

---

## Recommended Team Structure

| Role | WP Lead | Support |
|---|---|---|
| **LSP Engineer** | WP8.1 | WP8.4 (protocol integration) |
| **Formatter Engineer** | WP8.2 | WP8.4 (VS Code command) |
| **Test Framework Engineer** | WP8.3 | WP8.4 (test discovery in runner) |
| **IDE/Extension Engineer** | WP8.4–8.6 | All WPs (integration) |

**Total:** 3–4 engineers, 12–16 weeks

---

## Success Metrics

### Adoption

- VS Code extension: 10,000+ downloads in first month
- GitHub stars on extension repo: 500+
- Community feedback: 90%+ positive sentiment

### Quality

- Bug reports: < 5 per month
- Mean time to fix: < 1 week
- Crash-free sessions: 99.9%

### Satisfaction

- User survey: 4.5+ / 5.0 rating
- Feature request throughput: < 20 per month
- "Makes STARK enjoyable to use" ← post-Phase 8 goal

---

## Conclusion

Phase 8 transforms STARK from a research compiler into a professional development platform. By delivering LSP, formatter, testing, and IDE integration **before** native compilation, we ensure that Phase 6 engineers have excellent tools from day one, and external developers can start using STARK immediately after Phase 5.

**Next:** Phases 0–5 ✅, Phase 8 Plan ✅. Ready to execute.
