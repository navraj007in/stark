# WP-C8 — Semantic Language Services

**Assigned implementer:** Codex
**Parallel track:** Gate C7 may proceed concurrently under a separate owner.
**Starting point:** exact repository head agreed by both tracks after Gate C6 closure.
**Gate dependency:** C2 closed. C8 does not depend on C7 or native compilation.
**Primary outcome:** compiler-backed editor semantics through the STARK language server and VS Code extension.

---

## 0. Directive

Implement Gate C8 as a sequence of bounded work packages.

Do not treat protocol responses, cursor-coordinate echoes, text search, parser-only guesses, or placeholder data as completed language-service support.

Every semantic result must be derived from the shared compiler analysis pipeline established by C2:

```text
source files
    ↓
project/package discovery
    ↓
parse
    ↓
resolve
    ↓
type check
    ↓
structured compiler analysis
    ↓
LSP query
    ↓
editor response
```

Do not duplicate parsing, name resolution or type inference in the editor extension.

C8 is permitted to improve the shared compiler query APIs where necessary, but it must not:

* change STARK language semantics;
* broaden the normative standard-library surface;
* modify native lowering or backend behaviour;
* reopen Gate C6 decisions;
* implement C7 backend, optimisation, performance or release work;
* redesign package version resolution;
* generalise artifact providers;
* add tensor features beyond correctly exposing already-resolved extension symbols.

---

## 1. Parallel-work isolation

C7 and C8 are active concurrently. Prevent silent cross-track interference.

### 1.1 C8-owned areas

Codex owns, subject to the repository lease protocol:

```text
language-server / LSP implementation
VS Code extension integration
compiler semantic-query APIs required by LSP
LSP protocol tests
editor integration tests
C8 work-package documents
C8 evidence and exit report
```

Likely affected paths include, but must be verified against the repository map:

```text
starkc/src/lsp/**
starkc/src/project/**
starkc/src/analysis/**
starkc/src/diag/**
starkc/tests/*lsp*
editors/vscode/**
STARKLANG/docs/compiler/work-packages/WP-C8*.md
starkc/docs/compiler/evidence/c8/**
starkc/docs/compiler/C8-exit-report.md
```

Do not assume these paths exist exactly as written. Inspect first.

### 1.2 Shared files

Files such as these are potentially shared with C7:

```text
COMPILER-STATE.md
COMPILER-ROADMAP.md
COMPILER-CHARTER.md
Cargo.toml / Cargo.lock
shared CI workflows
compiler entry-point modules
project/package analysis
structured diagnostic types
```

Before editing a shared file:

1. record a lease in the integration ledger;
2. state the exact reason and expected change;
3. avoid broad formatting or unrelated cleanup;
4. release the lease immediately after the bounded commit;
5. rebase or reconcile against C7 before qualification.

### 1.3 Prohibited parallel behaviour

Do not:

* overwrite C7 state updates;
* modify backend or MIR files for convenience;
* introduce editor-only semantic behaviour;
* make the VS Code extension shell out through a second compiler path when an in-process project API exists;
* hide conflicts by accepting whichever result arrives last;
* perform repository-wide formatting during the parallel period.

---

## 2. Baseline audit — WP-C8.0

Before implementing features, establish the actual current state.

### 2.1 Inventory

Audit:

* language-server binary and entry point;
* JSON-RPC transport;
* initialisation capabilities currently advertised;
* document open/change/save/close handling;
* project-root and package discovery;
* source-text storage and version tracking;
* existing compiler invocation paths;
* diagnostic generation and publication;
* hover;
* definition;
* references;
* completion;
* signature help;
* rename;
* document symbols;
* workspace symbols;
* semantic tokens;
* inlay hints;
* formatting integration;
* VS Code extension commands and activation;
* subprocess use;
* placeholder or hard-coded responses;
* tests for each feature.

### 2.2 Capability truthfulness

Produce a table:

| Capability        | Advertised | Implemented semantically | Placeholder | Tested | Editor validated |
| ----------------- | ---------: | -----------------------: | ----------: | -----: | ---------------: |
| Diagnostics       |            |                          |             |        |                  |
| Hover             |            |                          |             |        |                  |
| Definition        |            |                          |             |        |                  |
| References        |            |                          |             |        |                  |
| Completion        |            |                          |             |        |                  |
| Signature help    |            |                          |             |        |                  |
| Rename            |            |                          |             |        |                  |
| Document symbols  |            |                          |             |        |                  |
| Workspace symbols |            |                          |             |        |                  |
| Semantic tokens   |            |                          |             |        |                  |
| Inlay hints       |            |                          |             |        |                  |
| Formatting        |            |                          |             |        |                  |

The server must not advertise an unsupported capability merely because a protocol handler exists.

### 2.3 Baseline evidence

Add raw JSON-RPC tests that record present behaviour before repair.

The baseline must identify:

* responses that are semantically correct;
* responses that are placeholders;
* responses based on text matching;
* responses derived from stale document versions;
* features that panic, hang or return invalid protocol data;
* duplicate compiler analysis;
* duplicate diagnostics;
* package and extension failures.

### 2.4 Deliverable

Create:

```text
STARKLANG/docs/compiler/work-packages/WP-C8.0-BASELINE.md
```

C8.0 closes only when every advertised capability is classified honestly.

---

## 3. Shared semantic-query foundation — WP-C8.0A

Do not implement each LSP feature as an independent walk over compiler internals.

Build or repair a shared query layer.

### 3.1 Project snapshot

Define an immutable or versioned analysis snapshot containing enough information for semantic queries:

```text
workspace/project identity
package graph
source-file identity
document version
parsed syntax
resolved symbol identities
type-check result
structured diagnostics
definition locations
reference occurrences
expression/type information
extension configuration
```

The exact internal representation may differ, but query results must correspond to one coherent snapshot.

### 3.2 Required query primitives

Provide stable internal APIs equivalent to:

```text
diagnostics_for_file(file_id)
symbol_at(file_id, position)
definition_of(symbol_id)
references_of(symbol_id)
type_at(file_id, position)
callable_signature_at(file_id, position)
visible_symbols_at(file_id, position)
members_of(type_id, receiver_context)
document_symbols(file_id)
workspace_symbols(query)
rename_edits(symbol_id, new_name)
semantic_classification(file_id)
```

Do not expose unstable compiler implementation details directly to protocol handlers where a query abstraction is practical.

### 3.3 Identity rule

Definitions, references and rename must use resolved symbol identity.

The following are forbidden as semantic implementations:

* matching equal source text;
* searching every file for the same identifier spelling;
* guessing by nearest declaration;
* matching method names without receiver/type resolution;
* treating imports or re-exports as unrelated textual occurrences.

### 3.4 Position mapping

Centralise conversion between:

* UTF-8 byte offsets;
* compiler spans;
* LSP UTF-16 positions;
* source lines and columns.

Test:

* ASCII;
* Punjabi and Hindi text in comments or strings;
* emoji;
* combining characters;
* multibyte identifiers if the grammar permits them;
* positions at line endings;
* positions outside the current document;
* stale versions.

No feature should maintain its own coordinate conversion.

---

## 4. WP-C8.1 — Diagnostic publication

Implement compiler-backed `publishDiagnostics`.

### 4.1 Triggers

Publish diagnostics on:

* document open;
* document change;
* document save;
* relevant dependency-file change;
* project configuration change;
* extension configuration change.

Clear diagnostics on close where appropriate.

### 4.2 Version discipline

Every analysis request and publication must be associated with the document version that produced it.

Required behaviour:

```text
version N analysis starts
version N+1 arrives
version N analysis completes
→ version N result must not replace version N+1 state
```

Implement stale-result suppression.

### 4.3 Diagnostic fidelity

Preserve:

* severity;
* stable diagnostic code;
* primary range;
* message;
* related information;
* secondary file locations;
* source/compiler identity;
* extension-specific context where enabled.

Cross-file errors must attach related information to the correct files.

### 4.4 Duplication prevention

If an existing subprocess checker remains temporarily:

* choose one publication authority; or
* clearly separate the two sources;
* never publish duplicate copies of the same diagnostic.

The final C8 state should use the shared project-analysis pipeline.

### 4.5 Tests

At minimum:

* syntax error appears and clears after correction;
* type error appears and clears;
* borrow error publishes the intended code and range;
* error in imported module appears with correct file;
* package dependency error appears at correct source;
* old-version diagnostics are suppressed;
* closing a file clears or retains diagnostics according to documented policy;
* Core-only mode rejects extension syntax;
* extension-enabled mode analyses extension syntax;
* malformed request cannot crash the server.

### 4.6 Done when

Diagnostics visible in VS Code agree with command-line/compiler diagnostics for the same project snapshot, modulo documented presentation differences.

---

## 5. WP-C8.2 — Hover and signature rendering

Implement real hover derived from compiler semantics.

### 5.1 Hover targets

Support:

* locals;
* parameters;
* constants;
* functions;
* methods;
* fields;
* structs;
* enums;
* enum variants;
* traits;
* type aliases;
* generic parameters;
* imported symbols;
* re-exported symbols;
* package symbols;
* inferred expression types where useful;
* enabled tensor/model symbols already known to the compiler.

### 5.2 Hover content

Where applicable, render:

```text
qualified symbol name
kind
resolved type
function or method signature
generic parameters and substitutions
receiver form
source module/package
documentation summary if compiler metadata exists
```

Do not invent documentation by parsing nearby comments unless the compiler owns that association.

### 5.3 Signature rendering

Use one renderer shared by:

* hover;
* completion detail;
* signature help;
* symbol display where appropriate.

The renderer must support:

* generic functions;
* trait methods;
* associated functions;
* reference and mutable-reference types;
* tuples;
* arrays;
* function types;
* nested generic types;
* associated types where available.

### 5.4 Tests

Include:

* inferred local;
* generic function after substitution;
* inherent method;
* trait method;
* imported function;
* re-export;
* field;
* enum variant;
* symbol shadowing;
* unresolved identifier returns no false hover;
* stale source does not return a hover from the previous version.

### 5.5 Done when

No advertised hover response is a cursor-coordinate string, token echo or parser-only guess.

---

## 6. WP-C8.3 — Definition and references

### 6.1 Go to definition

Support:

* same-file declarations;
* cross-file module declarations;
* cross-package public declarations;
* imports;
* re-exports;
* functions and methods;
* fields;
* types;
* traits;
* enum variants;
* generic parameters where useful;
* external package source locations when available.

Define and document behaviour for an imported symbol:

* go to import site;
* go to original declaration;
* or expose declaration links containing both.

Choose one consistent policy.

### 6.2 References

Return exact references based on resolved identity.

Support:

* declaration inclusion flag;
* local shadowing;
* same spelling referring to different symbols;
* re-exports;
* method calls;
* field accesses;
* trait and implementation relationships where compiler identity makes them available;
* references across package boundaries inside the workspace.

### 6.3 Method identity

Do not equate methods by name alone.

These must remain distinct:

```text
TypeA::len
TypeB::len
TraitA::run
TraitB::run
inherent run
trait-provided run
```

### 6.4 Tests

Include adversarial cases:

* two locals with same name in nested scopes;
* same function name in two modules;
* imported alias;
* re-export chain;
* two methods with same spelling on different types;
* trait method versus inherent method;
* field and local with same spelling;
* package dependency symbol;
* deleted or renamed declaration invalidates old result.

### 6.5 Done when

Definition and reference results remain correct under shadowing, aliases, imports, re-exports and same-spelling symbols.

---

## 7. WP-C8.4 — Completion and signature help

### 7.1 Completion contexts

Implement context-sensitive completion for:

* local scope;
* function parameters;
* module members;
* package paths;
* imported APIs;
* fields after member access;
* methods valid for the receiver type;
* enum variants;
* associated functions;
* types in type positions;
* values in expression positions;
* extension symbols only when the relevant extension is enabled.

### 7.2 Filtering

Completion must respect:

* visibility;
* package boundaries;
* current scope;
* shadowing;
* receiver type;
* generic applicability where known;
* Core-only versus extension-enabled mode.

Do not suggest private external-package items.

### 7.3 Completion edits

Support where practical:

* replacement ranges;
* deterministic ordering;
* insertion text;
* additional import edits only if import resolution is safe;
* detail/signature rendering;
* symbol kind.

Do not add speculative automatic imports in the first implementation unless identity and collision handling are reliable.

### 7.4 Signature help

Provide:

* active callable;
* parameter list;
* active parameter;
* generic substitutions where known;
* method receiver excluded or displayed consistently;
* nested-call handling.

Test commas inside nested expressions and generic arguments so the active parameter is not derived through naive character counting.

### 7.5 Tests

Include:

* local completion;
* module path;
* field completion;
* receiver method completion;
* enum variant completion;
* private item excluded;
* wrong-type method excluded;
* extension symbol absent in Core-only mode;
* active parameter in nested call;
* overload-like candidate ambiguity handled without fabricating certainty.

---

## 8. WP-C8.5 — Rename and symbols

### 8.1 Rename preparation

Before returning edits, verify:

* target resolves to a renameable symbol;
* requested name is lexically valid;
* symbol is not a fixed builtin or external immutable declaration;
* new name does not collide in affected scopes;
* edit set is based on one resolved identity.

Return a clear refusal when rename is unsafe.

### 8.2 Rename coverage

Support:

* locals;
* parameters;
* private and public functions;
* types;
* fields;
* enum variants;
* module items;
* imports and aliases where semantics are clear.

Define policy for:

* exported public API;
* external dependencies;
* trait methods and implementations;
* generated declarations;
* extension-provided declarations.

Do not perform a partial rename that leaves the workspace semantically inconsistent.

### 8.3 Edit properties

Edits must be:

* deterministic;
* non-overlapping;
* deduplicated;
* version-aware;
* sorted by file and descending source position where required for safe application.

### 8.4 Document symbols

Return hierarchical symbols reflecting real syntax and semantic identity:

```text
module
struct
  fields
  methods
enum
  variants
trait
  methods
functions
constants
```

### 8.5 Workspace symbols

Search indexed semantic symbols, not raw text.

Support:

* deterministic ranking;
* qualified names;
* package/module context;
* symbol kind;
* cancellation or limits for large workspaces.

### 8.6 Tests

Include:

* local rename with shadowing;
* collision rejection;
* cross-file public function rename;
* field rename;
* import alias;
* same-spelling unrelated symbol untouched;
* deterministic edit order;
* symbols update after document change.

---

## 9. WP-C8.6 — Semantic tokens and inlay information

Start only after C8.1–C8.5 query APIs are stable.

### 9.1 Semantic tokens

Classify semantically where possible:

* namespace/module;
* type;
* struct;
* enum;
* enum variant;
* trait/interface;
* type parameter;
* function;
* method;
* property/field;
* parameter;
* local variable;
* constant;
* builtin;
* extension symbol.

Use lexical classification only for categories that genuinely require no semantic distinction.

### 9.2 Token modifiers

Add only reliable modifiers, such as:

* declaration;
* definition;
* readonly;
* mutable;
* deprecated, if compiler metadata supports it;
* default library, if identity supports it.

### 9.3 Inlay hints

Optional, bounded hints:

* inferred local type where non-obvious;
* parameter names where unambiguous;
* generic substitutions where genuinely useful.

Do not flood ordinary code with redundant hints.

### 9.4 Incremental behaviour

A full-document token implementation is acceptable initially.

Do not advertise token deltas unless delta correctness is tested.

---

## 10. WP-C8.7 — Protocol and editor validation

Raw protocol tests are necessary but insufficient.

### 10.1 JSON-RPC test suite

Test:

* initialise;
* capability advertisement;
* initialized;
* didOpen;
* didChange;
* didSave;
* didClose;
* shutdown;
* exit;
* cancellation;
* malformed messages;
* unknown methods;
* invalid parameters;
* request ordering;
* concurrent requests;
* stale versions;
* clean server termination.

### 10.2 Golden semantic scenarios

Create small multi-file workspaces covering:

* modules;
* packages;
* imports;
* re-exports;
* generics;
* traits;
* borrowing;
* standard-library calls;
* Core-only mode;
* extension-enabled mode.

For each workspace, test the relevant LSP results against exact source ranges and semantic identities.

### 10.3 Real VS Code validation

Run at least one Extension Development Host or packaged-extension validation on a VS Code-capable environment.

Validate interactively:

* diagnostics appear and clear;
* hover content;
* go to definition;
* find references;
* completion;
* signature help;
* rename;
* document symbols;
* workspace symbols;
* semantic tokens if implemented;
* formatting;
* multi-file changes;
* package project loading.

Record:

```text
OS
architecture
VS Code version
extension version/commit
starkc commit
workspace fixture
features exercised
result
screenshots or structured record where practical
```

Protocol tests alone do not close this work package.

### 10.4 Extension packaging

Verify:

* activation events;
* server executable discovery;
* configuration;
* workspace trust behaviour;
* error shown when server cannot start;
* output/log channel;
* restart behaviour;
* no hard-coded developer machine paths;
* packaged extension contains required assets.

---

## 11. Performance and cancellation

C8 is not a performance gate, but editor features must remain usable.

### 11.1 Measurements

Record on fixed workspaces:

* cold project load;
* incremental edit-to-diagnostics latency;
* hover latency;
* definition latency;
* references latency;
* completion latency;
* workspace symbol latency;
* peak memory.

Do not make universal performance claims from a tiny fixture.

### 11.2 Cancellation

Long-running requests should observe cancellation where practical, especially:

* references;
* workspace symbols;
* project reanalysis;
* completion over large package graphs.

Cancellation must not corrupt the shared analysis snapshot.

### 11.3 No stale publication

A slow older analysis must never overwrite a newer result merely because it completed later.

---

## 12. Robustness and security boundaries

Review:

* arbitrary workspace paths;
* symlinks;
* package roots outside the workspace;
* malformed `file://` URIs;
* path traversal;
* enormous files;
* deeply nested syntax;
* rapid document changes;
* invalid UTF-16 positions;
* malformed JSON-RPC;
* subprocess paths, if any remain;
* workspace configuration supplied by untrusted repositories;
* extension command execution.

The language server must not execute project code merely to answer semantic queries.

---

## 13. Required permanent tests

At minimum, land permanent suites for:

```text
lsp_protocol
lsp_document_versions
lsp_diagnostics
lsp_hover
lsp_definition
lsp_references
lsp_completion
lsp_signature_help
lsp_rename
lsp_symbols
lsp_semantic_tokens
lsp_position_encoding
lsp_packages
lsp_extensions
vscode_extension_smoke
```

Suite names may differ, but their responsibilities must remain distinct and discoverable.

Each capability needs:

* positive cases;
* absent-result cases;
* stale-version cases where relevant;
* identity-confusion adversarial cases;
* malformed-input robustness.

---

## 14. Work-package sequence

Execute in this order unless evidence justifies a recorded revision:

```text
C8.0   baseline and advertised-capability audit
C8.0A  shared semantic-query and snapshot foundation
C8.1   diagnostics
C8.2   hover and signature rendering
C8.3   definition and references
C8.4   completion and signature help
C8.5   rename and symbols
C8.6   semantic tokens and optional inlay hints
C8.7   protocol and real-editor validation
C8.8   gate exit
```

Diagnostics may be developed alongside the query foundation where necessary, but do not build later features on separate ad hoc analysis paths.

---

## 15. Commit discipline

Use bounded commits corresponding to one semantic capability or infrastructure step.

Every material commit message must state:

```text
problem
semantic authority used
implementation path
tests added
known limitations
files or shared surfaces touched
whether C7 reconciliation is required
```

Do not combine:

* unrelated refactoring;
* formatting churn;
* extension redesign;
* native backend changes;
* package policy changes.

If a compiler defect is discovered:

1. reduce it;
2. classify whether it belongs to C8 or another gate;
3. add a DEV record;
4. do not quietly patch language semantics inside an LSP commit;
5. block only the affected capability unless the defect invalidates the shared analysis foundation.

---

## 16. Evidence requirements

For every advertised capability, the C8 evidence registry must include:

```text
capability
protocol method
compiler query used
semantic identities exercised
positive tests
negative/adversarial tests
multi-file/package tests
document-version tests
extension-mode tests where relevant
real-editor validation record
last verified commit
known limitations
```

The evidence checker must fail if:

* a capability is advertised but lacks implementation evidence;
* a named test does not exist;
* a protocol method points only to placeholder output;
* an editor-validation claim has no recorded environment;
* a capability is removed or renamed without updating the registry.

Do not manually claim completion based only on the existence of a handler.

---

## 17. Gate exit — WP-C8.8

Create:

```text
starkc/docs/compiler/C8-exit-report.md
```

The report must contain:

* exact qualified commit;
* toolchain;
* supported editors and tested versions;
* capability matrix;
* compiler-query architecture;
* package and extension behaviour;
* protocol-test results;
* real VS Code validation record;
* latency measurements;
* known deviations;
* carried items;
* shared-file reconciliation with C7;
* exact release claim.

### 17.1 Required exit capabilities

C8 cannot close unless these advertised capabilities reflect real compiler semantics:

* diagnostics;
* formatting;
* hover;
* definition;
* references;
* document symbols;
* workspace symbols.

Completion, signature help, rename and semantic tokens should also be implemented under this gate as specified. Any omission requires an explicit owner decision and capability advertisement must remain truthful.

### 17.2 Permitted exit conclusions

```text
C8-CLOSED
C8-CANDIDATE-COMPLETE
C8-BLOCKED
```

`C8-CLOSED` requires:

1. no advertised semantic capability remains a placeholder;
2. diagnostics are version-safe and package-aware;
3. hover is compiler-derived;
4. definition and references use resolved identity;
5. symbols are semantic rather than textual;
6. formatting uses the established compiler formatter;
7. raw protocol tests pass;
8. real VS Code validation passes;
9. C7 parallel changes are reconciled on the exact qualified commit;
10. the evidence registry contains no unresolved capability claim.

### 17.3 Exact closure claim

Use wording no stronger than:

> Gate C8 provides compiler-backed semantic language services for the documented STARK project and package configurations. Advertised diagnostics, hover, navigation, references, symbols, completion, signature and rename capabilities are derived from shared compiler analysis and have been validated through protocol tests and the recorded VS Code environment. Known limitations and unsupported configurations remain explicitly listed.

Do not claim:

* all editors are supported;
* every incomplete source file produces ideal results;
* semantic queries are formally complete;
* the language server proves compiler conformance;
* extension semantics beyond those already implemented by the compiler.

---

## 18. Immediate first task

Begin with WP-C8.0 only.

Do not start by implementing hover.

First produce:

1. the advertised-capability inventory;
2. the current LSP/compiler pipeline map;
3. the placeholder/stub list;
4. the existing-test matrix;
5. the proposed shared semantic-query API;
6. the C7/C8 file-ownership and lease list.

Commit that baseline separately before changing language-service behaviour.
