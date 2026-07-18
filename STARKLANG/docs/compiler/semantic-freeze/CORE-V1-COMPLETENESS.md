# Core v1 Completeness Inventory

Status: **WP-C2.6 complete — governance inventory, not normative semantics**
Updated: 2026-07-18

This ledger is the authoritative inventory of Core v1 semantic questions. It routes each
question to one normative home and one owning work package. It does not define language
behavior: a pending row becomes normative only after its owner approves the answer and the
answer is transferred to the named normative home.

## Reading the matrix

Each row has a stable granular ID. `Status/class` is
`complete|partial|absent|contradictory / specified|prohibited|implementation-defined|
target-defined|deliberately-unspecified|pending-classification`. Pending classification is an
inventory state, not a final behavior class. `Evidence` gives `positive; negative`; `none`
records a real evidence gap. `Cost/owner` gives compatibility cost and the WP that closes the
row. `Decision/dev` gives approval state, open-question ID, and known deviation.

Legacy evidence references (`LEX-*`, `SYN-*`, and so on) resolve to exact source and test paths
in `STARKLANG/conformance/core-v1-coverage.toml`. `U17` means the legacy entry exists but its
positive/negative role is not yet classified (DEV-017). The checked split from all 59 legacy
entries to these IDs is `STARKLANG/conformance/core-v1-rule-id-map.toml`.

Home abbreviations are exact file references: `01`–`07` mean the correspondingly numbered file
in `STARKLANG/docs/spec/`; `Abstract Machine` and `Future Boundaries` mean the two sole future
homes named in the authority map below. A section name after `§` is part of the home.

## Authoritative specification map

This routing decision is final for C2.6. Duplicate text elsewhere must be informative or a
cross-reference.

| Concept | Sole normative home |
| --- | --- |
| Source encoding, tokens, literals, comments | `spec/01-Lexical-Grammar.md` |
| Grammar, precedence, parse classification | `spec/02-Syntax-Grammar.md` |
| Type identity/well-formedness, inference, coercion, traits | `spec/03-Type-System.md` |
| Static names, scopes, flow, initialization, diagnostics | `spec/04-Semantic-Analysis.md` |
| Values, places, moves, temporaries, evaluation, destruction, language traps | `spec/CORE-V1-ABSTRACT-MACHINE.md` (C2.7) |
| Memory-safety guarantees | `spec/05-Memory-Model.md` |
| Required library APIs and compiler-recognized hooks | `spec/06-Standard-Library.md` |
| Modules, packages, identity, entry/process, target contracts | `spec/07-Modules-and-Packages.md` |
| Future-reserved and extension boundaries | `spec/CORE-V1-FUTURE-BOUNDARIES.md` (C2.10) |
| Overview and combined specification | Navigation/generated artifacts only; never authority |

## Source and lexical grammar

| ID | Exact normative question | Status/class | Home | Evidence (+; −) | Cost/owner | Decision/dev |
| --- | --- | --- | --- | --- | --- | --- |
| LEX-SOURCE-001 | Must source be valid UTF-8, and how is invalid input rejected? | partial/pending-classification | 01 §Source Encoding | LEX-001/U17; none | medium/C2.9 | Q018 pending/DEV-017 |
| LEX-IDENT-001 | Which characters and case rules form identifiers? | complete/specified | 01 §Identifiers | LEX-001/U17; LEX-001/U17 | high/C2.11 | settled/DEV-017 |
| LEX-IDENT-002 | What identifier-length limit applies, and how is excess length rejected? | partial/pending-classification | 01 §Identifiers | LEX-002/U17; none | medium/C2.9 | Q019 pending/DEV-017 |
| LEX-KEYWORD-001 | Which Core words always tokenize as keywords? | complete/specified | 01 §Keywords | LEX-001/U17; LEX-001/U17 | high/C2.11 | settled/DEV-017 |
| LEX-INT-001 | Which integer bases, digit forms, and maximal-munch boundaries tokenize? | complete/specified | 01 §Integer Literals | LEX-003; LEX-003 | high/C2.11 | settled/DEV-015 |
| LEX-INT-002 | Where may separators and integer suffixes occur? | complete/specified | 01 §Integer Literals | LEX-003; LEX-003 | high/C2.11 | settled/DEV-015 |
| LEX-FLOAT-001 | Which decimal/exponent/suffix forms are floating literals? | complete/specified | 01 §Floating-Point Literals | LEX-004; LEX-004 | high/C2.11 | settled/DEV-015 |
| LEX-STRING-001 | Which cooked and raw string forms terminate and what content is retained? | complete/specified | 01 §String Literals | LEX-005; LEX-005 | high/C2.11 | settled/none |
| LEX-ESCAPE-001 | Which escapes are legal and which Unicode scalar values may they produce? | partial/pending-classification | 01 §Escape Sequences | LEX-005; LEX-005 | high/C2.9 | Q011 pending/none |
| LEX-CHAR-001 | Must a character literal contain exactly one scalar value? | complete/specified | 01 §Character Literals | LEX-006; LEX-006 | high/C2.11 | settled/none |
| LEX-BOOL-001 | Do `true` and `false` tokenize as Boolean literals? | complete/specified | 01 §Boolean Literals | LEX-007/U17; LEX-007/U17 | high/C2.11 | settled/DEV-017 |
| LEX-OP-001 | Which operator spellings form tokens? | complete/specified | 01 §Operators | LEX-008/U17; LEX-008/U17 | ecosystem-breaking/C2.11 | settled/DEV-017 |
| LEX-DELIM-001 | Which delimiters and punctuation form tokens? | complete/specified | 01 §Delimiters | LEX-009/U17; LEX-009/U17 | ecosystem-breaking/C2.11 | settled/DEV-017 |
| LEX-COMMENT-001 | How do line, nested block, and documentation comments tokenize and attach? | partial/specified | 01 §Comments | LEX-010; LEX-010 | medium/C2.11 | pending-owner-approval/none |
| LEX-SPACE-001 | Which code points are whitespace? | complete/specified | 01 §Whitespace | LEX-011/U17; LEX-011/U17 | low/C2.11 | settled/DEV-017 |
| LEX-TOKEN-001 | What precedence resolves overlapping lexical tokens? | complete/specified | 01 §Token Precedence | LEX-012/U17; LEX-012/U17 | high/C2.11 | settled/DEV-017 |
| LEX-RESERVED-001 | Which future words are reserved and rejected as identifiers? | complete/prohibited | 01 §Reserved Tokens | LEX-013; LEX-013 | ecosystem-breaking/C2.10 | settled/none |
| LEX-ERROR-001 | Which lexical failures are mandatory rejection conditions? | partial/specified | 01 §Lexical Errors | LEX-014/U17; none | medium/C2.11 | pending-owner-approval/DEV-017 |

## Grammar and parsing

| ID | Exact normative question | Status/class | Home | Evidence (+; −) | Cost/owner | Decision/dev |
| --- | --- | --- | --- | --- | --- | --- |
| SYN-PROGRAM-001 | Which item forms constitute a compilation unit? | complete/specified | 02 §Program Structure | SYN-001/U17; SYN-001/U17 | high/C2.11 | settled/DEV-017 |
| SYN-VIS-001 | Where may visibility modifiers appear syntactically? | complete/specified | 02 §Visibility Modifiers | SYN-002/U17; SYN-002/U17 | high/C2.11 | settled/DEV-017 |
| SYN-FN-001 | Which function signatures, receivers, generics, and return forms parse? | complete/specified | 02 §Functions | SYN-002/U17; SYN-002/U17 | high/C2.11 | settled/DEV-017 |
| SYN-DATA-001 | Which struct, enum, trait, impl, and associated-item forms parse? | complete/specified | 02 §Type and Trait Declarations | SYN-003/U17; SYN-003/U17 | high/C2.11 | settled/DEV-017 |
| SYN-ALIAS-001 | Which type-alias declarations parse? | complete/specified | 02 §Type Alias | SYN-008/U17; SYN-008/U17 | high/C2.11 | settled/DEV-017 |
| SYN-CONST-001 | Which constant declarations parse? | complete/specified | 02 §Constant Declarations | SYN-009/U17; SYN-009/U17 | high/C2.11 | settled/DEV-017 |
| SYN-BLOCK-001 | How do statements and a trailing expression determine block syntax? | complete/specified | 02 §Blocks and Statements | SYN-004/U17; SYN-004/U17 | high/C2.11 | settled/DEV-017 |
| SYN-LET-001 | Which initialized and deferred `let` forms parse? | complete/specified | 02 §Let Statements | SYN-005/U17; SYN-005/U17 | high/C2.11 | settled/DEV-017 |
| SYN-EXPR-001 | What are expression precedence and associativity? | complete/specified | 02 §Expressions | SYN-006/U17; SYN-006/U17 | ecosystem-breaking/C2.11 | settled/DEV-017 |
| SYN-PLACE-001 | Which syntactic forms may appear as assignment targets? | partial/specified | 02 §Place Expressions | SYN-006/U17; none | high/C2.7 | Q006 pending/DEV-017 |
| SYN-CONTROL-001 | Which `if`, `match`, loop, `break`, `continue`, and `return` forms parse? | complete/specified | 02 §Control Flow | SYN-007/U17; SYN-007/U17 | high/C2.11 | settled/DEV-017 |
| SYN-PATTERN-001 | Which wildcard, binding, literal, tuple, struct, enum, range, and alternative patterns parse? | partial/specified | 02 §Patterns | SYN-007/U17; none | high/C2.8 | Q020 pending/DEV-017 |
| SYN-PATH-001 | Which paths, use trees, and explicit generic arguments parse? | complete/specified | 02 §Paths and Imports | SYN-008/U17; SYN-008/U17 | high/C2.11 | settled/DEV-017 |
| SYN-LITERAL-001 | Which tuple, array, repeat-array, and struct literal forms parse? | complete/specified | 02 §Literal Expressions | SYN-009/U17; SYN-009/U17 | high/C2.11 | settled/DEV-017 |
| SYN-GUARD-001 | Where must struct literals be parenthesized to avoid control-flow ambiguity? | complete/prohibited | 02 §Struct Literal Restrictions | SYN-009/U17; SYN-009/U17 | high/C2.11 | settled/DEV-017 |
| SYN-TYPE-001 | Which primitive, path, tuple, array, reference, function, and never types parse? | complete/specified | 02 §Types | SYN-010/U17; SYN-010/U17 | high/C2.11 | settled/DEV-017 |
| SYN-SPLIT-001 | How are generic-closing and tuple-field tokens reclassified after lexing? | complete/specified | 02 §Parsing Notes | SYN-013; SYN-013 | high/C2.11 | settled/DEV-018 |
| SYN-RECOVERY-001 | Is parser recovery observable beyond required acceptance/rejection and diagnostics? | absent/pending-classification | 02 §Syntax Errors | none; none | low/C2.9 | Q021 pending/none |

## Names, types, inference, and traits

| ID | Exact normative question | Status/class | Home | Evidence (+; −) | Cost/owner | Decision/dev |
| --- | --- | --- | --- | --- | --- | --- |
| NAME-SCOPE-001 | Which declarations introduce scopes and when do bindings become visible? | partial/specified | 04 §Scopes and Symbol Tables | SEM-001/U17; none | high/C2.8 | Q004 pending/DEV-017 |
| NAME-SHADOW-001 | Which declarations may shadow which earlier bindings? | partial/pending-classification | 04 §Name Resolution | SEM-001/U17; none | high/C2.8 | Q004 pending/DEV-017 |
| NAME-RESOLVE-001 | How are local, item, module, type, value, trait, and associated names resolved? | partial/pending-classification | 04 §Name Resolution | SEM-001,U17; PKG-003/U17 | ecosystem-breaking/C2.8 | Q004,Q005 pending/DEV-017 |
| TYPE-PRIM-001 | What are the identities of primitive, tuple, array, reference, function, and never types? | partial/specified | 03 §Types | TYP-001/U17; none | ecosystem-breaking/C2.8 | Q003 pending/DEV-017 |
| TYPE-NOMINAL-001 | What facts determine struct and enum type identity? | partial/pending-classification | 03 §User-Defined Types | TYP-001/U17; none | ecosystem-breaking/C2.8 | Q003,Q013A pending/DEV-017 |
| TYPE-ALIAS-001 | Does a type alias transparently denote its target or introduce identity? | absent/pending-classification | 03 §Type Aliases | none; none | ecosystem-breaking/C2.8 | Q002 pending/none |
| TYPE-WF-001 | Which recursive, unsized, zero-length, and edge-size types are well formed? | partial/pending-classification | 03 §Type Well-Formedness | TYP-001/U17; none | ecosystem-breaking/C2.8 | Q003,Q019 pending/DEV-017 |
| TYPE-INFER-001 | How do local constraints, expected types, later uses, and ambiguity determine inferred types? | partial/pending-classification | 03 §Type Inference | TYP-002/U17; none | high/C2.8 | Q004 pending/DEV-017 |
| TYPE-INFER-002 | When and to what types do unsuffixed numeric literals default? | complete/specified | 03 §Literal Type Inference | TYP-002/U17; TYP-002/U17 | high/C2.11 | settled/DEV-017 |
| TYPE-GENERIC-001 | How are generic parameters, substitutions, bounds, and explicit arguments normalized? | partial/pending-classification | 03 §Generics | TYP-002,U17; TYP-007/U17 | ecosystem-breaking/C2.8 | Q004 pending/DEV-017 |
| TYPE-COERCE-001 | Which numeric conversions are implicit? | complete/prohibited | 03 §Type Coercion | TYP-003/U17; TYP-003/U17 | high/C2.11 | settled/DEV-017 |
| TYPE-COERCE-002 | When may mutable references coerce to shared references? | complete/specified | 03 §Reference Coercions | TYP-003/U17; TYP-003/U17 | high/C2.11 | settled/DEV-017 |
| TYPE-COERCE-003 | When may an array reference coerce to a slice/view? | partial/specified | 03 §Array-to-Slice Coercion | TYP-003/U17; none | high/C2.8 | Q003 pending/DEV-017 |
| TYPE-COERCE-004 | How does the never type coerce at control-flow joins? | complete/specified | 03 §Never Type Coercion | TYP-003/U17; TYP-003/U17 | high/C2.11 | settled/DEV-017 |
| TYPE-CAST-001 | Which explicit numeric casts are legal and what values do they produce? | partial/pending-classification | 03 §Explicit Casts | TYP-003/U17; none | high/C2.9 | Q008,Q010 pending/DEV-017,DEV-024 |
| TYPE-CFLOW-001 | How do branches and `match` arms choose a common type? | complete/specified | 03 §Control-Flow Typing | TYP-004/U17; TYP-004/U17 | high/C2.11 | settled/DEV-017 |
| TYPE-LOOP-001 | How do loop kind and reachable `break` values determine expression type? | partial/specified | 03 §Loop Typing | TYP-004/U17; none | high/C2.8 | Q004 pending/DEV-017 |
| TYPE-METHOD-001 | How are inherent and trait method candidates collected and prioritized? | partial/pending-classification | 03 §Method Resolution | TYP-006/U17; none | ecosystem-breaking/C2.8 | Q004,Q005 pending/DEV-017 |
| TYPE-METHOD-002 | Which automatic borrow/dereference adjustments are allowed? | partial/pending-classification | 03 §Method Resolution | TYP-006/U17; none | high/C2.8 | Q004,Q006 pending/DEV-017 |
| TRAIT-DEF-001 | What declarations, default bodies, and impl obligations are legal? | partial/specified | 03 §Traits and Implementations | TYP-005/U17; none | ecosystem-breaking/C2.8 | Q005 pending/DEV-017 |
| TRAIT-ASSOC-001 | How are associated functions/types resolved and qualified? | partial/pending-classification | 03 §Associated Items | TYP-005/U17; none | high/C2.8 | Q005,Q015 pending/DEV-017 |
| TRAIT-COHERENCE-001 | What orphan rule applies across package identities? | partial/pending-classification | 03 §Coherence | SEM-007; SEM-007 | ecosystem-breaking/C2.8 | Q005,Q013A pending/DEV-021 |
| TRAIT-COHERENCE-002 | How are blanket impl overlap and candidate ambiguity decided? | partial/pending-classification | 03 §Coherence | SEM-007; SEM-007 | ecosystem-breaking/C2.8 | Q005 pending/DEV-021 |
| TRAIT-LAW-001 | What semantic laws bind `Eq`, `Ord`, and `Hash`, and what follows from violations? | absent/pending-classification | 03 §Standard Traits | none; none | ecosystem-breaking/C2.8 | Q005A pending/none |

## Flow, ownership, abstract execution, patterns, and constants

| ID | Exact normative question | Status/class | Home | Evidence (+; −) | Cost/owner | Decision/dev |
| --- | --- | --- | --- | --- | --- | --- |
| FLOW-RETURN-001 | Must every reachable non-unit path return a compatible value? | complete/specified | 04 §Return Analysis | SEM-002/U17; SEM-002/U17 | high/C2.11 | settled/DEV-017 |
| FLOW-LOOP-001 | Which statements are unreachable and which loops are known not to fall through? | partial/specified | 04 §Control Flow Analysis | SEM-003/U17; none | medium/C2.8 | Q004 pending/DEV-017 |
| FLOW-INIT-001 | When is a local definitely initialized on all paths? | complete/specified | 04 §Definite Assignment | none; none | high/C2.11 | settled/none |
| FLOW-MUT-001 | Which assignments require a mutable binding or mutable place? | complete/specified | 04 §Mutability Checking | none; none | high/C2.11 | settled/none |
| FLOW-BOUNDS-001 | Which statically provable indexing errors must be rejected? | partial/specified | 04 §Static Bounds Analysis | none; none | medium/C2.9 | Q017 pending/none |
| FLOW-TRY-001 | Where is `?` legal and what enclosing result type is required? | complete/specified | 04 §Error Propagation Analysis | none; none | high/C2.11 | settled/none |
| DIAG-CATALOG-001 | Which rejection categories and diagnostic codes are stable and collision-free? | partial/pending-classification | 04 §Diagnostics | SEM-006/U17; none | medium/C2.11 | Q022 pending/DEV-019 |
| OWN-MOVE-001 | Which value uses move, copy, or borrow an operand? | complete/specified | Abstract Machine §Moves, Partial Moves, and Reinitialization | MEM-001/U17; none | ecosystem-breaking/C2.11 | Q006 approved/DEV-017 |
| OWN-COPY-001 | Which types are implicitly copyable? | partial/pending-classification | 03 §Copy Types | MEM-002/U17; none | ecosystem-breaking/C2.8 | Q003,Q006 pending/DEV-017 |
| OWN-BORROW-001 | When may shared and exclusive borrows coexist? | partial/specified | 03 §Borrow Legality | MEM-003/U17; none | ecosystem-breaking/C2.8 | Q006 pending/DEV-017 |
| OWN-REGION-001 | When do local, argument-derived, and temporary borrows end? | partial/pending-classification | 03 §Borrow Regions | MEM-004/U17; none | ecosystem-breaking/C2.8 | Q006 pending/DEV-017 |
| OWN-RETURN-001 | Which references may escape through return values? | partial/pending-classification | 03 §Returned References | MEM-005/U17; none | ecosystem-breaking/C2.8 | Q006 pending/DEV-017 |
| OWN-CARRY-001 | How does reference provenance propagate through aggregates and calls? | partial/pending-classification | 03 §Reference Provenance | MEM-004/U17; none | ecosystem-breaking/C2.8 | Q006 pending/DEV-017 |
| OWN-PARTIAL-001 | Which field moves are permitted before later aggregate use or destruction? | complete/specified | Abstract Machine §Moves, Partial Moves, and Reinitialization | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| OWN-REINIT-001 | When does assignment reinitialize a moved place? | complete/specified | Abstract Machine §Moves, Partial Moves, and Reinitialization | none; none | high/C2.11 | Q006 approved/none |
| EXEC-EVAL-001 | What is the evaluation order of operands, calls, indexing, and aggregates? | complete/specified | Abstract Machine §Evaluation Order | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| EXEC-PLACE-001 | Which expressions denote values versus places, and when are places read? | complete/specified | Abstract Machine §Places and Projections | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| EXEC-TEMP-001 | When are temporaries created and destroyed on success and early exit? | complete/specified | Abstract Machine §Temporaries | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| EXEC-ASSIGN-001 | In what order are assignment target, right-hand side, old value, and write processed? | complete/specified | Abstract Machine §Assignment and Replacement | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| DROP-EXACT-001 | Which initialized values are destroyed exactly once? | complete/specified | Abstract Machine §Exactly-Once Rule | MEM-006/U17; none | ecosystem-breaking/C2.11 | Q006 approved/DEV-017 |
| DROP-ORDER-001 | What is destruction order for locals, fields, elements, parameters, and partial aggregates? | complete/specified | Abstract Machine §Deterministic Order | MEM-006/U17; none | ecosystem-breaking/C2.11 | Q006 approved/DEV-017 |
| DROP-ABORT-001 | Does a language trap perform destruction before termination? | complete/specified | Abstract Machine §Trap Termination | MEM-006/U17; none | high/C2.11 | Q006 approved; Q017 partial/DEV-017 |
| AM-VALUE-001 | What is a language value independently of runtime representation? | complete/specified | Abstract Machine §Values, Objects, Storage, and Ownership | none; not-applicable (definition) | ecosystem-breaking/C2.11 | Q006 approved/none |
| AM-OBJECT-001 | What gives a live object identity across ownership transfer? | complete/specified | Abstract Machine §Values, Objects, Storage, and Ownership | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| AM-STORAGE-001 | What is an abstract storage location/allocation without layout promises? | complete/deliberately-unspecified | Abstract Machine §Values, Objects, Storage, and Ownership | none; not-applicable (representation excluded) | high/C2.11 | Q006 approved/none |
| AM-OWNER-001 | Which entities may own an object and its destruction obligation? | complete/specified | Abstract Machine §Values, Objects, Storage, and Ownership | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| AM-LOCAL-001 | When is a local initialized or moved-from? | complete/specified | Abstract Machine §Values, Objects, Storage, and Ownership | none; none | high/C2.11 | Q006 approved/none |
| AM-TEMP-001 | What is a temporary owner? | complete/specified | Abstract Machine §Values, Objects, Storage, and Ownership | none; none | high/C2.11 | Q006 approved/none |
| EXEC-ONCE-001 | Must every evaluated source expression run exactly once? | complete/specified | Abstract Machine §Results and Full Expressions | none; none | ecosystem-breaking/C2.11 | Q006 approved/DEV-034 |
| EXEC-DISPATCH-001 | Must runtime execution invoke exactly the statically selected callable/protocol? | complete/specified | Abstract Machine §Results and Full Expressions | none; none | ecosystem-breaking/C2.11 | Q006 approved/DEV-024,DEV-026,DEV-027,DEV-038,DEV-043 |
| EXEC-FOR-001 | Must every statically accepted iterator execute through the selected iterator protocol? | complete/specified | Abstract Machine §Evaluation Order | none; none | ecosystem-breaking/C2.11 | Q006 approved/DEV-031 |
| MOVE-READ-001 | When does a place read copy, move, or borrow? | complete/specified | Abstract Machine §Value and Place Contexts | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| EXEC-AGG-001 | When do completed aggregate fields acquire ownership and in what order? | complete/specified | Abstract Machine §Aggregate Construction | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| EXEC-AGG-002 | What happens to a partial aggregate on normal transfer versus trap? | complete/specified | Abstract Machine §Aggregate Construction | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| EXEC-CFLOW-001 | Which early exits are normal transfers and how do they clean scopes? | complete/specified | Abstract Machine §Normal Control Transfer and Partial Evaluation | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| REF-IDENTITY-001 | What facts constitute reference identity and aliasing? | complete/specified | Abstract Machine §Identity and Derivation | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| REF-PROJECT-001 | Do reference projections and auto-dereference remain live views? | complete/specified | Abstract Machine §Projection, Auto-Borrow, and Auto-Dereference | none; none | ecosystem-breaking/C2.11 | Q006 approved/DEV-037 |
| REF-RETURN-001 | How does a returned receiver/parameter-derived reference survive callee exit? | complete/specified | Abstract Machine §Returned and Receiver-Derived References | none; none | ecosystem-breaking/C2.11 | Q006 approved/DEV-035 |
| REF-SLICE-001 | Is a range slice a live view or an element copy? | complete/specified | Abstract Machine §Slice References | none; none | ecosystem-breaking/C2.11 | Q006 approved/DEV-028,DEV-041,DEV-042 |
| REF-CARRY-001 | How do borrow-carrying values preserve contained reference identity? | complete/specified | Abstract Machine §Borrow-Carrying Values | none; none | ecosystem-breaking/C2.11 | Q006 approved/none |
| DROP-LOOP-001 | When are loop bindings, iterators, and unconsumed elements destroyed? | complete/specified | Abstract Machine §Loops and Iterators | none; none | ecosystem-breaking/C2.11 | Q006 approved/DEV-031,DEV-039 |
| DROP-COLLECTION-001 | How do collection destruction, clear, remove, replace, and duplicate discard destroy ownership? | complete/specified | Abstract Machine §Collections and Ownership-Discarding Operations | none; none | ecosystem-breaking/C2.11 | Q006 approved/DEV-040 |
| OBS-COMPARE-001 | Which observations must interpreter, MIR, and native execution compare? | complete/specified | Abstract Machine §Observable Execution and Differential Comparison | none; not-applicable (comparator definition) | ecosystem-breaking/C2.12 | Q006 approved/none |
| PAT-EXHAUST-001 | Which finite domains require exhaustive patterns? | partial/specified | 04 §Pattern Exhaustiveness | SEM-006/U17; none | high/C2.8 | Q020 pending/DEV-017 |
| PAT-USEFUL-001 | Which unreachable patterns require rejection or warning? | partial/pending-classification | 04 §Pattern Usefulness | SEM-006/U17; none | medium/C2.8 | Q020 pending/DEV-017 |
| PAT-OWN-001 | Do Core v1 pattern bindings move, copy, or preserve references for each matched component? | complete/specified | Abstract Machine §Pattern Execution | none; none | ecosystem-breaking/C2.11 | Q006 approved; Q020 runtime approved/none |
| PAT-DROP-001 | What is destroyed for failed arm tests, wildcards, and selected-arm exit? | complete/specified | Abstract Machine §Pattern Execution | none; none | ecosystem-breaking/C2.11 | Q006 approved; Q020 runtime approved/none |
| CONST-DECL-001 | Which declarations require compile-time evaluation? | partial/specified | 03 §Constant Contexts | SEM-008/U17; none | high/C2.8 | Q007 pending/DEV-017 |
| CONST-SUBSET-001 | Which deterministic expressions, calls, and operations are valid in constants? | partial/pending-classification | 03 §Constant Evaluation | SEM-008/U17; none | ecosystem-breaking/C2.8 | Q007 pending/DEV-017 |
| CONST-FAIL-001 | How do cycles, overflow, traps, and resource limits fail constant evaluation? | absent/pending-classification | 03 §Constant Evaluation Failures | none; none | high/C2.8 | Q007,Q017 pending/none |

## Numeric and text behavior

| ID | Exact normative question | Status/class | Home | Evidence (+; −) | Cost/owner | Decision/dev |
| --- | --- | --- | --- | --- | --- | --- |
| NUM-INT-TYPE-001 | What widths and value ranges do integer types have? | complete/specified | 03 §Integer Types | TYP-001/U17; TYP-001/U17 | ecosystem-breaking/C2.11 | settled/DEV-017 |
| NUM-INT-ARITH-001 | What happens on integer add/subtract/multiply/negate overflow? | partial/pending-classification | Abstract Machine §Integer Operations | STD-001/U17; none | ecosystem-breaking/C2.9 | Q008 pending/DEV-017 |
| NUM-INT-DIV-001 | What are signed division/remainder rounding, zero, and minimum/-1 results? | absent/pending-classification | Abstract Machine §Integer Operations | none; none | ecosystem-breaking/C2.9 | Q008 pending/none |
| NUM-SHIFT-001 | How are negative or out-of-width shift counts handled? | absent/pending-classification | Abstract Machine §Integer Operations | none; none | ecosystem-breaking/C2.9 | Q008 pending/none |
| NUM-CAST-001 | How do narrowing, sign-changing, and integer/float casts behave? | partial/pending-classification | Abstract Machine §Numeric Conversion | STD-001/U17; none | ecosystem-breaking/C2.9 | Q008,Q010 pending/DEV-024 |
| NUM-FLOAT-FORMAT-001 | Which IEEE formats correspond to `f32` and `f64`? | partial/specified | 03 §Floating-Point Types | TYP-001/U17; none | ecosystem-breaking/C2.9 | Q010 pending/DEV-017 |
| NUM-FLOAT-OP-001 | What rounding, NaN, infinity, signed-zero, and contraction rules apply? | absent/pending-classification | Abstract Machine §Floating-Point Operations | none; none | ecosystem-breaking/C2.9 | Q010 pending/none |
| NUM-FLOAT-TRAIT-001 | Which equality, ordering, and hashing contracts apply to floats? | contradictory/pending-classification | 03 §Floating-Point Traits | TYP-005/U17; none | ecosystem-breaking/C2.9 | Q009 pending/none |
| NUM-FLOAT-REPRO-001 | Which floating results must be reproducible across targets/backends? | absent/pending-classification | 07 §Target-Defined Numeric Behavior | none; none | high/C2.9 | Q010 pending/none |
| TEXT-UTF8-001 | Are strings valid UTF-8 at every observable boundary? | partial/specified | 06 §String | STD-003; none | ecosystem-breaking/C2.9 | Q011 pending/none |
| TEXT-CHAR-001 | Is `char` a Unicode scalar value? | complete/specified | 03 §Character Type | TYP-001/U17; TYP-001/U17 | ecosystem-breaking/C2.11 | settled/DEV-017 |
| TEXT-INDEX-001 | Are string offsets bytes, scalars, or another unit? | contradictory/pending-classification | 06 §String Index Units | STD-003; none | ecosystem-breaking/C2.9 | Q011 pending/none |
| TEXT-BOUNDARY-001 | What occurs at a non-scalar UTF-8 boundary? | partial/pending-classification | 06 §String Boundaries | STD-003; none | high/C2.9 | Q011 pending/none |
| TEXT-ITER-001 | What units and order do string iteration APIs expose? | absent/pending-classification | 06 §String Iteration | none; none | ecosystem-breaking/C2.9 | Q011 pending/none |
| TEXT-CASE-001 | Which Unicode data/version governs classification and case operations? | absent/pending-classification | 06 §Unicode Policy | none; none | high/C2.9 | Q011 pending/none |

## Modules, packages, process, library hooks, layout, limits, and extensions

| ID | Exact normative question | Status/class | Home | Evidence (+; −) | Cost/owner | Decision/dev |
| --- | --- | --- | --- | --- | --- | --- |
| MOD-DECL-001 | How do inline and file-backed modules form the module tree? | complete/specified | 07 §Module Declarations | PKG-001/U17; PKG-001/U17 | ecosystem-breaking/C2.11 | settled/DEV-017 |
| MOD-FILE-001 | How is a file-backed module located and bounded by its package/workspace? | partial/pending-classification | 07 §Module Files | PKG-004; PKG-004 | ecosystem-breaking/C2.9 | Q013A pending/DEV-036 |
| MOD-PATH-001 | How do absolute, relative, `self`, `super`, and package paths resolve? | partial/specified | 07 §Path Resolution | PKG-002/U17; none | ecosystem-breaking/C2.9 | Q013A pending/DEV-017 |
| MOD-USE-001 | How do aliases, nested imports, glob imports, and conflicts resolve? | partial/specified | 07 §Imports | PKG-003/U17; none | high/C2.9 | Q013A pending/DEV-017 |
| MOD-VIS-001 | Which items are visible across module and package boundaries? | partial/specified | 07 §Visibility | PKG-004/U17; none | ecosystem-breaking/C2.9 | Q023 pending/DEV-017 |
| MOD-REEXPORT-001 | May public APIs expose private or dependency-private types? | absent/pending-classification | 07 §Public API Reachability | none; none | ecosystem-breaking/C2.9 | Q023 pending/DEV-022 |
| MOD-CYCLE-001 | Which module/import cycles are permitted or rejected? | partial/pending-classification | 07 §Module Cycles | PKG-005/U17; none | high/C2.9 | Q013A pending/DEV-017 |
| PKG-MANIFEST-001 | Which manifest fields are required and how are they validated? | partial/specified | 07 §Package Manifest | PKG-006/U17; none | high/C2.9 | Q013A pending/DEV-017 |
| PKG-RESOLVE-001 | How are dependency constraints, sources, aliases, and features resolved? | partial/pending-classification | 07 §Dependency Resolution | PKG-006/U17; none | ecosystem-breaking/C2.9 | Q013A,Q013B pending/DEV-017 |
| PKG-VERSION-001 | Which compatible version is selected and how are ties handled? | partial/pending-classification | 07 §Version Selection | PKG-006/U17; none | ecosystem-breaking/C2.9 | Q013B pending/DEV-017 |
| PKG-CYCLE-001 | Which dependency cycles are rejected? | complete/prohibited | 07 §Dependency Cycles | PKG-005/U17; PKG-005/U17 | high/C2.11 | settled/DEV-017 |
| PKG-IDENTITY-001 | What relocation-stable facts identify a package instance and public item? | absent/pending-classification | 07 §Package Identity | none; none | ecosystem-breaking/C2.9 | Q013A pending/none |
| PKG-MULTIVER-001 | Which resolved versions may coexist in one dependency graph? | partial/pending-classification | 07 §Version Coexistence | PKG-006/U17; none | ecosystem-breaking/C2.9 | Q013B pending/DEV-017 |
| PKG-LOCK-001 | What lockfile facts are reproducible and authoritative? | absent/pending-classification | 07 §Lockfiles | none; none | ecosystem-breaking/C2.9 | Q013A,Q013B pending/none |
| PROC-MAIN-001 | Which `main` signatures are executable? | absent/pending-classification | 07 §Executable Entry | none; none | ecosystem-breaking/C2.9 | Q014 pending/none |
| PROC-EXIT-001 | How do return values, `Result`, and traps map to exit status? | absent/pending-classification | 07 §Process Termination | none; none | high/C2.9 | Q014,Q017 pending/none |
| PROC-STREAM-001 | What are the observable stdin/stdout/stderr and flush contracts? | absent/pending-classification | 07 §Process Streams | none; none | high/C2.9 | Q014 pending/DEV-009 |
| STD-PRELUDE-001 | Which names are implicitly available? | complete/specified | 06 §Prelude | STD-001; none | ecosystem-breaking/C2.11 | settled/none |
| STD-FORMAT-001 | What output and argument behavior do the Core formatting/printing facilities expose? | partial/specified | 06 §Formatting | STD-002; none | high/C2.9 | Q014 pending/DEV-023 |
| STD-PROFILE-001 | Which APIs are required in each Core profile? | partial/pending-classification | 06 §Conformance Profiles | STD-002/U17; none | ecosystem-breaking/C2.9 | Q024 pending/DEV-009 |
| STD-HOOK-001 | Which library items are compiler-recognized hooks rather than ordinary APIs? | partial/pending-classification | 06 §Language Hooks | STD-004; none | ecosystem-breaking/C2.8 | Q015 pending/DEV-023 |
| STD-TRAIT-001 | Which standard traits and required items belong to the Core profile? | partial/pending-classification | 06 §Core Traits | STD-004; STD-004 | ecosystem-breaking/C2.8 | Q005A,Q015 pending/DEV-023 |
| STD-OPTION-001 | What are `Option` representation-independent behavior and APIs? | partial/specified | 06 §Option | none; none | high/C2.11 | pending-owner-approval/none |
| STD-RESULT-001 | What are `Result`, propagation, and combinator behavior? | partial/specified | 06 §Result | none; none | high/C2.11 | pending-owner-approval/none |
| STD-ITER-001 | What is the iterator protocol and termination behavior? | partial/specified | 06 §Iterator | STD-003; none | high/C2.11 | pending-owner-approval/none |
| STD-VEC-001 | What are vector growth, indexing, bounds, and ownership contracts? | partial/specified | 06 §Vec | STD-003; none | high/C2.11 | pending-owner-approval/none |
| STD-HASH-001 | What equality, hashing, collision, and iteration-order contracts govern maps/sets? | partial/pending-classification | 06 §HashMap and HashSet | STD-003; none | ecosystem-breaking/C2.9 | Q005A pending/none |
| STD-IO-001 | Which I/O APIs and error mappings are required? | partial/pending-classification | 06 §I/O | none; none | high/C2.9 | Q014,Q017,Q024 pending/DEV-009 |
| STD-CONVERT-001 | Which parsing/conversion APIs share language numeric semantics? | partial/pending-classification | 06 §Conversion | none; none | ecosystem-breaking/C2.9 | Q008,Q010 pending/DEV-024 |
| STD-MATH-001 | Which mathematical operations, edge cases, and errors are required? | partial/pending-classification | 06 §Math | STD-005; STD-005 | high/C2.9 | Q008,Q010 pending/none |
| STD-RANDOM-001 | Which random APIs, seed determinism, and range contracts are required? | partial/pending-classification | 06 §Random | STD-005; none | high/C2.9 | Q024 pending/none |
| LAYOUT-QUERY-001 | Which size, alignment, discriminant, and address facts are observable? | contradictory/pending-classification | 07 §Target-Defined Layout | MEM-006/U17; none | ecosystem-breaking/C2.9 | Q012 pending/DEV-017 |
| LAYOUT-ABI-001 | Does Core v1 promise any stable data or calling ABI? | contradictory/pending-classification | 07 §ABI Boundary | none; none | ecosystem-breaking/C2.9 | Q012 pending/none |
| TRAP-CATEGORY-001 | Which failures are language traps, and how are host/process/resource failures excluded? | partial/pending-classification | Abstract Machine §Trap Categories | none; none | high/C2.9 | Q017 language-trap boundary approved; external classification pending/none |
| LIMIT-RESOURCE-001 | How are allocation, stack, recursion, and host-resource exhaustion classified? | absent/pending-classification | 07 §Implementation and Target Limits | none; none | high/C2.9 | Q017 pending/none |
| LIMIT-COMPILER-001 | Which source, nesting, object, array, and package limits may implementations impose? | absent/pending-classification | 07 §Implementation and Target Limits | none; none | high/C2.9 | Q017,Q019 pending/none |
| FUTURE-SYNTAX-001 | Which tokens and grammar space are reserved for future Core editions? | partial/pending-classification | Future Boundaries §Reserved Syntax | LEX-001/U17; none | ecosystem-breaking/C2.10 | Q016 pending/DEV-017 |
| FUTURE-CLOSURE-001 | What ownership boundary must future capturing closures preserve? | absent/pending-classification | Future Boundaries §Closures and Lifetimes | none; none | ecosystem-breaking/C2.10 | Q016 pending/none |
| FUTURE-THREAD-001 | What single-threaded guarantees must remain valid if concurrency is added later? | contradictory/pending-classification | Future Boundaries §Concurrency | none; none | ecosystem-breaking/C2.10 | Q016 pending/none |
| FUTURE-FFI-001 | What provenance, capability, ABI, and safety boundary applies to native providers/FFI? | absent/pending-classification | Future Boundaries §Native Extensions | none; none | ecosystem-breaking/C2.10 | Q016 pending/none |
| EXT-ISOLATION-001 | How are post-v1 extensions prevented from silently changing Core v1 behavior? | absent/pending-classification | Future Boundaries §Extension Isolation | none; none | ecosystem-breaking/C2.10 | Q016 pending/none |

## Exit accounting

The matrix contains every question found by the C2.6 sentence-level audit across chapters 00–07,
the reference-execution contract, the coverage database, and known deviations. Rows marked
`complete` still require C2.11 evidence classification where DEV-017 is linked. Rows marked
`partial`, `absent`, or `contradictory` are deliberately not promoted from current compiler
behavior. C2.7–C2.10 own the decisions; C2.11 owns final positive/negative evidence and status
alignment.
