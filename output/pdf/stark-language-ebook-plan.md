# Learning STARK

## Ebook plan and proposed table of contents

**Working title:** *Learning STARK*  
**Subtitle:** *Safe Systems Programming for Typed ML Deployment*  
**Edition:** Early Access - Core v1 specification edition  
**Format:** Original technical field guide, inspired by the clarity and pedagogy of classic O'Reilly programming books without copying O'Reilly branding or trade dress

## Editorial promise

This book introduces STARK as it exists in the repository today: a complete draft of the Core v1 language specification, an early Rust compiler scaffold, and an optional tensor/model extension aimed at safer ONNX deployment. It will not present proposed features as implemented or repeat superseded claims from the archived pre-pivot design.

Readers will learn to:

- Read and write idiomatic Core v1 source examples.
- Reason about STARK's static types, ownership, borrowing, and predictable runtime rules.
- Use structs, enums, pattern matching, traits, generics, modules, `Option`, and `Result`.
- Understand what the tensor extension adds for shape, dtype, device, and model-signature checking.
- Follow the planned path from source text to a checked native inference program.
- Explore the specification and contribute to the compiler without confusing roadmap work with shipped functionality.

## Audience and assumptions

The primary audience is software engineers, ML platform engineers, compiler-curious developers, and technical decision-makers. Familiarity with one typed language is useful but not required. Rust concepts are explained when introduced; Python comparisons are used selectively for ML readers.

All runnable-looking listings will carry one of these labels:

- **Core v1:** normative language syntax or semantics.
- **Tensor extension v0.1:** optional extension syntax or semantics.
- **Illustrative:** explanatory pseudocode or a proposed workflow.
- **Historical:** superseded material included only when its evolution is instructive.

## Proposed table of contents

### Preface

- Why this book exists
- What STARK is - and is not - today
- How to read examples before the compiler is runnable
- Core, extensions, implementation, and archive
- Conventions used in the book

### Part I - Meet STARK

#### 1. A Language for Safer Deployment

- The gap between research code and deployed systems
- STARK's current product wedge
- Shape, meaning, location, ownership, and constraints
- Core v1 versus the tensor extension
- Design influences and deliberate differences
- A realistic definition of success

#### 2. A Guided Tour of a STARK Program

- The anatomy of a source file
- `fn main` and expression-oriented blocks
- Immutable and mutable bindings
- Functions and explicit signatures
- Struct construction and method calls
- Reading compiler-style diagnostics
- A complete Core v1 example, annotated line by line

#### 3. Values, Types, and Expressions

- Integers, floating-point values, booleans, characters, and strings
- Arrays, slices, tuples, and ranges
- Local type inference
- Explicit casts and the absence of implicit numeric conversion
- Operators and precedence
- Overflow, bounds checks, and predictable failure
- Exercises: trace the type of each expression

#### 4. Control Flow and Pattern Matching

- `if` as an expression
- `loop`, `while`, and `for`
- `break`, `continue`, and `return`
- Enums and variants
- Destructuring with `match`
- Exhaustiveness and unreachable patterns
- Modeling state without null

### Part II - Build Reliable Core Programs

#### 5. Functions, Structs, Enums, and Methods

- Function parameters and return values
- Product types with structs
- Sum types with enums
- Associated functions and methods
- The `Self` type
- Separating data representation from behavior
- Case study: a typed image-processing configuration

#### 6. Errors as Values

- `Option<T>` for absence
- `Result<T, E>` for failure
- The `?` operator
- Designing error enums
- Propagating context without exceptions
- Panic and abort semantics
- Case study: validating an inference request

#### 7. Ownership, Moves, and Automatic Cleanup

- The single-owner rule
- Copy values and moved values
- Moves through assignment, calls, and returns
- Reinitialization and partial moves
- Stack, heap, and `Box<T>`
- Deterministic destruction and drop order
- Reading use-after-move diagnostics
- Exercises: predict which bindings remain valid

#### 8. Borrowing Without Dangling References

- Shared references and mutable references
- Aliasing rules
- Lexical lifetimes in Core v1
- Borrow scopes and temporary borrows
- References in function signatures
- Borrow-carrying types
- Common borrowing mistakes and repairs
- Case study: inspecting data without transferring ownership

#### 9. Generics, Traits, and Operator Meaning

- Generic functions and types
- Trait definitions and implementations
- Trait bounds
- Associated types
- Coherence and the orphan rule
- Operators as trait-driven behavior
- Auto-borrowing and method resolution
- Designing a small reusable abstraction

#### 10. Collections, Strings, Iteration, and I/O

- The `core-min` and `std-full` profiles
- `Vec<T>`, `HashMap<K, V>`, and `HashSet<T>`
- Owned `String` and borrowed string slices
- Iterators and ranges
- Indexing versus safe lookup
- Formatting, printing, and basic I/O
- What is specified versus what still needs implementation

#### 11. Modules and Packages

- File-based modules
- Paths and `use` declarations
- Visibility
- Name resolution
- Package layout
- The `starkpkg.json` manifest
- Dependencies and version resolution
- Organizing a multi-module application

### Part III - Tensor-Aware ML Deployment

#### 12. Why Tensors Belong in the Type System

- Runtime surprises in ordinary inference pipelines
- Shape, dtype, and device as compile-time facts
- Static and symbolic dimensions
- Broadcasting and dimension constraints
- Core isolation: why tensors remain an extension
- The limits of static guarantees

#### 13. Tensor and Model Types

- Enabling the `tensor` extension
- `Dim`, `DType`, shapes, and devices
- Tensor type construction
- Symbolic dimension unification
- Refinement at system boundaries
- Model signature declarations
- Type-checking common tensor operations
- Actionable mismatch diagnostics

#### 14. Importing an ONNX Model

- The role of an existing execution backend
- Inspecting an ONNX artifact
- Generating a typed STARK declaration
- Static and dynamic dimension mapping
- Detecting artifact/signature drift
- Where load-time verification begins
- Illustrative `stark import` workflow

#### 15. A Typed Computer-Vision Pipeline

- Input bytes to validated image data
- Resize, normalize, batch, and layout conversion
- Invoking a model through its generated signature
- Postprocessing typed outputs
- Handling errors at boundaries
- The complete illustrative pipeline
- What the planned prototype must prove

#### 16. Breaking the Pipeline on Purpose

- Incompatible dimensions
- Incorrect element types
- Incompatible device placement
- Stale or incorrect model signatures
- Following constraint provenance
- Turning diagnostics into design feedback
- A defect corpus for repeatable demonstrations

### Part IV - Inside the Project

#### 17. From Source Text to Meaning

- Lexing and nested comments
- Recursive-descent and Pratt parsing
- Arena-allocated AST nodes and source spans
- Name resolution and HIR
- Type checking and local inference
- Ownership and borrow analysis
- Why diagnostics are part of the language design

#### 18. From Checked Program to Execution

- The minimal tree-walking interpreter plan
- Runtime values, calls, control flow, and drop order
- Hosting native standard-library operations
- Generated Rust host glue and ONNX Runtime
- Why STARK is not building a tensor runtime yet
- Evidence-driven options after the prototype

#### 19. Roadmap, Trade-offs, and Open Questions

- The six delivery gates
- Exit criteria and evidence
- Explicit pre-prototype non-goals
- Safety versus implementation complexity
- What would justify a Core language change
- Go, revise, or stop at the decision checkpoint
- Reading performance claims responsibly

#### 20. Contributing to STARK

- Navigating the repository
- Normative, extension, implementation, and archived documents
- The specification-fixture workflow
- Finding and reporting specification defects
- Implementing the lexer and parser
- Tests, diagnostics, and conformance
- A first-contribution checklist

### Appendices

#### A. Core v1 Syntax Quick Reference

- Declarations, expressions, patterns, and types
- Operator precedence
- Reserved words and lexical conventions

#### B. Core Type and Safety Reference

- Primitive and composite types
- Copy versus move summary
- Borrowing rules
- Coercions, traps, and checks

#### C. Standard Library Quick Reference

- Prelude
- Core traits
- Collections, strings, iteration, math, I/O, and errors
- Conformance profiles

#### D. Tensor Extension Quick Reference

- Tensor type parameters
- Dimension notation
- Supported operation-typing rules
- Model declarations and ONNX mapping

#### E. Diagnostic Catalog

- Syntax, name-resolution, type, move, borrow, and extension errors
- How to read spans, causes, and suggested fixes

#### F. Glossary

#### G. Further Reading and Repository Map

## Chapter pattern

Most chapters will use a consistent teaching sequence:

1. A concrete problem or failure mode.
2. A small STARK example.
3. A line-by-line explanation.
4. The underlying rule.
5. A deliberately broken example and diagnostic.
6. A practical design note or comparison.
7. A concise recap and exercises.

## Visual and layout direction

- **Page size:** 7 x 9.25 inches, optimized for screen reading and print-on-demand.
- **Length target:** approximately 180-220 pages.
- **Typography:** readable serif body, compact sans-serif headings, monospaced code with clear bold/italic token emphasis.
- **Palette:** charcoal, warm paper, rust orange, and restrained tensor-blue accents; fully legible in grayscale.
- **Cover concept:** an original engraved-style Australian wedge-tailed eagle assembled from subtle geometric tensor contours. No copied publisher marks, colophons, or trade dress.
- **Interior devices:** chapter-opening illustrations, margin vocabulary, status callouts, warning boxes, specification notes, and full-width annotated code walkthroughs.
- **Accessibility:** tagged reading order where the PDF toolchain permits, high contrast, descriptive captions, no meaning conveyed by color alone, and selectable text.

## Source hierarchy

The manuscript will use sources in this order:

1. `STARKLANG/docs/spec/` for normative Core v1 behavior.
2. `STARKLANG/docs/extensions/Tensor-Model-Types.md` for tensor extension v0.1.
3. `STARKLANG/docs/ROADMAP.md` and `STARKLANG/docs/PLAN.md` for implementation direction.
4. Current `starkc/` source for implementation status.
5. Archived documents only for explicitly labeled history.

## Production plan

1. Freeze this outline and editorial conventions.
2. Build a source-of-truth fact sheet from the normative specifications.
3. Draft Parts I and II with a continuous Core v1 example.
4. Draft Part III as explicitly labeled extension and illustrative workflow material.
5. Draft Part IV from the active roadmap, plan, and compiler scaffold.
6. Create original cover art and a reusable interior design system.
7. Generate the PDF with bookmarks, internal links, code styling, headers, footers, and an index-ready glossary.
8. Render every page to images and inspect for overflow, clipped code, awkward breaks, and low-contrast elements.
9. Validate metadata, page count, selectable text, links, and terminology consistency.

## Acceptance criteria for the final PDF

- No statement implies that a compiler, package manager, tensor runtime, or ONNX importer exists unless verified in the repository at production time.
- Every code listing is labeled by status and is consistent with the relevant specification.
- Core and extension features are never blended without an explicit boundary.
- The PDF contains a polished cover, copyright/title pages, linked table of contents, chapter navigation, page numbers, syntax-highlighted code, glossary, and source notes.
- Final rendered pages contain no clipped text, overlaps, broken glyphs, orphaned headings, or unreadably small code.

