# STARK Core Language Specification Overview

## Introduction
This document provides an overview of the complete STARK core language specification. The documents in `docs/spec/` are normative for Core v1. The core language defines the general-purpose language surface (lexing, syntax, types, semantics, memory, modules, and standard library). Non-core extensions are defined separately.

## Design Philosophy

### Core Principles
1. **Memory Safety**: Prevent common memory errors through ownership and borrowing
2. **Performance**: Zero-cost abstractions and predictable execution
3. **Clarity**: Simple, explicit syntax and semantics
4. **Pragmatism**: A minimal, implementable Core v1
5. **Interoperability**: Clear boundaries for future extensions

### Language Goals
- A safe, compiled, general-purpose language core
- Compile-time guarantees for memory and type safety
- Simple, predictable semantics suitable for tooling
- Clear extension points for domain-specific features

## Specification Structure

### 1. Lexical Grammar ([01-Lexical-Grammar.md](./01-Lexical-Grammar.md))
Defines how source code is tokenized:
- **Keywords**: Control flow, declarations, types, operators
- **Identifiers**: Variables, functions, types (snake_case/PascalCase)
- **Literals**: Integers, floats, strings, characters, booleans
- **Operators**: Arithmetic, comparison, logical, bitwise, assignment
- **Comments**: Single-line (`//`) and multi-line (`/* */`)
- **Whitespace**: Space, tab, newline handling

### 2. Syntax Grammar ([02-Syntax-Grammar.md](./02-Syntax-Grammar.md))
Defines the concrete syntax using EBNF:
- **Program Structure**: Items (functions, structs, enums, traits)
- **Expressions**: Precedence, associativity
- **Statements**: Variable declarations, control flow, returns
- **Type Syntax**: Primitives, composites, references, functions
- **Pattern Matching**: Destructuring and exhaustiveness

### 3. Type System ([03-Type-System.md](./03-Type-System.md))
Comprehensive type system with safety guarantees:
- **Primitive Types**: Integers, floats, booleans, characters, strings
- **Composite Types**: Arrays, tuples, structs, enums
- **Reference Types**: Immutable and mutable references
- **Ownership Model**: Move semantics, borrowing rules
- **Type Inference**: Local and function return type inference
- **Trait System**: Interfaces and generic constraints

### 4. Semantic Analysis ([04-Semantic-Analysis.md](./04-Semantic-Analysis.md))
Rules for meaningful program validation:
- **Symbol Resolution**: Scoping, shadowing, name lookup
- **Type Checking**: Assignment compatibility, function calls
- **Ownership Analysis**: Move tracking, borrow checking
- **Control Flow**: Reachability, return path analysis
- **Pattern Exhaustiveness**: Match completeness checking
- **Error Reporting**: Comprehensive diagnostics with suggestions

### 5. Memory Model ([05-Memory-Model.md](./05-Memory-Model.md))
Memory safety through compile-time analysis:
- **Ownership Rules**: Single ownership, automatic cleanup
- **Move Semantics**: Explicit ownership transfer
- **Borrowing System**: Immutable and mutable references
- **Lifetime Tracking**: Reference validity guarantees
- **Stack vs Heap**: Allocation strategy and layout
- **Drop System**: Automatic and manual resource cleanup

### 6. Standard Library ([06-Standard-Library.md](./06-Standard-Library.md))
Essential types and functions for practical programming:
- **Core Types**: Option, Result, Box, Vec, HashMap
- **String Handling**: Unicode-aware string operations
- **Collections**: Dynamic arrays, hash tables, sets
- **IO Operations**: File handling, console output
- **Math Functions**: Arithmetic, trigonometric, random numbers
- **Error Handling**: Structured error types and propagation

### 7. Modules and Packages ([07-Modules-and-Packages.md](./07-Modules-and-Packages.md))
Defines module structure, visibility, and import resolution:
- **Modules**: File and directory layout
- **Visibility**: `pub`/`priv` rules
- **Imports**: `use` paths, trees, and aliasing
- **Packages**: Manifest and dependency resolution

### 8. Non-Core Extensions (../extensions/AI-Extensions.md)
Non-core language extensions live outside Core v1 and are optional. See `docs/extensions/AI-Extensions.md`.

## Core Language Features

### Variables and Mutability
```stark
let x = 42              // Immutable by default
let mut y = 10          // Explicitly mutable
const MAX_SIZE: Int32 = 1000  // Compile-time constant
```

### Functions
```stark
fn add(a: Int32, b: Int32) -> Int32 {
    a + b
}

fn greet(name: &str) {
    println("Hello, " + name)
}
```

### Ownership and Borrowing
```stark
fn consume(s: String) {
    // s is owned here
}

fn borrow(s: &String) -> Int32 {
    s.len()
}

fn mutate(s: &mut String) {
    s.push('!')
}
```

## Implementation Phases

### Phase 1: Core MVP
**Goal**: A complete, implementable Core v1
- Lexer and parser for core syntax
- Type checker with ownership analysis
- Module system and import resolution
- Minimal standard library

### Phase 2: Tooling and Stability
**Goal**: A stable core suitable for real use
- Improved diagnostics and error recovery
- Formatter and basic tooling support
- Expanded standard library coverage

## Success Criteria

### Correctness
- [ ] Memory safety enforced by ownership and borrowing
- [ ] Deterministic type checking and inference
- [ ] Exhaustive match checking

### Developer Experience
- [ ] Clear, actionable error messages
- [ ] Predictable module and import rules
- [ ] Stable core language surface

## Next Steps

1. **Finalize Core Grammar**: Resolve remaining ambiguities in syntax and lexing
2. **Solidify Type Rules**: Confirm inference and trait constraints
3. **Define Module Rules**: Ensure deterministic resolution and visibility
4. **Validate Stdlib Surface**: Confirm minimal APIs and behaviors

This specification provides a focused foundation for implementing a safe, performant, general-purpose language core.

## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
