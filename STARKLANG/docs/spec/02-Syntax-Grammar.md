# STARK Syntax Grammar Specification

## Overview
This document defines the concrete syntax grammar for STARK using Extended Backus-Naur Form (EBNF).

## Grammar Notation
```
::= means "is defined as"
|   means "or" (alternative)
?   means "optional" (zero or one)
*   means "zero or more"
+   means "one or more"
()  means "grouping"
[]  means "optional"
{}  means "zero or more repetitions"
```

## Top-Level Grammar

### Program
```ebnf
Program ::= Item*

Item ::= Visibility? (Function
       | Struct
       | Enum  
       | Trait
       | Impl
       | Const
       | TypeAlias
       | Use
       | Module)

Visibility ::= 'pub' | 'priv'
```

### Generic Parameters and Arguments
```ebnf
GenericParams ::= '<' GenericParam (',' GenericParam)* ','? '>'

GenericParam ::= IDENTIFIER (':' TraitBounds)?

TraitBounds ::= TraitBound ('+' TraitBound)*

TraitBound ::= Path GenericArgs?

GenericArgs ::= '<' GenericArg (',' GenericArg)* ','? '>'

GenericArg ::= Type
             | IDENTIFIER '=' Type          // Associated type binding, e.g. Iterator<Item = T>
```

Notes:
- Generic parameters may appear on functions, structs, enums, traits, impl blocks, and type aliases.
- Generic arguments in *expressions* are inferred at the use site. The only
  explicit form is `path::<Args>` on a path expression (see
  `PathExpression`), for calls where inference is impossible, e.g.
  `size_of::<Int32>()`.

### Function Definition
```ebnf
Function ::= FunctionSig Block

FunctionSig ::= 'fn' IDENTIFIER GenericParams? '(' ParameterList? ')' ('->' ReturnType)?

ReturnType ::= Type | '!'                   // '!' is the never type (diverging function)

ParameterList ::= Receiver (',' Parameter)* ','?
                | Parameter (',' Parameter)* ','?

Receiver ::= 'self'
           | '&' 'self'
           | '&' 'mut' 'self'

Parameter ::= IDENTIFIER ':' Type
            | 'mut' IDENTIFIER ':' Type

Block ::= '{' Statement* Expression? '}'
```

Block semantics:
- The optional final `Expression` has no terminating semicolon and is the
  block's value; a block without one evaluates to `Unit`.
- Disambiguation: at each position inside a block, the parser first attempts
  to parse a `Statement`; a trailing `Expression` is recognized only
  immediately before `}`. An expression followed by `;` is always a statement.

Notes:
- A receiver is only valid on functions declared inside a `trait` or `impl` block.
- A function without a `->` annotation returns `Unit`. Return types are never inferred.

### Data Type Definitions
```ebnf
Struct ::= 'struct' IDENTIFIER GenericParams? '{' FieldList? '}'

FieldList ::= Field (',' Field)* ','?

Field ::= IDENTIFIER ':' Type
        | 'pub' IDENTIFIER ':' Type

Enum ::= 'enum' IDENTIFIER GenericParams? '{' VariantList? '}'

VariantList ::= Variant (',' Variant)* ','?

Variant ::= IDENTIFIER
          | IDENTIFIER '(' TypeList ')'
          | IDENTIFIER '{' FieldList '}'

Trait ::= 'trait' IDENTIFIER GenericParams? '{' TraitItem* '}'

TraitItem ::= FunctionSig ';'               // Required method (signature only)
            | FunctionSig Block             // Method with default body
            | AssociatedType

AssociatedType ::= 'type' IDENTIFIER ';'

Impl ::= 'impl' GenericParams? Type '{' ImplItem* '}'
       | 'impl' GenericParams? TraitBound 'for' Type '{' ImplItem* '}'

ImplItem ::= Visibility? Function
           | 'type' IDENTIFIER '=' Type ';' // Associated type value
```

### Module Definition
```ebnf
Module ::= 'mod' IDENTIFIER ';'
        | 'mod' IDENTIFIER ModuleBlock

ModuleBlock ::= '{' Item* '}'
```

### Type Alias
```ebnf
TypeAlias ::= 'type' IDENTIFIER GenericParams? '=' Type ';'
```

### Statements
```ebnf
Statement ::= ';'                    // Empty statement
            | Expression ';'         // Expression statement
            | BlockExpression ';'?   // Block-formed expression statement
            | 'let' LetStatement
            | 'return' Expression? ';'
            | 'break' Expression? ';'
            | 'continue' ';'

BlockExpression ::= Block
                  | IfExpression
                  | MatchExpression
                  | LoopExpression

LetStatement ::= IDENTIFIER ':' Type ';'
               | 'mut' IDENTIFIER ':' Type ';'
               | IDENTIFIER ':' Type '=' Expression ';'
               | IDENTIFIER '=' Expression ';'
               | 'mut' IDENTIFIER ':' Type '=' Expression ';'
               | 'mut' IDENTIFIER '=' Expression ';'
```

Block-formed expression statements:
- An expression whose outermost form is a block, `if`, `match`, `loop`,
  `while`, or `for` may stand as a statement without a terminating
  semicolon (`if ready { start(); }` mid-block is a statement).
- Statement parsing is greedy: when a block-formed expression begins at
  statement position and is not followed by `;`, it is complete as a
  statement — a following token does NOT extend it into a larger
  expression (`{ if c { } - 1; }` is an `if` statement followed by an
  error, not a subtraction). Parenthesize to use the value:
  `(if c { 1 } else { 2 }) - 1`.
- Exception (trailing expression): a block-formed expression immediately
  before the closing `}` of its block and not followed by `;` is the
  block's trailing `Expression` (its value), not a statement — so
  `fn max(...) -> T { if a > b { a } else { b } }` returns the `if`'s
  value.

### Expressions
```ebnf
Expression ::= AssignmentExpression

AssignmentExpression ::= RangeExpression
                       | RangeExpression AssignOp AssignmentExpression

AssignOp ::= '=' | '+=' | '-=' | '*=' | '/=' | '%=' | '**='
           | '&=' | '|=' | '^=' | '<<=' | '>>='
```

### Place Expressions
The left-hand side of an assignment must be a *place expression* — an
expression that denotes a memory location. Place expressions are:

**SYN-PLACE-001.** The forms in `PlaceExpression` are the complete Core v1
assignment-target syntax. Parentheses do not change whether an expression is
a place. Field, tuple-field, index, and dereference forms are accepted by the
parser and then require the mutable-place typing rules in
`03-Type-System.md`; calls, literals, casts, ranges, aggregates, and other
computed values are never assignment targets.

```ebnf
PlaceExpression ::= Path                              // Variable
                  | PlaceExpression '.' IDENTIFIER    // Field
                  | PlaceExpression '.' INTEGER       // Tuple field
                  | PlaceExpression '[' Expression ']' // Index
                  | '*' Expression                    // Dereference
                  | '(' PlaceExpression ')'
```

Assignment to any other expression form is a compile-time error (checked
semantically; the grammar accepts any expression on the left).

```ebnf
RangeExpression ::= LogicalOrExpression (('..' | '..=') LogicalOrExpression)?

LogicalOrExpression ::= LogicalAndExpression ('||' LogicalAndExpression)*

LogicalAndExpression ::= EqualityExpression ('&&' EqualityExpression)*

EqualityExpression ::= RelationalExpression (('==' | '!=') RelationalExpression)?

RelationalExpression ::= BitwiseOrExpression (('<' | '<=' | '>' | '>=') BitwiseOrExpression)?

BitwiseOrExpression ::= BitwiseXorExpression ('|' BitwiseXorExpression)*

BitwiseXorExpression ::= BitwiseAndExpression ('^' BitwiseAndExpression)*

BitwiseAndExpression ::= ShiftExpression ('&' ShiftExpression)*

ShiftExpression ::= AdditiveExpression (('<<' | '>>') AdditiveExpression)*

AdditiveExpression ::= MultiplicativeExpression (('+' | '-') MultiplicativeExpression)*

MultiplicativeExpression ::= ExponentiationExpression (('*' | '/' | '%') ExponentiationExpression)*

ExponentiationExpression ::= CastExpression ('**' ExponentiationExpression)?

CastExpression ::= UnaryExpression ('as' Type)*

UnaryExpression ::= PostfixExpression
                  | '-' UnaryExpression
                  | '!' UnaryExpression
                  | '~' UnaryExpression
                  | '&' 'mut'? UnaryExpression // Reference (immutable or mutable borrow)
                  | '*' UnaryExpression        // Dereference

PostfixExpression ::= PrimaryExpression
                    | PostfixExpression '(' ArgumentList? ')'     // Function call
                    | PostfixExpression '[' Expression ']'        // Index (or slice, when the index is a range)
                    | PostfixExpression '.' IDENTIFIER            // Field access / method reference
                    | PostfixExpression '.' INTEGER               // Tuple access
                    | PostfixExpression '?'                       // Try operator

ArgumentList ::= Expression (',' Expression)* ','?

PrimaryExpression ::= PathExpression
                    | Literal
                    | '(' Expression ')'    // Grouping
                    | TupleLiteral
                    | ArrayLiteral
                    | StructLiteral
                    | IfExpression
                    | MatchExpression
                    | LoopExpression
                    | Block

PathExpression ::= Path ('::' GenericArgs)?  // e.g. x, String::from, Color::Red, size_of::<Int32>
```

Notes:
- `path::<Args>` (explicit generic arguments on a path expression) is
  permitted only when type inference cannot determine the arguments, e.g.
  `size_of::<Int32>()`. It is the only form of expression-level generic
  arguments in Core v1.
- Chained comparisons (`a < b < c`, `a == b == c`) are syntax errors:
  the relational and equality productions are non-associative (at most one
  operator). Parenthesize to compare a Bool result explicitly.

### Control Flow Expressions
```ebnf
IfExpression ::= 'if' Expression Block ('else' 'if' Expression Block)* ('else' Block)?

MatchExpression ::= 'match' Expression '{' MatchArm* '}'

MatchArm ::= Pattern '=>' Expression ','?

Pattern ::= Literal
          | '_'                                    // Wildcard
          | IDENTIFIER                             // Binding (or unit variant/const, see note)
          | Path                                   // Unit enum variant or const, e.g. Color::Red
          | Path '(' PatternList? ')'              // Tuple enum variant, e.g. Option::Some(x)
          | Path '{' FieldPatternList? '}'         // Struct or struct enum variant
          | '(' PatternList? ')'                   // Tuple
          | '[' PatternList? ']'                   // Array

PatternList ::= Pattern (',' Pattern)* ','?

FieldPatternList ::= FieldPattern (',' FieldPattern)* ','?

FieldPattern ::= IDENTIFIER ':' Pattern
               | IDENTIFIER                        // Shorthand: binds field to same-named variable

LoopExpression ::= 'loop' Block
                 | 'while' Expression Block
                 | 'for' IDENTIFIER 'in' Expression Block
```

**SYN-PATTERN-001.** The productions above are the complete Core v1 pattern
surface. Core v1 has no pattern guards, alternative (`|`) patterns, rest
patterns, binding-mode annotations, or range patterns. A named-field pattern
may list fields in any order but may not repeat a field. Static typing,
exhaustiveness, usefulness, and ownership are defined by
`04-Semantic-Analysis.md` and `CORE-V1-ABSTRACT-MACHINE.md`.

Note on pattern name resolution: a single `IDENTIFIER` pattern that resolves to a
unit enum variant or a constant in scope matches by value; otherwise it introduces
a new binding. Multi-segment `Path` patterns always match by value.

### Literals
```ebnf
Literal ::= INTEGER
          | FLOAT
          | STRING
          | CHAR
          | BOOLEAN

TupleLiteral ::= '(' ')'                                  // Unit value
               | '(' Expression ',' ')'                   // Single-element tuple
               | '(' Expression (',' Expression)+ ','? ')' // N-element tuple

ArrayLiteral ::= '[' ExpressionList? ']'
               | '[' Expression ';' Expression ']'        // Repeat: [value; count]

ExpressionList ::= Expression (',' Expression)* ','?

StructLiteral ::= Path '{' FieldInitList? '}'

FieldInitList ::= FieldInit (',' FieldInit)* ','?

FieldInit ::= IDENTIFIER ':' Expression
            | IDENTIFIER                    // Shorthand: field: field
```

Notes:
- `(expr)` is grouping; `(expr,)` is a single-element tuple; `()` is the unit
  value. The trailing comma distinguishes the 1-tuple from grouping.
- In `[value; count]`, `count` must be a compile-time constant expression of
  an unsigned integer type, and `value`'s type must implement `Copy` (the
  value is copied `count` times).

### Struct Literal Restriction
To avoid ambiguity with block-taking constructs, a struct literal may NOT appear
as the outermost expression in:
- the condition of an `if` or `while`,
- the scrutinee of a `match`,
- the iterable of a `for`.

Wrap the struct literal in parentheses in these positions:
```stark
if (Config { verbose: true }).verbose { do_thing(); }
```

### Types
```ebnf
Type ::= PrimitiveType
       | Path GenericArgs?                 // Named type, possibly generic: Vec<Int32>, Option<T>
       | '[' Type ';' INTEGER ']'          // Array type
       | '[' Type ']'                      // Slice type (unsized; used behind references)
       | TupleType
       | '(' Type ')'                      // Grouping (the type T, not a tuple)
       | '&' Type                          // Reference type
       | '&' 'mut' Type                    // Mutable reference type
       | 'fn' '(' TypeList? ')' ('->' Type)?  // Function type (non-capturing)

TupleType ::= '(' ')'                      // Unit type
            | '(' Type ',' ')'             // Single-element tuple type
            | '(' Type (',' Type)+ ','? ')' // N-element tuple type

PrimitiveType ::= 'Int8' | 'Int16' | 'Int32' | 'Int64'
                | 'UInt8' | 'UInt16' | 'UInt32' | 'UInt64'
                | 'Float32' | 'Float64'
                | 'Bool' | 'Char' | 'String' | 'Unit' | 'str'

TypeList ::= Type (',' Type)* ','?
```

Notes:
- A function type without `->` returns `Unit`.
- The never type `!` may appear only as a function return type (see `ReturnType`).
- `(T)` is the type `T` (grouping); the single-element tuple type is `(T,)`,
  mirroring `TupleLiteral` on the value side.

### Other Constructs
```ebnf
Const ::= 'const' IDENTIFIER ':' Type '=' Expression ';'

Use ::= 'use' UseTree ';'

UseTree ::= Path ('as' IDENTIFIER)?
          | Path '::' '*'
          | Path '::' 'self'
          | Path '::' '{' UseTreeList? '}'

UseTreeList ::= UseTree (',' UseTree)* ','?

Path ::= PathSegment ('::' PathSegment)*
PathSegment ::= IDENTIFIER | PrimitiveType | 'self' | 'Self' | 'super' | 'crate'
```

Notes:
- `Self` is valid only inside a `trait` or `impl` block, and only as the
  *first* segment of a path. As a complete one-segment path in type position
  it denotes the implementing type; with further segments it projects an
  associated item, e.g. `Self::Item`.
- `self`, `super`, and `crate` are likewise valid only as leading segments
  (`self`/`super` may repeat at the front: `super::super::x`).
- A `PrimitiveType` keyword is valid only as the *first* segment, where it
  names the primitive's associated items: `String::from(s)`. (Primitive type
  names are keywords in the lexical grammar, so without this rule such calls
  would be unparseable.)

## Operator Precedence (Highest to Lowest)
1. Primary expressions, field access, array access, function calls, try operator (`?`)
2. Unary operators (-, !, ~, &, &mut, *)
3. Cast (`as`)
4. Exponentiation (**)
5. Multiplicative (*, /, %)
6. Additive (+, -)
7. Shift (<<, >>)
8. Bitwise AND (&)
9. Bitwise XOR (^)
10. Bitwise OR (|)
11. Relational (<, <=, >, >=)
12. Equality (==, !=)
13. Logical AND (&&)
14. Logical OR (||)
15. Range (.., ..=)
16. Assignment (=, +=, -=, etc.)

## Associativity
- Left associative: Most binary operators
- Right associative: Assignment operators, exponentiation
- Non-associative: Comparison operators, range operators

## Statement vs Expression
- Statements do not return values and end with semicolons (block-formed
  expression statements may omit the semicolon — see Statements)
- Expressions return values and can be used as statements with semicolons
- Blocks are expressions (return value of last expression, or Unit if none)
- Control flow constructs are expressions

## Whitespace and Semicolons
- Semicolons are required to terminate statements, except after block-formed
  expression statements (see Statements)
- Semicolons are optional after the last expression in a block
- Newlines are not significant except for line comments
- Trailing commas are allowed in lists

## Parsing Notes
**SYN-RECOVERY-001.** For Core conformance, parsing observes only acceptance
or rejection and the required primary diagnostic category/location.
Token insertion/deletion, synchronization points, cascaded diagnostics,
partial syntax trees, and recovery continuation are implementation-defined
tooling behavior. Recovery must not accept a source that the grammar rejects
or reject a source that the grammar accepts.

- When a `>` is expected in generic-argument position and the next token is
  `>>`, `>>=`, or `>=` (maximal munch), the parser MUST split off a single
  `>` and re-tokenize the remainder (`>`, `>=`, or `=` respectively) — so
  `Vec<Vec<Int32>>`, `Vec<Vec<Int32>>= v`, and `Vec<Int32>= v` all parse.
- Nested tuple-field access `pair.0.1` lexes the `0.1` as a FLOAT token
  (maximal munch); after `.`, the parser MUST split an unsuffixed
  digits-`.`-digits FLOAT into two INTEGER tuple indices.
- In expression position, a bare `<` after an identifier is always the
  relational operator; explicit generic arguments require the `::<` form
  (turbofish), so no lookahead disambiguation is required.

## Informative Future Grammar Sketches (Non-Normative)
Reserved grammar constructs for later implementation:
```ebnf
// Async functions (future)
AsyncFunction ::= 'async' 'fn' IDENTIFIER '(' ParameterList? ')' ('->' Type)? Block

// Lambda expressions / capturing closures (future)
Lambda ::= '|' ParameterList? '|' (Expression | Block)

// Open-ended ranges (future)
OpenRange ::= Expression '..' | '..' Expression | '..'

// Lifetime annotations (future)
LifetimeParam ::= '\'' IDENTIFIER
```
## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
