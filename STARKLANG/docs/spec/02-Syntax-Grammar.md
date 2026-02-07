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

### Function Definition
```ebnf
Function ::= 'fn' IDENTIFIER '(' ParameterList? ')' ('->' Type)? Block

ParameterList ::= Parameter (',' Parameter)* ','?

Parameter ::= IDENTIFIER ':' Type
            | 'mut' IDENTIFIER ':' Type

Block ::= '{' Statement* '}'
```

### Data Type Definitions
```ebnf
Struct ::= 'struct' IDENTIFIER '{' FieldList? '}'

FieldList ::= Field (',' Field)* ','?

Field ::= IDENTIFIER ':' Type
        | 'pub' IDENTIFIER ':' Type

Enum ::= 'enum' IDENTIFIER '{' VariantList? '}'

VariantList ::= Variant (',' Variant)* ','?

Variant ::= IDENTIFIER
          | IDENTIFIER '(' TypeList ')'
          | IDENTIFIER '{' FieldList '}'

Trait ::= 'trait' IDENTIFIER '{' TraitItem* '}'

TraitItem ::= Function
            | Type

Impl ::= 'impl' Type '{' ImplItem* '}'
       | 'impl' IDENTIFIER 'for' Type '{' ImplItem* '}'

ImplItem ::= Function
```

### Module Definition
```ebnf
Module ::= 'mod' IDENTIFIER ';'
        | 'mod' IDENTIFIER ModuleBlock

ModuleBlock ::= '{' Item* '}'
```

### Type Alias
```ebnf
TypeAlias ::= 'type' IDENTIFIER '=' Type ';'
```

### Statements
```ebnf
Statement ::= ';'                    // Empty statement
            | Expression ';'         // Expression statement
            | 'let' LetStatement
            | 'return' Expression? ';'
            | 'break' Expression? ';'
            | 'continue' ';'
            | Block

LetStatement ::= IDENTIFIER ':' Type ';'
               | 'mut' IDENTIFIER ':' Type ';'
               | IDENTIFIER ':' Type '=' Expression ';'
               | IDENTIFIER '=' Expression ';'
               | 'mut' IDENTIFIER ':' Type '=' Expression ';'
               | 'mut' IDENTIFIER '=' Expression ';'
```

### Expressions
```ebnf
Expression ::= AssignmentExpression

AssignmentExpression ::= LogicalOrExpression
                       | LogicalOrExpression AssignOp AssignmentExpression

AssignOp ::= '=' | '+=' | '-=' | '*=' | '/=' | '%=' | '**='
           | '&=' | '|=' | '^=' | '<<=' | '>>='

LogicalOrExpression ::= LogicalAndExpression ('||' LogicalAndExpression)*

LogicalAndExpression ::= EqualityExpression ('&&' EqualityExpression)*

EqualityExpression ::= RelationalExpression (('==' | '!=') RelationalExpression)*

RelationalExpression ::= BitwiseOrExpression (('<' | '<=' | '>' | '>=') BitwiseOrExpression)*

BitwiseOrExpression ::= BitwiseXorExpression ('|' BitwiseXorExpression)*

BitwiseXorExpression ::= BitwiseAndExpression ('^' BitwiseAndExpression)*

BitwiseAndExpression ::= ShiftExpression ('&' ShiftExpression)*

ShiftExpression ::= AdditiveExpression (('<<' | '>>') AdditiveExpression)*

AdditiveExpression ::= MultiplicativeExpression (('+' | '-') MultiplicativeExpression)*

MultiplicativeExpression ::= ExponentiationExpression (('*' | '/' | '%') ExponentiationExpression)*

ExponentiationExpression ::= UnaryExpression ('**' ExponentiationExpression)?

UnaryExpression ::= PostfixExpression
                  | '-' UnaryExpression
                  | '!' UnaryExpression
                  | '~' UnaryExpression
                  | '&' UnaryExpression        // Reference
                  | '*' UnaryExpression        // Dereference

PostfixExpression ::= PrimaryExpression
                    | PostfixExpression '(' ArgumentList? ')'     // Function call
                    | PostfixExpression '[' Expression ']'        // Array access
                    | PostfixExpression '.' IDENTIFIER            // Field access
                    | PostfixExpression '.' INTEGER               // Tuple access
                    | PostfixExpression '?'                       // Try operator

ArgumentList ::= Expression (',' Expression)* ','?

PrimaryExpression ::= IDENTIFIER
                    | Literal
                    | '(' Expression ')'
                    | ArrayLiteral
                    | StructLiteral
                    | IfExpression
                    | MatchExpression
                    | LoopExpression
                    | Block
```

### Control Flow Expressions
```ebnf
IfExpression ::= 'if' Expression Block ('else' 'if' Expression Block)* ('else' Block)?

MatchExpression ::= 'match' Expression '{' MatchArm* '}'

MatchArm ::= Pattern '=>' Expression ','?

Pattern ::= IDENTIFIER
          | Literal
          | '_'                                    // Wildcard
          | IDENTIFIER '(' PatternList? ')'        // Enum variant
          | '(' PatternList? ')'                   // Tuple
          | '[' PatternList? ']'                   // Array

PatternList ::= Pattern (',' Pattern)* ','?

LoopExpression ::= 'loop' Block
                 | 'while' Expression Block
                 | 'for' IDENTIFIER 'in' Expression Block
```

### Literals
```ebnf
Literal ::= INTEGER
          | FLOAT
          | STRING
          | CHAR
          | BOOLEAN

ArrayLiteral ::= '[' ExpressionList? ']'

ExpressionList ::= Expression (',' Expression)* ','?

StructLiteral ::= IDENTIFIER '{' FieldInitList? '}'

FieldInitList ::= FieldInit (',' FieldInit)* ','?

FieldInit ::= IDENTIFIER ':' Expression
            | IDENTIFIER                    // Shorthand: field: field
```

### Types
```ebnf
Type ::= PrimitiveType
       | IDENTIFIER                        // Named type
       | '[' Type ';' INTEGER ']'          // Array type
       | '[' Type ']'                      // Slice type
       | '(' TypeList? ')'                 // Tuple type
       | '&' Type                          // Reference type
       | '&' 'mut' Type                    // Mutable reference type
       | Type '->' Type                    // Function type

PrimitiveType ::= 'Int8' | 'Int16' | 'Int32' | 'Int64'
                | 'UInt8' | 'UInt16' | 'UInt32' | 'UInt64'
                | 'Float32' | 'Float64'
                | 'Bool' | 'Char' | 'String' | 'Unit' | 'str'

TypeList ::= Type (',' Type)* ','?
```

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
PathSegment ::= IDENTIFIER | 'self' | 'super' | 'crate'
```

## Operator Precedence (Highest to Lowest)
1. Primary expressions, field access, array access, function calls, try operator (`?`)
2. Unary operators (-, !, ~, &, *)
3. Exponentiation (**)
4. Multiplicative (*, /, %)
5. Additive (+, -)
6. Shift (<<, >>)
7. Bitwise AND (&)
8. Bitwise XOR (^)
9. Bitwise OR (|)
10. Relational (<, <=, >, >=)
11. Equality (==, !=)
12. Logical AND (&&)
13. Logical OR (||)
14. Assignment (=, +=, -=, etc.)

## Associativity
- Left associative: Most binary operators
- Right associative: Assignment operators, exponentiation
- Non-associative: Comparison operators

## Statement vs Expression
- Statements do not return values and end with semicolons
- Expressions return values and can be used as statements with semicolons
- Blocks are expressions (return value of last expression, or Unit if none)
- Control flow constructs are expressions

## Whitespace and Semicolons
- Semicolons are required to terminate statements
- Semicolons are optional after the last expression in a block
- Newlines are not significant except for line comments
- Trailing commas are allowed in lists

## Grammar Extensions for Future Features
Reserved grammar constructs for later implementation:
```ebnf
// Generic types (future)
GenericType ::= IDENTIFIER '<' TypeList '>'

// Async functions (future)
AsyncFunction ::= 'async' 'fn' IDENTIFIER '(' ParameterList? ')' ('->' Type)? Block

// Lambda expressions (future)
Lambda ::= '|' ParameterList? '|' (Expression | Block)
```
## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
