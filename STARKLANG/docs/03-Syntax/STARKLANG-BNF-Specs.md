# STARKLANG Formal Grammar Specification (BNF/EBNF)

This document defines the core syntax of STARKLANG using a formal BNF/EBNF grammar notation. This grammar can be used for parser generation, IDE tooling, syntax validation, and compiler frontend implementation.

---

## 🎯 Grammar Notation Used: **EBNF** (Extended Backus–Naur Form)

- `|` → Alternation (OR)
- `{}` → Zero or more repetitions
- `[]` → Optional
- `()` → Grouping

---

## 📚 EBNF Grammar Definitions

```ebnf
program         ::= { import_stmt } { declaration }

import_stmt     ::= "import" identifier [ "as" identifier ] ";"

// Declarations

declaration     ::= function_decl
                 | struct_decl
                 | enum_decl
                 | trait_decl
                 | ml_pipeline_decl
                 | let_decl
                 | export_decl

export_decl     ::= "export" declaration

// Functions

function_decl   ::= [ "spawn" | "@serverless" | "@server" ] "fn" identifier
                    "(" [ param_list ] ")" "->" type ":" block

param_list      ::= param { "," param }
param           ::= identifier ":" type

// Traits

trait_decl      ::= "trait" identifier ":" INDENT { trait_function_decl } DEDENT
trait_function_decl ::= "fn" identifier "(" [ param_list ] ")" "->" type

// Types

type            ::= primitive_type
                 | composite_type
                 | generic_type
                 | "Model" | "JSON" | "Void"

primitive_type  ::= "Int8" | "Int16" | "Int32" | "Int64"
                 | "UInt8" | "Float32" | "Float64" | "Bool" | "Char"

composite_type  ::= "Array" "<" type ">"
                 | "List" "<" type ">"
                 | "Tuple" "<" type_list ">"
                 | "Map" "<" type "," type ">"
                 | struct_type | enum_type

generic_type    ::= identifier "<" type_list ">"
type_list       ::= type { "," type }

// Blocks

block           ::= INDENT { statement } DEDENT

statement       ::= let_decl
                 | assign_stmt
                 | if_stmt
                 | while_stmt
                 | for_stmt
                 | return_stmt
                 | expression ";"

// Let and Assignment

let_decl        ::= [ "immutable" ] "let" identifier [":" type] "=" expression ";"
assign_stmt     ::= identifier "=" expression ";"

// Control Flow

if_stmt         ::= "if" expression ":" block [ "else" ":" block ]
while_stmt      ::= "while" expression ":" block
for_stmt        ::= "for" identifier "in" expression ":" block
return_stmt     ::= "return" expression ";"

// Expressions

expression      ::= literal
                 | identifier
                 | function_call
                 | binary_expr
                 | unary_expr
                 | member_access
                 | index_access
                 | lambda_expr

literal         ::= number | string | "true" | "false" | "null"

function_call   ::= identifier "(" [ arg_list ] ")"
arg_list        ::= expression { "," expression }

binary_expr     ::= expression binary_op expression
binary_op       ::= "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "and" | "or"

unary_expr      ::= "-" expression | "not" expression

member_access   ::= expression "." identifier
index_access    ::= expression "[" expression "]"
lambda_expr     ::= "(" [ param_list ] ")" "->" expression

// Structs and Enums

struct_decl     ::= "struct" identifier ":" INDENT { field_decl } DEDENT
field_decl      ::= identifier ":" type

enum_decl       ::= "enum" identifier ":" INDENT { enum_variant } DEDENT
enum_variant    ::= identifier [ "(" type_list ")" ]

// ML Pipelines DSL

ml_pipeline_decl ::= "ml_pipeline" identifier ":" INDENT { ml_stage } DEDENT
ml_stage         ::= "load data from" string
                  | "preprocess with" ml_transform_list
                  | "train using" ml_model_def
                  | "validate with" ml_validation
                  | "deploy as service" string

ml_transform_list ::= ml_transform { "," ml_transform }
ml_transform     ::= identifier [ "(" arg_list ")" ]
ml_model_def     ::= identifier "(" arg_list ")"
ml_validation    ::= identifier "(" arg_list ")"
```

---

## 🧠 Compiler Implementation Notes

- INDENT/DEDENT tokens are lexical (Python-style block markers).
- Trait resolution and implementation checks are handled post-AST.
- Operator precedence and associativity are defined in parsing tables.

---

## 🔄 Future Grammar Extensions

- Pattern guards in `match`
- Type constraints inline in function params (`T: Trait`)
- Inline async blocks (`async { ... }`)
- Decorator support beyond `@serverless`, `@server`

---

## ✅ Summary

This EBNF grammar defines the full structural specification of STARKLANG. It provides the basis for writing parsers, syntax highlighters, and AST transformers. The grammar is cleanly extensible and future-safe for evolving STARK into a powerful AI-native, cloud-first language.

