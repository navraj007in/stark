# STARK Language Formal Grammar Specification

This document provides the complete formal grammar specification for the STARK programming language using Extended Backus-Naur Form (EBNF) notation.

## Grammar Notation

- `::=` - Definition
- `|` - Alternation (OR)
- `{}` - Zero or more repetitions
- `[]` - Optional element
- `()` - Grouping
- `<>` - Non-terminal symbols
- Terminal symbols are enclosed in quotes

## 1. Lexical Grammar

### 1.1 Whitespace and Comments

```ebnf
<whitespace>        ::= ' ' | '\t' | '\r' | '\n'
<comment>           ::= <line_comment> | <block_comment>
<line_comment>      ::= '//' {<any_char_except_newline>} '\n'
<block_comment>     ::= '/*' {<any_char>} '*/'
```

### 1.2 Identifiers and Keywords

```ebnf
<identifier>        ::= <letter> {<letter> | <digit> | '_'}
<letter>            ::= 'a'..'z' | 'A'..'Z'
<digit>             ::= '0'..'9'

<keyword>           ::= 'fn' | 'let' | 'immutable' | 'if' | 'else' | 'while' | 'for' | 'in'
                     | 'return' | 'break' | 'continue' | 'match' | 'case'
                     | 'struct' | 'enum' | 'trait' | 'impl' | 'self' | 'Self'
                     | 'import' | 'export' | 'as' | 'from' | 'module' | 'package'
                     | 'async' | 'await' | 'spawn' | 'actor' | 'send' | 'receive'
                     | 'true' | 'false' | 'null' | 'and' | 'or' | 'not'
                     | 'type' | 'alias' | 'where' | 'with'
                     | 'tensor' | 'model' | 'dataset' | 'pipeline'
                     | 'try' | 'catch' | 'finally' | 'throw'
                     | 'public' | 'private' | 'protected' | 'internal'
                     | 'cloud' | 'serverless' | 'service' | 'deploy'
```

### 1.3 Literals

```ebnf
<literal>           ::= <integer_literal> | <float_literal> | <string_literal> 
                     | <char_literal> | <bool_literal> | <null_literal>
                     | <tensor_literal>

<integer_literal>   ::= <decimal_int> | <hex_int> | <binary_int> | <octal_int>
<decimal_int>       ::= <digit> {<digit> | '_'}
<hex_int>           ::= '0x' <hex_digit> {<hex_digit> | '_'}
<binary_int>        ::= '0b' <binary_digit> {<binary_digit> | '_'}
<octal_int>         ::= '0o' <octal_digit> {<octal_digit> | '_'}

<float_literal>     ::= <decimal_int> '.' <decimal_int> [<exponent>]
                     | <decimal_int> <exponent>
<exponent>          ::= ('e' | 'E') ['+' | '-'] <decimal_int>

<string_literal>    ::= '"' {<string_char>} '"' 
                     | "'" {<string_char>} "'"
                     | 'f"' {<string_char> | '{' <expression> '}'} '"'
<char_literal>      ::= "'" <char> "'"

<bool_literal>      ::= 'true' | 'false'
<null_literal>      ::= 'null'

<tensor_literal>    ::= 'tensor' '[' <tensor_elements> ']' ['@' <device>]
<tensor_elements>   ::= <expression> {',' <expression>}
<device>            ::= 'cpu' | 'gpu' | 'tpu' | <identifier>
```

### 1.4 Operators and Delimiters

```ebnf
<operator>          ::= '+' | '-' | '*' | '/' | '%' | '**' | '@'
                     | '==' | '!=' | '<' | '<=' | '>' | '>='
                     | '=' | '+=' | '-=' | '*=' | '/=' | '%=' | '**='
                     | '&' | '|' | '^' | '~' | '<<' | '>>'
                     | '&&' | '||' | '!'
                     | '.' | '..' | '..=' | '->' | '=>' | '::'

<delimiter>         ::= '(' | ')' | '[' | ']' | '{' | '}' 
                     | ',' | ';' | ':' | '?' | '!'
```

## 2. Syntactic Grammar

### 2.1 Program Structure

```ebnf
<program>           ::= {<module_item>}

<module_item>       ::= <import_decl>
                     | <export_decl>
                     | <function_decl>
                     | <struct_decl>
                     | <enum_decl>
                     | <trait_decl>
                     | <impl_decl>
                     | <type_alias>
                     | <global_let>
                     | <actor_decl>
                     | <model_decl>
                     | <pipeline_decl>
```

### 2.2 Import and Export

```ebnf
<import_decl>       ::= 'import' <import_path> [<import_items>] ';'
<import_path>       ::= <identifier> {'.' <identifier>}
<import_items>      ::= '::' '{' <import_item> {',' <import_item>} '}'
                     | '::' '*'
                     | 'as' <identifier>
<import_item>       ::= <identifier> ['as' <identifier>]

<export_decl>       ::= 'export' <module_item>
                     | 'export' '{' <identifier> {',' <identifier>} '}'
```

### 2.3 Type System

```ebnf
<type>              ::= <primitive_type>
                     | <composite_type>
                     | <reference_type>
                     | <function_type>
                     | <generic_type>
                     | <ai_type>
                     | <qualified_type>
                     | '?'<type>

<primitive_type>    ::= 'i8' | 'i16' | 'i32' | 'i64' | 'i128'
                     | 'u8' | 'u16' | 'u32' | 'u64' | 'u128'
                     | 'f32' | 'f64'
                     | 'bool' | 'char' | 'str'

<composite_type>    ::= '[' <type> ']'
                     | '[' <type> ';' <expression> ']'
                     | 'List' '<' <type> '>'
                     | 'Array' '<' <type> ',' <expression> '>'
                     | 'Map' '<' <type> ',' <type> '>'
                     | 'Set' '<' <type> '>'
                     | '(' <type_list> ')'

<reference_type>    ::= '&' ['mut'] <type>
                     | '*' ['const' | 'mut'] <type>

<function_type>     ::= 'fn' '(' [<type_list>] ')' ['->' <type>]

<generic_type>      ::= <identifier> '<' <type_args> '>'
<type_args>         ::= <type> {',' <type>}
<type_list>         ::= <type> {',' <type>}

<ai_type>           ::= 'Tensor' '<' <type> ',' <shape_spec> '>'
                     | 'Model' '<' <input_spec> ',' <output_spec> '>'
                     | 'Dataset' '<' <type> '>'
                     | 'Graph' '<' <node_type> ',' <edge_type> '>'

<shape_spec>        ::= '[' <dimension> {',' <dimension>} ']'
<dimension>         ::= <integer_literal> | <identifier> | '?'

<qualified_type>    ::= <type> ['where' <type_constraints>]
<type_constraints>  ::= <type_constraint> {',' <type_constraint>}
<type_constraint>   ::= <identifier> ':' <trait_bounds>
<trait_bounds>      ::= <trait_bound> {'+' <trait_bound>}
<trait_bound>       ::= <identifier> ['<' <type_args> '>']
```

### 2.4 Declarations

```ebnf
<function_decl>     ::= [<attributes>] [<visibility>] ['async'] 'fn' <identifier> 
                        [<generic_params>] '(' [<parameters>] ')' ['->' <type>]
                        [<where_clause>] <block>

<struct_decl>       ::= [<attributes>] [<visibility>] 'struct' <identifier>
                        [<generic_params>] [<where_clause>] <struct_body>

<struct_body>       ::= '{' <field_list> '}'
                     | '(' <type_list> ')' ';'
                     | ';'

<field_list>        ::= <field> {',' <field>} [',']
<field>             ::= [<visibility>] <identifier> ':' <type>

<enum_decl>         ::= [<attributes>] [<visibility>] 'enum' <identifier>
                        [<generic_params>] [<where_clause>] '{' <variant_list> '}'

<variant_list>      ::= <variant> {',' <variant>} [',']
<variant>           ::= <identifier> [<variant_data>]
<variant_data>      ::= '(' <type_list> ')'
                     | '{' <field_list> '}'

<trait_decl>        ::= [<attributes>] [<visibility>] 'trait' <identifier>
                        [<generic_params>] [<where_clause>] '{' {<trait_item>} '}'

<trait_item>        ::= <trait_function>
                     | <trait_type>
                     | <trait_const>

<trait_function>    ::= ['async'] 'fn' <identifier> [<generic_params>]
                        '(' [<parameters>] ')' ['->' <type>] [<where_clause>] [<block>]

<impl_decl>         ::= [<attributes>] 'impl' [<generic_params>] <type>
                        ['for' <type>] [<where_clause>] '{' {<impl_item>} '}'

<impl_item>         ::= <function_decl>
                     | <type_alias>
                     | <const_decl>

<type_alias>        ::= [<visibility>] 'type' <identifier> [<generic_params>] 
                        '=' <type> [<where_clause>] ';'
```

### 2.5 Actor Model

```ebnf
<actor_decl>        ::= [<attributes>] [<visibility>] 'actor' <identifier>
                        [<generic_params>] '{' {<actor_item>} '}'

<actor_item>        ::= <state_decl>
                     | <message_handler>
                     | <function_decl>

<state_decl>        ::= [<visibility>] 'state' <identifier> ':' <type> ['=' <expression>] ';'

<message_handler>   ::= 'receive' <pattern> [<guard>] <block>

<actor_spawn>       ::= 'spawn' <expression> '(' [<expression_list>] ')'

<send_expr>         ::= <expression> '!' <expression>
                     | <expression> '.send' '(' <expression> ')'
```

### 2.6 AI/ML Constructs

```ebnf
<model_decl>        ::= [<attributes>] [<visibility>] 'model' <identifier>
                        [<generic_params>] '{' <model_body> '}'

<model_body>        ::= {<layer_decl>} [<forward_decl>]

<layer_decl>        ::= <identifier> ':' <layer_type> '(' <layer_params> ')' ';'

<layer_type>        ::= 'Linear' | 'Conv2d' | 'LSTM' | 'Transformer' 
                     | 'Attention' | <identifier>

<forward_decl>      ::= 'forward' '(' <parameters> ')' '->' <type> <block>

<pipeline_decl>     ::= [<attributes>] [<visibility>] 'pipeline' <identifier>
                        '{' {<pipeline_stage>} '}'

<pipeline_stage>    ::= <identifier> ':' <stage_type> <stage_config> ';'

<stage_type>        ::= 'load' | 'preprocess' | 'train' | 'validate' 
                     | 'deploy' | <identifier>

<tensor_ops>        ::= <expression> '@' <expression>  // Matrix multiplication
                     | <expression> '.T'               // Transpose
                     | <expression> '.reshape' '(' <shape_spec> ')'
                     | <expression> '.grad' '(' ')'
```

### 2.7 Statements

```ebnf
<statement>         ::= <let_stmt>
                     | <expression_stmt>
                     | <if_stmt>
                     | <while_stmt>
                     | <for_stmt>
                     | <match_stmt>
                     | <return_stmt>
                     | <break_stmt>
                     | <continue_stmt>
                     | <block>
                     | <async_block>
                     | <cloud_stmt>

<let_stmt>          ::= ['immutable'] 'let' <pattern> [':' <type>] '=' <expression> ';'

<expression_stmt>   ::= <expression> ';'

<if_stmt>           ::= 'if' <expression> <block> ['else' (<if_stmt> | <block>)]

<while_stmt>        ::= 'while' <expression> <block>

<for_stmt>          ::= 'for' <pattern> 'in' <expression> <block>

<match_stmt>        ::= 'match' <expression> '{' <match_arm_list> '}'
<match_arm_list>    ::= <match_arm> {',' <match_arm>} [',']
<match_arm>         ::= <pattern> [<guard>] '=>' (<expression> | <block>)
<guard>             ::= 'if' <expression>

<return_stmt>       ::= 'return' [<expression>] ';'
<break_stmt>        ::= 'break' [<label>] [<expression>] ';'
<continue_stmt>     ::= 'continue' [<label>] ';'

<block>             ::= '{' {<statement>} [<expression>] '}'
<async_block>       ::= 'async' <block>
```

### 2.8 Cloud-Native Constructs

```ebnf
<cloud_stmt>        ::= <serverless_decl>
                     | <service_decl>
                     | <deploy_stmt>

<serverless_decl>   ::= '@serverless' ['(' <serverless_config> ')'] <function_decl>

<serverless_config> ::= <config_item> {',' <config_item>}
<config_item>       ::= <identifier> ':' <literal>

<service_decl>      ::= '@service' ['(' <service_config> ')'] <struct_decl>

<deploy_stmt>       ::= 'deploy' <expression> 'to' <cloud_target> 
                        ['with' <deploy_config>] ';'

<cloud_target>      ::= 'aws' | 'gcp' | 'azure' | <string_literal>
```

### 2.9 Expressions

```ebnf
<expression>        ::= <assignment_expr>

<assignment_expr>   ::= <logical_or_expr> [<assignment_op> <assignment_expr>]
<assignment_op>     ::= '=' | '+=' | '-=' | '*=' | '/=' | '%=' | '**='

<logical_or_expr>   ::= <logical_and_expr> {'||' <logical_and_expr>}
<logical_and_expr>  ::= <bitwise_or_expr> {'&&' <bitwise_or_expr>}
<bitwise_or_expr>   ::= <bitwise_xor_expr> {'|' <bitwise_xor_expr>}
<bitwise_xor_expr>  ::= <bitwise_and_expr> {'^' <bitwise_and_expr>}
<bitwise_and_expr>  ::= <equality_expr> {'&' <equality_expr>}

<equality_expr>     ::= <relational_expr> {('==' | '!=') <relational_expr>}
<relational_expr>   ::= <shift_expr> {('<' | '<=' | '>' | '>=') <shift_expr>}
<shift_expr>        ::= <additive_expr> {('<<' | '>>') <additive_expr>}

<additive_expr>     ::= <multiplicative_expr> {('+' | '-') <multiplicative_expr>}
<multiplicative_expr> ::= <power_expr> {('*' | '/' | '%' | '@') <power_expr>}
<power_expr>        ::= <unary_expr> ['**' <power_expr>]

<unary_expr>        ::= <unary_op> <unary_expr>
                     | <postfix_expr>
<unary_op>          ::= '+' | '-' | '!' | '~' | '&' | '*'

<postfix_expr>      ::= <primary_expr> {<postfix_op>}
<postfix_op>        ::= '[' <expression> ']'
                     | '.' <identifier>
                     | '.' <integer_literal>
                     | '(' [<expression_list>] ')'
                     | '?' | '!'

<primary_expr>      ::= <literal>
                     | <identifier>
                     | <self_expr>
                     | <paren_expr>
                     | <tuple_expr>
                     | <array_expr>
                     | <struct_expr>
                     | <closure_expr>
                     | <if_expr>
                     | <match_expr>
                     | <await_expr>
                     | <tensor_expr>

<self_expr>         ::= 'self' | 'Self'
<paren_expr>        ::= '(' <expression> ')'
<tuple_expr>        ::= '(' [<expression_list>] ')'
<array_expr>        ::= '[' [<expression_list>] ']'
                     | '[' <expression> ';' <expression> ']'

<struct_expr>       ::= <identifier> '{' [<field_init_list>] '}'
<field_init_list>   ::= <field_init> {',' <field_init>} [',']
<field_init>        ::= <identifier> [':' <expression>]

<closure_expr>      ::= '|' [<parameters>] '|' ['->' <type>] <expression>
                     | '||' ['->' <type>] <expression>

<if_expr>           ::= 'if' <expression> <block> 'else' <block>
<match_expr>        ::= 'match' <expression> '{' <match_arm_list> '}'

<await_expr>        ::= <expression> '.await'

<expression_list>   ::= <expression> {',' <expression>}
```

### 2.10 Patterns

```ebnf
<pattern>           ::= <literal_pattern>
                     | <identifier_pattern>
                     | <wildcard_pattern>
                     | <tuple_pattern>
                     | <array_pattern>
                     | <struct_pattern>
                     | <enum_pattern>
                     | <range_pattern>
                     | <reference_pattern>

<literal_pattern>   ::= <literal>
<identifier_pattern> ::= ['ref'] ['mut'] <identifier>
<wildcard_pattern>  ::= '_'

<tuple_pattern>     ::= '(' [<pattern_list>] ')'
<array_pattern>     ::= '[' [<pattern_list>] ']'
<pattern_list>      ::= <pattern> {',' <pattern>}

<struct_pattern>    ::= <identifier> '{' [<field_pattern_list>] '}'
<field_pattern_list> ::= <field_pattern> {',' <field_pattern>} [',']
<field_pattern>     ::= <identifier> [':' <pattern>]

<enum_pattern>      ::= <identifier> [<enum_pattern_data>]
<enum_pattern_data> ::= '(' [<pattern_list>] ')'
                     | '{' [<field_pattern_list>] '}'

<range_pattern>     ::= <expression> '..' [<expression>]
                     | <expression> '..=' <expression>

<reference_pattern> ::= '&' ['mut'] <pattern>
```

### 2.11 Attributes

```ebnf
<attributes>        ::= {<attribute>}
<attribute>         ::= '#[' <attr_content> ']'
                     | '#![' <attr_content> ']'

<attr_content>      ::= <identifier> ['(' <attr_args> ')']
<attr_args>         ::= <literal> {',' <literal>}
                     | <identifier> '=' <literal> {',' <identifier> '=' <literal>}
```

### 2.12 Generic Parameters and Where Clauses

```ebnf
<generic_params>    ::= '<' <generic_param_list> '>'
<generic_param_list> ::= <generic_param> {',' <generic_param>}
<generic_param>     ::= <lifetime_param>
                     | <type_param>
                     | <const_param>

<lifetime_param>    ::= "'" <identifier> [':' <lifetime_bounds>]
<type_param>        ::= <identifier> [':' <trait_bounds>] ['=' <type>]
<const_param>       ::= 'const' <identifier> ':' <type>

<where_clause>      ::= 'where' <where_predicates>
<where_predicates>  ::= <where_predicate> {',' <where_predicate>}
<where_predicate>   ::= <type> ':' <trait_bounds>
                     | "'" <identifier> ':' <lifetime_bounds>

<lifetime_bounds>   ::= <lifetime> {'+' <lifetime>}
<lifetime>          ::= "'" <identifier> | "'static"
```

### 2.13 Visibility

```ebnf
<visibility>        ::= 'public'
                     | 'private'
                     | 'protected'
                     | 'internal'
                     | 'public' '(' <visibility_scope> ')'

<visibility_scope>  ::= 'crate' | 'super' | 'module'
```

## 3. Precedence and Associativity

| Precedence | Operator | Associativity |
|------------|----------|---------------|
| 1 (highest)| `()` `[]` `.` | Left |
| 2 | `!` `~` unary`-` unary`+` | Right |
| 3 | `**` | Right |
| 4 | `*` `/` `%` `@` | Left |
| 5 | `+` `-` | Left |
| 6 | `<<` `>>` | Left |
| 7 | `&` | Left |
| 8 | `^` | Left |
| 9 | `|` | Left |
| 10 | `<` `<=` `>` `>=` | Left |
| 11 | `==` `!=` | Left |
| 12 | `&&` | Left |
| 13 | `||` | Left |
| 14 | `..` `..=` | None |
| 15 | `=` `+=` `-=` etc. | Right |
| 16 (lowest) | `return` `break` | None |

## 4. Context-Sensitive Rules

1. **Indentation**: STARK uses significant indentation similar to Python. A new indentation level creates a new block scope.

2. **Type Inference**: Types can be omitted in many contexts where they can be inferred:
   - Local variable declarations
   - Closure parameters
   - Generic type arguments in some contexts

3. **Async Context**: `await` can only be used inside async functions or async blocks.

4. **Memory Safety**: The compiler enforces:
   - Ownership rules (single owner, move semantics)
   - Borrowing rules (multiple immutable OR single mutable reference)
   - Lifetime tracking

5. **Tensor Operations**: The `@` operator is overloaded for matrix multiplication and requires tensor types.

## 5. Examples

### Hello World
```stark
fn main() {
    print("Hello, World!")
}
```

### Async Actor
```stark
actor Counter {
    state count: i32 = 0
    
    receive Increment {
        self.count += 1
    }
    
    receive GetCount -> i32 {
        return self.count
    }
}

async fn main() {
    let counter = spawn Counter()
    counter ! Increment
    let count = (counter ! GetCount).await
    print(f"Count: {count}")
}
```

### ML Pipeline
```stark
pipeline ImageClassifier {
    load: Dataset<Image> from "data/images"
    
    preprocess: {
        resize(224, 224),
        normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    }
    
    model: ResNet(layers=50, num_classes=1000)
    
    train: {
        optimizer: Adam(lr=0.001),
        loss: CrossEntropy(),
        epochs: 100,
        batch_size: 32
    }
    
    validate: accuracy_score
    
    deploy: @serverless(memory=2048, timeout=300)
}
```

This formal grammar provides a complete specification for parsing STARK language source code and can be used as the basis for implementing a parser, syntax highlighter, or language server.