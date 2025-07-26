# STARK Lexical Grammar Specification

## Overview
This document defines the lexical structure of STARK - how source code is broken down into tokens.

## Token Categories

### 1. Keywords
```
// Control Flow
if, else, elif, match, case, default
for, while, loop, break, continue, return

// Declarations
fn, struct, enum, trait, impl, let, mut, const

// Types
Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
Float32, Float64, Bool, String, Char, Unit

// Memory & Ownership
ref, move, copy, drop

// Visibility
pub, priv

// Operators
and, or, not, in, is, as

// Literals
true, false, null
```

### 2. Identifiers
```
IDENTIFIER := [a-zA-Z_][a-zA-Z0-9_]*
```

Rules:
- Must start with letter or underscore
- Can contain letters, digits, underscores
- Case sensitive
- Cannot be keywords
- Maximum length: 255 characters

### 3. Literals

#### Integer Literals
```
DECIMAL_INT := [0-9][0-9_]*
HEX_INT     := 0[xX][0-9a-fA-F][0-9a-fA-F_]*
BINARY_INT  := 0[bB][01][01_]*
OCTAL_INT   := 0[oO][0-7][0-7_]*

// Type suffixes
INT_SUFFIX := (i8|i16|i32|i64|u8|u16|u32|u64)
```

Examples:
```stark
42
1_000_000
0xFF_FF
0b1010_1010
0o755
42i32
255u8
```

#### Floating Point Literals
```
FLOAT := DECIMAL_INT '.' [0-9][0-9_]* [EXPONENT]?
       | DECIMAL_INT EXPONENT
EXPONENT := [eE][+-]?[0-9][0-9_]*

// Type suffixes
FLOAT_SUFFIX := (f32|f64)
```

Examples:
```stark
3.14
1.0e10
2.5e-3
42.0f32
```

#### String Literals
```
STRING := '"' (CHAR | ESCAPE_SEQUENCE)* '"'
RAW_STRING := 'r"' .*? '"'

ESCAPE_SEQUENCE := '\' (n|t|r|0|\\|'|"|x[0-9a-fA-F]{2}|u{[0-9a-fA-F]{1,6}})
```

Examples:
```stark
"Hello, World!"
"Line 1\nLine 2"
"\x41\x42\x43"
"\u{1F600}"
r"Raw string with \n literal backslashes"
```

#### Character Literals
```
CHAR := '\'' (CHAR_CONTENT | ESCAPE_SEQUENCE) '\''
CHAR_CONTENT := [^'\\]
```

Examples:
```stark
'a'
'\n'
'\x41'
'\u{1F600}'
```

#### Boolean Literals
```
BOOL := true | false
```

### 4. Operators

#### Arithmetic
```
+ - * / % **
```

#### Comparison
```
== != < <= > >=
```

#### Logical
```
&& || !
```

#### Bitwise
```
& | ^ ~ << >>
```

#### Assignment
```
= += -= *= /= %= **= &= |= ^= <<= >>=
```

#### Other
```
? : :: . -> => @ # $
```

### 5. Delimiters
```
( ) [ ] { } , ; :
```

### 6. Comments
```
// Single line comment
/* Multi-line comment */
/** Documentation comment */
```

Rules:
- Single line comments start with `//` and continue to end of line
- Multi-line comments start with `/*` and end with `*/`
- Multi-line comments can be nested
- Documentation comments start with `/**` and are associated with following declaration

### 7. Whitespace
- Space (U+0020)
- Tab (U+0009)
- Newline (U+000A)
- Carriage Return (U+000D)

Newlines are significant for statement termination when not inside delimiters.

### 8. Token Precedence
When multiple token patterns could match:
1. Keywords take precedence over identifiers
2. Longer operators take precedence over shorter ones
3. Comments are ignored in token stream

### 9. Reserved Tokens
Reserved for future use:
```
async, await, yield, where, macro, unsafe, extern
```

## Lexical Analysis Rules

1. **Maximal Munch**: Always match the longest possible token
2. **Whitespace**: Ignored except for statement separation
3. **Line Continuation**: Backslash at end of line continues statement
4. **Encoding**: Source files must be valid UTF-8

## Error Handling

Invalid tokens should produce specific error messages:
- Invalid character: "Unexpected character 'X' at line Y:Z"
- Unterminated string: "Unterminated string literal at line Y:Z"
- Invalid number: "Invalid number format at line Y:Z"