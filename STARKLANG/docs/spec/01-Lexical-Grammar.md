# STARK Lexical Grammar Specification

## Overview
This document defines the lexical structure of STARK - how source code is broken down into tokens.

## Token Categories

### 1. Keywords
```
// Control Flow
if, else, match
for, while, loop, break, continue, return

// Declarations
fn, struct, enum, trait, impl, let, mut, const, type, use, mod

// Types
Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
Float32, Float64, Bool, String, Char, Unit, str

// Visibility
pub, priv

// Module Paths and Self Type
self, Self, super, crate

// Operators
in, as

// Literals
true, false
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

**LEX-IDENT-002.** Identifier length is measured in ASCII bytes. Core
identifiers are ASCII-only and may contain at most 255 bytes. The lexer must
reject an overlength identifier as one token and continue at its end; it may
not truncate, split, or intern a colliding prefix.

### 3. Literals

#### Integer Literals
```
DECIMAL_INT := [0-9] ('_'? [0-9])*
HEX_INT     := 0[xX] [0-9a-fA-F] ('_'? [0-9a-fA-F])*
BINARY_INT  := 0[bB] [01] ('_'? [01])*
OCTAL_INT   := 0[oO] [0-7] ('_'? [0-7])*

INT_SUFFIX := (i8|i16|i32|i64|u8|u16|u32|u64)

INTEGER := (DECIMAL_INT | HEX_INT | BINARY_INT | OCTAL_INT) INT_SUFFIX?
```

Rules:
- Underscores are digit separators and may appear only *between* two digits —
  never leading, trailing, or consecutive (`1__2` and `12_` are invalid).
- A suffix fixes the literal's type: `i8`→`Int8`, `i16`→`Int16`, `i32`→`Int32`,
  `i64`→`Int64`, `u8`→`UInt8`, `u16`→`UInt16`, `u32`→`UInt32`, `u64`→`UInt64`.
  A suffixed literal whose value does not fit the named type is a compile-time
  error.

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
FLOAT_BODY := DECIMAL_INT '.' [0-9] ('_'? [0-9])* EXPONENT?
            | DECIMAL_INT EXPONENT
EXPONENT := [eE] [+-]? [0-9] ('_'? [0-9])*

FLOAT_SUFFIX := (f32|f64)

FLOAT := FLOAT_BODY FLOAT_SUFFIX?
```

Rules:
- The underscore rules for integer literals apply (separators between digits only).
- A suffix fixes the literal's type: `f32`→`Float32`, `f64`→`Float64`.

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

**LEX-ESCAPE-001.** `\xNN` contributes exactly one byte and is legal in a
string only when the complete decoded string is valid UTF-8; in a character
literal it must denote one ASCII scalar. `\u{H...}` must contain one through
six hexadecimal digits and denote a Unicode scalar value, excluding surrogate
code points and values above U+10FFFF. An invalid escape rejects the literal;
no replacement character is inserted.

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

#### Range
```
.. ..=
```

#### Other
```
? :: . -> =>
```

Notes:
- `?` is the try operator (postfix). There is no ternary conditional operator; `if` is an expression.
- `:` appears only as a delimiter (type annotations, struct fields), not as an operator.

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

Whitespace separates tokens and is otherwise ignored. Statement termination is defined by semicolons in the syntax grammar.

### 8. Token Precedence
When multiple token patterns could match:
1. Keywords take precedence over identifiers
2. Longer operators take precedence over shorter ones
3. Comments are ignored in token stream

### 9. Reserved Tokens
Reserved for future use (recognized as keywords but not used by any Core v1 grammar production):
```
async, await, yield, where, macro, unsafe, extern, import, export, null,
and, or, not, is, dyn
```

**LEX-RESERVED-001.** Every spelling in this list tokenizes as a reserved
keyword and is rejected wherever Core requires an identifier. An
implementation may not treat an unused reserved word as an identifier or
silently enable its future meaning.

## Lexical Analysis Rules

1. **Maximal Munch**: Always match the longest possible token
2. **Whitespace**: Ignored except for token separation
3. **Encoding**: Source files must be valid UTF-8

**LEX-SOURCE-001.** A Core source is a sequence of UTF-8 bytes. A byte-order
mark at the start is not source whitespace and is rejected. Any invalid UTF-8
sequence rejects the source before tokenization at the first invalid byte
offset. Tools may transcode input before presenting it as Core source, but
that transport conversion is outside the language and cannot affect source
identity or diagnostics for the resulting bytes.

## Error Handling

Invalid tokens should produce specific error messages:
- Invalid character: "Unexpected character 'X' at line Y:Z"
- Unterminated string: "Unterminated string literal at line Y:Z"
- Invalid number: "Invalid number format at line Y:Z"
## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
