# ✍️ STARK Language — Basic Syntax & Grammar Rules

This document outlines the fundamental syntax and grammar of STARK. Designed for clarity, power, and elegance, STARK blends modern readability with high-performance semantics—giving developers a language that feels intuitive, yet razor-sharp.

---

## 📌 General Syntax Philosophy

- Minimal boilerplate, maximal expressiveness.
- Indentation-aware (like Python) — but not whitespace-prison.
- Consistent, human-readable structures.
- Optional trailing semicolons (`;`) — stylistic, not mandatory.

---

## 🔤 Variable Declarations

```stark
let age = 42           // Immutable by default
mut let counter = 0    // Mutable declaration


Type inference is automatic.
Explicit typing also allowed:

let name: String = "STARK"
🧠 Functions

fn greet(name: String) -> String:
    return "Hello, " + name
Multiple return types via Tuple:

fn stats() -> Tuple<Int32, Float32>:
    return (42, 99.8)
Generic Functions:

fn identity<T>(value: T) -> T:
    return value
🔁 Control Flow
If / Else

if score > 90:
    print("Excellent")
else:
    print("Needs work")
Match Expression

match user:
    Some(u) => print(u.name)
    None => print("Guest")
While Loop

while counter < 10:
    print(counter)
    counter += 1
For Loop

for item in dataset:
    print(item)
🧩 Pattern Matching
Pattern matching is native and expressive:

stark
Copy
match result:
    Ok(data) => handle(data)
    Err(e) => log("Error: " + e)
Destructuring tuples:

stark
Copy
let (a, b) = getPair()
📦 Modules & Imports
Modules = reusable code units, importable via keywords.


import utils/math
from utils/io import read_csv
🔐 Constants

const PI: Float32 = 3.14159
❗ Error Handling
Via Result and pattern matching:


fn divide(a: Int, b: Int) -> Result<Int, String>:
    if b == 0:
        return Err("Divide by zero")
    return Ok(a / b)
💬 Comments
Single line: // this is a comment
Multi-line:

/*
   This is a
   multi-line comment
*/
🧠 Syntax Highlights Summary
Construct	Syntax Style
Variable	let / mut let
Function	fn
Type Annotation	: (after variable)
Pattern Match	match block
Looping	for, while
Constants	const
Import Modules	import / from ... import
Error Handling	Result<T, E> + match
STARK syntax is built to be elegant, readable, and powerful. You write logic—not rituals.

