# STARKLANG Syntax Highlighting Specification for LSP/IDE Integration

This document defines the recommended syntax highlighting categories, token types, scopes, and language grammar rules for integrating STARKLANG into Language Server Protocol (LSP) environments and IDEs (e.g., VSCode, JetBrains, Neovim).

---

## 🎯 Objective
Provide a precise mapping of STARKLANG language constructs to syntax tokens for:
- Syntax coloring
- Semantic highlighting
- Token-based linting support

---

## 📚 Highlighting Token Categories

| Token Category        | Examples | LSP Semantic Token Type | Scope Suggestion         |
|-----------------------|---------|--------------------------|---------------------------|
| Keywords              | `fn`, `let`, `return`, `export`, `actor`, `match`, `trait`, `impl`, `async`, `await`, `spawn`, `immutable` | `keyword` | `keyword.control.stark` |
| Control Flow Keywords | `if`, `else`, `while`, `for`, `break`, `continue` | `keyword.control` | `keyword.control.flow.stark` |
| Data Types            | `Int32`, `Float32`, `Bool`, `String`, `Tensor`, `Dataset`, `Result`, `Option` | `type` | `storage.type.stark` |
| Function Names        | `calculate_tax`, `predict`, `main` | `function` | `entity.name.function.stark` |
| Function Calls        | `predict(...)`, `normalize(...)` | `function` | `meta.function-call.stark` |
| Struct/Enum Names     | `User`, `Transaction`, `StatusCode` | `type` | `entity.name.struct.stark` |
| Trait Names           | `Displayable`, `Predictable` | `interface` | `entity.name.trait.stark` |
| Variable Identifiers  | `x`, `data`, `model` | `variable` | `variable.other.stark` |
| Constants             | `PI`, `MAX_SIZE` | `variable.readonly` | `constant.other.stark` |
| Literals              | `"text"`, `42`, `true`, `false`, `null` | `string`, `number`, `boolean`, `null` | `constant.language.stark` |
| Comments              | `// inline comment`, `/* block comment */` | `comment` | `comment.line.stark` / `comment.block.stark` |
| Annotations           | `@serverless`, `@server` | `decorator` | `meta.annotation.stark` |
| Operators             | `+`, `-`, `==`, `<`, `>=`, `and`, `or`, `not` | `operator` | `keyword.operator.stark` |
| Module Imports        | `import utils.math` | `namespace` | `keyword.import.stark` |
| Attribute Access      | `object.property` | `property` | `variable.property.stark` |

---

## 📐 Semantic Highlighting Extension (LSP)

Use `semanticTokensProvider` in the STARK LSP:
```json
"semanticTokenTypes": [
  "keyword", "variable", "function", "type", "parameter", "property", "comment",
  "string", "number", "operator", "interface", "namespace", "decorator"
],
"semanticTokenModifiers": ["readonly", "static", "abstract"]
```

---

## 🧠 Special Highlighting Rules
- Differentiate trait names from type names using `interface`
- Highlight `fn` declarations differently from `fn` calls
- Highlight top-level module identifiers (imported namespaces) distinctly
- Emphasize ML pipeline keywords like `load data`, `train using`, `validate with`

---

## 🖍 VSCode Grammar JSON Snippet (Example)
```json
{
  "name": "Keyword",
  "match": "\\b(fn|let|return|export|actor|trait|spawn|immutable|async|await)\\b",
  "scope": "keyword.control.stark"
},
{
  "name": "Comment",
  "match": "//.*$",
  "scope": "comment.line.double-slash.stark"
},
{
  "name": "String",
  "begin": "\"",
  "end": "\"",
  "scope": "string.quoted.double.stark"
}
```

---

## 🔮 Future Enhancements
- Themed highlighting for ML pipeline stages
- Highlight actor lifecycle blocks (`start`, `listen`) with custom scopes
- Semantic token modifiers for async/await patterns

---

## ✅ Summary
This spec defines the visual language for STARKLANG in modern IDEs and editor integrations. These token categories and scopes provide strong semantic clarity and will improve developer productivity and syntax tooling capabilities.

