/**
 * A small STARK syntax highlighter.
 *
 * Written rather than imported because no highlighting library knows STARK. The token classes
 * below are taken from the compiler's own lexer keyword table and type model, not invented for
 * the website — if the language gains a keyword, this list is where it is added, and being wrong
 * here shows up as unhighlighted text rather than as a broken page.
 *
 * Deliberately small: it colours a fixed set of code samples on a marketing page. It is not a
 * parser, makes no claim to handle every construct, and never runs on user input.
 */

export type TokenKind =
  | 'keyword'
  | 'type'
  | 'string'
  | 'comment'
  | 'number'
  | 'function'
  | 'punct'
  | 'plain';

export interface Token {
  kind: TokenKind;
  text: string;
}

/** Reserved words, from `starkc/src/lexer.rs`. Includes those reserved-but-unused in Core v1. */
const KEYWORDS = new Set([
  'and', 'as', 'async', 'await', 'break', 'const', 'continue', 'crate', 'dyn', 'else', 'enum',
  'export', 'extern', 'false', 'fn', 'for', 'if', 'impl', 'import', 'in', 'is', 'let', 'loop',
  'macro', 'match', 'mod', 'mut', 'not', 'null', 'or', 'priv', 'pub', 'return', 'self', 'struct',
  'super', 'trait', 'true', 'type', 'unsafe', 'use', 'where', 'while', 'yield', 'model',
]);

/**
 * Primitives are PascalCase in STARK — `Int32`, never `i32`. Listed explicitly rather than matched
 * by "starts with a capital", so a user type reads as a type and a primitive reads as a primitive.
 */
const PRIMITIVES = new Set([
  'Int8', 'Int16', 'Int32', 'Int64',
  'UInt8', 'UInt16', 'UInt32', 'UInt64',
  'Float32', 'Float64',
  'Bool', 'Char', 'String', 'str', 'Unit',
  'Option', 'Result', 'Vec', 'Box', 'HashMap', 'HashSet', 'Ordering',
  'Tensor', 'TensorAny', 'Dim',
]);

const IDENT = /^[A-Za-z_][A-Za-z0-9_]*/;
const NUMBER = /^\d[\d_]*(\.\d[\d_]*)?([eE][+-]?\d+)?(i8|i16|i32|i64|u8|u16|u32|u64|f32|f64)?/;

/** Splits `source` into classified tokens. Unrecognised text is returned as `plain`. */
export function tokenize(source: string): Token[] {
  const out: Token[] = [];
  let i = 0;

  const push = (kind: TokenKind, text: string) => {
    // Merge runs of the same kind so the DOM stays small.
    const last = out[out.length - 1];
    if (last && last.kind === kind) last.text += text;
    else out.push({ kind, text });
  };

  while (i < source.length) {
    const rest = source.slice(i);

    // Line comment.
    if (rest.startsWith('//')) {
      const end = rest.indexOf('\n');
      const text = end === -1 ? rest : rest.slice(0, end);
      push('comment', text);
      i += text.length;
      continue;
    }

    // String and char literals, with escapes.
    if (rest[0] === '"' || rest[0] === "'") {
      const quote = rest[0];
      let j = 1;
      while (j < rest.length && rest[j] !== quote) {
        if (rest[j] === '\\') j++;
        j++;
      }
      const text = rest.slice(0, Math.min(j + 1, rest.length));
      push('string', text);
      i += text.length;
      continue;
    }

    const numberMatch = NUMBER.exec(rest);
    if (numberMatch && /\d/.test(rest[0])) {
      push('number', numberMatch[0]);
      i += numberMatch[0].length;
      continue;
    }

    const identMatch = IDENT.exec(rest);
    if (identMatch) {
      const word = identMatch[0];
      const after = rest.slice(word.length);
      // A name followed by `(` or `::<` reads as a call; this is a visual cue, not resolution.
      const isCall = /^\s*(\(|::<)/.test(after);

      if (KEYWORDS.has(word)) push('keyword', word);
      else if (PRIMITIVES.has(word)) push('type', word);
      else if (/^[A-Z]/.test(word)) push('type', word);
      else if (isCall) push('function', word);
      else push('plain', word);

      i += word.length;
      continue;
    }

    if ('{}()[]<>:;,.=+-*/%&|!?'.includes(rest[0])) {
      push('punct', rest[0]);
      i += 1;
      continue;
    }

    push('plain', rest[0]);
    i += 1;
  }

  return out;
}
