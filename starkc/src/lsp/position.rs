//! LSP position conversion.
//!
//! Compiler spans are UTF-8 byte offsets. LSP positions are zero-based lines
//! and UTF-16 code-unit character offsets.

use crate::source::{SourceFile, Span};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LspPosition {
    pub line: u32,
    pub character: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LspRange {
    pub start: LspPosition,
    pub end: LspPosition,
}

pub fn span_to_lsp_range(file: &SourceFile, span: Span) -> LspRange {
    LspRange {
        start: byte_offset_to_lsp_position(&file.src, span.lo),
        end: byte_offset_to_lsp_position(&file.src, span.hi),
    }
}

pub fn byte_offset_to_lsp_position(source: &str, offset: u32) -> LspPosition {
    let target = clamp_to_char_boundary(source, offset as usize);
    let mut line = 0u32;
    let mut line_start = 0usize;
    for (index, byte) in source.bytes().enumerate() {
        if index >= target {
            break;
        }
        if byte == b'\n' {
            line += 1;
            line_start = index + 1;
        }
    }
    LspPosition {
        line,
        character: utf16_units(&source[line_start..target]),
    }
}

pub fn lsp_position_to_byte_offset(source: &str, line: u32, character: u32) -> Option<u32> {
    let line_start = line_start(source, line)?;
    let line_end = source[line_start..]
        .find('\n')
        .map_or(source.len(), |index| line_start + index);
    let line_text = &source[line_start..line_end];
    let mut utf16 = 0u32;
    for (relative, ch) in line_text.char_indices() {
        if utf16 == character {
            return u32::try_from(line_start + relative).ok();
        }
        let width = ch.len_utf16() as u32;
        if utf16 + width > character {
            return None;
        }
        utf16 += width;
    }
    (utf16 == character)
        .then_some(line_end)
        .and_then(|offset| u32::try_from(offset).ok())
}

fn line_start(source: &str, line: u32) -> Option<usize> {
    if line == 0 {
        return Some(0);
    }
    let mut current = 0u32;
    for (index, byte) in source.bytes().enumerate() {
        if byte == b'\n' {
            current += 1;
            if current == line {
                return Some(index + 1);
            }
        }
    }
    None
}

fn clamp_to_char_boundary(source: &str, mut offset: usize) -> usize {
    offset = offset.min(source.len());
    while !source.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

fn utf16_units(text: &str) -> u32 {
    text.encode_utf16().count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_positions_round_trip() {
        let source = "let x = 1;\nlet y = 2;";
        assert_eq!(
            byte_offset_to_lsp_position(source, 15),
            LspPosition {
                line: 1,
                character: 4
            }
        );
        assert_eq!(lsp_position_to_byte_offset(source, 1, 4), Some(15));
    }

    #[test]
    fn devanagari_and_punjabi_comments_use_utf16_columns() {
        let source = "// नमस्ते\n// ਸਤਿ\nlet x = 1;";
        let x = source.find('x').unwrap();
        assert_eq!(
            byte_offset_to_lsp_position(source, x as u32),
            LspPosition {
                line: 2,
                character: 4
            }
        );
        assert_eq!(lsp_position_to_byte_offset(source, 2, 4), Some(x as u32));
    }

    #[test]
    fn emoji_counts_as_two_utf16_units() {
        let source = "// 😀a\nlet x = 1;";
        let after_emoji = source.find('a').unwrap();
        assert_eq!(
            byte_offset_to_lsp_position(source, after_emoji as u32),
            LspPosition {
                line: 0,
                character: 5
            }
        );
        assert_eq!(
            lsp_position_to_byte_offset(source, 0, 4),
            None,
            "positions inside a surrogate pair are invalid"
        );
        assert_eq!(
            lsp_position_to_byte_offset(source, 0, 5),
            Some(after_emoji as u32)
        );
    }

    #[test]
    fn combining_marks_count_as_separate_code_units() {
        let source = "// e\u{301}x\n";
        let x = source.find('x').unwrap();
        assert_eq!(
            byte_offset_to_lsp_position(source, x as u32),
            LspPosition {
                line: 0,
                character: 5
            }
        );
        assert_eq!(lsp_position_to_byte_offset(source, 0, 5), Some(x as u32));
    }

    #[test]
    fn rejects_positions_outside_document() {
        assert_eq!(lsp_position_to_byte_offset("one\n", 3, 0), None);
        assert_eq!(lsp_position_to_byte_offset("one", 0, 4), None);
    }
}
