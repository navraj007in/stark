//! Minimal Markdown-to-HTML renderer for doc comments.
//!
//! Deliberately not a full CommonMark implementation (no external
//! dependencies, per this project's convention) — just the subset doc
//! comments actually use per the plan's own example (`# Heading`
//! sections, paragraphs, inline `` `code` ``, fenced ` ```lang ` blocks,
//! `*`/`-` bullet lists). Anything outside that subset renders as a plain
//! escaped paragraph rather than being misinterpreted.

use super::highlight;

#[allow(unused_assignments)] // `list_open = false`'s last write (end of function) is intentionally unread
pub fn render(doc: &str) -> String {
    let mut out = String::new();
    let mut lines = doc.lines().peekable();
    let mut paragraph: Vec<&str> = Vec::new();
    let mut list_open = false;

    macro_rules! flush_paragraph {
        () => {
            if !paragraph.is_empty() {
                out.push_str("<p>");
                out.push_str(&render_inline(&paragraph.join(" ")));
                out.push_str("</p>\n");
                paragraph.clear();
            }
        };
    }
    macro_rules! close_list {
        () => {
            if list_open {
                out.push_str("</ul>\n");
                list_open = false;
            }
        };
    }

    while let Some(line) = lines.next() {
        let trimmed = line.trim_end();

        if trimmed.trim_start().starts_with("```") {
            flush_paragraph!();
            close_list!();
            let lang = trimmed
                .trim_start()
                .trim_start_matches("```")
                .trim()
                .to_string();
            let mut code = String::new();
            for inner in lines.by_ref() {
                if inner.trim_start().starts_with("```") {
                    break;
                }
                code.push_str(inner);
                code.push('\n');
            }
            out.push_str("<pre class=\"code-block\"><code>");
            if lang == "stark" {
                out.push_str(&highlight::highlight(&code));
            } else {
                out.push_str(&highlight::escape(&code));
            }
            out.push_str("</code></pre>\n");
            continue;
        }

        if let Some(rest) = heading_level(trimmed) {
            flush_paragraph!();
            close_list!();
            let (level, text) = rest;
            out.push_str(&format!("<h{level}>{}</h{level}>\n", render_inline(text)));
            continue;
        }

        let bullet = trimmed.trim_start();
        if bullet.starts_with("* ") || bullet.starts_with("- ") {
            flush_paragraph!();
            if !list_open {
                out.push_str("<ul>\n");
                list_open = true;
            }
            out.push_str("<li>");
            out.push_str(&render_inline(&bullet[2..]));
            out.push_str("</li>\n");
            continue;
        }

        if trimmed.trim().is_empty() {
            flush_paragraph!();
            close_list!();
            continue;
        }

        close_list!();
        paragraph.push(trimmed);
    }
    flush_paragraph!();
    close_list!();
    out
}

fn heading_level(line: &str) -> Option<(usize, &str)> {
    let hashes = line.chars().take_while(|&c| c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = line[hashes..].trim_start();
    if rest.is_empty() || line.as_bytes().get(hashes) != Some(&b' ') {
        return None;
    }
    Some((hashes, rest))
}

/// Inline spans: `` `code` `` only — the doc-comment subset this needs
/// doesn't use bold/italic/links, and guessing at them risks mangling
/// legitimate `*`/`_` characters in prose.
fn render_inline(text: &str) -> String {
    let mut out = String::new();
    let mut in_code = false;
    let mut buf = String::new();
    for ch in text.chars() {
        if ch == '`' {
            if in_code {
                out.push_str("<code>");
                out.push_str(&highlight::escape(&buf));
                out.push_str("</code>");
            } else {
                out.push_str(&highlight::escape(&buf));
            }
            buf.clear();
            in_code = !in_code;
        } else {
            buf.push(ch);
        }
    }
    if in_code {
        // Unterminated backtick: treat the buffered text as plain, not code.
        out.push('`');
        out.push_str(&highlight::escape(&buf));
    } else {
        out.push_str(&highlight::escape(&buf));
    }
    out
}
