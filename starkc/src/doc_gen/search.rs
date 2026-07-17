//! Search index: a flat JSON array of every documented item, consumed by
//! `search.html`'s client-side fuzzy/substring search (vanilla JS, no
//! external dependencies — see `html.rs`).

use super::extract::DocItem;
use super::html::item_url;
use std::path::Path;

pub struct SearchEntry {
    pub name: String,
    pub kind: &'static str,
    pub module: String,
    pub signature: String,
    pub url: String,
}

pub fn build_index(items: &[DocItem], package_name: &str) -> Vec<SearchEntry> {
    let mut entries = Vec::new();
    for item in items {
        if !item.kind.is_page_level() {
            continue;
        }
        entries.push(SearchEntry {
            name: item.name.clone(),
            kind: item.kind.label(),
            module: if item.module_path.is_empty() {
                package_name.to_string()
            } else {
                item.module_path.clone()
            },
            signature: item.signature.clone(),
            url: item_url(item),
        });
        for member in &item.members {
            entries.push(SearchEntry {
                name: format!("{}::{}", item.name, member.name),
                kind: member.kind.label(),
                module: if item.module_path.is_empty() {
                    package_name.to_string()
                } else {
                    item.module_path.clone()
                },
                signature: member.signature.clone(),
                url: format!("{}#{}", item_url(item), member.name),
            });
        }
    }
    entries
}

pub fn write_index(entries: &[SearchEntry], output_dir: &Path) -> std::io::Result<()> {
    let mut json = String::from("[\n");
    for (i, e) in entries.iter().enumerate() {
        if i > 0 {
            json.push_str(",\n");
        }
        json.push_str(&format!(
            "  {{\"name\":{},\"kind\":{},\"module\":{},\"signature\":{},\"url\":{}}}",
            json_string(&e.name),
            json_string(e.kind),
            json_string(&e.module),
            json_string(&e.signature),
            json_string(&e.url),
        ));
    }
    json.push_str("\n]\n");
    std::fs::write(output_dir.join("search.json"), json)
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
