//! Static HTML site generation: one directory+`index.html` per page-level
//! item (matching the plan's `std/option/index.html` layout), a package
//! `index.html`, and a `search.html` search UI over `search.json`
//! (`search.rs`).

use super::extract::{DocItem, ItemDocKind};
use super::highlight;
use super::markdown;
use std::path::Path;

/// Site-relative URL for `item`'s page (from the output root).
pub fn item_url(item: &DocItem) -> String {
    let mut parts: Vec<&str> = item
        .module_path
        .split("::")
        .filter(|s| !s.is_empty())
        .collect();
    parts.push(&item.name);
    format!("{}/index.html", parts.join("/"))
}

pub fn write_site(items: &[DocItem], package_name: &str, output_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    std::fs::write(output_dir.join("style.css"), STYLE_CSS)?;

    write_index_page(items, package_name, output_dir)?;
    write_search_page(package_name, output_dir)?;

    for item in items {
        if !item.kind.is_page_level() {
            continue;
        }
        let mut parts: Vec<&str> = item
            .module_path
            .split("::")
            .filter(|s| !s.is_empty())
            .collect();
        parts.push(&item.name);
        let dir = parts
            .iter()
            .fold(output_dir.to_path_buf(), |p, s| p.join(s));
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("index.html"), render_item_page(item, package_name))?;
    }

    Ok(())
}

fn page(title: &str, root_rel: &str, body: &str) -> String {
    format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>{title}</title>\n\
         <link rel=\"stylesheet\" href=\"{root_rel}style.css\">\n</head>\n<body>\n\
         <header class=\"site-header\"><a href=\"{root_rel}index.html\">Docs</a> \
         <a href=\"{root_rel}search.html\">Search</a></header>\n\
         <main>\n{body}\n</main>\n</body>\n</html>\n",
        title = highlight::escape(title),
    )
}

fn write_index_page(
    items: &[DocItem],
    package_name: &str,
    output_dir: &Path,
) -> std::io::Result<()> {
    let mut by_module: std::collections::BTreeMap<String, Vec<&DocItem>> =
        std::collections::BTreeMap::new();
    for item in items {
        if !item.kind.is_page_level() {
            continue;
        }
        by_module
            .entry(item.module_path.clone())
            .or_default()
            .push(item);
    }

    let mut body = format!("<h1>{}</h1>\n", highlight::escape(package_name));
    for (module, mut mod_items) in by_module {
        mod_items.sort_by(|a, b| a.name.cmp(&b.name));
        let heading = if module.is_empty() {
            "Top level".to_string()
        } else {
            module.clone()
        };
        body.push_str(&format!(
            "<h2>{}</h2>\n<ul class=\"item-list\">\n",
            highlight::escape(&heading)
        ));
        for item in mod_items {
            body.push_str(&format!(
                "<li><span class=\"badge badge-{kind}\">{kind}</span> <a href=\"{url}\">{name}</a></li>\n",
                kind = item.kind.label(),
                url = item_url(item),
                name = highlight::escape(&item.name),
            ));
        }
        body.push_str("</ul>\n");
    }

    std::fs::write(output_dir.join("index.html"), page(package_name, "", &body))
}

fn write_search_page(package_name: &str, output_dir: &Path) -> std::io::Result<()> {
    let body = format!(
        "<h1>Search {}</h1>\n\
         <input id=\"q\" type=\"search\" placeholder=\"Search by name…\" autofocus>\n\
         <ul id=\"results\" class=\"item-list\"></ul>\n\
         <script>{}</script>\n",
        highlight::escape(package_name),
        SEARCH_JS
    );
    std::fs::write(output_dir.join("search.html"), page("Search", "", &body))
}

fn render_item_page(item: &DocItem, package_name: &str) -> String {
    // Every page-level item lives at depth (module segments + 1); the
    // root-relative prefix walks back up that many directories.
    let depth = item
        .module_path
        .split("::")
        .filter(|s| !s.is_empty())
        .count()
        + 1;
    let root_rel = "../".repeat(depth);

    let mut body = format!(
        "<div class=\"item-header\"><span class=\"badge badge-{kind}\">{kind}</span> <h1>{name}</h1></div>\n",
        kind = item.kind.label(),
        name = highlight::escape(&item.name),
    );
    body.push_str(&format!(
        "<pre class=\"signature\"><code>{}</code></pre>\n",
        highlight::highlight(&item.signature)
    ));
    if !item.doc.trim().is_empty() {
        body.push_str(&markdown::render(&item.doc));
    }

    if !item.members.is_empty() {
        let heading = match item.kind {
            ItemDocKind::Struct => "Fields",
            ItemDocKind::Enum => "Variants",
            ItemDocKind::Trait => "Required methods",
            ItemDocKind::Model => "Ports",
            _ => "Members",
        };
        body.push_str(&format!("<h2>{heading}</h2>\n"));
        for member in &item.members {
            body.push_str(&format!(
                "<section class=\"member\" id=\"{anchor}\">\n\
                 <h3><span class=\"badge badge-{kind}\">{kind}</span> {name}</h3>\n\
                 <pre class=\"signature\"><code>{sig}</code></pre>\n",
                anchor = highlight::escape(&member.name),
                kind = member.kind.label(),
                name = highlight::escape(&member.name),
                sig = highlight::highlight(&member.signature),
            ));
            if !member.doc.trim().is_empty() {
                body.push_str(&markdown::render(&member.doc));
            }
            body.push_str("</section>\n");
        }
    }

    // Structs/enums also collect pub `impl` methods as members even
    // though the plan's own heading for those (`is_page_level` methods
    // merged from `impl` blocks) reads better as "Methods" than the
    // struct-field heading above when both are present; keep it simple —
    // real Core v1 struct+impl docs will read fine with one shared
    // "Members" section per WP8.5's initial scope.

    page(&format!("{} — {package_name}", item.name), &root_rel, &body)
}

const STYLE_CSS: &str = r#"
:root {
  color-scheme: light dark;
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #6b7280;
  --border: #e5e7eb;
  --code-bg: #f6f8fa;
  --link: #2563eb;
  --kw: #a626a4;
  --type: #c18401;
  --str: #50a14f;
  --num: #986801;
  --comment: #a0a1a7;
  --ident: #383a42;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117;
    --fg: #e6edf3;
    --muted: #8b949e;
    --border: #30363d;
    --code-bg: #161b22;
    --link: #58a6ff;
    --kw: #ff7b72;
    --type: #ffa657;
    --str: #a5d6ff;
    --num: #79c0ff;
    --comment: #8b949e;
    --ident: #e6edf3;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.6;
}
main { max-width: 860px; margin: 0 auto; padding: 1rem 1.5rem 4rem; }
.site-header {
  padding: 0.75rem 1.5rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  gap: 1rem;
}
.site-header a { color: var(--fg); text-decoration: none; font-weight: 600; }
.site-header a:hover { color: var(--link); }
a { color: var(--link); }
h1, h2, h3 { line-height: 1.25; }
h2 { margin-top: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 0.25rem; }
.item-header { display: flex; align-items: center; gap: 0.75rem; }
.item-header h1 { margin: 0; }
.item-list { list-style: none; padding: 0; }
.item-list li { padding: 0.25rem 0; }
.badge {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  padding: 0.1rem 0.45rem;
  border-radius: 0.3rem;
  background: var(--code-bg);
  color: var(--muted);
  border: 1px solid var(--border);
}
pre, code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
pre.signature, pre.code-block {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 0.4rem;
  padding: 0.75rem 1rem;
  overflow-x: auto;
}
p code { background: var(--code-bg); padding: 0.1rem 0.3rem; border-radius: 0.25rem; }
.member { border-top: 1px solid var(--border); padding-top: 1rem; margin-top: 1rem; }
.tok-kw { color: var(--kw); }
.tok-type { color: var(--type); }
.tok-str { color: var(--str); }
.tok-num { color: var(--num); }
.tok-comment { color: var(--comment); font-style: italic; }
.tok-ident { color: var(--ident); }
.tok-error { color: #e5534b; text-decoration: underline wavy; }
#q {
  width: 100%;
  font-size: 1.1rem;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--border);
  border-radius: 0.4rem;
  background: var(--bg);
  color: var(--fg);
}
"#;

const SEARCH_JS: &str = r#"
(function () {
  var q = document.getElementById('q');
  var results = document.getElementById('results');
  var items = [];
  fetch('search.json').then(function (r) { return r.json(); }).then(function (data) {
    items = data;
    render(q.value);
  });
  function score(name, query) {
    name = name.toLowerCase();
    query = query.toLowerCase();
    var idx = name.indexOf(query);
    if (idx === -1) return -1;
    return idx === 0 ? 0 : 1;
  }
  function render(query) {
    results.innerHTML = '';
    if (!query) return;
    var matches = items
      .map(function (it) { return { it: it, s: score(it.name, query) }; })
      .filter(function (m) { return m.s !== -1; })
      .sort(function (a, b) { return a.s - b.s || a.it.name.length - b.it.name.length; })
      .slice(0, 200);
    matches.forEach(function (m) {
      var li = document.createElement('li');
      var badge = document.createElement('span');
      badge.className = 'badge badge-' + m.it.kind;
      badge.textContent = m.it.kind;
      var a = document.createElement('a');
      a.href = m.it.url;
      a.textContent = m.it.name;
      var mod = document.createElement('span');
      mod.style.color = 'var(--muted)';
      mod.textContent = ' — ' + m.it.module;
      li.appendChild(badge);
      li.appendChild(document.createTextNode(' '));
      li.appendChild(a);
      li.appendChild(mod);
      results.appendChild(li);
    });
  }
  q.addEventListener('input', function () { render(q.value); });
})();
"#;
