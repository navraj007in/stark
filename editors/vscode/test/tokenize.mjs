// Tokenize STARK fixtures with the real VS Code TextMate engine and compare
// the scope sequences against committed snapshots (VSCODE_EXTENSION_PLAN.md
// §4.4). Run `node test/tokenize.mjs` to verify, or with `--update` to
// regenerate snapshots after an intentional grammar change.
//
// Requires devDependencies `vscode-textmate` and `vscode-oniguruma`
// (`npm install`). node_modules is gitignored; the snapshots are the golden
// artifacts under version control.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import oniguruma from "vscode-oniguruma";
import vsctm from "vscode-textmate";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "..");
const grammarPath = path.join(root, "syntaxes", "stark.tmLanguage.json");
const fixturesDir = path.join(here, "fixtures");
const snapshotsDir = path.join(here, "snapshots");
const update = process.argv.includes("--update");

const wasm = fs.readFileSync(
  path.join(path.dirname(require.resolve("vscode-oniguruma")), "onig.wasm")
).buffer;

const onig = oniguruma.loadWASM(wasm).then(() => ({
  createOnigScanner: (p) => new oniguruma.OnigScanner(p),
  createOnigString: (s) => new oniguruma.OnigString(s),
}));

const registry = new vsctm.Registry({
  onigLib: onig,
  loadGrammar: () =>
    Promise.resolve(
      vsctm.parseRawGrammar(fs.readFileSync(grammarPath, "utf8"), grammarPath)
    ),
});

function snapshotFor(grammar, src) {
  let stack = vsctm.INITIAL;
  const out = [];
  for (const line of src.split("\n")) {
    const r = grammar.tokenizeLine(line, stack);
    for (const t of r.tokens) {
      const text = line.slice(t.startIndex, t.endIndex);
      if (!text.trim()) continue;
      out.push(`${JSON.stringify(text)} : ${t.scopes.join(" ")}`);
    }
    stack = r.ruleStack;
  }
  return out.join("\n") + "\n";
}

const grammar = await registry.loadGrammar("source.stark");
fs.mkdirSync(snapshotsDir, { recursive: true });

let failed = 0;
for (const file of fs.readdirSync(fixturesDir).filter((f) => f.endsWith(".stark"))) {
  const src = fs.readFileSync(path.join(fixturesDir, file), "utf8");
  const snap = snapshotFor(grammar, src);
  const snapPath = path.join(snapshotsDir, file.replace(/\.stark$/, ".snap"));
  if (update || !fs.existsSync(snapPath)) {
    fs.writeFileSync(snapPath, snap);
    console.log(`${update ? "updated" : "created"}: ${path.basename(snapPath)}`);
  } else {
    const expected = fs.readFileSync(snapPath, "utf8");
    if (expected === snap) {
      console.log(`ok: ${file}`);
    } else {
      failed++;
      console.error(`MISMATCH: ${file} (run with --update to accept)`);
    }
  }
}
process.exit(failed === 0 ? 0 : 1);
