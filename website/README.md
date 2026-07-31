# starklang.com

The STARK language homepage. React + TypeScript, built with Vite, output is a static bundle.

```bash
npm install
npm run dev        # local development
npm run build      # static output in dist/
npm run preview    # serve the built output
npm run typecheck
```

## Deploying

`dist/` is plain static files with a relative `base`, so it serves from a domain root, a project
subpath, or `file://` without rebuilding. Any static host works — GitHub Pages, Netlify, Vercel,
Cloudflare Pages, S3.

## Where the content lives

**All copy is in `src/content.ts`**, deliberately, and it is the only file that needs editing when
the compiler's position changes.

A language homepage is mostly a claims document, and claims drift out of date faster than layout
does. Keeping them in one short file means they can be re-read against `COMPILER-STATE.md` in a few
minutes, rather than hunted through JSX. The rule applied throughout: **state what is true today,
with its limits attached.** The status section names what does not work as plainly as what does,
because a language that checks your assumptions should not misrepresent its own.

## Syntax highlighting

`src/stark-highlight.ts` is a small hand-written tokeniser, because no highlighting library knows
STARK. Its keyword and primitive lists come from the compiler's own lexer and type model — if the
language gains a keyword, that list is where it is added. Being wrong there shows up as
unhighlighted text, never as a broken page.

## Brand

The palette matches `editors/vscode/icons/stark.svg` — a near-black ground under a cyan-to-indigo
mark — so the site and the editor extension read as one product. Dark by default, light on request
via `prefers-color-scheme`.
