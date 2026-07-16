import esbuild from 'esbuild';
import process from 'process';

const args = process.argv.slice(2);
const watch = args.includes('--watch');
const minify = args.includes('--minify');

const ctx = await esbuild.context({
  entryPoints: ['src/extension.ts'],
  bundle: true,
  outfile: 'dist/extension.js',
  external: ['vscode'],
  format: 'cjs',
  platform: 'node',
  minify: minify,
  sourcemap: !minify,
  logLevel: 'info',
});

if (watch) {
  await ctx.watch();
} else {
  await ctx.rebuild();
  await ctx.dispose();
}
