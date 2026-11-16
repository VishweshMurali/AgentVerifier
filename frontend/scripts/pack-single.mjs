// Packs the built page (frontend/dist) into ONE html file that opens anywhere (a chat viewer, a phone,
// file://): the script, styles and example data are inlined, and the page's switches are set in
// window.__flags because a single file has no address to carry them.
//   npm run build && node scripts/pack-single.mjs ../design/frontend/sphere-opening.html sphere,tune
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dist = process.env.DIST ? resolve(process.env.DIST) : join(here, '..', 'dist');
const [out = '../design/frontend/sphere-opening.html', flagList = 'sphere,tune'] = process.argv.slice(2);
let html = readFileSync(join(dist, 'index.html'), 'utf8');
const safe = (s) => s.replace(/<\/(script|style)/gi, '<\\/$1');
const js = html.match(/<script type="module"[^>]*src="\.\/(assets\/[^"]+)"[^>]*><\/script>/);
const css = html.match(/<link rel="stylesheet"[^>]*href="\.\/(assets\/[^"]+)"[^>]*>/);
if (!js || !css) throw new Error('could not find the built script and stylesheet in dist/index.html');
const files = {};
for (const f of ['data/mock-run.json', 'data/quick-sample.json', 'data/story.json']) files[f] = readFileSync(join(dist, f), 'utf8');
const flags = Object.fromEntries(flagList.split(',').filter(Boolean).map((f) => [f, true]));
const shim = `<script>
window.__flags = ${JSON.stringify(flags)};
(function () {
  var files = ${safe(JSON.stringify(files))}, real = window.fetch.bind(window);
  window.fetch = function (u, o) {
    var k = String(u).split('?')[0];
    if (k.indexOf('./') === 0) k = k.slice(2);
    if (Object.prototype.hasOwnProperty.call(files, k)) return Promise.resolve(new Response(files[k], { status: 200, headers: { 'content-type': 'application/json' } }));
    return real(u, o);
  };
})();
</script>`;
// function replacers: the script text holds "$&"-style sequences that a string replacement would expand
const style = `<style>${safe(readFileSync(join(dist, css[1]), 'utf8'))}</style>`;
const app = `<script type="module">${safe(readFileSync(join(dist, js[1]), 'utf8'))}</script>\n`;
html = html.replace(js[0], () => '').replace(css[0], () => style);
html = html.replace('</head>', () => `${shim}\n</head>`).replace('</body>', () => `${app}</body>`);
const target = resolve(process.cwd(), out);
writeFileSync(target, html);
console.log(`wrote ${target} (${Math.round(html.length / 1024)} KB, flags: ${Object.keys(flags).join(', ') || 'none'})`);
