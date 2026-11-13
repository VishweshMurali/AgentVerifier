# Agent Verifier front end

React + Vite + TypeScript. One build, four pages chosen by the address:

| Address | Page |
|---|---|
| `/` | **the live page** (`LiveApp`, `components/MockPage.tsx`): a ten-screen scroll story. Describe your agent, pick its kinds of message, race it against simpler models, see one mistake, set your costs, then get the answer. With the local server it runs real quick tests; without it, it shows a saved example run. |
| `?story` | the earlier nine-scene story of the sealed Banking77 result (`story.json`) |
| `?test` | the quick-test flow prototype (`FlowApp`) |
| `?mock` | the static design mockup (`MockApp`) |

Every number on the pages comes from the pipeline's files or the local server; plain code picks the
robot's lines. No AI writes text on the main page. The live page says plainly when it shows the
example and when it shows your run, and which agent raced (a local Ollama model unless you bring
your own).

Stack: React + Vite + TypeScript only (exact versions are pinned in `package.json`).

## Run it locally

```sh
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Other scripts:

| Script | What it does |
|---|---|
| `npm run typecheck` | `tsc --noEmit` over `src/` |
| `npm test` | Node's built-in test runner on the pure logic (`src/**/*.test.ts`, Node 24 type stripping) |
| `npm run build` | typecheck, then `vite build` into `dist/` |
| `npm run preview` | serve the built `dist/` locally |

## Refresh the data

The data file comes from the pipeline. Run this from the **repository root**, not from `frontend/`:

```sh
uv run python scripts/export_frontend_data.py
```

It writes `frontend/public/data/story.json` from the sealed verdict (`results/verdict.json`) when
that exists, otherwise from the DEV rehearsal. Do not edit `story.json` by hand. Reload the dev
server, or rebuild, to pick it up.

## Build and deploy

```sh
npm run build
```

`dist/` is a fully static site (`index.html`, `assets/`, `data/story.json`). Vite's `base` is
`'./'`, so every path is relative and the site works on any static host and under any sub-path:

- **Netlify / Vercel / Cloudflare Pages**: build command `npm run build`, publish directory
  `frontend/dist` (base directory `frontend`).
- **GitHub Pages**: publish the contents of `dist/` (for example to a `gh-pages` branch). A project
  page such as `https://user.github.io/repo/` works as is.
- **S3 or any web server**: upload the contents of `dist/`, with `index.html` as the index document.

The story is rebuilt from data, so refresh `story.json` before each build you deploy.

## Code layout

```
src/
  LiveApp.tsx          the live page: loads the example, the server's menu and libraries, runs tests
  components/MockPage.tsx   the ten screens, the robot's lines, the costs, the answer
  components/UseCases.tsx   use-case cards, bring your own data, bring your agent
  components/ScrollText.tsx words and cards built by scrolling
  motion/brain.ts      the 3D message graph behind the page
  flow/                pure, tested helpers: run steps, time estimate, guide answers, run mapping

  data/types.ts        TypeScript types mirroring story.json
  data/provider.ts     DataProvider interface, StaticJsonProvider, validateStory (pure, tested)
  data/index.ts        picks the provider from VITE_DATA_PROVIDER
  story/scenes.ts      the 9 scenes: ids, names, robot side and mood
  story/cost.ts        expected cost = inference cost + (1 - accuracy) x error cost (pure, tested)
  story/lines.ts       the robot's fixed lines and the result sentences (pure, tested)
  story/format.ts      number formatting
  hooks/useActiveScene.ts  current scene via IntersectionObserver
  components/          Robot (SVG + moods), Timeline, StoryPage, RichText
  styles.css           the prototype's CSS: flight keyframes, moods, content beats, phone layout
```

## Adding a data provider (for example LangSmith)

The page never fetches data directly; it asks a `DataProvider`:

```ts
interface DataProvider { name: string; load(): Promise<Story> }
```

To add one:

1. Create `src/data/langsmithProvider.ts` with a class that implements `DataProvider`. In
   `load()`, fetch what you need, map it into the `Story` shape from `src/data/types.ts`, and
   return `validateStory(mapped)` so the page gets the same checked data. Keep it free of JSX and
   DOM so it can be tested with `node --test` and a fake fetch, like `src/data/provider.test.ts`.
2. Add a `case 'langsmith':` to `createProvider` in `src/data/index.ts`, reading its settings
   from `VITE_*` environment variables (never commit keys; anything `VITE_*` is public in the
   built page, so a secret needs a small server in between).
3. Build with `VITE_DATA_PROVIDER=langsmith npm run build`.

Only the `static` provider exists today. The scoring and the verdict still come from the
pipeline's deterministic code; a provider only moves the results to the page.
