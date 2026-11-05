// Picks the data provider from VITE_DATA_PROVIDER (build-time env). Only "static" exists now.
//
// Adding a provider later (for example LangSmith, so runs, traces and the verdict sit in one place):
//   1. Write a class in its own file, e.g. src/data/langsmithProvider.ts, that implements
//      DataProvider from ./provider.ts. Its load() fetches the runs and maps them into a Story,
//      then returns validateStory(mapped) so the page gets the same checked shape.
//   2. Add a case for it below (e.g. 'langsmith'), reading its settings from VITE_* env variables.
//   3. Build with VITE_DATA_PROVIDER=langsmith.
// Nothing here calls any service other than the page's own static file.
import { StaticJsonProvider, StoryDataError, type DataProvider } from './provider.ts';

export function createProvider(kind: string | undefined = import.meta.env.VITE_DATA_PROVIDER): DataProvider {
  const name = (kind ?? 'static').trim() || 'static';
  switch (name) {
    case 'static':
      return new StaticJsonProvider('./data/story.json');
    default:
      throw new StoryDataError(`Unknown data provider "${name}". Set VITE_DATA_PROVIDER to "static".`);
  }
}

export type { DataProvider } from './provider.ts';
export type * from './types.ts';
