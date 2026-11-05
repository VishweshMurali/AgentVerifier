import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { StaticJsonProvider, StoryDataError, validateStory, type FetchLike } from './provider.ts';

const raw = readFileSync(new URL('../../public/data/story.json', import.meta.url), 'utf8');
const good = () => JSON.parse(raw) as Record<string, any>;

function fakeFetch(body: unknown, status = 200): { fetch: FetchLike; urls: string[] } {
  const urls: string[] = [];
  return {
    urls,
    fetch: async (url: string) => {
      urls.push(url);
      return {
        ok: status >= 200 && status < 300,
        status,
        statusText: status === 404 ? 'Not Found' : '',
        json: async () => {
          if (typeof body === 'string') return JSON.parse(body);
          return body;
        },
      };
    },
  };
}

test('loads and validates the real story.json', async () => {
  const f = fakeFetch(good());
  const story = await new StaticJsonProvider('./data/story.json', f.fetch).load();
  assert.deepEqual(f.urls, ['./data/story.json']);
  assert.equal(story.source, 'rehearsal');
  assert.equal(story.verdict.agent_arm, 'agent');
  assert.equal(story.verdict.arms.length, 3);
});

test('a null case is allowed', () => {
  const d = good();
  d.case = null;
  assert.equal(validateStory(d).case, null);
});

test('missing fields give one readable error naming each field', () => {
  const d = good();
  delete d.verdict.accuracy_diff;
  d.verdict.arms[1].accuracy = 'high';
  delete d.default_price_table;
  try {
    validateStory(d);
    assert.fail('should have thrown');
  } catch (err) {
    assert.ok(err instanceof StoryDataError);
    assert.match(err.message, /"verdict\.accuracy_diff" is missing or not a number/);
    assert.match(err.message, /"verdict\.arms\[1\]\.accuracy" is missing or not a number from 0 to 1/);
    assert.match(err.message, /"default_price_table" is missing/);
    assert.equal(err.problems.length, 3);
  }
});

test('bad source, unknown arm, missing price for a table', () => {
  const d = good();
  d.source = 'live';
  d.verdict.simpler_arm = 'ghost';
  delete d.verdict.arms[0].inference_cost.cloud_large;
  assert.throws(() => validateStory(d), (err: unknown) => {
    const m = (err as Error).message;
    return (
      /"source"/.test(m) &&
      /"verdict\.simpler_arm" is missing or not the name of one of verdict\.arms/.test(m) &&
      /"verdict\.arms\[0\]\.inference_cost\.cloud_large"/.test(m)
    );
  });
});

test('not an object, HTTP error and bad JSON are readable errors', async () => {
  assert.throws(() => validateStory([1, 2]), /not a JSON object/);
  await assert.rejects(new StaticJsonProvider('./x.json', fakeFetch({}, 404).fetch).load(), /HTTP 404 Not Found/);
  await assert.rejects(new StaticJsonProvider('./x.json', fakeFetch('{nope').fetch).load(), /not valid JSON/);
  const down: FetchLike = async () => {
    throw new Error('offline');
  };
  await assert.rejects(new StaticJsonProvider('./x.json', down).load(), /Could not reach \.\/x\.json: offline/);
});
