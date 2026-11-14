import test from 'node:test';
import assert from 'node:assert/strict';
import { failureReason } from './run.ts';

test('a failed job shows its last output line', () => {
  assert.equal(failureReason(['a', 'Ollama is not reachable', '  ']), 'Ollama is not reachable');
  assert.equal(failureReason([]), 'the test ended without a result');
});

import { runTables } from './model.ts';

test('a run is priced with the tables in its own verdict, not the menu’s', () => {
  const known = [
    { name: 'local_amortized', input_usd_per_mtok: 0.05, output_usd_per_mtok: 0.1, assumed: true },
    { name: 'cloud_small', input_usd_per_mtok: 0.15, output_usd_per_mtok: 0.6, assumed: true },
  ];
  const verdict = { arms: [{ inference_cost: { cloud_small: 0.0002, own_table: 0.001 } }] } as never;
  const got = runTables(verdict, known);
  assert.deepEqual(got.map((t) => t.name), ['cloud_small', 'own_table']);
  assert.equal(got[0], known[1]);
});
