import test from 'node:test';
import assert from 'node:assert/strict';
import { failureReason } from './run.ts';

test('a failed job shows its last output line', () => {
  assert.equal(failureReason(['a', 'Ollama is not reachable', '  ']), 'Ollama is not reachable');
  assert.equal(failureReason([]), 'the test ended without a result');
});
