import test from 'node:test';
import assert from 'node:assert/strict';
import type { LiveStatus } from '../data/types.ts';
import { liveLine, sealedRunText, verdictStamp } from './live.ts';

const base = (over: Partial<LiveStatus['sealed_run']> = {}): LiveStatus => ({
  ollama: true,
  sealed_run: {
    state: 'not_started',
    started_at_commit: null,
    verdict: null,
    test_cases: 3271,
    answers_cached: {},
    ...over,
  },
  rehearsal: { verdict: { mtime: 5 } },
  job: null,
});

test('sealed run states read plainly', () => {
  assert.equal(sealedRunText(base()), 'Sealed run not started');
  assert.equal(sealedRunText(base({ state: 'finished' })), 'Sealed run finished');
  assert.equal(sealedRunText(base({ state: 'started_not_finished' })), 'Sealed run started, no answers yet');
  assert.equal(
    sealedRunText(base({ state: 'started_not_finished', answers_cached: { prompt: 1200, agent: 3 } })),
    'Sealed run started: agent 3, prompt 1,200 of ~3,271 answers',
  );
});

test('live line joins Ollama, the run and a job', () => {
  const s = { ...base(), ollama: false, job: { id: 'x', kind: 'rehearsal', state: 'running' } };
  assert.equal(liveLine(s), 'Ollama off · Sealed run not started · rehearsal running');
});

test('the verdict stamp changes when a verdict file is written', () => {
  const a = verdictStamp(base());
  const b = verdictStamp(base({ verdict: { mtime: 9 } }));
  assert.notEqual(a, b);
  assert.equal(a, verdictStamp(base()));
});
