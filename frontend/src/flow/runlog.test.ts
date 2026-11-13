import test from 'node:test';
import assert from 'node:assert/strict';
import { runLog } from './runlog.ts';

const texts = (p: Parameters<typeof runLog>[0]) => runLog(p).steps.map((s) => (s.now ? `> ${s.text}` : s.text));

test('before the plan prints, only the start is running', () => {
  assert.deepEqual(texts({ phase: 'starting' }), ['> Starting the test and picking messages…']);
  assert.equal(runLog(null).headline, 'Starting your test…');
});

test('training names the model in training, even between models', () => {
  assert.deepEqual(texts({ phase: 'training', drawn: 39, trained: ['majority'], training: 'linear' }), [
    'Picked 39 real messages to test',
    'Trained 1 of 6 free models',
    '> Training linear…',
  ]);
  // after "trained X" the server drops `training` but leaves the phase
  assert.deepEqual(texts({ phase: 'training', drawn: 39, trained: ['majority', 'linear'] }).at(-1), '> Training the free models…');
});

test('racing counts the agent and says how often it compared', () => {
  const p = { phase: 'racing', drawn: 39, trained: ['a', 'b', 'c', 'd', 'e', 'f'], simpler: 'linear_svm', scored: 16, looks: [{ n: 8 }] };
  assert.deepEqual(texts(p), [
    'Picked 39 real messages to test',
    'Trained 6 free models',
    'linear_svm is the one to beat',
    'Compared them 1 time: too close to stop yet',
    '> Your agent has answered 16 of 39…',
  ]);
  // the headline does not change with the count, so screen readers are not flooded
  assert.equal(runLog({ ...p, scored: 17 }).headline, runLog(p).headline);
});

test('an early stop and the end are named', () => {
  const p = { drawn: 168, trained: ['a'], simpler: 'linear', stopped_at: 67, of: 168 };
  assert.equal(texts({ ...p, phase: 'finishing' }).at(-2), 'Stopped early at 67 of 168: clear enough');
  assert.equal(runLog({ ...p, phase: 'done' }).headline, 'Done! Your result is ready.');
  assert.ok(runLog({ ...p, phase: 'done' }).steps.every((s) => !s.now));
});

test('all free models trained, before the one to beat is named', () => {
  const p = { phase: 'training', drawn: 168, trained: ['a', 'b', 'c', 'd', 'e', 'f'] };
  assert.equal(texts(p).at(-1), '> Picking the free model your agent must beat…');
});
