import test from 'node:test';
import assert from 'node:assert/strict';
import { badgeWord, exampleState, stripCounts } from './strip.ts';

const base = { invalid: false, simplerName: 'Linear', agentName: 'the agent', simplerRight: 18, simplerOf: 23, agentRight: 17, agentOf: 23 };

test('the strip counts both sides on the reader’s own kinds, and says nothing for an invalid run', () => {
  assert.equal(stripCounts(base), 'On your kinds: Linear 18 of 23 right · the agent 17 of 23');
  assert.equal(stripCounts({ ...base, invalid: true }), null);
  assert.equal(stripCounts({ ...base, simplerOf: 0 }), null);
});

test('one badge: EXAMPLE until the run is the reader’s, then YOUR RUN', () => {
  assert.equal(badgeWord('example'), 'EXAMPLE');
  assert.equal(badgeWord('mine'), 'YOUR RUN');
});

test('the example keeps saying what is happening to the reader’s own test', () => {
  assert.equal(exampleState({ saved: true }), 'saved, not run live');
  assert.ok(exampleState({ saved: false, state: 'running' }).startsWith('your test is running'));
  assert.ok(exampleState({ saved: false, state: 'failed' }).startsWith('your test stopped'));
  assert.equal(exampleState({ saved: false }), 'yours appears here after you run it');
  assert.equal(exampleState({ saved: false, ownTitle: 'Customer mood' }), 'yours (Customer mood) appears here after you run it');
});
