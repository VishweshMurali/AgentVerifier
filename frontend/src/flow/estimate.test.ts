import test from 'node:test';
import assert from 'node:assert/strict';
import { estimateMinutes, estimateText } from './estimate.ts';

test('saved models leave only the agent and the checks', () => {
  const e = estimateMinutes({ cases: 168, looks: [34, 67, 101, 135, 168], modelsSaved: true, fitSeconds: 300, agentSeconds: 2 });
  // 15 s + 60 s + 34 * 2 s = 2.4 min; 15 + 60 + 168 * 2 * 1.15 = 7.7 min
  assert.deepEqual([e.lo, e.hi, e.firstTime], [2, 8, false]);
});

test('a first test adds the measured or the usual training time', () => {
  const fresh = estimateMinutes({ cases: 39, modelsSaved: false, fitSeconds: null, agentSeconds: null });
  assert.ok(fresh.lo >= 4 && fresh.hi >= fresh.lo && fresh.firstTime);
  assert.match(estimateText({ cases: 39, modelsSaved: false, fitSeconds: null, agentSeconds: null }), /trains the free models, once/);
});

test('the tester’s 57-message run (about 8 minutes, nothing saved) falls inside the range', () => {
  const e = estimateMinutes({ cases: 142, looks: [29, 57, 85, 114, 142], modelsSaved: false, fitSeconds: 300, agentSeconds: 2.2 });
  assert.ok(e.lo <= 8 && 8 <= e.hi, `${e.lo}-${e.hi}`);
});
