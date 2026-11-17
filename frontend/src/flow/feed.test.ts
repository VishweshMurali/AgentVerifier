import test from 'node:test';
import assert from 'node:assert/strict';
import { clock, feedEvents, type CallStats } from './feed.ts';
import type { RunProgress } from './runlog.ts';

const name = (a: string) => ({ linear_svm: 'Linear SVM', naive_bayes: 'Naive Bayes' })[a] ?? a;
const calls = (o: Partial<CallStats> = {}): CallStats => ({ requests: 0, replayed: 0, failed: 0, tokens: 0, avg_s: 0, last_s: 0, ...o });
const lines = (prev: [RunProgress | null, CallStats | null], next: [RunProgress | null, CallStats | null], t = 5) =>
  feedEvents({ progress: prev[0], calls: prev[1] }, { progress: next[0], calls: next[1] }, t, name, 'The hosted test agent').map((e) => e.text);

test('the feed says nothing when nothing changed', () => {
  const p = { drawn: 39, trained: ['majority'], scored: 10 };
  assert.deepEqual(lines([p, calls({ requests: 4 })], [p, calls({ requests: 4 })]), []);
});

test('the feed lists the first things the test prints, once', () => {
  assert.deepEqual(lines([null, null], [{ drawn: 39, trained: ['naive_bayes'], simpler: 'linear_svm' }, null]), [
    'Picked 39 real messages to test',
    'Trained Naive Bayes, a free model',
    'Linear SVM is the free model the hosted test agent has to beat',
  ]);
});

test('requests, replays and errors are counted separately from the ledger', () => {
  const got = lines([{ drawn: 39 }, calls({ requests: 4 })], [{ drawn: 39 }, calls({ requests: 7, last_s: 1.26, replayed: 2, failed: 1 })]);
  assert.deepEqual(got, [
    '3 requests sent to the model (7 so far, last answer in 1.3 s)',
    '2 answers replayed from saved answers (no request needed)',
    '1 request ended in an error (counted as wrong)',
  ]);
});

test('a check-in shows the test’s own gain, and progress in messages', () => {
  const before: RunProgress = { drawn: 39, simpler: 'linear_svm', looks: [], scored: 0 };
  const after: RunProgress = { ...before, scored: 10, looks: [{ n: 8, gain: -0.043 }] };
  assert.deepEqual(lines([before, null], [after, null]), [
    'Check-in 1 at 8 messages: the hosted test agent is 4.3 points behind Linear SVM',
    'The hosted test agent has answered 10 of 39',
  ]);
});

test('the feed says when the test stopped early and when the answer is ready', () => {
  const prev: RunProgress = { drawn: 39, phase: 'racing' };
  assert.deepEqual(lines([prev, null], [{ ...prev, phase: 'finishing', stopped_at: 23, of: 39 }, null]), [
    'Stopped early at 23 of 39 messages: clear enough',
    'Scoring the other models and writing your answer',
  ]);
  assert.deepEqual(lines([{ ...prev, phase: 'finishing' }, null], [{ ...prev, phase: 'done' }, null]), ['Your answer is ready']);
});

test('the place in the waiting line shows only when it changes', () => {
  assert.deepEqual(lines([{ queued: 2 }, null], [{ queued: 1 }, null]), ['Waiting in line: 1 test ahead of yours']);
  assert.deepEqual(lines([{ queued: 1 }, null], [{ queued: 1 }, null]), []);
});

test('the clock reads minutes and seconds', () => {
  assert.equal(clock(75), '1:15');
  assert.equal(clock(-3), '0:00');
});
