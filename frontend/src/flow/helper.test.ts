import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { answer, readNumbers, type HelperContext, type Seg } from './helper.ts';

// the real refund quick test, as exported for the mockup
const run = JSON.parse(readFileSync(new URL('../../public/data/mock-run.json', import.meta.url), 'utf8'));
const ctx: HelperContext = {
  messages: run.messages,
  arms: run.verdict.arms,
  scope: run.scope,
  verdict: run.verdict.verdict,
  simpler: run.plan.comparator,
  gap: run.plan.gap,
  ci: run.verdict.accuracy_diff_ci,
  nCases: run.verdict.n_cases,
  planned: run.plan.population,
  looks: run.plan.early_stopping.looks,
  volume: 10000,
  mistake: 22.5,
  table: 'cloud_small',
  label: (a) => (a === 'linear' ? 'Linear' : a === 'agent' ? 'Your agent' : a),
};
const text = (segs: Seg[]) => segs.map((s) => (typeof s === 'string' ? s : s.b)).join('');

test('numbers are read from the question', () => {
  assert.deepEqual(readNumbers('what if a mistake costs $5?'), { money: 5, volume: null });
  assert.deepEqual(readNumbers('at 100k messages a month'), { money: null, volume: 100000 });
  assert.deepEqual(readNumbers('20,000 messages and 7 dollars'), { money: 7, volume: 20000 });
});

test('wrong answers: lists the real messages and lights them up', () => {
  const a = answer('Which messages did my agent get wrong?', ctx);
  const wrong = run.messages.filter((m: { agent_ok: boolean }) => !m.agent_ok).length;
  assert.match(text(a.text), new RegExp(`${wrong} of ${run.messages.length}`));
  assert.equal(a.focus?.length, wrong);
  assert.match(text(a.list?.[0] ?? []), /How long will it take to refund me/);
  assert.equal(a.go, 'm2');
});

test('the simpler model’s mistakes are asked for by name', () => {
  const a = answer('what did linear get wrong', ctx);
  assert.match(text(a.text), /^Linear got 1 of/);
});

test('patterns: what the agent mixed up, most common first', () => {
  const a = answer('what do these have in common', ctx);
  assert.ok((a.list?.length ?? 0) >= 1);
  assert.match(text(a.list?.[0] ?? []), /×/);
});

test('costs use the reader’s number and say whether the agent is ever cheaper', () => {
  const a = answer('What if a mistake costs $5?', ctx);
  assert.match(text(a.text), /\$5 per mistake/);
  assert.match(text(a.list?.[0] ?? []), /never cheaper/); // the agent is less accurate on this run
});

test('how sure: the interval, the stop and the wrong-winner limit', () => {
  const a = answer('how sure is this?', ctx);
  assert.match(text(a.text), /ruled out/);
  assert.match(text(a.list?.[0] ?? []), /check-in 4/);
  assert.match(text(a.list?.[1] ?? []), /5%/);
});

test('breakdown by kind covers every kind in the run', () => {
  const a = answer('break it down by kind', ctx);
  const kinds = new Set(run.messages.map((m: { gold: string }) => m.gold));
  assert.equal(a.list?.length, kinds.size);
});

test('an unknown question is answered honestly, with suggestions', () => {
  const a = answer('what is the weather', ctx);
  assert.match(text(a.text), /only answer from this run/);
  assert.ok(a.chips.length > 0);
});
