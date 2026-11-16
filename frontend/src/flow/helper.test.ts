import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { answer, readNumbers, STARTERS, type HelperContext, type Seg } from './helper.ts';

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

test('the guide answers about the reader’s own use case, not the example run', () => {
  const uc = { title: 'Customer mood', description: 'spots angry customers', kinds: ['anger', 'joy'], totalKinds: 6, gapPoints: 10, hasRun: false, running: false, planned: 168 };
  const a = answer('what will you test for my use case?', { ...ctx, useCase: uc });
  const t = text(a.text) + ' ' + (a.list ?? []).map(text).join(' ');
  assert.ok(a.own);
  assert.match(t, /Customer mood/);
  assert.match(t, /spots angry customers/);
  assert.match(t, /anger, joy/);
  assert.match(t, /168 messages/);
  assert.match(t, /No test has run/);
});

test('with no use case picked the guide says so instead of answering about the example', () => {
  const a = answer('what are you testing?', { ...ctx, useCase: { title: null, description: '', kinds: [], totalKinds: null, gapPoints: 10, hasRun: false, running: false, planned: null } });
  assert.match(text(a.text), /haven’t picked a use case/);
});

test('questions about the result still go to the result once there is a run', () => {
  const uc = { title: 'Bank support routing', description: '', kinds: [], totalKinds: 77, gapPoints: 10, hasRun: true, running: false, planned: null };
  const a = answer('which messages were wrong?', { ...ctx, useCase: uc });
  assert.ok(!a.own);
});

// every follow-up the guide offers must lead somewhere: never the "didn't catch that" fallback, and
// never the same answer that offered it (a dead-end loop), in each state the page can be in
const uc = (over: object) => ({ title: 'Customer mood', description: 'spots angry customers', kinds: ['anger'], totalKinds: 6, gapPoints: 10, hasRun: false, running: false, planned: 168, ...over });
for (const [state, useCase] of [
  ['the static site', undefined],
  ['a use case picked, no test yet', uc({})],
  ['no use case picked yet', uc({ title: null, kinds: [], totalKinds: null })],
  ['after their own test', uc({ hasRun: true })],
] as const) {
  test(`every chip leads somewhere: ${state}`, () => {
    const seen = new Set<string>();
    const queue = [...answer('hello', { ...ctx, useCase }).chips, ...STARTERS];
    while (queue.length) {
      const chip = queue.shift() as string;
      if (seen.has(chip)) continue;
      seen.add(chip);
      const a = answer(chip, { ...ctx, useCase });
      const t = text(a.text);
      assert.ok(!/didn’t catch that/.test(t), `“${chip}” hits the fallback`);
      const parent = [...seen].find((p) => p !== chip && answer(p, { ...ctx, useCase }).chips.includes(chip));
      if (parent) assert.notEqual(t, text(answer(parent, { ...ctx, useCase }).text), `“${chip}” repeats the answer that offered it`);
      queue.push(...a.chips);
    }
  });
}
