import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { plain } from '../story/lines.ts';
import * as F from './lines.ts';
import * as M from './model.ts';
import { FLOW_SCENES } from './scenes.ts';

// the preview data: exported by the project's own code from the saved Banking77 card-delivery quick test
const data = JSON.parse(readFileSync(new URL('../../public/data/quick-sample.json', import.meta.url), 'utf8')) as M.FlowData;
const run = data.saved_run as M.SavedRun;

const full: M.Choices = {
  description: 'tells customers where their new card is',
  dataset: 'banking77',
  labels: ['card_arrival'],
  arms: ['agent', 'linear'],
  gap: 0.1,
  volume: 10000,
  fixer: 'Support agent',
  hourly: 30,
  minutes: 5,
  table: 'cloud_small',
};

test('nothing is preselected: an empty flow cannot run, and says what comes first', () => {
  assert.deepEqual(M.missing(M.EMPTY), ['f1', 'say what your agent does']);
});

test('each missing choice is named in order, then the test can run', () => {
  assert.equal(M.missing(full), null);
  assert.deepEqual(M.missing({ ...full, dataset: null }), ['f2', 'pick a use case']);
  assert.deepEqual(M.missing({ ...full, labels: [] }), ['f3', 'pick at least one label']);
  assert.deepEqual(M.missing({ ...full, arms: ['linear'] }), ['f4', 'include your agent']);
  assert.deepEqual(M.missing({ ...full, arms: ['agent', 'prompt'] }), ['f4', 'add at least one free model to beat']);
  assert.deepEqual(M.missing({ ...full, gap: null }), ['f5', 'pick how big a gain matters']);
  assert.deepEqual(M.missing({ ...full, volume: null }), ['f6', 'enter messages a month']);
  assert.deepEqual(M.missing({ ...full, table: null }), ['f6', 'pick where the AI would run']);
});

test('amounts: empty or negative text is not a number yet', () => {
  assert.equal(M.parseAmount(''), null);
  assert.equal(M.parseAmount('abc'), null);
  assert.equal(M.parseAmount('-3'), null);
  assert.equal(M.parseAmount('0', 1), null);
  assert.equal(M.parseAmount('12.5'), 12.5);
});

test('one mistake and the monthly cost follow the stated formulas', () => {
  assert.equal(M.mistakeCost(30, 5), 2.5);
  const arm = { name: 'x', accuracy: 0.9, inference_cost: { t: 0.001 } };
  assert.ok(Math.abs(M.monthlyCost(arm, 't', 10000, 2.5) - 10000 * (0.001 + 0.1 * 2.5)) < 1e-9);
});

test('test size comes from the exported power table; minutes from the measured seconds', () => {
  assert.equal(M.plannedCases(data, 0.1), 168);
  assert.equal(M.plannedCases(data, 0.05), 658);
  assert.equal(M.plannedCases(data, 0.33), null);
  assert.ok(Math.abs(M.maxMinutes(data, 168, ['agent', 'linear']) - (168 * 1.05) / 60) < 1e-9);
});

test('the saved run stopped at check-in 2 of 5 (67 of 168 cases)', () => {
  assert.equal(M.stopLook(run), 1);
  assert.deepEqual(M.racedArms(run, ['boosting', 'linear', 'agent']), ['agent', 'linear']);
});

test('the verdict word comes from the engine verdict, never from a ranking', () => {
  assert.equal(F.verdictWord({ verdict: 'USE_SIMPLER', simpler_arm: 'linear' }), 'Use Linear');
  assert.equal(F.verdictWord({ verdict: 'DEPLOY_AGENT', simpler_arm: 'linear' }), 'Deploy your agent');
  assert.equal(F.verdictWord({ verdict: 'NEED_MORE_DATA', simpler_arm: 'linear' }), 'Not clear yet');
  assert.equal(F.verdictWord({ verdict: 'INVALID', simpler_arm: 'linear' }), 'Test not valid');
  assert.match(F.verdictLead({ verdict: 'NEED_MORE_DATA', simpler_arm: 'linear', accuracy_diff: 0.01, n_cases: 168 }, '', 168), /No winner/);
});

test('the saved run reads as USE_SIMPLER vs Linear, stopped early', () => {
  assert.equal(F.verdictWord(run.verdict), 'Use Linear');
  assert.match(F.verdictLead(run.verdict, 'x', run.plan.planned_cases), /Stopped early at 67 of 168 cases/);
  assert.equal(plain(F.scoresLine(run, ['agent', 'linear'])), 'Linear got 96% of your messages right. Your agent got 74%.');
});

test('robot lines react to the choices', () => {
  assert.match(plain(F.helloLine('')), /Tell me what your agent does/);
  assert.match(plain(F.contendersLine(['linear'], true)), /Tick your agent/);
  assert.match(plain(F.contendersLine(['agent'], false)), /free model/);
  assert.match(plain(F.gapLine(168, 2.9)), /Up to 168 cases, about 3 minutes/);
  assert.equal(F.minutesText(0.4), 'under a minute');
  assert.match(plain(F.raceLine('blocked', 'pick a use case')), /Still needed: pick a use case/);
});

test('9 scenes, alternating sides from the right, like the story', () => {
  assert.equal(FLOW_SCENES.length, 9);
  FLOW_SCENES.forEach((s, i) => assert.equal(s.side, i % 2 === 0 ? 'right' : 'left'));
});

test('only test-changing choices clear a finished run; cost answers just re-price it', () => {
  assert.equal(M.changesTheTest({ gap: 0.05 }), true);
  assert.equal(M.changesTheTest({ labels: [] }), true);
  assert.equal(M.changesTheTest({ volume: 5 }), false);
  assert.equal(M.changesTheTest({ table: 'cloud_large', hourly: 40 }), false);
});

test('the race band is the repeated confidence interval rebuilt from the printed z values', () => {
  // gain -0.119 at 67 cases, from the saved run's own output
  const band = M.lookInterval({ n: 67, gain: -0.119, z_agent: -2.75, z_ruled_out: -6.04, bound: 3.1 }, 0.1);
  assert.ok(band);
  const [lo, hi] = band as [number, number];
  assert.ok(Math.abs(lo - (-0.119 - 3.1 * (-0.119 / -2.75))) < 1e-12);
  assert.ok(hi < 0.1, 'a gain of 10 points is ruled out');
  assert.equal(M.lookInterval({ n: 34, gain: 0, z_agent: 0, z_ruled_out: -1, bound: 4.38 }, 0.1), null);
});

test('scores: in-scope accuracy for a scoped run, the engine accuracy for a full run', () => {
  assert.ok(Math.abs((M.shownAccuracy(run, 'linear') as number) - 0.9629629629629629) < 1e-12);
  const full = { ...run, scope: null };
  assert.equal(M.shownAccuracy(full, 'linear'), run.verdict.arms.find((a) => a.name === 'linear')?.accuracy);
  assert.equal(M.shownAccuracy(full, 'nope'), null);
});

test('stopped early only when fewer cases than planned were used', () => {
  assert.equal(M.stoppedEarly(run), true);
  const fullRun = { ...run, verdict: { ...run.verdict, n_cases: 168 } };
  assert.equal(M.stoppedEarly(fullRun), false);
  assert.match(plain(F.raceLine('done', null, 4, false)), /All five check-ins ran/);
  assert.doesNotMatch(F.verdictLead(fullRun.verdict, '', 168), /Stopped early/);
  assert.deepEqual(M.replayLooks(run), [34, 67]);
});

test('ready datasets: built in, or a built split', () => {
  assert.equal(M.datasetReady({ status: 'built in' }), true);
  assert.equal(M.datasetReady({ status: 'split built' }), true);
  assert.equal(M.datasetReady({ status: 'available' }), false);
  assert.equal(M.datasetReady({ status: 'chosen' }), false);
});
