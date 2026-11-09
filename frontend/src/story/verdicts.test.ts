import test from 'node:test';
// Every verdict the engine can return must read sensibly on the recap, not only USE_SIMPLER.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { Story } from '../data/types.ts';
import * as L from './lines.ts';

const base = JSON.parse(readFileSync(new URL('../fixtures/story.rehearsal.json', import.meta.url), 'utf8')) as Story;

function variant(over: Partial<Story['verdict']>): Story {
  return { ...base, source: 'sealed', verdict: { ...base.verdict, evidence_level: 'REAL', ...over } };
}

const deploy = variant({
  verdict: 'DEPLOY_AGENT',
  accuracy_diff: 0.05,
  accuracy_diff_ci: [0.03, 0.07],
  reasons: ['The agent beats the best simpler arm by at least the minimum gap.'],
  break_even: [{ price_table: 'cloud_small', value: 3.2, ci_low: 2.1, ci_high: 4.8 }],
});
const more = variant({
  verdict: 'NEED_MORE_DATA',
  accuracy_diff: -0.01,
  accuracy_diff_ci: [-0.03, 0.01],
  reasons: ['The accuracy difference CI includes 0 and the minimum gap.'],
});
const invalid = variant({
  verdict: 'INVALID',
  quality_floor_passed: false,
  reasons: ['The agent failed the quality floor, so the run is invalid.'],
});

for (const [name, s] of [['DEPLOY_AGENT', deploy], ['NEED_MORE_DATA', more], ['INVALID', invalid]] as const) {
  test(`${name}: recap heading and tag match the verdict`, () => {
    assert.equal(L.recapHeading(s), {
      DEPLOY_AGENT: 'The verdict: deploy your agent.',
      NEED_MORE_DATA: 'The verdict: we need more data.',
      INVALID: 'The verdict: this run is invalid.',
    }[name]);
    assert.match(L.verdictTag(s), new RegExp(`verdict ${name}`));
  });
}

test('INVALID recap does not crown a winner', () => {
  const lead = L.recapLead(invalid, base.default_price_table);
  assert.doesNotMatch(lead, /more accurate|less accurate|cheaper/);
});

test('NEED_MORE_DATA recap does not claim a winner', () => {
  const lead = L.recapLead(more, base.default_price_table);
  assert.doesNotMatch(lead, /more accurate|less accurate/);
});

test('DEPLOY_AGENT recap says the agent is ahead and shows the break-even', () => {
  const lead = L.recapLead(deploy, base.default_price_table);
  assert.match(lead, /less accurate|your agent/);
  assert.match(lead, /\$3\.20/);
});

// Scenes 4 and 6 read the sealed verdict too, so an undecided or broken run is never celebrated.
const invalidAhead = variant({ verdict: 'INVALID', accuracy_diff: 0.04, accuracy_diff_ci: [0.01, 0.07] });
const moreAhead = variant({ verdict: 'NEED_MORE_DATA', accuracy_diff: 0.01, accuracy_diff_ci: [-0.01, 0.03] });
const arms = [
  { name: 'agent', accuracy: 0.93 },
  { name: 'linear', accuracy: 0.9 },
];
const plain = (r: L.Rich) => L.plain(r);

test('INVALID: the reveal, scores and recap line do not celebrate', () => {
  assert.doesNotMatch(plain(L.revealLine(0.04, 'INVALID')), /ahead!/);
  assert.match(plain(L.revealLine(0.04, 'INVALID')), /invalid/);
  assert.equal(L.revealMood(0.04, 'INVALID'), 'thinking');
  assert.match(L.revealLead(invalidAhead), /invalid, so the gap decides nothing/);
  assert.match(plain(L.scoresPayoff('agent', arms, 'agent', 'INVALID')), /you were right.*invalid, so these scores don’t count/);
  assert.doesNotMatch(plain(L.recapLine('agent', arms, 'agent', 'INVALID')), /called it/);
});

test('NEED_MORE_DATA: an agent ahead is "not sure yet", not a win', () => {
  assert.match(plain(L.revealLine(0.01, 'NEED_MORE_DATA')), /not by enough to be sure/);
  assert.equal(L.revealMood(0.01, 'NEED_MORE_DATA'), 'thinking');
  assert.equal(L.revealMood(-0.01, 'NEED_MORE_DATA'), 'sad');
  assert.match(L.revealLead(moreAhead), /Too close to be sure/);
  assert.match(plain(L.scoresPayoff(null, arms, 'agent', 'NEED_MORE_DATA')), /too close to be sure yet/);
  assert.doesNotMatch(plain(L.recapLine('agent', arms, 'agent', 'NEED_MORE_DATA')), /called it/);
});

test('decided verdicts keep the plain lines', () => {
  assert.equal(plain(L.revealLine(0.04, 'DEPLOY_AGENT')), plain(L.revealLine(0.04)));
  assert.equal(L.revealMood(0.04, 'DEPLOY_AGENT'), 'proud');
  assert.match(plain(L.recapLine('agent', arms, 'agent', 'DEPLOY_AGENT')), /called it/);
});

test('USE_SIMPLER with the agent ahead says why: the minimum gap, and "less accurate but cheaper"', () => {
  const close = variant({ verdict: 'USE_SIMPLER', accuracy_diff: 0.004, accuracy_diff_ci: [-0.01, 0.018] });
  const lead = L.recapLead(close, base.default_price_table);
  assert.match(lead, /does not lead by the 2\.0 points set before the test, so the simpler model wins/);
  assert.doesNotMatch(lead, /less accurate \([^)]*\) and cheaper/);
  const behind = variant({ verdict: 'USE_SIMPLER' });
  assert.doesNotMatch(L.recapLead(behind, base.default_price_table), /set before the test/);
});

test('break-even CI uses the run’s CI level', () => {
  const s = { ...deploy, verdict: { ...deploy.verdict, ci_level: 0.9 } };
  assert.match(L.recapLead(s, 'cloud_small'), /90% CI \$2\.10 to \$4\.80/);
});

test('DEPLOY_AGENT recap names both pre-registered conditions: the gap and a lower cost somewhere', () => {
  const lead = L.recapLead(deploy, base.default_price_table);
  assert.match(lead, /leads by at least the 2\.0 points set before the test and costs less at some error cost/);
  const rehearsal = { ...deploy, source: 'rehearsal' as const };
  assert.doesNotMatch(L.recapLead(rehearsal, base.default_price_table), /set before the test/);
});

test('the per-scenario cost reason stays out of the plain lead; other reasons stay', () => {
  const s = variant({
    verdict: 'NEED_MORE_DATA',
    reasons: [
      'A gain of 2 points is still possible.',
      `${L.SCENARIO_REASON} (price table@error cost): agent: none; linear: cloud_small@$1, cloud_small@$100.`,
    ],
  });
  const lead = L.recapLead(s, base.default_price_table);
  assert.match(lead, /A gain of 2 points is still possible\./);
  assert.doesNotMatch(lead, /@\$|per scenario/);
});

test('the quality floor check reads as valid tool use', () => {
  assert.equal(L.checkName('quality_floor', 'agent'), 'Agent’s valid tool use');
});
