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
