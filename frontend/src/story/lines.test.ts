import test from 'node:test';
// Fixed DEV rehearsal data, so these copy tests do not change when the live story.json does.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { Story, StoryCase } from '../data/types.ts';
import * as L from './lines.ts';

const story = JSON.parse(readFileSync(new URL('../fixtures/story.rehearsal.json', import.meta.url), 'utf8')) as Story;

const arms = [
  { name: 'agent', accuracy: 0.8242 },
  { name: 'linear', accuracy: 0.9 },
  { name: 'prompt', accuracy: 0.7476 },
];
const text = (r: L.Rich) => L.plain(r);

test('reveal: behind is sad, ahead is proud, even is thinking', () => {
  assert.equal(L.revealState(-0.075), 'behind');
  assert.equal(L.revealMood(-0.075), 'sad');
  assert.match(text(L.revealLine(-0.075)), /That’s the gap/);
  assert.equal(L.revealState(0.03), 'ahead');
  assert.equal(L.revealMood(0.03), 'proud');
  assert.match(text(L.revealLine(0.03)), /ahead/);
  assert.equal(L.revealState(0.0001), 'even');
  assert.equal(L.revealMood(0.0001), 'thinking');
  assert.equal(L.revealNumber(-0.07523939808481532), '−7.5');
});

test('reveal lead: rehearsal says preview, sealed shows the CI', () => {
  assert.match(L.revealLead(story), /DEV rehearsal preview/);
  assert.match(L.revealLead(story), /trails Linear/);
  const sealed: Story = { ...story, source: 'sealed' };
  assert.match(L.revealLead(sealed), /95% CI: −9\.4 to −5\.7 pts/);
});

test('scores payoff: right guess, last place, middle, no guess', () => {
  assert.equal(
    text(L.scoresPayoff('linear', arms, 'agent')),
    'You picked Linear, and you were right: 90.0% beats your agent’s 82.4%.',
  );
  assert.equal(
    text(L.scoresPayoff('prompt', arms, 'agent')),
    'You picked Prompt, but it came last at 74.8%. Linear won with 90.0%.',
  );
  assert.equal(
    text(L.scoresPayoff('agent', arms, 'agent')),
    'You picked your agent. It got 82.4%, but Linear won with 90.0%.',
  );
  assert.equal(text(L.scoresPayoff(null, arms, 'agent')), 'Linear got 90.0%. Your agent got 82.4%. Hmm.');
});

test('scores payoff when the agent wins', () => {
  const won = [
    { name: 'agent', accuracy: 0.93 },
    { name: 'linear', accuracy: 0.9 },
    { name: 'prompt', accuracy: 0.7 },
  ];
  assert.equal(text(L.scoresPayoff('agent', won, 'agent')), 'You picked your agent, and you were right: 93.0% beats Linear’s 90.0%.');
  assert.equal(text(L.scoresPayoff(null, won, 'agent')), 'Your agent got 93.0%. Linear got 90.0%. Nice.');
  assert.equal(text(L.scoresPayoff('linear', won, 'agent')), 'You picked Linear. It got 90.0%, but your agent won with 93.0%.');
});

test('guess replies and recap recall the guess', () => {
  assert.match(text(L.guessReply('agent', 'agent')), /Rooting for your agent/);
  assert.match(text(L.guessReply('linear', 'agent')), /the quiet one/);
  assert.match(text(L.guessReply('prompt', 'agent')), /the single call/);
  assert.match(text(L.guessReply('other_arm', 'agent')), /^Other arm\. Interesting pick!/);
  assert.equal(text(L.recapLine(null, arms, 'agent')), 'Want the full audit bundle for your team?');
  assert.match(text(L.recapLine('linear', arms, 'agent')), /^You guessed Linear, and you called it\./);
  assert.match(text(L.recapLine('agent', arms, 'agent')), /^You guessed your agent\. Want/);
});

test('arm order: simpler arms first, the agent last; guess button says My agent', () => {
  assert.deepEqual(L.armOrder(arms, 'agent').map((a) => a.name), ['linear', 'prompt', 'agent']);
  assert.equal(L.guessButtonLabel('agent', 'agent'), 'My agent');
  assert.equal(L.guessButtonLabel('linear', 'agent'), 'Linear');
});

test('checks from the real data: counts, failures and the binding one', () => {
  const s = L.summarizeChecks(story.verdict.checks);
  assert.equal(s.total, story.verdict.checks.length);
  assert.equal(s.passed, story.verdict.checks.filter((c) => c.passed).length);
  assert.equal(L.checksHeading(s), `${s.passed} of ${s.total} checks passed.`);
  assert.deepEqual(s.binding.map((c) => c.name), ['power_cap']);
  assert.match(text(L.checksLine(s)), /power_cap did not pass, and that sets the verdict to NEED_MORE_DATA/);
  const allPass = L.summarizeChecks(story.verdict.checks.map((c) => ({ ...c, passed: true })));
  assert.match(text(L.checksLine(allPass)), /Everything checks out/);
  assert.equal(L.checksLead(allPass), 'Every pre-registered check passed.');
});

test('rules line uses the data counts', () => {
  assert.equal(text(L.rulesLine(13, 3)), 'I froze 13 checks and 3 price tables before looking at a single answer.');
});

test('cost line: more, saves, same', () => {
  assert.equal(text(L.costLine(5, 0.38)), 'At $5.00 a mistake, your agent costs $0.38 more per case. Drag the slider to try your own number.');
  assert.match(text(L.costLine(5, -0.2)), /saves \$0\.20 per case/);
  assert.match(text(L.costLine(5, 0)), /costs the same per case/);
});

test('case scene: real case, and null case hides the replay', () => {
  const c = story.case as StoryCase;
  const steps = L.replaySteps(c).map(text);
  assert.equal(steps[0], `Linear’s shortlist puts ${L.labelText(c.shortlist[0])} first.`);
  assert.equal(steps[1], `The agent searches “${c.search_query}”.`);
  assert.equal(steps[2], `The agent answers ${L.labelText(c.agent)}.`);
  assert.equal(steps[3], `The right answer was ${L.labelText(c.gold)}.`);
  assert.equal(L.replaySteps({ ...c, search_query: null })[1][0], 'The agent answers without searching.');
  assert.equal(L.caseHeading(null), 'No case to replay in this export.');
  assert.match(text(L.caseLine(null)), /nothing to replay/);
});

test('recap: rehearsal never reads as the final verdict; sealed shows the verdict', () => {
  assert.equal(L.recapHeading(story), 'On DEV, the simpler model wins.');
  assert.match(L.recapLead(story, 'cloud_small'), /Linear is more accurate \(89\.9% vs 82\.4%\) and cheaper at every error cost/);
  assert.match(L.recapLead(story, 'cloud_small'), /DEV rehearsal/);
  assert.match(L.verdictTag(story), /not a verdict/);
  assert.equal(L.sourceBanner('rehearsal'), 'DEV rehearsal preview: not the sealed verdict');
  const sealed: Story = { ...story, source: 'sealed' };
  assert.equal(L.recapHeading(sealed), 'The verdict: we need more data.');
  assert.equal(L.recapHeading({ ...sealed, verdict: { ...sealed.verdict, verdict: 'USE_SIMPLER' } }), 'The verdict: use the simpler model.');
  assert.match(L.recapLead(sealed, 'cloud_small'), /Evidence level: REAL/);
  assert.equal(L.auditLine(story), 'commit eb25fd1 · model 46e0c10c · layer f5292dc4');
});

test('check limits come from the bounds; dataset labels read as words', () => {
  assert.equal(L.checkLimit({ low: 0.07, high: 0.13 }), '0.07 to 0.13');
  assert.equal(L.checkLimit({ low: 0.9, high: null }), '≥ 0.9');
  assert.equal(L.checkLimit({ low: null, high: 0.005 }), '≤ 0.005');
  assert.equal(L.checkLimit({ low: null, high: null }), null);
  assert.equal(L.labelText('order_physical_card'), 'order physical card');
  const c = { id: 'x', text: 't', gold: 'card_linking', linear: 'card_linking', shortlist: ['card_linking'], search_query: null, agent: 'order_physical_card' };
  assert.doesNotMatch(L.replaySteps(c).map(text).join(' '), /_/);
});
