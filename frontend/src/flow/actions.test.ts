import test from 'node:test';
import assert from 'node:assert/strict';
import { chatActions, hasActions } from './actions.ts';

test('volumes in everyday words', () => {
  assert.equal(chatActions('We get 50,000 messages a month').volume, 50000);
  assert.equal(chatActions('about 50k tickets per month').volume, 50000);
  assert.equal(chatActions('2,000 a day').volume, 60000);
  assert.equal(chatActions('1.2 million requests').volume, 1200000);
});

test('the cost of a mistake, only when the words say it is a mistake', () => {
  assert.equal(chatActions('What if a mistake costs $5?').mistake, 5);
  assert.equal(chatActions('each error costs 12.5 dollars').mistake, 12.5);
  assert.equal(chatActions('$20 per wrong answer').mistake, 20);
  assert.equal(chatActions('what does it cost a month?').mistake, undefined);
  assert.equal(chatActions('our budget is $500').mistake, undefined);
});

test('what the agent does becomes the description', () => {
  assert.equal(chatActions('my agent handles refunds and lost cards').describe, 'handles refunds and lost cards');
  assert.equal(chatActions('It answers questions about card delivery.').describe, 'answers questions about card delivery');
  assert.equal(chatActions('which use case fits me?').describe, undefined);
});

test('difference and where the AI runs', () => {
  assert.equal(chatActions('look for a big difference').gap, 0.2);
  assert.equal(chatActions('I care about a small gap').gap, 0.05);
  assert.equal(chatActions('we run it locally').table, 'local_amortized');
  assert.equal(chatActions('it uses a large cloud model').table, 'cloud_large');
});

test('a plain question changes nothing', () => {
  assert.equal(hasActions(chatActions('How sure is this?')), false);
  assert.equal(hasActions(chatActions('Which messages did my agent get wrong?')), false);
});
