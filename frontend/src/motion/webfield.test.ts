import test from 'node:test';
import assert from 'node:assert/strict';
import { matchLabels } from './webfield.ts';

const lib = [
  { text: 'i am so mad right now', label: 'anger' },
  { text: 'what a lovely day', label: 'joy' },
  { text: 'i feel low today', label: 'sadness' },
  { text: 'switch the kitchen light off', label: 'iot_hue_lightoff' },
  { text: 'make the lights brighter', label: 'iot_hue_lightup' },
  { text: 'put on some songs', label: 'play_music' },
];

test('everyday words reach the dataset’s words even after stemming', () => {
  // "furious" stems to "furiou": the synonym table is looked up by stemmed keys
  assert.equal(matchLabels('customers who are furious', lib)[0], 'anger');
  assert.equal(matchLabels('detects angry customers', lib)[0], 'anger');
});

test('two asks in one sentence find both kinds', () => {
  const got = matchLabels('turn off the lights and play some jazz', lib);
  assert.ok(got.includes('iot_hue_lightoff') && got.includes('play_music'), got.join(','));
  assert.ok(got.indexOf('iot_hue_lightoff') < got.indexOf('iot_hue_lightup') || !got.includes('iot_hue_lightup'));
});
