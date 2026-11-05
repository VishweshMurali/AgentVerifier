import test from 'node:test';
import assert from 'node:assert/strict';
import { cheaperAcrossRange, costGap, defaultErrorCost, expectedCost, inferenceCost, sliderStep } from './cost.ts';
import { pct, signedPts, usd } from './format.ts';

const agent = { name: 'agent', accuracy: 0.8, inference_cost: { cheap: 0.001, dear: 0.5 } };
const linear = { name: 'linear', accuracy: 0.9, inference_cost: { cheap: 0, dear: 0 } };

const close = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-12, `${a} != ${b}`);

test('expected cost = inference cost + (1 - accuracy) * error cost', () => {
  close(expectedCost(agent, 'cheap', 5), 0.001 + 0.2 * 5);
  close(expectedCost(linear, 'cheap', 5), 0.1 * 5);
  close(expectedCost(agent, 'dear', 0), 0.5);
});

test('the price table changes only the inference part', () => {
  close(expectedCost(agent, 'dear', 10) - expectedCost(agent, 'cheap', 10), 0.5 - 0.001);
});

test('missing price table is an error, not a silent zero', () => {
  assert.throws(() => inferenceCost(agent, 'nope'), /no inference cost for price table "nope"/);
});

test('cost gap is agent minus simpler', () => {
  close(costGap(agent, linear, 'cheap', 5), 0.001 + 1 - 0.5);
  assert.ok(costGap(agent, linear, 'cheap', 5) > 0);
});

test('cheaper across the range checks both ends', () => {
  assert.equal(cheaperAcrossRange(agent, linear, 'cheap', [0.5, 20]), 'simpler');
  const better = { name: 'agent', accuracy: 0.95, inference_cost: { cheap: 0.4 } };
  // gap = 0.4 - 0.05 * e: positive at e = 1, negative at e = 20 -> crosses
  assert.equal(cheaperAcrossRange(better, linear, 'cheap', [1, 20]), 'crosses');
  assert.equal(cheaperAcrossRange(better, linear, 'cheap', [10, 20]), 'agent');
  assert.equal(cheaperAcrossRange(linear, linear, 'cheap', [1, 2]), 'equal');
});

test('the real data: agent costs more than linear at every error cost', () => {
  const realAgent = { name: 'agent', accuracy: 0.8242134062927496, inference_cost: { cloud_small: 0.0002052065663474692 } };
  const realLinear = { name: 'linear', accuracy: 0.899452804377565, inference_cost: { cloud_small: 0 } };
  close(expectedCost(realAgent, 'cloud_small', 5), 0.0002052065663474692 + (1 - 0.8242134062927496) * 5);
  assert.equal(cheaperAcrossRange(realAgent, realLinear, 'cloud_small', [0.5, 20]), 'simpler');
});

test('slider step and start stay inside the pre-registered range', () => {
  assert.equal(sliderStep([0.5, 20]), 0.5);
  assert.equal(sliderStep([0, 1]), 0.05);
  const start = defaultErrorCost([0.5, 20]);
  assert.equal(start, 5.5);
  assert.ok(start >= 0.5 && start <= 20);
  const s2 = defaultErrorCost([3, 4]);
  assert.ok(s2 >= 3 && s2 <= 4);
});

test('formatting: percent, signed points, dollars', () => {
  assert.equal(pct(0.899452804377565), '89.9%');
  assert.equal(signedPts(-0.07523939808481532), '−7.5');
  assert.equal(signedPts(0.031), '+3.1');
  assert.equal(signedPts(0.0001), '0.0');
  assert.equal(usd(0.8791), '$0.88');
  assert.equal(usd(0.000205), '$0.0002');
  assert.equal(usd(-0.5), '−$0.50');
  assert.equal(usd(0), '$0.00');
});
