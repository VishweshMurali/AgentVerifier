import assert from 'node:assert/strict';
import test from 'node:test';
import { birthState, makeBirthPoints, TUNE, TUNE_DEFAULTS, travelColour } from './birth.ts';

test('before the blast the sphere is whole and nothing has left it', () => {
  const B = birthState(0.5);
  assert.equal(B.e, 0);
  assert.equal(B.alive, 1);
  assert.equal(B.flash, 0);
});

test('the blast runs from its start for its length, then the graph is formed', () => {
  assert.equal(birthState(TUNE.burstAt).e, 0);
  assert.equal(birthState(TUNE.burstAt + TUNE.burstSpan).e, 1);
  assert.equal(birthState(TUNE.burstAt + TUNE.burstSpan + 1).e, 1);
  assert.equal(birthState(TUNE.burstAt + TUNE.burstSpan).alive, 0);
});

test('progress never goes backwards as the page scrolls on', () => {
  let last = birthState(0);
  for (let q = 0; q <= 4; q += 0.02) {
    const B = birthState(q);
    assert.ok(B.charge >= last.charge - 1e-9 && B.over >= last.over - 1e-9 && B.e >= last.e - 1e-9, `at ${q}`);
    last = B;
  }
});

test('the overcharge builds up before the blast and the flash is brief', () => {
  assert.ok(birthState(TUNE.burstAt - 0.1).over > birthState(TUNE.burstAt - 0.4).over);
  assert.ok(birthState(TUNE.burstAt + 0.01).flash > 0);
  assert.equal(birthState(TUNE.burstAt + 0.5).flash, 0);
});

test('a dot starts inside the sphere and ends exactly at its place in the graph', () => {
  const bp = makeBirthPoints(50), out = new Float32Array(3);
  for (let i = 0; i < 50; i++) {
    bp.place(i, 3, -2, 1, birthState(0), out, 0);
    assert.ok(Math.hypot(out[0], out[1], out[2]) <= TUNE.sphereSize * (1 + TUNE.chargeGrow) + 1e-6);
    bp.place(i, 3, -2, 1, birthState(TUNE.burstAt + TUNE.burstSpan), out, 0);
    assert.ok(Math.abs(out[0] - 3) < 1e-5 && Math.abs(out[1] + 2) < 1e-5 && Math.abs(out[2] - 1) < 1e-5, `dot ${i} lands`);
  }
});

test('a dot wears ice-blue in the sphere and its own colour once it has landed', () => {
  assert.equal(travelColour([10, 20, 30], 0), '186,230,253');
  assert.equal(travelColour([10, 20, 30], 1), '10,20,30');
});

test('the settings the owner chose on 2026-11-15 are the defaults', () => {
  assert.deepEqual(
    { ...TUNE_DEFAULTS },
    { sphereSize: 0.71, particles: 5000, density: 2.75, dotSize: 0.35, coreSize: 0.24, coreGlow: 0.2, chargeEnd: 2.2, chargeGrow: 0.7, neurons: 32, neuronSpeed: 3, overcharge: 0.55, squeeze: 0.6, burstAt: 1.46, burstSpan: 1.6, burstReach: 0.2, flash: 0.8, rays: 0.05, swirl: 2.5, trail: 0 },
  );
});
