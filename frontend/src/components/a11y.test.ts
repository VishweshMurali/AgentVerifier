import test from 'node:test';
// The components are TSX (not loadable by node --test), so these checks read their source.
// They pin the accessibility fixes: valid roles, a live region without controls, focusable scenes.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const src = (path: string) => readFileSync(new URL(path, import.meta.url), 'utf8');
const page = src('./StoryPage.tsx');
const robot = src('./Robot.tsx');
const flight = src('../motion/flight.ts');

test('no invalid role="text" and no aria-label on generic <b> elements', () => {
  assert.doesNotMatch(page, /role="text"/);
  assert.doesNotMatch(page, /<b aria-label/);
  assert.match(page, /role="heading" aria-level=\{2\}/);
});

test('the robot announces only its line; the controls sit outside the live region', () => {
  const status = robot.match(/<div className="say-line" role="status" aria-live="polite">([\s\S]*?)<\/div>/);
  assert.ok(status, 'say-line live region exists');
  assert.match(status[1], /\{line\}/);
  assert.doesNotMatch(status[1], /controls/);
  assert.doesNotMatch(robot, /className="say" role="status"/);
});

test('a focused bubble control is never hidden while the robot flies', () => {
  assert.match(flight, /p\.say\.contains\(document\.activeElement\)/);
});

test('scene links move focus and the address to the scene, like a normal jump link', () => {
  assert.equal((page.match(/<section id="s\d" tabIndex=\{-1\}/g) ?? []).length, 9);
  assert.match(flight, /history\.pushState\(null, '', `#\$\{sec\.id\}`\)/);
  assert.match(flight, /sec\.focus\(\{ preventScroll: true \}\)/);
});
