import test from 'node:test';
// Keyboard path to the robot's buttons and the header button's wording (source checks: the
// components are TSX, which node --test cannot load).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const page = readFileSync(new URL('./StoryPage.tsx', import.meta.url), 'utf8');
const robot = readFileSync(new URL('./Robot.tsx', import.meta.url), 'utf8');

test('the robot comes after <main> in the DOM, so Tab reaches the scene before its bubble', () => {
  assert.ok(page.indexOf('</main>') < page.indexOf('<Robot '), 'Robot is rendered after </main>');
});

test('every scene whose bubble has buttons ends with a focus-only button to reach them', () => {
  for (const id of ['s2', 's5', 's6', 's8', 's9']) assert.ok(page.includes(`{skipToRobot('${id}')}`), id);
  assert.match(page, /#robot-controls button, #robot-controls a/);
  assert.match(robot, /id="robot-controls"/);
});

test('the header button says what it does', () => {
  assert.match(page, /Try your error cost/);
  assert.doesNotMatch(page, /Verify an agent/);
});
