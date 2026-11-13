// The 9 scenes of the "test your agent" flow. Same rhythm as the story (one screen each, robot
// alternating sides, starting on the right), so the robot's flight and the ribbon work unchanged.
import type { Beats } from '../motion/ribbon.ts';
import type { SceneDef } from '../story/scenes.ts';

export const FLOW_SCENES: readonly SceneDef[] = [
  { id: 'f1', name: 'Your agent', side: 'right', mood: 'hello' },
  { id: 'f2', name: 'Use case', side: 'left', mood: 'curious' },
  { id: 'f3', name: 'Scope', side: 'right', mood: 'thinking' },
  { id: 'f4', name: 'Contenders', side: 'left', mood: 'curious' },
  { id: 'f5', name: 'Fair test', side: 'right', mood: 'thinking' },
  { id: 'f6', name: 'Your costs', side: 'left', mood: 'point' },
  { id: 'f7', name: 'The race', side: 'right', mood: 'nod' },
  { id: 'f8', name: 'Scores', side: 'left', mood: 'thinking' },
  { id: 'f9', name: 'The verdict', side: 'right', mood: 'proud' },
];

export const FLOW_IDS: readonly string[] = FLOW_SCENES.map((s) => s.id);

/** The ribbon splits at the contenders, charts the scores, and moves the agent lane at the verdict. */
export const FLOW_BEATS: Beats = { split: 3, chart: 7, reveal: 8, rejoin: 10 };

export function flowScene(id: string): SceneDef {
  return FLOW_SCENES.find((s) => s.id === id) ?? FLOW_SCENES[0];
}
