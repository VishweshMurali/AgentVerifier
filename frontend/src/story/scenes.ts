// The 9 scenes of the locked design (design/frontend/SPEC.md): order, names, robot side and mood.
export type Mood = 'hello' | 'curious' | 'thinking' | 'nod' | 'sad' | 'point' | 'proud';
export type Side = 'left' | 'right';

export interface SceneDef {
  id: string;
  name: string;
  side: Side;
  mood: Mood;
}

export const SCENES: readonly SceneDef[] = [
  { id: 's1', name: 'The question', side: 'right', mood: 'hello' },
  { id: 's2', name: 'Contenders', side: 'left', mood: 'curious' },
  { id: 's3', name: 'Fair rules', side: 'right', mood: 'thinking' },
  { id: 's4', name: 'Scores', side: 'left', mood: 'thinking' },
  { id: 's5', name: 'Integrity', side: 'right', mood: 'nod' },
  { id: 's6', name: 'The reveal', side: 'left', mood: 'sad' },
  { id: 's7', name: 'Your call', side: 'right', mood: 'point' },
  { id: 's8', name: 'One case', side: 'left', mood: 'curious' },
  { id: 's9', name: 'Recap', side: 'right', mood: 'proud' },
];

export const SCENE_IDS: readonly string[] = SCENES.map((s) => s.id);

export function sceneById(id: string): SceneDef {
  return SCENES.find((s) => s.id === id) ?? SCENES[0];
}
