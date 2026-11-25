// Words for the top strip and the example badge. Plain text logic only: the verdict word and the
// counts come from the engine's verdict, never from a ranking made here.

export interface StripInput {
  invalid: boolean;
  simplerName: string;
  agentName: string;
  simplerRight: number;
  simplerOf: number;
  agentRight: number;
  agentOf: number;
}

/** "On your kinds: Linear 18 of 23 right · agent 17 of 23", or null when there is nothing honest to count. */
export function stripCounts(s: StripInput): string | null {
  if (s.invalid || s.simplerOf <= 0 || s.agentOf <= 0) return null;
  return `On your kinds: ${s.simplerName} ${s.simplerRight} of ${s.simplerOf} right · ${s.agentName} ${s.agentRight} of ${s.agentOf}`;
}

export type BadgeKind = 'example' | 'mine';

export function badgeWord(kind: BadgeKind): string {
  return kind === 'example' ? 'EXAMPLE' : 'YOUR RUN';
}

/** What is happening next to the EXAMPLE badge: saved, running, stopped, or waiting for the reader. */
export function exampleState(o: { saved: boolean; state?: string; ownTitle?: string }): string {
  if (o.saved) return 'saved, not run live';
  if (o.state === 'running') return 'your test is running; your own result replaces this when it finishes';
  if (o.state === 'failed') return 'your test stopped, so this is still the example';
  return o.ownTitle ? `yours (${o.ownTitle}) appears here after you run it` : 'yours appears here after you run it';
}
