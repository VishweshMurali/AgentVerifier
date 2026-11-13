// The robot's lines and the result sentences of the "test your agent" flow: fixed templates,
// chosen deterministically from the reader's choices and the engine's verdict. No AI text.
import type { Verdict } from '../data/types.ts';
import { count, pct, usd } from '../story/format.ts';
import type { Rich } from '../story/lines.ts';
import { AGENT, modelLabel, shownAccuracy, type SavedRun } from './model.ts';

export function helloLine(description: string): Rich {
  return description.trim()
    ? ['Got it. Let’s see if ', { strong: 'your agent', tone: 'agent' }, ' beats simpler models at that.']
    : ['Hi! Tell me what your agent does. I’ll race it against ', { strong: 'simpler models', tone: 'linear' }, '.'];
}

export function useCaseLine(title: string | null): Rich {
  return title ? ['We’ll test on ', { strong: title }, '.'] : ['Which of these looks most like your agent’s job?'];
}

export function notSplitLine(title: string): Rich {
  return [
    { strong: title },
    ' needs a one-time split first. It locks away a hidden test part for good, so it waits for the owner’s go-ahead.',
  ];
}

export function scopeLine(picked: number, total: number): Rich {
  if (picked === 0) return ['Which messages are your agent’s? Everything else counts as ', { strong: '“not mine”' }, '.'];
  if (picked === total) return ['All ', { strong: String(total) }, ' labels: no scoping, the full test.'];
  return [
    { strong: `${picked} label${picked === 1 ? '' : 's'}` },
    ' picked. I’ll also check how often each model wrongly grabs other messages.',
  ];
}

export function contendersLine(arms: readonly string[], hasFree: boolean): Rich {
  if (!arms.includes(AGENT)) return ['Who races? Tick ', { strong: 'your agent', tone: 'agent' }, ' plus at least one free model.'];
  if (!hasFree) return ['Now add at least one ', { strong: 'free model', tone: 'linear' }, ' for your agent to beat.'];
  return [{ strong: `${arms.length} contenders` }, '. Free ones run locally and cost nothing.'];
}

export function gapLine(cases: number | null, minutes: number | null): Rich {
  if (cases === null || minutes === null) return ['How big a win would make the agent worth it? Smaller wins need longer tests.'];
  return ['Up to ', { strong: `${count(cases)} cases` }, `, about ${minutesText(minutes)} at most. It can stop early.`];
}

export function minutesText(minutes: number): string {
  if (minutes < 1) return 'under a minute';
  const m = Math.round(minutes);
  return `${m} minute${m === 1 ? '' : 's'}`;
}

export function costLine(mistake: number | null): Rich {
  return mistake === null
    ? ['Now the money. What does one wrong answer cost you?']
    : ['One wrong answer costs you about ', { strong: usd(mistake) }, '.'];
}

export function raceLine(state: 'blocked' | 'ready' | 'running' | 'done', todo: string | null, look = 0, early = true): Rich {
  if (state === 'blocked') return ['Almost. Still needed: ', { strong: todo ?? '' }, '.'];
  if (state === 'ready') return ['All set. Hit ', { strong: 'Run' }, ' and I’ll keep you posted.'];
  if (state === 'running') return look < 0 ? ['Warming up the models…'] : ['Check-in ', { strong: `${look + 1} of 5` }, '…'];
  return early ? ['Done. The answer was clear early, so I stopped.'] : ['Done. All five check-ins ran.'];
}

/** Big word on the verdict scene, straight from the engine's verdict (never from a ranking). */
export function verdictWord(v: Pick<Verdict, 'verdict' | 'simpler_arm'>): string {
  switch (v.verdict) {
    case 'DEPLOY_AGENT':
      return 'Deploy your agent';
    case 'USE_SIMPLER':
      return `Use ${modelLabel(v.simpler_arm)}`;
    case 'NEED_MORE_DATA':
      return 'Not clear yet';
    default:
      return 'Test not valid';
  }
}

export function verdictLead(v: Pick<Verdict, 'verdict' | 'simpler_arm' | 'accuracy_diff' | 'n_cases'>, description: string, planned: number): string {
  const what = description.trim() ? `For “${description.trim()}”: ` : '';
  const pts = Math.abs(v.accuracy_diff * 100).toFixed(1);
  const used = v.n_cases < planned ? ` Stopped early at ${count(v.n_cases)} of ${count(planned)} cases.` : '';
  switch (v.verdict) {
    case 'DEPLOY_AGENT':
      return `${what}your agent is ${pts} points more accurate than ${modelLabel(v.simpler_arm)}, enough to matter.${used}`;
    case 'USE_SIMPLER':
      return `${what}a big enough gain from your agent is ruled out (it was ${pts} points ${v.accuracy_diff < 0 ? 'behind' : 'ahead'}), so ${modelLabel(v.simpler_arm)} does the job.${used}`;
    case 'NEED_MORE_DATA':
      return `${what}the two are too close to call on this many cases. No winner is named.${used}`;
    default:
      return `${what}a check failed, so this run can’t pick a winner.`;
  }
}

export function verdictLine(v: Pick<Verdict, 'verdict'>): Rich {
  switch (v.verdict) {
    case 'DEPLOY_AGENT':
      return ['Your agent earns its place. ', { strong: 'Deploy it', tone: 'agent' }, '.'];
    case 'USE_SIMPLER':
      return ['You don’t need the agent here. ', { strong: 'The simpler model', tone: 'linear' }, ' wins.'];
    case 'NEED_MORE_DATA':
      return ['Too close to call. A bigger test, or a bigger gain, would settle it.'];
    default:
      return ['Something in the run broke a rule, so no winner.'];
  }
}

export function scoresLine(run: SavedRun, arms: readonly string[]): Rich {
  const acc = (a: string) => shownAccuracy(run, a);
  const agent = acc(AGENT);
  const best = arms
    .filter((a) => a !== AGENT && acc(a) !== null)
    .sort((a, b) => (acc(b) as number) - (acc(a) as number))[0];
  if (agent === null || !best) return ['Here’s how each one did.'];
  const where = run.scope ? ' of your messages' : '';
  return [
    { strong: modelLabel(best), tone: 'linear' },
    ` got ${pct(acc(best) as number, 0)}${where} right. `,
    { strong: 'Your agent', tone: 'agent' },
    ` got ${pct(agent, 0)}.`,
  ];
}

export function failedLine(reason: string): Rich {
  return ['The test stopped: ', { strong: reason }, '.'];
}
