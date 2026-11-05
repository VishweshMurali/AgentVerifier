// The robot's lines and the page's result sentences: fixed templates, chosen deterministically
// from the data. No text here is written or chosen by an AI.
import type { Arm, Check, Story, StoryCase, StorySource } from '../data/types.ts';
import { cheaperAcrossRange, type CostArm } from './cost.ts';
import { count, pct, signedPts, tableLabel, usd } from './format.ts';
import type { Mood } from './scenes.ts';

/** Rich text: plain strings and bold runs; tone colours a run as an arm. */
export type Segment = string | { strong: string; tone?: 'agent' | 'linear' };
export type Rich = Segment[];

export function plain(rich: Rich): string {
  return rich.map((s) => (typeof s === 'string' ? s : s.strong)).join('');
}

const KNOWN: Record<string, { label: string; blurb: string }> = {
  linear: { label: 'Linear', blurb: 'no LLM' },
  prompt: { label: 'Prompt', blurb: 'one LLM call' },
};

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Display name of an arm. The agent is "your agent" (capitalised at the start of a sentence). */
export function armLabel(name: string, agentArm: string, start = false): string {
  if (name === agentArm) return start ? 'Your agent' : 'your agent';
  return KNOWN[name]?.label ?? capitalize(name.replace(/[_-]+/g, ' '));
}

/** Label on the guess buttons: the reader's own agent is "My agent". */
export function guessButtonLabel(name: string, agentArm: string): string {
  return name === agentArm ? 'My agent' : armLabel(name, agentArm, true);
}

/** One-phrase description of an arm for the contenders scene, or null when unknown. */
export function armBlurb(name: string, agentArm: string): string | null {
  if (name === agentArm) return 'searches then decides';
  return KNOWN[name]?.blurb ?? null;
}

/** Simpler arms first (linear, prompt, then any others by name), the agent last. */
export function armOrder<T extends { name: string }>(arms: readonly T[], agentArm: string): T[] {
  const rank = (n: string) => (n === agentArm ? 3 : n === 'linear' ? 0 : n === 'prompt' ? 1 : 2);
  return [...arms].sort((a, b) => rank(a.name) - rank(b.name) || a.name.localeCompare(b.name));
}

/** Arms by accuracy, best first (ties broken by name so the order never changes). */
export function ranking<T extends { name: string; accuracy: number }>(arms: readonly T[]): T[] {
  return [...arms].sort((a, b) => b.accuracy - a.accuracy || a.name.localeCompare(b.name));
}

// ---------- scene 1: the question ----------
export function helloLine(): Rich {
  return ["Hi! I'm your verifier. Scroll down and I'll fly you through your agent's results."];
}

export function sourceBanner(source: StorySource): string {
  return source === 'rehearsal'
    ? 'DEV rehearsal preview: not the sealed verdict'
    : 'Sealed test: final verdict';
}

export function introLead(source: StorySource): string {
  const base = 'A fair race against simpler models on the same cases. Plain code decides, never an AI.';
  return source === 'rehearsal'
    ? `${base} You are looking at a DEV rehearsal preview; the sealed test has not been opened.`
    : base;
}

// ---------- scene 2: contenders ----------
export function contendersHeading(armCount: number, nCases: number): string {
  const words = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];
  const n = words[armCount] ?? String(armCount);
  return `${n} contenders. Same ${count(nCases)} cases.`;
}

export function contendersLine(): Rich {
  return ['Quick question: which one do you think wins?'];
}

export function guessReply(guess: string, agentArm: string): Rich {
  if (guess === agentArm) return ['Rooting for your agent, nice. Keep scrolling and we’ll see.'];
  const label = armLabel(guess, agentArm, true);
  if (guess === 'linear') return [`${label}, the quiet one. Bold pick! Keep scrolling and we’ll see.`];
  if (guess === 'prompt') return [`${label}, the single call. Interesting! Keep scrolling and we’ll see.`];
  return [`${label}. Interesting pick! Keep scrolling and we’ll see.`];
}

// ---------- scene 3: fair rules ----------
export function rulesLine(checks: number, priceTables: number): Rich {
  const c = checks === 1 ? 'check' : 'checks';
  const t = priceTables === 1 ? 'price table' : 'price tables';
  return [`I froze ${checks} ${c} and ${priceTables} ${t} before looking at a single answer.`];
}

// ---------- scene 4: scores ----------
type Scored = Pick<Arm, 'name' | 'accuracy'>;

function possessive(label: string): string {
  return label.endsWith('s') ? `${label}’` : `${label}’s`;
}

/** Pays off the reader's guess with the real accuracies (or comments on them with no guess). */
export function scoresPayoff(guess: string | null, arms: readonly Scored[], agentArm: string): Rich {
  const ranked = ranking(arms);
  const winner = ranked[0];
  const runnerUp = ranked[1];
  const label = (a: Scored, start = false) => armLabel(a.name, agentArm, start);
  const tie = runnerUp !== undefined && pct(runnerUp.accuracy) === pct(winner.accuracy);

  if (guess === null) {
    const agent = arms.find((a) => a.name === agentArm);
    if (!agent || !runnerUp) return [`${label(winner, true)} got ${pct(winner.accuracy)}.`];
    if (agent.name === winner.name) {
      return [`Your agent got ${pct(agent.accuracy)}. ${label(runnerUp, true)} got ${pct(runnerUp.accuracy)}. Nice.`];
    }
    return [`${label(winner, true)} got ${pct(winner.accuracy)}. Your agent got ${pct(agent.accuracy)}. Hmm.`];
  }

  const picked = arms.find((a) => a.name === guess);
  if (!picked) return [`${label(winner, true)} won with ${pct(winner.accuracy)}.`];
  const pickedLabel = picked.name === agentArm ? 'your agent' : label(picked, true);

  if (tie && (picked.name === winner.name || picked.name === runnerUp.name)) {
    return [`You picked ${pickedLabel}, and it tied for first at ${pct(winner.accuracy)}.`];
  }
  if (picked.name === winner.name && runnerUp) {
    return [
      `You picked ${pickedLabel}, and you were right: ${pct(winner.accuracy)} beats ${possessive(label(runnerUp))} ${pct(runnerUp.accuracy)}.`,
    ];
  }
  if (picked.name === winner.name) return [`You picked ${pickedLabel}, and you were right: ${pct(winner.accuracy)}.`];
  if (picked.name === ranked[ranked.length - 1].name && ranked.length > 2) {
    return [
      `You picked ${pickedLabel}, but it came last at ${pct(picked.accuracy)}. ${label(winner, true)} won with ${pct(winner.accuracy)}.`,
    ];
  }
  return [
    `You picked ${pickedLabel}. It got ${pct(picked.accuracy)}, but ${label(winner)} won with ${pct(winner.accuracy)}.`,
  ];
}

// ---------- scene 5: integrity ----------
export interface ChecksSummary {
  passed: number;
  total: number;
  failed: Check[];
  /** Failed checks whose effect changes the verdict (anything but REPORT). */
  binding: Check[];
}

export function summarizeChecks(checks: readonly Check[]): ChecksSummary {
  const failed = checks.filter((c) => !c.passed);
  return {
    passed: checks.length - failed.length,
    total: checks.length,
    failed,
    binding: failed.filter((c) => c.effect !== 'REPORT'),
  };
}

function listNames(names: string[]): string {
  if (names.length <= 1) return names.join('');
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

export function checksHeading(s: ChecksSummary): string {
  return `${s.passed} of ${s.total} checks passed.`;
}

export function checksLead(s: ChecksSummary): string {
  if (s.failed.length === 0) return 'Every pre-registered check passed.';
  const notes = s.failed.length - s.binding.length;
  const parts = [`Did not pass: ${listNames(s.failed.map((c) => c.name))}.`];
  if (s.binding.length === 0) parts.push('None of them changes the verdict; they are reported.');
  else if (notes > 0) parts.push(`${notes} ${notes === 1 ? 'is' : 'are'} only reported.`);
  return parts.join(' ');
}

export function checksLine(s: ChecksSummary): Rich {
  if (s.failed.length === 0) return [`Everything checks out: all ${s.total} checks passed.`];
  if (s.binding.length === 0) {
    return [
      'Everything that binds the verdict checks out. ',
      { strong: String(s.failed.length) },
      ` ${s.failed.length === 1 ? 'note is' : 'notes are'} only reported.`,
    ];
  }
  const out: Rich = ['Heads up: '];
  s.binding.forEach((c, i) => {
    if (i > 0) out.push(i === s.binding.length - 1 ? ' and ' : ', ');
    out.push({ strong: c.name });
  });
  const effects = [...new Set(s.binding.map((c) => c.effect))];
  out.push(` did not pass, and that sets the verdict to ${effects.join(' or ')}.`);
  return out;
}

export function checksToggleLabel(open: boolean): string {
  return open ? 'Hide checks' : 'Show all checks';
}

// ---------- scene 6: the reveal ----------
export type RevealState = 'ahead' | 'behind' | 'even';

/** Whether the agent is ahead of the best simpler arm, judged at the precision shown (0.1 pts). */
export function revealState(accuracyDiff: number): RevealState {
  const v = Number((accuracyDiff * 100).toFixed(1));
  return v > 0 ? 'ahead' : v < 0 ? 'behind' : 'even';
}

export function revealMood(accuracyDiff: number): Mood {
  const s = revealState(accuracyDiff);
  return s === 'behind' ? 'sad' : s === 'ahead' ? 'proud' : 'thinking';
}

export function revealLine(accuracyDiff: number): Rich {
  const s = revealState(accuracyDiff);
  if (s === 'behind') return ['Oh. That’s the gap. Want to see what it costs you?'];
  if (s === 'ahead') return ['Your agent is ahead! Want to see what that’s worth?'];
  return ['Dead even. Want to see what it costs you?'];
}

export function revealNumber(accuracyDiff: number): string {
  return signedPts(accuracyDiff);
}

export function revealLead(story: Story): string {
  const v = story.verdict;
  const simpler = armLabel(v.simpler_arm, v.agent_arm);
  const s = revealState(v.accuracy_diff);
  const first =
    s === 'behind'
      ? `Your agent trails ${simpler}, the best simpler model.`
      : s === 'ahead'
        ? `Your agent leads ${simpler}, the best simpler model.`
        : `Your agent and ${simpler} are level.`;
  if (story.source === 'rehearsal') {
    return `${first} This is the DEV rehearsal preview; the sealed test decides.`;
  }
  const [lo, hi] = v.accuracy_diff_ci;
  return `${first} ${Math.round(v.ci_level * 100)}% CI: ${signedPts(lo)} to ${signedPts(hi)} pts.`;
}

// ---------- scene 7: your call ----------
export function costLine(errorCost: number, gap: number): Rich {
  const tail = ' Drag the slider to try your own number.';
  const at = `At ${usd(errorCost)} a mistake, your agent `;
  if (gap > 0) return [at + 'costs ', { strong: usd(gap) }, ' more per case.' + tail];
  if (gap < 0) return [at + 'saves ', { strong: usd(-gap) }, ' per case.' + tail];
  return [at + 'costs the same per case.' + tail];
}

// ---------- scene 8: one case ----------
export function caseHeading(c: StoryCase | null): string {
  return c ? `“${c.text}”` : 'No case to replay in this export.';
}

export function caseLead(c: StoryCase | null): string {
  if (!c) return 'The data file has no disagreement case, so there is no trace to show here.';
  if (c.linear === c.gold) return 'Linear had it right. The agent picked another.';
  return 'The agent picked a different answer from the right one.';
}

export function caseLine(c: StoryCase | null): Rich {
  return c ? ['Want to watch how it went wrong?'] : ['No case came with this data, so there’s nothing to replay.'];
}

export function replayToggleLabel(open: boolean): string {
  return open ? 'Hide the replay' : 'Replay it';
}

export function replaySteps(c: StoryCase): Rich[] {
  const steps: Rich[] = [];
  if (c.shortlist.length > 0) {
    steps.push(['Linear’s shortlist puts ', { strong: c.shortlist[0] }, ' first.']);
  }
  steps.push(c.search_query ? [`The agent searches “${c.search_query}”.`] : ['The agent answers without searching.']);
  steps.push(['The agent answers ', { strong: c.agent, tone: 'agent' }, '.']);
  steps.push(['The right answer was ', { strong: c.gold }, '.']);
  return steps;
}

// ---------- scene 9: recap ----------
const VERDICT_HEADINGS: Record<string, string> = {
  DEPLOY_AGENT: 'The verdict: deploy your agent.',
  USE_SIMPLER: 'The verdict: use the simpler model.',
  NEED_MORE_DATA: 'The verdict: we need more data.',
  INVALID: 'The verdict: this run is invalid.',
};

export function recapHeading(story: Story): string {
  const v = story.verdict;
  if (story.source === 'sealed') return VERDICT_HEADINGS[v.verdict] ?? `The verdict: ${v.verdict}.`;
  const [lo, hi] = v.accuracy_diff_ci;
  if (hi < 0) return 'On DEV, the simpler model wins.';
  if (lo > 0) return 'On DEV, your agent wins.';
  return 'On DEV, it is too close to call.';
}

export function recapLead(story: Story, table: string): string {
  const v = story.verdict;
  const agent = v.arms.find((a) => a.name === v.agent_arm) as CostArm;
  const simpler = v.arms.find((a) => a.name === v.simpler_arm) as CostArm;
  const sLabel = armLabel(simpler.name, v.agent_arm, true);
  const accs = `(${pct(simpler.accuracy)} vs ${pct(agent.accuracy)})`;
  const s = revealState(v.accuracy_diff);
  const accuracy =
    s === 'behind'
      ? `${sLabel} is more accurate ${accs}`
      : s === 'ahead'
        ? `${sLabel} is less accurate ${accs}`
        : `${sLabel} is as accurate ${accs}`;
  const range = v.error_cost_range;
  const span = `from ${usd(range[0])} to ${usd(range[1])} a mistake (${tableLabel(table)} prices)`;
  const who = cheaperAcrossRange(agent, simpler, table, range);
  const costs =
    who === 'simpler'
      ? ` and cheaper at every error cost ${span}.`
      : who === 'agent'
        ? `, but your agent is cheaper at every error cost ${span}.`
        : who === 'equal'
          ? `, and both cost the same ${span}.`
          : `; which one is cheaper depends on the error cost ${span}.`;
  const close =
    story.source === 'rehearsal'
      ? ' This is a DEV rehearsal: the sealed test gives the final verdict.'
      : ` Evidence level: ${v.evidence_level}.`;
  return accuracy + costs + close;
}

export function recapLine(guess: string | null, arms: readonly Scored[], agentArm: string): Rich {
  const ask = 'Want the full audit bundle for your team?';
  if (guess === null) return [ask];
  const winner = ranking(arms)[0];
  const label = guess === agentArm ? 'your agent' : armLabel(guess, agentArm, true);
  return guess === winner.name ? [`You guessed ${label}, and you called it. ${ask}`] : [`You guessed ${label}. ${ask}`];
}

export function verdictTag(story: Story): string {
  const v = story.verdict;
  return story.source === 'rehearsal'
    ? `rehearsal outcome (not a verdict): ${v.verdict} · evidence ${v.evidence_level}`
    : `verdict ${v.verdict} · evidence ${v.evidence_level}`;
}

export function auditLine(story: Story): string {
  const a = story.verdict.audit;
  return `commit ${a.git_commit.slice(0, 7)} · model ${a.model_digest.slice(0, 8)} · layer ${a.layer_fingerprint.slice(0, 8)}`;
}

export function downloadReply(filename: string): Rich {
  return [
    'Saved ',
    { strong: filename },
    ': the verdict, its reasons, every check and the audit hashes. The predictions themselves stay with the run’s results.',
  ];
}
