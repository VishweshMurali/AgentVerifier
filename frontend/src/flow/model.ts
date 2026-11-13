// The "test your agent" flow: the reader's choices, what is still missing, and the plain-code
// sums the page shows (test size, minutes, cost of one mistake, cost per month). Pure and
// deterministic; no AI writes or chooses anything here.
import type { Arm, PriceTable, Verdict } from '../data/types.ts';
import { expectedCost } from '../story/cost.ts';

export interface ModelDef {
  id: string;
  label: string;
  blurb: string;
  /** True for the arms that call a language model (they cost per message). */
  ai: boolean;
}

/** The ladder the reader can race, in the order the page lists it. The agent comes first. */
export const MODELS: readonly ModelDef[] = [
  { id: 'agent', label: 'Your agent', blurb: 'Searches examples, reads label notes, then decides', ai: true },
  { id: 'prompt', label: 'Single prompt', blurb: 'One AI call with a shortlist', ai: true },
  { id: 'majority', label: 'Most common label', blurb: 'The floor: always the top label', ai: false },
  { id: 'naive_bayes', label: 'Naive Bayes', blurb: 'Basic, free', ai: false },
  { id: 'linear', label: 'Linear', blurb: 'Basic, free', ai: false },
  { id: 'linear_svm', label: 'Linear SVM', blurb: 'Tier 1, free', ai: false },
  { id: 'random_forest', label: 'Random forest', blurb: 'Tier 2, free', ai: false },
  { id: 'boosting', label: 'Gradient boosting', blurb: 'Tier 2, free', ai: false },
];

export const AGENT = 'agent';

export function modelLabel(id: string): string {
  return MODELS.find((m) => m.id === id)?.label ?? id.replace(/[_-]+/g, ' ');
}

export interface Dataset {
  option: string;
  title: string;
  description: string;
  status: string;
  labels: number | null;
}

/** A dataset can run only once its split exists ("built in" today; "built" after a split). */
export function datasetReady(d: Pick<Dataset, 'status'>): boolean {
  return d.status === 'built in' || d.status === 'split built';
}

export interface SavedRun {
  plan: {
    scenario: string;
    gap: number;
    planned_cases: number;
    arms: string[];
    comparator: string;
    population: number;
    early_stopping: { looks: number[] };
    scope?: { labels: string[]; near_misses: string[] };
  };
  /** Per-arm accuracy on the reader's labels and false grabs; null for a full (unscoped) test. */
  scope: Record<string, ScopeScore> | null;
  verdict: Verdict;
}

export interface ScopeScore {
  in_scope_cases: number;
  in_scope_accuracy: number;
  other_cases: number;
  false_grab_rate: number;
}

export interface FlowData {
  datasets: Dataset[];
  labels: Record<string, string[]>;
  /** Paired cases planned per gap, from the project's exact McNemar power code. */
  plan_sizes: Record<string, number>;
  seconds_per_case: Record<string, number>;
  price_tables: PriceTable[];
  /** The saved run the static preview replays; null when the page talks to the local server. */
  saved_run: SavedRun | null;
}

export interface Choices {
  description: string;
  dataset: string | null;
  labels: readonly string[];
  arms: readonly string[];
  gap: number | null;
  volume: number | null;
  fixer: string | null;
  hourly: number | null;
  minutes: number | null;
  table: string | null;
}

export const EMPTY: Choices = {
  description: '',
  dataset: null,
  labels: [],
  arms: [],
  gap: null,
  volume: null,
  fixer: null,
  hourly: null,
  minutes: null,
  table: null,
};

export const FIXERS = ['Support agent', 'Specialist', 'Nobody, the customer waits'] as const;

/** A number typed into a field, or null while it is empty or not a usable number. */
export function parseAmount(text: string, min = 0): number | null {
  if (text.trim() === '') return null;
  const x = Number(text);
  return Number.isFinite(x) && x >= min ? x : null;
}

/** The first thing still missing, as [scene id, what to do], or null when the test can run. */
export function missing(c: Choices): [string, string] | null {
  if (!c.description.trim()) return ['f1', 'say what your agent does'];
  if (!c.dataset) return ['f2', 'pick a use case'];
  if (c.labels.length === 0) return ['f3', 'pick at least one label'];
  if (!c.arms.includes(AGENT)) return ['f4', 'include your agent'];
  if (!c.arms.some((a) => !MODELS.find((m) => m.id === a)?.ai)) return ['f4', 'add at least one free model to beat'];
  if (c.gap === null) return ['f5', 'pick how big a gain matters'];
  if (c.volume === null || c.volume < 1) return ['f6', 'enter messages a month'];
  if (!c.fixer) return ['f6', 'say who fixes a wrong answer'];
  if (c.hourly === null) return ['f6', 'enter their cost per hour'];
  if (c.minutes === null) return ['f6', 'enter minutes per fix'];
  if (!c.table) return ['f6', 'pick where the AI would run'];
  return null;
}

/** Cost of one wrong answer: hourly cost x minutes per fix / 60. */
export function mistakeCost(hourly: number, minutes: number): number {
  return (hourly * minutes) / 60;
}

export function gapKey(gap: number): string {
  return gap.toFixed(2);
}

/** Planned cases for a gap, or null when the project has no plan size for it. */
export function plannedCases(data: Pick<FlowData, 'plan_sizes'>, gap: number): number | null {
  return data.plan_sizes[gapKey(gap)] ?? null;
}

/** Most minutes of language-model calls for n cases with these arms (it can stop early). */
export function maxMinutes(data: Pick<FlowData, 'seconds_per_case'>, n: number, arms: readonly string[]): number {
  return (n * arms.reduce((s, a) => s + (data.seconds_per_case[a] ?? 0), 0)) / 60;
}

/** Expected cost per month: messages x (AI price per message + share wrong x cost of a mistake). */
export function monthlyCost(arm: Pick<Arm, 'name' | 'accuracy' | 'inference_cost'>, table: string, volume: number, mistake: number): number {
  return volume * expectedCost(arm, table, mistake);
}

/** Which look of the saved run the test stopped at (0-based), from the cases it used. */
export function stopLook(run: SavedRun): number {
  const looks = run.plan.early_stopping.looks;
  const i = looks.indexOf(run.verdict.n_cases);
  return i >= 0 ? i : looks.length - 1;
}

/** The saved run's check-ins rebuilt for the preview replay: only the cases and the final gain are known. */
export function replayLooks(run: SavedRun): number[] {
  return run.plan.early_stopping.looks.slice(0, stopLook(run) + 1);
}

/** The arms the reader picked that the saved run also raced, in the page's order. */
export function racedArms(run: SavedRun, arms: readonly string[]): string[] {
  const ran = new Set(run.verdict.arms.map((a) => a.name));
  return MODELS.map((m) => m.id).filter((id) => arms.includes(id) && ran.has(id));
}

/** Did the change touch what the test measures? Cost answers only re-price a finished run. */
export function changesTheTest(patch: Partial<Choices>): boolean {
  return ['dataset', 'labels', 'arms', 'gap'].some((k) => k in patch);
}

/** One check-in of a running quick test, as the server parses it from the test's own output. */
export interface LookPoint {
  n: number;
  gain: number;
  z_agent: number;
  z_ruled_out: number;
  bound: number;
}

/**
 * The repeated confidence interval at a check-in (agent minus the simpler model), rebuilt from the
 * printed z values exactly as sequential.Look.interval does: gain - c*se0 to gain + c*se1, with
 * se0 = gain / z_agent and se1 = (gain - gap) / z_ruled_out. Null when a z is 0 (no spread yet).
 */
export function lookInterval(p: LookPoint, gap: number): [number, number] | null {
  if (p.z_agent === 0 || p.z_ruled_out === 0) return null;
  const se0 = p.gain / p.z_agent;
  const se1 = (p.gain - gap) / p.z_ruled_out;
  if (!(se0 > 0) || !(se1 > 0)) return null;
  return [p.gain - p.bound * se0, p.gain + p.bound * se1];
}

/** Accuracy of an arm as the page shows it: on the reader's labels when scoped, else on the test. */
export function shownAccuracy(run: SavedRun, arm: string): number | null {
  if (run.scope) return run.scope[arm]?.in_scope_accuracy ?? null;
  return run.verdict.arms.find((a) => a.name === arm)?.accuracy ?? null;
}

/** True when the test stopped before its planned size. */
export function stoppedEarly(run: SavedRun): boolean {
  return run.verdict.n_cases < run.plan.planned_cases;
}
