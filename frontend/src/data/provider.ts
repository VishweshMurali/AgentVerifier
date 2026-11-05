// Data providers: where the story's data comes from. Pure logic, no DOM, so node:test can run it.
import type { LiveStatus, Story } from './types.ts';

export interface DataProvider {
  /** Short name shown in errors, for example "static". */
  name: string;
  load(): Promise<Story>;
  /** Live project status, when the provider is connected to the local server. */
  status?(): Promise<LiveStatus>;
}

/** The part of fetch the providers use, so tests can pass a fake. */
export type FetchLike = (url: string) => Promise<{
  ok: boolean;
  status: number;
  statusText?: string;
  json(): Promise<unknown>;
}>;

export class StoryDataError extends Error {
  readonly problems: string[];
  constructor(message: string, problems: string[] = []) {
    super(problems.length ? `${message}\n- ${problems.join('\n- ')}` : message);
    this.name = 'StoryDataError';
    this.problems = problems;
  }
}

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isStr = (v: unknown): v is string => typeof v === 'string';

/**
 * Checks the minimal shape the page needs and throws one readable error listing every problem.
 * It does not recompute or change any value.
 */
export function validateStory(data: unknown): Story {
  const problems: string[] = [];
  const need = (ok: boolean, path: string, expected: string) => {
    if (!ok) problems.push(`"${path}" is missing or not ${expected}`);
    return ok;
  };

  if (!isObj(data)) throw new StoryDataError('story.json is not a JSON object');

  need(data.source === 'rehearsal' || data.source === 'sealed', 'source', '"rehearsal" or "sealed"');

  const v = data.verdict;
  if (need(isObj(v), 'verdict', 'an object') && isObj(v)) {
    need(isStr(v.verdict), 'verdict.verdict', 'a string');
    need(isStr(v.evidence_level), 'verdict.evidence_level', 'a string');
    need(isNum(v.accuracy_diff), 'verdict.accuracy_diff', 'a number');
    need(isNum(v.n_cases), 'verdict.n_cases', 'a number');
    need(isNum(v.ci_level), 'verdict.ci_level', 'a number');
    need(
      Array.isArray(v.accuracy_diff_ci) && v.accuracy_diff_ci.length === 2 && v.accuracy_diff_ci.every(isNum),
      'verdict.accuracy_diff_ci',
      'a pair of numbers',
    );
    const range = v.error_cost_range;
    if (
      need(
        Array.isArray(range) && range.length === 2 && range.every(isNum),
        'verdict.error_cost_range',
        'a pair of numbers',
      ) &&
      Array.isArray(range)
    ) {
      need((range[0] as number) < (range[1] as number), 'verdict.error_cost_range', 'an increasing range');
    }
    need(Array.isArray(v.reasons) && v.reasons.every(isStr), 'verdict.reasons', 'a list of strings');
    if (need(isObj(v.audit), 'verdict.audit', 'an object') && isObj(v.audit)) {
      for (const key of ['git_commit', 'model_digest', 'layer_fingerprint']) {
        need(isStr(v.audit[key]), `verdict.audit.${key}`, 'a string');
      }
    }
    if (need(Array.isArray(v.checks), 'verdict.checks', 'a list') && Array.isArray(v.checks)) {
      v.checks.forEach((c, i) => {
        if (!need(isObj(c), `verdict.checks[${i}]`, 'an object') || !isObj(c)) return;
        need(isStr(c.name), `verdict.checks[${i}].name`, 'a string');
        need(typeof c.passed === 'boolean', `verdict.checks[${i}].passed`, 'true or false');
        need(isStr(c.effect), `verdict.checks[${i}].effect`, 'a string');
      });
    }
    const armNames: string[] = [];
    if (need(Array.isArray(v.arms) && v.arms.length > 0, 'verdict.arms', 'a non-empty list') && Array.isArray(v.arms)) {
      v.arms.forEach((a, i) => {
        if (!need(isObj(a), `verdict.arms[${i}]`, 'an object') || !isObj(a)) return;
        if (need(isStr(a.name), `verdict.arms[${i}].name`, 'a string')) armNames.push(a.name as string);
        need(isNum(a.accuracy) && a.accuracy >= 0 && a.accuracy <= 1, `verdict.arms[${i}].accuracy`, 'a number from 0 to 1');
        need(isObj(a.inference_cost), `verdict.arms[${i}].inference_cost`, 'an object');
        if (isObj(a.inference_cost) && Array.isArray(data.price_tables)) {
          for (const t of data.price_tables) {
            if (isObj(t) && isStr(t.name)) {
              need(isNum(a.inference_cost[t.name]), `verdict.arms[${i}].inference_cost.${t.name}`, 'a number');
            }
          }
        }
      });
    }
    for (const key of ['agent_arm', 'simpler_arm']) {
      const name = v[key];
      need(isStr(name) && armNames.includes(name), `verdict.${key}`, 'the name of one of verdict.arms');
    }
  }

  const tables = data.price_tables;
  const tableNames: string[] = [];
  if (need(Array.isArray(tables) && tables.length > 0, 'price_tables', 'a non-empty list') && Array.isArray(tables)) {
    tables.forEach((t, i) => {
      if (!need(isObj(t), `price_tables[${i}]`, 'an object') || !isObj(t)) return;
      if (need(isStr(t.name), `price_tables[${i}].name`, 'a string')) tableNames.push(t.name as string);
    });
  }
  need(
    isStr(data.default_price_table) && tableNames.includes(data.default_price_table),
    'default_price_table',
    'the name of one of price_tables',
  );

  if (need(isObj(data.rules), 'rules', 'an object') && isObj(data.rules)) {
    need(isNum(data.rules.checks), 'rules.checks', 'a number');
    need(isNum(data.rules.price_tables), 'rules.price_tables', 'a number');
  }

  const c = data.case;
  if (c !== null) {
    if (need(isObj(c), 'case', 'an object or null') && isObj(c)) {
      for (const key of ['id', 'text', 'gold', 'linear', 'agent']) need(isStr(c[key]), `case.${key}`, 'a string');
      need(Array.isArray(c.shortlist) && c.shortlist.every(isStr), 'case.shortlist', 'a list of strings');
      need(c.search_query === null || isStr(c.search_query), 'case.search_query', 'a string or null');
    }
  }

  if (problems.length) throw new StoryDataError('story.json does not have the expected shape:', problems);
  return data as unknown as Story;
}

/** Reads the story from a static JSON file served next to the page (the default). */
export class StaticJsonProvider implements DataProvider {
  readonly name = 'static';
  private readonly url: string;
  private readonly fetchImpl: FetchLike;

  constructor(url: string, fetchImpl?: FetchLike) {
    this.url = url;
    this.fetchImpl = fetchImpl ?? ((u: string) => fetch(u));
  }

  async load(): Promise<Story> {
    return validateStory(await fetchJson(this.fetchImpl, this.url));
  }
}

/** GET a URL and parse JSON, turning every failure into one readable StoryDataError. */
export async function fetchJson(fetchImpl: FetchLike, url: string): Promise<unknown> {
  let res: Awaited<ReturnType<FetchLike>>;
  try {
    res = await fetchImpl(url);
  } catch (err) {
    throw new StoryDataError(`Could not reach ${url}: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (!res.ok) {
    throw new StoryDataError(`Could not load ${url}: HTTP ${res.status}${res.statusText ? ` ${res.statusText}` : ''}`);
  }
  try {
    return await res.json();
  } catch {
    throw new StoryDataError(`${url} is not valid JSON`);
  }
}

/**
 * Reads the story and the live status from the local server (server/, FastAPI). `base` is the
 * server's origin, or '' when the page is served by the server itself.
 */
export class ApiProvider implements DataProvider {
  readonly name = 'api';
  private readonly base: string;
  private readonly fetchImpl: FetchLike;

  constructor(base = '', fetchImpl?: FetchLike) {
    this.base = base.replace(/\/+$/, '');
    this.fetchImpl = fetchImpl ?? ((u: string) => fetch(u));
  }

  async load(): Promise<Story> {
    return validateStory(await fetchJson(this.fetchImpl, `${this.base}/api/story`));
  }

  async status(): Promise<LiveStatus> {
    const body = await fetchJson(this.fetchImpl, `${this.base}/api/status`);
    if (typeof body !== 'object' || body === null || !('sealed_run' in body)) {
      throw new StoryDataError('/api/status does not have the expected shape');
    }
    return body as LiveStatus;
  }
}
