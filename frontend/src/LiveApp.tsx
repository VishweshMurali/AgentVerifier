import { useCallback, useEffect, useRef, useState } from 'react';
import { MockPage, type LiveControl, type MockRun } from './components/MockPage.tsx';
import { caseCard, type AgentCheck, type MyAgent, type OwnUpload, type UseCase } from './components/UseCases.tsx';
import type { Arm, PriceTable, Verdict } from './data/types.ts';
import type { ScopeScore } from './flow/model.ts';
import { failureReason } from './flow/run.ts';

/**
 * The main page: the six-screen flow on live data. With the local server (VITE_DATA_PROVIDER=api)
 * the reader picks one of the ready use cases (or adds their own data), the message graph shows
 * that dataset's DEV sample, and "Run my test" starts a real quick test whose result replaces the
 * example. The static build shows the saved example only.
 */
// the free models plus the agent (the single-prompt arm is left out to keep a test short)
const ARMS = ['majority', 'linear', 'naive_bayes', 'linear_svm', 'random_forest', 'boosting', 'agent'];

interface FlowChoices {
  datasets: { option: string; title: string; status: string; own?: boolean }[];
  labels: Record<string, string[]>;
  price_tables: PriceTable[];
  plan_sizes: Record<string, number>;
  eval_sizes?: Record<string, number>;
  speeds?: Record<string, { models_saved: boolean; fit_seconds: number | null; agent_seconds_per_case: number | null }>;
}
interface Result {
  plan: MockRun['plan'] & { scenario?: string; scope?: { labels?: string[]; near_misses?: string[] } | null };
  scope: Record<string, ScopeScore> | null;
  verdict: Verdict;
  messages: MockRun['messages'];
}
type Library = MockRun['library'];

/** A finished quick test, shaped like the example the page already knows how to show. */
export function toRun(r: Result, flow: FlowChoices, library: Library, dataset: string): MockRun {
  // unscoped test: every label is the agent's, so "in scope" accuracy is plain accuracy
  const scope =
    r.scope ??
    Object.fromEntries(
      r.verdict.arms.map((a: Arm) => [a.name, { in_scope_cases: a.n, in_scope_accuracy: a.accuracy, other_cases: 0, false_grab_rate: 0 } as ScopeScore]),
    );
  return {
    labels: flow.labels[dataset] ?? [],
    plan: r.plan,
    scope,
    verdict: r.verdict,
    stream: r.messages.map((m) => ({ agent: m.agent_ok, simpler: m.simpler_ok })),
    price_tables: flow.price_tables,
    scope_labels: r.plan.scope?.labels ?? flow.labels[dataset] ?? [],
    library,
    messages: r.messages,
  };
}

export function LiveApp() {
  const api = import.meta.env.VITE_DATA_PROVIDER === 'api' ? (import.meta.env.VITE_API_URL ?? '') : null;
  const [example, setExample] = useState<MockRun | null>(null);
  const [flow, setFlow] = useState<FlowChoices | null>(null);
  const [libraries, setLibraries] = useState<Record<string, Library>>({});
  const [chosen, setChosen] = useState<string | null>(null);
  const [mine, setMine] = useState<MockRun | null>(null);
  const [state, setState] = useState<LiveControl['state']>('idle');
  const [progress, setProgress] = useState<LiveControl['progress']>(null);
  const [startedAt, setStartedAt] = useState(0);
  // the reader's own agent; null = our test agent (llama3.1:8b on Ollama)
  const [agent, setAgent] = useState<MyAgent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [upload, setUpload] = useState<{ busy: boolean; error: string | null; note: string | null }>({ busy: false, error: null, note: null });
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => void (alive.current = false);
  }, []);

  const getJson = useCallback(async <T,>(path: string): Promise<T> => {
    const r = await fetch(`${api}${path}`);
    if (!r.ok) throw new Error(`${r.status}`);
    return (await r.json()) as T;
  }, [api]);

  const loadFlow = useCallback(async () => {
    if (api === null) return null;
    const f = await getJson<FlowChoices>('/api/flow');
    setFlow(f);
    const ready = Object.keys(f.labels);
    const libs = await Promise.all(ready.map((d) => getJson<Library>(`/api/flow/library?dataset=${encodeURIComponent(d)}`).catch(() => [])));
    setLibraries(Object.fromEntries(ready.map((d, i) => [d, libs[i]])));
    return f;
  }, [api, getJson]);

  useEffect(() => {
    fetch('./data/mock-run.json')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setExample)
      .catch((e: unknown) => setLoadError(String(e)));
    loadFlow().catch(() => setFlow(null));
  }, [loadFlow]);

  const start = useCallback(
    async (scope: string[], gap: number, oneRoute = false) => {
      if (api === null || !flow || !chosen || !libraries[chosen]) return;
      const dataset = chosen;
      setStartedAt(Date.now());
      setState('running');
      setError(null);
      setProgress({ phase: 'starting' });
      try {
        const r = await fetch(`${api}/api/jobs/quick`, {
          method: 'POST',
          headers: { 'content-type': 'application/json', 'x-verifier-client': '1' },
          body: JSON.stringify({ scenario: dataset, gap, arms: ARMS, scope, one_route: oneRoute, ...(agent ? { agent: { url: agent.url, usd_per_message: agent.usd } } : {}) }),
        });
        if (!r.ok) throw new Error(((await r.json().catch(() => ({}))) as { detail?: string }).detail ?? `${r.status}`);
        const job = (await r.json()) as { id: string };
        for (;;) {
          await new Promise((res) => setTimeout(res, 1200));
          if (!alive.current) return;
          const view = await getJson<{ state: string; progress: LiveControl['progress']; output_tail?: string[] }>(`/api/jobs/${job.id}`);
          setProgress(view.progress ?? {});
          if (view.state === 'succeeded') break;
          if (view.state !== 'running' && view.state !== 'queued') throw new Error(failureReason(view.output_tail ?? []));
        }
        const res = await getJson<Result>(`/api/jobs/${job.id}/result`);
        if (!alive.current) return;
        setMine(toRun(res, flow, libraries[dataset], dataset));
        getJson<FlowChoices>('/api/flow').then((f) => alive.current && setFlow(f)).catch(() => undefined);
        setState('done');
      } catch (e) {
        if (!alive.current) return;
        setError(e instanceof Error ? e.message : String(e));
        setState('failed');
      }
    },
    [api, flow, chosen, libraries, getJson, agent],
  );

  const addOwn = useCallback(
    async (u: OwnUpload) => {
      if (api === null) return;
      setUpload({ busy: true, error: null, note: null });
      try {
        const fmt = u.file.name.toLowerCase().endsWith('.jsonl') ? 'jsonl' : 'csv';
        const q = new URLSearchParams({ name: u.name, format: fmt, item: u.item, items: u.items, label_kind: u.labelKind, title: u.title || u.name });
        const r = await fetch(`${api}/api/own?${q}`, { method: 'POST', headers: { 'x-verifier-client': '1' }, body: await u.file.arrayBuffer() });
        const out = (await r.json().catch(() => ({}))) as { detail?: string; name?: string; cases?: number; labels?: number; detectable_gap?: number | null };
        if (!r.ok) throw new Error(out.detail ?? `${r.status}`);
        await loadFlow();
        setChosen(out.name ?? u.name);
        const can = out.detectable_gap == null ? 'Too few messages for even a 20-point difference: add more rows.' : `It can spot a ${Math.round(out.detectable_gap * 100)}-point difference or bigger.`;
        setUpload({ busy: false, error: null, note: `Added ${out.cases} messages in ${out.labels} kinds and split them once. ${can}` });
      } catch (e) {
        setUpload({ busy: false, error: e instanceof Error ? e.message : String(e), note: null });
      }
    },
    [api, loadFlow],
  );

  const checkAgent = useCallback(
    async (url: string): Promise<AgentCheck> => {
      if (api === null || !chosen) throw new Error('pick a use case first');
      const r = await fetch(`${api}/api/agent/check`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-verifier-client': '1' },
        body: JSON.stringify({ dataset: chosen, url }),
      });
      const out = (await r.json().catch(() => ({}))) as AgentCheck & { detail?: string };
      if (!r.ok) throw new Error(out.detail ?? `${r.status}`);
      return out;
    },
    [api, chosen],
  );

  if (!example)
    return (
      <div className="boot">
        <span className="boot-dot" />
        {loadError ?? 'Loading…'}
      </div>
    );
  const cases: UseCase[] = (flow?.datasets ?? [])
    .filter((d) => flow?.labels[d.option])
    .map((d) => caseCard(d.option, d.title, flow!.labels[d.option].length, flow!.eval_sizes?.[d.option] ?? null, !!d.own));
  const live: LiveControl = {
    canRun: api !== null && !!flow && !!chosen && !!libraries[chosen],
    state,
    progress,
    startedAt,
    error,
    mine: !!mine,
    start,
    cases,
    chosen,
    choose: setChosen,
    libraries,
    planSizes: flow?.plan_sizes ?? {},
    speeds: flow?.speeds ?? {},
    canUpload: api !== null && !!flow,
    upload: addOwn,
    uploadState: upload,
    agent,
    setAgent,
    checkAgent,
  };
  return <MockPage run={mine ?? example} library={chosen ? libraries[chosen] : undefined} live={live} aiGuide={false} />;
}
