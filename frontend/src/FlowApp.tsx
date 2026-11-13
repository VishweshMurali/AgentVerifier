import { useEffect, useState } from 'react';
import { FlowPage } from './components/FlowPage.tsx';
import type { LiveStatus } from './data/types.ts';
import type { FlowData } from './flow/model.ts';

/**
 * The "test your agent" flow. With the local server (VITE_DATA_PROVIDER=api) its choices come from
 * /api/flow and Run starts a real quick test; the static build replays the saved preview run.
 */
export function FlowApp() {
  const api = import.meta.env.VITE_DATA_PROVIDER === 'api' ? (import.meta.env.VITE_API_URL ?? '') : null;
  const [data, setData] = useState<FlowData | null>(null);
  const [live, setLive] = useState<LiveStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(api === null ? './data/quick-sample.json' : `${api}/api/flow`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${r.statusText}`))))
      .then((d: FlowData) => setData(api === null ? d : { ...d, saved_run: null }))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [api]);

  useEffect(() => {
    if (api === null) return;
    let alive = true;
    const poll = () =>
      fetch(`${api}/api/status`)
        .then((r) => (r.ok ? (r.json() as Promise<LiveStatus>) : null))
        .then((s) => alive && setLive(s))
        .catch(() => alive && setLive(null));
    poll();
    const id = setInterval(poll, 10000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [api]);

  if (data) return <FlowPage data={data} api={api} live={live} storyHref={window.location.pathname} />;
  return (
    <main className="status-screen">
      <div className="card status-card" role={error ? 'alert' : 'status'}>
        <span className="brand">
          Agent Verifier<span className="dot">.</span>
        </span>
        <span className="eyebrow">{error ? 'Could not load the test flow' : 'Loading'}</span>
        <p className="lead">{error ?? 'Getting the test ready…'}</p>
      </div>
    </main>
  );
}
