import { useEffect, useMemo, useState } from 'react';
import { createProvider } from './data/index.ts';
import type { LiveStatus, Story } from './data/types.ts';
import { verdictStamp } from './story/live.ts';
import { StoryPage } from './components/StoryPage.tsx';

type State = { kind: 'loading' } | { kind: 'ready'; story: Story } | { kind: 'error'; message: string; provider: string };

export function App() {
  const [state, setState] = useState<State>({ kind: 'loading' });
  const [live, setLive] = useState<LiveStatus | null>(null);
  const [stamp, setStamp] = useState<string | null>(null);
  const provider = useMemo(() => {
    try {
      return createProvider();
    } catch (err) {
      return err instanceof Error ? err : new Error(String(err));
    }
  }, []);

  // load the story, and load it again whenever the server reports a new verdict file
  useEffect(() => {
    let alive = true;
    (async () => {
      if (provider instanceof Error) {
        setState({ kind: 'error', message: provider.message, provider: 'unknown' });
        return;
      }
      try {
        const story = await provider.load();
        if (alive) setState({ kind: 'ready', story });
      } catch (err) {
        if (alive) setState({ kind: 'error', message: err instanceof Error ? err.message : String(err), provider: provider.name });
      }
    })();
    return () => {
      alive = false;
    };
  }, [provider, stamp]);

  // live status from the local server (api provider only), every 10 seconds
  useEffect(() => {
    if (provider instanceof Error || !provider.status) return;
    const status = provider.status.bind(provider);
    let alive = true;
    const poll = async () => {
      try {
        const s = await status();
        if (!alive) return;
        setLive(s);
        setStamp(verdictStamp(s));
      } catch {
        if (alive) setLive(null);
      }
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [provider]);

  if (state.kind === 'ready') return <StoryPage story={state.story} live={live} />;

  return (
    <main className="status-screen">
      <div className="card status-card" role={state.kind === 'error' ? 'alert' : 'status'} aria-live="polite">
        <span className="brand">
          Agent Verifier<span className="dot">.</span>
        </span>
        {state.kind === 'loading' ? (
          <>
            <span className="eyebrow">Loading</span>
            <p className="lead">Reading the results…</p>
            <div className="bar loading-bar" aria-hidden="true">
              <div className="loading-fill" />
            </div>
          </>
        ) : (
          <>
            <span className="eyebrow">Could not load the results</span>
            <pre className="error-text">{state.message}</pre>
            <p className="note">
              Data provider: {state.provider}. Refresh the data from the repository root with{' '}
              <code>uv run python scripts/export_frontend_data.py</code>, then reload.
            </p>
          </>
        )}
      </div>
    </main>
  );
}
