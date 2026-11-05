import { useEffect, useState } from 'react';
import { createProvider } from './data/index.ts';
import type { Story } from './data/types.ts';
import { StoryPage } from './components/StoryPage.tsx';

type State = { kind: 'loading' } | { kind: 'ready'; story: Story } | { kind: 'error'; message: string; provider: string };

export function App() {
  const [state, setState] = useState<State>({ kind: 'loading' });

  useEffect(() => {
    let alive = true;
    let providerName = 'unknown';
    (async () => {
      try {
        const provider = createProvider();
        providerName = provider.name;
        const story = await provider.load();
        if (alive) setState({ kind: 'ready', story });
      } catch (err) {
        if (alive) setState({ kind: 'error', message: err instanceof Error ? err.message : String(err), provider: providerName });
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  if (state.kind === 'ready') return <StoryPage story={state.story} />;

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
