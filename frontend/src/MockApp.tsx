import { useEffect, useState } from 'react';
import { MockPage, type MockRun } from './components/MockPage.tsx';

/** Mockup for owner review: loads the real refund run exported to public/data/mock-run.json. */
export function MockApp() {
  const [run, setRun] = useState<MockRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch('./data/mock-run.json')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setRun)
      .catch((e: unknown) => setError(String(e)));
  }, []);
  return run ? <MockPage run={run} /> : <p style={{ padding: 40 }}>{error ?? 'Loading the mockup…'}</p>;
}
