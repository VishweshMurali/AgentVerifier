// Running a quick test from the page. With the local server, the choice is posted to
// /api/jobs/quick and the job is polled: each check-in arrives as the test prints it, then the
// finished plan, scope and verdict are read back. Without a server (the static preview), the saved
// run is replayed. Nothing here scores or decides; the test and the engine do.
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Choices, LookPoint, SavedRun } from './model.ts';
import { replayLooks } from './model.ts';

export type Phase = 'idle' | 'starting' | 'running' | 'done' | 'failed';

export interface RunState {
  phase: Phase;
  /** Check-ins so far: full points from the server, case counts only in the replay. */
  looks: LookPoint[];
  replayed: number[];
  planned: number | null;
  simpler: string | null;
  result: SavedRun | null;
  error: string | null;
}

const IDLE: RunState = { phase: 'idle', looks: [], replayed: [], planned: null, simpler: null, result: null, error: null };
const CLIENT_HEADER = { 'x-verifier-client': '1', 'content-type': 'application/json' };
const POLL_MS = 800;

interface JobView {
  id: string;
  state: 'running' | 'succeeded' | 'failed';
  output_tail: string[];
  progress: { looks?: LookPoint[]; planned?: number; simpler?: string; phase?: string };
}

async function readError(r: Response): Promise<string> {
  try {
    const body = (await r.json()) as { detail?: unknown };
    if (typeof body.detail === 'string') return body.detail;
  } catch {
    /* not JSON */
  }
  return `${r.status} ${r.statusText}`;
}

/** The last meaningful output line of a failed job, for the reader. */
export function failureReason(tail: readonly string[]): string {
  const lines = tail.map((l) => l.trim()).filter(Boolean);
  return lines.at(-1) ?? 'the test ended without a result';
}

export function useQuickRun(api: string | null, saved: SavedRun | null) {
  const [state, setState] = useState<RunState>(IDLE);
  const alive = useRef(true);
  const token = useRef(0);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const reset = useCallback(() => {
    token.current++;
    setState(IDLE);
  }, []);

  const start = useCallback(
    async (c: Choices) => {
      const my = ++token.current;
      const live = () => alive.current && token.current === my;
      setState({ ...IDLE, phase: 'starting' });

      if (api === null) {
        // static preview: replay the saved run's check-ins, one per beat
        if (!saved) return setState({ ...IDLE, phase: 'failed', error: 'no saved run to replay' });
        const looks = replayLooks(saved);
        setState({ ...IDLE, phase: 'running', planned: saved.plan.planned_cases, simpler: saved.plan.comparator });
        for (let i = 0; i < looks.length; i++) {
          await new Promise((r) => setTimeout(r, 1100));
          if (!live()) return;
          setState((s) => ({ ...s, replayed: looks.slice(0, i + 1) }));
        }
        await new Promise((r) => setTimeout(r, 700));
        if (live()) setState((s) => ({ ...s, phase: 'done', result: saved }));
        return;
      }

      try {
        const r = await fetch(`${api}/api/jobs/quick`, {
          method: 'POST',
          headers: CLIENT_HEADER,
          body: JSON.stringify({ scenario: c.dataset, gap: c.gap, arms: c.arms, scope: c.labels }),
        });
        if (!r.ok) throw new Error(await readError(r));
        const job = (await r.json()) as JobView;
        setState((s) => ({ ...s, phase: 'running' }));
        for (;;) {
          await new Promise((res) => setTimeout(res, POLL_MS));
          if (!live()) return;
          const pr = await fetch(`${api}/api/jobs/${job.id}`);
          if (!pr.ok) throw new Error(await readError(pr));
          const view = (await pr.json()) as JobView;
          const p = view.progress ?? {};
          setState((s) => ({ ...s, looks: p.looks ?? [], planned: p.planned ?? s.planned, simpler: p.simpler ?? s.simpler }));
          if (view.state === 'failed') throw new Error(failureReason(view.output_tail));
          if (view.state === 'succeeded') break;
        }
        const rr = await fetch(`${api}/api/jobs/${job.id}/result`);
        if (!rr.ok) throw new Error(await readError(rr));
        const result = (await rr.json()) as SavedRun;
        if (live()) setState((s) => ({ ...s, phase: 'done', result }));
      } catch (err) {
        if (live()) setState((s) => ({ ...s, phase: 'failed', error: err instanceof Error ? err.message : String(err) }));
      }
    },
    [api, saved],
  );

  return { state, start, reset };
}
