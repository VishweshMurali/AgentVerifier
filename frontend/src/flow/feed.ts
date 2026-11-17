// The live feed on the race screen: what the test is doing behind the page, in plain lines. Every
// line comes from the test's own output (its progress) or from counts the server read from the
// usage ledger (requests, replays, tokens, times). No message text, prompt or answer is shown, and
// nothing here is guessed: a line appears only when a number changed.
import type { RunProgress } from './runlog.ts';

/** Counts of the model requests a test has made so far (server/status.py call_stats). */
export type CallStats = { requests: number; replayed: number; failed: number; tokens: number; avg_s: number; last_s: number };

export type FeedEvent = { at: number; text: string };

const lowerFirst = (s: string) => s.charAt(0).toLowerCase() + s.slice(1);
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const points = (gain: number) => `${Math.abs(gain * 100).toFixed(1)} points ${gain >= 0 ? 'ahead of' : 'behind'}`;

/**
 * The new lines between two polls. `at` is the seconds since the test started; `name` turns an arm
 * id into the page's word for it; `agent` is the racing agent's name.
 */
export function feedEvents(
  prev: { progress: RunProgress | null; calls: CallStats | null },
  next: { progress: RunProgress | null; calls: CallStats | null },
  at: number,
  name: (arm: string) => string,
  agent: string,
): FeedEvent[] {
  const out: FeedEvent[] = [];
  const say = (text: string) => out.push({ at, text });
  const p = prev.progress ?? {};
  const n = next.progress ?? {};

  if (n.queued && n.queued !== p.queued) say(`Waiting in line: ${plural(n.queued, 'test')} ahead of yours`);
  const total = n.drawn ?? n.planned;
  if (total && !(p.drawn ?? p.planned)) say(`Picked ${total} real messages to test`);
  const trained = n.trained ?? [];
  for (const m of trained.slice((p.trained ?? []).length)) say(`Trained ${name(m)}, a free model`);
  if (n.simpler && n.simpler !== p.simpler) say(`${name(n.simpler)} is the free model ${lowerFirst(agent)} has to beat`);

  const lookedBefore = p.looks?.length ?? 0;
  (n.looks ?? []).slice(lookedBefore).forEach((l, i) => {
    const gain = l.gain;
    say(
      `Check-in ${lookedBefore + i + 1} at ${l.n} messages` +
        (typeof gain === 'number' ? `: ${lowerFirst(agent)} is ${points(gain)} ${name(n.simpler ?? 'the free model')}` : ''),
    );
  });
  if (n.scored && n.scored !== p.scored && total) say(`${agent} has answered ${n.scored} of ${total}`);

  const a = prev.calls;
  const b = next.calls;
  if (b && b.requests > (a?.requests ?? 0)) {
    const d = b.requests - (a?.requests ?? 0);
    say(`${plural(d, 'request')} sent to the model (${b.requests} so far, last answer in ${b.last_s.toFixed(1)} s)`);
  }
  if (b && b.replayed > (a?.replayed ?? 0)) {
    const d = b.replayed - (a?.replayed ?? 0);
    say(`${plural(d, 'answer')} replayed from saved answers (no request needed)`);
  }
  if (b && b.failed > (a?.failed ?? 0)) say(`${plural(b.failed - (a?.failed ?? 0), 'request')} ended in an error (counted as wrong)`);

  if (n.phase === 'finishing' && p.phase !== 'finishing') {
    say(n.stopped_at && n.of && n.stopped_at < n.of ? `Stopped early at ${n.stopped_at} of ${n.of} messages: clear enough` : 'All messages answered');
    say('Scoring the other models and writing your answer');
  }
  if (n.phase === 'done' && p.phase !== 'done') say('Your answer is ready');
  return out;
}

export function clock(secs: number): string {
  const s = Math.max(0, Math.round(secs));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
