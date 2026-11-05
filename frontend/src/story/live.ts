// The header's live line when the page runs against the local server. Fixed templates, pure.
import type { LiveStatus } from '../data/types.ts';
import { count } from './format.ts';

/** Changes whenever a verdict file is written, so the page knows to reload the story. */
export function verdictStamp(s: LiveStatus): string {
  return `${s.sealed_run.verdict?.mtime ?? '-'}|${s.rehearsal.verdict?.mtime ?? '-'}`;
}

export function sealedRunText(s: LiveStatus): string {
  const r = s.sealed_run;
  if (r.state === 'finished') return 'Sealed run finished';
  if (r.state === 'not_started') return 'Sealed run not started';
  const answers = Object.entries(r.answers_cached)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([arm, n]) => `${arm} ${count(n)}`)
    .join(', ');
  const of = r.test_cases ? ` of ~${count(r.test_cases)}` : '';
  return answers ? `Sealed run started: ${answers}${of} answers` : 'Sealed run started, no answers yet';
}

export function liveLine(s: LiveStatus): string {
  const parts = [s.ollama ? 'Ollama on' : 'Ollama off', sealedRunText(s)];
  if (s.job) parts.push(`${s.job.kind} ${s.job.state}`);
  return parts.join(' · ');
}
