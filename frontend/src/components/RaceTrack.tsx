// The race, drawn from what the test measures at each check-in: the agent's gain over the model
// to beat (dot) and its repeated confidence interval (band). The test stops when the band clears
// "no gain" (agent better) or falls short of "your gain" (gain ruled out). Positions only; every
// number comes from the quick test's own output.
import { lookInterval, type LookPoint } from '../flow/model.ts';

interface Props {
  looks: readonly LookPoint[];
  /** Case counts reached, when only those are known (the static replay). */
  reached: readonly number[];
  gates: readonly number[];
  gap: number;
  finalGain: number | null;
  simpler: string;
  agent: string;
}

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));

export function RaceTrack({ looks, reached, gates, gap, finalGain, simpler, agent }: Props) {
  const last = looks.at(-1) ?? null;
  const band = last ? lookInterval(last, gap) : null;
  const gain = last ? last.gain : finalGain;
  const lo = Math.min(-0.3, gain ?? 0, band?.[0] ?? 0) - 0.05;
  const hi = Math.max(gap + 0.15, gain ?? 0, band?.[1] ?? 0) + 0.05;
  const x = (v: number) => `${clamp(((v - lo) / (hi - lo)) * 100, 0, 100)}%`;
  const doneCount = Math.max(looks.length, reached.length);

  return (
    <div className="race" role="img" aria-label={gain === null ? 'The race has not started' : `Agent minus ${simpler}: ${(gain * 100).toFixed(1)} points`}>
      <div className="gates">
        {gates.map((n, i) => (
          <span key={n} className={i < doneCount ? 'hit' : ''}>
            <i />
            <small>{n}</small>
          </span>
        ))}
      </div>
      <div className="track">
        <div className="zone lose" style={{ left: 0, width: x(0) }} />
        <div className="zone win" style={{ left: x(gap), right: 0 }} />
        <div className="tick" style={{ left: x(0) }}>
          <small>no gain</small>
        </div>
        <div className="tick goal" style={{ left: x(gap) }}>
          <small>your gain</small>
        </div>
        {band && <div className="band" style={{ left: x(band[0]), width: `calc(${x(band[1])} - ${x(band[0])})` }} />}
        {gain !== null && <div className="dot" style={{ left: x(gain) }} />}
      </div>
      <div className="ends">
        <span>← {simpler} better</span>
        <span>{agent} better →</span>
      </div>
    </div>
  );
}
