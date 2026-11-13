// "About N minutes" before a quick test starts (plain code, an estimate, never a promise).
// Built from times measured on this laptop: how long fitting the free models took (0 when they
// are saved), the agent's seconds per answer (our test agent's last measured speed, or your
// agent's speed in the connection check), and a fixed allowance for the checks. A range: the
// test may stop at its first check-in, or run all its messages.

export interface EstimateInput {
  /** messages the test plans to use */
  cases: number;
  /** the planned check-ins (cases used at each look), or empty if unknown */
  looks?: number[];
  /** free models already saved for this dataset (then they load in seconds) */
  modelsSaved: boolean;
  /** how long fitting them took when last measured, in seconds; null = never measured */
  fitSeconds: number | null;
  /** seconds per answer of the agent being tested; null = unknown */
  agentSeconds: number | null;
}

const LOAD_SECONDS = 15; // loading saved models and the shared layer
const CHECKS_SECONDS = 60; // drift, leakage and false-alarm checks after the race
const UNKNOWN_FIT: [number, number] = [180, 480]; // 3-8 minutes, seen on this laptop
const UNKNOWN_AGENT = 2.0; // seconds per answer, from saved runs of our test agent

export function estimateMinutes(x: EstimateInput): { lo: number; hi: number; firstTime: boolean } {
  const fit: [number, number] = x.modelsSaved ? [LOAD_SECONDS, LOAD_SECONDS] : x.fitSeconds ? [x.fitSeconds, x.fitSeconds * 1.2] : UNKNOWN_FIT;
  const per = x.agentSeconds ?? UNKNOWN_AGENT;
  const first = x.looks?.length ? x.looks[0] : Math.ceil(x.cases / 5);
  const lo = (fit[0] + CHECKS_SECONDS + first * per) / 60;
  const hi = (fit[1] + CHECKS_SECONDS + x.cases * per * (x.agentSeconds ? 1.15 : 1.5)) / 60;
  return { lo: Math.max(1, Math.round(lo)), hi: Math.max(1, Math.round(hi)), firstTime: !x.modelsSaved };
}

export function estimateText(x: EstimateInput): string {
  const { lo, hi, firstTime } = estimateMinutes(x);
  const span = lo === hi ? `about ${lo} min` : `about ${lo}–${hi} min`;
  return `${span} (may stop early)${firstTime ? ' · the first test on this use case also trains the free models, once' : ''}`;
}
