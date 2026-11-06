// Types mirroring public/data/story.json, written by scripts/export_frontend_data.py.
// Every number the page shows comes from this data; nothing is computed by an AI.

export type StorySource = 'rehearsal' | 'sealed';

export type VerdictName = 'DEPLOY_AGENT' | 'USE_SIMPLER' | 'NEED_MORE_DATA' | 'INVALID';

export type EvidenceLevel = 'REAL' | 'SYNTHETIC_CHECKED' | 'SYNTHETIC';

export interface Audit {
  git_commit: string;
  prereg_sha256: string;
  dev_report_sha256: string;
  test_sha256: string;
  model_digest: string;
  layer_fingerprint: string;
  predictions_sha256: string;
  [key: string]: string;
}

export interface Check {
  name: string;
  value: number | null;
  low: number | null;
  high: number | null;
  passed: boolean;
  effect: string;
  detail: string;
}

export interface Arm {
  name: string;
  n: number;
  correct: number;
  accuracy: number;
  prompt_tokens_mean: number;
  completion_tokens_mean: number;
  latency_s_mean: number;
  flagged_accuracy: number | null;
  /** Inference cost per case in USD, keyed by price table name. */
  inference_cost: Record<string, number>;
}

/** Break-even error cost for one price table, with its CI (only when the agent is ahead). */
export interface BreakEven {
  price_table: string;
  value: number;
  ci_low: number;
  ci_high: number;
}

export interface Verdict {
  verdict: VerdictName | string;
  evidence_level: EvidenceLevel | string;
  audit: Audit;
  reasons: string[];
  scorer: string | null;
  quality_floor_passed: boolean;
  accuracy: Record<string, number>;
  flagged_rate: number;
  shortlist_coverage: number;
  flagged_accuracy: Record<string, number>;
  /** Agent accuracy minus the best simpler arm's accuracy. */
  accuracy_diff: number;
  accuracy_diff_ci: [number, number];
  mcnemar_p: number;
  aa_false_winner_rate: number;
  break_even: BreakEven[];
  latency_s: Record<string, number>;
  checks: Check[];
  agent_arm: string;
  simpler_arm: string;
  n_cases: number;
  n_dropped: number;
  mcnemar_counts: [number, number];
  arms: Arm[];
  error_cost_range: [number, number];
  rehearsal: boolean;
  alpha: number;
  ci_level: number;
  min_gap: number;
}

export interface PriceTable {
  name: string;
  input_usd_per_mtok: number;
  output_usd_per_mtok: number;
  assumed: boolean;
}

export interface Rules {
  alpha: number;
  min_gap: number;
  checks: number;
  price_tables: number;
}

export interface StoryCase {
  id: string;
  text: string;
  gold: string;
  linear: string;
  shortlist: string[];
  search_query: string | null;
  agent: string;
  /** Extra fields a newer export may add (for example the split the case comes from) are allowed. */
  [key: string]: unknown;
}

export interface Story {
  source: StorySource;
  verdict: Verdict;
  price_tables: PriceTable[];
  default_price_table: string;
  rules: Rules;
  case: StoryCase | null;
}

/** GET /api/status from the local server (server/), when the page runs against it. */
export interface LiveStatus {
  ollama: boolean;
  sealed_run: {
    state: 'not_started' | 'started_not_finished' | 'finished' | string;
    started_at_commit: string | null;
    verdict: { mtime: number } | null;
    /** Test cases before copies of DEV text are dropped. */
    test_cases: number | null;
    /** Run-phase answers cached so far, per arm. */
    answers_cached: Record<string, number>;
  };
  rehearsal: { verdict: { mtime: number } | null };
  job: { id: string; kind: string; state: string } | null;
}
