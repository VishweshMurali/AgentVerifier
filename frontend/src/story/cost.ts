// Expected cost per case, the decision rule's formula from prereg.yaml:
//   E[C] = inference cost (selected price table) + P(error) * error cost,   P(error) = 1 - accuracy.
// Pure and deterministic; the page only displays these numbers.
import type { Arm } from '../data/types.ts';

export type CostArm = Pick<Arm, 'name' | 'accuracy' | 'inference_cost'>;

export function inferenceCost(arm: CostArm, table: string): number {
  const c = arm.inference_cost[table];
  if (typeof c !== 'number' || !Number.isFinite(c)) {
    throw new Error(`arm "${arm.name}" has no inference cost for price table "${table}"`);
  }
  return c;
}

export function expectedCost(arm: CostArm, table: string, errorCost: number): number {
  return inferenceCost(arm, table) + (1 - arm.accuracy) * errorCost;
}

/** Agent's expected cost minus the simpler arm's: positive means the agent costs more per case. */
export function costGap(agent: CostArm, simpler: CostArm, table: string, errorCost: number): number {
  return expectedCost(agent, table, errorCost) - expectedCost(simpler, table, errorCost);
}

export type RangeWinner = 'agent' | 'simpler' | 'crosses' | 'equal';

/**
 * Which arm is cheaper across the whole error-cost range. Expected cost is linear in the error
 * cost, so checking both ends of the range is exact.
 */
export function cheaperAcrossRange(
  agent: CostArm,
  simpler: CostArm,
  table: string,
  range: readonly [number, number],
): RangeWinner {
  const lo = costGap(agent, simpler, table, range[0]);
  const hi = costGap(agent, simpler, table, range[1]);
  if (lo === 0 && hi === 0) return 'equal';
  if (lo >= 0 && hi >= 0) return 'simpler';
  if (lo <= 0 && hi <= 0) return 'agent';
  return 'crosses';
}

const STEPS = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100];

/** A round slider step giving at most ~40 positions across the range. */
export function sliderStep(range: readonly [number, number]): number {
  const span = range[1] - range[0];
  return STEPS.find((s) => span / s <= 40) ?? span / 40;
}

/**
 * Where the slider starts: a quarter of the way into the pre-registered range, snapped to the
 * step. A starting point for the reader to move, not an assumed error cost.
 */
export function defaultErrorCost(range: readonly [number, number]): number {
  const step = sliderStep(range);
  const raw = range[0] + (range[1] - range[0]) / 4;
  const snapped = range[0] + Math.round((raw - range[0]) / step) * step;
  return Math.min(range[1], Math.max(range[0], Number(snapped.toFixed(6))));
}
