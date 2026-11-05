// Plain number formatting for the page. Pure, deterministic.

const MINUS = '−';

/** 0.8994 -> "89.9%" */
export function pct(x: number, digits = 1): string {
  return `${(x * 100).toFixed(digits)}%`;
}

/** Signed points with a real minus sign: -0.0752 -> "−7.5", 0.03 -> "+3.0". */
export function signedPts(diff: number, digits = 1): string {
  const v = Number((diff * 100).toFixed(digits));
  if (v === 0) return (0).toFixed(digits);
  return (v < 0 ? MINUS : '+') + Math.abs(v).toFixed(digits);
}

/** USD: two decimals, or four when the amount is under a cent (so tiny costs do not read as $0.00). */
export function usd(x: number): string {
  const sign = x < 0 ? MINUS : '';
  const a = Math.abs(x);
  const digits = a !== 0 && a < 0.01 ? 4 : 2;
  return `${sign}$${a.toFixed(digits)}`;
}

/** 1462 -> "1,462" (fixed locale so output never depends on the viewer's machine). */
export function count(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}

/** "local_amortized" -> "Local amortized" */
export function tableLabel(name: string): string {
  const s = name.replace(/[_-]+/g, ' ').trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Up to 4 significant digits, trimmed: 0.006793 -> "0.006793", 1462 -> "1462". */
export function num(x: number | null): string {
  if (x === null || !Number.isFinite(x)) return 'n/a';
  if (Number.isInteger(x)) return String(x);
  return String(Number(x.toPrecision(4)));
}

const WORDS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten'];

/** 3 -> "Three" */
export function countWord(n: number): string {
  return WORDS[n] ?? String(n);
}
