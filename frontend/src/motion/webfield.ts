// Plain-code word matching for the live page (owner-approved, no AI): which kinds of message match
// what the reader typed, how well a whole dataset fits, and display names for kinds. The earlier
// 2D message-web renderer that lived here was removed (owner approved 2026-11-13); the page's graph
// is motion/brain.ts.

export interface WebMessage {
  text: string;
  true: string;
  gold: string;
  near: boolean;
  agent: string | null;
  agent_ok: boolean;
  simpler: string | null;
  simpler_ok: boolean;
  nb: number[];
}

const stem = (w: string) => w.toLowerCase().replace(/(ing|ed|es|s)$/, '');
const words = (text: string) =>
  text
    .split(/[^a-zA-Z]+/)
    .filter((w) => w.length > 2)
    .map(stem);
// everyday words people type, mapped to the words the datasets use (plain code, both directions
// not needed: only the reader's query is widened)
const FAMILY: Record<string, string[]> = {
  angry: ['anger'], mad: ['anger'], furious: ['anger'], annoy: ['anger'], frustrat: ['anger'], upset: ['sadness', 'anger'],
  happy: ['joy'], glad: ['joy'], pleas: ['joy'], delight: ['joy'], satisf: ['joy'],
  sad: ['sadness'], unhappy: ['sadness'], disappoint: ['sadness'],
  worri: ['fear'], scar: ['fear'], afraid: ['fear'], anxiou: ['fear'], nervou: ['fear'],
  surpris: ['surprise'], shock: ['surprise'], love: ['love'],
  mood: ['anger', 'joy', 'sadness', 'fear'], feel: ['anger', 'joy', 'sadness', 'fear'], emotion: ['anger', 'joy', 'sadness', 'fear'], tone: ['anger', 'joy'],
  song: ['music', 'play'], tune: ['music'], jazz: ['music'], rock: ['music'], playlist: ['music'], radio: ['radio'],
  light: ['light'], lamp: ['light'], off: ['lightoff'], dim: ['lightdim'], bright: ['lightup'], colour: ['lightchange'], color: ['lightchange'],
  alarm: ['alarm'], wake: ['alarm'],
  weather: ['weather'], flight: ['flight', 'travel'], hotel: ['travel'], restaurant: ['restaurant', 'reservation'], table: ['reservation'],
  money: ['transfer', 'balance'], pay: ['payment', 'card'], bank: ['card', 'transfer', 'account'], refund: ['refund'], lost: ['lost'], stolen: ['stolen', 'lost'],
};
const FAMILY_STEMS = Object.entries(FAMILY).map(([k, v]) => [stem(k), v] as const);
const widen = (list: string[]) =>
  [...new Set(list.flatMap((w) => [w, ...FAMILY_STEMS.filter(([k]) => w === k || w.startsWith(k)).flatMap(([, v]) => v).map(stem)]))];

/** Labels whose name or practice messages share words with the query, best first (plain code). */
const idfCache = new WeakMap<object, Map<string, number>>();
/** How rare each word is across labels: log(labels / labels using the word). */
function idf(library: { text: string; label: string }[]): Map<string, number> {
  const hit = idfCache.get(library);
  if (hit) return hit;
  const byWord = new Map<string, Set<string>>();
  for (const item of library) for (const w of [...words(item.text), ...words(item.label.replace(/_/g, ' '))]) {
    if (!byWord.has(w)) byWord.set(w, new Set());
    byWord.get(w)?.add(item.label);
  }
  const labels = new Set(library.map((i) => i.label)).size;
  const out = new Map([...byWord].map(([w, ls]) => [w, Math.log((labels + 1) / ls.size)]));
  idfCache.set(library, out);
  return out;
}

export function matchLabels(query: string, library: { text: string; label: string }[], max = 6): string[] {
  const q = new Set(widen(words(query)));
  if (!q.size) return [];
  const weight = idf(library);
  const score = new Map<string, number>();
  for (const item of library) {
    const w = (list: string[]) => list.filter((x) => q.has(x)).reduce((a, x) => a + (weight.get(x) ?? 0), 0);
    const hits = w(words(item.text)) + 3 * w(words(item.label.replace(/_/g, ' ')));
    if (hits) score.set(item.label, (score.get(item.label) ?? 0) + hits);
  }
  // keep only labels close to the best one, so one shared word does not drag in weak matches
  const best = Math.max(0, ...score.values());
  for (const [l, v] of score) if (v < best * 0.25) score.delete(l);
  return [...score.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, max)
    .map(([l]) => l);
}

export function hubName(label: string): string {
  return label === 'not_mine' ? 'not yours' : label.replace(/[_?]+/g, ' ').trim().toLowerCase();
}

/** How well a dataset fits the reader's words: its best label's score (plain code; 0 = none). */
export function matchScore(query: string, library: { text: string; label: string }[]): number {
  const q = new Set(widen(words(query)));
  if (!q.size || !library.length) return 0;
  const weight = idf(library);
  const score = new Map<string, number>();
  for (const item of library) {
    const w = (list: string[]) => list.filter((x) => q.has(x)).reduce((a, x) => a + (weight.get(x) ?? 0), 0);
    const hits = w(words(item.text)) + 3 * w(words(item.label.replace(/_/g, ' ')));
    if (hits) score.set(item.label, (score.get(item.label) ?? 0) + hits);
  }
  return Math.max(0, ...score.values());
}

/** The reader's words, widened by the everyday word families (for keyword checks elsewhere). */
export function queryWords(query: string): string[] {
  return widen(words(query));
}
