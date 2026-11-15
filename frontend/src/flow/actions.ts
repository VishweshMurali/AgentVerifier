// What the reader asks the guide to change on the page, read by plain code (no AI): what their
// agent does, messages a month, what one mistake costs, how big a difference to look for, and
// where the AI runs. The page applies these and says what it changed.

export interface ChatActions {
  /** what the reader's agent does: goes into "What does your agent do?" */
  describe?: string;
  /** messages a month */
  volume?: number;
  /** what one wrong answer costs, in dollars */
  mistake?: number;
  /** the difference to look for (0.2, 0.1 or 0.05) */
  gap?: number;
  /** where the AI runs (a price table name) */
  table?: string;
}

const num = (s: string) => Number(s.replace(/,/g, ''));
const scale = (unit: string | undefined) => (/^(k|thousand)$/i.test(unit ?? '') ? 1e3 : /^(m|mn|million)$/i.test(unit ?? '') ? 1e6 : 1);

export function chatActions(question: string): ChatActions {
  const q = question.trim();
  const out: ChatActions = {};

  // "my agent handles refunds", "it answers questions about cards", "we route billing tickets"
  const d = q.match(/^(?:my|our|the)?\s*(?:agent|bot|assistant|it|we)\s+(handles?|answers?|routes?|detects?|sorts?|classifies|reads?|helps?|does|deals? with|takes care of|replies to|responds to)\s+(.+)$/i);
  if (d) out.describe = `${d[1].toLowerCase()} ${d[2].replace(/[.!?]+$/, '')}`.trim();

  // "50,000 messages a month", "50k tickets per month", "2,000 a day"
  const v = q.match(/([\d][\d.,]*)\s*(k|thousand|m|mn|million)?\s*(?:messages|msgs|tickets|requests|chats|emails|conversations|queries)?\s*(?:a|per|each|every|\/)\s*(day|week|month)/i);
  if (v) {
    const per = v[3].toLowerCase() === 'day' ? 30 : v[3].toLowerCase() === 'week' ? 4.3 : 1;
    const n = num(v[1]) * scale(v[2]) * per;
    if (Number.isFinite(n) && n > 0) out.volume = Math.round(n);
  } else {
    const v2 = q.match(/([\d][\d.,]*)\s*(k|thousand|m|mn|million)?\s*(messages|msgs|tickets|requests|chats|emails|conversations|queries)\b/i);
    if (v2 && !/\$/.test(v2[0])) {
      const n = num(v2[1]) * scale(v2[2]);
      if (Number.isFinite(n) && n > 0) out.volume = Math.round(n);
    }
  }

  // "a mistake costs $5", "each error is $12.50", "$20 per wrong answer", "mistakes cost 8 dollars"
  if (/mistake|error|wrong|fix|redo|escalat/i.test(q)) {
    const m = q.match(/\$\s*([\d][\d.,]*)|([\d][\d.,]*)\s*(?:dollars?|usd|bucks)/i);
    if (m) {
      const n = num(m[1] ?? m[2]);
      if (Number.isFinite(n) && n >= 0) out.mistake = n;
    }
  }

  // "a big difference", "small gap", "clear one"
  if (/\b(big|large|huge)\s+(difference|gap|one)\b/i.test(q)) out.gap = 0.2;
  else if (/\b(clear|medium|moderate)\s+(difference|gap|one)\b/i.test(q)) out.gap = 0.1;
  else if (/\b(small|tiny|subtle)\s+(difference|gap|one)\b/i.test(q)) out.gap = 0.05;

  // "we run it locally", "on a small cloud model", "a large cloud model"
  if (/\b(local(ly)?|on[- ]prem|own (server|machine)|self[- ]hosted)\b/i.test(q)) out.table = 'local_amortized';
  else if (/\b(large|big|frontier)\s+(cloud\s+)?(model|llm|ai)\b|\bcloud large\b|\bgpt-?4|\bopus\b/i.test(q)) out.table = 'cloud_large';
  else if (/\b(small|cheap|mini)\s+(cloud\s+)?(model|llm|ai)\b|\bcloud small\b|\bin the cloud\b/i.test(q)) out.table = 'cloud_small';

  return out;
}

export const hasActions = (a: ChatActions) => Object.keys(a).length > 0;
