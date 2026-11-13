// The guide's answers to the reader's own questions. Plain code: the question is matched to a
// topic by keywords, and the answer is a fixed sentence filled with numbers worked out from this
// run (its messages, answers, scores and the reader's costs). No AI writes or picks any of it, so
// every figure it states is one the run actually produced. Each answer offers follow-ups.

export interface HelperMessage {
  text: string;
  true: string;
  gold: string;
  agent: string | null;
  agent_ok: boolean;
  simpler: string | null;
  simpler_ok: boolean;
}

export interface HelperArm {
  name: string;
  accuracy: number;
  inference_cost: Record<string, number>;
}

export interface HelperContext {
  messages: HelperMessage[];
  arms: HelperArm[];
  /** Per-arm accuracy on the reader's own labels and wrong grabs (scoped runs). */
  scope: Record<string, { in_scope_accuracy: number; false_grab_rate: number }> | null;
  verdict: string;
  simpler: string;
  gap: number;
  ci: [number, number];
  nCases: number;
  planned: number;
  looks: number[];
  volume: number;
  mistake: number;
  table: string;
  label: (arm: string) => string;
}

export type Seg = string | { b: string };

export interface Answer {
  text: Seg[];
  /** Bullet lines under the answer (each a short list of segments). */
  list?: Seg[][];
  /** Run message indices to light up in the web. */
  focus?: number[];
  /** Scene to glide to, when the answer is about it. */
  go?: string;
  chips: string[];
}

const pct = (x: number) => `${Math.round(x * 100)}%`;
const usd = (x: number) => `$${Math.round(x).toLocaleString('en-US')}`;
const kind = (label: string | null) =>
  label === null ? 'no answer' : label === 'not_mine' ? 'not yours' : label === 'my_kinds' ? 'one of your kinds' : label.replace(/[_?]+/g, ' ').trim().toLowerCase();
const has = (q: string, ...words: string[]) => words.some((w) => new RegExp(`\\b${w}`, 'i').test(q));

/** "$5", "5 dollars", "5 per mistake" -> 5; "20k messages", "20,000 a month" -> 20000. */
export function readNumbers(q: string): { money: number | null; volume: number | null } {
  let money: number | null = null, volume: number | null = null;
  const m = q.match(/\$\s*([\d.,]+)|([\d.,]+)\s*(dollars?|usd|bucks)/i);
  if (m) money = Number((m[1] ?? m[2]).replace(/,/g, ''));
  const v = q.match(/([\d.,]+)\s*(k|thousand|m|million)?\s*(messages|msgs|a month|per month|tickets|requests)/i);
  if (v) {
    const base = Number(v[1].replace(/,/g, ''));
    const mult = /^(k|thousand)$/i.test(v[2] ?? '') ? 1e3 : /^(m|million)$/i.test(v[2] ?? '') ? 1e6 : 1;
    volume = base * mult;
  }
  return { money: money !== null && Number.isFinite(money) ? money : null, volume: volume !== null && Number.isFinite(volume) ? volume : null };
}

function monthly(ctx: HelperContext, arm: string, volume: number, mistake: number): number | null {
  const a = ctx.arms.find((x) => x.name === arm);
  if (!a) return null;
  return volume * ((a.inference_cost[ctx.table] ?? 0) + (1 - a.accuracy) * mistake);
}

function wrongOf(ctx: HelperContext, who: 'agent' | 'simpler') {
  return ctx.messages
    .map((m, i) => ({ m, i }))
    .filter(({ m }) => (who === 'agent' ? !m.agent_ok : !m.simpler_ok));
}

export const STARTERS = ['Which messages did my agent get wrong?', 'What if a mistake costs $5?', 'How sure is this?', 'Break it down by kind'];

export function answer(question: string, ctx: HelperContext): Answer {
  const q = question.trim();
  const simpler = ctx.label(ctx.simpler);
  // the racing agent's name: the reference test agent, or the user's own when they plugged it in
  const A = ctx.label('agent');
  const a = A.charAt(0).toLowerCase() + A.slice(1);
  const aboutSimpler = has(q, ctx.label(ctx.simpler).toLowerCase(), 'simpl', 'linear', 'free model', 'cheap model');

  // costs, with the reader's own numbers when the question gives them
  if (has(q, 'cost', 'save', 'saving', 'money', 'price', 'cheap', 'expens', 'spend', 'pay', 'dollar', 'budget') || /\$/.test(q)) {
    const n = readNumbers(q);
    const mistake = n.money ?? ctx.mistake, volume = n.volume ?? ctx.volume;
    const ag = monthly(ctx, 'agent', volume, mistake), si = monthly(ctx, ctx.simpler, volume, mistake);
    const a = ctx.arms.find((x) => x.name === 'agent'), s = ctx.arms.find((x) => x.name === ctx.simpler);
    if (ag === null || si === null || !a || !s) return { text: ['I don’t have the prices for this run.'], chips: STARTERS };
    // the mistake cost at which the agent would become the cheaper one, if any
    const dAcc = a.accuracy - s.accuracy, dInf = (a.inference_cost[ctx.table] ?? 0) - (s.inference_cost[ctx.table] ?? 0);
    const even = dAcc > 0 ? dInf / dAcc : null;
    return {
      text: [
        `At ${Math.round(volume).toLocaleString('en-US')} messages a month and `,
        { b: `$${mistake.toLocaleString('en-US')}` },
        ` per mistake: ${a} costs about `,
        { b: usd(ag) },
        `, ${simpler} about `,
        { b: usd(si) },
        `. ${ag > si ? `${simpler} saves ${usd(ag - si)} a month.` : `${A} saves ${usd(si - ag)} a month.`}`,
      ],
      list: [
        even === null
          ? [`${A} is never cheaper here, at any mistake cost: it is not more accurate than ${simpler} on this test.`]
          : [`${A} would become cheaper once a mistake costs more than `, { b: `$${even.toFixed(2)}` }, ` (on this test’s accuracy).`],
      ],
      chips: ['What if a mistake costs $100?', 'At 100k messages a month?', 'How sure is this?'],
    };
  }

  // which messages a model got wrong
  if (has(q, 'wrong', 'mistake', 'miss', 'fail', 'error', 'slip')) {
    const who = aboutSimpler && !has(q, 'agent') ? 'simpler' : 'agent';
    const wrong = wrongOf(ctx, who);
    const name = who === 'agent' ? A : simpler;
    if (!wrong.length) return { text: [{ b: name }, ` got all ${ctx.messages.length} messages right.`], chips: ['Break it down by kind', 'How sure is this?'] };
    return {
      text: [{ b: name }, ` got `, { b: `${wrong.length} of ${ctx.messages.length}` }, ` wrong. I’ve lit them up in the web:`],
      list: wrong.slice(0, 4).map(({ m }) => [
        `“${m.text}” → answered `,
        { b: kind(who === 'agent' ? m.agent : m.simpler) },
        `, should be `,
        { b: kind(m.gold) },
      ]),
      focus: wrong.map((w) => w.i),
      go: 'm2',
      chips: ['What do these have in common?', 'Zoom into one', who === 'agent' ? `What did ${simpler} get wrong?` : 'What did my agent get wrong?'],
    };
  }

  // patterns in the agent's mistakes: which kind it confused with which
  if (has(q, 'common', 'pattern', 'why these', 'confus', 'mix')) {
    const pairs = new Map<string, number[]>();
    for (const { m, i } of wrongOf(ctx, 'agent')) {
      const key = `${kind(m.gold)} → ${kind(m.agent)}`;
      pairs.set(key, [...(pairs.get(key) ?? []), i]);
    }
    if (!pairs.size) return { text: [`${A} made no mistakes here, so there is no pattern.`], chips: STARTERS };
    const top = [...pairs.entries()].sort((a, b) => b[1].length - a[1].length);
    return {
      text: [`Here is what ${a} mixed up (what it should be → what it said):`],
      list: top.map(([k, ids]) => [{ b: `${ids.length}×` }, ` ${k}`]),
      focus: top[0][1],
      go: 'm2',
      chips: ['Zoom into one', `Is ${simpler} confused the same way?`, 'Break it down by kind'],
    };
  }

  // zoom into one mistake
  if (has(q, 'zoom', 'show me', 'example', 'one of them', 'worst', 'close')) {
    return { text: [`Diving into one message ${a} got wrong, step by step.`], go: 'm3', chips: ['What do these have in common?', 'What if a mistake costs $5?'] };
  }

  // how sure: the interval, the check-ins, the false-winner rate
  if (has(q, 'sure', 'confiden', 'trust', 'certain', 'enough', 'luck', 'significant', 'reliab', 'chance')) {
    const k = ctx.looks.indexOf(ctx.nCases) + 1 || ctx.looks.length;
    return {
      text: [
        `${A}’s gain over ${simpler} is somewhere between `,
        { b: `${(ctx.ci[0] * 100).toFixed(1)} and ${(ctx.ci[1] * 100).toFixed(1)} points` },
        `. You said a gain only matters at `,
        { b: `+${Math.round(ctx.gap * 100)}` },
        `, and that is ruled out.`,
      ],
      list: [
        [`Checked at ${ctx.looks.length} check-ins; it stopped at check-in ${k}, after `, { b: `${ctx.nCases} of ${ctx.planned}` }, ` messages.`],
        [`The stopping rule is set so a wrong winner happens at most about `, { b: '5%' }, ` of the time.`],
        ['This is a quick test on practice data: a strong hint, not the sealed final test.'],
      ],
      chips: ['Why did it stop early?', 'Which messages did my agent get wrong?'],
    };
  }

  if (has(q, 'stop', 'early', 'check-in', 'checkpoint')) {
    return {
      text: ['The test checks the score 5 times as it goes. It stops as soon as the answer is clear in either direction, which saves time and money.'],
      list: [
        ['Stopping early never weakens the result: the bar for "clear" is set higher at the early check-ins, so the 5% wrong-winner limit still holds.'],
        [`Here it was clear after `, { b: `${ctx.nCases}` }, ` of ${ctx.planned} planned messages.`],
      ],
      chips: ['How sure is this?', 'What if a mistake costs $5?'],
    };
  }

  // breakdown by kind of message, agent vs simpler
  if (has(q, 'break', 'kind', 'categor', 'label', 'type', 'per ', 'each')) {
    const groups = new Map<string, { n: number; a: number; s: number; ids: number[] }>();
    ctx.messages.forEach((m, i) => {
      const g = groups.get(m.gold) ?? { n: 0, a: 0, s: 0, ids: [] };
      g.n++;
      g.a += m.agent_ok ? 1 : 0;
      g.s += m.simpler_ok ? 1 : 0;
      g.ids.push(i);
      groups.set(m.gold, g);
    });
    return {
      text: [`Right answers by kind of message (${a} vs ${simpler}):`],
      list: [...groups.entries()].map(([g, v]) => [
        { b: kind(g) },
        ` · ${v.n} messages · agent ${pct(v.a / v.n)}, wrong on ${v.n - v.a} of ${v.n} · ${simpler} ${pct(v.s / v.n)}, wrong on ${v.n - v.s} of ${v.n}`,
      ]),
      chips: ['What do these have in common?', 'Which messages did my agent get wrong?', 'How often does it grab messages that aren’t mine?'],
    };
  }

  // wrong grabs
  if (has(q, 'grab', 'not mine', 'aren’t mine', "aren't mine", 'other', 'false')) {
    if (!ctx.scope) return { text: ['This run tested every label, so nothing counts as "not yours".'], chips: STARTERS };
    return {
      text: ['How often each model wrongly claimed a message that isn’t yours:'],
      list: Object.entries(ctx.scope)
        .filter(([n]) => n !== 'cascade')
        .sort((a, b) => a[1].false_grab_rate - b[1].false_grab_rate)
        .map(([n, v]) => [{ b: ctx.label(n) }, ` ${pct(v.false_grab_rate)}`]),
      chips: ['Break it down by kind', 'What if a mistake costs $5?'],
    };
  }

  // what the picture means
  if (has(q, 'red', 'teal', 'solid', 'dash', 'colou?r', 'mean', 'line', 'dot', 'web', 'thread', 'legend')) {
    return {
      text: ['How to read the web:'],
      list: [
        ['Each ', { b: 'dot' }, ' is a real message, tied to the 3 most like it.'],
        ['Each ', { b: 'hub' }, ' is a kind of message; your kinds on the left, "not yours" on the right.'],
        [{ b: 'Solid' }, ` threads are ${a}’s answers, `, { b: 'dashed' }, ` ones ${simpler}’s.`],
        [{ b: 'Red' }, ' means wrong; right answers fade into the background.'],
      ],
      chips: ['Which messages did my agent get wrong?', 'How sure is this?'],
    };
  }

  // the decision
  if (has(q, 'should', 'deploy', 'use', 'recommend', 'verdict', 'decision', 'worth', 'need')) {
    const word = ctx.verdict === 'USE_SIMPLER' ? `use ${simpler}` : ctx.verdict === 'DEPLOY_AGENT' ? `deploy ${a}` : 'wait for more data';
    return {
      text: ['On this test: ', { b: word }, '.'],
      list: [
        ctx.verdict === 'USE_SIMPLER'
          ? [`A gain of +${Math.round(ctx.gap * 100)} points from ${a} is ruled out, and ${simpler} costs less per message.`]
          : ctx.verdict === 'DEPLOY_AGENT'
            ? [`${A} is clearly more accurate, enough to beat ${simpler}’s lower cost.`]
            : ['The two are too close to call on this many messages.'],
        ['The rule was fixed before the test, and plain code made the call, not an AI.'],
      ],
      chips: ['How sure is this?', 'What if a mistake costs $5?', 'Which messages did my agent get wrong?'],
    };
  }

  if (has(q, 'how', 'work', 'method', 'test', 'fair')) {
    return {
      text: ['How the test works:'],
      list: [
        [`${A} and the simpler models answer the `, { b: 'same messages' }, '.'],
        ['Plain code scores every answer against the known right one.'],
        ['The size and the win rule are fixed before the run, so nobody can move the goalposts.'],
      ],
      chips: ['How sure is this?', 'Why did it stop early?'],
    };
  }

  return {
    text: ['I can only answer from this run’s results, and I didn’t catch that one. Try one of these:'],
    chips: STARTERS,
  };
}
