// MOCKUP (owner review): six screens, one idea each (ask, race, one mistake, answer, try your
// numbers, next steps), on the real robot and look, fed by the real refund quick test.
// Every number is worked out by plain code from the saved run; the verdict word is the engine's.
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import type { Check, PriceTable, Verdict } from '../data/types.ts';
import { AGENT, modelLabel as baseLabel, monthlyCost, type ScopeScore } from '../flow/model.ts';
import { useActiveScene, useScenesInView } from '../hooks/useActiveScene.ts';
import { prefersReducedMotion, startSmoothWheel } from '../motion/flight.ts';
import type { Flyer } from '../motion/ribbon.ts';
import { startBrain } from '../motion/brain.ts';
import { hubName, matchLabels, matchScore, queryWords } from '../motion/webfield.ts';
import { answer, STARTERS, type Seg } from '../flow/helper.ts';
import { revealMood } from '../story/lines.ts';
import { runLog, type RunProgress } from '../flow/runlog.ts';
import { estimateText } from '../flow/estimate.ts';
import { count, pct, tableLabel } from '../story/format.ts';
import type { Mood, SceneDef } from '../story/scenes.ts';
import { Icon } from './Icons.tsx';
import { Reveal, ScrollRise, ScrollTiles, ScrollWrite, type Part } from './ScrollText.tsx';
import { CASE_KEYWORDS, UseCases, type AgentCheck, type MyAgent, type OwnUpload, type UseCase } from './UseCases.tsx';
import { Decode } from './Motion.tsx';
import { Robot } from './Robot.tsx';
import { Timeline } from './Timeline.tsx';

interface Turn {
  who: 'you' | 'guide';
  text: string;
  by?: 'model' | 'code';
  list?: Seg[][];
  chips?: string[];
}

/** A live quick test, driven by the main page (LiveApp). */
export interface LiveControl {
  canRun: boolean;
  state: 'idle' | 'running' | 'done' | 'failed';
  progress: RunProgress | null;
  /** when the reader pressed Run (ms), for the elapsed time in the robot's bubble */
  startedAt?: number;
  error: string | null;
  /** true once the page shows the reader's own result instead of the example */
  mine: boolean;
  start: (scope: string[], gap: number, oneRoute?: boolean) => void;
  /** measured times per use case, for the estimate before Run */
  speeds?: Record<string, { models_saved: boolean; fit_seconds: number | null; agent_seconds_per_case: number | null }>;
  /** the ready use cases (none preselected), the reader's choice and each one's practice sample */
  cases?: UseCase[];
  chosen?: string | null;
  choose?: (option: string | null) => void;
  libraries?: Record<string, MockRun['library']>;
  /** cases a test needs per gap ("0.10" -> 168), to grey out tests the data can't hold */
  planSizes?: Record<string, number>;
  canUpload?: boolean;
  upload?: (u: OwnUpload) => void;
  uploadState?: { busy: boolean; error: string | null; note: string | null };
  /** the reader's own agent; null = our test agent (llama3.1:8b on Ollama) races */
  agent?: MyAgent | null;
  setAgent?: (a: MyAgent | null) => void;
  checkAgent?: (url: string) => Promise<AgentCheck>;
}

// how big a difference the reader wants to be able to see (the test's planned gap)
const GAP_CHOICES: [number, string][] = [
  [0.2, 'a big one'],
  [0.1, 'a clear one'],
  [0.05, 'a small one'],
];

export interface MockRun {
  labels: string[];
  plan: {
    gap: number;
    planned_cases: number;
    comparator: string;
    population: number;
    early_stopping: { looks: number[] };
    scenario?: string;
    scope?: { labels?: string[]; one_route?: boolean } | null;
    /** who raced: the reference test agent (Ollama) or the user's own agent over HTTP */
    agent?: { kind: 'reference' | 'external'; model?: string; endpoint?: string; usd_per_message?: number };
  };
  scope: Record<string, ScopeScore>;
  verdict: Verdict;
  stream: { agent: boolean; simpler: boolean }[];
  price_tables: PriceTable[];
  scope_labels: string[];
  library: { text: string; label: string }[];
  messages: { text: string; true: string; near: boolean; nb: number[]; gold: string; agent: string | null; agent_ok: boolean; simpler: string | null; simpler_ok: boolean; agent_calls: number; agent_latency: number; agent_tokens: number }[];
}

const SCENES: SceneDef[] = [
  { id: 't0', name: 'Welcome', side: 'right', mood: 'hello' },
  { id: 'm1', name: 'Your agent', side: 'right', mood: 'hello' },
  { id: 'm1b', name: 'Your kinds', side: 'left', mood: 'curious' },
  { id: 'm2', name: 'The race', side: 'right', mood: 'thinking' },
  { id: 'm3', name: 'One mistake', side: 'left', mood: 'curious' },
  { id: 'm5', name: 'Your costs', side: 'left', mood: 'point' },
  { id: 'm5b', name: 'Your volume', side: 'right', mood: 'thinking' },
  { id: 'm5c', name: 'Three ways', side: 'left', mood: 'point' },
  { id: 'm4', name: 'Your answer', side: 'right', mood: 'proud' },
  { id: 'm6', name: 'Next', side: 'right', mood: 'hello' },
];
SCENES.forEach((sc, i) => (sc.side = i % 2 ? 'left' : 'right'));
const IDS = SCENES.map((s) => s.id);
const sideOf = (id: string) => SCENES.find((x) => x.id === id)!.side;

// the reader's starting numbers (all editable on screen 5)
const VOLUME0 = 10000;
// who fixes a wrong answer, and a typical hourly cost for them (the reader can change it)
const FIXERS: [string, string, number][] = [
  ['support', 'A support agent', 30],
  ['specialist', 'A specialist', 60],
  ['nobody', 'Nobody: the customer just waits', 0],
];
// the three tiers of the answer: plain models, heavier non-AI models, the AI agent
const TIER_BASIC = ['majority', 'linear', 'naive_bayes', 'linear_svm'];
const TIER_ADVANCED = ['random_forest', 'boosting'];
const TABLE0 = 'cloud_small';
// messages a month on a log slider: 1,000 to 1,000,000
const volFromSlider = (s: number) => Math.round(10 ** (3 + s / 100) / 100) * 100;
const sliderFromVol = (v: number) => (Math.log10(v) - 3) * 100;

const CHECK_SHORT: Record<string, string> = {
  leakage: 'No copies of practice messages in the test',
  power_cap: 'Enough messages for the difference you asked for',
  quality_floor: 'The agent used its tools properly',
  null_rate: 'Share of low-confidence messages looks normal',
  score_drift: 'Test messages look like the practice ones (scores)',
  domain_auc: 'Test messages look like the practice ones (words)',
  label_mix: 'Mix of kinds looks like practice',
  false_alarm_rate: 'The drift check itself rarely cries wolf',
  per_label_bias: 'No kind of message is singled out',
  flagged_split: 'Low-confidence messages behave like the rest',
};
const checkName = (n: string) => CHECK_SHORT[n] ?? n.replace(/[_[\]]+/g, ' ');
const kind = (l: string | null) =>
  l === 'my_kinds' ? 'one of your kinds' : l === 'not_mine' ? 'not one of your kinds' : (l ?? 'no answer').replace(/_/g, ' ').toLowerCase();

/** A number that counts up (with thousands commas) each time its screen comes into view. */
function Counter({ to, run, prefix = '', suffix = '' }: { to: number; run: boolean; prefix?: string; suffix?: string }) {
  const [n, setN] = useState(to);
  useEffect(() => {
    if (!run || prefersReducedMotion()) {
      setN(to);
      return;
    }
    let raf = 0;
    const t0 = performance.now() + 350;
    const step = (now: number) => {
      const p = Math.min(1, Math.max(0, (now - t0) / 1400));
      setN(to * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    setN(0);
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [to, run]);
  return (
    <>
      {prefix}
      {count(n)}
      {suffix}
    </>
  );
}

export function MockPage({ run, library, live, aiGuide = true }: { run: MockRun; library?: MockRun['library']; live?: LiveControl; aiGuide?: boolean }) {
  // while the reader types, the graph shows the use case their words fit best (until they pick one)
  const [descEarly, setDescEarly] = useState('');
  const bestCase = useMemo(() => {
    if (!live?.cases?.length || !live.libraries || !descEarly.trim()) return null;
    let best: string | null = null, top = 0;
    const q = new Set(queryWords(descEarly));
    for (const c of live.cases) {
      const lab = live.libraries[c.option] ?? [];
      const kinds = new Set(lab.map((m) => m.label)).size || 1;
      const keys = (CASE_KEYWORDS[c.option] ?? []).filter((k) => queryWords(k).some((w) => q.has(w))).length;
      // the use case's own words count most; label matches (averaged per message, scaled to the
      // number of kinds) break ties within them
      const perLabel = lab.length / kinds || 1;
      const sc = matchScore(descEarly, lab) / perLabel / Math.log(kinds + 1) + 3 * keys;
      if (sc > top) [best, top] = [c.option, sc];
    }
    return best;
  }, [descEarly, live?.cases, live?.libraries]);
  const lib = library ?? (bestCase ? live?.libraries?.[bestCase] : undefined) ?? run.library;
  const chosenCase = live?.cases?.find((c) => c.option === live.chosen) ?? null;
  const [gap, setGap] = useState(0.1);
  // "my kinds all go to the same place": a mix-up between two picked kinds counts as right
  const [oneRoute, setOneRoute] = useState(false);
  const running = live?.state === 'running';
  const active = useActiveScene(IDS);
  const inView = useScenesInView(IDS);
  const [desc, setDesc] = useState('');
  const [picked, setPicked] = useState<string[]>([]);
  const [showAll, setShowAll] = useState(false);
  const [details, setDetails] = useState(false);
  const [night, setNight] = useState(true);
  const v = run.verdict;
  const simpler = run.plan.comparator;
  // one name for the racing agent: ours by default (llama3.1:8b on Ollama), "Your agent" once the
  // reader plugs in their own; while a test runs, the one being tested
  const extRun = run.plan.agent?.kind === 'external';
  const AG = extRun ? 'Your agent' : 'Local Ollama agent';
  const ag = extRun ? 'your agent' : 'the local Ollama agent';
  const agWho = extRun ? `your agent at ${run.plan.agent?.endpoint ?? 'your address'}` : 'llama3.1:8b, running on this computer through Ollama';
  const modelLabel = (id: string) => (id === AGENT ? AG : baseLabel(id));
  const runningLabel = (id: string) => (id === AGENT ? (live?.agent ? 'Your agent' : 'Local Ollama agent') : baseLabel(id));
  const sugg = useMemo(() => matchLabels(desc, lib), [desc, lib]);
  const [allKinds, setAllKinds] = useState(false);
  const everyKind = useMemo(() => [...new Set(lib.map((m) => m.label))].sort((x, y) => x.localeCompare(y, 'en', { sensitivity: 'base' })), [lib]);
  const kindChips = allKinds ? [...sugg, ...everyKind.filter((l) => !sugg.includes(l))] : [...sugg, ...picked.filter((l) => !sugg.includes(l))];
  useEffect(() => setDescEarly(desc), [desc]);
  const handPicked = useRef(false);
  const handChosen = useRef(false);
  // the page re-renders often (the race counter, hovering), so the pause is keyed on the words only
  const liveRef = useRef(live);
  liveRef.current = live;
  useEffect(() => {
    if (handChosen.current || desc.trim().length < 4 || bestCase === liveRef.current?.chosen) return;
    const t = setTimeout(() => liveRef.current?.choose?.(bestCase), 900);
    return () => clearTimeout(t);
  }, [bestCase, desc]);
  useEffect(() => {
    handPicked.current = false;
    setAllKinds(false);
    setPicked([]);
  }, [live?.chosen]);
  useEffect(() => {
    if (!handPicked.current && live?.chosen) setPicked(sugg);
  }, [sugg, live?.chosen]);
  const canRoute = picked.length >= 2 && (!chosenCase || picked.length < chosenCase.labels);
  const fits = (g: number) => !chosenCase?.evalSize || (live?.planSizes?.[g.toFixed(2)] ?? 0) <= chosenCase.evalSize;
  useEffect(() => {
    if (!fits(gap)) {
      const ok = GAP_CHOICES.map(([g]) => g).filter(fits).sort((a, b) => a - b)[0];
      if (ok !== undefined) setGap(ok);
    }
  });
  const descRef = useRef(desc);
  descRef.current = desc;
  const cv = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ kind: 'run' | 'lib'; index: number; x: number; y: number } | null>(null);
  // the guide's conversation, kept in a chat panel that opens from a small button
  const [ask, setAsk] = useState('');
  const [chat, setChat] = useState<Turn[]>([]);
  const [chatOpen, setChatOpen] = useState(false);
  const [thinking, setThinking] = useState(false);
  const chatEnd = useRef<HTMLDivElement>(null);
  const focusRef = useRef<number[]>([]);
  const [typing, setTyping] = useState(false);
  const typingTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // the reader's numbers
  const [volume, setVolume] = useState(VOLUME0);
  const [fixer, setFixer] = useState('support');
  const [hourly, setHourly] = useState(30);
  const [minutes, setMinutes] = useState(5);
  // one mistake = the fixer's hourly cost x minutes to fix / 60 (plain code)
  const mistake = (hourly * minutes) / 60;
  const [table, setTable] = useState(TABLE0);

  // screen 3: the messages worth a close-up (either side wrong), starting with the agent's first slip
  const mistakes = useMemo(() => run.messages.map((m, i) => (!m.agent_ok || !m.simpler_ok ? i : -1)).filter((i) => i >= 0), [run.messages]);
  const [pick, setPick] = useState(() => Math.max(0, run.messages.findIndex((m) => !m.agent_ok)));
  useEffect(() => setPick(Math.max(0, run.messages.findIndex((m) => !m.agent_ok))), [run]);
  const pickRef = useRef(pick);
  pickRef.current = pick;
  const fm = run.messages[pick];
  // screen 2: the counter follows the dots the background has scored
  const [raced, setRaced] = useState(0);
  const nightRef = useRef(night);
  nightRef.current = night;

  useEffect(() => (prefersReducedMotion() ? undefined : startSmoothWheel()), []);
  useEffect(() => {
    if (!cv.current) return;
    return startBrain(cv.current, {
      library: lib,
      query: () => descRef.current,
      focus: () => focusRef.current,
      messages: run.messages,
      featured: () => pickRef.current,
      onPick: (i) => {
        setPick(i);
        const here = document.getElementById('m3')?.getBoundingClientRect();
        if (here && Math.abs(here.top) > window.innerHeight / 2) document.getElementById('m3')?.scrollIntoView({ behavior: 'smooth' });
      },
      onRace: setRaced,
      onHover: setHover,
      still: prefersReducedMotion(),
      isNight: () => nightRef.current,
      // screens 1-2 (agent, kinds) are the graph's typing step; then race, close-up, answer as before
      // title and agent screens are the graph's quiet typing step; kinds eases toward the race
      remap: (q) => (q <= 1 ? 0 : q <= 2 ? (q - 1) * 0.3 : q <= 3 ? 0.3 + (q - 2) * 0.7 : q <= 4 ? q - 2 : q <= 8 ? 2 + (q - 4) / 4 : 3 + (q - 8)),
    });
  }, [run, lib, v.verdict]);
  const flyerRef = useRef<Flyer | null>(null);
  const onFly = useCallback((f: Flyer | null) => {
    flyerRef.current = f;
  }, []);

  const total = run.stream.length;
  const agentRight = run.stream.filter((s) => s.agent).length;
  const simplerRight = run.stream.filter((s) => s.simpler).length;
  const agentSoFar = run.stream.slice(0, raced).filter((s) => s.agent).length;
  const simplerSoFar = run.stream.slice(0, raced).filter((s) => s.simpler).length;
  const raceDone = raced >= total;

  const points = v.arms
    .filter((a) => run.scope[a.name] && a.name !== 'cascade')
    .map((a) => ({ name: a.name, acc: a.accuracy, right: Math.round(a.accuracy * v.n_cases), cost: monthlyCost(a, table, volume, mistake) }));
  const ranked = [...points].sort((a, b) => a.cost - b.cost || b.acc - a.acc);
  const top = new Set(ranked.slice(0, 3).map((p) => p.name).concat(AGENT));
  const shown = showAll ? points : points.filter((p) => top.has(p.name));
  const costs = shown.map((p) => p.cost);
  const accs = shown.map((p) => p.acc);
  const c0 = Math.min(...costs) * 0.85, c1 = Math.max(...costs) * 1.08;
  const a0 = Math.max(0, Math.min(...accs) - 0.03), a1 = Math.min(1, Math.max(...accs) + 0.01);
  const agentArm = v.arms.find((a) => a.name === AGENT);
  const perMsg = (t: string) => {
    const c = agentArm?.inference_cost[t] ?? 0;
    return c >= 0.01 ? c.toFixed(3) : c.toPrecision(2);
  };
  const agentCost = points.find((p) => p.name === AGENT)?.cost ?? 0;
  const simplerCost = points.find((p) => p.name === simpler)?.cost ?? 0;
  const saving = agentCost - simplerCost;
  const cheapest = (names: string[]) => points.filter((x) => names.includes(x.name)).sort((a, b) => a.cost - b.cost || b.acc - a.acc)[0];
  const tiers = [
    { tier: 'Tier 1 · basic model', p: cheapest(TIER_BASIC), cls: 'simpler' },
    { tier: 'Tier 2 · advanced model', p: cheapest(TIER_ADVANCED), cls: 'adv' },
    { tier: 'Tier 3 · AI agent', p: points.find((x) => x.name === AGENT), cls: 'agent' },
  ].filter((t) => t.p);
  const maxTier = Math.max(1, ...tiers.map((t) => t.p!.cost));
  const passed = v.checks.filter((c: Check) => c.passed).length;
  const failed = v.checks.filter((c: Check) => !c.passed && c.effect !== 'REPORT');
  const noted = v.checks.filter((c: Check) => !c.passed && c.effect === 'REPORT');
  const grabbed = (x?: ScopeScore) => Math.round((x?.false_grab_rate ?? 0) * (x?.other_cases ?? 0));
  const inRight = (x?: ScopeScore) => Math.round((x?.in_scope_accuracy ?? 0) * (x?.in_scope_cases ?? 0));
  const stopLook = run.plan.early_stopping.looks.indexOf(v.n_cases) + 1;
  const proven = v.accuracy_diff_ci[0] > 0;
  // why the agent lost, in plain code: fewer of the reader's messages right, or (just as right)
  // it takes messages that are not the reader's
  const sA = run.scope[AGENT] as ScopeScore | undefined, sS = run.scope[simpler] as ScopeScore | undefined;
  const lessAccurate = !!sA && !!sS && sA.in_scope_accuracy < sS.in_scope_accuracy;
  const grabsMore = !!sA && !!sS && !lessAccurate && sA.other_cases > 0 && sA.false_grab_rate > sS.false_grab_rate;
  const costPart: Part[] = saving > 0 ? [', and it costs ', { b: `$${count(saving)} more a month` }] : [];
  const reason: Part[] =
    v.verdict === 'USE_SIMPLER'
      ? [
          ...(grabsMore
            ? ([
                `${AG} is as good on your messages, but it wrongly takes `,
                { b: `${grabbed(sA)} of ${sA!.other_cases}` },
                ` messages that aren’t yours (${modelLabel(simpler)}: ${grabbed(sS)})`,
              ] as Part[])
            : lessAccurate
              ? [`${AG} gets fewer of your messages right`]
              : [`${AG} isn’t enough better on your messages`]),
          ...costPart,
          '.',
        ]
      : v.verdict === 'INVALID'
        ? ['This run can’t pick a winner: a safety check failed (', { b: failed.map((c) => checkName(c.name)).join('; ') || 'see the details' }, '). Run it again with more messages, or on other kinds.']
        : [
            'Neither is proven better on these ',
            { b: `${count(v.n_cases)} messages` },
            `: the agent’s gain is somewhere between ${(v.accuracy_diff_ci[0] * 100).toFixed(0)} and ${(v.accuracy_diff_ci[1] * 100).toFixed(0)} points. Test more messages to decide.`,
          ];
  const word = v.verdict === 'USE_SIMPLER' ? `Use ${modelLabel(simpler)}` : v.verdict === 'DEPLOY_AGENT' ? `Use ${ag}` : v.verdict === 'NEED_MORE_DATA' ? 'Too close to call' : 'Invalid run';

  const helperCtx = {
    messages: run.messages,
    arms: v.arms,
    scope: run.scope,
    verdict: v.verdict,
    simpler,
    gap: run.plan.gap,
    ci: v.accuracy_diff_ci,
    nCases: v.n_cases,
    planned: run.plan.population,
    looks: run.plan.early_stopping.looks,
    // the guide quotes the same numbers the reader set on screen 5
    volume,
    mistake,
    table,
    label: modelLabel,
  };
  const flat = (list: Seg[]) => list.map((x) => (typeof x === 'string' ? x : x.b)).join('');
  // every fact the guide may use, worked out by plain code from the run (the model only words it)
  const factsFor = (q: string) => {
    const topics = [
      'Which messages did my agent get wrong?',
      `What did ${modelLabel(simpler)} get wrong?`,
      'What do these have in common?',
      'How sure is this?',
      'Break it down by kind',
      'How often does it grab messages that aren’t mine?',
      'Should I deploy?',
      'What does it cost a month?',
      q,
    ];
    const lines = [
      `Kinds of message the agent handles: ${picked.length ? picked.join(', ') : run.scope_labels.join(', ')}.`,
      `Simpler model it is compared with: ${modelLabel(simpler)}.`,
    ];
    for (const t of topics) {
      const a = answer(t, helperCtx);
      lines.push(flat(a.text) + ' ' + (a.list ?? []).map((l) => flat(l)).join(' '));
    }
    return lines.join('\n');
  };
  const askGuide = async (q: string) => {
    if (!q.trim() || thinking) return;
    const a = answer(q, helperCtx);
    if (live && !live.mine)
      a.text = [
        running ? 'Your test is still running, so this is about the example run (a bank refund agent): ' : 'You haven’t run a test yet, so this is about the example run (a bank refund agent): ',
        ...a.text,
      ];
    focusRef.current = a.focus ?? [];
    setChatOpen(true);
    setAsk('');
    setChat((c) => [...c, { who: 'you', text: q }]);
    if (a.go && a.go !== active) setTimeout(() => document.getElementById(a.go as string)?.scrollIntoView({ behavior: 'smooth' }), 250);
    setThinking(true);
    const plain = flat(a.text) + (a.list ? ' ' + a.list.map((l) => flat(l)).join(' ') : '');
    let reply = plain, by: 'model' | 'code' = 'code';
    // the main page keeps the locked rule (no AI-written text) until the owner approves the guide
    if (aiGuide) try {
      const r = await fetch('/api/guide', {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-verifier-client': '1' },
        body: JSON.stringify({ question: q, description: desc, facts: factsFor(q), fallback: plain }),
      });
      if (r.ok) {
        const out = (await r.json()) as { reply: string; worded_by: 'model' | 'code' };
        reply = out.reply;
        by = out.worded_by;
      }
    } catch {
      /* no server: the plain answer stands */
    }
    setThinking(false);
    setChat((c) => [...c, by === 'code' ? { who: 'guide', text: flat(a.text), by, list: a.list, chips: a.chips } : { who: 'guide', text: reply, by, chips: a.chips }]);
  };
  useEffect(() => {
    const log = chatEnd.current?.parentElement;
    if (log) log.scrollTop = log.scrollHeight;
  }, [chat, thinking]);
  const segs = (list: Seg[]) => list.map((x, i) => (typeof x === 'string' ? <span key={i}>{x}</span> : <b key={i}>{x.b}</b>));
  const NEXT: Record<string, [string, string]> = {
    t0: ['m1', 'Let’s start'],
    m1: ['m1b', 'Pick my kinds'],
    m1b: ['m2', 'See the race'],
    m2: ['m3', 'Zoom into a mistake'],
    m3: ['m5', 'Now, your costs'],
    m5: ['m5b', 'Next: your volume'],
    m5b: ['m5c', 'Compare the three'],
    m5c: ['m4', 'Show my answer'],
    m4: ['m6', 'What next'],
  };
  const lastGuide = [...chat].reverse().find((t) => t.who === 'guide');
  const suggestions = lastGuide?.chips ?? STARTERS;
  const steps = running ? <RunSteps progress={live?.progress} startedAt={live?.startedAt} name={runningLabel} /> : null;
  const controls = () => (
    <>
      {steps}
      {nextButton()}
    </>
  );
  const nextButton = () =>
    NEXT[active] && (active !== 'm1' || picked.length > 0) ? (
      <div className="chips">
        <a href={`#${NEXT[active][0]}`} className="chip small primary">
          {NEXT[active][1]} ↓
        </a>
      </div>
    ) : null;

  // the robot reacts: it watches the race, leans to the leader, shrugs or cheers at the answer
  const mood: Mood =
    active === 'm2'
      ? raceDone
        ? 'nod'
        : 'thinking'
      : active === 'm4'
        ? revealMood(v.accuracy_diff, v.verdict)
        : SCENES.find((s) => s.id === active)!.mood;

  const line = () => {
    if (thinking) return <>Let me look at your run…</>;
    if (running) return <>{runLog(live?.progress, runningLabel).headline}</>;
    if (live?.state === 'failed' && (active === 'm1' || active === 'm1b' || active === 'm2')) return <>That test didn’t work: {live.error}</>;
    if (live?.state === 'done' && (active === 'm1' || active === 'm1b')) return <>Done! Your result is ready. The next screens now show your run.</>;
    if (active === 'm2')
      return raceDone ? (
        <>
          Done. {modelLabel(simpler)} got <b>{simplerRight}</b>, {ag} <b>{agentRight}</b>.
        </>
      ) : (
        <>Scoring every message… {agentSoFar >= simplerSoFar ? ag : modelLabel(simpler)} is ahead.</>
      );
    if (active === 'm3') return <>Tap any dot to see that message.</>;
    if (active === 'm4')
      return v.verdict === 'USE_SIMPLER' ? (
        <>{AG} didn’t earn its cost here.</>
      ) : v.verdict === 'DEPLOY_AGENT' ? (
        <>{AG} earned it!</>
      ) : v.verdict === 'INVALID' ? (
        <>A safety check failed, so I can’t name a winner from this run.</>
      ) : (
        <>Too close to call yet.</>
      );
    if (active === 't0') return <>Hi! I’m your guide. I’ll walk you through testing an AI agent, one step at a time.</>;
    if (active === 'm1b') return chosenCase ? <>Tap the kinds your agent handles. I’ll test only those.</> : <>Pick a use case first.</>;
    if (active === 'm5') return <>Who fixes a wrong answer? That’s what a mistake costs you.</>;
    if (active === 'm5b') return <>How many messages, and where the AI runs, set the bill.</>;
    if (active === 'm5c') return proven ? <>Here’s each way, priced with your numbers. {AG} is proven more accurate, so the price of a mistake matters.</> : <>Here’s each way, priced with your numbers. Your answer comes from the test; these prices don’t change it.</>;
    if (active === 'm6') return <>Take the evidence with you.</>;
    if (!desc.trim()) return <>Tell me what your agent does. I’ll light up the messages that match.</>;
    return sugg.length ? <>I found <b>{sugg.length} matching kinds</b> of message. Keep the ones that are yours.</> : <>No match yet. Try words your customers use.</>;
  };

  const stepMistake = (dir: number) => {
    const k = mistakes.indexOf(pick);
    const next = mistakes[(k < 0 ? 0 : k + dir + mistakes.length) % mistakes.length];
    if (next !== undefined) setPick(next);
  };

  return (
    <div className={`page flow mock mock5${night ? ' night' : ''}`} data-scene={active}>
      <div className="bg" aria-hidden="true">
        <div className="bg-night" />
        <canvas ref={cv} />
      </div>
      <header className="top">
        <span className="brand glass">
          Agent Verifier<span className="t-teal">.</span>
        </span>
        <span className="live glass">
          {live?.mine ? `Your result · ${live.cases?.find((c) => c.option === run.plan.scenario)?.title ?? 'exploratory'}` : running ? (live?.agent ? 'Testing your agent…' : 'Testing with the local Ollama agent…') : chosenCase ? `${chosenCase.title} · not run yet` : 'Example · a refund agent'}
        </span>
        <div className="top-actions">
          <button type="button" className="pill glass" aria-pressed={night} onClick={() => setNight(!night)}>
            {night ? 'Day' : 'Night'}
          </button>
        </div>
      </header>
      <Timeline active={active} scenes={SCENES} counted />

      <main>
        {/* 0 · Title: the product's name, what it does, and the guide flying in */}
        <section id="t0" tabIndex={-1} className={`scene robot-${sideOf('t0')} title-scene${inView.has('t0') ? ' in' : ''}`}>
          <span className="eyebrow title-eyebrow">An honest test for AI agents</span>
          <ScrollTiles as="h1" className="title-word grad" text="Agent Verifier" intro />
          <Reveal className="title-sub" items=".title-lead, .title-steps > *, .title-go" intro>
            <p className="lead title-lead">Is your AI agent worth what it costs? We race it against simpler models on real messages and tell you, in plain numbers, which one to use.</p>
            <div className="title-steps mono">
              <span>
                <b>1</b> Tell us what it does
              </span>
              <span>
                <b>2</b> Watch the race
              </span>
              <span>
                <b>3</b> Get your answer
              </span>
            </div>
            <a href="#m1" className="chip primary title-go">
              Start ↓
            </a>
          </Reveal>
        </section>

        {/* 1 · Ask */}
        <section id="m1" tabIndex={-1} className={`scene robot-${sideOf('m1')}${inView.has('m1') ? ' in' : ''}`}>
          <Decode className="eyebrow beat" text="01 · Your agent" run={inView.has('m1')} instantWhenStill />
          <ScrollTiles as="h2" className="mid" text="What does your agent do?" />
          <p className="lead muted agent-note">
            {live?.agent ? (
              <>We’ll race <b>your agent</b> at {live.agent.url.replace(/\?.*$/, '')}.</>
            ) : (
              <>
                The agent in this test is a <b>local Ollama model</b> (llama3.1:8b, running on this computer), standing in for yours. Have your own? Use{' '}
                <b>Bring your agent</b> below.
              </>
            )}
          </p>
          <label className="field hero-field">
            <span className="sr-only">What does your agent do?</span>
            <input
              value={desc}
              placeholder="e.g. handles refund requests"
              onChange={(e) => {
                setDesc(e.target.value);
                setTyping(true);
                if (typingTimer.current) clearTimeout(typingTimer.current);
                typingTimer.current = setTimeout(() => setTyping(false), 900);
              }}
            />
          </label>
          {live?.cases?.length ? (
            <UseCases
              cases={live.cases}
              chosen={live.chosen ?? null}
              best={bestCase}
              onChoose={(o) => {
                handChosen.current = true;
                live.choose?.(o);
                setTimeout(() => document.getElementById('m1b')?.scrollIntoView({ behavior: 'smooth' }), 350);
              }}
              canUpload={!!live.canUpload}
              onUpload={(u) => live.upload?.(u)}
              uploadState={live.uploadState ?? { busy: false, error: null, note: null }}
              agent={live.agent ?? null}
              setAgent={live.setAgent}
              checkAgent={live.checkAgent}
            />
          ) : null}
        </section>

        {/* 2 · Your kinds: which messages are the agent's, and how big a difference to look for */}
        <section id="m1b" tabIndex={-1} className={`scene robot-${sideOf('m1b')}${inView.has('m1b') ? ' in' : ''}`}>
          <Decode className="eyebrow beat" text="02 · Your kinds" run={inView.has('m1b')} instantWhenStill />
          <ScrollRise className="mid" text="Which messages are yours?" />
          {desc.trim() && chosenCase && bestCase === chosenCase.option ? (
            <p className="lead said">
              For “{desc.trim()}”, the closest use case is <b>{chosenCase.title}</b>.
            </p>
          ) : desc.trim() && !bestCase ? (
            <p className="lead said">
              None of our use cases matches “{desc.trim()}” yet. Pick the nearest one above, or bring your own data.
            </p>
          ) : null}
          {live?.cases?.length && !chosenCase ? (
            <p className="lead muted">Pick a use case above first, or bring your own data.</p>
          ) : null}
          {(live?.cases?.length ? !!chosenCase : true) && (
            <Reveal className="suggest glass" items=".suggest-head, .chips > *, .run-row">
              <div className="suggest-head">
                <Icon name={chosenCase && !chosenCase.own ? chosenCase.option : 'banking77'} />
                <div>
                  <b>{chosenCase ? chosenCase.title : 'Bank support routing'}</b>
                  <small className="mono">
                    {sugg.length ? `${sugg.length} of ${chosenCase?.labels ?? run.labels.length} kinds match your words` : 'Which kinds of message does your agent handle?'}
                  </small>
                </div>
                <button type="button" className="chip small primary" disabled={!sugg.length} onClick={() => setPicked(sugg)}>
                  Use these
                </button>
                {chosenCase && (
                  <button
                    type="button"
                    className="chip small"
                    onClick={() => {
                      handPicked.current = true;
                      setAllKinds(true);
                      setPicked(everyKind);
                    }}
                  >
                    All {chosenCase.labels}
                  </button>
                )}
              </div>
              <div className={`chips${allKinds ? ' all-kinds' : ''}`}>
                {kindChips.map((l, i) => (
                  <button
                    key={l}
                    type="button"
                    style={{ '--i': i } as CSSProperties}
                    className={`chip small mono pop${picked.includes(l) ? ' on' : ''}`}
                    aria-pressed={picked.includes(l)}
                    onClick={() => {
                      handPicked.current = true;
                      setPicked(picked.includes(l) ? picked.filter((x) => x !== l) : [...picked, l]);
                    }}
                  >
                    {kind(l)}
                  </button>
                ))}
              </div>
              <small className="mono muted">
                {picked.length ? `${picked.length} picked` : sugg.length ? 'Tap the kinds that are yours' : 'Describe your agent above, or take all of them'}
                {everyKind.length > sugg.length && (
                  <>
                    {' · '}
                    <button type="button" className="link" onClick={() => setAllKinds(!allKinds)}>
                      {allKinds ? 'show only the matches' : `show all ${everyKind.length} kinds`}
                    </button>
                  </>
                )}
              </small>
              {live?.canRun && picked.length > 0 && (
                <div className="run-row">
                  <span className="mono muted">Spot a difference that’s</span>
                  <div className="chips">
                    {GAP_CHOICES.map(([g, name]) => (
                      <button key={g} type="button" className={`chip small${gap === g ? ' on' : ''}`} onClick={() => setGap(g)} disabled={running || !fits(g)} title={fits(g) ? '' : 'Not enough messages in this data for so small a difference'}>
                        {name} · {Math.round(g * 100)} pts
                      </button>
                    ))}
                  </div>
                  <button
                    type="button"
                    className="chip primary run-btn"
                    disabled={running || !fits(gap)}
                    onClick={() => {
                      live.start(picked, gap, canRoute && oneRoute);
                      setTimeout(() => document.getElementById('m2')?.scrollIntoView({ behavior: 'smooth' }), 300);
                    }}
                  >
                    {running ? 'Testing…' : live.mine && run.plan.scenario === chosenCase?.option ? 'Run it again' : 'Run my test'}
                  </button>
                  {canRoute && (
                    <label className="route-pick">
                      <input type="checkbox" checked={oneRoute} onChange={(e) => setOneRoute(e.target.checked)} disabled={running} />
                      <span>
                        These kinds all go to the same place <small className="muted">(a mix-up between two of them counts as right, for every model)</small>
                      </span>
                    </label>
                  )}
                  {chosenCase && !running && fits(gap) && (
                    <small className="mono muted estimate">
                      ⏱{' '}
                      {estimateText({
                        cases: live.planSizes?.[gap.toFixed(2)] ?? 0,
                        modelsSaved: !!live.speeds?.[chosenCase.option]?.models_saved,
                        fitSeconds: live.speeds?.[chosenCase.option]?.fit_seconds ?? null,
                        agentSeconds: live.agent ? (live.agent.secs ?? null) : (live.speeds?.[chosenCase.option]?.agent_seconds_per_case ?? null),
                      })}
                    </small>
                  )}
                  <small className="mono muted racer">
                    Racing: {live.agent ? `your agent (${live.agent.url.replace(/\?.*$/, '')}, $${live.agent.usd}/answer)` : 'a local Ollama model (llama3.1:8b on this computer) standing in for your agent'} ·{' '}
                    <button type="button" className="link" onClick={() => document.getElementById('m1')?.scrollIntoView({ behavior: 'smooth' })}>
                      change
                    </button>
                  </small>
                  {!GAP_CHOICES.some(([g]) => fits(g)) && (
                    <small className="run-error">
                      This data leaves only {chosenCase?.evalSize} messages to test on; even a 20-point difference needs {live.planSizes?.['0.20'] ?? 39}. Add more rows.
                    </small>
                  )}
                  {live.state === 'failed' && <small className="run-error">{live.error}</small>}
                </div>
              )}
            </Reveal>
          )}
        </section>

        {/* 3 · The race: the counter ticks with the dots */}
        <section id="m2" tabIndex={-1} className={`scene robot-${sideOf('m2')}${inView.has('m2') ? ' in' : ''}`}>
          <Decode className="eyebrow beat" text="03 · The race" run={inView.has('m2')} instantWhenStill />
          {live && !live.mine ? (
            <div className="example-note mono">
              <span>
                Example run · a bank refund agent{chosenCase && chosenCase.option !== 'banking77' ? ` · yours (${chosenCase.title}) appears here after you run it` : ' · yours appears here after you run it'}
              </span>
              <button type="button" className="chip small" onClick={() => document.getElementById('m1b')?.scrollIntoView({ behavior: 'smooth' })}>
                Run mine ↑
              </button>
            </div>
          ) : null}
          <ScrollRise className="mid" text="Every message, checked." />
          {running && live?.progress && (
            <div className="live-progress glass">
              <span className="chat-dot" />
              <div>
                <b>{runLog(live.progress, runningLabel).steps.at(-1)?.text}</b>
                <small className="mono muted">{live.mine ? 'the screens below show your last run until this one finishes' : 'the race below is the example until yours finishes'}</small>
              </div>
            </div>
          )}
          <div className="duel race-duel">
            <div className={raceDone && agentRight > simplerRight ? 'lead' : ''}>
              <span className="mono">{AG}</span>
              <small className="mono muted who">{agWho}</small>
              <b>
                {agentSoFar}
                <small>/{total}</small>
              </b>
            </div>
            <div className="vs">vs</div>
            <div className={raceDone && simplerRight > agentRight ? 'lead' : ''}>
              <span className="mono">{modelLabel(simpler)}</span>
              <b>
                {simplerSoFar}
                <small>/{total}</small>
              </b>
            </div>
          </div>
          <div className="race-bar" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={raced} aria-label="Messages scored">
            <i style={{ width: `${(100 * raced) / total}%` }} />
          </div>
          <small className="mono muted">{raceDone ? `all ${total} messages scored` : `scoring message ${raced} of ${total}`}</small>
        </section>

        {/* 3 · One mistake: any dot can be picked */}
        <section id="m3" tabIndex={-1} className={`scene robot-${sideOf('m3')}${inView.has('m3') ? ' in' : ''}`}>
          <Decode className="eyebrow beat" text="04 · One mistake" run={inView.has('m3')} instantWhenStill />
          {live && !live.mine ? (
            <div className="example-note mono">
              <span>
                Example run · a bank refund agent{chosenCase && chosenCase.option !== 'banking77' ? ` · yours (${chosenCase.title}) appears here after you run it` : ' · yours appears here after you run it'}
              </span>
              <button type="button" className="chip small" onClick={() => document.getElementById('m1b')?.scrollIntoView({ behavior: 'smooth' })}>
                Run mine ↑
              </button>
            </div>
          ) : null}
          {fm && (
            <Reveal className="closeup glass" items=".closeup-text, .closeup-rows > div, .closeup-gold, .closeup-nav">
              <p className="closeup-text">“{fm.text}”</p>
              <div className="closeup-rows">
                <div className={fm.agent_ok ? 'ok' : 'no'}>
                  <span>
                    {AG} <small className="mono muted">· {extRun ? 'your agent' : 'llama3.1:8b via Ollama'}</small>
                  </span>
                  <b className="mono">{fm.agent === 'not_mine' ? 'not one of your kinds' : kind(fm.agent)}</b>
                  <small>
                    {fm.agent_ok ? 'right' : 'wrong'} ·{' '}
                    {extRun ? 'one call' : `${fm.agent_calls} tool ${fm.agent_calls === 1 ? 'call' : 'calls'}`} ·{' '}
                    {fm.agent_latency < 0.1 ? '<0.1' : fm.agent_latency.toFixed(1)} s
                  </small>
                </div>
                <div className={fm.simpler_ok ? 'ok' : 'no'}>
                  <span>{modelLabel(simpler)}</span>
                  <b className="mono">{fm.simpler === 'not_mine' ? 'not one of your kinds' : kind(fm.simpler)}</b>
                  <small>{fm.simpler_ok ? 'right' : 'wrong'} · instant · free</small>
                </div>
              </div>
              <small className="mono muted closeup-gold">
                correct answer: {fm.gold === 'not_mine' ? `not one of your kinds (it’s “${kind(fm.true)}”${fm.near ? ', a close neighbour' : ''})` : kind(fm.gold)}
              </small>
              <div className="closeup-nav">
                <button type="button" className="chip small" onClick={() => stepMistake(-1)}>
                  ‹ Previous
                </button>
                <small className="mono muted">
                  {mistakes.includes(pick) ? `mistake ${mistakes.indexOf(pick) + 1} of ${mistakes.length}` : 'both right on this one'}
                </small>
                <button type="button" className="chip small" onClick={() => stepMistake(1)}>
                  Next mistake ›
                </button>
              </div>
            </Reveal>
          )}
        </section>

        {/* 5 · Try your numbers: the costs move live; the verdict word stays the engine's */}
        <section id="m5" tabIndex={-1} className={`scene robot-${sideOf('m5')}${inView.has('m5') ? ' in' : ''}`}>
          <Decode className="eyebrow beat" text="05 · Your costs" run={inView.has('m5')} instantWhenStill />
          {live && !live.mine ? (
            <div className="example-note mono">
              <span>
                Example run · a bank refund agent{chosenCase && chosenCase.option !== 'banking77' ? ` · yours (${chosenCase.title}) appears here after you run it` : ' · yours appears here after you run it'}
              </span>
              <button type="button" className="chip small" onClick={() => document.getElementById('m1b')?.scrollIntoView({ behavior: 'smooth' })}>
                Run mine ↑
              </button>
            </div>
          ) : null}
          <ScrollRise className="mid" text="What’s a mistake worth to you?" />
          <Reveal className="tryit glass" items=".ask-q, .ask-two .slider, .mistake-is">
            <div className="ask-q">
              <span>Who fixes a wrong answer?</span>
              <div className="chips">
                {FIXERS.map(([k, name, rate]) => (
                  <button
                    key={k}
                    type="button"
                    className={`chip small${fixer === k ? ' on' : ''}`}
                    onClick={() => {
                      setFixer(k);
                      setHourly(rate);
                    }}
                  >
                    {name}
                  </button>
                ))}
              </div>
            </div>
            {fixer !== 'nobody' && (
              <div className="ask-two">
                <label className="slider">
                  <span>
                    They cost <b>${hourly}</b> an hour
                  </span>
                  <input type="range" min={10} max={200} step={5} value={hourly} onChange={(e) => setHourly(Number(e.target.value))} />
                </label>
                <label className="slider">
                  <span>
                    A fix takes <b>{minutes} min</b>
                  </span>
                  <input type="range" min={1} max={60} step={1} value={minutes} onChange={(e) => setMinutes(Number(e.target.value))} />
                </label>
              </div>
            )}
            <p className="mistake-is mono">
              = <b>${mistake.toFixed(2)}</b> per mistake{fixer === 'nobody' ? ' (no one is paid to fix it; the cost is your customer’s time)' : ''}
            </p>
          </Reveal>
        </section>

        {/* 7 · Your volume and who runs the AI */}
        <section id="m5b" tabIndex={-1} className={`scene robot-${sideOf('m5b')}${inView.has('m5b') ? ' in' : ''}`}>
          <Decode className="eyebrow beat" text="06 · Your volume" run={inView.has('m5b')} instantWhenStill />
          {live && !live.mine ? (
            <div className="example-note mono">
              <span>
                Example run · a bank refund agent{chosenCase && chosenCase.option !== 'banking77' ? ` · yours (${chosenCase.title}) appears here after you run it` : ' · yours appears here after you run it'}
              </span>
              <button type="button" className="chip small" onClick={() => document.getElementById('m1b')?.scrollIntoView({ behavior: 'smooth' })}>
                Run mine ↑
              </button>
            </div>
          ) : null}
          <ScrollRise className="mid" text="How busy is it, and who runs the AI?" />
          <Reveal className="tryit glass" items=".slider, .price-q, .tryit .chips > *">
            <label className="slider">
              <span>
                Messages a month <b>{count(volume)}</b>
              </span>
              <input type="range" min={0} max={300} step={1} value={sliderFromVol(volume)} onChange={(e) => setVolume(volFromSlider(Number(e.target.value)))} />
            </label>
            <p className="price-q">
              {extRun ? 'What your agent costs' : 'Where does the AI model run?'}{' '}
              <small className="muted">{extRun ? 'Your own figure, per answer.' : 'Prices are assumed averages, per message.'}</small>
            </p>
            <div className="chips">
              {extRun ? (
                <small className="mono muted">Your agent: ${run.plan.agent?.usd_per_message} per answer (your figure), whichever way it runs.</small>
              ) : (
                run.price_tables.map((p) => (
                  <button key={p.name} type="button" className={`chip small${p.name === table ? ' on' : ''}`} onClick={() => setTable(p.name)}>
                    {tableLabel(p.name)}
                    <small className="mono"> · ${perMsg(p.name)}/msg</small>
                  </button>
                ))
              )}
            </div>
          </Reveal>
        </section>

        {/* 8 · The three ways to do it, priced with the reader's numbers */}
        <section id="m5c" tabIndex={-1} className={`scene robot-${sideOf('m5c')}${inView.has('m5c') ? ' in' : ''}`}>
          <Decode className="eyebrow beat" text="07 · Three ways" run={inView.has('m5c')} instantWhenStill />
          {live && !live.mine ? (
            <div className="example-note mono">
              <span>
                Example run · a bank refund agent{chosenCase && chosenCase.option !== 'banking77' ? ` · yours (${chosenCase.title}) appears here after you run it` : ' · yours appears here after you run it'}
              </span>
              <button type="button" className="chip small" onClick={() => document.getElementById('m1b')?.scrollIntoView({ behavior: 'smooth' })}>
                Run mine ↑
              </button>
            </div>
          ) : null}
          <ScrollRise className="mid" text="Three ways to do it." />
          <Reveal className="tryit glass" items=".costbar, .tiers > small, .tryit-out">
            <div className="costbars tiers">
              {tiers.map((t) => (
                <div key={t.tier} className={`costbar ${t.cls}`}>
                  <span>
                    <em className="mono">{t.tier}</em>
                    {modelLabel(t.p!.name)} · {t.p!.right} of {v.n_cases} right
                  </span>
                  <i style={{ width: `${(100 * t.p!.cost) / maxTier}%` }} />
                  <b>${count(t.p!.cost)}</b>
                </div>
              ))}
              <small className="mono muted">
                a month: model costs plus mistakes, cheapest of each tier. The answer compares {ag} with {modelLabel(simpler)}, picked beforehand on separate tuning messages so the test stays fair.
              </small>
            </div>
            <p className="tryit-out">
              {proven ? (
                <>{AG} is proven more accurate, so a high enough mistake cost makes it worth it.</>
              ) : (
                <>
                  {v.accuracy_diff_ci[1] < 0 ? `${AG} is less accurate here (proven on these messages)` : `${AG} isn’t proven more accurate here`}, so no mistake cost changes the answer.
                  {agentCost < simplerCost ? ' It looks cheaper at these numbers, but that gap is within chance.' : ''}
                </>
              )}
            </p>
                    </Reveal>
        </section>

        {/* 8 · The answer, after the reader's own costs: one word, three numbers, nothing else */}
        <section id="m4" tabIndex={-1} className={`scene robot-${sideOf('m4')}${inView.has('m4') ? ' in' : ''}`}>
          <div className="verdict-hero">
            <span className="eyebrow">{live && !live.mine ? '08 · Example answer' : '08 · Your answer'}</span>
            <span className="proof mono" title="A quick test on a sample of practice messages. Only a sealed test, run once, is verified.">
              Exploratory · quick test on {count(v.n_cases)} messages{run.plan.scope?.one_route ? ' · your kinds scored as one route' : ''} · not a sealed test
            </span>
            <ScrollTiles className="verdict-word grad" text={word} />
            <ScrollWrite className="verdict-line" parts={reason} />
            <Reveal className="big-numbers" from={0.95} to={0.55}>
              {grabsMore ? (
                <div>
                  <b>
                    <Counter to={sA!.false_grab_rate * 100} run={inView.has('m4')} suffix="%" />
                  </b>
                  <small>
                    of other messages {ag} wrongly took ({grabbed(sA)} of {sA!.other_cases}) · {modelLabel(simpler)} {pct(sS!.false_grab_rate, 0)} ({grabbed(sS)})
                  </small>
                </div>
              ) : (
                <div>
                  <b>
                    <Counter to={(sS?.in_scope_accuracy ?? 0) * 100} run={inView.has('m4')} suffix="%" />
                  </b>
                  <small>
                    of your kinds {modelLabel(simpler)} got right ({inRight(sS)} of {sS?.in_scope_cases ?? 0}) · agent {pct(sA?.in_scope_accuracy ?? 0, 0)} ({inRight(sA)} of {sA?.in_scope_cases ?? 0})
                  </small>
                </div>
              )}
              <div>
                <b>
                  <Counter to={Math.abs(saving)} run={inView.has('m4')} prefix="$" />
                </b>
                <small>{saving >= 0 ? 'saved a month' : 'extra a month'}</small>
              </div>
              <div>
                <b>
                  <Counter to={v.n_cases} run={inView.has('m4')} />
                </b>
                <small>messages tested{v.n_cases < run.plan.population ? ` · stopped early (of ${count(run.plan.population)} planned)` : ''}</small>
              </div>
            </Reveal>
            <div className="chips">
              <button type="button" className="chip small" onClick={() => setDetails(true)}>
                See the details
              </button>
            </div>
          </div>
        </section>

        {/* 6 · Next steps */}
        <section id="m6" tabIndex={-1} className={`scene robot-${sideOf('m6')}${inView.has('m6') ? ' in' : ''}`}>
          <Decode className="eyebrow beat" text="09 · Next" run={inView.has('m6')} instantWhenStill />
          <ScrollRise className="mid" text="Take it with you." />
          <Reveal className="next-cards">
            <a className="glass next-card" href="?story">
              <b>See the full story</b>
              <small>Every number, every check, step by step</small>
            </a>
            <button type="button" className="glass next-card">
              <b>Download the audit file</b>
              <small>Re-run it and get the same answer</small>
            </button>
            <button type="button" className="glass next-card" onClick={() => document.getElementById('m1')?.scrollIntoView({ behavior: 'smooth' })}>
              <b>Test your own data</b>
              <small>Upload your messages</small>
            </button>
            <button type="button" className="glass next-card" onClick={() => setChatOpen(true)}>
              <b>Ask the guide</b>
              <small>About anything in your run</small>
            </button>
          </Reveal>
          <small className="mono muted">The audit file download isn’t wired up yet.</small>
        </section>
      </main>

      <aside className={`drawer glass${details ? ' open' : ''}`} aria-hidden={!details}>
        <div className="drawer-head">
          <b>The details</b>
          <button type="button" className="chip small" onClick={() => setDetails(false)}>
            Close
          </button>
        </div>
        <div className="plot-wrap">
          <svg className="plot" viewBox="0 0 400 190" role="img" aria-label="Accuracy against monthly cost per model">
            <text x="6" y="12" className="ax">more accurate ↑</text>
            <text x="394" y="186" className="ax" textAnchor="end">
              cost a month →
            </text>
            <line x1="0" y1="176" x2="400" y2="176" className="axl" />
            {shown.map((p) => {
              const x = 20 + ((p.cost - c0) / Math.max(1e-9, c1 - c0)) * 300;
              const y = 165 - ((p.acc - a0) / Math.max(1e-9, a1 - a0)) * 140;
              const best = p.name === simpler;
              return (
                <g key={p.name} className={`pt${best ? ' best' : ''}${p.name === AGENT ? ' agent' : ''}`} transform={`translate(${x.toFixed(1)} ${y.toFixed(1)})`}>
                  <circle r={best ? 8 : 6} />
                  <text x="12" y="4">
                    {modelLabel(p.name)}
                  </text>
                </g>
              );
            })}
          </svg>
          <button type="button" className="link" onClick={() => setShowAll(!showAll)}>
            {showAll ? 'Show top 3' : `Show all ${points.length}`}
          </button>
        </div>
        <dl>
          <dt>Checks</dt>
          <dd>
            {passed} of {v.checks.length} passed
            {failed.length ? ` · failed: ${failed.map((c) => checkName(c.name)).join('; ')}` : ''}
            {noted.length ? ` · noted (doesn’t change the answer): ${noted.map((c) => checkName(c.name)).join('; ')}` : ''}
          </dd>
          <dt>Stopped</dt>
          <dd>
            check-in {stopLook} of 5 · {count(v.n_cases)} of {count(run.plan.population)} messages
          </dd>
          <dt>Gain range</dt>
          <dd>
            {(v.accuracy_diff_ci[0] * 100).toFixed(1)} to {(v.accuracy_diff_ci[1] * 100).toFixed(1)} pts (needed +{Math.round(run.plan.gap * 100)})
          </dd>
          <dt>Compared with</dt>
          <dd>{modelLabel(simpler)}, the best free model in practice</dd>
          <dt>Model</dt>
          <dd className="mono">
            {extRun
              ? `your agent at ${run.plan.agent?.endpoint} · $${run.plan.agent?.usd_per_message}/message (your figure) · not pinned, a re-run may differ`
              : 'local Ollama agent: llama3.1:8b on this computer (used when you don’t bring yours)'}{' '}
            · {v.audit.git_commit.slice(0, 7)}
          </dd>
        </dl>
        <ul className="audit-checks">
          {v.checks.map((c) => (
            <li key={c.name}>
              <i className={c.passed ? 'ok' : 'no'} /> {checkName(c.name)}
              {!c.passed && (
                <small className={`mono check-fail${c.effect === 'REPORT' ? ' noted' : ''}`}>
                  {c.effect === 'REPORT' ? 'noted, doesn’t change the answer' : 'failed'} · {c.detail}
                </small>
              )}
            </li>
          ))}
        </ul>
      </aside>

      {hover && hover.kind === 'lib' && lib[hover.index] && (
        <div className="hovercard glass" style={{ left: Math.min(hover.x + 16, window.innerWidth - 300), top: hover.y + 16 }}>
          <p>“{lib[hover.index].text}”</p>
          <small className="mono muted">{hubName(lib[hover.index].label)}</small>
        </div>
      )}
      {hover && hover.kind === 'run' && run.messages[hover.index] && active !== 'm1' && active !== 'm3' && (
        <div className="hovercard glass" style={{ left: Math.min(hover.x + 16, window.innerWidth - 300), top: hover.y + 16 }}>
          <p>“{run.messages[hover.index].text}”</p>
          <small className="mono muted">{hubName(run.messages[hover.index].true)} · click to zoom in</small>
          <br />
          <small className="mono">
            <i className={run.messages[hover.index].agent_ok ? 'ok' : 'no'} /> agent · <i className={run.messages[hover.index].simpler_ok ? 'ok' : 'no'} /> {modelLabel(simpler).toLowerCase()}
          </small>
        </div>
      )}

      {active !== 'm1' && (
        <aside className={`chat glass${chatOpen ? ' open' : ' closed'}`} aria-label="Ask the guide">
          <button type="button" className="chat-head" onClick={() => setChatOpen(!chatOpen)} aria-expanded={chatOpen}>
            <span className="chat-dot" /> {chatOpen ? 'Ask about your result' : 'Ask'}
            {chatOpen && <span className="chat-caret">–</span>}
          </button>
          {chatOpen && (
            <>
              <div className="chat-log">
                {chat.length === 0 && (
                  <p className="chat-hint">
                    {live && !live.mine
                      ? 'You haven’t got a result yet, so I can only talk about the example run (a bank refund agent). Run your test and I’ll know every message in it.'
                      : `I know every message in your run${desc.trim() ? ` for “${desc.trim()}”` : ''}. Ask me anything about it, or pick one below.`}
                  </p>
                )}
                {chat.map((t, i) => (
                  <div key={i} className={`turn ${t.who}`}>
                    <p>{t.text}</p>
                    {t.list && (
                      <ul>
                        {t.list.map((l, k) => (
                          <li key={k}>{segs(l)}</li>
                        ))}
                      </ul>
                    )}
                    {t.who === 'guide' && (
                      <small className="mono" title="Every number in this reply was checked against your run by plain code.">
                        {t.by === 'model' ? '✓ numbers checked against your run' : '✓ worked out by plain code'}
                      </small>
                    )}
                  </div>
                ))}
                {thinking && (
                  <div className="turn guide typing">
                    <i />
                    <i />
                    <i />
                  </div>
                )}
                <div ref={chatEnd} />
              </div>
              <div className="chat-chips">
                {suggestions.map((c) => (
                  <button key={c} type="button" className="chip small" onClick={() => askGuide(c)}>
                    {c}
                  </button>
                ))}
              </div>
              <form
                className="chat-ask"
                onSubmit={(e) => {
                  e.preventDefault();
                  askGuide(ask);
                }}
              >
                <input value={ask} placeholder="Ask about your result…" aria-label="Ask the guide" onChange={(e) => setAsk(e.target.value)} />
                <button type="submit" className="chip small primary" disabled={thinking || !ask.trim()}>
                  Ask
                </button>
              </form>
            </>
          )}
        </aside>
      )}
      {(active === 'm2' || active === 'm3') && (
        <div className="legend glass mono">
          <span>
            <i className="dot both" /> both right
          </span>
          <span>
            <i className="dot agent" /> agent wrong
          </span>
          <span>
            <i className="dot simpler" /> {modelLabel(simpler).toLowerCase()} wrong
          </span>
        </div>
      )}

      <Robot controls={controls()} alive listening={typing} mood={mood} side={SCENES.find((s) => s.id === active)!.side} onFly={onFly} line={line()} />
    </div>
  );
}

/** The test's steps so far, under the robot's line (outside the announced part, so the ticking
 *  count and clock are not read out every second). */
function RunSteps({ progress, startedAt, name }: { progress?: RunProgress | null; startedAt?: number; name: (id: string) => string }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const secs = startedAt ? Math.max(0, Math.round((now - startedAt) / 1000)) : 0;
  const { steps } = runLog(progress, name);
  return (
    <div className="run-steps">
      <ol>
        {steps.map((st, i) => (
          <li key={i} className={st.now ? 'now' : 'done'}>
            <span aria-hidden="true">{st.now ? '' : '✓'}</span>
            {st.text}
          </li>
        ))}
      </ol>
      <small className="mono muted">
        {Math.floor(secs / 60)}:{String(secs % 60).padStart(2, '0')} so far · you can look around; the screens below show the example until yours is ready
      </small>
    </div>
  );
}
