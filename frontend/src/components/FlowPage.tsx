import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import type { LiveStatus } from '../data/types.ts';
import * as F from '../flow/lines.ts';
import {
  AGENT,
  EMPTY,
  FIXERS,
  MODELS,
  changesTheTest,
  datasetReady,
  maxMinutes,
  missing,
  mistakeCost,
  modelLabel,
  monthlyCost,
  parseAmount,
  plannedCases,
  racedArms,
  shownAccuracy,
  stoppedEarly,
  type Choices,
  type FlowData,
} from '../flow/model.ts';
import { useQuickRun } from '../flow/run.ts';
import { FLOW_BEATS, FLOW_IDS, FLOW_SCENES, flowScene } from '../flow/scenes.ts';
import { useActiveScene, useScenesInView } from '../hooks/useActiveScene.ts';
import { prefersReducedMotion, startSmoothWheel } from '../motion/flight.ts';
import type { Flyer, Lane } from '../motion/ribbon.ts';
import { count, pct, tableLabel, usd } from '../story/format.ts';
import { liveLine } from '../story/live.ts';
import { revealMood } from '../story/lines.ts';
import type { Mood } from '../story/scenes.ts';
import { Icon } from './Icons.tsx';
import { Background, CountUp, Decode, Words } from './Motion.tsx';
import { RaceTrack } from './RaceTrack.tsx';
import { RichText } from './Rich.tsx';
import { Robot } from './Robot.tsx';
import { Timeline } from './Timeline.tsx';

const GAPS = [0.02, 0.05, 0.1, 0.15, 0.2];
const DOTS = 240;
const RGB = { linear: [31, 110, 115], other: [110, 134, 150], agent: [37, 48, 49] } as const;
const PRE_LANES: Lane[] = [
  { name: 'Free models', rgb: [...RGB.linear], score: 60 },
  { name: 'Single prompt', rgb: [...RGB.other], score: 60 },
  { name: 'Your agent', rgb: [...RGB.agent], score: 60 },
];

/** The part of a menu title after its colon ("Banking77: bank support routing" -> "Bank support routing"). */
function shortTitle(title: string): [string, string] {
  const i = title.indexOf(':');
  if (i < 0) return [title, ''];
  const rest = title.slice(i + 1).trim();
  return [rest.charAt(0).toUpperCase() + rest.slice(1), title.slice(0, i)];
}

/** Is this scene the one across the middle of the window? */
function onScene(id: string): boolean {
  const r = document.getElementById(id)?.getBoundingClientRect();
  const mid = window.innerHeight / 2;
  return !!r && r.top <= mid && r.bottom >= mid;
}

/** Glide on to the next scene after a pause, but only if the reader is still on `from`. */
const go = (from: string, id: string, delay = 650) =>
  setTimeout(() => {
    if (onScene(from)) document.getElementById(id)?.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
  }, delay);

export function FlowPage({ data, api, live = null, storyHref }: { data: FlowData; api: string | null; live?: LiveStatus | null; storyHref: string }) {
  const active = useActiveScene(FLOW_IDS);
  const inView = useScenesInView(FLOW_IDS);
  const [c, setC] = useState<Choices>(EMPTY);
  const [text, setText] = useState({ volume: '', hourly: '', minutes: '' });
  const [find, setFind] = useState('');
  const [blocked, setBlocked] = useState<string | null>(null);
  const [night, setNight] = useState(false);
  const { state: rs, start, reset } = useQuickRun(api, data.saved_run);
  const set = (patch: Partial<Choices>) => {
    setC((p) => ({ ...p, ...patch }));
    if (changesTheTest(patch) && rs.phase !== 'running' && rs.phase !== 'starting') reset();
  };

  const dataset = data.datasets.find((d) => d.option === c.dataset) ?? null;
  const labels = (c.dataset && data.labels[c.dataset]) || [];
  const shown = labels.filter((l) => l.toLowerCase().includes(find.trim().toLowerCase()));
  const hasFree = c.arms.some((a) => !MODELS.find((m) => m.id === a)?.ai);
  const planned = c.gap === null ? null : plannedCases(data, c.gap);
  const minutes = planned === null ? null : maxMinutes(data, planned, c.arms);
  const mistake = c.hourly !== null && c.minutes !== null ? mistakeCost(c.hourly, c.minutes) : null;
  const todo = missing(c);
  const run = rs.result;
  const done = rs.phase === 'done' && run !== null;
  const busy = rs.phase === 'running' || rs.phase === 'starting';
  const v = run?.verdict ?? null;
  const raced = run ? racedArms(run, c.arms) : [];
  const notRaced = run ? c.arms.filter((a) => !raced.includes(a)) : [];
  const early = run ? stoppedEarly(run) : false;
  const gates = useMemo(() => {
    const total = run?.plan.planned_cases ?? rs.planned ?? planned ?? 0;
    return [1, 2, 3, 4, 5].map((k) => Math.round((total * k) / 5));
  }, [run, rs.planned, planned]);
  const lookIndex = Math.max(rs.looks.length, rs.replayed.length) - 1;

  // ribbon lanes: three generic lanes while choosing; the raced models at their scores once done
  const lanes = useMemo<Lane[]>(() => {
    if (!run) return PRE_LANES;
    const order = [...raced.filter((a) => a !== AGENT), ...(raced.includes(AGENT) ? [AGENT] : [])];
    return order.map((id) => ({
      name: modelLabel(id),
      rgb: [...(id === AGENT ? RGB.agent : id === 'linear' ? RGB.linear : RGB.other)] as [number, number, number],
      score: (shownAccuracy(run, id) ?? 0.6) * 100,
    }));
  }, [run]);

  const flyerRef = useRef<Flyer | null>(null);
  const onFly = useCallback((f: Flyer | null) => {
    flyerRef.current = f;
  }, []);
  const flyer = useCallback(() => flyerRef.current, []);
  useEffect(() => (prefersReducedMotion() ? undefined : startSmoothWheel()), []);
  // a finished race glides on to the scores
  useEffect(() => {
    if (rs.phase !== 'done') return;
    const t = go('f7', 'f8', 900);
    return () => clearTimeout(t);
  }, [rs.phase]);

  const scene = flowScene(active);
  const mood: Mood = active === 'f9' && v ? revealMood(v.accuracy_diff, v.verdict) : rs.phase === 'failed' && active === 'f7' ? 'sad' : scene.mood;
  const sceneCls = (id: string, extra: string) => `scene ${extra}${inView.has(id) ? ' in' : ''}`;
  const raceState = done ? 'done' : busy ? 'running' : todo ? 'blocked' : 'ready';
  const fill = (a: string) => (a === AGENT ? 'fill-agt' : a === 'linear' ? 'fill-lin' : 'fill-prm');

  const line = () => {
    switch (active) {
      case 'f2':
        return <RichText value={blocked ? F.notSplitLine(blocked) : F.useCaseLine(dataset ? shortTitle(dataset.title)[0] : null)} />;
      case 'f3':
        return <RichText value={F.scopeLine(c.labels.length, labels.length)} />;
      case 'f4':
        return <RichText value={F.contendersLine(c.arms, hasFree)} />;
      case 'f5':
        return <RichText value={F.gapLine(planned, minutes)} />;
      case 'f6':
        return <RichText value={F.costLine(mistake)} />;
      case 'f7':
        return <RichText value={rs.phase === 'failed' ? F.failedLine(rs.error ?? 'unknown error') : F.raceLine(raceState, todo?.[1] ?? null, lookIndex, early)} />;
      case 'f8':
        return <RichText value={done && run ? F.scoresLine(run, raced) : F.raceLine(raceState, todo?.[1] ?? null, lookIndex)} />;
      case 'f9':
        return <RichText value={done && v ? F.verdictLine(v) : F.raceLine(raceState, todo?.[1] ?? null, lookIndex)} />;
      default:
        return <RichText value={F.helloLine(c.description)} />;
    }
  };

  const next = (id: string, label = 'Next') => (
    <div className="chips">
      <a href={`#${id}`} className="chip primary">
        {label} ↓
      </a>
    </div>
  );
  const controls = () => {
    switch (active) {
      case 'f1':
        return c.description.trim() ? next('f2') : null;
      case 'f3':
        return c.labels.length ? next('f4') : null;
      case 'f4':
        return c.arms.includes(AGENT) && hasFree ? next('f5') : null;
      case 'f6':
        return todo === null || todo[0] === 'f7' ? next('f7', 'To the race') : null;
      case 'f7':
        if (busy) return null;
        return (
          <div className="chips">
            {todo ? (
              <a href={`#${todo[0]}`} className="chip">
                Go to {flowScene(todo[0]).name.toLowerCase()}
              </a>
            ) : (
              <button type="button" className="chip primary" onClick={() => start(c)}>
                {done || rs.phase === 'failed' ? 'Run again' : 'Run the test'}
              </button>
            )}
          </div>
        );
      case 'f8':
        return done ? next('f9', 'The verdict') : null;
      default:
        return null;
    }
  };
  const skipToRobot = (
    <button type="button" className="chip skip" onClick={() => document.querySelector<HTMLElement>('#robot-controls button, #robot-controls a')?.focus()}>
      Go to the robot’s buttons
    </button>
  );

  const toggle = (list: readonly string[], x: string) => (list.includes(x) ? list.filter((y) => y !== x) : [...list, x]);
  const dotsPer = planned === null ? 1 : Math.max(1, Math.ceil(planned / DOTS));
  const dotCount = planned === null ? 0 : Math.ceil(planned / dotsPer);
  const costs =
    done && v && c.table && c.volume !== null && mistake !== null
      ? raced
          .map((a) => ({ a, arm: v.arms.find((x) => x.name === a) }))
          .filter((x) => x.arm)
          .map((x) => ({ a: x.a, cost: monthlyCost(x.arm!, c.table as string, c.volume as number, mistake) }))
      : [];
  const maxCost = Math.max(1, ...costs.map((x) => x.cost));

  return (
    <div className={`page flow${night ? ' night' : ''}${active === 'f9' && done ? ' reveal' : ''}`}>
      <Background
        lanes={lanes}
        revealDirection={v ? (v.accuracy_diff < 0 ? 1 : v.accuracy_diff > 0 ? -1 : 0) : 0}
        nCases={v ? v.n_cases : (planned ?? 0)}
        sourceLabel={api !== null ? 'Quick test · DEV' : 'Quick test · preview'}
        sceneCount={FLOW_SCENES.length}
        night={night}
        flyer={flyer}
        beats={FLOW_BEATS}
      />

      <header className="top">
        <span className="brand glass">
          Agent Verifier<span className="t-teal">.</span>
        </span>
        {live && (
          <span className="live glass" role="status" title="Live status from the local server">
            <i className={live.ollama ? 'on' : 'off'} aria-hidden="true" />
            {liveLine(live)}
          </span>
        )}
        <div className="top-actions">
          <button type="button" className="pill glass" aria-pressed={night} onClick={() => setNight(!night)}>
            {night ? 'Day' : 'Night'}
          </button>
          <a href={storyHref} className="cta">
            Sealed story
          </a>
        </div>
      </header>

      <Timeline active={active} scenes={FLOW_SCENES} />

      <main>
        <section id="f1" tabIndex={-1} className={sceneCls('f1', 'robot-right')} aria-labelledby="f1-h">
          <Decode className="eyebrow beat" text="Agent Verifier" run={inView.has('f1')} />
          <h1 id="f1-h" className="big">
            <Words text="Do you really need an AI agent?" now />
          </h1>
          <label className="field hero-field beat-3">
            <span className="sr-only">What does your agent do?</span>
            <input
              type="text"
              value={c.description}
              placeholder="e.g. tells customers where their card is"
              onChange={(e) => set({ description: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && c.description.trim()) go('f1', 'f2', 0);
              }}
            />
          </label>
          {c.description.trim() && skipToRobot}
        </section>

        <section id="f2" tabIndex={-1} className={sceneCls('f2', 'robot-left')} aria-labelledby="f2-h">
          <Decode className="eyebrow beat" text="01 · Use case" run={inView.has('f2')} />
          <h2 id="f2-h" className="mid">
            <Words text="Closest to your job?" />
          </h2>
          <div className="pick-grid beat-3" role="group" aria-label="Use case">
            {data.datasets.map((d, i) => {
              const ready = datasetReady(d);
              const on = c.dataset === d.option;
              const [name, source] = shortTitle(d.title);
              return (
                <button
                  key={d.option}
                  type="button"
                  style={{ '--i': i } as CSSProperties}
                  className={`pick glass${on ? ' on' : ''}${ready ? '' : ' off'}`}
                  aria-pressed={on}
                  title={d.description}
                  onClick={() => {
                    if (!ready) return setBlocked(name);
                    setBlocked(null);
                    set({ dataset: d.option, labels: [] });
                    go('f2', 'f3');
                  }}
                >
                  <Icon name={d.option} />
                  <b>{name}</b>
                  <small className="mono">{ready ? `${source} · ${d.labels ?? '?'} labels` : `${source} · needs a split`}</small>
                </button>
              );
            })}
          </div>
        </section>

        <section id="f3" tabIndex={-1} className={sceneCls('f3', 'robot-right')} aria-labelledby="f3-h">
          <Decode className="eyebrow beat" text="02 · Scope" run={inView.has('f3')} />
          <h2 id="f3-h" className="mid">
            <Words text="Which messages are yours?" />
          </h2>
          <div className="card beat-3">
            {!c.dataset ? (
              <p className="note">Pick a use case first.</p>
            ) : (
              <>
                <div className="scope-head">
                  <div className="ring" style={{ '--p': labels.length ? c.labels.length / labels.length : 0 } as CSSProperties}>
                    <b>{c.labels.length}</b>
                    <small>/ {labels.length}</small>
                  </div>
                  <input
                    type="search"
                    className="field-input"
                    value={find}
                    placeholder="Search… try “card”"
                    aria-label="Search labels"
                    onChange={(e) => setFind(e.target.value)}
                  />
                </div>
                <div className="chips label-list scroll-y" role="group" aria-label="Labels your agent handles">
                  {shown.map((l) => (
                    <button
                      key={l}
                      type="button"
                      className={`chip small mono${c.labels.includes(l) ? ' on' : ''}`}
                      aria-pressed={c.labels.includes(l)}
                      onClick={() => set({ labels: toggle(c.labels, l) })}
                    >
                      {l.replace(/_/g, ' ')}
                    </button>
                  ))}
                </div>
                <p className="note">
                  <button type="button" className="link" onClick={() => set({ labels: [...labels] })}>
                    All of them
                  </button>{' '}
                  ·{' '}
                  <button type="button" className="link" onClick={() => set({ labels: [] })}>
                    Clear
                  </button>
                </p>
              </>
            )}
          </div>
          {c.labels.length > 0 && skipToRobot}
        </section>

        <section id="f4" tabIndex={-1} className={sceneCls('f4', 'robot-left')} aria-labelledby="f4-h">
          <Decode className="eyebrow beat" text="03 · Contenders" run={inView.has('f4')} />
          <h2 id="f4-h" className="mid">
            <Words text="Who races your agent?" />
          </h2>
          <div className="pick-grid small beat-3" role="group" aria-label="Models to race">
            {MODELS.map((m, i) => {
              const on = c.arms.includes(m.id);
              return (
                <button
                  key={m.id}
                  type="button"
                  style={{ '--i': i } as CSSProperties}
                  className={`pick glass${on ? ' on' : ''}${m.id === AGENT ? ' agent' : ''}`}
                  aria-pressed={on}
                  title={m.blurb}
                  onClick={() => set({ arms: toggle(c.arms, m.id) })}
                >
                  <Icon name={m.id} />
                  <b>{m.label}</b>
                  <small className="mono">{m.ai ? 'AI · paid' : 'free'}</small>
                </button>
              );
            })}
          </div>
          {c.arms.includes(AGENT) && hasFree && skipToRobot}
        </section>

        <section id="f5" tabIndex={-1} className={sceneCls('f5', 'robot-right')} aria-labelledby="f5-h">
          <Decode className="eyebrow beat" text="04 · Fair test" run={inView.has('f5')} />
          <h2 id="f5-h" className="mid">
            <Words text="How big a win is worth it?" />
          </h2>
          <div className="tables beat-3" role="group" aria-label="Smallest gain that matters">
            {GAPS.map((g) => (
              <button
                key={g}
                type="button"
                className={`chip${c.gap === g ? ' on' : ''}`}
                aria-pressed={c.gap === g}
                onClick={() => {
                  set({ gap: g });
                  go('f5', 'f6', 2800);
                }}
              >
                +{Math.round(g * 100)} pts
              </button>
            ))}
          </div>
          {planned !== null && minutes !== null && (
            <div className="sizer beat-3" key={planned}>
              <div className="dotgrid" aria-hidden="true">
                {Array.from({ length: dotCount }, (_, i) => (
                  <i key={i} style={{ '--i': i } as CSSProperties} />
                ))}
              </div>
              <div className="costs">
                <div>
                  <div className="cost-num">
                    <CountUp value={planned} decimals={0} run />
                  </div>
                  <div className="cost-name">cases at most</div>
                </div>
                <div>
                  <div className="cost-num">{F.minutesText(minutes).replace('about ', '')}</div>
                  <div className="cost-name">of AI time at most</div>
                </div>
              </div>
            </div>
          )}
          <details className="more beat-3">
            <summary>How it’s sized</summary>
            <p>
              Exact McNemar power: 80% chance to spot a real gain of this size, 5% false winners. It checks in 5 times and stops at
              the first clear answer. {dotsPer > 1 ? `One dot = ${dotsPer} cases. ` : ''}Minutes measured on this laptop, nothing cached.
            </p>
          </details>
        </section>

        <section id="f6" tabIndex={-1} className={sceneCls('f6', 'robot-left')} aria-labelledby="f6-h">
          <Decode className="eyebrow beat" text="05 · Your costs" run={inView.has('f6')} />
          <h2 id="f6-h" className="mid">
            <Words text="What does a mistake cost?" />
          </h2>
          <div className="card wide beat-3">
            <div className="field-grid">
              <label className="field">
                <span>Messages / month</span>
                <input
                  inputMode="numeric"
                  value={text.volume}
                  placeholder="10000"
                  onChange={(e) => {
                    setText({ ...text, volume: e.target.value });
                    set({ volume: parseAmount(e.target.value, 1) });
                  }}
                />
              </label>
              <label className="field">
                <span>Who fixes it?</span>
                <select value={c.fixer ?? ''} onChange={(e) => set({ fixer: e.target.value || null })}>
                  <option value="">Choose…</option>
                  {FIXERS.map((f) => (
                    <option key={f}>{f}</option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>$ / hour</span>
                <input
                  inputMode="decimal"
                  value={text.hourly}
                  placeholder="30"
                  onChange={(e) => {
                    setText({ ...text, hourly: e.target.value });
                    set({ hourly: parseAmount(e.target.value) });
                  }}
                />
              </label>
              <label className="field">
                <span>Minutes / fix</span>
                <input
                  inputMode="decimal"
                  value={text.minutes}
                  placeholder="5"
                  onChange={(e) => {
                    setText({ ...text, minutes: e.target.value });
                    set({ minutes: parseAmount(e.target.value) });
                  }}
                />
              </label>
            </div>
            <div className="tables" role="group" aria-label="Where the AI runs">
              {data.price_tables.map((t) => (
                <button
                  key={t.name}
                  type="button"
                  className={`chip small${c.table === t.name ? ' on' : ''}`}
                  aria-pressed={c.table === t.name}
                  onClick={() => set({ table: t.name })}
                >
                  {tableLabel(t.name)}
                </button>
              ))}
            </div>
            <div className="mistake">
              <span className="cost-name">One mistake</span>
              <b className="cost-num" key={mistake ?? 'none'}>
                {mistake === null ? '—' : usd(mistake)}
              </b>
            </div>
            <details className="more">
              <summary>How it’s worked out</summary>
              <p>$/hour × minutes per fix ÷ 60. AI prices per message come from the assumed price tables in prereg.yaml.</p>
            </details>
          </div>
          {(todo === null || todo[0] === 'f7') && skipToRobot}
        </section>

        <section id="f7" tabIndex={-1} className={sceneCls('f7', 'robot-right')} aria-labelledby="f7-h">
          <Decode className="eyebrow beat" text="06 · The race" run={inView.has('f7')} />
          <h2 id="f7-h" className="mid">
            <Words text={busy ? 'Racing…' : done ? 'Race over.' : rs.phase === 'failed' ? 'The race stopped.' : 'Ready?'} />
          </h2>
          <div className="beat-3 race-wrap">
            <RaceTrack
              looks={rs.looks}
              reached={rs.replayed}
              gates={gates}
              gap={c.gap ?? 0.1}
              finalGain={done && v ? v.accuracy_diff : null}
              simpler={modelLabel(rs.simpler ?? run?.plan.comparator ?? 'simpler')}
              agent="Your agent"
            />
            <p className="note" aria-live="polite">
              {rs.phase === 'starting'
                ? 'Starting…'
                : busy && lookIndex < 0
                  ? 'Training the free models and drawing the cases…'
                  : busy
                    ? `Check-in ${lookIndex + 1} of 5`
                    : done && run
                      ? `${early ? 'Stopped early' : 'Finished'} at ${count(run.verdict.n_cases)} of ${count(run.plan.planned_cases)} cases.`
                      : rs.phase === 'failed'
                        ? rs.error
                        : todo
                          ? `Still needed: ${todo[1]}.`
                          : api !== null
                            ? 'Runs on your machine with Ollama.'
                            : 'Preview: replays the saved Banking77 card-delivery test.'}
            </p>
          </div>
          {!busy && skipToRobot}
        </section>

        <section id="f8" tabIndex={-1} className={sceneCls('f8', 'robot-left')} aria-labelledby="f8-h">
          <Decode className="eyebrow beat" text="07 · The scores" run={inView.has('f8')} />
          <h2 id="f8-h" className={`mid${done ? '' : ' muted'}`}>
            <Words text={done ? (run?.scope ? 'Right on your messages.' : 'Right answers.') : 'Scores land here.'} />
          </h2>
          {done && run && (
            <div className="beat-3 stats">
              {raced
                .filter((a) => shownAccuracy(run, a) !== null)
                .sort((a, b) => (shownAccuracy(run, b) as number) - (shownAccuracy(run, a) as number))
                .map((a) => {
                  const acc = shownAccuracy(run, a) as number;
                  const grab = run.scope?.[a]?.false_grab_rate;
                  return (
                    <div className="stat" key={a}>
                      <span>{modelLabel(a)}</span>
                      <div className="bar" role="meter" aria-label={`${modelLabel(a)} accuracy`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Number((acc * 100).toFixed(1))}>
                        <div className={`fill ${fill(a)}`} style={{ width: `${acc * 100}%` }} />
                        {grab !== undefined && grab > 0 && <div className="grab" style={{ width: `${grab * 100}%` }} title={`Grabs ${pct(grab, 0)} of other messages`} />}
                      </div>
                      <b>
                        <CountUp value={acc * 100} suffix="%" run={inView.has('f8')} />
                      </b>
                    </div>
                  );
                })}
              <details className="more">
                <summary>What the bars show</summary>
                <p>
                  {run.scope
                    ? `Share right on ${count(run.scope[AGENT]?.in_scope_cases ?? 0)} of your messages. The red tip is how often a model wrongly grabbed someone else’s message: ${raced
                        .filter((a) => run.scope?.[a])
                        .map((a) => `${modelLabel(a)} ${pct(run.scope?.[a]?.false_grab_rate ?? 0, 0)}`)
                        .join(', ')}.`
                    : `Share right on ${count(run.verdict.n_cases)} test cases.`}
                  {notRaced.length > 0 && ` Not in this run: ${notRaced.map(modelLabel).join(', ')}.`}
                </p>
              </details>
            </div>
          )}
        </section>

        <section id="f9" tabIndex={-1} className={sceneCls('f9', 'robot-right')} aria-labelledby="f9-h">
          <div className="scan" aria-hidden="true" />
          <Decode className="eyebrow beat" text="08 · The verdict" run={inView.has('f9')} />
          <h2 id="f9-h" className={done ? 'big land' : 'mid muted'}>
            {done && v ? F.verdictWord(v) : <Words text="The verdict lands here." />}
          </h2>
          {done && v && run && (
            <>
              {costs.length > 0 && (
                <div className="cost-bars beat-3" aria-label="Estimated cost per month">
                  {[...costs]
                    .sort((a, b) => a.cost - b.cost)
                    .map((x) => (
                      <div key={x.a} className={`cb${x.a === AGENT ? ' agent' : ''}`}>
                        <span>{modelLabel(x.a)}</span>
                        <div className="cb-track">
                          <i style={{ width: `${(x.cost / maxCost) * 100}%` }} />
                        </div>
                        <b>${count(x.cost)}</b>
                      </div>
                    ))}
                  <small className="mono">per month · estimate · {tableLabel(c.table as string)}</small>
                </div>
              )}
              <details className="more beat-3">
                <summary>Why</summary>
                <p>{F.verdictLead(v, c.description, run.plan.planned_cases)}</p>
                <p>
                  Monthly cost = messages × (AI price per message + share wrong × {mistake === null ? '—' : usd(mistake)}). The share
                  wrong is on the whole test mix the engine judged. Change your costs above; this updates without re-running.
                </p>
                <ul className="reasons">
                  {v.reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </details>
              <p className="lead beat-3 mono small-mono">EXPLORATORY · quick test on DEV data · not the sealed test</p>
            </>
          )}
        </section>
      </main>

      <Robot mood={mood} side={scene.side} onFly={onFly} line={line()} controls={controls()} />
    </div>
  );
}
