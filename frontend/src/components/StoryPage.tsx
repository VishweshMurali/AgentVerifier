import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Arm, LiveStatus, Story } from '../data/types.ts';
import { liveLine } from '../story/live.ts';
import { costGap, defaultErrorCost, expectedCost, sliderStep } from '../story/cost.ts';
import { num, pct, tableLabel, usd } from '../story/format.ts';
import * as L from '../story/lines.ts';
import { SCENE_IDS, SCENES, sceneById, type Mood } from '../story/scenes.ts';
import { useActiveScene, useScenesInView } from '../hooks/useActiveScene.ts';
import { prefersReducedMotion, startSmoothWheel } from '../motion/flight.ts';
import type { Flyer, Lane } from '../motion/ribbon.ts';
import { Background, CountUp, Decode, Words } from './Motion.tsx';
import { RichText } from './Rich.tsx';
import { Robot } from './Robot.tsx';
import { Timeline } from './Timeline.tsx';

// Colours of the design (design/frontend/SPEC.md): Linear teal, Prompt grey, the agent in ink.
const LANE_RGB: Record<string, [number, number, number]> = { linear: [31, 110, 115], prompt: [110, 134, 150] };
const AGENT_RGB: [number, number, number] = [37, 48, 49];
const FILL_CLASS: Record<string, string> = { linear: 'fill-lin', prompt: 'fill-prm' };

export function StoryPage({ story, live = null }: { story: Story; live?: LiveStatus | null }) {
  const v = story.verdict;
  const active = useActiveScene(SCENE_IDS);
  const inView = useScenesInView(SCENE_IDS);
  const [guess, setGuess] = useState<string | null>(null);
  const [errorCost, setErrorCost] = useState(() => defaultErrorCost(v.error_cost_range));
  const [table, setTable] = useState(story.default_price_table);
  const [checksOpen, setChecksOpen] = useState(false);
  const [replayOpen, setReplayOpen] = useState(false);
  const [downloaded, setDownloaded] = useState<string | null>(null);
  const [night, setNight] = useState(false);

  const agent = v.arms.find((a) => a.name === v.agent_arm) as Arm;
  const simpler = v.arms.find((a) => a.name === v.simpler_arm) as Arm;
  const ordered = useMemo(() => L.armOrder(v.arms, v.agent_arm), [v]);
  const ranked = useMemo(() => L.ranking(v.arms), [v]);
  const checks = useMemo(() => L.summarizeChecks(v.checks), [v]);
  const label = (name: string, start = false) => L.armLabel(name, v.agent_arm, start);
  const fillClass = (name: string) => (name === v.agent_arm ? 'fill-agt' : (FILL_CLASS[name] ?? 'fill-prm'));

  // the background's lanes: simpler arms first, the agent last, each at its real accuracy
  const lanes = useMemo<Lane[]>(
    () =>
      ordered.map((a) => ({
        name: label(a.name, true),
        rgb: a.name === v.agent_arm ? AGENT_RGB : (LANE_RGB[a.name] ?? [110, 134, 150]),
        score: a.accuracy * 100,
      })),
    [ordered, v.agent_arm],
  );
  const reveal = L.revealState(v.accuracy_diff);
  const revealDirection = reveal === 'behind' ? 1 : reveal === 'ahead' ? -1 : 0;

  // the robot's position, shared with the background without re-rendering
  const flyerRef = useRef<Flyer | null>(null);
  const onFly = useCallback((f: Flyer | null) => {
    flyerRef.current = f;
  }, []);
  const flyer = useCallback(() => flyerRef.current, []);

  useEffect(() => (prefersReducedMotion() ? undefined : startSmoothWheel()), []);

  const scene = sceneById(active);
  const mood: Mood = active === 's6' ? L.revealMood(v.accuracy_diff) : scene.mood;
  const range = v.error_cost_range;
  const step = sliderStep(range);
  const gap = costGap(agent, simpler, table, errorCost);
  const priceTable = story.price_tables.find((t) => t.name === table);
  const rehearsal = story.source === 'rehearsal';
  const sceneCls = (id: string, extra: string) => `scene ${extra}${inView.has(id) ? ' in' : ''}`;
  const gapPts = Math.abs(Number((v.accuracy_diff * 100).toFixed(1)));
  const gapSign = reveal === 'behind' ? '−' : reveal === 'ahead' ? '+' : '';

  const download = () => {
    const filename = `audit-${story.source}-${v.audit.git_commit.slice(0, 7)}.json`;
    const bundle = { source: story.source, verdict: v, price_tables: story.price_tables, rules: story.rules };
    const url = URL.createObjectURL(new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setDownloaded(filename);
  };

  const bubble = () => {
    switch (active) {
      case 's2':
        return (
          <>
            <RichText value={L.contendersLine()} />
            <div className="chips" role="group" aria-label="Your guess">
              {ordered.map((a) => (
                <button
                  key={a.name}
                  type="button"
                  className={`chip${guess === a.name ? ' on' : ''}`}
                  aria-pressed={guess === a.name}
                  onClick={() => setGuess(a.name)}
                >
                  {L.guessButtonLabel(a.name, v.agent_arm)}
                </button>
              ))}
            </div>
            {guess && (
              <div className="reply">
                <RichText value={L.guessReply(guess, v.agent_arm)} />
              </div>
            )}
          </>
        );
      case 's3':
        return <RichText value={L.rulesLine(story.rules.checks, story.rules.price_tables)} />;
      case 's4':
        return <RichText value={L.scoresPayoff(guess, v.arms, v.agent_arm)} />;
      case 's5':
        return (
          <>
            <RichText value={L.checksLine(checks)} />
            <div className="chips">
              <button
                type="button"
                className="chip"
                aria-expanded={checksOpen}
                aria-controls="checks-list"
                onClick={() => setChecksOpen(!checksOpen)}
              >
                {L.checksToggleLabel(checksOpen)}
              </button>
            </div>
          </>
        );
      case 's6':
        return (
          <>
            <RichText value={L.revealLine(v.accuracy_diff)} />
            <div className="chips">
              <a href="#s7" className="chip primary">
                Yes, show me
              </a>
            </div>
          </>
        );
      case 's7':
        return <RichText value={L.costLine(errorCost, gap)} />;
      case 's8':
        return (
          <>
            <RichText value={L.caseLine(story.case)} />
            {story.case && (
              <>
                <div className="chips">
                  <button type="button" className="chip" aria-expanded={replayOpen} onClick={() => setReplayOpen(!replayOpen)}>
                    {L.replayToggleLabel(replayOpen)}
                  </button>
                </div>
                {replayOpen && (
                  <ol className="steps">
                    {L.replaySteps(story.case).map((s, i) => (
                      <li key={i}>
                        <RichText value={s} />
                      </li>
                    ))}
                  </ol>
                )}
              </>
            )}
          </>
        );
      case 's9':
        return (
          <>
            <RichText value={L.recapLine(guess, v.arms, v.agent_arm)} />
            <div className="chips">
              <button type="button" className="chip primary" onClick={download}>
                Download audit bundle
              </button>
            </div>
            {downloaded && (
              <div className="reply">
                <RichText value={L.downloadReply(downloaded)} />
              </div>
            )}
          </>
        );
      default:
        return <RichText value={L.helloLine()} />;
    }
  };

  return (
    <div className={`page${night ? ' night' : ''}${active === 's6' ? ' reveal' : ''}`}>
      <Background
        lanes={lanes}
        revealDirection={revealDirection}
        nCases={v.n_cases}
        sourceLabel={rehearsal ? 'Dev rehearsal' : 'Sealed test'}
        sceneCount={SCENES.length}
        night={night}
        flyer={flyer}
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
          <a href="#s7" className="cta">
            Verify an agent
          </a>
        </div>
      </header>

      <Robot mood={mood} side={scene.side} onFly={onFly}>
        {bubble()}
      </Robot>

      <Timeline active={active} />

      <main>
        <section id="s1" className={sceneCls('s1', 'robot-right')} aria-labelledby="s1-h">
          <Decode className="eyebrow beat" text="Agent Verifier" run={inView.has('s1')} />
          <h1 id="s1-h" className="big">
            <Words text="Is your agent worth deploying?" now />
          </h1>
          <p className="lead beat-3">{L.introLead(story.source)}</p>
        </section>

        <section id="s2" className={sceneCls('s2', 'robot-left')} aria-labelledby="s2-h">
          <Decode className="eyebrow beat" text="01 · The contenders" run={inView.has('s2')} />
          <h2 id="s2-h" className="mid">
            <Words text={L.contendersHeading(v.arms.length, v.n_cases)} />
          </h2>
          <p className="lead beat-3">
            {ordered.map((a, i) => {
              const blurb = L.armBlurb(a.name, v.agent_arm);
              return (
                <span key={a.name}>
                  {i > 0 && ' '}
                  <b className={a.name === v.agent_arm ? 'ink' : a.name === 'linear' ? 't-teal' : undefined}>
                    {label(a.name, true)}
                  </b>
                  {blurb ? `, ${blurb}.` : '.'}
                </span>
              );
            })}
          </p>
        </section>

        <section id="s3" className={sceneCls('s3', 'robot-right')} aria-labelledby="s3-h">
          <Decode className="eyebrow beat" text="02 · Fair rules" run={inView.has('s3')} />
          <h2 id="s3-h" className="mid">
            <Words text="The rules are locked before the test." />
          </h2>
          <p className="lead beat-3">Metric, thresholds, checks and prices are written down first.</p>
        </section>

        <section id="s4" className={sceneCls('s4', 'robot-left')} aria-labelledby="s4-h">
          <Decode className="eyebrow beat" text="03 · The scores" run={inView.has('s4')} />
          <h2 id="s4-h" className="mid">
            <Words text="How each one did." />
          </h2>
          <div className="beat-3 stats">
            {ranked.map((a) => (
              <div className="stat" key={a.name}>
                <span>{label(a.name, true)}</span>
                <div
                  className="bar"
                  role="meter"
                  aria-label={`${label(a.name, true)} accuracy`}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Number((a.accuracy * 100).toFixed(1))}
                >
                  <div className={`fill ${fillClass(a.name)}`} style={{ width: `${a.accuracy * 100}%` }} />
                </div>
                <b aria-label={pct(a.accuracy)}>
                  <CountUp value={a.accuracy * 100} suffix="%" run={inView.has('s4')} />
                </b>
              </div>
            ))}
            <p className="note">
              Accuracy on {v.n_cases.toLocaleString('en-US')} cases{rehearsal ? ' (DEV rehearsal)' : ' (sealed test)'}.
            </p>
          </div>
        </section>

        <section id="s5" className={sceneCls('s5', 'robot-right')} aria-labelledby="s5-h">
          <Decode className="eyebrow beat" text="04 · Integrity" run={inView.has('s5')} />
          <h2 id="s5-h" className="mid">
            <Words text={L.checksHeading(checks)} />
          </h2>
          <p className="lead beat-3">{L.checksLead(checks)}</p>
          {checksOpen && (
            <ul id="checks-list" className="checks glass" tabIndex={0} aria-label="All checks">
              {v.checks.map((c) => (
                <li key={c.name}>
                  <span className={`pass ${c.passed ? 'yes' : 'no'}`}>{c.passed ? 'Passed' : 'Failed'}</span>
                  <span className="mono">{c.name}</span>
                  <span className="check-val">{num(c.value)}</span>
                  <span className="check-detail">
                    {c.detail}
                    {c.passed ? '' : ` (${c.effect === 'REPORT' ? 'reported only' : `sets ${c.effect}`})`}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section id="s6" className={sceneCls('s6', 'robot-left center')} aria-labelledby="s6-h">
          <div className="scan" aria-hidden="true" />
          <Decode className="eyebrow beat" text="05 · The reveal" run={inView.has('s6')} />
          <div id="s6-h" className="land gap-num" role="text" aria-label={`${L.revealNumber(v.accuracy_diff)} points`}>
            <span aria-hidden="true">
              <CountUp value={gapPts} prefix={gapSign} run={inView.has('s6')} />
            </span>
            <span className="gap-pts" aria-hidden="true">
              {' '}
              pts
            </span>
          </div>
          <p className="lead beat-3">{L.revealLead(story)}</p>
        </section>

        <section id="s7" className={sceneCls('s7', 'robot-right center')} aria-labelledby="s7-h">
          <Decode className="eyebrow beat" text="06 · Your call" run={inView.has('s7')} />
          <h2 id="s7-h" className="mid">
            <Words text="What does one wrong answer cost you?" />
          </h2>
          <div className="card beat-3">
            <div className="tables" role="group" aria-label="Price table">
              {story.price_tables.map((t) => (
                <button
                  key={t.name}
                  type="button"
                  className={`chip small${t.name === table ? ' on' : ''}`}
                  aria-pressed={t.name === table}
                  onClick={() => setTable(t.name)}
                >
                  {tableLabel(t.name)}
                </button>
              ))}
            </div>
            <label htmlFor="cost" className="cost-label">
              <span>Cost per wrong answer</span>
              <b>{usd(errorCost)}</b>
            </label>
            <input
              id="cost"
              type="range"
              min={range[0]}
              max={range[1]}
              step={step}
              value={errorCost}
              onChange={(e) => setErrorCost(Number(e.target.value))}
            />
            <div className="costs" aria-live="polite">
              <div>
                <div className="cost-name">{label(simpler.name, true)}, per case</div>
                <div className="cost-num">{usd(expectedCost(simpler, table, errorCost))}</div>
              </div>
              <div>
                <div className="cost-name">Your agent, per case</div>
                <div className="cost-num ink">{usd(expectedCost(agent, table, errorCost))}</div>
              </div>
            </div>
            {priceTable && (
              <p className="note">
                {priceTable.assumed ? 'Assumed prices' : 'Prices'}: ${num(priceTable.input_usd_per_mtok)} in / $
                {num(priceTable.output_usd_per_mtok)} out per million tokens. Expected cost = inference + error rate × error
                cost.
              </p>
            )}
          </div>
        </section>

        <section id="s8" className={sceneCls('s8', 'robot-left')} aria-labelledby="s8-h">
          <Decode className="eyebrow beat" text="07 · One case" run={inView.has('s8')} />
          <h2 id="s8-h" className={`mid${story.case ? '' : ' muted'}`}>
            <Words text={L.caseHeading(story.case)} />
          </h2>
          <p className="lead beat-3">{L.caseLead(story.case)}</p>
        </section>

        <section id="s9" className={sceneCls('s9', 'robot-right')} aria-labelledby="s9-h">
          <Decode className="eyebrow beat" text="08 · The recap" run={inView.has('s9')} />
          <h2 id="s9-h" className="mid">
            <Words text={L.recapHeading(story)} />
          </h2>
          <p className="lead beat-3">{L.recapLead(story, table)}</p>
          <p className="lead beat-3 mono small-mono">{L.verdictTag(story)}</p>
          <p className="lead beat-3 mono small-mono">{L.auditLine(story)}</p>
        </section>
      </main>
    </div>
  );
}
