import { useMemo, useState } from 'react';
import type { Arm, Story } from '../data/types.ts';
import { costGap, defaultErrorCost, expectedCost, sliderStep } from '../story/cost.ts';
import { num, pct, tableLabel, usd } from '../story/format.ts';
import * as L from '../story/lines.ts';
import { SCENE_IDS, sceneById, type Mood } from '../story/scenes.ts';
import { useActiveScene } from '../hooks/useActiveScene.ts';
import { RichText } from './Rich.tsx';
import { Robot } from './Robot.tsx';
import { Timeline } from './Timeline.tsx';

const ARM_COLOURS: Record<string, string> = { linear: '#2242E8', prompt: '#8A8FA0' };
const AGENT_COLOUR = '#D6620A';

export function StoryPage({ story }: { story: Story }) {
  const v = story.verdict;
  const active = useActiveScene(SCENE_IDS);
  const [guess, setGuess] = useState<string | null>(null);
  const [errorCost, setErrorCost] = useState(() => defaultErrorCost(v.error_cost_range));
  const [table, setTable] = useState(story.default_price_table);
  const [checksOpen, setChecksOpen] = useState(false);
  const [replayOpen, setReplayOpen] = useState(false);
  const [downloaded, setDownloaded] = useState<string | null>(null);

  const agent = v.arms.find((a) => a.name === v.agent_arm) as Arm;
  const simpler = v.arms.find((a) => a.name === v.simpler_arm) as Arm;
  const ordered = useMemo(() => L.armOrder(v.arms, v.agent_arm), [v]);
  const ranked = useMemo(() => L.ranking(v.arms), [v]);
  const checks = useMemo(() => L.summarizeChecks(v.checks), [v]);
  const label = (name: string, start = false) => L.armLabel(name, v.agent_arm, start);
  const colour = (name: string) => (name === v.agent_arm ? AGENT_COLOUR : (ARM_COLOURS[name] ?? '#4A4E5C'));

  const scene = sceneById(active);
  const mood: Mood = active === 's6' ? L.revealMood(v.accuracy_diff) : scene.mood;
  const range = v.error_cost_range;
  const step = sliderStep(range);
  const gap = costGap(agent, simpler, table, errorCost);
  const priceTable = story.price_tables.find((t) => t.name === table);
  const rehearsal = story.source === 'rehearsal';

  const download = () => {
    const filename = `audit-${story.source}-${v.audit.git_commit.slice(0, 7)}.json`;
    const bundle = {
      source: story.source,
      verdict: v,
      price_tables: story.price_tables,
      rules: story.rules,
    };
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
              <button type="button" className="chip" aria-expanded={checksOpen} aria-controls="checks-list" onClick={() => setChecksOpen(!checksOpen)}>
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
    <div className="page">
      <header className="top">
        <span className="brand">
          Agent Verifier<span className="dot">.</span>
        </span>
        <span className={`source-badge ${rehearsal ? 'rehearsal' : 'sealed'}`}>{L.sourceBanner(story.source)}</span>
        <a href="#s7" className="cta">
          Verify an agent
        </a>
      </header>

      <Robot mood={mood} side={scene.side}>
        {bubble()}
      </Robot>

      <Timeline active={active} />

      <main>
        <section id="s1" className="scene robot-right" aria-labelledby="s1-h">
          <span className="eyebrow beat">Agent Verifier</span>
          <h1 id="s1-h" className="big beat-2">Is your agent worth deploying?</h1>
          <p className="lead beat-3">{L.introLead(story.source)}</p>
        </section>

        <section id="s2" className="scene robot-left tint" aria-labelledby="s2-h">
          <span className="eyebrow beat">01 · The contenders</span>
          <h2 id="s2-h" className="mid beat-2">{L.contendersHeading(v.arms.length, v.n_cases)}</h2>
          <p className="lead beat-3">
            {ordered.map((a, i) => {
              const blurb = L.armBlurb(a.name, v.agent_arm);
              return (
                <span key={a.name}>
                  {i > 0 && ' '}
                  <b className={a.name === v.agent_arm ? 'tone-agent' : a.name === 'linear' ? 'tone-linear' : undefined}>
                    {label(a.name, true)}
                  </b>
                  {blurb ? `, ${blurb}.` : '.'}
                </span>
              );
            })}
          </p>
        </section>

        <section id="s3" className="scene robot-right" aria-labelledby="s3-h">
          <span className="eyebrow beat">02 · Fair rules</span>
          <h2 id="s3-h" className="mid beat-2">The rules are locked before the test.</h2>
          <p className="lead beat-3">
            Metric, thresholds, checks and prices are written down first, so nobody can move the goalposts.
          </p>
        </section>

        <section id="s4" className="scene robot-left tint" aria-labelledby="s4-h">
          <span className="eyebrow beat">03 · The scores</span>
          <h2 id="s4-h" className="mid beat-2">How each one did.</h2>
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
                  <div className="fill" style={{ width: `${a.accuracy * 100}%`, background: colour(a.name) }} />
                </div>
                <b className="stat-num">{pct(a.accuracy)}</b>
              </div>
            ))}
            <p className="note">
              Accuracy on {v.n_cases.toLocaleString('en-US')} cases{rehearsal ? ' (DEV rehearsal)' : ' (sealed test)'}.
            </p>
          </div>
        </section>

        <section id="s5" className="scene robot-right" aria-labelledby="s5-h">
          <span className="eyebrow beat">04 · Integrity</span>
          <h2 id="s5-h" className="mid beat-2">{L.checksHeading(checks)}</h2>
          <p className="lead beat-3">{L.checksLead(checks)}</p>
          {checksOpen && (
            <ul id="checks-list" className="checks" tabIndex={0} aria-label="All checks">
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

        <section id="s6" className="scene robot-left dark center" aria-labelledby="s6-h">
          <span className="eyebrow beat on-dark">05 · The reveal</span>
          <div id="s6-h" className="land reveal-num" aria-label={`${L.revealNumber(v.accuracy_diff)} points`}>
            {L.revealNumber(v.accuracy_diff)}
            <span className="pts"> pts</span>
          </div>
          <p className="lead beat-3 on-dark-lead">{L.revealLead(story)}</p>
        </section>

        <section id="s7" className="scene robot-right center" aria-labelledby="s7-h">
          <span className="eyebrow beat">06 · Your call</span>
          <h2 id="s7-h" className="mid beat-2">What does one wrong answer cost you?</h2>
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
                <div className="cost-num tone-agent">{usd(expectedCost(agent, table, errorCost))}</div>
              </div>
            </div>
            {priceTable && (
              <p className="note">
                {priceTable.assumed ? 'Assumed prices' : 'Prices'}: ${num(priceTable.input_usd_per_mtok)} in / $
                {num(priceTable.output_usd_per_mtok)} out per million tokens. Expected cost = inference + error rate ×
                error cost.
              </p>
            )}
          </div>
        </section>

        <section id="s8" className="scene robot-left tint" aria-labelledby="s8-h">
          <span className="eyebrow beat">07 · One case</span>
          <h2 id="s8-h" className={`mid beat-2${story.case ? '' : ' muted'}`}>{L.caseHeading(story.case)}</h2>
          <p className="lead beat-3">{L.caseLead(story.case)}</p>
        </section>

        <section id="s9" className="scene robot-right" aria-labelledby="s9-h">
          <span className="eyebrow beat">08 · The recap</span>
          <h2 id="s9-h" className="mid beat-2">{L.recapHeading(story)}</h2>
          <p className="lead beat-3">{L.recapLead(story, table)}</p>
          {v.reasons.length > 0 && (
            <ul className="reasons beat-3">
              {v.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          )}
          <p className="lead beat-3 mono small-mono">{L.verdictTag(story)}</p>
          <p className="lead beat-3 mono small-mono">{L.auditLine(story)}</p>
        </section>
      </main>
    </div>
  );
}
