import { useState } from 'react';
import { TUNE, TUNE_DEFAULTS, type Tune } from '../motion/birth.ts';

// A live tuning panel for the sphere opening (?sphere&tune). Every slider moves one number the
// animation reads on the next frame; "Copy my settings" puts them on the clipboard so they can be
// sent back as exact values. Scene positions: 0 welcome, 1 the agent screen, 2 the kinds screen.
interface Control {
  key: keyof Tune;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
}
const GROUPS: { title: string; items: Control[] }[] = [
  {
    title: 'The sphere',
    items: [
      { key: 'sphereSize', label: 'Size', hint: 'radius of the sphere', min: 0.25, max: 0.9, step: 0.01 },
      { key: 'particles', label: 'Particles', hint: 'fine particles that make the ball', min: 0, max: 5000, step: 100 },
      { key: 'density', label: 'Core pull', hint: 'higher = dots packed closer to the core', min: 1, max: 4, step: 0.05 },
      { key: 'dotSize', label: 'Dot size', hint: 'graph dots inside the sphere', min: 0.15, max: 1.2, step: 0.01 },
      { key: 'coreSize', label: 'Core size', hint: 'the bright blue core', min: 0.1, max: 0.7, step: 0.01 },
      { key: 'coreGlow', label: 'Core colour', hint: 'how saturated the core is', min: 0.2, max: 1, step: 0.01 },
    ],
  },
  {
    title: 'Charging and firing',
    items: [
      { key: 'chargeEnd', label: 'Fully charged at', hint: 'scene position', min: 0.4, max: 2.2, step: 0.05 },
      { key: 'chargeGrow', label: 'Grows by', hint: 'extra size when charged', min: 0, max: 0.9, step: 0.01 },
      { key: 'neurons', label: 'Spark jets', hint: 'most firing at once', min: 0, max: 44, step: 1 },
      { key: 'neuronSpeed', label: 'Firing speed', hint: '', min: 0.3, max: 3, step: 0.05 },
    ],
  },
  {
    title: 'Overcharge',
    items: [
      { key: 'overcharge', label: 'Build-up', hint: 'shake, beat and white-hot core before the blast', min: 0, max: 1.5, step: 0.05 },
      { key: 'squeeze', label: 'Squeeze', hint: 'how far the ball tightens before it blasts', min: 0, max: 0.6, step: 0.01 },
    ],
  },
  {
    title: 'The blast',
    items: [
      { key: 'burstAt', label: 'Starts at', hint: 'scene position of the flash', min: 0.6, max: 2.6, step: 0.02 },
      { key: 'burstSpan', label: 'Lasts', hint: 'scroll until every dot has landed', min: 0.3, max: 1.6, step: 0.02 },
      { key: 'burstReach', label: 'Fly out', hint: 'how far dots and sparks go', min: 0.2, max: 3.5, step: 0.05 },
      { key: 'flash', label: 'Flash', hint: 'brightness', min: 0, max: 1.6, step: 0.05 },
      { key: 'rays', label: 'Rays', hint: 'streaking rays', min: 0, max: 1.5, step: 0.05 },
    ],
  },
  {
    title: 'Reforming into the graph',
    items: [
      { key: 'swirl', label: 'Swirl', hint: 'spiral on the way to the graph', min: 0, max: 2.5, step: 0.05 },
      { key: 'trail', label: 'Streaks', hint: 'length of trails behind dots', min: 0, max: 3, step: 0.05 },
    ],
  },
];
const PRESETS: [string, Partial<Tune>][] = [
  ['Calm', { overcharge: 0.4, squeeze: 0.25, burstReach: 0.2, flash: 0.6, rays: 0, swirl: 1.5, neuronSpeed: 1.2, neurons: 14 }],
  ['Dramatic', { overcharge: 1.2, squeeze: 0.45, burstReach: 1.4, flash: 1.2, rays: 1, swirl: 1, neuronSpeed: 2.2, neurons: 40, trail: 1 }],
  ['Minimal', { particles: 1500, neurons: 0, overcharge: 0, squeeze: 0, burstReach: 0.1, flash: 0.3, rays: 0, swirl: 0.5, trail: 0 }],
];
// the moments worth looking at, read from the current settings so they stay true as the numbers change
const jumps = (): [string, number][] => [
  ['Welcome', 0],
  ['Charged', Math.min(TUNE.chargeEnd, TUNE.burstAt - 0.3)],
  ['Overcharge', TUNE.burstAt - 0.1],
  ['Blast', TUNE.burstAt + 0.06],
  ['Landing', TUNE.burstAt + TUNE.burstSpan * 0.5],
  ['Graph', TUNE.burstAt + TUNE.burstSpan],
];

function jump(q: number) {
  const secs = Array.from(document.querySelectorAll<HTMLElement>('section.scene'));
  const i = Math.min(secs.length - 2, Math.floor(q)), y = window.scrollY;
  const top = (n: number) => secs[n].getBoundingClientRect().top + y;
  window.scrollTo({ top: top(i) + (q - i) * (top(i + 1) - top(i)), behavior: 'smooth' });
}

export function SphereTune() {
  const [, bump] = useState(0);
  const [open, setOpen] = useState(true);
  const [copied, setCopied] = useState(false);
  const set = (k: keyof Tune, v: number) => {
    TUNE[k] = v;
    bump((n) => n + 1);
  };
  const changed = (Object.keys(TUNE_DEFAULTS) as (keyof Tune)[]).filter((k) => TUNE[k] !== TUNE_DEFAULTS[k]);
  const copy = async () => {
    const text = `Sphere settings: ${JSON.stringify(TUNE)}`;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      window.prompt('Copy these settings', text);
    }
  };
  return (
    <aside className="tune glass" aria-label="Sphere settings">
      <header>
        <b>Sphere settings</b>
        <button type="button" className="chip small" onClick={() => setOpen(!open)} aria-expanded={open}>
          {open ? 'Hide' : 'Show'}
        </button>
      </header>
      {open && (
        <>
          <div className="tune-jumps">
            {jumps().map(([n, q]) => (
              <button key={n} type="button" className="chip small" onClick={() => jump(q)}>
                {n}
              </button>
            ))}
          </div>
          <div className="tune-jumps">
            {PRESETS.map(([n, v]) => (
              <button key={n} type="button" className="chip small" onClick={() => { Object.assign(TUNE, v); bump((c) => c + 1); }}>
                {n}
              </button>
            ))}
          </div>
          {GROUPS.map((g) => (
            <fieldset key={g.title}>
              <legend>{g.title}</legend>
              {g.items.map((c) => (
                <label key={c.key} title={c.hint}>
                  <span>
                    {c.label}
                    <i>{Number(TUNE[c.key]).toFixed(c.step < 0.1 ? 2 : 0)}</i>
                  </span>
                  <input type="range" min={c.min} max={c.max} step={c.step} value={TUNE[c.key]} onChange={(e) => set(c.key, Number(e.target.value))} />
                </label>
              ))}
            </fieldset>
          ))}
          <div className="tune-go">
            <button type="button" className="chip small primary" onClick={copy}>
              {copied ? 'Copied' : 'Copy my settings'}
            </button>
            <button
              type="button"
              className="chip small"
              onClick={() => {
                Object.assign(TUNE, TUNE_DEFAULTS);
                bump((n) => n + 1);
              }}
              disabled={changed.length === 0}
            >
              Reset
            </button>
          </div>
        </>
      )}
    </aside>
  );
}
