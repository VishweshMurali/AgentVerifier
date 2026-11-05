// Small motion pieces for the story: word-by-word headlines, numbers that count up, the canvas
// background with its readout. All driven by JavaScript so they work on every browser.
import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { prefersReducedMotion } from '../motion/flight.ts';
import { startRibbon, type Flyer, type Lane } from '../motion/ribbon.ts';

/** A headline split into words that rise in one after another (see .w in styles.css). */
export function Words({ text, now = false }: { text: string; now?: boolean }) {
  const words = text.split(/\s+/).filter(Boolean);
  return (
    <>
      {words.map((w, i) => (
        <span key={i}>
          {i > 0 && ' '}
          <span className={`w${now ? ' now' : ''}`} style={{ '--i': i } as CSSProperties}>
            {w}
          </span>
        </span>
      ))}
    </>
  );
}

/** A number that counts up from zero each time `run` turns true. Reduced motion shows it at once. */
export function CountUp({ value, decimals = 1, prefix = '', suffix = '', run }: {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  run: boolean;
}) {
  const still = prefersReducedMotion();
  const [shown, setShown] = useState(still ? value : 0);
  useEffect(() => {
    if (still) {
      setShown(value);
      return;
    }
    if (!run) return;
    let raf = 0;
    const t0 = performance.now() + 250;
    const step = (now: number) => {
      const p = Math.min(1, Math.max(0, (now - t0) / 1300));
      setShown(value * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [run, value, still]);
  return (
    <>
      {prefix}
      {shown.toFixed(decimals)}
      {suffix}
    </>
  );
}

interface BackgroundProps {
  lanes: Lane[];
  revealDirection: number;
  nCases: number;
  sourceLabel: string;
  sceneCount: number;
  night: boolean;
  flyer: () => Flyer | null;
}

/** The dotted-ribbon canvas, the reveal's corner brackets and the bottom-left readout. */
export function Background({ lanes, revealDirection, nCases, sourceLabel, sceneCount, night, flyer }: BackgroundProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const hudCase = useRef<HTMLElement>(null);
  const hudScene = useRef<HTMLElement>(null);
  const nightRef = useRef(night);
  nightRef.current = night;
  const flyerRef = useRef(flyer);
  flyerRef.current = flyer;

  useEffect(() => {
    if (!canvas.current) return;
    return startRibbon(canvas.current, {
      lanes,
      revealDirection,
      nCases,
      still: prefersReducedMotion(),
      isNight: () => nightRef.current,
      flyer: () => flyerRef.current(),
      hudCase: hudCase.current,
      hudScene: hudScene.current,
    });
  }, [lanes, revealDirection, nCases]);

  return (
    <>
      <div className="bg" aria-hidden="true">
        <div className="bg-night" />
        <canvas ref={canvas} />
      </div>
      <div className="frame" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
      </div>
      <div className="hud" aria-hidden="true">
        <span>{sourceLabel}</span>
        <span>
          Case <b ref={hudCase}>0000</b> / {nCases.toLocaleString('en-US')}
        </span>
        <span>
          Scene <b ref={hudScene}>01</b> / {String(sceneCount).padStart(2, '0')}
        </span>
      </div>
    </>
  );
}
